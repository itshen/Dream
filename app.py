from flask import Flask, request, jsonify, render_template, Response
import torch
from transformers import AutoModel, AutoTokenizer
import json
import time
import queue
import threading
import logging
import os
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Dream-Web")

# 检查是否是Flask的子进程重启
# 当Flask以debug=True模式运行时，会启动两个进程，这里识别主进程
is_flask_main_process = not os.environ.get('WERKZEUG_RUN_MAIN')

app = Flask(__name__)

# 全局变量
model_path = "Dream-org/Dream-v0-Instruct-7B"
tokenizer = None
model = None

# 用于存储生成任务的队列
generation_tasks = {}

# 加载模型和tokenizer
def load_model_and_tokenizer():
    global tokenizer, model
    
    if is_flask_main_process:
        logger.info("这是Flask调试模式下的主进程，仅加载tokenizer供代码检查")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("主进程Tokenizer加载完成")
        return

    logger.info(f"正在加载tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Tokenizer加载完成")
    
    logger.info(f"开始加载模型: {model_path}")
    logger.info("这可能需要几分钟时间...")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to("cuda").eval()
    logger.info("模型加载完成并已移至GPU")

# 在应用启动时加载模型
load_model_and_tokenizer()

@app.route('/')
def index():
    logger.info("访问首页")
    return render_template('index.html')

@app.route('/polish')
def polish():
    logger.info("访问文本润色页面")
    return render_template('polish.html')

@app.route('/api/polish', methods=['POST'])
def api_polish():
    """接收文本润色请求，预处理文本并返回任务ID"""
    data = request.json
    system_prompt = data.get('system_prompt', '请帮我润色文本，使其更加专业流畅。')
    user_input = data.get('user_input', '')
    
    # 检查输入格式
    if '\n\n' not in user_input:
        return jsonify({"error": "输入格式错误，必须包含原始文本和带掩码的文本，以空行分隔"}), 400
    
    # 分割原始文本和带掩码的文本
    parts = user_input.split('\n\n', 1)
    original_text = parts[0]
    masked_text = parts[1] if len(parts) > 1 else original_text
    
    # 限制[Mask]标记的数量，避免过多掩码导致结果异常
    mask_count = masked_text.count('<|mask|>')
    max_masks = 30  # 设置最大掩码数量
    
    if mask_count > max_masks:
        logger.warning(f"掩码数量过多 ({mask_count}), 超过限制 ({max_masks})，将进行限制")
        # 只保留前max_masks个掩码，其余的替换回原文的对应部分
        
        # 找出所有掩码位置
        mask_positions = []
        start_pos = 0
        while True:
            pos = masked_text.find('<|mask|>', start_pos)
            if pos == -1:
                break
            mask_positions.append(pos)
            start_pos = pos + len('<|mask|>')
        
        # 如果掩码过多，只保留前max_masks个
        if len(mask_positions) > max_masks:
            # 取前max_masks个掩码的最后一个位置
            last_kept_pos = mask_positions[max_masks - 1] + len('<|mask|>')
            # 截取到该位置的文本，后面的内容使用原文
            masked_text = masked_text[:last_kept_pos]
            
            # 处理原始文本对应位置，确保长度匹配
            if len(original_text) > last_kept_pos:
                masked_text += original_text[last_kept_pos:]
    
    # 构建聊天消息格式
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_text},
        {"role": "assistant", "content": masked_text}  # 助手回复中包含掩码
    ]
    
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 为该任务创建队列
    generation_tasks[task_id] = {
        "queue": queue.Queue(),
        "messages": messages,
        "params": data.get('params', {}),
        "start_time": time.time()
    }
    
    # 启动生成线程
    threading.Thread(target=polish_text, args=(task_id,)).start()
    
    # 返回任务ID
    return jsonify({"task_id": task_id})

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """接收生成请求，返回任务ID"""
    data = request.json
    system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
    user_input = data.get('user_input', '')
    
    # 获取用户自定义参数或使用默认值
    params = data.get('params', {})
    
    logger.info(f"收到生成请求，用户输入: '{user_input[:50]}...' (截断)")
    logger.info(f"系统提示: '{system_prompt[:50]}...' (截断)")
    logger.info(f"生成参数: {params}")
    
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 为该任务创建队列
    generation_tasks[task_id] = {
        "queue": queue.Queue(),
        "system_prompt": system_prompt, 
        "user_input": user_input,
        "params": params,
        "start_time": time.time()
    }
    
    # 启动生成线程
    threading.Thread(target=generate_text, args=(task_id,)).start()
    
    # 返回任务ID
    return jsonify({"task_id": task_id})

@app.route('/api/stop/<task_id>', methods=['POST'])
def stop_generation(task_id):
    """停止指定任务ID的生成过程"""
    if task_id not in generation_tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    logger.info(f"收到停止请求，任务ID: {task_id}")
    
    # 将任务标记为需要停止
    generation_tasks[task_id]["stop_requested"] = True
    
    # 直接发送停止消息到队列
    task_queue = generation_tasks[task_id]["queue"]
    task_queue.put((-4, "用户已停止生成"))
    
    return jsonify({"success": True, "message": "已发送停止信号"})

@app.route('/stream/<task_id>')
def stream_generate(task_id):
    """为特定任务ID创建SSE流"""
    if task_id not in generation_tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    logger.info(f"为任务 {task_id} 创建SSE流")
    
    # 初始化stop_requested标志
    generation_tasks[task_id]["stop_requested"] = False
    
    def generate():
        task = generation_tasks[task_id]
        task_queue = task["queue"]
        
        try:
            while True:
                try:
                    # 从队列中获取生成结果，最多等待60秒
                    result = task_queue.get(timeout=60)
                    
                    # 检查结果类型和长度
                    if isinstance(result, tuple):
                        if len(result) == 3:  # 带性能数据或温度数据的事件
                            step, text, extra_data = result
                            if "temperature" in extra_data:  # 是温度数据
                                performance_data = None
                                temperature = extra_data["temperature"]
                            else:  # 是性能数据
                                performance_data = extra_data
                                temperature = None
                        else:  # 普通事件，没有额外数据
                            step, text = result
                            performance_data = None
                            temperature = None
                    else:
                        logger.error(f"任务 {task_id} 收到非元组数据: {result}")
                        continue
                    
                    # 特殊标记-1表示生成完成
                    if step == -1:
                        if performance_data:
                            logger.info(f"任务 {task_id} 完成，生成了 {performance_data['tokens']} tokens，耗时 {performance_data['time']}s，TPS: {performance_data['tps']}")
                            data = json.dumps({
                                "step": "完成", 
                                "text": text,
                                "performance": performance_data
                            })
                        else:
                            logger.info(f"任务 {task_id} 完成")
                            data = json.dumps({"step": "完成", "text": text})
                        yield f"data: {data}\n\n"
                        # 清理任务数据，先检查任务是否仍存在
                        if task_id in generation_tasks:
                            del generation_tasks[task_id]
                        break
                    # 特殊标记-2表示发生错误
                    elif step == -2:
                        logger.error(f"任务 {task_id} 出错: {text}")
                        data = json.dumps({"step": "错误", "text": f"生成过程出错: {text}"})
                        yield f"data: {data}\n\n"
                        # 清理任务数据，先检查任务是否仍存在
                        if task_id in generation_tasks:
                            del generation_tasks[task_id]
                        break
                    # 特殊标记-3表示提前结束生成
                    elif step == -3:
                        logger.info(f"任务 {task_id} 提前结束生成")
                        data = json.dumps({"step": "完成", "text": text, "early_stopped": True})
                        yield f"data: {data}\n\n"
                        # 清理任务数据，先检查任务是否仍存在
                        if task_id in generation_tasks:
                            del generation_tasks[task_id]
                        break
                    # 特殊标记-4表示用户停止
                    elif step == -4:
                        logger.info(f"任务 {task_id} 被用户停止")
                        data = json.dumps({"step": "停止", "text": text, "user_stopped": True})
                        yield f"data: {data}\n\n"
                        # 清理任务数据，先检查任务是否仍存在
                        if task_id in generation_tasks:
                            del generation_tasks[task_id]
                        break
                    else:
                        # 对于常规步骤，只在特定间隔记录日志，避免日志过多
                        if step % 50 == 0:
                            logger.info(f"任务 {task_id} 步骤 {step}")
                        # 构建数据对象，如果有温度信息则包含
                        data_obj = {"step": step, "text": text}
                        if temperature is not None:
                            data_obj["temperature"] = temperature
                            
                        data = json.dumps(data_obj)
                        yield f"data: {data}\n\n"
                except queue.Empty:
                    # 超时退出
                    logger.warning(f"任务 {task_id} 超时")
                    data = json.dumps({"step": "超时", "text": "生成过程超时，请重试"})
                    yield f"data: {data}\n\n"
                    # 清理任务数据，先检查任务是否仍存在
                    if task_id in generation_tasks:
                        del generation_tasks[task_id]
                    break
        except GeneratorExit:
            # 客户端断开连接
            logger.info(f"任务 {task_id} 的客户端断开连接")
            # 不要立即删除任务数据，给生成进程一些时间完成或清理
    
    return Response(generate(), mimetype="text/event-stream")

def generate_text(task_id):
    """文本生成函数，在后台线程中运行"""
    task = generation_tasks[task_id]
    system_prompt = task["system_prompt"]
    user_input = task["user_input"]
    step_queue = task["queue"]
    params = task["params"]
    
    # 获取参数或使用默认值
    steps = params.get('steps', 8)
    temperature = params.get('temperature', 0.75)
    top_p = params.get('top_p', 0.7)
    max_new_tokens = params.get('max_new_tokens', 512)
    alg = params.get('alg', 'origin')
    alg_temp = params.get('alg_temp', 0.2)
    random_temp = params.get('random_temp', False)  # 新增随机温度参数
    temp_min = params.get('temp_min', 0.1)  # 最小温度值
    temp_max = params.get('temp_max', 1.0)  # 最大温度值
    
    try:
        logger.info(f"开始为任务 {task_id} 生成文本")
        if random_temp:
            logger.info(f"参数：steps={steps}, 随机温度范围=[{temp_min}-{temp_max}], top_p={top_p}, max_new_tokens={max_new_tokens}, alg={alg}, alg_temp={alg_temp}")
        else:
            logger.info(f"参数：steps={steps}, temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}, alg={alg}, alg_temp={alg_temp}")
        
        if random_temp:
            logger.info(f"启用随机温度：范围 {temp_min} - {temp_max}")
        start_time = time.time()
        last_log_time = start_time
        
        # 用于检测生成停滞的变量
        last_text = ""
        unchanged_count = 0
        
        # 定义自定义异常
        class EarlyStopGeneration(Exception):
            pass
        
        class UserStopGeneration(Exception):
            pass
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # 格式化输入
        logger.info(f"任务 {task_id} 应用聊天模板并准备输入")
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")
        
        logger.info(f"任务 {task_id} 输入长度: {len(input_ids[0])} tokens")
        
        # 跟踪当前使用的温度值
        current_temperature = temperature
        
        # 模型生成的钩子函数
        def generation_tokens_hook_func(step, x, logits):
            nonlocal last_log_time, last_text, unchanged_count, current_temperature
            current_time = time.time()
            current_text = tokenizer.decode(x[0].tolist())
            # 去除<|endoftext|>标记
            current_text = current_text.replace("<|endoftext|>", "")
            
            # 清理assistant回复中多余的mask标记
            if '<|im_start|>assistant' in current_text:
                # 尝试找到assistant部分
                parts = current_text.split('<|im_start|>assistant')
                if len(parts) > 1:
                    assistant_part = parts[-1]
                    # 如果包含<|im_end|>，只保留到<|im_end|>之前的内容
                    if '<|im_end|>' in assistant_part:
                        assistant_part = assistant_part.split('<|im_end|>')[0]
                    
                    # 检查[Mask]标记
                    # 如果找到第一个[Mask]，只保留[Mask]之前的内容加上当前[Mask]
                    if '[Mask]' in assistant_part:
                        clean_part = assistant_part.split('[Mask]')[0] + '[Mask]'
                        # 重新组装文本
                        current_text = parts[0] + '<|im_start|>assistant' + clean_part
            
            token_count = len(x[0]) - len(input_ids[0])
            
            # 检查是否有停止请求
            if task.get("stop_requested", False):
                logger.info(f"任务 {task_id} 收到停止请求，停止生成")
                raise UserStopGeneration("用户请求停止生成")
            
            # 处理step为None的情况
            if step is None:
                logger.info(f"任务 {task_id} 初始化步骤: 输入长度 {len(input_ids[0])} tokens")
                step_queue.put((0, current_text))
                last_text = current_text
                return x
            
            # 如果启用了随机温度，为每次迭代生成新的温度值
            if random_temp and step > 0:
                import random
                current_temperature = random.uniform(temp_min, temp_max)
                logger.info(f"任务 {task_id} 步骤 {step} 使用随机温度: {current_temperature:.2f}")
            
            # 检查文本是否变化
            if current_text == last_text:
                unchanged_count += 1
                # 如果连续50次没有变化，提前结束
                if unchanged_count >= 50:
                    logger.info(f"任务 {task_id} 连续50次迭代无变化，提前结束生成")
                    # 使用-3作为特殊标记，表示提前结束
                    step_queue.put((-3, current_text))
                    # 抛出异常以提前结束生成过程
                    raise EarlyStopGeneration("提前结束生成")
            else:
                unchanged_count = 0
                last_text = current_text
            
            # 每5步或者时间间隔超过1秒记录一次日志
            if step % 5 == 0 or current_time - last_log_time > 1.0:
                if random_temp:
                    logger.info(f"任务 {task_id} 步骤 {step}/{steps}: 温度 {current_temperature:.2f}, 已生成 {token_count} tokens, 耗时 {current_time - start_time:.2f}s")
                else:
                    logger.info(f"任务 {task_id} 步骤 {step}/{steps}: 已生成 {token_count} tokens，耗时 {current_time - start_time:.2f}s")
                last_log_time = current_time
            
            # 将当前步骤，生成的文本和当前温度值一起发送到队列
            if random_temp:
                # 如果使用随机温度，同时发送温度值
                step_queue.put((step, current_text, {"temperature": current_temperature}))
            else:
                step_queue.put((step, current_text))
            time.sleep(0.05)  # 稍微减慢速度以便观察
            return x
        
        # 生成文本
        logger.info(f"任务 {task_id} 开始扩散生成过程")
        # 如果使用随机温度，首次迭代使用初始温度，后续会在钩子函数中更新
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,  # 使用initial_temperature作为起始温度
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            generation_tokens_hook_func=generation_tokens_hook_func
        )
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 发送最终结果
        final_text = tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
        final_text = final_text.split(tokenizer.eos_token)[0].strip()
        
        # 记录生成结果信息
        token_count = len(output.sequences[0]) - len(input_ids[0])
        
        # 计算TPS (Tokens Per Second) - 使用最终生成的token总数除以总耗时
        tps = token_count / total_time if total_time > 0 else 0
        
        logger.info(f"任务 {task_id} 生成完成: 共 {token_count} tokens，总耗时 {total_time:.2f}s，TPS: {tps:.2f} tokens/s")
        
        # 使用-1表示完成，同时附加性能统计数据
        performance_data = {
            "tokens": token_count,
            "time": round(total_time, 2),
            "tps": round(tps, 2)
        }
        step_queue.put((-1, final_text, performance_data))
    except EarlyStopGeneration:
        # 处理提前结束生成的情况
        logger.info(f"任务 {task_id} 提前结束生成")
        # 不需要再次发送消息，因为已经在钩子函数中发送了
    except UserStopGeneration:
        # 处理用户停止的情况
        logger.info(f"任务 {task_id} 被用户停止")
        # 消息已在API调用中发送，无需再次发送
    except Exception as e:
        logger.error(f"任务 {task_id} 生成过程出错: {e}", exc_info=True)
        step_queue.put((-2, str(e)))  # 使用-2表示错误
        
        # 如果任务仍存在，清理它
        if task_id in generation_tasks:
            del generation_tasks[task_id]

def polish_text(task_id):
    """文本润色函数，在后台线程中运行"""
    task = generation_tasks[task_id]
    messages = task["messages"]
    step_queue = task["queue"]
    params = task["params"]
    
    # 获取参数或使用默认值
    steps = params.get('steps', 8)
    temperature = params.get('temperature', 0.75)
    top_p = params.get('top_p', 0.7)
    max_new_tokens = params.get('max_new_tokens', 512)
    alg = params.get('alg', 'origin')
    alg_temp = params.get('alg_temp', 0.2)
    random_temp = params.get('random_temp', False)
    temp_min = params.get('temp_min', 0.1)
    temp_max = params.get('temp_max', 1.0)
    
    try:
        logger.info(f"开始为润色任务 {task_id} 生成文本")
        if random_temp:
            logger.info(f"参数：steps={steps}, 随机温度范围=[{temp_min}-{temp_max}], top_p={top_p}, max_new_tokens={max_new_tokens}, alg={alg}, alg_temp={alg_temp}")
        else:
            logger.info(f"参数：steps={steps}, temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}, alg={alg}, alg_temp={alg_temp}")
        
        start_time = time.time()
        last_log_time = start_time
        
        # 用于检测生成停滞的变量
        last_text = ""
        unchanged_count = 0
        
        # 定义自定义异常
        class EarlyStopGeneration(Exception):
            pass
        
        class UserStopGeneration(Exception):
            pass
        
        # 格式化输入
        logger.info(f"润色任务 {task_id} 应用聊天模板并准备输入")
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        )
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")
        
        logger.info(f"润色任务 {task_id} 输入长度: {len(input_ids[0])} tokens")
        
        # 跟踪当前使用的温度值
        current_temperature = temperature
        
        # 模型生成的钩子函数
        def generation_tokens_hook_func(step, x, logits):
            nonlocal last_log_time, last_text, unchanged_count, current_temperature
            current_time = time.time()
            current_text = tokenizer.decode(x[0].tolist())
            # 去除<|endoftext|>标记
            current_text = current_text.replace("<|endoftext|>", "")
            
            # 清理assistant回复中多余的mask标记
            if '<|im_start|>assistant' in current_text:
                # 尝试找到assistant部分
                parts = current_text.split('<|im_start|>assistant')
                if len(parts) > 1:
                    assistant_part = parts[-1]
                    # 如果包含<|im_end|>，只保留到<|im_end|>之前的内容
                    if '<|im_end|>' in assistant_part:
                        assistant_part = assistant_part.split('<|im_end|>')[0]
                    
                    # 检查[Mask]标记
                    # 如果找到第一个[Mask]，只保留[Mask]之前的内容加上当前[Mask]
                    if '[Mask]' in assistant_part:
                        clean_part = assistant_part.split('[Mask]')[0] + '[Mask]'
                        # 重新组装文本
                        current_text = parts[0] + '<|im_start|>assistant' + clean_part
            
            token_count = len(x[0]) - len(input_ids[0])
            
            # 检查是否有停止请求
            if task.get("stop_requested", False):
                logger.info(f"润色任务 {task_id} 收到停止请求，停止生成")
                raise UserStopGeneration("用户请求停止生成")
            
            # 处理step为None的情况
            if step is None:
                logger.info(f"润色任务 {task_id} 初始化步骤: 输入长度 {len(input_ids[0])} tokens")
                step_queue.put((0, current_text))
                last_text = current_text
                return x
            
            # 如果启用了随机温度，为每次迭代生成新的温度值
            if random_temp and step > 0:
                import random
                current_temperature = random.uniform(temp_min, temp_max)
                logger.info(f"润色任务 {task_id} 步骤 {step} 使用随机温度: {current_temperature:.2f}")
            
            # 检查文本是否变化
            if current_text == last_text:
                unchanged_count += 1
                # 如果连续50次没有变化，提前结束
                if unchanged_count >= 50:
                    logger.info(f"润色任务 {task_id} 连续50次迭代无变化，提前结束生成")
                    # 使用-3作为特殊标记，表示提前结束
                    step_queue.put((-3, current_text))
                    # 抛出异常以提前结束生成过程
                    raise EarlyStopGeneration("提前结束生成")
            else:
                unchanged_count = 0
                last_text = current_text
            
            # 每5步或者时间间隔超过1秒记录一次日志
            if step % 5 == 0 or current_time - last_log_time > 1.0:
                if random_temp:
                    logger.info(f"润色任务 {task_id} 步骤 {step}/{steps}: 温度 {current_temperature:.2f}, 已生成 {token_count} tokens, 耗时 {current_time - start_time:.2f}s")
                else:
                    logger.info(f"润色任务 {task_id} 步骤 {step}/{steps}: 已生成 {token_count} tokens，耗时 {current_time - start_time:.2f}s")
                last_log_time = current_time
            
            # 将当前步骤，生成的文本和当前温度值一起发送到队列
            if random_temp:
                # 如果使用随机温度，同时发送温度值
                step_queue.put((step, current_text, {"temperature": current_temperature}))
            else:
                step_queue.put((step, current_text))
            time.sleep(0.05)  # 稍微减慢速度以便观察
            return x
        
        # 生成文本
        logger.info(f"润色任务 {task_id} 开始扩散生成过程")
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            generation_tokens_hook_func=generation_tokens_hook_func
        )
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 发送最终结果
        final_text = tokenizer.decode(output.sequences[0].tolist())
        
        # 尝试从最终文本中提取助手的回复部分
        try:
            # 清除<|endoftext|>标记
            final_text = final_text.replace("<|endoftext|>", "")
            
            # 尝试通过常见的分隔符提取助手回复
            if '<|im_start|>assistant' in final_text and '<|im_end|>' in final_text:
                # 找到最后一个assistant开始和第一个im_end结束
                assistant_start = final_text.rfind('<|im_start|>assistant')
                if assistant_start != -1:
                    assistant_part = final_text[assistant_start:]
                    # 清理可能存在的结束标记
                    if '<|im_end|>' in assistant_part:
                        end_pos = assistant_part.find('<|im_end|>')
                        assistant_part = assistant_part[:end_pos]
                    # 清理其他可能的标记
                    assistant_part = assistant_part.replace('<|im_start|>assistant', '').strip()
                    final_text = assistant_part
            else:
                # 如果无法通过标记分割，尝试从用户输入后找到助手回复
                user_input = messages[1]["content"]
                if user_input in final_text:
                    assistant_reply_start = final_text.find(user_input) + len(user_input)
                    # 查找结束标记如果存在
                    if '<|im_end|>' in final_text[assistant_reply_start:]:
                        end_pos = final_text.find('<|im_end|>', assistant_reply_start)
                        final_text = final_text[assistant_reply_start:end_pos].strip()
                    else:
                        final_text = final_text[assistant_reply_start:].strip()
            
            # 最后再次确保移除所有特殊标记
            final_text = final_text.replace("<|im_start|>assistant", "")
            final_text = final_text.replace("<|im_end|>", "")
            final_text = final_text.replace("<|endoftext|>", "")
            
            # 清除assistant回复之后的所有[Mask]标记
            # 这是为防止在输出中显示assistant回复后续的mask内容
            if '<|im_end|>' in final_text:
                final_text = final_text.split('<|im_end|>')[0].strip()
            
            # 如果是原始masked_text（即assistant回复），只取第一个<|im_end|>之前的内容
            original_assistant_content = messages[2]["content"]
            if original_assistant_content in final_text:
                # 只保留模型真正生成的部分，移除原始输入中的掩码内容
                cleaned_text = original_assistant_content
                if '<|im_end|>' in cleaned_text:
                    cleaned_text = cleaned_text.split('<|im_end|>')[0]
                final_text = cleaned_text
        except Exception as e:
            logger.warning(f"无法提取助手回复部分，使用完整生成文本: {str(e)}")
            # 至少清除特殊标记
            final_text = final_text.replace("<|endoftext|>", "")
            final_text = final_text.replace("<|im_start|>assistant", "")
            final_text = final_text.replace("<|im_end|>", "")
            
            # 确保不包含后续的[Mask]内容
            if '[Mask]' in final_text:
                parts = final_text.split('[Mask]')
                # 只保留第一部分，即没有[Mask]的内容
                final_text = parts[0].strip()
        
        # 记录生成结果信息
        token_count = len(output.sequences[0]) - len(input_ids[0])
        
        # 计算TPS (Tokens Per Second)
        tps = token_count / total_time if total_time > 0 else 0
        
        logger.info(f"润色任务 {task_id} 生成完成: 共 {token_count} tokens，总耗时 {total_time:.2f}s，TPS: {tps:.2f} tokens/s")
        
        # 使用-1表示完成，同时附加性能统计数据
        performance_data = {
            "tokens": token_count,
            "time": round(total_time, 2),
            "tps": round(tps, 2)
        }
        step_queue.put((-1, final_text, performance_data))
    except EarlyStopGeneration:
        # 处理提前结束生成的情况
        logger.info(f"润色任务 {task_id} 提前结束生成")
        # 不需要再次发送消息，因为已经在钩子函数中发送了
    except UserStopGeneration:
        # 处理用户停止的情况
        logger.info(f"润色任务 {task_id} 被用户停止")
        # 消息已在API调用中发送，无需再次发送
    except Exception as e:
        logger.error(f"润色任务 {task_id} 生成过程出错: {e}", exc_info=True)
        step_queue.put((-2, str(e)))  # 使用-2表示错误
        
        # 如果任务仍存在，清理它
        if task_id in generation_tasks:
            del generation_tasks[task_id]

if __name__ == '__main__':
    if not is_flask_main_process:
        logger.info("Dream-v0-Instruct-7B Web演示应用启动")
        logger.info("访问 http://127.0.0.1:5000/ 开始使用")
    app.run(debug=True) 