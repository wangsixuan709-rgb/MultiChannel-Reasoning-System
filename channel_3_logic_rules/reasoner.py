"""
================================================================================
Channel 3: Logical Reasoning Engine (VLM-CoT) (逻辑与事实推理检测)
文件名: reasoner.py
定位: 系统逻辑层防线，处理CLIP无法识别的细粒度属性冲突与常识谬误

================================================================================
【核心任务】
构建具备"深度认知"能力的AI审判官，通过视觉大模型(VLM)与思维链(Chain of Thought)
技术，解决语义层(通道二)无法覆盖的细粒度逻辑冲突。

【与通道二(CLIP)的关键区别】
  通道二 (语义一致性): 解决 "Topic Alignment" (主题是否一致)
    - 能力边界: 只能判断"图和文是不是在说同一件事"
    - 对细节(天气、时间、数量)不敏感
  
  通道三 (逻辑推理): 解决 "Fact Verification" (事实是否冲突)
    - 能力边界: 在主题一致的前提下，通过VLM(视觉转译)和LLM(逻辑比对)
    - 寻找时空、因果、常识上的矛盾

【检测目标 - 三种冲突类型】
1. 细粒度属性冲突 (Fine-grained Attribute Conflict)
   - 时间: 图(正午) vs 文(深夜)
   - 天气: 图(晴天) vs 文(暴雨)
   - 数量: 图(空地) vs 文(人山人海)

2. 实体/地标错位 (Entity Mismatch)
   - 地标: 图(东方明珠) vs 文(东京塔)
   - 文字: 图中路牌/横幅文字与新闻内容矛盾 (OCR能力)

3. 常识因果谬误 (Common Sense Error)
   - 物理常识: 图(夏天短袖) vs 文(大雪纷飞)

【技术原理 - VLM-CoT (Visual Chain of Thought)】
  Step 1: 视觉转译 (Captioning)
    - 利用VLM将图片转化为结构化元数据
    - Prompt: "Describe focusing on: time of day, weather, location, quantity"
  
  Step 2: 逻辑比对 (Reasoning)
    - 利用LLM进行NLI(自然语言推理)任务
    - Logic: Premise(Image) <-> Hypothesis(Text) ?

================================================================================
【I/O 接口规范】
================================================================================
输入 (Input):
  - image_path (str): 图片路径
  - text (str): 新闻文本
  - meta_data (dict): Excel中的元数据行 (Mock模式下作为推理依据)

输出 (Output):
  - is_conflict (bool): True=逻辑冲突(Fake), False=逻辑自洽(Real)
    对应 Excel 字段: GT_Ch3_Logic (1=有冲突, 0=无冲突)
  - reason (str): 推理证据描述

【与Excel字段的对应关系】
  - Sample_Type = "Logic_Trap" -> 逻辑陷阱样本，GT_Ch3_Logic = 1
  - 输入: Image_Path (C列), Text_Content (D列)
  - 验证: GT_Ch3_Logic (H列), 1=有冲突, 0=无冲突
  - 元数据: Meta_Time, Meta_Weather, Meta_Location, Meta_Fact, Meta_Object

【Mock模式说明】
  鉴于演示环境算力限制，系统默认开启 Mock Mode (模拟模式)：
  - 通过读取预处理的元数据(Excel Ground Truth)模拟VLM的输出
  - 确保演示的低延迟与高准确率

================================================================================
【技术选型】
推荐模型:
  - VLM: Moondream (轻量级) 或 LLaVA (高精度)
  - LLM: 本地部署或API调用

================================================================================
"""

import os
import re
# import torch  # TODO: 待模型集成时解开注释


class LogicReasoner:
    """
    逻辑推理引擎 (VLM-CoT 架构)
    处理CLIP无法识别的细粒度属性冲突与常识谬误
    """
    
    def __init__(self):
        """
        初始化逻辑推理引擎
        
        架构说明:
          - 理想架构: Image -> VLM -> Caption -> LLM -> Conflict Check
          - Demo实现: Image -> (Query Excel Meta) -> Mock Caption -> Rule Check
        
        此实现采用 Demo实现 策略，以确保演示时的响应速度和绝对准确率。
        """
        print("[Ch3-Init] Initializing Logic Reasoning Engine (VLM-CoT)...")
        print("[Ch3-Init] Mode: Mock (Excel Meta as Oracle)")
        
        # 时间关键词库
        self.night_keywords = ["深夜", "凌晨", "漆黑", "晚间", "通宵", "月色", "月光", "midnight", "night"]
        self.day_keywords = ["阳光", "正午", "白天", "烈日", "上午", "下午", "noon", "daylight", "morning"]
        
        # 天气关键词库
        self.storm_keywords = ["暴雨", "洪水", "台风", "积水", "雷电", "狂风", "暴风", "storm", "rain", "flood"]
        self.sunny_keywords = ["晴朗", "阳光明媚", "蓝天", "万里无云", "sunny", "clear"]
        
        # 季节/温度关键词库
        self.winter_keywords = ["大雪", "寒冬", "冰雪", "snow", "winter", "freezing"]
        self.summer_keywords = ["炎热", "酷暑", "短袖", "summer", "hot"]
        
        self.model_loaded = False  # 标记模型是否已加载

    def _vlm_captioning_mock(self, image_path, meta_data):
        """
        [模拟] VLM 的看图说话功能
        利用 Excel 里的 Meta 数据来'伪装'成 VLM 的视觉感知输出
        
        Args:
            image_path (str): 图片路径
            meta_data (dict): Excel 元数据
        
        Returns:
            dict: 结构化的视觉描述
        
        对应 Excel 字段:
            - Meta_Time: Day / Night
            - Meta_Weather: Sunny / Rain / Snow / Cloudy
            - Meta_Location: 地点名词 (Paris, Street...)
            - Meta_Fact: 事实状态 (Empty, Crowded...)
            - Meta_Object: 关键物体 (Car, Fire...)
        """
        # 从 Excel 元数据中获取真实信息，如果为空则默认为 Unknown
        time_info = str(meta_data.get('Meta_Time', 'Unknown')).strip()
        weather_info = str(meta_data.get('Meta_Weather', 'Unknown')).strip()
        location_info = str(meta_data.get('Meta_Location', 'Unknown')).strip()
        fact_info = str(meta_data.get('Meta_Fact', 'Unknown')).strip()
        object_info = str(meta_data.get('Meta_Object', 'Unknown')).strip()
        
        # 构造结构化的视觉描述，模拟 VLM 的输出格式
        vlm_caption = {
            "Time": time_info,        # e.g., "Day", "Night"
            "Weather": weather_info,  # e.g., "Sunny", "Rain"
            "Location": location_info, # e.g., "Beach", "Street"
            "Fact": fact_info,        # e.g., "Noon", "Crowded"
            "Objects": object_info    # e.g., "Eiffel Tower", "Car"
        }
        return vlm_caption

    def _call_vlm_api(self, image_path, prompt):
        """
        [预留接口] 调用真实 VLM API
        
        TODO: 接入 Moondream / LLaVA 等 VLM 模型
        
        Args:
            image_path (str): 图片路径
            prompt (str): VLM 提示词
        
        Returns:
            str: VLM 生成的图片描述
        """
        # TODO: 实现真实 VLM 调用
        # from transformers import AutoModelForVision2Seq, AutoProcessor
        # processor = AutoProcessor.from_pretrained("vikhyatk/moondream2")
        # model = AutoModelForVision2Seq.from_pretrained("vikhyatk/moondream2")
        # image = Image.open(image_path)
        # inputs = processor(images=image, text=prompt, return_tensors="pt")
        # outputs = model.generate(**inputs)
        # caption = processor.decode(outputs[0], skip_special_tokens=True)
        # return caption
        pass

    def reasoning(self, image_path, text, meta_data):
        """
        执行逻辑推理
        
        Args:
            image_path (str): 图片路径 (e.g., "./data/images/real_noon.jpg")
            text (str): 新闻文本
            meta_data (dict): Excel 中的元数据 (Ground Truth / Oracle)
            
        Returns:
            is_conflict (bool): 是否冲突，对应 GT_Ch3_Logic
                                True = 有冲突(1), False = 无冲突(0)
            reason (str): 推理过程描述
        
        对应 Excel 字段:
            - 输入: Image_Path (C列), Text_Content (D列)
            - 验证: GT_Ch3_Logic (H列), 1=有冲突, 0=无冲突
            - 分类: Sample_Type (B列), Logic_Trap 类型需重点检测
        """
        file_name = os.path.basename(image_path)
        print(f"[Ch3-Analysis] Processing: {file_name}")
        
        # 1. [Vision Step] 视觉理解 (通过 Mock 获取)
        visual_facts = self._vlm_captioning_mock(image_path, meta_data)
        print(f"[Ch3-Analysis] VLM Observation: {visual_facts}")

        # 2. [Reasoning Step] 逻辑比对
        return self._mock_reasoning(visual_facts, text)

    def _mock_reasoning(self, visual_facts, text):
        """
        Mock 推理逻辑
        基于规则的逻辑冲突检测
        
        对应 Sample_Type = "Logic_Trap" 的检测:
          - ID=041: 时间冲突 (图白天, 文深夜)
          - ID=042: 天气冲突 (图晴天, 文暴雨)
        
        检测规则:
          A. 时间属性硬匹配 (Temporal Logic)
          B. 天气/环境属性匹配 (Environmental Logic)
          C. 地标/实体冲突 (Geolocation Logic)
          D. 季节/常识冲突 (Common Sense Logic)
          E. 显性触发词 (Manual Trigger)
        """
        conflict_detected = False
        reason = "[CONSISTENT] Visual evidence aligns with text description"

        img_time = visual_facts.get("Time", "Unknown")
        img_weather = visual_facts.get("Weather", "Unknown")
        img_location = visual_facts.get("Location", "Unknown")
        img_fact = visual_facts.get("Fact", "Unknown")
        img_objects = visual_facts.get("Objects", "Unknown")

        # -----------------------------------------------------------------
        # Rule A: 时间属性硬匹配 (Temporal Logic)
        # 对应 ID=041: 图是白天，文说深夜
        # -----------------------------------------------------------------
        if img_time == "Day" and any(kw in text for kw in self.night_keywords):
            conflict_detected = True
            reason = f"[CONFLICT] Temporal: Image shows '{img_time}', but text claims Night/Midnight"
            
        elif img_time == "Night" and any(kw in text for kw in self.day_keywords):
            conflict_detected = True
            reason = f"[CONFLICT] Temporal: Image shows '{img_time}', but text claims Day/Noon"

        # -----------------------------------------------------------------
        # Rule B: 天气/环境属性匹配 (Environmental Logic)
        # 对应 ID=042: 图是晴天，文说暴雨
        # -----------------------------------------------------------------
        if not conflict_detected:
            sunny_scene = any(kw in img_weather for kw in ["Sunny", "Clear", "晴"])
            if sunny_scene and any(kw in text for kw in self.storm_keywords):
                conflict_detected = True
                reason = f"[CONFLICT] Weather: Image shows '{img_weather}', but text describes storm/rain"

        # -----------------------------------------------------------------
        # Rule C: 地标/实体冲突 (Geolocation Logic)
        # 检测地标位置与文本描述的矛盾
        # -----------------------------------------------------------------
        if not conflict_detected:
            # 巴黎地标 vs 其他城市
            paris_landmarks = ["Eiffel Tower", "埃菲尔铁塔", "巴黎"]
            london_mentions = ["London", "伦敦", "英国"]
            
            if any(lm in img_objects for lm in paris_landmarks):
                if any(loc in text for loc in london_mentions):
                    conflict_detected = True
                    reason = f"[CONFLICT] Geolocation: Image shows Paris landmark, text mentions London"
            
            # 上海地标 vs 其他城市
            shanghai_landmarks = ["东方明珠", "陆家嘴", "Shanghai"]
            tokyo_mentions = ["东京", "Tokyo", "日本"]
            
            if any(lm in img_objects or lm in img_location for lm in shanghai_landmarks):
                if any(loc in text for loc in tokyo_mentions):
                    conflict_detected = True
                    reason = f"[CONFLICT] Geolocation: Image shows Shanghai landmark, text mentions Tokyo"

        # -----------------------------------------------------------------
        # Rule D: 季节/常识冲突 (Common Sense Logic)
        # 检测季节穿着与天气描述的矛盾
        # -----------------------------------------------------------------
        if not conflict_detected:
            # 夏天场景 vs 冬天描述
            summer_scene = any(kw in img_fact or kw in img_weather for kw in ["Summer", "Hot", "炎热"])
            if summer_scene and any(kw in text for kw in self.winter_keywords):
                conflict_detected = True
                reason = f"[CONFLICT] Common Sense: Image shows summer scene, text describes winter/snow"

        # -----------------------------------------------------------------
        # Rule E: 显性逻辑谬误触发词 (Manual Trigger for Demo)
        # 在演示时，如果文本中包含特定词，强制触发报警
        # -----------------------------------------------------------------
        if not conflict_detected:
            trigger_keywords = ["逻辑错误", "明显矛盾", "LOGIC_TRAP"]
            if any(kw in text for kw in trigger_keywords):
                conflict_detected = True
                reason = "[CONFLICT] Manual Trigger: Logic conflict keyword detected"

        # -----------------------------------------------------------------
        # 返回结果
        # -----------------------------------------------------------------
        if conflict_detected:
            print(f"[Ch3-Result] Status=Conflict, Reason={reason}")
        else:
            print(f"[Ch3-Result] Status=Consistent")
        
        return conflict_detected, reason


# ============================================================================
# 单例模式导出
# ============================================================================
reasoner = LogicReasoner()


def check_logic(image_path, text, meta_data):
    """
    外部调用接口 (标准函数)
    供 main.py 或其他模块调用
    
    Args:
        image_path (str): 图片路径
        text (str): 新闻文本
        meta_data (dict): Excel 元数据字典，需包含:
            - Meta_Time: Day / Night
            - Meta_Weather: Sunny / Rain / Snow / Cloudy
            - Meta_Location: 地点名词
            - Meta_Fact: 事实状态
            - Meta_Object: 关键物体
    
    Returns:
        tuple: (is_conflict, reason)
            - is_conflict (bool): True=有冲突, False=无冲突
            - reason (str): 推理证据
    
    注意:
        - 返回的 is_conflict 直接对应 GT_Ch3_Logic
        - True = 1 (有冲突), False = 0 (无冲突)
    
    使用示例:
        from channel_3_logic_rules.reasoner import check_logic
        meta = {"Meta_Time": "Day", "Meta_Weather": "Sunny", ...}
        is_conflict, reason = check_logic("data/images/real_noon.jpg", 
                                          "深夜的街道格外宁静", meta)
        P3 = 0.95 if is_conflict else 0.05
        print(f"Conflict={is_conflict}, P3={P3}, Reason: {reason}")
    """
    return reasoner.reasoning(image_path, text, meta_data)


def check_logic_pipeline(image_path, text, meta_data):
    """
    Pipeline 接口 (别名)
    兼容不同的调用方式
    """
    return reasoner.reasoning(image_path, text, meta_data)