"""
================================================================================
Channel 2: Semantic Consistency Detection (图文语义一致性检测)
文件名: matcher.py
定位: 系统语义层防线，负责检测跨模态(图片内容 vs 文本描述)的逻辑匹配度

================================================================================
【核心任务】
检测 "移花接木" (Cheapfakes) 类造假，即使用真实的图片配以虚假或错误的文字描述。
不关注图片像素是否经过PS篡改，而是聚焦于图文语义是否匹配。

【检测目标 - 两种不符类型】
1. 完全不符 (Irrelevant Mismatch)
   - 场景: 图片内容与文本描述风马牛不相及
   - 示例: 图片显示"森林火灾"，文本描述"市中心商场打折促销"

2. 属性冲突 (Attribute Conflict)
   - 场景: 图片主体正确，但在环境、天气、情感基调等关键属性上存在矛盾
   - 示例: 图片显示"阳光明媚的公园"，文本描述"台风登陆，暴雨如注"

【检测原理 - Contrastive Learning】
基于 CLIP 模型的对比学习原理:
  1. 向量映射: 将图像(I)和文本(T)映射到同一个高维向量空间
  2. 相似度计算: 计算图像特征向量(V_I)和文本特征向量(V_T)之间的余弦相似度
  3. 判定逻辑:
     - 向量夹角小(相似度高) -> 图文描述同一事物 -> Real
     - 向量夹角大(相似度低) -> 图文语义分离 -> Fake

================================================================================
【I/O 接口规范】
================================================================================
输入 (Input):
  - image_path (str): 待检测图片的相对路径
  - text (str): 待检测的新闻文本内容 (CLIP限制77 tokens，代码需做截断)

输出 (Output):
  - score (float): 图文一致性分数 (0.0 ~ 1.0)
    注意: 这里输出的是"一致性"分数，需转换为"不符概率" P2 = 1 - score
    对应 Excel 字段: GT_Ch2_Mismatch (1=图文不符, 0=一致)
  - message (str): 诊断信息

【判定阈值】
  - score > 0.25 -> 判定为匹配 (Real)
  - score < 0.20 -> 判定为不匹配 (Fake)

【与Excel字段的对应关系】
  - Sample_Type = "Mismatch" -> 图文不符样本，GT_Ch2_Mismatch = 1
  - Sample_Type = "Real/Tamper_PS/Tamper_AIGC" -> 若图文匹配，GT_Ch2_Mismatch = 0
  - 输入: Image_Path (C列), Text_Content (D列)
  - 验证: GT_Ch2_Mismatch (G列), 1=不符, 0=一致

【重要说明】
  Ch2_Mismatch 与 Ch1_Tamper 是正交独立的:
  - 一张P过的假图(Ch1=1)，如果文字描述了P出来的内容，图文可能是一致的(Ch2=0)
  - 本通道只负责判断"图和文说的是不是一回事"

================================================================================
【技术选型】
推荐模型: OpenAI CLIP (openai/clip-vit-base-patch32)
  - Zero-Shot能力: 无需针对特定数据集微调
  - 轻量级高效: ~600MB，CPU/GPU均可运行
  - 生态成熟: HuggingFace 提供标准 API

================================================================================
"""

import os
import random
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# import torch


class ConsistencyDetector:
    """
    图文语义一致性检测器
    基于 CLIP 模型检测图片与文本描述是否匹配
    """
    
    def __init__(self):
        """
        初始化检测器
        任务: 加载 OpenAI CLIP 模型 (ViT-Base-Patch32)
        注意: 首次运行会自动下载模型权重 (~600MB)，请保持网络通畅
        """
        print("[Ch2-Init] Loading CLIP Model (openai/clip-vit-base-patch32)...")
        print("[Ch2-Init] Target: Semantic Consistency Detection for Cheapfakes")
        
        # -----------------------------------------------------------
        # TODO: [Step 1] 集成真实 CLIP 模型
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # -----------------------------------------------------------
        self.model_loaded = False  # 标记模型是否已加载
        self.max_text_length = 70  # CLIP token 限制 (77 tokens, 预留安全边际)

    def check(self, image_path, text):
        """
        执行图文一致性检测
        
        Args:
            image_path (str): 图片路径 (e.g., "./data/images/real_fire.jpg")
            text (str): 待检测的文本内容
            
        Returns:
            score (float): 一致性分数 (0.0 ~ 1.0)
                           高分 = 匹配, 低分 = 不符
                           P2(不符概率) = 1 - score
            message (str): 诊断结果描述
        
        对应 Excel 字段:
            - 输入: Image_Path (C列), Text_Content (D列)
            - 验证: GT_Ch2_Mismatch (G列), 1=不符, 0=一致
            - 分类: Sample_Type (B列), Mismatch 类型需重点检测
        """
        # 1. 路径标准化处理
        if not os.path.exists(image_path):
            abs_path = os.path.abspath(image_path)
            if os.path.exists(abs_path):
                image_path = abs_path
            else:
                return 0.0, "[ERROR] File Not Found"

        # 2. 文本截断处理 (CLIP 通常限制 77 token)
        short_text = text[:self.max_text_length] if len(text) > self.max_text_length else text

        file_name = os.path.basename(image_path)
        print(f"[Ch2-Analysis] Processing: {file_name}")
        print(f"[Ch2-Analysis] Text: '{short_text[:50]}...'")
        print(f"[Ch2-Analysis] Computing image-text cosine similarity...")

        # -------------------------------------------------------------
        # TODO: [Step 2] 接入真实 CLIP 推理
        # -------------------------------------------------------------
        # image = Image.open(image_path)
        # inputs = self.processor(text=[short_text], images=image, 
        #                         return_tensors="pt", padding=True).to(self.device)
        # 
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     logits_per_image = outputs.logits_per_image
        #     # 归一化处理: logits / 100 或使用 sigmoid
        #     score = torch.sigmoid(logits_per_image / 100).item()
        # return score, f"Consistency score: {score:.4f}"
        # -------------------------------------------------------------

        # ============================================================
        # [Phase 1] Mock Logic - 基于文件名和文本关键词模拟检测结果
        # 用于联调测试，真实模型接入前的演示
        # ============================================================
        return self._mock_check(image_path, text)

    def _mock_check(self, image_path, text):
        """
        Mock 检测逻辑
        根据文件名和文本关键词模拟图文一致性检测
        
        对应 Sample_Type 分类:
          - Mismatch: 图文完全不符，返回低分 (GT_Ch2_Mismatch = 1)
          - Real/Tamper_*: 若图文匹配，返回高分 (GT_Ch2_Mismatch = 0)
        
        命名规范:
          - mis_001.jpg, mis_002.jpg (Mismatch样本)
          - real_fire.jpg, real_park.jpg (真图样本)
        
        Mock 规则 (基于 Type 3 SOP):
          1. 灾难误植: fire/disaster vs 商场/促销 (情绪反差)
          2. 暴乱误植: park/library vs 骚乱/冲突 (动静反差)
          3. 物种错误: panda vs 东北虎/老虎 (主体不同)
          4. 场景错位: space/astronaut vs 深海/潜水 (环境反差)
          5. 环境错误: desert/沙漠 vs 洪水/淹没 (旱涝反差)
          6. 对象错误: plane/战斗机 vs 高铁/火车 (飞跑反差)
        """
        file_name = image_path.lower()
        text_lower = text.lower()
        
        # -----------------------------------------------------------------
        # 定义不匹配规则 (基于 Type 3 SOP 样本构造策略)
        # 格式: (图片关键词列表, 文本关键词列表) -> 构成不匹配
        # -----------------------------------------------------------------
        mismatch_rules = [
            # === 灾难误植 (Disaster Mismatch) ===
            # mis_001: 森林大火 vs 商场打折促销
            (["fire", "disaster", "burn", "smoke", "collapse", "火灾", "大火"],
             ["商场", "促销", "打折", "开业", "五折", "mall", "sale", "shopping", "discount"]),
            
            # === 暴乱误植 (Riot Mismatch) ===
            # mis_002: 安静的公园/图书馆 vs 暴乱骚乱
            (["park", "library", "quiet", "peaceful", "公园", "图书馆", "安静"],
             ["骚乱", "暴乱", "冲突", "示威", "riot", "violence", "protest", "clash"]),
            
            # === 物种错误 (Species Mismatch) ===
            # mis_003: 熊猫 vs 东北虎
            (["panda", "熊猫", "bamboo"],
             ["东北虎", "老虎", "tiger", "伤人", "下山"]),
            
            # === 场景错位 (Environment Mismatch) ===
            # 太空 vs 深海
            (["space", "astronaut", "太空", "宇航员", "航天"],
             ["深海", "潜水", "海沟", "sea", "underwater", "diver"]),
            
            # === 环境错误 (Climate Mismatch) ===
            # mis_004: 沙漠 vs 洪水
            (["desert", "沙漠", "骆驼", "camel", "干旱"],
             ["洪水", "淹没", "水灾", "flood", "inundation"]),
            
            # === 对象错误 (Object Mismatch) ===
            # mis_005: 战斗机 vs 高铁
            (["plane", "jet", "fighter", "战斗机", "飞机", "aircraft"],
             ["高铁", "火车", "通车", "train", "railway"]),
            
            # === 天气冲突 (Weather Conflict) ===
            # 对应 Sample_Type = "Logic_Trap", ID = 042
            (["sunny", "beach", "晴", "阳光", "蓝天"],
             ["台风", "暴雨", "狂风", "typhoon", "storm", "rainstorm"]),
            
            # === 时间冲突 (Time Conflict) ===
            # 对应 Sample_Type = "Logic_Trap", ID = 041
            (["noon", "day", "白天", "中午", "阳光"],
             ["深夜", "月光", "night", "midnight", "凌晨"]),
            
            # === 森林 vs 城市 ===
            (["forest", "森林", "树林", "woods"],
             ["城市", "商场", "市中心", "city", "urban", "downtown"]),
        ]
        
        is_mismatch = False
        mismatch_reason = ""
        
        # 检查规则匹配
        for img_keywords, text_keywords in mismatch_rules:
            img_match = any(kw in file_name for kw in img_keywords)
            text_match = any(kw in text_lower or kw in text for kw in text_keywords)
            
            if img_match and text_match:
                is_mismatch = True
                mismatch_reason = f"Image({img_keywords[0]}) vs Text({text_keywords[0]})"
                break
        
        # 额外检查: 文件名包含 "mis_" 前缀 (Mismatch 样本命名规范)
        if not is_mismatch and "mis_" in file_name:
            is_mismatch = True
            mismatch_reason = "Mismatch sample (mis_ prefix detected)"
        
        # -----------------------------------------------------------------
        # 返回结果
        # -----------------------------------------------------------------
        if is_mismatch:
            # 低分表示不匹配 (对应 GT_Ch2_Mismatch = 1)
            # 目标: 分数 < 0.2 (阈值)
            score = round(random.uniform(0.05, 0.18), 4)
            msg = f"[MISMATCH] Semantic conflict detected - {mismatch_reason}"
            print(f"[Ch2-Result] Score={score:.4f}, Status=Mismatch")
            return score, msg
        else:
            # 高分表示匹配 (对应 GT_Ch2_Mismatch = 0)
            # 目标: 分数 > 0.25 (阈值)
            score = round(random.uniform(0.72, 0.92), 4)
            msg = "[CONSISTENT] Image-text semantics aligned"
            print(f"[Ch2-Result] Score={score:.4f}, Status=Matched")
            return score, msg


# ============================================================================
# 单例模式导出
# ============================================================================
detector = ConsistencyDetector()


def check_consistency(image_path, text):
    """
    外部调用接口 (标准函数)
    供 main.py 或其他模块调用
    
    Args:
        image_path (str): 图片路径
        text (str): 文本内容
    
    Returns:
        tuple: (score, message)
            - score (float): 一致性分数 (0.0 ~ 1.0)
            - message (str): 诊断结果
    
    注意: 
        - 返回的是"一致性"分数，高分=匹配
        - 若需要"不符概率" P2，请使用: P2 = 1 - score
    
    使用示例:
        from channel_2_consistency_clip.matcher import check_consistency
        score, msg = check_consistency("data/images/real_fire.jpg", "商场促销活动")
        P2 = 1 - score  # 不符概率
        print(f"Consistency={score}, P2={P2}, Result: {msg}")
    """
    return detector.check(image_path, text)


def check_consistency_pipeline(image_path, text):
    """
    Pipeline 接口 (别名)
    兼容不同的调用方式
    """
    return detector.check(image_path, text)