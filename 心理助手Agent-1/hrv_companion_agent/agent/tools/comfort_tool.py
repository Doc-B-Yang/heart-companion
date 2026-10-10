# -*- coding: utf-8 -*-
"""
心理技能匹配引擎 (comfort_tool)
================================
对应计划书 §4.3 tools/comfort_tool.py，v2 由「情绪→话术表」升级为
「多维匹配引擎」。

**为什么把话术表改成匹配引擎**

v1 的做法是 `情绪 → 3 条固定话术`。这在演示里够用，但有三处不专业：

1. **同一情绪、不同唤醒度需要不同的技术。** 「高焦虑」的高唤醒形态
   需要**降生理唤醒**（延长呼气、生理性叹息），低唤醒形态（耗竭型焦虑）
   需要**先激活再梳理**，给同一套呼吸话术是错的。
2. **忽略个体差异。** 有人的焦虑是「灾难化思维」驱动，有人是「回避行为」
   驱动。同一句话对不同人的效果完全不同 —— 系统里已有 `feedback` 表
   记录哪些策略对这位用户真的有效，却没用上。
3. **忽略技术禁忌。** 例如渐进式肌肉放松在惊恐发作时可能因「关注躯体
   感受」而加重症状；蝴蝶拍对某些创伤经历者可能诱发不适。专业系统必须
   把禁忌写出来，而不是「什么情绪都推 54321」。

**匹配打分（加权和，全部可解释）**

    总分 = 4.0 × 情绪匹配
         + 3.0 × 唤醒适配
         + 2.0 × 强度适配
         + 2.5 × 该用户的历史有效率（文献4 反馈闭环）
         - 3.0 × 最近刚用过（避免重复感）

**参考来源**：CBT（贝克认知疗法）、DBT（辩证行为疗法）、
ACT（接纳承诺疗法）、正念减压（MBSR）、躯体体验（Somatic）、
行为激活（BA）、Kristin Neff 自我关怀、睡眠刺激控制等循证技术。
各条技术的适用与禁忌为本项目依循证文献整理，**不构成治疗建议**。

.. note::
   本文件所有对内嵌引文统一使用「」与（）—— 避免中文正文里出现
   ASCII 双引号与 Python 字符串定界符冲突（这是一个真实踩过的坑：
   技能库有 30+ 条内嵌引文，逐条转义既易错又难维护）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base_tool import BaseTool

# ---------------------------------------------------------------------------
# 技能库
# ---------------------------------------------------------------------------
# 字段说明:
#   id            机器名（用于 feedback 统计与去重）
#   name          人话名称
#   school        所属流派（便于解释「为什么这样做」）
#   targets       适用的 5 类情绪
#   arousal       "high" 需要降唤醒 / "low" 需要先激活 / "any" 都适用
#   intensity_min 最低适用强度（0~1，来自情感动力学 Et）
#   mechanism     起效机制（一句话，让模型知道「为什么」而非「照念」）
#   script        可直接融入对话的引导脚本
#   duration      建议时长
#   caution       禁忌 / 需要留意的情形（**必须尊重**）
SKILLS: List[dict] = [
    # ---------------------------------------------------- 躯体 / 降唤醒
    {
        "id": "physiological_sigh",
        "name": "生理性叹息",
        "school": "躯体 / 呼吸生理",
        "targets": ["高焦虑"],
        "arousal": "high",
        "intensity_min": 0.0,
        "mechanism": "双吸气 + 长呼气能快速复张塌陷的肺泡并提高迷走张力，"
                     "是目前起效最快的单次呼吸干预（一次即见效，无需练习）。",
        "script": "连着吸两口气——第二口短一点、把肺再撑满一些——"
                  "然后用嘴慢慢地、长长地呼出去，像叹气一样。"
                  "重复 3 次就好。",
        "duration": "30 秒",
        "caution": "几乎无禁忌，是所有高唤醒状态的首选第一步。",
    },
    {
        "id": "grounding_54321",
        "name": "5-4-3-2-1 感官着陆",
        "school": "正念 / 创伤知情（接地技术）",
        "targets": ["高焦虑"],
        "arousal": "high",
        "intensity_min": 0.0,
        "mechanism": "焦虑与解离的共同问题是注意力被困在内部（念头、躯体感）"
                     "与未来。把注意力强制分配到**外部、当下、具体**的感官信息上，"
                     "可直接打断反刍循环。",
        "script": "我们做个小练习，你慢慢跟着来："
                  "说出 5 个你能**看到**的东西……4 个能**摸到**的东西……"
                  "3 个能**听到**的声音……2 个能**闻到**的气味……"
                  "最后 1 个——你今天做得还行的一件事。不着急，一个一个来。",
        "duration": "2-3 分钟",
        "caution": "若用户正在解离或创伤闪回，改用以「感受到支撑」为主的"
                   "方式（脚踩地、背靠椅），不要让ta去注意身体内部感受。",
    },
    {
        "id": "butterfly_hug",
        "name": "蝴蝶拍（双侧自我安抚）",
        "school": "躯体 / EMDR 衍生",
        "targets": ["高焦虑", "低焦虑"],
        "arousal": "high",
        "intensity_min": 0.3,
        "mechanism": "双侧交替的节律性触觉输入配合缓慢呼吸，"
                     "对安抚系统有直接激活作用；且动作简单、无需认知资源。",
        "script": "把双臂交叉抱在胸前，双手轻轻搭在对侧肩膀上。"
                  "左手拍一下、右手拍一下，像哄小孩那样慢慢来，"
                  "一边拍一边在心里说「我现在是安全的」。做 1-2 分钟。",
        "duration": "1-2 分钟",
        "caution": "有创伤经历且被该动作引发不适时**立即停止**，"
                   "换成脚踩地、握握手边的物体。",
    },
    {
        "id": "progressive_relax",
        "name": "渐进式肌肉放松",
        "school": "躯体（Jacobson）",
        "targets": ["高焦虑", "疲劳"],
        "arousal": "high",
        "intensity_min": 0.0,
        "mechanism": "「先绷紧再释放」制造鲜明的对比感受，"
                     "让人第一次真切地认识到「原来我是绷着的」，"
                     "从而学会主动松下来。",
        "script": "从脚趾开始：用力蜷紧 5 秒……然后一下全松开，"
                  "感受那种沉下去的感觉，停 10 秒。再往上到小腿、大腿、"
                  "肚子、肩膀、手、脸。我们做到肩膀就够，不着急。",
        "duration": "5-8 分钟",
        "caution": "⚠️ **惊恐发作期间不适用** —— 关注躯体感受可能加重症状。"
                   "此时应优先用生理性叹息或外部着陆。",
    },
    {
        "id": "cold_water_anchor",
        "name": "冷水 / 温度锚定",
        "school": "躯体（迷走神经激活）",
        "targets": ["高焦虑"],
        "arousal": "high",
        "intensity_min": 0.5,
        "mechanism": "面部接触冷水会触发潜水反射，迅速降低心率 —— "
                     "这是少数能在 30 秒内直接改变生理唤醒状态的物理手段。",
        "script": "如果方便，去用凉水洗一下脸，或者用凉水冲一下手腕内侧，"
                  "停 30 秒。这个动作能从生理上直接把心跳压下来。",
        "duration": "1 分钟",
        "caution": "有心脑血管疾病者避免；不要用冰水，凉水即可。",
    },
    # ---------------------------------------------------- CBT（认知）
    {
        "id": "decatastrophizing",
        "name": "去灾难化（CBT）",
        "school": "CBT（认知重构）",
        "targets": ["高焦虑", "低焦虑"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "焦虑的维持机制是把最坏结果当成最可能结果的自动化预期。"
                     "把它写下来、拆成概率与代价两问，可削弱其情绪力量。",
        "script": "我们把它拆开看，不用急着结论："
                  "① 你现在最担心的那个结果是**具体**什么？"
                  "② 它发生的可能性，你估计有几成？"
                  "③ 万一真的发生，你能做的第一件事是什么？"
                  "（先问 ①，用户答完再问 ②，别一口气问三个。）",
        "duration": "5-10 分钟",
        "caution": "不要在用户情绪强度很高（Et>0.7）时直接做认知工作 —— "
                   "先降唤醒，再动认知。",
    },
    {
        "id": "evidence_check",
        "name": "证据检验（苏格拉底式提问）",
        "school": "CBT",
        "targets": ["低焦虑", "高焦虑"],
        "arousal": "low",
        "intensity_min": 0.0,
        "mechanism": "自动思维往往缺少证据支持却极具说服力。"
                     "让用户自己找支持证据与反对证据，比反驳更有效 —— "
                     "人是被**自己**想到的反对证据说服的。",
        "script": "如果让你当一次法官：**支持**这个想法的证据有哪些？"
                  "**反对**它的证据又有哪些？可以慢慢想，我等你。",
        "duration": "5-10 分钟",
        "caution": "不要在用户还没感受到被理解时使用，会显得像辩论。",
    },
    {
        "id": "worry_timebox",
        "name": "担忧时间盒",
        "school": "CBT（刺激控制）",
        "targets": ["低焦虑", "高焦虑"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "反刍之所以耗人，是因为它**随时发生**。"
                     "把担忧限制在固定时段，既保留其功能（担忧本身有准备作用），"
                     "又把它从全天候降为可控事件。",
        "script": "我们试试给它划个时间：每天固定 15 分钟，比如晚上 9 点，"
                  "专门用来想这些事 —— 想多久都行。其它时候它冒出来，"
                  "就先记在纸上，告诉自己「到点再想」。",
        "duration": "约定 + 每日执行",
        "caution": "睡前 2 小时内不要设置担忧时段（会影响入睡）。",
    },
    {
        "id": "cognitive_continuum",
        "name": "连续性技术（打破非黑即白）",
        "school": "CBT",
        "targets": ["低焦虑", "疲劳"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "「我不行」「我很失败」这类结论依赖全或无的判断方式。"
                     "把结果放到 0-100 的连续刻度上，极端结论会立刻变得不成立。",
        "script": "如果 0 分是「完全没做到」，100 分是「做到完美」，"
                  "你觉得这件事你现在落在几分？"
                  "（用户答完）—— 那说明它不是 0 分，对吗？",
        "duration": "5 分钟",
        "caution": "不要用这个技术否定用户的情绪（例如「你看你没那么差嘛」）。",
    },
    {
        "id": "double_standard",
        "name": "双重标准（对朋友视角）",
        "school": "CBT / 自我关怀",
        "targets": ["低焦虑", "疲劳"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "自责往往在「对自己」和「对别人」之间存在巨大落差。"
                     "借用这个落差，让用户听见自己的严苛，而不是被指出严苛。",
        "script": "如果是你最好的朋友跟你说了同样的事、同样的话，"
                  "你会怎么回ta？……那你能把这句话也说给自己听吗？",
        "duration": "3-5 分钟",
        "caution": "若用户随即出现强烈羞耻感，停下来先做接纳"
                   "（例如「这样对自己说话，真的很累吧」）。",
    },
    # ---------------------------------------------------- ACT（接纳 / 价值）
    {
        "id": "thought_defusion",
        "name": "认知解离（ACT）",
        "school": "ACT",
        "targets": ["低焦虑", "高焦虑"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "ACT 的目标不是改变想法内容，而是改变与想法的**关系**。"
                     "加上「我注意到我在想……」之后，人从「想法即现实」"
                     "变成「我在观察想法」。",
        "script": "试试在这个念头前面加一句：「我注意到我正在想——」。"
                  "完整念一遍：我注意到我正在想……"
                  "（等用户念）……感觉这句和刚才有什么不一样吗？",
        "duration": "2-3 分钟",
        "caution": "对高度解离者可能加重疏离感，此时改用着陆技术。",
    },
    {
        "id": "acceptance_turn",
        "name": "接纳转向（情绪的功能）",
        "school": "ACT / DBT",
        "targets": ["高焦虑", "低焦虑", "疲劳"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "「不要焦虑」这个指令本身会制造二次痛苦"
                     "（为焦虑而焦虑）。承认情绪的功能，痛苦就从故障"
                     "变回信号 —— 而信号是可以被读取、被使用的。",
        "script": "焦虑其实是有用的东西 —— 它在提醒你「这件事对你很重要」。"
                  "我们不用赶走它，只是不让它替你做决定。"
                  "你觉得它现在想让你做什么？",
        "duration": "3-5 分钟",
        "caution": "接纳不等于放弃行动 —— 说完要接一个**具体的下一步**，"
                   "否则会变成消极认命。",
    },
    {
        "id": "values_compass",
        "name": "价值罗盘（ACT）",
        "school": "ACT",
        "targets": ["疲劳", "平静", "低焦虑"],
        "arousal": "low",
        "intensity_min": 0.0,
        "mechanism": "迷茫与耗竭常源于目标与价值脱钩。"
                     "回到「我在乎什么」，比追问「我该做什么」更能恢复动力。",
        "script": "先不聊「该怎么办」。我想知道：在这件事上，"
                  "你真正在乎的是什么？比如是想被认可，还是想成为"
                  "某种自己看得起的人？",
        "duration": "5-10 分钟",
        "caution": "不要用价值去要求用户行动，会变成道德施压。",
    },
    # ---------------------------------------------------- 行为激活
    {
        "id": "behavioral_activation",
        "name": "最小行动（行为激活）",
        "school": "行为激活（BA）",
        "targets": ["疲劳", "低焦虑"],
        "arousal": "low",
        "intensity_min": 0.0,
        "mechanism": "低落时「先有动力再行动」是反的 —— 行为激活的原理是"
                     "**行动先于动力**。任务必须小到「不做反而更费劲」。",
        "script": "我们把门槛降到最低：不是「去学习」，而是"
                  "**打开书，读一行**。就一行。做完就不用管了。"
                  "你愿意试一下吗？（用户答应后要真的肯定完成，不要加码。）",
        "duration": "5 分钟",
        "caution": "⚠️ 完成后**不要**顺势加码（例如「既然打开书了不如多看两页」）"
                   "—— 这会立刻摧毁刚才建立的成功体验。",
    },
    {
        "id": "opposite_action",
        "name": "相反行动（DBT）",
        "school": "DBT（情绪调节）",
        "targets": ["高焦虑", "低焦虑"],
        "arousal": "any",
        "intensity_min": 0.3,
        "mechanism": "情绪会驱动行为（焦虑→回避、低落→躺平），"
                     "而这些行为又反过来维持情绪。有意识地做相反动作，"
                     "可以切断这个循环。",
        "script": "焦虑让你想做什么？……逃避、躲起来、刷手机，对吗。"
                  "那我们就故意做一点点相反的事：不要求你面对全部，"
                  "只是**先不关掉**那个页面，或者给那个人回一句话。",
        "duration": "5-10 分钟",
        "caution": "⚠️ **不适用于真正危险的情境** —— 若焦虑指向真实威胁"
                   "（如人身安全），相反行动是错的。先评估现实风险。",
    },
    # ---------------------------------------------------- 自我关怀 / 人际
    {
        "id": "self_compassion_break",
        "name": "自我关怀三要素",
        "school": "正念（Neff）",
        "targets": ["低焦虑", "疲劳"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "自我关怀由三部分组成：正念（承认痛苦）、共通人性"
                     "（我不是唯一这样的人）、自我友善（像对朋友那样对我）。"
                     "三者缺一都会滑向自怜或压抑。",
        "script": "三句话，你可以在心里说："
                  "①「这真的很难受」——先承认；"
                  "②「很多人在经历类似的事」——不是你一个人；"
                  "③「我可以对自己好一点」——像对朋友那样。",
        "duration": "3-5 分钟",
        "caution": "有些人对自我友善的第一反应是羞耻或想哭，"
                   "要把节奏放慢，并说明「想哭也很正常」。",
    },
    {
        "id": "affect_labeling",
        "name": "情绪命名（精确化）",
        "school": "情绪科学 / 跨诊断",
        "targets": ["高焦虑", "低焦虑", "疲劳", "平静"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "把模糊的不适准确命名，本身就能降低情绪强度"
                     "（情绪标注效应，与杏仁核活动下降相关）。"
                     "关键是**具体**：「委屈」比「难受」有用，"
                     "「被忽视的愤怒」比「生气」有用。",
        "script": "如果不用「难受」这个词，你会怎么形容？"
                  "是委屈、是着急、是失望、还是别的？"
                  "……说出来的那一刻，往往就已经松一点了。",
        "duration": "2-3 分钟",
        "caution": "不要替用户命名（例如「你其实是嫉妒吧」）—— "
                   "命名必须来自用户本人。",
    },
    {
        "id": "interpersonal_script",
        "name": "人际沟通脚本（DBT 简化）",
        "school": "DBT（人际效能）",
        "targets": ["低焦虑", "高焦虑"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "人际冲突的伤害多来自表达方式而非诉求本身。"
                     "「描述事实 → 表达感受 → 提出请求」这个结构"
                     "能显著降低对方的防御反应。",
        "script": "我们先把话排一下，不用现在就发："
                  "① **事实**（只说发生的，不加评价）：「你上次答应的事没做」；"
                  "② **感受**：「我有点失望」；"
                  "③ **请求**：「下次如果做不到，能提前告诉我吗？」",
        "duration": "5-10 分钟",
        "caution": "若对方存在人身威胁或控制行为，**不要**建议直接沟通 —— "
                   "安全优先，应转为求助与脱离方案。",
    },
    {
        "id": "micro_connection",
        "name": "微社交连接",
        "school": "行为激活 / 社会支持",
        "targets": ["疲劳", "低焦虑"],
        "arousal": "low",
        "intensity_min": 0.0,
        "mechanism": "孤独感的干预不需要「建立深厚关系」，"
                     "一次低风险的微小连接（一句问候）就能显著降低主观孤独。"
                     "门槛定得越低，越可能真的发生。",
        "script": "不用「找人谈心」这么大的动作。给一个老朋友发三个字："
                  "「在吗？」—— 发了就算完成，回不回都无所谓。"
                  "要不要试试？",
        "duration": "1 分钟",
        "caution": "若用户曾被重要他人伤害，建议先选最安全的对象，"
                   "而不是「最亲近的人」。",
    },
    # ---------------------------------------------------- 睡眠 / 积极
    {
        "id": "sleep_stimulus_control",
        "name": "睡眠刺激控制",
        "school": "失眠认知行为治疗（CBT-I）核心组件",
        "targets": ["疲劳", "高焦虑"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "躺着睡不着会让「床」与「清醒焦虑」建立条件反射。"
                     "刺激控制（睡不着就起来）是目前证据最强的单一失眠干预，"
                     "比「早点睡」有效得多。",
        "script": "今晚试试这个：躺下 20 分钟还没睡着，就起来，"
                  "去别的房间做点无聊的事（别刷手机），有困意了再回床上。"
                  "床只用来睡，不用来想事情。",
        "duration": "约 3-7 天见效",
        "caution": "⚠️ 白天极度嗜睡或有跌倒风险者慎用；"
                   "若长期严重失眠，应就医而不是自行调整。",
    },
    {
        "id": "savoring",
        "name": "积极体验放大",
        "school": "积极心理学（Savoring）",
        "targets": ["愉悦", "平静"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "积极体验若不主动加工，几分钟内就会被「适应」抹平。"
                     "刻意停留、复述、分享，能延长其情绪效应并强化记忆痕迹。",
        "script": "别急着往下说 —— 我们在这里多停一会儿。"
                  "那一刻你身体有什么感觉？周围是什么样子的？"
                  "把好的那部分在心里多放一会儿。",
        "duration": "2-3 分钟",
        "caution": "若用户正处于低落期，「被要求开心」会变成负担 —— "
                   "此时改用「今天有没有哪一小会儿没那么难」的问法。",
    },
    {
        "id": "resource_anchoring",
        "name": "资源锚定（回忆一次有效应对）",
        "school": "焦点解决（SFBT）",
        "targets": ["疲劳", "低焦虑", "平静"],
        "arousal": "any",
        "intensity_min": 0.0,
        "mechanism": "焦点解决的核心假设：用户**已经**拥有解决问题的资源。"
                     "找出「上一次你是怎么扛过来的」，比给新建议更能提升"
                     "效能感 —— 因为那是ta自己的成功经验。",
        "script": "以前有没有过类似的时刻，后来慢慢过去了？"
                  "那次是什么起了作用？……你看，你其实知道办法，"
                  "只是现在累得想不起来了。",
        "duration": "5 分钟",
        "caution": "若用户回答「从来没有过」，不要追问 —— "
                   "改为承认「一直一个人扛着，确实太累了」。",
    },
]

#: 通用兜底（不匹配任何特定情绪时）
GENERIC_SKILLS = ["affect_labeling", "acceptance_turn", "resource_anchoring"]

#: 通用前置框架 —— **任何技术之前都要先做这一步**
#:
#: 这不是"兜底话术"，而是一条临床常识：技术只有在用户感到被接纳之后
#: 才起效。跳过它直接给技巧，会被体验成"你只是想把我修好"。
#: 因此当情绪未识别（或识别结果不可信）时，强制把这两条放在最前面。
GENERIC_FRAMING = [
    ("无条件接纳",
     "先告诉用户：无论ta此刻是什么样子，在这里都是安全、被接纳的。"
     "先接住情绪，再谈事情 —— 顺序错了，后面所有技术都会打折。"),
    ("陪伴在场",
     "表达「我在」「我陪着你」的在场感，不要急着解决问题。"
     "有时被听见本身就是疗愈。"),
]

#: 各维权重（可解释：改这里就等于改匹配策略）
WEIGHT_EMOTION = 4.0
WEIGHT_AROUSAL = 3.0
WEIGHT_INTENSITY = 2.0
WEIGHT_HISTORY = 2.5
PENALTY_RECENT = 3.0

_BY_ID = {s["id"]: s for s in SKILLS}

#: 情绪 → 默认唤醒倾向（PAD 不可用时的兜底先验）
_EMOTION_AROUSAL = {"高焦虑": "high", "低焦虑": "high",
                    "疲劳": "low", "平静": "any", "愉悦": "any"}


def get_skill(skill_id: str) -> Optional[dict]:
    """按 id 取技能（供 feedback 统计与"再教我一次"使用）"""
    return _BY_ID.get(skill_id)


def all_skill_ids() -> List[str]:
    return [s["id"] for s in SKILLS]


def match_skills(emotion: str = "", arousal: str = "",
                 intensity: float = 0.0,
                 stats: Optional[Dict[str, dict]] = None,
                 recent_ids: Optional[List[str]] = None,
                 top_k: int = 3) -> List[dict]:
    """按「情绪 × 唤醒 × 强度 × 个人有效性 × 新鲜度」匹配技能

    Args:
        emotion: 5 类中文情绪
        arousal: "high" / "low" / "any"（可由 PADTool 的 A 轴给出）
        intensity: 情感动力学强度 Et ∈ [0,1]
        stats: {skill_id: {"total": n, "effective": m}} 个人历史有效性
        recent_ids: 最近已用过的技能 id（扣分，避免重复说同一套）
        top_k: 返回条数

    Returns:
        排序后的技能列表，每项额外带 ``_score`` 与 ``_why``（匹配理由）。
        打分全部来自显式加权，故可解释 —— 这对"为什么推荐这个技术"
        的可追溯性很重要。
    """
    stats = stats or {}
    recent = set(recent_ids or [])
    if not arousal:
        arousal = _EMOTION_AROUSAL.get(emotion, "any")

    scored = []
    for s in SKILLS:
        score = 0.0
        why: List[str] = []

        # ---- ① 情绪匹配 ----
        if emotion and emotion in s["targets"]:
            score += WEIGHT_EMOTION
            why.append("适用于「%s」" % emotion)
        elif not emotion:
            score += WEIGHT_EMOTION * 0.5
            why.append("通用适用")

        # ---- ② 唤醒适配（高唤醒优先降唤醒；低唤醒优先激活）----
        if s["arousal"] == "any":
            score += WEIGHT_AROUSAL * 0.6
        elif arousal == "any":
            score += WEIGHT_AROUSAL * 0.5
        elif s["arousal"] == arousal:
            score += WEIGHT_AROUSAL
            why.append("唤醒度匹配（%s）" % ("需降唤醒" if arousal == "high"
                                             else "需先激活"))

        # ---- ③ 强度适配 ----
        if intensity >= s["intensity_min"]:
            score += WEIGHT_INTENSITY
        else:
            # 强度不够时仍可给，但降权（如蝴蝶拍在轻焦虑下收益低）
            score += WEIGHT_INTENSITY * 0.3

        # ---- ④ 个人历史有效性（反馈闭环，文献4）----
        st = stats.get(s["id"]) or {}
        tot = int(st.get("total") or 0)
        if tot >= 2:
            rate = int(st.get("effective") or 0) / tot
            score += WEIGHT_HISTORY * rate
            if rate >= 0.6:
                why.append("对这位用户有效率高（%.0f%%，%d次）"
                           % (rate * 100, tot))
            elif rate < 0.35:
                why.append("此前效果一般（%.0f%%，%d次）" % (rate * 100, tot))
        else:
            score += WEIGHT_HISTORY * 0.5   # 数据不足时给中性分

        # ---- ⑤ 新鲜度惩罚 ----
        if s["id"] in recent:
            score -= PENALTY_RECENT
            why.append("最近刚用过（优先换一种）")

        scored.append({**s, "_score": round(score, 2),
                       "_why": "；".join(why)})

    scored.sort(key=lambda x: -x["_score"])
    return scored[:max(1, top_k)]


class ComfortTool(BaseTool):
    name = "comfort_tool"
    description = (
        "**心理技能匹配引擎**：按当前情绪 × 唤醒度 × 强度 × 该用户历史有效性，"
        "匹配最合适的循证心理技术（CBT/DBT/ACT/正念/躯体/行为激活等），"
        "并给出引导脚本与**技术禁忌**。"
        "调用格式: [TOOL_CALL:comfort_tool:emotion=高焦虑] 或 "
        "[TOOL_CALL:comfort_tool:emotion=疲劳,arousal=low,intensity=0.6] "
        "参数: emotion(5类) / arousal(high|low) / intensity(0~1) / "
        "top_k(默认3) / skill(直接指定技能id)。"
        "返回的 caution 字段必须遵守 —— 技术禁忌比适用更重要。")

    def __init__(self, skill_stats_provider=None, recent_provider=None):
        """
        Args:
            skill_stats_provider: 无参回调 → {skill_id: {total, effective}}
            recent_provider: 无参回调 → 最近用过的 skill_id 列表
        """
        self._stats_provider = skill_stats_provider
        self._recent_provider = recent_provider
        #: 最近一次匹配出的技能 id（供调用方记录"刚用过"，实现新鲜度惩罚）
        self.last_ids: List[str] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _norm_emotion(raw: str) -> str:
        """把 v1 写法与英文键都归一到 5 类情绪（保证向上兼容）"""
        t = (raw or "").strip()
        alias = {
            "anxiety": "高焦虑", "anxious": "高焦虑", "anxiety_high": "高焦虑",
            "high_anxiety": "高焦虑", "tense": "高焦虑", "紧张": "高焦虑",
            "焦虑": "高焦虑",
            "low_anxiety": "低焦虑", "mild_anxiety": "低焦虑",
            "relaxed": "平静", "calm": "平静", "neutral": "平静",
            "happy": "愉悦", "joy": "愉悦", "positive": "愉悦",
            "tired": "疲劳", "fatigue": "疲劳", "exhausted": "疲劳",
        }
        if t in ("平静", "低焦虑", "高焦虑", "愉悦", "疲劳"):
            return t
        return alias.get(t.lower(), t)

    def _safe(self, provider, default):
        if not provider:
            return default
        try:
            return provider()
        except Exception:  # noqa: BLE001 工具失败不应中断对话
            return default

    # ------------------------------------------------------------------
    def run(self, params: Dict) -> str:
        emotion = self._norm_emotion(
            str(params.get("emotion") or params.get("input") or ""))
        arousal = str(params.get("arousal") or "").strip().lower()
        if arousal not in ("high", "low", "any"):
            arousal = ""
        try:
            intensity = float(params.get("intensity") or 0.0)
        except (TypeError, ValueError):
            intensity = 0.0
        try:
            top_k = int(params.get("top_k") or 3)
        except (TypeError, ValueError):
            top_k = 3

        stats = self._safe(self._stats_provider, {}) or {}
        recent = self._safe(self._recent_provider, []) or []

        # 允许直接指定技能（用户或模型明确要求「再教我一次 XXX」）
        want = str(params.get("skill") or "").strip()
        picked = None
        if want:
            s = get_skill(want)
            if s:
                picked = [{**s, "_score": None, "_why": "调用方指定"}]
        if picked is None:
            picked = match_skills(emotion, arousal, intensity, stats,
                                  recent, top_k)
        # 记录本次匹配结果，供 agent 侧做"最近用过的技能"去重
        self.last_ids = [s["id"] for s in picked]

        emo_label = emotion or "未指定"
        eff_arousal = arousal or _EMOTION_AROUSAL.get(emotion, "any")
        arousal_label = {"high": "高唤醒（需要降温）",
                         "low": "低唤醒（需要温和激活）",
                         "any": "唤醒度中性"}.get(eff_arousal, "唤醒度未知")
        lines = ["【技能匹配结果】情绪=%s | %s | 强度=%.2f"
                 % (emo_label, arousal_label, intensity)]
        # 情绪未知 / 标签不在 5 类内 → 先给通用前置框架
        unknown = (not emotion) or (emotion not in _EMOTION_AROUSAL)
        if unknown:
            lines.append("（未识别的情绪标签，已按通用技能匹配）")
            lines.append("")
            lines.append("【通用前置框架 —— 先做这两步，再考虑任何技术】")
            for i, (name, script) in enumerate(GENERIC_FRAMING, 1):
                lines.append("  %d. 【%s】%s" % (i, name, script))
            lines.append("  说明: 情绪未识别（或识别不可信）时，"
                         "**不要**假装知道ta是什么情绪 —— "
                         "先用开放问题确认，而不是直接给技术。")

        for i, s in enumerate(picked, 1):
            lines.append("")
            lines.append("%d. 【%s】（%s）— 建议时长 %s"
                         % (i, s["name"], s["school"], s["duration"]))
            if s.get("_why"):
                lines.append("   匹配理由: %s" % s["_why"])
            lines.append("   起效机制: %s" % s["mechanism"])
            lines.append("   引导脚本: %s" % s["script"])
            lines.append("   ⚠️ 注意事项: %s" % s["caution"])

        lines.append("")
        lines.append("【使用要求】")
        lines.append("  1. 只挑**一个**技能带用户做，不要说「我给你三个选择」"
                     "—— 情绪负荷高时，做选择本身就是负担;")
        lines.append("  2. 脚本是**参考话术**不是台词，按用户的语言习惯改写，"
                     "语气要慢、要有停顿;")
        lines.append("  3. **必须遵守注意事项**：若技术禁忌命中了用户的情况，"
                     "直接换下一个，并就「为什么换」给一句人话解释;")
        lines.append("  4. 做完要问一句感受（「现在身体和刚才比，有什么"
                     "不一样吗？」），这既是陪伴也是有效性来源;")
        lines.append("  5. 不要承诺疗效（「做完就好了」）—— "
                     "这些技术是**缓解**不是**治疗**;")
        lines.append("  6. 全程不要提「你符合某某障碍的某某特征」"
                     "这类表述 —— 你只提供技术，不做诊断。")
        return "\n".join(lines)
