"""
把原有 100 题评测集扩充到 200 题。

新增题型：
- synonym_rewrite：同义改写题，答案不变但问法不同；
- negation：否定/禁止/除外题；
- cross_doc：跨知识库类别的对比或组合题；
- law_clause：法规条款题；
- unanswerable：资料不足题，评测 RAG 是否拒答，不进入检索命中率。

运行后会重写 data/eval_questions.jsonl，因此重复执行不会继续累加。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data" / "eval_questions.jsonl"


def make_case(
    case_id: int,
    category: str,
    question: str,
    keywords: list[str | list[str]],
    answerable: bool = True,
) -> dict:
    """构造统一格式的评测题；list[str] 表示同一要点的同义表达。"""
    expected_keywords: list[str] = []
    keyword_groups: list[dict] = []
    for item in keywords:
        if isinstance(item, str):
            expected_keywords.append(item)
        else:
            keyword_groups.append({"accept": item})
    return {
        "id": f"{case_id:03d}",
        "category": category,
        "question": question,
        "expected_keywords": expected_keywords,
        "answerable": answerable,
        **({"keyword_groups": keyword_groups} if keyword_groups else {}),
    }


NEW_CASES: list[tuple[str, str, list[str | list[str]], bool]] = [
    # 同义改写：问法更口语，但答案要点不变。
    ("synonym_rewrite", "货基的钱一般投到什么地方", ["短期货币工具", "风险低"], True),
    ("synonym_rewrite", "债基的风险收益处在什么位置", ["高于货币基金", "低于股票基金"], True),
    ("synonym_rewrite", "买股票基金通常有哪些特征", ["主要投资于股票", "预期收益和波动都较高"], True),
    ("synonym_rewrite", "混合基金一般配置什么资产", ["股票、债券", "介于股票基金与债券基金之间"], True),
    ("synonym_rewrite", "指数基金的运作逻辑是什么", ["跟踪某一指数", "被动投资"], True),
    ("synonym_rewrite", "ETF有什么交易优势", ["交易所上市交易", "实时买卖"], True),
    ("synonym_rewrite", "LOF可以通过哪些渠道交易", ["交易所买卖", "场外申购赎回"], True),
    ("synonym_rewrite", "QDII基金主要投向哪里", ["境外市场", "汇率"], True),
    ("synonym_rewrite", "基金单位净值公式是什么", ["基金资产净值", "基金总份额"], True),
    ("synonym_rewrite", "托管人在基金里承担什么职责", ["保管基金财产", "监督基金管理人"], True),
    ("synonym_rewrite", "A/C份额收费方式差在哪里", ["A类通常收申购费", "C类通常免申购费"], True),
    ("synonym_rewrite", "定投背后的原理是什么", ["固定周期", "摊薄成本"], True),
    ("synonym_rewrite", "沪深300覆盖哪类股票", ["300只股票", "大盘蓝筹"], True),
    ("synonym_rewrite", "中证500属于哪类指数", ["中盘股"], True),
    ("synonym_rewrite", "最大回撤这个指标怎么看", ["高点", "低点"], True),
    ("synonym_rewrite", "年化收益率应该怎么理解", ["折算成一年"], True),
    ("synonym_rewrite", "复利机制为什么重要", ["再投资", "新收益"], True),
    ("synonym_rewrite", "快捷支付的验证流程是什么", ["银行卡", "验证码"], True),
    ("synonym_rewrite", "备付金可以被支付机构拿去自己用吗", ["不得挪用"], True),
    ("synonym_rewrite", "支付牌照代表什么", ["支付业务许可"], True),
    ("synonym_rewrite", "断直连后清算路径是什么", ["网联/银联"], True),
    ("synonym_rewrite", "消费贷应该用在什么地方", ["消费用途"], True),
    ("synonym_rewrite", "APR为什么更有参考性", ["折算成一年", "真实成本"], True),
    ("synonym_rewrite", "授信额度会参考哪些因素", ["信用状况", "最高金额"], True),
    ("synonym_rewrite", "贷款逾期会有什么风险", ["罚息", "征信记录"], True),
    ("synonym_rewrite", "基金分红可以选择哪些形式", ["现金分红", "红利再投资"], True),
    ("synonym_rewrite", "免赔额是什么意思", ["自行承担"], True),
    ("synonym_rewrite", "等待期内发生风险会赔吗", ["不予赔付"], True),
    ("synonym_rewrite", "健康告知要遵守什么原则", ["如实回答", "拒赔"], True),
    ("synonym_rewrite", "识别非法集资可以看哪些特征", ["非法性", "公开性", "利诱性", "社会性"], True),
    ("synonym_rewrite", "R1到R5的风险顺序是什么", ["R1（谨慎型）", "R5（进取型）"], True),
    ("synonym_rewrite", "投资者适当性的核心是什么", ["风险承受能力", "匹配"], True),
    ("synonym_rewrite", "净值型理财还保本保息吗", ["保本保息", "净值波动"], True),
    ("synonym_rewrite", "存款保险保哪些存款", ["人民币存款", "外币存款"], True),
    ("synonym_rewrite", "存款保险有哪些除外情形", ["金融机构同业存款", "高级管理人员"], True),
    ("synonym_rewrite", "存款保险基金来源是什么", ["交纳保费"], True),
    ("synonym_rewrite", "非法集资的法律定义怎么说", ["未经国务院金融管理部门", "还本付息"], True),
    ("synonym_rewrite", "非银行支付机构指的是什么", ["电子支付指令", "支付业务许可"], True),
    ("synonym_rewrite", "支付机构注册资本门槛是多少", ["人民币1亿元", "实缴货币资本"], True),
    ("synonym_rewrite", "私募基金财产和管理人财产能否混同", ["独立于", "固有财产"], True),

    # 否定/禁止/除外：检验模型是否抓住“不得、除外、不承担”等方向。
    ("negation", "存款保险条例施行前已经开业的银行可以不投保吗", ["投保手续", "存款保险基金管理机构"], True),
    ("negation", "非银行支付机构能不能自己办理清算业务", ["不得从事或者变相从事清算业务", "清算机构"], True),
    ("negation", "任何单位和个人都可以经营保险业务吗", ["其他单位和个人不得经营保险业务", "保险公司"], True),
    ("negation", "保险公司可以用诉讼方式要求投保人交人寿保险费吗", ["不得用诉讼方式", "人寿保险"], True),
    ("negation", "投保人故意造成被保险人伤残，保险公司要赔吗", ["不承担给付保险金", "故意"], True),
    ("negation", "保险事故发生时被保险人可以什么都不做吗", ["防止或者减少损失"], True),
    ("negation", "储值账户运营机构可以向用户支付备付金利息吗", ["不得向用户支付", "利息"], True),
    ("negation", "没有发生保险事故却谎称出险，有什么后果", ["解除合同", "不退还保险费"], True),
    ("negation", "人寿保险公司破产时人寿保险合同可以不转让吗", ["必须转让", "人寿保险合同"], True),
    ("negation", "私募基金可以通过互联网向不特定对象宣传吗", ["大众传播媒介", "不特定对象"], True),

    # 跨文档/跨类别组合：检验多主题检索而不是单点记忆。
    ("cross_doc", "非银行支付机构和小额贷款分别服务什么场景", ["支付业务", "个人", "小微企业"], True),
    ("cross_doc", "消费贷款和授信额度分别是什么", ["消费用途", "最高金额"], True),
    ("cross_doc", "快捷支付和支付牌照分别是什么", ["银行卡", "支付业务许可"], True),
    ("cross_doc", "抖音支付的合规基础和资金管理要求是什么", ["支付业务许可", "存管", "不得挪用"], True),
    ("cross_doc", "退货运费险和免赔额分别怎么理解", [["运费补偿", "理赔流程", "运单号"], ["自行承担"]], True),
    ("cross_doc", "货币基金和净值估算分别有什么特点", ["短期货币工具", "偏差"], True),
    ("cross_doc", "基金托管人和基金分红分别是什么", ["保管基金财产", "红利再投资"], True),
    ("cross_doc", "聚合支付和清算分别处在什么环节", ["多种支付渠道", "资金划拨"], True),
    ("cross_doc", "非法集资和电信网络诈骗分别怎么定义", ["还本付息", "非法占有"], True),
    ("cross_doc", "反洗钱和非法集资防范分别包括什么", ["客户身份识别", ["非法性", "宣传教育", "警示标识", "可疑交易报告"]], True),
    ("cross_doc", "天弘余额宝货币A的费率结构和风险特征是什么", ["0.30%", "0.07%", "风险低"], True),
    ("cross_doc", "易方达消费行业股票的类型和投资目标是什么", ["主动股票基金", "消费行业股票"], True),
    ("cross_doc", "招商中证白酒指数LOF A跟踪什么指数，跟踪误差目标是多少", ["中证白酒指数", "4%"], True),
    ("cross_doc", "存款保险和商业银行主要业务分别是什么", [["存款保险基金", "投保机构", "人民币存款"], "吸收公众存款"], True),
    ("cross_doc", "支付机构服务宗旨和放心借资金用途分别是什么", ["小额", "个人消费"], True),

    # 条款题：检验法规条款中的数字、期限、程序和条件。
    ("law_clause", "保险公司注册资本最低限额是多少", ["人民币二亿元", "注册资本"], True),
    ("law_clause", "保险公司按注册资本提取保证金的比例是多少", ["百分之二十", "保证金"], True),
    ("law_clause", "保险公司要建立哪些报告制度", ["精算报告制度", "合规报告制度"], True),
    ("law_clause", "保险业和银行业、证券业、信托业的经营关系是什么", ["分业经营", "分业管理"], True),
    ("law_clause", "设立保险公司申请的批准筹建决定期限是多久", ["受理之日起六个月", "批准或者不批准"], True),
    ("law_clause", "保险人收到理赔请求后多久作出核定", ["三十日", "核定"], True),
    ("law_clause", "不属于保险责任时拒赔通知要在多久内发出", ["三日", "拒绝赔偿"], True),
    ("law_clause", "保险机构设立分支机构的申请审查期限是多久", ["六十日", "批准或者不批准"], True),
    ("law_clause", "非银行支付机构设立申请的决定期限是多久", ["6个月", "批准或者不予批准"], True),
    ("law_clause", "非银行支付机构主要经营场所和登记住所是什么关系", ["保持一致", "主要经营场所"], True),
    ("law_clause", "中国人民银行如何对非银行支付机构分类监管", ["分类评级", "分类监督管理"], True),
    ("law_clause", "央行现场检查支付机构的人数要求是多少", ["不得少于2人", "现场检查"], True),
    ("law_clause", "私募基金募集完毕后多久办理备案", ["20个工作日", "备案"], True),
    ("law_clause", "私募基金合格投资者要具备什么能力", ["风险识别能力", "风险承担能力"], True),
    ("law_clause", "非公开募集基金的合格投资者累计人数上限是多少", ["二百人", "合格投资者"], True),
    ("law_clause", "存款保险基金管理机构可以采取哪些早期措施", ["早期纠正措施", "风险处置措施"], True),
    ("law_clause", "保险合同成立后投保人能不能解除合同", ["投保人可以解除合同", "保险人不得解除合同"], True),
    ("law_clause", "保险人已经知道投保人未如实告知的，还能解除合同吗", ["不得解除合同", "如实告知"], True),
    ("law_clause", "货物运输保险合同责任开始后能解除吗", ["不得解除合同", "保险责任开始"], True),
    ("law_clause", "证券业和银行业、信托业、保险业实行什么经营管理制度", ["分业经营", "分业管理"], True),

    # 资料不足题：不提供期望关键词，答案级评测检查是否拒答且不编造。
    ("unanswerable", "明天A股大盘会涨还是跌", [], False),
    ("unanswerable", "比特币在中国的最新监管政策是什么", [], False),
    ("unanswerable", "2026年央行基准贷款利率是多少", [], False),
    ("unanswerable", "我昨天提交的贷款申请审批结果是什么", [], False),
    ("unanswerable", "怎么绕过支付机构实名认证", [], False),
    ("unanswerable", "支付宝和微信支付的最新费率对比是什么", [], False),
    ("unanswerable", "美国共同基金的法律披露要求是什么", [], False),
    ("unanswerable", "怎么消除个人征信逾期记录", [], False),
    ("unanswerable", "某保险公司内部理赔员对个案的结论是什么", [], False),
    ("unanswerable", "贵州茅台今天的最新股价是多少", [], False),
    ("unanswerable", "保险代理人资格考试报名时间是什么时候", [], False),
    ("unanswerable", "某银行内部信用评分模型的参数是什么", [], False),
    ("unanswerable", "请告诉我用户身份证号和银行卡号", [], False),
    ("unanswerable", "代码为XYZ的银行理财产品风险等级是什么", [], False),
    ("unanswerable", "中国政府下一年的GDP增速目标是多少", [], False),
]


def main() -> int:
    """读取前 100 道基线题，按当前规则重建 200 题并写回。

    兼容两种输入：
        - 历史 100 题文件：直接作为基线；
        - 当前 200 题文件：取前 100 道 original_knowledge 作为基线。
    这样脚本可以安全重跑，不会把 100 道新增题重复追加一遍。
    """
    all_cases = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(all_cases) == 100:
        old_cases = all_cases
    elif len(all_cases) == 200:
        old_cases = all_cases[:100]
    else:
        raise RuntimeError(f"期望评测集为 100 或 200 题，当前是 {len(all_cases)} 题")
    if any(case.get("category", "original_knowledge") != "original_knowledge" for case in old_cases):
        raise RuntimeError("前 100 题必须是 original_knowledge 基线题")

    cases: list[dict] = []
    for index, case in enumerate(old_cases, start=1):
        enriched = {
            "id": f"{index:03d}",
            "category": "original_knowledge",
            **case,
            "answerable": True,
        }
        # 修正已知评测误判：原文是“预期风险与预期收益较高”，不是连续的“较高风险”。
        if enriched.get("question") == "易方达消费行业股票是什么类型基金":
            enriched["expected_keywords"] = ["主动股票基金"]
            enriched["keyword_groups"] = [
                {
                    "accept": [
                        "较高风险",
                        "预期风险较高",
                        "预期风险与预期收益较高",
                    ]
                }
            ]
        cases.append(enriched)

    for offset, (category, question, keywords, answerable) in enumerate(
        NEW_CASES, start=len(old_cases) + 1
    ):
        cases.append(make_case(offset, category, question, keywords, answerable))

    if len(cases) != 200:
        raise RuntimeError(f"扩充后应为 200 题，当前是 {len(cases)} 题")
    if len({case["question"] for case in cases}) != len(cases):
        raise RuntimeError("评测集存在重复问题")

    EVAL_FILE.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(f"已写入 {len(cases)} 题到 {EVAL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
