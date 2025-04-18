import requests
import time

# API配置
API_URL = "http://localhost:8000/similarity"
TEST_CASES = [
    {
        "ground_truth": "今天天气真好",
        "answer_text": "今天的天气非常不错",
        "expected_score": 0.8  # 预期相似度大约值
    },
    {
        "ground_truth": "我喜欢吃苹果",
        "answer_text": "我爱吃香蕉",
        "expected_score": 0.5  # 预期相似度大约值
    },
    {
        "ground_truth": "深度学习是人工智能的一个分支",
        "answer_text": "机器学习属于AI领域",
        "expected_score": 0.7  # 预期相似度大约值
    },
    {
        "ground_truth": "北京是中国的首都",
        "answer_text": "中国的首都是北京",
        "expected_score": 0.9  # 预期相似度大约值
    },
    {
        "ground_truth": "这只猫很可爱",
        "answer_text": "这条狗很凶猛",
        "expected_score": 0.3  # 预期相似度大约值
    }
]


def test_single_case(test_case, show_details=True):
    """测试单个文本对"""
    payload = {
        "ground_truth": test_case["ground_truth"],
        "answer_text": test_case["answer_text"]
    }

    start_time = time.time()
    response = requests.post(API_URL, json=payload)
    elapsed_time = (time.time() - start_time) * 1000  # 毫秒

    if response.status_code != 200:
        print(f"❌ 请求失败 (状态码: {response.status_code}): {response.text}")
        return False

    result = response.json()
    similarity_score = result["similarity_score"]
    passed = abs(similarity_score - test_case["expected_score"]) < 0.2

    if show_details:
        print("\n" + "=" * 50)
        print(f"📝 原文: {test_case['ground_truth']}")
        print(f"📝 对比文本: {test_case['answer_text']}")
        print(f"✅ 相似度得分: {similarity_score:.4f}")
        print(f"⏱ 响应时间: {elapsed_time:.2f}ms")
        print(f"🔮 预期得分: ~{test_case['expected_score']}")
        print(f"🎯 结果: {'通过' if passed else '失败'}")

    return passed


def run_full_test():
    """运行完整测试套件"""
    print("\n🚀 开始测试BERT相似度API服务")
    print(f"🔗 API端点: {API_URL}")
    print(f"📊 测试用例数量: {len(TEST_CASES)}\n")

    passed_count = 0
    total_time = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n🔍 正在测试用例 {i}/{len(TEST_CASES)}...")
        start_time = time.time()
        success = test_single_case(test_case)
        elapsed_time = (time.time() - start_time) * 1000
        total_time += elapsed_time

        if success:
            passed_count += 1

    print("\n" + "=" * 50)
    print("📊 测试结果摘要")
    print(f"✅ 通过: {passed_count}/{len(TEST_CASES)}")
    print(f"❌ 失败: {len(TEST_CASES) - passed_count}/{len(TEST_CASES)}")
    print(f"⏱ 平均响应时间: {total_time / len(TEST_CASES):.2f}ms")
    print("=" * 50 + "\n")

    if passed_count == len(TEST_CASES):
        print("🎉 所有测试用例通过！API服务运行正常。")
    else:
        print("⚠️ 部分测试用例未通过，请检查API服务或测试用例。")


if __name__ == "__main__":
    run_full_test()