import re
import random
from typing import Dict, Tuple, Optional

import requests



def parse_model_answer(answer_text: str, do_print: bool) -> Optional[Dict[str, str]]:
    """Parses model's answer text into industry dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        do_print: Boolean indicating whether to print debug information

    Returns:
        Dictionary mapping industry to predicted class, or None if incomplete
    """
    industries_dict = {}

    if do_print:
        print("\n[Model Answer Parsing]")

    try:
        pattern = r'一级行业：(.+?)；'
        match = re.search(pattern, answer_text)
        ind = match.group(1) if match.group(1) else match.group(2)
        industries_dict["first_industry"] = ind.strip()
    except:
        if do_print:
            print(f"  [Error] Missing identification for first industry")

    try:
        pattern = r'；二级行业：(.+?)$'
        match = re.search(pattern, answer_text)
        ind = match.group(1) if match.group(1) else match.group(2)
        industries_dict["second_industry"] = ind.strip()
    except:
        if do_print:
            print(f"  [Error] Missing identification for second industry")

    return industries_dict


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    # if "Assistant:" in solution_str:
    #     processed_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     print("[Error] Failed to locate model response header")
    #     return None, solution_str
    processed_str = solution_str
    # Extract final answer using XML-style tags
    if "<answer>" in solution_str:
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

        if not matches:
            print("[Error] No valid answer tags found")
            return None, processed_str
        final_answer = matches[-1].group(1).strip()
    elif "</think>" in solution_str:
        final_answer = processed_str.split("/think")[1].strip()
    else:
        # print("No answer or think tags found")
        return processed_str, processed_str

    return final_answer, processed_str


def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model
        do_print: Boolean indicating whether to print debug information

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    elif do_print:
        print("  Tag sequence validation passed")

    return validation_passed


def validate_response_structure_ans(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model
        do_print: Boolean indicating whether to print debug information

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_begin': ('</think>', 0),
        'think_end': ('</think>', 0),
        'answer_start': ('<answer>', 0),
        'answer_end': ('</answer>', 0)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    return validation_passed


def validate_response_structure_ds(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model
        do_print: Boolean indicating whether to print debug information

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    elif do_print:
        print("  Tag sequence validation passed")

    return validation_passed


def rouge_l(reference, candidate, beta=1.0, mode='char'):
    """
    计算ROUGE-L分数（基于最长公共子序列LCS）。

    参数:
        reference (str/list): 参考文本（字符串或单词列表）
        candidate (str/list): 候选文本（字符串或单词列表）
        beta (float): F-score的权重参数（默认beta=1表示F1-score）
        mode (str): 计算模式，'word'（词级别）或'char'（字符级别）

    返回:
        dict: 包含Recall、Precision、F1-score的字典
    """
    # 预处理：根据模式将文本拆分为词或字符
    if mode == 'word':
        if isinstance(reference, str):
            ref_tokens = reference.split()
            can_tokens = candidate.split()
        else:
            ref_tokens = reference
            can_tokens = candidate
    elif mode == 'char':
        ref_tokens = list(reference)
        can_tokens = list(candidate)
    else:
        raise ValueError("mode必须是'word'或'char'")

    # 计算LCS长度
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs_len = lcs_length(ref_tokens, can_tokens)
    if lcs_len == 0:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}

    # 计算Recall、Precision、F1
    recall = lcs_len / len(ref_tokens)
    precision = lcs_len / len(can_tokens)
    f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)  # 避免除零

    return {'recall': recall, 'precision': precision, 'f1': f1}


def extract_links(text):
    """从文本中提取所有超链接"""
    if len(text.strip()) == 0:
        return []

    # markdown_links = re.findall(r'(https?://[^\s)]+)', str(text))
    markdown_links = re.findall(r'[($$](https?://[^\s)$$]+)[)\]]', str(text))

    return markdown_links


def zipngram(text: str, ngram_size: int):
    # words = text.lower().split()
    words = list(text)
    return zip(*[words[i:] for i in range(ngram_size)])


def get_similarity(ground_truth, answer_text):
    url = "http://localhost:8000/similarity"
    payload = {
        "ground_truth": ground_truth,
        "answer_text": answer_text
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()['similarity_score']
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

def compute_score(solution_str, ground_truth, extra_info):
    """The scoring function for open_industry task.

    Args:
        solution_str: the solution text
        ground_truth: the checked text
        extra_info

    Returns:
        Total score (sum of format and answer rewards)
    """

    # do_print = 1
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("\n" + "=" * 80)
        print(" Processing New Sample ".center(80, '='))

        print(f"[Ground Truth]: {ground_truth}")

    answer_text, processed_str = extract_solution(solution_str)

    if do_print:
        print(f"\n[Model Response]\n{processed_str}")

    format_correct = validate_response_structure_ans(processed_str, do_print)

    format_score = 1 if format_correct else -abs(1)

    if do_print:
        print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        print(f"  Format score: {format_score}")

    rouge_score = 0
    semantic_score = 0
    link_score = 0
    rep_score = 0
    if format_correct and answer_text:
        rouge_l_similarity = rouge_l(ground_truth, answer_text)
        rouge_score = rouge_l_similarity['f1']

        semantic_score = get_similarity(ground_truth, answer_text)

        pattern = re.compile(
            r'(?:(?:https?://)|(?:www\.))'  # 匹配 http://, https:// 或 www.
            r'[^\s<>"\'()\u4e00-\u9fa5]+'  # 匹配 URL 主体（排除空格、<>"'()和中文字符）
            r'(?=[\s"\'()\u4e00-\u9fa5]|$)',  # 确保 URL 后跟有效终止符
            re.IGNORECASE
        )

        # pattern = r'[($$](https?://[^\s)$$]+)[)\]]'
        content_all_links = pattern.findall(answer_text)
        links = list(set(content_all_links))
        context = extra_info["context"]
        if do_print:
            print(f"\n[Context]\n{context}")
        context_links = list(set(pattern.findall(context)))
        ground_truth_links = list(set(pattern.findall(ground_truth)))

        if not context_links:
            if do_print:
                print("no context links")
            link_score = max(-1.0, -0.5 * len(content_all_links)) if links else 1.0
        else:
            # # 漏召回
            # miss_recall_links = [link for link in ground_truth_links if link not in links]
            # miss_recall_penalty = 0.2 * len(miss_recall_links)

            # 计算错误链接的惩罚
            false_recall_links = [link for link in links if link not in context_links]
            false_penalty = 0.5 * len(false_recall_links)  # 每个错误链接扣0.5分

            penalty = false_penalty
            # 重复图片或图片数量过多的惩罚
            if len(content_all_links) > len(context_links) or len(links) < len(content_all_links):
                penalty += 0.2

            link_score = max(-1.0, -penalty)

            # reward 大于 0 时，计算正向reward
            if link_score > -1e-5:
                recall_in_ground_truth = sum(1.0 for link in links if link in ground_truth_links) / len(
                    ground_truth_links) if len(ground_truth_links) > 0 else 0.0
                recall_in_context = sum(
                    1.0 for link in links if link in context_links and link not in ground_truth_links) / len(
                    context_links) if len(context_links) > 0 else 0.0
                if recall_in_context < 1e-5:
                    link_score = recall_in_ground_truth
                else:
                    link_score = (recall_in_ground_truth * 5.0 + recall_in_context) / 6.0

        ngram_size = 3
        max_penalty = -1.0
        if answer_text == '':
            rep_score = 0
        elif len(list(answer_text)) < ngram_size:
            rep_score = 0
        else:
            ngrams = set()
            total = 0
            for ng in zipngram(answer_text, ngram_size):
                ngrams.add(ng)
                total += 1
            scaling = 1 - len(ngrams) / total
            rep_score = scaling * max_penalty

    else:
        if do_print:
            print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + rouge_score + semantic_score + link_score + rep_score
    if do_print:
        print("\n" + "-" * 80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Rouge_L: {rouge_score}")
        print(f"  Semantic: {semantic_score}")
        print(f"  Link: {link_score}")
        print(f"  Repetition: {rep_score}")
        print(f"  Total: {total_score}")
        print("=" * 80 + "\n")

    return {"score": total_score, "Format": format_score, "Rouge_L": rouge_score, "Semantic": semantic_score,
            "Link": link_score, "Repetition": rep_score, }
