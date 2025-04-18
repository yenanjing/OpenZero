import re
import random
from typing import Dict, Tuple, Optional


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
        print("[Error] No valid answer or think tags found")
        return None, processed_str

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


def compute_score(solution_str, ground_truth, method='strict', format_reward=1, answer_reward=1.):
    """The scoring function for open_industry task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer

    Returns:
        Total score (sum of format and answer rewards)
    """

    gt_industry = eval(ground_truth)
    # do_print = 1
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("\n" + "=" * 80)
        print(" Processing New Sample ".center(80, '='))

        print(f"[Ground Truth] Final industries: {gt_industry}")

    answer_text, processed_str = extract_solution(solution_str)

    if do_print:
        print(f"\n[Model Response]\n{processed_str}")

    if "<answer>" in processed_str:
        format_correct = validate_response_structure(processed_str, do_print)
    else:
        format_correct = validate_response_structure_ds(processed_str, do_print)

    format_score = format_reward if format_correct else -abs(format_reward)

    if do_print:
        print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        print(f"  Format score: {format_score}")

    if format_correct and answer_text:
        pred_industry = parse_model_answer(answer_text, do_print)
        if pred_industry:
            if do_print:
                print(f"\n[Content Validation]")
                print(f"  Expected: {gt_industry}")
                print(f"  Predicted: {pred_industry}")

            if pred_industry == gt_industry:
                answer_score = 2
                if do_print:
                    print("  Content validation: FULL MATCH")
            elif pred_industry.get('first_industry', '') == gt_industry['first_industry']:
                answer_score = 1.5
                if do_print:
                    print("  Content validation: first industry MATCH, second industry MISMATCH")
            elif pred_industry.get('second_industry', '') == gt_industry['second_industry']:
                answer_score = -0.5
                if do_print:
                    print("  Content validation: second industry MATCH, first industry MISMATCH")
            else:
                answer_score = -1.5
                if do_print:
                    print("  Content validation: FULL MISMATCH")
        else:
            answer_score = -2
            if do_print:
                print("Fail to parse answer")
    else:
        answer_score = -2
        if do_print:
            print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    if do_print:
        print("\n" + "-" * 80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("=" * 80 + "\n")

    return total_score
