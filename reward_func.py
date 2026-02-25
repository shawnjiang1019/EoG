import json
import re
from typing import List, Dict, Any, Union
import numpy as np
import ast
def compute_rog_cwq_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    
    
    if data_source != "rog_cwq":
        return 0.0
    
    try:
        gt_data = json.loads(ground_truth)
        correct_answers = gt_data["correct_answers"]
        
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]
        elif not isinstance(correct_answers, list):
            correct_answers = []
        
        answer_entities = gt_data.get("answer_entities", [])
        question_entities = gt_data.get("question_entities", [])
        graph_info = gt_data.get("graph_info", [])
        
        parsed_output = parse_model_output(solution_str)
        
        hits_at_1_score = evaluate_hits_at_1(
            parsed_output["final_answer"], 
            correct_answers
        )
        
        f1_score = evaluate_f1_score(
            parsed_output["final_answer"], 
            correct_answers
        )
        
        reasoning_score = evaluate_reasoning_quality(
            parsed_output["reasoning"], 
            graph_info,
            question_entities,
            answer_entities,
            gt_data['reasoning_path']
        )
        
        total_score = (
            hits_at_1_score * 0.0 + 
            f1_score * 0.0 + 
            reasoning_score * 1.0
        )
        
        return total_score
        
    except Exception as e:
        print(f"Scoring error: {e}")
        return 0.0

def parse_model_output(solution_str: str) -> Dict[str, str]:
    """Parse model output, extract reasoning process and final answer"""
    

    
    m = re.search(r"^(.*?)<answer>", solution_str, re.DOTALL | re.IGNORECASE)
    reasoning = m.group(1).strip() if m else ""
    
    
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    final_answer = answer_match.group(1).strip() if answer_match else ""
    final_answer = ast.literal_eval(final_answer)
    
    return {
        "reasoning": reasoning,
        "final_answer": final_answer
    }

def extract_answers_from_text(answer_input) -> List[str]:
    """
    Extract answer list from answer input, supports multiple formats and types
    
    Supported input formats:
    1. List: ["answer1", "answer2"]
    2. JSON string: '["answer1", "answer2"]'
    3. Tagged string: '<answer>["answer1", "answer2"]</answer>'
    4. Plain string: "single answer"
    """
    if not answer_input:
        return []
    
    if isinstance(answer_input, list):
        return [str(ans).strip() for ans in answer_input if str(ans).strip()]
    
    answer_text = str(answer_input).strip()
    if not answer_text:
        return []
    
    clean_text = answer_text.replace('<answer>', '').replace('</answer>', '').strip()
    
    try:
        if clean_text.startswith('[') and clean_text.endswith(']'):
            parsed_answers = json.loads(clean_text)
            if isinstance(parsed_answers, list):
                return [str(ans).strip() for ans in parsed_answers if str(ans).strip()]
    except json.JSONDecodeError:
        if clean_text.startswith('[') and clean_text.endswith(']'):
            content = clean_text[1:-1].strip()
            if content:
                answers = []
                current_answer = ""
                in_quotes = False
                for char in content:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        if current_answer.strip():
                            answers.append(current_answer.strip().strip('"'))
                        current_answer = ""
                    else:
                        current_answer += char
                
                if current_answer.strip():
                    answers.append(current_answer.strip().strip('"'))
                
                if answers:
                    return [ans for ans in answers if ans]
    if "," in clean_text:
        answers = [ans.strip() for ans in clean_text.split(",") if ans.strip()]
    return [clean_text] if clean_text else []

def evaluate_hits_at_1(predicted_answer, correct_answers: List[str]) -> float:
    """Evaluate Hits@1 score - exact match or containment relationship"""
    if not predicted_answer or not correct_answers:
        return 0.0
    
    predicted_answers = extract_answers_from_text(predicted_answer)
    
    if not predicted_answers:
        return 0.0
    
    for pred_ans in predicted_answers:
        pred_ans_lower = pred_ans.lower().strip()
        for correct_answer in correct_answers:
            correct_lower = correct_answer.lower().strip()
            
            if (pred_ans_lower == correct_lower or 
                correct_lower in pred_ans_lower or 
                pred_ans_lower in correct_lower):
                return 1.0
    
    return 0.0

def evaluate_f1_score(predicted_answer, correct_answers: List[str]) -> float:
    """Evaluate F1 score based on answer items"""
    if not predicted_answer or not correct_answers:
        return 0.0
    
    predicted_answers = extract_answers_from_text(predicted_answer)
    
    if not predicted_answers:
        return 0.0
    
    pred_set = set(ans.lower().strip() for ans in predicted_answers)
    gt_set = set(ans.lower().strip() for ans in correct_answers)
    
    intersection = pred_set & gt_set
    
    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0
    elif len(pred_set) == 0 or len(gt_set) == 0:
        return 0.0
    
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)
    
    if precision + recall == 0:
        return 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        return f1

def evaluate_reasoning_quality(reasoning: str, graph_info: List, 
                              question_entities: List[str], 
                              answer_entities: List[str],
                              reasoning_path: List = None) -> float:
    """Evaluate reasoning path quality"""
    if not reasoning:
        return 0.0
    
    score = 0.0
    path_match_score = evaluate_reasoning_path_match(reasoning, reasoning_path or [], graph_info)
    score += path_match_score
    
    return min(score, 1.0)

def evaluate_reasoning_path_match(reasoning: str, reasoning_path: List, graph_info: List) -> float:
    """Evaluate the match between reasoning path and standard path"""
    if not reasoning or not reasoning_path:
        return 0.0
    
    # 1. Extract triplets [subject, relation, object] from reasoning text
    extracted_triplets = extract_triplets_from_reasoning(reasoning)
    if not extracted_triplets:
        return 0.0
    print(extracted_triplets)
    # 2. Flatten reasoning_path from list of list of list to list of list
    flattened_path = flatten_reasoning_path(reasoning_path)
    if not flattened_path:
        return 0.0
    
    # 3. Calculate match score
    matched_count = 0
    for extracted_triplet in extracted_triplets:
        subj, rel, obj = extracted_triplet
        for path_triplet in flattened_path:
            if len(path_triplet) >= 3:
                path_subj, path_rel, path_obj = path_triplet[0], path_triplet[1], path_triplet[2]
                # Convert to string and compare in lowercase
                if (subj.lower() == path_subj.lower() and
                    rel.lower() == path_rel.lower() and
                    obj.lower() == path_obj.lower()):
                    matched_count += 1
                    break  # Each extracted triplet matches only once
    
    return matched_count / len(flattened_path) if extracted_triplets else 0.0


def extract_triplets_from_reasoning(reasoning: str) -> List[List[str]]:
    """Extract triplets [subject, relation, object] from reasoning text using rule matching"""
    triplets = []
    reasoning_lower = reasoning.lower()
    
    # Pattern 1: (subject, relation, object) or [subject, relation, object]
    pattern1 = r'[\[\(]([^,\[\(\)\]]+),\s*([^,\[\(\)\]]+),\s*([^,\[\(\)\]]+)[\]\)]'
    matches1 = re.finditer(pattern1, reasoning, re.IGNORECASE)
    for match in matches1:
        subj = match.group(1).strip()
        rel = match.group(2).strip()
        obj = match.group(3).strip()
        
        # Clean strings: remove quotes, escape characters, etc.
        subj = subj.strip('"\'`').strip()
        rel = rel.strip('"\'`').strip()
        obj = obj.strip('"\'`').strip()
        
        # If string looks like a list/tuple representation, try to parse
        for field in [subj, rel, obj]:
            if field.startswith('[') or field.startswith('('):
                try:
                    parsed = ast.literal_eval(field)
                    if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                        # If parsed result is a list/tuple, take the first element
                        field = str(parsed[0]) if parsed else field
                except (ValueError, SyntaxError):
                    pass
        
        if subj and rel and obj:
            triplets.append([subj, rel, obj])
    
    # Remove duplicates
    seen = set()
    unique_triplets = []
    for triplet in triplets:
        triplet_key = tuple([str(t).lower().strip() for t in triplet])
        if triplet_key not in seen:
            seen.add(triplet_key)
            unique_triplets.append(triplet)
    return unique_triplets


def flatten_reasoning_path(reasoning_path: List) -> List[List]:
    """Flatten nested list of list of list to list of list (each sublist is a triplet)"""
    flattened = []
    stack = [reasoning_path]
    
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            # Check if it's a triplet format [subject, relation, object]
            if len(node) >= 3 and all(not isinstance(node[i], (list, tuple)) for i in range(3)):
                subj, rel, obj = node[0], node[1], node[2]
                if isinstance(subj, str) and isinstance(rel, str) and isinstance(obj, str):
                    flattened.append([subj, rel, obj])
                else:
                    # Even if not strings, add as triplet
                    flattened.append([str(subj), str(rel), str(obj)])
            else:
                # Continue expanding nested lists
                stack.extend(reversed(node))  # reversed maintains order
    return flattened

def evaluate_answer_correctness(predicted_answer, correct_answers: List[str]) -> float:
    """Function retained for compatibility, actually uses Hits@1 evaluation"""
    return evaluate_hits_at_1(predicted_answer, correct_answers)