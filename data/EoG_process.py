

import argparse
import os
import json
import datasets
from datasets import Dataset

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on knowledge graphs. 
1. First, reason through the problem inside <think> and </think> tags. Here you can planning, memory, check for mistakes to reflect，prune the entities and relations.
2. When confident, output the final answer inside <answer> and </answer> tags. Your answer must strictly follow the rules provided by the user.

You will receive:
- question: A natural language question that needs to be answered
- Knowledge Graph: Knowledge graph information in the form of triples (subject, relation, object)
- Starting Entity: The starting entity in the knowledge graph that can be used as a reference point

You must generate a response that includes:

1. **Reasoning Chain** (<think> section):
   - Step-by-step logical reasoning process
   - Analysis of the question and relevant graph information
   - Identification of key entities and relationships, starting from the begin_entity
   - Show how you traverse from the begin_entity to reach the answer
   - Use of graph entities and relations in your reasoning

2. **Final Answer** (<answer> section):
   - Clear, concise answer to the question
   - Must match the provided correct answer
   - The answer MUST be one or more entities that exist in the provided graph
   - CRITICAL: Only use entities that are explicitly present in the graph triples

- Always start with the <think> section to show your reasoning
- Use the begin_entity as your starting point for reasoning
- Be explicit about which entities and relations you're considering
- Show logical connections between different pieces of information
- Ensure your answer only contains entities that exist in the provided graph triples
- Use clear, natural language that demonstrates good reasoning skills"""

USER_PROMPT = """Answer the given question based on the knowledge graph. You must first conduct reasoning step by step, and put your final answer inside <answer> and </answer>.

Rules:
1. You must always remember that the answer can only be selected from the entities from the graph.During reasoning,you must consistently keep this in mind.Only when the graph information is insufficient to answer the question,you can response or reasoing based on your own knowledge.
2. When you have the final answer, you can directly provide the answer inside <answer></answer>, without detailed illustrations. For example, <answer> ["Washington, D.C."] </answer>
3. If the knowledge graph information is insufficient to answer the question, you may use your own knowledge to supplement the reasoning process. 
4. If there are multiple possible answers, present them in list format using square brackets []. For example, <answer>["Beijing", "Washington DC"]</answer>.
5. Because questions usually have multiple answers, you should consider all possible answers and provide them in the list format.
6. If multiple starting entities are provided, you should analyze each one systematically，such as Consider how different starting points might lead to different answers ,combine insights from all starting entities to form answers and if starting entities are related, explore their connections in the knowledge graph.
You must use this format:
<think>...</think>
<answer>...</answer>

Question: {question}

Knowledge Graph:
{graph_text}

Starting Entity: {q_entity}

"""
def format_starting_entities(q_entity):
    if not q_entity:
        return "Unknown"
    elif len(q_entity) == 1:
        return q_entity[0]
    else:

        entities_list = []
        for i, entity in enumerate(q_entity, 1):
            entities_list.append(f"{i}. {entity}")
        return "\n".join(entities_list)
def build_prompt(question, graph_text, q_entity):
    q_entity_format = format_starting_entities(q_entity)
    user_content = USER_PROMPT.format(
        question=question,
        graph_text=graph_text,
        q_entity=q_entity_format
    )
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content+"\n"},
    ]

def process_rog_cwq_sample(example, idx):
    

    question = example['question']
    answer = example['answer']
    q_entity = example['q_entity']
    a_entity = example['a_entity']
    graph = example['graph']
    reasoning_path = example.get('reasoning_path', [])
    

    graph_text = str(graph) if graph else "No graph infomation is available"
    

    prompt = build_prompt(question, graph_text, q_entity)
    

    ground_truth = {
        "correct_answers": answer,
        "answer_entities": a_entity,
        "question_entities": q_entity,
        "graph_info": graph,
        "reasoning_path": reasoning_path
    }
    
    return {
        "data_source": "rog_cwq",
        "prompt": prompt,
        "reward_model": {
            "style": "function",
            "ground_truth": json.dumps(ground_truth)
        },
        "extra_info": {
            "split": "test",
            "index": idx,
            "original_id": example.get('qid', f'sample_{idx}'),
            "has_reasoning_path": len(reasoning_path) > 0
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/private/save/cwq-rog-rl/verl/verl/data/qald_10_en_test_original_2hop_remove_errors.jsonl', help='JSON file path')
    parser.add_argument('--output_dir', default='data/qald_en_test', help='Output directory')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS path')
    parser.add_argument('--test_size', default=0, type=float, help='Test set ratio (0.0-1.0)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    
    args = parser.parse_args()
    

    print(f"Loading JSONL file: {args.input_path}")
    data = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error at line {line_num}: {e}")
                    continue
    
    print(f"Successfully loaded {len(data)} samples")
    

    dataset = Dataset.from_list(data)
    

    if args.test_size > 0:
        split_dataset = dataset.train_test_split(
            test_size=args.test_size, 
            seed=args.seed,
            shuffle=True
        )
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']
        
        print(f"Dataset split: Training set {len(train_dataset)} samples, Test set {len(test_dataset)} samples")
    else:
        train_dataset = dataset
        test_dataset = None
        print(f"No test set split, all as training set: {len(train_dataset)} samples")
    

    train_processed = train_dataset.map(function=process_rog_cwq_sample, with_indices=True)
    

    local_dir = os.path.expanduser(args.output_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_processed.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    if test_dataset is not None:
        test_processed = test_dataset.map(function=process_rog_cwq_sample, with_indices=True)
        test_processed.to_parquet(os.path.join(local_dir, 'test.parquet'))
    

    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
    
    if test_dataset is not None:
        print(f"Data preprocessing completed! Training set: {len(train_processed)}, Test set: {len(test_processed)}")
    else:
        print(f"Data preprocessing completed! Training set: {len(train_processed)}")
    
    print(f"Output files saved to: {local_dir}")

if __name__ == '__main__':
    main()
