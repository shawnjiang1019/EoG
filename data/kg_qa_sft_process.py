
import json
import pandas as pd
import argparse
from transformers import AutoTokenizer

def convert_jsonl_to_multiturn_parquet(input_file, output_file, model_path="Qwen/Qwen2.5-0.5B-Instruct", max_tokens=10000):
    

    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    

    system_content = """You are a helpful assistant that answers questions based on knowledge graphs. 
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

    def format_entities(entities):
        if not entities:
            return "Unknown"
        elif len(entities) == 1:
            return entities[0]
        else:
            entities_list = []
            for i, entity in enumerate(entities, 1):
                entities_list.append(f"{i}. {entity}")
            return "\n".join(entities_list)

    def check_token_length(messages, max_tokens):
        try:

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            

            total_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
            
            return total_tokens <= max_tokens, total_tokens
            
        except Exception as e:
            print(f"Error checking token length: {e}")
            return False, 0

    processed_data = []
    filtered_count = 0
    total_count = 0
    token_stats = []
    
    print(f"Starting to process file: {input_file}")
    print(f"Max token limit: {max_tokens}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    total_count += 1
                    if total_count % 1000 == 0:
                        print(f"Processed {total_count} samples...")
                    
                    data = json.loads(line)
                    

                    question = data.get('question', '')
                    q_entity = data.get('q_entity', [])
                    graph = data.get('graph', [])
                    generated_raw_text = data.get('generated_raw_text', '')
                    

                    if not generated_raw_text:
                        generated_think = data.get('generated_think', '')
                        generated_answer = data.get('generated_answer', '')
                        answer = data.get('answer', [])
                        
                        if generated_think and generated_answer:
                            generated_raw_text = f"<think>\n{generated_think}\n</think>\n\n<answer>\n{generated_answer}\n</answer>"
                        else:
                            answer_text = ', '.join(answer) if answer else 'Unknown'
                            generated_raw_text = f"<think>\nI will analyze the question and knowledge graph to find the answer.\n</think>\n\n<answer>\n{answer_text}\n</answer>"
                    

                    graph_text = str(graph) if graph else "No graph infomation is available"
                    entity_text = format_entities(q_entity)
                    

                    user_content = f"""Answer the given question based on the knowledge graph. You must first conduct reasoning step by step, and put your final answer inside <answer> and </answer>.

Rules:
1. When you have the final answer, you can directly provide the answer inside <answer></answer>, without detailed illustrations. For example, <answer> ["Washington, D.C."] </answer>
2. CRITICAL: Your final answer must only contain entities that are explicitly present in the provided graph triples
3. If there are multiple possible answers, present them in list format using square brackets []. For example, <answer>["Beijing", "Washington DC"]</answer>
4. Because questions usually have multiple answers, you should consider all possible answers and provide them in the list format
5. If multiple starting entities are provided, you should analyze each one systematically，such as Consider how different starting points might lead to different answers ,combine insights from all starting entities to form answers and if starting entities are related, explore their connections in the knowledge graph.

You must use this format:
<think>...</think>
<answer>...</answer>

Question: {question}

Knowledge Graph:
{graph_text}

Starting Entity: {entity_text}"""
                    

                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": generated_raw_text},
                        
                    ]
                    

                    is_valid, token_count = check_token_length(messages, max_tokens)
                    token_stats.append(token_count)
                    
                    if is_valid:
                        processed_data.append({
                            'messages': messages,
                            'id': data.get('id', f'sample_{line_num}'),
                            'token_count': token_count
                        })
                    else:
                        filtered_count += 1
                        if filtered_count <= 10:
                            print(f"Filtered sample {line_num}: {token_count} tokens > {max_tokens}")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error at line {line_num}: {e}")
                    filtered_count += 1
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    filtered_count += 1
                    continue
    

    print(f"\n=== Processing Statistics ===")
    print(f"Total samples: {total_count}")
    print(f"Successfully processed: {len(processed_data)}")
    print(f"Filtered samples: {filtered_count}")
    print(f"Retention rate: {len(processed_data)/total_count*100:.2f}%")
    
    if token_stats:
        import numpy as np
        print(f"\n=== Token Statistics ===")
        print(f"Token count - Min: {min(token_stats)}")
        print(f"Token count - Max: {max(token_stats)}")
        print(f"Token count - Mean: {np.mean(token_stats):.0f}")
        print(f"Token count - Median: {np.median(token_stats):.0f}")
        print(f"Samples exceeding {max_tokens} tokens: {sum(1 for t in token_stats if t > max_tokens)}")
    

    if processed_data:
        df_data = [{'messages': item['messages'], 'id': item['id']} for item in processed_data]
        df = pd.DataFrame(df_data)
        df.to_parquet(output_file, index=False)
        print(f"\nData saved to: {output_file}")
        print(f"Number of samples saved: {len(df)}")
    else:
        print("Warning: No valid samples, file not generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL data to multiturn conversation parquet format and filter oversize samples")
    parser.add_argument("--input", default="/private/save/cwq-rog-rl/verl/verl/data/Fvcs/le2/rog_cwq_filtered_train_0.1_with_sft_new_add_list_le2.jsonl", help="/private/save/cwq-rog-rl/verl/verl/data/rog_cwq_filtered_train_0.1_with_sft_new.jsonl")
    parser.add_argument("--output", default="/private/save/cwq-rog-rl/verl/verl/data/qween2sftdata_cwqle2_15000token_addlist_byqwen3.parquet", help="/private/save/cwq-rog-rl/verl/verl/data")
    parser.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct", help="Model path for calculating token length")
    parser.add_argument("--max_tokens", type=int, default=15000, help="Maximum token length limit")
    
    args = parser.parse_args()
    
    convert_jsonl_to_multiturn_parquet(args.input, args.output, args.model_path, args.max_tokens)