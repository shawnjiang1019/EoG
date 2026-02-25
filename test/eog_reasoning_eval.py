import json
import time
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
from openai import OpenAI
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

@dataclass
class EvaluationCriteria:
    name: str
    description: str
    scoring_guide: str

class LLMEvaluator:
    def __init__(self):
        try:
            self.client = OpenAI(
                base_url="xx",
                api_key='xxx',
                timeout=120
            )
        except Exception as e:
            print(f"Error: Failed to initialize OpenAI client. {e}")
            raise e
        
        self.criteria = self._initialize_criteria()
        
    def _initialize_criteria(self) -> List[EvaluationCriteria]:
        return [
            EvaluationCriteria(
                "comprehensiveness",
                "whether the thinking considers all important aspects and is thorough",
                """Scoring Guide (0–10):
- 10: Extremely thorough, covering all relevant angles and considerations with depth.
- 8–9: Covers most key aspects clearly and thoughtfully; only minor omissions.
- 6–7: Covers some important aspects, but lacks depth or overlooks notable areas.
- 4–5: Touches on a few relevant points, but overall lacks substance or completeness.
- 1–3: Sparse or shallow treatment of the topic; misses most key aspects.
- 0: No comprehensiveness at all; completely superficial or irrelevant."""
            ),
            EvaluationCriteria(
                "knowledgeability",
                "whether the thinking is rich in insightful, domain-relevant knowledge",
                """Scoring Guide (0–10):
- 10: Demonstrates exceptional depth and insight with strong domain-specific knowledge.
- 8–9: Shows clear domain knowledge with good insight; mostly accurate and relevant.
- 6–7: Displays some understanding, but lacks depth or has notable gaps.
- 4–5: Limited knowledge shown; understanding is basic or somewhat flawed.
- 1–3: Poor grasp of relevant knowledge; superficial or mostly incorrect.
- 0: No evidence of meaningful knowledge."""
            ),
            EvaluationCriteria(
                "correctness",
                "whether the reasoning and answer are logically and factually correct",
                """Scoring Guide (0–10):
- 10: Fully accurate and logically sound; no flaws in reasoning or facts.
- 8–9: Mostly correct with minor inaccuracies or small logical gaps.
- 6–7: Partially correct; some key flaws or inconsistencies present.
- 4–5: Noticeable incorrect reasoning or factual errors throughout.
- 1–3: Largely incorrect, misleading, or illogical.
- 0: Entirely wrong or nonsensical."""
            ),
            EvaluationCriteria(
                "relevance",
                "whether the reasoning and answer are highly relevant and helpful to the question",
                """Scoring Guide (0–10):
- 10: Fully focused on the question; highly relevant and helpful.
- 8–9: Mostly on point; minor digressions but overall useful.
- 6–7: Generally relevant, but includes distractions or less helpful parts.
- 4–5: Limited relevance; much of the response is off-topic or unhelpful.
- 1–3: Barely related to the question or largely unhelpful.
- 0: Entirely irrelevant."""
            ),
            EvaluationCriteria(
                "diversity",
                "whether the reasoning is thought-provoking, offering varied or novel perspectives",
                """Scoring Guide (0–10):
- 10: Exceptionally rich and original; demonstrates multiple fresh and thought-provoking ideas.
- 8–9: Contains a few novel angles or interesting perspectives.
- 6–7: Some variety, but generally safe or conventional.
- 4–5: Mostly standard thinking; minimal diversity.
- 1–3: Very predictable or monotonous.
- 0: No diversity or originality at all."""
            ),
            EvaluationCriteria(
                "logical_coherence",
                "whether the reasoning is internally consistent, clear, and well-structured",
                """Scoring Guide (0–10):
- 10: Highly logical, clear, and easy to follow throughout.
- 8–9: Well-structured with minor lapses in flow or clarity.
- 6–7: Some structure and logic, but a few confusing or weakly connected parts.
- 4–5: Often disorganized or unclear; logic is hard to follow.
- 1–3: Poorly structured and incoherent.
- 0: Entirely illogical or unreadable."""
            ),
            EvaluationCriteria(
                "factuality",
                "whether the reasoning and answer are based on accurate and verifiable facts",
                """Scoring Guide (0–10):
- 10: All facts are accurate and verifiable.
- 8–9: Mostly accurate; only minor factual issues.
- 6–7: Contains some factual inaccuracies or unverified claims.
- 4–5: Several significant factual errors.
- 1–3: Mostly false or misleading.
- 0: Completely fabricated or factually wrong throughout."""
            ),
            EvaluationCriteria(
                "exploration",
                "whether the thinking process explore enough entities and relationships on the graph",
                """Scoring Guide (0–10):
- 10: Demonstrates highly systematic and in-depth understanding, leaving no crucial information or perspective untouched.
- 8–9: Presents a clear and thoughtful analysis, with only minor omissions.
- 6–7: Addresses relevant points but lacks depth or overlooks significant areas.
- 4–5: Touches on a few points but is generally incomplete and lacks detail.
- 1–3: The response is minimal, superficial, and misses most key elements.
- 0: The response is completely superficial or unrelated to the topic."""
            )
        ]
    
    def extract_reasoning(self, raw_output: str) -> str:
        answer_match = re.search(r'<answer>', raw_output, re.IGNORECASE)
        if answer_match:
            return raw_output[:answer_match.start()].strip()
        return raw_output.strip()
    
    def create_evaluation_prompt(self, question: str, reasoning: str, graph_info: List[List[str]], 
                               ground_truth: List[str], criterion: EvaluationCriteria) -> str:
        

        graph_sample = graph_info
        graph_str = "\n".join([f"- {triple}" for triple in graph_sample])
        
        prompt = f"""You are an expert evaluator tasked with scoring reasoning quality based on specific criteria.

**Question:** {question}

**Ground Truth Answer:** {ground_truth}

**Available Graph Information (sample):**
{graph_str}

**Reasoning to Evaluate:**
{reasoning}

**Evaluation Criterion:** {criterion.name}
**Description:** {criterion.description}

{criterion.scoring_guide}

Please evaluate the reasoning based ONLY on the "{criterion.name}" criterion. 

Your response should be in this exact format:
SCORE: [number from 0-10]
EXPLANATION: [brief explanation of why you gave this score, referencing specific aspects of the reasoning]

Be precise and objective in your evaluation."""

        return prompt
    
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"ERROR: Failed after {max_retries} attempts - {str(e)}"
    
    def parse_llm_response(self, response: str) -> tuple[int, str]:
        try:
            score_match = re.search(r'SCORE:\s*(\d+)', response)
            explanation_match = re.search(r'EXPLANATION:\s*(.*)', response, re.DOTALL)
            
            score = int(score_match.group(1)) if score_match else 0
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            

            score = max(0, min(10, score))
            
            return score, explanation
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            return 0, f"Parse error: {str(e)}"
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raw_output = sample.get("raw_model_output", "")
        reasoning = self.extract_reasoning(raw_output)
        
        if not reasoning.strip():
            return {
                "scores": {criterion.name: 0 for criterion in self.criteria},
                "explanations": {criterion.name: "No reasoning provided" for criterion in self.criteria},
                "overall_score": 0,
                "error": "No reasoning content found"
            }
        
        question = sample.get("question", "")
        graph_info = sample.get("graph_info", [])
        ground_truth = sample.get("ground_truth_answers", [])
        
        scores = {}
        explanations = {}
        

        for criterion in self.criteria:
            prompt = self.create_evaluation_prompt(question, reasoning, graph_info, ground_truth, criterion)
            response = self.call_llm(prompt)
            
            if not response.startswith("ERROR:"):
                score, explanation = self.parse_llm_response(response)
                scores[criterion.name] = score
                explanations[criterion.name] = explanation
            else:
                scores[criterion.name] = 0
                explanations[criterion.name] = response
            

            time.sleep(0.2)
        
        overall_score = sum(scores.values()) / len(scores) if scores else 0
        
        return {
            "scores": scores,
            "explanations": explanations,
            "overall_score": overall_score,
            "reasoning_length": len(reasoning.split())
        }

def evaluate_sample_worker(sample_data):
    try:
        evaluator = LLMEvaluator()
        sample, sample_index = sample_data
        
        print(f"Process {os.getpid()}: Evaluating sample {sample_index}")
        
        evaluation = evaluator.evaluate_sample(sample)
        
        result = {
            "index": sample.get("index", sample_index),
            "original_id": sample.get("original_id", ""),
            "evaluation": evaluation
        }
        
        print(f"Process {os.getpid()}: Completed sample {sample_index}, overall score: {evaluation['overall_score']:.2f}")
        
        return result
        
    except Exception as e:
        print(f"Process {os.getpid()}: Error evaluating sample {sample_index}: {e}")
        return {
            "index": sample.get("index", sample_index),
            "original_id": sample.get("original_id", ""),
            "evaluation": {
                "scores": {name: 0 for name in ["comprehensiveness", "knowledgeability", "correctness", 
                          "relevance", "diversity", "logical_coherence", "factuality", "exploration"]},
                "explanations": {name: f"Error: {str(e)}" for name in ["comprehensiveness", "knowledgeability", 
                               "correctness", "relevance", "diversity", "logical_coherence", "factuality", "exploration"]},
                "overall_score": 0,
                "error": str(e)
            }
        }

def evaluate_with_multiprocess(input_file: str, output_file: str, sample_size: Optional[int] = None,
                              num_processes: int = 4):
    

    samples = []
    print(f"Loading samples from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                sample = json.loads(line)
                samples.append(sample)
                
                if sample_size and len(samples) >= sample_size:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(samples)} samples")
    print(f"Using {num_processes} processes")
    

    sample_data = [(sample, i) for i, sample in enumerate(samples)]
    

    print("Starting multiprocess LLM evaluation...")
    start_time = time.time()
    
    results = []
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:

        future_to_index = {executor.submit(evaluate_sample_worker, data): i 
                          for i, data in enumerate(sample_data)}
        

        for future in as_completed(future_to_index):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                print(f"Completed {completed_count}/{len(samples)} samples")
                

                if completed_count % 50 == 0:
                    intermediate_file = f"/home/yanshiqi/my-project/intermediate_results_{completed_count}.json"

                    sorted_results = sorted(results, key=lambda x: x.get("index", 0))
                    with open(intermediate_file, 'w', encoding='utf-8') as f:
                        json.dump(sorted_results, f, indent=2, ensure_ascii=False)
                    print(f"Intermediate results saved to {intermediate_file}")
                    

                    valid_results = [r for r in sorted_results if "error" not in r["evaluation"]]
                    if valid_results:
                        avg_overall = sum(r["evaluation"]["overall_score"] for r in valid_results) / len(valid_results)
                        print(f"Current average overall score: {avg_overall:.2f}/10")
                
            except Exception as e:
                print(f"Error processing future: {e}")

                index = future_to_index[future]
                sample = samples[index]
                error_result = {
                    "index": sample.get("index", index),
                    "original_id": sample.get("original_id", ""),
                    "evaluation": {
                        "scores": {name: 0 for name in ["comprehensiveness", "knowledgeability", "correctness", 
                                  "relevance", "diversity", "logical_coherence", "factuality", "exploration"]},
                        "explanations": {name: f"Processing error: {str(e)}" for name in ["comprehensiveness", 
                                       "knowledgeability", "correctness", "relevance", "diversity", "logical_coherence", 
                                       "factuality", "exploration"]},
                        "overall_score": 0,
                        "error": str(e)
                    }
                }
                results.append(error_result)
                completed_count += 1
    
    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    

    results = sorted(results, key=lambda x: x.get("index", 0))
    

    evaluator = LLMEvaluator()
    total_scores = {criterion.name: 0 for criterion in evaluator.criteria}
    total_scores["overall"] = 0
    valid_results = [r for r in results if "error" not in r["evaluation"]]
    
    for result in valid_results:
        for criterion_name, score in result["evaluation"]["scores"].items():
            total_scores[criterion_name] += score
        total_scores["overall"] += result["evaluation"]["overall_score"]
    
    if valid_results:
        avg_scores = {k: v / len(valid_results) for k, v in total_scores.items()}
    else:
        avg_scores = {k: 0 for k in total_scores.keys()}
    

    output_data = {
        "metadata": {
            "total_samples": len(samples),
            "valid_evaluations": len(valid_results),
            "evaluation_time": end_time - start_time,
            "num_processes": num_processes,
            "api_base": "xx",
            "model": "gpt-4o-mini"
        },
        "summary": {
            "average_scores": avg_scores
        },
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print(f"Valid evaluations: {len(valid_results)}/{len(samples)}")
    print(f"Success rate: {len(valid_results)/len(samples)*100:.1f}%")
    print("\nAverage Scores:")
    for criterion_name, score in avg_scores.items():
        print(f"  {criterion_name}: {score:.2f}/10")

if __name__ == "__main__":

    input_file = "inputdata"
    output_file = "outputdata"
    

    sample_size = 3384
    num_processes = 8
    
    evaluate_with_multiprocess(
        input_file=input_file,
        output_file=output_file,
        sample_size=sample_size,
        num_processes=num_processes
    )