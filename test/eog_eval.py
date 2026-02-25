import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock
from openai import OpenAI

try:
    client = OpenAI(

    base_url="xx",

    api_key='xxx',
    timeout=120
    )
except Exception as e:

    exit()

class KGReasoner:
    def __init__(self, input_file, output_file, api_key, max_workers=5, batch_size=10):

        self.client = client
        self.input_file = input_file
        self.output_file = output_file

        self.model_name = "gemini-2.5-flash" 
        self.max_workers = max_workers
        self.batch_size = batch_size

        self.result_cache = []
        self.cache_lock = Lock()

        self.system_prompt = """You are a helpful assistant that answers questions based on knowledge graphs. 
                1. First, reason through the problem inside <think> and </think> tags. Here you can planning, memory, check for mistakes to reflect，prune the entities and relations.
                2. When confident, output the final answer inside <answer> and </answer> tags. Your answer must strictly follow the rules provided by the user.

                You will receive:
                - question: A natural language question that needs to be answered 
                - begin_entity: The starting entity in the knowledge graph that can be used as a reference point
                - graph: Knowledge graph information in the form of triples (subject, relation, object)

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
        self.user_prompt = """Answer the given question based on the knowledge graph. You must first conduct reasoning step by step, and put your final answer inside <answer> and </answer>.
                Rules:
                1. When you have the final answer, you can directly provide the answer inside <answer></answer>, without detailed illustrations. For example, <answer> ["Washington, D.C."] </answer>
                2. If the knowledge graph information is insufficient to answer the question, you may use your own knowledge to supplement the reasoning process. 
                3. If there are multiple possible answers, present them in list format using square brackets []. For example, <answer>["Beijing", "Washington DC"]</answer>.
                4. Because questions usually have multiple answers, you should consider all possible answers and provide them in the list format.
                5. If multiple starting entities are provided, you should analyze each one systematically, such as Consider how different starting points might lead to different answers ,combine insights from all starting entities to form answers and if starting entities are related, explore their connections in the knowledge graph.
                You must use this format:
                  <think>...</think> 
                  <answer>...</answer> """

    def _call_openai_api(self, question, begin_entities, graph):
        prompt = f"""Question: {question}
                Entities: {json.dumps(begin_entities)}
                Graph: {json.dumps(graph)}
                {self.user_prompt}."""

        messages = [
            {"role": "system", "content": self.system_prompt},

            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,

            timeout=120
        )

        return response.choices[0].message.content.strip()

    def _batch_write_results(self):
        if not self.result_cache:
            return

        with open(self.output_file, "a", encoding="utf-8") as out_f:
            for result in self.result_cache:
                json.dump(result, out_f, ensure_ascii=False)
                out_f.write("\n")

        self.result_cache.clear()

    def _process_single_question(self, data, index):
        try:
            begin_entities = data["q_entity"] if isinstance(data["q_entity"], list) else [data["q_entity"]]

            model_answer = self._call_openai_api(
                question=data["question"],
                begin_entities=begin_entities,
                graph=data["graph"]
            )
            result = {"id": data["id"], "answer": model_answer, "golden_answer": data['a_entity']}
            is_success = True
        except Exception as e:
            result = {
                "id": data.get("id", f"error_{index + 1}"),
                "question": data.get("question", ""),
                "error": str(e),
                "answer": None
            }
            is_success = False

        with self.cache_lock:
            self.result_cache.append(result)
            if len(self.result_cache) >= self.batch_size:
                self._batch_write_results()

        return {"is_success": is_success}

    def process_kg_questions(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        with open(self.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_questions = len(lines)
            print(f"Starting processing: {total_questions} questions total | Concurrent threads: {self.max_workers} | Batch write threshold: {self.batch_size}")
        questions_data = [(json.loads(line.strip()), idx) for idx, line in enumerate(lines)]

        success_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_single_question, data, idx)
                       for data, idx in questions_data]

            for future in tqdm(as_completed(futures), total=total_questions, desc="Processing progress"):
                task_result = future.result()
                if task_result["is_success"]:
                    success_count += 1

        with self.cache_lock:
            self._batch_write_results()

        error_count = total_questions - success_count
        print(f"\nProcessing completed! Total: {total_questions} | Success: {success_count} | Failed: {error_count}")

if __name__ == "__main__":

    reasoner = KGReasoner(
        input_file=r"inputdata",
        output_file=r"outputdata",

        api_key="sk-...",
        max_workers=10,
        batch_size=20
    )
    reasoner.process_kg_questions()