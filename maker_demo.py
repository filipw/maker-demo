import os
import re
from collections import Counter
from mlx_lm.utils import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from huggingface_hub.utils.tqdm import disable_progress_bars

# suppress noisy hugging face stuff
disable_progress_bars()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOCAL_MODEL_PATH = "mlx-community/Phi-4-mini-instruct-4bit"

class LocalAgent:
    def __init__(self, model_path):
        self.model, self.tokenizer = load(model_path) # type: ignore

    def run_inference(self, prompt: str, temp: float) -> str:
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a scheduling assistant. "
                    "You MUST end your response with exactly: 'Final Answer: <HH:MM AM/PM>'"
                )
            },
            {"role": "user", "content": prompt}
        ]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) # type: ignore
        else:
            full_prompt = prompt

        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=full_prompt, 
            max_tokens=400,
            verbose=False,
            sampler=make_sampler(temp) 
        )
        
        for token in ["<|assistant|>", "<|end|>", "<|user|>"]:
            response = response.replace(token, "")
            
        return response.strip()

    def extract_answer(self, text: str) -> str:
        match = re.search(r"Final Answer:\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)", text, re.IGNORECASE)
        if match:
            # normalize text to uppercase for voting (10:45 pm -> 10:45 PM)
            return match.group(1).upper().replace(" ", "")
        return "PARSE_ERROR"

    def solve_reliable(self, task: str, k_threshold: int = 2, max_attempts: int = 15):
        print(f"--- MAKER Protocol (Target: Ahead by {k_threshold}) ---")
        print(f"Task: {task}")
        
        votes = Counter()
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # 1. Generate (temp=0.7 allows the model to "take risks")
            raw_res = self.run_inference(task, temp=0.7)
            
            # 2. Extract
            answer_key = self.extract_answer(raw_res)
            
            if answer_key == "PARSE_ERROR":
                print(f"  Attempt {attempts}: [ERROR] Formatting error.")
                continue

            # 3. Vote
            votes[answer_key] += 1
            
            # 4. Check Standings
            sorted_votes = votes.most_common()
            leader_ans, leader_count = sorted_votes[0]
            
            runner_up_count = 0
            if len(sorted_votes) > 1:
                runner_up_count = sorted_votes[1][1]
            
            margin = leader_count - runner_up_count
            
            print(f"  Attempt {attempts}: Extracted '{answer_key}' | Leader: '{leader_ans}' (Ahead by {margin})")
            
            # 5. Convergence
            if margin >= k_threshold:
                print(f"\n🎉 CONVERGENCE REACHED: {leader_ans}")
                return leader_ans

        print(">>> WARNING: Max attempts reached.")
        return votes.most_common(1)[0][0]

def main():
    agent = LocalAgent(LOCAL_MODEL_PATH)
    
    # THE PROBLEM:
    # Depart Station A: 8:15 AM
    # Travel to B: 45 mins -> Arrive B at 9:00 AM
    # Stop at B: 10 mins   -> Depart B at 9:10 AM
    # Travel to C: 1 hour 50 mins -> 9:10 + 1:50 = 11:00 AM
    # Stop at C: 15 mins   -> Depart C at 11:15 AM
    # Travel to D: 30 mins -> Arrive D at 11:45 AM
    user_query = (
        "A train leaves Station A at 8:15 AM. "
        "It takes 45 minutes to reach Station B. "
        "It stops at Station B for 10 minutes. "
        "Then it takes 1 hour and 50 minutes to reach Station C. "
        "It stops at Station C for 15 minutes. "
        "Finally, it takes 30 minutes to reach Station D. "
        "What time does the train arrive at Station D?"
    )

    final = agent.solve_reliable(user_query, k_threshold=3)
    
    print(f"\n🤖 Final Consensus Answer: {final}")
    print("📝 Correct Answer: 11:45 AM")

if __name__ == "__main__":
    main()