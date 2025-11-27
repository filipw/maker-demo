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
                    "Think step by step and show your calculation logic. " 
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
            max_tokens=600, 
            verbose=False,
            sampler=make_sampler(temp) 
        )
        
        for token in ["<|assistant|>", "<|end|>", "<|user|>"]:
            response = response.replace(token, "")
            
        return response.strip()

    def extract_answer(self, text: str) -> str:
        match = re.search(r"Final Answer:\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)", text, re.IGNORECASE)
        if match:
            raw_time = match.group(1).upper().replace(" ", "")
            if len(raw_time) == 6: # e.g. 9:00AM
                raw_time = "0" + raw_time
            return raw_time
        return "PARSE_ERROR"

    def solve_reliable(self, task: str, k_threshold: int = 2, max_attempts: int = 15):
        print(f"--- MAKER Step (Target: Ahead by {k_threshold}) ---")
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
                # Red-flagging: silently discard malformed outputs
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
                print(f"🎉 CONVERGENCE REACHED: {leader_ans}\n")
                return leader_ans

        print(">>> WARNING: Max attempts reached.")
        return votes.most_common(1)[0][0]

def main():
    agent = LocalAgent(LOCAL_MODEL_PATH)
    
    print("\n🚀 STARTING DECOMPOSED PROTOCOL\n")

    # ---------------------------------------------------------
    # STEP 1: Travel from A to B
    # ---------------------------------------------------------
    prompt_1 = (
        "A train leaves Station A at 8:15 AM. "
        "It takes 45 minutes to reach Station B. "
        "What time does the train arrive at Station B?"
    )
    time_b_arrival = agent.solve_reliable(prompt_1, k_threshold=3)

    # ---------------------------------------------------------
    # STEP 2: Stop at B (calculate departure)
    # ---------------------------------------------------------
    # We inject the result from Step 1 into the prompt for Step 2
    prompt_2 = (
        f"A train arrives at Station B at {time_b_arrival}. "
        "It stops at Station B for 10 minutes. "
        "What time does the train depart Station B?"
    )
    time_b_depart = agent.solve_reliable(prompt_2, k_threshold=3)

    # ---------------------------------------------------------
    # STEP 3: Travel from B to C
    # ---------------------------------------------------------
    prompt_3 = (
        f"A train departs Station B at {time_b_depart}. "
        "It takes 1 hour and 50 minutes to reach Station C. "
        "What time does the train arrive at Station C?"
    )
    time_c_arrival = agent.solve_reliable(prompt_3, k_threshold=3)

    # ---------------------------------------------------------
    # STEP 4: Stop at C (calculate departure)
    # ---------------------------------------------------------
    prompt_4 = (
        f"A train arrives at Station C at {time_c_arrival}. "
        "It stops at Station C for 15 minutes. "
        "What time does the train depart Station C?"
    )
    time_c_depart = agent.solve_reliable(prompt_4, k_threshold=3)

    # ---------------------------------------------------------
    # STEP 5: Travel from C to D (Final Answer)
    # ---------------------------------------------------------
    prompt_5 = (
        f"A train departs Station C at {time_c_depart}. "
        "It takes 30 minutes to reach Station D. "
        "What time does the train arrive at Station D?"
    )
    final_result = agent.solve_reliable(prompt_5, k_threshold=3)

    print("==========================================")
    print(f"🤖 Final Consensus Answer: {final_result}")
    print("📝 Correct Answer: 11:45 AM")
    print("==========================================")

if __name__ == "__main__":
    main()