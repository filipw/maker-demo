# MAKER Protocol Demo

This sample is a simple Python demonstration of the **MAKER** framework (Massively Decomposed Agentic Processes), a method for building highly reliable workflows described in the research paper [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030).

> Elliot Meyerson and Giuseppe Paolo and Roberto Dailey and Hormoz Shahrzad and Olivier Francon and Conor F. Hayes and Xin Qiu and Babak Hodjat and Risto Miikkulainen (2025). *Solving a Million-Step LLM Task with Zero Errors*. arXiv.

Here, we are using Small Language Models (SLMs) entirely on-device. The script uses Apple's MLX framework to turn a stochastic, hallucination-prone local model into a reliable logic engine.

## Concept

This demo illustrates the two core pillars of the MAKER protocol that allow small models to solve complex tasks:

1.  **Massive Decomposition:** Instead of asking the model to solve a complex multi-stage word problem in one "monolithic" prompt (which often leads to reasoning errors), the task is broken down into a chain of atomic "micro-steps."
2.  **Reliability Engine:** Each individual micro-step is verified using the "First-to-Ahead-by-`k`" voting strategy.

The core idea is to trade variable compute time for accuracy. The system treats the model as a probabilistic generator, running a continuous loop with high temperature until one answer statistically dominates the others (pulls ahead by a margin of `k`). By combining this reliability with strict task decomposition, the system can maintain logic over long horizons without "derailing."

## Requirements

* **Hardware**: **A Mac with Apple Silicon is required** to run this demo, as it uses Apple's MLX framework for efficient local inference.
* **Python**: 3.9+

The model used is `Phi-4-mini-instruct-4bit` from Hugging Face, optimized for MLX to run efficiently on local hardware.

## Setup & Configuration

1.  **Create and activate a Python virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install mlx-lm
    ```

## Usage

Once the setup is complete, simply run the demo script from your terminal:

```bash
python maker_demo.py
```

*Note: On the first run, it will automatically download the Phi-4 model (approx. 2-3GB).*

## How It Works

The demo solves a "complex" (from the perspective of Small Language Models) train scheduling problem by executing a **Decomposed Reliability Loop**:

### 1\. The Outer Loop (Decomposition & Chaining)

The script breaks the journey into 5 dependent sub-tasks (e.g., "Travel A-\>B", "Stop at B", "Travel B-\>C"). It acts as an orchestrator, dynamically injecting the **result** of Step $N$ (e.g., "9:00 AM") into the **prompt** for Step $N+1$.

### 2\. The Inner Loop (Reliable Execution)

For *every single step* in the chain, the `solve_reliable` function engages the MAKER protocol to ensure accuracy before moving on:

  * **Diverse Generation:** The local model generates a solution for the micro-step using a high temperature (0.7). This forces answer diversity and prevents the model from getting stuck in a single wrong path.
  * **Red-Flagging:** A rule-based filter immediately discards outputs that fail basic checks (e.g., formatting errors) before they can pollute the vote.
  * **First-to-Ahead-by-`k` Voting:** Valid answers are tallied in real-time. The loop continues until a single answer leads the runner-up by a specific margin (`k=3`), guaranteeing a high-confidence consensus for that specific step.


## Sample Output

```plaintext
🚀 STARTING DECOMPOSED PROTOCOL

--- MAKER Step (Target: Ahead by 3) ---
Task: A train leaves Station A at 8:15 AM. It takes 45 minutes to reach Station B. What time does the train arrive at Station B?
  Attempt 1: Extracted '09:15AM' | Leader: '09:15AM' (Ahead by 1)
  Attempt 2: Extracted '09:00AM' | Leader: '09:15AM' (Ahead by 0)
  Attempt 3: Extracted '09:00AM' | Leader: '09:00AM' (Ahead by 1)
  Attempt 4: Extracted '09:15AM' | Leader: '09:15AM' (Ahead by 0)
  Attempt 5: Extracted '08:45AM' | Leader: '09:15AM' (Ahead by 0)
  Attempt 7: Extracted '09:00AM' | Leader: '09:00AM' (Ahead by 1)
  Attempt 8: Extracted '09:00AM' | Leader: '09:00AM' (Ahead by 2)
  Attempt 9: Extracted '09:00AM' | Leader: '09:00AM' (Ahead by 3)
🎉 CONVERGENCE REACHED: 09:00AM

--- MAKER Step (Target: Ahead by 3) ---
Task: A train arrives at Station B at 09:00AM. It stops at Station B for 10 minutes. What time does the train depart Station B?
  Attempt 1: Extracted '09:10AM' | Leader: '09:10AM' (Ahead by 1)
  Attempt 2: Extracted '09:10AM' | Leader: '09:10AM' (Ahead by 2)
  Attempt 3: Extracted '09:10AM' | Leader: '09:10AM' (Ahead by 3)
🎉 CONVERGENCE REACHED: 09:10AM

--- MAKER Step (Target: Ahead by 3) ---
Task: A train departs Station B at 09:10AM. It takes 1 hour and 50 minutes to reach Station C. What time does the train arrive at Station C?
  Attempt 1: Extracted '10:00AM' | Leader: '10:00AM' (Ahead by 1)
  Attempt 2: Extracted '10:40AM' | Leader: '10:00AM' (Ahead by 0)
  Attempt 3: Extracted '11:00AM' | Leader: '10:00AM' (Ahead by 0)
  Attempt 4: Extracted '11:00AM' | Leader: '11:00AM' (Ahead by 1)
  Attempt 5: Extracted '10:40AM' | Leader: '10:40AM' (Ahead by 0)
  Attempt 6: Extracted '10:00AM' | Leader: '10:00AM' (Ahead by 0)
  Attempt 7: Extracted '10:50AM' | Leader: '10:00AM' (Ahead by 0)
  Attempt 8: Extracted '11:00AM' | Leader: '11:00AM' (Ahead by 1)
  Attempt 9: Extracted '11:00AM' | Leader: '11:00AM' (Ahead by 2)
  Attempt 10: Extracted '11:00AM' | Leader: '11:00AM' (Ahead by 3)
🎉 CONVERGENCE REACHED: 11:00AM

--- MAKER Step (Target: Ahead by 3) ---
Task: A train arrives at Station C at 11:00AM. It stops at Station C for 15 minutes. What time does the train depart Station C?
  Attempt 1: Extracted '11:15AM' | Leader: '11:15AM' (Ahead by 1)
  Attempt 2: Extracted '11:15AM' | Leader: '11:15AM' (Ahead by 2)
  Attempt 3: Extracted '11:15AM' | Leader: '11:15AM' (Ahead by 3)
🎉 CONVERGENCE REACHED: 11:15AM

--- MAKER Step (Target: Ahead by 3) ---
Task: A train departs Station C at 11:15AM. It takes 30 minutes to reach Station D. What time does the train arrive at Station D?
  Attempt 1: Extracted '11:45AM' | Leader: '11:45AM' (Ahead by 1)
  Attempt 2: Extracted '12:45PM' | Leader: '11:45AM' (Ahead by 0)
  Attempt 3: Extracted '11:45AM' | Leader: '11:45AM' (Ahead by 1)
  Attempt 4: Extracted '11:45AM' | Leader: '11:45AM' (Ahead by 2)
  Attempt 5: Extracted '11:45AM' | Leader: '11:45AM' (Ahead by 3)
🎉 CONVERGENCE REACHED: 11:45AM

==========================================
🤖 Final Consensus Answer: 11:45AM
📝 Correct Answer: 11:45 AM
==========================================
```

If you look closely, you can see the "battle" between the correct answer and the hallucinations.