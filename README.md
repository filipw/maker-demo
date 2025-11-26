# MAKER Protocol Demo

This sample is a simple Python demonstration of the **MAKER** framework (Massively Decomposed Agentic Processes), a method for building highly reliable workflows using Small Language Models (SLMs) entirely on-device. The script uses Apple's MLX framework to turn a stochastic, hallucination-prone local model into a reliable logic engine.

## Concept

This demo implements the "First-to-Ahead-by-`k`" voting strategy described in the research paper [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030).

> Elliot Meyerson and Giuseppe Paolo and Roberto Dailey and Hormoz Shahrzad and Olivier Francon and Conor F. Hayes and Xin Qiu and Babak Hodjat and Risto Miikkulainen (2025). *Solving a Million-Step LLM Task with Zero Errors*. arXiv.

The core idea is to trade variable compute time for accuracy. Instead of running a Small Language Model (SLM) once and hoping the answer is correct, the system treats the model as a probabilistic generator. It runs a continuous generation loop with high temperature until one answer statistically dominates the others (pulls ahead by a margin of `k`). This filters out random "hallucinations" while converging on the reproducible "truth."

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

The demo executes the MAKER reliability loop in three stages:

1.  **Step 1: Diverse Generation:** The local model generates a solution using a high temperature (e.g., 0.7). This forces answer diversity, encourages model to take risks, leading to varied outputs.
2.  **Step 2: Red-Flagging:** A rule-based filter immediately discards outputs that fail basic checks (e.g., formatting errors, missing steps) before they can pollute the vote.
3.  **Step 3: First-to-Ahead-by-`k` Voting:** Valid answers are tallied in real-time. The loop continues until a single answer leads the runner-up by a specific margin (`k=3`), guaranteeing a high-confidence consensus.

## Acknowledgements

This project is a simple practical implementation of concepts from the research on reliable autonomous agents:

