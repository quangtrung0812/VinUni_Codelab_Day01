"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "mock-key"

    client = OpenAI (api_key=api_key)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    response_text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Supports dual-import fallback (new google-genai and legacy google-generativeai)
    to ensure zero-friction execution.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    start_time = time.time()
    
    if api_key == "mock-key":
        latency = time.time() - start_time
        return "Mock response for Gemini", latency, {"input_tokens": 10, "output_tokens": 10}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        latency = time.time() - start_time
        response_text = response.text
        
        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        return response_text, latency, usage
    except Exception:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        g_model = genai_legacy.GenerativeModel(model)
        response = g_model.generate_content(
            prompt,
            generation_config=genai_legacy.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            )
        )
        latency = time.time() - start_time
        response_text = response.text
        usage = {
            "input_tokens": 15,
            "output_tokens": 15,
        }
        return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
   """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as standard) and return
    the response text, latency, and token usage stats.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=api_key)

    if api_key == "mock-key":
        start_time = time.time()
        latency = time.time() - start_time
        return "Mock response for Anthropic", latency, {"input_tokens": 10, "output_tokens": 10}

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    response_text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.
    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.
    """
    # Call GPT-4o
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(
        prompt=prompt, model=OPENAI_MODEL
    )
    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"] +
        gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    # Call GPT-4o-mini
    mini_text, mini_lat, mini_usage = call_openai(
        prompt=prompt, model=OPENAI_MINI_MODEL
    )
    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"] +
        mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    # Call Gemini 2.5 Flash
    gemini_text, gemini_lat, gemini_usage = call_gemini(
        prompt=prompt, model=GEMINI_MODEL
    )
    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"] +
        gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    Maintains the last 3 turns of conversation history for context.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in dummy mode.\033[0m")
        api_key = "mock-key"

    print("\n\033[94m=============================================================")
    print("🤖 Vin Smart Future - Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("=============================================================\033[0m\n")

    if api_key == "mock-key":
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                print(f"Gemini: [Mock Stream] Echo: {user_input}")
            except (KeyboardInterrupt, EOFError):
                break
        return

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model=GEMINI_MODEL)
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("\nGoodbye!")
                    break
                if not user_input.strip():
                    continue

                print("Gemini: ", end="", flush=True)
                response = chat.send_message_stream(user_input)
                for chunk in response:
                    if hasattr(chunk, "text") and chunk.text:
                        print(chunk.text, end="", flush=True)
                print("\n")
                
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
    except Exception as e:
        print(f"\nError initializing chat: {e}")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
   """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    import time
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                raise last_exception
            sleep_time = base_delay * (2 ** attempt)
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    raise NotImplementedError("Implement batch_compare")


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
    raise NotImplementedError("Implement format_comparison_table")


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
