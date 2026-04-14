import numpy as np
from xai_explainer import FEATURE_NAMES, explain_values, format_explanation_report


def prompt_for_input():
    print("\nEnter soil feature values in the order below:")
    for idx, feature in enumerate(FEATURE_NAMES, 1):
        print(f"  {idx}. {feature}")

    values = []
    for feature in FEATURE_NAMES:
        while True:
            raw = input(f"{feature}: ").strip()
            if raw.lower() in {"quit", "exit"}:
                raise KeyboardInterrupt
            try:
                value = float(raw)
                values.append(value)
                break
            except ValueError:
                print("  Invalid value. Please enter a numeric value.")

    return values


def main():
    print("=" * 70)
    print("SOIL CONDITION PREDICTION WITH XAI EXPLANATION")
    print("=" * 70)
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            input_values = prompt_for_input()
            result = explain_values(input_values)
            print("\n" + "-" * 70)
            print(format_explanation_report(result))
            print("-" * 70 + "\n")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as exc:
            print(f"Error: {exc}\n")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
