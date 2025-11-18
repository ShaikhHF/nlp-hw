import json
import sys
import os

REQUIRED_FIELDS = ["type", "text", "answer"]

def validate_question(q, index):
    errors = []

    # must be an object
    if not isinstance(q, dict):
        errors.append(f"Item #{index} is not an object/dict.")
        return errors

    # required fields
    for field in REQUIRED_FIELDS:
        if field not in q:
            errors.append(f"Item #{index} missing required field: '{field}'.")
    if errors:
        return errors

    q_type = q["type"]
    q_text = q["text"]
    q_answer = q["answer"]

    # basic type checks
    if not isinstance(q_type, str):
        errors.append(f"Item #{index} field 'type' must be a string.")
    if not isinstance(q_text, str):
        errors.append(f"Item #{index} field 'text' must be a string.")

    # MC-specific checks
    if q_type == "mc":
        if "choices" not in q:
            errors.append(f"Item #{index} is mc but missing 'choices'.")
        else:
            choices = q["choices"]
            if not isinstance(choices, list):
                errors.append(f"Item #{index} 'choices' must be a list.")
            else:
                # unique set
                unique_choices = list(dict.fromkeys(choices))  # preserves order, removes dups
                if len(unique_choices) != 4:
                    errors.append(
                        f"Item #{index} must have exactly 4 unique choices, got {unique_choices}."
                    )
                # still check answer is in original choices
                if q_answer not in choices:
                    errors.append(
                        f"Item #{index} answer '{q_answer}' is not in choices {choices}."
                    )

    return errors


def validate_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"File is not valid JSON: {e}")
        return
    except FileNotFoundError:
        print(f"File not found: {path}")
        return

    if not isinstance(data, list):
        print("JSON must be a list of questions.")
        return

    all_errors = []
    for i, q in enumerate(data):
        all_errors.extend(validate_question(q, i))

    if not all_errors:
        print("All questions look valid!")
    else:
        print("Found problems:")
        for e in all_errors:
            print(" -", e)


if __name__ == "__main__":
    # get path from CLI if provided
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "./question.json"
        print(f"No path provided, using default: {path}")

    # check path exists before validating
    if not os.path.exists(path):
        print(f"Error: file does not exist: {path}")
        sys.exit(1)

    validate_file(path)

