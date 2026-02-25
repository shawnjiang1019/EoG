

import json
import os

NUM_LINES = 5

class TemplateFileError(Exception):
    pass

class PRBodyLoadError(Exception):
    pass

class PRDescriptionError(Exception):
    pass

template_file = os.path.join(os.getenv("GITHUB_WORKSPACE", "."), ".github", "PULL_REQUEST_TEMPLATE.md")

def load_template(path):
    lines = []
    try:
        with open(path, encoding="utf-8") as f:
            for _ in range(NUM_LINES):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
        return lines
    except Exception as e:
        raise TemplateFileError(f"Failed to read PR template (first {NUM_LINES} lines) at {path}: {e}") from e

def load_pr_body(event_path):
    try:
        with open(event_path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("pull_request", {}).get("body", "") or ""
    except Exception as e:
        raise PRBodyLoadError(f"Failed to read PR body from {event_path}: {e}") from e

def check_pr_description(body, template_lines):
    pr_lines = body.splitlines(keepends=True)
    pr_first = [x.strip() for x in pr_lines[:NUM_LINES]]
    if pr_first == template_lines:
        raise PRDescriptionError(
            "It looks like you haven't updated the '### What does this PR do?' section. Please replace "
            "the placeholder text with a concise description of what your PR does."
        )
    else:
        print(pr_first)
        print(template_lines)

def main():
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path:
        raise OSError("GITHUB_EVENT_PATH is not set.")

    template_lines = load_template(template_file)
    pr_body = load_pr_body(event_path)
    check_pr_description(pr_body, template_lines)

    print("✅ '### What does this PR do?' section has been filled out.")

if __name__ == "__main__":
    main()
