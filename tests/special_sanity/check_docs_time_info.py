

import sys
from pathlib import Path

ALLOW_LIST = {
    "docs/README.md",
    "docs/legacy/*.rst",
    "docs/index.rst",
    "docs/start/install.rst",
    "docs/start/quickstart.rst",
    "docs/README_vllm0.7.md",
}

DOCS_DIR = Path("docs")

def is_allowed(path: Path) -> bool:
    rel = str(path)
    for pattern in ALLOW_LIST:
        if Path(rel).match(pattern):
            return True
    return False

def main():
    if not DOCS_DIR.exists():
        print(f"Error: Documentation directory '{DOCS_DIR}' does not exist.", file=sys.stderr)
        sys.exit(1)

    missing = []

    for ext in ("*.md", "*.rst"):
        for path in DOCS_DIR.rglob(ext):
            if is_allowed(path):
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            if "Last updated" not in text:
                missing.append(path)

    if missing:
        print("\nThe following files are missing the 'Last updated' string:\n")
        for p in missing:
            print(f"  - {p}")
        print(f"\nTotal missing: {len(missing)}\n", file=sys.stderr)
        raise AssertionError(
            "Some documentation files lack a 'Last updated' line. Please include info such as "
            "'Last updated: mm/dd/yyyy' to indicate the last update time of the document."
        )
    else:
        print("✅ All checked files contain 'Last updated'.")

if __name__ == "__main__":
    main()
