

from argparse import ArgumentParser
from pathlib import Path

license_head_bytedance = "Copyright 2024 Bytedance Ltd. and/or its affiliates"
license_head_bytedance_25 = "Copyright 2025 Bytedance Ltd. and/or its affiliates"

license_head_prime = "Copyright 2024 PRIME team and/or its affiliates"
license_head_individual = "Copyright 2025 Individual Contributor:"
license_head_sglang = "Copyright 2023-2024 SGLang Team"
license_head_modelbest = "Copyright 2025 ModelBest Inc. and/or its affiliates"
license_head_amazon = "Copyright 2025 Amazon.com Inc and/or its affiliates"
license_headers = [
    license_head_bytedance,
    license_head_bytedance_25,
    license_head_prime,
    license_head_individual,
    license_head_sglang,
    license_head_modelbest,
    license_head_amazon,
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=str)
    args = parser.parse_args()
    directory_in_str = args.directory

    pathlist = Path(directory_in_str).glob("**/*.py")
    for path in pathlist:

        path_in_str = str(path.absolute())
        print(path_in_str)
        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

            has_license = False
            for lh in license_headers:
                if lh in file_content:
                    has_license = True
                    break
            assert has_license, f"file {path_in_str} does not contain license"
