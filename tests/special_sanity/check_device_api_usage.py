

import os
from argparse import ArgumentParser
from pathlib import Path

CUDA_KEYWORD_CHECK_WHITELIST = [
    "verl/utils/device.py",
    "recipe/prime/prime_ray_trainer.py",
    "recipe/spin/spin_trainer.py",
    "recipe/sppo/sppo_ray_trainer.py",
    "recipe/one_step_off_policy/ray_trainer.py",
    "verl/utils/profiler/nvtx_profile.py",
    "verl/utils/kernel/linear_cross_entropy.py",
    "verl/utils/rendezvous/ray_backend.py",
    "verl/single_controller/ray/base.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/reward_score/sandbox_fusion/utils.py",
    "verl/workers/reward_model/megatron/reward_model.py",
    "verl/workers/engine/fsdp/engine_impl.py",
]

NCCL_KEYWORD_CHECK_WHITELIST = [
    "verl/utils/device.py",
    "verl/third_party/sglang/parallel_state.py",
]

SEARCH_WHITELIST = CUDA_KEYWORD_CHECK_WHITELIST + NCCL_KEYWORD_CHECK_WHITELIST

SEARCH_KEYWORDS = [".cuda", '"cuda"', '"nccl"']

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=str)
    args = parser.parse_args()
    directory_in_str = args.directory

    pathlist = Path(directory_in_str).glob("**/*.py")
    for path in pathlist:
        path_in_str = str(path.absolute())

        path_in_whitelist = False

        for sw in SEARCH_WHITELIST:

            sw = sw.replace("/", os.sep)
            if sw in path_in_str:
                print(f"[SKIP] File {path_in_str} is in device api usage check whitelist, checking is skipped.")
                path_in_whitelist = True
                break

        if path_in_whitelist:
            continue

        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

            find_invalid_device_management = False

            for sk in SEARCH_KEYWORDS:
                if sk in file_content:
                    find_invalid_device_management = True
                    break

            print(
                f"[CHECK] File {path_in_str} is detected for device api usage check, check result: "
                f"{'success' if not find_invalid_device_management else f'failed, because detect {sk}'}."
            )

            assert not find_invalid_device_management, (
                f'file {path_in_str} contains .cuda/"cuda"/"nccl" usage, please use api in '
                f"verl/utils/device.py directly."
            )
