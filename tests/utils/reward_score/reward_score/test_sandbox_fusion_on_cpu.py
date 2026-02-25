

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import pytest

from verl.utils.reward_score.sandbox_fusion.utils import check_correctness

SANDBOX_URL = os.environ.get("SANDBOX_FUSION_URL")

skip_reason = "SANDBOX_FUSION_URL environment variable not set"
skip_condition = not SANDBOX_URL

CODE_SUCCESS = """
import sys
data = sys.stdin.read()
if data == 'input1':
    print('output1\\n', end='')
elif data == 'input2':
    print('output2\\n', end='')
else:
    print('unexpected input', end='')
"""

CODE_WRONG_OUTPUT = """
print('wrong_output\\n', end='')
"""

CODE_COMPILE_ERROR = """
a=b
"""

CODE_RUNTIME_ERROR = """
import sys
print("About to raise error", file=sys.stderr)
raise ValueError("This is a runtime error")
"""

CODE_TIMEOUT = """
import time
import sys
print("Sleeping...", file=sys.stderr)
time.sleep(10)
print("Finished sleeping", file=sys.stderr)
"""

INPUT_OUTPUT_VALID = {"inputs": ["input1", "input2"], "outputs": ["output1\n", "output2\n"]}

INPUT_OUTPUT_SINGLE = {"inputs": ["input1"], "outputs": ["output1\n"]}

INPUT_OUTPUT_MISMATCH = {"inputs": ["input1"], "outputs": ["output1\n", "output2\n"]}

INPUT_OUTPUT_INVALID_MISSING_KEY = {"inputs": ["input1"]}

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_correct():
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_SUCCESS)
    assert results == [True, True]
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[0]["stdout"] == "output1\n"
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[1]["stdout"] == "output2\n"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_wrong_output():
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_WRONG_OUTPUT)
    assert results == [False, False]
    assert metadata_list[0]["status"] == "wrong_answer"
    assert metadata_list[0]["stdout"] == "wrong_output\n"
    assert metadata_list[1]["status"] == "wrong_answer"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_compile_error():
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_COMPILE_ERROR, language="cpp")
    assert results == [-4, -4]
    assert metadata_list[0]["status"] == "compile_error"
    assert metadata_list[1]["status"] == "compile_error"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_error():
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_RUNTIME_ERROR)
    assert results == [-2]
    assert metadata_list[0]["status"] == "runtime_error"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_timeout():
    test_timeout = 5
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_TIMEOUT, timeout=test_timeout)
    assert results == [-3]
    assert metadata_list[0]["status"] == "timeout"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_concurrency_high_load():
    concurrency_level = 100

    wrong_answer_indices = {10, 25, 50}
    timeout_indices = {5, 30, 60, 90}

    high_load_inputs = []
    high_load_outputs = []
    expected_results_map = {}

    for i in range(concurrency_level):
        if i in timeout_indices:

            high_load_inputs.append(f"input_timeout_{i}")

            high_load_outputs.append(f"output_{i}\n")
            expected_results_map[i] = -3
        elif i in wrong_answer_indices:
            high_load_inputs.append(f"input_{i}")

            high_load_outputs.append(f"wrong_output_{i}\n")
            expected_results_map[i] = False
        else:
            high_load_inputs.append(f"input_{i}")

            high_load_outputs.append(f"output_{i}\n")
            expected_results_map[i] = True

    high_load_in_outs = {"inputs": high_load_inputs, "outputs": high_load_outputs}

    code_mixed_concurrent = """
import sys
import time
data = sys.stdin.read()
if data.startswith('input_timeout_'):
    time.sleep(20)
    print(f"output_{data.split('_')[-1]}\\n", end='')
elif data.startswith('input_'):
    print(f"output_{data.split('_')[-1]}\\n", end='')
else:
    print("unknown_input\\n", end='')
"""

    test_timeout = 15

    start_time = time.time()
    results, metadata_list = check_correctness(
        SANDBOX_URL,
        high_load_in_outs,
        code_mixed_concurrent,
        timeout=test_timeout,
    )
    end_time = time.time()
    duration = end_time - start_time
    print(
        f"\nHigh concurrency test ({concurrency_level} cases with {len(wrong_answer_indices)} wrong answers, "
        f"{len(timeout_indices)} timeouts) duration: {duration:.2f} seconds"
    )

    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"

    correct_count = 0
    wrong_count = 0
    timeout_count = 0
    unexpected_results = []
    for i, r in enumerate(results):
        expected = expected_results_map[i]
        if r == expected:
            if expected is True:
                correct_count += 1
            elif expected is False:
                wrong_count += 1
            elif expected == -3:
                timeout_count += 1
        else:
            unexpected_results.append((i, r, f"Expected {expected}"))

    print(
        f"Correct results (True): {correct_count}/"
        f"{concurrency_level - len(wrong_answer_indices) - len(timeout_indices)}"
    )
    print(f"Expected wrong answers (False, correctly identified): {wrong_count}/{len(wrong_answer_indices)}")
    print(f"Expected timeouts (-3, correctly identified): {timeout_count}/{len(timeout_indices)}")

    if unexpected_results:
        print("Unexpected results found:")
        for idx, res, expected_str in unexpected_results[:10]:
            print(f"  Index {idx}: Got {res}, {expected_str}. Metadata: {metadata_list[idx]}")
        raise AssertionError(f"Found {len(unexpected_results)} unexpected results.")

    assert correct_count == concurrency_level - len(wrong_answer_indices) - len(timeout_indices), (
        "Incorrect number of successful results"
    )
    assert wrong_count == len(wrong_answer_indices), "Incorrect number of identified wrong answers"
    assert timeout_count == len(timeout_indices), "Incorrect number of identified timeouts"

    assert len(metadata_list) == concurrency_level

    first_correct_index = next(
        i for i in range(concurrency_level) if i not in wrong_answer_indices and i not in timeout_indices
    )
    assert metadata_list[first_correct_index]["status"] == "success"
    assert metadata_list[first_correct_index]["stdout"] == f"output_{first_correct_index}\n"

    first_wrong_index = min(wrong_answer_indices)
    assert metadata_list[first_wrong_index]["status"] == "wrong_answer"
    assert metadata_list[first_wrong_index]["stdout"] == f"output_{first_wrong_index}\n"
    assert metadata_list[first_wrong_index]["expected_output"] == f"wrong_output_{first_wrong_index}\n"

    first_timeout_index = min(timeout_indices)
    assert metadata_list[first_timeout_index]["status"] == "timeout"

@patch("verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api")
def test_unit_concurrency_order(mock_call_sandbox_api):
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {"inputs": ["input1", "input2", "input3"], "outputs": ["output1", "output2", "output3"]}

    def side_effect(*args, **kwargs):
        stdin = kwargs.get("stdin")
        if stdin == "input1":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}},
                None,
            )
        elif stdin == "input2":
            time.sleep(0.1)
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output2", "return_code": 0}},
                None,
            )
        elif stdin == "input3":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}},
                None,
            )
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    assert results == [True, True, True]
    assert len(metadata_list) == 3
    assert metadata_list[0]["case_index"] == 0
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["case_index"] == 1
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[2]["case_index"] == 2
    assert metadata_list[2]["status"] == "success"
    assert mock_call_sandbox_api.call_count == 3

@patch("verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api")
def test_unit_api_timeout_error_concurrent(mock_call_sandbox_api):
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {"inputs": ["input1", "input2_timeout", "input3"], "outputs": ["output1", "output2", "output3"]}

    api_error_message = "API Call Failed: Gateway Timeout (504) on attempt 3/3"

    def side_effect(*args, **kwargs):
        stdin = kwargs.get("stdin")
        if stdin == "input1":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}},
                None,
            )
        elif stdin == "input2_timeout":
            return (None, api_error_message)
        elif stdin == "input3":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}},
                None,
            )
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    assert results == [True, -1, True]
    assert len(metadata_list) == 3
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["status"] == "api_error"
    assert metadata_list[1]["api_request_error"] == api_error_message
    assert metadata_list[2]["status"] == "success"
    assert mock_call_sandbox_api.call_count == 3

MAX_GLOBAL_CONCURRENCY_LIMIT_TEST = 5

NUM_PROCESSES_TEST = 4

NUM_TASKS_PER_PROCESS_TEST = 3

SIMULATED_API_CALL_DURATION_TEST = 0.2

def _mock_api_call_for_concurrency_tracking(
    active_calls_counter,
    max_calls_tracker,
    call_lock,

    sandbox_fusion_url,
    code,
    stdin,
    compile_timeout,
    run_timeout,
    memory_limit_mb,
    language,
):

    with call_lock:
        active_calls_counter.value += 1
        if active_calls_counter.value > max_calls_tracker.value:
            max_calls_tracker.value = active_calls_counter.value

    time.sleep(SIMULATED_API_CALL_DURATION_TEST)

    with call_lock:
        active_calls_counter.value -= 1

    return {
        "status": "Success",
        "run_result": {"status": "Finished", "stdout": f"mock_output_for_{stdin}", "return_code": 0},
    }, None

def _process_pool_worker_for_concurrency_test(
    sandbox_url,
    in_outs,
    generation,
    memory_limit_mb,
    language,
    timeout,
    mp_semaphore_for_check_correctness,
    active_calls_counter,
    max_calls_tracker,
    call_lock,
):

    curried_mock_api_call = (
        lambda sandbox_fusion_url, code, stdin, compile_timeout, run_timeout, memory_limit_mb, language: (
            _mock_api_call_for_concurrency_tracking(
                active_calls_counter,
                max_calls_tracker,
                call_lock,
                sandbox_fusion_url,
                code,
                stdin,
                compile_timeout,
                run_timeout,
                memory_limit_mb,
                language,
            )
        )
    )

    import os

    import verl.utils.reward_score.sandbox_fusion.utils

    print(
        f"[Worker PID:{os.getpid()}] Original call_sandbox_api: "
        f"{verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api}",
        flush=True,
    )

    with patch(
        "verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api", side_effect=curried_mock_api_call
    ) as mock_obj:

        print(
            f"[Worker PID:{os.getpid()}] Patched call_sandbox_api: "
            f"{verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api}",
            flush=True,
        )
        print(f"[Worker PID:{os.getpid()}] Mock object: {mock_obj}", flush=True)

        results, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_url,
            in_outs=in_outs,
            generation=generation,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            language=language,
            concurrent_semaphore=mp_semaphore_for_check_correctness,
        )

    return len(results)

def test_multiprocess_global_concurrency_limit_with_semaphore():
    manager = multiprocessing.Manager()
    active_calls_counter = manager.Value("i", 0)
    max_calls_tracker = manager.Value("i", 0)
    call_lock = manager.Lock()

    global_mp_semaphore = manager.Semaphore(MAX_GLOBAL_CONCURRENCY_LIMIT_TEST)

    mock_sandbox_url = "mock_url_for_concurrency_test"
    mock_generation = "pass"
    mock_memory_limit_mb = 1024
    mock_language = "python"
    mock_timeout = 5

    process_in_outs = {
        "inputs": [f"task_input_{i}" for i in range(NUM_TASKS_PER_PROCESS_TEST)],
        "outputs": [f"task_output_{i}" for i in range(NUM_TASKS_PER_PROCESS_TEST)],
    }

    futures = []
    total_tasks_expected_to_run = NUM_PROCESSES_TEST * NUM_TASKS_PER_PROCESS_TEST

    test_start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES_TEST) as executor:
        for i in range(NUM_PROCESSES_TEST):
            future = executor.submit(
                _process_pool_worker_for_concurrency_test,
                mock_sandbox_url,
                process_in_outs,
                mock_generation,
                mock_memory_limit_mb,
                mock_language,
                mock_timeout,
                global_mp_semaphore,
                active_calls_counter,
                max_calls_tracker,
                call_lock,
            )
            futures.append(future)

    num_tasks_processed_per_worker = [f.result() for f in futures]
    test_end_time = time.time()
    total_execution_time = test_end_time - test_start_time

    print("\n--- Global Concurrency Test Stats ---")
    print(f"Semaphore Limit (MAX_GLOBAL_CONCURRENCY_LIMIT_TEST): {MAX_GLOBAL_CONCURRENCY_LIMIT_TEST}")
    print(f"Number of Processes (NUM_PROCESSES_TEST): {NUM_PROCESSES_TEST}")
    print(f"Tasks per Process (NUM_TASKS_PER_PROCESS_TEST): {NUM_TASKS_PER_PROCESS_TEST}")
    print(f"Total Tasks Submitted: {total_tasks_expected_to_run}")
    print(f"Simulated API Call Duration: {SIMULATED_API_CALL_DURATION_TEST}s")
    print(f"Total Test Execution Time: {total_execution_time:.2f}s")
    print(f"Max Concurrent Mock API Calls Observed: {max_calls_tracker.value}")

    assert sum(num_tasks_processed_per_worker) == total_tasks_expected_to_run, (
        "Mismatch in the number of tasks processed."
    )

    assert max_calls_tracker.value > 0, "The mocked API call_sandbox_api was not called."

    assert max_calls_tracker.value <= MAX_GLOBAL_CONCURRENCY_LIMIT_TEST, (
        f"Observed concurrency ({max_calls_tracker.value}) exceeded semaphore limit "
        f"({MAX_GLOBAL_CONCURRENCY_LIMIT_TEST})."
    )

    min_expected_duration = (
        total_tasks_expected_to_run * SIMULATED_API_CALL_DURATION_TEST
    ) / MAX_GLOBAL_CONCURRENCY_LIMIT_TEST

    assert total_execution_time >= min_expected_duration * 0.8, (
        f"Total execution time ({total_execution_time:.2f}s) was unexpectedly short, suggesting the "
        f"semaphore might not be effectively limiting concurrency as expected "
        f"(min expected: {min_expected_duration * 0.8:.2f}s)."
    )

def test_unit_invalid_input_format():
    results, metadata_list = check_correctness(SANDBOX_URL, None, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

    results, metadata_list = check_correctness(SANDBOX_URL, {}, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_INVALID_MISSING_KEY, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_unit_input_output_mismatch():
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_MISMATCH, CODE_SUCCESS)
    assert results == [-1]
    assert len(metadata_list) == 1
    assert metadata_list[0]["error"] == "Input/output count mismatch"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_concurrency_all_timeout():
    concurrency_level = 100
    code_infinite_loop = """
def knight_moves(X, Y):
    MOD = 10**9 + 7
    dp = [[0] * (Y + 1) for _ in range(X + 1)]
    dp[0][0] = 1
    for i in range(1, X + 1):
        for j in range(1, Y + 1):
            dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % MOD
    return dp[X][Y]

def solve():
    X, Y = map(int, input().split())
    print(knight_moves(X, Y))

if __name__ == "__main__":
    solve()
    """

    timeout_inputs = ["324 384429" for i in range(concurrency_level)]
    timeout_outputs = [f"output_{i}\n" for i in range(concurrency_level)]
    timeout_in_outs = {"inputs": timeout_inputs, "outputs": timeout_outputs}

    test_timeout = 10

    start_time = time.time()
    results, metadata_list = check_correctness(SANDBOX_URL, timeout_in_outs, code_infinite_loop, timeout=test_timeout)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nHigh concurrency all timeout test ({concurrency_level} cases) duration: {duration:.2f} seconds")

    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"
    all_timed_out = all(r == -3 for r in results)
    if not all_timed_out:
        non_timeout_indices = [i for i, r in enumerate(results) if r != -3]
        print(f"Indices that did not time out: {non_timeout_indices}")

        for i in non_timeout_indices[:5]:
            print(f"Metadata for non-timeout case {i}: {metadata_list[i]}")
    assert all_timed_out, f"Not all {concurrency_level} concurrent tests resulted in timeout (-3). Results: {results}"

    assert len(metadata_list) == concurrency_level
    assert metadata_list[0]["status"] == "timeout"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_fn_name_success_single_case():
    generation_code = """
class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        positions = defaultdict(list)
        for idx, num in enumerate(nums):
            positions[num].append(idx)

        x_positions = positions[x]
        answer = []
        for k in queries:
            if k > len(x_positions):
                answer.append(-1)
            else:
                answer.append(x_positions[k-1])
        return answer
"""
    in_outs = {
        "fn_name": "occurrencesOfElement",
        "inputs": ["[1, 3, 1, 7]\n[1, 3, 2, 4]\n1", "[1, 2, 3]\n[10]\n5"],
        "outputs": ["[0, -1, 2, -1]", "[-1]"],
    }

    results, metadata_list = check_correctness(SANDBOX_URL, in_outs, generation_code, timeout=5)

    assert results == [True, True]
    assert "error" not in metadata_list[0]
    assert metadata_list[0].get("status") != "compilation error"
    assert metadata_list[0].get("status") != "runtime error"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_none_and_empty_stdin_passed_correctly():
    echo_code = """
import sys
print(f"You said '{sys.stdin.readline().strip()}'")
"""
    in_outs = {
        "inputs": [None, "", "hello"],
        "outputs": ["You said ''", "You said ''", "You said 'hello'"],
    }

    results, metadata_list = check_correctness(SANDBOX_URL, in_outs, echo_code, timeout=5)

    assert results == [True, True, True]
    assert "error" not in metadata_list[0]
    assert metadata_list[0].get("status") != "compilation error"
    assert metadata_list[0].get("status") != "runtime error"
