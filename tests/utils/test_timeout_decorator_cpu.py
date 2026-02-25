

import multiprocessing
import sys
import threading
import time

import pytest

from verl.utils.py_functional import timeout_limit as timeout

TEST_TIMEOUT_SECONDS = 1.5
LONG_TASK_DURATION = TEST_TIMEOUT_SECONDS + 0.5

@timeout(seconds=TEST_TIMEOUT_SECONDS)
def quick_task(x):
    time.sleep(0.1)
    return "quick_ok"

@timeout(seconds=TEST_TIMEOUT_SECONDS)
def slow_task(x):
    time.sleep(LONG_TASK_DURATION)
    return "slow_finished"

def task_raises_value_error():
    raise ValueError("Specific value error from task")

@timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
def top_level_decorated_quick_task_signal():

    time.sleep(0.1)
    return "quick_ok_signal_subprocess"

@timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
def top_level_decorated_slow_task_signal():
    time.sleep(LONG_TASK_DURATION)
    return "slow_finished"

def run_target_and_put_in_queue(target_func, q):
    try:
        result = target_func()
        q.put(("success", result))
    except Exception as e:
        q.put(("error", e))

@pytest.fixture(scope="module", autouse=True)
def set_macos_start_method():
    if sys.platform == "darwin":

        current_method = multiprocessing.get_start_method(allow_none=True)

        if current_method is None or current_method != "fork":
            try:
                multiprocessing.set_start_method("fork", force=True)
            except RuntimeError:

                pass

def test_quick_task():

    result = quick_task(1)
    assert result == "quick_ok"

def test_slow_task_timeout():

    with pytest.raises(TimeoutError) as excinfo:
        slow_task(1)

    assert f"timed out after {TEST_TIMEOUT_SECONDS} seconds" in str(excinfo.value)

def test_internal_exception():

    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS)(task_raises_value_error)
    with pytest.raises(ValueError) as excinfo:
        decorated_task()
    assert str(excinfo.value) == "Specific value error from task"

def test_signal_quick_task_main_process():

    def plain_quick_task_logic():
        time.sleep(0.1)
        return "quick_ok_signal"

    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)(plain_quick_task_logic)
    assert decorated_task() == "quick_ok_signal"

def test_signal_slow_task_main_process_timeout():

    def plain_slow_task_logic():
        time.sleep(LONG_TASK_DURATION)
        return "slow_finished_signal"

    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)(plain_slow_task_logic)
    with pytest.raises(TimeoutError) as excinfo:
        decorated_task()

    assert f"timed out after {TEST_TIMEOUT_SECONDS} seconds" in str(excinfo.value)

@pytest.mark.skip(reason="this test won't pass. Just to show why use_signals should not be used")
def test_signal_in_thread_does_not_timeout():
    result_container = []
    exception_container = []

    @timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
    def slow_task_in_thread():
        try:
            print("Thread: Starting slow task...")
            time.sleep(LONG_TASK_DURATION)
            print("Thread: Slow task finished.")
            return "slow_finished_in_thread"
        except Exception as e:

            print(f"Thread: Caught exception: {e}")
            exception_container.append(e)
            return None

    def thread_target():
        try:

            res = slow_task_in_thread()
            if res is not None:
                result_container.append(res)
        except Exception as e:

            print(f"Thread Target: Caught exception: {e}")
            exception_container.append(e)

    thread = threading.Thread(target=thread_target)
    print("Main: Starting thread...")
    thread.start()

    thread.join(timeout=LONG_TASK_DURATION + 1)

    assert len(exception_container) == 1
    assert isinstance(exception_container[0], TimeoutError)
    assert not result_container

def test_in_thread_timeout():
    result_container = []
    exception_container = []

    @timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=False)
    def slow_task_in_thread():
        try:
            print("Thread: Starting slow task...")
            time.sleep(LONG_TASK_DURATION)
            print("Thread: Slow task finished.")
            return "slow_finished_in_thread"
        except Exception as e:

            print(f"Thread: Caught exception: {e}")
            exception_container.append(e)
            return None

    def thread_target():
        try:

            res = slow_task_in_thread()
            if res is not None:
                result_container.append(res)
        except Exception as e:

            print(f"Thread Target: Caught exception: {e}")
            exception_container.append(e)

    thread = threading.Thread(target=thread_target)
    print("Main: Starting thread...")
    thread.start()

    thread.join(timeout=LONG_TASK_DURATION + 1)

    assert len(exception_container) == 1
    assert isinstance(exception_container[0], TimeoutError)
    assert not result_container
