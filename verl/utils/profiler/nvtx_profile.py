

import functools
from contextlib import contextmanager
from typing import Callable, Optional

import nvtx
import torch

from .profile import DistProfiler, ProfilerConfig

def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    return nvtx.start_range(message=message, color=color, domain=domain, category=category)

def mark_end_range(range_id: str) -> None:
    return nvtx.end_range(range_id)

def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:

    def decorator(func):
        profile_message = message or func.__name__
        return nvtx.annotate(profile_message, color=color, domain=domain, category=category)(func)

    return decorator

@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    mark_range = mark_start_range(message=name, color=color, domain=domain, category=category)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)

class NsightSystemsProfiler(DistProfiler):

    def __init__(self, rank: int, config: Optional[ProfilerConfig], **kwargs):

        if not config:
            config = ProfilerConfig(ranks=[])
        self.this_step: bool = False
        self.discrete: bool = config.discrete
        self.this_rank: bool = False
        if config.all_ranks:
            self.this_rank = True
        elif config.ranks:
            self.this_rank = rank in config.ranks

    def start(self, **kwargs):
        if self.this_rank:
            self.this_step = True
            if not self.discrete:
                torch.cuda.profiler.start()

    def stop(self):
        if self.this_rank:
            self.this_step = False
            if not self.discrete:
                torch.cuda.profiler.stop()

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> Callable:

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                profile_name = message or func.__name__

                if self.profiler.this_step:
                    if self.profiler.discrete:
                        torch.cuda.profiler.start()
                    mark_range = mark_start_range(message=profile_name, color=color, domain=domain, category=category)

                result = func(self, *args, **kwargs)

                if self.profiler.this_step:
                    mark_end_range(mark_range)
                    if self.profiler.discrete:
                        torch.cuda.profiler.stop()

                return result

            return wrapper

        return decorator
