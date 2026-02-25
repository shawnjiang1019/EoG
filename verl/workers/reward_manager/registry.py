

__all__ = ["register", "get_reward_manager_cls"]

REWARD_MANAGER_REGISTRY = {}

def register(name):

    def decorator(cls):
        if name in REWARD_MANAGER_REGISTRY and REWARD_MANAGER_REGISTRY[name] != cls:
            raise ValueError(
                f"Reward manager {name} has already been registered: {REWARD_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator

def get_reward_manager_cls(name):
    if name not in REWARD_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER_REGISTRY[name]
