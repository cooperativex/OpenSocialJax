from opensocialjax.environments import OpenCleanup, OpenHarvest

REGISTERED_ENVS = [
    "open_cleanup",   # OpenCleanup: Clean Up with a hidden crafting rule
    "open_harvest",   # OpenHarvest: Commons Harvest with hidden pay / regrowth rules
]

ENVS = {"open_cleanup": OpenCleanup, "open_harvest": OpenHarvest}


def make(env_id: str, **env_kwargs):
    """A JAX version of OpenAI's env.make(env_name), built off Gymnax."""
    if env_id not in ENVS:
        raise ValueError(f"{env_id} is not in registered OpenSocialJax environments")
    return ENVS[env_id](**env_kwargs)
