def can_build(env: dict) -> bool:
    return env.tools_enabled and env.platform not in ["android", "ios"]
