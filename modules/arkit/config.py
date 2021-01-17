def get_name() -> str:
    return 'arkit'

def can_build(env: dict) -> bool:
    return env.platform == "iphone"
