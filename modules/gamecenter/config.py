def get_name() -> str:
    return 'gamecenter'

def can_build(env: dict) -> bool:
    return env.platform == "iphone"
