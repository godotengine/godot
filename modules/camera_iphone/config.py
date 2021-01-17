def get_name() -> str:
    return 'camera_iphone'

def can_build(env: dict) -> bool:
    return env.platform == "iphone"
