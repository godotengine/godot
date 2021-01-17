def get_name() -> str:
    return 'icloud'
    
def can_build(env: dict) -> bool:
    return env.platform == "iphone"
