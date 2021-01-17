def get_name() -> str:
    return 'inappstore'
    
def can_build(env: dict) -> bool:
    return env.platform == "iphone"
