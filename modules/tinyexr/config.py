def get_name() -> str:
    return 'tinyexr'
    
def can_build(env: dict) -> bool:
    return env.tools_enabled
