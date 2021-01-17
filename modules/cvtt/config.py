def get_name() -> str:
    return 'cvtt'

def can_build(env: dict) -> bool:    
    return env.tools_enabled
