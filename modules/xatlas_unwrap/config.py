def get_name() -> str:
    return 'xatlas_unwrap'
    
def can_build(env: dict) -> bool:
    return env.tools_enabled and env.platform not in ["android", "ios"]
