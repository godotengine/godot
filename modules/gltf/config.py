def get_name() -> str:
    return 'gltf'
    
def can_build(env: dict) -> bool:
    return env.tools_enabled
