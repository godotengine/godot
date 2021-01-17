def get_name() -> str:
    return 'etc'

def can_build(env: dict) -> bool:
    return env.tools_enabled
