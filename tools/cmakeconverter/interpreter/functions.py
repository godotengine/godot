"""Additional SCons functions"""

def Glob(pattern):
    """Glob files"""
    print(f"DEBUG: Glob called with pattern={pattern}")
    import glob
    import os
    if pattern.startswith('#'):
        pattern = pattern[1:]  # Remove the # prefix
    if not pattern.startswith('/'):
        pattern = os.path.join(os.getcwd(), pattern)
    result = glob.glob(pattern)
    print(f"DEBUG: Glob result: {result}")
    return result

def Value(value):
    """Create a Value node"""
    print(f"DEBUG: Value called with value={value}")
    return value

def Run(command):
    """Run a command"""
    print(f"DEBUG: Run called with command={command}")
    return command

def CommandNoCache(target, source, action):
    """Run a command without caching"""
    print(f"DEBUG: CommandNoCache called with target={target}, source={source}, action={action}")
    return None