"""Mock SCons.Variables.BoolVariable module"""

def _text2bool(val):
    """Convert text string to boolean value"""
    val = val.lower()
    if val in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    elif val in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise ValueError(f"Invalid boolean value: {val}")

def BoolVariable(key, help, default):
    """Create a boolean SCons variable"""
    from interpreter.variables import BoolVariable as InterpreterBoolVariable
    return InterpreterBoolVariable(key, help, default)