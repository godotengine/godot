"""Mock SCons.Util package"""
import os
import sys
import shutil

def WhereIs(program, path=None, pathext=None, reject=[]):
    """Find an executable in the system path"""
    if path is None:
        path = os.environ.get('PATH', '').split(os.pathsep)
    
    if pathext is None:
        pathext = ['']
        if sys.platform == 'win32':
            pathext.extend(os.environ.get('PATHEXT', '').split(os.pathsep))
    
    for dir in path:
        for ext in pathext:
            full_path = os.path.join(dir, program + ext)
            if os.path.isfile(full_path) and full_path not in reject:
                return full_path
    
    # Fallback to shutil.which
    return shutil.which(program)

def is_String(obj):
    """Check if object is a string"""
    return isinstance(obj, str)

def is_List(obj):
    """Check if object is a list"""
    return isinstance(obj, list)

def is_Dict(obj):
    """Check if object is a dictionary"""
    return isinstance(obj, dict)

def flatten(sequence):
    """Flatten a sequence"""
    result = []
    for item in sequence:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def to_String(obj):
    """Convert object to string"""
    return str(obj)

def to_List(obj):
    """Convert object to list"""
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]

def to_Dict(obj):
    """Convert object to dictionary"""
    if isinstance(obj, dict):
        return obj
    else:
        raise TypeError(f"Cannot convert {type(obj)} to dict")