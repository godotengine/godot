"""Mock SCons module for the interpreter"""

__version__ = "4.0.0"

class Node:
    def __init__(self, path):
        self.path = path
        
    def abspath(self):
        return self.path
        
    def __str__(self):
        return self.path

class Action:
    def __init__(self, action, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs

class Builder:
    def __init__(self, action=None, suffix=None, src_suffix=None):
        self.action = action
        self.suffix = suffix
        self.src_suffix = src_suffix

def Scanner(*args, **kwargs):
    return None