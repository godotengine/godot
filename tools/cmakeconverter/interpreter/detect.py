"""Mock detect module for platform detection"""

def get_name():
    """Get platform name"""
    return "LinuxBSD"

def can_build():
    """Check if platform can be built"""
    return True

def get_opts():
    """Get platform options"""
    return []

def get_flags():
    """Get platform flags"""
    return []

def get_doc_classes():
    """Get platform documentation classes"""
    return []

def get_doc_path():
    """Get platform documentation path"""
    return ""