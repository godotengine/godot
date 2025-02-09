"""Mock core_builders module"""

def add_module_version_string(env):
    """Add module version string to environment"""
    env.module_version_string = ""

def get_version_info(version_string):
    """Get version info from version string"""
    return {
        'major': 4,
        'minor': 2,
        'patch': 0,
        'status': 'stable',
        'build': '',
        'year': 2024,
        'website': 'https://godotengine.org',
        'docs': 'https://docs.godotengine.org',
        'string': version_string,
    }

def Run(env, commands):
    """Run commands"""
    pass

def add_source_files(env, sources, files, warn_duplicates=True):
    """Add source files to list"""
    if isinstance(files, str):
        files = [files]
    sources.extend(files)

def add_library(env, name, sources, **args):
    """Add library target"""
    pass

def add_program(env, name, sources, **args):
    """Add program target"""
    pass

def add_shared_library(env, name, sources, **args):
    """Add shared library target"""
    pass

def add_static_library(env, name, sources, **args):
    """Add static library target"""
    pass

def add_thirdparty_library(env, name, thirdparty_dir, thirdparty_sources, thirdparty_env, **args):
    """Add third-party library target"""
    pass

def add_thirdparty_sources(env, thirdparty_dir, thirdparty_sources, thirdparty_env, **args):
    """Add third-party sources"""
    pass

def detect_modules():
    """Detect available modules"""
    return []

def get_doc_classes():
    """Get documentation classes"""
    return []

def get_doc_path():
    """Get documentation path"""
    return ""

def get_doc_data():
    """Get documentation data"""
    return {}