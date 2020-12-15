################################################################################
# Tools Scripts
################################################################################
# TODO part in editor/SCsub, part in editor/editor_builders.py
SCRIPT_DOCS = find_program('docs.py')
# In core/SCsub
SCRIPT_ENCRYPTION_KEY = find_program('encryption_key.py')
# Those 2 are heavily customized for meson, this is fine.
SCRIPT_MODULE_DB = find_program('module_db.py')
SCRIPT_MODULES_GEN = find_program('modules_gen.py')
# TODO use methods.save_active_platforms
SCRIPT_PLATFORM_LOGO = find_program('platform_logo.py')
# In editor/SCsub
SCRIPT_REGISTER_EXPORTERS = find_program('register_exporters.py')
# In platforms/SCsub
SCRIPT_REGISTER_PLATFORM_APIS = find_program('register_platform_apis.py')
# TODO use methods.update_version
SCRIPT_VERSION = find_program('version.py')

SCRIPT_GLSL_BUILDERS = find_program(MESON_SOURCE_ROOT / 'glsl_builders.py').full_path()
SCRIPT_DEFAULT_THEME_BUILDERS = find_program(
    MESON_SOURCE_ROOT / 'scene' / 'resources' / 'default_theme' / 'default_theme_builders.py'
).full_path()
SCRIPT_INPUT_BUILDERS = find_program(MESON_SOURCE_ROOT / 'core' / 'input' / 'input_builders.py').full_path()
SCRIPT_EDITOR_BUILDERS = find_program(MESON_SOURCE_ROOT / 'editor' / 'editor_builders.py').full_path()
SCRIPT_EDITOR_ICON_BUILDERS = find_program(
    MESON_SOURCE_ROOT / 'editor' / 'icons' / 'editor_icons_builders.py'
).full_path()
SCRIPT_CORE_BUILDERS = find_program(MESON_SOURCE_ROOT / 'core' / 'core_builders.py').full_path()
SCRIPT_MAIN_BUILDERS = find_program(MESON_SOURCE_ROOT / 'main' / 'main_builders.py').full_path()
SCRIPT_VIRTUALS = find_program(MESON_SOURCE_ROOT / 'core' / 'object' / 'make_virtuals.py').full_path()
SCRIPT_COMPAT = find_program('scons_compat.py')
