oxr_sources = files(run_command(GLOB, 'src/*.c', check: true).stdout().split('\n'))
oxr_sources += files(run_command(GLOB, 'src/common/*.cpp', check: true).stdout().split('\n'))
oxr_sources += files(
    'src/loader/android_utilities.cpp',
    'src/loader/api_layer_interface.cpp',
    'src/loader/loader_core.cpp',
    'src/loader/loader_instance.cpp',
    'src/loader/loader_logger.cpp',
    'src/loader/loader_logger_recorders.cpp',
    'src/loader/manifest_file.cpp',
    'src/loader/runtime_interface.cpp',
    'src/loader/xr_generated_loader.cpp',
    'src/external/jsoncpp/src/lib_json/json_reader.cpp',
    'src/external/jsoncpp/src/lib_json/json_value.cpp',
    'src/external/jsoncpp/src/lib_json/json_writer.cpp',
)

oxr_args = []
if host_machine.system() == 'linux'
  oxr_args += ['-DXR_OS_LINUX', '-DHAVE_SECURE_GETENV']
elif host_machine.system() == 'darwin'
  oxr_args += '-DXR_OS_APPLE'
elif host_machine.system() == 'windows'
  oxr_args += ['-DXR_OS_WINDOWS', '-DXR_USE_PLATFORM_WIN32']
else
  error('Unsupported platform.')
endif

DEP_OPENXR = declare_dependency(
    link_with: library('openxr',
        oxr_sources,
        cpp_args: oxr_args,
        include_directories: ['include', 'src', 'src/common', 'src/external/jsoncpp/include'],
        dependencies: DEP_VULKAN),
    include_directories: 'include'

)
