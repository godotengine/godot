module_gdnative_srcs += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

if 'webrtc' in MODULES_ENABLED
    module_gdnative_cpp_args += ['-DWEBRTC_GDNATIVE_ENABLED']
endif
