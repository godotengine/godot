DEPENDENCIES += [DEP_VULKAN, DEP_SPIRV_REFLECT]

SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n')) + files('../../thirdparty/volk/volk.c')

if PLATFORM == 'android'
   CPP_ARGS += '-DVK_USE_PLATFORM_ANDROID_KHR'
elif PLATFORM == 'iphone'
    CPP_ARGS += '-DVK_USE_PLATFORM_IOS_MVK'
elif PLATFORM == 'linuxbsd'
    CPP_ARGS += '-DVK_USE_PLATFORM_XLIB_KHR'
elif PLATFORM == 'osx'
    CPP_ARGS += '-DVK_USE_PLATFORM_MACOS_MVK'
elif PLATFORM == 'windows'
    CPP_ARGS += '-DVK_USE_PLATFORM_WIN32_KHR'
endif
