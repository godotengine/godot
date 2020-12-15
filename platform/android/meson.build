# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_android_logo = custom_target(
    'android_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'android'],
    build_by_default: true
)
SOURCES += [_target_android_logo]

_target_android_run_icon = custom_target(
    'android_run_icon',
    output: ['run_icon.gen.h'],
    input: ['run_icon.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'android'],
    build_by_default: true
)

if PLATFORM != 'android'
    subdir_done()
endif

#add_languages('java')
#
#jar('godot',
#  'java/lib/src/org/godotengine/godot/Dictionary.java',
#  'java/lib/src/org/godotengine/godot/FullScreenGodotApp.java',
#  'java/lib/src/org/godotengine/godot/GodotDownloaderAlarmReceiver.java',
#  'java/lib/src/org/godotengine/godot/GodotDownloaderService.java',
#  'java/lib/src/org/godotengine/godot/GodotGLRenderView.java',
#  'java/lib/src/org/godotengine/godot/GodotHost.java',
#  'java/lib/src/org/godotengine/godot/GodotInstrumentation.java',
#  'java/lib/src/org/godotengine/godot/GodotIO.java',
#  'java/lib/src/org/godotengine/godot/Godot.java',
#  'java/lib/src/org/godotengine/godot/GodotLib.java',
#  'java/lib/src/org/godotengine/godot/GodotRenderer.java',
#  'java/lib/src/org/godotengine/godot/GodotRenderView.java',
#  'java/lib/src/org/godotengine/godot/GodotVulkanRenderView.java',
#  'java/lib/src/org/godotengine/godot/input/GodotEditText.java',
#  'java/lib/src/org/godotengine/godot/input/GodotGestureHandler.java',
#  'java/lib/src/org/godotengine/godot/input/GodotInputHandler.java',
#  'java/lib/src/org/godotengine/godot/input/GodotTextInputWrapper.java',
#  'java/lib/src/org/godotengine/godot/input/InputManagerCompat.java',
#  'java/lib/src/org/godotengine/godot/input/InputManagerV16.java',
#  'java/lib/src/org/godotengine/godot/input/Joystick.java',
#  'java/lib/src/org/godotengine/godot/plugin/GodotPluginInfoProvider.java',
#  'java/lib/src/org/godotengine/godot/plugin/GodotPlugin.java',
#  'java/lib/src/org/godotengine/godot/plugin/GodotPluginRegistry.java',
#  'java/lib/src/org/godotengine/godot/plugin/SignalInfo.java',
#  'java/lib/src/org/godotengine/godot/plugin/UsedByGodot.java',
#  'java/lib/src/org/godotengine/godot/utils/Crypt.java',
#  'java/lib/src/org/godotengine/godot/utils/GLUtils.java',
#  'java/lib/src/org/godotengine/godot/utils/GodotNetUtils.java',
#  'java/lib/src/org/godotengine/godot/utils/PermissionsUtil.java',
#  'java/lib/src/org/godotengine/godot/xr/ovr/OvrConfigChooser.java',
#  'java/lib/src/org/godotengine/godot/xr/ovr/OvrContextFactory.java',
#  'java/lib/src/org/godotengine/godot/xr/ovr/OvrWindowSurfaceFactory.java',
#  'java/lib/src/org/godotengine/godot/xr/regular/RegularConfigChooser.java',
#  'java/lib/src/org/godotengine/godot/xr/regular/RegularContextFactory.java',
#  'java/lib/src/org/godotengine/godot/xr/regular/RegularFallbackConfigChooser.java',
#  'java/lib/src/org/godotengine/godot/xr/XRMode.java',
#)

################################################################################
# Host Machine Specifics
################################################################################
android_sources = files(
    'android_keys_utils.cpp',
    'api/api.cpp',
    'audio_driver_opensl.cpp',
    'dir_access_jandroid.cpp',
    'display_server_android.cpp',
    'file_access_android.cpp',
    'java_class_wrapper.cpp',
    'java_godot_io_wrapper.cpp',
    'java_godot_lib_jni.cpp',
    'java_godot_view_wrapper.cpp',
    'java_godot_wrapper.cpp',
    'jni_utils.cpp',
    'net_socket_android.cpp',
    'os_android.cpp',
    #'plugin/godot_plugin_jni.cpp',
    'thread_jandroid.cpp',
    'vulkan/vulkan_context_android.cpp',
)

SOURCES += [_target_android_run_icon, android_sources]
CPP_ARGS += ['-DANDROID_ENABLED', '-DNO_STATVFS', '-DUNIX_ENABLED']

ANDROID_NDK_DEPS = [cc.find_library('android'),
                    cc.find_library('OpenSLES'),
                    cc.find_library('log'),
                    ]
