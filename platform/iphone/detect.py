import os
import sys


def is_active():
    return True


def get_name():
    return "iOS"


def can_build():

    import sys
    import os
    if sys.platform == 'darwin' or os.environ.has_key("OSXCROSS_IOS"):
        return True

    return False


def get_opts():

    return [
        ('IPHONEPLATFORM', 'name of the iphone platform', 'iPhoneOS'),
        ('IPHONEPATH', 'the path to iphone toolchain', '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain'),
        ('IPHONESDK', 'path to the iphone SDK', '/Applications/Xcode.app/Contents/Developer/Platforms/${IPHONEPLATFORM}.platform/Developer/SDKs/${IPHONEPLATFORM}.sdk/'),
        ('game_center', 'Support for game center', 'yes'),
        ('store_kit', 'Support for in-app store', 'yes'),
        ('icloud', 'Support for iCloud', 'yes'),
        ('ios_gles22_override', 'Force GLES2.0 on iOS', 'yes'),
        ('ios_exceptions', 'Use exceptions when compiling on playbook', 'yes'),
        ('ios_triple', 'Triple for ios toolchain', ''),
        ('ios_sim', 'Build simulator binary', 'no'),
    ]


def get_flags():

    return [
        ('tools', 'no'),
    ]


def configure(env):

    env.Append(CPPPATH=['#platform/iphone'])

    env['ENV']['PATH'] = env['IPHONEPATH'] + "/Developer/usr/bin/:" + env['ENV']['PATH']

    env['CC'] = '$IPHONEPATH/usr/bin/${ios_triple}clang'
    env['CXX'] = '$IPHONEPATH/usr/bin/${ios_triple}clang++'
    env['AR'] = '$IPHONEPATH/usr/bin/${ios_triple}ar'
    env['RANLIB'] = '$IPHONEPATH/usr/bin/${ios_triple}ranlib'

    import string
    if (env["ios_sim"] == "yes" or env["arch"] == "x86"):  # i386, simulator
        env["arch"] = "x86"
        env["bits"] = "32"
        env['CCFLAGS'] = string.split('-arch i386 -fobjc-abi-version=2 -fobjc-legacy-dispatch -fmessage-length=0 -fpascal-strings -fasm-blocks -D__IPHONE_OS_VERSION_MIN_REQUIRED=40100 -isysroot $IPHONESDK -mios-simulator-version-min=4.3 -DCUSTOM_MATRIX_TRANSFORM_H=\\\"build/iphone/matrix4_iphone.h\\\" -DCUSTOM_VECTOR3_TRANSFORM_H=\\\"build/iphone/vector3_iphone.h\\\"')
    elif (env["arch"] == "arm64"):  # arm64
        env["bits"] = "64"
        env['CCFLAGS'] = string.split('-fno-objc-arc -arch arm64 -fmessage-length=0 -fno-strict-aliasing -fdiagnostics-print-source-range-info -fdiagnostics-show-category=id -fdiagnostics-parseable-fixits -fpascal-strings -fvisibility=hidden -MMD -MT dependencies -miphoneos-version-min=7.0 -isysroot $IPHONESDK')
        env.Append(CPPFLAGS=['-DNEED_LONG_INT'])
        env.Append(CPPFLAGS=['-DLIBYUV_DISABLE_NEON'])
    else:  # armv7
        env["arch"] = "arm"
        env["bits"] = "32"
        env['CCFLAGS'] = string.split('-fno-objc-arc -arch armv7 -fmessage-length=0 -fno-strict-aliasing -fdiagnostics-print-source-range-info -fdiagnostics-show-category=id -fdiagnostics-parseable-fixits -fpascal-strings -isysroot $IPHONESDK -fvisibility=hidden -mthumb "-DIBOutlet=__attribute__((iboutlet))" "-DIBOutletCollection(ClassName)=__attribute__((iboutletcollection(ClassName)))" "-DIBAction=void)__attribute__((ibaction)" -miphoneos-version-min=7.0 -MMD -MT dependencies -isysroot $IPHONESDK')

    if (env["arch"] == "x86"):
        env['IPHONEPLATFORM'] = 'iPhoneSimulator'
        env.Append(LINKFLAGS=['-arch', 'i386', '-mios-simulator-version-min=4.3',
                              '-isysroot', '$IPHONESDK',
                              #'-mmacosx-version-min=10.6',
                              '-Xlinker',
                              '-objc_abi_version',
                              '-Xlinker', '2',
                              '-framework', 'AudioToolbox',
                              '-framework', 'AVFoundation',
                              '-framework', 'CoreAudio',
                              '-framework', 'CoreGraphics',
                              '-framework', 'CoreMedia',
                              '-framework', 'CoreMotion',
                              '-framework', 'Foundation',
                              '-framework', 'Security',
                              '-framework', 'UIKit',
                              '-framework', 'MediaPlayer',
                              '-framework', 'OpenGLES',
                              '-framework', 'QuartzCore',
                              '-framework', 'SystemConfiguration',
                              '-F$IPHONESDK',
                              ])
    elif (env["arch"] == "arm64"):
        env.Append(LINKFLAGS=['-arch', 'arm64', '-Wl,-dead_strip', '-miphoneos-version-min=7.0',
                                                '-isysroot', '$IPHONESDK',
                                                #'-stdlib=libc++',
                                                '-framework', 'Foundation',
                                                '-framework', 'UIKit',
                                                '-framework', 'CoreGraphics',
                                                '-framework', 'OpenGLES',
                                                '-framework', 'QuartzCore',
                                                '-framework', 'CoreAudio',
                                                '-framework', 'AudioToolbox',
                                                '-framework', 'SystemConfiguration',
                                                '-framework', 'Security',
                                                #'-framework', 'AdSupport',
                                                '-framework', 'MediaPlayer',
                                                '-framework', 'AVFoundation',
                                                '-framework', 'CoreMedia',
                                                '-framework', 'CoreMotion',
                              ])
    else:
        env.Append(LINKFLAGS=['-arch', 'armv7', '-Wl,-dead_strip', '-miphoneos-version-min=7.0',
                                                '-isysroot', '$IPHONESDK',
                                                '-framework', 'Foundation',
                                                '-framework', 'UIKit',
                                                '-framework', 'CoreGraphics',
                                                '-framework', 'OpenGLES',
                                                '-framework', 'QuartzCore',
                                                '-framework', 'CoreAudio',
                                                '-framework', 'AudioToolbox',
                                                '-framework', 'SystemConfiguration',
                                                '-framework', 'Security',
                                                #'-framework', 'AdSupport',
                                                '-framework', 'MediaPlayer',
                                                '-framework', 'AVFoundation',
                                                '-framework', 'CoreMedia',
                                                '-framework', 'CoreMotion',
                              ])

    if env['game_center'] == 'yes':
        env.Append(CPPFLAGS=['-fblocks', '-DGAME_CENTER_ENABLED'])
        env.Append(LINKFLAGS=['-framework', 'GameKit'])

    if env['store_kit'] == 'yes':
        env.Append(CPPFLAGS=['-DSTOREKIT_ENABLED'])
        env.Append(LINKFLAGS=['-framework', 'StoreKit'])

    if env['icloud'] == 'yes':
        env.Append(CPPFLAGS=['-DICLOUD_ENABLED'])

    env.Append(CPPPATH=['$IPHONESDK/usr/include', '$IPHONESDK/System/Library/Frameworks/OpenGLES.framework/Headers', '$IPHONESDK/System/Library/Frameworks/AudioUnit.framework/Headers'])

    if (env["target"] == "release"):

        env.Append(CCFLAGS=['-O3', '-DNS_BLOCK_ASSERTIONS=1', '-gdwarf-2'])  # removed -ffast-math
        env.Append(LINKFLAGS=['-O3'])

    elif env["target"] == "release_debug":
        env.Append(CCFLAGS=['-Os', '-DNS_BLOCK_ASSERTIONS=1', '-DDEBUG_ENABLED'])
        env.Append(LINKFLAGS=['-Os'])
        env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ENABLED'])

    elif (env["target"] == "debug"):

        env.Append(CCFLAGS=['-D_DEBUG', '-DDEBUG=1', '-gdwarf-2', '-O0', '-DDEBUG_ENABLED'])
        env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ENABLED'])

    elif (env["target"] == "profile"):

        env.Append(CCFLAGS=['-g', '-pg', '-Os'])
        env.Append(LINKFLAGS=['-pg'])

    if (env["ios_sim"] == "yes"):  # TODO: Check if needed?
        env['ENV']['MACOSX_DEPLOYMENT_TARGET'] = '10.6'
    env['ENV']['CODESIGN_ALLOCATE'] = '/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/codesign_allocate'
    env.Append(CPPFLAGS=['-DIPHONE_ENABLED', '-DUNIX_ENABLED', '-DGLES2_ENABLED', '-DMPC_FIXED_POINT'])

    # TODO: Move that to opus module's config
    if("module_opus_enabled" in env and env["module_opus_enabled"] != "no"):
        env.opus_fixed_point = "yes"
        if env["arch"] == "x86":
            pass
        elif(env["arch"] == "arm64"):
            env.Append(CFLAGS=["-DOPUS_ARM64_OPT"])
        else:
            env.Append(CFLAGS=["-DOPUS_ARM_OPT"])

    if env['ios_exceptions'] == 'yes':
        env.Append(CPPFLAGS=['-fexceptions'])
    else:
        env.Append(CPPFLAGS=['-fno-exceptions'])
    # env['neon_enabled']=True
    env['S_compiler'] = '$IPHONEPATH/Developer/usr/bin/gcc'

    import methods
    env.Append(BUILDERS={'GLSL120': env.Builder(action=methods.build_legacygl_headers, suffix='glsl.h', src_suffix='.glsl')})
    env.Append(BUILDERS={'GLSL': env.Builder(action=methods.build_glsl_headers, suffix='glsl.h', src_suffix='.glsl')})
    env.Append(BUILDERS={'GLSL120GLES': env.Builder(action=methods.build_gles2_headers, suffix='glsl.h', src_suffix='.glsl')})
