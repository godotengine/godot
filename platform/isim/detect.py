import os
import sys


def is_active():
	return False
	
def get_name():
	return "iSIM"
	

def can_build():

	import sys
	if sys.platform == 'darwin':
		return True

	return False

def get_opts():

	return [
		('ISIMPLATFORM', 'name of the iphone platform', 'iPhoneSimulator'),
		('ISIMPATH', 'the path to iphone toolchain', '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain'),
		('ISIMSDK', 'path to the iphone SDK', '/Applications/Xcode.app/Contents/Developer/Platforms/${ISIMPLATFORM}.platform/Developer/SDKs/${ISIMPLATFORM}.sdk'),
		('game_center', 'Support for game center', 'yes'),
		('store_kit', 'Support for in-app store', 'yes'),
		('ios_gles22_override', 'Force GLES2.0 on iOS', 'yes'),
		('ios_GLES1_override', 'Force legacy GLES (1.1) on iOS', 'no'),
		('ios_appirater', 'Enable Appirater', 'no'),
		('ios_exceptions', 'Use exceptions when compiling on playbook', 'no'),
	]

def get_flags():

	return [
		('tools', 'yes'),
		('webp', 'yes'),
	]



def configure(env):

	env.Append(CPPPATH=['#platform/iphone'])

	env['ENV']['PATH'] = env['ISIMPATH']+"/Developer/usr/bin/:"+env['ENV']['PATH']

	env['CC'] = '$ISIMPATH/usr/bin/${ios_triple}clang'
	env['CXX'] = '$ISIMPATH/usr/bin/${ios_triple}clang++'
	env['AR'] = '$ISIMPATH/usr/bin/${ios_triple}ar'
	env['RANLIB'] = '$ISIMPATH/usr/bin/${ios_triple}ranlib'

	import string
	env['CCFLAGS'] = string.split('-arch i386 -fobjc-abi-version=2 -fobjc-legacy-dispatch -fmessage-length=0 -fpascal-strings -fasm-blocks  -Wall -D__IPHONE_OS_VERSION_MIN_REQUIRED=40100 -isysroot $ISIMSDK -mios-simulator-version-min=4.3 -DCUSTOM_MATRIX_TRANSFORM_H=\\\"build/iphone/matrix4_iphone.h\\\" -DCUSTOM_VECTOR3_TRANSFORM_H=\\\"build/iphone/vector3_iphone.h\\\"')

	env.Append(LINKFLAGS=['-arch', 'i386',
							'-mios-simulator-version-min=4.3',
							'-isysroot', '$ISIMSDK',
							#'-mmacosx-version-min=10.6',
							'-Xlinker',
							'-objc_abi_version',
							'-Xlinker', '2',
							'-framework', 'AudioToolbox',
							'-framework', 'AVFoundation',
							'-framework', 'CoreAudio',
							'-framework', 'CoreGraphics',
							'-framework', 'CoreMedia',
							'-framework', 'Foundation',
							'-framework', 'Security',
							'-framework', 'UIKit',
							'-framework', 'MediaPlayer',
							'-framework', 'OpenGLES',
							'-framework', 'QuartzCore',
							'-framework', 'SystemConfiguration',
							'-F$ISIMSDK',
							])

	env.Append(CPPPATH = ['$ISIMSDK/System/Library/Frameworks/OpenGLES.framework/Headers'])

	if (env["target"]=="release"):

		env.Append(CCFLAGS=['-O3', '-ffast-math'])
		env.Append(LINKFLAGS=['-O3', '-ffast-math'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['-DDEBUG', '-D_DEBUG', '-gdwarf-2', '-Wall', '-O0', '-DDEBUG_ENABLED'])
		env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC'])

	elif (env["target"]=="profile"):

		env.Append(CCFLAGS=['-g','-pg'])
		env.Append(LINKFLAGS=['-pg'])


	env['ENV']['MACOSX_DEPLOYMENT_TARGET'] = '10.6'
	env['ENV']['CODESIGN_ALLOCATE'] = '/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/codesign_allocate'
	env.Append(CPPFLAGS=['-DIPHONE_ENABLED', '-DUNIX_ENABLED', '-DGLES2_ENABLED', '-fexceptions'])

	import methods
	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )

