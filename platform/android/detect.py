import os
import sys
import string
import platform

def is_active():
	return True

def get_name():
	return "Android"

def can_build():

        import os
        if (not os.environ.has_key("ANDROID_NDK_ROOT")):
        	return False

	return True

def get_opts():

	return [
			('ANDROID_NDK_ROOT', 'the path to Android NDK', os.environ.get("ANDROID_NDK_ROOT", 0)),
			('NDK_TARGET', 'toolchain to use for the NDK',os.environ.get("NDK_TARGET", "arm-linux-androideabi-4.9")),
			('NDK_TARGET_X86', 'toolchain to use for the NDK x86',os.environ.get("NDK_TARGET_X86", "x86-4.9")),
			('ndk_platform', 'compile for platform: (android-<api> , example: android-14)',"android-14"),
			('android_arch', 'select compiler architecture: (armv7/armv6/x86)',"armv7"),
			('android_neon','enable neon (armv7 only)',"yes"),
			('android_stl','enable STL support in android port (for modules)',"no")
	]


def get_flags():

	return [
		('tools', 'no'),
		('openssl', 'builtin'), #use builtin openssl
	]


def create(env):
	tools = env['TOOLS']
	if "mingw" in tools:
		tools.remove('mingw')
	if "applelink" in tools:
		tools.remove("applelink")
		env.Tool('gcc')
	return env.Clone(tools=tools);

def configure(env):

	# Workaround for MinGW. See:
	# http://www.scons.org/wiki/LongCmdLinesOnWin32
	import os
	if (os.name=="nt"):

		import subprocess

		def mySubProcess(cmdline,env):
			#print "SPAWNED : " + cmdline
			startupinfo = subprocess.STARTUPINFO()
			startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
			proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
				stderr=subprocess.PIPE, startupinfo=startupinfo, shell = False, env = env)
			data, err = proc.communicate()
			rv = proc.wait()
			if rv:
				print "====="
				print err
				print "====="
			return rv

		def mySpawn(sh, escape, cmd, args, env):

			newargs = ' '.join(args[1:])
			cmdline = cmd + " " + newargs

			rv=0
			if len(cmdline) > 32000 and cmd.endswith("ar") :
				cmdline = cmd + " " + args[1] + " " + args[2] + " "
				for i in range(3,len(args)) :
					rv = mySubProcess( cmdline + args[i], env )
					if rv :
						break
			else:
				rv = mySubProcess( cmdline, env )

			return rv

		env['SPAWN'] = mySpawn

	ndk_platform=env['ndk_platform']

	if env['android_arch'] not in ['armv7','armv6','x86']:
		env['android_arch']='armv7'

	if env['android_arch']=='x86':
		env['NDK_TARGET']=env['NDK_TARGET_X86']
		env["x86_libtheora_opt_gcc"]=True

	if env['PLATFORM'] == 'win32':
		env.Tool('gcc')
		env['SHLIBSUFFIX'] = '.so'


	neon_text=""
	if env["android_arch"]=="armv7" and env['android_neon']=='yes':
		neon_text=" (with neon)"
	print("Godot Android!!!!! ("+env['android_arch']+")"+neon_text)

	env.Append(CPPPATH=['#platform/android'])

	if env['android_arch']=='x86':
		env.extra_suffix=".x86"+env.extra_suffix
	elif env['android_arch']=='armv6':
		env.extra_suffix=".armv6"+env.extra_suffix
	elif env["android_arch"]=="armv7":
		if env['android_neon']=='yes':
			env.extra_suffix=".armv7.neon"+env.extra_suffix
		else:
			env.extra_suffix=".armv7"+env.extra_suffix

	gcc_path=env["ANDROID_NDK_ROOT"]+"/toolchains/"+env["NDK_TARGET"]+"/prebuilt/";

	if (sys.platform.startswith("linux")):
		if (platform.machine().endswith('64')):
			gcc_path=gcc_path+"/linux-x86_64/bin"
		else:
			gcc_path=gcc_path+"/linux-x86/bin"
	elif (sys.platform.startswith("darwin")):
		gcc_path=gcc_path+"/darwin-x86_64/bin"
		env['SHLINKFLAGS'][1] = '-shared'
		env['SHLIBSUFFIX'] = '.so'
	elif (sys.platform.startswith('win')):
                if (platform.machine().endswith('64')):
			gcc_path=gcc_path+"/windows-x86_64/bin"
                else:
			gcc_path=gcc_path+"/windows-x86/bin"

	env['ENV']['PATH'] = gcc_path+":"+env['ENV']['PATH']
	if env['android_arch']=='x86':
		env['CC'] = gcc_path+'/i686-linux-android-gcc'
		env['CXX'] = gcc_path+'/i686-linux-android-g++'
		env['AR'] = gcc_path+"/i686-linux-android-ar"
		env['RANLIB'] = gcc_path+"/i686-linux-android-ranlib"
		env['AS'] = gcc_path+"/i686-linux-android-as"
	else:
		env['CC'] = gcc_path+'/arm-linux-androideabi-gcc'
		env['CXX'] = gcc_path+'/arm-linux-androideabi-g++'
		env['AR'] = gcc_path+"/arm-linux-androideabi-ar"
		env['RANLIB'] = gcc_path+"/arm-linux-androideabi-ranlib"
		env['AS'] = gcc_path+"/arm-linux-androideabi-as"

	if env['android_arch']=='x86':
		env['ARCH'] = 'arch-x86'
	else:
		env['ARCH'] = 'arch-arm'

	import string
	#include path
	gcc_include=env["ANDROID_NDK_ROOT"]+"/platforms/"+ndk_platform+"/"+env['ARCH'] +"/usr/include"
	ld_sysroot=env["ANDROID_NDK_ROOT"]+"/platforms/"+ndk_platform+"/"+env['ARCH']
	#glue_include=env["ANDROID_NDK_ROOT"]+"/sources/android/native_app_glue"
	ld_path=env["ANDROID_NDK_ROOT"]+"/platforms/"+ndk_platform+"/"+env['ARCH']+"/usr/lib"
	env.Append(CPPPATH=[gcc_include])
#	env['CCFLAGS'] = string.split('-DNO_THREADS -MMD -MP -MF -fpic -ffunction-sections -funwind-tables -fstack-protector -D__ARM_ARCH_5__ -D__ARM_ARCH_5T__ -D__ARM_ARCH_5E__ -D__ARM_ARCH_5TE__  -Wno-psabi -march=armv5te -mtune=xscale -msoft-float  -fno-exceptions -mthumb -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED ')

	env['neon_enabled']=False
	if env['android_arch']=='x86':
		env.Append(CCFLAGS=string.split('-DNO_STATVFS -fpic -ffunction-sections -funwind-tables -fstack-protector -fvisibility=hidden -D__GLIBC__  -Wno-psabi -ftree-vectorize -funsafe-math-optimizations -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED'))
	elif env["android_arch"]=="armv6":
		env.Append(CCFLAGS=string.split('-DNO_STATVFS -fpic -ffunction-sections -funwind-tables -fstack-protector -fvisibility=hidden -D__ARM_ARCH_6__ -D__GLIBC__  -Wno-psabi -march=armv6 -mfpu=vfp -mfloat-abi=softfp -funsafe-math-optimizations -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED'))
	elif env["android_arch"]=="armv7":
		env.Append(CCFLAGS=string.split('-DNO_STATVFS -fpic -ffunction-sections -funwind-tables -fstack-protector -fvisibility=hidden -D__ARM_ARCH_7__ -D__ARM_ARCH_7A__ -D__GLIBC__  -Wno-psabi -march=armv7-a -mfloat-abi=softfp -ftree-vectorize -funsafe-math-optimizations -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED'))
		if env['android_neon']=='yes':
			env['neon_enabled']=True
			env.Append(CCFLAGS=['-mfpu=neon','-D__ARM_NEON__'])
		else:
			env.Append(CCFLAGS=['-mfpu=vfpv3-d16'])

	env.Append(LDPATH=[ld_path])
	env.Append(LIBS=['OpenSLES'])
#	env.Append(LIBS=['c','m','stdc++','log','EGL','GLESv1_CM','GLESv2','OpenSLES','supc++','android'])
	env.Append(LIBS=['EGL','OpenSLES','android'])
	env.Append(LIBS=['c','m','stdc++','log','GLESv1_CM','GLESv2', 'z'])

	env["LINKFLAGS"]= string.split(" -g --sysroot="+ld_sysroot+" -Wl,--no-undefined -Wl,-z,noexecstack ")
	env.Append(LINKFLAGS=["-Wl,-soname,libgodot_android.so"])

	if (env["target"]=="release"):

		env.Append(CCFLAGS=['-O2', '-ffast-math','-fomit-frame-pointer'])

	elif (env["target"]=="release_debug"):

		env.Append(CCFLAGS=['-O2', '-ffast-math','-DDEBUG_ENABLED'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['-D_DEBUG', '-g1', '-Wall', '-O0', '-DDEBUG_ENABLED'])
		env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC'])

	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED', '-DNO_FCNTL','-DMPC_FIXED_POINT'])
#	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED','-DMPC_FIXED_POINT'])

	# TODO: Move that to opus module's config
	if("module_opus_enabled" in env and env["module_opus_enabled"] != "no"):
		if (env["android_arch"]=="armv6" or env["android_arch"]=="armv7"):
			env.Append(CFLAGS=["-DOPUS_ARM_OPT"])
		env.opus_fixed_point="yes"

	if (env['android_stl']=='yes'):
		#env.Append(CCFLAGS=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/system/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/include"])
		if env['android_arch']=='x86':
			env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/x86/include"])
			env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/x86"])
		elif env['android_arch']=='armv6':
			env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi/include"])
			env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi"])
		elif env["android_arch"]=="armv7":
			env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include"])
			env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a"])

		env.Append(LIBS=["gnustl_static","supc++"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cpufeatures"])

		#env.Append(CCFLAGS=["-I"+env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/stlport/stlport"])
		#env.Append(CCFLAGS=["-I"+env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/libs/armeabi/include"])
		#env.Append(LINKFLAGS=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/libs/armeabi/libstdc++.a"])
	else:

		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cpufeatures"])
		if env['android_arch']=='x86':
			env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/x86"])
		elif env["android_arch"]=="armv6":
			env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi"])
		elif env["android_arch"]=="armv7":
			env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a"])
		env.Append(LIBS=['gnustl_static'])
		env.Append(CCFLAGS=["-fno-exceptions",'-DNO_SAFE_CAST'])


	import methods
	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )

	env.use_windows_spawn_fix()
