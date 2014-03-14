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
             ('NDK_TOOLCHAIN', 'toolchain to use for the NDK',"arm-eabi-4.4.0"), 	                      
             #android 2.3       
		 ('ndk_platform', 'compile for platform: (2.2,2.3)',"2.2"),
		 ('NDK_TARGET', 'toolchain to use for the NDK',"arm-linux-androideabi-4.8"),
	     ('android_stl','enable STL support in android port (for modules)','no'),
	     ('armv6','compile for older phones running arm v6 (instead of v7+neon+smp)','no')

	]

def get_flags():

	return [
		('lua', 'no'),
		('tools', 'no'),
		('nedmalloc', 'no'),
		('builtin_zlib', 'no'),
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

	if env['PLATFORM'] == 'win32':
		import methods
		env.Tool('gcc')
		env['SPAWN'] = methods.win32_spawn

	env.android_source_modules.append("../libs/apk_expansion")	
	ndk_platform=""

	ndk_platform="android-15"

	print("Godot Android!!!!!")

	env.Append(CPPPATH=['#platform/android'])
	
	env['OBJSUFFIX'] = ".android.o"
	env['LIBSUFFIX'] = ".android.a"
	env['PROGSUFFIX'] = ".android"
	env['SHLIBSUFFIX'] = ".so"
	
	gcc_path=env["ANDROID_NDK_ROOT"]+"/toolchains/"+env["NDK_TARGET"]+"/prebuilt/";
	
	import os
	if (sys.platform.find("linux")==0):
		if (platform.architecture()[0]=='64bit' or os.path.isdir(gcc_path+"linux-x86_64/bin")): # check was not working
			gcc_path=gcc_path+"/linux-x86_64/bin"
		else:
			gcc_path=gcc_path+"/linux-x86/bin"
	elif (sys.platform=="darwin"):
		gcc_path=gcc_path+"/darwin-x86_64/bin" #this may be wrong
		env['SHLINKFLAGS'][1] = '-shared'
	elif (os.name=="nt"):
		gcc_path=gcc_path+"/windows/bin" #this may be wrong
	
	

	env['ENV']['PATH'] = gcc_path+":"+env['ENV']['PATH']

	env['CC'] = gcc_path+'/arm-linux-androideabi-gcc'
	env['CXX'] = gcc_path+'/arm-linux-androideabi-g++'
	env['AR'] = gcc_path+"/arm-linux-androideabi-ar"
	env['RANLIB'] = gcc_path+"/arm-linux-androideabi-ranlib"
	env['AS'] = gcc_path+"/arm-linux-androideabi-as"

	import string
	#include path
	gcc_include=env["ANDROID_NDK_ROOT"]+"/platforms/"+ndk_platform+"/arch-arm/usr/include"
	ld_sysroot=env["ANDROID_NDK_ROOT"]+"/platforms/"+ndk_platform+"/arch-arm"
	#glue_include=env["ANDROID_NDK_ROOT"]+"/sources/android/native_app_glue"
	ld_path=env["ANDROID_NDK_ROOT"]+"/platforms/"+ndk_platform+"/arch-arm/usr/lib"
	env.Append(CPPPATH=[gcc_include])
#	env['CCFLAGS'] = string.split('-DNO_THREADS -MMD -MP -MF -fpic -ffunction-sections -funwind-tables -fstack-protector -D__ARM_ARCH_5__ -D__ARM_ARCH_5T__ -D__ARM_ARCH_5E__ -D__ARM_ARCH_5TE__  -Wno-psabi -march=armv5te -mtune=xscale -msoft-float  -fno-exceptions -mthumb -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED ')
	print("********* armv6", env['armv6'])
	if env["armv6"]!="no":
		env['CCFLAGS'] = string.split('-DNO_STATVFS -MMD -MP -MF -fpic -ffunction-sections -funwind-tables -fstack-protector -D__ARM_ARCH_6__ -D__GLIBC__  -Wno-psabi -march=armv6 -mfpu=vfp -mfloat-abi=softfp -funsafe-math-optimizations -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED -DGLES1_ENABLED')
	else:
		env['CCFLAGS'] = string.split('-DNO_STATVFS -MMD -MP -MF -fpic -ffunction-sections -funwind-tables -fstack-protector -D__ARM_ARCH_7__ -D__GLIBC__  -Wno-psabi -march=armv6 -mfpu=neon -mfloat-abi=softfp -ftree-vectorize -funsafe-math-optimizations -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED -DGLES1_ENABLED')

	env.Append(LDPATH=[ld_path])
	env.Append(LIBS=['OpenSLES'])
#	env.Append(LIBS=['c','m','stdc++','log','EGL','GLESv1_CM','GLESv2','OpenSLES','supc++','android'])
	if (env["ndk_platform"]!="2.2"):
		env.Append(LIBS=['EGL','OpenSLES','android'])
	env.Append(LIBS=['c','m','stdc++','log','GLESv1_CM','GLESv2', 'z'])

	env["LINKFLAGS"]= string.split(" -g --sysroot="+ld_sysroot+" -Wl,--no-undefined -Wl,-z,noexecstack ")
	env.Append(LINKFLAGS=["-Wl,-soname,libgodot_android.so"])

	if (env["target"]=="release"):

		env.Append(CCFLAGS=['-O2', '-ffast-math','-fomit-frame-pointer'])
		env['OBJSUFFIX'] = "_opt"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = "_opt"+env['LIBSUFFIX']

	elif (env["target"]=="release_debug"):

		env.Append(CCFLAGS=['-O2', '-ffast-math','-DDEBUG_ENABLED'])
		env['OBJSUFFIX'] = "_optd"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = "_optd"+env['LIBSUFFIX']

	elif (env["target"]=="profile"):

		env.Append(CCFLAGS=['-O2', '-ffast-math','-fomit-frame-pointer', '-g1'])
		env.Append(LIBPATH=['#platform/android/armeabi'])
		env.Append(LIBS=['andprof'])
		env['OBJSUFFIX'] = "_prof"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = "_prof"+env['LIBSUFFIX']
		env['SHLIBSUFFIX'] = "_prof"+env['SHLIBSUFFIX']

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['-D_DEBUG', '-g1', '-Wall', '-O0', '-DDEBUG_ENABLED'])
		env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC'])

	if env["armv6"] == "no":
		env['neon_enabled']=True
	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED', '-DNO_FCNTL','-DMPC_FIXED_POINT'])
#	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED','-DMPC_FIXED_POINT'])
	if (env['android_stl']=='yes'):
		#env.Append(CCFLAGS=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/system/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.4.3/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.4.3/libs/armeabi/include"])
		env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.4.3/libs/armeabi"])
		env.Append(LIBS=["gnustl_static","supc++"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cpufeatures"])

		#env.Append(CCFLAGS=["-I"+env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/stlport/stlport"])
		#env.Append(CCFLAGS=["-I"+env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/libs/armeabi/include"])
		#env.Append(LINKFLAGS=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/libs/armeabi/libstdc++.a"])
	else:

		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gabi++/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cpufeatures"])
		env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gabi++/libs/armeabi"])
		env.Append(LIBS=['gabi++_static'])
		env.Append(CCFLAGS=["-fno-exceptions",'-DNO_SAFE_CAST'])

	import methods
	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )

