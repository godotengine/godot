import os
import sys
import string

def can_build():

        import os
        if (not os.environ.has_key("ANDROID_NDK_ROOT")):
        	print("ANDROID_NDK_ROOT not present, Android disabled.")
        	return False
	return True

def get_opts():

	return [
	     ('ANDROID_NDK_ROOT', 'the path to Android NDK', os.environ.get("ANDROID_NDK_ROOT", 0)), 
             ('NDK_TOOLCHAIN', 'toolchain to use for the NDK',"arm-eabi-4.4.0"), 	                      
             #android 2.2        
#             ('NDK_PLATFORM', 'platform to use for the NDK',"android-8"), 	                              
#             ('NDK_TARGET', 'toolchain to use for the NDK',"arm-linux-androideabi-4.4.3"), 	                              
	]

def get_flags():

	return [
		('lua', 'no'),
		('tools', 'no'),
		('nedmalloc', 'no'),
	]



def configure(env):


	print("Godot Android!!!!!")

	env.Append(CPPPATH=['#platform/android'])
	
	env['OBJSUFFIX'] = ".jandroid.o"
	env['LIBSUFFIX'] = ".jandroid.a"
	env['PROGSUFFIX'] = ".jandroid"
	
	gcc_path=env["ANDROID_NDK_ROOT"]+"/toolchains/"+env["NDK_TARGET"]+"/prebuilt/";
	
	if (sys.platform.find("linux")==0):
		gcc_path=gcc_path+"/linux-x86/bin"
	elif (sys.platform=="darwin"):
		gcc_path=gcc_path+"/darwin-x86/bin" #this may be wrong
	elif (os.name=="nt"):
		gcc_path=gcc_path+"/windows-x86/bin" #this may be wrong
	
	

	env['ENV']['PATH'] = gcc_path+":"+env['ENV']['PATH']

	env['CC'] = gcc_path+'/arm-linux-androideabi-gcc'
	env['CXX'] = gcc_path+'/arm-linux-androideabi-g++'
	env['AR'] = gcc_path+"/arm-linux-androideabi-ar"

	import string
	#include path
	gcc_include=env["ANDROID_NDK_ROOT"]+"/platforms/"+env["NDK_PLATFORM"]+"/arch-arm/usr/include"
	ld_sysroot=env["ANDROID_NDK_ROOT"]+"/platforms/"+env["NDK_PLATFORM"]+"/arch-arm"
	ld_path=env["ANDROID_NDK_ROOT"]+"/platforms/"+env["NDK_PLATFORM"]+"/arch-arm/usr/lib"
	#cxx_include=env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/system/include"
	#env.Append(CPPPATH=[gcc_include,cxx_include])
	env['CCFLAGS'] = string.split('-DNO_THREADS -DNO_STATVFS -MMD -MP -MF -fpic -ffunction-sections -funwind-tables -fstack-protector -D__ARM_ARCH_5__ -D__ARM_ARCH_5T__ -D__ARM_ARCH_5E__ -D__ARM_ARCH_5TE__  -Wno-psabi -march=armv5te -mtune=xscale -msoft-float  -fno-exceptions -mthumb -fno-strict-aliasing -DANDROID -Wa,--noexecstack -DGLES2_ENABLED ')

	env.Append(LDPATH=[ld_path])
	env.Append(LIBS=['c','m','stdc++','log','GLESv2'])
	
	env["LINKFLAGS"]= string.split(" -g --sysroot="+ld_sysroot+" -Wl,--no-undefined -Wl,-z,noexecstack")
	env.Append(LINKFLAGS=["-Wl,-soname,libgodot_android.so"])

	if (env["target"]=="release"):

		env.Append(CCFLAGS=['-O2', '-ffast-math','-fomit-frame-pointer'])
		env['OBJSUFFIX'] = "_opt"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = "_opt"+env['LIBSUFFIX']

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['-D_DEBUG', '-g', '-Wall', '-O0', '-DDEBUG_ENABLED'])
		env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC'])

	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED','-DNO_FCNTL'])

	env['neon_enabled']=True
	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED', '-DNO_FCNTL','-DMPC_FIXED_POINT'])
#	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED','-DMPC_FIXED_POINT'])
	if (env['android_stl']=='yes'):
		#env.Append(CCFLAGS=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/system/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.4.3/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.4.3/libs/armeabi/include"])
		env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/4.4.3/libs/armeabi"])
		env.Append(LIBS=["gnustl_static","supc++"])
		#env.Append(CCFLAGS=["-I"+env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/stlport/stlport"])
		#env.Append(CCFLAGS=["-I"+env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/libs/armeabi/include"])
		#env.Append(LINKFLAGS=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gnu-libstdc++/libs/armeabi/libstdc++.a"])
	else:

		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gabi++/include"])
		env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cpufeatures"])
		env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"]+"/sources/cxx-stl/gabi++/libs/armeabi"])
		env.Append(LIBS=['gabi++_static'])
		env.Append(CCFLAGS=["-fno-exceptions",'-DNO_SAFE_CAST'])
