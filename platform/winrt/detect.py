import os

import sys
import string
import methods


def is_active():
	return True

def get_name():
		return "WinRT"

def can_build():
	if (os.name=="nt"):
		#building natively on windows!
		if (os.getenv("VSINSTALLDIR")):

			if (os.getenv("ANGLE_SRC_PATH") == None):
				return False

			return True
	return False

def get_opts():
	return []

def get_flags():

	return [
	('tools', 'no'),
	('builtin_zlib', 'yes'),
	('openssl', 'builtin'),
	('xaudio2', 'yes'),
	]


def configure(env):

	if(env["bits"] != "default"):
		print "Error: bits argument is disabled for MSVC"
		print ("Bits argument is not supported for MSVC compilation. Architecture depends on the Native/Cross Compile Tools Prompt/Developer Console (or Visual Studio settings)"
			   +" that is being used to run SCons. As a consequence, bits argument is disabled. Run scons again without bits argument (example: scons p=winrt) and SCons will attempt to detect what MSVC compiler"
			   +" will be executed and inform you.")
		sys.exit()

	arch = ""
	env['ENV'] = os.environ;

	# ANGLE
	angle_root = os.getenv("ANGLE_SRC_PATH")
	env.Append(CPPPATH=[angle_root + '/include'])
	jobs = str(env.GetOption("num_jobs"))
	angle_build_cmd = "msbuild.exe " + angle_root + "/winrt/10/src/angle.sln /nologo /v:m /m:" + jobs + " /p:Configuration=Release /p:Platform="

	if os.path.isfile(str(os.getenv("ANGLE_SRC_PATH")) + "/winrt/10/src/angle.sln"):
		env["build_angle"] = True

	if os.getenv('Platform') == "ARM":

		print "Compiled program architecture will be an ARM executable. (forcing bits=32)."

		arch="arm"
		env["bits"]="32"
		env.Append(LINKFLAGS=['/MACHINE:ARM'])
		env.Append(LIBPATH=[os.environ['VCINSTALLDIR'] + 'lib/store/arm'])

		angle_build_cmd += "ARM"

		env.Append(LIBPATH=[angle_root + '/winrt/10/src/Release_ARM/lib'])

	else:

		compiler_version_str = methods.detect_visual_c_compiler_version(env['ENV'])

		if(compiler_version_str == "amd64" or compiler_version_str == "x86_amd64"):
			env["bits"]="64"
			print "Compiled program architecture will be a x64 executable (forcing bits=64)."
		elif (compiler_version_str=="x86" or compiler_version_str == "amd64_x86"):
			env["bits"]="32"
			print "Compiled program architecture will be a x86 executable. (forcing bits=32)."
		else:
			print "Failed to detect MSVC compiler architecture version... Defaulting to 32bit executable settings (forcing bits=32). Compilation attempt will continue, but SCons can not detect for what architecture this build is compiled for. You should check your settings/compilation setup."
			env["bits"]="32"

		if (env["bits"] == "32"):
			arch = "x86"

			angle_build_cmd += "Win32"

			env.Append(CPPFLAGS=['/DPNG_ABORT=abort'])
			env.Append(LINKFLAGS=['/MACHINE:X86'])
			env.Append(LIBPATH=[os.environ['VCINSTALLDIR'] + 'lib/store'])
			env.Append(LIBPATH=[angle_root + '/winrt/10/src/Release_Win32/lib'])

		else:
			arch = "x64"

			angle_build_cmd += "x64"

			env.Append(LINKFLAGS=['/MACHINE:X64'])
			env.Append(LIBPATH=[os.environ['VCINSTALLDIR'] + 'lib/store/amd64'])
			env.Append(LIBPATH=[angle_root + '/winrt/10/src/Release_x64/lib'])

	env.Append(CPPPATH=['#platform/winrt','#drivers/windows'])
	env.Append(LINKFLAGS=['/MANIFEST:NO', '/NXCOMPAT', '/DYNAMICBASE', '/WINMD', '/APPCONTAINER', '/ERRORREPORT:PROMPT', '/NOLOGO', '/TLBID:1', '/NODEFAULTLIB:"kernel32.lib"', '/NODEFAULTLIB:"ole32.lib"'])
	env.Append(CPPFLAGS=['/D','__WRL_NO_DEFAULT_LIB__','/D','WIN32'])
	env.Append(CPPFLAGS=['/FU', os.environ['VCINSTALLDIR'] + 'lib/store/references/platform.winmd'])
	env.Append(CPPFLAGS=['/AI', os.environ['VCINSTALLDIR'] + 'lib/store/references'])

	env.Append(LIBPATH=[os.environ['VCINSTALLDIR'] + 'lib/store/references'])

	if (env["target"]=="release"):

		env.Append(CPPFLAGS=['/O2', '/GL'])
		env.Append(CPPFLAGS=['/MD'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:WINDOWS', '/LTCG'])

	elif (env["target"]=="release_debug"):

		env.Append(CCFLAGS=['/O2','/Zi','/DDEBUG_ENABLED'])
		env.Append(CPPFLAGS=['/MD'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['/Zi','/DDEBUG_ENABLED','/DDEBUG_MEMORY_ENABLED'])
		env.Append(CPPFLAGS=['/MDd'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])
		env.Append(LINKFLAGS=['/DEBUG'])


	env.Append(CCFLAGS=string.split('/FS /MP /GS /wd"4453" /wd"28204" /wd"4291" /Zc:wchar_t /Gm- /fp:precise /D "_UNICODE" /D "UNICODE" /D "WINAPI_FAMILY=WINAPI_FAMILY_APP" /errorReport:prompt /WX- /Zc:forScope /Gd /EHsc /nologo'))
	env.Append(CXXFLAGS=string.split('/ZW /FS'))
	env.Append(CCFLAGS=['/AI', os.environ['VCINSTALLDIR']+'\\vcpackages', '/AI', os.environ['WINDOWSSDKDIR']+'\\References\\CommonConfiguration\\Neutral'])


	env["PROGSUFFIX"]="."+arch+env["PROGSUFFIX"]
	env["OBJSUFFIX"]="."+arch+env["OBJSUFFIX"]
	env["LIBSUFFIX"]="."+arch+env["LIBSUFFIX"]

	env.Append(CCFLAGS=['/DWINRT_ENABLED'])
	env.Append(CCFLAGS=['/DWINDOWS_ENABLED'])
	env.Append(CCFLAGS=['/DTYPED_METHOD_BIND'])

	env.Append(CCFLAGS=['/DGLES2_ENABLED','/DGL_GLEXT_PROTOTYPES','/DEGL_EGLEXT_PROTOTYPES','/DANGLE_ENABLED'])

	LIBS = [
		'WindowsApp',
		'mincore',
		'libANGLE',
		'libEGL',
		'libGLESv2',
		]
	env.Append(LINKFLAGS=[p+".lib" for p in LIBS])

	# Incremental linking fix
	env['BUILDERS']['ProgramOriginal'] = env['BUILDERS']['Program']
	env['BUILDERS']['Program'] = methods.precious_program

	env.Append( BUILDERS = { 'ANGLE' : env.Builder(action = angle_build_cmd) } )

	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'HLSL9' : env.Builder(action = methods.build_hlsl_dx9_headers, suffix = 'hlsl.h',src_suffix = '.hlsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
