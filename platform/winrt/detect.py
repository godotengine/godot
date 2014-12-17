

import os

import sys	
import string


def is_active():
	return True
        
def get_name():
		return "WinRT"

def can_build():
	if (os.name=="nt"):
		#building natively on windows!
		if (os.getenv("VSINSTALLDIR")):
			return True 
	return False
		
def get_opts():
	return []
  
def get_flags():

	return []


def configure(env):

	env.Append(CPPPATH=['#platform/winrt', '#platform/winrt/include'])

	env.Append(LINKFLAGS=['/MANIFEST:NO', '/NXCOMPAT', '/DYNAMICBASE', "kernel32.lib", '/MACHINE:X64', '/WINMD', '/APPCONTAINER', '/MANIFESTUAC:NO', '/ERRORREPORT:PROMPT', '/NOLOGO', '/TLBID:1'])

	env.Append(LIBPATH=['#platform/winrt/x64/lib'])


	if (env["target"]=="release"):

		env.Append(CCFLAGS=['/O2'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:WINDOWS'])
		env.Append(LINKFLAGS=['/ENTRY:mainCRTStartup'])

	elif (env["target"]=="test"):

		env.Append(CCFLAGS=['/O2','/DDEBUG_ENABLED','/DD3D_DEBUG_INFO'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['/Zi','/DDEBUG_ENABLED','/DD3D_DEBUG_INFO'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])
		env.Append(LINKFLAGS=['/DEBUG', '/D_DEBUG'])

	elif (env["target"]=="profile"):

		env.Append(CCFLAGS=['-g','-pg'])
		env.Append(LINKFLAGS=['-pg'])


	env.Append(CCFLAGS=string.split('/MP /GS /wd"4453" /wd"28204" /Zc:wchar_t /Gm- /Od /fp:precise /D "_UNICODE" /D "UNICODE" /D "WINAPI_FAMILY=WINAPI_FAMILY_APP" /errorReport:prompt /WX- /Zc:forScope /RTC1 /Gd /MDd /EHsc /nologo'))
	env.Append(CXXFLAGS=string.split('/ZW'))
	env.Append(CCFLAGS=['/AI', os.environ['VCINSTALLDIR']+'\\vcpackages', '/AI', os.environ['WINDOWSSDKDIR']+'\\References\\CommonConfiguration\\Neutral'])

	#env.Append(CCFLAGS=['/Gd','/GR','/nologo', '/EHsc'])
	#env.Append(CXXFLAGS=['/TP', '/ZW'])
	#env.Append(CPPFLAGS=['/DMSVC', '/GR', ])
	##env.Append(CCFLAGS=['/I'+os.getenv("WindowsSdkDir")+"/Include"])
	env.Append(CCFLAGS=['/DWINRT_ENABLED'])
	env.Append(CCFLAGS=['/DWINDOWS_ENABLED'])
	env.Append(CCFLAGS=['/DWINAPI_FAMILY=WINAPI_FAMILY_APP', '/D_WIN32_WINNT=0x0603', '/DNTDDI_VERSION=0x06030000'])
	env.Append(CCFLAGS=['/DRTAUDIO_ENABLED'])
	#env.Append(CCFLAGS=['/DWIN32'])
	env.Append(CCFLAGS=['/DTYPED_METHOD_BIND'])

	env.Append(CCFLAGS=['/DGLES2_ENABLED'])
	#env.Append(CCFLAGS=['/DGLES1_ENABLED'])

	LIBS=[
		#'winmm',
		'libEGL',
		'libGLESv2',
		'libANGLE',
		#'kernel32','ole32','user32', 'advapi32'
		]
	env.Append(LINKFLAGS=[p+".lib" for p in LIBS])

	import methods
	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'HLSL9' : env.Builder(action = methods.build_hlsl_dx9_headers, suffix = 'hlsl.h',src_suffix = '.hlsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )

	env['ENV'] = os.environ;

