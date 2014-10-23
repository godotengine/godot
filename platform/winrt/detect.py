

import os

import sys	


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

	env['OBJSUFFIX'] = ".rt" + env['OBJSUFFIX']
	env['LIBSUFFIX'] = ".rt" + env['LIBSUFFIX']

	env.Append(LIBPATH=['#platform/winrt/x64/lib'])

	if (env["target"]=="release"):

		env.Append(CCFLAGS=['/O2'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:WINDOWS'])
		env.Append(LINKFLAGS=['/ENTRY:mainCRTStartup'])

	elif (env["target"]=="test"):

		env.Append(CCFLAGS=['/O2','/DDEBUG_ENABLED','/DD3D_DEBUG_INFO'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['/Zi','/DDEBUG_ENABLED','/DD3D_DEBUG_INFO','/O1'])
		env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])
		env.Append(LINKFLAGS=['/DEBUG'])

	elif (env["target"]=="profile"):

		env.Append(CCFLAGS=['-g','-pg'])
		env.Append(LINKFLAGS=['-pg'])

	env.Append(CCFLAGS=['/Gd','/GR','/nologo', '/EHsc'])
	env.Append(CXXFLAGS=['/TP', '/ZW'])
	env.Append(CPPFLAGS=['/DMSVC', '/GR', ])
	#env.Append(CCFLAGS=['/I'+os.getenv("WindowsSdkDir")+"/Include"])
	env.Append(CCFLAGS=['/DWINRT_ENABLED'])
	env.Append(CCFLAGS=['/DWINDOWS_ENABLED'])
	env.Append(CCFLAGS=['/DWINAPI_FAMILY=WINAPI_FAMILY_APP', '/D_WIN32_WINNT=0x0603', '/DNTDDI_VERSION=0x06030000'])
	env.Append(CCFLAGS=['/DRTAUDIO_ENABLED'])
	#env.Append(CCFLAGS=['/DWIN32'])
	env.Append(CCFLAGS=['/DTYPED_METHOD_BIND'])

	env.Append(CCFLAGS=['/DGLES2_ENABLED'])
	#env.Append(CCFLAGS=['/DGLES1_ENABLED'])
	env.Append(LIBS=['winmm','opengl32','dsound','kernel32','ole32','user32','gdi32', 'IPHLPAPI', 'wsock32', 'shell32','advapi32'])
		
	import methods
	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'HLSL9' : env.Builder(action = methods.build_hlsl_dx9_headers, suffix = 'hlsl.h',src_suffix = '.hlsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )

	env['ENV'] = os.environ;


