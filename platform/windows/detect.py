

import os

import sys	


def is_active():
	return True
        
def get_name():
        return "Windows"

def can_build():
	
	
	if (os.name=="nt"):
		#building natively on windows!
		if (os.getenv("VSINSTALLDIR")):
			return True 
		else:
			print("MSVC Not detected, attempting mingw.")
			return True
			
			
			
	if (os.name=="posix"):
	
		if os.system("i586-mingw32msvc-gcc --version") == 0:
			return True

			
	return False
		
def get_opts():

	mwp=""
	mwp64=""
	if (os.name!="nt"):
		mwp="i586-mingw32msvc-"
		mwp64="x86_64-w64-mingw32-"

	return [
		('force_64_bits','Force 64 bits binary','no'),
		('force_32_bits','Force 32 bits binary','no'),
		('mingw_prefix','Mingw Prefix',mwp),
		('mingw_prefix_64','Mingw Prefix 64 bits',mwp64),
	]
  
def get_flags():

	return [
		('freetype','builtin'), #use builtin freetype
	]
			


def configure(env):

	if os.name == "posix":
		env['OBJSUFFIX'] = ".win"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = ".win"+env['LIBSUFFIX']

	env.Append(CPPPATH=['#platform/windows'])

	if (env["tools"]=="no"):
		#no tools suffix
		env['OBJSUFFIX'] = ".nt"+env['OBJSUFFIX']
		#env['LIBSUFFIX'] = ".nt"+env['LIBSUFFIX']
		env['platform_libsuffix'] = ".nt"+env['LIBSUFFIX']



	if (os.name=="nt" and os.getenv("VSINSTALLDIR")!=None):
		#build using visual studio
		env['ENV']['TMP'] = os.environ['TMP']
		env.Append(CPPPATH=['#platform/windows/include'])
		env.Append(LIBPATH=['#platform/windows/lib'])

		if (env["freetype"]!="no"):
			env.Append(CCFLAGS=['/DFREETYPE_ENABLED'])
			env.Append(CPPPATH=['#tools/freetype'])
			env.Append(CPPPATH=['#tools/freetype/freetype/include'])

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

		env.Append(CCFLAGS=['/MT','/Gd','/GR','/nologo'])
		env.Append(CXXFLAGS=['/TP'])
		env.Append(CPPFLAGS=['/DMSVC', '/GR', ])
		env.Append(CCFLAGS=['/I'+os.getenv("WindowsSdkDir")+"/Include"])
		env.Append(CCFLAGS=['/DWINDOWS_ENABLED'])
		env.Append(CCFLAGS=['/DRTAUDIO_ENABLED'])
		env.Append(CCFLAGS=['/DWIN32'])
		env.Append(CCFLAGS=['/DTYPED_METHOD_BIND'])

		env.Append(CCFLAGS=['/DGLES2_ENABLED'])
		env.Append(CCFLAGS=['/DGLES1_ENABLED'])
		env.Append(CCFLAGS=['/DGLEW_ENABLED'])
		env.Append(LIBS=['winmm','opengl32','dsound','kernel32','ole32','user32','gdi32','wsock32', 'shell32'])
		
		env.Append(LIBPATH=[os.getenv("WindowsSdkDir")+"/Lib"])
                if (os.getenv("DXSDK_DIR")):
                        DIRECTX_PATH=os.getenv("DXSDK_DIR")
                else:
                        DIRECTX_PATH="C:/Program Files/Microsoft DirectX SDK (March 2009)"

                if (os.getenv("VCINSTALLDIR")):
                        VC_PATH=os.getenv("VCINSTALLDIR")
                else:
                        VC_PATH=""

		env.Append(CCFLAGS=["/I" + p for p in os.getenv("INCLUDE").split(";")])
		env.Append(LIBPATH=[p for p in os.getenv("LIB").split(";")])
		env.Append(CCFLAGS=["/I"+DIRECTX_PATH+"/Include"])
		env.Append(LIBPATH=[DIRECTX_PATH+"/Lib/x86"])
		env['ENV'] = os.environ;
	else:
		#build using mingw
		if (os.name=="nt"):
			env['ENV']['TMP'] = os.environ['TMP'] #way to go scons, you can be so stupid sometimes

		mingw_prefix=""

		if (env["force_32_bits"]!="no"):
		    env['OBJSUFFIX'] = ".32"+env['OBJSUFFIX']
		    env['LIBSUFFIX'] = ".32"+env['LIBSUFFIX']
		    env.Append(CCFLAGS=['-m32'])
		    env.Append(LINKFLAGS=['-m32'])




		if (env["force_64_bits"]!="no"):
			mingw_prefix=env["mingw_prefix_64"];
			env['OBJSUFFIX'] = ".64"+env['OBJSUFFIX']
			env['LIBSUFFIX'] = ".64"+env['LIBSUFFIX']
			env.Append(LINKFLAGS=['-static'])
		else:
			mingw_prefix=env["mingw_prefix"];

		if (env["target"]=="release"):
			
			env.Append(CCFLAGS=['-O3','-ffast-math','-fomit-frame-pointer','-msse2'])
			env['OBJSUFFIX'] = "_opt"+env['OBJSUFFIX']
			env['LIBSUFFIX'] = "_opt"+env['LIBSUFFIX']
		elif (env["target"]=="release_debug"):

			env.Append(CCFLAGS=['-O2','-DDEBUG_ENABLED'])
			env['OBJSUFFIX'] = "_optd"+env['OBJSUFFIX']
			env['LIBSUFFIX'] = "_optd"+env['LIBSUFFIX']

		elif (env["target"]=="debug"):
					
			env.Append(CCFLAGS=['-g', '-Wall','-DDEBUG_ENABLED'])
		elif (env["target"]=="release_tools"):

			env.Append(CCFLAGS=['-O2','-Wall','-DDEBUG_ENABLED'])


		if (env["freetype"]!="no"):
			env.Append(CCFLAGS=['-DFREETYPE_ENABLED'])
			env.Append(CPPPATH=['#tools/freetype'])
			env.Append(CPPPATH=['#tools/freetype/freetype/include'])

		env["CC"]=mingw_prefix+"gcc"
		env['AS']=mingw_prefix+"as"
		env['CXX'] = mingw_prefix+"g++"
		env['AR'] = mingw_prefix+"ar"
		env['RANLIB'] = mingw_prefix+"ranlib"
		env['LD'] = mingw_prefix+"g++"

		#env['CC'] = "winegcc"
		#env['CXX'] = "wineg++"

		env.Append(CCFLAGS=['-DWINDOWS_ENABLED','-mwindows'])
		env.Append(CPPFLAGS=['-DRTAUDIO_ENABLED'])
		env.Append(CCFLAGS=['-DGLES2_ENABLED','-DGLES1_ENABLED','-DGLEW_ENABLED'])
		env.Append(LIBS=['mingw32','opengl32', 'dsound', 'ole32', 'd3d9','winmm','gdi32','wsock32','kernel32'])
		#'d3dx9d'
		env.Append(CPPFLAGS=['-DMINGW_ENABLED'])
		env.Append(LINKFLAGS=['-g'])

	import methods
	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'HLSL9' : env.Builder(action = methods.build_hlsl_dx9_headers, suffix = 'hlsl.h',src_suffix = '.hlsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )

	

