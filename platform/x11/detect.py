
import os
import sys	


def is_active():
	return True
        
def get_name():
        return "X11"


def can_build():

	if (os.name!="posix"):
		return False

	if sys.platform == "darwin":
		return False # no x11 on mac for now

	errorval=os.system("pkg-config --version > /dev/null")
	
	if (errorval):
		print("pkg-config not found.. x11 disabled.")
		return False
	
	x11_error=os.system("pkg-config x11 --modversion > /dev/null ")
	if (x11_error):
		print("X11 not found.. x11 disabled.")
		return False

        ssl_error=os.system("pkg-config openssl --modversion > /dev/null ")
        if (ssl_error):
                print("OpenSSL not found.. x11 disabled.")
                return False

	x11_error=os.system("pkg-config xcursor --modversion > /dev/null ")
	if (x11_error):
		print("xcursor not found.. x11 disabled.")
		return False

	
	return True # X11 enabled
  
def get_opts():

	return [
	('use_llvm','Use llvm compiler','no'),
	('use_sanitizer','Use llvm compiler sanitize address','no'),
	('force_32_bits','Force 32 bits binary','no')
	]
  
def get_flags():

	return [
	('opengl', 'no'),
	('legacygl', 'yes'),
	('builtin_zlib', 'no'),
	("openssl", "yes"),
        ]
			


def configure(env):

	env.Append(CPPPATH=['#platform/x11'])
	if (env["use_llvm"]=="yes"):
		env["CC"]="clang"
		env["CXX"]="clang++"
		env["LD"]="clang++"
		if (env["use_sanitizer"]=="yes"):
			env.Append(CXXFLAGS=['-fsanitize=address','-fno-omit-frame-pointer'])
			env.Append(LINKFLAGS=['-fsanitize=address'])



	if (env["tools"]=="no"):
		#no tools suffix
		env['OBJSUFFIX'] = ".nt"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = ".nt"+env['LIBSUFFIX']


	if (env["target"]=="release"):
		
		env.Append(CCFLAGS=['-O2','-ffast-math','-fomit-frame-pointer'])
		env['OBJSUFFIX'] = "_opt"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = "_opt"+env['LIBSUFFIX']

	elif (env["target"]=="release_debug"):

		env.Append(CCFLAGS=['-O2','-ffast-math','-DDEBUG_ENABLED'])
		env['OBJSUFFIX'] = "_optd"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = "_optd"+env['LIBSUFFIX']


#		env.Append(CCFLAGS=['-Os','-ffast-math','-fomit-frame-pointer'])
#does not seem to have much effect		
#		env.Append(CCFLAGS=['-fno-default-inline'])
#recommended by wxwidgets
#		env.Append(CCFLAGS=['-ffunction-sections','-fdata-sections'])
#		env.Append(LINKFLAGS=['-Wl','--gc-sections'])

	elif (env["target"]=="debug"):
				
		env.Append(CCFLAGS=['-g2', '-Wall','-DDEBUG_ENABLED','-DDEBUG_MEMORY_ENABLED'])
#does not seem to have much effect		
#		env.Append(CCFLAGS=['-fno-default-inline'])
#recommended by wxwidgets
#		env.Append(CCFLAGS=['-ffunction-sections','-fdata-sections'])
#		env.Append(LINKFLAGS=['-Wl','--gc-sections'])

	elif (env["target"]=="debug_light"):

		env.Append(CCFLAGS=['-g1', '-Wall','-DDEBUG_ENABLED','-DDEBUG_MEMORY_ENABLED'])


	elif (env["target"]=="profile"):
		
		env.Append(CCFLAGS=['-g','-pg'])
		env.Append(LINKFLAGS=['-pg'])		

	env.ParseConfig('pkg-config x11 --cflags --libs')
	env.ParseConfig('pkg-config xcursor --cflags --libs')
	env.ParseConfig('pkg-config openssl --cflags --libs')


	env.ParseConfig('pkg-config freetype2 --cflags --libs')
	env.Append(CCFLAGS=['-DFREETYPE_ENABLED'])

	
	if env['opengl'] == 'yes':
                env.Append(CPPFLAGS=['-DOPENGL_ENABLED','-DGLEW_ENABLED'])
	#env.Append(CPPFLAGS=["-DRTAUDIO_ENABLED"])
	env.Append(CPPFLAGS=["-DALSA_ENABLED"])
	env.Append(CPPFLAGS=['-DX11_ENABLED','-DUNIX_ENABLED','-DGLES2_ENABLED','-DGLES1_ENABLED','-DGLES_OVER_GL'])
#        env.Append(CPPFLAGS=['-DX11_ENABLED','-DUNIX_ENABLED','-DGLES2_ENABLED','-DGLES_OVER_GL'])
	env.Append(LIBS=['GL', 'GLU', 'pthread','asound','z']) #TODO detect linux/BSD!
	#env.Append(CPPFLAGS=['-DMPC_FIXED_POINT'])
	if (env["force_32_bits"]=="yes"):
		env.Append(CPPFLAGS=['-m32'])
		env.Append(LINKFLAGS=['-m32','-L/usr/lib/i386-linux-gnu'])
		env['OBJSUFFIX'] = ".32"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = ".32"+env['LIBSUFFIX']


	if (env["CXX"]=="clang++"):
		env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
		env["CC"]="clang"
		env["LD"]="clang++"

	import methods

	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	#env.Append( BUILDERS = { 'HLSL9' : env.Builder(action = methods.build_hlsl_dx9_headers, suffix = 'hlsl.h',src_suffix = '.hlsl') } )

