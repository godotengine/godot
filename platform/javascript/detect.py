import os
import sys
import string

def is_active():
	return True

def get_name():
	return "JavaScript"

def can_build():

	import os
	if (not os.environ.has_key("EMSCRIPTEN_ROOT")):
		return False
	return True

def get_opts():

	return [
		['compress','Compress JS Executable','no'],
		['javascript_eval','Enable JavaScript eval interface','yes']
	]

def get_flags():

	return [
		('tools', 'no'),
		('builtin_zlib', 'yes'),
		('module_etc1_enabled', 'no'),
		('module_mpc_enabled', 'no'),
		('module_theora_enabled', 'no'),
	]



def configure(env):
	env['ENV'] = os.environ;
	env.use_windows_spawn_fix('javascript')

	env.Append(CPPPATH=['#platform/javascript'])

	em_path=os.environ["EMSCRIPTEN_ROOT"]

	env['ENV']['PATH'] = em_path+":"+env['ENV']['PATH']

	env['CC'] = em_path+'/emcc'
	env['CXX'] = em_path+'/emcc'
	#env['AR'] = em_path+"/emar"
	env['AR'] = em_path+"/emcc"
	env['ARFLAGS'] = "-o"

#	env['RANLIB'] = em_path+"/emranlib"
	env['RANLIB'] = em_path + "/emcc"
	env['OBJSUFFIX'] = '.bc'
	env['LIBSUFFIX'] = '.bc'
	env['CCCOM'] = "$CC -o $TARGET $CFLAGS $CCFLAGS $_CCCOMCOM $SOURCES"
	env['CXXCOM'] = "$CC -o $TARGET $CFLAGS $CCFLAGS $_CCCOMCOM $SOURCES"

#	env.Append(LIBS=['c','m','stdc++','log','GLESv1_CM','GLESv2'])

#	env["LINKFLAGS"]= string.split(" -g --sysroot="+ld_sysroot+" -Wl,--no-undefined -Wl,-z,noexecstack ")

	if (env["target"]=="release"):
		env.Append(CCFLAGS=['-O2'])
	elif (env["target"]=="release_debug"):
		env.Append(CCFLAGS=['-O2','-DDEBUG_ENABLED'])
	elif (env["target"]=="debug"):
		env.Append(CCFLAGS=['-D_DEBUG', '-Wall', '-O2', '-DDEBUG_ENABLED'])
		#env.Append(CCFLAGS=['-D_DEBUG', '-Wall', '-g4', '-DDEBUG_ENABLED'])
		env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC'])

	# TODO: Move that to opus module's config
	if("module_opus_enabled" in env and env["module_opus_enabled"] != "no"):
		env.opus_fixed_point = "yes"

	env.Append(CPPFLAGS=["-fno-exceptions",'-DNO_SAFE_CAST','-fno-rtti'])
	env.Append(CPPFLAGS=['-DJAVASCRIPT_ENABLED', '-DUNIX_ENABLED', '-DPTHREAD_NO_RENAME', '-DNO_FCNTL','-DMPC_FIXED_POINT','-DTYPED_METHOD_BIND','-DNO_THREADS'])
	env.Append(CPPFLAGS=['-DGLES2_ENABLED'])
	env.Append(CPPFLAGS=['-DGLES_NO_CLIENT_ARRAYS'])
	env.Append(CPPFLAGS=['-s','ASM_JS=1'])
	env.Append(CPPFLAGS=['-s','FULL_ES2=1'])
#	env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED','-DMPC_FIXED_POINT'])

	if env['javascript_eval'] == 'yes':
		env.Append(CPPFLAGS=['-DJAVASCRIPT_EVAL_ENABLED'])

	if (env["compress"]=="yes"):
		lzma_binpath = em_path+"/third_party/lzma.js/lzma-native"
		lzma_decoder = em_path+"/third_party/lzma.js/lzma-decoder.js"
		lzma_dec = "LZMA.decompress"
		env.Append(LINKFLAGS=['--compression',lzma_binpath+","+lzma_decoder+","+lzma_dec])

	env.Append(LINKFLAGS=['-s','ASM_JS=1'])
	env.Append(LINKFLAGS=['-O2'])
	#env.Append(LINKFLAGS=['-g4'])

	#print "CCCOM is:", env.subst('$CCCOM')
	#print "P: ", env['p'], " Platofrm: ", env['platform']

	import methods

	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	#env.Append( BUILDERS = { 'HLSL9' : env.Builder(action = methods.build_hlsl_dx9_headers, suffix = 'hlsl.h',src_suffix = '.hlsl') } )
