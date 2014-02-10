
import os
import sys	


def is_active():
	return True
        
def get_name():
	return "Server"


def can_build():

	if (os.name!="posix"):
		return False

	return True # enabled
  
def get_opts():

	return [
	('use_llvm','Use llvm compiler','no'),
	('force_32_bits','Force 32 bits binary','no')
	]
  
def get_flags():

	return [
	('builtin_zlib', 'no'),
	]
			


def configure(env):

	env.Append(CPPPATH=['#platform/server'])
	if (env["use_llvm"]=="yes"):
		env["CC"]="clang"
		env["CXX"]="clang++"
		env["LD"]="clang++"

	env['OBJSUFFIX'] = ".srv"+env['OBJSUFFIX']
	env['LIBSUFFIX'] = ".srv"+env['LIBSUFFIX']

	if (env["force_32_bits"]!="no"):
		env['OBJSUFFIX'] = ".32"+env['OBJSUFFIX']
		env['LIBSUFFIX'] = ".32"+env['LIBSUFFIX']


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


	elif (env["target"]=="debug"):
				
		env.Append(CCFLAGS=['-g2', '-Wall','-DDEBUG_ENABLED','-DDEBUG_MEMORY_ENABLED'])

	elif (env["target"]=="profile"):
		
		env.Append(CCFLAGS=['-g','-pg'])
		env.Append(LINKFLAGS=['-pg'])		

	
	env.Append(CPPFLAGS=['-DSERVER_ENABLED','-DUNIX_ENABLED'])
	env.Append(LIBS=['pthread','z']) #TODO detect linux/BSD!

	if (env["force_32_bits"]=="yes"):
		env.Append(CPPFLAGS=['-m32'])
		env.Append(LINKFLAGS=['-m32','-L/usr/lib/i386-linux-gnu'])

	if (env["CXX"]=="clang++"):
		env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
		env["CC"]="clang"
		env["LD"]="clang++"

