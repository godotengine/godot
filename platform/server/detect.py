
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
	('theora','no'), #use builtin openssl
	]
			


def configure(env):

	env.Append(CPPPATH=['#platform/server'])
	if (env["use_llvm"]=="yes"):
		env["CC"]="clang"
		env["CXX"]="clang++"
		env["LD"]="clang++"
		if (env["colored"]=="yes"):
			if sys.stdout.isatty():
				env.Append(CXXFLAGS=["-fcolor-diagnostics"])

	is64=sys.maxsize > 2**32

	if (env["bits"]=="default"):
		if (is64):
			env["bits"]="64"
		else:
			env["bits"]="32"


	#if (env["tools"]=="no"):
	#	#no tools suffix
	#	env['OBJSUFFIX'] = ".nt"+env['OBJSUFFIX']
	#	env['LIBSUFFIX'] = ".nt"+env['LIBSUFFIX']


	if (env["target"]=="release"):

		env.Append(CCFLAGS=['-O2','-ffast-math','-fomit-frame-pointer'])

	elif (env["target"]=="release_debug"):

		env.Append(CCFLAGS=['-O2','-ffast-math','-DDEBUG_ENABLED'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['-g2', '-Wall','-DDEBUG_ENABLED','-DDEBUG_MEMORY_ENABLED'])

	env.Append(CPPFLAGS=['-DSERVER_ENABLED','-DUNIX_ENABLED'])
	env.Append(LIBS=['pthread','z']) #TODO detect linux/BSD!

	if (env["CXX"]=="clang++"):
		env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
		env["CC"]="clang"
		env["LD"]="clang++"

