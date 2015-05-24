import os
import sys

def is_active():
	return True

def get_name():
	return "Haiku"

def can_build():
	if (os.name != "posix"):
		return False
	
	if (sys.platform == "darwin"):
		return False
	
	return True

def get_opts():
	return []

def get_flags():
	return [
		('builtin_zlib', 'no')
	]

def configure(env):
	is64=sys.maxsize > 2**32

	if (env["bits"]=="default"):
		if (is64):
			env["bits"]="64"
		else:
			env["bits"]="32"
	
	env.Append(CPPPATH = ['#platform/haiku'])
	env["CC"] = "gcc-x86"
	env["CXX"] = "g++-x86"
	env.Append(CPPFLAGS = ['-DDEBUG_METHODS_ENABLED'])
	
	env.Append(CPPFLAGS = ['-DUNIX_ENABLED'])
	#env.Append(LIBS = ['be'])
