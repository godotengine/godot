import os
import sys
import string
import methods


def is_active():
	return True

def get_name():
	return "BlackBerry 10"

def can_build():

	import os
	if (not os.environ.has_key("QNX_TARGET")):
		return False
	return True

def get_opts():

	return [
		('QNX_HOST', 'path to qnx host', os.environ.get("QNX_HOST", 0)),
		('QNX_TARGET', 'path to qnx target', os.environ.get("QNX_TARGET", 0)),
		('QNX_CONFIGURATION', 'path to qnx configuration', os.environ.get("QNX_CONFIGURATION", 0)),
		('qnx_target', 'Qnx target (armle or x86', 'armle'),
		('bb10_payment_service', 'Enable Payment Service for BlackBerry10', 'yes'),
		('bb10_lgles_override', 'Force legacy GLES (1.1) on iOS', 'no'),
		('bb10_exceptions', 'Use exceptions when compiling on bb10', 'no'),
	]

def get_flags():

	return [
		('tools', 'no'),
		('builtin_zlib', 'yes'),
		('module_theora_enabled', 'no'),
	]

def configure(env):

	if env['PLATFORM'] == 'win32':
		env.Tool('mingw')
		env['SPAWN'] = methods.win32_spawn

	env['qnx_target_ver'] = env['qnx_target']
	if env['qnx_target'] == "armle":
		env['qnx_prefix'] = 'ntoarmv7'
		env['qnx_target_ver'] = 'armle-v7'
	else:
		env['qnx_prefix'] = 'ntox86'

	env['OBJSUFFIX'] = ".qnx.${qnx_target}.o"
	env['LIBSUFFIX'] = ".qnx.${qnx_target}.a"
	env['PROGSUFFIX'] = ".qnx.${qnx_target}"
	print("PROGSUFFIX: "+env['PROGSUFFIX']+" target: "+env['qnx_target'])

	env.PrependENVPath('PATH', env['QNX_CONFIGURATION'] + '/bin')
	env.PrependENVPath('PATH', env['QNX_CONFIGURATION'] + '/usr/bin')
	env['ENV']['QNX_HOST'] = env['QNX_HOST']
	env['ENV']['QNX_TARGET'] = env['QNX_TARGET']
	env['ENV']['QNX_CONFIGURATION'] = env['QNX_CONFIGURATION']

	env['CC'] = '$qnx_prefix-gcc'
	env['CXX'] = '$qnx_prefix-g++'
	env['AR'] = '$qnx_prefix-ar'
	env['RANLIB'] = '$qnx_prefix-ranlib'

	env.Append(CPPPATH = ['#platform/bb10'])
	env.Append(LIBPATH = ['#platform/bb10/lib/$qnx_target', '#platform/bb10/lib/$qnx_target_ver'])
	env.Append(CCFLAGS = string.split('-DBB10_ENABLED -DUNIX_ENABLED -DGLES2_ENABLED -DGLES1_ENABLED -D_LITTLE_ENDIAN -DNO_THREADS -DNO_FCNTL'))
	if env['bb10_exceptions']=="yes":
		env.Append(CCFLAGS = ['-fexceptions'])
	else:
		env.Append(CCFLAGS = ['-fno-exceptions'])

	#env.Append(LINKFLAGS = string.split()

	if (env["target"]=="release"):

		env.Append(CCFLAGS=['-O3','-DRELEASE_BUILD'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['-g', '-O0','-DDEBUG_ENABLED', '-D_DEBUG'])
		env.Append(LINKFLAGS=['-g'])

	env.Append(LIBS=['bps', 'pps', 'screen', 'socket', 'EGL', 'GLESv2', 'GLESv1_CM', 'm', 'asound'])

