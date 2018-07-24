import os
import sys

def is_active():
	return True


def get_name():
	return "Sailfish OS"


def can_build():

	if (os.name != "posix"):
		return False

	if (os.uname()[1] != "SailfishSDK"):
		return False

	errorval = os.system("pkg-config --version > /dev/null")
	if (errorval):
		print("pkg-config not found.")
		return False

	sailfishapp_error = os.system("pkg-config sailfishapp --modversion > /dev/null ")
	if (sailfishapp_error):
		print("libsailfishapp not found.")
		return False

	audioresource_error = os.system("pkg-config audioresource --modversion > /dev/null ")
	if (audioresource_error):
		print("libaudioresource not found.")
		return False

	qt5gui_error = os.system("pkg-config Qt5Gui --modversion > /dev/null ")
	if (qt5gui_error):
		print("Qt5Gui not found.")
		return False

	return True


def get_opts():

	return [
		('pulseaudio', 'Detect & Use pulseaudio', 'yes'),
		('debug_release', 'Add debug symbols to release version', 'no')
	]


def get_flags():

	return [
		('builtin_libpng', 'no'),
		('builtin_zlib', 'no')
	]


def configure(env):

	is64 = sys.maxsize > 2**32

	if (env["bits"] == "default"):
		if (is64):
			env["bits"] = "64"
		else:
			env["bits"] = "32"

	env.Append(CPPPATH=['#platform/sailfish'])

	if (env["target"] == "release"):

		if (env["debug_release"] == "yes"):
			env.Append(CCFLAGS=['-g2'])
		else:
			env.Append(CCFLAGS=['-O3', '-ffast-math'])

	elif (env["target"] == "release_debug"):

		env.Append(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED'])
		if (env["debug_release"] == "yes"):
			env.Append(CCFLAGS=['-g2'])

	elif (env["target"] == "debug"):

		env.Append(CCFLAGS=['-g2', '-Wall', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])

	if (env["pulseaudio"] == "yes"):
		if not os.system("pkg-config --exists libpulse-simple"):
			print("Enabling PulseAudio")
			env.Append(CPPFLAGS=["-DPULSEAUDIO_ENABLED"])
			env.ParseConfig('pkg-config --cflags --libs libpulse-simple')
		else:
			print("PulseAudio development libraries not found, disabling driver")

	if (env['builtin_libpng'] == 'yes' or env['builtin_zlib'] == 'yes'):
		env['builtin_libpng'] = 'yes'
		env['builtin_zlib'] = 'yes'

	if (env['builtin_libpng'] == 'no'):
		env.ParseConfig('pkg-config libpng --cflags --libs')

	if (env['builtin_zlib'] == 'no'):
		env.ParseConfig('pkg-config zlib --cflags --libs')

	env.ParseConfig('pkg-config sailfishapp --cflags --libs')
	env.ParseConfig('pkg-config audioresource --cflags --libs')
	env.ParseConfig('pkg-config Qt5Gui --cflags --libs')
	env.Append(CPPFLAGS=['-DUNIX_ENABLED', '-DOPENGL_ENABLED', '-DGLES2_ENABLED'])
	env.Append(LIBS=['dl', 'GLESv2', 'pthread'])

	import methods

	env.Append(BUILDERS={'GLSL120': env.Builder(action=methods.build_legacygl_headers, suffix='glsl.h', src_suffix='.glsl')})
	env.Append(BUILDERS={'GLSL': env.Builder(action=methods.build_glsl_headers, suffix='glsl.h', src_suffix='.glsl')})
	env.Append(BUILDERS={'GLSL120GLES': env.Builder(action=methods.build_gles2_headers, suffix='glsl.h', src_suffix='.glsl')})
