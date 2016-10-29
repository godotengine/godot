
import os
import sys
import platform


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

	x11_error=os.system("pkg-config xinerama --modversion > /dev/null ")
	if (x11_error):
		print("xinerama not found.. x11 disabled.")
		return False

	x11_error=os.system("pkg-config xrandr --modversion > /dev/null ")
	if (x11_error):
			print("xrandr not found.. x11 disabled.")
			return False


	return True # X11 enabled

def get_opts():

	return [
	('use_llvm','Use llvm compiler','no'),
	('use_static_cpp','link stdc++ statically','no'),
	('use_sanitizer','Use llvm compiler sanitize address','no'),
	('use_leak_sanitizer','Use llvm compiler sanitize memory leaks','no'),
	('pulseaudio','Detect & Use pulseaudio','yes'),
	('udev','Use udev for gamepad connection callbacks','no'),
	('debug_release', 'Add debug symbols to release version','no'),
	]

def get_flags():

	return [
	("openssl", "system"),
	('freetype', 'system'),
	('libpng', 'system'),
	]



def configure(env):

	is64=sys.maxsize > 2**32

	if (env["bits"]=="default"):
		if (is64):
			env["bits"]="64"
		else:
			env["bits"]="32"

	env.Append(CPPPATH=['#platform/x11'])
	if (env["use_llvm"]=="yes"):
		if 'clang++' not in env['CXX']:
			env["CC"]="clang"
			env["CXX"]="clang++"
			env["LD"]="clang++"
		env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
		env.extra_suffix=".llvm"

	if (env["use_sanitizer"]=="yes"):
		env.Append(CXXFLAGS=['-fsanitize=address','-fno-omit-frame-pointer'])
		env.Append(LINKFLAGS=['-fsanitize=address'])
		env.extra_suffix+="s"

	if (env["use_leak_sanitizer"]=="yes"):
		env.Append(CXXFLAGS=['-fsanitize=address','-fno-omit-frame-pointer'])
		env.Append(LINKFLAGS=['-fsanitize=address'])
		env.extra_suffix+="s"


	#if (env["tools"]=="no"):
	#	#no tools suffix
	#	env['OBJSUFFIX'] = ".nt"+env['OBJSUFFIX']
	#	env['LIBSUFFIX'] = ".nt"+env['LIBSUFFIX']


	if (env["target"]=="release"):

		if (env["debug_release"]=="yes"):
			env.Append(CCFLAGS=['-g2'])
		else:
			env.Append(CCFLAGS=['-O3','-ffast-math'])

	elif (env["target"]=="release_debug"):

		env.Append(CCFLAGS=['-O2','-ffast-math','-DDEBUG_ENABLED'])
		if (env["debug_release"]=="yes"):
			env.Append(CCFLAGS=['-g2'])

	elif (env["target"]=="debug"):

		env.Append(CCFLAGS=['-g2', '-Wall','-DDEBUG_ENABLED','-DDEBUG_MEMORY_ENABLED'])

	env.ParseConfig('pkg-config x11 --cflags --libs')
	env.ParseConfig('pkg-config xinerama --cflags --libs')
	env.ParseConfig('pkg-config xcursor --cflags --libs')
	env.ParseConfig('pkg-config xrandr --cflags --libs')

	if (env["openssl"] == "system"):
		env.ParseConfig('pkg-config openssl --cflags --libs')

	if (env["libwebp"] == "system"):
		env.ParseConfig('pkg-config libwebp --cflags --libs')

	if (env["freetype"] == "system"):
		env["libpng"] = "system"  # Freetype links against libpng
		env.ParseConfig('pkg-config freetype2 --cflags --libs')

	if (env["libpng"] == "system"):
		env.ParseConfig('pkg-config libpng --cflags --libs')

	if (env["enet"] == "system"):
		env.ParseConfig('pkg-config libenet --cflags --libs')

	if (env["squish"] == "system" and env["tools"] == "yes"):
		env.ParseConfig('pkg-config libsquish --cflags --libs')

	# Sound and video libraries
	# Keep the order as it triggers chained dependencies (ogg needed by others, etc.)

	if (env["libtheora"] == "system"):
		env["libogg"] = "system"  # Needed to link against system libtheora
		env["libvorbis"] = "system"  # Needed to link against system libtheora
		env.ParseConfig('pkg-config theora theoradec --cflags --libs')

	if (env["libvorbis"] == "system"):
		env["libogg"] = "system"  # Needed to link against system libvorbis
		env.ParseConfig('pkg-config vorbis vorbisfile --cflags --libs')

	if (env["opus"] == "system"):
		env["libogg"] = "system"  # Needed to link against system opus
		env.ParseConfig('pkg-config opus opusfile --cflags --libs')

	if (env["libogg"] == "system"):
		env.ParseConfig('pkg-config ogg --cflags --libs')


	env.Append(CPPFLAGS=['-DOPENGL_ENABLED'])

	if (env["glew"] == "system"):
		env.ParseConfig('pkg-config glew --cflags --libs')

	if os.system("pkg-config --exists alsa")==0:
		print("Enabling ALSA")
		env.Append(CPPFLAGS=["-DALSA_ENABLED"])
		env.ParseConfig('pkg-config alsa --cflags --libs')
	else:
		print("ALSA libraries not found, disabling driver")

	if (platform.system() == "Linux"):
		env.Append(CPPFLAGS=["-DJOYDEV_ENABLED"])
	if (env["udev"]=="yes"):
		# pkg-config returns 0 when the lib exists...
		found_udev = not os.system("pkg-config --exists libudev")

		if (found_udev):
			print("Enabling udev support")
			env.Append(CPPFLAGS=["-DUDEV_ENABLED"])
			env.ParseConfig('pkg-config libudev --cflags --libs')
		else:
			print("libudev development libraries not found, disabling udev support")

	if (env["pulseaudio"]=="yes"):
		if not os.system("pkg-config --exists libpulse-simple"):
			print("Enabling PulseAudio")
			env.Append(CPPFLAGS=["-DPULSEAUDIO_ENABLED"])
			env.ParseConfig('pkg-config --cflags --libs libpulse-simple')
		else:
			print("PulseAudio development libraries not found, disabling driver")

	env.Append(CPPFLAGS=['-DX11_ENABLED','-DUNIX_ENABLED','-DGLES2_ENABLED','-DGLES_OVER_GL'])
	env.Append(LIBS=['GL', 'pthread', 'z'])
	if (platform.system() == "Linux"):
		env.Append(LIBS='dl')
	#env.Append(CPPFLAGS=['-DMPC_FIXED_POINT'])

#host compiler is default..

	if (is64 and env["bits"]=="32"):
		env.Append(CPPFLAGS=['-m32'])
		env.Append(LINKFLAGS=['-m32','-L/usr/lib/i386-linux-gnu'])
	elif (not is64 and env["bits"]=="64"):
		env.Append(CPPFLAGS=['-m64'])
		env.Append(LINKFLAGS=['-m64','-L/usr/lib/i686-linux-gnu'])


	import methods

	env.Append( BUILDERS = { 'GLSL120' : env.Builder(action = methods.build_legacygl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL' : env.Builder(action = methods.build_glsl_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	env.Append( BUILDERS = { 'GLSL120GLES' : env.Builder(action = methods.build_gles2_headers, suffix = 'glsl.h',src_suffix = '.glsl') } )
	#env.Append( BUILDERS = { 'HLSL9' : env.Builder(action = methods.build_hlsl_dx9_headers, suffix = 'hlsl.h',src_suffix = '.hlsl') } )

	if (env["use_static_cpp"]=="yes"):
		env.Append(LINKFLAGS=['-static-libstdc++'])

	list_of_x86 = ['x86_64', 'x86', 'i386', 'i586']
	if any(platform.machine() in s for s in list_of_x86):
		env["x86_libtheora_opt_gcc"]=True

