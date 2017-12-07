AC_PREREQ(2.63)

# Version number. This is currently the only place.
m4_define([HIDAPI_MAJOR],   0)
m4_define([HIDAPI_MINOR],   8)
m4_define([HIDAPI_RELEASE], 0)
m4_define([HIDAPI_RC],      -rc1)
m4_define([VERSION_STRING], HIDAPI_MAJOR[.]HIDAPI_MINOR[.]HIDAPI_RELEASE[]HIDAPI_RC)

AC_INIT([hidapi],[VERSION_STRING],[alan@signal11.us])

# Library soname version
# Follow the following rules (particularly the ones in the second link):
#  http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
#  http://sourceware.org/autobook/autobook/autobook_91.html
lt_current="0"
lt_revision="0"
lt_age="0"
LTLDFLAGS="-version-info ${lt_current}:${lt_revision}:${lt_age}"

AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
AC_CONFIG_MACRO_DIR([m4])

m4_ifdef([AM_PROG_AR], [AM_PROG_AR])
LT_INIT

AC_PROG_CC
AC_PROG_CXX
AC_PROG_OBJC
PKG_PROG_PKG_CONFIG


m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

hidapi_lib_error() {
	echo ""
	echo "  Library $1 was not found on this system."
	echo "  Please install it and re-run ./configure"
	echo ""
	exit 1
}

hidapi_prog_error() {
	echo ""
	echo "  Program $1 was not found on this system."
	echo "  This program is part of $2."
	echo "  Please install it and re-run ./configure"
	echo ""
	exit 1
}

AC_MSG_CHECKING([operating system])
AC_MSG_RESULT($host)
case $host in
*-linux*)
	AC_MSG_RESULT([ (Linux back-end)])
	AC_DEFINE(OS_LINUX, 1, [Linux implementations])
	AC_SUBST(OS_LINUX)
	backend="linux"
	os="linux"
	threads="pthreads"

	# HIDAPI/hidraw libs
	PKG_CHECK_MODULES([libudev], [libudev], true, [hidapi_lib_error libudev])
	LIBS_HIDRAW_PR+=" $libudev_LIBS"
	CFLAGS_HIDRAW+=" $libudev_CFLAGS"

	# HIDAPI/libusb libs
	AC_CHECK_LIB([rt], [clock_gettime], [LIBS_LIBUSB_PRIVATE+=" -lrt"], [hidapi_lib_error librt])
	PKG_CHECK_MODULES([libusb], [libusb-1.0 >= 1.0.9], true, [hidapi_lib_error libusb-1.0])
	LIBS_LIBUSB_PRIVATE+=" $libusb_LIBS"
	CFLAGS_LIBUSB+=" $libusb_CFLAGS"
	;;
*-darwin*)
	AC_MSG_RESULT([ (Mac OS X back-end)])
	AC_DEFINE(OS_DARWIN, 1, [Mac implementation])
	AC_SUBST(OS_DARWIN)
	backend="mac"
	os="darwin"
	threads="pthreads"
	LIBS="${LIBS} -framework IOKit -framework CoreFoundation"
	;;
*-freebsd*)
	AC_MSG_RESULT([ (FreeBSD back-end)])
	AC_DEFINE(OS_FREEBSD, 1, [FreeBSD implementation])
	AC_SUBST(OS_FREEBSD)
	backend="libusb"
	os="freebsd"
	threads="pthreads"

	CFLAGS="$CFLAGS -I/usr/local/include"
	LDFLAGS="$LDFLAGS -L/usr/local/lib"
	LIBS="${LIBS}"
	AC_CHECK_LIB([usb], [libusb_init], [LIBS_LIBUSB_PRIVATE="${LIBS_LIBUSB_PRIVATE} -lusb"], [hidapi_lib_error libusb])
	AC_CHECK_LIB([iconv], [iconv_open], [LIBS_LIBUSB_PRIVATE="${LIBS_LIBUSB_PRIVATE} -liconv"], [hidapi_lib_error libiconv])
	echo libs_priv: $LIBS_LIBUSB_PRIVATE
	;;
*-kfreebsd*)
	AC_MSG_RESULT([ (kFreeBSD back-end)])
	AC_DEFINE(OS_KFREEBSD, 1, [kFreeBSD implementation])
	AC_SUBST(OS_KFREEBSD)
	backend="libusb"
	os="kfreebsd"
	threads="pthreads"

	AC_CHECK_LIB([usb], [libusb_init], [LIBS_LIBUSB_PRIVATE="${LIBS_LIBUSB_PRIVATE} -lusb"], [hidapi_lib_error libusb])
	echo libs_priv: $LIBS_LIBUSB_PRIVATE
	;;
*-mingw*)
	AC_MSG_RESULT([ (Windows back-end, using MinGW)])
	backend="windows"
	os="windows"
	threads="windows"
	win_implementation="mingw"
	;;
*-cygwin*)
	AC_MSG_RESULT([ (Windows back-end, using Cygwin)])
	backend="windows"
	os="windows"
	threads="windows"
	win_implementation="cygwin"
	;;
*)
	AC_MSG_ERROR([HIDAPI is not supported on your operating system yet])
esac

LIBS_HIDRAW="${LIBS} ${LIBS_HIDRAW_PR}"
LIBS_LIBUSB="${LIBS} ${LIBS_LIBUSB_PRIVATE}"
AC_SUBST([LIBS_HIDRAW])
AC_SUBST([LIBS_LIBUSB])
AC_SUBST([CFLAGS_LIBUSB])
AC_SUBST([CFLAGS_HIDRAW])

if test "x$os" = xwindows; then
	AC_DEFINE(OS_WINDOWS, 1, [Windows implementations])
	AC_SUBST(OS_WINDOWS)
	LDFLAGS="${LDFLAGS} -no-undefined"
	LIBS="${LIBS} -lsetupapi"
fi

if test "x$threads" = xpthreads; then
	AX_PTHREAD([found_pthreads=yes], [found_pthreads=no])

	if test "x$found_pthreads" = xyes; then
		if test "x$os" = xlinux; then
			# Only use pthreads for libusb implementation on Linux.
			LIBS_LIBUSB="$PTHREAD_LIBS $LIBS_LIBUSB"
			CFLAGS_LIBUSB="$CFLAGS_LIBUSB $PTHREAD_CFLAGS"
			# There's no separate CC on Linux for threading,
			# so it's ok that both implementations use $PTHREAD_CC
			CC="$PTHREAD_CC"
		else
			LIBS="$PTHREAD_LIBS $LIBS"
			CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
			CC="$PTHREAD_CC"
		fi
	fi
fi

# Test GUI
AC_ARG_ENABLE([testgui],
	[AS_HELP_STRING([--enable-testgui],
		[enable building of test GUI (default n)])],
	[testgui_enabled=$enableval],
	[testgui_enabled='no'])
AM_CONDITIONAL([BUILD_TESTGUI], [test "x$testgui_enabled" != "xno"])

# Configure the MacOS TestGUI app bundle
rm -Rf testgui/TestGUI.app
mkdir -p testgui/TestGUI.app
cp -R ${srcdir}/testgui/TestGUI.app.in/* testgui/TestGUI.app
chmod -R u+w testgui/TestGUI.app
mkdir testgui/TestGUI.app/Contents/MacOS/

if test "x$testgui_enabled" != "xno"; then
	if test "x$os" = xdarwin; then
		# On Mac OS, don't use pkg-config.
		AC_CHECK_PROG([foxconfig], [fox-config], [fox-config], false)
		if test "x$foxconfig" = "xfalse"; then
			hidapi_prog_error fox-config "FOX Toolkit"
		fi
		LIBS_TESTGUI+=`$foxconfig --libs`
		LIBS_TESTGUI+=" -framework Cocoa -L/usr/X11R6/lib"
		CFLAGS_TESTGUI+=`$foxconfig --cflags`
		OBJCFLAGS+=" -x objective-c++"
	elif test "x$os" = xwindows; then
		# On Windows, just set the paths for Fox toolkit
		if test "x$win_implementation" = xmingw; then
			CFLAGS_TESTGUI="-I\$(srcdir)/../../hidapi-externals/fox/include -g -c"
			LIBS_TESTGUI=" -mwindows \$(srcdir)/../../hidapi-externals/fox/lib/libFOX-1.6.a -lgdi32 -Wl,--enable-auto-import -static-libgcc -static-libstdc++ -lkernel32"
		else
			# Cygwin
			CFLAGS_TESTGUI="-DWIN32 -I\$(srcdir)/../../hidapi-externals/fox/include -g -c"
			LIBS_TESTGUI="\$(srcdir)/../../hidapi-externals/fox/lib/libFOX-cygwin-1.6.a -lgdi32 -Wl,--enable-auto-import -static-libgcc -static-libstdc++ -lkernel32"
		fi
	else
		# On Linux and FreeBSD platforms, use pkg-config to find fox.
		PKG_CHECK_MODULES([fox], [fox17], [], [PKG_CHECK_MODULES([fox], [fox])])
		LIBS_TESTGUI="${LIBS_TESTGUI} $fox_LIBS"
		if test "x$os" = xfreebsd; then
			LIBS_TESTGUI="${LIBS_TESTGUI} -L/usr/local/lib"
		fi
		CFLAGS_TESTGUI="${CFLAGS_TESTGUI} $fox_CFLAGS"
	fi
fi
AC_SUBST([LIBS_TESTGUI])
AC_SUBST([CFLAGS_TESTGUI])
AC_SUBST([backend])

# OS info for Automake
AM_CONDITIONAL(OS_LINUX, test "x$os" = xlinux)
AM_CONDITIONAL(OS_DARWIN, test "x$os" = xdarwin)
AM_CONDITIONAL(OS_FREEBSD, test "x$os" = xfreebsd)
AM_CONDITIONAL(OS_KFREEBSD, test "x$os" = xkfreebsd)
AM_CONDITIONAL(OS_WINDOWS, test "x$os" = xwindows)

AC_CONFIG_HEADERS([config.h])

if test "x$os" = "xlinux"; then
	AC_CONFIG_FILES([pc/hidapi-hidraw.pc])
	AC_CONFIG_FILES([pc/hidapi-libusb.pc])
else
	AC_CONFIG_FILES([pc/hidapi.pc])
fi

AC_SUBST(LTLDFLAGS)

AC_CONFIG_FILES([Makefile \
	hidtest/Makefile \
	libusb/Makefile \
	linux/Makefile \
	mac/Makefile \
	testgui/Makefile \
	windows/Makefile])
AC_OUTPUT
