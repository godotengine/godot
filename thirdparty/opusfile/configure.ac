# autoconf source script for generating configure

dnl The package_version file will be automatically synced to the git revision
dnl by the update_version script when configured in the repository, but will
dnl remain constant in tarball releases unless it is manually edited.
m4_define([CURRENT_VERSION],
          m4_esyscmd([ ./update_version 2>/dev/null || true
                       if test -e package_version; then
                           . ./package_version
                           printf "$PACKAGE_VERSION"
                       else
                           printf "unknown"
                       fi ]))

AC_INIT([opusfile],[CURRENT_VERSION],[opus@xiph.org])
AC_CONFIG_SRCDIR([src/opusfile.c])
AC_CONFIG_MACRO_DIR([m4])

AC_USE_SYSTEM_EXTENSIONS
AC_SYS_LARGEFILE

AM_INIT_AUTOMAKE([1.11 foreign no-define dist-zip subdir-objects])
AM_MAINTAINER_MODE([enable])
LT_INIT

m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

dnl Library versioning for libtool.
dnl Please update these for releases.
dnl CURRENT, REVISION, AGE
dnl - library source changed -> increment REVISION
dnl - interfaces added/removed/changed -> increment CURRENT, REVISION = 0
dnl - interfaces added -> increment AGE
dnl - interfaces removed -> AGE = 0

OP_LT_CURRENT=4
OP_LT_REVISION=5
OP_LT_AGE=4

AC_SUBST(OP_LT_CURRENT)
AC_SUBST(OP_LT_REVISION)
AC_SUBST(OP_LT_AGE)

CC_CHECK_CFLAGS_APPEND(
  [-std=c89 -pedantic -Wall -Wextra -Wno-parentheses -Wno-long-long])

# Platform-specific tweaks
case $host in
  *-mingw*)
    # -std=c89 causes some warnings under mingw.
    CC_CHECK_CFLAGS_APPEND([-U__STRICT_ANSI__])
    # We need WINNT>=0x501 (WindowsXP) for getaddrinfo/freeaddrinfo.
    # It's okay to define this even when HTTP support is disabled, as it only
    #  affects header declarations, not linking (unless we actually use some
    #  XP-only functions).
    AC_DEFINE_UNQUOTED(_WIN32_WINNT,0x501,
     [We need at least WindowsXP for getaddrinfo/freeaddrinfo])
    host_mingw=true
    ;;
esac
AM_CONDITIONAL(OP_WIN32, [test "$host_mingw" = "true"])

AC_ARG_ENABLE([assertions],
  AS_HELP_STRING([--enable-assertions], [Enable assertions in code]),,
  enable_assertions=no)

AS_IF([test "$enable_assertions" = "yes"], [
  AC_DEFINE([OP_ENABLE_ASSERTIONS], [1], [Enable assertions in code])
])

AC_ARG_ENABLE([http],
  AS_HELP_STRING([--disable-http], [Disable HTTP support]),,
  enable_http=yes)

AM_COND_IF(OP_WIN32, [
  AS_IF([test "$enable_http" != "no"], [
    AC_CHECK_HEADER([winsock2.h],, [
      AC_MSG_WARN([HTTP support requires a Winsock socket library.])
      enable_http=no
    ])
  ])
], [
  AS_IF([test "$enable_http" != "no"], [
    AC_CHECK_HEADER([sys/socket.h],, [
      AC_MSG_WARN([HTTP support requires a POSIX socket library.])
      enable_http=no
    ])
  ])
])

# HTTP support requires either clock_gettime or ftime.  clock_gettime is
# used only if time.h defines CLOCK_REALTIME and the function is available
# in the standard library; on platforms such as glibc < 2.17 where -lrt
# or another library would be required, ftime will be used.
AS_IF([test "$enable_http" != "no"], [
  AC_MSG_CHECKING([for clock_gettime])
  AC_LINK_IFELSE([
    AC_LANG_PROGRAM([[#include <time.h>]], [[
      struct timespec ts;
      return clock_gettime(CLOCK_REALTIME, &ts);
    ]])
  ], [
    AC_MSG_RESULT([yes])
    AC_DEFINE([OP_HAVE_CLOCK_GETTIME], [1],
      [Enable use of clock_gettime function])
  ], [
    AC_MSG_RESULT([no])
    AC_SEARCH_LIBS(ftime, [compat], , [enable_http=no])
  ])
])

m4_ifndef([PKG_PROG_PKG_CONFIG],
  [m4_fatal([Could not locate the pkg-config autoconf macros.
Please make sure pkg-config is installed and, if necessary, set the environment
variable ACLOCAL="aclocal -I/path/to/pkg.m4".])])

AS_IF([test "$enable_http" != "no"], [
   openssl="openssl"
   AC_DEFINE([OP_ENABLE_HTTP], [1], [Enable HTTP support])
   PKG_CHECK_MODULES([URL_DEPS], [openssl])
])
AM_CONDITIONAL(OP_ENABLE_HTTP, [test "$enable_http" != "no"])
AC_SUBST([openssl])

PKG_CHECK_MODULES([DEPS], [ogg >= 1.3 opus >= 1.0.1])

AC_ARG_ENABLE([fixed-point],
  AS_HELP_STRING([--enable-fixed-point], [Enable fixed-point calculation]),,
  enable_fixed_point=no)
AC_ARG_ENABLE([float],
  AS_HELP_STRING([--disable-float], [Disable floating-point API]),,
  enable_float=yes)

AS_IF([test "$enable_float" = "no"],
  [enable_fixed_point=yes
   AC_DEFINE([OP_DISABLE_FLOAT_API], [1], [Disable floating-point API])
  ]
)

AS_IF([test "$enable_fixed_point" = "yes"],
  [AC_DEFINE([OP_FIXED_POINT], [1], [Enable fixed-point calculation])],
  [dnl This only has to be tested for if float->fixed conversions are required
   saved_LIBS="$LIBS"
   AC_SEARCH_LIBS([lrintf], [m], [
      AC_DEFINE([OP_HAVE_LRINTF], [1], [Enable use of lrintf function])
      lrintf_notice="
    Library for lrintf() ......... ${ac_cv_search_lrintf}"
   ])
   LIBS="$saved_LIBS"
  ]
)

AC_ARG_ENABLE([examples],
  AS_HELP_STRING([--disable-examples], [Do not build example applications]),,
  enable_examples=yes)
AM_CONDITIONAL([OP_ENABLE_EXAMPLES], [test "$enable_examples" = "yes"])

AS_CASE(["$ac_cv_search_lrintf"],
  ["no"],[],
  ["none required"],[],
  [lrintf_lib="$ac_cv_search_lrintf"])

AC_SUBST([lrintf_lib])

CC_ATTRIBUTE_VISIBILITY([default], [
  CC_FLAG_VISIBILITY([CFLAGS="${CFLAGS} -fvisibility=hidden"])
])

dnl Check for doxygen
AC_ARG_ENABLE([doc],
  AS_HELP_STRING([--disable-doc], [Do not build API documentation]),,
  [enable_doc=yes]
)

AS_IF([test "$enable_doc" = "yes"], [
  AC_CHECK_PROG([HAVE_DOXYGEN], [doxygen], [yes], [no])
  AC_CHECK_PROG([HAVE_DOT], [dot], [yes], [no])
],[
  HAVE_DOXYGEN=no
])

AM_CONDITIONAL([HAVE_DOXYGEN], [test "$HAVE_DOXYGEN" = "yes"])

AC_CONFIG_FILES([
  Makefile
  opusfile.pc
  opusurl.pc
  opusfile-uninstalled.pc
  opusurl-uninstalled.pc
  doc/Doxyfile
])
AC_CONFIG_HEADERS([config.h])
AC_OUTPUT

AC_MSG_NOTICE([
------------------------------------------------------------------------
  $PACKAGE_NAME $PACKAGE_VERSION: Automatic configuration OK.

    Assertions ................... ${enable_assertions}

    HTTP support ................. ${enable_http}
    Fixed-point .................. ${enable_fixed_point}
    Floating-point API ........... ${enable_float}${lrintf_notice}

    Hidden visibility ............ ${cc_cv_flag_visibility}

    API code examples ............ ${enable_examples}
    API documentation ............ ${enable_doc}
------------------------------------------------------------------------
])
