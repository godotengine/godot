# IJG auto-configuration source file.
# Process this file with autoconf to produce a configure script.

#
# Configure script for IJG libjpeg
#

AC_INIT([libjpeg], [9.6.0])

# Directory where autotools helper scripts lives.
AC_CONFIG_AUX_DIR([.])

# Generate configuration headers.
AC_CONFIG_HEADERS([jconfig.h:jconfig.cfg])

# Hack: disable autoheader so that it doesn't overwrite our cfg template.
AUTOHEADER="echo autoheader ignored"

# Check system type
AC_CANONICAL_TARGET

# Initialize Automake
# Don't require all the GNU mandated files
AM_INIT_AUTOMAKE([-Wall -Werror no-dist foreign])

# Make --enable-silent-rules the default.
# To get verbose build output you may configure
# with --disable-silent-rules or use "make V=1".
AM_SILENT_RULES([yes])

# Add configure option --enable-maintainer-mode which enables
# dependency checking and generation useful to package maintainers.
# This is made an option to avoid confusing end users.
AM_MAINTAINER_MODE

# Check for programs
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_LN_S
AC_PROG_EGREP
AM_PROG_AR

# Check if LD supports linker scripts,
# and define automake conditional HAVE_LD_VERSION_SCRIPT if so.
AC_ARG_ENABLE([ld-version-script],
  AS_HELP_STRING([--enable-ld-version-script],
    [enable linker version script (default is enabled when possible)]),
    [have_ld_version_script=$enableval], [])
if test -z "$have_ld_version_script"; then
  AC_MSG_CHECKING([if LD -Wl,--version-script works])
  save_LDFLAGS="$LDFLAGS"
  LDFLAGS="$LDFLAGS -Wl,--version-script=conftest.map"
  cat > conftest.map <<EOF
VERS_1 {
        global: sym;
};

VERS_2 {
        global: sym;
} VERS_1;
EOF
  AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])],
                 [have_ld_version_script=yes], [have_ld_version_script=no])
  rm -f conftest.map
  LDFLAGS="$save_LDFLAGS"
  AC_MSG_RESULT($have_ld_version_script)
fi
AM_CONDITIONAL(HAVE_LD_VERSION_SCRIPT, test "$have_ld_version_script" = "yes")

# See if compiler supports prototypes.
AC_MSG_CHECKING([for function prototypes])
AC_CACHE_VAL([ijg_cv_have_prototypes],
[AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
int testfunction (int arg1, int * arg2); /* check prototypes */
struct methods_struct {		/* check method-pointer declarations */
  int (*error_exit) (char *msgtext);
  int (*trace_message) (char *msgtext);
  int (*another_method) (void);
};
int testfunction (int arg1, int * arg2) /* check definitions */
{ return arg2[arg1]; }
int test2function (void)	/* check void arg list */
{ return 0; }
]])],
        [ijg_cv_have_prototypes=yes],
        [ijg_cv_have_prototypes=no])])
AC_MSG_RESULT([$ijg_cv_have_prototypes])
if test $ijg_cv_have_prototypes = yes; then
  AC_DEFINE([HAVE_PROTOTYPES],[1],[Compiler supports function prototypes.])
else
  AC_MSG_WARN([Your compiler does not seem to know about function prototypes.
    Perhaps it needs a special switch to enable ANSI C mode.
    If so, we recommend running configure like this:
       ./configure  CC='cc -switch'
    where -switch is the proper switch.])
fi

# Check header files
AC_CHECK_HEADERS([stddef.h stdlib.h locale.h])
AC_CHECK_HEADER([string.h], [],
 [AC_DEFINE([NEED_BSD_STRINGS], [1],
            [Compiler has <strings.h> rather than standard <string.h>.])])

# See whether type size_t is defined in any ANSI-standard places;
# if not, perhaps it is defined in <sys/types.h>.
AC_MSG_CHECKING([for size_t])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#ifdef HAVE_STDDEF_H
#include <stddef.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#ifdef NEED_BSD_STRINGS
#include <strings.h>
#else
#include <string.h>
#endif
typedef size_t my_size_t;
]], [[ my_size_t foovar; ]])],
     [ijg_size_t_ok=yes],
     [ijg_size_t_ok="not ANSI, perhaps it is in sys/types.h"])
AC_MSG_RESULT([$ijg_size_t_ok])
if test "$ijg_size_t_ok" != yes; then
  AC_CHECK_HEADER([sys/types.h],
     [AC_DEFINE([NEED_SYS_TYPES_H], [1],
                [Need to include <sys/types.h> in order to obtain size_t.])
      AC_EGREP_CPP([size_t], [#include <sys/types.h>],
                   [ijg_size_t_ok="size_t is in sys/types.h"],
                   [ijg_size_t_ok=no])],
     [ijg_size_t_ok=no])
  AC_MSG_RESULT([$ijg_size_t_ok])
  if test "$ijg_size_t_ok" = no; then
    AC_MSG_WARN([Type size_t is not defined in any of the usual places.
      Try putting '"typedef unsigned int size_t;"' in jconfig.h.])
  fi
fi

# Check compiler characteristics
AC_MSG_CHECKING([for type unsigned char])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[ unsigned char un_char; ]])],
  [AC_MSG_RESULT(yes)
   AC_DEFINE([HAVE_UNSIGNED_CHAR], [1],
             [Compiler supports 'unsigned char'.])],
  [AC_MSG_RESULT(no)])

AC_MSG_CHECKING([for type unsigned short])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[ unsigned short un_short; ]])],
  [AC_MSG_RESULT(yes)
   AC_DEFINE([HAVE_UNSIGNED_SHORT], [1],
             [Compiler supports 'unsigned short'.])],
  [AC_MSG_RESULT(no)])

AC_MSG_CHECKING([for type void])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
/* Caution: a C++ compiler will insist on valid prototypes */
typedef void * void_ptr;	/* check void * */
#ifdef HAVE_PROTOTYPES		/* check ptr to function returning void */
typedef void (*void_func) (int a, int b);
#else
typedef void (*void_func) ();
#endif

#ifdef HAVE_PROTOTYPES		/* check void function result */
void test3function (void_ptr arg1, void_func arg2)
#else
void test3function (arg1, arg2)
     void_ptr arg1;
     void_func arg2;
#endif
{
  char * locptr = (char *) arg1; /* check casting to and from void * */
  arg1 = (void *) locptr;
  (*arg2) (1, 2);		/* check call of fcn returning void */
}
]], [[ ]])],
        [AC_MSG_RESULT(yes)],
        [AC_MSG_RESULT(no)
         AC_DEFINE([void], [char],
                   [Define 'void' as 'char' for archaic compilers
                    that don't understand it.])])
AC_C_CONST

# Check for non-broken inline under various spellings
AC_MSG_CHECKING([for inline])
ijg_cv_inline=""
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[} __inline__ int foo() { return 0; }
int bar() { return foo();]])], ijg_cv_inline="__inline__",
[AC_COMPILE_IFELSE([AC_LANG_PROGRAM(, [[} __inline int foo() { return 0; }
int bar() { return foo();]])], ijg_cv_inline="__inline",
[AC_COMPILE_IFELSE([AC_LANG_PROGRAM(, [[} inline int foo() { return 0; }
int bar() { return foo();]])], ijg_cv_inline="inline")])])
AC_MSG_RESULT($ijg_cv_inline)
AC_DEFINE_UNQUOTED([INLINE], [$ijg_cv_inline],
                   [How to obtain function inlining.])

# We cannot check for bogus warnings, but at least we can check for errors
AC_MSG_CHECKING([for broken incomplete types])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
typedef struct undefined_structure * undef_struct_ptr;
]], [[]])],
        [AC_MSG_RESULT(ok)],
        [AC_MSG_RESULT(broken)
         AC_DEFINE([INCOMPLETE_TYPES_BROKEN], [1],
                   [Compiler does not support pointers to unspecified
                    structures.])])

# Test whether global names are unique to at least 15 chars
AC_MSG_CHECKING([for short external names])
AC_LINK_IFELSE([AC_LANG_PROGRAM([[
int possibly_duplicate_function () { return 0; }
int possibly_dupli_function () { return 1; }
]], [[]])],
        [AC_MSG_RESULT(ok)],
        [AC_MSG_RESULT(short)
         AC_DEFINE([NEED_SHORT_EXTERNAL_NAMES], [1],
                   [Linker requires that global names be unique in
                    first 15 characters.])])

# Run-time checks
AC_MSG_CHECKING([to see if char is signed])
AC_RUN_IFELSE([AC_LANG_SOURCE([[
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#ifdef HAVE_PROTOTYPES
int is_char_signed (int arg)
#else
int is_char_signed (arg)
     int arg;
#endif
{
  if (arg == 189) {		/* expected result for unsigned char */
    return 0;			/* type char is unsigned */
  }
  else if (arg != -67) {	/* expected result for signed char */
    printf("Hmm, it seems 'char' is not eight bits wide on your machine.\n");
    printf("I fear the JPEG software will not work at all.\n\n");
  }
  return 1;			/* assume char is signed otherwise */
}
char signed_char_check = (char) (-67);
int main() {
  exit(is_char_signed((int) signed_char_check));
}]])], [AC_MSG_RESULT(no)
    AC_DEFINE([CHAR_IS_UNSIGNED], [1],
              [Characters are unsigned])],
              [AC_MSG_RESULT(yes)],
[AC_MSG_WARN([Assuming that char is signed on target machine.
    If it is unsigned, this will be a little bit inefficient.])
])

AC_MSG_CHECKING([to see if right shift is signed])
AC_RUN_IFELSE([AC_LANG_SOURCE([[
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#ifdef HAVE_PROTOTYPES
int is_shifting_signed (long arg)
#else
int is_shifting_signed (arg)
    long arg;
#endif
/* See whether right-shift on a long is signed or not. */
{
  long res = arg >> 4;

  if (res == -0x7F7E80CL) {	/* expected result for signed shift */
    return 1;			/* right shift is signed */
  }
  /* see if unsigned-shift hack will fix it. */
  /* we can't just test exact value since it depends on width of long... */
  res |= (~0L) << (32-4);
  if (res == -0x7F7E80CL) {	/* expected result now? */
    return 0;			/* right shift is unsigned */
  }
  printf("Right shift isn't acting as I expect it to.\n");
  printf("I fear the JPEG software will not work at all.\n\n");
  return 0;			/* try it with unsigned anyway */
}
int main() {
  exit(is_shifting_signed(-0x7F7E80B1L));
}]])],
        [AC_MSG_RESULT(no)
         AC_DEFINE([RIGHT_SHIFT_IS_UNSIGNED], [1],
                   [Broken compiler shifts signed values as an unsigned shift.])],
        [AC_MSG_RESULT(yes)],
        [AC_MSG_RESULT(Assuming that right shift is signed on target machine.)])

AC_MSG_CHECKING([to see if fopen accepts b spec])
AC_RUN_IFELSE([AC_LANG_SOURCE([[
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
int main() {
  if (fopen("conftestdata", "wb") != NULL)
    exit(0);
  exit(1);
}]])],
        [AC_MSG_RESULT(yes)],
        [AC_MSG_RESULT(no)
         AC_DEFINE([DONT_USE_B_MODE], [1],
                   [Don't open files in binary mode.])],
        [AC_MSG_RESULT(Assuming that it does.)])

# Configure libtool
LT_INIT([win32-dll])

# Select memory manager depending on user input.
# If no "-enable-maxmem", use jmemnobs
MEMORYMGR='jmemnobs'
MAXMEM="no"
AC_ARG_ENABLE([maxmem],
[  --enable-maxmem[=N]     enable use of temp files, set max mem usage to N MB],
[MAXMEM="$enableval"])
dnl [# support --with-maxmem for backwards compatibility with IJG V5.]
dnl AC_ARG_WITH(maxmem, , MAXMEM="$withval")
if test "x$MAXMEM" = xyes; then
  MAXMEM=1
fi
if test "x$MAXMEM" != xno; then
  if test -n "`echo $MAXMEM | sed 's/[[0-9]]//g'`"; then
    AC_MSG_ERROR(non-numeric argument to --enable-maxmem)
  fi
  DEFAULTMAXMEM=`expr $MAXMEM \* 1048576`
  AC_DEFINE_UNQUOTED([DEFAULT_MAX_MEM], [${DEFAULTMAXMEM}],
                     [Maximum data space library will allocate.])
  AC_MSG_CHECKING([for 'tmpfile()'])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <stdio.h>]], [[ FILE * tfile = tmpfile(); ]])],
                 [AC_MSG_RESULT(yes)
                 MEMORYMGR='jmemansi'],
                 [AC_MSG_RESULT(no)
                 dnl if tmpfile is not present, must use jmemname.
                 MEMORYMGR='jmemname'

    # Test for the need to remove temporary files using a signal handler
    # (for cjpeg/djpeg)
    AC_DEFINE([NEED_SIGNAL_CATCHER], [1],
              [Need signal handler to clean up temporary files.])
    AC_MSG_CHECKING([for 'mktemp()'])
    AC_LINK_IFELSE([AC_LANG_PROGRAM([[]], [[ char fname[80]; mktemp(fname); ]])],
                   [AC_MSG_RESULT(yes)],
                   [AC_MSG_RESULT(no)
                    AC_DEFINE([NO_MKTEMP], [1],
                              [The mktemp() function is not available.])])])
fi
AC_SUBST([MEMORYMGR])

# Extract the library version IDs from jpeglib.h.
AC_MSG_CHECKING([libjpeg version number])
[major=`sed -ne 's/^#define JPEG_LIB_VERSION_MAJOR *\([0-9][0-9]*\).*$/\1/p' $srcdir/jpeglib.h`
minor=`sed -ne 's/^#define JPEG_LIB_VERSION_MINOR *\([0-9][0-9]*\).*$/\1/p' $srcdir/jpeglib.h`]
AC_SUBST([JPEG_LIB_VERSION], [`expr $major + $minor`:0:$minor])
AC_SUBST([JPEG_LIB_VERSION_MAJOR], [$major])
AC_SUBST([JPEG_LIB_VERSION_MINOR], [$minor])
AC_MSG_RESULT([$major.$minor.0])

AC_CONFIG_FILES([Makefile libjpeg.pc])
AC_OUTPUT
