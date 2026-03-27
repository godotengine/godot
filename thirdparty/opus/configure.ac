dnl Process this file with autoconf to produce a configure script. -*-m4-*-

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

AC_INIT([opus],[CURRENT_VERSION],[opus@xiph.org])

AC_CONFIG_SRCDIR(src/opus_encoder.c)
AC_CONFIG_MACRO_DIR([m4])

dnl enable silent rules on automake 1.11 and later
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# For libtool.
dnl Please update these for releases.
OPUS_LT_CURRENT=11
OPUS_LT_REVISION=1
OPUS_LT_AGE=11

AC_SUBST(OPUS_LT_CURRENT)
AC_SUBST(OPUS_LT_REVISION)
AC_SUBST(OPUS_LT_AGE)

AM_INIT_AUTOMAKE([no-define])
AM_MAINTAINER_MODE([enable])

AC_CANONICAL_HOST
LT_INIT
AC_PROG_CC
AM_PROG_CC_C_O
AC_C_CONST
AC_C_INLINE

AM_PROG_AS

AC_DEFINE([OPUS_BUILD], [], [This is a build of OPUS])

#Use a hacked up version of autoconf's AC_C_RESTRICT because it's not
#strong enough a test to detect old buggy versions of GCC (e.g. 2.95.3)
AC_CACHE_CHECK([for C/C++ restrict keyword], ac_cv_c_restrict,
  [ac_cv_c_restrict=no
   # The order here caters to the fact that C++ does not require restrict.
   for ac_kw in __restrict __restrict__ _Restrict restrict; do
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
      [[typedef int * int_ptr;
        int foo (int_ptr $ac_kw ip, int * $ac_kw baz[]) {
        return ip[0];
       }]],
      [[int s[1];
        int * $ac_kw t = s;
        t[0] = 0;
        return foo(t, (void *)0)]])],
      [ac_cv_c_restrict=$ac_kw])
     test "$ac_cv_c_restrict" != no && break
   done
  ])

AH_VERBATIM([restrict],
[/* Define to the equivalent of the C99 'restrict' keyword, or to
   nothing if this is not supported.  Do not define if restrict is
   supported directly.  */
#undef restrict
/* Work around a bug in Sun C++: it does not support _Restrict or
   __restrict__, even though the corresponding Sun C compiler ends up with
   "#define restrict _Restrict" or "#define restrict __restrict__" in the
   previous line.  Perhaps some future version of Sun C++ will work with
   restrict; if so, hopefully it defines __RESTRICT like Sun C does.  */
#if defined __SUNPRO_CC && !defined __RESTRICT
# define _Restrict
# define __restrict__
#endif])

case $ac_cv_c_restrict in
   restrict) ;;
   no) AC_DEFINE([restrict], []) ;;
   *)  AC_DEFINE_UNQUOTED([restrict], [$ac_cv_c_restrict]) ;;
esac

AC_MSG_CHECKING(for C99 variable-size arrays)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],
                   [[static int x; char a[++x]; a[sizeof a - 1] = 0; int N; return a[0];]])],
    [ has_var_arrays=yes
      use_alloca="no (using var arrays)"
      AC_DEFINE([VAR_ARRAYS], [1], [Use C99 variable-size arrays])
    ],[
      has_var_arrays=no
    ])
AC_MSG_RESULT([$has_var_arrays])

AS_IF([test "$has_var_arrays" = "no"],
  [
   AC_CHECK_HEADERS([alloca.h])
   AC_MSG_CHECKING(for alloca)
   AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <alloca.h>]],
                                      [[int foo=10; int *array = alloca(foo);]])],
     [ use_alloca=yes;
       AC_DEFINE([USE_ALLOCA], [], [Make use of alloca])
     ],[
       use_alloca=no
     ])
   AC_MSG_RESULT([$use_alloca])
  ])

LT_LIB_M

AC_ARG_ENABLE([fixed-point],
    [AS_HELP_STRING([--enable-fixed-point],
                    [compile without floating point (for machines without a fast enough FPU)])],,
    [enable_fixed_point=no])

AS_IF([test "$enable_fixed_point" = "yes"],[
  enable_float="no"
  AC_DEFINE([FIXED_POINT], [1], [Compile as fixed-point (for machines without a fast enough FPU)])
  PC_BUILD="fixed-point"
],[
  enable_float="yes";
  PC_BUILD="floating-point"
])

AM_CONDITIONAL([FIXED_POINT], [test "$enable_fixed_point" = "yes"])

AC_ARG_ENABLE([fixed-point-debug],
    [AS_HELP_STRING([--enable-fixed-point-debug], [debug fixed-point implementation])],,
    [enable_fixed_point_debug=no])

AS_IF([test "$enable_fixed_point_debug" = "yes"],[
  AC_DEFINE([FIXED_DEBUG], [1], [Debug fixed-point implementation])
])

AC_ARG_ENABLE([float_api],
    [AS_HELP_STRING([--disable-float-api],
                    [compile without the floating point API (for machines with no float library)])],,
    [enable_float_api=yes])

AM_CONDITIONAL([DISABLE_FLOAT_API], [test "$enable_float_api" = "no"])

AS_IF([test "$enable_float_api" = "no"],[
  AC_DEFINE([DISABLE_FLOAT_API], [1], [Do not build the float API])
])

AC_ARG_ENABLE([fixed-res24],
	      [AS_HELP_STRING([--disable-fixed-res24], [use 16-bit (instead of 24-bit) internal resolution for fixed-point implementation])],,
    [enable_fixed_res24=yes])

AS_IF([test "$enable_fixed_res24" = "yes"],[
  AC_DEFINE([ENABLE_RES24], [1], [24-bit internal resolution for fixed-point])
])

AC_ARG_ENABLE([qext],
	      [AS_HELP_STRING([--enable-qext], [enable scalable quality extension])],,
    [enable_qext=no])

AS_IF([test "$enable_qext" = "yes"],[
  AC_DEFINE([ENABLE_QEXT], [1], [Scalable quality extension])
])
AM_CONDITIONAL([ENABLE_QEXT], [test "$enable_qext" = "yes"])

AC_ARG_ENABLE([custom-modes],
    [AS_HELP_STRING([--enable-custom-modes], [enable non-Opus modes, e.g. 44.1 kHz & 2^n frames])],,
    [enable_custom_modes=no])

AS_IF([test "$enable_custom_modes" = "yes"],[
  AC_DEFINE([CUSTOM_MODES], [1], [Custom modes])
  PC_BUILD="$PC_BUILD, custom modes"
])

AC_ARG_ENABLE([opus-custom-api],
    [AS_HELP_STRING([--enable-opus-custom-api], [enable Opus custom API])],,
    [enable_opus_custom_api=no])

AS_IF([test "$enable_opus_custom_api" = "yes"],[
  AC_DEFINE([ENABLE_OPUS_CUSTOM_API], [1], [Opus custom API])
  PC_BUILD="$PC_BUILD, Opus custom API"
])

AM_CONDITIONAL([CUSTOM_MODES], [test "$enable_custom_modes" = "yes" || test "$enable_opus_custom_api" = "yes"])

AC_ARG_ENABLE([dred],
    [AS_HELP_STRING([--enable-dred], [use Deep REDundancy (DRED)])],,
    [enable_dred=no])

AS_IF([test "$enable_dred" = "yes"],[
  AC_DEFINE([ENABLE_DRED], [1], [DRED])
])
AM_CONDITIONAL([ENABLE_DRED], [test "$enable_dred" = "yes"])

AC_ARG_ENABLE([deep-plc],
    [AS_HELP_STRING([--enable-deep-plc], [use deep PLC for SILK])],,
    [enable_deep_plc=no])

AS_IF([test "$enable_deep_plc" = "yes" || test "$enable_dred" = "yes" || test "$enable_osce" = "yes" || test "$enable_osce_training_data" = "yes"],[
  AC_DEFINE([ENABLE_DEEP_PLC], [1], [Deep PLC])
])
AM_CONDITIONAL([ENABLE_DEEP_PLC], [test "$enable_deep_plc" = "yes" || test "$enable_dred" = "yes" || test "$enable_osce" = "yes" || test "$enable_osce_training_data" = "yes"])

AC_ARG_ENABLE([lossgen],
    [AS_HELP_STRING([--enable-lossgen], [build opus_demo with packet loss simulator])],,
    [enable_lossgen=no])

AS_IF([test "$enable_lossgen" = "yes"],[
  AC_DEFINE([ENABLE_LOSSGEN], [1], [LOSSGEN])
])
AM_CONDITIONAL([ENABLE_LOSSGEN], [test "$enable_lossgen" = "yes"])

has_float_approx=no
case "$host_cpu" in
i[[3456]]86 | x86_64 | arm* | aarch64* | powerpc64 | powerpc32 | ia64)
  has_float_approx=yes
  ;;
esac

AC_ARG_ENABLE([float-approx],
    [AS_HELP_STRING([--enable-float-approx], [enable fast approximations for floating point])],
    [if test "$enable_float_approx" = "yes"; then
       AC_MSG_WARN([Floating point approximations are not supported on all platforms.])
     fi
    ],
    [enable_float_approx=$has_float_approx])

AS_IF([test "$enable_float_approx" = "yes"],[
  AC_DEFINE([FLOAT_APPROX], [1], [Float approximations])
])

AC_ARG_ENABLE([asm],
    [AS_HELP_STRING([--disable-asm], [disable assembly optimizations])],,
    [enable_asm=yes])

AC_ARG_ENABLE([rtcd],
    [AS_HELP_STRING([--disable-rtcd], [disable run-time CPU capabilities detection])],,
    [enable_rtcd=yes])

AC_ARG_ENABLE([intrinsics],
    [AS_HELP_STRING([--disable-intrinsics], [disable intrinsics optimizations])],,
    [enable_intrinsics=yes])

rtcd_support=no
cpu_arm=no
cpu_x86=no

AS_IF([test x"${enable_asm}" = x"yes"],[
    inline_optimization="No inline ASM for your platform, please send patches"
    case $host_cpu in
      arm*)
        dnl Currently we only have asm for fixed-point
        #AS_IF([test "$enable_float" != "yes"],[
            cpu_arm=yes
            AC_DEFINE([OPUS_ARM_ASM], [],  [Make use of ARM asm optimization])
            AS_GCC_INLINE_ASSEMBLY(
                [inline_optimization="ARM"],
                [inline_optimization="disabled"]
            )
            AS_ASM_ARM_EDSP([OPUS_ARM_INLINE_EDSP=1],[OPUS_ARM_INLINE_EDSP=0])
            AS_ASM_ARM_MEDIA([OPUS_ARM_INLINE_MEDIA=1],
                [OPUS_ARM_INLINE_MEDIA=0])
            AS_ASM_ARM_NEON([OPUS_ARM_INLINE_NEON=1],[OPUS_ARM_INLINE_NEON=0])
            AS_IF([test x"$inline_optimization" = x"ARM"],[
                AM_CONDITIONAL([OPUS_ARM_INLINE_ASM],[true])
                AC_DEFINE([OPUS_ARM_INLINE_ASM], 1,
                    [Use generic ARMv4 inline asm optimizations])
                AS_IF([test x"$OPUS_ARM_INLINE_EDSP" = x"1"],[
                    AC_DEFINE([OPUS_ARM_INLINE_EDSP], [1],
                        [Use ARMv5E inline asm optimizations])
                    inline_optimization="$inline_optimization (EDSP)"
                ])
                AS_IF([test x"$OPUS_ARM_INLINE_MEDIA" = x"1"],[
                    AC_DEFINE([OPUS_ARM_INLINE_MEDIA], [1],
                        [Use ARMv6 inline asm optimizations])
                    inline_optimization="$inline_optimization (Media)"
                ])
                AS_IF([test x"$OPUS_ARM_INLINE_NEON" = x"1"],[
                    AC_DEFINE([OPUS_ARM_INLINE_NEON], 1,
                        [Use ARM NEON inline asm optimizations])
                    inline_optimization="$inline_optimization (NEON)"
                ])
            ])
            dnl We need Perl to translate RVCT-syntax asm to gas syntax.
            AC_CHECK_PROG([HAVE_PERL], perl, yes, no)
            AS_IF([test x"$HAVE_PERL" = x"yes"],[
                AM_CONDITIONAL([OPUS_ARM_EXTERNAL_ASM],[true])
                asm_optimization="ARM"
                AS_IF([test x"$OPUS_ARM_INLINE_EDSP" = x"1"], [
                    OPUS_ARM_PRESUME_EDSP=1
                    OPUS_ARM_MAY_HAVE_EDSP=1
                ],
                [
                    OPUS_ARM_PRESUME_EDSP=0
                    OPUS_ARM_MAY_HAVE_EDSP=0
                ])
                AS_IF([test x"$OPUS_ARM_INLINE_MEDIA" = x"1"], [
                    OPUS_ARM_PRESUME_MEDIA=1
                    OPUS_ARM_MAY_HAVE_MEDIA=1
                ],
                [
                    OPUS_ARM_PRESUME_MEDIA=0
                    OPUS_ARM_MAY_HAVE_MEDIA=0
                ])
                AS_IF([test x"$OPUS_ARM_INLINE_NEON" = x"1"], [
                    OPUS_ARM_PRESUME_NEON=1
                    OPUS_ARM_MAY_HAVE_NEON=1
                ],
                [
                    OPUS_ARM_PRESUME_NEON=0
                    OPUS_ARM_MAY_HAVE_NEON=0
                ])
                AS_IF([test x"$enable_rtcd" = x"yes"],[
                    AS_IF([test x"$OPUS_ARM_MAY_HAVE_EDSP" != x"1"],[
                        AC_MSG_NOTICE(
                          [Trying to force-enable armv5e EDSP instructions...])
                        AS_ASM_ARM_EDSP_FORCE([OPUS_ARM_MAY_HAVE_EDSP=1])
                    ])
                    AS_IF([test x"$OPUS_ARM_MAY_HAVE_MEDIA" != x"1"],[
                        AC_MSG_NOTICE(
                          [Trying to force-enable ARMv6 media instructions...])
                        AS_ASM_ARM_MEDIA_FORCE([OPUS_ARM_MAY_HAVE_MEDIA=1])
                    ])
                    AS_IF([test x"$OPUS_ARM_MAY_HAVE_NEON" != x"1"],[
                        AC_MSG_NOTICE(
                          [Trying to force-enable NEON instructions...])
                        AS_ASM_ARM_NEON_FORCE([OPUS_ARM_MAY_HAVE_NEON=1])
                    ])
                ])
                rtcd_support=
                AS_IF([test x"$OPUS_ARM_MAY_HAVE_EDSP" = x"1"],[
                    AC_DEFINE(OPUS_ARM_MAY_HAVE_EDSP, 1,
                        [Define if assembler supports EDSP instructions])
                    AS_IF([test x"$OPUS_ARM_PRESUME_EDSP" = x"1"],[
                        AC_DEFINE(OPUS_ARM_PRESUME_EDSP, 1,
                          [Define if binary requires EDSP instruction support])
                        asm_optimization="$asm_optimization (EDSP)"
                    ],
                        [rtcd_support="$rtcd_support (EDSP)"]
                    )
                ])
                AC_SUBST(OPUS_ARM_MAY_HAVE_EDSP)
                AS_IF([test x"$OPUS_ARM_MAY_HAVE_MEDIA" = x"1"],[
                    AC_DEFINE(OPUS_ARM_MAY_HAVE_MEDIA, 1,
                      [Define if assembler supports ARMv6 media instructions])
                    AS_IF([test x"$OPUS_ARM_PRESUME_MEDIA" = x"1"],[
                        AC_DEFINE(OPUS_ARM_PRESUME_MEDIA, 1,
                          [Define if binary requires ARMv6 media instruction support])
                        asm_optimization="$asm_optimization (Media)"
                    ],
                        [rtcd_support="$rtcd_support (Media)"]
                    )
                ])
                AC_SUBST(OPUS_ARM_MAY_HAVE_MEDIA)
                AS_IF([test x"$OPUS_ARM_MAY_HAVE_NEON" = x"1"],[
                    AC_DEFINE(OPUS_ARM_MAY_HAVE_NEON, 1,
                      [Define if compiler supports NEON instructions])
                    AS_IF([test x"$OPUS_ARM_PRESUME_NEON" = x"1"], [
                        AC_DEFINE(OPUS_ARM_PRESUME_NEON, 1,
                          [Define if binary requires NEON instruction support])
                        asm_optimization="$asm_optimization (NEON)"
                    ],
                        [rtcd_support="$rtcd_support (NEON)"]
                    )
                ])
                AC_SUBST(OPUS_ARM_MAY_HAVE_NEON)
                AS_IF([test x"$OPUS_ARM_MAY_HAVE_DOTPROD" = x"1"],[
                    AC_DEFINE(OPUS_ARM_MAY_HAVE_DOTPROD, 1,
                      [Define if compiler supports DOTPROD instructions])
                    AS_IF([test x"$OPUS_ARM_PRESUME_DOTPROD" = x"1"], [
                        AC_DEFINE(OPUS_ARM_PRESUME_DOTPROD, 1,
                          [Define if binary requires DOTPROD instruction support])
                        asm_optimization="$asm_optimization (DOTPROD)"
                    ],
                        [rtcd_support="$rtcd_support (DOTPROD)"]
                    )
                ])
                AC_SUBST(OPUS_ARM_MAY_HAVE_DOTPROD)
                dnl Make sure turning on RTCD gets us at least one
                dnl instruction set.
                AS_IF([test x"$rtcd_support" != x""],
                    [rtcd_support=ARM"$rtcd_support"],
                    [rtcd_support="no"]
                )
                AC_MSG_CHECKING([for apple style tools])
                AC_PREPROC_IFELSE([AC_LANG_PROGRAM([
#ifndef __APPLE__
#error 1
#endif],[])],
                    [AC_MSG_RESULT([yes]); ARM2GNU_PARAMS="--apple"],
                    [AC_MSG_RESULT([no]); ARM2GNU_PARAMS=""])
                AC_SUBST(ARM2GNU_PARAMS)
            ],
            [
                AC_MSG_WARN(
                  [*** ARM assembly requires perl -- disabling optimizations])
                asm_optimization="(missing perl dependency for ARM)"
            ])
        #])
        ;;
    esac
],[
   inline_optimization="disabled"
   asm_optimization="disabled"
])

AM_CONDITIONAL([OPUS_ARM_INLINE_ASM],
    [test x"${inline_optimization%% *}" = x"ARM"])
AM_CONDITIONAL([OPUS_ARM_EXTERNAL_ASM],
    [test x"${asm_optimization%% *}" = x"ARM"])

AM_CONDITIONAL([HAVE_SSE], [false])
AM_CONDITIONAL([HAVE_SSE2], [false])
AM_CONDITIONAL([HAVE_SSE4_1], [false])
AM_CONDITIONAL([HAVE_AVX2], [false])

m4_define([DEFAULT_X86_SSE_CFLAGS], [-msse])
m4_define([DEFAULT_X86_SSE2_CFLAGS], [-msse2])
m4_define([DEFAULT_X86_SSE4_1_CFLAGS], [-msse4.1])
m4_define([DEFAULT_X86_AVX2_CFLAGS], [-mavx -mfma -mavx2])
m4_define([DEFAULT_ARM_NEON_INTR_CFLAGS], [-mfpu=neon])
m4_define([DEFAULT_ARM_DOTPROD_INTR_CFLAGS], ["-march=armv8.2-a+dotprod"])
# With GCC on ARM32 softfp architectures (e.g. Android, or older Ubuntu) you need to specify
# -mfloat-abi=softfp for -mfpu=neon to work.  However, on ARM32 hardfp architectures (e.g. newer Ubuntu),
# this option will break things.

# As a heuristic, if host matches arm*eabi* but not arm*hf*, it's probably soft-float.
m4_define([DEFAULT_ARM_NEON_SOFTFP_INTR_CFLAGS], [-mfpu=neon -mfloat-abi=softfp])

AS_CASE([$host],
        [arm*hf*], [AS_VAR_SET([RESOLVED_DEFAULT_ARM_NEON_INTR_CFLAGS], "DEFAULT_ARM_NEON_INTR_CFLAGS")],
        [arm*eabi*], [AS_VAR_SET([RESOLVED_DEFAULT_ARM_NEON_INTR_CFLAGS], "DEFAULT_ARM_NEON_SOFTFP_INTR_CFLAGS")],
        [AS_VAR_SET([RESOLVED_DEFAULT_ARM_NEON_INTR_CFLAGS], "DEFAULT_ARM_NEON_INTR_CFLAGS")])

AC_ARG_VAR([X86_SSE_CFLAGS], [C compiler flags to compile SSE intrinsics @<:@default=]DEFAULT_X86_SSE_CFLAGS[@:>@])
AC_ARG_VAR([X86_SSE2_CFLAGS], [C compiler flags to compile SSE2 intrinsics @<:@default=]DEFAULT_X86_SSE2_CFLAGS[@:>@])
AC_ARG_VAR([X86_SSE4_1_CFLAGS], [C compiler flags to compile SSE4.1 intrinsics @<:@default=]DEFAULT_X86_SSE4_1_CFLAGS[@:>@])
AC_ARG_VAR([X86_AVX2_CFLAGS], [C compiler flags to compile AVX2 intrinsics @<:@default=]DEFAULT_X86_AVX2_CFLAGS[@:>@])
AC_ARG_VAR([ARM_NEON_INTR_CFLAGS], [C compiler flags to compile ARM NEON intrinsics @<:@default=]DEFAULT_ARM_NEON_INTR_CFLAGS / DEFAULT_ARM_NEON_SOFTFP_INTR_CFLAGS[@:>@])
AC_ARG_VAR([ARM_DOTPROD_INTR_CFLAGS], [C compiler flags to compile ARM DOTPROD intrinsics @<:@default=]DEFAULT_ARM_DOTPROD_INTR_CFLAGS[@:>@])

AS_VAR_SET_IF([X86_SSE_CFLAGS], [], [AS_VAR_SET([X86_SSE_CFLAGS], "DEFAULT_X86_SSE_CFLAGS")])
AS_VAR_SET_IF([X86_SSE2_CFLAGS], [], [AS_VAR_SET([X86_SSE2_CFLAGS], "DEFAULT_X86_SSE2_CFLAGS")])
AS_VAR_SET_IF([X86_SSE4_1_CFLAGS], [], [AS_VAR_SET([X86_SSE4_1_CFLAGS], "DEFAULT_X86_SSE4_1_CFLAGS")])
AS_VAR_SET_IF([X86_AVX2_CFLAGS], [], [AS_VAR_SET([X86_AVX2_CFLAGS], "DEFAULT_X86_AVX2_CFLAGS")])
AS_VAR_SET_IF([ARM_NEON_INTR_CFLAGS], [], [AS_VAR_SET([ARM_NEON_INTR_CFLAGS], ["$RESOLVED_DEFAULT_ARM_NEON_INTR_CFLAGS"])])
AS_VAR_SET_IF([ARM_DOTPROD_INTR_CFLAGS], [], [AS_VAR_SET([ARM_DOTPROD_INTR_CFLAGS], ["DEFAULT_ARM_DOTPROD_INTR_CFLAGS"])])

AC_DEFUN([OPUS_PATH_NE10],
   [
      AC_ARG_WITH(NE10,
                  AS_HELP_STRING([--with-NE10=PFX],[prefix where libNE10 is installed (optional)]),
                  NE10_prefix="$withval", NE10_prefix="")
      AC_ARG_WITH(NE10-libraries,
                  AS_HELP_STRING([--with-NE10-libraries=DIR],
                                 [directory where libNE10 library is installed (optional)]),
                  NE10_libraries="$withval", NE10_libraries="")
      AC_ARG_WITH(NE10-includes,
                  AS_HELP_STRING([--with-NE10-includes=DIR],
                                 [directory where libNE10 header files are installed (optional)]),
                  NE10_includes="$withval", NE10_includes="")

      if test "x$NE10_libraries" != "x" ; then
         NE10_LIBS="-L$NE10_libraries"
      elif test "x$NE10_prefix" = "xno" || test "x$NE10_prefix" = "xyes" ; then
         NE10_LIBS=""
      elif test "x$NE10_prefix" != "x" ; then
         NE10_LIBS="-L$NE10_prefix/lib"
      elif test "x$prefix" != "xNONE" ; then
         NE10_LIBS="-L$prefix/lib"
      fi

      if test "x$NE10_prefix" != "xno" ; then
         NE10_LIBS="$NE10_LIBS -lNE10"
      fi

      if test "x$NE10_includes" != "x" ; then
         NE10_CFLAGS="-I$NE10_includes"
      elif test "x$NE10_prefix" = "xno" || test "x$NE10_prefix" = "xyes" ; then
         NE10_CFLAGS=""
      elif test "x$NE10_prefix" != "x" ; then
         NE10_CFLAGS="-I$NE10_prefix/include"
      elif test "x$prefix" != "xNONE"; then
         NE10_CFLAGS="-I$prefix/include"
      fi

      AC_MSG_CHECKING(for NE10)
      save_CFLAGS="$CFLAGS"; CFLAGS="$CFLAGS $NE10_CFLAGS"
      save_LIBS="$LIBS"; LIBS="$LIBS $NE10_LIBS $LIBM"
      AC_LINK_IFELSE(
         [
            AC_LANG_PROGRAM(
               [[#include <NE10_dsp.h>
               ]],
               [[
                  ne10_fft_cfg_float32_t cfg;
                  cfg = ne10_fft_alloc_c2c_float32_neon(480);
               ]]
            )
         ],[
            HAVE_ARM_NE10=1
            AC_MSG_RESULT([yes])
         ],[
            HAVE_ARM_NE10=0
            AC_MSG_RESULT([no])
            NE10_CFLAGS=""
            NE10_LIBS=""
         ]
      )
      CFLAGS="$save_CFLAGS"; LIBS="$save_LIBS"
      #Now we know if libNE10 is installed or not
      AS_IF([test x"$HAVE_ARM_NE10" = x"1"],
         [
            AC_DEFINE([HAVE_ARM_NE10], 1, [NE10 library is installed on host. Make sure it is on target!])
            AC_SUBST(HAVE_ARM_NE10)
            AC_SUBST(NE10_CFLAGS)
            AC_SUBST(NE10_LIBS)
         ]
      )
   ]
)

AS_IF([test x"$enable_intrinsics" = x"yes"],[
   intrinsics_support=""
   AS_CASE([$host_cpu],
   [arm*|aarch64*],
   [
      cpu_arm=yes
      OPUS_CHECK_INTRINSICS(
         [ARM Neon],
         [$ARM_NEON_INTR_CFLAGS],
         [OPUS_ARM_MAY_HAVE_NEON_INTR],
         [OPUS_ARM_PRESUME_NEON_INTR],
         [[#include <arm_neon.h>
         ]],
         [[
            static float32x4_t A0, A1, SUMM;
            SUMM = vmlaq_f32(SUMM, A0, A1);
            return (int)vgetq_lane_f32(SUMM, 0);
         ]]
      )
      AS_IF([test x"$OPUS_ARM_MAY_HAVE_NEON_INTR" = x"1" && test x"$OPUS_ARM_PRESUME_NEON_INTR" != x"1"],
          [
             OPUS_ARM_NEON_INTR_CFLAGS="$ARM_NEON_INTR_CFLAGS"
             AC_SUBST([OPUS_ARM_NEON_INTR_CFLAGS])
          ]
      )

      AS_IF([test x"$OPUS_ARM_MAY_HAVE_NEON_INTR" = x"1"],
      [
         AC_DEFINE([OPUS_ARM_MAY_HAVE_NEON_INTR], 1, [Compiler supports ARMv7/Aarch64 Neon Intrinsics])
         intrinsics_support="$intrinsics_support (NEON)"

         AS_IF([test x"$enable_rtcd" != x"no" && test x"$OPUS_ARM_PRESUME_NEON_INTR" != x"1"],
            [AS_IF([test x"$rtcd_support" = x"no"],
               [rtcd_support="ARM (NEON Intrinsics)"],
               [rtcd_support="$rtcd_support (NEON Intrinsics)"])])

         AS_IF([test x"$OPUS_ARM_PRESUME_NEON_INTR" = x"1"],
            [AC_DEFINE([OPUS_ARM_PRESUME_NEON_INTR], 1, [Define if binary requires NEON intrinsics support])])

         OPUS_PATH_NE10()
         AS_IF([test x"$NE10_LIBS" != x""],
         [
              intrinsics_support="$intrinsics_support (NE10)"
              AS_IF([test x"enable_rtcd" != x"" \
               && test x"$OPUS_ARM_PRESUME_NEON_INTR" != x"1"],
                 [rtcd_support="$rtcd_support (NE10)"])
         ])

         OPUS_CHECK_INTRINSICS(
            [Aarch64 Neon],
            [$ARM_NEON_INTR_CFLAGS],
            [OPUS_ARM_MAY_HAVE_AARCH64_NEON_INTR],
            [OPUS_ARM_PRESUME_AARCH64_NEON_INTR],
            [[#include <arm_neon.h>
            ]],
            [[
               static int32_t IN;
               static int16_t OUT;
               OUT = vqmovns_s32(IN);
            ]]
         )

         AS_IF([test x"$OPUS_ARM_PRESUME_AARCH64_NEON_INTR" = x"1"],
         [
            AC_DEFINE([OPUS_ARM_PRESUME_AARCH64_NEON_INTR], 1, [Define if binary requires Aarch64 Neon Intrinsics])
            intrinsics_support="$intrinsics_support (NEON [Aarch64])"
         ])

         OPUS_CHECK_INTRINSICS(
            [Aarch64 dotprod],
	    [$ARM_DOTPROD_INTR_CFLAGS],
            [OPUS_ARM_MAY_HAVE_DOTPROD],
            [OPUS_ARM_PRESUME_DOTPROD],
            [[#include <arm_neon.h>
            ]],
            [[
               static int32x4_t acc;
               static int8x16_t a, b;
               acc = vdotq_s32(acc, a, b);
            ]]
         )
         AS_IF([test x"$OPUS_ARM_MAY_HAVE_DOTPROD" = x"1" && test x"$OPUS_ARM_PRESUME_DOTPROD" != x"1"],
             [
                OPUS_ARM_DOTPROD_INTR_CFLAGS="$ARM_NEON_DOTPROD_CFLAGS"
                AC_SUBST([OPUS_ARM_DOTPROD_INTR_CFLAGS])
             ]
         )

         AS_IF([test x"$OPUS_ARM_MAY_HAVE_DOTPROD" = x"1"],
             [
                AC_DEFINE([OPUS_ARM_MAY_HAVE_DOTPROD], 1, [Compiler supports Aarch64 DOTPROD Intrinsics])
                intrinsics_support="$intrinsics_support (DOTPROD)"

                AS_IF([test x"$OPUS_ARM_PRESUME_DOTPROD" = x"1"],
                [
                   AC_DEFINE([OPUS_ARM_PRESUME_DOTPROD], 1, [Define if binary requires Aarch64 dotprod Intrinsics])
                   intrinsics_support="$intrinsics_support (DOTPROD [Aarch64])"
                ])

                AS_IF([test x"$enable_rtcd" != x"no" && test x"$OPUS_ARM_PRESUME_DOTPROD" != x"1"],
                   [AS_IF([test x"$rtcd_support" = x"no"],
                      [rtcd_support="ARM (DOTPROD Intrinsics)"],
                      [rtcd_support="$rtcd_support (DOTPROD Intrinsics)"])])

             ]
         )


         AS_IF([test x"$intrinsics_support" = x""],
            [intrinsics_support=no],
            [intrinsics_support="ARM$intrinsics_support"])
      ],
      [
         AC_MSG_WARN([Compiler does not support ARM intrinsics])
         intrinsics_support=no
      ])
   ],
   [i?86|x86_64],
   [
      cpu_x86=yes
      OPUS_CHECK_INTRINSICS(
         [SSE],
         [$X86_SSE_CFLAGS],
         [OPUS_X86_MAY_HAVE_SSE],
         [OPUS_X86_PRESUME_SSE],
         [[#include <xmmintrin.h>
           #include <time.h>
         ]],
         [[
             __m128 mtest;
             mtest = _mm_set1_ps((float)time(NULL));
             mtest = _mm_mul_ps(mtest, mtest);
             return _mm_cvtss_si32(mtest);
         ]]
      )
      AS_IF([test x"$OPUS_X86_MAY_HAVE_SSE" = x"1" && test x"$OPUS_X86_PRESUME_SSE" != x"1"],
          [
             OPUS_X86_SSE_CFLAGS="$X86_SSE_CFLAGS"
             AC_SUBST([OPUS_X86_SSE_CFLAGS])
          ]
      )
      OPUS_CHECK_INTRINSICS(
         [SSE2],
         [$X86_SSE2_CFLAGS],
         [OPUS_X86_MAY_HAVE_SSE2],
         [OPUS_X86_PRESUME_SSE2],
         [[#include <emmintrin.h>
           #include <time.h>
         ]],
         [[
            __m128i mtest;
            mtest = _mm_set1_epi32((int)time(NULL));
            mtest = _mm_mul_epu32(mtest, mtest);
            return _mm_cvtsi128_si32(mtest);
         ]]
      )
      AS_IF([test x"$OPUS_X86_MAY_HAVE_SSE2" = x"1" && test x"$OPUS_X86_PRESUME_SSE2" != x"1"],
          [
             OPUS_X86_SSE2_CFLAGS="$X86_SSE2_CFLAGS"
             AC_SUBST([OPUS_X86_SSE2_CFLAGS])
          ]
      )
      OPUS_CHECK_INTRINSICS(
         [SSE4.1],
         [$X86_SSE4_1_CFLAGS],
         [OPUS_X86_MAY_HAVE_SSE4_1],
         [OPUS_X86_PRESUME_SSE4_1],
         [[#include <smmintrin.h>
           #include <time.h>
         ]],
         [[
            __m128i mtest;
            mtest = _mm_set1_epi32((int)time(NULL));
            mtest = _mm_mul_epi32(mtest, mtest);
            return _mm_cvtsi128_si32(mtest);
         ]]
      )
      AS_IF([test x"$OPUS_X86_MAY_HAVE_SSE4_1" = x"1" && test x"$OPUS_X86_PRESUME_SSE4_1" != x"1"],
          [
             OPUS_X86_SSE4_1_CFLAGS="$X86_SSE4_1_CFLAGS"
             AC_SUBST([OPUS_X86_SSE4_1_CFLAGS])
          ]
      )
      OPUS_CHECK_INTRINSICS(
         [AVX2],
         [$X86_AVX2_CFLAGS],
         [OPUS_X86_MAY_HAVE_AVX2],
         [OPUS_X86_PRESUME_AVX2],
         [[#include <immintrin.h>
           #include <time.h>
         ]],
         [[
             unsigned char utest[[16]] = {1};
             __m256 mtest;
             __m256i mtest1;
             __m256i mtest2;
             mtest = _mm256_set1_ps((float)time(NULL));
             mtest = _mm256_fmadd_ps(mtest, mtest, mtest);
             mtest1 = _mm256_set_m128i(_mm_loadu_si64(utest), _mm_loadu_si64(utest));
             mtest2 =
              _mm256_cvtepi16_epi32(_mm_loadu_si128((void*)utest));
             return _mm256_extract_epi16(_mm256_xor_si256(
              _mm256_xor_si256(mtest1, mtest2), _mm256_cvttps_epi32(mtest)), 0);
         ]]
      )
      AS_IF([test x"$OPUS_X86_MAY_HAVE_AVX2" = x"1" && test x"$OPUS_X86_PRESUME_AVX2" != x"1"],
          [
             OPUS_X86_AVX2_CFLAGS="$X86_AVX2_CFLAGS"
             AC_SUBST([OPUS_X86_AVX2_CFLAGS])
          ]
      )
         AS_IF([test x"$rtcd_support" = x"no"], [rtcd_support=""])
         AS_IF([test x"$OPUS_X86_MAY_HAVE_SSE" = x"1"],
         [
            AC_DEFINE([OPUS_X86_MAY_HAVE_SSE], 1, [Compiler supports X86 SSE Intrinsics])
            intrinsics_support="$intrinsics_support SSE"

            AS_IF([test x"$OPUS_X86_PRESUME_SSE" = x"1"],
               [AC_DEFINE([OPUS_X86_PRESUME_SSE], 1, [Define if binary requires SSE intrinsics support])],
               [rtcd_support="$rtcd_support SSE"])
         ],
         [
            AC_MSG_WARN([Compiler does not support SSE intrinsics])
         ])

         AS_IF([test x"$OPUS_X86_MAY_HAVE_SSE2" = x"1"],
         [
            AC_DEFINE([OPUS_X86_MAY_HAVE_SSE2], 1, [Compiler supports X86 SSE2 Intrinsics])
            intrinsics_support="$intrinsics_support SSE2"

            AS_IF([test x"$OPUS_X86_PRESUME_SSE2" = x"1"],
               [AC_DEFINE([OPUS_X86_PRESUME_SSE2], 1, [Define if binary requires SSE2 intrinsics support])],
               [rtcd_support="$rtcd_support SSE2"])
         ],
         [
            AC_MSG_WARN([Compiler does not support SSE2 intrinsics])
         ])

         AS_IF([test x"$OPUS_X86_MAY_HAVE_SSE4_1" = x"1"],
         [
            AC_DEFINE([OPUS_X86_MAY_HAVE_SSE4_1], 1, [Compiler supports X86 SSE4.1 Intrinsics])
            intrinsics_support="$intrinsics_support SSE4.1"

            AS_IF([test x"$OPUS_X86_PRESUME_SSE4_1" = x"1"],
               [AC_DEFINE([OPUS_X86_PRESUME_SSE4_1], 1, [Define if binary requires SSE4.1 intrinsics support])],
               [rtcd_support="$rtcd_support SSE4.1"])
         ],
         [
            AC_MSG_WARN([Compiler does not support SSE4.1 intrinsics])
         ])
         AS_IF([test x"$OPUS_X86_MAY_HAVE_AVX2" = x"1"],
         [
            AC_DEFINE([OPUS_X86_MAY_HAVE_AVX2], 1, [Compiler supports X86 AVX2 Intrinsics])
            intrinsics_support="$intrinsics_support AVX2"

            AS_IF([test x"$OPUS_X86_PRESUME_AVX2" = x"1"],
               [AC_DEFINE([OPUS_X86_PRESUME_AVX2], 1, [Define if binary requires AVX2 intrinsics support])],
               [rtcd_support="$rtcd_support AVX2"])
         ],
         [
            AC_MSG_WARN([Compiler does not support AVX2 intrinsics])
         ])

         AS_IF([test x"$intrinsics_support" = x""],
            [intrinsics_support=no],
            [intrinsics_support="x86$intrinsics_support"]
         )
         AS_IF([test x"$rtcd_support" = x""],
            [rtcd_support=no],
            [rtcd_support="x86$rtcd_support"],
        )

    AS_IF([test x"$enable_rtcd" = x"yes" && test x"$rtcd_support" != x""],[
            get_cpuid_by_asm="no"
            AC_MSG_CHECKING([How to get X86 CPU Info])
            AC_LINK_IFELSE([AC_LANG_PROGRAM([[
                 #include <stdio.h>
            ]],[[
                 unsigned int CPUInfo0;
                 unsigned int CPUInfo1;
                 unsigned int CPUInfo2;
                 unsigned int CPUInfo3;
                 unsigned int InfoType;
                #if defined(__i386__) && defined(__PIC__)
                 __asm__ __volatile__ (
                 "xchg %%ebx, %1\n"
                 "cpuid\n"
                 "xchg %%ebx, %1\n":
                 "=a" (CPUInfo0),
                 "=r" (CPUInfo1),
                 "=c" (CPUInfo2),
                 "=d" (CPUInfo3) :
                 "a" (InfoType), "c" (0)
                );
               #else
                 __asm__ __volatile__ (
                 "cpuid":
                 "=a" (CPUInfo0),
                 "=b" (CPUInfo1),
                 "=c" (CPUInfo2),
                 "=d" (CPUInfo3) :
                 "a" (InfoType), "c" (0)
                );
               #endif
            ]])],
            [get_cpuid_by_asm="yes"
             AC_MSG_RESULT([Inline Assembly])
                 AC_DEFINE([CPU_INFO_BY_ASM], [1], [Get CPU Info by asm method])],
             [AC_LINK_IFELSE([AC_LANG_PROGRAM([[
                 #include <cpuid.h>
            ]],[[
                 unsigned int CPUInfo0;
                 unsigned int CPUInfo1;
                 unsigned int CPUInfo2;
                 unsigned int CPUInfo3;
                 unsigned int InfoType;
                 __get_cpuid_count(InfoType, 0, &CPUInfo0, &CPUInfo1, &CPUInfo2, &CPUInfo3);
            ]])],
            [AC_MSG_RESULT([C method])
                 AC_DEFINE([CPU_INFO_BY_C], [1], [Get CPU Info by c method])],
            [AC_MSG_ERROR([no supported Get CPU Info method, please disable run-time CPU capabilities detection or intrinsics])])])])
   ],
   [
      AC_MSG_WARN([No intrinsics support for your architecture])
      intrinsics_support="no"
   ])
],
[
   intrinsics_support="no"
])

AM_CONDITIONAL([CPU_ARM], [test "$cpu_arm" = "yes"])
AM_CONDITIONAL([HAVE_ARM_DOTPROD],
    [test x"$OPUS_ARM_MAY_HAVE_DOTPROD" = x"1"])
AM_CONDITIONAL([HAVE_ARM_NEON_INTR],
    [test x"$OPUS_ARM_MAY_HAVE_NEON_INTR" = x"1"])
AM_CONDITIONAL([HAVE_ARM_NE10],
    [test x"$HAVE_ARM_NE10" = x"1"])
AM_CONDITIONAL([CPU_X86], [test "$cpu_x86" = "yes"])
AM_CONDITIONAL([HAVE_SSE],
    [test x"$OPUS_X86_MAY_HAVE_SSE" = x"1"])
AM_CONDITIONAL([HAVE_SSE2],
    [test x"$OPUS_X86_MAY_HAVE_SSE2" = x"1"])
AM_CONDITIONAL([HAVE_SSE4_1],
    [test x"$OPUS_X86_MAY_HAVE_SSE4_1" = x"1"])
AM_CONDITIONAL([HAVE_AVX2],
    [test x"$OPUS_X86_MAY_HAVE_AVX2" = x"1"])

AM_CONDITIONAL([HAVE_RTCD],
 [test x"$enable_rtcd" = x"yes" && test x"$rtcd_support" != x"no"])
AS_IF([test x"$enable_rtcd" = x"yes"],[
    AS_IF([test x"$rtcd_support" != x"no"],[
        AC_DEFINE([OPUS_HAVE_RTCD], [1],
            [Use run-time CPU capabilities detection])
        OPUS_HAVE_RTCD=1
        AC_SUBST(OPUS_HAVE_RTCD)
    ])
],[
    rtcd_support="disabled"
])

AC_ARG_ENABLE([assertions],
    [AS_HELP_STRING([--enable-assertions],[enable additional software error checking])],,
    [enable_assertions=no])

AS_IF([test "$enable_assertions" = "yes"], [
  AC_DEFINE([ENABLE_ASSERTIONS], [1], [Assertions])
])

AC_ARG_ENABLE([hardening],
    [AS_HELP_STRING([--disable-hardening],[disable run-time checks that are cheap and safe for use in production])],,
    [enable_hardening=yes])

AS_IF([test "$enable_hardening" = "yes"], [
  AC_DEFINE([ENABLE_HARDENING], [1], [Hardening])
])

AC_ARG_ENABLE([fuzzing],
	      [AS_HELP_STRING([--enable-fuzzing],[cause the encoder to make random decisions (do not use in production)])],,
    [enable_fuzzing=no])

AS_IF([test "$enable_fuzzing" = "yes"], [
  AC_DEFINE([FUZZING], [1], [Fuzzing])
])

AC_ARG_ENABLE([check-asm],
    [AS_HELP_STRING([--enable-check-asm],
                    [enable bit-exactness checks between optimized and C implementations])],,
    [enable_check_asm=no])

AS_IF([test "$enable_check_asm" = "yes"], [
  AC_DEFINE([OPUS_CHECK_ASM], [1], [Run bit-exactness checks between optimized and C implementations])
])

AC_ARG_ENABLE([doc],
    [AS_HELP_STRING([--disable-doc], [do not build API documentation])],,
    [enable_doc=yes])

AS_IF([test "$enable_doc" = "yes"], [
  AC_CHECK_PROG(HAVE_DOXYGEN, [doxygen], [yes], [no])
  AC_CHECK_PROG(HAVE_DOT, [dot], [yes], [no])
],[
  HAVE_DOXYGEN=no
])

AC_ARG_ENABLE([dot-product],
	      AS_HELP_STRING([--disable-dot-product], [disable dot product implementation]),,
  enable_dot_product=yes)

AS_IF([test "$enable_dot_product" = "no"], [
       AC_DEFINE([DISABLE_DOT_PROD], [1], [Disable dot product instructions])
])

AC_ARG_ENABLE([dnn-debug-float],
	      AS_HELP_STRING([--enable-dnn-debug-float], [use floating-point DNN computation everywhere]),,
  enable_dnn_debug_float=no)

AS_IF([test "$enable_dnn_debug_float" = "no"], [
       AC_DEFINE([DISABLE_DEBUG_FLOAT], [1], [Disable DNN debug float])
])

AC_ARG_ENABLE([osce-training-data],
  AS_HELP_STRING([--enable-osce-training-data], [enable feature output for SILK enhancement]),,
  [enable_osc_training_data=no]
)

AS_IF([test "$enable_osce_training_data" = "yes"], [
       AC_DEFINE([ENABLE_OSCE_TRAINING_DATA], [1], [Enable dumping of OSCE training data])
])

AC_MSG_CHECKING([argument osce training data])
AS_IF([test "$enable_osce_training_data" = "yes"], [
       AC_MSG_RESULT([yes])
], [AC_MSG_RESULT([no])])

AC_ARG_ENABLE([osce],
  AS_HELP_STRING([--enable-osce], [enable speech coding enhancement]),,
  [enable_osce=no]
)

AS_IF([test "$enable_osce" = "yes" || test "$enable_osce_training_data" = "yes"], [
       AC_DEFINE([ENABLE_OSCE], [1], [Enable Opus Speech Coding Enhancement])
       AC_DEFINE([ENABLE_OSCE_BWE], [1], [Enable Opus Speech Coding Enhancement Blind BWE])
])

AM_CONDITIONAL([ENABLE_OSCE], [test "$enable_osce" = "yes" || test "$enable_osce_training_data" = "yes"])

AS_IF([test "$enable_fixed_point" = "yes"], [
  AS_IF([test "$enable_dred" = "yes" || test "$enable_deep_plc" = "yes" || test "$enable_osce" = "yes"], [
    AC_MSG_ERROR([--enable-fixed-point cannot be used with --enable-deep-plc, --enable-dred, and --enable-osce.])
  ])
])

AM_CONDITIONAL([HAVE_DOXYGEN], [test "$HAVE_DOXYGEN" = "yes"])

AC_ARG_ENABLE([extra-programs],
    [AS_HELP_STRING([--disable-extra-programs], [do not build extra programs (demo and tests)])],,
    [enable_extra_programs=yes])

AM_CONDITIONAL([EXTRA_PROGRAMS], [test "$enable_extra_programs" = "yes"])


AC_ARG_ENABLE([rfc8251],
	      AS_HELP_STRING([--disable-rfc8251], [disable bitstream fixes from RFC 8251]),,
  [enable_rfc8251=yes])

AS_IF([test "$enable_rfc8251" = "no"], [
       AC_DEFINE([DISABLE_UPDATE_DRAFT], [1], [Disable bitstream fixes from RFC 8251])
])


saved_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -fvisibility=hidden"
AC_MSG_CHECKING([if ${CC} supports -fvisibility=hidden])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[char foo;]])],
    [ AC_MSG_RESULT([yes]) ],
    [ AC_MSG_RESULT([no])
      CFLAGS="$saved_CFLAGS"
    ])

on_x86=no
case "$host_cpu" in
i[[3456]]86 | x86_64)
  on_x86=yes
  ;;
esac

on_windows=no
case $host in
*cygwin*|*mingw*)
  on_windows=yes
  ;;
esac

dnl Enable stack-protector-all only on x86 where it's well supported.
dnl on some platforms it causes crashes. Hopefully the OS's default's
dnl include this on platforms that work but have been missed here.
AC_ARG_ENABLE([stack-protector],
    [AS_HELP_STRING([--disable-stack-protector],[disable compiler stack hardening])],,
    [
      AS_IF([test "$ac_cv_c_compiler_gnu" = "yes" && test "$on_x86" = "yes" && test "$on_windows" = "no"],
            [enable_stack_protector=yes],[enable_stack_protector=no])
    ])

AS_IF([test "$enable_stack_protector" = "yes"],
 [
  saved_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS -fstack-protector-strong"
  AC_MSG_CHECKING([if ${CC} supports -fstack-protector-strong])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([],[[char foo;]])],
    [ AC_MSG_RESULT([yes]) ],
    [
      AC_MSG_RESULT([no])
      enable_stack_protector=no
      CFLAGS="$saved_CFLAGS"
    ])
 ])

AS_IF([test x$ac_cv_c_compiler_gnu = xyes],
    [AX_ADD_FORTIFY_SOURCE]
)

CFLAGS="$CFLAGS -W"

warn_CFLAGS="-Wall -Wextra -Wcast-align -Wnested-externs -Wshadow -Wstrict-prototypes"
saved_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS $warn_CFLAGS"
AC_MSG_CHECKING([if ${CC} supports ${warn_CFLAGS}])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[char foo;]])],
    [ AC_MSG_RESULT([yes]) ],
    [ AC_MSG_RESULT([no])
      CFLAGS="$saved_CFLAGS"
    ])

saved_LIBS="$LIBS"
LIBS="$LIBS $LIBM"
AC_CHECK_FUNCS([lrintf])
AC_CHECK_FUNCS([lrint])
LIBS="$saved_LIBS"

AC_CHECK_FUNCS([__malloc_hook elf_aux_info])

AC_SUBST([PC_BUILD])

AC_CONFIG_FILES([
    Makefile
    opus.pc
    opus-uninstalled.pc
    celt/arm/armopts.s
    doc/Makefile
    doc/Doxyfile
])
AC_CONFIG_HEADERS([config.h])

AC_OUTPUT

AC_MSG_NOTICE([
------------------------------------------------------------------------
  $PACKAGE_NAME $PACKAGE_VERSION:  Automatic configuration OK.

    Compiler support:

      C99 var arrays: ................ ${has_var_arrays}
      C99 lrintf: .................... ${ac_cv_func_lrintf}
      Use alloca: .................... ${use_alloca}

    General configuration:

      Floating point support: ........ ${enable_float}
      Fast float approximations: ..... ${enable_float_approx}
      Fixed point debugging: ......... ${enable_fixed_point_debug}
      Inline Assembly Optimizations: . ${inline_optimization}
      External Assembly Optimizations: ${asm_optimization}
      Intrinsics Optimizations: ...... ${intrinsics_support}
      Run-time CPU detection: ........ ${rtcd_support}
      Custom modes: .................. ${enable_custom_modes}
      Assertion checking: ............ ${enable_assertions}
      Hardening: ..................... ${enable_hardening}
      Fuzzing: ....................... ${enable_fuzzing}
      Check ASM: ..................... ${enable_check_asm}

      API documentation: ............. ${enable_doc}
      Extra programs: ................ ${enable_extra_programs}
------------------------------------------------------------------------

 Type "make; make install" to compile and install
 Type "make check" to run the test suite
])
