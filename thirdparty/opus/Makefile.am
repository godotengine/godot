# Provide the full test output for failed tests when using the parallel
# test suite (which is enabled by default with automake 1.13+).
export VERBOSE = yes

AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libopus.la

DIST_SUBDIRS = doc

AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/celt -I$(top_srcdir)/silk \
              -I$(top_srcdir)/silk/float -I$(top_srcdir)/silk/fixed $(NE10_CFLAGS) \
              -I$(top_srcdir)/dnn

include celt_sources.mk
include lpcnet_sources.mk
include silk_sources.mk
include opus_sources.mk

LPCNET_SOURCES =
if ENABLE_DEEP_PLC
LPCNET_SOURCES += $(DEEP_PLC_SOURCES)
endif
if ENABLE_DRED
LPCNET_SOURCES += $(DRED_SOURCES)
endif
if ENABLE_OSCE
LPCNET_SOURCES += $(OSCE_SOURCES)
endif

if FIXED_POINT
SILK_SOURCES += $(SILK_SOURCES_FIXED)
if HAVE_SSE4_1
SILK_SOURCES += $(SILK_SOURCES_SSE4_1) $(SILK_SOURCES_FIXED_SSE4_1)
endif
if HAVE_ARM_NEON_INTR
SILK_SOURCES += $(SILK_SOURCES_FIXED_ARM_NEON_INTR)
endif
else
SILK_SOURCES += $(SILK_SOURCES_FLOAT)
if HAVE_SSE4_1
SILK_SOURCES += $(SILK_SOURCES_SSE4_1)
endif
if HAVE_AVX2
SILK_SOURCES += $(SILK_SOURCES_FLOAT_AVX2)
endif
endif

if DISABLE_FLOAT_API
else
OPUS_SOURCES += $(OPUS_SOURCES_FLOAT)
endif

if CPU_X86
if HAVE_RTCD
CELT_SOURCES += $(CELT_SOURCES_X86_RTCD)
SILK_SOURCES += $(SILK_SOURCES_X86_RTCD)
if ENABLE_DEEP_PLC
LPCNET_SOURCES += $(DNN_SOURCES_X86_RTCD)
endif
endif
if HAVE_SSE
CELT_SOURCES += $(CELT_SOURCES_SSE)
endif
if HAVE_SSE2
CELT_SOURCES += $(CELT_SOURCES_SSE2)
if ENABLE_DEEP_PLC
LPCNET_SOURCES += $(DNN_SOURCES_SSE2)
endif
endif
if HAVE_SSE4_1
CELT_SOURCES += $(CELT_SOURCES_SSE4_1)
if ENABLE_DEEP_PLC
LPCNET_SOURCES += $(DNN_SOURCES_SSE4_1)
endif
endif
if HAVE_AVX2
SILK_SOURCES += $(SILK_SOURCES_AVX2)
CELT_SOURCES += $(CELT_SOURCES_AVX2)
if ENABLE_DEEP_PLC
LPCNET_SOURCES += $(DNN_SOURCES_AVX2)
endif
endif
endif

if CPU_ARM
if HAVE_RTCD
CELT_SOURCES += $(CELT_SOURCES_ARM_RTCD)
SILK_SOURCES += $(SILK_SOURCES_ARM_RTCD)
if ENABLE_DEEP_PLC
LPCNET_SOURCES += $(DNN_SOURCES_ARM_RTCD)
endif
endif

if ENABLE_DEEP_PLC
if HAVE_ARM_DOTPROD
LPCNET_SOURCES += $(DNN_SOURCES_DOTPROD)
endif
if HAVE_ARM_NEON_INTR
LPCNET_SOURCES += $(DNN_SOURCES_NEON)
endif
endif

if HAVE_ARM_NEON_INTR
CELT_SOURCES += $(CELT_SOURCES_ARM_NEON_INTR)
SILK_SOURCES += $(SILK_SOURCES_ARM_NEON_INTR)
endif

if HAVE_ARM_NE10
CELT_SOURCES += $(CELT_SOURCES_ARM_NE10)
endif

if OPUS_ARM_EXTERNAL_ASM
noinst_LTLIBRARIES = libarmasm.la
libarmasm_la_SOURCES = $(CELT_SOURCES_ARM_ASM:.s=-gnu.S)
BUILT_SOURCES = $(CELT_SOURCES_ARM_ASM:.s=-gnu.S) \
 $(CELT_AM_SOURCES_ARM_ASM:.s.in=.s) \
 $(CELT_AM_SOURCES_ARM_ASM:.s.in=-gnu.S)
endif
endif

CLEANFILES = $(CELT_SOURCES_ARM_ASM:.s=-gnu.S) \
 $(CELT_AM_SOURCES_ARM_ASM:.s.in=-gnu.S)

include celt_headers.mk
include lpcnet_headers.mk
include silk_headers.mk
include opus_headers.mk

LPCNET_HEAD =
if ENABLE_DEEP_PLC
LPCNET_HEAD += $(DEEP_PLC_HEAD)
endif
if ENABLE_DRED
LPCNET_HEAD += $(DRED_HEAD)
endif
if ENABLE_OSCE
LPCNET_HEAD += $(OSCE_HEAD)
endif
if ENABLE_LOSSGEN
LPCNET_HEAD += $(LOSSGEN_HEAD)
endif

libopus_la_SOURCES = $(CELT_SOURCES) $(SILK_SOURCES) $(LPCNET_SOURCES) $(OPUS_SOURCES)
libopus_la_LDFLAGS = -no-undefined -version-info @OPUS_LT_CURRENT@:@OPUS_LT_REVISION@:@OPUS_LT_AGE@
libopus_la_LIBADD = $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
libopus_la_LIBADD += libarmasm.la
endif

pkginclude_HEADERS = include/opus.h include/opus_multistream.h include/opus_types.h include/opus_defines.h include/opus_projection.h

noinst_HEADERS = $(OPUS_HEAD) $(SILK_HEAD) $(CELT_HEAD) $(LPCNET_HEAD)

if EXTRA_PROGRAMS
noinst_PROGRAMS = celt/tests/test_unit_cwrs32 \
                  celt/tests/test_unit_dft \
                  celt/tests/test_unit_mini_kfft \
                  celt/tests/test_unit_entropy \
                  celt/tests/test_unit_laplace \
                  celt/tests/test_unit_mathops \
                  celt/tests/test_unit_mdct \
                  celt/tests/test_unit_rotation \
                  celt/tests/test_unit_types \
                  opus_compare \
                  opus_demo \
                  repacketizer_demo \
                  silk/tests/test_unit_LPC_inv_pred_gain \
                  tests/test_opus_api \
                  tests/test_opus_decode \
                  tests/test_opus_dred \
                  tests/test_opus_encode \
                  tests/test_opus_extensions \
                  tests/test_opus_padding \
                  tests/test_opus_projection \
                  trivial_example

TESTS = celt/tests/test_unit_cwrs32 \
        celt/tests/test_unit_dft \
        celt/tests/test_unit_mini_kfft \
        celt/tests/test_unit_entropy \
        celt/tests/test_unit_laplace \
        celt/tests/test_unit_mathops \
        celt/tests/test_unit_mdct \
        celt/tests/test_unit_rotation \
        celt/tests/test_unit_types \
        silk/tests/test_unit_LPC_inv_pred_gain \
        tests/test_opus_api \
        tests/test_opus_decode \
        tests/test_opus_encode \
        tests/test_opus_extensions \
        tests/test_opus_padding \
        tests/test_opus_projection

opus_demo_SOURCES = src/opus_demo.c
if ENABLE_LOSSGEN
opus_demo_SOURCES += $(LOSSGEN_SOURCES)
endif

opus_demo_LDADD = libopus.la $(NE10_LIBS) $(LIBM)

repacketizer_demo_SOURCES = src/repacketizer_demo.c

repacketizer_demo_LDADD = libopus.la $(NE10_LIBS) $(LIBM)

opus_compare_SOURCES = src/opus_compare.c
opus_compare_LDADD = $(LIBM)

trivial_example_SOURCES = doc/trivial_example.c
trivial_example_LDADD = libopus.la $(LIBM)

tests_test_opus_api_SOURCES = tests/test_opus_api.c tests/test_opus_common.h
tests_test_opus_api_LDADD = libopus.la $(NE10_LIBS) $(LIBM)

tests_test_opus_encode_SOURCES = tests/test_opus_encode.c tests/opus_encode_regressions.c tests/test_opus_common.h
tests_test_opus_encode_LDADD = libopus.la $(NE10_LIBS) $(LIBM)

tests_test_opus_decode_SOURCES = tests/test_opus_decode.c tests/test_opus_common.h
tests_test_opus_decode_LDADD = libopus.la $(NE10_LIBS) $(LIBM)

tests_test_opus_padding_SOURCES = tests/test_opus_padding.c tests/test_opus_common.h
tests_test_opus_padding_LDADD = libopus.la $(NE10_LIBS) $(LIBM)

tests_test_opus_dred_SOURCES = tests/test_opus_dred.c tests/test_opus_common.h
tests_test_opus_dred_LDADD = libopus.la $(NE10_LIBS) $(LIBM)

if CUSTOM_MODES
tests_test_opus_custom_SOURCES = tests/test_opus_custom.c tests/test_opus_common.h
tests_test_opus_custom_LDADD = libopus.la $(NE10_LIBS) $(LIBM)
endif

CELT_OBJ = $(CELT_SOURCES:.c=.lo)
SILK_OBJ = $(SILK_SOURCES:.c=.lo)
LPCNET_OBJ = $(LPCNET_SOURCES:.c=.lo)
OPUS_OBJ = $(OPUS_SOURCES:.c=.lo)

tests_test_opus_extensions_SOURCES = tests/test_opus_extensions.c tests/test_opus_common.h
tests_test_opus_extensions_LDADD = $(OPUS_OBJ) $(SILK_OBJ) $(LPCNET_OBJ) $(CELT_OBJ) $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
tests_test_opus_extensions_LDADD += libarmasm.la
endif

tests_test_opus_projection_SOURCES = tests/test_opus_projection.c tests/test_opus_common.h
tests_test_opus_projection_LDADD = $(OPUS_OBJ) $(SILK_OBJ) $(LPCNET_OBJ) $(CELT_OBJ) $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
tests_test_opus_projection_LDADD += libarmasm.la
endif

silk_tests_test_unit_LPC_inv_pred_gain_SOURCES = silk/tests/test_unit_LPC_inv_pred_gain.c
silk_tests_test_unit_LPC_inv_pred_gain_LDADD = $(SILK_OBJ) $(LPCNET_OBJ) $(CELT_OBJ) $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
silk_tests_test_unit_LPC_inv_pred_gain_LDADD += libarmasm.la
endif

celt_tests_test_unit_cwrs32_SOURCES = celt/tests/test_unit_cwrs32.c
celt_tests_test_unit_cwrs32_LDADD = $(LIBM)

celt_tests_test_unit_dft_SOURCES = celt/tests/test_unit_dft.c
celt_tests_test_unit_dft_LDADD = $(CELT_OBJ) $(LPCNET_OBJ) $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
celt_tests_test_unit_dft_LDADD += libarmasm.la
endif

celt_tests_test_unit_mini_kfft_SOURCES = celt/tests/test_unit_mini_kfft.c
celt_tests_test_unit_mini_kfft_LDADD = $(LIBM)

celt_tests_test_unit_entropy_SOURCES = celt/tests/test_unit_entropy.c
celt_tests_test_unit_entropy_LDADD = $(LIBM)

celt_tests_test_unit_laplace_SOURCES = celt/tests/test_unit_laplace.c
celt_tests_test_unit_laplace_LDADD = $(LIBM)

celt_tests_test_unit_mathops_SOURCES = celt/tests/test_unit_mathops.c
celt_tests_test_unit_mathops_LDADD = $(CELT_OBJ) $(LPCNET_OBJ) $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
celt_tests_test_unit_mathops_LDADD += libarmasm.la
endif

celt_tests_test_unit_mdct_SOURCES = celt/tests/test_unit_mdct.c
celt_tests_test_unit_mdct_LDADD = $(CELT_OBJ) $(LPCNET_OBJ) $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
celt_tests_test_unit_mdct_LDADD += libarmasm.la
endif

celt_tests_test_unit_rotation_SOURCES = celt/tests/test_unit_rotation.c
celt_tests_test_unit_rotation_LDADD = $(CELT_OBJ) $(LPCNET_OBJ) $(NE10_LIBS) $(LIBM)
if OPUS_ARM_EXTERNAL_ASM
celt_tests_test_unit_rotation_LDADD += libarmasm.la
endif

celt_tests_test_unit_types_SOURCES = celt/tests/test_unit_types.c
celt_tests_test_unit_types_LDADD = $(LIBM)
endif

if CUSTOM_MODES
pkginclude_HEADERS += include/opus_custom.h
if EXTRA_PROGRAMS
noinst_PROGRAMS += opus_custom_demo
opus_custom_demo_SOURCES = celt/opus_custom_demo.c
opus_custom_demo_LDADD = libopus.la $(LIBM)

TESTS += tests/test_opus_custom
noinst_PROGRAMS += tests/test_opus_custom
endif
endif

if EXTRA_PROGRAMS
if ENABLE_DEEP_PLC
noinst_PROGRAMS += fargan_demo dump_data dump_weights_blob dred_compare
fargan_demo_SOURCES = dnn/fargan_demo.c
fargan_demo_LDADD = $(LPCNET_OBJ) $(CELT_OBJ) $(LIBM)

dred_compare_SOURCES = dnn/dred_compare.c
dred_compare_LDADD = $(LPCNET_OBJ) $(CELT_OBJ) $(LIBM)

dump_data_SOURCES = dnn/dump_data.c
dump_data_LDADD = $(LPCNET_OBJ) $(CELT_OBJ) $(LIBM)

dump_weights_blob_SOURCES = dnn/write_lpcnet_weights.c
dump_weights_blob_LDADD = $(LIBM)
dump_weights_blob_CFLAGS = $(AM_CFLAGS) -DDUMP_BINARY_WEIGHTS
endif
if ENABLE_DRED
TESTS += tests/test_opus_dred
endif

if ENABLE_LOSSGEN
noinst_PROGRAMS += lossgen_demo
lossgen_demo_SOURCES = dnn/lossgen_demo.c $(LOSSGEN_SOURCES)
lossgen_demo_LDADD = $(LIBM)
endif

if ENABLE_OSCE
noinst_PROGRAMS += bwe_demo
bwe_demo_SOURCES = dnn/bwe_demo.c
bwe_demo_LDADD = $(LPCNET_OBJ) $(CELT_OBJ) $(LIBM)
endif

if ENABLE_QEXT
noinst_PROGRAMS += qext_compare
qext_compare_SOURCES = src/qext_compare.c
qext_compare_LDADD = libopus.la $(LIBM)
endif

endif


EXTRA_DIST = opus.pc.in \
             opus-uninstalled.pc.in \
             opus.m4 \
             Makefile.unix \
             CMakeLists.txt \
             cmake/CFeatureCheck.cmake \
             cmake/OpusBuildtype.cmake \
             cmake/OpusConfig.cmake \
             cmake/OpusConfig.cmake.in \
             cmake/OpusFunctions.cmake \
             cmake/OpusPackageVersion.cmake \
             cmake/OpusSources.cmake \
             cmake/README.md \
             cmake/RunTest.cmake \
             cmake/config.h.cmake.in \
             cmake/vla.c \
             cmake/cpu_info_by_asm.c \
             cmake/cpu_info_by_c.c \
             meson/get-version.py \
             meson/read-sources-list.py \
             meson/README.md \
             meson.build \
             meson_options.txt \
             include/meson.build \
             celt/meson.build \
             celt/tests/meson.build \
             dnn/meson.build \
             dnn/README.md \
             silk/meson.build \
             silk/tests/meson.build \
             src/meson.build \
             tests/meson.build \
             doc/meson.build \
             tests/run_vectors.sh \
             celt/mini_kfft.c \
             celt/arm/meson.build \
             celt/arm/arm2gnu.pl \
             celt/arm/celt_pitch_xcorr_arm.s

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = opus.pc

m4datadir = $(datadir)/aclocal
m4data_DATA = opus.m4

# Targets to build and install just the library without the docs
opus check-opus install-opus: export NO_DOXYGEN = 1

opus: all
check-opus: check
install-opus: install


# Or just the docs
docs:
	( cd doc && $(MAKE) $(AM_MAKEFLAGS) )

install-docs:
	( cd doc && $(MAKE) $(AM_MAKEFLAGS) install )


# Or everything (by default)
all-local:
	@[ -n "$(NO_DOXYGEN)" ] || ( cd doc && $(MAKE) $(AM_MAKEFLAGS) )

install-data-local:
	@[ -n "$(NO_DOXYGEN)" ] || ( cd doc && $(MAKE) $(AM_MAKEFLAGS) install )

clean-local:
	-( cd doc && $(MAKE) $(AM_MAKEFLAGS) clean )

uninstall-local:
	( cd doc && $(MAKE) $(AM_MAKEFLAGS) uninstall )


# We check this every time make is run, with configure.ac being touched to
# trigger an update of the build system files if update_version changes the
# current PACKAGE_VERSION (or if package_version was modified manually by a
# user with either AUTO_UPDATE=no or no update_version script present - the
# latter being the normal case for tarball releases).
#
# We can't just add the package_version file to CONFIGURE_DEPENDENCIES since
# simply running autoconf will not actually regenerate configure for us when
# the content of that file changes (due to autoconf dependency checking not
# knowing about that without us creating yet another file for it to include).
#
# The MAKECMDGOALS check is a gnu-make'ism, but will degrade 'gracefully' for
# makes that don't support it.  The only loss of functionality is not forcing
# an update of package_version for `make dist` if AUTO_UPDATE=no, but that is
# unlikely to be a real problem for any real user.
$(top_srcdir)/configure.ac: force
	@case "$(MAKECMDGOALS)" in \
	    dist-hook)                             exit 0       ;; \
	    dist-* | dist | distcheck | distclean) _arg=release ;; \
	esac; \
	if ! $(top_srcdir)/update_version $$_arg 2> /dev/null; then \
	    if [ ! -e $(top_srcdir)/package_version ]; then \
		echo 'PACKAGE_VERSION="unknown"' > $(top_srcdir)/package_version; \
	    fi; \
	    . $(top_srcdir)/package_version || exit 1; \
	    [ "$(PACKAGE_VERSION)" != "$$PACKAGE_VERSION" ] || exit 0; \
	fi; \
	touch $@

force:

# Create a minimal package_version file when make dist is run.
dist-hook:
	echo 'PACKAGE_VERSION="$(PACKAGE_VERSION)"' > $(top_distdir)/package_version


.PHONY: opus check-opus install-opus docs install-docs

# automake doesn't do dependency tracking for asm files, that I can tell
$(CELT_SOURCES_ARM_ASM:%.s=%-gnu.S): celt/arm/armopts-gnu.S
$(CELT_SOURCES_ARM_ASM:%.s=%-gnu.S): $(top_srcdir)/celt/arm/arm2gnu.pl

# convert ARM asm to GNU as format
%-gnu.S: $(top_srcdir)/%.s
	$(top_srcdir)/celt/arm/arm2gnu.pl @ARM2GNU_PARAMS@ < $< > $@
# For autoconf-modified sources (e.g., armopts.s)
%-gnu.S: %.s
	$(top_srcdir)/celt/arm/arm2gnu.pl @ARM2GNU_PARAMS@ < $< > $@

OPT_UNIT_TEST_OBJ = $(celt_tests_test_unit_mathops_SOURCES:.c=.o) \
                    $(celt_tests_test_unit_rotation_SOURCES:.c=.o) \
                    $(celt_tests_test_unit_mdct_SOURCES:.c=.o) \
                    $(celt_tests_test_unit_dft_SOURCES:.c=.o) \
                    $(silk_tests_test_unit_LPC_inv_pred_gain_SOURCES:.c=.o)

if HAVE_SSE
SSE_OBJ = $(CELT_SOURCES_SSE:.c=.lo)
$(SSE_OBJ): CFLAGS += $(OPUS_X86_SSE_CFLAGS)
endif

if HAVE_SSE2
SSE2_OBJ = $(CELT_SOURCES_SSE2:.c=.lo) \
           $(DNN_SOURCES_SSE2:.c=.lo)
$(SSE2_OBJ): CFLAGS += $(OPUS_X86_SSE2_CFLAGS)
endif

if HAVE_SSE4_1
SSE4_1_OBJ = $(CELT_SOURCES_SSE4_1:.c=.lo) \
             $(DNN_SOURCES_SSE4_1:.c=.lo) \
             $(SILK_SOURCES_SSE4_1:.c=.lo) \
             $(SILK_SOURCES_FIXED_SSE4_1:.c=.lo)
$(SSE4_1_OBJ): CFLAGS += $(OPUS_X86_SSE4_1_CFLAGS)
endif

if HAVE_AVX2
AVX2_OBJ = $(CELT_SOURCES_AVX2:.c=.lo) \
           $(SILK_SOURCES_AVX2:.c=.lo) \
           $(SILK_SOURCES_FLOAT_AVX2:.c=.lo) \
           $(DNN_SOURCES_AVX2:.c=.lo)
$(AVX2_OBJ): CFLAGS += $(OPUS_X86_AVX2_CFLAGS)
endif

if HAVE_ARM_NEON_INTR
ARM_NEON_INTR_OBJ = $(CELT_SOURCES_ARM_NEON_INTR:.c=.lo) \
                    $(SILK_SOURCES_ARM_NEON_INTR:.c=.lo) \
                    $(DNN_SOURCES_NEON:.c=.lo) \
                    $(SILK_SOURCES_FIXED_ARM_NEON_INTR:.c=.lo)
$(ARM_NEON_INTR_OBJ): CFLAGS += \
 $(OPUS_ARM_NEON_INTR_CFLAGS)  $(NE10_CFLAGS)
endif

if HAVE_ARM_DOTPROD
ARM_DOTPROD_OBJ = $(DNN_SOURCES_DOTPROD:.c=.lo)
$(ARM_DOTPROD_OBJ): CFLAGS += $(ARM_DOTPROD_INTR_CFLAGS)
endif
