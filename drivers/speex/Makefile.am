# Disable automatic dependency tracking if using other tools than gcc and gmake
#AUTOMAKE_OPTIONS = no-dependencies


EXTRA_DIST=echo_diagnostic.m

INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(top_builddir) @OGG_CFLAGS@ @FFT_CFLAGS@

lib_LTLIBRARIES = libspeex.la libspeexdsp.la

# Sources for compilation in the library
libspeex_la_SOURCES = 	cb_search.c 	exc_10_32_table.c 	exc_8_128_table.c \
		filters.c 	gain_table.c 	hexc_table.c 	high_lsp_tables.c 	lsp.c \
		ltp.c 	speex.c 	stereo.c 	vbr.c 	vq.c bits.c exc_10_16_table.c \
	exc_20_32_table.c exc_5_256_table.c exc_5_64_table.c gain_table_lbr.c hexc_10_32_table.c \
	lpc.c lsp_tables_nb.c modes.c modes_wb.c nb_celp.c quant_lsp.c sb_celp.c \
	speex_callbacks.c speex_header.c window.c

if BUILD_KISS_FFT
  FFTSRC=kiss_fft.c _kiss_fft_guts.h kiss_fft.h kiss_fftr.c kiss_fftr.h 
else
if BUILD_SMALLFT
  FFTSRC=smallft.c
else
  FFTSRC=
endif
endif

libspeexdsp_la_SOURCES = preprocess.c jitter.c mdf.c fftwrap.c filterbank.c resample.c buffer.c scal.c $(FFTSRC)

noinst_HEADERS = 	arch.h 	cb_search_arm4.h 	cb_search_bfin.h 	cb_search_sse.h \
		filters.h 	filters_arm4.h 	filters_bfin.h 	filters_sse.h 	fixed_arm4.h \
		fixed_arm5e.h 	fixed_bfin.h 	fixed_debug.h 	lpc.h 	lpc_bfin.h 	ltp.h 	ltp_arm4.h \
		ltp_sse.h 	math_approx.h 		misc_bfin.h 	nb_celp.h 	quant_lsp.h 	sb_celp.h \
		stack_alloc.h 	vbr.h 	vq.h 	vq_arm4.h 	vq_bfin.h 	vq_sse.h cb_search.h fftwrap.h \
	filterbank.h fixed_generic.h lsp.h lsp_bfin.h ltp_bfin.h modes.h os_support.h \
	pseudofloat.h quant_lsp_bfin.h smallft.h vorbis_psy.h resample_sse.h


libspeex_la_LDFLAGS = -no-undefined -version-info @SPEEX_LT_CURRENT@:@SPEEX_LT_REVISION@:@SPEEX_LT_AGE@
libspeexdsp_la_LDFLAGS = -no-undefined -version-info @SPEEX_LT_CURRENT@:@SPEEX_LT_REVISION@:@SPEEX_LT_AGE@

noinst_PROGRAMS = testenc testenc_wb testenc_uwb testdenoise testecho testjitter
testenc_SOURCES = testenc.c
testenc_LDADD = libspeex.la
testenc_wb_SOURCES = testenc_wb.c
testenc_wb_LDADD = libspeex.la 
testenc_uwb_SOURCES = testenc_uwb.c
testenc_uwb_LDADD = libspeex.la
testdenoise_SOURCES = testdenoise.c
testdenoise_LDADD = libspeexdsp.la @FFT_LIBS@
testecho_SOURCES = testecho.c
testecho_LDADD = libspeexdsp.la @FFT_LIBS@
testjitter_SOURCES = testjitter.c
testjitter_LDADD = libspeexdsp.la @FFT_LIBS@
