CELT_SOURCES = \
celt/bands.c \
celt/celt.c \
celt/celt_encoder.c \
celt/celt_decoder.c \
celt/cwrs.c \
celt/entcode.c \
celt/entdec.c \
celt/entenc.c \
celt/kiss_fft.c \
celt/laplace.c \
celt/mathops.c \
celt/mdct.c \
celt/modes.c \
celt/pitch.c \
celt/celt_lpc.c \
celt/quant_bands.c \
celt/rate.c \
celt/vq.c

CELT_SOURCES_X86_RTCD = \
celt/x86/x86cpu.c \
celt/x86/x86_celt_map.c

CELT_SOURCES_SSE = \
celt/x86/pitch_sse.c

CELT_SOURCES_SSE2 = \
celt/x86/pitch_sse2.c \
celt/x86/vq_sse2.c

CELT_SOURCES_SSE4_1 = \
celt/x86/celt_lpc_sse4_1.c \
celt/x86/pitch_sse4_1.c

CELT_SOURCES_AVX2 = \
celt/x86/pitch_avx.c

CELT_SOURCES_ARM_RTCD = \
celt/arm/armcpu.c \
celt/arm/arm_celt_map.c

CELT_SOURCES_ARM_ASM = \
celt/arm/celt_pitch_xcorr_arm.s

CELT_AM_SOURCES_ARM_ASM = \
celt/arm/armopts.s.in

CELT_SOURCES_ARM_NEON_INTR = \
celt/arm/celt_neon_intr.c \
celt/arm/pitch_neon_intr.c

CELT_SOURCES_ARM_NE10 = \
celt/arm/celt_fft_ne10.c \
celt/arm/celt_mdct_ne10.c
