DEEP_PLC_SOURCES = \
dnn/burg.c \
dnn/freq.c \
dnn/fargan.c \
dnn/fargan_data.c \
dnn/lpcnet_enc.c \
dnn/lpcnet_plc.c \
dnn/lpcnet_tables.c \
dnn/nnet.c \
dnn/nnet_default.c \
dnn/plc_data.c \
dnn/parse_lpcnet_weights.c \
dnn/pitchdnn.c \
dnn/pitchdnn_data.c

DRED_SOURCES = \
dnn/dred_rdovae_enc.c \
dnn/dred_rdovae_enc_data.c \
dnn/dred_rdovae_dec.c \
dnn/dred_rdovae_dec_data.c \
dnn/dred_rdovae_stats_data.c \
dnn/dred_encoder.c \
dnn/dred_coding.c \
dnn/dred_decoder.c

OSCE_SOURCES = \
dnn/osce.c \
dnn/osce_features.c \
dnn/nndsp.c \
dnn/lace_data.c \
dnn/nolace_data.c \
dnn/bbwenet_data.c

LOSSGEN_SOURCES = \
dnn/lossgen.c \
dnn/lossgen_data.c

DNN_SOURCES_X86_RTCD = dnn/x86/x86_dnn_map.c
DNN_SOURCES_AVX2 = dnn/x86/nnet_avx2.c
DNN_SOURCES_SSE4_1 = dnn/x86/nnet_sse4_1.c
DNN_SOURCES_SSE2 = dnn/x86/nnet_sse2.c

DNN_SOURCES_ARM_RTCD = dnn/arm/arm_dnn_map.c
DNN_SOURCES_DOTPROD = dnn/arm/nnet_dotprod.c
DNN_SOURCES_NEON = dnn/arm/nnet_neon.c
