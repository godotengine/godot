!IF "$(PLATFORM)" == "x64" || ("$(PLATFORM)" == "arm64" && !defined(NO_ASM_GNU))
!IFNDEF NO_ASM
CFLAGS_C_SPEC = -DZ7_LZMA_DEC_OPT
ASM_OBJS = $(ASM_OBJS) \
  $O\LzmaDecOpt.obj
!ENDIF
!ENDIF
