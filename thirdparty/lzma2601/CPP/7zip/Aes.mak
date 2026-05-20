C_OBJS = $(C_OBJS) \
  $O\Aes.obj

!IF defined(USE_C_AES) || "$(PLATFORM)" == "arm" || "$(PLATFORM)" == "arm64"
C_OBJS = $(C_OBJS) \
  $O\AesOpt.obj
!ELSEIF "$(PLATFORM)" != "ia64" && "$(PLATFORM)" != "mips" && "$(PLATFORM)" != "arm" && "$(PLATFORM)" != "arm64"
ASM_OBJS = $(ASM_OBJS) \
  $O\AesOpt.obj
!ENDIF
