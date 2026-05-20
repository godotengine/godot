COMMON_OBJS = $(COMMON_OBJS) \
  $O\Sha256Prepare.obj

C_OBJS = $(C_OBJS) \
  $O\Sha256.obj

!IF defined(USE_C_SHA) || "$(PLATFORM)" == "arm" || "$(PLATFORM)" == "arm64"
C_OBJS = $(C_OBJS) \
  $O\Sha256Opt.obj
!ELSEIF "$(PLATFORM)" != "ia64" && "$(PLATFORM)" != "mips" && "$(PLATFORM)" != "arm" && "$(PLATFORM)" != "arm64"
ASM_OBJS = $(ASM_OBJS) \
  $O\Sha256Opt.obj
!ENDIF
