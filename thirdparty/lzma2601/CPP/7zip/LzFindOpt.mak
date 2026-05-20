!IF defined(USE_C_LZFINDOPT) || "$(PLATFORM)" != "x64"
C_OBJS = $(C_OBJS) \
  $O\LzFindOpt.obj
!ELSE
ASM_OBJS = $(ASM_OBJS) \
  $O\LzFindOpt.obj
!ENDIF
