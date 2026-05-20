!IFDEF ASM_OBJS
!IF "$(PLATFORM)" == "arm64"
$(ASM_OBJS): ../../../../Asm/arm64/$(*B).S
	$(COMPL_ASM_CLANG)
!ELSEIF "$(PLATFORM)" == "arm"
$(ASM_OBJS): ../../../../Asm/arm/$(*B).asm
	$(COMPL_ASM)
!ELSEIF "$(PLATFORM)" != "ia64" && "$(PLATFORM)" != "mips"
$(ASM_OBJS): ../../../../Asm/x86/$(*B).asm
	$(COMPL_ASM)
!ENDIF
!ENDIF
