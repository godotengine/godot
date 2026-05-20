; 7zAsm.asm -- ASM macros
; 2023-12-08 : Igor Pavlov : Public domain


; UASM can require these changes
; OPTION FRAMEPRESERVEFLAGS:ON
; OPTION PROLOGUE:NONE
; OPTION EPILOGUE:NONE

ifdef @wordsize
; @wordsize is defined only in JWASM and ASMC and is not defined in MASM
; @wordsize eq 8 for 64-bit x64
; @wordsize eq 2 for 32-bit x86
if @wordsize eq 8
  x64 equ 1
endif
else
ifdef RAX
  x64 equ 1
endif
endif


ifdef x64
  IS_X64 equ 1
else
  IS_X64 equ 0
endif

ifdef ABI_LINUX
  IS_LINUX equ 1
else
  IS_LINUX equ 0
endif

ifndef x64
; Use ABI_CDECL for x86 (32-bit) only
; if ABI_CDECL is not defined, we use fastcall abi
ifdef ABI_CDECL
  IS_CDECL equ 1
else
  IS_CDECL equ 0
endif
endif

OPTION PROLOGUE:NONE
OPTION EPILOGUE:NONE

MY_ASM_START macro
  ifdef x64
    .code
  else
    .386
    .model flat
    _TEXT$00 SEGMENT PARA PUBLIC 'CODE'
  endif
endm

MY_PROC macro name:req, numParams:req
  align 16
  proc_numParams = numParams
  if (IS_X64 gt 0)
    proc_name equ name
  elseif (IS_LINUX gt 0)
    proc_name equ name
  elseif (IS_CDECL gt 0)
    proc_name equ @CatStr(_,name)
  else
    proc_name equ @CatStr(@,name,@, %numParams * 4)
  endif
  proc_name PROC
endm

MY_ENDP macro
    if (IS_X64 gt 0)
        ret
    elseif (IS_CDECL gt 0)
        ret
    elseif (proc_numParams LT 3)
        ret
    else
        ret (proc_numParams - 2) * 4
    endif
  proc_name ENDP
endm


ifdef x64
  REG_SIZE equ 8
  REG_LOGAR_SIZE equ 3
else
  REG_SIZE equ 4
  REG_LOGAR_SIZE equ 2
endif

  x0 equ EAX
  x1 equ ECX
  x2 equ EDX
  x3 equ EBX
  x4 equ ESP
  x5 equ EBP
  x6 equ ESI
  x7 equ EDI

  x0_W equ AX
  x1_W equ CX
  x2_W equ DX
  x3_W equ BX

  x5_W equ BP
  x6_W equ SI
  x7_W equ DI

  x0_L equ AL
  x1_L equ CL
  x2_L equ DL
  x3_L equ BL

  x0_H equ AH
  x1_H equ CH
  x2_H equ DH
  x3_H equ BH

;  r0_L equ AL
;  r1_L equ CL
;  r2_L equ DL
;  r3_L equ BL

;  r0_H equ AH
;  r1_H equ CH
;  r2_H equ DH
;  r3_H equ BH


ifdef x64
  x5_L equ BPL
  x6_L equ SIL
  x7_L equ DIL
  x8_L equ r8b
  x9_L equ r9b
  x10_L equ r10b
  x11_L equ r11b
  x12_L equ r12b
  x13_L equ r13b
  x14_L equ r14b
  x15_L equ r15b

  r0 equ RAX
  r1 equ RCX
  r2 equ RDX
  r3 equ RBX
  r4 equ RSP
  r5 equ RBP
  r6 equ RSI
  r7 equ RDI
  x8 equ r8d
  x9 equ r9d
  x10 equ r10d
  x11 equ r11d
  x12 equ r12d
  x13 equ r13d
  x14 equ r14d
  x15 equ r15d
else
  r0 equ x0
  r1 equ x1
  r2 equ x2
  r3 equ x3
  r4 equ x4
  r5 equ x5
  r6 equ x6
  r7 equ x7
endif

  x0_R equ r0
  x1_R equ r1
  x2_R equ r2
  x3_R equ r3
  x4_R equ r4
  x5_R equ r5
  x6_R equ r6
  x7_R equ r7
  x8_R equ r8
  x9_R equ r9
  x10_R equ r10
  x11_R equ r11
  x12_R equ r12
  x13_R equ r13
  x14_R equ r14
  x15_R equ r15

ifdef x64
ifdef ABI_LINUX

MY_PUSH_2_REGS macro
    push    r3
    push    r5
endm

MY_POP_2_REGS macro
    pop     r5
    pop     r3
endm

endif
endif


MY_PUSH_4_REGS macro
    push    r3
    push    r5
    push    r6
    push    r7
endm

MY_POP_4_REGS macro
    pop     r7
    pop     r6
    pop     r5
    pop     r3
endm


; for fastcall and for WIN-x64
REG_PARAM_0_x   equ x1
REG_PARAM_0     equ r1
REG_PARAM_1_x   equ x2
REG_PARAM_1     equ r2

ifndef x64
; for x86-fastcall

REG_ABI_PARAM_0_x equ REG_PARAM_0_x
REG_ABI_PARAM_0   equ REG_PARAM_0
REG_ABI_PARAM_1_x equ REG_PARAM_1_x
REG_ABI_PARAM_1   equ REG_PARAM_1

MY_PUSH_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11 macro
        MY_PUSH_4_REGS
endm

MY_POP_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11 macro
        MY_POP_4_REGS
endm

else
; x64

if  (IS_LINUX eq 0)

; for WIN-x64:
REG_PARAM_2_x   equ x8
REG_PARAM_2     equ r8
REG_PARAM_3     equ r9

REG_ABI_PARAM_0_x equ REG_PARAM_0_x
REG_ABI_PARAM_0   equ REG_PARAM_0
REG_ABI_PARAM_1_x equ REG_PARAM_1_x
REG_ABI_PARAM_1   equ REG_PARAM_1
REG_ABI_PARAM_2_x equ REG_PARAM_2_x
REG_ABI_PARAM_2   equ REG_PARAM_2
REG_ABI_PARAM_3   equ REG_PARAM_3

else
; for LINUX-x64:
REG_LINUX_PARAM_0_x equ x7
REG_LINUX_PARAM_0   equ r7
REG_LINUX_PARAM_1_x equ x6
REG_LINUX_PARAM_1   equ r6
REG_LINUX_PARAM_2   equ r2
REG_LINUX_PARAM_3   equ r1
REG_LINUX_PARAM_4_x equ x8
REG_LINUX_PARAM_4   equ r8
REG_LINUX_PARAM_5   equ r9

REG_ABI_PARAM_0_x equ REG_LINUX_PARAM_0_x
REG_ABI_PARAM_0   equ REG_LINUX_PARAM_0
REG_ABI_PARAM_1_x equ REG_LINUX_PARAM_1_x
REG_ABI_PARAM_1   equ REG_LINUX_PARAM_1
REG_ABI_PARAM_2   equ REG_LINUX_PARAM_2
REG_ABI_PARAM_3   equ REG_LINUX_PARAM_3
REG_ABI_PARAM_4_x equ REG_LINUX_PARAM_4_x
REG_ABI_PARAM_4   equ REG_LINUX_PARAM_4
REG_ABI_PARAM_5   equ REG_LINUX_PARAM_5

MY_ABI_LINUX_TO_WIN_2 macro
        mov     r2, r6
        mov     r1, r7
endm

MY_ABI_LINUX_TO_WIN_3 macro
        mov     r8, r2
        mov     r2, r6
        mov     r1, r7
endm

MY_ABI_LINUX_TO_WIN_4 macro
        mov     r9, r1
        mov     r8, r2
        mov     r2, r6
        mov     r1, r7
endm

endif ; IS_LINUX


MY_PUSH_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11 macro
    if  (IS_LINUX gt 0)
        MY_PUSH_2_REGS
    else
        MY_PUSH_4_REGS
    endif
endm

MY_POP_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11 macro
    if  (IS_LINUX gt 0)
        MY_POP_2_REGS
    else
        MY_POP_4_REGS
    endif
endm


MY_PUSH_PRESERVED_ABI_REGS macro
    MY_PUSH_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11
        push    r12
        push    r13
        push    r14
        push    r15
endm


MY_POP_PRESERVED_ABI_REGS macro
        pop     r15
        pop     r14
        pop     r13
        pop     r12
    MY_POP_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11
endm

endif ; x64
