;*****************************************************************************
;* x86inc.asm: x264asm abstraction layer
;*****************************************************************************
;* Copyright (C) 2005-2016 x264 project
;*
;* Authors: Loren Merritt <lorenm@u.washington.edu>
;*          Anton Mitrofanov <BugMaster@narod.ru>
;*          Fiona Glaser <fiona@x264.com>
;*          Henrik Gramner <henrik@gramner.com>
;*
;* Permission to use, copy, modify, and/or distribute this software for any
;* purpose with or without fee is hereby granted, provided that the above
;* copyright notice and this permission notice appear in all copies.
;*
;* THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
;* WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
;* MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
;* ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
;* WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
;* ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
;* OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
;*****************************************************************************

; This is a header file for the x264ASM assembly language, which uses
; NASM/YASM syntax combined with a large number of macros to provide easy
; abstraction between different calling conventions (x86_32, win64, linux64).
; It also has various other useful features to simplify writing the kind of
; DSP functions that are most often used in x264.

; Unlike the rest of x264, this file is available under an ISC license, as it
; has significant usefulness outside of x264 and we want it to be available
; to the largest audience possible.  Of course, if you modify it for your own
; purposes to add a new feature, we strongly encourage contributing a patch
; as this feature might be useful for others as well.  Send patches or ideas
; to x264-devel@videolan.org .

%include "vpx_config.asm"

%ifndef private_prefix
    %define private_prefix vpx
%endif

%ifndef public_prefix
    %define public_prefix private_prefix
%endif

%ifndef STACK_ALIGNMENT
    %if ARCH_X86_64
        %define STACK_ALIGNMENT 16
    %else
        %define STACK_ALIGNMENT 4
    %endif
%endif

%define WIN64  0
%define UNIX64 0
%if ARCH_X86_64
    %ifidn __OUTPUT_FORMAT__,win32
        %define WIN64  1
    %elifidn __OUTPUT_FORMAT__,win64
        %define WIN64  1
    %elifidn __OUTPUT_FORMAT__,x64
        %define WIN64  1
    %else
        %define UNIX64 1
    %endif
%endif

%define FORMAT_ELF 0
%ifidn __OUTPUT_FORMAT__,elf
    %define FORMAT_ELF 1
%elifidn __OUTPUT_FORMAT__,elf32
    %define FORMAT_ELF 1
%elifidn __OUTPUT_FORMAT__,elf64
    %define FORMAT_ELF 1
%endif

%define FORMAT_MACHO 0
%ifidn __OUTPUT_FORMAT__,macho32
     %define FORMAT_MACHO 1
%elifidn __OUTPUT_FORMAT__,macho64
     %define FORMAT_MACHO 1
%endif

; Set PREFIX for libvpx builds.
%if FORMAT_ELF
    %undef PREFIX
%elif WIN64
    %undef PREFIX
%else
    %define PREFIX
%endif

%ifdef PREFIX
    %define mangle(x) _ %+ x
%else
    %define mangle(x) x
%endif

; In some instances macho32 tables get misaligned when using .rodata.
; When looking at the disassembly it appears that the offset is either
; correct or consistently off by 90. Placing them in the .text section
; works around the issue. It appears to be specific to the way libvpx
; handles the tables.
%macro SECTION_RODATA 0-1 16
    %ifidn __OUTPUT_FORMAT__,macho32
        SECTION .text align=%1
        fakegot:
    %elifidn __OUTPUT_FORMAT__,aout
        SECTION .text
    %else
        SECTION .rodata align=%1
    %endif
%endmacro

; PIC macros are copied from vpx_ports/x86_abi_support.asm. The "define PIC"
; from original code is added in for 64bit.
%ifidn __OUTPUT_FORMAT__,elf32
%define ABI_IS_32BIT 1
%elifidn __OUTPUT_FORMAT__,macho32
%define ABI_IS_32BIT 1
%elifidn __OUTPUT_FORMAT__,win32
%define ABI_IS_32BIT 1
%elifidn __OUTPUT_FORMAT__,aout
%define ABI_IS_32BIT 1
%else
%define ABI_IS_32BIT 0
%endif

%if ABI_IS_32BIT
    %if CONFIG_PIC=1
        %ifidn __OUTPUT_FORMAT__,elf32
            %define GET_GOT_DEFINED 1
            %define WRT_PLT wrt ..plt
            %macro GET_GOT 1
                extern _GLOBAL_OFFSET_TABLE_
                push %1
                call %%get_got
                %%sub_offset:
                jmp %%exitGG
                %%get_got:
                mov %1, [esp]
                add %1, _GLOBAL_OFFSET_TABLE_ + $$ - %%sub_offset wrt ..gotpc
                ret
                %%exitGG:
                %undef GLOBAL
                %define GLOBAL(x) x + %1 wrt ..gotoff
                %undef RESTORE_GOT
                %define RESTORE_GOT pop %1
            %endmacro
        %elifidn __OUTPUT_FORMAT__,macho32
            %define GET_GOT_DEFINED 1
            %macro GET_GOT 1
                push %1
                call %%get_got
                %%get_got:
                pop  %1
                %undef GLOBAL
                %define GLOBAL(x) x + %1 - %%get_got
                %undef RESTORE_GOT
                %define RESTORE_GOT pop %1
            %endmacro
        %else
            %define GET_GOT_DEFINED 0
        %endif
    %endif

    %if ARCH_X86_64 == 0
        %undef PIC
    %endif

%else
    %macro GET_GOT 1
    %endmacro
    %define GLOBAL(x) rel x
    %define WRT_PLT wrt ..plt

    %if WIN64
        %define PIC
    %elifidn __OUTPUT_FORMAT__,macho64
        %define PIC
    %elif CONFIG_PIC
        %define PIC
    %endif
%endif

%ifnmacro GET_GOT
    %macro GET_GOT 1
    %endmacro
    %define GLOBAL(x) x
%endif
%ifndef RESTORE_GOT
    %define RESTORE_GOT
%endif
%ifndef WRT_PLT
    %define WRT_PLT
%endif

%ifdef PIC
    default rel
%endif

%ifndef GET_GOT_DEFINED
    %define GET_GOT_DEFINED 0
%endif
; Done with PIC macros

%ifdef __NASM_VER__
    %use smartalign
%endif

; Macros to eliminate most code duplication between x86_32 and x86_64:
; Currently this works only for leaf functions which load all their arguments
; into registers at the start, and make no other use of the stack. Luckily that
; covers most of x264's asm.

; PROLOGUE:
; %1 = number of arguments. loads them from stack if needed.
; %2 = number of registers used. pushes callee-saved regs if needed.
; %3 = number of xmm registers used. pushes callee-saved xmm regs if needed.
; %4 = (optional) stack size to be allocated. The stack will be aligned before
;      allocating the specified stack size. If the required stack alignment is
;      larger than the known stack alignment the stack will be manually aligned
;      and an extra register will be allocated to hold the original stack
;      pointer (to not invalidate r0m etc.). To prevent the use of an extra
;      register as stack pointer, request a negative stack size.
; %4+/%5+ = list of names to define to registers
; PROLOGUE can also be invoked by adding the same options to cglobal

; e.g.
; cglobal foo, 2,3,7,0x40, dst, src, tmp
; declares a function (foo) that automatically loads two arguments (dst and
; src) into registers, uses one additional register (tmp) plus 7 vector
; registers (m0-m6) and allocates 0x40 bytes of stack space.

; TODO Some functions can use some args directly from the stack. If they're the
; last args then you can just not declare them, but if they're in the middle
; we need more flexible macro.

; RET:
; Pops anything that was pushed by PROLOGUE, and returns.

; REP_RET:
; Use this instead of RET if it's a branch target.

; registers:
; rN and rNq are the native-size register holding function argument N
; rNd, rNw, rNb are dword, word, and byte size
; rNh is the high 8 bits of the word size
; rNm is the original location of arg N (a register or on the stack), dword
; rNmp is native size

%macro DECLARE_REG 2-3
    %define r%1q %2
    %define r%1d %2d
    %define r%1w %2w
    %define r%1b %2b
    %define r%1h %2h
    %define %2q %2
    %if %0 == 2
        %define r%1m  %2d
        %define r%1mp %2
    %elif ARCH_X86_64 ; memory
        %define r%1m [rstk + stack_offset + %3]
        %define r%1mp qword r %+ %1 %+ m
    %else
        %define r%1m [rstk + stack_offset + %3]
        %define r%1mp dword r %+ %1 %+ m
    %endif
    %define r%1  %2
%endmacro

%macro DECLARE_REG_SIZE 3
    %define r%1q r%1
    %define e%1q r%1
    %define r%1d e%1
    %define e%1d e%1
    %define r%1w %1
    %define e%1w %1
    %define r%1h %3
    %define e%1h %3
    %define r%1b %2
    %define e%1b %2
    %if ARCH_X86_64 == 0
        %define r%1 e%1
    %endif
%endmacro

DECLARE_REG_SIZE ax, al, ah
DECLARE_REG_SIZE bx, bl, bh
DECLARE_REG_SIZE cx, cl, ch
DECLARE_REG_SIZE dx, dl, dh
DECLARE_REG_SIZE si, sil, null
DECLARE_REG_SIZE di, dil, null
DECLARE_REG_SIZE bp, bpl, null

; t# defines for when per-arch register allocation is more complex than just function arguments

%macro DECLARE_REG_TMP 1-*
    %assign %%i 0
    %rep %0
        CAT_XDEFINE t, %%i, r%1
        %assign %%i %%i+1
        %rotate 1
    %endrep
%endmacro

%macro DECLARE_REG_TMP_SIZE 0-*
    %rep %0
        %define t%1q t%1 %+ q
        %define t%1d t%1 %+ d
        %define t%1w t%1 %+ w
        %define t%1h t%1 %+ h
        %define t%1b t%1 %+ b
        %rotate 1
    %endrep
%endmacro

DECLARE_REG_TMP_SIZE 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14

%if ARCH_X86_64
    %define gprsize 8
%else
    %define gprsize 4
%endif

%macro PUSH 1
    push %1
    %ifidn rstk, rsp
        %assign stack_offset stack_offset+gprsize
    %endif
%endmacro

%macro POP 1
    pop %1
    %ifidn rstk, rsp
        %assign stack_offset stack_offset-gprsize
    %endif
%endmacro

%macro PUSH_IF_USED 1-*
    %rep %0
        %if %1 < regs_used
            PUSH r%1
        %endif
        %rotate 1
    %endrep
%endmacro

%macro POP_IF_USED 1-*
    %rep %0
        %if %1 < regs_used
            pop r%1
        %endif
        %rotate 1
    %endrep
%endmacro

%macro LOAD_IF_USED 1-*
    %rep %0
        %if %1 < num_args
            mov r%1, r %+ %1 %+ mp
        %endif
        %rotate 1
    %endrep
%endmacro

%macro SUB 2
    sub %1, %2
    %ifidn %1, rstk
        %assign stack_offset stack_offset+(%2)
    %endif
%endmacro

%macro ADD 2
    add %1, %2
    %ifidn %1, rstk
        %assign stack_offset stack_offset-(%2)
    %endif
%endmacro

%macro movifnidn 2
    %ifnidn %1, %2
        mov %1, %2
    %endif
%endmacro

%macro movsxdifnidn 2
    %ifnidn %1, %2
        movsxd %1, %2
    %endif
%endmacro

%macro ASSERT 1
    %if (%1) == 0
        %error assertion ``%1'' failed
    %endif
%endmacro

%macro DEFINE_ARGS 0-*
    %ifdef n_arg_names
        %assign %%i 0
        %rep n_arg_names
            CAT_UNDEF arg_name %+ %%i, q
            CAT_UNDEF arg_name %+ %%i, d
            CAT_UNDEF arg_name %+ %%i, w
            CAT_UNDEF arg_name %+ %%i, h
            CAT_UNDEF arg_name %+ %%i, b
            CAT_UNDEF arg_name %+ %%i, m
            CAT_UNDEF arg_name %+ %%i, mp
            CAT_UNDEF arg_name, %%i
            %assign %%i %%i+1
        %endrep
    %endif

    %xdefine %%stack_offset stack_offset
    %undef stack_offset ; so that the current value of stack_offset doesn't get baked in by xdefine
    %assign %%i 0
    %rep %0
        %xdefine %1q r %+ %%i %+ q
        %xdefine %1d r %+ %%i %+ d
        %xdefine %1w r %+ %%i %+ w
        %xdefine %1h r %+ %%i %+ h
        %xdefine %1b r %+ %%i %+ b
        %xdefine %1m r %+ %%i %+ m
        %xdefine %1mp r %+ %%i %+ mp
        CAT_XDEFINE arg_name, %%i, %1
        %assign %%i %%i+1
        %rotate 1
    %endrep
    %xdefine stack_offset %%stack_offset
    %assign n_arg_names %0
%endmacro

%define required_stack_alignment ((mmsize + 15) & ~15)

%macro ALLOC_STACK 1-2 0 ; stack_size, n_xmm_regs (for win64 only)
    %ifnum %1
        %if %1 != 0
            %assign %%pad 0
            %assign stack_size %1
            %if stack_size < 0
                %assign stack_size -stack_size
            %endif
            %if WIN64
                %assign %%pad %%pad + 32 ; shadow space
                %if mmsize != 8
                    %assign xmm_regs_used %2
                    %if xmm_regs_used > 8
                        %assign %%pad %%pad + (xmm_regs_used-8)*16 ; callee-saved xmm registers
                    %endif
                %endif
            %endif
            %if required_stack_alignment <= STACK_ALIGNMENT
                ; maintain the current stack alignment
                %assign stack_size_padded stack_size + %%pad + ((-%%pad-stack_offset-gprsize) & (STACK_ALIGNMENT-1))
                SUB rsp, stack_size_padded
            %else
                %assign %%reg_num (regs_used - 1)
                %xdefine rstk r %+ %%reg_num
                ; align stack, and save original stack location directly above
                ; it, i.e. in [rsp+stack_size_padded], so we can restore the
                ; stack in a single instruction (i.e. mov rsp, rstk or mov
                ; rsp, [rsp+stack_size_padded])
                %if %1 < 0 ; need to store rsp on stack
                    %xdefine rstkm [rsp + stack_size + %%pad]
                    %assign %%pad %%pad + gprsize
                %else ; can keep rsp in rstk during whole function
                    %xdefine rstkm rstk
                %endif
                %assign stack_size_padded stack_size + ((%%pad + required_stack_alignment-1) & ~(required_stack_alignment-1))
                mov rstk, rsp
                and rsp, ~(required_stack_alignment-1)
                sub rsp, stack_size_padded
                movifnidn rstkm, rstk
            %endif
            WIN64_PUSH_XMM
        %endif
    %endif
%endmacro

%macro SETUP_STACK_POINTER 1
    %ifnum %1
        %if %1 != 0 && required_stack_alignment > STACK_ALIGNMENT
            %if %1 > 0
                %assign regs_used (regs_used + 1)
            %endif
            %if ARCH_X86_64 && regs_used < 5 + UNIX64 * 3
                ; Ensure that we don't clobber any registers containing arguments
                %assign regs_used 5 + UNIX64 * 3
            %endif
        %endif
    %endif
%endmacro

%macro DEFINE_ARGS_INTERNAL 3+
    %ifnum %2
        DEFINE_ARGS %3
    %elif %1 == 4
        DEFINE_ARGS %2
    %elif %1 > 4
        DEFINE_ARGS %2, %3
    %endif
%endmacro

%if WIN64 ; Windows x64 ;=================================================

DECLARE_REG 0,  rcx
DECLARE_REG 1,  rdx
DECLARE_REG 2,  R8
DECLARE_REG 3,  R9
DECLARE_REG 4,  R10, 40
DECLARE_REG 5,  R11, 48
DECLARE_REG 6,  rax, 56
DECLARE_REG 7,  rdi, 64
DECLARE_REG 8,  rsi, 72
DECLARE_REG 9,  rbx, 80
DECLARE_REG 10, rbp, 88
DECLARE_REG 11, R12, 96
DECLARE_REG 12, R13, 104
DECLARE_REG 13, R14, 112
DECLARE_REG 14, R15, 120

%macro PROLOGUE 2-5+ 0 ; #args, #regs, #xmm_regs, [stack_size,] arg_names...
    %assign num_args %1
    %assign regs_used %2
    ASSERT regs_used >= num_args
    SETUP_STACK_POINTER %4
    ASSERT regs_used <= 15
    PUSH_IF_USED 7, 8, 9, 10, 11, 12, 13, 14
    ALLOC_STACK %4, %3
    %if mmsize != 8 && stack_size == 0
        WIN64_SPILL_XMM %3
    %endif
    LOAD_IF_USED 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    DEFINE_ARGS_INTERNAL %0, %4, %5
%endmacro

%macro WIN64_PUSH_XMM 0
    ; Use the shadow space to store XMM6 and XMM7, the rest needs stack space allocated.
    %if xmm_regs_used > 6
        movaps [rstk + stack_offset +  8], xmm6
    %endif
    %if xmm_regs_used > 7
        movaps [rstk + stack_offset + 24], xmm7
    %endif
    %if xmm_regs_used > 8
        %assign %%i 8
        %rep xmm_regs_used-8
            movaps [rsp + (%%i-8)*16 + stack_size + 32], xmm %+ %%i
            %assign %%i %%i+1
        %endrep
    %endif
%endmacro

%macro WIN64_SPILL_XMM 1
    %assign xmm_regs_used %1
    ASSERT xmm_regs_used <= 16
    %if xmm_regs_used > 8
        ; Allocate stack space for callee-saved xmm registers plus shadow space and align the stack.
        %assign %%pad (xmm_regs_used-8)*16 + 32
        %assign stack_size_padded %%pad + ((-%%pad-stack_offset-gprsize) & (STACK_ALIGNMENT-1))
        SUB rsp, stack_size_padded
    %endif
    WIN64_PUSH_XMM
%endmacro

%macro WIN64_RESTORE_XMM_INTERNAL 1
    %assign %%pad_size 0
    %if xmm_regs_used > 8
        %assign %%i xmm_regs_used
        %rep xmm_regs_used-8
            %assign %%i %%i-1
            movaps xmm %+ %%i, [%1 + (%%i-8)*16 + stack_size + 32]
        %endrep
    %endif
    %if stack_size_padded > 0
        %if stack_size > 0 && required_stack_alignment > STACK_ALIGNMENT
            mov rsp, rstkm
        %else
            add %1, stack_size_padded
            %assign %%pad_size stack_size_padded
        %endif
    %endif
    %if xmm_regs_used > 7
        movaps xmm7, [%1 + stack_offset - %%pad_size + 24]
    %endif
    %if xmm_regs_used > 6
        movaps xmm6, [%1 + stack_offset - %%pad_size +  8]
    %endif
%endmacro

%macro WIN64_RESTORE_XMM 1
    WIN64_RESTORE_XMM_INTERNAL %1
    %assign stack_offset (stack_offset-stack_size_padded)
    %assign xmm_regs_used 0
%endmacro

%define has_epilogue regs_used > 7 || xmm_regs_used > 6 || mmsize == 32 || stack_size > 0

%macro RET 0
    WIN64_RESTORE_XMM_INTERNAL rsp
    POP_IF_USED 14, 13, 12, 11, 10, 9, 8, 7
    %if mmsize == 32
        vzeroupper
    %endif
    AUTO_REP_RET
%endmacro

%elif ARCH_X86_64 ; *nix x64 ;=============================================

DECLARE_REG 0,  rdi
DECLARE_REG 1,  rsi
DECLARE_REG 2,  rdx
DECLARE_REG 3,  rcx
DECLARE_REG 4,  R8
DECLARE_REG 5,  R9
DECLARE_REG 6,  rax, 8
DECLARE_REG 7,  R10, 16
DECLARE_REG 8,  R11, 24
DECLARE_REG 9,  rbx, 32
DECLARE_REG 10, rbp, 40
DECLARE_REG 11, R12, 48
DECLARE_REG 12, R13, 56
DECLARE_REG 13, R14, 64
DECLARE_REG 14, R15, 72

%macro PROLOGUE 2-5+ ; #args, #regs, #xmm_regs, [stack_size,] arg_names...
    %assign num_args %1
    %assign regs_used %2
    ASSERT regs_used >= num_args
    SETUP_STACK_POINTER %4
    ASSERT regs_used <= 15
    PUSH_IF_USED 9, 10, 11, 12, 13, 14
    ALLOC_STACK %4
    LOAD_IF_USED 6, 7, 8, 9, 10, 11, 12, 13, 14
    DEFINE_ARGS_INTERNAL %0, %4, %5
%endmacro

%define has_epilogue regs_used > 9 || mmsize == 32 || stack_size > 0

%macro RET 0
    %if stack_size_padded > 0
        %if required_stack_alignment > STACK_ALIGNMENT
            mov rsp, rstkm
        %else
            add rsp, stack_size_padded
        %endif
    %endif
    POP_IF_USED 14, 13, 12, 11, 10, 9
    %if mmsize == 32
        vzeroupper
    %endif
    AUTO_REP_RET
%endmacro

%else ; X86_32 ;==============================================================

DECLARE_REG 0, eax, 4
DECLARE_REG 1, ecx, 8
DECLARE_REG 2, edx, 12
DECLARE_REG 3, ebx, 16
DECLARE_REG 4, esi, 20
DECLARE_REG 5, edi, 24
DECLARE_REG 6, ebp, 28
%define rsp esp

%macro DECLARE_ARG 1-*
    %rep %0
        %define r%1m [rstk + stack_offset + 4*%1 + 4]
        %define r%1mp dword r%1m
        %rotate 1
    %endrep
%endmacro

DECLARE_ARG 7, 8, 9, 10, 11, 12, 13, 14

%macro PROLOGUE 2-5+ ; #args, #regs, #xmm_regs, [stack_size,] arg_names...
    %assign num_args %1
    %assign regs_used %2
    ASSERT regs_used >= num_args
    %if num_args > 7
        %assign num_args 7
    %endif
    %if regs_used > 7
        %assign regs_used 7
    %endif
    SETUP_STACK_POINTER %4
    ASSERT regs_used <= 7
    PUSH_IF_USED 3, 4, 5, 6
    ALLOC_STACK %4
    LOAD_IF_USED 0, 1, 2, 3, 4, 5, 6
    DEFINE_ARGS_INTERNAL %0, %4, %5
%endmacro

%define has_epilogue regs_used > 3 || mmsize == 32 || stack_size > 0

%macro RET 0
    %if stack_size_padded > 0
        %if required_stack_alignment > STACK_ALIGNMENT
            mov rsp, rstkm
        %else
            add rsp, stack_size_padded
        %endif
    %endif
    POP_IF_USED 6, 5, 4, 3
    %if mmsize == 32
        vzeroupper
    %endif
    AUTO_REP_RET
%endmacro

%endif ;======================================================================

%if WIN64 == 0
    %macro WIN64_SPILL_XMM 1
    %endmacro
    %macro WIN64_RESTORE_XMM 1
    %endmacro
    %macro WIN64_PUSH_XMM 0
    %endmacro
%endif

; On AMD cpus <=K10, an ordinary ret is slow if it immediately follows either
; a branch or a branch target. So switch to a 2-byte form of ret in that case.
; We can automatically detect "follows a branch", but not a branch target.
; (SSSE3 is a sufficient condition to know that your cpu doesn't have this problem.)
%macro REP_RET 0
    %if has_epilogue
        RET
    %else
        rep ret
    %endif
    annotate_function_size
%endmacro

%define last_branch_adr $$
%macro AUTO_REP_RET 0
    %if notcpuflag(ssse3)
        times ((last_branch_adr-$)>>31)+1 rep ; times 1 iff $ == last_branch_adr.
    %endif
    ret
    annotate_function_size
%endmacro

%macro BRANCH_INSTR 0-*
    %rep %0
        %macro %1 1-2 %1
            %2 %1
            %if notcpuflag(ssse3)
                %%branch_instr equ $
                %xdefine last_branch_adr %%branch_instr
            %endif
        %endmacro
        %rotate 1
    %endrep
%endmacro

BRANCH_INSTR jz, je, jnz, jne, jl, jle, jnl, jnle, jg, jge, jng, jnge, ja, jae, jna, jnae, jb, jbe, jnb, jnbe, jc, jnc, js, jns, jo, jno, jp, jnp

%macro TAIL_CALL 2 ; callee, is_nonadjacent
    %if has_epilogue
        call %1
        RET
    %elif %2
        jmp %1
    %endif
    annotate_function_size
%endmacro

;=============================================================================
; arch-independent part
;=============================================================================

%assign function_align 16

; Begin a function.
; Applies any symbol mangling needed for C linkage, and sets up a define such that
; subsequent uses of the function name automatically refer to the mangled version.
; Appends cpuflags to the function name if cpuflags has been specified.
; The "" empty default parameter is a workaround for nasm, which fails if SUFFIX
; is empty and we call cglobal_internal with just %1 %+ SUFFIX (without %2).
%macro cglobal 1-2+ "" ; name, [PROLOGUE args]
    cglobal_internal 1, %1 %+ SUFFIX, %2
%endmacro
%macro cvisible 1-2+ "" ; name, [PROLOGUE args]
    cglobal_internal 0, %1 %+ SUFFIX, %2
%endmacro
%macro cglobal_internal 2-3+
    annotate_function_size
    %if %1
        %xdefine %%FUNCTION_PREFIX private_prefix
        ; libvpx explicitly sets visibility in shared object builds. Avoid
        ; setting visibility to hidden as it may break builds that split
        ; sources on e.g., directory boundaries.
        %ifdef CHROMIUM
            %xdefine %%VISIBILITY hidden
        %else
            %xdefine %%VISIBILITY
        %endif
    %else
        %xdefine %%FUNCTION_PREFIX public_prefix
        %xdefine %%VISIBILITY
    %endif
    %ifndef cglobaled_%2
        %xdefine %2 mangle(%%FUNCTION_PREFIX %+ _ %+ %2)
        %xdefine %2.skip_prologue %2 %+ .skip_prologue
        CAT_XDEFINE cglobaled_, %2, 1
    %endif
    %xdefine current_function %2
    %xdefine current_function_section __SECT__
    %if FORMAT_ELF
        global %2:function %%VISIBILITY
    %elif FORMAT_MACHO
        %ifdef __NASM_VER__
            global %2
        %else
            global %2:private_extern
        %endif
    %else
        global %2
    %endif
    align function_align
    %2:
    RESET_MM_PERMUTATION        ; needed for x86-64, also makes disassembly somewhat nicer
    %xdefine rstk rsp           ; copy of the original stack pointer, used when greater alignment than the known stack alignment is required
    %assign stack_offset 0      ; stack pointer offset relative to the return address
    %assign stack_size 0        ; amount of stack space that can be freely used inside a function
    %assign stack_size_padded 0 ; total amount of allocated stack space, including space for callee-saved xmm registers on WIN64 and alignment padding
    %assign xmm_regs_used 0     ; number of XMM registers requested, used for dealing with callee-saved registers on WIN64
    %ifnidn %3, ""
        PROLOGUE %3
    %endif
%endmacro

%macro cextern 1
    %xdefine %1 mangle(private_prefix %+ _ %+ %1)
    CAT_XDEFINE cglobaled_, %1, 1
    extern %1
%endmacro

; like cextern, but without the prefix
%macro cextern_naked 1
    %ifdef PREFIX
        %xdefine %1 mangle(%1)
    %endif
    CAT_XDEFINE cglobaled_, %1, 1
    extern %1
%endmacro

%macro const 1-2+
    %xdefine %1 mangle(private_prefix %+ _ %+ %1)
    %if FORMAT_ELF
        global %1:data hidden
    %else
        global %1
    %endif
    %1: %2
%endmacro

; This is needed for ELF, otherwise the GNU linker assumes the stack is executable by default.
%if FORMAT_ELF
    [SECTION .note.GNU-stack noalloc noexec nowrite progbits]
%endif

; Tell debuggers how large the function was.
; This may be invoked multiple times per function; we rely on later instances overriding earlier ones.
; This is invoked by RET and similar macros, and also cglobal does it for the previous function,
; but if the last function in a source file doesn't use any of the standard macros for its epilogue,
; then its size might be unspecified.
%macro annotate_function_size 0
    %ifdef __YASM_VER__
        %ifdef current_function
            %if FORMAT_ELF
                current_function_section
                %%ecf equ $
                size current_function %%ecf - current_function
                __SECT__
            %endif
        %endif
    %endif
%endmacro

; cpuflags

%assign cpuflags_mmx      (1<<0)
%assign cpuflags_mmx2     (1<<1) | cpuflags_mmx
%assign cpuflags_3dnow    (1<<2) | cpuflags_mmx
%assign cpuflags_3dnowext (1<<3) | cpuflags_3dnow
%assign cpuflags_sse      (1<<4) | cpuflags_mmx2
%assign cpuflags_sse2     (1<<5) | cpuflags_sse
%assign cpuflags_sse2slow (1<<6) | cpuflags_sse2
%assign cpuflags_sse3     (1<<7) | cpuflags_sse2
%assign cpuflags_ssse3    (1<<8) | cpuflags_sse3
%assign cpuflags_sse4     (1<<9) | cpuflags_ssse3
%assign cpuflags_sse42    (1<<10)| cpuflags_sse4
%assign cpuflags_avx      (1<<11)| cpuflags_sse42
%assign cpuflags_xop      (1<<12)| cpuflags_avx
%assign cpuflags_fma4     (1<<13)| cpuflags_avx
%assign cpuflags_fma3     (1<<14)| cpuflags_avx
%assign cpuflags_avx2     (1<<15)| cpuflags_fma3

%assign cpuflags_cache32  (1<<16)
%assign cpuflags_cache64  (1<<17)
%assign cpuflags_slowctz  (1<<18)
%assign cpuflags_lzcnt    (1<<19)
%assign cpuflags_aligned  (1<<20) ; not a cpu feature, but a function variant
%assign cpuflags_atom     (1<<21)
%assign cpuflags_bmi1     (1<<22)|cpuflags_lzcnt
%assign cpuflags_bmi2     (1<<23)|cpuflags_bmi1

; Returns a boolean value expressing whether or not the specified cpuflag is enabled.
%define    cpuflag(x) (((((cpuflags & (cpuflags_ %+ x)) ^ (cpuflags_ %+ x)) - 1) >> 31) & 1)
%define notcpuflag(x) (cpuflag(x) ^ 1)

; Takes an arbitrary number of cpuflags from the above list.
; All subsequent functions (up to the next INIT_CPUFLAGS) is built for the specified cpu.
; You shouldn't need to invoke this macro directly, it's a subroutine for INIT_MMX &co.
%macro INIT_CPUFLAGS 0-*
    %xdefine SUFFIX
    %undef cpuname
    %assign cpuflags 0

    %if %0 >= 1
        %rep %0
            %ifdef cpuname
                %xdefine cpuname cpuname %+ _%1
            %else
                %xdefine cpuname %1
            %endif
            %assign cpuflags cpuflags | cpuflags_%1
            %rotate 1
        %endrep
        %xdefine SUFFIX _ %+ cpuname

        %if cpuflag(avx)
            %assign avx_enabled 1
        %endif
        %if (mmsize == 16 && notcpuflag(sse2)) || (mmsize == 32 && notcpuflag(avx2))
            %define mova movaps
            %define movu movups
            %define movnta movntps
        %endif
        %if cpuflag(aligned)
            %define movu mova
        %elif cpuflag(sse3) && notcpuflag(ssse3)
            %define movu lddqu
        %endif
    %endif

    %if ARCH_X86_64 || cpuflag(sse2)
        %ifdef __NASM_VER__
            ALIGNMODE k8
        %else
            CPU amdnop
        %endif
    %else
        %ifdef __NASM_VER__
            ALIGNMODE nop
        %else
            CPU basicnop
        %endif
    %endif
%endmacro

; Merge mmx and sse*
; m# is a simd register of the currently selected size
; xm# is the corresponding xmm register if mmsize >= 16, otherwise the same as m#
; ym# is the corresponding ymm register if mmsize >= 32, otherwise the same as m#
; (All 3 remain in sync through SWAP.)

%macro CAT_XDEFINE 3
    %xdefine %1%2 %3
%endmacro

%macro CAT_UNDEF 2
    %undef %1%2
%endmacro

%macro INIT_MMX 0-1+
    %assign avx_enabled 0
    %define RESET_MM_PERMUTATION INIT_MMX %1
    %define mmsize 8
    %define num_mmregs 8
    %define mova movq
    %define movu movq
    %define movh movd
    %define movnta movntq
    %assign %%i 0
    %rep 8
        CAT_XDEFINE m, %%i, mm %+ %%i
        CAT_XDEFINE nnmm, %%i, %%i
        %assign %%i %%i+1
    %endrep
    %rep 8
        CAT_UNDEF m, %%i
        CAT_UNDEF nnmm, %%i
        %assign %%i %%i+1
    %endrep
    INIT_CPUFLAGS %1
%endmacro

%macro INIT_XMM 0-1+
    %assign avx_enabled 0
    %define RESET_MM_PERMUTATION INIT_XMM %1
    %define mmsize 16
    %define num_mmregs 8
    %if ARCH_X86_64
        %define num_mmregs 16
    %endif
    %define mova movdqa
    %define movu movdqu
    %define movh movq
    %define movnta movntdq
    %assign %%i 0
    %rep num_mmregs
        CAT_XDEFINE m, %%i, xmm %+ %%i
        CAT_XDEFINE nnxmm, %%i, %%i
        %assign %%i %%i+1
    %endrep
    INIT_CPUFLAGS %1
%endmacro

%macro INIT_YMM 0-1+
    %assign avx_enabled 1
    %define RESET_MM_PERMUTATION INIT_YMM %1
    %define mmsize 32
    %define num_mmregs 8
    %if ARCH_X86_64
        %define num_mmregs 16
    %endif
    %define mova movdqa
    %define movu movdqu
    %undef movh
    %define movnta movntdq
    %assign %%i 0
    %rep num_mmregs
        CAT_XDEFINE m, %%i, ymm %+ %%i
        CAT_XDEFINE nnymm, %%i, %%i
        %assign %%i %%i+1
    %endrep
    INIT_CPUFLAGS %1
%endmacro

INIT_XMM

%macro DECLARE_MMCAST 1
    %define  mmmm%1   mm%1
    %define  mmxmm%1  mm%1
    %define  mmymm%1  mm%1
    %define xmmmm%1   mm%1
    %define xmmxmm%1 xmm%1
    %define xmmymm%1 xmm%1
    %define ymmmm%1   mm%1
    %define ymmxmm%1 xmm%1
    %define ymmymm%1 ymm%1
    %define xm%1 xmm %+ m%1
    %define ym%1 ymm %+ m%1
%endmacro

%assign i 0
%rep 16
    DECLARE_MMCAST i
    %assign i i+1
%endrep

; I often want to use macros that permute their arguments. e.g. there's no
; efficient way to implement butterfly or transpose or dct without swapping some
; arguments.
;
; I would like to not have to manually keep track of the permutations:
; If I insert a permutation in the middle of a function, it should automatically
; change everything that follows. For more complex macros I may also have multiple
; implementations, e.g. the SSE2 and SSSE3 versions may have different permutations.
;
; Hence these macros. Insert a PERMUTE or some SWAPs at the end of a macro that
; permutes its arguments. It's equivalent to exchanging the contents of the
; registers, except that this way you exchange the register names instead, so it
; doesn't cost any cycles.

%macro PERMUTE 2-* ; takes a list of pairs to swap
    %rep %0/2
        %xdefine %%tmp%2 m%2
        %rotate 2
    %endrep
    %rep %0/2
        %xdefine m%1 %%tmp%2
        CAT_XDEFINE nn, m%1, %1
        %rotate 2
    %endrep
%endmacro

%macro SWAP 2+ ; swaps a single chain (sometimes more concise than pairs)
    %ifnum %1 ; SWAP 0, 1, ...
        SWAP_INTERNAL_NUM %1, %2
    %else ; SWAP m0, m1, ...
        SWAP_INTERNAL_NAME %1, %2
    %endif
%endmacro

%macro SWAP_INTERNAL_NUM 2-*
    %rep %0-1
        %xdefine %%tmp m%1
        %xdefine m%1 m%2
        %xdefine m%2 %%tmp
        CAT_XDEFINE nn, m%1, %1
        CAT_XDEFINE nn, m%2, %2
        %rotate 1
    %endrep
%endmacro

%macro SWAP_INTERNAL_NAME 2-*
    %xdefine %%args nn %+ %1
    %rep %0-1
        %xdefine %%args %%args, nn %+ %2
        %rotate 1
    %endrep
    SWAP_INTERNAL_NUM %%args
%endmacro

; If SAVE_MM_PERMUTATION is placed at the end of a function, then any later
; calls to that function will automatically load the permutation, so values can
; be returned in mmregs.
%macro SAVE_MM_PERMUTATION 0-1
    %if %0
        %xdefine %%f %1_m
    %else
        %xdefine %%f current_function %+ _m
    %endif
    %assign %%i 0
    %rep num_mmregs
        CAT_XDEFINE %%f, %%i, m %+ %%i
        %assign %%i %%i+1
    %endrep
%endmacro

%macro LOAD_MM_PERMUTATION 1 ; name to load from
    %ifdef %1_m0
        %assign %%i 0
        %rep num_mmregs
            CAT_XDEFINE m, %%i, %1_m %+ %%i
            CAT_XDEFINE nn, m %+ %%i, %%i
            %assign %%i %%i+1
        %endrep
    %endif
%endmacro

; Append cpuflags to the callee's name iff the appended name is known and the plain name isn't
%macro call 1
    call_internal %1 %+ SUFFIX, %1
%endmacro
%macro call_internal 2
    %xdefine %%i %2
    %ifndef cglobaled_%2
        %ifdef cglobaled_%1
            %xdefine %%i %1
        %endif
    %endif
    call %%i
    LOAD_MM_PERMUTATION %%i
%endmacro

; Substitutions that reduce instruction size but are functionally equivalent
%macro add 2
    %ifnum %2
        %if %2==128
            sub %1, -128
        %else
            add %1, %2
        %endif
    %else
        add %1, %2
    %endif
%endmacro

%macro sub 2
    %ifnum %2
        %if %2==128
            add %1, -128
        %else
            sub %1, %2
        %endif
    %else
        sub %1, %2
    %endif
%endmacro

;=============================================================================
; AVX abstraction layer
;=============================================================================

%assign i 0
%rep 16
    %if i < 8
        CAT_XDEFINE sizeofmm, i, 8
    %endif
    CAT_XDEFINE sizeofxmm, i, 16
    CAT_XDEFINE sizeofymm, i, 32
    %assign i i+1
%endrep
%undef i

%macro CHECK_AVX_INSTR_EMU 3-*
    %xdefine %%opcode %1
    %xdefine %%dst %2
    %rep %0-2
        %ifidn %%dst, %3
            %error non-avx emulation of ``%%opcode'' is not supported
        %endif
        %rotate 1
    %endrep
%endmacro

;%1 == instruction
;%2 == minimal instruction set
;%3 == 1 if float, 0 if int
;%4 == 1 if non-destructive or 4-operand (xmm, xmm, xmm, imm), 0 otherwise
;%5 == 1 if commutative (i.e. doesn't matter which src arg is which), 0 if not
;%6+: operands
%macro RUN_AVX_INSTR 6-9+
    %ifnum sizeof%7
        %assign __sizeofreg sizeof%7
    %elifnum sizeof%6
        %assign __sizeofreg sizeof%6
    %else
        %assign __sizeofreg mmsize
    %endif
    %assign __emulate_avx 0
    %if avx_enabled && __sizeofreg >= 16
        %xdefine __instr v%1
    %else
        %xdefine __instr %1
        %if %0 >= 8+%4
            %assign __emulate_avx 1
        %endif
    %endif
    %ifnidn %2, fnord
        %ifdef cpuname
            %if notcpuflag(%2)
                %error use of ``%1'' %2 instruction in cpuname function: current_function
            %elif cpuflags_%2 < cpuflags_sse && notcpuflag(sse2) && __sizeofreg > 8
                %error use of ``%1'' sse2 instruction in cpuname function: current_function
            %endif
        %endif
    %endif

    %if __emulate_avx
        %xdefine __src1 %7
        %xdefine __src2 %8
        %ifnidn %6, %7
            %if %0 >= 9
                CHECK_AVX_INSTR_EMU {%1 %6, %7, %8, %9}, %6, %8, %9
            %else
                CHECK_AVX_INSTR_EMU {%1 %6, %7, %8}, %6, %8
            %endif
            %if %5 && %4 == 0
                %ifnid %8
                    ; 3-operand AVX instructions with a memory arg can only have it in src2,
                    ; whereas SSE emulation prefers to have it in src1 (i.e. the mov).
                    ; So, if the instruction is commutative with a memory arg, swap them.
                    %xdefine __src1 %8
                    %xdefine __src2 %7
                %endif
            %endif
            %if __sizeofreg == 8
                MOVQ %6, __src1
            %elif %3
                MOVAPS %6, __src1
            %else
                MOVDQA %6, __src1
            %endif
        %endif
        %if %0 >= 9
            %1 %6, __src2, %9
        %else
            %1 %6, __src2
        %endif
    %elif %0 >= 9
        __instr %6, %7, %8, %9
    %elif %0 == 8
        __instr %6, %7, %8
    %elif %0 == 7
        __instr %6, %7
    %else
        __instr %6
    %endif
%endmacro

;%1 == instruction
;%2 == minimal instruction set
;%3 == 1 if float, 0 if int
;%4 == 1 if non-destructive or 4-operand (xmm, xmm, xmm, imm), 0 otherwise
;%5 == 1 if commutative (i.e. doesn't matter which src arg is which), 0 if not
%macro AVX_INSTR 1-5 fnord, 0, 1, 0
    %macro %1 1-10 fnord, fnord, fnord, fnord, %1, %2, %3, %4, %5
        %ifidn %2, fnord
            RUN_AVX_INSTR %6, %7, %8, %9, %10, %1
        %elifidn %3, fnord
            RUN_AVX_INSTR %6, %7, %8, %9, %10, %1, %2
        %elifidn %4, fnord
            RUN_AVX_INSTR %6, %7, %8, %9, %10, %1, %2, %3
        %elifidn %5, fnord
            RUN_AVX_INSTR %6, %7, %8, %9, %10, %1, %2, %3, %4
        %else
            RUN_AVX_INSTR %6, %7, %8, %9, %10, %1, %2, %3, %4, %5
        %endif
    %endmacro
%endmacro

; Instructions with both VEX and non-VEX encodings
; Non-destructive instructions are written without parameters
AVX_INSTR addpd, sse2, 1, 0, 1
AVX_INSTR addps, sse, 1, 0, 1
AVX_INSTR addsd, sse2, 1, 0, 1
AVX_INSTR addss, sse, 1, 0, 1
AVX_INSTR addsubpd, sse3, 1, 0, 0
AVX_INSTR addsubps, sse3, 1, 0, 0
AVX_INSTR aesdec, fnord, 0, 0, 0
AVX_INSTR aesdeclast, fnord, 0, 0, 0
AVX_INSTR aesenc, fnord, 0, 0, 0
AVX_INSTR aesenclast, fnord, 0, 0, 0
AVX_INSTR aesimc
AVX_INSTR aeskeygenassist
AVX_INSTR andnpd, sse2, 1, 0, 0
AVX_INSTR andnps, sse, 1, 0, 0
AVX_INSTR andpd, sse2, 1, 0, 1
AVX_INSTR andps, sse, 1, 0, 1
AVX_INSTR blendpd, sse4, 1, 0, 0
AVX_INSTR blendps, sse4, 1, 0, 0
AVX_INSTR blendvpd, sse4, 1, 0, 0
AVX_INSTR blendvps, sse4, 1, 0, 0
AVX_INSTR cmppd, sse2, 1, 1, 0
AVX_INSTR cmpps, sse, 1, 1, 0
AVX_INSTR cmpsd, sse2, 1, 1, 0
AVX_INSTR cmpss, sse, 1, 1, 0
AVX_INSTR comisd, sse2
AVX_INSTR comiss, sse
AVX_INSTR cvtdq2pd, sse2
AVX_INSTR cvtdq2ps, sse2
AVX_INSTR cvtpd2dq, sse2
AVX_INSTR cvtpd2ps, sse2
AVX_INSTR cvtps2dq, sse2
AVX_INSTR cvtps2pd, sse2
AVX_INSTR cvtsd2si, sse2
AVX_INSTR cvtsd2ss, sse2
AVX_INSTR cvtsi2sd, sse2
AVX_INSTR cvtsi2ss, sse
AVX_INSTR cvtss2sd, sse2
AVX_INSTR cvtss2si, sse
AVX_INSTR cvttpd2dq, sse2
AVX_INSTR cvttps2dq, sse2
AVX_INSTR cvttsd2si, sse2
AVX_INSTR cvttss2si, sse
AVX_INSTR divpd, sse2, 1, 0, 0
AVX_INSTR divps, sse, 1, 0, 0
AVX_INSTR divsd, sse2, 1, 0, 0
AVX_INSTR divss, sse, 1, 0, 0
AVX_INSTR dppd, sse4, 1, 1, 0
AVX_INSTR dpps, sse4, 1, 1, 0
AVX_INSTR extractps, sse4
AVX_INSTR haddpd, sse3, 1, 0, 0
AVX_INSTR haddps, sse3, 1, 0, 0
AVX_INSTR hsubpd, sse3, 1, 0, 0
AVX_INSTR hsubps, sse3, 1, 0, 0
AVX_INSTR insertps, sse4, 1, 1, 0
AVX_INSTR lddqu, sse3
AVX_INSTR ldmxcsr, sse
AVX_INSTR maskmovdqu, sse2
AVX_INSTR maxpd, sse2, 1, 0, 1
AVX_INSTR maxps, sse, 1, 0, 1
AVX_INSTR maxsd, sse2, 1, 0, 1
AVX_INSTR maxss, sse, 1, 0, 1
AVX_INSTR minpd, sse2, 1, 0, 1
AVX_INSTR minps, sse, 1, 0, 1
AVX_INSTR minsd, sse2, 1, 0, 1
AVX_INSTR minss, sse, 1, 0, 1
AVX_INSTR movapd, sse2
AVX_INSTR movaps, sse
AVX_INSTR movd, mmx
AVX_INSTR movddup, sse3
AVX_INSTR movdqa, sse2
AVX_INSTR movdqu, sse2
AVX_INSTR movhlps, sse, 1, 0, 0
AVX_INSTR movhpd, sse2, 1, 0, 0
AVX_INSTR movhps, sse, 1, 0, 0
AVX_INSTR movlhps, sse, 1, 0, 0
AVX_INSTR movlpd, sse2, 1, 0, 0
AVX_INSTR movlps, sse, 1, 0, 0
AVX_INSTR movmskpd, sse2
AVX_INSTR movmskps, sse
AVX_INSTR movntdq, sse2
AVX_INSTR movntdqa, sse4
AVX_INSTR movntpd, sse2
AVX_INSTR movntps, sse
AVX_INSTR movq, mmx
AVX_INSTR movsd, sse2, 1, 0, 0
AVX_INSTR movshdup, sse3
AVX_INSTR movsldup, sse3
AVX_INSTR movss, sse, 1, 0, 0
AVX_INSTR movupd, sse2
AVX_INSTR movups, sse
AVX_INSTR mpsadbw, sse4
AVX_INSTR mulpd, sse2, 1, 0, 1
AVX_INSTR mulps, sse, 1, 0, 1
AVX_INSTR mulsd, sse2, 1, 0, 1
AVX_INSTR mulss, sse, 1, 0, 1
AVX_INSTR orpd, sse2, 1, 0, 1
AVX_INSTR orps, sse, 1, 0, 1
AVX_INSTR pabsb, ssse3
AVX_INSTR pabsd, ssse3
AVX_INSTR pabsw, ssse3
AVX_INSTR packsswb, mmx, 0, 0, 0
AVX_INSTR packssdw, mmx, 0, 0, 0
AVX_INSTR packuswb, mmx, 0, 0, 0
AVX_INSTR packusdw, sse4, 0, 0, 0
AVX_INSTR paddb, mmx, 0, 0, 1
AVX_INSTR paddw, mmx, 0, 0, 1
AVX_INSTR paddd, mmx, 0, 0, 1
AVX_INSTR paddq, sse2, 0, 0, 1
AVX_INSTR paddsb, mmx, 0, 0, 1
AVX_INSTR paddsw, mmx, 0, 0, 1
AVX_INSTR paddusb, mmx, 0, 0, 1
AVX_INSTR paddusw, mmx, 0, 0, 1
AVX_INSTR palignr, ssse3
AVX_INSTR pand, mmx, 0, 0, 1
AVX_INSTR pandn, mmx, 0, 0, 0
AVX_INSTR pavgb, mmx2, 0, 0, 1
AVX_INSTR pavgw, mmx2, 0, 0, 1
AVX_INSTR pblendvb, sse4, 0, 0, 0
AVX_INSTR pblendw, sse4
AVX_INSTR pclmulqdq
AVX_INSTR pcmpestri, sse42
AVX_INSTR pcmpestrm, sse42
AVX_INSTR pcmpistri, sse42
AVX_INSTR pcmpistrm, sse42
AVX_INSTR pcmpeqb, mmx, 0, 0, 1
AVX_INSTR pcmpeqw, mmx, 0, 0, 1
AVX_INSTR pcmpeqd, mmx, 0, 0, 1
AVX_INSTR pcmpeqq, sse4, 0, 0, 1
AVX_INSTR pcmpgtb, mmx, 0, 0, 0
AVX_INSTR pcmpgtw, mmx, 0, 0, 0
AVX_INSTR pcmpgtd, mmx, 0, 0, 0
AVX_INSTR pcmpgtq, sse42, 0, 0, 0
AVX_INSTR pextrb, sse4
AVX_INSTR pextrd, sse4
AVX_INSTR pextrq, sse4
AVX_INSTR pextrw, mmx2
AVX_INSTR phaddw, ssse3, 0, 0, 0
AVX_INSTR phaddd, ssse3, 0, 0, 0
AVX_INSTR phaddsw, ssse3, 0, 0, 0
AVX_INSTR phminposuw, sse4
AVX_INSTR phsubw, ssse3, 0, 0, 0
AVX_INSTR phsubd, ssse3, 0, 0, 0
AVX_INSTR phsubsw, ssse3, 0, 0, 0
AVX_INSTR pinsrb, sse4
AVX_INSTR pinsrd, sse4
AVX_INSTR pinsrq, sse4
AVX_INSTR pinsrw, mmx2
AVX_INSTR pmaddwd, mmx, 0, 0, 1
AVX_INSTR pmaddubsw, ssse3, 0, 0, 0
AVX_INSTR pmaxsb, sse4, 0, 0, 1
AVX_INSTR pmaxsw, mmx2, 0, 0, 1
AVX_INSTR pmaxsd, sse4, 0, 0, 1
AVX_INSTR pmaxub, mmx2, 0, 0, 1
AVX_INSTR pmaxuw, sse4, 0, 0, 1
AVX_INSTR pmaxud, sse4, 0, 0, 1
AVX_INSTR pminsb, sse4, 0, 0, 1
AVX_INSTR pminsw, mmx2, 0, 0, 1
AVX_INSTR pminsd, sse4, 0, 0, 1
AVX_INSTR pminub, mmx2, 0, 0, 1
AVX_INSTR pminuw, sse4, 0, 0, 1
AVX_INSTR pminud, sse4, 0, 0, 1
AVX_INSTR pmovmskb, mmx2
AVX_INSTR pmovsxbw, sse4
AVX_INSTR pmovsxbd, sse4
AVX_INSTR pmovsxbq, sse4
AVX_INSTR pmovsxwd, sse4
AVX_INSTR pmovsxwq, sse4
AVX_INSTR pmovsxdq, sse4
AVX_INSTR pmovzxbw, sse4
AVX_INSTR pmovzxbd, sse4
AVX_INSTR pmovzxbq, sse4
AVX_INSTR pmovzxwd, sse4
AVX_INSTR pmovzxwq, sse4
AVX_INSTR pmovzxdq, sse4
AVX_INSTR pmuldq, sse4, 0, 0, 1
AVX_INSTR pmulhrsw, ssse3, 0, 0, 1
AVX_INSTR pmulhuw, mmx2, 0, 0, 1
AVX_INSTR pmulhw, mmx, 0, 0, 1
AVX_INSTR pmullw, mmx, 0, 0, 1
AVX_INSTR pmulld, sse4, 0, 0, 1
AVX_INSTR pmuludq, sse2, 0, 0, 1
AVX_INSTR por, mmx, 0, 0, 1
AVX_INSTR psadbw, mmx2, 0, 0, 1
AVX_INSTR pshufb, ssse3, 0, 0, 0
AVX_INSTR pshufd, sse2
AVX_INSTR pshufhw, sse2
AVX_INSTR pshuflw, sse2
AVX_INSTR psignb, ssse3, 0, 0, 0
AVX_INSTR psignw, ssse3, 0, 0, 0
AVX_INSTR psignd, ssse3, 0, 0, 0
AVX_INSTR psllw, mmx, 0, 0, 0
AVX_INSTR pslld, mmx, 0, 0, 0
AVX_INSTR psllq, mmx, 0, 0, 0
AVX_INSTR pslldq, sse2, 0, 0, 0
AVX_INSTR psraw, mmx, 0, 0, 0
AVX_INSTR psrad, mmx, 0, 0, 0
AVX_INSTR psrlw, mmx, 0, 0, 0
AVX_INSTR psrld, mmx, 0, 0, 0
AVX_INSTR psrlq, mmx, 0, 0, 0
AVX_INSTR psrldq, sse2, 0, 0, 0
AVX_INSTR psubb, mmx, 0, 0, 0
AVX_INSTR psubw, mmx, 0, 0, 0
AVX_INSTR psubd, mmx, 0, 0, 0
AVX_INSTR psubq, sse2, 0, 0, 0
AVX_INSTR psubsb, mmx, 0, 0, 0
AVX_INSTR psubsw, mmx, 0, 0, 0
AVX_INSTR psubusb, mmx, 0, 0, 0
AVX_INSTR psubusw, mmx, 0, 0, 0
AVX_INSTR ptest, sse4
AVX_INSTR punpckhbw, mmx, 0, 0, 0
AVX_INSTR punpckhwd, mmx, 0, 0, 0
AVX_INSTR punpckhdq, mmx, 0, 0, 0
AVX_INSTR punpckhqdq, sse2, 0, 0, 0
AVX_INSTR punpcklbw, mmx, 0, 0, 0
AVX_INSTR punpcklwd, mmx, 0, 0, 0
AVX_INSTR punpckldq, mmx, 0, 0, 0
AVX_INSTR punpcklqdq, sse2, 0, 0, 0
AVX_INSTR pxor, mmx, 0, 0, 1
AVX_INSTR rcpps, sse, 1, 0, 0
AVX_INSTR rcpss, sse, 1, 0, 0
AVX_INSTR roundpd, sse4
AVX_INSTR roundps, sse4
AVX_INSTR roundsd, sse4
AVX_INSTR roundss, sse4
AVX_INSTR rsqrtps, sse, 1, 0, 0
AVX_INSTR rsqrtss, sse, 1, 0, 0
AVX_INSTR shufpd, sse2, 1, 1, 0
AVX_INSTR shufps, sse, 1, 1, 0
AVX_INSTR sqrtpd, sse2, 1, 0, 0
AVX_INSTR sqrtps, sse, 1, 0, 0
AVX_INSTR sqrtsd, sse2, 1, 0, 0
AVX_INSTR sqrtss, sse, 1, 0, 0
AVX_INSTR stmxcsr, sse
AVX_INSTR subpd, sse2, 1, 0, 0
AVX_INSTR subps, sse, 1, 0, 0
AVX_INSTR subsd, sse2, 1, 0, 0
AVX_INSTR subss, sse, 1, 0, 0
AVX_INSTR ucomisd, sse2
AVX_INSTR ucomiss, sse
AVX_INSTR unpckhpd, sse2, 1, 0, 0
AVX_INSTR unpckhps, sse, 1, 0, 0
AVX_INSTR unpcklpd, sse2, 1, 0, 0
AVX_INSTR unpcklps, sse, 1, 0, 0
AVX_INSTR xorpd, sse2, 1, 0, 1
AVX_INSTR xorps, sse, 1, 0, 1

; 3DNow instructions, for sharing code between AVX, SSE and 3DN
AVX_INSTR pfadd, 3dnow, 1, 0, 1
AVX_INSTR pfsub, 3dnow, 1, 0, 0
AVX_INSTR pfmul, 3dnow, 1, 0, 1

; base-4 constants for shuffles
%assign i 0
%rep 256
    %assign j ((i>>6)&3)*1000 + ((i>>4)&3)*100 + ((i>>2)&3)*10 + (i&3)
    %if j < 10
        CAT_XDEFINE q000, j, i
    %elif j < 100
        CAT_XDEFINE q00, j, i
    %elif j < 1000
        CAT_XDEFINE q0, j, i
    %else
        CAT_XDEFINE q, j, i
    %endif
    %assign i i+1
%endrep
%undef i
%undef j

%macro FMA_INSTR 3
    %macro %1 4-7 %1, %2, %3
        %if cpuflag(xop)
            v%5 %1, %2, %3, %4
        %elifnidn %1, %4
            %6 %1, %2, %3
            %7 %1, %4
        %else
            %error non-xop emulation of ``%5 %1, %2, %3, %4'' is not supported
        %endif
    %endmacro
%endmacro

FMA_INSTR  pmacsww,  pmullw, paddw
FMA_INSTR  pmacsdd,  pmulld, paddd ; sse4 emulation
FMA_INSTR pmacsdql,  pmuldq, paddq ; sse4 emulation
FMA_INSTR pmadcswd, pmaddwd, paddd

; Macros for consolidating FMA3 and FMA4 using 4-operand (dst, src1, src2, src3) syntax.
; FMA3 is only possible if dst is the same as one of the src registers.
; Either src2 or src3 can be a memory operand.
%macro FMA4_INSTR 2-*
    %push fma4_instr
    %xdefine %$prefix %1
    %rep %0 - 1
        %macro %$prefix%2 4-6 %$prefix, %2
            %if notcpuflag(fma3) && notcpuflag(fma4)
                %error use of ``%5%6'' fma instruction in cpuname function: current_function
            %elif cpuflag(fma4)
                v%5%6 %1, %2, %3, %4
            %elifidn %1, %2
                ; If %3 or %4 is a memory operand it needs to be encoded as the last operand.
                %ifid %3
                    v%{5}213%6 %2, %3, %4
                %else
                    v%{5}132%6 %2, %4, %3
                %endif
            %elifidn %1, %3
                v%{5}213%6 %3, %2, %4
            %elifidn %1, %4
                v%{5}231%6 %4, %2, %3
            %else
                %error fma3 emulation of ``%5%6 %1, %2, %3, %4'' is not supported
            %endif
        %endmacro
        %rotate 1
    %endrep
    %pop
%endmacro

FMA4_INSTR fmadd,    pd, ps, sd, ss
FMA4_INSTR fmaddsub, pd, ps
FMA4_INSTR fmsub,    pd, ps, sd, ss
FMA4_INSTR fmsubadd, pd, ps
FMA4_INSTR fnmadd,   pd, ps, sd, ss
FMA4_INSTR fnmsub,   pd, ps, sd, ss

; workaround: vpbroadcastq is broken in x86_32 due to a yasm bug (fixed in 1.3.0)
%ifdef __YASM_VER__
    %if __YASM_VERSION_ID__ < 0x01030000 && ARCH_X86_64 == 0
        %macro vpbroadcastq 2
            %if sizeof%1 == 16
                movddup %1, %2
            %else
                vbroadcastsd %1, %2
            %endif
        %endmacro
    %endif
%endif
