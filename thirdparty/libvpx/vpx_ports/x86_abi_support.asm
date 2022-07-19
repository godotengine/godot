;
;  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;


%include "vpx_config.asm"

; 32/64 bit compatibility macros
;
; In general, we make the source use 64 bit syntax, then twiddle with it using
; the preprocessor to get the 32 bit syntax on 32 bit platforms.
;
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
%define rax eax
%define rbx ebx
%define rcx ecx
%define rdx edx
%define rsi esi
%define rdi edi
%define rsp esp
%define rbp ebp
%define movsxd mov
%macro movq 2
  %ifidn %1,eax
    movd %1,%2
  %elifidn %2,eax
    movd %1,%2
  %elifidn %1,ebx
    movd %1,%2
  %elifidn %2,ebx
    movd %1,%2
  %elifidn %1,ecx
    movd %1,%2
  %elifidn %2,ecx
    movd %1,%2
  %elifidn %1,edx
    movd %1,%2
  %elifidn %2,edx
    movd %1,%2
  %elifidn %1,esi
    movd %1,%2
  %elifidn %2,esi
    movd %1,%2
  %elifidn %1,edi
    movd %1,%2
  %elifidn %2,edi
    movd %1,%2
  %elifidn %1,esp
    movd %1,%2
  %elifidn %2,esp
    movd %1,%2
  %elifidn %1,ebp
    movd %1,%2
  %elifidn %2,ebp
    movd %1,%2
  %else
    movq %1,%2
  %endif
%endmacro
%endif


; LIBVPX_YASM_WIN64
; Set LIBVPX_YASM_WIN64 if output is Windows 64bit so the code will work if x64
; or win64 is defined on the Yasm command line.
%ifidn __OUTPUT_FORMAT__,win64
%define LIBVPX_YASM_WIN64 1
%elifidn __OUTPUT_FORMAT__,x64
%define LIBVPX_YASM_WIN64 1
%else
%define LIBVPX_YASM_WIN64 0
%endif

; sym()
; Return the proper symbol name for the target ABI.
;
; Certain ABIs, notably MS COFF and Darwin MACH-O, require that symbols
; with C linkage be prefixed with an underscore.
;
%ifidn   __OUTPUT_FORMAT__,elf32
%define sym(x) x
%elifidn __OUTPUT_FORMAT__,elf64
%define sym(x) x
%elifidn __OUTPUT_FORMAT__,elfx32
%define sym(x) x
%elif LIBVPX_YASM_WIN64
%define sym(x) x
%else
%define sym(x) _ %+ x
%endif

;  PRIVATE
;  Macro for the attribute to hide a global symbol for the target ABI.
;  This is only active if CHROMIUM is defined.
;
;  Chromium doesn't like exported global symbols due to symbol clashing with
;  plugins among other things.
;
;  Requires Chromium's patched copy of yasm:
;    http://src.chromium.org/viewvc/chrome?view=rev&revision=73761
;    http://www.tortall.net/projects/yasm/ticket/236
;
%ifdef CHROMIUM
  %ifidn   __OUTPUT_FORMAT__,elf32
    %define PRIVATE :hidden
  %elifidn __OUTPUT_FORMAT__,elf64
    %define PRIVATE :hidden
  %elifidn __OUTPUT_FORMAT__,elfx32
    %define PRIVATE :hidden
  %elif LIBVPX_YASM_WIN64
    %define PRIVATE
  %else
    %define PRIVATE :private_extern
  %endif
%else
  %define PRIVATE
%endif

; arg()
; Return the address specification of the given argument
;
%if ABI_IS_32BIT
  %define arg(x) [ebp+8+4*x]
%else
  ; 64 bit ABI passes arguments in registers. This is a workaround to get up
  ; and running quickly. Relies on SHADOW_ARGS_TO_STACK
  %if LIBVPX_YASM_WIN64
    %define arg(x) [rbp+16+8*x]
  %else
    %define arg(x) [rbp-8-8*x]
  %endif
%endif

; REG_SZ_BYTES, REG_SZ_BITS
; Size of a register
%if ABI_IS_32BIT
%define REG_SZ_BYTES 4
%define REG_SZ_BITS  32
%else
%define REG_SZ_BYTES 8
%define REG_SZ_BITS  64
%endif


; ALIGN_STACK <alignment> <register>
; This macro aligns the stack to the given alignment (in bytes). The stack
; is left such that the previous value of the stack pointer is the first
; argument on the stack (ie, the inverse of this macro is 'pop rsp.')
; This macro uses one temporary register, which is not preserved, and thus
; must be specified as an argument.
%macro ALIGN_STACK 2
    mov         %2, rsp
    and         rsp, -%1
    lea         rsp, [rsp - (%1 - REG_SZ_BYTES)]
    push        %2
%endmacro


;
; The Microsoft assembler tries to impose a certain amount of type safety in
; its register usage. YASM doesn't recognize these directives, so we just
; %define them away to maintain as much compatibility as possible with the
; original inline assembler we're porting from.
;
%idefine PTR
%idefine XMMWORD
%idefine MMWORD

; PIC macros
;
%if ABI_IS_32BIT
  %if CONFIG_PIC=1
  %ifidn __OUTPUT_FORMAT__,elf32
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
  %endif
  %endif

  %ifdef CHROMIUM
    %ifidn __OUTPUT_FORMAT__,macho32
      %define HIDDEN_DATA(x) x:private_extern
    %else
      %define HIDDEN_DATA(x) x
    %endif
  %else
    %define HIDDEN_DATA(x) x
  %endif
%else
  %macro GET_GOT 1
  %endmacro
  %define GLOBAL(x) rel x
  %ifidn __OUTPUT_FORMAT__,elf64
    %define WRT_PLT wrt ..plt
    %define HIDDEN_DATA(x) x:data hidden
  %elifidn __OUTPUT_FORMAT__,elfx32
    %define WRT_PLT wrt ..plt
    %define HIDDEN_DATA(x) x:data hidden
  %elifidn __OUTPUT_FORMAT__,macho64
    %ifdef CHROMIUM
      %define HIDDEN_DATA(x) x:private_extern
    %else
      %define HIDDEN_DATA(x) x
    %endif
  %else
    %define HIDDEN_DATA(x) x
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

%if ABI_IS_32BIT
  %macro SHADOW_ARGS_TO_STACK 1
  %endm
  %define UNSHADOW_ARGS
%else
%if LIBVPX_YASM_WIN64
  %macro SHADOW_ARGS_TO_STACK 1 ; argc
    %if %1 > 0
        mov arg(0),rcx
    %endif
    %if %1 > 1
        mov arg(1),rdx
    %endif
    %if %1 > 2
        mov arg(2),r8
    %endif
    %if %1 > 3
        mov arg(3),r9
    %endif
  %endm
%else
  %macro SHADOW_ARGS_TO_STACK 1 ; argc
    %if %1 > 0
        push rdi
    %endif
    %if %1 > 1
        push rsi
    %endif
    %if %1 > 2
        push rdx
    %endif
    %if %1 > 3
        push rcx
    %endif
    %if %1 > 4
        push r8
    %endif
    %if %1 > 5
        push r9
    %endif
    %if %1 > 6
      %assign i %1-6
      %assign off 16
      %rep i
        mov rax,[rbp+off]
        push rax
        %assign off off+8
      %endrep
    %endif
  %endm
%endif
  %define UNSHADOW_ARGS mov rsp, rbp
%endif

; Win64 ABI requires that XMM6:XMM15 are callee saved
; SAVE_XMM n, [u]
; store registers 6-n on the stack
; if u is specified, use unaligned movs.
; Win64 ABI requires 16 byte stack alignment, but then pushes an 8 byte return
; value. Typically we follow this up with 'push rbp' - re-aligning the stack -
; but in some cases this is not done and unaligned movs must be used.
%if LIBVPX_YASM_WIN64
%macro SAVE_XMM 1-2 a
  %if %1 < 6
    %error Only xmm registers 6-15 must be preserved
  %else
    %assign last_xmm %1
    %define movxmm movdq %+ %2
    %assign xmm_stack_space ((last_xmm - 5) * 16)
    sub rsp, xmm_stack_space
    %assign i 6
    %rep (last_xmm - 5)
      movxmm [rsp + ((i - 6) * 16)], xmm %+ i
      %assign i i+1
    %endrep
  %endif
%endmacro
%macro RESTORE_XMM 0
  %ifndef last_xmm
    %error RESTORE_XMM must be paired with SAVE_XMM n
  %else
    %assign i last_xmm
    %rep (last_xmm - 5)
      movxmm xmm %+ i, [rsp +((i - 6) * 16)]
      %assign i i-1
    %endrep
    add rsp, xmm_stack_space
    ; there are a couple functions which return from multiple places.
    ; otherwise, we could uncomment these:
    ; %undef last_xmm
    ; %undef xmm_stack_space
    ; %undef movxmm
  %endif
%endmacro
%else
%macro SAVE_XMM 1-2
%endmacro
%macro RESTORE_XMM 0
%endmacro
%endif

; Name of the rodata section
;
; .rodata seems to be an elf-ism, as it doesn't work on OSX.
;
%ifidn __OUTPUT_FORMAT__,macho64
%define SECTION_RODATA section .text
%elifidn __OUTPUT_FORMAT__,macho32
%macro SECTION_RODATA 0
section .text
%endmacro
%elifidn __OUTPUT_FORMAT__,aout
%define SECTION_RODATA section .data
%else
%define SECTION_RODATA section .rodata
%endif


; Tell GNU ld that we don't require an executable stack.
%ifidn __OUTPUT_FORMAT__,elf32
section .note.GNU-stack noalloc noexec nowrite progbits
section .text
%elifidn __OUTPUT_FORMAT__,elf64
section .note.GNU-stack noalloc noexec nowrite progbits
section .text
%elifidn __OUTPUT_FORMAT__,elfx32
section .note.GNU-stack noalloc noexec nowrite progbits
section .text
%endif

; On Android platforms use lrand48 when building postproc routines. Prior to L
; rand() was not available.
%if CONFIG_POSTPROC=1 || CONFIG_VP9_POSTPROC=1
%ifdef __ANDROID__
extern sym(lrand48)
%define LIBVPX_RAND lrand48
%else
extern sym(rand)
%define LIBVPX_RAND rand
%endif
%endif ; CONFIG_POSTPROC || CONFIG_VP9_POSTPROC
