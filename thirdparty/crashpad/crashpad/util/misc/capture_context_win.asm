; Copyright 2015 The Crashpad Authors. All rights reserved.
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

; Detect ml64 assembling for x86_64 by checking for rax.
ifdef rax
_M_X64 equ 1
else
_M_IX86 equ 1
endif

ifdef _M_IX86
.586
.xmm
.model flat
endif

offsetof macro structure, field
  exitm <structure.&field>
endm

; The CONTEXT structure definitions that follow are based on those in <winnt.h>.
; Field names are prefixed (as in c_Rax) to avoid colliding with the predefined
; register names (such as Rax).

ifdef _M_IX86

CONTEXT_i386 equ 10000h
CONTEXT_CONTROL equ CONTEXT_i386 or 1h
CONTEXT_INTEGER equ CONTEXT_i386 or 2h
CONTEXT_SEGMENTS equ CONTEXT_i386 or 4h
CONTEXT_FLOATING_POINT equ CONTEXT_i386 or 8h
CONTEXT_DEBUG_REGISTERS equ CONTEXT_i386 or 10h
CONTEXT_EXTENDED_REGISTERS equ CONTEXT_i386 or 20h
CONTEXT_XSTATE equ CONTEXT_i386 or 40h

MAXIMUM_SUPPORTED_EXTENSION equ 512

CONTEXT struct
  c_ContextFlags dword ?

  c_Dr0 dword ?
  c_Dr1 dword ?
  c_Dr2 dword ?
  c_Dr3 dword ?
  c_Dr6 dword ?
  c_Dr7 dword ?

  struct c_FloatSave
    f_ControlWord dword ?
    f_StatusWord dword ?
    f_TagWord dword ?
    f_ErrorOffset dword ?
    f_ErrorSelector dword ?
    f_DataOffset dword ?
    f_DataSelector dword ?
    f_RegisterArea byte 80 dup(?)

    union
      f_Spare0 dword ?  ; As in FLOATING_SAVE_AREA.
      f_Cr0NpxState dword ?  ; As in WOW64_FLOATING_SAVE_AREA.
    ends
  ends

  c_SegGs dword ?
  c_SegFs dword ?
  c_SegEs dword ?
  c_SegDs dword ?

  c_Edi dword ?
  c_Esi dword ?
  c_Ebx dword ?
  c_Edx dword ?
  c_Ecx dword ?
  c_Eax dword ?

  c_Ebp dword ?

  c_Eip dword ?
  c_SegCs dword ?

  c_EFlags dword ?

  c_Esp dword ?
  c_SegSs dword ?

  c_ExtendedRegisters byte MAXIMUM_SUPPORTED_EXTENSION dup(?)
CONTEXT ends

elseifdef _M_X64

M128A struct 16
  m_Low qword ?
  m_High qword ?
M128A ends

CONTEXT_AMD64 equ 100000h
CONTEXT_CONTROL equ CONTEXT_AMD64 or 1h
CONTEXT_INTEGER equ CONTEXT_AMD64 or 2h
CONTEXT_SEGMENTS equ CONTEXT_AMD64 or 4h
CONTEXT_FLOATING_POINT equ CONTEXT_AMD64 or 8h
CONTEXT_DEBUG_REGISTERS equ CONTEXT_AMD64 or 10h
CONTEXT_XSTATE equ CONTEXT_AMD64 or 40h

CONTEXT struct 16
  c_P1Home qword ?
  c_P2Home qword ?
  c_P3Home qword ?
  c_P4Home qword ?
  c_P5Home qword ?
  c_P6Home qword ?

  c_ContextFlags dword ?
  c_MxCsr dword ?

  c_SegCs word ?
  c_SegDs word ?
  c_SegEs word ?
  c_SegFs word ?
  c_SegGs word ?
  c_SegSs word ?

  c_EFlags dword ?

  c_Dr0 qword ?
  c_Dr1 qword ?
  c_Dr2 qword ?
  c_Dr3 qword ?
  c_Dr6 qword ?
  c_Dr7 qword ?

  c_Rax qword ?
  c_Rcx qword ?
  c_Rdx qword ?
  c_Rbx qword ?
  c_Rsp qword ?
  c_Rbp qword ?
  c_Rsi qword ?
  c_Rdi qword ?
  c_R8 qword ?
  c_R9 qword ?
  c_R10 qword ?
  c_R11 qword ?
  c_R12 qword ?
  c_R13 qword ?
  c_R14 qword ?
  c_R15 qword ?

  c_Rip qword ?

  union
    struct c_FltSave
      f_ControlWord word ?
      f_StatusWord word ?
      f_TagWord byte ?
      f_Reserved1 byte ?
      f_ErrorOpcode word ?
      f_ErrorOffset dword ?
      f_ErrorSelector word ?
      f_Reserved2 word ?
      f_DataOffset dword ?
      f_DataSelector word ?
      f_Reserved3 word ?
      f_MxCsr dword ?
      f_MxCsr_Mask dword ?
      f_FloatRegisters M128A 8 dup(<?>)
      f_XmmRegisters M128A 16 dup(<?>)
      f_Reserved4 byte 96 dup(?)
    ends
    struct
      fx_Header M128A 2 dup(<?>)
      fx_Legacy M128A 8 dup(<?>)
      fx_Xmm0 M128A <?>
      fx_Xmm1 M128A <?>
      fx_Xmm2 M128A <?>
      fx_Xmm3 M128A <?>
      fx_Xmm4 M128A <?>
      fx_Xmm5 M128A <?>
      fx_Xmm6 M128A <?>
      fx_Xmm7 M128A <?>
      fx_Xmm8 M128A <?>
      fx_Xmm9 M128A <?>
      fx_Xmm10 M128A <?>
      fx_Xmm11 M128A <?>
      fx_Xmm12 M128A <?>
      fx_Xmm13 M128A <?>
      fx_Xmm14 M128A <?>
      fx_Xmm15 M128A <?>
    ends
  ends

  c_VectorRegister M128A 26 dup(<?>)
  c_VectorControl qword ?

  c_DebugControl qword ?
  c_LastBranchToRip qword ?
  c_LastBranchFromRip qword ?
  c_LastExceptionToRip qword ?
  c_LastExceptionFromRip qword ?
CONTEXT ends

endif

; namespace crashpad {
; void CaptureContext(CONTEXT* context);
; }  // namespace crashpad
ifdef _M_IX86
CAPTURECONTEXT_SYMBOL equ ?CaptureContext@crashpad@@YAXPAU_CONTEXT@@@Z
elseifdef _M_X64
CAPTURECONTEXT_SYMBOL equ ?CaptureContext@crashpad@@YAXPEAU_CONTEXT@@@Z
endif

_TEXT segment
public CAPTURECONTEXT_SYMBOL

ifdef _M_IX86

CAPTURECONTEXT_SYMBOL proc

  push ebp
  mov ebp, esp

  ; pushfd first, because some instructions affect eflags. eflags will be in
  ; [ebp-4].
  pushfd

  ; Save the original value of ebx, and use ebx to hold the CONTEXT* argument.
  ; The original value of ebx will be in [ebp-8].
  push ebx
  mov ebx, [ebp+8]

  ; General-purpose registers whose values haven’t changed can be captured
  ; directly.
  mov [ebx.CONTEXT].c_Edi, edi
  mov [ebx.CONTEXT].c_Esi, esi
  mov [ebx.CONTEXT].c_Edx, edx
  mov [ebx.CONTEXT].c_Ecx, ecx
  mov [ebx.CONTEXT].c_Eax, eax

  ; Now that the original value of edx has been saved, it can be repurposed to
  ; hold other registers’ values.

  ; The original ebx was saved on the stack above.
  mov edx, dword ptr [ebp-8]
  mov [ebx.CONTEXT].c_Ebx, edx

  ; The original ebp was saved on the stack in this function’s prologue.
  mov edx, dword ptr [ebp]
  mov [ebx.CONTEXT].c_Ebp, edx

  ; eip can’t be accessed directly, but the return address saved on the stack
  ; by the call instruction that reached this function can be used.
  mov edx, dword ptr [ebp+4]
  mov [ebx.CONTEXT].c_Eip, edx

  ; The original eflags was saved on the stack above.
  mov edx, dword ptr [ebp-4]
  mov [ebx.CONTEXT].c_EFlags, edx

  ; esp was saved in ebp in this function’s prologue, but the caller’s esp is 8
  ; more than this value: 4 for the original ebp saved on the stack in this
  ; function’s prologue, and 4 for the return address saved on the stack by the
  ; call instruction that reached this function.
  lea edx, [ebp+8]
  mov [ebx.CONTEXT].c_Esp, edx

  ; The segment registers are 16 bits wide, but CONTEXT declares them as
  ; unsigned 32-bit values, so zero the top half.
  xor edx, edx
  mov dx, gs
  mov [ebx.CONTEXT].c_SegGs, edx
  mov dx, fs
  mov [ebx.CONTEXT].c_SegFs, edx
  mov dx, es
  mov [ebx.CONTEXT].c_SegEs, edx
  mov dx, ds
  mov [ebx.CONTEXT].c_SegDs, edx
  mov dx, cs
  mov [ebx.CONTEXT].c_SegCs, edx
  mov dx, ss
  mov [ebx.CONTEXT].c_SegSs, edx

  ; Prepare for the string move that will populate the ExtendedRegisters area,
  ; or the string store that will zero it.
  cld

  ; Use cpuid 1 to check whether fxsave is supported. If it is, perform it
  ; before fnsave because fxsave is a less-destructive operation.
  mov esi, ebx
  mov eax, 1
  cpuid
  mov ebx, esi

  test edx, 01000000  ; FXSR
  jnz $FXSave

  ; fxsave is not supported. Set ContextFlags to not include
  ; CONTEXT_EXTENDED_REGISTERS, and zero the ExtendedRegisters area.
  mov [ebx.CONTEXT].c_ContextFlags, CONTEXT_i386 or \
                                    CONTEXT_CONTROL or \
                                    CONTEXT_INTEGER or \
                                    CONTEXT_SEGMENTS or \
                                    CONTEXT_FLOATING_POINT
  lea edi, [ebx.CONTEXT].c_ExtendedRegisters
  xor eax, eax
  mov ecx, MAXIMUM_SUPPORTED_EXTENSION / sizeof(dword)  ; 128
  rep stosd
  jmp $FXSaveDone

$FXSave:
  ; fxsave is supported. Set ContextFlags to include CONTEXT_EXTENDED_REGISTERS.
  mov [ebx.CONTEXT].c_ContextFlags, CONTEXT_i386 or \
                                    CONTEXT_CONTROL or \
                                    CONTEXT_INTEGER or \
                                    CONTEXT_SEGMENTS or \
                                    CONTEXT_FLOATING_POINT or \
                                    CONTEXT_EXTENDED_REGISTERS

  ; fxsave requires a 16 byte-aligned destination memory area. Nothing
  ; guarantees the alignment of a CONTEXT structure, so create a temporary
  ; aligned fxsave destination on the stack.
  and esp, 0fffffff0h
  sub esp, MAXIMUM_SUPPORTED_EXTENSION

  ; Zero out the temporary fxsave area before performing the fxsave. Some of the
  ; fxsave area may not be written by fxsave, and some is definitely not written
  ; by fxsave.
  mov edi, esp
  xor eax, eax
  mov ecx, MAXIMUM_SUPPORTED_EXTENSION / sizeof(dword)  ; 128
  rep stosd

  fxsave [esp]

  ; Copy the temporary fxsave area into the CONTEXT structure.
  lea edi, [ebx.CONTEXT].c_ExtendedRegisters
  mov esi, esp
  mov ecx, MAXIMUM_SUPPORTED_EXTENSION / sizeof(dword)  ; 128
  rep movsd

  ; Free the stack space used for the temporary fxsave area.
  lea esp, [ebp-8]

  ; TODO(mark): AVX/xsave support. https://crashpad.chromium.org/bug/58

$FXSaveDone:
  ; fnsave reinitializes the FPU with an implicit finit operation, so use frstor
  ; to restore the original state.
  fnsave [ebx.CONTEXT].c_FloatSave
  frstor [ebx.CONTEXT].c_FloatSave

  ; cr0 is inaccessible from user code, and this field would not be used anyway.
  mov [ebx.CONTEXT].c_FloatSave.f_Cr0NpxState, 0

  ; The debug registers can’t be read from user code, so zero them out in the
  ; CONTEXT structure. context->ContextFlags doesn’t indicate that they are
  ; present.
  mov [ebx.CONTEXT].c_Dr0, 0
  mov [ebx.CONTEXT].c_Dr1, 0
  mov [ebx.CONTEXT].c_Dr2, 0
  mov [ebx.CONTEXT].c_Dr3, 0
  mov [ebx.CONTEXT].c_Dr6, 0
  mov [ebx.CONTEXT].c_Dr7, 0

  ; Clean up by restoring clobbered registers, even those considered volatile
  ; by the ABI, so that the captured context represents the state at this
  ; function’s exit.
  mov edi, [ebx.CONTEXT].c_Edi
  mov esi, [ebx.CONTEXT].c_Esi
  mov edx, [ebx.CONTEXT].c_Edx
  mov ecx, [ebx.CONTEXT].c_Ecx
  mov eax, [ebx.CONTEXT].c_Eax
  pop ebx
  popfd

  pop ebp

  ret

CAPTURECONTEXT_SYMBOL endp

elseifdef _M_X64

CAPTURECONTEXT_SYMBOL proc frame

  push rbp
  .pushreg rbp
  mov rbp, rsp
  .setframe rbp, 0

  ; Note that 16-byte stack alignment is not maintained because this function
  ; does not call out to any other.

  ; pushfq first, because some instructions affect rflags. rflags will be in
  ; [rbp-8].
  pushfq
  .allocstack 8
  .endprolog

  mov [rcx.CONTEXT].c_ContextFlags, CONTEXT_AMD64 or \
                                    CONTEXT_CONTROL or \
                                    CONTEXT_INTEGER or \
                                    CONTEXT_SEGMENTS or \
                                    CONTEXT_FLOATING_POINT

  ; General-purpose registers whose values haven’t changed can be captured
  ; directly.
  mov [rcx.CONTEXT].c_Rax, rax
  mov [rcx.CONTEXT].c_Rdx, rdx
  mov [rcx.CONTEXT].c_Rbx, rbx
  mov [rcx.CONTEXT].c_Rsi, rsi
  mov [rcx.CONTEXT].c_Rdi, rdi
  mov [rcx.CONTEXT].c_R8, r8
  mov [rcx.CONTEXT].c_R9, r9
  mov [rcx.CONTEXT].c_R10, r10
  mov [rcx.CONTEXT].c_R11, r11
  mov [rcx.CONTEXT].c_R12, r12
  mov [rcx.CONTEXT].c_R13, r13
  mov [rcx.CONTEXT].c_R14, r14
  mov [rcx.CONTEXT].c_R15, r15

  ; Because of the calling convention, there’s no way to recover the value of
  ; the caller’s rcx as it existed prior to calling this function. This
  ; function captures a snapshot of the register state at its return, which
  ; involves rcx containing a pointer to its first argument.
  mov [rcx.CONTEXT].c_Rcx, rcx

  ; Now that the original value of rax has been saved, it can be repurposed to
  ; hold other registers’ values.

  ; Save mxcsr. This is duplicated in context->FltSave.MxCsr, saved by fxsave
  ; below.
  stmxcsr [rcx.CONTEXT].c_MxCsr

  ; Segment registers.
  mov [rcx.CONTEXT].c_SegCs, cs
  mov [rcx.CONTEXT].c_SegDs, ds
  mov [rcx.CONTEXT].c_SegEs, es
  mov [rcx.CONTEXT].c_SegFs, fs
  mov [rcx.CONTEXT].c_SegGs, gs
  mov [rcx.CONTEXT].c_SegSs, ss

  ; The original rflags was saved on the stack above. Note that the CONTEXT
  ; structure only stores eflags, the low 32 bits. The high 32 bits in rflags
  ; are reserved.
  mov rax, qword ptr [rbp-8]
  mov [rcx.CONTEXT].c_EFlags, eax

  ; rsp was saved in rbp in this function’s prologue, but the caller’s rsp is
  ; 16 more than this value: 8 for the original rbp saved on the stack in this
  ; function’s prologue, and 8 for the return address saved on the stack by the
  ; call instruction that reached this function.
  lea rax, [rbp+16]
  mov [rcx.CONTEXT].c_Rsp, rax

  ; The original rbp was saved on the stack in this function’s prologue.
  mov rax, qword ptr [rbp]
  mov [rcx.CONTEXT].c_Rbp, rax

  ; rip can’t be accessed directly, but the return address saved on the stack by
  ; the call instruction that reached this function can be used.
  mov rax, qword ptr [rbp+8]
  mov [rcx.CONTEXT].c_Rip, rax

  ; Zero out the fxsave area before performing the fxsave. Some of the fxsave
  ; area may not be written by fxsave, and some is definitely not written by
  ; fxsave. This also zeroes out the rest of the CONTEXT structure to its end,
  ; including the unused VectorRegister and VectorControl fields, and the debug
  ; control register fields.
  mov rbx, rcx
  cld
  lea rdi, [rcx.CONTEXT].c_FltSave
  xor rax, rax
  mov rcx, (sizeof(CONTEXT) - offsetof(CONTEXT, c_FltSave)) / \
           sizeof(qword)  ; 122
  rep stosq
  mov rcx, rbx

  ; Save the floating point (including SSE) state. The CONTEXT structure is
  ; declared as 16-byte-aligned, which is correct for this operation.
  fxsave [rcx.CONTEXT].c_FltSave

  ; TODO(mark): AVX/xsave support. https://crashpad.chromium.org/bug/58

  ; The register parameter home address fields aren’t used, so zero them out.
  mov [rcx.CONTEXT].c_P1Home, 0
  mov [rcx.CONTEXT].c_P2Home, 0
  mov [rcx.CONTEXT].c_P3Home, 0
  mov [rcx.CONTEXT].c_P4Home, 0
  mov [rcx.CONTEXT].c_P5Home, 0
  mov [rcx.CONTEXT].c_P6Home, 0

  ; The debug registers can’t be read from user code, so zero them out in the
  ; CONTEXT structure. context->ContextFlags doesn’t indicate that they are
  ; present.
  mov [rcx.CONTEXT].c_Dr0, 0
  mov [rcx.CONTEXT].c_Dr1, 0
  mov [rcx.CONTEXT].c_Dr2, 0
  mov [rcx.CONTEXT].c_Dr3, 0
  mov [rcx.CONTEXT].c_Dr6, 0
  mov [rcx.CONTEXT].c_Dr7, 0

  ; Clean up by restoring clobbered registers, even those considered volatile by
  ; the ABI, so that the captured context represents the state at this
  ; function’s exit.
  mov rax, [rcx.CONTEXT].c_Rax
  mov rbx, [rcx.CONTEXT].c_Rbx
  mov rdi, [rcx.CONTEXT].c_Rdi
  popfq

  pop rbp

  ret

CAPTURECONTEXT_SYMBOL endp

endif

_TEXT ends
end
