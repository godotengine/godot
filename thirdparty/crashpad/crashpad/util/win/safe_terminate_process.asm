; Copyright 2017 The Crashpad Authors. All rights reserved.
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

includelib kernel32.lib

extern __imp__TerminateProcess@8:proc

; namespace crashpad {
; bool SafeTerminateProcess(HANDLE process, UINT exit_code);
; }  // namespace crashpad
SAFETERMINATEPROCESS_SYMBOL equ ?SafeTerminateProcess@crashpad@@YA_NPAXI@Z

_TEXT segment
public SAFETERMINATEPROCESS_SYMBOL

SAFETERMINATEPROCESS_SYMBOL proc

  ; This function is written in assembler source because it’s important for it
  ; to not be inlined, for it to allocate a stack frame, and most critically,
  ; for it to not trust esp on return from TerminateProcess().
  ; __declspec(noinline) can prevent inlining and #pragma optimize("y", off) can
  ; disable frame pointer omission, but there’s no way to force a C compiler to
  ; distrust esp, and even if there was a way, it’d probably be fragile.

  push ebp
  mov ebp, esp

  push [ebp+12]
  push [ebp+8]
  call dword ptr [__imp__TerminateProcess@8]

  ; Convert from BOOL to bool.
  test eax, eax
  setne al

  ; TerminateProcess() is supposed to be stdcall (callee clean-up), and esp and
  ; ebp are expected to already be equal. But if it’s been patched badly by
  ; something that’s cdecl (caller clean-up), this next move will get things
  ; back on track.
  mov esp, ebp
  pop ebp

  ret

SAFETERMINATEPROCESS_SYMBOL endp

_TEXT ends

endif

end
