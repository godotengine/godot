; XzCrc64Opt.asm -- CRC64 calculation : optimized version
; 2023-12-08 : Igor Pavlov : Public domain

include 7zAsm.asm

MY_ASM_START

NUM_WORDS       equ     3

if (NUM_WORDS lt 1) or (NUM_WORDS gt 64)
.err <num_words_IS_INCORRECT>
endif

NUM_SKIP_BYTES  equ     ((NUM_WORDS - 2) * 4)


MOVZXLO macro dest:req, src:req
        movzx   dest, @CatStr(src, _L)
endm

MOVZXHI macro dest:req, src:req
        movzx   dest, @CatStr(src, _H)
endm


ifdef x64

rD      equ  r11
rN      equ  r10
rT      equ  r9
  
CRC_OP macro op:req, dest:req, src:req, t:req
        op      dest, QWORD PTR [rT + @CatStr(src, _R) * 8 + 0800h * (t)]
endm
    
CRC_XOR macro dest:req, src:req, t:req
        CRC_OP  xor, dest, src, t
endm

CRC_MOV macro dest:req, src:req, t:req
        CRC_OP  mov, dest, src, t
endm

CRC1b macro
        movzx   x6, BYTE PTR [rD]
        inc     rD
        MOVZXLO x3, x0
        xor     x6, x3
        shr     r0, 8
        CRC_XOR r0, x6, 0
        dec     rN
endm


; ALIGN_MASK is 3 or 7 bytes alignment:
ALIGN_MASK      equ  (7 - (NUM_WORDS and 1) * 4)

if NUM_WORDS eq 1

src_rN_offset   equ  4
; + 4 for prefetching next 4-bytes after current iteration
NUM_BYTES_LIMIT equ  (NUM_WORDS * 4 + 4)
SRCDAT4         equ  DWORD PTR [rN + rD * 1]

XOR_NEXT macro
        mov     x1, [rD]
        xor     r0, r1
endm

else ; NUM_WORDS > 1

src_rN_offset   equ 8
; + 8 for prefetching next 8-bytes after current iteration
NUM_BYTES_LIMIT equ (NUM_WORDS * 4 + 8)

XOR_NEXT macro
        xor     r0, QWORD PTR [rD] ; 64-bit read, can be unaligned
endm

; 32-bit or 64-bit
LOAD_SRC_MULT4 macro dest:req, word_index:req
        mov     dest, [rN + rD * 1 + 4 * (word_index) - src_rN_offset];
endm

endif



MY_PROC @CatStr(XzCrc64UpdateT, %(NUM_WORDS * 4)), 4
        MY_PUSH_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11

        mov     r0, REG_ABI_PARAM_0   ; r0  <- r1 / r7
        mov     rD, REG_ABI_PARAM_1   ; r11 <- r2 / r6
        mov     rN, REG_ABI_PARAM_2   ; r10 <- r8 / r2
if  (IS_LINUX gt 0)
        mov     rT, REG_ABI_PARAM_3   ; r9  <- r9 / r1
endif

        cmp     rN, NUM_BYTES_LIMIT + ALIGN_MASK
        jb      crc_end
@@:
        test    rD, ALIGN_MASK
        jz      @F
        CRC1b
        jmp     @B
@@:
        XOR_NEXT
        lea     rN, [rD + rN * 1 - (NUM_BYTES_LIMIT - 1)]
        sub     rD, rN
        add     rN, src_rN_offset

align 16
@@:

if NUM_WORDS eq 1
  
        mov     x1, x0
        shr     x1, 8
        MOVZXLO x3, x1
        MOVZXLO x2, x0
        shr     x1, 8
        shr     r0, 32
        xor     x0, SRCDAT4
        CRC_XOR r0, x2, 3
        CRC_XOR r0, x3, 2
        MOVZXLO x2, x1
        shr     x1, 8
        CRC_XOR r0, x2, 1
        CRC_XOR r0, x1, 0

else ; NUM_WORDS > 1
    
if NUM_WORDS ne 2
  k = 2
  while k lt NUM_WORDS

        LOAD_SRC_MULT4  x1, k
    crc_op1  textequ <xor>

    if k eq 2
      if (NUM_WORDS and 1)
        LOAD_SRC_MULT4  x7, NUM_WORDS       ; aligned 32-bit
        LOAD_SRC_MULT4  x6, NUM_WORDS + 1   ; aligned 32-bit
        shl     r6, 32
      else
        LOAD_SRC_MULT4  r6, NUM_WORDS       ; aligned 64-bit
        crc_op1  textequ <mov>
      endif
    endif
        table = 4 * (NUM_WORDS - 1 - k)
        MOVZXLO x3, x1
        CRC_OP crc_op1, r7, x3, 3 + table
        MOVZXHI x3, x1
        shr     x1, 16
        CRC_XOR r6, x3, 2 + table
        MOVZXLO x3, x1
        shr     x1, 8
        CRC_XOR r7, x3, 1 + table
        CRC_XOR r6, x1, 0 + table
        k = k + 1
  endm
        crc_op2  textequ <xor>

else ; NUM_WORDS == 2
        LOAD_SRC_MULT4  r6, NUM_WORDS       ; aligned 64-bit
        crc_op2  textequ <mov>
endif ; NUM_WORDS == 2

        MOVZXHI x3, x0
        MOVZXLO x2, x0
        mov     r1, r0
        shr     r1, 32
        shr     x0, 16
        CRC_XOR r6, x2, NUM_SKIP_BYTES + 7
        CRC_OP  crc_op2, r7, x3, NUM_SKIP_BYTES + 6
        MOVZXLO x2, x0
        MOVZXHI x5, x1
        MOVZXLO x3, x1
        shr     x0, 8
        shr     x1, 16
        CRC_XOR r7, x2, NUM_SKIP_BYTES + 5
        CRC_XOR r6, x3, NUM_SKIP_BYTES + 3
        CRC_XOR r7, x0, NUM_SKIP_BYTES + 4
        CRC_XOR r6, x5, NUM_SKIP_BYTES + 2
        MOVZXLO x2, x1
        shr     x1, 8
        CRC_XOR r7, x2, NUM_SKIP_BYTES + 1
        CRC_MOV r0, x1, NUM_SKIP_BYTES + 0
        xor     r0, r6
        xor     r0, r7

endif ; NUM_WORDS > 1
        add     rD, NUM_WORDS * 4
        jnc     @B

        sub     rN, src_rN_offset
        add     rD, rN
        XOR_NEXT
        add     rN, NUM_BYTES_LIMIT - 1
        sub     rN, rD

crc_end:
        test    rN, rN
        jz      func_end
@@:
        CRC1b
        jnz      @B
func_end:
        MY_POP_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11
MY_ENDP



else
; ==================================================================
; x86 (32-bit)

rD      equ  r7
rN      equ  r1
rT      equ  r5

xA      equ  x6
xA_R    equ  r6

ifdef x64
    num_VAR     equ  r8
else

crc_OFFS  equ  (REG_SIZE * 5)

if (IS_CDECL gt 0) or (IS_LINUX gt 0)
    ; cdecl or (GNU fastcall) stack:
    ;   (UInt32 *) table
    ;   size_t     size
    ;   void *     data
    ;   (UInt64)   crc
    ;   ret-ip <-(r4)
    data_OFFS   equ  (8 + crc_OFFS)
    size_OFFS   equ  (REG_SIZE + data_OFFS)
    table_OFFS  equ  (REG_SIZE + size_OFFS)
    num_VAR     equ  [r4 + size_OFFS]
    table_VAR   equ  [r4 + table_OFFS]
else
    ; Windows fastcall:
    ;   r1 = data, r2 = size
    ; stack:
    ;   (UInt32 *) table
    ;   (UInt64)   crc
    ;   ret-ip <-(r4)
    table_OFFS  equ  (8 + crc_OFFS)
    table_VAR   equ  [r4 + table_OFFS]
    num_VAR     equ  table_VAR
endif
endif ; x64

SRCDAT4         equ     DWORD PTR [rN + rD * 1]

CRC_1 macro op:req, dest:req, src:req, t:req, word_index:req
        op      dest, DWORD PTR [rT + @CatStr(src, _R) * 8 + 0800h * (t) + (word_index) * 4]
endm

CRC macro op0:req, op1:req, dest0:req, dest1:req, src:req, t:req
        CRC_1   op0, dest0, src, t, 0
        CRC_1   op1, dest1, src, t, 1
endm

CRC_XOR macro dest0:req, dest1:req, src:req, t:req
        CRC xor, xor, dest0, dest1, src, t
endm


CRC1b macro
        movzx   xA, BYTE PTR [rD]
        inc     rD
        MOVZXLO x3, x0
        xor     xA, x3
        shrd    x0, x2, 8
        shr     x2, 8
        CRC_XOR x0, x2, xA, 0
        dec     rN
endm


MY_PROLOG_BASE macro
        MY_PUSH_4_REGS
ifdef x64
        mov     r0, REG_ABI_PARAM_0     ; r0 <- r1 / r7
        mov     rT, REG_ABI_PARAM_3     ; r5 <- r9 / r1
        mov     rN, REG_ABI_PARAM_2     ; r1 <- r8 / r2
        mov     rD, REG_ABI_PARAM_1     ; r7 <- r2 / r6
        mov     r2, r0
        shr     r2, 32
        mov     x0, x0
else
    if (IS_CDECL gt 0) or (IS_LINUX gt 0)
        proc_numParams = proc_numParams + 2 ; for ABI_LINUX
        mov     rN, [r4 + size_OFFS]
        mov     rD, [r4 + data_OFFS]
    else
        mov     rD, REG_ABI_PARAM_0     ; r7 <- r1 : (data)
        mov     rN, REG_ABI_PARAM_1     ; r1 <- r2 : (size)
    endif
        mov     x0, [r4 + crc_OFFS]
        mov     x2, [r4 + crc_OFFS + 4]
        mov     rT, table_VAR
endif
endm


MY_EPILOG_BASE macro crc_end:req, func_end:req
crc_end:
        test    rN, rN
        jz      func_end
@@:
        CRC1b
        jnz      @B
func_end:
ifdef x64
        shl     r2, 32
        xor     r0, r2
endif
        MY_POP_4_REGS
endm


; ALIGN_MASK is 3 or 7 bytes alignment:
ALIGN_MASK  equ     (7 - (NUM_WORDS and 1) * 4)

if (NUM_WORDS eq 1)

NUM_BYTES_LIMIT_T4 equ (NUM_WORDS * 4 + 4)

MY_PROC @CatStr(XzCrc64UpdateT, %(NUM_WORDS * 4)), 5
        MY_PROLOG_BASE

        cmp     rN, NUM_BYTES_LIMIT_T4 + ALIGN_MASK
        jb      crc_end_4
@@:
        test    rD, ALIGN_MASK
        jz      @F
        CRC1b
        jmp     @B
@@:
        xor     x0, [rD]
        lea     rN, [rD + rN * 1 - (NUM_BYTES_LIMIT_T4 - 1)]
        sub     rD, rN
        add     rN, 4

        MOVZXLO xA, x0
align 16
@@:
        mov     x3, SRCDAT4
        xor     x3, x2
        shr     x0, 8
        CRC xor, mov, x3, x2, xA, 3
        MOVZXLO xA, x0
        shr     x0, 8
        ; MOVZXHI  xA, x0
        ; shr     x0, 16
        CRC_XOR x3, x2, xA, 2

        MOVZXLO xA, x0
        shr     x0, 8
        CRC_XOR x3, x2, xA, 1
        CRC_XOR x3, x2, x0, 0
        MOVZXLO xA, x3
        mov     x0, x3

        add     rD, 4
        jnc     @B

        sub     rN, 4
        add     rD, rN
        xor     x0, [rD]
        add     rN, NUM_BYTES_LIMIT_T4 - 1
        sub     rN, rD
        MY_EPILOG_BASE crc_end_4, func_end_4
MY_ENDP

else ; NUM_WORDS > 1

SHR_X macro x, imm
        shr x, imm
endm


ITER_1 macro v0, v1, a, off
        MOVZXLO xA, a
        SHR_X   a, 8
        CRC_XOR v0, v1, xA, off
endm


ITER_4 macro v0, v1, a, off
if 0 eq 0
        ITER_1  v0, v1, a, off + 3
        ITER_1  v0, v1, a, off + 2
        ITER_1  v0, v1, a, off + 1
        CRC_XOR v0, v1, a, off
elseif 0 eq 0
        MOVZXLO xA, a
        CRC_XOR v0, v1, xA, off + 3
        mov     xA, a
        ror     a, 16   ; 32-bit ror
        shr     xA, 24
        CRC_XOR v0, v1, xA, off
        MOVZXLO xA, a
        SHR_X   a, 24
        CRC_XOR v0, v1, xA, off + 1
        CRC_XOR v0, v1, a, off + 2
else
        ; MOVZXHI provides smaller code, but MOVZX_HI_BYTE is not fast instruction
        MOVZXLO xA, a
        CRC_XOR v0, v1, xA, off + 3
        MOVZXHI xA, a
        SHR_X   a, 16
        CRC_XOR v0, v1, xA, off + 2
        MOVZXLO xA, a
        SHR_X   a, 8
        CRC_XOR v0, v1, xA, off + 1
        CRC_XOR v0, v1, a, off
endif
endm



ITER_1_PAIR macro v0, v1, a0, a1, off
        ITER_1 v0, v1, a0, off + 4
        ITER_1 v0, v1, a1, off
endm

src_rD_offset equ 8
STEP_SIZE       equ     (NUM_WORDS * 4)

ITER_12_NEXT macro op, index, v0, v1
        op     v0, DWORD PTR [rD + (index + 1) * STEP_SIZE     - src_rD_offset]
        op     v1, DWORD PTR [rD + (index + 1) * STEP_SIZE + 4 - src_rD_offset]
endm

ITER_12 macro index, a0, a1, v0, v1
  
  if NUM_SKIP_BYTES  eq 0
        ITER_12_NEXT mov, index, v0, v1
  else
    k = 0
    while k lt NUM_SKIP_BYTES
        movzx   xA, BYTE PTR [rD + (index) * STEP_SIZE + k + 8 - src_rD_offset]
      if k eq 0
        CRC mov, mov,   v0, v1, xA, NUM_SKIP_BYTES - 1 - k
      else
        CRC_XOR         v0, v1, xA, NUM_SKIP_BYTES - 1 - k
      endif
      k = k + 1
    endm
        ITER_12_NEXT xor, index, v0, v1
  endif

if 0 eq 0
        ITER_4  v0, v1, a0, NUM_SKIP_BYTES + 4
        ITER_4  v0, v1, a1, NUM_SKIP_BYTES
else ; interleave version is faster/slower for different processors
        ITER_1_PAIR v0, v1, a0, a1, NUM_SKIP_BYTES + 3
        ITER_1_PAIR v0, v1, a0, a1, NUM_SKIP_BYTES + 2
        ITER_1_PAIR v0, v1, a0, a1, NUM_SKIP_BYTES + 1
        CRC_XOR     v0, v1, a0,     NUM_SKIP_BYTES + 4
        CRC_XOR     v0, v1, a1,     NUM_SKIP_BYTES
endif
endm

; we use (UNROLL_CNT > 1) to reduce read ports pressure (num_VAR reads)
UNROLL_CNT      equ     (2 * 1)
NUM_BYTES_LIMIT equ     (STEP_SIZE * UNROLL_CNT + 8)

MY_PROC @CatStr(XzCrc64UpdateT, %(NUM_WORDS * 4)), 5
        MY_PROLOG_BASE

        cmp     rN, NUM_BYTES_LIMIT + ALIGN_MASK
        jb      crc_end_12
@@:
        test    rD, ALIGN_MASK
        jz      @F
        CRC1b
        jmp     @B
@@:
        xor     x0, [rD]
        xor     x2, [rD + 4]
        add     rD, src_rD_offset
        lea     rN, [rD + rN * 1 - (NUM_BYTES_LIMIT - 1)]
        mov     num_VAR, rN

align 16
@@:
    i = 0
    rept UNROLL_CNT
      if (i and 1) eq 0
        ITER_12     i, x0, x2,  x1, x3
      else
        ITER_12     i, x1, x3,  x0, x2
      endif
      i = i + 1
    endm

    if (UNROLL_CNT and 1)
        mov     x0, x1
        mov     x2, x3
    endif
        add     rD, STEP_SIZE * UNROLL_CNT
        cmp     rD, num_VAR
        jb      @B

        mov     rN, num_VAR
        add     rN, NUM_BYTES_LIMIT - 1
        sub     rN, rD
        sub     rD, src_rD_offset
        xor     x0, [rD]
        xor     x2, [rD + 4]

        MY_EPILOG_BASE crc_end_12, func_end_12
MY_ENDP

endif ; (NUM_WORDS > 1)
endif ; ! x64
end
