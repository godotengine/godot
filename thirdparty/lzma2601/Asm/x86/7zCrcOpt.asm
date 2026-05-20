; 7zCrcOpt.asm -- CRC32 calculation : optimized version
; 2023-12-08 : Igor Pavlov : Public domain

include 7zAsm.asm

MY_ASM_START

NUM_WORDS       equ     3
UNROLL_CNT      equ     2

if (NUM_WORDS lt 1) or (NUM_WORDS gt 64)
.err <NUM_WORDS_IS_INCORRECT>
endif
if (UNROLL_CNT lt 1)
.err <UNROLL_CNT_IS_INCORRECT>
endif

rD      equ  r2
rD_x    equ  x2
rN      equ  r7
rT      equ  r5

ifndef x64
    if (IS_CDECL gt 0)
        crc_OFFS    equ (REG_SIZE * 5)
        data_OFFS   equ (REG_SIZE + crc_OFFS)
        size_OFFS   equ (REG_SIZE + data_OFFS)
    else
        size_OFFS   equ (REG_SIZE * 5)
    endif
        table_OFFS  equ (REG_SIZE + size_OFFS)
endif

; rN + rD is same speed as rD, but we reduce one instruction in loop
SRCDAT_1        equ     rN + rD * 1 + 1 *
SRCDAT_4        equ     rN + rD * 1 + 4 *

CRC macro op:req, dest:req, src:req, t:req
        op      dest, dword ptr [rT + @CatStr(src, _R) * 4 + 0400h * (t)]
endm

CRC_XOR macro dest:req, src:req, t:req
        CRC     xor, dest, src, t
endm

CRC_MOV macro dest:req, src:req, t:req
        CRC     mov, dest, src, t
endm

MOVZXLO macro dest:req, src:req
        movzx   dest, @CatStr(src, _L)
endm

MOVZXHI macro dest:req, src:req
        movzx   dest, @CatStr(src, _H)
endm

; movzx x0, x0_L - is slow in some cpus (ivb), if same register for src and dest
; movzx x3, x0_L sometimes is 0   cycles latency (not always)
; movzx x3, x0_L sometimes is 0.5 cycles latency
; movzx x3, x0_H is 2 cycles latency in some cpus

CRC1b macro
        movzx   x6, byte ptr [rD]
        MOVZXLO x3, x0
        inc     rD
        shr     x0, 8
        xor     x6, x3
        CRC_XOR x0, x6, 0
        dec     rN
endm

LOAD_1 macro dest:req, t:req, iter:req, index:req
        movzx   dest, byte ptr [SRCDAT_1 (4 * (NUM_WORDS - 1 - t + iter * NUM_WORDS) + index)]
endm

LOAD_2 macro dest:req, t:req, iter:req, index:req
        movzx   dest, word ptr [SRCDAT_1 (4 * (NUM_WORDS - 1 - t + iter * NUM_WORDS) + index)]
endm

CRC_QUAD macro nn, t:req, iter:req
ifdef x64
        ; paired memory loads give 1-3% speed gain, but it uses more registers
        LOAD_2  x3, t, iter, 0
        LOAD_2  x9, t, iter, 2
        MOVZXLO x6, x3
        shr     x3, 8
        CRC_XOR nn, x6, t * 4 + 3
        MOVZXLO x6, x9
        shr     x9, 8
        CRC_XOR nn, x3, t * 4 + 2
        CRC_XOR nn, x6, t * 4 + 1
        CRC_XOR nn, x9, t * 4 + 0
elseif 0
        LOAD_2  x3, t, iter, 0
        MOVZXLO x6, x3
        shr     x3, 8
        CRC_XOR nn, x6, t * 4 + 3
        CRC_XOR nn, x3, t * 4 + 2
        LOAD_2  x3, t, iter, 2
        MOVZXLO x6, x3
        shr     x3, 8
        CRC_XOR nn, x6, t * 4 + 1
        CRC_XOR nn, x3, t * 4 + 0
elseif 0
        LOAD_1  x3, t, iter, 0
        LOAD_1  x6, t, iter, 1
        CRC_XOR nn, x3, t * 4 + 3
        CRC_XOR nn, x6, t * 4 + 2
        LOAD_1  x3, t, iter, 2
        LOAD_1  x6, t, iter, 3
        CRC_XOR nn, x3, t * 4 + 1
        CRC_XOR nn, x6, t * 4 + 0
else
        ; 32-bit load is better if there is only one read port (core2)
        ; but that code can be slower if there are 2 read ports (snb)
        mov     x3, dword ptr [SRCDAT_1 (4 * (NUM_WORDS - 1 - t + iter *  NUM_WORDS) + 0)]
        MOVZXLO x6, x3
        CRC_XOR nn, x6, t * 4 + 3
        MOVZXHI x6, x3
        shr     x3, 16
        CRC_XOR nn, x6, t * 4 + 2
        MOVZXLO x6, x3
        shr     x3, 8
        CRC_XOR nn, x6, t * 4 + 1
        CRC_XOR nn, x3, t * 4 + 0
endif
endm


LAST    equ     (4 * (NUM_WORDS - 1))

CRC_ITER macro qq, nn, iter
        mov     nn, [SRCDAT_4 (NUM_WORDS * (1 + iter))]

    i = 0
    rept NUM_WORDS - 1
        CRC_QUAD nn, i, iter
        i = i + 1
    endm

        MOVZXLO x6, qq
        mov     x3, qq
        shr     x3, 24
        CRC_XOR nn, x6, LAST + 3
        CRC_XOR nn, x3, LAST + 0
        ror     qq, 16
        MOVZXLO x6, qq
        shr     qq, 24
        CRC_XOR nn, x6, LAST + 1
if ((UNROLL_CNT and 1) eq 1) and (iter eq (UNROLL_CNT - 1))
        CRC_MOV qq, qq, LAST + 2
        xor     qq, nn
else
        CRC_XOR nn, qq, LAST + 2
endif
endm


; + 4 for prefetching next 4-bytes after current iteration
NUM_BYTES_LIMIT equ     (NUM_WORDS * 4 * UNROLL_CNT + 4)
ALIGN_MASK      equ     3


; MY_PROC @CatStr(CrcUpdateT, 12), 4
MY_PROC @CatStr(CrcUpdateT, %(NUM_WORDS * 4)), 4
        MY_PUSH_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11
    ifdef x64
        mov     x0, REG_ABI_PARAM_0_x   ; x0 = x1(win) / x7(linux)
        mov     rT, REG_ABI_PARAM_3     ; r5 = r9(win) / x1(linux)
        mov     rN, REG_ABI_PARAM_2     ; r7 = r8(win) / r2(linux)
        ; mov     rD, REG_ABI_PARAM_1     ; r2 = r2(win)
      if  (IS_LINUX gt 0)
        mov     rD, REG_ABI_PARAM_1     ; r2 = r6
      endif
    else
      if  (IS_CDECL gt 0)
        mov     x0, [r4 + crc_OFFS]
        mov     rD, [r4 + data_OFFS]
      else
        mov     x0, REG_ABI_PARAM_0_x
      endif
        mov     rN, [r4 + size_OFFS]
        mov     rT, [r4 + table_OFFS]
    endif
    
        cmp     rN, NUM_BYTES_LIMIT + ALIGN_MASK
        jb      crc_end
@@:
        test    rD_x, ALIGN_MASK    ; test    rD, ALIGN_MASK
        jz      @F
        CRC1b
        jmp     @B
@@:
        xor     x0, dword ptr [rD]
        lea     rN, [rD + rN * 1 - (NUM_BYTES_LIMIT - 1)]
        sub     rD, rN

align 16
@@:
unr_index = 0
while unr_index lt UNROLL_CNT
    if (unr_index and 1) eq 0
        CRC_ITER x0, x1, unr_index
    else
        CRC_ITER x1, x0, unr_index
    endif
        unr_index = unr_index + 1
endm

        add     rD, NUM_WORDS * 4 * UNROLL_CNT
        jnc     @B

if 0
        ; byte verson
        add     rD, rN
        xor     x0, dword ptr [rD]
        add     rN, NUM_BYTES_LIMIT - 1
else
        ; 4-byte version
        add     rN, 4 * NUM_WORDS * UNROLL_CNT
        sub     rD, 4 * NUM_WORDS * UNROLL_CNT
@@:
        MOVZXLO x3, x0
        MOVZXHI x1, x0
        shr     x0, 16
        MOVZXLO x6, x0
        shr     x0, 8
        CRC_MOV x0, x0, 0
        CRC_XOR x0, x3, 3
        CRC_XOR x0, x1, 2
        CRC_XOR x0, x6, 1

        add     rD, 4
if (NUM_WORDS * UNROLL_CNT) ne 1
        jc      @F
        xor     x0, [SRCDAT_4 0]
        jmp     @B
@@:
endif
        add     rD, rN
        add     rN, 4 - 1
        
endif
        
        sub     rN, rD
crc_end:
        test    rN, rN
        jz      func_end
@@:
        CRC1b
        jnz     @B

func_end:
        MY_POP_PRESERVED_ABI_REGS_UP_TO_INCLUDING_R11
MY_ENDP

end
