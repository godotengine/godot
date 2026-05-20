; Sha256Opt.asm -- SHA-256 optimized code for SHA-256 x86 hardware instructions
; 2024-06-16 : Igor Pavlov : Public domain

include 7zAsm.asm

MY_ASM_START

; .data
; public K

; we can use external SHA256_K_ARRAY defined in Sha256.c
; but we must guarantee that SHA256_K_ARRAY is aligned for 16-bytes

COMMENT @
ifdef x64
K_CONST equ SHA256_K_ARRAY
else
K_CONST equ _SHA256_K_ARRAY
endif
EXTRN   K_CONST:xmmword
@

CONST   SEGMENT READONLY

align 16
Reverse_Endian_Mask db 3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12

; COMMENT @
align 16
K_CONST \
DD 0428a2f98H, 071374491H, 0b5c0fbcfH, 0e9b5dba5H
DD 03956c25bH, 059f111f1H, 0923f82a4H, 0ab1c5ed5H
DD 0d807aa98H, 012835b01H, 0243185beH, 0550c7dc3H
DD 072be5d74H, 080deb1feH, 09bdc06a7H, 0c19bf174H
DD 0e49b69c1H, 0efbe4786H, 00fc19dc6H, 0240ca1ccH
DD 02de92c6fH, 04a7484aaH, 05cb0a9dcH, 076f988daH
DD 0983e5152H, 0a831c66dH, 0b00327c8H, 0bf597fc7H
DD 0c6e00bf3H, 0d5a79147H, 006ca6351H, 014292967H
DD 027b70a85H, 02e1b2138H, 04d2c6dfcH, 053380d13H
DD 0650a7354H, 0766a0abbH, 081c2c92eH, 092722c85H
DD 0a2bfe8a1H, 0a81a664bH, 0c24b8b70H, 0c76c51a3H
DD 0d192e819H, 0d6990624H, 0f40e3585H, 0106aa070H
DD 019a4c116H, 01e376c08H, 02748774cH, 034b0bcb5H
DD 0391c0cb3H, 04ed8aa4aH, 05b9cca4fH, 0682e6ff3H
DD 0748f82eeH, 078a5636fH, 084c87814H, 08cc70208H
DD 090befffaH, 0a4506cebH, 0bef9a3f7H, 0c67178f2H
; @

CONST   ENDS

; _TEXT$SHA256OPT SEGMENT 'CODE'

ifndef x64
    .686
    .xmm
endif
        
; jwasm-based assemblers for linux and linker from new versions of binutils
; can generate incorrect code for load [ARRAY + offset] instructions.
; 22.00: we load K_CONST offset to (rTable) register to avoid jwasm+binutils problem 
        rTable  equ r0
        ; rTable  equ K_CONST
        
ifdef x64
        rNum    equ REG_ABI_PARAM_2
    if (IS_LINUX eq 0)
        LOCAL_SIZE equ (16 * 2)
    endif
else
        rNum    equ r3
        LOCAL_SIZE equ (16 * 1)
endif

rState equ REG_ABI_PARAM_0
rData  equ REG_ABI_PARAM_1






MY_SHA_INSTR macro cmd, a1, a2
        db 0fH, 038H, cmd, (0c0H + a1 * 8 + a2)
endm

cmd_sha256rnds2 equ 0cbH
cmd_sha256msg1  equ 0ccH
cmd_sha256msg2  equ 0cdH

MY_sha256rnds2 macro a1, a2
        MY_SHA_INSTR  cmd_sha256rnds2, a1, a2
endm

MY_sha256msg1 macro a1, a2
        MY_SHA_INSTR  cmd_sha256msg1, a1, a2
endm

MY_sha256msg2 macro a1, a2
        MY_SHA_INSTR  cmd_sha256msg2, a1, a2
endm

MY_PROLOG macro
    ifdef x64
      if (IS_LINUX eq 0)
        movdqa  [r4 + 8], xmm6
        movdqa  [r4 + 8 + 16], xmm7
        sub     r4, LOCAL_SIZE + 8
        movdqa  [r4     ], xmm8
        movdqa  [r4 + 16], xmm9
      endif
    else ; x86
        push    r3
        push    r5
        mov     r5, r4
        NUM_PUSH_REGS   equ 2
        PARAM_OFFSET    equ (REG_SIZE * (1 + NUM_PUSH_REGS))
      if (IS_CDECL gt 0)
        mov     rState, [r4 + PARAM_OFFSET]
        mov     rData,  [r4 + PARAM_OFFSET + REG_SIZE * 1]
        mov     rNum,   [r4 + PARAM_OFFSET + REG_SIZE * 2]
      else ; fastcall
        mov     rNum,   [r4 + PARAM_OFFSET]
      endif
        and     r4, -16
        sub     r4, LOCAL_SIZE
    endif
endm

MY_EPILOG macro
    ifdef x64
      if (IS_LINUX eq 0)
        movdqa  xmm8, [r4]
        movdqa  xmm9, [r4 + 16]
        add     r4, LOCAL_SIZE + 8
        movdqa  xmm6, [r4 + 8]
        movdqa  xmm7, [r4 + 8 + 16]
      endif
    else ; x86
        mov     r4, r5
        pop     r5
        pop     r3
    endif
    MY_ENDP
endm


msg        equ xmm0
tmp        equ xmm0
state0_N   equ 2
state1_N   equ 3
w_regs     equ 4


state1_save equ xmm1
state0  equ @CatStr(xmm, %state0_N)
state1  equ @CatStr(xmm, %state1_N)


ifdef x64
        state0_save  equ  xmm8
        mask2        equ  xmm9
else
        state0_save  equ  [r4]
        mask2        equ  xmm0
endif

LOAD_MASK macro
        movdqa  mask2, XMMWORD PTR Reverse_Endian_Mask
endm

LOAD_W macro k:req
        movdqu  @CatStr(xmm, %(w_regs + k)), [rData + (16 * (k))]
        pshufb  @CatStr(xmm, %(w_regs + k)), mask2
endm


; pre1 <= 4 && pre2 >= 1 && pre1 > pre2 && (pre1 - pre2) <= 1
pre1 equ 3
pre2 equ 2
   


RND4 macro k
        movdqa  msg, xmmword ptr [rTable + (k) * 16]
        paddd   msg, @CatStr(xmm, %(w_regs + ((k + 0) mod 4)))
        MY_sha256rnds2 state0_N, state1_N
        pshufd   msg, msg, 0eH
        
    if (k GE (4 - pre1)) AND (k LT (16 - pre1))
        ; w4[0] = msg1(w4[-4], w4[-3])
        MY_sha256msg1 (w_regs + ((k + pre1) mod 4)), (w_regs + ((k + pre1 - 3) mod 4))
    endif
        
        MY_sha256rnds2 state1_N, state0_N

    if (k GE (4 - pre2)) AND (k LT (16 - pre2))
        movdqa  tmp, @CatStr(xmm, %(w_regs + ((k + pre2 - 1) mod 4)))
        palignr tmp, @CatStr(xmm, %(w_regs + ((k + pre2 - 2) mod 4))), 4
        paddd   @CatStr(xmm, %(w_regs + ((k + pre2) mod 4))), tmp
        ; w4[0] = msg2(w4[0], w4[-1])
        MY_sha256msg2 %(w_regs + ((k + pre2) mod 4)), %(w_regs + ((k + pre2 - 1) mod 4))
    endif
endm





REVERSE_STATE macro
                               ; state0 ; dcba
                               ; state1 ; hgfe
        pshufd      tmp, state0, 01bH   ; abcd
        pshufd   state0, state1, 01bH   ; efgh
        movdqa   state1, state0         ; efgh
        punpcklqdq  state0, tmp         ; cdgh
        punpckhqdq  state1, tmp         ; abef
endm


MY_PROC Sha256_UpdateBlocks_HW, 3
    MY_PROLOG

        lea     rTable, [K_CONST]

        cmp     rNum, 0
        je      end_c

        movdqu   state0, [rState]       ; dcba
        movdqu   state1, [rState + 16]  ; hgfe

        REVERSE_STATE
       
        ifdef x64
        LOAD_MASK
        endif

    align 16
    nextBlock:
        movdqa  state0_save, state0
        movdqa  state1_save, state1
        
        ifndef x64
        LOAD_MASK
        endif
        
        LOAD_W 0
        LOAD_W 1
        LOAD_W 2
        LOAD_W 3

        
        k = 0
        rept 16
          RND4 k
          k = k + 1
        endm

        paddd   state0, state0_save
        paddd   state1, state1_save

        add     rData, 64
        sub     rNum, 1
        jnz     nextBlock
        
        REVERSE_STATE

        movdqu  [rState], state0
        movdqu  [rState + 16], state1
       
  end_c:
MY_EPILOG

; _TEXT$SHA256OPT ENDS

end
