; AesOpt.asm -- AES optimized code for x86 AES hardware instructions
; 2021-12-25 : Igor Pavlov : Public domain

include 7zAsm.asm

ifdef __ASMC__
  use_vaes_256 equ 1
else
ifdef ymm0
  use_vaes_256 equ 1
endif
endif


ifdef use_vaes_256
  ECHO "++ VAES 256"
else
  ECHO "-- NO VAES 256"
endif

ifdef x64
  ECHO "x86-64"
else
  ECHO "x86"
if (IS_CDECL gt 0)
  ECHO "ABI : CDECL"
else
  ECHO "ABI : no CDECL : FASTCALL"
endif
endif

if (IS_LINUX gt 0)
  ECHO "ABI : LINUX"
else
  ECHO "ABI : WINDOWS"
endif

MY_ASM_START

ifndef x64
    .686
    .xmm
endif


; MY_ALIGN EQU ALIGN(64)
MY_ALIGN EQU

SEG_ALIGN EQU MY_ALIGN

MY_SEG_PROC macro name:req, numParams:req
    ; seg_name equ @CatStr(_TEXT$, name)
    ; seg_name SEGMENT SEG_ALIGN 'CODE'
    MY_PROC name, numParams
endm

MY_SEG_ENDP macro
    ; seg_name ENDS
endm


NUM_AES_KEYS_MAX equ 15

; the number of push operators in function PROLOG
if (IS_LINUX eq 0) or (IS_X64 eq 0)
num_regs_push   equ 2
stack_param_offset equ (REG_SIZE * (1 + num_regs_push))
endif

ifdef x64
    num_param   equ REG_ABI_PARAM_2
else
  if (IS_CDECL gt 0)
    ;   size_t     size
    ;   void *     data
    ;   UInt32 *   aes
    ;   ret-ip <- (r4)
    aes_OFFS    equ (stack_param_offset)
    data_OFFS   equ (REG_SIZE + aes_OFFS)
    size_OFFS   equ (REG_SIZE + data_OFFS)
    num_param   equ [r4 + size_OFFS]
  else
    num_param   equ [r4 + stack_param_offset]
  endif
endif

keys    equ  REG_PARAM_0  ; r1
rD      equ  REG_PARAM_1  ; r2
rN      equ  r0

koffs_x equ  x7
koffs_r equ  r7

ksize_x equ  x6
ksize_r equ  r6

keys2   equ  r3

state   equ  xmm0
key     equ  xmm0
key_ymm equ  ymm0
key_ymm_n equ   0

ifdef x64
        ways = 11
else
        ways = 4
endif

ways_start_reg equ 1

iv      equ     @CatStr(xmm, %(ways_start_reg + ways))
iv_ymm  equ     @CatStr(ymm, %(ways_start_reg + ways))


WOP macro op, op2
    i = 0
    rept ways
        op      @CatStr(xmm, %(ways_start_reg + i)), op2
        i = i + 1
    endm
endm


ifndef ABI_LINUX
ifdef x64

; we use 32 bytes of home space in stack in WIN64-x64
NUM_HOME_MM_REGS   equ (32 / 16)
; we preserve xmm registers starting from xmm6 in WIN64-x64
MM_START_SAVE_REG  equ 6

SAVE_XMM macro num_used_mm_regs:req
  num_save_mm_regs = num_used_mm_regs - MM_START_SAVE_REG
  if num_save_mm_regs GT 0
    num_save_mm_regs2 = num_save_mm_regs - NUM_HOME_MM_REGS
    ; RSP is (16*x + 8) after entering the function in WIN64-x64
    stack_offset = 16 * num_save_mm_regs2 + (stack_param_offset mod 16)
  
    i = 0
    rept num_save_mm_regs
      
      if i eq NUM_HOME_MM_REGS
        sub  r4, stack_offset
      endif
      
      if i lt NUM_HOME_MM_REGS
        movdqa  [r4 + stack_param_offset + i * 16], @CatStr(xmm, %(MM_START_SAVE_REG + i))
      else
        movdqa  [r4 + (i - NUM_HOME_MM_REGS) * 16], @CatStr(xmm, %(MM_START_SAVE_REG + i))
      endif
      
      i = i + 1
    endm
  endif
endm

RESTORE_XMM macro num_used_mm_regs:req
  if num_save_mm_regs GT 0
    i = 0
    if num_save_mm_regs2 GT 0
      rept num_save_mm_regs2
        movdqa  @CatStr(xmm, %(MM_START_SAVE_REG + NUM_HOME_MM_REGS + i)), [r4 + i * 16]
        i = i + 1
      endm
        add     r4, stack_offset
    endif

    num_low_regs = num_save_mm_regs - i
    i = 0
      rept num_low_regs
        movdqa  @CatStr(xmm, %(MM_START_SAVE_REG + i)), [r4 + stack_param_offset + i * 16]
        i = i + 1
      endm
  endif
endm

endif ; x64
endif ; ABI_LINUX


MY_PROLOG macro num_used_mm_regs:req
        ; num_regs_push: must be equal to the number of push operators
        ; push    r3
        ; push    r5
    if (IS_LINUX eq 0) or (IS_X64 eq 0)
        push    r6
        push    r7
    endif

        mov     rN, num_param  ; don't move it; num_param can use stack pointer (r4)

    if (IS_X64 eq 0)
      if (IS_CDECL gt 0)
        mov     rD,   [r4 + data_OFFS]
        mov     keys, [r4 + aes_OFFS]
      endif
    elseif (IS_LINUX gt 0)
        MY_ABI_LINUX_TO_WIN_2
    endif


    ifndef ABI_LINUX
    ifdef x64
        SAVE_XMM num_used_mm_regs
    endif
    endif
   
        mov     ksize_x, [keys + 16]
        shl     ksize_x, 5
endm


MY_EPILOG macro
    ifndef ABI_LINUX
    ifdef x64
        RESTORE_XMM num_save_mm_regs
    endif
    endif
    
    if (IS_LINUX eq 0) or (IS_X64 eq 0)
        pop     r7
        pop     r6
    endif
        ; pop     r5
        ; pop     r3
    MY_ENDP
endm


OP_KEY macro op:req, offs:req
        op      state, [keys + offs]
endm

 
WOP_KEY macro op:req, offs:req
        movdqa  key, [keys + offs]
        WOP     op, key
endm


; ---------- AES-CBC Decode ----------


XOR_WITH_DATA macro reg, _ppp_
        pxor    reg, [rD + i * 16]
endm

WRITE_TO_DATA macro reg, _ppp_
        movdqa  [rD + i * 16], reg
endm


; state0    equ  @CatStr(xmm, %(ways_start_reg))

key0            equ  @CatStr(xmm, %(ways_start_reg + ways + 1))
key0_ymm        equ  @CatStr(ymm, %(ways_start_reg + ways + 1))

key_last        equ  @CatStr(xmm, %(ways_start_reg + ways + 2))
key_last_ymm    equ  @CatStr(ymm, %(ways_start_reg + ways + 2))
key_last_ymm_n  equ                (ways_start_reg + ways + 2)

NUM_CBC_REGS    equ  (ways_start_reg + ways + 3)


MY_SEG_PROC AesCbc_Decode_HW, 3

    AesCbc_Decode_HW_start::
        MY_PROLOG NUM_CBC_REGS
   
    AesCbc_Decode_HW_start_2::
        movdqa  iv, [keys]
        add     keys, 32

        movdqa  key0, [keys + 1 * ksize_r]
        movdqa  key_last, [keys]
        sub     ksize_x, 16

        jmp     check2
    align 16
    nextBlocks2:
        WOP     movdqa, [rD + i * 16]
        mov     koffs_x, ksize_x
        ; WOP_KEY pxor, ksize_r + 16
        WOP     pxor, key0
    ; align 16
    @@:
        WOP_KEY aesdec, 1 * koffs_r
        sub     koffs_r, 16
        jnz     @B
        ; WOP_KEY aesdeclast, 0
        WOP     aesdeclast, key_last
        
        pxor    @CatStr(xmm, %(ways_start_reg)), iv
    i = 1
    rept ways - 1
        pxor    @CatStr(xmm, %(ways_start_reg + i)), [rD + i * 16 - 16]
        i = i + 1
    endm
        movdqa  iv, [rD + ways * 16 - 16]
        WOP     WRITE_TO_DATA

        add     rD, ways * 16
    AesCbc_Decode_HW_start_3::
    check2:
        sub     rN, ways
        jnc     nextBlocks2
        add     rN, ways

        sub     ksize_x, 16

        jmp     check
    nextBlock:
        movdqa  state, [rD]
        mov     koffs_x, ksize_x
        ; OP_KEY  pxor, 1 * ksize_r + 32
        pxor    state, key0
        ; movdqa  state0, [rD]
        ; movdqa  state, key0
        ; pxor    state, state0
    @@:
        OP_KEY  aesdec, 1 * koffs_r + 16
        OP_KEY  aesdec, 1 * koffs_r
        sub     koffs_r, 32
        jnz     @B
        OP_KEY  aesdec, 16
        ; OP_KEY  aesdeclast, 0
        aesdeclast state, key_last
        
        pxor    state, iv
        movdqa  iv, [rD]
        ; movdqa  iv, state0
        movdqa  [rD], state
        
        add     rD, 16
    check:
        sub     rN, 1
        jnc     nextBlock

        movdqa  [keys - 32], iv
MY_EPILOG




; ---------- AVX ----------


AVX__WOP_n macro op
    i = 0
    rept ways
        op      (ways_start_reg + i)
        i = i + 1
    endm
endm

AVX__WOP macro op
    i = 0
    rept ways
        op      @CatStr(ymm, %(ways_start_reg + i))
        i = i + 1
    endm
endm


AVX__WOP_KEY macro op:req, offs:req
        vmovdqa  key_ymm, ymmword ptr [keys2 + offs]
        AVX__WOP_n op
endm


AVX__CBC_START macro reg
        ; vpxor   reg, key_ymm, ymmword ptr [rD + 32 * i]
        vpxor   reg, key0_ymm, ymmword ptr [rD + 32 * i]
endm

AVX__CBC_END macro reg
    if i eq 0
        vpxor   reg, reg, iv_ymm
    else
        vpxor   reg, reg, ymmword ptr [rD + i * 32 - 16]
    endif
endm


AVX__WRITE_TO_DATA macro reg
        vmovdqu ymmword ptr [rD + 32 * i], reg
endm

AVX__XOR_WITH_DATA macro reg
        vpxor   reg, reg, ymmword ptr [rD + 32 * i]
endm

AVX__CTR_START macro reg
        vpaddq  iv_ymm, iv_ymm, one_ymm
        ; vpxor   reg, iv_ymm, key_ymm
        vpxor   reg, iv_ymm, key0_ymm
endm


MY_VAES_INSTR_2 macro cmd, dest, a1, a2
  db 0c4H
  db 2 + 040H + 020h * (1 - (a2) / 8) + 080h * (1 - (dest) / 8)
  db 5 + 8 * ((not (a1)) and 15)
  db cmd
  db 0c0H + 8 * ((dest) and 7) + ((a2) and 7)
endm

MY_VAES_INSTR macro cmd, dest, a
        MY_VAES_INSTR_2  cmd, dest, dest, a
endm

MY_vaesenc macro dest, a
        MY_VAES_INSTR  0dcH, dest, a
endm
MY_vaesenclast macro dest, a
        MY_VAES_INSTR  0ddH, dest, a
endm
MY_vaesdec macro dest, a
        MY_VAES_INSTR  0deH, dest, a
endm
MY_vaesdeclast macro dest, a
        MY_VAES_INSTR  0dfH, dest, a
endm


AVX__VAES_DEC macro reg
        MY_vaesdec reg, key_ymm_n
endm

AVX__VAES_DEC_LAST_key_last macro reg
        ; MY_vaesdeclast reg, key_ymm_n
        MY_vaesdeclast reg, key_last_ymm_n
endm

AVX__VAES_ENC macro reg
        MY_vaesenc reg, key_ymm_n
endm

AVX__VAES_ENC_LAST macro reg
        MY_vaesenclast reg, key_ymm_n
endm

AVX__vinserti128_TO_HIGH macro dest, src
        vinserti128  dest, dest, src, 1
endm


MY_PROC AesCbc_Decode_HW_256, 3
  ifdef use_vaes_256
        MY_PROLOG NUM_CBC_REGS
        
        cmp    rN, ways * 2
        jb     AesCbc_Decode_HW_start_2

        vmovdqa iv, xmmword ptr [keys]
        add     keys, 32

        vbroadcasti128  key0_ymm, xmmword ptr [keys + 1 * ksize_r]
        vbroadcasti128  key_last_ymm, xmmword ptr [keys]
        sub     ksize_x, 16
        mov     koffs_x, ksize_x
        add     ksize_x, ksize_x
        
        AVX_STACK_SUB = ((NUM_AES_KEYS_MAX + 1 - 2) * 32)
        push    keys2
        sub     r4, AVX_STACK_SUB
        ; sub     r4, 32
        ; sub     r4, ksize_r
        ; lea     keys2, [r4 + 32]
        mov     keys2, r4
        and     keys2, -32
    broad:
        vbroadcasti128  key_ymm, xmmword ptr [keys + 1 * koffs_r]
        vmovdqa         ymmword ptr [keys2 + koffs_r * 2], key_ymm
        sub     koffs_r, 16
        ; jnc     broad
        jnz     broad

        sub     rN, ways * 2

    align 16
    avx_cbcdec_nextBlock2:
        mov     koffs_x, ksize_x
        ; AVX__WOP_KEY    AVX__CBC_START, 1 * koffs_r + 32
        AVX__WOP    AVX__CBC_START
    @@:
        AVX__WOP_KEY    AVX__VAES_DEC, 1 * koffs_r
        sub     koffs_r, 32
        jnz     @B
        ; AVX__WOP_KEY    AVX__VAES_DEC_LAST, 0
        AVX__WOP_n   AVX__VAES_DEC_LAST_key_last

        AVX__vinserti128_TO_HIGH  iv_ymm, xmmword ptr [rD]
        AVX__WOP        AVX__CBC_END

        vmovdqa         iv, xmmword ptr [rD + ways * 32 - 16]
        AVX__WOP        AVX__WRITE_TO_DATA
        
        add     rD, ways * 32
        sub     rN, ways * 2
        jnc     avx_cbcdec_nextBlock2
        add     rN, ways * 2

        shr     ksize_x, 1
        
        ; lea     r4, [r4 + 1 * ksize_r + 32]
        add     r4, AVX_STACK_SUB
        pop     keys2

        vzeroupper
        jmp     AesCbc_Decode_HW_start_3
  else
        jmp     AesCbc_Decode_HW_start
  endif
MY_ENDP
MY_SEG_ENDP



    
; ---------- AES-CBC Encode ----------

e0  equ  xmm1

CENC_START_KEY     equ 2
CENC_NUM_REG_KEYS  equ (3 * 2)
; last_key equ @CatStr(xmm, %(CENC_START_KEY + CENC_NUM_REG_KEYS))

MY_SEG_PROC AesCbc_Encode_HW, 3
        MY_PROLOG (CENC_START_KEY + CENC_NUM_REG_KEYS + 0)

        movdqa  state, [keys]
        add     keys, 32
        
    i = 0
    rept CENC_NUM_REG_KEYS
        movdqa  @CatStr(xmm, %(CENC_START_KEY + i)), [keys + i * 16]
        i = i + 1
    endm
                                        
        add     keys, ksize_r
        neg     ksize_r
        add     ksize_r, (16 * CENC_NUM_REG_KEYS)
        ; movdqa  last_key, [keys]
        jmp     check_e

    align 16
    nextBlock_e:
        movdqa  e0, [rD]
        mov     koffs_r, ksize_r
        pxor    e0, @CatStr(xmm, %(CENC_START_KEY))
        pxor    state, e0
        
    i = 1
    rept (CENC_NUM_REG_KEYS - 1)
        aesenc  state, @CatStr(xmm, %(CENC_START_KEY + i))
        i = i + 1
    endm

    @@:
        OP_KEY  aesenc, 1 * koffs_r
        OP_KEY  aesenc, 1 * koffs_r + 16
        add     koffs_r, 32
        jnz     @B
        OP_KEY  aesenclast, 0
        ; aesenclast state, last_key
        
        movdqa  [rD], state
        add     rD, 16
    check_e:
        sub     rN, 1
        jnc     nextBlock_e

        ; movdqa  [keys - 32], state
        movdqa  [keys + 1 * ksize_r - (16 * CENC_NUM_REG_KEYS) - 32], state
MY_EPILOG
MY_SEG_ENDP


    
; ---------- AES-CTR ----------

ifdef x64
        ; ways = 11
endif

       
one             equ  @CatStr(xmm, %(ways_start_reg + ways + 1))
one_ymm         equ  @CatStr(ymm, %(ways_start_reg + ways + 1))
key0            equ  @CatStr(xmm, %(ways_start_reg + ways + 2))
key0_ymm        equ  @CatStr(ymm, %(ways_start_reg + ways + 2))
NUM_CTR_REGS    equ  (ways_start_reg + ways + 3)

INIT_CTR macro reg, _ppp_
        paddq   iv, one
        movdqa  reg, iv
endm


MY_SEG_PROC AesCtr_Code_HW, 3
    Ctr_start::
        MY_PROLOG NUM_CTR_REGS

    Ctr_start_2::
        movdqa  iv, [keys]
        add     keys, 32
        movdqa  key0, [keys]

        add     keys, ksize_r
        neg     ksize_r
        add     ksize_r, 16
        
    Ctr_start_3::
        mov     koffs_x, 1
        movd    one, koffs_x
        jmp     check2_c

    align 16
    nextBlocks2_c:
        WOP     INIT_CTR, 0
        mov     koffs_r, ksize_r
        ; WOP_KEY pxor, 1 * koffs_r -16
        WOP     pxor, key0
    @@:
        WOP_KEY aesenc, 1 * koffs_r
        add     koffs_r, 16
        jnz     @B
        WOP_KEY aesenclast, 0
        
        WOP     XOR_WITH_DATA
        WOP     WRITE_TO_DATA
        add     rD, ways * 16
    check2_c:
        sub     rN, ways
        jnc     nextBlocks2_c
        add     rN, ways

        sub     keys, 16
        add     ksize_r, 16
        
        jmp     check_c

    ; align 16
    nextBlock_c:
        paddq   iv, one
        ; movdqa  state, [keys + 1 * koffs_r - 16]
        movdqa  state, key0
        mov     koffs_r, ksize_r
        pxor    state, iv
        
    @@:
        OP_KEY  aesenc, 1 * koffs_r
        OP_KEY  aesenc, 1 * koffs_r + 16
        add     koffs_r, 32
        jnz     @B
        OP_KEY  aesenc, 0
        OP_KEY  aesenclast, 16
        
        pxor    state, [rD]
        movdqa  [rD], state
        add     rD, 16
    check_c:
        sub     rN, 1
        jnc     nextBlock_c

        ; movdqa  [keys - 32], iv
        movdqa  [keys + 1 * ksize_r - 16 - 32], iv
MY_EPILOG


MY_PROC AesCtr_Code_HW_256, 3
  ifdef use_vaes_256
        MY_PROLOG NUM_CTR_REGS

        cmp    rN, ways * 2
        jb     Ctr_start_2

        vbroadcasti128  iv_ymm, xmmword ptr [keys]
        add     keys, 32
        vbroadcasti128  key0_ymm, xmmword ptr [keys]
        mov     koffs_x, 1
        vmovd           one, koffs_x
        vpsubq  iv_ymm, iv_ymm, one_ymm
        vpaddq  one, one, one
        AVX__vinserti128_TO_HIGH     one_ymm, one
        
        add     keys, ksize_r
        sub     ksize_x, 16
        neg     ksize_r
        mov     koffs_r, ksize_r
        add     ksize_r, ksize_r

        AVX_STACK_SUB = ((NUM_AES_KEYS_MAX + 1 - 1) * 32)
        push    keys2
        lea     keys2, [r4 - 32]
        sub     r4, AVX_STACK_SUB
        and     keys2, -32
        vbroadcasti128  key_ymm, xmmword ptr [keys]
        vmovdqa         ymmword ptr [keys2], key_ymm
     @@:
        vbroadcasti128  key_ymm, xmmword ptr [keys + 1 * koffs_r]
        vmovdqa         ymmword ptr [keys2 + koffs_r * 2], key_ymm
        add     koffs_r, 16
        jnz     @B

        sub     rN, ways * 2
        
    align 16
    avx_ctr_nextBlock2:
        mov             koffs_r, ksize_r
        AVX__WOP        AVX__CTR_START
        ; AVX__WOP_KEY    AVX__CTR_START, 1 * koffs_r - 32
    @@:
        AVX__WOP_KEY    AVX__VAES_ENC, 1 * koffs_r
        add     koffs_r, 32
        jnz     @B
        AVX__WOP_KEY    AVX__VAES_ENC_LAST, 0
       
        AVX__WOP        AVX__XOR_WITH_DATA
        AVX__WOP        AVX__WRITE_TO_DATA
        
        add     rD, ways * 32
        sub     rN, ways * 2
        jnc     avx_ctr_nextBlock2
        add     rN, ways * 2
        
        vextracti128    iv, iv_ymm, 1
        sar     ksize_r, 1
       
        add     r4, AVX_STACK_SUB
        pop     keys2
        
        vzeroupper
        jmp     Ctr_start_3
  else
        jmp     Ctr_start
  endif
MY_ENDP
MY_SEG_ENDP

end
