// LzmaDecOpt.S -- ARM64-ASM version of LzmaDec_DecodeReal_3() function
// 2021-04-25 : Igor Pavlov : Public domain

/*
; 3 - is the code compatibility version of LzmaDec_DecodeReal_*()
; function for check at link time.
; That code is tightly coupled with LzmaDec_TryDummy()
; and with another functions in LzmaDec.c file.
; CLzmaDec structure, (probs) array layout, input and output of
; LzmaDec_DecodeReal_*() must be equal in both versions (C / ASM).
*/


#include "7zAsm.S"

	// .arch armv8-a
	// .file        "LzmaDecOpt.c"
	.text
	.align	2
	.p2align 4,,15
#ifdef __APPLE__
        .globl _LzmaDec_DecodeReal_3
#else        
	.global LzmaDec_DecodeReal_3
#endif        
	// .type LzmaDec_DecodeReal_3, %function

// #define _LZMA_SIZE_OPT 1

#define LZMA_USE_4BYTES_FILL 1
// #define LZMA_USE_2BYTES_COPY 1
// #define LZMA_USE_CMOV_LZ_WRAP 1
// #define _LZMA_PROB32 1

#define MY_ALIGN_FOR_ENTRY   MY_ALIGN_32
#define MY_ALIGN_FOR_LOOP    MY_ALIGN_32
#define MY_ALIGN_FOR_LOOP_16 MY_ALIGN_16

#ifdef _LZMA_PROB32
        .equ PSHIFT , 2
        .macro PLOAD dest:req, mem:req
                ldr     \dest, [\mem]
        .endm
        .macro PLOAD_PREINDEXED dest:req, mem:req, offset:req
                ldr     \dest, [\mem, \offset]!
        .endm
        .macro PLOAD_2 dest:req, mem1:req, mem2:req
                ldr     \dest, [\mem1, \mem2]
        .endm
        .macro PLOAD_LSL dest:req, mem1:req, mem2:req
                ldr     \dest, [\mem1, \mem2, lsl #PSHIFT]
        .endm
        .macro PSTORE src:req, mem:req
                str     \src, [\mem]
        .endm
        .macro PSTORE_2 src:req, mem1:req, mem2:req
                str     \src, [\mem1, \mem2]
        .endm
        .macro PSTORE_LSL src:req, mem1:req, mem2:req
                str     \src, [\mem1, \mem2, lsl #PSHIFT]
        .endm
        .macro PSTORE_LSL_M1 src:req, mem1:req, mem2:req, temp_reg:req
                // you must check that temp_reg is free register when macro is used
                add     \temp_reg, \mem1, \mem2
                str     \src, [\temp_reg, \mem2]
        .endm
#else
        // .equ PSHIFT  , 1
        #define PSHIFT  1
        .macro PLOAD dest:req, mem:req
                ldrh    \dest, [\mem]
        .endm
        .macro PLOAD_PREINDEXED dest:req, mem:req, offset:req
                ldrh    \dest, [\mem, \offset]!
        .endm
        .macro PLOAD_2 dest:req, mem1:req, mem2:req
                ldrh    \dest, [\mem1, \mem2]
        .endm
        .macro PLOAD_LSL dest:req, mem1:req, mem2:req
                ldrh    \dest, [\mem1, \mem2, lsl #PSHIFT]
        .endm
        .macro PSTORE src:req, mem:req
                strh    \src, [\mem]
        .endm
        .macro PSTORE_2 src:req, mem1:req, mem2:req
                strh    \src, [\mem1, \mem2]
        .endm
        .macro PSTORE_LSL src:req, mem1:req, mem2:req
                strh    \src, [\mem1, \mem2, lsl #PSHIFT]
        .endm
        .macro PSTORE_LSL_M1 src:req, mem1:req, mem2:req, temp_reg:req
                strh    \src, [\mem1, \mem2]
        .endm
#endif

.equ PMULT    , (1 << PSHIFT)
.equ PMULT_2  , (2 << PSHIFT)

.equ kMatchSpecLen_Error_Data , (1 << 9)

#       x7      t0 : NORM_CALC    : prob2 (IF_BIT_1)
#       x6      t1 : NORM_CALC    : probs_state
#       x8      t2 : (LITM) temp  : (TREE) temp
#       x4      t3 : (LITM) bit   : (TREE) temp : UPDATE_0/UPDATE_0 temp
#       x10     t4 : (LITM) offs  : (TREE) probs_PMULT : numBits
#       x9      t5 : (LITM) match : sym2 (ShortDist)
#       x1      t6 : (LITM) litm_prob : (TREE) prob_reg : pbPos
#       x2      t7 : (LITM) prm   : probBranch  : cnt
#       x3      sym : dist
#       x12     len
#       x0      range
#       x5      cod


#define range   w0

// t6
#define pbPos     w1
#define pbPos_R   r1
#define prob_reg  w1
#define litm_prob    prob_reg

// t7
#define probBranch    w2
#define cnt     w2
#define cnt_R   r2
#define prm     r2

#define sym     w3
#define sym_R   r3
#define dist       sym

#define t3      w4
#define bit     w4
#define bit_R   r4
#define update_temp_reg  r4

#define cod     w5

#define t1      w6
#define t1_R    r6
#define probs_state  t1_R

#define t0      w7
#define t0_R    r7
#define prob2      t0

#define t2      w8
#define t2_R    r8 

// t5
#define match   w9
#define sym2    w9
#define sym2_R  r9

#define t4      w10
#define t4_R    r10

#define offs    w10
#define offs_R  r10

#define probs   r11

#define len     w12
#define len_R   x12

#define state   w13
#define state_R r13

#define dicPos          r14
#define buf             r15
#define bufLimit        r16
#define dicBufSize      r17

#define limit           r19
#define rep0            w20
#define rep0_R          r20
#define rep1            w21
#define rep2            w22
#define rep3            w23
#define dic             r24
#define probs_IsMatch   r25
#define probs_Spec      r26
#define checkDicSize    w27
#define processedPos    w28
#define pbMask          w29
#define lc2_lpMask      w30


.equ kNumBitModelTotalBits   , 11
.equ kBitModelTotal          , (1 << kNumBitModelTotalBits)
.equ kNumMoveBits            , 5
.equ kBitModelOffset         , (kBitModelTotal - (1 << kNumMoveBits) + 1)

.macro NORM_2 macro
        ldrb    t0, [buf], 1
        shl     range, 8
        orr     cod, t0, cod, lsl 8
        /*
        mov     t0, cod
        ldrb    cod, [buf], 1
        shl     range, 8
        bfi	cod, t0, #8, #24
        */
.endm

.macro TEST_HIGH_BYTE_range macro
        tst     range, 0xFF000000
.endm   

.macro NORM macro
        TEST_HIGH_BYTE_range
        jnz     1f
        NORM_2
1:
.endm


# ---------- Branch MACROS ----------

.macro UPDATE_0__0
        sub     prob2, probBranch, kBitModelOffset
.endm

.macro UPDATE_0__1
        sub     probBranch, probBranch, prob2, asr #(kNumMoveBits)
.endm

.macro UPDATE_0__2 probsArray:req, probOffset:req, probDisp:req
     .if \probDisp == 0
        PSTORE_2  probBranch, \probsArray, \probOffset
    .elseif \probOffset == 0
        PSTORE_2  probBranch, \probsArray, \probDisp * PMULT
    .else
        .error "unsupported"
        // add     update_temp_reg, \probsArray, \probOffset
        PSTORE_2  probBranch, update_temp_reg, \probDisp * PMULT
    .endif
.endm

.macro UPDATE_0 probsArray:req, probOffset:req, probDisp:req
        UPDATE_0__0
        UPDATE_0__1
        UPDATE_0__2 \probsArray, \probOffset, \probDisp
.endm


.macro UPDATE_1 probsArray:req, probOffset:req, probDisp:req
        // sub     cod, cod, prob2
        // sub     range, range, prob2
        p2_sub  cod, range
        sub     range, prob2, range
        sub     prob2, probBranch, probBranch, lsr #(kNumMoveBits)
    .if \probDisp == 0
        PSTORE_2  prob2, \probsArray, \probOffset
    .elseif \probOffset == 0
        PSTORE_2  prob2, \probsArray, \probDisp * PMULT
    .else
        .error "unsupported"
        // add     update_temp_reg, \probsArray, \probOffset
        PSTORE_2  prob2, update_temp_reg, \probDisp * PMULT
    .endif
.endm


.macro CMP_COD_BASE
        NORM
        // lsr     prob2, range, kNumBitModelTotalBits
        // imul    prob2, probBranch
        // cmp     cod, prob2
        mov     prob2, range
        shr     range, kNumBitModelTotalBits
        imul    range, probBranch
        cmp     cod, range
.endm

.macro CMP_COD_1 probsArray:req
        PLOAD   probBranch, \probsArray
        CMP_COD_BASE
.endm

.macro CMP_COD_3 probsArray:req, probOffset:req, probDisp:req
    .if \probDisp == 0
        PLOAD_2 probBranch, \probsArray, \probOffset
    .elseif \probOffset == 0
        PLOAD_2 probBranch, \probsArray, \probDisp * PMULT
    .else
        .error "unsupported"
        add     update_temp_reg, \probsArray, \probOffset
        PLOAD_2 probBranch, update_temp_reg, \probDisp * PMULT
    .endif
        CMP_COD_BASE
.endm


.macro IF_BIT_1_NOUP probsArray:req, probOffset:req, probDisp:req, toLabel:req
        CMP_COD_3 \probsArray, \probOffset, \probDisp
        jae     \toLabel
.endm


.macro IF_BIT_1 probsArray:req, probOffset:req, probDisp:req, toLabel:req
        IF_BIT_1_NOUP \probsArray, \probOffset, \probDisp, \toLabel
        UPDATE_0 \probsArray, \probOffset, \probDisp
.endm


.macro IF_BIT_0_NOUP probsArray:req, probOffset:req, probDisp:req, toLabel:req
        CMP_COD_3 \probsArray, \probOffset, \probDisp
        jb      \toLabel
.endm

.macro IF_BIT_0_NOUP_1 probsArray:req, toLabel:req
        CMP_COD_1 \probsArray
        jb      \toLabel
.endm


# ---------- CMOV MACROS ----------

.macro NORM_LSR
        NORM
        lsr     t0, range, #kNumBitModelTotalBits
.endm

.macro COD_RANGE_SUB
        subs    t1, cod, t0
        p2_sub  range, t0
.endm

.macro RANGE_IMUL prob:req
        imul    t0, \prob
.endm

.macro NORM_CALC prob:req
        NORM_LSR
        RANGE_IMUL \prob
        COD_RANGE_SUB
.endm

.macro CMOV_range
        cmovb   range, t0
.endm

.macro CMOV_code
        cmovae  cod, t1
.endm

.macro CMOV_code_Model_Pre prob:req
        sub     t0, \prob, kBitModelOffset
        CMOV_code
        cmovae  t0, \prob
.endm
        

.macro PUP_BASE_2 prob:req, dest_reg:req
        # only sar works for both 16/32 bit prob modes
        sub     \dest_reg, \prob, \dest_reg, asr #(kNumMoveBits)
.endm

.macro PUP prob:req, probPtr:req, mem2:req
        PUP_BASE_2 \prob, t0
        PSTORE_2   t0, \probPtr, \mem2
.endm



#define probs_PMULT t4_R

.macro BIT_01
        add     probs_PMULT, probs, PMULT
.endm


.macro BIT_0_R prob:req
        PLOAD_2 \prob, probs, 1 * PMULT
        NORM_LSR
            sub     t3, \prob, kBitModelOffset
        RANGE_IMUL  \prob
            PLOAD_2 t2, probs, 1 * PMULT_2
        COD_RANGE_SUB
        CMOV_range
            cmovae  t3, \prob
        PLOAD_2 t0, probs, 1 * PMULT_2 + PMULT
            PUP_BASE_2 \prob, t3
        csel   \prob, t2, t0, lo
            CMOV_code
        mov     sym, 2
        PSTORE_2  t3, probs, 1 * PMULT
            adc     sym, sym, wzr
        BIT_01
.endm

.macro BIT_1_R prob:req
        NORM_LSR
            p2_add  sym, sym
            sub     t3, \prob, kBitModelOffset
        RANGE_IMUL  \prob
            PLOAD_LSL t2, probs, sym_R
        COD_RANGE_SUB
        CMOV_range
            cmovae  t3, \prob
        PLOAD_LSL t0, probs_PMULT, sym_R
            PUP_BASE_2 \prob, t3
        csel   \prob, t2, t0, lo
            CMOV_code
        PSTORE_LSL_M1  t3, probs, sym_R, t2_R
            adc     sym, sym, wzr
.endm


.macro BIT_2_R prob:req
        NORM_LSR
            p2_add  sym, sym
            sub     t3, \prob, kBitModelOffset
        RANGE_IMUL  \prob
        COD_RANGE_SUB
        CMOV_range
            cmovae  t3, \prob
            CMOV_code
            PUP_BASE_2 \prob, t3
        PSTORE_LSL_M1  t3, probs, sym_R, t2_R
            adc     sym, sym, wzr
.endm


# ---------- MATCHED LITERAL ----------

.macro LITM_0 macro
        shl     match, (PSHIFT + 1)
        and     bit, match, 256 * PMULT
        add     prm, probs, 256 * PMULT + 1 * PMULT
        p2_add  match, match
        p2_add  prm, bit_R
        eor     offs, bit, 256 * PMULT
        PLOAD   litm_prob, prm
        
        NORM_LSR
            sub     t2, litm_prob, kBitModelOffset
        RANGE_IMUL  litm_prob
        COD_RANGE_SUB
        cmovae  offs, bit
            CMOV_range
        and     bit, match, offs
            cmovae  t2, litm_prob
            CMOV_code
            mov     sym, 2
        PUP_BASE_2 litm_prob, t2
        PSTORE  t2, prm
        add     prm, probs, offs_R
        adc     sym, sym, wzr
.endm

.macro LITM macro
        p2_add  prm, bit_R
            xor     offs, bit
        PLOAD_LSL litm_prob, prm, sym_R
        
        NORM_LSR
            p2_add  match, match
            sub     t2, litm_prob, kBitModelOffset
        RANGE_IMUL  litm_prob
        COD_RANGE_SUB
        cmovae  offs, bit
            CMOV_range
        and     bit, match, offs
            cmovae  t2, litm_prob
            CMOV_code
        PUP_BASE_2 litm_prob, t2
        PSTORE_LSL t2, prm, sym_R
        add     prm, probs, offs_R
        adc     sym, sym, sym
.endm


.macro LITM_2 macro
        p2_add  prm, bit_R
        PLOAD_LSL litm_prob, prm, sym_R
        
        NORM_LSR
            sub     t2, litm_prob, kBitModelOffset
        RANGE_IMUL  litm_prob
        COD_RANGE_SUB
            CMOV_range
            cmovae  t2, litm_prob
            CMOV_code
        PUP_BASE_2 litm_prob, t2
        PSTORE_LSL t2, prm, sym_R
        adc     sym, sym, sym
.endm


# ---------- REVERSE BITS ----------

.macro REV_0 prob:req
        NORM_CALC \prob
        CMOV_range
        PLOAD   t2, sym2_R
        PLOAD_2 t3, probs, 3 * PMULT
        CMOV_code_Model_Pre \prob
        add     t1_R, probs, 3 * PMULT
        cmovae  sym2_R, t1_R
        PUP     \prob, probs, 1 * PMULT
        csel    \prob, t2, t3, lo
.endm


.macro REV_1 prob:req, step:req
        NORM_LSR
            PLOAD_PREINDEXED  t2, sym2_R, (\step * PMULT)
        RANGE_IMUL  \prob
        COD_RANGE_SUB
        CMOV_range
        PLOAD_2 t3, sym2_R, (\step * PMULT)
        sub     t0, \prob, kBitModelOffset
        CMOV_code
        add     t1_R, sym2_R, \step * PMULT
        cmovae  t0, \prob
        cmovae  sym2_R, t1_R
        PUP_BASE_2 \prob, t0
        csel    \prob, t2, t3, lo
        PSTORE_2   t0, t1_R, 0 - \step * PMULT_2
.endm


.macro REV_2 prob:req, step:req
        sub     t1_R, sym2_R, probs
        NORM_LSR
            orr     sym, sym, t1, lsr #PSHIFT
        RANGE_IMUL  \prob
        COD_RANGE_SUB
        sub     t2, sym, \step
        CMOV_range
        cmovb   sym, t2
        CMOV_code_Model_Pre \prob
        PUP     \prob, sym2_R, 0
.endm


.macro REV_1_VAR prob:req
        PLOAD   \prob, sym_R
        mov     probs, sym_R
        p2_add  sym_R, sym2_R
        NORM_LSR
            add     t2_R, sym_R, sym2_R
        RANGE_IMUL  \prob
        COD_RANGE_SUB
        cmovae  sym_R, t2_R
        CMOV_range
        CMOV_code_Model_Pre \prob
        p2_add  sym2, sym2
        PUP     \prob, probs, 0
.endm


.macro add_big dest:req, src:req, param:req
    .if (\param) < (1 << 12)
        add     \dest, \src, \param
    .else
        #ifndef _LZMA_PROB32    
          .error "unexpcted add_big expansion"
        #endif
        add     \dest, \src, (\param) / 2
        add     \dest, \dest, (\param) - (\param) / 2
    .endif
.endm

.macro sub_big dest:req, src:req, param:req
    .if (\param) < (1 << 12)
        sub     \dest, \src, \param
    .else
        #ifndef _LZMA_PROB32    
          .error "unexpcted sub_big expansion"
        #endif
        sub     \dest, \src, (\param) / 2
        sub     \dest, \dest, (\param) - (\param) / 2
    .endif
.endm


.macro SET_probs offset:req
        // add_big probs, probs_Spec, (\offset) * PMULT
        add     probs, probs_IsMatch, ((\offset) - IsMatch) * PMULT
.endm        


.macro LIT_PROBS
        add     sym, sym, processedPos, lsl 8
        inc     processedPos
        UPDATE_0__0
        shl     sym, lc2_lpMask
        SET_probs Literal
        p2_and  sym, lc2_lpMask
        // p2_add  probs_state, pbPos_R
        p2_add  probs, sym_R
        UPDATE_0__1
        add     probs, probs, sym_R, lsl 1
        UPDATE_0__2 probs_state, pbPos_R, 0
.endm



.equ kNumPosBitsMax       , 4
.equ kNumPosStatesMax     , (1 << kNumPosBitsMax)
                         
.equ kLenNumLowBits       , 3
.equ kLenNumLowSymbols    , (1 << kLenNumLowBits)
.equ kLenNumHighBits      , 8
.equ kLenNumHighSymbols   , (1 << kLenNumHighBits)
.equ kNumLenProbs         , (2 * kLenNumLowSymbols * kNumPosStatesMax + kLenNumHighSymbols)
                         
.equ LenLow               , 0
.equ LenChoice            , LenLow
.equ LenChoice2           , (LenLow + kLenNumLowSymbols)
.equ LenHigh              , (LenLow + 2 * kLenNumLowSymbols * kNumPosStatesMax)
                         
.equ kNumStates           , 12
.equ kNumStates2          , 16
.equ kNumLitStates        , 7
                         
.equ kStartPosModelIndex  , 4
.equ kEndPosModelIndex    , 14
.equ kNumFullDistances    , (1 << (kEndPosModelIndex >> 1))
                         
.equ kNumPosSlotBits      , 6
.equ kNumLenToPosStates   , 4
                         
.equ kNumAlignBits        , 4
.equ kAlignTableSize      , (1 << kNumAlignBits)
                         
.equ kMatchMinLen         , 2
.equ kMatchSpecLenStart   , (kMatchMinLen + kLenNumLowSymbols * 2 + kLenNumHighSymbols)

// .equ kStartOffset    , 1408
.equ kStartOffset    , 0
.equ SpecPos         , (-kStartOffset)
.equ IsRep0Long      , (SpecPos + kNumFullDistances)
.equ RepLenCoder     , (IsRep0Long + (kNumStates2 << kNumPosBitsMax))
.equ LenCoder        , (RepLenCoder + kNumLenProbs)
.equ IsMatch         , (LenCoder + kNumLenProbs)
.equ kAlign          , (IsMatch + (kNumStates2 << kNumPosBitsMax))
.equ IsRep           , (kAlign + kAlignTableSize)
.equ IsRepG0         , (IsRep + kNumStates)
.equ IsRepG1         , (IsRepG0 + kNumStates)
.equ IsRepG2         , (IsRepG1 + kNumStates)
.equ PosSlot         , (IsRepG2 + kNumStates)
.equ Literal         , (PosSlot + (kNumLenToPosStates << kNumPosSlotBits))
.equ NUM_BASE_PROBS  , (Literal + kStartOffset)

.if kStartOffset != 0   // && IsMatch != 0
  .error "Stop_Compiling_Bad_StartOffset"
.endif

.if NUM_BASE_PROBS != 1984
  .error "Stop_Compiling_Bad_LZMA_PROBS"
.endif

.equ offset_lc    , 0
.equ offset_lp    , 1
.equ offset_pb    , 2
.equ offset_dicSize       , 4
.equ offset_probs         , 4 + offset_dicSize
.equ offset_probs_1664    , 8 + offset_probs
.equ offset_dic           , 8 + offset_probs_1664
.equ offset_dicBufSize    , 8 + offset_dic
.equ offset_dicPos        , 8 + offset_dicBufSize
.equ offset_buf           , 8 + offset_dicPos
.equ offset_range         , 8 + offset_buf
.equ offset_code          , 4 + offset_range
.equ offset_processedPos  , 4 + offset_code
.equ offset_checkDicSize  , 4 + offset_processedPos
.equ offset_rep0          , 4 + offset_checkDicSize
.equ offset_rep1          , 4 + offset_rep0
.equ offset_rep2          , 4 + offset_rep1
.equ offset_rep3          , 4 + offset_rep2
.equ offset_state         , 4 + offset_rep3
.equ offset_remainLen     , 4 + offset_state
.equ offset_TOTAL_SIZE    , 4 + offset_remainLen

.if offset_TOTAL_SIZE != 96
  .error "Incorrect offset_TOTAL_SIZE"
.endif


.macro IsMatchBranch_Pre
        # prob = probs + IsMatch + (state << kNumPosBitsMax) + posState;
        and     pbPos, pbMask, processedPos, lsl #(kLenNumLowBits + 1 + PSHIFT)
        add     probs_state, probs_IsMatch, state_R
.endm


/*
.macro IsMatchBranch
        IsMatchBranch_Pre
        IF_BIT_1 probs_state, pbPos_R, (IsMatch - IsMatch), IsMatch_label
.endm
*/        

.macro CheckLimits
        cmp     buf, bufLimit
        jae     fin_OK
        cmp     dicPos, limit
        jae     fin_OK
.endm

#define  CheckLimits_lit  CheckLimits
/*
.macro CheckLimits_lit
        cmp     buf, bufLimit
        jae     fin_OK_lit
        cmp     dicPos, limit
        jae     fin_OK_lit
.endm
*/


#define PARAM_lzma      REG_ABI_PARAM_0
#define PARAM_limit     REG_ABI_PARAM_1
#define PARAM_bufLimit  REG_ABI_PARAM_2


.macro LOAD_LZMA_VAR reg:req, struct_offs:req
        ldr     \reg, [PARAM_lzma, \struct_offs]
.endm

.macro LOAD_LZMA_BYTE reg:req, struct_offs:req
        ldrb    \reg, [PARAM_lzma, \struct_offs]
.endm

.macro LOAD_LZMA_PAIR reg0:req, reg1:req, struct_offs:req
        ldp     \reg0, \reg1, [PARAM_lzma, \struct_offs]
.endm


LzmaDec_DecodeReal_3:
_LzmaDec_DecodeReal_3:
/*
.LFB0:
	.cfi_startproc  
*/

	stp	x19, x20, [sp, -128]!
	stp	x21, x22, [sp, 16]
	stp	x23, x24, [sp, 32]
	stp	x25, x26, [sp, 48]
	stp	x27, x28, [sp, 64]
	stp	x29, x30, [sp, 80]
        
        str     PARAM_lzma, [sp, 120]
        
        mov     bufLimit, PARAM_bufLimit
        mov     limit, PARAM_limit
        
        LOAD_LZMA_PAIR  dic, dicBufSize, offset_dic
        LOAD_LZMA_PAIR  dicPos, buf, offset_dicPos
        LOAD_LZMA_PAIR  rep0, rep1, offset_rep0
        LOAD_LZMA_PAIR  rep2, rep3, offset_rep2
        
        mov     t0, 1 << (kLenNumLowBits + 1 + PSHIFT)
        LOAD_LZMA_BYTE  pbMask, offset_pb
        p2_add  limit, dic
        mov     len, wzr    // we can set it in all requiread branches instead
        lsl     pbMask, t0, pbMask
        p2_add  dicPos, dic
        p2_sub  pbMask, t0

        LOAD_LZMA_BYTE  lc2_lpMask, offset_lc
        mov     t0, 256 << PSHIFT
        LOAD_LZMA_BYTE  t1, offset_lp
        p2_add  t1, lc2_lpMask
        p2_sub  lc2_lpMask, (256 << PSHIFT) - PSHIFT
        shl     t0, t1
        p2_add  lc2_lpMask, t0
        
        LOAD_LZMA_VAR   probs_Spec, offset_probs
        LOAD_LZMA_VAR   checkDicSize, offset_checkDicSize
        LOAD_LZMA_VAR   processedPos, offset_processedPos
        LOAD_LZMA_VAR   state, offset_state
        // range is r0 : this load must be last don't move        
        LOAD_LZMA_PAIR  range, cod, offset_range    
        mov     sym, wzr
        shl     state, PSHIFT

        add_big probs_IsMatch, probs_Spec, ((IsMatch - SpecPos) << PSHIFT)

        // if (processedPos != 0 || checkDicSize != 0)
        orr     t0, checkDicSize, processedPos
        cbz     t0, 1f
        add     t0_R, dicBufSize, dic
        cmp     dicPos, dic
        cmovne  t0_R, dicPos
        ldrb    sym, [t0_R, -1]
1:
        IsMatchBranch_Pre
        cmp     state, 4 * PMULT
        jb      lit_end
        cmp     state, kNumLitStates * PMULT
        jb      lit_matched_end
        jmp     lz_end
        

        
#define BIT_0  BIT_0_R prob_reg
#define BIT_1  BIT_1_R prob_reg
#define BIT_2  BIT_2_R prob_reg

# ---------- LITERAL ----------
MY_ALIGN_64
lit_start:
        mov     state, wzr
lit_start_2:
        LIT_PROBS

    #ifdef _LZMA_SIZE_OPT

        PLOAD_2 prob_reg, probs, 1 * PMULT
        mov     sym, 1
        BIT_01        
MY_ALIGN_FOR_LOOP
lit_loop:
        BIT_1
        tbz     sym, 7, lit_loop
        
    #else
        
        BIT_0
        BIT_1
        BIT_1
        BIT_1
        BIT_1
        BIT_1
        BIT_1
        
    #endif

        BIT_2
        IsMatchBranch_Pre
        strb    sym, [dicPos], 1
        p2_and  sym, 255
                
        CheckLimits_lit
lit_end:
        IF_BIT_0_NOUP probs_state, pbPos_R, (IsMatch - IsMatch), lit_start

        # jmp     IsMatch_label
        

#define FLAG_STATE_BITS (4 + PSHIFT)          

# ---------- MATCHES ----------
# MY_ALIGN_FOR_ENTRY
IsMatch_label:
        UPDATE_1 probs_state, pbPos_R, (IsMatch - IsMatch)
        IF_BIT_1 probs_state, 0, (IsRep - IsMatch), IsRep_label

        SET_probs LenCoder
        or      state, (1 << FLAG_STATE_BITS)

# ---------- LEN DECODE ----------
len_decode:
        mov     len, 8 - kMatchMinLen
        IF_BIT_0_NOUP_1 probs, len_mid_0
        UPDATE_1 probs, 0, 0
        p2_add  probs, (1 << (kLenNumLowBits + PSHIFT))
        mov     len, 0 - kMatchMinLen
        IF_BIT_0_NOUP_1 probs, len_mid_0
        UPDATE_1 probs, 0, 0
        p2_add  probs, LenHigh * PMULT - (1 << (kLenNumLowBits + PSHIFT))
        
    #if 0 == 1
        BIT_0
        BIT_1
        BIT_1
        BIT_1
        BIT_1
        BIT_1
   #else
        PLOAD_2 prob_reg, probs, 1 * PMULT
        mov     sym, 1
        BIT_01
MY_ALIGN_FOR_LOOP
len8_loop:
        BIT_1
        tbz     sym, 6, len8_loop
   #endif        
        
        mov     len, (kLenNumHighSymbols - kLenNumLowSymbols * 2) - kMatchMinLen
        jmp     len_mid_2 
        
MY_ALIGN_FOR_ENTRY
len_mid_0:
        UPDATE_0 probs, 0, 0
        p2_add  probs, pbPos_R
        BIT_0
len_mid_2:
        BIT_1
        BIT_2
        sub     len, sym, len
        tbz     state, FLAG_STATE_BITS, copy_match
        
# ---------- DECODE DISTANCE ----------
        // probs + PosSlot + ((len < kNumLenToPosStates ? len : kNumLenToPosStates - 1) << kNumPosSlotBits);

        mov     t0, 3 + kMatchMinLen
        cmp     len, 3 + kMatchMinLen
        cmovb   t0, len
        SET_probs PosSlot - (kMatchMinLen << (kNumPosSlotBits))
        add     probs, probs, t0_R, lsl #(kNumPosSlotBits + PSHIFT)
        
    #ifdef _LZMA_SIZE_OPT

        PLOAD_2 prob_reg, probs, 1 * PMULT
        mov     sym, 1
        BIT_01
MY_ALIGN_FOR_LOOP
slot_loop:
        BIT_1
        tbz     sym, 5, slot_loop
        
    #else
        
        BIT_0
        BIT_1
        BIT_1
        BIT_1
        BIT_1
        
    #endif
        
    #define numBits t4
        mov     numBits, sym
        BIT_2
        // we need only low bits
        p2_and  sym, 3
        cmp     numBits, 32 + kEndPosModelIndex / 2
        jb      short_dist

        SET_probs kAlign

        #  unsigned numDirectBits = (unsigned)(((distance >> 1) - 1));
        p2_sub  numBits, (32 + 1 + kNumAlignBits)
        #  distance = (2 | (distance & 1));
        or      sym, 2
        PLOAD_2 prob_reg, probs, 1 * PMULT
        add     sym2_R, probs, 2 * PMULT
        
# ---------- DIRECT DISTANCE ----------

.macro DIRECT_1
        shr     range, 1
        subs    t0, cod, range
        p2_add  sym, sym
        // add     t1, sym, 1
        csel    cod, cod, t0, mi
        csinc   sym, sym, sym, mi
        // csel    sym, t1, sym, pl
        // adc     sym, sym, sym // not 100% compatible for "corruptued-allowed" LZMA streams
        dec_s   numBits
        je      direct_end
.endm

    #ifdef _LZMA_SIZE_OPT

        jmp     direct_norm
MY_ALIGN_FOR_ENTRY
direct_loop:
        DIRECT_1
direct_norm:
        TEST_HIGH_BYTE_range
        jnz     direct_loop
        NORM_2
        jmp     direct_loop

    #else        

.macro DIRECT_2
        TEST_HIGH_BYTE_range
        jz      direct_unroll
        DIRECT_1
.endm

        DIRECT_2
        DIRECT_2
        DIRECT_2
        DIRECT_2
        DIRECT_2
        DIRECT_2
        DIRECT_2
        DIRECT_2
        
direct_unroll:
        NORM_2
        DIRECT_1
        DIRECT_1
        DIRECT_1
        DIRECT_1
        DIRECT_1
        DIRECT_1
        DIRECT_1
        DIRECT_1
        jmp     direct_unroll
    
    #endif

MY_ALIGN_FOR_ENTRY
direct_end:
        shl     sym, kNumAlignBits
        REV_0   prob_reg
        REV_1   prob_reg, 2
        REV_1   prob_reg, 4
        REV_2   prob_reg, 8

decode_dist_end:

    // if (distance >= (checkDicSize == 0 ? processedPos: checkDicSize))

        tst     checkDicSize, checkDicSize
        csel    t0, processedPos, checkDicSize, eq
        cmp     sym, t0
        jae     end_of_payload
        // jmp     end_of_payload # for debug
        
        mov     rep3, rep2
        mov     rep2, rep1
        mov     rep1, rep0
        add     rep0, sym, 1

.macro  STATE_UPDATE_FOR_MATCH
        // state = (state < kNumStates + kNumLitStates) ? kNumLitStates : kNumLitStates + 3;
        // cmp     state, (kNumStates + kNumLitStates) * PMULT
        cmp     state, kNumLitStates * PMULT + (1 << FLAG_STATE_BITS)
        mov     state, kNumLitStates * PMULT
        mov     t0, (kNumLitStates + 3) * PMULT
        cmovae  state, t0
.endm
        STATE_UPDATE_FOR_MATCH
        
# ---------- COPY MATCH ----------
copy_match:

    // if ((rem = limit - dicPos) == 0) break // return SZ_ERROR_DATA;
        subs    cnt_R, limit, dicPos
        // jz      fin_dicPos_LIMIT
        jz      fin_OK

    // curLen = ((rem < len) ? (unsigned)rem : len);
        cmp     cnt_R, len_R
        cmovae  cnt, len

        sub     t0_R, dicPos, dic
        p2_add  dicPos, cnt_R
        p2_add  processedPos, cnt
        p2_sub  len, cnt
        
    // pos = dicPos - rep0 + (dicPos < rep0 ? dicBufSize : 0);
        p2_sub_s  t0_R, rep0_R
        jae     1f

        cmn     t0_R, cnt_R
        p2_add  t0_R, dicBufSize
        ja      copy_match_cross
1:
# ---------- COPY MATCH FAST ----------
    # t0_R : src_pos
        p2_add  t0_R, dic
        ldrb    sym, [t0_R]
        p2_add  t0_R, cnt_R
        p1_neg  cnt_R

copy_common:
        dec     dicPos

    # dicPos  : (ptr_to_last_dest_BYTE)    
    # t0_R    : (src_lim)
    # cnt_R   : (-curLen)

        IsMatchBranch_Pre
        
        inc_s   cnt_R
        jz      copy_end
        
        cmp     rep0, 1
        je      copy_match_0
   
    #ifdef LZMA_USE_2BYTES_COPY
        strb    sym, [dicPos, cnt_R]
        dec     dicPos
    # dicPos  : (ptr_to_last_dest_16bitWORD)    
        p2_and  cnt_R, -2
        ldrh    sym, [t0_R, cnt_R]
        adds    cnt_R, cnt_R, 2
        jz      2f
MY_ALIGN_FOR_LOOP
1:
        /*
        strh    sym, [dicPos, cnt_R]
        ldrh    sym, [t0_R, cnt_R]
        adds    cnt_R, cnt_R, 2
        jz      2f
        */

        strh    sym, [dicPos, cnt_R]
        ldrh    sym, [t0_R, cnt_R]
        adds    cnt_R, cnt_R, 2
        jnz     1b
2:
        
        /*
        // for universal little/big endian code, but slow
        strh    sym, [dicPos]
        inc     dicPos 
        ldrb    sym, [t0_R, -1]
        */

        #if  __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // we must improve big-endian detection for another compilers 
        // for big-endian we need to revert bytes
        rev16   sym, sym         
        #endif
        
        // (sym) must represent as little-endian here:
        strb    sym, [dicPos], 1
        shr     sym, 8             

    #else

MY_ALIGN_FOR_LOOP
1:
        strb    sym, [dicPos, cnt_R]
        ldrb    sym, [t0_R, cnt_R]
        inc_s   cnt_R
        jz      copy_end

        strb    sym, [dicPos, cnt_R]
        ldrb    sym, [t0_R, cnt_R]
        inc_s   cnt_R
        jnz     1b
    #endif

copy_end:
lz_end_match:
        strb    sym, [dicPos], 1
  
        # IsMatchBranch_Pre
        CheckLimits
lz_end:
        IF_BIT_1_NOUP probs_state, pbPos_R, (IsMatch - IsMatch), IsMatch_label



# ---------- LITERAL MATCHED ----------
                
        LIT_PROBS
        
    // matchByte = dic[dicPos - rep0 + (dicPos < rep0 ? dicBufSize : 0)];

        sub     t0_R, dicPos, dic
        p2_sub_s t0_R, rep0_R
    
    #ifdef LZMA_USE_CMOV_LZ_WRAP
        add     t1_R, t0_R, dicBufSize
        cmovb   t0_R, t1_R
    #else                
        jae     1f
        p2_add  t0_R, dicBufSize
1:
    #endif                        

        ldrb    match, [dic, t0_R]

    // state -= (state < 10) ? 3 : 6;
        sub     sym, state, 6 * PMULT
        cmp     state, 10 * PMULT
        p2_sub  state, 3 * PMULT
        cmovae  state, sym

    #ifdef _LZMA_SIZE_OPT

        mov     offs, 256 * PMULT
        shl     match, (PSHIFT + 1)
        mov     sym, 1
        and     bit, match, offs
        add     prm, probs, offs_R

MY_ALIGN_FOR_LOOP
litm_loop:
        LITM
        tbz     sym, 8, litm_loop
        
    #else
        
        LITM_0
        LITM
        LITM
        LITM
        LITM
        LITM
        LITM
        LITM_2
        
    #endif
    
        IsMatchBranch_Pre
        strb    sym, [dicPos], 1
        p2_and  sym, 255
        
        // mov     len, wzr // LITM uses same regisetr (len / offs). So we clear it
        CheckLimits_lit
lit_matched_end:
        IF_BIT_1_NOUP probs_state, pbPos_R, (IsMatch - IsMatch), IsMatch_label
        # IsMatchBranch
        p2_sub  state, 3 * PMULT
        jmp     lit_start_2
        


# ---------- REP 0 LITERAL ----------
MY_ALIGN_FOR_ENTRY
IsRep0Short_label:
        UPDATE_0 probs_state, pbPos_R, 0

    // dic[dicPos] = dic[dicPos - rep0 + (dicPos < rep0 ? dicBufSize : 0)];
        sub     t0_R, dicPos, dic
        
        // state = state < kNumLitStates ? 9 : 11;
        or      state, 1 * PMULT
        
        # the caller doesn't allow (dicPos >= limit) case for REP_SHORT
        # so we don't need the following (dicPos == limit) check here:
        # cmp     dicPos, limit
        # jae     fin_dicPos_LIMIT_REP_SHORT
        # // jmp fin_dicPos_LIMIT_REP_SHORT // for testing/debug puposes

        inc     processedPos

        IsMatchBranch_Pre
       
        p2_sub_s t0_R, rep0_R
    #ifdef LZMA_USE_CMOV_LZ_WRAP
        add     sym_R, t0_R, dicBufSize
        cmovb   t0_R, sym_R
    #else       
        jae     1f
        p2_add  t0_R, dicBufSize
1:
    #endif
        
        ldrb    sym, [dic, t0_R]
        // mov     len, wzr
        jmp     lz_end_match
        
MY_ALIGN_FOR_ENTRY
IsRep_label:
        UPDATE_1 probs_state, 0, (IsRep - IsMatch)

        # The (checkDicSize == 0 && processedPos == 0) case was checked before in LzmaDec.c with kBadRepCode.
        # So we don't check it here.
        
        # mov     t0, processedPos
        # or      t0, checkDicSize
        # jz      fin_ERROR_2

        // state = state < kNumLitStates ? 8 : 11;
        cmp     state, kNumLitStates * PMULT
        mov     state, 8 * PMULT
        mov     probBranch, 11 * PMULT
        cmovae  state, probBranch

        SET_probs RepLenCoder
        
        IF_BIT_1 probs_state, 0, (IsRepG0 - IsMatch), IsRepG0_label
        sub_big  probs_state, probs_state, (IsMatch - IsRep0Long) << PSHIFT
        IF_BIT_0_NOUP probs_state, pbPos_R, 0, IsRep0Short_label
        UPDATE_1 probs_state, pbPos_R, 0
        jmp     len_decode

MY_ALIGN_FOR_ENTRY
IsRepG0_label:
        UPDATE_1 probs_state, 0, (IsRepG0 - IsMatch)
        IF_BIT_1 probs_state, 0, (IsRepG1 - IsMatch), IsRepG1_label
        mov     dist, rep1
        mov     rep1, rep0
        mov     rep0, dist
        jmp     len_decode
        
# MY_ALIGN_FOR_ENTRY
IsRepG1_label:
        UPDATE_1 probs_state, 0, (IsRepG1 - IsMatch)
        IF_BIT_1 probs_state, 0, (IsRepG2 - IsMatch), IsRepG2_label
        mov     dist, rep2
        mov     rep2, rep1
        mov     rep1, rep0
        mov     rep0, dist
        jmp     len_decode

# MY_ALIGN_FOR_ENTRY
IsRepG2_label:
        UPDATE_1 probs_state, 0, (IsRepG2 - IsMatch)
        mov     dist, rep3
        mov     rep3, rep2
        mov     rep2, rep1
        mov     rep1, rep0
        mov     rep0, dist
        jmp     len_decode

        

# ---------- SPEC SHORT DISTANCE ----------

MY_ALIGN_FOR_ENTRY
short_dist:
        p2_sub_s numBits, 32 + 1
        jbe     decode_dist_end
        or      sym, 2
        shl     sym, numBits
        add     sym_R, probs_Spec, sym_R, lsl #PSHIFT
        p2_add  sym_R, SpecPos * PMULT + 1 * PMULT
        mov     sym2, PMULT // # step
MY_ALIGN_FOR_LOOP
spec_loop:
        REV_1_VAR prob_reg
        dec_s   numBits
        jnz     spec_loop
        
        p2_add  sym2_R, probs_Spec
    .if SpecPos != 0
        p2_add  sym2_R, SpecPos * PMULT
    .endif
        p2_sub  sym_R, sym2_R
        shr     sym, PSHIFT
        
        jmp     decode_dist_end



# ---------- COPY MATCH 0 ----------
MY_ALIGN_FOR_ENTRY
copy_match_0:
    #ifdef LZMA_USE_4BYTES_FILL
        strb    sym, [dicPos, cnt_R]
        inc_s   cnt_R
        jz      copy_end
        
        strb    sym, [dicPos, cnt_R]
        inc_s   cnt_R
        jz      copy_end
        
        strb    sym, [dicPos, cnt_R]
        inc_s   cnt_R
        jz      copy_end
        
        orr     t3, sym, sym, lsl 8
        p2_and  cnt_R, -4
        orr     t3, t3, t3, lsl 16
MY_ALIGN_FOR_LOOP_16
1:
        /*
        str     t3, [dicPos, cnt_R]
        adds    cnt_R, cnt_R, 4
        jz      2f
        */

        str     t3, [dicPos, cnt_R]
        adds    cnt_R, cnt_R, 4
        jnz     1b
2:
        // p2_and  sym, 255
    #else

MY_ALIGN_FOR_LOOP
1:
        strb    sym, [dicPos, cnt_R]
        inc_s   cnt_R
        jz      copy_end

        strb    sym, [dicPos, cnt_R]
        inc_s   cnt_R
        jnz     1b
    #endif        

    jmp     copy_end


# ---------- COPY MATCH CROSS ----------
copy_match_cross:
        # t0_R  - src pos
        # cnt_R - total copy len

        p1_neg  cnt_R
1:
        ldrb    sym, [dic, t0_R]
        inc     t0_R
        strb    sym, [dicPos, cnt_R]
        inc     cnt_R
        cmp     t0_R, dicBufSize
        jne     1b
        
        ldrb    sym, [dic]
        sub     t0_R, dic, cnt_R
        jmp     copy_common




/*
fin_dicPos_LIMIT_REP_SHORT:
        mov     len, 1
        jmp     fin_OK
*/

/*
fin_dicPos_LIMIT:
        jmp     fin_OK
        # For more strict mode we can stop decoding with error
        # mov     sym, 1
        # jmp     fin
*/

fin_ERROR_MATCH_DIST:
        # rep0 = distance + 1;
        p2_add  len, kMatchSpecLen_Error_Data
        mov     rep3, rep2
        mov     rep2, rep1
        mov     rep1, rep0
        mov     rep0, sym
        STATE_UPDATE_FOR_MATCH
        # jmp     fin_OK
        mov     sym, 1
        jmp     fin

end_of_payload:
        inc_s   sym
        jnz     fin_ERROR_MATCH_DIST

        mov     len, kMatchSpecLenStart
        xor     state, (1 << FLAG_STATE_BITS)
        jmp     fin_OK

/*
fin_OK_lit:
        mov     len, wzr
*/

fin_OK:
        mov     sym, wzr

fin:
        NORM

    #define fin_lzma_reg  t0_R

   .macro STORE_LZMA_VAR reg:req, struct_offs:req
        str     \reg, [fin_lzma_reg, \struct_offs]
   .endm

   .macro STORE_LZMA_PAIR reg0:req, reg1:req, struct_offs:req
        stp     \reg0, \reg1, [fin_lzma_reg, \struct_offs]
   .endm

        ldr     fin_lzma_reg, [sp, 120]
        p2_sub  dicPos, dic
        shr     state, PSHIFT

        STORE_LZMA_PAIR   dicPos, buf,  offset_dicPos
        STORE_LZMA_PAIR   range, cod,   offset_range
        STORE_LZMA_VAR    processedPos, offset_processedPos
        STORE_LZMA_PAIR   rep0, rep1,   offset_rep0
        STORE_LZMA_PAIR   rep2, rep3,   offset_rep2
        STORE_LZMA_PAIR   state, len,   offset_state

        mov     w0, sym
        
	ldp	x29, x30, [sp, 80]
	ldp	x27, x28, [sp, 64]
	ldp	x25, x26, [sp, 48]
        ldp	x23, x24, [sp, 32]
	ldp	x21, x22, [sp, 16]
	ldp	x19, x20, [sp], 128

        ret
/*
	.cfi_endproc
.LFE0:
	.size	LzmaDec_DecodeReal_3, .-LzmaDec_DecodeReal_3
	.ident	"TAG_LZMA"
	.section	.note.GNU-stack,"",@progbits
*/        
