;********************************************************************
;*                                                                  *
;* THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
;* USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
;* GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
;* IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
;*                                                                  *
;* THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
;* by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
;*                                                                  *
;********************************************************************
;
; function:
;   last mod: $Id: mmxstate.c 17247 2010-05-28 05:35:32Z tterribe $
;
;********************************************************************

	AREA	|.text|, CODE, READONLY

	GET	armopts.s

 [ OC_ARM_ASM_NEON
	EXPORT	oc_enc_enquant_table_init_neon
	EXPORT	oc_enc_enquant_table_fixup_neon
	EXPORT	oc_enc_quantize_neon

oc_enc_enquant_table_init_neon PROC
	; r0 = void               *_enquant
	; r1 = const ogg_uint16_t  _dequant[64]
	STMFD r13!,{r0,r14}
	; Initialize the table using the C routine
	BLX	oc_enc_enquant_table_init_c
	LDR	r0, [r13],#4
	MOV	r1, #2
	; Now partially de-interleave it, so that the first row is all
	;  multipliers, the second row is all shift factors, etc.
	; Also, negate the shifts for use by VSHL.
oeeti_neon_lp
	SUBS	r1, r1, #1
	VLDMIA		r0, {D16-D31}
	VUZP.16		Q8, Q9
	VNEG.S16	Q9, Q9
	VUZP.16		Q10,Q11
	VNEG.S16	Q11,Q11
	VUZP.16		Q12,Q13
	VNEG.S16	Q13,Q13
	VUZP.16		Q14,Q15
	VNEG.S16	Q15,Q15
	VSTMIA		r0!,{D16-D31}
	BNE	oeeti_neon_lp
	LDR	PC, [r13],#4
	ENDP

oc_enc_enquant_table_fixup_neon PROC
	; r0 = void *_enquant[3][3][2]
	; r1 = int   _nqis
	STR	r14,[r13,#-4]!
oeetf_neon_lp1
	SUBS	r1, r1, #1
	BEQ	oeetf_neon_end1
	MOV	r14,#3
oeetf_neon_lp2
	LDR	r2, [r0]
	SUBS	r14,r14,#1
	LDRH	r3, [r2]
	LDRH	r12,[r2,#16]
	LDR	r2, [r0,#8]
	STRH	r3, [r2]
	STRH	r12,[r2,#16]
	LDR	r2, [r0,#4]
	LDRH	r3, [r2]
	LDRH	r12,[r2,#16]
	LDR	r2, [r0,#12]
	ADD	r0, r0, #24
	STRH	r3, [r2]
	STRH	r12,[r2,#16]
	BNE	oeetf_neon_lp2
	SUB	r0, r0, #64
	B	oeetf_neon_lp1
oeetf_neon_end1
	LDR	PC, [r13],#4
	ENDP

oc_enc_quantize_neon PROC
	; r0 = ogg_int16_t        _qdct[64]
	; r1 = const ogg_int16_t  _dct[64]
	; r2 = const ogg_int16_t  _dequant[64]
	; r3 = const void        *_enquant
	STMFD	r13!,{r4,r5,r14}
	; The loop counter goes in the high half of r14.
	MOV	r14,#0xFFFCFFFF
oeq_neon_lp
	; Load the next two rows of the data and the quant matrices.
	VLD1.64		{D16,D17,D18,D19},[r1@128]!
	VLD1.64		{D20,D21,D22,D23},[r2@128]!
	; Add in the signed rounding bias from the quantizers.
	; Note that the VHADD relies on the fact that the quantizers are all
	;  even (they're in fact multiples of four) in order to round correctly
	;  on the entries being negated.
	VSHR.S16	Q0, Q8, #15
	VSHR.S16	Q1, Q9, #15
	VLD1.64		{D24,D25,D26,D27},[r3@128]!
	VHADD.S16	Q10,Q0, Q10
	VHADD.S16	Q11,Q1, Q11
	VLD1.64		{D28,D29,D30,D31},[r3@128]!
	ADDS	r14,r14,#1<<16
	VEOR.S16	Q10,Q0, Q10
	VEOR.S16	Q11,Q1, Q11
	VADD.S16	Q8, Q8, Q10
	VADD.S16	Q9, Q9, Q11
	; Perform the actual division and save the result.
	VQDMULH.S16	Q12,Q8, Q12
	VQDMULH.S16	Q14,Q9, Q14
	VADD.S16	Q8, Q8, Q8
	VADD.S16	Q9, Q9, Q9
	VADD.S16	Q8, Q8, Q12
	VADD.S16	Q9, Q9, Q14
	VSHL.S16	Q8, Q13
	VSHL.S16	Q9, Q15
	VSUB.S16	Q8, Q8, Q0
	VSUB.S16	Q9, Q9, Q1
	VST1.64		{D16,D17,D18,D19},[r0@128]!
	; Now pull out a bitfield marking the non-zero coefficients.
	VQMOVN.S16	D16,Q8
	VQMOVN.S16	D17,Q9
	VCEQ.S8		Q8, #0
	; Sadly, NEON has no PMOVMSKB; emulating it requires 6 instructions.
	VNEG.S8		Q8, Q8          ; D16=.......3.......2.......1.......0
	                                ;     .......7.......6.......5.......4
	                                ; D17=.......B.......A.......9.......8
	                                ;     .......F.......E.......D.......C
	VZIP.8		D16,D17         ; D16=.......9.......1.......8.......0
	                                ;     .......B.......3.......A.......2
	                                ; D17=.......D.......5.......C.......4
	                                ;     .......F.......7.......E.......6
	VSLI.8		D16,D17,#4      ; D16=...D...9...5...1...C...8...4...0
	                                ;     ...F...B...7...3...E...A...6...2
	; Shift over the bitfields from previous iterations and
	;  finish compacting the bitfield from the last iteration.
	ORR	r4, r4, r5, LSL #2      ; r4 =.F.D.B.9.7.5.3.1.E.C.A.8.6.4.2.0
	ORR	r4, r4, r4, LSR #15     ; r4 =.F.D.B.9.7.5.3.1FEDCBA9876543210
	PKHTB	r14,r14,r12,ASR #16     ; r14=i|A
	PKHBT	r12,r4, r12,LSL #16     ; r12=B|C
	VMOV		r4, r5, D16
	BLT	oeq_neon_lp
	; Start with the low half while the NEON register transfers.
	PKHBT	r0, r14,r12             ; r0 =B|A
	MVNS	r0, r0
	CLZNE	r0, r0
	RSBNE	r0, r0, #31
	; Stall 8-10 more cycles waiting for the last transfer.
	ORR	r4, r4, r5, LSL #2      ; r4 =.F.D.B.9.7.5.3.1.E.C.A.8.6.4.2.0
	ORR	r4, r4, r4, LSR #15     ; r4 =.F.D.B.9.7.5.3.1FEDCBA9876543210
	PKHBT	r1, r12,r4, LSL #16     ; r1 = D|C
	MVNS	r1, r1
	CLZNE	r1, r1
	RSBNE	r0, r1, #63
	LDMFD	r13!,{r4,r5,PC}
	ENDP
 ]

	END
