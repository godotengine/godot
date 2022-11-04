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
	EXPORT	oc_enc_frag_satd_neon
	EXPORT	oc_enc_frag_satd2_neon
	EXPORT	oc_enc_frag_intra_satd_neon

oc_enc_frag_satd_neon PROC
	; r0 = int                 *_dc
	; r1 = const unsigned char *_src
	; r2 = const unsigned char *_ref
	; r3 = int                  _ystride
	; Load src and subtract ref, expanding to 16 bits.
	VLD1.64		{D16},[r1@64],r3
	VLD1.64		{D0}, [r2],r3
	VSUBL.U8	Q8, D16,D0
	VLD1.64		{D18},[r1@64],r3
	VLD1.64		{D1}, [r2],r3
	VSUBL.U8	Q9, D18,D1
	VLD1.64		{D20},[r1@64],r3
	VLD1.64		{D2}, [r2],r3
	VSUBL.U8	Q10,D20,D2
	VLD1.64		{D22},[r1@64],r3
	VLD1.64		{D3}, [r2],r3
	VSUBL.U8	Q11,D22,D3
	VLD1.64		{D24},[r1@64],r3
	VLD1.64		{D4}, [r2],r3
	VSUBL.U8	Q12,D24,D4
	VLD1.64		{D26},[r1@64],r3
	VLD1.64		{D5}, [r2],r3
	VSUBL.U8	Q13,D26,D5
	VLD1.64		{D28},[r1@64],r3
	VLD1.64		{D6}, [r2],r3
	VSUBL.U8	Q14,D28,D6
	VLD1.64		{D30},[r1@64]
	VLD1.64		{D7}, [r2]
	VSUBL.U8	Q15,D30,D7
oc_int_frag_satd_neon
	; Hadamard Stage A
	VADD.I16	Q0, Q8, Q12
	VSUB.I16	Q12,Q8, Q12
	VSUB.I16	Q1, Q9, Q13
	VADD.I16	Q9, Q9, Q13
	VSUB.I16	Q2, Q10,Q14
	VADD.I16	Q10,Q10,Q14
	VADD.I16	Q3, Q11,Q15
	VSUB.I16	Q15,Q11,Q15
	; Hadamard Stage B
	VADD.I16	Q8, Q0, Q10
	VSUB.I16	Q0, Q0, Q10
	VSUB.I16	Q11,Q9, Q3
	VADD.I16	Q3, Q9, Q3
	VSUB.I16	Q14,Q12,Q2
	VADD.I16	Q2, Q12,Q2
	VADD.I16	Q13,Q1, Q15
	VSUB.I16	Q1, Q1, Q15
	; Hadamard Stage C & Start 8x8 Transpose
	VSUB.I16	Q9, Q8, Q3
	VADD.I16	Q8, Q8, Q3
	VTRN.16		Q8, Q9
	VADD.I16	Q10,Q0, Q11
	VSUB.I16	Q11,Q0, Q11
	VTRN.16		Q10,Q11
	VADD.I16	Q12,Q2, Q13
	VTRN.32		Q8, Q10
	VSUB.I16	Q13,Q2, Q13
	VTRN.32		Q9, Q11
	VSUB.I16	Q15,Q14,Q1
	VTRN.16		Q12,Q13
	VADD.I16	Q14,Q14,Q1
	VTRN.16		Q14,Q15
	VTRN.32		Q12,Q14
	VSWP		D17,D24
	; Hadamard Stage A & Finish 8x8 Transpose
	VADD.I16	Q0, Q8, Q12
	VTRN.32		Q13,Q15
	VSUB.I16	Q12,Q8, Q12
	VSWP		D19,D26
	VSUB.I16	Q1, Q9, Q13
	VSWP		D21,D28
	VADD.I16	Q9, Q9, Q13
	VSWP		D23,D30
	VSUB.I16	Q2, Q10,Q14
	VADD.I16	Q10,Q10,Q14
	VADD.I16	Q3, Q11,Q15
	VSUB.I16	Q15,Q11,Q15
	; Hadamard Stage B
	VADD.I16	Q8, Q0, Q10
	VSUB.I16	Q0, Q0, Q10
	VSUB.I16	Q11,Q9, Q3
	VADD.I16	Q3, Q9, Q3
	VSUB.I16	Q14,Q12,Q2
	VADD.I16	Q2, Q12,Q2
	VADD.I16	Q13,Q1, Q15
	VSUB.I16	Q1, Q1, Q15
	; Hadamard Stage C & abs & accum
	VNEG.S16	Q9, Q3
	; Compute the (signed) DC component and save it off.
	VADDL.S16	Q10,D16,D6
	VABD.S16	Q12,Q8, Q9
	VABD.S16	Q15,Q11,Q0
	VST1.32		D20[0],[r0]
	; Remove the (abs) DC component from the total.
	MOV	r3,#0
	VMOV.I16	D24[0],r3
	VABA.S16	Q12,Q13,Q2
	VABA.S16	Q15,Q14,Q1
	VNEG.S16	Q0, Q0
	VNEG.S16	Q2, Q2
	VNEG.S16	Q1, Q1
	VABA.S16	Q12,Q8, Q3
	VABA.S16	Q15,Q11,Q0
	VABA.S16	Q12,Q13,Q2
	VABA.S16	Q15,Q14,Q1
	; We're now using all 16 bits of each value.
	VPADDL.U16	Q12,Q12
	VPADAL.U16	Q12,Q15
	VADD.U32	D24,D24,D25
	VPADDL.U32	D24,D24
	VMOV.U32	r0, D24[0]
	MOV	PC, r14
	ENDP

oc_enc_frag_satd2_neon PROC
	; r0 = int                 *_dc
	; r1 = const unsigned char *_src
	; r2 = const unsigned char *_ref1
	; r3 = const unsigned char *_ref2
	; r12= int                  _ystride
	LDR	r12,[r13]
	; Load src and subtract (ref1+ref2>>1), expanding to 16 bits.
	VLD1.64		{D0}, [r2],r12
	VLD1.64		{D1}, [r3],r12
	VLD1.64		{D16},[r1@64],r12
	VHADD.U8	D0, D0, D1
	VLD1.64		{D2}, [r2],r12
	VLD1.64		{D3}, [r3],r12
	VSUBL.U8	Q8, D16,D0
	VLD1.64		{D18},[r1@64],r12
	VHADD.U8	D2, D2, D3
	VLD1.64		{D4}, [r2],r12
	VLD1.64		{D5}, [r3],r12
	VSUBL.U8	Q9, D18,D2
	VLD1.64		{D20},[r1@64],r12
	VHADD.U8	D4, D4, D5
	VLD1.64		{D6}, [r2],r12
	VLD1.64		{D7}, [r3],r12
	VSUBL.U8	Q10,D20,D4
	VLD1.64		{D22},[r1@64],r12
	VHADD.U8	D6, D6, D7
	VLD1.64		{D0}, [r2],r12
	VLD1.64		{D1}, [r3],r12
	VSUBL.U8	Q11,D22,D6
	VLD1.64		{D24},[r1@64],r12
	VHADD.U8	D0, D0, D1
	VLD1.64		{D2}, [r2],r12
	VLD1.64		{D3}, [r3],r12
	VSUBL.U8	Q12,D24,D0
	VLD1.64		{D26},[r1@64],r12
	VHADD.U8	D2, D2, D3
	VLD1.64		{D4}, [r2],r12
	VLD1.64		{D5}, [r3],r12
	VSUBL.U8	Q13,D26,D2
	VLD1.64		{D28},[r1@64],r12
	VHADD.U8	D4, D4, D5
	VLD1.64		{D6}, [r2]
	VSUBL.U8	Q14,D28,D4
	VLD1.64		{D7}, [r3]
	VHADD.U8	D6, D6, D7
	VLD1.64		{D30},[r1@64]
	VSUBL.U8	Q15,D30,D6
	B	oc_int_frag_satd_neon
	ENDP

oc_enc_frag_intra_satd_neon PROC
	; r0 = int                 *_dc
	; r1 = const unsigned char *_src
	; r2 = int                  _ystride
	; Load and subtract 128 from src, expanding to 16 bits.
	VMOV.I8		D0,#128
	VLD1.64		{D16},[r1@64],r2
	VSUBL.U8	Q8, D16,D0
	VLD1.64		{D18},[r1@64],r2
	VSUBL.U8	Q9, D18,D0
	VLD1.64		{D20},[r1@64],r2
	VSUBL.U8	Q10,D20,D0
	VLD1.64		{D22},[r1@64],r2
	VSUBL.U8	Q11,D22,D0
	VLD1.64		{D24},[r1@64],r2
	VSUBL.U8	Q12,D24,D0
	VLD1.64		{D26},[r1@64],r2
	VSUBL.U8	Q13,D26,D0
	VLD1.64		{D28},[r1@64],r2
	VSUBL.U8	Q14,D28,D0
	VLD1.64		{D30},[r1@64]
	VSUBL.U8	Q15,D30,D0
	B	oc_int_frag_satd_neon
	ENDP
 ]

	END
