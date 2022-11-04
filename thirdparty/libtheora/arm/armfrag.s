;********************************************************************
;*                                                                  *
;* THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
;* USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
;* GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
;* IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
;*                                                                  *
;* THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2010                *
;* by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
;*                                                                  *
;********************************************************************
; Original implementation:
;  Copyright (C) 2009 Robin Watts for Pinknoise Productions Ltd
; last mod: $Id$
;********************************************************************

	AREA	|.text|, CODE, READONLY

	GET	armopts.s

; Vanilla ARM v4 versions
	EXPORT	oc_frag_copy_list_arm
	EXPORT	oc_frag_recon_intra_arm
	EXPORT	oc_frag_recon_inter_arm
	EXPORT	oc_frag_recon_inter2_arm

oc_frag_copy_list_arm PROC
	; r0 = _dst_frame
	; r1 = _src_frame
	; r2 = _ystride
	; r3 = _fragis
	; <> = _nfragis
	; <> = _frag_buf_offs
	LDR	r12,[r13]		; r12 = _nfragis
	STMFD	r13!,{r4-r6,r11,r14}
	SUBS	r12, r12, #1
	LDR	r4,[r3],#4		; r4 = _fragis[fragii]
	LDRGE	r14,[r13,#4*6]		; r14 = _frag_buf_offs
	BLT	ofcl_arm_end
	SUB	r2, r2, #4
ofcl_arm_lp
	LDR	r11,[r14,r4,LSL #2]	; r11 = _frag_buf_offs[_fragis[fragii]]
	SUBS	r12, r12, #1
	; Stall (on XScale)
	ADD	r4, r1, r11		; r4 = _src_frame+frag_buf_off
	LDR	r6, [r4], #4
	ADD	r11,r0, r11		; r11 = _dst_frame+frag_buf_off
	LDR	r5, [r4], r2
	STR	r6, [r11],#4
	LDR	r6, [r4], #4
	STR	r5, [r11],r2
	LDR	r5, [r4], r2
	STR	r6, [r11],#4
	LDR	r6, [r4], #4
	STR	r5, [r11],r2
	LDR	r5, [r4], r2
	STR	r6, [r11],#4
	LDR	r6, [r4], #4
	STR	r5, [r11],r2
	LDR	r5, [r4], r2
	STR	r6, [r11],#4
	LDR	r6, [r4], #4
	STR	r5, [r11],r2
	LDR	r5, [r4], r2
	STR	r6, [r11],#4
	LDR	r6, [r4], #4
	STR	r5, [r11],r2
	LDR	r5, [r4], r2
	STR	r6, [r11],#4
	LDR	r6, [r4], #4
	STR	r5, [r11],r2
	LDR	r5, [r4], r2
	STR	r6, [r11],#4
	LDR	r6, [r4], #4
	STR	r5, [r11],r2
	LDR	r5, [r4]
	LDRGE	r4,[r3],#4		; r4 = _fragis[fragii]
	STR	r6, [r11],#4
	STR	r5, [r11]
	BGE	ofcl_arm_lp
ofcl_arm_end
	LDMFD	r13!,{r4-r6,r11,PC}
oc_frag_recon_intra_arm
	; r0 =       unsigned char *_dst
	; r1 =       int            _ystride
	; r2 = const ogg_int16_t    _residue[64]
	STMFD	r13!,{r4,r5,r14}
	MOV	r14,#8
	MOV	r5, #255
	SUB	r1, r1, #7
ofrintra_lp_arm
	LDRSH	r3, [r2], #2
	LDRSH	r4, [r2], #2
	LDRSH	r12,[r2], #2
	ADDS	r3, r3, #128
	CMPGT	r5, r3
	EORLT	r3, r5, r3, ASR #32
	STRB	r3, [r0], #1
	ADDS	r4, r4, #128
	CMPGT	r5, r4
	EORLT	r4, r5, r4, ASR #32
	LDRSH	r3, [r2], #2
	STRB	r4, [r0], #1
	ADDS	r12,r12,#128
	CMPGT	r5, r12
	EORLT	r12,r5, r12,ASR #32
	LDRSH	r4, [r2], #2
	STRB	r12,[r0], #1
	ADDS	r3, r3, #128
	CMPGT	r5, r3
	EORLT	r3, r5, r3, ASR #32
	LDRSH	r12,[r2], #2
	STRB	r3, [r0], #1
	ADDS	r4, r4, #128
	CMPGT	r5, r4
	EORLT	r4, r5, r4, ASR #32
	LDRSH	r3, [r2], #2
	STRB	r4, [r0], #1
	ADDS	r12,r12,#128
	CMPGT	r5, r12
	EORLT	r12,r5, r12,ASR #32
	LDRSH	r4, [r2], #2
	STRB	r12,[r0], #1
	ADDS	r3, r3, #128
	CMPGT	r5, r3
	EORLT	r3, r5, r3, ASR #32
	STRB	r3, [r0], #1
	ADDS	r4, r4, #128
	CMPGT	r5, r4
	EORLT	r4, r5, r4, ASR #32
	STRB	r4, [r0], r1
	SUBS	r14,r14,#1
	BGT	ofrintra_lp_arm
	LDMFD	r13!,{r4,r5,PC}
	ENDP

oc_frag_recon_inter_arm PROC
	; r0 =       unsigned char *dst
	; r1 = const unsigned char *src
	; r2 =       int            ystride
	; r3 = const ogg_int16_t    residue[64]
	STMFD	r13!,{r5,r9-r11,r14}
	MOV	r9, #8
	MOV	r5, #255
	SUB	r2, r2, #7
ofrinter_lp_arm
	LDRSH	r12,[r3], #2
	LDRB	r14,[r1], #1
	LDRSH	r11,[r3], #2
	LDRB	r10,[r1], #1
	ADDS	r12,r12,r14
	CMPGT	r5, r12
	EORLT	r12,r5, r12,ASR #32
	STRB	r12,[r0], #1
	ADDS	r11,r11,r10
	CMPGT	r5, r11
	LDRSH	r12,[r3], #2
	LDRB	r14,[r1], #1
	EORLT	r11,r5, r11,ASR #32
	STRB	r11,[r0], #1
	ADDS	r12,r12,r14
	CMPGT	r5, r12
	LDRSH	r11,[r3], #2
	LDRB	r10,[r1], #1
	EORLT	r12,r5, r12,ASR #32
	STRB	r12,[r0], #1
	ADDS	r11,r11,r10
	CMPGT	r5, r11
	LDRSH	r12,[r3], #2
	LDRB	r14,[r1], #1
	EORLT	r11,r5, r11,ASR #32
	STRB	r11,[r0], #1
	ADDS	r12,r12,r14
	CMPGT	r5, r12
	LDRSH	r11,[r3], #2
	LDRB	r10,[r1], #1
	EORLT	r12,r5, r12,ASR #32
	STRB	r12,[r0], #1
	ADDS	r11,r11,r10
	CMPGT	r5, r11
	LDRSH	r12,[r3], #2
	LDRB	r14,[r1], #1
	EORLT	r11,r5, r11,ASR #32
	STRB	r11,[r0], #1
	ADDS	r12,r12,r14
	CMPGT	r5, r12
	LDRSH	r11,[r3], #2
	LDRB	r10,[r1], r2
	EORLT	r12,r5, r12,ASR #32
	STRB	r12,[r0], #1
	ADDS	r11,r11,r10
	CMPGT	r5, r11
	EORLT	r11,r5, r11,ASR #32
	STRB	r11,[r0], r2
	SUBS	r9, r9, #1
	BGT	ofrinter_lp_arm
	LDMFD	r13!,{r5,r9-r11,PC}
	ENDP

oc_frag_recon_inter2_arm PROC
	; r0 =       unsigned char *dst
	; r1 = const unsigned char *src1
	; r2 = const unsigned char *src2
	; r3 =       int            ystride
	LDR	r12,[r13]
	; r12= const ogg_int16_t    residue[64]
	STMFD	r13!,{r4-r8,r14}
	MOV	r14,#8
	MOV	r8, #255
	SUB	r3, r3, #7
ofrinter2_lp_arm
	LDRB	r5, [r1], #1
	LDRB	r6, [r2], #1
	LDRSH	r4, [r12],#2
	LDRB	r7, [r1], #1
	ADD	r5, r5, r6
	ADDS	r5, r4, r5, LSR #1
	CMPGT	r8, r5
	LDRB	r6, [r2], #1
	LDRSH	r4, [r12],#2
	EORLT	r5, r8, r5, ASR #32
	STRB	r5, [r0], #1
	ADD	r7, r7, r6
	ADDS	r7, r4, r7, LSR #1
	CMPGT	r8, r7
	LDRB	r5, [r1], #1
	LDRB	r6, [r2], #1
	LDRSH	r4, [r12],#2
	EORLT	r7, r8, r7, ASR #32
	STRB	r7, [r0], #1
	ADD	r5, r5, r6
	ADDS	r5, r4, r5, LSR #1
	CMPGT	r8, r5
	LDRB	r7, [r1], #1
	LDRB	r6, [r2], #1
	LDRSH	r4, [r12],#2
	EORLT	r5, r8, r5, ASR #32
	STRB	r5, [r0], #1
	ADD	r7, r7, r6
	ADDS	r7, r4, r7, LSR #1
	CMPGT	r8, r7
	LDRB	r5, [r1], #1
	LDRB	r6, [r2], #1
	LDRSH	r4, [r12],#2
	EORLT	r7, r8, r7, ASR #32
	STRB	r7, [r0], #1
	ADD	r5, r5, r6
	ADDS	r5, r4, r5, LSR #1
	CMPGT	r8, r5
	LDRB	r7, [r1], #1
	LDRB	r6, [r2], #1
	LDRSH	r4, [r12],#2
	EORLT	r5, r8, r5, ASR #32
	STRB	r5, [r0], #1
	ADD	r7, r7, r6
	ADDS	r7, r4, r7, LSR #1
	CMPGT	r8, r7
	LDRB	r5, [r1], #1
	LDRB	r6, [r2], #1
	LDRSH	r4, [r12],#2
	EORLT	r7, r8, r7, ASR #32
	STRB	r7, [r0], #1
	ADD	r5, r5, r6
	ADDS	r5, r4, r5, LSR #1
	CMPGT	r8, r5
	LDRB	r7, [r1], r3
	LDRB	r6, [r2], r3
	LDRSH	r4, [r12],#2
	EORLT	r5, r8, r5, ASR #32
	STRB	r5, [r0], #1
	ADD	r7, r7, r6
	ADDS	r7, r4, r7, LSR #1
	CMPGT	r8, r7
	EORLT	r7, r8, r7, ASR #32
	STRB	r7, [r0], r3
	SUBS	r14,r14,#1
	BGT	ofrinter2_lp_arm
	LDMFD	r13!,{r4-r8,PC}
	ENDP

 [ OC_ARM_ASM_EDSP
	EXPORT	oc_frag_copy_list_edsp

oc_frag_copy_list_edsp PROC
	; r0 = _dst_frame
	; r1 = _src_frame
	; r2 = _ystride
	; r3 = _fragis
	; <> = _nfragis
	; <> = _frag_buf_offs
	LDR	r12,[r13]		; r12 = _nfragis
	STMFD	r13!,{r4-r11,r14}
	SUBS	r12, r12, #1
	LDRGE	r5, [r3],#4		; r5 = _fragis[fragii]
	LDRGE	r14,[r13,#4*10]		; r14 = _frag_buf_offs
	BLT	ofcl_edsp_end
ofcl_edsp_lp
	MOV	r4, r1
	LDR	r5, [r14,r5, LSL #2]	; r5 = _frag_buf_offs[_fragis[fragii]]
	SUBS	r12, r12, #1
	; Stall (on XScale)
	LDRD	r6, [r4, r5]!		; r4 = _src_frame+frag_buf_off
	LDRD	r8, [r4, r2]!
	; Stall
	STRD	r6, [r5, r0]!		; r5 = _dst_frame+frag_buf_off
	STRD	r8, [r5, r2]!
	; Stall
	LDRD	r6, [r4, r2]!	; On Xscale at least, doing 3 consecutive
	LDRD	r8, [r4, r2]!	; loads causes a stall, but that's no worse
	LDRD	r10,[r4, r2]!	; than us only doing 2, and having to do
				; another pair of LDRD/STRD later on.
	; Stall
	STRD	r6, [r5, r2]!
	STRD	r8, [r5, r2]!
	STRD	r10,[r5, r2]!
	LDRD	r6, [r4, r2]!
	LDRD	r8, [r4, r2]!
	LDRD	r10,[r4, r2]!
	STRD	r6, [r5, r2]!
	STRD	r8, [r5, r2]!
	STRD	r10,[r5, r2]!
	LDRGE	r5, [r3],#4		; r5 = _fragis[fragii]
	BGE	ofcl_edsp_lp
ofcl_edsp_end
	LDMFD	r13!,{r4-r11,PC}
	ENDP
 ]

 [ OC_ARM_ASM_MEDIA
	EXPORT	oc_frag_recon_intra_v6
	EXPORT	oc_frag_recon_inter_v6
	EXPORT	oc_frag_recon_inter2_v6

oc_frag_recon_intra_v6 PROC
	; r0 =       unsigned char *_dst
	; r1 =       int            _ystride
	; r2 = const ogg_int16_t    _residue[64]
	STMFD	r13!,{r4-r6,r14}
	MOV	r14,#8
	MOV	r12,r2
	LDR	r6, =0x00800080
ofrintra_v6_lp
	LDRD	r2, [r12],#8	; r2 = 11110000 r3 = 33332222
	LDRD	r4, [r12],#8	; r4 = 55554444 r5 = 77776666
	SUBS	r14,r14,#1
	QADD16	r2, r2, r6
	QADD16	r3, r3, r6
	QADD16	r4, r4, r6
	QADD16	r5, r5, r6
	USAT16	r2, #8, r2		; r2 = __11__00
	USAT16	r3, #8, r3		; r3 = __33__22
	USAT16	r4, #8, r4		; r4 = __55__44
	USAT16	r5, #8, r5		; r5 = __77__66
	ORR	r2, r2, r2, LSR #8	; r2 = __111100
	ORR	r3, r3, r3, LSR #8	; r3 = __333322
	ORR	r4, r4, r4, LSR #8	; r4 = __555544
	ORR	r5, r5, r5, LSR #8	; r5 = __777766
	PKHBT   r2, r2, r3, LSL #16     ; r2 = 33221100
	PKHBT   r3, r4, r5, LSL #16     ; r3 = 77665544
	STRD	r2, [r0], r1
	BGT	ofrintra_v6_lp
	LDMFD	r13!,{r4-r6,PC}
	ENDP

oc_frag_recon_inter_v6 PROC
	; r0 =       unsigned char *_dst
	; r1 = const unsigned char *_src
	; r2 =       int            _ystride
	; r3 = const ogg_int16_t    _residue[64]
	STMFD	r13!,{r4-r7,r14}
	MOV	r14,#8
ofrinter_v6_lp
	LDRD	r6, [r3], #8		; r6 = 11110000 r7 = 33332222
	SUBS	r14,r14,#1
 [ OC_ARM_CAN_UNALIGN_LDRD
	LDRD	r4, [r1], r2	; Unaligned ; r4 = 33221100 r5 = 77665544
 |
	LDR	r5, [r1, #4]
	LDR	r4, [r1], r2
 ]
	PKHBT	r12,r6, r7, LSL #16	; r12= 22220000
	PKHTB	r7, r7, r6, ASR #16	; r7 = 33331111
	UXTB16	r6,r4			; r6 = __22__00
	UXTB16	r4,r4, ROR #8		; r4 = __33__11
	QADD16	r12,r12,r6		; r12= xx22xx00
	QADD16	r4, r7, r4		; r4 = xx33xx11
	LDRD	r6, [r3], #8		; r6 = 55554444 r7 = 77776666
	USAT16	r4, #8, r4		; r4 = __33__11
	USAT16	r12,#8,r12		; r12= __22__00
	ORR	r4, r12,r4, LSL #8	; r4 = 33221100
	PKHBT	r12,r6, r7, LSL #16	; r12= 66664444
	PKHTB	r7, r7, r6, ASR #16	; r7 = 77775555
	UXTB16	r6,r5			; r6 = __66__44
	UXTB16	r5,r5, ROR #8		; r5 = __77__55
	QADD16	r12,r12,r6		; r12= xx66xx44
	QADD16	r5, r7, r5		; r5 = xx77xx55
	USAT16	r12,#8, r12		; r12= __66__44
	USAT16	r5, #8, r5		; r4 = __77__55
	ORR	r5, r12,r5, LSL #8	; r5 = 33221100
	STRD	r4, [r0], r2
	BGT	ofrinter_v6_lp
	LDMFD	r13!,{r4-r7,PC}
	ENDP

oc_frag_recon_inter2_v6 PROC
	; r0 =       unsigned char *_dst
	; r1 = const unsigned char *_src1
	; r2 = const unsigned char *_src2
	; r3 =       int            _ystride
	LDR	r12,[r13]
	; r12= const ogg_int16_t    _residue[64]
	STMFD	r13!,{r4-r9,r14}
	MOV	r14,#8
ofrinter2_v6_lp
	LDRD	r6, [r12,#8]	; r6 = 55554444 r7 = 77776666
	SUBS	r14,r14,#1
	LDR	r4, [r1, #4]	; Unaligned	; r4 = src1[1] = 77665544
	LDR	r5, [r2, #4]	; Unaligned	; r5 = src2[1] = 77665544
	PKHBT	r8, r6, r7, LSL #16	; r8 = 66664444
	PKHTB	r9, r7, r6, ASR #16	; r9 = 77775555
	UHADD8	r4, r4, r5	; r4 = (src1[7,6,5,4] + src2[7,6,5,4])>>1
	UXTB16	r5, r4			; r5 = __66__44
	UXTB16	r4, r4, ROR #8		; r4 = __77__55
	QADD16	r8, r8, r5		; r8 = xx66xx44
	QADD16	r9, r9, r4		; r9 = xx77xx55
	LDRD	r6,[r12],#16	; r6 = 33332222 r7 = 11110000
	USAT16	r8, #8, r8		; r8 = __66__44
	LDR	r4, [r1], r3	; Unaligned	; r4 = src1[0] = 33221100
	USAT16	r9, #8, r9		; r9 = __77__55
	LDR	r5, [r2], r3	; Unaligned	; r5 = src2[0] = 33221100
	ORR	r9, r8, r9, LSL #8	; r9 = 77665544
	PKHBT	r8, r6, r7, LSL #16	; r8 = 22220000
	UHADD8	r4, r4, r5	; r4 = (src1[3,2,1,0] + src2[3,2,1,0])>>1
	PKHTB	r7, r7, r6, ASR #16	; r7 = 33331111
	UXTB16	r5, r4			; r5 = __22__00
	UXTB16	r4, r4, ROR #8		; r4 = __33__11
	QADD16	r8, r8, r5		; r8 = xx22xx00
	QADD16	r7, r7, r4		; r7 = xx33xx11
	USAT16	r8, #8, r8		; r8 = __22__00
	USAT16	r7, #8, r7		; r7 = __33__11
	ORR	r8, r8, r7, LSL #8	; r8 = 33221100
	STRD	r8, [r0], r3
	BGT	ofrinter2_v6_lp
	LDMFD	r13!,{r4-r9,PC}
	ENDP
 ]

 [ OC_ARM_ASM_NEON
	EXPORT	oc_frag_copy_list_neon
	EXPORT	oc_frag_recon_intra_neon
	EXPORT	oc_frag_recon_inter_neon
	EXPORT	oc_frag_recon_inter2_neon

oc_frag_copy_list_neon PROC
	; r0 = _dst_frame
	; r1 = _src_frame
	; r2 = _ystride
	; r3 = _fragis
	; <> = _nfragis
	; <> = _frag_buf_offs
	LDR	r12,[r13]		; r12 = _nfragis
	STMFD	r13!,{r4-r7,r14}
	CMP	r12, #1
	LDRGE	r6, [r3]		; r6 = _fragis[fragii]
	LDRGE	r14,[r13,#4*6]		; r14 = _frag_buf_offs
	BLT	ofcl_neon_end
	; Stall (2 on Xscale)
	LDR	r6, [r14,r6, LSL #2]	; r6 = _frag_buf_offs[_fragis[fragii]]
	; Stall (on XScale)
	MOV	r7, r6			; Guarantee PLD points somewhere valid.
ofcl_neon_lp
	ADD	r4, r1, r6
	VLD1.64	{D0}, [r4@64], r2
	ADD	r5, r0, r6
	VLD1.64	{D1}, [r4@64], r2
	SUBS	r12, r12, #1
	VLD1.64	{D2}, [r4@64], r2
	LDRGT	r6, [r3,#4]!		; r6 = _fragis[fragii]
	VLD1.64	{D3}, [r4@64], r2
	LDRGT	r6, [r14,r6, LSL #2]	; r6 = _frag_buf_offs[_fragis[fragii]]
	VLD1.64	{D4}, [r4@64], r2
	ADDGT	r7, r1, r6
	VLD1.64	{D5}, [r4@64], r2
	PLD	[r7]
	VLD1.64	{D6}, [r4@64], r2
	PLD	[r7, r2]
	VLD1.64	{D7}, [r4@64]
	PLD	[r7, r2, LSL #1]
	VST1.64	{D0}, [r5@64], r2
	ADDGT	r7, r7, r2, LSL #2
	VST1.64	{D1}, [r5@64], r2
	PLD	[r7, -r2]
	VST1.64	{D2}, [r5@64], r2
	PLD	[r7]
	VST1.64	{D3}, [r5@64], r2
	PLD	[r7, r2]
	VST1.64	{D4}, [r5@64], r2
	PLD	[r7, r2, LSL #1]
	VST1.64	{D5}, [r5@64], r2
	ADDGT	r7, r7, r2, LSL #2
	VST1.64	{D6}, [r5@64], r2
	PLD	[r7, -r2]
	VST1.64	{D7}, [r5@64]
	BGT	ofcl_neon_lp
ofcl_neon_end
	LDMFD	r13!,{r4-r7,PC}
	ENDP

oc_frag_recon_intra_neon PROC
	; r0 =       unsigned char *_dst
	; r1 =       int            _ystride
	; r2 = const ogg_int16_t    _residue[64]
	VMOV.I16	Q0, #128
	VLDMIA	r2,  {D16-D31}	; D16= 3333222211110000 etc	; 9(8) cycles
	VQADD.S16	Q8, Q8, Q0
	VQADD.S16	Q9, Q9, Q0
	VQADD.S16	Q10,Q10,Q0
	VQADD.S16	Q11,Q11,Q0
	VQADD.S16	Q12,Q12,Q0
	VQADD.S16	Q13,Q13,Q0
	VQADD.S16	Q14,Q14,Q0
	VQADD.S16	Q15,Q15,Q0
	VQMOVUN.S16	D16,Q8	; D16= 7766554433221100		; 1 cycle
	VQMOVUN.S16	D17,Q9	; D17= FFEEDDCCBBAA9988		; 1 cycle
	VQMOVUN.S16	D18,Q10	; D18= NNMMLLKKJJIIHHGG		; 1 cycle
	VST1.64	{D16},[r0@64], r1
	VQMOVUN.S16	D19,Q11	; D19= VVUUTTSSRRQQPPOO		; 1 cycle
	VST1.64	{D17},[r0@64], r1
	VQMOVUN.S16	D20,Q12	; D20= ddccbbaaZZYYXXWW		; 1 cycle
	VST1.64	{D18},[r0@64], r1
	VQMOVUN.S16	D21,Q13	; D21= llkkjjiihhggffee		; 1 cycle
	VST1.64	{D19},[r0@64], r1
	VQMOVUN.S16	D22,Q14	; D22= ttssrrqqppoonnmm		; 1 cycle
	VST1.64	{D20},[r0@64], r1
	VQMOVUN.S16	D23,Q15	; D23= !!@@zzyyxxwwvvuu		; 1 cycle
	VST1.64	{D21},[r0@64], r1
	VST1.64	{D22},[r0@64], r1
	VST1.64	{D23},[r0@64], r1
	MOV	PC,R14
	ENDP

oc_frag_recon_inter_neon PROC
	; r0 =       unsigned char *_dst
	; r1 = const unsigned char *_src
	; r2 =       int            _ystride
	; r3 = const ogg_int16_t    _residue[64]
	VLDMIA	r3, {D16-D31}	; D16= 3333222211110000 etc	; 9(8) cycles
	VLD1.64	{D0}, [r1], r2
	VLD1.64	{D2}, [r1], r2
	VMOVL.U8	Q0, D0	; Q0 = __77__66__55__44__33__22__11__00
	VLD1.64	{D4}, [r1], r2
	VMOVL.U8	Q1, D2	; etc
	VLD1.64	{D6}, [r1], r2
	VMOVL.U8	Q2, D4
	VMOVL.U8	Q3, D6
	VQADD.S16	Q8, Q8, Q0
	VLD1.64	{D0}, [r1], r2
	VQADD.S16	Q9, Q9, Q1
	VLD1.64	{D2}, [r1], r2
	VQADD.S16	Q10,Q10,Q2
	VLD1.64	{D4}, [r1], r2
	VQADD.S16	Q11,Q11,Q3
	VLD1.64	{D6}, [r1], r2
	VMOVL.U8	Q0, D0
	VMOVL.U8	Q1, D2
	VMOVL.U8	Q2, D4
	VMOVL.U8	Q3, D6
	VQADD.S16	Q12,Q12,Q0
	VQADD.S16	Q13,Q13,Q1
	VQADD.S16	Q14,Q14,Q2
	VQADD.S16	Q15,Q15,Q3
	VQMOVUN.S16	D16,Q8
	VQMOVUN.S16	D17,Q9
	VQMOVUN.S16	D18,Q10
	VST1.64	{D16},[r0@64], r2
	VQMOVUN.S16	D19,Q11
	VST1.64	{D17},[r0@64], r2
	VQMOVUN.S16	D20,Q12
	VST1.64	{D18},[r0@64], r2
	VQMOVUN.S16	D21,Q13
	VST1.64	{D19},[r0@64], r2
	VQMOVUN.S16	D22,Q14
	VST1.64	{D20},[r0@64], r2
	VQMOVUN.S16	D23,Q15
	VST1.64	{D21},[r0@64], r2
	VST1.64	{D22},[r0@64], r2
	VST1.64	{D23},[r0@64], r2
	MOV	PC,R14
	ENDP

oc_frag_recon_inter2_neon PROC
	; r0 =       unsigned char *_dst
	; r1 = const unsigned char *_src1
	; r2 = const unsigned char *_src2
	; r3 =       int            _ystride
	LDR	r12,[r13]
	; r12= const ogg_int16_t    _residue[64]
	VLDMIA	r12,{D16-D31}
	VLD1.64	{D0}, [r1], r3
	VLD1.64	{D4}, [r2], r3
	VLD1.64	{D1}, [r1], r3
	VLD1.64	{D5}, [r2], r3
	VHADD.U8	Q2, Q0, Q2	; Q2 = FFEEDDCCBBAA99887766554433221100
	VLD1.64	{D2}, [r1], r3
	VLD1.64	{D6}, [r2], r3
	VMOVL.U8	Q0, D4		; Q0 = __77__66__55__44__33__22__11__00
	VLD1.64	{D3}, [r1], r3
	VMOVL.U8	Q2, D5		; etc
	VLD1.64	{D7}, [r2], r3
	VHADD.U8	Q3, Q1, Q3
	VQADD.S16	Q8, Q8, Q0
	VQADD.S16	Q9, Q9, Q2
	VLD1.64	{D0}, [r1], r3
	VMOVL.U8	Q1, D6
	VLD1.64	{D4}, [r2], r3
	VMOVL.U8	Q3, D7
	VLD1.64	{D1}, [r1], r3
	VQADD.S16	Q10,Q10,Q1
	VLD1.64	{D5}, [r2], r3
	VQADD.S16	Q11,Q11,Q3
	VLD1.64	{D2}, [r1], r3
	VHADD.U8	Q2, Q0, Q2
	VLD1.64	{D6}, [r2], r3
	VLD1.64	{D3}, [r1], r3
	VMOVL.U8	Q0, D4
	VLD1.64	{D7}, [r2], r3
	VMOVL.U8	Q2, D5
	VHADD.U8	Q3, Q1, Q3
	VQADD.S16	Q12,Q12,Q0
	VQADD.S16	Q13,Q13,Q2
	VMOVL.U8	Q1, D6
	VMOVL.U8	Q3, D7
	VQADD.S16	Q14,Q14,Q1
	VQADD.S16	Q15,Q15,Q3
	VQMOVUN.S16	D16,Q8
	VQMOVUN.S16	D17,Q9
	VQMOVUN.S16	D18,Q10
	VST1.64	{D16},[r0@64], r3
	VQMOVUN.S16	D19,Q11
	VST1.64	{D17},[r0@64], r3
	VQMOVUN.S16	D20,Q12
	VST1.64	{D18},[r0@64], r3
	VQMOVUN.S16	D21,Q13
	VST1.64	{D19},[r0@64], r3
	VQMOVUN.S16	D22,Q14
	VST1.64	{D20},[r0@64], r3
	VQMOVUN.S16	D23,Q15
	VST1.64	{D21},[r0@64], r3
	VST1.64	{D22},[r0@64], r3
	VST1.64	{D23},[r0@64], r3
	MOV	PC,R14
	ENDP
 ]

	END
