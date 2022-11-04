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

	EXPORT	oc_idct8x8_1_arm
	EXPORT	oc_idct8x8_arm

oc_idct8x8_1_arm PROC
	; r0 = ogg_int16_t  *_y
	; r1 = ogg_uint16_t  _dc
	ORR	r1, r1, r1, LSL #16
	MOV	r2, r1
	MOV	r3, r1
	MOV	r12,r1
	STMIA	r0!,{r1,r2,r3,r12}
	STMIA	r0!,{r1,r2,r3,r12}
	STMIA	r0!,{r1,r2,r3,r12}
	STMIA	r0!,{r1,r2,r3,r12}
	STMIA	r0!,{r1,r2,r3,r12}
	STMIA	r0!,{r1,r2,r3,r12}
	STMIA	r0!,{r1,r2,r3,r12}
	STMIA	r0!,{r1,r2,r3,r12}
	MOV	PC, r14
	ENDP

oc_idct8x8_arm PROC
	; r0 = ogg_int16_t *_y
	; r1 = ogg_int16_t *_x
	; r2 = int          _last_zzi
	CMP	r2, #3
	BLE	oc_idct8x8_3_arm
	CMP	r2, #6
	BLE	oc_idct8x8_6_arm
	CMP	r2, #10
	BLE	oc_idct8x8_10_arm
oc_idct8x8_slow_arm
	STMFD	r13!,{r4-r11,r14}
	SUB	r13,r13,#64*2
; Row transforms
	STR	r0, [r13,#-4]!
	ADD	r0, r13, #4	; Write to temp storage.
	BL	idct8core_arm
	BL	idct8core_arm
	BL	idct8core_arm
	BL	idct8core_arm
	BL	idct8core_arm
	BL	idct8core_arm
	BL	idct8core_arm
	BL	idct8core_arm
	LDR	r0, [r13], #4	; Write to the final destination.
	SUB	r2, r1, #8*16
	; Clear input data for next block.
	MOV	r4, #0
	MOV	r5, #0
	MOV	r6, #0
	MOV	r7, #0
	STMIA	r2!,{r4,r5,r6,r7}
	STMIA	r2!,{r4,r5,r6,r7}
	STMIA	r2!,{r4,r5,r6,r7}
	STMIA	r2!,{r4,r5,r6,r7}
	STMIA	r2!,{r4,r5,r6,r7}
	STMIA	r2!,{r4,r5,r6,r7}
	STMIA	r2!,{r4,r5,r6,r7}
	STMIA	r2!,{r4,r5,r6,r7}
	MOV	r1, r13		; And read from temp storage.
; Column transforms
	BL	idct8core_down_arm
	BL	idct8core_down_arm
	BL	idct8core_down_arm
	BL	idct8core_down_arm
	BL	idct8core_down_arm
	BL	idct8core_down_arm
	BL	idct8core_down_arm
	BL	idct8core_down_arm
	ADD	r13,r13,#64*2
	LDMFD	r13!,{r4-r11,PC}
	ENDP

oc_idct8x8_10_arm PROC
	STMFD	r13!,{r4-r11,r14}
	SUB	r13,r13,#64*2
; Row transforms
	MOV	r2, r0
	MOV	r0, r13		; Write to temp storage.
	BL	idct4core_arm
	BL	idct3core_arm
	BL	idct2core_arm
	BL	idct1core_arm
	; Clear input data for next block.
	MOV	r4, #0
	STR	r4, [r1,#-4*16]!
	STR	r4, [r1,#4]
	STR	r4, [r1,#16]
	STR	r4, [r1,#20]
	STR	r4, [r1,#32]
	STR	r4, [r1,#48]
	MOV	r1, r13		; Read from temp storage.
	MOV	r0, r2		; Write to the final destination
oc_idct8x8_10_arm_cols
; Column transforms
	BL	idct4core_down_arm
	BL	idct4core_down_arm
	BL	idct4core_down_arm
	BL	idct4core_down_arm
	BL	idct4core_down_arm
	BL	idct4core_down_arm
	BL	idct4core_down_arm
	BL	idct4core_down_arm
	ADD	r13,r13,#64*2
	LDMFD	r13!,{r4-r11,PC}
	ENDP

oc_idct8x8_6_arm PROC
	STMFD	r13!,{r4-r7,r9-r11,r14}
	SUB	r13,r13,#64*2
; Row transforms
	MOV	r2, r0
	MOV	r0, r13		; Write to temp storage.
	BL	idct3core_arm
	BL	idct2core_arm
	BL	idct1core_arm
	; Clear input data for next block.
	MOV	r4, #0
	STR	r4, [r1,#-3*16]!
	STR	r4, [r1,#4]
	STR	r4, [r1,#16]
	STR	r4, [r1,#32]
	MOV	r1, r13		; Read from temp storage.
	MOV	r0, r2		; Write to the final destination
; Column transforms
	BL	idct3core_down_arm
	BL	idct3core_down_arm
	BL	idct3core_down_arm
	BL	idct3core_down_arm
	BL	idct3core_down_arm
	BL	idct3core_down_arm
	BL	idct3core_down_arm
	BL	idct3core_down_arm
	ADD	r13,r13,#64*2
	LDMFD	r13!,{r4-r7,r9-r11,PC}
	ENDP

oc_idct8x8_3_arm PROC
	STMFD	r13!,{r4-r7,r9-r11,r14}
	SUB	r13,r13,#64*2
; Row transforms
	MOV	r2, r0
	MOV	r0, r13		; Write to temp storage.
	BL	idct2core_arm
	BL	idct1core_arm
	; Clear input data for next block.
	MOV	r4, #0
	STR	r4, [r1,#-2*16]!
	STR	r4, [r1,#16]
	MOV	r1, r13		; Read from temp storage.
	MOV	r0, r2		; Write to the final destination
; Column transforms
	BL	idct2core_down_arm
	BL	idct2core_down_arm
	BL	idct2core_down_arm
	BL	idct2core_down_arm
	BL	idct2core_down_arm
	BL	idct2core_down_arm
	BL	idct2core_down_arm
	BL	idct2core_down_arm
	ADD	r13,r13,#64*2
	LDMFD	r13!,{r4-r7,r9-r11,PC}
	ENDP

idct1core_arm PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r3, [r1], #16
	MOV	r12,#0x05
	ORR	r12,r12,#0xB500
	MUL	r3, r12, r3
	; Stall ?
	MOV	r3, r3, ASR #16
	STRH	r3, [r0], #2
	STRH	r3, [r0, #14]
	STRH	r3, [r0, #30]
	STRH	r3, [r0, #46]
	STRH	r3, [r0, #62]
	STRH	r3, [r0, #78]
	STRH	r3, [r0, #94]
	STRH	r3, [r0, #110]
	MOV	PC,R14
	ENDP

idct2core_arm PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r9, [r1], #16		; r9 = x[0]
	LDR	r12,OC_C4S4
	LDRSH	r11,[r1, #-14]		; r11= x[1]
	LDR	r3, OC_C7S1
	MUL	r9, r12,r9		; r9 = t[0]<<16 = OC_C4S4*x[0]
	LDR	r10,OC_C1S7
	MUL	r3, r11,r3		; r3 = t[4]<<16 = OC_C7S1*x[1]
	MOV	r9, r9, ASR #16		; r9 = t[0]
	MUL	r11,r10,r11		; r11= t[7]<<16 = OC_C1S7*x[1]
	MOV	r3, r3, ASR #16		; r3 = t[4]
	MUL	r10,r12,r3		; r10= t[5]<<16 = OC_C4S4*t[4]
	MOV	r11,r11,ASR #16		; r11= t[7]
	MUL	r12,r11,r12		; r12= t[6]<<16 = OC_C4S4*t[7]
	MOV	r10,r10,ASR #16		; r10= t[5]
	ADD	r12,r9,r12,ASR #16	; r12= t[0]+t[6]
	ADD	r12,r12,r10		; r12= t[0]+t2[6] = t[0]+t[6]+t[5]
	SUB	r10,r12,r10,LSL #1	; r10= t[0]+t2[5] = t[0]+t[6]-t[5]
	ADD	r3, r3, r9		; r3 = t[0]+t[4]
	ADD	r11,r11,r9		; r11= t[0]+t[7]
	STRH	r11,[r0], #2		; y[0] = t[0]+t[7]
	STRH	r12,[r0, #14]		; y[1] = t[0]+t[6]
	STRH	r10,[r0, #30]		; y[2] = t[0]+t[5]
	STRH	r3, [r0, #46]		; y[3] = t[0]+t[4]
	RSB	r3, r3, r9, LSL #1	; r3 = t[0]*2-(t[0]+t[4])=t[0]-t[4]
	RSB	r10,r10,r9, LSL #1	; r10= t[0]*2-(t[0]+t[5])=t[0]-t[5]
	RSB	r12,r12,r9, LSL #1	; r12= t[0]*2-(t[0]+t[6])=t[0]-t[6]
	RSB	r11,r11,r9, LSL #1	; r1 = t[0]*2-(t[0]+t[7])=t[0]-t[7]
	STRH	r3, [r0, #62]		; y[4] = t[0]-t[4]
	STRH	r10,[r0, #78]		; y[5] = t[0]-t[5]
	STRH	r12,[r0, #94]		; y[6] = t[0]-t[6]
	STRH	r11,[r0, #110]		; y[7] = t[0]-t[7]
	MOV	PC,r14
	ENDP

idct2core_down_arm PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r9, [r1], #16		; r9 = x[0]
	LDR	r12,OC_C4S4
	LDRSH	r11,[r1, #-14]		; r11= x[1]
	LDR	r3, OC_C7S1
	MUL	r9, r12,r9		; r9 = t[0]<<16 = OC_C4S4*x[0]
	LDR	r10,OC_C1S7
	MUL	r3, r11,r3		; r3 = t[4]<<16 = OC_C7S1*x[1]
	MOV	r9, r9, ASR #16		; r9 = t[0]
	MUL	r11,r10,r11		; r11= t[7]<<16 = OC_C1S7*x[1]
	ADD	r9, r9, #8		; r9 = t[0]+8
	MOV	r3, r3, ASR #16		; r3 = t[4]
	MUL	r10,r12,r3		; r10= t[5]<<16 = OC_C4S4*t[4]
	MOV	r11,r11,ASR #16		; r11= t[7]
	MUL	r12,r11,r12		; r12= t[6]<<16 = OC_C4S4*t[7]
	MOV	r10,r10,ASR #16		; r10= t[5]
	ADD	r12,r9,r12,ASR #16	; r12= t[0]+t[6]+8
	ADD	r12,r12,r10		; r12= t[0]+t2[6] = t[0]+t[6]+t[5]+8
	SUB	r10,r12,r10,LSL #1	; r10= t[0]+t2[5] = t[0]+t[6]-t[5]+8
	ADD	r3, r3, r9		; r3 = t[0]+t[4]+8
	ADD	r11,r11,r9		; r11= t[0]+t[7]+8
	; TODO: This is wrong.
	; The C code truncates to 16 bits by storing to RAM and doing the
	;  shifts later; we've got an extra 4 bits here.
	MOV	r4, r11,ASR #4
	MOV	r5, r12,ASR #4
	MOV	r6, r10,ASR #4
	MOV	r7, r3, ASR #4
	RSB	r3, r3, r9, LSL #1	;r3 =t[0]*2+8-(t[0]+t[4])=t[0]-t[4]+8
	RSB	r10,r10,r9, LSL #1	;r10=t[0]*2+8-(t[0]+t[5])=t[0]-t[5]+8
	RSB	r12,r12,r9, LSL #1	;r12=t[0]*2+8-(t[0]+t[6])=t[0]-t[6]+8
	RSB	r11,r11,r9, LSL #1	;r11=t[0]*2+8-(t[0]+t[7])=t[0]-t[7]+8
	MOV	r3, r3, ASR #4
	MOV	r10,r10,ASR #4
	MOV	r12,r12,ASR #4
	MOV	r11,r11,ASR #4
	STRH	r4, [r0], #2		; y[0] = t[0]+t[7]
	STRH	r5, [r0, #14]		; y[1] = t[0]+t[6]
	STRH	r6, [r0, #30]		; y[2] = t[0]+t[5]
	STRH	r7, [r0, #46]		; y[3] = t[0]+t[4]
	STRH	r3, [r0, #62]		; y[4] = t[0]-t[4]
	STRH	r10,[r0, #78]		; y[5] = t[0]-t[5]
	STRH	r12,[r0, #94]		; y[6] = t[0]-t[6]
	STRH	r11,[r0, #110]		; y[7] = t[0]-t[7]
	MOV	PC,r14
	ENDP

idct3core_arm PROC
	LDRSH	r9, [r1], #16		; r9 = x[0]
	LDR	r12,OC_C4S4		; r12= OC_C4S4
	LDRSH	r3, [r1, #-12]		; r3 = x[2]
	LDR	r10,OC_C6S2		; r10= OC_C6S2
	MUL	r9, r12,r9		; r9 = t[0]<<16 = OC_C4S4*x[0]
	LDR	r4, OC_C2S6		; r4 = OC_C2S6
	MUL	r10,r3, r10		; r10= t[2]<<16 = OC_C6S2*x[2]
	LDRSH	r11,[r1, #-14]		; r11= x[1]
	MUL	r3, r4, r3		; r3 = t[3]<<16 = OC_C2S6*x[2]
	LDR	r4, OC_C7S1		; r4 = OC_C7S1
	LDR	r5, OC_C1S7		; r5 = OC_C1S7
	MOV	r9, r9, ASR #16		; r9 = t[0]
	MUL	r4, r11,r4		; r4 = t[4]<<16 = OC_C7S1*x[1]
	ADD	r3, r9, r3, ASR #16	; r3 = t[0]+t[3]
	MUL	r11,r5, r11		; r11= t[7]<<16 = OC_C1S7*x[1]
	MOV	r4, r4, ASR #16		; r4 = t[4]
	MUL	r5, r12,r4		; r5 = t[5]<<16 = OC_C4S4*t[4]
	MOV	r11,r11,ASR #16		; r11= t[7]
	MUL	r12,r11,r12		; r12= t[6]<<16 = OC_C4S4*t[7]
	ADD	r10,r9, r10,ASR #16	; r10= t[1] = t[0]+t[2]
	RSB	r6, r10,r9, LSL #1	; r6 = t[2] = t[0]-t[2]
					; r3 = t2[0] = t[0]+t[3]
	RSB	r9, r3, r9, LSL #1	; r9 = t2[3] = t[0]-t[3]
	MOV	r12,r12,ASR #16		; r12= t[6]
	ADD	r5, r12,r5, ASR #16	; r5 = t2[6] = t[6]+t[5]
	RSB	r12,r5, r12,LSL #1	; r12= t2[5] = t[6]-t[5]
	ADD	r11,r3, r11		; r11= t2[0]+t[7]
	ADD	r5, r10,r5		; r5 = t[1]+t2[6]
	ADD	r12,r6, r12		; r12= t[2]+t2[5]
	ADD	r4, r9, r4		; r4 = t2[3]+t[4]
	STRH	r11,[r0], #2		; y[0] = t[0]+t[7]
	STRH	r5, [r0, #14]		; y[1] = t[1]+t2[6]
	STRH	r12,[r0, #30]		; y[2] = t[2]+t2[5]
	STRH	r4, [r0, #46]		; y[3] = t2[3]+t[4]
	RSB	r11,r11,r3, LSL #1	; r11= t2[0] - t[7]
	RSB	r5, r5, r10,LSL #1	; r5 = t[1]  - t2[6]
	RSB	r12,r12,r6, LSL #1	; r6 = t[2]  - t2[5]
	RSB	r4, r4, r9, LSL #1	; r4 = t2[3] - t[4]
	STRH	r4, [r0, #62]		; y[4] = t2[3]-t[4]
	STRH	r12,[r0, #78]		; y[5] = t[2]-t2[5]
	STRH	r5, [r0, #94]		; y[6] = t[1]-t2[6]
	STRH	r11,[r0, #110]		; y[7] = t2[0]-t[7]
	MOV	PC,R14
	ENDP

idct3core_down_arm PROC
	LDRSH	r9, [r1], #16		; r9 = x[0]
	LDR	r12,OC_C4S4		; r12= OC_C4S4
	LDRSH	r3, [r1, #-12]		; r3 = x[2]
	LDR	r10,OC_C6S2		; r10= OC_C6S2
	MUL	r9, r12,r9		; r9 = t[0]<<16 = OC_C4S4*x[0]
	LDR	r4, OC_C2S6		; r4 = OC_C2S6
	MUL	r10,r3, r10		; r10= t[2]<<16 = OC_C6S2*x[2]
	LDRSH	r11,[r1, #-14]		; r11= x[1]
	MUL	r3, r4, r3		; r3 = t[3]<<16 = OC_C2S6*x[2]
	LDR	r4, OC_C7S1		; r4 = OC_C7S1
	LDR	r5, OC_C1S7		; r5 = OC_C1S7
	MOV	r9, r9, ASR #16		; r9 = t[0]
	MUL	r4, r11,r4		; r4 = t[4]<<16 = OC_C7S1*x[1]
	ADD	r9, r9, #8		; r9 = t[0]+8
	MUL	r11,r5, r11		; r11= t[7]<<16 = OC_C1S7*x[1]
	ADD	r3, r9, r3, ASR #16	; r3 = t[0]+t[3]+8
	MOV	r4, r4, ASR #16		; r4 = t[4]
	MUL	r5, r12,r4		; r5 = t[5]<<16 = OC_C4S4*t[4]
	MOV	r11,r11,ASR #16		; r11= t[7]
	MUL	r12,r11,r12		; r12= t[6]<<16 = OC_C4S4*t[7]
	ADD	r10,r9, r10,ASR #16	; r10= t[1]+8 = t[0]+t[2]+8
	RSB	r6, r10,r9, LSL #1	; r6 = t[2]+8 = t[0]-t[2]+8
					; r3 = t2[0]+8 = t[0]+t[3]+8
	RSB	r9, r3, r9, LSL #1	; r9 = t2[3]+8 = t[0]-t[3]+8
	MOV	r12,r12,ASR #16		; r12= t[6]
	ADD	r5, r12,r5, ASR #16	; r5 = t2[6] = t[6]+t[5]
	RSB	r12,r5, r12,LSL #1	; r12= t2[5] = t[6]-t[5]
	ADD	r11,r3, r11		; r11= t2[0]+t[7] +8
	ADD	r5, r10,r5		; r5 = t[1] +t2[6]+8
	ADD	r12,r6, r12		; r12= t[2] +t2[5]+8
	ADD	r4, r9, r4		; r4 = t2[3]+t[4] +8
	RSB	r3, r11,r3, LSL #1	; r11= t2[0] - t[7]  + 8
	RSB	r10,r5, r10,LSL #1	; r5 = t[1]  - t2[6] + 8
	RSB	r6, r12,r6, LSL #1	; r6 = t[2]  - t2[5] + 8
	RSB	r9, r4, r9, LSL #1	; r4 = t2[3] - t[4]  + 8
	; TODO: This is wrong.
	; The C code truncates to 16 bits by storing to RAM and doing the
	;  shifts later; we've got an extra 4 bits here.
	MOV	r11,r11,ASR #4
	MOV	r5, r5, ASR #4
	MOV	r12,r12,ASR #4
	MOV	r4, r4, ASR #4
	MOV	r9, r9, ASR #4
	MOV	r6, r6, ASR #4
	MOV	r10,r10,ASR #4
	MOV	r3, r3, ASR #4
	STRH	r11,[r0], #2		; y[0] = t[0]+t[7]
	STRH	r5, [r0, #14]		; y[1] = t[1]+t2[6]
	STRH	r12,[r0, #30]		; y[2] = t[2]+t2[5]
	STRH	r4, [r0, #46]		; y[3] = t2[3]+t[4]
	STRH	r9, [r0, #62]		; y[4] = t2[3]-t[4]
	STRH	r6, [r0, #78]		; y[5] = t[2]-t2[5]
	STRH	r10,[r0, #94]		; y[6] = t[1]-t2[6]
	STRH	r3, [r0, #110]		; y[7] = t2[0]-t[7]
	MOV	PC,R14
	ENDP

idct4core_arm PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r9, [r1], #16		; r9 = x[0]
	LDR	r10,OC_C4S4		; r10= OC_C4S4
	LDRSH	r12,[r1, #-12]		; r12= x[2]
	LDR	r4, OC_C6S2		; r4 = OC_C6S2
	MUL	r9, r10,r9		; r9 = t[0]<<16 = OC_C4S4*x[0]
	LDR	r5, OC_C2S6		; r5 = OC_C2S6
	MUL	r4, r12,r4		; r4 = t[2]<<16 = OC_C6S2*x[2]
	LDRSH	r3, [r1, #-14]		; r3 = x[1]
	MUL	r5, r12,r5		; r5 = t[3]<<16 = OC_C2S6*x[2]
	LDR	r6, OC_C7S1		; r6 = OC_C7S1
	LDR	r12,OC_C1S7		; r12= OC_C1S7
	LDRSH	r11,[r1, #-10]		; r11= x[3]
	MUL	r6, r3, r6		; r6 = t[4]<<16 = OC_C7S1*x[1]
	LDR	r7, OC_C5S3		; r7 = OC_C5S3
	MUL	r3, r12,r3		; r3 = t[7]<<16 = OC_C1S7*x[1]
	LDR	r8, OC_C3S5		; r8 = OC_C3S5
	MUL	r7, r11,r7		; r7 = -t[5]<<16 = OC_C5S3*x[3]
	MOV	r9, r9, ASR #16		; r9 = t[0]
	MUL	r11,r8, r11		; r11= t[6]<<16 = OC_C3S5*x[3]
	MOV	r6, r6, ASR #16		; r6 = t[4]
; TODO: This is wrong; t[4]-t[5] and t[7]-t[6] need to be truncated to 16-bit
; before multiplying, not after (this is not equivalent)
	SUB	r7, r6, r7, ASR #16	; r7 = t2[4]=t[4]+t[5] (as r7=-t[5])
	RSB	r6, r7, r6, LSL #1	; r6 = t[4]-t[5]
	MUL	r6, r10,r6		; r6 = t2[5]<<16 =OC_C4S4*(t[4]-t[5])
	MOV	r3, r3, ASR #16		; r3 = t[7]
	ADD	r11,r3, r11,ASR #16	; r11= t2[7]=t[7]+t[6]
	RSB	r3, r11,r3, LSL #1	; r3 = t[7]-t[6]
	MUL	r3, r10,r3		; r3 = t2[6]<<16 =OC_C4S4*(t[7]-t[6])
	ADD	r4, r9, r4, ASR #16	; r4 = t[1] = t[0] + t[2]
	RSB	r10,r4, r9, LSL #1	; r10= t[2] = t[0] - t[2]
	ADD	r5, r9, r5, ASR #16	; r5 = t[0] = t[0] + t[3]
	RSB	r9, r5, r9, LSL #1	; r9 = t[3] = t[0] - t[3]
	MOV	r3, r3, ASR #16		; r3 = t2[6]
	ADD	r6, r3, r6, ASR #16	; r6 = t3[6] = t2[6]+t2[5]
	RSB	r3, r6, r3, LSL #1	; r3 = t3[5] = t2[6]-t2[5]
	ADD	r11,r5, r11		; r11= t[0]+t2[7]
	ADD	r6, r4, r6		; r6 = t[1]+t3[6]
	ADD	r3, r10,r3		; r3 = t[2]+t3[5]
	ADD	r7, r9, r7		; r7 = t[3]+t2[4]
	STRH	r11,[r0], #2		; y[0] = t[0]+t[7]
	STRH	r6, [r0, #14]		; y[1] = t[1]+t2[6]
	STRH	r3, [r0, #30]		; y[2] = t[2]+t2[5]
	STRH	r7, [r0, #46]		; y[3] = t2[3]+t[4]
	RSB	r11,r11,r5, LSL #1	; r11= t[0]-t2[7]
	RSB	r6, r6, r4, LSL #1	; r6 = t[1]-t3[6]
	RSB	r3, r3, r10,LSL #1	; r3 = t[2]-t3[5]
	RSB	r7, r7, r9, LSL #1	; r7 = t[3]-t2[4]
	STRH	r7, [r0, #62]		; y[4] = t2[3]-t[4]
	STRH	r3, [r0, #78]		; y[5] = t[2]-t2[5]
	STRH	r6, [r0, #94]		; y[6] = t[1]-t2[6]
	STRH	r11, [r0, #110]		; y[7] = t2[0]-t[7]
	MOV	PC,r14
	ENDP

idct4core_down_arm PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r9, [r1], #16		; r9 = x[0]
	LDR	r10,OC_C4S4		; r10= OC_C4S4
	LDRSH	r12,[r1, #-12]		; r12= x[2]
	LDR	r4, OC_C6S2		; r4 = OC_C6S2
	MUL	r9, r10,r9		; r9 = t[0]<<16 = OC_C4S4*x[0]
	LDR	r5, OC_C2S6		; r5 = OC_C2S6
	MUL	r4, r12,r4		; r4 = t[2]<<16 = OC_C6S2*x[2]
	LDRSH	r3, [r1, #-14]		; r3 = x[1]
	MUL	r5, r12,r5		; r5 = t[3]<<16 = OC_C2S6*x[2]
	LDR	r6, OC_C7S1		; r6 = OC_C7S1
	LDR	r12,OC_C1S7		; r12= OC_C1S7
	LDRSH	r11,[r1, #-10]		; r11= x[3]
	MUL	r6, r3, r6		; r6 = t[4]<<16 = OC_C7S1*x[1]
	LDR	r7, OC_C5S3		; r7 = OC_C5S3
	MUL	r3, r12,r3		; r3 = t[7]<<16 = OC_C1S7*x[1]
	LDR	r8, OC_C3S5		; r8 = OC_C3S5
	MUL	r7, r11,r7		; r7 = -t[5]<<16 = OC_C5S3*x[3]
	MOV	r9, r9, ASR #16		; r9 = t[0]
	MUL	r11,r8, r11		; r11= t[6]<<16 = OC_C3S5*x[3]
	MOV	r6, r6, ASR #16		; r6 = t[4]
; TODO: This is wrong; t[4]-t[5] and t[7]-t[6] need to be truncated to 16-bit
; before multiplying, not after (this is not equivalent)
	SUB	r7, r6, r7, ASR #16	; r7 = t2[4]=t[4]+t[5] (as r7=-t[5])
	RSB	r6, r7, r6, LSL #1	; r6 = t[4]-t[5]
	MUL	r6, r10,r6		; r6 = t2[5]<<16 =OC_C4S4*(t[4]-t[5])
	MOV	r3, r3, ASR #16		; r3 = t[7]
	ADD	r11,r3, r11,ASR #16	; r11= t2[7]=t[7]+t[6]
	RSB	r3, r11,r3, LSL #1	; r3 = t[7]-t[6]
	ADD	r9, r9, #8		; r9 = t[0]+8
	MUL	r3, r10,r3		; r3 = t2[6]<<16 =OC_C4S4*(t[7]-t[6])
	ADD	r4, r9, r4, ASR #16	; r4 = t[1] = t[0] + t[2] + 8
	RSB	r10,r4, r9, LSL #1	; r10= t[2] = t[0] - t[2] + 8
	ADD	r5, r9, r5, ASR #16	; r5 = t[0] = t[0] + t[3] + 8
	RSB	r9, r5, r9, LSL #1	; r9 = t[3] = t[0] - t[3] + 8
	MOV	r3, r3, ASR #16		; r3 = t2[6]
	ADD	r6, r3, r6, ASR #16	; r6 = t3[6] = t2[6]+t2[5]
	RSB	r3, r6, r3, LSL #1	; r3 = t3[5] = t2[6]-t2[5]
	ADD	r5, r5, r11		; r5 = t[0]+t2[7]+8
	ADD	r4, r4, r6		; r4 = t[1]+t3[6]+8
	ADD	r10,r10,r3		; r10= t[2]+t3[5]+8
	ADD	r9, r9, r7		; r9 = t[3]+t2[4]+8
	SUB	r11,r5, r11,LSL #1	; r11= t[0]-t2[7]+8
	SUB	r6, r4, r6, LSL #1	; r6 = t[1]-t3[6]+8
	SUB	r3, r10,r3, LSL #1	; r3 = t[2]-t3[5]+8
	SUB	r7, r9, r7, LSL #1	; r7 = t[3]-t2[4]+8
	; TODO: This is wrong.
	; The C code truncates to 16 bits by storing to RAM and doing the
	;  shifts later; we've got an extra 4 bits here.
	MOV	r11,r11,ASR #4
	MOV	r6, r6, ASR #4
	MOV	r3, r3, ASR #4
	MOV	r7, r7, ASR #4
	MOV	r9, r9, ASR #4
	MOV	r10,r10,ASR #4
	MOV	r4, r4, ASR #4
	MOV	r5, r5, ASR #4
	STRH	r5,[r0], #2		; y[0] = t[0]+t[7]
	STRH	r4, [r0, #14]		; y[1] = t[1]+t2[6]
	STRH	r10,[r0, #30]		; y[2] = t[2]+t2[5]
	STRH	r9, [r0, #46]		; y[3] = t2[3]+t[4]
	STRH	r7, [r0, #62]		; y[4] = t2[3]-t[4]
	STRH	r3, [r0, #78]		; y[5] = t[2]-t2[5]
	STRH	r6, [r0, #94]		; y[6] = t[1]-t2[6]
	STRH	r11,[r0, #110]		; y[7] = t2[0]-t[7]
	MOV	PC,r14
	ENDP

idct8core_arm PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r2, [r1],#16		; r2 = x[0]
	STMFD	r13!,{r1,r14}
	LDRSH	r6, [r1, #-8]		; r6 = x[4]
	LDR	r12,OC_C4S4		; r12= C4S4
	LDRSH	r4, [r1, #-12]		; r4 = x[2]
	ADD	r2, r2, r6		; r2 = x[0] + x[4]
	SUB	r6, r2, r6, LSL #1	; r6 = x[0] - x[4]
	; For spec compliance, these sums must be truncated to 16-bit precision
	; _before_ the multiply (not after).
	; Sadly, ARMv4 provides no simple way to do that.
	MOV	r2, r2, LSL #16
	MOV	r6, r6, LSL #16
	MOV	r2, r2, ASR #16
	MOV	r6, r6, ASR #16
	MUL	r2, r12,r2		; r2 = t[0]<<16 = C4S4*(x[0]+x[4])
	LDRSH	r8, [r1, #-4]		; r8 = x[6]
	LDR	r7, OC_C6S2		; r7 = OC_C6S2
	MUL	r6, r12,r6		; r6 = t[1]<<16 = C4S4*(x[0]-x[4])
	LDR	r14,OC_C2S6		; r14= OC_C2S6
	MUL	r3, r4, r7		; r3 = OC_C6S2*x[2]
	LDR	r5, OC_C7S1		; r5 = OC_C7S1
	MUL	r4, r14,r4		; r4 = OC_C2S6*x[2]
	MOV	r3, r3, ASR #16		; r3 = OC_C6S2*x[2]>>16
	MUL	r14,r8, r14		; r14= OC_C2S6*x[6]
	MOV	r4, r4, ASR #16		; r4 = OC_C2S6*x[2]>>16
	MUL	r8, r7, r8		; r8 = OC_C6S2*x[6]
	LDR	r7, OC_C1S7		; r7 = OC_C1S7
	SUB	r3, r3, r14,ASR #16	; r3=t[2]=C6S2*x[2]>>16-C2S6*x[6]>>16
	LDRSH	r14,[r1, #-14]		; r14= x[1]
	ADD	r4, r4, r8, ASR #16	; r4=t[3]=C2S6*x[2]>>16+C6S2*x[6]>>16
	LDRSH	r8, [r1, #-2]		; r8 = x[7]
	MUL	r9, r5, r14		; r9 = OC_C7S1*x[1]
	LDRSH	r10,[r1, #-6]		; r10= x[5]
	MUL	r14,r7, r14		; r14= OC_C1S7*x[1]
	MOV	r9, r9, ASR #16		; r9 = OC_C7S1*x[1]>>16
	MUL	r7, r8, r7		; r7 = OC_C1S7*x[7]
	MOV	r14,r14,ASR #16		; r14= OC_C1S7*x[1]>>16
	MUL	r8, r5, r8		; r8 = OC_C7S1*x[7]
	LDRSH	r1, [r1, #-10]		; r1 = x[3]
	LDR	r5, OC_C3S5		; r5 = OC_C3S5
	LDR	r11,OC_C5S3		; r11= OC_C5S3
	ADD	r8, r14,r8, ASR #16	; r8=t[7]=C1S7*x[1]>>16+C7S1*x[7]>>16
	MUL	r14,r5, r10		; r14= OC_C3S5*x[5]
	SUB	r9, r9, r7, ASR #16	; r9=t[4]=C7S1*x[1]>>16-C1S7*x[7]>>16
	MUL	r10,r11,r10		; r10= OC_C5S3*x[5]
	MOV	r14,r14,ASR #16		; r14= OC_C3S5*x[5]>>16
	MUL	r11,r1, r11		; r11= OC_C5S3*x[3]
	MOV	r10,r10,ASR #16		; r10= OC_C5S3*x[5]>>16
	MUL	r1, r5, r1		; r1 = OC_C3S5*x[3]
	SUB	r14,r14,r11,ASR #16	;r14=t[5]=C3S5*x[5]>>16-C5S3*x[3]>>16
	ADD	r10,r10,r1, ASR #16	;r10=t[6]=C5S3*x[5]>>16+C3S5*x[3]>>16
	; r2=t[0]<<16 r3=t[2] r4=t[3] r6=t[1]<<16 r8=t[7] r9=t[4]
	; r10=t[6] r12=C4S4 r14=t[5]
; TODO: This is wrong; t[4]-t[5] and t[7]-t[6] need to be truncated to 16-bit
; before multiplying, not after (this is not equivalent)
	; Stage 2
	; 4-5 butterfly
	ADD	r9, r9, r14		; r9 = t2[4]     =       t[4]+t[5]
	SUB	r14,r9, r14, LSL #1	; r14=                   t[4]-t[5]
	MUL	r14,r12,r14		; r14= t2[5]<<16 = C4S4*(t[4]-t[5])
	; 7-6 butterfly
	ADD	r8, r8, r10		; r8 = t2[7]     =       t[7]+t[6]
	SUB	r10,r8, r10, LSL #1	; r10=                   t[7]-t[6]
	MUL	r10,r12,r10		; r10= t2[6]<<16 = C4S4*(t[7]+t[6])
	; r2=t[0]<<16 r3=t[2] r4=t[3] r6=t[1]<<16 r8=t2[7] r9=t2[4]
	; r10=t2[6]<<16 r12=C4S4 r14=t2[5]<<16
	; Stage 3
	; 0-3 butterfly
	ADD	r2, r4, r2, ASR #16	; r2 = t2[0] = t[0] + t[3]
	SUB	r4, r2, r4, LSL #1	; r4 = t2[3] = t[0] - t[3]
	; 1-2 butterfly
	ADD	r6, r3, r6, ASR #16	; r6 = t2[1] = t[1] + t[2]
	SUB	r3, r6, r3, LSL #1	; r3 = t2[2] = t[1] - t[2]
	; 6-5 butterfly
	MOV	r14,r14,ASR #16		; r14= t2[5]
	ADD	r10,r14,r10,ASR #16	; r10= t3[6] = t[6] + t[5]
	SUB	r14,r10,r14,LSL #1	; r14= t3[5] = t[6] - t[5]
	; r2=t2[0] r3=t2[2] r4=t2[3] r6=t2[1] r8=t2[7] r9=t2[4]
	; r10=t3[6] r14=t3[5]
	; Stage 4
	ADD	r2, r2, r8		; r2 = t[0] + t[7]
	ADD	r6, r6, r10		; r6 = t[1] + t[6]
	ADD	r3, r3, r14		; r3 = t[2] + t[5]
	ADD	r4, r4, r9		; r4 = t[3] + t[4]
	SUB	r8, r2, r8, LSL #1	; r8 = t[0] - t[7]
	SUB	r10,r6, r10,LSL #1	; r10= t[1] - t[6]
	SUB	r14,r3, r14,LSL #1	; r14= t[2] - t[5]
	SUB	r9, r4, r9, LSL #1	; r9 = t[3] - t[4]
	STRH	r2, [r0], #2		; y[0] = t[0]+t[7]
	STRH	r6, [r0, #14]		; y[1] = t[1]+t[6]
	STRH	r3, [r0, #30]		; y[2] = t[2]+t[5]
	STRH	r4, [r0, #46]		; y[3] = t[3]+t[4]
	STRH	r9, [r0, #62]		; y[4] = t[3]-t[4]
	STRH	r14,[r0, #78]		; y[5] = t[2]-t[5]
	STRH	r10,[r0, #94]		; y[6] = t[1]-t[6]
	STRH	r8, [r0, #110]		; y[7] = t[0]-t[7]
	LDMFD	r13!,{r1,PC}
	ENDP

idct8core_down_arm PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r2, [r1],#16		; r2 = x[0]
	STMFD	r13!,{r1,r14}
	LDRSH	r6, [r1, #-8]		; r6 = x[4]
	LDR	r12,OC_C4S4		; r12= C4S4
	LDRSH	r4, [r1, #-12]		; r4 = x[2]
	ADD	r2, r2, r6		; r2 = x[0] + x[4]
	SUB	r6, r2, r6, LSL #1	; r6 = x[0] - x[4]
	; For spec compliance, these sums must be truncated to 16-bit precision
	; _before_ the multiply (not after).
	; Sadly, ARMv4 provides no simple way to do that.
	MOV	r2, r2, LSL #16
	MOV	r6, r6, LSL #16
	MOV	r2, r2, ASR #16
	MOV	r6, r6, ASR #16
	MUL	r2, r12,r2		; r2 = t[0]<<16 = C4S4*(x[0]+x[4])
	LDRSH	r8, [r1, #-4]		; r8 = x[6]
	LDR	r7, OC_C6S2		; r7 = OC_C6S2
	MUL	r6, r12,r6		; r6 = t[1]<<16 = C4S4*(x[0]-x[4])
	LDR	r14,OC_C2S6		; r14= OC_C2S6
	MUL	r3, r4, r7		; r3 = OC_C6S2*x[2]
	LDR	r5, OC_C7S1		; r5 = OC_C7S1
	MUL	r4, r14,r4		; r4 = OC_C2S6*x[2]
	MOV	r3, r3, ASR #16		; r3 = OC_C6S2*x[2]>>16
	MUL	r14,r8, r14		; r14= OC_C2S6*x[6]
	MOV	r4, r4, ASR #16		; r4 = OC_C2S6*x[2]>>16
	MUL	r8, r7, r8		; r8 = OC_C6S2*x[6]
	LDR	r7, OC_C1S7		; r7 = OC_C1S7
	SUB	r3, r3, r14,ASR #16	; r3=t[2]=C6S2*x[2]>>16-C2S6*x[6]>>16
	LDRSH	r14,[r1, #-14]		; r14= x[1]
	ADD	r4, r4, r8, ASR #16	; r4=t[3]=C2S6*x[2]>>16+C6S2*x[6]>>16
	LDRSH	r8, [r1, #-2]		; r8 = x[7]
	MUL	r9, r5, r14		; r9 = OC_C7S1*x[1]
	LDRSH	r10,[r1, #-6]		; r10= x[5]
	MUL	r14,r7, r14		; r14= OC_C1S7*x[1]
	MOV	r9, r9, ASR #16		; r9 = OC_C7S1*x[1]>>16
	MUL	r7, r8, r7		; r7 = OC_C1S7*x[7]
	MOV	r14,r14,ASR #16		; r14= OC_C1S7*x[1]>>16
	MUL	r8, r5, r8		; r8 = OC_C7S1*x[7]
	LDRSH	r1, [r1, #-10]		; r1 = x[3]
	LDR	r5, OC_C3S5		; r5 = OC_C3S5
	LDR	r11,OC_C5S3		; r11= OC_C5S3
	ADD	r8, r14,r8, ASR #16	; r8=t[7]=C1S7*x[1]>>16+C7S1*x[7]>>16
	MUL	r14,r5, r10		; r14= OC_C3S5*x[5]
	SUB	r9, r9, r7, ASR #16	; r9=t[4]=C7S1*x[1]>>16-C1S7*x[7]>>16
	MUL	r10,r11,r10		; r10= OC_C5S3*x[5]
	MOV	r14,r14,ASR #16		; r14= OC_C3S5*x[5]>>16
	MUL	r11,r1, r11		; r11= OC_C5S3*x[3]
	MOV	r10,r10,ASR #16		; r10= OC_C5S3*x[5]>>16
	MUL	r1, r5, r1		; r1 = OC_C3S5*x[3]
	SUB	r14,r14,r11,ASR #16	;r14=t[5]=C3S5*x[5]>>16-C5S3*x[3]>>16
	ADD	r10,r10,r1, ASR #16	;r10=t[6]=C5S3*x[5]>>16+C3S5*x[3]>>16
	; r2=t[0]<<16 r3=t[2] r4=t[3] r6=t[1]<<16 r8=t[7] r9=t[4]
	; r10=t[6] r12=C4S4 r14=t[5]
	; Stage 2
; TODO: This is wrong; t[4]-t[5] and t[7]-t[6] need to be truncated to 16-bit
; before multiplying, not after (this is not equivalent)
	; 4-5 butterfly
	ADD	r9, r9, r14		; r9 = t2[4]     =       t[4]+t[5]
	SUB	r14,r9, r14, LSL #1	; r14=                   t[4]-t[5]
	MUL	r14,r12,r14		; r14= t2[5]<<16 = C4S4*(t[4]-t[5])
	; 7-6 butterfly
	ADD	r8, r8, r10		; r8 = t2[7]     =       t[7]+t[6]
	SUB	r10,r8, r10, LSL #1	; r10=                   t[7]-t[6]
	MUL	r10,r12,r10		; r10= t2[6]<<16 = C4S4*(t[7]+t[6])
	; r2=t[0]<<16 r3=t[2] r4=t[3] r6=t[1]<<16 r8=t2[7] r9=t2[4]
	; r10=t2[6]<<16 r12=C4S4 r14=t2[5]<<16
	; Stage 3
	ADD	r2, r2, #8<<16		; r2 = t[0]+8<<16
	ADD	r6, r6, #8<<16		; r6 = t[1]+8<<16
	; 0-3 butterfly
	ADD	r2, r4, r2, ASR #16	; r2 = t2[0] = t[0] + t[3] + 8
	SUB	r4, r2, r4, LSL #1	; r4 = t2[3] = t[0] - t[3] + 8
	; 1-2 butterfly
	ADD	r6, r3, r6, ASR #16	; r6 = t2[1] = t[1] + t[2] + 8
	SUB	r3, r6, r3, LSL #1	; r3 = t2[2] = t[1] - t[2] + 8
	; 6-5 butterfly
	MOV	r14,r14,ASR #16		; r14= t2[5]
	ADD	r10,r14,r10,ASR #16	; r10= t3[6] = t[6] + t[5]
	SUB	r14,r10,r14,LSL #1	; r14= t3[5] = t[6] - t[5]
	; r2=t2[0] r3=t2[2] r4=t2[3] r6=t2[1] r8=t2[7] r9=t2[4]
	; r10=t3[6] r14=t3[5]
	; Stage 4
	ADD	r2, r2, r8		; r2 = t[0] + t[7] + 8
	ADD	r6, r6, r10		; r6 = t[1] + t[6] + 8
	ADD	r3, r3, r14		; r3 = t[2] + t[5] + 8
	ADD	r4, r4, r9		; r4 = t[3] + t[4] + 8
	SUB	r8, r2, r8, LSL #1	; r8 = t[0] - t[7] + 8
	SUB	r10,r6, r10,LSL #1	; r10= t[1] - t[6] + 8
	SUB	r14,r3, r14,LSL #1	; r14= t[2] - t[5] + 8
	SUB	r9, r4, r9, LSL #1	; r9 = t[3] - t[4] + 8
	; TODO: This is wrong.
	; The C code truncates to 16 bits by storing to RAM and doing the
	;  shifts later; we've got an extra 4 bits here.
	MOV	r2, r2, ASR #4
	MOV	r6, r6, ASR #4
	MOV	r3, r3, ASR #4
	MOV	r4, r4, ASR #4
	MOV	r8, r8, ASR #4
	MOV	r10,r10,ASR #4
	MOV	r14,r14,ASR #4
	MOV	r9, r9, ASR #4
	STRH	r2, [r0], #2		; y[0] = t[0]+t[7]
	STRH	r6, [r0, #14]		; y[1] = t[1]+t[6]
	STRH	r3, [r0, #30]		; y[2] = t[2]+t[5]
	STRH	r4, [r0, #46]		; y[3] = t[3]+t[4]
	STRH	r9, [r0, #62]		; y[4] = t[3]-t[4]
	STRH	r14,[r0, #78]		; y[5] = t[2]-t[5]
	STRH	r10,[r0, #94]		; y[6] = t[1]-t[6]
	STRH	r8, [r0, #110]		; y[7] = t[0]-t[7]
	LDMFD	r13!,{r1,PC}
	ENDP

 [ OC_ARM_ASM_MEDIA
	EXPORT	oc_idct8x8_1_v6
	EXPORT	oc_idct8x8_v6

oc_idct8x8_1_v6 PROC
	; r0 = ogg_int16_t  *_y
	; r1 = ogg_uint16_t  _dc
	ORR	r2, r1, r1, LSL #16
	ORR	r3, r1, r1, LSL #16
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	STRD	r2, [r0], #8
	MOV	PC, r14
	ENDP

oc_idct8x8_v6 PROC
	; r0 = ogg_int16_t *_y
	; r1 = ogg_int16_t *_x
	; r2 = int          _last_zzi
	CMP	r2, #3
	BLE	oc_idct8x8_3_v6
	;CMP	r2, #6
	;BLE	oc_idct8x8_6_v6
	CMP	r2, #10
	BLE	oc_idct8x8_10_v6
oc_idct8x8_slow_v6
	STMFD	r13!,{r4-r11,r14}
	SUB	r13,r13,#64*2
; Row transforms
	STR	r0, [r13,#-4]!
	ADD	r0, r13, #4	; Write to temp storage.
	BL	idct8_8core_v6
	BL	idct8_8core_v6
	BL	idct8_8core_v6
	BL	idct8_8core_v6
	LDR	r0, [r13], #4	; Write to the final destination.
	; Clear input data for next block.
	MOV	r4, #0
	MOV	r5, #0
	STRD	r4, [r1,#-8*16]!
	STRD	r4, [r1,#8]
	STRD	r4, [r1,#16]
	STRD	r4, [r1,#24]
	STRD	r4, [r1,#32]
	STRD	r4, [r1,#40]
	STRD	r4, [r1,#48]
	STRD	r4, [r1,#56]
	STRD	r4, [r1,#64]
	STRD	r4, [r1,#72]
	STRD	r4, [r1,#80]
	STRD	r4, [r1,#88]
	STRD	r4, [r1,#96]
	STRD	r4, [r1,#104]
	STRD	r4, [r1,#112]
	STRD	r4, [r1,#120]
	MOV	r1, r13		; And read from temp storage.
; Column transforms
	BL	idct8_8core_down_v6
	BL	idct8_8core_down_v6
	BL	idct8_8core_down_v6
	BL	idct8_8core_down_v6
	ADD	r13,r13,#64*2
	LDMFD	r13!,{r4-r11,PC}
	ENDP

oc_idct8x8_10_v6 PROC
	STMFD	r13!,{r4-r11,r14}
	SUB	r13,r13,#64*2+4
; Row transforms
	MOV	r2, r13
	STR	r0, [r13,#-4]!
	AND	r0, r2, #4	; Align the stack.
	ADD	r0, r0, r2	; Write to temp storage.
	BL	idct4_3core_v6
	BL	idct2_1core_v6
	LDR	r0, [r13], #4	; Write to the final destination.
	; Clear input data for next block.
	MOV	r4, #0
	MOV	r5, #0
	STRD	r4, [r1,#-4*16]!
	STRD	r4, [r1,#16]
	STR	r4, [r1,#32]
	STR	r4, [r1,#48]
	AND	r1, r13,#4	; Align the stack.
	ADD	r1, r1, r13	; And read from temp storage.
; Column transforms
	BL	idct4_4core_down_v6
	BL	idct4_4core_down_v6
	BL	idct4_4core_down_v6
	BL	idct4_4core_down_v6
	ADD	r13,r13,#64*2+4
	LDMFD	r13!,{r4-r11,PC}
	ENDP

oc_idct8x8_3_v6 PROC
	STMFD	r13!,{r4-r8,r14}
	SUB	r13,r13,#64*2
; Row transforms
	MOV	r8, r0
	MOV	r0, r13		; Write to temp storage.
	BL	idct2_1core_v6
	; Clear input data for next block.
	MOV	r4, #0
	STR	r4, [r1,#-2*16]!
	STR	r4, [r1,#16]
	MOV	r1, r13		; Read from temp storage.
	MOV	r0, r8		; Write to the final destination.
; Column transforms
	BL	idct2_2core_down_v6
	BL	idct2_2core_down_v6
	BL	idct2_2core_down_v6
	BL	idct2_2core_down_v6
	ADD	r13,r13,#64*2
	LDMFD	r13!,{r4-r8,PC}
	ENDP

idct2_1core_v6 PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
; Stage 1:
	LDR	r2, [r1], #16		; r2 = <x[0,1]|x[0,0]>
	LDR	r3, OC_C4S4
	LDRSH	r6, [r1], #16		; r6 = x[1,0]
	SMULWB	r12,r3, r2		; r12= t[0,0]=OC_C4S4*x[0,0]>>16
	LDRD	r4, OC_C7S1		; r4 = OC_C7S1; r5 = OC_C1S7
	SMULWB	r6, r3, r6		; r6 = t[1,0]=OC_C4S4*x[1,0]>>16
	SMULWT	r4, r4, r2		; r4 = t[0,4]=OC_C7S1*x[0,1]>>16
	SMULWT	r7, r5, r2		; r7 = t[0,7]=OC_C1S7*x[0,1]>>16
; Stage 2:
	SMULWB	r5, r3, r4		; r5 = t[0,5]=OC_C4S4*t[0,4]>>16
	PKHBT	r12,r12,r6, LSL #16	; r12= <t[1,0]|t[0,0]>
	SMULWB	r6, r3, r7		; r6 = t[0,6]=OC_C4S4*t[0,7]>>16
	PKHBT	r7, r7, r3		; r7 = <0|t[0,7]>
; Stage 3:
	PKHBT	r5, r6, r5, LSL #16	; r5 = <t[0,5]|t[0,6]>
	PKHBT	r4, r4, r3		; r4 = <0|t[0,4]>
	SASX	r5, r5, r5		; r5 = <t[0,6]+t[0,5]|t[0,6]-t[0,5]>
; Stage 4:
	PKHTB	r6, r3, r5, ASR #16	; r6 = <0|t[0,6]>
	PKHBT	r5, r5, r3		; r5 = <0|t[0,5]>
	SADD16	r3, r12,r7		; r3 = t[0]+t[7]
	STR	r3, [r0], #4		; y[0<<3] = t[0]+t[7]
	SADD16	r3, r12,r6		; r3 = t[0]+t[6]
	STR	r3, [r0, #12]		; y[1<<3] = t[0]+t[6]
	SADD16	r3, r12,r5		; r3 = t[0]+t[5]
	STR	r3, [r0, #28]		; y[2<<3] = t[0]+t[5]
	SADD16	r3, r12,r4		; r3 = t[0]+t[4]
	STR	r3, [r0, #44]		; y[3<<3] = t[0]+t[4]
	SSUB16	r4, r12,r4		; r4 = t[0]-t[4]
	STR	r4, [r0, #60]		; y[4<<3] = t[0]-t[4]
	SSUB16	r5, r12,r5		; r5 = t[0]-t[5]
	STR	r5, [r0, #76]		; y[5<<3] = t[0]-t[5]
	SSUB16	r6, r12,r6		; r6 = t[0]-t[6]
	STR	r6, [r0, #92]		; y[6<<3] = t[0]-t[6]
	SSUB16	r7, r12,r7		; r7 = t[0]-t[7]
	STR	r7, [r0, #108]		; y[7<<3] = t[0]-t[7]
	MOV	PC,r14
	ENDP
 ]

	ALIGN 8
OC_C7S1
	DCD	12785 ; 31F1
OC_C1S7
	DCD	64277 ; FB15
OC_C6S2
	DCD	25080 ; 61F8
OC_C2S6
	DCD	60547 ; EC83
OC_C5S3
	DCD	36410 ; 8E3A
OC_C3S5
	DCD	54491 ; D4DB
OC_C4S4
	DCD	46341 ; B505

 [ OC_ARM_ASM_MEDIA
idct2_2core_down_v6 PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
; Stage 1:
	LDR	r2, [r1], #16		; r2 = <x[0,1]|x[0,0]>
	LDR	r3, OC_C4S4
	MOV	r7 ,#8			; r7  = 8
	LDR	r6, [r1], #16		; r6 = <x[1,1]|x[1,0]>
	SMLAWB	r12,r3, r2, r7		; r12= (t[0,0]=OC_C4S4*x[0,0]>>16)+8
	LDRD	r4, OC_C7S1		; r4 = OC_C7S1; r5 = OC_C1S7
	SMLAWB	r7, r3, r6, r7		; r7 = (t[1,0]=OC_C4S4*x[1,0]>>16)+8
	SMULWT  r5, r5, r2		; r2 = t[0,7]=OC_C1S7*x[0,1]>>16
	PKHBT	r12,r12,r7, LSL #16	; r12= <t[1,0]+8|t[0,0]+8>
	SMULWT	r4, r4, r2		; r4 = t[0,4]=OC_C7S1*x[0,1]>>16
; Here we cheat: row 1 had just a DC, so x[0,1]==x[1,1] by definition.
	PKHBT	r7, r5, r5, LSL #16	; r7 = <t[0,7]|t[0,7]>
; Stage 2:
	SMULWB	r6, r3, r7		; r6 = t[0,6]=OC_C4S4*t[0,7]>>16
	PKHBT	r4, r4, r4, LSL #16	; r4 = <t[0,4]|t[0,4]>
	SMULWT	r2, r3, r7		; r2 = t[1,6]=OC_C4S4*t[1,7]>>16
	SMULWB	r5, r3, r4		; r5 = t[0,5]=OC_C4S4*t[0,4]>>16
	PKHBT	r6, r6, r2, LSL #16	; r6 = <t[1,6]|t[0,6]>
	SMULWT	r2, r3, r4		; r2 = t[1,5]=OC_C4S4*t[1,4]>>16
	PKHBT	r2, r5, r2, LSL #16	; r2 = <t[1,5]|t[0,5]>
; Stage 3:
	SSUB16	r5, r6, r2		; r5 = <t[1,6]-t[1,5]|t[0,6]-t[0,5]>
	SADD16	r6, r6, r2		; r6 = <t[1,6]+t[1,5]|t[0,6]+t[0,5]>
; Stage 4:
	SADD16	r2, r12,r7		; r2 = t[0]+t[7]+8
	MOV	r3, r2, ASR #4
	MOV	r2, r2, LSL #16
	PKHTB	r3, r3, r2, ASR #20	; r3 = t[0]+t[7]+8>>4
	STR	r3, [r0], #4		; y[0<<3] = t[0]+t[7]+8>>4
	SADD16	r2, r12,r6		; r2 = t[0]+t[6]+8
	MOV	r3, r2, ASR #4
	MOV	r2, r2, LSL #16
	PKHTB	r3, r3, r2, ASR #20	; r3 = t[0]+t[6]+8>>4
	STR	r3, [r0, #12]		; y[1<<3] = t[0]+t[6]+8>>4
	SADD16	r2, r12,r5		; r2 = t[0]+t[5]+8
	MOV	r3, r2, ASR #4
	MOV	r2, r2, LSL #16
	PKHTB	r3, r3, r2, ASR #20	; r3 = t[0]+t[5]+8>>4
	STR	r3, [r0, #28]		; y[2<<3] = t[0]+t[5]+8>>4
	SADD16	r2, r12,r4		; r2 = t[0]+t[4]+8
	MOV	r3, r2, ASR #4
	MOV	r2, r2, LSL #16
	PKHTB	r3, r3, r2, ASR #20	; r3 = t[0]+t[4]+8>>4
	STR	r3, [r0, #44]		; y[3<<3] = t[0]+t[4]+8>>4
	SSUB16	r4, r12,r4		; r4 = t[0]-t[4]+8
	MOV	r3, r4, ASR #4
	MOV	r4, r4, LSL #16
	PKHTB	r3, r3, r4, ASR #20	; r3 = t[0]-t[4]+8>>4
	STR	r3, [r0, #60]		; y[4<<3] = t[0]-t[4]+8>>4
	SSUB16	r5, r12,r5		; r5 = t[0]-t[5]+8
	MOV	r3, r5, ASR #4
	MOV	r5, r5, LSL #16
	PKHTB	r3, r3, r5, ASR #20	; r3 = t[0]-t[5]+8>>4
	STR	r3, [r0, #76]		; y[5<<3] = t[0]-t[5]+8>>4
	SSUB16	r6, r12,r6		; r6 = t[0]-t[6]+8
	MOV	r3, r6, ASR #4
	MOV	r6, r6, LSL #16
	PKHTB	r3, r3, r6, ASR #20	; r3 = t[0]-t[6]+8>>4
	STR	r3, [r0, #92]		; y[6<<3] = t[0]-t[6]+8>>4
	SSUB16	r7, r12,r7		; r7 = t[0]-t[7]+8
	MOV	r3, r7, ASR #4
	MOV	r7, r7, LSL #16
	PKHTB	r3, r3, r7, ASR #20	; r3 = t[0]-t[7]+8>>4
	STR	r3, [r0, #108]		; y[7<<3] = t[0]-t[7]+8>>4
	MOV	PC,r14
	ENDP

; In theory this should save ~75 cycles over oc_idct8x8_10, more than enough to
;  pay for increased branch mis-prediction to get here, but in practice it
;  doesn't seem to slow anything down to take it out, and it's less code this
;  way.
 [ 0
oc_idct8x8_6_v6 PROC
	STMFD	r13!,{r4-r8,r10,r11,r14}
	SUB	r13,r13,#64*2+4
; Row transforms
	MOV	r8, r0
	AND	r0, r13,#4	; Align the stack.
	ADD	r0, r0, r13	; Write to temp storage.
	BL	idct3_2core_v6
	BL	idct1core_v6
	; Clear input data for next block.
	MOV	r4, #0
	MOV	r5, #0
	STRD	r4, [r1,#-3*16]!
	STR	r4, [r1,#16]
	STR	r4, [r1,#32]
	AND	r1, r13,#4	; Align the stack.
	MOV	r0, r8		; Write to the final destination.
	ADD	r1, r1, r13	; And read from temp storage.
; Column transforms
	BL	idct3_3core_down_v6
	BL	idct3_3core_down_v6
	BL	idct3_3core_down_v6
	BL	idct3_3core_down_v6
	ADD	r13,r13,#64*2+4
	LDMFD	r13!,{r4-r8,r10,r11,PC}
	ENDP

idct1core_v6 PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
	LDRSH	r3, [r1], #16
	MOV	r12,#0x05
	ORR	r12,r12,#0xB500
	MUL	r3, r12, r3
	; Stall ?
	MOV	r3, r3, ASR #16
	; Don't need to actually store the odd lines; they won't be read.
	STRH	r3, [r0], #2
	STRH	r3, [r0, #30]
	STRH	r3, [r0, #62]
	STRH	r3, [r0, #94]
	MOV	PC,R14
	ENDP

idct3_2core_v6 PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
; Stage 1:
	LDRD	r4, [r1], #16		; r4 = <x[0,1]|x[0,0]>; r5 = <*|x[0,2]>
	LDRD	r10,OC_C6S2_3_v6	; r10= OC_C6S2; r11= OC_C2S6
	; Stall
	SMULWB	r3, r11,r5		; r3 = t[0,3]=OC_C2S6*x[0,2]>>16
	LDR	r11,OC_C4S4
	SMULWB	r2, r10,r5		; r2 = t[0,2]=OC_C6S2*x[0,2]>>16
	LDR	r5, [r1], #16		; r5 = <x[1,1]|x[1,0]>
	SMULWB	r12,r11,r4		; r12= (t[0,0]=OC_C4S4*x[0,0]>>16)
	LDRD	r6, OC_C7S1_3_v6	; r6 = OC_C7S1; r7 = OC_C1S7
	SMULWB	r10,r11,r5		; r10= (t[1,0]=OC_C4S4*x[1,0]>>16)
	PKHBT	r12,r12,r10,LSL #16	; r12= <t[1,0]|t[0,0]>
	SMULWT  r10,r7, r5		; r10= t[1,7]=OC_C1S7*x[1,1]>>16
	PKHBT	r2, r2, r11		; r2 = <0|t[0,2]>
	SMULWT  r7, r7, r4		; r7 = t[0,7]=OC_C1S7*x[0,1]>>16
	PKHBT	r3, r3, r11		; r3 = <0|t[0,3]>
	SMULWT	r5, r6, r5		; r10= t[1,4]=OC_C7S1*x[1,1]>>16
	PKHBT	r7, r7, r10,LSL #16	; r7 = <t[1,7]|t[0,7]>
	SMULWT	r4, r6, r4		; r4 = t[0,4]=OC_C7S1*x[0,1]>>16
; Stage 2:
	SMULWB	r6, r11,r7		; r6 = t[0,6]=OC_C4S4*t[0,7]>>16
	PKHBT	r4, r4, r5, LSL #16	; r4 = <t[1,4]|t[0,4]>
	SMULWT	r10,r11,r7		; r10= t[1,6]=OC_C4S4*t[1,7]>>16
	SMULWB	r5, r11,r4		; r5 = t[0,5]=OC_C4S4*t[0,4]>>16
	PKHBT	r6, r6, r10,LSL #16	; r6 = <t[1,6]|t[0,6]>
	SMULWT	r10,r11,r4		; r10= t[1,5]=OC_C4S4*t[1,4]>>16
; Stage 3:
	B	idct4_3core_stage3_v6
	ENDP

; Another copy so the LDRD offsets are less than +/- 255.
	ALIGN 8
OC_C7S1_3_v6
	DCD	12785 ; 31F1
OC_C1S7_3_v6
	DCD	64277 ; FB15
OC_C6S2_3_v6
	DCD	25080 ; 61F8
OC_C2S6_3_v6
	DCD	60547 ; EC83

idct3_3core_down_v6 PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
; Stage 1:
	LDRD	r10,[r1], #16		; r10= <x[0,1]|x[0,0]>; r11= <??|x[0,2]>
	LDRD	r6, OC_C6S2_3_v6	; r6 = OC_C6S2; r7 = OC_C2S6
	LDR	r4, [r1], #16		; r4 = <x[1,1]|x[1,0]>
	SMULWB	r3, r7, r11		; r3 = t[0,3]=OC_C2S6*x[0,2]>>16
	MOV	r7,#8
	SMULWB	r2, r6, r11		; r2 = t[0,2]=OC_C6S2*x[0,2]>>16
	LDR	r11,OC_C4S4
	SMLAWB	r12,r11,r10,r7		; r12= t[0,0]+8=(OC_C4S4*x[0,0]>>16)+8
; Here we cheat: row 2 had just a DC, so x[0,2]==x[1,2] by definition.
	PKHBT	r3, r3, r3, LSL #16	; r3 = <t[0,3]|t[0,3]>
	SMLAWB	r5, r11,r4, r7		; r5 = t[1,0]+8=(OC_C4S4*x[1,0]>>16)+8
	PKHBT	r2, r2, r2, LSL #16	; r2 = <t[0,2]|t[0,2]>
	LDRD	r6, OC_C7S1_3_v6	; r6 = OC_C7S1; r7 = OC_C1S7
	PKHBT	r12,r12,r5, LSL #16	; r12= <t[1,0]+8|t[0,0]+8>
	SMULWT  r5, r7, r4		; r5 = t[1,7]=OC_C1S7*x[1,1]>>16
	SMULWT  r7, r7, r10		; r7 = t[0,7]=OC_C1S7*x[0,1]>>16
	SMULWT	r10,r6, r10		; r10= t[0,4]=OC_C7S1*x[0,1]>>16
	PKHBT	r7, r7, r5, LSL #16	; r7 = <t[1,7]|t[0,7]>
	SMULWT	r4, r6, r4		; r4 = t[1,4]=OC_C7S1*x[1,1]>>16
; Stage 2:
	SMULWB	r6, r11,r7		; r6 = t[0,6]=OC_C4S4*t[0,7]>>16
	PKHBT	r4, r10,r4, LSL #16	; r4 = <t[1,4]|t[0,4]>
	SMULWT	r10,r11,r7		; r10= t[1,6]=OC_C4S4*t[1,7]>>16
	SMULWB	r5, r11,r4		; r5 = t[0,5]=OC_C4S4*t[0,4]>>16
	PKHBT	r6, r6, r10,LSL #16	; r6 = <t[1,6]|t[0,6]>
	SMULWT	r10,r11,r4		; r10= t[1,5]=OC_C4S4*t[1,4]>>16
; Stage 3:
	B	idct4_4core_down_stage3_v6
	ENDP
 ]

idct4_3core_v6 PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
; Stage 1:
	LDRD	r10,[r1], #16	; r10= <x[0,1]|x[0,0]>; r11= <x[0,3]|x[0,2]>
	LDRD	r2, OC_C5S3_4_v6	; r2 = OC_C5S3; r3 = OC_C3S5
	LDRD	r4, [r1], #16		; r4 = <x[1,1]|x[1,0]>; r5 = <??|x[1,2]>
	SMULWT	r9, r3, r11		; r9 = t[0,6]=OC_C3S5*x[0,3]>>16
	SMULWT	r8, r2, r11		; r8 = -t[0,5]=OC_C5S3*x[0,3]>>16
	PKHBT	r9, r9, r2		; r9 = <0|t[0,6]>
	LDRD	r6, OC_C6S2_4_v6	; r6 = OC_C6S2; r7 = OC_C2S6
	PKHBT	r8, r8, r2		; r9 = <0|-t[0,5]>
	SMULWB	r3, r7, r11		; r3 = t[0,3]=OC_C2S6*x[0,2]>>16
	SMULWB	r2, r6, r11		; r2 = t[0,2]=OC_C6S2*x[0,2]>>16
	LDR	r11,OC_C4S4
	SMULWB	r12,r7, r5		; r12= t[1,3]=OC_C2S6*x[1,2]>>16
	SMULWB	r5, r6, r5		; r5 = t[1,2]=OC_C6S2*x[1,2]>>16
	PKHBT	r3, r3, r12,LSL #16	; r3 = <t[1,3]|t[0,3]>
	SMULWB	r12,r11,r10		; r12= t[0,0]=OC_C4S4*x[0,0]>>16
	PKHBT	r2, r2, r5, LSL #16	; r2 = <t[1,2]|t[0,2]>
	SMULWB	r5, r11,r4		; r5 = t[1,0]=OC_C4S4*x[1,0]>>16
	LDRD	r6, OC_C7S1_4_v6	; r6 = OC_C7S1; r7 = OC_C1S7
	PKHBT	r12,r12,r5, LSL #16	; r12= <t[1,0]|t[0,0]>
	SMULWT  r5, r7, r4		; r5 = t[1,7]=OC_C1S7*x[1,1]>>16
	SMULWT  r7, r7, r10		; r7 = t[0,7]=OC_C1S7*x[0,1]>>16
	SMULWT	r10,r6, r10		; r10= t[0,4]=OC_C7S1*x[0,1]>>16
	PKHBT	r7, r7, r5, LSL #16	; r7 = <t[1,7]|t[0,7]>
	SMULWT	r4, r6, r4		; r4 = t[1,4]=OC_C7S1*x[1,1]>>16
; Stage 2:
	SSUB16	r6, r7, r9		; r6 = t[7]-t[6]
	PKHBT	r4, r10,r4, LSL #16	; r4 = <t[1,4]|t[0,4]>
	SADD16	r7, r7, r9		; r7 = t[7]=t[7]+t[6]
	SMULWT	r9, r11,r6		; r9 = t[1,6]=OC_C4S4*r6T>>16
	SADD16	r5, r4, r8		; r5 = t[4]-t[5]
	SMULWB	r6, r11,r6		; r6 = t[0,6]=OC_C4S4*r6B>>16
	SSUB16	r4, r4, r8		; r4 = t[4]=t[4]+t[5]
	SMULWT	r10,r11,r5		; r10= t[1,5]=OC_C4S4*r5T>>16
	PKHBT	r6, r6, r9, LSL #16	; r6 = <t[1,6]|t[0,6]>
	SMULWB	r5, r11,r5		; r5 = t[0,5]=OC_C4S4*r5B>>16
; Stage 3:
idct4_3core_stage3_v6
	SADD16	r11,r12,r2		; r11= t[1]=t[0]+t[2]
	PKHBT	r10,r5, r10,LSL #16	; r10= <t[1,5]|t[0,5]>
	SSUB16	r2, r12,r2		; r2 = t[2]=t[0]-t[2]
idct4_3core_stage3_5_v6
	SSUB16	r5, r6, r10		; r5 = t[5]'=t[6]-t[5]
	SADD16	r6, r6, r10		; r6 = t[6]=t[6]+t[5]
	SADD16	r10,r12,r3		; r10= t[0]'=t[0]+t[3]
	SSUB16	r3, r12,r3		; r3 = t[3]=t[0]-t[3]
; Stage 4:
	SADD16	r12,r10,r7		; r12= t[0]+t[7]
	STR	r12,[r0], #4		; y[0<<3] = t[0]+t[7]
	SADD16	r12,r11,r6		; r12= t[1]+t[6]
	STR	r12,[r0, #12]		; y[1<<3] = t[1]+t[6]
	SADD16	r12,r2, r5		; r12= t[2]+t[5]
	STR	r12,[r0, #28]		; y[2<<3] = t[2]+t[5]
	SADD16	r12,r3, r4		; r12= t[3]+t[4]
	STR	r12,[r0, #44]		; y[3<<3] = t[3]+t[4]
	SSUB16	r4, r3, r4		; r4 = t[3]-t[4]
	STR	r4, [r0, #60]		; y[4<<3] = t[3]-t[4]
	SSUB16	r5, r2, r5		; r5 = t[2]-t[5]
	STR	r5, [r0, #76]		; y[5<<3] = t[2]-t[5]
	SSUB16	r6, r11,r6		; r6 = t[1]-t[6]
	STR	r6, [r0, #92]		; y[6<<3] = t[1]-t[6]
	SSUB16	r7, r10,r7		; r7 = t[0]-t[7]
	STR	r7, [r0, #108]		; y[7<<3] = t[0]-t[7]
	MOV	PC,r14
	ENDP

; Another copy so the LDRD offsets are less than +/- 255.
	ALIGN 8
OC_C7S1_4_v6
	DCD	12785 ; 31F1
OC_C1S7_4_v6
	DCD	64277 ; FB15
OC_C6S2_4_v6
	DCD	25080 ; 61F8
OC_C2S6_4_v6
	DCD	60547 ; EC83
OC_C5S3_4_v6
	DCD	36410 ; 8E3A
OC_C3S5_4_v6
	DCD	54491 ; D4DB

idct4_4core_down_v6 PROC
	; r0 =       ogg_int16_t *_y (destination)
	; r1 = const ogg_int16_t *_x (source)
; Stage 1:
	LDRD	r10,[r1], #16	; r10= <x[0,1]|x[0,0]>; r11= <x[0,3]|x[0,2]>
	LDRD	r2, OC_C5S3_4_v6	; r2 = OC_C5S3; r3 = OC_C3S5
	LDRD	r4, [r1], #16	; r4 = <x[1,1]|x[1,0]>; r5 = <x[1,3]|x[1,2]>
	SMULWT	r9, r3, r11		; r9 = t[0,6]=OC_C3S5*x[0,3]>>16
	LDRD	r6, OC_C6S2_4_v6	; r6 = OC_C6S2; r7 = OC_C2S6
	SMULWT	r8, r2, r11		; r8 = -t[0,5]=OC_C5S3*x[0,3]>>16
; Here we cheat: row 3 had just a DC, so x[0,3]==x[1,3] by definition.
	PKHBT	r9, r9, r9, LSL #16	; r9 = <t[0,6]|t[0,6]>
	SMULWB	r3, r7, r11		; r3 = t[0,3]=OC_C2S6*x[0,2]>>16
	PKHBT	r8, r8, r8, LSL #16	; r8 = <-t[0,5]|-t[0,5]>
	SMULWB	r2, r6, r11		; r2 = t[0,2]=OC_C6S2*x[0,2]>>16
	LDR	r11,OC_C4S4
	SMULWB	r12,r7, r5		; r12= t[1,3]=OC_C2S6*x[1,2]>>16
	MOV	r7,#8
	SMULWB	r5, r6, r5		; r5 = t[1,2]=OC_C6S2*x[1,2]>>16
	PKHBT	r3, r3, r12,LSL #16	; r3 = <t[1,3]|t[0,3]>
	SMLAWB	r12,r11,r10,r7		; r12= t[0,0]+8=(OC_C4S4*x[0,0]>>16)+8
	PKHBT	r2, r2, r5, LSL #16	; r2 = <t[1,2]|t[0,2]>
	SMLAWB	r5, r11,r4 ,r7		; r5 = t[1,0]+8=(OC_C4S4*x[1,0]>>16)+8
	LDRD	r6, OC_C7S1_4_v6	; r6 = OC_C7S1; r7 = OC_C1S7
	PKHBT	r12,r12,r5, LSL #16	; r12= <t[1,0]+8|t[0,0]+8>
	SMULWT  r5, r7, r4		; r5 = t[1,7]=OC_C1S7*x[1,1]>>16
	SMULWT  r7, r7, r10		; r7 = t[0,7]=OC_C1S7*x[0,1]>>16
	SMULWT	r10,r6, r10		; r10= t[0,4]=OC_C7S1*x[0,1]>>16
	PKHBT	r7, r7, r5, LSL #16	; r7 = <t[1,7]|t[0,7]>
	SMULWT	r4, r6, r4		; r4 = t[1,4]=OC_C7S1*x[1,1]>>16
; Stage 2:
	SSUB16	r6, r7, r9		; r6 = t[7]-t[6]
	PKHBT	r4, r10,r4, LSL #16	; r4 = <t[1,4]|t[0,4]>
	SADD16	r7, r7, r9		; r7 = t[7]=t[7]+t[6]
	SMULWT	r9, r11,r6		; r9 = t[1,6]=OC_C4S4*r6T>>16
	SADD16	r5, r4, r8		; r5 = t[4]-t[5]
	SMULWB	r6, r11,r6		; r6 = t[0,6]=OC_C4S4*r6B>>16
	SSUB16	r4, r4, r8		; r4 = t[4]=t[4]+t[5]
	SMULWT	r10,r11,r5		; r10= t[1,5]=OC_C4S4*r5T>>16
	PKHBT	r6, r6, r9, LSL #16	; r6 = <t[1,6]|t[0,6]>
	SMULWB	r5, r11,r5		; r5 = t[0,5]=OC_C4S4*r5B>>16
; Stage 3:
idct4_4core_down_stage3_v6
	SADD16	r11,r12,r2		; r11= t[1]+8=t[0]+t[2]+8
	PKHBT	r10,r5, r10,LSL #16	; r10= <t[1,5]|t[0,5]>
	SSUB16	r2, r12,r2		; r2 = t[2]+8=t[0]-t[2]+8
	B	idct8_8core_down_stage3_5_v6
	ENDP

idct8_8core_v6 PROC
	STMFD	r13!,{r0,r14}
; Stage 1:
	;5-6 rotation by 3pi/16
	LDRD	r10,OC_C5S3_4_v6	; r10= OC_C5S3, r11= OC_C3S5
	LDR	r4, [r1,#8]		; r4 = <x[0,5]|x[0,4]>
	LDR	r7, [r1,#24]		; r7 = <x[1,5]|x[1,4]>
	SMULWT	r5, r11,r4		; r5 = OC_C3S5*x[0,5]>>16
	LDR	r0, [r1,#4]		; r0 = <x[0,3]|x[0,2]>
	SMULWT	r3, r11,r7		; r3 = OC_C3S5*x[1,5]>>16
	LDR	r12,[r1,#20]		; r12= <x[1,3]|x[1,2]>
	SMULWT	r6, r11,r0		; r6 = OC_C3S5*x[0,3]>>16
	SMULWT	r11,r11,r12		; r11= OC_C3S5*x[1,3]>>16
	SMLAWT	r6, r10,r4, r6		; r6 = t[0,6]=r6+(OC_C5S3*x[0,5]>>16)
	PKHBT	r5, r5, r3, LSL #16	; r5 = <r3|r5>
	SMLAWT	r11,r10,r7, r11		; r11= t[1,6]=r11+(OC_C5S3*x[1,5]>>16)
	PKHBT	r4, r4, r7, LSL #16	; r4 = <x[1,4]|x[0,4]>
	SMULWT	r3, r10,r0		; r3 = OC_C5S3*x[0,3]>>16
	PKHBT	r6, r6, r11,LSL #16	; r6 = <t[1,6]|t[0,6]>
	SMULWT	r8, r10,r12		; r8 = OC_C5S3*x[1,3]>>16
	;2-3 rotation by 6pi/16
	LDRD	r10,OC_C6S2_4_v6	; r10= OC_C6S2, r11= OC_C2S6
	PKHBT	r3, r3, r8, LSL #16	; r3 = <r8|r3>
	LDR	r8, [r1,#12]		; r8 = <x[0,7]|x[0,6]>
	SMULWB	r2, r10,r0		; r2 = OC_C6S2*x[0,2]>>16
	SSUB16	r5, r5, r3		; r5 = <t[1,5]|t[0,5]>
	SMULWB	r9, r10,r12		; r9 = OC_C6S2*x[1,2]>>16
	LDR	r7, [r1,#28]		; r7 = <x[1,7]|x[1,6]>
	SMULWB	r3, r10,r8		; r3 = OC_C6S2*x[0,6]>>16
	SMULWB	r10,r10,r7		; r10= OC_C6S2*x[1,6]>>16
	PKHBT	r2, r2, r9, LSL #16	; r2 = <r2|r9>
	SMLAWB	r3, r11,r0, r3		; r3 = t[0,3]=r3+(OC_C2S6*x[0,2]>>16)
	SMLAWB	r10,r11,r12,r10		; r10= t[1,3]=r10+(OC_C2S6*x[1,2]>>16)
	SMULWB	r9, r11,r8		; r9 = OC_C2S6*x[0,6]>>16
	PKHBT	r3, r3, r10,LSL #16	; r3 = <t[1,6]|t[0,6]>
	SMULWB	r12,r11,r7		; r12= OC_C2S6*x[1,6]>>16
	;4-7 rotation by 7pi/16
	LDRD	r10,OC_C7S1_8_v6	; r10= OC_C7S1, r11= OC_C1S7
	PKHBT	r9, r9, r12,LSL #16	; r9 = <r9|r12>
	LDR	r0, [r1],#16		; r0 = <x[0,1]|x[0,0]>
	PKHTB	r7, r7, r8, ASR #16	; r7 = <x[1,7]|x[0,7]>
	SSUB16	r2, r2, r9		; r2 = <t[1,2]|t[0,2]>
	SMULWB	r9, r10,r7		; r9 = OC_C7S1*x[0,7]>>16
	LDR	r14,[r1],#16		; r14= <x[1,1]|x[1,0]>
	SMULWT	r12,r10,r7		; r12= OC_C7S1*x[1,7]>>16
	SMULWT	r8, r10,r0		; r8 = OC_C7S1*x[0,1]>>16
	SMULWT	r10,r10,r14		; r10= OC_C7S1*x[1,1]>>16
	SMLAWT	r9, r11,r0, r9		; r9 = t[0,7]=r9+(OC_C1S7*x[0,1]>>16)
	PKHBT	r8, r8, r10,LSL #16	; r8 = <r12|r8>
	SMLAWT	r12,r11,r14,r12		; r12= t[1,7]=r12+(OC_C1S7*x[1,1]>>16)
	PKHBT	r0, r0, r14,LSL #16	; r0 = <x[1,0]|x[0,0]>
	SMULWB	r10,r11,r7		; r10= OC_C1S7*x[0,6]>>16
	PKHBT	r9, r9, r12,LSL #16	; r9 = <t[1,7]|t[0,7]>
	SMULWT	r12,r11,r7		; r12= OC_C1S7*x[1,6]>>16
	;0-1 butterfly
	LDR	r11,OC_C4S4
	PKHBT	r10,r10,r12,LSL #16	; r10= <r12|r10>
	SADD16	r7, r0, r4		; r7 = x[0]+x[4]
	SSUB16	r10,r8, r10		; r10= <t[1,4]|t[0,4]>
	SSUB16	r4, r0, r4		; r4 = x[0]-x[4]
	SMULWB	r8, r11,r7		; r8 = t[0,0]=OC_C4S4*r7B>>16
	SMULWT	r12,r11,r7		; r12= t[1,0]=OC_C4S4*r7T>>16
	SMULWB	r7, r11,r4		; r7 = t[0,1]=OC_C4S4*r4B>>16
	PKHBT	r12,r8, r12,LSL #16	; r12= <t[1,0]|t[0,0]>
	SMULWT	r8, r11,r4		; r8 = t[1,1]=OC_C4S4*r4T>>16
; Stage 2:
	SADD16	r4, r10,r5		; r4 = t[4]'=t[4]+t[5]
	PKHBT	r8, r7, r8, LSL #16	; r8 = <t[1,0]|t[0,0]>
	SSUB16	r5, r10,r5		; r5 = t[4]-t[5]
	SMULWB	r10,r11,r5		; r10= t[0,5]=OC_C4S4*r5B>>16
	SADD16	r7, r9, r6		; r7 = t[7]'=t[7]+t[6]
	SMULWT	r5, r11,r5		; r5 = t[1,5]=OC_C4S4*r5T>>16
	SSUB16	r6, r9, r6		; r6 = t[7]-t[6]
	SMULWB	r9, r11,r6		; r9 = t[0,6]=OC_C4S4*r6B>>16
	PKHBT	r10,r10,r5, LSL #16	; r10= <t[1,5]|t[0,5]>
	SMULWT	r6, r11,r6		; r6 = t[1,6]=OC_C4S4*r6T>>16
; Stage 3:
	SADD16	r11,r8, r2		; r11= t[1]'=t[1]+t[2]
	PKHBT	r6, r9, r6, LSL #16	; r6 = <t[1,6]|t[0,6]>
	SSUB16	r2, r8, r2		; r2 = t[2]=t[1]-t[2]
	LDMFD	r13!,{r0,r14}
	B	idct4_3core_stage3_5_v6
	ENDP

; Another copy so the LDRD offsets are less than +/- 255.
	ALIGN 8
OC_C7S1_8_v6
	DCD	12785 ; 31F1
OC_C1S7_8_v6
	DCD	64277 ; FB15
OC_C6S2_8_v6
	DCD	25080 ; 61F8
OC_C2S6_8_v6
	DCD	60547 ; EC83
OC_C5S3_8_v6
	DCD	36410 ; 8E3A
OC_C3S5_8_v6
	DCD	54491 ; D4DB

idct8_8core_down_v6 PROC
	STMFD	r13!,{r0,r14}
; Stage 1:
	;5-6 rotation by 3pi/16
	LDRD	r10,OC_C5S3_8_v6	; r10= OC_C5S3, r11= OC_C3S5
	LDR	r4, [r1,#8]		; r4 = <x[0,5]|x[0,4]>
	LDR	r7, [r1,#24]		; r7 = <x[1,5]|x[1,4]>
	SMULWT	r5, r11,r4		; r5 = OC_C3S5*x[0,5]>>16
	LDR	r0, [r1,#4]		; r0 = <x[0,3]|x[0,2]>
	SMULWT	r3, r11,r7		; r3 = OC_C3S5*x[1,5]>>16
	LDR	r12,[r1,#20]		; r12= <x[1,3]|x[1,2]>
	SMULWT	r6, r11,r0		; r6 = OC_C3S5*x[0,3]>>16
	SMULWT	r11,r11,r12		; r11= OC_C3S5*x[1,3]>>16
	SMLAWT	r6, r10,r4, r6		; r6 = t[0,6]=r6+(OC_C5S3*x[0,5]>>16)
	PKHBT	r5, r5, r3, LSL #16	; r5 = <r3|r5>
	SMLAWT	r11,r10,r7, r11		; r11= t[1,6]=r11+(OC_C5S3*x[1,5]>>16)
	PKHBT	r4, r4, r7, LSL #16	; r4 = <x[1,4]|x[0,4]>
	SMULWT	r3, r10,r0		; r3 = OC_C5S3*x[0,3]>>16
	PKHBT	r6, r6, r11,LSL #16	; r6 = <t[1,6]|t[0,6]>
	SMULWT	r8, r10,r12		; r8 = OC_C5S3*x[1,3]>>16
	;2-3 rotation by 6pi/16
	LDRD	r10,OC_C6S2_8_v6	; r10= OC_C6S2, r11= OC_C2S6
	PKHBT	r3, r3, r8, LSL #16	; r3 = <r8|r3>
	LDR	r8, [r1,#12]		; r8 = <x[0,7]|x[0,6]>
	SMULWB	r2, r10,r0		; r2 = OC_C6S2*x[0,2]>>16
	SSUB16	r5, r5, r3		; r5 = <t[1,5]|t[0,5]>
	SMULWB	r9, r10,r12		; r9 = OC_C6S2*x[1,2]>>16
	LDR	r7, [r1,#28]		; r7 = <x[1,7]|x[1,6]>
	SMULWB	r3, r10,r8		; r3 = OC_C6S2*x[0,6]>>16
	SMULWB	r10,r10,r7		; r10= OC_C6S2*x[1,6]>>16
	PKHBT	r2, r2, r9, LSL #16	; r2 = <r2|r9>
	SMLAWB	r3, r11,r0, r3		; r3 = t[0,3]=r3+(OC_C2S6*x[0,2]>>16)
	SMLAWB	r10,r11,r12,r10		; r10= t[1,3]=r10+(OC_C2S6*x[1,2]>>16)
	SMULWB	r9, r11,r8		; r9 = OC_C2S6*x[0,6]>>16
	PKHBT	r3, r3, r10,LSL #16	; r3 = <t[1,6]|t[0,6]>
	SMULWB	r12,r11,r7		; r12= OC_C2S6*x[1,6]>>16
	;4-7 rotation by 7pi/16
	LDRD	r10,OC_C7S1_8_v6	; r10= OC_C7S1, r11= OC_C1S7
	PKHBT	r9, r9, r12,LSL #16	; r9 = <r9|r12>
	LDR	r0, [r1],#16		; r0 = <x[0,1]|x[0,0]>
	PKHTB	r7, r7, r8, ASR #16	; r7 = <x[1,7]|x[0,7]>
	SSUB16	r2, r2, r9		; r2 = <t[1,2]|t[0,2]>
	SMULWB	r9, r10,r7		; r9 = OC_C7S1*x[0,7]>>16
	LDR	r14,[r1],#16		; r14= <x[1,1]|x[1,0]>
	SMULWT	r12,r10,r7		; r12= OC_C7S1*x[1,7]>>16
	SMULWT	r8, r10,r0		; r8 = OC_C7S1*x[0,1]>>16
	SMULWT	r10,r10,r14		; r10= OC_C7S1*x[1,1]>>16
	SMLAWT	r9, r11,r0, r9		; r9 = t[0,7]=r9+(OC_C1S7*x[0,1]>>16)
	PKHBT	r8, r8, r10,LSL #16	; r8 = <r12|r8>
	SMLAWT	r12,r11,r14,r12		; r12= t[1,7]=r12+(OC_C1S7*x[1,1]>>16)
	PKHBT	r0, r0, r14,LSL #16	; r0 = <x[1,0]|x[0,0]>
	SMULWB	r10,r11,r7		; r10= OC_C1S7*x[0,6]>>16
	PKHBT	r9, r9, r12,LSL #16	; r9 = <t[1,7]|t[0,7]>
	SMULWT	r12,r11,r7		; r12= OC_C1S7*x[1,6]>>16
	;0-1 butterfly
	LDR	r11,OC_C4S4
	MOV	r14,#8
	PKHBT	r10,r10,r12,LSL #16	; r10= <r12|r10>
	SADD16	r7, r0, r4		; r7 = x[0]+x[4]
	SSUB16	r10,r8, r10		; r10= <t[1,4]|t[0,4]>
	SMLAWB	r8, r11,r7, r14		; r8 = t[0,0]+8=(OC_C4S4*r7B>>16)+8
	SSUB16	r4, r0, r4		; r4 = x[0]-x[4]
	SMLAWT	r12,r11,r7, r14		; r12= t[1,0]+8=(OC_C4S4*r7T>>16)+8
	SMLAWB	r7, r11,r4, r14		; r7 = t[0,1]+8=(OC_C4S4*r4B>>16)+8
	PKHBT	r12,r8, r12,LSL #16	; r12= <t[1,0]+8|t[0,0]+8>
	SMLAWT	r8, r11,r4, r14		; r8 = t[1,1]+8=(OC_C4S4*r4T>>16)+8
; Stage 2:
	SADD16	r4, r10,r5		; r4 = t[4]'=t[4]+t[5]
	PKHBT	r8, r7, r8, LSL #16	; r8 = <t[1,0]+8|t[0,0]+8>
	SSUB16	r5, r10,r5		; r5 = t[4]-t[5]
	SMULWB	r10,r11,r5		; r10= t[0,5]=OC_C4S4*r5B>>16
	SADD16	r7, r9, r6		; r7 = t[7]'=t[7]+t[6]
	SMULWT	r5, r11,r5		; r5 = t[1,5]=OC_C4S4*r5T>>16
	SSUB16	r6, r9, r6		; r6 = t[7]-t[6]
	SMULWB	r9, r11,r6		; r9 = t[0,6]=OC_C4S4*r6B>>16
	PKHBT	r10,r10,r5, LSL #16	; r10= <t[1,5]|t[0,5]>
	SMULWT	r6, r11,r6		; r6 = t[1,6]=OC_C4S4*r6T>>16
; Stage 3:
	SADD16	r11,r8, r2		; r11= t[1]'+8=t[1]+t[2]+8
	PKHBT	r6, r9, r6, LSL #16	; r6 = <t[1,6]|t[0,6]>
	SSUB16	r2, r8, r2		; r2 = t[2]+8=t[1]-t[2]+8
	LDMFD	r13!,{r0,r14}
idct8_8core_down_stage3_5_v6
	SSUB16	r5, r6, r10		; r5 = t[5]'=t[6]-t[5]
	SADD16	r6, r6, r10		; r6 = t[6]=t[6]+t[5]
	SADD16	r10,r12,r3		; r10= t[0]'+8=t[0]+t[3]+8
	SSUB16	r3, r12,r3		; r3 = t[3]+8=t[0]-t[3]+8
; Stage 4:
	SADD16	r12,r10,r7		; r12= t[0]+t[7]+8
	SSUB16	r7, r10,r7		; r7 = t[0]-t[7]+8
	MOV	r10,r12,ASR #4
	MOV	r12,r12,LSL #16
	PKHTB	r10,r10,r12,ASR #20	; r10= t[0]+t[7]+8>>4
	STR	r10,[r0], #4		; y[0<<3] = t[0]+t[7]+8>>4
	SADD16	r12,r11,r6		; r12= t[1]+t[6]+8
	SSUB16	r6, r11,r6		; r6 = t[1]-t[6]+8
	MOV	r10,r12,ASR #4
	MOV	r12,r12,LSL #16
	PKHTB	r10,r10,r12,ASR #20	; r10= t[1]+t[6]+8>>4
	STR	r10,[r0, #12]		; y[1<<3] = t[1]+t[6]+8>>4
	SADD16	r12,r2, r5		; r12= t[2]+t[5]+8
	SSUB16	r5, r2, r5		; r5 = t[2]-t[5]+8
	MOV	r10,r12,ASR #4
	MOV	r12,r12,LSL #16
	PKHTB	r10,r10,r12,ASR #20	; r10= t[2]+t[5]+8>>4
	STR	r10,[r0, #28]		; y[2<<3] = t[2]+t[5]+8>>4
	SADD16	r12,r3, r4		; r12= t[3]+t[4]+8
	SSUB16	r4, r3, r4		; r4 = t[3]-t[4]+8
	MOV	r10,r12,ASR #4
	MOV	r12,r12,LSL #16
	PKHTB	r10,r10,r12,ASR #20	; r10= t[3]+t[4]+8>>4
	STR	r10,[r0, #44]		; y[3<<3] = t[3]+t[4]+8>>4
	MOV	r10,r4, ASR #4
	MOV	r4, r4, LSL #16
	PKHTB	r10,r10,r4, ASR #20	; r10= t[3]-t[4]+8>>4
	STR	r10,[r0, #60]		; y[4<<3] = t[3]-t[4]+8>>4
	MOV	r10,r5, ASR #4
	MOV	r5, r5, LSL #16
	PKHTB	r10,r10,r5, ASR #20	; r10= t[2]-t[5]+8>>4
	STR	r10,[r0, #76]		; y[5<<3] = t[2]-t[5]+8>>4
	MOV	r10,r6, ASR #4
	MOV	r6, r6, LSL #16
	PKHTB	r10,r10,r6, ASR #20	; r10= t[1]-t[6]+8>>4
	STR	r10,[r0, #92]		; y[6<<3] = t[1]-t[6]+8>>4
	MOV	r10,r7, ASR #4
	MOV	r7, r7, LSL #16
	PKHTB	r10,r10,r7, ASR #20	; r10= t[0]-t[7]+8>>4
	STR	r10,[r0, #108]		; y[7<<3] = t[0]-t[7]+8>>4
	MOV	PC,r14
	ENDP
 ]

 [ OC_ARM_ASM_NEON
	EXPORT	oc_idct8x8_1_neon
	EXPORT	oc_idct8x8_neon

	ALIGN 16
OC_IDCT_CONSTS_NEON
	DCW	    8
	DCW	64277 ; FB15 (C1S7)
	DCW	60547 ; EC83 (C2S6)
	DCW	54491 ; D4DB (C3S5)
	DCW	46341 ; B505 (C4S4)
	DCW	36410 ; 471D (C5S3)
	DCW	25080 ; 30FC (C6S2)
	DCW	12785 ; 31F1 (C7S1)

oc_idct8x8_1_neon PROC
	; r0 = ogg_int16_t  *_y
	; r1 = ogg_uint16_t  _dc
	VDUP.S16	Q0, r1
	VMOV		Q1, Q0
	VST1.64		{D0, D1, D2, D3}, [r0@128]!
	VST1.64		{D0, D1, D2, D3}, [r0@128]!
	VST1.64		{D0, D1, D2, D3}, [r0@128]!
	VST1.64		{D0, D1, D2, D3}, [r0@128]
	MOV	PC, r14
	ENDP

oc_idct8x8_neon PROC
	; r0 = ogg_int16_t *_y
	; r1 = ogg_int16_t *_x
	; r2 = int          _last_zzi
	CMP	r2, #10
	BLE	oc_idct8x8_10_neon
oc_idct8x8_slow_neon
	VPUSH		{D8-D15}
	MOV	r2, r1
	ADR	r3, OC_IDCT_CONSTS_NEON
	; Row transforms (input is pre-transposed)
	VLD1.64		{D16,D17,D18,D19}, [r2@128]!
	VLD1.64		{D20,D21,D22,D23}, [r2@128]!
	VLD1.64		{D24,D25,D26,D27}, [r2@128]!
	VSUB.S16	Q1, Q8, Q12	; Q8 = x[0]-x[4]
	VLD1.64		{D28,D29,D30,D31}, [r2@128]
	VADD.S16	Q8, Q8, Q12	; Q1 = x[0]+x[4]
	VLD1.64		{D0,D1},           [r3@128]
	MOV	r12, r14
	BL	oc_idct8x8_stage123_neon
; Stage 4
	VSUB.S16	Q15,Q8, Q7	; Q15 = y[7]=t[0]'-t[7]'
	VADD.S16	Q8, Q8, Q7	; Q8  = y[0]=t[0]'+t[7]'
	VSUB.S16	Q14,Q9, Q3	; Q14 = y[6]=t[1]'-t[6]''
	VADD.S16	Q9, Q9, Q3	; Q9  = y[1]=t[1]'+t[6]''
	VSUB.S16	Q13,Q10,Q5	; Q13 = y[5]=t[2]'-t[5]''
	VADD.S16	Q10,Q10,Q5	; Q10 = y[2]=t[2]'+t[5]''
	VTRN.16		Q14,Q15
	VSUB.S16	Q12,Q11,Q4	; Q12 = y[4]=t[3]'-t[4]'
	VADD.S16	Q11,Q11,Q4	; Q11 = y[3]=t[3]'+t[4]'
	; 8x8 Transpose
	VTRN.16		Q8, Q9
	VTRN.16		Q10,Q11
	VTRN.16		Q12,Q13
	VTRN.32		Q8, Q10
	VTRN.32		Q9, Q11
	VTRN.32		Q12,Q14
	VTRN.32		Q13,Q15
	VSWP		D17,D24
	VSUB.S16	Q1, Q8, Q12	; Q8 = x[0]-x[4]
	VSWP		D19,D26
	VADD.S16	Q8, Q8, Q12	; Q1 = x[0]+x[4]
	VSWP		D21,D28
	VSWP		D23,D30
	; Column transforms
	BL	oc_idct8x8_stage123_neon
	; We have to put the return address back in the LR, or the branch
	;  predictor will not recognize the function return and mis-predict the
	;  entire call stack.
	MOV	r14, r12
; Stage 4
	VSUB.S16	Q15,Q8, Q7	; Q15 = y[7]=t[0]'-t[7]'
	VADD.S16	Q8, Q8, Q7	; Q8  = y[0]=t[0]'+t[7]'
	VSUB.S16	Q14,Q9, Q3	; Q14 = y[6]=t[1]'-t[6]''
	VADD.S16	Q9, Q9, Q3	; Q9  = y[1]=t[1]'+t[6]''
	VSUB.S16	Q13,Q10,Q5	; Q13 = y[5]=t[2]'-t[5]''
	VADD.S16	Q10,Q10,Q5	; Q10 = y[2]=t[2]'+t[5]''
	VSUB.S16	Q12,Q11,Q4	; Q12 = y[4]=t[3]'-t[4]'
	VADD.S16	Q11,Q11,Q4	; Q11 = y[3]=t[3]'+t[4]'
	VMOV.I8		Q2,#0
	VPOP		{D8-D15}
	VMOV.I8		Q3,#0
	VRSHR.S16	Q8, Q8, #4	; Q8  = y[0]+8>>4
	VST1.64		{D4, D5, D6, D7}, [r1@128]!
	VRSHR.S16	Q9, Q9, #4	; Q9  = y[1]+8>>4
	VRSHR.S16	Q10,Q10,#4	; Q10 = y[2]+8>>4
	VST1.64		{D4, D5, D6, D7}, [r1@128]!
	VRSHR.S16	Q11,Q11,#4	; Q11 = y[3]+8>>4
	VRSHR.S16	Q12,Q12,#4	; Q12 = y[4]+8>>4
	VST1.64		{D4, D5, D6, D7}, [r1@128]!
	VRSHR.S16	Q13,Q13,#4	; Q13 = y[5]+8>>4
	VRSHR.S16	Q14,Q14,#4	; Q14 = y[6]+8>>4
	VST1.64		{D4, D5, D6, D7}, [r1@128]
	VRSHR.S16	Q15,Q15,#4	; Q15 = y[7]+8>>4
	VSTMIA		r0, {D16-D31}
	MOV	PC, r14
	ENDP

oc_idct8x8_stage123_neon PROC
; Stages 1 & 2
	VMULL.S16	Q4, D18,D1[3]
	VMULL.S16	Q5, D19,D1[3]
	VMULL.S16	Q7, D30,D1[3]
	VMULL.S16	Q6, D31,D1[3]
	VMULL.S16	Q2, D30,D0[1]
	VMULL.S16	Q3, D31,D0[1]
	VSHRN.S32	D8, Q4, #16
	VSHRN.S32	D9, Q5, #16	; Q4 = (OC_C7S1*x[1]>>16)
	VSHRN.S32	D14,Q7, #16
	VSHRN.S32	D15,Q6, #16	; Q7 = (OC_C7S1*x[7]>>16)
	VSHRN.S32	D4, Q2, #16
	VSHRN.S32	D5, Q3, #16	; Q2 = (OC_C1S7*x[7]>>16)-x[7]
	VSUB.S16	Q4, Q4, Q15
	VADD.S16	Q7, Q7, Q9
	VSUB.S16	Q4, Q4, Q2	; Q4 = t[4]
	VMULL.S16	Q2, D18,D0[1]
	VMULL.S16	Q9, D19,D0[1]
	VMULL.S16	Q5, D26,D0[3]
	VMULL.S16	Q3, D27,D0[3]
	VMULL.S16	Q6, D22,D0[3]
	VMULL.S16	Q12,D23,D0[3]
	VSHRN.S32	D4, Q2, #16
	VSHRN.S32	D5, Q9, #16	; Q2 = (OC_C1S7*x[1]>>16)-x[1]
	VSHRN.S32	D10,Q5, #16
	VSHRN.S32	D11,Q3, #16	; Q5 = (OC_C3S5*x[5]>>16)-x[5]
	VSHRN.S32	D12,Q6, #16
	VSHRN.S32	D13,Q12,#16	; Q6 = (OC_C3S5*x[3]>>16)-x[3]
	VADD.S16	Q7, Q7, Q2	; Q7 = t[7]
	VSUB.S16	Q5, Q5, Q11
	VADD.S16	Q6, Q6, Q11
	VADD.S16	Q5, Q5, Q13
	VADD.S16	Q6, Q6, Q13
	VMULL.S16	Q9, D22,D1[1]
	VMULL.S16	Q11,D23,D1[1]
	VMULL.S16	Q15,D26,D1[1]
	VMULL.S16	Q13,D27,D1[1]
	VMULL.S16	Q2, D20,D1[2]
	VMULL.S16	Q12,D21,D1[2]
	VSHRN.S32	D18,Q9, #16
	VSHRN.S32	D19,Q11,#16	; Q9 = (OC_C5S3*x[3]>>16)-x[3]
	VSHRN.S32	D30,Q15,#16
	VSHRN.S32	D31,Q13,#16	; Q15= (OC_C5S3*x[5]>>16)-x[5]
	VSHRN.S32	D4, Q2, #16
	VSHRN.S32	D5, Q12,#16	; Q2 = (OC_C6S2*x[2]>>16)
	VSUB.S16	Q5, Q5, Q9	; Q5 = t[5]
	VADD.S16	Q6, Q6, Q15	; Q6 = t[6]
	VSUB.S16	Q2, Q2, Q14
	VMULL.S16	Q3, D28,D1[2]
	VMULL.S16	Q11,D29,D1[2]
	VMULL.S16	Q12,D28,D0[2]
	VMULL.S16	Q9, D29,D0[2]
	VMULL.S16	Q13,D20,D0[2]
	VMULL.S16	Q15,D21,D0[2]
	VSHRN.S32	D6, Q3, #16
	VSHRN.S32	D7, Q11,#16	; Q3 = (OC_C6S2*x[6]>>16)
	VSHRN.S32	D24,Q12,#16
	VSHRN.S32	D25,Q9, #16	; Q12= (OC_C2S6*x[6]>>16)-x[6]
	VSHRN.S32	D26,Q13,#16
	VSHRN.S32	D27,Q15,#16	; Q13= (OC_C2S6*x[2]>>16)-x[2]
	VSUB.S16	Q9, Q4, Q5	; Q9 = t[4]-t[5]
	VSUB.S16	Q11,Q7, Q6	; Q11= t[7]-t[6]
	VADD.S16	Q3, Q3, Q10
	VADD.S16	Q4, Q4, Q5	; Q4 = t[4]'=t[4]+t[5]
	VADD.S16	Q7, Q7, Q6	; Q7 = t[7]'=t[7]+t[6]
	VSUB.S16	Q2, Q2, Q12	; Q2 = t[2]
	VADD.S16	Q3, Q3, Q13	; Q3 = t[3]
	VMULL.S16	Q12,D16,D1[0]
	VMULL.S16	Q13,D17,D1[0]
	VMULL.S16	Q14,D2, D1[0]
	VMULL.S16	Q15,D3, D1[0]
	VMULL.S16	Q5, D18,D1[0]
	VMULL.S16	Q6, D22,D1[0]
	VSHRN.S32	D24,Q12,#16
	VSHRN.S32	D25,Q13,#16
	VSHRN.S32	D28,Q14,#16
	VSHRN.S32	D29,Q15,#16
	VMULL.S16	Q13,D19,D1[0]
	VMULL.S16	Q15,D23,D1[0]
	VADD.S16	Q8, Q8, Q12	; Q8 = t[0]
	VADD.S16	Q1, Q1, Q14	; Q1 = t[1]
	VSHRN.S32	D10,Q5, #16
	VSHRN.S32	D12,Q6, #16
	VSHRN.S32	D11,Q13,#16
	VSHRN.S32	D13,Q15,#16
	VADD.S16	Q5, Q5, Q9	; Q5 = t[5]'=OC_C4S4*(t[4]-t[5])>>16
	VADD.S16	Q6, Q6, Q11	; Q6 = t[6]'=OC_C4S4*(t[7]-t[6])>>16
; Stage 3
	VSUB.S16	Q11,Q8, Q3	; Q11 = t[3]''=t[0]-t[3]
	VADD.S16	Q8, Q8, Q3	; Q8  = t[0]''=t[0]+t[3]
	VADD.S16	Q9, Q1, Q2	; Q9  = t[1]''=t[1]+t[2]
	VADD.S16	Q3, Q6, Q5	; Q3  = t[6]''=t[6]'+t[5]'
	VSUB.S16	Q10,Q1, Q2	; Q10 = t[2]''=t[1]-t[2]
	VSUB.S16	Q5, Q6, Q5	; Q5  = t[5]''=t[6]'-t[5]'
	MOV	PC, r14
	ENDP

oc_idct8x8_10_neon PROC
	ADR	r3, OC_IDCT_CONSTS_NEON
	VLD1.64		{D0,D1},          [r3@128]
	MOV	r2, r1
	; Row transforms (input is pre-transposed)
; Stage 1
	VLD1.64		{D16,D17,D18,D19},[r2@128]!
	MOV	r12, #16
	VMULL.S16	Q15,D16,D1[0]	; Q15= OC_C4S4*x[0]-(x[0]<<16)
	VLD1.64		{D17},            [r2@64], r12
	VMULL.S16	Q2, D18,D0[1]	; Q2 = OC_C1S7*x[1]-(x[1]<<16)
	VLD1.64		{D19},            [r2@64]
	VMULL.S16	Q14,D17,D0[2]	; Q14= OC_C2S6*x[2]-(x[2]<<16)
	VMULL.S16	Q3, D19,D0[3]	; Q3 = OC_C3S5*x[3]-(x[3]<<16)
	VMULL.S16	Q13,D19,D1[1]	; Q13= OC_C5S3*x[3]-(x[3]<<16)
	VMULL.S16	Q12,D18,D1[3]	; Q12= OC_C7S1*x[1]
	VMULL.S16	Q1, D17,D1[2]	; Q1 = OC_C6S2*x[2]
	VSHRN.S32	D30,Q15,#16	; D30= t[0]-x[0]
	VSHRN.S32	D4, Q2, #16	; D4 = t[7]-x[1]
	VSHRN.S32	D31,Q14,#16	; D31= t[3]-x[2]
	VSHRN.S32	D6, Q3, #16	; D6 = t[6]-x[3]
	VSHRN.S32	D7, Q13,#16	; D7 = -t[5]-x[3]
	VSHRN.S32	D5, Q12,#16	; D5 = t[4]
	VSHRN.S32	D2, Q1, #16	; D2 = t[2]
	VADD.S16	D4, D4, D18	; D4 = t[7]
	VADD.S16	D6, D6, D19	; D6 = t[6]
	VADD.S16	D7, D7, D19	; D7 = -t[5]
	VADD.S16	Q15,Q15,Q8	; D30= t[0]
					; D31= t[3]
; Stages 2 & 3
	VSUB.S16	Q12,Q2, Q3	; D24= t[7]-t[6]
					; D25= t[4]'=t[4]+t[5]
	VADD.S16	Q13,Q2, Q3	; D26= t[7]'=t[7]+t[6]
					; D27= t[4]-t[5]
	VMULL.S16	Q11,D24,D1[0]	; Q11= OC_C4S4*(t[7]-t[6])
					;       -(t[7]-t[6]<<16)
	VMULL.S16	Q14,D27,D1[0]	; Q14= OC_C4S4*(t[4]-t[5])
					;       -(t[4]-t[5]<<16)
	VADD.S16	D16,D30,D31	; D16= t[0]'=t[0]+t[3]
	VSUB.S16	D17,D30,D2	; D17= t[2]'=t[0]-t[2]
	VADD.S16	D18,D30,D2	; D18= t[1]'=t[0]+t[2]
	VSHRN.S32	D22,Q11,#16	; D22= (OC_C4S4*(t[7]-t[6])>>16)
					;       -(t[7]-t[6])
	VSHRN.S32	D23,Q14,#16	; D23= (OC_C4S4*(t[4]-t[5])>>16)
					;       -(t[4]-t[5])
	VSUB.S16	D19,D30,D31	; D19= t[3]'=t[0]-t[3]
	VADD.S16	D22,D22,D24	; D22= t[6]'=OC_C4S4*(t[7]-t[6])>>16
	VADD.S16	D23,D23,D27	; D23= t[5]'=OC_C4S4*(t[4]-t[5])>>16
	VSUB.S16	D27,D22,D23	; D27= t[5]''=t[6]'-t[5]'
	VADD.S16	D24,D22,D23	; D24= t[6]''=t[6]'+t[5]'
; Stage 4
	VSUB.S16	Q11,Q8, Q13	; D22= y[7]=t[0]'-t[7]'
					; D23= y[5]=t[2]'-t[5]''
	VSUB.S16	Q10,Q9, Q12	; D20= y[6]=t[1]'-t[6]'
					; D21= y[4]=t[3]'-t[4]''
	VADD.S16	Q8, Q8, Q13	; D16= y[0]=t[0]'+t[7]'
					; D17= y[2]=t[2]'+t[5]''
	VADD.S16	Q9, Q9, Q12	; D18= y[1]=t[1]'-t[6]'
					; D19= y[3]=t[3]'-t[4]''
	; 8x4 transpose
	VTRN.16		Q10,Q11		; Q10= c5c4a5a4 c7c6a7a6
					; Q11= d5d4b5b4 d7d6b7b6
	VTRN.16		Q8, Q9		; Q8 = c3c2a3a2 c1c0a1a0
					; Q9 = d3d2b3b2 d1d0b1b0
	VSWP		D20,D21		; Q10= c7c6a7a6 c5c4a5a4
	VSWP		D22,D23		; Q11= d7d6b7b6 d5d4b5b4
	VUZP.32		Q9, Q11		; Q9 = b7b6b5b4 b3b2b1b0
					; Q11= d7d6d5d4 d3d2d1d0
	VMULL.S16	Q15,D18,D0[1]
	VMULL.S16	Q13,D22,D1[1]
	VUZP.32		Q8, Q10		; Q8 = a7a6a5a4 a3a2a1a0
					; Q10= c7c6c5c4 c3c2c1c0
	; Column transforms
; Stages 1, 2, & 3
	VMULL.S16	Q14,D19,D0[1]	; Q14:Q15= OC_C1S7*x[1]-(x[1]<<16)
	VMULL.S16	Q12,D23,D1[1]	; Q12:Q13= OC_C5S3*x[3]-(x[3]<<16)
	VMULL.S16	Q3, D22,D0[3]
	VMULL.S16	Q2, D23,D0[3]	;  Q2:Q3 = OC_C3S5*x[3]-(x[3]<<16)
	VSHRN.S32	D30,Q15,#16
	VSHRN.S32	D31,Q14,#16	; Q15= (OC_C1S7*x[1]>>16)-x[1]
	VSHRN.S32	D26,Q13,#16
	VSHRN.S32	D27,Q12,#16	; Q13= (OC_C5S3*x[3]>>16)-x[3]
	VSHRN.S32	D28,Q3, #16
	VSHRN.S32	D29,Q2, #16	; Q14= (OC_C3S5*x[3]>>16)-x[3]
	VADD.S16	Q15,Q15,Q9	; Q15= t[7]
	VADD.S16	Q13,Q13,Q11	; Q13= -t[5]
	VADD.S16	Q14,Q14,Q11	; Q14= t[6]
	VMULL.S16	Q12,D18,D1[3]
	VMULL.S16	Q2, D19,D1[3]	;  Q2:Q12= OC_C7S1*x[1]
	VMULL.S16	Q1, D16,D1[0]
	VMULL.S16	Q11,D17,D1[0]	; Q11:Q1 = OC_C4S4*x[0]-(x[0]<<16)
	VMULL.S16	Q3, D20,D0[2]
	VMULL.S16	Q9, D21,D0[2]	;  Q9:Q3 = OC_C2S6*x[2]-(x[2]<<16)
	VSHRN.S32	D24,Q12,#16
	VSHRN.S32	D25,Q2, #16	; Q12= t[4]
	VMULL.S16	Q2, D20,D1[2]
	VSHRN.S32	D2, Q1, #16
	VSHRN.S32	D3, Q11,#16	; Q1 = (OC_C4S4*x[0]>>16)-x[0]
	VMULL.S16	Q11,D21,D1[2]	;  Q2:Q11= OC_C6S2*x[2]
	VSHRN.S32	D6, Q3, #16
	VSHRN.S32	D7, Q9, #16	; Q3 = (OC_C2S6*x[2]>>16)-x[2]
	VSUB.S16	Q9, Q15,Q14	; Q9 = t[7]-t[6]
	VADD.S16	Q15,Q15,Q14	; Q15= t[7]'=t[7]+t[6]
	VSHRN.S32	D4, Q2, #16
	VSHRN.S32	D5, Q11,#16	; Q2 = t[2]
	VADD.S16	Q1, Q1, Q8	; Q1 = t[0]
	VADD.S16	Q8, Q12,Q13	; Q8 = t[4]-t[5]
	VADD.S16	Q3, Q3, Q10	; Q3 = t[3]
	VMULL.S16	Q10,D16,D1[0]
	VMULL.S16	Q11,D17,D1[0]	; Q11:Q10= OC_C4S4*(t[4]-t[5])
					;           -(t[4]-t[5]<<16)
	VSUB.S16	Q12,Q12,Q13	; Q12= t[4]'=t[4]+t[5]
	VMULL.S16	Q14,D18,D1[0]
	VMULL.S16	Q13,D19,D1[0]	; Q13:Q14= OC_C4S4*(t[6]-t[7])
					;           -(t[6]-t[7]<<16)
	VSHRN.S32	D20,Q10,#16
	VSHRN.S32	D21,Q11,#16	; Q10= (OC_C4S4*(t[4]-t[5])>>16)
					;       -(t[4]-t[5])
	VADD.S16	Q11,Q1, Q3	; Q11= t[0]'=t[0]+t[3]
	VSUB.S16	Q3, Q1, Q3	; Q3 = t[3]'=t[0]-t[3]
	VSHRN.S32	D28,Q14,#16
	VSHRN.S32	D29,Q13,#16	; Q14= (OC_C4S4*(t[7]-t[6])>>16)
					;       -(t[7]-t[6])
	VADD.S16	Q10,Q10,Q8	; Q10=t[5]'
	VADD.S16	Q14,Q14,Q9	; Q14=t[6]'
	VSUB.S16	Q13,Q14,Q10	; Q13=t[5]''=t[6]'-t[5]'
	VADD.S16	Q14,Q14,Q10	; Q14=t[6]''=t[6]'+t[5]'
	VADD.S16	Q10,Q1, Q2	; Q10= t[1]'=t[0]+t[2]
	VSUB.S16	Q2, Q1, Q2	; Q2 = t[2]'=t[0]-t[2]
; Stage 4
	VADD.S16	Q8, Q11,Q15	; Q8  = y[0]=t[0]'+t[7]'
	VADD.S16	Q9, Q10,Q14	; Q9  = y[1]=t[1]'+t[6]''
	VSUB.S16	Q15,Q11,Q15	; Q15 = y[7]=t[0]'-t[7]'
	VSUB.S16	Q14,Q10,Q14	; Q14 = y[6]=t[1]'-t[6]''
	VADD.S16	Q10,Q2, Q13	; Q10 = y[2]=t[2]'+t[5]''
	VADD.S16	Q11,Q3, Q12	; Q11 = y[3]=t[3]'+t[4]'
	VSUB.S16	Q12,Q3, Q12	; Q12 = y[4]=t[3]'-t[4]'
	VSUB.S16	Q13,Q2, Q13	; Q13 = y[5]=t[2]'-t[5]''
	VMOV.I8		D2, #0
	VRSHR.S16	Q8, Q8, #4	; Q8  = y[0]+8>>4
	VST1.64		{D2}, [r1@64], r12
	VRSHR.S16	Q9, Q9, #4	; Q9  = y[1]+8>>4
	VRSHR.S16	Q10,Q10,#4	; Q10 = y[2]+8>>4
	VST1.64		{D2}, [r1@64], r12
	VRSHR.S16	Q11,Q11,#4	; Q11 = y[3]+8>>4
	VRSHR.S16	Q12,Q12,#4	; Q12 = y[4]+8>>4
	VST1.64		{D2}, [r1@64], r12
	VRSHR.S16	Q13,Q13,#4	; Q13 = y[5]+8>>4
	VRSHR.S16	Q14,Q14,#4	; Q14 = y[6]+8>>4
	VST1.64		{D2}, [r1@64]
	VRSHR.S16	Q15,Q15,#4	; Q15 = y[7]+8>>4
	VSTMIA		r0, {D16-D31}
	MOV	PC, r14
	ENDP
 ]

	END
