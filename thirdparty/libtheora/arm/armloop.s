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

	EXPORT	oc_loop_filter_frag_rows_arm

; Which bit this is depends on the order of packing within a bitfield.
; Hopefully that doesn't change among any of the relevant compilers.
OC_FRAG_CODED_FLAG	*	1

	; Vanilla ARM v4 version
loop_filter_h_arm PROC
	; r0 = unsigned char *_pix
	; r1 = int            _ystride
	; r2 = int           *_bv
	; preserves r0-r3
	STMFD	r13!,{r3-r6,r14}
	MOV	r14,#8
	MOV	r6, #255
lfh_arm_lp
	LDRB	r3, [r0, #-2]		; r3 = _pix[0]
	LDRB	r12,[r0, #1]		; r12= _pix[3]
	LDRB	r4, [r0, #-1]		; r4 = _pix[1]
	LDRB	r5, [r0]		; r5 = _pix[2]
	SUB	r3, r3, r12		; r3 = _pix[0]-_pix[3]+4
	ADD	r3, r3, #4
	SUB	r12,r5, r4		; r12= _pix[2]-_pix[1]
	ADD	r12,r12,r12,LSL #1	; r12= 3*(_pix[2]-_pix[1])
	ADD	r12,r12,r3	; r12= _pix[0]-_pix[3]+3*(_pix[2]-_pix[1])+4
	MOV	r12,r12,ASR #3
	LDRSB	r12,[r2, r12]
	; Stall (2 on Xscale)
	ADDS	r4, r4, r12
	CMPGT	r6, r4
	EORLT	r4, r6, r4, ASR #32
	SUBS	r5, r5, r12
	CMPGT	r6, r5
	EORLT	r5, r6, r5, ASR #32
	STRB	r4, [r0, #-1]
	STRB	r5, [r0], r1
	SUBS	r14,r14,#1
	BGT	lfh_arm_lp
	SUB	r0, r0, r1, LSL #3
	LDMFD	r13!,{r3-r6,PC}
	ENDP

loop_filter_v_arm PROC
	; r0 = unsigned char *_pix
	; r1 = int            _ystride
	; r2 = int           *_bv
	; preserves r0-r3
	STMFD	r13!,{r3-r6,r14}
	MOV	r14,#8
	MOV	r6, #255
lfv_arm_lp
	LDRB	r3, [r0, -r1, LSL #1]	; r3 = _pix[0]
	LDRB	r12,[r0, r1]		; r12= _pix[3]
	LDRB	r4, [r0, -r1]		; r4 = _pix[1]
	LDRB	r5, [r0]		; r5 = _pix[2]
	SUB	r3, r3, r12		; r3 = _pix[0]-_pix[3]+4
	ADD	r3, r3, #4
	SUB	r12,r5, r4		; r12= _pix[2]-_pix[1]
	ADD	r12,r12,r12,LSL #1	; r12= 3*(_pix[2]-_pix[1])
	ADD	r12,r12,r3	; r12= _pix[0]-_pix[3]+3*(_pix[2]-_pix[1])+4
	MOV	r12,r12,ASR #3
	LDRSB	r12,[r2, r12]
	; Stall (2 on Xscale)
	ADDS	r4, r4, r12
	CMPGT	r6, r4
	EORLT	r4, r6, r4, ASR #32
	SUBS	r5, r5, r12
	CMPGT	r6, r5
	EORLT	r5, r6, r5, ASR #32
	STRB	r4, [r0, -r1]
	STRB	r5, [r0], #1
	SUBS	r14,r14,#1
	BGT	lfv_arm_lp
	SUB	r0, r0, #8
	LDMFD	r13!,{r3-r6,PC}
	ENDP

oc_loop_filter_frag_rows_arm PROC
	; r0 = _ref_frame_data
	; r1 = _ystride
	; r2 = _bv
	; r3 = _frags
	; r4 = _fragi0
	; r5 = _fragi0_end
	; r6 = _fragi_top
	; r7 = _fragi_bot
	; r8 = _frag_buf_offs
	; r9 = _nhfrags
	MOV	r12,r13
	STMFD	r13!,{r0,r4-r11,r14}
	LDMFD	r12,{r4-r9}
	ADD	r2, r2, #127	; _bv += 127
	CMP	r4, r5		; if(_fragi0>=_fragi0_end)
	BGE	oslffri_arm_end	;   bail
	SUBS	r9, r9, #1	; r9 = _nhfrags-1	if (r9<=0)
	BLE	oslffri_arm_end	;			  bail
	ADD	r3, r3, r4, LSL #2	; r3 = &_frags[fragi]
	ADD	r8, r8, r4, LSL #2	; r8 = &_frag_buf_offs[fragi]
	SUB	r7, r7, r9	; _fragi_bot -= _nhfrags;
oslffri_arm_lp1
	MOV	r10,r4		; r10= fragi = _fragi0
	ADD	r11,r4, r9	; r11= fragi_end-1=fragi+_nhfrags-1
oslffri_arm_lp2
	LDR	r14,[r3], #4	; r14= _frags[fragi]	_frags++
	LDR	r0, [r13]	; r0 = _ref_frame_data
	LDR	r12,[r8], #4	; r12= _frag_buf_offs[fragi]   _frag_buf_offs++
	TST	r14,#OC_FRAG_CODED_FLAG
	BEQ	oslffri_arm_uncoded
	CMP	r10,r4		; if (fragi>_fragi0)
	ADD	r0, r0, r12	; r0 = _ref_frame_data + _frag_buf_offs[fragi]
	BLGT	loop_filter_h_arm
	CMP	r4, r6		; if (_fragi0>_fragi_top)
	BLGT	loop_filter_v_arm
	CMP	r10,r11		; if(fragi+1<fragi_end)===(fragi<fragi_end-1)
	LDRLT	r12,[r3]	; r12 = _frags[fragi+1]
	ADD	r0, r0, #8
	ADD	r10,r10,#1	; r10 = fragi+1;
	ANDLT	r12,r12,#OC_FRAG_CODED_FLAG
	CMPLT	r12,#OC_FRAG_CODED_FLAG	; && _frags[fragi+1].coded==0
	BLLT	loop_filter_h_arm
	CMP	r10,r7		; if (fragi<_fragi_bot)
	LDRLT	r12,[r3, r9, LSL #2]	; r12 = _frags[fragi+1+_nhfrags-1]
	SUB	r0, r0, #8
	ADD	r0, r0, r1, LSL #3
	ANDLT	r12,r12,#OC_FRAG_CODED_FLAG
	CMPLT	r12,#OC_FRAG_CODED_FLAG
	BLLT	loop_filter_v_arm
	CMP	r10,r11		; while(fragi<=fragi_end-1)
	BLE	oslffri_arm_lp2
	MOV	r4, r10		; r4 = fragi0 += _nhfrags
	CMP	r4, r5
	BLT	oslffri_arm_lp1
oslffri_arm_end
	LDMFD	r13!,{r0,r4-r11,PC}
oslffri_arm_uncoded
	ADD	r10,r10,#1
	CMP	r10,r11
	BLE	oslffri_arm_lp2
	MOV	r4, r10		; r4 = _fragi0 += _nhfrags
	CMP	r4, r5
	BLT	oslffri_arm_lp1
	LDMFD	r13!,{r0,r4-r11,PC}
	ENDP

 [ OC_ARM_ASM_MEDIA
	EXPORT	oc_loop_filter_init_v6
	EXPORT	oc_loop_filter_frag_rows_v6

oc_loop_filter_init_v6 PROC
	; r0 = _bv
	; r1 = _flimit (=L from the spec)
	MVN	r1, r1, LSL #1		; r1 = <0xFFFFFF|255-2*L>
	AND	r1, r1, #255		; r1 = ll=r1&0xFF
	ORR	r1, r1, r1, LSL #8	; r1 = <ll|ll>
	PKHBT	r1, r1, r1, LSL #16	; r1 = <ll|ll|ll|ll>
	STR	r1, [r0]
	MOV	PC,r14
	ENDP

; We could use the same strategy as the v filter below, but that would require
;  40 instructions to load the data and transpose it into columns and another
;  32 to write out the results at the end, plus the 52 instructions to do the
;  filtering itself.
; This is slightly less, and less code, even assuming we could have shared the
;  52 instructions in the middle with the other function.
; It executes slightly fewer instructions than the ARMv6 approach David Conrad
;  proposed for FFmpeg, but not by much:
;  http://lists.mplayerhq.hu/pipermail/ffmpeg-devel/2010-February/083141.html
; His is a lot less code, though, because it only does two rows at once instead
;  of four.
loop_filter_h_v6 PROC
	; r0 = unsigned char *_pix
	; r1 = int            _ystride
	; r2 = int            _ll
	; preserves r0-r3
	STMFD	r13!,{r4-r11,r14}
	LDR	r12,=0x10003
	BL loop_filter_h_core_v6
	ADD	r0, r0, r1, LSL #2
	BL loop_filter_h_core_v6
	SUB	r0, r0, r1, LSL #2
	LDMFD	r13!,{r4-r11,PC}
	ENDP

loop_filter_h_core_v6 PROC
	; r0 = unsigned char *_pix
	; r1 = int            _ystride
	; r2 = int            _ll
	; r12= 0x10003
	; Preserves r0-r3, r12; Clobbers r4-r11.
	LDR	r4,[r0, #-2]!		; r4 = <p3|p2|p1|p0>
	; Single issue
	LDR	r5,[r0, r1]!		; r5 = <q3|q2|q1|q0>
	UXTB16	r6, r4, ROR #16		; r6 = <p0|p2>
	UXTB16	r4, r4, ROR #8		; r4 = <p3|p1>
	UXTB16	r7, r5, ROR #16		; r7 = <q0|q2>
	UXTB16	r5, r5, ROR #8		; r5 = <q3|q1>
	PKHBT	r8, r4, r5, LSL #16	; r8 = <__|q1|__|p1>
	PKHBT	r9, r6, r7, LSL #16	; r9 = <__|q2|__|p2>
	SSUB16	r6, r4, r6		; r6 = <p3-p0|p1-p2>
	SMLAD	r6, r6, r12,r12		; r6 = <????|(p3-p0)+3*(p1-p2)+3>
	SSUB16	r7, r5, r7		; r7 = <q3-q0|q1-q2>
	SMLAD	r7, r7, r12,r12		; r7 = <????|(q0-q3)+3*(q2-q1)+4>
	LDR	r4,[r0, r1]!		; r4 = <r3|r2|r1|r0>
	MOV	r6, r6, ASR #3		; r6 = <??????|(p3-p0)+3*(p1-p2)+3>>3>
	LDR	r5,[r0, r1]!		; r5 = <s3|s2|s1|s0>
	PKHBT	r11,r6, r7, LSL #13	; r11= <??|-R_q|??|-R_p>
	UXTB16	r6, r4, ROR #16		; r6 = <r0|r2>
	UXTB16	r11,r11			; r11= <__|-R_q|__|-R_p>
	UXTB16	r4, r4, ROR #8		; r4 = <r3|r1>
	UXTB16	r7, r5, ROR #16		; r7 = <s0|s2>
	PKHBT	r10,r6, r7, LSL #16	; r10= <__|s2|__|r2>
	SSUB16	r6, r4, r6		; r6 = <r3-r0|r1-r2>
	UXTB16	r5, r5, ROR #8		; r5 = <s3|s1>
	SMLAD	r6, r6, r12,r12		; r6 = <????|(r3-r0)+3*(r2-r1)+3>
	SSUB16	r7, r5, r7		; r7 = <r3-r0|r1-r2>
	SMLAD	r7, r7, r12,r12		; r7 = <????|(s0-s3)+3*(s2-s1)+4>
	ORR	r9, r9, r10, LSL #8	; r9 = <s2|q2|r2|p2>
	MOV	r6, r6, ASR #3		; r6 = <??????|(r0-r3)+3*(r2-r1)+4>>3>
	PKHBT	r10,r4, r5, LSL #16	; r10= <__|s1|__|r1>
	PKHBT	r6, r6, r7, LSL #13	; r6 = <??|-R_s|??|-R_r>
	ORR	r8, r8, r10, LSL #8	; r8 = <s1|q1|r1|p1>
	UXTB16	r6, r6			; r6 = <__|-R_s|__|-R_r>
	MOV	r10,#0
	ORR	r6, r11,r6, LSL #8	; r6 = <-R_s|-R_q|-R_r|-R_p>
	; Single issue
	; There's no min, max or abs instruction.
	; SSUB8 and SEL will work for abs, and we can do all the rest with
	;  unsigned saturated adds, which means the GE flags are still all
	;  set when we're done computing lflim(abs(R_i),L).
	; This allows us to both add and subtract, and split the results by
	;  the original sign of R_i.
	SSUB8	r7, r10,r6
	; Single issue
	SEL	r7, r7, r6		; r7 = abs(R_i)
	; Single issue
	UQADD8	r4, r7, r2		; r4 = 255-max(2*L-abs(R_i),0)
	; Single issue
	UQADD8	r7, r7, r4
	; Single issue
	UQSUB8	r7, r7, r4		; r7 = min(abs(R_i),max(2*L-abs(R_i),0))
	; Single issue
	UQSUB8	r4, r8, r7
	UQADD8	r5, r9, r7
	UQADD8	r8, r8, r7
	UQSUB8	r9, r9, r7
	SEL	r8, r8, r4		; r8 = p1+lflim(R_i,L)
	SEL	r9, r9, r5		; r9 = p2-lflim(R_i,L)
	MOV	r5, r9, LSR #24		; r5 = s2
	STRB	r5, [r0,#2]!
	MOV	r4, r8, LSR #24		; r4 = s1
	STRB	r4, [r0,#-1]
	MOV	r5, r9, LSR #8		; r5 = r2
	STRB	r5, [r0,-r1]!
	MOV	r4, r8, LSR #8		; r4 = r1
	STRB	r4, [r0,#-1]
	MOV	r5, r9, LSR #16		; r5 = q2
	STRB	r5, [r0,-r1]!
	MOV	r4, r8, LSR #16		; r4 = q1
	STRB	r4, [r0,#-1]
	; Single issue
	STRB	r9, [r0,-r1]!
	; Single issue
	STRB	r8, [r0,#-1]
	MOV	PC,r14
	ENDP

; This uses the same strategy as the MMXEXT version for x86, except that UHADD8
;  computes (a+b>>1) instead of (a+b+1>>1) like PAVGB.
; This works just as well, with the following procedure for computing the
;  filter value, f:
;   u = ~UHADD8(p1,~p2);
;   v = UHADD8(~p1,p2);
;   m = v-u;
;   a = m^UHADD8(m^p0,m^~p3);
;   f = UHADD8(UHADD8(a,u1),v1);
;  where f = 127+R, with R in [-127,128] defined as in the spec.
; This is exactly the same amount of arithmetic as the version that uses PAVGB
;  as the basic operator.
; It executes about 2/3 the number of instructions of David Conrad's approach,
;  but requires more code, because it does all eight columns at once, instead
;  of four at a time.
loop_filter_v_v6 PROC
	; r0 = unsigned char *_pix
	; r1 = int            _ystride
	; r2 = int            _ll
	; preserves r0-r11
	STMFD	r13!,{r4-r11,r14}
	LDRD	r6, [r0, -r1]!		; r7, r6 = <p5|p1>
	LDRD	r4, [r0, -r1]		; r5, r4 = <p4|p0>
	LDRD	r8, [r0, r1]!		; r9, r8 = <p6|p2>
	MVN	r14,r6			; r14= ~p1
	LDRD	r10,[r0, r1]		; r11,r10= <p7|p3>
	; Filter the first four columns.
	MVN	r12,r8			; r12= ~p2
	UHADD8	r14,r14,r8		; r14= v1=~p1+p2>>1
	UHADD8	r12,r12,r6		; r12= p1+~p2>>1
	MVN	r10, r10		; r10=~p3
	MVN	r12,r12			; r12= u1=~p1+p2+1>>1
	SSUB8	r14,r14,r12		; r14= m1=v1-u1
	; Single issue
	EOR	r4, r4, r14		; r4 = m1^p0
	EOR	r10,r10,r14		; r10= m1^~p3
	UHADD8	r4, r4, r10		; r4 = (m1^p0)+(m1^~p3)>>1
	; Single issue
	EOR	r4, r4, r14		; r4 = a1=m1^((m1^p0)+(m1^~p3)>>1)
	SADD8	r14,r14,r12		; r14= v1=m1+u1
	UHADD8	r4, r4, r12		; r4 = a1+u1>>1
	MVN	r12,r9			; r12= ~p6
	UHADD8	r4, r4, r14		; r4 = f1=(a1+u1>>1)+v1>>1
	; Filter the second four columns.
	MVN	r14,r7			; r14= ~p5
	UHADD8	r12,r12,r7		; r12= p5+~p6>>1
	UHADD8	r14,r14,r9		; r14= v2=~p5+p6>>1
	MVN	r12,r12			; r12= u2=~p5+p6+1>>1
	MVN	r11,r11			; r11=~p7
	SSUB8	r10,r14,r12		; r10= m2=v2-u2
	; Single issue
	EOR	r5, r5, r10		; r5 = m2^p4
	EOR	r11,r11,r10		; r11= m2^~p7
	UHADD8	r5, r5, r11		; r5 = (m2^p4)+(m2^~p7)>>1
	; Single issue
	EOR	r5, r5, r10		; r5 = a2=m2^((m2^p4)+(m2^~p7)>>1)
	; Single issue
	UHADD8	r5, r5, r12		; r5 = a2+u2>>1
	LDR	r12,=0x7F7F7F7F		; r12 = {127}x4
	UHADD8	r5, r5, r14		; r5 = f2=(a2+u2>>1)+v2>>1
	; Now split f[i] by sign.
	; There's no min or max instruction.
	; We could use SSUB8 and SEL, but this is just as many instructions and
	;  dual issues more (for v7 without NEON).
	UQSUB8	r10,r4, r12		; r10= R_i>0?R_i:0
	UQSUB8	r4, r12,r4		; r4 = R_i<0?-R_i:0
	UQADD8	r11,r10,r2		; r11= 255-max(2*L-abs(R_i<0),0)
	UQADD8	r14,r4, r2		; r14= 255-max(2*L-abs(R_i>0),0)
	UQADD8	r10,r10,r11
	UQADD8	r4, r4, r14
	UQSUB8	r10,r10,r11		; r10= min(abs(R_i<0),max(2*L-abs(R_i<0),0))
	UQSUB8	r4, r4, r14		; r4 = min(abs(R_i>0),max(2*L-abs(R_i>0),0))
	UQSUB8	r11,r5, r12		; r11= R_i>0?R_i:0
	UQADD8	r6, r6, r10
	UQSUB8	r8, r8, r10
	UQSUB8	r5, r12,r5		; r5 = R_i<0?-R_i:0
	UQSUB8	r6, r6, r4		; r6 = p1+lflim(R_i,L)
	UQADD8	r8, r8, r4		; r8 = p2-lflim(R_i,L)
	UQADD8	r10,r11,r2		; r10= 255-max(2*L-abs(R_i<0),0)
	UQADD8	r14,r5, r2		; r14= 255-max(2*L-abs(R_i>0),0)
	UQADD8	r11,r11,r10
	UQADD8	r5, r5, r14
	UQSUB8	r11,r11,r10		; r11= min(abs(R_i<0),max(2*L-abs(R_i<0),0))
	UQSUB8	r5, r5, r14		; r5 = min(abs(R_i>0),max(2*L-abs(R_i>0),0))
	UQADD8	r7, r7, r11
	UQSUB8	r9, r9, r11
	UQSUB8	r7, r7, r5		; r7 = p5+lflim(R_i,L)
	STRD	r6, [r0, -r1]		; [p5:p1] = [r7: r6]
	UQADD8	r9, r9, r5		; r9 = p6-lflim(R_i,L)
	STRD	r8, [r0]		; [p6:p2] = [r9: r8]
	LDMFD	r13!,{r4-r11,PC}
	ENDP

oc_loop_filter_frag_rows_v6 PROC
	; r0 = _ref_frame_data
	; r1 = _ystride
	; r2 = _bv
	; r3 = _frags
	; r4 = _fragi0
	; r5 = _fragi0_end
	; r6 = _fragi_top
	; r7 = _fragi_bot
	; r8 = _frag_buf_offs
	; r9 = _nhfrags
	MOV	r12,r13
	STMFD	r13!,{r0,r4-r11,r14}
	LDMFD	r12,{r4-r9}
	LDR	r2, [r2]	; ll = *(int *)_bv
	CMP	r4, r5		; if(_fragi0>=_fragi0_end)
	BGE	oslffri_v6_end	;   bail
	SUBS	r9, r9, #1	; r9 = _nhfrags-1	if (r9<=0)
	BLE	oslffri_v6_end	;			  bail
	ADD	r3, r3, r4, LSL #2	; r3 = &_frags[fragi]
	ADD	r8, r8, r4, LSL #2	; r8 = &_frag_buf_offs[fragi]
	SUB	r7, r7, r9	; _fragi_bot -= _nhfrags;
oslffri_v6_lp1
	MOV	r10,r4		; r10= fragi = _fragi0
	ADD	r11,r4, r9	; r11= fragi_end-1=fragi+_nhfrags-1
oslffri_v6_lp2
	LDR	r14,[r3], #4	; r14= _frags[fragi]	_frags++
	LDR	r0, [r13]	; r0 = _ref_frame_data
	LDR	r12,[r8], #4	; r12= _frag_buf_offs[fragi]   _frag_buf_offs++
	TST	r14,#OC_FRAG_CODED_FLAG
	BEQ	oslffri_v6_uncoded
	CMP	r10,r4		; if (fragi>_fragi0)
	ADD	r0, r0, r12	; r0 = _ref_frame_data + _frag_buf_offs[fragi]
	BLGT	loop_filter_h_v6
	CMP	r4, r6		; if (fragi0>_fragi_top)
	BLGT	loop_filter_v_v6
	CMP	r10,r11		; if(fragi+1<fragi_end)===(fragi<fragi_end-1)
	LDRLT	r12,[r3]	; r12 = _frags[fragi+1]
	ADD	r0, r0, #8
	ADD	r10,r10,#1	; r10 = fragi+1;
	ANDLT	r12,r12,#OC_FRAG_CODED_FLAG
	CMPLT	r12,#OC_FRAG_CODED_FLAG	; && _frags[fragi+1].coded==0
	BLLT	loop_filter_h_v6
	CMP	r10,r7		; if (fragi<_fragi_bot)
	LDRLT	r12,[r3, r9, LSL #2]	; r12 = _frags[fragi+1+_nhfrags-1]
	SUB	r0, r0, #8
	ADD	r0, r0, r1, LSL #3
	ANDLT	r12,r12,#OC_FRAG_CODED_FLAG
	CMPLT	r12,#OC_FRAG_CODED_FLAG
	BLLT	loop_filter_v_v6
	CMP	r10,r11		; while(fragi<=fragi_end-1)
	BLE	oslffri_v6_lp2
	MOV	r4, r10		; r4 = fragi0 += nhfrags
	CMP	r4, r5
	BLT	oslffri_v6_lp1
oslffri_v6_end
	LDMFD	r13!,{r0,r4-r11,PC}
oslffri_v6_uncoded
	ADD	r10,r10,#1
	CMP	r10,r11
	BLE	oslffri_v6_lp2
	MOV	r4, r10		; r4 = fragi0 += nhfrags
	CMP	r4, r5
	BLT	oslffri_v6_lp1
	LDMFD	r13!,{r0,r4-r11,PC}
	ENDP
 ]

 [ OC_ARM_ASM_NEON
	EXPORT	oc_loop_filter_init_neon
	EXPORT	oc_loop_filter_frag_rows_neon

oc_loop_filter_init_neon PROC
	; r0 = _bv
	; r1 = _flimit (=L from the spec)
	MOV		r1, r1, LSL #1  ; r1 = 2*L
	VDUP.S16	Q15, r1		; Q15= 2L in U16s
	VST1.64		{D30,D31}, [r0@128]
	MOV	PC,r14
	ENDP

loop_filter_h_neon PROC
	; r0 = unsigned char *_pix
	; r1 = int            _ystride
	; r2 = int           *_bv
	; preserves r0-r3
	; We assume Q15= 2*L in U16s
	;                    My best guesses at cycle counts (and latency)--vvv
	SUB	r12,r0, #2
	; Doing a 2-element structure load saves doing two VTRN's below, at the
	;  cost of using two more slower single-lane loads vs. the faster
	;  all-lane loads.
	; It's less code this way, though, and benches a hair faster, but it
	;  leaves D2 and D4 swapped.
	VLD2.16	{D0[],D2[]},  [r12], r1		; D0 = ____________1100     2,1
						; D2 = ____________3322
	VLD2.16	{D4[],D6[]},  [r12], r1		; D4 = ____________5544     2,1
						; D6 = ____________7766
	VLD2.16	{D0[1],D2[1]},[r12], r1		; D0 = ________99881100     3,1
						; D2 = ________BBAA3322
	VLD2.16	{D4[1],D6[1]},[r12], r1		; D4 = ________DDCC5544     3,1
						; D6 = ________FFEE7766
	VLD2.16	{D0[2],D2[2]},[r12], r1		; D0 = ____GGHH99881100     3,1
						; D2 = ____JJIIBBAA3322
	VLD2.16	{D4[2],D6[2]},[r12], r1		; D4 = ____KKLLDDCC5544     3,1
						; D6 = ____NNMMFFEE7766
	VLD2.16	{D0[3],D2[3]},[r12], r1		; D0 = PPOOGGHH99881100     3,1
						; D2 = RRQQJJIIBBAA3322
	VLD2.16	{D4[3],D6[3]},[r12], r1		; D4 = TTSSKKLLDDCC5544     3,1
						; D6 = VVUUNNMMFFEE7766
	VTRN.8	D0, D4	; D0 = SSOOKKGGCC884400 D4 = TTPPLLHHDD995511       1,1
	VTRN.8	D2, D6	; D2 = UUQQMMIIEEAA6622 D6 = VVRRNNJJFFBB7733       1,1
	VSUBL.U8	Q0, D0, D6	; Q0 = 00 - 33 in S16s              1,3
	VSUBL.U8	Q8, D2, D4	; Q8 = 22 - 11 in S16s              1,3
	ADD	r12,r0, #8
	VADD.S16	Q0, Q0, Q8	;                                   1,3
	PLD	[r12]
	VADD.S16	Q0, Q0, Q8	;                                   1,3
	PLD	[r12,r1]
	VADD.S16	Q0, Q0, Q8	; Q0 = [0-3]+3*[2-1]                1,3
	PLD	[r12,r1, LSL #1]
	VRSHR.S16	Q0, Q0, #3	; Q0 = f = ([0-3]+3*[2-1]+4)>>3     1,4
	ADD	r12,r12,r1, LSL #2
	;  We want to do
	; f =             CLAMP(MIN(-2L-f,0), f, MAX(2L-f,0))
	;   = ((f >= 0) ? MIN( f ,MAX(2L- f ,0)) : MAX(  f , MIN(-2L- f ,0)))
	;   = ((f >= 0) ? MIN(|f|,MAX(2L-|f|,0)) : MAX(-|f|, MIN(-2L+|f|,0)))
	;   = ((f >= 0) ? MIN(|f|,MAX(2L-|f|,0)) :-MIN( |f|,-MIN(-2L+|f|,0)))
	;   = ((f >= 0) ? MIN(|f|,MAX(2L-|f|,0)) :-MIN( |f|, MAX( 2L-|f|,0)))
	; So we've reduced the left and right hand terms to be the same, except
	; for a negation.
	; Stall x3
	VABS.S16	Q9, Q0		; Q9 = |f| in U16s                  1,4
	PLD	[r12,-r1]
	VSHR.S16	Q0, Q0, #15	; Q0 = -1 or 0 according to sign    1,3
	PLD	[r12]
	VQSUB.U16	Q10,Q15,Q9	; Q10= MAX(2L-|f|,0) in U16s        1,4
	PLD	[r12,r1]
	VMOVL.U8	Q1, D2	   ; Q2 = __UU__QQ__MM__II__EE__AA__66__22  2,3
	PLD	[r12,r1,LSL #1]
	VMIN.U16	Q9, Q10,Q9	; Q9 = MIN(|f|,MAX(2L-|f|))         1,4
	ADD	r12,r12,r1, LSL #2
	; Now we need to correct for the sign of f.
	; For negative elements of Q0, we want to subtract the appropriate
	; element of Q9. For positive elements we want to add them. No NEON
	; instruction exists to do this, so we need to negate the negative
	; elements, and we can then just add them. a-b = a-(1+!b) = a-1+!b
	VADD.S16	Q9, Q9, Q0	;				    1,3
	PLD	[r12,-r1]
	VEOR.S16	Q9, Q9, Q0	; Q9 = real value of f              1,3
	; Bah. No VRSBW.U8
	; Stall (just 1 as Q9 not needed to second pipeline stage. I think.)
	VADDW.U8	Q2, Q9, D4 ; Q1 = xxTTxxPPxxLLxxHHxxDDxx99xx55xx11  1,3
	VSUB.S16	Q1, Q1, Q9 ; Q2 = xxUUxxQQxxMMxxIIxxEExxAAxx66xx22  1,3
	VQMOVUN.S16	D4, Q2		; D4 = TTPPLLHHDD995511		    1,1
	VQMOVUN.S16	D2, Q1		; D2 = UUQQMMIIEEAA6622		    1,1
	SUB	r12,r0, #1
	VTRN.8	D4, D2		; D4 = QQPPIIHHAA992211	D2 = MMLLEEDD6655   1,1
	VST1.16	{D4[0]}, [r12], r1
	VST1.16	{D2[0]}, [r12], r1
	VST1.16	{D4[1]}, [r12], r1
	VST1.16	{D2[1]}, [r12], r1
	VST1.16	{D4[2]}, [r12], r1
	VST1.16	{D2[2]}, [r12], r1
	VST1.16	{D4[3]}, [r12], r1
	VST1.16	{D2[3]}, [r12], r1
	MOV	PC,r14
	ENDP

loop_filter_v_neon PROC
	; r0 = unsigned char *_pix
	; r1 = int            _ystride
	; r2 = int           *_bv
	; preserves r0-r3
	; We assume Q15= 2*L in U16s
	;                    My best guesses at cycle counts (and latency)--vvv
	SUB	r12,r0, r1, LSL #1
	VLD1.64	{D0}, [r12@64], r1		; D0 = SSOOKKGGCC884400     2,1
	VLD1.64	{D2}, [r12@64], r1		; D2 = TTPPLLHHDD995511     2,1
	VLD1.64	{D4}, [r12@64], r1		; D4 = UUQQMMIIEEAA6622     2,1
	VLD1.64	{D6}, [r12@64]			; D6 = VVRRNNJJFFBB7733     2,1
	VSUBL.U8	Q8, D4, D2	; Q8 = 22 - 11 in S16s              1,3
	VSUBL.U8	Q0, D0, D6	; Q0 = 00 - 33 in S16s              1,3
	ADD	r12, #8
	VADD.S16	Q0, Q0, Q8	;                                   1,3
	PLD	[r12]
	VADD.S16	Q0, Q0, Q8	;                                   1,3
	PLD	[r12,r1]
	VADD.S16	Q0, Q0, Q8	; Q0 = [0-3]+3*[2-1]                1,3
	SUB	r12, r0, r1
	VRSHR.S16	Q0, Q0, #3	; Q0 = f = ([0-3]+3*[2-1]+4)>>3     1,4
	;  We want to do
	; f =             CLAMP(MIN(-2L-f,0), f, MAX(2L-f,0))
	;   = ((f >= 0) ? MIN( f ,MAX(2L- f ,0)) : MAX(  f , MIN(-2L- f ,0)))
	;   = ((f >= 0) ? MIN(|f|,MAX(2L-|f|,0)) : MAX(-|f|, MIN(-2L+|f|,0)))
	;   = ((f >= 0) ? MIN(|f|,MAX(2L-|f|,0)) :-MIN( |f|,-MIN(-2L+|f|,0)))
	;   = ((f >= 0) ? MIN(|f|,MAX(2L-|f|,0)) :-MIN( |f|, MAX( 2L-|f|,0)))
	; So we've reduced the left and right hand terms to be the same, except
	; for a negation.
	; Stall x3
	VABS.S16	Q9, Q0		; Q9 = |f| in U16s                  1,4
	VSHR.S16	Q0, Q0, #15	; Q0 = -1 or 0 according to sign    1,3
	; Stall x2
	VQSUB.U16	Q10,Q15,Q9	; Q10= MAX(2L-|f|,0) in U16s        1,4
	VMOVL.U8	Q2, D4	   ; Q2 = __UU__QQ__MM__II__EE__AA__66__22  2,3
	; Stall x2
	VMIN.U16	Q9, Q10,Q9	; Q9 = MIN(|f|,MAX(2L-|f|))         1,4
	; Now we need to correct for the sign of f.
	; For negative elements of Q0, we want to subtract the appropriate
	; element of Q9. For positive elements we want to add them. No NEON
	; instruction exists to do this, so we need to negate the negative
	; elements, and we can then just add them. a-b = a-(1+!b) = a-1+!b
	; Stall x3
	VADD.S16	Q9, Q9, Q0	;				    1,3
	; Stall x2
	VEOR.S16	Q9, Q9, Q0	; Q9 = real value of f              1,3
	; Bah. No VRSBW.U8
	; Stall (just 1 as Q9 not needed to second pipeline stage. I think.)
	VADDW.U8	Q1, Q9, D2 ; Q1 = xxTTxxPPxxLLxxHHxxDDxx99xx55xx11  1,3
	VSUB.S16	Q2, Q2, Q9 ; Q2 = xxUUxxQQxxMMxxIIxxEExxAAxx66xx22  1,3
	VQMOVUN.S16	D2, Q1		; D2 = TTPPLLHHDD995511		    1,1
	VQMOVUN.S16	D4, Q2		; D4 = UUQQMMIIEEAA6622		    1,1
	VST1.64	{D2}, [r12@64], r1
	VST1.64	{D4}, [r12@64], r1
	MOV	PC,r14
	ENDP

oc_loop_filter_frag_rows_neon PROC
	; r0 = _ref_frame_data
	; r1 = _ystride
	; r2 = _bv
	; r3 = _frags
	; r4 = _fragi0
	; r5 = _fragi0_end
	; r6 = _fragi_top
	; r7 = _fragi_bot
	; r8 = _frag_buf_offs
	; r9 = _nhfrags
	MOV	r12,r13
	STMFD	r13!,{r0,r4-r11,r14}
	LDMFD	r12,{r4-r9}
	CMP	r4, r5		; if(_fragi0>=_fragi0_end)
	BGE	oslffri_neon_end;   bail
	SUBS	r9, r9, #1	; r9 = _nhfrags-1	if (r9<=0)
	BLE	oslffri_neon_end	;		  bail
	VLD1.64	{D30,D31}, [r2@128]	; Q15= 2L in U16s
	ADD	r3, r3, r4, LSL #2	; r3 = &_frags[fragi]
	ADD	r8, r8, r4, LSL #2	; r8 = &_frag_buf_offs[fragi]
	SUB	r7, r7, r9	; _fragi_bot -= _nhfrags;
oslffri_neon_lp1
	MOV	r10,r4		; r10= fragi = _fragi0
	ADD	r11,r4, r9	; r11= fragi_end-1=fragi+_nhfrags-1
oslffri_neon_lp2
	LDR	r14,[r3], #4	; r14= _frags[fragi]	_frags++
	LDR	r0, [r13]	; r0 = _ref_frame_data
	LDR	r12,[r8], #4	; r12= _frag_buf_offs[fragi]   _frag_buf_offs++
	TST	r14,#OC_FRAG_CODED_FLAG
	BEQ	oslffri_neon_uncoded
	CMP	r10,r4		; if (fragi>_fragi0)
	ADD	r0, r0, r12	; r0 = _ref_frame_data + _frag_buf_offs[fragi]
	BLGT	loop_filter_h_neon
	CMP	r4, r6		; if (_fragi0>_fragi_top)
	BLGT	loop_filter_v_neon
	CMP	r10,r11		; if(fragi+1<fragi_end)===(fragi<fragi_end-1)
	LDRLT	r12,[r3]	; r12 = _frags[fragi+1]
	ADD	r0, r0, #8
	ADD	r10,r10,#1	; r10 = fragi+1;
	ANDLT	r12,r12,#OC_FRAG_CODED_FLAG
	CMPLT	r12,#OC_FRAG_CODED_FLAG	; && _frags[fragi+1].coded==0
	BLLT	loop_filter_h_neon
	CMP	r10,r7		; if (fragi<_fragi_bot)
	LDRLT	r12,[r3, r9, LSL #2]	; r12 = _frags[fragi+1+_nhfrags-1]
	SUB	r0, r0, #8
	ADD	r0, r0, r1, LSL #3
	ANDLT	r12,r12,#OC_FRAG_CODED_FLAG
	CMPLT	r12,#OC_FRAG_CODED_FLAG
	BLLT	loop_filter_v_neon
	CMP	r10,r11		; while(fragi<=fragi_end-1)
	BLE	oslffri_neon_lp2
	MOV	r4, r10		; r4 = _fragi0 += _nhfrags
	CMP	r4, r5
	BLT	oslffri_neon_lp1
oslffri_neon_end
	LDMFD	r13!,{r0,r4-r11,PC}
oslffri_neon_uncoded
	ADD	r10,r10,#1
	CMP	r10,r11
	BLE	oslffri_neon_lp2
	MOV	r4, r10		; r4 = _fragi0 += _nhfrags
	CMP	r4, r5
	BLT	oslffri_neon_lp1
	LDMFD	r13!,{r0,r4-r11,PC}
	ENDP
 ]

	END
