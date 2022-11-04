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
;
; function:
;   last mod: $Id$
;
;********************************************************************

	AREA	|.text|, CODE, READONLY

	EXPORT oc_pack_read_arm
	EXPORT oc_pack_read1_arm
	EXPORT oc_huff_token_decode_arm

oc_pack_read1_arm PROC
	; r0 = oc_pack_buf *_b
	ADD r12,r0,#8
	LDMIA r12,{r2,r3}      ; r2 = window
	; Stall...             ; r3 = available
	; Stall...
	SUBS r3,r3,#1          ; r3 = available-1, available<1 => LT
	BLT oc_pack_read1_refill
	MOV r0,r2,LSR #31      ; r0 = window>>31
	MOV r2,r2,LSL #1       ; r2 = window<<=1
	STMIA r12,{r2,r3}      ; window = r2
	                       ; available = r3
	MOV PC,r14
	ENDP

oc_pack_read_arm PROC
	; r0 = oc_pack_buf *_b
	; r1 = int          _bits
	ADD r12,r0,#8
	LDMIA r12,{r2,r3}      ; r2 = window
	; Stall...             ; r3 = available
	; Stall...
	SUBS r3,r3,r1          ; r3 = available-_bits, available<_bits => LT
	BLT oc_pack_read_refill
	RSB r0,r1,#32          ; r0 = 32-_bits
	MOV r0,r2,LSR r0       ; r0 = window>>32-_bits
	MOV r2,r2,LSL r1       ; r2 = window<<=_bits
	STMIA r12,{r2,r3}      ; window = r2
	                       ; available = r3
	MOV PC,r14

; We need to refill window.
oc_pack_read1_refill
	MOV r1,#1
oc_pack_read_refill
	STMFD r13!,{r10,r11,r14}
	LDMIA r0,{r10,r11}     ; r10 = stop
	                       ; r11 = ptr
	RSB r0,r1,#32          ; r0 = 32-_bits
	RSB r3,r3,r0           ; r3 = 32-available
; We can use unsigned compares for both the pointers and for available
;  (allowing us to chain condition codes) because available will never be
;  larger than 32 (or we wouldn't be here), and thus 32-available will never be
;  negative.
	CMP r10,r11            ; ptr<stop => HI
	CMPHI r3,#7            ;   available<=24 => HI
	LDRHIB r14,[r11],#1    ;     r14 = *ptr++
	SUBHI r3,#8            ;     available += 8
	; (HI) Stall...
	ORRHI r2,r14,LSL r3    ;     r2 = window|=r14<<32-available
	CMPHI r10,r11          ;     ptr<stop => HI
	CMPHI r3,#7            ;       available<=24 => HI
	LDRHIB r14,[r11],#1    ;         r14 = *ptr++
	SUBHI r3,#8            ;         available += 8
	; (HI) Stall...
	ORRHI r2,r14,LSL r3    ;         r2 = window|=r14<<32-available
	CMPHI r10,r11          ;         ptr<stop => HI
	CMPHI r3,#7            ;           available<=24 => HI
	LDRHIB r14,[r11],#1    ;             r14 = *ptr++
	SUBHI r3,#8            ;             available += 8
	; (HI) Stall...
	ORRHI r2,r14,LSL r3    ;             r2 = window|=r14<<32-available
	CMPHI r10,r11          ;             ptr<stop => HI
	CMPHI r3,#7            ;               available<=24 => HI
	LDRHIB r14,[r11],#1    ;                 r14 = *ptr++
	SUBHI r3,#8            ;                 available += 8
	; (HI) Stall...
	ORRHI r2,r14,LSL r3    ;                 r2 = window|=r14<<32-available
	SUBS r3,r0,r3          ; r3 = available-=_bits, available<bits => GT
	BLT oc_pack_read_refill_last
	MOV r0,r2,LSR r0       ; r0 = window>>32-_bits
	MOV r2,r2,LSL r1       ; r2 = window<<=_bits
	STR r11,[r12,#-4]      ; ptr = r11
	STMIA r12,{r2,r3}      ; window = r2
	                       ; available = r3
	LDMFD r13!,{r10,r11,PC}

; Either we wanted to read more than 24 bits and didn't have enough room to
;  stuff the last byte into the window, or we hit the end of the packet.
oc_pack_read_refill_last
	CMP r11,r10            ; ptr<stop => LO
; If we didn't hit the end of the packet, then pull enough of the next byte to
;  to fill up the window.
	LDRLOB r14,[r11]       ; (LO) r14 = *ptr
; Otherwise, set the EOF flag and pretend we have lots of available bits.
	MOVHS r14,#1           ; (HS) r14 = 1
	ADDLO r10,r3,r1        ; (LO) r10 = available
	STRHS r14,[r12,#8]     ; (HS) eof = 1
	ANDLO r10,r10,#7       ; (LO) r10 = available&7
	MOVHS r3,#1<<30        ; (HS) available = OC_LOTS_OF_BITS
	ORRLO r2,r14,LSL r10   ; (LO) r2 = window|=*ptr>>(available&7)
	MOV r0,r2,LSR r0       ; r0 = window>>32-_bits
	MOV r2,r2,LSL r1       ; r2 = window<<=_bits
	STR r11,[r12,#-4]      ; ptr = r11
	STMIA r12,{r2,r3}      ; window = r2
	                       ; available = r3
	LDMFD r13!,{r10,r11,PC}
	ENDP



oc_huff_token_decode_arm PROC
	; r0 = oc_pack_buf       *_b
	; r1 = const ogg_int16_t *_tree
	STMFD r13!,{r4,r5,r10,r14}
	LDRSH r10,[r1]         ; r10 = n=_tree[0]
	LDMIA r0,{r2-r5}       ; r2 = stop
	; Stall...             ; r3 = ptr
	; Stall...             ; r4 = window
	                       ; r5 = available
	CMP r10,r5             ; n>available => GT
	BGT oc_huff_token_decode_refill0
	RSB r14,r10,#32        ; r14 = 32-n
	MOV r14,r4,LSR r14     ; r14 = bits=window>>32-n
	ADD r14,r1,r14,LSL #1  ; r14 = _tree+bits
	LDRSH r12,[r14,#2]     ; r12 = node=_tree[1+bits]
	; Stall...
	; Stall...
	RSBS r14,r12,#0        ; r14 = -node, node>0 => MI
	BMI oc_huff_token_decode_continue
	MOV r10,r14,LSR #8     ; r10 = n=node>>8
	MOV r4,r4,LSL r10      ; r4 = window<<=n
	SUB r5,r10             ; r5 = available-=n
	STMIB r0,{r3-r5}       ; ptr = r3
	                       ; window = r4
	                       ; available = r5
	AND r0,r14,#255        ; r0 = node&255
	LDMFD r13!,{r4,r5,r10,pc}

; The first tree node wasn't enough to reach a leaf, read another
oc_huff_token_decode_continue
	ADD r12,r1,r12,LSL #1  ; r12 = _tree+node
	MOV r4,r4,LSL r10      ; r4 = window<<=n
	SUB r5,r5,r10          ; r5 = available-=n
	LDRSH r10,[r12],#2     ; r10 = n=_tree[node]
	; Stall...             ; r12 = _tree+node+1
	; Stall...
	CMP r10,r5             ; n>available => GT
	BGT oc_huff_token_decode_refill
	RSB r14,r10,#32        ; r14 = 32-n
	MOV r14,r4,LSR r14     ; r14 = bits=window>>32-n
	ADD r12,r12,r14        ;
	LDRSH r12,[r12,r14]    ; r12 = node=_tree[node+1+bits]
	; Stall...
	; Stall...
	RSBS r14,r12,#0        ; r14 = -node, node>0 => MI
	BMI oc_huff_token_decode_continue
	MOV r10,r14,LSR #8     ; r10 = n=node>>8
	MOV r4,r4,LSL r10      ; r4 = window<<=n
	SUB r5,r10             ; r5 = available-=n
	STMIB r0,{r3-r5}       ; ptr = r3
	                       ; window = r4
	                       ; available = r5
	AND r0,r14,#255        ; r0 = node&255
	LDMFD r13!,{r4,r5,r10,pc}

oc_huff_token_decode_refill0
	ADD r12,r1,#2          ; r12 = _tree+1
oc_huff_token_decode_refill
; We can't possibly need more than 15 bits, so available must be <= 15.
; Therefore we can load at least two bytes without checking it.
	CMP r2,r3              ; ptr<stop => HI
	LDRHIB r14,[r3],#1     ;   r14 = *ptr++
	RSBHI r5,r5,#24        ; (HI) available = 32-(available+=8)
	RSBLS r5,r5,#32        ; (LS) r5 = 32-available
	ORRHI r4,r14,LSL r5    ;   r4 = window|=r14<<32-available
	CMPHI r2,r3            ;   ptr<stop => HI
	LDRHIB r14,[r3],#1     ;     r14 = *ptr++
	SUBHI r5,#8            ;     available += 8
	; (HI) Stall...
	ORRHI r4,r14,LSL r5    ;     r4 = window|=r14<<32-available
; We can use unsigned compares for both the pointers and for available
;  (allowing us to chain condition codes) because available will never be
;  larger than 32 (or we wouldn't be here), and thus 32-available will never be
;  negative.
	CMPHI r2,r3            ;     ptr<stop => HI
	CMPHI r5,#7            ;       available<=24 => HI
	LDRHIB r14,[r3],#1     ;         r14 = *ptr++
	SUBHI r5,#8            ;         available += 8
	; (HI) Stall...
	ORRHI r4,r14,LSL r5    ;         r4 = window|=r14<<32-available
	CMP r2,r3              ; ptr<stop => HI
	MOVLS r5,#-1<<30       ; (LS) available = OC_LOTS_OF_BITS+32
	CMPHI r5,#7            ; (HI) available<=24 => HI
	LDRHIB r14,[r3],#1     ; (HI)   r14 = *ptr++
	SUBHI r5,#8            ; (HI)   available += 8
	; (HI) Stall...
	ORRHI r4,r14,LSL r5    ; (HI)   r4 = window|=r14<<32-available
	RSB r14,r10,#32        ; r14 = 32-n
	MOV r14,r4,LSR r14     ; r14 = bits=window>>32-n
	ADD r12,r12,r14        ;
	LDRSH r12,[r12,r14]    ; r12 = node=_tree[node+1+bits]
	RSB r5,r5,#32          ; r5 = available
	; Stall...
	RSBS r14,r12,#0        ; r14 = -node, node>0 => MI
	BMI oc_huff_token_decode_continue
	MOV r10,r14,LSR #8     ; r10 = n=node>>8
	MOV r4,r4,LSL r10      ; r4 = window<<=n
	SUB r5,r10             ; r5 = available-=n
	STMIB r0,{r3-r5}       ; ptr = r3
	                       ; window = r4
	                       ; available = r5
	AND r0,r14,#255        ; r0 = node&255
	LDMFD r13!,{r4,r5,r10,pc}
	ENDP

	END
