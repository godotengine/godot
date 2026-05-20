	CODE32

	EXPORT	|CrcUpdateT4@16|

	AREA	|.text|, CODE, ARM

	MACRO
	CRC32_STEP_1

	ldrb    r4, [r1], #1
	subs    r2, r2, #1
	eor     r4, r4, r0
	and     r4, r4, #0xFF
	ldr     r4, [r3, +r4, lsl #2]
	eor     r0, r4, r0, lsr #8

	MEND


	MACRO
	CRC32_STEP_4 $STREAM_WORD
	
	eor     r7, r7, r8
	eor     r7, r7, r9
	eor     r0, r0, r7
	eor     r0, r0, $STREAM_WORD
	ldr     $STREAM_WORD, [r1], #4
	
	and     r7, r0, #0xFF
	and     r8, r0, #0xFF00
	and     r9, r0, #0xFF0000
	and     r0, r0, #0xFF000000

	ldr     r7, [r6, +r7, lsl #2]
	ldr     r8, [r5, +r8, lsr #6]
	ldr     r9, [r4, +r9, lsr #14]
	ldr     r0, [r3, +r0, lsr #22]
	
	MEND


|CrcUpdateT4@16| PROC

	stmdb   sp!, {r4-r11, lr}
	cmp     r2, #0
	beq     |$fin|

|$v1|
	tst     r1, #7
	beq     |$v2|
	CRC32_STEP_1
	bne     |$v1|

|$v2|
	cmp     r2, #16
	blo     |$v3|

	ldr     r10, [r1], #4
	ldr     r11, [r1], #4

	add     r4, r3, #0x400 
	add     r5, r3, #0x800
	add     r6, r3, #0xC00

	mov     r7, #0
	mov     r8, #0
	mov     r9, #0

	sub     r2, r2, #16

|$loop|
	; pld     [r1, #0x40]

	CRC32_STEP_4 r10
	CRC32_STEP_4 r11

	subs    r2, r2, #8
	bhs     |$loop|

	sub     r1, r1, #8
	add     r2, r2, #16

	eor     r7, r7, r8
	eor     r7, r7, r9
	eor     r0, r0, r7

|$v3|
	cmp     r2, #0
	beq     |$fin|

|$v4|
	CRC32_STEP_1
	bne     |$v4|

|$fin|
	ldmia   sp!, {r4-r11, pc}

|CrcUpdateT4@16| ENDP

	END
