// 7zAsm.S -- ASM macros for arm64
// : Igor Pavlov : Public domain

#if defined(__clang_major__) && __clang_major__ >= 20
  // clang21 : error: identifier 'and' conflicts with a C++ keyword [-Wc++-keyword]
  // #pragma GCC diagnostic ignored "-Wc++-keyword" // clang 21
  #pragma GCC diagnostic ignored "-Wc++-compat"  // clang 12+
#endif

#define  r0 x0
#define  r1 x1
#define  r2 x2
#define  r3 x3
#define  r4 x4
#define  r5 x5
#define  r6 x6
#define  r7 x7
#define  r8 x8
#define  r9 x9
#define  r10 x10
#define  r11 x11
#define  r12 x12
#define  r13 x13
#define  r14 x14
#define  r15 x15
#define  r16 x16
#define  r17 x17
#define  r18 x18
#define  r19 x19
#define  r20 x20
#define  r21 x21
#define  r22 x22
#define  r23 x23
#define  r24 x24
#define  r25 x25
#define  r26 x26
#define  r27 x27
#define  r28 x28
#define  r29 x29
#define  r30 x30

#define  REG_ABI_PARAM_0 r0
#define  REG_ABI_PARAM_1 r1
#define  REG_ABI_PARAM_2 r2


.macro p2_add reg:req, param:req
        add     \reg, \reg, \param
.endm

.macro p2_sub reg:req, param:req
        sub     \reg, \reg, \param
.endm

.macro p2_sub_s reg:req, param:req
        subs    \reg, \reg, \param
.endm

.macro p2_and reg:req, param:req
        and     \reg, \reg, \param
.endm

.macro xor reg:req, param:req
        eor     \reg, \reg, \param
.endm

.macro or reg:req, param:req
        orr     \reg, \reg, \param
.endm

.macro shl reg:req, param:req
        lsl     \reg, \reg, \param
.endm

.macro shr reg:req, param:req
        lsr     \reg, \reg, \param
.endm

.macro sar reg:req, param:req
        asr     \reg, \reg, \param
.endm

.macro p1_neg reg:req
        neg     \reg, \reg
.endm

.macro dec reg:req
        sub     \reg, \reg, 1
.endm

.macro dec_s reg:req
        subs    \reg, \reg, 1
.endm

.macro inc reg:req
        add     \reg, \reg, 1
.endm

.macro inc_s reg:req
        adds    \reg, \reg, 1
.endm


.macro imul reg:req, param:req
        mul     \reg, \reg, \param
.endm

/*
arm64 and arm use reverted c flag after subs/cmp instructions:
  arm64-arm   :     x86
 b.lo / b.cc  :  jb  / jc
 b.hs / b.cs  :  jae / jnc
*/

.macro jmp lab:req
        b       \lab
.endm

.macro je lab:req
        b.eq    \lab
.endm

.macro jz lab:req
        b.eq    \lab
.endm

.macro jnz lab:req
        b.ne    \lab
.endm

.macro jne lab:req
        b.ne    \lab
.endm

.macro jb lab:req
        b.lo    \lab
.endm

.macro jbe lab:req
        b.ls    \lab
.endm

.macro ja lab:req
        b.hi    \lab
.endm

.macro jae lab:req
        b.hs    \lab
.endm


.macro cmove dest:req, srcTrue:req
        csel    \dest, \srcTrue, \dest, eq
.endm

.macro cmovne dest:req, srcTrue:req
        csel    \dest, \srcTrue, \dest, ne
.endm

.macro cmovs dest:req, srcTrue:req
        csel    \dest, \srcTrue, \dest, mi
.endm

.macro cmovns dest:req, srcTrue:req
        csel    \dest, \srcTrue, \dest, pl
.endm

.macro cmovb dest:req, srcTrue:req
        csel    \dest, \srcTrue, \dest, lo
.endm

.macro cmovae dest:req, srcTrue:req
        csel    \dest, \srcTrue, \dest, hs
.endm


.macro MY_ALIGN_16 macro
	.p2align 4,, (1 << 4) - 1
.endm

.macro MY_ALIGN_32 macro
        .p2align 5,, (1 << 5) - 1
.endm

.macro MY_ALIGN_64 macro
        .p2align 6,, (1 << 6) - 1
.endm
