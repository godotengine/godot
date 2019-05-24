#version 400

#define ON

float sum = 0.0;

void main()
{

#ifdef ON
//yes
sum += 1.0;
#endif

#ifdef OFF
//no
sum += 20.0;
#endif

#if defined(ON)
//yes
sum += 300.0;
#endif

#if defined(OFF)
//no
sum += 4000.0;
#endif

#if !defined(ON)
//no
sum += 50000.0;
#endif

#ifndef OFF
//yes
sum += 600000.0;
#else
//no
sum += 0.6;
#endif

#if defined(ON) && defined(OFF)
//no
sum += 0.7;
#elif !defined(OFF)
//yes
sum += 7000000.0;
#endif

#if defined(ON) && !defined(OFF)
//yes
sum += 80000000.0;
#endif

#if defined(OFF) || defined(ON)
//yes
sum += 900000000.0;
#endif

#if NEVER_DEFINED
//no
sum += 0.04;
#else
sum += 0.05;
#endif

// sum should be 987600301.7
    gl_Position = vec4(sum);
}

#define A 0
# define B 0
 #   define C 0

#if (A == B) || (A == C)
#error good1
#endif

#if A == B || (A == C)
#error good2
#endif

#if (A == B || (A == C))
#error good3
#endif

#if (AA == BB) || (AA == CC)
#error good4
#endif

#if AA == BB || (AA == CC)
#error good5
#endif

#if ((AA == BB || (AA == CC)))
#error good6
#endif

#if (A == B || (A == C)
#error bad1
#endif

#if A == B || A == C)
#error bad2
#endif

#if (A == B || (A == C)
#error bad3
#endif

#if AA == BB) || (AA == CC)
#error bad4
#endif

#if AA == BB || (AA == CC
#error bad5
#endif

#if ((AA == BB || (AA == CC))))
#error bad6
#endif extra tokens

int linenumber = __LINE__;
int filenumber = __FILE__;
int version = __VERSION__;

#define PI (3.14)
#define TWOPI (2.0 * PI)
float twoPi = TWOPI;

//#define PASTE(a,b) a ## b
//float PASTE(tod, ay) = 17;

"boo" // ERROR
int a = length("aoenatuh");  // ERROR
#define QUOTE "abcd"  // okay
'int';  // ERROR
#define SINGLE 'a'   // okay
// ERROR: all the following are reserved
#define GL_
#define GL_Macro 1
#define __M 
#define M__
#define ABC__DE abc

#if 4
#else extra
#elif
// ERROR elif after else
#endif

#if blah
  #if 0
  #else extra
    #ifdef M
    #else
    #else
    // ERROR else after else
    #endif  extra
  #endif
#endif

#define m1(a,a)  // ERROR
#define m2(a,b)

// okay
#define m3 (a)
#define m3 (a)

// ERROR
#define m4(b)
#define m4

// ERROR
#define m5 (b)
#define m5(b)

// ERROR
#define m6(a)
#define m6(a,b)

// ERROR (whitespace)
#define m7 (a)
#define m7 ( a)

#define m80(a,b) is + exactly m3 the same
#define m80(a,b) is + exactly m3 the same

// ERROR
#define m8(a,b) almost + exactly m3 the same
#define m8(a,b) almost + exactly m3 thee same

// ERROR
#define m9(a,b,c) aoe
#define m9(a,d,c) aoe

#define n1 0xf
int n = n1;

#define f1 .08e-2Lf
double f = f1;

#undef __VERSION__
#undef GL_ARB_texture_rectangle

#
 # 
		#	
##
# # 
# 0x25
####
####ff
#########ff fg 0x25
#pragma
#pragma(aoent)
	#	pragma
#pragma STDGL
#pragma	 optimize(	on)
#pragma  optimize(off)
#pragma debug( on)
#pragma debug(off	)
#pragma	 optimize(	on) anoteun
#pragma  optimize(off
#pragma debug( on) (
#pragma debug(off	aoeua)
#pragma	 optimize(	on)
#pragma  optimize(off,)
#pragma debug( on, aoeu)
#pragma debugoff	)
#pragma aontheu natoeh uantheo uasotea noeahuonea uonethau onethuanoeth aunotehau noeth anthoeua  anoethuantoeh uantoehu natoehu naoteh unotaehu noethua onetuh aou
# \

# \
 error good continuation

#flizbit

#define directive error

#directive directive was expanded

#line 12000
#error line should be 12000
#line 13000 7
#error line should be 13000, string 7
#define L1 14000
#define L2 13
#define F1 5
#define F2 7
#line L1 + L2
#error line should be 14013, string 7
#line L1 + L2 F1 + F2 //  antoeuh sat  comment
#error line should be 14013, string 12
#line L1 + L2 + F1 + F2
#error line should be 14025, string 12
#line 1234 F1 + F2 extra
#define empty_extra
#line 1235 F1 + F2 empty_extra
#define moreEmpty empty_extra
#line 1236 F1 + F2 moreEmpty empty_extra // okay, lots of nothin
#line 1237 F1 + F2 moreEmpty empty_extra extra  // ERROR, 'extra'
#line 1238 F1 + F2 moreEmpty empty_extra
#line 1239 empty_extra F1 empty_extra + empty_extra F2 empty_extra moreEmpty empty_extra
#line (20000)
#error line should be 20000
#line (20000+10)
#error line should be 20010
#line +20020
#error line should be 20020

#define VAL1 1.0
#define VAL2 2.0

#define RES2 /* test a multiline
                comment in a macro definition */ (RES1 * VAL2)
#define RES1    (VAL2 / VAL1) 
#define RES2    /* comment */(RES1 * VAL2)
#define /* */SUM_VALUES   (RES2 + RES1)

void foo234()
{
    gl_Position = vec4(SUM_VALUES);
}

// more whitespace recording tests
#define SPACE_AT_END(a,b) spaceAtEndIsOkay
#define SPACE_AT_END(a,b) spaceAtEndIsOkay // space at end

#define SPACE_AT_BEGIN(a,b)spaceAtBeginIsOkay
#define SPACE_AT_BEGIN(a,b) spaceAtBeginIsOkay

// space in middle is an error
#define SPACE_IN_MIDDLE(a,b) space +in middle
#define SPACE_IN_MIDDLE(a,b) space + in middle

#define FIRSTPART 17
#define SECONDPART + 5

#if FIRSTPART SECONDPART == 22
#error good evaluation 1
#endif

#if moreEmpty FIRSTPART moreEmpty SECONDPART moreEmpty == moreEmpty 22 moreEmpty
#error good evaluation 2
#endif

// ERRORS...
#line 9000
#if defined(OUNH
#endif
#if defined OUNH)
#endif

// recursion (okay)
#define RECURSE RECURSE
int RECURSE;
#define R2 R1
#define R1 R2
#undef RECURSE
int R1 = RECURSE;

#define FOOOM(a,b) a + b
int aoeua = FOOOM;
#if FOOOM
#endif

#line 9500
#if\376
#endif
#if \376
#endif
#if \377
#endif
#error\377
#error \ 376
#error \377

// ERROR for macro expansion to yield 'defined'
#line 9600
#define DEF_MAC
#define DEF_DEFINED defined
#if DEF_DEFINED DEF_MAC
#error DEF_DEFINED then
#else
#error DEF_DEFINED else
#endif

#line 10000
#if 1
#else
// ERROR, missing #endif