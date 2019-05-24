#version 450

// side test verifies multiple rounds of argument expansion
#define bear SecondExpansion
#define mmmB bear
#define mmmA(a) a
int mmmA(mmmB);                    // mmmB -> bear, and then in mmmA(), bear -> SecondExpansion

// pasting skips the first round of expansion
#define mmcatmmdog PostPasteExpansion
#define mmcat cat
#define mmdog dog
#define mmp(a,b) a## b
int mmp(mmcat, mmdog);             // mmcat/mmdog not expanded, mmcatmmdog -> PostPasteExpansion

// multi-token pre
#define mmtokpastepre(a) a##27
mmtokpastepre(float foo);          // should declare "float foo27;"

// multi-token post
#define mmtokpastepost(a) uni ##a
mmtokpastepost(form float foo155); // should declare "uniform float foo155;"

// non-first argument
#define foo ShouldntExpandToThis
#define semi ;
#define bothpaste(a,b) a##b
float bothpaste(foo, 719);          // should declare "float foo719;"
#define secpaste(a,b) a bar ## b
secpaste(uniform float, foo semi)   // should declare "uniform float barfoo;"

// no args
#define noArg fl##oat
noArg argless;

// bad location
#define bad1 ## float
bad1 dc1;
#define bad2 float ##
bad2 dc2;

// multiple ##
#define multiPaste(a, b, c) a##or##b flo##at foo##c
multiPaste(unif, m, 875);

// too long
#define simplePaste(a,b) a##b
// 1020 + 5 characters
float simplePaste(ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF012345, 12345);

// non-identifiers
int a = simplePaste(11,12);

// operators
#define MAKE_OP(L, R) L ## R
const int aop = 10;
const int bop = 4;
int cop = aop MAKE_OP(<, <) bop;
bool dop = aop MAKE_OP(!,=) bop;

#define MAKE_OP3(L, M, R) L ## M ## R

void foo()
{
    int e = 16;
    e MAKE_OP3(>,>,=) 2;

    // recovery from bad op
    bool f = e MAKE_OP(>,!) 5;
}

// arguments: should make 'uniform int argPaste2;'
#define M_NEST(q) int q
#define M_OUTER(p) M_NEST(p##2)
uniform M_OUTER(argPaste);
// should make 'uniform int argPaste20suff;'
#define M_NEST2(q) int q ## suff
#define M_OUTER2(p) M_NEST2(p ## 20)
uniform M_OUTER2(argPaste);

#define rec(x)##
rec(rec())

#define bax(bay)
#define baz bax(/##)
baz
