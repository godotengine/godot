#version 300 es

// this file cont\
ains no errors other than the #error which are there to see if line numbering for errors is correct

#error e1

float f\
oo;  // same as 'float foo;'

#error e2

#define MAIN void main() \
   {                     \
gl_Position = vec4(foo); \
} 

#error e3

MAIN

vec4 foo2(vec4 a)
{                                
  vec4 b = a;       \
  return b;                   
}

// aoeuntheo unatehutna \ antaehnathe 
// anteonuth $ natohe " '
// anteonuth     natohe

#define FOO int /* \
*/ goodDecl;

FOO

#define A int q1 = \ 1
#define B int q2 = \1
#define C int q3 = $ 1
#define D int q4 = @ 1

const highp int a1 = \ 4;  // ERROR
const highp int a2 = @ 3;  // ERROR
const highp int a3 = $4;   // ERROR
const highp int a4 = a2\;  // ERROR

A;
B;
C;
D;

# \

# \
    error good continuation

#define AA1 a \ b
#define AA2 a \\ b
#define AA3 a \\\ b
#define AA4 a \\\\ b

// anoetuh nonaetu \\\\\\
still in comment

const int abdece = 10;
const int aoeuntaoehu = abd\
\
\
\
\
\
ece;

float funkyf = \
.\
1\
2\
3\
e\
+\
1\
7\
;\
int funkyh\
=\
0\
x\
f\
4\
;
int funkyo =\
0\
4\
2\
;
int c = \
11;
int d = 1\
2;

#define FOOM(a,b) a + b

#if FO\
OM(2\
,\
3)
int bar103 = 17;
#endif

// ERROR
#if FOOM(2,
3)
int bar104 = 19;
#endif

// ERROR
#if FOOM(
2,3)
int bar105 = 19;
#endif

int bar106 = FOOM(5,7);
int bar107 = FOOM  // okay
    (
    2
    ,
    3
    )
    ;

void foo203209409()
{
    bar107 \
+= 37;
    bar107 *\
= 38;
    bar107 /=\
39;
    bar107 +\
41;
}

#define QUOTE "ab\
cd"

void foo230920394()
{
    // syntax error
    bar107 +\
 = 42;
}
