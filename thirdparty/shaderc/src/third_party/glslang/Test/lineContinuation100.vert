#version 100

// non-line continuation comment \
#error good error



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
/*@*/
/* *@/*/
//@

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
    error bad continuation

#define QUOTE "ab\
cd"
