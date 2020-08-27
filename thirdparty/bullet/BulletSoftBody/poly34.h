// poly34.h : solution of cubic and quartic equation
// (c) Khashin S.I. http://math.ivanovo.ac.ru/dalgebra/Khashin/index.html
// khash2 (at) gmail.com

#ifndef POLY_34
#define POLY_34
#include "LinearMath/btScalar.h"
// x - array of size 2
// return 2: 2 real roots x[0], x[1]
// return 0: pair of complex roots: x[0]i*x[1]
int SolveP2(btScalar* x, btScalar a, btScalar b); // solve equation x^2 + a*x + b = 0

// x - array of size 3
// return 3: 3 real roots x[0], x[1], x[2]
// return 1: 1 real root x[0] and pair of complex roots: x[1]i*x[2]
int SolveP3(btScalar* x, btScalar a, btScalar b, btScalar c); // solve cubic equation x^3 + a*x^2 + b*x + c = 0

// x - array of size 4
// return 4: 4 real roots x[0], x[1], x[2], x[3], possible multiple roots
// return 2: 2 real roots x[0], x[1] and complex x[2]i*x[3],
// return 0: two pair of complex roots: x[0]i*x[1],  x[2]i*x[3],
int SolveP4(btScalar* x, btScalar a, btScalar b, btScalar c, btScalar d); // solve equation x^4 + a*x^3 + b*x^2 + c*x + d = 0 by Dekart-Euler method

// x - array of size 5
// return 5: 5 real roots x[0], x[1], x[2], x[3], x[4], possible multiple roots
// return 3: 3 real roots x[0], x[1], x[2] and complex x[3]i*x[4],
// return 1: 1 real root x[0] and two pair of complex roots: x[1]i*x[2],  x[3]i*x[4],
int SolveP5(btScalar* x, btScalar a, btScalar b, btScalar c, btScalar d, btScalar e); // solve equation x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0

//-----------------------------------------------------------------------------
// And some additional functions for internal use.
// Your may remove this definitions from here
int SolveP4Bi(btScalar* x, btScalar b, btScalar d); // solve equation x^4 + b*x^2 + d = 0
int SolveP4De(btScalar* x, btScalar b, btScalar c, btScalar d); // solve equation x^4 + b*x^2 + c*x + d = 0
void CSqrt(btScalar x, btScalar y, btScalar& a, btScalar& b); // returns as a+i*s,  sqrt(x+i*y)
btScalar N4Step(btScalar x, btScalar a, btScalar b, btScalar c, btScalar d); // one Newton step for x^4 + a*x^3 + b*x^2 + c*x + d
btScalar SolveP5_1(btScalar a, btScalar b, btScalar c, btScalar d, btScalar e); // return real root of x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0
#endif
