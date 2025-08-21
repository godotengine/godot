#pragma once

extern float test_fadd(float a, float b);
extern float test_fsub(float a, float b);
extern float test_fmul(float a, float b);
extern float test_fdiv(float a, float b);
extern float test_fmax(float a, float b);
extern float test_fmin(float a, float b);
extern double test_ftod(float val);
extern float test_dtof(double val);

extern float  test_fneg(float val);
extern double test_dneg(double val);

extern float test_fmadd(float a, float b, float c);
extern float test_fmsub(float a, float b, float c);
extern float test_fnmadd(float a, float b, float c);
extern float test_fnmsub(float a, float b, float c);

extern float  test_fsqrt(float val);
extern double test_dsqrt(double val);
extern float  test_fpow(float val, float);
extern double test_dpow(double val, double);

extern float test_sinf(float val);
extern float test_cosf(float val);
extern float test_tanf(float val);
