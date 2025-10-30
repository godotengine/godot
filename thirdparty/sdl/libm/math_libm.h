/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef math_libm_h_
#define math_libm_h_

#include "SDL_internal.h"

/* Math routines from uClibc: http://www.uclibc.org */

extern double SDL_uclibc_atan(double x);
extern double SDL_uclibc_atan2(double y, double x);
extern double SDL_uclibc_copysign(double x, double y);
extern double SDL_uclibc_cos(double x);
extern double SDL_uclibc_exp(double x);
extern double SDL_uclibc_fabs(double x);
extern double SDL_uclibc_floor(double x);
extern double SDL_uclibc_fmod(double x, double y);
extern int SDL_uclibc_isinf(double x);
extern int SDL_uclibc_isinff(float x);
extern int SDL_uclibc_isnan(double x);
extern int SDL_uclibc_isnanf(float x);
extern double SDL_uclibc_log(double x);
extern double SDL_uclibc_log10(double x);
extern double SDL_uclibc_modf(double x, double *y);
extern double SDL_uclibc_pow(double x, double y);
extern double SDL_uclibc_scalbn(double x, int n);
extern double SDL_uclibc_sin(double x);
extern double SDL_uclibc_sqrt(double x);
extern double SDL_uclibc_tan(double x);

#endif /* math_libm_h_ */
