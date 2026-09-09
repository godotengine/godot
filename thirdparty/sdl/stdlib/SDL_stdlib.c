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
#include "SDL_internal.h"

// This file contains portable stdlib functions for SDL

#include "../libm/math_libm.h"

double SDL_atan(double x)
{
#ifdef HAVE_ATAN
    return atan(x);
#else
    return SDL_uclibc_atan(x);
#endif
}

float SDL_atanf(float x)
{
#ifdef HAVE_ATANF
    return atanf(x);
#else
    return (float)SDL_atan((double)x);
#endif
}

double SDL_atan2(double y, double x)
{
#ifdef HAVE_ATAN2
    return atan2(y, x);
#else
    return SDL_uclibc_atan2(y, x);
#endif
}

float SDL_atan2f(float y, float x)
{
#ifdef HAVE_ATAN2F
    return atan2f(y, x);
#else
    return (float)SDL_atan2((double)y, (double)x);
#endif
}

double SDL_acos(double val)
{
#ifdef HAVE_ACOS
    return acos(val);
#else
    double result;
    if (val == -1.0) {
        result = SDL_PI_D;
    } else {
        result = SDL_atan(SDL_sqrt(1.0 - val * val) / val);
        if (result < 0.0) {
            result += SDL_PI_D;
        }
    }
    return result;
#endif
}

float SDL_acosf(float val)
{
#ifdef HAVE_ACOSF
    return acosf(val);
#else
    return (float)SDL_acos((double)val);
#endif
}

double SDL_asin(double val)
{
#ifdef HAVE_ASIN
    return asin(val);
#else
    double result;
    if (val == -1.0) {
        result = -(SDL_PI_D / 2.0);
    } else {
        result = (SDL_PI_D / 2.0) - SDL_acos(val);
    }
    return result;
#endif
}

float SDL_asinf(float val)
{
#ifdef HAVE_ASINF
    return asinf(val);
#else
    return (float)SDL_asin((double)val);
#endif
}

double SDL_ceil(double x)
{
#ifdef HAVE_CEIL
    return ceil(x);
#else
    double integer = SDL_floor(x);
    double fraction = x - integer;
    if (fraction > 0.0) {
        integer += 1.0;
    }
    return integer;
#endif // HAVE_CEIL
}

float SDL_ceilf(float x)
{
#ifdef HAVE_CEILF
    return ceilf(x);
#else
    return (float)SDL_ceil((double)x);
#endif
}

double SDL_copysign(double x, double y)
{
#ifdef HAVE_COPYSIGN
    return copysign(x, y);
#elif defined(HAVE__COPYSIGN)
    return _copysign(x, y);
#elif defined(__WATCOMC__) && defined(__386__)
    // this is nasty as hell, but it works..
    unsigned int *xi = (unsigned int *)&x,
                 *yi = (unsigned int *)&y;
    xi[1] = (yi[1] & 0x80000000) | (xi[1] & 0x7fffffff);
    return x;
#else
    return SDL_uclibc_copysign(x, y);
#endif // HAVE_COPYSIGN
}

float SDL_copysignf(float x, float y)
{
#ifdef HAVE_COPYSIGNF
    return copysignf(x, y);
#else
    return (float)SDL_copysign((double)x, (double)y);
#endif
}

double SDL_cos(double x)
{
#ifdef HAVE_COS
    return cos(x);
#else
    return SDL_uclibc_cos(x);
#endif
}

float SDL_cosf(float x)
{
#ifdef HAVE_COSF
    return cosf(x);
#else
    return (float)SDL_cos((double)x);
#endif
}

double SDL_exp(double x)
{
#ifdef HAVE_EXP
    return exp(x);
#else
    return SDL_uclibc_exp(x);
#endif
}

float SDL_expf(float x)
{
#ifdef HAVE_EXPF
    return expf(x);
#else
    return (float)SDL_exp((double)x);
#endif
}

double SDL_fabs(double x)
{
#ifdef HAVE_FABS
    return fabs(x);
#else
    return SDL_uclibc_fabs(x);
#endif
}

float SDL_fabsf(float x)
{
#ifdef HAVE_FABSF
    return fabsf(x);
#else
    return (float)SDL_fabs((double)x);
#endif
}

double SDL_floor(double x)
{
#ifdef HAVE_FLOOR
    return floor(x);
#else
    return SDL_uclibc_floor(x);
#endif
}

float SDL_floorf(float x)
{
#ifdef HAVE_FLOORF
    return floorf(x);
#else
    return (float)SDL_floor((double)x);
#endif
}

double SDL_trunc(double x)
{
#ifdef HAVE_TRUNC
    return trunc(x);
#else
    if (x >= 0.0f) {
        return SDL_floor(x);
    } else {
        return SDL_ceil(x);
    }
#endif
}

float SDL_truncf(float x)
{
#ifdef HAVE_TRUNCF
    return truncf(x);
#else
    return (float)SDL_trunc((double)x);
#endif
}

double SDL_fmod(double x, double y)
{
#ifdef HAVE_FMOD
    return fmod(x, y);
#else
    return SDL_uclibc_fmod(x, y);
#endif
}

float SDL_fmodf(float x, float y)
{
#ifdef HAVE_FMODF
    return fmodf(x, y);
#else
    return (float)SDL_fmod((double)x, (double)y);
#endif
}

int SDL_isinf(double x)
{
#ifdef HAVE_ISINF
    return isinf(x);
#else
    return SDL_uclibc_isinf(x);
#endif
}

int SDL_isinff(float x)
{
#ifdef HAVE_ISINF_FLOAT_MACRO
    return isinf(x);
#elif defined(HAVE_ISINFF)
    return isinff(x);
#else
    return SDL_uclibc_isinff(x);
#endif
}

int SDL_isnan(double x)
{
#ifdef HAVE_ISNAN
    return isnan(x);
#else
    return SDL_uclibc_isnan(x);
#endif
}

int SDL_isnanf(float x)
{
#ifdef HAVE_ISNAN_FLOAT_MACRO
    return isnan(x);
#elif defined(HAVE_ISNANF)
    return isnanf(x);
#else
    return SDL_uclibc_isnanf(x);
#endif
}

double SDL_log(double x)
{
#ifdef HAVE_LOG
    return log(x);
#else
    return SDL_uclibc_log(x);
#endif
}

float SDL_logf(float x)
{
#ifdef HAVE_LOGF
    return logf(x);
#else
    return (float)SDL_log((double)x);
#endif
}

double SDL_log10(double x)
{
#ifdef HAVE_LOG10
    return log10(x);
#else
    return SDL_uclibc_log10(x);
#endif
}

float SDL_log10f(float x)
{
#ifdef HAVE_LOG10F
    return log10f(x);
#else
    return (float)SDL_log10((double)x);
#endif
}

double SDL_modf(double x, double *y)
{
#ifdef HAVE_MODF
    return modf(x, y);
#else
    return SDL_uclibc_modf(x, y);
#endif
}

float SDL_modff(float x, float *y)
{
#ifdef HAVE_MODFF
    return modff(x, y);
#else
    double double_result, double_y;
    double_result = SDL_modf((double)x, &double_y);
    *y = (float)double_y;
    return (float)double_result;
#endif
}

double SDL_pow(double x, double y)
{
#ifdef HAVE_POW
    return pow(x, y);
#else
    return SDL_uclibc_pow(x, y);
#endif
}

float SDL_powf(float x, float y)
{
#ifdef HAVE_POWF
    return powf(x, y);
#else
    return (float)SDL_pow((double)x, (double)y);
#endif
}

double SDL_round(double arg)
{
#if defined HAVE_ROUND
    return round(arg);
#else
    if (arg >= 0.0) {
        return SDL_floor(arg + 0.5);
    } else {
        return SDL_ceil(arg - 0.5);
    }
#endif
}

float SDL_roundf(float arg)
{
#if defined HAVE_ROUNDF
    return roundf(arg);
#else
    return (float)SDL_round((double)arg);
#endif
}

long SDL_lround(double arg)
{
#if defined HAVE_LROUND
    return lround(arg);
#else
    return (long)SDL_round(arg);
#endif
}

long SDL_lroundf(float arg)
{
#if defined HAVE_LROUNDF
    return lroundf(arg);
#else
    return (long)SDL_round((double)arg);
#endif
}

double SDL_scalbn(double x, int n)
{
#ifdef HAVE_SCALBN
    return scalbn(x, n);
#elif defined(HAVE__SCALB)
    return _scalb(x, n);
#elif defined(HAVE_LIBC) && defined(HAVE_FLOAT_H) && (FLT_RADIX == 2)
    /* from scalbn(3): If FLT_RADIX equals 2 (which is
     * usual), then scalbn() is equivalent to ldexp(3). */
    return ldexp(x, n);
#else
    return SDL_uclibc_scalbn(x, n);
#endif
}

float SDL_scalbnf(float x, int n)
{
#ifdef HAVE_SCALBNF
    return scalbnf(x, n);
#else
    return (float)SDL_scalbn((double)x, n);
#endif
}

double SDL_sin(double x)
{
#ifdef HAVE_SIN
    return sin(x);
#else
    return SDL_uclibc_sin(x);
#endif
}

float SDL_sinf(float x)
{
#ifdef HAVE_SINF
    return sinf(x);
#else
    return (float)SDL_sin((double)x);
#endif
}

double SDL_sqrt(double x)
{
#ifdef HAVE_SQRT
    return sqrt(x);
#else
    return SDL_uclibc_sqrt(x);
#endif
}

float SDL_sqrtf(float x)
{
#ifdef HAVE_SQRTF
    return sqrtf(x);
#else
    return (float)SDL_sqrt((double)x);
#endif
}

double SDL_tan(double x)
{
#ifdef HAVE_TAN
    return tan(x);
#else
    return SDL_uclibc_tan(x);
#endif
}

float SDL_tanf(float x)
{
#ifdef HAVE_TANF
    return tanf(x);
#else
    return (float)SDL_tan((double)x);
#endif
}

int SDL_abs(int x)
{
#ifdef HAVE_ABS
    return abs(x);
#else
    return (x < 0) ? -x : x;
#endif
}

int SDL_isalpha(int x) { return (SDL_isupper(x)) || (SDL_islower(x)); }
int SDL_isalnum(int x) { return (SDL_isalpha(x)) || (SDL_isdigit(x)); }
int SDL_isdigit(int x) { return ((x) >= '0') && ((x) <= '9'); }
int SDL_isxdigit(int x) { return (((x) >= 'A') && ((x) <= 'F')) || (((x) >= 'a') && ((x) <= 'f')) || (SDL_isdigit(x)); }
int SDL_ispunct(int x) { return (SDL_isgraph(x)) && (!SDL_isalnum(x)); }
int SDL_isspace(int x) { return ((x) == ' ') || ((x) == '\t') || ((x) == '\r') || ((x) == '\n') || ((x) == '\f') || ((x) == '\v'); }
int SDL_isupper(int x) { return ((x) >= 'A') && ((x) <= 'Z'); }
int SDL_islower(int x) { return ((x) >= 'a') && ((x) <= 'z'); }
int SDL_isprint(int x) { return ((x) >= ' ') && ((x) < '\x7f'); }
int SDL_isgraph(int x) { return (SDL_isprint(x)) && ((x) != ' '); }
int SDL_iscntrl(int x) { return (((x) >= '\0') && ((x) <= '\x1f')) || ((x) == '\x7f'); }
int SDL_toupper(int x) { return ((x) >= 'a') && ((x) <= 'z') ? ('A' + ((x) - 'a')) : (x); }
int SDL_tolower(int x) { return ((x) >= 'A') && ((x) <= 'Z') ? ('a' + ((x) - 'A')) : (x); }
int SDL_isblank(int x) { return ((x) == ' ') || ((x) == '\t'); }

void *SDL_aligned_alloc(size_t alignment, size_t size)
{
    size_t padding;
    Uint8 *result = NULL;
    size_t requested_size = size;

    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    padding = (alignment - (size % alignment));

    if (SDL_size_add_check_overflow(size, alignment, &size) &&
        SDL_size_add_check_overflow(size, sizeof(void *), &size) &&
        SDL_size_add_check_overflow(size, padding, &size)) {
        void *original = SDL_malloc(size);
        if (original) {
            // Make sure we have enough space to store the original pointer
            result = (Uint8 *)original + sizeof(original);

            // Align the pointer we're going to return
            result += alignment - (((size_t)result) % alignment);

            // Store the original pointer right before the returned value
            SDL_memcpy(result - sizeof(original), &original, sizeof(original));

            // Initialize the padding to zero
            if (padding > 0) {
                SDL_memset(result + requested_size, 0, padding);
            }
        }
    }
    return result;
}

void SDL_aligned_free(void *mem)
{
    if (mem) {
        void *original;
        SDL_memcpy(&original, ((Uint8 *)mem - sizeof(original)), sizeof(original));
        SDL_free(original);
    }
}
