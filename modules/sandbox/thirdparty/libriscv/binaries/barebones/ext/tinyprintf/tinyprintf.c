/*
File: tinyprintf.c

Copyright (C) 2004  Kustaa Nyholm

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#include "tinyprintf.h"


/*
 * Configuration
 */

/* Enable long int support */
#define PRINTF_LONG_SUPPORT

/* Enable long long int support (implies long int support) */
#define PRINTF_LONG_LONG_SUPPORT

/* Enable %z (size_t) support */
#define PRINTF_SIZE_T_SUPPORT

/*
 * Configuration adjustments
 */
#ifdef PRINTF_SIZE_T_SUPPORT
#include <sys/types.h>
#endif

#include <limits.h>

#ifdef PRINTF_LONG_LONG_SUPPORT
# define PRINTF_LONG_SUPPORT
#endif

/* __SIZEOF_<type>__ defined at least by gcc */
#ifdef __SIZEOF_POINTER__
# define SIZEOF_POINTER __SIZEOF_POINTER__
#endif
#ifdef __SIZEOF_LONG_LONG__
# define SIZEOF_LONG_LONG __SIZEOF_LONG_LONG__
#endif
#ifdef __SIZEOF_LONG__
# define SIZEOF_LONG __SIZEOF_LONG__
#endif
#ifdef __SIZEOF_INT__
# define SIZEOF_INT __SIZEOF_INT__
#endif

#ifdef __GNUC__
# define _TFP_GCC_NO_INLINE_  __attribute__ ((noinline))
#else
# define _TFP_GCC_NO_INLINE_
#endif

/*
 * Implementation
 */
struct param {
    char l:1;               /**<  Add leading character */
    char alt:1;             /**<  alternate form */
    char uc:1;              /**<  Upper case (for base16 only) */
    char align_left:1;      /**<  0 == align right (default), 1 == align left */
    char lchr;              /**<  Leading character */
    unsigned int width;     /**<  field width */
    int precision;          /**<  field precision */
    char sign;              /**<  The sign to display (if any) */
    unsigned int base;      /**<  number base (e.g.: 8, 10, 16) */
    char *bf;               /**<  Buffer to output */
};


#ifdef PRINTF_LONG_LONG_SUPPORT
static void _TFP_GCC_NO_INLINE_ ulli2a(
    unsigned long long int num, struct param *p)
{
    int n = 0;
    unsigned long long int d = 1;
    char *bf = p->bf;
    while (num / d >= p->base)
        d *= p->base;
    while (d != 0) {
        int dgt = num / d;
        num %= d;
        d /= p->base;
        if (n || dgt > 0 || d == 0) {
            *bf++ = dgt + (dgt < 10 ? '0' : (p->uc ? 'A' : 'a') - 10);
            ++n;
        }
    }
    *bf = 0;
}

static void lli2a(long long int num, struct param *p)
{
    if (num < 0) {
        num = -num;
        p->sign = '-';
    }
    ulli2a(num, p);
}
#endif

#ifdef PRINTF_LONG_SUPPORT
static void uli2a(unsigned long int num, struct param *p)
{
    int n = 0;
    unsigned long int d = 1;
    char *bf = p->bf;
    while (num / d >= p->base)
        d *= p->base;
    while (d != 0) {
        int dgt = num / d;
        num %= d;
        d /= p->base;
        if (n || dgt > 0 || d == 0) {
            *bf++ = dgt + (dgt < 10 ? '0' : (p->uc ? 'A' : 'a') - 10);
            ++n;
        }
    }
    *bf = 0;
}

static void li2a(long num, struct param *p)
{
    if (num < 0) {
        num = -num;
        p->sign = '-';
    }
    uli2a(num, p);
}
#endif

static void ui2a(unsigned int num, struct param *p)
{
    int n = 0;
    unsigned int d = 1;
    char *bf = p->bf;
    while (num / d >= p->base)
        d *= p->base;
    while (d != 0) {
        int dgt = num / d;
        num %= d;
        d /= p->base;
        if (n || dgt > 0 || d == 0) {
            *bf++ = dgt + (dgt < 10 ? '0' : (p->uc ? 'A' : 'a') - 10);
            ++n;
        }
    }
    *bf = 0;
}

static void i2a(int num, struct param *p)
{
    if (num < 0) {
        num = -num;
        p->sign = '-';
    }
    ui2a(num, p);
}

static int a2d(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    else if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    else if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    else
        return -1;
}

static char a2u(char ch, const char **src, int base, unsigned int *nump)
{
    const char *p = *src;
    unsigned int num = 0;
    int digit;
    while ((digit = a2d(ch)) >= 0) {
        if (digit > base)
            break;
        num = num * base + digit;
        ch = *p++;
    }
    *src = p;
    *nump = num;
    return ch;
}

static void putchw(void *putp, putcf putf, const struct param *p)
{
    char ch;
    int n = p->width;
    char *bf = p->bf;
    int precision = p->precision;

    /* Number of filling characters */
    while (precision-- > 0 && *bf++ && n-- > 0);
    if (p->sign)
        n--;
    if (p->alt && p->base == 16)
        n -= 2;
    else if (p->alt && p->base == 8)
        n--;

    /* Fill with space to align to the right, before alternate or sign */
    if (!p->l && !p->align_left) {
        while (n-- > 0)
            putf(putp, ' ');
    }

    /* print sign */
    if (p->sign)
        putf(putp, p->sign);

    /* Alternate */
    if (p->alt && p->base == 16) {
        putf(putp, '0');
        putf(putp, (p->uc ? 'X' : 'x'));
    } else if (p->alt && p->base == 8) {
        putf(putp, '0');
    }

    /* Fill with zeros, after alternate or sign */
    if (p->l) {
        while (n-- > 0)
            putf(putp, p->lchr);
    }

    /* Put actual buffer */
    bf = p->bf;
    precision = p->precision;
    while (precision-- > 0 && (ch = *bf++))
        putf(putp, ch);

    /* Fill with space to align to the left, after string */
    if (!p->l && p->align_left) {
        while (n-- > 0)
            putf(putp, ' ');
    }
}

void tfp_format(void *putp, putcf putf, const char *fmt, va_list va)
{
    struct param p;
#ifdef PRINTF_LONG_SUPPORT
    char bf[23];  /* long = 64b on some architectures */
#else
    char bf[12];  /* int = 32b on some architectures */
#endif
    char ch;
    int w;
    p.bf = bf;

    while ((ch = *(fmt++))) {
        if (ch != '%') {
            putf(putp, ch);
        } else {
#ifdef PRINTF_LONG_SUPPORT
            char lng = 0;  /* 1 for long, 2 for long long */
#endif
            /* Init parameter struct */
            p.l = 0;
            p.lchr = ' ';
            p.alt = 0;
            p.width = 0;
            p.precision = INT_MAX;
            p.align_left = 0;
            p.sign = 0;

            /* Flags */
            while ((ch = *(fmt++))) {
                switch (ch) {
                case '-':
                    p.align_left = 1;
                    continue;
                case '0':
                case ' ':
                    p.l = 1;
                    p.lchr = ch;
                    continue;
                case '#':
                    p.alt = 1;
                    continue;
                case '*':
                    w = va_arg(va, int);
                    if (w < 0) {
                        p.align_left = 1;
                        p.width = (unsigned int) -w;
                    } else {
                        p.width = (unsigned int) w;
                    }
                    continue;
                default:
                    break;
                }
                break;
            }

            /* Width */
            if (ch >= '0' && ch <= '9') {
                ch = a2u(ch, &fmt, 10, &(p.width));
            }

            /* We accept 'x.y' format but don't support it completely:
             * we ignore the 'y' digit => this ignores 0-fill
             * size and makes it == width (ie. 'x'), but use
             * its value to set fill character to '0' if greater than 1.
             */
            if (ch == '.') {
              p.l = 1;
              ch = *(fmt++);
              if (ch >= '0' && ch <= '9') {
                unsigned int num;
                ch = a2u(ch, &fmt, 10, &num);
                if (num > 1) {
                  p.lchr = '0';
                }
              } else if (ch == '*') {
                  ch = *(fmt++);
                  int precision = va_arg(va, int);
                  if (precision >= 0) {
                      p.precision = precision;
                  }
              }
            }

#ifdef PRINTF_SIZE_T_SUPPORT
# ifdef PRINTF_LONG_SUPPORT
            if (ch == 'z') {
                ch = *(fmt++);
                if (sizeof(size_t) == sizeof(unsigned long int))
                    lng = 1;
#  ifdef PRINTF_LONG_LONG_SUPPORT
                else if (sizeof(size_t) == sizeof(unsigned long long int))
                    lng = 2;
#  endif
            } else
# endif
#endif

#ifdef PRINTF_LONG_SUPPORT
            if (ch == 'l') {
                ch = *(fmt++);
                lng = 1;
#ifdef PRINTF_LONG_LONG_SUPPORT
                if (ch == 'l') {
                  ch = *(fmt++);
                  lng = 2;
                }
#endif
            }
#endif
            switch (ch) {
            case 0:
                goto abort;
            case 'u':
                p.base = 10;
#ifdef PRINTF_LONG_SUPPORT
#ifdef PRINTF_LONG_LONG_SUPPORT
                if (2 == lng)
                    ulli2a(va_arg(va, unsigned long long int), &p);
                else
#endif
                  if (1 == lng)
                    uli2a(va_arg(va, unsigned long int), &p);
                else
#endif
                    ui2a(va_arg(va, unsigned int), &p);
                putchw(putp, putf, &p);
                break;
            case 'd':
            case 'i':
                p.base = 10;
#ifdef PRINTF_LONG_SUPPORT
#ifdef PRINTF_LONG_LONG_SUPPORT
                if (2 == lng)
                    lli2a(va_arg(va, long long int), &p);
                else
#endif
                  if (1 == lng)
                    li2a(va_arg(va, long int), &p);
                else
#endif
                    i2a(va_arg(va, int), &p);
                putchw(putp, putf, &p);
                break;
#ifdef SIZEOF_POINTER
            case 'p':
                p.alt = 1;
# if defined(SIZEOF_INT) && SIZEOF_POINTER <= SIZEOF_INT
                lng = 0;
# elif defined(SIZEOF_LONG) && SIZEOF_POINTER <= SIZEOF_LONG
                lng = 1;
# elif defined(SIZEOF_LONG_LONG) && SIZEOF_POINTER <= SIZEOF_LONG_LONG
                lng = 2;
# endif
#endif
				// fallthrough
            case 'x':
            case 'X':
                p.base = 16;
                p.uc = (ch == 'X')?1:0;
#ifdef PRINTF_LONG_SUPPORT
#ifdef PRINTF_LONG_LONG_SUPPORT
                if (2 == lng)
                    ulli2a(va_arg(va, unsigned long long int), &p);
                else
#endif
                  if (1 == lng)
                    uli2a(va_arg(va, unsigned long int), &p);
                else
#endif
                    ui2a(va_arg(va, unsigned int), &p);
                putchw(putp, putf, &p);
                break;
            case 'o':
                p.base = 8;
                ui2a(va_arg(va, unsigned int), &p);
                putchw(putp, putf, &p);
                break;
            case 'c':
                putf(putp, (char)(va_arg(va, int)));
                break;
            case 's':
                p.bf = va_arg(va, char *);
                putchw(putp, putf, &p);
                p.bf = bf;
                break;
            case '%':
                putf(putp, ch);
            default:
                break;
            }
        }
    }
 abort:;
}

#if TINYPRINTF_DEFINE_TFP_PRINTF
static putcf stdout_putf;
static void *stdout_putp;

void init_printf(void *putp, putcf putf)
{
    stdout_putf = putf;
    stdout_putp = putp;
}

void tfp_printf(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    tfp_format(stdout_putp, stdout_putf, fmt, va);
    va_end(va);
}
#endif

#if TINYPRINTF_DEFINE_TFP_SPRINTF
struct _vsnprintf_putcf_data
{
  size_t dest_capacity;
  char *dest;
  size_t num_chars;
};

static void _vsnprintf_putcf(void *p, char c)
{
  struct _vsnprintf_putcf_data *data = (struct _vsnprintf_putcf_data*)p;
  if (data->num_chars < data->dest_capacity)
    data->dest[data->num_chars] = c;
  data->num_chars ++;
}

int tfp_vsnprintf(char *str, size_t size, const char *format, va_list ap)
{
  struct _vsnprintf_putcf_data data;

  if (size < 1)
    return 0;

  data.dest = str;
  data.dest_capacity = size-1;
  data.num_chars = 0;
  tfp_format(&data, _vsnprintf_putcf, format, ap);

  if (data.num_chars < data.dest_capacity)
    data.dest[data.num_chars] = '\0';
  else
    data.dest[data.dest_capacity] = '\0';

  return data.num_chars;
}

int tfp_snprintf(char *str, size_t size, const char *format, ...)
{
  va_list ap;
  int retval;

  va_start(ap, format);
  retval = tfp_vsnprintf(str, size, format, ap);
  va_end(ap);
  return retval;
}

struct _vsprintf_putcf_data
{
  char *dest;
  size_t num_chars;
};

static void _vsprintf_putcf(void *p, char c)
{
  struct _vsprintf_putcf_data *data = (struct _vsprintf_putcf_data*)p;
  data->dest[data->num_chars++] = c;
}

int tfp_vsprintf(char *str, const char *format, va_list ap)
{
  struct _vsprintf_putcf_data data;
  data.dest = str;
  data.num_chars = 0;
  tfp_format(&data, _vsprintf_putcf, format, ap);
  data.dest[data.num_chars] = '\0';
  return data.num_chars;
}

int tfp_sprintf(char *str, const char *format, ...)
{
  va_list ap;
  int retval;

  va_start(ap, format);
  retval = tfp_vsprintf(str, format, ap);
  va_end(ap);
  return retval;
}
#endif
