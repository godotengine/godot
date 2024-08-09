/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                     Written by Philip Hazel
            Copyright (c) 2023 University of Cambridge

-----------------------------------------------------------------------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the University of Cambridge nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------
*/

/* This file contains functions to implement checked integer operation */

#ifndef PCRE2_PCRE2TEST
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pcre2_internal.h"
#endif

/*************************************************
*        Checked Integer Multiplication          *
*************************************************/

/*
Arguments:
  r         A pointer to PCRE2_SIZE to store the answer
  a, b      Two integers

Returns:    Bool indicating if the operation overflows

It is modeled after C23's <stdckdint.h> interface
The INT64_OR_DOUBLE type is a 64-bit integer type when available,
otherwise double. */

BOOL
PRIV(ckd_smul)(PCRE2_SIZE *r, int a, int b)
{
#ifdef HAVE_BUILTIN_MUL_OVERFLOW
PCRE2_SIZE m;

if (__builtin_mul_overflow(a, b, &m)) return TRUE;

*r = m;
#else
INT64_OR_DOUBLE m;

#ifdef PCRE2_DEBUG
if (a < 0 || b < 0) abort();
#endif

m = (INT64_OR_DOUBLE)a * (INT64_OR_DOUBLE)b;

#if defined INT64_MAX || defined int64_t
if (sizeof(m) > sizeof(*r) && m > (INT64_OR_DOUBLE)PCRE2_SIZE_MAX) return TRUE;
*r = (PCRE2_SIZE)m;
#else
if (m > PCRE2_SIZE_MAX) return TRUE;
*r = m;
#endif

#endif

return FALSE;
}

/* End of pcre_chkdint.c */
