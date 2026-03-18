/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE2 is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2024 University of Cambridge

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

#ifndef PCRE2_UTIL_H_IDEMPOTENT_GUARD
#define PCRE2_UTIL_H_IDEMPOTENT_GUARD

/* Assertion macros */

#ifdef PCRE2_DEBUG

#if defined(HAVE_ASSERT_H) && !defined(NDEBUG)
#include <assert.h>
#endif

/* PCRE2_ASSERT(x) can be used to inject an assert() for conditions
that the code below doesn't support. It is a NOP for non debug builds
but in debug builds will print information about the location of the
code where it triggered and crash.

It is meant to work like assert(), and therefore the expression used
should indicate what the expected state is, and shouldn't have any
side-effects. */

#if defined(HAVE_ASSERT_H) && !defined(NDEBUG)
#define PCRE2_ASSERT(x) assert(x)
#else
#define PCRE2_ASSERT(x) do                                            \
{                                                                     \
  if (!(x))                                                           \
  {                                                                   \
  fprintf(stderr, "Assertion failed at " __FILE__ ":%d\n", __LINE__); \
  abort();                                                            \
  }                                                                   \
} while(0)
#endif

/* PCRE2_UNREACHABLE() can be used to mark locations on the code that
shouldn't be reached. In non debug builds is defined as a hint for
the compiler to eliminate any code after it, so it is useful also for
performance reasons, but should be used with care because if it is
ever reached will trigger Undefined Behaviour and if you are lucky a
crash. In debug builds it will report the location where it was triggered
and crash. One important point to consider when using this macro, is
that it is only implemented for a few compilers, and therefore can't
be relied on to always be active either, so if it is followed by some
code it is important to make sure that the whole thing is safe to
use even if the macro is not there (ex: make sure there is a `break`
after it if used at the end of a `case`) and to test your code also
with a configuration where the macro will be a NOP. */

#if defined(HAVE_ASSERT_H) && !defined(NDEBUG)
#define PCRE2_UNREACHABLE()                                         \
assert(((void)"Execution reached unexpected point", 0))
#else
#define PCRE2_UNREACHABLE() do                                      \
{                                                                   \
fprintf(stderr, "Execution reached unexpected point at " __FILE__   \
                ":%d\n", __LINE__);                                 \
abort();                                                            \
} while(0)
#endif

/* PCRE2_DEBUG_UNREACHABLE() is a debug only version of the previous
macro. It is meant to be used in places where the code is handling
an error situation in code that shouldn't be reached, but that has
some sort of fallback code to normally handle the error. When in
doubt you should use this instead of the previous macro. Like in
the previous case, it is a good idea to document as much as possible
the reason and the actions that should be taken if it ever triggers. */

#define PCRE2_DEBUG_UNREACHABLE() PCRE2_UNREACHABLE()

#endif /* PCRE2_DEBUG */

#ifndef PCRE2_DEBUG_UNREACHABLE
#define PCRE2_DEBUG_UNREACHABLE() do {} while(0)
#endif

#ifndef PCRE2_UNREACHABLE
#ifdef HAVE_BUILTIN_UNREACHABLE
#define PCRE2_UNREACHABLE() __builtin_unreachable()
#elif defined(HAVE_BUILTIN_ASSUME)
#define PCRE2_UNREACHABLE() __assume(0)
#else
#define PCRE2_UNREACHABLE() do {} while(0)
#endif
#endif /* !PCRE2_UNREACHABLE */

#ifndef PCRE2_ASSERT
#define PCRE2_ASSERT(x) do {} while(0)
#endif

#endif /* PCRE2_UTIL_H_IDEMPOTENT_GUARD */

/* End of pcre2_util.h */
