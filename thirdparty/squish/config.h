/* -----------------------------------------------------------------------------

    Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   -------------------------------------------------------------------------- */

#ifndef SQUISH_CONFIG_H
#define SQUISH_CONFIG_H

// Set to 1 when building squish to use Altivec instructions.
#ifndef SQUISH_USE_ALTIVEC
#define SQUISH_USE_ALTIVEC 0
#endif

// Set to 1 or 2 when building squish to use SSE or SSE2 instructions.
#ifndef SQUISH_USE_SSE
#define SQUISH_USE_SSE 0
#endif

// Internally set SQUISH_USE_SIMD when either Altivec or SSE is available.
#if SQUISH_USE_ALTIVEC && SQUISH_USE_SSE
#error "Cannot enable both Altivec and SSE!"
#endif
#if SQUISH_USE_ALTIVEC || SQUISH_USE_SSE
#define SQUISH_USE_SIMD 1
#else
#define SQUISH_USE_SIMD 0
#endif

#endif // ndef SQUISH_CONFIG_H
