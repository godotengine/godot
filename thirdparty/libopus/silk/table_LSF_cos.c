/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
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
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tables.h"

/* Cosine approximation table for LSF conversion */
/* Q12 values (even) */
const opus_int16 silk_LSFCosTab_FIX_Q12[ LSF_COS_TAB_SZ_FIX + 1 ] = {
            8192,             8190,             8182,             8170,
            8152,             8130,             8104,             8072,
            8034,             7994,             7946,             7896,
            7840,             7778,             7714,             7644,
            7568,             7490,             7406,             7318,
            7226,             7128,             7026,             6922,
            6812,             6698,             6580,             6458,
            6332,             6204,             6070,             5934,
            5792,             5648,             5502,             5352,
            5198,             5040,             4880,             4718,
            4552,             4382,             4212,             4038,
            3862,             3684,             3502,             3320,
            3136,             2948,             2760,             2570,
            2378,             2186,             1990,             1794,
            1598,             1400,             1202,             1002,
             802,              602,              402,              202,
               0,             -202,             -402,             -602,
            -802,            -1002,            -1202,            -1400,
           -1598,            -1794,            -1990,            -2186,
           -2378,            -2570,            -2760,            -2948,
           -3136,            -3320,            -3502,            -3684,
           -3862,            -4038,            -4212,            -4382,
           -4552,            -4718,            -4880,            -5040,
           -5198,            -5352,            -5502,            -5648,
           -5792,            -5934,            -6070,            -6204,
           -6332,            -6458,            -6580,            -6698,
           -6812,            -6922,            -7026,            -7128,
           -7226,            -7318,            -7406,            -7490,
           -7568,            -7644,            -7714,            -7778,
           -7840,            -7896,            -7946,            -7994,
           -8034,            -8072,            -8104,            -8130,
           -8152,            -8170,            -8182,            -8190,
           -8192
};
