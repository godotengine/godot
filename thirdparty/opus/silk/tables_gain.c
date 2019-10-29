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

#ifdef __cplusplus
extern "C"
{
#endif

const opus_uint8 silk_gain_iCDF[ 3 ][ N_LEVELS_QGAIN / 8 ] =
{
{
       224,    112,     44,     15,      3,      2,      1,      0
},
{
       254,    237,    192,    132,     70,     23,      4,      0
},
{
       255,    252,    226,    155,     61,     11,      2,      0
}
};

const opus_uint8 silk_delta_gain_iCDF[ MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1 ] = {
       250,    245,    234,    203,     71,     50,     42,     38,
        35,     33,     31,     29,     28,     27,     26,     25,
        24,     23,     22,     21,     20,     19,     18,     17,
        16,     15,     14,     13,     12,     11,     10,      9,
         8,      7,      6,      5,      4,      3,      2,      1,
         0
};

#ifdef __cplusplus
}
#endif
