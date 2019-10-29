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

const opus_uint8 silk_pitch_lag_iCDF[ 2 * ( PITCH_EST_MAX_LAG_MS - PITCH_EST_MIN_LAG_MS ) ] = {
       253,    250,    244,    233,    212,    182,    150,    131,
       120,    110,     98,     85,     72,     60,     49,     40,
        32,     25,     19,     15,     13,     11,      9,      8,
         7,      6,      5,      4,      3,      2,      1,      0
};

const opus_uint8 silk_pitch_delta_iCDF[21] = {
       210,    208,    206,    203,    199,    193,    183,    168,
       142,    104,     74,     52,     37,     27,     20,     14,
        10,      6,      4,      2,      0
};

const opus_uint8 silk_pitch_contour_iCDF[34] = {
       223,    201,    183,    167,    152,    138,    124,    111,
        98,     88,     79,     70,     62,     56,     50,     44,
        39,     35,     31,     27,     24,     21,     18,     16,
        14,     12,     10,      8,      6,      4,      3,      2,
         1,      0
};

const opus_uint8 silk_pitch_contour_NB_iCDF[11] = {
       188,    176,    155,    138,    119,     97,     67,     43,
        26,     10,      0
};

const opus_uint8 silk_pitch_contour_10_ms_iCDF[12] = {
       165,    119,     80,     61,     47,     35,     27,     20,
        14,      9,      4,      0
};

const opus_uint8 silk_pitch_contour_10_ms_NB_iCDF[3] = {
       113,     63,      0
};


