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

static const opus_uint8 silk_NLSF_CB1_NB_MB_Q8[ 320 ] = {
        12,     35,     60,     83,    108,    132,    157,    180,
       206,    228,     15,     32,     55,     77,    101,    125,
       151,    175,    201,    225,     19,     42,     66,     89,
       114,    137,    162,    184,    209,    230,     12,     25,
        50,     72,     97,    120,    147,    172,    200,    223,
        26,     44,     69,     90,    114,    135,    159,    180,
       205,    225,     13,     22,     53,     80,    106,    130,
       156,    180,    205,    228,     15,     25,     44,     64,
        90,    115,    142,    168,    196,    222,     19,     24,
        62,     82,    100,    120,    145,    168,    190,    214,
        22,     31,     50,     79,    103,    120,    151,    170,
       203,    227,     21,     29,     45,     65,    106,    124,
       150,    171,    196,    224,     30,     49,     75,     97,
       121,    142,    165,    186,    209,    229,     19,     25,
        52,     70,     93,    116,    143,    166,    192,    219,
        26,     34,     62,     75,     97,    118,    145,    167,
       194,    217,     25,     33,     56,     70,     91,    113,
       143,    165,    196,    223,     21,     34,     51,     72,
        97,    117,    145,    171,    196,    222,     20,     29,
        50,     67,     90,    117,    144,    168,    197,    221,
        22,     31,     48,     66,     95,    117,    146,    168,
       196,    222,     24,     33,     51,     77,    116,    134,
       158,    180,    200,    224,     21,     28,     70,     87,
       106,    124,    149,    170,    194,    217,     26,     33,
        53,     64,     83,    117,    152,    173,    204,    225,
        27,     34,     65,     95,    108,    129,    155,    174,
       210,    225,     20,     26,     72,     99,    113,    131,
       154,    176,    200,    219,     34,     43,     61,     78,
        93,    114,    155,    177,    205,    229,     23,     29,
        54,     97,    124,    138,    163,    179,    209,    229,
        30,     38,     56,     89,    118,    129,    158,    178,
       200,    231,     21,     29,     49,     63,     85,    111,
       142,    163,    193,    222,     27,     48,     77,    103,
       133,    158,    179,    196,    215,    232,     29,     47,
        74,     99,    124,    151,    176,    198,    220,    237,
        33,     42,     61,     76,     93,    121,    155,    174,
       207,    225,     29,     53,     87,    112,    136,    154,
       170,    188,    208,    227,     24,     30,     52,     84,
       131,    150,    166,    186,    203,    229,     37,     48,
        64,     84,    104,    118,    156,    177,    201,    230
};

static const opus_uint8 silk_NLSF_CB1_iCDF_NB_MB[ 64 ] = {
       212,    178,    148,    129,    108,     96,     85,     82,
        79,     77,     61,     59,     57,     56,     51,     49,
        48,     45,     42,     41,     40,     38,     36,     34,
        31,     30,     21,     12,     10,      3,      1,      0,
       255,    245,    244,    236,    233,    225,    217,    203,
       190,    176,    175,    161,    149,    136,    125,    114,
       102,     91,     81,     71,     60,     52,     43,     35,
        28,     20,     19,     18,     12,     11,      5,      0
};

static const opus_uint8 silk_NLSF_CB2_SELECT_NB_MB[ 160 ] = {
        16,      0,      0,      0,      0,     99,     66,     36,
        36,     34,     36,     34,     34,     34,     34,     83,
        69,     36,     52,     34,    116,    102,     70,     68,
        68,    176,    102,     68,     68,     34,     65,     85,
        68,     84,     36,    116,    141,    152,    139,    170,
       132,    187,    184,    216,    137,    132,    249,    168,
       185,    139,    104,    102,    100,     68,     68,    178,
       218,    185,    185,    170,    244,    216,    187,    187,
       170,    244,    187,    187,    219,    138,    103,    155,
       184,    185,    137,    116,    183,    155,    152,    136,
       132,    217,    184,    184,    170,    164,    217,    171,
       155,    139,    244,    169,    184,    185,    170,    164,
       216,    223,    218,    138,    214,    143,    188,    218,
       168,    244,    141,    136,    155,    170,    168,    138,
       220,    219,    139,    164,    219,    202,    216,    137,
       168,    186,    246,    185,    139,    116,    185,    219,
       185,    138,    100,    100,    134,    100,    102,     34,
        68,     68,    100,     68,    168,    203,    221,    218,
       168,    167,    154,    136,    104,     70,    164,    246,
       171,    137,    139,    137,    155,    218,    219,    139
};

static const opus_uint8 silk_NLSF_CB2_iCDF_NB_MB[ 72 ] = {
       255,    254,    253,    238,     14,      3,      2,      1,
         0,    255,    254,    252,    218,     35,      3,      2,
         1,      0,    255,    254,    250,    208,     59,      4,
         2,      1,      0,    255,    254,    246,    194,     71,
        10,      2,      1,      0,    255,    252,    236,    183,
        82,      8,      2,      1,      0,    255,    252,    235,
       180,     90,     17,      2,      1,      0,    255,    248,
       224,    171,     97,     30,      4,      1,      0,    255,
       254,    236,    173,     95,     37,      7,      1,      0
};

static const opus_uint8 silk_NLSF_CB2_BITS_NB_MB_Q5[ 72 ] = {
       255,    255,    255,    131,      6,    145,    255,    255,
       255,    255,    255,    236,     93,     15,     96,    255,
       255,    255,    255,    255,    194,     83,     25,     71,
       221,    255,    255,    255,    255,    162,     73,     34,
        66,    162,    255,    255,    255,    210,    126,     73,
        43,     57,    173,    255,    255,    255,    201,    125,
        71,     48,     58,    130,    255,    255,    255,    166,
       110,     73,     57,     62,    104,    210,    255,    255,
       251,    123,     65,     55,     68,    100,    171,    255
};

static const opus_uint8 silk_NLSF_PRED_NB_MB_Q8[ 18 ] = {
       179,    138,    140,    148,    151,    149,    153,    151,
       163,    116,     67,     82,     59,     92,     72,    100,
        89,     92
};

static const opus_int16 silk_NLSF_DELTA_MIN_NB_MB_Q15[ 11 ] = {
       250,      3,      6,      3,      3,      3,      4,      3,
         3,      3,    461
};

const silk_NLSF_CB_struct silk_NLSF_CB_NB_MB =
{
    32,
    10,
    SILK_FIX_CONST( 0.18, 16 ),
    SILK_FIX_CONST( 1.0 / 0.18, 6 ),
    silk_NLSF_CB1_NB_MB_Q8,
    silk_NLSF_CB1_iCDF_NB_MB,
    silk_NLSF_PRED_NB_MB_Q8,
    silk_NLSF_CB2_SELECT_NB_MB,
    silk_NLSF_CB2_iCDF_NB_MB,
    silk_NLSF_CB2_BITS_NB_MB_Q5,
    silk_NLSF_DELTA_MIN_NB_MB_Q15,
};
