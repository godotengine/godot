/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "quant_common.h"

static const int dc_qlookup[QINDEX_RANGE] =
{
    4,    5,    6,    7,    8,    9,   10,   10,   11,   12,   13,   14,   15,   16,   17,   17,
    18,   19,   20,   20,   21,   21,   22,   22,   23,   23,   24,   25,   25,   26,   27,   28,
    29,   30,   31,   32,   33,   34,   35,   36,   37,   37,   38,   39,   40,   41,   42,   43,
    44,   45,   46,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,
    59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,
    75,   76,   76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,
    91,   93,   95,   96,   98,  100,  101,  102,  104,  106,  108,  110,  112,  114,  116,  118,
    122,  124,  126,  128,  130,  132,  134,  136,  138,  140,  143,  145,  148,  151,  154,  157,
};

static const int ac_qlookup[QINDEX_RANGE] =
{
    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
    20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,
    36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,
    52,   53,   54,   55,   56,   57,   58,   60,   62,   64,   66,   68,   70,   72,   74,   76,
    78,   80,   82,   84,   86,   88,   90,   92,   94,   96,   98,  100,  102,  104,  106,  108,
    110,  112,  114,  116,  119,  122,  125,  128,  131,  134,  137,  140,  143,  146,  149,  152,
    155,  158,  161,  164,  167,  170,  173,  177,  181,  185,  189,  193,  197,  201,  205,  209,
    213,  217,  221,  225,  229,  234,  239,  245,  249,  254,  259,  264,  269,  274,  279,  284,
};


int vp8_dc_quant(int QIndex, int Delta)
{
    int retval;

    QIndex = QIndex + Delta;

    if (QIndex > 127)
        QIndex = 127;
    else if (QIndex < 0)
        QIndex = 0;

    retval = dc_qlookup[ QIndex ];
    return retval;
}

int vp8_dc2quant(int QIndex, int Delta)
{
    int retval;

    QIndex = QIndex + Delta;

    if (QIndex > 127)
        QIndex = 127;
    else if (QIndex < 0)
        QIndex = 0;

    retval = dc_qlookup[ QIndex ] * 2;
    return retval;

}
int vp8_dc_uv_quant(int QIndex, int Delta)
{
    int retval;

    QIndex = QIndex + Delta;

    if (QIndex > 127)
        QIndex = 127;
    else if (QIndex < 0)
        QIndex = 0;

    retval = dc_qlookup[ QIndex ];

    if (retval > 132)
        retval = 132;

    return retval;
}

int vp8_ac_yquant(int QIndex)
{
    int retval;

    if (QIndex > 127)
        QIndex = 127;
    else if (QIndex < 0)
        QIndex = 0;

    retval = ac_qlookup[ QIndex ];
    return retval;
}

int vp8_ac2quant(int QIndex, int Delta)
{
    int retval;

    QIndex = QIndex + Delta;

    if (QIndex > 127)
        QIndex = 127;
    else if (QIndex < 0)
        QIndex = 0;

    /* For all x in [0..284], x*155/100 is bitwise equal to (x*101581) >> 16.
     * The smallest precision for that is '(x*6349) >> 12' but 16 is a good
     * word size. */
    retval = (ac_qlookup[ QIndex ] * 101581) >> 16;

    if (retval < 8)
        retval = 8;

    return retval;
}
int vp8_ac_uv_quant(int QIndex, int Delta)
{
    int retval;

    QIndex = QIndex + Delta;

    if (QIndex > 127)
        QIndex = 127;
    else if (QIndex < 0)
        QIndex = 0;

    retval = ac_qlookup[ QIndex ];
    return retval;
}
