/*
  Copyright (c) 2005-2009, The Musepack Development Team
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

  * Neither the name of the The Musepack Development Team nor the
  names of its contributors may be used to endorse or promote
  products derived from this software without specific prior
  written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/// \file huffman.c
/// Implementations of sv7/sv8 huffman decoding functions.
#include "huffman.h"


// sv7 huffman tables
static const mpc_huffman mpc_table_HuffHdr [10] = {
{0x8000, 1, 0}, {0x6000, 3, 1}, {0x5e00, 7, -4}, {0x5d80, 9, 3}, {0x5d00, 9, 4}, {0x5c00, 8, -5}, {0x5800, 6, 2}, {0x5000, 5, -3}, {0x4000, 4, -2}, {0x0, 2, -1}
};
mpc_lut_data mpc_HuffHdr = {mpc_table_HuffHdr};

const mpc_huffman mpc_table_HuffSCFI [4] = {
	{0x8000, 1, 1}, {0x6000, 3, 2}, {0x4000, 3, 0}, {0x0, 2, 3}
};

static const mpc_huffman mpc_table_HuffDSCF [16] = {
	{0xf800, 5, 5}, {0xf000, 5, -4}, {0xe000, 4, 3}, {0xd000, 4, -3}, {0xc000, 4, 8}, {0xa000, 3, 1}, {0x9000, 4, 0}, {0x8800, 5, -5}, {0x8400, 6, 7}, {0x8000, 6, -7}, {0x6000, 3, -1}, {0x4000, 3, 2}, {0x3000, 4, 4}, {0x2800, 5, 6}, {0x2000, 5, -6}, {0x0, 3, -2}
};
mpc_lut_data mpc_HuffDSCF = {mpc_table_HuffDSCF};

static const mpc_huffman mpc_table_HuffQ1 [2] [27] = {
	{
		{0xe000, 3, 13}, {0xdc00, 6, 26}, {0xd800, 6, 0}, {0xd400, 6, 20}, {0xd000, 6, 6}, {0xc000, 4, 14}, {0xb000, 4, 12}, {0xa000, 4, 4}, {0x9000, 4, 22}, {0x8c00, 6, 8}, {0x8800, 6, 18}, {0x8400, 6, 24}, {0x8000, 6, 2}, {0x7000, 4, 16}, {0x6000, 4, 10}, {0x5800, 5, 17}, {0x5000, 5, 9}, {0x4800, 5, 1}, {0x4000, 5, 25}, {0x3800, 5, 5}, {0x3000, 5, 21}, {0x2800, 5, 3}, {0x2000, 5, 11}, {0x1800, 5, 15}, {0x1000, 5, 23}, {0x800, 5, 19}, {0x0, 5, 7}
	}, {
		{0x8000, 1, 13}, {0x7e00, 7, 15}, {0x7c00, 7, 1}, {0x7a00, 7, 11}, {0x7800, 7, 7}, {0x7600, 7, 17}, {0x7400, 7, 25}, {0x7200, 7, 19}, {0x7180, 9, 8}, {0x7100, 9, 18}, {0x7080, 9, 2}, {0x7000, 9, 24}, {0x6e00, 7, 3}, {0x6c00, 7, 23}, {0x6a00, 7, 21}, {0x6800, 7, 5}, {0x6700, 8, 0}, {0x6600, 8, 26}, {0x6500, 8, 6}, {0x6400, 8, 20}, {0x6000, 6, 9}, {0x5000, 4, 14}, {0x4000, 4, 12}, {0x3000, 4, 4}, {0x2000, 4, 22}, {0x1000, 4, 16}, {0x0, 4, 10}
	}
};

static const mpc_huffman mpc_table_HuffQ2 [2] [25] = {
	{
		{0xf000, 4, 13}, {0xe000, 4, 17}, {0xd000, 4, 7}, {0xc000, 4, 11}, {0xbc00, 6, 1}, {0xb800, 6, 23}, {0xb600, 7, 4}, {0xb400, 7, 20}, {0xb200, 7, 0}, {0xb000, 7, 24}, {0xa800, 5, 22}, {0xa000, 5, 10}, {0x8000, 3, 12}, {0x7800, 5, 2}, {0x7000, 5, 14}, {0x6000, 4, 6}, {0x5000, 4, 18}, {0x4000, 4, 8}, {0x3000, 4, 16}, {0x2800, 5, 9}, {0x2000, 5, 5}, {0x1800, 5, 15}, {0x1000, 5, 21}, {0x800, 5, 19}, {0x0, 5, 3}
	}, {
		{0xf800, 5, 18}, {0xf000, 5, 6}, {0xe800, 5, 8}, {0xe700, 8, 3}, {0xe6c0, 10, 24}, {0xe680, 10, 4}, {0xe640, 10, 0}, {0xe600, 10, 20}, {0xe400, 7, 23}, {0xe200, 7, 1}, {0xe000, 7, 19}, {0xd800, 5, 16}, {0xd600, 7, 15}, {0xd400, 7, 21}, {0xd200, 7, 9}, {0xd000, 7, 5}, {0xcc00, 6, 2}, {0xc800, 6, 10}, {0xc400, 6, 14}, {0xc000, 6, 22}, {0x8000, 2, 12}, {0x6000, 3, 13}, {0x4000, 3, 17}, {0x2000, 3, 11}, {0x0, 3, 7}
	}
};

static const mpc_huffman mpc_table_HuffQ3 [2] [7] = {
	{
		{0xe000, 3, 1}, {0xd000, 4, 3}, {0xc000, 4, -3}, {0xa000, 3, 2}, {0x8000, 3, -2}, {0x4000, 2, 0}, {0x0, 2, -1}
	}, {
		{0xc000, 2, 0}, {0x8000, 2, -1}, {0x4000, 2, 1}, {0x3000, 4, -2}, {0x2800, 5, 3}, {0x2000, 5, -3}, {0x0, 3, 2}
	}
};

static const mpc_huffman mpc_table_HuffQ4 [2] [9] = {
	{
		{0xe000, 3, 0}, {0xc000, 3, -1}, {0xa000, 3, 1}, {0x8000, 3, -2}, {0x6000, 3, 2}, {0x5000, 4, -4}, {0x4000, 4, 4}, {0x2000, 3, 3}, {0x0, 3, -3}
	}, {
		{0xe000, 3, 1}, {0xd000, 4, 2}, {0xc000, 4, -3}, {0x8000, 2, 0}, {0x6000, 3, -2}, {0x5000, 4, 3}, {0x4800, 5, -4}, {0x4000, 5, 4}, {0x0, 2, -1}
	}
};

static const mpc_huffman mpc_table_HuffQ5 [2] [15] = {
	{
		{0xf000, 4, 2}, {0xe800, 5, 5}, {0xe400, 6, -7}, {0xe000, 6, 7}, {0xd000, 4, -3}, {0xc000, 4, 3}, {0xb800, 5, -6}, {0xb000, 5, 6}, {0xa000, 4, -4}, {0x9000, 4, 4}, {0x8000, 4, -5}, {0x6000, 3, 0}, {0x4000, 3, -1}, {0x2000, 3, 1}, {0x0, 3, -2}
	}, {
		{0xf000, 4, 3}, {0xe800, 5, 4}, {0xe600, 7, 6}, {0xe500, 8, -7}, {0xe400, 8, 7}, {0xe000, 6, -6}, {0xc000, 3, 0}, {0xa000, 3, -1}, {0x8000, 3, 1}, {0x6000, 3, -2}, {0x4000, 3, 2}, {0x3800, 5, -5}, {0x3000, 5, 5}, {0x2000, 4, -4}, {0x0, 3, -3}
	}
};

static const mpc_huffman mpc_table_HuffQ6 [2] [31] = {
	{
		{0xf800, 5, 3}, {0xf000, 5, -4}, {0xec00, 6, -11}, {0xe800, 6, 12}, {0xe000, 5, 4}, {0xd800, 5, 6}, {0xd000, 5, -5}, {0xc800, 5, 5}, {0xc000, 5, 7}, {0xb800, 5, -7}, {0xb400, 6, -12}, {0xb000, 6, -13}, {0xa800, 5, -6}, {0xa000, 5, 8}, {0x9800, 5, -8}, {0x9000, 5, 9}, {0x8800, 5, -9}, {0x8400, 6, 13}, {0x8200, 7, -15}, {0x8000, 7, 15}, {0x7000, 4, 0}, {0x6800, 5, -10}, {0x6000, 5, 10}, {0x5000, 4, -1}, {0x4000, 4, 2}, {0x3000, 4, 1}, {0x2000, 4, -2}, {0x1c00, 6, 14}, {0x1800, 6, -14}, {0x1000, 5, 11}, {0x0, 4, -3}
	}, {
		{0xf800, 5, -6}, {0xf000, 5, 6}, {0xe000, 4, 1}, {0xd000, 4, -1}, {0xce00, 7, 10}, {0xcc00, 7, -10}, {0xcb00, 8, -11}, {0xca80, 9, -12}, {0xca60, 11, 13}, {0xca58, 13, 15}, {0xca50, 13, -14}, {0xca48, 13, 14}, {0xca40, 13, -15}, {0xca00, 10, -13}, {0xc900, 8, 11}, {0xc800, 8, 12}, {0xc400, 6, -9}, {0xc000, 6, 9}, {0xb000, 4, -2}, {0xa000, 4, 2}, {0x9000, 4, 3}, {0x8000, 4, -3}, {0x7800, 5, -7}, {0x7000, 5, 7}, {0x6000, 4, -4}, {0x5000, 4, 4}, {0x4800, 5, -8}, {0x4000, 5, 8}, {0x3000, 4, 5}, {0x2000, 4, -5}, {0x0, 3, 0}
	}
};

static const mpc_huffman mpc_table_HuffQ7 [2] [63] = {
	{
		{0xfc00, 6, 7}, {0xf800, 6, 8}, {0xf400, 6, 9}, {0xf000, 6, -8}, {0xec00, 6, 11}, {0xea00, 7, 21}, {0xe900, 8, -28}, {0xe800, 8, 28}, {0xe400, 6, -9}, {0xe200, 7, -22}, {0xe000, 7, -21}, {0xdc00, 6, -10}, {0xd800, 6, -11}, {0xd400, 6, 10}, {0xd000, 6, 12}, {0xcc00, 6, -13}, {0xca00, 7, 22}, {0xc800, 7, 23}, {0xc400, 6, -12}, {0xc000, 6, 13}, {0xbc00, 6, 14}, {0xb800, 6, -14}, {0xb600, 7, -23}, {0xb500, 8, -29}, {0xb400, 8, 29}, {0xb000, 6, -15}, {0xac00, 6, 15}, {0xa800, 6, 16}, {0xa400, 6, -16}, {0xa200, 7, -24}, {0xa000, 7, 24}, {0x9c00, 6, 17}, {0x9a00, 7, -25}, {0x9900, 8, -30}, {0x9800, 8, 30}, {0x9400, 6, -17}, {0x9000, 6, 18}, {0x8c00, 6, -18}, {0x8a00, 7, 25}, {0x8800, 7, 26}, {0x8400, 6, 19}, {0x8200, 7, -26}, {0x8000, 7, -27}, {0x7800, 5, 2}, {0x7400, 6, -19}, {0x7000, 6, 20}, {0x6800, 5, -1}, {0x6700, 8, -31}, {0x6600, 8, 31}, {0x6400, 7, 27}, {0x6000, 6, -20}, {0x5800, 5, 1}, {0x5000, 5, -5}, {0x4800, 5, -3}, {0x4000, 5, 3}, {0x3800, 5, 0}, {0x3000, 5, -2}, {0x2800, 5, -4}, {0x2000, 5, 4}, {0x1800, 5, 5}, {0x1000, 5, -6}, {0x800, 5, 6}, {0x0, 5, -7}
	}, {
		{0xf800, 5, -1}, {0xf000, 5, 2}, {0xe800, 5, -2}, {0xe000, 5, 3}, {0xdf00, 8, -20}, {0xdec0, 10, 24}, {0xdebc, 14, 28}, {0xdeb8, 14, -28}, {0xdeb4, 14, -30}, {0xdeb0, 14, 30}, {0xdea0, 12, -27}, {0xde9c, 14, 29}, {0xde98, 14, -29}, {0xde94, 14, 31}, {0xde90, 14, -31}, {0xde80, 12, 27}, {0xde00, 9, -22}, {0xdc00, 7, -17}, {0xd800, 6, -11}, {0xd000, 5, -3}, {0xc800, 5, 4}, {0xc000, 5, -4}, {0xbe00, 7, 17}, {0xbd00, 8, 20}, {0xbc80, 9, 22}, {0xbc40, 10, -25}, {0xbc00, 10, -26}, {0xb800, 6, 12}, {0xb000, 5, 5}, {0xa800, 5, -5}, {0xa000, 5, 6}, {0x9800, 5, -6}, {0x9400, 6, -12}, {0x9200, 7, -18}, {0x9000, 7, 18}, {0x8c00, 6, 13}, {0x8800, 6, -13}, {0x8000, 5, -7}, {0x7c00, 6, 14}, {0x7b00, 8, 21}, {0x7a00, 8, -21}, {0x7800, 7, -19}, {0x7000, 5, 7}, {0x6800, 5, 8}, {0x6400, 6, -14}, {0x6000, 6, -15}, {0x5800, 5, -8}, {0x5400, 6, 15}, {0x5200, 7, 19}, {0x51c0, 10, 25}, {0x5180, 10, 26}, {0x5100, 9, -23}, {0x5080, 9, 23}, {0x5000, 9, -24}, {0x4800, 5, -9}, {0x4000, 5, 9}, {0x3c00, 6, 16}, {0x3800, 6, -16}, {0x3000, 5, 10}, {0x2000, 4, 0}, {0x1800, 5, -10}, {0x1000, 5, 11}, {0x0, 4, 1}
	}
};

mpc_lut_data mpc_HuffQ [7] [2] = {
	{{mpc_table_HuffQ1[0]}, {mpc_table_HuffQ1[1]}},
	{{mpc_table_HuffQ2[0]}, {mpc_table_HuffQ2[1]}},
	{{mpc_table_HuffQ3[0]}, {mpc_table_HuffQ3[1]}},
	{{mpc_table_HuffQ4[0]}, {mpc_table_HuffQ4[1]}},
	{{mpc_table_HuffQ5[0]}, {mpc_table_HuffQ5[1]}},
	{{mpc_table_HuffQ6[0]}, {mpc_table_HuffQ6[1]}},
	{{mpc_table_HuffQ7[0]},	{mpc_table_HuffQ7[1]}}
};


// sv8 huffman tables
static const mpc_huffman mpc_huff_SCFI_1 [3] = {
	{0x8000, 1, 1}, {0x4000, 2, 2}, {0x0, 3, 3}
}; // 3
static const mpc_int8_t mpc_sym_SCFI_1 [4] = {
	2, 3, 1, 0
};
static const mpc_huffman mpc_huff_SCFI_2 [5] = {
	{0x8000, 2, 3}, {0x4000, 3, 5}, {0x1800, 5, 11}, {0x400, 6, 14}, {0x0, 7, 15}
}; // 5
static const mpc_int8_t mpc_sym_SCFI_2 [16] = {
	15, 10, 14, 11, 13, 9, 7, 6, 5, 12, 8, 3, 2, 0, 4, 1
};
mpc_can_data mpc_can_SCFI[2] = {{mpc_huff_SCFI_1, mpc_sym_SCFI_1}, {mpc_huff_SCFI_2, mpc_sym_SCFI_2}};

static const mpc_huffman mpc_huff_DSCF_1 [12] = {
	{0xa000, 3, 7}, {0x4000, 4, 12}, {0x2800, 5, 16}, {0x1800, 6, 21}, {0xe00, 7, 27}, {0x700, 8, 34}, {0x380, 9, 41}, {0x140, 10, 48}, {0x80, 11, 53}, {0x30, 12, 57}, {0x18, 13, 60}, {0x0, 14, 63}
}; // 12
static const mpc_int8_t mpc_sym_DSCF_1 [64] = {
	35, 34, 33, 36, 32, 30, 29, 27, 26, 37, 28, 25, 39, 38, 24, 23, 40, 22, 21, 20, 19, 43, 42, 41, 18, 17, 16, 15, 46, 45, 44, 14, 13, 12, 11, 49, 48, 47, 31, 10, 9, 8, 7, 6, 52, 51, 50, 5, 4, 3, 54, 53, 2, 1, 0, 57, 56, 55, 63, 62, 61, 60, 59, 58
};
static const mpc_huffman mpc_huff_DSCF_2 [13] = {
	{0x6000, 3, 7}, {0x3000, 4, 10}, {0x1800, 5, 13}, {0x1000, 6, 16}, {0xa00, 7, 20}, {0x600, 8, 25}, {0x380, 9, 31}, {0x1c0, 10, 38}, {0xe0, 11, 45}, {0x50, 12, 52}, {0x20, 13, 57}, {0xc, 14, 61}, {0x0, 15, 64}
}; // 13
static const mpc_int8_t mpc_sym_DSCF_2 [65] = {
	33, 32, 31, 30, 29, 34, 28, 27, 36, 35, 26, 37, 25, 38, 24, 23, 40, 39, 22, 21, 42, 41, 20, 19, 18, 45, 44, 43, 17, 16, 15, 14, 48, 47, 46, 13, 12, 11, 10, 64, 52, 51, 50, 49, 9, 8, 7, 6, 55, 54, 53, 5, 4, 3, 58, 57, 56, 2, 1, 63, 62, 61, 60, 59, 0
};
mpc_can_data mpc_can_DSCF[2] = {{mpc_huff_DSCF_1, mpc_sym_DSCF_1}, {mpc_huff_DSCF_2, mpc_sym_DSCF_2}};

static const mpc_huffman mpc_huff_Bands [12] = {
	{0x8000, 1, 1}, {0x4000, 2, 2}, {0x2000, 3, 3}, {0x1000, 5, 6}, {0x800, 6, 8}, {0x600, 7, 10}, {0x300, 8, 13}, {0x200, 9, 16}, {0x140, 10, 20}, {0xc0, 11, 25}, {0x10, 12, 31}, {0x0, 13, 32}
}; // 12
static const mpc_int8_t mpc_sym_Bands [33] = {
	0, 32, 1, 31, 2, 30, 3, 4, 29, 6, 5, 28, 7, 27, 26, 8, 25, 24, 23, 9, 22, 21, 20, 18, 17, 16, 15, 14, 12, 11, 10, 19, 13
};
mpc_can_data mpc_can_Bands = {mpc_huff_Bands, mpc_sym_Bands};

static const mpc_huffman mpc_huff_Res_1 [16] = {
	{0x8000, 1, 1}, {0x4000, 2, 2}, {0x2000, 3, 3}, {0x1000, 4, 4}, {0x800, 5, 5}, {0x400, 6, 6}, {0x200, 7, 7}, {0x100, 8, 8}, {0x80, 9, 9}, {0x40, 10, 10}, {0x20, 11, 11}, {0x10, 12, 12}, {0x8, 13, 13}, {0x4, 14, 14}, {0x2, 15, 15}, {0x0, 16, 16}
}; // 16
static const mpc_int8_t mpc_sym_Res_1 [17] = {
	0, 1, 16, 2, 3, 4, 5, 15, 6, 7, 8, 9, 10, 11, 12, 14, 13
};
static const mpc_huffman mpc_huff_Res_2 [12] = {
	{0x4000, 2, 3}, {0x2000, 3, 4}, {0x1000, 4, 5}, {0x800, 5, 6}, {0x400, 6, 7}, {0x200, 7, 8}, {0x100, 8, 9}, {0x80, 9, 10}, {0x40, 10, 11}, {0x20, 11, 12}, {0x10, 12, 13}, {0x0, 14, 16}
}; // 12
static const mpc_int8_t mpc_sym_Res_2 [17] = {
	16, 1, 0, 2, 15, 3, 14, 4, 5, 13, 6, 12, 7, 11, 10, 9, 8
};
mpc_can_data mpc_can_Res[2] = {{mpc_huff_Res_1, mpc_sym_Res_1}, {mpc_huff_Res_2, mpc_sym_Res_2}};

static const mpc_huffman mpc_huff_Q1 [10] = {
	{0x6000, 3, 7}, {0x1000, 4, 10}, {0x800, 5, 11}, {0x400, 6, 12}, {0x200, 7, 13}, {0x100, 8, 14}, {0x80, 9, 15}, {0x40, 10, 16}, {0x20, 11, 17}, {0x0, 12, 18}
}; // 10
static const mpc_int8_t mpc_sym_Q1 [19] = {
	7, 6, 5, 4, 3, 10, 9, 8, 2, 1, 11, 0, 12, 13, 14, 15, 16, 18, 17
};
mpc_can_data mpc_can_Q1 = {mpc_huff_Q1, mpc_sym_Q1};

static const mpc_huffman mpc_huff_Q2_1 [10] = {
	{0xe000, 3, 7}, {0x8000, 4, 14}, {0x3c00, 6, 38}, {0x2a00, 7, 53}, {0x1200, 8, 74}, {0x600, 9, 92}, {0x3c0, 10, 104}, {0x60, 11, 119}, {0x20, 12, 122}, {0x0, 13, 124}
}; // 10
static const mpc_int8_t mpc_sym_Q2_1 [125] = {
	62, 87, 67, 63, 61, 57, 37, 93, 92, 88, 86, 83, 82, 81, 68, 66, 58, 56, 42, 41, 38, 36, 32, 31, 112, 91, 72, 64, 60, 52, 43, 33, 12, 117, 113, 111, 107, 97, 89, 85, 77, 73, 71, 69, 65, 59, 55, 53, 51, 47, 39, 35, 27, 17, 13, 11, 7, 118, 116, 108, 106, 98, 96, 94, 90, 84, 80, 78, 76, 48, 46, 44, 40, 34, 30, 28, 26, 18, 16, 8, 6, 122, 110, 102, 74, 70, 54, 50, 22, 2, 123, 121, 119, 115, 114, 109, 105, 103, 101, 99, 95, 79, 75, 49, 45, 29, 25, 23, 21, 19, 15, 14, 10, 9, 5, 3, 1, 124, 104, 20, 0, 120, 100, 24, 4
};
static const mpc_huffman mpc_huff_Q2_2 [9] = {
	{0xf000, 4, 15}, {0x7000, 5, 30}, {0x4800, 6, 44}, {0x3c00, 7, 62}, {0xc00, 8, 92}, {0x780, 9, 104}, {0xc0, 10, 119}, {0x40, 11, 122}, {0x0, 12, 124}
}; // 9
static const mpc_int8_t mpc_sym_Q2_2 [125] = {
	62, 92, 87, 86, 82, 68, 67, 66, 63, 61, 58, 57, 56, 42, 38, 37, 32, 93, 91, 88, 83, 81, 43, 41, 36, 33, 31, 112, 72, 64, 60, 52, 12, 118, 117, 116, 113, 111, 108, 107, 106, 98, 97, 96, 94, 90, 89, 85, 84, 80, 78, 77, 76, 73, 71, 69, 65, 59, 55, 53, 51, 48, 47, 46, 44, 40, 39, 35, 34, 30, 28, 27, 26, 18, 17, 16, 13, 11, 8, 7, 6, 122, 110, 74, 70, 54, 50, 22, 14, 2, 123, 121, 119, 115, 114, 109, 105, 103, 102, 101, 99, 95, 79, 75, 49, 45, 29, 25, 23, 21, 19, 15, 10, 9, 5, 3, 1, 124, 104, 20, 0, 120, 100, 24, 4
};

static const mpc_huffman mpc_huff_Q3 [7] = {
	{0xe000, 3, 7}, {0x8000, 4, 14}, {0x5000, 5, 22}, {0x2400, 6, 32}, {0xa00, 7, 41}, {0x200, 8, 46}, {0x0, 9, 48}
}; // 7
static const mpc_int8_t mpc_sym_Q3 [49] = {
	0, 17, 16, 1, 15, -16, -1, 32, 31, 2, 14, -15, -32, 34, 33, 47, 46, 18, 30, -14, -2, -31, -17, -18, 49, 48, 63, 19, 29, 3, 13, -13, -3, -30, -47, -48, -33, 50, 62, 35, 45, -29, -19, -46, -34, 51, 61, -45, -35
};

static const mpc_huffman mpc_huff_Q4 [8] = {
	{0xf000, 4, 15}, {0x9000, 5, 30}, {0x3400, 6, 48}, {0x1800, 7, 61}, {0x500, 8, 73}, {0x100, 9, 78}, {0x0, 10, 80}, {0x0, 0, 90}
}; // 8
static const mpc_int8_t mpc_sym_Q4 [91] = {
	0, 32, 17, 16, 31, 2, 1, 15, 14, -15, -16, -1, -32, 49, 48, 34, 33, 47, 46, 19, 18, 30, 29, 3, 13, -13, -14, -2, -3, -30, -31, -17, -18, -47, -48, -33, 64, 50, 63, 62, 35, 45, 4, 12, -29, -19, -46, -34, -64, -49, 66, 65, 79, 78, 51, 61, 36, 44, 20, 28, -12, -4, -28, -20, -45, -35, -62, -63, -50, 67, 77, 52, 60, -44, -36, -61, -51, 68, 76, -60, -52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static const mpc_huffman mpc_huff_Q5_1 [6] = {
	{0xc000, 2, 3}, {0x4000, 3, 6}, {0x2000, 4, 8}, {0x1000, 5, 10}, {0x800, 6, 12}, {0x0, 7, 14}
}; // 6
static const mpc_int8_t mpc_sym_Q5_1 [15] = {
	0, 2, 1, -1, -2, 3, -3, 4, -4, 5, -5, 7, 6, -6, -7
};
static const mpc_huffman mpc_huff_Q5_2 [4] = {
	{0x6000, 3, 7}, {0x2000, 4, 10}, {0x1000, 5, 12}, {0x0, 6, 14}
}; // 4
static const mpc_int8_t mpc_sym_Q5_2 [15] = {
	2, 1, 0, -1, -2, 4, 3, -3, -4, 5, -5, 7, 6, -6, -7
};

static const mpc_huffman mpc_huff_Q6_1 [8] = {
	{0xc000, 2, 3}, {0x8000, 3, 6}, {0x4000, 4, 10}, {0x2800, 5, 14}, {0xc00, 6, 19}, {0x800, 7, 22}, {0x400, 8, 26}, {0x0, 9, 30}
}; // 8
static const mpc_int8_t mpc_sym_Q6_1 [31] = {
	0, 1, -1, 3, 2, -2, -3, 4, -4, -5, 8, 7, 6, 5, -6, -7, -8, 9, -9, 11, 10, -10, -11, 15, 14, 13, 12, -12, -13, -14, -15
};
static const mpc_huffman mpc_huff_Q6_2 [5] = {
	{0x5000, 4, 15}, {0x2000, 5, 20}, {0x1000, 6, 24}, {0x400, 7, 28}, {0x0, 8, 30}
}; // 5
static const mpc_int8_t mpc_sym_Q6_2 [31] = {
	5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, 8, 7, 6, -6, -7, -8, 10, 9, -9, -10, 13, 12, 11, -11, -12, -13, 15, 14, -14, -15
};

static const mpc_huffman mpc_huff_Q7_1 [9] = {
	{0xc000, 2, 3}, {0x8000, 3, 6}, {0x6000, 4, 10}, {0x4000, 5, 16}, {0x2800, 6, 24}, {0x1400, 7, 34}, {0xa00, 8, 44}, {0x400, 9, 54}, {0x0, 10, 62}
}; // 9
static const mpc_int8_t mpc_sym_Q7_1 [63] = {
	0, 1, -1, 2, -2, 4, 3, -3, -4, 7, 6, 5, -5, -6, -7, 13, 11, 10, 9, 8, -8, -9, -10, -11, -12, 17, 16, 15, 14, 12, -13, -14, -15, -16, -17, 28, 27, 21, 20, 19, 18, -18, -19, -20, -21, -27, -28, 31, 30, 29, 26, 25, 24, 23, 22, -22, -23, -24, -25, -26, -29, -30, -31
};
static const mpc_huffman mpc_huff_Q7_2 [5] = {
	{0x6000, 5, 31}, {0x2400, 6, 43}, {0x1000, 7, 52}, {0x200, 8, 60}, {0x0, 9, 62}
}; // 5
static const mpc_int8_t mpc_sym_Q7_2 [63] = {
	10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, 17, 16, 15, 14, 13, 12, 11, -10, -11, -12, -13, -14, -15, -16, -17, 22, 21, 20, 19, 18, -18, -19, -20, -21, -22, 29, 28, 27, 26, 25, 24, 23, -23, -24, -25, -26, -27, -28, -29, 31, 30, -30, -31
};

static const mpc_huffman mpc_huff_Q8_1 [11] = {
	{0xc000, 2, 3}, {0x8000, 3, 6}, {0x7000, 4, 10}, {0x5800, 5, 17}, {0x3800, 6, 28}, {0x2800, 7, 42}, {0x1900, 8, 62}, {0xd00, 9, 87}, {0x280, 10, 113}, {0x60, 11, 123}, {0x0, 12, 126}
}; // 11
static const mpc_int8_t mpc_sym_Q8_1 [127] = {
	0, 1, -1, -2, 3, 2, -3, 7, 6, 5, 4, -4, -5, -6, -7, 11, 10, 9, 8, -8, -9, -10, -11, 19, 18, 17, 16, 15, 14, 13, 12, -12, -13, -14, -15, -16, -17, -19, 56, 55, 31, 28, 27, 26, 25, 24, 23, 22, 21, 20, -18, -20, -21, -22, -23, -24, -25, -26, -27, -33, -54, -56, 63, 62, 61, 60, 59, 58, 57, 54, 53, 43, 40, 39, 38, 37, 36, 35, 34, 33, 32, 30, 29, -28, -29, -30, -31, -32, -34, -35, -36, -37, -38, -39, -40, -41, -43, -53, -55, -57, -58, -59, -60, -61, 49, 47, 46, 45, 44, 42, 41, -42, -44, -45, -46, -47, -48, -49, -50, -62, -63, 52, 51, 50, 48, -51, -52
};
static const mpc_huffman mpc_huff_Q8_2 [4] = {
	{0x9800, 6, 63}, {0x2a00, 7, 101}, {0x400, 8, 122}, {0x0, 9, 126}
}; // 4
static const mpc_int8_t mpc_sym_Q8_2 [127] = {
	13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 12, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, 63, 62, 61, 60, -60, -61, -62, -63
};

static const mpc_huffman mpc_huff_Q9up [6] = {
	{0xf800, 6, 63}, {0xac00, 7, 125}, {0x2600, 8, -45}, {0x280, 9, -7}, {0x40, 10, -2}, {0x0, 11, -1}
}; // 6
static const mpc_int8_t mpc_sym_Q9up [256] = {
	-128, 127, -108, -110, -111, -112, -113, -114, -115, -116, -117, -118, -119, -120, -121, -122, -123, -124, -125, -126, -127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68, -69, -70, -71, -72, -73, -74, -75, -76, -77, -78, -79, -80, -81, -82, -83, -84, -85, -86, -87, -88, -89, -90, -91, -92, -93, -94, -95, -96, -97, -98, -99, -100, -101, -102, -103, -104, -105, -106, -107, -109, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 40, 20, 19, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, 41, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, -3, -4, -5, -6, 4, 3, 2, 1, 0, -1, -2
};
mpc_can_data mpc_can_Q9up = {mpc_huff_Q9up, mpc_sym_Q9up};


mpc_can_data mpc_can_Q [6][2] = {
	{{mpc_huff_Q2_1, mpc_sym_Q2_1}, {mpc_huff_Q2_2, mpc_sym_Q2_2}},
	{{mpc_huff_Q3, mpc_sym_Q3},     {mpc_huff_Q4, mpc_sym_Q4}},
	{{mpc_huff_Q5_1, mpc_sym_Q5_1}, {mpc_huff_Q5_2, mpc_sym_Q5_2}},
	{{mpc_huff_Q6_1, mpc_sym_Q6_1}, {mpc_huff_Q6_2, mpc_sym_Q6_2}},
	{{mpc_huff_Q7_1, mpc_sym_Q7_1}, {mpc_huff_Q7_2, mpc_sym_Q7_2}},
	{{mpc_huff_Q8_1, mpc_sym_Q8_1}, {mpc_huff_Q8_2, mpc_sym_Q8_2}}
};

static void huff_fill_lut(const mpc_huffman * table, mpc_huff_lut * lut, const int bits)
{
	int i, idx = 0;
	const int shift = 16 - bits;
	for (i = (1 << bits) - 1; i >= 0 ; i--) {
		if ((table[idx].Code >> shift) < i) {
			lut[i].Length = table[idx].Length;
			lut[i].Value = table[idx].Value;
		} else {
			if (table[idx].Length <= bits) {
				lut[i].Length = table[idx].Length;
				lut[i].Value = table[idx].Value;
			} else {
				lut[i].Length = 0;
				lut[i].Value = idx;
			}
			if (i != 0)
				do {
					idx++;
				} while ((table[idx].Code >> shift) == i);
		}
	}
}

static void can_fill_lut(mpc_can_data * data, const int bits)
{
	int i, idx = 0;
	const int shift = 16 - bits;
	const mpc_huffman * table = data->table;
	const mpc_int8_t * sym = data->sym;
	mpc_huff_lut * lut = data->lut;
	for (i = (1 << bits) - 1; i >= 0 ; i--) {
		if ((table[idx].Code >> shift) < i) {
			if (table[idx].Length <= bits) {
				lut[i].Length = table[idx].Length;
				lut[i].Value = sym[(table[idx].Value - (i >> (bits - table[idx].Length))) & 0xFF];
			} else {
				lut[i].Length = 0;
				lut[i].Value = idx;
			}
		} else {
			if (table[idx].Length <= bits) {
				lut[i].Length = table[idx].Length;
				lut[i].Value = sym[(table[idx].Value - (i >> (bits - table[idx].Length))) & 0xFF];
			} else {
				lut[i].Length = 0;
				lut[i].Value = idx;
			}
			if (i != 0)
				do {
					idx++;
				} while ((table[idx].Code >> shift) == i);
		}
	}
}

void huff_init_lut(const int bits)
{
	int i, j;

	huff_fill_lut(mpc_HuffDSCF.table, mpc_HuffDSCF.lut, bits);
	huff_fill_lut(mpc_HuffHdr.table, mpc_HuffHdr.lut, bits);

	can_fill_lut(&mpc_can_SCFI[0], bits);
	can_fill_lut(&mpc_can_SCFI[1], bits);
	can_fill_lut(&mpc_can_DSCF[0], bits);
	can_fill_lut(&mpc_can_DSCF[1], bits);
	can_fill_lut(&mpc_can_Res[0], bits);
	can_fill_lut(&mpc_can_Res[1], bits);
	can_fill_lut(&mpc_can_Q1, bits);
	can_fill_lut(&mpc_can_Q9up, bits);


	for( i = 0; i < 7; i++){
		for( j = 0; j < 2; j++){
			if (i != 6) can_fill_lut(&mpc_can_Q[i][j], bits);
			huff_fill_lut(mpc_HuffQ[i][j].table, mpc_HuffQ[i][j].lut, bits);
		}
	}
}


