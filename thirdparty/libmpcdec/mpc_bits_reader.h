/*
  Copyright (c) 2007-2009, The Musepack Development Team
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

#define MAX_ENUM 32

MPC_API int mpc_bits_get_block(mpc_bits_reader * r, mpc_block * p_block);
mpc_int32_t mpc_bits_golomb_dec(mpc_bits_reader * r, const mpc_uint_t k);
MPC_API unsigned int mpc_bits_get_size(mpc_bits_reader * r, mpc_uint64_t * p_size);
mpc_uint32_t mpc_bits_log_dec(mpc_bits_reader * r, mpc_uint_t max);

extern const mpc_uint32_t Cnk[MAX_ENUM / 2][MAX_ENUM];
extern const mpc_uint8_t Cnk_len[MAX_ENUM / 2][MAX_ENUM];
extern const mpc_uint32_t Cnk_lost[MAX_ENUM / 2][MAX_ENUM];

// can read up to 31 bits
static mpc_inline mpc_uint32_t mpc_bits_read(mpc_bits_reader * r, const unsigned int nb_bits)
{
	mpc_uint32_t ret;

	r->buff -= (int)(r->count - nb_bits) >> 3;
	r->count = (r->count - nb_bits) & 0x07;

	ret = (r->buff[0] | (r->buff[-1] << 8)) >> r->count;
	if (nb_bits > (16 - r->count)) {
		ret |= (mpc_uint32_t)((r->buff[-2] << 16) | (r->buff[-3] << 24)) >> r->count;
		if (nb_bits > 24 && r->count != 0)
			ret |= r->buff[-4] << (32 - r->count);
	}

	return ret & ((1 << nb_bits) - 1);
}

// basic huffman decoding routine
// works with maximum lengths up to 16
static mpc_inline mpc_int32_t mpc_bits_huff_dec(mpc_bits_reader * r, const mpc_huffman *Table)
{
	mpc_uint16_t code;
	code = (mpc_uint16_t)((((r->buff[0] << 16) | (r->buff[1] << 8) | r->buff[2]) >> r->count) & 0xFFFF);

	while (code < Table->Code) Table++;

	r->buff -= (int)(r->count - Table->Length) >> 3;
	r->count = (r->count - Table->Length) & 0x07;

	return Table->Value;
}

static mpc_inline mpc_int32_t mpc_bits_can_dec(mpc_bits_reader * r, const mpc_can_data *can)
{
	mpc_uint16_t code;
	mpc_huff_lut tmp;
	const mpc_huffman * Table;
	code = (mpc_uint16_t)((((r->buff[0] << 16) | (r->buff[1] << 8) | r->buff[2]) >> r->count) & 0xFFFF);

	tmp = can->lut[code >> (16 - LUT_DEPTH)];
	if (tmp.Length != 0) {
		r->buff -= (int)(r->count - tmp.Length) >> 3;
		r->count = (r->count - tmp.Length) & 0x07;
		return tmp.Value;
	}

	Table = can->table + (unsigned char)tmp.Value;
	while (code < Table->Code) Table++;

	r->buff -= (int)(r->count - Table->Length) >> 3;
	r->count = (r->count - Table->Length) & 0x07;

	return can->sym[(Table->Value - (code >> (16 - Table->Length))) & 0xFF] ;
}

// LUT-based huffman decoding routine
// works with maximum lengths up to 16
static mpc_inline mpc_int32_t mpc_bits_huff_lut(mpc_bits_reader * r, const mpc_lut_data *lut)
{
	mpc_uint16_t code;
	mpc_huff_lut tmp;
	const mpc_huffman * Table;
	code = (mpc_uint16_t)((((r->buff[0] << 16) | (r->buff[1] << 8) | r->buff[2]) >> r->count) & 0xFFFF);

	tmp = lut->lut[code >> (16 - LUT_DEPTH)];
	if (tmp.Length != 0) {
		r->buff -= (int)(r->count - tmp.Length) >> 3;
		r->count = (r->count - tmp.Length) & 0x07;
		return tmp.Value;
	}

	Table = lut->table + (unsigned char)tmp.Value;
	while (code < Table->Code) Table++;

	r->buff -= (int)(r->count - Table->Length) >> 3;
	r->count = (r->count - Table->Length) & 0x07;

	return Table->Value;
}

static mpc_inline mpc_uint32_t mpc_bits_enum_dec(mpc_bits_reader * r, mpc_uint_t k, mpc_uint_t n)
{
	mpc_uint32_t bits = 0;
	mpc_uint32_t code;
	const mpc_uint32_t * C = Cnk[k-1];

	code = mpc_bits_read(r, Cnk_len[k-1][n-1] - 1);

	if (code >= Cnk_lost[k-1][n-1])
		code = ((code << 1) | mpc_bits_read(r, 1)) - Cnk_lost[k-1][n-1];

	do {
		n--;
		if (code >= C[n]) {
			bits |= 1 << n;
			code -= C[n];
			C -= MAX_ENUM;
			k--;
		}
	} while(k > 0);

	return bits;
}
