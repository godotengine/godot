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
/// \file mpc_decoder.c
/// Core decoding routines and logic.

#include <string.h>
#include <mpc/mpcdec.h>
#include <mpc/minimax.h>
#include "decoder.h"
#include "huffman.h"
#include "internal.h"
#include "mpcdec_math.h"
#include "requant.h"
#include "mpc_bits_reader.h"

//SV7 tables
extern const mpc_lut_data   mpc_HuffQ [7] [2];
extern const mpc_lut_data   mpc_HuffHdr;
extern const mpc_huffman    mpc_table_HuffSCFI [ 4];
extern const mpc_lut_data   mpc_HuffDSCF;

//SV8 tables
extern const mpc_can_data mpc_can_Bands;
extern const mpc_can_data mpc_can_SCFI[2];
extern const mpc_can_data mpc_can_DSCF[2];
extern const mpc_can_data mpc_can_Res [2];
extern const mpc_can_data mpc_can_Q [8][2];
extern const mpc_can_data mpc_can_Q1;
extern const mpc_can_data mpc_can_Q9up;

//------------------------------------------------------------------------------
// types
//------------------------------------------------------------------------------
enum
{
    MEMSIZE   = MPC_DECODER_MEMSIZE, // overall buffer size
    MEMSIZE2  = (MEMSIZE/2),         // size of one buffer
    MEMMASK   = (MEMSIZE-1)
};

//------------------------------------------------------------------------------
// forward declarations
//------------------------------------------------------------------------------
void mpc_decoder_read_bitstream_sv7(mpc_decoder * d, mpc_bits_reader * r);
void mpc_decoder_read_bitstream_sv8(mpc_decoder * d, mpc_bits_reader * r,
									mpc_bool_t is_key_frame);
static void mpc_decoder_requantisierung(mpc_decoder *d);

/**
 * set the scf indexes for seeking use
 * needed only for sv7 seeking
 * @param d
 */
void mpc_decoder_reset_scf(mpc_decoder * d, int value)
{
	memset(d->SCF_Index_L, value, sizeof(d->SCF_Index_L) );
	memset(d->SCF_Index_R, value, sizeof(d->SCF_Index_R) );
}


void mpc_decoder_setup(mpc_decoder *d)
{
	memset(d, 0, sizeof *d);

	d->__r1 = 1;
	d->__r2 = 1;

	mpc_decoder_init_quant(d, 1.0f);
}

void mpc_decoder_set_streaminfo(mpc_decoder *d, mpc_streaminfo *si)
{
	d->stream_version     = si->stream_version;
	d->ms                 = si->ms;
	d->max_band           = si->max_band;
	d->channels           = si->channels;
	d->samples_to_skip    = MPC_DECODER_SYNTH_DELAY + si->beg_silence;

	if (si->stream_version == 7 && si->is_true_gapless)
		d->samples = ((si->samples + MPC_FRAME_LENGTH - 1) / MPC_FRAME_LENGTH) * MPC_FRAME_LENGTH;
	else
		d->samples = si->samples;
}

mpc_decoder * mpc_decoder_init(mpc_streaminfo *si)
{
	mpc_decoder* p_tmp = malloc(sizeof(mpc_decoder));

	if (p_tmp != 0) {
		mpc_decoder_setup(p_tmp);
		mpc_decoder_set_streaminfo(p_tmp, si);
		huff_init_lut(LUT_DEPTH); // FIXME : this needs to be called only once when the library is loaded
	}

	return p_tmp;
}

void mpc_decoder_exit(mpc_decoder *d)
{
	free(d);
}

void mpc_decoder_decode_frame(mpc_decoder * d,
							  mpc_bits_reader * r,
							  mpc_frame_info * i)
{
	mpc_bits_reader r_sav = *r;
	mpc_int64_t samples_left;

	samples_left = d->samples - d->decoded_samples + MPC_DECODER_SYNTH_DELAY;

	if (samples_left <= 0 && d->samples != 0) {
		i->samples = 0;
		i->bits = -1;
		return;
	}

	if (d->stream_version == 8)
		mpc_decoder_read_bitstream_sv8(d, r, i->is_key_frame);
	else
		mpc_decoder_read_bitstream_sv7(d, r);

	if (d->samples_to_skip < MPC_FRAME_LENGTH + MPC_DECODER_SYNTH_DELAY) {
		mpc_decoder_requantisierung(d);
		mpc_decoder_synthese_filter_float(d, i->buffer, d->channels);
	}

	d->decoded_samples += MPC_FRAME_LENGTH;

    // reconstruct exact filelength
	if (d->decoded_samples - d->samples < MPC_FRAME_LENGTH && d->stream_version == 7) {
		int last_frame_samples = mpc_bits_read(r, 11);
		if (d->decoded_samples == d->samples) {
			if (last_frame_samples == 0) last_frame_samples = MPC_FRAME_LENGTH;
			d->samples += last_frame_samples - MPC_FRAME_LENGTH;
			samples_left += last_frame_samples - MPC_FRAME_LENGTH;
		}
	}

	i->samples = samples_left > MPC_FRAME_LENGTH ? MPC_FRAME_LENGTH : samples_left < 0 ? 0 : (mpc_uint32_t) samples_left;
	i->bits = (mpc_uint32_t) (((r->buff - r_sav.buff) << 3) + r_sav.count - r->count);

	if (d->samples_to_skip) {
		if (i->samples <= d->samples_to_skip) {
			d->samples_to_skip -= i->samples;
			i->samples = 0;
		} else {
			i->samples -= d->samples_to_skip;
			memmove(i->buffer, i->buffer + d->samples_to_skip * d->channels,
					i->samples * d->channels * sizeof (MPC_SAMPLE_FORMAT));
			d->samples_to_skip = 0;
		}
	}
}

void
mpc_decoder_requantisierung(mpc_decoder *d)
{
    mpc_int32_t     Band;
    mpc_int32_t     n;
    MPC_SAMPLE_FORMAT facL;
    MPC_SAMPLE_FORMAT facR;
    MPC_SAMPLE_FORMAT templ;
    MPC_SAMPLE_FORMAT tempr;
    MPC_SAMPLE_FORMAT* YL;
    MPC_SAMPLE_FORMAT* YR;
    mpc_int16_t*    L;
    mpc_int16_t*    R;
	const mpc_int32_t Last_Band = d->max_band;

#ifdef MPC_FIXED_POINT
#if MPC_FIXED_POINT_FRACTPART == 14
#define MPC_MULTIPLY_SCF(CcVal, SCF_idx) \
    MPC_MULTIPLY_EX(CcVal, d->SCF[SCF_idx], d->SCF_shift[SCF_idx])
#else

#error FIXME, Cc table is in 18.14 format

#endif
#else
#define MPC_MULTIPLY_SCF(CcVal, SCF_idx) \
    MPC_MULTIPLY(CcVal, d->SCF[SCF_idx])
#endif
    // requantization and scaling of subband-samples
    for ( Band = 0; Band <= Last_Band; Band++ ) {   // setting pointers
        YL = d->Y_L[0] + Band;
        YR = d->Y_R[0] + Band;
        L  = d->Q[Band].L;
        R  = d->Q[Band].R;
        /************************** MS-coded **************************/
        if ( d->MS_Flag [Band] ) {
            if ( d->Res_L [Band] ) {
                if ( d->Res_R [Band] ) {    // M!=0, S!=0
                    facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][0] & 0xFF);
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][0] & 0xFF);
                    for ( n = 0; n < 12; n++, YL += 32, YR += 32 ) {
                        *YL   = (templ = MPC_MULTIPLY_FLOAT_INT(facL,*L++))+(tempr = MPC_MULTIPLY_FLOAT_INT(facR,*R++));
                        *YR   = templ - tempr;
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][1] & 0xFF);
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][1] & 0xFF);
                    for ( ; n < 24; n++, YL += 32, YR += 32 ) {
                        *YL   = (templ = MPC_MULTIPLY_FLOAT_INT(facL,*L++))+(tempr = MPC_MULTIPLY_FLOAT_INT(facR,*R++));
                        *YR   = templ - tempr;
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][2] & 0xFF);
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][2] & 0xFF);
                    for ( ; n < 36; n++, YL += 32, YR += 32 ) {
                        *YL   = (templ = MPC_MULTIPLY_FLOAT_INT(facL,*L++))+(tempr = MPC_MULTIPLY_FLOAT_INT(facR,*R++));
                        *YR   = templ - tempr;
                    }
                } else {    // M!=0, S==0
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][0] & 0xFF);
                    for ( n = 0; n < 12; n++, YL += 32, YR += 32 ) {
                        *YR = *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][1] & 0xFF);
                    for ( ; n < 24; n++, YL += 32, YR += 32 ) {
                        *YR = *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][2] & 0xFF);
                    for ( ; n < 36; n++, YL += 32, YR += 32 ) {
                        *YR = *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                    }
                }
            } else {
                if (d->Res_R[Band])    // M==0, S!=0
                {
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][0] & 0xFF);
                    for ( n = 0; n < 12; n++, YL += 32, YR += 32 ) {
                        *YR = - (*YL = MPC_MULTIPLY_FLOAT_INT(facR,*(R++)));
                    }
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][1] & 0xFF);
                    for ( ; n < 24; n++, YL += 32, YR += 32 ) {
                        *YR = - (*YL = MPC_MULTIPLY_FLOAT_INT(facR,*(R++)));
                    }
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][2] & 0xFF);
                    for ( ; n < 36; n++, YL += 32, YR += 32 ) {
                        *YR = - (*YL = MPC_MULTIPLY_FLOAT_INT(facR,*(R++)));
                    }
                } else {    // M==0, S==0
                    for ( n = 0; n < 36; n++, YL += 32, YR += 32 ) {
                        *YR = *YL = 0;
                    }
                }
            }
        }
        /************************** LR-coded **************************/
        else {
            if ( d->Res_L [Band] ) {
                if ( d->Res_R [Band] ) {    // L!=0, R!=0
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][0] & 0xFF);
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][0] & 0xFF);
                    for (n = 0; n < 12; n++, YL += 32, YR += 32 ) {
                        *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                        *YR = MPC_MULTIPLY_FLOAT_INT(facR,*R++);
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][1] & 0xFF);
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][1] & 0xFF);
                    for (; n < 24; n++, YL += 32, YR += 32 ) {
                        *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                        *YR = MPC_MULTIPLY_FLOAT_INT(facR,*R++);
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][2] & 0xFF);
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][2] & 0xFF);
                    for (; n < 36; n++, YL += 32, YR += 32 ) {
                        *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                        *YR = MPC_MULTIPLY_FLOAT_INT(facR,*R++);
                    }
                } else {     // L!=0, R==0
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][0] & 0xFF);
                    for ( n = 0; n < 12; n++, YL += 32, YR += 32 ) {
                        *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                        *YR = 0;
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][1] & 0xFF);
                    for ( ; n < 24; n++, YL += 32, YR += 32 ) {
                        *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                        *YR = 0;
                    }
					facL = MPC_MULTIPLY_SCF( Cc[d->Res_L[Band]] , d->SCF_Index_L[Band][2] & 0xFF);
                    for ( ; n < 36; n++, YL += 32, YR += 32 ) {
                        *YL = MPC_MULTIPLY_FLOAT_INT(facL,*L++);
                        *YR = 0;
                    }
                }
            }
            else {
                if ( d->Res_R [Band] ) {    // L==0, R!=0
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][0] & 0xFF);
                    for ( n = 0; n < 12; n++, YL += 32, YR += 32 ) {
                        *YL = 0;
                        *YR = MPC_MULTIPLY_FLOAT_INT(facR,*R++);
                    }
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][1] & 0xFF);
                    for ( ; n < 24; n++, YL += 32, YR += 32 ) {
                        *YL = 0;
                        *YR = MPC_MULTIPLY_FLOAT_INT(facR,*R++);
                    }
					facR = MPC_MULTIPLY_SCF( Cc[d->Res_R[Band]] , d->SCF_Index_R[Band][2] & 0xFF);
                    for ( ; n < 36; n++, YL += 32, YR += 32 ) {
                        *YL = 0;
                        *YR = MPC_MULTIPLY_FLOAT_INT(facR,*R++);
                    }
                } else {    // L==0, R==0
                    for ( n = 0; n < 36; n++, YL += 32, YR += 32 ) {
                        *YR = *YL = 0;
                    }
                }
            }
        }
    }
}

void mpc_decoder_read_bitstream_sv7(mpc_decoder * d, mpc_bits_reader * r)
{
    // these arrays hold decoding results for bundled quantizers (3- and 5-step)
    static const mpc_int32_t idx30[] = { -1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1};
    static const mpc_int32_t idx31[] = { -1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1};
    static const mpc_int32_t idx32[] = { -1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static const mpc_int32_t idx50[] = { -2,-1, 0, 1, 2,-2,-1, 0, 1, 2,-2,-1, 0, 1, 2,-2,-1, 0, 1, 2,-2,-1, 0, 1, 2};
    static const mpc_int32_t idx51[] = { -2,-2,-2,-2,-2,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};

    mpc_int32_t n, idx, Max_used_Band = 0;

    /***************************** Header *****************************/

    // first subband
	d->Res_L[0] = mpc_bits_read(r, 4);
	d->Res_R[0] = mpc_bits_read(r, 4);
	if (!(d->Res_L[0] == 0 && d->Res_R[0] == 0)) {
		if (d->ms)
        	d->MS_Flag[0] = mpc_bits_read(r, 1);
		Max_used_Band = 1;
	}

    // consecutive subbands
	for ( n = 1; n <= d->max_band; n++ ) {
		idx   = mpc_bits_huff_lut(r, & mpc_HuffHdr);
		d->Res_L[n] = (idx!=4) ? d->Res_L[n - 1] + idx : (int) mpc_bits_read(r, 4);

		idx   = mpc_bits_huff_lut(r, & mpc_HuffHdr);
		d->Res_R[n] = (idx!=4) ? d->Res_R[n - 1] + idx : (int) mpc_bits_read(r, 4);

		if (!(d->Res_L[n] == 0 && d->Res_R[n] == 0)) {
			if (d->ms)
            	d->MS_Flag[n] = mpc_bits_read(r, 1);
			Max_used_Band = n + 1;
		}
    }

    /****************************** SCFI ******************************/
    for ( n = 0; n < Max_used_Band; n++ ) {
		if (d->Res_L[n])
			d->SCFI_L[n] = mpc_bits_huff_dec(r, mpc_table_HuffSCFI);
		if (d->Res_R[n])
			d->SCFI_R[n] = mpc_bits_huff_dec(r, mpc_table_HuffSCFI);
    }

    /**************************** SCF/DSCF ****************************/
    for ( n = 0; n < Max_used_Band; n++ ) {
		mpc_int32_t * SCF = d->SCF_Index_L[n];
		mpc_uint32_t Res  = d->Res_L[n], SCFI = d->SCFI_L[n];
		do {
			if (Res) {
				switch (SCFI) {
					case 1:
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[0] = (idx!=8) ? SCF[2] + idx : (int) mpc_bits_read(r, 6);
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[1] = (idx!=8) ? SCF[0] + idx : (int) mpc_bits_read(r, 6);
						SCF[2] = SCF[1];
						break;
					case 3:
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[0] = (idx!=8) ? SCF[2] + idx : (int) mpc_bits_read(r, 6);
						SCF[1] = SCF[0];
						SCF[2] = SCF[1];
						break;
					case 2:
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[0] = (idx!=8) ? SCF[2] + idx : (int) mpc_bits_read(r, 6);
						SCF[1] = SCF[0];
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[2] = (idx!=8) ? SCF[1] + idx : (int) mpc_bits_read(r, 6);
						break;
					case 0:
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[0] = (idx!=8) ? SCF[2] + idx : (int) mpc_bits_read(r, 6);
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[1] = (idx!=8) ? SCF[0] + idx : (int) mpc_bits_read(r, 6);
						idx  = mpc_bits_huff_lut(r, & mpc_HuffDSCF);
						SCF[2] = (idx!=8) ? SCF[1] + idx : (int) mpc_bits_read(r, 6);
						break;
					default:
						return;
				}
				if (SCF[0] > 1024)
					SCF[0] = 0x8080;
				if (SCF[1] > 1024)
					SCF[1] = 0x8080;
				if (SCF[2] > 1024)
					SCF[2] = 0x8080;
			}
			Res = d->Res_R[n];
			SCFI = d->SCFI_R[n];
		} while ( SCF == d->SCF_Index_L[n] && (SCF = d->SCF_Index_R[n]));
    }

//     if (d->seeking == TRUE)
//         return;

    /***************************** Samples ****************************/
    for ( n = 0; n < Max_used_Band; n++ ) {
		mpc_int16_t *q = d->Q[n].L, Res = d->Res_L[n];
		do {
			mpc_int32_t k;
			const mpc_lut_data *Table;
			switch (Res) {
				case  -2: case  -3: case  -4: case  -5: case  -6: case  -7: case  -8: case  -9:
				case -10: case -11: case -12: case -13: case -14: case -15: case -16: case -17: case 0:
					break;
				case -1:
					for (k=0; k<36; k++ ) {
						mpc_uint32_t tmp = mpc_random_int(d);
						q[k] = ((tmp >> 24) & 0xFF) + ((tmp >> 16) & 0xFF) + ((tmp >>  8) & 0xFF) + ((tmp >>  0) & 0xFF) - 510;
					}
					break;
				case 1:
					Table = & mpc_HuffQ[0][mpc_bits_read(r, 1)];
					for ( k = 0; k < 36; k += 3) {
						idx = mpc_bits_huff_lut(r, Table);
						q[k] = idx30[idx];
						q[k + 1] = idx31[idx];
						q[k + 2] = idx32[idx];
					}
					break;
				case 2:
					Table = & mpc_HuffQ[1][mpc_bits_read(r, 1)];
					for ( k = 0; k < 36; k += 2) {
						idx = mpc_bits_huff_lut(r, Table);
						q[k] = idx50[idx];
						q[k + 1] = idx51[idx];
					}
					break;
				case 3:
				case 4:
				case 5:
				case 6:
				case 7:
					Table = & mpc_HuffQ[Res - 1][mpc_bits_read(r, 1)];
					for ( k = 0; k < 36; k++ )
						q[k] = mpc_bits_huff_lut(r, Table);
					break;
				case 8: case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16: case 17:
					for ( k = 0; k < 36; k++ )
						q[k] = (mpc_int32_t)mpc_bits_read(r, Res_bit[Res]) - Dc[Res];
					break;
				default:
					return;
			}

			Res = d->Res_R[n];
		} while (q == d->Q[n].L && (q = d->Q[n].R));
    }
}

void mpc_decoder_read_bitstream_sv8(mpc_decoder * d, mpc_bits_reader * r, mpc_bool_t is_key_frame)
{
    // these arrays hold decoding results for bundled quantizers (3- and 5-step)
	static const mpc_int8_t idx50[125] = {-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2};
	static const mpc_int8_t idx51[125] = {-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
	static const mpc_int8_t idx52[125] = {-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

	mpc_int32_t n, Max_used_Band;
	const mpc_can_data * Table, * Tables[2];

	/***************************** Header *****************************/

	if (is_key_frame == MPC_TRUE) {
		Max_used_Band = mpc_bits_log_dec(r, d->max_band + 1);
	} else {
		Max_used_Band = d->last_max_band + mpc_bits_can_dec(r, & mpc_can_Bands);
		if (Max_used_Band > 32) Max_used_Band -= 33;
	}
	d->last_max_band = Max_used_Band;

	if (Max_used_Band) {
		d->Res_L[Max_used_Band-1] = mpc_bits_can_dec(r, & mpc_can_Res[0]);
		d->Res_R[Max_used_Band-1] = mpc_bits_can_dec(r, & mpc_can_Res[0]);
		if (d->Res_L[Max_used_Band-1] > 15) d->Res_L[Max_used_Band-1] -= 17;
		if (d->Res_R[Max_used_Band-1] > 15) d->Res_R[Max_used_Band-1] -= 17;
		for ( n = Max_used_Band - 2; n >= 0; n--) {
			d->Res_L[n] = mpc_bits_can_dec(r, & mpc_can_Res[d->Res_L[n + 1] > 2]) + d->Res_L[n + 1];
			if (d->Res_L[n] > 15) d->Res_L[n] -= 17;
			d->Res_R[n] = mpc_bits_can_dec(r, & mpc_can_Res[d->Res_R[n + 1] > 2]) + d->Res_R[n + 1];
			if (d->Res_R[n] > 15) d->Res_R[n] -= 17;
		}

		if (d->ms) {
			int cnt = 0, tot = 0;
			mpc_uint32_t tmp = 0;
			for( n = 0; n < Max_used_Band; n++)
				if ( d->Res_L[n] != 0 || d->Res_R[n] != 0 )
					tot++;
			cnt = mpc_bits_log_dec(r, tot);
			if (cnt != 0 && cnt != tot)
				tmp = mpc_bits_enum_dec(r, mini(cnt, tot-cnt), tot);
			if (cnt * 2 > tot) tmp = ~tmp;
			for( n = Max_used_Band - 1; n >= 0; n--)
				if ( d->Res_L[n] != 0 || d->Res_R[n] != 0 ) {
					d->MS_Flag[n] = tmp & 1;
					tmp >>= 1;
				}
		}
	}

	for( n = Max_used_Band; n <= d->max_band; n++)
		d->Res_L[n] = d->Res_R[n] = 0;

	/****************************** SCFI ******************************/
	if (is_key_frame == MPC_TRUE){
		for( n = 0; n < 32; n++)
			d->DSCF_Flag_L[n] = d->DSCF_Flag_R[n] = 1; // new block -> force key frame
	}

	Tables[0] = & mpc_can_SCFI[0];
	Tables[1] = & mpc_can_SCFI[1];
	for ( n = 0; n < Max_used_Band; n++ ) {
		int tmp = 0, cnt = -1;
		if (d->Res_L[n]) cnt++;
		if (d->Res_R[n]) cnt++;
		if (cnt >= 0) {
			tmp = mpc_bits_can_dec(r, Tables[cnt]);
			if (d->Res_L[n]) d->SCFI_L[n] = tmp >> (2 * cnt);
			if (d->Res_R[n]) d->SCFI_R[n] = tmp & 3;
		}
	}

	/**************************** SCF/DSCF ****************************/

	for ( n = 0; n < Max_used_Band; n++ ) {
		mpc_int32_t * SCF = d->SCF_Index_L[n];
		mpc_uint32_t Res = d->Res_L[n], SCFI = d->SCFI_L[n];
		mpc_bool_t * DSCF_Flag = &d->DSCF_Flag_L[n];

		do {
			if ( Res ) {
				int m;
				if (*DSCF_Flag == 1) {
					SCF[0] = (mpc_int32_t)mpc_bits_read(r, 7) - 6;
					*DSCF_Flag = 0;
				} else {
					mpc_uint_t tmp = mpc_bits_can_dec(r, & mpc_can_DSCF[1]);
					if (tmp == 64)
						tmp += mpc_bits_read(r, 6);
					SCF[0] = ((SCF[2] - 25 + tmp) & 127) - 6;
				}
				for( m = 0; m < 2; m++){
					if (((SCFI << m) & 2) == 0) {
						mpc_uint_t tmp = mpc_bits_can_dec(r, & mpc_can_DSCF[0]);
						if (tmp == 31)
							tmp = 64 + mpc_bits_read(r, 6);
						SCF[m + 1] = ((SCF[m] - 25 + tmp) & 127) - 6;
					} else
						SCF[m + 1] = SCF[m];
				}
			}
			Res = d->Res_R[n];
			SCFI = d->SCFI_R[n];
			DSCF_Flag = &d->DSCF_Flag_R[n];
		} while ( SCF == d->SCF_Index_L[n] && (SCF = d->SCF_Index_R[n]));
	}

	/***************************** Samples ****************************/
	for ( n = 0; n < Max_used_Band; n++ ) {
		mpc_int16_t *q = d->Q[n].L, Res = d->Res_L[n];
		static const unsigned int thres[] = {0, 0, 3, 0, 0, 1, 3, 4, 8};
		static const mpc_int8_t HuffQ2_var[5*5*5] =
		{6, 5, 4, 5, 6, 5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6, 5, 4, 5, 6, 5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6, 5, 4, 5, 6, 5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6, 5, 4, 5, 6};

		do {
			mpc_int32_t k = 0, idx = 1;
			if (Res != 0) {
				if (Res == 2) {
					Tables[0] = & mpc_can_Q [0][0];
					Tables[1] = & mpc_can_Q [0][1];
					idx = 2 * thres[Res];
					for ( ; k < 36; k += 3) {
						int tmp = mpc_bits_can_dec(r, Tables[idx > thres[Res]]);
						q[k] = idx50[tmp];
						q[k + 1] = idx51[tmp];
						q[k + 2] = idx52[tmp];
						idx = (idx >> 1) + HuffQ2_var[tmp];
					}
				} else if (Res == 1) {
					Table = & mpc_can_Q1;
					for( ; k < 36; ){
						int kmax = k + 18;
						mpc_uint_t cnt = mpc_bits_can_dec(r, Table);
						idx = 0;
						if (cnt > 0 && cnt < 18)
							idx = mpc_bits_enum_dec(r, cnt <= 9 ? cnt : 18 - cnt, 18);
						if (cnt > 9) idx = ~idx;
						for ( ; k < kmax; k++) {
							q[k] = 0;
							if ( idx & (1 << 17) )
								q[k] = (mpc_bits_read(r, 1) << 1) - 1;
							idx <<= 1;
						}
					}
				} else if (Res == -1) {
					for ( ; k<36; k++ ) {
						mpc_uint32_t tmp = mpc_random_int(d);
						q[k] = ((tmp >> 24) & 0xFF) + ((tmp >> 16) & 0xFF) + ((tmp >>  8) & 0xFF) + ((tmp >>  0) & 0xFF) - 510;
					}
				} else if (Res <= 4) {
					Table = & mpc_can_Q[1][Res - 3];
					for ( ; k < 36; k += 2 ) {
						union {
							mpc_int8_t sym;
							struct { mpc_int8_t s1:4, s2:4; };
						} tmp;
						tmp.sym = mpc_bits_can_dec(r, Table);
						q[k] = tmp.s1;
						q[k + 1] = tmp.s2;
					}
				} else if (Res <= 8) {
					Tables[0] = & mpc_can_Q [Res - 3][0];
					Tables[1] = & mpc_can_Q [Res - 3][1];
					idx = 2 * thres[Res];
					for ( ; k < 36; k++ ) {
						q[k] = mpc_bits_can_dec(r, Tables[idx > thres[Res]]);
						idx = (idx >> 1) + absi(q[k]);
					}
				} else {
					for ( ; k < 36; k++ ) {
						q[k] = (unsigned char) mpc_bits_can_dec(r, & mpc_can_Q9up);
						if (Res != 9)
							q[k] = (q[k] << (Res - 9)) | mpc_bits_read(r, Res - 9);
						q[k] -= Dc[Res];
					}
				}
			}

			Res = d->Res_R[n];
		} while (q == d->Q[n].L && (q = d->Q[n].R));
	}
}

