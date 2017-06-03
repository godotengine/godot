/*************************************************************************/
/*  reverb_sw.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "reverb_sw.h"
#include "print_string.h"
#include "stdlib.h"
#define SETMIN(x, y) (x) = MIN((x), (y))
#define rangeloop(c, min, max) \
	for ((c) = (min); (c) < (max); (c)++)

#define ABSDIFF(x, y) \
	(((x) < (y)) ? ((y) - (x)) : ((x) - (y)))

#ifdef bleh_MSC_VER

#if _MSC_VER >= 1400
_FORCE_INLINE_ int32_tMULSHIFT_S32(
		int32_t Factor1,
		int32_t Factor2,
		uint8_t Bits) {

	return __ll_rshift(
			__emul(Factor1, Factor2),
			Bits);
}
#endif

#else
#define MULSHIFT_S32(Factor1, Factor2, Bits) \
	((int)(((int64_t)(Factor1) * (Factor2)) >> (Bits)))
#endif

struct ReverbParamsSW {
	unsigned int BufferSize; // Required buffer size
	int gLPF; // Coefficient
	int gEcho0; // Coefficient
	int gEcho1; // Coefficient
	int gEcho2; // Coefficient
	int gEcho3; // Coefficient
	int gWall; // Coefficient
	int gReva; // Coefficient
	int gRevb; // Coefficient
	int gInputL; // Coefficient
	int gInputR; // Coefficient
	unsigned int nRevaOldL; // Offset
	unsigned int nRevaOldR; // Offset
	unsigned int nRevbOldL; // Offset
	unsigned int nRevbOldR; // Offset
	unsigned int nLwlNew; // Offset
	unsigned int nRwrNew; // Offset
	unsigned int nEcho0L; // Offset
	unsigned int nEcho0R; // Offset
	unsigned int nEcho1L; // Offset
	unsigned int nEcho1R; // Offset
	unsigned int nLwlOld; // Offset
	unsigned int nRwrOld; // Offset
	unsigned int nLwrNew; // Offset
	unsigned int nRwlNew; // Offset
	unsigned int nEcho2L; // Offset
	unsigned int nEcho2R; // Offset
	unsigned int nEcho3L; // Offset
	unsigned int nEcho3R; // Offset
	unsigned int nLwrOld; // Offset
	unsigned int nRwlOld; // Offset
	unsigned int nRevaNewL; // Offset
	unsigned int nRevaNewR; // Offset
	unsigned int nRevbNewL; // Offset
	unsigned int nRevbNewR; // Offset
};

static ReverbParamsSW reverb_params_Room = {
	0x26C0 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x6D80, 0x54B8, -0x4130, 0x0000, 0x0000, -0x4580,
	//gReva		gRevb		gInputL		gInputR
	0x5800, 0x5300, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x01B4 - 0x007D, 0x0136 - 0x007D, 0x00B8 - 0x005B, 0x005C - 0x005B,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x04D6, 0x0333, 0x03F0, 0x0227, 0x0374, 0x01EF,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x0334, 0x01B5, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x0000, 0x0000, 0x01B4, 0x0136, 0x00B8, 0x005C
};

static ReverbParamsSW reverb_params_StudioSmall = {
	0x1F40 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x70F0, 0x4FA8, -0x4320, 0x4410, -0x3F10, -0x6400,
	//gReva		gRevb		gInputL		gInputR
	0x5280, 0x4EC0, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x00B4 - 0x0033, 0x0080 - 0x0033, 0x004C - 0x0025, 0x0026 - 0x0025,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x03E4, 0x031B, 0x03A4, 0x02AF, 0x0372, 0x0266,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x031C, 0x025D, 0x025C, 0x018E, 0x022F, 0x0135, 0x01D2, 0x00B7,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x018F, 0x00B5, 0x00B4, 0x0080, 0x004C, 0x0026
};

static ReverbParamsSW reverb_params_StudioMedium = {
	0x4840 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x70F0, 0x4FA8, -0x4320, 0x4510, -0x4110, -0x4B40,
	//gReva		gRevb		gInputL		gInputR
	0x5280, 0x4EC0, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x0264 - 0x00B1, 0x01B2 - 0x00B1, 0x0100 - 0x007F, 0x0080 - 0x007F,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x0904, 0x076B, 0x0824, 0x065F, 0x07A2, 0x0616,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x076C, 0x05ED, 0x05EC, 0x042E, 0x050F, 0x0305, 0x0462, 0x02B7,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x042F, 0x0265, 0x0264, 0x01B2, 0x0100, 0x0080
};

static ReverbParamsSW reverb_params_StudioLarge = {
	0x6FE0 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x6F60, 0x4FA8, -0x4320, 0x4510, -0x4110, -0x5980,
	//gReva		gRevb		gInputL		gInputR
	0x5680, 0x52C0, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x031C - 0x00E3, 0x0238 - 0x00E3, 0x0154 - 0x00A9, 0x00AA - 0x00A9,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x0DFB, 0x0B58, 0x0D09, 0x0A3C, 0x0BD9, 0x0973,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x0B59, 0x08DA, 0x08D9, 0x05E9, 0x07EC, 0x04B0, 0x06EF, 0x03D2,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x05EA, 0x031D, 0x031C, 0x0238, 0x0154, 0x00AA
};

static ReverbParamsSW reverb_params_Hall = {
	0xADE0 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x6000, 0x5000, 0x4C00, -0x4800, -0x4400, -0x4000,
	//gReva		gRevb		gInputL		gInputR
	0x6000, 0x5C00, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x05C0 - 0x01A5, 0x041A - 0x01A5, 0x0274 - 0x0139, 0x013A - 0x0139,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x15BA, 0x11BB, 0x14C2, 0x10BD, 0x11BC, 0x0DC1,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x11C0, 0x0DC3, 0x0DC0, 0x09C1, 0x0BC4, 0x07C1, 0x0A00, 0x06CD,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x09C2, 0x05C1, 0x05C0, 0x041A, 0x0274, 0x013A
};

static ReverbParamsSW reverb_params_SpaceEcho = {
	0xF6C0 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x7E00, 0x5000, -0x4C00, -0x5000, 0x4C00, -0x5000,
	//gReva		gRevb		gInputL		gInputR
	0x6000, 0x5400, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x0AE0 - 0x033D, 0x07A2 - 0x033D, 0x0464 - 0x0231, 0x0232 - 0x0231,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x1ED6, 0x1A31, 0x1D14, 0x183B, 0x1BC2, 0x16B2,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x1A32, 0x15EF, 0x15EE, 0x1055, 0x1334, 0x0F2D, 0x11F6, 0x0C5D,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x1056, 0x0AE1, 0x0AE0, 0x07A2, 0x0464, 0x0232
};

static ReverbParamsSW reverb_params_Echo = {
	0x18040 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x7FFF, 0x7FFF, 0x0000, 0x0000, 0x0000, -0x7F00,
	//gReva		gRevb		gInputL		gInputR
	0x0000, 0x0000, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x1004 - 0x0001, 0x1002 - 0x0001, 0x0004 - 0x0001, 0x0002 - 0x0001,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x1FFF, 0x0FFF, 0x1005, 0x0005, 0x0000, 0x0000,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x1005, 0x0005, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x0000, 0x0000, 0x1004, 0x1002, 0x0004, 0x0002
};

static ReverbParamsSW reverb_params_Delay = {
	0x18040 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x7FFF, 0x7FFF, 0x0000, 0x0000, 0x0000, 0x0000,
	//gReva		gRevb		gInputL		gInputR
	0x0000, 0x0000, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x1004 - 0x0001, 0x1002 - 0x0001, 0x0004 - 0x0001, 0x0002 - 0x0001,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x1FFF, 0x0FFF, 0x1005, 0x0005, 0x0000, 0x0000,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x1005, 0x0005, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x0000, 0x0000, 0x1004, 0x1002, 0x0004, 0x0002
};

static ReverbParamsSW reverb_params_HalfEcho = {
	0x3C00 / 2,
	//gLPF		gEcho0		gEcho1		gEcho2		gEcho3		gWall
	0x70F0, 0x4FA8, -0x4320, 0x4510, -0x4110, -0x7B00,
	//gReva		gRevb		gInputL		gInputR
	0x5F80, 0x54C0, -0x8000, -0x8000,
	//nRevaOldL			nRevaOldR			nRevbOldL			nRevbOldR
	0x0058 - 0x0017, 0x0040 - 0x0017, 0x0028 - 0x0013, 0x0014 - 0x0013,
	//nLwlNew	nRwrNew		nEcho0L		nEcho0R		nEcho1L		nEcho1R
	0x0371, 0x02AF, 0x02E5, 0x01DF, 0x02B0, 0x01D7,
	//nLwlOld	nRwrOld		nLwrNew		nRwlNew		nEcho2L		nEcho2R		nEcho3L		nEcho3R
	0x0358, 0x026A, 0x01D6, 0x011E, 0x012D, 0x00B1, 0x011F, 0x0059,
	//nLwrOld	nRwlOld		nRevaNewL	nRevaNewR	nRevbNewL	nRevbNewR
	0x01A0, 0x00E3, 0x0058, 0x0040, 0x0028, 0x0014
};

static ReverbParamsSW *reverb_param_modes[] = {
	&reverb_params_Room,
	&reverb_params_StudioSmall,
	&reverb_params_StudioMedium,
	&reverb_params_StudioLarge,
	&reverb_params_Hall,
	&reverb_params_SpaceEcho,
	&reverb_params_Echo,
	&reverb_params_Delay,
	&reverb_params_HalfEcho,
};

bool ReverbSW::process(int *p_input, int *p_output, int p_frames, int p_stereo_stride) {

	if (!reverb_buffer)
		return false;

//
// p_input must point to a non-looping buffer.
// BOTH p_input and p_output must be touched (use ClearModuleBuffer).

// LOCAL MACROS

#undef LM_SETSRCOFFSET
#define LM_SETSRCOFFSET(x)            \
	(x) = current_params->x + Offset; \
	if ((x) >= reverb_buffer_size) {  \
		(x) -= reverb_buffer_size;    \
	}                                 \
	SETMIN(aSample, reverb_buffer_size - (x));

/*
#undef LM_SETSRCOFFSET2
#define LM_SETSRCOFFSET2(x,y)							\
			(x) = ((y) << 3) >> HZShift;					\
			(x) += Offset;									\
			if ( (x) >= reverb_buffer_size ) {						\
				(x) -= reverb_buffer_size;							\
	}												\
			SETMIN ( aSample, reverb_buffer_size - (x) );
*/
#undef LM_SRCADVANCE
#define LM_SRCADVANCE(x) \
	(x) += aSample;

#undef LM_MUL
#define LM_MUL(x, y) \
	MULSHIFT_S32(x, current_params->y, 15)

#undef LM_REVERB
#define LM_REVERB(x) reverb_buffer[(x) + cSample]

	// LOCAL VARIABLES

	unsigned int Offset;

	int lwl, lwr, rwl, rwr;
	//unsigned char HZShift;

	// CODE

	lwl = state.lwl;
	lwr = state.lwr;
	rwl = state.rwl;
	rwr = state.rwr;
	Offset = state.Offset;

	int max = 0;

	while (p_frames) {

		// Offsets

		unsigned int nLwlOld;
		unsigned int nRwrOld;
		unsigned int nLwlNew;
		unsigned int nRwrNew;

		unsigned int nLwrOld;
		unsigned int nRwlOld;
		unsigned int nLwrNew;
		unsigned int nRwlNew;

		unsigned int nEcho0L;
		unsigned int nEcho1L;
		unsigned int nEcho2L;
		unsigned int nEcho3L;

		unsigned int nEcho0R;
		unsigned int nEcho1R;
		unsigned int nEcho2R;
		unsigned int nEcho3R;

		unsigned int nRevaOldL;
		unsigned int nRevaOldR;
		unsigned int nRevbOldL;
		unsigned int nRevbOldR;

		unsigned int nRevaNewL;
		unsigned int nRevaNewR;
		unsigned int nRevbNewL;
		unsigned int nRevbNewR;

		// Other variables

		unsigned int aSample = p_frames;

		// Set initial offsets

		LM_SETSRCOFFSET(nLwlOld);
		LM_SETSRCOFFSET(nRwrOld);
		LM_SETSRCOFFSET(nLwlNew);
		LM_SETSRCOFFSET(nRwrNew);
		LM_SETSRCOFFSET(nLwrOld);
		LM_SETSRCOFFSET(nRwlOld);
		LM_SETSRCOFFSET(nLwrNew);
		LM_SETSRCOFFSET(nRwlNew);
		LM_SETSRCOFFSET(nEcho0L);
		LM_SETSRCOFFSET(nEcho1L);
		LM_SETSRCOFFSET(nEcho2L);
		LM_SETSRCOFFSET(nEcho3L);
		LM_SETSRCOFFSET(nEcho0R);
		LM_SETSRCOFFSET(nEcho1R);
		LM_SETSRCOFFSET(nEcho2R);
		LM_SETSRCOFFSET(nEcho3R);
		LM_SETSRCOFFSET(nRevaOldL);
		LM_SETSRCOFFSET(nRevaOldR);
		LM_SETSRCOFFSET(nRevbOldL);
		LM_SETSRCOFFSET(nRevbOldR);
		LM_SETSRCOFFSET(nRevaNewL);
		LM_SETSRCOFFSET(nRevaNewR);
		LM_SETSRCOFFSET(nRevbNewL);
		LM_SETSRCOFFSET(nRevbNewR);

		//SETMIN ( aSample, p_output.Size - p_output.Offset );

		for (unsigned int cSample = 0; cSample < aSample; cSample++) {

			int tempL0, tempL1, tempR0, tempR1;

			tempL1 = p_input[(cSample << p_stereo_stride)] >> 8;
			tempR1 = p_input[(cSample << p_stereo_stride) + 1] >> 8;

			tempL0 = LM_MUL(tempL1, gInputL);
			tempR0 = LM_MUL(tempR1, gInputR);

			/*
			Left -> Wall -> Left Reflection
			*/
			tempL1 = tempL0 + LM_MUL(LM_REVERB(nLwlOld), gWall);
			tempR1 = tempR0 + LM_MUL(LM_REVERB(nRwrOld), gWall);
			lwl += LM_MUL(tempL1 - lwl, gLPF);
			rwr += LM_MUL(tempR1 - rwr, gLPF);
			LM_REVERB(nLwlNew) = lwl;
			LM_REVERB(nRwrNew) = rwr;
			/*
			Left -> Wall -> Right Reflection
			*/
			tempL1 = tempL0 + LM_MUL(LM_REVERB(nRwlOld), gWall);
			tempR1 = tempR0 + LM_MUL(LM_REVERB(nLwrOld), gWall);
			lwr += LM_MUL(tempL1 - lwr, gLPF);
			rwl += LM_MUL(tempR1 - rwl, gLPF);
			LM_REVERB(nLwrNew) = lwr;
			LM_REVERB(nRwlNew) = rwl;
			/*
			Early Echo(Early Reflection)
			*/
			tempL0 =
					LM_MUL(LM_REVERB(nEcho0L), gEcho0) +
					LM_MUL(LM_REVERB(nEcho1L), gEcho1) +
					LM_MUL(LM_REVERB(nEcho2L), gEcho2) +
					LM_MUL(LM_REVERB(nEcho3L), gEcho3);
			tempR0 =
					LM_MUL(LM_REVERB(nEcho0R), gEcho0) +
					LM_MUL(LM_REVERB(nEcho1R), gEcho1) +
					LM_MUL(LM_REVERB(nEcho2R), gEcho2) +
					LM_MUL(LM_REVERB(nEcho3R), gEcho3);
			/*
			Late Reverb
			*/
			tempL1 = LM_REVERB(nRevaOldL);
			tempR1 = LM_REVERB(nRevaOldR);
			tempL0 -= LM_MUL(tempL1, gReva);
			tempR0 -= LM_MUL(tempR1, gReva);
			LM_REVERB(nRevaNewL) = tempL0;
			LM_REVERB(nRevaNewR) = tempR0;
			tempL0 = LM_MUL(tempL0, gReva) + tempL1;
			tempR0 = LM_MUL(tempR0, gReva) + tempR1;
			tempL1 = LM_REVERB(nRevbOldL);
			tempR1 = LM_REVERB(nRevbOldR);
			tempL0 -= LM_MUL(tempL1, gRevb);
			tempR0 -= LM_MUL(tempR1, gRevb);
			LM_REVERB(nRevbNewL) = tempL0;
			LM_REVERB(nRevbNewR) = tempR0;
			tempL0 = LM_MUL(tempL0, gRevb) + tempL1;
			tempR0 = LM_MUL(tempR0, gRevb) + tempR1;
			/*
			Output
			*/

			max |= abs(tempL0);
			max |= abs(tempR0);

			p_output[(cSample << p_stereo_stride)] += tempL0 << 8;
			p_output[(cSample << p_stereo_stride) + 1] += tempR0 << 8;
		}

		// Advance offsets

		Offset += aSample;
		if (Offset >= reverb_buffer_size) {
			Offset -= reverb_buffer_size;
		}

		p_input += aSample << p_stereo_stride;
		p_output += aSample << p_stereo_stride;

		p_frames -= aSample;
	}

	state.lwl = lwl;
	state.lwr = lwr;
	state.rwl = rwl;
	state.rwr = rwr;
	state.Offset = Offset;

	return (max & 0x7FFFFF00) != 0; // audio was mixed?
}

void ReverbSW::adjust_current_params() {

	*current_params = *reverb_param_modes[mode];

	uint32_t maxofs = 0;

#define LM_CONFIG_PARAM(x)                                                                             \
	current_params->x = (int)(((int64_t)current_params->x * (int64_t)mix_rate * 8L) / (int64_t)44100); \
	if (current_params->x > maxofs)                                                                    \
		maxofs = current_params->x;

	LM_CONFIG_PARAM(nLwlOld);
	LM_CONFIG_PARAM(nRwrOld);
	LM_CONFIG_PARAM(nLwlNew);
	LM_CONFIG_PARAM(nRwrNew);
	LM_CONFIG_PARAM(nLwrOld);
	LM_CONFIG_PARAM(nRwlOld);
	LM_CONFIG_PARAM(nLwrNew);
	LM_CONFIG_PARAM(nRwlNew);
	LM_CONFIG_PARAM(nEcho0L);
	LM_CONFIG_PARAM(nEcho1L);
	LM_CONFIG_PARAM(nEcho2L);
	LM_CONFIG_PARAM(nEcho3L);
	LM_CONFIG_PARAM(nEcho0R);
	LM_CONFIG_PARAM(nEcho1R);
	LM_CONFIG_PARAM(nEcho2R);
	LM_CONFIG_PARAM(nEcho3R);
	LM_CONFIG_PARAM(nRevaOldL);
	LM_CONFIG_PARAM(nRevaOldR);
	LM_CONFIG_PARAM(nRevbOldL);
	LM_CONFIG_PARAM(nRevbOldR);
	LM_CONFIG_PARAM(nRevaNewL);
	LM_CONFIG_PARAM(nRevaNewR);
	LM_CONFIG_PARAM(nRevbNewL);
	LM_CONFIG_PARAM(nRevbNewR);

	int needed_buffer_size = maxofs + 1;
	if (reverb_buffer)
		memdelete_arr(reverb_buffer);

	reverb_buffer = memnew_arr(int, needed_buffer_size);
	reverb_buffer_size = needed_buffer_size;

	for (uint32_t i = 0; i < reverb_buffer_size; i++)
		reverb_buffer[i] = 0;

	state.reset();
}

void ReverbSW::set_mode(ReverbMode p_mode) {

	if (mode == p_mode)
		return;

	mode = p_mode;

	adjust_current_params();
}

void ReverbSW::set_mix_rate(int p_mix_rate) {

	if (p_mix_rate == mix_rate)
		return;

	mix_rate = p_mix_rate;

	adjust_current_params();
}

ReverbSW::ReverbSW() {

	reverb_buffer = 0;
	reverb_buffer_size = 0;
	mode = REVERB_MODE_ROOM;
	mix_rate = 1;
	current_params = memnew(ReverbParamsSW);
}

ReverbSW::~ReverbSW() {

	if (reverb_buffer)
		memdelete_arr(reverb_buffer);

	memdelete(current_params);
}
