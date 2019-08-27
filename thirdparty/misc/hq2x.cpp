/*
 * Copyright 2016 Bruno Ribeiro
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "hq2x.h"

#include "core/math/math_funcs.h"

static const uint32_t AMASK = 0xFF000000;
static const uint32_t YMASK = 0x00FF0000;
static const uint32_t UMASK = 0x0000FF00;
static const uint32_t VMASK = 0x000000FF;

_FORCE_INLINE_ static uint32_t ARGBtoAYUV(
	uint32_t value )
{
    uint32_t A, R, G, B, Y, U, V;
//todo big endian check
    A = value >> 24;
    R = (value >> 16) & 0xFF;
    G = (value >> 8) & 0xFF;
    B = value & 0xFF;

    Y = Math::fast_ftoi( 0.299 * R + 0.587 * G + 0.114 * B);
    U = Math::fast_ftoi(-0.169 * R - 0.331 * G +   0.5 * B) + 128;
    V = Math::fast_ftoi(   0.5 * R - 0.419 * G - 0.081 * B) + 128;
    return (A << 24) + (Y << 16) + (U << 8) + V;
}


/*
 * Use this function for sharper images (good for cartoon style, used by DOSBOX)
 */

_FORCE_INLINE_ static bool isDifferent(
	uint32_t color1,
	uint32_t color2,
	uint32_t trY,
	uint32_t trU,
	uint32_t trV,
	uint32_t trA )
{
	color1 = ARGBtoAYUV(color1);
	color2 = ARGBtoAYUV(color2);

	uint32_t value;

	value = ((color1 & YMASK) - (color2 & YMASK));
	value = (value ^ (value >> 31)) - (value >> 31);
	if (value > trY) return true;

	value = ((color1 & UMASK) - (color2 & UMASK));
	value = (value ^ (value >> 31)) - (value >> 31);
	if (value > trU) return true;

	value = ((color1 & VMASK) - (color2 & VMASK));
	value = (value ^ (value >> 31)) - (value >> 31);
	if (value > trV) return true;

	value = ((color1 & AMASK) - (color2 & AMASK));
	value = (value ^ (value >> 31)) - (value >> 31);
	if (value > trA) return true;

	return false;

}



#define MASK_RB   0x00FF00FF
#define MASK_G    0x0000FF00
#define MASK_A    0xFF000000


/**
 * @brief Mixes two colors using the given weights.
 */
#define HQX_MIX_2(C0,C1,W0,W1) \
	((((C0 & MASK_RB) * W0 + (C1 & MASK_RB) * W1) / (W0 + W1)) & MASK_RB) | \
	((((C0 & MASK_G)  * W0 + (C1 & MASK_G)  * W1) / (W0 + W1)) & MASK_G)  | \
	((((((C0 & MASK_A) >> 8)  * W0 + ((C1 & MASK_A) >> 8) * W1) / (W0 + W1)) << 8) & MASK_A)

/**
 * @brief Mixes three colors using the given weights.
 */
#define HQX_MIX_3(C0,C1,C2,W0,W1,W2) \
	((((C0 & MASK_RB) * W0 + (C1 & MASK_RB) * W1 + (C2 & MASK_RB) * W2) / (W0 + W1 + W2)) & MASK_RB) | \
	((((C0 & MASK_G)  * W0 + (C1 & MASK_G)  * W1 + (C2 & MASK_G)  * W2) / (W0 + W1 + W2)) & MASK_G)  | \
	((((((C0 & MASK_A) >> 8) * W0 + ((C1 & MASK_A) >> 8) * W1 + ((C2 & MASK_A) >> 8) * W2) / (W0 + W1 + W2)) << 8) & MASK_A)


#define MIX_00_4				*output = w[4];
#define MIX_00_MIX_00_4_0_3_1	*output = HQX_MIX_2(w[4],w[0],3U,1U);
#define MIX_00_4_3_3_1			*output = HQX_MIX_2(w[4],w[3],3U,1U);
#define MIX_00_4_1_3_1			*output = HQX_MIX_2(w[4],w[1],3U,1U);
#define MIX_00_3_1_1_1			*output = HQX_MIX_2(w[3],w[1],1U,1U);
#define MIX_00_4_3_1_2_1_1		*output = HQX_MIX_3(w[4],w[3],w[1],2U,1U,1U);
#define MIX_00_4_3_1_2_7_7 		*output = HQX_MIX_3(w[4],w[3],w[1],2U,7U,7U);
#define MIX_00_4_0_1_2_1_1		*output = HQX_MIX_3(w[4],w[0],w[1],2U,1U,1U);
#define MIX_00_4_0_3_2_1_1		*output = HQX_MIX_3(w[4],w[0],w[3],2U,1U,1U);
#define MIX_00_4_1_3_5_2_1		*output = HQX_MIX_3(w[4],w[1],w[3],5U,2U,1U);
#define MIX_00_4_3_1_5_2_1		*output = HQX_MIX_3(w[4],w[3],w[1],5U,2U,1U);
#define MIX_00_4_3_1_6_1_1		*output = HQX_MIX_3(w[4],w[3],w[1],6U,1U,1U);
#define MIX_00_4_3_1_2_3_3		*output = HQX_MIX_3(w[4],w[3],w[1],2U,3U,3U);
#define MIX_00_MIX_00_4_0_3_10	*output = HQX_MIX_3(w[4],w[3],w[1],14U,1U,1U);

#define MIX_01_4			*(output + 1) = w[4];
#define MIX_01_4_2_3_1		*(output + 1) = HQX_MIX_2(w[4],w[2],3U,1U);
#define MIX_01_4_1_3_1		*(output + 1) = HQX_MIX_2(w[4],w[1],3U,1U);
#define MIX_01_1_4_3_1		*(output + 1) = HQX_MIX_2(w[1],w[4],3U,1U);
#define MIX_01_4_5_3_1		*(output + 1) = HQX_MIX_2(w[4],w[5],3U,1U);
#define MIX_01_4_1_7_1		*(output + 1) = HQX_MIX_2(w[4],w[1],7U,1U);
#define MIX_01_4_1_5_2_1_1	*(output + 1) = HQX_MIX_3(w[4],w[1],w[5],2U,1U,1U);
#define MIX_01_4_2_5_2_1_1	*(output + 1) = HQX_MIX_3(w[4],w[2],w[5],2U,1U,1U);
#define MIX_01_4_2_1_2_1_1	*(output + 1) = HQX_MIX_3(w[4],w[2],w[1],2U,1U,1U);
#define MIX_01_4_5_1_5_2_1	*(output + 1) = HQX_MIX_3(w[4],w[5],w[1],5U,2U,1U);
#define MIX_01_4_1_5_5_2_1	*(output + 1) = HQX_MIX_3(w[4],w[1],w[5],5U,2U,1U);
#define MIX_01_4_1_5_6_1_1	*(output + 1) = HQX_MIX_3(w[4],w[1],w[5],6U,1U,1U);
#define MIX_01_4_1_5_2_3_3	*(output + 1) = HQX_MIX_3(w[4],w[1],w[5],2U,3U,3U);
#define MIX_01_4_2_3_10		*(output + 1) = HQX_MIX_3(w[4],w[1],w[5],14U,1U,1U);

#define MIX_02_4			*(output + 2) = w[4];
#define MIX_02_4_2_3_1		*(output + 2) = HQX_MIX_2(w[4],w[2],3U,1U);
#define MIX_02_4_1_3_1		*(output + 2) = HQX_MIX_2(w[4],w[1],3U,1U);
#define MIX_02_4_5_3_1  	*(output + 2) = HQX_MIX_2(w[4],w[5],3U,1U);
#define MIX_02_4_1_5_2_1_1	*(output + 2) = HQX_MIX_3(w[4],w[1],w[5],2U,1U,1U);
#define MIX_02_4_1_5_2_7_7	*(output + 2) = HQX_MIX_3(w[4],w[1],w[5],2U,7U,7U);
#define MIX_02_1_5_1_1		*(output + 2) = HQX_MIX_2(w[1],w[5],1U,1U);

#define MIX_10_4			*(output + lineSize) = w[4];
#define MIX_10_4_6_3_1		*(output + lineSize) = HQX_MIX_2(w[4],w[6],3U,1U);
#define MIX_10_4_7_3_1		*(output + lineSize) = HQX_MIX_2(w[4],w[7],3U,1U);
#define MIX_10_4_3_3_1		*(output + lineSize) = HQX_MIX_2(w[4],w[3],3U,1U);
#define MIX_10_4_7_3_2_1_1	*(output + lineSize) = HQX_MIX_3(w[4],w[7],w[3],2U,1U,1U);
#define MIX_10_4_6_3_2_1_1	*(output + lineSize) = HQX_MIX_3(w[4],w[6],w[3],2U,1U,1U);
#define MIX_10_4_6_7_2_1_1	*(output + lineSize) = HQX_MIX_3(w[4],w[6],w[7],2U,1U,1U);
#define MIX_10_4_3_7_5_2_1	*(output + lineSize) = HQX_MIX_3(w[4],w[3],w[7],5U,2U,1U);
#define MIX_10_4_7_3_5_2_1	*(output + lineSize) = HQX_MIX_3(w[4],w[7],w[3],5U,2U,1U);
#define MIX_10_4_7_3_6_1_1	*(output + lineSize) = HQX_MIX_3(w[4],w[7],w[3],6U,1U,1U);
#define MIX_10_4_7_3_2_3_3	*(output + lineSize) = HQX_MIX_3(w[4],w[7],w[3],2U,3U,3U);
#define MIX_10_4_6_3_10		*(output + lineSize) = HQX_MIX_3(w[4],w[7],w[3],14U,1U,1U);
#define MIX_10_4_3_7_1  	*(output + lineSize) = HQX_MIX_2(w[4],w[3],7U,1U);
#define MIX_10_3_4_3_1  	*(output + lineSize) = HQX_MIX_2(w[3],w[4],3U,1U);

#define MIX_11_4			*(output + lineSize + 1) = w[4];
#define MIX_11_4_8_3_1		*(output + lineSize + 1) = HQX_MIX_2(w[4],w[8],3U,1U);
#define MIX_11_4_5_3_1		*(output + lineSize + 1) = HQX_MIX_2(w[4],w[5],3U,1U);
#define MIX_11_4_7_3_1		*(output + lineSize + 1) = HQX_MIX_2(w[4],w[7],3U,1U);
#define MIX_11_4_5_7_2_1_1	*(output + lineSize + 1) = HQX_MIX_3(w[4],w[5],w[7],2U,1U,1U);
#define MIX_11_4_8_7_2_1_1	*(output + lineSize + 1) = HQX_MIX_3(w[4],w[8],w[7],2U,1U,1U);
#define MIX_11_4_8_5_2_1_1	*(output + lineSize + 1) = HQX_MIX_3(w[4],w[8],w[5],2U,1U,1U);
#define MIX_11_4_7_5_5_2_1	*(output + lineSize + 1) = HQX_MIX_3(w[4],w[7],w[5],5U,2U,1U);
#define MIX_11_4_5_7_5_2_1	*(output + lineSize + 1) = HQX_MIX_3(w[4],w[5],w[7],5U,2U,1U);
#define MIX_11_4_5_7_6_1_1	*(output + lineSize + 1) = HQX_MIX_3(w[4],w[5],w[7],6U,1U,1U);
#define MIX_11_4_5_7_2_3_3	*(output + lineSize + 1) = HQX_MIX_3(w[4],w[5],w[7],2U,3U,3U);
#define MIX_11_4_8_3_10		*(output + lineSize + 1) = HQX_MIX_3(w[4],w[5],w[7],14U,1U,1U);

#define MIX_12_4			*(output + lineSize + 2) = w[4];
#define MIX_12_4_5_3_1		*(output + lineSize + 2) = HQX_MIX_2(w[4],w[5],3U,1U);
#define MIX_12_4_5_7_1		*(output + lineSize + 2) = HQX_MIX_2(w[4],w[5],7U,1U);
#define MIX_12_5_4_3_1		*(output + lineSize + 2) = HQX_MIX_2(w[5],w[4],3U,1U);

#define MIX_20_4			*(output + lineSize + lineSize) = w[4];
#define MIX_20_4_6_3_1		*(output + lineSize + lineSize) = HQX_MIX_2(w[4],w[6],3U,1U);
#define MIX_20_4_7_3_1		*(output + lineSize + lineSize) = HQX_MIX_2(w[4],w[7],3U,1U);
#define MIX_20_4_3_3_1		*(output + lineSize + lineSize) = HQX_MIX_2(w[4],w[3],3U,1U);
#define MIX_20_4_7_3_2_1_1	*(output + lineSize + lineSize) = HQX_MIX_3(w[4],w[7],w[3],2U,1U,1U);
#define MIX_20_4_7_3_2_7_7	*(output + lineSize + lineSize) = HQX_MIX_3(w[4],w[7],w[3],2U,7U,7U);
#define MIX_20_7_3_1_1		*(output + lineSize + lineSize) = HQX_MIX_2(w[7],w[3],1U,1U);

#define MIX_21_4			*(output + lineSize + lineSize + 1) = w[4];
#define MIX_21_4_7_3_1		*(output + lineSize + lineSize + 1) = HQX_MIX_2(w[4],w[7],3U,1U);
#define MIX_21_4_7_7_1		*(output + lineSize + lineSize + 1) = HQX_MIX_2(w[4],w[7],7U,1U);
#define MIX_21_7_4_3_1		*(output + lineSize + lineSize + 1) = HQX_MIX_2(w[7],w[4],3U,1U);

#define MIX_22_4			*(output + lineSize + lineSize + 2) = w[4];
#define MIX_22_4_8_3_1		*(output + lineSize + lineSize + 2) = HQX_MIX_2(w[4],w[8],3U,1U);
#define MIX_22_4_7_3_1		*(output + lineSize + lineSize + 2) = HQX_MIX_2(w[4],w[7],3U,1U);
#define MIX_22_4_5_3_1		*(output + lineSize + lineSize + 2) = HQX_MIX_2(w[4],w[5],3U,1U);
#define MIX_22_4_5_7_2_1_1	*(output + lineSize + lineSize + 2) = HQX_MIX_3(w[4],w[5],w[7],2U,1U,1U);
#define MIX_22_4_5_7_2_7_7	*(output + lineSize + lineSize + 2) = HQX_MIX_3(w[4],w[5],w[7],2U,7U,7U);
#define MIX_22_5_7_1_1		*(output + lineSize + lineSize + 2) = HQX_MIX_2(w[5],w[7],1U,1U);



uint32_t *hq2x_resize(
	const uint32_t *image,
	uint32_t width,
	uint32_t height,
	uint32_t *output,
	uint32_t trY,
	uint32_t trU,
	uint32_t trV,
	uint32_t trA,
	bool wrapX,
	bool wrapY )
{
	int lineSize = width * 2;

	int previous, next;
	uint32_t w[9];

	trY <<= 16;
	trU <<= 8;
	trA <<= 24;

	// iterates between the lines
	for (uint32_t row = 0; row < height; row++)
	{
		/*
		 * Note: this function uses a 3x3 sliding window over the original image.
		 *
		 *   +----+----+----+
		 *   |    |    |    |
		 *   | w0 | w1 | w2 |
		 *   +----+----+----+
		 *   |    |    |    |
		 *   | w3 | w4 | w5 |
		 *   +----+----+----+
		 *   |    |    |    |
		 *   | w6 | w7 | w8 |
		 *   +----+----+----+
		 */

		// adjusts the previous and next line pointers
		if (row > 0)
			previous = -width;
		else
		{
			if (wrapY)
				previous = width * (height - 1);
			else
				previous = 0;
		}
		if (row < height - 1)
			next = width;
		else
		{
			if (wrapY)
				next = -(width * (height - 1));
			else
				next = 0;
		}

		// iterates between the columns
		for (uint32_t col = 0; col < width; col++)
		{
			w[1] = *(image + previous);
			w[4] = *image;
			w[7] = *(image + next);

			if (col > 0)
			{
				w[0] = *(image + previous - 1);
				w[3] = *(image - 1);
				w[6] = *(image + next - 1);
			}
			else
			{
				if (wrapX)
				{
					w[0] = *(image + previous + width - 1);
					w[3] = *(image + width - 1);
					w[6] = *(image + next + width - 1);
				}
				else
				{
					w[0] = w[1];
					w[3] = w[4];
					w[6] = w[7];
				}
			}

			if (col < width - 1)
			{
				w[2] = *(image + previous + 1);
				w[5] = *(image + 1);
				w[8] = *(image + next + 1);
			}
			else
			{
				if (wrapX)
				{
					w[2] = *(image + previous - width + 1);
					w[5] = *(image - width + 1);
					w[8] = *(image + next - width + 1);
				}
				else
				{
					w[2] = w[1];
					w[5] = w[4];
					w[8] = w[7];
				}
			}

			int pattern = 0;

			// computes the pattern to be used considering the neighbor pixels
			for (int k = 0, flag = 1; k < 9; k++)
			{
				// ignores the central pixel
				if (k == 4) continue;

				if (w[k] != w[4])
					if (isDifferent(w[4], w[k], trY, trU, trV, trA)) pattern |= flag;
				flag <<= 1;
			}

			switch (pattern)
			{
				case 0:
				case 1:
				case 4:
				case 32:
				case 128:
				case 5:
				case 132:
				case 160:
				case 33:
				case 129:
				case 36:
				case 133:
				case 164:
				case 161:
				case 37:
				case 165:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 2:
				case 34:
				case 130:
				case 162:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 16:
				case 17:
				case 48:
				case 49:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 64:
				case 65:
				case 68:
				case 69:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_8_5_2_1_1
					break;
				case 8:
				case 12:
				case 136:
				case 140:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 3:
				case 35:
				case 131:
				case 163:
					MIX_00_4_3_3_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 6:
				case 38:
				case 134:
				case 166:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_5_3_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 20:
				case 21:
				case 52:
				case 53:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 144:
				case 145:
				case 176:
				case 177:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_7_3_1
					break;
				case 192:
				case 193:
				case 196:
				case 197:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_5_3_1
					break;
				case 96:
				case 97:
				case 100:
				case 101:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_3_3_1
					MIX_11_4_8_5_2_1_1
					break;
				case 40:
				case 44:
				case 168:
				case 172:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_7_3_1
					MIX_11_4_5_7_2_1_1
					break;
				case 9:
				case 13:
				case 137:
				case 141:
					MIX_00_4_1_3_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 18:
				case 50:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_7_3_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 80:
				case 81:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 72:
				case 76:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 10:
				case 138:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 66:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_8_5_2_1_1
					break;
				case 24:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 7:
				case 39:
				case 135:
					MIX_00_4_3_3_1
					MIX_01_4_5_3_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 148:
				case 149:
				case 180:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_7_3_1
					break;
				case 224:
				case 228:
				case 225:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_3_3_1
					MIX_11_4_5_3_1
					break;
				case 41:
				case 169:
				case 45:
					MIX_00_4_1_3_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_7_3_1
					MIX_11_4_5_7_2_1_1
					break;
				case 22:
				case 54:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_7_3_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 208:
				case 209:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 104:
				case 108:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 11:
				case 139:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 19:
				case 51:
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
					MIX_00_4_3_3_1
					MIX_01_4_2_3_1
					}
					else
					{
					MIX_00_4_1_3_5_2_1
					MIX_01_4_1_5_2_3_3
					}
					MIX_10_4_7_3_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 146:
				case 178:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
					MIX_01_4_2_3_1
					MIX_11_4_7_3_1
					}
					else
					{
					MIX_01_4_1_5_2_3_3
					MIX_11_4_5_7_5_2_1
					}
					MIX_10_4_7_3_2_1_1
					break;
				case 84:
				case 85:
					MIX_00_4_3_1_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
					MIX_01_4_1_3_1
					MIX_11_4_8_3_1
					}
					else
					{
					MIX_01_4_5_1_5_2_1
					MIX_11_4_5_7_2_3_3
					}
					MIX_10_4_6_3_2_1_1
					break;
				case 112:
				case 113:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_2_1_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
					MIX_10_4_3_3_1
					MIX_11_4_8_3_1
					}
					else
					{
					MIX_10_4_7_3_5_2_1
					MIX_11_4_5_7_2_3_3
					}
					break;
				case 200:
				case 204:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
					MIX_10_4_6_3_1
					MIX_11_4_5_3_1
					}
					else
					{
					MIX_10_4_7_3_2_3_3
					MIX_11_4_7_5_5_2_1
					}
					break;
				case 73:
				case 77:
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
					MIX_00_4_1_3_1
					MIX_10_4_6_3_1
					}
					else
					{
					MIX_00_4_3_1_5_2_1
					MIX_10_4_7_3_2_3_3
					}
					MIX_01_4_1_5_2_1_1
					MIX_11_4_8_5_2_1_1
					break;
				case 42:
				case 170:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
					MIX_00_MIX_00_4_0_3_1
					MIX_10_4_7_3_1
					}
					else
					{
					MIX_00_4_3_1_2_3_3
					MIX_10_4_3_7_5_2_1
					}
					MIX_01_4_2_5_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 14:
				case 142:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
					MIX_00_MIX_00_4_0_3_1
					MIX_01_4_5_3_1
					}
					else
					{
					MIX_00_4_3_1_2_3_3
					MIX_01_4_1_5_5_2_1
					}
					MIX_10_4_6_7_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 67:
					MIX_00_4_3_3_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_8_5_2_1_1
					break;
				case 70:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_5_3_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_8_5_2_1_1
					break;
				case 28:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 152:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 194:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_5_3_1
					break;
				case 98:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_3_3_1
					MIX_11_4_8_5_2_1_1
					break;
				case 56:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 25:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 26:
				case 31:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_7_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 82:
				case 214:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 88:
				case 248:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_2_1_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 74:
				case 107:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 27:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_3_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 86:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_3_2_1_1
					MIX_11_4_8_3_1
					break;
				case 216:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 106:
					MIX_00_MIX_00_4_0_3_1
					MIX_01_4_2_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 30:
					MIX_00_MIX_00_4_0_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_7_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 210:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_2_3_1
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 120:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_2_1_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_3_1
					break;
				case 75:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_3_1
					MIX_11_4_8_5_2_1_1
					break;
				case 29:
					MIX_00_4_1_3_1
					MIX_01_4_1_3_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 198:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_5_3_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_5_3_1
					break;
				case 184:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_7_3_1
					MIX_11_4_7_3_1
					break;
				case 99:
					MIX_00_4_3_3_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_3_3_1
					MIX_11_4_8_5_2_1_1
					break;
				case 57:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 71:
					MIX_00_4_3_3_1
					MIX_01_4_5_3_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_8_5_2_1_1
					break;
				case 156:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 226:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_3_3_1
					MIX_11_4_5_3_1
					break;
				case 60:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 195:
					MIX_00_4_3_3_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_5_3_1
					break;
				case 102:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_5_3_1
					MIX_10_4_3_3_1
					MIX_11_4_8_5_2_1_1
					break;
				case 153:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 58:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 83:
					MIX_00_4_3_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 92:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 202:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					MIX_01_4_2_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					MIX_11_4_5_3_1
					break;
				case 78:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					MIX_01_4_5_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 154:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 114:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_3_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 89:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 90:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 55:
				case 23:
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
					MIX_00_4_3_3_1
					MIX_01_4
					}
					else
					{
					MIX_00_4_1_3_5_2_1
					MIX_01_4_1_5_2_3_3
					}
					MIX_10_4_7_3_2_1_1
					MIX_11_4_8_7_2_1_1
					break;
				case 182:
				case 150:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
					MIX_01_4
					MIX_11_4_7_3_1
					}
					else
					{
					MIX_01_4_1_5_2_3_3
					MIX_11_4_5_7_5_2_1
					}
					MIX_10_4_7_3_2_1_1
					break;
				case 213:
				case 212:
					MIX_00_4_3_1_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
					MIX_01_4_1_3_1
					MIX_11_4
					}
					else
					{
					MIX_01_4_5_1_5_2_1
					MIX_11_4_5_7_2_3_3
					}
					MIX_10_4_6_3_2_1_1
					break;
				case 241:
				case 240:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_2_1_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
					MIX_10_4_3_3_1
					MIX_11_4
					}
					else
					{
					MIX_10_4_7_3_5_2_1
					MIX_11_4_5_7_2_3_3
					}
					break;
				case 236:
				case 232:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
					MIX_10_4
					MIX_11_4_5_3_1
					}
					else
					{
					MIX_10_4_7_3_2_3_3
					MIX_11_4_7_5_5_2_1
					}
					break;
				case 109:
				case 105:
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
					MIX_00_4_1_3_1
					MIX_10_4
					}
					else
					{
					MIX_00_4_3_1_5_2_1
					MIX_10_4_7_3_2_3_3
					}
					MIX_01_4_1_5_2_1_1
					MIX_11_4_8_5_2_1_1
					break;
				case 171:
				case 43:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
					MIX_00_4
					MIX_10_4_7_3_1
					}
					else
					{
					MIX_00_4_3_1_2_3_3
					MIX_10_4_3_7_5_2_1
					}
					MIX_01_4_2_5_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 143:
				case 15:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
					MIX_00_4
					MIX_01_4_5_3_1
					}
					else
					{
					MIX_00_4_3_1_2_3_3
					MIX_01_4_1_5_5_2_1
					}
					MIX_10_4_6_7_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 124:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_3_1
					break;
				case 203:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_5_2_1_1
					MIX_10_4_6_3_1
					MIX_11_4_5_3_1
					break;
				case 62:
					MIX_00_MIX_00_4_0_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 211:
					MIX_00_4_3_3_1
					MIX_01_4_2_3_1
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 118:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_3_3_1
					MIX_11_4_8_3_1
					break;
				case 217:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_6_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 110:
					MIX_00_MIX_00_4_0_3_1
					MIX_01_4_5_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 155:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_3_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 188:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_7_3_1
					MIX_11_4_7_3_1
					break;
				case 185:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					MIX_10_4_7_3_1
					MIX_11_4_7_3_1
					break;
				case 61:
					MIX_00_4_1_3_1
					MIX_01_4_1_3_1
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 157:
					MIX_00_4_1_3_1
					MIX_01_4_1_3_1
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 103:
					MIX_00_4_3_3_1
					MIX_01_4_5_3_1
					MIX_10_4_3_3_1
					MIX_11_4_8_5_2_1_1
					break;
				case 227:
					MIX_00_4_3_3_1
					MIX_01_4_2_5_2_1_1
					MIX_10_4_3_3_1
					MIX_11_4_5_3_1
					break;
				case 230:
					MIX_00_4_0_3_2_1_1
					MIX_01_4_5_3_1
					MIX_10_4_3_3_1
					MIX_11_4_5_3_1
					break;
				case 199:
					MIX_00_4_3_3_1
					MIX_01_4_5_3_1
					MIX_10_4_6_3_2_1_1
					MIX_11_4_5_3_1
					break;
				case 220:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 158:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 234:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					MIX_01_4_2_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_5_3_1
					break;
				case 242:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_3_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 59:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 121:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 87:
					MIX_00_4_3_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 79:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_5_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 122:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 94:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 218:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 91:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 229:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_3_3_1
					MIX_11_4_5_3_1
					break;
				case 167:
					MIX_00_4_3_3_1
					MIX_01_4_5_3_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_5_7_2_1_1
					break;
				case 173:
					MIX_00_4_1_3_1
					MIX_01_4_1_5_2_1_1
					MIX_10_4_7_3_1
					MIX_11_4_5_7_2_1_1
					break;
				case 181:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_7_3_2_1_1
					MIX_11_4_7_3_1
					break;
				case 186:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_7_3_1
					MIX_11_4_7_3_1
					break;
				case 115:
					MIX_00_4_3_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_3_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 93:
					MIX_00_4_1_3_1
					MIX_01_4_1_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 206:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					MIX_01_4_5_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					MIX_11_4_5_3_1
					break;
				case 205:
				case 201:
					MIX_00_4_1_3_1
					MIX_01_4_1_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4_6_3_1
					}
					else
					{
						MIX_10_4_7_3_6_1_1
					}
					MIX_11_4_5_3_1
					break;
				case 174:
				case 46:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_MIX_00_4_0_3_1
					}
					else
					{
						MIX_00_4_3_1_6_1_1
					}
					MIX_01_4_5_3_1
					MIX_10_4_7_3_1
					MIX_11_4_5_7_2_1_1
					break;
				case 179:
				case 147:
					MIX_00_4_3_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4_2_3_1
					}
					else
					{
						MIX_01_4_1_5_6_1_1
					}
					MIX_10_4_7_3_2_1_1
					MIX_11_4_7_3_1
					break;
				case 117:
				case 116:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_3_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4_8_3_1
					}
					else
					{
						MIX_11_4_5_7_6_1_1
					}
					break;
				case 189:
					MIX_00_4_1_3_1
					MIX_01_4_1_3_1
					MIX_10_4_7_3_1
					MIX_11_4_7_3_1
					break;
				case 231:
					MIX_00_4_3_3_1
					MIX_01_4_5_3_1
					MIX_10_4_3_3_1
					MIX_11_4_5_3_1
					break;
				case 126:
					MIX_00_MIX_00_4_0_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_3_1
					break;
				case 219:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_3_1
					MIX_10_4_6_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 125:
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
					MIX_00_4_1_3_1
					MIX_10_4
					}
					else
					{
					MIX_00_4_3_1_5_2_1
					MIX_10_4_7_3_2_3_3
					}
					MIX_01_4_1_3_1
					MIX_11_4_8_3_1
					break;
				case 221:
					MIX_00_4_1_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
					MIX_01_4_1_3_1
					MIX_11_4
					}
					else
					{
					MIX_01_4_5_1_5_2_1
					MIX_11_4_5_7_2_3_3
					}
					MIX_10_4_6_3_1
					break;
				case 207:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
					MIX_00_4
					MIX_01_4_5_3_1
					}
					else
					{
					MIX_00_4_3_1_2_3_3
					MIX_01_4_1_5_5_2_1
					}
					MIX_10_4_6_3_1
					MIX_11_4_5_3_1
					break;
				case 238:
					MIX_00_MIX_00_4_0_3_1
					MIX_01_4_5_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
					MIX_10_4
					MIX_11_4_5_3_1
					}
					else
					{
					MIX_10_4_7_3_2_3_3
					MIX_11_4_7_5_5_2_1
					}
					break;
				case 190:
					MIX_00_MIX_00_4_0_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
					MIX_01_4
					MIX_11_4_7_3_1
					}
					else
					{
					MIX_01_4_1_5_2_3_3
					MIX_11_4_5_7_5_2_1
					}
					MIX_10_4_7_3_1
					break;
				case 187:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
					MIX_00_4
					MIX_10_4_7_3_1
					}
					else
					{
					MIX_00_4_3_1_2_3_3
					MIX_10_4_3_7_5_2_1
					}
					MIX_01_4_2_3_1
					MIX_11_4_7_3_1
					break;
				case 243:
					MIX_00_4_3_3_1
					MIX_01_4_2_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
					MIX_10_4_3_3_1
					MIX_11_4
					}
					else
					{
					MIX_10_4_7_3_5_2_1
					MIX_11_4_5_7_2_3_3
					}
					break;
				case 119:
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
					MIX_00_4_3_3_1
					MIX_01_4
					}
					else
					{
					MIX_00_4_1_3_5_2_1
					MIX_01_4_1_5_2_3_3
					}
					MIX_10_4_3_3_1
					MIX_11_4_8_3_1
					break;
				case 237:
				case 233:
					MIX_00_4_1_3_1
					MIX_01_4_1_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_6_3_10
					}
					MIX_11_4_5_3_1
					break;
				case 175:
				case 47:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_MIX_00_4_0_3_10
					}
					MIX_01_4_5_3_1
					MIX_10_4_7_3_1
					MIX_11_4_5_7_2_1_1
					break;
				case 183:
				case 151:
					MIX_00_4_3_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_2_3_10
					}
					MIX_10_4_7_3_2_1_1
					MIX_11_4_7_3_1
					break;
				case 245:
				case 244:
					MIX_00_4_3_1_2_1_1
					MIX_01_4_1_3_1
					MIX_10_4_3_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_8_3_10
					}
					break;
				case 250:
					MIX_00_MIX_00_4_0_3_1
					MIX_01_4_2_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 123:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_3_1
					break;
				case 95:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_3_1
					MIX_11_4_8_3_1
					break;
				case 222:
					MIX_00_MIX_00_4_0_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_6_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 252:
					MIX_00_4_0_1_2_1_1
					MIX_01_4_1_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_8_3_10
					}
					break;
				case 249:
					MIX_00_4_1_3_1
					MIX_01_4_2_1_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_6_3_10
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 235:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_5_2_1_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_6_3_10
					}
					MIX_11_4_5_3_1
					break;
				case 111:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_MIX_00_4_0_3_10
					}
					MIX_01_4_5_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_5_2_1_1
					break;
				case 63:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_MIX_00_4_0_3_10
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_7_3_1
					MIX_11_4_8_7_2_1_1
					break;
				case 159:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_2_3_10
					}
					MIX_10_4_6_7_2_1_1
					MIX_11_4_7_3_1
					break;
				case 215:
					MIX_00_4_3_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_2_3_10
					}
					MIX_10_4_6_3_2_1_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 246:
					MIX_00_4_0_3_2_1_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					MIX_10_4_3_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_8_3_10
					}
					break;
				case 254:
					MIX_00_MIX_00_4_0_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_8_3_10
					}
					break;
				case 253:
					MIX_00_4_1_3_1
					MIX_01_4_1_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_6_3_10
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_8_3_10
					}
					break;
				case 251:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					MIX_01_4_2_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_6_3_10
					}
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 239:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_MIX_00_4_0_3_10
					}
					MIX_01_4_5_3_1
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_6_3_10
					}
					MIX_11_4_5_3_1
					break;
				case 127:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_MIX_00_4_0_3_10
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_1_5_2_1_1
					}
					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
					{
						MIX_10_4
					}
					else
					{
						MIX_10_4_7_3_2_1_1
					}
					MIX_11_4_8_3_1
					break;
				case 191:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_MIX_00_4_0_3_10
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_2_3_10
					}
					MIX_10_4_7_3_1
					MIX_11_4_7_3_1
					break;
				case 223:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
					{
						MIX_00_4
					}
					else
					{
						MIX_00_4_3_1_2_1_1
					}
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_2_3_10
					}
					MIX_10_4_6_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_5_7_2_1_1
					}
					break;
				case 247:
					MIX_00_4_3_3_1
					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
					{
						MIX_01_4
					}
					else
					{
						MIX_01_4_2_3_10
					}
					MIX_10_4_3_3_1
					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
					{
						MIX_11_4
					}
					else
					{
						MIX_11_4_8_3_10
					}
					break;
				case 255:
					if (isDifferent(w[3], w[1], trY, trU, trV, trA))
						MIX_00_4
					else
						MIX_00_MIX_00_4_0_3_10

					if (isDifferent(w[1], w[5], trY, trU, trV, trA))
						MIX_01_4
					else
						MIX_01_4_2_3_10

					if (isDifferent(w[7], w[3], trY, trU, trV, trA))
						MIX_10_4
					else
						MIX_10_4_6_3_10

					if (isDifferent(w[5], w[7], trY, trU, trV, trA))
						MIX_11_4
					else
						MIX_11_4_8_3_10
					break;
			}
			image++;
			output += 2;
		}
		output += lineSize;
	}

	return output;
}
