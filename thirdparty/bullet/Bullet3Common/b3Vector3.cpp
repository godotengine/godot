/*
 Copyright (c) 2011-213 Apple Inc. http://bulletphysics.org

 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it freely,
 subject to the following restrictions:

 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.

 This source version has been altered.
 */

#if defined(_WIN32) || defined(__i386__)
#define B3_USE_SSE_IN_API
#endif

#include "b3Vector3.h"

#if defined(B3_USE_SSE) || defined(B3_USE_NEON)

#ifdef __APPLE__
#include <stdint.h>
typedef float float4 __attribute__((vector_size(16)));
#else
#define float4 __m128
#endif
//typedef  uint32_t uint4 __attribute__ ((vector_size(16)));

#if defined B3_USE_SSE || defined _WIN32

#define LOG2_ARRAY_SIZE 6
#define STACK_ARRAY_COUNT (1UL << LOG2_ARRAY_SIZE)

#include <emmintrin.h>

long b3_maxdot_large(const float *vv, const float *vec, unsigned long count, float *dotResult);
long b3_maxdot_large(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	const float4 *vertices = (const float4 *)vv;
	static const unsigned char indexTable[16] = {(unsigned char)-1, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
	float4 dotMax = b3Assign128(-B3_INFINITY, -B3_INFINITY, -B3_INFINITY, -B3_INFINITY);
	float4 vvec = _mm_loadu_ps(vec);
	float4 vHi = b3CastiTo128f(_mm_shuffle_epi32(b3CastfTo128i(vvec), 0xaa));  /// zzzz
	float4 vLo = _mm_movelh_ps(vvec, vvec);                                    /// xyxy

	long maxIndex = -1L;

	size_t segment = 0;
	float4 stack_array[STACK_ARRAY_COUNT];

#if DEBUG
	// memset( stack_array, -1, STACK_ARRAY_COUNT * sizeof(stack_array[0]) );
#endif

	size_t index;
	float4 max;
	// Faster loop without cleanup code for full tiles
	for (segment = 0; segment + STACK_ARRAY_COUNT * 4 <= count; segment += STACK_ARRAY_COUNT * 4)
	{
		max = dotMax;

		for (index = 0; index < STACK_ARRAY_COUNT; index += 4)
		{  // do four dot products at a time. Carefully avoid touching the w element.
			float4 v0 = vertices[0];
			float4 v1 = vertices[1];
			float4 v2 = vertices[2];
			float4 v3 = vertices[3];
			vertices += 4;

			float4 lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			float4 hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			float4 lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			float4 hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			float4 z = _mm_shuffle_ps(hi0, hi1, 0x88);
			float4 x = _mm_shuffle_ps(lo0, lo1, 0x88);
			float4 y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 1] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 2] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 3] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			// It is too costly to keep the index of the max here. We will look for it again later.  We save a lot of work this way.
		}

		// If we found a new max
		if (0xf != _mm_movemask_ps((float4)_mm_cmpeq_ps(max, dotMax)))
		{
			// copy the new max across all lanes of our max accumulator
			max = _mm_max_ps(max, (float4)_mm_shuffle_ps(max, max, 0x4e));
			max = _mm_max_ps(max, (float4)_mm_shuffle_ps(max, max, 0xb1));

			dotMax = max;

			// find first occurrence of that max
			size_t test;
			for (index = 0; 0 == (test = _mm_movemask_ps(_mm_cmpeq_ps(stack_array[index], max))); index++)  // local_count must be a multiple of 4
			{
			}
			// record where it is.
			maxIndex = 4 * index + segment + indexTable[test];
		}
	}

	// account for work we've already done
	count -= segment;

	// Deal with the last < STACK_ARRAY_COUNT vectors
	max = dotMax;
	index = 0;

	if (b3Unlikely(count > 16))
	{
		for (; index + 4 <= count / 4; index += 4)
		{  // do four dot products at a time. Carefully avoid touching the w element.
			float4 v0 = vertices[0];
			float4 v1 = vertices[1];
			float4 v2 = vertices[2];
			float4 v3 = vertices[3];
			vertices += 4;

			float4 lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			float4 hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			float4 lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			float4 hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			float4 z = _mm_shuffle_ps(hi0, hi1, 0x88);
			float4 x = _mm_shuffle_ps(lo0, lo1, 0x88);
			float4 y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 1] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 2] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 3] = x;
			max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan

			// It is too costly to keep the index of the max here. We will look for it again later.  We save a lot of work this way.
		}
	}

	size_t localCount = (count & -4L) - 4 * index;
	if (localCount)
	{
#ifdef __APPLE__
		float4 t0, t1, t2, t3, t4;
		float4 *sap = &stack_array[index + localCount / 4];
		vertices += localCount;  // counter the offset
		size_t byteIndex = -(localCount) * sizeof(float);
		//AT&T Code style assembly
		asm volatile(
			".align 4                                                                   \n\
             0: movaps  %[max], %[t2]                            // move max out of the way to avoid propagating NaNs in max \n\
          movaps  (%[vertices], %[byteIndex], 4),    %[t0]    // vertices[0]      \n\
          movaps  16(%[vertices], %[byteIndex], 4),  %[t1]    // vertices[1]      \n\
          movaps  %[t0], %[max]                               // vertices[0]      \n\
          movlhps %[t1], %[max]                               // x0y0x1y1         \n\
         movaps  32(%[vertices], %[byteIndex], 4),  %[t3]    // vertices[2]      \n\
         movaps  48(%[vertices], %[byteIndex], 4),  %[t4]    // vertices[3]      \n\
          mulps   %[vLo], %[max]                              // x0y0x1y1 * vLo   \n\
         movhlps %[t0], %[t1]                                // z0w0z1w1         \n\
         movaps  %[t3], %[t0]                                // vertices[2]      \n\
         movlhps %[t4], %[t0]                                // x2y2x3y3         \n\
         mulps   %[vLo], %[t0]                               // x2y2x3y3 * vLo   \n\
          movhlps %[t3], %[t4]                                // z2w2z3w3         \n\
          shufps  $0x88, %[t4], %[t1]                         // z0z1z2z3         \n\
          mulps   %[vHi], %[t1]                               // z0z1z2z3 * vHi   \n\
         movaps  %[max], %[t3]                               // x0y0x1y1 * vLo   \n\
         shufps  $0x88, %[t0], %[max]                        // x0x1x2x3 * vLo.x \n\
         shufps  $0xdd, %[t0], %[t3]                         // y0y1y2y3 * vLo.y \n\
         addps   %[t3], %[max]                               // x + y            \n\
         addps   %[t1], %[max]                               // x + y + z        \n\
         movaps  %[max], (%[sap], %[byteIndex])              // record result for later scrutiny \n\
         maxps   %[t2], %[max]                               // record max, restore max   \n\
         add     $16, %[byteIndex]                           // advance loop counter\n\
         jnz     0b                                          \n\
     "
			: [max] "+x"(max), [t0] "=&x"(t0), [t1] "=&x"(t1), [t2] "=&x"(t2), [t3] "=&x"(t3), [t4] "=&x"(t4), [byteIndex] "+r"(byteIndex)
			: [vLo] "x"(vLo), [vHi] "x"(vHi), [vertices] "r"(vertices), [sap] "r"(sap)
			: "memory", "cc");
		index += localCount / 4;
#else
		{
			for (unsigned int i = 0; i < localCount / 4; i++, index++)
			{  // do four dot products at a time. Carefully avoid touching the w element.
				float4 v0 = vertices[0];
				float4 v1 = vertices[1];
				float4 v2 = vertices[2];
				float4 v3 = vertices[3];
				vertices += 4;

				float4 lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
				float4 hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
				float4 lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
				float4 hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

				lo0 = lo0 * vLo;
				lo1 = lo1 * vLo;
				float4 z = _mm_shuffle_ps(hi0, hi1, 0x88);
				float4 x = _mm_shuffle_ps(lo0, lo1, 0x88);
				float4 y = _mm_shuffle_ps(lo0, lo1, 0xdd);
				z = z * vHi;
				x = x + y;
				x = x + z;
				stack_array[index] = x;
				max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan
			}
		}
#endif  //__APPLE__
	}

	// process the last few points
	if (count & 3)
	{
		float4 v0, v1, v2, x, y, z;
		switch (count & 3)
		{
			case 3:
			{
				v0 = vertices[0];
				v1 = vertices[1];
				v2 = vertices[2];

				// Calculate 3 dot products, transpose, duplicate v2
				float4 lo0 = _mm_movelh_ps(v0, v1);  // xyxy.lo
				float4 hi0 = _mm_movehl_ps(v1, v0);  // z?z?.lo
				lo0 = lo0 * vLo;
				z = _mm_shuffle_ps(hi0, v2, 0xa8);  // z0z1z2z2
				z = z * vHi;
				float4 lo1 = _mm_movelh_ps(v2, v2);  // xyxy
				lo1 = lo1 * vLo;
				x = _mm_shuffle_ps(lo0, lo1, 0x88);
				y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			}
			break;
			case 2:
			{
				v0 = vertices[0];
				v1 = vertices[1];
				float4 xy = _mm_movelh_ps(v0, v1);
				z = _mm_movehl_ps(v1, v0);
				xy = xy * vLo;
				z = _mm_shuffle_ps(z, z, 0xa8);
				x = _mm_shuffle_ps(xy, xy, 0xa8);
				y = _mm_shuffle_ps(xy, xy, 0xfd);
				z = z * vHi;
			}
			break;
			case 1:
			{
				float4 xy = vertices[0];
				z = _mm_shuffle_ps(xy, xy, 0xaa);
				xy = xy * vLo;
				z = z * vHi;
				x = _mm_shuffle_ps(xy, xy, 0);
				y = _mm_shuffle_ps(xy, xy, 0x55);
			}
			break;
		}
		x = x + y;
		x = x + z;
		stack_array[index] = x;
		max = _mm_max_ps(x, max);  // control the order here so that max is never NaN even if x is nan
		index++;
	}

	// if we found a new max.
	if (0 == segment || 0xf != _mm_movemask_ps((float4)_mm_cmpeq_ps(max, dotMax)))
	{  // we found a new max. Search for it
		// find max across the max vector, place in all elements of max -- big latency hit here
		max = _mm_max_ps(max, (float4)_mm_shuffle_ps(max, max, 0x4e));
		max = _mm_max_ps(max, (float4)_mm_shuffle_ps(max, max, 0xb1));

		// It is slightly faster to do this part in scalar code when count < 8. However, the common case for
		// this where it actually makes a difference is handled in the early out at the top of the function,
		// so it is less than a 1% difference here. I opted for improved code size, fewer branches and reduced
		// complexity, and removed it.

		dotMax = max;

		// scan for the first occurence of max in the array
		size_t test;
		for (index = 0; 0 == (test = _mm_movemask_ps(_mm_cmpeq_ps(stack_array[index], max))); index++)  // local_count must be a multiple of 4
		{
		}
		maxIndex = 4 * index + segment + indexTable[test];
	}

	_mm_store_ss(dotResult, dotMax);
	return maxIndex;
}

long b3_mindot_large(const float *vv, const float *vec, unsigned long count, float *dotResult);

long b3_mindot_large(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	const float4 *vertices = (const float4 *)vv;
	static const unsigned char indexTable[16] = {(unsigned char)-1, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};

	float4 dotmin = b3Assign128(B3_INFINITY, B3_INFINITY, B3_INFINITY, B3_INFINITY);
	float4 vvec = _mm_loadu_ps(vec);
	float4 vHi = b3CastiTo128f(_mm_shuffle_epi32(b3CastfTo128i(vvec), 0xaa));  /// zzzz
	float4 vLo = _mm_movelh_ps(vvec, vvec);                                    /// xyxy

	long minIndex = -1L;

	size_t segment = 0;
	float4 stack_array[STACK_ARRAY_COUNT];

#if DEBUG
	// memset( stack_array, -1, STACK_ARRAY_COUNT * sizeof(stack_array[0]) );
#endif

	size_t index;
	float4 min;
	// Faster loop without cleanup code for full tiles
	for (segment = 0; segment + STACK_ARRAY_COUNT * 4 <= count; segment += STACK_ARRAY_COUNT * 4)
	{
		min = dotmin;

		for (index = 0; index < STACK_ARRAY_COUNT; index += 4)
		{  // do four dot products at a time. Carefully avoid touching the w element.
			float4 v0 = vertices[0];
			float4 v1 = vertices[1];
			float4 v2 = vertices[2];
			float4 v3 = vertices[3];
			vertices += 4;

			float4 lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			float4 hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			float4 lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			float4 hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			float4 z = _mm_shuffle_ps(hi0, hi1, 0x88);
			float4 x = _mm_shuffle_ps(lo0, lo1, 0x88);
			float4 y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 1] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 2] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 3] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			// It is too costly to keep the index of the min here. We will look for it again later.  We save a lot of work this way.
		}

		// If we found a new min
		if (0xf != _mm_movemask_ps((float4)_mm_cmpeq_ps(min, dotmin)))
		{
			// copy the new min across all lanes of our min accumulator
			min = _mm_min_ps(min, (float4)_mm_shuffle_ps(min, min, 0x4e));
			min = _mm_min_ps(min, (float4)_mm_shuffle_ps(min, min, 0xb1));

			dotmin = min;

			// find first occurrence of that min
			size_t test;
			for (index = 0; 0 == (test = _mm_movemask_ps(_mm_cmpeq_ps(stack_array[index], min))); index++)  // local_count must be a multiple of 4
			{
			}
			// record where it is.
			minIndex = 4 * index + segment + indexTable[test];
		}
	}

	// account for work we've already done
	count -= segment;

	// Deal with the last < STACK_ARRAY_COUNT vectors
	min = dotmin;
	index = 0;

	if (b3Unlikely(count > 16))
	{
		for (; index + 4 <= count / 4; index += 4)
		{  // do four dot products at a time. Carefully avoid touching the w element.
			float4 v0 = vertices[0];
			float4 v1 = vertices[1];
			float4 v2 = vertices[2];
			float4 v3 = vertices[3];
			vertices += 4;

			float4 lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			float4 hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			float4 lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			float4 hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			float4 z = _mm_shuffle_ps(hi0, hi1, 0x88);
			float4 x = _mm_shuffle_ps(lo0, lo1, 0x88);
			float4 y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 1] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 2] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			v0 = vertices[0];
			v1 = vertices[1];
			v2 = vertices[2];
			v3 = vertices[3];
			vertices += 4;

			lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
			hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
			lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
			hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

			lo0 = lo0 * vLo;
			lo1 = lo1 * vLo;
			z = _mm_shuffle_ps(hi0, hi1, 0x88);
			x = _mm_shuffle_ps(lo0, lo1, 0x88);
			y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			z = z * vHi;
			x = x + y;
			x = x + z;
			stack_array[index + 3] = x;
			min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan

			// It is too costly to keep the index of the min here. We will look for it again later.  We save a lot of work this way.
		}
	}

	size_t localCount = (count & -4L) - 4 * index;
	if (localCount)
	{
#ifdef __APPLE__
		vertices += localCount;  // counter the offset
		float4 t0, t1, t2, t3, t4;
		size_t byteIndex = -(localCount) * sizeof(float);
		float4 *sap = &stack_array[index + localCount / 4];

		asm volatile(
			".align 4                                                                   \n\
             0: movaps  %[min], %[t2]                            // move min out of the way to avoid propagating NaNs in min \n\
             movaps  (%[vertices], %[byteIndex], 4),    %[t0]    // vertices[0]      \n\
             movaps  16(%[vertices], %[byteIndex], 4),  %[t1]    // vertices[1]      \n\
             movaps  %[t0], %[min]                               // vertices[0]      \n\
             movlhps %[t1], %[min]                               // x0y0x1y1         \n\
             movaps  32(%[vertices], %[byteIndex], 4),  %[t3]    // vertices[2]      \n\
             movaps  48(%[vertices], %[byteIndex], 4),  %[t4]    // vertices[3]      \n\
             mulps   %[vLo], %[min]                              // x0y0x1y1 * vLo   \n\
             movhlps %[t0], %[t1]                                // z0w0z1w1         \n\
             movaps  %[t3], %[t0]                                // vertices[2]      \n\
             movlhps %[t4], %[t0]                                // x2y2x3y3         \n\
             movhlps %[t3], %[t4]                                // z2w2z3w3         \n\
             mulps   %[vLo], %[t0]                               // x2y2x3y3 * vLo   \n\
             shufps  $0x88, %[t4], %[t1]                         // z0z1z2z3         \n\
             mulps   %[vHi], %[t1]                               // z0z1z2z3 * vHi   \n\
             movaps  %[min], %[t3]                               // x0y0x1y1 * vLo   \n\
             shufps  $0x88, %[t0], %[min]                        // x0x1x2x3 * vLo.x \n\
             shufps  $0xdd, %[t0], %[t3]                         // y0y1y2y3 * vLo.y \n\
             addps   %[t3], %[min]                               // x + y            \n\
             addps   %[t1], %[min]                               // x + y + z        \n\
             movaps  %[min], (%[sap], %[byteIndex])              // record result for later scrutiny \n\
             minps   %[t2], %[min]                               // record min, restore min   \n\
             add     $16, %[byteIndex]                           // advance loop counter\n\
             jnz     0b                                          \n\
             "
			: [min] "+x"(min), [t0] "=&x"(t0), [t1] "=&x"(t1), [t2] "=&x"(t2), [t3] "=&x"(t3), [t4] "=&x"(t4), [byteIndex] "+r"(byteIndex)
			: [vLo] "x"(vLo), [vHi] "x"(vHi), [vertices] "r"(vertices), [sap] "r"(sap)
			: "memory", "cc");
		index += localCount / 4;
#else
		{
			for (unsigned int i = 0; i < localCount / 4; i++, index++)
			{  // do four dot products at a time. Carefully avoid touching the w element.
				float4 v0 = vertices[0];
				float4 v1 = vertices[1];
				float4 v2 = vertices[2];
				float4 v3 = vertices[3];
				vertices += 4;

				float4 lo0 = _mm_movelh_ps(v0, v1);  // x0y0x1y1
				float4 hi0 = _mm_movehl_ps(v1, v0);  // z0?0z1?1
				float4 lo1 = _mm_movelh_ps(v2, v3);  // x2y2x3y3
				float4 hi1 = _mm_movehl_ps(v3, v2);  // z2?2z3?3

				lo0 = lo0 * vLo;
				lo1 = lo1 * vLo;
				float4 z = _mm_shuffle_ps(hi0, hi1, 0x88);
				float4 x = _mm_shuffle_ps(lo0, lo1, 0x88);
				float4 y = _mm_shuffle_ps(lo0, lo1, 0xdd);
				z = z * vHi;
				x = x + y;
				x = x + z;
				stack_array[index] = x;
				min = _mm_min_ps(x, min);  // control the order here so that max is never NaN even if x is nan
			}
		}

#endif
	}

	// process the last few points
	if (count & 3)
	{
		float4 v0, v1, v2, x, y, z;
		switch (count & 3)
		{
			case 3:
			{
				v0 = vertices[0];
				v1 = vertices[1];
				v2 = vertices[2];

				// Calculate 3 dot products, transpose, duplicate v2
				float4 lo0 = _mm_movelh_ps(v0, v1);  // xyxy.lo
				float4 hi0 = _mm_movehl_ps(v1, v0);  // z?z?.lo
				lo0 = lo0 * vLo;
				z = _mm_shuffle_ps(hi0, v2, 0xa8);  // z0z1z2z2
				z = z * vHi;
				float4 lo1 = _mm_movelh_ps(v2, v2);  // xyxy
				lo1 = lo1 * vLo;
				x = _mm_shuffle_ps(lo0, lo1, 0x88);
				y = _mm_shuffle_ps(lo0, lo1, 0xdd);
			}
			break;
			case 2:
			{
				v0 = vertices[0];
				v1 = vertices[1];
				float4 xy = _mm_movelh_ps(v0, v1);
				z = _mm_movehl_ps(v1, v0);
				xy = xy * vLo;
				z = _mm_shuffle_ps(z, z, 0xa8);
				x = _mm_shuffle_ps(xy, xy, 0xa8);
				y = _mm_shuffle_ps(xy, xy, 0xfd);
				z = z * vHi;
			}
			break;
			case 1:
			{
				float4 xy = vertices[0];
				z = _mm_shuffle_ps(xy, xy, 0xaa);
				xy = xy * vLo;
				z = z * vHi;
				x = _mm_shuffle_ps(xy, xy, 0);
				y = _mm_shuffle_ps(xy, xy, 0x55);
			}
			break;
		}
		x = x + y;
		x = x + z;
		stack_array[index] = x;
		min = _mm_min_ps(x, min);  // control the order here so that min is never NaN even if x is nan
		index++;
	}

	// if we found a new min.
	if (0 == segment || 0xf != _mm_movemask_ps((float4)_mm_cmpeq_ps(min, dotmin)))
	{  // we found a new min. Search for it
		// find min across the min vector, place in all elements of min -- big latency hit here
		min = _mm_min_ps(min, (float4)_mm_shuffle_ps(min, min, 0x4e));
		min = _mm_min_ps(min, (float4)_mm_shuffle_ps(min, min, 0xb1));

		// It is slightly faster to do this part in scalar code when count < 8. However, the common case for
		// this where it actually makes a difference is handled in the early out at the top of the function,
		// so it is less than a 1% difference here. I opted for improved code size, fewer branches and reduced
		// complexity, and removed it.

		dotmin = min;

		// scan for the first occurence of min in the array
		size_t test;
		for (index = 0; 0 == (test = _mm_movemask_ps(_mm_cmpeq_ps(stack_array[index], min))); index++)  // local_count must be a multiple of 4
		{
		}
		minIndex = 4 * index + segment + indexTable[test];
	}

	_mm_store_ss(dotResult, dotmin);
	return minIndex;
}

#elif defined B3_USE_NEON
#define ARM_NEON_GCC_COMPATIBILITY 1
#include <arm_neon.h>

static long b3_maxdot_large_v0(const float *vv, const float *vec, unsigned long count, float *dotResult);
static long b3_maxdot_large_v1(const float *vv, const float *vec, unsigned long count, float *dotResult);
static long b3_maxdot_large_sel(const float *vv, const float *vec, unsigned long count, float *dotResult);
static long b3_mindot_large_v0(const float *vv, const float *vec, unsigned long count, float *dotResult);
static long b3_mindot_large_v1(const float *vv, const float *vec, unsigned long count, float *dotResult);
static long b3_mindot_large_sel(const float *vv, const float *vec, unsigned long count, float *dotResult);

long (*b3_maxdot_large)(const float *vv, const float *vec, unsigned long count, float *dotResult) = b3_maxdot_large_sel;
long (*b3_mindot_large)(const float *vv, const float *vec, unsigned long count, float *dotResult) = b3_mindot_large_sel;

extern "C"
{
	int _get_cpu_capabilities(void);
}

static long b3_maxdot_large_sel(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	if (_get_cpu_capabilities() & 0x2000)
		b3_maxdot_large = _maxdot_large_v1;
	else
		b3_maxdot_large = _maxdot_large_v0;

	return b3_maxdot_large(vv, vec, count, dotResult);
}

static long b3_mindot_large_sel(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	if (_get_cpu_capabilities() & 0x2000)
		b3_mindot_large = _mindot_large_v1;
	else
		b3_mindot_large = _mindot_large_v0;

	return b3_mindot_large(vv, vec, count, dotResult);
}

#define vld1q_f32_aligned_postincrement(_ptr) ({ float32x4_t _r; asm( "vld1.f32  {%0}, [%1, :128]!\n" : "=w" (_r), "+r" (_ptr) ); /*return*/ _r; })

long b3_maxdot_large_v0(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	unsigned long i = 0;
	float32x4_t vvec = vld1q_f32_aligned_postincrement(vec);
	float32x2_t vLo = vget_low_f32(vvec);
	float32x2_t vHi = vdup_lane_f32(vget_high_f32(vvec), 0);
	float32x2_t dotMaxLo = (float32x2_t){-B3_INFINITY, -B3_INFINITY};
	float32x2_t dotMaxHi = (float32x2_t){-B3_INFINITY, -B3_INFINITY};
	uint32x2_t indexLo = (uint32x2_t){0, 1};
	uint32x2_t indexHi = (uint32x2_t){2, 3};
	uint32x2_t iLo = (uint32x2_t){-1, -1};
	uint32x2_t iHi = (uint32x2_t){-1, -1};
	const uint32x2_t four = (uint32x2_t){4, 4};

	for (; i + 8 <= count; i += 8)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
		float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);
		float32x2_t xy2 = vmul_f32(vget_low_f32(v2), vLo);
		float32x2_t xy3 = vmul_f32(vget_low_f32(v3), vLo);

		float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x2x2_t z1 = vtrn_f32(vget_high_f32(v2), vget_high_f32(v3));
		float32x2_t zLo = vmul_f32(z0.val[0], vHi);
		float32x2_t zHi = vmul_f32(z1.val[0], vHi);

		float32x2_t rLo = vpadd_f32(xy0, xy1);
		float32x2_t rHi = vpadd_f32(xy2, xy3);
		rLo = vadd_f32(rLo, zLo);
		rHi = vadd_f32(rHi, zHi);

		uint32x2_t maskLo = vcgt_f32(rLo, dotMaxLo);
		uint32x2_t maskHi = vcgt_f32(rHi, dotMaxHi);
		dotMaxLo = vbsl_f32(maskLo, rLo, dotMaxLo);
		dotMaxHi = vbsl_f32(maskHi, rHi, dotMaxHi);
		iLo = vbsl_u32(maskLo, indexLo, iLo);
		iHi = vbsl_u32(maskHi, indexHi, iHi);
		indexLo = vadd_u32(indexLo, four);
		indexHi = vadd_u32(indexHi, four);

		v0 = vld1q_f32_aligned_postincrement(vv);
		v1 = vld1q_f32_aligned_postincrement(vv);
		v2 = vld1q_f32_aligned_postincrement(vv);
		v3 = vld1q_f32_aligned_postincrement(vv);

		xy0 = vmul_f32(vget_low_f32(v0), vLo);
		xy1 = vmul_f32(vget_low_f32(v1), vLo);
		xy2 = vmul_f32(vget_low_f32(v2), vLo);
		xy3 = vmul_f32(vget_low_f32(v3), vLo);

		z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
		z1 = vtrn_f32(vget_high_f32(v2), vget_high_f32(v3));
		zLo = vmul_f32(z0.val[0], vHi);
		zHi = vmul_f32(z1.val[0], vHi);

		rLo = vpadd_f32(xy0, xy1);
		rHi = vpadd_f32(xy2, xy3);
		rLo = vadd_f32(rLo, zLo);
		rHi = vadd_f32(rHi, zHi);

		maskLo = vcgt_f32(rLo, dotMaxLo);
		maskHi = vcgt_f32(rHi, dotMaxHi);
		dotMaxLo = vbsl_f32(maskLo, rLo, dotMaxLo);
		dotMaxHi = vbsl_f32(maskHi, rHi, dotMaxHi);
		iLo = vbsl_u32(maskLo, indexLo, iLo);
		iHi = vbsl_u32(maskHi, indexHi, iHi);
		indexLo = vadd_u32(indexLo, four);
		indexHi = vadd_u32(indexHi, four);
	}

	for (; i + 4 <= count; i += 4)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
		float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);
		float32x2_t xy2 = vmul_f32(vget_low_f32(v2), vLo);
		float32x2_t xy3 = vmul_f32(vget_low_f32(v3), vLo);

		float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x2x2_t z1 = vtrn_f32(vget_high_f32(v2), vget_high_f32(v3));
		float32x2_t zLo = vmul_f32(z0.val[0], vHi);
		float32x2_t zHi = vmul_f32(z1.val[0], vHi);

		float32x2_t rLo = vpadd_f32(xy0, xy1);
		float32x2_t rHi = vpadd_f32(xy2, xy3);
		rLo = vadd_f32(rLo, zLo);
		rHi = vadd_f32(rHi, zHi);

		uint32x2_t maskLo = vcgt_f32(rLo, dotMaxLo);
		uint32x2_t maskHi = vcgt_f32(rHi, dotMaxHi);
		dotMaxLo = vbsl_f32(maskLo, rLo, dotMaxLo);
		dotMaxHi = vbsl_f32(maskHi, rHi, dotMaxHi);
		iLo = vbsl_u32(maskLo, indexLo, iLo);
		iHi = vbsl_u32(maskHi, indexHi, iHi);
		indexLo = vadd_u32(indexLo, four);
		indexHi = vadd_u32(indexHi, four);
	}

	switch (count & 3)
	{
		case 3:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);

			float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
			float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);
			float32x2_t xy2 = vmul_f32(vget_low_f32(v2), vLo);

			float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
			float32x2_t zLo = vmul_f32(z0.val[0], vHi);
			float32x2_t zHi = vmul_f32(vdup_lane_f32(vget_high_f32(v2), 0), vHi);

			float32x2_t rLo = vpadd_f32(xy0, xy1);
			float32x2_t rHi = vpadd_f32(xy2, xy2);
			rLo = vadd_f32(rLo, zLo);
			rHi = vadd_f32(rHi, zHi);

			uint32x2_t maskLo = vcgt_f32(rLo, dotMaxLo);
			uint32x2_t maskHi = vcgt_f32(rHi, dotMaxHi);
			dotMaxLo = vbsl_f32(maskLo, rLo, dotMaxLo);
			dotMaxHi = vbsl_f32(maskHi, rHi, dotMaxHi);
			iLo = vbsl_u32(maskLo, indexLo, iLo);
			iHi = vbsl_u32(maskHi, indexHi, iHi);
		}
		break;
		case 2:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);

			float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
			float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);

			float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
			float32x2_t zLo = vmul_f32(z0.val[0], vHi);

			float32x2_t rLo = vpadd_f32(xy0, xy1);
			rLo = vadd_f32(rLo, zLo);

			uint32x2_t maskLo = vcgt_f32(rLo, dotMaxLo);
			dotMaxLo = vbsl_f32(maskLo, rLo, dotMaxLo);
			iLo = vbsl_u32(maskLo, indexLo, iLo);
		}
		break;
		case 1:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
			float32x2_t z0 = vdup_lane_f32(vget_high_f32(v0), 0);
			float32x2_t zLo = vmul_f32(z0, vHi);
			float32x2_t rLo = vpadd_f32(xy0, xy0);
			rLo = vadd_f32(rLo, zLo);
			uint32x2_t maskLo = vcgt_f32(rLo, dotMaxLo);
			dotMaxLo = vbsl_f32(maskLo, rLo, dotMaxLo);
			iLo = vbsl_u32(maskLo, indexLo, iLo);
		}
		break;

		default:
			break;
	}

	// select best answer between hi and lo results
	uint32x2_t mask = vcgt_f32(dotMaxHi, dotMaxLo);
	dotMaxLo = vbsl_f32(mask, dotMaxHi, dotMaxLo);
	iLo = vbsl_u32(mask, iHi, iLo);

	// select best answer between even and odd results
	dotMaxHi = vdup_lane_f32(dotMaxLo, 1);
	iHi = vdup_lane_u32(iLo, 1);
	mask = vcgt_f32(dotMaxHi, dotMaxLo);
	dotMaxLo = vbsl_f32(mask, dotMaxHi, dotMaxLo);
	iLo = vbsl_u32(mask, iHi, iLo);

	*dotResult = vget_lane_f32(dotMaxLo, 0);
	return vget_lane_u32(iLo, 0);
}

long b3_maxdot_large_v1(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	float32x4_t vvec = vld1q_f32_aligned_postincrement(vec);
	float32x4_t vLo = vcombine_f32(vget_low_f32(vvec), vget_low_f32(vvec));
	float32x4_t vHi = vdupq_lane_f32(vget_high_f32(vvec), 0);
	const uint32x4_t four = (uint32x4_t){4, 4, 4, 4};
	uint32x4_t local_index = (uint32x4_t){0, 1, 2, 3};
	uint32x4_t index = (uint32x4_t){-1, -1, -1, -1};
	float32x4_t maxDot = (float32x4_t){-B3_INFINITY, -B3_INFINITY, -B3_INFINITY, -B3_INFINITY};

	unsigned long i = 0;
	for (; i + 8 <= count; i += 8)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		// the next two lines should resolve to a single vswp d, d
		float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
		float32x4_t xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v3));
		// the next two lines should resolve to a single vswp d, d
		float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x4_t z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));

		xy0 = vmulq_f32(xy0, vLo);
		xy1 = vmulq_f32(xy1, vLo);

		float32x4x2_t zb = vuzpq_f32(z0, z1);
		float32x4_t z = vmulq_f32(zb.val[0], vHi);
		float32x4x2_t xy = vuzpq_f32(xy0, xy1);
		float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
		x = vaddq_f32(x, z);

		uint32x4_t mask = vcgtq_f32(x, maxDot);
		maxDot = vbslq_f32(mask, x, maxDot);
		index = vbslq_u32(mask, local_index, index);
		local_index = vaddq_u32(local_index, four);

		v0 = vld1q_f32_aligned_postincrement(vv);
		v1 = vld1q_f32_aligned_postincrement(vv);
		v2 = vld1q_f32_aligned_postincrement(vv);
		v3 = vld1q_f32_aligned_postincrement(vv);

		// the next two lines should resolve to a single vswp d, d
		xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
		xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v3));
		// the next two lines should resolve to a single vswp d, d
		z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
		z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));

		xy0 = vmulq_f32(xy0, vLo);
		xy1 = vmulq_f32(xy1, vLo);

		zb = vuzpq_f32(z0, z1);
		z = vmulq_f32(zb.val[0], vHi);
		xy = vuzpq_f32(xy0, xy1);
		x = vaddq_f32(xy.val[0], xy.val[1]);
		x = vaddq_f32(x, z);

		mask = vcgtq_f32(x, maxDot);
		maxDot = vbslq_f32(mask, x, maxDot);
		index = vbslq_u32(mask, local_index, index);
		local_index = vaddq_u32(local_index, four);
	}

	for (; i + 4 <= count; i += 4)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		// the next two lines should resolve to a single vswp d, d
		float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
		float32x4_t xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v3));
		// the next two lines should resolve to a single vswp d, d
		float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x4_t z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));

		xy0 = vmulq_f32(xy0, vLo);
		xy1 = vmulq_f32(xy1, vLo);

		float32x4x2_t zb = vuzpq_f32(z0, z1);
		float32x4_t z = vmulq_f32(zb.val[0], vHi);
		float32x4x2_t xy = vuzpq_f32(xy0, xy1);
		float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
		x = vaddq_f32(x, z);

		uint32x4_t mask = vcgtq_f32(x, maxDot);
		maxDot = vbslq_f32(mask, x, maxDot);
		index = vbslq_u32(mask, local_index, index);
		local_index = vaddq_u32(local_index, four);
	}

	switch (count & 3)
	{
		case 3:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);

			// the next two lines should resolve to a single vswp d, d
			float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
			float32x4_t xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v2));
			// the next two lines should resolve to a single vswp d, d
			float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
			float32x4_t z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v2));

			xy0 = vmulq_f32(xy0, vLo);
			xy1 = vmulq_f32(xy1, vLo);

			float32x4x2_t zb = vuzpq_f32(z0, z1);
			float32x4_t z = vmulq_f32(zb.val[0], vHi);
			float32x4x2_t xy = vuzpq_f32(xy0, xy1);
			float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
			x = vaddq_f32(x, z);

			uint32x4_t mask = vcgtq_f32(x, maxDot);
			maxDot = vbslq_f32(mask, x, maxDot);
			index = vbslq_u32(mask, local_index, index);
			local_index = vaddq_u32(local_index, four);
		}
		break;

		case 2:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);

			// the next two lines should resolve to a single vswp d, d
			float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
			// the next two lines should resolve to a single vswp d, d
			float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));

			xy0 = vmulq_f32(xy0, vLo);

			float32x4x2_t zb = vuzpq_f32(z0, z0);
			float32x4_t z = vmulq_f32(zb.val[0], vHi);
			float32x4x2_t xy = vuzpq_f32(xy0, xy0);
			float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
			x = vaddq_f32(x, z);

			uint32x4_t mask = vcgtq_f32(x, maxDot);
			maxDot = vbslq_f32(mask, x, maxDot);
			index = vbslq_u32(mask, local_index, index);
			local_index = vaddq_u32(local_index, four);
		}
		break;

		case 1:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);

			// the next two lines should resolve to a single vswp d, d
			float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v0));
			// the next two lines should resolve to a single vswp d, d
			float32x4_t z = vdupq_lane_f32(vget_high_f32(v0), 0);

			xy0 = vmulq_f32(xy0, vLo);

			z = vmulq_f32(z, vHi);
			float32x4x2_t xy = vuzpq_f32(xy0, xy0);
			float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
			x = vaddq_f32(x, z);

			uint32x4_t mask = vcgtq_f32(x, maxDot);
			maxDot = vbslq_f32(mask, x, maxDot);
			index = vbslq_u32(mask, local_index, index);
			local_index = vaddq_u32(local_index, four);
		}
		break;

		default:
			break;
	}

	// select best answer between hi and lo results
	uint32x2_t mask = vcgt_f32(vget_high_f32(maxDot), vget_low_f32(maxDot));
	float32x2_t maxDot2 = vbsl_f32(mask, vget_high_f32(maxDot), vget_low_f32(maxDot));
	uint32x2_t index2 = vbsl_u32(mask, vget_high_u32(index), vget_low_u32(index));

	// select best answer between even and odd results
	float32x2_t maxDotO = vdup_lane_f32(maxDot2, 1);
	uint32x2_t indexHi = vdup_lane_u32(index2, 1);
	mask = vcgt_f32(maxDotO, maxDot2);
	maxDot2 = vbsl_f32(mask, maxDotO, maxDot2);
	index2 = vbsl_u32(mask, indexHi, index2);

	*dotResult = vget_lane_f32(maxDot2, 0);
	return vget_lane_u32(index2, 0);
}

long b3_mindot_large_v0(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	unsigned long i = 0;
	float32x4_t vvec = vld1q_f32_aligned_postincrement(vec);
	float32x2_t vLo = vget_low_f32(vvec);
	float32x2_t vHi = vdup_lane_f32(vget_high_f32(vvec), 0);
	float32x2_t dotMinLo = (float32x2_t){B3_INFINITY, B3_INFINITY};
	float32x2_t dotMinHi = (float32x2_t){B3_INFINITY, B3_INFINITY};
	uint32x2_t indexLo = (uint32x2_t){0, 1};
	uint32x2_t indexHi = (uint32x2_t){2, 3};
	uint32x2_t iLo = (uint32x2_t){-1, -1};
	uint32x2_t iHi = (uint32x2_t){-1, -1};
	const uint32x2_t four = (uint32x2_t){4, 4};

	for (; i + 8 <= count; i += 8)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
		float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);
		float32x2_t xy2 = vmul_f32(vget_low_f32(v2), vLo);
		float32x2_t xy3 = vmul_f32(vget_low_f32(v3), vLo);

		float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x2x2_t z1 = vtrn_f32(vget_high_f32(v2), vget_high_f32(v3));
		float32x2_t zLo = vmul_f32(z0.val[0], vHi);
		float32x2_t zHi = vmul_f32(z1.val[0], vHi);

		float32x2_t rLo = vpadd_f32(xy0, xy1);
		float32x2_t rHi = vpadd_f32(xy2, xy3);
		rLo = vadd_f32(rLo, zLo);
		rHi = vadd_f32(rHi, zHi);

		uint32x2_t maskLo = vclt_f32(rLo, dotMinLo);
		uint32x2_t maskHi = vclt_f32(rHi, dotMinHi);
		dotMinLo = vbsl_f32(maskLo, rLo, dotMinLo);
		dotMinHi = vbsl_f32(maskHi, rHi, dotMinHi);
		iLo = vbsl_u32(maskLo, indexLo, iLo);
		iHi = vbsl_u32(maskHi, indexHi, iHi);
		indexLo = vadd_u32(indexLo, four);
		indexHi = vadd_u32(indexHi, four);

		v0 = vld1q_f32_aligned_postincrement(vv);
		v1 = vld1q_f32_aligned_postincrement(vv);
		v2 = vld1q_f32_aligned_postincrement(vv);
		v3 = vld1q_f32_aligned_postincrement(vv);

		xy0 = vmul_f32(vget_low_f32(v0), vLo);
		xy1 = vmul_f32(vget_low_f32(v1), vLo);
		xy2 = vmul_f32(vget_low_f32(v2), vLo);
		xy3 = vmul_f32(vget_low_f32(v3), vLo);

		z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
		z1 = vtrn_f32(vget_high_f32(v2), vget_high_f32(v3));
		zLo = vmul_f32(z0.val[0], vHi);
		zHi = vmul_f32(z1.val[0], vHi);

		rLo = vpadd_f32(xy0, xy1);
		rHi = vpadd_f32(xy2, xy3);
		rLo = vadd_f32(rLo, zLo);
		rHi = vadd_f32(rHi, zHi);

		maskLo = vclt_f32(rLo, dotMinLo);
		maskHi = vclt_f32(rHi, dotMinHi);
		dotMinLo = vbsl_f32(maskLo, rLo, dotMinLo);
		dotMinHi = vbsl_f32(maskHi, rHi, dotMinHi);
		iLo = vbsl_u32(maskLo, indexLo, iLo);
		iHi = vbsl_u32(maskHi, indexHi, iHi);
		indexLo = vadd_u32(indexLo, four);
		indexHi = vadd_u32(indexHi, four);
	}

	for (; i + 4 <= count; i += 4)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
		float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);
		float32x2_t xy2 = vmul_f32(vget_low_f32(v2), vLo);
		float32x2_t xy3 = vmul_f32(vget_low_f32(v3), vLo);

		float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x2x2_t z1 = vtrn_f32(vget_high_f32(v2), vget_high_f32(v3));
		float32x2_t zLo = vmul_f32(z0.val[0], vHi);
		float32x2_t zHi = vmul_f32(z1.val[0], vHi);

		float32x2_t rLo = vpadd_f32(xy0, xy1);
		float32x2_t rHi = vpadd_f32(xy2, xy3);
		rLo = vadd_f32(rLo, zLo);
		rHi = vadd_f32(rHi, zHi);

		uint32x2_t maskLo = vclt_f32(rLo, dotMinLo);
		uint32x2_t maskHi = vclt_f32(rHi, dotMinHi);
		dotMinLo = vbsl_f32(maskLo, rLo, dotMinLo);
		dotMinHi = vbsl_f32(maskHi, rHi, dotMinHi);
		iLo = vbsl_u32(maskLo, indexLo, iLo);
		iHi = vbsl_u32(maskHi, indexHi, iHi);
		indexLo = vadd_u32(indexLo, four);
		indexHi = vadd_u32(indexHi, four);
	}
	switch (count & 3)
	{
		case 3:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);

			float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
			float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);
			float32x2_t xy2 = vmul_f32(vget_low_f32(v2), vLo);

			float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
			float32x2_t zLo = vmul_f32(z0.val[0], vHi);
			float32x2_t zHi = vmul_f32(vdup_lane_f32(vget_high_f32(v2), 0), vHi);

			float32x2_t rLo = vpadd_f32(xy0, xy1);
			float32x2_t rHi = vpadd_f32(xy2, xy2);
			rLo = vadd_f32(rLo, zLo);
			rHi = vadd_f32(rHi, zHi);

			uint32x2_t maskLo = vclt_f32(rLo, dotMinLo);
			uint32x2_t maskHi = vclt_f32(rHi, dotMinHi);
			dotMinLo = vbsl_f32(maskLo, rLo, dotMinLo);
			dotMinHi = vbsl_f32(maskHi, rHi, dotMinHi);
			iLo = vbsl_u32(maskLo, indexLo, iLo);
			iHi = vbsl_u32(maskHi, indexHi, iHi);
		}
		break;
		case 2:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);

			float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
			float32x2_t xy1 = vmul_f32(vget_low_f32(v1), vLo);

			float32x2x2_t z0 = vtrn_f32(vget_high_f32(v0), vget_high_f32(v1));
			float32x2_t zLo = vmul_f32(z0.val[0], vHi);

			float32x2_t rLo = vpadd_f32(xy0, xy1);
			rLo = vadd_f32(rLo, zLo);

			uint32x2_t maskLo = vclt_f32(rLo, dotMinLo);
			dotMinLo = vbsl_f32(maskLo, rLo, dotMinLo);
			iLo = vbsl_u32(maskLo, indexLo, iLo);
		}
		break;
		case 1:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x2_t xy0 = vmul_f32(vget_low_f32(v0), vLo);
			float32x2_t z0 = vdup_lane_f32(vget_high_f32(v0), 0);
			float32x2_t zLo = vmul_f32(z0, vHi);
			float32x2_t rLo = vpadd_f32(xy0, xy0);
			rLo = vadd_f32(rLo, zLo);
			uint32x2_t maskLo = vclt_f32(rLo, dotMinLo);
			dotMinLo = vbsl_f32(maskLo, rLo, dotMinLo);
			iLo = vbsl_u32(maskLo, indexLo, iLo);
		}
		break;

		default:
			break;
	}

	// select best answer between hi and lo results
	uint32x2_t mask = vclt_f32(dotMinHi, dotMinLo);
	dotMinLo = vbsl_f32(mask, dotMinHi, dotMinLo);
	iLo = vbsl_u32(mask, iHi, iLo);

	// select best answer between even and odd results
	dotMinHi = vdup_lane_f32(dotMinLo, 1);
	iHi = vdup_lane_u32(iLo, 1);
	mask = vclt_f32(dotMinHi, dotMinLo);
	dotMinLo = vbsl_f32(mask, dotMinHi, dotMinLo);
	iLo = vbsl_u32(mask, iHi, iLo);

	*dotResult = vget_lane_f32(dotMinLo, 0);
	return vget_lane_u32(iLo, 0);
}

long b3_mindot_large_v1(const float *vv, const float *vec, unsigned long count, float *dotResult)
{
	float32x4_t vvec = vld1q_f32_aligned_postincrement(vec);
	float32x4_t vLo = vcombine_f32(vget_low_f32(vvec), vget_low_f32(vvec));
	float32x4_t vHi = vdupq_lane_f32(vget_high_f32(vvec), 0);
	const uint32x4_t four = (uint32x4_t){4, 4, 4, 4};
	uint32x4_t local_index = (uint32x4_t){0, 1, 2, 3};
	uint32x4_t index = (uint32x4_t){-1, -1, -1, -1};
	float32x4_t minDot = (float32x4_t){B3_INFINITY, B3_INFINITY, B3_INFINITY, B3_INFINITY};

	unsigned long i = 0;
	for (; i + 8 <= count; i += 8)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		// the next two lines should resolve to a single vswp d, d
		float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
		float32x4_t xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v3));
		// the next two lines should resolve to a single vswp d, d
		float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x4_t z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));

		xy0 = vmulq_f32(xy0, vLo);
		xy1 = vmulq_f32(xy1, vLo);

		float32x4x2_t zb = vuzpq_f32(z0, z1);
		float32x4_t z = vmulq_f32(zb.val[0], vHi);
		float32x4x2_t xy = vuzpq_f32(xy0, xy1);
		float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
		x = vaddq_f32(x, z);

		uint32x4_t mask = vcltq_f32(x, minDot);
		minDot = vbslq_f32(mask, x, minDot);
		index = vbslq_u32(mask, local_index, index);
		local_index = vaddq_u32(local_index, four);

		v0 = vld1q_f32_aligned_postincrement(vv);
		v1 = vld1q_f32_aligned_postincrement(vv);
		v2 = vld1q_f32_aligned_postincrement(vv);
		v3 = vld1q_f32_aligned_postincrement(vv);

		// the next two lines should resolve to a single vswp d, d
		xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
		xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v3));
		// the next two lines should resolve to a single vswp d, d
		z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
		z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));

		xy0 = vmulq_f32(xy0, vLo);
		xy1 = vmulq_f32(xy1, vLo);

		zb = vuzpq_f32(z0, z1);
		z = vmulq_f32(zb.val[0], vHi);
		xy = vuzpq_f32(xy0, xy1);
		x = vaddq_f32(xy.val[0], xy.val[1]);
		x = vaddq_f32(x, z);

		mask = vcltq_f32(x, minDot);
		minDot = vbslq_f32(mask, x, minDot);
		index = vbslq_u32(mask, local_index, index);
		local_index = vaddq_u32(local_index, four);
	}

	for (; i + 4 <= count; i += 4)
	{
		float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);
		float32x4_t v3 = vld1q_f32_aligned_postincrement(vv);

		// the next two lines should resolve to a single vswp d, d
		float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
		float32x4_t xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v3));
		// the next two lines should resolve to a single vswp d, d
		float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
		float32x4_t z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));

		xy0 = vmulq_f32(xy0, vLo);
		xy1 = vmulq_f32(xy1, vLo);

		float32x4x2_t zb = vuzpq_f32(z0, z1);
		float32x4_t z = vmulq_f32(zb.val[0], vHi);
		float32x4x2_t xy = vuzpq_f32(xy0, xy1);
		float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
		x = vaddq_f32(x, z);

		uint32x4_t mask = vcltq_f32(x, minDot);
		minDot = vbslq_f32(mask, x, minDot);
		index = vbslq_u32(mask, local_index, index);
		local_index = vaddq_u32(local_index, four);
	}

	switch (count & 3)
	{
		case 3:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v2 = vld1q_f32_aligned_postincrement(vv);

			// the next two lines should resolve to a single vswp d, d
			float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
			float32x4_t xy1 = vcombine_f32(vget_low_f32(v2), vget_low_f32(v2));
			// the next two lines should resolve to a single vswp d, d
			float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
			float32x4_t z1 = vcombine_f32(vget_high_f32(v2), vget_high_f32(v2));

			xy0 = vmulq_f32(xy0, vLo);
			xy1 = vmulq_f32(xy1, vLo);

			float32x4x2_t zb = vuzpq_f32(z0, z1);
			float32x4_t z = vmulq_f32(zb.val[0], vHi);
			float32x4x2_t xy = vuzpq_f32(xy0, xy1);
			float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
			x = vaddq_f32(x, z);

			uint32x4_t mask = vcltq_f32(x, minDot);
			minDot = vbslq_f32(mask, x, minDot);
			index = vbslq_u32(mask, local_index, index);
			local_index = vaddq_u32(local_index, four);
		}
		break;

		case 2:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);
			float32x4_t v1 = vld1q_f32_aligned_postincrement(vv);

			// the next two lines should resolve to a single vswp d, d
			float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
			// the next two lines should resolve to a single vswp d, d
			float32x4_t z0 = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));

			xy0 = vmulq_f32(xy0, vLo);

			float32x4x2_t zb = vuzpq_f32(z0, z0);
			float32x4_t z = vmulq_f32(zb.val[0], vHi);
			float32x4x2_t xy = vuzpq_f32(xy0, xy0);
			float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
			x = vaddq_f32(x, z);

			uint32x4_t mask = vcltq_f32(x, minDot);
			minDot = vbslq_f32(mask, x, minDot);
			index = vbslq_u32(mask, local_index, index);
			local_index = vaddq_u32(local_index, four);
		}
		break;

		case 1:
		{
			float32x4_t v0 = vld1q_f32_aligned_postincrement(vv);

			// the next two lines should resolve to a single vswp d, d
			float32x4_t xy0 = vcombine_f32(vget_low_f32(v0), vget_low_f32(v0));
			// the next two lines should resolve to a single vswp d, d
			float32x4_t z = vdupq_lane_f32(vget_high_f32(v0), 0);

			xy0 = vmulq_f32(xy0, vLo);

			z = vmulq_f32(z, vHi);
			float32x4x2_t xy = vuzpq_f32(xy0, xy0);
			float32x4_t x = vaddq_f32(xy.val[0], xy.val[1]);
			x = vaddq_f32(x, z);

			uint32x4_t mask = vcltq_f32(x, minDot);
			minDot = vbslq_f32(mask, x, minDot);
			index = vbslq_u32(mask, local_index, index);
			local_index = vaddq_u32(local_index, four);
		}
		break;

		default:
			break;
	}

	// select best answer between hi and lo results
	uint32x2_t mask = vclt_f32(vget_high_f32(minDot), vget_low_f32(minDot));
	float32x2_t minDot2 = vbsl_f32(mask, vget_high_f32(minDot), vget_low_f32(minDot));
	uint32x2_t index2 = vbsl_u32(mask, vget_high_u32(index), vget_low_u32(index));

	// select best answer between even and odd results
	float32x2_t minDotO = vdup_lane_f32(minDot2, 1);
	uint32x2_t indexHi = vdup_lane_u32(index2, 1);
	mask = vclt_f32(minDotO, minDot2);
	minDot2 = vbsl_f32(mask, minDotO, minDot2);
	index2 = vbsl_u32(mask, indexHi, index2);

	*dotResult = vget_lane_f32(minDot2, 0);
	return vget_lane_u32(index2, 0);
}

#else
#error Unhandled __APPLE__ arch
#endif

#endif /* __APPLE__ */
