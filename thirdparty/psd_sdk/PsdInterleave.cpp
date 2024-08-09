// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdInterleave.h"

#include "PsdUnionCast.h"

#if !defined(PSD_USE_SSE)
	#if defined(_M_IX86) || defined(_M_X64)
		#define PSD_USE_SSE 1
	#else
		#define PSD_USE_SSE 0
	#endif
#endif

#if PSD_USE_SSE
	#include <emmintrin.h>
#endif


PSD_NAMESPACE_BEGIN

#if PSD_USE_SSE
// splats a single 8-bit, 16-bit or 32-bit value into a SSE2 register
namespace
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	__m128i SplatValue(T value);


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <>
	__m128i SplatValue<uint8_t>(uint8_t value)
	{
		return _mm_set1_epi8(util::union_cast<char>(value));
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <>
	__m128i SplatValue<uint16_t>(uint16_t value)
	{
		return _mm_set1_epi16(util::union_cast<short>(value));
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <>
	__m128i SplatValue<float32_t>(float32_t value)
	{
		return _mm_castps_si128(_mm_set_ps1(value));
	}
}


// interleaves either 8-bit, 16-bit, 32-bit or 64-bit values from two SSE2 registers
namespace
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <unsigned int N>
	__m128i InterleaveLo(__m128i a, __m128i b);

	template <> __m128i InterleaveLo<1>(__m128i a, __m128i b) { return _mm_unpacklo_epi8(a, b); }
	template <> __m128i InterleaveLo<2>(__m128i a, __m128i b) { return _mm_unpacklo_epi16(a, b); }
	template <> __m128i InterleaveLo<4>(__m128i a, __m128i b) { return _mm_unpacklo_epi32(a, b); }
	template <> __m128i InterleaveLo<8>(__m128i a, __m128i b) { return _mm_unpacklo_epi64(a, b); }


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <unsigned int N>
	__m128i InterleaveHi(__m128i a, __m128i b);

	template <> __m128i InterleaveHi<1>(__m128i a, __m128i b) { return _mm_unpackhi_epi8(a, b); }
	template <> __m128i InterleaveHi<2>(__m128i a, __m128i b) { return _mm_unpackhi_epi16(a, b); }
	template <> __m128i InterleaveHi<4>(__m128i a, __m128i b) { return _mm_unpackhi_epi32(a, b); }
	template <> __m128i InterleaveHi<8>(__m128i a, __m128i b) { return _mm_unpackhi_epi64(a, b); }
}
#endif


namespace imageUtil
{
#if PSD_USE_SSE
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	unsigned int InterleaveBlocks(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, T alpha, T* PSD_RESTRICT dest, unsigned int width, unsigned int height, unsigned int blockSize)
	{
		const unsigned int pixelCount = width*height;
		const unsigned int blockCount = pixelCount / blockSize;
		const __m128i va = SplatValue(alpha);

		for (unsigned int i=0; i < blockCount; ++i, srcR += blockSize, srcG += blockSize, srcB += blockSize, dest += blockSize*4u)
		{
			// load pixels from R, G, B
			const __m128i vr = _mm_load_si128(reinterpret_cast<const __m128i*>(srcR));
			const __m128i vg = _mm_load_si128(reinterpret_cast<const __m128i*>(srcG));
			const __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(srcB));

			// interleave R and G
			const __m128i rg_interleaved_lo = InterleaveLo<sizeof(T)>(vr, vg);
			const __m128i rg_interleaved_hi = InterleaveHi<sizeof(T)>(vr, vg);

			// interleave B and A
			const __m128i ba_interleaved_lo = InterleaveLo<sizeof(T)>(vb, va);
			const __m128i ba_interleaved_hi = InterleaveHi<sizeof(T)>(vb, va);

			// interleave RG and BA
			const __m128i rgba_1 = InterleaveLo<sizeof(T)*2>(rg_interleaved_lo, ba_interleaved_lo);
			const __m128i rgba_2 = InterleaveHi<sizeof(T)*2>(rg_interleaved_lo, ba_interleaved_lo);
			const __m128i rgba_3 = InterleaveLo<sizeof(T)*2>(rg_interleaved_hi, ba_interleaved_hi);
			const __m128i rgba_4 = InterleaveHi<sizeof(T)*2>(rg_interleaved_hi, ba_interleaved_hi);

			// store to memory non-temporal, bypassing cache
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest), rgba_1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest + blockSize*1u), rgba_2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest + blockSize*2u), rgba_3);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest + blockSize*3u), rgba_4);
		}

		return blockCount;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	unsigned int InterleaveBlocks(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, const T* PSD_RESTRICT srcA, T* PSD_RESTRICT dest, unsigned int width, unsigned int height, unsigned int blockSize)
	{
		const unsigned int pixelCount = width*height;
		const unsigned int blockCount = pixelCount / blockSize;

		for (unsigned int i=0; i < blockCount; ++i, srcR += blockSize, srcG += blockSize, srcB += blockSize, srcA += blockSize, dest += blockSize*4u)
		{
			// load pixels from R, G, B, and A
			const __m128i vr = _mm_load_si128(reinterpret_cast<const __m128i*>(srcR));
			const __m128i vg = _mm_load_si128(reinterpret_cast<const __m128i*>(srcG));
			const __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(srcB));
			const __m128i va = _mm_load_si128(reinterpret_cast<const __m128i*>(srcA));

			// interleave R and G
			const __m128i rg_interleaved_lo = InterleaveLo<sizeof(T)>(vr, vg);
			const __m128i rg_interleaved_hi = InterleaveHi<sizeof(T)>(vr, vg);

			// interleave B and A
			const __m128i ba_interleaved_lo = InterleaveLo<sizeof(T)>(vb, va);
			const __m128i ba_interleaved_hi = InterleaveHi<sizeof(T)>(vb, va);

			// interleave RG and BA
			const __m128i rgba_1 = InterleaveLo<sizeof(T)*2>(rg_interleaved_lo, ba_interleaved_lo);
			const __m128i rgba_2 = InterleaveHi<sizeof(T)*2>(rg_interleaved_lo, ba_interleaved_lo);
			const __m128i rgba_3 = InterleaveLo<sizeof(T)*2>(rg_interleaved_hi, ba_interleaved_hi);
			const __m128i rgba_4 = InterleaveHi<sizeof(T)*2>(rg_interleaved_hi, ba_interleaved_hi);

			// store to memory non-temporal, bypassing cache
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest), rgba_1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest + blockSize*1u), rgba_2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest + blockSize*2u), rgba_3);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dest + blockSize*3u), rgba_4);
		}

		return blockCount;
	}
#endif


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void CopyRemainingPixels(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, T alpha, T* PSD_RESTRICT dest, unsigned int count)
	{
		for (unsigned int i=0; i < count; ++i)
		{
			const T r = srcR[i];
			const T g = srcG[i];
			const T b = srcB[i];

			dest[0] = r;
			dest[1] = g;
			dest[2] = b;
			dest[3] = alpha;
			dest += 4;
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void CopyRemainingPixels(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, const T* PSD_RESTRICT srcA, T* PSD_RESTRICT dest, unsigned int count)
	{
		for (unsigned int i=0; i < count; ++i)
		{
			const T r = srcR[i];
			const T g = srcG[i];
			const T b = srcB[i];
			const T a = srcA[i];

			dest[0] = r;
			dest[1] = g;
			dest[2] = b;
			dest[3] = a;
			dest += 4;
		}
	}


#if PSD_USE_SSE
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void InterleaveRGB(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, T alpha, T* PSD_RESTRICT dest, unsigned int width, unsigned int height, unsigned int blockSize)
	{
		// do blocks first, and then copy remaining pixels
		const unsigned int blockCount = InterleaveBlocks(srcR, srcG, srcB, alpha, dest, width, height, blockSize);
		const unsigned int remaining = width*height - blockCount*blockSize;
		CopyRemainingPixels(srcR + blockCount*blockSize, srcG + blockCount*blockSize, srcB + blockCount*blockSize, alpha, dest + blockCount*blockSize*4u, remaining);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void InterleaveRGBA(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, const T* PSD_RESTRICT srcA, T* PSD_RESTRICT dest, unsigned int width, unsigned int height, unsigned int blockSize)
	{
		// do blocks first, and then copy remaining pixels
		const unsigned int blockCount = InterleaveBlocks(srcR, srcG, srcB, srcA, dest, width, height, blockSize);
		const unsigned int remaining = width*height - blockCount*blockSize;
		CopyRemainingPixels(srcR + blockCount*blockSize, srcG + blockCount*blockSize, srcB + blockCount*blockSize, srcA + blockCount*blockSize, dest + blockCount*blockSize*4u, remaining);
	}
#else
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void InterleaveRGB(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, T alpha, T* PSD_RESTRICT dest, unsigned int width, unsigned int height, unsigned int /* blockSize */)
	{
		// copy pixels
		const unsigned int count = width * height;
		CopyRemainingPixels(srcR, srcG, srcB, alpha, dest, count);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void InterleaveRGBA(const T* PSD_RESTRICT srcR, const T* PSD_RESTRICT srcG, const T* PSD_RESTRICT srcB, const T* PSD_RESTRICT srcA, T* PSD_RESTRICT dest, unsigned int width, unsigned int height, unsigned int /* blockSize */)
	{
		// copy pixels
		const unsigned int count = width * height;
		CopyRemainingPixels(srcR, srcG, srcB, srcA, dest, count);
	}
#endif


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void InterleaveRGB(const uint8_t* PSD_RESTRICT srcR, const uint8_t* PSD_RESTRICT srcG, const uint8_t* PSD_RESTRICT srcB, uint8_t alpha, uint8_t* PSD_RESTRICT dest, unsigned int width, unsigned int height)
	{
		InterleaveRGB(srcR, srcG, srcB, alpha, dest, width, height, 16u);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void InterleaveRGBA(const uint8_t* PSD_RESTRICT srcR, const uint8_t* PSD_RESTRICT srcG, const uint8_t* PSD_RESTRICT srcB, const uint8_t* PSD_RESTRICT srcA, uint8_t* PSD_RESTRICT dest, unsigned int width, unsigned int height)
	{
		InterleaveRGBA(srcR, srcG, srcB, srcA, dest, width, height, 16u);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void InterleaveRGB(const uint16_t* PSD_RESTRICT srcR, const uint16_t* PSD_RESTRICT srcG, const uint16_t* PSD_RESTRICT srcB, uint16_t alpha, uint16_t* PSD_RESTRICT dest, unsigned int width, unsigned int height)
	{
		InterleaveRGB(srcR, srcG, srcB, alpha, dest, width, height, 8u);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void InterleaveRGBA(const uint16_t* PSD_RESTRICT srcR, const uint16_t* PSD_RESTRICT srcG, const uint16_t* PSD_RESTRICT srcB, const uint16_t* PSD_RESTRICT srcA, uint16_t* PSD_RESTRICT dest, unsigned int width, unsigned int height)
	{
		InterleaveRGBA(srcR, srcG, srcB, srcA, dest, width, height, 8u);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void InterleaveRGB(const float32_t* PSD_RESTRICT srcR, const float32_t* PSD_RESTRICT srcG, const float32_t* PSD_RESTRICT srcB, float32_t alpha, float32_t* PSD_RESTRICT dest, unsigned int width, unsigned int height)
	{
		InterleaveRGB(srcR, srcG, srcB, alpha, dest, width, height, 4u);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void InterleaveRGBA(const float32_t* PSD_RESTRICT srcR, const float32_t* PSD_RESTRICT srcG, const float32_t* PSD_RESTRICT srcB, const float32_t* PSD_RESTRICT srcA, float32_t* PSD_RESTRICT dest, unsigned int width, unsigned int height)
	{
		InterleaveRGBA(srcR, srcG, srcB, srcA, dest, width, height, 4u);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void DeinterleaveRGB(const T* PSD_RESTRICT rgb, T* PSD_RESTRICT destR, T* PSD_RESTRICT destG, T* PSD_RESTRICT destB, unsigned int count)
	{
		for (unsigned int i = 0u; i < count; ++i)
		{
			const T r = rgb[i * 3 + 0];
			const T g = rgb[i * 3 + 1];
			const T b = rgb[i * 3 + 2];

			destR[i] = r;
			destG[i] = g;
			destB[i] = b;
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void DeinterleaveRGBA(const T* PSD_RESTRICT rgba, T* PSD_RESTRICT destR, T* PSD_RESTRICT destG, T* PSD_RESTRICT destB, T* PSD_RESTRICT destA, unsigned int count)
	{
		for (unsigned int i = 0u; i < count; ++i)
		{
			const T r = rgba[i * 4 + 0];
			const T g = rgba[i * 4 + 1];
			const T b = rgba[i * 4 + 2];
			const T a = rgba[i * 4 + 3];

			destR[i] = r;
			destG[i] = g;
			destB[i] = b;
			destA[i] = a;
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void DeinterleaveRGB(const uint8_t* PSD_RESTRICT rgb, uint8_t* PSD_RESTRICT destR, uint8_t* PSD_RESTRICT destG, uint8_t* PSD_RESTRICT destB, unsigned int width, unsigned int height)
	{
		const unsigned int count = width*height;
		DeinterleaveRGB(rgb, destR, destG, destB, count);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void DeinterleaveRGBA(const uint8_t* PSD_RESTRICT rgba, uint8_t* PSD_RESTRICT destR, uint8_t* PSD_RESTRICT destG, uint8_t* PSD_RESTRICT destB, uint8_t* PSD_RESTRICT destA, unsigned int width, unsigned int height)
	{
		const unsigned int count = width*height;
		DeinterleaveRGBA(rgba, destR, destG, destB, destA, count);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void DeinterleaveRGB(const uint16_t* PSD_RESTRICT rgb, uint16_t* PSD_RESTRICT destR, uint16_t* PSD_RESTRICT destG, uint16_t* PSD_RESTRICT destB, unsigned int width, unsigned int height)
	{
		const unsigned int count = width*height;
		DeinterleaveRGB(rgb, destR, destG, destB, count);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void DeinterleaveRGBA(const uint16_t* PSD_RESTRICT rgba, uint16_t* PSD_RESTRICT destR, uint16_t* PSD_RESTRICT destG, uint16_t* PSD_RESTRICT destB, uint16_t* PSD_RESTRICT destA, unsigned int width, unsigned int height)
	{
		const unsigned int count = width*height;
		DeinterleaveRGBA(rgba, destR, destG, destB, destA, count);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void DeinterleaveRGB(const float32_t* PSD_RESTRICT rgb, float32_t* PSD_RESTRICT destR, float32_t* PSD_RESTRICT destG, float32_t* PSD_RESTRICT destB, unsigned int width, unsigned int height)
	{
		const unsigned int count = width*height;
		DeinterleaveRGB(rgb, destR, destG, destB, count);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void DeinterleaveRGBA(const float32_t* PSD_RESTRICT rgba, float32_t* PSD_RESTRICT destR, float32_t* PSD_RESTRICT destG, float32_t* PSD_RESTRICT destB, float32_t* PSD_RESTRICT destA, unsigned int width, unsigned int height)
	{
		const unsigned int count = width*height;
		DeinterleaveRGBA(rgba, destR, destG, destB, destA, count);
	}
}

PSD_NAMESPACE_END
