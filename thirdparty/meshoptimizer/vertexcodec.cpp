// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

// The block below auto-detects SIMD ISA that can be used on the target platform
#ifndef MESHOPTIMIZER_NO_SIMD

// The SIMD implementation requires SSSE3, which can be enabled unconditionally through compiler settings
#if defined(__AVX__) || defined(__SSSE3__)
#define SIMD_SSE
#endif

// An experimental implementation using AVX512 instructions; it's only enabled when AVX512 is enabled through compiler settings
#if defined(__AVX512VBMI2__) && defined(__AVX512VBMI__) && defined(__AVX512VL__) && defined(__POPCNT__)
#undef SIMD_SSE
#define SIMD_AVX
#endif

// MSVC supports compiling SSSE3 code regardless of compile options; we use a cpuid-based scalar fallback
#if !defined(SIMD_SSE) && !defined(SIMD_AVX) && defined(_MSC_VER) && !defined(__clang__) && (defined(_M_IX86) || defined(_M_X64))
#define SIMD_SSE
#define SIMD_FALLBACK
#endif

// GCC 4.9+ and clang 3.8+ support targeting SIMD ISA from individual functions; we use a cpuid-based scalar fallback
#if !defined(SIMD_SSE) && !defined(SIMD_AVX) && ((defined(__clang__) && __clang_major__ * 100 + __clang_minor__ >= 308) || (defined(__GNUC__) && __GNUC__ * 100 + __GNUC_MINOR__ >= 409)) && (defined(__i386__) || defined(__x86_64__))
#define SIMD_SSE
#define SIMD_FALLBACK
#define SIMD_TARGET __attribute__((target("ssse3")))
#endif

// GCC/clang define these when NEON support is available
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define SIMD_NEON
#endif

// On MSVC, we assume that ARM builds always target NEON-capable devices
#if !defined(SIMD_NEON) && defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
#define SIMD_NEON
#endif

// When targeting Wasm SIMD we can't use runtime cpuid checks so we unconditionally enable SIMD
#if defined(__wasm_simd128__)
#define SIMD_WASM
// Prevent compiling other variant when wasm simd compilation is active
#undef SIMD_NEON
#undef SIMD_SSE
#undef SIMD_AVX
#endif

#ifndef SIMD_TARGET
#define SIMD_TARGET
#endif

// When targeting AArch64/x64, optimize for latency to allow decoding of individual 16-byte groups to overlap
// We don't do this for 32-bit systems because we need 64-bit math for this and this will hurt in-order CPUs
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
#define SIMD_LATENCYOPT
#endif

#endif // !MESHOPTIMIZER_NO_SIMD

#ifdef SIMD_SSE
#include <tmmintrin.h>
#endif

#if defined(SIMD_SSE) && defined(SIMD_FALLBACK)
#ifdef _MSC_VER
#include <intrin.h> // __cpuid
#else
#include <cpuid.h> // __cpuid
#endif
#endif

#ifdef SIMD_AVX
#include <immintrin.h>
#endif

#ifdef SIMD_NEON
#if defined(_MSC_VER) && defined(_M_ARM64)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#endif

#ifdef SIMD_WASM
#include <wasm_simd128.h>
#endif

#ifdef SIMD_WASM
#define wasmx_splat_v32x4(v, i) wasm_i32x4_shuffle(v, v, i, i, i, i)
#define wasmx_unpacklo_v8x16(a, b) wasm_i8x16_shuffle(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23)
#define wasmx_unpackhi_v8x16(a, b) wasm_i8x16_shuffle(a, b, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31)
#define wasmx_unpacklo_v16x8(a, b) wasm_i16x8_shuffle(a, b, 0, 8, 1, 9, 2, 10, 3, 11)
#define wasmx_unpackhi_v16x8(a, b) wasm_i16x8_shuffle(a, b, 4, 12, 5, 13, 6, 14, 7, 15)
#define wasmx_unpacklo_v64x2(a, b) wasm_i64x2_shuffle(a, b, 0, 2)
#define wasmx_unpackhi_v64x2(a, b) wasm_i64x2_shuffle(a, b, 1, 3)
#endif

namespace meshopt
{

const unsigned char kVertexHeader = 0xa0;

static int gEncodeVertexVersion = 0;

const size_t kVertexBlockSizeBytes = 8192;
const size_t kVertexBlockMaxSize = 256;
const size_t kByteGroupSize = 16;
const size_t kByteGroupDecodeLimit = 24;
const size_t kTailMaxSize = 32;

static size_t getVertexBlockSize(size_t vertex_size)
{
	// make sure the entire block fits into the scratch buffer
	size_t result = kVertexBlockSizeBytes / vertex_size;

	// align to byte group size; we encode each byte as a byte group
	// if vertex block is misaligned, it results in wasted bytes, so just truncate the block size
	result &= ~(kByteGroupSize - 1);

	return (result < kVertexBlockMaxSize) ? result : kVertexBlockMaxSize;
}

inline unsigned char zigzag8(unsigned char v)
{
	return ((signed char)(v) >> 7) ^ (v << 1);
}

inline unsigned char unzigzag8(unsigned char v)
{
	return -(v & 1) ^ (v >> 1);
}

static bool encodeBytesGroupZero(const unsigned char* buffer)
{
	for (size_t i = 0; i < kByteGroupSize; ++i)
		if (buffer[i])
			return false;

	return true;
}

static size_t encodeBytesGroupMeasure(const unsigned char* buffer, int bits)
{
	assert(bits >= 1 && bits <= 8);

	if (bits == 1)
		return encodeBytesGroupZero(buffer) ? 0 : size_t(-1);

	if (bits == 8)
		return kByteGroupSize;

	size_t result = kByteGroupSize * bits / 8;

	unsigned char sentinel = (1 << bits) - 1;

	for (size_t i = 0; i < kByteGroupSize; ++i)
		result += buffer[i] >= sentinel;

	return result;
}

static unsigned char* encodeBytesGroup(unsigned char* data, const unsigned char* buffer, int bits)
{
	assert(bits >= 1 && bits <= 8);

	if (bits == 1)
		return data;

	if (bits == 8)
	{
		memcpy(data, buffer, kByteGroupSize);
		return data + kByteGroupSize;
	}

	size_t byte_size = 8 / bits;
	assert(kByteGroupSize % byte_size == 0);

	// fixed portion: bits bits for each value
	// variable portion: full byte for each out-of-range value (using 1...1 as sentinel)
	unsigned char sentinel = (1 << bits) - 1;

	for (size_t i = 0; i < kByteGroupSize; i += byte_size)
	{
		unsigned char byte = 0;

		for (size_t k = 0; k < byte_size; ++k)
		{
			unsigned char enc = (buffer[i + k] >= sentinel) ? sentinel : buffer[i + k];

			byte <<= bits;
			byte |= enc;
		}

		*data++ = byte;
	}

	for (size_t i = 0; i < kByteGroupSize; ++i)
	{
		if (buffer[i] >= sentinel)
		{
			*data++ = buffer[i];
		}
	}

	return data;
}

static unsigned char* encodeBytes(unsigned char* data, unsigned char* data_end, const unsigned char* buffer, size_t buffer_size)
{
	assert(buffer_size % kByteGroupSize == 0);

	unsigned char* header = data;

	// round number of groups to 4 to get number of header bytes
	size_t header_size = (buffer_size / kByteGroupSize + 3) / 4;

	if (size_t(data_end - data) < header_size)
		return NULL;

	data += header_size;

	memset(header, 0, header_size);

	for (size_t i = 0; i < buffer_size; i += kByteGroupSize)
	{
		if (size_t(data_end - data) < kByteGroupDecodeLimit)
			return NULL;

		int best_bits = 8;
		size_t best_size = encodeBytesGroupMeasure(buffer + i, 8);

		for (int bits = 1; bits < 8; bits *= 2)
		{
			size_t size = encodeBytesGroupMeasure(buffer + i, bits);

			if (size < best_size)
			{
				best_bits = bits;
				best_size = size;
			}
		}

		int bitslog2 = (best_bits == 1) ? 0 : (best_bits == 2) ? 1 : (best_bits == 4) ? 2 : 3;
		assert((1 << bitslog2) == best_bits);

		size_t header_offset = i / kByteGroupSize;

		header[header_offset / 4] |= bitslog2 << ((header_offset % 4) * 2);

		unsigned char* next = encodeBytesGroup(data, buffer + i, best_bits);

		assert(data + best_size == next);
		data = next;
	}

	return data;
}

static unsigned char* encodeVertexBlock(unsigned char* data, unsigned char* data_end, const unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, unsigned char last_vertex[256])
{
	assert(vertex_count > 0 && vertex_count <= kVertexBlockMaxSize);

	unsigned char buffer[kVertexBlockMaxSize];
	assert(sizeof(buffer) % kByteGroupSize == 0);

	// we sometimes encode elements we didn't fill when rounding to kByteGroupSize
	memset(buffer, 0, sizeof(buffer));

	for (size_t k = 0; k < vertex_size; ++k)
	{
		size_t vertex_offset = k;

		unsigned char p = last_vertex[k];

		for (size_t i = 0; i < vertex_count; ++i)
		{
			buffer[i] = zigzag8(vertex_data[vertex_offset] - p);

			p = vertex_data[vertex_offset];

			vertex_offset += vertex_size;
		}

		data = encodeBytes(data, data_end, buffer, (vertex_count + kByteGroupSize - 1) & ~(kByteGroupSize - 1));
		if (!data)
			return NULL;
	}

	memcpy(last_vertex, &vertex_data[vertex_size * (vertex_count - 1)], vertex_size);

	return data;
}

#if defined(SIMD_FALLBACK) || (!defined(SIMD_SSE) && !defined(SIMD_NEON) && !defined(SIMD_AVX) && !defined(SIMD_WASM))
static const unsigned char* decodeBytesGroup(const unsigned char* data, unsigned char* buffer, int bitslog2)
{
#define READ() byte = *data++
#define NEXT(bits) enc = byte >> (8 - bits), byte <<= bits, encv = *data_var, *buffer++ = (enc == (1 << bits) - 1) ? encv : enc, data_var += (enc == (1 << bits) - 1)

	unsigned char byte, enc, encv;
	const unsigned char* data_var;

	switch (bitslog2)
	{
	case 0:
		memset(buffer, 0, kByteGroupSize);
		return data;
	case 1:
		data_var = data + 4;

		// 4 groups with 4 2-bit values in each byte
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);

		return data_var;
	case 2:
		data_var = data + 8;

		// 8 groups with 2 4-bit values in each byte
		READ(), NEXT(4), NEXT(4);
		READ(), NEXT(4), NEXT(4);
		READ(), NEXT(4), NEXT(4);
		READ(), NEXT(4), NEXT(4);
		READ(), NEXT(4), NEXT(4);
		READ(), NEXT(4), NEXT(4);
		READ(), NEXT(4), NEXT(4);
		READ(), NEXT(4), NEXT(4);

		return data_var;
	case 3:
		memcpy(buffer, data, kByteGroupSize);
		return data + kByteGroupSize;
	default:
		assert(!"Unexpected bit length"); // unreachable since bitslog2 is a 2-bit value
		return data;
	}

#undef READ
#undef NEXT
}

static const unsigned char* decodeBytes(const unsigned char* data, const unsigned char* data_end, unsigned char* buffer, size_t buffer_size)
{
	assert(buffer_size % kByteGroupSize == 0);

	const unsigned char* header = data;

	// round number of groups to 4 to get number of header bytes
	size_t header_size = (buffer_size / kByteGroupSize + 3) / 4;

	if (size_t(data_end - data) < header_size)
		return NULL;

	data += header_size;

	for (size_t i = 0; i < buffer_size; i += kByteGroupSize)
	{
		if (size_t(data_end - data) < kByteGroupDecodeLimit)
			return NULL;

		size_t header_offset = i / kByteGroupSize;

		int bitslog2 = (header[header_offset / 4] >> ((header_offset % 4) * 2)) & 3;

		data = decodeBytesGroup(data, buffer + i, bitslog2);
	}

	return data;
}

static const unsigned char* decodeVertexBlock(const unsigned char* data, const unsigned char* data_end, unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, unsigned char last_vertex[256])
{
	assert(vertex_count > 0 && vertex_count <= kVertexBlockMaxSize);

	unsigned char buffer[kVertexBlockMaxSize];
	unsigned char transposed[kVertexBlockSizeBytes];

	size_t vertex_count_aligned = (vertex_count + kByteGroupSize - 1) & ~(kByteGroupSize - 1);

	for (size_t k = 0; k < vertex_size; ++k)
	{
		data = decodeBytes(data, data_end, buffer, vertex_count_aligned);
		if (!data)
			return NULL;

		size_t vertex_offset = k;

		unsigned char p = last_vertex[k];

		for (size_t i = 0; i < vertex_count; ++i)
		{
			unsigned char v = unzigzag8(buffer[i]) + p;

			transposed[vertex_offset] = v;
			p = v;

			vertex_offset += vertex_size;
		}
	}

	memcpy(vertex_data, transposed, vertex_count * vertex_size);

	memcpy(last_vertex, &transposed[vertex_size * (vertex_count - 1)], vertex_size);

	return data;
}
#endif

#if defined(SIMD_SSE) || defined(SIMD_NEON) || defined(SIMD_WASM)
static unsigned char kDecodeBytesGroupShuffle[256][8];
static unsigned char kDecodeBytesGroupCount[256];

#ifdef __wasm__
__attribute__((cold)) // this saves 500 bytes in the output binary - we don't need to vectorize this loop!
#endif
static bool
decodeBytesGroupBuildTables()
{
	for (int mask = 0; mask < 256; ++mask)
	{
		unsigned char shuffle[8];
		unsigned char count = 0;

		for (int i = 0; i < 8; ++i)
		{
			int maski = (mask >> i) & 1;
			shuffle[i] = maski ? count : 0x80;
			count += (unsigned char)(maski);
		}

		memcpy(kDecodeBytesGroupShuffle[mask], shuffle, 8);
		kDecodeBytesGroupCount[mask] = count;
	}

	return true;
}

static bool gDecodeBytesGroupInitialized = decodeBytesGroupBuildTables();
#endif

#ifdef SIMD_SSE
SIMD_TARGET
static __m128i decodeShuffleMask(unsigned char mask0, unsigned char mask1)
{
	__m128i sm0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&kDecodeBytesGroupShuffle[mask0]));
	__m128i sm1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&kDecodeBytesGroupShuffle[mask1]));
	__m128i sm1off = _mm_set1_epi8(kDecodeBytesGroupCount[mask0]);

	__m128i sm1r = _mm_add_epi8(sm1, sm1off);

	return _mm_unpacklo_epi64(sm0, sm1r);
}

SIMD_TARGET
static const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int bitslog2)
{
	switch (bitslog2)
	{
	case 0:
	{
		__m128i result = _mm_setzero_si128();

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data;
	}

	case 1:
	{
#ifdef __GNUC__
		typedef int __attribute__((aligned(1))) unaligned_int;
#else
		typedef int unaligned_int;
#endif

#ifdef SIMD_LATENCYOPT
		unsigned int data32;
		memcpy(&data32, data, 4);
		data32 &= data32 >> 1;

		// arrange bits such that low bits of nibbles of data64 contain all 2-bit elements of data32
		unsigned long long data64 = ((unsigned long long)data32 << 30) | (data32 & 0x3fffffff);

		// adds all 1-bit nibbles together; the sum fits in 4 bits because datacnt=16 would have used mode 3
		int datacnt = int(((data64 & 0x1111111111111111ull) * 0x1111111111111111ull) >> 60);
#endif

		__m128i sel2 = _mm_cvtsi32_si128(*reinterpret_cast<const unaligned_int*>(data));
		__m128i rest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 4));

		__m128i sel22 = _mm_unpacklo_epi8(_mm_srli_epi16(sel2, 4), sel2);
		__m128i sel2222 = _mm_unpacklo_epi8(_mm_srli_epi16(sel22, 2), sel22);
		__m128i sel = _mm_and_si128(sel2222, _mm_set1_epi8(3));

		__m128i mask = _mm_cmpeq_epi8(sel, _mm_set1_epi8(3));
		int mask16 = _mm_movemask_epi8(mask);
		unsigned char mask0 = (unsigned char)(mask16 & 255);
		unsigned char mask1 = (unsigned char)(mask16 >> 8);

		__m128i shuf = decodeShuffleMask(mask0, mask1);

		__m128i result = _mm_or_si128(_mm_shuffle_epi8(rest, shuf), _mm_andnot_si128(mask, sel));

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

#ifdef SIMD_LATENCYOPT
		return data + 4 + datacnt;
#else
		return data + 4 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
#endif
	}

	case 2:
	{
#ifdef SIMD_LATENCYOPT
		unsigned long long data64;
		memcpy(&data64, data, 8);
		data64 &= data64 >> 1;
		data64 &= data64 >> 2;

		// adds all 1-bit nibbles together; the sum fits in 4 bits because datacnt=16 would have used mode 3
		int datacnt = int(((data64 & 0x1111111111111111ull) * 0x1111111111111111ull) >> 60);
#endif

		__m128i sel4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(data));
		__m128i rest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 8));

		__m128i sel44 = _mm_unpacklo_epi8(_mm_srli_epi16(sel4, 4), sel4);
		__m128i sel = _mm_and_si128(sel44, _mm_set1_epi8(15));

		__m128i mask = _mm_cmpeq_epi8(sel, _mm_set1_epi8(15));
		int mask16 = _mm_movemask_epi8(mask);
		unsigned char mask0 = (unsigned char)(mask16 & 255);
		unsigned char mask1 = (unsigned char)(mask16 >> 8);

		__m128i shuf = decodeShuffleMask(mask0, mask1);

		__m128i result = _mm_or_si128(_mm_shuffle_epi8(rest, shuf), _mm_andnot_si128(mask, sel));

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

#ifdef SIMD_LATENCYOPT
		return data + 8 + datacnt;
#else
		return data + 8 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
#endif
	}

	case 3:
	{
		__m128i result = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data + 16;
	}

	default:
		assert(!"Unexpected bit length"); // unreachable since bitslog2 is a 2-bit value
		return data;
	}
}
#endif

#ifdef SIMD_AVX
static const __m128i decodeBytesGroupConfig[] = {
    _mm_set1_epi8(3),
    _mm_set1_epi8(15),
    _mm_setr_epi8(6, 4, 2, 0, 14, 12, 10, 8, 22, 20, 18, 16, 30, 28, 26, 24),
    _mm_setr_epi8(4, 0, 12, 8, 20, 16, 28, 24, 36, 32, 44, 40, 52, 48, 60, 56),
};

static const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int bitslog2)
{
	switch (bitslog2)
	{
	case 0:
	{
		__m128i result = _mm_setzero_si128();

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data;
	}

	case 1:
	case 2:
	{
		const unsigned char* skip = data + (bitslog2 << 2);

		__m128i selb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(data));
		__m128i rest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(skip));

		__m128i sent = decodeBytesGroupConfig[bitslog2 - 1];
		__m128i ctrl = decodeBytesGroupConfig[bitslog2 + 1];

		__m128i selw = _mm_shuffle_epi32(selb, 0x44);
		__m128i sel = _mm_and_si128(sent, _mm_multishift_epi64_epi8(ctrl, selw));
		__mmask16 mask16 = _mm_cmp_epi8_mask(sel, sent, _MM_CMPINT_EQ);

		__m128i result = _mm_mask_expand_epi8(sel, mask16, rest);

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return skip + _mm_popcnt_u32(mask16);
	}

	case 3:
	{
		__m128i result = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data + 16;
	}

	default:
		assert(!"Unexpected bit length"); // unreachable since bitslog2 is a 2-bit value
		return data;
	}
}
#endif

#ifdef SIMD_NEON
static uint8x16_t shuffleBytes(unsigned char mask0, unsigned char mask1, uint8x8_t rest0, uint8x8_t rest1)
{
	uint8x8_t sm0 = vld1_u8(kDecodeBytesGroupShuffle[mask0]);
	uint8x8_t sm1 = vld1_u8(kDecodeBytesGroupShuffle[mask1]);

	uint8x8_t r0 = vtbl1_u8(rest0, sm0);
	uint8x8_t r1 = vtbl1_u8(rest1, sm1);

	return vcombine_u8(r0, r1);
}

static void neonMoveMask(uint8x16_t mask, unsigned char& mask0, unsigned char& mask1)
{
	// magic constant found using z3 SMT assuming mask has 8 groups of 0xff or 0x00
	const uint64_t magic = 0x000103070f1f3f80ull;

	uint64x2_t mask2 = vreinterpretq_u64_u8(mask);

	mask0 = uint8_t((vgetq_lane_u64(mask2, 0) * magic) >> 56);
	mask1 = uint8_t((vgetq_lane_u64(mask2, 1) * magic) >> 56);
}

static const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int bitslog2)
{
	switch (bitslog2)
	{
	case 0:
	{
		uint8x16_t result = vdupq_n_u8(0);

		vst1q_u8(buffer, result);

		return data;
	}

	case 1:
	{
#ifdef SIMD_LATENCYOPT
		unsigned int data32;
		memcpy(&data32, data, 4);
		data32 &= data32 >> 1;

		// arrange bits such that low bits of nibbles of data64 contain all 2-bit elements of data32
		unsigned long long data64 = ((unsigned long long)data32 << 30) | (data32 & 0x3fffffff);

		// adds all 1-bit nibbles together; the sum fits in 4 bits because datacnt=16 would have used mode 3
		int datacnt = int(((data64 & 0x1111111111111111ull) * 0x1111111111111111ull) >> 60);
#endif

		uint8x8_t sel2 = vld1_u8(data);
		uint8x8_t sel22 = vzip_u8(vshr_n_u8(sel2, 4), sel2).val[0];
		uint8x8x2_t sel2222 = vzip_u8(vshr_n_u8(sel22, 2), sel22);
		uint8x16_t sel = vandq_u8(vcombine_u8(sel2222.val[0], sel2222.val[1]), vdupq_n_u8(3));

		uint8x16_t mask = vceqq_u8(sel, vdupq_n_u8(3));
		unsigned char mask0, mask1;
		neonMoveMask(mask, mask0, mask1);

		uint8x8_t rest0 = vld1_u8(data + 4);
		uint8x8_t rest1 = vld1_u8(data + 4 + kDecodeBytesGroupCount[mask0]);

		uint8x16_t result = vbslq_u8(mask, shuffleBytes(mask0, mask1, rest0, rest1), sel);

		vst1q_u8(buffer, result);

#ifdef SIMD_LATENCYOPT
		return data + 4 + datacnt;
#else
		return data + 4 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
#endif
	}

	case 2:
	{
#ifdef SIMD_LATENCYOPT
		unsigned long long data64;
		memcpy(&data64, data, 8);
		data64 &= data64 >> 1;
		data64 &= data64 >> 2;

		// adds all 1-bit nibbles together; the sum fits in 4 bits because datacnt=16 would have used mode 3
		int datacnt = int(((data64 & 0x1111111111111111ull) * 0x1111111111111111ull) >> 60);
#endif

		uint8x8_t sel4 = vld1_u8(data);
		uint8x8x2_t sel44 = vzip_u8(vshr_n_u8(sel4, 4), vand_u8(sel4, vdup_n_u8(15)));
		uint8x16_t sel = vcombine_u8(sel44.val[0], sel44.val[1]);

		uint8x16_t mask = vceqq_u8(sel, vdupq_n_u8(15));
		unsigned char mask0, mask1;
		neonMoveMask(mask, mask0, mask1);

		uint8x8_t rest0 = vld1_u8(data + 8);
		uint8x8_t rest1 = vld1_u8(data + 8 + kDecodeBytesGroupCount[mask0]);

		uint8x16_t result = vbslq_u8(mask, shuffleBytes(mask0, mask1, rest0, rest1), sel);

		vst1q_u8(buffer, result);

#ifdef SIMD_LATENCYOPT
		return data + 8 + datacnt;
#else
		return data + 8 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
#endif
	}

	case 3:
	{
		uint8x16_t result = vld1q_u8(data);

		vst1q_u8(buffer, result);

		return data + 16;
	}

	default:
		assert(!"Unexpected bit length"); // unreachable since bitslog2 is a 2-bit value
		return data;
	}
}
#endif

#ifdef SIMD_WASM
SIMD_TARGET
static v128_t decodeShuffleMask(unsigned char mask0, unsigned char mask1)
{
	v128_t sm0 = wasm_v128_load(&kDecodeBytesGroupShuffle[mask0]);
	v128_t sm1 = wasm_v128_load(&kDecodeBytesGroupShuffle[mask1]);

	v128_t sm1off = wasm_v128_load(&kDecodeBytesGroupCount[mask0]);
	sm1off = wasm_i8x16_shuffle(sm1off, sm1off, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	v128_t sm1r = wasm_i8x16_add(sm1, sm1off);

	return wasmx_unpacklo_v64x2(sm0, sm1r);
}

SIMD_TARGET
static void wasmMoveMask(v128_t mask, unsigned char& mask0, unsigned char& mask1)
{
	// magic constant found using z3 SMT assuming mask has 8 groups of 0xff or 0x00
	const uint64_t magic = 0x000103070f1f3f80ull;

	mask0 = uint8_t((wasm_i64x2_extract_lane(mask, 0) * magic) >> 56);
	mask1 = uint8_t((wasm_i64x2_extract_lane(mask, 1) * magic) >> 56);
}

SIMD_TARGET
static const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int bitslog2)
{
	switch (bitslog2)
	{
	case 0:
	{
		v128_t result = wasm_i8x16_splat(0);

		wasm_v128_store(buffer, result);

		return data;
	}

	case 1:
	{
		v128_t sel2 = wasm_v128_load(data);
		v128_t rest = wasm_v128_load(data + 4);

		v128_t sel22 = wasmx_unpacklo_v8x16(wasm_i16x8_shr(sel2, 4), sel2);
		v128_t sel2222 = wasmx_unpacklo_v8x16(wasm_i16x8_shr(sel22, 2), sel22);
		v128_t sel = wasm_v128_and(sel2222, wasm_i8x16_splat(3));

		v128_t mask = wasm_i8x16_eq(sel, wasm_i8x16_splat(3));

		unsigned char mask0, mask1;
		wasmMoveMask(mask, mask0, mask1);

		v128_t shuf = decodeShuffleMask(mask0, mask1);

		v128_t result = wasm_v128_bitselect(wasm_i8x16_swizzle(rest, shuf), sel, mask);

		wasm_v128_store(buffer, result);

		return data + 4 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
	}

	case 2:
	{
		v128_t sel4 = wasm_v128_load(data);
		v128_t rest = wasm_v128_load(data + 8);

		v128_t sel44 = wasmx_unpacklo_v8x16(wasm_i16x8_shr(sel4, 4), sel4);
		v128_t sel = wasm_v128_and(sel44, wasm_i8x16_splat(15));

		v128_t mask = wasm_i8x16_eq(sel, wasm_i8x16_splat(15));

		unsigned char mask0, mask1;
		wasmMoveMask(mask, mask0, mask1);

		v128_t shuf = decodeShuffleMask(mask0, mask1);

		v128_t result = wasm_v128_bitselect(wasm_i8x16_swizzle(rest, shuf), sel, mask);

		wasm_v128_store(buffer, result);

		return data + 8 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
	}

	case 3:
	{
		v128_t result = wasm_v128_load(data);

		wasm_v128_store(buffer, result);

		return data + 16;
	}

	default:
		assert(!"Unexpected bit length"); // unreachable since bitslog2 is a 2-bit value
		return data;
	}
}
#endif

#if defined(SIMD_SSE) || defined(SIMD_AVX)
SIMD_TARGET
static void transpose8(__m128i& x0, __m128i& x1, __m128i& x2, __m128i& x3)
{
	__m128i t0 = _mm_unpacklo_epi8(x0, x1);
	__m128i t1 = _mm_unpackhi_epi8(x0, x1);
	__m128i t2 = _mm_unpacklo_epi8(x2, x3);
	__m128i t3 = _mm_unpackhi_epi8(x2, x3);

	x0 = _mm_unpacklo_epi16(t0, t2);
	x1 = _mm_unpackhi_epi16(t0, t2);
	x2 = _mm_unpacklo_epi16(t1, t3);
	x3 = _mm_unpackhi_epi16(t1, t3);
}

SIMD_TARGET
static __m128i unzigzag8(__m128i v)
{
	__m128i xl = _mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi8(1)));
	__m128i xr = _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi8(127));

	return _mm_xor_si128(xl, xr);
}
#endif

#ifdef SIMD_NEON
static void transpose8(uint8x16_t& x0, uint8x16_t& x1, uint8x16_t& x2, uint8x16_t& x3)
{
	uint8x16x2_t t01 = vzipq_u8(x0, x1);
	uint8x16x2_t t23 = vzipq_u8(x2, x3);

	uint16x8x2_t x01 = vzipq_u16(vreinterpretq_u16_u8(t01.val[0]), vreinterpretq_u16_u8(t23.val[0]));
	uint16x8x2_t x23 = vzipq_u16(vreinterpretq_u16_u8(t01.val[1]), vreinterpretq_u16_u8(t23.val[1]));

	x0 = vreinterpretq_u8_u16(x01.val[0]);
	x1 = vreinterpretq_u8_u16(x01.val[1]);
	x2 = vreinterpretq_u8_u16(x23.val[0]);
	x3 = vreinterpretq_u8_u16(x23.val[1]);
}

static uint8x16_t unzigzag8(uint8x16_t v)
{
	uint8x16_t xl = vreinterpretq_u8_s8(vnegq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(1)))));
	uint8x16_t xr = vshrq_n_u8(v, 1);

	return veorq_u8(xl, xr);
}
#endif

#ifdef SIMD_WASM
SIMD_TARGET
static void transpose8(v128_t& x0, v128_t& x1, v128_t& x2, v128_t& x3)
{
	v128_t t0 = wasmx_unpacklo_v8x16(x0, x1);
	v128_t t1 = wasmx_unpackhi_v8x16(x0, x1);
	v128_t t2 = wasmx_unpacklo_v8x16(x2, x3);
	v128_t t3 = wasmx_unpackhi_v8x16(x2, x3);

	x0 = wasmx_unpacklo_v16x8(t0, t2);
	x1 = wasmx_unpackhi_v16x8(t0, t2);
	x2 = wasmx_unpacklo_v16x8(t1, t3);
	x3 = wasmx_unpackhi_v16x8(t1, t3);
}

SIMD_TARGET
static v128_t unzigzag8(v128_t v)
{
	v128_t xl = wasm_i8x16_neg(wasm_v128_and(v, wasm_i8x16_splat(1)));
	v128_t xr = wasm_u8x16_shr(v, 1);

	return wasm_v128_xor(xl, xr);
}
#endif

#if defined(SIMD_SSE) || defined(SIMD_AVX) || defined(SIMD_NEON) || defined(SIMD_WASM)
SIMD_TARGET
static const unsigned char* decodeBytesSimd(const unsigned char* data, const unsigned char* data_end, unsigned char* buffer, size_t buffer_size)
{
	assert(buffer_size % kByteGroupSize == 0);
	assert(kByteGroupSize == 16);

	const unsigned char* header = data;

	// round number of groups to 4 to get number of header bytes
	size_t header_size = (buffer_size / kByteGroupSize + 3) / 4;

	if (size_t(data_end - data) < header_size)
		return NULL;

	data += header_size;

	size_t i = 0;

	// fast-path: process 4 groups at a time, do a shared bounds check - each group reads <=24b
	for (; i + kByteGroupSize * 4 <= buffer_size && size_t(data_end - data) >= kByteGroupDecodeLimit * 4; i += kByteGroupSize * 4)
	{
		size_t header_offset = i / kByteGroupSize;
		unsigned char header_byte = header[header_offset / 4];

		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 0, (header_byte >> 0) & 3);
		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 1, (header_byte >> 2) & 3);
		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 2, (header_byte >> 4) & 3);
		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 3, (header_byte >> 6) & 3);
	}

	// slow-path: process remaining groups
	for (; i < buffer_size; i += kByteGroupSize)
	{
		if (size_t(data_end - data) < kByteGroupDecodeLimit)
			return NULL;

		size_t header_offset = i / kByteGroupSize;

		int bitslog2 = (header[header_offset / 4] >> ((header_offset % 4) * 2)) & 3;

		data = decodeBytesGroupSimd(data, buffer + i, bitslog2);
	}

	return data;
}

SIMD_TARGET
static const unsigned char* decodeVertexBlockSimd(const unsigned char* data, const unsigned char* data_end, unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, unsigned char last_vertex[256])
{
	assert(vertex_count > 0 && vertex_count <= kVertexBlockMaxSize);

	unsigned char buffer[kVertexBlockMaxSize * 4];
	unsigned char transposed[kVertexBlockSizeBytes];

	size_t vertex_count_aligned = (vertex_count + kByteGroupSize - 1) & ~(kByteGroupSize - 1);

	for (size_t k = 0; k < vertex_size; k += 4)
	{
		for (size_t j = 0; j < 4; ++j)
		{
			data = decodeBytesSimd(data, data_end, buffer + j * vertex_count_aligned, vertex_count_aligned);
			if (!data)
				return NULL;
		}

#if defined(SIMD_SSE) || defined(SIMD_AVX)
#define TEMP __m128i
#define PREP() __m128i pi = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(last_vertex + k))
#define LOAD(i) __m128i r##i = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buffer + j + i * vertex_count_aligned))
#define GRP4(i) t0 = _mm_shuffle_epi32(r##i, 0), t1 = _mm_shuffle_epi32(r##i, 1), t2 = _mm_shuffle_epi32(r##i, 2), t3 = _mm_shuffle_epi32(r##i, 3)
#define FIXD(i) t##i = pi = _mm_add_epi8(pi, t##i)
#define SAVE(i) *reinterpret_cast<int*>(savep) = _mm_cvtsi128_si32(t##i), savep += vertex_size
#endif

#ifdef SIMD_NEON
#define TEMP uint8x8_t
#define PREP() uint8x8_t pi = vreinterpret_u8_u32(vld1_lane_u32(reinterpret_cast<uint32_t*>(last_vertex + k), vdup_n_u32(0), 0))
#define LOAD(i) uint8x16_t r##i = vld1q_u8(buffer + j + i * vertex_count_aligned)
#define GRP4(i) t0 = vget_low_u8(r##i), t1 = vreinterpret_u8_u32(vdup_lane_u32(vreinterpret_u32_u8(t0), 1)), t2 = vget_high_u8(r##i), t3 = vreinterpret_u8_u32(vdup_lane_u32(vreinterpret_u32_u8(t2), 1))
#define FIXD(i) t##i = pi = vadd_u8(pi, t##i)
#define SAVE(i) vst1_lane_u32(reinterpret_cast<uint32_t*>(savep), vreinterpret_u32_u8(t##i), 0), savep += vertex_size
#endif

#ifdef SIMD_WASM
#define TEMP v128_t
#define PREP() v128_t pi = wasm_v128_load(last_vertex + k)
#define LOAD(i) v128_t r##i = wasm_v128_load(buffer + j + i * vertex_count_aligned)
#define GRP4(i) t0 = wasmx_splat_v32x4(r##i, 0), t1 = wasmx_splat_v32x4(r##i, 1), t2 = wasmx_splat_v32x4(r##i, 2), t3 = wasmx_splat_v32x4(r##i, 3)
#define FIXD(i) t##i = pi = wasm_i8x16_add(pi, t##i)
#define SAVE(i) *reinterpret_cast<int*>(savep) = wasm_i32x4_extract_lane(t##i, 0), savep += vertex_size
#endif

		PREP();

		unsigned char* savep = transposed + k;

		for (size_t j = 0; j < vertex_count_aligned; j += 16)
		{
			LOAD(0);
			LOAD(1);
			LOAD(2);
			LOAD(3);

			r0 = unzigzag8(r0);
			r1 = unzigzag8(r1);
			r2 = unzigzag8(r2);
			r3 = unzigzag8(r3);

			transpose8(r0, r1, r2, r3);

			TEMP t0, t1, t2, t3;

			GRP4(0);
			FIXD(0), FIXD(1), FIXD(2), FIXD(3);
			SAVE(0), SAVE(1), SAVE(2), SAVE(3);

			GRP4(1);
			FIXD(0), FIXD(1), FIXD(2), FIXD(3);
			SAVE(0), SAVE(1), SAVE(2), SAVE(3);

			GRP4(2);
			FIXD(0), FIXD(1), FIXD(2), FIXD(3);
			SAVE(0), SAVE(1), SAVE(2), SAVE(3);

			GRP4(3);
			FIXD(0), FIXD(1), FIXD(2), FIXD(3);
			SAVE(0), SAVE(1), SAVE(2), SAVE(3);

#undef TEMP
#undef PREP
#undef LOAD
#undef GRP4
#undef FIXD
#undef SAVE
		}
	}

	memcpy(vertex_data, transposed, vertex_count * vertex_size);

	memcpy(last_vertex, &transposed[vertex_size * (vertex_count - 1)], vertex_size);

	return data;
}
#endif

#if defined(SIMD_SSE) && defined(SIMD_FALLBACK)
static unsigned int getCpuFeatures()
{
	int cpuinfo[4] = {};
#ifdef _MSC_VER
	__cpuid(cpuinfo, 1);
#else
	__cpuid(1, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
	return cpuinfo[2];
}

static unsigned int cpuid = getCpuFeatures();
#endif

} // namespace meshopt

size_t meshopt_encodeVertexBuffer(unsigned char* buffer, size_t buffer_size, const void* vertices, size_t vertex_count, size_t vertex_size)
{
	using namespace meshopt;

	assert(vertex_size > 0 && vertex_size <= 256);
	assert(vertex_size % 4 == 0);

	const unsigned char* vertex_data = static_cast<const unsigned char*>(vertices);

	unsigned char* data = buffer;
	unsigned char* data_end = buffer + buffer_size;

	if (size_t(data_end - data) < 1 + vertex_size)
		return 0;

	int version = gEncodeVertexVersion;

	*data++ = (unsigned char)(kVertexHeader | version);

	unsigned char first_vertex[256] = {};
	if (vertex_count > 0)
		memcpy(first_vertex, vertex_data, vertex_size);

	unsigned char last_vertex[256] = {};
	memcpy(last_vertex, first_vertex, vertex_size);

	size_t vertex_block_size = getVertexBlockSize(vertex_size);

	size_t vertex_offset = 0;

	while (vertex_offset < vertex_count)
	{
		size_t block_size = (vertex_offset + vertex_block_size < vertex_count) ? vertex_block_size : vertex_count - vertex_offset;

		data = encodeVertexBlock(data, data_end, vertex_data + vertex_offset * vertex_size, block_size, vertex_size, last_vertex);
		if (!data)
			return 0;

		vertex_offset += block_size;
	}

	size_t tail_size = vertex_size < kTailMaxSize ? kTailMaxSize : vertex_size;

	if (size_t(data_end - data) < tail_size)
		return 0;

	// write first vertex to the end of the stream and pad it to 32 bytes; this is important to simplify bounds checks in decoder
	if (vertex_size < kTailMaxSize)
	{
		memset(data, 0, kTailMaxSize - vertex_size);
		data += kTailMaxSize - vertex_size;
	}

	memcpy(data, first_vertex, vertex_size);
	data += vertex_size;

	assert(data >= buffer + tail_size);
	assert(data <= buffer + buffer_size);

	return data - buffer;
}

size_t meshopt_encodeVertexBufferBound(size_t vertex_count, size_t vertex_size)
{
	using namespace meshopt;

	assert(vertex_size > 0 && vertex_size <= 256);
	assert(vertex_size % 4 == 0);

	size_t vertex_block_size = getVertexBlockSize(vertex_size);
	size_t vertex_block_count = (vertex_count + vertex_block_size - 1) / vertex_block_size;

	size_t vertex_block_header_size = (vertex_block_size / kByteGroupSize + 3) / 4;
	size_t vertex_block_data_size = vertex_block_size;

	size_t tail_size = vertex_size < kTailMaxSize ? kTailMaxSize : vertex_size;

	return 1 + vertex_block_count * vertex_size * (vertex_block_header_size + vertex_block_data_size) + tail_size;
}

void meshopt_encodeVertexVersion(int version)
{
	assert(unsigned(version) <= 0);

	meshopt::gEncodeVertexVersion = version;
}

int meshopt_decodeVertexBuffer(void* destination, size_t vertex_count, size_t vertex_size, const unsigned char* buffer, size_t buffer_size)
{
	using namespace meshopt;

	assert(vertex_size > 0 && vertex_size <= 256);
	assert(vertex_size % 4 == 0);

	const unsigned char* (*decode)(const unsigned char*, const unsigned char*, unsigned char*, size_t, size_t, unsigned char[256]) = NULL;

#if defined(SIMD_SSE) && defined(SIMD_FALLBACK)
	decode = (cpuid & (1 << 9)) ? decodeVertexBlockSimd : decodeVertexBlock;
#elif defined(SIMD_SSE) || defined(SIMD_AVX) || defined(SIMD_NEON) || defined(SIMD_WASM)
	decode = decodeVertexBlockSimd;
#else
	decode = decodeVertexBlock;
#endif

#if defined(SIMD_SSE) || defined(SIMD_NEON) || defined(SIMD_WASM)
	assert(gDecodeBytesGroupInitialized);
	(void)gDecodeBytesGroupInitialized;
#endif

	unsigned char* vertex_data = static_cast<unsigned char*>(destination);

	const unsigned char* data = buffer;
	const unsigned char* data_end = buffer + buffer_size;

	if (size_t(data_end - data) < 1 + vertex_size)
		return -2;

	unsigned char data_header = *data++;

	if ((data_header & 0xf0) != kVertexHeader)
		return -1;

	int version = data_header & 0x0f;
	if (version > 0)
		return -1;

	unsigned char last_vertex[256];
	memcpy(last_vertex, data_end - vertex_size, vertex_size);

	size_t vertex_block_size = getVertexBlockSize(vertex_size);

	size_t vertex_offset = 0;

	while (vertex_offset < vertex_count)
	{
		size_t block_size = (vertex_offset + vertex_block_size < vertex_count) ? vertex_block_size : vertex_count - vertex_offset;

		data = decode(data, data_end, vertex_data + vertex_offset * vertex_size, block_size, vertex_size, last_vertex);
		if (!data)
			return -2;

		vertex_offset += block_size;
	}

	size_t tail_size = vertex_size < kTailMaxSize ? kTailMaxSize : vertex_size;

	if (size_t(data_end - data) != tail_size)
		return -3;

	return 0;
}

#undef SIMD_NEON
#undef SIMD_SSE
#undef SIMD_AVX
#undef SIMD_WASM
#undef SIMD_FALLBACK
#undef SIMD_TARGET
