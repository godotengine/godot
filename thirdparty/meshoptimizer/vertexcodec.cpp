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

// In switch dispatch, marking default case as unreachable allows to remove redundant bounds checks
#if defined(__GNUC__)
#define SIMD_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define SIMD_UNREACHABLE() __assume(false)
#else
#define SIMD_UNREACHABLE() assert(!"Unreachable")
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

#ifndef TRACE
#define TRACE 0
#endif

#if TRACE
#include <stdio.h>
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

static int gEncodeVertexVersion = 1;
const int kDecodeVertexVersion = 1;

const size_t kVertexBlockSizeBytes = 8192;
const size_t kVertexBlockMaxSize = 256;
const size_t kByteGroupSize = 16;
const size_t kByteGroupDecodeLimit = 24;
const size_t kTailMinSizeV0 = 32;
const size_t kTailMinSizeV1 = 24;

static const int kBitsV0[4] = {0, 2, 4, 8};
static const int kBitsV1[5] = {0, 1, 2, 4, 8};

const int kEncodeDefaultLevel = 2;

static size_t getVertexBlockSize(size_t vertex_size)
{
	// make sure the entire block fits into the scratch buffer and is aligned to byte group size
	// note: the block size is implicitly part of the format, so we can't change it without breaking compatibility
	size_t result = (kVertexBlockSizeBytes / vertex_size) & ~(kByteGroupSize - 1);

	return (result < kVertexBlockMaxSize) ? result : kVertexBlockMaxSize;
}

inline unsigned int rotate(unsigned int v, int r)
{
	return (v << r) | (v >> ((32 - r) & 31));
}

template <typename T>
inline T zigzag(T v)
{
	return (0 - (v >> (sizeof(T) * 8 - 1))) ^ (v << 1);
}

template <typename T>
inline T unzigzag(T v)
{
	return (0 - (v & 1)) ^ (v >> 1);
}

#if TRACE
struct Stats
{
	size_t size;
	size_t header;  // bytes for header
	size_t bitg[9]; // bytes for bit groups
	size_t bitc[8]; // bit consistency: how many bits are shared between all bytes in a group
	size_t ctrl[4]; // number of control groups
};

static Stats* bytestats = NULL;
static Stats vertexstats[256];
#endif

static bool encodeBytesGroupZero(const unsigned char* buffer)
{
	assert(kByteGroupSize == sizeof(unsigned long long) * 2);

	unsigned long long v[2];
	memcpy(v, buffer, sizeof(v));

	return (v[0] | v[1]) == 0;
}

static size_t encodeBytesGroupMeasure(const unsigned char* buffer, int bits)
{
	assert(bits >= 0 && bits <= 8);

	if (bits == 0)
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
	assert(bits >= 0 && bits <= 8);
	assert(kByteGroupSize % 8 == 0);

	if (bits == 0)
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

		// encode 1-bit groups in reverse bit order
		// this makes them faster to decode alongside other groups
		if (bits == 1)
			byte = (unsigned char)(((byte * 0x80200802ull) & 0x0884422110ull) * 0x0101010101ull >> 32);

		*data++ = byte;
	}

	for (size_t i = 0; i < kByteGroupSize; ++i)
	{
		unsigned char v = buffer[i];

		// branchless append of out-of-range values
		*data = v;
		data += v >= sentinel;
	}

	return data;
}

static unsigned char* encodeBytes(unsigned char* data, unsigned char* data_end, const unsigned char* buffer, size_t buffer_size, const int bits[4])
{
	assert(buffer_size % kByteGroupSize == 0);

	unsigned char* header = data;

	// round number of groups to 4 to get number of header bytes
	size_t header_size = (buffer_size / kByteGroupSize + 3) / 4;

	if (size_t(data_end - data) < header_size)
		return NULL;

	data += header_size;

	memset(header, 0, header_size);

	int last_bits = -1;

	for (size_t i = 0; i < buffer_size; i += kByteGroupSize)
	{
		if (size_t(data_end - data) < kByteGroupDecodeLimit)
			return NULL;

		int best_bitk = 3;
		size_t best_size = encodeBytesGroupMeasure(buffer + i, bits[best_bitk]);

		for (int bitk = 0; bitk < 3; ++bitk)
		{
			size_t size = encodeBytesGroupMeasure(buffer + i, bits[bitk]);

			// favor consistent bit selection across groups, but never replace literals
			if (size < best_size || (size == best_size && bits[bitk] == last_bits && bits[best_bitk] != 8))
			{
				best_bitk = bitk;
				best_size = size;
			}
		}

		size_t header_offset = i / kByteGroupSize;
		header[header_offset / 4] |= best_bitk << ((header_offset % 4) * 2);

		int best_bits = bits[best_bitk];
		unsigned char* next = encodeBytesGroup(data, buffer + i, best_bits);

		assert(data + best_size == next);
		data = next;
		last_bits = best_bits;

#if TRACE
		bytestats->bitg[best_bits] += best_size;
#endif
	}

#if TRACE
	bytestats->header += header_size;
#endif

	return data;
}

template <typename T, bool Xor>
static void encodeDeltas1(unsigned char* buffer, const unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, const unsigned char last_vertex[256], size_t k, int rot)
{
	size_t k0 = k & ~(sizeof(T) - 1);
	int ks = (k & (sizeof(T) - 1)) * 8;

	T p = last_vertex[k0];
	for (size_t j = 1; j < sizeof(T); ++j)
		p |= T(last_vertex[k0 + j]) << (j * 8);

	const unsigned char* vertex = vertex_data + k0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		T v = vertex[0];
		for (size_t j = 1; j < sizeof(T); ++j)
			v |= vertex[j] << (j * 8);

		T d = Xor ? T(rotate(v ^ p, rot)) : zigzag(T(v - p));

		buffer[i] = (unsigned char)(d >> ks);
		p = v;
		vertex += vertex_size;
	}
}

static void encodeDeltas(unsigned char* buffer, const unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, const unsigned char last_vertex[256], size_t k, int channel)
{
	switch (channel & 3)
	{
	case 0:
		return encodeDeltas1<unsigned char, false>(buffer, vertex_data, vertex_count, vertex_size, last_vertex, k, 0);
	case 1:
		return encodeDeltas1<unsigned short, false>(buffer, vertex_data, vertex_count, vertex_size, last_vertex, k, 0);
	case 2:
		return encodeDeltas1<unsigned int, true>(buffer, vertex_data, vertex_count, vertex_size, last_vertex, k, channel >> 4);
	default:
		assert(!"Unsupported channel encoding"); // unreachable
	}
}

static int estimateBits(unsigned char v)
{
	return v <= 15 ? (v <= 3 ? (v == 0 ? 0 : 2) : 4) : 8;
}

static int estimateRotate(const unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, size_t k, size_t group_size)
{
	size_t sizes[8] = {};

	const unsigned char* vertex = vertex_data + k;
	unsigned int last = vertex[0] | (vertex[1] << 8) | (vertex[2] << 16) | (vertex[3] << 24);

	for (size_t i = 0; i < vertex_count; i += group_size)
	{
		unsigned int bitg = 0;

		// calculate bit consistency mask for the group
		for (size_t j = 0; j < group_size && i + j < vertex_count; ++j)
		{
			unsigned int v = vertex[0] | (vertex[1] << 8) | (vertex[2] << 16) | (vertex[3] << 24);
			unsigned int d = v ^ last;

			bitg |= d;
			last = v;
			vertex += vertex_size;
		}

#if TRACE
		for (int j = 0; j < 32; ++j)
			vertexstats[k + (j / 8)].bitc[j % 8] += (i + group_size < vertex_count ? group_size : vertex_count - i) * (1 - ((bitg >> j) & 1));
#endif

		for (int j = 0; j < 8; ++j)
		{
			unsigned int bitr = rotate(bitg, j);

			sizes[j] += estimateBits((unsigned char)(bitr >> 0)) + estimateBits((unsigned char)(bitr >> 8));
			sizes[j] += estimateBits((unsigned char)(bitr >> 16)) + estimateBits((unsigned char)(bitr >> 24));
		}
	}

	int best_rot = 0;
	for (int rot = 1; rot < 8; ++rot)
		best_rot = (sizes[rot] < sizes[best_rot]) ? rot : best_rot;

	return best_rot;
}

static int estimateChannel(const unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, size_t k, size_t vertex_block_size, size_t block_skip, int max_channel, int xor_rot)
{
	unsigned char block[kVertexBlockMaxSize];
	assert(vertex_block_size <= kVertexBlockMaxSize);

	unsigned char last_vertex[256] = {};

	size_t sizes[3] = {};
	assert(max_channel <= 3);

	for (size_t i = 0; i < vertex_count; i += vertex_block_size * block_skip)
	{
		size_t block_size = i + vertex_block_size < vertex_count ? vertex_block_size : vertex_count - i;
		size_t block_size_aligned = (block_size + kByteGroupSize - 1) & ~(kByteGroupSize - 1);

		memcpy(last_vertex, vertex_data + (i == 0 ? 0 : i - 1) * vertex_size, vertex_size);

		// we sometimes encode elements we didn't fill when rounding to kByteGroupSize
		if (block_size < block_size_aligned)
			memset(block + block_size, 0, block_size_aligned - block_size);

		for (int channel = 0; channel < max_channel; ++channel)
			for (size_t j = 0; j < 4; ++j)
			{
				encodeDeltas(block, vertex_data + i * vertex_size, block_size, vertex_size, last_vertex, k + j, channel | (xor_rot << 4));

				for (size_t ig = 0; ig < block_size; ig += kByteGroupSize)
				{
					// to maximize encoding performance we only evaluate 1/2/4/8 bit groups
					size_t size1 = encodeBytesGroupMeasure(block + ig, 1);
					size_t size2 = encodeBytesGroupMeasure(block + ig, 2);
					size_t size4 = encodeBytesGroupMeasure(block + ig, 4);
					size_t size8 = encodeBytesGroupMeasure(block + ig, 8);

					size_t best_size = size1 < size2 ? size1 : size2;
					best_size = best_size < size4 ? best_size : size4;
					best_size = best_size < size8 ? best_size : size8;

					sizes[channel] += best_size;
				}
			}
	}

	int best_channel = 0;
	for (int channel = 1; channel < max_channel; ++channel)
		best_channel = (sizes[channel] < sizes[best_channel]) ? channel : best_channel;

	return best_channel == 2 ? best_channel | (xor_rot << 4) : best_channel;
}

static bool estimateControlZero(const unsigned char* buffer, size_t vertex_count_aligned)
{
	for (size_t i = 0; i < vertex_count_aligned; i += kByteGroupSize)
		if (!encodeBytesGroupZero(buffer + i))
			return false;

	return true;
}

static int estimateControl(const unsigned char* buffer, size_t vertex_count, size_t vertex_count_aligned, int level)
{
	if (estimateControlZero(buffer, vertex_count_aligned))
		return 2; // zero encoding

	if (level == 0)
		return 1; // 1248 encoding in level 0 for encoding speed

	// round number of groups to 4 to get number of header bytes
	size_t header_size = (vertex_count_aligned / kByteGroupSize + 3) / 4;

	size_t est_bytes0 = header_size, est_bytes1 = header_size;

	for (size_t i = 0; i < vertex_count_aligned; i += kByteGroupSize)
	{
		// assumes kBitsV1[] = {0, 1, 2, 4, 8} for performance
		size_t size0 = encodeBytesGroupMeasure(buffer + i, 0);
		size_t size1 = encodeBytesGroupMeasure(buffer + i, 1);
		size_t size2 = encodeBytesGroupMeasure(buffer + i, 2);
		size_t size4 = encodeBytesGroupMeasure(buffer + i, 4);
		size_t size8 = encodeBytesGroupMeasure(buffer + i, 8);

		// both control modes have access to 1/2/4 bit encoding
		size_t size12 = size1 < size2 ? size1 : size2;
		size_t size124 = size12 < size4 ? size12 : size4;

		// each control mode has access to 0/8 bit encoding respectively
		est_bytes0 += size124 < size0 ? size124 : size0;
		est_bytes1 += size124 < size8 ? size124 : size8;
	}

	// pick shortest control entry but prefer literal encoding
	if (est_bytes0 < vertex_count || est_bytes1 < vertex_count)
		return est_bytes0 < est_bytes1 ? 0 : 1;
	else
		return 3; // literal encoding
}

static unsigned char* encodeVertexBlock(unsigned char* data, unsigned char* data_end, const unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, unsigned char last_vertex[256], const unsigned char* channels, int version, int level)
{
	assert(vertex_count > 0 && vertex_count <= kVertexBlockMaxSize);
	assert(vertex_size % 4 == 0);

	unsigned char buffer[kVertexBlockMaxSize];
	assert(sizeof(buffer) % kByteGroupSize == 0);

	size_t vertex_count_aligned = (vertex_count + kByteGroupSize - 1) & ~(kByteGroupSize - 1);

	// we sometimes encode elements we didn't fill when rounding to kByteGroupSize
	memset(buffer, 0, sizeof(buffer));

	size_t control_size = version == 0 ? 0 : vertex_size / 4;
	if (size_t(data_end - data) < control_size)
		return NULL;

	unsigned char* control = data;
	data += control_size;

	memset(control, 0, control_size);

	for (size_t k = 0; k < vertex_size; ++k)
	{
		encodeDeltas(buffer, vertex_data, vertex_count, vertex_size, last_vertex, k, version == 0 ? 0 : channels[k / 4]);

#if TRACE
		const unsigned char* olddata = data;
		bytestats = &vertexstats[k];
#endif

		int ctrl = 0;

		if (version != 0)
		{
			ctrl = estimateControl(buffer, vertex_count, vertex_count_aligned, level);

			assert(unsigned(ctrl) < 4);
			control[k / 4] |= ctrl << ((k % 4) * 2);

#if TRACE
			vertexstats[k].ctrl[ctrl]++;
#endif
		}

		if (ctrl == 3)
		{
			// literal encoding
			if (size_t(data_end - data) < vertex_count)
				return NULL;

			memcpy(data, buffer, vertex_count);
			data += vertex_count;
		}
		else if (ctrl != 2) // non-zero encoding
		{
			data = encodeBytes(data, data_end, buffer, vertex_count_aligned, version == 0 ? kBitsV0 : kBitsV1 + ctrl);
			if (!data)
				return NULL;
		}

#if TRACE
		bytestats = NULL;
		vertexstats[k].size += data - olddata;
#endif
	}

	memcpy(last_vertex, &vertex_data[vertex_size * (vertex_count - 1)], vertex_size);

	return data;
}

#if defined(SIMD_FALLBACK) || (!defined(SIMD_SSE) && !defined(SIMD_NEON) && !defined(SIMD_AVX) && !defined(SIMD_WASM))
static const unsigned char* decodeBytesGroup(const unsigned char* data, unsigned char* buffer, int bits)
{
#define READ() byte = *data++
#define NEXT(bits) enc = byte >> (8 - bits), byte <<= bits, encv = *data_var, *buffer++ = (enc == (1 << bits) - 1) ? encv : enc, data_var += (enc == (1 << bits) - 1)

	unsigned char byte, enc, encv;
	const unsigned char* data_var;

	switch (bits)
	{
	case 0:
		memset(buffer, 0, kByteGroupSize);
		return data;
	case 1:
		data_var = data + 2;

		// 2 groups with 8 1-bit values in each byte (reversed from the order in other groups)
		READ();
		byte = (unsigned char)(((byte * 0x80200802ull) & 0x0884422110ull) * 0x0101010101ull >> 32);
		NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1);
		READ();
		byte = (unsigned char)(((byte * 0x80200802ull) & 0x0884422110ull) * 0x0101010101ull >> 32);
		NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1), NEXT(1);

		return data_var;
	case 2:
		data_var = data + 4;

		// 4 groups with 4 2-bit values in each byte
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);
		READ(), NEXT(2), NEXT(2), NEXT(2), NEXT(2);

		return data_var;
	case 4:
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
	case 8:
		memcpy(buffer, data, kByteGroupSize);
		return data + kByteGroupSize;
	default:
		assert(!"Unexpected bit length"); // unreachable
		return data;
	}

#undef READ
#undef NEXT
}

static const unsigned char* decodeBytes(const unsigned char* data, const unsigned char* data_end, unsigned char* buffer, size_t buffer_size, const int* bits)
{
	assert(buffer_size % kByteGroupSize == 0);

	// round number of groups to 4 to get number of header bytes
	size_t header_size = (buffer_size / kByteGroupSize + 3) / 4;
	if (size_t(data_end - data) < header_size)
		return NULL;

	const unsigned char* header = data;
	data += header_size;

	for (size_t i = 0; i < buffer_size; i += kByteGroupSize)
	{
		if (size_t(data_end - data) < kByteGroupDecodeLimit)
			return NULL;

		size_t header_offset = i / kByteGroupSize;
		int bitsk = (header[header_offset / 4] >> ((header_offset % 4) * 2)) & 3;

		data = decodeBytesGroup(data, buffer + i, bits[bitsk]);
	}

	return data;
}

template <typename T, bool Xor>
static void decodeDeltas1(const unsigned char* buffer, unsigned char* transposed, size_t vertex_count, size_t vertex_size, const unsigned char* last_vertex, int rot)
{
	for (size_t k = 0; k < 4; k += sizeof(T))
	{
		size_t vertex_offset = k;

		T p = last_vertex[0];
		for (size_t j = 1; j < sizeof(T); ++j)
			p |= last_vertex[j] << (8 * j);

		for (size_t i = 0; i < vertex_count; ++i)
		{
			T v = buffer[i];
			for (size_t j = 1; j < sizeof(T); ++j)
				v |= buffer[i + vertex_count * j] << (8 * j);

			v = Xor ? T(rotate(v, rot)) ^ p : unzigzag(v) + p;

			for (size_t j = 0; j < sizeof(T); ++j)
				transposed[vertex_offset + j] = (unsigned char)(v >> (j * 8));

			p = v;

			vertex_offset += vertex_size;
		}

		buffer += vertex_count * sizeof(T);
		last_vertex += sizeof(T);
	}
}

static const unsigned char* decodeVertexBlock(const unsigned char* data, const unsigned char* data_end, unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, unsigned char last_vertex[256], const unsigned char* channels, int version)
{
	assert(vertex_count > 0 && vertex_count <= kVertexBlockMaxSize);

	unsigned char buffer[kVertexBlockMaxSize * 4];
	unsigned char transposed[kVertexBlockSizeBytes];

	size_t vertex_count_aligned = (vertex_count + kByteGroupSize - 1) & ~(kByteGroupSize - 1);
	assert(vertex_count <= vertex_count_aligned);

	size_t control_size = version == 0 ? 0 : vertex_size / 4;
	if (size_t(data_end - data) < control_size)
		return NULL;

	const unsigned char* control = data;
	data += control_size;

	for (size_t k = 0; k < vertex_size; k += 4)
	{
		unsigned char ctrl_byte = version == 0 ? 0 : control[k / 4];

		for (size_t j = 0; j < 4; ++j)
		{
			int ctrl = (ctrl_byte >> (j * 2)) & 3;

			if (ctrl == 3)
			{
				// literal encoding
				if (size_t(data_end - data) < vertex_count)
					return NULL;

				memcpy(buffer + j * vertex_count, data, vertex_count);
				data += vertex_count;
			}
			else if (ctrl == 2)
			{
				// zero encoding
				memset(buffer + j * vertex_count, 0, vertex_count);
			}
			else
			{
				data = decodeBytes(data, data_end, buffer + j * vertex_count, vertex_count_aligned, version == 0 ? kBitsV0 : kBitsV1 + ctrl);
				if (!data)
					return NULL;
			}
		}

		int channel = version == 0 ? 0 : channels[k / 4];

		switch (channel & 3)
		{
		case 0:
			decodeDeltas1<unsigned char, false>(buffer, transposed + k, vertex_count, vertex_size, last_vertex + k, 0);
			break;
		case 1:
			decodeDeltas1<unsigned short, false>(buffer, transposed + k, vertex_count, vertex_size, last_vertex + k, 0);
			break;
		case 2:
			decodeDeltas1<unsigned int, true>(buffer, transposed + k, vertex_count, vertex_size, last_vertex + k, (32 - (channel >> 4)) & 31);
			break;
		default:
			return NULL; // invalid channel type
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
inline __m128i decodeShuffleMask(unsigned char mask0, unsigned char mask1)
{
	__m128i sm0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&kDecodeBytesGroupShuffle[mask0]));
	__m128i sm1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&kDecodeBytesGroupShuffle[mask1]));
	__m128i sm1off = _mm_set1_epi8(kDecodeBytesGroupCount[mask0]);

	__m128i sm1r = _mm_add_epi8(sm1, sm1off);

	return _mm_unpacklo_epi64(sm0, sm1r);
}

SIMD_TARGET
inline const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int hbits)
{
	switch (hbits)
	{
	case 0:
	case 4:
	{
		__m128i result = _mm_setzero_si128();

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data;
	}

	case 1:
	case 6:
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
	case 7:
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
	case 8:
	{
		__m128i result = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data + 16;
	}

	case 5:
	{
		__m128i rest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 2));

		unsigned char mask0 = data[0];
		unsigned char mask1 = data[1];

		__m128i shuf = decodeShuffleMask(mask0, mask1);
		__m128i result = _mm_shuffle_epi8(rest, shuf);

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data + 2 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
	}

	default:
		SIMD_UNREACHABLE(); // unreachable
	}
}
#endif

#ifdef SIMD_AVX
static const __m128i kDecodeBytesGroupConfig[8][2] = {
    {_mm_setzero_si128(), _mm_setzero_si128()},
    {_mm_set1_epi8(3), _mm_setr_epi8(6, 4, 2, 0, 14, 12, 10, 8, 22, 20, 18, 16, 30, 28, 26, 24)},
    {_mm_set1_epi8(15), _mm_setr_epi8(4, 0, 12, 8, 20, 16, 28, 24, 36, 32, 44, 40, 52, 48, 60, 56)},
    {_mm_setzero_si128(), _mm_setzero_si128()},
    {_mm_setzero_si128(), _mm_setzero_si128()},
    {_mm_set1_epi8(1), _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)},
    {_mm_set1_epi8(3), _mm_setr_epi8(6, 4, 2, 0, 14, 12, 10, 8, 22, 20, 18, 16, 30, 28, 26, 24)},
    {_mm_set1_epi8(15), _mm_setr_epi8(4, 0, 12, 8, 20, 16, 28, 24, 36, 32, 44, 40, 52, 48, 60, 56)},
};

SIMD_TARGET
inline const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int hbits)
{
	switch (hbits)
	{
	case 0:
	case 4:
	{
		__m128i result = _mm_setzero_si128();

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data;
	}

	case 5: // 1-bit
	case 1: // 2-bit
	case 6:
	case 2: // 4-bit
	case 7:
	{
		const unsigned char* skip = data + (2 << (hbits < 3 ? hbits : hbits - 5));

		__m128i selb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(data));
		__m128i rest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(skip));

		__m128i sent = kDecodeBytesGroupConfig[hbits][0];
		__m128i ctrl = kDecodeBytesGroupConfig[hbits][1];

		__m128i selw = _mm_shuffle_epi32(selb, 0x44);
		__m128i sel = _mm_and_si128(sent, _mm_multishift_epi64_epi8(ctrl, selw));
		__mmask16 mask16 = _mm_cmp_epi8_mask(sel, sent, _MM_CMPINT_EQ);

		__m128i result = _mm_mask_expand_epi8(sel, mask16, rest);

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return skip + _mm_popcnt_u32(mask16);
	}

	case 3:
	case 8:
	{
		__m128i result = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));

		_mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), result);

		return data + 16;
	}

	default:
		SIMD_UNREACHABLE(); // unreachable
	}
}
#endif

#ifdef SIMD_NEON
SIMD_TARGET
inline uint8x16_t shuffleBytes(unsigned char mask0, unsigned char mask1, uint8x8_t rest0, uint8x8_t rest1)
{
	uint8x8_t sm0 = vld1_u8(kDecodeBytesGroupShuffle[mask0]);
	uint8x8_t sm1 = vld1_u8(kDecodeBytesGroupShuffle[mask1]);

	uint8x8_t r0 = vtbl1_u8(rest0, sm0);
	uint8x8_t r1 = vtbl1_u8(rest1, sm1);

	return vcombine_u8(r0, r1);
}

SIMD_TARGET
inline void neonMoveMask(uint8x16_t mask, unsigned char& mask0, unsigned char& mask1)
{
	// magic constant found using z3 SMT assuming mask has 8 groups of 0xff or 0x00
	const uint64_t magic = 0x000103070f1f3f80ull;

	uint64x2_t mask2 = vreinterpretq_u64_u8(mask);

	mask0 = uint8_t((vgetq_lane_u64(mask2, 0) * magic) >> 56);
	mask1 = uint8_t((vgetq_lane_u64(mask2, 1) * magic) >> 56);
}

SIMD_TARGET
inline const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int hbits)
{
	switch (hbits)
	{
	case 0:
	case 4:
	{
		uint8x16_t result = vdupq_n_u8(0);

		vst1q_u8(buffer, result);

		return data;
	}

	case 1:
	case 6:
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
	case 7:
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
	case 8:
	{
		uint8x16_t result = vld1q_u8(data);

		vst1q_u8(buffer, result);

		return data + 16;
	}

	case 5:
	{
		unsigned char mask0 = data[0];
		unsigned char mask1 = data[1];

		uint8x8_t rest0 = vld1_u8(data + 2);
		uint8x8_t rest1 = vld1_u8(data + 2 + kDecodeBytesGroupCount[mask0]);

		uint8x16_t result = shuffleBytes(mask0, mask1, rest0, rest1);

		vst1q_u8(buffer, result);

		return data + 2 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
	}

	default:
		SIMD_UNREACHABLE(); // unreachable
	}
}
#endif

#ifdef SIMD_WASM
SIMD_TARGET
inline v128_t decodeShuffleMask(unsigned char mask0, unsigned char mask1)
{
	v128_t sm0 = wasm_v128_load(&kDecodeBytesGroupShuffle[mask0]);
	v128_t sm1 = wasm_v128_load(&kDecodeBytesGroupShuffle[mask1]);

	v128_t sm1off = wasm_v128_load8_splat(&kDecodeBytesGroupCount[mask0]);
	v128_t sm1r = wasm_i8x16_add(sm1, sm1off);

	return wasmx_unpacklo_v64x2(sm0, sm1r);
}

SIMD_TARGET
inline void wasmMoveMask(v128_t mask, unsigned char& mask0, unsigned char& mask1)
{
	// magic constant found using z3 SMT assuming mask has 8 groups of 0xff or 0x00
	const uint64_t magic = 0x000103070f1f3f80ull;

	mask0 = uint8_t((wasm_i64x2_extract_lane(mask, 0) * magic) >> 56);
	mask1 = uint8_t((wasm_i64x2_extract_lane(mask, 1) * magic) >> 56);
}

SIMD_TARGET
inline const unsigned char* decodeBytesGroupSimd(const unsigned char* data, unsigned char* buffer, int hbits)
{
	switch (hbits)
	{
	case 0:
	case 4:
	{
		v128_t result = wasm_i8x16_splat(0);

		wasm_v128_store(buffer, result);

		return data;
	}

	case 1:
	case 6:
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
	case 7:
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
	case 8:
	{
		v128_t result = wasm_v128_load(data);

		wasm_v128_store(buffer, result);

		return data + 16;
	}

	case 5:
	{
		v128_t rest = wasm_v128_load(data + 2);

		unsigned char mask0 = data[0];
		unsigned char mask1 = data[1];

		v128_t shuf = decodeShuffleMask(mask0, mask1);
		v128_t result = wasm_i8x16_swizzle(rest, shuf);

		wasm_v128_store(buffer, result);

		return data + 2 + kDecodeBytesGroupCount[mask0] + kDecodeBytesGroupCount[mask1];
	}

	default:
		SIMD_UNREACHABLE(); // unreachable
	}
}
#endif

#if defined(SIMD_SSE) || defined(SIMD_AVX)
SIMD_TARGET
inline void transpose8(__m128i& x0, __m128i& x1, __m128i& x2, __m128i& x3)
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
inline __m128i unzigzag8(__m128i v)
{
	__m128i xl = _mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi8(1)));
	__m128i xr = _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi8(127));

	return _mm_xor_si128(xl, xr);
}

SIMD_TARGET
inline __m128i unzigzag16(__m128i v)
{
	__m128i xl = _mm_sub_epi16(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi16(1)));
	__m128i xr = _mm_srli_epi16(v, 1);

	return _mm_xor_si128(xl, xr);
}

SIMD_TARGET
inline __m128i rotate32(__m128i v, int r)
{
	return _mm_or_si128(_mm_slli_epi32(v, r), _mm_srli_epi32(v, 32 - r));
}
#endif

#ifdef SIMD_NEON
SIMD_TARGET
inline void transpose8(uint8x16_t& x0, uint8x16_t& x1, uint8x16_t& x2, uint8x16_t& x3)
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

SIMD_TARGET
inline uint8x16_t unzigzag8(uint8x16_t v)
{
	uint8x16_t xl = vreinterpretq_u8_s8(vnegq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(1)))));
	uint8x16_t xr = vshrq_n_u8(v, 1);

	return veorq_u8(xl, xr);
}

SIMD_TARGET
inline uint8x16_t unzigzag16(uint8x16_t v)
{
	uint16x8_t vv = vreinterpretq_u16_u8(v);
	uint8x16_t xl = vreinterpretq_u8_s16(vnegq_s16(vreinterpretq_s16_u16(vandq_u16(vv, vdupq_n_u16(1)))));
	uint8x16_t xr = vreinterpretq_u8_u16(vshrq_n_u16(vv, 1));

	return veorq_u8(xl, xr);
}

SIMD_TARGET
inline uint8x16_t rotate32(uint8x16_t v, int r)
{
	uint32x4_t v32 = vreinterpretq_u32_u8(v);
	return vreinterpretq_u8_u32(vorrq_u32(vshlq_u32(v32, vdupq_n_s32(r)), vshlq_u32(v32, vdupq_n_s32(r - 32))));
}

template <int Channel>
SIMD_TARGET inline uint8x8_t rebase(uint8x8_t npi, uint8x16_t r0, uint8x16_t r1, uint8x16_t r2, uint8x16_t r3)
{
	switch (Channel)
	{
	case 0:
	{
		uint8x16_t rsum = vaddq_u8(vaddq_u8(r0, r1), vaddq_u8(r2, r3));
		uint8x8_t rsumx = vadd_u8(vget_low_u8(rsum), vget_high_u8(rsum));
		return vadd_u8(vadd_u8(npi, rsumx), vext_u8(rsumx, rsumx, 4));
	}
	case 1:
	{
		uint16x8_t rsum = vaddq_u16(vaddq_u16(vreinterpretq_u16_u8(r0), vreinterpretq_u16_u8(r1)), vaddq_u16(vreinterpretq_u16_u8(r2), vreinterpretq_u16_u8(r3)));
		uint16x4_t rsumx = vadd_u16(vget_low_u16(rsum), vget_high_u16(rsum));
		return vreinterpret_u8_u16(vadd_u16(vadd_u16(vreinterpret_u16_u8(npi), rsumx), vext_u16(rsumx, rsumx, 2)));
	}
	case 2:
	{
		uint8x16_t rsum = veorq_u8(veorq_u8(r0, r1), veorq_u8(r2, r3));
		uint8x8_t rsumx = veor_u8(vget_low_u8(rsum), vget_high_u8(rsum));
		return veor_u8(veor_u8(npi, rsumx), vext_u8(rsumx, rsumx, 4));
	}
	default:
		return npi;
	}
}
#endif

#ifdef SIMD_WASM
SIMD_TARGET
inline void transpose8(v128_t& x0, v128_t& x1, v128_t& x2, v128_t& x3)
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
inline v128_t unzigzag8(v128_t v)
{
	v128_t xl = wasm_i8x16_neg(wasm_v128_and(v, wasm_i8x16_splat(1)));
	v128_t xr = wasm_u8x16_shr(v, 1);

	return wasm_v128_xor(xl, xr);
}

SIMD_TARGET
inline v128_t unzigzag16(v128_t v)
{
	v128_t xl = wasm_i16x8_neg(wasm_v128_and(v, wasm_i16x8_splat(1)));
	v128_t xr = wasm_u16x8_shr(v, 1);

	return wasm_v128_xor(xl, xr);
}

SIMD_TARGET
inline v128_t rotate32(v128_t v, int r)
{
	return wasm_v128_or(wasm_i32x4_shl(v, r), wasm_i32x4_shr(v, 32 - r));
}
#endif

#if defined(SIMD_SSE) || defined(SIMD_AVX) || defined(SIMD_NEON) || defined(SIMD_WASM)
SIMD_TARGET
static const unsigned char* decodeBytesSimd(const unsigned char* data, const unsigned char* data_end, unsigned char* buffer, size_t buffer_size, int hshift)
{
	assert(buffer_size % kByteGroupSize == 0);
	assert(kByteGroupSize == 16);

	// round number of groups to 4 to get number of header bytes
	size_t header_size = (buffer_size / kByteGroupSize + 3) / 4;
	if (size_t(data_end - data) < header_size)
		return NULL;

	const unsigned char* header = data;
	data += header_size;

	size_t i = 0;

	// fast-path: process 4 groups at a time, do a shared bounds check
	for (; i + kByteGroupSize * 4 <= buffer_size && size_t(data_end - data) >= kByteGroupDecodeLimit * 4; i += kByteGroupSize * 4)
	{
		size_t header_offset = i / kByteGroupSize;
		unsigned char header_byte = header[header_offset / 4];

		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 0, hshift + ((header_byte >> 0) & 3));
		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 1, hshift + ((header_byte >> 2) & 3));
		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 2, hshift + ((header_byte >> 4) & 3));
		data = decodeBytesGroupSimd(data, buffer + i + kByteGroupSize * 3, hshift + ((header_byte >> 6) & 3));
	}

	// slow-path: process remaining groups
	for (; i < buffer_size; i += kByteGroupSize)
	{
		if (size_t(data_end - data) < kByteGroupDecodeLimit)
			return NULL;

		size_t header_offset = i / kByteGroupSize;
		unsigned char header_byte = header[header_offset / 4];

		data = decodeBytesGroupSimd(data, buffer + i, hshift + ((header_byte >> ((header_offset % 4) * 2)) & 3));
	}

	return data;
}

template <int Channel>
SIMD_TARGET static void
decodeDeltas4Simd(const unsigned char* buffer, unsigned char* transposed, size_t vertex_count_aligned, size_t vertex_size, unsigned char last_vertex[4], int rot)
{
#if defined(SIMD_SSE) || defined(SIMD_AVX)
#define TEMP __m128i
#define PREP() __m128i pi = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(last_vertex))
#define LOAD(i) __m128i r##i = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buffer + j + i * vertex_count_aligned))
#define GRP4(i) t0 = r##i, t1 = _mm_shuffle_epi32(r##i, 1), t2 = _mm_shuffle_epi32(r##i, 2), t3 = _mm_shuffle_epi32(r##i, 3)
#define FIXD(i) t##i = pi = Channel == 0 ? _mm_add_epi8(pi, t##i) : (Channel == 1 ? _mm_add_epi16(pi, t##i) : _mm_xor_si128(pi, t##i))
#define SAVE(i) *reinterpret_cast<int*>(savep) = _mm_cvtsi128_si32(t##i), savep += vertex_size
#endif

#ifdef SIMD_NEON
#define TEMP uint8x8_t
#define PREP() uint8x8_t pi = vreinterpret_u8_u32(vld1_lane_u32(reinterpret_cast<uint32_t*>(last_vertex), vdup_n_u32(0), 0))
#define LOAD(i) uint8x16_t r##i = vld1q_u8(buffer + j + i * vertex_count_aligned)
#define GRP4(i) t0 = vget_low_u8(r##i), t1 = vreinterpret_u8_u32(vdup_lane_u32(vreinterpret_u32_u8(t0), 1)), t2 = vget_high_u8(r##i), t3 = vreinterpret_u8_u32(vdup_lane_u32(vreinterpret_u32_u8(t2), 1))
#define FIXD(i) t##i = pi = Channel == 0 ? vadd_u8(pi, t##i) : (Channel == 1 ? vreinterpret_u8_u16(vadd_u16(vreinterpret_u16_u8(pi), vreinterpret_u16_u8(t##i))) : veor_u8(pi, t##i))
#define SAVE(i) vst1_lane_u32(reinterpret_cast<uint32_t*>(savep), vreinterpret_u32_u8(t##i), 0), savep += vertex_size
#endif

#ifdef SIMD_WASM
#define TEMP v128_t
#define PREP() v128_t pi = wasm_v128_load(last_vertex)
#define LOAD(i) v128_t r##i = wasm_v128_load(buffer + j + i * vertex_count_aligned)
#define GRP4(i) t0 = r##i, t1 = wasmx_splat_v32x4(r##i, 1), t2 = wasmx_splat_v32x4(r##i, 2), t3 = wasmx_splat_v32x4(r##i, 3)
#define FIXD(i) t##i = pi = Channel == 0 ? wasm_i8x16_add(pi, t##i) : (Channel == 1 ? wasm_i16x8_add(pi, t##i) : wasm_v128_xor(pi, t##i))
#define SAVE(i) wasm_v128_store32_lane(savep, t##i, 0), savep += vertex_size
#endif

#define UNZR(i) r##i = Channel == 0 ? unzigzag8(r##i) : (Channel == 1 ? unzigzag16(r##i) : rotate32(r##i, rot))

	PREP();

	unsigned char* savep = transposed;

	for (size_t j = 0; j < vertex_count_aligned; j += 16)
	{
		LOAD(0);
		LOAD(1);
		LOAD(2);
		LOAD(3);

		transpose8(r0, r1, r2, r3);

		TEMP t0, t1, t2, t3;
		TEMP npi = pi;

		UNZR(0);
		GRP4(0);
		FIXD(0), FIXD(1), FIXD(2), FIXD(3);
		SAVE(0), SAVE(1), SAVE(2), SAVE(3);

		UNZR(1);
		GRP4(1);
		FIXD(0), FIXD(1), FIXD(2), FIXD(3);
		SAVE(0), SAVE(1), SAVE(2), SAVE(3);

		UNZR(2);
		GRP4(2);
		FIXD(0), FIXD(1), FIXD(2), FIXD(3);
		SAVE(0), SAVE(1), SAVE(2), SAVE(3);

		UNZR(3);
		GRP4(3);
		FIXD(0), FIXD(1), FIXD(2), FIXD(3);
		SAVE(0), SAVE(1), SAVE(2), SAVE(3);

#if defined(SIMD_LATENCYOPT) && defined(SIMD_NEON) && (defined(__APPLE__) || defined(_WIN32))
		// instead of relying on accumulated pi, recompute it from scratch from r0..r3; this shortens dependency between loop iterations
		pi = rebase<Channel>(npi, r0, r1, r2, r3);
#else
		(void)npi;
#endif

#undef UNZR
#undef TEMP
#undef PREP
#undef LOAD
#undef GRP4
#undef FIXD
#undef SAVE
	}
}

SIMD_TARGET
static const unsigned char* decodeVertexBlockSimd(const unsigned char* data, const unsigned char* data_end, unsigned char* vertex_data, size_t vertex_count, size_t vertex_size, unsigned char last_vertex[256], const unsigned char* channels, int version)
{
	assert(vertex_count > 0 && vertex_count <= kVertexBlockMaxSize);

	unsigned char buffer[kVertexBlockMaxSize * 4];
	unsigned char transposed[kVertexBlockSizeBytes];

	size_t vertex_count_aligned = (vertex_count + kByteGroupSize - 1) & ~(kByteGroupSize - 1);

	size_t control_size = version == 0 ? 0 : vertex_size / 4;
	if (size_t(data_end - data) < control_size)
		return NULL;

	const unsigned char* control = data;
	data += control_size;

	for (size_t k = 0; k < vertex_size; k += 4)
	{
		unsigned char ctrl_byte = version == 0 ? 0 : control[k / 4];

		for (size_t j = 0; j < 4; ++j)
		{
			int ctrl = (ctrl_byte >> (j * 2)) & 3;

			if (ctrl == 3)
			{
				// literal encoding; safe to over-copy due to tail
				if (size_t(data_end - data) < vertex_count_aligned)
					return NULL;

				memcpy(buffer + j * vertex_count_aligned, data, vertex_count_aligned);
				data += vertex_count;
			}
			else if (ctrl == 2)
			{
				// zero encoding
				memset(buffer + j * vertex_count_aligned, 0, vertex_count_aligned);
			}
			else
			{
				// for v0, headers are mapped to 0..3; for v1, headers are mapped to 4..8
				int hshift = version == 0 ? 0 : 4 + ctrl;

				data = decodeBytesSimd(data, data_end, buffer + j * vertex_count_aligned, vertex_count_aligned, hshift);
				if (!data)
					return NULL;
			}
		}

		int channel = version == 0 ? 0 : channels[k / 4];

		switch (channel & 3)
		{
		case 0:
			decodeDeltas4Simd<0>(buffer, transposed + k, vertex_count_aligned, vertex_size, last_vertex + k, 0);
			break;
		case 1:
			decodeDeltas4Simd<1>(buffer, transposed + k, vertex_count_aligned, vertex_size, last_vertex + k, 0);
			break;
		case 2:
			decodeDeltas4Simd<2>(buffer, transposed + k, vertex_count_aligned, vertex_size, last_vertex + k, (32 - (channel >> 4)) & 31);
			break;
		default:
			return NULL; // invalid channel type
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

size_t meshopt_encodeVertexBufferLevel(unsigned char* buffer, size_t buffer_size, const void* vertices, size_t vertex_count, size_t vertex_size, int level, int version)
{
	using namespace meshopt;

	assert(vertex_size > 0 && vertex_size <= 256);
	assert(vertex_size % 4 == 0);
	assert(level >= 0 && level <= 9); // only a subset of this range is used right now
	assert(version < 0 || unsigned(version) <= kDecodeVertexVersion);

	version = version < 0 ? gEncodeVertexVersion : version;

#if TRACE
	memset(vertexstats, 0, sizeof(vertexstats));
#endif

	const unsigned char* vertex_data = static_cast<const unsigned char*>(vertices);

	unsigned char* data = buffer;
	unsigned char* data_end = buffer + buffer_size;

	if (size_t(data_end - data) < 1)
		return 0;

	*data++ = (unsigned char)(kVertexHeader | version);

	unsigned char first_vertex[256] = {};
	if (vertex_count > 0)
		memcpy(first_vertex, vertex_data, vertex_size);

	unsigned char last_vertex[256] = {};
	memcpy(last_vertex, first_vertex, vertex_size);

	size_t vertex_block_size = getVertexBlockSize(vertex_size);

	unsigned char channels[64] = {};
	if (version != 0 && level > 1 && vertex_count > 1)
		for (size_t k = 0; k < vertex_size; k += 4)
		{
			int rot = level >= 3 ? estimateRotate(vertex_data, vertex_count, vertex_size, k, /* group_size= */ 16) : 0;
			int channel = estimateChannel(vertex_data, vertex_count, vertex_size, k, vertex_block_size, /* block_skip= */ 3, /* max_channels= */ level >= 3 ? 3 : 2, rot);

			assert(unsigned(channel) < 2 || ((channel & 3) == 2 && unsigned(channel >> 4) < 8));
			channels[k / 4] = (unsigned char)channel;
		}

	size_t vertex_offset = 0;

	while (vertex_offset < vertex_count)
	{
		size_t block_size = (vertex_offset + vertex_block_size < vertex_count) ? vertex_block_size : vertex_count - vertex_offset;

		data = encodeVertexBlock(data, data_end, vertex_data + vertex_offset * vertex_size, block_size, vertex_size, last_vertex, channels, version, level);
		if (!data)
			return 0;

		vertex_offset += block_size;
	}

	size_t tail_size = vertex_size + (version == 0 ? 0 : vertex_size / 4);
	size_t tail_size_min = version == 0 ? kTailMinSizeV0 : kTailMinSizeV1;
	size_t tail_size_pad = tail_size < tail_size_min ? tail_size_min : tail_size;

	if (size_t(data_end - data) < tail_size_pad)
		return 0;

	if (tail_size < tail_size_pad)
	{
		memset(data, 0, tail_size_pad - tail_size);
		data += tail_size_pad - tail_size;
	}

	memcpy(data, first_vertex, vertex_size);
	data += vertex_size;

	if (version != 0)
	{
		memcpy(data, channels, vertex_size / 4);
		data += vertex_size / 4;
	}

	assert(data >= buffer + tail_size);
	assert(data <= buffer + buffer_size);

#if TRACE
	size_t total_size = data - buffer;

	for (size_t k = 0; k < vertex_size; ++k)
	{
		const Stats& vsk = vertexstats[k];

		printf("%2d: %7d bytes [%4.1f%%] %.1f bpv", int(k), int(vsk.size), double(vsk.size) / double(total_size) * 100, double(vsk.size) / double(vertex_count) * 8);

		size_t total_k = vsk.header + vsk.bitg[1] + vsk.bitg[2] + vsk.bitg[4] + vsk.bitg[8];
		double total_kr = total_k ? 1.0 / double(total_k) : 0;

		if (version != 0)
		{
			int channel = channels[k / 4];

			if ((channel & 3) == 2 && k % 4 == 0)
				printf(" | ^%d", channel >> 4);
			else
				printf(" | %2s", channel == 0 ? "1" : (channel == 1 && k % 2 == 0 ? "2" : "."));
		}

		printf(" | hdr [%5.1f%%] bitg [1 %4.1f%% 2 %4.1f%% 4 %4.1f%% 8 %4.1f%%]",
		    double(vsk.header) * total_kr * 100,
		    double(vsk.bitg[1]) * total_kr * 100, double(vsk.bitg[2]) * total_kr * 100,
		    double(vsk.bitg[4]) * total_kr * 100, double(vsk.bitg[8]) * total_kr * 100);

		size_t total_ctrl = vsk.ctrl[0] + vsk.ctrl[1] + vsk.ctrl[2] + vsk.ctrl[3];

		if (total_ctrl)
		{
			printf(" | ctrl %3.0f%% %3.0f%% %3.0f%% %3.0f%%",
			    double(vsk.ctrl[0]) / double(total_ctrl) * 100, double(vsk.ctrl[1]) / double(total_ctrl) * 100,
			    double(vsk.ctrl[2]) / double(total_ctrl) * 100, double(vsk.ctrl[3]) / double(total_ctrl) * 100);
		}

		if (level >= 3)
			printf(" | bitc [%3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%%]",
			    double(vsk.bitc[0]) / double(vertex_count) * 100, double(vsk.bitc[1]) / double(vertex_count) * 100,
			    double(vsk.bitc[2]) / double(vertex_count) * 100, double(vsk.bitc[3]) / double(vertex_count) * 100,
			    double(vsk.bitc[4]) / double(vertex_count) * 100, double(vsk.bitc[5]) / double(vertex_count) * 100,
			    double(vsk.bitc[6]) / double(vertex_count) * 100, double(vsk.bitc[7]) / double(vertex_count) * 100);

		printf("\n");
	}
#endif

	return data - buffer;
}

size_t meshopt_encodeVertexBuffer(unsigned char* buffer, size_t buffer_size, const void* vertices, size_t vertex_count, size_t vertex_size)
{
	return meshopt_encodeVertexBufferLevel(buffer, buffer_size, vertices, vertex_count, vertex_size, meshopt::kEncodeDefaultLevel, meshopt::gEncodeVertexVersion);
}

size_t meshopt_encodeVertexBufferBound(size_t vertex_count, size_t vertex_size)
{
	using namespace meshopt;

	assert(vertex_size > 0 && vertex_size <= 256);
	assert(vertex_size % 4 == 0);

	size_t vertex_block_size = getVertexBlockSize(vertex_size);
	size_t vertex_block_count = (vertex_count + vertex_block_size - 1) / vertex_block_size;

	size_t vertex_block_control_size = vertex_size / 4;
	size_t vertex_block_header_size = (vertex_block_size / kByteGroupSize + 3) / 4;
	size_t vertex_block_data_size = vertex_block_size;

	size_t tail_size = vertex_size + (vertex_size / 4);
	size_t tail_size_min = kTailMinSizeV0 > kTailMinSizeV1 ? kTailMinSizeV0 : kTailMinSizeV1;
	size_t tail_size_pad = tail_size < tail_size_min ? tail_size_min : tail_size;
	assert(tail_size_pad >= kByteGroupDecodeLimit);

	return 1 + vertex_block_count * vertex_size * (vertex_block_control_size + vertex_block_header_size + vertex_block_data_size) + tail_size_pad;
}

void meshopt_encodeVertexVersion(int version)
{
	assert(unsigned(version) <= unsigned(meshopt::kDecodeVertexVersion));

	meshopt::gEncodeVertexVersion = version;
}

int meshopt_decodeVertexVersion(const unsigned char* buffer, size_t buffer_size)
{
	if (buffer_size < 1)
		return -1;

	unsigned char header = buffer[0];

	if ((header & 0xf0) != meshopt::kVertexHeader)
		return -1;

	int version = header & 0x0f;
	if (version > meshopt::kDecodeVertexVersion)
		return -1;

	return version;
}

int meshopt_decodeVertexBuffer(void* destination, size_t vertex_count, size_t vertex_size, const unsigned char* buffer, size_t buffer_size)
{
	using namespace meshopt;

	assert(vertex_size > 0 && vertex_size <= 256);
	assert(vertex_size % 4 == 0);

	const unsigned char* (*decode)(const unsigned char*, const unsigned char*, unsigned char*, size_t, size_t, unsigned char[256], const unsigned char*, int) = NULL;

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

	if (size_t(data_end - data) < 1)
		return -2;

	unsigned char data_header = *data++;

	if ((data_header & 0xf0) != kVertexHeader)
		return -1;

	int version = data_header & 0x0f;
	if (version > kDecodeVertexVersion)
		return -1;

	size_t tail_size = vertex_size + (version == 0 ? 0 : vertex_size / 4);
	size_t tail_size_min = version == 0 ? kTailMinSizeV0 : kTailMinSizeV1;
	size_t tail_size_pad = tail_size < tail_size_min ? tail_size_min : tail_size;

	if (size_t(data_end - data) < tail_size_pad)
		return -2;

	const unsigned char* tail = data_end - tail_size;

	unsigned char last_vertex[256];
	memcpy(last_vertex, tail, vertex_size);

	const unsigned char* channels = version == 0 ? NULL : tail + vertex_size;

	size_t vertex_block_size = getVertexBlockSize(vertex_size);

	size_t vertex_offset = 0;

	while (vertex_offset < vertex_count)
	{
		size_t block_size = (vertex_offset + vertex_block_size < vertex_count) ? vertex_block_size : vertex_count - vertex_offset;

		data = decode(data, data_end, vertex_data + vertex_offset * vertex_size, block_size, vertex_size, last_vertex, channels, version);
		if (!data)
			return -2;

		vertex_offset += block_size;
	}

	if (size_t(data_end - data) != tail_size_pad)
		return -3;

	return 0;
}

#undef SIMD_NEON
#undef SIMD_SSE
#undef SIMD_AVX
#undef SIMD_WASM
#undef SIMD_FALLBACK
#undef SIMD_TARGET
#undef SIMD_LATENCYOPT
