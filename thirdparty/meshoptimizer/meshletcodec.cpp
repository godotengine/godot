// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

// The block below auto-detects SIMD ISA that can be used on the target platform
#ifndef MESHOPTIMIZER_NO_SIMD

// The SIMD implementation requires SSE4.1, which can be enabled unconditionally through compiler settings
#if defined(__AVX__) || defined(__SSE4_1__)
#define SIMD_SSE
#endif

// MSVC supports compiling SSE4.1 code regardless of compile options; we use a cpuid-based scalar fallback
#if !defined(SIMD_SSE) && defined(_MSC_VER) && !defined(__clang__) && (defined(_M_IX86) || (defined(_M_X64) && !defined(_M_ARM64EC)))
#define SIMD_SSE
#define SIMD_FALLBACK
#endif

// GCC 4.9+ and clang 3.8+ support targeting SIMD ISA from individual functions; we use a cpuid-based scalar fallback
#if !defined(SIMD_SSE) && ((defined(__clang__) && __clang_major__ * 100 + __clang_minor__ >= 308) || (defined(__GNUC__) && __GNUC__ * 100 + __GNUC_MINOR__ >= 409)) && (defined(__i386__) || defined(__x86_64__))
#define SIMD_SSE
#define SIMD_FALLBACK
#define SIMD_TARGET __attribute__((target("sse4.1")))
#endif

// When targeting AArch64, enable NEON SIMD unconditionally; we do not support SIMD decoding for 32-bit ARM
#if defined(__aarch64__) || (defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC)) && _MSC_VER >= 1922)
#define SIMD_NEON
#endif

#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER > 1930
#define SIMD_FLATTEN [[msvc::flatten]]
#elif defined(__GNUC__) || defined(__clang__)
#define SIMD_FLATTEN __attribute__((flatten))
#else
#define SIMD_FLATTEN
#endif

#ifndef SIMD_TARGET
#define SIMD_TARGET
#endif

#endif // !MESHOPTIMIZER_NO_SIMD

#ifdef SIMD_SSE
#include <smmintrin.h>
#endif

#ifdef SIMD_NEON
#include <arm_neon.h>
#endif

#if defined(SIMD_SSE) && defined(SIMD_FALLBACK)
#ifdef _MSC_VER
#include <intrin.h> // __cpuid
#else
#include <cpuid.h> // __cpuid
#endif
#endif

#ifndef TRACE
#define TRACE 0
#endif

#if TRACE
#include <stdio.h>
#endif

namespace meshopt
{

typedef unsigned int EdgeFifo8[8][2];

static int rotateTriangle(unsigned int a, unsigned int b, unsigned int c)
{
	return (a > b && a > c) ? 1 : (b > c ? 2 : 0);
}

static int getEdgeFifo8(EdgeFifo8 fifo, unsigned int a, unsigned int b, unsigned int c, size_t offset)
{
	for (int i = 0; i < 8; ++i)
	{
		size_t index = (offset - 1 - i) & 7;

		unsigned int e0 = fifo[index][0];
		unsigned int e1 = fifo[index][1];

		if (e0 == a && e1 == b)
			return (i << 2) | 0;
		if (e0 == b && e1 == c)
			return (i << 2) | 1;
		if (e0 == c && e1 == a)
			return (i << 2) | 2;
	}

	return -1;
}

static void pushEdgeFifo8(EdgeFifo8 fifo, unsigned int a, unsigned int b, size_t& offset)
{
	fifo[offset][0] = a;
	fifo[offset][1] = b;
	offset = (offset + 1) & 7;
}

static size_t encodeTriangles(unsigned char* codes, unsigned char* extra, const unsigned char* triangles, size_t triangle_count)
{
	EdgeFifo8 edgefifo;
	memset(edgefifo, -1, sizeof(edgefifo));

	size_t edgefifooffset = 0;

	unsigned int next = 0;

	// 4-bit triangle codes give us 16 options that we use as follows:
	// 3*2 edge reuse (2 edges * 3 last triangles) * 2 next/explicit = 12 options
	// 4 remaining options = next bits; 000, 001, 011, 111.
	// triangles are rotated to make next bits line up.
	memset(codes, 0, (triangle_count + 1) / 2);

	static const int rotations[] = {0, 1, 2, 0, 1};

	unsigned char* start = extra;

	for (size_t i = 0; i < triangle_count; ++i)
	{
#if TRACE > 1
		unsigned int last = next;
#endif

		int fer = getEdgeFifo8(edgefifo, triangles[i * 3 + 0], triangles[i * 3 + 1], triangles[i * 3 + 2], edgefifooffset);

		if (fer >= 0 && (fer >> 2) < 6)
		{
			// note: getEdgeFifo8 implicitly rotates triangles by matching a/b to existing edge
			const int* order = rotations + (fer & 3);

			unsigned int a = triangles[i * 3 + order[0]], b = triangles[i * 3 + order[1]], c = triangles[i * 3 + order[2]];

			int fec = (c == next) ? (next++, 0) : 1;

#if TRACE > 1
			printf("%3d+ | %3d %3d %3d | edge: e%d c%d\n", last, a, b, c, fer >> 2, fec);
#endif

			unsigned int code = (fer >> 2) * 2 + fec;

			codes[i / 2] |= (unsigned char)(code << ((i & 1) * 4));

			if (fec)
				*extra++ = (unsigned char)c;

			pushEdgeFifo8(edgefifo, c, b, edgefifooffset);
			pushEdgeFifo8(edgefifo, a, c, edgefifooffset);
		}
		else
		{
			// rotate triangles to minimize the need for extra vertices
			int rotation = rotateTriangle(triangles[i * 3 + 0], triangles[i * 3 + 1], triangles[i * 3 + 2]);
			const int* order = rotations + rotation;

			unsigned int a = triangles[i * 3 + order[0]], b = triangles[i * 3 + order[1]], c = triangles[i * 3 + order[2]];

			// fe must be continuous: once a vertex is encoded with next, further vertices must also be encoded with next
			int fea = (a == next && b == next + 1 && c == next + 2) ? (next++, 0) : 1;
			int feb = (b == next && c == next + 1) ? (next++, 0) : 1;
			int fec = (c == next) ? (next++, 0) : 1;

			assert(fea == 1 || feb == 0);
			assert(feb == 1 || fec == 0);

#if TRACE > 1
			printf("%3d+ | %3d %3d %3d | restart: %d%d%d\n", last, a, b, c, fea, feb, fec);
#endif

			unsigned int code = 12 + (fea + feb + fec);

			codes[i / 2] |= (unsigned char)(code << ((i & 1) * 4));

			if (fea)
				*extra++ = (unsigned char)a;
			if (feb)
				*extra++ = (unsigned char)b;
			if (fec)
				*extra++ = (unsigned char)c;

			pushEdgeFifo8(edgefifo, c, b, edgefifooffset);
			pushEdgeFifo8(edgefifo, a, c, edgefifooffset);
		}
	}

	return extra - start;
}

static size_t encodeVertices(unsigned char* ctrl, unsigned char* data, const unsigned int* vertices, size_t vertex_count)
{
	// grouped varint, 2 bit per value to indicate 0/1/2/3 byte deltas, with per-group 4-byte fallback
	memset(ctrl, 0, (vertex_count + 3) / 4);

	unsigned char* start = data;

	unsigned int last = ~0u;

	for (size_t i = 0; i < vertex_count; i += 4)
	{
		unsigned int gv[4] = {};

		for (int k = 0; k < 4 && i + k < vertex_count; ++k)
		{
			unsigned int d = vertices[i + k] - last - 1;
			unsigned int v = (d << 1) ^ (int(d) >> 31);

			gv[k] = v;
			last = vertices[i + k];
		}

		// if any value needs 4 bytes, or if *all* values need 3 bytes, we use 4 bytes for all values
		// this allows us to encode most 3-byte deltas with 3 bytes which saves space overall
		bool use4 = (gv[0] | gv[1] | gv[2] | gv[3]) > 0xffffff || (gv[0] > 0xffff && gv[1] > 0xffff && gv[2] > 0xffff && gv[3] > 0xffff);

		for (int k = 0; k < 4; ++k)
		{
			unsigned int v = gv[k];

			// 0/1/2/3 bytes per value, or all 4 values use 4 bytes
			int code = use4 ? 3 : (v == 0 ? 0 : (v < 256 ? 1 : (v < 65536 ? 2 : 3)));

			if (code > 0)
				*data++ = (unsigned char)(v & 0xff);
			if (code > 1)
				*data++ = (unsigned char)((v >> 8) & 0xff);
			if (code > 2)
				*data++ = (unsigned char)((v >> 16) & 0xff);
			if (use4)
				*data++ = (unsigned char)((v >> 24) & 0xff);

			// split low and high bits into two nibbles for better packing
			ctrl[i / 4] |= ((code & 1) << k) | ((code >> 1) << (k + 4));
		}
	}

	return data - start;
}

#if defined(SIMD_FALLBACK) || (!defined(SIMD_SSE) && !defined(SIMD_NEON))
inline void writeTriangle(unsigned int* triangles, size_t i, unsigned int fifo)
{
	// output triangle is stored without extra edge vertex (0xcbac => 0xcba)
	triangles[i] = fifo >> 8;
}

inline void writeTriangle(unsigned char* triangles, size_t i, unsigned int fifo)
{
	triangles[i * 3 + 0] = (unsigned char)(fifo >> 8);
	triangles[i * 3 + 1] = (unsigned char)(fifo >> 16);
	triangles[i * 3 + 2] = (unsigned char)(fifo >> 24);
}

template <typename T>
static const unsigned char* decodeTriangles(T* triangles, const unsigned char* codes, const unsigned char* extra, const unsigned char* bound, size_t triangle_count)
{
	// branchlessly read next or extra vertex and advance pointers
#define NEXT(var, ec) \
	e = *extra; \
	unsigned int var = (ec) ? e : next; \
	extra += (ec), next += 1 - (ec)

	unsigned int next = 0;
	unsigned int fifo[3] = {}; // two edge fifo entries in one uint: 0xcbac

	for (size_t i = 0; i < triangle_count; ++i)
	{
		if (extra > bound)
			return NULL;

		unsigned int code = (codes[i / 2] >> ((i & 1) * 4)) & 0xF;
		unsigned int tri;

		if (code < 12)
		{
			// reuse
			unsigned int edge = fifo[code / 4];
			edge >>= (code << 3) & 16; // shift by 16 if bit 1 is set (odd edge for each triangle)

			// 0-1 extra vertices
			unsigned int e;
			NEXT(c, code & 1);

			// repack triangle into edge format (0xcbac)
			tri = ((edge & 0xff) << 16) | (edge & 0xff00) | c | (c << 24);
		}
		else
		{
			// restart
			int fea = code > 12;
			int feb = code > 13;
			int fec = code > 14;

			// 0-3 extra vertices
			unsigned int e;
			NEXT(a, fea);
			NEXT(b, feb);
			NEXT(c, fec);

			// repack triangle into edge format (0xcbac)
			tri = c | (a << 8) | (b << 16) | (c << 24);
		}

		writeTriangle(triangles, i, tri);

		fifo[2] = fifo[1];
		fifo[1] = fifo[0];
		fifo[0] = tri;
	}

	return extra;

#undef NEXT
}

template <typename V>
static const unsigned char* decodeVertices(V* vertices, const unsigned char* ctrl, const unsigned char* data, const unsigned char* bound, size_t vertex_count)
{
	unsigned int last = ~0u;

	for (size_t i = 0; i < vertex_count; i += 4)
	{
		if (data > bound)
			return NULL;

		unsigned char code4 = ctrl[i / 4];

		for (int k = 0; k < 4; ++k)
		{
			int code = ((code4 >> k) & 1) | ((code4 >> (k + 3)) & 2);
			int length = code4 == 0xff ? 4 : code;

			// branchlessly read up to 4 bytes
			unsigned int mask = (length == 4) ? ~0u : (1 << (8 * length)) - 1;
			unsigned int v = (data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)) & mask;

			// unzigzag + 1
			unsigned int d = (v >> 1) ^ -int(v & 1);
			unsigned int r = last + d + 1;

			if (i + k < vertex_count)
				vertices[i + k] = V(r);

			data += length;
			last = r;
		}
	}

	return data;
}

static int decodeMeshlet(void* vertices, void* triangles, const unsigned char* codes, const unsigned char* ctrl, const unsigned char* data, const unsigned char* bound, size_t vertex_count, size_t triangle_count, size_t vertex_size, size_t triangle_size)
{
	if (vertex_size == 4)
		data = decodeVertices(static_cast<unsigned int*>(vertices), ctrl, data, bound, vertex_count);
	else
		data = decodeVertices(static_cast<unsigned short*>(vertices), ctrl, data, bound, vertex_count);
	if (!data)
		return -2;

	if (triangle_size == 4)
		data = decodeTriangles(static_cast<unsigned int*>(triangles), codes, data, bound, triangle_count);
	else
		data = decodeTriangles(static_cast<unsigned char*>(triangles), codes, data, bound, triangle_count);
	if (!data)
		return -2;

	return (data == bound) ? 0 : -3;
}
#endif

#if defined(SIMD_SSE) || defined(SIMD_NEON)
// SIMD state is stored in a single 16b register as follows:
// 0..5: 6 next extra bytes
// 6..14: 9 bytes = 3 triangles worth of index data
// 15: 'next' byte

// upon reading each triangle pair we need to transform this state such that the 9 bytes with triangle data contain the newly decoded triangles,
// which is a permutation of original state modulo per-element additions
// this transform can be chained to decode second triangle from original state; we create tables for 256 combinations of two 4-bit triangle codes
// the actual decoding becomes shuffle+add per triangle pair, plus management of extra bytes
static unsigned char kDecodeTableMasks[256][16];
static unsigned char kDecodeTableExtra[256];

// for SIMD vertex decoding we need to unpack 4 values with 0-4 bytes in each
// this can be done with a single control-dependent shuffle per group
static unsigned char kDecodeTableVerts[256][16];
static unsigned char kDecodeTableLength[256];

static bool decodeBuildTables()
{
#define NEXT(var, ec) \
	shuf[var] = (ec) ? (unsigned char)extra : 15; \
	next[var] = (ec) ? 0 : (unsigned char)nextoff; \
	extra += (ec), nextoff += 1 - (ec)

	// check for SSE4.1 support if we have a fallback path
#if defined(SIMD_SSE) && defined(SIMD_FALLBACK)
	int cpuinfo[4] = {};
#ifdef _MSC_VER
	__cpuid(cpuinfo, 1);
#else
	__cpuid(1, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
	// bit 19 = SSE4.1
	if ((cpuinfo[2] & (1 << 19)) == 0)
		return false;
#endif

	// fill triangle decoding tables for each combination of two triangle codes
	for (int code = 0; code < 256; ++code)
	{
		unsigned char shuf[16] = {};
		unsigned char next[16] = {};
		int extra = 0;
		int nextoff = 0;

		// state 0..5 will be refilled every iteration, so we ignore that
		// state 6..8 will always contain the last decoded triangle because every triangle shifts fifo equally, so we can decode it independently
		shuf[6] = 12;
		shuf[7] = 13;
		shuf[8] = 14;

		// state 15 will contain next (potentially incremented a few times)
		shuf[15] = 15;

		// state 9..11 will contain the first decoded triangle (tri0), which can refer to extra/next and the original triangle history
		// state 12..14 will contain the second decoded triangle (tri1); when decoding edge reuse, we need to handle edge 0/1 specially as it was just decoded earlier
		for (int k = 0; k < 2; ++k)
		{
			int tri = (code >> (k * 4)) & 0xf;

			if (tri < 12)
			{
				if (k == 1 && tri / 4 == 0)
				{
					// we need to decode one of two edges from the triangle we just decoded earlier
					// for that we simply need to copy shuf/next values for the two decoded indices
					shuf[9 + k * 3] = shuf[9 + ((tri & 2) ? 2 : 0)];
					next[9 + k * 3] = next[9 + ((tri & 2) ? 2 : 0)];

					shuf[10 + k * 3] = shuf[9 + ((tri & 2) ? 1 : 2)];
					next[10 + k * 3] = next[9 + ((tri & 2) ? 1 : 2)];
				}
				else
				{
					// reuse: edge comes from the history based on edge index
					// note: we reuse with an offset because last triangle in the original history was consumed by tri0
					int trioff = 6 + k * 3 + (2 - tri / 4) * 3;

					// edge cb or ac
					shuf[9 + k * 3] = (unsigned char)(trioff + ((tri & 2) ? 2 : 0));
					shuf[10 + k * 3] = (unsigned char)(trioff + ((tri & 2) ? 1 : 2));
				}

				// third vertex is either next or comes from extra
				NEXT(11 + k * 3, tri & 1);
			}
			else
			{
				// restart: three vertices, each comes from next or extra
				int fea = tri > 12;
				int feb = tri > 13;
				int fec = tri > 14;

				NEXT(9 + k * 3, fea);
				NEXT(10 + k * 3, feb);
				NEXT(11 + k * 3, fec);
			}
		}

		// next needs to advance
		next[15] = (unsigned char)nextoff;

		// next[0..8] = 0 trivially (never written to); next[9] must also be 0 because nextoff is 0 initially
		// shuf[0..5] is not used, which allows us to pack next[10..15] + shuf[6..15] into a single 16-byte entry
		assert(next[9] == 0);
		memcpy(&kDecodeTableMasks[code][0], &next[10], 6);
		memcpy(&kDecodeTableMasks[code][6], &shuf[6], 10);
		kDecodeTableExtra[code] = (unsigned char)extra;
	}

	// fill vertex decoding tables for each combination of four vertex references
	for (unsigned int i = 0; i < 256; ++i)
	{
		unsigned char shuf[16] = {};
		int offset = 0;

		for (int k = 0; k < 4; ++k)
		{
			int code = ((i >> k) & 1) | ((i >> (k + 3)) & 2);
			int length = i == 0xff ? 4 : code; // 0/1/2/3 bytes, or all 4 bytes if code==0xff

			shuf[k * 4 + 0] = (length > 0) ? (unsigned char)(offset + 0) : 0x80;
			shuf[k * 4 + 1] = (length > 1) ? (unsigned char)(offset + 1) : 0x80;
			shuf[k * 4 + 2] = (length > 2) ? (unsigned char)(offset + 2) : 0x80;
			shuf[k * 4 + 3] = (length > 3) ? (unsigned char)(offset + 3) : 0x80;

			offset += length;
		}

		memcpy(kDecodeTableVerts[i], shuf, sizeof(shuf));
		kDecodeTableLength[i] = (unsigned char)offset;
	}

	return true;

#undef NEXT
}

static bool gDecodeTablesInitialized = decodeBuildTables();
#endif

#if defined(SIMD_SSE)
SIMD_TARGET
inline __m128i decodeTriangleGroup(__m128i state, unsigned char code, const unsigned char*& extra)
{
	__m128i shuf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kDecodeTableMasks[code]));
	__m128i next = _mm_slli_si128(shuf, 10);

	// patch first 6 bytes with current extra and roll state forward
	__m128i ext = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(extra));
	state = _mm_blend_epi16(state, ext, 7);
	state = _mm_add_epi8(_mm_shuffle_epi8(state, shuf), next);

	extra += kDecodeTableExtra[code];

	return state;
}

SIMD_TARGET
inline __m128i decodeVertexGroup(__m128i last, unsigned char code, const unsigned char*& data)
{
	__m128i word = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
	__m128i shuf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kDecodeTableVerts[code]));

	__m128i v = _mm_shuffle_epi8(word, shuf);

	// unzigzag+1
	__m128i xl = _mm_sub_epi32(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi32(1)));
	__m128i xr = _mm_srli_epi32(v, 1);
	__m128i x = _mm_add_epi32(_mm_xor_si128(xl, xr), _mm_set1_epi32(1));

	// prefix sum
	x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
	x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
	x = _mm_add_epi32(x, _mm_shuffle_epi32(last, 0xff));

	data += kDecodeTableLength[code];

	return x;
}
#endif

#if defined(SIMD_NEON)
SIMD_TARGET
inline uint8x16_t decodeTriangleGroup(uint8x16_t state, unsigned char code, const unsigned char*& extra)
{
	uint8x16_t shuf = vld1q_u8(kDecodeTableMasks[code]);
	uint8x16_t next = vextq_u8(vdupq_n_u8(0), shuf, 6);

	// patch first 6 bytes with current extra and roll state forward
	uint8x8_t extl = vld1_u8(extra);
	uint8x16_t ext = vcombine_u8(extl, vdup_n_u8(0));
	state = vbslq_u8(vcombine_u8(vcreate_u8(0xffffffffffffull), vdup_n_u8(0)), ext, state);
	state = vaddq_u8(vqtbl1q_u8(state, shuf), next);

	extra += kDecodeTableExtra[code];

	return state;
}

SIMD_TARGET
inline uint32x4_t decodeVertexGroup(uint32x4_t last, unsigned char code, const unsigned char*& data)
{
	uint8x16_t word = vld1q_u8(data);
	uint8x16_t shuf = vld1q_u8(kDecodeTableVerts[code]);

	uint32x4_t v = vreinterpretq_u32_u8(vqtbl1q_u8(word, shuf));

	// unzigzag+1
	uint32x4_t xl = vsubq_u32(vdupq_n_u32(0), vandq_u32(v, vdupq_n_u32(1)));
	uint32x4_t xr = vshrq_n_u32(v, 1);
	uint32x4_t x = vaddq_u32(veorq_u32(xl, xr), vdupq_n_u32(1));

	// prefix sum
	x = vaddq_u32(x, vextq_u32(vdupq_n_u32(0), x, 2));
	x = vaddq_u32(x, vextq_u32(vdupq_n_u32(0), x, 3));
	x = vaddq_u32(x, vdupq_n_u32(vgetq_lane_u32(last, 3)));

	data += kDecodeTableLength[code];

	return x;
}
#endif

#if defined(SIMD_SSE)
#ifdef __GNUC__
typedef int __attribute__((aligned(1))) unaligned_int;
#else
typedef int unaligned_int;
#endif
#endif

#if defined(SIMD_SSE) || defined(SIMD_NEON)
SIMD_TARGET
static const unsigned char* decodeTrianglesSimd(unsigned int* triangles, const unsigned char* codes, const unsigned char* extra, const unsigned char* bound, size_t triangle_count)
{
#if defined(SIMD_SSE)
	__m128i repack = _mm_setr_epi8(9, 10, 11, -1, 12, 13, 14, -1, 0, 0, 0, 0, 0, 0, 0, 0);
	__m128i state = _mm_setzero_si128();
#elif defined(SIMD_NEON)
	uint8x8_t repack = vcreate_u8(0xff0e0d0cff0b0a09ull);
	uint8x16_t state = vdupq_n_u8(0);
#endif

	size_t groups = triangle_count / 2;

	// process all complete groups
	for (size_t i = 0; i < groups; ++i)
	{
		unsigned char code = *codes++;

		if (extra > bound)
			return NULL;

		state = decodeTriangleGroup(state, code, extra);

		// write 6 bytes of new triangle data into output, formatted as 8 bytes with 0 padding
#if defined(SIMD_SSE)
		__m128i r = _mm_shuffle_epi8(state, repack);
		_mm_storel_epi64(reinterpret_cast<__m128i*>(&triangles[i * 2]), r);
#elif defined(SIMD_NEON)
		uint32x2_t r = vreinterpret_u32_u8(vqtbl1_u8(state, repack));
		vst1_u32(&triangles[i * 2], r);
#endif
	}

	// process a 1 triangle tail; to maintain the memory safety guarantee we have to write a 32-bit element
	if (triangle_count & 1)
	{
		unsigned char code = *codes++;

		if (extra > bound)
			return NULL;

		state = decodeTriangleGroup(state, code, extra);

		unsigned int* tail = &triangles[triangle_count & ~1u];

#if defined(SIMD_SSE)
		__m128i r = _mm_shuffle_epi8(state, repack);
		*tail = unsigned(_mm_cvtsi128_si32(r));
#elif defined(SIMD_NEON)
		uint32x2_t r = vreinterpret_u32_u8(vqtbl1_u8(state, repack));
		vst1_lane_u32(tail, r, 0);
#endif
	}

	return extra;
}

SIMD_TARGET
static const unsigned char* decodeTrianglesSimd(unsigned char* triangles, const unsigned char* codes, const unsigned char* extra, const unsigned char* bound, size_t triangle_count)
{
#if defined(SIMD_SSE)
	__m128i state = _mm_setzero_si128();
#elif defined(SIMD_NEON)
	uint8x16_t state = vdupq_n_u8(0);
#endif

	// because the output buffer is guaranteed to have 32-bit aligned size available, we can optimize writes and tail processing
	// instead of processing triangles 2 at a time, we process 2 *pairs* at a time (12-byte write) followed by a tail pair, if present
	// if the number of triangles mod 4 is 3, we'd normally need to write 12k+9 bytes, but we can instead overwrite up to 3 bytes in the main loop
	size_t groups = (triangle_count + 1) / 4;

	// process all complete groups
	for (size_t i = 0; i < groups; ++i)
	{
		unsigned char code0 = *codes++;
		unsigned char code1 = *codes++;

		// each triangle pair reads <=6 bytes from extra, so two pairs need <=12 bytes and gap guarantees 16 byte of overread
		if (extra > bound)
			return NULL;

		state = decodeTriangleGroup(state, code0, extra);

		// write first decoded triangle and first index of second decoded triangle
#if defined(SIMD_SSE)
		__m128i r0 = _mm_srli_si128(state, 9);
		*reinterpret_cast<unaligned_int*>(&triangles[i * 12]) = _mm_cvtsi128_si32(r0);
#elif defined(SIMD_NEON)
		uint8x16_t r0 = vextq_u8(state, vdupq_n_u8(0), 9);
		vst1q_lane_u32(reinterpret_cast<unsigned int*>(&triangles[i * 12]), vreinterpretq_u32_u8(r0), 0);
#endif

		state = decodeTriangleGroup(state, code1, extra);

		// write last two indices of second decoded triangle that we didn't write above plus two new ones
		// note that the second decoded triangle has shifted down to 6-8 bytes, hence shift by 7
#if defined(SIMD_SSE)
		__m128i r1 = _mm_srli_si128(state, 7);
		_mm_storel_epi64(reinterpret_cast<__m128i*>(&triangles[i * 12 + 4]), r1);
#elif defined(SIMD_NEON)
		uint8x16_t r1 = vextq_u8(state, vdupq_n_u8(0), 7);
		vst1_u8(&triangles[i * 12 + 4], vget_low_u8(r1));
#endif
	}

	// process a 1-2 triangle tail; to maintain the memory safety guarantee we have to write 1-2 32-bit elements
	if (groups * 4 < triangle_count)
	{
		unsigned char code = *codes++;

		if (extra > bound)
			return NULL;

		state = decodeTriangleGroup(state, code, extra);

		unsigned char* tail = &triangles[(triangle_count & ~3u) * 3];

#if defined(SIMD_SSE)
		__m128i r = _mm_srli_si128(state, 9);

		*reinterpret_cast<unaligned_int*>(tail) = _mm_cvtsi128_si32(r);
		if ((triangle_count & 3) > 1)
			*reinterpret_cast<unaligned_int*>(tail + 4) = _mm_extract_epi32(r, 1);
#elif defined(SIMD_NEON)
		uint8x16_t r = vextq_u8(state, vdupq_n_u8(0), 9);

		vst1q_lane_u32(reinterpret_cast<unsigned int*>(tail), vreinterpretq_u32_u8(r), 0);
		if ((triangle_count & 3) > 1)
			vst1q_lane_u32(reinterpret_cast<unsigned int*>(tail + 4), vreinterpretq_u32_u8(r), 1);
#endif
	}

	return extra;
}

SIMD_TARGET
static const unsigned char* decodeVerticesSimd(unsigned int* vertices, const unsigned char* ctrl, const unsigned char* data, const unsigned char* bound, size_t vertex_count)
{
#if defined(SIMD_SSE)
	__m128i last = _mm_set1_epi32(-1);
#elif defined(SIMD_NEON)
	uint32x4_t last = vdupq_n_u32(~0u);
#endif

	size_t groups = vertex_count / 4;

	// process all complete groups
	for (size_t i = 0; i < groups; ++i)
	{
		unsigned char code = *ctrl++;
		if (data > bound)
			return NULL;

		last = decodeVertexGroup(last, code, data);

#if defined(SIMD_SSE)
		_mm_storeu_si128(reinterpret_cast<__m128i*>(&vertices[i * 4]), last);
#elif defined(SIMD_NEON)
		vst1q_u32(&vertices[i * 4], last);
#endif
	}

	// process a 1-3 vertex tail; to maintain the memory safety guarantee we have to write individual elements
	if (vertex_count & 3)
	{
		unsigned char code = *ctrl++;

		if (data > bound)
			return NULL;

		last = decodeVertexGroup(last, code, data);

		unsigned int* tail = &vertices[vertex_count & ~3u];

#if defined(SIMD_SSE)
		tail[0] = _mm_cvtsi128_si32(last);
		if ((vertex_count & 3) > 1)
			tail[1] = _mm_extract_epi32(last, 1);
		if ((vertex_count & 3) > 2)
			tail[2] = _mm_extract_epi32(last, 2);
#elif defined(SIMD_NEON)
		vst1q_lane_u32(&tail[0], last, 0);
		if ((vertex_count & 3) > 1)
			vst1q_lane_u32(&tail[1], last, 1);
		if ((vertex_count & 3) > 2)
			vst1q_lane_u32(&tail[2], last, 2);
#endif
	}

	return data;
}

SIMD_TARGET
static const unsigned char* decodeVerticesSimd(unsigned short* vertices, const unsigned char* ctrl, const unsigned char* data, const unsigned char* bound, size_t vertex_count)
{
#if defined(SIMD_SSE)
	__m128i repack = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0);
	__m128i last = _mm_set1_epi32(-1);
#elif defined(SIMD_NEON)
	uint32x4_t last = vdupq_n_u32(~0u);
#endif

	// because the output buffer is guaranteed to have 32-bit aligned size available, we can simplify tail processing
	// if the number of vertices mod 4 is 3, we'd normally need to write 8+6 bytes, but we can instead overwrite up to 2 bytes in the main loop
	size_t groups = (vertex_count + 1) / 4;

	// process all complete groups
	for (size_t i = 0; i < groups; ++i)
	{
		unsigned char code = *ctrl++;

		if (data > bound)
			return NULL;

		last = decodeVertexGroup(last, code, data);

#if defined(SIMD_SSE)
		__m128i r = _mm_shuffle_epi8(last, repack);
		_mm_storel_epi64(reinterpret_cast<__m128i*>(&vertices[i * 4]), r);
#elif defined(SIMD_NEON)
		uint16x4_t r = vmovn_u32(last);
		vst1_u16(&vertices[i * 4], r);
#endif
	}

	// process a 1-2 vertex tail; to maintain the memory safety guarantee we have to write a 32-bit element
	if (groups * 4 < vertex_count)
	{
		unsigned char code = *ctrl++;

		if (data > bound)
			return NULL;

		last = decodeVertexGroup(last, code, data);

		unsigned short* tail = &vertices[vertex_count & ~3u];

#if defined(SIMD_SSE)
		__m128i r = _mm_shufflelo_epi16(last, 8);
		*reinterpret_cast<unaligned_int*>(tail) = _mm_cvtsi128_si32(r);
#elif defined(SIMD_NEON)
		uint16x4_t r = vmovn_u32(last);
		vst1_lane_u32(reinterpret_cast<unsigned int*>(tail), vreinterpret_u32_u16(r), 0);
#endif
	}

	return data;
}

template <int Raw>
SIMD_TARGET SIMD_FLATTEN static int
decodeMeshletSimd(void* vertices, void* triangles, const unsigned char* codes, const unsigned char* ctrl, const unsigned char* data, const unsigned char* bound, size_t vertex_count, size_t triangle_count, size_t vertex_size, size_t triangle_size)
{
	assert(gDecodeTablesInitialized);
	(void)gDecodeTablesInitialized;

#ifdef __clang__
	// data is guaranteed to be non-null initially; if decode loops never hit bounds errors, it remains non-null
	__builtin_assume(data);
#endif

	// decodes 4 vertices at a time with tail processing; writes up to align(vertex_size * vertex_count, 4)
	// raw decoding skips tail processing by rounding up vertex count; it's safe because output buffer is guaranteed to have extra space, and tail control data is 0
	if (vertex_size == 4 || Raw)
		data = decodeVerticesSimd(static_cast<unsigned int*>(vertices), ctrl, data, bound, Raw ? (vertex_count + 3) & ~3 : vertex_count);
	else
		data = decodeVerticesSimd(static_cast<unsigned short*>(vertices), ctrl, data, bound, vertex_count);
	if (!data)
		return -2;

	// decodes 2/4 triangles at a time with tail processing; writes up to align(triangle_size * triangle_count, 4)
	// raw decoding skips tail processing by rounding up triangle count; it's safe because output buffer is guaranteed to have extra space, and tail code data is 0
	if (triangle_size == 4 || Raw)
		data = decodeTrianglesSimd(static_cast<unsigned int*>(triangles), codes, data, bound, Raw ? (triangle_count + 1) & ~1 : triangle_count);
	else
		data = decodeTrianglesSimd(static_cast<unsigned char*>(triangles), codes, data, bound, triangle_count);
	if (!data)
		return -2;

	return (data == bound) ? 0 : -3;
}
#endif

} // namespace meshopt

size_t meshopt_encodeMeshletBound(size_t max_vertices, size_t max_triangles)
{
	size_t codes_size = (max_triangles + 1) / 2;
	size_t extra_size = max_triangles * 3;

	size_t ctrl_size = (max_vertices + 3) / 4;
	size_t data_size = (max_vertices + 3) / 4 * 16; // worst case: 16 bytes per vertex group

	size_t gap_size = (codes_size + ctrl_size < 16) ? 16 - (codes_size + ctrl_size) : 0;

	return codes_size + extra_size + ctrl_size + data_size + gap_size;
}

size_t meshopt_encodeMeshlet(unsigned char* buffer, size_t buffer_size, const unsigned int* vertices, size_t vertex_count, const unsigned char* triangles, size_t triangle_count)
{
	using namespace meshopt;

	assert(triangle_count <= 256 && vertex_count <= 256);

	// 4 bits per triangle + up to three bytes of extra data
	unsigned char codes[256 / 2];
	unsigned char extra[256 * 3];
	size_t codes_size = (triangle_count + 1) / 2;
	size_t extra_size = encodeTriangles(codes, extra, triangles, triangle_count);
	assert(extra_size <= sizeof(extra));

	// 2 bits per vertex + up to 4 bytes of actual data
	unsigned char ctrl[256 / 4];
	unsigned char data[256 * 4];
	size_t ctrl_size = (vertex_count + 3) / 4;
	size_t data_size = encodeVertices(ctrl, data, vertices, vertex_count);
	assert(data_size <= sizeof(data));

	// we need to ensure that up to 16 bytes after extra+data are available for SIMD decoding
	// to minimize overhead, we place fixed-size codes+control at the end of the buffer
	size_t gap_size = (codes_size + ctrl_size < 16) ? 16 - (codes_size + ctrl_size) : 0;

	size_t result = codes_size + extra_size + ctrl_size + data_size + gap_size;

	if (result > buffer_size)
		return 0;

	// variable-size data first
	memcpy(buffer, data, data_size);
	buffer += data_size;
	memcpy(buffer, extra, extra_size);
	buffer += extra_size;

	// gap (for accelerated decoding) separates variable-size and fixed-size data
	memset(buffer, 0, gap_size);
	buffer += gap_size;

	// fixed-size data last; it can be located from buffer end during decoding
	memcpy(buffer, ctrl, ctrl_size);
	buffer += ctrl_size;
	memcpy(buffer, codes, codes_size);
	buffer += codes_size;

#if TRACE > 1
	printf("extra:");
	for (size_t i = 0; i < extra_size; ++i)
		printf(" %d", extra[i]);
	printf("\n");

	unsigned int minv = ~0u;
	for (size_t i = 0; i < vertex_count; ++i)
		minv = minv < vertices[i] ? minv : vertices[i];

	printf("vertices: [%d+]", minv);
	for (size_t i = 0; i < vertex_count; ++i)
		printf(" %d", vertices[i] - minv);
	printf("\n");
#endif

#if TRACE
	printf("stats: %d vertices, %d triangles => %d bytes (triangles: %d codes, %d extra; vertices: %d control, %d data; %d gap)\n",
	    int(vertex_count), int(triangle_count), int(result),
	    int(codes_size), int(extra_size), int(ctrl_size), int(data_size), int(gap_size));
#endif

	return result;
}

int meshopt_decodeMeshlet(void* vertices, size_t vertex_count, size_t vertex_size, void* triangles, size_t triangle_count, size_t triangle_size, const unsigned char* buffer, size_t buffer_size)
{
	using namespace meshopt;

	assert(triangle_count <= 256 && vertex_count <= 256);
	assert(vertex_size == 4 || vertex_size == 2);
	assert(triangle_size == 4 || triangle_size == 3);

	// layout must match encoding
	size_t codes_size = (triangle_count + 1) / 2;
	size_t ctrl_size = (vertex_count + 3) / 4;
	size_t gap_size = (codes_size + ctrl_size < 16) ? 16 - (codes_size + ctrl_size) : 0;

	if (buffer_size < codes_size + ctrl_size + gap_size)
		return -2;

	const unsigned char* end = buffer + buffer_size;
	const unsigned char* codes = end - codes_size;
	const unsigned char* ctrl = codes - ctrl_size;
	const unsigned char* data = buffer;

	// gap ensures we have at least 16 bytes available after bound; this allows SIMD decoders to over-read safely
	const unsigned char* bound = ctrl - gap_size;
	assert(bound >= buffer && bound + 16 <= buffer + buffer_size);

#if defined(SIMD_FALLBACK)
	return (gDecodeTablesInitialized ? decodeMeshletSimd<0> : decodeMeshlet)(vertices, triangles, codes, ctrl, data, bound, vertex_count, triangle_count, vertex_size, triangle_size);
#elif defined(SIMD_SSE) || defined(SIMD_NEON)
	return decodeMeshletSimd<0>(vertices, triangles, codes, ctrl, data, bound, vertex_count, triangle_count, vertex_size, triangle_size);
#else
	return decodeMeshlet(vertices, triangles, codes, ctrl, data, bound, vertex_count, triangle_count, vertex_size, triangle_size);
#endif
}

int meshopt_decodeMeshletRaw(unsigned int* vertices, size_t vertex_count, unsigned int* triangles, size_t triangle_count, const unsigned char* buffer, size_t buffer_size)
{
	using namespace meshopt;

	assert(triangle_count <= 256 && vertex_count <= 256);

	// layout must match encoding
	size_t codes_size = (triangle_count + 1) / 2;
	size_t ctrl_size = (vertex_count + 3) / 4;
	size_t gap_size = (codes_size + ctrl_size < 16) ? 16 - (codes_size + ctrl_size) : 0;

	if (buffer_size < codes_size + ctrl_size + gap_size)
		return -2;

	const unsigned char* end = buffer + buffer_size;
	const unsigned char* codes = end - codes_size;
	const unsigned char* ctrl = codes - ctrl_size;
	const unsigned char* data = buffer;

	// gap ensures we have at least 16 bytes available after bound; this allows SIMD decoders to over-read safely
	const unsigned char* bound = ctrl - gap_size;
	assert(bound >= buffer && bound + 16 <= buffer + buffer_size);

#if defined(SIMD_FALLBACK)
	return (gDecodeTablesInitialized ? decodeMeshletSimd<1> : decodeMeshlet)(vertices, triangles, codes, ctrl, data, bound, vertex_count, triangle_count, 4, 4);
#elif defined(SIMD_SSE) || defined(SIMD_NEON)
	return decodeMeshletSimd<1>(vertices, triangles, codes, ctrl, data, bound, vertex_count, triangle_count, 4, 4);
#else
	return decodeMeshlet(vertices, triangles, codes, ctrl, data, bound, vertex_count, triangle_count, 4, 4);
#endif
}

#undef SIMD_SSE
#undef SIMD_NEON
#undef SIMD_FALLBACK
#undef SIMD_FLATTEN
#undef SIMD_TARGET
