// basisu.h
// Copyright (C) 2019 Binomial LLC. All Rights Reserved.
// Important: If compiling with gcc, be sure strict aliasing is disabled: -fno-strict-aliasing
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#ifdef _MSC_VER

	#pragma warning (disable : 4201)
	#pragma warning (disable : 4127) // warning C4127: conditional expression is constant
	#pragma warning (disable : 4530) // C++ exception handler used, but unwind semantics are not enabled.

	#ifndef BASISU_NO_ITERATOR_DEBUG_LEVEL
		//#define _HAS_ITERATOR_DEBUGGING 0

		#if defined(_DEBUG) || defined(DEBUG)
			// This is madness, but we need to disable iterator debugging in debug builds or the encoder is unsable because MSVC's iterator debugging implementation is totally broken.
			#ifndef _ITERATOR_DEBUG_LEVEL
			#define _ITERATOR_DEBUG_LEVEL 1
			#endif
			#ifndef _SECURE_SCL
			#define _SECURE_SCL 1
			#endif
		#else // defined(_DEBUG) || defined(DEBUG)
			#ifndef _SECURE_SCL
			#define _SECURE_SCL 0
			#endif
			#ifndef _ITERATOR_DEBUG_LEVEL
			#define _ITERATOR_DEBUG_LEVEL 0
			#endif
		#endif // defined(_DEBUG) || defined(DEBUG)

		#ifndef NOMINMAX
			#define NOMINMAX
		#endif

	#endif // BASISU_NO_ITERATOR_DEBUG_LEVEL

#endif // _MSC_VER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <memory.h>
#include <limits.h>
#include <stdint.h>

#include <algorithm>
#include <limits>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>
#include <assert.h>
#include <random>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

// Set to one to enable debug printf()'s when any errors occur, for development/debugging.
#ifndef BASISU_DEVEL_MESSAGES
#define BASISU_DEVEL_MESSAGES 0
#endif

#define BASISU_NOTE_UNUSED(x) (void)(x)
#define BASISU_ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))
#define BASISU_NO_EQUALS_OR_COPY_CONSTRUCT(x) x(const x &) = delete; x& operator= (const x &) = delete;
#define BASISU_ASSUME(x) static_assert(x, #x);
#define BASISU_OFFSETOF(s, m) (uint32_t)(intptr_t)(&((s *)(0))->m)
#define BASISU_STRINGIZE(x) #x
#define BASISU_STRINGIZE2(x) BASISU_STRINGIZE(x)

#if BASISU_DEVEL_MESSAGES
	#define BASISU_DEVEL_ERROR(...) do { basisu::debug_printf(__VA_ARGS__); } while(0)
#else
	#define BASISU_DEVEL_ERROR(...)
#endif

namespace basisu
{
	// Types/utilities

#ifdef _WIN32
	const char BASISU_PATH_SEPERATOR_CHAR = '\\';
#else
	const char BASISU_PATH_SEPERATOR_CHAR = '/';
#endif

	typedef std::vector<uint8_t> uint8_vec;
	typedef std::vector<int16_t> int16_vec;
	typedef std::vector<uint16_t> uint16_vec;
	typedef std::vector<uint32_t> uint_vec;
	typedef std::vector<uint64_t> uint64_vec;
	typedef std::vector<int> int_vec;
	typedef std::vector<bool> bool_vec;

	void enable_debug_printf(bool enabled);
	void debug_printf(const char *pFmt, ...);
		
	template <typename T> inline void clear_obj(T& obj) { memset(&obj, 0, sizeof(obj)); }

	template <typename T0, typename T1> inline T0 lerp(T0 a, T0 b, T1 c) { return a + (b - a) * c; }

	template <typename S> inline S maximum(S a, S b) { return (a > b) ? a : b; }
	template <typename S> inline S maximum(S a, S b, S c) { return maximum(maximum(a, b), c); }
	
	template <typename S> inline S minimum(S a, S b) {	return (a < b) ? a : b; }
	template <typename S> inline S minimum(S a, S b, S c) {	return minimum(minimum(a, b), c); }

	template <typename S> inline S clamp(S value, S low, S high) { return (value < low) ? low : ((value > high) ? high : value); }

	inline uint32_t iabs(int32_t i) { return (i < 0) ? static_cast<uint32_t>(-i) : static_cast<uint32_t>(i);	}
	inline uint64_t iabs64(int64_t i) {	return (i < 0) ? static_cast<uint64_t>(-i) : static_cast<uint64_t>(i); }

	template<typename T> inline void clear_vector(T &vec) { vec.erase(vec.begin(), vec.end()); }		
	template<typename T> inline typename T::value_type *enlarge_vector(T &vec, size_t n) { size_t cs = vec.size(); vec.resize(cs + n); return &vec[cs]; }

	template<typename S> inline S square(S val) { return val * val; }

	inline bool is_pow2(uint32_t x) { return x && ((x & (x - 1U)) == 0U); }
	inline bool is_pow2(uint64_t x) { return x && ((x & (x - 1U)) == 0U); }

	template<typename T> inline T open_range_check(T v, T minv, T maxv) { assert(v >= minv && v < maxv); return v; }
	template<typename T> inline T open_range_check(T v, T maxv) { assert(v < maxv); BASISU_NOTE_UNUSED(maxv); return v; }

	inline uint32_t total_bits(uint32_t v) { uint32_t l = 0; for ( ; v > 0U; ++l) v >>= 1; return l; }

	template<typename T> inline T saturate(T val) { return clamp(val, 0.0f, 1.0f); }

	template<typename T, typename R> inline void append_vector(T &vec, const R *pObjs, size_t n) 
	{ 
		if (n)
		{
			const size_t cur_s = vec.size();
			vec.resize(cur_s + n);
			memcpy(&vec[cur_s], pObjs, sizeof(R) * n);
		}
	}

	template<typename T> inline void append_vector(T &vec, const T &other_vec)
	{
		if (other_vec.size())
			append_vector(vec, &other_vec[0], other_vec.size());
	}

	template<typename T> inline void vector_ensure_element_is_valid(T &vec, size_t idx)
	{
		if (idx >= vec.size())
			vec.resize(idx + 1);
	}

	template<typename T> inline void vector_sort(T &vec)
	{
		if (vec.size())
			std::sort(vec.begin(), vec.end());
	}

	template<typename T, typename U> inline bool unordered_set_contains(T& set, const U&obj)
	{
		return set.find(obj) != set.end();
	}

	template<typename T> int vector_find(const T &vec, const typename T::value_type &obj)
	{
		assert(vec.size() <= INT_MAX);
		for (size_t i = 0; i < vec.size(); i++)
			if (vec[i] == obj)
				return static_cast<int>(i);
		return -1;
	}

	template<typename T> void vector_set_all(T &vec, const typename T::value_type &obj)
	{
		for (size_t i = 0; i < vec.size(); i++)
			vec[i] = obj;
	}
		
	inline uint64_t read_be64(const void *p)
	{
		uint64_t val = 0;
		for (uint32_t i = 0; i < 8; i++)
			val |= (static_cast<uint64_t>(static_cast<const uint8_t *>(p)[7 - i]) << (i * 8));
		return val;
	}

	inline void write_be64(void *p, uint64_t x)
	{
		for (uint32_t i = 0; i < 8; i++)
			static_cast<uint8_t *>(p)[7 - i] = static_cast<uint8_t>(x >> (i * 8));
	}

	static inline uint16_t byteswap16(uint16_t x) { return static_cast<uint16_t>((x << 8) | (x >> 8)); }
	static inline uint32_t byteswap32(uint32_t x) { return ((x << 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x >> 24)); }

	inline uint32_t floor_log2i(uint32_t v)
	{
		uint32_t b = 0;
		for (; v > 1U; ++b)
			v >>= 1;
		return b;
	}

	inline uint32_t ceil_log2i(uint32_t v)
	{
		uint32_t b = floor_log2i(v);
		if ((b != 32) && (v > (1U << b)))
			++b;
		return b;
	}

	inline int posmod(int x, int y)
	{
		if (x >= 0)
			return (x < y) ? x : (x % y);
		int m = (-x) % y;
		return (m != 0) ? (y - m) : m;
	}

	inline bool do_excl_ranges_overlap(int la, int ha, int lb, int hb)
	{
		assert(la < ha && lb < hb);
		if ((ha <= lb) || (la >= hb)) return false;
		return true;
	}
		
	// Always little endian 2-4 byte unsigned int
	template<uint32_t NumBytes>
	struct packed_uint
	{
		uint8_t m_bytes[NumBytes];

		inline packed_uint() { static_assert(NumBytes <= 4, "NumBytes <= 4"); }
		inline packed_uint(uint32_t v) { *this = v; }
		inline packed_uint(const packed_uint& other) { *this = other; }

		inline packed_uint& operator= (uint32_t v) { for (uint32_t i = 0; i < NumBytes; i++) m_bytes[i] = static_cast<uint8_t>(v >> (i * 8)); return *this; }

		inline operator uint32_t() const
		{
			switch (NumBytes)
			{
				case 1:  return  m_bytes[0];
				case 2:  return (m_bytes[1] << 8U) | m_bytes[0];
				case 3:  return (m_bytes[2] << 16U) | (m_bytes[1] << 8U) | (m_bytes[0]);
				default: return (m_bytes[3] << 24U) | (m_bytes[2] << 16U) | (m_bytes[1] << 8U) | (m_bytes[0]);
			}
		}
	};

	enum eZero { cZero };
	enum eNoClamp { cNoClamp };
	
	// Rice/Huffman entropy coding
		
	// This is basically Deflate-style canonical Huffman, except we allow for a lot more symbols.
	enum
	{
		cHuffmanMaxSupportedCodeSize = 16, cHuffmanMaxSupportedInternalCodeSize = 31, 
		cHuffmanFastLookupBits = 10, cHuffmanFastLookupSize = 1 << cHuffmanFastLookupBits,
		cHuffmanMaxSymsLog2 = 14, cHuffmanMaxSyms = 1 << cHuffmanMaxSymsLog2,

		// Small zero runs
		cHuffmanSmallZeroRunSizeMin = 3, cHuffmanSmallZeroRunSizeMax = 10, cHuffmanSmallZeroRunExtraBits = 3,

		// Big zero run
		cHuffmanBigZeroRunSizeMin = 11, cHuffmanBigZeroRunSizeMax = 138, cHuffmanBigZeroRunExtraBits = 7,

		// Small non-zero run
		cHuffmanSmallRepeatSizeMin = 3, cHuffmanSmallRepeatSizeMax = 6, cHuffmanSmallRepeatExtraBits = 2,

		// Big non-zero run
		cHuffmanBigRepeatSizeMin = 7, cHuffmanBigRepeatSizeMax = 134, cHuffmanBigRepeatExtraBits = 7,

		cHuffmanTotalCodelengthCodes = 21, cHuffmanSmallZeroRunCode = 17, cHuffmanBigZeroRunCode = 18, cHuffmanSmallRepeatCode = 19, cHuffmanBigRepeatCode = 20
	};

	static const uint8_t g_huffman_sorted_codelength_codes[] = { cHuffmanSmallZeroRunCode, cHuffmanBigZeroRunCode,	cHuffmanSmallRepeatCode, cHuffmanBigRepeatCode, 0, 8, 7, 9, 6, 0xA, 5, 0xB, 4, 0xC, 3, 0xD, 2, 0xE, 1, 0xF, 0x10 };
	const uint32_t cHuffmanTotalSortedCodelengthCodes = sizeof(g_huffman_sorted_codelength_codes) / sizeof(g_huffman_sorted_codelength_codes[0]);

	// GPU texture formats

	enum class texture_format
	{
		cInvalidTextureFormat = -1,
		
		// Block-based formats
		cETC1,			// ETC1
		cETC1S,			// ETC1 (subset: diff colors only, no subblocks)
		cETC2_RGB,		// ETC2 color block
		cETC2_RGBA,		// ETC2 alpha block followed by ETC2 color block
		cETC2_ALPHA,	// ETC2 EAC alpha block 
		cBC1,				// DXT1
		cBC3,				// DXT5 (DXT5A block followed by a DXT1 block)
		cBC4,				// DXT5A
		cBC5,				// 3DC/DXN (two DXT5A blocks)
		cBC7,
		cASTC4x4,		
		cPVRTC1_4_RGB,
		cPVRTC1_4_RGBA,
		cATC_RGB,
		cATC_RGBA_INTERPOLATED_ALPHA,
		cFXT1_RGB,
		cPVRTC2_4_RGBA,
		cETC2_R11_EAC,
		cETC2_RG11_EAC,
		
		// Uncompressed/raw pixels
		cRGBA32,
		cRGB565,
		cBGR565,
		cRGBA4444,
		cABGR4444
	};

	inline uint32_t get_bytes_per_block(texture_format fmt)
	{
		switch (fmt)
		{
		case texture_format::cETC1:
		case texture_format::cETC1S:
		case texture_format::cETC2_RGB:
		case texture_format::cETC2_ALPHA:
		case texture_format::cBC1:
		case texture_format::cBC4:
		case texture_format::cPVRTC1_4_RGB:
		case texture_format::cPVRTC1_4_RGBA:
		case texture_format::cATC_RGB:
		case texture_format::cPVRTC2_4_RGBA:
		case texture_format::cETC2_R11_EAC:
			return 8;
		case texture_format::cRGBA32:
			return sizeof(uint32_t) * 16;
		default:
			break;
		}
		return 16;
	}

	inline uint32_t get_qwords_per_block(texture_format fmt)
	{
		return get_bytes_per_block(fmt) >> 3;
	}

	inline uint32_t get_block_width(texture_format fmt)
	{
		BASISU_NOTE_UNUSED(fmt);
		switch (fmt)
		{
		case texture_format::cFXT1_RGB:
			return 8;
		default:
			break;
		}
		return 4;
	}

	inline uint32_t get_block_height(texture_format fmt)
	{
		BASISU_NOTE_UNUSED(fmt);
		return 4;
	}
							
} // namespace basisu

