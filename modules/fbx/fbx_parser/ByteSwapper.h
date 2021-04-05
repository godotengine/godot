/*************************************************************************/
/*  ByteSwapper.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

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

----------------------------------------------------------------------
*/

/** @file Helper class tp perform various byte oder swappings
   (e.g. little to big endian) */
#ifndef BYTE_SWAPPER_H
#define BYTE_SWAPPER_H

#include <stdint.h>
#include <algorithm>
#include <locale>

namespace FBXDocParser {
// --------------------------------------------------------------------------------------
/** Defines some useful byte order swap routines.
 *
 * This is required to read big-endian model formats on little-endian machines,
 * and vice versa. Direct use of this class is DEPRECATED. Use #StreamReader instead. */
// --------------------------------------------------------------------------------------
class ByteSwap {
	ByteSwap() {}

public:
	// ----------------------------------------------------------------------
	/** Swap two bytes of data
	 *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
	static inline void Swap2(void *_szOut) {
		uint8_t *const szOut = reinterpret_cast<uint8_t *>(_szOut);
		std::swap(szOut[0], szOut[1]);
	}

	// ----------------------------------------------------------------------
	/** Swap four bytes of data
	 *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
	static inline void Swap4(void *_szOut) {
		uint8_t *const szOut = reinterpret_cast<uint8_t *>(_szOut);
		std::swap(szOut[0], szOut[3]);
		std::swap(szOut[1], szOut[2]);
	}

	// ----------------------------------------------------------------------
	/** Swap eight bytes of data
	 *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
	static inline void Swap8(void *_szOut) {
		uint8_t *const szOut = reinterpret_cast<uint8_t *>(_szOut);
		std::swap(szOut[0], szOut[7]);
		std::swap(szOut[1], szOut[6]);
		std::swap(szOut[2], szOut[5]);
		std::swap(szOut[3], szOut[4]);
	}

	// ----------------------------------------------------------------------
	/** ByteSwap a float. Not a joke.
	 *  @param[inout] fOut ehm. .. */
	static inline void Swap(float *fOut) {
		Swap4(fOut);
	}

	// ----------------------------------------------------------------------
	/** ByteSwap a double. Not a joke.
	 *  @param[inout] fOut ehm. .. */
	static inline void Swap(double *fOut) {
		Swap8(fOut);
	}

	// ----------------------------------------------------------------------
	/** ByteSwap an int16t. Not a joke.
	 *  @param[inout] fOut ehm. .. */
	static inline void Swap(int16_t *fOut) {
		Swap2(fOut);
	}

	static inline void Swap(uint16_t *fOut) {
		Swap2(fOut);
	}

	// ----------------------------------------------------------------------
	/** ByteSwap an int32t. Not a joke.
	 *  @param[inout] fOut ehm. .. */
	static inline void Swap(int32_t *fOut) {
		Swap4(fOut);
	}

	static inline void Swap(uint32_t *fOut) {
		Swap4(fOut);
	}

	// ----------------------------------------------------------------------
	/** ByteSwap an int64t. Not a joke.
	 *  @param[inout] fOut ehm. .. */
	static inline void Swap(int64_t *fOut) {
		Swap8(fOut);
	}

	static inline void Swap(uint64_t *fOut) {
		Swap8(fOut);
	}

	// ----------------------------------------------------------------------
	//! Templatized ByteSwap
	//! \returns param tOut as swapped
	template <typename Type>
	static inline Type Swapped(Type tOut) {
		return _swapper<Type, sizeof(Type)>()(tOut);
	}

private:
	template <typename T, size_t size>
	struct _swapper;
};

template <typename T>
struct ByteSwap::_swapper<T, 2> {
	T operator()(T tOut) {
		Swap2(&tOut);
		return tOut;
	}
};

template <typename T>
struct ByteSwap::_swapper<T, 4> {
	T operator()(T tOut) {
		Swap4(&tOut);
		return tOut;
	}
};

template <typename T>
struct ByteSwap::_swapper<T, 8> {
	T operator()(T tOut) {
		Swap8(&tOut);
		return tOut;
	}
};

// --------------------------------------------------------------------------------------
// ByteSwap macros for BigEndian/LittleEndian support
// --------------------------------------------------------------------------------------
#if (defined AI_BUILD_BIG_ENDIAN)
#define AI_LE(t) (t)
#define AI_BE(t) ByteSwap::Swapped(t)
#define AI_LSWAP2(p)
#define AI_LSWAP4(p)
#define AI_LSWAP8(p)
#define AI_LSWAP2P(p)
#define AI_LSWAP4P(p)
#define AI_LSWAP8P(p)
#define LE_NCONST const
#define AI_SWAP2(p) ByteSwap::Swap2(&(p))
#define AI_SWAP4(p) ByteSwap::Swap4(&(p))
#define AI_SWAP8(p) ByteSwap::Swap8(&(p))
#define AI_SWAP2P(p) ByteSwap::Swap2((p))
#define AI_SWAP4P(p) ByteSwap::Swap4((p))
#define AI_SWAP8P(p) ByteSwap::Swap8((p))
#define BE_NCONST
#else
#define AI_BE(t) (t)
#define AI_LE(t) ByteSwap::Swapped(t)
#define AI_SWAP2(p)
#define AI_SWAP4(p)
#define AI_SWAP8(p)
#define AI_SWAP2P(p)
#define AI_SWAP4P(p)
#define AI_SWAP8P(p)
#define BE_NCONST const
#define AI_LSWAP2(p) ByteSwap::Swap2(&(p))
#define AI_LSWAP4(p) ByteSwap::Swap4(&(p))
#define AI_LSWAP8(p) ByteSwap::Swap8(&(p))
#define AI_LSWAP2P(p) ByteSwap::Swap2((p))
#define AI_LSWAP4P(p) ByteSwap::Swap4((p))
#define AI_LSWAP8P(p) ByteSwap::Swap8((p))
#define LE_NCONST
#endif

namespace Intern {

// --------------------------------------------------------------------------------------------
template <typename T, bool doit>
struct ByteSwapper {
	void operator()(T *inout) {
		ByteSwap::Swap(inout);
	}
};

template <typename T>
struct ByteSwapper<T, false> {
	void operator()(T *) {
	}
};

// --------------------------------------------------------------------------------------------
template <bool SwapEndianess, typename T, bool RuntimeSwitch>
struct Getter {
	void operator()(T *inout, bool le) {
		le = !le;
		if (le) {
			ByteSwapper<T, (sizeof(T) > 1 ? true : false)>()(inout);
		} else {
			ByteSwapper<T, false>()(inout);
		}
	}
};

template <bool SwapEndianess, typename T>
struct Getter<SwapEndianess, T, false> {
	void operator()(T *inout, bool /*le*/) {
		// static branch
		ByteSwapper<T, (SwapEndianess && sizeof(T) > 1)>()(inout);
	}
};
} // namespace Intern
} // namespace FBXDocParser

#endif // BYTE_SWAPPER_H
