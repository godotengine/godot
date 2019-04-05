/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


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
#ifndef AI_BYTESWAPPER_H_INC
#define AI_BYTESWAPPER_H_INC

#include <assimp/ai_assert.h>
#include <assimp/types.h>
#include <stdint.h>

#if _MSC_VER >= 1400
#include <stdlib.h>
#endif

namespace Assimp    {
// --------------------------------------------------------------------------------------
/** Defines some useful byte order swap routines.
 *
 * This is required to read big-endian model formats on little-endian machines,
 * and vice versa. Direct use of this class is DEPRECATED. Use #StreamReader instead. */
// --------------------------------------------------------------------------------------
class ByteSwap {
    ByteSwap() AI_NO_EXCEPT {}

public:

    // ----------------------------------------------------------------------
    /** Swap two bytes of data
     *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
    static inline void Swap2(void* _szOut)
    {
        ai_assert(_szOut);

#if _MSC_VER >= 1400
        uint16_t* const szOut = reinterpret_cast<uint16_t*>(_szOut);
        *szOut = _byteswap_ushort(*szOut);
#else
        uint8_t* const szOut = reinterpret_cast<uint8_t*>(_szOut);
        std::swap(szOut[0],szOut[1]);
#endif
    }

    // ----------------------------------------------------------------------
    /** Swap four bytes of data
     *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
    static inline void Swap4(void* _szOut)
    {
        ai_assert(_szOut);

#if _MSC_VER >= 1400
        uint32_t* const szOut = reinterpret_cast<uint32_t*>(_szOut);
        *szOut = _byteswap_ulong(*szOut);
#else
        uint8_t* const szOut = reinterpret_cast<uint8_t*>(_szOut);
        std::swap(szOut[0],szOut[3]);
        std::swap(szOut[1],szOut[2]);
#endif
    }

    // ----------------------------------------------------------------------
    /** Swap eight bytes of data
     *  @param[inout] _szOut A void* to save the reintcasts for the caller. */
    static inline void Swap8(void* _szOut)
    {
    ai_assert(_szOut);

#if _MSC_VER >= 1400
        uint64_t* const szOut = reinterpret_cast<uint64_t*>(_szOut);
        *szOut = _byteswap_uint64(*szOut);
#else
        uint8_t* const szOut = reinterpret_cast<uint8_t*>(_szOut);
        std::swap(szOut[0],szOut[7]);
        std::swap(szOut[1],szOut[6]);
        std::swap(szOut[2],szOut[5]);
        std::swap(szOut[3],szOut[4]);
#endif
    }

    // ----------------------------------------------------------------------
    /** ByteSwap a float. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(float* fOut) {
        Swap4(fOut);
    }

    // ----------------------------------------------------------------------
    /** ByteSwap a double. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(double* fOut) {
        Swap8(fOut);
    }


    // ----------------------------------------------------------------------
    /** ByteSwap an int16t. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(int16_t* fOut) {
        Swap2(fOut);
    }

    static inline void Swap(uint16_t* fOut) {
        Swap2(fOut);
    }

    // ----------------------------------------------------------------------
    /** ByteSwap an int32t. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(int32_t* fOut){
        Swap4(fOut);
    }

    static inline void Swap(uint32_t* fOut){
        Swap4(fOut);
    }

    // ----------------------------------------------------------------------
    /** ByteSwap an int64t. Not a joke.
     *  @param[inout] fOut ehm. .. */
    static inline void Swap(int64_t* fOut) {
        Swap8(fOut);
    }

    static inline void Swap(uint64_t* fOut) {
        Swap8(fOut);
    }

    // ----------------------------------------------------------------------
    //! Templatized ByteSwap
    //! \returns param tOut as swapped
    template<typename Type>
    static inline Type Swapped(Type tOut)
    {
        return _swapper<Type,sizeof(Type)>()(tOut);
    }

private:

    template <typename T, size_t size> struct _swapper;
};

template <typename T> struct ByteSwap::_swapper<T,2> {
    T operator() (T tOut) {
        Swap2(&tOut);
        return tOut;
    }
};

template <typename T> struct ByteSwap::_swapper<T,4> {
    T operator() (T tOut) {
        Swap4(&tOut);
        return tOut;
    }
};

template <typename T> struct ByteSwap::_swapper<T,8> {
    T operator() (T tOut) {
        Swap8(&tOut);
        return tOut;
    }
};


// --------------------------------------------------------------------------------------
// ByteSwap macros for BigEndian/LittleEndian support
// --------------------------------------------------------------------------------------
#if (defined AI_BUILD_BIG_ENDIAN)
#   define AI_LE(t) (t)
#   define AI_BE(t) ByteSwap::Swapped(t)
#   define AI_LSWAP2(p)
#   define AI_LSWAP4(p)
#   define AI_LSWAP8(p)
#   define AI_LSWAP2P(p)
#   define AI_LSWAP4P(p)
#   define AI_LSWAP8P(p)
#   define LE_NCONST const
#   define AI_SWAP2(p) ByteSwap::Swap2(&(p))
#   define AI_SWAP4(p) ByteSwap::Swap4(&(p))
#   define AI_SWAP8(p) ByteSwap::Swap8(&(p))
#   define AI_SWAP2P(p) ByteSwap::Swap2((p))
#   define AI_SWAP4P(p) ByteSwap::Swap4((p))
#   define AI_SWAP8P(p) ByteSwap::Swap8((p))
#   define BE_NCONST
#else
#   define AI_BE(t) (t)
#   define AI_LE(t) ByteSwap::Swapped(t)
#   define AI_SWAP2(p)
#   define AI_SWAP4(p)
#   define AI_SWAP8(p)
#   define AI_SWAP2P(p)
#   define AI_SWAP4P(p)
#   define AI_SWAP8P(p)
#   define BE_NCONST const
#   define AI_LSWAP2(p)     ByteSwap::Swap2(&(p))
#   define AI_LSWAP4(p)     ByteSwap::Swap4(&(p))
#   define AI_LSWAP8(p)     ByteSwap::Swap8(&(p))
#   define AI_LSWAP2P(p)    ByteSwap::Swap2((p))
#   define AI_LSWAP4P(p)    ByteSwap::Swap4((p))
#   define AI_LSWAP8P(p)    ByteSwap::Swap8((p))
#   define LE_NCONST
#endif


namespace Intern {

// --------------------------------------------------------------------------------------------
template <typename T, bool doit>
struct ByteSwapper  {
    void operator() (T* inout) {
        ByteSwap::Swap(inout);
    }
};

template <typename T>
struct ByteSwapper<T,false> {
    void operator() (T*) {
    }
};

// --------------------------------------------------------------------------------------------
template <bool SwapEndianess, typename T, bool RuntimeSwitch>
struct Getter {
    void operator() (T* inout, bool le) {
#ifdef AI_BUILD_BIG_ENDIAN
        le =  le;
#else
        le =  !le;
#endif
        if (le) {
            ByteSwapper<T,(sizeof(T)>1?true:false)> () (inout);
        }
        else ByteSwapper<T,false> () (inout);
    }
};

template <bool SwapEndianess, typename T>
struct Getter<SwapEndianess,T,false> {

    void operator() (T* inout, bool /*le*/) {
        // static branch
        ByteSwapper<T,(SwapEndianess && sizeof(T)>1)> () (inout);
    }
};
} // end Intern
} // end Assimp

#endif //!! AI_BYTESWAPPER_H_INC
