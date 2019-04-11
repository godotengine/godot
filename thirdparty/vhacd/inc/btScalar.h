/*
Copyright (c) 2003-2009 Erwin Coumans  http://bullet.googlecode.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_SCALAR_H
#define BT_SCALAR_H

#ifdef BT_MANAGED_CODE
//Aligned data types not supported in managed code
#pragma unmanaged
#endif

#include <float.h>
#include <math.h>
#include <stdlib.h> //size_t for MSVC 6.0
#include <stdint.h>

/* SVN $Revision$ on $Date$ from http://bullet.googlecode.com*/
#define BT_BULLET_VERSION 279

inline int32_t btGetVersion()
{
    return BT_BULLET_VERSION;
}

#if defined(DEBUG) || defined(_DEBUG)
#define BT_DEBUG
#endif

#ifdef _WIN32

#if defined(__MINGW32__) || defined(__CYGWIN__) || (defined(_MSC_VER) && _MSC_VER < 1300)

#define SIMD_FORCE_INLINE inline
#define ATTRIBUTE_ALIGNED16(a) a
#define ATTRIBUTE_ALIGNED64(a) a
#define ATTRIBUTE_ALIGNED128(a) a
#else
//#define BT_HAS_ALIGNED_ALLOCATOR
#pragma warning(disable : 4324) // disable padding warning
//			#pragma warning(disable:4530) // Disable the exception disable but used in MSCV Stl warning.
//			#pragma warning(disable:4996) //Turn off warnings about deprecated C routines
//			#pragma warning(disable:4786) // Disable the "debug name too long" warning

#define SIMD_FORCE_INLINE __forceinline
#define ATTRIBUTE_ALIGNED16(a) __declspec(align(16)) a
#define ATTRIBUTE_ALIGNED64(a) __declspec(align(64)) a
#define ATTRIBUTE_ALIGNED128(a) __declspec(align(128)) a
#ifdef _XBOX
#define BT_USE_VMX128

#include <ppcintrinsics.h>
#define BT_HAVE_NATIVE_FSEL
#define btFsel(a, b, c) __fsel((a), (b), (c))
#else

#if (defined(_WIN32) && (_MSC_VER) && _MSC_VER >= 1400) && (!defined(BT_USE_DOUBLE_PRECISION))
#define BT_USE_SSE
#include <emmintrin.h>
#endif

#endif //_XBOX

#endif //__MINGW32__

#include <assert.h>
#ifdef BT_DEBUG
#define btAssert assert
#else
#define btAssert(x)
#endif
//btFullAssert is optional, slows down a lot
#define btFullAssert(x)

#define btLikely(_c) _c
#define btUnlikely(_c) _c

#else

#if defined(__CELLOS_LV2__)
#define SIMD_FORCE_INLINE inline __attribute__((always_inline))
#define ATTRIBUTE_ALIGNED16(a) a __attribute__((aligned(16)))
#define ATTRIBUTE_ALIGNED64(a) a __attribute__((aligned(64)))
#define ATTRIBUTE_ALIGNED128(a) a __attribute__((aligned(128)))
#ifndef assert
#include <assert.h>
#endif
#ifdef BT_DEBUG
#ifdef __SPU__
#include <spu_printf.h>
#define printf spu_printf
#define btAssert(x)                                                \
    {                                                              \
        if (!(x)) {                                                \
            printf("Assert " __FILE__ ":%u (" #x ")\n", __LINE__); \
            spu_hcmpeq(0, 0);                                      \
        }                                                          \
    }
#else
#define btAssert assert
#endif

#else
#define btAssert(x)
#endif
//btFullAssert is optional, slows down a lot
#define btFullAssert(x)

#define btLikely(_c) _c
#define btUnlikely(_c) _c

#else

#ifdef USE_LIBSPE2

#define SIMD_FORCE_INLINE __inline
#define ATTRIBUTE_ALIGNED16(a) a __attribute__((aligned(16)))
#define ATTRIBUTE_ALIGNED64(a) a __attribute__((aligned(64)))
#define ATTRIBUTE_ALIGNED128(a) a __attribute__((aligned(128)))
#ifndef assert
#include <assert.h>
#endif
#ifdef BT_DEBUG
#define btAssert assert
#else
#define btAssert(x)
#endif
//btFullAssert is optional, slows down a lot
#define btFullAssert(x)

#define btLikely(_c) __builtin_expect((_c), 1)
#define btUnlikely(_c) __builtin_expect((_c), 0)

#else
//non-windows systems

#if (defined(__APPLE__) && defined(__i386__) && (!defined(BT_USE_DOUBLE_PRECISION)))
#define BT_USE_SSE
#include <emmintrin.h>

#define SIMD_FORCE_INLINE inline
///@todo: check out alignment methods for other platforms/compilers
#define ATTRIBUTE_ALIGNED16(a) a __attribute__((aligned(16)))
#define ATTRIBUTE_ALIGNED64(a) a __attribute__((aligned(64)))
#define ATTRIBUTE_ALIGNED128(a) a __attribute__((aligned(128)))
#ifndef assert
#include <assert.h>
#endif

#if defined(DEBUG) || defined(_DEBUG)
#define btAssert assert
#else
#define btAssert(x)
#endif

//btFullAssert is optional, slows down a lot
#define btFullAssert(x)
#define btLikely(_c) _c
#define btUnlikely(_c) _c

#else

#define SIMD_FORCE_INLINE inline
///@todo: check out alignment methods for other platforms/compilers
///#define ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
///#define ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
///#define ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
#define ATTRIBUTE_ALIGNED16(a) a
#define ATTRIBUTE_ALIGNED64(a) a
#define ATTRIBUTE_ALIGNED128(a) a
#ifndef assert
#include <assert.h>
#endif

#if defined(DEBUG) || defined(_DEBUG)
#define btAssert assert
#else
#define btAssert(x)
#endif

//btFullAssert is optional, slows down a lot
#define btFullAssert(x)
#define btLikely(_c) _c
#define btUnlikely(_c) _c
#endif //__APPLE__

#endif // LIBSPE2

#endif //__CELLOS_LV2__
#endif

///The btScalar type abstracts floating point numbers, to easily switch between double and single floating point precision.
#if defined(BT_USE_DOUBLE_PRECISION)
typedef double btScalar;
//this number could be bigger in double precision
#define BT_LARGE_FLOAT 1e30
#else
typedef float btScalar;
//keep BT_LARGE_FLOAT*BT_LARGE_FLOAT < FLT_MAX
#define BT_LARGE_FLOAT 1e18f
#endif

#define BT_DECLARE_ALIGNED_ALLOCATOR()                                                                     \
    SIMD_FORCE_INLINE void* operator new(size_t sizeInBytes) { return btAlignedAlloc(sizeInBytes, 16); }   \
    SIMD_FORCE_INLINE void operator delete(void* ptr) { btAlignedFree(ptr); }                              \
    SIMD_FORCE_INLINE void* operator new(size_t, void* ptr) { return ptr; }                                \
    SIMD_FORCE_INLINE void operator delete(void*, void*) {}                                                \
    SIMD_FORCE_INLINE void* operator new[](size_t sizeInBytes) { return btAlignedAlloc(sizeInBytes, 16); } \
    SIMD_FORCE_INLINE void operator delete[](void* ptr) { btAlignedFree(ptr); }                            \
    SIMD_FORCE_INLINE void* operator new[](size_t, void* ptr) { return ptr; }                              \
    SIMD_FORCE_INLINE void operator delete[](void*, void*) {}

#if defined(BT_USE_DOUBLE_PRECISION) || defined(BT_FORCE_DOUBLE_FUNCTIONS)

SIMD_FORCE_INLINE btScalar btSqrt(btScalar x)
{
    return sqrt(x);
}
SIMD_FORCE_INLINE btScalar btFabs(btScalar x) { return fabs(x); }
SIMD_FORCE_INLINE btScalar btCos(btScalar x) { return cos(x); }
SIMD_FORCE_INLINE btScalar btSin(btScalar x) { return sin(x); }
SIMD_FORCE_INLINE btScalar btTan(btScalar x) { return tan(x); }
SIMD_FORCE_INLINE btScalar btAcos(btScalar x)
{
    if (x < btScalar(-1))
        x = btScalar(-1);
    if (x > btScalar(1))
        x = btScalar(1);
    return acos(x);
}
SIMD_FORCE_INLINE btScalar btAsin(btScalar x)
{
    if (x < btScalar(-1))
        x = btScalar(-1);
    if (x > btScalar(1))
        x = btScalar(1);
    return asin(x);
}
SIMD_FORCE_INLINE btScalar btAtan(btScalar x) { return atan(x); }
SIMD_FORCE_INLINE btScalar btAtan2(btScalar x, btScalar y) { return atan2(x, y); }
SIMD_FORCE_INLINE btScalar btExp(btScalar x) { return exp(x); }
SIMD_FORCE_INLINE btScalar btLog(btScalar x) { return log(x); }
SIMD_FORCE_INLINE btScalar btPow(btScalar x, btScalar y) { return pow(x, y); }
SIMD_FORCE_INLINE btScalar btFmod(btScalar x, btScalar y) { return fmod(x, y); }

#else

SIMD_FORCE_INLINE btScalar btSqrt(btScalar y)
{
#ifdef USE_APPROXIMATION
    double x, z, tempf;
    unsigned long* tfptr = ((unsigned long*)&tempf) + 1;

    tempf = y;
    *tfptr = (0xbfcdd90a - *tfptr) >> 1; /* estimate of 1/sqrt(y) */
    x = tempf;
    z = y * btScalar(0.5);
    x = (btScalar(1.5) * x) - (x * x) * (x * z); /* iteration formula     */
    x = (btScalar(1.5) * x) - (x * x) * (x * z);
    x = (btScalar(1.5) * x) - (x * x) * (x * z);
    x = (btScalar(1.5) * x) - (x * x) * (x * z);
    x = (btScalar(1.5) * x) - (x * x) * (x * z);
    return x * y;
#else
    return sqrtf(y);
#endif
}
SIMD_FORCE_INLINE btScalar btFabs(btScalar x) { return fabsf(x); }
SIMD_FORCE_INLINE btScalar btCos(btScalar x) { return cosf(x); }
SIMD_FORCE_INLINE btScalar btSin(btScalar x) { return sinf(x); }
SIMD_FORCE_INLINE btScalar btTan(btScalar x) { return tanf(x); }
SIMD_FORCE_INLINE btScalar btAcos(btScalar x)
{
    if (x < btScalar(-1))
        x = btScalar(-1);
    if (x > btScalar(1))
        x = btScalar(1);
    return acosf(x);
}
SIMD_FORCE_INLINE btScalar btAsin(btScalar x)
{
    if (x < btScalar(-1))
        x = btScalar(-1);
    if (x > btScalar(1))
        x = btScalar(1);
    return asinf(x);
}
SIMD_FORCE_INLINE btScalar btAtan(btScalar x) { return atanf(x); }
SIMD_FORCE_INLINE btScalar btAtan2(btScalar x, btScalar y) { return atan2f(x, y); }
SIMD_FORCE_INLINE btScalar btExp(btScalar x) { return expf(x); }
SIMD_FORCE_INLINE btScalar btLog(btScalar x) { return logf(x); }
SIMD_FORCE_INLINE btScalar btPow(btScalar x, btScalar y) { return powf(x, y); }
SIMD_FORCE_INLINE btScalar btFmod(btScalar x, btScalar y) { return fmodf(x, y); }

#endif

#define SIMD_2_PI btScalar(6.283185307179586232)
#define SIMD_PI (SIMD_2_PI * btScalar(0.5))
#define SIMD_HALF_PI (SIMD_2_PI * btScalar(0.25))
#define SIMD_RADS_PER_DEG (SIMD_2_PI / btScalar(360.0))
#define SIMD_DEGS_PER_RAD (btScalar(360.0) / SIMD_2_PI)
#define SIMDSQRT12 btScalar(0.7071067811865475244008443621048490)

#define btRecipSqrt(x) ((btScalar)(btScalar(1.0) / btSqrt(btScalar(x)))) /* reciprocal square root */

#ifdef BT_USE_DOUBLE_PRECISION
#define SIMD_EPSILON DBL_EPSILON
#define SIMD_INFINITY DBL_MAX
#else
#define SIMD_EPSILON FLT_EPSILON
#define SIMD_INFINITY FLT_MAX
#endif

SIMD_FORCE_INLINE btScalar btAtan2Fast(btScalar y, btScalar x)
{
    btScalar coeff_1 = SIMD_PI / 4.0f;
    btScalar coeff_2 = 3.0f * coeff_1;
    btScalar abs_y = btFabs(y);
    btScalar angle;
    if (x >= 0.0f) {
        btScalar r = (x - abs_y) / (x + abs_y);
        angle = coeff_1 - coeff_1 * r;
    }
    else {
        btScalar r = (x + abs_y) / (abs_y - x);
        angle = coeff_2 - coeff_1 * r;
    }
    return (y < 0.0f) ? -angle : angle;
}

SIMD_FORCE_INLINE bool btFuzzyZero(btScalar x) { return btFabs(x) < SIMD_EPSILON; }

SIMD_FORCE_INLINE bool btEqual(btScalar a, btScalar eps)
{
    return (((a) <= eps) && !((a) < -eps));
}
SIMD_FORCE_INLINE bool btGreaterEqual(btScalar a, btScalar eps)
{
    return (!((a) <= eps));
}

SIMD_FORCE_INLINE int32_t btIsNegative(btScalar x)
{
    return x < btScalar(0.0) ? 1 : 0;
}

SIMD_FORCE_INLINE btScalar btRadians(btScalar x) { return x * SIMD_RADS_PER_DEG; }
SIMD_FORCE_INLINE btScalar btDegrees(btScalar x) { return x * SIMD_DEGS_PER_RAD; }

#define BT_DECLARE_HANDLE(name) \
    typedef struct name##__ {   \
        int32_t unused;             \
    } * name

#ifndef btFsel
SIMD_FORCE_INLINE btScalar btFsel(btScalar a, btScalar b, btScalar c)
{
    return a >= 0 ? b : c;
}
#endif
#define btFsels(a, b, c) (btScalar) btFsel(a, b, c)

SIMD_FORCE_INLINE bool btMachineIsLittleEndian()
{
    long int i = 1;
    const char* p = (const char*)&i;
    if (p[0] == 1) // Lowest address contains the least significant byte
        return true;
    else
        return false;
}

///btSelect avoids branches, which makes performance much better for consoles like Playstation 3 and XBox 360
///Thanks Phil Knight. See also http://www.cellperformance.com/articles/2006/04/more_techniques_for_eliminatin_1.html
SIMD_FORCE_INLINE unsigned btSelect(unsigned condition, unsigned valueIfConditionNonZero, unsigned valueIfConditionZero)
{
    // Set testNz to 0xFFFFFFFF if condition is nonzero, 0x00000000 if condition is zero
    // Rely on positive value or'ed with its negative having sign bit on
    // and zero value or'ed with its negative (which is still zero) having sign bit off
    // Use arithmetic shift right, shifting the sign bit through all 32 bits
    unsigned testNz = (unsigned)(((int32_t)condition | -(int32_t)condition) >> 31);
    unsigned testEqz = ~testNz;
    return ((valueIfConditionNonZero & testNz) | (valueIfConditionZero & testEqz));
}
SIMD_FORCE_INLINE int32_t btSelect(unsigned condition, int32_t valueIfConditionNonZero, int32_t valueIfConditionZero)
{
    unsigned testNz = (unsigned)(((int32_t)condition | -(int32_t)condition) >> 31);
    unsigned testEqz = ~testNz;
    return static_cast<int32_t>((valueIfConditionNonZero & testNz) | (valueIfConditionZero & testEqz));
}
SIMD_FORCE_INLINE float btSelect(unsigned condition, float valueIfConditionNonZero, float valueIfConditionZero)
{
#ifdef BT_HAVE_NATIVE_FSEL
    return (float)btFsel((btScalar)condition - btScalar(1.0f), valueIfConditionNonZero, valueIfConditionZero);
#else
    return (condition != 0) ? valueIfConditionNonZero : valueIfConditionZero;
#endif
}

template <typename T>
SIMD_FORCE_INLINE void btSwap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

//PCK: endian swapping functions
SIMD_FORCE_INLINE unsigned btSwapEndian(unsigned val)
{
    return (((val & 0xff000000) >> 24) | ((val & 0x00ff0000) >> 8) | ((val & 0x0000ff00) << 8) | ((val & 0x000000ff) << 24));
}

SIMD_FORCE_INLINE unsigned short btSwapEndian(unsigned short val)
{
    return static_cast<unsigned short>(((val & 0xff00) >> 8) | ((val & 0x00ff) << 8));
}

SIMD_FORCE_INLINE unsigned btSwapEndian(int32_t val)
{
    return btSwapEndian((unsigned)val);
}

SIMD_FORCE_INLINE unsigned short btSwapEndian(short val)
{
    return btSwapEndian((unsigned short)val);
}

///btSwapFloat uses using char pointers to swap the endianness
////btSwapFloat/btSwapDouble will NOT return a float, because the machine might 'correct' invalid floating point values
///Not all values of sign/exponent/mantissa are valid floating point numbers according to IEEE 754.
///When a floating point unit is faced with an invalid value, it may actually change the value, or worse, throw an exception.
///In most systems, running user mode code, you wouldn't get an exception, but instead the hardware/os/runtime will 'fix' the number for you.
///so instead of returning a float/double, we return integer/long long integer
SIMD_FORCE_INLINE uint32_t btSwapEndianFloat(float d)
{
    uint32_t a = 0;
    unsigned char* dst = (unsigned char*)&a;
    unsigned char* src = (unsigned char*)&d;

    dst[0] = src[3];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[0];
    return a;
}

// unswap using char pointers
SIMD_FORCE_INLINE float btUnswapEndianFloat(uint32_t a)
{
    float d = 0.0f;
    unsigned char* src = (unsigned char*)&a;
    unsigned char* dst = (unsigned char*)&d;

    dst[0] = src[3];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[0];

    return d;
}

// swap using char pointers
SIMD_FORCE_INLINE void btSwapEndianDouble(double d, unsigned char* dst)
{
    unsigned char* src = (unsigned char*)&d;

    dst[0] = src[7];
    dst[1] = src[6];
    dst[2] = src[5];
    dst[3] = src[4];
    dst[4] = src[3];
    dst[5] = src[2];
    dst[6] = src[1];
    dst[7] = src[0];
}

// unswap using char pointers
SIMD_FORCE_INLINE double btUnswapEndianDouble(const unsigned char* src)
{
    double d = 0.0;
    unsigned char* dst = (unsigned char*)&d;

    dst[0] = src[7];
    dst[1] = src[6];
    dst[2] = src[5];
    dst[3] = src[4];
    dst[4] = src[3];
    dst[5] = src[2];
    dst[6] = src[1];
    dst[7] = src[0];

    return d;
}

// returns normalized value in range [-SIMD_PI, SIMD_PI]
SIMD_FORCE_INLINE btScalar btNormalizeAngle(btScalar angleInRadians)
{
    angleInRadians = btFmod(angleInRadians, SIMD_2_PI);
    if (angleInRadians < -SIMD_PI) {
        return angleInRadians + SIMD_2_PI;
    }
    else if (angleInRadians > SIMD_PI) {
        return angleInRadians - SIMD_2_PI;
    }
    else {
        return angleInRadians;
    }
}

///rudimentary class to provide type info
struct btTypedObject {
    btTypedObject(int32_t objectType)
        : m_objectType(objectType)
    {
    }
    int32_t m_objectType;
    inline int32_t getObjectType() const
    {
        return m_objectType;
    }
};
#endif //BT_SCALAR_H
