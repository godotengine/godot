/*
Copyright (c) 2003-2013 Gino van den Bergen / Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/



#ifndef B3_SCALAR_H
#define B3_SCALAR_H

#ifdef B3_MANAGED_CODE
//Aligned data types not supported in managed code
#pragma unmanaged
#endif



#include <math.h>
#include <stdlib.h>//size_t for MSVC 6.0
#include <float.h>

//Original repository is at http://github.com/erwincoumans/bullet3
#define B3_BULLET_VERSION 300

inline int	b3GetVersion()
{
	return B3_BULLET_VERSION;
}

#if defined(DEBUG) || defined (_DEBUG)
#define B3_DEBUG
#endif

#include "b3Logging.h"//for b3Error


#ifdef _WIN32

		#if defined(__MINGW32__) || defined(__CYGWIN__) || (defined (_MSC_VER) && _MSC_VER < 1300)

			#define B3_FORCE_INLINE inline
			#define B3_ATTRIBUTE_ALIGNED16(a) a
			#define B3_ATTRIBUTE_ALIGNED64(a) a
			#define B3_ATTRIBUTE_ALIGNED128(a) a
		#else
			//#define B3_HAS_ALIGNED_ALLOCATOR
			#pragma warning(disable : 4324) // disable padding warning
//			#pragma warning(disable:4530) // Disable the exception disable but used in MSCV Stl warning.
			#pragma warning(disable:4996) //Turn off warnings about deprecated C routines
//			#pragma warning(disable:4786) // Disable the "debug name too long" warning

			#define B3_FORCE_INLINE __forceinline
			#define B3_ATTRIBUTE_ALIGNED16(a) __declspec(align(16)) a
			#define B3_ATTRIBUTE_ALIGNED64(a) __declspec(align(64)) a
			#define B3_ATTRIBUTE_ALIGNED128(a) __declspec (align(128)) a
		#ifdef _XBOX
			#define B3_USE_VMX128

			#include <ppcintrinsics.h>
 			#define B3_HAVE_NATIVE_FSEL
 			#define b3Fsel(a,b,c) __fsel((a),(b),(c))
		#else

#if (defined (_WIN32) && (_MSC_VER) && _MSC_VER >= 1400) && (!defined (B3_USE_DOUBLE_PRECISION))
	#if (defined (_M_IX86) || defined (_M_X64))
			#define B3_USE_SSE
			#ifdef B3_USE_SSE
			//B3_USE_SSE_IN_API is disabled under Windows by default, because 
			//it makes it harder to integrate Bullet into your application under Windows 
			//(structured embedding Bullet structs/classes need to be 16-byte aligned)
			//with relatively little performance gain
			//If you are not embedded Bullet data in your classes, or make sure that you align those classes on 16-byte boundaries
			//you can manually enable this line or set it in the build system for a bit of performance gain (a few percent, dependent on usage)
			//#define B3_USE_SSE_IN_API
			#endif //B3_USE_SSE
			#include <emmintrin.h>
	#endif
#endif

		#endif//_XBOX

		#endif //__MINGW32__

#ifdef B3_DEBUG
	#ifdef _MSC_VER
		#include <stdio.h>
		#define b3Assert(x) { if(!(x)){b3Error("Assert "__FILE__ ":%u ("#x")\n", __LINE__);__debugbreak();	}}
	#else//_MSC_VER
		#include <assert.h>
		#define b3Assert assert
	#endif//_MSC_VER
#else
		#define b3Assert(x)
#endif
		//b3FullAssert is optional, slows down a lot
		#define b3FullAssert(x)

		#define b3Likely(_c)  _c
		#define b3Unlikely(_c) _c

#else
	
#if defined	(__CELLOS_LV2__)
		#define B3_FORCE_INLINE inline __attribute__((always_inline))
		#define B3_ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
		#define B3_ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
		#define B3_ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
		#ifndef assert
		#include <assert.h>
		#endif
#ifdef B3_DEBUG
#ifdef __SPU__
#include <spu_printf.h>
#define printf spu_printf
	#define b3Assert(x) {if(!(x)){b3Error("Assert "__FILE__ ":%u ("#x")\n", __LINE__);spu_hcmpeq(0,0);}}
#else
	#define b3Assert assert
#endif
	
#else
		#define b3Assert(x)
#endif
		//b3FullAssert is optional, slows down a lot
		#define b3FullAssert(x)

		#define b3Likely(_c)  _c
		#define b3Unlikely(_c) _c

#else

#ifdef USE_LIBSPE2

		#define B3_FORCE_INLINE __inline
		#define B3_ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
		#define B3_ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
		#define B3_ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
		#ifndef assert
		#include <assert.h>
		#endif
#ifdef B3_DEBUG
		#define b3Assert assert
#else
		#define b3Assert(x)
#endif
		//b3FullAssert is optional, slows down a lot
		#define b3FullAssert(x)


		#define b3Likely(_c)   __builtin_expect((_c), 1)
		#define b3Unlikely(_c) __builtin_expect((_c), 0)
		

#else
	//non-windows systems

#if (defined (__APPLE__) && (!defined (B3_USE_DOUBLE_PRECISION)))
    #if defined (__i386__) || defined (__x86_64__)
        #define B3_USE_SSE
		//B3_USE_SSE_IN_API is enabled on Mac OSX by default, because memory is automatically aligned on 16-byte boundaries
		//if apps run into issues, we will disable the next line
		#define B3_USE_SSE_IN_API
        #ifdef B3_USE_SSE
            // include appropriate SSE level
            #if defined (__SSE4_1__)
                #include <smmintrin.h>
            #elif defined (__SSSE3__)
                #include <tmmintrin.h>
            #elif defined (__SSE3__)
                #include <pmmintrin.h>
            #else
                #include <emmintrin.h>
            #endif
        #endif //B3_USE_SSE
    #elif defined( __armv7__ )
        #ifdef __clang__
            #define B3_USE_NEON 1

            #if defined B3_USE_NEON && defined (__clang__)
                #include <arm_neon.h>
            #endif//B3_USE_NEON
       #endif //__clang__
    #endif//__arm__

	#define B3_FORCE_INLINE inline __attribute__ ((always_inline))
///@todo: check out alignment methods for other platforms/compilers
	#define B3_ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
	#define B3_ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
	#define B3_ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
	#ifndef assert
	#include <assert.h>
	#endif

	#if defined(DEBUG) || defined (_DEBUG)
	 #if defined (__i386__) || defined (__x86_64__)
	#include <stdio.h>
	 #define b3Assert(x)\
	{\
	if(!(x))\
	{\
		b3Error("Assert %s in line %d, file %s\n",#x, __LINE__, __FILE__);\
		asm volatile ("int3");\
	}\
	}
	#else//defined (__i386__) || defined (__x86_64__)
		#define b3Assert assert
	#endif//defined (__i386__) || defined (__x86_64__)
	#else//defined(DEBUG) || defined (_DEBUG)
		#define b3Assert(x)
	#endif//defined(DEBUG) || defined (_DEBUG)

	//b3FullAssert is optional, slows down a lot
	#define b3FullAssert(x)
	#define b3Likely(_c)  _c
	#define b3Unlikely(_c) _c

#else

		#define B3_FORCE_INLINE inline
		///@todo: check out alignment methods for other platforms/compilers
		#define B3_ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
		#define B3_ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
		#define B3_ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
		///#define B3_ATTRIBUTE_ALIGNED16(a) a
		///#define B3_ATTRIBUTE_ALIGNED64(a) a
		///#define B3_ATTRIBUTE_ALIGNED128(a) a
		#ifndef assert
		#include <assert.h>
		#endif

#if defined(DEBUG) || defined (_DEBUG)
		#define b3Assert assert
#else
		#define b3Assert(x)
#endif

		//b3FullAssert is optional, slows down a lot
		#define b3FullAssert(x)
		#define b3Likely(_c)  _c
		#define b3Unlikely(_c) _c
#endif //__APPLE__ 

#endif // LIBSPE2

#endif	//__CELLOS_LV2__
#endif


///The b3Scalar type abstracts floating point numbers, to easily switch between double and single floating point precision.
#if defined(B3_USE_DOUBLE_PRECISION)
typedef double b3Scalar;
//this number could be bigger in double precision
#define B3_LARGE_FLOAT 1e30
#else
typedef float b3Scalar;
//keep B3_LARGE_FLOAT*B3_LARGE_FLOAT < FLT_MAX
#define B3_LARGE_FLOAT 1e18f
#endif

#ifdef B3_USE_SSE
typedef __m128 b3SimdFloat4;
#endif//B3_USE_SSE

#if defined B3_USE_SSE_IN_API && defined (B3_USE_SSE)
#ifdef _WIN32

#ifndef B3_NAN
static int b3NanMask = 0x7F800001;
#define B3_NAN (*(float*)&b3NanMask)
#endif

#ifndef B3_INFINITY_MASK
static  int b3InfinityMask = 0x7F800000;
#define B3_INFINITY_MASK (*(float*)&b3InfinityMask)
#endif

inline __m128 operator + (const __m128 A, const __m128 B)
{
    return _mm_add_ps(A, B);
}

inline __m128 operator - (const __m128 A, const __m128 B)
{
    return _mm_sub_ps(A, B);
}

inline __m128 operator * (const __m128 A, const __m128 B)
{
    return _mm_mul_ps(A, B);
}

#define b3CastfTo128i(a) (_mm_castps_si128(a))
#define b3CastfTo128d(a) (_mm_castps_pd(a))
#define b3CastiTo128f(a) (_mm_castsi128_ps(a))
#define b3CastdTo128f(a) (_mm_castpd_ps(a))
#define b3CastdTo128i(a) (_mm_castpd_si128(a))
#define b3Assign128(r0,r1,r2,r3) _mm_setr_ps(r0,r1,r2,r3)

#else//_WIN32

#define b3CastfTo128i(a) ((__m128i)(a))
#define b3CastfTo128d(a) ((__m128d)(a))
#define b3CastiTo128f(a)  ((__m128) (a))
#define b3CastdTo128f(a) ((__m128) (a))
#define b3CastdTo128i(a) ((__m128i)(a))
#define b3Assign128(r0,r1,r2,r3) (__m128){r0,r1,r2,r3}
#endif//_WIN32
#endif //B3_USE_SSE_IN_API

#ifdef B3_USE_NEON
#include <arm_neon.h>

typedef float32x4_t b3SimdFloat4;
#define B3_INFINITY INFINITY
#define B3_NAN NAN
#define b3Assign128(r0,r1,r2,r3) (float32x4_t){r0,r1,r2,r3}
#endif





#define B3_DECLARE_ALIGNED_ALLOCATOR() \
   B3_FORCE_INLINE void* operator new(size_t sizeInBytes)   { return b3AlignedAlloc(sizeInBytes,16); }   \
   B3_FORCE_INLINE void  operator delete(void* ptr)         { b3AlignedFree(ptr); }   \
   B3_FORCE_INLINE void* operator new(size_t, void* ptr)   { return ptr; }   \
   B3_FORCE_INLINE void  operator delete(void*, void*)      { }   \
   B3_FORCE_INLINE void* operator new[](size_t sizeInBytes)   { return b3AlignedAlloc(sizeInBytes,16); }   \
   B3_FORCE_INLINE void  operator delete[](void* ptr)         { b3AlignedFree(ptr); }   \
   B3_FORCE_INLINE void* operator new[](size_t, void* ptr)   { return ptr; }   \
   B3_FORCE_INLINE void  operator delete[](void*, void*)      { }   \



#if defined(B3_USE_DOUBLE_PRECISION) || defined(B3_FORCE_DOUBLE_FUNCTIONS)
		
B3_FORCE_INLINE b3Scalar b3Sqrt(b3Scalar x) { return sqrt(x); }
B3_FORCE_INLINE b3Scalar b3Fabs(b3Scalar x) { return fabs(x); }
B3_FORCE_INLINE b3Scalar b3Cos(b3Scalar x) { return cos(x); }
B3_FORCE_INLINE b3Scalar b3Sin(b3Scalar x) { return sin(x); }
B3_FORCE_INLINE b3Scalar b3Tan(b3Scalar x) { return tan(x); }
B3_FORCE_INLINE b3Scalar b3Acos(b3Scalar x) { if (x<b3Scalar(-1))	x=b3Scalar(-1); if (x>b3Scalar(1))	x=b3Scalar(1); return acos(x); }
B3_FORCE_INLINE b3Scalar b3Asin(b3Scalar x) { if (x<b3Scalar(-1))	x=b3Scalar(-1); if (x>b3Scalar(1))	x=b3Scalar(1); return asin(x); }
B3_FORCE_INLINE b3Scalar b3Atan(b3Scalar x) { return atan(x); }
B3_FORCE_INLINE b3Scalar b3Atan2(b3Scalar x, b3Scalar y) { return atan2(x, y); }
B3_FORCE_INLINE b3Scalar b3Exp(b3Scalar x) { return exp(x); }
B3_FORCE_INLINE b3Scalar b3Log(b3Scalar x) { return log(x); }
B3_FORCE_INLINE b3Scalar b3Pow(b3Scalar x,b3Scalar y) { return pow(x,y); }
B3_FORCE_INLINE b3Scalar b3Fmod(b3Scalar x,b3Scalar y) { return fmod(x,y); }

#else
		
B3_FORCE_INLINE b3Scalar b3Sqrt(b3Scalar y) 
{ 
#ifdef USE_APPROXIMATION
    double x, z, tempf;
    unsigned long *tfptr = ((unsigned long *)&tempf) + 1;

	tempf = y;
	*tfptr = (0xbfcdd90a - *tfptr)>>1; /* estimate of 1/sqrt(y) */
	x =  tempf;
	z =  y*b3Scalar(0.5);
	x = (b3Scalar(1.5)*x)-(x*x)*(x*z);         /* iteration formula     */
	x = (b3Scalar(1.5)*x)-(x*x)*(x*z);
	x = (b3Scalar(1.5)*x)-(x*x)*(x*z);
	x = (b3Scalar(1.5)*x)-(x*x)*(x*z);
	x = (b3Scalar(1.5)*x)-(x*x)*(x*z);
	return x*y;
#else
	return sqrtf(y); 
#endif
}
B3_FORCE_INLINE b3Scalar b3Fabs(b3Scalar x) { return fabsf(x); }
B3_FORCE_INLINE b3Scalar b3Cos(b3Scalar x) { return cosf(x); }
B3_FORCE_INLINE b3Scalar b3Sin(b3Scalar x) { return sinf(x); }
B3_FORCE_INLINE b3Scalar b3Tan(b3Scalar x) { return tanf(x); }
B3_FORCE_INLINE b3Scalar b3Acos(b3Scalar x) { 
	if (x<b3Scalar(-1))	
		x=b3Scalar(-1); 
	if (x>b3Scalar(1))	
		x=b3Scalar(1);
	return acosf(x); 
}
B3_FORCE_INLINE b3Scalar b3Asin(b3Scalar x) { 
	if (x<b3Scalar(-1))	
		x=b3Scalar(-1); 
	if (x>b3Scalar(1))	
		x=b3Scalar(1);
	return asinf(x); 
}
B3_FORCE_INLINE b3Scalar b3Atan(b3Scalar x) { return atanf(x); }
B3_FORCE_INLINE b3Scalar b3Atan2(b3Scalar x, b3Scalar y) { return atan2f(x, y); }
B3_FORCE_INLINE b3Scalar b3Exp(b3Scalar x) { return expf(x); }
B3_FORCE_INLINE b3Scalar b3Log(b3Scalar x) { return logf(x); }
B3_FORCE_INLINE b3Scalar b3Pow(b3Scalar x,b3Scalar y) { return powf(x,y); }
B3_FORCE_INLINE b3Scalar b3Fmod(b3Scalar x,b3Scalar y) { return fmodf(x,y); }
	
#endif

#define B3_2_PI         b3Scalar(6.283185307179586232)
#define B3_PI           (B3_2_PI * b3Scalar(0.5))
#define B3_HALF_PI      (B3_2_PI * b3Scalar(0.25))
#define B3_RADS_PER_DEG (B3_2_PI / b3Scalar(360.0))
#define B3_DEGS_PER_RAD  (b3Scalar(360.0) / B3_2_PI)
#define B3_SQRT12 b3Scalar(0.7071067811865475244008443621048490)

#define b3RecipSqrt(x) ((b3Scalar)(b3Scalar(1.0)/b3Sqrt(b3Scalar(x))))		/* reciprocal square root */


#ifdef B3_USE_DOUBLE_PRECISION
#define B3_EPSILON      DBL_EPSILON
#define B3_INFINITY     DBL_MAX
#else
#define B3_EPSILON      FLT_EPSILON
#define B3_INFINITY     FLT_MAX
#endif

B3_FORCE_INLINE b3Scalar b3Atan2Fast(b3Scalar y, b3Scalar x) 
{
	b3Scalar coeff_1 = B3_PI / 4.0f;
	b3Scalar coeff_2 = 3.0f * coeff_1;
	b3Scalar abs_y = b3Fabs(y);
	b3Scalar angle;
	if (x >= 0.0f) {
		b3Scalar r = (x - abs_y) / (x + abs_y);
		angle = coeff_1 - coeff_1 * r;
	} else {
		b3Scalar r = (x + abs_y) / (abs_y - x);
		angle = coeff_2 - coeff_1 * r;
	}
	return (y < 0.0f) ? -angle : angle;
}

B3_FORCE_INLINE bool      b3FuzzyZero(b3Scalar x) { return b3Fabs(x) < B3_EPSILON; }

B3_FORCE_INLINE bool	b3Equal(b3Scalar a, b3Scalar eps) {
	return (((a) <= eps) && !((a) < -eps));
}
B3_FORCE_INLINE bool	b3GreaterEqual (b3Scalar a, b3Scalar eps) {
	return (!((a) <= eps));
}


B3_FORCE_INLINE int       b3IsNegative(b3Scalar x) {
    return x < b3Scalar(0.0) ? 1 : 0;
}

B3_FORCE_INLINE b3Scalar b3Radians(b3Scalar x) { return x * B3_RADS_PER_DEG; }
B3_FORCE_INLINE b3Scalar b3Degrees(b3Scalar x) { return x * B3_DEGS_PER_RAD; }

#define B3_DECLARE_HANDLE(name) typedef struct name##__ { int unused; } *name

#ifndef b3Fsel
B3_FORCE_INLINE b3Scalar b3Fsel(b3Scalar a, b3Scalar b, b3Scalar c)
{
	return a >= 0 ? b : c;
}
#endif
#define b3Fsels(a,b,c) (b3Scalar)b3Fsel(a,b,c)


B3_FORCE_INLINE bool b3MachineIsLittleEndian()
{
   long int i = 1;
   const char *p = (const char *) &i;
   if (p[0] == 1)  // Lowest address contains the least significant byte
	   return true;
   else
	   return false;
}



///b3Select avoids branches, which makes performance much better for consoles like Playstation 3 and XBox 360
///Thanks Phil Knight. See also http://www.cellperformance.com/articles/2006/04/more_techniques_for_eliminatin_1.html
B3_FORCE_INLINE unsigned b3Select(unsigned condition, unsigned valueIfConditionNonZero, unsigned valueIfConditionZero) 
{
    // Set testNz to 0xFFFFFFFF if condition is nonzero, 0x00000000 if condition is zero
    // Rely on positive value or'ed with its negative having sign bit on
    // and zero value or'ed with its negative (which is still zero) having sign bit off 
    // Use arithmetic shift right, shifting the sign bit through all 32 bits
    unsigned testNz = (unsigned)(((int)condition | -(int)condition) >> 31);
    unsigned testEqz = ~testNz;
    return ((valueIfConditionNonZero & testNz) | (valueIfConditionZero & testEqz)); 
}
B3_FORCE_INLINE int b3Select(unsigned condition, int valueIfConditionNonZero, int valueIfConditionZero)
{
    unsigned testNz = (unsigned)(((int)condition | -(int)condition) >> 31);
    unsigned testEqz = ~testNz; 
    return static_cast<int>((valueIfConditionNonZero & testNz) | (valueIfConditionZero & testEqz));
}
B3_FORCE_INLINE float b3Select(unsigned condition, float valueIfConditionNonZero, float valueIfConditionZero)
{
#ifdef B3_HAVE_NATIVE_FSEL
    return (float)b3Fsel((b3Scalar)condition - b3Scalar(1.0f), valueIfConditionNonZero, valueIfConditionZero);
#else
    return (condition != 0) ? valueIfConditionNonZero : valueIfConditionZero; 
#endif
}

template<typename T> B3_FORCE_INLINE void b3Swap(T& a, T& b)
{
	T tmp = a;
	a = b;
	b = tmp;
}


//PCK: endian swapping functions
B3_FORCE_INLINE unsigned b3SwapEndian(unsigned val)
{
	return (((val & 0xff000000) >> 24) | ((val & 0x00ff0000) >> 8) | ((val & 0x0000ff00) << 8)  | ((val & 0x000000ff) << 24));
}

B3_FORCE_INLINE unsigned short b3SwapEndian(unsigned short val)
{
	return static_cast<unsigned short>(((val & 0xff00) >> 8) | ((val & 0x00ff) << 8));
}

B3_FORCE_INLINE unsigned b3SwapEndian(int val)
{
	return b3SwapEndian((unsigned)val);
}

B3_FORCE_INLINE unsigned short b3SwapEndian(short val)
{
	return b3SwapEndian((unsigned short) val);
}

///b3SwapFloat uses using char pointers to swap the endianness
////b3SwapFloat/b3SwapDouble will NOT return a float, because the machine might 'correct' invalid floating point values
///Not all values of sign/exponent/mantissa are valid floating point numbers according to IEEE 754. 
///When a floating point unit is faced with an invalid value, it may actually change the value, or worse, throw an exception. 
///In most systems, running user mode code, you wouldn't get an exception, but instead the hardware/os/runtime will 'fix' the number for you. 
///so instead of returning a float/double, we return integer/long long integer
B3_FORCE_INLINE unsigned int  b3SwapEndianFloat(float d)
{
    unsigned int a = 0;
    unsigned char *dst = (unsigned char *)&a;
    unsigned char *src = (unsigned char *)&d;

    dst[0] = src[3];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[0];
    return a;
}

// unswap using char pointers
B3_FORCE_INLINE float b3UnswapEndianFloat(unsigned int a) 
{
    float d = 0.0f;
    unsigned char *src = (unsigned char *)&a;
    unsigned char *dst = (unsigned char *)&d;

    dst[0] = src[3];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[0];

    return d;
}


// swap using char pointers
B3_FORCE_INLINE void  b3SwapEndianDouble(double d, unsigned char* dst)
{
    unsigned char *src = (unsigned char *)&d;

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
B3_FORCE_INLINE double b3UnswapEndianDouble(const unsigned char *src) 
{
    double d = 0.0;
    unsigned char *dst = (unsigned char *)&d;

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

// returns normalized value in range [-B3_PI, B3_PI]
B3_FORCE_INLINE b3Scalar b3NormalizeAngle(b3Scalar angleInRadians) 
{
	angleInRadians = b3Fmod(angleInRadians, B3_2_PI);
	if(angleInRadians < -B3_PI)
	{
		return angleInRadians + B3_2_PI;
	}
	else if(angleInRadians > B3_PI)
	{
		return angleInRadians - B3_2_PI;
	}
	else
	{
		return angleInRadians;
	}
}

///rudimentary class to provide type info
struct b3TypedObject
{
	b3TypedObject(int objectType)
		:m_objectType(objectType)
	{
	}
	int	m_objectType;
	inline int getObjectType() const
	{
		return m_objectType;
	}
};


  
///align a pointer to the provided alignment, upwards
template <typename T>T* b3AlignPointer(T* unalignedPtr, size_t alignment)
{
		
	struct b3ConvertPointerSizeT
	{
		union 
		{
				T* ptr;
				size_t integer;
		};
	};
    b3ConvertPointerSizeT converter;
    
    
	const size_t bit_mask = ~(alignment - 1);
    converter.ptr = unalignedPtr;
	converter.integer += alignment-1;
	converter.integer &= bit_mask;
	return converter.ptr;
}

#endif //B3_SCALAR_H
