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

#include <math.h>
#include <stdlib.h>  //size_t for MSVC 6.0
#include <float.h>

/* SVN $Revision$ on $Date$ from http://bullet.googlecode.com*/
#define BT_BULLET_VERSION 287

inline int btGetVersion()
{
	return BT_BULLET_VERSION;
}


// The following macro "BT_NOT_EMPTY_FILE" can be put into a file
// in order suppress the MS Visual C++ Linker warning 4221
//
// warning LNK4221: no public symbols found; archive member will be inaccessible
//
// This warning occurs on PC and XBOX when a file compiles out completely
// has no externally visible symbols which may be dependant on configuration
// #defines and options.
//
// see more https://stackoverflow.com/questions/1822887/what-is-the-best-way-to-eliminate-ms-visual-c-linker-warning-warning-lnk422

#if defined (_MSC_VER)
	#define BT_NOT_EMPTY_FILE_CAT_II(p, res) res
	#define BT_NOT_EMPTY_FILE_CAT_I(a, b) BT_NOT_EMPTY_FILE_CAT_II(~, a ## b)
	#define BT_NOT_EMPTY_FILE_CAT(a, b) BT_NOT_EMPTY_FILE_CAT_I(a, b)
	#define BT_NOT_EMPTY_FILE namespace { char BT_NOT_EMPTY_FILE_CAT(NoEmptyFileDummy, __COUNTER__); }
#else
	#define BT_NOT_EMPTY_FILE 
#endif


// clang and most formatting tools don't support indentation of preprocessor guards, so turn it off
// clang-format off
#if defined(DEBUG) || defined (_DEBUG)
	#define BT_DEBUG
#endif

#ifdef _WIN32
	#if defined(__MINGW32__) || defined(__CYGWIN__) || (defined (_MSC_VER) && _MSC_VER < 1300)
		#define SIMD_FORCE_INLINE inline
		#define ATTRIBUTE_ALIGNED16(a) a
		#define ATTRIBUTE_ALIGNED64(a) a
		#define ATTRIBUTE_ALIGNED128(a) a
	#elif defined(_M_ARM)
		#define SIMD_FORCE_INLINE __forceinline
		#define ATTRIBUTE_ALIGNED16(a) __declspec() a
		#define ATTRIBUTE_ALIGNED64(a) __declspec() a
		#define ATTRIBUTE_ALIGNED128(a) __declspec () a
	#else//__MINGW32__
		//#define BT_HAS_ALIGNED_ALLOCATOR
		#pragma warning(disable : 4324) // disable padding warning
//			#pragma warning(disable:4530) // Disable the exception disable but used in MSCV Stl warning.
		#pragma warning(disable:4996) //Turn off warnings about deprecated C routines
//			#pragma warning(disable:4786) // Disable the "debug name too long" warning

		#define SIMD_FORCE_INLINE __forceinline
		#define ATTRIBUTE_ALIGNED16(a) __declspec(align(16)) a
		#define ATTRIBUTE_ALIGNED64(a) __declspec(align(64)) a
		#define ATTRIBUTE_ALIGNED128(a) __declspec (align(128)) a
		#ifdef _XBOX
			#define BT_USE_VMX128

			#include <ppcintrinsics.h>
 			#define BT_HAVE_NATIVE_FSEL
 			#define btFsel(a,b,c) __fsel((a),(b),(c))
		#else

#if defined (_M_ARM)
            //Do not turn SSE on for ARM (may want to turn on BT_USE_NEON however)
#elif (defined (_WIN32) && (_MSC_VER) && _MSC_VER >= 1400) && (!defined (BT_USE_DOUBLE_PRECISION))
			#if _MSC_VER>1400
				#define BT_USE_SIMD_VECTOR3
			#endif

			#define BT_USE_SSE
			#ifdef BT_USE_SSE

#if (_MSC_FULL_VER >= 170050727)//Visual Studio 2012 can compile SSE4/FMA3 (but SSE4/FMA3 is not enabled by default)
			#define BT_ALLOW_SSE4
#endif //(_MSC_FULL_VER >= 160040219)

			//BT_USE_SSE_IN_API is disabled under Windows by default, because 
			//it makes it harder to integrate Bullet into your application under Windows 
			//(structured embedding Bullet structs/classes need to be 16-byte aligned)
			//with relatively little performance gain
			//If you are not embedded Bullet data in your classes, or make sure that you align those classes on 16-byte boundaries
			//you can manually enable this line or set it in the build system for a bit of performance gain (a few percent, dependent on usage)
			//#define BT_USE_SSE_IN_API
			#endif //BT_USE_SSE
			#include <emmintrin.h>
#endif

		#endif//_XBOX

	#endif //__MINGW32__

	#ifdef BT_DEBUG
		#ifdef _MSC_VER
			#include <stdio.h>
			#define btAssert(x) { if(!(x)){printf("Assert "__FILE__ ":%u (%s)\n", __LINE__, #x);__debugbreak();	}}
		#else//_MSC_VER
			#include <assert.h>
			#define btAssert assert
		#endif//_MSC_VER
	#else
		#define btAssert(x)
	#endif
		//btFullAssert is optional, slows down a lot
		#define btFullAssert(x)

		#define btLikely(_c)  _c
		#define btUnlikely(_c) _c

#else//_WIN32
	
	#if defined	(__CELLOS_LV2__)
		#define SIMD_FORCE_INLINE inline __attribute__((always_inline))
		#define ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
		#define ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
		#define ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
		#ifndef assert
		#include <assert.h>
		#endif
		#ifdef BT_DEBUG
			#ifdef __SPU__
				#include <spu_printf.h>
				#define printf spu_printf
				#define btAssert(x) {if(!(x)){printf("Assert "__FILE__ ":%u ("#x")\n", __LINE__);spu_hcmpeq(0,0);}}
			#else
				#define btAssert assert
			#endif
	
		#else//BT_DEBUG
				#define btAssert(x)
		#endif//BT_DEBUG
		//btFullAssert is optional, slows down a lot
		#define btFullAssert(x)

		#define btLikely(_c)  _c
		#define btUnlikely(_c) _c

	#else//defined	(__CELLOS_LV2__)

		#ifdef USE_LIBSPE2

			#define SIMD_FORCE_INLINE __inline
			#define ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
			#define ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
			#define ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
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


			#define btLikely(_c)   __builtin_expect((_c), 1)
			#define btUnlikely(_c) __builtin_expect((_c), 0)
		

		#else//USE_LIBSPE2
	//non-windows systems

			#if (defined (__APPLE__) && (!defined (BT_USE_DOUBLE_PRECISION)))
				#if defined (__i386__) || defined (__x86_64__)
					#define BT_USE_SIMD_VECTOR3
					#define BT_USE_SSE
					//BT_USE_SSE_IN_API is enabled on Mac OSX by default, because memory is automatically aligned on 16-byte boundaries
					//if apps run into issues, we will disable the next line
					#define BT_USE_SSE_IN_API
					#ifdef BT_USE_SSE
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
					#endif //BT_USE_SSE
				#elif defined( __ARM_NEON__ )
					#ifdef __clang__
						#define BT_USE_NEON 1
						#define BT_USE_SIMD_VECTOR3
		
						#if defined BT_USE_NEON && defined (__clang__)
							#include <arm_neon.h>
						#endif//BT_USE_NEON
				   #endif //__clang__
				#endif//__arm__

				#define SIMD_FORCE_INLINE inline __attribute__ ((always_inline))
			///@todo: check out alignment methods for other platforms/compilers
				#define ATTRIBUTE_ALIGNED16(a) a __attribute__ ((aligned (16)))
				#define ATTRIBUTE_ALIGNED64(a) a __attribute__ ((aligned (64)))
				#define ATTRIBUTE_ALIGNED128(a) a __attribute__ ((aligned (128)))
				#ifndef assert
				#include <assert.h>
				#endif

				#if defined(DEBUG) || defined (_DEBUG)
				 #if defined (__i386__) || defined (__x86_64__)
				#include <stdio.h>
				 #define btAssert(x)\
				{\
				if(!(x))\
				{\
					printf("Assert %s in line %d, file %s\n",#x, __LINE__, __FILE__);\
					asm volatile ("int3");\
				}\
				}
				#else//defined (__i386__) || defined (__x86_64__)
					#define btAssert assert
				#endif//defined (__i386__) || defined (__x86_64__)
				#else//defined(DEBUG) || defined (_DEBUG)
					#define btAssert(x)
				#endif//defined(DEBUG) || defined (_DEBUG)

				//btFullAssert is optional, slows down a lot
				#define btFullAssert(x)
				#define btLikely(_c)  _c
				#define btUnlikely(_c) _c

			#else//__APPLE__

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

				#if defined(DEBUG) || defined (_DEBUG)
					#define btAssert assert
				#else
					#define btAssert(x)
				#endif

				//btFullAssert is optional, slows down a lot
				#define btFullAssert(x)
				#define btLikely(_c)  _c
				#define btUnlikely(_c) _c
			#endif //__APPLE__ 
		#endif // LIBSPE2
	#endif	//__CELLOS_LV2__
#endif//_WIN32


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

#ifdef BT_USE_SSE
	typedef __m128 btSimdFloat4;
#endif  //BT_USE_SSE

#if defined(BT_USE_SSE)
	//#if defined BT_USE_SSE_IN_API && defined (BT_USE_SSE)
	#ifdef _WIN32

		#ifndef BT_NAN
			static int btNanMask = 0x7F800001;
			#define BT_NAN (*(float *)&btNanMask)
		#endif

		#ifndef BT_INFINITY
			static int btInfinityMask = 0x7F800000;
			#define BT_INFINITY (*(float *)&btInfinityMask)
			inline int btGetInfinityMask()  //suppress stupid compiler warning
			{
				return btInfinityMask;
			}
		#endif



	//use this, in case there are clashes (such as xnamath.h)
	#ifndef BT_NO_SIMD_OPERATOR_OVERLOADS
	inline __m128 operator+(const __m128 A, const __m128 B)
	{
		return _mm_add_ps(A, B);
	}

	inline __m128 operator-(const __m128 A, const __m128 B)
	{
		return _mm_sub_ps(A, B);
	}

	inline __m128 operator*(const __m128 A, const __m128 B)
	{
		return _mm_mul_ps(A, B);
	}
	#endif  //BT_NO_SIMD_OPERATOR_OVERLOADS

	#define btCastfTo128i(a) (_mm_castps_si128(a))
	#define btCastfTo128d(a) (_mm_castps_pd(a))
	#define btCastiTo128f(a) (_mm_castsi128_ps(a))
	#define btCastdTo128f(a) (_mm_castpd_ps(a))
	#define btCastdTo128i(a) (_mm_castpd_si128(a))
	#define btAssign128(r0, r1, r2, r3) _mm_setr_ps(r0, r1, r2, r3)

	#else  //_WIN32

		#define btCastfTo128i(a) ((__m128i)(a))
		#define btCastfTo128d(a) ((__m128d)(a))
		#define btCastiTo128f(a) ((__m128)(a))
		#define btCastdTo128f(a) ((__m128)(a))
		#define btCastdTo128i(a) ((__m128i)(a))
		#define btAssign128(r0, r1, r2, r3) \
			(__m128) { r0, r1, r2, r3 }
		#define BT_INFINITY INFINITY
		#define BT_NAN NAN
	#endif  //_WIN32
#else//BT_USE_SSE

	#ifdef BT_USE_NEON
	#include <arm_neon.h>

	typedef float32x4_t btSimdFloat4;
	#define BT_INFINITY INFINITY
	#define BT_NAN NAN
	#define btAssign128(r0, r1, r2, r3) \
		(float32x4_t) { r0, r1, r2, r3 }
	#else  //BT_USE_NEON

	#ifndef BT_INFINITY
	struct btInfMaskConverter
	{
		union {
			float mask;
			int intmask;
		};
		btInfMaskConverter(int mask = 0x7F800000)
			: intmask(mask)
		{
		}
	};
	static btInfMaskConverter btInfinityMask = 0x7F800000;
	#define BT_INFINITY (btInfinityMask.mask)
	inline int btGetInfinityMask()  //suppress stupid compiler warning
	{
		return btInfinityMask.intmask;
	}
	#endif
	#endif  //BT_USE_NEON

#endif  //BT_USE_SSE

#ifdef BT_USE_NEON
	#include <arm_neon.h>

	typedef float32x4_t btSimdFloat4;
	#define BT_INFINITY INFINITY
	#define BT_NAN NAN
	#define btAssign128(r0, r1, r2, r3) \
		(float32x4_t) { r0, r1, r2, r3 }
#endif//BT_USE_NEON

#define BT_DECLARE_ALIGNED_ALLOCATOR()                                                                     \
	SIMD_FORCE_INLINE void *operator new(size_t sizeInBytes) { return btAlignedAlloc(sizeInBytes, 16); }   \
	SIMD_FORCE_INLINE void operator delete(void *ptr) { btAlignedFree(ptr); }                              \
	SIMD_FORCE_INLINE void *operator new(size_t, void *ptr) { return ptr; }                                \
	SIMD_FORCE_INLINE void operator delete(void *, void *) {}                                              \
	SIMD_FORCE_INLINE void *operator new[](size_t sizeInBytes) { return btAlignedAlloc(sizeInBytes, 16); } \
	SIMD_FORCE_INLINE void operator delete[](void *ptr) { btAlignedFree(ptr); }                            \
	SIMD_FORCE_INLINE void *operator new[](size_t, void *ptr) { return ptr; }                              \
	SIMD_FORCE_INLINE void operator delete[](void *, void *) {}

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
		if (x < btScalar(-1)) x = btScalar(-1);
		if (x > btScalar(1)) x = btScalar(1);
		return acos(x);
	}
	SIMD_FORCE_INLINE btScalar btAsin(btScalar x)
	{
		if (x < btScalar(-1)) x = btScalar(-1);
		if (x > btScalar(1)) x = btScalar(1);
		return asin(x);
	}
	SIMD_FORCE_INLINE btScalar btAtan(btScalar x) { return atan(x); }
	SIMD_FORCE_INLINE btScalar btAtan2(btScalar x, btScalar y) { return atan2(x, y); }
	SIMD_FORCE_INLINE btScalar btExp(btScalar x) { return exp(x); }
	SIMD_FORCE_INLINE btScalar btLog(btScalar x) { return log(x); }
	SIMD_FORCE_INLINE btScalar btPow(btScalar x, btScalar y) { return pow(x, y); }
	SIMD_FORCE_INLINE btScalar btFmod(btScalar x, btScalar y) { return fmod(x, y); }

#else//BT_USE_DOUBLE_PRECISION

	SIMD_FORCE_INLINE btScalar btSqrt(btScalar y)
	{
	#ifdef USE_APPROXIMATION
	#ifdef __LP64__
		float xhalf = 0.5f * y;
		int i = *(int *)&y;
		i = 0x5f375a86 - (i >> 1);
		y = *(float *)&i;
		y = y * (1.5f - xhalf * y * y);
		y = y * (1.5f - xhalf * y * y);
		y = y * (1.5f - xhalf * y * y);
		y = 1 / y;
		return y;
	#else
		double x, z, tempf;
		unsigned long *tfptr = ((unsigned long *)&tempf) + 1;
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
	#endif
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

#endif//BT_USE_DOUBLE_PRECISION

#define SIMD_PI btScalar(3.1415926535897932384626433832795029)
#define SIMD_2_PI (btScalar(2.0) * SIMD_PI)
#define SIMD_HALF_PI (SIMD_PI * btScalar(0.5))
#define SIMD_RADS_PER_DEG (SIMD_2_PI / btScalar(360.0))
#define SIMD_DEGS_PER_RAD (btScalar(360.0) / SIMD_2_PI)
#define SIMDSQRT12 btScalar(0.7071067811865475244008443621048490)
#define btRecipSqrt(x) ((btScalar)(btScalar(1.0) / btSqrt(btScalar(x)))) /* reciprocal square root */
#define btRecip(x) (btScalar(1.0) / btScalar(x))

#ifdef BT_USE_DOUBLE_PRECISION
	#define SIMD_EPSILON DBL_EPSILON
	#define SIMD_INFINITY DBL_MAX
	#define BT_ONE 1.0
	#define BT_ZERO 0.0
	#define BT_TWO 2.0
	#define BT_HALF 0.5
#else
	#define SIMD_EPSILON FLT_EPSILON
	#define SIMD_INFINITY FLT_MAX
	#define BT_ONE 1.0f
	#define BT_ZERO 0.0f
	#define BT_TWO 2.0f
	#define BT_HALF 0.5f
#endif

// clang-format on

SIMD_FORCE_INLINE btScalar btAtan2Fast(btScalar y, btScalar x)
{
	btScalar coeff_1 = SIMD_PI / 4.0f;
	btScalar coeff_2 = 3.0f * coeff_1;
	btScalar abs_y = btFabs(y);
	btScalar angle;
	if (x >= 0.0f)
	{
		btScalar r = (x - abs_y) / (x + abs_y);
		angle = coeff_1 - coeff_1 * r;
	}
	else
	{
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

SIMD_FORCE_INLINE int btIsNegative(btScalar x)
{
	return x < btScalar(0.0) ? 1 : 0;
}

SIMD_FORCE_INLINE btScalar btRadians(btScalar x) { return x * SIMD_RADS_PER_DEG; }
SIMD_FORCE_INLINE btScalar btDegrees(btScalar x) { return x * SIMD_DEGS_PER_RAD; }

#define BT_DECLARE_HANDLE(name) \
	typedef struct name##__     \
	{                           \
		int unused;             \
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
	const char *p = (const char *)&i;
	if (p[0] == 1)  // Lowest address contains the least significant byte
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
	unsigned testNz = (unsigned)(((int)condition | -(int)condition) >> 31);
	unsigned testEqz = ~testNz;
	return ((valueIfConditionNonZero & testNz) | (valueIfConditionZero & testEqz));
}
SIMD_FORCE_INLINE int btSelect(unsigned condition, int valueIfConditionNonZero, int valueIfConditionZero)
{
	unsigned testNz = (unsigned)(((int)condition | -(int)condition) >> 31);
	unsigned testEqz = ~testNz;
	return static_cast<int>((valueIfConditionNonZero & testNz) | (valueIfConditionZero & testEqz));
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
SIMD_FORCE_INLINE void btSwap(T &a, T &b)
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

SIMD_FORCE_INLINE unsigned btSwapEndian(int val)
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
SIMD_FORCE_INLINE unsigned int btSwapEndianFloat(float d)
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
SIMD_FORCE_INLINE float btUnswapEndianFloat(unsigned int a)
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
SIMD_FORCE_INLINE void btSwapEndianDouble(double d, unsigned char *dst)
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
SIMD_FORCE_INLINE double btUnswapEndianDouble(const unsigned char *src)
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

template <typename T>
SIMD_FORCE_INLINE void btSetZero(T *a, int n)
{
	T *acurr = a;
	size_t ncurr = n;
	while (ncurr > 0)
	{
		*(acurr++) = 0;
		--ncurr;
	}
}

SIMD_FORCE_INLINE btScalar btLargeDot(const btScalar *a, const btScalar *b, int n)
{
	btScalar p0, q0, m0, p1, q1, m1, sum;
	sum = 0;
	n -= 2;
	while (n >= 0)
	{
		p0 = a[0];
		q0 = b[0];
		m0 = p0 * q0;
		p1 = a[1];
		q1 = b[1];
		m1 = p1 * q1;
		sum += m0;
		sum += m1;
		a += 2;
		b += 2;
		n -= 2;
	}
	n += 2;
	while (n > 0)
	{
		sum += (*a) * (*b);
		a++;
		b++;
		n--;
	}
	return sum;
}

// returns normalized value in range [-SIMD_PI, SIMD_PI]
SIMD_FORCE_INLINE btScalar btNormalizeAngle(btScalar angleInRadians)
{
	angleInRadians = btFmod(angleInRadians, SIMD_2_PI);
	if (angleInRadians < -SIMD_PI)
	{
		return angleInRadians + SIMD_2_PI;
	}
	else if (angleInRadians > SIMD_PI)
	{
		return angleInRadians - SIMD_2_PI;
	}
	else
	{
		return angleInRadians;
	}
}

///rudimentary class to provide type info
struct btTypedObject
{
	btTypedObject(int objectType)
		: m_objectType(objectType)
	{
	}
	int m_objectType;
	inline int getObjectType() const
	{
		return m_objectType;
	}
};

///align a pointer to the provided alignment, upwards
template <typename T>
T *btAlignPointer(T *unalignedPtr, size_t alignment)
{
	struct btConvertPointerSizeT
	{
		union {
			T *ptr;
			size_t integer;
		};
	};
	btConvertPointerSizeT converter;

	const size_t bit_mask = ~(alignment - 1);
	converter.ptr = unalignedPtr;
	converter.integer += alignment - 1;
	converter.integer &= bit_mask;
	return converter.ptr;
}

#endif  //BT_SCALAR_H
