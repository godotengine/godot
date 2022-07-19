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

#ifndef B3_SIMD_QUADWORD_H
#define B3_SIMD_QUADWORD_H

#include "b3Scalar.h"
#include "b3MinMax.h"

#if defined(__CELLOS_LV2) && defined(__SPU__)
#include <altivec.h>
#endif

/**@brief The b3QuadWord class is base class for b3Vector3 and b3Quaternion. 
 * Some issues under PS3 Linux with IBM 2.1 SDK, gcc compiler prevent from using aligned quadword.
 */
#ifndef USE_LIBSPE2
B3_ATTRIBUTE_ALIGNED16(class)
b3QuadWord
#else
class b3QuadWord
#endif
{
protected:
#if defined(__SPU__) && defined(__CELLOS_LV2__)
	union {
		vec_float4 mVec128;
		b3Scalar m_floats[4];
	};

public:
	vec_float4 get128() const
	{
		return mVec128;
	}

#else  //__CELLOS_LV2__ __SPU__

#if defined(B3_USE_SSE) || defined(B3_USE_NEON)
public:
	union {
		b3SimdFloat4 mVec128;
		b3Scalar m_floats[4];
		struct
		{
			b3Scalar x, y, z, w;
		};
	};

public:
	B3_FORCE_INLINE b3SimdFloat4 get128() const
	{
		return mVec128;
	}
	B3_FORCE_INLINE void set128(b3SimdFloat4 v128)
	{
		mVec128 = v128;
	}
#else
public:
	union {
		b3Scalar m_floats[4];
		struct
		{
			b3Scalar x, y, z, w;
		};
	};
#endif  // B3_USE_SSE

#endif  //__CELLOS_LV2__ __SPU__

public:
#if defined(B3_USE_SSE) || defined(B3_USE_NEON)

	// Set Vector
	B3_FORCE_INLINE b3QuadWord(const b3SimdFloat4 vec)
	{
		mVec128 = vec;
	}

	// Copy constructor
	B3_FORCE_INLINE b3QuadWord(const b3QuadWord& rhs)
	{
		mVec128 = rhs.mVec128;
	}

	// Assignment Operator
	B3_FORCE_INLINE b3QuadWord&
	operator=(const b3QuadWord& v)
	{
		mVec128 = v.mVec128;

		return *this;
	}

#endif

	/**@brief Return the x value */
	B3_FORCE_INLINE const b3Scalar& getX() const { return m_floats[0]; }
	/**@brief Return the y value */
	B3_FORCE_INLINE const b3Scalar& getY() const { return m_floats[1]; }
	/**@brief Return the z value */
	B3_FORCE_INLINE const b3Scalar& getZ() const { return m_floats[2]; }
	/**@brief Set the x value */
	B3_FORCE_INLINE void setX(b3Scalar _x) { m_floats[0] = _x; };
	/**@brief Set the y value */
	B3_FORCE_INLINE void setY(b3Scalar _y) { m_floats[1] = _y; };
	/**@brief Set the z value */
	B3_FORCE_INLINE void setZ(b3Scalar _z) { m_floats[2] = _z; };
	/**@brief Set the w value */
	B3_FORCE_INLINE void setW(b3Scalar _w) { m_floats[3] = _w; };
	/**@brief Return the x value */

	//B3_FORCE_INLINE b3Scalar&       operator[](int i)       { return (&m_floats[0])[i];	}
	//B3_FORCE_INLINE const b3Scalar& operator[](int i) const { return (&m_floats[0])[i]; }
	///operator b3Scalar*() replaces operator[], using implicit conversion. We added operator != and operator == to avoid pointer comparisons.
	B3_FORCE_INLINE operator b3Scalar*() { return &m_floats[0]; }
	B3_FORCE_INLINE operator const b3Scalar*() const { return &m_floats[0]; }

	B3_FORCE_INLINE bool operator==(const b3QuadWord& other) const
	{
#ifdef B3_USE_SSE
		return (0xf == _mm_movemask_ps((__m128)_mm_cmpeq_ps(mVec128, other.mVec128)));
#else
		return ((m_floats[3] == other.m_floats[3]) &&
				(m_floats[2] == other.m_floats[2]) &&
				(m_floats[1] == other.m_floats[1]) &&
				(m_floats[0] == other.m_floats[0]));
#endif
	}

	B3_FORCE_INLINE bool operator!=(const b3QuadWord& other) const
	{
		return !(*this == other);
	}

	/**@brief Set x,y,z and zero w 
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   */
	B3_FORCE_INLINE void setValue(const b3Scalar& _x, const b3Scalar& _y, const b3Scalar& _z)
	{
		m_floats[0] = _x;
		m_floats[1] = _y;
		m_floats[2] = _z;
		m_floats[3] = 0.f;
	}

	/*		void getValue(b3Scalar *m) const 
		{
			m[0] = m_floats[0];
			m[1] = m_floats[1];
			m[2] = m_floats[2];
		}
*/
	/**@brief Set the values 
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   * @param w Value of w
   */
	B3_FORCE_INLINE void setValue(const b3Scalar& _x, const b3Scalar& _y, const b3Scalar& _z, const b3Scalar& _w)
	{
		m_floats[0] = _x;
		m_floats[1] = _y;
		m_floats[2] = _z;
		m_floats[3] = _w;
	}
	/**@brief No initialization constructor */
	B3_FORCE_INLINE b3QuadWord()
	//	:m_floats[0](b3Scalar(0.)),m_floats[1](b3Scalar(0.)),m_floats[2](b3Scalar(0.)),m_floats[3](b3Scalar(0.))
	{
	}

	/**@brief Three argument constructor (zeros w)
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   */
	B3_FORCE_INLINE b3QuadWord(const b3Scalar& _x, const b3Scalar& _y, const b3Scalar& _z)
	{
		m_floats[0] = _x, m_floats[1] = _y, m_floats[2] = _z, m_floats[3] = 0.0f;
	}

	/**@brief Initializing constructor
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   * @param w Value of w
   */
	B3_FORCE_INLINE b3QuadWord(const b3Scalar& _x, const b3Scalar& _y, const b3Scalar& _z, const b3Scalar& _w)
	{
		m_floats[0] = _x, m_floats[1] = _y, m_floats[2] = _z, m_floats[3] = _w;
	}

	/**@brief Set each element to the max of the current values and the values of another b3QuadWord
   * @param other The other b3QuadWord to compare with 
   */
	B3_FORCE_INLINE void setMax(const b3QuadWord& other)
	{
#ifdef B3_USE_SSE
		mVec128 = _mm_max_ps(mVec128, other.mVec128);
#elif defined(B3_USE_NEON)
		mVec128 = vmaxq_f32(mVec128, other.mVec128);
#else
		b3SetMax(m_floats[0], other.m_floats[0]);
		b3SetMax(m_floats[1], other.m_floats[1]);
		b3SetMax(m_floats[2], other.m_floats[2]);
		b3SetMax(m_floats[3], other.m_floats[3]);
#endif
	}
	/**@brief Set each element to the min of the current values and the values of another b3QuadWord
   * @param other The other b3QuadWord to compare with 
   */
	B3_FORCE_INLINE void setMin(const b3QuadWord& other)
	{
#ifdef B3_USE_SSE
		mVec128 = _mm_min_ps(mVec128, other.mVec128);
#elif defined(B3_USE_NEON)
		mVec128 = vminq_f32(mVec128, other.mVec128);
#else
		b3SetMin(m_floats[0], other.m_floats[0]);
		b3SetMin(m_floats[1], other.m_floats[1]);
		b3SetMin(m_floats[2], other.m_floats[2]);
		b3SetMin(m_floats[3], other.m_floats[3]);
#endif
	}
};

#endif  //B3_SIMD_QUADWORD_H
