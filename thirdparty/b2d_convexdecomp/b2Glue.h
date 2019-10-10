/*
* Copyright (c) 2006-2009 Erin Catto http://www.gphysics.com
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B2GLUE_H
#define B2GLUE_H

#include "core/math/vector2.h"

#include <limits.h>

namespace b2ConvexDecomp {

typedef real_t float32;
typedef int32_t int32;

static inline float32 b2Sqrt(float32 val) { return Math::sqrt(val); }
#define	b2_maxFloat		FLT_MAX
#define	b2_epsilon		CMP_EPSILON
#define b2_pi			3.14159265359f
#define b2_maxPolygonVertices 16
#define b2Max MAX
#define b2Min MIN
#define b2Clamp CLAMP
#define b2Abs ABS
/// A small length used as a collision and constraint tolerance. Usually it is
/// chosen to be numerically significant, but visually insignificant.
#define b2_linearSlop                   0.005f

/// A small angle used as a collision and constraint tolerance. Usually it is
/// chosen to be numerically significant, but visually insignificant.
#define b2_angularSlop                  (2.0f / 180.0f * b2_pi)

/// A 2D column vector.
struct b2Vec2
{
	/// Default constructor does nothing (for performance).
	b2Vec2() {}

	/// Construct using coordinates.
	b2Vec2(float32 x, float32 y) : x(x), y(y) {}

	/// Set this vector to all zeros.
	void SetZero() { x = 0.0f; y = 0.0f; }

	/// Set this vector to some specified coordinates.
	void Set(float32 x_, float32 y_) { x = x_; y = y_; }

	/// Negate this vector.
	b2Vec2 operator -() const { b2Vec2 v; v.Set(-x, -y); return v; }

	/// Read from and indexed element.
	float32 operator () (int32 i) const
	{
		return (&x)[i];
	}

	/// Write to an indexed element.
	float32& operator () (int32 i)
	{
		return (&x)[i];
	}

	/// Add a vector to this vector.
	void operator += (const b2Vec2& v)
	{
		x += v.x; y += v.y;
	}

	/// Subtract a vector from this vector.
	void operator -= (const b2Vec2& v)
	{
		x -= v.x; y -= v.y;
	}

	/// Multiply this vector by a scalar.
	void operator *= (float32 a)
	{
		x *= a; y *= a;
	}

	/// Get the length of this vector (the norm).
	float32 Length() const
	{
		return b2Sqrt(x * x + y * y);
	}

	/// Get the length squared. For performance, use this instead of
	/// b2Vec2::Length (if possible).
	float32 LengthSquared() const
	{
		return x * x + y * y;
	}

	bool operator==(const b2Vec2& p_v) const {
		return x==p_v.x && y==p_v.y;
	}
	b2Vec2 operator+(const b2Vec2& p_v) const {
		return b2Vec2(x+p_v.x,y+p_v.y);
	}
	b2Vec2 operator-(const b2Vec2& p_v) const {
		return b2Vec2(x-p_v.x,y-p_v.y);
	}

	b2Vec2 operator*(float32 f) const {
		return b2Vec2(f*x,f*y);
	}

	/// Convert this vector into a unit vector. Returns the length.
	float32 Normalize()
	{
		float32 length = Length();
		if (length < b2_epsilon)
		{
			return 0.0f;
		}
		float32 invLength = 1.0f / length;
		x *= invLength;
		y *= invLength;

		return length;
	}

	/*
	/// Does this vector contain finite coordinates?
	bool IsValid() const
	{
		return b2IsValid(x) && b2IsValid(y);
	}
	*/

	float32 x, y;
};

inline b2Vec2 operator*(float32 f,const b2Vec2& p_v)  {
	return b2Vec2(f*p_v.x,f*p_v.y);
}

/// Perform the dot product on two vectors.
inline float32 b2Dot(const b2Vec2& a, const b2Vec2& b)
{
	return a.x * b.x + a.y * b.y;
}

/// Perform the cross product on two vectors. In 2D this produces a scalar.
inline float32 b2Cross(const b2Vec2& a, const b2Vec2& b)
{
	return a.x * b.y - a.y * b.x;
}

/// Perform the cross product on a vector and a scalar. In 2D this produces
/// a vector.
inline b2Vec2 b2Cross(const b2Vec2& a, float32 s)
{
	return b2Vec2(s * a.y, -s * a.x);
}

}

#endif
