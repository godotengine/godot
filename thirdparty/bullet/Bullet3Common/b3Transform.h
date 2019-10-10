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

#ifndef B3_TRANSFORM_H
#define B3_TRANSFORM_H

#include "b3Matrix3x3.h"

#ifdef B3_USE_DOUBLE_PRECISION
#define b3TransformData b3TransformDoubleData
#else
#define b3TransformData b3TransformFloatData
#endif

/**@brief The b3Transform class supports rigid transforms with only translation and rotation and no scaling/shear.
 *It can be used in combination with b3Vector3, b3Quaternion and b3Matrix3x3 linear algebra classes. */
B3_ATTRIBUTE_ALIGNED16(class)
b3Transform
{
	///Storage for the rotation
	b3Matrix3x3 m_basis;
	///Storage for the translation
	b3Vector3 m_origin;

public:
	/**@brief No initialization constructor */
	b3Transform() {}
	/**@brief Constructor from b3Quaternion (optional b3Vector3 )
   * @param q Rotation from quaternion 
   * @param c Translation from Vector (default 0,0,0) */
	explicit B3_FORCE_INLINE b3Transform(const b3Quaternion& q,
										 const b3Vector3& c = b3MakeVector3(b3Scalar(0), b3Scalar(0), b3Scalar(0)))
		: m_basis(q),
		  m_origin(c)
	{
	}

	/**@brief Constructor from b3Matrix3x3 (optional b3Vector3)
   * @param b Rotation from Matrix 
   * @param c Translation from Vector default (0,0,0)*/
	explicit B3_FORCE_INLINE b3Transform(const b3Matrix3x3& b,
										 const b3Vector3& c = b3MakeVector3(b3Scalar(0), b3Scalar(0), b3Scalar(0)))
		: m_basis(b),
		  m_origin(c)
	{
	}
	/**@brief Copy constructor */
	B3_FORCE_INLINE b3Transform(const b3Transform& other)
		: m_basis(other.m_basis),
		  m_origin(other.m_origin)
	{
	}
	/**@brief Assignment Operator */
	B3_FORCE_INLINE b3Transform& operator=(const b3Transform& other)
	{
		m_basis = other.m_basis;
		m_origin = other.m_origin;
		return *this;
	}

	/**@brief Set the current transform as the value of the product of two transforms
   * @param t1 Transform 1
   * @param t2 Transform 2
   * This = Transform1 * Transform2 */
	B3_FORCE_INLINE void mult(const b3Transform& t1, const b3Transform& t2)
	{
		m_basis = t1.m_basis * t2.m_basis;
		m_origin = t1(t2.m_origin);
	}

	/*		void multInverseLeft(const b3Transform& t1, const b3Transform& t2) {
			b3Vector3 v = t2.m_origin - t1.m_origin;
			m_basis = b3MultTransposeLeft(t1.m_basis, t2.m_basis);
			m_origin = v * t1.m_basis;
		}
		*/

	/**@brief Return the transform of the vector */
	B3_FORCE_INLINE b3Vector3 operator()(const b3Vector3& x) const
	{
		return x.dot3(m_basis[0], m_basis[1], m_basis[2]) + m_origin;
	}

	/**@brief Return the transform of the vector */
	B3_FORCE_INLINE b3Vector3 operator*(const b3Vector3& x) const
	{
		return (*this)(x);
	}

	/**@brief Return the transform of the b3Quaternion */
	B3_FORCE_INLINE b3Quaternion operator*(const b3Quaternion& q) const
	{
		return getRotation() * q;
	}

	/**@brief Return the basis matrix for the rotation */
	B3_FORCE_INLINE b3Matrix3x3& getBasis() { return m_basis; }
	/**@brief Return the basis matrix for the rotation */
	B3_FORCE_INLINE const b3Matrix3x3& getBasis() const { return m_basis; }

	/**@brief Return the origin vector translation */
	B3_FORCE_INLINE b3Vector3& getOrigin() { return m_origin; }
	/**@brief Return the origin vector translation */
	B3_FORCE_INLINE const b3Vector3& getOrigin() const { return m_origin; }

	/**@brief Return a quaternion representing the rotation */
	b3Quaternion getRotation() const
	{
		b3Quaternion q;
		m_basis.getRotation(q);
		return q;
	}

	/**@brief Set from an array 
   * @param m A pointer to a 15 element array (12 rotation(row major padded on the right by 1), and 3 translation */
	void setFromOpenGLMatrix(const b3Scalar* m)
	{
		m_basis.setFromOpenGLSubMatrix(m);
		m_origin.setValue(m[12], m[13], m[14]);
	}

	/**@brief Fill an array representation
   * @param m A pointer to a 15 element array (12 rotation(row major padded on the right by 1), and 3 translation */
	void getOpenGLMatrix(b3Scalar * m) const
	{
		m_basis.getOpenGLSubMatrix(m);
		m[12] = m_origin.getX();
		m[13] = m_origin.getY();
		m[14] = m_origin.getZ();
		m[15] = b3Scalar(1.0);
	}

	/**@brief Set the translational element
   * @param origin The vector to set the translation to */
	B3_FORCE_INLINE void setOrigin(const b3Vector3& origin)
	{
		m_origin = origin;
	}

	B3_FORCE_INLINE b3Vector3 invXform(const b3Vector3& inVec) const;

	/**@brief Set the rotational element by b3Matrix3x3 */
	B3_FORCE_INLINE void setBasis(const b3Matrix3x3& basis)
	{
		m_basis = basis;
	}

	/**@brief Set the rotational element by b3Quaternion */
	B3_FORCE_INLINE void setRotation(const b3Quaternion& q)
	{
		m_basis.setRotation(q);
	}

	/**@brief Set this transformation to the identity */
	void setIdentity()
	{
		m_basis.setIdentity();
		m_origin.setValue(b3Scalar(0.0), b3Scalar(0.0), b3Scalar(0.0));
	}

	/**@brief Multiply this Transform by another(this = this * another) 
   * @param t The other transform */
	b3Transform& operator*=(const b3Transform& t)
	{
		m_origin += m_basis * t.m_origin;
		m_basis *= t.m_basis;
		return *this;
	}

	/**@brief Return the inverse of this transform */
	b3Transform inverse() const
	{
		b3Matrix3x3 inv = m_basis.transpose();
		return b3Transform(inv, inv * -m_origin);
	}

	/**@brief Return the inverse of this transform times the other transform
   * @param t The other transform 
   * return this.inverse() * the other */
	b3Transform inverseTimes(const b3Transform& t) const;

	/**@brief Return the product of this transform and the other */
	b3Transform operator*(const b3Transform& t) const;

	/**@brief Return an identity transform */
	static const b3Transform& getIdentity()
	{
		static const b3Transform identityTransform(b3Matrix3x3::getIdentity());
		return identityTransform;
	}

	void serialize(struct b3TransformData & dataOut) const;

	void serializeFloat(struct b3TransformFloatData & dataOut) const;

	void deSerialize(const struct b3TransformData& dataIn);

	void deSerializeDouble(const struct b3TransformDoubleData& dataIn);

	void deSerializeFloat(const struct b3TransformFloatData& dataIn);
};

B3_FORCE_INLINE b3Vector3
b3Transform::invXform(const b3Vector3& inVec) const
{
	b3Vector3 v = inVec - m_origin;
	return (m_basis.transpose() * v);
}

B3_FORCE_INLINE b3Transform
b3Transform::inverseTimes(const b3Transform& t) const
{
	b3Vector3 v = t.getOrigin() - m_origin;
	return b3Transform(m_basis.transposeTimes(t.m_basis),
					   v * m_basis);
}

B3_FORCE_INLINE b3Transform
	b3Transform::operator*(const b3Transform& t) const
{
	return b3Transform(m_basis * t.m_basis,
					   (*this)(t.m_origin));
}

/**@brief Test if two transforms have all elements equal */
B3_FORCE_INLINE bool operator==(const b3Transform& t1, const b3Transform& t2)
{
	return (t1.getBasis() == t2.getBasis() &&
			t1.getOrigin() == t2.getOrigin());
}

///for serialization
struct b3TransformFloatData
{
	b3Matrix3x3FloatData m_basis;
	b3Vector3FloatData m_origin;
};

struct b3TransformDoubleData
{
	b3Matrix3x3DoubleData m_basis;
	b3Vector3DoubleData m_origin;
};

B3_FORCE_INLINE void b3Transform::serialize(b3TransformData& dataOut) const
{
	m_basis.serialize(dataOut.m_basis);
	m_origin.serialize(dataOut.m_origin);
}

B3_FORCE_INLINE void b3Transform::serializeFloat(b3TransformFloatData& dataOut) const
{
	m_basis.serializeFloat(dataOut.m_basis);
	m_origin.serializeFloat(dataOut.m_origin);
}

B3_FORCE_INLINE void b3Transform::deSerialize(const b3TransformData& dataIn)
{
	m_basis.deSerialize(dataIn.m_basis);
	m_origin.deSerialize(dataIn.m_origin);
}

B3_FORCE_INLINE void b3Transform::deSerializeFloat(const b3TransformFloatData& dataIn)
{
	m_basis.deSerializeFloat(dataIn.m_basis);
	m_origin.deSerializeFloat(dataIn.m_origin);
}

B3_FORCE_INLINE void b3Transform::deSerializeDouble(const b3TransformDoubleData& dataIn)
{
	m_basis.deSerializeDouble(dataIn.m_basis);
	m_origin.deSerializeDouble(dataIn.m_origin);
}

#endif  //B3_TRANSFORM_H
