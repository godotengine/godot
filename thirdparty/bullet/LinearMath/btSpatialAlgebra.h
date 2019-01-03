/*
Copyright (c) 2003-2015 Erwin Coumans, Jakub Stepien

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

///These spatial algebra classes are used for btMultiBody,
///see BulletDynamics/Featherstone

#ifndef BT_SPATIAL_ALGEBRA_H
#define BT_SPATIAL_ALGEBRA_H

#include "btMatrix3x3.h"

struct btSpatialForceVector
{
	btVector3 m_topVec, m_bottomVec;
	//
	btSpatialForceVector() { setZero(); }
	btSpatialForceVector(const btVector3 &angular, const btVector3 &linear) : m_topVec(linear), m_bottomVec(angular) {}
	btSpatialForceVector(const btScalar &ax, const btScalar &ay, const btScalar &az, const btScalar &lx, const btScalar &ly, const btScalar &lz)
	{
		setValue(ax, ay, az, lx, ly, lz);
	}
	//
	void setVector(const btVector3 &angular, const btVector3 &linear)
	{
		m_topVec = linear;
		m_bottomVec = angular;
	}
	void setValue(const btScalar &ax, const btScalar &ay, const btScalar &az, const btScalar &lx, const btScalar &ly, const btScalar &lz)
	{
		m_bottomVec.setValue(ax, ay, az);
		m_topVec.setValue(lx, ly, lz);
	}
	//
	void addVector(const btVector3 &angular, const btVector3 &linear)
	{
		m_topVec += linear;
		m_bottomVec += angular;
	}
	void addValue(const btScalar &ax, const btScalar &ay, const btScalar &az, const btScalar &lx, const btScalar &ly, const btScalar &lz)
	{
		m_bottomVec[0] += ax;
		m_bottomVec[1] += ay;
		m_bottomVec[2] += az;
		m_topVec[0] += lx;
		m_topVec[1] += ly;
		m_topVec[2] += lz;
	}
	//
	const btVector3 &getLinear() const { return m_topVec; }
	const btVector3 &getAngular() const { return m_bottomVec; }
	//
	void setLinear(const btVector3 &linear) { m_topVec = linear; }
	void setAngular(const btVector3 &angular) { m_bottomVec = angular; }
	//
	void addAngular(const btVector3 &angular) { m_bottomVec += angular; }
	void addLinear(const btVector3 &linear) { m_topVec += linear; }
	//
	void setZero()
	{
		m_topVec.setZero();
		m_bottomVec.setZero();
	}
	//
	btSpatialForceVector &operator+=(const btSpatialForceVector &vec)
	{
		m_topVec += vec.m_topVec;
		m_bottomVec += vec.m_bottomVec;
		return *this;
	}
	btSpatialForceVector &operator-=(const btSpatialForceVector &vec)
	{
		m_topVec -= vec.m_topVec;
		m_bottomVec -= vec.m_bottomVec;
		return *this;
	}
	btSpatialForceVector operator-(const btSpatialForceVector &vec) const { return btSpatialForceVector(m_bottomVec - vec.m_bottomVec, m_topVec - vec.m_topVec); }
	btSpatialForceVector operator+(const btSpatialForceVector &vec) const { return btSpatialForceVector(m_bottomVec + vec.m_bottomVec, m_topVec + vec.m_topVec); }
	btSpatialForceVector operator-() const { return btSpatialForceVector(-m_bottomVec, -m_topVec); }
	btSpatialForceVector operator*(const btScalar &s) const { return btSpatialForceVector(s * m_bottomVec, s * m_topVec); }
	//btSpatialForceVector & operator = (const btSpatialForceVector &vec) { m_topVec = vec.m_topVec; m_bottomVec = vec.m_bottomVec; return *this; }
};

struct btSpatialMotionVector
{
	btVector3 m_topVec, m_bottomVec;
	//
	btSpatialMotionVector() { setZero(); }
	btSpatialMotionVector(const btVector3 &angular, const btVector3 &linear) : m_topVec(angular), m_bottomVec(linear) {}
	//
	void setVector(const btVector3 &angular, const btVector3 &linear)
	{
		m_topVec = angular;
		m_bottomVec = linear;
	}
	void setValue(const btScalar &ax, const btScalar &ay, const btScalar &az, const btScalar &lx, const btScalar &ly, const btScalar &lz)
	{
		m_topVec.setValue(ax, ay, az);
		m_bottomVec.setValue(lx, ly, lz);
	}
	//
	void addVector(const btVector3 &angular, const btVector3 &linear)
	{
		m_topVec += linear;
		m_bottomVec += angular;
	}
	void addValue(const btScalar &ax, const btScalar &ay, const btScalar &az, const btScalar &lx, const btScalar &ly, const btScalar &lz)
	{
		m_topVec[0] += ax;
		m_topVec[1] += ay;
		m_topVec[2] += az;
		m_bottomVec[0] += lx;
		m_bottomVec[1] += ly;
		m_bottomVec[2] += lz;
	}
	//
	const btVector3 &getAngular() const { return m_topVec; }
	const btVector3 &getLinear() const { return m_bottomVec; }
	//
	void setAngular(const btVector3 &angular) { m_topVec = angular; }
	void setLinear(const btVector3 &linear) { m_bottomVec = linear; }
	//
	void addAngular(const btVector3 &angular) { m_topVec += angular; }
	void addLinear(const btVector3 &linear) { m_bottomVec += linear; }
	//
	void setZero()
	{
		m_topVec.setZero();
		m_bottomVec.setZero();
	}
	//
	btScalar dot(const btSpatialForceVector &b) const
	{
		return m_bottomVec.dot(b.m_topVec) + m_topVec.dot(b.m_bottomVec);
	}
	//
	template <typename SpatialVectorType>
	void cross(const SpatialVectorType &b, SpatialVectorType &out) const
	{
		out.m_topVec = m_topVec.cross(b.m_topVec);
		out.m_bottomVec = m_bottomVec.cross(b.m_topVec) + m_topVec.cross(b.m_bottomVec);
	}
	template <typename SpatialVectorType>
	SpatialVectorType cross(const SpatialVectorType &b) const
	{
		SpatialVectorType out;
		out.m_topVec = m_topVec.cross(b.m_topVec);
		out.m_bottomVec = m_bottomVec.cross(b.m_topVec) + m_topVec.cross(b.m_bottomVec);
		return out;
	}
	//
	btSpatialMotionVector &operator+=(const btSpatialMotionVector &vec)
	{
		m_topVec += vec.m_topVec;
		m_bottomVec += vec.m_bottomVec;
		return *this;
	}
	btSpatialMotionVector &operator-=(const btSpatialMotionVector &vec)
	{
		m_topVec -= vec.m_topVec;
		m_bottomVec -= vec.m_bottomVec;
		return *this;
	}
	btSpatialMotionVector &operator*=(const btScalar &s)
	{
		m_topVec *= s;
		m_bottomVec *= s;
		return *this;
	}
	btSpatialMotionVector operator-(const btSpatialMotionVector &vec) const { return btSpatialMotionVector(m_topVec - vec.m_topVec, m_bottomVec - vec.m_bottomVec); }
	btSpatialMotionVector operator+(const btSpatialMotionVector &vec) const { return btSpatialMotionVector(m_topVec + vec.m_topVec, m_bottomVec + vec.m_bottomVec); }
	btSpatialMotionVector operator-() const { return btSpatialMotionVector(-m_topVec, -m_bottomVec); }
	btSpatialMotionVector operator*(const btScalar &s) const { return btSpatialMotionVector(s * m_topVec, s * m_bottomVec); }
};

struct btSymmetricSpatialDyad
{
	btMatrix3x3 m_topLeftMat, m_topRightMat, m_bottomLeftMat;
	//
	btSymmetricSpatialDyad() { setIdentity(); }
	btSymmetricSpatialDyad(const btMatrix3x3 &topLeftMat, const btMatrix3x3 &topRightMat, const btMatrix3x3 &bottomLeftMat) { setMatrix(topLeftMat, topRightMat, bottomLeftMat); }
	//
	void setMatrix(const btMatrix3x3 &topLeftMat, const btMatrix3x3 &topRightMat, const btMatrix3x3 &bottomLeftMat)
	{
		m_topLeftMat = topLeftMat;
		m_topRightMat = topRightMat;
		m_bottomLeftMat = bottomLeftMat;
	}
	//
	void addMatrix(const btMatrix3x3 &topLeftMat, const btMatrix3x3 &topRightMat, const btMatrix3x3 &bottomLeftMat)
	{
		m_topLeftMat += topLeftMat;
		m_topRightMat += topRightMat;
		m_bottomLeftMat += bottomLeftMat;
	}
	//
	void setIdentity()
	{
		m_topLeftMat.setIdentity();
		m_topRightMat.setIdentity();
		m_bottomLeftMat.setIdentity();
	}
	//
	btSymmetricSpatialDyad &operator-=(const btSymmetricSpatialDyad &mat)
	{
		m_topLeftMat -= mat.m_topLeftMat;
		m_topRightMat -= mat.m_topRightMat;
		m_bottomLeftMat -= mat.m_bottomLeftMat;
		return *this;
	}
	//
	btSpatialForceVector operator*(const btSpatialMotionVector &vec)
	{
		return btSpatialForceVector(m_bottomLeftMat * vec.m_topVec + m_topLeftMat.transpose() * vec.m_bottomVec, m_topLeftMat * vec.m_topVec + m_topRightMat * vec.m_bottomVec);
	}
};

struct btSpatialTransformationMatrix
{
	btMatrix3x3 m_rotMat;  //btMatrix3x3 m_trnCrossMat;
	btVector3 m_trnVec;
	//
	enum eOutputOperation
	{
		None = 0,
		Add = 1,
		Subtract = 2
	};
	//
	template <typename SpatialVectorType>
	void transform(const SpatialVectorType &inVec,
				   SpatialVectorType &outVec,
				   eOutputOperation outOp = None)
	{
		if (outOp == None)
		{
			outVec.m_topVec = m_rotMat * inVec.m_topVec;
			outVec.m_bottomVec = -m_trnVec.cross(outVec.m_topVec) + m_rotMat * inVec.m_bottomVec;
		}
		else if (outOp == Add)
		{
			outVec.m_topVec += m_rotMat * inVec.m_topVec;
			outVec.m_bottomVec += -m_trnVec.cross(outVec.m_topVec) + m_rotMat * inVec.m_bottomVec;
		}
		else if (outOp == Subtract)
		{
			outVec.m_topVec -= m_rotMat * inVec.m_topVec;
			outVec.m_bottomVec -= -m_trnVec.cross(outVec.m_topVec) + m_rotMat * inVec.m_bottomVec;
		}
	}

	template <typename SpatialVectorType>
	void transformRotationOnly(const SpatialVectorType &inVec,
							   SpatialVectorType &outVec,
							   eOutputOperation outOp = None)
	{
		if (outOp == None)
		{
			outVec.m_topVec = m_rotMat * inVec.m_topVec;
			outVec.m_bottomVec = m_rotMat * inVec.m_bottomVec;
		}
		else if (outOp == Add)
		{
			outVec.m_topVec += m_rotMat * inVec.m_topVec;
			outVec.m_bottomVec += m_rotMat * inVec.m_bottomVec;
		}
		else if (outOp == Subtract)
		{
			outVec.m_topVec -= m_rotMat * inVec.m_topVec;
			outVec.m_bottomVec -= m_rotMat * inVec.m_bottomVec;
		}
	}

	template <typename SpatialVectorType>
	void transformInverse(const SpatialVectorType &inVec,
						  SpatialVectorType &outVec,
						  eOutputOperation outOp = None)
	{
		if (outOp == None)
		{
			outVec.m_topVec = m_rotMat.transpose() * inVec.m_topVec;
			outVec.m_bottomVec = m_rotMat.transpose() * (inVec.m_bottomVec + m_trnVec.cross(inVec.m_topVec));
		}
		else if (outOp == Add)
		{
			outVec.m_topVec += m_rotMat.transpose() * inVec.m_topVec;
			outVec.m_bottomVec += m_rotMat.transpose() * (inVec.m_bottomVec + m_trnVec.cross(inVec.m_topVec));
		}
		else if (outOp == Subtract)
		{
			outVec.m_topVec -= m_rotMat.transpose() * inVec.m_topVec;
			outVec.m_bottomVec -= m_rotMat.transpose() * (inVec.m_bottomVec + m_trnVec.cross(inVec.m_topVec));
		}
	}

	template <typename SpatialVectorType>
	void transformInverseRotationOnly(const SpatialVectorType &inVec,
									  SpatialVectorType &outVec,
									  eOutputOperation outOp = None)
	{
		if (outOp == None)
		{
			outVec.m_topVec = m_rotMat.transpose() * inVec.m_topVec;
			outVec.m_bottomVec = m_rotMat.transpose() * inVec.m_bottomVec;
		}
		else if (outOp == Add)
		{
			outVec.m_topVec += m_rotMat.transpose() * inVec.m_topVec;
			outVec.m_bottomVec += m_rotMat.transpose() * inVec.m_bottomVec;
		}
		else if (outOp == Subtract)
		{
			outVec.m_topVec -= m_rotMat.transpose() * inVec.m_topVec;
			outVec.m_bottomVec -= m_rotMat.transpose() * inVec.m_bottomVec;
		}
	}

	void transformInverse(const btSymmetricSpatialDyad &inMat,
						  btSymmetricSpatialDyad &outMat,
						  eOutputOperation outOp = None)
	{
		const btMatrix3x3 r_cross(0, -m_trnVec[2], m_trnVec[1],
								  m_trnVec[2], 0, -m_trnVec[0],
								  -m_trnVec[1], m_trnVec[0], 0);

		if (outOp == None)
		{
			outMat.m_topLeftMat = m_rotMat.transpose() * (inMat.m_topLeftMat - inMat.m_topRightMat * r_cross) * m_rotMat;
			outMat.m_topRightMat = m_rotMat.transpose() * inMat.m_topRightMat * m_rotMat;
			outMat.m_bottomLeftMat = m_rotMat.transpose() * (r_cross * (inMat.m_topLeftMat - inMat.m_topRightMat * r_cross) + inMat.m_bottomLeftMat - inMat.m_topLeftMat.transpose() * r_cross) * m_rotMat;
		}
		else if (outOp == Add)
		{
			outMat.m_topLeftMat += m_rotMat.transpose() * (inMat.m_topLeftMat - inMat.m_topRightMat * r_cross) * m_rotMat;
			outMat.m_topRightMat += m_rotMat.transpose() * inMat.m_topRightMat * m_rotMat;
			outMat.m_bottomLeftMat += m_rotMat.transpose() * (r_cross * (inMat.m_topLeftMat - inMat.m_topRightMat * r_cross) + inMat.m_bottomLeftMat - inMat.m_topLeftMat.transpose() * r_cross) * m_rotMat;
		}
		else if (outOp == Subtract)
		{
			outMat.m_topLeftMat -= m_rotMat.transpose() * (inMat.m_topLeftMat - inMat.m_topRightMat * r_cross) * m_rotMat;
			outMat.m_topRightMat -= m_rotMat.transpose() * inMat.m_topRightMat * m_rotMat;
			outMat.m_bottomLeftMat -= m_rotMat.transpose() * (r_cross * (inMat.m_topLeftMat - inMat.m_topRightMat * r_cross) + inMat.m_bottomLeftMat - inMat.m_topLeftMat.transpose() * r_cross) * m_rotMat;
		}
	}

	template <typename SpatialVectorType>
	SpatialVectorType operator*(const SpatialVectorType &vec)
	{
		SpatialVectorType out;
		transform(vec, out);
		return out;
	}
};

template <typename SpatialVectorType>
void symmetricSpatialOuterProduct(const SpatialVectorType &a, const SpatialVectorType &b, btSymmetricSpatialDyad &out)
{
	//output op maybe?

	out.m_topLeftMat = outerProduct(a.m_topVec, b.m_bottomVec);
	out.m_topRightMat = outerProduct(a.m_topVec, b.m_topVec);
	out.m_topLeftMat = outerProduct(a.m_bottomVec, b.m_bottomVec);
	//maybe simple a*spatTranspose(a) would be nicer?
}

template <typename SpatialVectorType>
btSymmetricSpatialDyad symmetricSpatialOuterProduct(const SpatialVectorType &a, const SpatialVectorType &b)
{
	btSymmetricSpatialDyad out;

	out.m_topLeftMat = outerProduct(a.m_topVec, b.m_bottomVec);
	out.m_topRightMat = outerProduct(a.m_topVec, b.m_topVec);
	out.m_bottomLeftMat = outerProduct(a.m_bottomVec, b.m_bottomVec);

	return out;
	//maybe simple a*spatTranspose(a) would be nicer?
}

#endif  //BT_SPATIAL_ALGEBRA_H
