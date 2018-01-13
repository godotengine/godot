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


#ifndef B3_TRANSFORM_UTIL_H
#define B3_TRANSFORM_UTIL_H

#include "b3Transform.h"
#define B3_ANGULAR_MOTION_THRESHOLD b3Scalar(0.5)*B3_HALF_PI




B3_FORCE_INLINE b3Vector3 b3AabbSupport(const b3Vector3& halfExtents,const b3Vector3& supportDir)
{
	return b3MakeVector3(supportDir.getX() < b3Scalar(0.0) ? -halfExtents.getX() : halfExtents.getX(),
      supportDir.getY() < b3Scalar(0.0) ? -halfExtents.getY() : halfExtents.getY(),
      supportDir.getZ() < b3Scalar(0.0) ? -halfExtents.getZ() : halfExtents.getZ()); 
}






/// Utils related to temporal transforms
class b3TransformUtil
{

public:

	static void integrateTransform(const b3Transform& curTrans,const b3Vector3& linvel,const b3Vector3& angvel,b3Scalar timeStep,b3Transform& predictedTransform)
	{
		predictedTransform.setOrigin(curTrans.getOrigin() + linvel * timeStep);
//	#define QUATERNION_DERIVATIVE
	#ifdef QUATERNION_DERIVATIVE
		b3Quaternion predictedOrn = curTrans.getRotation();
		predictedOrn += (angvel * predictedOrn) * (timeStep * b3Scalar(0.5));
		predictedOrn.normalize();
	#else
		//Exponential map
		//google for "Practical Parameterization of Rotations Using the Exponential Map", F. Sebastian Grassia

		b3Vector3 axis;
		b3Scalar	fAngle = angvel.length(); 
		//limit the angular motion
		if (fAngle*timeStep > B3_ANGULAR_MOTION_THRESHOLD)
		{
			fAngle = B3_ANGULAR_MOTION_THRESHOLD / timeStep;
		}

		if ( fAngle < b3Scalar(0.001) )
		{
			// use Taylor's expansions of sync function
			axis   = angvel*( b3Scalar(0.5)*timeStep-(timeStep*timeStep*timeStep)*(b3Scalar(0.020833333333))*fAngle*fAngle );
		}
		else
		{
			// sync(fAngle) = sin(c*fAngle)/t
			axis   = angvel*( b3Sin(b3Scalar(0.5)*fAngle*timeStep)/fAngle );
		}
		b3Quaternion dorn (axis.getX(),axis.getY(),axis.getZ(),b3Cos( fAngle*timeStep*b3Scalar(0.5) ));
		b3Quaternion orn0 = curTrans.getRotation();

		b3Quaternion predictedOrn = dorn * orn0;
		predictedOrn.normalize();
	#endif
		predictedTransform.setRotation(predictedOrn);
	}

	static void	calculateVelocityQuaternion(const b3Vector3& pos0,const b3Vector3& pos1,const b3Quaternion& orn0,const b3Quaternion& orn1,b3Scalar timeStep,b3Vector3& linVel,b3Vector3& angVel)
	{
		linVel = (pos1 - pos0) / timeStep;
		b3Vector3 axis;
		b3Scalar  angle;
		if (orn0 != orn1)
		{
			calculateDiffAxisAngleQuaternion(orn0,orn1,axis,angle);
			angVel = axis * angle / timeStep;
		} else
		{
			angVel.setValue(0,0,0);
		}
	}

	static void calculateDiffAxisAngleQuaternion(const b3Quaternion& orn0,const b3Quaternion& orn1a,b3Vector3& axis,b3Scalar& angle)
	{
		b3Quaternion orn1 = orn0.nearest(orn1a);
		b3Quaternion dorn = orn1 * orn0.inverse();
		angle = dorn.getAngle();
		axis = b3MakeVector3(dorn.getX(),dorn.getY(),dorn.getZ());
		axis[3] = b3Scalar(0.);
		//check for axis length
		b3Scalar len = axis.length2();
		if (len < B3_EPSILON*B3_EPSILON)
			axis = b3MakeVector3(b3Scalar(1.),b3Scalar(0.),b3Scalar(0.));
		else
			axis /= b3Sqrt(len);
	}

	static void	calculateVelocity(const b3Transform& transform0,const b3Transform& transform1,b3Scalar timeStep,b3Vector3& linVel,b3Vector3& angVel)
	{
		linVel = (transform1.getOrigin() - transform0.getOrigin()) / timeStep;
		b3Vector3 axis;
		b3Scalar  angle;
		calculateDiffAxisAngle(transform0,transform1,axis,angle);
		angVel = axis * angle / timeStep;
	}

	static void calculateDiffAxisAngle(const b3Transform& transform0,const b3Transform& transform1,b3Vector3& axis,b3Scalar& angle)
	{
		b3Matrix3x3 dmat = transform1.getBasis() * transform0.getBasis().inverse();
		b3Quaternion dorn;
		dmat.getRotation(dorn);

		///floating point inaccuracy can lead to w component > 1..., which breaks 
		dorn.normalize();
		
		angle = dorn.getAngle();
		axis = b3MakeVector3(dorn.getX(),dorn.getY(),dorn.getZ());
		axis[3] = b3Scalar(0.);
		//check for axis length
		b3Scalar len = axis.length2();
		if (len < B3_EPSILON*B3_EPSILON)
			axis = b3MakeVector3(b3Scalar(1.),b3Scalar(0.),b3Scalar(0.));
		else
			axis /= b3Sqrt(len);
	}

};


///The b3ConvexSeparatingDistanceUtil can help speed up convex collision detection 
///by conservatively updating a cached separating distance/vector instead of re-calculating the closest distance
class	b3ConvexSeparatingDistanceUtil
{
	b3Quaternion	m_ornA;
	b3Quaternion	m_ornB;
	b3Vector3	m_posA;
	b3Vector3	m_posB;
	
	b3Vector3	m_separatingNormal;

	b3Scalar	m_boundingRadiusA;
	b3Scalar	m_boundingRadiusB;
	b3Scalar	m_separatingDistance;

public:

	b3ConvexSeparatingDistanceUtil(b3Scalar	boundingRadiusA,b3Scalar	boundingRadiusB)
		:m_boundingRadiusA(boundingRadiusA),
		m_boundingRadiusB(boundingRadiusB),
		m_separatingDistance(0.f)
	{
	}

	b3Scalar	getConservativeSeparatingDistance()
	{
		return m_separatingDistance;
	}

	void	updateSeparatingDistance(const b3Transform& transA,const b3Transform& transB)
	{
		const b3Vector3& toPosA = transA.getOrigin();
		const b3Vector3& toPosB = transB.getOrigin();
		b3Quaternion toOrnA = transA.getRotation();
		b3Quaternion toOrnB = transB.getRotation();

		if (m_separatingDistance>0.f)
		{
			

			b3Vector3 linVelA,angVelA,linVelB,angVelB;
			b3TransformUtil::calculateVelocityQuaternion(m_posA,toPosA,m_ornA,toOrnA,b3Scalar(1.),linVelA,angVelA);
			b3TransformUtil::calculateVelocityQuaternion(m_posB,toPosB,m_ornB,toOrnB,b3Scalar(1.),linVelB,angVelB);
			b3Scalar maxAngularProjectedVelocity = angVelA.length() * m_boundingRadiusA + angVelB.length() * m_boundingRadiusB;
			b3Vector3 relLinVel = (linVelB-linVelA);
			b3Scalar relLinVelocLength = relLinVel.dot(m_separatingNormal);
			if (relLinVelocLength<0.f)
			{
				relLinVelocLength = 0.f;
			}
	
			b3Scalar	projectedMotion = maxAngularProjectedVelocity +relLinVelocLength;
			m_separatingDistance -= projectedMotion;
		}
	
		m_posA = toPosA;
		m_posB = toPosB;
		m_ornA = toOrnA;
		m_ornB = toOrnB;
	}

	void	initSeparatingDistance(const b3Vector3& separatingVector,b3Scalar separatingDistance,const b3Transform& transA,const b3Transform& transB)
	{
		m_separatingDistance = separatingDistance;

		if (m_separatingDistance>0.f)
		{
			m_separatingNormal = separatingVector;
			
			const b3Vector3& toPosA = transA.getOrigin();
			const b3Vector3& toPosB = transB.getOrigin();
			b3Quaternion toOrnA = transA.getRotation();
			b3Quaternion toOrnB = transB.getRotation();
			m_posA = toPosA;
			m_posB = toPosB;
			m_ornA = toOrnA;
			m_ornB = toOrnB;
		}
	}

};


#endif //B3_TRANSFORM_UTIL_H

