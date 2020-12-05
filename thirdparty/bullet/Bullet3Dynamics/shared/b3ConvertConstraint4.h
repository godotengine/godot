

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"
#include "Bullet3Dynamics/shared/b3ContactConstraint4.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

void b3PlaneSpace1(b3Float4ConstArg n, b3Float4* p, b3Float4* q);
void b3PlaneSpace1(b3Float4ConstArg n, b3Float4* p, b3Float4* q)
{
	if (b3Fabs(n.z) > 0.70710678f)
	{
		// choose p in y-z plane
		float a = n.y * n.y + n.z * n.z;
		float k = 1.f / sqrt(a);
		p[0].x = 0;
		p[0].y = -n.z * k;
		p[0].z = n.y * k;
		// set q = n x p
		q[0].x = a * k;
		q[0].y = -n.x * p[0].z;
		q[0].z = n.x * p[0].y;
	}
	else
	{
		// choose p in x-y plane
		float a = n.x * n.x + n.y * n.y;
		float k = 1.f / sqrt(a);
		p[0].x = -n.y * k;
		p[0].y = n.x * k;
		p[0].z = 0;
		// set q = n x p
		q[0].x = -n.z * p[0].y;
		q[0].y = n.z * p[0].x;
		q[0].z = a * k;
	}
}

void setLinearAndAngular(b3Float4ConstArg n, b3Float4ConstArg r0, b3Float4ConstArg r1, b3Float4* linear, b3Float4* angular0, b3Float4* angular1)
{
	*linear = b3MakeFloat4(n.x, n.y, n.z, 0.f);
	*angular0 = b3Cross3(r0, n);
	*angular1 = -b3Cross3(r1, n);
}

float calcRelVel(b3Float4ConstArg l0, b3Float4ConstArg l1, b3Float4ConstArg a0, b3Float4ConstArg a1, b3Float4ConstArg linVel0,
				 b3Float4ConstArg angVel0, b3Float4ConstArg linVel1, b3Float4ConstArg angVel1)
{
	return b3Dot3F4(l0, linVel0) + b3Dot3F4(a0, angVel0) + b3Dot3F4(l1, linVel1) + b3Dot3F4(a1, angVel1);
}

float calcJacCoeff(b3Float4ConstArg linear0, b3Float4ConstArg linear1, b3Float4ConstArg angular0, b3Float4ConstArg angular1,
				   float invMass0, const b3Mat3x3* invInertia0, float invMass1, const b3Mat3x3* invInertia1)
{
	//	linear0,1 are normlized
	float jmj0 = invMass0;  //b3Dot3F4(linear0, linear0)*invMass0;
	float jmj1 = b3Dot3F4(mtMul3(angular0, *invInertia0), angular0);
	float jmj2 = invMass1;  //b3Dot3F4(linear1, linear1)*invMass1;
	float jmj3 = b3Dot3F4(mtMul3(angular1, *invInertia1), angular1);
	return -1.f / (jmj0 + jmj1 + jmj2 + jmj3);
}

void setConstraint4(b3Float4ConstArg posA, b3Float4ConstArg linVelA, b3Float4ConstArg angVelA, float invMassA, b3Mat3x3ConstArg invInertiaA,
					b3Float4ConstArg posB, b3Float4ConstArg linVelB, b3Float4ConstArg angVelB, float invMassB, b3Mat3x3ConstArg invInertiaB,
					__global struct b3Contact4Data* src, float dt, float positionDrift, float positionConstraintCoeff,
					b3ContactConstraint4_t* dstC)
{
	dstC->m_bodyA = abs(src->m_bodyAPtrAndSignBit);
	dstC->m_bodyB = abs(src->m_bodyBPtrAndSignBit);

	float dtInv = 1.f / dt;
	for (int ic = 0; ic < 4; ic++)
	{
		dstC->m_appliedRambdaDt[ic] = 0.f;
	}
	dstC->m_fJacCoeffInv[0] = dstC->m_fJacCoeffInv[1] = 0.f;

	dstC->m_linear = src->m_worldNormalOnB;
	dstC->m_linear.w = 0.7f;  //src->getFrictionCoeff() );
	for (int ic = 0; ic < 4; ic++)
	{
		b3Float4 r0 = src->m_worldPosB[ic] - posA;
		b3Float4 r1 = src->m_worldPosB[ic] - posB;

		if (ic >= src->m_worldNormalOnB.w)  //npoints
		{
			dstC->m_jacCoeffInv[ic] = 0.f;
			continue;
		}

		float relVelN;
		{
			b3Float4 linear, angular0, angular1;
			setLinearAndAngular(src->m_worldNormalOnB, r0, r1, &linear, &angular0, &angular1);

			dstC->m_jacCoeffInv[ic] = calcJacCoeff(linear, -linear, angular0, angular1,
												   invMassA, &invInertiaA, invMassB, &invInertiaB);

			relVelN = calcRelVel(linear, -linear, angular0, angular1,
								 linVelA, angVelA, linVelB, angVelB);

			float e = 0.f;  //src->getRestituitionCoeff();
			if (relVelN * relVelN < 0.004f) e = 0.f;

			dstC->m_b[ic] = e * relVelN;
			//float penetration = src->m_worldPosB[ic].w;
			dstC->m_b[ic] += (src->m_worldPosB[ic].w + positionDrift) * positionConstraintCoeff * dtInv;
			dstC->m_appliedRambdaDt[ic] = 0.f;
		}
	}

	if (src->m_worldNormalOnB.w > 0)  //npoints
	{                                 //	prepare friction
		b3Float4 center = b3MakeFloat4(0.f, 0.f, 0.f, 0.f);
		for (int i = 0; i < src->m_worldNormalOnB.w; i++)
			center += src->m_worldPosB[i];
		center /= (float)src->m_worldNormalOnB.w;

		b3Float4 tangent[2];
		b3PlaneSpace1(src->m_worldNormalOnB, &tangent[0], &tangent[1]);

		b3Float4 r[2];
		r[0] = center - posA;
		r[1] = center - posB;

		for (int i = 0; i < 2; i++)
		{
			b3Float4 linear, angular0, angular1;
			setLinearAndAngular(tangent[i], r[0], r[1], &linear, &angular0, &angular1);

			dstC->m_fJacCoeffInv[i] = calcJacCoeff(linear, -linear, angular0, angular1,
												   invMassA, &invInertiaA, invMassB, &invInertiaB);
			dstC->m_fAppliedRambdaDt[i] = 0.f;
		}
		dstC->m_center = center;
	}

	for (int i = 0; i < 4; i++)
	{
		if (i < src->m_worldNormalOnB.w)
		{
			dstC->m_worldPos[i] = src->m_worldPosB[i];
		}
		else
		{
			dstC->m_worldPos[i] = b3MakeFloat4(0.f, 0.f, 0.f, 0.f);
		}
	}
}
