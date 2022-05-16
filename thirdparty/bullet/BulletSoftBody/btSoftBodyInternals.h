/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
///btSoftBody implementation by Nathanael Presson

#ifndef _BT_SOFT_BODY_INTERNALS_H
#define _BT_SOFT_BODY_INTERNALS_H

#include "btSoftBody.h"
#include "LinearMath/btQuickprof.h"
#include "LinearMath/btPolarDecomposition.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h"
#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "BulletCollision/CollisionShapes/btConvexInternalShape.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkEpa2.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraint.h"
#include <string.h>  //for memset
#include <cmath>
#include "poly34.h"

// Given a multibody link, a contact point and a contact direction, fill in the jacobian data needed to calculate the velocity change given an impulse in the contact direction
static SIMD_FORCE_INLINE void findJacobian(const btMultiBodyLinkCollider* multibodyLinkCol,
										   btMultiBodyJacobianData& jacobianData,
										   const btVector3& contact_point,
										   const btVector3& dir)
{
	const int ndof = multibodyLinkCol->m_multiBody->getNumDofs() + 6;
	jacobianData.m_jacobians.resize(ndof);
	jacobianData.m_deltaVelocitiesUnitImpulse.resize(ndof);
	btScalar* jac = &jacobianData.m_jacobians[0];

	multibodyLinkCol->m_multiBody->fillContactJacobianMultiDof(multibodyLinkCol->m_link, contact_point, dir, jac, jacobianData.scratch_r, jacobianData.scratch_v, jacobianData.scratch_m);
	multibodyLinkCol->m_multiBody->calcAccelerationDeltasMultiDof(&jacobianData.m_jacobians[0], &jacobianData.m_deltaVelocitiesUnitImpulse[0], jacobianData.scratch_r, jacobianData.scratch_v);
}
static SIMD_FORCE_INLINE btVector3 generateUnitOrthogonalVector(const btVector3& u)
{
	btScalar ux = u.getX();
	btScalar uy = u.getY();
	btScalar uz = u.getZ();
	btScalar ax = std::abs(ux);
	btScalar ay = std::abs(uy);
	btScalar az = std::abs(uz);
	btVector3 v;
	if (ax <= ay && ax <= az)
		v = btVector3(0, -uz, uy);
	else if (ay <= ax && ay <= az)
		v = btVector3(-uz, 0, ux);
	else
		v = btVector3(-uy, ux, 0);
	v.normalize();
	return v;
}

static SIMD_FORCE_INLINE bool proximityTest(const btVector3& x1, const btVector3& x2, const btVector3& x3, const btVector3& x4, const btVector3& normal, const btScalar& mrg, btVector3& bary)
{
	btVector3 x43 = x4 - x3;
	if (std::abs(x43.dot(normal)) > mrg)
		return false;
	btVector3 x13 = x1 - x3;
	btVector3 x23 = x2 - x3;
	btScalar a11 = x13.length2();
	btScalar a22 = x23.length2();
	btScalar a12 = x13.dot(x23);
	btScalar b1 = x13.dot(x43);
	btScalar b2 = x23.dot(x43);
	btScalar det = a11 * a22 - a12 * a12;
	if (det < SIMD_EPSILON)
		return false;
	btScalar w1 = (b1 * a22 - b2 * a12) / det;
	btScalar w2 = (b2 * a11 - b1 * a12) / det;
	btScalar w3 = 1 - w1 - w2;
	btScalar delta = mrg / std::sqrt(0.5 * std::abs(x13.cross(x23).safeNorm()));
	bary = btVector3(w1, w2, w3);
	for (int i = 0; i < 3; ++i)
	{
		if (bary[i] < -delta || bary[i] > 1 + delta)
			return false;
	}
	return true;
}
static const int KDOP_COUNT = 13;
static btVector3 dop[KDOP_COUNT] = {btVector3(1, 0, 0),
									btVector3(0, 1, 0),
									btVector3(0, 0, 1),
									btVector3(1, 1, 0),
									btVector3(1, 0, 1),
									btVector3(0, 1, 1),
									btVector3(1, -1, 0),
									btVector3(1, 0, -1),
									btVector3(0, 1, -1),
									btVector3(1, 1, 1),
									btVector3(1, -1, 1),
									btVector3(1, 1, -1),
									btVector3(1, -1, -1)};

static inline int getSign(const btVector3& n, const btVector3& x)
{
	btScalar d = n.dot(x);
	if (d > SIMD_EPSILON)
		return 1;
	if (d < -SIMD_EPSILON)
		return -1;
	return 0;
}

static SIMD_FORCE_INLINE bool hasSeparatingPlane(const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt)
{
	btVector3 hex[6] = {face->m_n[0]->m_x - node->m_x,
						face->m_n[1]->m_x - node->m_x,
						face->m_n[2]->m_x - node->m_x,
						face->m_n[0]->m_x + dt * face->m_n[0]->m_v - node->m_x,
						face->m_n[1]->m_x + dt * face->m_n[1]->m_v - node->m_x,
						face->m_n[2]->m_x + dt * face->m_n[2]->m_v - node->m_x};
	btVector3 segment = dt * node->m_v;
	for (int i = 0; i < KDOP_COUNT; ++i)
	{
		int s = getSign(dop[i], segment);
		int j = 0;
		for (; j < 6; ++j)
		{
			if (getSign(dop[i], hex[j]) == s)
				break;
		}
		if (j == 6)
			return true;
	}
	return false;
}

static SIMD_FORCE_INLINE bool nearZero(const btScalar& a)
{
	return (a > -SAFE_EPSILON && a < SAFE_EPSILON);
}
static SIMD_FORCE_INLINE bool sameSign(const btScalar& a, const btScalar& b)
{
	return (nearZero(a) || nearZero(b) || (a > SAFE_EPSILON && b > SAFE_EPSILON) || (a < -SAFE_EPSILON && b < -SAFE_EPSILON));
}
static SIMD_FORCE_INLINE bool diffSign(const btScalar& a, const btScalar& b)
{
	return !sameSign(a, b);
}
inline btScalar evaluateBezier2(const btScalar& p0, const btScalar& p1, const btScalar& p2, const btScalar& t, const btScalar& s)
{
	btScalar s2 = s * s;
	btScalar t2 = t * t;

	return p0 * s2 + p1 * btScalar(2.0) * s * t + p2 * t2;
}
inline btScalar evaluateBezier(const btScalar& p0, const btScalar& p1, const btScalar& p2, const btScalar& p3, const btScalar& t, const btScalar& s)
{
	btScalar s2 = s * s;
	btScalar s3 = s2 * s;
	btScalar t2 = t * t;
	btScalar t3 = t2 * t;

	return p0 * s3 + p1 * btScalar(3.0) * s2 * t + p2 * btScalar(3.0) * s * t2 + p3 * t3;
}
static SIMD_FORCE_INLINE bool getSigns(bool type_c, const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btScalar& t0, const btScalar& t1, btScalar& lt0, btScalar& lt1)
{
	if (sameSign(t0, t1))
	{
		lt0 = t0;
		lt1 = t0;
		return true;
	}

	if (type_c || diffSign(k0, k3))
	{
		btScalar ft = evaluateBezier(k0, k1, k2, k3, t0, -t1);
		if (t0 < -0)
			ft = -ft;

		if (sameSign(ft, k0))
		{
			lt0 = t1;
			lt1 = t1;
		}
		else
		{
			lt0 = t0;
			lt1 = t0;
		}
		return true;
	}

	if (!type_c)
	{
		btScalar ft = evaluateBezier(k0, k1, k2, k3, t0, -t1);
		if (t0 < -0)
			ft = -ft;

		if (diffSign(ft, k0))
		{
			lt0 = t0;
			lt1 = t1;
			return true;
		}

		btScalar fk = evaluateBezier2(k1 - k0, k2 - k1, k3 - k2, t0, -t1);

		if (sameSign(fk, k1 - k0))
			lt0 = lt1 = t1;
		else
			lt0 = lt1 = t0;

		return true;
	}
	return false;
}

static SIMD_FORCE_INLINE void getBernsteinCoeff(const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt, btScalar& k0, btScalar& k1, btScalar& k2, btScalar& k3)
{
	const btVector3& n0 = face->m_n0;
	const btVector3& n1 = face->m_n1;
	btVector3 n_hat = n0 + n1 - face->m_vn;
	btVector3 p0ma0 = node->m_x - face->m_n[0]->m_x;
	btVector3 p1ma1 = node->m_q - face->m_n[0]->m_q;
	k0 = (p0ma0).dot(n0) * 3.0;
	k1 = (p0ma0).dot(n_hat) + (p1ma1).dot(n0);
	k2 = (p1ma1).dot(n_hat) + (p0ma0).dot(n1);
	k3 = (p1ma1).dot(n1) * 3.0;
}

static SIMD_FORCE_INLINE void polyDecomposition(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btScalar& j0, const btScalar& j1, const btScalar& j2, btScalar& u0, btScalar& u1, btScalar& v0, btScalar& v1)
{
	btScalar denom = 4.0 * (j1 - j2) * (j1 - j0) + (j2 - j0) * (j2 - j0);
	u0 = (2.0 * (j1 - j2) * (3.0 * k1 - 2.0 * k0 - k3) - (j0 - j2) * (3.0 * k2 - 2.0 * k3 - k0)) / denom;
	u1 = (2.0 * (j1 - j0) * (3.0 * k2 - 2.0 * k3 - k0) - (j2 - j0) * (3.0 * k1 - 2.0 * k0 - k3)) / denom;
	v0 = k0 - u0 * j0;
	v1 = k3 - u1 * j2;
}

static SIMD_FORCE_INLINE bool rootFindingLemma(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3)
{
	btScalar u0, u1, v0, v1;
	btScalar j0 = 3.0 * (k1 - k0);
	btScalar j1 = 3.0 * (k2 - k1);
	btScalar j2 = 3.0 * (k3 - k2);
	polyDecomposition(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1);
	if (sameSign(v0, v1))
	{
		btScalar Ypa = j0 * (1.0 - v0) * (1.0 - v0) + 2.0 * j1 * v0 * (1.0 - v0) + j2 * v0 * v0;  // Y'(v0)
		if (sameSign(Ypa, j0))
		{
			return (diffSign(k0, v1));
		}
	}
	return diffSign(k0, v0);
}

static SIMD_FORCE_INLINE void getJs(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btSoftBody::Node* a, const btSoftBody::Node* b, const btSoftBody::Node* c, const btSoftBody::Node* p, const btScalar& dt, btScalar& j0, btScalar& j1, btScalar& j2)
{
	const btVector3& a0 = a->m_x;
	const btVector3& b0 = b->m_x;
	const btVector3& c0 = c->m_x;
	const btVector3& va = a->m_v;
	const btVector3& vb = b->m_v;
	const btVector3& vc = c->m_v;
	const btVector3 a1 = a0 + dt * va;
	const btVector3 b1 = b0 + dt * vb;
	const btVector3 c1 = c0 + dt * vc;
	btVector3 n0 = (b0 - a0).cross(c0 - a0);
	btVector3 n1 = (b1 - a1).cross(c1 - a1);
	btVector3 n_hat = n0 + n1 - dt * dt * (vb - va).cross(vc - va);
	const btVector3& p0 = p->m_x;
	const btVector3& vp = p->m_v;
	btVector3 p1 = p0 + dt * vp;
	btVector3 m0 = (b0 - p0).cross(c0 - p0);
	btVector3 m1 = (b1 - p1).cross(c1 - p1);
	btVector3 m_hat = m0 + m1 - dt * dt * (vb - vp).cross(vc - vp);
	btScalar l0 = m0.dot(n0);
	btScalar l1 = 0.25 * (m0.dot(n_hat) + m_hat.dot(n0));
	btScalar l2 = btScalar(1) / btScalar(6) * (m0.dot(n1) + m_hat.dot(n_hat) + m1.dot(n0));
	btScalar l3 = 0.25 * (m_hat.dot(n1) + m1.dot(n_hat));
	btScalar l4 = m1.dot(n1);

	btScalar k1p = 0.25 * k0 + 0.75 * k1;
	btScalar k2p = 0.5 * k1 + 0.5 * k2;
	btScalar k3p = 0.75 * k2 + 0.25 * k3;

	btScalar s0 = (l1 * k0 - l0 * k1p) * 4.0;
	btScalar s1 = (l2 * k0 - l0 * k2p) * 2.0;
	btScalar s2 = (l3 * k0 - l0 * k3p) * btScalar(4) / btScalar(3);
	btScalar s3 = l4 * k0 - l0 * k3;

	j0 = (s1 * k0 - s0 * k1) * 3.0;
	j1 = (s2 * k0 - s0 * k2) * 1.5;
	j2 = (s3 * k0 - s0 * k3);
}

static SIMD_FORCE_INLINE bool signDetermination1Internal(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btScalar& u0, const btScalar& u1, const btScalar& v0, const btScalar& v1)
{
	btScalar Yu0 = k0 * (1.0 - u0) * (1.0 - u0) * (1.0 - u0) + 3.0 * k1 * u0 * (1.0 - u0) * (1.0 - u0) + 3.0 * k2 * u0 * u0 * (1.0 - u0) + k3 * u0 * u0 * u0;  // Y(u0)
	btScalar Yv0 = k0 * (1.0 - v0) * (1.0 - v0) * (1.0 - v0) + 3.0 * k1 * v0 * (1.0 - v0) * (1.0 - v0) + 3.0 * k2 * v0 * v0 * (1.0 - v0) + k3 * v0 * v0 * v0;  // Y(v0)

	btScalar sign_Ytp = (u0 > u1) ? Yu0 : -Yu0;
	btScalar L = sameSign(sign_Ytp, k0) ? u1 : u0;
	sign_Ytp = (v0 > v1) ? Yv0 : -Yv0;
	btScalar K = (sameSign(sign_Ytp, k0)) ? v1 : v0;
	return diffSign(L, K);
}

static SIMD_FORCE_INLINE bool signDetermination2Internal(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btScalar& j0, const btScalar& j1, const btScalar& j2, const btScalar& u0, const btScalar& u1, const btScalar& v0, const btScalar& v1)
{
	btScalar Yu0 = k0 * (1.0 - u0) * (1.0 - u0) * (1.0 - u0) + 3.0 * k1 * u0 * (1.0 - u0) * (1.0 - u0) + 3.0 * k2 * u0 * u0 * (1.0 - u0) + k3 * u0 * u0 * u0;  // Y(u0)
	btScalar sign_Ytp = (u0 > u1) ? Yu0 : -Yu0, L1, L2;
	if (diffSign(sign_Ytp, k0))
	{
		L1 = u0;
		L2 = u1;
	}
	else
	{
		btScalar Yp_u0 = j0 * (1.0 - u0) * (1.0 - u0) + 2.0 * j1 * (1.0 - u0) * u0 + j2 * u0 * u0;
		if (sameSign(Yp_u0, j0))
		{
			L1 = u1;
			L2 = u1;
		}
		else
		{
			L1 = u0;
			L2 = u0;
		}
	}
	btScalar Yv0 = k0 * (1.0 - v0) * (1.0 - v0) * (1.0 - v0) + 3.0 * k1 * v0 * (1.0 - v0) * (1.0 - v0) + 3.0 * k2 * v0 * v0 * (1.0 - v0) + k3 * v0 * v0 * v0;  // Y(uv0)
	sign_Ytp = (v0 > v1) ? Yv0 : -Yv0;
	btScalar K1, K2;
	if (diffSign(sign_Ytp, k0))
	{
		K1 = v0;
		K2 = v1;
	}
	else
	{
		btScalar Yp_v0 = j0 * (1.0 - v0) * (1.0 - v0) + 2.0 * j1 * (1.0 - v0) * v0 + j2 * v0 * v0;
		if (sameSign(Yp_v0, j0))
		{
			K1 = v1;
			K2 = v1;
		}
		else
		{
			K1 = v0;
			K2 = v0;
		}
	}
	return (diffSign(K1, L1) || diffSign(L2, K2));
}

static SIMD_FORCE_INLINE bool signDetermination1(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt)
{
	btScalar j0, j1, j2, u0, u1, v0, v1;
	// p1
	getJs(k0, k1, k2, k3, face->m_n[0], face->m_n[1], face->m_n[2], node, dt, j0, j1, j2);
	if (nearZero(j0 + j2 - j1 * 2.0))
	{
		btScalar lt0, lt1;
		getSigns(true, k0, k1, k2, k3, j0, j2, lt0, lt1);
		if (lt0 < -SAFE_EPSILON)
			return false;
	}
	else
	{
		polyDecomposition(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1);
		if (!signDetermination1Internal(k0, k1, k2, k3, u0, u1, v0, v1))
			return false;
	}
	// p2
	getJs(k0, k1, k2, k3, face->m_n[1], face->m_n[2], face->m_n[0], node, dt, j0, j1, j2);
	if (nearZero(j0 + j2 - j1 * 2.0))
	{
		btScalar lt0, lt1;
		getSigns(true, k0, k1, k2, k3, j0, j2, lt0, lt1);
		if (lt0 < -SAFE_EPSILON)
			return false;
	}
	else
	{
		polyDecomposition(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1);
		if (!signDetermination1Internal(k0, k1, k2, k3, u0, u1, v0, v1))
			return false;
	}
	// p3
	getJs(k0, k1, k2, k3, face->m_n[2], face->m_n[0], face->m_n[1], node, dt, j0, j1, j2);
	if (nearZero(j0 + j2 - j1 * 2.0))
	{
		btScalar lt0, lt1;
		getSigns(true, k0, k1, k2, k3, j0, j2, lt0, lt1);
		if (lt0 < -SAFE_EPSILON)
			return false;
	}
	else
	{
		polyDecomposition(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1);
		if (!signDetermination1Internal(k0, k1, k2, k3, u0, u1, v0, v1))
			return false;
	}
	return true;
}

static SIMD_FORCE_INLINE bool signDetermination2(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt)
{
	btScalar j0, j1, j2, u0, u1, v0, v1;
	// p1
	getJs(k0, k1, k2, k3, face->m_n[0], face->m_n[1], face->m_n[2], node, dt, j0, j1, j2);
	if (nearZero(j0 + j2 - j1 * 2.0))
	{
		btScalar lt0, lt1;
		bool bt0 = true, bt1 = true;
		getSigns(false, k0, k1, k2, k3, j0, j2, lt0, lt1);
		if (lt0 < -SAFE_EPSILON)
			bt0 = false;
		if (lt1 < -SAFE_EPSILON)
			bt1 = false;
		if (!bt0 && !bt1)
			return false;
	}
	else
	{
		polyDecomposition(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1);
		if (!signDetermination2Internal(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1))
			return false;
	}
	// p2
	getJs(k0, k1, k2, k3, face->m_n[1], face->m_n[2], face->m_n[0], node, dt, j0, j1, j2);
	if (nearZero(j0 + j2 - j1 * 2.0))
	{
		btScalar lt0, lt1;
		bool bt0 = true, bt1 = true;
		getSigns(false, k0, k1, k2, k3, j0, j2, lt0, lt1);
		if (lt0 < -SAFE_EPSILON)
			bt0 = false;
		if (lt1 < -SAFE_EPSILON)
			bt1 = false;
		if (!bt0 && !bt1)
			return false;
	}
	else
	{
		polyDecomposition(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1);
		if (!signDetermination2Internal(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1))
			return false;
	}
	// p3
	getJs(k0, k1, k2, k3, face->m_n[2], face->m_n[0], face->m_n[1], node, dt, j0, j1, j2);
	if (nearZero(j0 + j2 - j1 * 2.0))
	{
		btScalar lt0, lt1;
		bool bt0 = true, bt1 = true;
		getSigns(false, k0, k1, k2, k3, j0, j2, lt0, lt1);
		if (lt0 < -SAFE_EPSILON)
			bt0 = false;
		if (lt1 < -SAFE_EPSILON)
			bt1 = false;
		if (!bt0 && !bt1)
			return false;
	}
	else
	{
		polyDecomposition(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1);
		if (!signDetermination2Internal(k0, k1, k2, k3, j0, j1, j2, u0, u1, v0, v1))
			return false;
	}
	return true;
}

static SIMD_FORCE_INLINE bool coplanarAndInsideTest(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt)
{
	// Coplanar test
	if (diffSign(k1 - k0, k3 - k2))
	{
		// Case b:
		if (sameSign(k0, k3) && !rootFindingLemma(k0, k1, k2, k3))
			return false;
		// inside test
		return signDetermination2(k0, k1, k2, k3, face, node, dt);
	}
	else
	{
		// Case c:
		if (sameSign(k0, k3))
			return false;
		// inside test
		return signDetermination1(k0, k1, k2, k3, face, node, dt);
	}
	return false;
}
static SIMD_FORCE_INLINE bool conservativeCulling(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btScalar& mrg)
{
	if (k0 > mrg && k1 > mrg && k2 > mrg && k3 > mrg)
		return true;
	if (k0 < -mrg && k1 < -mrg && k2 < -mrg && k3 < -mrg)
		return true;
	return false;
}

static SIMD_FORCE_INLINE bool bernsteinVFTest(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btScalar& mrg, const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt)
{
	if (conservativeCulling(k0, k1, k2, k3, mrg))
		return false;
	return coplanarAndInsideTest(k0, k1, k2, k3, face, node, dt);
}

static SIMD_FORCE_INLINE void deCasteljau(const btScalar& k0, const btScalar& k1, const btScalar& k2, const btScalar& k3, const btScalar& t0, btScalar& k10, btScalar& k20, btScalar& k30, btScalar& k21, btScalar& k12)
{
	k10 = k0 * (1.0 - t0) + k1 * t0;
	btScalar k11 = k1 * (1.0 - t0) + k2 * t0;
	k12 = k2 * (1.0 - t0) + k3 * t0;
	k20 = k10 * (1.0 - t0) + k11 * t0;
	k21 = k11 * (1.0 - t0) + k12 * t0;
	k30 = k20 * (1.0 - t0) + k21 * t0;
}
static SIMD_FORCE_INLINE bool bernsteinVFTest(const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt, const btScalar& mrg)
{
	btScalar k0, k1, k2, k3;
	getBernsteinCoeff(face, node, dt, k0, k1, k2, k3);
	if (conservativeCulling(k0, k1, k2, k3, mrg))
		return false;
	return true;
	if (diffSign(k2 - 2.0 * k1 + k0, k3 - 2.0 * k2 + k1))
	{
		btScalar k10, k20, k30, k21, k12;
		btScalar t0 = (k2 - 2.0 * k1 + k0) / (k0 - 3.0 * k1 + 3.0 * k2 - k3);
		deCasteljau(k0, k1, k2, k3, t0, k10, k20, k30, k21, k12);
		return bernsteinVFTest(k0, k10, k20, k30, mrg, face, node, dt) || bernsteinVFTest(k30, k21, k12, k3, mrg, face, node, dt);
	}
	return coplanarAndInsideTest(k0, k1, k2, k3, face, node, dt);
}

static SIMD_FORCE_INLINE bool continuousCollisionDetection(const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt, const btScalar& mrg, btVector3& bary)
{
	if (hasSeparatingPlane(face, node, dt))
		return false;
	btVector3 x21 = face->m_n[1]->m_x - face->m_n[0]->m_x;
	btVector3 x31 = face->m_n[2]->m_x - face->m_n[0]->m_x;
	btVector3 x41 = node->m_x - face->m_n[0]->m_x;
	btVector3 v21 = face->m_n[1]->m_v - face->m_n[0]->m_v;
	btVector3 v31 = face->m_n[2]->m_v - face->m_n[0]->m_v;
	btVector3 v41 = node->m_v - face->m_n[0]->m_v;
	btVector3 a = x21.cross(x31);
	btVector3 b = x21.cross(v31) + v21.cross(x31);
	btVector3 c = v21.cross(v31);
	btVector3 d = x41;
	btVector3 e = v41;
	btScalar a0 = a.dot(d);
	btScalar a1 = a.dot(e) + b.dot(d);
	btScalar a2 = c.dot(d) + b.dot(e);
	btScalar a3 = c.dot(e);
	btScalar eps = SAFE_EPSILON;
	int num_roots = 0;
	btScalar roots[3];
	if (std::abs(a3) < eps)
	{
		// cubic term is zero
		if (std::abs(a2) < eps)
		{
			if (std::abs(a1) < eps)
			{
				if (std::abs(a0) < eps)
				{
					num_roots = 2;
					roots[0] = 0;
					roots[1] = dt;
				}
			}
			else
			{
				num_roots = 1;
				roots[0] = -a0 / a1;
			}
		}
		else
		{
			num_roots = SolveP2(roots, a1 / a2, a0 / a2);
		}
	}
	else
	{
		num_roots = SolveP3(roots, a2 / a3, a1 / a3, a0 / a3);
	}
	//    std::sort(roots, roots+num_roots);
	if (num_roots > 1)
	{
		if (roots[0] > roots[1])
			btSwap(roots[0], roots[1]);
	}
	if (num_roots > 2)
	{
		if (roots[0] > roots[2])
			btSwap(roots[0], roots[2]);
		if (roots[1] > roots[2])
			btSwap(roots[1], roots[2]);
	}
	for (int r = 0; r < num_roots; ++r)
	{
		double root = roots[r];
		if (root <= 0)
			continue;
		if (root > dt + SIMD_EPSILON)
			return false;
		btVector3 x1 = face->m_n[0]->m_x + root * face->m_n[0]->m_v;
		btVector3 x2 = face->m_n[1]->m_x + root * face->m_n[1]->m_v;
		btVector3 x3 = face->m_n[2]->m_x + root * face->m_n[2]->m_v;
		btVector3 x4 = node->m_x + root * node->m_v;
		btVector3 normal = (x2 - x1).cross(x3 - x1);
		normal.safeNormalize();
		if (proximityTest(x1, x2, x3, x4, normal, mrg, bary))
			return true;
	}
	return false;
}
static SIMD_FORCE_INLINE bool bernsteinCCD(const btSoftBody::Face* face, const btSoftBody::Node* node, const btScalar& dt, const btScalar& mrg, btVector3& bary)
{
	if (!bernsteinVFTest(face, node, dt, mrg))
		return false;
	if (!continuousCollisionDetection(face, node, dt, 1e-6, bary))
		return false;
	return true;
}

//
// btSymMatrix
//
template <typename T>
struct btSymMatrix
{
	btSymMatrix() : dim(0) {}
	btSymMatrix(int n, const T& init = T()) { resize(n, init); }
	void resize(int n, const T& init = T())
	{
		dim = n;
		store.resize((n * (n + 1)) / 2, init);
	}
	int index(int c, int r) const
	{
		if (c > r) btSwap(c, r);
		btAssert(r < dim);
		return ((r * (r + 1)) / 2 + c);
	}
	T& operator()(int c, int r) { return (store[index(c, r)]); }
	const T& operator()(int c, int r) const { return (store[index(c, r)]); }
	btAlignedObjectArray<T> store;
	int dim;
};

//
// btSoftBodyCollisionShape
//
class btSoftBodyCollisionShape : public btConcaveShape
{
public:
	btSoftBody* m_body;

	btSoftBodyCollisionShape(btSoftBody* backptr)
	{
		m_shapeType = SOFTBODY_SHAPE_PROXYTYPE;
		m_body = backptr;
	}

	virtual ~btSoftBodyCollisionShape()
	{
	}

	void processAllTriangles(btTriangleCallback* /*callback*/, const btVector3& /*aabbMin*/, const btVector3& /*aabbMax*/) const
	{
		//not yet
		btAssert(0);
	}

	///getAabb returns the axis aligned bounding box in the coordinate frame of the given transform t.
	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const
	{
		/* t is usually identity, except when colliding against btCompoundShape. See Issue 512 */
		const btVector3 mins = m_body->m_bounds[0];
		const btVector3 maxs = m_body->m_bounds[1];
		const btVector3 crns[] = {t * btVector3(mins.x(), mins.y(), mins.z()),
								  t * btVector3(maxs.x(), mins.y(), mins.z()),
								  t * btVector3(maxs.x(), maxs.y(), mins.z()),
								  t * btVector3(mins.x(), maxs.y(), mins.z()),
								  t * btVector3(mins.x(), mins.y(), maxs.z()),
								  t * btVector3(maxs.x(), mins.y(), maxs.z()),
								  t * btVector3(maxs.x(), maxs.y(), maxs.z()),
								  t * btVector3(mins.x(), maxs.y(), maxs.z())};
		aabbMin = aabbMax = crns[0];
		for (int i = 1; i < 8; ++i)
		{
			aabbMin.setMin(crns[i]);
			aabbMax.setMax(crns[i]);
		}
	}

	virtual void setLocalScaling(const btVector3& /*scaling*/)
	{
		///na
	}
	virtual const btVector3& getLocalScaling() const
	{
		static const btVector3 dummy(1, 1, 1);
		return dummy;
	}
	virtual void calculateLocalInertia(btScalar /*mass*/, btVector3& /*inertia*/) const
	{
		///not yet
		btAssert(0);
	}
	virtual const char* getName() const
	{
		return "SoftBody";
	}
};

//
// btSoftClusterCollisionShape
//
class btSoftClusterCollisionShape : public btConvexInternalShape
{
public:
	const btSoftBody::Cluster* m_cluster;

	btSoftClusterCollisionShape(const btSoftBody::Cluster* cluster) : m_cluster(cluster) { setMargin(0); }

	virtual btVector3 localGetSupportingVertex(const btVector3& vec) const
	{
		btSoftBody::Node* const* n = &m_cluster->m_nodes[0];
		btScalar d = btDot(vec, n[0]->m_x);
		int j = 0;
		for (int i = 1, ni = m_cluster->m_nodes.size(); i < ni; ++i)
		{
			const btScalar k = btDot(vec, n[i]->m_x);
			if (k > d)
			{
				d = k;
				j = i;
			}
		}
		return (n[j]->m_x);
	}
	virtual btVector3 localGetSupportingVertexWithoutMargin(const btVector3& vec) const
	{
		return (localGetSupportingVertex(vec));
	}
	//notice that the vectors should be unit length
	virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors, btVector3* supportVerticesOut, int numVectors) const
	{
	}

	virtual void calculateLocalInertia(btScalar mass, btVector3& inertia) const
	{
	}

	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const
	{
	}

	virtual int getShapeType() const { return SOFTBODY_SHAPE_PROXYTYPE; }

	//debugging
	virtual const char* getName() const { return "SOFTCLUSTER"; }

	virtual void setMargin(btScalar margin)
	{
		btConvexInternalShape::setMargin(margin);
	}
	virtual btScalar getMargin() const
	{
		return btConvexInternalShape::getMargin();
	}
};

//
// Inline's
//

//
template <typename T>
static inline void ZeroInitialize(T& value)
{
	memset(&value, 0, sizeof(T));
}
//
template <typename T>
static inline bool CompLess(const T& a, const T& b)
{
	return (a < b);
}
//
template <typename T>
static inline bool CompGreater(const T& a, const T& b)
{
	return (a > b);
}
//
template <typename T>
static inline T Lerp(const T& a, const T& b, btScalar t)
{
	return (a + (b - a) * t);
}
//
template <typename T>
static inline T InvLerp(const T& a, const T& b, btScalar t)
{
	return ((b + a * t - b * t) / (a * b));
}
//
static inline btMatrix3x3 Lerp(const btMatrix3x3& a,
							   const btMatrix3x3& b,
							   btScalar t)
{
	btMatrix3x3 r;
	r[0] = Lerp(a[0], b[0], t);
	r[1] = Lerp(a[1], b[1], t);
	r[2] = Lerp(a[2], b[2], t);
	return (r);
}
//
static inline btVector3 Clamp(const btVector3& v, btScalar maxlength)
{
	const btScalar sql = v.length2();
	if (sql > (maxlength * maxlength))
		return ((v * maxlength) / btSqrt(sql));
	else
		return (v);
}
//
template <typename T>
static inline T Clamp(const T& x, const T& l, const T& h)
{
	return (x < l ? l : x > h ? h : x);
}
//
template <typename T>
static inline T Sq(const T& x)
{
	return (x * x);
}
//
template <typename T>
static inline T Cube(const T& x)
{
	return (x * x * x);
}
//
template <typename T>
static inline T Sign(const T& x)
{
	return ((T)(x < 0 ? -1 : +1));
}
//
template <typename T>
static inline bool SameSign(const T& x, const T& y)
{
	return ((x * y) > 0);
}
//
static inline btScalar ClusterMetric(const btVector3& x, const btVector3& y)
{
	const btVector3 d = x - y;
	return (btFabs(d[0]) + btFabs(d[1]) + btFabs(d[2]));
}
//
static inline btMatrix3x3 ScaleAlongAxis(const btVector3& a, btScalar s)
{
	const btScalar xx = a.x() * a.x();
	const btScalar yy = a.y() * a.y();
	const btScalar zz = a.z() * a.z();
	const btScalar xy = a.x() * a.y();
	const btScalar yz = a.y() * a.z();
	const btScalar zx = a.z() * a.x();
	btMatrix3x3 m;
	m[0] = btVector3(1 - xx + xx * s, xy * s - xy, zx * s - zx);
	m[1] = btVector3(xy * s - xy, 1 - yy + yy * s, yz * s - yz);
	m[2] = btVector3(zx * s - zx, yz * s - yz, 1 - zz + zz * s);
	return (m);
}
//
static inline btMatrix3x3 Cross(const btVector3& v)
{
	btMatrix3x3 m;
	m[0] = btVector3(0, -v.z(), +v.y());
	m[1] = btVector3(+v.z(), 0, -v.x());
	m[2] = btVector3(-v.y(), +v.x(), 0);
	return (m);
}
//
static inline btMatrix3x3 Diagonal(btScalar x)
{
	btMatrix3x3 m;
	m[0] = btVector3(x, 0, 0);
	m[1] = btVector3(0, x, 0);
	m[2] = btVector3(0, 0, x);
	return (m);
}

static inline btMatrix3x3 Diagonal(const btVector3& v)
{
	btMatrix3x3 m;
	m[0] = btVector3(v.getX(), 0, 0);
	m[1] = btVector3(0, v.getY(), 0);
	m[2] = btVector3(0, 0, v.getZ());
	return (m);
}

static inline btScalar Dot(const btScalar* a, const btScalar* b, int ndof)
{
	btScalar result = 0;
	for (int i = 0; i < ndof; ++i)
		result += a[i] * b[i];
	return result;
}

static inline btMatrix3x3 OuterProduct(const btScalar* v1, const btScalar* v2, const btScalar* v3,
									   const btScalar* u1, const btScalar* u2, const btScalar* u3, int ndof)
{
	btMatrix3x3 m;
	btScalar a11 = Dot(v1, u1, ndof);
	btScalar a12 = Dot(v1, u2, ndof);
	btScalar a13 = Dot(v1, u3, ndof);

	btScalar a21 = Dot(v2, u1, ndof);
	btScalar a22 = Dot(v2, u2, ndof);
	btScalar a23 = Dot(v2, u3, ndof);

	btScalar a31 = Dot(v3, u1, ndof);
	btScalar a32 = Dot(v3, u2, ndof);
	btScalar a33 = Dot(v3, u3, ndof);
	m[0] = btVector3(a11, a12, a13);
	m[1] = btVector3(a21, a22, a23);
	m[2] = btVector3(a31, a32, a33);
	return (m);
}

static inline btMatrix3x3 OuterProduct(const btVector3& v1, const btVector3& v2)
{
	btMatrix3x3 m;
	btScalar a11 = v1[0] * v2[0];
	btScalar a12 = v1[0] * v2[1];
	btScalar a13 = v1[0] * v2[2];

	btScalar a21 = v1[1] * v2[0];
	btScalar a22 = v1[1] * v2[1];
	btScalar a23 = v1[1] * v2[2];

	btScalar a31 = v1[2] * v2[0];
	btScalar a32 = v1[2] * v2[1];
	btScalar a33 = v1[2] * v2[2];
	m[0] = btVector3(a11, a12, a13);
	m[1] = btVector3(a21, a22, a23);
	m[2] = btVector3(a31, a32, a33);
	return (m);
}

//
static inline btMatrix3x3 Add(const btMatrix3x3& a,
							  const btMatrix3x3& b)
{
	btMatrix3x3 r;
	for (int i = 0; i < 3; ++i) r[i] = a[i] + b[i];
	return (r);
}
//
static inline btMatrix3x3 Sub(const btMatrix3x3& a,
							  const btMatrix3x3& b)
{
	btMatrix3x3 r;
	for (int i = 0; i < 3; ++i) r[i] = a[i] - b[i];
	return (r);
}
//
static inline btMatrix3x3 Mul(const btMatrix3x3& a,
							  btScalar b)
{
	btMatrix3x3 r;
	for (int i = 0; i < 3; ++i) r[i] = a[i] * b;
	return (r);
}
//
static inline void Orthogonalize(btMatrix3x3& m)
{
	m[2] = btCross(m[0], m[1]).normalized();
	m[1] = btCross(m[2], m[0]).normalized();
	m[0] = btCross(m[1], m[2]).normalized();
}
//
static inline btMatrix3x3 MassMatrix(btScalar im, const btMatrix3x3& iwi, const btVector3& r)
{
	const btMatrix3x3 cr = Cross(r);
	return (Sub(Diagonal(im), cr * iwi * cr));
}

//
static inline btMatrix3x3 ImpulseMatrix(btScalar dt,
										btScalar ima,
										btScalar imb,
										const btMatrix3x3& iwi,
										const btVector3& r)
{
	return (Diagonal(1 / dt) * Add(Diagonal(ima), MassMatrix(imb, iwi, r)).inverse());
}

//
static inline btMatrix3x3 ImpulseMatrix(btScalar dt,
										const btMatrix3x3& effective_mass_inv,
										btScalar imb,
										const btMatrix3x3& iwi,
										const btVector3& r)
{
	return (Diagonal(1 / dt) * Add(effective_mass_inv, MassMatrix(imb, iwi, r)).inverse());
	//    btMatrix3x3 iimb = MassMatrix(imb, iwi, r);
	//    if (iimb.determinant() == 0)
	//        return effective_mass_inv.inverse();
	//    return effective_mass_inv.inverse() *  Add(effective_mass_inv.inverse(), iimb.inverse()).inverse() * iimb.inverse();
}

//
static inline btMatrix3x3 ImpulseMatrix(btScalar ima, const btMatrix3x3& iia, const btVector3& ra,
										btScalar imb, const btMatrix3x3& iib, const btVector3& rb)
{
	return (Add(MassMatrix(ima, iia, ra), MassMatrix(imb, iib, rb)).inverse());
}

//
static inline btMatrix3x3 AngularImpulseMatrix(const btMatrix3x3& iia,
											   const btMatrix3x3& iib)
{
	return (Add(iia, iib).inverse());
}

//
static inline btVector3 ProjectOnAxis(const btVector3& v,
									  const btVector3& a)
{
	return (a * btDot(v, a));
}
//
static inline btVector3 ProjectOnPlane(const btVector3& v,
									   const btVector3& a)
{
	return (v - ProjectOnAxis(v, a));
}

//
static inline void ProjectOrigin(const btVector3& a,
								 const btVector3& b,
								 btVector3& prj,
								 btScalar& sqd)
{
	const btVector3 d = b - a;
	const btScalar m2 = d.length2();
	if (m2 > SIMD_EPSILON)
	{
		const btScalar t = Clamp<btScalar>(-btDot(a, d) / m2, 0, 1);
		const btVector3 p = a + d * t;
		const btScalar l2 = p.length2();
		if (l2 < sqd)
		{
			prj = p;
			sqd = l2;
		}
	}
}
//
static inline void ProjectOrigin(const btVector3& a,
								 const btVector3& b,
								 const btVector3& c,
								 btVector3& prj,
								 btScalar& sqd)
{
	const btVector3& q = btCross(b - a, c - a);
	const btScalar m2 = q.length2();
	if (m2 > SIMD_EPSILON)
	{
		const btVector3 n = q / btSqrt(m2);
		const btScalar k = btDot(a, n);
		const btScalar k2 = k * k;
		if (k2 < sqd)
		{
			const btVector3 p = n * k;
			if ((btDot(btCross(a - p, b - p), q) > 0) &&
				(btDot(btCross(b - p, c - p), q) > 0) &&
				(btDot(btCross(c - p, a - p), q) > 0))
			{
				prj = p;
				sqd = k2;
			}
			else
			{
				ProjectOrigin(a, b, prj, sqd);
				ProjectOrigin(b, c, prj, sqd);
				ProjectOrigin(c, a, prj, sqd);
			}
		}
	}
}

//
static inline bool rayIntersectsTriangle(const btVector3& origin, const btVector3& dir, const btVector3& v0, const btVector3& v1, const btVector3& v2, btScalar& t)
{
	btScalar a, f, u, v;

	btVector3 e1 = v1 - v0;
	btVector3 e2 = v2 - v0;
	btVector3 h = dir.cross(e2);
	a = e1.dot(h);

	if (a > -0.00001 && a < 0.00001)
		return (false);

	f = btScalar(1) / a;
	btVector3 s = origin - v0;
	u = f * s.dot(h);

	if (u < 0.0 || u > 1.0)
		return (false);

	btVector3 q = s.cross(e1);
	v = f * dir.dot(q);
	if (v < 0.0 || u + v > 1.0)
		return (false);
	// at this stage we can compute t to find out where
	// the intersection point is on the line
	t = f * e2.dot(q);
	if (t > 0)  // ray intersection
		return (true);
	else  // this means that there is a line intersection
		// but not a ray intersection
		return (false);
}

static inline bool lineIntersectsTriangle(const btVector3& rayStart, const btVector3& rayEnd, const btVector3& p1, const btVector3& p2, const btVector3& p3, btVector3& sect, btVector3& normal)
{
	btVector3 dir = rayEnd - rayStart;
	btScalar dir_norm = dir.norm();
	if (dir_norm < SIMD_EPSILON)
		return false;
	dir.normalize();
	btScalar t;
	bool ret = rayIntersectsTriangle(rayStart, dir, p1, p2, p3, t);

	if (ret)
	{
		if (t <= dir_norm)
		{
			sect = rayStart + dir * t;
		}
		else
		{
			ret = false;
		}
	}

	if (ret)
	{
		btVector3 n = (p3 - p1).cross(p2 - p1);
		n.safeNormalize();
		if (n.dot(dir) < 0)
			normal = n;
		else
			normal = -n;
	}
	return ret;
}

//
template <typename T>
static inline T BaryEval(const T& a,
						 const T& b,
						 const T& c,
						 const btVector3& coord)
{
	return (a * coord.x() + b * coord.y() + c * coord.z());
}
//
static inline btVector3 BaryCoord(const btVector3& a,
								  const btVector3& b,
								  const btVector3& c,
								  const btVector3& p)
{
	const btScalar w[] = {btCross(a - p, b - p).length(),
						  btCross(b - p, c - p).length(),
						  btCross(c - p, a - p).length()};
	const btScalar isum = 1 / (w[0] + w[1] + w[2]);
	return (btVector3(w[1] * isum, w[2] * isum, w[0] * isum));
}

//
inline static btScalar ImplicitSolve(btSoftBody::ImplicitFn* fn,
									 const btVector3& a,
									 const btVector3& b,
									 const btScalar accuracy,
									 const int maxiterations = 256)
{
	btScalar span[2] = {0, 1};
	btScalar values[2] = {fn->Eval(a), fn->Eval(b)};
	if (values[0] > values[1])
	{
		btSwap(span[0], span[1]);
		btSwap(values[0], values[1]);
	}
	if (values[0] > -accuracy) return (-1);
	if (values[1] < +accuracy) return (-1);
	for (int i = 0; i < maxiterations; ++i)
	{
		const btScalar t = Lerp(span[0], span[1], values[0] / (values[0] - values[1]));
		const btScalar v = fn->Eval(Lerp(a, b, t));
		if ((t <= 0) || (t >= 1)) break;
		if (btFabs(v) < accuracy) return (t);
		if (v < 0)
		{
			span[0] = t;
			values[0] = v;
		}
		else
		{
			span[1] = t;
			values[1] = v;
		}
	}
	return (-1);
}

inline static void EvaluateMedium(const btSoftBodyWorldInfo* wfi,
								  const btVector3& x,
								  btSoftBody::sMedium& medium)
{
	medium.m_velocity = btVector3(0, 0, 0);
	medium.m_pressure = 0;
	medium.m_density = wfi->air_density;
	if (wfi->water_density > 0)
	{
		const btScalar depth = -(btDot(x, wfi->water_normal) + wfi->water_offset);
		if (depth > 0)
		{
			medium.m_density = wfi->water_density;
			medium.m_pressure = depth * wfi->water_density * wfi->m_gravity.length();
		}
	}
}

//
static inline btVector3 NormalizeAny(const btVector3& v)
{
	const btScalar l = v.length();
	if (l > SIMD_EPSILON)
		return (v / l);
	else
		return (btVector3(0, 0, 0));
}

//
static inline btDbvtVolume VolumeOf(const btSoftBody::Face& f,
									btScalar margin)
{
	const btVector3* pts[] = {&f.m_n[0]->m_x,
							  &f.m_n[1]->m_x,
							  &f.m_n[2]->m_x};
	btDbvtVolume vol = btDbvtVolume::FromPoints(pts, 3);
	vol.Expand(btVector3(margin, margin, margin));
	return (vol);
}

//
static inline btVector3 CenterOf(const btSoftBody::Face& f)
{
	return ((f.m_n[0]->m_x + f.m_n[1]->m_x + f.m_n[2]->m_x) / 3);
}

//
static inline btScalar AreaOf(const btVector3& x0,
							  const btVector3& x1,
							  const btVector3& x2)
{
	const btVector3 a = x1 - x0;
	const btVector3 b = x2 - x0;
	const btVector3 cr = btCross(a, b);
	const btScalar area = cr.length();
	return (area);
}

//
static inline btScalar VolumeOf(const btVector3& x0,
								const btVector3& x1,
								const btVector3& x2,
								const btVector3& x3)
{
	const btVector3 a = x1 - x0;
	const btVector3 b = x2 - x0;
	const btVector3 c = x3 - x0;
	return (btDot(a, btCross(b, c)));
}

//

//
static inline void ApplyClampedForce(btSoftBody::Node& n,
									 const btVector3& f,
									 btScalar dt)
{
	const btScalar dtim = dt * n.m_im;
	if ((f * dtim).length2() > n.m_v.length2())
	{ /* Clamp	*/
		n.m_f -= ProjectOnAxis(n.m_v, f.normalized()) / dtim;
	}
	else
	{ /* Apply	*/
		n.m_f += f;
	}
}

//
static inline int MatchEdge(const btSoftBody::Node* a,
							const btSoftBody::Node* b,
							const btSoftBody::Node* ma,
							const btSoftBody::Node* mb)
{
	if ((a == ma) && (b == mb)) return (0);
	if ((a == mb) && (b == ma)) return (1);
	return (-1);
}

//
// btEigen : Extract eigen system,
// straitforward implementation of http://math.fullerton.edu/mathews/n2003/JacobiMethodMod.html
// outputs are NOT sorted.
//
struct btEigen
{
	static int system(btMatrix3x3& a, btMatrix3x3* vectors, btVector3* values = 0)
	{
		static const int maxiterations = 16;
		static const btScalar accuracy = (btScalar)0.0001;
		btMatrix3x3& v = *vectors;
		int iterations = 0;
		vectors->setIdentity();
		do
		{
			int p = 0, q = 1;
			if (btFabs(a[p][q]) < btFabs(a[0][2]))
			{
				p = 0;
				q = 2;
			}
			if (btFabs(a[p][q]) < btFabs(a[1][2]))
			{
				p = 1;
				q = 2;
			}
			if (btFabs(a[p][q]) > accuracy)
			{
				const btScalar w = (a[q][q] - a[p][p]) / (2 * a[p][q]);
				const btScalar z = btFabs(w);
				const btScalar t = w / (z * (btSqrt(1 + w * w) + z));
				if (t == t) /* [WARNING] let hope that one does not get thrown aways by some compilers... */
				{
					const btScalar c = 1 / btSqrt(t * t + 1);
					const btScalar s = c * t;
					mulPQ(a, c, s, p, q);
					mulTPQ(a, c, s, p, q);
					mulPQ(v, c, s, p, q);
				}
				else
					break;
			}
			else
				break;
		} while ((++iterations) < maxiterations);
		if (values)
		{
			*values = btVector3(a[0][0], a[1][1], a[2][2]);
		}
		return (iterations);
	}

private:
	static inline void mulTPQ(btMatrix3x3& a, btScalar c, btScalar s, int p, int q)
	{
		const btScalar m[2][3] = {{a[p][0], a[p][1], a[p][2]},
								  {a[q][0], a[q][1], a[q][2]}};
		int i;

		for (i = 0; i < 3; ++i) a[p][i] = c * m[0][i] - s * m[1][i];
		for (i = 0; i < 3; ++i) a[q][i] = c * m[1][i] + s * m[0][i];
	}
	static inline void mulPQ(btMatrix3x3& a, btScalar c, btScalar s, int p, int q)
	{
		const btScalar m[2][3] = {{a[0][p], a[1][p], a[2][p]},
								  {a[0][q], a[1][q], a[2][q]}};
		int i;

		for (i = 0; i < 3; ++i) a[i][p] = c * m[0][i] - s * m[1][i];
		for (i = 0; i < 3; ++i) a[i][q] = c * m[1][i] + s * m[0][i];
	}
};

//
// Polar decomposition,
// "Computing the Polar Decomposition with Applications", Nicholas J. Higham, 1986.
//
static inline int PolarDecompose(const btMatrix3x3& m, btMatrix3x3& q, btMatrix3x3& s)
{
	static const btPolarDecomposition polar;
	return polar.decompose(m, q, s);
}

//
// btSoftColliders
//
struct btSoftColliders
{
	//
	// ClusterBase
	//
	struct ClusterBase : btDbvt::ICollide
	{
		btScalar erp;
		btScalar idt;
		btScalar m_margin;
		btScalar friction;
		btScalar threshold;
		ClusterBase()
		{
			erp = (btScalar)1;
			idt = 0;
			m_margin = 0;
			friction = 0;
			threshold = (btScalar)0;
		}
		bool SolveContact(const btGjkEpaSolver2::sResults& res,
						  btSoftBody::Body ba, const btSoftBody::Body bb,
						  btSoftBody::CJoint& joint)
		{
			if (res.distance < m_margin)
			{
				btVector3 norm = res.normal;
				norm.normalize();  //is it necessary?

				const btVector3 ra = res.witnesses[0] - ba.xform().getOrigin();
				const btVector3 rb = res.witnesses[1] - bb.xform().getOrigin();
				const btVector3 va = ba.velocity(ra);
				const btVector3 vb = bb.velocity(rb);
				const btVector3 vrel = va - vb;
				const btScalar rvac = btDot(vrel, norm);
				btScalar depth = res.distance - m_margin;

				//				printf("depth=%f\n",depth);
				const btVector3 iv = norm * rvac;
				const btVector3 fv = vrel - iv;
				joint.m_bodies[0] = ba;
				joint.m_bodies[1] = bb;
				joint.m_refs[0] = ra * ba.xform().getBasis();
				joint.m_refs[1] = rb * bb.xform().getBasis();
				joint.m_rpos[0] = ra;
				joint.m_rpos[1] = rb;
				joint.m_cfm = 1;
				joint.m_erp = 1;
				joint.m_life = 0;
				joint.m_maxlife = 0;
				joint.m_split = 1;

				joint.m_drift = depth * norm;

				joint.m_normal = norm;
				//				printf("normal=%f,%f,%f\n",res.normal.getX(),res.normal.getY(),res.normal.getZ());
				joint.m_delete = false;
				joint.m_friction = fv.length2() < (rvac * friction * rvac * friction) ? 1 : friction;
				joint.m_massmatrix = ImpulseMatrix(ba.invMass(), ba.invWorldInertia(), joint.m_rpos[0],
												   bb.invMass(), bb.invWorldInertia(), joint.m_rpos[1]);

				return (true);
			}
			return (false);
		}
	};
	//
	// CollideCL_RS
	//
	struct CollideCL_RS : ClusterBase
	{
		btSoftBody* psb;
		const btCollisionObjectWrapper* m_colObjWrap;

		void Process(const btDbvtNode* leaf)
		{
			btSoftBody::Cluster* cluster = (btSoftBody::Cluster*)leaf->data;
			btSoftClusterCollisionShape cshape(cluster);

			const btConvexShape* rshape = (const btConvexShape*)m_colObjWrap->getCollisionShape();

			///don't collide an anchored cluster with a static/kinematic object
			if (m_colObjWrap->getCollisionObject()->isStaticOrKinematicObject() && cluster->m_containsAnchor)
				return;

			btGjkEpaSolver2::sResults res;
			if (btGjkEpaSolver2::SignedDistance(&cshape, btTransform::getIdentity(),
												rshape, m_colObjWrap->getWorldTransform(),
												btVector3(1, 0, 0), res))
			{
				btSoftBody::CJoint joint;
				if (SolveContact(res, cluster, m_colObjWrap->getCollisionObject(), joint))  //prb,joint))
				{
					btSoftBody::CJoint* pj = new (btAlignedAlloc(sizeof(btSoftBody::CJoint), 16)) btSoftBody::CJoint();
					*pj = joint;
					psb->m_joints.push_back(pj);
					if (m_colObjWrap->getCollisionObject()->isStaticOrKinematicObject())
					{
						pj->m_erp *= psb->m_cfg.kSKHR_CL;
						pj->m_split *= psb->m_cfg.kSK_SPLT_CL;
					}
					else
					{
						pj->m_erp *= psb->m_cfg.kSRHR_CL;
						pj->m_split *= psb->m_cfg.kSR_SPLT_CL;
					}
				}
			}
		}
		void ProcessColObj(btSoftBody* ps, const btCollisionObjectWrapper* colObWrap)
		{
			psb = ps;
			m_colObjWrap = colObWrap;
			idt = ps->m_sst.isdt;
			m_margin = m_colObjWrap->getCollisionShape()->getMargin() + psb->getCollisionShape()->getMargin();
			///Bullet rigid body uses multiply instead of minimum to determine combined friction. Some customization would be useful.
			friction = btMin(psb->m_cfg.kDF, m_colObjWrap->getCollisionObject()->getFriction());
			btVector3 mins;
			btVector3 maxs;

			ATTRIBUTE_ALIGNED16(btDbvtVolume)
			volume;
			colObWrap->getCollisionShape()->getAabb(colObWrap->getWorldTransform(), mins, maxs);
			volume = btDbvtVolume::FromMM(mins, maxs);
			volume.Expand(btVector3(1, 1, 1) * m_margin);
			ps->m_cdbvt.collideTV(ps->m_cdbvt.m_root, volume, *this);
		}
	};
	//
	// CollideCL_SS
	//
	struct CollideCL_SS : ClusterBase
	{
		btSoftBody* bodies[2];
		void Process(const btDbvtNode* la, const btDbvtNode* lb)
		{
			btSoftBody::Cluster* cla = (btSoftBody::Cluster*)la->data;
			btSoftBody::Cluster* clb = (btSoftBody::Cluster*)lb->data;

			bool connected = false;
			if ((bodies[0] == bodies[1]) && (bodies[0]->m_clusterConnectivity.size()))
			{
				connected = bodies[0]->m_clusterConnectivity[cla->m_clusterIndex + bodies[0]->m_clusters.size() * clb->m_clusterIndex];
			}

			if (!connected)
			{
				btSoftClusterCollisionShape csa(cla);
				btSoftClusterCollisionShape csb(clb);
				btGjkEpaSolver2::sResults res;
				if (btGjkEpaSolver2::SignedDistance(&csa, btTransform::getIdentity(),
													&csb, btTransform::getIdentity(),
													cla->m_com - clb->m_com, res))
				{
					btSoftBody::CJoint joint;
					if (SolveContact(res, cla, clb, joint))
					{
						btSoftBody::CJoint* pj = new (btAlignedAlloc(sizeof(btSoftBody::CJoint), 16)) btSoftBody::CJoint();
						*pj = joint;
						bodies[0]->m_joints.push_back(pj);
						pj->m_erp *= btMax(bodies[0]->m_cfg.kSSHR_CL, bodies[1]->m_cfg.kSSHR_CL);
						pj->m_split *= (bodies[0]->m_cfg.kSS_SPLT_CL + bodies[1]->m_cfg.kSS_SPLT_CL) / 2;
					}
				}
			}
			else
			{
				static int count = 0;
				count++;
				//printf("count=%d\n",count);
			}
		}
		void ProcessSoftSoft(btSoftBody* psa, btSoftBody* psb)
		{
			idt = psa->m_sst.isdt;
			//m_margin		=	(psa->getCollisionShape()->getMargin()+psb->getCollisionShape()->getMargin())/2;
			m_margin = (psa->getCollisionShape()->getMargin() + psb->getCollisionShape()->getMargin());
			friction = btMin(psa->m_cfg.kDF, psb->m_cfg.kDF);
			bodies[0] = psa;
			bodies[1] = psb;
			psa->m_cdbvt.collideTT(psa->m_cdbvt.m_root, psb->m_cdbvt.m_root, *this);
		}
	};
	//
	// CollideSDF_RS
	//
	struct CollideSDF_RS : btDbvt::ICollide
	{
		void Process(const btDbvtNode* leaf)
		{
			btSoftBody::Node* node = (btSoftBody::Node*)leaf->data;
			DoNode(*node);
		}
		void DoNode(btSoftBody::Node& n) const
		{
			const btScalar m = n.m_im > 0 ? dynmargin : stamargin;
			btSoftBody::RContact c;

			if ((!n.m_battach) &&
				psb->checkContact(m_colObj1Wrap, n.m_x, m, c.m_cti))
			{
				const btScalar ima = n.m_im;
				const btScalar imb = m_rigidBody ? m_rigidBody->getInvMass() : 0.f;
				const btScalar ms = ima + imb;
				if (ms > 0)
				{
					const btTransform& wtr = m_rigidBody ? m_rigidBody->getWorldTransform() : m_colObj1Wrap->getCollisionObject()->getWorldTransform();
					static const btMatrix3x3 iwiStatic(0, 0, 0, 0, 0, 0, 0, 0, 0);
					const btMatrix3x3& iwi = m_rigidBody ? m_rigidBody->getInvInertiaTensorWorld() : iwiStatic;
					const btVector3 ra = n.m_x - wtr.getOrigin();
					const btVector3 va = m_rigidBody ? m_rigidBody->getVelocityInLocalPoint(ra) * psb->m_sst.sdt : btVector3(0, 0, 0);
					const btVector3 vb = n.m_x - n.m_q;
					const btVector3 vr = vb - va;
					const btScalar dn = btDot(vr, c.m_cti.m_normal);
					const btVector3 fv = vr - c.m_cti.m_normal * dn;
					const btScalar fc = psb->m_cfg.kDF * m_colObj1Wrap->getCollisionObject()->getFriction();
					c.m_node = &n;
					c.m_c0 = ImpulseMatrix(psb->m_sst.sdt, ima, imb, iwi, ra);
					c.m_c1 = ra;
					c.m_c2 = ima * psb->m_sst.sdt;
					c.m_c3 = fv.length2() < (dn * fc * dn * fc) ? 0 : 1 - fc;
					c.m_c4 = m_colObj1Wrap->getCollisionObject()->isStaticOrKinematicObject() ? psb->m_cfg.kKHR : psb->m_cfg.kCHR;
					psb->m_rcontacts.push_back(c);
					if (m_rigidBody)
						m_rigidBody->activate();
				}
			}
		}
		btSoftBody* psb;
		const btCollisionObjectWrapper* m_colObj1Wrap;
		btRigidBody* m_rigidBody;
		btScalar dynmargin;
		btScalar stamargin;
	};

	//
	// CollideSDF_RD
	//
	struct CollideSDF_RD : btDbvt::ICollide
	{
		void Process(const btDbvtNode* leaf)
		{
			btSoftBody::Node* node = (btSoftBody::Node*)leaf->data;
			DoNode(*node);
		}
		void DoNode(btSoftBody::Node& n) const
		{
			const btScalar m = n.m_im > 0 ? dynmargin : stamargin;
			btSoftBody::DeformableNodeRigidContact c;

			if (!n.m_battach)
			{
				// check for collision at x_{n+1}^*
				if (psb->checkDeformableContact(m_colObj1Wrap, n.m_q, m, c.m_cti, /*predict = */ true))
				{
					const btScalar ima = n.m_im;
					// todo: collision between multibody and fixed deformable node will be missed.
					const btScalar imb = m_rigidBody ? m_rigidBody->getInvMass() : 0.f;
					const btScalar ms = ima + imb;
					if (ms > 0)
					{
						// resolve contact at x_n
						psb->checkDeformableContact(m_colObj1Wrap, n.m_x, m, c.m_cti, /*predict = */ false);
						btSoftBody::sCti& cti = c.m_cti;
						c.m_node = &n;
						const btScalar fc = psb->m_cfg.kDF * m_colObj1Wrap->getCollisionObject()->getFriction();
						c.m_c2 = ima;
						c.m_c3 = fc;
						c.m_c4 = m_colObj1Wrap->getCollisionObject()->isStaticOrKinematicObject() ? psb->m_cfg.kKHR : psb->m_cfg.kCHR;
						c.m_c5 = n.m_effectiveMass_inv;

						if (cti.m_colObj->getInternalType() == btCollisionObject::CO_RIGID_BODY)
						{
							const btTransform& wtr = m_rigidBody ? m_rigidBody->getWorldTransform() : m_colObj1Wrap->getCollisionObject()->getWorldTransform();
							const btVector3 ra = n.m_x - wtr.getOrigin();

							static const btMatrix3x3 iwiStatic(0, 0, 0, 0, 0, 0, 0, 0, 0);
							const btMatrix3x3& iwi = m_rigidBody ? m_rigidBody->getInvInertiaTensorWorld() : iwiStatic;
							if (psb->m_reducedModel)
							{
								c.m_c0 = MassMatrix(imb, iwi, ra); //impulse factor K of the rigid body only (not the inverse)
							}
							else
							{
								c.m_c0 = ImpulseMatrix(1, n.m_effectiveMass_inv, imb, iwi, ra);
								//                            c.m_c0 = ImpulseMatrix(1, ima, imb, iwi, ra);
							}
							c.m_c1 = ra;
						}
						else if (cti.m_colObj->getInternalType() == btCollisionObject::CO_FEATHERSTONE_LINK)
						{
							btMultiBodyLinkCollider* multibodyLinkCol = (btMultiBodyLinkCollider*)btMultiBodyLinkCollider::upcast(cti.m_colObj);
							if (multibodyLinkCol)
							{
								btVector3 normal = cti.m_normal;
								btVector3 t1 = generateUnitOrthogonalVector(normal);
								btVector3 t2 = btCross(normal, t1);
								btMultiBodyJacobianData jacobianData_normal, jacobianData_t1, jacobianData_t2;
								findJacobian(multibodyLinkCol, jacobianData_normal, c.m_node->m_x, normal);
								findJacobian(multibodyLinkCol, jacobianData_t1, c.m_node->m_x, t1);
								findJacobian(multibodyLinkCol, jacobianData_t2, c.m_node->m_x, t2);

								btScalar* J_n = &jacobianData_normal.m_jacobians[0];
								btScalar* J_t1 = &jacobianData_t1.m_jacobians[0];
								btScalar* J_t2 = &jacobianData_t2.m_jacobians[0];

								btScalar* u_n = &jacobianData_normal.m_deltaVelocitiesUnitImpulse[0];
								btScalar* u_t1 = &jacobianData_t1.m_deltaVelocitiesUnitImpulse[0];
								btScalar* u_t2 = &jacobianData_t2.m_deltaVelocitiesUnitImpulse[0];

								btMatrix3x3 rot(normal.getX(), normal.getY(), normal.getZ(),
												t1.getX(), t1.getY(), t1.getZ(),
												t2.getX(), t2.getY(), t2.getZ());  // world frame to local frame
								const int ndof = multibodyLinkCol->m_multiBody->getNumDofs() + 6;
								
								btMatrix3x3 local_impulse_matrix;
								if (psb->m_reducedModel)
								{
									local_impulse_matrix = OuterProduct(J_n, J_t1, J_t2, u_n, u_t1, u_t2, ndof);
								}
								else
								{
									local_impulse_matrix = (n.m_effectiveMass_inv + OuterProduct(J_n, J_t1, J_t2, u_n, u_t1, u_t2, ndof)).inverse();
								}
								c.m_c0 = rot.transpose() * local_impulse_matrix * rot;
								c.jacobianData_normal = jacobianData_normal;
								c.jacobianData_t1 = jacobianData_t1;
								c.jacobianData_t2 = jacobianData_t2;
								c.t1 = t1;
								c.t2 = t2;
							}
						}
						psb->m_nodeRigidContacts.push_back(c);
					}
				}
			}
		}
		btSoftBody* psb;
		const btCollisionObjectWrapper* m_colObj1Wrap;
		btRigidBody* m_rigidBody;
		btScalar dynmargin;
		btScalar stamargin;
	};

	//
	// CollideSDF_RDF
	//
	struct CollideSDF_RDF : btDbvt::ICollide
	{
		void Process(const btDbvtNode* leaf)
		{
			btSoftBody::Face* face = (btSoftBody::Face*)leaf->data;
			DoNode(*face);
		}
		void DoNode(btSoftBody::Face& f) const
		{
			btSoftBody::Node* n0 = f.m_n[0];
			btSoftBody::Node* n1 = f.m_n[1];
			btSoftBody::Node* n2 = f.m_n[2];
			const btScalar m = (n0->m_im > 0 && n1->m_im > 0 && n2->m_im > 0) ? dynmargin : stamargin;
			btSoftBody::DeformableFaceRigidContact c;
			btVector3 contact_point;
			btVector3 bary;
			if (psb->checkDeformableFaceContact(m_colObj1Wrap, f, contact_point, bary, m, c.m_cti, true))
			{
				btScalar ima = n0->m_im + n1->m_im + n2->m_im;
				const btScalar imb = m_rigidBody ? m_rigidBody->getInvMass() : 0.f;
				// todo: collision between multibody and fixed deformable face will be missed.
				const btScalar ms = ima + imb;
				if (ms > 0)
				{
					// resolve contact at x_n
					//                    psb->checkDeformableFaceContact(m_colObj1Wrap, f, contact_point, bary, m, c.m_cti, /*predict = */ false);
					btSoftBody::sCti& cti = c.m_cti;
					c.m_contactPoint = contact_point;
					c.m_bary = bary;
					// todo xuchenhan@: this is assuming mass of all vertices are the same. Need to modify if mass are different for distinct vertices
					c.m_weights = btScalar(2) / (btScalar(1) + bary.length2()) * bary;
					c.m_face = &f;
					// friction is handled by the nodes to prevent sticking
					//                    const btScalar fc = 0;
					const btScalar fc = psb->m_cfg.kDF * m_colObj1Wrap->getCollisionObject()->getFriction();

					// the effective inverse mass of the face as in https://graphics.stanford.edu/papers/cloth-sig02/cloth.pdf
					ima = bary.getX() * c.m_weights.getX() * n0->m_im + bary.getY() * c.m_weights.getY() * n1->m_im + bary.getZ() * c.m_weights.getZ() * n2->m_im;
					c.m_c2 = ima;
					c.m_c3 = fc;
					c.m_c4 = m_colObj1Wrap->getCollisionObject()->isStaticOrKinematicObject() ? psb->m_cfg.kKHR : psb->m_cfg.kCHR;
					c.m_c5 = Diagonal(ima);
					if (cti.m_colObj->getInternalType() == btCollisionObject::CO_RIGID_BODY)
					{
						const btTransform& wtr = m_rigidBody ? m_rigidBody->getWorldTransform() : m_colObj1Wrap->getCollisionObject()->getWorldTransform();
						static const btMatrix3x3 iwiStatic(0, 0, 0, 0, 0, 0, 0, 0, 0);
						const btMatrix3x3& iwi = m_rigidBody ? m_rigidBody->getInvInertiaTensorWorld() : iwiStatic;
						const btVector3 ra = contact_point - wtr.getOrigin();

						// we do not scale the impulse matrix by dt
						c.m_c0 = ImpulseMatrix(1, ima, imb, iwi, ra);
						c.m_c1 = ra;
					}
					else if (cti.m_colObj->getInternalType() == btCollisionObject::CO_FEATHERSTONE_LINK)
					{
						btMultiBodyLinkCollider* multibodyLinkCol = (btMultiBodyLinkCollider*)btMultiBodyLinkCollider::upcast(cti.m_colObj);
						if (multibodyLinkCol)
						{
							btVector3 normal = cti.m_normal;
							btVector3 t1 = generateUnitOrthogonalVector(normal);
							btVector3 t2 = btCross(normal, t1);
							btMultiBodyJacobianData jacobianData_normal, jacobianData_t1, jacobianData_t2;
							findJacobian(multibodyLinkCol, jacobianData_normal, contact_point, normal);
							findJacobian(multibodyLinkCol, jacobianData_t1, contact_point, t1);
							findJacobian(multibodyLinkCol, jacobianData_t2, contact_point, t2);

							btScalar* J_n = &jacobianData_normal.m_jacobians[0];
							btScalar* J_t1 = &jacobianData_t1.m_jacobians[0];
							btScalar* J_t2 = &jacobianData_t2.m_jacobians[0];

							btScalar* u_n = &jacobianData_normal.m_deltaVelocitiesUnitImpulse[0];
							btScalar* u_t1 = &jacobianData_t1.m_deltaVelocitiesUnitImpulse[0];
							btScalar* u_t2 = &jacobianData_t2.m_deltaVelocitiesUnitImpulse[0];

							btMatrix3x3 rot(normal.getX(), normal.getY(), normal.getZ(),
											t1.getX(), t1.getY(), t1.getZ(),
											t2.getX(), t2.getY(), t2.getZ());  // world frame to local frame
							const int ndof = multibodyLinkCol->m_multiBody->getNumDofs() + 6;
							btMatrix3x3 local_impulse_matrix = (Diagonal(ima) + OuterProduct(J_n, J_t1, J_t2, u_n, u_t1, u_t2, ndof)).inverse();
							c.m_c0 = rot.transpose() * local_impulse_matrix * rot;
							c.jacobianData_normal = jacobianData_normal;
							c.jacobianData_t1 = jacobianData_t1;
							c.jacobianData_t2 = jacobianData_t2;
							c.t1 = t1;
							c.t2 = t2;
						}
					}
					psb->m_faceRigidContacts.push_back(c);
				}
			}
			// Set caching barycenters to be false after collision detection.
			// Only turn on when contact is static.
			f.m_pcontact[3] = 0;
		}
		btSoftBody* psb;
		const btCollisionObjectWrapper* m_colObj1Wrap;
		btRigidBody* m_rigidBody;
		btScalar dynmargin;
		btScalar stamargin;
	};

	//
	// CollideVF_SS
	//
	struct CollideVF_SS : btDbvt::ICollide
	{
		void Process(const btDbvtNode* lnode,
					 const btDbvtNode* lface)
		{
			btSoftBody::Node* node = (btSoftBody::Node*)lnode->data;
			btSoftBody::Face* face = (btSoftBody::Face*)lface->data;
			for (int i = 0; i < 3; ++i)
			{
				if (face->m_n[i] == node)
					continue;
			}

			btVector3 o = node->m_x;
			btVector3 p;
			btScalar d = SIMD_INFINITY;
			ProjectOrigin(face->m_n[0]->m_x - o,
						  face->m_n[1]->m_x - o,
						  face->m_n[2]->m_x - o,
						  p, d);
			const btScalar m = mrg + (o - node->m_q).length() * 2;
			if (d < (m * m))
			{
				const btSoftBody::Node* n[] = {face->m_n[0], face->m_n[1], face->m_n[2]};
				const btVector3 w = BaryCoord(n[0]->m_x, n[1]->m_x, n[2]->m_x, p + o);
				const btScalar ma = node->m_im;
				btScalar mb = BaryEval(n[0]->m_im, n[1]->m_im, n[2]->m_im, w);
				if ((n[0]->m_im <= 0) ||
					(n[1]->m_im <= 0) ||
					(n[2]->m_im <= 0))
				{
					mb = 0;
				}
				const btScalar ms = ma + mb;
				if (ms > 0)
				{
					btSoftBody::SContact c;
					c.m_normal = p / -btSqrt(d);
					c.m_margin = m;
					c.m_node = node;
					c.m_face = face;
					c.m_weights = w;
					c.m_friction = btMax(psb[0]->m_cfg.kDF, psb[1]->m_cfg.kDF);
					c.m_cfm[0] = ma / ms * psb[0]->m_cfg.kSHR;
					c.m_cfm[1] = mb / ms * psb[1]->m_cfg.kSHR;
					psb[0]->m_scontacts.push_back(c);
				}
			}
		}
		btSoftBody* psb[2];
		btScalar mrg;
	};

	//
	// CollideVF_DD
	//
	struct CollideVF_DD : btDbvt::ICollide
	{
		void Process(const btDbvtNode* lnode,
					 const btDbvtNode* lface)
		{
			btSoftBody::Node* node = (btSoftBody::Node*)lnode->data;
			btSoftBody::Face* face = (btSoftBody::Face*)lface->data;
			btVector3 bary;
			if (proximityTest(face->m_n[0]->m_x, face->m_n[1]->m_x, face->m_n[2]->m_x, node->m_x, face->m_normal, mrg, bary))
			{
				const btSoftBody::Node* n[] = {face->m_n[0], face->m_n[1], face->m_n[2]};
				const btVector3 w = bary;
				const btScalar ma = node->m_im;
				btScalar mb = BaryEval(n[0]->m_im, n[1]->m_im, n[2]->m_im, w);
				if ((n[0]->m_im <= 0) ||
					(n[1]->m_im <= 0) ||
					(n[2]->m_im <= 0))
				{
					mb = 0;
				}
				const btScalar ms = ma + mb;
				if (ms > 0)
				{
					btSoftBody::DeformableFaceNodeContact c;
					c.m_normal = face->m_normal;
					if (!useFaceNormal && c.m_normal.dot(node->m_x - face->m_n[2]->m_x) < 0)
						c.m_normal = -face->m_normal;
					c.m_margin = mrg;
					c.m_node = node;
					c.m_face = face;
					c.m_bary = w;
					c.m_friction = psb[0]->m_cfg.kDF * psb[1]->m_cfg.kDF;
					// Initialize unused fields.
					c.m_weights = btVector3(0, 0, 0);
					c.m_imf = 0;
					c.m_c0 = 0;
					c.m_colObj = psb[1];
					psb[0]->m_faceNodeContacts.push_back(c);
				}
			}
		}
		btSoftBody* psb[2];
		btScalar mrg;
		bool useFaceNormal;
	};

	//
	// CollideFF_DD
	//
	struct CollideFF_DD : btDbvt::ICollide
	{
		void Process(const btDbvntNode* lface1,
					 const btDbvntNode* lface2)
		{
			btSoftBody::Face* f1 = (btSoftBody::Face*)lface1->data;
			btSoftBody::Face* f2 = (btSoftBody::Face*)lface2->data;
			if (f1 != f2)
			{
				Repel(f1, f2);
				Repel(f2, f1);
			}
		}
		void Repel(btSoftBody::Face* f1, btSoftBody::Face* f2)
		{
			//#define REPEL_NEIGHBOR 1
#ifndef REPEL_NEIGHBOR
			for (int node_id = 0; node_id < 3; ++node_id)
			{
				btSoftBody::Node* node = f1->m_n[node_id];
				for (int i = 0; i < 3; ++i)
				{
					if (f2->m_n[i] == node)
						return;
				}
			}
#endif
			bool skip = false;
			for (int node_id = 0; node_id < 3; ++node_id)
			{
				btSoftBody::Node* node = f1->m_n[node_id];
#ifdef REPEL_NEIGHBOR
				for (int i = 0; i < 3; ++i)
				{
					if (f2->m_n[i] == node)
					{
						skip = true;
						break;
					}
				}
				if (skip)
				{
					skip = false;
					continue;
				}
#endif
				btSoftBody::Face* face = f2;
				btVector3 bary;
				if (!proximityTest(face->m_n[0]->m_x, face->m_n[1]->m_x, face->m_n[2]->m_x, node->m_x, face->m_normal, mrg, bary))
					continue;
				btSoftBody::DeformableFaceNodeContact c;
				c.m_normal = face->m_normal;
				if (!useFaceNormal && c.m_normal.dot(node->m_x - face->m_n[2]->m_x) < 0)
					c.m_normal = -face->m_normal;
				c.m_margin = mrg;
				c.m_node = node;
				c.m_face = face;
				c.m_bary = bary;
				c.m_friction = psb[0]->m_cfg.kDF * psb[1]->m_cfg.kDF;
				// Initialize unused fields.
				c.m_weights = btVector3(0, 0, 0);
				c.m_imf = 0;
				c.m_c0 = 0;
				c.m_colObj = psb[1];
				psb[0]->m_faceNodeContacts.push_back(c);
			}
		}
		btSoftBody* psb[2];
		btScalar mrg;
		bool useFaceNormal;
	};

	struct CollideCCD : btDbvt::ICollide
	{
		void Process(const btDbvtNode* lnode,
					 const btDbvtNode* lface)
		{
			btSoftBody::Node* node = (btSoftBody::Node*)lnode->data;
			btSoftBody::Face* face = (btSoftBody::Face*)lface->data;
			btVector3 bary;
			if (bernsteinCCD(face, node, dt, SAFE_EPSILON, bary))
			{
				btSoftBody::DeformableFaceNodeContact c;
				c.m_normal = face->m_normal;
				if (!useFaceNormal && c.m_normal.dot(node->m_x - face->m_n[2]->m_x) < 0)
					c.m_normal = -face->m_normal;
				c.m_node = node;
				c.m_face = face;
				c.m_bary = bary;
				c.m_friction = psb[0]->m_cfg.kDF * psb[1]->m_cfg.kDF;
				// Initialize unused fields.
				c.m_weights = btVector3(0, 0, 0);
				c.m_margin = mrg;
				c.m_imf = 0;
				c.m_c0 = 0;
				c.m_colObj = psb[1];
				psb[0]->m_faceNodeContactsCCD.push_back(c);
			}
		}
		void Process(const btDbvntNode* lface1,
					 const btDbvntNode* lface2)
		{
			btSoftBody::Face* f1 = (btSoftBody::Face*)lface1->data;
			btSoftBody::Face* f2 = (btSoftBody::Face*)lface2->data;
			if (f1 != f2)
			{
				Repel(f1, f2);
				Repel(f2, f1);
			}
		}
		void Repel(btSoftBody::Face* f1, btSoftBody::Face* f2)
		{
			//#define REPEL_NEIGHBOR 1
#ifndef REPEL_NEIGHBOR
			for (int node_id = 0; node_id < 3; ++node_id)
			{
				btSoftBody::Node* node = f1->m_n[node_id];
				for (int i = 0; i < 3; ++i)
				{
					if (f2->m_n[i] == node)
						return;
				}
			}
#endif
			bool skip = false;
			for (int node_id = 0; node_id < 3; ++node_id)
			{
				btSoftBody::Node* node = f1->m_n[node_id];
#ifdef REPEL_NEIGHBOR
				for (int i = 0; i < 3; ++i)
				{
					if (f2->m_n[i] == node)
					{
						skip = true;
						break;
					}
				}
				if (skip)
				{
					skip = false;
					continue;
				}
#endif
				btSoftBody::Face* face = f2;
				btVector3 bary;
				if (bernsteinCCD(face, node, dt, SAFE_EPSILON, bary))
				{
					btSoftBody::DeformableFaceNodeContact c;
					c.m_normal = face->m_normal;
					if (!useFaceNormal && c.m_normal.dot(node->m_x - face->m_n[2]->m_x) < 0)
						c.m_normal = -face->m_normal;
					c.m_node = node;
					c.m_face = face;
					c.m_bary = bary;
					c.m_friction = psb[0]->m_cfg.kDF * psb[1]->m_cfg.kDF;
					// Initialize unused fields.
					c.m_weights = btVector3(0, 0, 0);
					c.m_margin = mrg;
					c.m_imf = 0;
					c.m_c0 = 0;
					c.m_colObj = psb[1];
					psb[0]->m_faceNodeContactsCCD.push_back(c);
				}
			}
		}
		btSoftBody* psb[2];
		btScalar dt, mrg;
		bool useFaceNormal;
	};
};
#endif  //_BT_SOFT_BODY_INTERNALS_H
