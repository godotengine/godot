
/***
 * ---------------------------------
 * Copyright (c)2012 Daniel Fiser <danfis@danfis.cz>
 *
 *  This file was ported from mpr.c file, part of libccd.
 *  The Minkoski Portal Refinement implementation was ported 
 *  to OpenCL by Erwin Coumans for the Bullet 3 Physics library.
 *  The original MPR idea and implementation is by Gary Snethen
 *  in XenoCollide, see http://github.com/erwincoumans/xenocollide
 *
 *  Distributed under the OSI-approved BSD License (the "License");
 *  see <http://www.opensource.org/licenses/bsd-license.php>.
 *  This software is distributed WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the License for more information.
 */

///2014 Oct, Erwin Coumans, Use templates to avoid void* casts

#ifndef BT_MPR_PENETRATION_H
#define BT_MPR_PENETRATION_H

#define BT_DEBUG_MPR1

#include "LinearMath/btTransform.h"
#include "LinearMath/btAlignedObjectArray.h"

//#define MPR_AVERAGE_CONTACT_POSITIONS

struct btMprCollisionDescription
{
	btVector3 m_firstDir;
	int m_maxGjkIterations;
	btScalar m_maximumDistanceSquared;
	btScalar m_gjkRelError2;

	btMprCollisionDescription()
		: m_firstDir(0, 1, 0),
		  m_maxGjkIterations(1000),
		  m_maximumDistanceSquared(1e30f),
		  m_gjkRelError2(1.0e-6)
	{
	}
	virtual ~btMprCollisionDescription()
	{
	}
};

struct btMprDistanceInfo
{
	btVector3 m_pointOnA;
	btVector3 m_pointOnB;
	btVector3 m_normalBtoA;
	btScalar m_distance;
};

#ifdef __cplusplus
#define BT_MPR_SQRT sqrtf
#else
#define BT_MPR_SQRT sqrt
#endif
#define BT_MPR_FMIN(x, y) ((x) < (y) ? (x) : (y))
#define BT_MPR_FABS fabs

#define BT_MPR_TOLERANCE 1E-6f
#define BT_MPR_MAX_ITERATIONS 1000

struct _btMprSupport_t
{
	btVector3 v;   //!< Support point in minkowski sum
	btVector3 v1;  //!< Support point in obj1
	btVector3 v2;  //!< Support point in obj2
};
typedef struct _btMprSupport_t btMprSupport_t;

struct _btMprSimplex_t
{
	btMprSupport_t ps[4];
	int last;  //!< index of last added point
};
typedef struct _btMprSimplex_t btMprSimplex_t;

inline btMprSupport_t *btMprSimplexPointW(btMprSimplex_t *s, int idx)
{
	return &s->ps[idx];
}

inline void btMprSimplexSetSize(btMprSimplex_t *s, int size)
{
	s->last = size - 1;
}

#ifdef DEBUG_MPR
inline void btPrintPortalVertex(_btMprSimplex_t *portal, int index)
{
	printf("portal[%d].v = %f,%f,%f, v1=%f,%f,%f, v2=%f,%f,%f\n", index, portal->ps[index].v.x(), portal->ps[index].v.y(), portal->ps[index].v.z(),
		   portal->ps[index].v1.x(), portal->ps[index].v1.y(), portal->ps[index].v1.z(),
		   portal->ps[index].v2.x(), portal->ps[index].v2.y(), portal->ps[index].v2.z());
}
#endif  //DEBUG_MPR

inline int btMprSimplexSize(const btMprSimplex_t *s)
{
	return s->last + 1;
}

inline const btMprSupport_t *btMprSimplexPoint(const btMprSimplex_t *s, int idx)
{
	// here is no check on boundaries
	return &s->ps[idx];
}

inline void btMprSupportCopy(btMprSupport_t *d, const btMprSupport_t *s)
{
	*d = *s;
}

inline void btMprSimplexSet(btMprSimplex_t *s, size_t pos, const btMprSupport_t *a)
{
	btMprSupportCopy(s->ps + pos, a);
}

inline void btMprSimplexSwap(btMprSimplex_t *s, size_t pos1, size_t pos2)
{
	btMprSupport_t supp;

	btMprSupportCopy(&supp, &s->ps[pos1]);
	btMprSupportCopy(&s->ps[pos1], &s->ps[pos2]);
	btMprSupportCopy(&s->ps[pos2], &supp);
}

inline int btMprIsZero(float val)
{
	return BT_MPR_FABS(val) < FLT_EPSILON;
}

inline int btMprEq(float _a, float _b)
{
	float ab;
	float a, b;

	ab = BT_MPR_FABS(_a - _b);
	if (BT_MPR_FABS(ab) < FLT_EPSILON)
		return 1;

	a = BT_MPR_FABS(_a);
	b = BT_MPR_FABS(_b);
	if (b > a)
	{
		return ab < FLT_EPSILON * b;
	}
	else
	{
		return ab < FLT_EPSILON * a;
	}
}

inline int btMprVec3Eq(const btVector3 *a, const btVector3 *b)
{
	return btMprEq((*a).x(), (*b).x()) && btMprEq((*a).y(), (*b).y()) && btMprEq((*a).z(), (*b).z());
}

template <typename btConvexTemplate>
inline void btFindOrigin(const btConvexTemplate &a, const btConvexTemplate &b, const btMprCollisionDescription &colDesc, btMprSupport_t *center)
{
	center->v1 = a.getObjectCenterInWorld();
	center->v2 = b.getObjectCenterInWorld();
	center->v = center->v1 - center->v2;
}

inline void btMprVec3Set(btVector3 *v, float x, float y, float z)
{
	v->setValue(x, y, z);
}

inline void btMprVec3Add(btVector3 *v, const btVector3 *w)
{
	*v += *w;
}

inline void btMprVec3Copy(btVector3 *v, const btVector3 *w)
{
	*v = *w;
}

inline void btMprVec3Scale(btVector3 *d, float k)
{
	*d *= k;
}

inline float btMprVec3Dot(const btVector3 *a, const btVector3 *b)
{
	float dot;

	dot = btDot(*a, *b);
	return dot;
}

inline float btMprVec3Len2(const btVector3 *v)
{
	return btMprVec3Dot(v, v);
}

inline void btMprVec3Normalize(btVector3 *d)
{
	float k = 1.f / BT_MPR_SQRT(btMprVec3Len2(d));
	btMprVec3Scale(d, k);
}

inline void btMprVec3Cross(btVector3 *d, const btVector3 *a, const btVector3 *b)
{
	*d = btCross(*a, *b);
}

inline void btMprVec3Sub2(btVector3 *d, const btVector3 *v, const btVector3 *w)
{
	*d = *v - *w;
}

inline void btPortalDir(const btMprSimplex_t *portal, btVector3 *dir)
{
	btVector3 v2v1, v3v1;

	btMprVec3Sub2(&v2v1, &btMprSimplexPoint(portal, 2)->v,
				  &btMprSimplexPoint(portal, 1)->v);
	btMprVec3Sub2(&v3v1, &btMprSimplexPoint(portal, 3)->v,
				  &btMprSimplexPoint(portal, 1)->v);
	btMprVec3Cross(dir, &v2v1, &v3v1);
	btMprVec3Normalize(dir);
}

inline int portalEncapsulesOrigin(const btMprSimplex_t *portal,
								  const btVector3 *dir)
{
	float dot;
	dot = btMprVec3Dot(dir, &btMprSimplexPoint(portal, 1)->v);
	return btMprIsZero(dot) || dot > 0.f;
}

inline int portalReachTolerance(const btMprSimplex_t *portal,
								const btMprSupport_t *v4,
								const btVector3 *dir)
{
	float dv1, dv2, dv3, dv4;
	float dot1, dot2, dot3;

	// find the smallest dot product of dir and {v1-v4, v2-v4, v3-v4}

	dv1 = btMprVec3Dot(&btMprSimplexPoint(portal, 1)->v, dir);
	dv2 = btMprVec3Dot(&btMprSimplexPoint(portal, 2)->v, dir);
	dv3 = btMprVec3Dot(&btMprSimplexPoint(portal, 3)->v, dir);
	dv4 = btMprVec3Dot(&v4->v, dir);

	dot1 = dv4 - dv1;
	dot2 = dv4 - dv2;
	dot3 = dv4 - dv3;

	dot1 = BT_MPR_FMIN(dot1, dot2);
	dot1 = BT_MPR_FMIN(dot1, dot3);

	return btMprEq(dot1, BT_MPR_TOLERANCE) || dot1 < BT_MPR_TOLERANCE;
}

inline int portalCanEncapsuleOrigin(const btMprSimplex_t *portal,
									const btMprSupport_t *v4,
									const btVector3 *dir)
{
	float dot;
	dot = btMprVec3Dot(&v4->v, dir);
	return btMprIsZero(dot) || dot > 0.f;
}

inline void btExpandPortal(btMprSimplex_t *portal,
						   const btMprSupport_t *v4)
{
	float dot;
	btVector3 v4v0;

	btMprVec3Cross(&v4v0, &v4->v, &btMprSimplexPoint(portal, 0)->v);
	dot = btMprVec3Dot(&btMprSimplexPoint(portal, 1)->v, &v4v0);
	if (dot > 0.f)
	{
		dot = btMprVec3Dot(&btMprSimplexPoint(portal, 2)->v, &v4v0);
		if (dot > 0.f)
		{
			btMprSimplexSet(portal, 1, v4);
		}
		else
		{
			btMprSimplexSet(portal, 3, v4);
		}
	}
	else
	{
		dot = btMprVec3Dot(&btMprSimplexPoint(portal, 3)->v, &v4v0);
		if (dot > 0.f)
		{
			btMprSimplexSet(portal, 2, v4);
		}
		else
		{
			btMprSimplexSet(portal, 1, v4);
		}
	}
}
template <typename btConvexTemplate>
inline void btMprSupport(const btConvexTemplate &a, const btConvexTemplate &b,
						 const btMprCollisionDescription &colDesc,
						 const btVector3 &dir, btMprSupport_t *supp)
{
	btVector3 separatingAxisInA = dir * a.getWorldTransform().getBasis();
	btVector3 separatingAxisInB = -dir * b.getWorldTransform().getBasis();

	btVector3 pInA = a.getLocalSupportWithMargin(separatingAxisInA);
	btVector3 qInB = b.getLocalSupportWithMargin(separatingAxisInB);

	supp->v1 = a.getWorldTransform()(pInA);
	supp->v2 = b.getWorldTransform()(qInB);
	supp->v = supp->v1 - supp->v2;
}

template <typename btConvexTemplate>
static int btDiscoverPortal(const btConvexTemplate &a, const btConvexTemplate &b,
							const btMprCollisionDescription &colDesc,
							btMprSimplex_t *portal)
{
	btVector3 dir, va, vb;
	float dot;
	int cont;

	// vertex 0 is center of portal
	btFindOrigin(a, b, colDesc, btMprSimplexPointW(portal, 0));

	// vertex 0 is center of portal
	btMprSimplexSetSize(portal, 1);

	btVector3 zero = btVector3(0, 0, 0);
	btVector3 *org = &zero;

	if (btMprVec3Eq(&btMprSimplexPoint(portal, 0)->v, org))
	{
		// Portal's center lies on origin (0,0,0) => we know that objects
		// intersect but we would need to know penetration info.
		// So move center little bit...
		btMprVec3Set(&va, FLT_EPSILON * 10.f, 0.f, 0.f);
		btMprVec3Add(&btMprSimplexPointW(portal, 0)->v, &va);
	}

	// vertex 1 = support in direction of origin
	btMprVec3Copy(&dir, &btMprSimplexPoint(portal, 0)->v);
	btMprVec3Scale(&dir, -1.f);
	btMprVec3Normalize(&dir);

	btMprSupport(a, b, colDesc, dir, btMprSimplexPointW(portal, 1));

	btMprSimplexSetSize(portal, 2);

	// test if origin isn't outside of v1
	dot = btMprVec3Dot(&btMprSimplexPoint(portal, 1)->v, &dir);

	if (btMprIsZero(dot) || dot < 0.f)
		return -1;

	// vertex 2
	btMprVec3Cross(&dir, &btMprSimplexPoint(portal, 0)->v,
				   &btMprSimplexPoint(portal, 1)->v);
	if (btMprIsZero(btMprVec3Len2(&dir)))
	{
		if (btMprVec3Eq(&btMprSimplexPoint(portal, 1)->v, org))
		{
			// origin lies on v1
			return 1;
		}
		else
		{
			// origin lies on v0-v1 segment
			return 2;
		}
	}

	btMprVec3Normalize(&dir);
	btMprSupport(a, b, colDesc, dir, btMprSimplexPointW(portal, 2));

	dot = btMprVec3Dot(&btMprSimplexPoint(portal, 2)->v, &dir);
	if (btMprIsZero(dot) || dot < 0.f)
		return -1;

	btMprSimplexSetSize(portal, 3);

	// vertex 3 direction
	btMprVec3Sub2(&va, &btMprSimplexPoint(portal, 1)->v,
				  &btMprSimplexPoint(portal, 0)->v);
	btMprVec3Sub2(&vb, &btMprSimplexPoint(portal, 2)->v,
				  &btMprSimplexPoint(portal, 0)->v);
	btMprVec3Cross(&dir, &va, &vb);
	btMprVec3Normalize(&dir);

	// it is better to form portal faces to be oriented "outside" origin
	dot = btMprVec3Dot(&dir, &btMprSimplexPoint(portal, 0)->v);
	if (dot > 0.f)
	{
		btMprSimplexSwap(portal, 1, 2);
		btMprVec3Scale(&dir, -1.f);
	}

	while (btMprSimplexSize(portal) < 4)
	{
		btMprSupport(a, b, colDesc, dir, btMprSimplexPointW(portal, 3));

		dot = btMprVec3Dot(&btMprSimplexPoint(portal, 3)->v, &dir);
		if (btMprIsZero(dot) || dot < 0.f)
			return -1;

		cont = 0;

		// test if origin is outside (v1, v0, v3) - set v2 as v3 and
		// continue
		btMprVec3Cross(&va, &btMprSimplexPoint(portal, 1)->v,
					   &btMprSimplexPoint(portal, 3)->v);
		dot = btMprVec3Dot(&va, &btMprSimplexPoint(portal, 0)->v);
		if (dot < 0.f && !btMprIsZero(dot))
		{
			btMprSimplexSet(portal, 2, btMprSimplexPoint(portal, 3));
			cont = 1;
		}

		if (!cont)
		{
			// test if origin is outside (v3, v0, v2) - set v1 as v3 and
			// continue
			btMprVec3Cross(&va, &btMprSimplexPoint(portal, 3)->v,
						   &btMprSimplexPoint(portal, 2)->v);
			dot = btMprVec3Dot(&va, &btMprSimplexPoint(portal, 0)->v);
			if (dot < 0.f && !btMprIsZero(dot))
			{
				btMprSimplexSet(portal, 1, btMprSimplexPoint(portal, 3));
				cont = 1;
			}
		}

		if (cont)
		{
			btMprVec3Sub2(&va, &btMprSimplexPoint(portal, 1)->v,
						  &btMprSimplexPoint(portal, 0)->v);
			btMprVec3Sub2(&vb, &btMprSimplexPoint(portal, 2)->v,
						  &btMprSimplexPoint(portal, 0)->v);
			btMprVec3Cross(&dir, &va, &vb);
			btMprVec3Normalize(&dir);
		}
		else
		{
			btMprSimplexSetSize(portal, 4);
		}
	}

	return 0;
}

template <typename btConvexTemplate>
static int btRefinePortal(const btConvexTemplate &a, const btConvexTemplate &b, const btMprCollisionDescription &colDesc,
						  btMprSimplex_t *portal)
{
	btVector3 dir;
	btMprSupport_t v4;

	for (int i = 0; i < BT_MPR_MAX_ITERATIONS; i++)
	//while (1)
	{
		// compute direction outside the portal (from v0 through v1,v2,v3
		// face)
		btPortalDir(portal, &dir);

		// test if origin is inside the portal
		if (portalEncapsulesOrigin(portal, &dir))
			return 0;

		// get next support point

		btMprSupport(a, b, colDesc, dir, &v4);

		// test if v4 can expand portal to contain origin and if portal
		// expanding doesn't reach given tolerance
		if (!portalCanEncapsuleOrigin(portal, &v4, &dir) || portalReachTolerance(portal, &v4, &dir))
		{
			return -1;
		}

		// v1-v2-v3 triangle must be rearranged to face outside Minkowski
		// difference (direction from v0).
		btExpandPortal(portal, &v4);
	}

	return -1;
}

static void btFindPos(const btMprSimplex_t *portal, btVector3 *pos)
{
	btVector3 zero = btVector3(0, 0, 0);
	btVector3 *origin = &zero;

	btVector3 dir;
	size_t i;
	float b[4], sum, inv;
	btVector3 vec, p1, p2;

	btPortalDir(portal, &dir);

	// use barycentric coordinates of tetrahedron to find origin
	btMprVec3Cross(&vec, &btMprSimplexPoint(portal, 1)->v,
				   &btMprSimplexPoint(portal, 2)->v);
	b[0] = btMprVec3Dot(&vec, &btMprSimplexPoint(portal, 3)->v);

	btMprVec3Cross(&vec, &btMprSimplexPoint(portal, 3)->v,
				   &btMprSimplexPoint(portal, 2)->v);
	b[1] = btMprVec3Dot(&vec, &btMprSimplexPoint(portal, 0)->v);

	btMprVec3Cross(&vec, &btMprSimplexPoint(portal, 0)->v,
				   &btMprSimplexPoint(portal, 1)->v);
	b[2] = btMprVec3Dot(&vec, &btMprSimplexPoint(portal, 3)->v);

	btMprVec3Cross(&vec, &btMprSimplexPoint(portal, 2)->v,
				   &btMprSimplexPoint(portal, 1)->v);
	b[3] = btMprVec3Dot(&vec, &btMprSimplexPoint(portal, 0)->v);

	sum = b[0] + b[1] + b[2] + b[3];

	if (btMprIsZero(sum) || sum < 0.f)
	{
		b[0] = 0.f;

		btMprVec3Cross(&vec, &btMprSimplexPoint(portal, 2)->v,
					   &btMprSimplexPoint(portal, 3)->v);
		b[1] = btMprVec3Dot(&vec, &dir);
		btMprVec3Cross(&vec, &btMprSimplexPoint(portal, 3)->v,
					   &btMprSimplexPoint(portal, 1)->v);
		b[2] = btMprVec3Dot(&vec, &dir);
		btMprVec3Cross(&vec, &btMprSimplexPoint(portal, 1)->v,
					   &btMprSimplexPoint(portal, 2)->v);
		b[3] = btMprVec3Dot(&vec, &dir);

		sum = b[1] + b[2] + b[3];
	}

	inv = 1.f / sum;

	btMprVec3Copy(&p1, origin);
	btMprVec3Copy(&p2, origin);
	for (i = 0; i < 4; i++)
	{
		btMprVec3Copy(&vec, &btMprSimplexPoint(portal, i)->v1);
		btMprVec3Scale(&vec, b[i]);
		btMprVec3Add(&p1, &vec);

		btMprVec3Copy(&vec, &btMprSimplexPoint(portal, i)->v2);
		btMprVec3Scale(&vec, b[i]);
		btMprVec3Add(&p2, &vec);
	}
	btMprVec3Scale(&p1, inv);
	btMprVec3Scale(&p2, inv);
#ifdef MPR_AVERAGE_CONTACT_POSITIONS
	btMprVec3Copy(pos, &p1);
	btMprVec3Add(pos, &p2);
	btMprVec3Scale(pos, 0.5);
#else
	btMprVec3Copy(pos, &p2);
#endif  //MPR_AVERAGE_CONTACT_POSITIONS
}

inline float btMprVec3Dist2(const btVector3 *a, const btVector3 *b)
{
	btVector3 ab;
	btMprVec3Sub2(&ab, a, b);
	return btMprVec3Len2(&ab);
}

inline float _btMprVec3PointSegmentDist2(const btVector3 *P,
										 const btVector3 *x0,
										 const btVector3 *b,
										 btVector3 *witness)
{
	// The computation comes from solving equation of segment:
	//      S(t) = x0 + t.d
	//          where - x0 is initial point of segment
	//                - d is direction of segment from x0 (|d| > 0)
	//                - t belongs to <0, 1> interval
	//
	// Than, distance from a segment to some point P can be expressed:
	//      D(t) = |x0 + t.d - P|^2
	//          which is distance from any point on segment. Minimization
	//          of this function brings distance from P to segment.
	// Minimization of D(t) leads to simple quadratic equation that's
	// solving is straightforward.
	//
	// Bonus of this method is witness point for free.

	float dist, t;
	btVector3 d, a;

	// direction of segment
	btMprVec3Sub2(&d, b, x0);

	// precompute vector from P to x0
	btMprVec3Sub2(&a, x0, P);

	t = -1.f * btMprVec3Dot(&a, &d);
	t /= btMprVec3Len2(&d);

	if (t < 0.f || btMprIsZero(t))
	{
		dist = btMprVec3Dist2(x0, P);
		if (witness)
			btMprVec3Copy(witness, x0);
	}
	else if (t > 1.f || btMprEq(t, 1.f))
	{
		dist = btMprVec3Dist2(b, P);
		if (witness)
			btMprVec3Copy(witness, b);
	}
	else
	{
		if (witness)
		{
			btMprVec3Copy(witness, &d);
			btMprVec3Scale(witness, t);
			btMprVec3Add(witness, x0);
			dist = btMprVec3Dist2(witness, P);
		}
		else
		{
			// recycling variables
			btMprVec3Scale(&d, t);
			btMprVec3Add(&d, &a);
			dist = btMprVec3Len2(&d);
		}
	}

	return dist;
}

inline float btMprVec3PointTriDist2(const btVector3 *P,
									const btVector3 *x0, const btVector3 *B,
									const btVector3 *C,
									btVector3 *witness)
{
	// Computation comes from analytic expression for triangle (x0, B, C)
	//      T(s, t) = x0 + s.d1 + t.d2, where d1 = B - x0 and d2 = C - x0 and
	// Then equation for distance is:
	//      D(s, t) = | T(s, t) - P |^2
	// This leads to minimization of quadratic function of two variables.
	// The solution from is taken only if s is between 0 and 1, t is
	// between 0 and 1 and t + s < 1, otherwise distance from segment is
	// computed.

	btVector3 d1, d2, a;
	float u, v, w, p, q, r;
	float s, t, dist, dist2;
	btVector3 witness2;

	btMprVec3Sub2(&d1, B, x0);
	btMprVec3Sub2(&d2, C, x0);
	btMprVec3Sub2(&a, x0, P);

	u = btMprVec3Dot(&a, &a);
	v = btMprVec3Dot(&d1, &d1);
	w = btMprVec3Dot(&d2, &d2);
	p = btMprVec3Dot(&a, &d1);
	q = btMprVec3Dot(&a, &d2);
	r = btMprVec3Dot(&d1, &d2);

	btScalar div = (w * v - r * r);
	if (btMprIsZero(div))
	{
		s = -1;
	}
	else
	{
		s = (q * r - w * p) / div;
		t = (-s * r - q) / w;
	}

	if ((btMprIsZero(s) || s > 0.f) && (btMprEq(s, 1.f) || s < 1.f) && (btMprIsZero(t) || t > 0.f) && (btMprEq(t, 1.f) || t < 1.f) && (btMprEq(t + s, 1.f) || t + s < 1.f))
	{
		if (witness)
		{
			btMprVec3Scale(&d1, s);
			btMprVec3Scale(&d2, t);
			btMprVec3Copy(witness, x0);
			btMprVec3Add(witness, &d1);
			btMprVec3Add(witness, &d2);

			dist = btMprVec3Dist2(witness, P);
		}
		else
		{
			dist = s * s * v;
			dist += t * t * w;
			dist += 2.f * s * t * r;
			dist += 2.f * s * p;
			dist += 2.f * t * q;
			dist += u;
		}
	}
	else
	{
		dist = _btMprVec3PointSegmentDist2(P, x0, B, witness);

		dist2 = _btMprVec3PointSegmentDist2(P, x0, C, &witness2);
		if (dist2 < dist)
		{
			dist = dist2;
			if (witness)
				btMprVec3Copy(witness, &witness2);
		}

		dist2 = _btMprVec3PointSegmentDist2(P, B, C, &witness2);
		if (dist2 < dist)
		{
			dist = dist2;
			if (witness)
				btMprVec3Copy(witness, &witness2);
		}
	}

	return dist;
}

template <typename btConvexTemplate>
static void btFindPenetr(const btConvexTemplate &a, const btConvexTemplate &b,
						 const btMprCollisionDescription &colDesc,
						 btMprSimplex_t *portal,
						 float *depth, btVector3 *pdir, btVector3 *pos)
{
	btVector3 dir;
	btMprSupport_t v4;
	unsigned long iterations;

	btVector3 zero = btVector3(0, 0, 0);
	btVector3 *origin = &zero;

	iterations = 1UL;
	for (int i = 0; i < BT_MPR_MAX_ITERATIONS; i++)
	//while (1)
	{
		// compute portal direction and obtain next support point
		btPortalDir(portal, &dir);

		btMprSupport(a, b, colDesc, dir, &v4);

		// reached tolerance -> find penetration info
		if (portalReachTolerance(portal, &v4, &dir) || iterations == BT_MPR_MAX_ITERATIONS)
		{
			*depth = btMprVec3PointTriDist2(origin, &btMprSimplexPoint(portal, 1)->v, &btMprSimplexPoint(portal, 2)->v, &btMprSimplexPoint(portal, 3)->v, pdir);
			*depth = BT_MPR_SQRT(*depth);

			if (btMprIsZero((*pdir).x()) && btMprIsZero((*pdir).y()) && btMprIsZero((*pdir).z()))
			{
				*pdir = dir;
			}
			btMprVec3Normalize(pdir);

			// barycentric coordinates:
			btFindPos(portal, pos);

			return;
		}

		btExpandPortal(portal, &v4);

		iterations++;
	}
}

static void btFindPenetrTouch(btMprSimplex_t *portal, float *depth, btVector3 *dir, btVector3 *pos)
{
	// Touching contact on portal's v1 - so depth is zero and direction
	// is unimportant and pos can be guessed
	*depth = 0.f;
	btVector3 zero = btVector3(0, 0, 0);
	btVector3 *origin = &zero;

	btMprVec3Copy(dir, origin);
#ifdef MPR_AVERAGE_CONTACT_POSITIONS
	btMprVec3Copy(pos, &btMprSimplexPoint(portal, 1)->v1);
	btMprVec3Add(pos, &btMprSimplexPoint(portal, 1)->v2);
	btMprVec3Scale(pos, 0.5);
#else
	btMprVec3Copy(pos, &btMprSimplexPoint(portal, 1)->v2);
#endif
}

static void btFindPenetrSegment(btMprSimplex_t *portal,
								float *depth, btVector3 *dir, btVector3 *pos)
{
	// Origin lies on v0-v1 segment.
	// Depth is distance to v1, direction also and position must be
	// computed
#ifdef MPR_AVERAGE_CONTACT_POSITIONS
	btMprVec3Copy(pos, &btMprSimplexPoint(portal, 1)->v1);
	btMprVec3Add(pos, &btMprSimplexPoint(portal, 1)->v2);
	btMprVec3Scale(pos, 0.5f);
#else
	btMprVec3Copy(pos, &btMprSimplexPoint(portal, 1)->v2);
#endif  //MPR_AVERAGE_CONTACT_POSITIONS

	btMprVec3Copy(dir, &btMprSimplexPoint(portal, 1)->v);
	*depth = BT_MPR_SQRT(btMprVec3Len2(dir));
	btMprVec3Normalize(dir);
}

template <typename btConvexTemplate>
inline int btMprPenetration(const btConvexTemplate &a, const btConvexTemplate &b,
							const btMprCollisionDescription &colDesc,
							float *depthOut, btVector3 *dirOut, btVector3 *posOut)
{
	btMprSimplex_t portal;

	// Phase 1: Portal discovery
	int result = btDiscoverPortal(a, b, colDesc, &portal);

	//sepAxis[pairIndex] = *pdir;//or -dir?

	switch (result)
	{
		case 0:
		{
			// Phase 2: Portal refinement

			result = btRefinePortal(a, b, colDesc, &portal);
			if (result < 0)
				return -1;

			// Phase 3. Penetration info
			btFindPenetr(a, b, colDesc, &portal, depthOut, dirOut, posOut);

			break;
		}
		case 1:
		{
			// Touching contact on portal's v1.
			btFindPenetrTouch(&portal, depthOut, dirOut, posOut);
			result = 0;
			break;
		}
		case 2:
		{
			btFindPenetrSegment(&portal, depthOut, dirOut, posOut);
			result = 0;
			break;
		}
		default:
		{
			//if (res < 0)
			//{
			// Origin isn't inside portal - no collision.
			result = -1;
			//}
		}
	};

	return result;
};

template <typename btConvexTemplate, typename btMprDistanceTemplate>
inline int btComputeMprPenetration(const btConvexTemplate &a, const btConvexTemplate &b, const btMprCollisionDescription &colDesc, btMprDistanceTemplate *distInfo)
{
	btVector3 dir, pos;
	float depth;

	int res = btMprPenetration(a, b, colDesc, &depth, &dir, &pos);
	if (res == 0)
	{
		distInfo->m_distance = -depth;
		distInfo->m_pointOnB = pos;
		distInfo->m_normalBtoA = -dir;
		distInfo->m_pointOnA = pos - distInfo->m_distance * dir;
		return 0;
	}

	return -1;
}

#endif  //BT_MPR_PENETRATION_H
