
/***
 * ---------------------------------
 * Copyright (c)2012 Daniel Fiser <danfis@danfis.cz>
 *
 *  This file was ported from mpr.c file, part of libccd.
 *  The Minkoski Portal Refinement implementation was ported 
 *  to OpenCL by Erwin Coumans for the Bullet 3 Physics library.
 *  at http://github.com/erwincoumans/bullet3
 *
 *  Distributed under the OSI-approved BSD License (the "License");
 *  see <http://www.opensource.org/licenses/bsd-license.php>.
 *  This software is distributed WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the License for more information.
 */

#ifndef B3_MPR_PENETRATION_H
#define B3_MPR_PENETRATION_H

#include "Bullet3Common/shared/b3PlatformDefinitions.h"
#include "Bullet3Common/shared/b3Float4.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"

#ifdef __cplusplus
#define B3_MPR_SQRT sqrtf
#else
#define B3_MPR_SQRT sqrt
#endif
#define B3_MPR_FMIN(x, y) ((x) < (y) ? (x) : (y))
#define B3_MPR_FABS fabs

#define B3_MPR_TOLERANCE 1E-6f
#define B3_MPR_MAX_ITERATIONS 1000

struct _b3MprSupport_t
{
	b3Float4 v;   //!< Support point in minkowski sum
	b3Float4 v1;  //!< Support point in obj1
	b3Float4 v2;  //!< Support point in obj2
};
typedef struct _b3MprSupport_t b3MprSupport_t;

struct _b3MprSimplex_t
{
	b3MprSupport_t ps[4];
	int last;  //!< index of last added point
};
typedef struct _b3MprSimplex_t b3MprSimplex_t;

inline b3MprSupport_t *b3MprSimplexPointW(b3MprSimplex_t *s, int idx)
{
	return &s->ps[idx];
}

inline void b3MprSimplexSetSize(b3MprSimplex_t *s, int size)
{
	s->last = size - 1;
}

inline int b3MprSimplexSize(const b3MprSimplex_t *s)
{
	return s->last + 1;
}

inline const b3MprSupport_t *b3MprSimplexPoint(const b3MprSimplex_t *s, int idx)
{
	// here is no check on boundaries
	return &s->ps[idx];
}

inline void b3MprSupportCopy(b3MprSupport_t *d, const b3MprSupport_t *s)
{
	*d = *s;
}

inline void b3MprSimplexSet(b3MprSimplex_t *s, size_t pos, const b3MprSupport_t *a)
{
	b3MprSupportCopy(s->ps + pos, a);
}

inline void b3MprSimplexSwap(b3MprSimplex_t *s, size_t pos1, size_t pos2)
{
	b3MprSupport_t supp;

	b3MprSupportCopy(&supp, &s->ps[pos1]);
	b3MprSupportCopy(&s->ps[pos1], &s->ps[pos2]);
	b3MprSupportCopy(&s->ps[pos2], &supp);
}

inline int b3MprIsZero(float val)
{
	return B3_MPR_FABS(val) < FLT_EPSILON;
}

inline int b3MprEq(float _a, float _b)
{
	float ab;
	float a, b;

	ab = B3_MPR_FABS(_a - _b);
	if (B3_MPR_FABS(ab) < FLT_EPSILON)
		return 1;

	a = B3_MPR_FABS(_a);
	b = B3_MPR_FABS(_b);
	if (b > a)
	{
		return ab < FLT_EPSILON * b;
	}
	else
	{
		return ab < FLT_EPSILON * a;
	}
}

inline int b3MprVec3Eq(const b3Float4 *a, const b3Float4 *b)
{
	return b3MprEq((*a).x, (*b).x) && b3MprEq((*a).y, (*b).y) && b3MprEq((*a).z, (*b).z);
}

inline b3Float4 b3LocalGetSupportVertex(b3Float4ConstArg supportVec, __global const b3ConvexPolyhedronData_t *hull, b3ConstArray(b3Float4) verticesA)
{
	b3Float4 supVec = b3MakeFloat4(0, 0, 0, 0);
	float maxDot = -B3_LARGE_FLOAT;

	if (0 < hull->m_numVertices)
	{
		const b3Float4 scaled = supportVec;
		int index = b3MaxDot(scaled, &verticesA[hull->m_vertexOffset], hull->m_numVertices, &maxDot);
		return verticesA[hull->m_vertexOffset + index];
	}

	return supVec;
}

B3_STATIC void b3MprConvexSupport(int pairIndex, int bodyIndex, b3ConstArray(b3RigidBodyData_t) cpuBodyBuf,
								  b3ConstArray(b3ConvexPolyhedronData_t) cpuConvexData,
								  b3ConstArray(b3Collidable_t) cpuCollidables,
								  b3ConstArray(b3Float4) cpuVertices,
								  __global b3Float4 *sepAxis,
								  const b3Float4 *_dir, b3Float4 *outp, int logme)
{
	//dir is in worldspace, move to local space

	b3Float4 pos = cpuBodyBuf[bodyIndex].m_pos;
	b3Quat orn = cpuBodyBuf[bodyIndex].m_quat;

	b3Float4 dir = b3MakeFloat4((*_dir).x, (*_dir).y, (*_dir).z, 0.f);

	const b3Float4 localDir = b3QuatRotate(b3QuatInverse(orn), dir);

	//find local support vertex
	int colIndex = cpuBodyBuf[bodyIndex].m_collidableIdx;

	b3Assert(cpuCollidables[colIndex].m_shapeType == SHAPE_CONVEX_HULL);
	__global const b3ConvexPolyhedronData_t *hull = &cpuConvexData[cpuCollidables[colIndex].m_shapeIndex];

	b3Float4 pInA;
	if (logme)
	{
		//	b3Float4 supVec = b3MakeFloat4(0,0,0,0);
		float maxDot = -B3_LARGE_FLOAT;

		if (0 < hull->m_numVertices)
		{
			const b3Float4 scaled = localDir;
			int index = b3MaxDot(scaled, &cpuVertices[hull->m_vertexOffset], hull->m_numVertices, &maxDot);
			pInA = cpuVertices[hull->m_vertexOffset + index];
		}
	}
	else
	{
		pInA = b3LocalGetSupportVertex(localDir, hull, cpuVertices);
	}

	//move vertex to world space
	*outp = b3TransformPoint(pInA, pos, orn);
}

inline void b3MprSupport(int pairIndex, int bodyIndexA, int bodyIndexB, b3ConstArray(b3RigidBodyData_t) cpuBodyBuf,
						 b3ConstArray(b3ConvexPolyhedronData_t) cpuConvexData,
						 b3ConstArray(b3Collidable_t) cpuCollidables,
						 b3ConstArray(b3Float4) cpuVertices,
						 __global b3Float4 *sepAxis,
						 const b3Float4 *_dir, b3MprSupport_t *supp)
{
	b3Float4 dir;
	dir = *_dir;
	b3MprConvexSupport(pairIndex, bodyIndexA, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &dir, &supp->v1, 0);
	dir = *_dir * -1.f;
	b3MprConvexSupport(pairIndex, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &dir, &supp->v2, 0);
	supp->v = supp->v1 - supp->v2;
}

inline void b3FindOrigin(int bodyIndexA, int bodyIndexB, b3ConstArray(b3RigidBodyData_t) cpuBodyBuf, b3MprSupport_t *center)
{
	center->v1 = cpuBodyBuf[bodyIndexA].m_pos;
	center->v2 = cpuBodyBuf[bodyIndexB].m_pos;
	center->v = center->v1 - center->v2;
}

inline void b3MprVec3Set(b3Float4 *v, float x, float y, float z)
{
	(*v).x = x;
	(*v).y = y;
	(*v).z = z;
	(*v).w = 0.f;
}

inline void b3MprVec3Add(b3Float4 *v, const b3Float4 *w)
{
	(*v).x += (*w).x;
	(*v).y += (*w).y;
	(*v).z += (*w).z;
}

inline void b3MprVec3Copy(b3Float4 *v, const b3Float4 *w)
{
	*v = *w;
}

inline void b3MprVec3Scale(b3Float4 *d, float k)
{
	*d *= k;
}

inline float b3MprVec3Dot(const b3Float4 *a, const b3Float4 *b)
{
	float dot;

	dot = b3Dot3F4(*a, *b);
	return dot;
}

inline float b3MprVec3Len2(const b3Float4 *v)
{
	return b3MprVec3Dot(v, v);
}

inline void b3MprVec3Normalize(b3Float4 *d)
{
	float k = 1.f / B3_MPR_SQRT(b3MprVec3Len2(d));
	b3MprVec3Scale(d, k);
}

inline void b3MprVec3Cross(b3Float4 *d, const b3Float4 *a, const b3Float4 *b)
{
	*d = b3Cross3(*a, *b);
}

inline void b3MprVec3Sub2(b3Float4 *d, const b3Float4 *v, const b3Float4 *w)
{
	*d = *v - *w;
}

inline void b3PortalDir(const b3MprSimplex_t *portal, b3Float4 *dir)
{
	b3Float4 v2v1, v3v1;

	b3MprVec3Sub2(&v2v1, &b3MprSimplexPoint(portal, 2)->v,
				  &b3MprSimplexPoint(portal, 1)->v);
	b3MprVec3Sub2(&v3v1, &b3MprSimplexPoint(portal, 3)->v,
				  &b3MprSimplexPoint(portal, 1)->v);
	b3MprVec3Cross(dir, &v2v1, &v3v1);
	b3MprVec3Normalize(dir);
}

inline int portalEncapsulesOrigin(const b3MprSimplex_t *portal,
								  const b3Float4 *dir)
{
	float dot;
	dot = b3MprVec3Dot(dir, &b3MprSimplexPoint(portal, 1)->v);
	return b3MprIsZero(dot) || dot > 0.f;
}

inline int portalReachTolerance(const b3MprSimplex_t *portal,
								const b3MprSupport_t *v4,
								const b3Float4 *dir)
{
	float dv1, dv2, dv3, dv4;
	float dot1, dot2, dot3;

	// find the smallest dot product of dir and {v1-v4, v2-v4, v3-v4}

	dv1 = b3MprVec3Dot(&b3MprSimplexPoint(portal, 1)->v, dir);
	dv2 = b3MprVec3Dot(&b3MprSimplexPoint(portal, 2)->v, dir);
	dv3 = b3MprVec3Dot(&b3MprSimplexPoint(portal, 3)->v, dir);
	dv4 = b3MprVec3Dot(&v4->v, dir);

	dot1 = dv4 - dv1;
	dot2 = dv4 - dv2;
	dot3 = dv4 - dv3;

	dot1 = B3_MPR_FMIN(dot1, dot2);
	dot1 = B3_MPR_FMIN(dot1, dot3);

	return b3MprEq(dot1, B3_MPR_TOLERANCE) || dot1 < B3_MPR_TOLERANCE;
}

inline int portalCanEncapsuleOrigin(const b3MprSimplex_t *portal,
									const b3MprSupport_t *v4,
									const b3Float4 *dir)
{
	float dot;
	dot = b3MprVec3Dot(&v4->v, dir);
	return b3MprIsZero(dot) || dot > 0.f;
}

inline void b3ExpandPortal(b3MprSimplex_t *portal,
						   const b3MprSupport_t *v4)
{
	float dot;
	b3Float4 v4v0;

	b3MprVec3Cross(&v4v0, &v4->v, &b3MprSimplexPoint(portal, 0)->v);
	dot = b3MprVec3Dot(&b3MprSimplexPoint(portal, 1)->v, &v4v0);
	if (dot > 0.f)
	{
		dot = b3MprVec3Dot(&b3MprSimplexPoint(portal, 2)->v, &v4v0);
		if (dot > 0.f)
		{
			b3MprSimplexSet(portal, 1, v4);
		}
		else
		{
			b3MprSimplexSet(portal, 3, v4);
		}
	}
	else
	{
		dot = b3MprVec3Dot(&b3MprSimplexPoint(portal, 3)->v, &v4v0);
		if (dot > 0.f)
		{
			b3MprSimplexSet(portal, 2, v4);
		}
		else
		{
			b3MprSimplexSet(portal, 1, v4);
		}
	}
}

B3_STATIC int b3DiscoverPortal(int pairIndex, int bodyIndexA, int bodyIndexB, b3ConstArray(b3RigidBodyData_t) cpuBodyBuf,
							   b3ConstArray(b3ConvexPolyhedronData_t) cpuConvexData,
							   b3ConstArray(b3Collidable_t) cpuCollidables,
							   b3ConstArray(b3Float4) cpuVertices,
							   __global b3Float4 *sepAxis,
							   __global int *hasSepAxis,
							   b3MprSimplex_t *portal)
{
	b3Float4 dir, va, vb;
	float dot;
	int cont;

	// vertex 0 is center of portal
	b3FindOrigin(bodyIndexA, bodyIndexB, cpuBodyBuf, b3MprSimplexPointW(portal, 0));
	// vertex 0 is center of portal
	b3MprSimplexSetSize(portal, 1);

	b3Float4 zero = b3MakeFloat4(0, 0, 0, 0);
	b3Float4 *b3mpr_vec3_origin = &zero;

	if (b3MprVec3Eq(&b3MprSimplexPoint(portal, 0)->v, b3mpr_vec3_origin))
	{
		// Portal's center lies on origin (0,0,0) => we know that objects
		// intersect but we would need to know penetration info.
		// So move center little bit...
		b3MprVec3Set(&va, FLT_EPSILON * 10.f, 0.f, 0.f);
		b3MprVec3Add(&b3MprSimplexPointW(portal, 0)->v, &va);
	}

	// vertex 1 = support in direction of origin
	b3MprVec3Copy(&dir, &b3MprSimplexPoint(portal, 0)->v);
	b3MprVec3Scale(&dir, -1.f);
	b3MprVec3Normalize(&dir);

	b3MprSupport(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &dir, b3MprSimplexPointW(portal, 1));

	b3MprSimplexSetSize(portal, 2);

	// test if origin isn't outside of v1
	dot = b3MprVec3Dot(&b3MprSimplexPoint(portal, 1)->v, &dir);

	if (b3MprIsZero(dot) || dot < 0.f)
		return -1;

	// vertex 2
	b3MprVec3Cross(&dir, &b3MprSimplexPoint(portal, 0)->v,
				   &b3MprSimplexPoint(portal, 1)->v);
	if (b3MprIsZero(b3MprVec3Len2(&dir)))
	{
		if (b3MprVec3Eq(&b3MprSimplexPoint(portal, 1)->v, b3mpr_vec3_origin))
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

	b3MprVec3Normalize(&dir);
	b3MprSupport(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &dir, b3MprSimplexPointW(portal, 2));

	dot = b3MprVec3Dot(&b3MprSimplexPoint(portal, 2)->v, &dir);
	if (b3MprIsZero(dot) || dot < 0.f)
		return -1;

	b3MprSimplexSetSize(portal, 3);

	// vertex 3 direction
	b3MprVec3Sub2(&va, &b3MprSimplexPoint(portal, 1)->v,
				  &b3MprSimplexPoint(portal, 0)->v);
	b3MprVec3Sub2(&vb, &b3MprSimplexPoint(portal, 2)->v,
				  &b3MprSimplexPoint(portal, 0)->v);
	b3MprVec3Cross(&dir, &va, &vb);
	b3MprVec3Normalize(&dir);

	// it is better to form portal faces to be oriented "outside" origin
	dot = b3MprVec3Dot(&dir, &b3MprSimplexPoint(portal, 0)->v);
	if (dot > 0.f)
	{
		b3MprSimplexSwap(portal, 1, 2);
		b3MprVec3Scale(&dir, -1.f);
	}

	while (b3MprSimplexSize(portal) < 4)
	{
		b3MprSupport(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &dir, b3MprSimplexPointW(portal, 3));

		dot = b3MprVec3Dot(&b3MprSimplexPoint(portal, 3)->v, &dir);
		if (b3MprIsZero(dot) || dot < 0.f)
			return -1;

		cont = 0;

		// test if origin is outside (v1, v0, v3) - set v2 as v3 and
		// continue
		b3MprVec3Cross(&va, &b3MprSimplexPoint(portal, 1)->v,
					   &b3MprSimplexPoint(portal, 3)->v);
		dot = b3MprVec3Dot(&va, &b3MprSimplexPoint(portal, 0)->v);
		if (dot < 0.f && !b3MprIsZero(dot))
		{
			b3MprSimplexSet(portal, 2, b3MprSimplexPoint(portal, 3));
			cont = 1;
		}

		if (!cont)
		{
			// test if origin is outside (v3, v0, v2) - set v1 as v3 and
			// continue
			b3MprVec3Cross(&va, &b3MprSimplexPoint(portal, 3)->v,
						   &b3MprSimplexPoint(portal, 2)->v);
			dot = b3MprVec3Dot(&va, &b3MprSimplexPoint(portal, 0)->v);
			if (dot < 0.f && !b3MprIsZero(dot))
			{
				b3MprSimplexSet(portal, 1, b3MprSimplexPoint(portal, 3));
				cont = 1;
			}
		}

		if (cont)
		{
			b3MprVec3Sub2(&va, &b3MprSimplexPoint(portal, 1)->v,
						  &b3MprSimplexPoint(portal, 0)->v);
			b3MprVec3Sub2(&vb, &b3MprSimplexPoint(portal, 2)->v,
						  &b3MprSimplexPoint(portal, 0)->v);
			b3MprVec3Cross(&dir, &va, &vb);
			b3MprVec3Normalize(&dir);
		}
		else
		{
			b3MprSimplexSetSize(portal, 4);
		}
	}

	return 0;
}

B3_STATIC int b3RefinePortal(int pairIndex, int bodyIndexA, int bodyIndexB, b3ConstArray(b3RigidBodyData_t) cpuBodyBuf,
							 b3ConstArray(b3ConvexPolyhedronData_t) cpuConvexData,
							 b3ConstArray(b3Collidable_t) cpuCollidables,
							 b3ConstArray(b3Float4) cpuVertices,
							 __global b3Float4 *sepAxis,
							 b3MprSimplex_t *portal)
{
	b3Float4 dir;
	b3MprSupport_t v4;

	for (int i = 0; i < B3_MPR_MAX_ITERATIONS; i++)
	//while (1)
	{
		// compute direction outside the portal (from v0 throught v1,v2,v3
		// face)
		b3PortalDir(portal, &dir);

		// test if origin is inside the portal
		if (portalEncapsulesOrigin(portal, &dir))
			return 0;

		// get next support point

		b3MprSupport(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &dir, &v4);

		// test if v4 can expand portal to contain origin and if portal
		// expanding doesn't reach given tolerance
		if (!portalCanEncapsuleOrigin(portal, &v4, &dir) || portalReachTolerance(portal, &v4, &dir))
		{
			return -1;
		}

		// v1-v2-v3 triangle must be rearranged to face outside Minkowski
		// difference (direction from v0).
		b3ExpandPortal(portal, &v4);
	}

	return -1;
}

B3_STATIC void b3FindPos(const b3MprSimplex_t *portal, b3Float4 *pos)
{
	b3Float4 zero = b3MakeFloat4(0, 0, 0, 0);
	b3Float4 *b3mpr_vec3_origin = &zero;

	b3Float4 dir;
	size_t i;
	float b[4], sum, inv;
	b3Float4 vec, p1, p2;

	b3PortalDir(portal, &dir);

	// use barycentric coordinates of tetrahedron to find origin
	b3MprVec3Cross(&vec, &b3MprSimplexPoint(portal, 1)->v,
				   &b3MprSimplexPoint(portal, 2)->v);
	b[0] = b3MprVec3Dot(&vec, &b3MprSimplexPoint(portal, 3)->v);

	b3MprVec3Cross(&vec, &b3MprSimplexPoint(portal, 3)->v,
				   &b3MprSimplexPoint(portal, 2)->v);
	b[1] = b3MprVec3Dot(&vec, &b3MprSimplexPoint(portal, 0)->v);

	b3MprVec3Cross(&vec, &b3MprSimplexPoint(portal, 0)->v,
				   &b3MprSimplexPoint(portal, 1)->v);
	b[2] = b3MprVec3Dot(&vec, &b3MprSimplexPoint(portal, 3)->v);

	b3MprVec3Cross(&vec, &b3MprSimplexPoint(portal, 2)->v,
				   &b3MprSimplexPoint(portal, 1)->v);
	b[3] = b3MprVec3Dot(&vec, &b3MprSimplexPoint(portal, 0)->v);

	sum = b[0] + b[1] + b[2] + b[3];

	if (b3MprIsZero(sum) || sum < 0.f)
	{
		b[0] = 0.f;

		b3MprVec3Cross(&vec, &b3MprSimplexPoint(portal, 2)->v,
					   &b3MprSimplexPoint(portal, 3)->v);
		b[1] = b3MprVec3Dot(&vec, &dir);
		b3MprVec3Cross(&vec, &b3MprSimplexPoint(portal, 3)->v,
					   &b3MprSimplexPoint(portal, 1)->v);
		b[2] = b3MprVec3Dot(&vec, &dir);
		b3MprVec3Cross(&vec, &b3MprSimplexPoint(portal, 1)->v,
					   &b3MprSimplexPoint(portal, 2)->v);
		b[3] = b3MprVec3Dot(&vec, &dir);

		sum = b[1] + b[2] + b[3];
	}

	inv = 1.f / sum;

	b3MprVec3Copy(&p1, b3mpr_vec3_origin);
	b3MprVec3Copy(&p2, b3mpr_vec3_origin);
	for (i = 0; i < 4; i++)
	{
		b3MprVec3Copy(&vec, &b3MprSimplexPoint(portal, i)->v1);
		b3MprVec3Scale(&vec, b[i]);
		b3MprVec3Add(&p1, &vec);

		b3MprVec3Copy(&vec, &b3MprSimplexPoint(portal, i)->v2);
		b3MprVec3Scale(&vec, b[i]);
		b3MprVec3Add(&p2, &vec);
	}
	b3MprVec3Scale(&p1, inv);
	b3MprVec3Scale(&p2, inv);

	b3MprVec3Copy(pos, &p1);
	b3MprVec3Add(pos, &p2);
	b3MprVec3Scale(pos, 0.5);
}

inline float b3MprVec3Dist2(const b3Float4 *a, const b3Float4 *b)
{
	b3Float4 ab;
	b3MprVec3Sub2(&ab, a, b);
	return b3MprVec3Len2(&ab);
}

inline float _b3MprVec3PointSegmentDist2(const b3Float4 *P,
										 const b3Float4 *x0,
										 const b3Float4 *b,
										 b3Float4 *witness)
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
	b3Float4 d, a;

	// direction of segment
	b3MprVec3Sub2(&d, b, x0);

	// precompute vector from P to x0
	b3MprVec3Sub2(&a, x0, P);

	t = -1.f * b3MprVec3Dot(&a, &d);
	t /= b3MprVec3Len2(&d);

	if (t < 0.f || b3MprIsZero(t))
	{
		dist = b3MprVec3Dist2(x0, P);
		if (witness)
			b3MprVec3Copy(witness, x0);
	}
	else if (t > 1.f || b3MprEq(t, 1.f))
	{
		dist = b3MprVec3Dist2(b, P);
		if (witness)
			b3MprVec3Copy(witness, b);
	}
	else
	{
		if (witness)
		{
			b3MprVec3Copy(witness, &d);
			b3MprVec3Scale(witness, t);
			b3MprVec3Add(witness, x0);
			dist = b3MprVec3Dist2(witness, P);
		}
		else
		{
			// recycling variables
			b3MprVec3Scale(&d, t);
			b3MprVec3Add(&d, &a);
			dist = b3MprVec3Len2(&d);
		}
	}

	return dist;
}

inline float b3MprVec3PointTriDist2(const b3Float4 *P,
									const b3Float4 *x0, const b3Float4 *B,
									const b3Float4 *C,
									b3Float4 *witness)
{
	// Computation comes from analytic expression for triangle (x0, B, C)
	//      T(s, t) = x0 + s.d1 + t.d2, where d1 = B - x0 and d2 = C - x0 and
	// Then equation for distance is:
	//      D(s, t) = | T(s, t) - P |^2
	// This leads to minimization of quadratic function of two variables.
	// The solution from is taken only if s is between 0 and 1, t is
	// between 0 and 1 and t + s < 1, otherwise distance from segment is
	// computed.

	b3Float4 d1, d2, a;
	float u, v, w, p, q, r;
	float s, t, dist, dist2;
	b3Float4 witness2;

	b3MprVec3Sub2(&d1, B, x0);
	b3MprVec3Sub2(&d2, C, x0);
	b3MprVec3Sub2(&a, x0, P);

	u = b3MprVec3Dot(&a, &a);
	v = b3MprVec3Dot(&d1, &d1);
	w = b3MprVec3Dot(&d2, &d2);
	p = b3MprVec3Dot(&a, &d1);
	q = b3MprVec3Dot(&a, &d2);
	r = b3MprVec3Dot(&d1, &d2);

	s = (q * r - w * p) / (w * v - r * r);
	t = (-s * r - q) / w;

	if ((b3MprIsZero(s) || s > 0.f) && (b3MprEq(s, 1.f) || s < 1.f) && (b3MprIsZero(t) || t > 0.f) && (b3MprEq(t, 1.f) || t < 1.f) && (b3MprEq(t + s, 1.f) || t + s < 1.f))
	{
		if (witness)
		{
			b3MprVec3Scale(&d1, s);
			b3MprVec3Scale(&d2, t);
			b3MprVec3Copy(witness, x0);
			b3MprVec3Add(witness, &d1);
			b3MprVec3Add(witness, &d2);

			dist = b3MprVec3Dist2(witness, P);
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
		dist = _b3MprVec3PointSegmentDist2(P, x0, B, witness);

		dist2 = _b3MprVec3PointSegmentDist2(P, x0, C, &witness2);
		if (dist2 < dist)
		{
			dist = dist2;
			if (witness)
				b3MprVec3Copy(witness, &witness2);
		}

		dist2 = _b3MprVec3PointSegmentDist2(P, B, C, &witness2);
		if (dist2 < dist)
		{
			dist = dist2;
			if (witness)
				b3MprVec3Copy(witness, &witness2);
		}
	}

	return dist;
}

B3_STATIC void b3FindPenetr(int pairIndex, int bodyIndexA, int bodyIndexB, b3ConstArray(b3RigidBodyData_t) cpuBodyBuf,
							b3ConstArray(b3ConvexPolyhedronData_t) cpuConvexData,
							b3ConstArray(b3Collidable_t) cpuCollidables,
							b3ConstArray(b3Float4) cpuVertices,
							__global b3Float4 *sepAxis,
							b3MprSimplex_t *portal,
							float *depth, b3Float4 *pdir, b3Float4 *pos)
{
	b3Float4 dir;
	b3MprSupport_t v4;
	unsigned long iterations;

	b3Float4 zero = b3MakeFloat4(0, 0, 0, 0);
	b3Float4 *b3mpr_vec3_origin = &zero;

	iterations = 1UL;
	for (int i = 0; i < B3_MPR_MAX_ITERATIONS; i++)
	//while (1)
	{
		// compute portal direction and obtain next support point
		b3PortalDir(portal, &dir);

		b3MprSupport(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &dir, &v4);

		// reached tolerance -> find penetration info
		if (portalReachTolerance(portal, &v4, &dir) || iterations == B3_MPR_MAX_ITERATIONS)
		{
			*depth = b3MprVec3PointTriDist2(b3mpr_vec3_origin, &b3MprSimplexPoint(portal, 1)->v, &b3MprSimplexPoint(portal, 2)->v, &b3MprSimplexPoint(portal, 3)->v, pdir);
			*depth = B3_MPR_SQRT(*depth);

			if (b3MprIsZero((*pdir).x) && b3MprIsZero((*pdir).y) && b3MprIsZero((*pdir).z))
			{
				*pdir = dir;
			}
			b3MprVec3Normalize(pdir);

			// barycentric coordinates:
			b3FindPos(portal, pos);

			return;
		}

		b3ExpandPortal(portal, &v4);

		iterations++;
	}
}

B3_STATIC void b3FindPenetrTouch(b3MprSimplex_t *portal, float *depth, b3Float4 *dir, b3Float4 *pos)
{
	// Touching contact on portal's v1 - so depth is zero and direction
	// is unimportant and pos can be guessed
	*depth = 0.f;
	b3Float4 zero = b3MakeFloat4(0, 0, 0, 0);
	b3Float4 *b3mpr_vec3_origin = &zero;

	b3MprVec3Copy(dir, b3mpr_vec3_origin);

	b3MprVec3Copy(pos, &b3MprSimplexPoint(portal, 1)->v1);
	b3MprVec3Add(pos, &b3MprSimplexPoint(portal, 1)->v2);
	b3MprVec3Scale(pos, 0.5);
}

B3_STATIC void b3FindPenetrSegment(b3MprSimplex_t *portal,
								   float *depth, b3Float4 *dir, b3Float4 *pos)
{
	// Origin lies on v0-v1 segment.
	// Depth is distance to v1, direction also and position must be
	// computed

	b3MprVec3Copy(pos, &b3MprSimplexPoint(portal, 1)->v1);
	b3MprVec3Add(pos, &b3MprSimplexPoint(portal, 1)->v2);
	b3MprVec3Scale(pos, 0.5f);

	b3MprVec3Copy(dir, &b3MprSimplexPoint(portal, 1)->v);
	*depth = B3_MPR_SQRT(b3MprVec3Len2(dir));
	b3MprVec3Normalize(dir);
}

inline int b3MprPenetration(int pairIndex, int bodyIndexA, int bodyIndexB,
							b3ConstArray(b3RigidBodyData_t) cpuBodyBuf,
							b3ConstArray(b3ConvexPolyhedronData_t) cpuConvexData,
							b3ConstArray(b3Collidable_t) cpuCollidables,
							b3ConstArray(b3Float4) cpuVertices,
							__global b3Float4 *sepAxis,
							__global int *hasSepAxis,
							float *depthOut, b3Float4 *dirOut, b3Float4 *posOut)
{
	b3MprSimplex_t portal;

	//	if (!hasSepAxis[pairIndex])
	//	return -1;

	hasSepAxis[pairIndex] = 0;
	int res;

	// Phase 1: Portal discovery
	res = b3DiscoverPortal(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, hasSepAxis, &portal);

	//sepAxis[pairIndex] = *pdir;//or -dir?

	switch (res)
	{
		case 0:
		{
			// Phase 2: Portal refinement

			res = b3RefinePortal(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &portal);
			if (res < 0)
				return -1;

			// Phase 3. Penetration info
			b3FindPenetr(pairIndex, bodyIndexA, bodyIndexB, cpuBodyBuf, cpuConvexData, cpuCollidables, cpuVertices, sepAxis, &portal, depthOut, dirOut, posOut);
			hasSepAxis[pairIndex] = 1;
			sepAxis[pairIndex] = -*dirOut;
			break;
		}
		case 1:
		{
			// Touching contact on portal's v1.
			b3FindPenetrTouch(&portal, depthOut, dirOut, posOut);
			break;
		}
		case 2:
		{
			b3FindPenetrSegment(&portal, depthOut, dirOut, posOut);
			break;
		}
		default:
		{
			hasSepAxis[pairIndex] = 0;
			//if (res < 0)
			//{
			// Origin isn't inside portal - no collision.
			return -1;
			//}
		}
	};

	return 0;
};

#endif  //B3_MPR_PENETRATION_H
