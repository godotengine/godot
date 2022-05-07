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

#include "btGjkPairDetector.h"
#include "BulletCollision/CollisionShapes/btConvexShape.h"
#include "BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h"
#include "BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h"

#if defined(DEBUG) || defined(_DEBUG)
//#define TEST_NON_VIRTUAL 1
#include <stdio.h>  //for debug printf
#ifdef __SPU__
#include <spu_printf.h>
#define printf spu_printf
#endif  //__SPU__
#endif

//must be above the machine epsilon
#ifdef BT_USE_DOUBLE_PRECISION
#define REL_ERROR2 btScalar(1.0e-12)
btScalar gGjkEpaPenetrationTolerance = 1.0e-12;
#else
#define REL_ERROR2 btScalar(1.0e-6)
btScalar gGjkEpaPenetrationTolerance = 0.001;
#endif


btGjkPairDetector::btGjkPairDetector(const btConvexShape *objectA, const btConvexShape *objectB, btSimplexSolverInterface *simplexSolver, btConvexPenetrationDepthSolver *penetrationDepthSolver)
	: m_cachedSeparatingAxis(btScalar(0.), btScalar(1.), btScalar(0.)),
	  m_penetrationDepthSolver(penetrationDepthSolver),
	  m_simplexSolver(simplexSolver),
	  m_minkowskiA(objectA),
	  m_minkowskiB(objectB),
	  m_shapeTypeA(objectA->getShapeType()),
	  m_shapeTypeB(objectB->getShapeType()),
	  m_marginA(objectA->getMargin()),
	  m_marginB(objectB->getMargin()),
	  m_ignoreMargin(false),
	  m_lastUsedMethod(-1),
	  m_catchDegeneracies(1),
	  m_fixContactNormalDirection(1)
{
}
btGjkPairDetector::btGjkPairDetector(const btConvexShape *objectA, const btConvexShape *objectB, int shapeTypeA, int shapeTypeB, btScalar marginA, btScalar marginB, btSimplexSolverInterface *simplexSolver, btConvexPenetrationDepthSolver *penetrationDepthSolver)
	: m_cachedSeparatingAxis(btScalar(0.), btScalar(1.), btScalar(0.)),
	  m_penetrationDepthSolver(penetrationDepthSolver),
	  m_simplexSolver(simplexSolver),
	  m_minkowskiA(objectA),
	  m_minkowskiB(objectB),
	  m_shapeTypeA(shapeTypeA),
	  m_shapeTypeB(shapeTypeB),
	  m_marginA(marginA),
	  m_marginB(marginB),
	  m_ignoreMargin(false),
	  m_lastUsedMethod(-1),
	  m_catchDegeneracies(1),
	  m_fixContactNormalDirection(1)
{
}

void btGjkPairDetector::getClosestPoints(const ClosestPointInput &input, Result &output, class btIDebugDraw *debugDraw, bool swapResults)
{
	(void)swapResults;

	getClosestPointsNonVirtual(input, output, debugDraw);
}

static void btComputeSupport(const btConvexShape *convexA, const btTransform &localTransA, const btConvexShape *convexB, const btTransform &localTransB, const btVector3 &dir, bool check2d, btVector3 &supAworld, btVector3 &supBworld, btVector3 &aMinb)
{
	btVector3 separatingAxisInA = (dir)*localTransA.getBasis();
	btVector3 separatingAxisInB = (-dir) * localTransB.getBasis();

	btVector3 pInANoMargin = convexA->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInA);
	btVector3 qInBNoMargin = convexB->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInB);

	btVector3 pInA = pInANoMargin;
	btVector3 qInB = qInBNoMargin;

	supAworld = localTransA(pInA);
	supBworld = localTransB(qInB);

	if (check2d)
	{
		supAworld[2] = 0.f;
		supBworld[2] = 0.f;
	}

	aMinb = supAworld - supBworld;
}

struct btSupportVector
{
	btVector3 v;   //!< Support point in minkowski sum
	btVector3 v1;  //!< Support point in obj1
	btVector3 v2;  //!< Support point in obj2
};

struct btSimplex
{
	btSupportVector ps[4];
	int last;  //!< index of last added point
};

static btVector3 ccd_vec3_origin(0, 0, 0);

inline void btSimplexInit(btSimplex *s)
{
	s->last = -1;
}

inline int btSimplexSize(const btSimplex *s)
{
	return s->last + 1;
}

inline const btSupportVector *btSimplexPoint(const btSimplex *s, int idx)
{
	// here is no check on boundaries
	return &s->ps[idx];
}
inline void btSupportCopy(btSupportVector *d, const btSupportVector *s)
{
	*d = *s;
}

inline void btVec3Copy(btVector3 *v, const btVector3 *w)
{
	*v = *w;
}

inline void ccdVec3Add(btVector3 *v, const btVector3 *w)
{
	v->m_floats[0] += w->m_floats[0];
	v->m_floats[1] += w->m_floats[1];
	v->m_floats[2] += w->m_floats[2];
}

inline void ccdVec3Sub(btVector3 *v, const btVector3 *w)
{
	*v -= *w;
}
inline void btVec3Sub2(btVector3 *d, const btVector3 *v, const btVector3 *w)
{
	*d = (*v) - (*w);
}
inline btScalar btVec3Dot(const btVector3 *a, const btVector3 *b)
{
	btScalar dot;
	dot = a->dot(*b);

	return dot;
}

inline btScalar ccdVec3Dist2(const btVector3 *a, const btVector3 *b)
{
	btVector3 ab;
	btVec3Sub2(&ab, a, b);
	return btVec3Dot(&ab, &ab);
}

inline void btVec3Scale(btVector3 *d, btScalar k)
{
	d->m_floats[0] *= k;
	d->m_floats[1] *= k;
	d->m_floats[2] *= k;
}

inline void btVec3Cross(btVector3 *d, const btVector3 *a, const btVector3 *b)
{
	d->m_floats[0] = (a->m_floats[1] * b->m_floats[2]) - (a->m_floats[2] * b->m_floats[1]);
	d->m_floats[1] = (a->m_floats[2] * b->m_floats[0]) - (a->m_floats[0] * b->m_floats[2]);
	d->m_floats[2] = (a->m_floats[0] * b->m_floats[1]) - (a->m_floats[1] * b->m_floats[0]);
}

inline void btTripleCross(const btVector3 *a, const btVector3 *b,
						  const btVector3 *c, btVector3 *d)
{
	btVector3 e;
	btVec3Cross(&e, a, b);
	btVec3Cross(d, &e, c);
}

inline int ccdEq(btScalar _a, btScalar _b)
{
	btScalar ab;
	btScalar a, b;

	ab = btFabs(_a - _b);
	if (btFabs(ab) < SIMD_EPSILON)
		return 1;

	a = btFabs(_a);
	b = btFabs(_b);
	if (b > a)
	{
		return ab < SIMD_EPSILON * b;
	}
	else
	{
		return ab < SIMD_EPSILON * a;
	}
}

btScalar ccdVec3X(const btVector3 *v)
{
	return v->x();
}

btScalar ccdVec3Y(const btVector3 *v)
{
	return v->y();
}

btScalar ccdVec3Z(const btVector3 *v)
{
	return v->z();
}
inline int btVec3Eq(const btVector3 *a, const btVector3 *b)
{
	return ccdEq(ccdVec3X(a), ccdVec3X(b)) && ccdEq(ccdVec3Y(a), ccdVec3Y(b)) && ccdEq(ccdVec3Z(a), ccdVec3Z(b));
}

inline void btSimplexAdd(btSimplex *s, const btSupportVector *v)
{
	// here is no check on boundaries in sake of speed
	++s->last;
	btSupportCopy(s->ps + s->last, v);
}

inline void btSimplexSet(btSimplex *s, size_t pos, const btSupportVector *a)
{
	btSupportCopy(s->ps + pos, a);
}

inline void btSimplexSetSize(btSimplex *s, int size)
{
	s->last = size - 1;
}

inline const btSupportVector *ccdSimplexLast(const btSimplex *s)
{
	return btSimplexPoint(s, s->last);
}

inline int ccdSign(btScalar val)
{
	if (btFuzzyZero(val))
	{
		return 0;
	}
	else if (val < btScalar(0))
	{
		return -1;
	}
	return 1;
}

inline btScalar btVec3PointSegmentDist2(const btVector3 *P,
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

	btScalar dist, t;
	btVector3 d, a;

	// direction of segment
	btVec3Sub2(&d, b, x0);

	// precompute vector from P to x0
	btVec3Sub2(&a, x0, P);

	t = -btScalar(1.) * btVec3Dot(&a, &d);
	t /= btVec3Dot(&d, &d);

	if (t < btScalar(0) || btFuzzyZero(t))
	{
		dist = ccdVec3Dist2(x0, P);
		if (witness)
			btVec3Copy(witness, x0);
	}
	else if (t > btScalar(1) || ccdEq(t, btScalar(1)))
	{
		dist = ccdVec3Dist2(b, P);
		if (witness)
			btVec3Copy(witness, b);
	}
	else
	{
		if (witness)
		{
			btVec3Copy(witness, &d);
			btVec3Scale(witness, t);
			ccdVec3Add(witness, x0);
			dist = ccdVec3Dist2(witness, P);
		}
		else
		{
			// recycling variables
			btVec3Scale(&d, t);
			ccdVec3Add(&d, &a);
			dist = btVec3Dot(&d, &d);
		}
	}

	return dist;
}

btScalar btVec3PointTriDist2(const btVector3 *P,
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
	double u, v, w, p, q, r;
	double s, t, dist, dist2;
	btVector3 witness2;

	btVec3Sub2(&d1, B, x0);
	btVec3Sub2(&d2, C, x0);
	btVec3Sub2(&a, x0, P);

	u = btVec3Dot(&a, &a);
	v = btVec3Dot(&d1, &d1);
	w = btVec3Dot(&d2, &d2);
	p = btVec3Dot(&a, &d1);
	q = btVec3Dot(&a, &d2);
	r = btVec3Dot(&d1, &d2);

	s = (q * r - w * p) / (w * v - r * r);
	t = (-s * r - q) / w;

	if ((btFuzzyZero(s) || s > btScalar(0)) && (ccdEq(s, btScalar(1)) || s < btScalar(1)) && (btFuzzyZero(t) || t > btScalar(0)) && (ccdEq(t, btScalar(1)) || t < btScalar(1)) && (ccdEq(t + s, btScalar(1)) || t + s < btScalar(1)))
	{
		if (witness)
		{
			btVec3Scale(&d1, s);
			btVec3Scale(&d2, t);
			btVec3Copy(witness, x0);
			ccdVec3Add(witness, &d1);
			ccdVec3Add(witness, &d2);

			dist = ccdVec3Dist2(witness, P);
		}
		else
		{
			dist = s * s * v;
			dist += t * t * w;
			dist += btScalar(2.) * s * t * r;
			dist += btScalar(2.) * s * p;
			dist += btScalar(2.) * t * q;
			dist += u;
		}
	}
	else
	{
		dist = btVec3PointSegmentDist2(P, x0, B, witness);

		dist2 = btVec3PointSegmentDist2(P, x0, C, &witness2);
		if (dist2 < dist)
		{
			dist = dist2;
			if (witness)
				btVec3Copy(witness, &witness2);
		}

		dist2 = btVec3PointSegmentDist2(P, B, C, &witness2);
		if (dist2 < dist)
		{
			dist = dist2;
			if (witness)
				btVec3Copy(witness, &witness2);
		}
	}

	return dist;
}

static int btDoSimplex2(btSimplex *simplex, btVector3 *dir)
{
	const btSupportVector *A, *B;
	btVector3 AB, AO, tmp;
	btScalar dot;

	// get last added as A
	A = ccdSimplexLast(simplex);
	// get the other point
	B = btSimplexPoint(simplex, 0);
	// compute AB oriented segment
	btVec3Sub2(&AB, &B->v, &A->v);
	// compute AO vector
	btVec3Copy(&AO, &A->v);
	btVec3Scale(&AO, -btScalar(1));

	// dot product AB . AO
	dot = btVec3Dot(&AB, &AO);

	// check if origin doesn't lie on AB segment
	btVec3Cross(&tmp, &AB, &AO);
	if (btFuzzyZero(btVec3Dot(&tmp, &tmp)) && dot > btScalar(0))
	{
		return 1;
	}

	// check if origin is in area where AB segment is
	if (btFuzzyZero(dot) || dot < btScalar(0))
	{
		// origin is in outside are of A
		btSimplexSet(simplex, 0, A);
		btSimplexSetSize(simplex, 1);
		btVec3Copy(dir, &AO);
	}
	else
	{
		// origin is in area where AB segment is

		// keep simplex untouched and set direction to
		// AB x AO x AB
		btTripleCross(&AB, &AO, &AB, dir);
	}

	return 0;
}

static int btDoSimplex3(btSimplex *simplex, btVector3 *dir)
{
	const btSupportVector *A, *B, *C;
	btVector3 AO, AB, AC, ABC, tmp;
	btScalar dot, dist;

	// get last added as A
	A = ccdSimplexLast(simplex);
	// get the other points
	B = btSimplexPoint(simplex, 1);
	C = btSimplexPoint(simplex, 0);

	// check touching contact
	dist = btVec3PointTriDist2(&ccd_vec3_origin, &A->v, &B->v, &C->v, 0);
	if (btFuzzyZero(dist))
	{
		return 1;
	}

	// check if triangle is really triangle (has area > 0)
	// if not simplex can't be expanded and thus no itersection is found
	if (btVec3Eq(&A->v, &B->v) || btVec3Eq(&A->v, &C->v))
	{
		return -1;
	}

	// compute AO vector
	btVec3Copy(&AO, &A->v);
	btVec3Scale(&AO, -btScalar(1));

	// compute AB and AC segments and ABC vector (perpendircular to triangle)
	btVec3Sub2(&AB, &B->v, &A->v);
	btVec3Sub2(&AC, &C->v, &A->v);
	btVec3Cross(&ABC, &AB, &AC);

	btVec3Cross(&tmp, &ABC, &AC);
	dot = btVec3Dot(&tmp, &AO);
	if (btFuzzyZero(dot) || dot > btScalar(0))
	{
		dot = btVec3Dot(&AC, &AO);
		if (btFuzzyZero(dot) || dot > btScalar(0))
		{
			// C is already in place
			btSimplexSet(simplex, 1, A);
			btSimplexSetSize(simplex, 2);
			btTripleCross(&AC, &AO, &AC, dir);
		}
		else
		{
			dot = btVec3Dot(&AB, &AO);
			if (btFuzzyZero(dot) || dot > btScalar(0))
			{
				btSimplexSet(simplex, 0, B);
				btSimplexSet(simplex, 1, A);
				btSimplexSetSize(simplex, 2);
				btTripleCross(&AB, &AO, &AB, dir);
			}
			else
			{
				btSimplexSet(simplex, 0, A);
				btSimplexSetSize(simplex, 1);
				btVec3Copy(dir, &AO);
			}
		}
	}
	else
	{
		btVec3Cross(&tmp, &AB, &ABC);
		dot = btVec3Dot(&tmp, &AO);
		if (btFuzzyZero(dot) || dot > btScalar(0))
		{
			dot = btVec3Dot(&AB, &AO);
			if (btFuzzyZero(dot) || dot > btScalar(0))
			{
				btSimplexSet(simplex, 0, B);
				btSimplexSet(simplex, 1, A);
				btSimplexSetSize(simplex, 2);
				btTripleCross(&AB, &AO, &AB, dir);
			}
			else
			{
				btSimplexSet(simplex, 0, A);
				btSimplexSetSize(simplex, 1);
				btVec3Copy(dir, &AO);
			}
		}
		else
		{
			dot = btVec3Dot(&ABC, &AO);
			if (btFuzzyZero(dot) || dot > btScalar(0))
			{
				btVec3Copy(dir, &ABC);
			}
			else
			{
				btSupportVector tmp;
				btSupportCopy(&tmp, C);
				btSimplexSet(simplex, 0, B);
				btSimplexSet(simplex, 1, &tmp);

				btVec3Copy(dir, &ABC);
				btVec3Scale(dir, -btScalar(1));
			}
		}
	}

	return 0;
}

static int btDoSimplex4(btSimplex *simplex, btVector3 *dir)
{
	const btSupportVector *A, *B, *C, *D;
	btVector3 AO, AB, AC, AD, ABC, ACD, ADB;
	int B_on_ACD, C_on_ADB, D_on_ABC;
	int AB_O, AC_O, AD_O;
	btScalar dist;

	// get last added as A
	A = ccdSimplexLast(simplex);
	// get the other points
	B = btSimplexPoint(simplex, 2);
	C = btSimplexPoint(simplex, 1);
	D = btSimplexPoint(simplex, 0);

	// check if tetrahedron is really tetrahedron (has volume > 0)
	// if it is not simplex can't be expanded and thus no intersection is
	// found
	dist = btVec3PointTriDist2(&A->v, &B->v, &C->v, &D->v, 0);
	if (btFuzzyZero(dist))
	{
		return -1;
	}

	// check if origin lies on some of tetrahedron's face - if so objects
	// intersect
	dist = btVec3PointTriDist2(&ccd_vec3_origin, &A->v, &B->v, &C->v, 0);
	if (btFuzzyZero(dist))
		return 1;
	dist = btVec3PointTriDist2(&ccd_vec3_origin, &A->v, &C->v, &D->v, 0);
	if (btFuzzyZero(dist))
		return 1;
	dist = btVec3PointTriDist2(&ccd_vec3_origin, &A->v, &B->v, &D->v, 0);
	if (btFuzzyZero(dist))
		return 1;
	dist = btVec3PointTriDist2(&ccd_vec3_origin, &B->v, &C->v, &D->v, 0);
	if (btFuzzyZero(dist))
		return 1;

	// compute AO, AB, AC, AD segments and ABC, ACD, ADB normal vectors
	btVec3Copy(&AO, &A->v);
	btVec3Scale(&AO, -btScalar(1));
	btVec3Sub2(&AB, &B->v, &A->v);
	btVec3Sub2(&AC, &C->v, &A->v);
	btVec3Sub2(&AD, &D->v, &A->v);
	btVec3Cross(&ABC, &AB, &AC);
	btVec3Cross(&ACD, &AC, &AD);
	btVec3Cross(&ADB, &AD, &AB);

	// side (positive or negative) of B, C, D relative to planes ACD, ADB
	// and ABC respectively
	B_on_ACD = ccdSign(btVec3Dot(&ACD, &AB));
	C_on_ADB = ccdSign(btVec3Dot(&ADB, &AC));
	D_on_ABC = ccdSign(btVec3Dot(&ABC, &AD));

	// whether origin is on same side of ACD, ADB, ABC as B, C, D
	// respectively
	AB_O = ccdSign(btVec3Dot(&ACD, &AO)) == B_on_ACD;
	AC_O = ccdSign(btVec3Dot(&ADB, &AO)) == C_on_ADB;
	AD_O = ccdSign(btVec3Dot(&ABC, &AO)) == D_on_ABC;

	if (AB_O && AC_O && AD_O)
	{
		// origin is in tetrahedron
		return 1;
		// rearrange simplex to triangle and call btDoSimplex3()
	}
	else if (!AB_O)
	{
		// B is farthest from the origin among all of the tetrahedron's
		// points, so remove it from the list and go on with the triangle
		// case

		// D and C are in place
		btSimplexSet(simplex, 2, A);
		btSimplexSetSize(simplex, 3);
	}
	else if (!AC_O)
	{
		// C is farthest
		btSimplexSet(simplex, 1, D);
		btSimplexSet(simplex, 0, B);
		btSimplexSet(simplex, 2, A);
		btSimplexSetSize(simplex, 3);
	}
	else
	{  // (!AD_O)
		btSimplexSet(simplex, 0, C);
		btSimplexSet(simplex, 1, B);
		btSimplexSet(simplex, 2, A);
		btSimplexSetSize(simplex, 3);
	}

	return btDoSimplex3(simplex, dir);
}

static int btDoSimplex(btSimplex *simplex, btVector3 *dir)
{
	if (btSimplexSize(simplex) == 2)
	{
		// simplex contains segment only one segment
		return btDoSimplex2(simplex, dir);
	}
	else if (btSimplexSize(simplex) == 3)
	{
		// simplex contains triangle
		return btDoSimplex3(simplex, dir);
	}
	else
	{  // btSimplexSize(simplex) == 4
		// tetrahedron - this is the only shape which can encapsule origin
		// so btDoSimplex4() also contains test on it
		return btDoSimplex4(simplex, dir);
	}
}

#ifdef __SPU__
void btGjkPairDetector::getClosestPointsNonVirtual(const ClosestPointInput &input, Result &output, class btIDebugDraw *debugDraw)
#else
void btGjkPairDetector::getClosestPointsNonVirtual(const ClosestPointInput &input, Result &output, class btIDebugDraw *debugDraw)
#endif
{
	m_cachedSeparatingDistance = 0.f;

	btScalar distance = btScalar(0.);
	btVector3 normalInB(btScalar(0.), btScalar(0.), btScalar(0.));

	btVector3 pointOnA, pointOnB;
	btTransform localTransA = input.m_transformA;
	btTransform localTransB = input.m_transformB;
	btVector3 positionOffset = (localTransA.getOrigin() + localTransB.getOrigin()) * btScalar(0.5);
	localTransA.getOrigin() -= positionOffset;
	localTransB.getOrigin() -= positionOffset;

	bool check2d = m_minkowskiA->isConvex2d() && m_minkowskiB->isConvex2d();

	btScalar marginA = m_marginA;
	btScalar marginB = m_marginB;


	//for CCD we don't use margins
	if (m_ignoreMargin)
	{
		marginA = btScalar(0.);
		marginB = btScalar(0.);
	}

	m_curIter = 0;
	int gGjkMaxIter = 1000;  //this is to catch invalid input, perhaps check for #NaN?
	m_cachedSeparatingAxis.setValue(0, 1, 0);

	bool isValid = false;
	bool checkSimplex = false;
	bool checkPenetration = true;
	m_degenerateSimplex = 0;

	m_lastUsedMethod = -1;
	int status = -2;
	btVector3 orgNormalInB(0, 0, 0);
	btScalar margin = marginA + marginB;

	//we add a separate implementation to check if the convex shapes intersect
	//See also "Real-time Collision Detection with Implicit Objects" by Leif Olvang
	//Todo: integrate the simplex penetration check directly inside the Bullet btVoronoiSimplexSolver
	//and remove this temporary code from libCCD
	//this fixes issue https://github.com/bulletphysics/bullet3/issues/1703
	//note, for large differences in shapes, use double precision build!
	{
		btScalar squaredDistance = BT_LARGE_FLOAT;
		btScalar delta = btScalar(0.);

		btSimplex simplex1;
		btSimplex *simplex = &simplex1;
		btSimplexInit(simplex);

		btVector3 dir(1, 0, 0);

		{
			btVector3 lastSupV;
			btVector3 supAworld;
			btVector3 supBworld;
			btComputeSupport(m_minkowskiA, localTransA, m_minkowskiB, localTransB, dir, check2d, supAworld, supBworld, lastSupV);

			btSupportVector last;
			last.v = lastSupV;
			last.v1 = supAworld;
			last.v2 = supBworld;

			btSimplexAdd(simplex, &last);

			dir = -lastSupV;

			// start iterations
			for (int iterations = 0; iterations < gGjkMaxIter; iterations++)
			{
				// obtain support point
				btComputeSupport(m_minkowskiA, localTransA, m_minkowskiB, localTransB, dir, check2d, supAworld, supBworld, lastSupV);

				// check if farthest point in Minkowski difference in direction dir
				// isn't somewhere before origin (the test on negative dot product)
				// - because if it is, objects are not intersecting at all.
				btScalar delta = lastSupV.dot(dir);
				if (delta < 0)
				{
					//no intersection, besides margin
					status = -1;
					break;
				}

				// add last support vector to simplex
				last.v = lastSupV;
				last.v1 = supAworld;
				last.v2 = supBworld;

				btSimplexAdd(simplex, &last);

				// if btDoSimplex returns 1 if objects intersect, -1 if objects don't
				// intersect and 0 if algorithm should continue

				btVector3 newDir;
				int do_simplex_res = btDoSimplex(simplex, &dir);

				if (do_simplex_res == 1)
				{
					status = 0;  // intersection found
					break;
				}
				else if (do_simplex_res == -1)
				{
					// intersection not found
					status = -1;
					break;
				}

				if (btFuzzyZero(btVec3Dot(&dir, &dir)))
				{
					// intersection not found
					status = -1;
				}

				if (dir.length2() < SIMD_EPSILON)
				{
					//no intersection, besides margin
					status = -1;
					break;
				}

				if (dir.fuzzyZero())
				{
					// intersection not found
					status = -1;
					break;
				}
			}
		}

		m_simplexSolver->reset();
		if (status == 0)
		{
			//status = 0;
			//printf("Intersect!\n");
		}

		if (status == -1)
		{
			//printf("not intersect\n");
		}
		//printf("dir=%f,%f,%f\n",dir[0],dir[1],dir[2]);
		if (1)
		{
			for (;;)
			//while (true)
			{
				btVector3 separatingAxisInA = (-m_cachedSeparatingAxis) * localTransA.getBasis();
				btVector3 separatingAxisInB = m_cachedSeparatingAxis * localTransB.getBasis();

				btVector3 pInA = m_minkowskiA->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInA);
				btVector3 qInB = m_minkowskiB->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInB);

				btVector3 pWorld = localTransA(pInA);
				btVector3 qWorld = localTransB(qInB);

				if (check2d)
				{
					pWorld[2] = 0.f;
					qWorld[2] = 0.f;
				}

				btVector3 w = pWorld - qWorld;
				delta = m_cachedSeparatingAxis.dot(w);

				// potential exit, they don't overlap
				if ((delta > btScalar(0.0)) && (delta * delta > squaredDistance * input.m_maximumDistanceSquared))
				{
					m_degenerateSimplex = 10;
					checkSimplex = true;
					//checkPenetration = false;
					break;
				}

				//exit 0: the new point is already in the simplex, or we didn't come any closer
				if (m_simplexSolver->inSimplex(w))
				{
					m_degenerateSimplex = 1;
					checkSimplex = true;
					break;
				}
				// are we getting any closer ?
				btScalar f0 = squaredDistance - delta;
				btScalar f1 = squaredDistance * REL_ERROR2;

				if (f0 <= f1)
				{
					if (f0 <= btScalar(0.))
					{
						m_degenerateSimplex = 2;
					}
					else
					{
						m_degenerateSimplex = 11;
					}
					checkSimplex = true;
					break;
				}

				//add current vertex to simplex
				m_simplexSolver->addVertex(w, pWorld, qWorld);
				btVector3 newCachedSeparatingAxis;

				//calculate the closest point to the origin (update vector v)
				if (!m_simplexSolver->closest(newCachedSeparatingAxis))
				{
					m_degenerateSimplex = 3;
					checkSimplex = true;
					break;
				}

				if (newCachedSeparatingAxis.length2() < REL_ERROR2)
				{
					m_cachedSeparatingAxis = newCachedSeparatingAxis;
					m_degenerateSimplex = 6;
					checkSimplex = true;
					break;
				}

				btScalar previousSquaredDistance = squaredDistance;
				squaredDistance = newCachedSeparatingAxis.length2();
#if 0
				///warning: this termination condition leads to some problems in 2d test case see Bullet/Demos/Box2dDemo
				if (squaredDistance > previousSquaredDistance)
				{
					m_degenerateSimplex = 7;
					squaredDistance = previousSquaredDistance;
					checkSimplex = false;
					break;
				}
#endif  //

				//redundant m_simplexSolver->compute_points(pointOnA, pointOnB);

				//are we getting any closer ?
				if (previousSquaredDistance - squaredDistance <= SIMD_EPSILON * previousSquaredDistance)
				{
					//				m_simplexSolver->backup_closest(m_cachedSeparatingAxis);
					checkSimplex = true;
					m_degenerateSimplex = 12;

					break;
				}

				m_cachedSeparatingAxis = newCachedSeparatingAxis;

				//degeneracy, this is typically due to invalid/uninitialized worldtransforms for a btCollisionObject
				if (m_curIter++ > gGjkMaxIter)
				{
#if defined(DEBUG) || defined(_DEBUG)

					printf("btGjkPairDetector maxIter exceeded:%i\n", m_curIter);
					printf("sepAxis=(%f,%f,%f), squaredDistance = %f, shapeTypeA=%i,shapeTypeB=%i\n",
						   m_cachedSeparatingAxis.getX(),
						   m_cachedSeparatingAxis.getY(),
						   m_cachedSeparatingAxis.getZ(),
						   squaredDistance,
						   m_minkowskiA->getShapeType(),
						   m_minkowskiB->getShapeType());

#endif
					break;
				}

				bool check = (!m_simplexSolver->fullSimplex());
				//bool check = (!m_simplexSolver->fullSimplex() && squaredDistance > SIMD_EPSILON * m_simplexSolver->maxVertex());

				if (!check)
				{
					//do we need this backup_closest here ?
					//				m_simplexSolver->backup_closest(m_cachedSeparatingAxis);
					m_degenerateSimplex = 13;
					break;
				}
			}

			if (checkSimplex)
			{
				m_simplexSolver->compute_points(pointOnA, pointOnB);
				normalInB = m_cachedSeparatingAxis;

				btScalar lenSqr = m_cachedSeparatingAxis.length2();

				//valid normal
				if (lenSqr < REL_ERROR2)
				{
					m_degenerateSimplex = 5;
				}
				if (lenSqr > SIMD_EPSILON * SIMD_EPSILON)
				{
					btScalar rlen = btScalar(1.) / btSqrt(lenSqr);
					normalInB *= rlen;  //normalize

					btScalar s = btSqrt(squaredDistance);

					btAssert(s > btScalar(0.0));
					pointOnA -= m_cachedSeparatingAxis * (marginA / s);
					pointOnB += m_cachedSeparatingAxis * (marginB / s);
					distance = ((btScalar(1.) / rlen) - margin);
					isValid = true;
					orgNormalInB = normalInB;

					m_lastUsedMethod = 1;
				}
				else
				{
					m_lastUsedMethod = 2;
				}
			}
		}

		bool catchDegeneratePenetrationCase =
			(m_catchDegeneracies && m_penetrationDepthSolver && m_degenerateSimplex && ((distance + margin) < gGjkEpaPenetrationTolerance));

		//if (checkPenetration && !isValid)
		if ((checkPenetration && (!isValid || catchDegeneratePenetrationCase)) || (status == 0))
		{
			//penetration case

			//if there is no way to handle penetrations, bail out
			if (m_penetrationDepthSolver)
			{
				// Penetration depth case.
				btVector3 tmpPointOnA, tmpPointOnB;

				m_cachedSeparatingAxis.setZero();

				bool isValid2 = m_penetrationDepthSolver->calcPenDepth(
					*m_simplexSolver,
					m_minkowskiA, m_minkowskiB,
					localTransA, localTransB,
					m_cachedSeparatingAxis, tmpPointOnA, tmpPointOnB,
					debugDraw);

				if (m_cachedSeparatingAxis.length2())
				{
					if (isValid2)
					{
						btVector3 tmpNormalInB = tmpPointOnB - tmpPointOnA;
						btScalar lenSqr = tmpNormalInB.length2();
						if (lenSqr <= (SIMD_EPSILON * SIMD_EPSILON))
						{
							tmpNormalInB = m_cachedSeparatingAxis;
							lenSqr = m_cachedSeparatingAxis.length2();
						}

						if (lenSqr > (SIMD_EPSILON * SIMD_EPSILON))
						{
							tmpNormalInB /= btSqrt(lenSqr);
							btScalar distance2 = -(tmpPointOnA - tmpPointOnB).length();
							m_lastUsedMethod = 3;
							//only replace valid penetrations when the result is deeper (check)
							if (!isValid || (distance2 < distance))
							{
								distance = distance2;
								pointOnA = tmpPointOnA;
								pointOnB = tmpPointOnB;
								normalInB = tmpNormalInB;
								isValid = true;
							}
							else
							{
								m_lastUsedMethod = 8;
							}
						}
						else
						{
							m_lastUsedMethod = 9;
						}
					}
					else

					{
						///this is another degenerate case, where the initial GJK calculation reports a degenerate case
						///EPA reports no penetration, and the second GJK (using the supporting vector without margin)
						///reports a valid positive distance. Use the results of the second GJK instead of failing.
						///thanks to Jacob.Langford for the reproduction case
						///http://code.google.com/p/bullet/issues/detail?id=250

						if (m_cachedSeparatingAxis.length2() > btScalar(0.))
						{
							btScalar distance2 = (tmpPointOnA - tmpPointOnB).length() - margin;
							//only replace valid distances when the distance is less
							if (!isValid || (distance2 < distance))
							{
								distance = distance2;
								pointOnA = tmpPointOnA;
								pointOnB = tmpPointOnB;
								pointOnA -= m_cachedSeparatingAxis * marginA;
								pointOnB += m_cachedSeparatingAxis * marginB;
								normalInB = m_cachedSeparatingAxis;
								normalInB.normalize();

								isValid = true;
								m_lastUsedMethod = 6;
							}
							else
							{
								m_lastUsedMethod = 5;
							}
						}
					}
				}
				else
				{
					//printf("EPA didn't return a valid value\n");
				}
			}
		}
	}

	if (isValid && ((distance < 0) || (distance * distance < input.m_maximumDistanceSquared)))
	{
		m_cachedSeparatingAxis = normalInB;
		m_cachedSeparatingDistance = distance;
		if (1)
		{
			///todo: need to track down this EPA penetration solver degeneracy
			///the penetration solver reports penetration but the contact normal
			///connecting the contact points is pointing in the opposite direction
			///until then, detect the issue and revert the normal

			btScalar d2 = 0.f;
			{
				btVector3 separatingAxisInA = (-orgNormalInB) * localTransA.getBasis();
				btVector3 separatingAxisInB = orgNormalInB * localTransB.getBasis();

				btVector3 pInA = m_minkowskiA->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInA);
				btVector3 qInB = m_minkowskiB->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInB);

				btVector3 pWorld = localTransA(pInA);
				btVector3 qWorld = localTransB(qInB);
				btVector3 w = pWorld - qWorld;
				d2 = orgNormalInB.dot(w) - margin;
			}

			btScalar d1 = 0;
			{
				btVector3 separatingAxisInA = (normalInB)*localTransA.getBasis();
				btVector3 separatingAxisInB = -normalInB * localTransB.getBasis();

				btVector3 pInA = m_minkowskiA->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInA);
				btVector3 qInB = m_minkowskiB->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInB);

				btVector3 pWorld = localTransA(pInA);
				btVector3 qWorld = localTransB(qInB);
				btVector3 w = pWorld - qWorld;
				d1 = (-normalInB).dot(w) - margin;
			}
			btScalar d0 = 0.f;
			{
				btVector3 separatingAxisInA = (-normalInB) * input.m_transformA.getBasis();
				btVector3 separatingAxisInB = normalInB * input.m_transformB.getBasis();

				btVector3 pInA = m_minkowskiA->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInA);
				btVector3 qInB = m_minkowskiB->localGetSupportVertexWithoutMarginNonVirtual(separatingAxisInB);

				btVector3 pWorld = localTransA(pInA);
				btVector3 qWorld = localTransB(qInB);
				btVector3 w = pWorld - qWorld;
				d0 = normalInB.dot(w) - margin;
			}

			if (d1 > d0)
			{
				m_lastUsedMethod = 10;
				normalInB *= -1;
			}

			if (orgNormalInB.length2())
			{
				if (d2 > d0 && d2 > d1 && d2 > distance)
				{
					normalInB = orgNormalInB;
					distance = d2;
				}
			}
		}

		output.addContactPoint(
			normalInB,
			pointOnB + positionOffset,
			distance);
	}
	else
	{
		//printf("invalid gjk query\n");
	}
}
