#ifndef GIM_BASIC_GEOMETRY_OPERATIONS_H_INCLUDED
#define GIM_BASIC_GEOMETRY_OPERATIONS_H_INCLUDED

/*! \file gim_basic_geometry_operations.h
*\author Francisco Leon Najera
type independant geometry routines

*/
/*
-----------------------------------------------------------------------------
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2006 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com

 This library is free software; you can redistribute it and/or
 modify it under the terms of EITHER:
   (1) The GNU Lesser General Public License as published by the Free
       Software Foundation; either version 2.1 of the License, or (at
       your option) any later version. The text of the GNU Lesser
       General Public License is included with this library in the
       file GIMPACT-LICENSE-LGPL.TXT.
   (2) The BSD-style license that is included with this library in
       the file GIMPACT-LICENSE-BSD.TXT.
   (3) The zlib/libpng license that is included with this library in
       the file GIMPACT-LICENSE-ZLIB.TXT.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
 GIMPACT-LICENSE-LGPL.TXT, GIMPACT-LICENSE-ZLIB.TXT and GIMPACT-LICENSE-BSD.TXT for more details.

-----------------------------------------------------------------------------
*/

#include "gim_linear_math.h"

#ifndef PLANEDIREPSILON
#define PLANEDIREPSILON 0.0000001f
#endif

#ifndef PARALELENORMALS
#define PARALELENORMALS 0.000001f
#endif

#define TRIANGLE_NORMAL(v1, v2, v3, n) \
	{                                  \
		vec3f _dif1, _dif2;            \
		VEC_DIFF(_dif1, v2, v1);       \
		VEC_DIFF(_dif2, v3, v1);       \
		VEC_CROSS(n, _dif1, _dif2);    \
		VEC_NORMALIZE(n);              \
	}

#define TRIANGLE_NORMAL_FAST(v1, v2, v3, n) \
	{                                       \
		vec3f _dif1, _dif2;                 \
		VEC_DIFF(_dif1, v2, v1);            \
		VEC_DIFF(_dif2, v3, v1);            \
		VEC_CROSS(n, _dif1, _dif2);         \
	}

/// plane is a vec4f
#define TRIANGLE_PLANE(v1, v2, v3, plane)   \
	{                                       \
		TRIANGLE_NORMAL(v1, v2, v3, plane); \
		plane[3] = VEC_DOT(v1, plane);      \
	}

/// plane is a vec4f
#define TRIANGLE_PLANE_FAST(v1, v2, v3, plane)   \
	{                                            \
		TRIANGLE_NORMAL_FAST(v1, v2, v3, plane); \
		plane[3] = VEC_DOT(v1, plane);           \
	}

/// Calc a plane from an edge an a normal. plane is a vec4f
#define EDGE_PLANE(e1, e2, n, plane)   \
	{                                  \
		vec3f _dif;                    \
		VEC_DIFF(_dif, e2, e1);        \
		VEC_CROSS(plane, _dif, n);     \
		VEC_NORMALIZE(plane);          \
		plane[3] = VEC_DOT(e1, plane); \
	}

#define DISTANCE_PLANE_POINT(plane, point) (VEC_DOT(plane, point) - plane[3])

#define PROJECT_POINT_PLANE(point, plane, projected) \
	{                                                \
		GREAL _dis;                                  \
		_dis = DISTANCE_PLANE_POINT(plane, point);   \
		VEC_SCALE(projected, -_dis, plane);          \
		VEC_SUM(projected, projected, point);        \
	}

//! Verifies if a point is in the plane hull
template <typename CLASS_POINT, typename CLASS_PLANE>
SIMD_FORCE_INLINE bool POINT_IN_HULL(
	const CLASS_POINT &point, const CLASS_PLANE *planes, GUINT plane_count)
{
	GREAL _dis;
	for (GUINT _i = 0; _i < plane_count; ++_i)
	{
		_dis = DISTANCE_PLANE_POINT(planes[_i], point);
		if (_dis > 0.0f) return false;
	}
	return true;
}

template <typename CLASS_POINT, typename CLASS_PLANE>
SIMD_FORCE_INLINE void PLANE_CLIP_SEGMENT(
	const CLASS_POINT &s1,
	const CLASS_POINT &s2, const CLASS_PLANE &plane, CLASS_POINT &clipped)
{
	GREAL _dis1, _dis2;
	_dis1 = DISTANCE_PLANE_POINT(plane, s1);
	VEC_DIFF(clipped, s2, s1);
	_dis2 = VEC_DOT(clipped, plane);
	VEC_SCALE(clipped, -_dis1 / _dis2, clipped);
	VEC_SUM(clipped, clipped, s1);
}

enum ePLANE_INTERSECTION_TYPE
{
	G_BACK_PLANE = 0,
	G_COLLIDE_PLANE,
	G_FRONT_PLANE
};

enum eLINE_PLANE_INTERSECTION_TYPE
{
	G_FRONT_PLANE_S1 = 0,
	G_FRONT_PLANE_S2,
	G_BACK_PLANE_S1,
	G_BACK_PLANE_S2,
	G_COLLIDE_PLANE_S1,
	G_COLLIDE_PLANE_S2
};

//! Confirms if the plane intersect the edge or nor
/*!
intersection type must have the following values
<ul>
<li> 0 : Segment in front of plane, s1 closest
<li> 1 : Segment in front of plane, s2 closest
<li> 2 : Segment in back of plane, s1 closest
<li> 3 : Segment in back of plane, s2 closest
<li> 4 : Segment collides plane, s1 in back
<li> 5 : Segment collides plane, s2 in back
</ul>
*/

template <typename CLASS_POINT, typename CLASS_PLANE>
SIMD_FORCE_INLINE eLINE_PLANE_INTERSECTION_TYPE PLANE_CLIP_SEGMENT2(
	const CLASS_POINT &s1,
	const CLASS_POINT &s2,
	const CLASS_PLANE &plane, CLASS_POINT &clipped)
{
	GREAL _dis1 = DISTANCE_PLANE_POINT(plane, s1);
	GREAL _dis2 = DISTANCE_PLANE_POINT(plane, s2);
	if (_dis1 > -G_EPSILON && _dis2 > -G_EPSILON)
	{
		if (_dis1 < _dis2) return G_FRONT_PLANE_S1;
		return G_FRONT_PLANE_S2;
	}
	else if (_dis1 < G_EPSILON && _dis2 < G_EPSILON)
	{
		if (_dis1 > _dis2) return G_BACK_PLANE_S1;
		return G_BACK_PLANE_S2;
	}

	VEC_DIFF(clipped, s2, s1);
	_dis2 = VEC_DOT(clipped, plane);
	VEC_SCALE(clipped, -_dis1 / _dis2, clipped);
	VEC_SUM(clipped, clipped, s1);
	if (_dis1 < _dis2) return G_COLLIDE_PLANE_S1;
	return G_COLLIDE_PLANE_S2;
}

//! Confirms if the plane intersect the edge or not
/*!
clipped1 and clipped2 are the vertices behind the plane.
clipped1 is the closest

intersection_type must have the following values
<ul>
<li> 0 : Segment in front of plane, s1 closest
<li> 1 : Segment in front of plane, s2 closest
<li> 2 : Segment in back of plane, s1 closest
<li> 3 : Segment in back of plane, s2 closest
<li> 4 : Segment collides plane, s1 in back
<li> 5 : Segment collides plane, s2 in back
</ul>
*/
template <typename CLASS_POINT, typename CLASS_PLANE>
SIMD_FORCE_INLINE eLINE_PLANE_INTERSECTION_TYPE PLANE_CLIP_SEGMENT_CLOSEST(
	const CLASS_POINT &s1,
	const CLASS_POINT &s2,
	const CLASS_PLANE &plane,
	CLASS_POINT &clipped1, CLASS_POINT &clipped2)
{
	eLINE_PLANE_INTERSECTION_TYPE intersection_type = PLANE_CLIP_SEGMENT2(s1, s2, plane, clipped1);
	switch (intersection_type)
	{
		case G_FRONT_PLANE_S1:
			VEC_COPY(clipped1, s1);
			VEC_COPY(clipped2, s2);
			break;
		case G_FRONT_PLANE_S2:
			VEC_COPY(clipped1, s2);
			VEC_COPY(clipped2, s1);
			break;
		case G_BACK_PLANE_S1:
			VEC_COPY(clipped1, s1);
			VEC_COPY(clipped2, s2);
			break;
		case G_BACK_PLANE_S2:
			VEC_COPY(clipped1, s2);
			VEC_COPY(clipped2, s1);
			break;
		case G_COLLIDE_PLANE_S1:
			VEC_COPY(clipped2, s1);
			break;
		case G_COLLIDE_PLANE_S2:
			VEC_COPY(clipped2, s2);
			break;
	}
	return intersection_type;
}

//! Finds the 2 smallest cartesian coordinates of a plane normal
#define PLANE_MINOR_AXES(plane, i0, i1) VEC_MINOR_AXES(plane, i0, i1)

//! Ray plane collision in one way
/*!
Intersects plane in one way only. The ray must face the plane (normals must be in opossite directions).<br/>
It uses the PLANEDIREPSILON constant.
*/
template <typename T, typename CLASS_POINT, typename CLASS_PLANE>
SIMD_FORCE_INLINE bool RAY_PLANE_COLLISION(
	const CLASS_PLANE &plane,
	const CLASS_POINT &vDir,
	const CLASS_POINT &vPoint,
	CLASS_POINT &pout, T &tparam)
{
	GREAL _dis, _dotdir;
	_dotdir = VEC_DOT(plane, vDir);
	if (_dotdir < PLANEDIREPSILON)
	{
		return false;
	}
	_dis = DISTANCE_PLANE_POINT(plane, vPoint);
	tparam = -_dis / _dotdir;
	VEC_SCALE(pout, tparam, vDir);
	VEC_SUM(pout, vPoint, pout);
	return true;
}

//! line collision
/*!
*\return
	-0  if the ray never intersects
	-1 if the ray collides in front
	-2 if the ray collides in back
*/
template <typename T, typename CLASS_POINT, typename CLASS_PLANE>
SIMD_FORCE_INLINE GUINT LINE_PLANE_COLLISION(
	const CLASS_PLANE &plane,
	const CLASS_POINT &vDir,
	const CLASS_POINT &vPoint,
	CLASS_POINT &pout,
	T &tparam,
	T tmin, T tmax)
{
	GREAL _dis, _dotdir;
	_dotdir = VEC_DOT(plane, vDir);
	if (btFabs(_dotdir) < PLANEDIREPSILON)
	{
		tparam = tmax;
		return 0;
	}
	_dis = DISTANCE_PLANE_POINT(plane, vPoint);
	char returnvalue = _dis < 0.0f ? 2 : 1;
	tparam = -_dis / _dotdir;

	if (tparam < tmin)
	{
		returnvalue = 0;
		tparam = tmin;
	}
	else if (tparam > tmax)
	{
		returnvalue = 0;
		tparam = tmax;
	}

	VEC_SCALE(pout, tparam, vDir);
	VEC_SUM(pout, vPoint, pout);
	return returnvalue;
}

/*! \brief Returns the Ray on which 2 planes intersect if they do.
    Written by Rodrigo Hernandez on ODE convex collision

  \param p1 Plane 1
  \param p2 Plane 2
  \param p Contains the origin of the ray upon returning if planes intersect
  \param d Contains the direction of the ray upon returning if planes intersect
  \return true if the planes intersect, 0 if paralell.

*/
template <typename CLASS_POINT, typename CLASS_PLANE>
SIMD_FORCE_INLINE bool INTERSECT_PLANES(
	const CLASS_PLANE &p1,
	const CLASS_PLANE &p2,
	CLASS_POINT &p,
	CLASS_POINT &d)
{
	VEC_CROSS(d, p1, p2);
	GREAL denom = VEC_DOT(d, d);
	if (GIM_IS_ZERO(denom)) return false;
	vec3f _n;
	_n[0] = p1[3] * p2[0] - p2[3] * p1[0];
	_n[1] = p1[3] * p2[1] - p2[3] * p1[1];
	_n[2] = p1[3] * p2[2] - p2[3] * p1[2];
	VEC_CROSS(p, _n, d);
	p[0] /= denom;
	p[1] /= denom;
	p[2] /= denom;
	return true;
}

//***************** SEGMENT and LINE FUNCTIONS **********************************///

/*! Finds the closest point(cp) to (v) on a segment (e1,e2)
 */
template <typename CLASS_POINT>
SIMD_FORCE_INLINE void CLOSEST_POINT_ON_SEGMENT(
	CLASS_POINT &cp, const CLASS_POINT &v,
	const CLASS_POINT &e1, const CLASS_POINT &e2)
{
	vec3f _n;
	VEC_DIFF(_n, e2, e1);
	VEC_DIFF(cp, v, e1);
	GREAL _scalar = VEC_DOT(cp, _n);
	_scalar /= VEC_DOT(_n, _n);
	if (_scalar < 0.0f)
	{
		VEC_COPY(cp, e1);
	}
	else if (_scalar > 1.0f)
	{
		VEC_COPY(cp, e2);
	}
	else
	{
		VEC_SCALE(cp, _scalar, _n);
		VEC_SUM(cp, cp, e1);
	}
}

/*! \brief Finds the line params where these lines intersect.

\param dir1 Direction of line 1
\param point1 Point of line 1
\param dir2 Direction of line 2
\param point2 Point of line 2
\param t1 Result Parameter for line 1
\param t2 Result Parameter for line 2
\param dointersect  0  if the lines won't intersect, else 1

*/
template <typename T, typename CLASS_POINT>
SIMD_FORCE_INLINE bool LINE_INTERSECTION_PARAMS(
	const CLASS_POINT &dir1,
	CLASS_POINT &point1,
	const CLASS_POINT &dir2,
	CLASS_POINT &point2,
	T &t1, T &t2)
{
	GREAL det;
	GREAL e1e1 = VEC_DOT(dir1, dir1);
	GREAL e1e2 = VEC_DOT(dir1, dir2);
	GREAL e2e2 = VEC_DOT(dir2, dir2);
	vec3f p1p2;
	VEC_DIFF(p1p2, point1, point2);
	GREAL p1p2e1 = VEC_DOT(p1p2, dir1);
	GREAL p1p2e2 = VEC_DOT(p1p2, dir2);
	det = e1e2 * e1e2 - e1e1 * e2e2;
	if (GIM_IS_ZERO(det)) return false;
	t1 = (e1e2 * p1p2e2 - e2e2 * p1p2e1) / det;
	t2 = (e1e1 * p1p2e2 - e1e2 * p1p2e1) / det;
	return true;
}

//! Find closest points on segments
template <typename CLASS_POINT>
SIMD_FORCE_INLINE void SEGMENT_COLLISION(
	const CLASS_POINT &vA1,
	const CLASS_POINT &vA2,
	const CLASS_POINT &vB1,
	const CLASS_POINT &vB2,
	CLASS_POINT &vPointA,
	CLASS_POINT &vPointB)
{
	CLASS_POINT _AD, _BD, n;
	vec4f _M;  //plane
	VEC_DIFF(_AD, vA2, vA1);
	VEC_DIFF(_BD, vB2, vB1);
	VEC_CROSS(n, _AD, _BD);
	GREAL _tp = VEC_DOT(n, n);
	if (_tp < G_EPSILON)  //ARE PARALELE
	{
		//project B over A
		bool invert_b_order = false;
		_M[0] = VEC_DOT(vB1, _AD);
		_M[1] = VEC_DOT(vB2, _AD);
		if (_M[0] > _M[1])
		{
			invert_b_order = true;
			GIM_SWAP_NUMBERS(_M[0], _M[1]);
		}
		_M[2] = VEC_DOT(vA1, _AD);
		_M[3] = VEC_DOT(vA2, _AD);
		//mid points
		n[0] = (_M[0] + _M[1]) * 0.5f;
		n[1] = (_M[2] + _M[3]) * 0.5f;

		if (n[0] < n[1])
		{
			if (_M[1] < _M[2])
			{
				vPointB = invert_b_order ? vB1 : vB2;
				vPointA = vA1;
			}
			else if (_M[1] < _M[3])
			{
				vPointB = invert_b_order ? vB1 : vB2;
				CLOSEST_POINT_ON_SEGMENT(vPointA, vPointB, vA1, vA2);
			}
			else
			{
				vPointA = vA2;
				CLOSEST_POINT_ON_SEGMENT(vPointB, vPointA, vB1, vB2);
			}
		}
		else
		{
			if (_M[3] < _M[0])
			{
				vPointB = invert_b_order ? vB2 : vB1;
				vPointA = vA2;
			}
			else if (_M[3] < _M[1])
			{
				vPointA = vA2;
				CLOSEST_POINT_ON_SEGMENT(vPointB, vPointA, vB1, vB2);
			}
			else
			{
				vPointB = invert_b_order ? vB1 : vB2;
				CLOSEST_POINT_ON_SEGMENT(vPointA, vPointB, vA1, vA2);
			}
		}
		return;
	}

	VEC_CROSS(_M, n, _BD);
	_M[3] = VEC_DOT(_M, vB1);

	LINE_PLANE_COLLISION(_M, _AD, vA1, vPointA, _tp, btScalar(0), btScalar(1));
	/*Closest point on segment*/
	VEC_DIFF(vPointB, vPointA, vB1);
	_tp = VEC_DOT(vPointB, _BD);
	_tp /= VEC_DOT(_BD, _BD);
	_tp = GIM_CLAMP(_tp, 0.0f, 1.0f);
	VEC_SCALE(vPointB, _tp, _BD);
	VEC_SUM(vPointB, vPointB, vB1);
}

//! Line box intersection in one dimension
/*!

*\param pos Position of the ray
*\param dir Projection of the Direction of the ray
*\param bmin Minimum bound of the box
*\param bmax Maximum bound of the box
*\param tfirst the minimum projection. Assign to 0 at first.
*\param tlast the maximum projection. Assign to INFINITY at first.
*\return true if there is an intersection.
*/
template <typename T>
SIMD_FORCE_INLINE bool BOX_AXIS_INTERSECT(T pos, T dir, T bmin, T bmax, T &tfirst, T &tlast)
{
	if (GIM_IS_ZERO(dir))
	{
		return !(pos < bmin || pos > bmax);
	}
	GREAL a0 = (bmin - pos) / dir;
	GREAL a1 = (bmax - pos) / dir;
	if (a0 > a1) GIM_SWAP_NUMBERS(a0, a1);
	tfirst = GIM_MAX(a0, tfirst);
	tlast = GIM_MIN(a1, tlast);
	if (tlast < tfirst) return false;
	return true;
}

//! Sorts 3 componets
template <typename T>
SIMD_FORCE_INLINE void SORT_3_INDICES(
	const T *values,
	GUINT *order_indices)
{
	//get minimum
	order_indices[0] = values[0] < values[1] ? (values[0] < values[2] ? 0 : 2) : (values[1] < values[2] ? 1 : 2);

	//get second and third
	GUINT i0 = (order_indices[0] + 1) % 3;
	GUINT i1 = (i0 + 1) % 3;

	if (values[i0] < values[i1])
	{
		order_indices[1] = i0;
		order_indices[2] = i1;
	}
	else
	{
		order_indices[1] = i1;
		order_indices[2] = i0;
	}
}

#endif  // GIM_VECTOR_H_INCLUDED
