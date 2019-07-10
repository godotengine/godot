#ifndef GIM_TRI_COLLISION_H_INCLUDED
#define GIM_TRI_COLLISION_H_INCLUDED

/*! \file gim_tri_collision.h
\author Francisco Leon Najera
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

#include "gim_box_collision.h"
#include "gim_clip_polygon.h"

#ifndef MAX_TRI_CLIPPING
#define MAX_TRI_CLIPPING 16
#endif

//! Structure for collision
struct GIM_TRIANGLE_CONTACT_DATA
{
	GREAL m_penetration_depth;
	GUINT m_point_count;
	btVector4 m_separating_normal;
	btVector3 m_points[MAX_TRI_CLIPPING];

	SIMD_FORCE_INLINE void copy_from(const GIM_TRIANGLE_CONTACT_DATA &other)
	{
		m_penetration_depth = other.m_penetration_depth;
		m_separating_normal = other.m_separating_normal;
		m_point_count = other.m_point_count;
		GUINT i = m_point_count;
		while (i--)
		{
			m_points[i] = other.m_points[i];
		}
	}

	GIM_TRIANGLE_CONTACT_DATA()
	{
	}

	GIM_TRIANGLE_CONTACT_DATA(const GIM_TRIANGLE_CONTACT_DATA &other)
	{
		copy_from(other);
	}

	//! classify points that are closer
	template <typename DISTANCE_FUNC, typename CLASS_PLANE>
	SIMD_FORCE_INLINE void mergepoints_generic(const CLASS_PLANE &plane,
											   GREAL margin, const btVector3 *points, GUINT point_count, DISTANCE_FUNC distance_func)
	{
		m_point_count = 0;
		m_penetration_depth = -1000.0f;

		GUINT point_indices[MAX_TRI_CLIPPING];

		GUINT _k;

		for (_k = 0; _k < point_count; _k++)
		{
			GREAL _dist = -distance_func(plane, points[_k]) + margin;

			if (_dist >= 0.0f)
			{
				if (_dist > m_penetration_depth)
				{
					m_penetration_depth = _dist;
					point_indices[0] = _k;
					m_point_count = 1;
				}
				else if ((_dist + G_EPSILON) >= m_penetration_depth)
				{
					point_indices[m_point_count] = _k;
					m_point_count++;
				}
			}
		}

		for (_k = 0; _k < m_point_count; _k++)
		{
			m_points[_k] = points[point_indices[_k]];
		}
	}

	//! classify points that are closer
	SIMD_FORCE_INLINE void merge_points(const btVector4 &plane, GREAL margin,
										const btVector3 *points, GUINT point_count)
	{
		m_separating_normal = plane;
		mergepoints_generic(plane, margin, points, point_count, DISTANCE_PLANE_3D_FUNC());
	}
};

//! Class for colliding triangles
class GIM_TRIANGLE
{
public:
	btScalar m_margin;
	btVector3 m_vertices[3];

	GIM_TRIANGLE() : m_margin(0.1f)
	{
	}

	SIMD_FORCE_INLINE GIM_AABB get_box() const
	{
		return GIM_AABB(m_vertices[0], m_vertices[1], m_vertices[2], m_margin);
	}

	SIMD_FORCE_INLINE void get_normal(btVector3 &normal) const
	{
		TRIANGLE_NORMAL(m_vertices[0], m_vertices[1], m_vertices[2], normal);
	}

	SIMD_FORCE_INLINE void get_plane(btVector4 &plane) const
	{
		TRIANGLE_PLANE(m_vertices[0], m_vertices[1], m_vertices[2], plane);
		;
	}

	SIMD_FORCE_INLINE void apply_transform(const btTransform &trans)
	{
		m_vertices[0] = trans(m_vertices[0]);
		m_vertices[1] = trans(m_vertices[1]);
		m_vertices[2] = trans(m_vertices[2]);
	}

	SIMD_FORCE_INLINE void get_edge_plane(GUINT edge_index, const btVector3 &triangle_normal, btVector4 &plane) const
	{
		const btVector3 &e0 = m_vertices[edge_index];
		const btVector3 &e1 = m_vertices[(edge_index + 1) % 3];
		EDGE_PLANE(e0, e1, triangle_normal, plane);
	}

	//! Gets the relative transformation of this triangle
	/*!
    The transformation is oriented to the triangle normal , and aligned to the 1st edge of this triangle. The position corresponds to vertice 0:
    - triangle normal corresponds to Z axis.
    - 1st normalized edge corresponds to X axis,

    */
	SIMD_FORCE_INLINE void get_triangle_transform(btTransform &triangle_transform) const
	{
		btMatrix3x3 &matrix = triangle_transform.getBasis();

		btVector3 zaxis;
		get_normal(zaxis);
		MAT_SET_Z(matrix, zaxis);

		btVector3 xaxis = m_vertices[1] - m_vertices[0];
		VEC_NORMALIZE(xaxis);
		MAT_SET_X(matrix, xaxis);

		//y axis
		xaxis = zaxis.cross(xaxis);
		MAT_SET_Y(matrix, xaxis);

		triangle_transform.setOrigin(m_vertices[0]);
	}

	//! Test triangles by finding separating axis
	/*!
	\param other Triangle for collide
	\param contact_data Structure for holding contact points, normal and penetration depth; The normal is pointing toward this triangle from the other triangle
	*/
	bool collide_triangle_hard_test(
		const GIM_TRIANGLE &other,
		GIM_TRIANGLE_CONTACT_DATA &contact_data) const;

	//! Test boxes before doing hard test
	/*!
	\param other Triangle for collide
	\param contact_data Structure for holding contact points, normal and penetration depth; The normal is pointing toward this triangle from the other triangle
	\
	*/
	SIMD_FORCE_INLINE bool collide_triangle(
		const GIM_TRIANGLE &other,
		GIM_TRIANGLE_CONTACT_DATA &contact_data) const
	{
		//test box collisioin
		GIM_AABB boxu(m_vertices[0], m_vertices[1], m_vertices[2], m_margin);
		GIM_AABB boxv(other.m_vertices[0], other.m_vertices[1], other.m_vertices[2], other.m_margin);
		if (!boxu.has_collision(boxv)) return false;

		//do hard test
		return collide_triangle_hard_test(other, contact_data);
	}

	/*!

	Solve the System for u,v parameters:

	u*axe1[i1] + v*axe2[i1] = vecproj[i1]
	u*axe1[i2] + v*axe2[i2] = vecproj[i2]

	sustitute:
	v = (vecproj[i2] - u*axe1[i2])/axe2[i2]

	then the first equation in terms of 'u':

	--> u*axe1[i1] + ((vecproj[i2] - u*axe1[i2])/axe2[i2])*axe2[i1] = vecproj[i1]

	--> u*axe1[i1] + vecproj[i2]*axe2[i1]/axe2[i2] - u*axe1[i2]*axe2[i1]/axe2[i2] = vecproj[i1]

	--> u*(axe1[i1]  - axe1[i2]*axe2[i1]/axe2[i2]) = vecproj[i1] - vecproj[i2]*axe2[i1]/axe2[i2]

	--> u*((axe1[i1]*axe2[i2]  - axe1[i2]*axe2[i1])/axe2[i2]) = (vecproj[i1]*axe2[i2] - vecproj[i2]*axe2[i1])/axe2[i2]

	--> u*(axe1[i1]*axe2[i2]  - axe1[i2]*axe2[i1]) = vecproj[i1]*axe2[i2] - vecproj[i2]*axe2[i1]

	--> u = (vecproj[i1]*axe2[i2] - vecproj[i2]*axe2[i1]) /(axe1[i1]*axe2[i2]  - axe1[i2]*axe2[i1])

if 0.0<= u+v <=1.0 then they are inside of triangle

	\return false if the point is outside of triangle.This function  doesn't take the margin
	*/
	SIMD_FORCE_INLINE bool get_uv_parameters(
		const btVector3 &point,
		const btVector3 &tri_plane,
		GREAL &u, GREAL &v) const
	{
		btVector3 _axe1 = m_vertices[1] - m_vertices[0];
		btVector3 _axe2 = m_vertices[2] - m_vertices[0];
		btVector3 _vecproj = point - m_vertices[0];
		GUINT _i1 = (tri_plane.closestAxis() + 1) % 3;
		GUINT _i2 = (_i1 + 1) % 3;
		if (btFabs(_axe2[_i2]) < G_EPSILON)
		{
			u = (_vecproj[_i2] * _axe2[_i1] - _vecproj[_i1] * _axe2[_i2]) / (_axe1[_i2] * _axe2[_i1] - _axe1[_i1] * _axe2[_i2]);
			v = (_vecproj[_i1] - u * _axe1[_i1]) / _axe2[_i1];
		}
		else
		{
			u = (_vecproj[_i1] * _axe2[_i2] - _vecproj[_i2] * _axe2[_i1]) / (_axe1[_i1] * _axe2[_i2] - _axe1[_i2] * _axe2[_i1]);
			v = (_vecproj[_i2] - u * _axe1[_i2]) / _axe2[_i2];
		}

		if (u < -G_EPSILON)
		{
			return false;
		}
		else if (v < -G_EPSILON)
		{
			return false;
		}
		else
		{
			btScalar sumuv;
			sumuv = u + v;
			if (sumuv < -G_EPSILON)
			{
				return false;
			}
			else if (sumuv - 1.0f > G_EPSILON)
			{
				return false;
			}
		}
		return true;
	}

	//! is point in triangle beam?
	/*!
	Test if point is in triangle, with m_margin tolerance
	*/
	SIMD_FORCE_INLINE bool is_point_inside(const btVector3 &point, const btVector3 &tri_normal) const
	{
		//Test with edge 0
		btVector4 edge_plane;
		this->get_edge_plane(0, tri_normal, edge_plane);
		GREAL dist = DISTANCE_PLANE_POINT(edge_plane, point);
		if (dist - m_margin > 0.0f) return false;  // outside plane

		this->get_edge_plane(1, tri_normal, edge_plane);
		dist = DISTANCE_PLANE_POINT(edge_plane, point);
		if (dist - m_margin > 0.0f) return false;  // outside plane

		this->get_edge_plane(2, tri_normal, edge_plane);
		dist = DISTANCE_PLANE_POINT(edge_plane, point);
		if (dist - m_margin > 0.0f) return false;  // outside plane
		return true;
	}

	//! Bidireccional ray collision
	SIMD_FORCE_INLINE bool ray_collision(
		const btVector3 &vPoint,
		const btVector3 &vDir, btVector3 &pout, btVector3 &triangle_normal,
		GREAL &tparam, GREAL tmax = G_REAL_INFINITY)
	{
		btVector4 faceplane;
		{
			btVector3 dif1 = m_vertices[1] - m_vertices[0];
			btVector3 dif2 = m_vertices[2] - m_vertices[0];
			VEC_CROSS(faceplane, dif1, dif2);
			faceplane[3] = m_vertices[0].dot(faceplane);
		}

		GUINT res = LINE_PLANE_COLLISION(faceplane, vDir, vPoint, pout, tparam, btScalar(0), tmax);
		if (res == 0) return false;
		if (!is_point_inside(pout, faceplane)) return false;

		if (res == 2)  //invert normal
		{
			triangle_normal.setValue(-faceplane[0], -faceplane[1], -faceplane[2]);
		}
		else
		{
			triangle_normal.setValue(faceplane[0], faceplane[1], faceplane[2]);
		}

		VEC_NORMALIZE(triangle_normal);

		return true;
	}

	//! one direccion ray collision
	SIMD_FORCE_INLINE bool ray_collision_front_side(
		const btVector3 &vPoint,
		const btVector3 &vDir, btVector3 &pout, btVector3 &triangle_normal,
		GREAL &tparam, GREAL tmax = G_REAL_INFINITY)
	{
		btVector4 faceplane;
		{
			btVector3 dif1 = m_vertices[1] - m_vertices[0];
			btVector3 dif2 = m_vertices[2] - m_vertices[0];
			VEC_CROSS(faceplane, dif1, dif2);
			faceplane[3] = m_vertices[0].dot(faceplane);
		}

		GUINT res = LINE_PLANE_COLLISION(faceplane, vDir, vPoint, pout, tparam, btScalar(0), tmax);
		if (res != 1) return false;

		if (!is_point_inside(pout, faceplane)) return false;

		triangle_normal.setValue(faceplane[0], faceplane[1], faceplane[2]);

		VEC_NORMALIZE(triangle_normal);

		return true;
	}
};

#endif  // GIM_TRI_COLLISION_H_INCLUDED
