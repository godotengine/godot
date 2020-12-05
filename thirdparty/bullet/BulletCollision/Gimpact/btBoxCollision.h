#ifndef BT_BOX_COLLISION_H_INCLUDED
#define BT_BOX_COLLISION_H_INCLUDED

/*! \file gim_box_collision.h
\author Francisco Leon Najera
*/
/*
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2007 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com


This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "LinearMath/btTransform.h"

///Swap numbers
#define BT_SWAP_NUMBERS(a, b) \
	{                         \
		a = a + b;            \
		b = a - b;            \
		a = a - b;            \
	}

#define BT_MAX(a, b) (a < b ? b : a)
#define BT_MIN(a, b) (a > b ? b : a)

#define BT_GREATER(x, y) btFabs(x) > (y)

#define BT_MAX3(a, b, c) BT_MAX(a, BT_MAX(b, c))
#define BT_MIN3(a, b, c) BT_MIN(a, BT_MIN(b, c))

enum eBT_PLANE_INTERSECTION_TYPE
{
	BT_CONST_BACK_PLANE = 0,
	BT_CONST_COLLIDE_PLANE,
	BT_CONST_FRONT_PLANE
};

//SIMD_FORCE_INLINE bool test_cross_edge_box(
//	const btVector3 & edge,
//	const btVector3 & absolute_edge,
//	const btVector3 & pointa,
//	const btVector3 & pointb, const btVector3 & extend,
//	int dir_index0,
//	int dir_index1
//	int component_index0,
//	int component_index1)
//{
//	// dir coords are -z and y
//
//	const btScalar dir0 = -edge[dir_index0];
//	const btScalar dir1 = edge[dir_index1];
//	btScalar pmin = pointa[component_index0]*dir0 + pointa[component_index1]*dir1;
//	btScalar pmax = pointb[component_index0]*dir0 + pointb[component_index1]*dir1;
//	//find minmax
//	if(pmin>pmax)
//	{
//		BT_SWAP_NUMBERS(pmin,pmax);
//	}
//	//find extends
//	const btScalar rad = extend[component_index0] * absolute_edge[dir_index0] +
//					extend[component_index1] * absolute_edge[dir_index1];
//
//	if(pmin>rad || -rad>pmax) return false;
//	return true;
//}
//
//SIMD_FORCE_INLINE bool test_cross_edge_box_X_axis(
//	const btVector3 & edge,
//	const btVector3 & absolute_edge,
//	const btVector3 & pointa,
//	const btVector3 & pointb, btVector3 & extend)
//{
//
//	return test_cross_edge_box(edge,absolute_edge,pointa,pointb,extend,2,1,1,2);
//}
//
//
//SIMD_FORCE_INLINE bool test_cross_edge_box_Y_axis(
//	const btVector3 & edge,
//	const btVector3 & absolute_edge,
//	const btVector3 & pointa,
//	const btVector3 & pointb, btVector3 & extend)
//{
//
//	return test_cross_edge_box(edge,absolute_edge,pointa,pointb,extend,0,2,2,0);
//}
//
//SIMD_FORCE_INLINE bool test_cross_edge_box_Z_axis(
//	const btVector3 & edge,
//	const btVector3 & absolute_edge,
//	const btVector3 & pointa,
//	const btVector3 & pointb, btVector3 & extend)
//{
//
//	return test_cross_edge_box(edge,absolute_edge,pointa,pointb,extend,1,0,0,1);
//}

#define TEST_CROSS_EDGE_BOX_MCR(edge, absolute_edge, pointa, pointb, _extend, i_dir_0, i_dir_1, i_comp_0, i_comp_1) \
	{                                                                                                               \
		const btScalar dir0 = -edge[i_dir_0];                                                                       \
		const btScalar dir1 = edge[i_dir_1];                                                                        \
		btScalar pmin = pointa[i_comp_0] * dir0 + pointa[i_comp_1] * dir1;                                          \
		btScalar pmax = pointb[i_comp_0] * dir0 + pointb[i_comp_1] * dir1;                                          \
		if (pmin > pmax)                                                                                            \
		{                                                                                                           \
			BT_SWAP_NUMBERS(pmin, pmax);                                                                            \
		}                                                                                                           \
		const btScalar abs_dir0 = absolute_edge[i_dir_0];                                                           \
		const btScalar abs_dir1 = absolute_edge[i_dir_1];                                                           \
		const btScalar rad = _extend[i_comp_0] * abs_dir0 + _extend[i_comp_1] * abs_dir1;                           \
		if (pmin > rad || -rad > pmax) return false;                                                                \
	}

#define TEST_CROSS_EDGE_BOX_X_AXIS_MCR(edge, absolute_edge, pointa, pointb, _extend)       \
	{                                                                                      \
		TEST_CROSS_EDGE_BOX_MCR(edge, absolute_edge, pointa, pointb, _extend, 2, 1, 1, 2); \
	}

#define TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(edge, absolute_edge, pointa, pointb, _extend)       \
	{                                                                                      \
		TEST_CROSS_EDGE_BOX_MCR(edge, absolute_edge, pointa, pointb, _extend, 0, 2, 2, 0); \
	}

#define TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(edge, absolute_edge, pointa, pointb, _extend)       \
	{                                                                                      \
		TEST_CROSS_EDGE_BOX_MCR(edge, absolute_edge, pointa, pointb, _extend, 1, 0, 0, 1); \
	}

//! Returns the dot product between a vec3f and the col of a matrix
SIMD_FORCE_INLINE btScalar bt_mat3_dot_col(
	const btMatrix3x3 &mat, const btVector3 &vec3, int colindex)
{
	return vec3[0] * mat[0][colindex] + vec3[1] * mat[1][colindex] + vec3[2] * mat[2][colindex];
}

//!  Class for transforming a model1 to the space of model0
ATTRIBUTE_ALIGNED16(class)
BT_BOX_BOX_TRANSFORM_CACHE
{
public:
	btVector3 m_T1to0;    //!< Transforms translation of model1 to model 0
	btMatrix3x3 m_R1to0;  //!< Transforms Rotation of model1 to model 0, equal  to R0' * R1
	btMatrix3x3 m_AR;     //!< Absolute value of m_R1to0

	SIMD_FORCE_INLINE void calc_absolute_matrix()
	{
		//		static const btVector3 vepsi(1e-6f,1e-6f,1e-6f);
		//		m_AR[0] = vepsi + m_R1to0[0].absolute();
		//		m_AR[1] = vepsi + m_R1to0[1].absolute();
		//		m_AR[2] = vepsi + m_R1to0[2].absolute();

		int i, j;

		for (i = 0; i < 3; i++)
		{
			for (j = 0; j < 3; j++)
			{
				m_AR[i][j] = 1e-6f + btFabs(m_R1to0[i][j]);
			}
		}
	}

	BT_BOX_BOX_TRANSFORM_CACHE()
	{
	}

	//! Calc the transformation relative  1 to 0. Inverts matrics by transposing
	SIMD_FORCE_INLINE void calc_from_homogenic(const btTransform &trans0, const btTransform &trans1)
	{
		btTransform temp_trans = trans0.inverse();
		temp_trans = temp_trans * trans1;

		m_T1to0 = temp_trans.getOrigin();
		m_R1to0 = temp_trans.getBasis();

		calc_absolute_matrix();
	}

	//! Calcs the full invertion of the matrices. Useful for scaling matrices
	SIMD_FORCE_INLINE void calc_from_full_invert(const btTransform &trans0, const btTransform &trans1)
	{
		m_R1to0 = trans0.getBasis().inverse();
		m_T1to0 = m_R1to0 * (-trans0.getOrigin());

		m_T1to0 += m_R1to0 * trans1.getOrigin();
		m_R1to0 *= trans1.getBasis();

		calc_absolute_matrix();
	}

	SIMD_FORCE_INLINE btVector3 transform(const btVector3 &point) const
	{
		return point.dot3(m_R1to0[0], m_R1to0[1], m_R1to0[2]) + m_T1to0;
	}
};

#define BOX_PLANE_EPSILON 0.000001f

//! Axis aligned box
ATTRIBUTE_ALIGNED16(class)
btAABB
{
public:
	btVector3 m_min;
	btVector3 m_max;

	btAABB()
	{
	}

	btAABB(const btVector3 &V1,
		   const btVector3 &V2,
		   const btVector3 &V3)
	{
		m_min[0] = BT_MIN3(V1[0], V2[0], V3[0]);
		m_min[1] = BT_MIN3(V1[1], V2[1], V3[1]);
		m_min[2] = BT_MIN3(V1[2], V2[2], V3[2]);

		m_max[0] = BT_MAX3(V1[0], V2[0], V3[0]);
		m_max[1] = BT_MAX3(V1[1], V2[1], V3[1]);
		m_max[2] = BT_MAX3(V1[2], V2[2], V3[2]);
	}

	btAABB(const btVector3 &V1,
		   const btVector3 &V2,
		   const btVector3 &V3,
		   btScalar margin)
	{
		m_min[0] = BT_MIN3(V1[0], V2[0], V3[0]);
		m_min[1] = BT_MIN3(V1[1], V2[1], V3[1]);
		m_min[2] = BT_MIN3(V1[2], V2[2], V3[2]);

		m_max[0] = BT_MAX3(V1[0], V2[0], V3[0]);
		m_max[1] = BT_MAX3(V1[1], V2[1], V3[1]);
		m_max[2] = BT_MAX3(V1[2], V2[2], V3[2]);

		m_min[0] -= margin;
		m_min[1] -= margin;
		m_min[2] -= margin;
		m_max[0] += margin;
		m_max[1] += margin;
		m_max[2] += margin;
	}

	btAABB(const btAABB &other) : m_min(other.m_min), m_max(other.m_max)
	{
	}

	btAABB(const btAABB &other, btScalar margin) : m_min(other.m_min), m_max(other.m_max)
	{
		m_min[0] -= margin;
		m_min[1] -= margin;
		m_min[2] -= margin;
		m_max[0] += margin;
		m_max[1] += margin;
		m_max[2] += margin;
	}

	SIMD_FORCE_INLINE void invalidate()
	{
		m_min[0] = SIMD_INFINITY;
		m_min[1] = SIMD_INFINITY;
		m_min[2] = SIMD_INFINITY;
		m_max[0] = -SIMD_INFINITY;
		m_max[1] = -SIMD_INFINITY;
		m_max[2] = -SIMD_INFINITY;
	}

	SIMD_FORCE_INLINE void increment_margin(btScalar margin)
	{
		m_min[0] -= margin;
		m_min[1] -= margin;
		m_min[2] -= margin;
		m_max[0] += margin;
		m_max[1] += margin;
		m_max[2] += margin;
	}

	SIMD_FORCE_INLINE void copy_with_margin(const btAABB &other, btScalar margin)
	{
		m_min[0] = other.m_min[0] - margin;
		m_min[1] = other.m_min[1] - margin;
		m_min[2] = other.m_min[2] - margin;

		m_max[0] = other.m_max[0] + margin;
		m_max[1] = other.m_max[1] + margin;
		m_max[2] = other.m_max[2] + margin;
	}

	template <typename CLASS_POINT>
	SIMD_FORCE_INLINE void calc_from_triangle(
		const CLASS_POINT &V1,
		const CLASS_POINT &V2,
		const CLASS_POINT &V3)
	{
		m_min[0] = BT_MIN3(V1[0], V2[0], V3[0]);
		m_min[1] = BT_MIN3(V1[1], V2[1], V3[1]);
		m_min[2] = BT_MIN3(V1[2], V2[2], V3[2]);

		m_max[0] = BT_MAX3(V1[0], V2[0], V3[0]);
		m_max[1] = BT_MAX3(V1[1], V2[1], V3[1]);
		m_max[2] = BT_MAX3(V1[2], V2[2], V3[2]);
	}

	template <typename CLASS_POINT>
	SIMD_FORCE_INLINE void calc_from_triangle_margin(
		const CLASS_POINT &V1,
		const CLASS_POINT &V2,
		const CLASS_POINT &V3, btScalar margin)
	{
		m_min[0] = BT_MIN3(V1[0], V2[0], V3[0]);
		m_min[1] = BT_MIN3(V1[1], V2[1], V3[1]);
		m_min[2] = BT_MIN3(V1[2], V2[2], V3[2]);

		m_max[0] = BT_MAX3(V1[0], V2[0], V3[0]);
		m_max[1] = BT_MAX3(V1[1], V2[1], V3[1]);
		m_max[2] = BT_MAX3(V1[2], V2[2], V3[2]);

		m_min[0] -= margin;
		m_min[1] -= margin;
		m_min[2] -= margin;
		m_max[0] += margin;
		m_max[1] += margin;
		m_max[2] += margin;
	}

	//! Apply a transform to an AABB
	SIMD_FORCE_INLINE void appy_transform(const btTransform &trans)
	{
		btVector3 center = (m_max + m_min) * 0.5f;
		btVector3 extends = m_max - center;
		// Compute new center
		center = trans(center);

		btVector3 textends = extends.dot3(trans.getBasis().getRow(0).absolute(),
										  trans.getBasis().getRow(1).absolute(),
										  trans.getBasis().getRow(2).absolute());

		m_min = center - textends;
		m_max = center + textends;
	}

	//! Apply a transform to an AABB
	SIMD_FORCE_INLINE void appy_transform_trans_cache(const BT_BOX_BOX_TRANSFORM_CACHE &trans)
	{
		btVector3 center = (m_max + m_min) * 0.5f;
		btVector3 extends = m_max - center;
		// Compute new center
		center = trans.transform(center);

		btVector3 textends = extends.dot3(trans.m_R1to0.getRow(0).absolute(),
										  trans.m_R1to0.getRow(1).absolute(),
										  trans.m_R1to0.getRow(2).absolute());

		m_min = center - textends;
		m_max = center + textends;
	}

	//! Merges a Box
	SIMD_FORCE_INLINE void merge(const btAABB &box)
	{
		m_min[0] = BT_MIN(m_min[0], box.m_min[0]);
		m_min[1] = BT_MIN(m_min[1], box.m_min[1]);
		m_min[2] = BT_MIN(m_min[2], box.m_min[2]);

		m_max[0] = BT_MAX(m_max[0], box.m_max[0]);
		m_max[1] = BT_MAX(m_max[1], box.m_max[1]);
		m_max[2] = BT_MAX(m_max[2], box.m_max[2]);
	}

	//! Merges a point
	template <typename CLASS_POINT>
	SIMD_FORCE_INLINE void merge_point(const CLASS_POINT &point)
	{
		m_min[0] = BT_MIN(m_min[0], point[0]);
		m_min[1] = BT_MIN(m_min[1], point[1]);
		m_min[2] = BT_MIN(m_min[2], point[2]);

		m_max[0] = BT_MAX(m_max[0], point[0]);
		m_max[1] = BT_MAX(m_max[1], point[1]);
		m_max[2] = BT_MAX(m_max[2], point[2]);
	}

	//! Gets the extend and center
	SIMD_FORCE_INLINE void get_center_extend(btVector3 & center, btVector3 & extend) const
	{
		center = (m_max + m_min) * 0.5f;
		extend = m_max - center;
	}

	//! Finds the intersecting box between this box and the other.
	SIMD_FORCE_INLINE void find_intersection(const btAABB &other, btAABB &intersection) const
	{
		intersection.m_min[0] = BT_MAX(other.m_min[0], m_min[0]);
		intersection.m_min[1] = BT_MAX(other.m_min[1], m_min[1]);
		intersection.m_min[2] = BT_MAX(other.m_min[2], m_min[2]);

		intersection.m_max[0] = BT_MIN(other.m_max[0], m_max[0]);
		intersection.m_max[1] = BT_MIN(other.m_max[1], m_max[1]);
		intersection.m_max[2] = BT_MIN(other.m_max[2], m_max[2]);
	}

	SIMD_FORCE_INLINE bool has_collision(const btAABB &other) const
	{
		if (m_min[0] > other.m_max[0] ||
			m_max[0] < other.m_min[0] ||
			m_min[1] > other.m_max[1] ||
			m_max[1] < other.m_min[1] ||
			m_min[2] > other.m_max[2] ||
			m_max[2] < other.m_min[2])
		{
			return false;
		}
		return true;
	}

	/*! \brief Finds the Ray intersection parameter.
	\param aabb Aligned box
	\param vorigin A vec3f with the origin of the ray
	\param vdir A vec3f with the direction of the ray
	*/
	SIMD_FORCE_INLINE bool collide_ray(const btVector3 &vorigin, const btVector3 &vdir) const
	{
		btVector3 extents, center;
		this->get_center_extend(center, extents);
		;

		btScalar Dx = vorigin[0] - center[0];
		if (BT_GREATER(Dx, extents[0]) && Dx * vdir[0] >= 0.0f) return false;
		btScalar Dy = vorigin[1] - center[1];
		if (BT_GREATER(Dy, extents[1]) && Dy * vdir[1] >= 0.0f) return false;
		btScalar Dz = vorigin[2] - center[2];
		if (BT_GREATER(Dz, extents[2]) && Dz * vdir[2] >= 0.0f) return false;

		btScalar f = vdir[1] * Dz - vdir[2] * Dy;
		if (btFabs(f) > extents[1] * btFabs(vdir[2]) + extents[2] * btFabs(vdir[1])) return false;
		f = vdir[2] * Dx - vdir[0] * Dz;
		if (btFabs(f) > extents[0] * btFabs(vdir[2]) + extents[2] * btFabs(vdir[0])) return false;
		f = vdir[0] * Dy - vdir[1] * Dx;
		if (btFabs(f) > extents[0] * btFabs(vdir[1]) + extents[1] * btFabs(vdir[0])) return false;
		return true;
	}

	SIMD_FORCE_INLINE void projection_interval(const btVector3 &direction, btScalar &vmin, btScalar &vmax) const
	{
		btVector3 center = (m_max + m_min) * 0.5f;
		btVector3 extend = m_max - center;

		btScalar _fOrigin = direction.dot(center);
		btScalar _fMaximumExtent = extend.dot(direction.absolute());
		vmin = _fOrigin - _fMaximumExtent;
		vmax = _fOrigin + _fMaximumExtent;
	}

	SIMD_FORCE_INLINE eBT_PLANE_INTERSECTION_TYPE plane_classify(const btVector4 &plane) const
	{
		btScalar _fmin, _fmax;
		this->projection_interval(plane, _fmin, _fmax);

		if (plane[3] > _fmax + BOX_PLANE_EPSILON)
		{
			return BT_CONST_BACK_PLANE;  // 0
		}

		if (plane[3] + BOX_PLANE_EPSILON >= _fmin)
		{
			return BT_CONST_COLLIDE_PLANE;  //1
		}
		return BT_CONST_FRONT_PLANE;  //2
	}

	SIMD_FORCE_INLINE bool overlapping_trans_conservative(const btAABB &box, btTransform &trans1_to_0) const
	{
		btAABB tbox = box;
		tbox.appy_transform(trans1_to_0);
		return has_collision(tbox);
	}

	SIMD_FORCE_INLINE bool overlapping_trans_conservative2(const btAABB &box,
														   const BT_BOX_BOX_TRANSFORM_CACHE &trans1_to_0) const
	{
		btAABB tbox = box;
		tbox.appy_transform_trans_cache(trans1_to_0);
		return has_collision(tbox);
	}

	//! transcache is the transformation cache from box to this AABB
	SIMD_FORCE_INLINE bool overlapping_trans_cache(
		const btAABB &box, const BT_BOX_BOX_TRANSFORM_CACHE &transcache, bool fulltest) const
	{
		//Taken from OPCODE
		btVector3 ea, eb;  //extends
		btVector3 ca, cb;  //extends
		get_center_extend(ca, ea);
		box.get_center_extend(cb, eb);

		btVector3 T;
		btScalar t, t2;
		int i;

		// Class I : A's basis vectors
		for (i = 0; i < 3; i++)
		{
			T[i] = transcache.m_R1to0[i].dot(cb) + transcache.m_T1to0[i] - ca[i];
			t = transcache.m_AR[i].dot(eb) + ea[i];
			if (BT_GREATER(T[i], t)) return false;
		}
		// Class II : B's basis vectors
		for (i = 0; i < 3; i++)
		{
			t = bt_mat3_dot_col(transcache.m_R1to0, T, i);
			t2 = bt_mat3_dot_col(transcache.m_AR, ea, i) + eb[i];
			if (BT_GREATER(t, t2)) return false;
		}
		// Class III : 9 cross products
		if (fulltest)
		{
			int j, m, n, o, p, q, r;
			for (i = 0; i < 3; i++)
			{
				m = (i + 1) % 3;
				n = (i + 2) % 3;
				o = i == 0 ? 1 : 0;
				p = i == 2 ? 1 : 2;
				for (j = 0; j < 3; j++)
				{
					q = j == 2 ? 1 : 2;
					r = j == 0 ? 1 : 0;
					t = T[n] * transcache.m_R1to0[m][j] - T[m] * transcache.m_R1to0[n][j];
					t2 = ea[o] * transcache.m_AR[p][j] + ea[p] * transcache.m_AR[o][j] +
						 eb[r] * transcache.m_AR[i][q] + eb[q] * transcache.m_AR[i][r];
					if (BT_GREATER(t, t2)) return false;
				}
			}
		}
		return true;
	}

	//! Simple test for planes.
	SIMD_FORCE_INLINE bool collide_plane(
		const btVector4 &plane) const
	{
		eBT_PLANE_INTERSECTION_TYPE classify = plane_classify(plane);
		return (classify == BT_CONST_COLLIDE_PLANE);
	}

	//! test for a triangle, with edges
	SIMD_FORCE_INLINE bool collide_triangle_exact(
		const btVector3 &p1,
		const btVector3 &p2,
		const btVector3 &p3,
		const btVector4 &triangle_plane) const
	{
		if (!collide_plane(triangle_plane)) return false;

		btVector3 center, extends;
		this->get_center_extend(center, extends);

		const btVector3 v1(p1 - center);
		const btVector3 v2(p2 - center);
		const btVector3 v3(p3 - center);

		//First axis
		btVector3 diff(v2 - v1);
		btVector3 abs_diff = diff.absolute();
		//Test With X axis
		TEST_CROSS_EDGE_BOX_X_AXIS_MCR(diff, abs_diff, v1, v3, extends);
		//Test With Y axis
		TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(diff, abs_diff, v1, v3, extends);
		//Test With Z axis
		TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(diff, abs_diff, v1, v3, extends);

		diff = v3 - v2;
		abs_diff = diff.absolute();
		//Test With X axis
		TEST_CROSS_EDGE_BOX_X_AXIS_MCR(diff, abs_diff, v2, v1, extends);
		//Test With Y axis
		TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(diff, abs_diff, v2, v1, extends);
		//Test With Z axis
		TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(diff, abs_diff, v2, v1, extends);

		diff = v1 - v3;
		abs_diff = diff.absolute();
		//Test With X axis
		TEST_CROSS_EDGE_BOX_X_AXIS_MCR(diff, abs_diff, v3, v2, extends);
		//Test With Y axis
		TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(diff, abs_diff, v3, v2, extends);
		//Test With Z axis
		TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(diff, abs_diff, v3, v2, extends);

		return true;
	}
};

//! Compairison of transformation objects
SIMD_FORCE_INLINE bool btCompareTransformsEqual(const btTransform &t1, const btTransform &t2)
{
	if (!(t1.getOrigin() == t2.getOrigin())) return false;

	if (!(t1.getBasis().getRow(0) == t2.getBasis().getRow(0))) return false;
	if (!(t1.getBasis().getRow(1) == t2.getBasis().getRow(1))) return false;
	if (!(t1.getBasis().getRow(2) == t2.getBasis().getRow(2))) return false;
	return true;
}

#endif  // GIM_BOX_COLLISION_H_INCLUDED
