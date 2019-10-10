#ifndef GIM_BOX_SET_H_INCLUDED
#define GIM_BOX_SET_H_INCLUDED

/*! \file gim_box_set.h
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

#include "gim_array.h"
#include "gim_radixsort.h"
#include "gim_box_collision.h"
#include "gim_tri_collision.h"
#include "gim_pair.h"

//! A pairset array
class gim_pair_set : public gim_array<GIM_PAIR>
{
public:
	gim_pair_set() : gim_array<GIM_PAIR>(32)
	{
	}
	inline void push_pair(GUINT index1, GUINT index2)
	{
		push_back(GIM_PAIR(index1, index2));
	}

	inline void push_pair_inv(GUINT index1, GUINT index2)
	{
		push_back(GIM_PAIR(index2, index1));
	}
};

//! Prototype Base class for primitive classification
/*!
This class is a wrapper for primitive collections.
This tells relevant info for the Bounding Box set classes, which take care of space classification.
This class can manage Compound shapes and trimeshes, and if it is managing trimesh then the  Hierarchy Bounding Box classes will take advantage of primitive Vs Box overlapping tests for getting optimal results and less Per Box compairisons.
*/
class GIM_PRIMITIVE_MANAGER_PROTOTYPE
{
public:
	virtual ~GIM_PRIMITIVE_MANAGER_PROTOTYPE() {}
	//! determines if this manager consist on only triangles, which special case will be optimized
	virtual bool is_trimesh() = 0;
	virtual GUINT get_primitive_count() = 0;
	virtual void get_primitive_box(GUINT prim_index, GIM_AABB& primbox) = 0;
	virtual void get_primitive_triangle(GUINT prim_index, GIM_TRIANGLE& triangle) = 0;
};

struct GIM_AABB_DATA
{
	GIM_AABB m_bound;
	GUINT m_data;
};

//! Node Structure for trees
struct GIM_BOX_TREE_NODE
{
	GIM_AABB m_bound;
	GUINT m_left;         //!< Left subtree
	GUINT m_right;        //!< Right subtree
	GUINT m_escapeIndex;  //!< Scape index for traversing
	GUINT m_data;         //!< primitive index if apply

	GIM_BOX_TREE_NODE()
	{
		m_left = 0;
		m_right = 0;
		m_escapeIndex = 0;
		m_data = 0;
	}

	SIMD_FORCE_INLINE bool is_leaf_node() const
	{
		return (!m_left && !m_right);
	}
};

//! Basic Box tree structure
class GIM_BOX_TREE
{
protected:
	GUINT m_num_nodes;
	gim_array<GIM_BOX_TREE_NODE> m_node_array;

protected:
	GUINT _sort_and_calc_splitting_index(
		gim_array<GIM_AABB_DATA>& primitive_boxes,
		GUINT startIndex, GUINT endIndex, GUINT splitAxis);

	GUINT _calc_splitting_axis(gim_array<GIM_AABB_DATA>& primitive_boxes, GUINT startIndex, GUINT endIndex);

	void _build_sub_tree(gim_array<GIM_AABB_DATA>& primitive_boxes, GUINT startIndex, GUINT endIndex);

public:
	GIM_BOX_TREE()
	{
		m_num_nodes = 0;
	}

	//! prototype functions for box tree management
	//!@{
	void build_tree(gim_array<GIM_AABB_DATA>& primitive_boxes);

	SIMD_FORCE_INLINE void clearNodes()
	{
		m_node_array.clear();
		m_num_nodes = 0;
	}

	//! node count
	SIMD_FORCE_INLINE GUINT getNodeCount() const
	{
		return m_num_nodes;
	}

	//! tells if the node is a leaf
	SIMD_FORCE_INLINE bool isLeafNode(GUINT nodeindex) const
	{
		return m_node_array[nodeindex].is_leaf_node();
	}

	SIMD_FORCE_INLINE GUINT getNodeData(GUINT nodeindex) const
	{
		return m_node_array[nodeindex].m_data;
	}

	SIMD_FORCE_INLINE void getNodeBound(GUINT nodeindex, GIM_AABB& bound) const
	{
		bound = m_node_array[nodeindex].m_bound;
	}

	SIMD_FORCE_INLINE void setNodeBound(GUINT nodeindex, const GIM_AABB& bound)
	{
		m_node_array[nodeindex].m_bound = bound;
	}

	SIMD_FORCE_INLINE GUINT getLeftNodeIndex(GUINT nodeindex) const
	{
		return m_node_array[nodeindex].m_left;
	}

	SIMD_FORCE_INLINE GUINT getRightNodeIndex(GUINT nodeindex) const
	{
		return m_node_array[nodeindex].m_right;
	}

	SIMD_FORCE_INLINE GUINT getScapeNodeIndex(GUINT nodeindex) const
	{
		return m_node_array[nodeindex].m_escapeIndex;
	}

	//!@}
};

//! Generic Box Tree Template
/*!
This class offers an structure for managing a box tree of primitives.
Requires a Primitive prototype (like GIM_PRIMITIVE_MANAGER_PROTOTYPE ) and
a Box tree structure ( like GIM_BOX_TREE).
*/
template <typename _GIM_PRIMITIVE_MANAGER_PROTOTYPE, typename _GIM_BOX_TREE_PROTOTYPE>
class GIM_BOX_TREE_TEMPLATE_SET
{
protected:
	_GIM_PRIMITIVE_MANAGER_PROTOTYPE m_primitive_manager;
	_GIM_BOX_TREE_PROTOTYPE m_box_tree;

protected:
	//stackless refit
	SIMD_FORCE_INLINE void refit()
	{
		GUINT nodecount = getNodeCount();
		while (nodecount--)
		{
			if (isLeafNode(nodecount))
			{
				GIM_AABB leafbox;
				m_primitive_manager.get_primitive_box(getNodeData(nodecount), leafbox);
				setNodeBound(nodecount, leafbox);
			}
			else
			{
				//get left bound
				GUINT childindex = getLeftNodeIndex(nodecount);
				GIM_AABB bound;
				getNodeBound(childindex, bound);
				//get right bound
				childindex = getRightNodeIndex(nodecount);
				GIM_AABB bound2;
				getNodeBound(childindex, bound2);
				bound.merge(bound2);

				setNodeBound(nodecount, bound);
			}
		}
	}

public:
	GIM_BOX_TREE_TEMPLATE_SET()
	{
	}

	SIMD_FORCE_INLINE GIM_AABB getGlobalBox() const
	{
		GIM_AABB totalbox;
		getNodeBound(0, totalbox);
		return totalbox;
	}

	SIMD_FORCE_INLINE void setPrimitiveManager(const _GIM_PRIMITIVE_MANAGER_PROTOTYPE& primitive_manager)
	{
		m_primitive_manager = primitive_manager;
	}

	const _GIM_PRIMITIVE_MANAGER_PROTOTYPE& getPrimitiveManager() const
	{
		return m_primitive_manager;
	}

	_GIM_PRIMITIVE_MANAGER_PROTOTYPE& getPrimitiveManager()
	{
		return m_primitive_manager;
	}

	//! node manager prototype functions
	///@{

	//! this attemps to refit the box set.
	SIMD_FORCE_INLINE void update()
	{
		refit();
	}

	//! this rebuild the entire set
	SIMD_FORCE_INLINE void buildSet()
	{
		//obtain primitive boxes
		gim_array<GIM_AABB_DATA> primitive_boxes;
		primitive_boxes.resize(m_primitive_manager.get_primitive_count(), false);

		for (GUINT i = 0; i < primitive_boxes.size(); i++)
		{
			m_primitive_manager.get_primitive_box(i, primitive_boxes[i].m_bound);
			primitive_boxes[i].m_data = i;
		}

		m_box_tree.build_tree(primitive_boxes);
	}

	//! returns the indices of the primitives in the m_primitive_manager
	SIMD_FORCE_INLINE bool boxQuery(const GIM_AABB& box, gim_array<GUINT>& collided_results) const
	{
		GUINT curIndex = 0;
		GUINT numNodes = getNodeCount();

		while (curIndex < numNodes)
		{
			GIM_AABB bound;
			getNodeBound(curIndex, bound);

			//catch bugs in tree data

			bool aabbOverlap = bound.has_collision(box);
			bool isleafnode = isLeafNode(curIndex);

			if (isleafnode && aabbOverlap)
			{
				collided_results.push_back(getNodeData(curIndex));
			}

			if (aabbOverlap || isleafnode)
			{
				//next subnode
				curIndex++;
			}
			else
			{
				//skip node
				curIndex += getScapeNodeIndex(curIndex);
			}
		}
		if (collided_results.size() > 0) return true;
		return false;
	}

	//! returns the indices of the primitives in the m_primitive_manager
	SIMD_FORCE_INLINE bool boxQueryTrans(const GIM_AABB& box,
										 const btTransform& transform, gim_array<GUINT>& collided_results) const
	{
		GIM_AABB transbox = box;
		transbox.appy_transform(transform);
		return boxQuery(transbox, collided_results);
	}

	//! returns the indices of the primitives in the m_primitive_manager
	SIMD_FORCE_INLINE bool rayQuery(
		const btVector3& ray_dir, const btVector3& ray_origin,
		gim_array<GUINT>& collided_results) const
	{
		GUINT curIndex = 0;
		GUINT numNodes = getNodeCount();

		while (curIndex < numNodes)
		{
			GIM_AABB bound;
			getNodeBound(curIndex, bound);

			//catch bugs in tree data

			bool aabbOverlap = bound.collide_ray(ray_origin, ray_dir);
			bool isleafnode = isLeafNode(curIndex);

			if (isleafnode && aabbOverlap)
			{
				collided_results.push_back(getNodeData(curIndex));
			}

			if (aabbOverlap || isleafnode)
			{
				//next subnode
				curIndex++;
			}
			else
			{
				//skip node
				curIndex += getScapeNodeIndex(curIndex);
			}
		}
		if (collided_results.size() > 0) return true;
		return false;
	}

	//! tells if this set has hierarcht
	SIMD_FORCE_INLINE bool hasHierarchy() const
	{
		return true;
	}

	//! tells if this set is a trimesh
	SIMD_FORCE_INLINE bool isTrimesh() const
	{
		return m_primitive_manager.is_trimesh();
	}

	//! node count
	SIMD_FORCE_INLINE GUINT getNodeCount() const
	{
		return m_box_tree.getNodeCount();
	}

	//! tells if the node is a leaf
	SIMD_FORCE_INLINE bool isLeafNode(GUINT nodeindex) const
	{
		return m_box_tree.isLeafNode(nodeindex);
	}

	SIMD_FORCE_INLINE GUINT getNodeData(GUINT nodeindex) const
	{
		return m_box_tree.getNodeData(nodeindex);
	}

	SIMD_FORCE_INLINE void getNodeBound(GUINT nodeindex, GIM_AABB& bound) const
	{
		m_box_tree.getNodeBound(nodeindex, bound);
	}

	SIMD_FORCE_INLINE void setNodeBound(GUINT nodeindex, const GIM_AABB& bound)
	{
		m_box_tree.setNodeBound(nodeindex, bound);
	}

	SIMD_FORCE_INLINE GUINT getLeftNodeIndex(GUINT nodeindex) const
	{
		return m_box_tree.getLeftNodeIndex(nodeindex);
	}

	SIMD_FORCE_INLINE GUINT getRightNodeIndex(GUINT nodeindex) const
	{
		return m_box_tree.getRightNodeIndex(nodeindex);
	}

	SIMD_FORCE_INLINE GUINT getScapeNodeIndex(GUINT nodeindex) const
	{
		return m_box_tree.getScapeNodeIndex(nodeindex);
	}

	SIMD_FORCE_INLINE void getNodeTriangle(GUINT nodeindex, GIM_TRIANGLE& triangle) const
	{
		m_primitive_manager.get_primitive_triangle(getNodeData(nodeindex), triangle);
	}
};

//! Class for Box Tree Sets
/*!
this has the GIM_BOX_TREE implementation for bounding boxes.
*/
template <typename _GIM_PRIMITIVE_MANAGER_PROTOTYPE>
class GIM_BOX_TREE_SET : public GIM_BOX_TREE_TEMPLATE_SET<_GIM_PRIMITIVE_MANAGER_PROTOTYPE, GIM_BOX_TREE>
{
public:
};

/// GIM_BOX_SET collision methods
template <typename BOX_SET_CLASS0, typename BOX_SET_CLASS1>
class GIM_TREE_TREE_COLLIDER
{
public:
	gim_pair_set* m_collision_pairs;
	BOX_SET_CLASS0* m_boxset0;
	BOX_SET_CLASS1* m_boxset1;
	GUINT current_node0;
	GUINT current_node1;
	bool node0_is_leaf;
	bool node1_is_leaf;
	bool t0_is_trimesh;
	bool t1_is_trimesh;
	bool node0_has_triangle;
	bool node1_has_triangle;
	GIM_AABB m_box0;
	GIM_AABB m_box1;
	GIM_BOX_BOX_TRANSFORM_CACHE trans_cache_1to0;
	btTransform trans_cache_0to1;
	GIM_TRIANGLE m_tri0;
	btVector4 m_tri0_plane;
	GIM_TRIANGLE m_tri1;
	btVector4 m_tri1_plane;

public:
	GIM_TREE_TREE_COLLIDER()
	{
		current_node0 = G_UINT_INFINITY;
		current_node1 = G_UINT_INFINITY;
	}

protected:
	SIMD_FORCE_INLINE void retrieve_node0_triangle(GUINT node0)
	{
		if (node0_has_triangle) return;
		m_boxset0->getNodeTriangle(node0, m_tri0);
		//transform triangle
		m_tri0.m_vertices[0] = trans_cache_0to1(m_tri0.m_vertices[0]);
		m_tri0.m_vertices[1] = trans_cache_0to1(m_tri0.m_vertices[1]);
		m_tri0.m_vertices[2] = trans_cache_0to1(m_tri0.m_vertices[2]);
		m_tri0.get_plane(m_tri0_plane);

		node0_has_triangle = true;
	}

	SIMD_FORCE_INLINE void retrieve_node1_triangle(GUINT node1)
	{
		if (node1_has_triangle) return;
		m_boxset1->getNodeTriangle(node1, m_tri1);
		//transform triangle
		m_tri1.m_vertices[0] = trans_cache_1to0.transform(m_tri1.m_vertices[0]);
		m_tri1.m_vertices[1] = trans_cache_1to0.transform(m_tri1.m_vertices[1]);
		m_tri1.m_vertices[2] = trans_cache_1to0.transform(m_tri1.m_vertices[2]);
		m_tri1.get_plane(m_tri1_plane);

		node1_has_triangle = true;
	}

	SIMD_FORCE_INLINE void retrieve_node0_info(GUINT node0)
	{
		if (node0 == current_node0) return;
		m_boxset0->getNodeBound(node0, m_box0);
		node0_is_leaf = m_boxset0->isLeafNode(node0);
		node0_has_triangle = false;
		current_node0 = node0;
	}

	SIMD_FORCE_INLINE void retrieve_node1_info(GUINT node1)
	{
		if (node1 == current_node1) return;
		m_boxset1->getNodeBound(node1, m_box1);
		node1_is_leaf = m_boxset1->isLeafNode(node1);
		node1_has_triangle = false;
		current_node1 = node1;
	}

	SIMD_FORCE_INLINE bool node_collision(GUINT node0, GUINT node1)
	{
		retrieve_node0_info(node0);
		retrieve_node1_info(node1);
		bool result = m_box0.overlapping_trans_cache(m_box1, trans_cache_1to0, true);
		if (!result) return false;

		if (t0_is_trimesh && node0_is_leaf)
		{
			//perform primitive vs box collision
			retrieve_node0_triangle(node0);
			//do triangle vs box collision
			m_box1.increment_margin(m_tri0.m_margin);

			result = m_box1.collide_triangle_exact(
				m_tri0.m_vertices[0], m_tri0.m_vertices[1], m_tri0.m_vertices[2], m_tri0_plane);

			m_box1.increment_margin(-m_tri0.m_margin);

			if (!result) return false;
			return true;
		}
		else if (t1_is_trimesh && node1_is_leaf)
		{
			//perform primitive vs box collision
			retrieve_node1_triangle(node1);
			//do triangle vs box collision
			m_box0.increment_margin(m_tri1.m_margin);

			result = m_box0.collide_triangle_exact(
				m_tri1.m_vertices[0], m_tri1.m_vertices[1], m_tri1.m_vertices[2], m_tri1_plane);

			m_box0.increment_margin(-m_tri1.m_margin);

			if (!result) return false;
			return true;
		}
		return true;
	}

	//stackless collision routine
	void find_collision_pairs()
	{
		gim_pair_set stack_collisions;
		stack_collisions.reserve(32);

		//add the first pair
		stack_collisions.push_pair(0, 0);

		while (stack_collisions.size())
		{
			//retrieve the last pair and pop
			GUINT node0 = stack_collisions.back().m_index1;
			GUINT node1 = stack_collisions.back().m_index2;
			stack_collisions.pop_back();
			if (node_collision(node0, node1))  // a collision is found
			{
				if (node0_is_leaf)
				{
					if (node1_is_leaf)
					{
						m_collision_pairs->push_pair(m_boxset0->getNodeData(node0), m_boxset1->getNodeData(node1));
					}
					else
					{
						//collide left
						stack_collisions.push_pair(node0, m_boxset1->getLeftNodeIndex(node1));

						//collide right
						stack_collisions.push_pair(node0, m_boxset1->getRightNodeIndex(node1));
					}
				}
				else
				{
					if (node1_is_leaf)
					{
						//collide left
						stack_collisions.push_pair(m_boxset0->getLeftNodeIndex(node0), node1);
						//collide right
						stack_collisions.push_pair(m_boxset0->getRightNodeIndex(node0), node1);
					}
					else
					{
						GUINT left0 = m_boxset0->getLeftNodeIndex(node0);
						GUINT right0 = m_boxset0->getRightNodeIndex(node0);
						GUINT left1 = m_boxset1->getLeftNodeIndex(node1);
						GUINT right1 = m_boxset1->getRightNodeIndex(node1);
						//collide left
						stack_collisions.push_pair(left0, left1);
						//collide right
						stack_collisions.push_pair(left0, right1);
						//collide left
						stack_collisions.push_pair(right0, left1);
						//collide right
						stack_collisions.push_pair(right0, right1);

					}  // else if node1 is not a leaf
				}      // else if node0 is not a leaf

			}  // if(node_collision(node0,node1))
		}      //while(stack_collisions.size())
	}

public:
	void find_collision(BOX_SET_CLASS0* boxset1, const btTransform& trans1,
						BOX_SET_CLASS1* boxset2, const btTransform& trans2,
						gim_pair_set& collision_pairs, bool complete_primitive_tests = true)
	{
		m_collision_pairs = &collision_pairs;
		m_boxset0 = boxset1;
		m_boxset1 = boxset2;

		trans_cache_1to0.calc_from_homogenic(trans1, trans2);

		trans_cache_0to1 = trans2.inverse();
		trans_cache_0to1 *= trans1;

		if (complete_primitive_tests)
		{
			t0_is_trimesh = boxset1->getPrimitiveManager().is_trimesh();
			t1_is_trimesh = boxset2->getPrimitiveManager().is_trimesh();
		}
		else
		{
			t0_is_trimesh = false;
			t1_is_trimesh = false;
		}

		find_collision_pairs();
	}
};

#endif  // GIM_BOXPRUNING_H_INCLUDED
