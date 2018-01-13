#ifndef GIM_BOX_SET_H_INCLUDED
#define GIM_BOX_SET_H_INCLUDED

/*! \file gim_box_set.h
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


#include "LinearMath/btAlignedObjectArray.h"

#include "btBoxCollision.h"
#include "btTriangleShapeEx.h"
#include "btGImpactBvhStructs.h"

//! A pairset array
class btPairSet: public btAlignedObjectArray<GIM_PAIR>
{
public:
	btPairSet()
	{
		reserve(32);
	}
	inline void push_pair(int index1,int index2)
	{
		push_back(GIM_PAIR(index1,index2));
	}

	inline void push_pair_inv(int index1,int index2)
	{
		push_back(GIM_PAIR(index2,index1));
	}
};

class GIM_BVH_DATA_ARRAY:public btAlignedObjectArray<GIM_BVH_DATA>
{
};


class GIM_BVH_TREE_NODE_ARRAY:public btAlignedObjectArray<GIM_BVH_TREE_NODE>
{
};




//! Basic Box tree structure
class btBvhTree
{
protected:
	int m_num_nodes;
	GIM_BVH_TREE_NODE_ARRAY m_node_array;
protected:
	int _sort_and_calc_splitting_index(
		GIM_BVH_DATA_ARRAY & primitive_boxes,
		 int startIndex,  int endIndex, int splitAxis);

	int _calc_splitting_axis(GIM_BVH_DATA_ARRAY & primitive_boxes, int startIndex,  int endIndex);

	void _build_sub_tree(GIM_BVH_DATA_ARRAY & primitive_boxes, int startIndex,  int endIndex);
public:
	btBvhTree()
	{
		m_num_nodes = 0;
	}

	//! prototype functions for box tree management
	//!@{
	void build_tree(GIM_BVH_DATA_ARRAY & primitive_boxes);

	SIMD_FORCE_INLINE void clearNodes()
	{
		m_node_array.clear();
		m_num_nodes = 0;
	}

	//! node count
	SIMD_FORCE_INLINE int getNodeCount() const
	{
		return m_num_nodes;
	}

	//! tells if the node is a leaf
	SIMD_FORCE_INLINE bool isLeafNode(int nodeindex) const
	{
		return m_node_array[nodeindex].isLeafNode();
	}

	SIMD_FORCE_INLINE int getNodeData(int nodeindex) const
	{
		return m_node_array[nodeindex].getDataIndex();
	}

	SIMD_FORCE_INLINE void getNodeBound(int nodeindex, btAABB & bound) const
	{
		bound = m_node_array[nodeindex].m_bound;
	}

	SIMD_FORCE_INLINE void setNodeBound(int nodeindex, const btAABB & bound)
	{
		m_node_array[nodeindex].m_bound = bound;
	}

	SIMD_FORCE_INLINE int getLeftNode(int nodeindex) const
	{
		return nodeindex+1;
	}

	SIMD_FORCE_INLINE int getRightNode(int nodeindex) const
	{
		if(m_node_array[nodeindex+1].isLeafNode()) return nodeindex+2;
		return nodeindex+1 + m_node_array[nodeindex+1].getEscapeIndex();
	}

	SIMD_FORCE_INLINE int getEscapeNodeIndex(int nodeindex) const
	{
		return m_node_array[nodeindex].getEscapeIndex();
	}

	SIMD_FORCE_INLINE const GIM_BVH_TREE_NODE * get_node_pointer(int index = 0) const
	{
		return &m_node_array[index];
	}

	//!@}
};


//! Prototype Base class for primitive classification
/*!
This class is a wrapper for primitive collections.
This tells relevant info for the Bounding Box set classes, which take care of space classification.
This class can manage Compound shapes and trimeshes, and if it is managing trimesh then the  Hierarchy Bounding Box classes will take advantage of primitive Vs Box overlapping tests for getting optimal results and less Per Box compairisons.
*/
class btPrimitiveManagerBase
{
public:

	virtual ~btPrimitiveManagerBase() {}

	//! determines if this manager consist on only triangles, which special case will be optimized
	virtual bool is_trimesh() const = 0;
	virtual int get_primitive_count() const = 0;
	virtual void get_primitive_box(int prim_index ,btAABB & primbox) const = 0;
	//! retrieves only the points of the triangle, and the collision margin
	virtual void get_primitive_triangle(int prim_index,btPrimitiveTriangle & triangle) const= 0;
};


//! Structure for containing Boxes
/*!
This class offers an structure for managing a box tree of primitives.
Requires a Primitive prototype (like btPrimitiveManagerBase )
*/
class btGImpactBvh
{
protected:
	btBvhTree m_box_tree;
	btPrimitiveManagerBase * m_primitive_manager;

protected:
	//stackless refit
	void refit();
public:

	//! this constructor doesn't build the tree. you must call	buildSet
	btGImpactBvh()
	{
		m_primitive_manager = NULL;
	}

	//! this constructor doesn't build the tree. you must call	buildSet
	btGImpactBvh(btPrimitiveManagerBase * primitive_manager)
	{
		m_primitive_manager = primitive_manager;
	}

	SIMD_FORCE_INLINE btAABB getGlobalBox()  const
	{
		btAABB totalbox;
		getNodeBound(0, totalbox);
		return totalbox;
	}

	SIMD_FORCE_INLINE void setPrimitiveManager(btPrimitiveManagerBase * primitive_manager)
	{
		m_primitive_manager = primitive_manager;
	}

	SIMD_FORCE_INLINE btPrimitiveManagerBase * getPrimitiveManager() const
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
	void buildSet();

	//! returns the indices of the primitives in the m_primitive_manager
	bool boxQuery(const btAABB & box, btAlignedObjectArray<int> & collided_results) const;

	//! returns the indices of the primitives in the m_primitive_manager
	SIMD_FORCE_INLINE bool boxQueryTrans(const btAABB & box,
		 const btTransform & transform, btAlignedObjectArray<int> & collided_results) const
	{
		btAABB transbox=box;
		transbox.appy_transform(transform);
		return boxQuery(transbox,collided_results);
	}

	//! returns the indices of the primitives in the m_primitive_manager
	bool rayQuery(
		const btVector3 & ray_dir,const btVector3 & ray_origin ,
		btAlignedObjectArray<int> & collided_results) const;

	//! tells if this set has hierarcht
	SIMD_FORCE_INLINE bool hasHierarchy() const
	{
		return true;
	}

	//! tells if this set is a trimesh
	SIMD_FORCE_INLINE bool isTrimesh()  const
	{
		return m_primitive_manager->is_trimesh();
	}

	//! node count
	SIMD_FORCE_INLINE int getNodeCount() const
	{
		return m_box_tree.getNodeCount();
	}

	//! tells if the node is a leaf
	SIMD_FORCE_INLINE bool isLeafNode(int nodeindex) const
	{
		return m_box_tree.isLeafNode(nodeindex);
	}

	SIMD_FORCE_INLINE int getNodeData(int nodeindex) const
	{
		return m_box_tree.getNodeData(nodeindex);
	}

	SIMD_FORCE_INLINE void getNodeBound(int nodeindex, btAABB & bound)  const
	{
		m_box_tree.getNodeBound(nodeindex, bound);
	}

	SIMD_FORCE_INLINE void setNodeBound(int nodeindex, const btAABB & bound)
	{
		m_box_tree.setNodeBound(nodeindex, bound);
	}


	SIMD_FORCE_INLINE int getLeftNode(int nodeindex) const
	{
		return m_box_tree.getLeftNode(nodeindex);
	}

	SIMD_FORCE_INLINE int getRightNode(int nodeindex) const
	{
		return m_box_tree.getRightNode(nodeindex);
	}

	SIMD_FORCE_INLINE int getEscapeNodeIndex(int nodeindex) const
	{
		return m_box_tree.getEscapeNodeIndex(nodeindex);
	}

	SIMD_FORCE_INLINE void getNodeTriangle(int nodeindex,btPrimitiveTriangle & triangle) const
	{
		m_primitive_manager->get_primitive_triangle(getNodeData(nodeindex),triangle);
	}


	SIMD_FORCE_INLINE const GIM_BVH_TREE_NODE * get_node_pointer(int index = 0) const
	{
		return m_box_tree.get_node_pointer(index);
	}

#ifdef TRI_COLLISION_PROFILING
	static float getAverageTreeCollisionTime();
#endif //TRI_COLLISION_PROFILING

	static void find_collision(btGImpactBvh * boxset1, const btTransform & trans1,
		btGImpactBvh * boxset2, const btTransform & trans2,
		btPairSet & collision_pairs);
};

#endif // GIM_BOXPRUNING_H_INCLUDED
