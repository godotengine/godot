#ifndef GIM_QUANTIZED_SET_H_INCLUDED
#define GIM_QUANTIZED_SET_H_INCLUDED

/*! \file btGImpactQuantizedBvh.h
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

#include "btGImpactBvh.h"
#include "btQuantization.h"
#include "btGImpactQuantizedBvhStructs.h"

class GIM_QUANTIZED_BVH_NODE_ARRAY:public btAlignedObjectArray<BT_QUANTIZED_BVH_NODE>
{
};




//! Basic Box tree structure
class btQuantizedBvhTree
{
protected:
	int m_num_nodes;
	GIM_QUANTIZED_BVH_NODE_ARRAY m_node_array;
	btAABB m_global_bound;
	btVector3 m_bvhQuantization;
protected:
	void calc_quantization(GIM_BVH_DATA_ARRAY & primitive_boxes, btScalar boundMargin = btScalar(1.0) );

	int _sort_and_calc_splitting_index(
		GIM_BVH_DATA_ARRAY & primitive_boxes,
		 int startIndex,  int endIndex, int splitAxis);

	int _calc_splitting_axis(GIM_BVH_DATA_ARRAY & primitive_boxes, int startIndex,  int endIndex);

	void _build_sub_tree(GIM_BVH_DATA_ARRAY & primitive_boxes, int startIndex,  int endIndex);
public:
	btQuantizedBvhTree()
	{
		m_num_nodes = 0;
	}

	//! prototype functions for box tree management
	//!@{
	void build_tree(GIM_BVH_DATA_ARRAY & primitive_boxes);

	SIMD_FORCE_INLINE void quantizePoint(
		unsigned short * quantizedpoint, const btVector3 & point) const
	{
		bt_quantize_clamp(quantizedpoint,point,m_global_bound.m_min,m_global_bound.m_max,m_bvhQuantization);
	}


	SIMD_FORCE_INLINE bool testQuantizedBoxOverlapp(
		int node_index,
		unsigned short * quantizedMin,unsigned short * quantizedMax) const
	{
		return m_node_array[node_index].testQuantizedBoxOverlapp(quantizedMin,quantizedMax);
	}

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
		bound.m_min = bt_unquantize(
			m_node_array[nodeindex].m_quantizedAabbMin,
			m_global_bound.m_min,m_bvhQuantization);

		bound.m_max = bt_unquantize(
			m_node_array[nodeindex].m_quantizedAabbMax,
			m_global_bound.m_min,m_bvhQuantization);
	}

	SIMD_FORCE_INLINE void setNodeBound(int nodeindex, const btAABB & bound)
	{
		bt_quantize_clamp(	m_node_array[nodeindex].m_quantizedAabbMin,
							bound.m_min,
							m_global_bound.m_min,
							m_global_bound.m_max,
							m_bvhQuantization);

		bt_quantize_clamp(	m_node_array[nodeindex].m_quantizedAabbMax,
							bound.m_max,
							m_global_bound.m_min,
							m_global_bound.m_max,
							m_bvhQuantization);
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

	SIMD_FORCE_INLINE const BT_QUANTIZED_BVH_NODE * get_node_pointer(int index = 0) const
	{
		return &m_node_array[index];
	}

	//!@}
};



//! Structure for containing Boxes
/*!
This class offers an structure for managing a box tree of primitives.
Requires a Primitive prototype (like btPrimitiveManagerBase )
*/
class btGImpactQuantizedBvh
{
protected:
	btQuantizedBvhTree m_box_tree;
	btPrimitiveManagerBase * m_primitive_manager;

protected:
	//stackless refit
	void refit();
public:

	//! this constructor doesn't build the tree. you must call	buildSet
	btGImpactQuantizedBvh()
	{
		m_primitive_manager = NULL;
	}

	//! this constructor doesn't build the tree. you must call	buildSet
	btGImpactQuantizedBvh(btPrimitiveManagerBase * primitive_manager)
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


	SIMD_FORCE_INLINE const BT_QUANTIZED_BVH_NODE * get_node_pointer(int index = 0) const
	{
		return m_box_tree.get_node_pointer(index);
	}

#ifdef TRI_COLLISION_PROFILING
	static float getAverageTreeCollisionTime();
#endif //TRI_COLLISION_PROFILING

	static void find_collision(const btGImpactQuantizedBvh * boxset1, const btTransform & trans1,
		const btGImpactQuantizedBvh * boxset2, const btTransform & trans2,
		btPairSet & collision_pairs);
};

#endif // GIM_BOXPRUNING_H_INCLUDED
