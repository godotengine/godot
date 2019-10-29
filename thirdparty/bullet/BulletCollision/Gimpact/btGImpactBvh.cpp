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
#include "btGImpactBvh.h"
#include "LinearMath/btQuickprof.h"

#ifdef TRI_COLLISION_PROFILING

btClock g_tree_clock;

float g_accum_tree_collision_time = 0;
int g_count_traversing = 0;

void bt_begin_gim02_tree_time()
{
	g_tree_clock.reset();
}

void bt_end_gim02_tree_time()
{
	g_accum_tree_collision_time += g_tree_clock.getTimeMicroseconds();
	g_count_traversing++;
}

//! Gets the average time in miliseconds of tree collisions
float btGImpactBvh::getAverageTreeCollisionTime()
{
	if (g_count_traversing == 0) return 0;

	float avgtime = g_accum_tree_collision_time;
	avgtime /= (float)g_count_traversing;

	g_accum_tree_collision_time = 0;
	g_count_traversing = 0;
	return avgtime;

	//	float avgtime = g_count_traversing;
	//	g_count_traversing = 0;
	//	return avgtime;
}

#endif  //TRI_COLLISION_PROFILING

/////////////////////// btBvhTree /////////////////////////////////

int btBvhTree::_calc_splitting_axis(
	GIM_BVH_DATA_ARRAY& primitive_boxes, int startIndex, int endIndex)
{
	int i;

	btVector3 means(btScalar(0.), btScalar(0.), btScalar(0.));
	btVector3 variance(btScalar(0.), btScalar(0.), btScalar(0.));
	int numIndices = endIndex - startIndex;

	for (i = startIndex; i < endIndex; i++)
	{
		btVector3 center = btScalar(0.5) * (primitive_boxes[i].m_bound.m_max +
											primitive_boxes[i].m_bound.m_min);
		means += center;
	}
	means *= (btScalar(1.) / (btScalar)numIndices);

	for (i = startIndex; i < endIndex; i++)
	{
		btVector3 center = btScalar(0.5) * (primitive_boxes[i].m_bound.m_max +
											primitive_boxes[i].m_bound.m_min);
		btVector3 diff2 = center - means;
		diff2 = diff2 * diff2;
		variance += diff2;
	}
	variance *= (btScalar(1.) / ((btScalar)numIndices - 1));

	return variance.maxAxis();
}

int btBvhTree::_sort_and_calc_splitting_index(
	GIM_BVH_DATA_ARRAY& primitive_boxes, int startIndex,
	int endIndex, int splitAxis)
{
	int i;
	int splitIndex = startIndex;
	int numIndices = endIndex - startIndex;

	// average of centers
	btScalar splitValue = 0.0f;

	btVector3 means(btScalar(0.), btScalar(0.), btScalar(0.));
	for (i = startIndex; i < endIndex; i++)
	{
		btVector3 center = btScalar(0.5) * (primitive_boxes[i].m_bound.m_max +
											primitive_boxes[i].m_bound.m_min);
		means += center;
	}
	means *= (btScalar(1.) / (btScalar)numIndices);

	splitValue = means[splitAxis];

	//sort leafNodes so all values larger then splitValue comes first, and smaller values start from 'splitIndex'.
	for (i = startIndex; i < endIndex; i++)
	{
		btVector3 center = btScalar(0.5) * (primitive_boxes[i].m_bound.m_max +
											primitive_boxes[i].m_bound.m_min);
		if (center[splitAxis] > splitValue)
		{
			//swap
			primitive_boxes.swap(i, splitIndex);
			//swapLeafNodes(i,splitIndex);
			splitIndex++;
		}
	}

	//if the splitIndex causes unbalanced trees, fix this by using the center in between startIndex and endIndex
	//otherwise the tree-building might fail due to stack-overflows in certain cases.
	//unbalanced1 is unsafe: it can cause stack overflows
	//bool unbalanced1 = ((splitIndex==startIndex) || (splitIndex == (endIndex-1)));

	//unbalanced2 should work too: always use center (perfect balanced trees)
	//bool unbalanced2 = true;

	//this should be safe too:
	int rangeBalancedIndices = numIndices / 3;
	bool unbalanced = ((splitIndex <= (startIndex + rangeBalancedIndices)) || (splitIndex >= (endIndex - 1 - rangeBalancedIndices)));

	if (unbalanced)
	{
		splitIndex = startIndex + (numIndices >> 1);
	}

	btAssert(!((splitIndex == startIndex) || (splitIndex == (endIndex))));

	return splitIndex;
}

void btBvhTree::_build_sub_tree(GIM_BVH_DATA_ARRAY& primitive_boxes, int startIndex, int endIndex)
{
	int curIndex = m_num_nodes;
	m_num_nodes++;

	btAssert((endIndex - startIndex) > 0);

	if ((endIndex - startIndex) == 1)
	{
		//We have a leaf node
		setNodeBound(curIndex, primitive_boxes[startIndex].m_bound);
		m_node_array[curIndex].setDataIndex(primitive_boxes[startIndex].m_data);

		return;
	}
	//calculate Best Splitting Axis and where to split it. Sort the incoming 'leafNodes' array within range 'startIndex/endIndex'.

	//split axis
	int splitIndex = _calc_splitting_axis(primitive_boxes, startIndex, endIndex);

	splitIndex = _sort_and_calc_splitting_index(
		primitive_boxes, startIndex, endIndex,
		splitIndex  //split axis
	);

	//calc this node bounding box

	btAABB node_bound;
	node_bound.invalidate();

	for (int i = startIndex; i < endIndex; i++)
	{
		node_bound.merge(primitive_boxes[i].m_bound);
	}

	setNodeBound(curIndex, node_bound);

	//build left branch
	_build_sub_tree(primitive_boxes, startIndex, splitIndex);

	//build right branch
	_build_sub_tree(primitive_boxes, splitIndex, endIndex);

	m_node_array[curIndex].setEscapeIndex(m_num_nodes - curIndex);
}

//! stackless build tree
void btBvhTree::build_tree(
	GIM_BVH_DATA_ARRAY& primitive_boxes)
{
	// initialize node count to 0
	m_num_nodes = 0;
	// allocate nodes
	m_node_array.resize(primitive_boxes.size() * 2);

	_build_sub_tree(primitive_boxes, 0, primitive_boxes.size());
}

////////////////////////////////////class btGImpactBvh

void btGImpactBvh::refit()
{
	int nodecount = getNodeCount();
	while (nodecount--)
	{
		if (isLeafNode(nodecount))
		{
			btAABB leafbox;
			m_primitive_manager->get_primitive_box(getNodeData(nodecount), leafbox);
			setNodeBound(nodecount, leafbox);
		}
		else
		{
			//const GIM_BVH_TREE_NODE * nodepointer = get_node_pointer(nodecount);
			//get left bound
			btAABB bound;
			bound.invalidate();

			btAABB temp_box;

			int child_node = getLeftNode(nodecount);
			if (child_node)
			{
				getNodeBound(child_node, temp_box);
				bound.merge(temp_box);
			}

			child_node = getRightNode(nodecount);
			if (child_node)
			{
				getNodeBound(child_node, temp_box);
				bound.merge(temp_box);
			}

			setNodeBound(nodecount, bound);
		}
	}
}

//! this rebuild the entire set
void btGImpactBvh::buildSet()
{
	//obtain primitive boxes
	GIM_BVH_DATA_ARRAY primitive_boxes;
	primitive_boxes.resize(m_primitive_manager->get_primitive_count());

	for (int i = 0; i < primitive_boxes.size(); i++)
	{
		m_primitive_manager->get_primitive_box(i, primitive_boxes[i].m_bound);
		primitive_boxes[i].m_data = i;
	}

	m_box_tree.build_tree(primitive_boxes);
}

//! returns the indices of the primitives in the m_primitive_manager
bool btGImpactBvh::boxQuery(const btAABB& box, btAlignedObjectArray<int>& collided_results) const
{
	int curIndex = 0;
	int numNodes = getNodeCount();

	while (curIndex < numNodes)
	{
		btAABB bound;
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
			curIndex += getEscapeNodeIndex(curIndex);
		}
	}
	if (collided_results.size() > 0) return true;
	return false;
}

//! returns the indices of the primitives in the m_primitive_manager
bool btGImpactBvh::rayQuery(
	const btVector3& ray_dir, const btVector3& ray_origin,
	btAlignedObjectArray<int>& collided_results) const
{
	int curIndex = 0;
	int numNodes = getNodeCount();

	while (curIndex < numNodes)
	{
		btAABB bound;
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
			curIndex += getEscapeNodeIndex(curIndex);
		}
	}
	if (collided_results.size() > 0) return true;
	return false;
}

SIMD_FORCE_INLINE bool _node_collision(
	btGImpactBvh* boxset0, btGImpactBvh* boxset1,
	const BT_BOX_BOX_TRANSFORM_CACHE& trans_cache_1to0,
	int node0, int node1, bool complete_primitive_tests)
{
	btAABB box0;
	boxset0->getNodeBound(node0, box0);
	btAABB box1;
	boxset1->getNodeBound(node1, box1);

	return box0.overlapping_trans_cache(box1, trans_cache_1to0, complete_primitive_tests);
	//	box1.appy_transform_trans_cache(trans_cache_1to0);
	//	return box0.has_collision(box1);
}

//stackless recursive collision routine
static void _find_collision_pairs_recursive(
	btGImpactBvh* boxset0, btGImpactBvh* boxset1,
	btPairSet* collision_pairs,
	const BT_BOX_BOX_TRANSFORM_CACHE& trans_cache_1to0,
	int node0, int node1, bool complete_primitive_tests)
{
	if (_node_collision(
			boxset0, boxset1, trans_cache_1to0,
			node0, node1, complete_primitive_tests) == false) return;  //avoid colliding internal nodes

	if (boxset0->isLeafNode(node0))
	{
		if (boxset1->isLeafNode(node1))
		{
			// collision result
			collision_pairs->push_pair(
				boxset0->getNodeData(node0), boxset1->getNodeData(node1));
			return;
		}
		else
		{
			//collide left recursive

			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				node0, boxset1->getLeftNode(node1), false);

			//collide right recursive
			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				node0, boxset1->getRightNode(node1), false);
		}
	}
	else
	{
		if (boxset1->isLeafNode(node1))
		{
			//collide left recursive
			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				boxset0->getLeftNode(node0), node1, false);

			//collide right recursive

			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				boxset0->getRightNode(node0), node1, false);
		}
		else
		{
			//collide left0 left1

			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				boxset0->getLeftNode(node0), boxset1->getLeftNode(node1), false);

			//collide left0 right1

			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				boxset0->getLeftNode(node0), boxset1->getRightNode(node1), false);

			//collide right0 left1

			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				boxset0->getRightNode(node0), boxset1->getLeftNode(node1), false);

			//collide right0 right1

			_find_collision_pairs_recursive(
				boxset0, boxset1,
				collision_pairs, trans_cache_1to0,
				boxset0->getRightNode(node0), boxset1->getRightNode(node1), false);

		}  // else if node1 is not a leaf
	}      // else if node0 is not a leaf
}

void btGImpactBvh::find_collision(btGImpactBvh* boxset0, const btTransform& trans0,
								  btGImpactBvh* boxset1, const btTransform& trans1,
								  btPairSet& collision_pairs)
{
	if (boxset0->getNodeCount() == 0 || boxset1->getNodeCount() == 0) return;

	BT_BOX_BOX_TRANSFORM_CACHE trans_cache_1to0;

	trans_cache_1to0.calc_from_homogenic(trans0, trans1);

#ifdef TRI_COLLISION_PROFILING
	bt_begin_gim02_tree_time();
#endif  //TRI_COLLISION_PROFILING

	_find_collision_pairs_recursive(
		boxset0, boxset1,
		&collision_pairs, trans_cache_1to0, 0, 0, true);
#ifdef TRI_COLLISION_PROFILING
	bt_end_gim02_tree_time();
#endif  //TRI_COLLISION_PROFILING
}
