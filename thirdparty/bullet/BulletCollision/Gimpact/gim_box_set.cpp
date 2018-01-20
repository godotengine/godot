
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


#include "gim_box_set.h"


GUINT GIM_BOX_TREE::_calc_splitting_axis(
	gim_array<GIM_AABB_DATA> & primitive_boxes, GUINT startIndex,  GUINT endIndex)
{
	GUINT i;

	btVector3 means(btScalar(0.),btScalar(0.),btScalar(0.));
	btVector3 variance(btScalar(0.),btScalar(0.),btScalar(0.));
	GUINT numIndices = endIndex-startIndex;

	for (i=startIndex;i<endIndex;i++)
	{
		btVector3 center = btScalar(0.5)*(primitive_boxes[i].m_bound.m_max +
					 primitive_boxes[i].m_bound.m_min);
		means+=center;
	}
	means *= (btScalar(1.)/(btScalar)numIndices);

	for (i=startIndex;i<endIndex;i++)
	{
		btVector3 center = btScalar(0.5)*(primitive_boxes[i].m_bound.m_max +
					 primitive_boxes[i].m_bound.m_min);
		btVector3 diff2 = center-means;
		diff2 = diff2 * diff2;
		variance += diff2;
	}
	variance *= (btScalar(1.)/	((btScalar)numIndices-1)	);

	return variance.maxAxis();
}


GUINT GIM_BOX_TREE::_sort_and_calc_splitting_index(
	gim_array<GIM_AABB_DATA> & primitive_boxes, GUINT startIndex,
	GUINT endIndex, GUINT splitAxis)
{
	GUINT i;
	GUINT splitIndex =startIndex;
	GUINT numIndices = endIndex - startIndex;

	// average of centers
	btScalar splitValue = 0.0f;
	for (i=startIndex;i<endIndex;i++)
	{
		splitValue+= 0.5f*(primitive_boxes[i].m_bound.m_max[splitAxis] +
					 primitive_boxes[i].m_bound.m_min[splitAxis]);
	}
	splitValue /= (btScalar)numIndices;

	//sort leafNodes so all values larger then splitValue comes first, and smaller values start from 'splitIndex'.
	for (i=startIndex;i<endIndex;i++)
	{
		btScalar center = 0.5f*(primitive_boxes[i].m_bound.m_max[splitAxis] +
					 primitive_boxes[i].m_bound.m_min[splitAxis]);
		if (center > splitValue)
		{
			//swap
			primitive_boxes.swap(i,splitIndex);
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
	GUINT rangeBalancedIndices = numIndices/3;
	bool unbalanced = ((splitIndex<=(startIndex+rangeBalancedIndices)) || (splitIndex >=(endIndex-1-rangeBalancedIndices)));

	if (unbalanced)
	{
		splitIndex = startIndex+ (numIndices>>1);
	}

	btAssert(!((splitIndex==startIndex) || (splitIndex == (endIndex))));

	return splitIndex;
}


void GIM_BOX_TREE::_build_sub_tree(gim_array<GIM_AABB_DATA> & primitive_boxes, GUINT startIndex,  GUINT endIndex)
{
	GUINT current_index = m_num_nodes++;

	btAssert((endIndex-startIndex)>0);

	if((endIndex-startIndex) == 1) //we got a leaf
	{		
		m_node_array[current_index].m_left = 0;
		m_node_array[current_index].m_right = 0;
		m_node_array[current_index].m_escapeIndex = 0;

		m_node_array[current_index].m_bound = primitive_boxes[startIndex].m_bound;
		m_node_array[current_index].m_data = primitive_boxes[startIndex].m_data;
		return;
	}

	//configure inner node

	GUINT splitIndex;

	//calc this node bounding box
	m_node_array[current_index].m_bound.invalidate();	
	for (splitIndex=startIndex;splitIndex<endIndex;splitIndex++)
	{
		m_node_array[current_index].m_bound.merge(primitive_boxes[splitIndex].m_bound);
	}

	//calculate Best Splitting Axis and where to split it. Sort the incoming 'leafNodes' array within range 'startIndex/endIndex'.

	//split axis
	splitIndex = _calc_splitting_axis(primitive_boxes,startIndex,endIndex);

	splitIndex = _sort_and_calc_splitting_index(
			primitive_boxes,startIndex,endIndex,splitIndex);

	//configure this inner node : the left node index
	m_node_array[current_index].m_left = m_num_nodes;
	//build left child tree
	_build_sub_tree(primitive_boxes, startIndex, splitIndex );

	//configure this inner node : the right node index
	m_node_array[current_index].m_right = m_num_nodes;

	//build right child tree
	_build_sub_tree(primitive_boxes, splitIndex ,endIndex);

	//configure this inner node : the escape index
	m_node_array[current_index].m_escapeIndex  = m_num_nodes - current_index;
}

//! stackless build tree
void GIM_BOX_TREE::build_tree(
	gim_array<GIM_AABB_DATA> & primitive_boxes)
{
	// initialize node count to 0
	m_num_nodes = 0;
	// allocate nodes
	m_node_array.resize(primitive_boxes.size()*2);
	
	_build_sub_tree(primitive_boxes, 0, primitive_boxes.size());
}


