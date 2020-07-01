// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "bvh_rotate.h"

namespace embree
{
  namespace isa 
  {
    /*! Computes half surface area of box. */
    __forceinline float halfArea3f(const BBox<vfloat4>& box) {
      const vfloat4 d = box.size();
      const vfloat4 a = d*shuffle<1,2,0,3>(d);
      return a[0]+a[1]+a[2];
    }
    
    size_t BVHNRotate<4>::rotate(NodeRef parentRef, size_t depth)
    {
      /*! nothing to rotate if we reached a leaf node. */
      if (parentRef.isBarrier()) return 0;
      if (parentRef.isLeaf()) return 0;
      AlignedNode* parent = parentRef.alignedNode();
      
      /*! rotate all children first */
      vint4 cdepth;
      for (size_t c=0; c<4; c++)
	cdepth[c] = (int)rotate(parent->child(c),depth+1);
      
      /* compute current areas of all children */
      vfloat4 sizeX = parent->upper_x-parent->lower_x;
      vfloat4 sizeY = parent->upper_y-parent->lower_y;
      vfloat4 sizeZ = parent->upper_z-parent->lower_z;
      vfloat4 childArea = madd(sizeX,(sizeY + sizeZ),sizeY*sizeZ);
      
      /*! get node bounds */
      BBox<vfloat4> child1_0,child1_1,child1_2,child1_3;
      parent->bounds(child1_0,child1_1,child1_2,child1_3);
      
      /*! Find best rotation. We pick a first child (child1) and a sub-child 
	(child2child) of a different second child (child2), and swap child1 
	and child2child. We perform the best such swap. */
      float bestArea = 0;
      size_t bestChild1 = -1, bestChild2 = -1, bestChild2Child = -1;
      for (size_t c2=0; c2<4; c2++)
      {
	/*! ignore leaf nodes as we cannot descent into them */
	if (parent->child(c2).isBarrier()) continue;
	if (parent->child(c2).isLeaf()) continue;
	AlignedNode* child2 = parent->child(c2).alignedNode();
	
	/*! transpose child bounds */
	BBox<vfloat4> child2c0,child2c1,child2c2,child2c3;
	child2->bounds(child2c0,child2c1,child2c2,child2c3);
	
	/*! put child1_0 at each child2 position */
	float cost00 = halfArea3f(merge(child1_0,child2c1,child2c2,child2c3));
	float cost01 = halfArea3f(merge(child2c0,child1_0,child2c2,child2c3));
	float cost02 = halfArea3f(merge(child2c0,child2c1,child1_0,child2c3));
	float cost03 = halfArea3f(merge(child2c0,child2c1,child2c2,child1_0));
	vfloat4 cost0 = vfloat4(cost00,cost01,cost02,cost03);
	vfloat4 min0 = vreduce_min(cost0);
	int pos0 = (int)bsf(movemask(min0 == cost0));
	
	/*! put child1_1 at each child2 position */
	float cost10 = halfArea3f(merge(child1_1,child2c1,child2c2,child2c3));
	float cost11 = halfArea3f(merge(child2c0,child1_1,child2c2,child2c3));
	float cost12 = halfArea3f(merge(child2c0,child2c1,child1_1,child2c3));
	float cost13 = halfArea3f(merge(child2c0,child2c1,child2c2,child1_1));
	vfloat4 cost1 = vfloat4(cost10,cost11,cost12,cost13);
	vfloat4 min1 = vreduce_min(cost1);
	int pos1 = (int)bsf(movemask(min1 == cost1));
	
	/*! put child1_2 at each child2 position */
	float cost20 = halfArea3f(merge(child1_2,child2c1,child2c2,child2c3));
	float cost21 = halfArea3f(merge(child2c0,child1_2,child2c2,child2c3));
	float cost22 = halfArea3f(merge(child2c0,child2c1,child1_2,child2c3));
	float cost23 = halfArea3f(merge(child2c0,child2c1,child2c2,child1_2));
	vfloat4 cost2 = vfloat4(cost20,cost21,cost22,cost23);
	vfloat4 min2 = vreduce_min(cost2);
	int pos2 = (int)bsf(movemask(min2 == cost2));
	
	/*! put child1_3 at each child2 position */
	float cost30 = halfArea3f(merge(child1_3,child2c1,child2c2,child2c3));
	float cost31 = halfArea3f(merge(child2c0,child1_3,child2c2,child2c3));
	float cost32 = halfArea3f(merge(child2c0,child2c1,child1_3,child2c3));
	float cost33 = halfArea3f(merge(child2c0,child2c1,child2c2,child1_3));
	vfloat4 cost3 = vfloat4(cost30,cost31,cost32,cost33);
	vfloat4 min3 = vreduce_min(cost3);
	int pos3 = (int)bsf(movemask(min3 == cost3));
	
	/*! find best other child */
	vfloat4 area0123 = vfloat4(extract<0>(min0),extract<0>(min1),extract<0>(min2),extract<0>(min3)) - vfloat4(childArea[c2]);
	int pos[4] = { pos0,pos1,pos2,pos3 };
	const size_t mbd = BVH4::maxBuildDepth;
	vbool4 valid = vint4(int(depth+1))+cdepth <= vint4(mbd); // only select swaps that fulfill depth constraints
	valid &= vint4(int(c2)) != vint4(step);
	if (none(valid)) continue;
	size_t c1 = select_min(valid,area0123);
	float area = area0123[c1]; 
        if (c1 == c2) continue; // can happen if bounds are NANs
	
	/*! accept a swap when it reduces cost and is not swapping a node with itself */
	if (area < bestArea) {
	  bestArea = area;
	  bestChild1 = c1;
	  bestChild2 = c2;
	  bestChild2Child = pos[c1];
	}
      }
      
      /*! if we did not find a swap that improves the SAH then do nothing */
      if (bestChild1 == size_t(-1)) return 1+reduce_max(cdepth);
      
      /*! perform the best found tree rotation */
      AlignedNode* child2 = parent->child(bestChild2).alignedNode();
      BVH4::swap(parent,bestChild1,child2,bestChild2Child);
      parent->setBounds(bestChild2,child2->bounds());
      BVH4::compact(parent);
      BVH4::compact(child2);
      
      /*! This returned depth is conservative as the child that was
       *  pulled up in the tree could have been on the critical path. */
      cdepth[bestChild1]++; // bestChild1 was pushed down one level
      return 1+reduce_max(cdepth); 
    }
  }
}
