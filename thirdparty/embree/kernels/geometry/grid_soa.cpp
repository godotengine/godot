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
 
#include "grid_soa.h"

namespace embree
{
  namespace isa
  {  
    GridSOA::GridSOA(const SubdivPatch1Base* patches, unsigned time_steps,
                     const unsigned x0, const unsigned x1, const unsigned y0, const unsigned y1, const unsigned swidth, const unsigned sheight,
                     const SubdivMesh* const geom, const size_t gridOffset, const size_t gridBytes, BBox3fa* bounds_o)
      : troot(BVH4::emptyNode),
        time_steps(time_steps), width(x1-x0+1), height(y1-y0+1), dim_offset(width*height),
        _geomID(patches->geomID()), _primID(patches->primID()), 
        gridOffset(unsigned(gridOffset)), gridBytes(unsigned(gridBytes)), rootOffset(unsigned(gridOffset+time_steps*gridBytes))
    {
      /* the generate loops need padded arrays, thus first store into these temporary arrays */
      unsigned temp_size = width*height+VSIZEX;
      dynamic_large_stack_array(float,local_grid_u,temp_size,32*32*sizeof(float));
      dynamic_large_stack_array(float,local_grid_v,temp_size,32*32*sizeof(float));
      dynamic_large_stack_array(float,local_grid_x,temp_size,32*32*sizeof(float));
      dynamic_large_stack_array(float,local_grid_y,temp_size,32*32*sizeof(float));
      dynamic_large_stack_array(float,local_grid_z,temp_size,32*32*sizeof(float));
      dynamic_large_stack_array(int,local_grid_uv,temp_size,32*32*sizeof(int));

      /* first create the grids for each time step */
      for (size_t t=0; t<time_steps; t++)
      {
        /* compute vertex grid (+displacement) */
        evalGrid(patches[t],x0,x1,y0,y1,swidth,sheight,
                 local_grid_x,local_grid_y,local_grid_z,local_grid_u,local_grid_v,geom);
        
        /* encode UVs */
        for (unsigned i=0; i<dim_offset; i+=VSIZEX) {
          const vintx iu = (vintx) clamp(vfloatx::load(&local_grid_u[i])*(0x10000/8.0f), vfloatx(0.0f), vfloatx(0xFFFF));
          const vintx iv = (vintx) clamp(vfloatx::load(&local_grid_v[i])*(0x10000/8.0f), vfloatx(0.0f), vfloatx(0xFFFF));
          vintx::storeu(&local_grid_uv[i], (iv << 16) | iu);
        }

        /* copy temporary data to compact grid */
        float* const grid_x  = (float*)(gridData(t) + 0*dim_offset);
        float* const grid_y  = (float*)(gridData(t) + 1*dim_offset);
        float* const grid_z  = (float*)(gridData(t) + 2*dim_offset);
        int  * const grid_uv = (int*  )(gridData(t) + 3*dim_offset);
	
	for (size_t i=0; i<width*height; i++)
	{
	  grid_x[i]  = local_grid_x[i];
	  grid_y[i]  = local_grid_y[i];
	  grid_z[i]  = local_grid_z[i];
	  grid_uv[i] = local_grid_uv[i];
	}
      }

      /* create normal BVH when no motion blur is active */
      if (time_steps == 1)
        root(0) = buildBVH(bounds_o).first;

      /* otherwise build MBlur BVH */
      else {
        BBox3fa gbounds[RTC_MAX_TIME_STEP_COUNT];
        troot = buildMSMBlurBVH(make_range(0,int(time_steps-1)),gbounds).first;

        if (bounds_o)
          for (size_t i=0; i<time_steps; i++) 
            bounds_o[i] = gbounds[i];
      }
    }

    size_t GridSOA::getBVHBytes(const GridRange& range, const size_t nodeBytes, const size_t leafBytes)
    {
      if (range.hasLeafSize()) 
        return leafBytes;
      
      __aligned(64) GridRange r[4];
      const size_t children = range.splitIntoSubRanges(r);
      
      size_t bytes = nodeBytes;
      for (size_t i=0; i<children; i++)
        bytes += getBVHBytes(r[i],nodeBytes,leafBytes);
      return bytes;
    }

    size_t GridSOA::getTemporalBVHBytes(const range<int> time_range, const size_t nodeBytes)
    {
      if (time_range.size() <= 1)
        return 0;

      size_t bytes = nodeBytes;
      for (int i=0; i<4; i++) {
        const int begin = time_range.begin() + (i+0)*time_range.size()/4;
        const int end   = time_range.begin() + (i+1)*time_range.size()/4;
        bytes += getTemporalBVHBytes(make_range(begin,end),nodeBytes);
      }
      return bytes;
    }

    std::pair<BVH4::NodeRef,BBox3fa> GridSOA::buildBVH(const GridRange& range, size_t& allocator)
    {
      /*! create leaf node */
      if (unlikely(range.hasLeafSize()))
      {
        /* we store index of first subgrid vertex as leaf node */
        BVH4::NodeRef curNode = BVH4::encodeTypedLeaf(encodeLeaf(range.u_start,range.v_start),0);

        /* return bounding box */
        return std::make_pair(curNode,calculateBounds(0,range));
      }
      
      /* create internal node */
      else 
      {
        /* allocate new bvh4 node */
        BVH4::AlignedNode* node = (BVH4::AlignedNode *)&bvhData()[allocator];
        allocator += sizeof(BVH4::AlignedNode);
        node->clear();
        
        /* split range */
        GridRange r[4];
        const unsigned children = range.splitIntoSubRanges(r);
      
        /* recurse into subtrees */
        BBox3fa bounds( empty );
        for (unsigned i=0; i<children; i++)
        {
          std::pair<BVH4::NodeRef,BBox3fa> node_bounds = buildBVH(r[i], allocator);
          node->set(i,node_bounds.first,node_bounds.second);
          bounds.extend(node_bounds.second);
        }
        assert(is_finite(bounds));
        return std::make_pair(BVH4::encodeNode(node),bounds);
      }
    }

    std::pair<BVH4::NodeRef,BBox3fa> GridSOA::buildBVH(BBox3fa* bounds_o)
    {
      size_t allocator = 0;
      GridRange range(0,width-1,0,height-1);
      std::pair<BVH4::NodeRef,BBox3fa> root_bounds = buildBVH(range,allocator);
      if (bounds_o) *bounds_o = root_bounds.second;
      assert(allocator == gridOffset);
      return root_bounds;
    }

    std::pair<BVH4::NodeRef,LBBox3fa> GridSOA::buildMBlurBVH(size_t time, const GridRange& range, size_t& allocator)
    {
      /*! create leaf node */
      if (unlikely(range.hasLeafSize()))
      {
        /* we store index of first subgrid vertex as leaf node */
        BVH4::NodeRef curNode = BVH4::encodeTypedLeaf(encodeLeaf(range.u_start,range.v_start),0);

        /* return bounding box */
        const BBox3fa b0 = calculateBounds(time+0,range);
        const BBox3fa b1 = calculateBounds(time+1,range);
        return std::make_pair(curNode,LBBox3fa(b0,b1));
      }
      
      /* create internal node */
      else 
      {
        /* allocate new bvh4 node */
        BVH4::AlignedNodeMB* node = (BVH4::AlignedNodeMB *)&bvhData()[allocator];
        allocator += sizeof(BVH4::AlignedNodeMB);
        node->clear();
        
        /* split range */
        GridRange r[4];
        const unsigned children = range.splitIntoSubRanges(r);
      
        /* recurse into subtrees */
        LBBox3fa bounds(empty);
        for (unsigned i=0; i<children; i++)
        {
          const BBox1f time_range(float(time+0)/float(time_steps-1),
                                  float(time+1)/float(time_steps-1));
          std::pair<BVH4::NodeRef,LBBox3fa> node_bounds = buildMBlurBVH(time, r[i], allocator);
          node->setRef(i,node_bounds.first);
          node->setBounds(i,node_bounds.second.global(time_range));
          bounds.extend(node_bounds.second);
        }
        assert(is_finite(bounds.bounds0));
        assert(is_finite(bounds.bounds1));
        
        return std::make_pair(BVH4::encodeNode(node),bounds);
      }
    }

    std::pair<BVH4::NodeRef,LBBox3fa> GridSOA::buildMSMBlurBVH(const range<int> time_range, size_t& allocator, BBox3fa* bounds_o)
    {
      assert(time_range.size() > 0);
      if (time_range.size() == 1) 
      {
        size_t t = time_range.begin();
        GridRange range(0,width-1,0,height-1);
        std::pair<BVH4::NodeRef,LBBox3fa> root_bounds = buildMBlurBVH(t,range,allocator);
        root(t) = root_bounds.first;
        bounds_o[t+0] = root_bounds.second.bounds0;
        bounds_o[t+1] = root_bounds.second.bounds1;
        return root_bounds;
      }

      /* allocate new bvh4 node */
      BVH4::AlignedNodeMB4D* node = (BVH4::AlignedNodeMB4D*)&bvhData()[allocator];
      allocator += sizeof(BVH4::AlignedNodeMB4D);
      node->clear();

      for (int i=0, j=0; i<4; i++) 
      {
        const int begin = time_range.begin() + (i+0)*time_range.size()/4;
        const int end   = time_range.begin() + (i+1)*time_range.size()/4;
        if (end-begin <= 0) continue;
        std::pair<BVH4::NodeRef,LBBox3fa> node_bounds = buildMSMBlurBVH(make_range(begin,end),allocator,bounds_o);
        const float t0 = float(begin)/float(time_steps-1);
        const float t1 = float(end  )/float(time_steps-1);
        node->set(j,node_bounds.first,node_bounds.second,BBox1f(t0,t1));
        j++;
      }

      const LBBox3fa lbounds = LBBox3fa([&] ( int i ) { return bounds_o[i]; }, time_range, time_steps-1);
      return std::make_pair(BVH4::encodeNode(node),lbounds);
    }

    std::pair<BVH4::NodeRef,LBBox3fa> GridSOA::buildMSMBlurBVH(const range<int> time_range, BBox3fa* bounds_o)
    {
      size_t allocator = 0;
      std::pair<BVH4::NodeRef,LBBox3fa> root = buildMSMBlurBVH(time_range,allocator,bounds_o);
      assert(allocator == gridOffset);
      return root;
    }
  }
}
