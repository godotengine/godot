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

#include "subdivpatch1base.h"

namespace embree
{
  namespace isa
  {
    Vec3fa patchEval(const SubdivPatch1Base& patch, const float uu, const float vv) 
    {
      if (likely(patch.type == SubdivPatch1Base::BEZIER_PATCH))
        return ((BezierPatch3fa*)patch.patch_v)->eval(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BSPLINE_PATCH))
        return ((BSplinePatch3fa*)patch.patch_v)->eval(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::GREGORY_PATCH))
        return ((DenseGregoryPatch3fa*)patch.patch_v)->eval(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BILINEAR_PATCH))
        return ((BilinearPatch3fa*)patch.patch_v)->eval(uu,vv);
      return Vec3fa( zero );
    }

    Vec3fa patchNormal(const SubdivPatch1Base& patch, const float uu, const float vv) 
    {
      if (likely(patch.type == SubdivPatch1Base::BEZIER_PATCH))
        return ((BezierPatch3fa*)patch.patch_v)->normal(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BSPLINE_PATCH))
        return ((BSplinePatch3fa*)patch.patch_v)->normal(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::GREGORY_PATCH))
        return ((DenseGregoryPatch3fa*)patch.patch_v)->normal(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BILINEAR_PATCH))
        return ((BilinearPatch3fa*)patch.patch_v)->normal(uu,vv);
      return Vec3fa( zero );
    }

    template<typename simdf>
      Vec3<simdf> patchEval(const SubdivPatch1Base& patch, const simdf& uu, const simdf& vv) 
    {
      if (likely(patch.type == SubdivPatch1Base::BEZIER_PATCH))
        return ((BezierPatch3fa*)patch.patch_v)->eval(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BSPLINE_PATCH))
        return ((BSplinePatch3fa*)patch.patch_v)->eval(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::GREGORY_PATCH))
        return ((DenseGregoryPatch3fa*)patch.patch_v)->eval(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BILINEAR_PATCH))
        return ((BilinearPatch3fa*)patch.patch_v)->eval(uu,vv);
      return Vec3<simdf>( zero );
    }

    template<typename simdf>
      Vec3<simdf> patchNormal(const SubdivPatch1Base& patch, const simdf& uu, const simdf& vv) 
    {
      if (likely(patch.type == SubdivPatch1Base::BEZIER_PATCH))
        return ((BezierPatch3fa*)patch.patch_v)->normal(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BSPLINE_PATCH))
        return ((BSplinePatch3fa*)patch.patch_v)->normal(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::GREGORY_PATCH))
        return ((DenseGregoryPatch3fa*)patch.patch_v)->normal(uu,vv);
      else if (likely(patch.type == SubdivPatch1Base::BILINEAR_PATCH))
        return ((BilinearPatch3fa*)patch.patch_v)->normal(uu,vv);
      return Vec3<simdf>( zero );
    }

    /* eval grid over patch and stich edges when required */      
    void evalGrid(const SubdivPatch1Base& patch,
                  const unsigned x0, const unsigned x1,
                  const unsigned y0, const unsigned y1,
                  const unsigned swidth, const unsigned sheight,
                  float *__restrict__ const grid_x,
                  float *__restrict__ const grid_y,
                  float *__restrict__ const grid_z,
                  float *__restrict__ const grid_u,
                  float *__restrict__ const grid_v,
                  const SubdivMesh* const geom)
    {
      const unsigned dwidth  = x1-x0+1;
      const unsigned dheight = y1-y0+1;
      const unsigned M = dwidth*dheight+VSIZEX;
      const unsigned grid_size_simd_blocks = (M-1)/VSIZEX;

      if (unlikely(patch.type == SubdivPatch1Base::EVAL_PATCH))
      {
        const bool displ = geom->displFunc;
        const unsigned N = displ ? M : 0;
        dynamic_large_stack_array(float,grid_Ng_x,N,32*32*sizeof(float));
        dynamic_large_stack_array(float,grid_Ng_y,N,32*32*sizeof(float));
        dynamic_large_stack_array(float,grid_Ng_z,N,32*32*sizeof(float));
        
        if (geom->patch_eval_trees.size())
        {
          feature_adaptive_eval_grid<PatchEvalGrid> 
            (geom->patch_eval_trees[geom->numTimeSteps*patch.primID()+patch.time()], patch.subPatch(), patch.needsStitching() ? patch.level : nullptr,
             x0,x1,y0,y1,swidth,sheight,
             grid_x,grid_y,grid_z,grid_u,grid_v,
             displ ? (float*)grid_Ng_x : nullptr, displ ? (float*)grid_Ng_y : nullptr, displ ? (float*)grid_Ng_z : nullptr,
             dwidth,dheight);
        }
        else 
        {
          GeneralCatmullClarkPatch3fa ccpatch(patch.edge(),geom->getVertexBuffer(patch.time()));
          
          feature_adaptive_eval_grid<FeatureAdaptiveEvalGrid,GeneralCatmullClarkPatch3fa> 
            (ccpatch, patch.subPatch(), patch.needsStitching() ? patch.level : nullptr,
            x0,x1,y0,y1,swidth,sheight,
            grid_x,grid_y,grid_z,grid_u,grid_v,
            displ ? (float*)grid_Ng_x : nullptr, displ ? (float*)grid_Ng_y : nullptr, displ ? (float*)grid_Ng_z : nullptr,
            dwidth,dheight);
        }

        /* convert sub-patch UVs to patch UVs*/
        const Vec2f uv0 = patch.getUV(0);
        const Vec2f uv1 = patch.getUV(1);
        const Vec2f uv2 = patch.getUV(2);
        const Vec2f uv3 = patch.getUV(3);
        for (unsigned i=0; i<grid_size_simd_blocks; i++)
        {
          const vfloatx u = vfloatx::load(&grid_u[i*VSIZEX]);
          const vfloatx v = vfloatx::load(&grid_v[i*VSIZEX]);
          const vfloatx patch_u = lerp2(uv0.x,uv1.x,uv3.x,uv2.x,u,v);
          const vfloatx patch_v = lerp2(uv0.y,uv1.y,uv3.y,uv2.y,u,v);
          vfloatx::store(&grid_u[i*VSIZEX],patch_u);
          vfloatx::store(&grid_v[i*VSIZEX],patch_v);
        }

        /* call displacement shader */
        if (unlikely(geom->displFunc)) {
          RTCDisplacementFunctionNArguments args;
          args.geometryUserPtr = geom->userPtr;
          args.geometry = (RTCGeometry)geom;
          //args.geomID = patch.geomID();
          args.primID = patch.primID();
          args.timeStep = patch.time();
          args.u = grid_u;
          args.v = grid_v;
          args.Ng_x = grid_Ng_x;
          args.Ng_y = grid_Ng_y;
          args.Ng_z = grid_Ng_z;
          args.P_x = grid_x;
          args.P_y = grid_y;
          args.P_z = grid_z;
          args.N = dwidth*dheight;
          geom->displFunc(&args);
        }

        /* set last elements in u,v array to 1.0f */
        const float last_u = grid_u[dwidth*dheight-1];
        const float last_v = grid_v[dwidth*dheight-1];
        const float last_x = grid_x[dwidth*dheight-1];
        const float last_y = grid_y[dwidth*dheight-1];
        const float last_z = grid_z[dwidth*dheight-1];
        for (unsigned i=dwidth*dheight;i<grid_size_simd_blocks*VSIZEX;i++)
        {
          grid_u[i] = last_u;
          grid_v[i] = last_v;
          grid_x[i] = last_x;
          grid_y[i] = last_y;
          grid_z[i] = last_z;
        }
      }
      else
      {
        /* grid_u, grid_v need to be padded as we write with SIMD granularity */
        gridUVTessellator(patch.level,swidth,sheight,x0,y0,dwidth,dheight,grid_u,grid_v);
      
        /* set last elements in u,v array to last valid point */
        const float last_u = grid_u[dwidth*dheight-1];
        const float last_v = grid_v[dwidth*dheight-1];
        for (unsigned i=dwidth*dheight;i<grid_size_simd_blocks*VSIZEX;i++) {
          grid_u[i] = last_u;
          grid_v[i] = last_v;
        }

        /* stitch edges if necessary */
        if (unlikely(patch.needsStitching()))
          stitchUVGrid(patch.level,swidth,sheight,x0,y0,dwidth,dheight,grid_u,grid_v);
      
        /* iterates over all grid points */
        for (unsigned i=0; i<grid_size_simd_blocks; i++)
        {
          const vfloatx u = vfloatx::load(&grid_u[i*VSIZEX]);
          const vfloatx v = vfloatx::load(&grid_v[i*VSIZEX]);
          Vec3vfx vtx = patchEval(patch,u,v);
        
          /* evaluate displacement function */
          if (unlikely(geom->displFunc != nullptr))
          {
            const Vec3vfx normal = normalize_safe(patchNormal(patch, u, v));
            RTCDisplacementFunctionNArguments args;
            args.geometryUserPtr = geom->userPtr;
            args.geometry = (RTCGeometry)geom;
            //args.geomID = patch.geomID();
            args.primID = patch.primID();
            args.timeStep = patch.time();
            args.u = &u[0];
            args.v = &v[0];
            args.Ng_x = &normal.x[0];
            args.Ng_y = &normal.y[0];
            args.Ng_z = &normal.z[0];
            args.P_x = &vtx.x[0];
            args.P_y = &vtx.y[0];
            args.P_z = &vtx.z[0];
            args.N = VSIZEX;
            geom->displFunc(&args);
          }

          vfloatx::store(&grid_x[i*VSIZEX],vtx.x);
          vfloatx::store(&grid_y[i*VSIZEX],vtx.y);
          vfloatx::store(&grid_z[i*VSIZEX],vtx.z);
        }
      }
    }


    /* eval grid over patch and stich edges when required */      
    BBox3fa evalGridBounds(const SubdivPatch1Base& patch,
                           const unsigned x0, const unsigned x1,
                           const unsigned y0, const unsigned y1,
                           const unsigned swidth, const unsigned sheight,
                           const SubdivMesh* const geom)
    {
      BBox3fa b(empty);
      const unsigned dwidth  = x1-x0+1;
      const unsigned dheight = y1-y0+1;
      const unsigned M = dwidth*dheight+VSIZEX;
      const unsigned grid_size_simd_blocks = (M-1)/VSIZEX;
      dynamic_large_stack_array(float,grid_u,M,64*64*sizeof(float));
      dynamic_large_stack_array(float,grid_v,M,64*64*sizeof(float));

      if (unlikely(patch.type == SubdivPatch1Base::EVAL_PATCH))
      {
        const bool displ = geom->displFunc;
        dynamic_large_stack_array(float,grid_x,M,64*64*sizeof(float));
        dynamic_large_stack_array(float,grid_y,M,64*64*sizeof(float));
        dynamic_large_stack_array(float,grid_z,M,64*64*sizeof(float));
        dynamic_large_stack_array(float,grid_Ng_x,displ ? M : 0,64*64*sizeof(float));
        dynamic_large_stack_array(float,grid_Ng_y,displ ? M : 0,64*64*sizeof(float));
        dynamic_large_stack_array(float,grid_Ng_z,displ ? M : 0,64*64*sizeof(float));

        if (geom->patch_eval_trees.size())
        {
          feature_adaptive_eval_grid<PatchEvalGrid> 
            (geom->patch_eval_trees[geom->numTimeSteps*patch.primID()+patch.time()], patch.subPatch(), patch.needsStitching() ? patch.level : nullptr,
             x0,x1,y0,y1,swidth,sheight,
             grid_x,grid_y,grid_z,grid_u,grid_v,
             displ ? (float*)grid_Ng_x : nullptr, displ ? (float*)grid_Ng_y : nullptr, displ ? (float*)grid_Ng_z : nullptr,
             dwidth,dheight);
        } 
        else 
        {
          GeneralCatmullClarkPatch3fa ccpatch(patch.edge(),geom->getVertexBuffer(patch.time()));
          
          feature_adaptive_eval_grid <FeatureAdaptiveEvalGrid,GeneralCatmullClarkPatch3fa>
            (ccpatch, patch.subPatch(), patch.needsStitching() ? patch.level : nullptr,
            x0,x1,y0,y1,swidth,sheight,
            grid_x,grid_y,grid_z,grid_u,grid_v,
            displ ? (float*)grid_Ng_x : nullptr, displ ? (float*)grid_Ng_y : nullptr, displ ? (float*)grid_Ng_z : nullptr,
            dwidth,dheight);
        }

        /* call displacement shader */
        if (unlikely(geom->displFunc))
        {
          RTCDisplacementFunctionNArguments args;
          args.geometryUserPtr = geom->userPtr;
          args.geometry = (RTCGeometry)geom;
          //args.geomID = patch.geomID();
          args.primID = patch.primID();
          args.timeStep = patch.time();
          args.u = grid_u;
          args.v = grid_v;
          args.Ng_x = grid_Ng_x;
          args.Ng_y = grid_Ng_y;
          args.Ng_z = grid_Ng_z;
          args.P_x = grid_x;
          args.P_y = grid_y;
          args.P_z = grid_z;
          args.N = dwidth*dheight;
          geom->displFunc(&args);
        }

        /* set last elements in u,v array to 1.0f */
        const float last_u = grid_u[dwidth*dheight-1];
        const float last_v = grid_v[dwidth*dheight-1];
        const float last_x = grid_x[dwidth*dheight-1];
        const float last_y = grid_y[dwidth*dheight-1];
        const float last_z = grid_z[dwidth*dheight-1];
        for (unsigned i=dwidth*dheight;i<grid_size_simd_blocks*VSIZEX;i++)
        {
          grid_u[i] = last_u;
          grid_v[i] = last_v;
          grid_x[i] = last_x;
          grid_y[i] = last_y;
          grid_z[i] = last_z;
        }

        vfloatx bounds_min_x = pos_inf;
        vfloatx bounds_min_y = pos_inf;
        vfloatx bounds_min_z = pos_inf;
        vfloatx bounds_max_x = neg_inf;
        vfloatx bounds_max_y = neg_inf;
        vfloatx bounds_max_z = neg_inf;
        for (unsigned i = 0; i<grid_size_simd_blocks; i++)
        {
          vfloatx x = vfloatx::loadu(&grid_x[i * VSIZEX]);
          vfloatx y = vfloatx::loadu(&grid_y[i * VSIZEX]);
          vfloatx z = vfloatx::loadu(&grid_z[i * VSIZEX]);

	  bounds_min_x = min(bounds_min_x,x);
	  bounds_min_y = min(bounds_min_y,y);
	  bounds_min_z = min(bounds_min_z,z);

	  bounds_max_x = max(bounds_max_x,x);
	  bounds_max_y = max(bounds_max_y,y);
	  bounds_max_z = max(bounds_max_z,z);
        }

        b.lower.x = reduce_min(bounds_min_x);  
        b.lower.y = reduce_min(bounds_min_y);
        b.lower.z = reduce_min(bounds_min_z);
        b.upper.x = reduce_max(bounds_max_x);
        b.upper.y = reduce_max(bounds_max_y);
        b.upper.z = reduce_max(bounds_max_z);
        b.lower.a = 0;
        b.upper.a = 0;
      }
      else
      {
        /* grid_u, grid_v need to be padded as we write with SIMD granularity */
        gridUVTessellator(patch.level,swidth,sheight,x0,y0,dwidth,dheight,grid_u,grid_v);
      
        /* set last elements in u,v array to last valid point */
        const float last_u = grid_u[dwidth*dheight-1];
        const float last_v = grid_v[dwidth*dheight-1];
        for (unsigned i=dwidth*dheight;i<grid_size_simd_blocks*VSIZEX;i++) {
          grid_u[i] = last_u;
          grid_v[i] = last_v;
        }

        /* stitch edges if necessary */
        if (unlikely(patch.needsStitching()))
          stitchUVGrid(patch.level,swidth,sheight,x0,y0,dwidth,dheight,grid_u,grid_v);
      
        /* iterates over all grid points */
        Vec3vfx bounds_min;
        bounds_min[0] = pos_inf;
        bounds_min[1] = pos_inf;
        bounds_min[2] = pos_inf;

        Vec3vfx bounds_max;
        bounds_max[0] = neg_inf;
        bounds_max[1] = neg_inf;
        bounds_max[2] = neg_inf;

        for (unsigned i=0; i<grid_size_simd_blocks; i++)
        {
          const vfloatx u = vfloatx::load(&grid_u[i*VSIZEX]);
          const vfloatx v = vfloatx::load(&grid_v[i*VSIZEX]);
          Vec3vfx vtx = patchEval(patch,u,v);
        
          /* evaluate displacement function */
          if (unlikely(geom->displFunc != nullptr))
          {
            const Vec3vfx normal = normalize_safe(patchNormal(patch,u,v));
            RTCDisplacementFunctionNArguments args;
            args.geometryUserPtr = geom->userPtr;
            args.geometry = (RTCGeometry)geom;
            //args.geomID = patch.geomID();
            args.primID = patch.primID();
            args.timeStep = patch.time();
            args.u = &u[0];
            args.v = &v[0];
            args.Ng_x = &normal.x[0];
            args.Ng_y = &normal.y[0];
            args.Ng_z = &normal.z[0];
            args.P_x = &vtx.x[0];
            args.P_y = &vtx.y[0];
            args.P_z = &vtx.z[0];
            args.N = VSIZEX;
            geom->displFunc(&args);
          }

          bounds_min[0] = min(bounds_min[0],vtx.x);
          bounds_max[0] = max(bounds_max[0],vtx.x);
          bounds_min[1] = min(bounds_min[1],vtx.y);
          bounds_max[1] = max(bounds_max[1],vtx.y);
          bounds_min[2] = min(bounds_min[2],vtx.z);
          bounds_max[2] = max(bounds_max[2],vtx.z);      
        }

        b.lower.x = reduce_min(bounds_min[0]);
        b.lower.y = reduce_min(bounds_min[1]);
        b.lower.z = reduce_min(bounds_min[2]);
        b.upper.x = reduce_max(bounds_max[0]);
        b.upper.y = reduce_max(bounds_max[1]);
        b.upper.z = reduce_max(bounds_max[2]);
        b.lower.a = 0;
        b.upper.a = 0;
      }

      assert( std::isfinite(b.lower.x) );
      assert( std::isfinite(b.lower.y) );
      assert( std::isfinite(b.lower.z) );

      assert( std::isfinite(b.upper.x) );
      assert( std::isfinite(b.upper.y) );
      assert( std::isfinite(b.upper.z) );


      assert(b.lower.x <= b.upper.x);
      assert(b.lower.y <= b.upper.y);
      assert(b.lower.z <= b.upper.z);
      return b;
    }
  }
}
