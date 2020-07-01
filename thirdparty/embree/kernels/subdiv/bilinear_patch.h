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

#pragma once

#include "catmullclark_patch.h"
#include "bezier_curve.h"

namespace embree
{
  template<typename Vertex, typename Vertex_t = Vertex>
    class __aligned(64) BilinearPatchT
    {
      typedef CatmullClark1RingT<Vertex,Vertex_t> CatmullClarkRing;
      typedef CatmullClarkPatchT<Vertex,Vertex_t> CatmullClarkPatch;
      
    public:
      Vertex v[4];
      
    public:
      
      __forceinline BilinearPatchT () {}

      __forceinline BilinearPatchT (const HalfEdge* edge, const BufferView<Vertex>& vertices) {
        init(edge,vertices.getPtr(),vertices.getStride());
      }
      
      __forceinline BilinearPatchT (const HalfEdge* edge, const char* vertices, size_t stride) {
        init(edge,vertices,stride);
      }

      __forceinline void init (const HalfEdge* edge, const char* vertices, size_t stride)
      {
        v[0] = Vertex::loadu(vertices+edge->getStartVertexIndex()*stride); edge = edge->next();
        v[1] = Vertex::loadu(vertices+edge->getStartVertexIndex()*stride); edge = edge->next();
        v[2] = Vertex::loadu(vertices+edge->getStartVertexIndex()*stride); edge = edge->next();
        v[3] = Vertex::loadu(vertices+edge->getStartVertexIndex()*stride); edge = edge->next();
      }

      __forceinline BilinearPatchT (const CatmullClarkPatch& patch)
      {
        v[0] = patch.ring[0].getLimitVertex();
        v[1] = patch.ring[1].getLimitVertex();
        v[2] = patch.ring[2].getLimitVertex();
        v[3] = patch.ring[3].getLimitVertex();
      }

      __forceinline BBox<Vertex> bounds() const
      {
        
        BBox<Vertex> bounds (v[0]);
        bounds.extend(v[1]);
        bounds.extend(v[2]);
        bounds.extend(v[3]);
        return bounds;
      }
      
      __forceinline Vertex eval(const float uu, const float vv) const {
        return lerp(lerp(v[0],v[1],uu),lerp(v[3],v[2],uu),vv);
      }

      __forceinline Vertex eval_du(const float uu, const float vv) const {
        return lerp(v[1]-v[0],v[2]-v[3],vv);
      }

      __forceinline Vertex eval_dv(const float uu, const float vv) const {
        return lerp(v[3]-v[0],v[2]-v[1],uu);
      }

      __forceinline Vertex eval_dudu(const float uu, const float vv) const {
        return Vertex(zero);
      }

      __forceinline Vertex eval_dvdv(const float uu, const float vv) const {
        return Vertex(zero);
      }

      __forceinline Vertex eval_dudv(const float uu, const float vv) const {
        return (v[2]-v[3]) - (v[1]-v[0]);
      }

      __forceinline Vertex normal(const float uu, const float vv) const {
        return cross(eval_du(uu,vv),eval_dv(uu,vv));
      }
      
      __forceinline void eval(const float u, const float v, 
                              Vertex* P, Vertex* dPdu, Vertex* dPdv, Vertex* ddPdudu, Vertex* ddPdvdv, Vertex* ddPdudv,
                              const float dscale = 1.0f) const
      {
        if (P) {
          *P = eval(u,v); 
        }
        if (dPdu) {
          assert(dPdu); *dPdu = eval_du(u,v)*dscale; 
          assert(dPdv); *dPdv = eval_dv(u,v)*dscale; 
        }
        if (ddPdudu) {
          assert(ddPdudu); *ddPdudu = eval_dudu(u,v)*sqr(dscale); 
          assert(ddPdvdv); *ddPdvdv = eval_dvdv(u,v)*sqr(dscale); 
          assert(ddPdudv); *ddPdudv = eval_dudv(u,v)*sqr(dscale); 
        }
      }

      template<class vfloat>
      __forceinline Vec3<vfloat> eval(const vfloat& uu, const vfloat& vv) const
      {
        const vfloat x = lerp(lerp(v[0].x,v[1].x,uu),lerp(v[3].x,v[2].x,uu),vv);
        const vfloat y = lerp(lerp(v[0].y,v[1].y,uu),lerp(v[3].y,v[2].y,uu),vv);
        const vfloat z = lerp(lerp(v[0].z,v[1].z,uu),lerp(v[3].z,v[2].z,uu),vv);
        return Vec3<vfloat>(x,y,z);
      }

      template<class vfloat>
      __forceinline Vec3<vfloat> eval_du(const vfloat& uu, const vfloat& vv) const
      {
        const vfloat x = lerp(v[1].x-v[0].x,v[2].x-v[3].x,vv);
        const vfloat y = lerp(v[1].y-v[0].y,v[2].y-v[3].y,vv);
        const vfloat z = lerp(v[1].z-v[0].z,v[2].z-v[3].z,vv);
        return Vec3<vfloat>(x,y,z);
      }

      template<class vfloat>
      __forceinline Vec3<vfloat> eval_dv(const vfloat& uu, const vfloat& vv) const
      {
        const vfloat x = lerp(v[3].x-v[0].x,v[2].x-v[1].x,uu);
        const vfloat y = lerp(v[3].y-v[0].y,v[2].y-v[1].y,uu);
        const vfloat z = lerp(v[3].z-v[0].z,v[2].z-v[1].z,uu);
        return Vec3<vfloat>(x,y,z);
      }

      template<typename vfloat>
      __forceinline Vec3<vfloat> normal(const vfloat& uu, const vfloat& vv) const {
        return cross(eval_du(uu,vv),eval_dv(uu,vv));
      }

       template<class vfloat>
      __forceinline vfloat eval(const size_t i, const vfloat& uu, const vfloat& vv) const {
        return lerp(lerp(v[0][i],v[1][i],uu),lerp(v[3][i],v[2][i],uu),vv);
      }

      template<class vfloat>
      __forceinline vfloat eval_du(const size_t i, const vfloat& uu, const vfloat& vv) const {
        return lerp(v[1][i]-v[0][i],v[2][i]-v[3][i],vv);
      }

      template<class vfloat>
      __forceinline vfloat eval_dv(const size_t i, const vfloat& uu, const vfloat& vv) const {
        return lerp(v[3][i]-v[0][i],v[2][i]-v[1][i],uu);
      }
      
      template<class vfloat>
      __forceinline vfloat eval_dudu(const size_t i, const vfloat& uu, const vfloat& vv) const {
        return vfloat(zero);
      }

      template<class vfloat>
      __forceinline vfloat eval_dvdv(const size_t i, const vfloat& uu, const vfloat& vv) const {
        return vfloat(zero);
      }

      template<class vfloat>
      __forceinline vfloat eval_dudv(const size_t i, const vfloat& uu, const vfloat& vv) const {
        return (v[2][i]-v[3][i]) - (v[1][i]-v[0][i]);
      }

      template<typename vbool, typename vfloat>
      __forceinline void eval(const vbool& valid, const vfloat& uu, const vfloat& vv, 
                              float* P, float* dPdu, float* dPdv, float* ddPdudu, float* ddPdvdv, float* ddPdudv,
                              const float dscale, const size_t dstride, const size_t N) const
      {
        if (P) {
          for (size_t i=0; i<N; i++) vfloat::store(valid,P+i*dstride,eval(i,uu,vv));
        }
        if (dPdu) {
          for (size_t i=0; i<N; i++) {
            assert(dPdu); vfloat::store(valid,dPdu+i*dstride,eval_du(i,uu,vv)*dscale);
            assert(dPdv); vfloat::store(valid,dPdv+i*dstride,eval_dv(i,uu,vv)*dscale);
          }
        }
        if (ddPdudu) {
          for (size_t i=0; i<N; i++) {
            assert(ddPdudu); vfloat::store(valid,ddPdudu+i*dstride,eval_dudu(i,uu,vv)*sqr(dscale));
            assert(ddPdvdv); vfloat::store(valid,ddPdvdv+i*dstride,eval_dvdv(i,uu,vv)*sqr(dscale));
            assert(ddPdudv); vfloat::store(valid,ddPdudv+i*dstride,eval_dudv(i,uu,vv)*sqr(dscale));
          }
        }
      }
    };
  
  typedef BilinearPatchT<Vec3fa,Vec3fa_t> BilinearPatch3fa;
}
