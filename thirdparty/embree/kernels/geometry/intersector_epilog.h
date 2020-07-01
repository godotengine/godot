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

#include "../common/ray.h"
#include "../common/context.h"
#include "filter.h"

namespace embree
{
  namespace isa
  {
    template<int M>
    struct UVIdentity {
      __forceinline void operator() (vfloat<M>& u, vfloat<M>& v) const {}
    };

    template<bool filter>
    struct Intersect1Epilog1
    {
      RayHit& ray;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Intersect1Epilog1(RayHit& ray,
                                      IntersectContext* context,
                                      const unsigned int geomID,
                                      const unsigned int primID)
        : ray(ray), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (Hit& hit) const
      {
        /* ray mask test */
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        if ((geometry->mask & ray.mask) == 0) return false;
#endif
        hit.finalize();

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter())) {
            HitK<1> h(context->instID,geomID,primID,hit.u,hit.v,hit.Ng);
            const float old_t = ray.tfar;
            ray.tfar = hit.t;
            bool found = runIntersectionFilter1(geometry,ray,context,h);
            if (!found) ray.tfar = old_t;
            return found;
          }
        }
#endif

        /* update hit information */
        ray.tfar = hit.t;
        ray.Ng = hit.Ng;
        ray.u = hit.u;
        ray.v = hit.v;
        ray.primID = primID;
        ray.geomID = geomID;
        ray.instID = context->instID;
        return true;
      }
    };

    template<bool filter>
    struct Occluded1Epilog1
    {
      Ray& ray;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Occluded1Epilog1(Ray& ray,
                                     IntersectContext* context,
                                     const unsigned int geomID,
                                     const unsigned int primID)
        : ray(ray), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (Hit& hit) const
      {
        /* ray mask test */
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);


#if defined(EMBREE_RAY_MASK)
        if ((geometry->mask & ray.mask) == 0) return false;
#endif
        hit.finalize();

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter())) {
            HitK<1> h(context->instID,geomID,primID,hit.u,hit.v,hit.Ng);
            const float old_t = ray.tfar;
            ray.tfar = hit.t;
            const bool found = runOcclusionFilter1(geometry,ray,context,h);
            if (!found) ray.tfar = old_t;
            return found;
          }
        }
#endif
        return true;
      }
    };

    template<int K, bool filter>
    struct Intersect1KEpilog1
    {
      RayHitK<K>& ray;
      size_t k;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Intersect1KEpilog1(RayHitK<K>& ray, size_t k,
                                       IntersectContext* context,
                                       const unsigned int geomID,
                                       const unsigned int primID)
        : ray(ray), k(k), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (Hit& hit) const
      {
        /* ray mask test */
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        if ((geometry->mask & ray.mask[k]) == 0)
          return false;
#endif
        hit.finalize();

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter())) {
            HitK<K> h(context->instID,geomID,primID,hit.u,hit.v,hit.Ng);
            const float old_t = ray.tfar[k];
            ray.tfar[k] = hit.t;
            const bool found = any(runIntersectionFilter(vbool<K>(1<<k),geometry,ray,context,h));
            if (!found) ray.tfar[k] = old_t;
            return found;
          }
        }
#endif

        /* update hit information */
        ray.tfar[k] = hit.t;
        ray.Ng.x[k] = hit.Ng.x;
        ray.Ng.y[k] = hit.Ng.y;
        ray.Ng.z[k] = hit.Ng.z;
        ray.u[k] = hit.u;
        ray.v[k] = hit.v;
        ray.primID[k] = primID;
        ray.geomID[k] = geomID;
        ray.instID[k] = context->instID;
        return true;
      }
    };
    
    template<int K, bool filter>
    struct Occluded1KEpilog1
    {
      RayK<K>& ray;
      size_t k;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Occluded1KEpilog1(RayK<K>& ray, size_t k,
                                      IntersectContext* context,
                                      const unsigned int geomID,
                                      const unsigned int primID)
        : ray(ray), k(k), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (Hit& hit) const
      {
        /* ray mask test */
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        if ((geometry->mask & ray.mask[k]) == 0)
          return false;
#endif

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter())) {
            hit.finalize();
            HitK<K> h(context->instID,geomID,primID,hit.u,hit.v,hit.Ng);
            const float old_t = ray.tfar[k];
            ray.tfar[k] = hit.t;
            const bool found = any(runOcclusionFilter(vbool<K>(1<<k),geometry,ray,context,h));
            if (!found) ray.tfar[k] = old_t;
            return found;
          }
        }
#endif 
        return true;
      }
    };
    
    template<int M, int Mx, bool filter>
    struct Intersect1EpilogM
    {
      RayHit& ray;
      IntersectContext* context;
      const vuint<M>& geomIDs;
      const vuint<M>& primIDs;

      __forceinline Intersect1EpilogM(RayHit& ray,
                                      IntersectContext* context,
                                      const vuint<M>& geomIDs,
                                      const vuint<M>& primIDs)
        : ray(ray), context(context), geomIDs(geomIDs), primIDs(primIDs) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<Mx>& valid_i, Hit& hit) const
      {
        Scene* scene = context->scene;
        vbool<Mx> valid = valid_i;
        if (Mx > M) valid &= (1<<M)-1;
        hit.finalize();
        size_t i = select_min(valid,hit.vt);
        unsigned int geomID = geomIDs[i];

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION) || defined(EMBREE_RAY_MASK)
        bool foundhit = false;
        goto entry;
        while (true)
        {
          if (unlikely(none(valid))) return foundhit;
          i = select_min(valid,hit.vt);

          geomID = geomIDs[i];
        entry:
          Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

#if defined(EMBREE_RAY_MASK)
          /* goto next hit if mask test fails */
          if ((geometry->mask & ray.mask) == 0) {
            clear(valid,i);
            continue;
          }
#endif

#if defined(EMBREE_FILTER_FUNCTION) 
          /* call intersection filter function */
          if (filter) {
            if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter())) {
              const Vec2f uv = hit.uv(i);
              HitK<1> h(context->instID,geomID,primIDs[i],uv.x,uv.y,hit.Ng(i));
              const float old_t = ray.tfar;
              ray.tfar = hit.t(i);
              const bool found = runIntersectionFilter1(geometry,ray,context,h);
              if (!found) ray.tfar = old_t;
              foundhit |= found;
              clear(valid,i);
              valid &= hit.vt <= ray.tfar; // intersection filters may modify tfar value
              continue;
            }
          }
#endif
          break;
        }
#endif

        /* update hit information */
        const Vec2f uv = hit.uv(i);
        ray.tfar = hit.vt[i];
        ray.Ng.x = hit.vNg.x[i];
        ray.Ng.y = hit.vNg.y[i];
        ray.Ng.z = hit.vNg.z[i];
        ray.u = uv.x;
        ray.v = uv.y;
        ray.primID = primIDs[i];
        ray.geomID = geomID;
        ray.instID = context->instID;
        return true;

      }
    };

#if 0 && defined(__AVX512F__) // do not enable, this reduced frequency for BVH4
    template<int M, bool filter>
    struct Intersect1EpilogM<M,16,filter>
    {
      static const size_t Mx = 16;
      RayHit& ray;
      IntersectContext* context;
      const vuint<M>& geomIDs;
      const vuint<M>& primIDs;

      __forceinline Intersect1EpilogM(RayHit& ray,
                                      IntersectContext* context,
                                      const vuint<M>& geomIDs,
                                      const vuint<M>& primIDs)
        : ray(ray), context(context), geomIDs(geomIDs), primIDs(primIDs) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<Mx>& valid_i, Hit& hit) const
      {
        Scene* scene = context->scene;
        vbool<Mx> valid = valid_i;
        if (Mx > M) valid &= (1<<M)-1;
        hit.finalize();
        size_t i = select_min(valid,hit.vt);
        unsigned int geomID = geomIDs[i];

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION) || defined(EMBREE_RAY_MASK)
        bool foundhit = false;
        goto entry;
        while (true)
        {
          if (unlikely(none(valid))) return foundhit;
          i = select_min(valid,hit.vt);

          geomID = geomIDs[i];
        entry:
          Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

#if defined(EMBREE_RAY_MASK)
          /* goto next hit if mask test fails */
          if ((geometry->mask & ray.mask) == 0) {
            clear(valid,i);
            continue;
          }
#endif

#if defined(EMBREE_FILTER_FUNCTION) 
          /* call intersection filter function */
          if (filter) {
            if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter())) {
              const Vec2f uv = hit.uv(i);
              HitK<1> h(context->instID,geomID,primIDs[i],uv.x,uv.y,hit.Ng(i));
              const float old_t = ray.tfar;
              ray.tfar = hit.t(i);
              const bool found = runIntersectionFilter1(geometry,ray,context,h);
              if (!found) ray.tfar = old_t;
              foundhit |= found;
              clear(valid,i);
              valid &= hit.vt <= ray.tfar; // intersection filters may modify tfar value
              continue;
            }
          }
#endif
          break;
        }
#endif

        vbool<Mx> finalMask(((unsigned int)1 << i));
        ray.update(finalMask,hit.vt,hit.vu,hit.vv,hit.vNg.x,hit.vNg.y,hit.vNg.z,geomID,primIDs);
        ray.instID = context->instID;
        return true;

      }
    };
#endif    
    
    template<int M, int Mx, bool filter>
    struct Occluded1EpilogM
    {
      Ray& ray;
      IntersectContext* context;
      const vuint<M>& geomIDs;
      const vuint<M>& primIDs;

      __forceinline Occluded1EpilogM(Ray& ray,
                                     IntersectContext* context,
                                     const vuint<M>& geomIDs,
                                     const vuint<M>& primIDs)
        : ray(ray), context(context), geomIDs(geomIDs), primIDs(primIDs) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<Mx>& valid_i, Hit& hit) const
      {
        Scene* scene = context->scene;
        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION) || defined(EMBREE_RAY_MASK)
        if (unlikely(filter))
          hit.finalize(); /* called only once */

        vbool<Mx> valid = valid_i;
        if (Mx > M) valid &= (1<<M)-1;
        size_t m=movemask(valid);
        goto entry;
        while (true)
        {
          if (unlikely(m == 0)) return false;
        entry:
          size_t i=bsf(m);

          const unsigned int geomID = geomIDs[i];
          Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

#if defined(EMBREE_RAY_MASK)
          /* goto next hit if mask test fails */
          if ((geometry->mask & ray.mask) == 0) {
            m=btc(m,i);
            continue;
          }
#endif

#if defined(EMBREE_FILTER_FUNCTION)
          /* if we have no filter then the test passed */
          if (filter) {
            if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter()))
            {
              const Vec2f uv = hit.uv(i);
              HitK<1> h(context->instID,geomID,primIDs[i],uv.x,uv.y,hit.Ng(i));
              const float old_t = ray.tfar;
              ray.tfar = hit.t(i);
              if (runOcclusionFilter1(geometry,ray,context,h)) return true;
              ray.tfar = old_t;
              m=btc(m,i);
              continue;
            }
          }
#endif
          break;
        }
#endif

        return true;
      }
    };

    
    template<int M, bool filter>
    struct Intersect1EpilogMU
    {
      RayHit& ray;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Intersect1EpilogMU(RayHit& ray,
                                       IntersectContext* context,
                                       const unsigned int geomID,
                                       const unsigned int primID)
        : ray(ray), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<M>& valid_i, Hit& hit) const
      {
        /* ray mask test */
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        if ((geometry->mask & ray.mask) == 0) return false;
#endif

        vbool<M> valid = valid_i;
        hit.finalize();

        size_t i = select_min(valid,hit.vt);

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter()))
        {
          bool foundhit = false;
          while (true)
          {
            /* call intersection filter function */
            Vec2f uv = hit.uv(i);
            const float old_t = ray.tfar;
            ray.tfar = hit.t(i);
            HitK<1> h(context->instID,geomID,primID,uv.x,uv.y,hit.Ng(i));
            const bool found = runIntersectionFilter1(geometry,ray,context,h);
            if (!found) ray.tfar = old_t;
            foundhit |= found;
            clear(valid,i);
            valid &= hit.vt <= ray.tfar; // intersection filters may modify tfar value
            if (unlikely(none(valid))) break;
            i = select_min(valid,hit.vt);
          }
          return foundhit;
        }
#endif

        /* update hit information */
        const Vec2f uv = hit.uv(i);
        const Vec3fa Ng = hit.Ng(i);
        ray.tfar = hit.t(i);
        ray.Ng.x = Ng.x;
        ray.Ng.y = Ng.y;
        ray.Ng.z = Ng.z;
        ray.u = uv.x;
        ray.v = uv.y;
        ray.primID = primID;
        ray.geomID = geomID;
        ray.instID = context->instID;
        return true;
      }
    };
    
    template<int M, bool filter>
    struct Occluded1EpilogMU
    {
      Ray& ray;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Occluded1EpilogMU(Ray& ray,
                                      IntersectContext* context,
                                      const unsigned int geomID,
                                      const unsigned int primID)
        : ray(ray), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<M>& valid, Hit& hit) const
      {
        /* ray mask test */
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        if ((geometry->mask & ray.mask) == 0) return false;
#endif

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter()))
        {
          hit.finalize();
          for (size_t m=movemask(valid), i=bsf(m); m!=0; m=btc(m,i), i=bsf(m))
          {
            const Vec2f uv = hit.uv(i);
            const float old_t = ray.tfar;
            ray.tfar = hit.t(i);
            HitK<1> h(context->instID,geomID,primID,uv.x,uv.y,hit.Ng(i));
            if (runOcclusionFilter1(geometry,ray,context,h)) return true;
            ray.tfar = old_t;
          }
          return false;
        }
#endif
        return true;
      }
    };
        
    template<int M, int K, bool filter>
    struct IntersectKEpilogM
    {
      RayHitK<K>& ray;
      IntersectContext* context;
      const vuint<M>& geomIDs;
      const vuint<M>& primIDs;
      const size_t i;

      __forceinline IntersectKEpilogM(RayHitK<K>& ray,
                                      IntersectContext* context,
                                     const vuint<M>& geomIDs,
                                     const vuint<M>& primIDs,
                                     size_t i)
        : ray(ray), context(context), geomIDs(geomIDs), primIDs(primIDs), i(i) {}

      template<typename Hit>
      __forceinline vbool<K> operator() (const vbool<K>& valid_i, const Hit& hit) const
      {
        Scene* scene = context->scene;

        vfloat<K> u, v, t;
        Vec3vf<K> Ng;
        vbool<K> valid = valid_i;

        std::tie(u,v,t,Ng) = hit();

        const unsigned int geomID = geomIDs[i];
        const unsigned int primID = primIDs[i];
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

        /* ray masking test */
#if defined(EMBREE_RAY_MASK)
        valid &= (geometry->mask & ray.mask) != 0;
        if (unlikely(none(valid))) return false;
#endif

        /* occlusion filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter())) {
            HitK<K> h(context->instID,geomID,primID,u,v,Ng);
            const vfloat<K> old_t = ray.tfar;
            ray.tfar = select(valid,t,ray.tfar);
            const vbool<K> m_accept = runIntersectionFilter(valid,geometry,ray,context,h);
            ray.tfar = select(m_accept,ray.tfar,old_t);
            return m_accept;
          }
        }
#endif

        /* update hit information */
        vfloat<K>::store(valid,&ray.tfar,t);
        vfloat<K>::store(valid,&ray.Ng.x,Ng.x);
        vfloat<K>::store(valid,&ray.Ng.y,Ng.y);
        vfloat<K>::store(valid,&ray.Ng.z,Ng.z);
        vfloat<K>::store(valid,&ray.u,u);
        vfloat<K>::store(valid,&ray.v,v);
        vuint<K>::store(valid,&ray.primID,primID);
        vuint<K>::store(valid,&ray.geomID,geomID);
        vuint<K>::store(valid,&ray.instID,context->instID);
        return valid;
      }
    };
    
    template<int M, int K, bool filter>
    struct OccludedKEpilogM
    {
      vbool<K>& valid0;
      RayK<K>& ray;
      IntersectContext* context;
      const vuint<M>& geomIDs;
      const vuint<M>& primIDs;
      const size_t i;

      __forceinline OccludedKEpilogM(vbool<K>& valid0,
                                     RayK<K>& ray,
                                     IntersectContext* context,
                                     const vuint<M>& geomIDs,
                                     const vuint<M>& primIDs,
                                     size_t i)
        : valid0(valid0), ray(ray), context(context), geomIDs(geomIDs), primIDs(primIDs), i(i) {}

      template<typename Hit>
      __forceinline vbool<K> operator() (const vbool<K>& valid_i, const Hit& hit) const
      {
        vbool<K> valid = valid_i;

        /* ray masking test */
        Scene* scene = context->scene;
        const unsigned int geomID = geomIDs[i];
        const unsigned int primID = primIDs[i];
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        valid &= (geometry->mask & ray.mask) != 0;
        if (unlikely(none(valid))) return valid;
#endif

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter()))
          {
            vfloat<K> u, v, t;
            Vec3vf<K> Ng;
            std::tie(u,v,t,Ng) = hit();
            HitK<K> h(context->instID,geomID,primID,u,v,Ng);
            const vfloat<K> old_t = ray.tfar;
            ray.tfar = select(valid,t,ray.tfar);
            valid = runOcclusionFilter(valid,geometry,ray,context,h);
            ray.tfar = select(valid,ray.tfar,old_t);
          }
        }
#endif

        /* update occlusion */
        valid0 = valid0 & !valid;
        return valid;
      }
    };
    
    template<int M, int K, bool filter>
    struct IntersectKEpilogMU
    {
      RayHitK<K>& ray;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline IntersectKEpilogMU(RayHitK<K>& ray,
                                       IntersectContext* context,
                                       const unsigned int geomID,
                                       const unsigned int primID)
        : ray(ray), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline vbool<K> operator() (const vbool<K>& valid_org, const Hit& hit) const
      {
        vbool<K> valid = valid_org;
        vfloat<K> u, v, t;
        Vec3vf<K> Ng;
        std::tie(u,v,t,Ng) = hit();

        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

        /* ray masking test */
#if defined(EMBREE_RAY_MASK)
        valid &= (geometry->mask & ray.mask) != 0;
        if (unlikely(none(valid))) return false;
#endif

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter())) {
            HitK<K> h(context->instID,geomID,primID,u,v,Ng);
            const vfloat<K> old_t = ray.tfar;
            ray.tfar = select(valid,t,ray.tfar);
            const vbool<K> m_accept = runIntersectionFilter(valid,geometry,ray,context,h);
            ray.tfar = select(m_accept,ray.tfar,old_t);
            return m_accept;
          }
        }
#endif

        /* update hit information */
        vfloat<K>::store(valid,&ray.tfar,t);
        vfloat<K>::store(valid,&ray.Ng.x,Ng.x);
        vfloat<K>::store(valid,&ray.Ng.y,Ng.y);
        vfloat<K>::store(valid,&ray.Ng.z,Ng.z);
        vfloat<K>::store(valid,&ray.u,u);
        vfloat<K>::store(valid,&ray.v,v);
        vuint<K>::store(valid,&ray.primID,primID);
        vuint<K>::store(valid,&ray.geomID,geomID);
        vuint<K>::store(valid,&ray.instID,context->instID);
        return valid;
      }
    };
    
    template<int M, int K, bool filter>
    struct OccludedKEpilogMU
    {
      vbool<K>& valid0;
      RayK<K>& ray;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline OccludedKEpilogMU(vbool<K>& valid0,
                                      RayK<K>& ray,
                                      IntersectContext* context,
                                      const unsigned int geomID,
                                      const unsigned int primID)
        : valid0(valid0), ray(ray), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline vbool<K> operator() (const vbool<K>& valid_i, const Hit& hit) const
      {
        vbool<K> valid = valid_i;
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

#if defined(EMBREE_RAY_MASK)
        valid &= (geometry->mask & ray.mask) != 0;
        if (unlikely(none(valid))) return false;
#endif

        /* occlusion filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter()))
          {
            vfloat<K> u, v, t;
            Vec3vf<K> Ng;
            std::tie(u,v,t,Ng) = hit();
            HitK<K> h(context->instID,geomID,primID,u,v,Ng);
            const vfloat<K> old_t = ray.tfar;
            ray.tfar = select(valid,t,ray.tfar);
            valid = runOcclusionFilter(valid,geometry,ray,context,h);
            ray.tfar = select(valid,ray.tfar,old_t);
          }
        }
#endif

        /* update occlusion */
        valid0 = valid0 & !valid;
        return valid;
      }
    };
    
    template<int M, int Mx, int K, bool filter>
    struct Intersect1KEpilogM
    {
      RayHitK<K>& ray;
      size_t k;
      IntersectContext* context;
      const vuint<M>& geomIDs;
      const vuint<M>& primIDs;

      __forceinline Intersect1KEpilogM(RayHitK<K>& ray, size_t k,
                                       IntersectContext* context,
                                       const vuint<M>& geomIDs,
                                       const vuint<M>& primIDs)
        : ray(ray), k(k), context(context), geomIDs(geomIDs), primIDs(primIDs) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<Mx>& valid_i, Hit& hit) const
      {
        Scene* scene = context->scene;
        vbool<Mx> valid = valid_i;
        hit.finalize();
        if (Mx > M) valid &= (1<<M)-1;
        size_t i = select_min(valid,hit.vt);
        assert(i<M);
        unsigned int geomID = geomIDs[i];

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION) || defined(EMBREE_RAY_MASK)
        bool foundhit = false;
        goto entry;
        while (true)
        {
          if (unlikely(none(valid))) return foundhit;
          i = select_min(valid,hit.vt);
          assert(i<M);
          geomID = geomIDs[i];
        entry:
          Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

#if defined(EMBREE_RAY_MASK)
          /* goto next hit if mask test fails */
          if ((geometry->mask & ray.mask[k]) == 0) {
            clear(valid,i);
            continue;
          }
#endif

#if defined(EMBREE_FILTER_FUNCTION) 
          /* call intersection filter function */
          if (filter) {
            if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter())) {
              assert(i<M);
              const Vec2f uv = hit.uv(i);
              HitK<K> h(context->instID,geomID,primIDs[i],uv.x,uv.y,hit.Ng(i));
              const float old_t = ray.tfar[k];
              ray.tfar[k] = hit.t(i);
              const bool found = any(runIntersectionFilter(vbool<K>(1<<k),geometry,ray,context,h));
              if (!found) ray.tfar[k] = old_t;
              foundhit = foundhit | found;
              clear(valid,i);
              valid &= hit.vt <= ray.tfar[k]; // intersection filters may modify tfar value
              continue;
            }
          }
#endif
          break;
        }
#endif
        assert(i<M);
        /* update hit information */
#if 0 && defined(__AVX512F__) // do not enable, this reduced frequency for BVH4
        ray.updateK(i,k,hit.vt,hit.vu,hit.vv,vfloat<Mx>(hit.vNg.x),vfloat<Mx>(hit.vNg.y),vfloat<Mx>(hit.vNg.z),geomID,vuint<Mx>(primIDs));
        ray.instID[k] = context->instID;
#else
        const Vec2f uv = hit.uv(i);
        ray.tfar[k] = hit.t(i);
        ray.Ng.x[k] = hit.vNg.x[i];
        ray.Ng.y[k] = hit.vNg.y[i];
        ray.Ng.z[k] = hit.vNg.z[i];
        ray.u[k] = uv.x;
        ray.v[k] = uv.y;
        ray.primID[k] = primIDs[i];
        ray.geomID[k] = geomID;
        ray.instID[k] = context->instID;
#endif
        return true;
      }
    };
    
    template<int M, int Mx, int K, bool filter>
    struct Occluded1KEpilogM
    {
      RayK<K>& ray;
      size_t k;
      IntersectContext* context;
      const vuint<M>& geomIDs;
      const vuint<M>& primIDs;

      __forceinline Occluded1KEpilogM(RayK<K>& ray, size_t k,
                                      IntersectContext* context,
                                      const vuint<M>& geomIDs,
                                      const vuint<M>& primIDs)
        : ray(ray), k(k), context(context), geomIDs(geomIDs), primIDs(primIDs) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<Mx>& valid_i, Hit& hit) const
      {
        Scene* scene = context->scene;

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION) || defined(EMBREE_RAY_MASK)
        if (unlikely(filter))
          hit.finalize(); /* called only once */

        vbool<Mx> valid = valid_i;
        if (Mx > M) valid &= (1<<M)-1;
        size_t m=movemask(valid);
        goto entry;
        while (true)
        {
          if (unlikely(m == 0)) return false;
        entry:
          size_t i=bsf(m);

          const unsigned int geomID = geomIDs[i];
          Geometry* geometry MAYBE_UNUSED = scene->get(geomID);

#if defined(EMBREE_RAY_MASK)
          /* goto next hit if mask test fails */
          if ((geometry->mask & ray.mask[k]) == 0) {
            m=btc(m,i);
            continue;
          }
#endif

#if defined(EMBREE_FILTER_FUNCTION)
          /* execute occlusion filer */
          if (filter) {
            if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter()))
            {
              const Vec2f uv = hit.uv(i);
              const float old_t = ray.tfar[k];
              ray.tfar[k] = hit.t(i);
              HitK<K> h(context->instID,geomID,primIDs[i],uv.x,uv.y,hit.Ng(i));
              if (any(runOcclusionFilter(vbool<K>(1<<k),geometry,ray,context,h))) return true;
              ray.tfar[k] = old_t;
              m=btc(m,i);
              continue;
            }
          }
#endif
          break;
        }
#endif
        return true;
      }
    };
    
    template<int M, int K, bool filter>
    struct Intersect1KEpilogMU
    {
      RayHitK<K>& ray;
      size_t k;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Intersect1KEpilogMU(RayHitK<K>& ray, size_t k,
                                        IntersectContext* context,
                                        const unsigned int geomID,
                                        const unsigned int primID)
        : ray(ray), k(k), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<M>& valid_i, Hit& hit) const
      {
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        /* ray mask test */
        if ((geometry->mask & ray.mask[k]) == 0)
          return false;
#endif

        /* finalize hit calculation */
        vbool<M> valid = valid_i;
        hit.finalize();
        size_t i = select_min(valid,hit.vt);

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasIntersectionFilter()))
          {
            bool foundhit = false;
            while (true)
            {
              const Vec2f uv = hit.uv(i);
              const float old_t = ray.tfar[k];
              ray.tfar[k] = hit.t(i);
              HitK<K> h(context->instID,geomID,primID,uv.x,uv.y,hit.Ng(i));
              const bool found = any(runIntersectionFilter(vbool<K>(1<<k),geometry,ray,context,h));
              if (!found) ray.tfar[k] = old_t;
              foundhit = foundhit | found;
              clear(valid,i);
              valid &= hit.vt <= ray.tfar[k]; // intersection filters may modify tfar value
              if (unlikely(none(valid))) break;
              i = select_min(valid,hit.vt);
            }
            return foundhit;
          }
        }
#endif

        /* update hit information */
#if 0 && defined(__AVX512F__) // do not enable, this reduced frequency for BVH4
        const Vec3fa Ng = hit.Ng(i);
        ray.updateK(i,k,hit.vt,hit.vu,hit.vv,vfloat<M>(Ng.x),vfloat<M>(Ng.y),vfloat<M>(Ng.z),geomID,vuint<M>(primID));
        ray.instID[k] = context->instID;
#else
        const Vec2f uv = hit.uv(i);
        const Vec3fa Ng = hit.Ng(i);
        ray.tfar[k] = hit.t(i);
        ray.Ng.x[k] = Ng.x;
        ray.Ng.y[k] = Ng.y;
        ray.Ng.z[k] = Ng.z;
        ray.u[k] = uv.x;
        ray.v[k] = uv.y;
        ray.primID[k] = primID;
        ray.geomID[k] = geomID;
        ray.instID[k] = context->instID;
#endif
        return true;
      }
    };
    
    template<int M, int K, bool filter>
    struct Occluded1KEpilogMU
    {
      RayK<K>& ray;
      size_t k;
      IntersectContext* context;
      const unsigned int geomID;
      const unsigned int primID;

      __forceinline Occluded1KEpilogMU(RayK<K>& ray, size_t k,
                                       IntersectContext* context,
                                       const unsigned int geomID,
                                       const unsigned int primID)
        : ray(ray), k(k), context(context), geomID(geomID), primID(primID) {}

      template<typename Hit>
      __forceinline bool operator() (const vbool<M>& valid_i, Hit& hit) const
      {
        Scene* scene = context->scene;
        Geometry* geometry MAYBE_UNUSED = scene->get(geomID);
#if defined(EMBREE_RAY_MASK)
        /* ray mask test */
        if ((geometry->mask & ray.mask[k]) == 0)
          return false;
#endif

        /* intersection filter test */
#if defined(EMBREE_FILTER_FUNCTION)
        if (filter) {
          if (unlikely(context->hasContextFilter() || geometry->hasOcclusionFilter()))
          {
            hit.finalize();
            for (size_t m=movemask(valid_i), i=bsf(m); m!=0; m=btc(m,i), i=bsf(m))
            {
              const Vec2f uv = hit.uv(i);
              const float old_t = ray.tfar[k];
              ray.tfar[k] = hit.t(i);
              HitK<K> h(context->instID,geomID,primID,uv.x,uv.y,hit.Ng(i));
              if (any(runOcclusionFilter(vbool<K>(1<<k),geometry,ray,context,h))) return true;
              ray.tfar[k] = old_t;
            }
            return false;
          }
        }
#endif 
        return true;
      }
    };
  }
}
