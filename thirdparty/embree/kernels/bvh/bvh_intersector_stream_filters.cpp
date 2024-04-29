// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector_stream_filters.h"
#include "bvh_intersector_stream.h"

namespace embree
{
  namespace isa
  {
    template<int K, bool intersect>
    __noinline void RayStreamFilter::filterAOS(Scene* scene, void* _rayN, size_t N, size_t stride, IntersectContext* context)
    {
      RayStreamAOS rayN(_rayN);

      /* use fast path for coherent ray mode */
      if (unlikely(context->isCoherent()))
      {
        __aligned(64) RayTypeK<K, intersect> rays[MAX_INTERNAL_STREAM_SIZE / K];
        __aligned(64) RayTypeK<K, intersect>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

        for (size_t i = 0; i < N; i += MAX_INTERNAL_STREAM_SIZE)
        {
          const size_t size = min(N - i, MAX_INTERNAL_STREAM_SIZE);

          /* convert from AOS to SOA */
          for (size_t j = 0; j < size; j += K)
          {
            const vint<K> vij = vint<K>(int(i+j)) + vint<K>(step);
            const vbool<K> valid = vij < vint<K>(int(N));
            const vint<K> offset = vij * int(stride);
            const size_t packetIndex = j / K;

            RayTypeK<K, intersect> ray = rayN.getRayByOffset<K>(valid, offset);
            ray.tnear() = select(valid, ray.tnear(), zero);
            ray.tfar  = select(valid, ray.tfar,  neg_inf);

            rays[packetIndex] = ray;
            rayPtrs[packetIndex] = &rays[packetIndex]; // rayPtrs might get reordered for occludedN
          }

          /* trace stream */
          scene->intersectors.intersectN(rayPtrs, size, context);

          /* convert from SOA to AOS */
          for (size_t j = 0; j < size; j += K)
          {
            const vint<K> vij = vint<K>(int(i+j)) + vint<K>(step);
            const vbool<K> valid = vij < vint<K>(int(N));
            const vint<K> offset = vij * int(stride);
            const size_t packetIndex = j / K;
            rayN.setHitByOffset(valid, offset, rays[packetIndex]);
          }
        }
      }
      else if (unlikely(!intersect))
      {
        /* octant sorting for occlusion rays */
        __aligned(64) unsigned int octants[8][MAX_INTERNAL_STREAM_SIZE];
        __aligned(64) RayK<K> rays[MAX_INTERNAL_STREAM_SIZE / K];
        __aligned(64) RayK<K>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

        unsigned int raysInOctant[8];
        for (unsigned int i = 0; i < 8; i++)
          raysInOctant[i] = 0;
        size_t inputRayID = 0;

        for (;;)
        {
          int curOctant = -1;

          /* sort rays into octants */
          for (; inputRayID < N;)
          {
            const Ray& ray = rayN.getRayByOffset(inputRayID * stride);

            /* skip invalid rays */
            if (unlikely(ray.tnear() > ray.tfar || ray.tfar < 0.0f)) { inputRayID++; continue; } // ignore invalid or already occluded rays
#if defined(EMBREE_IGNORE_INVALID_RAYS)
            if (unlikely(!ray.valid())) { inputRayID++; continue; }
#endif

            const unsigned int octantID = movemask(vfloat4(Vec3fa(ray.dir)) < 0.0f) & 0x7;

            assert(octantID < 8);
            octants[octantID][raysInOctant[octantID]++] = (unsigned int)inputRayID;
            inputRayID++;
            if (unlikely(raysInOctant[octantID] == MAX_INTERNAL_STREAM_SIZE))
            {
              curOctant = octantID;
              break;
            }
          }

          /* need to flush rays in octant? */
          if (unlikely(curOctant == -1))
          {
            for (unsigned int i = 0; i < 8; i++)
              if (raysInOctant[i]) { curOctant = i; break; }
          }

          /* all rays traced? */
          if (unlikely(curOctant == -1))
            break;
        
          unsigned int* const rayIDs = &octants[curOctant][0];
          const unsigned int numOctantRays = raysInOctant[curOctant];
          assert(numOctantRays);

          for (unsigned int j = 0; j < numOctantRays; j += K)
          {
            const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
            const vbool<K> valid = vi < vint<K>(int(numOctantRays));
            const vint<K> offset = *(vint<K>*)&rayIDs[j] * int(stride);
            RayK<K>& ray = rays[j/K];
            rayPtrs[j/K] = &ray;
            ray = rayN.getRayByOffset<K>(valid, offset);
            ray.tnear() = select(valid, ray.tnear(), zero);
            ray.tfar  = select(valid, ray.tfar,  neg_inf);
          }

          scene->intersectors.occludedN(rayPtrs, numOctantRays, context);

          for (unsigned int j = 0; j < numOctantRays; j += K)
          {
            const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
            const vbool<K> valid = vi < vint<K>(int(numOctantRays));
            const vint<K> offset = *(vint<K>*)&rayIDs[j] * int(stride);
            rayN.setHitByOffset<K>(valid, offset, rays[j/K]);
          }

          raysInOctant[curOctant] = 0;
        }
      }
      else
      {
        /* fallback to packets */
        for (size_t i = 0; i < N; i += K)
        {
          const vint<K> vi = vint<K>(int(i)) + vint<K>(step);
          vbool<K> valid = vi < vint<K>(int(N));
          const vint<K> offset = vi * int(stride);

          RayTypeK<K, intersect> ray = rayN.getRayByOffset<K>(valid, offset);
          valid &= ray.tnear() <= ray.tfar;

          scene->intersectors.intersect(valid, ray, context);

          rayN.setHitByOffset<K>(valid, offset, ray);
        }
      }
    }

    template<int K, bool intersect>
    __noinline void RayStreamFilter::filterAOP(Scene* scene, void** _rayN, size_t N, IntersectContext* context)
    {
      RayStreamAOP rayN(_rayN);

      /* use fast path for coherent ray mode */
      if (unlikely(context->isCoherent()))
      {
        __aligned(64) RayTypeK<K, intersect> rays[MAX_INTERNAL_STREAM_SIZE / K];
        __aligned(64) RayTypeK<K, intersect>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

        for (size_t i = 0; i < N; i += MAX_INTERNAL_STREAM_SIZE)
        {
          const size_t size = min(N - i, MAX_INTERNAL_STREAM_SIZE);

          /* convert from AOP to SOA */
          for (size_t j = 0; j < size; j += K)
          {
            const vint<K> vij = vint<K>(int(i+j)) + vint<K>(step);
            const vbool<K> valid = vij < vint<K>(int(N));
            const size_t packetIndex = j / K;

            RayTypeK<K, intersect> ray = rayN.getRayByIndex<K>(valid, vij);
            ray.tnear() = select(valid, ray.tnear(), zero);
            ray.tfar  = select(valid, ray.tfar,  neg_inf);

            rays[packetIndex] = ray;
            rayPtrs[packetIndex] = &rays[packetIndex]; // rayPtrs might get reordered for occludedN
          }

          /* trace stream */
          scene->intersectors.intersectN(rayPtrs, size, context);

          /* convert from SOA to AOP */
          for (size_t j = 0; j < size; j += K)
          {
            const vint<K> vij = vint<K>(int(i+j)) + vint<K>(step);
            const vbool<K> valid = vij < vint<K>(int(N));
            const size_t packetIndex = j / K;

            rayN.setHitByIndex<K>(valid, vij, rays[packetIndex]);
          }
        }
      }
      else if (unlikely(!intersect))
      {
        /* octant sorting for occlusion rays */
        __aligned(64) unsigned int octants[8][MAX_INTERNAL_STREAM_SIZE];
        __aligned(64) RayK<K> rays[MAX_INTERNAL_STREAM_SIZE / K];
        __aligned(64) RayK<K>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

        unsigned int raysInOctant[8];
        for (unsigned int i = 0; i < 8; i++)
          raysInOctant[i] = 0;
        size_t inputRayID = 0;

        for (;;)
        {
          int curOctant = -1;

          /* sort rays into octants */
          for (; inputRayID < N;)
          {
            const Ray& ray = rayN.getRayByIndex(inputRayID);

            /* skip invalid rays */
            if (unlikely(ray.tnear() > ray.tfar || ray.tfar < 0.0f)) { inputRayID++; continue; } // ignore invalid or already occluded rays
#if defined(EMBREE_IGNORE_INVALID_RAYS)
            if (unlikely(!ray.valid())) { inputRayID++; continue; }
#endif

            const unsigned int octantID = movemask(lt_mask(ray.dir,Vec3fa(0.0f)));

            assert(octantID < 8);
            octants[octantID][raysInOctant[octantID]++] = (unsigned int)inputRayID;
            inputRayID++;
            if (unlikely(raysInOctant[octantID] == MAX_INTERNAL_STREAM_SIZE))
            {
              curOctant = octantID;
              break;
            }
          }

          /* need to flush rays in octant? */
          if (unlikely(curOctant == -1))
          {
            for (unsigned int i = 0; i < 8; i++)
              if (raysInOctant[i]) { curOctant = i; break; }
          }

          /* all rays traced? */
          if (unlikely(curOctant == -1))
            break;

          unsigned int* const rayIDs = &octants[curOctant][0];
          const unsigned int numOctantRays = raysInOctant[curOctant];
          assert(numOctantRays);

          for (unsigned int j = 0; j < numOctantRays; j += K)
          {
            const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
            const vbool<K> valid = vi < vint<K>(int(numOctantRays));
            const vint<K> index = *(vint<K>*)&rayIDs[j];
            RayK<K>& ray = rays[j/K];
            rayPtrs[j/K] = &ray;
            ray = rayN.getRayByIndex<K>(valid, index);
            ray.tnear() = select(valid, ray.tnear(), zero);
            ray.tfar  = select(valid, ray.tfar,  neg_inf);
          }

          scene->intersectors.occludedN(rayPtrs, numOctantRays, context);

          for (unsigned int j = 0; j < numOctantRays; j += K)
          {
            const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
            const vbool<K> valid = vi < vint<K>(int(numOctantRays));
            const vint<K> index = *(vint<K>*)&rayIDs[j];
            rayN.setHitByIndex<K>(valid, index, rays[j/K]);
          }

          raysInOctant[curOctant] = 0;
        }
      }
      else
      {
        /* fallback to packets */
        for (size_t i = 0; i < N; i += K)
        {
          const vint<K> vi = vint<K>(int(i)) + vint<K>(step);
          vbool<K> valid = vi < vint<K>(int(N));

          RayTypeK<K, intersect> ray = rayN.getRayByIndex<K>(valid, vi);
          valid &= ray.tnear() <= ray.tfar;

          scene->intersectors.intersect(valid, ray, context);

          rayN.setHitByIndex<K>(valid, vi, ray);
        }
      }
    }

    template<int K, bool intersect>
    __noinline void RayStreamFilter::filterSOA(Scene* scene, char* rayData, size_t N, size_t numPackets, size_t stride, IntersectContext* context)
    {
      const size_t rayDataAlignment = (size_t)rayData % (K*sizeof(float));
      const size_t offsetAlignment  = (size_t)stride  % (K*sizeof(float));

      /* fast path for packets with the correct width and data alignment */
      if (likely(N == K &&
                 !rayDataAlignment &&
                 !offsetAlignment))
      {
        if (unlikely(context->isCoherent()))
        {
          __aligned(64) RayTypeK<K, intersect>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

          size_t packetIndex = 0;
          for (size_t i = 0; i < numPackets; i++)
          {
            const size_t offset = i * stride;
            RayTypeK<K, intersect>& ray = *(RayTypeK<K, intersect>*)(rayData + offset);
            rayPtrs[packetIndex++] = &ray;

            /* trace as stream */
            if (unlikely(packetIndex == MAX_INTERNAL_STREAM_SIZE / K))
            {
              const size_t size = packetIndex*K;
              scene->intersectors.intersectN(rayPtrs, size, context);
              packetIndex = 0;
            }
          }

          /* flush remaining packets */
          if (unlikely(packetIndex > 0))
          {
            const size_t size = packetIndex*K;
            scene->intersectors.intersectN(rayPtrs, size, context);
          }
        }
        else if (unlikely(!intersect))
        {
          /* octant sorting for occlusion rays */
          RayStreamSOA rayN(rayData, K);

          __aligned(64) unsigned int octants[8][MAX_INTERNAL_STREAM_SIZE];
          __aligned(64) RayK<K> rays[MAX_INTERNAL_STREAM_SIZE / K];
          __aligned(64) RayK<K>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

          unsigned int raysInOctant[8];
          for (unsigned int i = 0; i < 8; i++)
            raysInOctant[i] = 0;
          size_t inputRayID = 0;

          for (;;)
          {
            int curOctant = -1;

            /* sort rays into octants */
            for (; inputRayID < N*numPackets;)
            {
              const size_t offset = (inputRayID / K) * stride + (inputRayID % K) * sizeof(float);

              /* skip invalid rays */
              if (unlikely(!rayN.isValidByOffset(offset))) { inputRayID++; continue; } // ignore invalid or already occluded rays
  #if defined(EMBREE_IGNORE_INVALID_RAYS)
              __aligned(64) Ray ray = rayN.getRayByOffset(offset);
              if (unlikely(!ray.valid())) { inputRayID++; continue; }
  #endif

              const unsigned int octantID = (unsigned int)rayN.getOctantByOffset(offset);

              assert(octantID < 8);
              octants[octantID][raysInOctant[octantID]++] = (unsigned int)offset;
              inputRayID++;
              if (unlikely(raysInOctant[octantID] == MAX_INTERNAL_STREAM_SIZE))
              {
                curOctant = octantID;
                break;
              }
            }

            /* need to flush rays in octant? */
            if (unlikely(curOctant == -1))
            {
              for (unsigned int i = 0; i < 8; i++)
                if (raysInOctant[i]) { curOctant = i; break; }
            }

            /* all rays traced? */
            if (unlikely(curOctant == -1))
              break;

            unsigned int* const rayOffsets = &octants[curOctant][0];
            const unsigned int numOctantRays = raysInOctant[curOctant];
            assert(numOctantRays);

            for (unsigned int j = 0; j < numOctantRays; j += K)
            {
              const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
              const vbool<K> valid = vi < vint<K>(int(numOctantRays));
              const vint<K> offset = *(vint<K>*)&rayOffsets[j];
              RayK<K>& ray = rays[j/K];
              rayPtrs[j/K] = &ray;
              ray = rayN.getRayByOffset<K>(valid, offset);
              ray.tnear() = select(valid, ray.tnear(), zero);
              ray.tfar  = select(valid, ray.tfar,  neg_inf);
            }

            scene->intersectors.occludedN(rayPtrs, numOctantRays, context);

            for (unsigned int j = 0; j < numOctantRays; j += K)
            {
              const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
              const vbool<K> valid = vi < vint<K>(int(numOctantRays));
              const vint<K> offset = *(vint<K>*)&rayOffsets[j];
              rayN.setHitByOffset(valid, offset, rays[j/K]);
            }
            raysInOctant[curOctant] = 0;
          }
        }
        else
        {
          /* fallback to packets */
          for (size_t i = 0; i < numPackets; i++)
          {
            const size_t offset = i * stride;
            RayTypeK<K, intersect>& ray = *(RayTypeK<K, intersect>*)(rayData + offset);
            const vbool<K> valid = ray.tnear() <= ray.tfar;

            scene->intersectors.intersect(valid, ray, context);
          }
        }
      }
      else
      {
        /* fallback to packets for arbitrary packet size and alignment */
        for (size_t i = 0; i < numPackets; i++)
        {
          const size_t offsetN = i * stride;
          RayStreamSOA rayN(rayData + offsetN, N);

          for (size_t j = 0; j < N; j += K)
          {
            const size_t offset = j * sizeof(float);
            vbool<K> valid = (vint<K>(int(j)) + vint<K>(step)) < vint<K>(int(N));
            RayTypeK<K, intersect> ray = rayN.getRayByOffset<K>(valid, offset);
            valid &= ray.tnear() <= ray.tfar;

            scene->intersectors.intersect(valid, ray, context);

            rayN.setHitByOffset(valid, offset, ray);
          }
        }
      }
    }

    template<int K, bool intersect>
    __noinline void RayStreamFilter::filterSOP(Scene* scene, const void* _rayN, size_t N, IntersectContext* context)
    { 
      RayStreamSOP& rayN = *(RayStreamSOP*)_rayN;

      /* use fast path for coherent ray mode */
      if (unlikely(context->isCoherent()))
      {
        __aligned(64) RayTypeK<K, intersect> rays[MAX_INTERNAL_STREAM_SIZE / K];
        __aligned(64) RayTypeK<K, intersect>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

        for (size_t i = 0; i < N; i += MAX_INTERNAL_STREAM_SIZE)
        {
          const size_t size = min(N - i, MAX_INTERNAL_STREAM_SIZE);

          /* convert from SOP to SOA */
          for (size_t j = 0; j < size; j += K)
          {
            const vint<K> vij = vint<K>(int(i+j)) + vint<K>(step);
            const vbool<K> valid = vij < vint<K>(int(N));
            const size_t offset = (i+j) * sizeof(float);
            const size_t packetIndex = j / K;

            RayTypeK<K, intersect> ray = rayN.getRayByOffset<K>(valid, offset);
            ray.tnear() = select(valid, ray.tnear(), zero);
            ray.tfar  = select(valid, ray.tfar,  neg_inf);

            rays[packetIndex] = ray;
            rayPtrs[packetIndex] = &rays[packetIndex]; // rayPtrs might get reordered for occludedN
          }

          /* trace stream */
          scene->intersectors.intersectN(rayPtrs, size, context);

          /* convert from SOA to SOP */
          for (size_t j = 0; j < size; j += K)
          {
            const vint<K> vij = vint<K>(int(i+j)) + vint<K>(step);
            const vbool<K> valid = vij < vint<K>(int(N));
            const size_t offset = (i+j) * sizeof(float);
            const size_t packetIndex = j / K;

            rayN.setHitByOffset(valid, offset, rays[packetIndex]);
          }
        }
      }
      else if (unlikely(!intersect))
      {
        /* octant sorting for occlusion rays */
        __aligned(64) unsigned int octants[8][MAX_INTERNAL_STREAM_SIZE];
        __aligned(64) RayK<K> rays[MAX_INTERNAL_STREAM_SIZE / K];
        __aligned(64) RayK<K>* rayPtrs[MAX_INTERNAL_STREAM_SIZE / K];

        unsigned int raysInOctant[8];
        for (unsigned int i = 0; i < 8; i++)
          raysInOctant[i] = 0;
        size_t inputRayID = 0;

        for (;;)
        {
          int curOctant = -1;

          /* sort rays into octants */
          for (; inputRayID < N;)
          {
            const size_t offset = inputRayID * sizeof(float);
            /* skip invalid rays */
            if (unlikely(!rayN.isValidByOffset(offset))) { inputRayID++; continue; } // ignore invalid or already occluded rays
#if defined(EMBREE_IGNORE_INVALID_RAYS)
            __aligned(64) Ray ray = rayN.getRayByOffset(offset);
            if (unlikely(!ray.valid())) { inputRayID++; continue; }
#endif

            const unsigned int octantID = (unsigned int)rayN.getOctantByOffset(offset);

            assert(octantID < 8);
            octants[octantID][raysInOctant[octantID]++] = (unsigned int)offset;
            inputRayID++;
            if (unlikely(raysInOctant[octantID] == MAX_INTERNAL_STREAM_SIZE))
            {
              curOctant = octantID;
              break;
            }
          }

          /* need to flush rays in octant? */
          if (unlikely(curOctant == -1))
          {
            for (unsigned int i = 0; i < 8; i++)
              if (raysInOctant[i]) { curOctant = i; break; }
          }

          /* all rays traced? */
          if (unlikely(curOctant == -1))
            break;

          unsigned int* const rayOffsets = &octants[curOctant][0];
          const unsigned int numOctantRays = raysInOctant[curOctant];
          assert(numOctantRays);

          for (unsigned int j = 0; j < numOctantRays; j += K)
          {
            const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
            const vbool<K> valid = vi < vint<K>(int(numOctantRays));
            const vint<K> offset = *(vint<K>*)&rayOffsets[j];
            RayK<K>& ray = rays[j/K];
            rayPtrs[j/K] = &ray;
            ray = rayN.getRayByOffset<K>(valid, offset);
            ray.tnear() = select(valid, ray.tnear(), zero);
            ray.tfar  = select(valid, ray.tfar,  neg_inf);
          }

          scene->intersectors.occludedN(rayPtrs, numOctantRays, context);

          for (unsigned int j = 0; j < numOctantRays; j += K)
          {
            const vint<K> vi = vint<K>(int(j)) + vint<K>(step);
            const vbool<K> valid = vi < vint<K>(int(numOctantRays));
            const vint<K> offset = *(vint<K>*)&rayOffsets[j];
            rayN.setHitByOffset(valid, offset, rays[j/K]);
          }

          raysInOctant[curOctant] = 0;
        }
      }
      else
      {
        /* fallback to packets */
        for (size_t i = 0; i < N; i += K)
        {
          const vint<K> vi = vint<K>(int(i)) + vint<K>(step);
          vbool<K> valid = vi < vint<K>(int(N));
          const size_t offset = i * sizeof(float);

          RayTypeK<K, intersect> ray = rayN.getRayByOffset<K>(valid, offset);
          valid &= ray.tnear() <= ray.tfar;

          scene->intersectors.intersect(valid, ray, context);

          rayN.setHitByOffset(valid, offset, ray);
        }
      }
    }


    void RayStreamFilter::intersectAOS(Scene* scene, RTCRayHit* _rayN, size_t N, size_t stride, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterAOS<VSIZEL, true>(scene, _rayN, N, stride, context);
      else
        filterAOS<VSIZEX, true>(scene, _rayN, N, stride, context);
    }

    void RayStreamFilter::occludedAOS(Scene* scene, RTCRay* _rayN, size_t N, size_t stride, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterAOS<VSIZEL, false>(scene, _rayN, N, stride, context);
      else
        filterAOS<VSIZEX, false>(scene, _rayN, N, stride, context);
    }

    void RayStreamFilter::intersectAOP(Scene* scene, RTCRayHit** _rayN, size_t N, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterAOP<VSIZEL, true>(scene, (void**)_rayN, N, context);
      else
        filterAOP<VSIZEX, true>(scene, (void**)_rayN, N, context);
    }

    void RayStreamFilter::occludedAOP(Scene* scene, RTCRay** _rayN, size_t N, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterAOP<VSIZEL, false>(scene, (void**)_rayN, N, context);
      else
        filterAOP<VSIZEX, false>(scene, (void**)_rayN, N, context);
    }

    void RayStreamFilter::intersectSOA(Scene* scene, char* rayData, size_t N, size_t numPackets, size_t stride, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterSOA<VSIZEL, true>(scene, rayData, N, numPackets, stride, context);
      else
        filterSOA<VSIZEX, true>(scene, rayData, N, numPackets, stride, context);
    }

    void RayStreamFilter::occludedSOA(Scene* scene, char* rayData, size_t N, size_t numPackets, size_t stride, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterSOA<VSIZEL, false>(scene, rayData, N, numPackets, stride, context);
      else
        filterSOA<VSIZEX, false>(scene, rayData, N, numPackets, stride, context);
    }

    void RayStreamFilter::intersectSOP(Scene* scene, const RTCRayHitNp* _rayN, size_t N, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterSOP<VSIZEL, true>(scene, _rayN, N, context);
      else
        filterSOP<VSIZEX, true>(scene, _rayN, N, context);
    }

    void RayStreamFilter::occludedSOP(Scene* scene, const RTCRayNp* _rayN, size_t N, IntersectContext* context) {
      if (unlikely(context->isCoherent()))
        filterSOP<VSIZEL, false>(scene, _rayN, N, context);
      else
        filterSOP<VSIZEX, false>(scene, _rayN, N, context);
    }


    RayStreamFilterFuncs rayStreamFilterFuncs() {
      return RayStreamFilterFuncs(RayStreamFilter::intersectAOS, RayStreamFilter::intersectAOP, RayStreamFilter::intersectSOA, RayStreamFilter::intersectSOP,
                                  RayStreamFilter::occludedAOS,  RayStreamFilter::occludedAOP,  RayStreamFilter::occludedSOA,  RayStreamFilter::occludedSOP);
    }
  };
};
