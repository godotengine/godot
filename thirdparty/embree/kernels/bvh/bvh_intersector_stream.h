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

#include "node_intersector_packet_stream.h"
#include "node_intersector_frustum.h"
#include "bvh_traverser_stream.h"

namespace embree
{
  namespace isa 
  {
    /*! BVH ray stream intersector. */
    template<int N, int Nx, int types, bool robust, typename PrimitiveIntersector>
    class BVHNIntersectorStream
    {
      static const int Nxd = (Nx == N) ? N : Nx/2;

      /* shortcuts for frequently used types */
      template<int K> using PrimitiveIntersectorK = typename PrimitiveIntersector::template Type<K>;
      template<int K> using PrimitiveK = typename PrimitiveIntersectorK<K>::PrimitiveK;
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::BaseNode BaseNode;
      typedef typename BVH::AlignedNode AlignedNode;
      typedef typename BVH::AlignedNodeMB AlignedNodeMB;

      template<int K>
      __forceinline static size_t initPacketsAndFrustum(RayK<K>** inputPackets, size_t numOctantRays,
                                                        TravRayKStream<K, robust>* packets, Frustum<robust>& frustum, bool& commonOctant)
      {
        const size_t numPackets = (numOctantRays+K-1)/K;

        Vec3vf<K> tmp_min_rdir(pos_inf);
        Vec3vf<K> tmp_max_rdir(neg_inf);
        Vec3vf<K> tmp_min_org(pos_inf);
        Vec3vf<K> tmp_max_org(neg_inf);
        vfloat<K> tmp_min_dist(pos_inf);
        vfloat<K> tmp_max_dist(neg_inf);

        size_t m_active = 0;
        for (size_t i = 0; i < numPackets; i++)
        {
          const vfloat<K> tnear = inputPackets[i]->tnear();
          const vfloat<K> tfar  = inputPackets[i]->tfar;
          vbool<K> m_valid = (tnear <= tfar) & (tnear >= 0.0f);

#if defined(EMBREE_IGNORE_INVALID_RAYS)
          m_valid &= inputPackets[i]->valid();
#endif

          m_active |= (size_t)movemask(m_valid) << (i*K);

          vfloat<K> packet_min_dist = max(tnear, 0.0f);
          vfloat<K> packet_max_dist = select(m_valid, tfar, neg_inf);
          tmp_min_dist = min(tmp_min_dist, packet_min_dist);
          tmp_max_dist = max(tmp_max_dist, packet_max_dist);

          const Vec3vf<K>& org = inputPackets[i]->org;
          const Vec3vf<K>& dir = inputPackets[i]->dir;

          new (&packets[i]) TravRayKStream<K, robust>(org, dir, packet_min_dist, packet_max_dist);

          tmp_min_rdir = min(tmp_min_rdir, select(m_valid, packets[i].rdir, Vec3vf<K>(pos_inf)));
          tmp_max_rdir = max(tmp_max_rdir, select(m_valid, packets[i].rdir, Vec3vf<K>(neg_inf)));
          tmp_min_org  = min(tmp_min_org , select(m_valid,org , Vec3vf<K>(pos_inf)));
          tmp_max_org  = max(tmp_max_org , select(m_valid,org , Vec3vf<K>(neg_inf)));
        }

        m_active &= (numOctantRays == (8 * sizeof(size_t))) ? (size_t)-1 : (((size_t)1 << numOctantRays)-1);

        
        const Vec3fa reduced_min_rdir(reduce_min(tmp_min_rdir.x),
                                      reduce_min(tmp_min_rdir.y),
                                      reduce_min(tmp_min_rdir.z));

        const Vec3fa reduced_max_rdir(reduce_max(tmp_max_rdir.x),
                                      reduce_max(tmp_max_rdir.y),
                                      reduce_max(tmp_max_rdir.z));

        const Vec3fa reduced_min_origin(reduce_min(tmp_min_org.x),
                                        reduce_min(tmp_min_org.y),
                                        reduce_min(tmp_min_org.z));

        const Vec3fa reduced_max_origin(reduce_max(tmp_max_org.x),
                                        reduce_max(tmp_max_org.y),
                                        reduce_max(tmp_max_org.z));

        commonOctant =
          (reduced_max_rdir.x < 0.0f || reduced_min_rdir.x >= 0.0f) &&
          (reduced_max_rdir.y < 0.0f || reduced_min_rdir.y >= 0.0f) &&
          (reduced_max_rdir.z < 0.0f || reduced_min_rdir.z >= 0.0f);
        
        const float frustum_min_dist = reduce_min(tmp_min_dist);
        const float frustum_max_dist = reduce_max(tmp_max_dist);

        frustum.init(reduced_min_origin, reduced_max_origin,
                     reduced_min_rdir, reduced_max_rdir,
                     frustum_min_dist, frustum_max_dist,
                     N);
        
        return m_active;
      }

      template<int K>
      __forceinline static size_t intersectAlignedNodePacket(size_t m_active,
                                                             const TravRayKStream<K,robust>* packets,
                                                             const AlignedNode* __restrict__ node,
                                                             size_t boxID,
                                                             const NearFarPrecalculations& nf)
      {
        assert(m_active);
        const size_t startPacketID = bsf(m_active) / K;
        const size_t endPacketID   = bsr(m_active) / K;
        size_t m_trav_active = 0;
        for (size_t i = startPacketID; i <= endPacketID; i++)
        {
          const size_t m_hit = intersectNodeK<N>(node, boxID, packets[i], nf);
          m_trav_active |= m_hit << (i*K);
        } 
        return m_trav_active;
      }
      
      template<int K>
      __forceinline static size_t traverseCoherentStream(size_t m_active,
                                                         TravRayKStream<K, robust>* packets,
                                                         const AlignedNode* __restrict__ node,
                                                         const Frustum<robust>& frustum,
                                                         size_t* maskK,
                                                         vfloat<Nx>& dist)
      {
        size_t m_node_hit = intersectNodeFrustum<N,Nx>(node, frustum, dist);
        const size_t first_index    = bsf(m_active);
        const size_t first_packetID = first_index / K;
        const size_t first_rayID    = first_index % K;
        size_t m_first_hit = intersectNode1<N,Nx>(node, packets[first_packetID], first_rayID, frustum.nf);

        /* this make traversal independent of the ordering of rays */
        size_t m_node = m_node_hit ^ m_first_hit;
        while (unlikely(m_node))
        {
          const size_t boxID = bscf(m_node);
          const size_t m_current = m_active & intersectAlignedNodePacket(m_active, packets, node, boxID, frustum.nf);
          m_node_hit ^= m_current ? (size_t)0 : ((size_t)1 << boxID);
          maskK[boxID] = m_current;
        }
        return m_node_hit;
      }
      
      // TODO: explicit 16-wide path for KNL
      template<int K>
      __forceinline static vint<Nx> traverseIncoherentStream(size_t m_active,
                                                             TravRayKStreamFast<K>* __restrict__ packets,
                                                             const AlignedNode* __restrict__ node,
                                                             const NearFarPrecalculations& nf,
                                                             const int shiftTable[32])
      {
        const vfloat<Nx> bminX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearX));
        const vfloat<Nx> bminY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearY));
        const vfloat<Nx> bminZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearZ));
        const vfloat<Nx> bmaxX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farX));
        const vfloat<Nx> bmaxY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farY));
        const vfloat<Nx> bmaxZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farZ));
        assert(m_active);
        vint<Nx> vmask(zero);
        do
        {   
          STAT3(shadow.trav_nodes,1,1,1);
          const size_t rayID = bscf(m_active);
          assert(rayID < MAX_INTERNAL_STREAM_SIZE);
          TravRayKStream<K,robust> &p = packets[rayID / K];
          const size_t i = rayID % K;
          const vint<Nx> bitmask(shiftTable[rayID]);
          const vfloat<Nx> tNearX = msub(bminX, p.rdir.x[i], p.org_rdir.x[i]);
          const vfloat<Nx> tNearY = msub(bminY, p.rdir.y[i], p.org_rdir.y[i]);
          const vfloat<Nx> tNearZ = msub(bminZ, p.rdir.z[i], p.org_rdir.z[i]);
          const vfloat<Nx> tFarX  = msub(bmaxX, p.rdir.x[i], p.org_rdir.x[i]);
          const vfloat<Nx> tFarY  = msub(bmaxY, p.rdir.y[i], p.org_rdir.y[i]);
          const vfloat<Nx> tFarZ  = msub(bmaxZ, p.rdir.z[i], p.org_rdir.z[i]); 
          const vfloat<Nx> tNear  = maxi(tNearX, tNearY, tNearZ, vfloat<Nx>(p.tnear[i]));
          const vfloat<Nx> tFar   = mini(tFarX , tFarY , tFarZ,  vfloat<Nx>(p.tfar[i]));      

#if defined(__AVX512ER__)
          const vboolx m_node((1 << N)-1);
          const vbool<Nx> hit_mask = le(m_node, tNear, tFar);
          vmask = mask_or(hit_mask, vmask, vmask, bitmask);
#else
          const vbool<Nx> hit_mask = tNear <= tFar;
#if defined(__AVX2__)
          vmask = vmask | (bitmask & vint<Nx>(hit_mask));
#else
          vmask = select(hit_mask, vmask | bitmask, vmask);
#endif
#endif
        } while(m_active);
        return vmask;        
      }

      template<int K>
      __forceinline static vint<Nx> traverseIncoherentStream(size_t m_active,
                                                             TravRayKStreamRobust<K>* __restrict__ packets,
                                                             const AlignedNode* __restrict__ node,
                                                             const NearFarPrecalculations& nf,
                                                             const int shiftTable[32])
      {
        const vfloat<Nx> bminX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearX));
        const vfloat<Nx> bminY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearY));
        const vfloat<Nx> bminZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearZ));
        const vfloat<Nx> bmaxX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farX));
        const vfloat<Nx> bmaxY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farY));
        const vfloat<Nx> bmaxZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farZ));
        assert(m_active);
        vint<Nx> vmask(zero);
        do
        {   
          STAT3(shadow.trav_nodes,1,1,1);
          const size_t rayID = bscf(m_active);
          assert(rayID < MAX_INTERNAL_STREAM_SIZE);
          TravRayKStream<K,robust> &p = packets[rayID / K];
          const size_t i = rayID % K;
          const vint<Nx> bitmask(shiftTable[rayID]);
          const vfloat<Nx> tNearX = (bminX - p.org.x[i]) * p.rdir.x[i];
          const vfloat<Nx> tNearY = (bminY - p.org.y[i]) * p.rdir.y[i];
          const vfloat<Nx> tNearZ = (bminZ - p.org.z[i]) * p.rdir.z[i];
          const vfloat<Nx> tFarX  = (bmaxX - p.org.x[i]) * p.rdir.x[i];
          const vfloat<Nx> tFarY  = (bmaxY - p.org.y[i]) * p.rdir.y[i];
          const vfloat<Nx> tFarZ  = (bmaxZ - p.org.z[i]) * p.rdir.z[i];
          const vfloat<Nx> tNear  = maxi(tNearX, tNearY, tNearZ, vfloat<Nx>(p.tnear[i]));
          const vfloat<Nx> tFar   = mini(tFarX , tFarY , tFarZ,  vfloat<Nx>(p.tfar[i]));
          const float round_down  = 1.0f-2.0f*float(ulp);
          const float round_up    = 1.0f+2.0f*float(ulp);
#if defined(__AVX512ER__)
          const vboolx m_node((1 << N)-1);
          const vbool<Nx> hit_mask = le(m_node, round_down*tNear, round_up*tFar);
          vmask = mask_or(hit_mask, vmask, vmask, bitmask);
#else
          const vbool<Nx> hit_mask = round_down*tNear <= round_up*tFar;
#if defined(__AVX2__)
          vmask = vmask | (bitmask & vint<Nx>(hit_mask));
#else
          vmask = select(hit_mask, vmask | bitmask, vmask);
#endif
#endif
        } while(m_active);
        return vmask;
      }
                                                         

      static const size_t stackSizeSingle = 1+(N-1)*BVH::maxDepth;

    public:
      static void intersect(Accel::Intersectors* This, RayHitN** inputRays, size_t numRays, IntersectContext* context);
      static void occluded (Accel::Intersectors* This, RayN** inputRays, size_t numRays, IntersectContext* context);

    private:
      template<int K>
      static void intersectCoherent(Accel::Intersectors* This, RayHitK<K>** inputRays, size_t numRays, IntersectContext* context);

      template<int K>
      static void occludedCoherent(Accel::Intersectors* This, RayK<K>** inputRays, size_t numRays, IntersectContext* context);

      template<int K>
      static void occludedIncoherent(Accel::Intersectors* This, RayK<K>** inputRays, size_t numRays, IntersectContext* context);
    };


    /*! BVH ray stream intersector with direct fallback to packets. */
    template<int N, int Nx>
    class BVHNIntersectorStreamPacketFallback
    {
    public:
      static void intersect(Accel::Intersectors* This, RayHitN** inputRays, size_t numRays, IntersectContext* context);
      static void occluded (Accel::Intersectors* This, RayN** inputRays, size_t numRays, IntersectContext* context);

    private:
      template<int K>
      static void intersectK(Accel::Intersectors* This, RayHitK<K>** inputRays, size_t numRays, IntersectContext* context);

      template<int K>
      static void occludedK(Accel::Intersectors* This, RayK<K>** inputRays, size_t numRays, IntersectContext* context);
    };
  }
}
