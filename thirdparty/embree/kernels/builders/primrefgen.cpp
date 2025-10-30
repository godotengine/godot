// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "primrefgen.h"
#include "primrefgen_presplit.h"

#include "../../common/algorithms/parallel_for_for.h"
#include "../../common/algorithms/parallel_for_for_prefix_sum.h"

namespace embree
{
  namespace isa
  {
    PrimInfo createPrimRefArray(Geometry* geometry, unsigned int geomID, const size_t numPrimRefs, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor)
    {
      ParallelPrefixSumState<PrimInfo> pstate;
      
      /* first try */
      progressMonitor(0);
      PrimInfo pinfo = parallel_prefix_sum( pstate, size_t(0), geometry->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo {
          return geometry->createPrimRefArray(prims,r,r.begin(),geomID);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });

      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
      {
        progressMonitor(0);
        pinfo = parallel_prefix_sum( pstate, size_t(0), geometry->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo {
          return geometry->createPrimRefArray(prims,r,base.size(),geomID);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      }
      return pinfo;
    }

    PrimInfo createPrimRefArray(Scene* scene, Geometry::GTypeMask types, bool mblur, const size_t numPrimRefs, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor)
    {
      ParallelForForPrefixSumState<PrimInfo> pstate;
      Scene::Iterator2 iter(scene,types,mblur);
      
      /* first try */
      progressMonitor(0);
      pstate.init(iter,size_t(1024));
      PrimInfo pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfo {
          return mesh->createPrimRefArray(prims,r,k,(unsigned)geomID);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      
      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
      {
        progressMonitor(0);
        pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfo& base) -> PrimInfo {
            return mesh->createPrimRefArray(prims,r,base.size(),(unsigned)geomID);
          }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      }
      return pinfo;
    }

    PrimInfo createPrimRefArray(Scene* scene, Geometry::GTypeMask types, bool mblur, const size_t numPrimRefs, mvector<PrimRef>& prims, mvector<SubGridBuildData>& sgrids, BuildProgressMonitor& progressMonitor)
    {
      ParallelForForPrefixSumState<PrimInfo> pstate;
      Scene::Iterator2 iter(scene,types,mblur);
      
      /* first try */
      progressMonitor(0);
      pstate.init(iter,size_t(1024));
      PrimInfo pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfo {
         return mesh->createPrimRefArray(prims,sgrids,r,k,(unsigned)geomID);
       }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      
      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
      {
        progressMonitor(0);
        pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfo& base) -> PrimInfo {
          return mesh->createPrimRefArray(prims,sgrids,r,base.size(),(unsigned)geomID);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      }
      return pinfo;
    }

    PrimInfo createPrimRefArrayMBlur(Scene* scene, Geometry::GTypeMask types, const size_t numPrimRefs, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor, size_t itime)
    {
      ParallelForForPrefixSumState<PrimInfo> pstate;
      Scene::Iterator2 iter(scene,types,true);
      
      /* first try */
      progressMonitor(0);
      pstate.init(iter,size_t(1024));
      PrimInfo pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfo {
          return mesh->createPrimRefArrayMB(prims,itime,r,k,(unsigned)geomID);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      
      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
      {
        progressMonitor(0);
        pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfo& base) -> PrimInfo {
            return mesh->createPrimRefArrayMB(prims,itime,r,base.size(),(unsigned)geomID);
          }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      }
      return pinfo;
    }

    PrimInfoMB createPrimRefArrayMSMBlur(Scene* scene, Geometry::GTypeMask types, const size_t numPrimRefs, mvector<PrimRefMB>& prims, BuildProgressMonitor& progressMonitor, BBox1f t0t1)
    {
      ParallelForForPrefixSumState<PrimInfoMB> pstate;
      Scene::Iterator2 iter(scene,types,true);
      
      /* first try */
      progressMonitor(0);
      pstate.init(iter,size_t(1024));
      PrimInfoMB pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfoMB(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfoMB {
          return mesh->createPrimRefMBArray(prims,t0t1,r,k,(unsigned)geomID);
      }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
      
      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
      {
        progressMonitor(0);
        pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfoMB(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfoMB& base) -> PrimInfoMB {
            return mesh->createPrimRefMBArray(prims,t0t1,r,base.size(),(unsigned)geomID);
        }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
      }

      /* the BVH starts with that time range, even though primitives might have smaller/larger time range */
      pinfo.time_range = t0t1;
      return pinfo;
    }

    PrimInfoMB createPrimRefArrayMSMBlur(Scene* scene, Geometry::GTypeMask types, const size_t numPrimRefs, mvector<PrimRefMB>& prims, mvector<SubGridBuildData>& sgrids, BuildProgressMonitor& progressMonitor, BBox1f t0t1)
    {
      ParallelForForPrefixSumState<PrimInfoMB> pstate;
      Scene::Iterator2 iter(scene,types,true);
      
      /* first try */
      progressMonitor(0);
      pstate.init(iter,size_t(1024));
      PrimInfoMB pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfoMB(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfoMB {
         return mesh->createPrimRefMBArray(prims,sgrids,t0t1,r,k,(unsigned)geomID);
      }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
      
      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
      {
        progressMonitor(0);
        pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfoMB(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfoMB& base) -> PrimInfoMB {
          return mesh->createPrimRefMBArray(prims,sgrids,t0t1,r,base.size(),(unsigned)geomID);
        }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
      }

      /* the BVH starts with that time range, even though primitives might have smaller/larger time range */
      pinfo.time_range = t0t1;
      return pinfo;
    }

    template<typename Mesh>
    size_t createMortonCodeArray(Mesh* mesh, mvector<BVHBuilderMorton::BuildPrim>& morton, BuildProgressMonitor& progressMonitor)
    {
      size_t numPrimitives = morton.size();

      /* compute scene bounds */
      std::pair<size_t,BBox3fa> cb_empty(0,empty);
      auto cb = parallel_reduce 
        ( size_t(0), numPrimitives, size_t(1024), cb_empty, [&](const range<size_t>& r) -> std::pair<size_t,BBox3fa>
          {
            size_t num = 0;
            BBox3fa bounds = empty;
            
            for (size_t j=r.begin(); j<r.end(); j++)
            {
              BBox3fa prim_bounds = empty;
              if (unlikely(!mesh->buildBounds(j,&prim_bounds))) continue;
              bounds.extend(center2(prim_bounds));
              num++;
            }
            return std::make_pair(num,bounds);
          }, [] (const std::pair<size_t,BBox3fa>& a, const std::pair<size_t,BBox3fa>& b) {
          return std::make_pair(a.first + b.first,merge(a.second,b.second)); 
        });
      
      
      size_t numPrimitivesGen = cb.first;
      const BBox3fa centBounds = cb.second;
      
      /* compute morton codes */
      if (likely(numPrimitivesGen == numPrimitives))
      {
        /* fast path if all primitives were valid */
        BVHBuilderMorton::MortonCodeMapping mapping(centBounds);
        parallel_for( size_t(0), numPrimitives, size_t(1024), [&](const range<size_t>& r) -> void {
            BVHBuilderMorton::MortonCodeGenerator generator(mapping,&morton.data()[r.begin()]);
            for (size_t j=r.begin(); j<r.end(); j++)
              generator(mesh->bounds(j),unsigned(j));
          });
      }
      else
      {
        /* slow path, fallback in case some primitives were invalid */
        ParallelPrefixSumState<size_t> pstate;
        BVHBuilderMorton::MortonCodeMapping mapping(centBounds);
        parallel_prefix_sum( pstate, size_t(0), numPrimitives, size_t(1024), size_t(0), [&](const range<size_t>& r, const size_t base) -> size_t {
            size_t num = 0;
            BVHBuilderMorton::MortonCodeGenerator generator(mapping,&morton.data()[r.begin()]);
            for (size_t j=r.begin(); j<r.end(); j++)
            {
              BBox3fa bounds = empty;
              if (unlikely(!mesh->buildBounds(j,&bounds))) continue;
              generator(bounds,unsigned(j));
              num++;
            }
            return num;
          }, std::plus<size_t>());
        
        parallel_prefix_sum( pstate, size_t(0), numPrimitives, size_t(1024), size_t(0), [&](const range<size_t>& r, const size_t base) -> size_t {
            size_t num = 0;
            BVHBuilderMorton::MortonCodeGenerator generator(mapping,&morton.data()[base]);
            for (size_t j=r.begin(); j<r.end(); j++)
            {
              BBox3fa bounds = empty;
              if (!mesh->buildBounds(j,&bounds)) continue;
              generator(bounds,unsigned(j));
              num++;
            }
            return num;
          }, std::plus<size_t>());          
      }
      return numPrimitivesGen;
    }

    // ====================================================================================================
    // ====================================================================================================
    // ====================================================================================================

    // special variants for grid meshes

#if defined(EMBREE_GEOMETRY_GRID)
    PrimInfo createPrimRefArrayGrids(Scene* scene, mvector<PrimRef>& prims, mvector<SubGridBuildData>& sgrids)
    {
      PrimInfo pinfo(empty);
      size_t numPrimitives = 0;
      
      /* first run to get #primitives */

      ParallelForForPrefixSumState<PrimInfo> pstate;
      Scene::Iterator<GridMesh,false> iter(scene);

      pstate.init(iter,size_t(1024));

      /* iterate over all meshes in the scene */
      pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfo {
          PrimInfo pinfo(empty);
          for (size_t j=r.begin(); j<r.end(); j++)
          {
            if (!mesh->valid(j)) continue;
            BBox3fa bounds = empty;
            const PrimRef prim(bounds,(unsigned)geomID,(unsigned)j);
            if (!mesh->valid(j)) continue;
            pinfo.add_center2(prim,mesh->getNumSubGrids(j));
          }
          return pinfo;
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      numPrimitives = pinfo.size();
          
      /* resize arrays */
      sgrids.resize(numPrimitives); 
      prims.resize(numPrimitives); 

      /* second run to fill primrefs and SubGridBuildData arrays */
      pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfo& base) -> PrimInfo {
        return mesh->createPrimRefArray(prims,sgrids,r,base.size(),geomID);
      }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      assert(pinfo.size() == numPrimitives);
      return pinfo;
    }

    PrimInfo createPrimRefArrayGrids(GridMesh* mesh, mvector<PrimRef>& prims, mvector<SubGridBuildData>& sgrids)
    {
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max ();

      PrimInfo pinfo(empty);
      size_t numPrimitives = 0;
      
      ParallelPrefixSumState<PrimInfo> pstate;
      /* iterate over all grids in a single mesh */
      pinfo = parallel_prefix_sum( pstate, size_t(0), mesh->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo
                                   {
                                     PrimInfo pinfo(empty);
                                     for (size_t j=r.begin(); j<r.end(); j++)
                                     {
                                       if (!mesh->valid(j)) continue;
                                       BBox3fa bounds = empty;
                                       const PrimRef prim(bounds,geomID_,unsigned(j));
                                       pinfo.add_center2(prim,mesh->getNumSubGrids(j));
                                     }
                                     return pinfo;
                                   }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      numPrimitives = pinfo.size();
      /* resize arrays */
      sgrids.resize(numPrimitives); 
      prims.resize(numPrimitives); 

      /* second run to fill primrefs and SubGridBuildData arrays */
      pinfo = parallel_prefix_sum( pstate, size_t(0), mesh->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo {
        return mesh->createPrimRefArray(prims,sgrids,r,base.size(),geomID_);
      }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });

      return pinfo;
    }

    PrimInfoMB createPrimRefArrayMSMBlurGrid(Scene* scene, mvector<PrimRefMB>& prims, mvector<SubGridBuildData>& sgrids, BuildProgressMonitor& progressMonitor, BBox1f t0t1)
    {
      /* first run to get #primitives */
      ParallelForForPrefixSumState<PrimInfoMB> pstate;
      Scene::Iterator<GridMesh,true> iter(scene);
      
      pstate.init(iter,size_t(1024));
      /* iterate over all meshes in the scene */
      PrimInfoMB pinfoMB = parallel_for_for_prefix_sum0( pstate, iter, PrimInfoMB(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t /*geomID*/) -> PrimInfoMB {
                                                                                            
         PrimInfoMB pinfoMB(empty);
         for (size_t j=r.begin(); j<r.end(); j++)
         {
           if (!mesh->valid(j, mesh->timeSegmentRange(t0t1))) continue;
           LBBox3fa bounds(empty);
           PrimInfoMB gridMB(0,mesh->getNumSubGrids(j));
           pinfoMB.merge(gridMB);
         }
         return pinfoMB;
      }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
      
      size_t numPrimitives = pinfoMB.size();
      if (numPrimitives == 0) return pinfoMB;
      
      /* resize arrays */
      sgrids.resize(numPrimitives); 
      prims.resize(numPrimitives); 
      /* second run to fill primrefs and SubGridBuildData arrays */
      pinfoMB = parallel_for_for_prefix_sum1( pstate, iter, PrimInfoMB(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfoMB& base) -> PrimInfoMB {
        return mesh->createPrimRefMBArray(prims,sgrids,t0t1,r,base.size(),(unsigned)geomID);                                                                                 
      }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
      
      assert(pinfoMB.size() == numPrimitives);
      pinfoMB.time_range = t0t1;
      return pinfoMB;
    }
    
#endif
    
    // ====================================================================================================
    // ====================================================================================================
    // ====================================================================================================
    
    IF_ENABLED_TRIS (template size_t createMortonCodeArray<TriangleMesh>(TriangleMesh* mesh COMMA mvector<BVHBuilderMorton::BuildPrim>& morton COMMA BuildProgressMonitor& progressMonitor));
    IF_ENABLED_QUADS(template size_t createMortonCodeArray<QuadMesh>(QuadMesh* mesh COMMA mvector<BVHBuilderMorton::BuildPrim>& morton COMMA BuildProgressMonitor& progressMonitor));
    IF_ENABLED_USER (template size_t createMortonCodeArray<UserGeometry>(UserGeometry* mesh COMMA mvector<BVHBuilderMorton::BuildPrim>& morton COMMA BuildProgressMonitor& progressMonitor));
    IF_ENABLED_INSTANCE (template size_t createMortonCodeArray<Instance>(Instance* mesh COMMA mvector<BVHBuilderMorton::BuildPrim>& morton COMMA BuildProgressMonitor& progressMonitor));
    IF_ENABLED_INSTANCE_ARRAY (template size_t createMortonCodeArray<InstanceArray>(InstanceArray* mesh COMMA mvector<BVHBuilderMorton::BuildPrim>& morton COMMA BuildProgressMonitor& progressMonitor));
  }
}
