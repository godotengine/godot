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

#include "bvh_builder_twolevel.h"
#include "bvh_statistics.h"
#include "../builders/bvh_builder_sah.h"
#include "../common/scene_line_segments.h"
#include "../common/scene_triangle_mesh.h"
#include "../common/scene_quad_mesh.h"

#define PROFILE 0

/* new open/merge builder */
#define ENABLE_DIRECT_SAH_MERGE_BUILDER 1
#define ENABLE_OPEN_SEQUENTIAL 0
#define SPLIT_MEMORY_RESERVE_FACTOR 1000
#define SPLIT_MEMORY_RESERVE_SCALE 2
#define SPLIT_MIN_EXT_SPACE 1000

namespace embree
{
  namespace isa
  {
    template<int N, typename Mesh>
    BVHNBuilderTwoLevel<N,Mesh>::BVHNBuilderTwoLevel (BVH* bvh, Scene* scene, const createMeshAccelTy createMeshAccel, const size_t singleThreadThreshold)
      : bvh(bvh), objects(bvh->objects), scene(scene), createMeshAccel(createMeshAccel), refs(scene->device,0), prims(scene->device,0), singleThreadThreshold(singleThreadThreshold) {}
    
    template<int N, typename Mesh>
    BVHNBuilderTwoLevel<N,Mesh>::~BVHNBuilderTwoLevel () {
    }

    // ===========================================================================
    // ===========================================================================
    // ===========================================================================

    template<int N, typename Mesh>
    void BVHNBuilderTwoLevel<N,Mesh>::build()
    {
      /* delete some objects */
      size_t num = scene->size();
      if (num < objects.size()) {
        parallel_for(num, objects.size(), [&] (const range<size_t>& r) {
            for (size_t i=r.begin(); i<r.end(); i++) {
              builders[i].clear();
              delete objects[i]; objects[i] = nullptr;
            }
          });
      }
      
#if PROFILE
      while(1) 
#endif
      {
      /* reset memory allocator */
      bvh->alloc.reset();
      
      /* skip build for empty scene */
      const size_t numPrimitives = scene->getNumPrimitives<Mesh,false>();

      if (numPrimitives == 0) {
        prims.resize(0);
        bvh->set(BVH::emptyNode,empty,0);
        return;
      }

      double t0 = bvh->preBuild(TOSTRING(isa) "::BVH" + toString(N) + "BuilderTwoLevel");

      /* resize object array if scene got larger */
      if (objects.size()  < num) objects.resize(num);
      if (builders.size() < num) builders.resize(num);
      if (refs.size()     < num) refs.resize(num);
      nextRef.store(0);
      
      /* create acceleration structures */
      parallel_for(size_t(0), num, [&] (const range<size_t>& r)
      {
        for (size_t objectID=r.begin(); objectID<r.end(); objectID++)
        {
          Mesh* mesh = scene->getSafe<Mesh>(objectID);
          
          /* ignore meshes we do not support */
          if (mesh == nullptr || mesh->numTimeSteps != 1)
            continue;
          
          /* create BVH and builder for new meshes */
          if (objects[objectID] == nullptr) {
            Builder* builder = nullptr;
            createMeshAccel(mesh,(AccelData*&)objects[objectID],builder);
            builders[objectID] = BuilderState(builder,mesh->quality);
          }

          /* re-create when build quality changed */
          else if (mesh->quality != builders[objectID].quality) {
            Builder* builder = nullptr;
            delete objects[objectID]; 
            createMeshAccel(mesh,(AccelData*&)objects[objectID],builder);
            builders[objectID] = BuilderState(builder,mesh->quality);
          }
        }
      });

      /* parallel build of acceleration structures */
      parallel_for(size_t(0), num, [&] (const range<size_t>& r)
      {
        for (size_t objectID=r.begin(); objectID<r.end(); objectID++)
        {
          /* ignore if no triangle mesh or not enabled */
          Mesh* mesh = scene->getSafe<Mesh>(objectID);
          if (mesh == nullptr || !mesh->isEnabled() || mesh->numTimeSteps != 1) 
            continue;
        
          BVH*     object  = objects [objectID]; assert(object);
          Ref<Builder>& builder = builders[objectID].builder; assert(builder);
          
          /* build object if it got modified */
          if (mesh->isModified())
            builder->build();

          /* create build primitive */
          if (!object->getBounds().empty())
          {
#if ENABLE_DIRECT_SAH_MERGE_BUILDER
            refs[nextRef++] = BVHNBuilderTwoLevel::BuildRef(object->getBounds(),object->root,(unsigned int)objectID,(unsigned int)mesh->size());
#else
            refs[nextRef++] = BVHNBuilderTwoLevel::BuildRef(object->getBounds(),object->root);
#endif
          }
        }
      });


#if PROFILE
      double d0 = getSeconds();
#endif
      /* fast path for single geometry scenes */
      if (nextRef == 1) { 
        bvh->set(refs[0].node,LBBox3fa(refs[0].bounds()),numPrimitives);
      }

      else
      {     
        /* open all large nodes */
        refs.resize(nextRef);

        /* this probably needs some more tuning */
        const size_t extSize = max(max((size_t)SPLIT_MIN_EXT_SPACE,refs.size()*SPLIT_MEMORY_RESERVE_SCALE),size_t((float)numPrimitives / SPLIT_MEMORY_RESERVE_FACTOR));
        //PRINT(extSize);
 
#if !ENABLE_DIRECT_SAH_MERGE_BUILDER

#if ENABLE_OPEN_SEQUENTIAL
        open_sequential(extSize); 
#endif
        /* compute PrimRefs */
        prims.resize(refs.size());
#endif

        /* calculate the size of the entire BVH */
        const size_t node_bytes = numPrimitives*sizeof(typename BVH::AlignedNodeMB)/(4*N);
        const size_t leaf_bytes = size_t(1.2*44*numPrimitives); // assumes triangles
        bvh->alloc.init_estimate(node_bytes+leaf_bytes); 

#if defined(TASKING_TBB) && defined(__AVX512ER__) && USE_TASK_ARENA // KNL
        tbb::task_arena limited(min(32,(int)TaskScheduler::threadCount()));
        limited.execute([&]
#endif
        {
#if ENABLE_DIRECT_SAH_MERGE_BUILDER

          const PrimInfo pinfo = parallel_reduce(size_t(0), refs.size(),  PrimInfo(empty), [&] (const range<size_t>& r) -> PrimInfo {

              PrimInfo pinfo(empty);
              for (size_t i=r.begin(); i<r.end(); i++) {
                pinfo.add_center2(refs[i]);
              }
              return pinfo;
            }, [] (const PrimInfo& a, const PrimInfo& b) { return PrimInfo::merge(a,b); });
          
#else
          const PrimInfo pinfo = parallel_reduce(size_t(0), refs.size(),  PrimInfo(empty), [&] (const range<size_t>& r) -> PrimInfo {

              PrimInfo pinfo(empty);
              for (size_t i=r.begin(); i<r.end(); i++) {
                pinfo.add(refs[i].bounds());
                prims[i] = PrimRef(refs[i].bounds(),(size_t)refs[i].node);
              }
              return pinfo;
            }, [] (const PrimInfo& a, const PrimInfo& b) { return PrimInfo::merge(a,b); });
#endif   
       
          /* skip if all objects where empty */
          if (pinfo.size() == 0)
            bvh->set(BVH::emptyNode,empty,0);
        
          /* otherwise build toplevel hierarchy */
          else
          {
            /* settings for BVH build */
            GeneralBVHBuilder::Settings settings;
            settings.branchingFactor = N;
            settings.maxDepth = BVH::maxBuildDepthLeaf;
            settings.logBlockSize = bsr(N);
            settings.minLeafSize = 1;
            settings.maxLeafSize = 1;
            settings.travCost = 1.0f;
            settings.intCost = 1.0f;
            settings.singleThreadThreshold = singleThreadThreshold;
      
#if ENABLE_DIRECT_SAH_MERGE_BUILDER
            refs.resize(extSize); 
         
            NodeRef root = BVHBuilderBinnedOpenMergeSAH::build<NodeRef,BuildRef>(
              typename BVH::CreateAlloc(bvh),
              typename BVH::AlignedNode::Create2(),
              typename BVH::AlignedNode::Set2(),
              
              [&] (const BuildRef* refs, const range<size_t>& range, const FastAllocator::CachedAllocator& alloc) -> NodeRef  {
                assert(range.size() == 1);
                return (NodeRef) refs[range.begin()].node;
              },
              [&] (BuildRef &bref, BuildRef *refs) -> size_t { 
                return openBuildRef(bref,refs);
              },              
              [&] (size_t dn) { bvh->scene->progressMonitor(0); },
              refs.data(),extSize,pinfo,settings);
#else
            NodeRef root = BVHBuilderBinnedSAH::build<NodeRef>(
              typename BVH::CreateAlloc(bvh),
              typename BVH::AlignedNode::Create2(),
              typename BVH::AlignedNode::Set2(),
              
              [&] (const PrimRef* pims, const range<size_t>& range, const FastAllocator::CachedAllocator& alloc) -> NodeRef {
                assert(range.size() == 1);
                return (NodeRef) prims[range.begin()].ID();
              },
              [&] (size_t dn) { bvh->scene->progressMonitor(0); },
              prims.data(),pinfo,settings);
#endif

            
            bvh->set(root,LBBox3fa(pinfo.geomBounds),numPrimitives);
          }
        }
#if defined(TASKING_TBB) && defined(__AVX512ER__) && USE_TASK_ARENA // KNL
          );
#endif

      }  
        
      bvh->alloc.cleanup();
      bvh->postBuild(t0);
#if PROFILE
      double d1 = getSeconds();
      std::cout << "TOP_LEVEL OPENING/REBUILD TIME " << 1000.0*(d1-d0) << " ms" << std::endl;
#endif
      }

    }
    
    template<int N, typename Mesh>
    void BVHNBuilderTwoLevel<N,Mesh>::deleteGeometry(size_t geomID)
    {
      if (geomID >= objects.size()) return;
      builders[geomID].clear();
      delete objects [geomID]; objects [geomID] = nullptr;
    }

    template<int N, typename Mesh>
    void BVHNBuilderTwoLevel<N,Mesh>::clear()
    {
      for (size_t i=0; i<objects.size(); i++) 
        if (objects[i]) objects[i]->clear();

      for (size_t i=0; i<builders.size(); i++) 
	if (builders[i].builder) builders[i].builder->clear();

      refs.clear();
    }

    template<int N, typename Mesh>
    void BVHNBuilderTwoLevel<N,Mesh>::open_sequential(const size_t extSize)
    {
      if (refs.size() == 0)
	return;

      refs.reserve(extSize);

#if 1
      for (size_t i=0;i<refs.size();i++)
      {
        NodeRef ref = refs[i].node;
        if (ref.isAlignedNode())
          ref.prefetch();
      }
#endif

      std::make_heap(refs.begin(),refs.end());
      while (refs.size()+N-1 <= extSize)
      {
        std::pop_heap (refs.begin(),refs.end()); 
        NodeRef ref = refs.back().node;
        if (ref.isLeaf()) break;
        refs.pop_back();    
        
        AlignedNode* node = ref.alignedNode();
        for (size_t i=0; i<N; i++) {
          if (node->child(i) == BVH::emptyNode) continue;
          refs.push_back(BuildRef(node->bounds(i),node->child(i)));
         
#if 1
          NodeRef ref_pre = node->child(i);
          if (ref_pre.isAlignedNode())
            ref_pre.prefetch();
#endif
          std::push_heap (refs.begin(),refs.end()); 
        }
      }
    }

#if defined(EMBREE_GEOMETRY_TRIANGLE)
    Builder* BVH4BuilderTwoLevelTriangleMeshSAH (void* bvh, Scene* scene, const createTriangleMeshAccelTy createMeshAccel) {
      return new BVHNBuilderTwoLevel<4,TriangleMesh>((BVH4*)bvh,scene,createMeshAccel);
    }
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
    Builder* BVH4BuilderTwoLevelQuadMeshSAH (void* bvh, Scene* scene, const createQuadMeshAccelTy createMeshAccel) {
    return new BVHNBuilderTwoLevel<4,QuadMesh>((BVH4*)bvh,scene,createMeshAccel);
    }
#endif

#if defined(EMBREE_GEOMETRY_USER)
    Builder* BVH4BuilderTwoLevelVirtualSAH (void* bvh, Scene* scene, const createUserGeometryAccelTy createMeshAccel) {
    return new BVHNBuilderTwoLevel<4,UserGeometry>((BVH4*)bvh,scene,createMeshAccel);
    }
#endif


#if defined(__AVX__)
#if defined(EMBREE_GEOMETRY_TRIANGLE)
    Builder* BVH8BuilderTwoLevelTriangleMeshSAH (void* bvh, Scene* scene, const createTriangleMeshAccelTy createMeshAccel) {
      return new BVHNBuilderTwoLevel<8,TriangleMesh>((BVH8*)bvh,scene,createMeshAccel);
    }
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
    Builder* BVH8BuilderTwoLevelQuadMeshSAH (void* bvh, Scene* scene, const createQuadMeshAccelTy createMeshAccel) {
      return new BVHNBuilderTwoLevel<8,QuadMesh>((BVH8*)bvh,scene,createMeshAccel);
    }
#endif

#if defined(EMBREE_GEOMETRY_USER)
    Builder* BVH8BuilderTwoLevelVirtualSAH (void* bvh, Scene* scene, const createUserGeometryAccelTy createMeshAccel) {
      return new BVHNBuilderTwoLevel<8,UserGeometry>((BVH8*)bvh,scene,createMeshAccel);
    }
#endif
#endif
  }
}
