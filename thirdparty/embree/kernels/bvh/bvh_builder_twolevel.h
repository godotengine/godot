// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "bvh_builder_twolevel_internal.h"
#include "bvh.h"
#include "../builders/priminfo.h"
#include "../builders/primrefgen.h"

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
    template<int N, typename Mesh, typename Primitive>
    class BVHNBuilderTwoLevel : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::AABBNode AABBNode;
      typedef typename BVH::NodeRef NodeRef;

      __forceinline static bool isSmallGeometry(Mesh* mesh) {
        return mesh->size() <= 4;
      }

    public:

      typedef void (*createMeshAccelTy)(Scene* scene, unsigned int geomID, AccelData*& accel, Builder*& builder);

      struct BuildRef : public PrimRef
      {
      public:
        __forceinline BuildRef () {}

        __forceinline BuildRef (const BBox3fa& bounds, NodeRef node)
          : PrimRef(bounds,(size_t)node), node(node)
        {
          if (node.isLeaf())
            bounds_area = 0.0f;
          else
            bounds_area = area(this->bounds());
        }

        /* used by the open/merge bvh builder */
        __forceinline BuildRef (const BBox3fa& bounds, NodeRef node, const unsigned int geomID, const unsigned int numPrimitives)
          : PrimRef(bounds,geomID,numPrimitives), node(node)
        {
          /* important for relative buildref ordering */
          if (node.isLeaf())
            bounds_area = 0.0f;
          else
            bounds_area = area(this->bounds());
        }

        __forceinline size_t size() const {
          return primID();
        }

        friend bool operator< (const BuildRef& a, const BuildRef& b) {
          return a.bounds_area < b.bounds_area;
        }

        friend __forceinline embree_ostream operator<<(embree_ostream cout, const BuildRef& ref) {
          return cout << "{ lower = " << ref.lower << ", upper = " << ref.upper << ", center2 = " << ref.center2() << ", geomID = " << ref.geomID() << ", numPrimitives = " << ref.numPrimitives() << ", bounds_area = " << ref.bounds_area << " }";
        }

        __forceinline unsigned int numPrimitives() const { return primID(); }

      public:
        NodeRef node;
        float bounds_area;
      };


      __forceinline size_t openBuildRef(BuildRef &bref, BuildRef *const refs) {
        if (bref.node.isLeaf())
        {
          refs[0] = bref;
          return 1;
        }
        NodeRef ref = bref.node;
        unsigned int geomID   = bref.geomID();
        unsigned int numPrims = max((unsigned int)bref.numPrimitives() / N,(unsigned int)1);
        AABBNode* node = ref.getAABBNode();
        size_t n = 0;
        for (size_t i=0; i<N; i++) {
          if (node->child(i) == BVH::emptyNode) continue;
          refs[i] = BuildRef(node->bounds(i),node->child(i),geomID,numPrims);
          n++;
        }
        assert(n > 1);
        return n;        
      }
      
      /*! Constructor. */
      BVHNBuilderTwoLevel (BVH* bvh, Scene* scene, Geometry::GTypeMask gtype = Mesh::geom_type, bool useMortonBuilder = false, const size_t singleThreadThreshold = DEFAULT_SINGLE_THREAD_THRESHOLD);
      
      /*! Destructor */
      ~BVHNBuilderTwoLevel ();
      
      /*! builder entry point */
      void build();
      void deleteGeometry(size_t geomID);
      void clear();

      void open_sequential(const size_t extSize);
      
    private:

      class RefBuilderBase {
      public:
        virtual ~RefBuilderBase () {}
        virtual void attachBuildRefs (BVHNBuilderTwoLevel* builder) = 0;
        virtual bool meshQualityChanged (RTCBuildQuality currQuality) = 0;
      };

      class RefBuilderSmall : public RefBuilderBase {
      public:

        RefBuilderSmall (size_t objectID)
          : objectID_ (objectID) {}

        void attachBuildRefs (BVHNBuilderTwoLevel* topBuilder) {

          Mesh* mesh = topBuilder->scene->template getSafe<Mesh>(objectID_);
          size_t meshSize = mesh->size();
          assert(isSmallGeometry(mesh));
          
          mvector<PrimRef> prefs(topBuilder->scene->device, meshSize);
          auto pinfo = createPrimRefArray(mesh,objectID_,meshSize,prefs,topBuilder->bvh->scene->progressInterface);

          size_t begin=0;
          while (begin < pinfo.size())
          {
            Primitive* accel = (Primitive*) topBuilder->bvh->alloc.getCachedAllocator().malloc1(sizeof(Primitive),BVH::byteAlignment);
            typename BVH::NodeRef node = BVH::encodeLeaf((char*)accel,1);
            accel->fill(prefs.data(),begin,pinfo.size(),topBuilder->bvh->scene);
            
            /* create build primitive */
#if ENABLE_DIRECT_SAH_MERGE_BUILDER
            topBuilder->refs[topBuilder->nextRef++] = BVHNBuilderTwoLevel::BuildRef(pinfo.geomBounds,node,(unsigned int)objectID_,1);
#else
            topBuilder->refs[topBuilder->nextRef++] = BVHNBuilderTwoLevel::BuildRef(pinfo.geomBounds,node);
#endif
          }
          assert(begin == pinfo.size());
        }

        bool meshQualityChanged (RTCBuildQuality /*currQuality*/) {
          return false;
        }
        
        size_t  objectID_;
      };

      class RefBuilderLarge : public RefBuilderBase {
      public:
        
        RefBuilderLarge (size_t objectID, const Ref<Builder>& builder, RTCBuildQuality quality)
        : objectID_ (objectID), builder_ (builder), quality_ (quality) {}

        void attachBuildRefs (BVHNBuilderTwoLevel* topBuilder)
        {
          BVH* object  = topBuilder->getBVH(objectID_); assert(object);
          
          /* build object if it got modified */
          if (topBuilder->isGeometryModified(objectID_))
            builder_->build();

          /* create build primitive */
          if (!object->getBounds().empty())
          {
#if ENABLE_DIRECT_SAH_MERGE_BUILDER
            Mesh* mesh = topBuilder->getMesh(objectID_);
            topBuilder->refs[topBuilder->nextRef++] = BVHNBuilderTwoLevel::BuildRef(object->getBounds(),object->root,(unsigned int)objectID_,(unsigned int)mesh->size());
#else
            topBuilder->refs[topBuilder->nextRef++] = BVHNBuilderTwoLevel::BuildRef(object->getBounds(),object->root);
#endif
          }
        }

        bool meshQualityChanged (RTCBuildQuality currQuality) {
          return currQuality != quality_;
        }

      private:
        size_t          objectID_;
        Ref<Builder>    builder_;
        RTCBuildQuality quality_;
      };

      void setupLargeBuildRefBuilder (size_t objectID, Mesh const * const mesh);
      void setupSmallBuildRefBuilder (size_t objectID, Mesh const * const mesh);

      BVH*  getBVH (size_t objectID) {
        return this->bvh->objects[objectID];
      }
      Mesh* getMesh (size_t objectID) {
        return this->scene->template getSafe<Mesh>(objectID);
      }
      bool  isGeometryModified (size_t objectID) {
        return this->scene->isGeometryModified(objectID);
      }

      void resizeRefsList ()
      {
        size_t num = parallel_reduce (size_t(0), scene->size(), size_t(0), 
          [this](const range<size_t>& r)->size_t {
            size_t c = 0;
            for (auto i=r.begin(); i<r.end(); ++i) {
              Mesh* mesh = scene->getSafe<Mesh>(i);
              if (mesh == nullptr || mesh->numTimeSteps != 1)
                continue;
              size_t meshSize = mesh->size();
              c += isSmallGeometry(mesh) ? Primitive::blocks(meshSize) : 1;
            }
            return c;
          },
          std::plus<size_t>()
        );

        if (refs.size() < num) {
          refs.resize(num);
        }
      }

      void createMeshAccel (size_t geomID, Builder*& builder)
      {
        bvh->objects[geomID] = new BVH(Primitive::type,scene);
        BVH* accel = bvh->objects[geomID];
        auto mesh = scene->getSafe<Mesh>(geomID);
        if (nullptr == mesh) {
          throw_RTCError(RTC_ERROR_INVALID_ARGUMENT,"geomID does not return correct type");
          return;
        }

        __internal_two_level_builder__::MeshBuilder<N,Mesh,Primitive>()(accel, mesh, geomID, this->gtype, this->useMortonBuilder_, builder);
      }      

      using BuilderList = std::vector<std::unique_ptr<RefBuilderBase>>;

      BuilderList         builders;
      BVH*                bvh;
      Scene*              scene;      
      mvector<BuildRef>   refs;
      mvector<PrimRef>    prims;
      std::atomic<int>    nextRef;
      const size_t        singleThreadThreshold;
      Geometry::GTypeMask gtype;
      bool                useMortonBuilder_ = false;
    };
  }
}
