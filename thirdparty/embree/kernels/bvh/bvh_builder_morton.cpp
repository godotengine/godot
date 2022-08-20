// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh.h"
#include "bvh_statistics.h"
#include "bvh_rotate.h"
#include "../common/profile.h"
#include "../../common/algorithms/parallel_prefix_sum.h"

#include "../builders/primrefgen.h"
#include "../builders/bvh_builder_morton.h"

#include "../geometry/triangle.h"
#include "../geometry/trianglev.h"
#include "../geometry/trianglei.h"
#include "../geometry/quadv.h"
#include "../geometry/quadi.h"
#include "../geometry/object.h"
#include "../geometry/instance.h"

#if defined(__64BIT__)
#  define ROTATE_TREE 1 // specifies number of tree rotation rounds to perform
#else
#  define ROTATE_TREE 0 // do not use tree rotations on 32 bit platforms, barrier bit in NodeRef will cause issues
#endif

namespace embree 
{
  namespace isa
  {
    template<int N>
    struct SetBVHNBounds
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;
      typedef typename BVH::AABBNode AABBNode;

      BVH* bvh;
      __forceinline SetBVHNBounds (BVH* bvh) : bvh(bvh) {}

      __forceinline NodeRecord operator() (NodeRef ref, const NodeRecord* children, size_t num)
      {
        AABBNode* node = ref.getAABBNode();

        BBox3fa res = empty;
        for (size_t i=0; i<num; i++) {
          const BBox3fa b = children[i].bounds;
          res.extend(b);
          node->setRef(i,children[i].ref);
          node->setBounds(i,b);
        }

        BBox3fx result = (BBox3fx&)res;
#if ROTATE_TREE
        if (N == 4)
        {
          size_t n = 0;
          for (size_t i=0; i<num; i++)
            n += children[i].bounds.lower.a;

          if (n >= 4096) {
            for (size_t i=0; i<num; i++) {
              if (children[i].bounds.lower.a < 4096) {
                for (int j=0; j<ROTATE_TREE; j++)
                  BVHNRotate<N>::rotate(node->child(i));
                node->child(i).setBarrier();
              }
            }
          }
          result.lower.a = unsigned(n);
        }
#endif

        return NodeRecord(ref,result);
      }
    };

    template<int N, typename Primitive>
    struct CreateMortonLeaf;

    template<int N>
    struct CreateMortonLeaf<N,Triangle4>
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;

      __forceinline CreateMortonLeaf (TriangleMesh* mesh, unsigned int geomID, BVHBuilderMorton::BuildPrim* morton)
        : mesh(mesh), morton(morton), geomID_(geomID) {}

      __noinline NodeRecord operator() (const range<unsigned>& current, const FastAllocator::CachedAllocator& alloc)
      {
        vfloat4 lower(pos_inf);
        vfloat4 upper(neg_inf);
        size_t items = current.size();
        size_t start = current.begin();
        assert(items<=4);
        
        /* allocate leaf node */
        Triangle4* accel = (Triangle4*) alloc.malloc1(sizeof(Triangle4),BVH::byteAlignment);
        NodeRef ref = BVH::encodeLeaf((char*)accel,1);
        vuint4 vgeomID = -1, vprimID = -1;
        Vec3vf4 v0 = zero, v1 = zero, v2 = zero;
        const TriangleMesh* __restrict__ const mesh = this->mesh;

        for (size_t i=0; i<items; i++)
        {
          const unsigned int primID = morton[start+i].index;
          const TriangleMesh::Triangle& tri = mesh->triangle(primID);
          const Vec3fa& p0 = mesh->vertex(tri.v[0]);
          const Vec3fa& p1 = mesh->vertex(tri.v[1]);
          const Vec3fa& p2 = mesh->vertex(tri.v[2]);
          lower = min(lower,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2);
          upper = max(upper,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2);
          vgeomID [i] = geomID_;
          vprimID [i] = primID;
          v0.x[i] = p0.x; v0.y[i] = p0.y; v0.z[i] = p0.z;
          v1.x[i] = p1.x; v1.y[i] = p1.y; v1.z[i] = p1.z;
          v2.x[i] = p2.x; v2.y[i] = p2.y; v2.z[i] = p2.z;
        }

        Triangle4::store_nt(accel,Triangle4(v0,v1,v2,vgeomID,vprimID));
        BBox3fx box_o = BBox3fx((Vec3fx)lower,(Vec3fx)upper);
#if ROTATE_TREE
        if (N == 4)
          box_o.lower.a = unsigned(current.size());
#endif
        return NodeRecord(ref,box_o);
      }
    
    private:
      TriangleMesh* mesh;
      BVHBuilderMorton::BuildPrim* morton;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
    };
    
    template<int N>
    struct CreateMortonLeaf<N,Triangle4v>
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;

      __forceinline CreateMortonLeaf (TriangleMesh* mesh, unsigned int geomID, BVHBuilderMorton::BuildPrim* morton)
        : mesh(mesh), morton(morton), geomID_(geomID) {}
      
      __noinline NodeRecord operator() (const range<unsigned>& current, const FastAllocator::CachedAllocator& alloc)
      {
        vfloat4 lower(pos_inf);
        vfloat4 upper(neg_inf);
        size_t items = current.size();
        size_t start = current.begin();
        assert(items<=4);
        
        /* allocate leaf node */
        Triangle4v* accel = (Triangle4v*) alloc.malloc1(sizeof(Triangle4v),BVH::byteAlignment);
        NodeRef ref = BVH::encodeLeaf((char*)accel,1);       
        vuint4 vgeomID = -1, vprimID = -1;
        Vec3vf4 v0 = zero, v1 = zero, v2 = zero;
        const TriangleMesh* __restrict__ mesh = this->mesh;

        for (size_t i=0; i<items; i++)
        {
          const unsigned int primID = morton[start+i].index;
          const TriangleMesh::Triangle& tri = mesh->triangle(primID);
          const Vec3fa& p0 = mesh->vertex(tri.v[0]);
          const Vec3fa& p1 = mesh->vertex(tri.v[1]);
          const Vec3fa& p2 = mesh->vertex(tri.v[2]);
          lower = min(lower,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2);
          upper = max(upper,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2);
          vgeomID [i] = geomID_;
          vprimID [i] = primID;
          v0.x[i] = p0.x; v0.y[i] = p0.y; v0.z[i] = p0.z;
          v1.x[i] = p1.x; v1.y[i] = p1.y; v1.z[i] = p1.z;
          v2.x[i] = p2.x; v2.y[i] = p2.y; v2.z[i] = p2.z;
        }
        Triangle4v::store_nt(accel,Triangle4v(v0,v1,v2,vgeomID,vprimID));
        BBox3fx box_o = BBox3fx((Vec3fx)lower,(Vec3fx)upper);
#if ROTATE_TREE
        if (N == 4)
          box_o.lower.a = current.size();
#endif
        return NodeRecord(ref,box_o);
      }
    private:
      TriangleMesh* mesh;
      BVHBuilderMorton::BuildPrim* morton;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
    };

    template<int N>
    struct CreateMortonLeaf<N,Triangle4i>
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;

      __forceinline CreateMortonLeaf (TriangleMesh* mesh, unsigned int geomID, BVHBuilderMorton::BuildPrim* morton)
        : mesh(mesh), morton(morton), geomID_(geomID) {}
      
      __noinline NodeRecord operator() (const range<unsigned>& current, const FastAllocator::CachedAllocator& alloc)
      {
        vfloat4 lower(pos_inf);
        vfloat4 upper(neg_inf);
        size_t items = current.size();
        size_t start = current.begin();
        assert(items<=4);
        
        /* allocate leaf node */
        Triangle4i* accel = (Triangle4i*) alloc.malloc1(sizeof(Triangle4i),BVH::byteAlignment);
        NodeRef ref = BVH::encodeLeaf((char*)accel,1);
        
        vuint4 v0 = zero, v1 = zero, v2 = zero;
        vuint4 vgeomID = -1, vprimID = -1;
        const TriangleMesh* __restrict__ const mesh = this->mesh;
        
        for (size_t i=0; i<items; i++)
        {
          const unsigned int primID = morton[start+i].index;
          const TriangleMesh::Triangle& tri = mesh->triangle(primID);
          const Vec3fa& p0 = mesh->vertex(tri.v[0]);
          const Vec3fa& p1 = mesh->vertex(tri.v[1]);
          const Vec3fa& p2 = mesh->vertex(tri.v[2]);
          lower = min(lower,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2);
          upper = max(upper,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2);
          vgeomID[i] = geomID_;
          vprimID[i] = primID;
          unsigned int int_stride = mesh->vertices0.getStride()/4;
          v0[i] = tri.v[0] * int_stride; 
          v1[i] = tri.v[1] * int_stride;
          v2[i] = tri.v[2] * int_stride;
        }
        
        for (size_t i=items; i<4; i++)
        {
          vgeomID[i] = vgeomID[0];
          vprimID[i] = -1;
          v0[i] = 0;
          v1[i] = 0; 
          v2[i] = 0;
        }
        Triangle4i::store_nt(accel,Triangle4i(v0,v1,v2,vgeomID,vprimID));
        BBox3fx box_o = BBox3fx((Vec3fx)lower,(Vec3fx)upper);
#if ROTATE_TREE
        if (N == 4)
          box_o.lower.a = current.size();
#endif
        return NodeRecord(ref,box_o);
      }
    private:
      TriangleMesh* mesh;
      BVHBuilderMorton::BuildPrim* morton;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
    };

    template<int N>
    struct CreateMortonLeaf<N,Quad4v>
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;

      __forceinline CreateMortonLeaf (QuadMesh* mesh, unsigned int geomID, BVHBuilderMorton::BuildPrim* morton)
        : mesh(mesh), morton(morton), geomID_(geomID) {}
      
      __noinline NodeRecord operator() (const range<unsigned>& current, const FastAllocator::CachedAllocator& alloc)
      {
        vfloat4 lower(pos_inf);
        vfloat4 upper(neg_inf);
        size_t items = current.size();
        size_t start = current.begin();
        assert(items<=4);
        
        /* allocate leaf node */
        Quad4v* accel = (Quad4v*) alloc.malloc1(sizeof(Quad4v),BVH::byteAlignment);
        NodeRef ref = BVH::encodeLeaf((char*)accel,1);
        
        vuint4 vgeomID = -1, vprimID = -1;
        Vec3vf4 v0 = zero, v1 = zero, v2 = zero, v3 = zero;
        const QuadMesh* __restrict__ mesh = this->mesh;

        for (size_t i=0; i<items; i++)
        {
          const unsigned int primID = morton[start+i].index;
          const QuadMesh::Quad& tri = mesh->quad(primID);
          const Vec3fa& p0 = mesh->vertex(tri.v[0]);
          const Vec3fa& p1 = mesh->vertex(tri.v[1]);
          const Vec3fa& p2 = mesh->vertex(tri.v[2]);
          const Vec3fa& p3 = mesh->vertex(tri.v[3]);
          lower = min(lower,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2,(vfloat4)p3);
          upper = max(upper,(vfloat4)p0,(vfloat4)p1,(vfloat4)p2,(vfloat4)p3);
          vgeomID [i] = geomID_;
          vprimID [i] = primID;
          v0.x[i] = p0.x; v0.y[i] = p0.y; v0.z[i] = p0.z;
          v1.x[i] = p1.x; v1.y[i] = p1.y; v1.z[i] = p1.z;
          v2.x[i] = p2.x; v2.y[i] = p2.y; v2.z[i] = p2.z;
          v3.x[i] = p3.x; v3.y[i] = p3.y; v3.z[i] = p3.z;
        }
        Quad4v::store_nt(accel,Quad4v(v0,v1,v2,v3,vgeomID,vprimID));
        BBox3fx box_o = BBox3fx((Vec3fx)lower,(Vec3fx)upper);
#if ROTATE_TREE
        if (N == 4)
          box_o.lower.a = current.size();
#endif
        return NodeRecord(ref,box_o);
      }
    private:
      QuadMesh* mesh;
      BVHBuilderMorton::BuildPrim* morton;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
    };

    template<int N>
    struct CreateMortonLeaf<N,Object>
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;

      __forceinline CreateMortonLeaf (UserGeometry* mesh, unsigned int geomID, BVHBuilderMorton::BuildPrim* morton)
        : mesh(mesh), morton(morton), geomID_(geomID) {}
      
      __noinline NodeRecord operator() (const range<unsigned>& current, const FastAllocator::CachedAllocator& alloc)
      {
        vfloat4 lower(pos_inf);
        vfloat4 upper(neg_inf);
        size_t items = current.size();
        size_t start = current.begin();
        
        /* allocate leaf node */
        Object* accel = (Object*) alloc.malloc1(items*sizeof(Object),BVH::byteAlignment);
        NodeRef ref = BVH::encodeLeaf((char*)accel,items);
        const UserGeometry* mesh = this->mesh;
        
        BBox3fa bounds = empty;
        for (size_t i=0; i<items; i++)
        {
          const unsigned int index = morton[start+i].index;
          const unsigned int primID = index; 
          bounds.extend(mesh->bounds(primID));
          new (&accel[i]) Object(geomID_,primID);
        }

        BBox3fx box_o = (BBox3fx&)bounds;
#if ROTATE_TREE
        if (N == 4)
          box_o.lower.a = current.size();
#endif
        return NodeRecord(ref,box_o);
      }
    private:
      UserGeometry* mesh;
      BVHBuilderMorton::BuildPrim* morton;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
    };

    template<int N>
    struct CreateMortonLeaf<N,InstancePrimitive>
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;

      __forceinline CreateMortonLeaf (Instance* mesh, unsigned int geomID, BVHBuilderMorton::BuildPrim* morton)
        : mesh(mesh), morton(morton), geomID_(geomID) {}
      
      __noinline NodeRecord operator() (const range<unsigned>& current, const FastAllocator::CachedAllocator& alloc)
      {
        vfloat4 lower(pos_inf);
        vfloat4 upper(neg_inf);
        size_t items = current.size();
        size_t start = current.begin();
        assert(items <= 1);
        
        /* allocate leaf node */
        InstancePrimitive* accel = (InstancePrimitive*) alloc.malloc1(items*sizeof(InstancePrimitive),BVH::byteAlignment);
        NodeRef ref = BVH::encodeLeaf((char*)accel,items);
        const Instance* instance = this->mesh;
        
        BBox3fa bounds = empty;
        for (size_t i=0; i<items; i++)
        {
          const unsigned int primID = morton[start+i].index; 
          bounds.extend(instance->bounds(primID));
          new (&accel[i]) InstancePrimitive(instance, geomID_);
        }

        BBox3fx box_o = (BBox3fx&)bounds;
#if ROTATE_TREE
        if (N == 4)
          box_o.lower.a = current.size();
#endif
        return NodeRecord(ref,box_o);
      }
    private:
      Instance* mesh;
      BVHBuilderMorton::BuildPrim* morton;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
    };

    template<typename Mesh>
    struct CalculateMeshBounds
    {
      __forceinline CalculateMeshBounds (Mesh* mesh)
        : mesh(mesh) {}
      
      __forceinline const BBox3fa operator() (const BVHBuilderMorton::BuildPrim& morton) {
        return mesh->bounds(morton.index);
      }
      
    private:
      Mesh* mesh;
    };        

    template<int N, typename Mesh, typename Primitive>
    class BVHNMeshBuilderMorton : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::AABBNode AABBNode;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecord NodeRecord;

    public:
      
      BVHNMeshBuilderMorton (BVH* bvh, Mesh* mesh, unsigned int geomID, const size_t minLeafSize, const size_t maxLeafSize, const size_t singleThreadThreshold = DEFAULT_SINGLE_THREAD_THRESHOLD)
        : bvh(bvh), mesh(mesh), morton(bvh->device,0), settings(N,BVH::maxBuildDepth,minLeafSize,min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks),singleThreadThreshold), geomID_(geomID) {}
      
      /* build function */
      void build() 
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh->numPrimitives != numPreviousPrimitives) {
          bvh->alloc.clear();
          morton.clear();
        }
        size_t numPrimitives = mesh->size();
        numPreviousPrimitives = numPrimitives;
        
        /* skip build for empty scene */
        if (numPrimitives == 0) {
          bvh->set(BVH::emptyNode,empty,0);
          return;
        }
        
        /* preallocate arrays */
        morton.resize(numPrimitives);
        size_t bytesEstimated = numPrimitives*sizeof(AABBNode)/(4*N) + size_t(1.2f*Primitive::blocks(numPrimitives)*sizeof(Primitive));
        size_t bytesMortonCodes = numPrimitives*sizeof(BVHBuilderMorton::BuildPrim);
        bytesEstimated = max(bytesEstimated,bytesMortonCodes); // the first allocation block is reused to sort the morton codes
        bvh->alloc.init(bytesMortonCodes,bytesMortonCodes,bytesEstimated);

        /* create morton code array */
        BVHBuilderMorton::BuildPrim* dest = (BVHBuilderMorton::BuildPrim*) bvh->alloc.specialAlloc(bytesMortonCodes);
        size_t numPrimitivesGen = createMortonCodeArray<Mesh>(mesh,morton,bvh->scene->progressInterface);

        /* create BVH */
        SetBVHNBounds<N> setBounds(bvh);
        CreateMortonLeaf<N,Primitive> createLeaf(mesh,geomID_,morton.data());
        CalculateMeshBounds<Mesh> calculateBounds(mesh);
        auto root = BVHBuilderMorton::build<NodeRecord>(
          typename BVH::CreateAlloc(bvh), 
          typename BVH::AABBNode::Create(),
          setBounds,createLeaf,calculateBounds,bvh->scene->progressInterface,
          morton.data(),dest,numPrimitivesGen,settings);
        
        bvh->set(root.ref,LBBox3fa(root.bounds),numPrimitives);
        
#if ROTATE_TREE
        if (N == 4)
        {
          for (int i=0; i<ROTATE_TREE; i++)
            BVHNRotate<N>::rotate(bvh->root);
          bvh->clearBarrier(bvh->root);
        }
#endif

        /* clear temporary data for static geometry */
        if (bvh->scene->isStaticAccel()) {
          morton.clear();
        }
        bvh->cleanup();
      }
      
      void clear() {
        morton.clear();
      }
      
    private:
      BVH* bvh;
      Mesh* mesh;
      mvector<BVHBuilderMorton::BuildPrim> morton;
      BVHBuilderMorton::Settings settings;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
      unsigned int numPreviousPrimitives = 0;
    };

#if defined(EMBREE_GEOMETRY_TRIANGLE)
    Builder* BVH4Triangle4MeshBuilderMortonGeneral  (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<4,TriangleMesh,Triangle4> ((BVH4*)bvh,mesh,geomID,4,4); }
    Builder* BVH4Triangle4vMeshBuilderMortonGeneral (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<4,TriangleMesh,Triangle4v>((BVH4*)bvh,mesh,geomID,4,4); }
    Builder* BVH4Triangle4iMeshBuilderMortonGeneral (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<4,TriangleMesh,Triangle4i>((BVH4*)bvh,mesh,geomID,4,4); }
#if defined(__AVX__)
    Builder* BVH8Triangle4MeshBuilderMortonGeneral  (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<8,TriangleMesh,Triangle4> ((BVH8*)bvh,mesh,geomID,4,4); }
    Builder* BVH8Triangle4vMeshBuilderMortonGeneral (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<8,TriangleMesh,Triangle4v>((BVH8*)bvh,mesh,geomID,4,4); }
    Builder* BVH8Triangle4iMeshBuilderMortonGeneral (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<8,TriangleMesh,Triangle4i>((BVH8*)bvh,mesh,geomID,4,4); }
#endif
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
    Builder* BVH4Quad4vMeshBuilderMortonGeneral (void* bvh, QuadMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<4,QuadMesh,Quad4v>((BVH4*)bvh,mesh,geomID,4,4); }
#if defined(__AVX__)
    Builder* BVH8Quad4vMeshBuilderMortonGeneral (void* bvh, QuadMesh* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<8,QuadMesh,Quad4v>((BVH8*)bvh,mesh,geomID,4,4); }
#endif
#endif

#if defined(EMBREE_GEOMETRY_USER)
    Builder* BVH4VirtualMeshBuilderMortonGeneral (void* bvh, UserGeometry* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<4,UserGeometry,Object>((BVH4*)bvh,mesh,geomID,1,BVH4::maxLeafBlocks); }
#if defined(__AVX__)
    Builder* BVH8VirtualMeshBuilderMortonGeneral (void* bvh, UserGeometry* mesh, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<8,UserGeometry,Object>((BVH8*)bvh,mesh,geomID,1,BVH4::maxLeafBlocks); }    
#endif
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE)
    Builder* BVH4InstanceMeshBuilderMortonGeneral (void* bvh, Instance* mesh, Geometry::GTypeMask gtype, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<4,Instance,InstancePrimitive>((BVH4*)bvh,mesh,gtype,geomID,1,BVH4::maxLeafBlocks); }
#if defined(__AVX__)
    Builder* BVH8InstanceMeshBuilderMortonGeneral (void* bvh, Instance* mesh, Geometry::GTypeMask gtype, unsigned int geomID, size_t mode) { return new class BVHNMeshBuilderMorton<8,Instance,InstancePrimitive>((BVH8*)bvh,mesh,gtype,geomID,1,BVH4::maxLeafBlocks); }    
#endif
#endif

  }
}
