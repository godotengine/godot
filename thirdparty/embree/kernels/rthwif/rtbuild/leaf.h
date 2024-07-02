// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ZE_RAYTRACING)
#include "sys/sysinfo.h"
#include "sys/vector.h"
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/bbox.h"
#include "math/affinespace.h"
#else
#include "../../../common/sys/sysinfo.h"
#include "../../../common/sys/vector.h"
#include "../../../common/math/vec2.h"
#include "../../../common/math/vec3.h"
#include "../../../common/math/bbox.h"
#include "../../../common/math/lbbox.h"
#include "../../../common/math/affinespace.h"
#endif

#include "node_type.h"

#include <map>
#include <bitset>

namespace embree
{
  /*

    Internal representation for GeometryFlags.

  */
  
#undef OPAQUE     // Windows defines OPAQUE in gdi.h
  enum class GeometryFlags : uint32_t
  {
    NONE = 0x0,
    OPAQUE = 0x1
  };

  inline bool operator& (GeometryFlags a, GeometryFlags b) {
    return (int(a) & int(b)) ? true : false;
  }

  /* output operator for GeometryFlags */
  inline std::ostream& operator<<(std::ostream& cout, const GeometryFlags& gflags)
  {
#if !defined(__SYCL_DEVICE_ONLY__)
    if (gflags == GeometryFlags::NONE) return cout << "NONE";
    if (gflags & GeometryFlags::OPAQUE) cout << "OPAQUE ";
#endif
    return cout;
  }

  /*

    This structure is a header for each leaf type. Only the
    InstanceLeaf has a slightly different header.

    All primitives inside a leaf are of the same geometry, thus have
    the same geometry index (geomIndex), the same shader index
    (shaderIndex), the same geometry mask (geomMask), and the same
    geometry flags (geomFlags).

    The shaderIndex is used to calculate the shader record to
    invoke. This is an extension to DXR where the geomIndex is used
    for that purpose. For DXR we can always set the shaderIndex to be
    equal to the geomIndex.

   */
  
  struct PrimLeafDesc 
  {
    static const uint32_t MAX_GEOM_INDEX = 0x3FFFFFFF;
    static const uint32_t MAX_SHADER_INDEX = 0xFFFFFF;
    
    enum Type : uint32_t
    {
      TYPE_NONE = 0,

      /* For a node type of NODE_TYPE_PROCEDURAL we support enabling
       * and disabling the opaque/non_opaque culling. */
        
      TYPE_OPACITY_CULLING_ENABLED = 0,
      TYPE_OPACITY_CULLING_DISABLED = 1
    };
    
    PrimLeafDesc() {}
    
    PrimLeafDesc(uint32_t shaderIndex, uint32_t geomIndex, GeometryFlags gflags, uint32_t geomMask, Type type = TYPE_NONE)
    : shaderIndex(shaderIndex), geomMask(geomMask), geomIndex(geomIndex), type(type), geomFlags((uint32_t)gflags)
    {
      if (shaderIndex > MAX_SHADER_INDEX)
        throw std::runtime_error("too large shader ID");
      
      if (geomIndex > MAX_GEOM_INDEX)
        throw std::runtime_error("too large geometry ID");
    }

    /* compares two PrimLeafDesc's for equality */
    friend bool operator ==(const PrimLeafDesc& a, const PrimLeafDesc& b)
    {
      if (a.geomIndex != b.geomIndex) return false;
      assert(a.shaderIndex == b.shaderIndex);
      assert(a.geomMask == b.geomMask);
      assert(a.type == b.type);
      assert(a.geomFlags == b.geomFlags);
      return true;
    }

    friend bool operator !=(const PrimLeafDesc& a, const PrimLeafDesc& b) {
      return !(a == b);
    }

    void print(std::ostream& cout, uint32_t depth) const
    {
#if !defined(__SYCL_DEVICE_ONLY__)
      cout << tab(depth) << "PrimLeafDesc {" << std::endl;
      cout << tab(depth) << "  shaderIndex = " << shaderIndex << std::endl;
      cout << tab(depth) << "  geomMask = " << std::bitset<8>(geomMask) << std::endl;
      cout << tab(depth) << "  geomFlags = " << getGeomFlags() << std::endl;
      cout << tab(depth) << "  geomIndex = " << geomIndex << std::endl;
      cout << tab(depth) << "}";
#endif
    }

    friend inline std::ostream& operator<<(std::ostream& cout, const PrimLeafDesc& desc) {
      desc.print(cout,0); return cout;
    }

    /* Checks if opaque culling is enabled. */
    bool opaqueCullingEnabled() const {
      return type == TYPE_OPACITY_CULLING_ENABLED;
    }

    /* procedural instances store some valid shader index */
    bool isProceduralInstance() const {
      return shaderIndex != 0xFFFFFF;
    }

    /* returns geometry flags */
    GeometryFlags getGeomFlags() const {
      return (GeometryFlags) geomFlags;
    }

  public:
    uint32_t shaderIndex : 24;    // shader index used for shader record calculations
    uint32_t geomMask    : 8;     // geometry mask used for ray masking
 
    uint32_t geomIndex      : 29; // the geometry index specifies the n'th geometry of the scene
    /*Type*/ uint32_t type  : 1;  // enable/disable culling for procedurals and instances
    /*GeometryFlags*/ uint32_t geomFlags : 2;  // geometry flags of this geometry
  };

  /*

    The QuadLeaf structure stores a single quad. A quad is a triangle
    pair with a shared edge. The first triangle has vertices v0,v1,v2,
    while the second triangle has vertices v[j0],v[j1],v[j2], thus the
    second triangle used local triangle indices.

   */
  
  struct QuadLeaf
  {
    QuadLeaf() {}

    QuadLeaf (Vec3f v0, Vec3f v1, Vec3f v2, Vec3f v3,
              uint8_t j0, uint8_t j1, uint8_t j2,
              uint32_t shaderIndex, uint32_t geomIndex, uint32_t primIndex0, uint32_t primIndex1,
              GeometryFlags gflags, uint32_t geomMask, bool last)

      : leafDesc(shaderIndex,geomIndex,gflags,geomMask),
        primIndex0(primIndex0), 
        primIndex1Delta(primIndex1-primIndex0), pad1(0),
        j0(j0),j1(j1),j2(j2),last(last),pad(0),
        v0(v0), v1(v1), v2(v2), v3(v3)
    {
      /* There are some constraints on the primitive indices. The
       * second primitive index always has to be the largest and the
       * distance between them can be at most 0xFFFF as we use 16 bits
       * to encode that difference. */
      assert(primIndex0 <= primIndex1 && primIndex1 - primIndex0 < 0xFFFF);
    }

    /* returns the i'th vertex */
    __forceinline Vec3f vertex(size_t i) const {
      assert(i < 4); return (&v0)[i];
    }

    /* Checks if the specified triange is the last inside a leaf
     * list. */
    bool isLast(uint32_t i = 1) const
    {
      assert(i<2);
      if (i == 0) return false; // the first triangle is never the last
      else return last;         // the last bit tags the second triangle to be last
    }

    /* Checks if the second triangle exists. */
    bool valid2() const {
      return !(j0 == 0 && j1 == 0 && j2 == 0);
    }

    /* Calculates the number of stored triangles. */
    size_t size() const {
      return 1 + valid2();
    }

    /* Calculates the effectively used bytes. If we store only one
     * triangle we waste the storage of one vertex. */
    size_t usedBytes() const
    {
      if (valid2()) return sizeof(QuadLeaf);
      else          return sizeof(QuadLeaf)-sizeof(Vec3f);
    }

    /* Calculates to delta to add to primIndex0 to get the primitive
     * index of the i'th triangle. */
    uint32_t primIndexDelta(uint32_t i) const
    {
      assert(i<2);
      return i*primIndex1Delta;
    }

    /* Calculates the primitive index of the i'th triangle. */
    uint32_t primIndex(uint32_t i) const
    {
      assert(i<2);
      return primIndex0 + primIndexDelta(i);
    }   

    /* Quad mode is a special mode where the uv's over the quad are
     * defined over the entire range [0,1]x[0,1]. */
    bool quadMode() const {
      return primIndex1Delta == 0;
    }

    /* Calculates the bounding box of this leaf. */
    BBox3f bounds() const
    {
      BBox3f b = empty;
      b.extend(v0);
      b.extend(v1);
      b.extend(v2);
      if (valid2())
        b.extend(v3);
      return b;
    }

    /* output of quad leaf */
    void print(std::ostream& cout, uint32_t depth) const
    {
#if !defined(__SYCL_DEVICE_ONLY__)
      cout << tab(depth) << "QuadLeaf {" << std::endl;
      cout << tab(depth) << "  addr = " << this << std::endl;
      cout << tab(depth) << "  shaderIndex = " << leafDesc.shaderIndex << std::endl;
      cout << tab(depth) << "  geomMask = " << std::bitset<8>(leafDesc.geomMask) << std::endl;
      cout << tab(depth) << "  geomFlags = " << leafDesc.getGeomFlags() << std::endl;
      cout << tab(depth) << "  geomIndex = " << leafDesc.geomIndex << std::endl;
      cout << tab(depth) << "  triangle0 = { " << std::endl;
      cout << tab(depth) << "    primIndex = " << primIndex(0) << std::endl;
      cout << tab(depth) << "    v0 = " << v0 << std::endl;
      cout << tab(depth) << "    v1 = " << v1 << std::endl;
      cout << tab(depth) << "    v2 = " << v2 << std::endl;
      cout << tab(depth) << "  }" << std::endl;
      if (valid2()) {
        cout << tab(depth) << "  triangle1 = { " << std::endl;
        cout << tab(depth) << "    primIndex = " << primIndex(1) << std::endl;
        cout << tab(depth) << "    v0 = " << vertex(j0) << std::endl;
        cout << tab(depth) << "    v1 = " << vertex(j1) << std::endl;
        cout << tab(depth) << "    v2 = " << vertex(j2) << std::endl;
        cout << tab(depth) << "  }" << std::endl;
      }
      cout << tab(depth) << "}";
#endif
    }

    /* output operator for QuadLeaf */
    friend inline std::ostream& operator<<(std::ostream& cout, const QuadLeaf& leaf) {
      leaf.print(cout,0); return cout;
    }

  public:
    PrimLeafDesc leafDesc;  // the leaf header

    uint32_t primIndex0;    // primitive index of first triangle
    struct {
      uint32_t primIndex1Delta : 5;  // delta encoded primitive index of second triangle
      uint32_t pad1            : 11; // MBZ
      uint32_t j0              : 2;   // specifies first vertex of second triangle
      uint32_t j1              : 2;   // specified second vertex of second triangle
      uint32_t j2              : 2;   // specified third vertex of second triangle    
      uint32_t last            : 1;   // true if the second triangle is the last triangle in a leaf list
      uint32_t pad             : 9;   // unused bits
    };
    
    Vec3f v0;  // first vertex of first triangle
    Vec3f v1;  // second vertex of first triangle
    Vec3f v2;  // third vertex of first triangle
    Vec3f v3;  // forth vertex used for second triangle
  };

  static_assert(sizeof(QuadLeaf) == 64, "QuadLeaf must be 64 bytes large");

  /* 

     Internal instance flags definition.

  */
  
  struct InstanceFlags
  {
    enum Flags : uint8_t
    {
      NONE = 0x0,
      TRIANGLE_CULL_DISABLE = 0x1,              // disables culling of front and back facing triangles through ray flags
      TRIANGLE_FRONT_COUNTERCLOCKWISE = 0x2,    // for mirroring transformations the instance can switch front and backface of triangles
      FORCE_OPAQUE = 0x4,                       // forces all primitives inside this instance to be opaque
      FORCE_NON_OPAQUE = 0x8                    // forces all primitives inside this instane to be non-opaque
    };

    InstanceFlags() {}

    InstanceFlags(Flags rflags)
      : flags(rflags) {}

    InstanceFlags(uint8_t rflags)
      : flags((Flags)rflags) {}

    operator Flags () const {
      return flags;
    }

    /* output operator for InstanceFlags */
    friend inline std::ostream& operator<<(std::ostream& cout, const InstanceFlags& iflags)
    {
#if !defined(__SYCL_DEVICE_ONLY__)
      if (iflags == InstanceFlags::NONE) return cout << "NONE";
      if (iflags.triangle_cull_disable) cout << "TRIANGLE_CULL_DISABLE ";
      if (iflags.triangle_front_counterclockwise) cout << "TRIANGLE_FRONT_COUNTERCLOCKWISE ";
      if (iflags.force_opaque) cout << "FORCE_OPAQUE ";
      if (iflags.force_non_opaque) cout << "FORCE_NON_OPAQUE ";
#endif
      return cout;
    }

  public:
    union
    {
      Flags flags;
      struct
      {
        bool triangle_cull_disable : 1;
        bool triangle_front_counterclockwise : 1;
        bool force_opaque : 1;
        bool force_non_opaque : 1;
      };
    };
  };

  inline InstanceFlags::Flags operator| (InstanceFlags::Flags a,InstanceFlags::Flags b) {
    return (InstanceFlags::Flags)(int(a) | int(b));
  }
  
  /* 

     The instance leaf represent an instance. It essentially stores
     transformation matrices (local to world as well as world to
     local) of the instance as well as a pointer to the start node
     of some BVH.

     The instance leaf consists of two parts, part0 (first 64 bytes)
     and part1 (second 64 bytes). Part0 will only get accessed by
     hardware and stores the world to local transformation as well as
     the BVH node to start traversal. Part1 stores additional data
     that is only read by the shader, e.g. it stores the local to
     world transformation of the instance.

     The layout of the first part of the InstanceLeaf is compatible
     with a ProceduralLeaf, thus we can use the same layout for
     software instancing if we want.

  */
  
  struct InstanceLeaf
  {
    InstanceLeaf() {}
    
    InstanceLeaf (AffineSpace3f obj2world, uint64_t startNodePtr, uint32_t instID, uint32_t instUserID, uint8_t instMask)
    {
      part0.shaderIndex = 0; //InstShaderRecordID;
      part0.geomMask = instMask;
      
      part0.instanceContributionToHitGroupIndex = 0; //desc.InstanceContributionToHitGroupIndex;
      part0.pad0 = 0;
      part0.type = PrimLeafDesc::TYPE_OPACITY_CULLING_ENABLED;
      part0.geomFlags = (uint32_t) GeometryFlags::NONE;
    
      part0.startNodePtr = startNodePtr;
      assert((startNodePtr >> 48) == 0);
      part0.instFlags = (InstanceFlags) 0;
      part0.pad1 = 0;
      
      part1.instanceID = instUserID;
      part1.instanceIndex = instID;
      part1.bvhPtr = (uint64_t) 0;
      part1.pad = 0;

      part1.obj2world_vx = obj2world.l.vx;
      part1.obj2world_vy = obj2world.l.vy;
      part1.obj2world_vz = obj2world.l.vz;
      part0.obj2world_p = obj2world.p;
      
      const AffineSpace3f world2obj = rcp(obj2world);
      part0.world2obj_vx = world2obj.l.vx;
      part0.world2obj_vy = world2obj.l.vy;
      part0.world2obj_vz = world2obj.l.vz;
      part1.world2obj_p = world2obj.p;
    }

    /* Returns the address of the start node pointer. We need this
     * address to calculate relocation tables when dumping the BVH to
     * disk. */
    const uint64_t startNodePtrAddr() const {
      return (uint64_t)((char*)&part0 + 8);
    }

    /* Returns the address of the BVH that contains the start node. */
    const uint64_t bvhPtrAddr() const {
      return (uint64_t)&part1;
    }

    /* returns the world to object space transformation matrix. */
    const AffineSpace3f World2Obj() const {
      return AffineSpace3f(part0.world2obj_vx,part0.world2obj_vy,part0.world2obj_vz,part1.world2obj_p);
    }

    /* returns the object to world space transformation matrix. */
    const AffineSpace3f Obj2World() const {
      return AffineSpace3f(part1.obj2world_vx,part1.obj2world_vy,part1.obj2world_vz,part0.obj2world_p);
    }

    /* output operator for instance leaf */
    void print (std::ostream& cout, uint32_t depth) const
    {
#if !defined(__SYCL_DEVICE_ONLY__)
      if (!part0.type) cout << tab(depth) << "InstanceLeaf {" << std::endl;
      else             cout << tab(depth) << "ProceduralInstanceLeaf {" << std::endl;
        
      cout << tab(depth) << "  addr = " << this << std::endl;
      cout << tab(depth) << "  shaderIndex = " << part0.shaderIndex << std::endl;
      cout << tab(depth) << "  geomMask = " << std::bitset<8>(part0.geomMask) << std::endl;
      cout << tab(depth) << "  geomIndex = " << part1.instanceIndex << std::endl;
      cout << tab(depth) << "  instanceID = " << part1.instanceID << std::endl;
      cout << tab(depth) << "  instFlags = " << InstanceFlags(part0.instFlags) << std::endl;
      cout << tab(depth) << "  startNodePtr = " << (void*)(size_t)part0.startNodePtr << std::endl;
      cout << tab(depth) << "  obj2world.vx = " << part1.obj2world_vx << std::endl;
      cout << tab(depth) << "  obj2world.vy = " << part1.obj2world_vy << std::endl;
      cout << tab(depth) << "  obj2world.vz = " << part1.obj2world_vz << std::endl;
      cout << tab(depth) << "  obj2world.p = " << part0.obj2world_p << std::endl;
      cout << tab(depth) << "  world2obj.vx = " << part0.world2obj_vx << std::endl;
      cout << tab(depth) << "  world2obj.vy = " << part0.world2obj_vy << std::endl;
      cout << tab(depth) << "  world2obj.vz = " << part0.world2obj_vz << std::endl;
      cout << tab(depth) << "  world2obj.p = " << part1.world2obj_p << std::endl;
      cout << tab(depth) << "  instanceContributionToHitGroupIndex = " << part0.instanceContributionToHitGroupIndex << std::endl;
      cout << tab(depth) << "}";
#endif
    }

    /* output operator for InstanceLeaf */
    friend inline std::ostream& operator<<(std::ostream& cout, const InstanceLeaf& leaf) {
      leaf.print(cout,0); return cout;
    }

    /* first 64 bytes accessed during traversal by hardware */
    struct Part0
    {
      /* Checks if opaque culling is enabled. */
      bool opaqueCullingEnabled() const {
        return type == PrimLeafDesc::TYPE_OPACITY_CULLING_ENABLED;
      }

    public:
      uint32_t shaderIndex : 24;  // shader index used to calculate instancing shader in case of software instancing
      uint32_t geomMask : 8;      // geometry mask used for ray masking
      
      uint32_t instanceContributionToHitGroupIndex : 24;
      uint32_t pad0 : 5;

      /* the following two entries are only used for procedural instances */
      /*PrimLeafDesc::Type*/ uint32_t type : 1; // enables/disables opaque culling
      /*GeometryFlags*/ uint32_t geomFlags : 2; // unused for instances
      
      uint64_t startNodePtr : 48;  // start node where to continue traversal of the instanced object
      uint64_t instFlags : 8;      // flags for the instance (see InstanceFlags)
      uint64_t pad1 : 8;           // unused bits
      
      Vec3f world2obj_vx;   // 1st column of Worl2Obj transform
      Vec3f world2obj_vy;   // 2nd column of Worl2Obj transform
      Vec3f world2obj_vz;   // 3rd column of Worl2Obj transform
      Vec3f obj2world_p;    // translation of Obj2World transform (on purpose in first 64 bytes)
    } part0;

    /* second 64 bytes accessed during shading */
    struct Part1
    {
      uint64_t bvhPtr : 48;   // pointer to BVH where start node belongs too
      uint64_t pad : 16;      // unused bits
      
      uint32_t instanceID;    // user defined value per DXR spec
      uint32_t instanceIndex; // geometry index of the instance (n'th geometry in scene)
      
      Vec3f obj2world_vx;   // 1st column of Obj2World transform
      Vec3f obj2world_vy;   // 2nd column of Obj2World transform
      Vec3f obj2world_vz;   // 3rd column of Obj2World transform
      Vec3f world2obj_p;    // translation of World2Obj transform
    } part1;
  };

  static_assert(sizeof(InstanceLeaf) == 128, "InstanceLeaf must be 128 bytes large");


  /*
    Leaf type for procedural geometry. This leaf only contains the
    leaf header (which identifices the geometry) and a list of
    primitive indices.

    The BVH will typically reference only some of the primitives
    stores inside this leaf. The range is specified by a start
    primitive and the last primitive is tagged with a bit.

   */
  
  struct ProceduralLeaf
  {
    static const uint32_t N = 13;

    /* Creates an empty procedural leaf. */
    ProceduralLeaf ()
      : leafDesc(PrimLeafDesc::MAX_SHADER_INDEX,PrimLeafDesc::MAX_GEOM_INDEX,GeometryFlags::NONE,0), numPrimitives(0), pad(0), last(0)
    {
      for (auto& id : _primIndex) id = 0xFFFFFFFF;
    }

    /* Creates a procedural leaf with one primitive. More primitives
     * of the same geometry can get added later using the add
     * function. */
    
    ProceduralLeaf (PrimLeafDesc leafDesc, uint32_t primIndex, bool last)
    : leafDesc(leafDesc), numPrimitives(1), pad(0), last(last ? 0xFFFFFFFF : 0xFFFFFFFE)
    {
      for (auto& id : _primIndex) id = 0xFFFFFFFF;
      _primIndex[0] = primIndex;
    }

    /* returns the number of primitives stored inside this leaf */
    uint32_t size() const  {
      return numPrimitives;
    }

    /* Calculates the effectively used bytes. */
    size_t usedBytes() const
    {
      /*if (leafDesc.isProceduralInstance())
        return sizeof(InstanceLeaf);
      else*/
        return sizeof(PrimLeafDesc)+4+4*numPrimitives;
    }
    
    /* if possible adds a new primitive to this leaf */
    bool add(PrimLeafDesc leafDesc_in, uint32_t primIndex_in, bool last_in)
    {
      assert(primIndex_in != 0xFFFFFFFF);
      if (numPrimitives >= N) return false;
      if (!numPrimitives) leafDesc = leafDesc_in;
      if (leafDesc != leafDesc_in) return false;
      _primIndex[numPrimitives] = primIndex_in;
      if (last_in) last |=   1 << numPrimitives;
      else         last &= ~(1 << numPrimitives);
      numPrimitives++;
      return true;
    }

    /* returns the primitive index of the i'th primitive */
    uint32_t primIndex(uint32_t i) const
    {
      assert(i < N);
      return _primIndex[i];
    }

    /* checks if the i'th primitive is the last in a leaf list */
    bool isLast(uint32_t i) const {
      if (i >= N) return true; // just to make some verify tests happy
      else return (last >> i) & 1;
    }

    /* output operator for procedural leaf */
    void print (std::ostream& cout, uint32_t i, uint32_t depth) const
    {
#if !defined(__SYCL_DEVICE_ONLY__)
      cout << tab(depth) << "ProceduralLeaf {" << std::endl;
      cout << tab(depth) << "  addr = " << this << std::endl;
      cout << tab(depth) << "  slot = " << i << std::endl;
      if (i < N) {
        cout << tab(depth) << "  shaderIndex = " << leafDesc.shaderIndex << std::endl;
        cout << tab(depth) << "  geomMask = " << std::bitset<8>(leafDesc.geomMask) << std::endl;
        cout << tab(depth) << "  geomFlags = " << leafDesc.getGeomFlags() << std::endl;
        cout << tab(depth) << "  geomIndex = " << leafDesc.geomIndex << std::endl;
        cout << tab(depth) << "  primIndex = " << primIndex(i) << std::endl;
      } else {
        cout << tab(depth) << " INVALID" << std::endl;
      }
      cout << tab(depth) << "}";
#endif
    }

  public:
    PrimLeafDesc leafDesc;           // leaf header identifying the geometry
    uint32_t numPrimitives : 4;      // number of stored primitives
    uint32_t pad           : 32-4-N;
    uint32_t last          : N;      // bit vector with a last bit per primitive
    uint32_t _primIndex[N];          // primitive indices of all primitives stored inside the leaf
  };

  static_assert(sizeof(ProceduralLeaf) == 64, "ProceduralLeaf must be 64 bytes large");
}
