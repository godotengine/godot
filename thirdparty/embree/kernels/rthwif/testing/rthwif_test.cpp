// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define NOMINMAX

// prevents "'__thiscall' calling convention is not supported for this target" warning from TBB
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"

#include <CL/sycl.hpp>
#include "tbb/tbb.h"

#if defined(ZE_RAYTRACING)
#include "../rtbuild/sys/sysinfo.h"
#include "../rtbuild/sys/vector.h"
#include "../rtbuild/math/vec2.h"
#include "../rtbuild/math/vec3.h"
#include "../rtbuild/math/bbox.h"
#include "../rtbuild/math/affinespace.h"
#else
#include "../../../common/sys/sysinfo.h"
#include "../../../common/sys/vector.h"
#include "../../../common/math/vec2.h"
#include "../../../common/math/vec3.h"
#include "../../../common/math/bbox.h"
#include "../../../common/math/lbbox.h"
#include "../../../common/math/affinespace.h"
#endif

#define _USE_MATH_DEFINES
#include <math.h>

#include "../rttrace/rttrace.h"

#include <level_zero/ze_wrapper.h>

#include <vector>
#include <map>
#include <iostream>
#include <fstream>

namespace embree {
  double getSeconds();
}

sycl::device device;
sycl::context context;
void* dispatchGlobalsPtr = nullptr;

struct RandomSampler {
  unsigned int s;
};

unsigned int MurmurHash3_mix(unsigned int hash, unsigned int k)
{
  const unsigned int c1 = 0xcc9e2d51;
  const unsigned int c2 = 0x1b873593;
  const unsigned int r1 = 15;
  const unsigned int r2 = 13;
  const unsigned int m = 5;
  const unsigned int n = 0xe6546b64;

  k *= c1;
  k = (k << r1) | (k >> (32 - r1));
  k *= c2;

  hash ^= k;
  hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;

  return hash;
}

unsigned int MurmurHash3_finalize(unsigned int hash)
{
  hash ^= hash >> 16;
  hash *= 0x85ebca6b;
  hash ^= hash >> 13;
  hash *= 0xc2b2ae35;
  hash ^= hash >> 16;
  return hash;
}

unsigned int LCG_next(unsigned int value)
{
  const unsigned int m = 1664525;
  const unsigned int n = 1013904223;
  return value * m + n;
}

void RandomSampler_init(RandomSampler& self, int id)
{
  unsigned int hash = 0;
  hash = MurmurHash3_mix(hash, id);
  hash = MurmurHash3_finalize(hash);
  self.s = hash;
}

int RandomSampler_getInt(RandomSampler& self) {
  self.s = LCG_next(self.s); return self.s >> 1;
}

unsigned int RandomSampler_getUInt(RandomSampler& self) {
  self.s = LCG_next(self.s); return self.s;
}

float RandomSampler_getFloat(RandomSampler& self) {
  return (float)RandomSampler_getInt(self) * 4.656612873077392578125e-10f;
}

sycl::float3 RandomSampler_getFloat3(RandomSampler& self)
{
  const float x = RandomSampler_getFloat(self);
  const float y = RandomSampler_getFloat(self);
  const float z = RandomSampler_getFloat(self);
  return sycl::float3(x,y,z);
}

RandomSampler rng;

ze_rtas_builder_exp_handle_t hBuilder = nullptr;
ze_rtas_parallel_operation_exp_handle_t parallelOperation = nullptr;

enum class InstancingType
{
  NONE,
  SW_INSTANCING,
  HW_INSTANCING
};

enum class TestType
{
  TRIANGLES_COMMITTED_HIT,           // triangles
  TRIANGLES_POTENTIAL_HIT,           // triangles + filter + check potential hit
  TRIANGLES_ANYHIT_SHADER_COMMIT,    // triangles + filter + commit
  TRIANGLES_ANYHIT_SHADER_REJECT,    // triangles + filter + reject
  PROCEDURALS_COMMITTED_HIT,         // procedural triangles
  BUILD_TEST_TRIANGLES,              // test BVH builder with triangles
  BUILD_TEST_PROCEDURALS,            // test BVH builder with procedurals
  BUILD_TEST_INSTANCES,              // test BVH builder with instances
  BUILD_TEST_MIXED,                  // test BVH builder with mixed scene (triangles, procedurals, and instances)
  BENCHMARK_TRIANGLES,               // benchmark BVH builder with triangles
  BENCHMARK_PROCEDURALS,             // benchmark BVH builder with procedurals
};

enum class BuildMode
{
  BUILD_EXPECTED_SIZE,
  BUILD_WORST_CASE_SIZE
};

struct TestInput
{
  sycl::float3 org;
  sycl::float3 dir;
  float tnear;
  float tfar;
  uint32_t mask;
  uint32_t flags;
};

enum TestHitType {
  TEST_COMMITTED_HIT,
  TEST_POTENTIAL_HIT,
  TEST_MISS
};

struct TestOutput
{
  // Ray data at level 0
  sycl::float3 ray0_org;
  sycl::float3 ray0_dir;
  float ray0_tnear;
  uint32_t ray0_mask;
  uint32_t ray0_flags;
  
  // Ray data at hit bvh_level
  sycl::float3 rayN_org;
  sycl::float3 rayN_dir;
  float rayN_tnear;
  uint32_t rayN_mask;
  uint32_t rayN_flags;
  
  // Hit data
  TestHitType hit_type;
  uint32_t bvh_level;
  uint32_t hit_candidate;
  float t;
  float u;
  float v;
  bool front_face;
  uint32_t geomID;
  uint32_t primID;
  uint32_t instID;
  uint32_t instUserID;
  sycl::float3 v0;
  sycl::float3 v1;
  sycl::float3 v2;

  intel_float4x3 world_to_object;
  intel_float4x3 object_to_world;
};

std::ostream& operator<<(std::ostream& out, const intel_float3& v) {
  return out << "(" << v.x << "," << v.y << "," << v.z  << ")";
}

void compareTestOutput(uint32_t tid, uint32_t& errors, const TestOutput& test, const TestOutput& expected)
{
#define COMPARE(member)                 \
  if (test.member != expected.member) { \
    if (errors < 16)                                                    \
      std::cout << "test" << tid << " " #member " mismatch: output " << test.member << " != expected " << expected.member << std::endl; \
    errors++;                                                           \
  }
#define COMPARE1(member,eps)               \
  if (fabs(test.member-expected.member) > eps) {                              \
    if (errors < 16)                                                    \
      std::cout << "test" << tid << " " #member " mismatch: output " << test.member << " != expected " << expected.member << std::endl; \
    errors++;                                                           \
  }
#define COMPARE3(member,eps) {                                          \
    const bool x = fabs(test.member.x()-expected.member.x()) > eps;     \
    const bool y = fabs(test.member.y()-expected.member.y()) > eps;     \
    const bool z = fabs(test.member.z()-expected.member.z()) > eps;     \
    if (x || y || z) {                                                  \
      if (errors < 16)                                                  \
        std::cout << "test" << tid << " " #member " mismatch: output " << test.member << " != expected " << expected.member << std::endl; \
      errors++;                                                         \
    }                                                                   \
  }
#define COMPARE3I(member,eps) {                                          \
    const bool x = test.member.x != expected.member.x;     \
    const bool y = test.member.y != expected.member.y;     \
    const bool z = test.member.z != expected.member.z;     \
    if (x || y || z) {                                                  \
      if (errors < 16)                                                  \
        std::cout << "test" << tid << " " #member " mismatch: output " << test.member << " != expected " << expected.member << std::endl; \
      errors++;                                                         \
    }                                                                   \
  }

  float eps = 2E-4;

  COMPARE3(ray0_org,0);
  COMPARE3(ray0_dir,0);
  COMPARE1(ray0_tnear,0);
  COMPARE(ray0_mask);
  COMPARE(ray0_flags);
  COMPARE3(rayN_org,eps);
  COMPARE3(rayN_dir,eps);
  COMPARE1(rayN_tnear,eps);
  COMPARE(rayN_mask);
  COMPARE(rayN_flags);
  COMPARE(hit_type);
  COMPARE(bvh_level);
  COMPARE(hit_candidate);
  COMPARE1(t,eps);
  COMPARE1(u,eps);
  COMPARE1(v,eps);
  COMPARE(front_face);
  COMPARE(geomID);
  COMPARE(primID);
  COMPARE(instID);
  COMPARE(instUserID);
  COMPARE3(v0,eps);
  COMPARE3(v1,eps);
  COMPARE3(v2,eps);
  COMPARE3I(world_to_object.vx,eps);
  COMPARE3I(world_to_object.vy,eps);
  COMPARE3I(world_to_object.vz,eps);
  COMPARE3I(world_to_object.p ,eps);
  COMPARE3I(object_to_world.vx,eps);
  COMPARE3I(object_to_world.vy,eps);
  COMPARE3I(object_to_world.vz,eps);
  COMPARE3I(object_to_world.p ,eps);
}

struct LinearSpace3f
{
  /*! matrix construction from column vectors */
  LinearSpace3f(const sycl::float3& vx, const sycl::float3& vy, const sycl::float3& vz)
    : vx(vx), vy(vy), vz(vz) {}
  
  /*! matrix construction from row mayor data */
  LinearSpace3f(const float m00, const float m01, const float m02,
                const float m10, const float m11, const float m12,
                const float m20, const float m21, const float m22)
    : vx(m00,m10,m20), vy(m01,m11,m21), vz(m02,m12,m22) {}
  
  /*! compute the determinant of the matrix */
  const float det() const { return sycl::dot(vx,sycl::cross(vy,vz)); }
  
  /*! compute adjoint matrix */
  const LinearSpace3f adjoint() const { return LinearSpace3f(sycl::cross(vy,vz),sycl::cross(vz,vx),sycl::cross(vx,vy)).transposed(); }
  
  /*! compute inverse matrix */
  const LinearSpace3f inverse() const
  {
    const float d = det();
    const LinearSpace3f a = adjoint();
    return { a.vx/d, a.vy/d, a.vz/d };
  }

  /*! compute transposed matrix */
  const LinearSpace3f transposed() const { return LinearSpace3f(vx.x(),vx.y(),vx.z(),vy.x(),vy.y(),vy.z(),vz.x(),vz.y(),vz.z()); }

  /*! return matrix for rotation around arbitrary axis */
  static LinearSpace3f rotate(const sycl::float3 _u, const float r) {
    sycl::float3 u = normalize(_u);
    float s = sinf(r), c = cosf(r);
    return LinearSpace3f(u.x()*u.x()+(1-u.x()*u.x())*c,  u.x()*u.y()*(1-c)-u.z()*s,    u.x()*u.z()*(1-c)+u.y()*s,
                         u.x()*u.y()*(1-c)+u.z()*s,    u.y()*u.y()+(1-u.y()*u.y())*c,  u.y()*u.z()*(1-c)-u.x()*s,
                         u.x()*u.z()*(1-c)-u.y()*s,    u.y()*u.z()*(1-c)+u.x()*s,    u.z()*u.z()+(1-u.z()*u.z())*c);
  }

public:
  sycl::float3 vx,vy,vz;
};

sycl::float3 xfmPoint (const LinearSpace3f& m, const sycl::float3& p) {
  return p.x()*m.vx + (p.y()*m.vy + p.z()*m.vz);
}

struct Transform
{
  Transform ()
    : vx(1,0,0), vy(0,1,0), vz(0,0,1), p(0,0,0) {}

  Transform ( sycl::float3 vx, sycl::float3 vy, sycl::float3 vz, sycl::float3 p )
    : vx(vx), vy(vy), vz(vz), p(p) {}

  Transform ( intel_float4x3 xfm )
    : vx(xfm.vx), vy(xfm.vy), vz(xfm.vz), p(xfm.p) {}

  operator intel_float4x3 () const {
    return { vx, vy, vz, p };
  }

  sycl::float3 vx,vy,vz,p;
};

std::ostream& operator<<(std::ostream& out, const Transform& t) {
  return out << " Transform {" << t.vx << ", " << t.vy << ", " << t.vz << ", " << t.p  << "}";
}

sycl::float3 xfmPoint (const Transform& m, const sycl::float3& p) {
  return p.x()*m.vx + (p.y()*m.vy + (p.z()*m.vz + m.p));
}

sycl::float3 xfmVector (const Transform& m, const sycl::float3& v) {
  return v.x()*m.vx + (v.y()*m.vy + v.z()*m.vz);
}

Transform operator* (const Transform& a, const Transform& b) {
  return Transform(xfmVector(a,b.vx),xfmVector(a,b.vy),xfmVector(a,b.vz),xfmPoint(a,b.p));
}

Transform rcp( const Transform& a )
{
#if 1 // match builder math for rcp to have bit accurate data to compare against
  embree::Vec3f vx(a.vx.x(), a.vx.y(), a.vx.z());
  embree::Vec3f vy(a.vy.x(), a.vy.y(), a.vy.z());
  embree::Vec3f vz(a.vz.x(), a.vz.y(), a.vz.z());
  embree::Vec3f  p(a. p.x(), a. p.y(), a. p.z());
  embree::AffineSpace3f l(embree::LinearSpace3f(vx,vy,vz),p);
  embree::AffineSpace3f il = rcp(l);
  sycl::float3 ivx(il.l.vx.x, il.l.vx.y, il.l.vx.z);
  sycl::float3 ivy(il.l.vy.x, il.l.vy.y, il.l.vy.z);
  sycl::float3 ivz(il.l.vz.x, il.l.vz.y, il.l.vz.z);
  sycl::float3  ip(il.p.x, il.p.y, il.p.z);
  return Transform(ivx,ivy,ivz,ip);
#else
  const LinearSpace3f l = { a.vx, a.vy, a.vz };
  const LinearSpace3f il = l.inverse();
  return Transform(il.vx, il.vy, il.vz, -xfmPoint(il,a.p));
#endif
}

Transform RandomSampler_getTransform(RandomSampler& self)
{
  const sycl::float3 u = RandomSampler_getFloat3(self) + sycl::float3(0.01f);
  const float r = 2.0f*M_PI*RandomSampler_getFloat(self);
  const sycl::float3 p = 10.0f*RandomSampler_getFloat3(self);
  const LinearSpace3f xfm = LinearSpace3f::rotate(u,r);
  return Transform(xfm.vx,xfm.vy,xfm.vz,p);
}

struct Bounds3f
{
  void extend( sycl::float3 p ) {
    lower = sycl::min(lower,p);
    upper = sycl::max(upper,p);
  }

  static Bounds3f empty() {
    return { sycl::float3(INFINITY), sycl::float3(-INFINITY) };
  }

  operator ze_rtas_aabb_exp_t () const {
    return { { lower.x(), lower.y(), lower.z() }, { upper.x(), upper.y(), upper.z() } };
  }
  
  sycl::float3 lower;
  sycl::float3 upper;
};

std::ostream& operator<<(std::ostream& out, const Bounds3f& b) {
  return out << "Bounds3f {" << b.lower << "," << b.upper  << "}";
}

const Bounds3f xfmBounds(const Transform& m, const Bounds3f& b) 
{ 
  Bounds3f dst = Bounds3f::empty();
  const sycl::float3 p0(b.lower.x(),b.lower.y(),b.lower.z()); dst.extend(xfmPoint(m,p0));
  const sycl::float3 p1(b.lower.x(),b.lower.y(),b.upper.z()); dst.extend(xfmPoint(m,p1));
  const sycl::float3 p2(b.lower.x(),b.upper.y(),b.lower.z()); dst.extend(xfmPoint(m,p2));
  const sycl::float3 p3(b.lower.x(),b.upper.y(),b.upper.z()); dst.extend(xfmPoint(m,p3));
  const sycl::float3 p4(b.upper.x(),b.lower.y(),b.lower.z()); dst.extend(xfmPoint(m,p4));
  const sycl::float3 p5(b.upper.x(),b.lower.y(),b.upper.z()); dst.extend(xfmPoint(m,p5));
  const sycl::float3 p6(b.upper.x(),b.upper.y(),b.lower.z()); dst.extend(xfmPoint(m,p6));
  const sycl::float3 p7(b.upper.x(),b.upper.y(),b.upper.z()); dst.extend(xfmPoint(m,p7));
  return dst;
}

struct Triangle
{
  Triangle()
    : v0(0.f,0.f,0.f), v1(0.f,0.f,0.f), v2(0.f,0.f,0.f), index(0) {}
  
  Triangle (sycl::float3 v0, sycl::float3 v1, sycl::float3 v2, uint32_t index)
    : v0(v0), v1(v1), v2(v2), index(index) {}

  sycl::float3 sample(float u, float v) const {
    return (1.0f-u-v)*v0 + u*v1 + v*v2;
  }

  sycl::float3 center() const {
    return (v0+v1+v2)/3.0f;
  }

  Bounds3f bounds() const
  {
    const sycl::float3 lower = sycl::min(v0,sycl::min(v1,v2));
    const sycl::float3 upper = sycl::max(v0,sycl::max(v1,v2));
    return { lower, upper };
  }

  const Triangle transform( Transform xfm ) const {
    return Triangle(xfmPoint(xfm,v0), xfmPoint(xfm,v1), xfmPoint(xfm,v2), index);
  }

  sycl::float3 v0;
  sycl::float3 v1;
  sycl::float3 v2;
  uint32_t index;
};

struct less_float3 {
  bool operator() ( const sycl::float3& a, const sycl::float3& b ) const {
    if (a.x() != b.x()) return a.x()  < b.x();
    if (a.y() != b.y()) return a.y()  < b.y();
    if (a.z() != b.z()) return a.z()  < b.z();
    return false;
  }
};

std::ostream& operator<<(std::ostream& out, const Triangle& tri) {
  return out << "Triangle {" << tri.v0 << "," << tri.v1 << "," << tri.v2  << "}";
}

struct Hit
{
  Transform local_to_world;
  Triangle triangle;
  bool procedural_triangle = false;
  bool procedural_instance = false;
  uint32_t instUserID = -1;
  uint32_t instID = -1;
  uint32_t geomID = -1;
  uint32_t primID = -1;
};


struct GEOMETRY_INSTANCE_DESC : ze_rtas_builder_instance_geometry_info_exp_t
{
  ze_rtas_transform_float3x4_aligned_column_major_exp_t xfmdata;
};

typedef union GEOMETRY_DESC
{
  ze_rtas_builder_geometry_type_exp_t geometryType;
  ze_rtas_builder_triangles_geometry_info_exp_t Triangles;
  ze_rtas_builder_quads_geometry_info_exp_t Quads;
  ze_rtas_builder_procedural_geometry_info_exp_t AABBs;
  GEOMETRY_INSTANCE_DESC Instance;

} GEOMETRY_DESC;

struct Geometry
{
  enum Type {
    TRIANGLE_MESH,
    INSTANCE
  };

  Geometry (Type type)
    : type(type) {}

  virtual void getDesc(GEOMETRY_DESC* desc) = 0;

  virtual void transform( const Transform xfm) {
    throw std::runtime_error("Geometry::transform not implemented");
  }

  virtual void buildAccel(sycl::device& device, sycl::context& context, BuildMode buildMode, ze_rtas_builder_build_quality_hint_exp_t quality) {
  };

  virtual void buildTriMap(Transform local_to_world, std::vector<uint32_t> id_stack, uint32_t instUserID, bool procedural_instance, std::vector<Hit>& tri_map) = 0;

  virtual size_t getNumPrimitives() const = 0;

  Type type;
};

struct TriangleMesh : public Geometry
{
public:

  TriangleMesh (ze_rtas_builder_geometry_exp_flags_t gflags = 0, bool procedural = false)
    : Geometry(Type::TRIANGLE_MESH),
      gflags(gflags), procedural(procedural),
      triangles_alloc(context,device,sycl::ext::oneapi::property::usm::device_read_only()), triangles(0,triangles_alloc),
      vertices_alloc (context,device,sycl::ext::oneapi::property::usm::device_read_only()), vertices(0,vertices_alloc) {}

  virtual ~TriangleMesh() {}

  void* operator new(size_t size) {
    return sycl::aligned_alloc_shared(64,size,device,context,sycl::ext::oneapi::property::usm::device_read_only());
  }
  void operator delete(void* ptr) {
    sycl::free(ptr,context);
  }

  size_t size() const {
    return triangles.size();
  }

  virtual void transform( const Transform xfm) override
  {
    for (size_t i=0; i<vertices.size(); i++)
      vertices[i] = xfmPoint(xfm,vertices[i]);
  }

  static void getBoundsCallback (ze_rtas_geometry_aabbs_exp_cb_params_t* params)
  {
    assert(params->stype == ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS);
    const TriangleMesh* mesh = (TriangleMesh*) params->pGeomUserPtr;

    for (uint32_t i=0; i<params->primIDCount; i++)
    {
      const uint32_t primID = params->primID+i;
      const Bounds3f bounds = mesh->getBounds(primID);
      ze_rtas_aabb_exp_t* boundsOut = params->pBoundsOut;
      boundsOut[i].lower.x = bounds.lower.x();
      boundsOut[i].lower.y = bounds.lower.y();
      boundsOut[i].lower.z = bounds.lower.z();
      boundsOut[i].upper.x = bounds.upper.x();
      boundsOut[i].upper.y = bounds.upper.y();
      boundsOut[i].upper.z = bounds.upper.z();
    }
  }
  
  virtual void getDesc(GEOMETRY_DESC* desc) override
  {
    if (procedural)
    {
      ze_rtas_builder_procedural_geometry_info_exp_t& out =  desc->AABBs;
      memset(&out,0,sizeof(out));
      out.geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL;
      out.geometryFlags = gflags;
      out.geometryMask = 0xFF;
      out.primCount = triangles.size();
      out.pfnGetBoundsCb = TriangleMesh::getBoundsCallback;
      out.pGeomUserPtr = this;
    }
    else
    {
      ze_rtas_builder_triangles_geometry_info_exp_t& out = desc->Triangles;
      memset(&out,0,sizeof(out));
      out.geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES;
      out.geometryFlags = gflags;
      out.geometryMask = 0xFF;
      out.triangleFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32;
      out.vertexFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3;
      out.pTriangleBuffer = (ze_rtas_triangle_indices_uint32_exp_t*) triangles.data();
      out.triangleCount = triangles.size();
      out.triangleStride = sizeof(sycl::int4);
      out.pVertexBuffer = (ze_rtas_float3_exp_t*) vertices.data();
      out.vertexCount = vertices.size();
      out.vertexStride = sizeof(sycl::float3);
    }
  }

  Triangle getTriangle( const uint32_t primID ) const
  {
    const sycl::float3 v0 = vertices[triangles[primID].x()];
    const sycl::float3 v1 = vertices[triangles[primID].y()];
    const sycl::float3 v2 = vertices[triangles[primID].z()];
    const uint32_t index = triangles[primID].w();
    return Triangle(v0,v1,v2,index);
  }

  Bounds3f getBounds( const uint32_t primID ) const {
    return getTriangle(primID).bounds();
  }

  uint32_t addVertex( const sycl::float3& v )
  {
    auto e = vertex_map.find(v);
    if (e != vertex_map.end()) return e->second;
    vertices.push_back(v);
    vertex_map[v] = vertices.size()-1;
    return vertices.size()-1;
  }

  void addTriangle( const Triangle& tri )
  {
    const uint32_t v0 = addVertex(tri.v0);
    const uint32_t v1 = addVertex(tri.v1);
    const uint32_t v2 = addVertex(tri.v2);
    triangles.push_back(sycl::int4(v0,v1,v2,tri.index));
  }

  void split(const sycl::float3 P, const sycl::float3 N, std::shared_ptr<TriangleMesh>& mesh0, std::shared_ptr<TriangleMesh>& mesh1)
  {
    mesh0 = std::shared_ptr<TriangleMesh>(new TriangleMesh(gflags,procedural));
    mesh1 = std::shared_ptr<TriangleMesh>(new TriangleMesh(gflags,procedural));
    
    for (uint32_t primID=0; primID<(uint32_t) size(); primID++)
    {
      const Triangle tri = getTriangle(primID);
      if (sycl::dot(tri.center()-P,N) < 0.0f) mesh0->addTriangle(tri);
      else                                    mesh1->addTriangle(tri);
    }
  }

  void split(std::shared_ptr<TriangleMesh>& mesh0, std::shared_ptr<TriangleMesh>& mesh1)
  {
    uint32_t N = (uint32_t) size();
    mesh0 = std::shared_ptr<TriangleMesh>(new TriangleMesh(gflags,procedural));
    mesh1 = std::shared_ptr<TriangleMesh>(new TriangleMesh(gflags,procedural));
    mesh0->triangles.reserve(triangles.size()/2+1);
    mesh1->triangles.reserve(triangles.size()/2+1);
    mesh0->vertices.reserve(vertices.size()/2+8);
    mesh1->vertices.reserve(vertices.size()/2+8);
    
    for (uint32_t primID=0; primID<N; primID++)
    {
      const Triangle tri = getTriangle(primID);
      if (primID<N/2) mesh0->addTriangle(tri);
      else            mesh1->addTriangle(tri);
    }
  }

  /* selects random sub-set of triangles */
  void selectRandom(const uint32_t numTriangles)
  {
    assert(numTriangles <= size());

    /* first randomize triangles */
    for (size_t i=0; i<size(); i++) {
      uint32_t j = RandomSampler_getUInt(rng) % size();
      std::swap(triangles[i],triangles[j]);
    }

    /* now we can easily select a random set of triangles */
    triangles.resize(numTriangles);

    /* now we sort the triangles again */
    std::sort(triangles.begin(), triangles.end(), []( sycl::int4 a, sycl::int4 b ) { return a.w() < b.w(); });

    /* and assign consecutive IDs */
    for (uint32_t i=0; i<numTriangles; i++)
      triangles[i].w() = i;
  }

  /* selects sequential sub-set of triangles */
  void selectSequential(const uint32_t numTriangles)
  {
    assert(numTriangles <= size());

    /* now we can easily select a random set of triangles */
    triangles.resize(numTriangles);
  }
  
  /* creates separate vertives for triangles */
  void unshareVertices()
  {
    vertices.reserve(vertices.size()+3*triangles.size());
    for (size_t i=0; i<triangles.size(); i++) {
      const sycl::int4 tri = triangles[i];
      const uint32_t v0 = (uint32_t) vertices.size();
      vertices.push_back(vertices[tri.x()]);
      const uint32_t v1 = (uint32_t) vertices.size();
      vertices.push_back(vertices[tri.y()]);
      const uint32_t v2 = (uint32_t) vertices.size();
      vertices.push_back(vertices[tri.z()]);
      triangles[i] = sycl::int4(v0,v1,v2,tri.w());
    }
  }

  virtual void buildTriMap(Transform local_to_world, std::vector<uint32_t> id_stack, uint32_t instUserID, bool procedural_instance, std::vector<Hit>& tri_map) override
  {
    uint32_t instID = -1;
    uint32_t geomID = -1;
    
    if (id_stack.size()) {
      geomID = id_stack.back();
      id_stack.pop_back();
    }
    
    if (id_stack.size()) {
      instID = id_stack.back();
      id_stack.pop_back();
    }

    assert(id_stack.size() == 0);
    
    for (uint32_t primID=0; primID<triangles.size(); primID++)
    {
      const Triangle tri = getTriangle(primID);
      assert(tri_map[tri.index].geomID == -1);
      tri_map[tri.index].instUserID = instUserID;
      tri_map[tri.index].primID = primID;
      tri_map[tri.index].geomID = geomID;
      tri_map[tri.index].instID = instID;
      tri_map[tri.index].procedural_triangle = procedural;
      tri_map[tri.index].procedural_instance = procedural_instance;
      tri_map[tri.index].triangle = tri;
      tri_map[tri.index].local_to_world = local_to_world;
    }
  }
  
  size_t getNumPrimitives() const override {
    return triangles.size();
  }

  
public:
  ze_rtas_builder_geometry_exp_flags_t gflags = 0;
  bool procedural = false;
  
  typedef sycl::usm_allocator<sycl::int4, sycl::usm::alloc::shared> triangles_alloc_ty;
  triangles_alloc_ty triangles_alloc;
  std::vector<sycl::int4, triangles_alloc_ty> triangles;

  typedef sycl::usm_allocator<sycl::float3, sycl::usm::alloc::shared> vertices_alloc_ty;
  vertices_alloc_ty vertices_alloc;
  std::vector<sycl::float3, vertices_alloc_ty> vertices;
  
  std::map<sycl::float3,uint32_t,less_float3> vertex_map;
};

template<typename Scene>
struct InstanceGeometryT : public Geometry
{
  InstanceGeometryT(const Transform& local2world, std::shared_ptr<Scene> scene, bool procedural, uint32_t instUserID)
    : Geometry(Type::INSTANCE), procedural(procedural), instUserID(instUserID), local2world(local2world), scene(scene) {}

  virtual ~InstanceGeometryT() {}

  void* operator new(size_t size) {
    return sycl::aligned_alloc_shared(64,size,device,context,sycl::ext::oneapi::property::usm::device_read_only());
  }
  void operator delete(void* ptr) {
    sycl::free(ptr,context);
  }

  static void getBoundsCallback (ze_rtas_geometry_aabbs_exp_cb_params_t* params)
  {
    assert(params->stype == ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS);
    assert(params->primID == 0);
    assert(params->primIDCount == 1);
    const InstanceGeometryT* inst = (InstanceGeometryT*) params->pGeomUserPtr;
    const Bounds3f scene_bounds = inst->scene->getBounds();
    const Bounds3f bounds = xfmBounds(inst->local2world, scene_bounds);
    ze_rtas_aabb_exp_t* boundsOut = params->pBoundsOut;
    boundsOut->lower.x = bounds.lower.x();
    boundsOut->lower.y = bounds.lower.y();
    boundsOut->lower.z = bounds.lower.z();
    boundsOut->upper.x = bounds.upper.x();
    boundsOut->upper.y = bounds.upper.y();
    boundsOut->upper.z = bounds.upper.z();
  }

  virtual void getDesc(GEOMETRY_DESC* desc) override
  {
    if (procedural)
    {
      ze_rtas_builder_procedural_geometry_info_exp_t& out = desc->AABBs;
      memset(&out,0,sizeof(out));
      out.geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL;
      out.geometryFlags = 0;
      out.geometryMask = 0xFF;
      out.primCount = 1;
      out.pfnGetBoundsCb = InstanceGeometryT::getBoundsCallback;
      out.pGeomUserPtr = this;
    }
    else
    {
      GEOMETRY_INSTANCE_DESC& out = desc->Instance;
      memset(&out,0,sizeof(GEOMETRY_INSTANCE_DESC));
      out.geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE;
      out.instanceFlags = 0;
      out.geometryMask = 0xFF;
      out.instanceUserID = instUserID;
      out.transformFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ALIGNED_COLUMN_MAJOR;
      out.pTransform = (float*)&out.xfmdata;
      out.xfmdata.vx_x = local2world.vx.x();
      out.xfmdata.vx_y = local2world.vx.y();
      out.xfmdata.vx_z = local2world.vx.z();
      out.xfmdata.pad0 = 0.0f;
      out.xfmdata.vy_x = local2world.vy.x();
      out.xfmdata.vy_y = local2world.vy.y();
      out.xfmdata.vy_z = local2world.vy.z();
      out.xfmdata.pad1 = 0.0f;
      out.xfmdata.vz_x = local2world.vz.x();
      out.xfmdata.vz_y = local2world.vz.y();
      out.xfmdata.vz_z = local2world.vz.z();
      out.xfmdata.pad2 = 0.0f;
      out.xfmdata.p_x  = local2world.p.x();
      out.xfmdata.p_y  = local2world.p.y();
      out.xfmdata.p_z  = local2world.p.z();
      out.xfmdata.pad3  = 0.0f;
      out.pBounds = &scene->bounds;
      out.pAccelerationStructure = scene->getAccel();
    }
  }

  virtual void buildAccel(sycl::device& device, sycl::context& context, BuildMode buildMode, ze_rtas_builder_build_quality_hint_exp_t quality) override {
    scene->buildAccel(device,context,buildMode);
  }

  virtual void buildTriMap(Transform local_to_world_in, std::vector<uint32_t> id_stack, uint32_t instUserID, bool procedural_instance, std::vector<Hit>& tri_map) override {
    instUserID = this->instUserID;
    scene->buildTriMap(local_to_world_in * local2world, id_stack, instUserID, procedural, tri_map);
  }

  size_t getNumPrimitives() const override {
    return 1;
  }

  bool procedural;
  uint32_t instUserID = -1;
  Transform local2world;
  std::shared_ptr<Scene> scene;
};

std::shared_ptr<TriangleMesh> createTrianglePlane (const sycl::float3& p0, const sycl::float3& dx, const sycl::float3& dy, size_t width, size_t height)
{
  std::shared_ptr<TriangleMesh> mesh(new TriangleMesh);
  mesh->triangles.resize(2*width*height);
  mesh->vertices.resize((width+1)*(height+1));
  
  for (size_t y=0; y<=height; y++) {
    for (size_t x=0; x<=width; x++) {
      sycl::float3 p = p0+float(x)/float(width)*dx+float(y)/float(height)*dy;
      size_t i = y*(width+1)+x;
      mesh->vertices[i] = p;
    }
  }
  for (size_t y=0; y<height; y++) {
    for (size_t x=0; x<width; x++) {
      size_t i = 2*y*width+2*x;
      size_t p00 = (y+0)*(width+1)+(x+0);
      size_t p01 = (y+0)*(width+1)+(x+1);
      size_t p10 = (y+1)*(width+1)+(x+0);
      size_t p11 = (y+1)*(width+1)+(x+1);
      mesh->triangles[i+0] = sycl::int4((int)p00,(int)p01,(int)p10,i+0);
      mesh->triangles[i+1] = sycl::int4((int)p11,(int)p10,(int)p01,i+1);
    }
  }
  return mesh;
}

void* alloc_accel_buffer_internal(size_t bytes, sycl::device device, sycl::context context)
{
  ze_context_handle_t hContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  ze_device_handle_t  hDevice  = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);

  ze_rtas_device_exp_properties_t rtasProp = { ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES };
  ze_device_properties_t devProp = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &rtasProp };
  ze_result_t err = ZeWrapper::zeDeviceGetProperties(hDevice, &devProp );
  if (err != ZE_RESULT_SUCCESS)
    throw std::runtime_error("zeDeviceGetProperties failed");
  
  ze_raytracing_mem_alloc_ext_desc_t rt_desc;
  rt_desc.stype = ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC;
  rt_desc.pNext = nullptr;
  rt_desc.flags = 0;
    
  ze_device_mem_alloc_desc_t device_desc;
  device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  device_desc.pNext = &rt_desc;
  device_desc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
  device_desc.ordinal = 0;

  ze_host_mem_alloc_desc_t host_desc;
  host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  host_desc.pNext = nullptr;
  host_desc.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED;
  
  void* ptr = nullptr;
  ze_result_t result = ZeWrapper::zeMemAllocShared(hContext,&device_desc,&host_desc,bytes,rtasProp.rtasBufferAlignment,hDevice,&ptr);
  if (result != ZE_RESULT_SUCCESS)
    throw std::runtime_error("accel allocation failed");
  return ptr;
}

void free_accel_buffer_internal(void* ptr, sycl::context context)
{
  if (ptr == nullptr) return;
  ze_context_handle_t hContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  ze_result_t result = ZeWrapper::zeMemFree(hContext,ptr);
  if (result != ZE_RESULT_SUCCESS)
    throw std::runtime_error("accel free failed");
}

struct Block {
  Block (size_t bytes, sycl::device device, sycl::context context)
    : base((char*)alloc_accel_buffer_internal(bytes,device,context)), total(bytes), cur(0) {}
  ~Block() {
    free_accel_buffer_internal((void*)base,context);
  }
  void* alloc(size_t bytes) {
    bytes &= -128;
    if (cur+bytes > total) return nullptr;
    void* ptr = &base[cur];
    cur += bytes;
    return ptr;
  }
  char* base = nullptr;
  size_t total = 0;
  size_t cur = 0;
  
};

bool g_use_accel_blocks = true;
std::vector<std::shared_ptr<Block>> g_blocks;

void* alloc_accel_buffer(size_t bytes, sycl::device device, sycl::context context)
{
  if (!g_use_accel_blocks)
    return alloc_accel_buffer_internal(bytes,device,context);

  if (g_blocks.size() == 0)
    g_blocks.push_back(std::shared_ptr<Block>(new Block(1024*1024,device,context)));

  if (bytes > 1024*1024) {
    g_blocks.push_back(std::shared_ptr<Block>(new Block(bytes,device,context)));
    void* ptr = g_blocks.back()->alloc(bytes);
    assert(ptr);
    return ptr;
  }
    
  void* ptr = g_blocks.back()->alloc(bytes);
  if (ptr) return ptr;
  
  g_blocks.push_back(std::shared_ptr<Block>(new Block(1024*1024,device,context)));
  ptr = g_blocks.back()->alloc(bytes);
  assert(ptr);
  return ptr;
}

void free_accel_buffer(void* ptr, sycl::context context)
{
  if (!g_use_accel_blocks)
    return free_accel_buffer_internal(ptr,context);
}

struct Scene
{
  typedef InstanceGeometryT<Scene> InstanceGeometry;
  
  Scene()
    : geometries_alloc(context,device,sycl::ext::oneapi::property::usm::device_read_only()), geometries(0,geometries_alloc), bounds(Bounds3f::empty()), accel(nullptr) {}
      
  Scene(uint32_t width, uint32_t height, bool opaque, bool procedural)
    : geometries_alloc(context,device,sycl::ext::oneapi::property::usm::device_read_only()), geometries(0,geometries_alloc), bounds(Bounds3f::empty()), accel(nullptr) 
  {
    std::shared_ptr<TriangleMesh> plane = createTrianglePlane(sycl::float3(0,0,0), sycl::float3(width,0,0), sycl::float3(0,height,0), width, height);
    plane->gflags = opaque ? (ze_rtas_builder_geometry_exp_flag_t) 0 : ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_NON_OPAQUE;
    plane->procedural = procedural;
    geometries.push_back(plane);
  }

  ~Scene() {
    free_accel_buffer(accel,context);
  }

  void* operator new(size_t size) {
    return sycl::aligned_alloc_shared(64,size,device,context,sycl::ext::oneapi::property::usm::device_read_only());
  }

  void operator delete(void* ptr) {
    sycl::free(ptr,context);
  }

  void add(std::shared_ptr<TriangleMesh> mesh) {
    geometries.push_back(mesh);
  }

  void splitIntoGeometries(uint32_t numGeometries)
  {
    bool progress = true;
    while (progress)
    {
      size_t N = geometries.size();
      progress = false;
      for (uint32_t i=0; i<N; i++)
      {
        if (std::shared_ptr<TriangleMesh> mesh = std::dynamic_pointer_cast<TriangleMesh>(geometries[i]))
        {
          if (mesh->size() <= 1) continue;
          progress = true;
          
          /*const Triangle tri = mesh->getTriangle(RandomSampler_getUInt(rng)%mesh->size());
            const float u = 2.0f*M_PI*RandomSampler_getFloat(rng);
            const sycl::float3 P = tri.center();
            const sycl::float3 N(cosf(u),sinf(u),0.0f);
            
            std::shared_ptr<TriangleMesh> mesh0, mesh1;
            mesh->split(P,N,mesh0,mesh1);*/
          
          std::shared_ptr<TriangleMesh> mesh0, mesh1;
          mesh->split(mesh0,mesh1);
          geometries[i] = std::dynamic_pointer_cast<Geometry>(mesh0);
          geometries.push_back(std::dynamic_pointer_cast<Geometry>(mesh1));

          if (geometries.size() >= numGeometries)
            return;
        }
      }
    }
    assert(geometries.size() == numGeometries);
  }

  /* splits each primitive into a geometry */
  void splitIntoGeometries()
  {
    /* count number of triangles */
    uint32_t numTriangles = 0;
    for (uint32_t i=0; i<geometries.size(); i++)
    {
      if (std::shared_ptr<TriangleMesh> mesh = std::dynamic_pointer_cast<TriangleMesh>(geometries[i])) {
        numTriangles++;
      }
    }
        
    std::vector<std::shared_ptr<Geometry>, geometries_alloc_ty> new_geometries(0,geometries_alloc);
    new_geometries.reserve(numTriangles);
    
    for (uint32_t i=0; i<geometries.size(); i++)
    {
      if (std::shared_ptr<TriangleMesh> mesh = std::dynamic_pointer_cast<TriangleMesh>(geometries[i]))
      {
        if (mesh->size() <= 1) {
          new_geometries.push_back(geometries[i]);
          continue;
        }

        for (uint32_t j=0; j<mesh->size(); j++) {
          std::shared_ptr<TriangleMesh> mesh0(new TriangleMesh(mesh->gflags,mesh->procedural));
          mesh0->triangles.reserve(1);
          mesh->vertices.reserve(3);
          mesh0->addTriangle(mesh->getTriangle(j));
          new_geometries.push_back(mesh0);
        }
      }
    }

    geometries = new_geometries;
  }

  void createInstances(uint32_t maxInstances, uint32_t blockSize = 1, bool procedural = false)
  {
    std::vector<std::shared_ptr<Geometry>, geometries_alloc_ty> instances(0,geometries_alloc);
    
    for (uint32_t i=0; i<geometries.size(); i+=blockSize)
    {
      const uint32_t begin = i;
      const uint32_t end   = std::min((uint32_t)geometries.size(),i+blockSize);
      
      if (instances.size() >= maxInstances)
      {
        for (uint32_t j=begin; j<end; j++) {
          instances.push_back(geometries[j]);
        }
        continue;
      }
      
      const Transform local2world = RandomSampler_getTransform(rng);
      const Transform world2local = rcp(local2world);

      std::shared_ptr<Scene> scene(new Scene);
      for (size_t j=begin; j<end; j++) {
        geometries[j]->transform(world2local);
        scene->geometries.push_back(geometries[j]);
      }

      //std::shared_ptr<InstanceGeometry> instance = std::make_shared<InstanceGeometry>(local2world,scene,procedural);
      uint32_t instUserID = RandomSampler_getUInt(rng);
      std::shared_ptr<InstanceGeometry> instance(new InstanceGeometry(local2world,scene,procedural,instUserID));
      instances.push_back(instance);
    }

    geometries = instances;
  }

  void mixTrianglesAndProcedurals()
  {
    for (uint32_t i=0; i<geometries.size(); i++)
      if (std::shared_ptr<TriangleMesh> mesh = std::dynamic_pointer_cast<TriangleMesh>(geometries[i]))
        mesh->procedural = i%2;
  }

  void addNullGeometries(uint32_t D)
  {
    size_t N = geometries.size();
    geometries.resize(N+D);
    if (N == 0) return;

    for (size_t g=N; g<N+D; g++) {
      uint32_t k = RandomSampler_getUInt(rng) % N;
      std::swap(geometries[g],geometries[k]);
    }
  }

  void buildAccel(sycl::device& device, sycl::context& context, BuildMode buildMode, bool benchmark = false)
  {
     ze_rtas_builder_build_quality_hint_exp_t quality = (ze_rtas_builder_build_quality_hint_exp_t) (RandomSampler_getUInt(rng) % 3);
     
    /* fill geometry descriptor buffer */
    std::vector<GEOMETRY_DESC> desc(size());
    std::vector<const ze_rtas_builder_geometry_info_exp_t*> geom(size());
    size_t numPrimitives = 0;
    for (size_t geomID=0; geomID<size(); geomID++)
    {
      const std::shared_ptr<Geometry>& g = geometries[geomID];
      
      /* skip NULL geometries */
      if (g == nullptr) {
        geom[geomID] = nullptr;
        continue;
      }

      numPrimitives += g->getNumPrimitives();
      g->buildAccel(device,context,buildMode,quality);
      g->getDesc(&desc[geomID]);
      geom[geomID] = (const ze_rtas_builder_geometry_info_exp_t*) &desc[geomID];
    }

    ze_device_handle_t hDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);

    ze_rtas_device_exp_properties_t rtasProp = { ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES };
    ze_device_properties_t devProp = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &rtasProp };
    ze_result_t err = ZeWrapper::zeDeviceGetProperties(hDevice, &devProp );
    if (err != ZE_RESULT_SUCCESS)
      throw std::runtime_error("zeDeviceGetProperties failed");
    
    /* estimate accel size */
    size_t accelBufferBytesOut = 0;
    ze_rtas_aabb_exp_t bounds;
    ze_rtas_builder_build_op_exp_desc_t args;
    memset(&args,0,sizeof(args));
    args.stype = ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC;
    args.pNext = nullptr;
    args.rtasFormat = rtasProp.rtasFormat;
    args.buildQuality = quality;
    args.buildFlags = 0;
    args.ppGeometries = (const ze_rtas_builder_geometry_info_exp_t**) geom.data();
    args.numGeometries = geom.size();

    /* just for debugging purposes */
#if defined(EMBREE_SYCL_ALLOC_DISPATCH_GLOBALS)
    ze_rtas_builder_build_op_debug_exp_desc_t buildOpDebug = { ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_DEBUG_EXP_DESC };
    buildOpDebug.dispatchGlobalsPtr = dispatchGlobalsPtr;
    args.pNext = &buildOpDebug;
#endif
    
    ze_rtas_builder_exp_properties_t size = { ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES };
    err = ZeWrapper::zeRTASBuilderGetBuildPropertiesExp(hBuilder,&args,&size);
    if (err != ZE_RESULT_SUCCESS)
      throw std::runtime_error("BVH size estimate failed");

    if (size.rtasBufferSizeBytesExpected > size.rtasBufferSizeBytesMaxRequired)
      throw std::runtime_error("expected larger than worst case");

    /* allocate scratch buffer */
    size_t sentinelBytes = 1024; // add that many zero bytes to catch buffer overruns
    std::vector<char> scratchBuffer(size.scratchBufferSizeBytes+sentinelBytes);
    memset(scratchBuffer.data(),0,scratchBuffer.size());

    accel = nullptr;
    size_t accelBytes = 0;
    
    /* build with different modes */
    switch (buildMode)
    {
    case BuildMode::BUILD_WORST_CASE_SIZE: {

      accelBytes = size.rtasBufferSizeBytesMaxRequired;
      accel = alloc_accel_buffer(accelBytes+sentinelBytes,device,context);
      memset(accel,0,accelBytes+sentinelBytes);

      /* build accel */
      double t0 = embree::getSeconds();
      size_t numIterations = benchmark ? 16 : 1;

      for (size_t i=0; i<numIterations; i++)
      {
        err = ZeWrapper::zeRTASBuilderBuildExp(hBuilder,&args,
                                        scratchBuffer.data(),scratchBuffer.size(),
                                        accel, accelBytes,
                                        parallelOperation,
                                        nullptr, &bounds, &accelBufferBytesOut);

        if (parallelOperation)
        {
          assert(err == ZE_RESULT_EXP_RTAS_BUILD_DEFERRED);
          
          ze_rtas_parallel_operation_exp_properties_t prop = { ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES };
          err = ZeWrapper::zeRTASParallelOperationGetPropertiesExp(parallelOperation,&prop);
          if (err != ZE_RESULT_SUCCESS)
            throw std::runtime_error("get max concurrency failed");
          
          tbb::parallel_for(0u, prop.maxConcurrency, 1u, [&](uint32_t) {
            err = ZeWrapper::zeRTASParallelOperationJoinExp(parallelOperation);
          });
        }
      
        if (err != ZE_RESULT_SUCCESS)
          throw std::runtime_error("build error");
      }
      double t1 = embree::getSeconds();

      if (benchmark) {
        double dt = (t1-t0)/double(numIterations);
        std::cout << double(numPrimitives)/dt*1E-6 << " Mprims/s" << std::endl;
      }
      break;
    }
    case BuildMode::BUILD_EXPECTED_SIZE: {
      
      size_t bytes = size.rtasBufferSizeBytesExpected;
      for (size_t i=0; i<=16; i++) // FIXME: reduce worst cast iteration number
      {
        if (i == 16)
          throw std::runtime_error("build requires more than 16 iterations");
        
        /* allocate BVH data */
        free_accel_buffer(accel,context);
        accelBytes = bytes;
        accel = alloc_accel_buffer(accelBytes+sentinelBytes,device,context);
        memset(accel,0,accelBytes+sentinelBytes);

        /* build accel */
        err = ZeWrapper::zeRTASBuilderBuildExp(hBuilder,&args,
                                        scratchBuffer.data(),scratchBuffer.size(),
                                        accel, accelBytes,
                                        parallelOperation,
                                        nullptr, &bounds, &accelBufferBytesOut);

        if (parallelOperation)
        {
          assert(err == ZE_RESULT_EXP_RTAS_BUILD_DEFERRED);
          
          ze_rtas_parallel_operation_exp_properties_t prop = { ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES };
          err = ZeWrapper::zeRTASParallelOperationGetPropertiesExp(parallelOperation,&prop);
          if (err != ZE_RESULT_SUCCESS)
            throw std::runtime_error("get max concurrency failed");
          
          tbb::parallel_for(0u, prop.maxConcurrency, 1u, [&](uint32_t) {
            err = ZeWrapper::zeRTASParallelOperationJoinExp(parallelOperation);
          });
        }
        
        if (err != ZE_RESULT_EXP_RTAS_BUILD_RETRY)
          break;

        if (accelBufferBytesOut < bytes || size.rtasBufferSizeBytesMaxRequired < accelBufferBytesOut )
          throw std::runtime_error("failed build returned wrong new estimate");

        bytes = accelBufferBytesOut;
      }
      
      if (err != ZE_RESULT_SUCCESS)
        throw std::runtime_error("build error");

      break;
    }
    }

    this->bounds = bounds;

    if (!benchmark)
    {
      /* scratch buffer bounds check */
      for (size_t i=size.scratchBufferSizeBytes; i<size.scratchBufferSizeBytes+sentinelBytes; i++) {
        if (scratchBuffer[i] == 0x00) continue;
        throw std::runtime_error("scratch buffer bounds check failed");
      }
      /* acceleration structure bounds check */
      for (size_t i=accelBytes; i<accelBytes+sentinelBytes; i++) {
        if (((char*)accel)[i] == 0x00) continue;
        throw std::runtime_error("acceleration buffer bounds check failed");
      }
      /* check if returned size of acceleration structure is correct */
      for (size_t i=accelBufferBytesOut; i<accelBytes; i++) {
        if (((char*)accel)[i] == 0x00) continue;
        throw std::runtime_error("wrong acceleration structure size returned");
      }
    }
  }
  
  void buildTriMap(Transform local_to_world, std::vector<uint32_t> id_stack, uint32_t instUserID, bool procedural_instance, std::vector<Hit>& tri_map)
  {    
    for (uint32_t geomID=0; geomID<geometries.size(); geomID++)
    {
      if (geometries[geomID] == nullptr)
        continue;
      
      id_stack.push_back(geomID);
      geometries[geomID]->buildTriMap(local_to_world,id_stack,instUserID,procedural_instance,tri_map);
      id_stack.pop_back();
    }
  }

  size_t size() const {
    return geometries.size();
  }

  Bounds3f getBounds() {
    return {
      { bounds.lower.x, bounds.lower.y, bounds.lower.z },
      { bounds.upper.x, bounds.upper.y, bounds.upper.z }
    };
  }
  
  void* getAccel() {
    return accel;
  }

  std::shared_ptr<Geometry> operator[] ( size_t i ) { return geometries[i]; }

  typedef sycl::usm_allocator<std::shared_ptr<Geometry>, sycl::usm::alloc::shared> geometries_alloc_ty;
  geometries_alloc_ty geometries_alloc;
  std::vector<std::shared_ptr<Geometry>, geometries_alloc_ty> geometries;

  ze_rtas_aabb_exp_t bounds;
  void* accel;
};

void exception_handler(sycl::exception_list exceptions)
{
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch(sycl::exception const& e) {
      std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
    }
  }
};

void render(uint32_t i, const TestInput& in, TestOutput& out, intel_raytracing_acceleration_structure_t accel)
{
  intel_raytracing_ext_flag_t flags = intel_get_raytracing_ext_flag();
  if (!(flags & intel_raytracing_ext_flag_ray_query))
    return;
  
  /* setup ray */
  intel_ray_desc_t ray;
  ray.origin = in.org;
  ray.direction = in.dir;
  ray.tmin = in.tnear;
  ray.tmax = in.tfar;
  ray.mask = in.mask;
  ray.flags = (intel_ray_flags_t) in.flags;

  /* trace ray */
  intel_ray_query_t query = intel_ray_query_init(ray,accel);
  intel_ray_query_start_traversal(query);
  intel_ray_query_sync(query);

  /* return ray data of level 0 */
  out.ray0_org = intel_get_ray_origin(query,0);
  out.ray0_dir = intel_get_ray_direction(query,0);
  out.ray0_tnear = intel_get_ray_tmin(query,0);
  out.ray0_mask = intel_get_ray_mask(query,0);
  out.ray0_flags = intel_get_ray_flags(query,0);
  
  /* clear ray data of level N */
  out.rayN_org = sycl::float3(0.f,0.f,0.f);
  out.rayN_dir = sycl::float3(0.f,0.f,0.f);
  out.rayN_tnear = 0.0f;
  out.rayN_mask = 0;
  out.rayN_flags = 0;

  /* potential hit */
  if (!intel_is_traversal_done(query))
  {
    out.hit_type = TEST_POTENTIAL_HIT;
    out.bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_potential_hit );
    out.hit_candidate = intel_get_hit_candidate( query, intel_hit_type_potential_hit );
    out.t = intel_get_hit_distance(query, intel_hit_type_potential_hit);
    out.u = intel_get_hit_barycentrics(query, intel_hit_type_potential_hit).x;
    out.v = intel_get_hit_barycentrics(query, intel_hit_type_potential_hit).y;
    out.front_face = intel_get_hit_front_face( query, intel_hit_type_potential_hit );
    out.instUserID = intel_get_hit_instance_user_id( query, intel_hit_type_potential_hit );
    out.instID = intel_get_hit_instance_id( query, intel_hit_type_potential_hit );
    out.geomID = intel_get_hit_geometry_id( query, intel_hit_type_potential_hit );
    if (i%2) out.primID = intel_get_hit_triangle_primitive_id( query, intel_hit_type_potential_hit );
    else     out.primID = intel_get_hit_primitive_id         ( query, intel_hit_type_potential_hit );
    intel_float3 vertex_out[3];
    intel_get_hit_triangle_vertices(query, vertex_out, intel_hit_type_potential_hit);
    out.v0 = vertex_out[0];
    out.v1 = vertex_out[1];
    out.v2 = vertex_out[2];

    /* return ray data at current level */
    uint32_t bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_potential_hit );
    out.rayN_org = intel_get_ray_origin(query,bvh_level);
    out.rayN_dir = intel_get_ray_direction(query,bvh_level);
    out.rayN_tnear = intel_get_ray_tmin(query,bvh_level);
    out.rayN_mask = intel_get_ray_mask(query,bvh_level);
    out.rayN_flags = intel_get_ray_flags(query,bvh_level);

    /* return instance transformations */
    out.world_to_object = intel_get_hit_world_to_object(query,intel_hit_type_potential_hit);
    out.object_to_world = intel_get_hit_object_to_world(query,intel_hit_type_potential_hit);
  }

  /* committed hit */
  else if (intel_has_committed_hit(query))
  {
    out.hit_type = TEST_COMMITTED_HIT;
    out.bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_committed_hit );
    out.hit_candidate = intel_get_hit_candidate( query, intel_hit_type_committed_hit );
    out.t = intel_get_hit_distance(query, intel_hit_type_committed_hit);
    out.u = intel_get_hit_barycentrics(query, intel_hit_type_committed_hit).x;
    out.v = intel_get_hit_barycentrics(query, intel_hit_type_committed_hit).y;
    out.front_face = intel_get_hit_front_face( query, intel_hit_type_committed_hit );
    out.instUserID = intel_get_hit_instance_user_id( query, intel_hit_type_committed_hit );
    out.instID = intel_get_hit_instance_id( query, intel_hit_type_committed_hit );
    out.geomID = intel_get_hit_geometry_id( query, intel_hit_type_committed_hit );
    if (i%2) out.primID = intel_get_hit_triangle_primitive_id( query, intel_hit_type_committed_hit );
    else     out.primID = intel_get_hit_primitive_id         ( query, intel_hit_type_committed_hit );
    intel_float3 vertex_out[3];
    intel_get_hit_triangle_vertices(query, vertex_out, intel_hit_type_committed_hit);
    out.v0 = vertex_out[0];
    out.v1 = vertex_out[1];
    out.v2 = vertex_out[2];

    /* return instance transformations */
    out.world_to_object = intel_get_hit_world_to_object(query,intel_hit_type_committed_hit);
    out.object_to_world = intel_get_hit_object_to_world(query,intel_hit_type_committed_hit);
  }

  /* miss */
  else {
    out.hit_type = TEST_MISS;
  }

  /* abandon ray query */
  intel_ray_query_abandon(query);
}

void render_loop(uint32_t i, const TestInput& in, TestOutput& out, size_t scene_in, intel_raytracing_acceleration_structure_t accel, TestType test)
{
  intel_raytracing_ext_flag_t flags = intel_get_raytracing_ext_flag();
  if (!(flags & intel_raytracing_ext_flag_ray_query))
    return;
  
  /* setup ray */
  intel_ray_desc_t ray;
  ray.origin = in.org;
  ray.direction = in.dir;
  ray.tmin = in.tnear;
  ray.tmax = in.tfar;
  ray.mask = in.mask;
  ray.flags = (intel_ray_flags_t) in.flags;
  
  /* trace ray */
  intel_ray_query_t query = intel_ray_query_init(ray,accel);
  intel_ray_query_start_traversal(query);
  intel_ray_query_sync(query);
  
  /* return ray data of level 0 */
  out.ray0_org = intel_get_ray_origin(query,0);
  out.ray0_dir = intel_get_ray_direction(query,0);
  out.ray0_tnear = intel_get_ray_tmin(query,0);
  out.ray0_mask = intel_get_ray_mask(query,0);
  out.ray0_flags = intel_get_ray_flags(query,0);
  
  /* clear ray data of level N */
  out.rayN_org = sycl::float3(0.f,0.f,0.f);
  out.rayN_dir = sycl::float3(0.f,0.f,0.f);
  out.rayN_tnear = 0.0f;
  out.rayN_mask = 0;
  out.rayN_flags = 0;

  Scene* scenes[2];
  scenes[0] = (Scene*) scene_in;
  scenes[1] = nullptr;

  /* traversal loop */
  while (!intel_is_traversal_done(query))
  {
    const intel_candidate_type_t candidate = intel_get_hit_candidate(query, intel_hit_type_potential_hit);

    if (candidate == intel_candidate_type_triangle)
    {
      if (test == TestType::TRIANGLES_POTENTIAL_HIT)
      {
        out.hit_type = TEST_POTENTIAL_HIT;
        out.bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_potential_hit );
        out.hit_candidate = intel_get_hit_candidate( query, intel_hit_type_potential_hit );
        out.t = intel_get_hit_distance(query, intel_hit_type_potential_hit);
        out.u = intel_get_hit_barycentrics(query, intel_hit_type_potential_hit).x;
        out.v = intel_get_hit_barycentrics(query, intel_hit_type_potential_hit).y;
        out.front_face = intel_get_hit_front_face( query, intel_hit_type_potential_hit );
        out.instUserID = intel_get_hit_instance_user_id( query, intel_hit_type_potential_hit );
        out.instID = intel_get_hit_instance_id( query, intel_hit_type_potential_hit );
        out.geomID = intel_get_hit_geometry_id( query, intel_hit_type_potential_hit );
        if (i%2) out.primID = intel_get_hit_triangle_primitive_id( query, intel_hit_type_potential_hit );
        else     out.primID = intel_get_hit_primitive_id         ( query, intel_hit_type_potential_hit );
        intel_float3 vertex_out[3];
        intel_get_hit_triangle_vertices(query, vertex_out, intel_hit_type_potential_hit);
        out.v0 = vertex_out[0];
        out.v1 = vertex_out[1];
        out.v2 = vertex_out[2];

        /* return instance transformations */
        out.world_to_object = intel_get_hit_world_to_object(query,intel_hit_type_committed_hit);
        out.object_to_world = intel_get_hit_object_to_world(query,intel_hit_type_committed_hit);
        
        /* return ray data at current level */
        uint32_t bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_potential_hit );
        out.rayN_org = intel_get_ray_origin(query,bvh_level);
        out.rayN_dir = intel_get_ray_direction(query,bvh_level);
        out.rayN_tnear = intel_get_ray_tmin(query,bvh_level);
        out.rayN_mask = intel_get_ray_mask(query,bvh_level);
        out.rayN_flags = intel_get_ray_flags(query,bvh_level);
        return;
      }
    
      if (test == TestType::TRIANGLES_ANYHIT_SHADER_COMMIT)
        intel_ray_query_commit_potential_hit(query);
    }

    else if (candidate == intel_candidate_type_procedural)
    {
      const uint32_t bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_potential_hit );
      
      const uint32_t instID = intel_get_hit_instance_id( query, intel_hit_type_potential_hit );
      const uint32_t geomID = intel_get_hit_geometry_id( query, intel_hit_type_potential_hit );
      const uint32_t primID = intel_get_hit_primitive_id( query, intel_hit_type_potential_hit );

      Geometry* geom = nullptr;
      if (instID != -1) {
        Scene::InstanceGeometry* instance = (Scene::InstanceGeometry*) (scenes[0]->geometries.data() + instID)->get();
        geom = (instance->scene->geometries.data() + geomID)->get();
      } else {
        geom = (scenes[bvh_level]->geometries.data() + geomID)->get();
      }

      if (geom->type == Geometry::TRIANGLE_MESH)
      {
        const TriangleMesh* mesh = (TriangleMesh*) geom;

        const sycl::int4 tri = *(mesh->triangles.data() + primID);
        const sycl::float3 tri_v0 = *(mesh->vertices.data() + tri.x());
        const sycl::float3 tri_v1 = *(mesh->vertices.data() + tri.y());
        const sycl::float3 tri_v2 = *(mesh->vertices.data() + tri.z());

        /* calculate vertices relative to ray origin */
        const sycl::float3 O = intel_get_ray_origin(query,bvh_level);
        const sycl::float3 D = intel_get_ray_direction(query,bvh_level);
        const float tnear = intel_get_ray_tmin(query,bvh_level);
        const float tfar = intel_get_hit_distance(query, intel_hit_type_committed_hit);
        const sycl::float3 v0 = tri_v0-O;
        const sycl::float3 v1 = tri_v1-O;
        const sycl::float3 v2 = tri_v2-O;
        
        /* calculate triangle edges */
        const sycl::float3 e0 = v2-v0;
        const sycl::float3 e1 = v0-v1;
        const sycl::float3 e2 = v1-v2;
        
        /* perform edge tests */
        const float U = sycl::dot(cross(e0,v2+v0),D);
        const float V = sycl::dot(cross(e1,v0+v1),D);
        const float W = sycl::dot(cross(e2,v1+v2),D);
        const float UVW = U+V+W;
        bool valid = (std::min(U,std::min(V,W)) >= -0.0f) || (std::max(U,std::max(V,W)) <= 0.0f);
        
        /* calculate geometry normal and denominator */
        const sycl::float3 Ng = sycl::cross(e2,e1);
        const float den = 2.0f*(dot(Ng,D));
        
        /* perform depth test */
        const float T = 2.0f*dot(v0,Ng);
        const float t = T/den;
        const float u = U/UVW;
        const float v = V/UVW;
        valid &= tnear <= t & t <= tfar;
        valid &= den != 0.0f;

        /* commit hit */
        if (valid)
          intel_ray_query_commit_potential_hit_override(query,t,sycl::float2(u,v));
      }
      else if (geom->type == Geometry::INSTANCE)
      {
        const Scene::InstanceGeometry* inst = (Scene::InstanceGeometry*) geom;
        const Transform local2world = inst->local2world;
        const Transform world2local = rcp(local2world);
        
        /* load ray */
        const uint32_t bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_potential_hit );
        const sycl::float3 O = intel_get_ray_origin(query,bvh_level);
        const sycl::float3 D = intel_get_ray_direction(query,bvh_level);

        /* transform ray */
        const sycl::float3 O1 = xfmPoint(world2local, O);
        const sycl::float3 D1 = xfmVector(world2local, D);

        scenes[bvh_level+1] = inst->scene.get();
        intel_raytracing_acceleration_structure_t inst_accel = (intel_raytracing_acceleration_structure_t) inst->scene->getAccel();

        /* continue traversal */
        intel_ray_desc_t ray;
        ray.origin = O1;
        ray.direction = D1;
        ray.tmin = intel_get_ray_tmin(query,bvh_level);
        ray.tmax = 0.0f; // unused
        ray.mask = intel_get_ray_mask(query,bvh_level);
        ray.flags = intel_get_ray_flags(query,bvh_level);
        intel_ray_query_forward_ray(query, ray, inst_accel);
      }
    }
    
    intel_ray_query_start_traversal(query);
    intel_ray_query_sync(query);
  }

  /* committed hit */
  if (intel_has_committed_hit(query))
  {
    out.hit_type = TEST_COMMITTED_HIT;
    out.bvh_level = intel_get_hit_bvh_level( query, intel_hit_type_committed_hit );
    out.hit_candidate = intel_get_hit_candidate( query, intel_hit_type_committed_hit );
    out.t = intel_get_hit_distance(query, intel_hit_type_committed_hit);
    out.u = intel_get_hit_barycentrics(query, intel_hit_type_committed_hit).x;
    out.v = intel_get_hit_barycentrics(query, intel_hit_type_committed_hit).y;
    out.front_face = intel_get_hit_front_face( query, intel_hit_type_committed_hit );
    out.instUserID = intel_get_hit_instance_user_id( query, intel_hit_type_committed_hit );
    out.instID = intel_get_hit_instance_id( query, intel_hit_type_committed_hit );
    out.geomID = intel_get_hit_geometry_id( query, intel_hit_type_committed_hit );
    out.primID = intel_get_hit_primitive_id( query, intel_hit_type_committed_hit );

    out.v0 = sycl::float3(0.f,0.f,0.f);
    out.v1 = sycl::float3(0.f,0.f,0.f);
    out.v2 = sycl::float3(0.f,0.f,0.f);
    if (intel_get_hit_candidate( query, intel_hit_type_committed_hit ) == intel_candidate_type_triangle)
    {
      intel_float3 vertex_out[3];
      intel_get_hit_triangle_vertices(query, vertex_out, intel_hit_type_committed_hit);
      out.v0 = vertex_out[0];
      out.v1 = vertex_out[1];
      out.v2 = vertex_out[2];
    }

    /* return instance transformations */
    out.world_to_object = intel_get_hit_world_to_object(query,intel_hit_type_committed_hit);
    out.object_to_world = intel_get_hit_object_to_world(query,intel_hit_type_committed_hit);
  }

  /* miss */
  else {
    out.hit_type = TEST_MISS;
  }

  /* abandon ray query */
  intel_ray_query_abandon(query);
}

void buildTestExpectedInputAndOutput(std::shared_ptr<Scene> scene, size_t numTests, TestType test, TestInput* in, TestOutput* out_expected)
{
  std::vector<Hit> tri_map;
  tri_map.resize(numTests);
  std::vector<uint32_t> id_stack;
  Transform local_to_world;
  scene->buildTriMap(local_to_world,id_stack,-1,false,tri_map);
  
  TestHitType hit_type = TEST_MISS;
  switch (test) {
  case TestType::TRIANGLES_COMMITTED_HIT: hit_type = TEST_COMMITTED_HIT; break;
  case TestType::TRIANGLES_POTENTIAL_HIT: hit_type = TEST_POTENTIAL_HIT; break;
  case TestType::TRIANGLES_ANYHIT_SHADER_COMMIT: hit_type = TEST_COMMITTED_HIT; break;
  case TestType::TRIANGLES_ANYHIT_SHADER_REJECT: hit_type = TEST_MISS; break;
  case TestType::PROCEDURALS_COMMITTED_HIT: hit_type = TEST_COMMITTED_HIT; break;
  default: assert(false); break;
  };

  //for (size_t y=0; y<height; y++)
  {
    //for (size_t x=0; x<width; x++)
    {
      //for (size_t i=0; i<2; i++)
      for (size_t tid=0; tid<numTests; tid++)
      {
        //size_t tid = 2*(y*width+x)+i;
        assert(tid < numTests);

        Hit hit = tri_map[tid];
        const Triangle tri = hit.triangle.transform(hit.local_to_world);
        const sycl::float3 p = tri.sample(0.1f,0.6f);
        const Transform world_to_local = rcp(hit.local_to_world);
        
        in[tid].org = p + sycl::float3(0.f,0.f,-1.f);
        in[tid].dir = sycl::float3(0.f,0.f,1.f);
        in[tid].tnear = 0.0f;
        in[tid].tfar = 10000.0f;
        in[tid].mask = 0xFF;
        in[tid].flags = intel_ray_flags_none;

        // Ray data at level 0
        out_expected[tid].ray0_org = in[tid].org;
        out_expected[tid].ray0_dir = in[tid].dir;
        out_expected[tid].ray0_tnear = in[tid].tnear;
        out_expected[tid].ray0_mask = in[tid].mask;
        out_expected[tid].ray0_flags = in[tid].flags;
        
        // Ray data at hit bvh_level
        switch (test) {
        default: break;
        case TestType::TRIANGLES_POTENTIAL_HIT:
          out_expected[tid].rayN_org = xfmPoint (world_to_local,in[tid].org);
          out_expected[tid].rayN_dir = xfmVector(world_to_local,in[tid].dir);
          out_expected[tid].rayN_tnear = in[tid].tnear;
          out_expected[tid].rayN_mask = in[tid].mask;
          out_expected[tid].rayN_flags = in[tid].flags;
          break;
        }
                 
        // Hit data
        out_expected[tid].hit_type = hit_type;
        switch (test) {
        default: break;
        case TestType::TRIANGLES_COMMITTED_HIT:
        case TestType::TRIANGLES_POTENTIAL_HIT:
        case TestType::TRIANGLES_ANYHIT_SHADER_COMMIT:
        case TestType::PROCEDURALS_COMMITTED_HIT:

          if (hit.instID != -1)
            out_expected[tid].bvh_level = 1;
          else
            out_expected[tid].bvh_level = 0;
          
          if (hit.procedural_triangle)
            out_expected[tid].hit_candidate = intel_candidate_type_procedural;
          else
            out_expected[tid].hit_candidate = intel_candidate_type_triangle;
          
          out_expected[tid].t = 1.0f;
          out_expected[tid].u = 0.1f;
          out_expected[tid].v = 0.6f;
          out_expected[tid].front_face = 0;
          out_expected[tid].geomID = hit.geomID;
          out_expected[tid].primID = hit.primID;

          if (hit.procedural_instance) {
            out_expected[tid].instID = -1;
            out_expected[tid].instUserID = -1;
          }
          else {
            out_expected[tid].instID = hit.instID;
            out_expected[tid].instUserID = hit.instUserID;
          }
           
          if (hit.procedural_triangle) {
            out_expected[tid].v0 = sycl::float3(0.f,0.f,0.f);
            out_expected[tid].v1 = sycl::float3(0.f,0.f,0.f);
            out_expected[tid].v2 = sycl::float3(0.f,0.f,0.f);
          } else {
            out_expected[tid].v0 = hit.triangle.v0;
            out_expected[tid].v1 = hit.triangle.v1;
            out_expected[tid].v2 = hit.triangle.v2;
          }
          if (hit.procedural_instance) {
            out_expected[tid].world_to_object = Transform();
            out_expected[tid].object_to_world = Transform();
          } else {
            out_expected[tid].world_to_object = world_to_local;
            out_expected[tid].object_to_world = hit.local_to_world;
          }
          break;
        }
      }
    }
  }
}

uint32_t executeTest(sycl::device& device, sycl::queue& queue, sycl::context& context, InstancingType inst, TestType test)
{
  const int width = 128;
  const int height = 128;
  const size_t numTests = 2*width*height;

  bool opaque = true;
  bool procedural = false;
  switch (test) {
  case TestType::TRIANGLES_COMMITTED_HIT       : opaque = true;  procedural=false; break;
  case TestType::TRIANGLES_POTENTIAL_HIT       : opaque = false; procedural=false; break;
  case TestType::TRIANGLES_ANYHIT_SHADER_COMMIT: opaque = false; procedural=false; break;
  case TestType::TRIANGLES_ANYHIT_SHADER_REJECT: opaque = false; procedural=false; break;
  case TestType::PROCEDURALS_COMMITTED_HIT     : opaque = false; procedural=true;  break;
  default: assert(false); break;
  };

  //std::shared_ptr<Scene> scene = std::make_shared<Scene>(width,height,opaque,procedural);
  std::shared_ptr<Scene> scene(new Scene(width,height,opaque,procedural));
  scene->splitIntoGeometries(16);
  if (inst != InstancingType::NONE)
    scene->createInstances(scene->size(),3, inst == InstancingType::SW_INSTANCING);

  scene->addNullGeometries(16);

  scene->buildAccel(device,context,BuildMode::BUILD_EXPECTED_SIZE,false);

  /* calculate test input and expected output */
  TestInput* in = (TestInput*) sycl::aligned_alloc(64,numTests*sizeof(TestInput),device,context,sycl::usm::alloc::shared);
  memset(in, 0, numTests*sizeof(TestInput));
  TestOutput* out_test = (TestOutput*) sycl::aligned_alloc(64,numTests*sizeof(TestOutput),device,context,sycl::usm::alloc::shared);
  memset(out_test, 0, numTests*sizeof(TestOutput));
  TestOutput* out_expected = (TestOutput*) sycl::aligned_alloc(64,numTests*sizeof(TestOutput),device,context,sycl::usm::alloc::shared);
  memset(out_expected, 0, numTests*sizeof(TestOutput));

  buildTestExpectedInputAndOutput(scene,numTests,test,in,out_expected);
 
  /* execute test */
  intel_raytracing_acceleration_structure_t accel = (intel_raytracing_acceleration_structure_t) scene->getAccel();
  size_t scene_ptr = (size_t) scene.get();

  if (inst != InstancingType::SW_INSTANCING &&
      (test == TestType::TRIANGLES_COMMITTED_HIT || test == TestType::TRIANGLES_POTENTIAL_HIT))
  {
#if defined(ZE_RAYTRACING_RT_SIMULATION)
    tbb::parallel_for(size_t(0),numTests, [&](size_t i) {
      render(i,in[i],out_test[i],accel);
     });
#else
    queue.submit([&](sycl::handler& cgh) {
                   const sycl::range<1> range(numTests);
                   cgh.parallel_for(range, [=](sycl::item<1> item) {
                                             const uint32_t i = item.get_id(0);
                                             render(i,in[i],out_test[i],accel);
                                           });
                 });
    queue.wait_and_throw();
#endif
  }
  else
  {
#if defined(ZE_RAYTRACING_RT_SIMULATION)
    tbb::parallel_for(size_t(0),numTests, [&](size_t i) {
      render_loop(i,in[i],out_test[i],scene_ptr,accel,test);
     });
#else
    queue.submit([&](sycl::handler& cgh) {
                   const sycl::range<1> range(numTests);
                   cgh.parallel_for(range, [=](sycl::item<1> item) {
                                             const uint32_t i = item.get_id(0);
                                             render_loop(i,in[i],out_test[i],scene_ptr,accel,test);
                                           });
                 });
    queue.wait_and_throw();
#endif
  }
    
  /* verify result */
  uint32_t numErrors = 0;
  for (size_t tid=0; tid<numTests; tid++)
    compareTestOutput(tid,numErrors,out_test[tid],out_expected[tid]);

  sycl::free(in,context);
  sycl::free(out_test,context);
  sycl::free(out_expected,context);

  return numErrors;
}

uint32_t executeBuildTest(sycl::device& device, sycl::queue& queue, sycl::context& context, TestType test, BuildMode buildMode, uint32_t numPrimitives, int testID)
{
  const uint32_t width = 2*(uint32_t)ceilf(sqrtf(numPrimitives));
  std::shared_ptr<TriangleMesh> plane = createTrianglePlane(sycl::float3(0,0,0), sycl::float3(width,0,0), sycl::float3(0,width,0), width, width);
  if (test == TestType::BUILD_TEST_PROCEDURALS) plane->procedural = true;
  plane->selectRandom(numPrimitives);
  if (testID%2) plane->unshareVertices();
    
  std::shared_ptr<Scene> scene(new Scene);
  scene->add(plane);
  
  if (test == TestType::BUILD_TEST_PROCEDURALS) {
    if (testID%3==0)
      scene->splitIntoGeometries();
  }
  else if (test == TestType::BUILD_TEST_MIXED) {
    scene->splitIntoGeometries(std::max(1u,std::min(1024u,numPrimitives)));
    scene->mixTrianglesAndProcedurals();
    scene->createInstances(scene->size()/2);
  }
  else if (test == TestType::BUILD_TEST_INSTANCES) {
    scene->splitIntoGeometries(std::max(1u,std::min(1024u,numPrimitives)));
    scene->createInstances(scene->size());
  }

  scene->addNullGeometries(16);
  scene->buildAccel(device,context,buildMode,false);

  /* calculate test input and expected output */
  TestInput* in = (TestInput*) sycl::aligned_alloc(64,numPrimitives*sizeof(TestInput),device,context,sycl::usm::alloc::shared);
  memset(in, 0, numPrimitives*sizeof(TestInput));
  TestOutput* out_test = (TestOutput*) sycl::aligned_alloc(64,numPrimitives*sizeof(TestOutput),device,context,sycl::usm::alloc::shared);
  memset(out_test, 0, numPrimitives*sizeof(TestOutput));
  TestOutput* out_expected = (TestOutput*) sycl::aligned_alloc(64,numPrimitives*sizeof(TestOutput),device,context,sycl::usm::alloc::shared);
  memset(out_expected, 0, numPrimitives*sizeof(TestOutput));

  buildTestExpectedInputAndOutput(scene,numPrimitives,TestType::TRIANGLES_COMMITTED_HIT,in,out_expected);

  /* execute test */
  intel_raytracing_acceleration_structure_t accel = (intel_raytracing_acceleration_structure_t) scene->getAccel();
  size_t scene_ptr = (size_t) scene.get();

  if (numPrimitives)
  {
#if defined(ZE_RAYTRACING_RT_SIMULATION)
    tbb::parallel_for(size_t(0),size_t(numPrimitives), [&](size_t i) {
      render_loop(i,in[i],out_test[i],scene_ptr,accel,TestType::TRIANGLES_COMMITTED_HIT);
    });
#else
    queue.submit([&](sycl::handler& cgh) {
                   const sycl::range<1> range(numPrimitives);
                   cgh.parallel_for(range, [=](sycl::item<1> item) {
                                             const uint32_t i = item.get_id(0);
                                             render_loop(i,in[i],out_test[i],scene_ptr,accel,TestType::TRIANGLES_COMMITTED_HIT);
                                           });
                 });
    queue.wait_and_throw();
#endif
  }
    
  /* verify result */
  uint32_t numErrors = 0;
  for (size_t tid=0; tid<numPrimitives; tid++)
    compareTestOutput(tid,numErrors,out_test[tid],out_expected[tid]);
  
  sycl::free(in,context);
  sycl::free(out_test,context);
  sycl::free(out_expected,context);

  return numErrors;
}

uint32_t executeBuildTest(sycl::device& device, sycl::queue& queue, sycl::context& context, TestType test, BuildMode buildMode)
{
  uint32_t numErrors = 0;
  for (uint32_t i=0; i<128; i++) {
    const uint32_t numPrimitives = i>10 ? i*i : i;
    std::cout << "testing " << numPrimitives << " primitives" << std::endl;
    numErrors += executeBuildTest(device,queue,context,test,buildMode,numPrimitives,i);
  }
  return numErrors;
}

uint32_t executeBenchmark(sycl::device& device, sycl::queue& queue, sycl::context& context, TestType test)
{
  for (uint32_t i=0; i<=20; i++)
  {
    const uint32_t numPrimitives = 1<<i;

    switch (test) {
    default: break;
    case TestType::BENCHMARK_TRIANGLES  : std::cout << "benchmarking " << numPrimitives << " triangles: "; break;
    case TestType::BENCHMARK_PROCEDURALS: std::cout << "benchmarking " << numPrimitives << " procedurals: "; break;
    };

    const uint32_t width = 2*(uint32_t)ceilf(sqrtf(numPrimitives));
    std::shared_ptr<TriangleMesh> plane = createTrianglePlane(sycl::float3(0,0,0), sycl::float3(width,0,0), sycl::float3(0,width,0), width, width);
    if (test == TestType::BENCHMARK_PROCEDURALS) plane->procedural = true;
    plane->selectSequential(numPrimitives);
    
    std::shared_ptr<Scene> scene(new Scene);
    scene->add(plane);
    
    scene->buildAccel(device,context,BuildMode::BUILD_WORST_CASE_SIZE,true);
  }
  return 0;
}

enum Flags : uint32_t {
  FLAGS_NONE,
  DEPTH_TEST_LESS_EQUAL = 1 << 0  // when set we use <= for depth test, otherwise <
};

struct DispatchGlobals
{
  uint64_t rtMemBasePtr;               // base address of the allocated stack memory
  uint64_t callStackHandlerKSP;             // this is the KSP of the continuation handler that is invoked by BTD when the read KSP is 0
  uint32_t asyncStackSize;             // async-RT stack size in 64 byte blocks
  uint32_t numDSSRTStacks : 16;        // number of stacks per DSS
  uint32_t syncRayQueryCount : 4;      // number of ray queries in the sync-RT stack: 0-15 mapped to: 1-16
  unsigned _reserved_mbz : 12;
  uint32_t maxBVHLevels;               // the maximal number of supported instancing levels (0->8, 1->1, 2->2, ...)
  Flags flags;                         // per context control flags
};

void* allocDispatchGlobals(sycl::device device, sycl::context context)
{
  size_t maxBVHLevels = 2; //RTC_MAX_INSTANCE_LEVEL_COUNT+1;
  
  size_t rtstack_bytes = (64+maxBVHLevels*(64+32)+63)&-64;
  size_t num_rtstacks = 1<<17; // this is sufficiently large also for PVC
  size_t dispatchGlobalSize = 128+num_rtstacks*rtstack_bytes;
  
  void* dispatchGlobalsPtr = alloc_accel_buffer(dispatchGlobalSize,device,context);
  memset(dispatchGlobalsPtr, 0, dispatchGlobalSize);

  DispatchGlobals* dg = (DispatchGlobals*) dispatchGlobalsPtr;
  dg->rtMemBasePtr = (uint64_t) dispatchGlobalsPtr + dispatchGlobalSize;
  dg->callStackHandlerKSP = 0;
  dg->asyncStackSize = 0;
  dg->numDSSRTStacks = 0;
  dg->syncRayQueryCount = 0;
  dg->_reserved_mbz = 0;
  dg->maxBVHLevels = maxBVHLevels;
  dg->flags = DEPTH_TEST_LESS_EQUAL;
  
  return dispatchGlobalsPtr;
}

int main(int argc, char* argv[]) try
{
  TestType test = TestType::TRIANGLES_COMMITTED_HIT;
  InstancingType inst = InstancingType::NONE;
  BuildMode buildMode = BuildMode::BUILD_EXPECTED_SIZE;

#if defined(EMBREE_SYCL_L0_RTAS_BUILDER)
  ZeWrapper::RTAS_BUILD_MODE rtas_build_mode = ZeWrapper::RTAS_BUILD_MODE::LEVEL_ZERO;
#else
  ZeWrapper::RTAS_BUILD_MODE rtas_build_mode = ZeWrapper::RTAS_BUILD_MODE::INTERNAL;
#endif
  
  bool jit_cache = false;
  uint32_t numThreads = tbb::this_task_arena::max_concurrency();
  
  /* command line parsing */
  if (argc == 1) {
    std::cout << "ERROR: no test specified" << std::endl;
    return 1;
  }

  /* parse all command line options */
  for (size_t i=1; i<argc; i++)
  {
    if (strcmp(argv[i], "--internal-rtas-builder") == 0) {
      rtas_build_mode = ZeWrapper::RTAS_BUILD_MODE::INTERNAL;
    }
    else if (strcmp(argv[i], "--level-zero-rtas-builder") == 0) {
      rtas_build_mode = ZeWrapper::RTAS_BUILD_MODE::LEVEL_ZERO;
    }
    else if (strcmp(argv[i], "--default-rtas-builder") == 0) {
      rtas_build_mode = ZeWrapper::RTAS_BUILD_MODE::AUTO;
    }
    else if (strcmp(argv[i], "--triangles-committed-hit") == 0) {
      test = TestType::TRIANGLES_COMMITTED_HIT;
    }
    else if (strcmp(argv[i], "--triangles-potential-hit") == 0) {
      test = TestType::TRIANGLES_POTENTIAL_HIT;
    }
    else if (strcmp(argv[i], "--triangles-anyhit-shader-commit") == 0) {
      test = TestType::TRIANGLES_ANYHIT_SHADER_COMMIT;
    }
    else if (strcmp(argv[i], "--triangles-anyhit-shader-reject") == 0) {
      test = TestType::TRIANGLES_ANYHIT_SHADER_REJECT;
    }
    else if (strcmp(argv[i], "--procedurals-committed-hit") == 0) {
      test = TestType::PROCEDURALS_COMMITTED_HIT;
    }
    else if (strcmp(argv[i], "--build_test_triangles") == 0) {
      test = TestType::BUILD_TEST_TRIANGLES;
    }
    else if (strcmp(argv[i], "--build_test_procedurals") == 0) {
      test = TestType::BUILD_TEST_PROCEDURALS;
    }
    else if (strcmp(argv[i], "--build_test_instances") == 0) {
      test = TestType::BUILD_TEST_INSTANCES;
    }
    else if (strcmp(argv[i], "--build_test_mixed") == 0) {
      test = TestType::BUILD_TEST_MIXED;
    }
    else if (strcmp(argv[i], "--benchmark_triangles") == 0) {
      test = TestType::BENCHMARK_TRIANGLES;
    }
    else if (strcmp(argv[i], "--benchmark_procedurals") == 0) {
      test = TestType::BENCHMARK_PROCEDURALS;
    }
    else if (strcmp(argv[i], "--no-instancing") == 0) {
      inst = InstancingType::NONE;
    }
    else if (strcmp(argv[i], "--hw-instancing") == 0) {
      inst = InstancingType::HW_INSTANCING;
    }
    else if (strcmp(argv[i], "--sw-instancing") == 0) {
      inst = InstancingType::SW_INSTANCING;
    }
    else if (strcmp(argv[i], "--build_mode_worst_case") == 0) {
      buildMode = BuildMode::BUILD_WORST_CASE_SIZE;
    }
    else if (strcmp(argv[i], "--build_mode_expected") == 0) {
      buildMode = BuildMode::BUILD_EXPECTED_SIZE;
    }
    else if (strcmp(argv[i], "--jit-cache") == 0) {
      if (++i >= argc) throw std::runtime_error("Error: --jit-cache <int>: syntax error");
      jit_cache = atoi(argv[i]);
    }
    else if (strcmp(argv[i], "--threads") == 0) {
      if (++i >= argc) throw std::runtime_error("Error: --threads <int>: syntax error");
      numThreads = atoi(argv[i]);
    }
    else {
      std::cout << "ERROR: invalid command line option " << argv[i] << std::endl;
      return 1;
    }
  }

  if (jit_cache)
    std::cout << "WARNING: JIT caching is not supported!" << std::endl;

  if (ZeWrapper::init() != ZE_RESULT_SUCCESS) {
    std::cerr << "ZeWrapper not successfully initialized" << std::endl;
    return 1;
  }

#if defined(ZE_RAYTRACING_RT_SIMULATION)
  RTCore::Init();
  RTCore::SetXeVersion((RTCore::XeVersion)ZE_RAYTRACING_DEVICE);
#endif

#if TBB_INTERFACE_VERSION >= 11005
  tbb::global_control tbb_threads(tbb::global_control::max_allowed_parallelism,numThreads);
#else
  tbb::task_scheduler_init tbb_threads(tbb::task_scheduler_init::deferred);
  tbb_threads.initialize(int(numThreads));
#endif

  /* initialize SYCL device */
  device = sycl::device(sycl::gpu_selector_v);
  sycl::queue queue = sycl::queue(device,exception_handler);
  context = queue.get_context();

#if defined(EMBREE_SYCL_ALLOC_DISPATCH_GLOBALS)
  dispatchGlobalsPtr = allocDispatchGlobals(device,context);
#endif

  /* execute test */
  RandomSampler_init(rng,0x56FE238A);

  ze_result_t result = ZE_RESULT_SUCCESS;
  sycl::platform platform = device.get_platform();
  ze_driver_handle_t hDriver = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(platform);

  /* enable RTAS extension only when enabled */
  if (rtas_build_mode == ZeWrapper::RTAS_BUILD_MODE::AUTO)
  {
    uint32_t count = 0;
    std::vector<ze_driver_extension_properties_t> extensions;
    result = ZeWrapper::zeDriverGetExtensionProperties(hDriver,&count,extensions.data());
    if (result != ZE_RESULT_SUCCESS)
      throw std::runtime_error("zeDriverGetExtensionProperties failed");
    
    extensions.resize(count);
    result = ZeWrapper::zeDriverGetExtensionProperties(hDriver,&count,extensions.data());
    if (result != ZE_RESULT_SUCCESS)
      throw std::runtime_error("zeDriverGetExtensionProperties failed");
    
    bool ze_rtas_builder = false;
    for (uint32_t i=0; i<extensions.size(); i++)
    {
      if (strncmp("ZE_experimental_rtas_builder",extensions[i].name,sizeof(extensions[i].name)) == 0)
        ze_rtas_builder = true;
    }

    if (ze_rtas_builder)
      result = ZeWrapper::initRTASBuilder(hDriver,ZeWrapper::RTAS_BUILD_MODE::AUTO);
    else
      result = ZeWrapper::initRTASBuilder(hDriver,ZeWrapper::RTAS_BUILD_MODE::INTERNAL);
  }
  else
    result = ZeWrapper::initRTASBuilder(hDriver,rtas_build_mode);

  if (result == ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE)
    throw std::runtime_error("cannot load ZE_experimental_rtas_builder extension");
  
  if (result != ZE_RESULT_SUCCESS)
    throw std::runtime_error("cannot initialize ZE_experimental_rtas_builder extension");
  
  if (ZeWrapper::rtas_builder == ZeWrapper::INTERNAL)
    std::cout << "using internal RTAS builder" << std::endl;
  else
    std::cout << "using Level Zero RTAS builder" << std::endl;

    
  /* create L0 builder object */
  ze_rtas_builder_exp_desc_t builderDesc = { ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC };
  ze_result_t err = ZeWrapper::zeRTASBuilderCreateExp(hDriver, &builderDesc, &hBuilder);
  if (err != ZE_RESULT_SUCCESS)
    throw std::runtime_error("ze_rtas_builder creation failed");

  err = ZeWrapper::zeRTASParallelOperationCreateExp(hDriver,&parallelOperation);
  if (err != ZE_RESULT_SUCCESS)
    throw std::runtime_error("parallel operation creation failed");
  
  uint32_t numErrors = 0;
  if (test >= TestType::BENCHMARK_TRIANGLES)
    numErrors = executeBenchmark(device,queue,context,test);
  else if (test >= TestType::BUILD_TEST_TRIANGLES)
    numErrors = executeBuildTest(device,queue,context,test,buildMode);
  else
    numErrors = executeTest(device,queue,context,inst,test);

  err = ZeWrapper::zeRTASParallelOperationDestroyExp(parallelOperation);
  if (err != ZE_RESULT_SUCCESS)
    throw std::runtime_error("parallel operation destruction failed");

  /* destroy rtas builder again */
  err = ZeWrapper::zeRTASBuilderDestroyExp(hBuilder);
  if (err != ZE_RESULT_SUCCESS)
    throw std::runtime_error("ze_rtas_builder destruction failed");
  
#if defined(EMBREE_SYCL_ALLOC_DISPATCH_GLOBALS)
  free_accel_buffer(dispatchGlobalsPtr, context);
#endif

#if defined(ZE_RAYTRACING_RT_SIMULATION)
  RTCore::Cleanup();
#endif
  
  return numErrors ? 1 : 0;
}
catch (std::runtime_error e) {
  std::cerr << "std::runtime_error: " << e.what() << std::endl;
  return 1;
}

#pragma clang diagnostic pop
