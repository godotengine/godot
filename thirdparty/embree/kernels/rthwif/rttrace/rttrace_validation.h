// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-W#pragma-messages"

#include <sycl/sycl.hpp>

#pragma clang diagnostic pop

enum intel_ray_flags_t
{
  intel_ray_flags_none = 0x00,
  intel_ray_flags_force_opaque = 0x01,                      // forces geometry to be opaque (no anyhit shader invokation)
  intel_ray_flags_force_non_opaque = 0x02,                  // forces geometry to be non-opqaue (invoke anyhit shader)
  intel_ray_flags_accept_first_hit_and_end_search = 0x04,   // terminates traversal on the first hit found (shadow rays)
  intel_ray_flags_skip_closest_hit_shader = 0x08,           // skip execution of the closest hit shader
  intel_ray_flags_cull_back_facing_triangles = 0x10,        // back facing triangles to not produce a hit
  intel_ray_flags_cull_front_facing_triangles = 0x20,       // front facing triangles do not produce a hit
  intel_ray_flags_cull_opaque = 0x40,                       // opaque geometry does not produce a hit
  intel_ray_flags_cull_non_opaque = 0x80,                   // non-opaque geometry does not produce a hit
  intel_ray_flags_skip_triangles = 0x100,                   // treat all triangle intersections as misses.
  intel_ray_flags_skip_procedural_primitives = 0x200,       // skip execution of intersection shaders
};

enum intel_hit_type_t
{
  intel_hit_type_committed_hit = 0,
  intel_hit_type_potential_hit = 1,
};

enum intel_raytracing_ext_flag_t
{
  intel_raytracing_ext_flag_ray_query   = 1 << 0,        // true if ray queries are supported
};

struct intel_float2
{
  float x, y;

  intel_float2() {}

  intel_float2(float x, float y)
    : x(x), y(y) {}
  
  intel_float2(sycl::float2 v)
    : x(v.x()), y(v.y()) {}

  operator sycl::float2() {
    return sycl::float2(x,y);
  }
};

struct intel_float3
{
  float x, y, z;

  intel_float3() {}

  intel_float3(float x, float y, float z)
    : x(x), y(y), z(z) {}

  intel_float3(sycl::float3 v)
    : x(v.x()), y(v.y()), z(v.z()) {}

  operator sycl::float3() {
    return sycl::float3(x,y,z);
  }
};

struct intel_float4x3 {
  intel_float3 vx, vy, vz, p;
};

struct intel_ray_desc_t
{
  intel_float3 origin;
  intel_float3 direction;
  float tmin;
  float tmax;
  unsigned int mask;
  intel_ray_flags_t flags;
};

#include "rttrace_internal.h"

// opaque types
struct intel_ray_query_t {
  void* opaque0; void* opaque1; void* opaque2; uint32_t ctrl; uint32_t bvh_level;
  MemHit& hit(intel_hit_type_t ty) {
    struct RTStack* rtStack = (struct RTStack*) opaque2;
    return rtStack->hit[ty];
  }
};
typedef __attribute__((opencl_global )) struct intel_raytracing_acceleration_structure_opaque_t* intel_raytracing_acceleration_structure_t;

// check supported ray tracing features
SYCL_EXTERNAL intel_raytracing_ext_flag_t intel_get_raytracing_ext_flag();

// initializes a ray query
SYCL_EXTERNAL intel_ray_query_t intel_ray_query_init(
  intel_ray_desc_t ray,
  intel_raytracing_acceleration_structure_t accel
);

// setup for instance traversal using a transformed ray and bottom-level AS
SYCL_EXTERNAL void intel_ray_query_forward_ray(
  intel_ray_query_t& query,
  intel_ray_desc_t ray,
  intel_raytracing_acceleration_structure_t accel
);

// commit the potential hit
SYCL_EXTERNAL void intel_ray_query_commit_potential_hit(
  intel_ray_query_t& query
);

// commit the potential hit and override hit distance and UVs
SYCL_EXTERNAL void intel_ray_query_commit_potential_hit_override(
  intel_ray_query_t& query,
  float override_hit_distance,
  intel_float2 override_uv
);

// start traversal of a ray query
SYCL_EXTERNAL void intel_ray_query_start_traversal( intel_ray_query_t& query );

// synchronize rayquery execution.  If a ray was dispatched, 
//  This must be called prior to calling any of the accessors below.
SYCL_EXTERNAL void intel_ray_query_sync( intel_ray_query_t& query );

// signal that a ray query will not be used further.  This is the moral equaivalent of a delete
// this function does an implicit sync
SYCL_EXTERNAL void intel_ray_query_abandon( intel_ray_query_t& query );

// read hit information during shader execution
SYCL_EXTERNAL unsigned int intel_get_hit_bvh_level( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL float intel_get_hit_distance( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL intel_float2 intel_get_hit_barycentrics( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL bool intel_get_hit_front_face( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL unsigned int intel_get_hit_geometry_id(intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL unsigned int intel_get_hit_primitive_id( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL unsigned int intel_get_hit_triangle_primitive_id( intel_ray_query_t& query, intel_hit_type_t hit_type );  // fast path for quad leaves
SYCL_EXTERNAL unsigned int intel_get_hit_procedural_primitive_id( intel_ray_query_t& query, intel_hit_type_t hit_type ); // fast path for procedural leaves
SYCL_EXTERNAL unsigned int intel_get_hit_instance_id( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL unsigned int intel_get_hit_instance_user_id( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL intel_float4x3 intel_get_hit_world_to_object( intel_ray_query_t& query, intel_hit_type_t hit_type );
SYCL_EXTERNAL intel_float4x3 intel_get_hit_object_to_world( intel_ray_query_t& query, intel_hit_type_t hit_type );

// fetch triangle vertices for a hit
SYCL_EXTERNAL void intel_get_hit_triangle_vertices( intel_ray_query_t& query, intel_float3 vertices_out[3], intel_hit_type_t hit_type );

// Read ray-data. This is used to read transformed rays produced by HW instancing pipeline
// during any-hit or intersection shader execution.
SYCL_EXTERNAL intel_float3 intel_get_ray_origin( intel_ray_query_t& query, unsigned int bvh_level );
SYCL_EXTERNAL intel_float3 intel_get_ray_direction( intel_ray_query_t& query, unsigned int bvh_level );
SYCL_EXTERNAL float intel_get_ray_tmin( intel_ray_query_t& query, unsigned int bvh_level );
SYCL_EXTERNAL intel_ray_flags_t intel_get_ray_flags( intel_ray_query_t& query, unsigned int bvh_level );
SYCL_EXTERNAL unsigned int intel_get_ray_mask( intel_ray_query_t& query, unsigned int bvh_level );

// if traversal returns one can test if a triangle or procedural is hit
enum intel_candidate_type_t
{
  intel_candidate_type_triangle,
  intel_candidate_type_procedural
};

SYCL_EXTERNAL intel_candidate_type_t intel_get_hit_candidate( intel_ray_query_t& query, intel_hit_type_t hit_type );

// test whether traversal has terminated.  If false, the ray has reached
//  a procedural leaf or a non-opaque triangle leaf, and requires shader processing
SYCL_EXTERNAL bool intel_is_traversal_done( intel_ray_query_t& query );

// if traversal is done one can test for the presence of a committed hit to either invoke miss or closest hit shader
SYCL_EXTERNAL bool intel_has_committed_hit( intel_ray_query_t& query );
