// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../include/embree4/rtcore_config.h"

// #cmakedefine EMBREE_RAY_MASK
// #cmakedefine EMBREE_STAT_COUNTERS
// #cmakedefine EMBREE_BACKFACE_CULLING
// #cmakedefine EMBREE_BACKFACE_CULLING_CURVES
// #cmakedefine EMBREE_BACKFACE_CULLING_SPHERES
#define EMBREE_FILTER_FUNCTION
// #cmakedefine EMBREE_IGNORE_INVALID_RAYS
#define EMBREE_GEOMETRY_TRIANGLE
// #cmakedefine EMBREE_GEOMETRY_QUAD
// #cmakedefine EMBREE_GEOMETRY_CURVE
// #cmakedefine EMBREE_GEOMETRY_SUBDIVISION
// #cmakedefine EMBREE_GEOMETRY_USER
// #cmakedefine EMBREE_GEOMETRY_INSTANCE
// EMBREE_GEOMETRY_INSTANCE_ARRAY is defined in rtcore_config.h
// #cmakedefine EMBREE_GEOMETRY_GRID
// #cmakedefine EMBREE_GEOMETRY_POINT
#define EMBREE_RAY_PACKETS
// #cmakedefine EMBREE_COMPACT_POLYS

#define EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR 2.0
#define EMBREE_DISC_POINT_SELF_INTERSECTION_AVOIDANCE

#if defined(EMBREE_GEOMETRY_TRIANGLE)
  #define IF_ENABLED_TRIS(x) x
#else
  #define IF_ENABLED_TRIS(x)
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
  #define IF_ENABLED_QUADS(x) x
#else
  #define IF_ENABLED_QUADS(x)
#endif

#if defined(EMBREE_GEOMETRY_CURVE) || defined(EMBREE_GEOMETRY_POINT)
  #define IF_ENABLED_CURVES_OR_POINTS(x) x
#else
  #define IF_ENABLED_CURVES_OR_POINTS(x)
#endif

#if defined(EMBREE_GEOMETRY_CURVE)
  #define IF_ENABLED_CURVES(x) x
#else
  #define IF_ENABLED_CURVES(x)
#endif

#if defined(EMBREE_GEOMETRY_POINT)
  #define IF_ENABLED_POINTS(x) x
#else
  #define IF_ENABLED_POINTS(x)
#endif

#if defined(EMBREE_GEOMETRY_SUBDIVISION)
  #define IF_ENABLED_SUBDIV(x) x
#else
  #define IF_ENABLED_SUBDIV(x)
#endif

#if defined(EMBREE_GEOMETRY_USER)
  #define IF_ENABLED_USER(x) x
#else
  #define IF_ENABLED_USER(x)
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE)
  #define IF_ENABLED_INSTANCE(x) x
#else
  #define IF_ENABLED_INSTANCE(x)
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE_ARRAY)
  #define IF_ENABLED_INSTANCE_ARRAY(x) x
#else
  #define IF_ENABLED_INSTANCE_ARRAY(x)
#endif

#if defined(EMBREE_GEOMETRY_GRID)
  #define IF_ENABLED_GRIDS(x) x
#else
  #define IF_ENABLED_GRIDS(x)
#endif




