// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore_scene.h"

RTC_NAMESPACE_BEGIN
  
/* Opaque BVH type */
typedef struct RTCBVHTy* RTCBVH;

/* Input build primitives for the builder */
struct RTC_ALIGN(32) RTCBuildPrimitive
{
  float lower_x, lower_y, lower_z; 
  unsigned int geomID;
  float upper_x, upper_y, upper_z;
  unsigned int primID;
};

/* Opaque thread local allocator type */
typedef struct RTCThreadLocalAllocatorTy* RTCThreadLocalAllocator;

/* Callback to create a node */
typedef void* (*RTCCreateNodeFunction) (RTCThreadLocalAllocator allocator, unsigned int childCount, void* userPtr);

/* Callback to set the pointer to all children */
typedef void (*RTCSetNodeChildrenFunction) (void* nodePtr, void** children, unsigned int childCount, void* userPtr);

/* Callback to set the bounds of all children */
typedef void (*RTCSetNodeBoundsFunction) (void* nodePtr, const struct RTCBounds** bounds, unsigned int childCount, void* userPtr);

/* Callback to create a leaf node */
typedef void* (*RTCCreateLeafFunction) (RTCThreadLocalAllocator allocator, const struct RTCBuildPrimitive* primitives, size_t primitiveCount, void* userPtr);

/* Callback to split a build primitive */
typedef void (*RTCSplitPrimitiveFunction) (const struct RTCBuildPrimitive* primitive, unsigned int dimension, float position, struct RTCBounds* leftBounds, struct RTCBounds* rightBounds, void* userPtr);

/* Build flags */
enum RTCBuildFlags
{
  RTC_BUILD_FLAG_NONE    = 0,
  RTC_BUILD_FLAG_DYNAMIC = (1 << 0),
};

enum RTCBuildConstants
{
  RTC_BUILD_MAX_PRIMITIVES_PER_LEAF = 32
};

/* Input for builders */
struct RTCBuildArguments
{
  size_t byteSize;
  
  enum RTCBuildQuality buildQuality;
  enum RTCBuildFlags buildFlags;
  unsigned int maxBranchingFactor;
  unsigned int maxDepth;
  unsigned int sahBlockSize;
  unsigned int minLeafSize;
  unsigned int maxLeafSize;
  float traversalCost;
  float intersectionCost;
  
  RTCBVH bvh;
  struct RTCBuildPrimitive* primitives;
  size_t primitiveCount;
  size_t primitiveArrayCapacity;
  
  RTCCreateNodeFunction createNode;
  RTCSetNodeChildrenFunction setNodeChildren;
  RTCSetNodeBoundsFunction setNodeBounds;
  RTCCreateLeafFunction createLeaf;
  RTCSplitPrimitiveFunction splitPrimitive;
  RTCProgressMonitorFunction buildProgress;
  void* userPtr;
};

/* Returns the default build settings.  */
RTC_FORCEINLINE struct RTCBuildArguments rtcDefaultBuildArguments()
{
  struct RTCBuildArguments args;
  args.byteSize = sizeof(args);
  args.buildQuality = RTC_BUILD_QUALITY_MEDIUM;
  args.buildFlags = RTC_BUILD_FLAG_NONE;
  args.maxBranchingFactor = 2;
  args.maxDepth = 32;
  args.sahBlockSize = 1;
  args.minLeafSize = 1;
  args.maxLeafSize = RTC_BUILD_MAX_PRIMITIVES_PER_LEAF;
  args.traversalCost = 1.0f;
  args.intersectionCost = 1.0f;
  args.bvh = NULL;
  args.primitives = NULL;
  args.primitiveCount = 0;
  args.primitiveArrayCapacity = 0;
  args.createNode = NULL;
  args.setNodeChildren = NULL;
  args.setNodeBounds = NULL;
  args.createLeaf = NULL;
  args.splitPrimitive = NULL;
  args.buildProgress = NULL;
  args.userPtr = NULL;
  return args;
}

/* Creates a new BVH. */
RTC_API RTCBVH rtcNewBVH(RTCDevice device);

/* Builds a BVH. */
RTC_API void* rtcBuildBVH(const struct RTCBuildArguments* args);

/* Allocates memory using the thread local allocator. */
RTC_API void* rtcThreadLocalAlloc(RTCThreadLocalAllocator allocator, size_t bytes, size_t align);

/* Retains the BVH (increments reference count). */
RTC_API void rtcRetainBVH(RTCBVH bvh);

/* Releases the BVH (decrements reference count). */
RTC_API void rtcReleaseBVH(RTCBVH bvh);

RTC_NAMESPACE_END

