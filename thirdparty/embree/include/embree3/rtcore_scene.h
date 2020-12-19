// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore_device.h"

RTC_NAMESPACE_BEGIN
  
/* Forward declarations for ray structures */
struct RTCRayHit;
struct RTCRayHit4;
struct RTCRayHit8;
struct RTCRayHit16;
struct RTCRayHitNp;

/* Scene flags */
enum RTCSceneFlags
{
  RTC_SCENE_FLAG_NONE                    = 0,
  RTC_SCENE_FLAG_DYNAMIC                 = (1 << 0),
  RTC_SCENE_FLAG_COMPACT                 = (1 << 1),
  RTC_SCENE_FLAG_ROBUST                  = (1 << 2),
  RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION = (1 << 3)
};

/* Creates a new scene. */
RTC_API RTCScene rtcNewScene(RTCDevice device);

/* Returns the device the scene got created in. The reference count of
 * the device is incremented by this function. */
RTC_API RTCDevice rtcGetSceneDevice(RTCScene hscene);
   
/* Retains the scene (increments the reference count). */
RTC_API void rtcRetainScene(RTCScene scene);

/* Releases the scene (decrements the reference count). */
RTC_API void rtcReleaseScene(RTCScene scene);


/* Attaches the geometry to a scene. */
RTC_API unsigned int rtcAttachGeometry(RTCScene scene, RTCGeometry geometry);

/* Attaches the geometry to a scene using the specified geometry ID. */
RTC_API void rtcAttachGeometryByID(RTCScene scene, RTCGeometry geometry, unsigned int geomID);

/* Detaches the geometry from the scene. */
RTC_API void rtcDetachGeometry(RTCScene scene, unsigned int geomID);

/* Gets a geometry handle from the scene. */
RTC_API RTCGeometry rtcGetGeometry(RTCScene scene, unsigned int geomID);


/* Commits the scene. */
RTC_API void rtcCommitScene(RTCScene scene);

/* Commits the scene from multiple threads. */
RTC_API void rtcJoinCommitScene(RTCScene scene);


/* Progress monitor callback function */
typedef bool (*RTCProgressMonitorFunction)(void* ptr, double n);

/* Sets the progress monitor callback function of the scene. */
RTC_API void rtcSetSceneProgressMonitorFunction(RTCScene scene, RTCProgressMonitorFunction progress, void* ptr);

/* Sets the build quality of the scene. */
RTC_API void rtcSetSceneBuildQuality(RTCScene scene, enum RTCBuildQuality quality);

/* Sets the scene flags. */
RTC_API void rtcSetSceneFlags(RTCScene scene, enum RTCSceneFlags flags);

/* Returns the scene flags. */
RTC_API enum RTCSceneFlags rtcGetSceneFlags(RTCScene scene);

/* Returns the axis-aligned bounds of the scene. */
RTC_API void rtcGetSceneBounds(RTCScene scene, struct RTCBounds* bounds_o);

/* Returns the linear axis-aligned bounds of the scene. */
RTC_API void rtcGetSceneLinearBounds(RTCScene scene, struct RTCLinearBounds* bounds_o);


/* Perform a closest point query of the scene. */
RTC_API bool rtcPointQuery(RTCScene scene, struct RTCPointQuery* query, struct RTCPointQueryContext* context, RTCPointQueryFunction queryFunc, void* userPtr);

/* Perform a closest point query with a packet of 4 points with the scene. */
RTC_API bool rtcPointQuery4(const int* valid, RTCScene scene, struct RTCPointQuery4* query, struct RTCPointQueryContext* context, RTCPointQueryFunction queryFunc, void** userPtr);

/* Perform a closest point query with a packet of 4 points with the scene. */
RTC_API bool rtcPointQuery8(const int* valid, RTCScene scene, struct RTCPointQuery8* query, struct RTCPointQueryContext* context, RTCPointQueryFunction queryFunc, void** userPtr);

/* Perform a closest point query with a packet of 4 points with the scene. */
RTC_API bool rtcPointQuery16(const int* valid, RTCScene scene, struct RTCPointQuery16* query, struct RTCPointQueryContext* context, RTCPointQueryFunction queryFunc, void** userPtr);

/* Intersects a single ray with the scene. */
RTC_API void rtcIntersect1(RTCScene scene, struct RTCIntersectContext* context, struct RTCRayHit* rayhit);

/* Intersects a packet of 4 rays with the scene. */
RTC_API void rtcIntersect4(const int* valid, RTCScene scene, struct RTCIntersectContext* context, struct RTCRayHit4* rayhit);

/* Intersects a packet of 8 rays with the scene. */
RTC_API void rtcIntersect8(const int* valid, RTCScene scene, struct RTCIntersectContext* context, struct RTCRayHit8* rayhit);

/* Intersects a packet of 16 rays with the scene. */
RTC_API void rtcIntersect16(const int* valid, RTCScene scene, struct RTCIntersectContext* context, struct RTCRayHit16* rayhit);

/* Intersects a stream of M rays with the scene. */
RTC_API void rtcIntersect1M(RTCScene scene, struct RTCIntersectContext* context, struct RTCRayHit* rayhit, unsigned int M, size_t byteStride);

/* Intersects a stream of pointers to M rays with the scene. */
RTC_API void rtcIntersect1Mp(RTCScene scene, struct RTCIntersectContext* context, struct RTCRayHit** rayhit, unsigned int M);

/* Intersects a stream of M ray packets of size N in SOA format with the scene. */
RTC_API void rtcIntersectNM(RTCScene scene, struct RTCIntersectContext* context, struct RTCRayHitN* rayhit, unsigned int N, unsigned int M, size_t byteStride);

/* Intersects a stream of M ray packets of size N in SOA format with the scene. */
RTC_API void rtcIntersectNp(RTCScene scene, struct RTCIntersectContext* context, const struct RTCRayHitNp* rayhit, unsigned int N);

/* Tests a single ray for occlusion with the scene. */
RTC_API void rtcOccluded1(RTCScene scene, struct RTCIntersectContext* context, struct RTCRay* ray);

/* Tests a packet of 4 rays for occlusion occluded with the scene. */
RTC_API void rtcOccluded4(const int* valid, RTCScene scene, struct RTCIntersectContext* context, struct RTCRay4* ray);

/* Tests a packet of 8 rays for occlusion with the scene. */
RTC_API void rtcOccluded8(const int* valid, RTCScene scene, struct RTCIntersectContext* context, struct RTCRay8* ray);

/* Tests a packet of 16 rays for occlusion with the scene. */
RTC_API void rtcOccluded16(const int* valid, RTCScene scene, struct RTCIntersectContext* context, struct RTCRay16* ray);

/* Tests a stream of M rays for occlusion with the scene. */
RTC_API void rtcOccluded1M(RTCScene scene, struct RTCIntersectContext* context, struct RTCRay* ray, unsigned int M, size_t byteStride);

/* Tests a stream of pointers to M rays for occlusion with the scene. */
RTC_API void rtcOccluded1Mp(RTCScene scene, struct RTCIntersectContext* context, struct RTCRay** ray, unsigned int M);

/* Tests a stream of M ray packets of size N in SOA format for occlusion with the scene. */
RTC_API void rtcOccludedNM(RTCScene scene, struct RTCIntersectContext* context, struct RTCRayN* ray, unsigned int N, unsigned int M, size_t byteStride);

/* Tests a stream of M ray packets of size N in SOA format for occlusion with the scene. */
RTC_API void rtcOccludedNp(RTCScene scene, struct RTCIntersectContext* context, const struct RTCRayNp* ray, unsigned int N);

/*! collision callback */
struct RTCCollision { unsigned int geomID0; unsigned int primID0; unsigned int geomID1; unsigned int primID1; };
typedef void (*RTCCollideFunc) (void* userPtr, struct RTCCollision* collisions, unsigned int num_collisions);

/*! Performs collision detection of two scenes */
RTC_API void rtcCollide (RTCScene scene0, RTCScene scene1, RTCCollideFunc callback, void* userPtr);
 
#if defined(__cplusplus)

/* Helper for easily combining scene flags */
inline RTCSceneFlags operator|(RTCSceneFlags a, RTCSceneFlags b) {
  return (RTCSceneFlags)((size_t)a | (size_t)b);
}

#endif

RTC_NAMESPACE_END

