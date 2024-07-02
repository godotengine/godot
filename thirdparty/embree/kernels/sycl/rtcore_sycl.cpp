// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define RTC_EXPORT_API

#include "../common/default.h"
#include "../common/device.h"
#include "../common/scene.h"
#include "../common/context.h"
#include "../geometry/filter.h"
#include "rthwif_embree.h"
using namespace embree;

#define DBG(x)

RTC_NAMESPACE_BEGIN;

RTC_API_EXTERN_C RTCDevice rtcNewSYCLDeviceInternal(sycl::context context, const char* config);

void use_rthwif_embree();
void use_rthwif_production();

/* we define rtcNewSYCLDevice in libembree_sycl.a to avoid drop of rtcore_sycl.o during linking of libembree_sycl.a file */
RTC_API_EXTERN_C RTCDevice rtcNewSYCLDevice(sycl::context context, const char* config)
{
  use_rthwif_embree();     // to avoid drop of rthwif_embree.o during linking of libembree_sycl.a file
#if defined(EMBREE_SYCL_RT_VALIDATION_API)
  use_rthwif_production(); // to avoid drop of rthwif_production.o during linking of libembree_sycl.a file
#endif
  return rtcNewSYCLDeviceInternal(context, config);
}

#if defined(EMBREE_SYCL_SUPPORT) && (defined(__SYCL_DEVICE_ONLY__) || defined(EMBREE_SYCL_RT_SIMULATION))

SYCL_EXTERNAL __attribute__((always_inline)) void rtcIntersect1(RTCScene hscene, struct RTCRayHit* rayhit, struct RTCIntersectArguments* args)
{
  RTCIntersectArguments default_args;
  if (args == nullptr) {
    rtcInitIntersectArguments(&default_args);
    args = &default_args;
  }
  RTCRayQueryContext* context = args->context;

  RTCRayQueryContext defaultContext;
  if (unlikely(context == nullptr)) {
    rtcInitRayQueryContext(&defaultContext);
    context = &defaultContext;
  }
    
  rtcIntersectRTHW(hscene, context, rayhit, args); 
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcForwardIntersect1(const RTCIntersectFunctionNArguments* args_, RTCScene scene, struct RTCRay* iray, unsigned int instID)
{
  return rtcForwardIntersect1Ex(args_, scene, iray, instID, 0);
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcForwardIntersect1Ex(const RTCIntersectFunctionNArguments* args_, RTCScene scene, struct RTCRay* iray, unsigned int instID, unsigned int instPrimID)
{
  IntersectFunctionNArguments* args = (IntersectFunctionNArguments*) args_;
  assert(args->N == 1);
  assert(args->forward_scene == nullptr);
  
  Ray* oray = (Ray*)args->rayhit;
  oray->org.x = iray->org_x;
  oray->org.y = iray->org_y;
  oray->org.z = iray->org_z;
  oray->dir.x = iray->dir_x;
  oray->dir.y = iray->dir_y;
  oray->dir.z = iray->dir_z;
  args->forward_scene = scene;
  instance_id_stack::push(args->context, instID, instPrimID);
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcOccluded1(RTCScene hscene, struct RTCRay* ray, struct RTCOccludedArguments* args)
{
  RTCOccludedArguments default_args;
  if (args == nullptr) {
    rtcInitOccludedArguments(&default_args);
    args = &default_args;
  }
  RTCRayQueryContext* context = args->context;

  RTCRayQueryContext defaultContext;
  if (unlikely(context == nullptr)) {
    rtcInitRayQueryContext(&defaultContext);
    context = &defaultContext;
  }
  
  rtcOccludedRTHW(hscene, context, ray, args);
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcForwardOccluded1(const RTCOccludedFunctionNArguments *args_, RTCScene scene, struct RTCRay *iray, unsigned int instID){
  return rtcForwardOccluded1Ex(args_, scene, iray, instID, 0);
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcForwardOccluded1Ex(const RTCOccludedFunctionNArguments *args_, RTCScene scene, struct RTCRay *iray, unsigned int instID, unsigned int instPrimID)
{
  OccludedFunctionNArguments* args = (OccludedFunctionNArguments*) args_;
  assert(args->N == 1);
  assert(args->forward_scene == nullptr);
   
  Ray* oray = (Ray*)args->ray;
  oray->org.x = iray->org_x;
  oray->org.y = iray->org_y;
  oray->org.z = iray->org_z;
  oray->dir.x = iray->dir_x;
  oray->dir.y = iray->dir_y;
  oray->dir.z = iray->dir_z;
  args->forward_scene = scene;
  instance_id_stack::push(args->context, instID, instPrimID);
}

SYCL_EXTERNAL __attribute__((always_inline)) void* rtcGetGeometryUserDataFromScene (RTCScene hscene, unsigned int geomID)
{
  Scene* scene = (Scene*) hscene;
  //RTC_CATCH_BEGIN;
  //RTC_TRACE(rtcGetGeometryUserDataFromScene);
#if defined(DEBUG)
  //RTC_VERIFY_HANDLE(hscene);
  //RTC_VERIFY_GEOMID(geomID);
#endif
  //RTC_ENTER_DEVICE(hscene); // do not enable for performance reasons
  return scene->get(geomID)->getUserData();
  //RTC_CATCH_END2(scene);
  //return nullptr;
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcGetGeometryTransformFromScene(RTCScene hscene, unsigned int geomID, float time, enum RTCFormat format, void* xfm)
{
  Scene* scene = (Scene*) hscene;
  //RTC_CATCH_BEGIN;
  //RTC_TRACE(rtcGetGeometryTransformFromScene);
  //RTC_ENTER_DEVICE(hscene);
  AffineSpace3fa transform = one;
  Geometry* geom = scene->get(geomID);
  if (geom->getTypeMask() & Geometry::MTY_INSTANCE) {
    Instance* instance = (Instance*) geom;
    if (likely(instance->numTimeSteps <= 1))
      transform = instance->getLocal2World();
    else
      transform = instance->getLocal2World(time);
  }
  storeTransform(transform, format, (float*)xfm);
  //RTC_CATCH_END2(geometry);
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcInvokeIntersectFilterFromGeometry(const RTCIntersectFunctionNArguments* args_i, const RTCFilterFunctionNArguments* filter_args)
{
#if EMBREE_SYCL_GEOMETRY_CALLBACK
  IntersectFunctionNArguments* args = (IntersectFunctionNArguments*) args_i;
  if (args->geometry->intersectionFilterN)
    args->geometry->intersectionFilterN(filter_args);
#endif
}

SYCL_EXTERNAL __attribute__((always_inline)) void rtcInvokeOccludedFilterFromGeometry(const RTCOccludedFunctionNArguments* args_i, const RTCFilterFunctionNArguments* filter_args)
{
#if EMBREE_SYCL_GEOMETRY_CALLBACK
  OccludedFunctionNArguments* args = (OccludedFunctionNArguments*) args_i;
  if (args->geometry->occlusionFilterN)
    args->geometry->occlusionFilterN(filter_args);
#endif
}

#endif

RTC_NAMESPACE_END;
