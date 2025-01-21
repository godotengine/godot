// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "accel.h"

namespace embree
{
  /*! merges N acceleration structures together, by processing them in order */
  class AccelN : public Accel
  {
  public:
    AccelN ();
    ~AccelN();

  public:
    void accels_add(Accel* accel);
    void accels_init();

  public:
    static bool pointQuery (Accel::Intersectors* This, PointQuery* query, PointQueryContext* context);

  public:
    static void intersect (Accel::Intersectors* This, RTCRayHit& ray, RayQueryContext* context);
    static void intersect4 (const void* valid, Accel::Intersectors* This, RTCRayHit4& ray, RayQueryContext* context);
    static void intersect8 (const void* valid, Accel::Intersectors* This, RTCRayHit8& ray, RayQueryContext* context);
    static void intersect16 (const void* valid, Accel::Intersectors* This, RTCRayHit16& ray, RayQueryContext* context);

  public:
    static void occluded (Accel::Intersectors* This, RTCRay& ray, RayQueryContext* context);
    static void occluded4 (const void* valid, Accel::Intersectors* This, RTCRay4& ray, RayQueryContext* context);
    static void occluded8 (const void* valid, Accel::Intersectors* This, RTCRay8& ray, RayQueryContext* context);
    static void occluded16 (const void* valid, Accel::Intersectors* This, RTCRay16& ray, RayQueryContext* context);

  public:
    void accels_print(size_t ident);
    void accels_immutable();
    void accels_build ();
    void accels_select(bool filter);
    void accels_deleteGeometry(size_t geomID);
    void accels_clear ();

  public:
    std::vector<Accel*> accels;
  };
}
