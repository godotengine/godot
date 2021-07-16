// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "accel.h"
#include "builder.h"

namespace embree
{
  class AccelInstance : public Accel
  {
  public:
    AccelInstance (AccelData* accel, Builder* builder, Intersectors& intersectors)
      : Accel(AccelData::TY_ACCEL_INSTANCE,intersectors), accel(accel), builder(builder) {}

    void immutable () {
      builder.reset(nullptr);
    }

  public:
    void build () {
      if (builder) builder->build();
      bounds = accel->bounds;
    }

    void deleteGeometry(size_t geomID) {
      if (accel  ) accel->deleteGeometry(geomID);
      if (builder) builder->deleteGeometry(geomID);
    }
    
    void clear() {
      if (accel) accel->clear();
      if (builder) builder->clear();
    }

  private:
    std::unique_ptr<AccelData> accel;
    std::unique_ptr<Builder> builder;
  };
}
