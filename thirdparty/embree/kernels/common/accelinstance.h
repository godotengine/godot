// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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
