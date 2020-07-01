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

#include "accelset.h"

namespace embree
{
  /*! User geometry with user defined intersection functions */
  struct UserGeometry : public AccelSet
  {
    /*! type of this geometry */
    static const Geometry::GTypeMask geom_type = Geometry::MTY_USER_GEOMETRY;

  public:
    UserGeometry (Device* device, unsigned int items = 0, unsigned int numTimeSteps = 1);
    virtual void enabling ();
    virtual void disabling();
    virtual void setMask (unsigned mask);
    virtual void setBoundsFunction (RTCBoundsFunction bounds, void* userPtr);
    virtual void setIntersectFunctionN (RTCIntersectFunctionN intersect);
    virtual void setOccludedFunctionN (RTCOccludedFunctionN occluded);
    virtual void build() {}
  };

  namespace isa
  {
    struct UserGeometryISA : public UserGeometry
    {
      UserGeometryISA (Device* device)
        : UserGeometry(device) {}

      PrimInfo createPrimRefArray(mvector<PrimRef>& prims, const range<size_t>& r, size_t k) const
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          BBox3fa bounds = empty;
          if (!buildBounds(j,&bounds)) continue;
          const PrimRef prim(bounds,geomID,unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k) const
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          BBox3fa bounds = empty;
          if (!buildBounds(j,itime,bounds)) continue;
          const PrimRef prim(bounds,geomID,unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }
      
      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k) const
      {
        PrimInfoMB pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(j, timeSegmentRange(t0t1))) continue;
          const PrimRefMB prim(linearBounds(j,t0t1),this->numTimeSegments(),this->time_range,this->numTimeSegments(),this->geomID,unsigned(j));
          pinfo.add_primref(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }
    };
  }
  
  DECLARE_ISA_FUNCTION(UserGeometry*, createUserGeometry, Device*);
}
