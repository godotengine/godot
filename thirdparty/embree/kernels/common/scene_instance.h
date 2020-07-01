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

#include "geometry.h"
#include "accel.h"

namespace embree
{
  /*! Instanced acceleration structure */
  struct Instance : public Geometry
  {
    ALIGNED_STRUCT_(16);
    static const Geometry::GTypeMask geom_type = Geometry::MTY_INSTANCE;
    
  public:
    Instance (Device* device, Accel* object = nullptr, unsigned int numTimeSteps = 1);
    ~Instance();

  private:
    Instance (const Instance& other) DELETED; // do not implement
    Instance& operator= (const Instance& other) DELETED; // do not implement
    
  public:
    virtual void enabling ();
    virtual void disabling();
    virtual void setNumTimeSteps (unsigned int numTimeSteps);
    virtual void setInstancedScene(const Ref<Scene>& scene);
    virtual void setTransform(const AffineSpace3fa& local2world, unsigned int timeStep);
    virtual AffineSpace3fa getTransform(float time);
    virtual void setMask (unsigned mask);
    virtual void build() {}

  public:

     /*! calculates the bounds of instance */
    __forceinline BBox3fa bounds(size_t i) const {
      assert(i == 0);
      return xfmBounds(local2world[0],object->bounds.bounds());
    }

     /*! calculates the bounds of instance */
    __forceinline BBox3fa bounds(size_t i, size_t itime) const {
      assert(i == 0);
      return xfmBounds(local2world[itime],object->getBounds(float(itime)/fnumTimeSegments));
    }

     /*! calculates the linear bounds at the itimeGlobal'th time segment */
    __forceinline LBBox3fa linearBounds(size_t i, size_t itime) const {
      assert(i == 0);
      return LBBox3fa(bounds(i,itime+0),bounds(i,itime+1));
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(size_t i, const BBox1f& dt) const {
      assert(i == 0);
      return LBBox3fa([&] (size_t itime) { return bounds(i, itime); }, dt, time_range, fnumTimeSegments);
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
    {
      assert(i == 0);
      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
        if (!isvalid(bounds(i,itime))) return false;
      
      return true;
    }
      
    __forceinline AffineSpace3fa getLocal2World() const {
      return local2world[0];
    }

    __forceinline AffineSpace3fa getLocal2World(float t) const
    {
      float ftime; const unsigned int itime = timeSegment(t, ftime);
      return lerp(local2world[itime+0],local2world[itime+1],ftime);
    }

    __forceinline AffineSpace3fa getWorld2Local() const {
      return world2local0;
    }

    __forceinline AffineSpace3fa getWorld2Local(float t) const {
      return rcp(getLocal2World(t));
    }

    template<int K>
    __forceinline AffineSpace3vf<K> getWorld2Local(const vbool<K>& valid, const vfloat<K>& t) const
    { 
      vfloat<K> ftime;
      const vint<K> itime_k = timeSegment(t, ftime);
      assert(any(valid));
      const size_t index = bsf(movemask(valid));
      const int itime = itime_k[index];
      const vfloat<K> t0 = vfloat<K>(1.0f)-ftime, t1 = ftime;
      if (likely(all(valid, itime_k == vint<K>(itime)))) {
        return rcp(t0*AffineSpace3vf<K>(local2world[itime+0]) + t1*AffineSpace3vf<K>(local2world[itime+1]));
      } else {
        AffineSpace3vf<K> space0,space1;
        vbool<K> valid1 = valid;
        while (any(valid1)) {
          vbool<K> valid2;
          const int itime = next_unique(valid1, itime_k, valid2);
          space0 = select(valid2, AffineSpace3vf<K>(local2world[itime+0]), space0);
          space1 = select(valid2, AffineSpace3vf<K>(local2world[itime+1]), space1);
        }
        return rcp(t0*space0 + t1*space1);
      }
    }
    
  public:
    Accel* object;                 //!< pointer to instanced acceleration structure
    AffineSpace3fa* local2world;   //!< transformation from local space to world space for each timestep
    AffineSpace3fa world2local0;   //!< transformation from world space to local space for timestep 0
  };

  namespace isa
  {
    struct InstanceISA : public Instance
    {
      InstanceISA (Device* device)
        : Instance(device) {}

      PrimInfo createPrimRefArray(mvector<PrimRef>& prims, const range<size_t>& r, size_t k) const
      {
        assert(r.begin() == 0);
        assert(r.end()   == 1);
        
        PrimInfo pinfo(empty);
        const BBox3fa b = bounds(0);
        if (!isvalid(b)) return pinfo;
        
        const PrimRef prim(b,geomID,unsigned(0));
        pinfo.add_center2(prim);
        prims[k++] = prim;
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k) const
      {
        assert(r.begin() == 0);
        assert(r.end()   == 1);
        
        PrimInfo pinfo(empty);
        if (!valid(0,range<size_t>(itime))) return pinfo;
        const PrimRef prim(linearBounds(0,itime).bounds(),geomID,unsigned(0));
        pinfo.add_center2(prim);
        prims[k++] = prim;
        return pinfo;
      }
      
      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k) const
      {
        assert(r.begin() == 0);
        assert(r.end()   == 1);
        
        PrimInfoMB pinfo(empty);
        if (!valid(0, timeSegmentRange(t0t1))) return pinfo;
        const PrimRefMB prim(linearBounds(0,t0t1),this->numTimeSegments(),this->time_range,this->numTimeSegments(),this->geomID,unsigned(0));
        pinfo.add_primref(prim);
        prims[k++] = prim;
        return pinfo;
      }
    };
  }
  
  DECLARE_ISA_FUNCTION(Instance*, createInstance, Device*);
}
