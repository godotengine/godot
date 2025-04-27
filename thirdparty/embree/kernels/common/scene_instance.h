// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry.h"
#include "accel.h"

namespace embree
{
  struct MotionDerivativeCoefficients;

  /*! Instanced acceleration structure */
  struct Instance : public Geometry
  {
    static const Geometry::GTypeMask geom_type = Geometry::MTY_INSTANCE;

  public:
    Instance (Device* device, Accel* object = nullptr, unsigned int numTimeSteps = 1);
    ~Instance();

  private:
    Instance (const Instance& other) DELETED; // do not implement
    Instance& operator= (const Instance& other) DELETED; // do not implement

  private:
    LBBox3fa nonlinearBounds(const BBox1f& time_range_in,
                             const BBox1f& geom_time_range,
                             float geom_time_segments) const;

    BBox3fa boundSegment(size_t itime,
      BBox3fa const& obbox0, BBox3fa const& obbox1,
      BBox3fa const& bbox0, BBox3fa const& bbox1,
      float t_min, float t_max) const;

    /* calculates the (correct) interpolated bounds */
    __forceinline BBox3fa bounds(size_t itime0, size_t itime1, float f) const
    {
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return xfmBounds(slerp(local2world[itime0], local2world[itime1], f),
                         lerp(getObjectBounds(itime0), getObjectBounds(itime1), f));
      return xfmBounds(lerp(local2world[itime0], local2world[itime1], f),
                        lerp(getObjectBounds(itime0), getObjectBounds(itime1), f));
    }

  public:
    virtual void setNumTimeSteps (unsigned int numTimeSteps) override;
    virtual void setInstancedScene(const Ref<Scene>& scene) override;
    virtual void setTransform(const AffineSpace3fa& local2world, unsigned int timeStep) override;
    virtual void setQuaternionDecomposition(const AffineSpace3ff& qd, unsigned int timeStep) override;
    virtual AffineSpace3fa getTransform(float time) override;
    virtual AffineSpace3fa getTransform(size_t, float time) override;
    virtual void setMask (unsigned mask) override;
    virtual void build() {}
    virtual void addElementsToCount (GeometryCounts & counts) const override;
    virtual void commit() override;
    virtual size_t getGeometryDataDeviceByteSize() const override;
    virtual void convertToDeviceRepresentation(size_t offset, char* data_host, char* data_device) const override;

  public:

     /*! calculates the bounds of instance */
    __forceinline BBox3fa bounds(size_t i) const {
      assert(i == 0);
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return xfmBounds(quaternionDecompositionToAffineSpace(local2world[0]),object->bounds.bounds());
      return xfmBounds(local2world[0],object->bounds.bounds());
    }

    /*! gets the bounds of the instanced scene */
    __forceinline BBox3fa getObjectBounds(size_t itime) const {
      return object->getBounds(timeStep(itime));
    }

     /*! calculates the bounds of instance */
    __forceinline BBox3fa bounds(size_t i, size_t itime) const {
      assert(i == 0);
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return xfmBounds(quaternionDecompositionToAffineSpace(local2world[itime]),getObjectBounds(itime));
      return xfmBounds(local2world[itime],getObjectBounds(itime));
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(size_t i, const BBox1f& dt) const {
      assert(i == 0);
      LBBox3fa lbbox = nonlinearBounds(dt, time_range, fnumTimeSegments);
      return lbbox;
    }

    /*! calculates the build bounds of the i'th item, if it's valid */
    __forceinline bool buildBounds(size_t i, BBox3fa* bbox = nullptr) const
    {
      assert(i==0);
      const BBox3fa b = bounds(i);
      if (bbox) *bbox = b;
      return isvalid(b);
    }

     /*! calculates the build bounds of the i'th item at the itime'th time segment, if it's valid */
    __forceinline bool buildBounds(size_t i, size_t itime, BBox3fa& bbox) const
    {
      assert(i==0);
      const LBBox3fa bounds = linearBounds(i,itime);
      bbox = bounds.bounds ();
      return isvalid(bounds);
    }

    /* gets version info of topology */
    unsigned int getTopologyVersion() const {
      return numPrimitives;
    }
  
    /* returns true if topology changed */
    bool topologyChanged(unsigned int otherVersion) const {
      return numPrimitives != otherVersion;
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
    {
      assert(i == 0);
      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
        if (!isvalid(bounds(i,itime))) return false;

      return true;
    }

    __forceinline AffineSpace3fa getLocal2World() const
    {
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return quaternionDecompositionToAffineSpace(local2world[0]);
      return local2world[0];
    }

    __forceinline AffineSpace3fa getLocal2World(float t) const
    {
      if (numTimeSegments() > 0) {
        float ftime; const unsigned int itime = timeSegment(t, ftime);
        if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
          return slerp(local2world[itime+0],local2world[itime+1],ftime);
        return lerp(local2world[itime+0],local2world[itime+1],ftime);
      }
      return getLocal2World();
    }

    __forceinline AffineSpace3fa getWorld2Local() const {
      return world2local0;
    }

    __forceinline AffineSpace3fa getWorld2Local(float t) const {
      if (numTimeSegments() > 0)
        return rcp(getLocal2World(t));
      return getWorld2Local();
    }

    template<int K>
    __forceinline AffineSpace3vf<K> getWorld2Local(const vbool<K>& valid, const vfloat<K>& t) const
    {
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return getWorld2LocalSlerp<K>(valid, t);
      return getWorld2LocalLerp<K>(valid, t);
    }

    __forceinline float projectedPrimitiveArea(const size_t i) const {
      return area(bounds(i));
    }

    private:

    template<int K>
    __forceinline AffineSpace3vf<K> getWorld2LocalSlerp(const vbool<K>& valid, const vfloat<K>& t) const
    {
      vfloat<K> ftime;
      const vint<K> itime_k = timeSegment<K>(t, ftime);
      assert(any(valid));
      const size_t index = bsf(movemask(valid));
      const int itime = itime_k[index];
      if (likely(all(valid, itime_k == vint<K>(itime)))) {
        return rcp(slerp(AffineSpace3vff<K>(local2world[itime+0]),
                         AffineSpace3vff<K>(local2world[itime+1]),
                         ftime));
      }
      else {
        AffineSpace3vff<K> space0,space1;
        vbool<K> valid1 = valid;
        while (any(valid1)) {
          vbool<K> valid2;
          const int itime = next_unique(valid1, itime_k, valid2);
          space0 = select(valid2, AffineSpace3vff<K>(local2world[itime+0]), space0);
          space1 = select(valid2, AffineSpace3vff<K>(local2world[itime+1]), space1);
        }
        return rcp(slerp(space0, space1, ftime));
      }
    }

    template<int K>
    __forceinline AffineSpace3vf<K> getWorld2LocalLerp(const vbool<K>& valid, const vfloat<K>& t) const
    {
      vfloat<K> ftime;
      const vint<K> itime_k = timeSegment<K>(t, ftime);
      assert(any(valid));
      const size_t index = bsf(movemask(valid));
      const int itime = itime_k[index];
      if (likely(all(valid, itime_k == vint<K>(itime)))) {
        return rcp(lerp(AffineSpace3vf<K>((AffineSpace3fa)local2world[itime+0]),
                        AffineSpace3vf<K>((AffineSpace3fa)local2world[itime+1]),
                        ftime));
      } else {
        AffineSpace3vf<K> space0,space1;
        vbool<K> valid1 = valid;
        while (any(valid1)) {
          vbool<K> valid2;
          const int itime = next_unique(valid1, itime_k, valid2);
          space0 = select(valid2, AffineSpace3vf<K>((AffineSpace3fa)local2world[itime+0]), space0);
          space1 = select(valid2, AffineSpace3vf<K>((AffineSpace3fa)local2world[itime+1]), space1);
        }
        return rcp(lerp(space0, space1, ftime));
      }
    }

  public:
    Accel* object;                 //!< pointer to instanced acceleration structure
    AffineSpace3ff* local2world;   //!< transformation from local space to world space for each timestep (either normal matrix or quaternion decomposition)
    AffineSpace3fa world2local0;   //!< transformation from world space to local space for timestep 0
  };

  namespace isa
  {
    struct InstanceISA : public Instance
    {
      InstanceISA (Device* device)
        : Instance(device) {}

      LBBox3fa vlinearBounds(size_t primID, const BBox1f& time_range) const {
        return linearBounds(primID,time_range);
      }

      PrimInfo createPrimRefArray(PrimRef* prims, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        assert(r.begin() == 0);
        assert(r.end()   == 1);

        PrimInfo pinfo(empty);
        BBox3fa b = empty;
        if (!buildBounds(0,&b)) return pinfo;
        // const BBox3fa b = bounds(0);
        // if (!isvalid(b)) return pinfo;

        const PrimRef prim(b,geomID,unsigned(0));
        pinfo.add_center2(prim);
        prims[k++] = prim;
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        assert(r.begin() == 0);
        assert(r.end()   == 1);

        PrimInfo pinfo(empty);
        BBox3fa b = empty;
        if (!buildBounds(0,&b)) return pinfo;
        // if (!valid(0,range<size_t>(itime))) return pinfo;
        // const PrimRef prim(linearBounds(0,itime).bounds(),geomID,unsigned(0));
        const PrimRef prim(b,geomID,unsigned(0));
        pinfo.add_center2(prim);
        prims[k++] = prim;
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(PrimRef* prims, const BBox1f& time_range, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        assert(r.begin() == 0);
        assert(r.end()   == 1);

        PrimInfo pinfo(empty);
        const BBox1f t0t1 = intersect(getTimeRange(), time_range);
        if (t0t1.empty()) return pinfo;
        
        const BBox3fa bounds = linearBounds(0, t0t1).bounds();
        const PrimRef prim(bounds, geomID, unsigned(0));
        pinfo.add_center2(prim);
        prims[k++] = prim;
        return pinfo;
      }

      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        assert(r.begin() == 0);
        assert(r.end()   == 1);

        PrimInfoMB pinfo(empty);
        if (!valid(0, timeSegmentRange(t0t1))) return pinfo;
        const PrimRefMB prim(linearBounds(0,t0t1),this->numTimeSegments(),this->time_range,this->numTimeSegments(),geomID,unsigned(0));
        pinfo.add_primref(prim);
        prims[k++] = prim;
        return pinfo;
      }
    };
  }

  DECLARE_ISA_FUNCTION(Instance*, createInstance, Device*);
}
