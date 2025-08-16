// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry.h"
#include "accel.h"

namespace embree
{
  struct MotionDerivativeCoefficients;

  /*! Instanced acceleration structure */
  struct InstanceArray : public Geometry
  {
    static const Geometry::GTypeMask geom_type = Geometry::MTY_INSTANCE_ARRAY;

  public:
    InstanceArray (Device* device, unsigned int numTimeSteps = 1);
    ~InstanceArray();

  private:
    InstanceArray (const InstanceArray& other) DELETED; // do not implement
    InstanceArray& operator= (const InstanceArray& other) DELETED; // do not implement

  private:
    LBBox3fa nonlinearBounds(size_t i,
                             const BBox1f& time_range_in,
                             const BBox1f& geom_time_range,
                             float geom_time_segments) const;

    BBox3fa boundSegment(size_t i, size_t itime,
      BBox3fa const& obbox0, BBox3fa const& obbox1,
      BBox3fa const& bbox0, BBox3fa const& bbox1,
      float t_min, float t_max) const;

    /* calculates the (correct) interpolated bounds */
    __forceinline BBox3fa bounds(size_t i, size_t itime0, size_t itime1, float f) const
    {
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return xfmBounds(slerp(l2w(i, itime0), l2w(i, itime1), f),
                         lerp(getObjectBounds(i, itime0), getObjectBounds(i, itime1), f));
      return xfmBounds(lerp(l2w(i, itime0), l2w(i, itime1), f),
                        lerp(getObjectBounds(i, itime0), getObjectBounds(i, itime1), f));
    }

  public:

    virtual void setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num) override;
    virtual void* getBufferData(RTCBufferType type, unsigned int slot, BufferDataPointerType pointerType) override;
    virtual void updateBuffer(RTCBufferType type, unsigned int slot) override;

    virtual void setNumTimeSteps (unsigned int numTimeSteps) override;
    virtual void setInstancedScene(const Ref<Scene>& scene) override;
    virtual void setInstancedScenes(const RTCScene* scenes, size_t numScenes) override;
    virtual AffineSpace3fa getTransform(size_t, float time) override;
    virtual void setMask (unsigned mask) override;
    virtual void build() {}
    virtual void addElementsToCount (GeometryCounts & counts) const override;
    virtual void commit() override;
    size_t getGeometryDataDeviceByteSize() const override;
    void convertToDeviceRepresentation(size_t offset, char* data_host, char* data_device) const override;

  public:

     /*! calculates the bounds of instance */
    __forceinline BBox3fa bounds(size_t i) const {
      if (!valid(i))
        return BBox3fa();

      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return xfmBounds(quaternionDecompositionToAffineSpace(l2w(i, 0)),getObject(i)->bounds.bounds());
      return xfmBounds(l2w(i, 0),getObject(i)->bounds.bounds());
    }

    /*! gets the bounds of the instanced scene */
    __forceinline BBox3fa getObjectBounds(size_t i, size_t itime) const {
      if (!valid(i))
        return BBox3fa();

      return getObject(i)->getBounds(timeStep(itime));
    }

     /*! calculates the bounds of instance */
    __forceinline BBox3fa bounds(size_t i, size_t itime) const {
      if (!valid(i))
        return BBox3fa();

      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return xfmBounds(quaternionDecompositionToAffineSpace(l2w(i, itime)),getObjectBounds(i, itime));
      return xfmBounds(l2w(i, itime),getObjectBounds(i, itime));
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(size_t i, const BBox1f& dt) const {
      if (!valid(i))
        return LBBox3fa();

      LBBox3fa lbbox = nonlinearBounds(i, dt, time_range, fnumTimeSegments);
      return lbbox;
    }

    /*! calculates the build bounds of the i'th item, if it's valid */
    __forceinline bool buildBounds(size_t i, BBox3fa* bbox = nullptr) const
    {
      if (!valid(i))
        return false;

      const BBox3fa b = bounds(i);
      if (bbox) *bbox = b;
      return isvalid(b);
    }

     /*! calculates the build bounds of the i'th item at the itime'th time segment, if it's valid */
    __forceinline bool buildBounds(size_t i, size_t itime, BBox3fa& bbox) const
    {
      if (!valid(i))
        return false;

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
    __forceinline bool valid(size_t i) const
    {
      if (object) return true;
      return (object_ids[i] != (unsigned int)(-1));
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
    {
      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
        if (!isvalid(bounds(i,itime))) return false;

      return true;
    }

    __forceinline AffineSpace3fa getLocal2World(size_t i) const
    {
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return quaternionDecompositionToAffineSpace(l2w(i,0));
      return l2w(i, 0);
    }

    __forceinline AffineSpace3fa getLocal2World(size_t i, float t) const
    {
      if (numTimeSegments() > 0) {
        float ftime; const unsigned int itime = timeSegment(t, ftime);
        if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
          return slerp(l2w(i, itime+0),l2w(i, itime+1),ftime);
        return lerp(l2w(i, itime+0),l2w(i, itime+1),ftime);
      }
      return getLocal2World(i);
    }

    __forceinline AffineSpace3fa getWorld2Local(size_t i) const {
      return rcp(getLocal2World(i));
    }

    __forceinline AffineSpace3fa getWorld2Local(size_t i, float t) const {
      return rcp(getLocal2World(i, t));
    }

    template<int K>
    __forceinline AffineSpace3vf<K> getWorld2Local(size_t i, const vbool<K>& valid, const vfloat<K>& t) const
    {
      if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
        return getWorld2LocalSlerp<K>(i, valid, t);
      return getWorld2LocalLerp<K>(i, valid, t);
    }

    __forceinline float projectedPrimitiveArea(const size_t i) const {
      return area(bounds(i));
    }

    inline Accel* getObject(size_t i) const {
      if (object) {
        return object;
      }

      assert(objects);
      assert(i < numPrimitives);
      if (object_ids[i] == (unsigned int)(-1))
        return nullptr;

      assert(object_ids[i] < numObjects);
      return objects[object_ids[i]];
    }

    private:

    template<int K>
    __forceinline AffineSpace3vf<K> getWorld2LocalSlerp(size_t i, const vbool<K>& valid, const vfloat<K>& t) const
    {
      vfloat<K> ftime;
      const vint<K> itime_k = timeSegment<K>(t, ftime);
      assert(any(valid));
      const size_t index = bsf(movemask(valid));
      const int itime = itime_k[index];
      if (likely(all(valid, itime_k == vint<K>(itime)))) {
        return rcp(slerp(AffineSpace3vff<K>(l2w(i, itime+0)),
                         AffineSpace3vff<K>(l2w(i, itime+1)),
                         ftime));
      }
      else {
        AffineSpace3vff<K> space0,space1;
        vbool<K> valid1 = valid;
        while (any(valid1)) {
          vbool<K> valid2;
          const int itime = next_unique(valid1, itime_k, valid2);
          space0 = select(valid2, AffineSpace3vff<K>(l2w(i, itime+0)), space0);
          space1 = select(valid2, AffineSpace3vff<K>(l2w(i, itime+1)), space1);
        }
        return rcp(slerp(space0, space1, ftime));
      }
    }

    template<int K>
    __forceinline AffineSpace3vf<K> getWorld2LocalLerp(size_t i, const vbool<K>& valid, const vfloat<K>& t) const
    {
      vfloat<K> ftime;
      const vint<K> itime_k = timeSegment<K>(t, ftime);
      assert(any(valid));
      const size_t index = bsf(movemask(valid));
      const int itime = itime_k[index];
      if (likely(all(valid, itime_k == vint<K>(itime)))) {
        return rcp(lerp(AffineSpace3vf<K>((AffineSpace3fa)l2w(i, itime+0)),
                        AffineSpace3vf<K>((AffineSpace3fa)l2w(i, itime+1)),
                        ftime));
      } else {
        AffineSpace3vf<K> space0,space1;
        vbool<K> valid1 = valid;
        while (any(valid1)) {
          vbool<K> valid2;
          const int itime = next_unique(valid1, itime_k, valid2);
          space0 = select(valid2, AffineSpace3vf<K>((AffineSpace3fa)l2w(i, itime+0)), space0);
          space1 = select(valid2, AffineSpace3vf<K>((AffineSpace3fa)l2w(i, itime+1)), space1);
        }
        return rcp(lerp(space0, space1, ftime));
      }
    }

  private:

    __forceinline AffineSpace3ff l2w(size_t i, size_t itime) const {
      if (l2w_buf[itime].getFormat() == RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR) {
        return *(AffineSpace3ff*)(l2w_buf[itime].getPtr(i));
      }
      else if(l2w_buf[itime].getFormat() == RTC_FORMAT_QUATERNION_DECOMPOSITION) {
        AffineSpace3ff transform;
        QuaternionDecomposition* qd = (QuaternionDecomposition*)l2w_buf[itime].getPtr(i);
        transform.l.vx.x = qd->scale_x;
        transform.l.vy.y = qd->scale_y;
        transform.l.vz.z = qd->scale_z;
        transform.l.vy.x = qd->skew_xy;
        transform.l.vz.x = qd->skew_xz;
        transform.l.vz.y = qd->skew_yz;
        transform.l.vx.y = qd->translation_x;
        transform.l.vx.z = qd->translation_y;
        transform.l.vy.z = qd->translation_z;
        transform.p.x    = qd->shift_x;
        transform.p.y    = qd->shift_y;
        transform.p.z    = qd->shift_z;
        // normalize quaternion
        Quaternion3f q(qd->quaternion_r, qd->quaternion_i, qd->quaternion_j, qd->quaternion_k);
        q = normalize(q);
        transform.l.vx.w = q.i;
        transform.l.vy.w = q.j;
        transform.l.vz.w = q.k;
        transform.p.w    = q.r;
        return transform;
      }
      else if (l2w_buf[itime].getFormat() == RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR) {
        AffineSpace3f* l2w = reinterpret_cast<AffineSpace3f*>(l2w_buf[itime].getPtr(i));
        return AffineSpace3ff(*l2w);
      }
      else if (l2w_buf[itime].getFormat() == RTC_FORMAT_FLOAT3X4_ROW_MAJOR) {
        float* data = reinterpret_cast<float*>(l2w_buf[itime].getPtr(i));
        AffineSpace3f l2w;
        l2w.l.vx.x = data[0]; l2w.l.vy.x = data[1]; l2w.l.vz.x = data[2]; l2w.p.x = data[3];
        l2w.l.vx.y = data[4]; l2w.l.vy.y = data[5]; l2w.l.vz.y = data[6]; l2w.p.y = data[7];
        l2w.l.vx.z = data[8]; l2w.l.vy.z = data[9]; l2w.l.vz.z = data[10]; l2w.p.z = data[11];
        return l2w;
      }
      assert(false);
      return AffineSpace3ff();
    }

    inline AffineSpace3ff l2w(size_t i) const {
      return l2w(i, 0);
    }

  private:
    Accel* object;                   //!< fast path if only one scene is instanced
    Accel** objects;
    uint32_t numObjects;
    Device::vector<RawBufferView> l2w_buf = device; //!< transformation from local space to world space for each timestep (either normal matrix or quaternion decomposition)
    BufferView<uint32_t> object_ids; //!< array of scene ids per instance array primitive
  };

  namespace isa
  {
    struct InstanceArrayISA : public InstanceArray
    {
      InstanceArrayISA (Device* device)
        : InstanceArray(device) {}

      LBBox3fa vlinearBounds(size_t primID, const BBox1f& time_range) const {
        return linearBounds(primID,time_range);
      }

      PrimInfo createPrimRefArray(PrimRef* prims, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        for (size_t j = r.begin(); j < r.end(); j++) {
          BBox3fa bounds = empty;
          if (!buildBounds(j, &bounds) || !valid(j))
            continue;
          const PrimRef prim(bounds, geomID, unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        for (size_t j = r.begin(); j < r.end(); j++) {
          BBox3fa bounds = empty;
          if (!buildBounds(j, itime, bounds))
            continue;
          const PrimRef prim(bounds, geomID, unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(PrimRef* prims, const BBox1f& time_range, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        const BBox1f t0t1 = BBox1f::intersect(getTimeRange(), time_range);
        if (t0t1.empty()) return pinfo;

        for (size_t j = r.begin(); j < r.end(); j++) {
          LBBox3fa lbounds = linearBounds(j, t0t1);
          if (!isvalid(lbounds.bounds()))
            continue;
          const PrimRef prim(lbounds.bounds(), geomID, unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfoMB pinfo(empty);
        for (size_t j = r.begin(); j < r.end(); j++) {
          if (!valid(j, timeSegmentRange(t0t1)))
            continue;
          const PrimRefMB prim(linearBounds(j, t0t1), this->numTimeSegments(), this->time_range, this->numTimeSegments(), geomID, unsigned(j));
          pinfo.add_primref(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }
    };
  }

  DECLARE_ISA_FUNCTION(InstanceArray*, createInstanceArray, Device*);
}
