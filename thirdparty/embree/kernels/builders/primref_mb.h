// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/default.h"

#define MBLUR_BIN_LBBOX 1

namespace embree
{
#if MBLUR_BIN_LBBOX

  /*! A primitive reference stores the bounds of the primitive and its ID. */
  struct PrimRefMB
  {
    typedef LBBox3fa BBox;

    __forceinline PrimRefMB () {}

    __forceinline PrimRefMB (const LBBox3fa& lbounds_i, unsigned int activeTimeSegments, BBox1f time_range, unsigned int totalTimeSegments, unsigned int geomID, unsigned int primID)
      : lbounds((LBBox3fx)lbounds_i), time_range(time_range)
    {
      assert(activeTimeSegments > 0);
      lbounds.bounds0.lower.a = geomID;
      lbounds.bounds0.upper.a = primID;
      lbounds.bounds1.lower.a = activeTimeSegments;
      lbounds.bounds1.upper.a = totalTimeSegments;
    }

    __forceinline PrimRefMB (EmptyTy empty, const LBBox3fa& lbounds_i, unsigned int activeTimeSegments, BBox1f time_range, unsigned int totalTimeSegments, size_t id)
      : lbounds((LBBox3fx)lbounds_i), time_range(time_range)
    {
      assert(activeTimeSegments > 0);
#if defined(__64BIT__)
      lbounds.bounds0.lower.a = id & 0xFFFFFFFF;
      lbounds.bounds0.upper.a = (id >> 32) & 0xFFFFFFFF;
#else
      lbounds.bounds0.lower.a = id;
      lbounds.bounds0.upper.a = 0;
#endif
      lbounds.bounds1.lower.a = activeTimeSegments;
      lbounds.bounds1.upper.a = totalTimeSegments;
    }
    
    __forceinline PrimRefMB (const LBBox3fa& lbounds_i, unsigned int activeTimeSegments, BBox1f time_range, unsigned int totalTimeSegments, size_t id)
      : lbounds((LBBox3fx)lbounds_i), time_range(time_range)
    {
      assert(activeTimeSegments > 0);
#if defined(__64BIT__)
      lbounds.bounds0.lower.u = id & 0xFFFFFFFF;
      lbounds.bounds0.upper.u = (id >> 32) & 0xFFFFFFFF;
#else
      lbounds.bounds0.lower.u = id;
      lbounds.bounds0.upper.u = 0;
#endif
      lbounds.bounds1.lower.a = activeTimeSegments;
      lbounds.bounds1.upper.a = totalTimeSegments;
    }

    /*! returns bounds for binning */
    __forceinline LBBox3fa bounds() const {
      return lbounds;
    }

    /*! returns the number of time segments of this primref */
    __forceinline unsigned size() const {
      return lbounds.bounds1.lower.a;
    }

    __forceinline unsigned totalTimeSegments() const {
      return lbounds.bounds1.upper.a;
    }

     /* calculate overlapping time segment range */
    __forceinline range<int> timeSegmentRange(const BBox1f& range) const {
      return getTimeSegmentRange(range,time_range,float(totalTimeSegments()));
    }

     /* returns time that corresponds to time step */
    __forceinline float timeStep(const int i) const {
      assert(i>=0 && i<=(int)totalTimeSegments());
      return time_range.lower + time_range.size()*float(i)/float(totalTimeSegments());
    }
    
    /*! checks if time range overlaps */
    __forceinline bool time_range_overlap(const BBox1f& range) const
    {
      if (0.9999f*time_range.upper <= range.lower) return false;
      if (1.0001f*time_range.lower >= range.upper) return false;
      return true;
    }

    /*! returns center for binning */
    __forceinline Vec3fa binCenter() const {
      return center2(lbounds.interpolate(0.5f));
    }

    /*! returns bounds and centroid used for binning */
    __forceinline void binBoundsAndCenter(LBBox3fa& bounds_o, Vec3fa& center_o) const
    {
      bounds_o = bounds();
      center_o = binCenter();
    }

    /*! returns the geometry ID */
    __forceinline unsigned geomID() const {
      return lbounds.bounds0.lower.a;
    }

    /*! returns the primitive ID */
    __forceinline unsigned primID() const {
      return lbounds.bounds0.upper.a;
    }

    /*! returns an size_t sized ID */
    __forceinline size_t ID() const {
#if defined(__64BIT__)
      return size_t(lbounds.bounds0.lower.u) + (size_t(lbounds.bounds0.upper.u) << 32);
#else
      return size_t(lbounds.bounds0.lower.u);
#endif
    }

    /*! special function for operator< */
    __forceinline uint64_t ID64() const {
      return (((uint64_t)primID()) << 32) + (uint64_t)geomID();
    }

    /*! allows sorting the primrefs by ID */
    friend __forceinline bool operator<(const PrimRefMB& p0, const PrimRefMB& p1) {
      return p0.ID64() < p1.ID64();
    }

    /*! Outputs primitive reference to a stream. */
    friend __forceinline embree_ostream operator<<(embree_ostream cout, const PrimRefMB& ref) {
      return cout << "{ time_range = " << ref.time_range << ", bounds = " << ref.bounds() << ", geomID = " << ref.geomID() << ", primID = " << ref.primID() << ", active_segments = " << ref.size() << ",  total_segments = " << ref.totalTimeSegments() << " }";
    }

  public:
    LBBox3fx lbounds;
    BBox1f time_range; // entire geometry time range
  };

#else

  /*! A primitive reference stores the bounds of the primitive and its ID. */
  struct __aligned(16) PrimRefMB
  {
    typedef BBox3fa BBox;

    __forceinline PrimRefMB () {}

    __forceinline PrimRefMB (const LBBox3fa& bounds, unsigned int activeTimeSegments, BBox1f time_range, unsigned int totalTimeSegments, unsigned int geomID, unsigned int primID)
      : bbox(bounds.interpolate(0.5f)), _activeTimeSegments(activeTimeSegments), _totalTimeSegments(totalTimeSegments), time_range(time_range)
    {
      assert(activeTimeSegments > 0);
      bbox.lower.a = geomID;
      bbox.upper.a = primID;
    }
    
    __forceinline PrimRefMB (EmptyTy empty, const LBBox3fa& bounds, unsigned int activeTimeSegments, BBox1f time_range, unsigned int totalTimeSegments, size_t id)
      : bbox(bounds.interpolate(0.5f)), _activeTimeSegments(activeTimeSegments), _totalTimeSegments(totalTimeSegments), time_range(time_range)
    {
      assert(activeTimeSegments > 0);
#if defined(__64BIT__)
      bbox.lower.u = id & 0xFFFFFFFF;
      bbox.upper.u = (id >> 32) & 0xFFFFFFFF;
#else
      bbox.lower.u = id;
      bbox.upper.u = 0;
#endif
    }
    
    /*! returns bounds for binning */
    __forceinline BBox3fa bounds() const {
      return bbox;
    }

    /*! returns the number of time segments of this primref */
    __forceinline unsigned int size() const { 
      return _activeTimeSegments;
    }

    __forceinline unsigned int totalTimeSegments() const { 
      return _totalTimeSegments;
    }

     /* calculate overlapping time segment range */
    __forceinline range<int> timeSegmentRange(const BBox1f& range) const {
      return getTimeSegmentRange(range,time_range,float(_totalTimeSegments));
    }

     /* returns time that corresponds to time step */
    __forceinline float timeStep(const int i) const {
      assert(i>=0 && i<=(int)_totalTimeSegments);
      return time_range.lower + time_range.size()*float(i)/float(_totalTimeSegments);
    }
    
    /*! checks if time range overlaps */
    __forceinline bool time_range_overlap(const BBox1f& range) const
    {
      if (0.9999f*time_range.upper <= range.lower) return false;
      if (1.0001f*time_range.lower >= range.upper) return false;
      return true;
    }

    /*! returns center for binning */
    __forceinline Vec3fa binCenter() const {
      return center2(bounds());
    }

    /*! returns bounds and centroid used for binning */
    __forceinline void binBoundsAndCenter(BBox3fa& bounds_o, Vec3fa& center_o) const
    {
      bounds_o = bounds();
      center_o = center2(bounds());
    }

    /*! returns the geometry ID */
    __forceinline unsigned int geomID() const { 
      return bbox.lower.a;
    }

    /*! returns the primitive ID */
    __forceinline unsigned int primID() const { 
      return bbox.upper.a;
    }

    /*! returns an size_t sized ID */
    __forceinline size_t ID() const { 
#if defined(__64BIT__)
      return size_t(bbox.lower.u) + (size_t(bbox.upper.u) << 32);
#else
      return size_t(bbox.lower.u);
#endif
    }

    /*! special function for operator< */
    __forceinline uint64_t ID64() const {
      return (((uint64_t)primID()) << 32) + (uint64_t)geomID();
    }
    
    /*! allows sorting the primrefs by ID */
    friend __forceinline bool operator<(const PrimRefMB& p0, const PrimRefMB& p1) {
      return p0.ID64() < p1.ID64();
    }

    /*! Outputs primitive reference to a stream. */
    friend __forceinline embree_ostream operator<<(embree_ostream cout, const PrimRefMB& ref) {
      return cout << "{ bounds = " << ref.bounds() << ", geomID = " << ref.geomID() << ", primID = " << ref.primID() << ", active_segments = " << ref.size() << ",  total_segments = " << ref.totalTimeSegments() << " }";
    }

  public:
    BBox3fa bbox; // bounds, geomID, primID
    unsigned int _activeTimeSegments;
    unsigned int _totalTimeSegments;
    BBox1f time_range; // entire geometry time range
  };

#endif
}
