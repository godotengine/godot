// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scene_instance.h"
#include "scene.h"
#include "motion_derivative.h"
namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  Instance::Instance (Device* device, Accel* object, unsigned int numTimeSteps)
    : Geometry(device,Geometry::GTY_INSTANCE_CHEAP,1,numTimeSteps)
    , object(object)
    , local2world(nullptr)
  {
    if (object) object->refInc();
    gsubtype = GTY_SUBTYPE_INSTANCE_LINEAR;
    world2local0 = one;
    device->memoryMonitor(numTimeSteps*sizeof(AffineSpace3ff), false);
    local2world = (AffineSpace3ff*) device->malloc(numTimeSteps*sizeof(AffineSpace3ff),16);
    for (size_t i = 0; i < numTimeSteps; i++)
      local2world[i] = one;
    device->memoryMonitor(sizeof(*this), false);
  }

  Instance::~Instance()
  {
    device->free(local2world);
    device->memoryMonitor(-ssize_t(numTimeSteps*sizeof(AffineSpace3ff)), true);
    if (object) object->refDec();
    device->memoryMonitor(-sizeof(*this), false);
  }

  void Instance::setNumTimeSteps (unsigned int numTimeSteps_in)
  {
    if (numTimeSteps_in == numTimeSteps)
      return;

    device->memoryMonitor(numTimeSteps_in*sizeof(AffineSpace3ff), false);
    AffineSpace3ff* local2world2 = (AffineSpace3ff*) device->malloc(numTimeSteps_in*sizeof(AffineSpace3ff),16);

    for (size_t i = 0; i < min(numTimeSteps, numTimeSteps_in); i++) {
      local2world2[i] = local2world[i];
    }

    for (size_t i = numTimeSteps; i < numTimeSteps_in; i++) {
      local2world2[i] = one;
    }

    device->free(local2world);
    device->memoryMonitor(-ssize_t(numTimeSteps*sizeof(AffineSpace3ff)), true);
    local2world = local2world2;

    Geometry::setNumTimeSteps(numTimeSteps_in);
  }

  void Instance::setInstancedScene(const Ref<Scene>& scene)
  {
    if (object) object->refDec();
    object = scene.ptr;
    if (object) object->refInc();
    Geometry::update();
  }

#if 0
  void Instance::preCommit()
  {
#if 0 // disable expensive instance optimization for now
    // decide whether we're an expensive instance or not
    auto numExpensiveGeo =  static_cast<Scene*> (object)->getNumPrimitives(CurveGeometry::geom_type, false)
                          + static_cast<Scene*> (object)->getNumPrimitives(CurveGeometry::geom_type, true)
                          + static_cast<Scene*> (object)->getNumPrimitives(UserGeometry::geom_type, false)
                          + static_cast<Scene*> (object)->getNumPrimitives(UserGeometry::geom_type, true);
    if (numExpensiveGeo > 0) {
      this->gtype = GTY_INSTANCE_EXPENSIVE;
    }
#endif

    Geometry::preCommit();
  }
#endif

  void Instance::addElementsToCount (GeometryCounts & counts) const 
  {
    if (Geometry::GTY_INSTANCE_CHEAP == this->gtype) {
      if (1 == numTimeSteps) {
        counts.numInstancesCheap += numPrimitives;
      } else {
        counts.numMBInstancesCheap += numPrimitives;
      }
    } else {
      if (1 == numTimeSteps) {
        counts.numInstancesExpensive += numPrimitives;
      } else {
        counts.numMBInstancesExpensive += numPrimitives;
      }
    }
  }

  void Instance::setTransform(const AffineSpace3fa& xfm, unsigned int timeStep)
  {
    if (timeStep >= numTimeSteps)
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"invalid timestep");

    local2world[timeStep] = xfm;
    gsubtype = GTY_SUBTYPE_INSTANCE_LINEAR;
    Geometry::update();
  }

  void Instance::setQuaternionDecomposition(const AffineSpace3ff& qd, unsigned int timeStep)
  {
    if (timeStep >= numTimeSteps)
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"invalid timestep");

    local2world[timeStep] = qd;
    gsubtype = GTY_SUBTYPE_INSTANCE_QUATERNION;
    Geometry::update();
  }

  AffineSpace3fa Instance::getTransform(float time)
  {
    if (likely(numTimeSteps <= 1))
      return getLocal2World();
    else
      return getLocal2World(time);
  }

  AffineSpace3fa Instance::getTransform(size_t i, float time)
  {
    if (i != 0)
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "instance has only primitive 0.");

    if (likely(numTimeSteps <= 1))
      return getLocal2World();
    else
      return getLocal2World(time);
  }

  void Instance::setMask (unsigned mask)
  {
    this->mask = mask;
    Geometry::update();
  }

  void Instance::commit()
  {
    if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION))
      world2local0 = rcp(quaternionDecompositionToAffineSpace(local2world[0]));
    else
      world2local0 = rcp(local2world[0]);

    Geometry::commit();
  }

  /* 

     This function calculates the correction for the linear bounds
     bbox0/bbox1 to properly bound the motion obtained when linearly
     blending the transformation and applying the resulting
     transformation to the linearly blended positions
     lerp(xfm0,xfm1,t)*lerp(p0,p1,t). The extrema of the error to the
     linearly blended bounds have to get calculates
     f = lerp(xfm0,xfm1,t)*lerp(p0,p1,t) - lerp(bounds0,bounds1,t). For
     the position where this error gets extreme we have to correct the
     linear bounds. The derivative of this function f can get
     calculates as

     f' = (lerp(A0,A1,t) lerp(p0,p1,t))` - (lerp(bounds0,bounds1,t))`
        = lerp'(A0,A1,t) lerp(p0,p1,t) + lerp(A0,A1,t) lerp'(p0,p1,t) - (bounds1-bounds0)
        = (A1-A0) lerp(p0,p1,t) + lerp(A0,A1,t) (p1-p0) - (bounds1-bounds0)
        = (A1-A0) (p0 + t*(p1-p0)) + (A0 + t*(A1-A0)) (p1-p0) - (bounds1-bounds0)
        = (A1-A0) * p0 + t*(A1-A0)*(p1-p0) + A0*(p1-p0) + t*(A1-A0)*(p1-p0) - (bounds1-bounds0)
        = (A1-A0) * p0 + A0*(p1-p0) - (bounds1-bounds0) + t* ((A1-A0)*(p1-p0) + (A1-A0)*(p1-p0))

   The t value where this function has an extremal point is thus:

    => t = - ((A1-A0) * p0 + A0*(p1-p0) + (bounds1-bounds0)) / (2*(A1-A0)*(p1-p0))
         = (2*A0*p0 - A1*p0 - A0*p1 + (bounds1-bounds0)) / (2*(A1-A0)*(p1-p0))

   */

  BBox3fa boundSegmentLinear(AffineSpace3fa const& xfm0,
                             AffineSpace3fa const& xfm1,
                             BBox3fa const& obbox0,
                             BBox3fa const& obbox1,
                             BBox3fa const& bbox0,
                             BBox3fa const& bbox1,
                             float tmin,
                             float tmax)
  {
    BBox3fa delta(Vec3fa(0.f), Vec3fa(0.f));

    // loop over bounding box corners
    for (int ii = 0; ii < 2; ++ii)
    for (int jj = 0; jj < 2; ++jj)
    for (int kk = 0; kk < 2; ++kk)
    {
      Vec3fa p0(ii == 0 ? obbox0.lower.x : obbox0.upper.x,
                jj == 0 ? obbox0.lower.y : obbox0.upper.y,
                kk == 0 ? obbox0.lower.z : obbox0.upper.z);
      Vec3fa p1(ii == 0 ? obbox1.lower.x : obbox1.upper.x,
                jj == 0 ? obbox1.lower.y : obbox1.upper.y,
                kk == 0 ? obbox1.lower.z : obbox1.upper.z);

      // get extrema of motion of bounding box corner for each dimension
      const Vec3fa denom = 2.0 * xfmVector(xfm0 - xfm1, p0 - p1);
      const Vec3fa nom   = 2.0 * xfmPoint (xfm0, p0) - xfmPoint(xfm0, p1) - xfmPoint(xfm1, p0);
      for (int dim = 0; dim < 3; ++dim)
      {
        if (!(std::abs(denom[dim]) > 0)) continue;

        const float tl = (nom[dim] + (bbox1.lower[dim] - bbox0.lower[dim])) / denom[dim];
        if (tmin <= tl && tl <= tmax) {
          const BBox3fa bt = lerp(bbox0, bbox1, tl);
          const Vec3fa  pt = xfmPoint (lerp(xfm0, xfm1, tl), lerp(p0, p1, tl));
          delta.lower[dim] = std::min(delta.lower[dim], pt[dim] - bt.lower[dim]);
        }
        const float tu = (nom[dim] + (bbox1.upper[dim] - bbox0.upper[dim])) / denom[dim];
        if (tmin <= tu && tu <= tmax) {
          const BBox3fa bt = lerp(bbox0, bbox1, tu);
          const Vec3fa  pt = xfmPoint(lerp(xfm0, xfm1, tu), lerp(p0, p1, tu));
          delta.upper[dim] = std::max(delta.upper[dim], pt[dim] - bt.upper[dim]);
        }
      }
    }
    return delta;
  }

  /* 
     This function calculates the correction for the linear bounds
     bbox0/bbox1 to properly bound the motion obtained by linearly
     blending the quaternion transformations and applying the
     resulting transformation to the linearly blended positions. The
     extrema of the error to the linearly blended bounds has to get
     calclated, the the linear bounds get corrected at the extremal
     points. In difference to the previous function the extremal
     points cannot get calculated analytically, thus we fall back to
     some root solver. 
  */
 
  BBox3fa boundSegmentNonlinear(MotionDerivativeCoefficients const& motionDerivCoeffs,
                                AffineSpace3fa const& xfm0,
                                AffineSpace3fa const& xfm1,
                                BBox3fa const& obbox0,
                                BBox3fa const& obbox1,
                                BBox3fa const& bbox0,
                                BBox3fa const& bbox1,
                                float tmin,
                                float tmax)
  {
    BBox3fa delta(Vec3fa(0.f), Vec3fa(0.f));
    float roots[32];
    unsigned int maxNumRoots = 32;
    unsigned int numRoots;
    const Interval1f interval(tmin, tmax);

    // loop over bounding box corners
    for (int ii = 0; ii < 2; ++ii)
    for (int jj = 0; jj < 2; ++jj)
    for (int kk = 0; kk < 2; ++kk)
    {
      Vec3fa p0(ii == 0 ? obbox0.lower.x : obbox0.upper.x,
                jj == 0 ? obbox0.lower.y : obbox0.upper.y,
                kk == 0 ? obbox0.lower.z : obbox0.upper.z);
      Vec3fa p1(ii == 0 ? obbox1.lower.x : obbox1.upper.x,
                jj == 0 ? obbox1.lower.y : obbox1.upper.y,
                kk == 0 ? obbox1.lower.z : obbox1.upper.z);

      // get extrema of motion of bounding box corner for each dimension
      for (int dim = 0; dim < 3; ++dim)
      {
        MotionDerivative motionDerivative(motionDerivCoeffs, dim, p0, p1);

        numRoots = motionDerivative.findRoots(interval, bbox0.lower[dim] - bbox1.lower[dim], roots, maxNumRoots);
        for (unsigned int r = 0; r < numRoots; ++r) {
          float t = roots[r];
          const BBox3fa bt = lerp(bbox0, bbox1, t);
          const Vec3fa  pt = xfmPoint(slerp(xfm0, xfm1, t), lerp(p0, p1, t));
          delta.lower[dim] = std::min(delta.lower[dim], pt[dim] - bt.lower[dim]);
        }

        numRoots = motionDerivative.findRoots(interval, bbox0.upper[dim] - bbox1.upper[dim], roots, maxNumRoots);
        for (unsigned int r = 0; r < numRoots; ++r) {
          float t = roots[r];
          const BBox3fa bt = lerp(bbox0, bbox1, t);
          const Vec3fa  pt = xfmPoint(slerp(xfm0, xfm1, t), lerp(p0, p1, t));
          delta.upper[dim] = std::max(delta.upper[dim], pt[dim] - bt.upper[dim]);
        }
      }
    }

    return delta;
  }

  BBox3fa Instance::boundSegment(size_t itime,
      BBox3fa const& obbox0, BBox3fa const& obbox1,
      BBox3fa const& bbox0, BBox3fa const& bbox1,
      float tmin, float tmax) const
  {
    if (unlikely(gsubtype == GTY_SUBTYPE_INSTANCE_QUATERNION)) {
      auto const& xfm0 = local2world[itime];
      auto const& xfm1 = local2world[itime+1];
      MotionDerivativeCoefficients motionDerivCoeffs(local2world[itime+0], local2world[itime+1]);
      return boundSegmentNonlinear(motionDerivCoeffs, xfm0, xfm1, obbox0, obbox1, bbox0, bbox1, tmin, tmax);
    } else {
      auto const& xfm0 = local2world[itime];
      auto const& xfm1 = local2world[itime+1];
      return boundSegmentLinear(xfm0, xfm1, obbox0, obbox1, bbox0, bbox1, tmin, tmax);
    }
  }

  LBBox3fa Instance::nonlinearBounds(const BBox1f& time_range_in,
                                     const BBox1f& geom_time_range,
                                     float geom_time_segments) const
  {
    LBBox3fa lbbox = empty;
    /* normalize global time_range_in to local geom_time_range */
    const BBox1f time_range((time_range_in.lower-geom_time_range.lower)/geom_time_range.size(),
                            (time_range_in.upper-geom_time_range.lower)/geom_time_range.size());

    const float lower = time_range.lower*geom_time_segments;
    const float upper = time_range.upper*geom_time_segments;
    const float ilowerf = floor(lower);
    const float iupperf = ceil(upper);
    const float ilowerfc = max(0.0f,ilowerf);
    const float iupperfc = min(iupperf,geom_time_segments);
    const int   ilowerc = (int)ilowerfc;
    const int   iupperc = (int)iupperfc;
    assert(iupperc-ilowerc > 0);

    /* this larger iteration range guarantees that we process borders of geom_time_range is (partially) inside time_range_in */
    const int ilower_iter = max(-1,(int)ilowerf);
    const int iupper_iter = min((int)iupperf,(int)geom_time_segments+1);

    if (iupper_iter-ilower_iter == 1)
    {
      const float f0 = (ilowerc / geom_time_segments - time_range.lower) / time_range.size();
      const float f1 = (iupperc / geom_time_segments - time_range.lower) / time_range.size();

      lbbox.bounds0 = bounds(ilowerc, iupperc, max(0.0f,lower-ilowerfc));
      lbbox.bounds1 = bounds(iupperc, ilowerc, max(0.0f,iupperfc-upper));

      const BBox3fa d = boundSegment(ilowerc, getObjectBounds(ilowerc), getObjectBounds(iupperc),
        lerp(lbbox.bounds0, lbbox.bounds1, f0), lerp(lbbox.bounds0, lbbox.bounds1, f1),
        max(0.f, lower-ilowerfc), 1.f - max(0.f, iupperfc-upper));

      lbbox.bounds0.lower += d.lower; lbbox.bounds1.lower += d.lower;
      lbbox.bounds0.upper += d.upper; lbbox.bounds1.upper += d.upper;
    }
    else
    {
      BBox3fa b0 = bounds(ilowerc, ilowerc+1, lower-ilowerfc);
      BBox3fa b1 = bounds(iupperc, iupperc-1, iupperfc-upper);

      for (int i = ilower_iter+1; i < iupper_iter; i++)
      {
        const float f = (float(i) / geom_time_segments - time_range.lower) / time_range.size();
        const BBox3fa bt = lerp(b0, b1, f);
        const BBox3fa bi = bounds(0, i);
        const Vec3fa dlower = min(bi.lower-bt.lower, Vec3fa(zero));
        const Vec3fa dupper = max(bi.upper-bt.upper, Vec3fa(zero));
        b0.lower += dlower; b1.lower += dlower;
        b0.upper += dupper; b1.upper += dupper;
      }

      BBox3fa delta(Vec3fa(0.f), Vec3fa(0.f));
      for (int i = max(1, ilower_iter+1); i <= min((int)fnumTimeSegments, iupper_iter); i++)
      {
        // compute local times for local itimes
        const float f0 = ((i-1) / geom_time_segments - time_range.lower) / time_range.size();
        const float f1 = ((i  ) / geom_time_segments - time_range.lower) / time_range.size();
        const float tmin = (i == max(1, ilower_iter+1))                           ?       max(0.f, lower-ilowerfc) : 0.f;
        const float tmax = (i == max(1, min((int)fnumTimeSegments, iupper_iter))) ? 1.f - max(0.f, iupperfc-upper) : 1.f;
        const BBox3fa d = boundSegment(i-1, getObjectBounds(i-1), getObjectBounds(i),
          lerp(b0, b1, f0), lerp(b0, b1, f1), tmin, tmax);
        delta.lower = min(delta.lower, d.lower);
        delta.upper = max(delta.upper, d.upper);
      }
      b0.lower += delta.lower; b1.lower += delta.lower;
      b0.upper += delta.upper; b1.upper += delta.upper;

      lbbox.bounds0 = b0;
      lbbox.bounds1 = b1;
    }
    return lbbox;
  }
#endif

  namespace isa
  {
    Instance* createInstance(Device* device) {
      return new InstanceISA(device);
    }
  }
}
