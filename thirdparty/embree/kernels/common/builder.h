// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "accel.h"

namespace embree
{  
#define MODE_HIGH_QUALITY (1<<8)

  /*! virtual interface for all hierarchy builders */
  class Builder : public RefCount {
  public:

    static const size_t DEFAULT_SINGLE_THREAD_THRESHOLD = 1024;

    /*! initiates the hierarchy builder */
    virtual void build() = 0;

    /*! notifies the builder about the deletion of some geometry */
    virtual void deleteGeometry(size_t geomID) {};

    /*! clears internal builder state */
    virtual void clear() = 0;
  };

  /*! virtual interface for progress monitor class */
  struct BuildProgressMonitor {
    virtual void operator() (size_t dn) const = 0;
  };

  /*! build the progress monitor interface from a closure */
  template<typename Closure>
    struct ProgressMonitorClosure : BuildProgressMonitor
  {
  public:
    ProgressMonitorClosure (const Closure& closure) : closure(closure) {}
    void operator() (size_t dn) const { closure(dn); }
  private:
    const Closure closure;
  };
  template<typename Closure> __forceinline const ProgressMonitorClosure<Closure> BuildProgressMonitorFromClosure(const Closure& closure) {
    return ProgressMonitorClosure<Closure>(closure);
  }

  struct LineSegments;
  struct TriangleMesh;
  struct QuadMesh;
  struct UserGeometry;

  class Scene;

  typedef void (*createLineSegmentsAccelTy)(Scene* scene, LineSegments* mesh, AccelData*& accel, Builder*& builder);
  typedef void (*createTriangleMeshAccelTy)(Scene* scene, unsigned int geomID, AccelData*& accel, Builder*& builder);
  typedef void (*createQuadMeshAccelTy)(Scene* scene, unsigned int geomID, AccelData*& accel, Builder*& builder);
  typedef void (*createUserGeometryAccelTy)(Scene* scene, unsigned int geomID, AccelData*& accel, Builder*& builder);

}
