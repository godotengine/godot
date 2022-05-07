// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primitive.h"

namespace embree
{
  struct Object
  {
    struct Type : public PrimitiveType 
    {
      const char* name() const;
      size_t sizeActive(const char* This) const;
      size_t sizeTotal(const char* This) const;
      size_t getBytes(const char* This) const;
    };
    static Type type;

  public:

    /* primitive supports multiple time segments */
    static const bool singleTimeSegment = false;

    /* Returns maximum number of stored primitives */
    static __forceinline size_t max_size() { return 1; }

    /* Returns required number of primitive blocks for N primitives */
    static __forceinline size_t blocks(size_t N) { return N; }

  public:

    /*! constructs a virtual object */
    Object (unsigned geomID, unsigned primID) 
    : _geomID(geomID), _primID(primID) {}

    __forceinline unsigned geomID() const {
      return _geomID;
    }

    __forceinline unsigned primID() const {
      return _primID;
    }

    /*! fill triangle from triangle list */
    __forceinline void fill(const PrimRef* prims, size_t& i, size_t end, Scene* scene)
    {
      const PrimRef& prim = prims[i]; i++;
      new (this) Object(prim.geomID(), prim.primID());
    }

    /*! fill triangle from triangle list */
    __forceinline LBBox3fa fillMB(const PrimRef* prims, size_t& i, size_t end, Scene* scene, size_t itime)
    {
      const PrimRef& prim = prims[i]; i++;
      const unsigned geomID = prim.geomID();
      const unsigned primID = prim.primID();
      new (this) Object(geomID, primID);
      AccelSet* accel = (AccelSet*) scene->get(geomID);
      return accel->linearBounds(primID,itime);
    }

    /*! fill triangle from triangle list */
    __forceinline LBBox3fa fillMB(const PrimRefMB* prims, size_t& i, size_t end, Scene* scene, const BBox1f time_range)
    {
      const PrimRefMB& prim = prims[i]; i++;
      const unsigned geomID = prim.geomID();
      const unsigned primID = prim.primID();
      new (this) Object(geomID, primID);
      AccelSet* accel = (AccelSet*) scene->get(geomID);
      return accel->linearBounds(primID,time_range);
    }

    /* Updates the primitive */
    __forceinline BBox3fa update(AccelSet* mesh) {
      return mesh->bounds(primID());
    }

  private:
    unsigned int _geomID;  //!< geometry ID
    unsigned int _primID;  //!< primitive ID
  };
}
