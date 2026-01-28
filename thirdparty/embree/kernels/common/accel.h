// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "ray.h"
#include "point_query.h"
#include "context.h"

namespace embree
{
  class Scene;

  /*! Base class for the acceleration structure data. */
  class AccelData : public RefCount 
  {
    ALIGNED_CLASS_(16);
  public:
    enum Type { TY_UNKNOWN = 0, TY_ACCELN = 1, TY_ACCEL_INSTANCE = 2, TY_BVH4 = 3, TY_BVH8 = 4, TY_GPU = 5 };

  public:
    AccelData (const Type type) 
      : bounds(empty), type(type) {}

    /*! notifies the acceleration structure about the deletion of some geometry */
    virtual void deleteGeometry(size_t geomID) {};
   
    /*! clears the acceleration structure data */
    virtual void clear() = 0;

    /*! returns normal bounds */
    __forceinline BBox3fa getBounds() const {
      return bounds.bounds();
    }

    /*! returns bounds for some time */
    __forceinline BBox3fa getBounds(float t) const {
      return bounds.interpolate(t);
    }

    /*! returns linear bounds */
    __forceinline LBBox3fa getLinearBounds() const {
      return bounds;
    }

    /*! checks if acceleration structure is empty */
    __forceinline bool isEmpty() const {
      return bounds.bounds0.lower.x == float(pos_inf);
    }

  public:
    LBBox3fa bounds; // linear bounds
    Type type;
  };

  /*! Base class for all intersectable and buildable acceleration structures. */
  class Accel : public AccelData
  {
     ALIGNED_CLASS_(16);
  public:

    struct Intersectors;

    /*! Type of collide function */
    typedef void (*CollideFunc)(void* bvh0, void* bvh1, RTCCollideFunc callback, void* userPtr);

    /*! Type of point query function */
    typedef bool(*PointQueryFunc)(Intersectors* This,          /*!< this pointer to accel */
                                  PointQuery* query,        /*!< point query for lookup */
                                  PointQueryContext* context); /*!< point query context */

    /*! Type of intersect function pointer for single rays. */
    typedef void (*IntersectFunc)(Intersectors* This,  /*!< this pointer to accel */
                                  RTCRayHit& ray,      /*!< ray to intersect */
                                  RayQueryContext* context);
    
    /*! Type of intersect function pointer for ray packets of size 4. */
    typedef void (*IntersectFunc4)(const void* valid,  /*!< pointer to valid mask */
                                   Intersectors* This, /*!< this pointer to accel */
                                   RTCRayHit4& ray,    /*!< ray packet to intersect */
                                   RayQueryContext* context);
    
    /*! Type of intersect function pointer for ray packets of size 8. */
    typedef void (*IntersectFunc8)(const void* valid,  /*!< pointer to valid mask */
                                   Intersectors* This, /*!< this pointer to accel */
                                   RTCRayHit8& ray,    /*!< ray packet to intersect */
                                   RayQueryContext* context);
    
    /*! Type of intersect function pointer for ray packets of size 16. */
    typedef void (*IntersectFunc16)(const void* valid,  /*!< pointer to valid mask */
                                    Intersectors* This, /*!< this pointer to accel */
                                    RTCRayHit16& ray,   /*!< ray packet to intersect */
                                    RayQueryContext* context);

    /*! Type of occlusion function pointer for single rays. */
    typedef void (*OccludedFunc) (Intersectors* This, /*!< this pointer to accel */
                                  RTCRay& ray,        /*!< ray to test occlusion */
                                  RayQueryContext* context);
    
    /*! Type of occlusion function pointer for ray packets of size 4. */
    typedef void (*OccludedFunc4) (const void* valid,  /*!< pointer to valid mask */
                                   Intersectors* This, /*!< this pointer to accel */
                                   RTCRay4& ray,       /*!< ray packet to test occlusion. */
                                   RayQueryContext* context);
    
    /*! Type of occlusion function pointer for ray packets of size 8. */
    typedef void (*OccludedFunc8) (const void* valid,  /*!< pointer to valid mask */
                                   Intersectors* This, /*!< this pointer to accel */
                                   RTCRay8& ray,       /*!< ray packet to test occlusion. */
                                   RayQueryContext* context);
    
    /*! Type of occlusion function pointer for ray packets of size 16. */
    typedef void (*OccludedFunc16) (const void* valid,  /*!< pointer to valid mask */
                                    Intersectors* This, /*!< this pointer to accel */
                                    RTCRay16& ray,      /*!< ray packet to test occlusion. */
                                    RayQueryContext* context);

    typedef void (*ErrorFunc) ();

    struct Collider
    {
      Collider (ErrorFunc error = nullptr) 
      : collide((CollideFunc)error), name(nullptr) {}

      Collider (CollideFunc collide, const char* name)
      : collide(collide), name(name) {}

      operator bool() const { return name; }

    public:
      CollideFunc collide;  
      const char* name;
    };
    
    struct Intersector1
    {
      Intersector1 (ErrorFunc error = nullptr)
      : intersect((IntersectFunc)error), occluded((OccludedFunc)error), name(nullptr) {}
      
      Intersector1 (IntersectFunc intersect, OccludedFunc occluded, const char* name)
      : intersect(intersect), occluded(occluded), pointQuery(nullptr), name(name) {}
      
      Intersector1 (IntersectFunc intersect, OccludedFunc occluded, PointQueryFunc pointQuery, const char* name)
      : intersect(intersect), occluded(occluded), pointQuery(pointQuery), name(name) {}

      operator bool() const { return name; }

    public:
      static const char* type;
      IntersectFunc intersect;
      OccludedFunc occluded;
      PointQueryFunc pointQuery;
      const char* name;
    };
    
    struct Intersector4 
    {
      Intersector4 (ErrorFunc error = nullptr)
      : intersect((IntersectFunc4)error), occluded((OccludedFunc4)error), name(nullptr) {}

      Intersector4 (IntersectFunc4 intersect, OccludedFunc4 occluded, const char* name)
      : intersect(intersect), occluded(occluded), name(name) {}

      operator bool() const { return name; }
      
    public:
      static const char* type;
      IntersectFunc4 intersect;
      OccludedFunc4 occluded;
      const char* name;
    };
    
    struct Intersector8 
    {
      Intersector8 (ErrorFunc error = nullptr)
      : intersect((IntersectFunc8)error), occluded((OccludedFunc8)error), name(nullptr) {}

      Intersector8 (IntersectFunc8 intersect, OccludedFunc8 occluded, const char* name)
      : intersect(intersect), occluded(occluded), name(name) {}

      operator bool() const { return name; }
      
    public:
      static const char* type;
      IntersectFunc8 intersect;
      OccludedFunc8 occluded;
      const char* name;
    };
    
    struct Intersector16 
    {
      Intersector16 (ErrorFunc error = nullptr)
      : intersect((IntersectFunc16)error), occluded((OccludedFunc16)error), name(nullptr) {}

      Intersector16 (IntersectFunc16 intersect, OccludedFunc16 occluded, const char* name)
      : intersect(intersect), occluded(occluded), name(name) {}

      operator bool() const { return name; }
      
    public:
      static const char* type;
      IntersectFunc16 intersect;
      OccludedFunc16 occluded;
      const char* name;
    };

    struct Intersectors 
    {
      Intersectors() 
      : ptr(nullptr), leafIntersector(nullptr), collider(nullptr), intersector1(nullptr), intersector4(nullptr), intersector8(nullptr), intersector16(nullptr) {}

      Intersectors (ErrorFunc error) 
      : ptr(nullptr), leafIntersector(nullptr), collider(error), intersector1(error), intersector4(error), intersector8(error), intersector16(error) {}

      void print(size_t ident) 
      {
        if (collider.name) {
          for (size_t i=0; i<ident; i++) std::cout << " ";
          std::cout << "collider  = " << collider.name << std::endl;
        }
        if (intersector1.name) {
          for (size_t i=0; i<ident; i++) std::cout << " ";
          std::cout << "intersector1  = " << intersector1.name << std::endl;
        }
        if (intersector4.name) {
          for (size_t i=0; i<ident; i++) std::cout << " ";
          std::cout << "intersector4  = " << intersector4.name << std::endl;
        }
        if (intersector8.name) {
          for (size_t i=0; i<ident; i++) std::cout << " ";
          std::cout << "intersector8  = " << intersector8.name << std::endl;
        }
        if (intersector16.name) {
          for (size_t i=0; i<ident; i++) std::cout << " ";
          std::cout << "intersector16 = " << intersector16.name << std::endl;
        }
      }

      void select(bool filter)
      {
        if (intersector4_filter) {
          if (filter) intersector4 = intersector4_filter;
          else        intersector4 = intersector4_nofilter;
        }
        if (intersector8_filter) {
          if (filter) intersector8 = intersector8_filter;
          else        intersector8 = intersector8_nofilter;
        }
        if (intersector16_filter) {
          if (filter) intersector16 = intersector16_filter;
          else         intersector16 = intersector16_nofilter;
        }
      }

      __forceinline bool pointQuery (PointQuery* query, PointQueryContext* context) {
        assert(intersector1.pointQuery);
        return intersector1.pointQuery(this,query,context);
      }

      /*! collides two scenes */
      __forceinline void collide (Accel* scene0, Accel* scene1, RTCCollideFunc callback, void* userPtr) {
        assert(collider.collide);
        collider.collide(scene0->intersectors.ptr,scene1->intersectors.ptr,callback,userPtr);
      }

      /*! Intersects a single ray with the scene. */
      __forceinline void intersect (RTCRayHit& ray, RayQueryContext* context) {
        assert(intersector1.intersect);
        intersector1.intersect(this,ray,context);
      }

      /*! Intersects a packet of 4 rays with the scene. */
      __forceinline void intersect4 (const void* valid, RTCRayHit4& ray, RayQueryContext* context) {
        assert(intersector4.intersect);
        intersector4.intersect(valid,this,ray,context);
      }
      
      /*! Intersects a packet of 8 rays with the scene. */
      __forceinline void intersect8 (const void* valid, RTCRayHit8& ray, RayQueryContext* context) {
        assert(intersector8.intersect);
        intersector8.intersect(valid,this,ray,context);
      }
      
      /*! Intersects a packet of 16 rays with the scene. */
      __forceinline void intersect16 (const void* valid, RTCRayHit16& ray, RayQueryContext* context) {
        assert(intersector16.intersect);
        intersector16.intersect(valid,this,ray,context);
      }

      /*! Intersects a packet of 4 rays with the scene. */
      __forceinline void intersect (const void* valid, RTCRayHit4& ray, RayQueryContext* context) {
        assert(intersector4.intersect);
        intersector4.intersect(valid,this,ray,context);
      }
      
      /*! Intersects a packet of 8 rays with the scene. */
      __forceinline void intersect (const void* valid, RTCRayHit8& ray, RayQueryContext* context) {
        assert(intersector8.intersect);
        intersector8.intersect(valid,this,ray,context);
      }
      
      /*! Intersects a packet of 16 rays with the scene. */
      __forceinline void intersect (const void* valid, RTCRayHit16& ray, RayQueryContext* context) {
        assert(intersector16.intersect);
        intersector16.intersect(valid,this,ray,context);
      }
      
#if defined(__SSE__) || defined(__ARM_NEON)
      __forceinline void intersect(const vbool4& valid, RayHitK<4>& ray, RayQueryContext* context) {
        const vint<4> mask = valid.mask32();
        intersect4(&mask,(RTCRayHit4&)ray,context);
      }
#endif
#if defined(__AVX__)
      __forceinline void intersect(const vbool8& valid, RayHitK<8>& ray, RayQueryContext* context) {
        const vint<8> mask = valid.mask32();
        intersect8(&mask,(RTCRayHit8&)ray,context);
      }
#endif
#if defined(__AVX512F__)
      __forceinline void intersect(const vbool16& valid, RayHitK<16>& ray, RayQueryContext* context) {
        const vint<16> mask = valid.mask32();
        intersect16(&mask,(RTCRayHit16&)ray,context);
      }
#endif
      
      /*! Tests if single ray is occluded by the scene. */
      __forceinline void occluded (RTCRay& ray, RayQueryContext* context) {
        assert(intersector1.occluded);
        intersector1.occluded(this,ray,context);
      }
      
      /*! Tests if a packet of 4 rays is occluded by the scene. */
      __forceinline void occluded4 (const void* valid, RTCRay4& ray, RayQueryContext* context) {
        assert(intersector4.occluded);
        intersector4.occluded(valid,this,ray,context);
      }
      
      /*! Tests if a packet of 8 rays is occluded by the scene. */
      __forceinline void occluded8 (const void* valid, RTCRay8& ray, RayQueryContext* context) {
        assert(intersector8.occluded);
        intersector8.occluded(valid,this,ray,context);
      }
      
      /*! Tests if a packet of 16 rays is occluded by the scene. */
      __forceinline void occluded16 (const void* valid, RTCRay16& ray, RayQueryContext* context) {
        assert(intersector16.occluded);
        intersector16.occluded(valid,this,ray,context);
      }

      /*! Tests if a packet of 4 rays is occluded by the scene. */
      __forceinline void occluded (const void* valid, RTCRay4& ray, RayQueryContext* context) {
        assert(intersector4.occluded);
        intersector4.occluded(valid,this,ray,context);
      }
      
      /*! Tests if a packet of 8 rays is occluded by the scene. */
      __forceinline void occluded (const void* valid, RTCRay8& ray, RayQueryContext* context) {
        assert(intersector8.occluded);
        intersector8.occluded(valid,this,ray,context);
      }
      
      /*! Tests if a packet of 16 rays is occluded by the scene. */
      __forceinline void occluded (const void* valid, RTCRay16& ray, RayQueryContext* context) {
        assert(intersector16.occluded);
        intersector16.occluded(valid,this,ray,context);
      }
      
#if defined(__SSE__) || defined(__ARM_NEON)
      __forceinline void occluded(const vbool4& valid, RayK<4>& ray, RayQueryContext* context) {
        const vint<4> mask = valid.mask32();
        occluded4(&mask,(RTCRay4&)ray,context);
      }
#endif
#if defined(__AVX__)
      __forceinline void occluded(const vbool8& valid, RayK<8>& ray, RayQueryContext* context) {
        const vint<8> mask = valid.mask32();
        occluded8(&mask,(RTCRay8&)ray,context);
      }
#endif
#if defined(__AVX512F__)
      __forceinline void occluded(const vbool16& valid, RayK<16>& ray, RayQueryContext* context) {
        const vint<16> mask = valid.mask32();
        occluded16(&mask,(RTCRay16&)ray,context);
      }
#endif

      /*! Tests if single ray is occluded by the scene. */
      __forceinline void intersect(RTCRay& ray, RayQueryContext* context) {
        occluded(ray, context);
      }

      /*! Tests if a packet of K rays is occluded by the scene. */
      template<int K>
      __forceinline void intersect(const vbool<K>& valid, RayK<K>& ray, RayQueryContext* context) {
        occluded(valid, ray, context);
      }

      
    public:
      AccelData* ptr;
      void* leafIntersector;
      Collider collider;
      Intersector1 intersector1;
      Intersector4 intersector4;
      Intersector4 intersector4_filter;
      Intersector4 intersector4_nofilter;
      Intersector8 intersector8;
      Intersector8 intersector8_filter;
      Intersector8 intersector8_nofilter;
      Intersector16 intersector16;
      Intersector16 intersector16_filter;
      Intersector16 intersector16_nofilter;
    };
  
  public:

    /*! Construction */
    Accel (const AccelData::Type type) 
      : AccelData(type) {}
    
    /*! Construction */
    Accel (const AccelData::Type type, const Intersectors& intersectors) 
      : AccelData(type), intersectors(intersectors) {}

    /*! Virtual destructor */
    virtual ~Accel() {}

    /*! makes the acceleration structure immutable */
    virtual void immutable () {}
    
    /*! build acceleration structure */
    virtual void build () = 0;

  public:
    Intersectors intersectors;
  };

#define DEFINE_COLLIDER(symbol,collider)                                \
  Accel::Collider symbol() {                                            \
    return Accel::Collider((Accel::CollideFunc)collider::collide,       \
                           TOSTRING(isa) "::" TOSTRING(symbol));        \
  }

#define DEFINE_INTERSECTOR1(symbol,intersector)                               \
  Accel::Intersector1 symbol() {                                              \
    return Accel::Intersector1((Accel::IntersectFunc )intersector::intersect, \
                               (Accel::OccludedFunc  )intersector::occluded,  \
                               (Accel::PointQueryFunc)intersector::pointQuery,\
                               TOSTRING(isa) "::" TOSTRING(symbol));          \
  }
  
#define DEFINE_INTERSECTOR4(symbol,intersector)                               \
  Accel::Intersector4 symbol() {                                              \
    return Accel::Intersector4((Accel::IntersectFunc4)intersector::intersect, \
                               (Accel::OccludedFunc4)intersector::occluded,   \
                               TOSTRING(isa) "::" TOSTRING(symbol));          \
  }
  
#define DEFINE_INTERSECTOR8(symbol,intersector)                               \
  Accel::Intersector8 symbol() {                                              \
    return Accel::Intersector8((Accel::IntersectFunc8)intersector::intersect, \
                               (Accel::OccludedFunc8)intersector::occluded,   \
                               TOSTRING(isa) "::" TOSTRING(symbol));          \
  }

#define DEFINE_INTERSECTOR16(symbol,intersector)                                \
  Accel::Intersector16 symbol() {                                               \
    return Accel::Intersector16((Accel::IntersectFunc16)intersector::intersect, \
                                (Accel::OccludedFunc16)intersector::occluded,   \
                                TOSTRING(isa) "::" TOSTRING(symbol));           \
  }
}
