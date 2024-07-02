// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define RTHWIF_EXPORT_API

#include "rtbuild.h"
#include "qbvh6_builder_sah.h"

// get definition of debug extension
#if defined(EMBREE_SYCL_ALLOC_DISPATCH_GLOBALS)
#include "../../level_zero/ze_wrapper.h"
#endif

namespace embree
{
  using namespace embree::isa;

  static tbb::task_arena g_arena(tbb::this_task_arena::max_concurrency(),tbb::this_task_arena::max_concurrency());
  
  inline ze_rtas_triangle_indices_uint32_exp_t getPrimitive(const ze_rtas_builder_triangles_geometry_info_exp_t* geom, uint32_t primID) {
    assert(primID < geom->triangleCount);
    return *(ze_rtas_triangle_indices_uint32_exp_t*)((char*)geom->pTriangleBuffer + uint64_t(primID)*geom->triangleStride);
  }
  
  inline Vec3f getVertex(const ze_rtas_builder_triangles_geometry_info_exp_t* geom, uint32_t vertexID) {
    assert(vertexID < geom->vertexCount);
    return *(Vec3f*)((char*)geom->pVertexBuffer + uint64_t(vertexID)*geom->vertexStride);
  }
  
  inline ze_rtas_quad_indices_uint32_exp_t getPrimitive(const ze_rtas_builder_quads_geometry_info_exp_t* geom, uint32_t primID) {
    assert(primID < geom->quadCount);
    return *(ze_rtas_quad_indices_uint32_exp_t*)((char*)geom->pQuadBuffer + uint64_t(primID)*geom->quadStride);
  }
  
  inline Vec3f getVertex(const ze_rtas_builder_quads_geometry_info_exp_t* geom, uint32_t vertexID) {
    assert(vertexID < geom->vertexCount);
    return *(Vec3f*)((char*)geom->pVertexBuffer + uint64_t(vertexID)*geom->vertexStride);
  }

  inline AffineSpace3fa getTransform(const ze_rtas_builder_instance_geometry_info_exp_t* geom)
  {
    switch (geom->transformFormat)
    {
    case ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_COLUMN_MAJOR: {
      const ze_rtas_transform_float3x4_column_major_exp_t* xfm = (const ze_rtas_transform_float3x4_column_major_exp_t*) geom->pTransform;
      return {
        { xfm->vx_x, xfm->vx_y, xfm->vx_z },
        { xfm->vy_x, xfm->vy_y, xfm->vy_z },
        { xfm->vz_x, xfm->vz_y, xfm->vz_z },
        { xfm-> p_x, xfm-> p_y, xfm-> p_z }
      };
    }
    case ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ALIGNED_COLUMN_MAJOR: {
      const ze_rtas_transform_float3x4_aligned_column_major_exp_t* xfm = (const ze_rtas_transform_float3x4_aligned_column_major_exp_t*) geom->pTransform;
      return {
        { xfm->vx_x, xfm->vx_y, xfm->vx_z },
        { xfm->vy_x, xfm->vy_y, xfm->vy_z },
        { xfm->vz_x, xfm->vz_y, xfm->vz_z },
        { xfm-> p_x, xfm-> p_y, xfm-> p_z }
      };
    }
    case ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ROW_MAJOR: {
      const ze_rtas_transform_float3x4_row_major_exp_t* xfm = (const ze_rtas_transform_float3x4_row_major_exp_t*) geom->pTransform;
      return {
        { xfm->vx_x, xfm->vx_y, xfm->vx_z },
        { xfm->vy_x, xfm->vy_y, xfm->vy_z },
        { xfm->vz_x, xfm->vz_y, xfm->vz_z },
        { xfm-> p_x, xfm-> p_y, xfm-> p_z }
      };
    }
    default:
      throw std::runtime_error("invalid transform format");
    }
  }
  
  inline void verifyGeometryDesc(const ze_rtas_builder_triangles_geometry_info_exp_t* geom)
  {
    if (geom->triangleFormat != ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32)
      throw std::runtime_error("triangle format must be ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32");
    
    if (geom->vertexFormat != ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3)
      throw std::runtime_error("vertex format must be ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3");
 
    if (geom->triangleCount && geom->pTriangleBuffer == nullptr) throw std::runtime_error("no triangle buffer specified");
    if (geom->vertexCount   && geom->pVertexBuffer   == nullptr) throw std::runtime_error("no vertex buffer specified");
  }

  inline void verifyGeometryDesc(const ze_rtas_builder_quads_geometry_info_exp_t* geom)
  {
    if (geom->quadFormat != ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32)
      throw std::runtime_error("quad format must be ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32");
    
    if (geom->vertexFormat != ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3)
      throw std::runtime_error("vertex format must be ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3");
 
    if (geom->quadCount   && geom->pQuadBuffer   == nullptr) throw std::runtime_error("no quad buffer specified");
    if (geom->vertexCount && geom->pVertexBuffer == nullptr) throw std::runtime_error("no vertex buffer specified");
  }

  inline void verifyGeometryDesc(const ze_rtas_builder_procedural_geometry_info_exp_t* geom)
  {
    if (geom->primCount   && geom->pfnGetBoundsCb == nullptr) throw std::runtime_error("no bounds function specified");
    if (geom->reserved != 0) throw std::runtime_error("reserved value must be zero");
  }

  inline void verifyGeometryDesc(const ze_rtas_builder_instance_geometry_info_exp_t* geom)
  {
    if (geom->pTransform == nullptr) throw std::runtime_error("no instance transformation specified");
    if (geom->pBounds == nullptr) throw std::runtime_error("no acceleration structure bounds specified");
    if (geom->pAccelerationStructure == nullptr) throw std::runtime_error("no acceleration structure to instanciate specified");
  }

  inline bool buildBounds(const ze_rtas_builder_triangles_geometry_info_exp_t* geom, uint32_t primID, BBox3fa& bbox, void* buildUserPtr)
  {
    if (primID >= geom->triangleCount) return false;
    const ze_rtas_triangle_indices_uint32_exp_t tri = getPrimitive(geom,primID);
    if (unlikely(tri.v0 >= geom->vertexCount)) return false;
    if (unlikely(tri.v1 >= geom->vertexCount)) return false;
    if (unlikely(tri.v2 >= geom->vertexCount)) return false;
    
    const Vec3f p0 = getVertex(geom,tri.v0);
    const Vec3f p1 = getVertex(geom,tri.v1);
    const Vec3f p2 = getVertex(geom,tri.v2);
    if (unlikely(!isvalid(p0))) return false;
    if (unlikely(!isvalid(p1))) return false;
    if (unlikely(!isvalid(p2))) return false;
    
    bbox = BBox3fa(min(p0,p1,p2),max(p0,p1,p2));
    return true;
  }

  inline bool buildBounds(const ze_rtas_builder_quads_geometry_info_exp_t* geom, uint32_t primID, BBox3fa& bbox, void* buildUserPtr)
  {
    if (primID >= geom->quadCount) return false;
    const ze_rtas_quad_indices_uint32_exp_t tri = getPrimitive(geom,primID);
    if (unlikely(tri.v0 >= geom->vertexCount)) return false;
    if (unlikely(tri.v1 >= geom->vertexCount)) return false;
    if (unlikely(tri.v2 >= geom->vertexCount)) return false;
    if (unlikely(tri.v3 >= geom->vertexCount)) return false;
    
    const Vec3f p0 = getVertex(geom,tri.v0);
    const Vec3f p1 = getVertex(geom,tri.v1);
    const Vec3f p2 = getVertex(geom,tri.v2);
    const Vec3f p3 = getVertex(geom,tri.v3);
    if (unlikely(!isvalid(p0))) return false;
    if (unlikely(!isvalid(p1))) return false;
    if (unlikely(!isvalid(p2))) return false;
    if (unlikely(!isvalid(p3))) return false;
    
    bbox = BBox3fa(min(p0,p1,p2,p3),max(p0,p1,p2,p3));
    return true;
  }

  inline bool buildBounds(const ze_rtas_builder_procedural_geometry_info_exp_t* geom, uint32_t primID, BBox3fa& bbox, void* buildUserPtr)
  {
    if (primID >= geom->primCount) return false;
    if (geom->pfnGetBoundsCb == nullptr) return false;

    BBox3f bounds;
    ze_rtas_geometry_aabbs_exp_cb_params_t params = { ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS };
    params.primID = primID;
    params.primIDCount = 1;
    params.pGeomUserPtr = geom->pGeomUserPtr;
    params.pBuildUserPtr = buildUserPtr;
    params.pBoundsOut = (ze_rtas_aabb_exp_t*) &bounds;
    (geom->pfnGetBoundsCb)(&params);
    
    if (unlikely(!isvalid(bounds.lower))) return false;
    if (unlikely(!isvalid(bounds.upper))) return false;
    if (unlikely(bounds.empty())) return false;
    
    bbox = (BBox3f&) bounds;
    return true;
  }

  inline bool buildBounds(const ze_rtas_builder_instance_geometry_info_exp_t* geom, uint32_t primID, BBox3fa& bbox, void* buildUserPtr)
  {
    if (primID >= 1) return false;
    if (geom->pAccelerationStructure == nullptr) return false;
    if (geom->pTransform == nullptr) return false;
    
    const AffineSpace3fa local2world = getTransform(geom);
    const Vec3fa lower(geom->pBounds->lower.x,geom->pBounds->lower.y,geom->pBounds->lower.z);
    const Vec3fa upper(geom->pBounds->upper.x,geom->pBounds->upper.y,geom->pBounds->upper.z);
    const BBox3fa bounds = xfmBounds(local2world,BBox3fa(lower,upper));
     
    if (unlikely(!isvalid(bounds.lower))) return false;
    if (unlikely(!isvalid(bounds.upper))) return false;
    if (unlikely(bounds.empty())) return false;
    
    bbox = bounds;
    return true;
  }

  template<typename GeometryType>
  PrimInfo createGeometryPrimRefArray(const GeometryType* geom, void* buildUserPtr, evector<PrimRef>& prims, const range<size_t>& r, size_t k, unsigned int geomID)
  {
    PrimInfo pinfo(empty);
    for (uint32_t primID=r.begin(); primID<r.end(); primID++)
    {
      BBox3fa bounds = empty;
      if (!buildBounds(geom,primID,bounds,buildUserPtr)) continue;
      const PrimRef prim(bounds,geomID,primID);
      pinfo.add_center2(prim);
      prims[k++] = prim;
    }
    return pinfo;
  }
  
  typedef struct _zet_base_desc_t
  {
    /** [in] type of this structure */
    ze_structure_type_t stype;
    
    /** [in,out][optional] must be null or a pointer to an extension-specific structure */
    const void* pNext;
    
  } zet_base_desc_t_;

  #define VALIDATE(arg) \
  {\
  ze_result_t result = validate(arg);\
  if (result != ZE_RESULT_SUCCESS) return result; \
  }

#define VALIDATE_PTR(arg)                       \
  {                                                                     \
    if ((arg) == nullptr) return ZE_RESULT_ERROR_INVALID_NULL_POINTER; \
  }                                                                     \

   ze_result_t validate(ze_driver_handle_t hDriver)
  {
    if (hDriver == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    
    return ZE_RESULT_SUCCESS;
  }

  ze_result_t validate(ze_device_handle_t hDevice)
  {
    if (hDevice == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    
    return ZE_RESULT_SUCCESS;
  }
  
  bool checkDescChain(zet_base_desc_t_* desc)
  {
    /* supporting maximal 1024 to also detect cycles */
    for (size_t i=0; i<1024; i++) {
      if (desc->pNext == nullptr) return true;
      desc = (zet_base_desc_t_*) desc->pNext;
    }
    return false;
  }

  struct ze_rtas_builder
  {
    ze_rtas_builder () {
    }
    
    ~ze_rtas_builder() {
      magick = 0x0;
    }

    bool verify() const {
      return magick == MAGICK;
    }
    
    enum { MAGICK = 0x45FE67E1 };
    uint32_t magick = MAGICK;
  };

  ze_result_t validate(ze_rtas_builder_exp_handle_t hBuilder)
  {
    if (hBuilder == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    
    if (!((ze_rtas_builder*)hBuilder)->verify())
      return ZE_RESULT_ERROR_INVALID_ARGUMENT;

    return ZE_RESULT_SUCCESS;
  }

  struct ze_rtas_parallel_operation_t
  {
    ze_rtas_parallel_operation_t() {
    }

    ~ze_rtas_parallel_operation_t() {
      magick = 0x0;
    }

    ze_result_t verify() const
    {
      if (magick != MAGICK)
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;

      return ZE_RESULT_SUCCESS;
    }
    
    enum { MAGICK = 0xE84567E1 };
    uint32_t magick = MAGICK;
    std::atomic<bool> object_in_use = false;
    ze_result_t errorCode = ZE_RESULT_SUCCESS;
    tbb::task_group group;
  };

  ze_result_t validate(ze_rtas_parallel_operation_exp_handle_t hParallelOperation)
  {
    if (hParallelOperation == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    
    return ((ze_rtas_parallel_operation_t*)hParallelOperation)->verify();
  }

  ze_result_t validate(const ze_rtas_builder_exp_desc_t* pDescriptor)
  {
    if (pDescriptor == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_POINTER;

    if (pDescriptor->stype != ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    if (!checkDescChain((zet_base_desc_t_*)pDescriptor))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    if (uint32_t(ZE_RTAS_BUILDER_EXP_VERSION_CURRENT) < uint32_t(pDescriptor->builderVersion))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;
    
    return ZE_RESULT_SUCCESS;
  }

  ze_result_t validate(ze_rtas_device_exp_properties_t* pProperties)
  { 
    if (pProperties == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_POINTER;

    if (pProperties->stype != ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;
    
    if (!checkDescChain((zet_base_desc_t_*)pProperties))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;
    
    return ZE_RESULT_SUCCESS;
  }

  ze_result_t validate(ze_rtas_format_exp_t rtasFormat)
  {
    if (rtasFormat == ZE_RTAS_FORMAT_EXP_INVALID)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;
      
    if (uint32_t(rtasFormat) > uint32_t(ZE_RTAS_DEVICE_FORMAT_EXP_VERSION_MAX))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    return ZE_RESULT_SUCCESS;
  }
  
  ze_result_t validate(const ze_rtas_builder_build_op_exp_desc_t* args)
  {
    /* check for valid pointers */
    if (args == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_POINTER;

    /* check if input descriptor has proper type */
    if (args->stype != ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    /* check valid pNext chain */
    if (!checkDescChain((zet_base_desc_t_*)args))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    /* check if acceleration structure format is supported */
    VALIDATE(args->rtasFormat);

    /* check for valid geometries array */
    if (args->ppGeometries == nullptr && args->numGeometries > 0)
      return ZE_RESULT_ERROR_INVALID_NULL_POINTER;

    /* validate that number of geometries are in range */
    if (args->numGeometries > 0x00FFFFFF)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    /* validate build quality */
    if (args->buildQuality < 0 || ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH < args->buildQuality)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    /* validate build flags */
    if (args->buildFlags >= (ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION<<1))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;
    
    return ZE_RESULT_SUCCESS;
  }

  ze_result_t validate(ze_rtas_builder_exp_properties_t* pProp)
  {
    /* check for valid pointers */
    if (pProp == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    
    /* check if return property has proper type */
    if (pProp->stype != ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    /* check valid pNext chain */
    if (!checkDescChain((zet_base_desc_t_*)pProp))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    return ZE_RESULT_SUCCESS;
  }

  ze_result_t validate(ze_rtas_parallel_operation_exp_properties_t* pProperties)
  {
    /* check for valid pointer */
    if (pProperties == nullptr)
      return ZE_RESULT_ERROR_INVALID_NULL_POINTER;

    /* check for proper property */
    if (pProperties->stype != ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES)
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    /* check valid pNext chain */
    if (!checkDescChain((zet_base_desc_t_*)pProperties))
      return ZE_RESULT_ERROR_INVALID_ENUMERATION;

    return ZE_RESULT_SUCCESS;
  }
  
  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASBuilderCreateExpImpl(ze_driver_handle_t hDriver, const ze_rtas_builder_exp_desc_t *pDescriptor, ze_rtas_builder_exp_handle_t *phBuilder)
  {
    /* input validation */
    VALIDATE(hDriver);
    VALIDATE(pDescriptor);
    VALIDATE_PTR(phBuilder);

    *phBuilder = (ze_rtas_builder_exp_handle_t) new ze_rtas_builder();
    return ZE_RESULT_SUCCESS;
  }

  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASBuilderDestroyExpImpl(ze_rtas_builder_exp_handle_t hBuilder)
  {
    VALIDATE(hBuilder);
    delete (ze_rtas_builder*) hBuilder;
    return ZE_RESULT_SUCCESS;
  }

  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeDriverRTASFormatCompatibilityCheckExpImpl( ze_driver_handle_t hDriver,
                                                                                        const ze_rtas_format_exp_t accelFormat,
                                                                                        const ze_rtas_format_exp_t otherAccelFormat )
  {
    /* input validation */
    VALIDATE(hDriver);
    VALIDATE(accelFormat);
    VALIDATE(otherAccelFormat);

    /* check if rtas formats are compatible */
    if (accelFormat == otherAccelFormat)
      return ZE_RESULT_SUCCESS;

    /* report incompatible format */
    return ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE;
  }

  uint32_t getNumPrimitives(const ze_rtas_builder_geometry_info_exp_t* geom)
  {
    switch (geom->geometryType) {
    case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES  : return ((ze_rtas_builder_triangles_geometry_info_exp_t*) geom)->triangleCount;
    case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL : return ((ze_rtas_builder_procedural_geometry_info_exp_t*) geom)->primCount;
    case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS      : return ((ze_rtas_builder_quads_geometry_info_exp_t*) geom)->quadCount;
    case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE   : return 1;
    default                              : return 0;
    };
  }
  
  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASBuilderGetBuildPropertiesExpImpl(ze_rtas_builder_exp_handle_t hBuilder,
                                                                                  const ze_rtas_builder_build_op_exp_desc_t* args,
                                                                                  ze_rtas_builder_exp_properties_t* pProp)
  {
    /* input validation */
    VALIDATE(hBuilder);
    VALIDATE(args);
    VALIDATE(pProp);

    const ze_rtas_builder_geometry_info_exp_t** geometries = args->ppGeometries;
    const size_t numGeometries = args->numGeometries;

    auto getSize = [&](uint32_t geomID) -> size_t {
      const ze_rtas_builder_geometry_info_exp_t* geom = geometries[geomID];
      if (geom == nullptr) return 0;
      return getNumPrimitives(geom);
    };
    
    auto getType = [&](unsigned int geomID)
    {
      const ze_rtas_builder_geometry_info_exp_t* geom = geometries[geomID];
      assert(geom);
      switch (geom->geometryType) {
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES : return QBVH6BuilderSAH::TRIANGLE;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS: return QBVH6BuilderSAH::QUAD;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL: return QBVH6BuilderSAH::PROCEDURAL;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE: return QBVH6BuilderSAH::INSTANCE;
      default: throw std::runtime_error("invalid geometry type");
      };
    };

    /* query memory requirements from builder */
    size_t expectedBytes = 0;
    size_t worstCaseBytes = 0;
    size_t scratchBytes = 0;
    QBVH6BuilderSAH::estimateSize(numGeometries, getSize, getType, args->rtasFormat, args->buildQuality, args->buildFlags, expectedBytes, worstCaseBytes, scratchBytes);
    
    /* fill return struct */
    pProp->flags = 0;
    pProp->rtasBufferSizeBytesExpected = expectedBytes;
    pProp->rtasBufferSizeBytesMaxRequired = worstCaseBytes;
    pProp->scratchBufferSizeBytes = scratchBytes;
    return ZE_RESULT_SUCCESS;
  }
  
  ze_result_t zeRTASBuilderBuildExpBody(const ze_rtas_builder_build_op_exp_desc_t* args,
                                            void *pScratchBuffer, size_t scratchBufferSizeBytes,
                                            void *pRtasBuffer, size_t rtasBufferSizeBytes,
                                            void *pBuildUserPtr, ze_rtas_aabb_exp_t *pBounds, size_t *pRtasBufferSizeBytes) try
  {
    const ze_rtas_builder_geometry_info_exp_t** geometries = args->ppGeometries;
    const uint32_t numGeometries = args->numGeometries;

    /* verify input descriptors */
    parallel_for(numGeometries,[&](uint32_t geomID) {
      const ze_rtas_builder_geometry_info_exp_t* geom = geometries[geomID];
      if (geom == nullptr) return;
      
      switch (geom->geometryType) {
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES  : verifyGeometryDesc((ze_rtas_builder_triangles_geometry_info_exp_t*)geom); break;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS      : verifyGeometryDesc((ze_rtas_builder_quads_geometry_info_exp_t*    )geom); break;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL : verifyGeometryDesc((ze_rtas_builder_procedural_geometry_info_exp_t*)geom); break;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE   : verifyGeometryDesc((ze_rtas_builder_instance_geometry_info_exp_t* )geom); break;
      default: throw std::runtime_error("invalid geometry type");
      };
    });
    
    auto getSize = [&](uint32_t geomID) -> size_t {
      const ze_rtas_builder_geometry_info_exp_t* geom = geometries[geomID];
      if (geom == nullptr) return 0;
      return getNumPrimitives(geom);
    };
    
    auto getType = [&](unsigned int geomID)
    {
      const ze_rtas_builder_geometry_info_exp_t* geom = geometries[geomID];
      assert(geom);
      switch (geom->geometryType) {
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES : return QBVH6BuilderSAH::TRIANGLE;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS: return QBVH6BuilderSAH::QUAD;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL: return QBVH6BuilderSAH::PROCEDURAL;
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE: return QBVH6BuilderSAH::INSTANCE;
      default: throw std::runtime_error("invalid geometry type");
      };
    };
    
    auto createPrimRefArray = [&] (evector<PrimRef>& prims, BBox1f time_range, const range<size_t>& r, size_t k, unsigned int geomID) -> PrimInfo
    {
      const ze_rtas_builder_geometry_info_exp_t* geom = geometries[geomID];
      assert(geom);

      switch (geom->geometryType) {
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES  : return createGeometryPrimRefArray((ze_rtas_builder_triangles_geometry_info_exp_t*)geom,pBuildUserPtr,prims,r,k,geomID);
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS      : return createGeometryPrimRefArray((ze_rtas_builder_quads_geometry_info_exp_t*    )geom,pBuildUserPtr,prims,r,k,geomID);
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL: return createGeometryPrimRefArray((ze_rtas_builder_procedural_geometry_info_exp_t*)geom,pBuildUserPtr,prims,r,k,geomID);
      case ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE: return createGeometryPrimRefArray((ze_rtas_builder_instance_geometry_info_exp_t* )geom,pBuildUserPtr,prims,r,k,geomID);
      default: throw std::runtime_error("invalid geometry type");
      };
    };

    auto convertGeometryFlags = [&] (ze_rtas_builder_packed_geometry_exp_flags_t flags) -> GeometryFlags {
      return (flags & ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_NON_OPAQUE) ? GeometryFlags::NONE : GeometryFlags::OPAQUE;
    };
    
    auto getTriangle = [&](unsigned int geomID, unsigned int primID)
    {
      const ze_rtas_builder_triangles_geometry_info_exp_t* geom = (const ze_rtas_builder_triangles_geometry_info_exp_t*) geometries[geomID];
      assert(geom);
      
      const ze_rtas_triangle_indices_uint32_exp_t tri = getPrimitive(geom,primID);
      if (unlikely(tri.v0 >= geom->vertexCount)) return QBVH6BuilderSAH::Triangle();
      if (unlikely(tri.v1 >= geom->vertexCount)) return QBVH6BuilderSAH::Triangle();
      if (unlikely(tri.v2 >= geom->vertexCount)) return QBVH6BuilderSAH::Triangle();
      
      const Vec3f p0 = getVertex(geom,tri.v0);
      const Vec3f p1 = getVertex(geom,tri.v1);
      const Vec3f p2 = getVertex(geom,tri.v2);
      if (unlikely(!isvalid(p0))) return QBVH6BuilderSAH::Triangle();
      if (unlikely(!isvalid(p1))) return QBVH6BuilderSAH::Triangle();
      if (unlikely(!isvalid(p2))) return QBVH6BuilderSAH::Triangle();

      const GeometryFlags gflags = convertGeometryFlags(geom->geometryFlags);
      return QBVH6BuilderSAH::Triangle(tri.v0,tri.v1,tri.v2,p0,p1,p2,gflags,geom->geometryMask);
    };
    
    auto getTriangleIndices = [&] (uint32_t geomID, uint32_t primID) {
      const ze_rtas_builder_triangles_geometry_info_exp_t* geom = (const ze_rtas_builder_triangles_geometry_info_exp_t*) geometries[geomID];
      assert(geom);
      const ze_rtas_triangle_indices_uint32_exp_t tri = getPrimitive(geom,primID);
      return Vec3<uint32_t>(tri.v0,tri.v1,tri.v2);
    };
    
    auto getQuad = [&](unsigned int geomID, unsigned int primID)
    {
      const ze_rtas_builder_quads_geometry_info_exp_t* geom = (const ze_rtas_builder_quads_geometry_info_exp_t*) geometries[geomID];
      assert(geom);
                     
      const ze_rtas_quad_indices_uint32_exp_t quad = getPrimitive(geom,primID);
      const Vec3f p0 = getVertex(geom,quad.v0);
      const Vec3f p1 = getVertex(geom,quad.v1);
      const Vec3f p2 = getVertex(geom,quad.v2);
      const Vec3f p3 = getVertex(geom,quad.v3);

      const GeometryFlags gflags = convertGeometryFlags(geom->geometryFlags);
      return QBVH6BuilderSAH::Quad(p0,p1,p2,p3,gflags,geom->geometryMask);
    };
    
    auto getProcedural = [&](unsigned int geomID, unsigned int primID) {
      const ze_rtas_builder_procedural_geometry_info_exp_t* geom = (const ze_rtas_builder_procedural_geometry_info_exp_t*) geometries[geomID];
      assert(geom);
      return QBVH6BuilderSAH::Procedural(geom->geometryMask); // FIXME: pass gflags
    };
    
    auto getInstance = [&](unsigned int geomID, unsigned int primID)
    {
      assert(geometries[geomID]);
      assert(geometries[geomID]->geometryType == ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE);
      const ze_rtas_builder_instance_geometry_info_exp_t* geom = (const ze_rtas_builder_instance_geometry_info_exp_t*) geometries[geomID];
      void* accel = geom->pAccelerationStructure;
      const AffineSpace3fa local2world = getTransform(geom);
      return QBVH6BuilderSAH::Instance(local2world,accel,geom->geometryMask,geom->instanceUserID); // FIXME: pass instance flags
    };

    /* dispatch globals ptr for debugging purposes */
    void* dispatchGlobalsPtr = nullptr;
#if defined(EMBREE_SYCL_ALLOC_DISPATCH_GLOBALS)
    if (args->pNext) {
      zet_base_desc_t_* next = (zet_base_desc_t_*) args->pNext;
      if (next->stype == ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_DEBUG_EXP_DESC) {
        ze_rtas_builder_build_op_debug_exp_desc_t* debug_ext = (ze_rtas_builder_build_op_debug_exp_desc_t*) next;
        dispatchGlobalsPtr = debug_ext->dispatchGlobalsPtr;
      }
    }
#endif

    bool verbose = false;
    bool success = QBVH6BuilderSAH::build(numGeometries, nullptr, 
                           getSize, getType, 
                           createPrimRefArray, getTriangle, getTriangleIndices, getQuad, getProcedural, getInstance,
                           (char*)pRtasBuffer, rtasBufferSizeBytes,
                           pScratchBuffer, scratchBufferSizeBytes,
                           (BBox3f*) pBounds, pRtasBufferSizeBytes,
                           args->rtasFormat, args->buildQuality, args->buildFlags, verbose, dispatchGlobalsPtr);
    if (!success) {
      return ZE_RESULT_EXP_RTAS_BUILD_RETRY;
    }
    return ZE_RESULT_SUCCESS;
  }
  catch (std::exception& e) {
    //std::cerr << "caught exception during BVH build: " << e.what() << std::endl;
    return ZE_RESULT_ERROR_UNKNOWN;
  }
  
  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASBuilderBuildExpImpl(ze_rtas_builder_exp_handle_t hBuilder,
                                                                     const ze_rtas_builder_build_op_exp_desc_t* args,
                                                                     void *pScratchBuffer, size_t scratchBufferSizeBytes,
                                                                     void *pRtasBuffer, size_t rtasBufferSizeBytes,
                                                                     ze_rtas_parallel_operation_exp_handle_t hParallelOperation,
                                                                     void *pBuildUserPtr, ze_rtas_aabb_exp_t *pBounds, size_t *pRtasBufferSizeBytes)
  {
    /* input validation */
    VALIDATE(hBuilder);
    VALIDATE(args);
    VALIDATE_PTR(pScratchBuffer);
    VALIDATE_PTR(pRtasBuffer);
    
    /* if parallel operation is provided then execute using thread arena inside task group ... */
    if (hParallelOperation)
    {
      VALIDATE(hParallelOperation);
      
      ze_rtas_parallel_operation_t* op = (ze_rtas_parallel_operation_t*) hParallelOperation;
      
      if (op->object_in_use.load())
        return ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE;
      
      op->object_in_use.store(true);
      
      g_arena.execute([&](){ op->group.run([=](){
         op->errorCode = zeRTASBuilderBuildExpBody(args,
                                                       pScratchBuffer, scratchBufferSizeBytes,
                                                       pRtasBuffer, rtasBufferSizeBytes,
                                                       pBuildUserPtr, pBounds, pRtasBufferSizeBytes);
                                            });
                       });
      return ZE_RESULT_EXP_RTAS_BUILD_DEFERRED;
    }
    /* ... otherwise we just execute inside task arena to avoid spawning of TBB worker threads */
    else
    {
      ze_result_t errorCode = ZE_RESULT_SUCCESS;
      g_arena.execute([&](){ errorCode = zeRTASBuilderBuildExpBody(args,
                                                                        pScratchBuffer, scratchBufferSizeBytes,
                                                                        pRtasBuffer, rtasBufferSizeBytes,
                                                                        pBuildUserPtr, pBounds, pRtasBufferSizeBytes);
                       });
      return errorCode;
    }
  }

  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASParallelOperationCreateExpImpl(ze_driver_handle_t hDriver, ze_rtas_parallel_operation_exp_handle_t* phParallelOperation)
  {
    /* input validation */
    VALIDATE(hDriver);
    VALIDATE_PTR(phParallelOperation);

    /* create parallel operation object */
    *phParallelOperation = (ze_rtas_parallel_operation_exp_handle_t) new ze_rtas_parallel_operation_t();
    return ZE_RESULT_SUCCESS;
  }
  
  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASParallelOperationDestroyExpImpl( ze_rtas_parallel_operation_exp_handle_t hParallelOperation )
  {
    /* input validation */
    VALIDATE(hParallelOperation);

    /* delete parallel operation */
    delete (ze_rtas_parallel_operation_t*) hParallelOperation;
    return ZE_RESULT_SUCCESS;
  }
  
  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASParallelOperationGetPropertiesExpImpl( ze_rtas_parallel_operation_exp_handle_t hParallelOperation, ze_rtas_parallel_operation_exp_properties_t* pProperties )
  {
    /* input validation */
    VALIDATE(hParallelOperation);
    VALIDATE(pProperties);

    ze_rtas_parallel_operation_t* op = (ze_rtas_parallel_operation_t*) hParallelOperation;
    if (!op->object_in_use.load())
      return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    
    /* return properties */
    pProperties->flags = 0;
    pProperties->maxConcurrency = tbb::this_task_arena::max_concurrency();
    return ZE_RESULT_SUCCESS;
  }
  
  RTHWIF_API_EXPORT ze_result_t ZE_APICALL zeRTASParallelOperationJoinExpImpl( ze_rtas_parallel_operation_exp_handle_t hParallelOperation)
  {
    /* check for valid handle */
    VALIDATE(hParallelOperation);
    
    ze_rtas_parallel_operation_t* op = (ze_rtas_parallel_operation_t*) hParallelOperation;
    g_arena.execute([&](){ op->group.wait(); });
    op->object_in_use.store(false); // this is slighty too early
    return op->errorCode;
  }
}
