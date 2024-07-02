// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ze_api.h"

#if !defined(ZE_RTAS_BUILDER_EXP_NAME)
#include "ze_rtas.h"
#endif

//////////////////////
// Debug extension

#define ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_DEBUG_EXP_DESC ((ze_structure_type_t)0x00020020)  ///< ::ze_rtas_builder_build_op_debug_exp_desc_t

typedef struct _ze_rtas_builder_build_op_debug_exp_desc_t
{
  ze_structure_type_t stype;                                              ///< [in] type of this structure
  const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                          ///< structure (i.e. contains stype and pNext).
  void* dispatchGlobalsPtr;
  
} ze_rtas_builder_build_op_debug_exp_desc_t;

////////////////////

struct ZeWrapper
{
  enum RTAS_BUILD_MODE {
    AUTO = 0,       // try L0 implementation first and fallback to internal implementation
    INTERNAL = 1,   // use internal RTAS build implementation
    LEVEL_ZERO = 2, // use Level Zero provided RTAS build implementation
  };

  ~ZeWrapper();

  static ze_result_t init();
  static ze_result_t initRTASBuilder(ze_driver_handle_t hDriver, RTAS_BUILD_MODE rtas_build_mode = RTAS_BUILD_MODE::AUTO);
  
  static ze_result_t zeMemFree(ze_context_handle_t, void*);
  static ze_result_t zeMemAllocShared(ze_context_handle_t, const ze_device_mem_alloc_desc_t*, const ze_host_mem_alloc_desc_t*, size_t, size_t, ze_device_handle_t, void**);
  static ze_result_t zeDriverGetExtensionProperties(ze_driver_handle_t, uint32_t*, ze_driver_extension_properties_t*);
  static ze_result_t zeDeviceGetProperties(ze_device_handle_t, ze_device_properties_t*);
  static ze_result_t zeDeviceGetModuleProperties(ze_device_handle_t, ze_device_module_properties_t*);

  static ze_result_t zeRTASBuilderCreateExp(ze_driver_handle_t hDriver, const ze_rtas_builder_exp_desc_t *pDescriptor, ze_rtas_builder_exp_handle_t *phBuilder);
  static ze_result_t zeRTASBuilderDestroyExp(ze_rtas_builder_exp_handle_t hBuilder);
  static ze_result_t zeDriverRTASFormatCompatibilityCheckExp( ze_driver_handle_t hDriver,
                                                              const ze_rtas_format_exp_t accelFormat,
                                                              const ze_rtas_format_exp_t otherAccelFormat);
  static ze_result_t zeRTASBuilderGetBuildPropertiesExp(ze_rtas_builder_exp_handle_t hBuilder,
                                                        const ze_rtas_builder_build_op_exp_desc_t* args,
                                                        ze_rtas_builder_exp_properties_t* pProp);
  static ze_result_t zeRTASBuilderBuildExp(ze_rtas_builder_exp_handle_t hBuilder,
                                           const ze_rtas_builder_build_op_exp_desc_t* args,
                                           void *pScratchBuffer, size_t scratchBufferSizeBytes,
                                           void *pRtasBuffer, size_t rtasBufferSizeBytes,
                                           ze_rtas_parallel_operation_exp_handle_t hParallelOperation,
                                           void *pBuildUserPtr, ze_rtas_aabb_exp_t *pBounds, size_t *pRtasBufferSizeBytes);
  
  static ze_result_t zeRTASParallelOperationCreateExp(ze_driver_handle_t hDriver, ze_rtas_parallel_operation_exp_handle_t* phParallelOperation);
  static ze_result_t zeRTASParallelOperationDestroyExp( ze_rtas_parallel_operation_exp_handle_t hParallelOperation );
  static ze_result_t zeRTASParallelOperationGetPropertiesExp( ze_rtas_parallel_operation_exp_handle_t hParallelOperation, ze_rtas_parallel_operation_exp_properties_t* pProperties );
  static ze_result_t zeRTASParallelOperationJoinExp( ze_rtas_parallel_operation_exp_handle_t hParallelOperation);

  static RTAS_BUILD_MODE rtas_builder;
};

