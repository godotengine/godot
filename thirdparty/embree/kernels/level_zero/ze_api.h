/*
 *
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ze_api.h
 * @version v1.7-r1.7.9
 *
 */
#ifndef _ZE_API_H
#define _ZE_API_H
#if defined(__cplusplus)
#pragma once
#endif

// standard headers
#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Level-Zero API common types
#if !defined(__GNUC__)
#pragma region common
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAKE_VERSION
/// @brief Generates generic 'oneAPI' API versions
#define ZE_MAKE_VERSION( _major, _minor )  (( _major << 16 )|( _minor & 0x0000ffff))
#endif // ZE_MAKE_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAJOR_VERSION
/// @brief Extracts 'oneAPI' API major version
#define ZE_MAJOR_VERSION( _ver )  ( _ver >> 16 )
#endif // ZE_MAJOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MINOR_VERSION
/// @brief Extracts 'oneAPI' API minor version
#define ZE_MINOR_VERSION( _ver )  ( _ver & 0x0000ffff )
#endif // ZE_MINOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_APICALL
#if defined(_WIN32)
/// @brief Calling convention for all API functions
#define ZE_APICALL  __cdecl
#else
#define ZE_APICALL  
#endif // defined(_WIN32)
#endif // ZE_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_APIEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define ZE_APIEXPORT  __declspec(dllexport)
#endif // defined(_WIN32)
#endif // ZE_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_APIEXPORT
#if __GNUC__ >= 4
/// @brief GCC-specific dllexport storage-class attribute
#define ZE_APIEXPORT  __attribute__ ((visibility ("default")))
#else
#define ZE_APIEXPORT  
#endif // __GNUC__ >= 4
#endif // ZE_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_DLLEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define ZE_DLLEXPORT  __declspec(dllexport)
#endif // defined(_WIN32)
#endif // ZE_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_DLLEXPORT
#if __GNUC__ >= 4
/// @brief GCC-specific dllexport storage-class attribute
#define ZE_DLLEXPORT  __attribute__ ((visibility ("default")))
#else
#define ZE_DLLEXPORT  
#endif // __GNUC__ >= 4
#endif // ZE_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief compiler-independent type
typedef uint8_t ze_bool_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a driver instance
typedef struct _ze_driver_handle_t *ze_driver_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's device object
typedef struct _ze_device_handle_t *ze_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's context object
typedef struct _ze_context_handle_t *ze_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's command queue object
typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's command list object
typedef struct _ze_command_list_handle_t *ze_command_list_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's fence object
typedef struct _ze_fence_handle_t *ze_fence_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's event pool object
typedef struct _ze_event_pool_handle_t *ze_event_pool_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's event object
typedef struct _ze_event_handle_t *ze_event_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's image object
typedef struct _ze_image_handle_t *ze_image_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's module object
typedef struct _ze_module_handle_t *ze_module_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of module's build log object
typedef struct _ze_module_build_log_handle_t *ze_module_build_log_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's kernel object
typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's sampler object
typedef struct _ze_sampler_handle_t *ze_sampler_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of physical memory object
typedef struct _ze_physical_mem_handle_t *ze_physical_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's fabric vertex object
typedef struct _ze_fabric_vertex_handle_t *ze_fabric_vertex_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's fabric edge object
typedef struct _ze_fabric_edge_handle_t *ze_fabric_edge_handle_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_IPC_HANDLE_SIZE
/// @brief Maximum IPC handle size
#define ZE_MAX_IPC_HANDLE_SIZE  64
#endif // ZE_MAX_IPC_HANDLE_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief IPC handle to a memory allocation
typedef struct _ze_ipc_mem_handle_t
{
    char data[ZE_MAX_IPC_HANDLE_SIZE];                                      ///< [out] Opaque data representing an IPC handle

} ze_ipc_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief IPC handle to a event pool allocation
typedef struct _ze_ipc_event_pool_handle_t
{
    char data[ZE_MAX_IPC_HANDLE_SIZE];                                      ///< [out] Opaque data representing an IPC handle

} ze_ipc_event_pool_handle_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_BIT
/// @brief Generic macro for enumerator bit masks
#define ZE_BIT( _i )  ( 1 << _i )
#endif // ZE_BIT

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes
typedef enum _ze_result_t
{
    ZE_RESULT_SUCCESS = 0,                                                  ///< [Core] success
    ZE_RESULT_NOT_READY = 1,                                                ///< [Core] synchronization primitive not signaled
    ZE_RESULT_ERROR_DEVICE_LOST = 0x70000001,                               ///< [Core] device hung, reset, was removed, or driver update occurred
    ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY = 0x70000002,                        ///< [Core] insufficient host memory to satisfy call
    ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 0x70000003,                      ///< [Core] insufficient device memory to satisfy call
    ZE_RESULT_ERROR_MODULE_BUILD_FAILURE = 0x70000004,                      ///< [Core] error occurred when building module, see build log for details
    ZE_RESULT_ERROR_MODULE_LINK_FAILURE = 0x70000005,                       ///< [Core] error occurred when linking modules, see build log for details
    ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET = 0x70000006,                     ///< [Core] device requires a reset
    ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 0x70000007,                 ///< [Core] device currently in low power state
    ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX = 0x7ff00001,                  ///< [Core, Experimental] device is not represented by a fabric vertex
    ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE = 0x7ff00002,                  ///< [Core, Experimental] fabric vertex does not represent a device
    ZE_RESULT_EXP_ERROR_REMOTE_DEVICE = 0x7ff00003,                         ///< [Core, Experimental] fabric vertex represents a remote device or
                                                                            ///< subdevice
    ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE = 0x7ff00004,                 ///< [Core, Experimental] operands of comparison are not compatible
    ZE_RESULT_EXP_RTAS_BUILD_RETRY = 0x7ff00005,                            ///< [Core, Experimental] ray tracing acceleration structure build
                                                                            ///< operation failed due to insufficient resources, retry with a larger
                                                                            ///< acceleration structure buffer allocation
    ZE_RESULT_EXP_RTAS_BUILD_DEFERRED = 0x7ff00006,                         ///< [Core, Experimental] ray tracing acceleration structure build
                                                                            ///< operation deferred to parallel operation join
    ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 0x70010000,                  ///< [Sysman] access denied due to permission level
    ZE_RESULT_ERROR_NOT_AVAILABLE = 0x70010001,                             ///< [Sysman] resource already in use and simultaneous access not allowed
                                                                            ///< or resource was removed
    ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE = 0x70020000,                    ///< [Common] external required dependency is unavailable or missing
    ZE_RESULT_WARNING_DROPPED_DATA = 0x70020001,                            ///< [Tools] data may have been dropped
    ZE_RESULT_ERROR_UNINITIALIZED = 0x78000001,                             ///< [Validation] driver is not initialized
    ZE_RESULT_ERROR_UNSUPPORTED_VERSION = 0x78000002,                       ///< [Validation] generic error code for unsupported versions
    ZE_RESULT_ERROR_UNSUPPORTED_FEATURE = 0x78000003,                       ///< [Validation] generic error code for unsupported features
    ZE_RESULT_ERROR_INVALID_ARGUMENT = 0x78000004,                          ///< [Validation] generic error code for invalid arguments
    ZE_RESULT_ERROR_INVALID_NULL_HANDLE = 0x78000005,                       ///< [Validation] handle argument is not valid
    ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 0x78000006,                      ///< [Validation] object pointed to by handle still in-use by device
    ZE_RESULT_ERROR_INVALID_NULL_POINTER = 0x78000007,                      ///< [Validation] pointer argument may not be nullptr
    ZE_RESULT_ERROR_INVALID_SIZE = 0x78000008,                              ///< [Validation] size argument is invalid (e.g., must not be zero)
    ZE_RESULT_ERROR_UNSUPPORTED_SIZE = 0x78000009,                          ///< [Validation] size argument is not supported by the device (e.g., too
                                                                            ///< large)
    ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 0x7800000a,                     ///< [Validation] alignment argument is not supported by the device (e.g.,
                                                                            ///< too small)
    ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 0x7800000b,            ///< [Validation] synchronization object in invalid state
    ZE_RESULT_ERROR_INVALID_ENUMERATION = 0x7800000c,                       ///< [Validation] enumerator argument is not valid
    ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 0x7800000d,                   ///< [Validation] enumerator argument is not supported by the device
    ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 0x7800000e,                  ///< [Validation] image format is not supported by the device
    ZE_RESULT_ERROR_INVALID_NATIVE_BINARY = 0x7800000f,                     ///< [Validation] native binary is not supported by the device
    ZE_RESULT_ERROR_INVALID_GLOBAL_NAME = 0x78000010,                       ///< [Validation] global variable is not found in the module
    ZE_RESULT_ERROR_INVALID_KERNEL_NAME = 0x78000011,                       ///< [Validation] kernel name is not found in the module
    ZE_RESULT_ERROR_INVALID_FUNCTION_NAME = 0x78000012,                     ///< [Validation] function name is not found in the module
    ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 0x78000013,              ///< [Validation] group size dimension is not valid for the kernel or
                                                                            ///< device
    ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 0x78000014,            ///< [Validation] global width dimension is not valid for the kernel or
                                                                            ///< device
    ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 0x78000015,             ///< [Validation] kernel argument index is not valid for kernel
    ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 0x78000016,              ///< [Validation] kernel argument size does not match kernel
    ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 0x78000017,            ///< [Validation] value of kernel attribute is not valid for the kernel or
                                                                            ///< device
    ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED = 0x78000018,                   ///< [Validation] module with imports needs to be linked before kernels can
                                                                            ///< be created from it.
    ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 0x78000019,                 ///< [Validation] command list type does not match command queue type
    ZE_RESULT_ERROR_OVERLAPPING_REGIONS = 0x7800001a,                       ///< [Validation] copy operations do not support overlapping regions of
                                                                            ///< memory
    ZE_RESULT_WARNING_ACTION_REQUIRED = 0x7800001b,                         ///< [Sysman] an action is required to complete the desired operation
    ZE_RESULT_ERROR_UNKNOWN = 0x7ffffffe,                                   ///< [Core] unknown or internal error
    ZE_RESULT_FORCE_UINT32 = 0x7fffffff

} ze_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _ze_structure_type_t
{
    ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 0x1,                              ///< ::ze_driver_properties_t
    ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES = 0x2,                          ///< ::ze_driver_ipc_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3,                              ///< ::ze_device_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 0x4,                      ///< ::ze_device_compute_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES = 0x5,                       ///< ::ze_device_module_properties_t
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 0x6,                 ///< ::ze_command_queue_group_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 0x7,                       ///< ::ze_device_memory_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES = 0x8,                ///< ::ze_device_memory_access_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES = 0x9,                        ///< ::ze_device_cache_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES = 0xa,                        ///< ::ze_device_image_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES = 0xb,                          ///< ::ze_device_p2p_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 0xc,              ///< ::ze_device_external_memory_properties_t
    ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xd,                                   ///< ::ze_context_desc_t
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0xe,                             ///< ::ze_command_queue_desc_t
    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0xf,                              ///< ::ze_command_list_desc_t
    ZE_STRUCTURE_TYPE_EVENT_POOL_DESC = 0x10,                               ///< ::ze_event_pool_desc_t
    ZE_STRUCTURE_TYPE_EVENT_DESC = 0x11,                                    ///< ::ze_event_desc_t
    ZE_STRUCTURE_TYPE_FENCE_DESC = 0x12,                                    ///< ::ze_fence_desc_t
    ZE_STRUCTURE_TYPE_IMAGE_DESC = 0x13,                                    ///< ::ze_image_desc_t
    ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES = 0x14,                              ///< ::ze_image_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15,                         ///< ::ze_device_mem_alloc_desc_t
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16,                           ///< ::ze_host_mem_alloc_desc_t
    ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x17,                  ///< ::ze_memory_allocation_properties_t
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 0x18,                   ///< ::ze_external_memory_export_desc_t
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD = 0x19,                     ///< ::ze_external_memory_import_fd_t
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD = 0x1a,                     ///< ::ze_external_memory_export_fd_t
    ZE_STRUCTURE_TYPE_MODULE_DESC = 0x1b,                                   ///< ::ze_module_desc_t
    ZE_STRUCTURE_TYPE_MODULE_PROPERTIES = 0x1c,                             ///< ::ze_module_properties_t
    ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x1d,                                   ///< ::ze_kernel_desc_t
    ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES = 0x1e,                             ///< ::ze_kernel_properties_t
    ZE_STRUCTURE_TYPE_SAMPLER_DESC = 0x1f,                                  ///< ::ze_sampler_desc_t
    ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC = 0x20,                             ///< ::ze_physical_mem_desc_t
    ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES = 0x21,        ///< ::ze_kernel_preferred_group_size_properties_t
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32 = 0x22,                  ///< ::ze_external_memory_import_win32_handle_t
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32 = 0x23,                  ///< ::ze_external_memory_export_win32_handle_t
    ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES = 0x00010001,        ///< ::ze_device_raytracing_ext_properties_t
    ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC = 0x10002,              ///< ::ze_raytracing_mem_alloc_ext_desc_t
    ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES = 0x10003,                ///< ::ze_float_atomic_ext_properties_t
    ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC = 0x10004,                 ///< ::ze_cache_reservation_ext_desc_t
    ZE_STRUCTURE_TYPE_EU_COUNT_EXT = 0x10005,                               ///< ::ze_eu_count_ext_t
    ZE_STRUCTURE_TYPE_SRGB_EXT_DESC = 0x10006,                              ///< ::ze_srgb_ext_desc_t
    ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC = 0x10007,                ///< ::ze_linkage_inspection_ext_desc_t
    ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES = 0x10008,                         ///< ::ze_pci_ext_properties_t
    ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES = 0x10009,          ///< ::ze_driver_memory_free_ext_properties_t
    ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC = 0x1000a,                       ///< ::ze_memory_free_ext_desc_t
    ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC = 0x1000b,          ///< ::ze_memory_compression_hints_ext_desc_t
    ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES = 0x1000c,            ///< ::ze_image_allocation_ext_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES = 0x1000d,                 ///< ::ze_device_luid_ext_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES = 0x1000e,               ///< ::ze_device_memory_ext_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT = 0x1000f,                      ///< ::ze_device_ip_version_ext_t
    ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXT_DESC = 0x10010,                 ///< ::ze_image_view_planar_ext_desc_t
    ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES = 0x10011,   ///< ::ze_event_query_kernel_timestamps_ext_properties_t
    ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES = 0x10012,   ///< ::ze_event_query_kernel_timestamps_results_ext_properties_t
    ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES = 0x10013,       ///< ::ze_kernel_max_group_size_ext_properties_t
    ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC = 0x00020001,      ///< ::ze_relaxed_allocation_limits_exp_desc_t
    ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC = 0x00020002,                 ///< ::ze_module_program_exp_desc_t
    ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES = 0x00020003,          ///< ::ze_scheduling_hint_exp_properties_t
    ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC = 0x00020004,                ///< ::ze_scheduling_hint_exp_desc_t
    ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC = 0x00020005,              ///< ::ze_image_view_planar_exp_desc_t
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 = 0x00020006,                   ///< ::ze_device_properties_t
    ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES = 0x00020007,             ///< ::ze_image_memory_properties_exp_t
    ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC = 0x00020008,              ///< ::ze_context_power_saving_hint_exp_desc_t
    ZE_STRUCTURE_TYPE_COPY_BANDWIDTH_EXP_PROPERTIES = 0x00020009,           ///< ::ze_copy_bandwidth_exp_properties_t
    ZE_STRUCTURE_TYPE_DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES = 0x0002000A,     ///< ::ze_device_p2p_bandwidth_exp_properties_t
    ZE_STRUCTURE_TYPE_FABRIC_VERTEX_EXP_PROPERTIES = 0x0002000B,            ///< ::ze_fabric_vertex_exp_properties_t
    ZE_STRUCTURE_TYPE_FABRIC_EDGE_EXP_PROPERTIES = 0x0002000C,              ///< ::ze_fabric_edge_exp_properties_t
    ZE_STRUCTURE_TYPE_MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES = 0x0002000D,   ///< ::ze_memory_sub_allocations_exp_properties_t
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC = 0x0002000E,                   ///< ::ze_rtas_builder_exp_desc_t
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC = 0x0002000F,          ///< ::ze_rtas_builder_build_op_exp_desc_t
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES = 0x00020010,             ///< ::ze_rtas_builder_exp_properties_t
    ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES = 0x00020011,  ///< ::ze_rtas_parallel_operation_exp_properties_t
    ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES = 0x00020012,              ///< ::ze_rtas_device_exp_properties_t
    ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS = 0x00020013,       ///< ::ze_rtas_geometry_aabbs_exp_cb_params_t
    ZE_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief External memory type flags
typedef uint32_t ze_external_memory_type_flags_t;
typedef enum _ze_external_memory_type_flag_t
{
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD = ZE_BIT(0),                     ///< an opaque POSIX file descriptor handle
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = ZE_BIT(1),                       ///< a file descriptor handle for a Linux dma_buf
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32 = ZE_BIT(2),                  ///< an NT handle
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT = ZE_BIT(3),              ///< a global share (KMT) handle
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE = ZE_BIT(4),                 ///< an NT handle referring to a Direct3D 10 or 11 texture resource
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT = ZE_BIT(5),             ///< a global share (KMT) handle referring to a Direct3D 10 or 11 texture
                                                                            ///< resource
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP = ZE_BIT(6),                    ///< an NT handle referring to a Direct3D 12 heap resource
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE = ZE_BIT(7),                ///< an NT handle referring to a Direct3D 12 committed resource
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_external_memory_type_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Bandwidth unit
typedef enum _ze_bandwidth_unit_t
{
    ZE_BANDWIDTH_UNIT_UNKNOWN = 0,                                          ///< The unit used for bandwidth is unknown
    ZE_BANDWIDTH_UNIT_BYTES_PER_NANOSEC = 1,                                ///< Bandwidth is provided in bytes/nanosec
    ZE_BANDWIDTH_UNIT_BYTES_PER_CLOCK = 2,                                  ///< Bandwidth is provided in bytes/clock
    ZE_BANDWIDTH_UNIT_FORCE_UINT32 = 0x7fffffff

} ze_bandwidth_unit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Latency unit
typedef enum _ze_latency_unit_t
{
    ZE_LATENCY_UNIT_UNKNOWN = 0,                                            ///< The unit used for latency is unknown
    ZE_LATENCY_UNIT_NANOSEC = 1,                                            ///< Latency is provided in nanosecs
    ZE_LATENCY_UNIT_CLOCK = 2,                                              ///< Latency is provided in clocks
    ZE_LATENCY_UNIT_HOP = 3,                                                ///< Latency is provided in hops (normalized so that the lowest latency
                                                                            ///< link has a latency of 1 hop)
    ZE_LATENCY_UNIT_FORCE_UINT32 = 0x7fffffff

} ze_latency_unit_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_UUID_SIZE
/// @brief Maximum universal unique id (UUID) size in bytes
#define ZE_MAX_UUID_SIZE  16
#endif // ZE_MAX_UUID_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Universal unique id (UUID)
typedef struct _ze_uuid_t
{
    uint8_t id[ZE_MAX_UUID_SIZE];                                           ///< [out] opaque data representing a UUID

} ze_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all callback function parameter types
typedef struct _ze_base_cb_params_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} ze_base_cb_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct _ze_base_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} ze_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct _ze_base_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} ze_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forces driver to only report devices (and sub-devices) as specified by
///        values

///////////////////////////////////////////////////////////////////////////////
/// @brief Forces driver to report devices from lowest to highest PCI bus ID

///////////////////////////////////////////////////////////////////////////////
/// @brief Forces all shared allocations into device memory

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines the device hierarchy model exposed by Level Zero driver
///        implementation

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_ipc_mem_handle_t
typedef struct _ze_ipc_mem_handle_t ze_ipc_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_ipc_event_pool_handle_t
typedef struct _ze_ipc_event_pool_handle_t ze_ipc_event_pool_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_uuid_t
typedef struct _ze_uuid_t ze_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_base_cb_params_t
typedef struct _ze_base_cb_params_t ze_base_cb_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_base_properties_t
typedef struct _ze_base_properties_t ze_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_base_desc_t
typedef struct _ze_base_desc_t ze_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_driver_uuid_t
typedef struct _ze_driver_uuid_t ze_driver_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_driver_properties_t
typedef struct _ze_driver_properties_t ze_driver_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_driver_ipc_properties_t
typedef struct _ze_driver_ipc_properties_t ze_driver_ipc_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_driver_extension_properties_t
typedef struct _ze_driver_extension_properties_t ze_driver_extension_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_uuid_t
typedef struct _ze_device_uuid_t ze_device_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_properties_t
typedef struct _ze_device_properties_t ze_device_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_thread_t
typedef struct _ze_device_thread_t ze_device_thread_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_compute_properties_t
typedef struct _ze_device_compute_properties_t ze_device_compute_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_native_kernel_uuid_t
typedef struct _ze_native_kernel_uuid_t ze_native_kernel_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_module_properties_t
typedef struct _ze_device_module_properties_t ze_device_module_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_command_queue_group_properties_t
typedef struct _ze_command_queue_group_properties_t ze_command_queue_group_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_memory_properties_t
typedef struct _ze_device_memory_properties_t ze_device_memory_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_memory_access_properties_t
typedef struct _ze_device_memory_access_properties_t ze_device_memory_access_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_cache_properties_t
typedef struct _ze_device_cache_properties_t ze_device_cache_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_image_properties_t
typedef struct _ze_device_image_properties_t ze_device_image_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_external_memory_properties_t
typedef struct _ze_device_external_memory_properties_t ze_device_external_memory_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_p2p_properties_t
typedef struct _ze_device_p2p_properties_t ze_device_p2p_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_context_desc_t
typedef struct _ze_context_desc_t ze_context_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_command_queue_desc_t
typedef struct _ze_command_queue_desc_t ze_command_queue_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_command_list_desc_t
typedef struct _ze_command_list_desc_t ze_command_list_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_copy_region_t
typedef struct _ze_copy_region_t ze_copy_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_region_t
typedef struct _ze_image_region_t ze_image_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_event_pool_desc_t
typedef struct _ze_event_pool_desc_t ze_event_pool_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_event_desc_t
typedef struct _ze_event_desc_t ze_event_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_kernel_timestamp_data_t
typedef struct _ze_kernel_timestamp_data_t ze_kernel_timestamp_data_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_kernel_timestamp_result_t
typedef struct _ze_kernel_timestamp_result_t ze_kernel_timestamp_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_fence_desc_t
typedef struct _ze_fence_desc_t ze_fence_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_format_t
typedef struct _ze_image_format_t ze_image_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_desc_t
typedef struct _ze_image_desc_t ze_image_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_properties_t
typedef struct _ze_image_properties_t ze_image_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_mem_alloc_desc_t
typedef struct _ze_device_mem_alloc_desc_t ze_device_mem_alloc_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_host_mem_alloc_desc_t
typedef struct _ze_host_mem_alloc_desc_t ze_host_mem_alloc_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_memory_allocation_properties_t
typedef struct _ze_memory_allocation_properties_t ze_memory_allocation_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_external_memory_export_desc_t
typedef struct _ze_external_memory_export_desc_t ze_external_memory_export_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_external_memory_import_fd_t
typedef struct _ze_external_memory_import_fd_t ze_external_memory_import_fd_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_external_memory_export_fd_t
typedef struct _ze_external_memory_export_fd_t ze_external_memory_export_fd_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_external_memory_import_win32_handle_t
typedef struct _ze_external_memory_import_win32_handle_t ze_external_memory_import_win32_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_external_memory_export_win32_handle_t
typedef struct _ze_external_memory_export_win32_handle_t ze_external_memory_export_win32_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_module_constants_t
typedef struct _ze_module_constants_t ze_module_constants_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_module_desc_t
typedef struct _ze_module_desc_t ze_module_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_module_properties_t
typedef struct _ze_module_properties_t ze_module_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_kernel_desc_t
typedef struct _ze_kernel_desc_t ze_kernel_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_kernel_uuid_t
typedef struct _ze_kernel_uuid_t ze_kernel_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_kernel_properties_t
typedef struct _ze_kernel_properties_t ze_kernel_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_kernel_preferred_group_size_properties_t
typedef struct _ze_kernel_preferred_group_size_properties_t ze_kernel_preferred_group_size_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_group_count_t
typedef struct _ze_group_count_t ze_group_count_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_module_program_exp_desc_t
typedef struct _ze_module_program_exp_desc_t ze_module_program_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_raytracing_ext_properties_t
typedef struct _ze_device_raytracing_ext_properties_t ze_device_raytracing_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_raytracing_mem_alloc_ext_desc_t
typedef struct _ze_raytracing_mem_alloc_ext_desc_t ze_raytracing_mem_alloc_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_sampler_desc_t
typedef struct _ze_sampler_desc_t ze_sampler_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_physical_mem_desc_t
typedef struct _ze_physical_mem_desc_t ze_physical_mem_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_float_atomic_ext_properties_t
typedef struct _ze_float_atomic_ext_properties_t ze_float_atomic_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_relaxed_allocation_limits_exp_desc_t
typedef struct _ze_relaxed_allocation_limits_exp_desc_t ze_relaxed_allocation_limits_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_cache_reservation_ext_desc_t
typedef struct _ze_cache_reservation_ext_desc_t ze_cache_reservation_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_memory_properties_exp_t
typedef struct _ze_image_memory_properties_exp_t ze_image_memory_properties_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_view_planar_ext_desc_t
typedef struct _ze_image_view_planar_ext_desc_t ze_image_view_planar_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_view_planar_exp_desc_t
typedef struct _ze_image_view_planar_exp_desc_t ze_image_view_planar_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_scheduling_hint_exp_properties_t
typedef struct _ze_scheduling_hint_exp_properties_t ze_scheduling_hint_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_scheduling_hint_exp_desc_t
typedef struct _ze_scheduling_hint_exp_desc_t ze_scheduling_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_context_power_saving_hint_exp_desc_t
typedef struct _ze_context_power_saving_hint_exp_desc_t ze_context_power_saving_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_eu_count_ext_t
typedef struct _ze_eu_count_ext_t ze_eu_count_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_pci_address_ext_t
typedef struct _ze_pci_address_ext_t ze_pci_address_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_pci_speed_ext_t
typedef struct _ze_pci_speed_ext_t ze_pci_speed_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_pci_ext_properties_t
typedef struct _ze_pci_ext_properties_t ze_pci_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_srgb_ext_desc_t
typedef struct _ze_srgb_ext_desc_t ze_srgb_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_image_allocation_ext_properties_t
typedef struct _ze_image_allocation_ext_properties_t ze_image_allocation_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_linkage_inspection_ext_desc_t
typedef struct _ze_linkage_inspection_ext_desc_t ze_linkage_inspection_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_memory_compression_hints_ext_desc_t
typedef struct _ze_memory_compression_hints_ext_desc_t ze_memory_compression_hints_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_driver_memory_free_ext_properties_t
typedef struct _ze_driver_memory_free_ext_properties_t ze_driver_memory_free_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_memory_free_ext_desc_t
typedef struct _ze_memory_free_ext_desc_t ze_memory_free_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_p2p_bandwidth_exp_properties_t
typedef struct _ze_device_p2p_bandwidth_exp_properties_t ze_device_p2p_bandwidth_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_copy_bandwidth_exp_properties_t
typedef struct _ze_copy_bandwidth_exp_properties_t ze_copy_bandwidth_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_luid_ext_t
typedef struct _ze_device_luid_ext_t ze_device_luid_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_luid_ext_properties_t
typedef struct _ze_device_luid_ext_properties_t ze_device_luid_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_fabric_vertex_pci_exp_address_t
typedef struct _ze_fabric_vertex_pci_exp_address_t ze_fabric_vertex_pci_exp_address_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_fabric_vertex_exp_properties_t
typedef struct _ze_fabric_vertex_exp_properties_t ze_fabric_vertex_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_fabric_edge_exp_properties_t
typedef struct _ze_fabric_edge_exp_properties_t ze_fabric_edge_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_memory_ext_properties_t
typedef struct _ze_device_memory_ext_properties_t ze_device_memory_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_device_ip_version_ext_t
typedef struct _ze_device_ip_version_ext_t ze_device_ip_version_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_kernel_max_group_size_properties_ext_t
typedef struct _ze_kernel_max_group_size_properties_ext_t ze_kernel_max_group_size_properties_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_sub_allocation_t
typedef struct _ze_sub_allocation_t ze_sub_allocation_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_memory_sub_allocations_exp_properties_t
typedef struct _ze_memory_sub_allocations_exp_properties_t ze_memory_sub_allocations_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_event_query_kernel_timestamps_ext_properties_t
typedef struct _ze_event_query_kernel_timestamps_ext_properties_t ze_event_query_kernel_timestamps_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_synchronized_timestamp_data_ext_t
typedef struct _ze_synchronized_timestamp_data_ext_t ze_synchronized_timestamp_data_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_synchronized_timestamp_result_ext_t
typedef struct _ze_synchronized_timestamp_result_ext_t ze_synchronized_timestamp_result_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_event_query_kernel_timestamps_results_ext_properties_t
typedef struct _ze_event_query_kernel_timestamps_results_ext_properties_t ze_event_query_kernel_timestamps_results_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_exp_desc_t
typedef struct _ze_rtas_builder_exp_desc_t ze_rtas_builder_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_exp_properties_t
typedef struct _ze_rtas_builder_exp_properties_t ze_rtas_builder_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_parallel_operation_exp_properties_t
typedef struct _ze_rtas_parallel_operation_exp_properties_t ze_rtas_parallel_operation_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_device_exp_properties_t
typedef struct _ze_rtas_device_exp_properties_t ze_rtas_device_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_float3_exp_t
typedef struct _ze_rtas_float3_exp_t ze_rtas_float3_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_transform_float3x4_column_major_exp_t
typedef struct _ze_rtas_transform_float3x4_column_major_exp_t ze_rtas_transform_float3x4_column_major_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_transform_float3x4_aligned_column_major_exp_t
typedef struct _ze_rtas_transform_float3x4_aligned_column_major_exp_t ze_rtas_transform_float3x4_aligned_column_major_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_transform_float3x4_row_major_exp_t
typedef struct _ze_rtas_transform_float3x4_row_major_exp_t ze_rtas_transform_float3x4_row_major_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_aabb_exp_t
typedef struct _ze_rtas_aabb_exp_t ze_rtas_aabb_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_triangle_indices_uint32_exp_t
typedef struct _ze_rtas_triangle_indices_uint32_exp_t ze_rtas_triangle_indices_uint32_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_quad_indices_uint32_exp_t
typedef struct _ze_rtas_quad_indices_uint32_exp_t ze_rtas_quad_indices_uint32_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_geometry_info_exp_t
typedef struct _ze_rtas_builder_geometry_info_exp_t ze_rtas_builder_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_triangles_geometry_info_exp_t
typedef struct _ze_rtas_builder_triangles_geometry_info_exp_t ze_rtas_builder_triangles_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_quads_geometry_info_exp_t
typedef struct _ze_rtas_builder_quads_geometry_info_exp_t ze_rtas_builder_quads_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_geometry_aabbs_exp_cb_params_t
typedef struct _ze_rtas_geometry_aabbs_exp_cb_params_t ze_rtas_geometry_aabbs_exp_cb_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_procedural_geometry_info_exp_t
typedef struct _ze_rtas_builder_procedural_geometry_info_exp_t ze_rtas_builder_procedural_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_instance_geometry_info_exp_t
typedef struct _ze_rtas_builder_instance_geometry_info_exp_t ze_rtas_builder_instance_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_rtas_builder_build_op_exp_desc_t
typedef struct _ze_rtas_builder_build_op_exp_desc_t ze_rtas_builder_build_op_exp_desc_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs
#if !defined(__GNUC__)
#pragma region driver
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported initialization flags
typedef uint32_t ze_init_flags_t;
typedef enum _ze_init_flag_t
{
    ZE_INIT_FLAG_GPU_ONLY = ZE_BIT(0),                                      ///< only initialize GPU drivers
    ZE_INIT_FLAG_VPU_ONLY = ZE_BIT(1),                                      ///< only initialize VPU drivers
    ZE_INIT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_init_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Initialize the 'oneAPI' driver(s)
/// 
/// @details
///     - The application must call this function before calling any other
///       function.
///     - If this function is not called then all other functions will return
///       ::ZE_RESULT_ERROR_UNINITIALIZED.
///     - Only one instance of each driver will be initialized per process.
///     - The application may call this function multiple times with different
///       flags or environment variables enabled.
///     - The application must call this function after forking new processes.
///       Each forked process must call this function.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe for scenarios
///       where multiple libraries may initialize the driver(s) simultaneously.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < flags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeInit(
    ze_init_flags_t flags                                                   ///< [in] initialization flags.
                                                                            ///< must be 0 (default) or a combination of ::ze_init_flag_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves driver instances
/// 
/// @details
///     - A driver represents a collection of physical devices.
///     - Multiple calls to this function will return identical driver handles,
///       in the same order.
///     - The application may pass nullptr for pDrivers when only querying the
///       number of drivers.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clGetPlatformIDs
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGet(
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of driver instances.
                                                                            ///< if count is zero, then the loader shall update the value with the
                                                                            ///< total number of drivers available.
                                                                            ///< if count is greater than the number of drivers available, then the
                                                                            ///< loader shall update the value with the correct number of drivers available.
    ze_driver_handle_t* phDrivers                                           ///< [in,out][optional][range(0, *pCount)] array of driver instance handles.
                                                                            ///< if count is less than the number of drivers available, then the loader
                                                                            ///< shall only retrieve that number of drivers.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported API versions
/// 
/// @details
///     - API versions contain major and minor attributes, use
///       ::ZE_MAJOR_VERSION and ::ZE_MINOR_VERSION
typedef enum _ze_api_version_t
{
    ZE_API_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                           ///< version 1.0
    ZE_API_VERSION_1_1 = ZE_MAKE_VERSION( 1, 1 ),                           ///< version 1.1
    ZE_API_VERSION_1_2 = ZE_MAKE_VERSION( 1, 2 ),                           ///< version 1.2
    ZE_API_VERSION_1_3 = ZE_MAKE_VERSION( 1, 3 ),                           ///< version 1.3
    ZE_API_VERSION_1_4 = ZE_MAKE_VERSION( 1, 4 ),                           ///< version 1.4
    ZE_API_VERSION_1_5 = ZE_MAKE_VERSION( 1, 5 ),                           ///< version 1.5
    ZE_API_VERSION_1_6 = ZE_MAKE_VERSION( 1, 6 ),                           ///< version 1.6
    ZE_API_VERSION_1_7 = ZE_MAKE_VERSION( 1, 7 ),                           ///< version 1.7
    ZE_API_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 7 ),                       ///< latest known version
    ZE_API_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_api_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the API version supported by the specified driver
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == version`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetApiVersion(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    ze_api_version_t* version                                               ///< [out] api version
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_DRIVER_UUID_SIZE
/// @brief Maximum driver universal unique id (UUID) size in bytes
#define ZE_MAX_DRIVER_UUID_SIZE  16
#endif // ZE_MAX_DRIVER_UUID_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Driver universal unique id (UUID)
typedef struct _ze_driver_uuid_t
{
    uint8_t id[ZE_MAX_DRIVER_UUID_SIZE];                                    ///< [out] opaque data representing a driver UUID

} ze_driver_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Driver properties queried using ::zeDriverGetProperties
typedef struct _ze_driver_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_driver_uuid_t uuid;                                                  ///< [out] universal unique identifier.
    uint32_t driverVersion;                                                 ///< [out] driver version
                                                                            ///< The driver version is a non-zero, monotonically increasing value where
                                                                            ///< higher values always indicate a more recent version.

} ze_driver_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves properties of the driver.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clGetPlatformInfo**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pDriverProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetProperties(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    ze_driver_properties_t* pDriverProperties                               ///< [in,out] query result for driver properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported IPC property flags
typedef uint32_t ze_ipc_property_flags_t;
typedef enum _ze_ipc_property_flag_t
{
    ZE_IPC_PROPERTY_FLAG_MEMORY = ZE_BIT(0),                                ///< Supports passing memory allocations between processes. See
                                                                            ///< ::zeMemGetIpcHandle.
    ZE_IPC_PROPERTY_FLAG_EVENT_POOL = ZE_BIT(1),                            ///< Supports passing event pools between processes. See
                                                                            ///< ::zeEventPoolGetIpcHandle.
    ZE_IPC_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_ipc_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief IPC properties queried using ::zeDriverGetIpcProperties
typedef struct _ze_driver_ipc_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_ipc_property_flags_t flags;                                          ///< [out] 0 (none) or a valid combination of ::ze_ipc_property_flag_t

} ze_driver_ipc_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves IPC attributes of the driver
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pIpcProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetIpcProperties(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    ze_driver_ipc_properties_t* pIpcProperties                              ///< [in,out] query result for IPC properties
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_EXTENSION_NAME
/// @brief Maximum extension name string size
#define ZE_MAX_EXTENSION_NAME  256
#endif // ZE_MAX_EXTENSION_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension properties queried using ::zeDriverGetExtensionProperties
typedef struct _ze_driver_extension_properties_t
{
    char name[ZE_MAX_EXTENSION_NAME];                                       ///< [out] extension name
    uint32_t version;                                                       ///< [out] extension version using ::ZE_MAKE_VERSION

} ze_driver_extension_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves extension properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **vkEnumerateInstanceExtensionProperties**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetExtensionProperties(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of extension properties.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of extension properties available.
                                                                            ///< if count is greater than the number of extension properties available,
                                                                            ///< then the driver shall update the value with the correct number of
                                                                            ///< extension properties available.
    ze_driver_extension_properties_t* pExtensionProperties                  ///< [in,out][optional][range(0, *pCount)] array of query results for
                                                                            ///< extension properties.
                                                                            ///< if count is less than the number of extension properties available,
                                                                            ///< then driver shall only retrieve that number of extension properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves function pointer for vendor-specific or experimental
///        extensions
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == name`
///         + `nullptr == ppFunctionAddress`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetExtensionFunctionAddress(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    const char* name,                                                       ///< [in] extension function name
    void** ppFunctionAddress                                                ///< [out] pointer to function pointer
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves a string describing the last error code returned by the
///        driver in the current thread.
/// 
/// @details
///     - String returned is thread local.
///     - String is only updated on calls returning an error, i.e., not on calls
///       returning ::ZE_RESULT_SUCCESS.
///     - String may be empty if driver considers error code is already explicit
///       enough to describe cause.
///     - Memory pointed to by ppString is owned by the driver.
///     - String returned is null-terminated.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ppString`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetLastErrorDescription(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    const char** ppString                                                   ///< [in,out] pointer to a null-terminated array of characters describing
                                                                            ///< cause of error.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Device
#if !defined(__GNUC__)
#pragma region device
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves devices within a driver
/// 
/// @details
///     - Multiple calls to this function will return identical device handles,
///       in the same order.
///     - The number and order of handles returned from this function is
///       affected by the ::ZE_AFFINITY_MASK and ::ZE_ENABLE_PCI_ID_DEVICE_ORDER
///       environment variables.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGet(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of devices.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of devices available.
                                                                            ///< if count is greater than the number of devices available, then the
                                                                            ///< driver shall update the value with the correct number of devices available.
    ze_device_handle_t* phDevices                                           ///< [in,out][optional][range(0, *pCount)] array of handle of devices.
                                                                            ///< if count is less than the number of devices available, then driver
                                                                            ///< shall only retrieve that number of devices.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves the root-device of a device handle
/// 
/// @details
///     - When the device handle passed does not belong to any root-device,
///       nullptr is returned.
///     - Multiple calls to this function will return the same device handle.
///     - The root-device handle returned by this function does not have access
///       automatically to the resources
///       created with the associated sub-device, unless those resources have
///       been created with a context
///       explicitly containing both handles.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phRootDevice`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetRootDevice(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    ze_device_handle_t* phRootDevice                                        ///< [in,out] parent root device.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves a sub-device from a device
/// 
/// @details
///     - When the device handle passed does not contain any sub-device, a
///       pCount of 0 is returned.
///     - Multiple calls to this function will return identical device handles,
///       in the same order.
///     - The number of handles returned from this function is affected by the
///       ::ZE_AFFINITY_MASK environment variable.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clCreateSubDevices
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetSubDevices(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of sub-devices.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of sub-devices available.
                                                                            ///< if count is greater than the number of sub-devices available, then the
                                                                            ///< driver shall update the value with the correct number of sub-devices available.
    ze_device_handle_t* phSubdevices                                        ///< [in,out][optional][range(0, *pCount)] array of handle of sub-devices.
                                                                            ///< if count is less than the number of sub-devices available, then driver
                                                                            ///< shall only retrieve that number of sub-devices.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum _ze_device_type_t
{
    ZE_DEVICE_TYPE_GPU = 1,                                                 ///< Graphics Processing Unit
    ZE_DEVICE_TYPE_CPU = 2,                                                 ///< Central Processing Unit
    ZE_DEVICE_TYPE_FPGA = 3,                                                ///< Field Programmable Gate Array
    ZE_DEVICE_TYPE_MCA = 4,                                                 ///< Memory Copy Accelerator
    ZE_DEVICE_TYPE_VPU = 5,                                                 ///< Vision Processing Unit
    ZE_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_device_type_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_DEVICE_UUID_SIZE
/// @brief Maximum device universal unique id (UUID) size in bytes
#define ZE_MAX_DEVICE_UUID_SIZE  16
#endif // ZE_MAX_DEVICE_UUID_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Device universal unique id (UUID)
typedef struct _ze_device_uuid_t
{
    uint8_t id[ZE_MAX_DEVICE_UUID_SIZE];                                    ///< [out] opaque data representing a device UUID

} ze_device_uuid_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_DEVICE_NAME
/// @brief Maximum device name string size
#define ZE_MAX_DEVICE_NAME  256
#endif // ZE_MAX_DEVICE_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device property flags
typedef uint32_t ze_device_property_flags_t;
typedef enum _ze_device_property_flag_t
{
    ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = ZE_BIT(0),                         ///< Device is integrated with the Host.
    ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE = ZE_BIT(1),                          ///< Device handle used for query represents a sub-device.
    ZE_DEVICE_PROPERTY_FLAG_ECC = ZE_BIT(2),                                ///< Device supports error correction memory access.
    ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = ZE_BIT(3),                     ///< Device supports on-demand page-faulting.
    ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device properties queried using ::zeDeviceGetProperties
typedef struct _ze_device_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_type_t type;                                                  ///< [out] generic device type
    uint32_t vendorId;                                                      ///< [out] vendor id from PCI configuration
    uint32_t deviceId;                                                      ///< [out] device id from PCI configuration
                                                                            ///< Note, the device id uses little-endian format.
    ze_device_property_flags_t flags;                                       ///< [out] 0 (none) or a valid combination of ::ze_device_property_flag_t
    uint32_t subdeviceId;                                                   ///< [out] sub-device id. Only valid if ::ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE
                                                                            ///< is set.
    uint32_t coreClockRate;                                                 ///< [out] Clock rate for device core.
    uint64_t maxMemAllocSize;                                               ///< [out] Maximum memory allocation size.
    uint32_t maxHardwareContexts;                                           ///< [out] Maximum number of logical hardware contexts.
    uint32_t maxCommandQueuePriority;                                       ///< [out] Maximum priority for command queues. Higher value is higher
                                                                            ///< priority.
    uint32_t numThreadsPerEU;                                               ///< [out] Maximum number of threads per EU.
    uint32_t physicalEUSimdWidth;                                           ///< [out] The physical EU simd width.
    uint32_t numEUsPerSubslice;                                             ///< [out] Maximum number of EUs per sub-slice.
    uint32_t numSubslicesPerSlice;                                          ///< [out] Maximum number of sub-slices per slice.
    uint32_t numSlices;                                                     ///< [out] Maximum number of slices.
    uint64_t timerResolution;                                               ///< [out] Returns the resolution of device timer used for profiling,
                                                                            ///< timestamps, etc. When stype==::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES the
                                                                            ///< units are in nanoseconds. When
                                                                            ///< stype==::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 units are in
                                                                            ///< cycles/sec
    uint32_t timestampValidBits;                                            ///< [out] Returns the number of valid bits in the timestamp value.
    uint32_t kernelTimestampValidBits;                                      ///< [out] Returns the number of valid bits in the kernel timestamp values
    ze_device_uuid_t uuid;                                                  ///< [out] universal unique identifier. Note: Subdevices will have their
                                                                            ///< own uuid.
    char name[ZE_MAX_DEVICE_NAME];                                          ///< [out] Device name

} ze_device_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device thread identifier.
typedef struct _ze_device_thread_t
{
    uint32_t slice;                                                         ///< [in,out] the slice number.
                                                                            ///< Must be `UINT32_MAX` (all) or less than the `numSlices` member of ::ze_device_properties_t.
    uint32_t subslice;                                                      ///< [in,out] the sub-slice number within its slice.
                                                                            ///< Must be `UINT32_MAX` (all) or less than the `numSubslicesPerSlice`
                                                                            ///< member of ::ze_device_properties_t.
    uint32_t eu;                                                            ///< [in,out] the EU number within its sub-slice.
                                                                            ///< Must be `UINT32_MAX` (all) or less than the `numEUsPerSubslice` member
                                                                            ///< of ::ze_device_properties_t.
    uint32_t thread;                                                        ///< [in,out] the thread number within its EU.
                                                                            ///< Must be `UINT32_MAX` (all) or less than the `numThreadsPerEU` member
                                                                            ///< of ::ze_device_properties_t.

} ze_device_thread_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves properties of the device.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clGetDeviceInfo
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pDeviceProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_device_properties_t* pDeviceProperties                               ///< [in,out] query result for device properties
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_SUBGROUPSIZE_COUNT
/// @brief Maximum number of subgroup sizes supported.
#define ZE_SUBGROUPSIZE_COUNT  8
#endif // ZE_SUBGROUPSIZE_COUNT

///////////////////////////////////////////////////////////////////////////////
/// @brief Device compute properties queried using ::zeDeviceGetComputeProperties
typedef struct _ze_device_compute_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t maxTotalGroupSize;                                             ///< [out] Maximum items per compute group. (groupSizeX * groupSizeY *
                                                                            ///< groupSizeZ) <= maxTotalGroupSize
    uint32_t maxGroupSizeX;                                                 ///< [out] Maximum items for X dimension in group
    uint32_t maxGroupSizeY;                                                 ///< [out] Maximum items for Y dimension in group
    uint32_t maxGroupSizeZ;                                                 ///< [out] Maximum items for Z dimension in group
    uint32_t maxGroupCountX;                                                ///< [out] Maximum groups that can be launched for x dimension
    uint32_t maxGroupCountY;                                                ///< [out] Maximum groups that can be launched for y dimension
    uint32_t maxGroupCountZ;                                                ///< [out] Maximum groups that can be launched for z dimension
    uint32_t maxSharedLocalMemory;                                          ///< [out] Maximum shared local memory per group.
    uint32_t numSubGroupSizes;                                              ///< [out] Number of subgroup sizes supported. This indicates number of
                                                                            ///< entries in subGroupSizes.
    uint32_t subGroupSizes[ZE_SUBGROUPSIZE_COUNT];                          ///< [out] Size group sizes supported.

} ze_device_compute_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves compute properties of the device.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clGetDeviceInfo
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pComputeProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetComputeProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_device_compute_properties_t* pComputeProperties                      ///< [in,out] query result for compute properties
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_NATIVE_KERNEL_UUID_SIZE
/// @brief Maximum native kernel universal unique id (UUID) size in bytes
#define ZE_MAX_NATIVE_KERNEL_UUID_SIZE  16
#endif // ZE_MAX_NATIVE_KERNEL_UUID_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Native kernel universal unique id (UUID)
typedef struct _ze_native_kernel_uuid_t
{
    uint8_t id[ZE_MAX_NATIVE_KERNEL_UUID_SIZE];                             ///< [out] opaque data representing a native kernel UUID

} ze_native_kernel_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device module flags
typedef uint32_t ze_device_module_flags_t;
typedef enum _ze_device_module_flag_t
{
    ZE_DEVICE_MODULE_FLAG_FP16 = ZE_BIT(0),                                 ///< Device supports 16-bit floating-point operations
    ZE_DEVICE_MODULE_FLAG_FP64 = ZE_BIT(1),                                 ///< Device supports 64-bit floating-point operations
    ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS = ZE_BIT(2),                        ///< Device supports 64-bit atomic operations
    ZE_DEVICE_MODULE_FLAG_DP4A = ZE_BIT(3),                                 ///< Device supports four component dot product and accumulate operations
    ZE_DEVICE_MODULE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_module_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported floating-Point capability flags
typedef uint32_t ze_device_fp_flags_t;
typedef enum _ze_device_fp_flag_t
{
    ZE_DEVICE_FP_FLAG_DENORM = ZE_BIT(0),                                   ///< Supports denorms
    ZE_DEVICE_FP_FLAG_INF_NAN = ZE_BIT(1),                                  ///< Supports INF and quiet NaNs
    ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST = ZE_BIT(2),                         ///< Supports rounding to nearest even rounding mode
    ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO = ZE_BIT(3),                            ///< Supports rounding to zero.
    ZE_DEVICE_FP_FLAG_ROUND_TO_INF = ZE_BIT(4),                             ///< Supports rounding to both positive and negative INF.
    ZE_DEVICE_FP_FLAG_FMA = ZE_BIT(5),                                      ///< Supports IEEE754-2008 fused multiply-add.
    ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT = ZE_BIT(6),                      ///< Supports rounding as defined by IEEE754 for divide and sqrt
                                                                            ///< operations.
    ZE_DEVICE_FP_FLAG_SOFT_FLOAT = ZE_BIT(7),                               ///< Uses software implementation for basic floating-point operations.
    ZE_DEVICE_FP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_fp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device module properties queried using ::zeDeviceGetModuleProperties
typedef struct _ze_device_module_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t spirvVersionSupported;                                         ///< [out] Maximum supported SPIR-V version.
                                                                            ///< Returns zero if SPIR-V is not supported.
                                                                            ///< Contains major and minor attributes, use ::ZE_MAJOR_VERSION and ::ZE_MINOR_VERSION.
    ze_device_module_flags_t flags;                                         ///< [out] 0 or a valid combination of ::ze_device_module_flag_t
    ze_device_fp_flags_t fp16flags;                                         ///< [out] Capabilities for half-precision floating-point operations.
                                                                            ///< returns 0 (if ::ZE_DEVICE_MODULE_FLAG_FP16 is not set) or a
                                                                            ///< combination of ::ze_device_fp_flag_t.
    ze_device_fp_flags_t fp32flags;                                         ///< [out] Capabilities for single-precision floating-point operations.
                                                                            ///< returns a combination of ::ze_device_fp_flag_t.
    ze_device_fp_flags_t fp64flags;                                         ///< [out] Capabilities for double-precision floating-point operations.
                                                                            ///< returns 0 (if ::ZE_DEVICE_MODULE_FLAG_FP64 is not set) or a
                                                                            ///< combination of ::ze_device_fp_flag_t.
    uint32_t maxArgumentsSize;                                              ///< [out] Maximum kernel argument size that is supported.
    uint32_t printfBufferSize;                                              ///< [out] Maximum size of internal buffer that holds output of printf
                                                                            ///< calls from kernel.
    ze_native_kernel_uuid_t nativeKernelSupported;                          ///< [out] Compatibility UUID of supported native kernel.
                                                                            ///< UUID may or may not be the same across driver release, devices, or
                                                                            ///< operating systems.
                                                                            ///< Application is responsible for ensuring UUID matches before creating
                                                                            ///< module using
                                                                            ///< previously created native kernel.

} ze_device_module_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves module properties of the device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pModuleProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetModuleProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_device_module_properties_t* pModuleProperties                        ///< [in,out] query result for module properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported command queue group property flags
typedef uint32_t ze_command_queue_group_property_flags_t;
typedef enum _ze_command_queue_group_property_flag_t
{
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = ZE_BIT(0),               ///< Command queue group supports enqueing compute commands.
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY = ZE_BIT(1),                  ///< Command queue group supports enqueing copy commands.
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS = ZE_BIT(2),   ///< Command queue group supports cooperative kernels.
                                                                            ///< See ::zeCommandListAppendLaunchCooperativeKernel for more details.
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS = ZE_BIT(3),               ///< Command queue groups supports metric queries.
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_group_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command queue group properties queried using
///        ::zeDeviceGetCommandQueueGroupProperties
typedef struct _ze_command_queue_group_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_command_queue_group_property_flags_t flags;                          ///< [out] 0 (none) or a valid combination of
                                                                            ///< ::ze_command_queue_group_property_flag_t
    size_t maxMemoryFillPatternSize;                                        ///< [out] maximum `pattern_size` supported by command queue group.
                                                                            ///< See ::zeCommandListAppendMemoryFill for more details.
    uint32_t numQueues;                                                     ///< [out] the number of physical engines within the group.

} ze_command_queue_group_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves command queue group properties of the device.
/// 
/// @details
///     - Properties are reported for each physical command queue type supported
///       by the device.
///     - Multiple calls to this function will return properties in the same
///       order.
///     - The order in which the properties are returned defines the command
///       queue group's ordinal.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **vkGetPhysicalDeviceQueueFamilyProperties**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetCommandQueueGroupProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of command queue group properties.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of command queue group properties available.
                                                                            ///< if count is greater than the number of command queue group properties
                                                                            ///< available, then the driver shall update the value with the correct
                                                                            ///< number of command queue group properties available.
    ze_command_queue_group_properties_t* pCommandQueueGroupProperties       ///< [in,out][optional][range(0, *pCount)] array of query results for
                                                                            ///< command queue group properties.
                                                                            ///< if count is less than the number of command queue group properties
                                                                            ///< available, then driver shall only retrieve that number of command
                                                                            ///< queue group properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device memory property flags
typedef uint32_t ze_device_memory_property_flags_t;
typedef enum _ze_device_memory_property_flag_t
{
    ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD = ZE_BIT(0),                         ///< reserved for future use
    ZE_DEVICE_MEMORY_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_memory_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device local memory properties queried using
///        ::zeDeviceGetMemoryProperties
typedef struct _ze_device_memory_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_memory_property_flags_t flags;                                ///< [out] 0 (none) or a valid combination of
                                                                            ///< ::ze_device_memory_property_flag_t
    uint32_t maxClockRate;                                                  ///< [out] Maximum clock rate for device memory.
    uint32_t maxBusWidth;                                                   ///< [out] Maximum bus width between device and memory.
    uint64_t totalSize;                                                     ///< [out] Total memory size in bytes that is available to the device.
    char name[ZE_MAX_DEVICE_NAME];                                          ///< [out] Memory name

} ze_device_memory_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves local memory properties of the device.
/// 
/// @details
///     - Properties are reported for each physical memory type supported by the
///       device.
///     - Multiple calls to this function will return properties in the same
///       order.
///     - The order in which the properties are returned defines the device's
///       local memory ordinal.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clGetDeviceInfo
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetMemoryProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of memory properties.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of memory properties available.
                                                                            ///< if count is greater than the number of memory properties available,
                                                                            ///< then the driver shall update the value with the correct number of
                                                                            ///< memory properties available.
    ze_device_memory_properties_t* pMemProperties                           ///< [in,out][optional][range(0, *pCount)] array of query results for
                                                                            ///< memory properties.
                                                                            ///< if count is less than the number of memory properties available, then
                                                                            ///< driver shall only retrieve that number of memory properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory access capability flags
/// 
/// @details
///     - Supported access capabilities for different types of memory
///       allocations
typedef uint32_t ze_memory_access_cap_flags_t;
typedef enum _ze_memory_access_cap_flag_t
{
    ZE_MEMORY_ACCESS_CAP_FLAG_RW = ZE_BIT(0),                               ///< Supports load/store access
    ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC = ZE_BIT(1),                           ///< Supports atomic access
    ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT = ZE_BIT(2),                       ///< Supports concurrent access
    ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC = ZE_BIT(3),                ///< Supports concurrent atomic access
    ZE_MEMORY_ACCESS_CAP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_memory_access_cap_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device memory access properties queried using
///        ::zeDeviceGetMemoryAccessProperties
typedef struct _ze_device_memory_access_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_memory_access_cap_flags_t hostAllocCapabilities;                     ///< [out] host memory capabilities.
                                                                            ///< returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
    ze_memory_access_cap_flags_t deviceAllocCapabilities;                   ///< [out] device memory capabilities.
                                                                            ///< returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
    ze_memory_access_cap_flags_t sharedSingleDeviceAllocCapabilities;       ///< [out] shared, single-device memory capabilities.
                                                                            ///< returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
    ze_memory_access_cap_flags_t sharedCrossDeviceAllocCapabilities;        ///< [out] shared, cross-device memory capabilities.
                                                                            ///< returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
    ze_memory_access_cap_flags_t sharedSystemAllocCapabilities;             ///< [out] shared, system memory capabilities.
                                                                            ///< returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.

} ze_device_memory_access_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves memory access properties of the device.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clGetDeviceInfo
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pMemAccessProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetMemoryAccessProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_device_memory_access_properties_t* pMemAccessProperties              ///< [in,out] query result for memory access properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported cache control property flags
typedef uint32_t ze_device_cache_property_flags_t;
typedef enum _ze_device_cache_property_flag_t
{
    ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL = ZE_BIT(0),                 ///< Device support User Cache Control (i.e. SLM section vs Generic Cache)
    ZE_DEVICE_CACHE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_cache_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device cache properties queried using ::zeDeviceGetCacheProperties
typedef struct _ze_device_cache_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_cache_property_flags_t flags;                                 ///< [out] 0 (none) or a valid combination of
                                                                            ///< ::ze_device_cache_property_flag_t
    size_t cacheSize;                                                       ///< [out] Per-cache size, in bytes

} ze_device_cache_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves cache properties of the device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clGetDeviceInfo
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetCacheProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of cache properties.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of cache properties available.
                                                                            ///< if count is greater than the number of cache properties available,
                                                                            ///< then the driver shall update the value with the correct number of
                                                                            ///< cache properties available.
    ze_device_cache_properties_t* pCacheProperties                          ///< [in,out][optional][range(0, *pCount)] array of query results for cache properties.
                                                                            ///< if count is less than the number of cache properties available, then
                                                                            ///< driver shall only retrieve that number of cache properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Device image properties queried using ::zeDeviceGetImageProperties
typedef struct _ze_device_image_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t maxImageDims1D;                                                ///< [out] Maximum image dimensions for 1D resources. if 0, then 1D images
                                                                            ///< are unsupported.
    uint32_t maxImageDims2D;                                                ///< [out] Maximum image dimensions for 2D resources. if 0, then 2D images
                                                                            ///< are unsupported.
    uint32_t maxImageDims3D;                                                ///< [out] Maximum image dimensions for 3D resources. if 0, then 3D images
                                                                            ///< are unsupported.
    uint64_t maxImageBufferSize;                                            ///< [out] Maximum image buffer size in bytes. if 0, then buffer images are
                                                                            ///< unsupported.
    uint32_t maxImageArraySlices;                                           ///< [out] Maximum image array slices. if 0, then image arrays are
                                                                            ///< unsupported.
    uint32_t maxSamplers;                                                   ///< [out] Max samplers that can be used in kernel. if 0, then sampling is
                                                                            ///< unsupported.
    uint32_t maxReadImageArgs;                                              ///< [out] Returns the maximum number of simultaneous image objects that
                                                                            ///< can be read from by a kernel. if 0, then reading images is
                                                                            ///< unsupported.
    uint32_t maxWriteImageArgs;                                             ///< [out] Returns the maximum number of simultaneous image objects that
                                                                            ///< can be written to by a kernel. if 0, then writing images is
                                                                            ///< unsupported.

} ze_device_image_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves image properties of the device
/// 
/// @details
///     - See ::zeImageGetProperties for format-specific capabilities.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pImageProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetImageProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_device_image_properties_t* pImageProperties                          ///< [in,out] query result for image properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Device external memory import and export properties
typedef struct _ze_device_external_memory_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_external_memory_type_flags_t memoryAllocationImportTypes;            ///< [out] Supported external memory import types for memory allocations.
    ze_external_memory_type_flags_t memoryAllocationExportTypes;            ///< [out] Supported external memory export types for memory allocations.
    ze_external_memory_type_flags_t imageImportTypes;                       ///< [out] Supported external memory import types for images.
    ze_external_memory_type_flags_t imageExportTypes;                       ///< [out] Supported external memory export types for images.

} ze_device_external_memory_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves external memory import and export of the device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pExternalMemoryProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetExternalMemoryProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_device_external_memory_properties_t* pExternalMemoryProperties       ///< [in,out] query result for external memory properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device peer-to-peer property flags
typedef uint32_t ze_device_p2p_property_flags_t;
typedef enum _ze_device_p2p_property_flag_t
{
    ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS = ZE_BIT(0),                         ///< Device supports access between peer devices.
    ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS = ZE_BIT(1),                        ///< Device supports atomics between peer devices.
    ZE_DEVICE_P2P_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_p2p_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device peer-to-peer properties queried using
///        ::zeDeviceGetP2PProperties
typedef struct _ze_device_p2p_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_p2p_property_flags_t flags;                                   ///< [out] 0 (none) or a valid combination of
                                                                            ///< ::ze_device_p2p_property_flag_t

} ze_device_p2p_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves peer-to-peer properties between one device and a peer
///        devices
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///         + `nullptr == hPeerDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pP2PProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetP2PProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device performing the access
    ze_device_handle_t hPeerDevice,                                         ///< [in] handle of the peer device with the allocation
    ze_device_p2p_properties_t* pP2PProperties                              ///< [in,out] Peer-to-Peer properties between source and peer device
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries if one device can directly access peer device allocations
/// 
/// @details
///     - Any device can access any other device within a node through a
///       scale-up fabric.
///     - The following are conditions for CanAccessPeer query.
///         + If both device and peer device are the same then return true.
///         + If both sub-device and peer sub-device are the same then return
///           true.
///         + If both are sub-devices and share the same parent device then
///           return true.
///         + If both device and remote device are connected by a direct or
///           indirect scale-up fabric or over PCIe (same root complex or shared
///           PCIe switch) then true.
///         + If both sub-device and remote parent device (and vice-versa) are
///           connected by a direct or indirect scale-up fabric or over PCIe
///           (same root complex or shared PCIe switch) then true.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///         + `nullptr == hPeerDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == value`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceCanAccessPeer(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device performing the access
    ze_device_handle_t hPeerDevice,                                         ///< [in] handle of the peer device with the allocation
    ze_bool_t* value                                                        ///< [out] returned access capability
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns current status of the device.
/// 
/// @details
///     - Once a device is reset, this call will update the OS handle attached
///       to the device handle.
///     - The application may call this function from simultaneous threads with
///       the same device handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_SUCCESS
///         + Device is available for use.
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///         + Device is lost; must be reset for use.
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetStatus(
    ze_device_handle_t hDevice                                              ///< [in] handle of the device
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns synchronized Host and device global timestamps.
/// 
/// @details
///     - The application may call this function from simultaneous threads with
///       the same device handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == hostTimestamp`
///         + `nullptr == deviceTimestamp`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetGlobalTimestamps(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    uint64_t* hostTimestamp,                                                ///< [out] value of the Host's global timestamp that correlates with the
                                                                            ///< Device's global timestamp value
    uint64_t* deviceTimestamp                                               ///< [out] value of the Device's global timestamp that correlates with the
                                                                            ///< Host's global timestamp value
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Context
#if !defined(__GNUC__)
#pragma region context
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported context creation flags
typedef uint32_t ze_context_flags_t;
typedef enum _ze_context_flag_t
{
    ZE_CONTEXT_FLAG_TBD = ZE_BIT(0),                                        ///< reserved for future use
    ZE_CONTEXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_context_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Context descriptor
typedef struct _ze_context_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_context_flags_t flags;                                               ///< [in] creation flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_context_flag_t;
                                                                            ///< default behavior may use implicit driver-based heuristics.

} ze_context_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a context for the driver.
/// 
/// @details
///     - The application must only use the context for the driver which was
///       provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phContext`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < desc->flags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextCreate(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver object
    const ze_context_desc_t* desc,                                          ///< [in] pointer to context descriptor
    ze_context_handle_t* phContext                                          ///< [out] pointer to handle of context object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a context for the driver.
/// 
/// @details
///     - The application must only use the context for the driver which was
///       provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phContext`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < desc->flags`
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phDevices) && (0 < numDevices)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextCreateEx(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver object
    const ze_context_desc_t* desc,                                          ///< [in] pointer to context descriptor
    uint32_t numDevices,                                                    ///< [in][optional] number of device handles; must be 0 if `nullptr ==
                                                                            ///< phDevices`
    ze_device_handle_t* phDevices,                                          ///< [in][optional][range(0, numDevices)] array of device handles which
                                                                            ///< context has visibility.
                                                                            ///< if nullptr, then all devices and any sub-devices supported by the
                                                                            ///< driver instance are
                                                                            ///< visible to the context.
                                                                            ///< otherwise, the context only has visibility to the devices and any
                                                                            ///< sub-devices of the
                                                                            ///< devices in this array.
    ze_context_handle_t* phContext                                          ///< [out] pointer to handle of context object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a context.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the context before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this context.
///     - The application must **not** call this function from simultaneous
///       threads with the same context handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextDestroy(
    ze_context_handle_t hContext                                            ///< [in][release] handle of context object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns current status of the context.
/// 
/// @details
///     - The application may call this function from simultaneous threads with
///       the same context handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_SUCCESS
///         + Context is available for use.
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///         + Context is invalid; due to device lost or reset.
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextGetStatus(
    ze_context_handle_t hContext                                            ///< [in] handle of context object
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Command Queue
#if !defined(__GNUC__)
#pragma region cmdqueue
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported command queue flags
typedef uint32_t ze_command_queue_flags_t;
typedef enum _ze_command_queue_flag_t
{
    ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY = ZE_BIT(0),                        ///< command queue should be optimized for submission to a single device engine.
                                                                            ///< driver **must** disable any implicit optimizations for distributing
                                                                            ///< work across multiple engines.
                                                                            ///< this flag should be used when applications want full control over
                                                                            ///< multi-engine submission and scheduling.
    ZE_COMMAND_QUEUE_FLAG_IN_ORDER = ZE_BIT(1),                             ///< To be used only when creating immediate command lists. Commands
                                                                            ///< appended to the immediate command
                                                                            ///< list are executed in-order, with driver implementation enforcing
                                                                            ///< dependencies between them.
                                                                            ///< Application is not required to have the signal event of a given
                                                                            ///< command being the wait event of
                                                                            ///< the next to define an in-order list, and application is allowed to
                                                                            ///< pass signal and wait events
                                                                            ///< to each appended command to implement more complex dependency graphs.
    ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported command queue modes
typedef enum _ze_command_queue_mode_t
{
    ZE_COMMAND_QUEUE_MODE_DEFAULT = 0,                                      ///< implicit default behavior; uses driver-based heuristics
    ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1,                                  ///< Device execution always completes immediately on execute;
                                                                            ///< Host thread is blocked using wait on implicit synchronization object
    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2,                                 ///< Device execution is scheduled and will complete in future;
                                                                            ///< explicit synchronization object must be used to determine completeness
    ZE_COMMAND_QUEUE_MODE_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported command queue priorities
typedef enum _ze_command_queue_priority_t
{
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0,                                   ///< [default] normal priority
    ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW = 1,                             ///< lower priority than normal
    ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH = 2,                            ///< higher priority than normal
    ZE_COMMAND_QUEUE_PRIORITY_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_priority_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command Queue descriptor
typedef struct _ze_command_queue_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t ordinal;                                                       ///< [in] command queue group ordinal
    uint32_t index;                                                         ///< [in] command queue index within the group;
                                                                            ///< must be zero if ::ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY is not set
    ze_command_queue_flags_t flags;                                         ///< [in] usage flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_command_queue_flag_t;
                                                                            ///< default behavior may use implicit driver-based heuristics to balance
                                                                            ///< latency and throughput.
    ze_command_queue_mode_t mode;                                           ///< [in] operation mode
    ze_command_queue_priority_t priority;                                   ///< [in] priority

} ze_command_queue_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a command queue on the context.
/// 
/// @details
///     - A command queue represents a logical input stream to the device, tied
///       to a physical input stream.
///     - The application must only use the command queue for the device, or its
///       sub-devices, which was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @remarks
///   _Analogues_
///     - **clCreateCommandQueue**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phCommandQueue`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < desc->flags`
///         + `::ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS < desc->mode`
///         + `::ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH < desc->priority`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueCreate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    const ze_command_queue_desc_t* desc,                                    ///< [in] pointer to command queue descriptor
    ze_command_queue_handle_t* phCommandQueue                               ///< [out] pointer to handle of command queue object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a command queue.
/// 
/// @details
///     - The application must destroy all fence handles created from the
///       command queue before destroying the command queue itself
///     - The application must ensure the device is not currently referencing
///       the command queue before it is deleted
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this command queue
///     - The application must **not** call this function from simultaneous
///       threads with the same command queue handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @remarks
///   _Analogues_
///     - **clReleaseCommandQueue**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandQueue`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueDestroy(
    ze_command_queue_handle_t hCommandQueue                                 ///< [in][release] handle of command queue object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Executes a command list in a command queue.
/// 
/// @details
///     - The command lists are submitted to the device in the order they are
///       received, whether from multiple calls (on the same or different
///       threads) or a single call with multiple command lists.
///     - The application must ensure the command lists are accessible by the
///       device on which the command queue was created.
///     - The application must ensure the device is not currently referencing
///       the command list since the implementation is allowed to modify the
///       contents of the command list for submission.
///     - The application must only execute command lists created with an
///       identical command queue group ordinal to the command queue.
///     - The application must use a fence created using the same command queue.
///     - The application must ensure the command queue, command list and fence
///       were created on the same context.
///     - The application must ensure the command lists being executed are not
///       immediate command lists.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - vkQueueSubmit
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandQueue`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phCommandLists`
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `0 == numCommandLists`
///     - ::ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueExecuteCommandLists(
    ze_command_queue_handle_t hCommandQueue,                                ///< [in] handle of the command queue
    uint32_t numCommandLists,                                               ///< [in] number of command lists to execute
    ze_command_list_handle_t* phCommandLists,                               ///< [in][range(0, numCommandLists)] list of handles of the command lists
                                                                            ///< to execute
    ze_fence_handle_t hFence                                                ///< [in][optional] handle of the fence to signal on completion
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Synchronizes a command queue by waiting on the host.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandQueue`
///     - ::ZE_RESULT_NOT_READY
///         + timeout expired
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueSynchronize(
    ze_command_queue_handle_t hCommandQueue,                                ///< [in] handle of the command queue
    uint64_t timeout                                                        ///< [in] if non-zero, then indicates the maximum time (in nanoseconds) to
                                                                            ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                                            ///< if zero, then immediately returns the status of the command queue;
                                                                            ///< if `UINT64_MAX`, then function will not return until complete or
                                                                            ///< device is lost.
                                                                            ///< Due to external dependencies, timeout may be rounded to the closest
                                                                            ///< value allowed by the accuracy of those dependencies.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Command List
#if !defined(__GNUC__)
#pragma region cmdlist
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported command list creation flags
typedef uint32_t ze_command_list_flags_t;
typedef enum _ze_command_list_flag_t
{
    ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING = ZE_BIT(0),                      ///< driver may reorder commands (e.g., kernels, copies) between barriers
                                                                            ///< and synchronization primitives.
                                                                            ///< using this flag may increase Host overhead of ::zeCommandListClose.
                                                                            ///< therefore, this flag should **not** be set for low-latency usage-models.
    ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT = ZE_BIT(1),                   ///< driver may perform additional optimizations that increase execution
                                                                            ///< throughput. 
                                                                            ///< using this flag may increase Host overhead of ::zeCommandListClose and ::zeCommandQueueExecuteCommandLists.
                                                                            ///< therefore, this flag should **not** be set for low-latency usage-models.
    ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY = ZE_BIT(2),                         ///< command list should be optimized for submission to a single command
                                                                            ///< queue and device engine.
                                                                            ///< driver **must** disable any implicit optimizations for distributing
                                                                            ///< work across multiple engines.
                                                                            ///< this flag should be used when applications want full control over
                                                                            ///< multi-engine submission and scheduling.
    ZE_COMMAND_LIST_FLAG_IN_ORDER = ZE_BIT(3),                              ///< commands appended to this command list are executed in-order, with
                                                                            ///< driver implementation
                                                                            ///< enforcing dependencies between them. Application is not required to
                                                                            ///< have the signal event
                                                                            ///< of a given command being the wait event of the next to define an
                                                                            ///< in-order list, and
                                                                            ///< application is allowed to pass signal and wait events to each appended
                                                                            ///< command to implement
                                                                            ///< more complex dependency graphs. Cannot be combined with ::ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING.
    ZE_COMMAND_LIST_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_command_list_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command List descriptor
typedef struct _ze_command_list_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t commandQueueGroupOrdinal;                                      ///< [in] command queue group ordinal to which this command list will be
                                                                            ///< submitted
    ze_command_list_flags_t flags;                                          ///< [in] usage flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_command_list_flag_t;
                                                                            ///< default behavior may use implicit driver-based heuristics to balance
                                                                            ///< latency and throughput.

} ze_command_list_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a command list on the context.
/// 
/// @details
///     - A command list represents a sequence of commands for execution on a
///       command queue.
///     - The command list is created in the 'open' state.
///     - The application must only use the command list for the device, or its
///       sub-devices, which was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0xf < desc->flags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListCreate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    const ze_command_list_desc_t* desc,                                     ///< [in] pointer to command list descriptor
    ze_command_list_handle_t* phCommandList                                 ///< [out] pointer to handle of command list object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates an immediate command list on the context.
/// 
/// @details
///     - An immediate command list is used for low-latency submission of
///       commands.
///     - An immediate command list creates an implicit command queue.
///     - Immediate command lists must not be passed to
///       ::zeCommandQueueExecuteCommandLists.
///     - Commands appended into an immediate command list may execute
///       synchronously, by blocking until the command is complete.
///     - The command list is created in the 'open' state and never needs to be
///       closed.
///     - The application must only use the command list for the device, or its
///       sub-devices, which was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == altdesc`
///         + `nullptr == phCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < altdesc->flags`
///         + `::ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS < altdesc->mode`
///         + `::ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH < altdesc->priority`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListCreateImmediate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    const ze_command_queue_desc_t* altdesc,                                 ///< [in] pointer to command queue descriptor
    ze_command_list_handle_t* phCommandList                                 ///< [out] pointer to handle of command list object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a command list.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the command list before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this command list.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListDestroy(
    ze_command_list_handle_t hCommandList                                   ///< [in][release] handle of command list object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Closes a command list; ready to be executed by a command queue.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListClose(
    ze_command_list_handle_t hCommandList                                   ///< [in] handle of command list object to close
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Reset a command list to initial (empty) state; ready for appending
///        commands.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the command list before it is reset
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListReset(
    ze_command_list_handle_t hCommandList                                   ///< [in] handle of command list object to reset
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends a memory write of the device's global timestamp value into a
///        command list.
/// 
/// @details
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The timestamp frequency can be queried from the `timerResolution`
///       member of ::ze_device_properties_t.
///     - The number of valid bits in the timestamp value can be queried from
///       the `timestampValidBits` member of ::ze_device_properties_t.
///     - The application must ensure the memory pointed to by dstptr is
///       accessible by the device on which the command list was created.
///     - The application must ensure the command list and events were created,
///       and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendWriteGlobalTimestamp(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    uint64_t* dstptr,                                                       ///< [in,out] pointer to memory where timestamp value will be written; must
                                                                            ///< be 8byte-aligned.
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before executing query;
                                                                            ///< must be 0 if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before executing query
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Synchronizes an immediate command list by waiting on the host for the
///        completion of all commands previously submitted to it.
/// 
/// @details
///     - The application must call this function only with command lists
///       created with ::zeCommandListCreateImmediate.
///     - Waiting on one immediate command list shall not block the concurrent
///       execution of commands appended to other
///       immediate command lists created with either a different ordinal or
///       different index.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_NOT_READY
///         + timeout expired
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + handle does not correspond to an immediate command list
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListHostSynchronize(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the immediate command list
    uint64_t timeout                                                        ///< [in] if non-zero, then indicates the maximum time (in nanoseconds) to
                                                                            ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                                            ///< if zero, then immediately returns the status of the immediate command list;
                                                                            ///< if `UINT64_MAX`, then function will not return until complete or
                                                                            ///< device is lost.
                                                                            ///< Due to external dependencies, timeout may be rounded to the closest
                                                                            ///< value allowed by the accuracy of those dependencies.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Barrier
#if !defined(__GNUC__)
#pragma region barrier
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Appends an execution and global memory barrier into a command list.
/// 
/// @details
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - If numWaitEvents is zero, then all previous commands, enqueued on same
///       command queue, must complete prior to the execution of the barrier.
///       This is not the case when numWaitEvents is non-zero.
///     - If numWaitEvents is non-zero, then only all phWaitEvents must be
///       signaled prior to the execution of the barrier.
///     - This command blocks all following commands from beginning until the
///       execution of the barrier completes.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **vkCmdPipelineBarrier**
///     - clEnqueueBarrierWithWaitList
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendBarrier(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before executing barrier;
                                                                            ///< must be 0 if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before executing barrier
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends a global memory ranges barrier into a command list.
/// 
/// @details
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - If numWaitEvents is zero, then all previous commands are completed
///       prior to the execution of the barrier.
///     - If numWaitEvents is non-zero, then then all phWaitEvents must be
///       signaled prior to the execution of the barrier.
///     - This command blocks all following commands from beginning until the
///       execution of the barrier completes.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRangeSizes`
///         + `nullptr == pRanges`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryRangesBarrier(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    uint32_t numRanges,                                                     ///< [in] number of memory ranges
    const size_t* pRangeSizes,                                              ///< [in][range(0, numRanges)] array of sizes of memory range
    const void** pRanges,                                                   ///< [in][range(0, numRanges)] array of memory ranges
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before executing barrier;
                                                                            ///< must be 0 if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before executing barrier
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Ensures in-bound writes to the device are globally observable.
/// 
/// @details
///     - This is a special-case system level barrier that can be used to ensure
///       global observability of writes; 
///       typically needed after a producer (e.g., NIC) performs direct writes
///       to the device's memory (e.g., Direct RDMA writes).
///       This is typically required when the memory corresponding to the writes
///       is subsequently accessed from a remote device.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextSystemBarrier(
    ze_context_handle_t hContext,                                           ///< [in] handle of context object
    ze_device_handle_t hDevice                                              ///< [in] handle of the device
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Copies
#if !defined(__GNUC__)
#pragma region copy
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Copies host, device, or shared memory.
/// 
/// @details
///     - The application must ensure the memory pointed to by dstptr and srcptr
///       is accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by dstptr and
///       srcptr as they are free to be modified by either the Host or device up
///       until execution.
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The application must ensure the command list and events were created,
///       and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyBuffer**
///     - **clEnqueueReadBuffer**
///     - **clEnqueueWriteBuffer**
///     - **clEnqueueSVMMemcpy**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstptr`
///         + `nullptr == srcptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryCopy(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    void* dstptr,                                                           ///< [in] pointer to destination memory to copy to
    const void* srcptr,                                                     ///< [in] pointer to source memory to copy from
    size_t size,                                                            ///< [in] size in bytes to copy
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Initializes host, device, or shared memory.
/// 
/// @details
///     - The application must ensure the memory pointed to by dstptr is
///       accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by dstptr as
///       it is free to be modified by either the Host or device up until
///       execution.
///     - The value to initialize memory to is described by the pattern and the
///       pattern size.
///     - The pattern size must be a power-of-two and less than or equal to the
///       `maxMemoryFillPatternSize` member of
///       ::ze_command_queue_group_properties_t.
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The application must ensure the command list and events were created,
///       and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clEnqueueFillBuffer**
///     - **clEnqueueSVMMemFill**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == pattern`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryFill(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    void* ptr,                                                              ///< [in] pointer to memory to initialize
    const void* pattern,                                                    ///< [in] pointer to value to initialize memory to
    size_t pattern_size,                                                    ///< [in] size in bytes of the value to initialize memory to
    size_t size,                                                            ///< [in] size in bytes to initialize
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Copy region descriptor
typedef struct _ze_copy_region_t
{
    uint32_t originX;                                                       ///< [in] The origin x offset for region in bytes
    uint32_t originY;                                                       ///< [in] The origin y offset for region in rows
    uint32_t originZ;                                                       ///< [in] The origin z offset for region in slices
    uint32_t width;                                                         ///< [in] The region width relative to origin in bytes
    uint32_t height;                                                        ///< [in] The region height relative to origin in rows
    uint32_t depth;                                                         ///< [in] The region depth relative to origin in slices. Set this to 0 for
                                                                            ///< 2D copy.

} ze_copy_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies a region from a 2D or 3D array of host, device, or shared
///        memory.
/// 
/// @details
///     - The application must ensure the memory pointed to by dstptr and srcptr
///       is accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by dstptr and
///       srcptr as they are free to be modified by either the Host or device up
///       until execution.
///     - The region width, height, and depth for both src and dst must be same.
///       The origins can be different.
///     - The src and dst regions cannot be overlapping.
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The application must ensure the command list and events were created,
///       and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstptr`
///         + `nullptr == dstRegion`
///         + `nullptr == srcptr`
///         + `nullptr == srcRegion`
///     - ::ZE_RESULT_ERROR_OVERLAPPING_REGIONS
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryCopyRegion(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    void* dstptr,                                                           ///< [in] pointer to destination memory to copy to
    const ze_copy_region_t* dstRegion,                                      ///< [in] pointer to destination region to copy to
    uint32_t dstPitch,                                                      ///< [in] destination pitch in bytes
    uint32_t dstSlicePitch,                                                 ///< [in] destination slice pitch in bytes. This is required for 3D region
                                                                            ///< copies where the `depth` member of ::ze_copy_region_t is not 0,
                                                                            ///< otherwise it's ignored.
    const void* srcptr,                                                     ///< [in] pointer to source memory to copy from
    const ze_copy_region_t* srcRegion,                                      ///< [in] pointer to source region to copy from
    uint32_t srcPitch,                                                      ///< [in] source pitch in bytes
    uint32_t srcSlicePitch,                                                 ///< [in] source slice pitch in bytes. This is required for 3D region
                                                                            ///< copies where the `depth` member of ::ze_copy_region_t is not 0,
                                                                            ///< otherwise it's ignored.
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies host, device, or shared memory from another context.
/// 
/// @details
///     - The current active and source context must be from the same driver.
///     - The application must ensure the memory pointed to by dstptr and srcptr
///       is accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by dstptr and
///       srcptr as they are free to be modified by either the Host or device up
///       until execution.
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The application must ensure the command list and events were created,
///       and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hContextSrc`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstptr`
///         + `nullptr == srcptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryCopyFromContext(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    void* dstptr,                                                           ///< [in] pointer to destination memory to copy to
    ze_context_handle_t hContextSrc,                                        ///< [in] handle of source context object
    const void* srcptr,                                                     ///< [in] pointer to source memory to copy from
    size_t size,                                                            ///< [in] size in bytes to copy
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies an image.
/// 
/// @details
///     - The application must ensure the image and events are accessible by the
///       device on which the command list was created.
///     - The application must ensure the image format descriptors for both
///       source and destination images are the same.
///     - The application must ensure the command list, images and events were
///       created on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyImage**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hDstImage`
///         + `nullptr == hSrcImage`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopy(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    ze_image_handle_t hDstImage,                                            ///< [in] handle of destination image to copy to
    ze_image_handle_t hSrcImage,                                            ///< [in] handle of source image to copy from
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Region descriptor
typedef struct _ze_image_region_t
{
    uint32_t originX;                                                       ///< [in] The origin x offset for region in pixels
    uint32_t originY;                                                       ///< [in] The origin y offset for region in pixels
    uint32_t originZ;                                                       ///< [in] The origin z offset for region in pixels
    uint32_t width;                                                         ///< [in] The region width relative to origin in pixels
    uint32_t height;                                                        ///< [in] The region height relative to origin in pixels
    uint32_t depth;                                                         ///< [in] The region depth relative to origin. For 1D or 2D images, set
                                                                            ///< this to 1.

} ze_image_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies a region of an image to another image.
/// 
/// @details
///     - The application must ensure the image and events are accessible by the
///       device on which the command list was created.
///     - The region width and height for both src and dst must be same. The
///       origins can be different.
///     - The src and dst regions cannot be overlapping.
///     - The application must ensure the image format descriptors for both
///       source and destination images are the same.
///     - The application must ensure the command list, images and events were
///       created, and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hDstImage`
///         + `nullptr == hSrcImage`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_OVERLAPPING_REGIONS
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyRegion(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    ze_image_handle_t hDstImage,                                            ///< [in] handle of destination image to copy to
    ze_image_handle_t hSrcImage,                                            ///< [in] handle of source image to copy from
    const ze_image_region_t* pDstRegion,                                    ///< [in][optional] destination region descriptor
    const ze_image_region_t* pSrcRegion,                                    ///< [in][optional] source region descriptor
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies from an image to device or shared memory.
/// 
/// @details
///     - The application must ensure the memory pointed to by dstptr is
///       accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by dstptr as
///       it is free to be modified by either the Host or device up until
///       execution.
///     - The application must ensure the image and events are accessible by the
///       device on which the command list was created.
///     - The application must ensure the image format descriptor for the source
///       image is a single-planar format.
///     - The application must ensure the command list, image and events were
///       created, and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clEnqueueReadImage
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hSrcImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyToMemory(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    void* dstptr,                                                           ///< [in] pointer to destination memory to copy to
    ze_image_handle_t hSrcImage,                                            ///< [in] handle of source image to copy from
    const ze_image_region_t* pSrcRegion,                                    ///< [in][optional] source region descriptor
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies to an image from device or shared memory.
/// 
/// @details
///     - The application must ensure the memory pointed to by srcptr is
///       accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by srcptr as
///       it is free to be modified by either the Host or device up until
///       execution.
///     - The application must ensure the image and events are accessible by the
///       device on which the command list was created.
///     - The application must ensure the image format descriptor for the
///       destination image is a single-planar format.
///     - The application must ensure the command list, image and events were
///       created, and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clEnqueueWriteImage
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hDstImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == srcptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyFromMemory(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    ze_image_handle_t hDstImage,                                            ///< [in] handle of destination image to copy to
    const void* srcptr,                                                     ///< [in] pointer to source memory to copy from
    const ze_image_region_t* pDstRegion,                                    ///< [in][optional] destination region descriptor
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Asynchronously prefetches shared memory to the device associated with
///        the specified command list
/// 
/// @details
///     - This is a hint to improve performance only and is not required for
///       correctness.
///     - Only prefetching to the device associated with the specified command
///       list is supported.
///       Prefetching to the host or to a peer device is not supported.
///     - Prefetching may not be supported for all allocation types for all devices.
///       If memory prefetching is not supported for the specified memory range
///       the prefetch hint may be ignored.
///     - Prefetching may only be supported at a device-specific granularity,
///       such as at a page boundary.
///       In this case, the memory range may be expanded such that the start and
///       end of the range satisfy granularity requirements.
///     - The application must ensure the memory pointed to by ptr is accessible
///       by the device on which the command list was created.
///     - The application must ensure the command list was created, and the
///       memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clEnqueueSVMMigrateMem
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryPrefetch(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    const void* ptr,                                                        ///< [in] pointer to start of the memory range to prefetch
    size_t size                                                             ///< [in] size in bytes of the memory range to prefetch
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported memory advice hints
typedef enum _ze_memory_advice_t
{
    ZE_MEMORY_ADVICE_SET_READ_MOSTLY = 0,                                   ///< hint that memory will be read from frequently and written to rarely
    ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY = 1,                                 ///< removes the effect of ::ZE_MEMORY_ADVICE_SET_READ_MOSTLY
    ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION = 2,                            ///< hint that the preferred memory location is the specified device
    ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION = 3,                          ///< removes the effect of ::ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION
    ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY = 4,                             ///< hints that memory will mostly be accessed non-atomically
    ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY = 5,                           ///< removes the effect of ::ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY
    ZE_MEMORY_ADVICE_BIAS_CACHED = 6,                                       ///< hints that memory should be cached
    ZE_MEMORY_ADVICE_BIAS_UNCACHED = 7,                                     ///< hints that memory should be not be cached
    ZE_MEMORY_ADVICE_SET_SYSTEM_MEMORY_PREFERRED_LOCATION = 8,              ///< hint that the preferred memory location is host memory
    ZE_MEMORY_ADVICE_CLEAR_SYSTEM_MEMORY_PREFERRED_LOCATION = 9,            ///< removes the effect of
                                                                            ///< ::ZE_MEMORY_ADVICE_SET_SYSTEM_MEMORY_PREFERRED_LOCATION
    ZE_MEMORY_ADVICE_FORCE_UINT32 = 0x7fffffff

} ze_memory_advice_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Provides advice about the use of a shared memory range
/// 
/// @details
///     - Memory advice is a performance hint only and is not required for
///       functional correctness.
///     - Memory advice can be used to override driver heuristics to explicitly
///       control shared memory behavior.
///     - Not all memory advice hints may be supported for all allocation types
///       for all devices.
///       If a memory advice hint is not supported by the device it will be ignored.
///     - Memory advice may only be supported at a device-specific granularity,
///       such as at a page boundary.
///       In this case, the memory range may be expanded such that the start and
///       end of the range satisfy granularity requirements.
///     - The application must ensure the memory pointed to by ptr is accessible
///       by the device on which the command list was created.
///     - The application must ensure the command list was created, and memory
///       was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle, and the memory was
///       allocated.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_MEMORY_ADVICE_CLEAR_SYSTEM_MEMORY_PREFERRED_LOCATION < advice`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemAdvise(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    ze_device_handle_t hDevice,                                             ///< [in] device associated with the memory advice
    const void* ptr,                                                        ///< [in] Pointer to the start of the memory range
    size_t size,                                                            ///< [in] Size in bytes of the memory range
    ze_memory_advice_t advice                                               ///< [in] Memory advice for the memory range
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Event
#if !defined(__GNUC__)
#pragma region event
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported event pool creation flags
typedef uint32_t ze_event_pool_flags_t;
typedef enum _ze_event_pool_flag_t
{
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE = ZE_BIT(0),                            ///< signals and waits are also visible to host
    ZE_EVENT_POOL_FLAG_IPC = ZE_BIT(1),                                     ///< signals and waits may be shared across processes
    ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP = ZE_BIT(2),                        ///< Indicates all events in pool will contain kernel timestamps
    ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP = ZE_BIT(3),                 ///< Indicates all events in pool will contain kernel timestamps
                                                                            ///< synchronized to host time domain; cannot be combined with
                                                                            ///< ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP
    ZE_EVENT_POOL_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_event_pool_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event pool descriptor
typedef struct _ze_event_pool_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_event_pool_flags_t flags;                                            ///< [in] creation flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_event_pool_flag_t;
                                                                            ///< default behavior is signals and waits are visible to the entire device
                                                                            ///< and peer devices.
    uint32_t count;                                                         ///< [in] number of events within the pool; must be greater than 0

} ze_event_pool_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a pool of events on the context.
/// 
/// @details
///     - The application must only use events within the pool for the
///       device(s), or their sub-devices, which were provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phEventPool`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0xf < desc->flags`
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `0 == desc->count`
///         + `(nullptr == phDevices) && (0 < numDevices)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolCreate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const ze_event_pool_desc_t* desc,                                       ///< [in] pointer to event pool descriptor
    uint32_t numDevices,                                                    ///< [in][optional] number of device handles; must be 0 if `nullptr ==
                                                                            ///< phDevices`
    ze_device_handle_t* phDevices,                                          ///< [in][optional][range(0, numDevices)] array of device handles which
                                                                            ///< have visibility to the event pool.
                                                                            ///< if nullptr, then event pool is visible to all devices supported by the
                                                                            ///< driver instance.
    ze_event_pool_handle_t* phEventPool                                     ///< [out] pointer handle of event pool object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Deletes an event pool object.
/// 
/// @details
///     - The application must destroy all event handles created from the pool
///       before destroying the pool itself.
///     - The application must ensure the device is not currently referencing
///       the any event within the pool before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this event pool.
///     - The application must **not** call this function from simultaneous
///       threads with the same event pool handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEventPool`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolDestroy(
    ze_event_pool_handle_t hEventPool                                       ///< [in][release] handle of event pool object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported event scope flags
typedef uint32_t ze_event_scope_flags_t;
typedef enum _ze_event_scope_flag_t
{
    ZE_EVENT_SCOPE_FLAG_SUBDEVICE = ZE_BIT(0),                              ///< cache hierarchies are flushed or invalidated sufficient for local
                                                                            ///< sub-device access
    ZE_EVENT_SCOPE_FLAG_DEVICE = ZE_BIT(1),                                 ///< cache hierarchies are flushed or invalidated sufficient for global
                                                                            ///< device access and peer device access
    ZE_EVENT_SCOPE_FLAG_HOST = ZE_BIT(2),                                   ///< cache hierarchies are flushed or invalidated sufficient for device and
                                                                            ///< host access
    ZE_EVENT_SCOPE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_event_scope_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event descriptor
typedef struct _ze_event_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t index;                                                         ///< [in] index of the event within the pool; must be less than the count
                                                                            ///< specified during pool creation
    ze_event_scope_flags_t signal;                                          ///< [in] defines the scope of relevant cache hierarchies to flush on a
                                                                            ///< signal action before the event is triggered.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_event_scope_flag_t;
                                                                            ///< default behavior is synchronization within the command list only, no
                                                                            ///< additional cache hierarchies are flushed.
    ze_event_scope_flags_t wait;                                            ///< [in] defines the scope of relevant cache hierarchies to invalidate on
                                                                            ///< a wait action after the event is complete.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_event_scope_flag_t;
                                                                            ///< default behavior is synchronization within the command list only, no
                                                                            ///< additional cache hierarchies are invalidated.

} ze_event_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates an event from the pool.
/// 
/// @details
///     - An event is used to communicate fine-grain host-to-device,
///       device-to-host or device-to-device dependencies have completed.
///     - The application must ensure the location in the pool is not being used
///       by another event.
///     - The application must **not** call this function from simultaneous
///       threads with the same event pool handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clCreateUserEvent**
///     - vkCreateEvent
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEventPool`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phEvent`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7 < desc->signal`
///         + `0x7 < desc->wait`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventCreate(
    ze_event_pool_handle_t hEventPool,                                      ///< [in] handle of the event pool
    const ze_event_desc_t* desc,                                            ///< [in] pointer to event descriptor
    ze_event_handle_t* phEvent                                              ///< [out] pointer to handle of event object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Deletes an event object.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the event before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this event.
///     - The application must **not** call this function from simultaneous
///       threads with the same event handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clReleaseEvent**
///     - vkDestroyEvent
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventDestroy(
    ze_event_handle_t hEvent                                                ///< [in][release] handle of event object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Gets an IPC event pool handle for the specified event handle that can
///        be shared with another process.
/// 
/// @details
///     - Event pool must have been created with ::ZE_EVENT_POOL_FLAG_IPC.
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEventPool`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phIpc`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolGetIpcHandle(
    ze_event_pool_handle_t hEventPool,                                      ///< [in] handle of event pool object
    ze_ipc_event_pool_handle_t* phIpc                                       ///< [out] Returned IPC event handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns an IPC event pool handle to the driver
/// 
/// @details
///     - This call must be used for IPC handles previously obtained with
///       ::zeEventPoolGetIpcHandle.
///     - Upon call, driver may release any underlying resources associated with
///       the IPC handle.
///       For instance, it may close the file descriptor contained in the IPC
///       handle, if such type of handle is being used by the driver.
///     - This call does not destroy the original event pool for which the IPC
///       handle was created.
///     - This function may **not** be called from simultaneous threads with the
///       same IPC handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolPutIpcHandle(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object associated with the IPC event pool
                                                                            ///< handle
    ze_ipc_event_pool_handle_t hIpc                                         ///< [in] IPC event pool handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Opens an IPC event pool handle to retrieve an event pool handle from
///        another process.
/// 
/// @details
///     - Multiple calls to this function with the same IPC handle will return
///       unique event pool handles.
///     - The event handle in this process should not be freed with
///       ::zeEventPoolDestroy, but rather with ::zeEventPoolCloseIpcHandle.
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phEventPool`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolOpenIpcHandle(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object to associate with the IPC event pool
                                                                            ///< handle
    ze_ipc_event_pool_handle_t hIpc,                                        ///< [in] IPC event pool handle
    ze_event_pool_handle_t* phEventPool                                     ///< [out] pointer handle of event pool object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Closes an IPC event handle in the current process.
/// 
/// @details
///     - Closes an IPC event handle by destroying events that were opened in
///       this process using ::zeEventPoolOpenIpcHandle.
///     - The application must **not** call this function from simultaneous
///       threads with the same event pool handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEventPool`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolCloseIpcHandle(
    ze_event_pool_handle_t hEventPool                                       ///< [in][release] handle of event pool object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends a signal of the event from the device into a command list.
/// 
/// @details
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The duration of an event created from an event pool that was created
///       using ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP or
///       ::ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP flags is undefined.
///       However, for consistency and orthogonality the event will report
///       correctly as signaled when used by other event API functionality.
///     - The application must ensure the command list and events were created
///       on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clSetUserEventStatus**
///     - vkCmdSetEvent
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendSignalEvent(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    ze_event_handle_t hEvent                                                ///< [in] handle of the event
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends wait on event(s) on the device into a command list.
/// 
/// @details
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The application must ensure the command list and events were created
///       on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phEvents`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendWaitOnEvents(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    uint32_t numEvents,                                                     ///< [in] number of events to wait on before continuing
    ze_event_handle_t* phEvents                                             ///< [in][range(0, numEvents)] handles of the events to wait on before
                                                                            ///< continuing
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Signals a event from host.
/// 
/// @details
///     - The duration of an event created from an event pool that was created
///       using ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP or
///       ::ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP flags is undefined.
///       However, for consistency and orthogonality the event will report
///       correctly as signaled when used by other event API functionality.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clSetUserEventStatus
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventHostSignal(
    ze_event_handle_t hEvent                                                ///< [in] handle of the event
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief The current host thread waits on an event to be signaled.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clWaitForEvents
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_NOT_READY
///         + timeout expired
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventHostSynchronize(
    ze_event_handle_t hEvent,                                               ///< [in] handle of the event
    uint64_t timeout                                                        ///< [in] if non-zero, then indicates the maximum time (in nanoseconds) to
                                                                            ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                                            ///< if zero, then operates exactly like ::zeEventQueryStatus;
                                                                            ///< if `UINT64_MAX`, then function will not return until complete or
                                                                            ///< device is lost.
                                                                            ///< Due to external dependencies, timeout may be rounded to the closest
                                                                            ///< value allowed by the accuracy of those dependencies.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries an event object's status on the host.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clGetEventInfo**
///     - vkGetEventStatus
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_NOT_READY
///         + not signaled
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventQueryStatus(
    ze_event_handle_t hEvent                                                ///< [in] handle of the event
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends a reset of an event back to not signaled state into a command
///        list.
/// 
/// @details
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The application must ensure the command list and events were created
///       on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - vkResetEvent
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendEventReset(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    ze_event_handle_t hEvent                                                ///< [in] handle of the event
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief The current host thread resets an event back to not signaled state.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - vkResetEvent
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventHostReset(
    ze_event_handle_t hEvent                                                ///< [in] handle of the event
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel timestamp clock data
/// 
/// @details
///     - The timestamp frequency can be queried from the `timerResolution`
///       member of ::ze_device_properties_t.
///     - The number of valid bits in the timestamp value can be queried from
///       the `kernelTimestampValidBits` member of ::ze_device_properties_t.
typedef struct _ze_kernel_timestamp_data_t
{
    uint64_t kernelStart;                                                   ///< [out] device clock at start of kernel execution
    uint64_t kernelEnd;                                                     ///< [out] device clock at end of kernel execution

} ze_kernel_timestamp_data_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel timestamp result
typedef struct _ze_kernel_timestamp_result_t
{
    ze_kernel_timestamp_data_t global;                                      ///< [out] wall-clock data
    ze_kernel_timestamp_data_t context;                                     ///< [out] context-active data; only includes clocks while device context
                                                                            ///< was actively executing.

} ze_kernel_timestamp_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries an event's timestamp value on the host.
/// 
/// @details
///     - The application must ensure the event was created from an event pool
///       that was created using ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP or
///       ::ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP flag.
///     - The destination memory will be unmodified if the event has not been
///       signaled.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_NOT_READY
///         + not signaled
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventQueryKernelTimestamp(
    ze_event_handle_t hEvent,                                               ///< [in] handle of the event
    ze_kernel_timestamp_result_t* dstptr                                    ///< [in,out] pointer to memory for where timestamp result will be written.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends a query of an events' timestamp value(s) into a command list.
/// 
/// @details
///     - The application must ensure the events are accessible by the device on
///       which the command list was created.
///     - The application must ensure the events were created from an event pool
///       that was created using ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP flag.
///     - The application must ensure the memory pointed to by both dstptr and
///       pOffsets is accessible by the device on which the command list was
///       created.
///     - The value(s) written to the destination buffer are undefined if any
///       timestamp event has not been signaled.
///     - If pOffsets is nullptr, then multiple results will be appended
///       sequentially into memory in the same order as phEvents.
///     - The application must ensure the command list and events were created,
///       and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phEvents`
///         + `nullptr == dstptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendQueryKernelTimestamps(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    uint32_t numEvents,                                                     ///< [in] the number of timestamp events to query
    ze_event_handle_t* phEvents,                                            ///< [in][range(0, numEvents)] handles of timestamp events to query
    void* dstptr,                                                           ///< [in,out] pointer to memory where ::ze_kernel_timestamp_result_t will
                                                                            ///< be written; must be size-aligned.
    const size_t* pOffsets,                                                 ///< [in][optional][range(0, numEvents)] offset, in bytes, to write
                                                                            ///< results; address must be 4byte-aligned and offsets must be
                                                                            ///< size-aligned.
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before executing query;
                                                                            ///< must be 0 if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before executing query
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Fence
#if !defined(__GNUC__)
#pragma region fence
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported fence creation flags
typedef uint32_t ze_fence_flags_t;
typedef enum _ze_fence_flag_t
{
    ZE_FENCE_FLAG_SIGNALED = ZE_BIT(0),                                     ///< fence is created in the signaled state, otherwise not signaled.
    ZE_FENCE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_fence_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fence descriptor
typedef struct _ze_fence_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_fence_flags_t flags;                                                 ///< [in] creation flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_fence_flag_t.

} ze_fence_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a fence for the command queue.
/// 
/// @details
///     - A fence is a heavyweight synchronization primitive used to communicate
///       to the host that command list execution has completed.
///     - The application must only use the fence for the command queue which
///       was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @remarks
///   _Analogues_
///     - **vkCreateFence**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandQueue`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phFence`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < desc->flags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceCreate(
    ze_command_queue_handle_t hCommandQueue,                                ///< [in] handle of command queue
    const ze_fence_desc_t* desc,                                            ///< [in] pointer to fence descriptor
    ze_fence_handle_t* phFence                                              ///< [out] pointer to handle of fence object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Deletes a fence object.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the fence before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this fence.
///     - The application must **not** call this function from simultaneous
///       threads with the same fence handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @remarks
///   _Analogues_
///     - **vkDestroyFence**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFence`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceDestroy(
    ze_fence_handle_t hFence                                                ///< [in][release] handle of fence object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief The current host thread waits on a fence to be signaled.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **vkWaitForFences**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFence`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_NOT_READY
///         + timeout expired
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceHostSynchronize(
    ze_fence_handle_t hFence,                                               ///< [in] handle of the fence
    uint64_t timeout                                                        ///< [in] if non-zero, then indicates the maximum time (in nanoseconds) to
                                                                            ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                                            ///< if zero, then operates exactly like ::zeFenceQueryStatus;
                                                                            ///< if `UINT64_MAX`, then function will not return until complete or
                                                                            ///< device is lost.
                                                                            ///< Due to external dependencies, timeout may be rounded to the closest
                                                                            ///< value allowed by the accuracy of those dependencies.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries a fence object's status.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **vkGetFenceStatus**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFence`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_NOT_READY
///         + not signaled
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceQueryStatus(
    ze_fence_handle_t hFence                                                ///< [in] handle of the fence
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Reset a fence back to the not signaled state.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **vkResetFences**
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFence`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceReset(
    ze_fence_handle_t hFence                                                ///< [in] handle of the fence
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Images
#if !defined(__GNUC__)
#pragma region image
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported image creation flags
typedef uint32_t ze_image_flags_t;
typedef enum _ze_image_flag_t
{
    ZE_IMAGE_FLAG_KERNEL_WRITE = ZE_BIT(0),                                 ///< kernels will write contents
    ZE_IMAGE_FLAG_BIAS_UNCACHED = ZE_BIT(1),                                ///< device should not cache contents
    ZE_IMAGE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_image_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported image types
typedef enum _ze_image_type_t
{
    ZE_IMAGE_TYPE_1D = 0,                                                   ///< 1D
    ZE_IMAGE_TYPE_1DARRAY = 1,                                              ///< 1D array
    ZE_IMAGE_TYPE_2D = 2,                                                   ///< 2D
    ZE_IMAGE_TYPE_2DARRAY = 3,                                              ///< 2D array
    ZE_IMAGE_TYPE_3D = 4,                                                   ///< 3D
    ZE_IMAGE_TYPE_BUFFER = 5,                                               ///< Buffer
    ZE_IMAGE_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_image_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported image format layouts
typedef enum _ze_image_format_layout_t
{
    ZE_IMAGE_FORMAT_LAYOUT_8 = 0,                                           ///< 8-bit single component layout
    ZE_IMAGE_FORMAT_LAYOUT_16 = 1,                                          ///< 16-bit single component layout
    ZE_IMAGE_FORMAT_LAYOUT_32 = 2,                                          ///< 32-bit single component layout
    ZE_IMAGE_FORMAT_LAYOUT_8_8 = 3,                                         ///< 2-component 8-bit layout
    ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 = 4,                                     ///< 4-component 8-bit layout
    ZE_IMAGE_FORMAT_LAYOUT_16_16 = 5,                                       ///< 2-component 16-bit layout
    ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 = 6,                                 ///< 4-component 16-bit layout
    ZE_IMAGE_FORMAT_LAYOUT_32_32 = 7,                                       ///< 2-component 32-bit layout
    ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 = 8,                                 ///< 4-component 32-bit layout
    ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2 = 9,                                  ///< 4-component 10_10_10_2 layout
    ZE_IMAGE_FORMAT_LAYOUT_11_11_10 = 10,                                   ///< 3-component 11_11_10 layout
    ZE_IMAGE_FORMAT_LAYOUT_5_6_5 = 11,                                      ///< 3-component 5_6_5 layout
    ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1 = 12,                                    ///< 4-component 5_5_5_1 layout
    ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4 = 13,                                    ///< 4-component 4_4_4_4 layout
    ZE_IMAGE_FORMAT_LAYOUT_Y8 = 14,                                         ///< Media Format: Y8. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_NV12 = 15,                                       ///< Media Format: NV12. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_YUYV = 16,                                       ///< Media Format: YUYV. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_VYUY = 17,                                       ///< Media Format: VYUY. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_YVYU = 18,                                       ///< Media Format: YVYU. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_UYVY = 19,                                       ///< Media Format: UYVY. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_AYUV = 20,                                       ///< Media Format: AYUV. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_P010 = 21,                                       ///< Media Format: P010. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_Y410 = 22,                                       ///< Media Format: Y410. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_P012 = 23,                                       ///< Media Format: P012. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_Y16 = 24,                                        ///< Media Format: Y16. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_P016 = 25,                                       ///< Media Format: P016. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_Y216 = 26,                                       ///< Media Format: Y216. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_P216 = 27,                                       ///< Media Format: P216. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_P8 = 28,                                         ///< Media Format: P8. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_YUY2 = 29,                                       ///< Media Format: YUY2. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_A8P8 = 30,                                       ///< Media Format: A8P8. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_IA44 = 31,                                       ///< Media Format: IA44. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_AI44 = 32,                                       ///< Media Format: AI44. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_Y416 = 33,                                       ///< Media Format: Y416. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_Y210 = 34,                                       ///< Media Format: Y210. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_I420 = 35,                                       ///< Media Format: I420. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_YV12 = 36,                                       ///< Media Format: YV12. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_400P = 37,                                       ///< Media Format: 400P. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_422H = 38,                                       ///< Media Format: 422H. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_422V = 39,                                       ///< Media Format: 422V. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_444P = 40,                                       ///< Media Format: 444P. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_RGBP = 41,                                       ///< Media Format: RGBP. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_BRGP = 42,                                       ///< Media Format: BRGP. Format type and swizzle is ignored for this.
    ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32 = 0x7fffffff

} ze_image_format_layout_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported image format types
typedef enum _ze_image_format_type_t
{
    ZE_IMAGE_FORMAT_TYPE_UINT = 0,                                          ///< Unsigned integer
    ZE_IMAGE_FORMAT_TYPE_SINT = 1,                                          ///< Signed integer
    ZE_IMAGE_FORMAT_TYPE_UNORM = 2,                                         ///< Unsigned normalized integer
    ZE_IMAGE_FORMAT_TYPE_SNORM = 3,                                         ///< Signed normalized integer
    ZE_IMAGE_FORMAT_TYPE_FLOAT = 4,                                         ///< Float
    ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_image_format_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported image format component swizzle into channel
typedef enum _ze_image_format_swizzle_t
{
    ZE_IMAGE_FORMAT_SWIZZLE_R = 0,                                          ///< Red component
    ZE_IMAGE_FORMAT_SWIZZLE_G = 1,                                          ///< Green component
    ZE_IMAGE_FORMAT_SWIZZLE_B = 2,                                          ///< Blue component
    ZE_IMAGE_FORMAT_SWIZZLE_A = 3,                                          ///< Alpha component
    ZE_IMAGE_FORMAT_SWIZZLE_0 = 4,                                          ///< Zero
    ZE_IMAGE_FORMAT_SWIZZLE_1 = 5,                                          ///< One
    ZE_IMAGE_FORMAT_SWIZZLE_X = 6,                                          ///< Don't care
    ZE_IMAGE_FORMAT_SWIZZLE_FORCE_UINT32 = 0x7fffffff

} ze_image_format_swizzle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image format 
typedef struct _ze_image_format_t
{
    ze_image_format_layout_t layout;                                        ///< [in] image format component layout (e.g. N-component layouts and media
                                                                            ///< formats)
    ze_image_format_type_t type;                                            ///< [in] image format type
    ze_image_format_swizzle_t x;                                            ///< [in] image component swizzle into channel x
    ze_image_format_swizzle_t y;                                            ///< [in] image component swizzle into channel y
    ze_image_format_swizzle_t z;                                            ///< [in] image component swizzle into channel z
    ze_image_format_swizzle_t w;                                            ///< [in] image component swizzle into channel w

} ze_image_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image descriptor
typedef struct _ze_image_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_image_flags_t flags;                                                 ///< [in] creation flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_image_flag_t;
                                                                            ///< default is read-only, cached access.
    ze_image_type_t type;                                                   ///< [in] image type. Media format layouts are unsupported for
                                                                            ///< ::ZE_IMAGE_TYPE_BUFFER
    ze_image_format_t format;                                               ///< [in] image format
    uint64_t width;                                                         ///< [in] width dimension.
                                                                            ///< ::ZE_IMAGE_TYPE_BUFFER: size in bytes; see the `maxImageBufferSize`
                                                                            ///< member of ::ze_device_image_properties_t for limits.
                                                                            ///< ::ZE_IMAGE_TYPE_1D, ::ZE_IMAGE_TYPE_1DARRAY: width in pixels; see the
                                                                            ///< `maxImageDims1D` member of ::ze_device_image_properties_t for limits.
                                                                            ///< ::ZE_IMAGE_TYPE_2D, ::ZE_IMAGE_TYPE_2DARRAY: width in pixels; see the
                                                                            ///< `maxImageDims2D` member of ::ze_device_image_properties_t for limits.
                                                                            ///< ::ZE_IMAGE_TYPE_3D: width in pixels; see the `maxImageDims3D` member
                                                                            ///< of ::ze_device_image_properties_t for limits.
    uint32_t height;                                                        ///< [in] height dimension.
                                                                            ///< ::ZE_IMAGE_TYPE_2D, ::ZE_IMAGE_TYPE_2DARRAY: height in pixels; see the
                                                                            ///< `maxImageDims2D` member of ::ze_device_image_properties_t for limits.
                                                                            ///< ::ZE_IMAGE_TYPE_3D: height in pixels; see the `maxImageDims3D` member
                                                                            ///< of ::ze_device_image_properties_t for limits.
                                                                            ///< other: ignored.
    uint32_t depth;                                                         ///< [in] depth dimension.
                                                                            ///< ::ZE_IMAGE_TYPE_3D: depth in pixels; see the `maxImageDims3D` member
                                                                            ///< of ::ze_device_image_properties_t for limits.
                                                                            ///< other: ignored.
    uint32_t arraylevels;                                                   ///< [in] array levels.
                                                                            ///< ::ZE_IMAGE_TYPE_1DARRAY, ::ZE_IMAGE_TYPE_2DARRAY: see the
                                                                            ///< `maxImageArraySlices` member of ::ze_device_image_properties_t for limits.
                                                                            ///< other: ignored.
    uint32_t miplevels;                                                     ///< [in] mipmap levels (must be 0)

} ze_image_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported sampler filtering flags
typedef uint32_t ze_image_sampler_filter_flags_t;
typedef enum _ze_image_sampler_filter_flag_t
{
    ZE_IMAGE_SAMPLER_FILTER_FLAG_POINT = ZE_BIT(0),                         ///< device supports point filtering
    ZE_IMAGE_SAMPLER_FILTER_FLAG_LINEAR = ZE_BIT(1),                        ///< device supports linear filtering
    ZE_IMAGE_SAMPLER_FILTER_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_image_sampler_filter_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image properties
typedef struct _ze_image_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_image_sampler_filter_flags_t samplerFilterFlags;                     ///< [out] supported sampler filtering.
                                                                            ///< returns 0 (unsupported) or a combination of ::ze_image_sampler_filter_flag_t.

} ze_image_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves supported properties of an image.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == pImageProperties`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < desc->flags`
///         + `::ZE_IMAGE_TYPE_BUFFER < desc->type`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageGetProperties(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    const ze_image_desc_t* desc,                                            ///< [in] pointer to image descriptor
    ze_image_properties_t* pImageProperties                                 ///< [out] pointer to image properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates an image on the context.
/// 
/// @details
///     - The application must only use the image for the device, or its
///       sub-devices, which was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @remarks
///   _Analogues_
///     - clCreateImage
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phImage`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < desc->flags`
///         + `::ZE_IMAGE_TYPE_BUFFER < desc->type`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageCreate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    const ze_image_desc_t* desc,                                            ///< [in] pointer to image descriptor
    ze_image_handle_t* phImage                                              ///< [out] pointer to handle of image object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Deletes an image object.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the image before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this image.
///     - The application must **not** call this function from simultaneous
///       threads with the same image handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hImage`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageDestroy(
    ze_image_handle_t hImage                                                ///< [in][release] handle of image object to destroy
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Memory
#if !defined(__GNUC__)
#pragma region memory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported memory allocation flags
typedef uint32_t ze_device_mem_alloc_flags_t;
typedef enum _ze_device_mem_alloc_flag_t
{
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED = ZE_BIT(0),                       ///< device should cache allocation
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED = ZE_BIT(1),                     ///< device should not cache allocation (UC)
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = ZE_BIT(2),            ///< optimize shared allocation for first access on the device
    ZE_DEVICE_MEM_ALLOC_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_mem_alloc_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device memory allocation descriptor
typedef struct _ze_device_mem_alloc_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_mem_alloc_flags_t flags;                                      ///< [in] flags specifying additional allocation controls.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_device_mem_alloc_flag_t;
                                                                            ///< default behavior may use implicit driver-based heuristics.
    uint32_t ordinal;                                                       ///< [in] ordinal of the device's local memory to allocate from.
                                                                            ///< must be less than the count returned from ::zeDeviceGetMemoryProperties.

} ze_device_mem_alloc_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported host memory allocation flags
typedef uint32_t ze_host_mem_alloc_flags_t;
typedef enum _ze_host_mem_alloc_flag_t
{
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED = ZE_BIT(0),                         ///< host should cache allocation
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED = ZE_BIT(1),                       ///< host should not cache allocation (UC)
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED = ZE_BIT(2),                 ///< host memory should be allocated write-combined (WC)
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = ZE_BIT(3),              ///< optimize shared allocation for first access on the host
    ZE_HOST_MEM_ALLOC_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_host_mem_alloc_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Host memory allocation descriptor
typedef struct _ze_host_mem_alloc_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_host_mem_alloc_flags_t flags;                                        ///< [in] flags specifying additional allocation controls.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_host_mem_alloc_flag_t;
                                                                            ///< default behavior may use implicit driver-based heuristics.

} ze_host_mem_alloc_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Allocates shared memory on the context.
/// 
/// @details
///     - Shared allocations share ownership between the host and one or more
///       devices.
///     - Shared allocations may optionally be associated with a device by
///       passing a handle to the device.
///     - Devices supporting only single-device shared access capabilities may
///       access shared memory associated with the device.
///       For these devices, ownership of the allocation is shared between the
///       host and the associated device only.
///     - Passing nullptr as the device handle does not associate the shared
///       allocation with any device.
///       For allocations with no associated device, ownership of the allocation
///       is shared between the host and all devices supporting cross-device
///       shared access capabilities.
///     - The application must only use the memory allocation for the context
///       and device, or its sub-devices, which was provided during allocation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == device_desc`
///         + `nullptr == host_desc`
///         + `nullptr == pptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7 < device_desc->flags`
///         + `0xf < host_desc->flags`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
///         + Must be zero or a power-of-two
///         + `0 != (alignment & (alignment - 1))`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemAllocShared(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const ze_device_mem_alloc_desc_t* device_desc,                          ///< [in] pointer to device memory allocation descriptor
    const ze_host_mem_alloc_desc_t* host_desc,                              ///< [in] pointer to host memory allocation descriptor
    size_t size,                                                            ///< [in] size in bytes to allocate; must be less than or equal to the
                                                                            ///< `maxMemAllocSize` member of ::ze_device_properties_t
    size_t alignment,                                                       ///< [in] minimum alignment in bytes for the allocation; must be a power of
                                                                            ///< two
    ze_device_handle_t hDevice,                                             ///< [in][optional] device handle to associate with
    void** pptr                                                             ///< [out] pointer to shared allocation
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Allocates device memory on the context.
/// 
/// @details
///     - Device allocations are owned by a specific device.
///     - In general, a device allocation may only be accessed by the device
///       that owns it.
///     - The application must only use the memory allocation for the context
///       and device, or its sub-devices, which was provided during allocation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == device_desc`
///         + `nullptr == pptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7 < device_desc->flags`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
///         + Must be zero or a power-of-two
///         + `0 != (alignment & (alignment - 1))`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemAllocDevice(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const ze_device_mem_alloc_desc_t* device_desc,                          ///< [in] pointer to device memory allocation descriptor
    size_t size,                                                            ///< [in] size in bytes to allocate; must be less than or equal to the
                                                                            ///< `maxMemAllocSize` member of ::ze_device_properties_t
    size_t alignment,                                                       ///< [in] minimum alignment in bytes for the allocation; must be a power of
                                                                            ///< two
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    void** pptr                                                             ///< [out] pointer to device allocation
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Allocates host memory on the context.
/// 
/// @details
///     - Host allocations are owned by the host process.
///     - Host allocations are accessible by the host and all devices within the
///       driver's context.
///     - Host allocations are frequently used as staging areas to transfer data
///       to or from devices.
///     - The application must only use the memory allocation for the context
///       which was provided during allocation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == host_desc`
///         + `nullptr == pptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0xf < host_desc->flags`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
///         + Must be zero or a power-of-two
///         + `0 != (alignment & (alignment - 1))`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemAllocHost(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const ze_host_mem_alloc_desc_t* host_desc,                              ///< [in] pointer to host memory allocation descriptor
    size_t size,                                                            ///< [in] size in bytes to allocate; must be less than or equal to the
                                                                            ///< `maxMemAllocSize` member of ::ze_device_properties_t
    size_t alignment,                                                       ///< [in] minimum alignment in bytes for the allocation; must be a power of
                                                                            ///< two
    void** pptr                                                             ///< [out] pointer to host allocation
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Frees allocated host memory, device memory, or shared memory on the
///        context.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the memory before it is freed
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this memory
///     - The application must **not** call this function from simultaneous
///       threads with the same pointer.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemFree(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    void* ptr                                                               ///< [in][release] pointer to memory to free
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory allocation type
typedef enum _ze_memory_type_t
{
    ZE_MEMORY_TYPE_UNKNOWN = 0,                                             ///< the memory pointed to is of unknown type
    ZE_MEMORY_TYPE_HOST = 1,                                                ///< the memory pointed to is a host allocation
    ZE_MEMORY_TYPE_DEVICE = 2,                                              ///< the memory pointed to is a device allocation
    ZE_MEMORY_TYPE_SHARED = 3,                                              ///< the memory pointed to is a shared ownership allocation
    ZE_MEMORY_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_memory_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory allocation properties queried using ::zeMemGetAllocProperties
typedef struct _ze_memory_allocation_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_memory_type_t type;                                                  ///< [out] type of allocated memory
    uint64_t id;                                                            ///< [out] identifier for this allocation
    uint64_t pageSize;                                                      ///< [out] page size used for allocation

} ze_memory_allocation_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves attributes of a memory allocation
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The application may query attributes of a memory allocation unrelated
///       to the context.
///       When this occurs, the returned allocation type will be
///       ::ZE_MEMORY_TYPE_UNKNOWN, and the returned identifier and associated
///       device is unspecified.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == pMemAllocProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetAllocProperties(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] memory pointer to query
    ze_memory_allocation_properties_t* pMemAllocProperties,                 ///< [in,out] query result for memory allocation properties
    ze_device_handle_t* phDevice                                            ///< [out][optional] device associated with this allocation
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves the base address and/or size of an allocation
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetAddressRange(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] memory pointer to query
    void** pBase,                                                           ///< [in,out][optional] base address of the allocation
    size_t* pSize                                                           ///< [in,out][optional] size of the allocation
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates an IPC memory handle for the specified allocation
/// 
/// @details
///     - Takes a pointer to a device memory allocation and creates an IPC
///       memory handle for exporting it for use in another process.
///     - The pointer must be base pointer of a device or host memory
///       allocation; i.e. the value returned from ::zeMemAllocDevice or from
///       ::zeMemAllocHost, respectively.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == pIpcHandle`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetIpcHandle(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] pointer to the device memory allocation
    ze_ipc_mem_handle_t* pIpcHandle                                         ///< [out] Returned IPC memory handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates an IPC memory handle out of a file descriptor
/// 
/// @details
///     - Handle passed must be a valid file descriptor obtained with
///       ::ze_external_memory_export_fd_t via ::zeMemGetAllocProperties.
///     - Returned IPC handle may contain metadata in addition to the file
///       descriptor.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pIpcHandle`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetIpcHandleFromFileDescriptorExp(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    uint64_t handle,                                                        ///< [in] file descriptor
    ze_ipc_mem_handle_t* pIpcHandle                                         ///< [out] Returned IPC memory handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Gets the file descriptor contained in an IPC memory handle
/// 
/// @details
///     - IPC memory handle must be a valid handle obtained with
///       ::zeMemGetIpcHandle.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pHandle`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetFileDescriptorFromIpcHandleExp(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_ipc_mem_handle_t ipcHandle,                                          ///< [in] IPC memory handle
    uint64_t* pHandle                                                       ///< [out] Returned file descriptor
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns an IPC memory handle to the driver
/// 
/// @details
///     - This call may be used for IPC handles previously obtained with either
///       ::zeMemGetIpcHandle or with ::ze_external_memory_export_fd_t via ::zeMemGetAllocProperties.
///     - Upon call, driver may release any underlying resources associated with
///       the IPC handle.
///       For instance, it may close the file descriptor contained in the IPC
///       handle, if such type of handle is being used by the driver.
///     - This call does not free the original allocation for which the IPC
///       handle was created.
///     - This function may **not** be called from simultaneous threads with the
///       same IPC handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemPutIpcHandle(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_ipc_mem_handle_t handle                                              ///< [in] IPC memory handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported IPC memory flags
typedef uint32_t ze_ipc_memory_flags_t;
typedef enum _ze_ipc_memory_flag_t
{
    ZE_IPC_MEMORY_FLAG_BIAS_CACHED = ZE_BIT(0),                             ///< device should cache allocation
    ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED = ZE_BIT(1),                           ///< device should not cache allocation (UC)
    ZE_IPC_MEMORY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_ipc_memory_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Opens an IPC memory handle to retrieve a device pointer on the
///        context.
/// 
/// @details
///     - Takes an IPC memory handle from a remote process and associates it
///       with a device pointer usable in this process.
///     - The device pointer in this process should not be freed with
///       ::zeMemFree, but rather with ::zeMemCloseIpcHandle.
///     - Multiple calls to this function with the same IPC handle will return
///       unique pointers.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < flags`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pptr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemOpenIpcHandle(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device to associate with the IPC memory handle
    ze_ipc_mem_handle_t handle,                                             ///< [in] IPC memory handle
    ze_ipc_memory_flags_t flags,                                            ///< [in] flags controlling the operation.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_ipc_memory_flag_t.
    void** pptr                                                             ///< [out] pointer to device allocation in this process
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Closes an IPC memory handle
/// 
/// @details
///     - Closes an IPC memory handle by unmapping memory that was opened in
///       this process using ::zeMemOpenIpcHandle.
///     - The application must **not** call this function from simultaneous
///       threads with the same pointer.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemCloseIpcHandle(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr                                                         ///< [in][release] pointer to device allocation in this process
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Additional allocation descriptor for exporting external memory
/// 
/// @details
///     - This structure may be passed to ::zeMemAllocDevice and
///       ::zeMemAllocHost, via the `pNext` member of
///       ::ze_device_mem_alloc_desc_t or ::ze_host_mem_alloc_desc_t,
///       respectively, to indicate an exportable memory allocation.
///     - This structure may be passed to ::zeImageCreate, via the `pNext`
///       member of ::ze_image_desc_t, to indicate an exportable image.
typedef struct _ze_external_memory_export_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_external_memory_type_flags_t flags;                                  ///< [in] flags specifying memory export types for this allocation.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t

} ze_external_memory_export_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Additional allocation descriptor for importing external memory as a
///        file descriptor
/// 
/// @details
///     - This structure may be passed to ::zeMemAllocDevice or
///       ::zeMemAllocHost, via the `pNext` member of
///       ::ze_device_mem_alloc_desc_t or of ::ze_host_mem_alloc_desc_t,
///       respectively, to import memory from a file descriptor.
///     - This structure may be passed to ::zeImageCreate, via the `pNext`
///       member of ::ze_image_desc_t, to import memory from a file descriptor.
typedef struct _ze_external_memory_import_fd_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_external_memory_type_flags_t flags;                                  ///< [in] flags specifying the memory import type for the file descriptor.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
    int fd;                                                                 ///< [in] the file descriptor handle to import

} ze_external_memory_import_fd_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exports an allocation as a file descriptor
/// 
/// @details
///     - This structure may be passed to ::zeMemGetAllocProperties, via the
///       `pNext` member of ::ze_memory_allocation_properties_t, to export a
///       memory allocation as a file descriptor.
///     - This structure may be passed to ::zeImageGetAllocPropertiesExt, via
///       the `pNext` member of ::ze_image_allocation_ext_properties_t, to
///       export an image as a file descriptor.
///     - The requested memory export type must have been specified when the
///       allocation was made.
typedef struct _ze_external_memory_export_fd_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_external_memory_type_flags_t flags;                                  ///< [in] flags specifying the memory export type for the file descriptor.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
    int fd;                                                                 ///< [out] the exported file descriptor handle representing the allocation.

} ze_external_memory_export_fd_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Additional allocation descriptor for importing external memory as a
///        Win32 handle
/// 
/// @details
///     - When `handle` is `nullptr`, `name` must not be `nullptr`.
///     - When `name` is `nullptr`, `handle` must not be `nullptr`.
///     - When `flags` is ::ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT,
///       `name` must be `nullptr`.
///     - This structure may be passed to ::zeMemAllocDevice or
///       ::zeMemAllocHost, via the `pNext` member of
///       ::ze_device_mem_alloc_desc_t or of ::ze_host_mem_alloc_desc_t,
///       respectively, to import memory from a Win32 handle.
///     - This structure may be passed to ::zeImageCreate, via the `pNext`
///       member of ::ze_image_desc_t, to import memory from a Win32 handle.
typedef struct _ze_external_memory_import_win32_handle_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_external_memory_type_flags_t flags;                                  ///< [in] flags specifying the memory import type for the Win32 handle.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
    void* handle;                                                           ///< [in][optional] the Win32 handle to import
    const void* name;                                                       ///< [in][optional] name of a memory object to import

} ze_external_memory_import_win32_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exports an allocation as a Win32 handle
/// 
/// @details
///     - This structure may be passed to ::zeMemGetAllocProperties, via the
///       `pNext` member of ::ze_memory_allocation_properties_t, to export a
///       memory allocation as a Win32 handle.
///     - This structure may be passed to ::zeImageGetAllocPropertiesExt, via
///       the `pNext` member of ::ze_image_allocation_ext_properties_t, to
///       export an image as a Win32 handle.
///     - The requested memory export type must have been specified when the
///       allocation was made.
typedef struct _ze_external_memory_export_win32_handle_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_external_memory_type_flags_t flags;                                  ///< [in] flags specifying the memory export type for the Win32 handle.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
    void* handle;                                                           ///< [out] the exported Win32 handle representing the allocation.

} ze_external_memory_export_win32_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief atomic access attribute flags
typedef uint32_t ze_memory_atomic_attr_exp_flags_t;
typedef enum _ze_memory_atomic_attr_exp_flag_t
{
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_ATOMICS = ZE_BIT(0),                  ///< Atomics on the pointer are not allowed
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_HOST_ATOMICS = ZE_BIT(1),             ///< Host atomics on the pointer are not allowed
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_HOST_ATOMICS = ZE_BIT(2),                ///< Host atomics on the pointer are allowed. Requires
                                                                            ///< ::ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC returned by
                                                                            ///< ::zeDeviceGetMemoryAccessProperties.
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_DEVICE_ATOMICS = ZE_BIT(3),           ///< Device atomics on the pointer are not allowed
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_DEVICE_ATOMICS = ZE_BIT(4),              ///< Device atomics on the pointer are allowed. Requires
                                                                            ///< ::ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC returned by
                                                                            ///< ::zeDeviceGetMemoryAccessProperties.
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_SYSTEM_ATOMICS = ZE_BIT(5),           ///< Concurrent atomics on the pointer from both host and device are not
                                                                            ///< allowed
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_SYSTEM_ATOMICS = ZE_BIT(6),              ///< Concurrent atomics on the pointer from both host and device are
                                                                            ///< allowed. Requires ::ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC
                                                                            ///< returned by ::zeDeviceGetMemoryAccessProperties.
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_memory_atomic_attr_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets atomic access attributes for a shared allocation
/// 
/// @details
///     - If the shared-allocation is owned by multiple devices (i.e. nullptr
///       was passed to ::zeMemAllocShared when creating it), then hDevice may be
///       passed to set the attributes in that specific device. If nullptr is
///       passed in hDevice, then the atomic attributes are set in all devices
///       associated with the allocation.
///     - If the atomic access attribute select is not supported by the driver,
///       ::ZE_RESULT_INVALID_ARGUMENT is returned.
///     - The atomic access attribute may be only supported at a device-specific
///       granularity, such as at a page boundary. In this case, the memory range
///       may be expanded such that the start and end of the range satisfy granularity
///       requirements.
///     - When calling this function multiple times with different flags, only the
///       attributes from last call are honored.
///     - The application must not call this function for shared-allocations currently
///       being used by the device.
///     - The application must **not** call this function from simultaneous threads
///       with the same pointer.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7f < attr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemSetAtomicAccessAttributeExp(
    ze_context_handle_t hContext,                                           ///< [in] handle of context
    ze_device_handle_t hDevice,                                             ///< [in] device associated with the memory advice
    const void* ptr,                                                        ///< [in] Pointer to the start of the memory range
    size_t size,                                                            ///< [in] Size in bytes of the memory range
    ze_memory_atomic_attr_exp_flags_t attr                                  ///< [in] Atomic access attributes to set for the specified range.
                                                                            ///< Must be 0 (default) or a valid combination of ::ze_memory_atomic_attr_exp_flag_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves the atomic access attributes previously set for a shared
///        allocation
/// 
/// @details
///     - The application may call this function from simultaneous threads
///       with the same pointer.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == pAttr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetAtomicAccessAttributeExp(
    ze_context_handle_t hContext,                                           ///< [in] handle of context
    ze_device_handle_t hDevice,                                             ///< [in] device associated with the memory advice
    const void* ptr,                                                        ///< [in] Pointer to the start of the memory range
    size_t size,                                                            ///< [in] Size in bytes of the memory range
    ze_memory_atomic_attr_exp_flags_t* pAttr                                ///< [out] Atomic access attributes for the specified range
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Module
#if !defined(__GNUC__)
#pragma region module
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported module creation input formats
typedef enum _ze_module_format_t
{
    ZE_MODULE_FORMAT_IL_SPIRV = 0,                                          ///< Format is SPIRV IL format
    ZE_MODULE_FORMAT_NATIVE = 1,                                            ///< Format is device native format
    ZE_MODULE_FORMAT_FORCE_UINT32 = 0x7fffffff

} ze_module_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Specialization constants - User defined constants
typedef struct _ze_module_constants_t
{
    uint32_t numConstants;                                                  ///< [in] Number of specialization constants.
    const uint32_t* pConstantIds;                                           ///< [in][range(0, numConstants)] Array of IDs that is sized to
                                                                            ///< numConstants.
    const void** pConstantValues;                                           ///< [in][range(0, numConstants)] Array of pointers to values that is sized
                                                                            ///< to numConstants.

} ze_module_constants_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Module descriptor
typedef struct _ze_module_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_module_format_t format;                                              ///< [in] Module format passed in with pInputModule
    size_t inputSize;                                                       ///< [in] size of input IL or ISA from pInputModule.
    const uint8_t* pInputModule;                                            ///< [in] pointer to IL or ISA
    const char* pBuildFlags;                                                ///< [in][optional] string containing one or more (comma-separated)
                                                                            ///< compiler flags. If unsupported, flag is ignored with a warning.
                                                                            ///<  - "-ze-opt-disable"
                                                                            ///<       - Disable optimizations
                                                                            ///<  - "-ze-opt-level"
                                                                            ///<       - Specifies optimization level for compiler. Levels are
                                                                            ///< implementation specific.
                                                                            ///<           - 0 is no optimizations (equivalent to -ze-opt-disable)
                                                                            ///<           - 1 is optimize minimally (may be the same as 2)
                                                                            ///<           - 2 is optimize more (default)
                                                                            ///<  - "-ze-opt-greater-than-4GB-buffer-required"
                                                                            ///<       - Use 64-bit offset calculations for buffers.
                                                                            ///<  - "-ze-opt-large-register-file"
                                                                            ///<       - Increase number of registers available to threads.
                                                                            ///<  - "-ze-opt-has-buffer-offset-arg"
                                                                            ///<       - Extend stateless to stateful optimization to more
                                                                            ///<         cases with the use of additional offset (e.g. 64-bit
                                                                            ///<         pointer to binding table with 32-bit offset).
                                                                            ///<  - "-g"
                                                                            ///<       - Include debugging information.
    const ze_module_constants_t* pConstants;                                ///< [in][optional] pointer to specialization constants. Valid only for
                                                                            ///< SPIR-V input. This must be set to nullptr if no specialization
                                                                            ///< constants are provided.

} ze_module_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a module on the context.
/// 
/// @details
///     - Compiles the module for execution on the device.
///     - The application must only use the module for the device, or its
///       sub-devices, which was provided during creation.
///     - The module can be copied to other devices and contexts within the same
///       driver instance by using ::zeModuleGetNativeBinary.
///     - A build log can optionally be returned to the caller. The caller is
///       responsible for destroying build log using ::zeModuleBuildLogDestroy.
///     - The module descriptor constants are only supported for SPIR-V
///       specialization constants.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == desc->pInputModule`
///         + `nullptr == phModule`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_MODULE_FORMAT_NATIVE < desc->format`
///     - ::ZE_RESULT_ERROR_INVALID_NATIVE_BINARY
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `0 == desc->inputSize`
///     - ::ZE_RESULT_ERROR_MODULE_BUILD_FAILURE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleCreate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    const ze_module_desc_t* desc,                                           ///< [in] pointer to module descriptor
    ze_module_handle_t* phModule,                                           ///< [out] pointer to handle of module object created
    ze_module_build_log_handle_t* phBuildLog                                ///< [out][optional] pointer to handle of module's build log.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys module
/// 
/// @details
///     - The application must destroy all kernel handles created from the
///       module before destroying the module itself.
///     - The application must ensure the device is not currently referencing
///       the module before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this module.
///     - The application must **not** call this function from simultaneous
///       threads with the same module handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleDestroy(
    ze_module_handle_t hModule                                              ///< [in][release] handle of the module
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Dynamically link modules together that share import/export linkage
///        dependencies.
/// 
/// @details
///     - Modules support SPIR-V import and export linkage types for functions
///       and global variables. See the SPIR-V specification for linkage
///       details.
///     - Modules can have both import and export linkage.
///     - Modules that do not have any imports or exports do not need to be
///       linked.
///     - All module import requirements must be satisfied via linking before
///       kernel objects can be created from them.
///     - Modules cannot be partially linked. Unsatisfiable import dependencies
///       in the set of modules passed to ::zeModuleDynamicLink will result in 
///       ::ZE_RESULT_ERROR_MODULE_LINK_FAILURE being returned.
///     - Modules will only be linked once. A module can be used in multiple
///       link calls if it has exports but its imports will not be re-linked.
///     - Ambiguous dependencies, where multiple modules satisfy the same import
///       dependencies for a module, are not allowed.
///     - The application must ensure the modules being linked were created on
///       the same context.
///     - The application may call this function from simultaneous threads as
///       long as the import modules being linked are not the same.
///     - ModuleGetNativeBinary can be called on any module regardless of
///       whether it is linked or not.
///     - A link log can optionally be returned to the caller. The caller is
///       responsible for destroying the link log using
///       ::zeModuleBuildLogDestroy.
///     - The link log may contain a list of the unresolved import dependencies
///       if present.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phModules`
///     - ::ZE_RESULT_ERROR_MODULE_LINK_FAILURE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleDynamicLink(
    uint32_t numModules,                                                    ///< [in] number of modules to be linked pointed to by phModules.
    ze_module_handle_t* phModules,                                          ///< [in][range(0, numModules)] pointer to an array of modules to
                                                                            ///< dynamically link together.
    ze_module_build_log_handle_t* phLinkLog                                 ///< [out][optional] pointer to handle of dynamic link log.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys module build log object
/// 
/// @details
///     - The implementation of this function may immediately free all Host
///       allocations associated with this object.
///     - The application must **not** call this function from simultaneous
///       threads with the same build log handle.
///     - The implementation of this function should be lock-free.
///     - This function can be called before or after ::zeModuleDestroy for the
///       associated module.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModuleBuildLog`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleBuildLogDestroy(
    ze_module_build_log_handle_t hModuleBuildLog                            ///< [in][release] handle of the module build log object.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves text string for build log.
/// 
/// @details
///     - The caller can pass nullptr for pBuildLog when querying only for size.
///     - The caller must provide memory for build log.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModuleBuildLog`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleBuildLogGetString(
    ze_module_build_log_handle_t hModuleBuildLog,                           ///< [in] handle of the module build log object.
    size_t* pSize,                                                          ///< [in,out] size of build log string.
    char* pBuildLog                                                         ///< [in,out][optional] pointer to null-terminated string of the log.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve native binary from Module.
/// 
/// @details
///     - The native binary output can be cached to disk and new modules can be
///       later constructed from the cached copy.
///     - The native binary will retain debugging information that is associated
///       with a module.
///     - The caller can pass nullptr for pModuleNativeBinary when querying only
///       for size.
///     - The implementation will copy the native binary into a buffer supplied
///       by the caller.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetNativeBinary(
    ze_module_handle_t hModule,                                             ///< [in] handle of the module
    size_t* pSize,                                                          ///< [in,out] size of native binary in bytes.
    uint8_t* pModuleNativeBinary                                            ///< [in,out][optional] byte pointer to native binary
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve global variable pointer from Module.
/// 
/// @details
///     - The application may query global pointer from any module that either
///       exports or imports it.
///     - The application must dynamically link a module that imports a global
///       before the global pointer can be queried from it.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pGlobalName`
///     - ::ZE_RESULT_ERROR_INVALID_GLOBAL_NAME
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetGlobalPointer(
    ze_module_handle_t hModule,                                             ///< [in] handle of the module
    const char* pGlobalName,                                                ///< [in] name of global variable in module
    size_t* pSize,                                                          ///< [in,out][optional] size of global variable
    void** pptr                                                             ///< [in,out][optional] device visible pointer
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve all kernel names in the module.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetKernelNames(
    ze_module_handle_t hModule,                                             ///< [in] handle of the module
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of names.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of names available.
                                                                            ///< if count is greater than the number of names available, then the
                                                                            ///< driver shall update the value with the correct number of names available.
    const char** pNames                                                     ///< [in,out][optional][range(0, *pCount)] array of names of functions.
                                                                            ///< if count is less than the number of names available, then driver shall
                                                                            ///< only retrieve that number of names.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported module property flags
typedef uint32_t ze_module_property_flags_t;
typedef enum _ze_module_property_flag_t
{
    ZE_MODULE_PROPERTY_FLAG_IMPORTS = ZE_BIT(0),                            ///< Module has imports (i.e. imported global variables and/or kernels).
                                                                            ///< See ::zeModuleDynamicLink.
    ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_module_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Module properties
typedef struct _ze_module_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_module_property_flags_t flags;                                       ///< [out] 0 (none) or a valid combination of ::ze_module_property_flag_t

} ze_module_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve module properties.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pModuleProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetProperties(
    ze_module_handle_t hModule,                                             ///< [in] handle of the module
    ze_module_properties_t* pModuleProperties                               ///< [in,out] query result for module properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported kernel creation flags
typedef uint32_t ze_kernel_flags_t;
typedef enum _ze_kernel_flag_t
{
    ZE_KERNEL_FLAG_FORCE_RESIDENCY = ZE_BIT(0),                             ///< force all device allocations to be resident during execution
    ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY = ZE_BIT(1),                          ///< application is responsible for all residency of device allocations.
                                                                            ///< driver may disable implicit residency management.
    ZE_KERNEL_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_kernel_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel descriptor
typedef struct _ze_kernel_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_kernel_flags_t flags;                                                ///< [in] creation flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_kernel_flag_t;
                                                                            ///< default behavior may use driver-based residency.
    const char* pKernelName;                                                ///< [in] null-terminated name of kernel in module

} ze_kernel_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a kernel from the module.
/// 
/// @details
///     - Modules that have unresolved imports need to be dynamically linked
///       before a kernel can be created from them. (See ::zeModuleDynamicLink)
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == desc->pKernelName`
///         + `nullptr == phKernel`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < desc->flags`
///     - ::ZE_RESULT_ERROR_INVALID_KERNEL_NAME
///     - ::ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelCreate(
    ze_module_handle_t hModule,                                             ///< [in] handle of the module
    const ze_kernel_desc_t* desc,                                           ///< [in] pointer to kernel descriptor
    ze_kernel_handle_t* phKernel                                            ///< [out] handle of the Function object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a kernel object
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the kernel before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this kernel.
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelDestroy(
    ze_kernel_handle_t hKernel                                              ///< [in][release] handle of the kernel object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve a function pointer from a module by name
/// 
/// @details
///     - The function pointer is unique for the device on which the module was
///       created.
///     - The function pointer is no longer valid if module is destroyed.
///     - The function name should only refer to callable functions within the
///       module.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pFunctionName`
///         + `nullptr == pfnFunction`
///     - ::ZE_RESULT_ERROR_INVALID_FUNCTION_NAME
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetFunctionPointer(
    ze_module_handle_t hModule,                                             ///< [in] handle of the module
    const char* pFunctionName,                                              ///< [in] Name of function to retrieve function pointer for.
    void** pfnFunction                                                      ///< [out] pointer to function.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set group size for a kernel.
/// 
/// @details
///     - The group size will be used when a ::zeCommandListAppendLaunchKernel
///       variant is called.
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetGroupSize(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    uint32_t groupSizeX,                                                    ///< [in] group size for X dimension to use for this kernel
    uint32_t groupSizeY,                                                    ///< [in] group size for Y dimension to use for this kernel
    uint32_t groupSizeZ                                                     ///< [in] group size for Z dimension to use for this kernel
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Query a suggested group size for a kernel given a global size for each
///        dimension.
/// 
/// @details
///     - This function ignores the group size that is set using
///       ::zeKernelSetGroupSize.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == groupSizeX`
///         + `nullptr == groupSizeY`
///         + `nullptr == groupSizeZ`
///     - ::ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSuggestGroupSize(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    uint32_t globalSizeX,                                                   ///< [in] global width for X dimension
    uint32_t globalSizeY,                                                   ///< [in] global width for Y dimension
    uint32_t globalSizeZ,                                                   ///< [in] global width for Z dimension
    uint32_t* groupSizeX,                                                   ///< [out] recommended size of group for X dimension
    uint32_t* groupSizeY,                                                   ///< [out] recommended size of group for Y dimension
    uint32_t* groupSizeZ                                                    ///< [out] recommended size of group for Z dimension
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Query a suggested max group count for a cooperative kernel.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == totalGroupCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSuggestMaxCooperativeGroupCount(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    uint32_t* totalGroupCount                                               ///< [out] recommended total group count.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set kernel argument for a kernel.
/// 
/// @details
///     - The argument values will be used when a
///       ::zeCommandListAppendLaunchKernel variant is called.
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetArgumentValue(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    uint32_t argIndex,                                                      ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                                         ///< [in] size of argument type
    const void* pArgValue                                                   ///< [in][optional] argument value represented as matching arg type. If
                                                                            ///< null then argument value is considered null.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel indirect access flags
typedef uint32_t ze_kernel_indirect_access_flags_t;
typedef enum _ze_kernel_indirect_access_flag_t
{
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST = ZE_BIT(0),                        ///< Indicates that the kernel accesses host allocations indirectly.
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE = ZE_BIT(1),                      ///< Indicates that the kernel accesses device allocations indirectly.
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED = ZE_BIT(2),                      ///< Indicates that the kernel accesses shared allocations indirectly.
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_kernel_indirect_access_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets kernel indirect access flags.
/// 
/// @details
///     - The application should specify which allocations will be indirectly
///       accessed by the kernel to allow driver to optimize which allocations
///       are made resident
///     - This function may **not** be called from simultaneous threads with the
///       same Kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7 < flags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetIndirectAccess(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    ze_kernel_indirect_access_flags_t flags                                 ///< [in] kernel indirect access flags
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve kernel indirect access flags.
/// 
/// @details
///     - This function may be called from simultaneous threads with the same
///       Kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pFlags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetIndirectAccess(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    ze_kernel_indirect_access_flags_t* pFlags                               ///< [out] query result for kernel indirect access flags.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve all declared kernel attributes (i.e. can be specified with
///        __attribute__ in runtime language).
/// 
/// @details
///     - This function may be called from simultaneous threads with the same
///       Kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetSourceAttributes(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    uint32_t* pSize,                                                        ///< [in,out] pointer to size of string in bytes, including
                                                                            ///< null-terminating character.
    char** pString                                                          ///< [in,out][optional] pointer to application-managed character array
                                                                            ///< (string data).
                                                                            ///< If NULL, the string length of the kernel source attributes, including
                                                                            ///< a null-terminating character, is returned in pSize.
                                                                            ///< Otherwise, pString must point to valid application memory that is
                                                                            ///< greater than or equal to *pSize bytes in length, and on return the
                                                                            ///< pointed-to string will contain a space-separated list of kernel source attributes.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported Cache Config flags
typedef uint32_t ze_cache_config_flags_t;
typedef enum _ze_cache_config_flag_t
{
    ZE_CACHE_CONFIG_FLAG_LARGE_SLM = ZE_BIT(0),                             ///< Large SLM size
    ZE_CACHE_CONFIG_FLAG_LARGE_DATA = ZE_BIT(1),                            ///< Large General Data size
    ZE_CACHE_CONFIG_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_cache_config_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets the preferred cache configuration.
/// 
/// @details
///     - The cache configuration will be used when a
///       ::zeCommandListAppendLaunchKernel variant is called.
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < flags`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetCacheConfig(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    ze_cache_config_flags_t flags                                           ///< [in] cache configuration.
                                                                            ///< must be 0 (default configuration) or a valid combination of ::ze_cache_config_flag_t.
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_KERNEL_UUID_SIZE
/// @brief Maximum kernel universal unique id (UUID) size in bytes
#define ZE_MAX_KERNEL_UUID_SIZE  16
#endif // ZE_MAX_KERNEL_UUID_SIZE

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_MODULE_UUID_SIZE
/// @brief Maximum module universal unique id (UUID) size in bytes
#define ZE_MAX_MODULE_UUID_SIZE  16
#endif // ZE_MAX_MODULE_UUID_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel universal unique id (UUID)
typedef struct _ze_kernel_uuid_t
{
    uint8_t kid[ZE_MAX_KERNEL_UUID_SIZE];                                   ///< [out] opaque data representing a kernel UUID
    uint8_t mid[ZE_MAX_MODULE_UUID_SIZE];                                   ///< [out] opaque data representing the kernel's module UUID

} ze_kernel_uuid_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel properties
typedef struct _ze_kernel_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t numKernelArgs;                                                 ///< [out] number of kernel arguments.
    uint32_t requiredGroupSizeX;                                            ///< [out] required group size in the X dimension,
                                                                            ///< or zero if there is no required group size
    uint32_t requiredGroupSizeY;                                            ///< [out] required group size in the Y dimension,
                                                                            ///< or zero if there is no required group size
    uint32_t requiredGroupSizeZ;                                            ///< [out] required group size in the Z dimension,
                                                                            ///< or zero if there is no required group size
    uint32_t requiredNumSubGroups;                                          ///< [out] required number of subgroups per thread group,
                                                                            ///< or zero if there is no required number of subgroups
    uint32_t requiredSubgroupSize;                                          ///< [out] required subgroup size,
                                                                            ///< or zero if there is no required subgroup size
    uint32_t maxSubgroupSize;                                               ///< [out] maximum subgroup size
    uint32_t maxNumSubgroups;                                               ///< [out] maximum number of subgroups per thread group
    uint32_t localMemSize;                                                  ///< [out] local memory size used by each thread group
    uint32_t privateMemSize;                                                ///< [out] private memory size allocated by compiler used by each thread
    uint32_t spillMemSize;                                                  ///< [out] spill memory size allocated by compiler
    ze_kernel_uuid_t uuid;                                                  ///< [out] universal unique identifier.

} ze_kernel_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Additional kernel preferred group size properties
/// 
/// @details
///     - This structure may be passed to ::zeKernelGetProperties, via the
///       `pNext` member of ::ze_kernel_properties_t, to query additional kernel
///       preferred group size properties.
typedef struct _ze_kernel_preferred_group_size_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t preferredMultiple;                                             ///< [out] preferred group size multiple

} ze_kernel_preferred_group_size_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve kernel properties.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pKernelProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetProperties(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    ze_kernel_properties_t* pKernelProperties                               ///< [in,out] query result for kernel properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve kernel name from Kernel.
/// 
/// @details
///     - The caller can pass nullptr for pName when querying only for size.
///     - The implementation will copy the kernel name into a buffer supplied by
///       the caller.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetName(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    size_t* pSize,                                                          ///< [in,out] size of kernel name string, including null terminator, in
                                                                            ///< bytes.
    char* pName                                                             ///< [in,out][optional] char pointer to kernel name.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel dispatch group count.
typedef struct _ze_group_count_t
{
    uint32_t groupCountX;                                                   ///< [in] number of thread groups in X dimension
    uint32_t groupCountY;                                                   ///< [in] number of thread groups in Y dimension
    uint32_t groupCountZ;                                                   ///< [in] number of thread groups in Z dimension

} ze_group_count_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Launch kernel over one or more work groups.
/// 
/// @details
///     - The application must ensure the kernel and events are accessible by
///       the device on which the command list was created.
///     - This may **only** be called for a command list created with command
///       queue group ordinal that supports compute.
///     - The application must ensure the command list, kernel and events were
///       created on the same context.
///     - This function may **not** be called from simultaneous threads with the
///       same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLaunchFuncArgs`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchKernel(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    const ze_group_count_t* pLaunchFuncArgs,                                ///< [in] thread group launch arguments
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Launch kernel cooperatively over one or more work groups.
/// 
/// @details
///     - The application must ensure the kernel and events are accessible by
///       the device on which the command list was created.
///     - This may **only** be called for a command list created with command
///       queue group ordinal that supports compute.
///     - This may only be used for a command list that are submitted to command
///       queue with cooperative flag set.
///     - The application must ensure the command list, kernel and events were
///       created on the same context.
///     - This function may **not** be called from simultaneous threads with the
///       same command list handle.
///     - The implementation of this function should be lock-free.
///     - Use ::zeKernelSuggestMaxCooperativeGroupCount to recommend max group
///       count for device for cooperative functions that device supports.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLaunchFuncArgs`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchCooperativeKernel(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    const ze_group_count_t* pLaunchFuncArgs,                                ///< [in] thread group launch arguments
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Launch kernel over one or more work groups using indirect arguments.
/// 
/// @details
///     - The application must ensure the kernel and events are accessible by
///       the device on which the command list was created.
///     - The application must ensure the launch arguments are visible to the
///       device on which the command list was created.
///     - The implementation must not access the contents of the launch
///       arguments as they are free to be modified by either the Host or device
///       up until execution.
///     - This may **only** be called for a command list created with command
///       queue group ordinal that supports compute.
///     - The application must ensure the command list, kernel and events were
///       created, and the memory was allocated, on the same context.
///     - This function may **not** be called from simultaneous threads with the
///       same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLaunchArgumentsBuffer`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchKernelIndirect(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    const ze_group_count_t* pLaunchArgumentsBuffer,                         ///< [in] pointer to device buffer that will contain thread group launch
                                                                            ///< arguments
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Launch multiple kernels over one or more work groups using an array of
///        indirect arguments.
/// 
/// @details
///     - The application must ensure the kernel and events are accessible by
///       the device on which the command list was created.
///     - The application must ensure the array of launch arguments and count
///       buffer are visible to the device on which the command list was
///       created.
///     - The implementation must not access the contents of the array of launch
///       arguments or count buffer as they are free to be modified by either
///       the Host or device up until execution.
///     - This may **only** be called for a command list created with command
///       queue group ordinal that supports compute.
///     - The application must enusre the command list, kernel and events were
///       created, and the memory was allocated, on the same context.
///     - This function may **not** be called from simultaneous threads with the
///       same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phKernels`
///         + `nullptr == pCountBuffer`
///         + `nullptr == pLaunchArgumentsBuffer`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchMultipleKernelsIndirect(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of the command list
    uint32_t numKernels,                                                    ///< [in] maximum number of kernels to launch
    ze_kernel_handle_t* phKernels,                                          ///< [in][range(0, numKernels)] handles of the kernel objects
    const uint32_t* pCountBuffer,                                           ///< [in] pointer to device memory location that will contain the actual
                                                                            ///< number of kernels to launch; value must be less than or equal to
                                                                            ///< numKernels
    const ze_group_count_t* pLaunchArgumentsBuffer,                         ///< [in][range(0, numKernels)] pointer to device buffer that will contain
                                                                            ///< a contiguous array of thread group launch arguments
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting module programs.
#if !defined(__GNUC__)
#pragma region program
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MODULE_PROGRAM_EXP_NAME
/// @brief Module Program Extension Name
#define ZE_MODULE_PROGRAM_EXP_NAME  "ZE_experimental_module_program"
#endif // ZE_MODULE_PROGRAM_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Module Program Extension Version(s)
typedef enum _ze_module_program_exp_version_t
{
    ZE_MODULE_PROGRAM_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),            ///< version 1.0
    ZE_MODULE_PROGRAM_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),        ///< latest known version
    ZE_MODULE_PROGRAM_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_module_program_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Module extended descriptor to support multiple input modules.
/// 
/// @details
///     - Implementation must support ::ZE_experimental_module_program extension
///     - Modules support import and export linkage for functions and global
///       variables.
///     - SPIR-V import and export linkage types are used. See SPIR-V
///       specification for linkage details.
///     - pInputModules, pBuildFlags, and pConstants from ::ze_module_desc_t is
///       ignored.
///     - Format in ::ze_module_desc_t needs to be set to
///       ::ZE_MODULE_FORMAT_IL_SPIRV.
typedef struct _ze_module_program_exp_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t count;                                                         ///< [in] Count of input modules
    const size_t* inputSizes;                                               ///< [in][range(0, count)] sizes of each input IL module in pInputModules.
    const uint8_t** pInputModules;                                          ///< [in][range(0, count)] pointer to an array of IL (e.g. SPIR-V modules).
                                                                            ///< Valid only for SPIR-V input.
    const char** pBuildFlags;                                               ///< [in][optional][range(0, count)] array of strings containing build
                                                                            ///< flags. See pBuildFlags in ::ze_module_desc_t.
    const ze_module_constants_t** pConstants;                               ///< [in][optional][range(0, count)] pointer to array of specialization
                                                                            ///< constant strings. Valid only for SPIR-V input. This must be set to
                                                                            ///< nullptr if no specialization constants are provided.

} ze_module_program_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Raytracing
#if !defined(__GNUC__)
#pragma region raytracing
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_RAYTRACING_EXT_NAME
/// @brief Raytracing Extension Name
#define ZE_RAYTRACING_EXT_NAME  "ZE_extension_raytracing"
#endif // ZE_RAYTRACING_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Raytracing Extension Version(s)
typedef enum _ze_raytracing_ext_version_t
{
    ZE_RAYTRACING_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                ///< version 1.0
    ZE_RAYTRACING_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),            ///< latest known version
    ZE_RAYTRACING_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_raytracing_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported raytracing capability flags
typedef uint32_t ze_device_raytracing_ext_flags_t;
typedef enum _ze_device_raytracing_ext_flag_t
{
    ZE_DEVICE_RAYTRACING_EXT_FLAG_RAYQUERY = ZE_BIT(0),                     ///< Supports rayquery
    ZE_DEVICE_RAYTRACING_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_raytracing_ext_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Raytracing properties queried using ::zeDeviceGetModuleProperties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetModuleProperties, via
///       the `pNext` member of ::ze_device_module_properties_t.
typedef struct _ze_device_raytracing_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_raytracing_ext_flags_t flags;                                 ///< [out] 0 or a valid combination of ::ze_device_raytracing_ext_flags_t
    uint32_t maxBVHLevels;                                                  ///< [out] Maximum number of BVH levels supported

} ze_device_raytracing_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported raytracing memory allocation flags
typedef uint32_t ze_raytracing_mem_alloc_ext_flags_t;
typedef enum _ze_raytracing_mem_alloc_ext_flag_t
{
    ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_TBD = ZE_BIT(0),                       ///< reserved for future use
    ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_raytracing_mem_alloc_ext_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Raytracing memory allocation descriptor
/// 
/// @details
///     - This structure must be passed to ::zeMemAllocShared or
///       ::zeMemAllocDevice, via the `pNext` member of
///       ::ze_device_mem_alloc_desc_t, for any memory allocation that is to be
///       accessed by raytracing fixed-function of the device.
typedef struct _ze_raytracing_mem_alloc_ext_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_raytracing_mem_alloc_ext_flags_t flags;                              ///< [in] flags specifying additional allocation controls.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_raytracing_mem_alloc_ext_flag_t;
                                                                            ///< default behavior may use implicit driver-based heuristics.

} ze_raytracing_mem_alloc_ext_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Memory Residency
#if !defined(__GNUC__)
#pragma region residency
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Makes memory resident for the device.
/// 
/// @details
///     - The application must ensure the memory is resident before being
///       referenced by the device
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextMakeMemoryResident(
    ze_context_handle_t hContext,                                           ///< [in] handle of context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    void* ptr,                                                              ///< [in] pointer to memory to make resident
    size_t size                                                             ///< [in] size in bytes to make resident
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Allows memory to be evicted from the device.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the memory before it is evicted
///     - The application may free the memory without evicting; the memory is
///       implicitly evicted when freed.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextEvictMemory(
    ze_context_handle_t hContext,                                           ///< [in] handle of context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    void* ptr,                                                              ///< [in] pointer to memory to evict
    size_t size                                                             ///< [in] size in bytes to evict
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes image resident for the device.
/// 
/// @details
///     - The application must ensure the image is resident before being
///       referenced by the device
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///         + `nullptr == hImage`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextMakeImageResident(
    ze_context_handle_t hContext,                                           ///< [in] handle of context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_image_handle_t hImage                                                ///< [in] handle of image to make resident
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Allows image to be evicted from the device.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the image before it is evicted
///     - The application may destroy the image without evicting; the image is
///       implicitly evicted when destroyed.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///         + `nullptr == hImage`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextEvictImage(
    ze_context_handle_t hContext,                                           ///< [in] handle of context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_image_handle_t hImage                                                ///< [in] handle of image to make evict
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Sampler
#if !defined(__GNUC__)
#pragma region sampler
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler addressing modes
typedef enum _ze_sampler_address_mode_t
{
    ZE_SAMPLER_ADDRESS_MODE_NONE = 0,                                       ///< No coordinate modifications for out-of-bounds image access.
    ZE_SAMPLER_ADDRESS_MODE_REPEAT = 1,                                     ///< Out-of-bounds coordinates are wrapped back around.
    ZE_SAMPLER_ADDRESS_MODE_CLAMP = 2,                                      ///< Out-of-bounds coordinates are clamped to edge.
    ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3,                            ///< Out-of-bounds coordinates are clamped to border color which is (0.0f,
                                                                            ///< 0.0f, 0.0f, 0.0f) if image format swizzle contains alpha, otherwise
                                                                            ///< (0.0f, 0.0f, 0.0f, 1.0f).
    ZE_SAMPLER_ADDRESS_MODE_MIRROR = 4,                                     ///< Out-of-bounds coordinates are mirrored starting from edge.
    ZE_SAMPLER_ADDRESS_MODE_FORCE_UINT32 = 0x7fffffff

} ze_sampler_address_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler filtering modes
typedef enum _ze_sampler_filter_mode_t
{
    ZE_SAMPLER_FILTER_MODE_NEAREST = 0,                                     ///< No coordinate modifications for out of bounds image access.
    ZE_SAMPLER_FILTER_MODE_LINEAR = 1,                                      ///< Out-of-bounds coordinates are wrapped back around.
    ZE_SAMPLER_FILTER_MODE_FORCE_UINT32 = 0x7fffffff

} ze_sampler_filter_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler descriptor
typedef struct _ze_sampler_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_sampler_address_mode_t addressMode;                                  ///< [in] Sampler addressing mode to determine how out-of-bounds
                                                                            ///< coordinates are handled.
    ze_sampler_filter_mode_t filterMode;                                    ///< [in] Sampler filter mode to determine how samples are filtered.
    ze_bool_t isNormalized;                                                 ///< [in] Are coordinates normalized [0, 1] or not.

} ze_sampler_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates sampler on the context.
/// 
/// @details
///     - The application must only use the sampler for the device, or its
///       sub-devices, which was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phSampler`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_SAMPLER_ADDRESS_MODE_MIRROR < desc->addressMode`
///         + `::ZE_SAMPLER_FILTER_MODE_LINEAR < desc->filterMode`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeSamplerCreate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    const ze_sampler_desc_t* desc,                                          ///< [in] pointer to sampler descriptor
    ze_sampler_handle_t* phSampler                                          ///< [out] handle of the sampler
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys sampler object
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the sampler before it is deleted.
///     - The implementation of this function may immediately free all Host and
///       Device allocations associated with this sampler.
///     - The application must **not** call this function from simultaneous
///       threads with the same sampler handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hSampler`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeSamplerDestroy(
    ze_sampler_handle_t hSampler                                            ///< [in][release] handle of the sampler
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs for Virtual Memory Management
#if !defined(__GNUC__)
#pragma region virtual
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Virtual memory page access attributes
typedef enum _ze_memory_access_attribute_t
{
    ZE_MEMORY_ACCESS_ATTRIBUTE_NONE = 0,                                    ///< Indicates the memory page is inaccessible.
    ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1,                               ///< Indicates the memory page supports read write access.
    ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY = 2,                                ///< Indicates the memory page supports read-only access.
    ZE_MEMORY_ACCESS_ATTRIBUTE_FORCE_UINT32 = 0x7fffffff

} ze_memory_access_attribute_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Reserves pages in virtual address space.
/// 
/// @details
///     - The application must only use the memory allocation on the context for
///       which it was created.
///     - The starting address and size must be page aligned. See
///       ::zeVirtualMemQueryPageSize.
///     - If pStart is not null then implementation will attempt to reserve
///       starting from that address. If not available then will find another
///       suitable starting address.
///     - The application may call this function from simultaneous threads.
///     - The access attributes will default to none to indicate reservation is
///       inaccessible.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pptr`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemReserve(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* pStart,                                                     ///< [in][optional] pointer to start of region to reserve. If nullptr then
                                                                            ///< implementation will choose a start address.
    size_t size,                                                            ///< [in] size in bytes to reserve; must be page aligned.
    void** pptr                                                             ///< [out] pointer to virtual reservation.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Free pages in a reserved virtual address range.
/// 
/// @details
///     - Any existing virtual mappings for the range will be unmapped.
///     - Physical allocations objects that were mapped to this range will not
///       be destroyed. These need to be destroyed explicitly.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemFree(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] pointer to start of region to free.
    size_t size                                                             ///< [in] size in bytes to free; must be page aligned.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries page size to use for aligning virtual memory reservations and
///        physical memory allocations.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pagesize`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemQueryPageSize(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    size_t size,                                                            ///< [in] unaligned allocation size in bytes
    size_t* pagesize                                                        ///< [out] pointer to page size to use for start address and size
                                                                            ///< alignments.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported physical memory creation flags
typedef uint32_t ze_physical_mem_flags_t;
typedef enum _ze_physical_mem_flag_t
{
    ZE_PHYSICAL_MEM_FLAG_TBD = ZE_BIT(0),                                   ///< reserved for future use.
    ZE_PHYSICAL_MEM_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_physical_mem_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Physical memory descriptor
typedef struct _ze_physical_mem_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_physical_mem_flags_t flags;                                          ///< [in] creation flags.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_physical_mem_flag_t.
    size_t size;                                                            ///< [in] size in bytes to reserve; must be page aligned.

} ze_physical_mem_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a physical memory object for the context.
/// 
/// @details
///     - The application must only use the physical memory object on the
///       context for which it was created.
///     - The size must be page aligned. See ::zeVirtualMemQueryPageSize.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phPhysicalMemory`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < desc->flags`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == desc->size`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
ZE_APIEXPORT ze_result_t ZE_APICALL
zePhysicalMemCreate(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    ze_physical_mem_desc_t* desc,                                           ///< [in] pointer to physical memory descriptor.
    ze_physical_mem_handle_t* phPhysicalMemory                              ///< [out] pointer to handle of physical memory object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a physical memory object.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the physical memory object before it is deleted
///     - The application must **not** call this function from simultaneous
///       threads with the same physical memory handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hPhysicalMemory`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zePhysicalMemDestroy(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_physical_mem_handle_t hPhysicalMemory                                ///< [in][release] handle of physical memory object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Maps pages in virtual address space to pages from physical memory
///        object.
/// 
/// @details
///     - The virtual address range must have been reserved using
///       ::zeVirtualMemReserve.
///     - The application must only use the mapped memory allocation on the
///       context for which it was created.
///     - The virtual start address and size must be page aligned. See
///       ::zeVirtualMemQueryPageSize.
///     - The application should use, for the starting address and size, the
///       same size alignment used for the physical allocation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hPhysicalMemory`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY < access`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemMap(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] pointer to start of virtual address range to map.
    size_t size,                                                            ///< [in] size in bytes of virtual address range to map; must be page
                                                                            ///< aligned.
    ze_physical_mem_handle_t hPhysicalMemory,                               ///< [in] handle to physical memory object.
    size_t offset,                                                          ///< [in] offset into physical memory allocation object; must be page
                                                                            ///< aligned.
    ze_memory_access_attribute_t access                                     ///< [in] specifies page access attributes to apply to the virtual address
                                                                            ///< range.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Unmaps pages in virtual address space from pages from a physical
///        memory object.
/// 
/// @details
///     - The page access attributes for virtual address range will revert back
///       to none.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
///         + Address must be page aligned
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///         + Size must be page aligned
ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemUnmap(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] pointer to start of region to unmap.
    size_t size                                                             ///< [in] size in bytes to unmap; must be page aligned.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set memory access attributes for a virtual address range.
/// 
/// @details
///     - This function may be called from simultaneous threads with the same
///       function handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY < access`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
///         + Address must be page aligned
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///         + Size must be page aligned
ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemSetAccessAttribute(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] pointer to start of reserved virtual address region.
    size_t size,                                                            ///< [in] size in bytes; must be page aligned.
    ze_memory_access_attribute_t access                                     ///< [in] specifies page access attributes to apply to the virtual address
                                                                            ///< range.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get memory access attribute for a virtual address range.
/// 
/// @details
///     - If size and outSize are equal then the pages in the specified virtual
///       address range have the same access attributes.
///     - This function may be called from simultaneous threads with the same
///       function handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == access`
///         + `nullptr == outSize`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT
///         + Address must be page aligned
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_SIZE
///         + `0 == size`
///         + Size must be page aligned
ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemGetAccessAttribute(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const void* ptr,                                                        ///< [in] pointer to start of virtual address region for query.
    size_t size,                                                            ///< [in] size in bytes; must be page aligned.
    ze_memory_access_attribute_t* access,                                   ///< [out] query result for page access attribute.
    size_t* outSize                                                         ///< [out] query result for size of virtual address range, starting at ptr,
                                                                            ///< that shares same access attribute.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Floating-Point Atomics
#if !defined(__GNUC__)
#pragma region floatAtomics
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_FLOAT_ATOMICS_EXT_NAME
/// @brief Floating-Point Atomics Extension Name
#define ZE_FLOAT_ATOMICS_EXT_NAME  "ZE_extension_float_atomics"
#endif // ZE_FLOAT_ATOMICS_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Floating-Point Atomics Extension Version(s)
typedef enum _ze_float_atomics_ext_version_t
{
    ZE_FLOAT_ATOMICS_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),             ///< version 1.0
    ZE_FLOAT_ATOMICS_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),         ///< latest known version
    ZE_FLOAT_ATOMICS_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_float_atomics_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported floating-point atomic capability flags
typedef uint32_t ze_device_fp_atomic_ext_flags_t;
typedef enum _ze_device_fp_atomic_ext_flag_t
{
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE = ZE_BIT(0),             ///< Supports atomic load, store, and exchange
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD = ZE_BIT(1),                    ///< Supports atomic add and subtract
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX = ZE_BIT(2),                ///< Supports atomic min and max
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE = ZE_BIT(16),             ///< Supports atomic load, store, and exchange
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD = ZE_BIT(17),                    ///< Supports atomic add and subtract
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX = ZE_BIT(18),                ///< Supports atomic min and max
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_fp_atomic_ext_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device floating-point atomic properties queried using
///        ::zeDeviceGetModuleProperties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetModuleProperties, via
///       the `pNext` member of ::ze_device_module_properties_t.
typedef struct _ze_float_atomic_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_fp_atomic_ext_flags_t fp16Flags;                              ///< [out] Capabilities for half-precision floating-point atomic operations
    ze_device_fp_atomic_ext_flags_t fp32Flags;                              ///< [out] Capabilities for single-precision floating-point atomic
                                                                            ///< operations
    ze_device_fp_atomic_ext_flags_t fp64Flags;                              ///< [out] Capabilities for double-precision floating-point atomic
                                                                            ///< operations

} ze_float_atomic_ext_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting kernel global work offset.
#if !defined(__GNUC__)
#pragma region globaloffset
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_GLOBAL_OFFSET_EXP_NAME
/// @brief Global Offset Extension Name
#define ZE_GLOBAL_OFFSET_EXP_NAME  "ZE_experimental_global_offset"
#endif // ZE_GLOBAL_OFFSET_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Global Offset Extension Version(s)
typedef enum _ze_global_offset_exp_version_t
{
    ZE_GLOBAL_OFFSET_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),             ///< version 1.0
    ZE_GLOBAL_OFFSET_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),         ///< latest known version
    ZE_GLOBAL_OFFSET_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_global_offset_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set global work offset for a kernel.
/// 
/// @details
///     - The global work offset will be used when a
///       ::zeCommandListAppendLaunchKernel() variant is called.
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetGlobalOffsetExp(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    uint32_t offsetX,                                                       ///< [in] global offset for X dimension to use for this kernel
    uint32_t offsetY,                                                       ///< [in] global offset for Y dimension to use for this kernel
    uint32_t offsetZ                                                        ///< [in] global offset for Z dimension to use for this kernel
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting relaxed allocation limits.
#if !defined(__GNUC__)
#pragma region relaxedAllocLimits
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME
/// @brief Relaxed Allocation Limits Extension Name
#define ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME  "ZE_experimental_relaxed_allocation_limits"
#endif // ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Relaxed Allocation Limits Extension Version(s)
typedef enum _ze_relaxed_allocation_limits_exp_version_t
{
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), ///< version 1.0
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), ///< latest known version
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_relaxed_allocation_limits_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported relaxed memory allocation flags
typedef uint32_t ze_relaxed_allocation_limits_exp_flags_t;
typedef enum _ze_relaxed_allocation_limits_exp_flag_t
{
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE = ZE_BIT(0),             ///< Allocation size may exceed the `maxMemAllocSize` member of
                                                                            ///< ::ze_device_properties_t.
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_relaxed_allocation_limits_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Relaxed limits memory allocation descriptor
/// 
/// @details
///     - This structure may be passed to ::zeMemAllocShared or
///       ::zeMemAllocDevice, via the `pNext` member of
///       ::ze_device_mem_alloc_desc_t.
///     - This structure may also be passed to ::zeMemAllocHost, via the `pNext`
///       member of ::ze_host_mem_alloc_desc_t.
typedef struct _ze_relaxed_allocation_limits_exp_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_relaxed_allocation_limits_exp_flags_t flags;                         ///< [in] flags specifying allocation limits to relax.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_relaxed_allocation_limits_exp_flag_t;

} ze_relaxed_allocation_limits_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Cache Reservation
#if !defined(__GNUC__)
#pragma region cacheReservation
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_CACHE_RESERVATION_EXT_NAME
/// @brief Cache_Reservation Extension Name
#define ZE_CACHE_RESERVATION_EXT_NAME  "ZE_extension_cache_reservation"
#endif // ZE_CACHE_RESERVATION_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Cache_Reservation Extension Version(s)
typedef enum _ze_cache_reservation_ext_version_t
{
    ZE_CACHE_RESERVATION_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),         ///< version 1.0
    ZE_CACHE_RESERVATION_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),     ///< latest known version
    ZE_CACHE_RESERVATION_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_cache_reservation_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Cache Reservation Region
typedef enum _ze_cache_ext_region_t
{
    ZE_CACHE_EXT_REGION_ZE_CACHE_REGION_DEFAULT = 0,                        ///< [DEPRECATED] utilize driver default scheme. Use
                                                                            ///< ::ZE_CACHE_EXT_REGION_DEFAULT.
    ZE_CACHE_EXT_REGION_ZE_CACHE_RESERVE_REGION = 1,                        ///< [DEPRECATED] utilize reserved region. Use
                                                                            ///< ::ZE_CACHE_EXT_REGION_RESERVED.
    ZE_CACHE_EXT_REGION_ZE_CACHE_NON_RESERVED_REGION = 2,                   ///< [DEPRECATED] utilize non-reserverd region. Use
                                                                            ///< ::ZE_CACHE_EXT_REGION_NON_RESERVED.
    ZE_CACHE_EXT_REGION_DEFAULT = 0,                                        ///< utilize driver default scheme
    ZE_CACHE_EXT_REGION_RESERVED = 1,                                       ///< utilize reserved region
    ZE_CACHE_EXT_REGION_NON_RESERVED = 2,                                   ///< utilize non-reserverd region
    ZE_CACHE_EXT_REGION_FORCE_UINT32 = 0x7fffffff

} ze_cache_ext_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief CacheReservation structure
/// 
/// @details
///     - This structure must be passed to ::zeDeviceGetCacheProperties via the
///       `pNext` member of ::ze_device_cache_properties_t
///     - Used for determining the max cache reservation allowed on device. Size
///       of zero means no reservation available.
typedef struct _ze_cache_reservation_ext_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    size_t maxCacheReservationSize;                                         ///< [out] max cache reservation size

} ze_cache_reservation_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Reserve Cache on Device
/// 
/// @details
///     - The application may call this function but may not be successful as
///       some other application may have reserve prior
/// 
/// @remarks
///   _Analogues_
///     - None
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceReserveCacheExt(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    size_t cacheLevel,                                                      ///< [in] cache level where application want to reserve. If zero, then the
                                                                            ///< driver shall default to last level of cache and attempt to reserve in
                                                                            ///< that cache.
    size_t cacheReservationSize                                             ///< [in] value for reserving size, in bytes. If zero, then the driver
                                                                            ///< shall remove prior reservation
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Assign VA section to use reserved section
/// 
/// @details
///     - The application may call this function to assign VA to particular
///       reservartion region
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_CACHE_EXT_REGION_NON_RESERVED < cacheRegion`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceSetCacheAdviceExt(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object
    void* ptr,                                                              ///< [in] memory pointer to query
    size_t regionSize,                                                      ///< [in] region size, in pages
    ze_cache_ext_region_t cacheRegion                                       ///< [in] reservation region
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting event query timestamps.
#if !defined(__GNUC__)
#pragma region eventquerytimestamps
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_EVENT_QUERY_TIMESTAMPS_EXP_NAME
/// @brief Event Query Timestamps Extension Name
#define ZE_EVENT_QUERY_TIMESTAMPS_EXP_NAME  "ZE_experimental_event_query_timestamps"
#endif // ZE_EVENT_QUERY_TIMESTAMPS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Event Query Timestamps Extension Version(s)
typedef enum _ze_event_query_timestamps_exp_version_t
{
    ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),    ///< version 1.0
    ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),///< latest known version
    ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_event_query_timestamps_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query event timestamps for a device or sub-device.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///     - The implementation must support
///       ::ZE_experimental_event_query_timestamps.
///     - The implementation must return all timestamps for the specified event
///       and device pair.
///     - The implementation must return all timestamps for all sub-devices when
///       device handle is parent device.
///     - The implementation may return all timestamps for sub-devices when
///       device handle is sub-device or may return 0 for count.
/// 
/// @remarks
///   _Analogues_
///     - None
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventQueryTimestampsExp(
    ze_event_handle_t hEvent,                                               ///< [in] handle of the event
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device to query
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of timestamp results.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of timestamps available.
                                                                            ///< if count is greater than the number of timestamps available, then the
                                                                            ///< driver shall update the value with the correct number of timestamps available.
    ze_kernel_timestamp_result_t* pTimestamps                               ///< [in,out][optional][range(0, *pCount)] array of timestamp results.
                                                                            ///< if count is less than the number of timestamps available, then driver
                                                                            ///< shall only retrieve that number of timestamps.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting image memory properties.
#if !defined(__GNUC__)
#pragma region imagememoryproperties
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_IMAGE_MEMORY_PROPERTIES_EXP_NAME
/// @brief Image Memory Properties Extension Name
#define ZE_IMAGE_MEMORY_PROPERTIES_EXP_NAME  "ZE_experimental_image_memory_properties"
#endif // ZE_IMAGE_MEMORY_PROPERTIES_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Image Memory Properties Extension Version(s)
typedef enum _ze_image_memory_properties_exp_version_t
{
    ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),   ///< version 1.0
    ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),   ///< latest known version
    ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_memory_properties_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image memory properties
typedef struct _ze_image_memory_properties_exp_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t size;                                                          ///< [out] size of image allocation in bytes.
    uint64_t rowPitch;                                                      ///< [out] size of image row in bytes.
    uint64_t slicePitch;                                                    ///< [out] size of image slice in bytes.

} ze_image_memory_properties_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query image memory properties.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///     - The implementation must support
///       ::ZE_experimental_image_memory_properties extension.
/// 
/// @remarks
///   _Analogues_
///     - None
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pMemoryProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageGetMemoryPropertiesExp(
    ze_image_handle_t hImage,                                               ///< [in] handle of image object
    ze_image_memory_properties_exp_t* pMemoryProperties                     ///< [in,out] query result for image memory properties.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting image views.
#if !defined(__GNUC__)
#pragma region imageview
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_IMAGE_VIEW_EXT_NAME
/// @brief Image View Extension Name
#define ZE_IMAGE_VIEW_EXT_NAME  "ZE_extension_image_view"
#endif // ZE_IMAGE_VIEW_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Image View Extension Version(s)
typedef enum _ze_image_view_ext_version_t
{
    ZE_IMAGE_VIEW_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                ///< version 1.0
    ZE_IMAGE_VIEW_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),            ///< latest known version
    ZE_IMAGE_VIEW_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_view_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create image view on the context.
/// 
/// @details
///     - The application must only use the image view for the device, or its
///       sub-devices, which was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///     - The implementation must support ::ZE_extension_image_view extension.
///     - Image views are treated as images from the API.
///     - Image views provide a mechanism to redescribe how an image is
///       interpreted (e.g. different format).
///     - Image views become disabled when their corresponding image resource is
///       destroyed.
///     - Use ::zeImageDestroy to destroy image view objects.
/// 
/// @remarks
///   _Analogues_
///     - None
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///         + `nullptr == hImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phImageView`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < desc->flags`
///         + `::ZE_IMAGE_TYPE_BUFFER < desc->type`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageViewCreateExt(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    const ze_image_desc_t* desc,                                            ///< [in] pointer to image descriptor
    ze_image_handle_t hImage,                                               ///< [in] handle of image object to create view from
    ze_image_handle_t* phImageView                                          ///< [out] pointer to handle of image object created for view
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_IMAGE_VIEW_EXP_NAME
/// @brief Image View Extension Name
#define ZE_IMAGE_VIEW_EXP_NAME  "ZE_experimental_image_view"
#endif // ZE_IMAGE_VIEW_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Image View Extension Version(s)
typedef enum _ze_image_view_exp_version_t
{
    ZE_IMAGE_VIEW_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                ///< version 1.0
    ZE_IMAGE_VIEW_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),            ///< latest known version
    ZE_IMAGE_VIEW_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_view_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create image view on the context.
/// 
/// @details
///     - The application must only use the image view for the device, or its
///       sub-devices, which was provided during creation.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///     - The implementation must support ::ZE_experimental_image_view
///       extension.
///     - Image views are treated as images from the API.
///     - Image views provide a mechanism to redescribe how an image is
///       interpreted (e.g. different format).
///     - Image views become disabled when their corresponding image resource is
///       destroyed.
///     - Use ::zeImageDestroy to destroy image view objects.
///     - Note: This function is deprecated and replaced by
///       ::zeImageViewCreateExt.
/// 
/// @remarks
///   _Analogues_
///     - None
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///         + `nullptr == hImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phImageView`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < desc->flags`
///         + `::ZE_IMAGE_TYPE_BUFFER < desc->type`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT
ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageViewCreateExp(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    const ze_image_desc_t* desc,                                            ///< [in] pointer to image descriptor
    ze_image_handle_t hImage,                                               ///< [in] handle of image object to create view from
    ze_image_handle_t* phImageView                                          ///< [out] pointer to handle of image object created for view
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting image views for planar images.
#if !defined(__GNUC__)
#pragma region imageviewplanar
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_IMAGE_VIEW_PLANAR_EXT_NAME
/// @brief Image View Planar Extension Name
#define ZE_IMAGE_VIEW_PLANAR_EXT_NAME  "ZE_extension_image_view_planar"
#endif // ZE_IMAGE_VIEW_PLANAR_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Image View Planar Extension Version(s)
typedef enum _ze_image_view_planar_ext_version_t
{
    ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),         ///< version 1.0
    ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),     ///< latest known version
    ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_view_planar_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image view planar descriptor
typedef struct _ze_image_view_planar_ext_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t planeIndex;                                                    ///< [in] the 0-based plane index (e.g. NV12 is 0 = Y plane, 1 UV plane)

} ze_image_view_planar_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_IMAGE_VIEW_PLANAR_EXP_NAME
/// @brief Image View Planar Extension Name
#define ZE_IMAGE_VIEW_PLANAR_EXP_NAME  "ZE_experimental_image_view_planar"
#endif // ZE_IMAGE_VIEW_PLANAR_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Image View Planar Extension Version(s)
typedef enum _ze_image_view_planar_exp_version_t
{
    ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),         ///< version 1.0
    ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),     ///< latest known version
    ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_view_planar_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image view planar descriptor
typedef struct _ze_image_view_planar_exp_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t planeIndex;                                                    ///< [in] the 0-based plane index (e.g. NV12 is 0 = Y plane, 1 UV plane)

} ze_image_view_planar_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for specifying kernel scheduling hints.
#if !defined(__GNUC__)
#pragma region kernelSchedulingHints
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_KERNEL_SCHEDULING_HINTS_EXP_NAME
/// @brief Kernel Scheduling Hints Extension Name
#define ZE_KERNEL_SCHEDULING_HINTS_EXP_NAME  "ZE_experimental_scheduling_hints"
#endif // ZE_KERNEL_SCHEDULING_HINTS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel Scheduling Hints Extension Version(s)
typedef enum _ze_scheduling_hints_exp_version_t
{
    ZE_SCHEDULING_HINTS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),          ///< version 1.0
    ZE_SCHEDULING_HINTS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),      ///< latest known version
    ZE_SCHEDULING_HINTS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_scheduling_hints_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported kernel scheduling hint flags
typedef uint32_t ze_scheduling_hint_exp_flags_t;
typedef enum _ze_scheduling_hint_exp_flag_t
{
    ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST = ZE_BIT(0),                   ///< Hint that the kernel prefers oldest-first scheduling
    ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN = ZE_BIT(1),                    ///< Hint that the kernel prefers round-robin scheduling
    ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN = ZE_BIT(2),        ///< Hint that the kernel prefers stall-based round-robin scheduling
    ZE_SCHEDULING_HINT_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_scheduling_hint_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device kernel scheduling hint properties queried using
///        ::zeDeviceGetModuleProperties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetModuleProperties, via
///       the `pNext` member of ::ze_device_module_properties_t.
typedef struct _ze_scheduling_hint_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_scheduling_hint_exp_flags_t schedulingHintFlags;                     ///< [out] Supported kernel scheduling hints.
                                                                            ///< May be 0 (none) or a valid combination of ::ze_scheduling_hint_exp_flag_t.

} ze_scheduling_hint_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel scheduling hint descriptor
/// 
/// @details
///     - This structure may be passed to ::zeKernelSchedulingHintExp.
typedef struct _ze_scheduling_hint_exp_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_scheduling_hint_exp_flags_t flags;                                   ///< [in] flags specifying kernel scheduling hints.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_scheduling_hint_exp_flag_t.

} ze_scheduling_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Provide kernel scheduling hints that may improve performance
/// 
/// @details
///     - The scheduling hints may improve performance only and are not required
///       for correctness.
///     - If a specified scheduling hint is unsupported it will be silently
///       ignored.
///     - If two conflicting scheduling hints are specified there is no defined behavior;
///       the hints may be ignored or one hint may be chosen arbitrarily.
///     - The application must not call this function from simultaneous threads
///       with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pHint`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7 < pHint->flags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSchedulingHintExp(
    ze_kernel_handle_t hKernel,                                             ///< [in] handle of the kernel object
    ze_scheduling_hint_exp_desc_t* pHint                                    ///< [in] pointer to kernel scheduling hint descriptor
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for One-Definition-Rule Linkage Types
#if !defined(__GNUC__)
#pragma region linkonceodr
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_LINKONCE_ODR_EXT_NAME
/// @brief Linkonce ODR Extension Name
#define ZE_LINKONCE_ODR_EXT_NAME  "ZE_extension_linkonce_odr"
#endif // ZE_LINKONCE_ODR_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Linkonce ODR Extension Version(s)
typedef enum _ze_linkonce_odr_ext_version_t
{
    ZE_LINKONCE_ODR_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),              ///< version 1.0
    ZE_LINKONCE_ODR_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),          ///< latest known version
    ZE_LINKONCE_ODR_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_linkonce_odr_ext_version_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting power saving hint.
#if !defined(__GNUC__)
#pragma region powersavinghint
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_CONTEXT_POWER_SAVING_HINT_EXP_NAME
/// @brief Power Saving Hint Extension Name
#define ZE_CONTEXT_POWER_SAVING_HINT_EXP_NAME  "ZE_experimental_power_saving_hint"
#endif // ZE_CONTEXT_POWER_SAVING_HINT_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Power Saving Hint Extension Version(s)
typedef enum _ze_power_saving_hint_exp_version_t
{
    ZE_POWER_SAVING_HINT_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),         ///< version 1.0
    ZE_POWER_SAVING_HINT_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),     ///< latest known version
    ZE_POWER_SAVING_HINT_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_power_saving_hint_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum _ze_power_saving_hint_type_t
{
    ZE_POWER_SAVING_HINT_TYPE_MIN = 0,                                      ///< Minumum power savings. The device will make no attempt to save power
                                                                            ///< while executing work submitted to this context.
    ZE_POWER_SAVING_HINT_TYPE_MAX = 100,                                    ///< Maximum power savings. The device will do everything to bring power to
                                                                            ///< a minimum while executing work submitted to this context.
    ZE_POWER_SAVING_HINT_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_power_saving_hint_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extended context descriptor containing power saving hint.
typedef struct _ze_context_power_saving_hint_exp_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t hint;                                                          ///< [in] power saving hint (default value = 0). This is value from [0,100]
                                                                            ///< and can use pre-defined settings from ::ze_power_saving_hint_type_t.

} ze_context_power_saving_hint_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Subgroups
#if !defined(__GNUC__)
#pragma region subgroups
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_SUBGROUPS_EXT_NAME
/// @brief Subgroups Extension Name
#define ZE_SUBGROUPS_EXT_NAME  "ZE_extension_subgroups"
#endif // ZE_SUBGROUPS_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Subgroups Extension Version(s)
typedef enum _ze_subgroup_ext_version_t
{
    ZE_SUBGROUP_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                  ///< version 1.0
    ZE_SUBGROUP_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),              ///< latest known version
    ZE_SUBGROUP_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_subgroup_ext_version_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for EU Count
#if !defined(__GNUC__)
#pragma region EUCount
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_EU_COUNT_EXT_NAME
/// @brief EU Count Extension Name
#define ZE_EU_COUNT_EXT_NAME  "ZE_extension_eu_count"
#endif // ZE_EU_COUNT_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief EU Count Extension Version(s)
typedef enum _ze_eu_count_ext_version_t
{
    ZE_EU_COUNT_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                  ///< version 1.0
    ZE_EU_COUNT_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),              ///< latest known version
    ZE_EU_COUNT_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_eu_count_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief EU count queried using ::zeDeviceGetProperties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetProperties via the
///       `pNext` member of ::ze_device_properties_t.
///     - Used for determining the total number of EUs available on device.
typedef struct _ze_eu_count_ext_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t numTotalEUs;                                                   ///< [out] Total number of EUs available

} ze_eu_count_ext_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for PCI Properties
#if !defined(__GNUC__)
#pragma region PCIProperties
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_PCI_PROPERTIES_EXT_NAME
/// @brief PCI Properties Extension Name
#define ZE_PCI_PROPERTIES_EXT_NAME  "ZE_extension_pci_properties"
#endif // ZE_PCI_PROPERTIES_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI Properties Extension Version(s)
typedef enum _ze_pci_properties_ext_version_t
{
    ZE_PCI_PROPERTIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),            ///< version 1.0
    ZE_PCI_PROPERTIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),        ///< latest known version
    ZE_PCI_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_pci_properties_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device PCI address
/// 
/// @details
///     - This structure may be passed to ::zeDevicePciGetPropertiesExt as an
///       attribute of ::ze_pci_ext_properties_t.
///     - A PCI BDF address is the bus:device:function address of the device and
///       is useful for locating the device in the PCI switch fabric.
typedef struct _ze_pci_address_ext_t
{
    uint32_t domain;                                                        ///< [out] PCI domain number
    uint32_t bus;                                                           ///< [out] PCI BDF bus number
    uint32_t device;                                                        ///< [out] PCI BDF device number
    uint32_t function;                                                      ///< [out] PCI BDF function number

} ze_pci_address_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device PCI speed
typedef struct _ze_pci_speed_ext_t
{
    int32_t genVersion;                                                     ///< [out] The link generation. A value of -1 means that this property is
                                                                            ///< unknown.
    int32_t width;                                                          ///< [out] The number of lanes. A value of -1 means that this property is
                                                                            ///< unknown.
    int64_t maxBandwidth;                                                   ///< [out] The theoretical maximum bandwidth in bytes/sec (sum of all
                                                                            ///< lanes). A value of -1 means that this property is unknown.

} ze_pci_speed_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Static PCI properties
typedef struct _ze_pci_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_pci_address_ext_t address;                                           ///< [out] The BDF address
    ze_pci_speed_ext_t maxSpeed;                                            ///< [out] Fastest port configuration supported by the device (sum of all
                                                                            ///< lanes)

} ze_pci_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get PCI properties - address, max speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - None
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pPciProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDevicePciGetPropertiesExt(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device object.
    ze_pci_ext_properties_t* pPciProperties                                 ///< [in,out] returns the PCI properties of the device.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for sRGB
#if !defined(__GNUC__)
#pragma region SRGB
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_SRGB_EXT_NAME
/// @brief sRGB Extension Name
#define ZE_SRGB_EXT_NAME  "ZE_extension_srgb"
#endif // ZE_SRGB_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief sRGB Extension Version(s)
typedef enum _ze_srgb_ext_version_t
{
    ZE_SRGB_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                      ///< version 1.0
    ZE_SRGB_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),                  ///< latest known version
    ZE_SRGB_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_srgb_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief sRGB image descriptor
/// 
/// @details
///     - This structure may be passed to ::zeImageCreate via the `pNext` member
///       of ::ze_image_desc_t
///     - Used for specifying that the image is in sRGB format.
typedef struct _ze_srgb_ext_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_bool_t sRGB;                                                         ///< [in] Is sRGB.

} ze_srgb_ext_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Image Copy To/From Memory
#if !defined(__GNUC__)
#pragma region imageCopy
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_IMAGE_COPY_EXT_NAME
/// @brief Image Copy Extension Name
#define ZE_IMAGE_COPY_EXT_NAME  "ZE_extension_image_copy"
#endif // ZE_IMAGE_COPY_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Image Copy Extension Version(s)
typedef enum _ze_image_copy_ext_version_t
{
    ZE_IMAGE_COPY_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                ///< version 1.0
    ZE_IMAGE_COPY_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),            ///< latest known version
    ZE_IMAGE_COPY_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_copy_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies from an image to device or shared memory.
/// 
/// @details
///     - The application must ensure the memory pointed to by dstptr is
///       accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by dstptr as
///       it is free to be modified by either the Host or device up until
///       execution.
///     - The application must ensure the image and events are accessible by the
///       device on which the command list was created.
///     - The application must ensure the image format descriptor for the source
///       image is a single-planar format.
///     - The application must ensure that the rowPitch is set to 0 if image is
///       a 1D image. Otherwise the rowPitch must be greater than or equal to
///       the element size in bytes x width.
///     - If rowPitch is set to 0, the appropriate row pitch is calculated based
///       on the size of each element in bytes multiplied by width
///     - The application must ensure that the slicePitch is set to 0 if image
///       is a 1D or 2D image. Otherwise this value must be greater than or
///       equal to rowPitch x height.
///     - If slicePitch is set to 0, the appropriate slice pitch is calculated
///       based on the rowPitch x height.
///     - The application must ensure the command list, image and events were
///       created, and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clEnqueueReadImage
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hSrcImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyToMemoryExt(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    void* dstptr,                                                           ///< [in] pointer to destination memory to copy to
    ze_image_handle_t hSrcImage,                                            ///< [in] handle of source image to copy from
    const ze_image_region_t* pSrcRegion,                                    ///< [in][optional] source region descriptor
    uint32_t destRowPitch,                                                  ///< [in] size in bytes of the 1D slice of the 2D region of a 2D or 3D
                                                                            ///< image or each image of a 1D or 2D image array being written
    uint32_t destSlicePitch,                                                ///< [in] size in bytes of the 2D slice of the 3D region of a 3D image or
                                                                            ///< each image of a 1D or 2D image array being written
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Copies to an image from device or shared memory.
/// 
/// @details
///     - The application must ensure the memory pointed to by srcptr is
///       accessible by the device on which the command list was created.
///     - The implementation must not access the memory pointed to by srcptr as
///       it is free to be modified by either the Host or device up until
///       execution.
///     - The application must ensure the image and events are accessible by the
///       device on which the command list was created.
///     - The application must ensure the image format descriptor for the
///       destination image is a single-planar format.
///     - The application must ensure that the rowPitch is set to 0 if image is
///       a 1D image. Otherwise the rowPitch must be greater than or equal to
///       the element size in bytes x width.
///     - If rowPitch is set to 0, the appropriate row pitch is calculated based
///       on the size of each element in bytes multiplied by width
///     - The application must ensure that the slicePitch is set to 0 if image
///       is a 1D or 2D image. Otherwise this value must be greater than or
///       equal to rowPitch x height.
///     - If slicePitch is set to 0, the appropriate slice pitch is calculated
///       based on the rowPitch x height.
///     - The application must ensure the command list, image and events were
///       created, and the memory was allocated, on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - clEnqueueWriteImage
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hDstImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == srcptr`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyFromMemoryExt(
    ze_command_list_handle_t hCommandList,                                  ///< [in] handle of command list
    ze_image_handle_t hDstImage,                                            ///< [in] handle of destination image to copy to
    const void* srcptr,                                                     ///< [in] pointer to source memory to copy from
    const ze_image_region_t* pDstRegion,                                    ///< [in][optional] destination region descriptor
    uint32_t srcRowPitch,                                                   ///< [in] size in bytes of the 1D slice of the 2D region of a 2D or 3D
                                                                            ///< image or each image of a 1D or 2D image array being read
    uint32_t srcSlicePitch,                                                 ///< [in] size in bytes of the 2D slice of the 3D region of a 3D image or
                                                                            ///< each image of a 1D or 2D image array being read
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in][optional] number of events to wait on before launching; must be 0
                                                                            ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                                         ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                                            ///< on before launching
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for Querying Image Allocation Properties.
#if !defined(__GNUC__)
#pragma region imageQueryAllocProperties
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_NAME
/// @brief Image Query Allocation Properties Extension Name
#define ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_NAME  "ZE_extension_image_query_alloc_properties"
#endif // ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Image Query Allocation Properties Extension Version(s)
typedef enum _ze_image_query_alloc_properties_ext_version_t
{
    ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_query_alloc_properties_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image allocation properties queried using
///        ::zeImageGetAllocPropertiesExt
typedef struct _ze_image_allocation_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t id;                                                            ///< [out] identifier for this allocation

} ze_image_allocation_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves attributes of an image allocation
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hImage`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pImageAllocProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageGetAllocPropertiesExt(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    ze_image_handle_t hImage,                                               ///< [in] handle of image object to query
    ze_image_allocation_ext_properties_t* pImageAllocProperties             ///< [in,out] query result for image allocation properties
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Linkage Inspection
#if !defined(__GNUC__)
#pragma region linkageInspection
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_LINKAGE_INSPECTION_EXT_NAME
/// @brief Linkage Inspection Extension Name
#define ZE_LINKAGE_INSPECTION_EXT_NAME  "ZE_extension_linkage_inspection"
#endif // ZE_LINKAGE_INSPECTION_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Linkage Inspection Extension Version(s)
typedef enum _ze_linkage_inspection_ext_version_t
{
    ZE_LINKAGE_INSPECTION_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),        ///< version 1.0
    ZE_LINKAGE_INSPECTION_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),    ///< latest known version
    ZE_LINKAGE_INSPECTION_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_linkage_inspection_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported module linkage inspection flags
typedef uint32_t ze_linkage_inspection_ext_flags_t;
typedef enum _ze_linkage_inspection_ext_flag_t
{
    ZE_LINKAGE_INSPECTION_EXT_FLAG_IMPORTS = ZE_BIT(0),                     ///< List all imports of modules
    ZE_LINKAGE_INSPECTION_EXT_FLAG_UNRESOLVABLE_IMPORTS = ZE_BIT(1),        ///< List all imports of modules that do not have a corresponding export
    ZE_LINKAGE_INSPECTION_EXT_FLAG_EXPORTS = ZE_BIT(2),                     ///< List all exports of modules
    ZE_LINKAGE_INSPECTION_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_linkage_inspection_ext_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Module linkage inspection descriptor
/// 
/// @details
///     - This structure may be passed to ::zeModuleInspectLinkageExt.
typedef struct _ze_linkage_inspection_ext_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_linkage_inspection_ext_flags_t flags;                                ///< [in] flags specifying module linkage inspection.
                                                                            ///< must be 0 (default) or a valid combination of ::ze_linkage_inspection_ext_flag_t.

} ze_linkage_inspection_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief List Imports & Exports
/// 
/// @details
///     - List all the import & unresolveable import dependencies & exports of a
///       set of modules
/// 
/// @remarks
///   _Analogues_
///     - None
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pInspectDesc`
///         + `nullptr == phModules`
///         + `nullptr == phLog`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7 < pInspectDesc->flags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleInspectLinkageExt(
    ze_linkage_inspection_ext_desc_t* pInspectDesc,                         ///< [in] pointer to linkage inspection descriptor structure.
    uint32_t numModules,                                                    ///< [in] number of modules to be inspected pointed to by phModules.
    ze_module_handle_t* phModules,                                          ///< [in][range(0, numModules)] pointer to an array of modules to be
                                                                            ///< inspected for import dependencies.
    ze_module_build_log_handle_t* phLog                                     ///< [out] pointer to handle of linkage inspection log. Log object will
                                                                            ///< contain separate lists of imports, un-resolvable imports, and exports.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting memory compression hints.
#if !defined(__GNUC__)
#pragma region memoryCompressionHints
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME
/// @brief Memory Compression Hints Extension Name
#define ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME  "ZE_extension_memory_compression_hints"
#endif // ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory Compression Hints Extension Version(s)
typedef enum _ze_memory_compression_hints_ext_version_t
{
    ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_memory_compression_hints_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported memory compression hints flags
typedef uint32_t ze_memory_compression_hints_ext_flags_t;
typedef enum _ze_memory_compression_hints_ext_flag_t
{
    ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED = ZE_BIT(0),            ///< Hint Driver implementation to make allocation compressible
    ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_UNCOMPRESSED = ZE_BIT(1),          ///< Hint Driver implementation to make allocation not compressible
    ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_memory_compression_hints_ext_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Compression hints memory allocation descriptor
/// 
/// @details
///     - This structure may be passed to ::zeMemAllocShared or
///       ::zeMemAllocDevice, via the `pNext` member of
///       ::ze_device_mem_alloc_desc_t.
///     - This structure may be passed to ::zeMemAllocHost, via the `pNext`
///       member of ::ze_host_mem_alloc_desc_t.
///     - This structure may be passed to ::zeImageCreate, via the `pNext`
///       member of ::ze_image_desc_t.
typedef struct _ze_memory_compression_hints_ext_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_memory_compression_hints_ext_flags_t flags;                          ///< [in] flags specifying if allocation should be compressible or not.
                                                                            ///< Must be set to one of the ::ze_memory_compression_hints_ext_flag_t;

} ze_memory_compression_hints_ext_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Memory Free Policies
#if !defined(__GNUC__)
#pragma region memoryFreePolicies
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MEMORY_FREE_POLICIES_EXT_NAME
/// @brief Memory Free Policies Extension Name
#define ZE_MEMORY_FREE_POLICIES_EXT_NAME  "ZE_extension_memory_free_policies"
#endif // ZE_MEMORY_FREE_POLICIES_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory Free Policies Extension Version(s)
typedef enum _ze_memory_free_policies_ext_version_t
{
    ZE_MEMORY_FREE_POLICIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),      ///< version 1.0
    ZE_MEMORY_FREE_POLICIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZE_MEMORY_FREE_POLICIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_memory_free_policies_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported memory free policy capability flags
typedef uint32_t ze_driver_memory_free_policy_ext_flags_t;
typedef enum _ze_driver_memory_free_policy_ext_flag_t
{
    ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE = ZE_BIT(0),        ///< blocks until all commands using the memory are complete before freeing
    ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE = ZE_BIT(1),           ///< schedules the memory to be freed but does not free immediately
    ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_driver_memory_free_policy_ext_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Driver memory free properties queried using ::zeDriverGetProperties
/// 
/// @details
///     - All drivers must support an immediate free policy, which is the
///       default free policy.
///     - This structure may be returned from ::zeDriverGetProperties, via the
///       `pNext` member of ::ze_driver_properties_t.
typedef struct _ze_driver_memory_free_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_driver_memory_free_policy_ext_flags_t freePolicies;                  ///< [out] Supported memory free policies.
                                                                            ///< must be 0 or a combination of ::ze_driver_memory_free_policy_ext_flag_t.

} ze_driver_memory_free_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory free descriptor with free policy
typedef struct _ze_memory_free_ext_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_driver_memory_free_policy_ext_flags_t freePolicy;                    ///< [in] flags specifying the memory free policy.
                                                                            ///< must be 0 (default) or a supported ::ze_driver_memory_free_policy_ext_flag_t;
                                                                            ///< default behavior is to free immediately.

} ze_memory_free_ext_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frees allocated host memory, device memory, or shared memory using the
///        specified free policy.
/// 
/// @details
///     - The memory free policy is specified by the memory free descriptor.
///     - The application must **not** call this function from simultaneous
///       threads with the same pointer.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pMemFreeDesc`
///         + `nullptr == ptr`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < pMemFreeDesc->freePolicy`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemFreeExt(
    ze_context_handle_t hContext,                                           ///< [in] handle of the context object
    const ze_memory_free_ext_desc_t* pMemFreeDesc,                          ///< [in] pointer to memory free descriptor
    void* ptr                                                               ///< [in][release] pointer to memory to free
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Bandwidth
#if !defined(__GNUC__)
#pragma region bandwidth
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_BANDWIDTH_PROPERTIES_EXP_NAME
/// @brief Bandwidth Extension Name
#define ZE_BANDWIDTH_PROPERTIES_EXP_NAME  "ZE_experimental_bandwidth_properties"
#endif // ZE_BANDWIDTH_PROPERTIES_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief P2P Bandwidth Properties
/// 
/// @details
///     - This structure may be passed to ::zeDeviceGetP2PProperties by having
///       the pNext member of ::ze_device_p2p_properties_t point at this struct.
typedef struct _ze_device_p2p_bandwidth_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t logicalBandwidth;                                              ///< [out] total logical design bandwidth for all links connecting the two
                                                                            ///< devices
    uint32_t physicalBandwidth;                                             ///< [out] total physical design bandwidth for all links connecting the two
                                                                            ///< devices
    ze_bandwidth_unit_t bandwidthUnit;                                      ///< [out] bandwidth unit
    uint32_t logicalLatency;                                                ///< [out] average logical design latency for all links connecting the two
                                                                            ///< devices
    uint32_t physicalLatency;                                               ///< [out] average physical design latency for all links connecting the two
                                                                            ///< devices
    ze_latency_unit_t latencyUnit;                                          ///< [out] latency unit

} ze_device_p2p_bandwidth_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Copy Bandwidth Properties
/// 
/// @details
///     - This structure may be passed to
///       ::zeDeviceGetCommandQueueGroupProperties by having the pNext member of
///       ::ze_command_queue_group_properties_t point at this struct.
typedef struct _ze_copy_bandwidth_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t copyBandwidth;                                                 ///< [out] design bandwidth supported by this engine type for copy
                                                                            ///< operations
    ze_bandwidth_unit_t copyBandwidthUnit;                                  ///< [out] copy bandwidth unit

} ze_copy_bandwidth_exp_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Device Local Identifier (LUID)
#if !defined(__GNUC__)
#pragma region deviceLUID
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_DEVICE_LUID_EXT_NAME
/// @brief Device Local Identifier (LUID) Extension Name
#define ZE_DEVICE_LUID_EXT_NAME  "ZE_extension_device_luid"
#endif // ZE_DEVICE_LUID_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Device Local Identifier (LUID) Extension Version(s)
typedef enum _ze_device_luid_ext_version_t
{
    ZE_DEVICE_LUID_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),               ///< version 1.0
    ZE_DEVICE_LUID_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),           ///< latest known version
    ZE_DEVICE_LUID_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_device_luid_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_DEVICE_LUID_SIZE_EXT
/// @brief Maximum device local identifier (LUID) size in bytes
#define ZE_MAX_DEVICE_LUID_SIZE_EXT  8
#endif // ZE_MAX_DEVICE_LUID_SIZE_EXT

///////////////////////////////////////////////////////////////////////////////
/// @brief Device local identifier (LUID)
typedef struct _ze_device_luid_ext_t
{
    uint8_t id[ZE_MAX_DEVICE_LUID_SIZE_EXT];                                ///< [out] opaque data representing a device LUID

} ze_device_luid_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device LUID properties queried using ::zeDeviceGetProperties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetProperties, via the
///       `pNext` member of ::ze_device_properties_t.
typedef struct _ze_device_luid_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_luid_ext_t luid;                                              ///< [out] locally unique identifier (LUID).
                                                                            ///< The returned LUID can be cast to a LUID object and must be equal to
                                                                            ///< the locally
                                                                            ///< unique identifier of an IDXGIAdapter1 object that corresponds to the device.
    uint32_t nodeMask;                                                      ///< [out] node mask.
                                                                            ///< The returned node mask must contain exactly one bit.
                                                                            ///< If the device is running on an operating system that supports the
                                                                            ///< Direct3D 12 API
                                                                            ///< and the device corresponds to an individual device in a linked device
                                                                            ///< adapter, the
                                                                            ///< returned node mask identifies the Direct3D 12 node corresponding to
                                                                            ///< the device.
                                                                            ///< Otherwise, the returned node mask must be 1.

} ze_device_luid_ext_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Fabric Topology Discovery
#if !defined(__GNUC__)
#pragma region fabric
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_FABRIC_EXP_NAME
/// @brief Fabric Topology Discovery Extension Name
#define ZE_FABRIC_EXP_NAME  "ZE_experimental_fabric"
#endif // ZE_FABRIC_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_FABRIC_EDGE_MODEL_EXP_SIZE
/// @brief Maximum fabric edge model string size
#define ZE_MAX_FABRIC_EDGE_MODEL_EXP_SIZE  256
#endif // ZE_MAX_FABRIC_EDGE_MODEL_EXP_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric Vertex types
typedef enum _ze_fabric_vertex_exp_type_t
{
    ZE_FABRIC_VERTEX_EXP_TYPE_UNKNOWN = 0,                                  ///< Fabric vertex type is unknown
    ZE_FABRIC_VERTEX_EXP_TYPE_DEVICE = 1,                                   ///< Fabric vertex represents a device
    ZE_FABRIC_VERTEX_EXP_TYPE_SUBDEVICE = 2,                                ///< Fabric vertex represents a subdevice
    ZE_FABRIC_VERTEX_EXP_TYPE_SWITCH = 3,                                   ///< Fabric vertex represents a switch
    ZE_FABRIC_VERTEX_EXP_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_fabric_vertex_exp_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric edge duplexity
typedef enum _ze_fabric_edge_exp_duplexity_t
{
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_UNKNOWN = 0,                               ///< Fabric edge duplexity is unknown
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_HALF_DUPLEX = 1,                           ///< Fabric edge is half duplex, i.e. stated bandwidth is obtained in only
                                                                            ///< one direction at time
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_FULL_DUPLEX = 2,                           ///< Fabric edge is full duplex, i.e. stated bandwidth is supported in both
                                                                            ///< directions simultaneously
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_FORCE_UINT32 = 0x7fffffff

} ze_fabric_edge_exp_duplexity_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI address
/// 
/// @details
///     - A PCI BDF address is the bus:device:function address of the device and
///       is useful for locating the device in the PCI switch fabric.
typedef struct _ze_fabric_vertex_pci_exp_address_t
{
    uint32_t domain;                                                        ///< [out] PCI domain number
    uint32_t bus;                                                           ///< [out] PCI BDF bus number
    uint32_t device;                                                        ///< [out] PCI BDF device number
    uint32_t function;                                                      ///< [out] PCI BDF function number

} ze_fabric_vertex_pci_exp_address_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric Vertex properties
typedef struct _ze_fabric_vertex_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_uuid_t uuid;                                                         ///< [out] universal unique identifier. If the vertex is co-located with a
                                                                            ///< device/subdevice, then this uuid will match that of the corresponding
                                                                            ///< device/subdevice
    ze_fabric_vertex_exp_type_t type;                                       ///< [out] does the fabric vertex represent a device, subdevice, or switch?
    ze_bool_t remote;                                                       ///< [out] does the fabric vertex live on the local node or on a remote
                                                                            ///< node?
    ze_fabric_vertex_pci_exp_address_t address;                             ///< [out] B/D/F address of fabric vertex & associated device/subdevice if
                                                                            ///< available

} ze_fabric_vertex_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric Edge properties
typedef struct _ze_fabric_edge_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_uuid_t uuid;                                                         ///< [out] universal unique identifier.
    char model[ZE_MAX_FABRIC_EDGE_MODEL_EXP_SIZE];                          ///< [out] Description of fabric edge technology. Will be set to the string
                                                                            ///< "unkown" if this cannot be determined for this edge
    uint32_t bandwidth;                                                     ///< [out] design bandwidth
    ze_bandwidth_unit_t bandwidthUnit;                                      ///< [out] bandwidth unit
    uint32_t latency;                                                       ///< [out] design latency
    ze_latency_unit_t latencyUnit;                                          ///< [out] latency unit
    ze_fabric_edge_exp_duplexity_t duplexity;                               ///< [out] Duplexity of the fabric edge

} ze_fabric_edge_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves fabric vertices within a driver
/// 
/// @details
///     - A fabric vertex represents either a device or a switch connected to
///       other fabric vertices.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFabricVertexGetExp(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of fabric vertices.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of fabric vertices available.
                                                                            ///< if count is greater than the number of fabric vertices available, then
                                                                            ///< the driver shall update the value with the correct number of fabric
                                                                            ///< vertices available.
    ze_fabric_vertex_handle_t* phVertices                                   ///< [in,out][optional][range(0, *pCount)] array of handle of fabric vertices.
                                                                            ///< if count is less than the number of fabric vertices available, then
                                                                            ///< driver shall only retrieve that number of fabric vertices.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves a fabric sub-vertex from a fabric vertex
/// 
/// @details
///     - Multiple calls to this function will return identical fabric vertex
///       handles, in the same order.
///     - The number of handles returned from this function is affected by the
///       ::ZE_AFFINITY_MASK environment variable.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hVertex`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFabricVertexGetSubVerticesExp(
    ze_fabric_vertex_handle_t hVertex,                                      ///< [in] handle of the fabric vertex object
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of sub-vertices.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of sub-vertices available.
                                                                            ///< if count is greater than the number of sub-vertices available, then
                                                                            ///< the driver shall update the value with the correct number of
                                                                            ///< sub-vertices available.
    ze_fabric_vertex_handle_t* phSubvertices                                ///< [in,out][optional][range(0, *pCount)] array of handle of sub-vertices.
                                                                            ///< if count is less than the number of sub-vertices available, then
                                                                            ///< driver shall only retrieve that number of sub-vertices.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves properties of the fabric vertex.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hVertex`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pVertexProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFabricVertexGetPropertiesExp(
    ze_fabric_vertex_handle_t hVertex,                                      ///< [in] handle of the fabric vertex
    ze_fabric_vertex_exp_properties_t* pVertexProperties                    ///< [in,out] query result for fabric vertex properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns device handle from fabric vertex handle.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hVertex`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phDevice`
///     - ::ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE
///         + Provided fabric vertex handle does not correspond to a device or subdevice.
///     - ::ZE_RESULT_EXP_ERROR_REMOTE_DEVICE
///         + Provided fabric vertex handle corresponds to remote device or subdevice.
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFabricVertexGetDeviceExp(
    ze_fabric_vertex_handle_t hVertex,                                      ///< [in] handle of the fabric vertex
    ze_device_handle_t* phDevice                                            ///< [out] device handle corresponding to fabric vertex
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns fabric vertex handle from device handle.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phVertex`
///     - ::ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX
///         + Provided device handle does not correspond to a fabric vertex.
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetFabricVertexExp(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_fabric_vertex_handle_t* phVertex                                     ///< [out] fabric vertex handle corresponding to device
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all fabric edges between provided pair of fabric vertices
/// 
/// @details
///     - A fabric edge represents one or more physical links between two fabric
///       vertices.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hVertexA`
///         + `nullptr == hVertexB`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFabricEdgeGetExp(
    ze_fabric_vertex_handle_t hVertexA,                                     ///< [in] handle of first fabric vertex instance
    ze_fabric_vertex_handle_t hVertexB,                                     ///< [in] handle of second fabric vertex instance
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of fabric edges.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of fabric edges available.
                                                                            ///< if count is greater than the number of fabric edges available, then
                                                                            ///< the driver shall update the value with the correct number of fabric
                                                                            ///< edges available.
    ze_fabric_edge_handle_t* phEdges                                        ///< [in,out][optional][range(0, *pCount)] array of handle of fabric edges.
                                                                            ///< if count is less than the number of fabric edges available, then
                                                                            ///< driver shall only retrieve that number of fabric edges.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves fabric vertices connected by a fabric edge
/// 
/// @details
///     - A fabric vertex represents either a device or a switch connected to
///       other fabric vertices via a fabric edge.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEdge`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phVertexA`
///         + `nullptr == phVertexB`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFabricEdgeGetVerticesExp(
    ze_fabric_edge_handle_t hEdge,                                          ///< [in] handle of the fabric edge instance
    ze_fabric_vertex_handle_t* phVertexA,                                   ///< [out] fabric vertex connected to one end of the given fabric edge.
    ze_fabric_vertex_handle_t* phVertexB                                    ///< [out] fabric vertex connected to other end of the given fabric edge.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves properties of the fabric edge.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEdge`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pEdgeProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFabricEdgeGetPropertiesExp(
    ze_fabric_edge_handle_t hEdge,                                          ///< [in] handle of the fabric edge
    ze_fabric_edge_exp_properties_t* pEdgeProperties                        ///< [in,out] query result for fabric edge properties
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Device Memory Properties
#if !defined(__GNUC__)
#pragma region memoryProperties
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_DEVICE_MEMORY_PROPERTIES_EXT_NAME
/// @brief Device Memory Properties Extension Name
#define ZE_DEVICE_MEMORY_PROPERTIES_EXT_NAME  "ZE_extension_device_memory_properties"
#endif // ZE_DEVICE_MEMORY_PROPERTIES_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Device Memory Properties Extension Version(s)
typedef enum _ze_device_memory_properties_ext_version_t
{
    ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_device_memory_properties_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory module types
typedef enum _ze_device_memory_ext_type_t
{
    ZE_DEVICE_MEMORY_EXT_TYPE_HBM = 0,                                      ///< HBM memory
    ZE_DEVICE_MEMORY_EXT_TYPE_HBM2 = 1,                                     ///< HBM2 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR = 2,                                      ///< DDR memory
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR2 = 3,                                     ///< DDR2 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR3 = 4,                                     ///< DDR3 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR4 = 5,                                     ///< DDR4 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR5 = 6,                                     ///< DDR5 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR = 7,                                    ///< LPDDR memory
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR3 = 8,                                   ///< LPDDR3 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR4 = 9,                                   ///< LPDDR4 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR5 = 10,                                  ///< LPDDR5 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_SRAM = 11,                                    ///< SRAM memory
    ZE_DEVICE_MEMORY_EXT_TYPE_L1 = 12,                                      ///< L1 cache
    ZE_DEVICE_MEMORY_EXT_TYPE_L3 = 13,                                      ///< L3 cache
    ZE_DEVICE_MEMORY_EXT_TYPE_GRF = 14,                                     ///< Execution unit register file
    ZE_DEVICE_MEMORY_EXT_TYPE_SLM = 15,                                     ///< Execution unit shared local memory
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR4 = 16,                                   ///< GDDR4 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5 = 17,                                   ///< GDDR5 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5X = 18,                                  ///< GDDR5X memory
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6 = 19,                                   ///< GDDR6 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6X = 20,                                  ///< GDDR6X memory
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR7 = 21,                                   ///< GDDR7 memory
    ZE_DEVICE_MEMORY_EXT_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_device_memory_ext_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory properties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetMemoryProperties via
///       the `pNext` member of ::ze_device_memory_properties_t
typedef struct _ze_device_memory_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_device_memory_ext_type_t type;                                       ///< [out] The memory type
    uint64_t physicalSize;                                                  ///< [out] Physical memory size in bytes. A value of 0 indicates that this
                                                                            ///< property is not known. However, a call to $sMemoryGetState() will
                                                                            ///< correctly return the total size of usable memory.
    uint32_t readBandwidth;                                                 ///< [out] Design bandwidth for reads
    uint32_t writeBandwidth;                                                ///< [out] Design bandwidth for writes
    ze_bandwidth_unit_t bandwidthUnit;                                      ///< [out] bandwidth unit

} ze_device_memory_ext_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Bfloat16 Conversions
#if !defined(__GNUC__)
#pragma region bfloat16conversions
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_BFLOAT16_CONVERSIONS_EXT_NAME
/// @brief Bfloat16 Conversions Extension Name
#define ZE_BFLOAT16_CONVERSIONS_EXT_NAME  "ZE_extension_bfloat16_conversions"
#endif // ZE_BFLOAT16_CONVERSIONS_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Bfloat16 Conversions Extension Version(s)
typedef enum _ze_bfloat16_conversions_ext_version_t
{
    ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),      ///< version 1.0
    ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_bfloat16_conversions_ext_version_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension APIs for Device IP Version
#if !defined(__GNUC__)
#pragma region deviceipversion
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_DEVICE_IP_VERSION_EXT_NAME
/// @brief Device IP Version Extension Name
#define ZE_DEVICE_IP_VERSION_EXT_NAME  "ZE_extension_device_ip_version"
#endif // ZE_DEVICE_IP_VERSION_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Device IP Version Extension Version(s)
typedef enum _ze_device_ip_version_version_t
{
    ZE_DEVICE_IP_VERSION_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),             ///< version 1.0
    ZE_DEVICE_IP_VERSION_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),         ///< latest known version
    ZE_DEVICE_IP_VERSION_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_device_ip_version_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device IP version queried using ::zeDeviceGetProperties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetProperties via the
///       `pNext` member of ::ze_device_properties_t
typedef struct _ze_device_ip_version_ext_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t ipVersion;                                                     ///< [out] Device IP version. The meaning of the device IP version is
                                                                            ///< implementation-defined, but newer devices should have a higher
                                                                            ///< version than older devices.

} ze_device_ip_version_ext_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for querying kernel max group size properties.
#if !defined(__GNUC__)
#pragma region kernelMaxGroupSizeProperties
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_NAME
/// @brief Kernel Max Group Size Properties Extension Name
#define ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_NAME  "ZE_extension_kernel_max_group_size_properties"
#endif // ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel Max Group Size Properties Extension Version(s)
typedef enum _ze_kernel_max_group_size_properties_ext_version_t
{
    ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_kernel_max_group_size_properties_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Additional kernel max group size properties
/// 
/// @details
///     - This structure may be passed to ::zeKernelGetProperties, via the
///       `pNext` member of ::ze_kernel_properties_t, to query additional kernel
///       max group size properties.
typedef struct _ze_kernel_max_group_size_properties_ext_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t maxGroupSize;                                                  ///< [out] maximum group size that can be used to execute the kernel. This
                                                                            ///< value may be less than or equal to the `maxTotalGroupSize` member of
                                                                            ///< ::ze_device_compute_properties_t.

} ze_kernel_max_group_size_properties_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief compiler-independent type
typedef ze_kernel_max_group_size_properties_ext_t ze_kernel_max_group_size_ext_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for querying sub-allocations properties.
#if !defined(__GNUC__)
#pragma region subAllocationsProperties
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_SUB_ALLOCATIONS_EXP_NAME
/// @brief Sub-Allocations Properties Extension Name
#define ZE_SUB_ALLOCATIONS_EXP_NAME  "ZE_experimental_sub_allocations"
#endif // ZE_SUB_ALLOCATIONS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Sub-Allocations Properties Extension Version(s)
typedef enum _ze_sub_allocations_exp_version_t
{
    ZE_SUB_ALLOCATIONS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),           ///< version 1.0
    ZE_SUB_ALLOCATIONS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),       ///< latest known version
    ZE_SUB_ALLOCATIONS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_sub_allocations_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties returned for a sub-allocation
typedef struct _ze_sub_allocation_t
{
    void* base;                                                             ///< [in,out][optional] base address of the sub-allocation
    size_t size;                                                            ///< [in,out][optional] size of the allocation

} ze_sub_allocation_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sub-Allocations Properties
/// 
/// @details
///     - This structure may be passed to ::zeMemGetAllocProperties, via the
///       `pNext` member of ::ze_memory_allocation_properties_t.
typedef struct _ze_memory_sub_allocations_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t* pCount;                                                       ///< [in,out] pointer to the number of sub-allocations.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of sub-allocations on which the allocation has been divided.
                                                                            ///< if count is greater than the number of sub-allocations, then the
                                                                            ///< driver shall update the value with the correct number of sub-allocations.
    ze_sub_allocation_t* pSubAllocations;                                   ///< [in,out][optional][range(0, *pCount)] array of properties for sub-allocations.
                                                                            ///< if count is less than the number of sub-allocations available, then
                                                                            ///< driver shall only retrieve properties for that number of sub-allocations.

} ze_memory_sub_allocations_exp_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting the querying of synchronized event timestamps.
#if !defined(__GNUC__)
#pragma region eventQueryKernelTimestamps
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_NAME
/// @brief Event Query Kernel Timestamps Extension Name
#define ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_NAME  "ZE_extension_event_query_kernel_timestamps"
#endif // ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Event Query Kernel Timestamps Extension Version(s)
typedef enum _ze_event_query_kernel_timestamps_ext_version_t
{
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), ///< version 1.0
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), ///< latest known version
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_event_query_kernel_timestamps_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event query kernel timestamps flags
typedef uint32_t ze_event_query_kernel_timestamps_ext_flags_t;
typedef enum _ze_event_query_kernel_timestamps_ext_flag_t
{
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_KERNEL = ZE_BIT(0),           ///< Kernel timestamp results
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_SYNCHRONIZED = ZE_BIT(1),     ///< Device event timestamps synchronized to the host time domain
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_event_query_kernel_timestamps_ext_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event query kernel timestamps properties
/// 
/// @details
///     - This structure may be returned from ::zeDeviceGetProperties, via the
///       `pNext` member of ::ze_device_properties_t.
typedef struct _ze_event_query_kernel_timestamps_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_event_query_kernel_timestamps_ext_flags_t flags;                     ///< [out] 0 or some combination of
                                                                            ///< ::ze_event_query_kernel_timestamps_ext_flag_t flags

} ze_event_query_kernel_timestamps_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel timestamp clock data synchronized to the host time domain
typedef struct _ze_synchronized_timestamp_data_ext_t
{
    uint64_t kernelStart;                                                   ///< [out] synchronized clock at start of kernel execution
    uint64_t kernelEnd;                                                     ///< [out] synchronized clock at end of kernel execution

} ze_synchronized_timestamp_data_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Synchronized kernel timestamp result
typedef struct _ze_synchronized_timestamp_result_ext_t
{
    ze_synchronized_timestamp_data_ext_t global;                            ///< [out] wall-clock data
    ze_synchronized_timestamp_data_ext_t context;                           ///< [out] context-active data; only includes clocks while device context
                                                                            ///< was actively executing.

} ze_synchronized_timestamp_result_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event query kernel timestamps results properties
typedef struct _ze_event_query_kernel_timestamps_results_ext_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_kernel_timestamp_result_t* pKernelTimestampsBuffer;                  ///< [in,out][optional][range(0, *pCount)] pointer to destination buffer of
                                                                            ///< kernel timestamp results
    ze_synchronized_timestamp_result_ext_t* pSynchronizedTimestampsBuffer;  ///< [in,out][optional][range(0, *pCount)] pointer to destination buffer of
                                                                            ///< synchronized timestamp results

} ze_event_query_kernel_timestamps_results_ext_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query an event's timestamp value on the host, with domain preference.
/// 
/// @details
///     - For collecting *only* kernel timestamps, the application must ensure
///       the event was created from an event pool that was created using
///       ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP flag.
///     - For collecting synchronized timestamps, the application must ensure
///       the event was created from an event pool that was created using
///       ::ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP flag. Kernel timestamps
///       are also available from this type of event pool, but there is a
///       performance cost.
///     - The destination memory will be unmodified if the event has not been
///       signaled.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///     - The implementation must support
///       ::ZE_extension_event_query_kernel_timestamps.
///     - The implementation must return all timestamps for the specified event
///       and device pair.
///     - The implementation must return all timestamps for all sub-devices when
///       device handle is parent device.
///     - The implementation may return all timestamps for sub-devices when
///       device handle is sub-device or may return 0 for count.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventQueryKernelTimestampsExt(
    ze_event_handle_t hEvent,                                               ///< [in] handle of the event
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device to query
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of event packets available.
                                                                            ///<    - This value is implementation specific.
                                                                            ///<    - if `*pCount` is zero, then the driver shall update the value with
                                                                            ///< the total number of event packets available.
                                                                            ///<    - if `*pCount` is greater than the number of event packets
                                                                            ///< available, the driver shall update the value with the correct value.
                                                                            ///<    - Buffer(s) for query results must be sized by the application to
                                                                            ///< accommodate a minimum of `*pCount` elements.
    ze_event_query_kernel_timestamps_results_ext_properties_t* pResults     ///< [in,out][optional][range(0, *pCount)] pointer to event query
                                                                            ///< properties structure(s).
                                                                            ///<    - This parameter may be null when `*pCount` is zero.
                                                                            ///<    - if `*pCount` is less than the number of event packets available,
                                                                            ///< the driver may only update `*pCount` elements, starting at element zero.
                                                                            ///<    - if `*pCount` is greater than the number of event packets
                                                                            ///< available, the driver may only update the valid elements.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Extension for supporting ray tracing acceleration structure builder.
#if !defined(__GNUC__)
#pragma region RTASBuilder
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_RTAS_BUILDER_EXP_NAME
/// @brief Ray Tracing Acceleration Structure Builder Extension Name
#define ZE_RTAS_BUILDER_EXP_NAME  "ZE_experimental_rtas_builder"
#endif // ZE_RTAS_BUILDER_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray Tracing Acceleration Structure Builder Extension Version(s)
typedef enum _ze_rtas_builder_exp_version_t
{
    ZE_RTAS_BUILDER_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),              ///< version 1.0
    ZE_RTAS_BUILDER_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),          ///< latest known version
    ZE_RTAS_BUILDER_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure device flags
typedef uint32_t ze_rtas_device_exp_flags_t;
typedef enum _ze_rtas_device_exp_flag_t
{
    ZE_RTAS_DEVICE_EXP_FLAG_RESERVED = ZE_BIT(0),                           ///< reserved for future use
    ZE_RTAS_DEVICE_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_rtas_device_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure format
/// 
/// @details
///     - This is an opaque ray tracing acceleration structure format
///       identifier.
typedef enum _ze_rtas_format_exp_t
{
    ZE_RTAS_FORMAT_EXP_INVALID = 0,                                         ///< Invalid acceleration structure format
    ZE_RTAS_FORMAT_EXP_FORCE_UINT32 = 0x7fffffff

} ze_rtas_format_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder flags
typedef uint32_t ze_rtas_builder_exp_flags_t;
typedef enum _ze_rtas_builder_exp_flag_t
{
    ZE_RTAS_BUILDER_EXP_FLAG_RESERVED = ZE_BIT(0),                          ///< Reserved for future use
    ZE_RTAS_BUILDER_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder parallel operation flags
typedef uint32_t ze_rtas_parallel_operation_exp_flags_t;
typedef enum _ze_rtas_parallel_operation_exp_flag_t
{
    ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_RESERVED = ZE_BIT(0),               ///< Reserved for future use
    ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_rtas_parallel_operation_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder geometry flags
typedef uint32_t ze_rtas_builder_geometry_exp_flags_t;
typedef enum _ze_rtas_builder_geometry_exp_flag_t
{
    ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_NON_OPAQUE = ZE_BIT(0),               ///< non-opaque geometries invoke an any-hit shader
    ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_geometry_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Packed ray tracing acceleration structure builder geometry flags (see
///        ::ze_rtas_builder_geometry_exp_flags_t)
typedef uint8_t ze_rtas_builder_packed_geometry_exp_flags_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder instance flags
typedef uint32_t ze_rtas_builder_instance_exp_flags_t;
typedef enum _ze_rtas_builder_instance_exp_flag_t
{
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_CULL_DISABLE = ZE_BIT(0),    ///< disables culling of front-facing and back-facing triangles
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE = ZE_BIT(1),  ///< reverses front and back face of triangles
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_OPAQUE = ZE_BIT(2),    ///< forces instanced geometry to be opaque, unless ray flag forces it to
                                                                            ///< be non-opaque
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_NON_OPAQUE = ZE_BIT(3),///< forces instanced geometry to be non-opaque, unless ray flag forces it
                                                                            ///< to be opaque
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_instance_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Packed ray tracing acceleration structure builder instance flags (see
///        ::ze_rtas_builder_instance_exp_flags_t)
typedef uint8_t ze_rtas_builder_packed_instance_exp_flags_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder build operation flags
/// 
/// @details
///     - These flags allow the application to tune the acceleration structure
///       build operation.
///     - The acceleration structure builder implementation might choose to use
///       spatial splitting to split large or long primitives into smaller
///       pieces. This may result in any-hit shaders being invoked multiple
///       times for non-opaque primitives, unless
///       ::ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION is specified.
///     - Usage of any of these flags may reduce ray tracing performance.
typedef uint32_t ze_rtas_builder_build_op_exp_flags_t;
typedef enum _ze_rtas_builder_build_op_exp_flag_t
{
    ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_COMPACT = ZE_BIT(0),                  ///< build more compact acceleration structure
    ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION = ZE_BIT(1),   ///< guarantees single any-hit shader invocation per primitive
    ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_build_op_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder build quality hint
/// 
/// @details
///     - Depending on use case different quality modes for acceleration
///       structure build are supported.
///     - A low-quality build builds an acceleration structure fast, but at the
///       cost of some reduction in ray tracing performance. This mode is
///       recommended for dynamic content, such as animated characters.
///     - A medium-quality build uses a compromise between build quality and ray
///       tracing performance. This mode should be used by default.
///     - Higher ray tracing performance can be achieved by using a high-quality
///       build, but acceleration structure build performance might be
///       significantly reduced.
typedef enum _ze_rtas_builder_build_quality_hint_exp_t
{
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_LOW = 0,                         ///< build low-quality acceleration structure (fast)
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_MEDIUM = 1,                      ///< build medium-quality acceleration structure (slower)
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH = 2,                        ///< build high-quality acceleration structure (slow)
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_build_quality_hint_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder geometry type
typedef enum _ze_rtas_builder_geometry_type_exp_t
{
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES = 0,                        ///< triangle mesh geometry type
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS = 1,                            ///< quad mesh geometry type
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL = 2,                       ///< procedural geometry type
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE = 3,                         ///< instance geometry type
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_geometry_type_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Packed ray tracing acceleration structure builder geometry type (see
///        ::ze_rtas_builder_geometry_type_exp_t)
typedef uint8_t ze_rtas_builder_packed_geometry_type_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure data buffer element format
/// 
/// @details
///     - Specifies the format of data buffer elements.
///     - Data buffers may contain instancing transform matrices, triangle/quad
///       vertex indices, etc...
typedef enum _ze_rtas_builder_input_data_format_exp_t
{
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3 = 0,                       ///< 3-component float vector (see ::ze_rtas_float3_exp_t)
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_COLUMN_MAJOR = 1,        ///< 3x4 affine transformation in column-major format (see
                                                                            ///< ::ze_rtas_transform_float3x4_column_major_exp_t)
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ALIGNED_COLUMN_MAJOR = 2,///< 3x4 affine transformation in column-major format (see
                                                                            ///< ::ze_rtas_transform_float3x4_aligned_column_major_exp_t)
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ROW_MAJOR = 3,           ///< 3x4 affine transformation in row-major format (see
                                                                            ///< ::ze_rtas_transform_float3x4_row_major_exp_t)
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_AABB = 4,                         ///< 3-dimensional axis-aligned bounding-box (see ::ze_rtas_aabb_exp_t)
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32 = 5,      ///< Unsigned 32-bit triangle indices (see
                                                                            ///< ::ze_rtas_triangle_indices_uint32_exp_t)
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32 = 6,          ///< Unsigned 32-bit quad indices (see ::ze_rtas_quad_indices_uint32_exp_t)
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FORCE_UINT32 = 0x7fffffff

} ze_rtas_builder_input_data_format_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Packed ray tracing acceleration structure data buffer element format
///        (see ::ze_rtas_builder_input_data_format_exp_t)
typedef uint8_t ze_rtas_builder_packed_input_data_format_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of ray tracing acceleration structure builder object
typedef struct _ze_rtas_builder_exp_handle_t *ze_rtas_builder_exp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of ray tracing acceleration structure builder parallel
///        operation object
typedef struct _ze_rtas_parallel_operation_exp_handle_t *ze_rtas_parallel_operation_exp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder descriptor
typedef struct _ze_rtas_builder_exp_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_rtas_builder_exp_version_t builderVersion;                           ///< [in] ray tracing acceleration structure builder version

} ze_rtas_builder_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder properties
typedef struct _ze_rtas_builder_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_rtas_builder_exp_flags_t flags;                                      ///< [out] ray tracing acceleration structure builder flags
    size_t rtasBufferSizeBytesExpected;                                     ///< [out] expected size (in bytes) required for acceleration structure buffer
                                                                            ///<    - When using an acceleration structure buffer of this size, the
                                                                            ///< build is expected to succeed; however, it is possible that the build
                                                                            ///< may fail with ::ZE_RESULT_EXP_RTAS_BUILD_RETRY
    size_t rtasBufferSizeBytesMaxRequired;                                  ///< [out] worst-case size (in bytes) required for acceleration structure buffer
                                                                            ///<    - When using an acceleration structure buffer of this size, the
                                                                            ///< build is guaranteed to not run out of memory.
    size_t scratchBufferSizeBytes;                                          ///< [out] scratch buffer size (in bytes) required for acceleration
                                                                            ///< structure build.

} ze_rtas_builder_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder parallel operation
///        properties
typedef struct _ze_rtas_parallel_operation_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_rtas_parallel_operation_exp_flags_t flags;                           ///< [out] ray tracing acceleration structure builder parallel operation
                                                                            ///< flags
    uint32_t maxConcurrency;                                                ///< [out] maximum number of threads that may join the parallel operation

} ze_rtas_parallel_operation_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure device properties
/// 
/// @details
///     - This structure may be passed to ::zeDeviceGetProperties, via `pNext`
///       member of ::ze_device_properties_t.
///     - The implementation shall populate `format` with a value other than
///       ::ZE_RTAS_FORMAT_EXP_INVALID when the device supports ray tracing.
typedef struct _ze_rtas_device_exp_properties_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_rtas_device_exp_flags_t flags;                                       ///< [out] ray tracing acceleration structure device flags
    ze_rtas_format_exp_t rtasFormat;                                        ///< [out] ray tracing acceleration structure format
    uint32_t rtasBufferAlignment;                                           ///< [out] required alignment of acceleration structure buffer

} ze_rtas_device_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief A 3-component vector type
typedef struct _ze_rtas_float3_exp_t
{
    float x;                                                                ///< [in] x-coordinate of float3 vector
    float y;                                                                ///< [in] y-coordinate of float3 vector
    float z;                                                                ///< [in] z-coordinate of float3 vector

} ze_rtas_float3_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3x4 affine transformation in column-major layout
/// 
/// @details
///     - A 3x4 affine transformation in column major layout, consisting of vectors
///          - vx=(vx_x, vx_y, vx_z),
///          - vy=(vy_x, vy_y, vy_z),
///          - vz=(vz_x, vz_y, vz_z), and
///          - p=(p_x, p_y, p_z)
///     - The transformation transforms a point (x, y, z) to: `x*vx + y*vy +
///       z*vz + p`.
typedef struct _ze_rtas_transform_float3x4_column_major_exp_t
{
    float vx_x;                                                             ///< [in] element 0 of column 0 of 3x4 matrix
    float vx_y;                                                             ///< [in] element 1 of column 0 of 3x4 matrix
    float vx_z;                                                             ///< [in] element 2 of column 0 of 3x4 matrix
    float vy_x;                                                             ///< [in] element 0 of column 1 of 3x4 matrix
    float vy_y;                                                             ///< [in] element 1 of column 1 of 3x4 matrix
    float vy_z;                                                             ///< [in] element 2 of column 1 of 3x4 matrix
    float vz_x;                                                             ///< [in] element 0 of column 2 of 3x4 matrix
    float vz_y;                                                             ///< [in] element 1 of column 2 of 3x4 matrix
    float vz_z;                                                             ///< [in] element 2 of column 2 of 3x4 matrix
    float p_x;                                                              ///< [in] element 0 of column 3 of 3x4 matrix
    float p_y;                                                              ///< [in] element 1 of column 3 of 3x4 matrix
    float p_z;                                                              ///< [in] element 2 of column 3 of 3x4 matrix

} ze_rtas_transform_float3x4_column_major_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3x4 affine transformation in column-major layout with aligned column
///        vectors
/// 
/// @details
///     - A 3x4 affine transformation in column major layout, consisting of vectors
///        - vx=(vx_x, vx_y, vx_z),
///        - vy=(vy_x, vy_y, vy_z),
///        - vz=(vz_x, vz_y, vz_z), and
///        - p=(p_x, p_y, p_z)
///     - The transformation transforms a point (x, y, z) to: `x*vx + y*vy +
///       z*vz + p`.
///     - The column vectors are aligned to 16-bytes and pad members are
///       ignored.
typedef struct _ze_rtas_transform_float3x4_aligned_column_major_exp_t
{
    float vx_x;                                                             ///< [in] element 0 of column 0 of 3x4 matrix
    float vx_y;                                                             ///< [in] element 1 of column 0 of 3x4 matrix
    float vx_z;                                                             ///< [in] element 2 of column 0 of 3x4 matrix
    float pad0;                                                             ///< [in] ignored padding
    float vy_x;                                                             ///< [in] element 0 of column 1 of 3x4 matrix
    float vy_y;                                                             ///< [in] element 1 of column 1 of 3x4 matrix
    float vy_z;                                                             ///< [in] element 2 of column 1 of 3x4 matrix
    float pad1;                                                             ///< [in] ignored padding
    float vz_x;                                                             ///< [in] element 0 of column 2 of 3x4 matrix
    float vz_y;                                                             ///< [in] element 1 of column 2 of 3x4 matrix
    float vz_z;                                                             ///< [in] element 2 of column 2 of 3x4 matrix
    float pad2;                                                             ///< [in] ignored padding
    float p_x;                                                              ///< [in] element 0 of column 3 of 3x4 matrix
    float p_y;                                                              ///< [in] element 1 of column 3 of 3x4 matrix
    float p_z;                                                              ///< [in] element 2 of column 3 of 3x4 matrix
    float pad3;                                                             ///< [in] ignored padding

} ze_rtas_transform_float3x4_aligned_column_major_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3x4 affine transformation in row-major layout
/// 
/// @details
///     - A 3x4 affine transformation in row-major layout, consisting of vectors
///          - vx=(vx_x, vx_y, vx_z),
///          - vy=(vy_x, vy_y, vy_z),
///          - vz=(vz_x, vz_y, vz_z), and
///          - p=(p_x, p_y, p_z)
///     - The transformation transforms a point (x, y, z) to: `x*vx + y*vy +
///       z*vz + p`.
typedef struct _ze_rtas_transform_float3x4_row_major_exp_t
{
    float vx_x;                                                             ///< [in] element 0 of row 0 of 3x4 matrix
    float vy_x;                                                             ///< [in] element 1 of row 0 of 3x4 matrix
    float vz_x;                                                             ///< [in] element 2 of row 0 of 3x4 matrix
    float p_x;                                                              ///< [in] element 3 of row 0 of 3x4 matrix
    float vx_y;                                                             ///< [in] element 0 of row 1 of 3x4 matrix
    float vy_y;                                                             ///< [in] element 1 of row 1 of 3x4 matrix
    float vz_y;                                                             ///< [in] element 2 of row 1 of 3x4 matrix
    float p_y;                                                              ///< [in] element 3 of row 1 of 3x4 matrix
    float vx_z;                                                             ///< [in] element 0 of row 2 of 3x4 matrix
    float vy_z;                                                             ///< [in] element 1 of row 2 of 3x4 matrix
    float vz_z;                                                             ///< [in] element 2 of row 2 of 3x4 matrix
    float p_z;                                                              ///< [in] element 3 of row 2 of 3x4 matrix

} ze_rtas_transform_float3x4_row_major_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief A 3-dimensional axis-aligned bounding-box with lower and upper bounds
///        in each dimension
typedef struct _ze_rtas_aabb_exp_t
{
    ze_rtas_float3_exp_t lower;                                             ///< [in] lower bounds of AABB
    ze_rtas_float3_exp_t upper;                                             ///< [in] upper bounds of AABB

} ze_rtas_aabb_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Triangle represented using 3 vertex indices
/// 
/// @details
///     - Represents a triangle using 3 vertex indices that index into a vertex
///       array that needs to be provided together with the index array.
///     - The linear barycentric u/v parametrization of the triangle is defined as:
///          - (u=0, v=0) at v0,
///          - (u=1, v=0) at v1, and
///          - (u=0, v=1) at v2
typedef struct _ze_rtas_triangle_indices_uint32_exp_t
{
    uint32_t v0;                                                            ///< [in] first index pointing to the first triangle vertex in vertex array
    uint32_t v1;                                                            ///< [in] second index pointing to the second triangle vertex in vertex
                                                                            ///< array
    uint32_t v2;                                                            ///< [in] third index pointing to the third triangle vertex in vertex array

} ze_rtas_triangle_indices_uint32_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Quad represented using 4 vertex indices
/// 
/// @details
///     - Represents a quad composed of 4 indices that index into a vertex array
///       that needs to be provided together with the index array.
///     - A quad is a triangle pair represented using 4 vertex indices v0, v1,
///       v2, v3.
///       The first triangle is made out of indices v0, v1, v3 and the second triangle
///       from indices v2, v3, v1. The piecewise linear barycentric u/v parametrization
///       of the quad is defined as:
///          - (u=0, v=0) at v0,
///          - (u=1, v=0) at v1,
///          - (u=0, v=1) at v3, and
///          - (u=1, v=1) at v2
///       This is achieved by correcting the u'/v' coordinates of the second
///       triangle by
///       *u = 1-u'* and *v = 1-v'*, yielding a piecewise linear parametrization.
typedef struct _ze_rtas_quad_indices_uint32_exp_t
{
    uint32_t v0;                                                            ///< [in] first index pointing to the first quad vertex in vertex array
    uint32_t v1;                                                            ///< [in] second index pointing to the second quad vertex in vertex array
    uint32_t v2;                                                            ///< [in] third index pointing to the third quad vertex in vertex array
    uint32_t v3;                                                            ///< [in] fourth index pointing to the fourth quad vertex in vertex array

} ze_rtas_quad_indices_uint32_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder geometry info
typedef struct _ze_rtas_builder_geometry_info_exp_t
{
    ze_rtas_builder_packed_geometry_type_exp_t geometryType;                ///< [in] geometry type

} ze_rtas_builder_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder triangle mesh geometry info
/// 
/// @details
///     - The linear barycentric u/v parametrization of the triangle is defined as:
///          - (u=0, v=0) at v0,
///          - (u=1, v=0) at v1, and
///          - (u=0, v=1) at v2
typedef struct _ze_rtas_builder_triangles_geometry_info_exp_t
{
    ze_rtas_builder_packed_geometry_type_exp_t geometryType;                ///< [in] geometry type, must be
                                                                            ///< ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES
    ze_rtas_builder_packed_geometry_exp_flags_t geometryFlags;              ///< [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                            ///< bits representing the geometry flags for all primitives of this
                                                                            ///< geometry
    uint8_t geometryMask;                                                   ///< [in] 8-bit geometry mask for ray masking
    ze_rtas_builder_packed_input_data_format_exp_t triangleFormat;          ///< [in] format of triangle buffer data, must be
                                                                            ///< ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32
    ze_rtas_builder_packed_input_data_format_exp_t vertexFormat;            ///< [in] format of vertex buffer data, must be
                                                                            ///< ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3
    uint32_t triangleCount;                                                 ///< [in] number of triangles in triangle buffer
    uint32_t vertexCount;                                                   ///< [in] number of vertices in vertex buffer
    uint32_t triangleStride;                                                ///< [in] stride (in bytes) of triangles in triangle buffer
    uint32_t vertexStride;                                                  ///< [in] stride (in bytes) of vertices in vertex buffer
    void* pTriangleBuffer;                                                  ///< [in] pointer to array of triangle indices in specified format
    void* pVertexBuffer;                                                    ///< [in] pointer to array of triangle vertices in specified format

} ze_rtas_builder_triangles_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder quad mesh geometry info
/// 
/// @details
///     - A quad is a triangle pair represented using 4 vertex indices v0, v1,
///       v2, v3.
///       The first triangle is made out of indices v0, v1, v3 and the second triangle
///       from indices v2, v3, v1. The piecewise linear barycentric u/v parametrization
///       of the quad is defined as:
///          - (u=0, v=0) at v0,
///          - (u=1, v=0) at v1,
///          - (u=0, v=1) at v3, and
///          - (u=1, v=1) at v2
///       This is achieved by correcting the u'/v' coordinates of the second
///       triangle by
///       *u = 1-u'* and *v = 1-v'*, yielding a piecewise linear parametrization.
typedef struct _ze_rtas_builder_quads_geometry_info_exp_t
{
    ze_rtas_builder_packed_geometry_type_exp_t geometryType;                ///< [in] geometry type, must be ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS
    ze_rtas_builder_packed_geometry_exp_flags_t geometryFlags;              ///< [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                            ///< bits representing the geometry flags for all primitives of this
                                                                            ///< geometry
    uint8_t geometryMask;                                                   ///< [in] 8-bit geometry mask for ray masking
    ze_rtas_builder_packed_input_data_format_exp_t quadFormat;              ///< [in] format of quad buffer data, must be
                                                                            ///< ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32
    ze_rtas_builder_packed_input_data_format_exp_t vertexFormat;            ///< [in] format of vertex buffer data, must be
                                                                            ///< ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3
    uint32_t quadCount;                                                     ///< [in] number of quads in quad buffer
    uint32_t vertexCount;                                                   ///< [in] number of vertices in vertex buffer
    uint32_t quadStride;                                                    ///< [in] stride (in bytes) of quads in quad buffer
    uint32_t vertexStride;                                                  ///< [in] stride (in bytes) of vertices in vertex buffer
    void* pQuadBuffer;                                                      ///< [in] pointer to array of quad indices in specified format
    void* pVertexBuffer;                                                    ///< [in] pointer to array of quad vertices in specified format

} ze_rtas_builder_quads_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief AABB callback function parameters
typedef struct _ze_rtas_geometry_aabbs_exp_cb_params_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t primID;                                                        ///< [in] first primitive to return bounds for
    uint32_t primIDCount;                                                   ///< [in] number of primitives to return bounds for
    void* pGeomUserPtr;                                                     ///< [in] pointer provided through geometry descriptor
    void* pBuildUserPtr;                                                    ///< [in] pointer provided through ::zeRTASBuilderBuildExp function
    ze_rtas_aabb_exp_t* pBoundsOut;                                         ///< [out] destination buffer to write AABB bounds to

} ze_rtas_geometry_aabbs_exp_cb_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function pointer type to return AABBs for a range of
///        procedural primitives
typedef void (*ze_rtas_geometry_aabbs_cb_exp_t)(
        ze_rtas_geometry_aabbs_exp_cb_params_t* params                          ///< [in] callback function parameters structure
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder procedural primitives
///        geometry info
/// 
/// @details
///     - A host-side bounds callback function is invoked by the acceleration
///       structure builder to query the bounds of procedural primitives on
///       demand. The callback is passed some `pGeomUserPtr` that can point to
///       an application-side representation of the procedural primitives.
///       Further, a second `pBuildUserPtr`, which is set by a parameter to
///       ::zeRTASBuilderBuildExp, is passed to the callback. This allows the
///       build to change the bounds of the procedural geometry, for example, to
///       build a BVH only over a short time range to implement multi-segment
///       motion blur.
typedef struct _ze_rtas_builder_procedural_geometry_info_exp_t
{
    ze_rtas_builder_packed_geometry_type_exp_t geometryType;                ///< [in] geometry type, must be
                                                                            ///< ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL
    ze_rtas_builder_packed_geometry_exp_flags_t geometryFlags;              ///< [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                            ///< bits representing the geometry flags for all primitives of this
                                                                            ///< geometry
    uint8_t geometryMask;                                                   ///< [in] 8-bit geometry mask for ray masking
    uint8_t reserved;                                                       ///< [in] reserved for future use
    uint32_t primCount;                                                     ///< [in] number of primitives in geometry
    ze_rtas_geometry_aabbs_cb_exp_t pfnGetBoundsCb;                         ///< [in] pointer to callback function to get the axis-aligned bounding-box
                                                                            ///< for a range of primitives
    void* pGeomUserPtr;                                                     ///< [in] user data pointer passed to callback

} ze_rtas_builder_procedural_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Ray tracing acceleration structure builder instance geometry info
typedef struct _ze_rtas_builder_instance_geometry_info_exp_t
{
    ze_rtas_builder_packed_geometry_type_exp_t geometryType;                ///< [in] geometry type, must be
                                                                            ///< ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE
    ze_rtas_builder_packed_instance_exp_flags_t instanceFlags;              ///< [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                            ///< bits representing the geometry flags for all primitives of this
                                                                            ///< geometry
    uint8_t geometryMask;                                                   ///< [in] 8-bit geometry mask for ray masking
    ze_rtas_builder_packed_input_data_format_exp_t transformFormat;         ///< [in] format of the specified transformation
    uint32_t instanceUserID;                                                ///< [in] user-specified identifier for the instance
    void* pTransform;                                                       ///< [in] object-to-world instance transformation in specified format
    ze_rtas_aabb_exp_t* pBounds;                                            ///< [in] object-space axis-aligned bounding-box of the instanced
                                                                            ///< acceleration structure
    void* pAccelerationStructure;                                           ///< [in] pointer to acceleration structure to instantiate

} ze_rtas_builder_instance_geometry_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 
typedef struct _ze_rtas_builder_build_op_exp_desc_t
{
    ze_structure_type_t stype;                                              ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    ze_rtas_format_exp_t rtasFormat;                                        ///< [in] ray tracing acceleration structure format
    ze_rtas_builder_build_quality_hint_exp_t buildQuality;                  ///< [in] acceleration structure build quality hint
    ze_rtas_builder_build_op_exp_flags_t buildFlags;                        ///< [in] 0 or some combination of ::ze_rtas_builder_build_op_exp_flag_t
                                                                            ///< flags
    const ze_rtas_builder_geometry_info_exp_t** ppGeometries;               ///< [in][optional][range(0, `numGeometries`)] NULL or a valid array of
                                                                            ///< pointers to geometry infos
    uint32_t numGeometries;                                                 ///< [in] number of geometries in geometry infos array, can be zero when
                                                                            ///< `ppGeometries` is NULL

} ze_rtas_builder_build_op_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a ray tracing acceleration structure builder object
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///     - The implementation must support ::ZE_experimental_rtas_builder
///       extension.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pDescriptor`
///         + `nullptr == phBuilder`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_RTAS_BUILDER_EXP_VERSION_CURRENT < pDescriptor->builderVersion`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASBuilderCreateExp(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of driver object
    const ze_rtas_builder_exp_desc_t* pDescriptor,                          ///< [in] pointer to builder descriptor
    ze_rtas_builder_exp_handle_t* phBuilder                                 ///< [out] handle of builder object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves ray tracing acceleration structure builder properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hBuilder`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pBuildOpDescriptor`
///         + `nullptr == pProperties`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_RTAS_FORMAT_EXP_INVALID < pBuildOpDescriptor->rtasFormat`
///         + `::ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH < pBuildOpDescriptor->buildQuality`
///         + `0x3 < pBuildOpDescriptor->buildFlags`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASBuilderGetBuildPropertiesExp(
    ze_rtas_builder_exp_handle_t hBuilder,                                  ///< [in] handle of builder object
    const ze_rtas_builder_build_op_exp_desc_t* pBuildOpDescriptor,          ///< [in] pointer to build operation descriptor
    ze_rtas_builder_exp_properties_t* pProperties                           ///< [in,out] query result for builder properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Checks ray tracing acceleration structure format compatibility
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_RTAS_FORMAT_EXP_INVALID < rtasFormatA`
///         + `::ZE_RTAS_FORMAT_EXP_INVALID < rtasFormatB`
///     - ::ZE_RESULT_SUCCESS
///         + An acceleration structure built with `rtasFormatA` is compatible with devices that report `rtasFormatB`.
///     - ::ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE
///         + An acceleration structure built with `rtasFormatA` is **not** compatible with devices that report `rtasFormatB`.
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverRTASFormatCompatibilityCheckExp(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of driver object
    ze_rtas_format_exp_t rtasFormatA,                                       ///< [in] operand A
    ze_rtas_format_exp_t rtasFormatB                                        ///< [in] operand B
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Build ray tracing acceleration structure
/// 
/// @details
///     - This function builds an acceleration structure of the scene consisting
///       of the specified geometry information and writes the acceleration
///       structure to the provided destination buffer. All types of geometries
///       can get freely mixed inside a scene.
///     - It is the user's responsibility to manage the acceleration structure
///       buffer allocation, de-allocation, and potential prefetching to the
///       device memory. The required size of the acceleration structure buffer
///       can be queried with the ::zeRTASBuilderGetBuildPropertiesExp function.
///       The acceleration structure buffer must be a shared USM allocation and
///       should be present on the host at build time. The referenced scene data
///       (index- and vertex- buffers) can be standard host allocations, and
///       will not be referenced into by the build acceleration structure.
///     - Before an acceleration structure can be built, the user must allocate
///       the memory for the acceleration structure buffer and scratch buffer
///       using sizes based on a query for the estimated size properties.
///     - When using the "worst-case" size for the acceleration structure
///       buffer, the acceleration structure construction will never fail with ::ZE_RESULT_EXP_RTAS_BUILD_RETRY.
///     - When using the "expected" size for the acceleration structure buffer,
///       the acceleration structure construction may fail with
///       ::ZE_RESULT_EXP_RTAS_BUILD_RETRY. If this happens, the user may resize
///       their acceleration structure buffer using the returned
///       `*pRtasBufferSizeBytes` value, which will be updated with an improved
///       size estimate that will likely result in a successful build.
///     - The acceleration structure construction is run on the host and is
///       synchronous, thus after the function returns with a successful result,
///       the acceleration structure may be used.
///     - All provided data buffers must be host-accessible.
///     - The acceleration structure buffer must be a USM allocation.
///     - A successfully constructed acceleration structure is entirely
///       self-contained. There is no requirement for input data to persist
///       beyond build completion.
///     - A successfully constructed acceleration structure is non-copyable.
///     - Acceleration structure construction may be parallelized by passing a
///       valid handle to a parallel operation object and joining that parallel
///       operation using ::zeRTASParallelOperationJoinExp with user-provided
///       worker threads.
///     - **Additional Notes**
///        - "The geometry infos array, geometry infos, and scratch buffer must
///       all be standard host memory allocations."
///        - "A pointer to a geometry info can be a null pointer, in which case
///       the geometry is treated as empty."
///        - "If no parallel operation handle is provided, the build is run
///       sequentially on the current thread."
///        - "A parallel operation object may only be associated with a single
///       acceleration structure build at a time."
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hBuilder`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pBuildOpDescriptor`
///         + `nullptr == pScratchBuffer`
///         + `nullptr == pRtasBuffer`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZE_RTAS_FORMAT_EXP_INVALID < pBuildOpDescriptor->rtasFormat`
///         + `::ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH < pBuildOpDescriptor->buildQuality`
///         + `0x3 < pBuildOpDescriptor->buildFlags`
///     - ::ZE_RESULT_EXP_RTAS_BUILD_DEFERRED
///         + Acceleration structure build completion is deferred to parallel operation join.
///     - ::ZE_RESULT_EXP_RTAS_BUILD_RETRY
///         + Acceleration structure build failed due to insufficient resources, retry the build operation with a larger acceleration structure buffer allocation.
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + Acceleration structure build failed due to parallel operation object participation in another build operation.
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASBuilderBuildExp(
    ze_rtas_builder_exp_handle_t hBuilder,                                  ///< [in] handle of builder object
    const ze_rtas_builder_build_op_exp_desc_t* pBuildOpDescriptor,          ///< [in] pointer to build operation descriptor
    void* pScratchBuffer,                                                   ///< [in][range(0, `scratchBufferSizeBytes`)] scratch buffer to be used
                                                                            ///< during acceleration structure construction
    size_t scratchBufferSizeBytes,                                          ///< [in] size of scratch buffer, in bytes
    void* pRtasBuffer,                                                      ///< [in] pointer to destination buffer
    size_t rtasBufferSizeBytes,                                             ///< [in] destination buffer size, in bytes
    ze_rtas_parallel_operation_exp_handle_t hParallelOperation,             ///< [in][optional] handle to parallel operation object
    void* pBuildUserPtr,                                                    ///< [in][optional] pointer passed to callbacks
    ze_rtas_aabb_exp_t* pBounds,                                            ///< [in,out][optional] pointer to destination address for acceleration
                                                                            ///< structure bounds
    size_t* pRtasBufferSizeBytes                                            ///< [out][optional] updated acceleration structure size requirement, in
                                                                            ///< bytes
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a ray tracing acceleration structure builder object
/// 
/// @details
///     - The implementation of this function may immediately release any
///       internal Host and Device resources associated with this builder.
///     - The application must **not** call this function from simultaneous
///       threads with the same builder handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hBuilder`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASBuilderDestroyExp(
    ze_rtas_builder_exp_handle_t hBuilder                                   ///< [in][release] handle of builder object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a ray tracing acceleration structure builder parallel
///        operation object
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///     - The implementation must support ::ZE_experimental_rtas_builder
///       extension.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phParallelOperation`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASParallelOperationCreateExp(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of driver object
    ze_rtas_parallel_operation_exp_handle_t* phParallelOperation            ///< [out] handle of parallel operation object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves ray tracing acceleration structure builder parallel
///        operation properties
/// 
/// @details
///     - The application must first bind the parallel operation object to a
///       build operation before it may query the parallel operation properties.
///       In other words, the application must first call
///       ::zeRTASBuilderBuildExp with **hParallelOperation** before calling
///       this function.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hParallelOperation`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASParallelOperationGetPropertiesExp(
    ze_rtas_parallel_operation_exp_handle_t hParallelOperation,             ///< [in] handle of parallel operation object
    ze_rtas_parallel_operation_exp_properties_t* pProperties                ///< [in,out] query result for parallel operation properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Joins a parallel build operation
/// 
/// @details
///     - All worker threads return the same error code for the parallel build
///       operation upon build completion
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hParallelOperation`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASParallelOperationJoinExp(
    ze_rtas_parallel_operation_exp_handle_t hParallelOperation              ///< [in] handle of parallel operation object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a ray tracing acceleration structure builder parallel
///        operation object
/// 
/// @details
///     - The implementation of this function may immediately release any
///       internal Host and Device resources associated with this parallel
///       operation.
///     - The application must **not** call this function from simultaneous
///       threads with the same parallel operation handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hParallelOperation`
ZE_APIEXPORT ze_result_t ZE_APICALL
zeRTASParallelOperationDestroyExp(
    ze_rtas_parallel_operation_exp_handle_t hParallelOperation              ///< [in][release] handle of parallel operation object to destroy
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero API Callbacks
#if !defined(__GNUC__)
#pragma region callbacks
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeInit 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_init_params_t
{
    ze_init_flags_t* pflags;
} ze_init_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeInit 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnInitCb_t)(
    ze_init_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Global callback functions pointers
typedef struct _ze_global_callbacks_t
{
    ze_pfnInitCb_t                                                  pfnInitCb;
} ze_global_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDriverGet 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_driver_get_params_t
{
    uint32_t** ppCount;
    ze_driver_handle_t** pphDrivers;
} ze_driver_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDriverGet 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDriverGetCb_t)(
    ze_driver_get_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDriverGetApiVersion 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_driver_get_api_version_params_t
{
    ze_driver_handle_t* phDriver;
    ze_api_version_t** pversion;
} ze_driver_get_api_version_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDriverGetApiVersion 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDriverGetApiVersionCb_t)(
    ze_driver_get_api_version_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDriverGetProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_driver_get_properties_params_t
{
    ze_driver_handle_t* phDriver;
    ze_driver_properties_t** ppDriverProperties;
} ze_driver_get_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDriverGetProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDriverGetPropertiesCb_t)(
    ze_driver_get_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDriverGetIpcProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_driver_get_ipc_properties_params_t
{
    ze_driver_handle_t* phDriver;
    ze_driver_ipc_properties_t** ppIpcProperties;
} ze_driver_get_ipc_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDriverGetIpcProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDriverGetIpcPropertiesCb_t)(
    ze_driver_get_ipc_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDriverGetExtensionProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_driver_get_extension_properties_params_t
{
    ze_driver_handle_t* phDriver;
    uint32_t** ppCount;
    ze_driver_extension_properties_t** ppExtensionProperties;
} ze_driver_get_extension_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDriverGetExtensionProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDriverGetExtensionPropertiesCb_t)(
    ze_driver_get_extension_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Driver callback functions pointers
typedef struct _ze_driver_callbacks_t
{
    ze_pfnDriverGetCb_t                                             pfnGetCb;
    ze_pfnDriverGetApiVersionCb_t                                   pfnGetApiVersionCb;
    ze_pfnDriverGetPropertiesCb_t                                   pfnGetPropertiesCb;
    ze_pfnDriverGetIpcPropertiesCb_t                                pfnGetIpcPropertiesCb;
    ze_pfnDriverGetExtensionPropertiesCb_t                          pfnGetExtensionPropertiesCb;
} ze_driver_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGet 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_params_t
{
    ze_driver_handle_t* phDriver;
    uint32_t** ppCount;
    ze_device_handle_t** pphDevices;
} ze_device_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGet 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetCb_t)(
    ze_device_get_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetSubDevices 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_sub_devices_params_t
{
    ze_device_handle_t* phDevice;
    uint32_t** ppCount;
    ze_device_handle_t** pphSubdevices;
} ze_device_get_sub_devices_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetSubDevices 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetSubDevicesCb_t)(
    ze_device_get_sub_devices_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_properties_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_properties_t** ppDeviceProperties;
} ze_device_get_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetPropertiesCb_t)(
    ze_device_get_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetComputeProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_compute_properties_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_compute_properties_t** ppComputeProperties;
} ze_device_get_compute_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetComputeProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetComputePropertiesCb_t)(
    ze_device_get_compute_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetModuleProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_module_properties_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_module_properties_t** ppModuleProperties;
} ze_device_get_module_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetModuleProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetModulePropertiesCb_t)(
    ze_device_get_module_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetCommandQueueGroupProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_command_queue_group_properties_params_t
{
    ze_device_handle_t* phDevice;
    uint32_t** ppCount;
    ze_command_queue_group_properties_t** ppCommandQueueGroupProperties;
} ze_device_get_command_queue_group_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetCommandQueueGroupProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t)(
    ze_device_get_command_queue_group_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetMemoryProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_memory_properties_params_t
{
    ze_device_handle_t* phDevice;
    uint32_t** ppCount;
    ze_device_memory_properties_t** ppMemProperties;
} ze_device_get_memory_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetMemoryProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetMemoryPropertiesCb_t)(
    ze_device_get_memory_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetMemoryAccessProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_memory_access_properties_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_memory_access_properties_t** ppMemAccessProperties;
} ze_device_get_memory_access_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetMemoryAccessProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetMemoryAccessPropertiesCb_t)(
    ze_device_get_memory_access_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetCacheProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_cache_properties_params_t
{
    ze_device_handle_t* phDevice;
    uint32_t** ppCount;
    ze_device_cache_properties_t** ppCacheProperties;
} ze_device_get_cache_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetCacheProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetCachePropertiesCb_t)(
    ze_device_get_cache_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetImageProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_image_properties_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_image_properties_t** ppImageProperties;
} ze_device_get_image_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetImageProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetImagePropertiesCb_t)(
    ze_device_get_image_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetExternalMemoryProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_external_memory_properties_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_external_memory_properties_t** ppExternalMemoryProperties;
} ze_device_get_external_memory_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetExternalMemoryProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetExternalMemoryPropertiesCb_t)(
    ze_device_get_external_memory_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetP2PProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_p2_p_properties_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_handle_t* phPeerDevice;
    ze_device_p2p_properties_t** ppP2PProperties;
} ze_device_get_p2_p_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetP2PProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetP2PPropertiesCb_t)(
    ze_device_get_p2_p_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceCanAccessPeer 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_can_access_peer_params_t
{
    ze_device_handle_t* phDevice;
    ze_device_handle_t* phPeerDevice;
    ze_bool_t** pvalue;
} ze_device_can_access_peer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceCanAccessPeer 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceCanAccessPeerCb_t)(
    ze_device_can_access_peer_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeDeviceGetStatus 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_device_get_status_params_t
{
    ze_device_handle_t* phDevice;
} ze_device_get_status_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeDeviceGetStatus 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnDeviceGetStatusCb_t)(
    ze_device_get_status_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Device callback functions pointers
typedef struct _ze_device_callbacks_t
{
    ze_pfnDeviceGetCb_t                                             pfnGetCb;
    ze_pfnDeviceGetSubDevicesCb_t                                   pfnGetSubDevicesCb;
    ze_pfnDeviceGetPropertiesCb_t                                   pfnGetPropertiesCb;
    ze_pfnDeviceGetComputePropertiesCb_t                            pfnGetComputePropertiesCb;
    ze_pfnDeviceGetModulePropertiesCb_t                             pfnGetModulePropertiesCb;
    ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t                  pfnGetCommandQueueGroupPropertiesCb;
    ze_pfnDeviceGetMemoryPropertiesCb_t                             pfnGetMemoryPropertiesCb;
    ze_pfnDeviceGetMemoryAccessPropertiesCb_t                       pfnGetMemoryAccessPropertiesCb;
    ze_pfnDeviceGetCachePropertiesCb_t                              pfnGetCachePropertiesCb;
    ze_pfnDeviceGetImagePropertiesCb_t                              pfnGetImagePropertiesCb;
    ze_pfnDeviceGetExternalMemoryPropertiesCb_t                     pfnGetExternalMemoryPropertiesCb;
    ze_pfnDeviceGetP2PPropertiesCb_t                                pfnGetP2PPropertiesCb;
    ze_pfnDeviceCanAccessPeerCb_t                                   pfnCanAccessPeerCb;
    ze_pfnDeviceGetStatusCb_t                                       pfnGetStatusCb;
} ze_device_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_create_params_t
{
    ze_driver_handle_t* phDriver;
    const ze_context_desc_t** pdesc;
    ze_context_handle_t** pphContext;
} ze_context_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextCreateCb_t)(
    ze_context_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_destroy_params_t
{
    ze_context_handle_t* phContext;
} ze_context_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextDestroyCb_t)(
    ze_context_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextGetStatus 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_get_status_params_t
{
    ze_context_handle_t* phContext;
} ze_context_get_status_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextGetStatus 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextGetStatusCb_t)(
    ze_context_get_status_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextSystemBarrier 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_system_barrier_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
} ze_context_system_barrier_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextSystemBarrier 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextSystemBarrierCb_t)(
    ze_context_system_barrier_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextMakeMemoryResident 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_make_memory_resident_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    void** pptr;
    size_t* psize;
} ze_context_make_memory_resident_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextMakeMemoryResident 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextMakeMemoryResidentCb_t)(
    ze_context_make_memory_resident_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextEvictMemory 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_evict_memory_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    void** pptr;
    size_t* psize;
} ze_context_evict_memory_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextEvictMemory 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextEvictMemoryCb_t)(
    ze_context_evict_memory_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextMakeImageResident 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_make_image_resident_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    ze_image_handle_t* phImage;
} ze_context_make_image_resident_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextMakeImageResident 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextMakeImageResidentCb_t)(
    ze_context_make_image_resident_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeContextEvictImage 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_context_evict_image_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    ze_image_handle_t* phImage;
} ze_context_evict_image_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeContextEvictImage 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnContextEvictImageCb_t)(
    ze_context_evict_image_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Context callback functions pointers
typedef struct _ze_context_callbacks_t
{
    ze_pfnContextCreateCb_t                                         pfnCreateCb;
    ze_pfnContextDestroyCb_t                                        pfnDestroyCb;
    ze_pfnContextGetStatusCb_t                                      pfnGetStatusCb;
    ze_pfnContextSystemBarrierCb_t                                  pfnSystemBarrierCb;
    ze_pfnContextMakeMemoryResidentCb_t                             pfnMakeMemoryResidentCb;
    ze_pfnContextEvictMemoryCb_t                                    pfnEvictMemoryCb;
    ze_pfnContextMakeImageResidentCb_t                              pfnMakeImageResidentCb;
    ze_pfnContextEvictImageCb_t                                     pfnEvictImageCb;
} ze_context_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandQueueCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_queue_create_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    const ze_command_queue_desc_t** pdesc;
    ze_command_queue_handle_t** pphCommandQueue;
} ze_command_queue_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandQueueCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandQueueCreateCb_t)(
    ze_command_queue_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandQueueDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_queue_destroy_params_t
{
    ze_command_queue_handle_t* phCommandQueue;
} ze_command_queue_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandQueueDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandQueueDestroyCb_t)(
    ze_command_queue_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandQueueExecuteCommandLists 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_queue_execute_command_lists_params_t
{
    ze_command_queue_handle_t* phCommandQueue;
    uint32_t* pnumCommandLists;
    ze_command_list_handle_t** pphCommandLists;
    ze_fence_handle_t* phFence;
} ze_command_queue_execute_command_lists_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandQueueExecuteCommandLists 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandQueueExecuteCommandListsCb_t)(
    ze_command_queue_execute_command_lists_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandQueueSynchronize 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_queue_synchronize_params_t
{
    ze_command_queue_handle_t* phCommandQueue;
    uint64_t* ptimeout;
} ze_command_queue_synchronize_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandQueueSynchronize 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandQueueSynchronizeCb_t)(
    ze_command_queue_synchronize_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of CommandQueue callback functions pointers
typedef struct _ze_command_queue_callbacks_t
{
    ze_pfnCommandQueueCreateCb_t                                    pfnCreateCb;
    ze_pfnCommandQueueDestroyCb_t                                   pfnDestroyCb;
    ze_pfnCommandQueueExecuteCommandListsCb_t                       pfnExecuteCommandListsCb;
    ze_pfnCommandQueueSynchronizeCb_t                               pfnSynchronizeCb;
} ze_command_queue_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_create_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    const ze_command_list_desc_t** pdesc;
    ze_command_list_handle_t** pphCommandList;
} ze_command_list_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListCreateCb_t)(
    ze_command_list_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListCreateImmediate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_create_immediate_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    const ze_command_queue_desc_t** paltdesc;
    ze_command_list_handle_t** pphCommandList;
} ze_command_list_create_immediate_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListCreateImmediate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListCreateImmediateCb_t)(
    ze_command_list_create_immediate_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_destroy_params_t
{
    ze_command_list_handle_t* phCommandList;
} ze_command_list_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListDestroyCb_t)(
    ze_command_list_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListClose 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_close_params_t
{
    ze_command_list_handle_t* phCommandList;
} ze_command_list_close_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListClose 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListCloseCb_t)(
    ze_command_list_close_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListReset 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_reset_params_t
{
    ze_command_list_handle_t* phCommandList;
} ze_command_list_reset_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListReset 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListResetCb_t)(
    ze_command_list_reset_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendWriteGlobalTimestamp 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_write_global_timestamp_params_t
{
    ze_command_list_handle_t* phCommandList;
    uint64_t** pdstptr;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_write_global_timestamp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendWriteGlobalTimestamp 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendWriteGlobalTimestampCb_t)(
    ze_command_list_append_write_global_timestamp_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendBarrier 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_barrier_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_barrier_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendBarrier 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendBarrierCb_t)(
    ze_command_list_append_barrier_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendMemoryRangesBarrier 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_memory_ranges_barrier_params_t
{
    ze_command_list_handle_t* phCommandList;
    uint32_t* pnumRanges;
    const size_t** ppRangeSizes;
    const void*** ppRanges;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_ranges_barrier_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendMemoryRangesBarrier 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryRangesBarrierCb_t)(
    ze_command_list_append_memory_ranges_barrier_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendMemoryCopy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_memory_copy_params_t
{
    ze_command_list_handle_t* phCommandList;
    void** pdstptr;
    const void** psrcptr;
    size_t* psize;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendMemoryCopy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyCb_t)(
    ze_command_list_append_memory_copy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendMemoryFill 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_memory_fill_params_t
{
    ze_command_list_handle_t* phCommandList;
    void** pptr;
    const void** ppattern;
    size_t* ppattern_size;
    size_t* psize;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_fill_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendMemoryFill 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryFillCb_t)(
    ze_command_list_append_memory_fill_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendMemoryCopyRegion 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_memory_copy_region_params_t
{
    ze_command_list_handle_t* phCommandList;
    void** pdstptr;
    const ze_copy_region_t** pdstRegion;
    uint32_t* pdstPitch;
    uint32_t* pdstSlicePitch;
    const void** psrcptr;
    const ze_copy_region_t** psrcRegion;
    uint32_t* psrcPitch;
    uint32_t* psrcSlicePitch;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_copy_region_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendMemoryCopyRegion 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyRegionCb_t)(
    ze_command_list_append_memory_copy_region_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendMemoryCopyFromContext 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_memory_copy_from_context_params_t
{
    ze_command_list_handle_t* phCommandList;
    void** pdstptr;
    ze_context_handle_t* phContextSrc;
    const void** psrcptr;
    size_t* psize;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_copy_from_context_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendMemoryCopyFromContext 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyFromContextCb_t)(
    ze_command_list_append_memory_copy_from_context_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendImageCopy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_image_copy_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_image_handle_t* phDstImage;
    ze_image_handle_t* phSrcImage;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendImageCopy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyCb_t)(
    ze_command_list_append_image_copy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendImageCopyRegion 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_image_copy_region_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_image_handle_t* phDstImage;
    ze_image_handle_t* phSrcImage;
    const ze_image_region_t** ppDstRegion;
    const ze_image_region_t** ppSrcRegion;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_region_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendImageCopyRegion 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyRegionCb_t)(
    ze_command_list_append_image_copy_region_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendImageCopyToMemory 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_image_copy_to_memory_params_t
{
    ze_command_list_handle_t* phCommandList;
    void** pdstptr;
    ze_image_handle_t* phSrcImage;
    const ze_image_region_t** ppSrcRegion;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_to_memory_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendImageCopyToMemory 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyToMemoryCb_t)(
    ze_command_list_append_image_copy_to_memory_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendImageCopyFromMemory 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_image_copy_from_memory_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_image_handle_t* phDstImage;
    const void** psrcptr;
    const ze_image_region_t** ppDstRegion;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_from_memory_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendImageCopyFromMemory 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyFromMemoryCb_t)(
    ze_command_list_append_image_copy_from_memory_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendMemoryPrefetch 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_memory_prefetch_params_t
{
    ze_command_list_handle_t* phCommandList;
    const void** pptr;
    size_t* psize;
} ze_command_list_append_memory_prefetch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendMemoryPrefetch 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryPrefetchCb_t)(
    ze_command_list_append_memory_prefetch_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendMemAdvise 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_mem_advise_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_device_handle_t* phDevice;
    const void** pptr;
    size_t* psize;
    ze_memory_advice_t* padvice;
} ze_command_list_append_mem_advise_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendMemAdvise 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendMemAdviseCb_t)(
    ze_command_list_append_mem_advise_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendSignalEvent 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_signal_event_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_event_handle_t* phEvent;
} ze_command_list_append_signal_event_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendSignalEvent 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendSignalEventCb_t)(
    ze_command_list_append_signal_event_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendWaitOnEvents 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_wait_on_events_params_t
{
    ze_command_list_handle_t* phCommandList;
    uint32_t* pnumEvents;
    ze_event_handle_t** pphEvents;
} ze_command_list_append_wait_on_events_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendWaitOnEvents 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendWaitOnEventsCb_t)(
    ze_command_list_append_wait_on_events_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendEventReset 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_event_reset_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_event_handle_t* phEvent;
} ze_command_list_append_event_reset_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendEventReset 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendEventResetCb_t)(
    ze_command_list_append_event_reset_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendQueryKernelTimestamps 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_query_kernel_timestamps_params_t
{
    ze_command_list_handle_t* phCommandList;
    uint32_t* pnumEvents;
    ze_event_handle_t** pphEvents;
    void** pdstptr;
    const size_t** ppOffsets;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_query_kernel_timestamps_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendQueryKernelTimestamps 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendQueryKernelTimestampsCb_t)(
    ze_command_list_append_query_kernel_timestamps_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendLaunchKernel 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_launch_kernel_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_kernel_handle_t* phKernel;
    const ze_group_count_t** ppLaunchFuncArgs;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_kernel_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendLaunchKernel 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchKernelCb_t)(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendLaunchCooperativeKernel 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_launch_cooperative_kernel_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_kernel_handle_t* phKernel;
    const ze_group_count_t** ppLaunchFuncArgs;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_cooperative_kernel_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendLaunchCooperativeKernel 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchCooperativeKernelCb_t)(
    ze_command_list_append_launch_cooperative_kernel_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendLaunchKernelIndirect 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_launch_kernel_indirect_params_t
{
    ze_command_list_handle_t* phCommandList;
    ze_kernel_handle_t* phKernel;
    const ze_group_count_t** ppLaunchArgumentsBuffer;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_kernel_indirect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendLaunchKernelIndirect 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchKernelIndirectCb_t)(
    ze_command_list_append_launch_kernel_indirect_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeCommandListAppendLaunchMultipleKernelsIndirect 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_command_list_append_launch_multiple_kernels_indirect_params_t
{
    ze_command_list_handle_t* phCommandList;
    uint32_t* pnumKernels;
    ze_kernel_handle_t** pphKernels;
    const uint32_t** ppCountBuffer;
    const ze_group_count_t** ppLaunchArgumentsBuffer;
    ze_event_handle_t* phSignalEvent;
    uint32_t* pnumWaitEvents;
    ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_multiple_kernels_indirect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeCommandListAppendLaunchMultipleKernelsIndirect 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t)(
    ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of CommandList callback functions pointers
typedef struct _ze_command_list_callbacks_t
{
    ze_pfnCommandListCreateCb_t                                     pfnCreateCb;
    ze_pfnCommandListCreateImmediateCb_t                            pfnCreateImmediateCb;
    ze_pfnCommandListDestroyCb_t                                    pfnDestroyCb;
    ze_pfnCommandListCloseCb_t                                      pfnCloseCb;
    ze_pfnCommandListResetCb_t                                      pfnResetCb;
    ze_pfnCommandListAppendWriteGlobalTimestampCb_t                 pfnAppendWriteGlobalTimestampCb;
    ze_pfnCommandListAppendBarrierCb_t                              pfnAppendBarrierCb;
    ze_pfnCommandListAppendMemoryRangesBarrierCb_t                  pfnAppendMemoryRangesBarrierCb;
    ze_pfnCommandListAppendMemoryCopyCb_t                           pfnAppendMemoryCopyCb;
    ze_pfnCommandListAppendMemoryFillCb_t                           pfnAppendMemoryFillCb;
    ze_pfnCommandListAppendMemoryCopyRegionCb_t                     pfnAppendMemoryCopyRegionCb;
    ze_pfnCommandListAppendMemoryCopyFromContextCb_t                pfnAppendMemoryCopyFromContextCb;
    ze_pfnCommandListAppendImageCopyCb_t                            pfnAppendImageCopyCb;
    ze_pfnCommandListAppendImageCopyRegionCb_t                      pfnAppendImageCopyRegionCb;
    ze_pfnCommandListAppendImageCopyToMemoryCb_t                    pfnAppendImageCopyToMemoryCb;
    ze_pfnCommandListAppendImageCopyFromMemoryCb_t                  pfnAppendImageCopyFromMemoryCb;
    ze_pfnCommandListAppendMemoryPrefetchCb_t                       pfnAppendMemoryPrefetchCb;
    ze_pfnCommandListAppendMemAdviseCb_t                            pfnAppendMemAdviseCb;
    ze_pfnCommandListAppendSignalEventCb_t                          pfnAppendSignalEventCb;
    ze_pfnCommandListAppendWaitOnEventsCb_t                         pfnAppendWaitOnEventsCb;
    ze_pfnCommandListAppendEventResetCb_t                           pfnAppendEventResetCb;
    ze_pfnCommandListAppendQueryKernelTimestampsCb_t                pfnAppendQueryKernelTimestampsCb;
    ze_pfnCommandListAppendLaunchKernelCb_t                         pfnAppendLaunchKernelCb;
    ze_pfnCommandListAppendLaunchCooperativeKernelCb_t              pfnAppendLaunchCooperativeKernelCb;
    ze_pfnCommandListAppendLaunchKernelIndirectCb_t                 pfnAppendLaunchKernelIndirectCb;
    ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t        pfnAppendLaunchMultipleKernelsIndirectCb;
} ze_command_list_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeImageGetProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_image_get_properties_params_t
{
    ze_device_handle_t* phDevice;
    const ze_image_desc_t** pdesc;
    ze_image_properties_t** ppImageProperties;
} ze_image_get_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeImageGetProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnImageGetPropertiesCb_t)(
    ze_image_get_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeImageCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_image_create_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    const ze_image_desc_t** pdesc;
    ze_image_handle_t** pphImage;
} ze_image_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeImageCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnImageCreateCb_t)(
    ze_image_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeImageDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_image_destroy_params_t
{
    ze_image_handle_t* phImage;
} ze_image_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeImageDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnImageDestroyCb_t)(
    ze_image_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Image callback functions pointers
typedef struct _ze_image_callbacks_t
{
    ze_pfnImageGetPropertiesCb_t                                    pfnGetPropertiesCb;
    ze_pfnImageCreateCb_t                                           pfnCreateCb;
    ze_pfnImageDestroyCb_t                                          pfnDestroyCb;
} ze_image_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeFenceCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_fence_create_params_t
{
    ze_command_queue_handle_t* phCommandQueue;
    const ze_fence_desc_t** pdesc;
    ze_fence_handle_t** pphFence;
} ze_fence_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeFenceCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnFenceCreateCb_t)(
    ze_fence_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeFenceDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_fence_destroy_params_t
{
    ze_fence_handle_t* phFence;
} ze_fence_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeFenceDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnFenceDestroyCb_t)(
    ze_fence_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeFenceHostSynchronize 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_fence_host_synchronize_params_t
{
    ze_fence_handle_t* phFence;
    uint64_t* ptimeout;
} ze_fence_host_synchronize_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeFenceHostSynchronize 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnFenceHostSynchronizeCb_t)(
    ze_fence_host_synchronize_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeFenceQueryStatus 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_fence_query_status_params_t
{
    ze_fence_handle_t* phFence;
} ze_fence_query_status_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeFenceQueryStatus 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnFenceQueryStatusCb_t)(
    ze_fence_query_status_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeFenceReset 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_fence_reset_params_t
{
    ze_fence_handle_t* phFence;
} ze_fence_reset_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeFenceReset 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnFenceResetCb_t)(
    ze_fence_reset_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Fence callback functions pointers
typedef struct _ze_fence_callbacks_t
{
    ze_pfnFenceCreateCb_t                                           pfnCreateCb;
    ze_pfnFenceDestroyCb_t                                          pfnDestroyCb;
    ze_pfnFenceHostSynchronizeCb_t                                  pfnHostSynchronizeCb;
    ze_pfnFenceQueryStatusCb_t                                      pfnQueryStatusCb;
    ze_pfnFenceResetCb_t                                            pfnResetCb;
} ze_fence_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventPoolCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_pool_create_params_t
{
    ze_context_handle_t* phContext;
    const ze_event_pool_desc_t** pdesc;
    uint32_t* pnumDevices;
    ze_device_handle_t** pphDevices;
    ze_event_pool_handle_t** pphEventPool;
} ze_event_pool_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventPoolCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventPoolCreateCb_t)(
    ze_event_pool_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventPoolDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_pool_destroy_params_t
{
    ze_event_pool_handle_t* phEventPool;
} ze_event_pool_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventPoolDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventPoolDestroyCb_t)(
    ze_event_pool_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventPoolGetIpcHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_pool_get_ipc_handle_params_t
{
    ze_event_pool_handle_t* phEventPool;
    ze_ipc_event_pool_handle_t** pphIpc;
} ze_event_pool_get_ipc_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventPoolGetIpcHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventPoolGetIpcHandleCb_t)(
    ze_event_pool_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventPoolOpenIpcHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_pool_open_ipc_handle_params_t
{
    ze_context_handle_t* phContext;
    ze_ipc_event_pool_handle_t* phIpc;
    ze_event_pool_handle_t** pphEventPool;
} ze_event_pool_open_ipc_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventPoolOpenIpcHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventPoolOpenIpcHandleCb_t)(
    ze_event_pool_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventPoolCloseIpcHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_pool_close_ipc_handle_params_t
{
    ze_event_pool_handle_t* phEventPool;
} ze_event_pool_close_ipc_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventPoolCloseIpcHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventPoolCloseIpcHandleCb_t)(
    ze_event_pool_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of EventPool callback functions pointers
typedef struct _ze_event_pool_callbacks_t
{
    ze_pfnEventPoolCreateCb_t                                       pfnCreateCb;
    ze_pfnEventPoolDestroyCb_t                                      pfnDestroyCb;
    ze_pfnEventPoolGetIpcHandleCb_t                                 pfnGetIpcHandleCb;
    ze_pfnEventPoolOpenIpcHandleCb_t                                pfnOpenIpcHandleCb;
    ze_pfnEventPoolCloseIpcHandleCb_t                               pfnCloseIpcHandleCb;
} ze_event_pool_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_create_params_t
{
    ze_event_pool_handle_t* phEventPool;
    const ze_event_desc_t** pdesc;
    ze_event_handle_t** pphEvent;
} ze_event_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventCreateCb_t)(
    ze_event_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_destroy_params_t
{
    ze_event_handle_t* phEvent;
} ze_event_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventDestroyCb_t)(
    ze_event_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventHostSignal 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_host_signal_params_t
{
    ze_event_handle_t* phEvent;
} ze_event_host_signal_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventHostSignal 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventHostSignalCb_t)(
    ze_event_host_signal_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventHostSynchronize 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_host_synchronize_params_t
{
    ze_event_handle_t* phEvent;
    uint64_t* ptimeout;
} ze_event_host_synchronize_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventHostSynchronize 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventHostSynchronizeCb_t)(
    ze_event_host_synchronize_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventQueryStatus 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_query_status_params_t
{
    ze_event_handle_t* phEvent;
} ze_event_query_status_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventQueryStatus 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventQueryStatusCb_t)(
    ze_event_query_status_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventHostReset 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_host_reset_params_t
{
    ze_event_handle_t* phEvent;
} ze_event_host_reset_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventHostReset 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventHostResetCb_t)(
    ze_event_host_reset_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeEventQueryKernelTimestamp 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_event_query_kernel_timestamp_params_t
{
    ze_event_handle_t* phEvent;
    ze_kernel_timestamp_result_t** pdstptr;
} ze_event_query_kernel_timestamp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeEventQueryKernelTimestamp 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnEventQueryKernelTimestampCb_t)(
    ze_event_query_kernel_timestamp_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Event callback functions pointers
typedef struct _ze_event_callbacks_t
{
    ze_pfnEventCreateCb_t                                           pfnCreateCb;
    ze_pfnEventDestroyCb_t                                          pfnDestroyCb;
    ze_pfnEventHostSignalCb_t                                       pfnHostSignalCb;
    ze_pfnEventHostSynchronizeCb_t                                  pfnHostSynchronizeCb;
    ze_pfnEventQueryStatusCb_t                                      pfnQueryStatusCb;
    ze_pfnEventHostResetCb_t                                        pfnHostResetCb;
    ze_pfnEventQueryKernelTimestampCb_t                             pfnQueryKernelTimestampCb;
} ze_event_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_create_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    const ze_module_desc_t** pdesc;
    ze_module_handle_t** pphModule;
    ze_module_build_log_handle_t** pphBuildLog;
} ze_module_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleCreateCb_t)(
    ze_module_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_destroy_params_t
{
    ze_module_handle_t* phModule;
} ze_module_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleDestroyCb_t)(
    ze_module_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleDynamicLink 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_dynamic_link_params_t
{
    uint32_t* pnumModules;
    ze_module_handle_t** pphModules;
    ze_module_build_log_handle_t** pphLinkLog;
} ze_module_dynamic_link_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleDynamicLink 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleDynamicLinkCb_t)(
    ze_module_dynamic_link_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleGetNativeBinary 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_get_native_binary_params_t
{
    ze_module_handle_t* phModule;
    size_t** ppSize;
    uint8_t** ppModuleNativeBinary;
} ze_module_get_native_binary_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleGetNativeBinary 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleGetNativeBinaryCb_t)(
    ze_module_get_native_binary_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleGetGlobalPointer 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_get_global_pointer_params_t
{
    ze_module_handle_t* phModule;
    const char** ppGlobalName;
    size_t** ppSize;
    void*** ppptr;
} ze_module_get_global_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleGetGlobalPointer 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleGetGlobalPointerCb_t)(
    ze_module_get_global_pointer_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleGetKernelNames 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_get_kernel_names_params_t
{
    ze_module_handle_t* phModule;
    uint32_t** ppCount;
    const char*** ppNames;
} ze_module_get_kernel_names_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleGetKernelNames 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleGetKernelNamesCb_t)(
    ze_module_get_kernel_names_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleGetProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_get_properties_params_t
{
    ze_module_handle_t* phModule;
    ze_module_properties_t** ppModuleProperties;
} ze_module_get_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleGetProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleGetPropertiesCb_t)(
    ze_module_get_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleGetFunctionPointer 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_get_function_pointer_params_t
{
    ze_module_handle_t* phModule;
    const char** ppFunctionName;
    void*** ppfnFunction;
} ze_module_get_function_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleGetFunctionPointer 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleGetFunctionPointerCb_t)(
    ze_module_get_function_pointer_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Module callback functions pointers
typedef struct _ze_module_callbacks_t
{
    ze_pfnModuleCreateCb_t                                          pfnCreateCb;
    ze_pfnModuleDestroyCb_t                                         pfnDestroyCb;
    ze_pfnModuleDynamicLinkCb_t                                     pfnDynamicLinkCb;
    ze_pfnModuleGetNativeBinaryCb_t                                 pfnGetNativeBinaryCb;
    ze_pfnModuleGetGlobalPointerCb_t                                pfnGetGlobalPointerCb;
    ze_pfnModuleGetKernelNamesCb_t                                  pfnGetKernelNamesCb;
    ze_pfnModuleGetPropertiesCb_t                                   pfnGetPropertiesCb;
    ze_pfnModuleGetFunctionPointerCb_t                              pfnGetFunctionPointerCb;
} ze_module_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleBuildLogDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_build_log_destroy_params_t
{
    ze_module_build_log_handle_t* phModuleBuildLog;
} ze_module_build_log_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleBuildLogDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleBuildLogDestroyCb_t)(
    ze_module_build_log_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeModuleBuildLogGetString 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_module_build_log_get_string_params_t
{
    ze_module_build_log_handle_t* phModuleBuildLog;
    size_t** ppSize;
    char** ppBuildLog;
} ze_module_build_log_get_string_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeModuleBuildLogGetString 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnModuleBuildLogGetStringCb_t)(
    ze_module_build_log_get_string_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of ModuleBuildLog callback functions pointers
typedef struct _ze_module_build_log_callbacks_t
{
    ze_pfnModuleBuildLogDestroyCb_t                                 pfnDestroyCb;
    ze_pfnModuleBuildLogGetStringCb_t                               pfnGetStringCb;
} ze_module_build_log_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_create_params_t
{
    ze_module_handle_t* phModule;
    const ze_kernel_desc_t** pdesc;
    ze_kernel_handle_t** pphKernel;
} ze_kernel_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelCreateCb_t)(
    ze_kernel_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_destroy_params_t
{
    ze_kernel_handle_t* phKernel;
} ze_kernel_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelDestroyCb_t)(
    ze_kernel_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelSetCacheConfig 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_set_cache_config_params_t
{
    ze_kernel_handle_t* phKernel;
    ze_cache_config_flags_t* pflags;
} ze_kernel_set_cache_config_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelSetCacheConfig 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelSetCacheConfigCb_t)(
    ze_kernel_set_cache_config_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelSetGroupSize 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_set_group_size_params_t
{
    ze_kernel_handle_t* phKernel;
    uint32_t* pgroupSizeX;
    uint32_t* pgroupSizeY;
    uint32_t* pgroupSizeZ;
} ze_kernel_set_group_size_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelSetGroupSize 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelSetGroupSizeCb_t)(
    ze_kernel_set_group_size_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelSuggestGroupSize 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_suggest_group_size_params_t
{
    ze_kernel_handle_t* phKernel;
    uint32_t* pglobalSizeX;
    uint32_t* pglobalSizeY;
    uint32_t* pglobalSizeZ;
    uint32_t** pgroupSizeX;
    uint32_t** pgroupSizeY;
    uint32_t** pgroupSizeZ;
} ze_kernel_suggest_group_size_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelSuggestGroupSize 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelSuggestGroupSizeCb_t)(
    ze_kernel_suggest_group_size_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelSuggestMaxCooperativeGroupCount 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_suggest_max_cooperative_group_count_params_t
{
    ze_kernel_handle_t* phKernel;
    uint32_t** ptotalGroupCount;
} ze_kernel_suggest_max_cooperative_group_count_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelSuggestMaxCooperativeGroupCount 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t)(
    ze_kernel_suggest_max_cooperative_group_count_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelSetArgumentValue 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_set_argument_value_params_t
{
    ze_kernel_handle_t* phKernel;
    uint32_t* pargIndex;
    size_t* pargSize;
    const void** ppArgValue;
} ze_kernel_set_argument_value_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelSetArgumentValue 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelSetArgumentValueCb_t)(
    ze_kernel_set_argument_value_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelSetIndirectAccess 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_set_indirect_access_params_t
{
    ze_kernel_handle_t* phKernel;
    ze_kernel_indirect_access_flags_t* pflags;
} ze_kernel_set_indirect_access_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelSetIndirectAccess 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelSetIndirectAccessCb_t)(
    ze_kernel_set_indirect_access_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelGetIndirectAccess 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_get_indirect_access_params_t
{
    ze_kernel_handle_t* phKernel;
    ze_kernel_indirect_access_flags_t** ppFlags;
} ze_kernel_get_indirect_access_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelGetIndirectAccess 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelGetIndirectAccessCb_t)(
    ze_kernel_get_indirect_access_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelGetSourceAttributes 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_get_source_attributes_params_t
{
    ze_kernel_handle_t* phKernel;
    uint32_t** ppSize;
    char*** ppString;
} ze_kernel_get_source_attributes_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelGetSourceAttributes 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelGetSourceAttributesCb_t)(
    ze_kernel_get_source_attributes_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelGetProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_get_properties_params_t
{
    ze_kernel_handle_t* phKernel;
    ze_kernel_properties_t** ppKernelProperties;
} ze_kernel_get_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelGetProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelGetPropertiesCb_t)(
    ze_kernel_get_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeKernelGetName 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_kernel_get_name_params_t
{
    ze_kernel_handle_t* phKernel;
    size_t** ppSize;
    char** ppName;
} ze_kernel_get_name_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeKernelGetName 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnKernelGetNameCb_t)(
    ze_kernel_get_name_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Kernel callback functions pointers
typedef struct _ze_kernel_callbacks_t
{
    ze_pfnKernelCreateCb_t                                          pfnCreateCb;
    ze_pfnKernelDestroyCb_t                                         pfnDestroyCb;
    ze_pfnKernelSetCacheConfigCb_t                                  pfnSetCacheConfigCb;
    ze_pfnKernelSetGroupSizeCb_t                                    pfnSetGroupSizeCb;
    ze_pfnKernelSuggestGroupSizeCb_t                                pfnSuggestGroupSizeCb;
    ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t                 pfnSuggestMaxCooperativeGroupCountCb;
    ze_pfnKernelSetArgumentValueCb_t                                pfnSetArgumentValueCb;
    ze_pfnKernelSetIndirectAccessCb_t                               pfnSetIndirectAccessCb;
    ze_pfnKernelGetIndirectAccessCb_t                               pfnGetIndirectAccessCb;
    ze_pfnKernelGetSourceAttributesCb_t                             pfnGetSourceAttributesCb;
    ze_pfnKernelGetPropertiesCb_t                                   pfnGetPropertiesCb;
    ze_pfnKernelGetNameCb_t                                         pfnGetNameCb;
} ze_kernel_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeSamplerCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_sampler_create_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    const ze_sampler_desc_t** pdesc;
    ze_sampler_handle_t** pphSampler;
} ze_sampler_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeSamplerCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnSamplerCreateCb_t)(
    ze_sampler_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeSamplerDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_sampler_destroy_params_t
{
    ze_sampler_handle_t* phSampler;
} ze_sampler_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeSamplerDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnSamplerDestroyCb_t)(
    ze_sampler_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Sampler callback functions pointers
typedef struct _ze_sampler_callbacks_t
{
    ze_pfnSamplerCreateCb_t                                         pfnCreateCb;
    ze_pfnSamplerDestroyCb_t                                        pfnDestroyCb;
} ze_sampler_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zePhysicalMemCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_physical_mem_create_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    ze_physical_mem_desc_t** pdesc;
    ze_physical_mem_handle_t** pphPhysicalMemory;
} ze_physical_mem_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zePhysicalMemCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnPhysicalMemCreateCb_t)(
    ze_physical_mem_create_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zePhysicalMemDestroy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_physical_mem_destroy_params_t
{
    ze_context_handle_t* phContext;
    ze_physical_mem_handle_t* phPhysicalMemory;
} ze_physical_mem_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zePhysicalMemDestroy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnPhysicalMemDestroyCb_t)(
    ze_physical_mem_destroy_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of PhysicalMem callback functions pointers
typedef struct _ze_physical_mem_callbacks_t
{
    ze_pfnPhysicalMemCreateCb_t                                     pfnCreateCb;
    ze_pfnPhysicalMemDestroyCb_t                                    pfnDestroyCb;
} ze_physical_mem_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemAllocShared 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_alloc_shared_params_t
{
    ze_context_handle_t* phContext;
    const ze_device_mem_alloc_desc_t** pdevice_desc;
    const ze_host_mem_alloc_desc_t** phost_desc;
    size_t* psize;
    size_t* palignment;
    ze_device_handle_t* phDevice;
    void*** ppptr;
} ze_mem_alloc_shared_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemAllocShared 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemAllocSharedCb_t)(
    ze_mem_alloc_shared_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemAllocDevice 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_alloc_device_params_t
{
    ze_context_handle_t* phContext;
    const ze_device_mem_alloc_desc_t** pdevice_desc;
    size_t* psize;
    size_t* palignment;
    ze_device_handle_t* phDevice;
    void*** ppptr;
} ze_mem_alloc_device_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemAllocDevice 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemAllocDeviceCb_t)(
    ze_mem_alloc_device_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemAllocHost 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_alloc_host_params_t
{
    ze_context_handle_t* phContext;
    const ze_host_mem_alloc_desc_t** phost_desc;
    size_t* psize;
    size_t* palignment;
    void*** ppptr;
} ze_mem_alloc_host_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemAllocHost 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemAllocHostCb_t)(
    ze_mem_alloc_host_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemFree 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_free_params_t
{
    ze_context_handle_t* phContext;
    void** pptr;
} ze_mem_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemFree 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemFreeCb_t)(
    ze_mem_free_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemGetAllocProperties 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_get_alloc_properties_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    ze_memory_allocation_properties_t** ppMemAllocProperties;
    ze_device_handle_t** pphDevice;
} ze_mem_get_alloc_properties_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemGetAllocProperties 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemGetAllocPropertiesCb_t)(
    ze_mem_get_alloc_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemGetAddressRange 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_get_address_range_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    void*** ppBase;
    size_t** ppSize;
} ze_mem_get_address_range_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemGetAddressRange 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemGetAddressRangeCb_t)(
    ze_mem_get_address_range_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemGetIpcHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_get_ipc_handle_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    ze_ipc_mem_handle_t** ppIpcHandle;
} ze_mem_get_ipc_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemGetIpcHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemGetIpcHandleCb_t)(
    ze_mem_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemOpenIpcHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_open_ipc_handle_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    ze_ipc_mem_handle_t* phandle;
    ze_ipc_memory_flags_t* pflags;
    void*** ppptr;
} ze_mem_open_ipc_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemOpenIpcHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemOpenIpcHandleCb_t)(
    ze_mem_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeMemCloseIpcHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_mem_close_ipc_handle_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
} ze_mem_close_ipc_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeMemCloseIpcHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnMemCloseIpcHandleCb_t)(
    ze_mem_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Mem callback functions pointers
typedef struct _ze_mem_callbacks_t
{
    ze_pfnMemAllocSharedCb_t                                        pfnAllocSharedCb;
    ze_pfnMemAllocDeviceCb_t                                        pfnAllocDeviceCb;
    ze_pfnMemAllocHostCb_t                                          pfnAllocHostCb;
    ze_pfnMemFreeCb_t                                               pfnFreeCb;
    ze_pfnMemGetAllocPropertiesCb_t                                 pfnGetAllocPropertiesCb;
    ze_pfnMemGetAddressRangeCb_t                                    pfnGetAddressRangeCb;
    ze_pfnMemGetIpcHandleCb_t                                       pfnGetIpcHandleCb;
    ze_pfnMemOpenIpcHandleCb_t                                      pfnOpenIpcHandleCb;
    ze_pfnMemCloseIpcHandleCb_t                                     pfnCloseIpcHandleCb;
} ze_mem_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeVirtualMemReserve 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_virtual_mem_reserve_params_t
{
    ze_context_handle_t* phContext;
    const void** ppStart;
    size_t* psize;
    void*** ppptr;
} ze_virtual_mem_reserve_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeVirtualMemReserve 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnVirtualMemReserveCb_t)(
    ze_virtual_mem_reserve_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeVirtualMemFree 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_virtual_mem_free_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    size_t* psize;
} ze_virtual_mem_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeVirtualMemFree 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnVirtualMemFreeCb_t)(
    ze_virtual_mem_free_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeVirtualMemQueryPageSize 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_virtual_mem_query_page_size_params_t
{
    ze_context_handle_t* phContext;
    ze_device_handle_t* phDevice;
    size_t* psize;
    size_t** ppagesize;
} ze_virtual_mem_query_page_size_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeVirtualMemQueryPageSize 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnVirtualMemQueryPageSizeCb_t)(
    ze_virtual_mem_query_page_size_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeVirtualMemMap 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_virtual_mem_map_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    size_t* psize;
    ze_physical_mem_handle_t* phPhysicalMemory;
    size_t* poffset;
    ze_memory_access_attribute_t* paccess;
} ze_virtual_mem_map_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeVirtualMemMap 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnVirtualMemMapCb_t)(
    ze_virtual_mem_map_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeVirtualMemUnmap 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_virtual_mem_unmap_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    size_t* psize;
} ze_virtual_mem_unmap_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeVirtualMemUnmap 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnVirtualMemUnmapCb_t)(
    ze_virtual_mem_unmap_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeVirtualMemSetAccessAttribute 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_virtual_mem_set_access_attribute_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    size_t* psize;
    ze_memory_access_attribute_t* paccess;
} ze_virtual_mem_set_access_attribute_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeVirtualMemSetAccessAttribute 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnVirtualMemSetAccessAttributeCb_t)(
    ze_virtual_mem_set_access_attribute_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zeVirtualMemGetAccessAttribute 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _ze_virtual_mem_get_access_attribute_params_t
{
    ze_context_handle_t* phContext;
    const void** pptr;
    size_t* psize;
    ze_memory_access_attribute_t** paccess;
    size_t** poutSize;
} ze_virtual_mem_get_access_attribute_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zeVirtualMemGetAccessAttribute 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZE_APICALL *ze_pfnVirtualMemGetAccessAttributeCb_t)(
    ze_virtual_mem_get_access_attribute_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of VirtualMem callback functions pointers
typedef struct _ze_virtual_mem_callbacks_t
{
    ze_pfnVirtualMemReserveCb_t                                     pfnReserveCb;
    ze_pfnVirtualMemFreeCb_t                                        pfnFreeCb;
    ze_pfnVirtualMemQueryPageSizeCb_t                               pfnQueryPageSizeCb;
    ze_pfnVirtualMemMapCb_t                                         pfnMapCb;
    ze_pfnVirtualMemUnmapCb_t                                       pfnUnmapCb;
    ze_pfnVirtualMemSetAccessAttributeCb_t                          pfnSetAccessAttributeCb;
    ze_pfnVirtualMemGetAccessAttributeCb_t                          pfnGetAccessAttributeCb;
} ze_virtual_mem_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Container for all callbacks
typedef struct _ze_callbacks_t
{
    ze_global_callbacks_t               Global;
    ze_driver_callbacks_t               Driver;
    ze_device_callbacks_t               Device;
    ze_context_callbacks_t              Context;
    ze_command_queue_callbacks_t        CommandQueue;
    ze_command_list_callbacks_t         CommandList;
    ze_fence_callbacks_t                Fence;
    ze_event_pool_callbacks_t           EventPool;
    ze_event_callbacks_t                Event;
    ze_image_callbacks_t                Image;
    ze_module_callbacks_t               Module;
    ze_module_build_log_callbacks_t     ModuleBuildLog;
    ze_kernel_callbacks_t               Kernel;
    ze_sampler_callbacks_t              Sampler;
    ze_physical_mem_callbacks_t         PhysicalMem;
    ze_mem_callbacks_t                  Mem;
    ze_virtual_mem_callbacks_t          VirtualMem;
} ze_callbacks_t;
#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZE_API_H