// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes

#define ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE ((ze_result_t)0x7ff00004) ///< [Core, Experimental] operands of comparison are not compatible
#define ZE_RESULT_EXP_RTAS_BUILD_RETRY ((ze_result_t) 0x7ff00005)           ///< [Core, Experimental] ray tracing acceleration structure build failed
                                                                            ///< due to insufficient resources, retry with a larger buffer allocation
#define ZE_RESULT_EXP_RTAS_BUILD_DEFERRED ((ze_result_t) 0x7ff00006)        ///< [Core, Experimental] ray tracing acceleration structure build
                                                                            ///< operation deferred to parallel operation join

#define ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC ((ze_structure_type_t)0x0002000E)   ///< ::ze_rtas_builder_exp_desc_t
#define ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC ((ze_structure_type_t)0x0002000F)  ///< ::ze_rtas_builder_build_op_exp_desc_t
#define ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES ((ze_structure_type_t)0x00020010) ///< ::ze_rtas_builder_exp_properties_t
#define ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES ((ze_structure_type_t)0x00020011)  ///< ::ze_rtas_parallel_operation_exp_properties_t
#define ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES ((ze_structure_type_t)0x00020012)  ///< ::ze_rtas_device_exp_properties_t
#define ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS ((ze_structure_type_t)0x00020013)   ///< ::ze_rtas_geometry_aabbs_exp_cb_params_t


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
                                                                            ///< structure (i.e. contains sType and pNext).
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
///       can be queried with the ::zeRTASBuilderGetPropertiesExp function. The
///       acceleration structure buffer must be a shared USM allocation and
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
