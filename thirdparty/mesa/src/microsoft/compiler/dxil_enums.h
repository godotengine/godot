/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef DIXL_ENUMS_H
#define DIXL_ENUMS_H

enum dxil_signature_kind {
   DXIL_SIG_INVALID = 0,
   DXIL_SIG_INPUT,
   DXIL_SIG_OUTPUT,
   DXIL_SIG_PATCH_CONST_OR_PRIM
};

/* These enums are taken from
 * DirectXShaderCompiler/lib/dxc/DXIL/DxilConstants.h
 */
enum dxil_semantic_kind {
  DXIL_SEM_ARBITRARY,
  DXIL_SEM_VERTEX_ID,
  DXIL_SEM_INSTANCE_ID,
  DXIL_SEM_POSITION,
  DXIL_SEM_RENDERTARGET_ARRAY_INDEX,
  DXIL_SEM_VIEWPORT_ARRAY_INDEX,
  DXIL_SEM_CLIP_DISTANCE,
  DXIL_SEM_CULL_DISTANCE,
  DXIL_SEM_OUTPUT_CONTROL_POINT_ID,
  DXIL_SEM_DOMAIN_LOCATION,
  DXIL_SEM_PRIMITIVE_ID,
  DXIL_SEM_GS_INSTANCE_ID,
  DXIL_SEM_SAMPLE_INDEX,
  DXIL_SEM_IS_FRONT_FACE,
  DXIL_SEM_COVERAGE,
  DXIL_SEM_INNER_COVERAGE,
  DXIL_SEM_TARGET,
  DXIL_SEM_DEPTH,
  DXIL_SEM_DEPTH_LE,
  DXIL_SEM_DEPTH_GE,
  DXIL_SEM_STENCIL_REF,
  DXIL_SEM_DISPATCH_THREAD_ID,
  DXIL_SEM_GROUP_ID,
  DXIL_SEM_GROUP_INDEX,
  DXIL_SEM_GROUP_THREAD_ID,
  DXIL_SEM_TESS_FACTOR,
  DXIL_SEM_INSIDE_TESS_FACTOR,
  DXIL_SEM_VIEW_ID,
  DXIL_SEM_BARYCENTRICS,
  DXIL_SEM_SHADING_RATE,
  DXIL_SEM_CULL_PRIMITIVE,
  DXIL_SEM_INVALID
};

enum dxil_prog_sig_semantic {
   DXIL_PROG_SEM_UNDEFINED = 0,
   DXIL_PROG_SEM_POSITION = 1,
   DXIL_PROG_SEM_CLIP_DISTANCE = 2,
   DXIL_PROG_SEM_CULL_DISTANCE = 3,
   DXIL_PROG_SEM_RENDERTARGET_ARRAY_INDEX = 4,
   DXIL_PROG_SEM_VIEWPORT_ARRAY_INDEX = 5,
   DXIL_PROG_SEM_VERTEX_ID = 6,
   DXIL_PROG_SEM_PRIMITIVE_ID = 7,
   DXIL_PROG_SEM_INSTANCE_ID = 8,
   DXIL_PROG_SEM_IS_FRONTFACE = 9,
   DXIL_PROG_SEM_SAMPLE_INDEX = 10,
   DXIL_PROG_SEM_FINAL_QUAD_EDGE_TESSFACTOR = 11,
   DXIL_PROG_SEM_FINAL_QUAD_INSIDE_EDGE_TESSFACTOR = 12,
   DXIL_PROG_SEM_FINAL_TRI_EDGE_TESSFACTOR = 13,
   DXIL_PROG_SEM_FINAL_TRI_INSIDE_EDGE_TESSFACTOR = 14,
   DXIL_PROG_SEM_FINAL_LINE_DETAIL_TESSFACTOR = 15,
   DXIL_PROG_SEM_FINAL_LINE_DENSITY_TESSFACTOR = 16,
   DXIL_PROG_SEM_BARYCENTRICS = 23,
   DXIL_PROG_SEM_SHADING_RATE = 24,
   DXIL_PROG_SEM_CULL_PRIMITIVE = 25,
   DXIL_PROG_SEM_TARGET = 64,
   DXIL_PROG_SEM_DEPTH = 65,
   DXIL_PROG_SEM_COVERAGE = 66,
   DXIL_PROG_SEM_DEPTH_GE = 67,
   DXIL_PROG_SEM_DEPTH_LE = 68,
   DXIL_PROG_SEM_STENCIL_REF = 69,
   DXIL_PROG_SEM_INNER_COVERAGE = 70
};

enum dxil_prog_sig_comp_type {
   DXIL_PROG_SIG_COMP_TYPE_UNKNOWN = 0,
   DXIL_PROG_SIG_COMP_TYPE_UINT32 = 1,
   DXIL_PROG_SIG_COMP_TYPE_SINT32 = 2,
   DXIL_PROG_SIG_COMP_TYPE_FLOAT32 = 3,
   DXIL_PROG_SIG_COMP_TYPE_UINT16 = 4,
   DXIL_PROG_SIG_COMP_TYPE_SINT16 = 5,
   DXIL_PROG_SIG_COMP_TYPE_FLOAT16 = 6,
   DXIL_PROG_SIG_COMP_TYPE_UINT64 = 7,
   DXIL_PROG_SIG_COMP_TYPE_SINT64 = 8,
   DXIL_PROG_SIG_COMP_TYPE_FLOAT64 = 9,
   DXIL_PROG_SIG_COMP_TYPE_COUNT
};


enum dxil_sig_point_kind {
   DXIL_SIG_POINT_VSIN, // Ordinary Vertex Shader input from Input Assembler
   DXIL_SIG_POINT_VSOUT, // Ordinary Vertex Shader output that may feed Rasterizer
   DXIL_SIG_POINT_PCIN, // Patch Constant function non-patch inputs
   DXIL_SIG_POINT_HSIN, // Hull Shader function non-patch inputs
   DXIL_SIG_POINT_HSCPIN, // Hull Shader patch inputs - Control Points
   DXIL_SIG_POINT_HSCPOut, // Hull Shader function output - Control Point
   DXIL_SIG_POINT_PCOUT, // Patch Constant function output - Patch Constant data passed to Domain Shader
   DXIL_SIG_POINT_DSIN, // Domain Shader regular input - Patch Constant data plus system values
   DXIL_SIG_POINT_DSCPIN, // Domain Shader patch input - Control Points
   DXIL_SIG_POINT_DSOUT, // Domain Shader output - vertex data that may feed Rasterizer
   DXIL_SIG_POINT_GSVIN, // Geometry Shader vertex input - qualified with primitive type
   DXIL_SIG_POINT_GSIN, // Geometry Shader non-vertex inputs (system values)
   DXIL_SIG_POINT_GSOUT, // Geometry Shader output - vertex data that may feed Rasterizer
   DXIL_SIG_POINT_PSIN, // Pixel Shader input
   DXIL_SIG_POINT_PSOUT, // Pixel Shader output
   DXIL_SIG_POINT_CSIN, // Compute Shader input
   DXIL_SIG_POINT_MSIN, // Mesh Shader input
   DXIL_SIG_POINT_MSOUT, // Mesh Shader vertices output
   DXIL_SIG_POINT_MSPOUT, // Mesh Shader primitives output
   DXIL_SIG_POINT_ASIN, // Amplification Shader input
   DXIL_SIG_POINT_INVALID
};

enum dxil_min_precision  {
  DXIL_MIN_PREC_DEFAULT = 0,
  DXIL_MIN_PREC_FLOAT16 = 1,
  DXIL_MIN_PREC_FLOAT2_8 = 2,
  DXIL_MIN_PREC_RESERVED = 3,
  DXIL_MIN_PREC_SINT16 = 4,
  DXIL_MIN_PREC_UINT16 = 5,
  DXIL_MIN_PREC_ANY16 = 0xf0,
  DXIL_MIN_PREC_ANY10 = 0xf1
};

enum dxil_semantic_interpret_kind {
   DXIL_SEM_INTERP_NA, // Not Available
   DXIL_SEM_INTERP_SV, // Normal System Value
   DXIL_SEM_INTERP_SGV, // System Generated Value (sorted last)
   DXIL_SEM_INTERP_ARB, // Treated as Arbitrary
   DXIL_SEM_INTERP_NOT_IN_SIG, // Not included in signature (intrinsic access)
   DXIL_SEM_INTERP_NOT_PACKED, // Included in signature, but does not contribute to packing
   DXIL_SEM_INTERP_TARGET, // Special handling for SV_Target
   DXIL_SEM_INTERP_TESSFACTOR, // Special handling for tessellation factors
   DXIL_SEM_INTERP_SHADOW, // Shadow element must be added to a signature for compatibility
   DXIL_SEM_INTERP_CLIPCULL, // Special packing rules for SV_ClipDistance or SV_CullDistance
   DXIL_SEM_INTERP_INVALID
};

enum dxil_component_type {
   DXIL_COMP_TYPE_INVALID = 0,
   DXIL_COMP_TYPE_I1 = 1,
   DXIL_COMP_TYPE_I16 = 2,
   DXIL_COMP_TYPE_U16 = 3,
   DXIL_COMP_TYPE_I32 = 4,
   DXIL_COMP_TYPE_U32 = 5,
   DXIL_COMP_TYPE_I64 = 6,
   DXIL_COMP_TYPE_U64 = 7,
   DXIL_COMP_TYPE_F16 = 8,
   DXIL_COMP_TYPE_F32 = 9,
   DXIL_COMP_TYPE_F64 = 10,
   DXIL_COMP_TYPE_SNORMF16 = 11,
   DXIL_COMP_TYPE_UNORMF16 = 12,
   DXIL_COMP_TYPE_SNORMF32 = 13,
   DXIL_COMP_TYPE_UNORMF32 = 14,
   DXIL_COMP_TYPE_SNORMF64 = 15,
   DXIL_COMP_TYPE_UNORMF64 = 16
};

enum dxil_interpolation_mode  {
  DXIL_INTERP_UNDEFINED                   = 0,
  DXIL_INTERP_CONSTANT                    = 1,
  DXIL_INTERP_LINEAR                      = 2,
  DXIL_INTERP_LINEAR_CENTROID             = 3,
  DXIL_INTERP_LINEAR_NOPERSPECTIVE        = 4,
  DXIL_INTERP_LINEAR_NOPERSPECTIVE_CENTROID  = 5,
  DXIL_INTERP_LINEAR_SAMPLE               = 6,
  DXIL_INTERP_LINEAR_NOPERSPECTIVE_SAMPLE = 7,
  DXIL_INTERP_INVALID                     = 8
};

enum overload_type {
   DXIL_NONE,
   DXIL_I16,
   DXIL_I32,
   DXIL_I64,
   DXIL_F16,
   DXIL_F32,
   DXIL_F64,
   DXIL_NUM_OVERLOADS
};

enum dxil_resource_class {
   DXIL_RESOURCE_CLASS_SRV     = 0,
   DXIL_RESOURCE_CLASS_UAV     = 1,
   DXIL_RESOURCE_CLASS_CBV     = 2,
   DXIL_RESOURCE_CLASS_SAMPLER = 3
};

enum dxil_resource_kind {
   DXIL_RESOURCE_KIND_INVALID           = 0,
   DXIL_RESOURCE_KIND_TEXTURE1D         = 1,
   DXIL_RESOURCE_KIND_TEXTURE2D         = 2,
   DXIL_RESOURCE_KIND_TEXTURE2DMS       = 3,
   DXIL_RESOURCE_KIND_TEXTURE3D         = 4,
   DXIL_RESOURCE_KIND_TEXTURECUBE       = 5,
   DXIL_RESOURCE_KIND_TEXTURE1D_ARRAY   = 6,
   DXIL_RESOURCE_KIND_TEXTURE2D_ARRAY   = 7,
   DXIL_RESOURCE_KIND_TEXTURE2DMS_ARRAY = 8,
   DXIL_RESOURCE_KIND_TEXTURECUBE_ARRAY = 9,
   DXIL_RESOURCE_KIND_TYPED_BUFFER      = 10,
   DXIL_RESOURCE_KIND_RAW_BUFFER        = 11,
   DXIL_RESOURCE_KIND_STRUCTURED_BUFFER = 12,
   DXIL_RESOURCE_KIND_CBUFFER           = 13,
   DXIL_RESOURCE_KIND_SAMPLER           = 14,
   DXIL_RESOURCE_KIND_TBUFFER           = 15,
};

enum dxil_sampler_kind {
   DXIL_SAMPLER_KIND_DEFAULT    = 0,
   DXIL_SAMPLER_KIND_COMPARISON = 1,
   DXIL_SAMPLER_KIND_MONO       = 2,
   DXIL_SAMPLER_KIND_INVALID    = 3,
};

enum dxil_attr_kind {
   DXIL_ATTR_KIND_NONE = 0,
   DXIL_ATTR_KIND_NO_DUPLICATE = 12,
   DXIL_ATTR_KIND_NO_UNWIND = 18,
   DXIL_ATTR_KIND_READ_NONE = 20,
   DXIL_ATTR_KIND_READ_ONLY = 21,
};

enum dxil_input_primitive {
   DXIL_INPUT_PRIMITIVE_UNDEFINED         = 0,
   DXIL_INPUT_PRIMITIVE_POINT             = 1,
   DXIL_INPUT_PRIMITIVE_LINE              = 2,
   DXIL_INPUT_PRIMITIVE_TRIANGLE          = 3,
   DXIL_INPUT_PRIMITIVE_LINES_ADJENCY     = 6,
   DXIL_INPUT_PRIMITIVE_TRIANGLES_ADJENCY = 7,
};

enum dxil_primitive_topology {
   DXIL_PRIMITIVE_TOPOLOGY_UNDEFINED      = 0,
   DXIL_PRIMITIVE_TOPOLOGY_POINT_LIST     = 1,
   DXIL_PRIMITIVE_TOPOLOGY_LINE_LIST      = 2,
   DXIL_PRIMITIVE_TOPOLOGY_LINE_STRIP     = 3,
   DXIL_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST  = 4,
   DXIL_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 5,
};

enum dxil_shader_tag {
   DXIL_SHADER_TAG_FLAGS       = 0,
   DXIL_SHADER_TAG_GS_STATE    = 1,
   DXIL_SHADER_TAG_DS_STATE    = 2,
   DXIL_SHADER_TAG_HS_STATE    = 3,
   DXIL_SHADER_TAG_NUM_THREADS = 4,
};

enum dxil_barrier_mode {
   DXIL_BARRIER_MODE_SYNC_THREAD_GROUP = 1,
   DXIL_BARRIER_MODE_UAV_FENCE_GLOBAL = 2,
   DXIL_BARRIER_MODE_UAV_FENCE_THREAD_GROUP = 4,
   DXIL_BARRIER_MODE_GROUPSHARED_MEM_FENCE = 8,
};

enum dxil_address_space {
   DXIL_AS_DEFAULT = 0,
   DXIL_AS_DEVMEM = 1,
   DXIL_AS_CBUF = 2,
   DXIL_AS_GROUPSHARED = 3,
};

enum dxil_rmw_op {
   DXIL_RMWOP_XCHG = 0,
   DXIL_RMWOP_ADD = 1,
   DXIL_RMWOP_SUB = 2,
   DXIL_RMWOP_AND = 3,
   DXIL_RMWOP_NAND = 4,
   DXIL_RMWOP_OR = 5,
   DXIL_RMWOP_XOR = 6,
   DXIL_RMWOP_MAX = 7,
   DXIL_RMWOP_MIN = 8,
   DXIL_RMWOP_UMAX = 9,
   DXIL_RMWOP_UMIN = 10,
};

enum dxil_atomic_ordering {
   DXIL_ATOMIC_ORDERING_NOTATOMIC = 0,
   DXIL_ATOMIC_ORDERING_UNORDERED = 1,
   DXIL_ATOMIC_ORDERING_MONOTONIC = 2,
   DXIL_ATOMIC_ORDERING_ACQUIRE = 3,
   DXIL_ATOMIC_ORDERING_RELEASE = 4,
   DXIL_ATOMIC_ORDERING_ACQREL = 5,
   DXIL_ATOMIC_ORDERING_SEQCST = 6,
};

enum dxil_sync_scope {
   DXIL_SYNC_SCOPE_SINGLETHREAD = 0,
   DXIL_SYNC_SCOPE_CROSSTHREAD = 1,
};

enum dxil_tessellator_domain {
   DXIL_TESSELLATOR_DOMAIN_UNDEFINED = 0,
   DXIL_TESSELLATOR_DOMAIN_ISOLINE = 1,
   DXIL_TESSELLATOR_DOMAIN_TRI = 2,
   DXIL_TESSELLATOR_DOMAIN_QUAD = 3,
};

enum dxil_tessellator_output_primitive {
   DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_UNDEFINED = 0,
   DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_POINT = 1,
   DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_LINE = 2,
   DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_TRIANGLE_CW = 3,
   DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_TRIANGLE_CCW = 4,
};

enum dxil_tessellator_partitioning {
   DXIL_TESSELLATOR_PARTITIONING_UNDEFINED = 0,
   DXIL_TESSELLATOR_PARTITIONING_INTEGER = 1,
   DXIL_TESSELLATOR_PARTITIONING_POW2 = 2,
   DXIL_TESSELLATOR_PARTITIONING_FRACTIONAL_ODD = 3,
   DXIL_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN = 4,
};

enum dxil_signature_element_extended_properties {
   DXIL_SIGNATURE_ELEMENT_OUTPUT_STREAM = 0,
   DXIL_SIGNATURE_ELEMENT_GLOBAL_SYMBOL = 1,
   DXIL_SIGNATURE_ELEMENT_DYNAMIC_INDEX_COMPONENT_MASK = 2,
   DXIL_SIGNATURE_ELEMENT_USAGE_COMPONENT_MASK = 3,
};

#ifdef __cplusplus
extern "C" {
#endif

struct glsl_type;

enum dxil_component_type dxil_get_comp_type(const struct glsl_type *type);

enum dxil_prog_sig_comp_type dxil_get_prog_sig_comp_type(const struct glsl_type *type);

enum dxil_resource_kind dxil_get_resource_kind(const struct glsl_type *type);

enum dxil_primitive_topology dxil_get_primitive_topology(unsigned topology);

enum dxil_input_primitive dxil_get_input_primitive(unsigned primitive);

const char *dxil_overload_suffix( enum overload_type overload);

#ifdef __cplusplus
}
#endif


#endif // DXIL_ENUMS_H
