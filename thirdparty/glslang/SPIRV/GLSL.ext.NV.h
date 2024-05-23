/*
** Copyright (c) 2014-2017 The Khronos Group Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and/or associated documentation files (the "Materials"),
** to deal in the Materials without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Materials, and to permit persons to whom the
** Materials are furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Materials.
**
** MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS KHRONOS
** STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS SPECIFICATIONS AND
** HEADER INFORMATION ARE LOCATED AT https://www.khronos.org/registry/
**
** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
** OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
** THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
** IN THE MATERIALS.
*/

#ifndef GLSLextNV_H
#define GLSLextNV_H

enum BuiltIn;
enum Decoration;
enum Op;
enum Capability;

static const int GLSLextNVVersion = 100;
static const int GLSLextNVRevision = 11;

//SPV_NV_sample_mask_override_coverage
const char* const E_SPV_NV_sample_mask_override_coverage = "SPV_NV_sample_mask_override_coverage";

//SPV_NV_geometry_shader_passthrough
const char* const E_SPV_NV_geometry_shader_passthrough = "SPV_NV_geometry_shader_passthrough";

//SPV_NV_viewport_array2
const char* const E_SPV_NV_viewport_array2 = "SPV_NV_viewport_array2";
const char* const E_ARB_shader_viewport_layer_array = "SPV_ARB_shader_viewport_layer_array";

//SPV_NV_stereo_view_rendering
const char* const E_SPV_NV_stereo_view_rendering = "SPV_NV_stereo_view_rendering";

//SPV_NVX_multiview_per_view_attributes
const char* const E_SPV_NVX_multiview_per_view_attributes = "SPV_NVX_multiview_per_view_attributes";

//SPV_NV_shader_subgroup_partitioned
const char* const E_SPV_NV_shader_subgroup_partitioned = "SPV_NV_shader_subgroup_partitioned";

//SPV_NV_fragment_shader_barycentric
const char* const E_SPV_NV_fragment_shader_barycentric = "SPV_NV_fragment_shader_barycentric";

//SPV_NV_compute_shader_derivatives
const char* const E_SPV_NV_compute_shader_derivatives = "SPV_NV_compute_shader_derivatives";

//SPV_NV_shader_image_footprint
const char* const E_SPV_NV_shader_image_footprint = "SPV_NV_shader_image_footprint";

//SPV_NV_mesh_shader
const char* const E_SPV_NV_mesh_shader = "SPV_NV_mesh_shader";

//SPV_NV_raytracing
const char* const E_SPV_NV_ray_tracing = "SPV_NV_ray_tracing";

//SPV_NV_ray_tracing_motion_blur
const char* const E_SPV_NV_ray_tracing_motion_blur = "SPV_NV_ray_tracing_motion_blur";

//SPV_NV_shading_rate
const char* const E_SPV_NV_shading_rate = "SPV_NV_shading_rate";

//SPV_NV_cooperative_matrix
const char* const E_SPV_NV_cooperative_matrix = "SPV_NV_cooperative_matrix";

//SPV_NV_shader_sm_builtins
const char* const E_SPV_NV_shader_sm_builtins = "SPV_NV_shader_sm_builtins";

//SPV_NV_shader_execution_reorder
const char* const E_SPV_NV_shader_invocation_reorder = "SPV_NV_shader_invocation_reorder";

//SPV_NV_displacement_micromap
const char* const E_SPV_NV_displacement_micromap = "SPV_NV_displacement_micromap";

#endif  // #ifndef GLSLextNV_H
