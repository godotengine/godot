//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FeaturesMtl.h: Optional features for the Metal renderer.
//

#ifndef ANGLE_PLATFORM_FEATURESMTL_H_
#define ANGLE_PLATFORM_FEATURESMTL_H_

#include "platform/Feature.h"

namespace angle
{

struct FeaturesMtl : FeatureSetBase
{
    // BaseVertex/Instanced draw support:
    Feature hasBaseVertexInstancedDraw = {
        "has_base_vertex_instanced_draw", FeatureCategory::MetalFeatures,
        "The renderer supports base vertex instanced draw", &members};

    // Support depth texture filtering
    Feature hasDepthTextureFiltering = {
        "has_depth_texture_filtering", FeatureCategory::MetalFeatures,
        "The renderer supports depth texture's filtering other than nearest", &members};

    // Support explicit memory barrier
    Feature hasExplicitMemBarrier = {"has_explicit_mem_barrier", FeatureCategory::MetalFeatures,
                                     "The renderer supports explicit memory barrier", &members};

    // Non-uniform compute shader dispatch support, i.e. Group size is not necessarily to be fixed:
    Feature hasNonUniformDispatch = {
        "has_non_uniform_dispatch", FeatureCategory::MetalFeatures,
        "The renderer supports non uniform compute shader dispatch's group size", &members};

    // fragment stencil output support
    Feature hasStencilOutput = {"has_stencil_output", FeatureCategory::MetalFeatures,
                                "The renderer supports stencil output from fragment shader",
                                &members};

    // Texture swizzle support:
    Feature hasTextureSwizzle = {"has_texture_swizzle", FeatureCategory::MetalFeatures,
                                 "The renderer supports texture swizzle", &members};

    Feature hasDepthAutoResolve = {
        "has_msaa_depth_auto_resolve", FeatureCategory::MetalFeatures,
        "The renderer supports MSAA depth auto resolve at the end of render pass", &members};

    Feature hasStencilAutoResolve = {
        "has_msaa_stencil_auto_resolve", FeatureCategory::MetalFeatures,
        "The renderer supports MSAA stencil auto resolve at the end of render pass", &members};

    Feature allowInlineConstVertexData = {
        "allow_inline_const_vertex_data", FeatureCategory::MetalFeatures,
        "The renderer supports using inline constant data for small client vertex data", &members};

    // On macos, separate depth & stencil buffers are not supproted. However, on iOS devices,
    // they are supproted:
    Feature allowSeparatedDepthStencilBuffers = {
        "allow_separate_depth_stencil_buffers", FeatureCategory::MetalFeatures,
        "Some Apple platforms such as iOS allows separate depth & stencil buffers, "
        "whereas others such as macOS don't",
        &members};

    Feature allowRuntimeSamplerCompareMode = {
        "allow_runtime_sampler_compare_mode", FeatureCategory::MetalFeatures,
        "The renderer supports changing sampler's compare mode outside shaders", &members};

    Feature allowBufferReadWrite = {"allow_buffer_read_write", FeatureCategory::MetalFeatures,
                                    "The renderer supports buffer read & write in the same shader",
                                    &members};

    Feature allowMultisampleStoreAndResolve = {
        "allow_msaa_store_and_resolve", FeatureCategory::MetalFeatures,
        "The renderer supports MSAA store and resolve in the same pass", &members};

    Feature breakRenderPassIsCheap = {"break_render_pass_is_cheap", FeatureCategory::MetalFeatures,
                                      "Breaking render pass is a cheap operation", &members};

    Feature forceBufferGPUStorage = {
        "force_buffer_gpu_storage", FeatureCategory::MetalFeatures,
        "On systems that support both buffer' memory allocation on GPU and shared memory (such as "
        "macOS), force using GPU memory allocation for buffers everytime or not.",
        &members};

    Feature forceNonCSBaseMipmapGeneration = {
        "force_non_cs_mipmap_gen", FeatureCategory::MetalFeatures,
        "Turn this feature on to disallow Compute Shader based mipmap generation. Compute Shader "
        "based mipmap generation might cause GPU hang on some older iOS devices.",
        &members};

    Feature emulateDepthRangeMappingInShader = {
        "emulate_depth_range_mapping", FeatureCategory::MetalFeatures,
        "Enable linear depth range mapping in shader. This is work-around for older GPUs where "
        "viewport's depth range is simply a clamp instead of a map",
        &members};
};

}  // namespace angle

#endif  // ANGLE_PLATFORM_FEATURESMTL_H_
