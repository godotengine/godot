//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Wrapper for Khronos glslang compiler. This file is used by Vulkan and Metal backends.
//

#ifndef LIBANGLE_RENDERER_GLSLANG_WRAPPER_UTILS_H_
#define LIBANGLE_RENDERER_GLSLANG_WRAPPER_UTILS_H_

#include <functional>

#include "libANGLE/renderer/ProgramImpl.h"

namespace rx
{
enum class GlslangError
{
    InvalidShader,
};

struct GlslangSourceOptions
{
    // Uniforms set index:
    uint32_t uniformsAndXfbDescriptorSetIndex = 0;
    // Textures set index:
    uint32_t textureDescriptorSetIndex = 1;
    // Other shader resources set index:
    uint32_t shaderResourceDescriptorSetIndex = 2;
    // ANGLE driver uniforms set index:
    uint32_t driverUniformsDescriptorSetIndex = 3;

    // Binding index start for transform feedback buffers:
    uint32_t xfbBindingIndexStart = 16;
};

using GlslangErrorCallback = std::function<angle::Result(GlslangError)>;

void GlslangInitialize();
void GlslangRelease();

// Get the mapped sampler name after the soure is transformed by GlslangGetShaderSource()
std::string GlslangGetMappedSamplerName(const std::string &originalName);

// Transform the source to include actual binding points for various shader
// resources (textures, buffers, xfb, etc)
void GlslangGetShaderSource(const GlslangSourceOptions &options,
                            bool useOldRewriteStructSamplers,
                            const gl::ProgramState &programState,
                            const gl::ProgramLinkedResources &resources,
                            gl::ShaderMap<std::string> *shaderSourcesOut);

angle::Result GlslangGetShaderSpirvCode(GlslangErrorCallback callback,
                                        const gl::Caps &glCaps,
                                        bool enableLineRasterEmulation,
                                        bool enableXfbEmulation,
                                        const gl::ShaderMap<std::string> &shaderSources,
                                        gl::ShaderMap<std::vector<uint32_t>> *shaderCodesOut);

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GLSLANG_WRAPPER_UTILS_H_
