//
// Copyright (c) 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// GlslangUtils: Wrapper for Khronos's glslang compiler.
//

#ifndef LIBANGLE_RENDERER_METAL_GLSLANGWRAPPER_H_
#define LIBANGLE_RENDERER_METAL_GLSLANGWRAPPER_H_

#include "libANGLE/BinaryStream.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/ProgramImpl.h"
#include "libANGLE/renderer/metal/mtl_common.h"

namespace rx
{
namespace mtl
{

struct SamplerBinding
{
    uint32_t textureBinding = 0;
    uint32_t samplerBinding = 0;
};

struct TranslatedShaderInfo
{
    void reset();

    void save(gl::BinaryOutputStream *stream);
    void load(gl::BinaryInputStream *stream);

    // Translated Metal source code
    std::string metalShaderSource;
    // Metal library compiled from source code above. Used by ProgramMtl.
    AutoObjCPtr<id<MTLLibrary>> metalLibrary;

    std::array<SamplerBinding, kMaxGLSamplerBindings> actualSamplerBindings;
    std::array<uint32_t, kMaxGLUBOBindings> actualUBOBindings;
    std::array<uint32_t, kMaxShaderXFBs> actualXFBBindings;
    bool hasUBOArgumentBuffer;
};

void GlslangGetShaderSource(const gl::ProgramState &programState,
                            const gl::ProgramLinkedResources &resources,
                            gl::ShaderMap<std::string> *shaderSourcesOut);

angle::Result GlslangGetShaderSpirvCode(
    ErrorHandler *context,
    const gl::Caps &glCaps,
    const gl::ProgramState &programState,
    bool enableLineRasterEmulation,
    const gl::ShaderMap<std::string> &shaderSources,
    gl::ShaderMap<std::vector<uint32_t>> *shaderCodeOut,
    std::vector<uint32_t> *xfbOnlyShaderCodeOut /** nullable */);

// Translate from SPIR-V code to Metal shader source code.
angle::Result SpirvCodeToMsl(Context *context,
                             const gl::ProgramState &programState,
                             gl::ShaderMap<std::vector<uint32_t>> *spirvShaderCode,
                             std::vector<uint32_t> *xfbOnlySpirvCode /** nullable */,
                             gl::ShaderMap<TranslatedShaderInfo> *mslShaderInfoOut,
                             TranslatedShaderInfo *mslXfbOnlyShaderInfoOut /** nullable */);

// Get equivalent shadow compare mode that is used in translated msl shader.
uint MslGetShaderShadowCompareMode(GLenum mode, GLenum func);

}  // namespace mtl
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_GLSLANGWRAPPER_H_ */
