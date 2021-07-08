//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// translator_fuzzer.cpp: A libfuzzer fuzzer for the shader translator.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "angle_gl.h"
#include "anglebase/no_destructor.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/util.h"

using namespace sh;

struct TranslatorCacheKey
{
    bool operator==(const TranslatorCacheKey &other) const
    {
        return type == other.type && spec == other.spec && output == other.output;
    }

    uint32_t type   = 0;
    uint32_t spec   = 0;
    uint32_t output = 0;
};

namespace std
{

template <>
struct hash<TranslatorCacheKey>
{
    std::size_t operator()(const TranslatorCacheKey &k) const
    {
        return (hash<uint32_t>()(k.type) << 1) ^ (hash<uint32_t>()(k.spec) >> 1) ^
               hash<uint32_t>()(k.output);
    }
};
}  // namespace std

struct TCompilerDeleter
{
    void operator()(TCompiler *compiler) const { DeleteCompiler(compiler); }
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    // Reserve some size for future compile options
    const size_t kHeaderSize = 128;

    if (size <= kHeaderSize)
    {
        return 0;
    }

    // Make sure the rest of data will be a valid C string so that we don't have to copy it.
    if (data[size - 1] != 0)
    {
        return 0;
    }

    uint32_t type    = *reinterpret_cast<const uint32_t *>(data);
    uint32_t spec    = *reinterpret_cast<const uint32_t *>(data + 4);
    uint32_t output  = *reinterpret_cast<const uint32_t *>(data + 8);
    uint64_t options = *reinterpret_cast<const uint64_t *>(data + 12);

    if (type != GL_FRAGMENT_SHADER && type != GL_VERTEX_SHADER)
    {
        return 0;
    }

    if (spec != SH_GLES2_SPEC && type != SH_WEBGL_SPEC && spec != SH_GLES3_SPEC &&
        spec != SH_WEBGL2_SPEC)
    {
        return 0;
    }

    ShShaderOutput shaderOutput = static_cast<ShShaderOutput>(output);
    if (!(IsOutputGLSL(shaderOutput) || IsOutputESSL(shaderOutput)) &&
        (options & SH_SELECT_VIEW_IN_NV_GLSL_VERTEX_SHADER) != 0u)
    {
        // This compiler option is only available in ESSL and GLSL.
        return 0;
    }

    std::vector<uint32_t> validOutputs;
    validOutputs.push_back(SH_ESSL_OUTPUT);
    validOutputs.push_back(SH_GLSL_COMPATIBILITY_OUTPUT);
    validOutputs.push_back(SH_GLSL_130_OUTPUT);
    validOutputs.push_back(SH_GLSL_140_OUTPUT);
    validOutputs.push_back(SH_GLSL_150_CORE_OUTPUT);
    validOutputs.push_back(SH_GLSL_330_CORE_OUTPUT);
    validOutputs.push_back(SH_GLSL_400_CORE_OUTPUT);
    validOutputs.push_back(SH_GLSL_410_CORE_OUTPUT);
    validOutputs.push_back(SH_GLSL_420_CORE_OUTPUT);
    validOutputs.push_back(SH_GLSL_430_CORE_OUTPUT);
    validOutputs.push_back(SH_GLSL_440_CORE_OUTPUT);
    validOutputs.push_back(SH_GLSL_450_CORE_OUTPUT);
    validOutputs.push_back(SH_HLSL_3_0_OUTPUT);
    validOutputs.push_back(SH_HLSL_4_1_OUTPUT);
    validOutputs.push_back(SH_HLSL_4_0_FL9_3_OUTPUT);
    bool found = false;
    for (auto valid : validOutputs)
    {
        found = found || (valid == output);
    }
    if (!found)
    {
        return 0;
    }

    size -= kHeaderSize;
    data += kHeaderSize;

    if (!sh::Initialize())
    {
        return 0;
    }

    TranslatorCacheKey key;
    key.type   = type;
    key.spec   = spec;
    key.output = output;

    using UniqueTCompiler = std::unique_ptr<TCompiler, TCompilerDeleter>;
    static angle::base::NoDestructor<std::unordered_map<TranslatorCacheKey, UniqueTCompiler>>
        translators;

    if (translators->find(key) == translators->end())
    {
        UniqueTCompiler translator(
            ConstructCompiler(type, static_cast<ShShaderSpec>(spec), shaderOutput));

        if (translator == nullptr)
        {
            return 0;
        }

        ShBuiltInResources resources;
        sh::InitBuiltInResources(&resources);

        // Enable all the extensions to have more coverage
        resources.OES_standard_derivatives        = 1;
        resources.OES_EGL_image_external          = 1;
        resources.OES_EGL_image_external_essl3    = 1;
        resources.NV_EGL_stream_consumer_external = 1;
        resources.ARB_texture_rectangle           = 1;
        resources.EXT_blend_func_extended         = 1;
        resources.EXT_draw_buffers                = 1;
        resources.EXT_frag_depth                  = 1;
        resources.EXT_shader_texture_lod          = 1;
        resources.WEBGL_debug_shader_precision    = 1;
        resources.EXT_shader_framebuffer_fetch    = 1;
        resources.NV_shader_framebuffer_fetch     = 1;
        resources.ARM_shader_framebuffer_fetch    = 1;
        resources.EXT_YUV_target                  = 1;
        resources.APPLE_clip_distance             = 1;
        resources.MaxDualSourceDrawBuffers        = 1;
        resources.MaxClipDistances                = 1;

        if (!translator->Init(resources))
        {
            return 0;
        }

        (*translators)[key] = std::move(translator);
    }

    auto &translator = (*translators)[key];

    const char *shaderStrings[] = {reinterpret_cast<const char *>(data)};
    translator->compile(shaderStrings, 1, options);

    return 0;
}
