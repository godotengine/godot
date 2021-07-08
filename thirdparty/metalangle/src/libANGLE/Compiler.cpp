//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Compiler.cpp: implements the gl::Compiler class.

#include "libANGLE/Compiler.h"

#include "common/debug.h"
#include "libANGLE/State.h"
#include "libANGLE/renderer/CompilerImpl.h"
#include "libANGLE/renderer/GLImplFactory.h"

namespace gl
{

namespace
{

// To know when to call sh::Initialize and sh::Finalize.
size_t gActiveCompilers = 0;

ShShaderSpec SelectShaderSpec(GLint majorVersion,
                              GLint minorVersion,
                              bool isWebGL,
                              EGLenum clientType)
{
    // For Desktop GL
    if (clientType == EGL_OPENGL_API)
    {
        return SH_GL_COMPATIBILITY_SPEC;
    }

    if (majorVersion >= 3)
    {
        if (minorVersion == 1)
        {
            return isWebGL ? SH_WEBGL3_SPEC : SH_GLES3_1_SPEC;
        }
        else
        {
            return isWebGL ? SH_WEBGL2_SPEC : SH_GLES3_SPEC;
        }
    }

    // GLES1 emulation: Use GLES3 shader spec.
    if (!isWebGL && majorVersion == 1)
    {
        return SH_GLES3_SPEC;
    }

    return isWebGL ? SH_WEBGL_SPEC : SH_GLES2_SPEC;
}

}  // anonymous namespace

Compiler::Compiler(rx::GLImplFactory *implFactory, const State &state)
    : mImplementation(implFactory->createCompiler()),
      mSpec(SelectShaderSpec(state.getClientMajorVersion(),
                             state.getClientMinorVersion(),
                             state.getExtensions().webglCompatibility,
                             state.getClientType())),
      mOutputType(mImplementation->getTranslatorOutputType()),
      mResources()
{
    // TODO(http://anglebug.com/3819): Update for GL version specific validation
    ASSERT(state.getClientMajorVersion() == 1 || state.getClientMajorVersion() == 2 ||
           state.getClientMajorVersion() == 3 || state.getClientMajorVersion() == 4);

    const gl::Caps &caps             = state.getCaps();
    const gl::Extensions &extensions = state.getExtensions();

    if (gActiveCompilers == 0)
    {
        sh::Initialize();
    }
    ++gActiveCompilers;

    sh::InitBuiltInResources(&mResources);
    mResources.MaxVertexAttribs             = caps.maxVertexAttributes;
    mResources.MaxVertexUniformVectors      = caps.maxVertexUniformVectors;
    mResources.MaxVaryingVectors            = caps.maxVaryingVectors;
    mResources.MaxVertexTextureImageUnits   = caps.maxShaderTextureImageUnits[ShaderType::Vertex];
    mResources.MaxCombinedTextureImageUnits = caps.maxCombinedTextureImageUnits;
    mResources.MaxTextureImageUnits         = caps.maxShaderTextureImageUnits[ShaderType::Fragment];
    mResources.MaxFragmentUniformVectors    = caps.maxFragmentUniformVectors;
    mResources.MaxDrawBuffers               = caps.maxDrawBuffers;
    mResources.OES_standard_derivatives     = extensions.standardDerivatives;
    mResources.EXT_draw_buffers             = extensions.drawBuffers;
    mResources.EXT_shader_texture_lod       = extensions.shaderTextureLOD;
    mResources.OES_EGL_image_external       = extensions.eglImageExternal;
    mResources.OES_EGL_image_external_essl3 = extensions.eglImageExternalEssl3;
    mResources.NV_EGL_stream_consumer_external = extensions.eglStreamConsumerExternal;
    mResources.ARB_texture_rectangle           = extensions.textureRectangle;
    mResources.OES_texture_storage_multisample_2d_array =
        extensions.textureStorageMultisample2DArray;
    mResources.OES_texture_3D                  = extensions.texture3DOES;
    mResources.ANGLE_texture_multisample       = extensions.textureMultisample;
    mResources.ANGLE_multi_draw                = extensions.multiDraw;
    mResources.ANGLE_base_vertex_base_instance = extensions.baseVertexBaseInstance;
    mResources.APPLE_clip_distance             = extensions.clipDistanceAPPLE;

    // TODO: use shader precision caps to determine if high precision is supported?
    mResources.FragmentPrecisionHigh = 1;
    mResources.EXT_frag_depth        = extensions.fragDepth;

    // OVR_multiview state
    mResources.OVR_multiview = extensions.multiview;

    // OVR_multiview2 state
    mResources.OVR_multiview2 = extensions.multiview2;
    mResources.MaxViewsOVR    = extensions.maxViews;

    // EXT_multisampled_render_to_texture
    mResources.EXT_multisampled_render_to_texture = extensions.multisampledRenderToTexture;

    // GLSL ES 3.0 constants
    mResources.MaxVertexOutputVectors  = caps.maxVertexOutputComponents / 4;
    mResources.MaxFragmentInputVectors = caps.maxFragmentInputComponents / 4;
    mResources.MinProgramTexelOffset   = caps.minProgramTexelOffset;
    mResources.MaxProgramTexelOffset   = caps.maxProgramTexelOffset;

    // EXT_blend_func_extended
    mResources.EXT_blend_func_extended  = extensions.blendFuncExtended;
    mResources.MaxDualSourceDrawBuffers = extensions.maxDualSourceDrawBuffers;

    // APPLE_clip_distance/EXT_clip_cull_distance
    mResources.MaxClipDistances = caps.maxClipDistances;

    // GLSL ES 3.1 constants
    mResources.MaxProgramTextureGatherOffset    = caps.maxProgramTextureGatherOffset;
    mResources.MinProgramTextureGatherOffset    = caps.minProgramTextureGatherOffset;
    mResources.MaxImageUnits                    = caps.maxImageUnits;
    mResources.MaxVertexImageUniforms           = caps.maxShaderImageUniforms[ShaderType::Vertex];
    mResources.MaxFragmentImageUniforms         = caps.maxShaderImageUniforms[ShaderType::Fragment];
    mResources.MaxComputeImageUniforms          = caps.maxShaderImageUniforms[ShaderType::Compute];
    mResources.MaxCombinedImageUniforms         = caps.maxCombinedImageUniforms;
    mResources.MaxCombinedShaderOutputResources = caps.maxCombinedShaderOutputResources;
    mResources.MaxUniformLocations              = caps.maxUniformLocations;

    for (size_t index = 0u; index < 3u; ++index)
    {
        mResources.MaxComputeWorkGroupCount[index] = caps.maxComputeWorkGroupCount[index];
        mResources.MaxComputeWorkGroupSize[index]  = caps.maxComputeWorkGroupSize[index];
    }

    mResources.MaxComputeUniformComponents = caps.maxShaderUniformComponents[ShaderType::Compute];
    mResources.MaxComputeTextureImageUnits = caps.maxShaderTextureImageUnits[ShaderType::Compute];

    mResources.MaxComputeAtomicCounters = caps.maxShaderAtomicCounters[ShaderType::Compute];
    mResources.MaxComputeAtomicCounterBuffers =
        caps.maxShaderAtomicCounterBuffers[ShaderType::Compute];

    mResources.MaxVertexAtomicCounters   = caps.maxShaderAtomicCounters[ShaderType::Vertex];
    mResources.MaxFragmentAtomicCounters = caps.maxShaderAtomicCounters[ShaderType::Fragment];
    mResources.MaxCombinedAtomicCounters = caps.maxCombinedAtomicCounters;
    mResources.MaxAtomicCounterBindings  = caps.maxAtomicCounterBufferBindings;
    mResources.MaxVertexAtomicCounterBuffers =
        caps.maxShaderAtomicCounterBuffers[ShaderType::Vertex];
    mResources.MaxFragmentAtomicCounterBuffers =
        caps.maxShaderAtomicCounterBuffers[ShaderType::Fragment];
    mResources.MaxCombinedAtomicCounterBuffers = caps.maxCombinedAtomicCounterBuffers;
    mResources.MaxAtomicCounterBufferSize      = caps.maxAtomicCounterBufferSize;

    mResources.MaxUniformBufferBindings       = caps.maxUniformBufferBindings;
    mResources.MaxShaderStorageBufferBindings = caps.maxShaderStorageBufferBindings;

    // Needed by point size clamping workaround
    mResources.MaxPointSize = caps.maxAliasedPointSize;

    if (state.getClientMajorVersion() == 2 && !extensions.drawBuffers)
    {
        mResources.MaxDrawBuffers = 1;
    }

    // Geometry Shader constants
    mResources.EXT_geometry_shader          = extensions.geometryShader;
    mResources.MaxGeometryUniformComponents = caps.maxShaderUniformComponents[ShaderType::Geometry];
    mResources.MaxGeometryUniformBlocks     = caps.maxShaderUniformBlocks[ShaderType::Geometry];
    mResources.MaxGeometryInputComponents   = caps.maxGeometryInputComponents;
    mResources.MaxGeometryOutputComponents  = caps.maxGeometryOutputComponents;
    mResources.MaxGeometryOutputVertices    = caps.maxGeometryOutputVertices;
    mResources.MaxGeometryTotalOutputComponents = caps.maxGeometryTotalOutputComponents;
    mResources.MaxGeometryTextureImageUnits = caps.maxShaderTextureImageUnits[ShaderType::Geometry];

    mResources.MaxGeometryAtomicCounterBuffers =
        caps.maxShaderAtomicCounterBuffers[ShaderType::Geometry];
    mResources.MaxGeometryAtomicCounters      = caps.maxShaderAtomicCounters[ShaderType::Geometry];
    mResources.MaxGeometryShaderStorageBlocks = caps.maxShaderStorageBlocks[ShaderType::Geometry];
    mResources.MaxGeometryShaderInvocations   = caps.maxGeometryShaderInvocations;
    mResources.MaxGeometryImageUniforms       = caps.maxShaderImageUniforms[ShaderType::Geometry];
}

Compiler::~Compiler()
{
    for (auto &pool : mPools)
    {
        for (ShCompilerInstance &instance : pool)
        {
            instance.destroy();
        }
    }
    --gActiveCompilers;
    if (gActiveCompilers == 0)
    {
        sh::Finalize();
    }
}

ShCompilerInstance Compiler::getInstance(ShaderType type)
{
    ASSERT(type != ShaderType::InvalidEnum);
    auto &pool = mPools[type];
    if (pool.empty())
    {
        ShHandle handle = sh::ConstructCompiler(ToGLenum(type), mSpec, mOutputType, &mResources);
        ASSERT(handle);
        return ShCompilerInstance(handle, mOutputType, type);
    }
    else
    {
        ShCompilerInstance instance = std::move(pool.back());
        pool.pop_back();
        return instance;
    }
}

void Compiler::putInstance(ShCompilerInstance &&instance)
{
    static constexpr size_t kMaxPoolSize = 32;
    auto &pool                           = mPools[instance.getShaderType()];
    if (pool.size() < kMaxPoolSize)
    {
        pool.push_back(std::move(instance));
    }
    else
    {
        instance.destroy();
    }
}

ShCompilerInstance::ShCompilerInstance() : mHandle(nullptr) {}

ShCompilerInstance::ShCompilerInstance(ShHandle handle,
                                       ShShaderOutput outputType,
                                       ShaderType shaderType)
    : mHandle(handle), mOutputType(outputType), mShaderType(shaderType)
{}

ShCompilerInstance::~ShCompilerInstance()
{
    ASSERT(mHandle == nullptr);
}

void ShCompilerInstance::destroy()
{
    if (mHandle != nullptr)
    {
        sh::Destruct(mHandle);
        mHandle = nullptr;
    }
}

ShCompilerInstance::ShCompilerInstance(ShCompilerInstance &&other)
    : mHandle(other.mHandle), mOutputType(other.mOutputType), mShaderType(other.mShaderType)
{
    other.mHandle = nullptr;
}

ShCompilerInstance &ShCompilerInstance::operator=(ShCompilerInstance &&other)
{
    mHandle       = other.mHandle;
    mOutputType   = other.mOutputType;
    mShaderType   = other.mShaderType;
    other.mHandle = nullptr;
    return *this;
}

ShHandle ShCompilerInstance::getHandle()
{
    return mHandle;
}

ShaderType ShCompilerInstance::getShaderType() const
{
    return mShaderType;
}

const std::string &ShCompilerInstance::getBuiltinResourcesString()
{
    return sh::GetBuiltInResourcesString(mHandle);
}

ShShaderOutput ShCompilerInstance::getShaderOutputType() const
{
    return mOutputType;
}

}  // namespace gl
