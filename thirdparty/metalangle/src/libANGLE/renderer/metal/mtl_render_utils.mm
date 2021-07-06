//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_render_utils.mm:
//    Implements the class methods for RenderUtils.
//

#include "libANGLE/renderer/metal/mtl_render_utils.h"

#include <utility>

#include "common/debug.h"
#include "libANGLE/renderer/metal/BufferMtl.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/QueryMtl.h"
#include "libANGLE/renderer/metal/mtl_common.h"
#include "libANGLE/renderer/metal/mtl_utils.h"

namespace rx
{
namespace mtl
{
namespace
{

#define NUM_COLOR_OUTPUTS_CONSTANT_NAME @"kNumColorOutputs"
#define SOURCE_BUFFER_ALIGNED_CONSTANT_NAME @"kSourceBufferAligned"
#define SOURCE_IDX_IS_U8_CONSTANT_NAME @"kSourceIndexIsU8"
#define SOURCE_IDX_IS_U16_CONSTANT_NAME @"kSourceIndexIsU16"
#define SOURCE_IDX_IS_U32_CONSTANT_NAME @"kSourceIndexIsU32"
#define PREMULTIPLY_ALPHA_CONSTANT_NAME @"kPremultiplyAlpha"
#define UNMULTIPLY_ALPHA_CONSTANT_NAME @"kUnmultiplyAlpha"
#define SOURCE_TEXTURE_TYPE_CONSTANT_NAME @"kSourceTextureType"
#define SOURCE_TEXTURE2_TYPE_CONSTANT_NAME @"kSourceTexture2Type"
#define COPY_FORMAT_TYPE_CONSTANT_NAME @"kCopyFormatType"
#define PIXEL_COPY_TEXTURE_TYPE_CONSTANT_NAME @"kCopyTextureType"
#define VISIBILITY_RESULT_KEEP_OLD_VAL_CONSTANT_NAME @"kCombineWithExistingResult"

// See libANGLE/renderer/metal/shaders/clear.metal
struct ClearParamsUniform
{
    float clearColor[4];
    float clearDepth;
    float padding[3];
};

// See libANGLE/renderer/metal/shaders/blit.metal
struct BlitParamsUniform
{
    // 0: lower left, 1: lower right, 2: upper left
    float srcTexCoords[3][2];
    int srcLevel         = 0;
    int srcLayer         = 0;
    int srcLevel2        = 0;
    int srcLayer2        = 0;
    uint8_t dstFlipX     = 0;
    uint8_t dstFlipY     = 0;
    uint8_t dstLuminance = 0;  // dest texture is luminace
    uint8_t padding[9];
};

struct BlitStencilToBufferParamsUniform
{
    float srcStartTexCoords[2];
    float srcTexCoordSteps[2];
    uint32_t srcLevel;
    uint32_t srcLayer;

    uint32_t dstSize[2];
    uint32_t dstBufferRowPitch;
    uint8_t resolveMS;

    uint8_t padding[11];
};

// See libANGLE/renderer/metal/shaders/genIndices.metal
struct TriFanOrLineLoopArrayParams
{
    uint firstVertex;
    uint vertexCount;
    uint padding[2];
};

struct IndexConversionUniform
{
    uint32_t srcOffset;
    uint32_t indexCount;
    uint8_t primitiveRestartEnabled;
    uint8_t padding[7];
};

// See libANGLE/renderer/metal/shaders/misc.metal
struct CombineVisibilityResultUniform
{
    uint32_t startOffset;
    uint32_t numOffsets;
    uint32_t padding[2];
};

// See libANGLE/renderer/metal/shaders/gen_mipmap.metal
struct Generate3DMipmapUniform
{
    uint32_t srcLevel;
    uint32_t numMipmapsToGenerate;
    uint32_t padding[2];
};

// See libANGLE/renderer/metal/shaders/copy_buffer.metal
struct CopyPixelFromBufferUniforms
{
    uint32_t copySize[3];
    uint32_t padding1;
    uint32_t textureOffset[3];
    uint32_t padding2;
    uint32_t bufferStartOffset;
    uint32_t pixelSize;
    uint32_t bufferRowPitch;
    uint32_t bufferDepthPitch;
};
struct WritePixelToBufferUniforms
{
    uint32_t copySize[2];
    uint32_t textureOffset[2];

    uint32_t bufferStartOffset;
    uint32_t pixelSize;
    uint32_t bufferRowPitch;

    uint32_t textureLevel;
    uint32_t textureLayer;
    uint8_t reverseTextureRowOrder;

    uint8_t padding[11];
};

struct CopyVertexUniforms
{
    uint32_t srcBufferStartOffset;
    uint32_t srcStride;
    uint32_t srcComponentBytes;
    uint32_t srcComponents;
    uint32_t srcDefaultAlphaData;

    uint32_t dstBufferStartOffset;
    uint32_t dstStride;
    uint32_t dstComponents;

    uint32_t vertexCount;

    uint32_t padding[3];
};

// Class to automatically disable occlusion query upon entering block and re-able it upon
// exiting block.
struct ScopedDisableOcclusionQuery
{
    ScopedDisableOcclusionQuery(ContextMtl *contextMtl,
                                RenderCommandEncoder *encoder,
                                angle::Result *resultOut)
        : mContextMtl(contextMtl), mEncoder(encoder), mResultOut(resultOut)
    {
#ifndef NDEBUG
        if (contextMtl->hasActiveOcclusionQuery())
        {
            encoder->pushDebugGroup(@"Disabled OcclusionQuery");
        }
#endif
        // temporarily disable occlusion query
        contextMtl->disableActiveOcclusionQueryInRenderPass();
    }
    ~ScopedDisableOcclusionQuery()
    {
        *mResultOut = mContextMtl->restartActiveOcclusionQueryInRenderPass();
#ifndef NDEBUG
        if (mContextMtl->hasActiveOcclusionQuery())
        {
            mEncoder->popDebugGroup();
        }
#else
        ANGLE_UNUSED_VARIABLE(mEncoder);
#endif
    }

  private:
    ContextMtl *mContextMtl;
    RenderCommandEncoder *mEncoder;

    angle::Result *mResultOut;
};

void GetBlitTexCoords(uint32_t srcWidth,
                      uint32_t srcHeight,
                      const gl::Rectangle &srcRect,
                      bool srcYFlipped,
                      bool unpackFlipX,
                      bool unpackFlipY,
                      float *u0,
                      float *v0,
                      float *u1,
                      float *v1)
{
    int x0 = srcRect.x0();  // left
    int x1 = srcRect.x1();  // right
    int y0 = srcRect.y0();  // lower
    int y1 = srcRect.y1();  // upper
    if (srcYFlipped)
    {
        // If source's Y has been flipped, such as default framebuffer, then adjust the real source
        // rectangle.
        y0 = srcHeight - y1;
        y1 = y0 + srcRect.height;
        std::swap(y0, y1);
    }

    if (unpackFlipX)
    {
        std::swap(x0, x1);
    }

    if (unpackFlipY)
    {
        std::swap(y0, y1);
    }

    *u0 = static_cast<float>(x0) / srcWidth;
    *u1 = static_cast<float>(x1) / srcWidth;
    *v0 = static_cast<float>(y0) / srcHeight;
    *v1 = static_cast<float>(y1) / srcHeight;
}

template <typename T>
angle::Result GenTriFanFromClientElements(ContextMtl *contextMtl,
                                          GLsizei count,
                                          bool primitiveRestartEnabled,
                                          const T *indices,
                                          const BufferRef &dstBuffer,
                                          uint32_t dstOffset)
{
    ASSERT(count > 2);

    uint32_t genIndicesCount;
    ANGLE_TRY(mtl::GetTriangleFanIndicesCount(contextMtl, count, &genIndicesCount));

    constexpr T kSrcPrimitiveRestartIndex    = std::numeric_limits<T>::max();
    const uint32_t kDstPrimitiveRestartIndex = std::numeric_limits<uint32_t>::max();

    uint32_t *dstPtr = reinterpret_cast<uint32_t *>(dstBuffer->map(contextMtl) + dstOffset);
    T triFirstIdx = 0;  // Vertex index of trianlge's 1st vertex
    T srcPrevIdx  = 0;  // Vertex index of trianlge's 2nd vertex
    memcpy(&triFirstIdx, indices, sizeof(triFirstIdx));

    if (primitiveRestartEnabled)
    {
        GLsizei triFirstIdxLoc = 0;
        while (triFirstIdx == kSrcPrimitiveRestartIndex)
        {
            memcpy(&dstPtr[triFirstIdxLoc++], &kDstPrimitiveRestartIndex,
                   sizeof(kDstPrimitiveRestartIndex));
            memcpy(&triFirstIdx, indices + triFirstIdxLoc, sizeof(triFirstIdx));
        }

        if (triFirstIdxLoc + 2 >= count)
        {
            // Not enough indices.
            for (GLsizei i = triFirstIdxLoc; i < count; ++i)
            {
                memcpy(&dstPtr[i], &kDstPrimitiveRestartIndex, sizeof(kDstPrimitiveRestartIndex));
            }
        }
        else if (triFirstIdxLoc + 1 < count)
        {
            memcpy(&srcPrevIdx, indices + triFirstIdxLoc + 1, sizeof(srcPrevIdx));
        }

        for (GLsizei i = triFirstIdxLoc + 2; i < count; ++i)
        {
            uint32_t triIndices[3];

            T srcIdx;
            memcpy(&srcIdx, indices + i, sizeof(srcIdx));
            if (srcPrevIdx == kSrcPrimitiveRestartIndex || srcIdx == kSrcPrimitiveRestartIndex)
            {
                // Incomplete triangle.
                triIndices[0]  = kDstPrimitiveRestartIndex;
                triIndices[1]  = kDstPrimitiveRestartIndex;
                triIndices[2]  = kDstPrimitiveRestartIndex;
                triFirstIdx    = srcIdx;
                triFirstIdxLoc = i;
            }
            else if (i < triFirstIdxLoc + 2)
            {
                // Incomplete triangle
                triIndices[0] = kDstPrimitiveRestartIndex;
                triIndices[1] = kDstPrimitiveRestartIndex;
                triIndices[2] = kDstPrimitiveRestartIndex;
            }
            else
            {
                triIndices[0] = triFirstIdx;
                triIndices[1] = srcPrevIdx;
                triIndices[2] = srcIdx;
            }
            srcPrevIdx = srcIdx;

            memcpy(dstPtr + 3 * (i - 2), triIndices, sizeof(triIndices));
        }
    }
    else
    {
        memcpy(&srcPrevIdx, indices + 1, sizeof(srcPrevIdx));

        for (GLsizei i = 2; i < count; ++i)
        {
            T srcIdx;
            memcpy(&srcIdx, indices + i, sizeof(srcIdx));

            uint32_t triIndices[3];
            triIndices[0] = triFirstIdx;
            triIndices[1] = srcPrevIdx;
            triIndices[2] = srcIdx;
            srcPrevIdx    = srcIdx;

            memcpy(dstPtr + 3 * (i - 2), triIndices, sizeof(triIndices));
        }
    }
    dstBuffer->unmap(contextMtl, dstOffset, genIndicesCount * sizeof(uint32_t));

    return angle::Result::Continue;
}

template <typename T>
angle::Result GenLineLoopFromClientElements(ContextMtl *contextMtl,
                                            GLsizei count,
                                            bool primitiveRestartEnabled,
                                            const T *indices,
                                            const BufferRef &dstBuffer,
                                            uint32_t dstOffset,
                                            uint32_t *indicesGenerated)
{
    ASSERT(count >= 2);
    constexpr T kSrcPrimitiveRestartIndex    = std::numeric_limits<T>::max();
    const uint32_t kDstPrimitiveRestartIndex = std::numeric_limits<uint32_t>::max();

    uint32_t *dstPtr = reinterpret_cast<uint32_t *>(dstBuffer->map(contextMtl) + dstOffset);
    T lineLoopFirstIdx;
    memcpy(&lineLoopFirstIdx, indices, sizeof(lineLoopFirstIdx));

    if (primitiveRestartEnabled)
    {
        GLsizei lineLoopFirstIdxLoc = 0;
        while (lineLoopFirstIdx == kSrcPrimitiveRestartIndex)
        {
            memcpy(&dstPtr[lineLoopFirstIdxLoc++], &kDstPrimitiveRestartIndex,
                   sizeof(kDstPrimitiveRestartIndex));
            memcpy(&lineLoopFirstIdx, indices + lineLoopFirstIdxLoc, sizeof(lineLoopFirstIdx));
        }

        uint32_t dstIdx = lineLoopFirstIdx;
        memcpy(&dstPtr[lineLoopFirstIdxLoc], &dstIdx, sizeof(dstIdx));
        uint32_t dstWritten = lineLoopFirstIdxLoc + 1;

        for (GLsizei i = lineLoopFirstIdxLoc + 1; i < count; ++i)
        {
            T srcIdx;
            memcpy(&srcIdx, indices + i, sizeof(srcIdx));
            if (srcIdx == kSrcPrimitiveRestartIndex)
            {
                // breaking line strip
                dstIdx = lineLoopFirstIdx;
                memcpy(&dstPtr[dstWritten++], &dstIdx, sizeof(dstIdx));
                memcpy(&dstPtr[dstWritten++], &kDstPrimitiveRestartIndex,
                       sizeof(kDstPrimitiveRestartIndex));
                lineLoopFirstIdxLoc = i + 1;
            }
            else
            {
                dstIdx = srcIdx;
                memcpy(&dstPtr[dstWritten++], &dstIdx, sizeof(dstIdx));
                if (lineLoopFirstIdxLoc == i)
                {
                    lineLoopFirstIdx = srcIdx;
                }
            }
        }

        if (lineLoopFirstIdxLoc < count)
        {
            // last segment
            dstIdx = lineLoopFirstIdx;
            memcpy(&dstPtr[dstWritten++], &dstIdx, sizeof(dstIdx));
        }

        *indicesGenerated = dstWritten;
    }
    else
    {
        uint32_t dstIdx = lineLoopFirstIdx;
        memcpy(dstPtr, &dstIdx, sizeof(dstIdx));
        memcpy(dstPtr + count, &dstIdx, sizeof(dstIdx));
        for (GLsizei i = 1; i < count; ++i)
        {
            T srcIdx;
            memcpy(&srcIdx, indices + i, sizeof(srcIdx));

            dstIdx = srcIdx;
            memcpy(dstPtr + i, &dstIdx, sizeof(dstIdx));
        }

        *indicesGenerated = count + 1;
    }
    dstBuffer->unmap(contextMtl, dstOffset, (*indicesGenerated) * sizeof(uint32_t));

    return angle::Result::Continue;
}

template <typename T>
void GetFirstLastIndicesFromClientElements(GLsizei count,
                                           const T *indices,
                                           uint32_t *firstOut,
                                           uint32_t *lastOut)
{
    *firstOut = 0;
    *lastOut  = 0;
    memcpy(firstOut, indices, sizeof(indices[0]));
    memcpy(lastOut, indices + count - 1, sizeof(indices[0]));
}

int GetShaderTextureType(const TextureRef &texture)
{
    if (!texture)
    {
        return -1;
    }
    switch (texture->textureType())
    {
        case MTLTextureType2D:
            return mtl_shader::kTextureType2D;
        case MTLTextureType2DArray:
            return mtl_shader::kTextureType2DArray;
        case MTLTextureType2DMultisample:
            return mtl_shader::kTextureType2DMultisample;
        case MTLTextureTypeCube:
            return mtl_shader::kTextureTypeCube;
        case MTLTextureType3D:
            return mtl_shader::kTextureType3D;
        default:
            UNREACHABLE();
    }

    return 0;
}

int GetPixelTypeIndex(const angle::Format &angleFormat)
{
    if (angleFormat.isSint())
    {
        return static_cast<int>(PixelType::Int);
    }
    else if (angleFormat.isUint())
    {
        return static_cast<int>(PixelType::UInt);
    }
    else
    {
        return static_cast<int>(PixelType::Float);
    }
}

ANGLE_INLINE
void EnsureComputePipelineInitialized(DisplayMtl *display,
                                      NSString *functionName,
                                      AutoObjCPtr<id<MTLComputePipelineState>> *pipelineOut)
{
    AutoObjCPtr<id<MTLComputePipelineState>> &pipeline = *pipelineOut;
    if (pipeline)
    {
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        id<MTLDevice> metalDevice = display->getMetalDevice();
        auto shaderLib            = display->getDefaultShadersLib();
        NSError *err              = nil;
        id<MTLFunction> shader    = [shaderLib newFunctionWithName:functionName];

        [shader ANGLE_MTL_AUTORELEASE];

        pipeline = [[metalDevice newComputePipelineStateWithFunction:shader
                                                               error:&err] ANGLE_MTL_AUTORELEASE];
        if (err && !pipeline)
        {
            ERR() << "Internal error: " << err.localizedDescription.UTF8String << "\n";
        }

        ASSERT(pipeline);
    }
}

ANGLE_INLINE
void EnsureSpecializedComputePipelineInitialized(
    DisplayMtl *display,
    NSString *functionName,
    MTLFunctionConstantValues *funcConstants,
    AutoObjCPtr<id<MTLComputePipelineState>> *pipelineOut)
{
    if (!funcConstants)
    {
        // Non specialized constants provided, use default creation function.
        EnsureComputePipelineInitialized(display, functionName, pipelineOut);
        return;
    }

    AutoObjCPtr<id<MTLComputePipelineState>> &pipeline = *pipelineOut;
    if (pipeline)
    {
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        id<MTLDevice> metalDevice = display->getMetalDevice();
        auto shaderLib            = display->getDefaultShadersLib();
        NSError *err              = nil;

        id<MTLFunction> shader = [shaderLib newFunctionWithName:functionName
                                                 constantValues:funcConstants
                                                          error:&err];
        if (err && !shader)
        {
            ERR() << "Internal error: " << err.localizedDescription.UTF8String << "\n";
        }
        ASSERT([shader ANGLE_MTL_AUTORELEASE]);

        pipeline = [[metalDevice newComputePipelineStateWithFunction:shader
                                                               error:&err] ANGLE_MTL_AUTORELEASE];
        if (err && !pipeline)
        {
            ERR() << "Internal error: " << err.localizedDescription.UTF8String << "\n";
        }
        ASSERT(pipeline);
    }
}

// Function to initialize render pipeline cache with only vertex shader.
ANGLE_INLINE
void EnsureVertexShaderOnlyPipelineCacheInitialized(Context *context,
                                                    NSString *vertexFunctionName,
                                                    RenderPipelineCache *pipelineCacheOut)
{
    RenderPipelineCache &pipelineCache = *pipelineCacheOut;
    if (pipelineCache.getVertexShader())
    {
        // Already initialized
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        DisplayMtl *display    = context->getDisplay();
        auto shaderLib         = display->getDefaultShadersLib();
        id<MTLFunction> shader = [shaderLib newFunctionWithName:vertexFunctionName];

        ASSERT([shader ANGLE_MTL_AUTORELEASE]);

        pipelineCache.setVertexShader(context, shader);
    }
}

// Function to initialize specialized render pipeline cache with only vertex shader.
ANGLE_INLINE
void EnsureSpecializedVertexShaderOnlyPipelineCacheInitialized(
    Context *context,
    NSString *vertexFunctionName,
    MTLFunctionConstantValues *funcConstants,
    RenderPipelineCache *pipelineCacheOut)
{
    if (!funcConstants)
    {
        // Non specialized constants provided, use default creation function.
        EnsureVertexShaderOnlyPipelineCacheInitialized(context, vertexFunctionName,
                                                       pipelineCacheOut);
        return;
    }

    RenderPipelineCache &pipelineCache = *pipelineCacheOut;
    if (pipelineCache.getVertexShader())
    {
        // Already initialized
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        DisplayMtl *display = context->getDisplay();
        auto shaderLib      = display->getDefaultShadersLib();
        NSError *err        = nil;

        id<MTLFunction> shader = [shaderLib newFunctionWithName:vertexFunctionName
                                                 constantValues:funcConstants
                                                          error:&err];
        if (err && !shader)
        {
            ERR() << "Internal error: " << err.localizedDescription.UTF8String << "\n";
        }
        ASSERT([shader ANGLE_MTL_AUTORELEASE]);

        pipelineCache.setVertexShader(context, shader);
    }
}

// Get pipeline descriptor for render pipeline that contains vertex shader acting as compute shader.
ANGLE_INLINE
RenderPipelineDesc GetComputingVertexShaderOnlyRenderPipelineDesc(RenderCommandEncoder *cmdEncoder)
{
    RenderPipelineDesc pipelineDesc;
    const RenderPassDesc &renderPassDesc = cmdEncoder->renderPassDesc();

    renderPassDesc.populateRenderPipelineOutputDesc(&pipelineDesc.outputDescriptor);
    pipelineDesc.rasterizationType      = RenderPipelineRasterization::Disabled;
    pipelineDesc.inputPrimitiveTopology = kPrimitiveTopologyClassPoint;

    return pipelineDesc;
}

template <typename T>
void ClearRenderPipelineCacheArray(T *pipelineCacheArray)
{
    for (RenderPipelineCache &pipelineCache : *pipelineCacheArray)
    {
        pipelineCache.clear();
    }
}

template <typename T>
void ClearRenderPipelineCache2DArray(T *pipelineCache2DArray)
{
    for (auto &level1Array : *pipelineCache2DArray)
    {
        ClearRenderPipelineCacheArray(&level1Array);
    }
}

template <typename T>
void ClearPipelineStateArray(T *pipelineCacheArray)
{
    for (auto &pipeline : *pipelineCacheArray)
    {
        pipeline = nil;
    }
}

template <typename T>
void ClearPipelineState2DArray(T *pipelineCache2DArray)
{
    for (auto &level1Array : *pipelineCache2DArray)
    {
        ClearPipelineStateArray(&level1Array);
    }
}

void DispatchCompute(ContextMtl *contextMtl,
                     ComputeCommandEncoder *encoder,
                     bool allowNonUniform,
                     const MTLSize &numThreads,
                     const MTLSize &threadsPerThreadgroup)
{
    if (allowNonUniform && contextMtl->getDisplay()->getFeatures().hasNonUniformDispatch.enabled)
    {
        encoder->dispatchNonUniform(numThreads, threadsPerThreadgroup);
    }
    else
    {
        MTLSize groups = MTLSizeMake(
            (numThreads.width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            (numThreads.height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            (numThreads.depth + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth);
        encoder->dispatch(groups, threadsPerThreadgroup);
    }
}

void DispatchCompute(ContextMtl *contextMtl,
                     ComputeCommandEncoder *cmdEncoder,
                     id<MTLComputePipelineState> pipelineState,
                     size_t numThreads)
{
    NSUInteger w = std::min<NSUInteger>(pipelineState.threadExecutionWidth, numThreads);
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, 1, 1);

    if (contextMtl->getDisplay()->getFeatures().hasNonUniformDispatch.enabled)
    {
        MTLSize threads = MTLSizeMake(numThreads, 1, 1);
        cmdEncoder->dispatchNonUniform(threads, threadsPerThreadgroup);
    }
    else
    {
        MTLSize groups = MTLSizeMake((numThreads + w - 1) / w, 1, 1);
        cmdEncoder->dispatch(groups, threadsPerThreadgroup);
    }
}

void SetupFullscreenQuadDrawCommonStates(RenderCommandEncoder *cmdEncoder)
{
    cmdEncoder->setCullMode(MTLCullModeNone);
    cmdEncoder->setTriangleFillMode(MTLTriangleFillModeFill);
    cmdEncoder->setDepthBias(0, 0, 0);
}

void SetupBlitWithDrawUniformData(RenderCommandEncoder *cmdEncoder,
                                  const BlitParams &params,
                                  bool isColorBlit)
{

    BlitParamsUniform uniformParams;
    uniformParams.dstFlipX = params.dstFlipX ? 1 : 0;
    uniformParams.dstFlipY = params.dstFlipY ? 1 : 0;
    uniformParams.srcLevel = params.srcLevel;
    uniformParams.srcLayer = params.srcLayer;
    if (isColorBlit)
    {
        const auto colorParams     = static_cast<const ColorBlitParams *>(&params);
        uniformParams.dstLuminance = colorParams->dstLuminance ? 1 : 0;
    }
    else
    {
        const auto dsParams     = static_cast<const DepthStencilBlitParams *>(&params);
        uniformParams.srcLevel2 = dsParams->srcStencilLevel;
        uniformParams.srcLayer2 = dsParams->srcStencilLayer;
    }

    // Compute source texCoords
    uint32_t srcWidth = 0, srcHeight = 0;
    if (params.src)
    {
        srcWidth  = params.src->width(params.srcLevel);
        srcHeight = params.src->height(params.srcLevel);
    }
    else if (!isColorBlit)
    {
        const DepthStencilBlitParams *dsParams =
            static_cast<const DepthStencilBlitParams *>(&params);
        srcWidth  = dsParams->srcStencil->width(dsParams->srcStencilLevel);
        srcHeight = dsParams->srcStencil->height(dsParams->srcStencilLevel);
    }
    else
    {
        UNREACHABLE();
    }

    float u0, v0, u1, v1;
    GetBlitTexCoords(srcWidth, srcHeight, params.srcRect, params.srcYFlipped, params.unpackFlipX,
                     params.unpackFlipY, &u0, &v0, &u1, &v1);

    auto du = u1 - u0;
    auto dv = v1 - v0;

    // lower left
    uniformParams.srcTexCoords[0][0] = u0;
    uniformParams.srcTexCoords[0][1] = v0;

    // lower right
    uniformParams.srcTexCoords[1][0] = u1 + du;
    uniformParams.srcTexCoords[1][1] = v0;

    // upper left
    uniformParams.srcTexCoords[2][0] = u0;
    uniformParams.srcTexCoords[2][1] = v1 + dv;

    cmdEncoder->setVertexData(uniformParams, 0);
    cmdEncoder->setFragmentData(uniformParams, 0);
}

void SetupCommonBlitWithDrawStates(const gl::Context *context,
                                   RenderCommandEncoder *cmdEncoder,
                                   const BlitParams &params,
                                   bool isColorBlit)
{
    // Setup states
    SetupFullscreenQuadDrawCommonStates(cmdEncoder);

    // Viewport
    MTLViewport viewportMtl =
        GetViewport(params.dstRect, params.dstTextureSize.height, params.dstFlipY);
    MTLScissorRect scissorRectMtl =
        GetScissorRect(params.dstScissorRect, params.dstTextureSize.height, params.dstFlipY);
    cmdEncoder->setViewport(viewportMtl);
    cmdEncoder->setScissorRect(scissorRectMtl);

    if (params.src)
    {
        cmdEncoder->setFragmentTexture(params.src, 0);
    }

    // Uniform
    SetupBlitWithDrawUniformData(cmdEncoder, params, isColorBlit);
}

// Overloaded functions to be used with both compute and render command encoder.
ANGLE_INLINE void SetComputeOrVertexBuffer(RenderCommandEncoder *encoder,
                                           const BufferRef &buffer,
                                           uint32_t offset,
                                           uint32_t index)
{
    encoder->setBuffer(gl::ShaderType::Vertex, buffer, offset, index);
}
ANGLE_INLINE void SetComputeOrVertexBufferForWrite(RenderCommandEncoder *encoder,
                                                   const BufferRef &buffer,
                                                   uint32_t offset,
                                                   uint32_t index)
{
    encoder->setBufferForWrite(gl::ShaderType::Vertex, buffer, offset, index);
}
ANGLE_INLINE void SetComputeOrVertexBuffer(ComputeCommandEncoder *encoder,
                                           const BufferRef &buffer,
                                           uint32_t offset,
                                           uint32_t index)
{
    encoder->setBuffer(buffer, offset, index);
}
ANGLE_INLINE void SetComputeOrVertexBufferForWrite(ComputeCommandEncoder *encoder,
                                                   const BufferRef &buffer,
                                                   uint32_t offset,
                                                   uint32_t index)
{
    encoder->setBufferForWrite(buffer, offset, index);
}

template <typename T>
ANGLE_INLINE void SetComputeOrVertexData(RenderCommandEncoder *encoder,
                                         const T &data,
                                         uint32_t index)
{
    encoder->setData(gl::ShaderType::Vertex, data, index);
}
template <typename T>
ANGLE_INLINE void SetComputeOrVertexData(ComputeCommandEncoder *encoder,
                                         const T &data,
                                         uint32_t index)
{
    encoder->setData(data, index);
}

ANGLE_INLINE void SetPipelineState(RenderCommandEncoder *encoder,
                                   id<MTLRenderPipelineState> pipeline)
{
    encoder->setRenderPipelineState(pipeline);
}
ANGLE_INLINE void SetPipelineState(ComputeCommandEncoder *encoder,
                                   id<MTLComputePipelineState> pipeline)
{
    encoder->setComputePipelineState(pipeline);
}

}  // namespace

// StencilBlitViaBufferParams implementation
StencilBlitViaBufferParams::StencilBlitViaBufferParams() {}

StencilBlitViaBufferParams::StencilBlitViaBufferParams(const DepthStencilBlitParams &src)
{
    dstTextureSize = src.dstTextureSize;
    dstRect        = src.dstRect;
    dstScissorRect = src.dstScissorRect;
    dstFlipY       = src.dstFlipY;
    dstFlipX       = src.dstFlipX;
    srcRect        = src.srcRect;
    srcYFlipped    = src.srcYFlipped;
    unpackFlipX    = src.unpackFlipX;
    unpackFlipY    = src.unpackFlipY;

    srcStencil      = src.srcStencil;
    srcStencilLevel = src.srcStencilLevel;
    srcStencilLayer = src.srcStencilLayer;
}

// RenderUtils implementation
RenderUtils::RenderUtils(DisplayMtl *display)
    : Context(display),
      mClearUtils(
          {ClearUtils("clearIntFS"), ClearUtils("clearUIntFS"), ClearUtils("clearFloatFS")}),
      mColorBlitUtils({ColorBlitUtils("blitIntFS"), ColorBlitUtils("blitUIntFS"),
                       ColorBlitUtils("blitFloatFS")}),
      mCopyPixelsUtils(
          {CopyPixelsUtils("readFromBufferToIntTexture", "writeFromIntTextureToBuffer"),
           CopyPixelsUtils("readFromBufferToUIntTexture", "writeFromUIntTextureToBuffer"),
           CopyPixelsUtils("readFromBufferToFloatTexture", "writeFromFloatTextureToBuffer")})
{}

RenderUtils::~RenderUtils() {}

angle::Result RenderUtils::initialize()
{
    return angle::Result::Continue;
}

void RenderUtils::onDestroy()
{
    mDepthStencilBlitUtils.onDestroy();
    mIndexUtils.onDestroy();
    mVisibilityResultUtils.onDestroy();
    mMipmapUtils.onDestroy();
    mVertexFormatUtils.onDestroy();

    for (ClearUtils &util : mClearUtils)
    {
        util.onDestroy();
    }
    for (ColorBlitUtils &util : mColorBlitUtils)
    {
        util.onDestroy();
    }
    for (CopyPixelsUtils &util : mCopyPixelsUtils)
    {
        util.onDestroy();
    }
}

// override ErrorHandler
void RenderUtils::handleError(GLenum glErrorCode,
                              const char *file,
                              const char *function,
                              unsigned int line)
{
    ERR() << "Metal backend encountered an internal error. Code=" << glErrorCode << ".";
}

void RenderUtils::handleError(NSError *nserror,
                              const char *file,
                              const char *function,
                              unsigned int line)
{
    if (!nserror)
    {
        return;
    }

    std::stringstream errorStream;
    ERR() << "Metal backend encountered an internal error: \n"
          << nserror.localizedDescription.UTF8String;
}

// Clear current framebuffer
angle::Result RenderUtils::clearWithDraw(const gl::Context *context,
                                         RenderCommandEncoder *cmdEncoder,
                                         const ClearRectParams &params)
{
    int index = 0;
    if (params.clearColor.valid())
    {
        index = static_cast<int>(params.clearColor.value().type);
    }
    else if (params.colorFormat)
    {
        index = GetPixelTypeIndex(params.colorFormat->actualAngleFormat());
    }
    return mClearUtils[index].clearWithDraw(context, cmdEncoder, params);
}

// Blit texture data to current framebuffer
angle::Result RenderUtils::blitColorWithDraw(const gl::Context *context,
                                             RenderCommandEncoder *cmdEncoder,
                                             const angle::Format &srcAngleFormat,
                                             const ColorBlitParams &params)
{
    int index = GetPixelTypeIndex(srcAngleFormat);
    return mColorBlitUtils[index].blitColorWithDraw(context, cmdEncoder, params);
}

angle::Result RenderUtils::blitColorWithDraw(const gl::Context *context,
                                             RenderCommandEncoder *cmdEncoder,
                                             const angle::Format &srcAngleFormat,
                                             const TextureRef &srcTexture)
{
    if (!srcTexture)
    {
        return angle::Result::Continue;
    }
    ColorBlitParams params;
    params.enabledBuffers.set(0);
    params.src = srcTexture;
    params.dstTextureSize =
        gl::Extents(static_cast<int>(srcTexture->width()), static_cast<int>(srcTexture->height()),
                    static_cast<int>(srcTexture->depth()));
    params.dstRect = params.dstScissorRect = params.srcRect =
        gl::Rectangle(0, 0, params.dstTextureSize.width, params.dstTextureSize.height);

    return blitColorWithDraw(context, cmdEncoder, srcAngleFormat, params);
}

angle::Result RenderUtils::blitDepthStencilWithDraw(const gl::Context *context,
                                                    RenderCommandEncoder *cmdEncoder,
                                                    const DepthStencilBlitParams &params)
{
    return mDepthStencilBlitUtils.blitDepthStencilWithDraw(context, cmdEncoder, params);
}

angle::Result RenderUtils::blitStencilViaCopyBuffer(const gl::Context *context,
                                                    const StencilBlitViaBufferParams &params)
{
    return mDepthStencilBlitUtils.blitStencilViaCopyBuffer(context, params);
}

angle::Result RenderUtils::convertIndexBufferGPU(ContextMtl *contextMtl,
                                                 const IndexConversionParams &params)
{
    return mIndexUtils.convertIndexBufferGPU(contextMtl, params);
}
angle::Result RenderUtils::generateTriFanBufferFromArrays(
    ContextMtl *contextMtl,
    const TriFanOrLineLoopFromArrayParams &params)
{
    return mIndexUtils.generateTriFanBufferFromArrays(contextMtl, params);
}
angle::Result RenderUtils::generateTriFanBufferFromElementsArray(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params)
{
    return mIndexUtils.generateTriFanBufferFromElementsArray(contextMtl, params);
}

angle::Result RenderUtils::generateLineLoopBufferFromArrays(
    ContextMtl *contextMtl,
    const TriFanOrLineLoopFromArrayParams &params)
{
    return mIndexUtils.generateLineLoopBufferFromArrays(contextMtl, params);
}
angle::Result RenderUtils::generateLineLoopLastSegment(ContextMtl *contextMtl,
                                                       uint32_t firstVertex,
                                                       uint32_t lastVertex,
                                                       const BufferRef &dstBuffer,
                                                       uint32_t dstOffset)
{
    return mIndexUtils.generateLineLoopLastSegment(contextMtl, firstVertex, lastVertex, dstBuffer,
                                                   dstOffset);
}
angle::Result RenderUtils::generateLineLoopBufferFromElementsArray(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params,
    uint32_t *indicesGenerated)
{
    return mIndexUtils.generateLineLoopBufferFromElementsArray(contextMtl, params,
                                                               indicesGenerated);
}
angle::Result RenderUtils::generateLineLoopLastSegmentFromElementsArray(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params)
{
    return mIndexUtils.generateLineLoopLastSegmentFromElementsArray(contextMtl, params);
}

void RenderUtils::combineVisibilityResult(
    ContextMtl *contextMtl,
    bool keepOldValue,
    const VisibilityBufferOffsetsMtl &renderPassResultBufOffsets,
    const BufferRef &renderPassResultBuf,
    const BufferRef &finalResultBuf)
{
    return mVisibilityResultUtils.combineVisibilityResult(
        contextMtl, keepOldValue, renderPassResultBufOffsets, renderPassResultBuf, finalResultBuf);
}

// Compute based mipmap generation
angle::Result RenderUtils::generateMipmapCS(ContextMtl *contextMtl,
                                            const TextureRef &srcTexture,
                                            gl::TexLevelArray<mtl::TextureRef> *mipmapOutputViews)
{
    return mMipmapUtils.generateMipmapCS(contextMtl, srcTexture, mipmapOutputViews);
}

angle::Result RenderUtils::unpackPixelsFromBufferToTexture(ContextMtl *contextMtl,
                                                           const angle::Format &srcAngleFormat,
                                                           const CopyPixelsFromBufferParams &params)
{
    int index = GetPixelTypeIndex(srcAngleFormat);
    return mCopyPixelsUtils[index].unpackPixelsFromBufferToTexture(contextMtl, srcAngleFormat,
                                                                   params);
}
angle::Result RenderUtils::packPixelsFromTextureToBuffer(ContextMtl *contextMtl,
                                                         const angle::Format &dstAngleFormat,
                                                         const CopyPixelsToBufferParams &params)
{
    int index = GetPixelTypeIndex(dstAngleFormat);
    return mCopyPixelsUtils[index].packPixelsFromTextureToBuffer(contextMtl, dstAngleFormat,
                                                                 params);
}

angle::Result RenderUtils::convertVertexFormatToFloatCS(ContextMtl *contextMtl,
                                                        const angle::Format &srcAngleFormat,
                                                        const VertexFormatConvertParams &params)
{
    return mVertexFormatUtils.convertVertexFormatToFloatCS(contextMtl, srcAngleFormat, params);
}

angle::Result RenderUtils::convertVertexFormatToFloatVS(const gl::Context *context,
                                                        RenderCommandEncoder *encoder,
                                                        const angle::Format &srcAngleFormat,
                                                        const VertexFormatConvertParams &params)
{
    return mVertexFormatUtils.convertVertexFormatToFloatVS(context, encoder, srcAngleFormat,
                                                           params);
}

// Expand number of components per vertex's attribute
angle::Result RenderUtils::expandVertexFormatComponentsCS(ContextMtl *contextMtl,
                                                          const angle::Format &srcAngleFormat,
                                                          const VertexFormatConvertParams &params)
{
    return mVertexFormatUtils.expandVertexFormatComponentsCS(contextMtl, srcAngleFormat, params);
}

angle::Result RenderUtils::expandVertexFormatComponentsVS(const gl::Context *context,
                                                          RenderCommandEncoder *encoder,
                                                          const angle::Format &srcAngleFormat,
                                                          const VertexFormatConvertParams &params)
{
    return mVertexFormatUtils.expandVertexFormatComponentsVS(context, encoder, srcAngleFormat,
                                                             params);
}

// ClearUtils implementation
ClearUtils::ClearUtils(const std::string &fragmentShaderName)
    : mFragmentShaderName(fragmentShaderName)
{}

ClearUtils::ClearUtils(const ClearUtils &src) : ClearUtils(src.mFragmentShaderName) {}

void ClearUtils::onDestroy()
{
    ClearRenderPipelineCacheArray(&mClearRenderPipelineCache);
}

void ClearUtils::ensureRenderPipelineStateInitialized(ContextMtl *ctx, uint32_t numOutputs)
{
    RenderPipelineCache &cache = mClearRenderPipelineCache[numOutputs];
    if (cache.getVertexShader() && cache.getFragmentShader())
    {
        // Already initialized.
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        NSError *err       = nil;
        auto shaderLib     = ctx->getDisplay()->getDefaultShadersLib();
        auto vertexShader  = [[shaderLib newFunctionWithName:@"clearVS"] ANGLE_MTL_AUTORELEASE];
        auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

        // Create clear shader pipeline cache for each number of color outputs.
        // So clear k color outputs will use mClearRenderPipelineCache[k] for example:
        [funcConstants setConstantValue:&numOutputs
                                   type:MTLDataTypeUInt
                               withName:NUM_COLOR_OUTPUTS_CONSTANT_NAME];

        auto fragmentShader = [[shaderLib
            newFunctionWithName:[NSString stringWithUTF8String:mFragmentShaderName.c_str()]
                 constantValues:funcConstants
                          error:&err] ANGLE_MTL_AUTORELEASE];
        ASSERT(fragmentShader);

        cache.setVertexShader(ctx, vertexShader);
        cache.setFragmentShader(ctx, fragmentShader);
    }
}

id<MTLDepthStencilState> ClearUtils::getClearDepthStencilState(const gl::Context *context,
                                                               const ClearRectParams &params)
{
    ContextMtl *contextMtl = GetImpl(context);

    if (!params.clearDepth.valid() && !params.clearStencil.valid())
    {
        // Doesn't clear depth nor stencil
        return contextMtl->getDisplay()->getStateCache().getNullDepthStencilState(contextMtl);
    }

    DepthStencilDesc desc;
    desc.reset();

    if (params.clearDepth.valid())
    {
        // Clear depth state
        desc.depthWriteEnabled = true;
    }
    else
    {
        desc.depthWriteEnabled = false;
    }

    if (params.clearStencil.valid())
    {
        // Clear stencil state
        desc.frontFaceStencil.depthStencilPassOperation = MTLStencilOperationReplace;
        desc.frontFaceStencil.writeMask                 = contextMtl->getStencilMask();
        desc.backFaceStencil.depthStencilPassOperation  = MTLStencilOperationReplace;
        desc.backFaceStencil.writeMask                  = contextMtl->getStencilMask();
    }

    return contextMtl->getDisplay()->getStateCache().getDepthStencilState(
        contextMtl->getMetalDevice(), desc);
}

id<MTLRenderPipelineState> ClearUtils::getClearRenderPipelineState(const gl::Context *context,
                                                                   RenderCommandEncoder *cmdEncoder,
                                                                   const ClearRectParams &params)
{
    ContextMtl *contextMtl = GetImpl(context);
    // The color mask to be applied to every color attachment:
    MTLColorWriteMask globalColorMask = params.clearColorMask;
    if (!params.clearColor.valid())
    {
        globalColorMask = MTLColorWriteMaskNone;
    }

    RenderPipelineDesc pipelineDesc;
    const RenderPassDesc &renderPassDesc = cmdEncoder->renderPassDesc();

    renderPassDesc.populateRenderPipelineOutputDesc(globalColorMask,
                                                    &pipelineDesc.outputDescriptor);

    // Disable clear for some outputs that are not enabled
    pipelineDesc.outputDescriptor.updateEnabledDrawBuffers(params.enabledBuffers);

    pipelineDesc.inputPrimitiveTopology = kPrimitiveTopologyClassTriangle;

    ensureRenderPipelineStateInitialized(contextMtl, renderPassDesc.numColorAttachments);
    RenderPipelineCache &cache = mClearRenderPipelineCache[renderPassDesc.numColorAttachments];

    return cache.getRenderPipelineState(contextMtl, pipelineDesc);
}

void ClearUtils::setupClearWithDraw(const gl::Context *context,
                                    RenderCommandEncoder *cmdEncoder,
                                    const ClearRectParams &params)
{
    // Generate render pipeline state
    auto renderPipelineState = getClearRenderPipelineState(context, cmdEncoder, params);
    ASSERT(renderPipelineState);
    // Setup states
    SetupFullscreenQuadDrawCommonStates(cmdEncoder);
    cmdEncoder->setRenderPipelineState(renderPipelineState);

    id<MTLDepthStencilState> dsState = getClearDepthStencilState(context, params);
    cmdEncoder->setDepthStencilState(dsState).setStencilRefVal(params.clearStencil.value());

    // Viewports
    MTLViewport viewport;
    MTLScissorRect scissorRect;

    viewport = GetViewport(params.clearArea, params.dstTextureSize.height, params.flipY);

    scissorRect = GetScissorRect(params.clearArea, params.dstTextureSize.height, params.flipY);

    cmdEncoder->setViewport(viewport);
    cmdEncoder->setScissorRect(scissorRect);

    // uniform
    ClearParamsUniform uniformParams;
    // ClearColorValue is an int, uint, float union so it's safe to use only floats.
    // The Shader will do the bit cast based on appropriate format type.
    uniformParams.clearColor[0] = params.clearColor.value().red;
    uniformParams.clearColor[1] = params.clearColor.value().green;
    uniformParams.clearColor[2] = params.clearColor.value().blue;
    uniformParams.clearColor[3] = params.clearColor.value().alpha;
    uniformParams.clearDepth    = params.clearDepth.value();

    cmdEncoder->setVertexData(uniformParams, 0);
    cmdEncoder->setFragmentData(uniformParams, 0);
}

angle::Result ClearUtils::clearWithDraw(const gl::Context *context,
                                        RenderCommandEncoder *cmdEncoder,
                                        const ClearRectParams &params)
{
    auto overridedParams = params;
    // Make sure we don't clear attachment that doesn't exist
    const RenderPassDesc &renderPassDesc = cmdEncoder->renderPassDesc();
    if (renderPassDesc.numColorAttachments == 0)
    {
        overridedParams.clearColor.reset();
    }
    if (!renderPassDesc.depthAttachment.texture())
    {
        overridedParams.clearDepth.reset();
    }
    if (!renderPassDesc.stencilAttachment.texture())
    {
        overridedParams.clearStencil.reset();
    }

    if (!overridedParams.clearColor.valid() && !overridedParams.clearDepth.valid() &&
        !overridedParams.clearStencil.valid())
    {
        return angle::Result::Continue;
    }
    auto contextMtl = GetImpl(context);
    setupClearWithDraw(context, cmdEncoder, overridedParams);

    angle::Result result;
    {
        // Need to disable occlusion query, otherwise clearing will affect the occlusion counting
        ScopedDisableOcclusionQuery disableOcclusionQuery(contextMtl, cmdEncoder, &result);
        // Draw the screen aligned triangle
        cmdEncoder->draw(MTLPrimitiveTypeTriangle, 0, 3);
    }

    // Invalidate current context's state
    contextMtl->invalidateState(context);

    return result;
}

// ColorBlitUtils implementation
ColorBlitUtils::ColorBlitUtils(const std::string &fragmentShaderName)
    : mFragmentShaderName(fragmentShaderName)
{}

ColorBlitUtils::ColorBlitUtils(const ColorBlitUtils &src) : ColorBlitUtils(src.mFragmentShaderName)
{}

void ColorBlitUtils::onDestroy()
{
    ClearRenderPipelineCache2DArray(&mBlitRenderPipelineCache);
    ClearRenderPipelineCache2DArray(&mBlitPremultiplyAlphaRenderPipelineCache);
    ClearRenderPipelineCache2DArray(&mBlitUnmultiplyAlphaRenderPipelineCache);
}

void ColorBlitUtils::ensureRenderPipelineStateInitialized(ContextMtl *ctx,
                                                          uint32_t numOutputs,
                                                          int alphaPremultiplyType,
                                                          int textureType,
                                                          RenderPipelineCache *cacheOut)
{
    RenderPipelineCache &pipelineCache = *cacheOut;
    if (pipelineCache.getVertexShader() && pipelineCache.getFragmentShader())
    {
        // Already initialized.
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        NSError *err       = nil;
        auto shaderLib     = ctx->getDisplay()->getDefaultShadersLib();
        auto vertexShader  = [[shaderLib newFunctionWithName:@"blitVS"] ANGLE_MTL_AUTORELEASE];
        auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

        constexpr BOOL multiplyAlphaFlags[][2] = {// premultiply, unmultiply

                                                  // Normal blit
                                                  {NO, NO},
                                                  // Blit premultiply-alpha
                                                  {YES, NO},
                                                  // Blit unmultiply alpha
                                                  {NO, YES}};

        // Set alpha multiply flags
        [funcConstants setConstantValue:&multiplyAlphaFlags[alphaPremultiplyType][0]
                                   type:MTLDataTypeBool
                               withName:PREMULTIPLY_ALPHA_CONSTANT_NAME];
        [funcConstants setConstantValue:&multiplyAlphaFlags[alphaPremultiplyType][1]
                                   type:MTLDataTypeBool
                               withName:UNMULTIPLY_ALPHA_CONSTANT_NAME];

        // We create blit shader pipeline cache for each number of color outputs.
        // So blit k color outputs will use mBlitRenderPipelineCache[k-1] for example:
        [funcConstants setConstantValue:&numOutputs
                                   type:MTLDataTypeUInt
                               withName:NUM_COLOR_OUTPUTS_CONSTANT_NAME];

        // Set texture type constant
        [funcConstants setConstantValue:&textureType
                                   type:MTLDataTypeInt
                               withName:SOURCE_TEXTURE_TYPE_CONSTANT_NAME];

        auto fragmentShader = [[shaderLib
            newFunctionWithName:[NSString stringWithUTF8String:mFragmentShaderName.c_str()]
                 constantValues:funcConstants
                          error:&err] ANGLE_MTL_AUTORELEASE];

        ASSERT(fragmentShader);
        pipelineCache.setVertexShader(ctx, vertexShader);
        pipelineCache.setFragmentShader(ctx, fragmentShader);
    }
}

id<MTLRenderPipelineState> ColorBlitUtils::getColorBlitRenderPipelineState(
    const gl::Context *context,
    RenderCommandEncoder *cmdEncoder,
    const ColorBlitParams &params)
{
    ContextMtl *contextMtl = GetImpl(context);
    RenderPipelineDesc pipelineDesc;
    const RenderPassDesc &renderPassDesc = cmdEncoder->renderPassDesc();

    renderPassDesc.populateRenderPipelineOutputDesc(params.blitColorMask,
                                                    &pipelineDesc.outputDescriptor);

    // Disable blit for some outputs that are not enabled
    pipelineDesc.outputDescriptor.updateEnabledDrawBuffers(params.enabledBuffers);

    pipelineDesc.inputPrimitiveTopology = kPrimitiveTopologyClassTriangle;

    RenderPipelineCache *pipelineCache;
    int alphaPremultiplyType;
    uint32_t nOutputIndex = renderPassDesc.numColorAttachments - 1;
    int textureType       = GetShaderTextureType(params.src);
    if (params.unpackPremultiplyAlpha == params.unpackUnmultiplyAlpha)
    {
        alphaPremultiplyType = 0;
        pipelineCache        = &mBlitRenderPipelineCache[nOutputIndex][textureType];
    }
    else if (params.unpackPremultiplyAlpha)
    {
        alphaPremultiplyType = 1;
        pipelineCache        = &mBlitPremultiplyAlphaRenderPipelineCache[nOutputIndex][textureType];
    }
    else
    {
        alphaPremultiplyType = 2;
        pipelineCache        = &mBlitUnmultiplyAlphaRenderPipelineCache[nOutputIndex][textureType];
    }

    ensureRenderPipelineStateInitialized(contextMtl, renderPassDesc.numColorAttachments,
                                         alphaPremultiplyType, textureType, pipelineCache);

    return pipelineCache->getRenderPipelineState(contextMtl, pipelineDesc);
}

void ColorBlitUtils::setupColorBlitWithDraw(const gl::Context *context,
                                            RenderCommandEncoder *cmdEncoder,
                                            const ColorBlitParams &params)
{
    ASSERT(cmdEncoder->renderPassDesc().numColorAttachments >= 1 && params.src);

    ContextMtl *contextMtl = mtl::GetImpl(context);

    // Generate render pipeline state
    auto renderPipelineState = getColorBlitRenderPipelineState(context, cmdEncoder, params);
    ASSERT(renderPipelineState);
    // Setup states
    cmdEncoder->setRenderPipelineState(renderPipelineState);
    cmdEncoder->setDepthStencilState(
        contextMtl->getDisplay()->getStateCache().getNullDepthStencilState(contextMtl));

    SetupCommonBlitWithDrawStates(context, cmdEncoder, params, true);

    // Set sampler state
    SamplerDesc samplerDesc;
    samplerDesc.reset();
    samplerDesc.minFilter = samplerDesc.magFilter = GetFilter(params.filter);

    cmdEncoder->setFragmentSamplerState(contextMtl->getDisplay()->getStateCache().getSamplerState(
                                            contextMtl->getMetalDevice(), samplerDesc),
                                        0, FLT_MAX, 0);
}

angle::Result ColorBlitUtils::blitColorWithDraw(const gl::Context *context,
                                                RenderCommandEncoder *cmdEncoder,
                                                const ColorBlitParams &params)
{
    if (!params.src)
    {
        return angle::Result::Continue;
    }
    ContextMtl *contextMtl = GetImpl(context);
    setupColorBlitWithDraw(context, cmdEncoder, params);

    angle::Result result;
    {
        // Need to disable occlusion query, otherwise blitting will affect the occlusion counting
        ScopedDisableOcclusionQuery disableOcclusionQuery(contextMtl, cmdEncoder, &result);
        // Draw the screen aligned triangle
        cmdEncoder->draw(MTLPrimitiveTypeTriangle, 0, 3);
    }

    // Invalidate current context's state
    contextMtl->invalidateState(context);

    return result;
}

// DepthStencilBlitUtils implementation
void DepthStencilBlitUtils::onDestroy()
{
    ClearRenderPipelineCacheArray(&mDepthBlitRenderPipelineCache);
    ClearRenderPipelineCacheArray(&mStencilBlitRenderPipelineCache);
    ClearRenderPipelineCache2DArray(&mDepthStencilBlitRenderPipelineCache);

    ClearPipelineStateArray(&mStencilBlitToBufferComPipelineCache);

    mStencilCopyBuffer = nullptr;
}

void DepthStencilBlitUtils::ensureRenderPipelineStateInitialized(ContextMtl *ctx,
                                                                 int sourceDepthTextureType,
                                                                 int sourceStencilTextureType,
                                                                 RenderPipelineCache *cacheOut)
{
    RenderPipelineCache &cache = *cacheOut;
    if (cache.getVertexShader() && cache.getFragmentShader())
    {
        // Already initialized.
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        NSError *err       = nil;
        auto shaderLib     = ctx->getDisplay()->getDefaultShadersLib();
        auto vertexShader  = [[shaderLib newFunctionWithName:@"blitVS"] ANGLE_MTL_AUTORELEASE];
        auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

        NSString *shaderName;
        if (sourceDepthTextureType != -1 && sourceStencilTextureType != -1)
        {
            shaderName = @"blitDepthStencilFS";
        }
        else if (sourceDepthTextureType != -1)
        {
            shaderName = @"blitDepthFS";
        }
        else
        {
            shaderName = @"blitStencilFS";
        }

        if (sourceDepthTextureType != -1)
        {
            [funcConstants setConstantValue:&sourceDepthTextureType
                                       type:MTLDataTypeInt
                                   withName:SOURCE_TEXTURE_TYPE_CONSTANT_NAME];
        }
        if (sourceStencilTextureType != -1)
        {

            [funcConstants setConstantValue:&sourceStencilTextureType
                                       type:MTLDataTypeInt
                                   withName:SOURCE_TEXTURE2_TYPE_CONSTANT_NAME];
        }

        auto fragmentShader = [[shaderLib newFunctionWithName:shaderName
                                               constantValues:funcConstants
                                                        error:&err] ANGLE_MTL_AUTORELEASE];
        ASSERT(fragmentShader);

        cache.setVertexShader(ctx, vertexShader);
        cache.setFragmentShader(ctx, fragmentShader);
    }
}

id<MTLComputePipelineState> DepthStencilBlitUtils::getStencilToBufferComputePipelineState(
    ContextMtl *contextMtl,
    const StencilBlitViaBufferParams &params)
{
    int sourceStencilTextureType = GetShaderTextureType(params.srcStencil);
    AutoObjCPtr<id<MTLComputePipelineState>> &cache =
        mStencilBlitToBufferComPipelineCache[sourceStencilTextureType];
    if (cache)
    {
        return cache;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

        [funcConstants setConstantValue:&sourceStencilTextureType
                                   type:MTLDataTypeInt
                               withName:SOURCE_TEXTURE2_TYPE_CONSTANT_NAME];

        EnsureSpecializedComputePipelineInitialized(
            contextMtl->getDisplay(), @"blitStencilToBufferCS", funcConstants, &cache);
    }

    return cache;
}

id<MTLRenderPipelineState> DepthStencilBlitUtils::getDepthStencilBlitRenderPipelineState(
    const gl::Context *context,
    RenderCommandEncoder *cmdEncoder,
    const DepthStencilBlitParams &params)
{
    ContextMtl *contextMtl = GetImpl(context);
    RenderPipelineDesc pipelineDesc;
    const RenderPassDesc &renderPassDesc = cmdEncoder->renderPassDesc();

    renderPassDesc.populateRenderPipelineOutputDesc(&pipelineDesc.outputDescriptor);

    // Disable all color outputs
    pipelineDesc.outputDescriptor.updateEnabledDrawBuffers(gl::DrawBufferMask());

    pipelineDesc.inputPrimitiveTopology = kPrimitiveTopologyClassTriangle;

    RenderPipelineCache *pipelineCache;

    int depthTextureType   = GetShaderTextureType(params.src);
    int stencilTextureType = GetShaderTextureType(params.srcStencil);
    if (params.src && params.srcStencil)
    {
        pipelineCache = &mDepthStencilBlitRenderPipelineCache[depthTextureType][stencilTextureType];
    }
    else if (params.src)
    {
        // Only depth blit
        pipelineCache = &mDepthBlitRenderPipelineCache[depthTextureType];
    }
    else
    {
        // Only stencil blit
        pipelineCache = &mStencilBlitRenderPipelineCache[stencilTextureType];
    }

    ensureRenderPipelineStateInitialized(contextMtl, depthTextureType, stencilTextureType,
                                         pipelineCache);

    return pipelineCache->getRenderPipelineState(contextMtl, pipelineDesc);
}

void DepthStencilBlitUtils::setupDepthStencilBlitWithDraw(const gl::Context *context,
                                                          RenderCommandEncoder *cmdEncoder,
                                                          const DepthStencilBlitParams &params)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);

    ASSERT(params.src || params.srcStencil);

    SetupCommonBlitWithDrawStates(context, cmdEncoder, params, false);

    // Generate render pipeline state
    auto renderPipelineState = getDepthStencilBlitRenderPipelineState(context, cmdEncoder, params);
    ASSERT(renderPipelineState);
    // Setup states
    cmdEncoder->setRenderPipelineState(renderPipelineState);

    // Depth stencil state
    mtl::DepthStencilDesc dsStateDesc;
    dsStateDesc.reset();
    dsStateDesc.depthCompareFunction = MTLCompareFunctionAlways;

    if (params.src)
    {
        // Enable depth write
        dsStateDesc.depthWriteEnabled = true;
    }
    else
    {
        // Disable depth write
        dsStateDesc.depthWriteEnabled = false;
    }

    if (params.srcStencil)
    {
        cmdEncoder->setFragmentTexture(params.srcStencil, 1);

        if (!contextMtl->getDisplay()->getFeatures().hasStencilOutput.enabled)
        {
            // Hardware must support stencil writing directly in shader.
            UNREACHABLE();
        }
        // Enable stencil write to framebuffer
        dsStateDesc.frontFaceStencil.stencilCompareFunction = MTLCompareFunctionAlways;
        dsStateDesc.backFaceStencil.stencilCompareFunction  = MTLCompareFunctionAlways;

        dsStateDesc.frontFaceStencil.depthStencilPassOperation = MTLStencilOperationReplace;
        dsStateDesc.backFaceStencil.depthStencilPassOperation  = MTLStencilOperationReplace;

        dsStateDesc.frontFaceStencil.writeMask = kStencilMaskAll;
        dsStateDesc.backFaceStencil.writeMask  = kStencilMaskAll;
    }

    cmdEncoder->setDepthStencilState(contextMtl->getDisplay()->getStateCache().getDepthStencilState(
        contextMtl->getMetalDevice(), dsStateDesc));
}

angle::Result DepthStencilBlitUtils::blitDepthStencilWithDraw(const gl::Context *context,
                                                              RenderCommandEncoder *cmdEncoder,
                                                              const DepthStencilBlitParams &params)
{
    if (!params.src && !params.srcStencil)
    {
        return angle::Result::Continue;
    }
    ContextMtl *contextMtl = GetImpl(context);

    setupDepthStencilBlitWithDraw(context, cmdEncoder, params);

    angle::Result result;
    {
        // Need to disable occlusion query, otherwise blitting will affect the occlusion counting
        ScopedDisableOcclusionQuery disableOcclusionQuery(contextMtl, cmdEncoder, &result);
        // Draw the screen aligned triangle
        cmdEncoder->draw(MTLPrimitiveTypeTriangle, 0, 3);
    }

    // Invalidate current context's state
    contextMtl->invalidateState(context);

    return result;
}

angle::Result DepthStencilBlitUtils::blitStencilViaCopyBuffer(
    const gl::Context *context,
    const StencilBlitViaBufferParams &params)
{
    // Depth texture must be omitted.
    ASSERT(!params.src);
    if (!params.srcStencil || !params.dstStencil)
    {
        return angle::Result::Continue;
    }
    ContextMtl *contextMtl = GetImpl(context);

    // Create intermediate buffer.
    uint32_t bufferRequiredRowPitch =
        static_cast<uint32_t>(params.dstRect.width) * params.dstStencil->samples();
    uint32_t bufferRequiredSize =
        bufferRequiredRowPitch * static_cast<uint32_t>(params.dstRect.height);
    if (!mStencilCopyBuffer || mStencilCopyBuffer->size() < bufferRequiredSize)
    {
        ANGLE_TRY(Buffer::MakeBuffer(contextMtl, bufferRequiredSize, nullptr, &mStencilCopyBuffer));
    }

    // Copy stencil data to buffer via compute shader
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    id<MTLComputePipelineState> pipeline =
        getStencilToBufferComputePipelineState(contextMtl, params);

    cmdEncoder->setComputePipelineState(pipeline);

    uint32_t srcWidth  = params.srcStencil->width(params.srcLevel);
    uint32_t srcHeight = params.srcStencil->height(params.srcLevel);

    float u0, v0, u1, v1;
    bool unpackFlipX = params.unpackFlipX;
    bool unpackFlipY = params.unpackFlipY;
    if (params.dstFlipX)
    {
        unpackFlipX = !unpackFlipX;
    }
    if (params.dstFlipY)
    {
        unpackFlipY = !unpackFlipY;
    }
    GetBlitTexCoords(srcWidth, srcHeight, params.srcRect, params.srcYFlipped, unpackFlipX,
                     unpackFlipY, &u0, &v0, &u1, &v1);

    BlitStencilToBufferParamsUniform uniform;
    uniform.srcTexCoordSteps[0]  = (u1 - u0) / params.dstRect.width;
    uniform.srcTexCoordSteps[1]  = (v1 - v0) / params.dstRect.height;
    uniform.srcStartTexCoords[0] = u0 + uniform.srcTexCoordSteps[0] * 0.5f;
    uniform.srcStartTexCoords[1] = v0 + uniform.srcTexCoordSteps[1] * 0.5f;
    uniform.srcLevel             = params.srcStencilLevel;
    uniform.srcLayer             = params.srcStencilLayer;
    uniform.dstSize[0]           = params.dstRect.width;
    uniform.dstSize[1]           = params.dstRect.height;
    uniform.dstBufferRowPitch    = bufferRequiredRowPitch;
    uniform.resolveMS            = params.dstStencil->samples() == 1;

    cmdEncoder->setTexture(params.srcStencil, 1);

    cmdEncoder->setData(uniform, 0);
    cmdEncoder->setBufferForWrite(mStencilCopyBuffer, 0, 1);

    NSUInteger w                  = pipeline.threadExecutionWidth;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, 1, 1);
    DispatchCompute(contextMtl, cmdEncoder, /** allowNonUniform */ true,
                    MTLSizeMake(params.dstRect.width, params.dstRect.height, 1),
                    threadsPerThreadgroup);

    // Copy buffer to real destination texture
    ASSERT(params.dstStencil->textureType() != MTLTextureType3D);

    mtl::BlitCommandEncoder *blitEncoder = contextMtl->getBlitCommandEncoder();

    // Only copy the scissored area of the buffer.
    MTLScissorRect viewportRectMtl =
        GetScissorRect(params.dstRect, params.dstTextureSize.height, params.dstFlipY);
    MTLScissorRect scissorRectMtl =
        GetScissorRect(params.dstScissorRect, params.dstTextureSize.height, params.dstFlipY);

    uint32_t dx = static_cast<uint32_t>(scissorRectMtl.x - viewportRectMtl.x);
    uint32_t dy = static_cast<uint32_t>(scissorRectMtl.y - viewportRectMtl.y);

    uint32_t bufferStartReadableOffset = dx + bufferRequiredRowPitch * dy;
    blitEncoder->copyBufferToTexture(
        mStencilCopyBuffer, bufferStartReadableOffset, bufferRequiredRowPitch, 0,
        MTLSizeMake(scissorRectMtl.width, scissorRectMtl.height, 1), params.dstStencil,
        params.dstStencilLayer, params.dstStencilLevel,
        MTLOriginMake(scissorRectMtl.x, scissorRectMtl.y, 0),
        params.dstPackedDepthStencilFormat ? MTLBlitOptionStencilFromDepthStencil
                                           : MTLBlitOptionNone);

    return angle::Result::Continue;
}

// IndexGeneratorUtils implementation
void IndexGeneratorUtils::onDestroy()
{
    ClearPipelineState2DArray(&mIndexConversionPipelineCaches);
    ClearPipelineState2DArray(&mTriFanFromElemArrayGeneratorPipelineCaches);
    ClearPipelineState2DArray(&mLineLoopFromElemArrayGeneratorPipelineCaches);

    mTriFanFromArraysGeneratorPipeline   = nil;
    mLineLoopFromArraysGeneratorPipeline = nil;
}

AutoObjCPtr<id<MTLComputePipelineState>> IndexGeneratorUtils::getIndexConversionPipeline(
    ContextMtl *contextMtl,
    gl::DrawElementsType srcType,
    uint32_t srcOffset)
{
    size_t elementSize = gl::GetDrawElementsTypeSize(srcType);
    BOOL aligned       = (srcOffset % elementSize) == 0;
    int srcTypeKey     = static_cast<int>(srcType);
    auto &cache        = mIndexConversionPipelineCaches[srcTypeKey][aligned ? 1 : 0];

    if (!cache)
    {
        ANGLE_MTL_OBJC_SCOPE
        {
            auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

            [funcConstants setConstantValue:&aligned
                                       type:MTLDataTypeBool
                                   withName:SOURCE_BUFFER_ALIGNED_CONSTANT_NAME];

            NSString *shaderName = nil;
            switch (srcType)
            {
                case gl::DrawElementsType::UnsignedByte:
                    // No need for specialized shader
                    funcConstants = nil;
                    shaderName    = @"convertIndexU8ToU16";
                    break;
                case gl::DrawElementsType::UnsignedShort:
                    shaderName = @"convertIndexU16";
                    break;
                case gl::DrawElementsType::UnsignedInt:
                    shaderName = @"convertIndexU32";
                    break;
                default:
                    UNREACHABLE();
            }

            EnsureSpecializedComputePipelineInitialized(contextMtl->getDisplay(), shaderName,
                                                        funcConstants, &cache);
        }
    }

    return cache;
}

AutoObjCPtr<id<MTLComputePipelineState>>
IndexGeneratorUtils::getIndicesFromElemArrayGeneratorPipeline(
    ContextMtl *contextMtl,
    gl::DrawElementsType srcType,
    uint32_t srcOffset,
    NSString *shaderName,
    IndexConversionPipelineArray *pipelineCacheArray)
{
    size_t elementSize = gl::GetDrawElementsTypeSize(srcType);
    BOOL aligned       = (srcOffset % elementSize) == 0;
    int srcTypeKey     = static_cast<int>(srcType);

    auto &cache = (*pipelineCacheArray)[srcTypeKey][aligned ? 1 : 0];

    if (!cache)
    {
        ANGLE_MTL_OBJC_SCOPE
        {
            auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

            bool isU8  = false;
            bool isU16 = false;
            bool isU32 = false;

            switch (srcType)
            {
                case gl::DrawElementsType::UnsignedByte:
                    isU8 = true;
                    break;
                case gl::DrawElementsType::UnsignedShort:
                    isU16 = true;
                    break;
                case gl::DrawElementsType::UnsignedInt:
                    isU32 = true;
                    break;
                default:
                    UNREACHABLE();
            }

            [funcConstants setConstantValue:&aligned
                                       type:MTLDataTypeBool
                                   withName:SOURCE_BUFFER_ALIGNED_CONSTANT_NAME];
            [funcConstants setConstantValue:&isU8
                                       type:MTLDataTypeBool
                                   withName:SOURCE_IDX_IS_U8_CONSTANT_NAME];
            [funcConstants setConstantValue:&isU16
                                       type:MTLDataTypeBool
                                   withName:SOURCE_IDX_IS_U16_CONSTANT_NAME];
            [funcConstants setConstantValue:&isU32
                                       type:MTLDataTypeBool
                                   withName:SOURCE_IDX_IS_U32_CONSTANT_NAME];

            EnsureSpecializedComputePipelineInitialized(contextMtl->getDisplay(), shaderName,
                                                        funcConstants, &cache);
        }
    }

    return cache;
}

void IndexGeneratorUtils::ensureTriFanFromArrayGeneratorInitialized(ContextMtl *contextMtl)
{
    EnsureComputePipelineInitialized(contextMtl->getDisplay(), @"genTriFanIndicesFromArray",
                                     &mTriFanFromArraysGeneratorPipeline);
}

void IndexGeneratorUtils::ensureLineLoopFromArrayGeneratorInitialized(ContextMtl *contextMtl)
{
    EnsureComputePipelineInitialized(contextMtl->getDisplay(), @"genLineLoopIndicesFromArray",
                                     &mLineLoopFromArraysGeneratorPipeline);
}

angle::Result IndexGeneratorUtils::convertIndexBufferGPU(ContextMtl *contextMtl,
                                                         const IndexConversionParams &params)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    AutoObjCPtr<id<MTLComputePipelineState>> pipelineState =
        getIndexConversionPipeline(contextMtl, params.srcType, params.srcOffset);

    ASSERT(pipelineState);

    cmdEncoder->setComputePipelineState(pipelineState);

    ASSERT((params.dstOffset % kIndexBufferOffsetAlignment) == 0);

    IndexConversionUniform uniform;
    uniform.srcOffset               = params.srcOffset;
    uniform.indexCount              = params.indexCount;
    uniform.primitiveRestartEnabled = params.primitiveRestartEnabled;

    cmdEncoder->setData(uniform, 0);
    cmdEncoder->setBuffer(params.srcBuffer, 0, 1);
    cmdEncoder->setBufferForWrite(params.dstBuffer, params.dstOffset, 2);

    DispatchCompute(contextMtl, cmdEncoder, pipelineState, params.indexCount);

    return angle::Result::Continue;
}

angle::Result IndexGeneratorUtils::generateTriFanBufferFromArrays(
    ContextMtl *contextMtl,
    const TriFanOrLineLoopFromArrayParams &params)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);
    ensureTriFanFromArrayGeneratorInitialized(contextMtl);

    ASSERT(params.vertexCount > 2);

    cmdEncoder->setComputePipelineState(mTriFanFromArraysGeneratorPipeline);

    ASSERT((params.dstOffset % kIndexBufferOffsetAlignment) == 0);

    TriFanOrLineLoopArrayParams uniform;

    uniform.firstVertex = params.firstVertex;
    uniform.vertexCount = params.vertexCount - 2;

    cmdEncoder->setData(uniform, 0);
    cmdEncoder->setBufferForWrite(params.dstBuffer, params.dstOffset, 2);

    DispatchCompute(contextMtl, cmdEncoder, mTriFanFromArraysGeneratorPipeline,
                    uniform.vertexCount);

    return angle::Result::Continue;
}

angle::Result IndexGeneratorUtils::generateTriFanBufferFromElementsArray(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params)
{
    const gl::VertexArray *vertexArray = contextMtl->getState().getVertexArray();
    const gl::Buffer *elementBuffer    = vertexArray->getElementArrayBuffer();
    if (elementBuffer)
    {
        BufferMtl *elementBufferMtl = GetImpl(elementBuffer);
        size_t srcOffset            = reinterpret_cast<size_t>(params.indices);
        ANGLE_CHECK(contextMtl, srcOffset <= std::numeric_limits<uint32_t>::max(),
                    "Index offset is too large", GL_INVALID_VALUE);
        if (params.primitiveRestartEnabled ||
            (!contextMtl->getDisplay()->getFeatures().breakRenderPassIsCheap.enabled &&
             contextMtl->getRenderCommandEncoder()))
        {
            IndexGenerationParams cpuPathParams = params;
            cpuPathParams.indices =
                elementBufferMtl->getClientShadowCopyData(contextMtl) + srcOffset;
            return generateTriFanBufferFromElementsArrayCPU(contextMtl, cpuPathParams);
        }
        else
        {
            return generateTriFanBufferFromElementsArrayGPU(
                contextMtl, params.srcType, params.indexCount, elementBufferMtl->getCurrentBuffer(),
                static_cast<uint32_t>(srcOffset), params.dstBuffer, params.dstOffset);
        }
    }
    else
    {
        return generateTriFanBufferFromElementsArrayCPU(contextMtl, params);
    }
}

angle::Result IndexGeneratorUtils::generateTriFanBufferFromElementsArrayGPU(
    ContextMtl *contextMtl,
    gl::DrawElementsType srcType,
    uint32_t indexCount,
    const BufferRef &srcBuffer,
    uint32_t srcOffset,
    const BufferRef &dstBuffer,
    // Must be multiples of kIndexBufferOffsetAlignment
    uint32_t dstOffset)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    AutoObjCPtr<id<MTLComputePipelineState>> pipelineState =
        getIndicesFromElemArrayGeneratorPipeline(contextMtl, srcType, srcOffset,
                                                 @"genTriFanIndicesFromElements",
                                                 &mTriFanFromElemArrayGeneratorPipelineCaches);

    ASSERT(pipelineState);

    cmdEncoder->setComputePipelineState(pipelineState);

    ASSERT((dstOffset % kIndexBufferOffsetAlignment) == 0);
    ASSERT(indexCount > 2);

    IndexConversionUniform uniform;
    uniform.srcOffset  = srcOffset;
    uniform.indexCount = indexCount - 2;  // Only start from the 3rd element.

    cmdEncoder->setData(uniform, 0);
    cmdEncoder->setBuffer(srcBuffer, 0, 1);
    cmdEncoder->setBufferForWrite(dstBuffer, dstOffset, 2);

    DispatchCompute(contextMtl, cmdEncoder, pipelineState, uniform.indexCount);

    return angle::Result::Continue;
}

angle::Result IndexGeneratorUtils::generateTriFanBufferFromElementsArrayCPU(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params)
{
    switch (params.srcType)
    {
        case gl::DrawElementsType::UnsignedByte:
            return GenTriFanFromClientElements(
                contextMtl, params.indexCount, params.primitiveRestartEnabled,
                static_cast<const uint8_t *>(params.indices), params.dstBuffer, params.dstOffset);
        case gl::DrawElementsType::UnsignedShort:
            return GenTriFanFromClientElements(
                contextMtl, params.indexCount, params.primitiveRestartEnabled,
                static_cast<const uint16_t *>(params.indices), params.dstBuffer, params.dstOffset);
        case gl::DrawElementsType::UnsignedInt:
            return GenTriFanFromClientElements(
                contextMtl, params.indexCount, params.primitiveRestartEnabled,
                static_cast<const uint32_t *>(params.indices), params.dstBuffer, params.dstOffset);
        default:
            UNREACHABLE();
    }

    return angle::Result::Stop;
}

angle::Result IndexGeneratorUtils::generateLineLoopBufferFromArrays(
    ContextMtl *contextMtl,
    const TriFanOrLineLoopFromArrayParams &params)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);
    ensureLineLoopFromArrayGeneratorInitialized(contextMtl);

    cmdEncoder->setComputePipelineState(mLineLoopFromArraysGeneratorPipeline);

    ASSERT((params.dstOffset % kIndexBufferOffsetAlignment) == 0);

    TriFanOrLineLoopArrayParams uniform;

    uniform.firstVertex = params.firstVertex;
    uniform.vertexCount = params.vertexCount;

    cmdEncoder->setData(uniform, 0);
    cmdEncoder->setBufferForWrite(params.dstBuffer, params.dstOffset, 2);

    DispatchCompute(contextMtl, cmdEncoder, mLineLoopFromArraysGeneratorPipeline,
                    uniform.vertexCount + 1);

    return angle::Result::Continue;
}

angle::Result IndexGeneratorUtils::generateLineLoopBufferFromElementsArray(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params,
    uint32_t *indicesGenerated)
{
    const gl::VertexArray *vertexArray = contextMtl->getState().getVertexArray();
    const gl::Buffer *elementBuffer    = vertexArray->getElementArrayBuffer();
    if (elementBuffer)
    {
        BufferMtl *elementBufferMtl = GetImpl(elementBuffer);
        size_t srcOffset            = reinterpret_cast<size_t>(params.indices);
        ANGLE_CHECK(contextMtl, srcOffset <= std::numeric_limits<uint32_t>::max(),
                    "Index offset is too large", GL_INVALID_VALUE);
        if (params.primitiveRestartEnabled ||
            (!contextMtl->getDisplay()->getFeatures().breakRenderPassIsCheap.enabled &&
             contextMtl->getRenderCommandEncoder()))
        {
            IndexGenerationParams cpuPathParams = params;
            cpuPathParams.indices =
                elementBufferMtl->getClientShadowCopyData(contextMtl) + srcOffset;
            return generateLineLoopBufferFromElementsArrayCPU(contextMtl, cpuPathParams,
                                                              indicesGenerated);
        }
        else
        {
            *indicesGenerated = params.indexCount + 1;
            return generateLineLoopBufferFromElementsArrayGPU(
                contextMtl, params.srcType, params.indexCount, elementBufferMtl->getCurrentBuffer(),
                static_cast<uint32_t>(srcOffset), params.dstBuffer, params.dstOffset);
        }
    }
    else
    {
        return generateLineLoopBufferFromElementsArrayCPU(contextMtl, params, indicesGenerated);
    }
}

angle::Result IndexGeneratorUtils::generateLineLoopBufferFromElementsArrayGPU(
    ContextMtl *contextMtl,
    gl::DrawElementsType srcType,
    uint32_t indexCount,
    const BufferRef &srcBuffer,
    uint32_t srcOffset,
    const BufferRef &dstBuffer,
    // Must be multiples of kIndexBufferOffsetAlignment
    uint32_t dstOffset)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    AutoObjCPtr<id<MTLComputePipelineState>> pipelineState =
        getIndicesFromElemArrayGeneratorPipeline(contextMtl, srcType, srcOffset,
                                                 @"genLineLoopIndicesFromElements",
                                                 &mLineLoopFromElemArrayGeneratorPipelineCaches);

    ASSERT(pipelineState);

    cmdEncoder->setComputePipelineState(pipelineState);

    ASSERT((dstOffset % kIndexBufferOffsetAlignment) == 0);
    ASSERT(indexCount >= 2);

    IndexConversionUniform uniform;
    uniform.srcOffset  = srcOffset;
    uniform.indexCount = indexCount;

    cmdEncoder->setData(uniform, 0);
    cmdEncoder->setBuffer(srcBuffer, 0, 1);
    cmdEncoder->setBufferForWrite(dstBuffer, dstOffset, 2);

    DispatchCompute(contextMtl, cmdEncoder, pipelineState, uniform.indexCount + 1);

    return angle::Result::Continue;
}

angle::Result IndexGeneratorUtils::generateLineLoopBufferFromElementsArrayCPU(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params,
    uint32_t *indicesGenerated)
{
    switch (params.srcType)
    {
        case gl::DrawElementsType::UnsignedByte:
            return GenLineLoopFromClientElements(
                contextMtl, params.indexCount, params.primitiveRestartEnabled,
                static_cast<const uint8_t *>(params.indices), params.dstBuffer, params.dstOffset,
                indicesGenerated);
        case gl::DrawElementsType::UnsignedShort:
            return GenLineLoopFromClientElements(
                contextMtl, params.indexCount, params.primitiveRestartEnabled,
                static_cast<const uint16_t *>(params.indices), params.dstBuffer, params.dstOffset,
                indicesGenerated);
        case gl::DrawElementsType::UnsignedInt:
            return GenLineLoopFromClientElements(
                contextMtl, params.indexCount, params.primitiveRestartEnabled,
                static_cast<const uint32_t *>(params.indices), params.dstBuffer, params.dstOffset,
                indicesGenerated);
        default:
            UNREACHABLE();
    }

    return angle::Result::Stop;
}

angle::Result IndexGeneratorUtils::generateLineLoopLastSegment(ContextMtl *contextMtl,
                                                               uint32_t firstVertex,
                                                               uint32_t lastVertex,
                                                               const BufferRef &dstBuffer,
                                                               uint32_t dstOffset)
{
    uint8_t *ptr = dstBuffer->map(contextMtl) + dstOffset;

    uint32_t indices[2] = {lastVertex, firstVertex};
    memcpy(ptr, indices, sizeof(indices));

    dstBuffer->unmap(contextMtl, dstOffset, sizeof(indices));

    return angle::Result::Continue;
}

angle::Result IndexGeneratorUtils::generateLineLoopLastSegmentFromElementsArray(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params)
{
    ASSERT(!params.primitiveRestartEnabled);
    const gl::VertexArray *vertexArray = contextMtl->getState().getVertexArray();
    const gl::Buffer *elementBuffer    = vertexArray->getElementArrayBuffer();
    if (elementBuffer)
    {
        size_t srcOffset = reinterpret_cast<size_t>(params.indices);
        ANGLE_CHECK(contextMtl, srcOffset <= std::numeric_limits<uint32_t>::max(),
                    "Index offset is too large", GL_INVALID_VALUE);

        BufferMtl *bufferMtl = GetImpl(elementBuffer);
        std::pair<uint32_t, uint32_t> firstLast;
        ANGLE_TRY(bufferMtl->getFirstLastIndices(contextMtl, params.srcType,
                                                 static_cast<uint32_t>(srcOffset),
                                                 params.indexCount, &firstLast));

        return generateLineLoopLastSegment(contextMtl, firstLast.first, firstLast.second,
                                           params.dstBuffer, params.dstOffset);
    }
    else
    {
        return generateLineLoopLastSegmentFromElementsArrayCPU(contextMtl, params);
    }
}

angle::Result IndexGeneratorUtils::generateLineLoopLastSegmentFromElementsArrayCPU(
    ContextMtl *contextMtl,
    const IndexGenerationParams &params)
{
    ASSERT(!params.primitiveRestartEnabled);

    uint32_t first, last;

    switch (params.srcType)
    {
        case gl::DrawElementsType::UnsignedByte:
            GetFirstLastIndicesFromClientElements(
                params.indexCount, static_cast<const uint8_t *>(params.indices), &first, &last);
            break;
        case gl::DrawElementsType::UnsignedShort:
            GetFirstLastIndicesFromClientElements(
                params.indexCount, static_cast<const uint16_t *>(params.indices), &first, &last);
            break;
        case gl::DrawElementsType::UnsignedInt:
            GetFirstLastIndicesFromClientElements(
                params.indexCount, static_cast<const uint32_t *>(params.indices), &first, &last);
            break;
        default:
            UNREACHABLE();
            return angle::Result::Stop;
    }

    return generateLineLoopLastSegment(contextMtl, first, last, params.dstBuffer, params.dstOffset);
}

// VisibilityResultUtils implementation
void VisibilityResultUtils::onDestroy()
{
    ClearPipelineStateArray(&mVisibilityResultCombPipelines);
}

AutoObjCPtr<id<MTLComputePipelineState>> VisibilityResultUtils::getVisibilityResultCombPipeline(
    ContextMtl *contextMtl,
    bool keepOldValue)
{
    // There is no guarantee Objective-C's BOOL is equal to bool, so casting just in case.
    BOOL keepOldValueVal = keepOldValue;
    AutoObjCPtr<id<MTLComputePipelineState>> &cache =
        mVisibilityResultCombPipelines[keepOldValueVal];
    if (cache)
    {
        return cache;
    }
    ANGLE_MTL_OBJC_SCOPE
    {
        auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

        [funcConstants setConstantValue:&keepOldValueVal
                                   type:MTLDataTypeBool
                               withName:VISIBILITY_RESULT_KEEP_OLD_VAL_CONSTANT_NAME];

        EnsureSpecializedComputePipelineInitialized(
            contextMtl->getDisplay(), @"combineVisibilityResult", funcConstants, &cache);
    }

    return cache;
}

void VisibilityResultUtils::combineVisibilityResult(
    ContextMtl *contextMtl,
    bool keepOldValue,
    const VisibilityBufferOffsetsMtl &renderPassResultBufOffsets,
    const BufferRef &renderPassResultBuf,
    const BufferRef &finalResultBuf)
{
    ASSERT(!renderPassResultBufOffsets.empty());

    if (renderPassResultBufOffsets.size() == 1 && !keepOldValue)
    {
        // Use blit command to copy directly
        BlitCommandEncoder *blitEncoder = contextMtl->getBlitCommandEncoder();

        blitEncoder->copyBuffer(renderPassResultBuf, renderPassResultBufOffsets.front(),
                                finalResultBuf, 0, kOcclusionQueryResultSize);
        return;
    }

    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    id<MTLComputePipelineState> pipeline =
        getVisibilityResultCombPipeline(contextMtl, keepOldValue);
    cmdEncoder->setComputePipelineState(pipeline);

    CombineVisibilityResultUniform options;
    // Offset is viewed as 64 bit unit in compute shader.
    options.startOffset = renderPassResultBufOffsets.front() / kOcclusionQueryResultSize;
    options.numOffsets  = renderPassResultBufOffsets.size();

    cmdEncoder->setData(options, 0);
    cmdEncoder->setBuffer(renderPassResultBuf, 0, 1);
    cmdEncoder->setBufferForWrite(finalResultBuf, 0, 2);

    DispatchCompute(contextMtl, cmdEncoder, pipeline, 1);
}

// MipmapUtils implementation
void MipmapUtils::onDestroy()
{
    m3DMipGeneratorPipeline      = nil;
    m2DMipGeneratorPipeline      = nil;
    m2DArrayMipGeneratorPipeline = nil;
    mCubeMipGeneratorPipeline    = nil;
}

void MipmapUtils::ensure3DMipGeneratorPipelineInitialized(ContextMtl *contextMtl)
{
    EnsureComputePipelineInitialized(contextMtl->getDisplay(), @"generate3DMipmaps",
                                     &m3DMipGeneratorPipeline);
}

void MipmapUtils::ensure2DMipGeneratorPipelineInitialized(ContextMtl *contextMtl)
{
    EnsureComputePipelineInitialized(contextMtl->getDisplay(), @"generate2DMipmaps",
                                     &m2DMipGeneratorPipeline);
}

void MipmapUtils::ensure2DArrayMipGeneratorPipelineInitialized(ContextMtl *contextMtl)
{
    EnsureComputePipelineInitialized(contextMtl->getDisplay(), @"generate2DArrayMipmaps",
                                     &m2DArrayMipGeneratorPipeline);
}

void MipmapUtils::ensureCubeMipGeneratorPipelineInitialized(ContextMtl *contextMtl)
{
    EnsureComputePipelineInitialized(contextMtl->getDisplay(), @"generateCubeMipmaps",
                                     &mCubeMipGeneratorPipeline);
}

angle::Result MipmapUtils::generateMipmapCS(ContextMtl *contextMtl,
                                            const TextureRef &srcTexture,
                                            gl::TexLevelArray<mtl::TextureRef> *mipmapOutputViews)
{

    MTLSize threadGroupSize;
    uint32_t slices                           = 1;
    id<MTLComputePipelineState> computePiline = nil;
    switch (srcTexture->textureType())
    {
        case MTLTextureType2D:
            ensure2DMipGeneratorPipelineInitialized(contextMtl);
            computePiline   = m2DMipGeneratorPipeline;
            threadGroupSize = MTLSizeMake(kGenerateMipThreadGroupSizePerDim,
                                          kGenerateMipThreadGroupSizePerDim, 1);
            break;
        case MTLTextureType2DArray:
            ensure2DArrayMipGeneratorPipelineInitialized(contextMtl);
            computePiline   = m2DArrayMipGeneratorPipeline;
            slices          = srcTexture->arrayLength();
            threadGroupSize = MTLSizeMake(kGenerateMipThreadGroupSizePerDim,
                                          kGenerateMipThreadGroupSizePerDim, 1);
            break;
        case MTLTextureTypeCube:
            ensureCubeMipGeneratorPipelineInitialized(contextMtl);
            computePiline   = mCubeMipGeneratorPipeline;
            slices          = 6;
            threadGroupSize = MTLSizeMake(kGenerateMipThreadGroupSizePerDim,
                                          kGenerateMipThreadGroupSizePerDim, 1);
            break;
        case MTLTextureType3D:
            ensure3DMipGeneratorPipelineInitialized(contextMtl);
            computePiline = m3DMipGeneratorPipeline;
            threadGroupSize =
                MTLSizeMake(kGenerateMipThreadGroupSizePerDim, kGenerateMipThreadGroupSizePerDim,
                            kGenerateMipThreadGroupSizePerDim);
            break;
        default:
            UNREACHABLE();
    }

    if (threadGroupSize.width * threadGroupSize.height * threadGroupSize.depth >
        computePiline.maxTotalThreadsPerThreadgroup)
    {
        // HACK: use blit command encoder to generate mipmaps if it is not possible
        // to use compute shader due to hardware limits.
        BlitCommandEncoder *blitEncoder = contextMtl->getBlitCommandEncoder();
        blitEncoder->generateMipmapsForTexture(srcTexture);
        return angle::Result::Continue;
    }

    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);
    cmdEncoder->setComputePipelineState(computePiline);

    Generate3DMipmapUniform options;
    uint32_t maxMipsPerBatch = 4;

    uint32_t remainMips = srcTexture->mipmapLevels() - 1;
    options.srcLevel    = 0;

    cmdEncoder->setTexture(srcTexture, 0);
    cmdEncoder->markResourceBeingWrittenByGPU(srcTexture);
    while (remainMips)
    {
        const TextureRef &firstMipView = mipmapOutputViews->at(options.srcLevel + 1);
        gl::Extents size               = firstMipView->size();
        bool isPow2 = gl::isPow2(size.width) && gl::isPow2(size.height) && gl::isPow2(size.depth);

        // Currently multiple mipmaps generation is only supported for power of two base level.
        if (isPow2)
        {
            options.numMipmapsToGenerate = std::min(remainMips, maxMipsPerBatch);
        }
        else
        {
            options.numMipmapsToGenerate = 1;
        }

        cmdEncoder->setData(options, 0);

        for (uint32_t i = 1; i <= options.numMipmapsToGenerate; ++i)
        {
            cmdEncoder->setTexture(mipmapOutputViews->at(options.srcLevel + i), i);
        }

        uint32_t threadsPerZ = std::max(slices, firstMipView->depth());

        DispatchCompute(contextMtl, cmdEncoder,
                        /** allowNonUniform */ false,
                        MTLSizeMake(firstMipView->width(), firstMipView->height(), threadsPerZ),
                        threadGroupSize);

        remainMips -= options.numMipmapsToGenerate;
        options.srcLevel += options.numMipmapsToGenerate;
    }

    return angle::Result::Continue;
}

// CopyPixelsUtils implementation
CopyPixelsUtils::CopyPixelsUtils(const std::string &readShaderName,
                                 const std::string &writeShaderName)
    : mReadShaderName(readShaderName), mWriteShaderName(writeShaderName)
{}
CopyPixelsUtils::CopyPixelsUtils(const CopyPixelsUtils &src)
    : CopyPixelsUtils(src.mReadShaderName, src.mWriteShaderName)
{}

void CopyPixelsUtils::onDestroy()
{
    ClearPipelineState2DArray(&mPixelsCopyPipelineCaches);
}

AutoObjCPtr<id<MTLComputePipelineState>> CopyPixelsUtils::getPixelsCopyPipeline(
    ContextMtl *contextMtl,
    const angle::Format &angleFormat,
    const TextureRef &texture,
    bool bufferWrite)
{
    int formatIDValue     = static_cast<int>(angleFormat.id);
    int shaderTextureType = GetShaderTextureType(texture);
    int index2 = mtl_shader::kTextureTypeCount * (bufferWrite ? 1 : 0) + shaderTextureType;

    auto &cache = mPixelsCopyPipelineCaches[formatIDValue][index2];

    if (!cache)
    {
        // Pipeline not cached, create it now:
        ANGLE_MTL_OBJC_SCOPE
        {
            auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

            [funcConstants setConstantValue:&formatIDValue
                                       type:MTLDataTypeInt
                                   withName:COPY_FORMAT_TYPE_CONSTANT_NAME];
            [funcConstants setConstantValue:&shaderTextureType
                                       type:MTLDataTypeInt
                                   withName:PIXEL_COPY_TEXTURE_TYPE_CONSTANT_NAME];

            NSString *shaderName = nil;
            if (bufferWrite)
            {
                shaderName = [NSString stringWithUTF8String:mWriteShaderName.c_str()];
            }
            else
            {
                shaderName = [NSString stringWithUTF8String:mReadShaderName.c_str()];
            }

            EnsureSpecializedComputePipelineInitialized(contextMtl->getDisplay(), shaderName,
                                                        funcConstants, &cache);
        }
    }

    return cache;
}

angle::Result CopyPixelsUtils::unpackPixelsFromBufferToTexture(
    ContextMtl *contextMtl,
    const angle::Format &srcAngleFormat,
    const CopyPixelsFromBufferParams &params)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    AutoObjCPtr<id<MTLComputePipelineState>> pipeline =
        getPixelsCopyPipeline(contextMtl, srcAngleFormat, params.texture, false);

    cmdEncoder->setComputePipelineState(pipeline);
    cmdEncoder->setBuffer(params.buffer, 0, 1);
    cmdEncoder->setTextureForWrite(params.texture, 0);

    CopyPixelFromBufferUniforms options;
    options.copySize[0]       = params.textureArea.width;
    options.copySize[1]       = params.textureArea.height;
    options.copySize[2]       = params.textureArea.depth;
    options.bufferStartOffset = params.bufferStartOffset;
    options.pixelSize         = srcAngleFormat.pixelBytes;
    options.bufferRowPitch    = params.bufferRowPitch;
    options.bufferDepthPitch  = params.bufferDepthPitch;
    options.textureOffset[0]  = params.textureArea.x;
    options.textureOffset[1]  = params.textureArea.y;
    options.textureOffset[2]  = params.textureArea.z;
    cmdEncoder->setData(options, 0);

    NSUInteger w                  = pipeline.get().threadExecutionWidth;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, 1, 1);

    MTLSize threads =
        MTLSizeMake(params.textureArea.width, params.textureArea.height, params.textureArea.depth);

    DispatchCompute(contextMtl, cmdEncoder,
                    /** allowNonUniform */ true, threads, threadsPerThreadgroup);

    return angle::Result::Continue;
}

angle::Result CopyPixelsUtils::packPixelsFromTextureToBuffer(ContextMtl *contextMtl,
                                                             const angle::Format &dstAngleFormat,
                                                             const CopyPixelsToBufferParams &params)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    AutoObjCPtr<id<MTLComputePipelineState>> pipeline =
        getPixelsCopyPipeline(contextMtl, dstAngleFormat, params.texture, true);

    cmdEncoder->setComputePipelineState(pipeline);
    cmdEncoder->setTexture(params.texture, 0);
    cmdEncoder->setBufferForWrite(params.buffer, 0, 1);

    WritePixelToBufferUniforms options;
    options.copySize[0]            = params.textureArea.width;
    options.copySize[1]            = params.textureArea.height;
    options.bufferStartOffset      = params.bufferStartOffset;
    options.pixelSize              = dstAngleFormat.pixelBytes;
    options.bufferRowPitch         = params.bufferRowPitch;
    options.textureOffset[0]       = params.textureArea.x;
    options.textureOffset[1]       = params.textureArea.y;
    options.textureLevel           = params.textureLevel;
    options.textureLayer           = params.textureSliceOrDeph;
    options.reverseTextureRowOrder = params.reverseTextureRowOrder;
    cmdEncoder->setData(options, 0);

    NSUInteger w                  = pipeline.get().threadExecutionWidth;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, 1, 1);

    MTLSize threads = MTLSizeMake(params.textureArea.width, params.textureArea.height, 1);

    DispatchCompute(contextMtl, cmdEncoder,
                    /** allowNonUniform */ true, threads, threadsPerThreadgroup);

    return angle::Result::Continue;
}

// VertexFormatConversionUtils implementation
void VertexFormatConversionUtils::onDestroy()
{
    ClearPipelineStateArray(&mConvertToFloatCompPipelineCaches);
    ClearRenderPipelineCacheArray(&mConvertToFloatRenderPipelineCaches);

    mComponentsExpandCompPipeline = nil;
    mComponentsExpandRenderPipelineCache.clear();
}

angle::Result VertexFormatConversionUtils::convertVertexFormatToFloatCS(
    ContextMtl *contextMtl,
    const angle::Format &srcAngleFormat,
    const VertexFormatConvertParams &params)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    AutoObjCPtr<id<MTLComputePipelineState>> pipeline =
        getFloatConverstionComputePipeline(contextMtl, srcAngleFormat);

    ANGLE_TRY(setupCommonConvertVertexFormatToFloat(contextMtl, cmdEncoder, pipeline,
                                                    srcAngleFormat, params));

    DispatchCompute(contextMtl, cmdEncoder, pipeline, params.vertexCount);
    return angle::Result::Continue;
}

angle::Result VertexFormatConversionUtils::convertVertexFormatToFloatVS(
    const gl::Context *context,
    RenderCommandEncoder *cmdEncoder,
    const angle::Format &srcAngleFormat,
    const VertexFormatConvertParams &params)
{
    ContextMtl *contextMtl = GetImpl(context);
    ASSERT(cmdEncoder);
    ASSERT(contextMtl->getDisplay()->getFeatures().hasExplicitMemBarrier.enabled);

    AutoObjCPtr<id<MTLRenderPipelineState>> pipeline =
        getFloatConverstionRenderPipeline(contextMtl, cmdEncoder, srcAngleFormat);

    ANGLE_TRY(setupCommonConvertVertexFormatToFloat(contextMtl, cmdEncoder, pipeline,
                                                    srcAngleFormat, params));

    cmdEncoder->draw(MTLPrimitiveTypePoint, 0, params.vertexCount);

    cmdEncoder->memoryBarrierWithResource(params.dstBuffer, kRenderStageVertex, kRenderStageVertex);

    // Invalidate current context's state.
    // NOTE(hqle): Consider invalidating only affected states.
    contextMtl->invalidateState(context);

    return angle::Result::Continue;
}

template <typename EncoderType, typename PipelineType>
angle::Result VertexFormatConversionUtils::setupCommonConvertVertexFormatToFloat(
    ContextMtl *contextMtl,
    EncoderType cmdEncoder,
    const PipelineType &pipeline,
    const angle::Format &srcAngleFormat,
    const VertexFormatConvertParams &params)
{
    SetPipelineState(cmdEncoder, pipeline);
    SetComputeOrVertexBuffer(cmdEncoder, params.srcBuffer, 0, 1);
    SetComputeOrVertexBufferForWrite(cmdEncoder, params.dstBuffer, 0, 2);

    CopyVertexUniforms options;
    options.srcBufferStartOffset = params.srcBufferStartOffset;
    options.srcStride            = params.srcStride;

    options.dstBufferStartOffset = params.dstBufferStartOffset;
    options.dstStride            = params.dstStride;
    options.dstComponents        = params.dstComponents;

    options.vertexCount = params.vertexCount;
    SetComputeOrVertexData(cmdEncoder, options, 0);

    return angle::Result::Continue;
}

// Expand number of components per vertex's attribute
angle::Result VertexFormatConversionUtils::expandVertexFormatComponentsCS(
    ContextMtl *contextMtl,
    const angle::Format &srcAngleFormat,
    const VertexFormatConvertParams &params)
{
    ComputeCommandEncoder *cmdEncoder = contextMtl->getComputeCommandEncoder();
    ASSERT(cmdEncoder);

    ensureComponentsExpandComputePipelineCreated(contextMtl);

    ANGLE_TRY(setupCommonExpandVertexFormatComponents(
        contextMtl, cmdEncoder, mComponentsExpandCompPipeline, srcAngleFormat, params));

    DispatchCompute(contextMtl, cmdEncoder, mComponentsExpandCompPipeline, params.vertexCount);
    return angle::Result::Continue;
}

angle::Result VertexFormatConversionUtils::expandVertexFormatComponentsVS(
    const gl::Context *context,
    RenderCommandEncoder *cmdEncoder,
    const angle::Format &srcAngleFormat,
    const VertexFormatConvertParams &params)
{
    ContextMtl *contextMtl = GetImpl(context);
    ASSERT(cmdEncoder);
    ASSERT(contextMtl->getDisplay()->getFeatures().hasExplicitMemBarrier.enabled);

    AutoObjCPtr<id<MTLRenderPipelineState>> pipeline =
        getComponentsExpandRenderPipeline(contextMtl, cmdEncoder);

    ANGLE_TRY(setupCommonExpandVertexFormatComponents(contextMtl, cmdEncoder, pipeline,
                                                      srcAngleFormat, params));

    cmdEncoder->draw(MTLPrimitiveTypePoint, 0, params.vertexCount);

    cmdEncoder->memoryBarrierWithResource(params.dstBuffer, kRenderStageVertex, kRenderStageVertex);

    // Invalidate current context's state.
    // NOTE(hqle): Consider invalidating only affected states.
    contextMtl->invalidateState(context);

    return angle::Result::Continue;
}

template <typename EncoderType, typename PipelineType>
angle::Result VertexFormatConversionUtils::setupCommonExpandVertexFormatComponents(
    ContextMtl *contextMtl,
    EncoderType cmdEncoder,
    const PipelineType &pipeline,
    const angle::Format &srcAngleFormat,
    const VertexFormatConvertParams &params)
{
    SetPipelineState(cmdEncoder, pipeline);
    SetComputeOrVertexBuffer(cmdEncoder, params.srcBuffer, 0, 1);
    SetComputeOrVertexBufferForWrite(cmdEncoder, params.dstBuffer, 0, 2);

    CopyVertexUniforms options;
    options.srcBufferStartOffset = params.srcBufferStartOffset;
    options.srcStride            = params.srcStride;
    options.srcComponentBytes    = srcAngleFormat.pixelBytes / srcAngleFormat.channelCount;
    options.srcComponents        = srcAngleFormat.channelCount;
    options.srcDefaultAlphaData  = params.srcDefaultAlphaData;

    options.dstBufferStartOffset = params.dstBufferStartOffset;
    options.dstStride            = params.dstStride;
    options.dstComponents        = params.dstComponents;

    options.vertexCount = params.vertexCount;
    SetComputeOrVertexData(cmdEncoder, options, 0);

    return angle::Result::Continue;
}

void VertexFormatConversionUtils::ensureComponentsExpandComputePipelineCreated(
    ContextMtl *contextMtl)
{
    EnsureComputePipelineInitialized(contextMtl->getDisplay(), @"expandVertexFormatComponentsCS",
                                     &mComponentsExpandCompPipeline);
}

AutoObjCPtr<id<MTLRenderPipelineState>>
VertexFormatConversionUtils::getComponentsExpandRenderPipeline(ContextMtl *contextMtl,
                                                               RenderCommandEncoder *cmdEncoder)
{
    EnsureVertexShaderOnlyPipelineCacheInitialized(contextMtl, @"expandVertexFormatComponentsVS",
                                                   &mComponentsExpandRenderPipelineCache);

    RenderPipelineDesc pipelineDesc = GetComputingVertexShaderOnlyRenderPipelineDesc(cmdEncoder);

    return mComponentsExpandRenderPipelineCache.getRenderPipelineState(contextMtl, pipelineDesc);
}

AutoObjCPtr<id<MTLComputePipelineState>>
VertexFormatConversionUtils::getFloatConverstionComputePipeline(ContextMtl *contextMtl,
                                                                const angle::Format &srcAngleFormat)
{
    int formatIDValue = static_cast<int>(srcAngleFormat.id);

    auto &cache = mConvertToFloatCompPipelineCaches[formatIDValue];

    if (!cache)
    {
        // Pipeline not cached, create it now:
        ANGLE_MTL_OBJC_SCOPE
        {
            auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

            [funcConstants setConstantValue:&formatIDValue
                                       type:MTLDataTypeInt
                                   withName:COPY_FORMAT_TYPE_CONSTANT_NAME];

            EnsureSpecializedComputePipelineInitialized(
                contextMtl->getDisplay(), @"convertToFloatVertexFormatCS", funcConstants, &cache);
        }
    }

    return cache;
}

AutoObjCPtr<id<MTLRenderPipelineState>>
VertexFormatConversionUtils::getFloatConverstionRenderPipeline(ContextMtl *contextMtl,
                                                               RenderCommandEncoder *cmdEncoder,
                                                               const angle::Format &srcAngleFormat)
{
    int formatIDValue = static_cast<int>(srcAngleFormat.id);

    RenderPipelineCache &cache = mConvertToFloatRenderPipelineCaches[formatIDValue];

    if (!cache.getVertexShader())
    {
        // Pipeline cache not intialized, do it now:
        ANGLE_MTL_OBJC_SCOPE
        {
            auto funcConstants = [[[MTLFunctionConstantValues alloc] init] ANGLE_MTL_AUTORELEASE];

            [funcConstants setConstantValue:&formatIDValue
                                       type:MTLDataTypeInt
                                   withName:COPY_FORMAT_TYPE_CONSTANT_NAME];

            EnsureSpecializedVertexShaderOnlyPipelineCacheInitialized(
                contextMtl, @"convertToFloatVertexFormatVS", funcConstants, &cache);
        }
    }

    RenderPipelineDesc pipelineDesc = GetComputingVertexShaderOnlyRenderPipelineDesc(cmdEncoder);

    return cache.getRenderPipelineState(contextMtl, pipelineDesc);
}

}  // namespace mtl
}  // namespace rx
