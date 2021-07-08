//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_command_buffer.h:
//    Defines the wrapper classes for Metal's MTLCommandEncoder, MTLCommandQueue and
//    MTLCommandBuffer.
//

#ifndef LIBANGLE_RENDERER_METAL_COMMANDENBUFFERMTL_H_
#define LIBANGLE_RENDERER_METAL_COMMANDENBUFFERMTL_H_

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/FixedVector.h"
#include "common/angleutils.h"
#include "libANGLE/renderer/metal/mtl_common.h"
#include "libANGLE/renderer/metal/mtl_resources.h"
#include "libANGLE/renderer/metal/mtl_state_cache.h"

namespace rx
{
namespace mtl
{

class CommandBuffer;
class CommandEncoder;
class RenderCommandEncoder;
class OcclusionQueryPool;

class CommandQueue final : public WrappedObject<id<MTLCommandQueue>>, angle::NonCopyable
{
  public:
    void reset();
    void set(id<MTLCommandQueue> metalQueue);

    void finishAllCommands();

    // This method will ensure that every GPU command buffer using this resource will finish before
    // returning. Note: this doesn't include the "in-progress" command buffer, i.e. the one hasn't
    // been commmitted yet. It's the responsibility of caller to make sure that command buffer is
    // commited/flushed first before calling this method.
    void ensureResourceReadyForCPU(const ResourceRef &resource);
    void ensureResourceReadyForCPU(Resource *resource);

    // Check whether the resource is being used by any command buffer still running on GPU.
    // This must be called before attempting to read the content of resource on CPU side.
    bool isResourceBeingUsedByGPU(const ResourceRef &resource) const
    {
        return isResourceBeingUsedByGPU(resource.get());
    }
    bool isResourceBeingUsedByGPU(const Resource *resource) const;

    // Checks whether the last command buffer that uses the given resource has been committed or not
    bool resourceHasPendingWorks(const Resource *resource) const;

    CommandQueue &operator=(id<MTLCommandQueue> metalQueue)
    {
        set(metalQueue);
        return *this;
    }

    AutoObjCPtr<id<MTLCommandBuffer>> makeMetalCommandBuffer(uint64_t *queueSerialOut);
    void onCommandBufferCommitted(id<MTLCommandBuffer> buf, uint64_t serial);

  private:
    void onCommandBufferCompleted(id<MTLCommandBuffer> buf, uint64_t serial);
    using ParentClass = WrappedObject<id<MTLCommandQueue>>;

    struct CmdBufferQueueEntry
    {
        AutoObjCPtr<id<MTLCommandBuffer>> buffer;
        uint64_t serial;
    };
    std::deque<CmdBufferQueueEntry> mMetalCmdBuffers;
    std::deque<CmdBufferQueueEntry> mMetalCmdBuffersTmp;

    uint64_t mQueueSerialCounter = 1;
    std::atomic<uint64_t> mCommittedBufferSerial{0};
    std::atomic<uint64_t> mCompletedBufferSerial{0};

    mutable std::mutex mLock;
};

class CommandBuffer final : public WrappedObject<id<MTLCommandBuffer>>, angle::NonCopyable
{
  public:
    CommandBuffer(CommandQueue *cmdQueue);
    ~CommandBuffer();

    // This method must be called so that command encoder can be used.
    void restart();

    // Return true if command buffer can be encoded into. Return false if it has been committed
    // and hasn't been restarted.
    bool valid() const;
    void commit();
    // wait for committed command buffer to finish.
    void finish();

    void present(id<CAMetalDrawable> presentationDrawable);

    void setWriteDependency(const ResourceRef &resource);
    void setReadDependency(const ResourceRef &resource);

    void insertDebugSign(const std::string &marker);
    void pushDebugGroup(const std::string &marker);
    void popDebugGroup();

    void queueEventSignal(const mtl::SharedEventRef &event, uint64_t value);
    void serverWaitEvent(const mtl::SharedEventRef &event, uint64_t value);

    CommandQueue &cmdQueue() { return mCmdQueue; }

    // Private use only
    void setActiveCommandEncoder(CommandEncoder *encoder);
    void invalidateActiveCommandEncoder(CommandEncoder *encoder);

  private:
    void set(id<MTLCommandBuffer> metalBuffer);
    void cleanup();

    bool validImpl() const;
    void commitImpl();
    void forceEndingCurrentEncoder();

    void setPendingEvents();
    void setEventImpl(const mtl::SharedEventRef &event, uint64_t value);
    void waitEventImpl(const mtl::SharedEventRef &event, uint64_t value);

    using ParentClass = WrappedObject<id<MTLCommandBuffer>>;

    CommandQueue &mCmdQueue;

    CommandEncoder *mActiveCommandEncoder = nullptr;

    uint64_t mQueueSerial = 0;

    std::vector<std::string> mPendingDebugSigns;
    std::vector<std::pair<mtl::SharedEventRef, uint64_t>> mPendingSignalEvents;

    bool mCommitted = false;
};

class CommandEncoder : public WrappedObject<id<MTLCommandEncoder>>, angle::NonCopyable
{
  public:
    enum Type
    {
        RENDER,
        BLIT,
        COMPUTE,
    };

    virtual ~CommandEncoder();

    virtual void endEncoding();

    virtual void reset();
    Type getType() const { return mType; }

    CommandEncoder &markResourceBeingWrittenByGPU(const BufferRef &buffer);
    CommandEncoder &markResourceBeingWrittenByGPU(const TextureRef &texture);

    void insertDebugSign(NSString *marker);

  protected:
    using ParentClass = WrappedObject<id<MTLCommandEncoder>>;

    CommandEncoder(CommandBuffer *cmdBuffer, Type type);

    CommandBuffer &cmdBuffer() { return mCmdBuffer; }
    CommandQueue &cmdQueue() { return mCmdBuffer.cmdQueue(); }

    void set(id<MTLCommandEncoder> metalCmdEncoder);

    virtual void insertDebugSignImpl(NSString *marker);

  private:
    const Type mType;
    CommandBuffer &mCmdBuffer;
};

// Per shader stage's states
struct RenderCommandEncoderShaderStates
{
    RenderCommandEncoderShaderStates();

    void reset();

    std::array<AutoObjCPtr<id<MTLBuffer>>, kMaxShaderBuffers> buffers;
    std::array<uint32_t, kMaxShaderBuffers> bufferOffsets;
    std::array<AutoObjCPtr<id<MTLSamplerState>>, kMaxShaderSamplers> samplers;
    std::array<Optional<std::pair<float, float>>, kMaxShaderSamplers> samplerLodClamps;
    std::array<AutoObjCPtr<id<MTLTexture>>, kMaxShaderSamplers> textures;
};

// Per render pass's states
struct RenderCommandEncoderStates
{
    RenderCommandEncoderStates();

    void reset();

    AutoObjCPtr<id<MTLRenderPipelineState>> renderPipeline;

    MTLTriangleFillMode triangleFillMode;
    MTLWinding winding;
    MTLCullMode cullMode;

    AutoObjCPtr<id<MTLDepthStencilState>> depthStencilState;
    float depthBias, depthSlopeScale, depthClamp;

    uint32_t stencilFrontRef, stencilBackRef;

    Optional<MTLViewport> viewport;
    Optional<MTLScissorRect> scissorRect;

    std::array<float, 4> blendColor;

    gl::ShaderMap<RenderCommandEncoderShaderStates> perShaderStates;

    MTLVisibilityResultMode visibilityResultMode;
    size_t visibilityResultBufferOffset;
};

// Encoder for encoding render commands
class RenderCommandEncoder final : public CommandEncoder
{
  public:
    RenderCommandEncoder(CommandBuffer *cmdBuffer, const OcclusionQueryPool &queryPool);
    ~RenderCommandEncoder() override;

    // override CommandEncoder
    bool valid() const { return mRecording; }
    void reset() override;
    void endEncoding() override;

    // Restart the encoder so that new commands can be encoded.
    // NOTE: parent CommandBuffer's restart() must be called before this.
    RenderCommandEncoder &restart(const RenderPassDesc &desc);

    RenderCommandEncoder &setRenderPipelineState(id<MTLRenderPipelineState> state);
    RenderCommandEncoder &setTriangleFillMode(MTLTriangleFillMode mode);
    RenderCommandEncoder &setFrontFacingWinding(MTLWinding winding);
    RenderCommandEncoder &setCullMode(MTLCullMode mode);

    RenderCommandEncoder &setDepthStencilState(id<MTLDepthStencilState> state);
    RenderCommandEncoder &setDepthBias(float depthBias, float slopeScale, float clamp);
    RenderCommandEncoder &setStencilRefVals(uint32_t frontRef, uint32_t backRef);
    RenderCommandEncoder &setStencilRefVal(uint32_t ref);

    RenderCommandEncoder &setViewport(const MTLViewport &viewport);
    RenderCommandEncoder &setScissorRect(const MTLScissorRect &rect);

    RenderCommandEncoder &setBlendColor(float r, float g, float b, float a);

    RenderCommandEncoder &setVertexBuffer(const BufferRef &buffer, uint32_t offset, uint32_t index)
    {
        return setBuffer(gl::ShaderType::Vertex, buffer, offset, index);
    }
    RenderCommandEncoder &setVertexBytes(const uint8_t *bytes, size_t size, uint32_t index)
    {
        return setBytes(gl::ShaderType::Vertex, bytes, size, index);
    }
    template <typename T>
    RenderCommandEncoder &setVertexData(const T &data, uint32_t index)
    {
        return setVertexBytes(reinterpret_cast<const uint8_t *>(&data), sizeof(T), index);
    }
    RenderCommandEncoder &setVertexSamplerState(id<MTLSamplerState> state,
                                                float lodMinClamp,
                                                float lodMaxClamp,
                                                uint32_t index)
    {
        return setSamplerState(gl::ShaderType::Vertex, state, lodMinClamp, lodMaxClamp, index);
    }
    RenderCommandEncoder &setVertexTexture(const TextureRef &texture, uint32_t index)
    {
        return setTexture(gl::ShaderType::Vertex, texture, index);
    }

    RenderCommandEncoder &setFragmentBuffer(const BufferRef &buffer,
                                            uint32_t offset,
                                            uint32_t index)
    {
        return setBuffer(gl::ShaderType::Fragment, buffer, offset, index);
    }
    RenderCommandEncoder &setFragmentBytes(const uint8_t *bytes, size_t size, uint32_t index)
    {
        return setBytes(gl::ShaderType::Fragment, bytes, size, index);
    }
    template <typename T>
    RenderCommandEncoder &setFragmentData(const T &data, uint32_t index)
    {
        return setFragmentBytes(reinterpret_cast<const uint8_t *>(&data), sizeof(T), index);
    }
    RenderCommandEncoder &setFragmentSamplerState(id<MTLSamplerState> state,
                                                  float lodMinClamp,
                                                  float lodMaxClamp,
                                                  uint32_t index)
    {
        return setSamplerState(gl::ShaderType::Fragment, state, lodMinClamp, lodMaxClamp, index);
    }
    RenderCommandEncoder &setFragmentTexture(const TextureRef &texture, uint32_t index)
    {
        return setTexture(gl::ShaderType::Fragment, texture, index);
    }

    RenderCommandEncoder &setBuffer(gl::ShaderType shaderType,
                                    const BufferRef &buffer,
                                    uint32_t offset,
                                    uint32_t index);
    RenderCommandEncoder &setBufferForWrite(gl::ShaderType shaderType,
                                            const BufferRef &buffer,
                                            uint32_t offset,
                                            uint32_t index);
    RenderCommandEncoder &setBytes(gl::ShaderType shaderType,
                                   const uint8_t *bytes,
                                   size_t size,
                                   uint32_t index);
    template <typename T>
    RenderCommandEncoder &setData(gl::ShaderType shaderType, const T &data, uint32_t index)
    {
        return setBytes(shaderType, reinterpret_cast<const uint8_t *>(&data), sizeof(T), index);
    }
    RenderCommandEncoder &setSamplerState(gl::ShaderType shaderType,
                                          id<MTLSamplerState> state,
                                          float lodMinClamp,
                                          float lodMaxClamp,
                                          uint32_t index);
    RenderCommandEncoder &setTexture(gl::ShaderType shaderType,
                                     const TextureRef &texture,
                                     uint32_t index);

    RenderCommandEncoder &draw(MTLPrimitiveType primitiveType,
                               uint32_t vertexStart,
                               uint32_t vertexCount);
    RenderCommandEncoder &drawInstanced(MTLPrimitiveType primitiveType,
                                        uint32_t vertexStart,
                                        uint32_t vertexCount,
                                        uint32_t instances);
    RenderCommandEncoder &drawIndexed(MTLPrimitiveType primitiveType,
                                      uint32_t indexCount,
                                      MTLIndexType indexType,
                                      const BufferRef &indexBuffer,
                                      size_t bufferOffset);
    RenderCommandEncoder &drawIndexedInstanced(MTLPrimitiveType primitiveType,
                                               uint32_t indexCount,
                                               MTLIndexType indexType,
                                               const BufferRef &indexBuffer,
                                               size_t bufferOffset,
                                               uint32_t instances);
    RenderCommandEncoder &drawIndexedInstancedBaseVertex(MTLPrimitiveType primitiveType,
                                                         uint32_t indexCount,
                                                         MTLIndexType indexType,
                                                         const BufferRef &indexBuffer,
                                                         size_t bufferOffset,
                                                         uint32_t instances,
                                                         uint32_t baseVertex);

    RenderCommandEncoder &setVisibilityResultMode(MTLVisibilityResultMode mode, size_t offset);

    RenderCommandEncoder &useResource(const BufferRef &resource,
                                      MTLResourceUsage usage,
                                      mtl::RenderStages stages);

    RenderCommandEncoder &memoryBarrierWithResource(const BufferRef &resource,
                                                    mtl::RenderStages after,
                                                    mtl::RenderStages before);

    RenderCommandEncoder &setColorStoreAction(MTLStoreAction action, uint32_t colorAttachmentIndex);
    // Set store action for every color attachment.
    RenderCommandEncoder &setColorStoreAction(MTLStoreAction action);

    RenderCommandEncoder &setDepthStencilStoreAction(MTLStoreAction depthStoreAction,
                                                     MTLStoreAction stencilStoreAction);
    RenderCommandEncoder &setDepthStoreAction(MTLStoreAction action);
    RenderCommandEncoder &setStencilStoreAction(MTLStoreAction action);

    // Set storeaction for every color & depth & stencil attachment.
    RenderCommandEncoder &setStoreAction(MTLStoreAction action);

    // Change the render pass's loadAction. Note that this operation is only allowed when
    // canChangeLoadAction() returns true.
    RenderCommandEncoder &setColorLoadAction(MTLLoadAction action,
                                             const MTLClearColor &clearValue,
                                             uint32_t colorAttachmentIndex);
    RenderCommandEncoder &setDepthLoadAction(MTLLoadAction action, double clearValue);
    RenderCommandEncoder &setStencilLoadAction(MTLLoadAction action, uint32_t clearValue);

    void pushDebugGroup(NSString *label);
    void popDebugGroup();

    void setLabel(NSString *label);

    const RenderPassDesc &renderPassDesc() const { return mRenderPassDesc; }
    bool canChangeLoadAction() const { return get() == nil; }

    // Enable warning when scissor rect is out of render target's bound
    void enableScissorRectOOBWarn(bool e) { mWarnOutOfBoundScissorRect = e; }

  private:
    // Override CommandEncoder
    id<MTLRenderCommandEncoder> get() const
    {
        return static_cast<id<MTLRenderCommandEncoder>>(CommandEncoder::get());
    }
    void insertDebugSignImpl(NSString *label) override;

    void initAttachmentWriteDependencyAndScissorRect(const RenderPassAttachmentDesc &attachment);

    MTLStoreAction correctStoreAction(MTLRenderPassAttachmentDescriptor *objCRenderPassAttachment,
                                      MTLStoreAction finalStoreAction);

    void ensureMetalEncoderStarted();
    void simulateDiscardFramebuffer();
    void endEncodingImpl(bool considerDiscardSimulation);
    void applyStates();

    RenderCommandEncoder &commonSetBuffer(gl::ShaderType shaderType,
                                          id<MTLBuffer> mtlBuffer,
                                          uint32_t offset,
                                          uint32_t index);

    RenderCommandEncoder &mtlSetRenderPipelineState(id<MTLRenderPipelineState> state);
    RenderCommandEncoder &mtlSetTriangleFillMode(MTLTriangleFillMode mode);
    RenderCommandEncoder &mtlSetFrontFacingWinding(MTLWinding winding);
    RenderCommandEncoder &mtlSetCullMode(MTLCullMode mode);

    RenderCommandEncoder &mtlSetDepthStencilState(id<MTLDepthStencilState> state);
    RenderCommandEncoder &mtlSetDepthBias(float depthBias, float slopeScale, float clamp);
    RenderCommandEncoder &mtlSetStencilRefVals(uint32_t frontRef, uint32_t backRef);

    RenderCommandEncoder &mtlSetViewport(const MTLViewport &viewport);
    RenderCommandEncoder &mtlSetScissorRect(const MTLScissorRect &rect);

    RenderCommandEncoder &mtlSetBlendColor(float r, float g, float b, float a);

    RenderCommandEncoder &mtlSetVertexBuffer(id<MTLBuffer> buffer, uint32_t offset, uint32_t index);
    RenderCommandEncoder &mtlSetVertexBufferOffset(uint32_t offset, uint32_t index);
    RenderCommandEncoder &mtlSetVertexBytes(const uint8_t *bytes, size_t size, uint32_t index);
    RenderCommandEncoder &mtlSetVertexSamplerState(id<MTLSamplerState> state, uint32_t index);
    RenderCommandEncoder &mtlSetVertexSamplerState(id<MTLSamplerState> state,
                                                   float lodMinClamp,
                                                   float lodMaxClamp,
                                                   uint32_t index);
    RenderCommandEncoder &mtlSetVertexTexture(id<MTLTexture> texture, uint32_t index);

    RenderCommandEncoder &mtlSetFragmentBuffer(id<MTLBuffer> buffer,
                                               uint32_t offset,
                                               uint32_t index);
    RenderCommandEncoder &mtlSetFragmentBufferOffset(uint32_t offset, uint32_t index);
    RenderCommandEncoder &mtlSetFragmentBytes(const uint8_t *bytes, size_t size, uint32_t index);
    RenderCommandEncoder &mtlSetFragmentSamplerState(id<MTLSamplerState> state, uint32_t index);
    RenderCommandEncoder &mtlSetFragmentSamplerState(id<MTLSamplerState> state,
                                                     float lodMinClamp,
                                                     float lodMaxClamp,
                                                     uint32_t index);
    RenderCommandEncoder &mtlSetFragmentTexture(id<MTLTexture> texture, uint32_t index);

    RenderCommandEncoder &mtlSetVisibilityResultMode(MTLVisibilityResultMode mode, size_t offset);

    RenderCommandEncoder &mtlUseResource(id<MTLBuffer> resource,
                                         MTLResourceUsage usage,
                                         mtl::RenderStages stages);

    RenderCommandEncoder &mtlMemoryBarrierWithResource(id<MTLBuffer> resource,
                                                       mtl::RenderStages after,
                                                       mtl::RenderStages before);

    RenderCommandEncoder &mtlInsertDebugSign(NSString *label);

    RenderCommandEncoder &mtlPushDebugGroup(NSString *label);
    RenderCommandEncoder &mtlPopDebugGroup();

    RenderCommandEncoder &mtlSetLabel(NSString *label);

    RenderPassDesc mRenderPassDesc;
    // Cached Objective-C render pass desc to avoid re-allocate every frame.
    mtl::AutoObjCObj<MTLRenderPassDescriptor> mCachedRenderPassDescObjC;

    mtl::AutoObjCObj<NSString> mDeferredLabel;

    mtl::AutoObjCObj<NSString> mDeferredDebugSign;
    std::vector<mtl::AutoObjCObj<NSString>> mDeferredDebugGroups;

    MTLScissorRect mRenderPassMaxScissorRect;

    bool mWarnOutOfBoundScissorRect;

    const OcclusionQueryPool &mOcclusionQueryPool;
    bool mRecording = false;

    using SetBufferFunc       = RenderCommandEncoder &(RenderCommandEncoder::*)(id<MTLBuffer>,
                                                                          uint32_t,
                                                                          uint32_t);
    using SetBufferOffsetFunc = RenderCommandEncoder &(RenderCommandEncoder::*)(uint32_t, uint32_t);
    using SetBytesFunc        = RenderCommandEncoder &(RenderCommandEncoder::*)(const uint8_t *,
                                                                         size_t,
                                                                         uint32_t);
    using SetSamplerStateWithoutLodFunc =
        RenderCommandEncoder &(RenderCommandEncoder::*)(id<MTLSamplerState>, uint32_t);
    using SetSamplerStateFunc = RenderCommandEncoder &(RenderCommandEncoder::*)(id<MTLSamplerState>,
                                                                                float,
                                                                                float,
                                                                                uint32_t);
    using SetTextureFunc      = RenderCommandEncoder &(RenderCommandEncoder::*)(id<MTLTexture>,
                                                                           uint32_t);

    gl::ShaderMap<SetBufferFunc> mSetBufferFuncs;
    gl::ShaderMap<SetBufferOffsetFunc> mSetBufferOffsetFuncs;
    gl::ShaderMap<SetBytesFunc> mSetBytesFuncs;
    gl::ShaderMap<SetTextureFunc> mSetTextureFuncs;
    gl::ShaderMap<SetSamplerStateFunc> mSetSamplerFuncs;
    gl::ShaderMap<SetSamplerStateWithoutLodFunc> mSetSamplerWithoutLodFuncs;

    RenderCommandEncoderStates mStateCache = {};
};

class BlitCommandEncoder final : public CommandEncoder
{
  public:
    BlitCommandEncoder(CommandBuffer *cmdBuffer);
    ~BlitCommandEncoder() override;

    // Restart the encoder so that new commands can be encoded.
    // NOTE: parent CommandBuffer's restart() must be called before this.
    BlitCommandEncoder &restart();

    BlitCommandEncoder &copyBuffer(const BufferRef &src,
                                   size_t srcOffset,
                                   const BufferRef &dst,
                                   size_t dstOffset,
                                   size_t size);

    BlitCommandEncoder &copyBufferToTexture(const BufferRef &src,
                                            size_t srcOffset,
                                            size_t srcBytesPerRow,
                                            size_t srcBytesPerImage,
                                            MTLSize srcSize,
                                            const TextureRef &dst,
                                            uint32_t dstSlice,
                                            uint32_t dstLevel,
                                            MTLOrigin dstOrigin,
                                            MTLBlitOption blitOption);

    BlitCommandEncoder &copyTextureToBuffer(const TextureRef &src,
                                            uint32_t srcSlice,
                                            uint32_t srcLevel,
                                            MTLOrigin srcOrigin,
                                            MTLSize srcSize,
                                            const BufferRef &dst,
                                            size_t dstOffset,
                                            size_t dstBytesPerRow,
                                            size_t dstBytesPerImage,
                                            MTLBlitOption blitOption);

    BlitCommandEncoder &copyTexture(const TextureRef &src,
                                    uint32_t srcSlice,
                                    uint32_t srcLevel,
                                    MTLOrigin srcOrigin,
                                    MTLSize srcSize,
                                    const TextureRef &dst,
                                    uint32_t dstSlice,
                                    uint32_t dstLevel,
                                    MTLOrigin dstOrigin);

    BlitCommandEncoder &copyTexture(const TextureRef &src,
                                    uint32_t srcSlice,
                                    uint32_t srcLevel,
                                    const TextureRef &dst,
                                    uint32_t dstSlice,
                                    uint32_t dstLevel,
                                    uint32_t sliceCount,
                                    uint32_t levelCount);

    BlitCommandEncoder &fillBuffer(const BufferRef &buffer, NSRange range, uint8_t value);

    BlitCommandEncoder &generateMipmapsForTexture(const TextureRef &texture);
    BlitCommandEncoder &synchronizeResource(const BufferRef &buffer);
    BlitCommandEncoder &synchronizeResource(const TextureRef &texture);

  private:
    id<MTLBlitCommandEncoder> get()
    {
        return static_cast<id<MTLBlitCommandEncoder>>(CommandEncoder::get());
    }
};

class ComputeCommandEncoder final : public CommandEncoder
{
  public:
    ComputeCommandEncoder(CommandBuffer *cmdBuffer);
    ~ComputeCommandEncoder() override;

    // Restart the encoder so that new commands can be encoded.
    // NOTE: parent CommandBuffer's restart() must be called before this.
    ComputeCommandEncoder &restart();

    ComputeCommandEncoder &setComputePipelineState(id<MTLComputePipelineState> state);

    ComputeCommandEncoder &setBuffer(const BufferRef &buffer, uint32_t offset, uint32_t index);
    ComputeCommandEncoder &setBufferForWrite(const BufferRef &buffer,
                                             uint32_t offset,
                                             uint32_t index);
    ComputeCommandEncoder &setBytes(const uint8_t *bytes, size_t size, uint32_t index);
    template <typename T>
    ComputeCommandEncoder &setData(const T &data, uint32_t index)
    {
        return setBytes(reinterpret_cast<const uint8_t *>(&data), sizeof(T), index);
    }
    ComputeCommandEncoder &setSamplerState(id<MTLSamplerState> state,
                                           float lodMinClamp,
                                           float lodMaxClamp,
                                           uint32_t index);
    ComputeCommandEncoder &setTexture(const TextureRef &texture, uint32_t index);
    ComputeCommandEncoder &setTextureForWrite(const TextureRef &texture, uint32_t index);

    ComputeCommandEncoder &dispatch(const MTLSize &threadGroupsPerGrid,
                                    const MTLSize &threadsPerGroup);

    ComputeCommandEncoder &dispatchNonUniform(const MTLSize &threadsPerGrid,
                                              const MTLSize &threadsPerGroup);

  private:
    id<MTLComputeCommandEncoder> get()
    {
        return static_cast<id<MTLComputeCommandEncoder>>(CommandEncoder::get());
    }
};

}  // namespace mtl
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_COMMANDENBUFFERMTL_H_ */
