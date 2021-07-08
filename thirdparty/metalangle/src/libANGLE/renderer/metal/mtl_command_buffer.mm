//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_command_buffer.mm:
//      Implementations of Metal framework's MTLCommandBuffer, MTLCommandQueue,
//      MTLCommandEncoder's wrappers.
//

#include "libANGLE/renderer/metal/mtl_command_buffer.h"

#include <cassert>
#if ANGLE_MTL_SIMULATE_DISCARD_FRAMEBUFFER
#    include <random>
#endif

#include "common/debug.h"
#include "libANGLE/renderer/metal/mtl_occlusion_query_pool.h"
#include "libANGLE/renderer/metal/mtl_resources.h"

// Use to compare the new values with the values already set in the command encoder:
static inline bool operator==(const MTLViewport &lhs, const MTLViewport &rhs)
{
    return memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

static inline bool operator==(const MTLScissorRect &lhs, const MTLScissorRect &rhs)
{
    return memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

namespace rx
{
namespace mtl
{

namespace
{

NSString *cppLabelToObjC(const std::string &marker)
{
    auto label = [NSString stringWithUTF8String:marker.c_str()];
    if (!label)
    {
        // This can happen if the string is not a valid ascii string.
        label = @"Invalid ASCII string";
    }
    return label;
}

}

// CommandQueue implementation
void CommandQueue::reset()
{
    finishAllCommands();
    ParentClass::reset();
}

void CommandQueue::set(id<MTLCommandQueue> metalQueue)
{
    finishAllCommands();

    ParentClass::set(metalQueue);
}

void CommandQueue::finishAllCommands()
{
    {
        // Copy to temp list
        std::lock_guard<std::mutex> lg(mLock);

        for (CmdBufferQueueEntry &metalBufferEntry : mMetalCmdBuffers)
        {
            mMetalCmdBuffersTmp.push_back(metalBufferEntry);
        }

        mMetalCmdBuffers.clear();
    }

    // Wait for command buffers to finish
    for (CmdBufferQueueEntry &metalBufferEntry : mMetalCmdBuffersTmp)
    {
        [metalBufferEntry.buffer waitUntilCompleted];
    }
    mMetalCmdBuffersTmp.clear();
}

void CommandQueue::ensureResourceReadyForCPU(const ResourceRef &resource)
{
    if (!resource)
    {
        return;
    }

    ensureResourceReadyForCPU(resource.get());
}

void CommandQueue::ensureResourceReadyForCPU(Resource *resource)
{
    mLock.lock();
    while (isResourceBeingUsedByGPU(resource) && !mMetalCmdBuffers.empty())
    {
        CmdBufferQueueEntry metalBufferEntry = mMetalCmdBuffers.front();
        mMetalCmdBuffers.pop_front();
        mLock.unlock();

        ANGLE_MTL_LOG("Waiting for MTLCommandBuffer %llu:%p", metalBufferEntry.serial,
                      metalBufferEntry.buffer.get());
        [metalBufferEntry.buffer waitUntilCompleted];

        mLock.lock();
    }
    mLock.unlock();

    // This can happen if the resource is read then write in the same command buffer.
    // So it is the responsitibily of outer code to ensure the command buffer is commit before
    // the resource can be read or written again
    ASSERT(!isResourceBeingUsedByGPU(resource));
}

bool CommandQueue::isResourceBeingUsedByGPU(const Resource *resource) const
{
    if (!resource)
    {
        return false;
    }

    return mCompletedBufferSerial.load(std::memory_order_relaxed) <
           resource->getCommandBufferQueueSerial();
}

bool CommandQueue::resourceHasPendingWorks(const Resource *resource) const
{
    if (!resource)
    {
        return false;
    }

    return mCommittedBufferSerial.load(std::memory_order_relaxed) <
           resource->getCommandBufferQueueSerial();
}

AutoObjCPtr<id<MTLCommandBuffer>> CommandQueue::makeMetalCommandBuffer(uint64_t *queueSerialOut)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        AutoObjCPtr<id<MTLCommandBuffer>> metalCmdBuffer = [get() commandBuffer];

        std::lock_guard<std::mutex> lg(mLock);

        uint64_t serial = mQueueSerialCounter++;

        mMetalCmdBuffers.push_back({metalCmdBuffer, serial});

        ANGLE_MTL_LOG("Created MTLCommandBuffer %llu:%p", serial, metalCmdBuffer.get());

        [metalCmdBuffer addCompletedHandler:^(id<MTLCommandBuffer> buf) {
          onCommandBufferCompleted(buf, serial);
        }];

        [metalCmdBuffer enqueue];

        ASSERT(metalCmdBuffer);

        *queueSerialOut = serial;

        return metalCmdBuffer;
    }
}

void CommandQueue::onCommandBufferCommitted(id<MTLCommandBuffer> buf, uint64_t serial)
{
    std::lock_guard<std::mutex> lg(mLock);

    ANGLE_MTL_LOG("Committed MTLCommandBuffer %llu:%p", serial, buf);

    mCommittedBufferSerial.store(
        std::max(mCommittedBufferSerial.load(std::memory_order_relaxed), serial),
        std::memory_order_relaxed);
}

void CommandQueue::onCommandBufferCompleted(id<MTLCommandBuffer> buf, uint64_t serial)
{
    std::lock_guard<std::mutex> lg(mLock);

    ANGLE_MTL_LOG("Completed MTLCommandBuffer %llu:%p", serial, buf);

    if (mCompletedBufferSerial >= serial)
    {
        // Already handled.
        return;
    }

    while (!mMetalCmdBuffers.empty() && mMetalCmdBuffers.front().serial <= serial)
    {
        CmdBufferQueueEntry metalBufferEntry = mMetalCmdBuffers.front();
        ANGLE_UNUSED_VARIABLE(metalBufferEntry);
        ANGLE_MTL_LOG("Popped MTLCommandBuffer %llu:%p", metalBufferEntry.serial,
                      metalBufferEntry.buffer.get());

        mMetalCmdBuffers.pop_front();
    }

    mCompletedBufferSerial.store(
        std::max(mCompletedBufferSerial.load(std::memory_order_relaxed), serial),
        std::memory_order_relaxed);
}

// CommandBuffer implementation
CommandBuffer::CommandBuffer(CommandQueue *cmdQueue) : mCmdQueue(*cmdQueue) {}

CommandBuffer::~CommandBuffer()
{
    finish();
    cleanup();
}

bool CommandBuffer::valid() const
{
    return validImpl();
}

void CommandBuffer::commit()
{
    commitImpl();
}

void CommandBuffer::finish()
{
    commit();
    [get() waitUntilCompleted];
}

void CommandBuffer::present(id<CAMetalDrawable> presentationDrawable)
{
    [get() presentDrawable:presentationDrawable];
}

void CommandBuffer::setWriteDependency(const ResourceRef &resource)
{
    if (!resource)
    {
        return;
    }

    if (!validImpl())
    {
        return;
    }

    resource->setUsedByCommandBufferWithQueueSerial(mQueueSerial, true);
}

void CommandBuffer::setReadDependency(const ResourceRef &resource)
{
    if (!resource)
    {
        return;
    }

    if (!validImpl())
    {
        return;
    }

    resource->setUsedByCommandBufferWithQueueSerial(mQueueSerial, false);
}

void CommandBuffer::restart()
{
    uint64_t serial     = 0;
    auto metalCmdBuffer = mCmdQueue.makeMetalCommandBuffer(&serial);

    set(metalCmdBuffer);
    mQueueSerial = serial;
    mCommitted   = false;

    ASSERT(metalCmdBuffer);
}

void CommandBuffer::insertDebugSign(const std::string &marker)
{
    mtl::CommandEncoder *currentEncoder = mActiveCommandEncoder;
    if (currentEncoder)
    {
        ANGLE_MTL_OBJC_SCOPE
        {
            auto label = cppLabelToObjC(marker);
            currentEncoder->insertDebugSign(label);
        }
    }
    else
    {
        mPendingDebugSigns.push_back(marker);
    }
}

void CommandBuffer::pushDebugGroup(const std::string &marker)
{
    // NOTE(hqle): to implement this
}

void CommandBuffer::popDebugGroup()
{
    // NOTE(hqle): to implement this
}

void CommandBuffer::queueEventSignal(const mtl::SharedEventRef &event, uint64_t value)
{
    ASSERT(validImpl());

    if (mActiveCommandEncoder && mActiveCommandEncoder->getType() == CommandEncoder::RENDER)
    {
        // We cannot set event when there is an active render pass, defer the setting until the
        // pass end.
        mPendingSignalEvents.push_back({event, value});
    }
    else
    {
        setEventImpl(event, value);
    }
}

void CommandBuffer::serverWaitEvent(const mtl::SharedEventRef &event, uint64_t value)
{
    ASSERT(validImpl());

    waitEventImpl(event, value);
}

/** private use only */
void CommandBuffer::set(id<MTLCommandBuffer> metalBuffer)
{
    ParentClass::set(metalBuffer);
}

void CommandBuffer::setActiveCommandEncoder(CommandEncoder *encoder)
{
    mActiveCommandEncoder = encoder;
    for (auto &marker : mPendingDebugSigns)
    {
        ANGLE_MTL_OBJC_SCOPE
        {
            auto label = cppLabelToObjC(marker);
            encoder->insertDebugSign(label);
        }
    }
    mPendingDebugSigns.clear();
}

void CommandBuffer::invalidateActiveCommandEncoder(CommandEncoder *encoder)
{
    if (mActiveCommandEncoder == encoder)
    {
        mActiveCommandEncoder = nullptr;

        // No active command encoder, we can safely encode event signalling now.
        setPendingEvents();
    }
}

void CommandBuffer::cleanup()
{
    mActiveCommandEncoder = nullptr;

    ParentClass::set(nil);
}

bool CommandBuffer::validImpl() const
{
    if (!ParentClass::valid())
    {
        return false;
    }

    return !mCommitted;
}

void CommandBuffer::commitImpl()
{
    if (!validImpl())
    {
        return;
    }

    // End the current encoder
    forceEndingCurrentEncoder();

    // Encoding any pending event's signalling.
    setPendingEvents();

    // Notify command queue
    mCmdQueue.onCommandBufferCommitted(get(), mQueueSerial);

    // Do the actual commit
    [get() commit];

    mCommitted = true;
}

void CommandBuffer::forceEndingCurrentEncoder()
{
    if (mActiveCommandEncoder)
    {
        mActiveCommandEncoder->endEncoding();
        mActiveCommandEncoder = nullptr;
    }
}

void CommandBuffer::setPendingEvents()
{
    for (const std::pair<mtl::SharedEventRef, uint64_t> &eventEntry : mPendingSignalEvents)
    {
        setEventImpl(eventEntry.first, eventEntry.second);
    }
    mPendingSignalEvents.clear();
}

void CommandBuffer::setEventImpl(const mtl::SharedEventRef &event, uint64_t value)
{
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
    ASSERT(!mActiveCommandEncoder || mActiveCommandEncoder->getType() != CommandEncoder::RENDER);
    // For non-render command encoder, we can safely end it, so that we can encode a signal
    // event.
    forceEndingCurrentEncoder();

    [get() encodeSignalEvent:event value:value];
#endif  // #if defined(__IPHONE_12_0) || defined(__MAC_10_14)
}

void CommandBuffer::waitEventImpl(const mtl::SharedEventRef &event, uint64_t value)
{
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
    ASSERT(!mActiveCommandEncoder || mActiveCommandEncoder->getType() != CommandEncoder::RENDER);

    forceEndingCurrentEncoder();

    // Encoding any pending event's signalling.
    setPendingEvents();

    [get() encodeWaitForEvent:event value:value];
#endif  // #if defined(__IPHONE_12_0) || defined(__MAC_10_14)
}

// CommandEncoder implementation
CommandEncoder::CommandEncoder(CommandBuffer *cmdBuffer, Type type)
    : mType(type), mCmdBuffer(*cmdBuffer)
{}

CommandEncoder::~CommandEncoder()
{
    reset();
}

void CommandEncoder::endEncoding()
{
    [get() endEncoding];
    reset();
}

void CommandEncoder::reset()
{
    ParentClass::reset();

    mCmdBuffer.invalidateActiveCommandEncoder(this);
}

void CommandEncoder::set(id<MTLCommandEncoder> metalCmdEncoder)
{
    ParentClass::set(metalCmdEncoder);

    // Set this as active encoder
    cmdBuffer().setActiveCommandEncoder(this);
}

CommandEncoder &CommandEncoder::markResourceBeingWrittenByGPU(const BufferRef &buffer)
{
    cmdBuffer().setWriteDependency(buffer);
    return *this;
}

CommandEncoder &CommandEncoder::markResourceBeingWrittenByGPU(const TextureRef &texture)
{
    cmdBuffer().setWriteDependency(texture);
    return *this;
}

void CommandEncoder::insertDebugSign(NSString *label)
{
    insertDebugSignImpl(label);
}

void CommandEncoder::insertDebugSignImpl(NSString *label)
{
    // Default implementation
    [get() insertDebugSignpost:label];
}

// RenderCommandEncoderShaderStates implementation
RenderCommandEncoderShaderStates::RenderCommandEncoderShaderStates()
{
    reset();
}

void RenderCommandEncoderShaderStates::reset()
{
    for (AutoObjCPtr<id<MTLBuffer>> &buffer : buffers)
    {
        buffer = nil;
    }

    for (uint32_t &offset : bufferOffsets)
    {
        offset = 0;
    }

    for (AutoObjCPtr<id<MTLSamplerState>> &sampler : samplers)
    {
        sampler = nil;
    }

    for (Optional<std::pair<float, float>> &lodClampRange : samplerLodClamps)
    {
        lodClampRange.reset();
    }

    for (AutoObjCPtr<id<MTLTexture>> &texture : textures)
    {
        texture = nil;
    }
}

// RenderCommandEncoderStates implementation
RenderCommandEncoderStates::RenderCommandEncoderStates()
{
    reset();
}

void RenderCommandEncoderStates::reset()
{
    renderPipeline = nil;

    triangleFillMode = MTLTriangleFillModeFill;
    winding          = MTLWindingClockwise;
    cullMode         = MTLCullModeNone;

    depthStencilState = nil;
    depthBias = depthSlopeScale = depthClamp = 0;

    stencilFrontRef = stencilBackRef = 0;

    viewport.reset();
    scissorRect.reset();

    blendColor = {0, 0, 0, 0};

    for (RenderCommandEncoderShaderStates &shaderStates : perShaderStates)
    {
        shaderStates.reset();
    }

    visibilityResultMode         = MTLVisibilityResultModeDisabled;
    visibilityResultBufferOffset = 0;
}

// RenderCommandEncoder implemtation
RenderCommandEncoder::RenderCommandEncoder(CommandBuffer *cmdBuffer,
                                           const OcclusionQueryPool &queryPool)
    : CommandEncoder(cmdBuffer, RENDER), mOcclusionQueryPool(queryPool)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        mCachedRenderPassDescObjC = [MTLRenderPassDescriptor renderPassDescriptor];
    }

    for (gl::ShaderType shaderType : gl::AllShaderTypes())
    {
        mSetBufferFuncs[shaderType]            = nullptr;
        mSetBufferOffsetFuncs[shaderType]      = nullptr;
        mSetBytesFuncs[shaderType]             = nullptr;
        mSetTextureFuncs[shaderType]           = nullptr;
        mSetSamplerFuncs[shaderType]           = nullptr;
        mSetSamplerWithoutLodFuncs[shaderType] = nullptr;
    }

    mSetBufferFuncs[gl::ShaderType::Vertex]   = &RenderCommandEncoder::mtlSetVertexBuffer;
    mSetBufferFuncs[gl::ShaderType::Fragment] = &RenderCommandEncoder::mtlSetFragmentBuffer;

    mSetBufferOffsetFuncs[gl::ShaderType::Vertex] = &RenderCommandEncoder::mtlSetVertexBufferOffset;
    mSetBufferOffsetFuncs[gl::ShaderType::Fragment] =
        &RenderCommandEncoder::mtlSetFragmentBufferOffset;

    mSetBytesFuncs[gl::ShaderType::Vertex]   = &RenderCommandEncoder::mtlSetVertexBytes;
    mSetBytesFuncs[gl::ShaderType::Fragment] = &RenderCommandEncoder::mtlSetFragmentBytes;

    mSetTextureFuncs[gl::ShaderType::Vertex]   = &RenderCommandEncoder::mtlSetVertexTexture;
    mSetTextureFuncs[gl::ShaderType::Fragment] = &RenderCommandEncoder::mtlSetFragmentTexture;

    mSetSamplerFuncs[gl::ShaderType::Vertex]   = &RenderCommandEncoder::mtlSetVertexSamplerState;
    mSetSamplerFuncs[gl::ShaderType::Fragment] = &RenderCommandEncoder::mtlSetFragmentSamplerState;

    mSetSamplerWithoutLodFuncs[gl::ShaderType::Vertex] =
        &RenderCommandEncoder::mtlSetVertexSamplerState;
    mSetSamplerWithoutLodFuncs[gl::ShaderType::Fragment] =
        &RenderCommandEncoder::mtlSetFragmentSamplerState;
}
RenderCommandEncoder::~RenderCommandEncoder() {}

void RenderCommandEncoder::reset()
{
    CommandEncoder::reset();

    if (mRecording)
    {
        mStateCache.reset();

        mDeferredLabel     = nil;
        mDeferredDebugSign = nil;
        mDeferredDebugGroups.clear();
    }

    mRecording = false;
}

MTLStoreAction RenderCommandEncoder::correctStoreAction(
    MTLRenderPassAttachmentDescriptor *objCRenderPassAttachment,
    MTLStoreAction finalStoreAction)
{
    MTLStoreAction storeAction = finalStoreAction;

    if (objCRenderPassAttachment.resolveTexture)
    {
        if (finalStoreAction == MTLStoreActionStore)
        {
            // NOTE(hqle): Currently if the store action with implicit MS texture is MTLStoreAction,
            // it is automatically convert to store and resolve action. It might introduce
            // unnecessary overhead.
            // Consider an improvement such as only store the MS texture, and resolve only at
            // the end of real render pass (not render pass the was interrupted by compute pass)
            // or before glBlitFramebuffer operation starts.
            storeAction = MTLStoreActionStoreAndMultisampleResolve;
        }
        else if (finalStoreAction == MTLStoreActionDontCare)
        {
            // NOTE(hqle): If there is a resolve texture in the render pass, we cannot set
            // storeAction=MTLStoreActionDontCare. Use MTLStoreActionMultisampleResolve instead.
            storeAction = MTLStoreActionMultisampleResolve;
        }
    }

    if (finalStoreAction == MTLStoreActionUnknown)
    {
        // If storeAction hasn't been set for this attachment, we set to dontcare.
        storeAction = MTLStoreActionDontCare;
    }

    return storeAction;
}

void RenderCommandEncoder::endEncoding()
{
    endEncodingImpl(true);
}

void RenderCommandEncoder::endEncodingImpl(bool considerDiscardSimulation)
{
    if (!valid())
        return;

    ensureMetalEncoderStarted();

    // Last minute correcting the store options.
    MTLRenderPassDescriptor *objCRenderPassDesc = mCachedRenderPassDescObjC.get();
    for (uint32_t i = 0; i < mRenderPassDesc.numColorAttachments; ++i)
    {
        if (!objCRenderPassDesc.colorAttachments[i].texture)
        {
            continue;
        }
        // Update store action set between restart() and endEncoding()
        MTLStoreAction storeAction =
            correctStoreAction(objCRenderPassDesc.colorAttachments[i],
                               mRenderPassDesc.colorAttachments[i].storeAction);
        ANGLE_MTL_LOG("setColorStoreAction:%lu atIndex:%u", storeAction, i);
        [get() setColorStoreAction:storeAction atIndex:i];
    }

    // Update store action set between restart() and endEncoding()
    if (objCRenderPassDesc.depthAttachment.texture)
    {
        MTLStoreAction storeAction = correctStoreAction(
            objCRenderPassDesc.depthAttachment, mRenderPassDesc.depthAttachment.storeAction);
        [get() setDepthStoreAction:storeAction];
    }
    // Update store action set between restart() and endEncoding()
    if (objCRenderPassDesc.stencilAttachment.texture)
    {
        MTLStoreAction storeAction = correctStoreAction(
            objCRenderPassDesc.stencilAttachment, mRenderPassDesc.stencilAttachment.storeAction);
        [get() setStencilStoreAction:storeAction];
    }

    CommandEncoder::endEncoding();

#if ANGLE_MTL_SIMULATE_DISCARD_FRAMEBUFFER
    if (considerDiscardSimulation)
    {
        simulateDiscardFramebuffer();
    }
#endif

    // reset state
    mRenderPassDesc = RenderPassDesc();
}

inline void RenderCommandEncoder::initAttachmentWriteDependencyAndScissorRect(
    const RenderPassAttachmentDesc &attachment)
{
    TextureRef texture = attachment.texture();
    if (texture)
    {
        cmdBuffer().setWriteDependency(texture);

        uint32_t mipLevel = attachment.level();

        mRenderPassMaxScissorRect.width =
            std::min<NSUInteger>(mRenderPassMaxScissorRect.width, texture->width(mipLevel));
        mRenderPassMaxScissorRect.height =
            std::min<NSUInteger>(mRenderPassMaxScissorRect.height, texture->height(mipLevel));
    }
}

void RenderCommandEncoder::simulateDiscardFramebuffer()
{
    // Simulate true framebuffer discard operation by clearing the framebuffer
#if ANGLE_MTL_SIMULATE_DISCARD_FRAMEBUFFER
    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0f);
    bool hasDiscard = false;
    for (uint32_t i = 0; i < mRenderPassDesc.numColorAttachments; ++i)
    {
        if (mRenderPassDesc.colorAttachments[i].storeAction == MTLStoreActionDontCare)
        {
            hasDiscard                                     = true;
            mRenderPassDesc.colorAttachments[i].loadAction = MTLLoadActionClear;
            mRenderPassDesc.colorAttachments[i].clearColor =
                MTLClearColorMake(dis(gen), dis(gen), dis(gen), dis(gen));
        }
        else
        {
            mRenderPassDesc.colorAttachments[i].loadAction = MTLLoadActionLoad;
        }
    }

    if (mRenderPassDesc.depthAttachment.storeAction == MTLStoreActionDontCare)
    {
        hasDiscard                                 = true;
        mRenderPassDesc.depthAttachment.loadAction = MTLLoadActionClear;
        mRenderPassDesc.depthAttachment.clearDepth = dis(gen);
    }
    else
    {
        mRenderPassDesc.depthAttachment.loadAction = MTLLoadActionLoad;
    }

    if (mRenderPassDesc.stencilAttachment.storeAction == MTLStoreActionDontCare)
    {
        hasDiscard                                     = true;
        mRenderPassDesc.stencilAttachment.loadAction   = MTLLoadActionClear;
        mRenderPassDesc.stencilAttachment.clearStencil = rand();
    }
    else
    {
        mRenderPassDesc.stencilAttachment.loadAction = MTLLoadActionLoad;
    }

    if (hasDiscard)
    {
        auto tmpDesc = mRenderPassDesc;
        restart(tmpDesc);
        endEncodingImpl(false);
    }
#endif  // ANGLE_MTL_SIMULATE_DISCARD_FRAMEBUFFER
}

void RenderCommandEncoder::ensureMetalEncoderStarted()
{
    if (get())
    {
        return;
    }

    ANGLE_MTL_OBJC_SCOPE
    {
        // Set store action to unknown so that we can set proper value later.
        MTLRenderPassDescriptor *objCRenderPassDesc = mCachedRenderPassDescObjC.get();
        for (uint32_t i = 0; i < mRenderPassDesc.numColorAttachments; ++i)
        {
            if (objCRenderPassDesc.colorAttachments[i].texture)
            {
                objCRenderPassDesc.colorAttachments[i].storeAction = MTLStoreActionUnknown;
            }
        }

        if (objCRenderPassDesc.depthAttachment.texture)
        {
            objCRenderPassDesc.depthAttachment.storeAction = MTLStoreActionUnknown;
        }
        if (objCRenderPassDesc.stencilAttachment.texture)
        {
            objCRenderPassDesc.stencilAttachment.storeAction = MTLStoreActionUnknown;
        }

        // Set visibility result buffer
        if (mOcclusionQueryPool.getRenderPassVisibilityPoolBuffer())
        {
            objCRenderPassDesc.visibilityResultBuffer =
                mOcclusionQueryPool.getRenderPassVisibilityPoolBuffer()->get();
        }
        else
        {
            objCRenderPassDesc.visibilityResultBuffer = nil;
        }

        ANGLE_MTL_LOG("Creating new render command encoder with desc: %@", objCRenderPassDesc);

        id<MTLRenderCommandEncoder> metalCmdEncoder =
            [cmdBuffer().get() renderCommandEncoderWithDescriptor:objCRenderPassDesc];

        set(metalCmdEncoder);

        // Verify that it was created successfully
        ASSERT(get());

        applyStates();
    }
}

void RenderCommandEncoder::applyStates()
{
    // Apply the current cached states
    if (mDeferredLabel)
    {
        mtlSetLabel(mDeferredLabel);
    }
    mtlInsertDebugSign(mDeferredDebugSign);

    for (mtl::AutoObjCObj<NSString> &group : mDeferredDebugGroups)
    {
        mtlPushDebugGroup(group);
    }

    mtlSetTriangleFillMode(mStateCache.triangleFillMode);
    mtlSetFrontFacingWinding(mStateCache.winding);
    mtlSetCullMode(mStateCache.cullMode);
    mtlSetDepthBias(mStateCache.depthBias, mStateCache.depthSlopeScale, mStateCache.depthClamp);
    mtlSetStencilRefVals(mStateCache.stencilFrontRef, mStateCache.stencilBackRef);
    mtlSetBlendColor(mStateCache.blendColor[0], mStateCache.blendColor[1],
                     mStateCache.blendColor[2], mStateCache.blendColor[3]);

    if (mStateCache.renderPipeline)
    {
        mtlSetRenderPipelineState(mStateCache.renderPipeline);
    }
    if (mStateCache.depthStencilState)
    {
        mtlSetDepthStencilState(mStateCache.depthStencilState);
    }
    if (mStateCache.viewport.valid())
    {
        mtlSetViewport(mStateCache.viewport.value());
    }
    if (mStateCache.scissorRect.valid())
    {
        mtlSetScissorRect(mStateCache.scissorRect.value());
    }
    if (mStateCache.visibilityResultMode != MTLVisibilityResultModeDisabled)
    {
        mtlSetVisibilityResultMode(mStateCache.visibilityResultMode,
                                   mStateCache.visibilityResultBufferOffset);
    }

    for (gl::ShaderType shaderType : gl::AllGLES2ShaderTypes())
    {
        const RenderCommandEncoderShaderStates &shaderStates =
            mStateCache.perShaderStates[shaderType];
        for (uint32_t i = 0; i < kMaxShaderBuffers; ++i)
        {
            if (shaderStates.buffers[i])
            {
                (this->*mSetBufferFuncs[shaderType])(shaderStates.buffers[i],
                                                     shaderStates.bufferOffsets[i], i);
            }
        }

        for (uint32_t i = 0; i < kMaxShaderSamplers; ++i)
        {
            if (shaderStates.samplers[i])
            {
                if (shaderStates.samplerLodClamps[i].valid())
                {
                    const std::pair<float, float> &clamps =
                        shaderStates.samplerLodClamps[i].value();
                    (this->*mSetSamplerFuncs[shaderType])(shaderStates.samplers[i], clamps.first,
                                                          clamps.second, i);
                }
                else
                {
                    (this->*mSetSamplerWithoutLodFuncs[shaderType])(shaderStates.samplers[i], i);
                }
            }

            if (shaderStates.textures[i])
            {
                (this->*mSetTextureFuncs[shaderType])(shaderStates.textures[i], i);
            }
        }
    }
}

RenderCommandEncoder &RenderCommandEncoder::restart(const RenderPassDesc &desc)
{
    if (valid())
    {
        if (mRenderPassDesc == desc)
        {
            // no change, skip
            return *this;
        }

        // finish current encoder
        endEncoding();
    }

    if (!cmdBuffer().valid())
    {
        reset();
        return *this;
    }

    mRenderPassDesc            = desc;
    mRecording                 = true;
    mWarnOutOfBoundScissorRect = true;
    mRenderPassMaxScissorRect  = {.x      = 0,
                                 .y      = 0,
                                 .width  = std::numeric_limits<NSUInteger>::max(),
                                 .height = std::numeric_limits<NSUInteger>::max()};

    // mask writing dependency & set appropriate store options
    for (uint32_t i = 0; i < mRenderPassDesc.numColorAttachments; ++i)
    {
        initAttachmentWriteDependencyAndScissorRect(mRenderPassDesc.colorAttachments[i]);
    }

    initAttachmentWriteDependencyAndScissorRect(mRenderPassDesc.depthAttachment);

    initAttachmentWriteDependencyAndScissorRect(mRenderPassDesc.stencilAttachment);

    // Convert to Objective-C descriptor
    mRenderPassDesc.convertToMetalDesc(mCachedRenderPassDescObjC);

    // The actual Objective-C encoder will be created later in endEncoding(), we do so in order
    // to be able to sort the commands or do the preprocessing before the actual encoding.

    // Since we defer the native encoder creation, we need to explicitly tell command buffer that
    // this object is the active encoder:
    cmdBuffer().setActiveCommandEncoder(this);

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setRenderPipelineState(id<MTLRenderPipelineState> state)
{
    if (mStateCache.renderPipeline == state)
    {
        return *this;
    }
    mStateCache.renderPipeline.retainAssign(state);

    if (get())
    {
        mtlSetRenderPipelineState(state);
    }

    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::setTriangleFillMode(MTLTriangleFillMode mode)
{
    if (mStateCache.triangleFillMode == mode)
    {
        return *this;
    }
    mStateCache.triangleFillMode = mode;

    if (get())
    {
        mtlSetTriangleFillMode(mode);
    }

    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::setFrontFacingWinding(MTLWinding winding)
{
    if (mStateCache.winding == winding)
    {
        return *this;
    }
    mStateCache.winding = winding;

    if (get())
    {
        mtlSetFrontFacingWinding(winding);
    }

    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::setCullMode(MTLCullMode mode)
{
    if (mStateCache.cullMode == mode)
    {
        return *this;
    }
    mStateCache.cullMode = mode;

    if (get())
    {
        mtlSetCullMode(mode);
    }

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setDepthStencilState(id<MTLDepthStencilState> state)
{
    if (mStateCache.depthStencilState == state)
    {
        return *this;
    }
    mStateCache.depthStencilState.retainAssign(state);

    if (get())
    {
        mtlSetDepthStencilState(state);
    }

    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::setDepthBias(float depthBias,
                                                         float slopeScale,
                                                         float clamp)
{
    if (mStateCache.depthBias == depthBias && mStateCache.depthSlopeScale == slopeScale &&
        mStateCache.depthClamp == clamp)
    {
        return *this;
    }
    mStateCache.depthBias       = depthBias;
    mStateCache.depthSlopeScale = slopeScale;
    mStateCache.depthClamp      = clamp;

    if (get())
    {
        mtlSetDepthBias(depthBias, slopeScale, clamp);
    }

    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::setStencilRefVals(uint32_t frontRef, uint32_t backRef)
{
    // Metal has some bugs when reference values are larger than 0xff
    ASSERT(frontRef == (frontRef & kStencilMaskAll));
    ASSERT(backRef == (backRef & kStencilMaskAll));

    if (mStateCache.stencilFrontRef == frontRef && mStateCache.stencilBackRef == backRef)
    {
        return *this;
    }
    mStateCache.stencilFrontRef = frontRef;
    mStateCache.stencilBackRef  = backRef;

    if (get())
    {
        mtlSetStencilRefVals(frontRef, backRef);
    }

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setStencilRefVal(uint32_t ref)
{
    return setStencilRefVals(ref, ref);
}

RenderCommandEncoder &RenderCommandEncoder::setViewport(const MTLViewport &viewport)
{
    if (mStateCache.viewport.valid() && mStateCache.viewport.value() == viewport)
    {
        return *this;
    }
    mStateCache.viewport = viewport;

    if (get())
    {
        mtlSetViewport(viewport);
    }

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setScissorRect(const MTLScissorRect &rect)
{
    if (mStateCache.scissorRect.valid() && mStateCache.scissorRect.value() == rect)
    {
        return *this;
    }

    if (rect.x + rect.width > mRenderPassMaxScissorRect.width ||
        rect.y + rect.height > mRenderPassMaxScissorRect.height)
    {
        if (mWarnOutOfBoundScissorRect)
        {
            WARN() << "Out of bound scissor rect detected " << rect.x << " " << rect.y << " "
                   << rect.width << " " << rect.height;
        }
        // Out of bound rect will crash the metal runtime, ignore it.
        return *this;
    }

    mStateCache.scissorRect = rect;

    if (get())
    {
        mtlSetScissorRect(rect);
    }

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setBlendColor(float r, float g, float b, float a)
{
    if (mStateCache.blendColor[0] == r && mStateCache.blendColor[1] == g &&
        mStateCache.blendColor[2] == b && mStateCache.blendColor[3] == a)
    {
        return *this;
    }
    mStateCache.blendColor[0] = r;
    mStateCache.blendColor[1] = g;
    mStateCache.blendColor[2] = b;
    mStateCache.blendColor[3] = a;

    if (get())
    {
        mtlSetBlendColor(r, g, b, a);
    }

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setBuffer(gl::ShaderType shaderType,
                                                      const BufferRef &buffer,
                                                      uint32_t offset,
                                                      uint32_t index)
{
    if (index >= kMaxShaderBuffers)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(buffer);

    id<MTLBuffer> mtlBuffer = (buffer ? buffer->get() : nil);

    return commonSetBuffer(shaderType, mtlBuffer, offset, index);
}

RenderCommandEncoder &RenderCommandEncoder::setBufferForWrite(gl::ShaderType shaderType,
                                                              const BufferRef &buffer,
                                                              uint32_t offset,
                                                              uint32_t index)
{
    if (index >= kMaxShaderBuffers)
    {
        return *this;
    }

    cmdBuffer().setWriteDependency(buffer);

    id<MTLBuffer> mtlBuffer = (buffer ? buffer->get() : nil);

    return commonSetBuffer(shaderType, mtlBuffer, offset, index);
}

RenderCommandEncoder &RenderCommandEncoder::commonSetBuffer(gl::ShaderType shaderType,
                                                            id<MTLBuffer> mtlBuffer,
                                                            uint32_t offset,
                                                            uint32_t index)
{
    RenderCommandEncoderShaderStates &shaderStates = mStateCache.perShaderStates[shaderType];
    if (shaderStates.buffers[index] == mtlBuffer)
    {
        if (shaderStates.bufferOffsets[index] == offset)
        {
            return *this;
        }

        // If buffer already bound but with different offset, then update the offer only.
        shaderStates.bufferOffsets[index] = offset;

        if (get())
        {
            (this->*mSetBufferOffsetFuncs[shaderType])(offset, index);
        }
        return *this;
    }

    shaderStates.buffers[index].retainAssign(mtlBuffer);
    shaderStates.bufferOffsets[index] = offset;

    if (get())
    {
        (this->*mSetBufferFuncs[shaderType])(mtlBuffer, offset, index);
    }

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setBytes(gl::ShaderType shaderType,
                                                     const uint8_t *bytes,
                                                     size_t size,
                                                     uint32_t index)
{
    if (index >= kMaxShaderBuffers)
    {
        return *this;
    }

    // NOTE(hqle): find an efficient way to cache inline data.
    ensureMetalEncoderStarted();

    RenderCommandEncoderShaderStates &shaderStates = mStateCache.perShaderStates[shaderType];
    shaderStates.buffers[index]                    = nil;
    shaderStates.bufferOffsets[index]              = 0;

    (this->*mSetBytesFuncs[shaderType])(bytes, size, index);

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setSamplerState(gl::ShaderType shaderType,
                                                            id<MTLSamplerState> state,
                                                            float lodMinClamp,
                                                            float lodMaxClamp,
                                                            uint32_t index)
{
    if (index >= kMaxShaderSamplers)
    {
        return *this;
    }

    RenderCommandEncoderShaderStates &shaderStates = mStateCache.perShaderStates[shaderType];
    if (shaderStates.samplers[index] == state && shaderStates.samplerLodClamps[index].valid())
    {
        const std::pair<float, float> &currentLodClampRange =
            shaderStates.samplerLodClamps[index].value();
        if (currentLodClampRange.first == lodMinClamp && currentLodClampRange.second == lodMaxClamp)
        {
            return *this;
        }
    }

    shaderStates.samplers[index].retainAssign(state);
    shaderStates.samplerLodClamps[index] = {lodMinClamp, lodMaxClamp};

    if (get())
    {
        (this->*mSetSamplerFuncs[shaderType])(state, lodMinClamp, lodMaxClamp, index);
    }

    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::setTexture(gl::ShaderType shaderType,
                                                       const TextureRef &texture,
                                                       uint32_t index)
{
    if (index >= kMaxShaderSamplers)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(texture);

    id<MTLTexture> mtlTexture = (texture ? texture->get() : nil);

    RenderCommandEncoderShaderStates &shaderStates = mStateCache.perShaderStates[shaderType];
    if (shaderStates.textures[index] == mtlTexture)
    {
        return *this;
    }
    shaderStates.textures[index].retainAssign(mtlTexture);

    if (get())
    {
        (this->*mSetTextureFuncs[shaderType])(mtlTexture, index);
    }

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::draw(MTLPrimitiveType primitiveType,
                                                 uint32_t vertexStart,
                                                 uint32_t vertexCount)
{
    ensureMetalEncoderStarted();
    [get() drawPrimitives:primitiveType vertexStart:vertexStart vertexCount:vertexCount];

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::drawInstanced(MTLPrimitiveType primitiveType,
                                                          uint32_t vertexStart,
                                                          uint32_t vertexCount,
                                                          uint32_t instances)
{
    ensureMetalEncoderStarted();
    [get() drawPrimitives:primitiveType
              vertexStart:vertexStart
              vertexCount:vertexCount
            instanceCount:instances];

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::drawIndexed(MTLPrimitiveType primitiveType,
                                                        uint32_t indexCount,
                                                        MTLIndexType indexType,
                                                        const BufferRef &indexBuffer,
                                                        size_t bufferOffset)
{
    if (!indexBuffer)
    {
        return *this;
    }

    ensureMetalEncoderStarted();
    cmdBuffer().setReadDependency(indexBuffer);

    [get() drawIndexedPrimitives:primitiveType
                      indexCount:indexCount
                       indexType:indexType
                     indexBuffer:indexBuffer->get()
               indexBufferOffset:bufferOffset];

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::drawIndexedInstanced(MTLPrimitiveType primitiveType,
                                                                 uint32_t indexCount,
                                                                 MTLIndexType indexType,
                                                                 const BufferRef &indexBuffer,
                                                                 size_t bufferOffset,
                                                                 uint32_t instances)
{
    if (!indexBuffer)
    {
        return *this;
    }

    ensureMetalEncoderStarted();
    cmdBuffer().setReadDependency(indexBuffer);

    [get() drawIndexedPrimitives:primitiveType
                      indexCount:indexCount
                       indexType:indexType
                     indexBuffer:indexBuffer->get()
               indexBufferOffset:bufferOffset
                   instanceCount:instances];

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::drawIndexedInstancedBaseVertex(
    MTLPrimitiveType primitiveType,
    uint32_t indexCount,
    MTLIndexType indexType,
    const BufferRef &indexBuffer,
    size_t bufferOffset,
    uint32_t instances,
    uint32_t baseVertex)
{
    if (!indexBuffer)
    {
        return *this;
    }

    ensureMetalEncoderStarted();
    cmdBuffer().setReadDependency(indexBuffer);

    [get() drawIndexedPrimitives:primitiveType
                      indexCount:indexCount
                       indexType:indexType
                     indexBuffer:indexBuffer->get()
               indexBufferOffset:bufferOffset
                   instanceCount:instances
                      baseVertex:baseVertex
                    baseInstance:0];

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setVisibilityResultMode(MTLVisibilityResultMode mode,
                                                                    size_t offset)
{
    if (mStateCache.visibilityResultMode == mode &&
        mStateCache.visibilityResultBufferOffset == offset)
    {
        return *this;
    }
    mStateCache.visibilityResultMode         = mode;
    mStateCache.visibilityResultBufferOffset = offset;

    if (get())
    {
        mtlSetVisibilityResultMode(mode, offset);
    }
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::useResource(const BufferRef &resource,
                                                        MTLResourceUsage usage,
                                                        mtl::RenderStages stages)
{
    if (!resource)
    {
        return *this;
    }

    // NOTE(hqle): Find an efficient way to cache resource usage.
    ensureMetalEncoderStarted();

    cmdBuffer().setReadDependency(resource);

    mtlUseResource(resource->get(), usage, stages);

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::memoryBarrierWithResource(const BufferRef &resource,
                                                                      mtl::RenderStages after,
                                                                      mtl::RenderStages before)
{
    if (!resource)
    {
        return *this;
    }

    // NOTE(hqle): Find an efficient way to cache resource barrier.
    ensureMetalEncoderStarted();
    cmdBuffer().setWriteDependency(resource);

    mtlMemoryBarrierWithResource(resource->get(), after, before);

    return *this;
}

void RenderCommandEncoder::insertDebugSignImpl(NSString *label)
{
    if (get())
    {
        mtlInsertDebugSign(label);
        return;
    }
    // Defer the insertion
    mDeferredDebugSign.retainAssign(label);
}

void RenderCommandEncoder::pushDebugGroup(NSString *label)
{
    if (get())
    {
        mtlPushDebugGroup(label);
        return;
    }
    // Defer the insertion
    mtl::AutoObjCObj<NSString> retainedLabel;
    retainedLabel.retainAssign(label);
    mDeferredDebugGroups.push_back(retainedLabel);
}
void RenderCommandEncoder::popDebugGroup()
{
    if (get())
    {
        mtlPopDebugGroup();
        return;
    }

    mDeferredDebugGroups.pop_back();
}

void RenderCommandEncoder::setLabel(NSString *label)
{
    if (get())
    {
        mtlSetLabel(label);
        return;
    }
    mDeferredLabel.retainAssign(label);
}

RenderCommandEncoder &RenderCommandEncoder::setColorStoreAction(MTLStoreAction action,
                                                                uint32_t colorAttachmentIndex)
{
    if (colorAttachmentIndex >= mRenderPassDesc.numColorAttachments)
    {
        return *this;
    }

    // We only store the options, will defer the actual setting until the encoder finishes
    mRenderPassDesc.colorAttachments[colorAttachmentIndex].storeAction = action;

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setColorStoreAction(MTLStoreAction action)
{
    for (uint32_t i = 0; i < mRenderPassDesc.numColorAttachments; ++i)
    {
        setColorStoreAction(action, i);
    }
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setDepthStencilStoreAction(
    MTLStoreAction depthStoreAction,
    MTLStoreAction stencilStoreAction)
{
    // We only store the options, will defer the actual setting until the encoder finishes
    mRenderPassDesc.depthAttachment.storeAction   = depthStoreAction;
    mRenderPassDesc.stencilAttachment.storeAction = stencilStoreAction;

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setDepthStoreAction(MTLStoreAction action)
{
    // We only store the options, will defer the actual setting until the encoder finishes
    mRenderPassDesc.depthAttachment.storeAction = action;

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setStencilStoreAction(MTLStoreAction action)
{
    // We only store the options, will defer the actual setting until the encoder finishes
    mRenderPassDesc.stencilAttachment.storeAction = action;

    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setStoreAction(MTLStoreAction action)
{
    setColorStoreAction(action);
    setDepthStencilStoreAction(action, action);
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setColorLoadAction(MTLLoadAction action,
                                                               const MTLClearColor &clearValue,
                                                               uint32_t colorAttachmentIndex)
{
    ASSERT(canChangeLoadAction());
    if (mCachedRenderPassDescObjC.get().colorAttachments[colorAttachmentIndex].texture)
    {
        mCachedRenderPassDescObjC.get().colorAttachments[colorAttachmentIndex].loadAction = action;
        mCachedRenderPassDescObjC.get().colorAttachments[colorAttachmentIndex].clearColor =
            clearValue;
    }
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setDepthLoadAction(MTLLoadAction action,
                                                               double clearVal)
{
    ASSERT(canChangeLoadAction());
    if (mCachedRenderPassDescObjC.get().depthAttachment.texture)
    {
        mCachedRenderPassDescObjC.get().depthAttachment.loadAction = action;
        mCachedRenderPassDescObjC.get().depthAttachment.clearDepth = clearVal;
    }
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::setStencilLoadAction(MTLLoadAction action,
                                                                 uint32_t clearVal)
{
    ASSERT(canChangeLoadAction());
    if (mCachedRenderPassDescObjC.get().stencilAttachment.texture)
    {
        mCachedRenderPassDescObjC.get().stencilAttachment.loadAction   = action;
        mCachedRenderPassDescObjC.get().stencilAttachment.clearStencil = clearVal;
    }
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetRenderPipelineState(
    id<MTLRenderPipelineState> state)
{
    [get() setRenderPipelineState:state];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetTriangleFillMode(MTLTriangleFillMode mode)
{
    [get() setTriangleFillMode:mode];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetFrontFacingWinding(MTLWinding winding)
{
    [get() setFrontFacingWinding:winding];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetCullMode(MTLCullMode mode)
{
    [get() setCullMode:mode];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetDepthStencilState(id<MTLDepthStencilState> state)
{
    [get() setDepthStencilState:state];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetDepthBias(float depthBias,
                                                            float slopeScale,
                                                            float clamp)
{
    [get() setDepthBias:depthBias slopeScale:slopeScale clamp:clamp];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetStencilRefVals(uint32_t frontRef,
                                                                 uint32_t backRef)
{
    [get() setStencilFrontReferenceValue:frontRef backReferenceValue:backRef];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetViewport(const MTLViewport &viewport)
{
    [get() setViewport:viewport];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetScissorRect(const MTLScissorRect &rect)
{
    [get() setScissorRect:rect];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetBlendColor(float r, float g, float b, float a)
{
    [get() setBlendColorRed:r green:g blue:b alpha:a];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetVertexBuffer(id<MTLBuffer> buffer,
                                                               uint32_t offset,
                                                               uint32_t index)
{
    [get() setVertexBuffer:buffer offset:offset atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetVertexBufferOffset(uint32_t offset,
                                                                     uint32_t index)
{
    [get() setVertexBufferOffset:offset atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetVertexBytes(const uint8_t *bytes,
                                                              size_t size,
                                                              uint32_t index)
{
    [get() setVertexBytes:bytes length:size atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetVertexSamplerState(id<MTLSamplerState> state,
                                                                     uint32_t index)
{
    [get() setVertexSamplerState:state atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetVertexSamplerState(id<MTLSamplerState> state,
                                                                     float lodMinClamp,
                                                                     float lodMaxClamp,
                                                                     uint32_t index)
{
    [get() setVertexSamplerState:state
                     lodMinClamp:lodMinClamp
                     lodMaxClamp:lodMaxClamp
                         atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetVertexTexture(id<MTLTexture> texture,
                                                                uint32_t index)
{
    [get() setVertexTexture:texture atIndex:index];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetFragmentBuffer(id<MTLBuffer> buffer,
                                                                 uint32_t offset,
                                                                 uint32_t index)
{
    [get() setFragmentBuffer:buffer offset:offset atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetFragmentBufferOffset(uint32_t offset,
                                                                       uint32_t index)
{
    [get() setFragmentBufferOffset:offset atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetFragmentBytes(const uint8_t *bytes,
                                                                size_t size,
                                                                uint32_t index)
{
    [get() setFragmentBytes:bytes length:size atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetFragmentSamplerState(id<MTLSamplerState> state,
                                                                       uint32_t index)
{
    [get() setFragmentSamplerState:state atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetFragmentSamplerState(id<MTLSamplerState> state,
                                                                       float lodMinClamp,
                                                                       float lodMaxClamp,
                                                                       uint32_t index)
{
    [get() setFragmentSamplerState:state
                       lodMinClamp:lodMinClamp
                       lodMaxClamp:lodMaxClamp
                           atIndex:index];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlSetFragmentTexture(id<MTLTexture> texture,
                                                                  uint32_t index)
{
    [get() setFragmentTexture:texture atIndex:index];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetVisibilityResultMode(MTLVisibilityResultMode mode,
                                                                       size_t offset)
{
    [get() setVisibilityResultMode:mode offset:offset];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlUseResource(id<MTLBuffer> resource,
                                                           MTLResourceUsage usage,
                                                           mtl::RenderStages stages)
{
    ANGLE_UNUSED_VARIABLE(stages);
#if defined(__IPHONE_13_0) || defined(__MAC_10_15)
    if (ANGLE_APPLE_AVAILABLE_XCI(10.15, 13.0, 13.0))
    {
        [get() useResource:resource usage:usage stages:stages];
    }
    else
#endif
    {
        [get() useResource:resource usage:usage];
    }
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlMemoryBarrierWithResource(id<MTLBuffer> resource,
                                                                         mtl::RenderStages after,
                                                                         mtl::RenderStages before)
{
    ANGLE_UNUSED_VARIABLE(after);
    ANGLE_UNUSED_VARIABLE(before);
#if defined(__MAC_10_14) && (TARGET_OS_OSX || TARGET_OS_MACCATALYST)
    if (ANGLE_APPLE_AVAILABLE_XC(10.14, 13.0))
    {
        [get() memoryBarrierWithResources:&resource count:1 afterStages:after beforeStages:before];
    }
#endif
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlInsertDebugSign(NSString *label)
{
    if (!label)
    {
        return *this;
    }
    [get() insertDebugSignpost:label];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlPushDebugGroup(NSString *label)
{
    [get() pushDebugGroup:label];
    return *this;
}
RenderCommandEncoder &RenderCommandEncoder::mtlPopDebugGroup()
{
    [get() popDebugGroup];
    return *this;
}

RenderCommandEncoder &RenderCommandEncoder::mtlSetLabel(NSString *label)
{
    get().label = label;
    return *this;
}

// BlitCommandEncoder
BlitCommandEncoder::BlitCommandEncoder(CommandBuffer *cmdBuffer) : CommandEncoder(cmdBuffer, BLIT)
{}

BlitCommandEncoder::~BlitCommandEncoder() {}

BlitCommandEncoder &BlitCommandEncoder::restart()
{
    ANGLE_MTL_OBJC_SCOPE
    {
        if (valid())
        {
            // no change, skip
            return *this;
        }

        if (!cmdBuffer().valid())
        {
            reset();
            return *this;
        }

        // Create objective C object
        set([cmdBuffer().get() blitCommandEncoder]);

        // Verify that it was created successfully
        ASSERT(get());

        return *this;
    }
}

BlitCommandEncoder &BlitCommandEncoder::copyBuffer(const BufferRef &src,
                                                   size_t srcOffset,
                                                   const BufferRef &dst,
                                                   size_t dstOffset,
                                                   size_t size)
{
    if (!src || !dst)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(src);
    cmdBuffer().setWriteDependency(dst);

    [get() copyFromBuffer:src->get()
             sourceOffset:srcOffset
                 toBuffer:dst->get()
        destinationOffset:dstOffset
                     size:size];

    return *this;
}

BlitCommandEncoder &BlitCommandEncoder::copyBufferToTexture(const BufferRef &src,
                                                            size_t srcOffset,
                                                            size_t srcBytesPerRow,
                                                            size_t srcBytesPerImage,
                                                            MTLSize srcSize,
                                                            const TextureRef &dst,
                                                            uint32_t dstSlice,
                                                            uint32_t dstLevel,
                                                            MTLOrigin dstOrigin,
                                                            MTLBlitOption blitOption)
{
    if (!src || !dst)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(src);
    cmdBuffer().setWriteDependency(dst);

    [get() copyFromBuffer:src->get()
               sourceOffset:srcOffset
          sourceBytesPerRow:srcBytesPerRow
        sourceBytesPerImage:srcBytesPerImage
                 sourceSize:srcSize
                  toTexture:dst->get()
           destinationSlice:dstSlice
           destinationLevel:dstLevel
          destinationOrigin:dstOrigin
                    options:blitOption];

    return *this;
}

BlitCommandEncoder &BlitCommandEncoder::copyTextureToBuffer(const TextureRef &src,
                                                            uint32_t srcSlice,
                                                            uint32_t srcLevel,
                                                            MTLOrigin srcOrigin,
                                                            MTLSize srcSize,
                                                            const BufferRef &dst,
                                                            size_t dstOffset,
                                                            size_t dstBytesPerRow,
                                                            size_t dstBytesPerImage,
                                                            MTLBlitOption blitOption)
{

    if (!src || !dst)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(src);
    cmdBuffer().setWriteDependency(dst);

    [get() copyFromTexture:src->get()
                     sourceSlice:srcSlice
                     sourceLevel:srcLevel
                    sourceOrigin:srcOrigin
                      sourceSize:srcSize
                        toBuffer:dst->get()
               destinationOffset:dstOffset
          destinationBytesPerRow:dstBytesPerRow
        destinationBytesPerImage:dstBytesPerImage
                         options:blitOption];

    return *this;
}

BlitCommandEncoder &BlitCommandEncoder::copyTexture(const TextureRef &src,
                                                    uint32_t srcSlice,
                                                    uint32_t srcLevel,
                                                    MTLOrigin srcOrigin,
                                                    MTLSize srcSize,
                                                    const TextureRef &dst,
                                                    uint32_t dstSlice,
                                                    uint32_t dstLevel,
                                                    MTLOrigin dstOrigin)
{
    if (!src || !dst)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(src);
    cmdBuffer().setWriteDependency(dst);
    [get() copyFromTexture:src->get()
               sourceSlice:srcSlice
               sourceLevel:srcLevel
              sourceOrigin:srcOrigin
                sourceSize:srcSize
                 toTexture:dst->get()
          destinationSlice:dstSlice
          destinationLevel:dstLevel
         destinationOrigin:dstOrigin];

    return *this;
}

BlitCommandEncoder &BlitCommandEncoder::copyTexture(const TextureRef &src,
                                                    uint32_t srcStartSlice,
                                                    uint32_t srcStartLevel,
                                                    const TextureRef &dst,
                                                    uint32_t dstStartSlice,
                                                    uint32_t dstStartLevel,
                                                    uint32_t sliceCount,
                                                    uint32_t levelCount)
{
    if (!src || !dst)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(src);
    cmdBuffer().setWriteDependency(dst);

#if defined(__IPHONE_13_0) || defined(__MAC_10_15)
    if (ANGLE_APPLE_AVAILABLE_XCI(10.15, 13.0, 13.0))
    {
        [get() copyFromTexture:src->get()
                   sourceSlice:srcStartSlice
                   sourceLevel:srcStartLevel
                     toTexture:dst->get()
              destinationSlice:dstStartSlice
              destinationLevel:dstStartLevel
                    sliceCount:sliceCount
                    levelCount:levelCount];
    }
    else
#endif
    {
        MTLOrigin origin = MTLOriginMake(0, 0, 0);
        for (uint32_t slice = 0; slice < sliceCount; ++slice)
        {
            uint32_t srcSlice = srcStartSlice + slice;
            uint32_t dstSlice = dstStartSlice + slice;
            for (uint32_t level = 0; level < levelCount; ++level)
            {
                uint32_t srcLevel = srcStartLevel + level;
                uint32_t dstLevel = dstStartLevel + level;
                MTLSize srcSize =
                    MTLSizeMake(src->width(srcLevel), src->height(srcLevel), src->depth(srcLevel));

                [get() copyFromTexture:src->get()
                           sourceSlice:srcSlice
                           sourceLevel:srcLevel
                          sourceOrigin:origin
                            sourceSize:srcSize
                             toTexture:dst->get()
                      destinationSlice:dstSlice
                      destinationLevel:dstLevel
                     destinationOrigin:origin];
            }
        }
    }

    return *this;
}

BlitCommandEncoder &BlitCommandEncoder::fillBuffer(const BufferRef &buffer,
                                                   NSRange range,
                                                   uint8_t value)
{
    if (!buffer)
    {
        return *this;
    }

    [get() fillBuffer:buffer->get() range:range value:value];
    return *this;
}

BlitCommandEncoder &BlitCommandEncoder::generateMipmapsForTexture(const TextureRef &texture)
{
    if (!texture)
    {
        return *this;
    }

    cmdBuffer().setWriteDependency(texture);
    [get() generateMipmapsForTexture:texture->get()];

    return *this;
}
BlitCommandEncoder &BlitCommandEncoder::synchronizeResource(const BufferRef &buffer)
{
    if (!buffer)
    {
        return *this;
    }

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    // Only MacOS has separated storage for resource on CPU and GPU and needs explicit
    // synchronization
    cmdBuffer().setReadDependency(buffer);
    [get() synchronizeResource:buffer->get()];
#endif
    return *this;
}
BlitCommandEncoder &BlitCommandEncoder::synchronizeResource(const TextureRef &texture)
{
    if (!texture)
    {
        return *this;
    }

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    // Only MacOS has separated storage for resource on CPU and GPU and needs explicit
    // synchronization
    cmdBuffer().setReadDependency(texture);
    if (texture->get().parentTexture)
    {
        [get() synchronizeResource:texture->get().parentTexture];
    }
    else
    {
        [get() synchronizeResource:texture->get()];
    }
#endif
    return *this;
}

// ComputeCommandEncoder implementation
ComputeCommandEncoder::ComputeCommandEncoder(CommandBuffer *cmdBuffer)
    : CommandEncoder(cmdBuffer, COMPUTE)
{}
ComputeCommandEncoder::~ComputeCommandEncoder() {}

ComputeCommandEncoder &ComputeCommandEncoder::restart()
{
    ANGLE_MTL_OBJC_SCOPE
    {
        if (valid())
        {
            // no change, skip
            return *this;
        }

        if (!cmdBuffer().valid())
        {
            reset();
            return *this;
        }

        // Create objective C object
        set([cmdBuffer().get() computeCommandEncoder]);

        // Verify that it was created successfully
        ASSERT(get());

        return *this;
    }
}

ComputeCommandEncoder &ComputeCommandEncoder::setComputePipelineState(
    id<MTLComputePipelineState> state)
{
    [get() setComputePipelineState:state];
    return *this;
}

ComputeCommandEncoder &ComputeCommandEncoder::setBuffer(const BufferRef &buffer,
                                                        uint32_t offset,
                                                        uint32_t index)
{
    if (index >= kMaxShaderBuffers)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(buffer);

    [get() setBuffer:(buffer ? buffer->get() : nil) offset:offset atIndex:index];

    return *this;
}

ComputeCommandEncoder &ComputeCommandEncoder::setBufferForWrite(const BufferRef &buffer,
                                                                uint32_t offset,
                                                                uint32_t index)
{
    if (index >= kMaxShaderBuffers)
    {
        return *this;
    }

    cmdBuffer().setWriteDependency(buffer);
    return setBuffer(buffer, offset, index);
}

ComputeCommandEncoder &ComputeCommandEncoder::setBytes(const uint8_t *bytes,
                                                       size_t size,
                                                       uint32_t index)
{
    if (index >= kMaxShaderBuffers)
    {
        return *this;
    }

    [get() setBytes:bytes length:size atIndex:index];

    return *this;
}

ComputeCommandEncoder &ComputeCommandEncoder::setSamplerState(id<MTLSamplerState> state,
                                                              float lodMinClamp,
                                                              float lodMaxClamp,
                                                              uint32_t index)
{
    if (index >= kMaxShaderSamplers)
    {
        return *this;
    }

    [get() setSamplerState:state lodMinClamp:lodMinClamp lodMaxClamp:lodMaxClamp atIndex:index];

    return *this;
}
ComputeCommandEncoder &ComputeCommandEncoder::setTexture(const TextureRef &texture, uint32_t index)
{
    if (index >= kMaxShaderSamplers)
    {
        return *this;
    }

    cmdBuffer().setReadDependency(texture);
    [get() setTexture:(texture ? texture->get() : nil) atIndex:index];

    return *this;
}
ComputeCommandEncoder &ComputeCommandEncoder::setTextureForWrite(const TextureRef &texture,
                                                                 uint32_t index)
{
    if (index >= kMaxShaderSamplers)
    {
        return *this;
    }

    cmdBuffer().setWriteDependency(texture);
    return setTexture(texture, index);
}

ComputeCommandEncoder &ComputeCommandEncoder::dispatch(const MTLSize &threadGroupsPerGrid,
                                                       const MTLSize &threadsPerGroup)
{
    [get() dispatchThreadgroups:threadGroupsPerGrid threadsPerThreadgroup:threadsPerGroup];
    return *this;
}

ComputeCommandEncoder &ComputeCommandEncoder::dispatchNonUniform(const MTLSize &threadsPerGrid,
                                                                 const MTLSize &threadsPerGroup)
{
#if TARGET_OS_TV
    UNREACHABLE();
#else
    [get() dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
#endif
    return *this;
}

}
}
