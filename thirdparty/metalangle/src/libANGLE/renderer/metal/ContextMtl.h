//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ContextMtl.h:
//    Defines the class interface for ContextMtl, implementing ContextImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_CONTEXTMTL_H_
#define LIBANGLE_RENDERER_METAL_CONTEXTMTL_H_

#import <Metal/Metal.h>

#include "common/Optional.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/ContextImpl.h"
#include "libANGLE/renderer/metal/mtl_buffer_pool.h"
#include "libANGLE/renderer/metal/mtl_command_buffer.h"
#include "libANGLE/renderer/metal/mtl_occlusion_query_pool.h"
#include "libANGLE/renderer/metal/mtl_resources.h"
#include "libANGLE/renderer/metal/mtl_state_cache.h"
#include "libANGLE/renderer/metal/mtl_utils.h"

namespace rx
{
class DisplayMtl;
class FramebufferMtl;
class VertexArrayMtl;
class ProgramMtl;
class RenderTargetMtl;
class WindowSurfaceMtl;
class TransformFeedbackMtl;

class ContextMtl : public ContextImpl, public mtl::Context
{
  public:
    ContextMtl(const gl::State &state, gl::ErrorSet *errorSet, DisplayMtl *display);
    ~ContextMtl() override;

    angle::Result initialize() override;

    void onDestroy(const gl::Context *context) override;

    // Flush and finish.
    angle::Result flush(const gl::Context *context) override;
    angle::Result finish(const gl::Context *context) override;

    // Drawing methods.
    angle::Result drawArrays(const gl::Context *context,
                             gl::PrimitiveMode mode,
                             GLint first,
                             GLsizei count) override;
    angle::Result drawArraysInstanced(const gl::Context *context,
                                      gl::PrimitiveMode mode,
                                      GLint first,
                                      GLsizei count,
                                      GLsizei instanceCount) override;
    angle::Result drawArraysInstancedBaseInstance(const gl::Context *context,
                                                  gl::PrimitiveMode mode,
                                                  GLint first,
                                                  GLsizei count,
                                                  GLsizei instanceCount,
                                                  GLuint baseInstance) override;

    angle::Result drawElements(const gl::Context *context,
                               gl::PrimitiveMode mode,
                               GLsizei count,
                               gl::DrawElementsType type,
                               const void *indices) override;
    angle::Result drawElementsInstanced(const gl::Context *context,
                                        gl::PrimitiveMode mode,
                                        GLsizei count,
                                        gl::DrawElementsType type,
                                        const void *indices,
                                        GLsizei instanceCount) override;
    angle::Result drawElementsInstancedBaseVertexBaseInstance(const gl::Context *context,
                                                              gl::PrimitiveMode mode,
                                                              GLsizei count,
                                                              gl::DrawElementsType type,
                                                              const void *indices,
                                                              GLsizei instances,
                                                              GLint baseVertex,
                                                              GLuint baseInstance) override;
    angle::Result drawRangeElements(const gl::Context *context,
                                    gl::PrimitiveMode mode,
                                    GLuint start,
                                    GLuint end,
                                    GLsizei count,
                                    gl::DrawElementsType type,
                                    const void *indices) override;
    angle::Result drawArraysIndirect(const gl::Context *context,
                                     gl::PrimitiveMode mode,
                                     const void *indirect) override;
    angle::Result drawElementsIndirect(const gl::Context *context,
                                       gl::PrimitiveMode mode,
                                       gl::DrawElementsType type,
                                       const void *indirect) override;

    // Device loss
    gl::GraphicsResetStatus getResetStatus() override;

    // Vendor and description strings.
    std::string getVendorString() const override;
    std::string getRendererDescription() const override;

    // EXT_debug_marker
    void insertEventMarker(GLsizei length, const char *marker) override;
    void pushGroupMarker(GLsizei length, const char *marker) override;
    void popGroupMarker() override;

    // KHR_debug
    void pushDebugGroup(GLenum source, GLuint id, const std::string &message) override;
    void popDebugGroup() override;

    // State sync with dirty bits.
    angle::Result syncState(const gl::Context *context,
                            const gl::State::DirtyBits &dirtyBits,
                            const gl::State::DirtyBits &bitMask) override;

    // Disjoint timer queries
    GLint getGPUDisjoint() override;
    GLint64 getTimestamp() override;

    // Context switching
    angle::Result onMakeCurrent(const gl::Context *context) override;
    angle::Result onUnMakeCurrent(const gl::Context *context) override;

    // Native capabilities, unmodified by gl::Context.
    gl::Caps getNativeCaps() const override;
    const gl::TextureCapsMap &getNativeTextureCaps() const override;
    const gl::Extensions &getNativeExtensions() const override;
    const gl::Limitations &getNativeLimitations() const override;

    // Shader creation
    CompilerImpl *createCompiler() override;
    ShaderImpl *createShader(const gl::ShaderState &state) override;
    ProgramImpl *createProgram(const gl::ProgramState &state) override;

    // Framebuffer creation
    FramebufferImpl *createFramebuffer(const gl::FramebufferState &state) override;

    // Texture creation
    TextureImpl *createTexture(const gl::TextureState &state) override;

    // Renderbuffer creation
    RenderbufferImpl *createRenderbuffer(const gl::RenderbufferState &state) override;

    // Buffer creation
    BufferImpl *createBuffer(const gl::BufferState &state) override;

    // Vertex Array creation
    VertexArrayImpl *createVertexArray(const gl::VertexArrayState &state) override;

    // Query and Fence creation
    QueryImpl *createQuery(gl::QueryType type) override;
    FenceNVImpl *createFenceNV() override;
    SyncImpl *createSync() override;

    // Transform Feedback creation
    TransformFeedbackImpl *createTransformFeedback(
        const gl::TransformFeedbackState &state) override;

    // Sampler object creation
    SamplerImpl *createSampler(const gl::SamplerState &state) override;

    // Program Pipeline object creation
    ProgramPipelineImpl *createProgramPipeline(const gl::ProgramPipelineState &data) override;

    // Path object creation
    std::vector<PathImpl *> createPaths(GLsizei) override;

    // Memory object creation.
    MemoryObjectImpl *createMemoryObject() override;

    // Semaphore creation.
    SemaphoreImpl *createSemaphore() override;

    // Overlay creation.
    OverlayImpl *createOverlay(const gl::OverlayState &state) override;

    angle::Result dispatchCompute(const gl::Context *context,
                                  GLuint numGroupsX,
                                  GLuint numGroupsY,
                                  GLuint numGroupsZ) override;
    angle::Result dispatchComputeIndirect(const gl::Context *context, GLintptr indirect) override;

    angle::Result memoryBarrier(const gl::Context *context, GLbitfield barriers) override;
    angle::Result memoryBarrierByRegion(const gl::Context *context, GLbitfield barriers) override;

    // override mtl::ErrorHandler
    void handleError(GLenum error,
                     const char *file,
                     const char *function,
                     unsigned int line) override;
    void handleError(NSError *_Nullable error,
                     const char *file,
                     const char *function,
                     unsigned int line) override;

    using ContextImpl::handleError;

    void invalidateState(const gl::Context *context);
    void invalidateDefaultAttribute(size_t attribIndex);
    void invalidateDefaultAttributes(const gl::AttributesMask &dirtyMask);
    void invalidateCurrentTextures();
    void invalidateDriverUniforms();
    void invalidateRenderPipeline();

    // Call this to notify ContextMtl whenever FramebufferMtl's state changed
    void onDrawFrameBufferChangedState(const gl::Context *context,
                                       FramebufferMtl *framebuffer,
                                       bool renderPassChanged);
    void onBackbufferResized(const gl::Context *context, WindowSurfaceMtl *backbuffer);

    // Invoke by QueryMtl
    angle::Result onOcclusionQueryBegan(const gl::Context *context, QueryMtl *query);
    void onOcclusionQueryEnded(const gl::Context *context, QueryMtl *query);
    void onOcclusionQueryDestroyed(const gl::Context *context, QueryMtl *query);

    // Useful for temporarily pause then restart occlusion query during clear/blit with draw.
    bool hasActiveOcclusionQuery() const { return mOcclusionQuery; }
    // Disable the occlusion query in the current render pass.
    // The render pass must already started.
    void disableActiveOcclusionQueryInRenderPass();
    // Re-enable the occlusion query in the current render pass.
    // The render pass must already started.
    // NOTE: the old query's result will be retained and combined with the new result.
    angle::Result restartActiveOcclusionQueryInRenderPass();

    // Invoke by mtl::Sync
    void queueEventSignal(const mtl::SharedEventRef &event, uint64_t value);
    void serverWaitEvent(const mtl::SharedEventRef &event, uint64_t value);

    // Invoke by TransformFeedbackMtl
    void onTransformFeedbackActive(const gl::Context *context, TransformFeedbackMtl *xfb);
    void onTransformFeedbackInactive(const gl::Context *context, TransformFeedbackMtl *xfb);

    const mtl::ClearColorValue &getClearColorValue() const;
    MTLColorWriteMask getColorMask() const;
    float getClearDepthValue() const;
    uint32_t getClearStencilValue() const;
    // Return front facing stencil write mask
    uint32_t getStencilMask() const;
    bool getDepthMask() const;

    const mtl::Format &getPixelFormat(angle::FormatID angleFormatId) const;
    const mtl::FormatCaps &getNativeFormatCaps(MTLPixelFormat mtlFormat) const;
    // See mtl::FormatTable::getVertexFormat()
    const mtl::VertexFormat &getVertexFormat(angle::FormatID angleFormatId,
                                             bool tightlyPacked) const;

    angle::Result getNullTexture(const gl::Context *context,
                                 gl::TextureType type,
                                 gl::Texture **textureOut);

    mtl::BufferPool &getClientIndexBufferPool() { return mClientIndexBufferPool; }

    // Recommended to call these methods to end encoding instead of invoking the encoder's
    // endEncoding() directly.
    void endEncoding(mtl::RenderCommandEncoder *encoder);
    // Ends any active command encoder
    void endEncoding(bool forceSaveRenderPassContent);

    void flushCommandBufer();
    void present(const gl::Context *context, id<CAMetalDrawable> presentationDrawable);
    angle::Result finishCommandBuffer();

    // Check whether compatible render pass has been started.
    bool hasStartedRenderPass(const mtl::RenderPassDesc &desc);

    // Get current render encoder. May be nullptr if no render pass has been started.
    mtl::RenderCommandEncoder *getRenderCommandEncoder();

    // Will end current command encoder if it is valid, then start new encoder.
    // Unless hasStartedRenderPass(desc) returns true.
    mtl::RenderCommandEncoder *getRenderCommandEncoder(const mtl::RenderPassDesc &desc);

    // Utilities to quickly create render command encoder to a specific texture:
    mtl::RenderCommandEncoder *getRenderCommandEncoder(const mtl::TextureRef &textureTarget,
                                                       const gl::ImageIndex &index);
    // The previous content of texture will be loaded if clearColor is not provided
    mtl::RenderCommandEncoder *getRenderCommandEncoder(const RenderTargetMtl &renderTarget,
                                                       const Optional<MTLClearColor> &clearColor);
    // The previous content of texture will be loaded
    mtl::RenderCommandEncoder *getRenderCommandEncoder(const RenderTargetMtl &renderTarget);

    // Will end current command encoder and start new blit command encoder. Unless a blit comamnd
    // encoder is already started.
    mtl::BlitCommandEncoder *getBlitCommandEncoder();

    // Will end current command encoder and start new compute command encoder. Unless a compute
    // command encoder is already started.
    mtl::ComputeCommandEncoder *getComputeCommandEncoder();

  private:
    void ensureCommandBufferValid();
    void releaseInFlightBuffers();
    angle::Result ensureIncompleteTexturesCreated(const gl::Context *context);
    angle::Result setupDraw(const gl::Context *context,
                            gl::PrimitiveMode mode,
                            GLint firstVertex,
                            GLsizei vertexOrIndexCount,
                            GLsizei instanceCount,
                            gl::DrawElementsType indexTypeOrNone,
                            const void *indices,
                            bool xfbPass);

    angle::Result drawTriFanArrays(const gl::Context *context,
                                   GLint first,
                                   GLsizei count,
                                   GLsizei instances);
    angle::Result drawTriFanArraysWithBaseVertex(const gl::Context *context,
                                                 GLint first,
                                                 GLsizei count,
                                                 GLsizei instances);
    angle::Result drawTriFanArraysLegacy(const gl::Context *context,
                                         GLint first,
                                         GLsizei count,
                                         GLsizei instances);
    angle::Result drawTriFanElements(const gl::Context *context,
                                     GLsizei count,
                                     gl::DrawElementsType type,
                                     const void *indices,
                                     GLsizei instances);

    angle::Result drawLineLoopArraysNonInstanced(const gl::Context *context,
                                                 GLint first,
                                                 GLsizei count);
    angle::Result drawLineLoopArrays(const gl::Context *context,
                                     GLint first,
                                     GLsizei count,
                                     GLsizei instances);
    angle::Result drawLineLoopElementsNonInstancedNoPrimitiveRestart(const gl::Context *context,
                                                                     GLsizei count,
                                                                     gl::DrawElementsType type,
                                                                     const void *indices);
    angle::Result drawLineLoopElements(const gl::Context *context,
                                       GLsizei count,
                                       gl::DrawElementsType type,
                                       const void *indices,
                                       GLsizei instances);

    angle::Result drawArraysImpl(const gl::Context *context,
                                 gl::PrimitiveMode mode,
                                 GLint first,
                                 GLsizei count,
                                 GLsizei instanceCount);

    angle::Result drawElementsImpl(const gl::Context *context,
                                   gl::PrimitiveMode mode,
                                   GLsizei count,
                                   gl::DrawElementsType type,
                                   const void *indices,
                                   GLsizei instanceCount);

    angle::Result execDrawInstanced(MTLPrimitiveType primitiveType,
                                    uint32_t vertexStart,
                                    uint32_t vertexCount,
                                    uint32_t instances);

    void execDrawIndexedInstanced(MTLPrimitiveType primitiveType,
                                  uint32_t indexCount,
                                  MTLIndexType indexType,
                                  const mtl::BufferRef &indexBuffer,
                                  size_t bufferOffset,
                                  uint32_t instances);

    void updateExtendedState(const gl::State &glState);

    void updateViewport(FramebufferMtl *framebufferMtl,
                        const gl::Rectangle &viewport,
                        float nearPlane,
                        float farPlane);
    void updateDepthRange(float nearPlane, float farPlane);
    void updateScissor(const gl::State &glState);
    void updateCullMode(const gl::State &glState);
    void updateFrontFace(const gl::State &glState);
    void updateDepthBias(const gl::State &glState);
    void updateDrawFrameBufferBinding(const gl::Context *context);
    void updateProgramExecutable(const gl::Context *context);
    void updateVertexArray(const gl::Context *context);
    void updatePrimitiRestart(const gl::State &glState);

    angle::Result updateDefaultAttribute(size_t attribIndex);
    void filterOutXFBOnlyDirtyBits(const gl::Context *context);
    angle::Result handleDirtyActiveTextures(const gl::Context *context);
    angle::Result handleDirtyDefaultAttribs(const gl::Context *context);
    angle::Result handleDirtyDriverUniforms(const gl::Context *context,
                                            GLint drawCallFirstVertex,
                                            uint32_t verticesPerInstance);
    angle::Result fillDriverXFBUniforms(GLint drawCallFirstVertex,
                                        uint32_t verticesPerInstance,
                                        uint32_t skippedInstances);
    angle::Result handleDirtyDepthStencilState(const gl::Context *context);
    angle::Result handleDirtyDepthBias(const gl::Context *context);
    angle::Result handleDirtyRenderPass(const gl::Context *context);
    angle::Result checkIfPipelineChanged(const gl::Context *context,
                                         gl::PrimitiveMode primitiveMode,
                                         bool xfbPass,
                                         bool *pipelineDescChanged);

    angle::Result startOcclusionQueryInRenderPass(QueryMtl *query, bool clearOldValue);
    // ensure that occlusion query pool can allocate new offset.
    angle::Result ensureOcclusionQueryPoolCapacity();

    // Dirty bits.
    enum DirtyBitType : size_t
    {
        DIRTY_BIT_DEFAULT_ATTRIBS,
        DIRTY_BIT_TEXTURES,
        DIRTY_BIT_DRIVER_UNIFORMS,
        DIRTY_BIT_DEPTH_STENCIL_DESC,
        DIRTY_BIT_DEPTH_BIAS,
        DIRTY_BIT_STENCIL_REF,
        DIRTY_BIT_BLEND_COLOR,
        DIRTY_BIT_VIEWPORT,
        DIRTY_BIT_SCISSOR,
        DIRTY_BIT_DRAW_FRAMEBUFFER,
        DIRTY_BIT_CULL_MODE,
        DIRTY_BIT_WINDING,
        DIRTY_BIT_RENDER_PIPELINE,
        DIRTY_BIT_UNIFORM_BUFFERS_BINDING,
        DIRTY_BIT_RASTERIZER_DISCARD,
        DIRTY_BIT_MAX,
    };

    // See compiler/translator/TranslatorVulkan.cpp: AddDriverUniformsToShader()
    struct DriverUniforms
    {
        float viewport[4];

        float halfRenderAreaHeight;
        float viewportYScale;
        float negViewportYScale;

        // 32 bits for 32 clip distances
        uint32_t enabledClipDistances;

        uint32_t xfbActiveUnpaused;
        uint32_t xfbVerticesPerDraw;
        // NOTE: Explicit padding. Fill in with useful data when needed in the future.
        int32_t padding[2];

        int32_t xfbBufferOffsets[4];
        uint32_t acbBufferOffsets[4];

        // We'll use x, y, z, w for near / far / diff / zscale respectively.
        float depthRange[4];

        uint32_t coverageMask;

        int32_t emulatedInstanceID;

        float padding2[2];

        // Adjusted depth range used for depth range mapping emulation.
        // x, y, z is near / far / diff
        float adjustedDepthRange[4];
    };

    struct DefaultAttribute
    {
        float values[4];
    };

    mtl::OcclusionQueryPool mOcclusionQueryPool;

    mtl::CommandBuffer mCmdBuffer;
    mtl::RenderCommandEncoder mRenderEncoder;
    mtl::BlitCommandEncoder mBlitEncoder;
    mtl::ComputeCommandEncoder mComputeEncoder;

    // Cached back-end objects
    FramebufferMtl *mDrawFramebuffer = nullptr;
    VertexArrayMtl *mVertexArray     = nullptr;
    ProgramMtl *mProgram             = nullptr;
    QueryMtl *mOcclusionQuery        = nullptr;

    using DirtyBits = angle::BitSet<DIRTY_BIT_MAX>;

    gl::AttributesMask mDirtyDefaultAttribsMask;
    DirtyBits mDirtyBits;

    // State
    mtl::RenderPipelineDesc mRenderPipelineDesc;
    mtl::DepthStencilDesc mDepthStencilDesc;
    mtl::BlendDesc mBlendDesc;
    mtl::ClearColorValue mClearColor;
    uint32_t mClearStencil    = 0;
    uint32_t mStencilRefFront = 0;
    uint32_t mStencilRefBack  = 0;
    MTLViewport mViewport;
    MTLScissorRect mScissorRect;
    MTLWinding mWinding;
    MTLCullMode mCullMode;
    bool mCullAllPolygons = false;

    mtl::BufferPool mClientIndexBufferPool;

    // Lineloop and TriFan index buffer
    mtl::BufferPool mLineLoopIndexBufferPool;
    mtl::BufferPool mLineLoopLastSegmentIndexBufferPool;
    mtl::BufferPool mTriFanIndexBufferPool;
    mtl::BufferPool mTriFanClientIndexBufferPool;

    // LRU cache to store generate index buffer from client data. Note: if primitive restart is
    // changed, this cache must be invalidated
    mtl::ClientIndexBufferCache mTriFanClientIndexBufferCache;

    // one buffer can be reused for multiple DrawArrays()
    mtl::BufferRef mTriFanArraysIndexBuffer;
    GLint mTriFanArraysIndexBufferFirstVertex = 0;
    uint32_t mTriFanArraysIndexBufferOffset   = 0;

    // Dummy texture to be used for transform feedback only pass.
    mtl::TextureRef mDummyXFBRenderTexture;

    DriverUniforms mDriverUniforms;

    DefaultAttribute mDefaultAttributes[mtl::kMaxVertexAttribs];

    IncompleteTextureSet mIncompleteTextures;
    bool mIncompleteTexturesInitialized = false;
};

}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_CONTEXTMTL_H_ */
