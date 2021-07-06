//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ContextGL:
//   OpenGL-specific functionality associated with a GL Context.
//

#ifndef LIBANGLE_RENDERER_GL_CONTEXTGL_H_
#define LIBANGLE_RENDERER_GL_CONTEXTGL_H_

#include "libANGLE/renderer/ContextImpl.h"
#include "libANGLE/renderer/gl/RendererGL.h"

namespace angle
{
struct FeaturesGL;
}  // namespace angle

namespace sh
{
struct BlockMemberInfo;
}  // namespace sh

namespace rx
{
class BlitGL;
class ClearMultiviewGL;
class FunctionsGL;
class RendererGL;
class StateManagerGL;

class ContextGL : public ContextImpl
{
  public:
    ContextGL(const gl::State &state,
              gl::ErrorSet *errorSet,
              const std::shared_ptr<RendererGL> &renderer);
    ~ContextGL() override;

    angle::Result initialize() override;

    // Shader creation
    CompilerImpl *createCompiler() override;
    ShaderImpl *createShader(const gl::ShaderState &data) override;
    ProgramImpl *createProgram(const gl::ProgramState &data) override;

    // Framebuffer creation
    FramebufferImpl *createFramebuffer(const gl::FramebufferState &data) override;

    // Texture creation
    TextureImpl *createTexture(const gl::TextureState &state) override;

    // Renderbuffer creation
    RenderbufferImpl *createRenderbuffer(const gl::RenderbufferState &state) override;

    // Buffer creation
    BufferImpl *createBuffer(const gl::BufferState &state) override;

    // Vertex Array creation
    VertexArrayImpl *createVertexArray(const gl::VertexArrayState &data) override;

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
    std::vector<PathImpl *> createPaths(GLsizei range) override;

    // Memory object creation.
    MemoryObjectImpl *createMemoryObject() override;

    // Semaphore creation.
    SemaphoreImpl *createSemaphore() override;

    // Overlay creation.
    OverlayImpl *createOverlay(const gl::OverlayState &state) override;

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
                                        GLsizei instances) override;
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

    // CHROMIUM_path_rendering implementation
    void stencilFillPath(const gl::Path *path, GLenum fillMode, GLuint mask) override;
    void stencilStrokePath(const gl::Path *path, GLint reference, GLuint mask) override;
    void coverFillPath(const gl::Path *path, GLenum coverMode) override;
    void coverStrokePath(const gl::Path *path, GLenum coverMode) override;
    void stencilThenCoverFillPath(const gl::Path *path,
                                  GLenum fillMode,
                                  GLuint mask,
                                  GLenum coverMode) override;
    void stencilThenCoverStrokePath(const gl::Path *path,
                                    GLint reference,
                                    GLuint mask,
                                    GLenum coverMode) override;
    void coverFillPathInstanced(const std::vector<gl::Path *> &paths,
                                GLenum coverMode,
                                GLenum transformType,
                                const GLfloat *transformValues) override;
    void coverStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                  GLenum coverMode,
                                  GLenum transformType,
                                  const GLfloat *transformValues) override;
    void stencilFillPathInstanced(const std::vector<gl::Path *> &paths,
                                  GLenum fillMode,
                                  GLuint mask,
                                  GLenum transformType,
                                  const GLfloat *transformValues) override;
    void stencilStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                    GLint reference,
                                    GLuint mask,
                                    GLenum transformType,
                                    const GLfloat *transformValues) override;
    void stencilThenCoverFillPathInstanced(const std::vector<gl::Path *> &paths,
                                           GLenum coverMode,
                                           GLenum fillMode,
                                           GLuint mask,
                                           GLenum transformType,
                                           const GLfloat *transformValues) override;
    void stencilThenCoverStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                             GLenum coverMode,
                                             GLint reference,
                                             GLuint mask,
                                             GLenum transformType,
                                             const GLfloat *transformValues) override;

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

    // Caps queries
    gl::Caps getNativeCaps() const override;
    const gl::TextureCapsMap &getNativeTextureCaps() const override;
    const gl::Extensions &getNativeExtensions() const override;
    const gl::Limitations &getNativeLimitations() const override;

    // Handle helpers
    ANGLE_INLINE const FunctionsGL *getFunctions() const { return mRenderer->getFunctions(); }

    StateManagerGL *getStateManager();
    const angle::FeaturesGL &getFeaturesGL() const;
    BlitGL *getBlitter() const;
    ClearMultiviewGL *getMultiviewClearer() const;

    angle::Result dispatchCompute(const gl::Context *context,
                                  GLuint numGroupsX,
                                  GLuint numGroupsY,
                                  GLuint numGroupsZ) override;
    angle::Result dispatchComputeIndirect(const gl::Context *context, GLintptr indirect) override;

    angle::Result memoryBarrier(const gl::Context *context, GLbitfield barriers) override;
    angle::Result memoryBarrierByRegion(const gl::Context *context, GLbitfield barriers) override;

    void setMaxShaderCompilerThreads(GLuint count) override;

    void invalidateTexture(gl::TextureType target) override;

    void validateState() const;

  private:
    angle::Result setDrawArraysState(const gl::Context *context,
                                     GLint first,
                                     GLsizei count,
                                     GLsizei instanceCount);

    angle::Result setDrawElementsState(const gl::Context *context,
                                       GLsizei count,
                                       gl::DrawElementsType type,
                                       const void *indices,
                                       GLsizei instanceCount,
                                       const void **outIndices);

    gl::AttributesMask updateAttributesForBaseInstance(const gl::Program *program,
                                                       GLuint baseInstance);
    void resetUpdatedAttributes(gl::AttributesMask attribMask);

  protected:
    std::shared_ptr<RendererGL> mRenderer;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_CONTEXTGL_H_
