//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ContextGL:
//   OpenGL-specific functionality associated with a GL Context.
//

#include "libANGLE/renderer/gl/ContextGL.h"

#include "libANGLE/Context.h"
#include "libANGLE/renderer/OverlayImpl.h"
#include "libANGLE/renderer/gl/BufferGL.h"
#include "libANGLE/renderer/gl/CompilerGL.h"
#include "libANGLE/renderer/gl/FenceNVGL.h"
#include "libANGLE/renderer/gl/FramebufferGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/MemoryObjectGL.h"
#include "libANGLE/renderer/gl/PathGL.h"
#include "libANGLE/renderer/gl/ProgramGL.h"
#include "libANGLE/renderer/gl/ProgramPipelineGL.h"
#include "libANGLE/renderer/gl/QueryGL.h"
#include "libANGLE/renderer/gl/RenderbufferGL.h"
#include "libANGLE/renderer/gl/RendererGL.h"
#include "libANGLE/renderer/gl/SamplerGL.h"
#include "libANGLE/renderer/gl/SemaphoreGL.h"
#include "libANGLE/renderer/gl/ShaderGL.h"
#include "libANGLE/renderer/gl/StateManagerGL.h"
#include "libANGLE/renderer/gl/SyncGL.h"
#include "libANGLE/renderer/gl/TextureGL.h"
#include "libANGLE/renderer/gl/TransformFeedbackGL.h"
#include "libANGLE/renderer/gl/VertexArrayGL.h"

namespace rx
{

ContextGL::ContextGL(const gl::State &state,
                     gl::ErrorSet *errorSet,
                     const std::shared_ptr<RendererGL> &renderer)
    : ContextImpl(state, errorSet), mRenderer(renderer)
{}

ContextGL::~ContextGL() {}

angle::Result ContextGL::initialize()
{
    return angle::Result::Continue;
}

CompilerImpl *ContextGL::createCompiler()
{
    return new CompilerGL(getFunctions());
}

ShaderImpl *ContextGL::createShader(const gl::ShaderState &data)
{
    const FunctionsGL *functions = getFunctions();
    GLuint shader                = functions->createShader(ToGLenum(data.getShaderType()));

    return new ShaderGL(data, shader, mRenderer->getMultiviewImplementationType(), mRenderer);
}

ProgramImpl *ContextGL::createProgram(const gl::ProgramState &data)
{
    return new ProgramGL(data, getFunctions(), getFeaturesGL(), getStateManager(),
                         getExtensions().pathRendering, mRenderer);
}

FramebufferImpl *ContextGL::createFramebuffer(const gl::FramebufferState &data)
{
    const FunctionsGL *funcs = getFunctions();

    GLuint fbo = 0;
    funcs->genFramebuffers(1, &fbo);

    return new FramebufferGL(data, fbo, false, false);
}

TextureImpl *ContextGL::createTexture(const gl::TextureState &state)
{
    const FunctionsGL *functions = getFunctions();
    StateManagerGL *stateManager = getStateManager();

    GLuint texture = 0;
    functions->genTextures(1, &texture);
    stateManager->bindTexture(state.getType(), texture);

    return new TextureGL(state, texture);
}

RenderbufferImpl *ContextGL::createRenderbuffer(const gl::RenderbufferState &state)
{
    const FunctionsGL *functions = getFunctions();
    StateManagerGL *stateManager = getStateManager();

    GLuint renderbuffer = 0;
    functions->genRenderbuffers(1, &renderbuffer);
    stateManager->bindRenderbuffer(GL_RENDERBUFFER, renderbuffer);

    return new RenderbufferGL(state, renderbuffer);
}

BufferImpl *ContextGL::createBuffer(const gl::BufferState &state)
{
    return new BufferGL(state, getFunctions(), getStateManager());
}

VertexArrayImpl *ContextGL::createVertexArray(const gl::VertexArrayState &data)
{
    return new VertexArrayGL(data, getFunctions(), getStateManager());
}

QueryImpl *ContextGL::createQuery(gl::QueryType type)
{
    switch (type)
    {
        case gl::QueryType::CommandsCompleted:
            return new SyncQueryGL(type, getFunctions());

        default:
            return new StandardQueryGL(type, getFunctions(), getStateManager());
    }
}

FenceNVImpl *ContextGL::createFenceNV()
{
    const FunctionsGL *functions = getFunctions();
    if (FenceNVGL::Supported(functions))
    {
        return new FenceNVGL(functions);
    }
    else
    {
        ASSERT(FenceNVSyncGL::Supported(functions));
        return new FenceNVSyncGL(functions);
    }
}

SyncImpl *ContextGL::createSync()
{
    return new SyncGL(getFunctions());
}

TransformFeedbackImpl *ContextGL::createTransformFeedback(const gl::TransformFeedbackState &state)
{
    return new TransformFeedbackGL(state, getFunctions(), getStateManager());
}

SamplerImpl *ContextGL::createSampler(const gl::SamplerState &state)
{
    return new SamplerGL(state, getFunctions(), getStateManager());
}

ProgramPipelineImpl *ContextGL::createProgramPipeline(const gl::ProgramPipelineState &data)
{
    return new ProgramPipelineGL(data, getFunctions());
}

std::vector<PathImpl *> ContextGL::createPaths(GLsizei range)
{
    const FunctionsGL *funcs = getFunctions();

    std::vector<PathImpl *> ret;
    ret.reserve(range);

    const GLuint first = funcs->genPathsNV(range);
    if (first == 0)
        return ret;

    for (GLsizei i = 0; i < range; ++i)
    {
        const auto id = first + i;
        ret.push_back(new PathGL(funcs, id));
    }

    return ret;
}

MemoryObjectImpl *ContextGL::createMemoryObject()
{
    const FunctionsGL *functions = getFunctions();

    GLuint memoryObject = 0;
    functions->createMemoryObjectsEXT(1, &memoryObject);

    return new MemoryObjectGL(memoryObject);
}

SemaphoreImpl *ContextGL::createSemaphore()
{
    const FunctionsGL *functions = getFunctions();

    GLuint semaphore = 0;
    functions->genSemaphoresEXT(1, &semaphore);

    return new SemaphoreGL(semaphore);
}

OverlayImpl *ContextGL::createOverlay(const gl::OverlayState &state)
{
    // Not implemented.
    return new OverlayImpl(state);
}

angle::Result ContextGL::flush(const gl::Context *context)
{
    return mRenderer->flush();
}

angle::Result ContextGL::finish(const gl::Context *context)
{
    return mRenderer->finish();
}

ANGLE_INLINE angle::Result ContextGL::setDrawArraysState(const gl::Context *context,
                                                         GLint first,
                                                         GLsizei count,
                                                         GLsizei instanceCount)
{
    if (context->getStateCache().hasAnyActiveClientAttrib())
    {
        const gl::State &glState   = context->getState();
        const gl::Program *program = glState.getProgram();
        const gl::VertexArray *vao = glState.getVertexArray();
        const VertexArrayGL *vaoGL = GetImplAs<VertexArrayGL>(vao);

        ANGLE_TRY(vaoGL->syncClientSideData(context, program->getActiveAttribLocationsMask(), first,
                                            count, instanceCount));

#if defined(ANGLE_STATE_VALIDATION_ENABLED)
        vaoGL->validateState();
#endif  // ANGLE_STATE_VALIDATION_ENABLED
    }

    return angle::Result::Continue;
}

ANGLE_INLINE angle::Result ContextGL::setDrawElementsState(const gl::Context *context,
                                                           GLsizei count,
                                                           gl::DrawElementsType type,
                                                           const void *indices,
                                                           GLsizei instanceCount,
                                                           const void **outIndices)
{
    const gl::State &glState = context->getState();

    const gl::Program *program = glState.getProgram();

    const gl::VertexArray *vao = glState.getVertexArray();

    const gl::StateCache &stateCache = context->getStateCache();
    if (stateCache.hasAnyActiveClientAttrib() || vao->getElementArrayBuffer() == nullptr)
    {
        const VertexArrayGL *vaoGL = GetImplAs<VertexArrayGL>(vao);
        ANGLE_TRY(vaoGL->syncDrawElementsState(context, program->getActiveAttribLocationsMask(),
                                               count, type, indices, instanceCount,
                                               glState.isPrimitiveRestartEnabled(), outIndices));
    }
    else
    {
        *outIndices = indices;
    }

#if defined(ANGLE_STATE_VALIDATION_ENABLED)
    const VertexArrayGL *vaoGL = GetImplAs<VertexArrayGL>(vao);
    vaoGL->validateState();
#endif  // ANGLE_STATE_VALIDATION_ENABLED

    return angle::Result::Continue;
}

angle::Result ContextGL::drawArrays(const gl::Context *context,
                                    gl::PrimitiveMode mode,
                                    GLint first,
                                    GLsizei count)
{
    const gl::Program *program  = context->getState().getProgram();
    const bool usesMultiview    = program->usesMultiview();
    const GLsizei instanceCount = usesMultiview ? program->getNumViews() : 0;

#if defined(ANGLE_STATE_VALIDATION_ENABLED)
    validateState();
#endif

    ANGLE_TRY(setDrawArraysState(context, first, count, instanceCount));
    if (!usesMultiview)
    {
        getFunctions()->drawArrays(ToGLenum(mode), first, count);
    }
    else
    {
        getFunctions()->drawArraysInstanced(ToGLenum(mode), first, count, instanceCount);
    }
    return angle::Result::Continue;
}

angle::Result ContextGL::drawArraysInstanced(const gl::Context *context,
                                             gl::PrimitiveMode mode,
                                             GLint first,
                                             GLsizei count,
                                             GLsizei instanceCount)
{
    GLsizei adjustedInstanceCount = instanceCount;
    const gl::Program *program    = context->getState().getProgram();
    if (program->usesMultiview())
    {
        adjustedInstanceCount *= program->getNumViews();
    }

    ANGLE_TRY(setDrawArraysState(context, first, count, adjustedInstanceCount));
    getFunctions()->drawArraysInstanced(ToGLenum(mode), first, count, adjustedInstanceCount);
    return angle::Result::Continue;
}

gl::AttributesMask ContextGL::updateAttributesForBaseInstance(const gl::Program *program,
                                                              GLuint baseInstance)
{
    gl::AttributesMask attribToUpdateMask;

    if (baseInstance != 0)
    {
        const FunctionsGL *functions = getFunctions();
        const auto &attribs          = mState.getVertexArray()->getVertexAttributes();
        const auto &bindings         = mState.getVertexArray()->getVertexBindings();
        for (GLuint attribIndex = 0; attribIndex < gl::MAX_VERTEX_ATTRIBS; attribIndex++)
        {
            const gl::VertexAttribute &attrib = attribs[attribIndex];
            const gl::VertexBinding &binding  = bindings[attrib.bindingIndex];
            if (program->isAttribLocationActive(attribIndex) && binding.getDivisor() != 0)
            {
                attribToUpdateMask.set(attribIndex);
                const char *p             = static_cast<const char *>(attrib.pointer);
                const size_t sourceStride = gl::ComputeVertexAttributeStride(attrib, binding);
                const void *newPointer    = p + sourceStride * baseInstance;

                const BufferGL *buffer = GetImplAs<BufferGL>(binding.getBuffer().get());
                // We often stream data from scratch buffers when client side data is being used
                // and that information is in VertexArrayGL.
                // Assert that the buffer is non-null because this case isn't handled.
                ASSERT(buffer);
                getStateManager()->bindBuffer(gl::BufferBinding::Array, buffer->getBufferID());
                if (attrib.format->isPureInt())
                {
                    functions->vertexAttribIPointer(attribIndex, attrib.format->channelCount,
                                                    gl::ToGLenum(attrib.format->vertexAttribType),
                                                    attrib.vertexAttribArrayStride, newPointer);
                }
                else
                {
                    functions->vertexAttribPointer(attribIndex, attrib.format->channelCount,
                                                   gl::ToGLenum(attrib.format->vertexAttribType),
                                                   attrib.format->isNorm(),
                                                   attrib.vertexAttribArrayStride, newPointer);
                }
            }
        }
    }

    return attribToUpdateMask;
}

void ContextGL::resetUpdatedAttributes(gl::AttributesMask attribMask)
{
    const FunctionsGL *functions = getFunctions();
    for (size_t attribIndex : attribMask)
    {
        const gl::VertexAttribute &attrib =
            mState.getVertexArray()->getVertexAttributes()[attribIndex];
        const gl::VertexBinding &binding =
            (mState.getVertexArray()->getVertexBindings())[attrib.bindingIndex];
        getStateManager()->bindBuffer(
            gl::BufferBinding::Array,
            GetImplAs<BufferGL>(binding.getBuffer().get())->getBufferID());
        if (attrib.format->isPureInt())
        {
            functions->vertexAttribIPointer(static_cast<GLuint>(attribIndex),
                                            attrib.format->channelCount,
                                            gl::ToGLenum(attrib.format->vertexAttribType),
                                            attrib.vertexAttribArrayStride, attrib.pointer);
        }
        else
        {
            functions->vertexAttribPointer(
                static_cast<GLuint>(attribIndex), attrib.format->channelCount,
                gl::ToGLenum(attrib.format->vertexAttribType), attrib.format->isNorm(),
                attrib.vertexAttribArrayStride, attrib.pointer);
        }
    }
}

angle::Result ContextGL::drawArraysInstancedBaseInstance(const gl::Context *context,
                                                         gl::PrimitiveMode mode,
                                                         GLint first,
                                                         GLsizei count,
                                                         GLsizei instanceCount,
                                                         GLuint baseInstance)
{
    GLsizei adjustedInstanceCount = instanceCount;
    const gl::Program *program    = context->getState().getProgram();
    if (program->usesMultiview())
    {
        adjustedInstanceCount *= program->getNumViews();
    }

    ANGLE_TRY(setDrawArraysState(context, first, count, adjustedInstanceCount));

    const FunctionsGL *functions = getFunctions();

    if (functions->drawArraysInstancedBaseInstance)
    {
        // GL 4.2+ or GL_EXT_base_instance
        functions->drawArraysInstancedBaseInstance(ToGLenum(mode), first, count,
                                                   adjustedInstanceCount, baseInstance);
    }
    else
    {
        // GL 3.3+ or GLES 3.2+
        // TODO(http://anglebug.com/3910): This is a temporary solution by setting and resetting
        // pointer offset calling vertexAttribPointer Will refactor stateCache and pass baseInstance
        // to setDrawArraysState to set pointer offset

        gl::AttributesMask attribToResetMask =
            updateAttributesForBaseInstance(program, baseInstance);

        functions->drawArraysInstanced(ToGLenum(mode), first, count, adjustedInstanceCount);

        resetUpdatedAttributes(attribToResetMask);
    }

    return angle::Result::Continue;
}

angle::Result ContextGL::drawElements(const gl::Context *context,
                                      gl::PrimitiveMode mode,
                                      GLsizei count,
                                      gl::DrawElementsType type,
                                      const void *indices)
{
    const gl::State &glState    = context->getState();
    const gl::Program *program  = glState.getProgram();
    const bool usesMultiview    = program->usesMultiview();
    const GLsizei instanceCount = usesMultiview ? program->getNumViews() : 0;
    const void *drawIndexPtr    = nullptr;

#if defined(ANGLE_STATE_VALIDATION_ENABLED)
    validateState();
#endif  // ANGLE_STATE_VALIDATION_ENABLED

    ANGLE_TRY(setDrawElementsState(context, count, type, indices, instanceCount, &drawIndexPtr));
    if (!usesMultiview)
    {
        getFunctions()->drawElements(ToGLenum(mode), count, ToGLenum(type), drawIndexPtr);
    }
    else
    {
        getFunctions()->drawElementsInstanced(ToGLenum(mode), count, ToGLenum(type), drawIndexPtr,
                                              instanceCount);
    }
    return angle::Result::Continue;
}

angle::Result ContextGL::drawElementsInstanced(const gl::Context *context,
                                               gl::PrimitiveMode mode,
                                               GLsizei count,
                                               gl::DrawElementsType type,
                                               const void *indices,
                                               GLsizei instances)
{
    GLsizei adjustedInstanceCount = instances;
    const gl::Program *program    = context->getState().getProgram();
    if (program->usesMultiview())
    {
        adjustedInstanceCount *= program->getNumViews();
    }
    const void *drawIndexPointer = nullptr;

    ANGLE_TRY(setDrawElementsState(context, count, type, indices, adjustedInstanceCount,
                                   &drawIndexPointer));
    getFunctions()->drawElementsInstanced(ToGLenum(mode), count, ToGLenum(type), drawIndexPointer,
                                          adjustedInstanceCount);
    return angle::Result::Continue;
}

angle::Result ContextGL::drawElementsInstancedBaseVertexBaseInstance(const gl::Context *context,
                                                                     gl::PrimitiveMode mode,
                                                                     GLsizei count,
                                                                     gl::DrawElementsType type,
                                                                     const void *indices,
                                                                     GLsizei instances,
                                                                     GLint baseVertex,
                                                                     GLuint baseInstance)
{
    GLsizei adjustedInstanceCount = instances;
    const gl::Program *program    = context->getState().getProgram();
    if (program->usesMultiview())
    {
        adjustedInstanceCount *= program->getNumViews();
    }
    const void *drawIndexPointer = nullptr;

    ANGLE_TRY(setDrawElementsState(context, count, type, indices, adjustedInstanceCount,
                                   &drawIndexPointer));

    const FunctionsGL *functions = getFunctions();

    if (functions->drawElementsInstancedBaseVertexBaseInstance)
    {
        // GL 4.2+ or GL_EXT_base_instance
        functions->drawElementsInstancedBaseVertexBaseInstance(
            ToGLenum(mode), count, ToGLenum(type), drawIndexPointer, adjustedInstanceCount,
            baseVertex, baseInstance);
    }
    else
    {
        // GL 3.3+ or GLES 3.2+
        // TODO(http://anglebug.com/3910): same as above
        gl::AttributesMask attribToResetMask =
            updateAttributesForBaseInstance(program, baseInstance);

        functions->drawElementsInstancedBaseVertex(ToGLenum(mode), count, ToGLenum(type),
                                                   drawIndexPointer, adjustedInstanceCount,
                                                   baseVertex);

        resetUpdatedAttributes(attribToResetMask);
    }

    return angle::Result::Continue;
}

angle::Result ContextGL::drawRangeElements(const gl::Context *context,
                                           gl::PrimitiveMode mode,
                                           GLuint start,
                                           GLuint end,
                                           GLsizei count,
                                           gl::DrawElementsType type,
                                           const void *indices)
{
    const gl::Program *program   = context->getState().getProgram();
    const bool usesMultiview     = program->usesMultiview();
    const GLsizei instanceCount  = usesMultiview ? program->getNumViews() : 0;
    const void *drawIndexPointer = nullptr;

    ANGLE_TRY(
        setDrawElementsState(context, count, type, indices, instanceCount, &drawIndexPointer));
    if (!usesMultiview)
    {
        getFunctions()->drawRangeElements(ToGLenum(mode), start, end, count, ToGLenum(type),
                                          drawIndexPointer);
    }
    else
    {
        getFunctions()->drawElementsInstanced(ToGLenum(mode), count, ToGLenum(type),
                                              drawIndexPointer, instanceCount);
    }
    return angle::Result::Continue;
}

angle::Result ContextGL::drawArraysIndirect(const gl::Context *context,
                                            gl::PrimitiveMode mode,
                                            const void *indirect)
{
    getFunctions()->drawArraysIndirect(ToGLenum(mode), indirect);
    return angle::Result::Continue;
}

angle::Result ContextGL::drawElementsIndirect(const gl::Context *context,
                                              gl::PrimitiveMode mode,
                                              gl::DrawElementsType type,
                                              const void *indirect)
{
    getFunctions()->drawElementsIndirect(ToGLenum(mode), ToGLenum(type), indirect);
    return angle::Result::Continue;
}

void ContextGL::stencilFillPath(const gl::Path *path, GLenum fillMode, GLuint mask)
{
    mRenderer->stencilFillPath(mState, path, fillMode, mask);
}

void ContextGL::stencilStrokePath(const gl::Path *path, GLint reference, GLuint mask)
{
    mRenderer->stencilStrokePath(mState, path, reference, mask);
}

void ContextGL::coverFillPath(const gl::Path *path, GLenum coverMode)
{
    mRenderer->coverFillPath(mState, path, coverMode);
}

void ContextGL::coverStrokePath(const gl::Path *path, GLenum coverMode)
{
    mRenderer->coverStrokePath(mState, path, coverMode);
}

void ContextGL::stencilThenCoverFillPath(const gl::Path *path,
                                         GLenum fillMode,
                                         GLuint mask,
                                         GLenum coverMode)
{
    mRenderer->stencilThenCoverFillPath(mState, path, fillMode, mask, coverMode);
}

void ContextGL::stencilThenCoverStrokePath(const gl::Path *path,
                                           GLint reference,
                                           GLuint mask,
                                           GLenum coverMode)
{
    mRenderer->stencilThenCoverStrokePath(mState, path, reference, mask, coverMode);
}

void ContextGL::coverFillPathInstanced(const std::vector<gl::Path *> &paths,
                                       GLenum coverMode,
                                       GLenum transformType,
                                       const GLfloat *transformValues)
{
    mRenderer->coverFillPathInstanced(mState, paths, coverMode, transformType, transformValues);
}

void ContextGL::coverStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                         GLenum coverMode,
                                         GLenum transformType,
                                         const GLfloat *transformValues)
{
    mRenderer->coverStrokePathInstanced(mState, paths, coverMode, transformType, transformValues);
}

void ContextGL::stencilFillPathInstanced(const std::vector<gl::Path *> &paths,
                                         GLenum fillMode,
                                         GLuint mask,
                                         GLenum transformType,
                                         const GLfloat *transformValues)
{
    mRenderer->stencilFillPathInstanced(mState, paths, fillMode, mask, transformType,
                                        transformValues);
}

void ContextGL::stencilStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                           GLint reference,
                                           GLuint mask,
                                           GLenum transformType,
                                           const GLfloat *transformValues)
{
    mRenderer->stencilStrokePathInstanced(mState, paths, reference, mask, transformType,
                                          transformValues);
}

void ContextGL::stencilThenCoverFillPathInstanced(const std::vector<gl::Path *> &paths,
                                                  GLenum coverMode,
                                                  GLenum fillMode,
                                                  GLuint mask,
                                                  GLenum transformType,
                                                  const GLfloat *transformValues)
{
    mRenderer->stencilThenCoverFillPathInstanced(mState, paths, coverMode, fillMode, mask,
                                                 transformType, transformValues);
}

void ContextGL::stencilThenCoverStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                                    GLenum coverMode,
                                                    GLint reference,
                                                    GLuint mask,
                                                    GLenum transformType,
                                                    const GLfloat *transformValues)
{
    mRenderer->stencilThenCoverStrokePathInstanced(mState, paths, coverMode, reference, mask,
                                                   transformType, transformValues);
}

gl::GraphicsResetStatus ContextGL::getResetStatus()
{
    return mRenderer->getResetStatus();
}

std::string ContextGL::getVendorString() const
{
    return mRenderer->getVendorString();
}

std::string ContextGL::getRendererDescription() const
{
    return mRenderer->getRendererDescription();
}

void ContextGL::insertEventMarker(GLsizei length, const char *marker)
{
    mRenderer->insertEventMarker(length, marker);
}

void ContextGL::pushGroupMarker(GLsizei length, const char *marker)
{
    mRenderer->pushGroupMarker(length, marker);
}

void ContextGL::popGroupMarker()
{
    mRenderer->popGroupMarker();
}

void ContextGL::pushDebugGroup(GLenum source, GLuint id, const std::string &message)
{
    mRenderer->pushDebugGroup(source, id, message);
}

void ContextGL::popDebugGroup()
{
    mRenderer->popDebugGroup();
}

angle::Result ContextGL::syncState(const gl::Context *context,
                                   const gl::State::DirtyBits &dirtyBits,
                                   const gl::State::DirtyBits &bitMask)
{
    mRenderer->getStateManager()->syncState(context, dirtyBits, bitMask);
    return angle::Result::Continue;
}

GLint ContextGL::getGPUDisjoint()
{
    return mRenderer->getGPUDisjoint();
}

GLint64 ContextGL::getTimestamp()
{
    return mRenderer->getTimestamp();
}

angle::Result ContextGL::onMakeCurrent(const gl::Context *context)
{
    // Queries need to be paused/resumed on context switches
    return mRenderer->getStateManager()->onMakeCurrent(context);
}

gl::Caps ContextGL::getNativeCaps() const
{
    return mRenderer->getNativeCaps();
}

const gl::TextureCapsMap &ContextGL::getNativeTextureCaps() const
{
    return mRenderer->getNativeTextureCaps();
}

const gl::Extensions &ContextGL::getNativeExtensions() const
{
    return mRenderer->getNativeExtensions();
}

const gl::Limitations &ContextGL::getNativeLimitations() const
{
    return mRenderer->getNativeLimitations();
}

StateManagerGL *ContextGL::getStateManager()
{
    return mRenderer->getStateManager();
}

const angle::FeaturesGL &ContextGL::getFeaturesGL() const
{
    return mRenderer->getFeatures();
}

BlitGL *ContextGL::getBlitter() const
{
    return mRenderer->getBlitter();
}

ClearMultiviewGL *ContextGL::getMultiviewClearer() const
{
    return mRenderer->getMultiviewClearer();
}

angle::Result ContextGL::dispatchCompute(const gl::Context *context,
                                         GLuint numGroupsX,
                                         GLuint numGroupsY,
                                         GLuint numGroupsZ)
{
    return mRenderer->dispatchCompute(context, numGroupsX, numGroupsY, numGroupsZ);
}

angle::Result ContextGL::dispatchComputeIndirect(const gl::Context *context, GLintptr indirect)
{
    return mRenderer->dispatchComputeIndirect(context, indirect);
}

angle::Result ContextGL::memoryBarrier(const gl::Context *context, GLbitfield barriers)
{
    return mRenderer->memoryBarrier(barriers);
}
angle::Result ContextGL::memoryBarrierByRegion(const gl::Context *context, GLbitfield barriers)
{
    return mRenderer->memoryBarrierByRegion(barriers);
}

void ContextGL::setMaxShaderCompilerThreads(GLuint count)
{
    mRenderer->setMaxShaderCompilerThreads(count);
}

void ContextGL::invalidateTexture(gl::TextureType target)
{
    mRenderer->getStateManager()->invalidateTexture(target);
}

void ContextGL::validateState() const
{
    const StateManagerGL *stateManager = mRenderer->getStateManager();
    stateManager->validateState();
}

}  // namespace rx
