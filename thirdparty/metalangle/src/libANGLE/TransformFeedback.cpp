//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "libANGLE/TransformFeedback.h"

#include "common/mathutil.h"
#include "libANGLE/Buffer.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Context.h"
#include "libANGLE/Program.h"
#include "libANGLE/State.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/TransformFeedbackImpl.h"

#include <limits>

namespace gl
{

angle::CheckedNumeric<GLsizeiptr> GetVerticesNeededForDraw(PrimitiveMode primitiveMode,
                                                           GLsizei count,
                                                           GLsizei primcount)
{
    if (count < 0 || primcount < 0)
    {
        return 0;
    }
    // Transform feedback only outputs complete primitives, so we need to round down to the nearest
    // complete primitive before multiplying by the number of instances.
    angle::CheckedNumeric<GLsizeiptr> checkedCount     = count;
    angle::CheckedNumeric<GLsizeiptr> checkedPrimcount = primcount;
    switch (primitiveMode)
    {
        case PrimitiveMode::Triangles:
            return checkedPrimcount * (checkedCount - checkedCount % 3);
        case PrimitiveMode::Lines:
            return checkedPrimcount * (checkedCount - checkedCount % 2);
        case PrimitiveMode::Points:
            return checkedPrimcount * checkedCount;
        default:
            UNREACHABLE();
            return checkedPrimcount * checkedCount;
    }
}

TransformFeedbackState::TransformFeedbackState(size_t maxIndexedBuffers)
    : mLabel(),
      mActive(false),
      mPrimitiveMode(PrimitiveMode::InvalidEnum),
      mPaused(false),
      mVerticesDrawn(0),
      mVertexCapacity(0),
      mProgram(nullptr),
      mIndexedBuffers(maxIndexedBuffers)
{}

TransformFeedbackState::~TransformFeedbackState() {}

const OffsetBindingPointer<Buffer> &TransformFeedbackState::getIndexedBuffer(size_t idx) const
{
    return mIndexedBuffers[idx];
}

const std::vector<OffsetBindingPointer<Buffer>> &TransformFeedbackState::getIndexedBuffers() const
{
    return mIndexedBuffers;
}

GLsizeiptr TransformFeedbackState::getPrimitivesDrawn() const
{
    switch (mPrimitiveMode)
    {
        case gl::PrimitiveMode::Points:
            return mVerticesDrawn;
        case gl::PrimitiveMode::Lines:
            return mVerticesDrawn / 2;
        case gl::PrimitiveMode::Triangles:
            return mVerticesDrawn / 3;
        default:
            return 0;
    }
}

TransformFeedback::TransformFeedback(rx::GLImplFactory *implFactory,
                                     TransformFeedbackID id,
                                     const Caps &caps)
    : RefCountObject(id),
      mState(caps.maxTransformFeedbackSeparateAttributes),
      mImplementation(implFactory->createTransformFeedback(mState))
{
    ASSERT(mImplementation != nullptr);
}

void TransformFeedback::onDestroy(const Context *context)
{
    ASSERT(!context || !context->isCurrentTransformFeedback(this));
    if (mState.mProgram)
    {
        mState.mProgram->release(context);
        mState.mProgram = nullptr;
    }

    ASSERT(!mState.mProgram);
    for (size_t i = 0; i < mState.mIndexedBuffers.size(); i++)
    {
        mState.mIndexedBuffers[i].set(context, nullptr, 0, 0);
    }
}

TransformFeedback::~TransformFeedback()
{
    SafeDelete(mImplementation);
}

void TransformFeedback::setLabel(const Context *context, const std::string &label)
{
    mState.mLabel = label;
}

const std::string &TransformFeedback::getLabel() const
{
    return mState.mLabel;
}

angle::Result TransformFeedback::begin(const Context *context,
                                       PrimitiveMode primitiveMode,
                                       Program *program)
{
    ANGLE_TRY(mImplementation->begin(context, primitiveMode));
    mState.mActive        = true;
    mState.mPrimitiveMode = primitiveMode;
    mState.mPaused        = false;
    mState.mVerticesDrawn = 0;
    bindProgram(context, program);

    if (program)
    {
        // Compute the number of vertices we can draw before overflowing the bound buffers.
        auto strides = program->getTransformFeedbackStrides();
        ASSERT(strides.size() <= mState.mIndexedBuffers.size() && !strides.empty());
        GLsizeiptr minCapacity = std::numeric_limits<GLsizeiptr>::max();
        for (size_t index = 0; index < strides.size(); index++)
        {
            GLsizeiptr capacity =
                GetBoundBufferAvailableSize(mState.mIndexedBuffers[index]) / strides[index];
            minCapacity = std::min(minCapacity, capacity);
        }
        mState.mVertexCapacity = minCapacity;
    }
    else
    {
        mState.mVertexCapacity = 0;
    }
    return angle::Result::Continue;
}

angle::Result TransformFeedback::end(const Context *context)
{
    ANGLE_TRY(mImplementation->end(context));
    mState.mActive         = false;
    mState.mPrimitiveMode  = PrimitiveMode::InvalidEnum;
    mState.mPaused         = false;
    mState.mVerticesDrawn  = 0;
    mState.mVertexCapacity = 0;
    if (mState.mProgram)
    {
        mState.mProgram->release(context);
        mState.mProgram = nullptr;
    }
    return angle::Result::Continue;
}

angle::Result TransformFeedback::pause(const Context *context)
{
    ANGLE_TRY(mImplementation->pause(context));
    mState.mPaused = true;
    return angle::Result::Continue;
}

angle::Result TransformFeedback::resume(const Context *context)
{
    ANGLE_TRY(mImplementation->resume(context));
    mState.mPaused = false;
    return angle::Result::Continue;
}

bool TransformFeedback::isPaused() const
{
    return mState.mPaused;
}

PrimitiveMode TransformFeedback::getPrimitiveMode() const
{
    return mState.mPrimitiveMode;
}

bool TransformFeedback::checkBufferSpaceForDraw(GLsizei count, GLsizei primcount) const
{
    auto vertices =
        mState.mVerticesDrawn + GetVerticesNeededForDraw(mState.mPrimitiveMode, count, primcount);
    return vertices.IsValid() && vertices.ValueOrDie() <= mState.mVertexCapacity;
}

void TransformFeedback::onVerticesDrawn(const Context *context, GLsizei count, GLsizei primcount)
{
    ASSERT(mState.mActive && !mState.mPaused);
    // All draws should be validated with checkBufferSpaceForDraw so ValueOrDie should never fail.
    mState.mVerticesDrawn =
        (mState.mVerticesDrawn + GetVerticesNeededForDraw(mState.mPrimitiveMode, count, primcount))
            .ValueOrDie();

    for (auto &buffer : mState.mIndexedBuffers)
    {
        if (buffer.get() != nullptr)
        {
            buffer->onDataChanged();
        }
    }
}

void TransformFeedback::bindProgram(const Context *context, Program *program)
{
    if (mState.mProgram != program)
    {
        if (mState.mProgram != nullptr)
        {
            mState.mProgram->release(context);
        }
        mState.mProgram = program;
        if (mState.mProgram != nullptr)
        {
            mState.mProgram->addRef();
        }
    }
}

bool TransformFeedback::hasBoundProgram(ShaderProgramID program) const
{
    return mState.mProgram != nullptr && mState.mProgram->id().value == program.value;
}

angle::Result TransformFeedback::detachBuffer(const Context *context, BufferID bufferID)
{
    bool isBound = context->isCurrentTransformFeedback(this);
    for (size_t index = 0; index < mState.mIndexedBuffers.size(); index++)
    {
        if (mState.mIndexedBuffers[index].id() == bufferID)
        {
            if (isBound)
            {
                mState.mIndexedBuffers[index]->onTFBindingChanged(context, false, true);
            }
            mState.mIndexedBuffers[index].set(context, nullptr, 0, 0);
            ANGLE_TRY(
                mImplementation->bindIndexedBuffer(context, index, mState.mIndexedBuffers[index]));
        }
    }

    return angle::Result::Continue;
}

angle::Result TransformFeedback::bindIndexedBuffer(const Context *context,
                                                   size_t index,
                                                   Buffer *buffer,
                                                   size_t offset,
                                                   size_t size)
{
    ASSERT(index < mState.mIndexedBuffers.size());
    bool isBound = context && context->isCurrentTransformFeedback(this);
    if (isBound && mState.mIndexedBuffers[index].get())
    {
        mState.mIndexedBuffers[index]->onTFBindingChanged(context, false, true);
    }
    mState.mIndexedBuffers[index].set(context, buffer, offset, size);
    if (isBound && buffer)
    {
        buffer->onTFBindingChanged(context, true, true);
    }

    return mImplementation->bindIndexedBuffer(context, index, mState.mIndexedBuffers[index]);
}

const OffsetBindingPointer<Buffer> &TransformFeedback::getIndexedBuffer(size_t index) const
{
    ASSERT(index < mState.mIndexedBuffers.size());
    return mState.mIndexedBuffers[index];
}

size_t TransformFeedback::getIndexedBufferCount() const
{
    return mState.mIndexedBuffers.size();
}

bool TransformFeedback::buffersBoundForOtherUse() const
{
    for (auto &buffer : mState.mIndexedBuffers)
    {
        if (buffer.get() && buffer->isBoundForTransformFeedbackAndOtherUse())
        {
            return true;
        }
    }
    return false;
}

rx::TransformFeedbackImpl *TransformFeedback::getImplementation() const
{
    return mImplementation;
}

void TransformFeedback::onBindingChanged(const Context *context, bool bound)
{
    for (auto &buffer : mState.mIndexedBuffers)
    {
        if (buffer.get())
        {
            buffer->onTFBindingChanged(context, bound, true);
        }
    }
}
}  // namespace gl
