//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Buffer.cpp: Implements the gl::Buffer class, representing storage of vertex and/or
// index data. Implements GL buffer objects and related functionality.
// [OpenGL ES 2.0.24] section 2.9 page 21.

#include "libANGLE/Buffer.h"

#include "libANGLE/Context.h"
#include "libANGLE/renderer/BufferImpl.h"
#include "libANGLE/renderer/GLImplFactory.h"

namespace gl
{
namespace
{
constexpr angle::SubjectIndex kImplementationSubjectIndex = 0;
}  // anonymous namespace

BufferState::BufferState()
    : mLabel(),
      mUsage(BufferUsage::StaticDraw),
      mSize(0),
      mAccessFlags(0),
      mAccess(GL_WRITE_ONLY_OES),
      mMapped(GL_FALSE),
      mMapPointer(nullptr),
      mMapOffset(0),
      mMapLength(0),
      mBindingCount(0),
      mTransformFeedbackIndexedBindingCount(0),
      mTransformFeedbackGenericBindingCount(0)
{}

BufferState::~BufferState() {}

Buffer::Buffer(rx::GLImplFactory *factory, BufferID id)
    : RefCountObject(id),
      mImpl(factory->createBuffer(mState)),
      mImplObserver(this, kImplementationSubjectIndex)
{
    mImplObserver.bind(mImpl);
}

Buffer::~Buffer()
{
    SafeDelete(mImpl);
}

void Buffer::onDestroy(const Context *context)
{
    // In tests, mImpl might be null.
    if (mImpl)
        mImpl->destroy(context);
}

void Buffer::setLabel(const Context *context, const std::string &label)
{
    mState.mLabel = label;
}

const std::string &Buffer::getLabel() const
{
    return mState.mLabel;
}

angle::Result Buffer::bufferData(Context *context,
                                 BufferBinding target,
                                 const void *data,
                                 GLsizeiptr size,
                                 BufferUsage usage)
{
    const void *dataForImpl = data;

    // If we are using robust resource init, make sure the buffer starts cleared.
    // Note: the Context is checked for nullptr because of some testing code.
    // TODO(jmadill): Investigate lazier clearing.
    if (context && context->getState().isRobustResourceInitEnabled() && !data && size > 0)
    {
        angle::MemoryBuffer *scratchBuffer = nullptr;
        ANGLE_CHECK_GL_ALLOC(
            context, context->getZeroFilledBuffer(static_cast<size_t>(size), &scratchBuffer));
        dataForImpl = scratchBuffer->data();
    }

    ANGLE_TRY(mImpl->setData(context, target, dataForImpl, size, usage));

    mIndexRangeCache.clear();
    mState.mUsage = usage;
    mState.mSize  = size;

    // Notify when storage changes.
    onStateChange(angle::SubjectMessage::SubjectChanged);

    return angle::Result::Continue;
}

angle::Result Buffer::bufferSubData(const Context *context,
                                    BufferBinding target,
                                    const void *data,
                                    GLsizeiptr size,
                                    GLintptr offset)
{
    ANGLE_TRY(mImpl->setSubData(context, target, data, size, offset));

    mIndexRangeCache.invalidateRange(static_cast<unsigned int>(offset),
                                     static_cast<unsigned int>(size));

    // Notify when data changes.
    onStateChange(angle::SubjectMessage::ContentsChanged);

    return angle::Result::Continue;
}

angle::Result Buffer::copyBufferSubData(const Context *context,
                                        Buffer *source,
                                        GLintptr sourceOffset,
                                        GLintptr destOffset,
                                        GLsizeiptr size)
{
    ANGLE_TRY(
        mImpl->copySubData(context, source->getImplementation(), sourceOffset, destOffset, size));

    mIndexRangeCache.invalidateRange(static_cast<unsigned int>(destOffset),
                                     static_cast<unsigned int>(size));

    // Notify when data changes.
    onStateChange(angle::SubjectMessage::ContentsChanged);

    return angle::Result::Continue;
}

angle::Result Buffer::map(const Context *context, GLenum access)
{
    ASSERT(!mState.mMapped);

    mState.mMapPointer = nullptr;
    ANGLE_TRY(mImpl->map(context, access, &mState.mMapPointer));

    ASSERT(access == GL_WRITE_ONLY_OES);

    mState.mMapped      = GL_TRUE;
    mState.mMapOffset   = 0;
    mState.mMapLength   = mState.mSize;
    mState.mAccess      = access;
    mState.mAccessFlags = GL_MAP_WRITE_BIT;
    mIndexRangeCache.clear();

    // Notify when state changes.
    onStateChange(angle::SubjectMessage::SubjectMapped);

    return angle::Result::Continue;
}

angle::Result Buffer::mapRange(const Context *context,
                               GLintptr offset,
                               GLsizeiptr length,
                               GLbitfield access)
{
    ASSERT(!mState.mMapped);
    ASSERT(offset + length <= mState.mSize);

    mState.mMapPointer = nullptr;
    ANGLE_TRY(mImpl->mapRange(context, offset, length, access, &mState.mMapPointer));

    mState.mMapped      = GL_TRUE;
    mState.mMapOffset   = static_cast<GLint64>(offset);
    mState.mMapLength   = static_cast<GLint64>(length);
    mState.mAccess      = GL_WRITE_ONLY_OES;
    mState.mAccessFlags = access;

    // The OES_mapbuffer extension states that GL_WRITE_ONLY_OES is the only valid
    // value for GL_BUFFER_ACCESS_OES because it was written against ES2.  Since there is
    // no update for ES3 and the GL_READ_ONLY and GL_READ_WRITE enums don't exist for ES,
    // we cannot properly set GL_BUFFER_ACCESS_OES when glMapBufferRange is called.

    if ((access & GL_MAP_WRITE_BIT) > 0)
    {
        mIndexRangeCache.invalidateRange(static_cast<unsigned int>(offset),
                                         static_cast<unsigned int>(length));
    }

    // Notify when state changes.
    onStateChange(angle::SubjectMessage::SubjectMapped);

    return angle::Result::Continue;
}

angle::Result Buffer::unmap(const Context *context, GLboolean *result)
{
    ASSERT(mState.mMapped);

    *result = GL_FALSE;
    ANGLE_TRY(mImpl->unmap(context, result));

    mState.mMapped      = GL_FALSE;
    mState.mMapPointer  = nullptr;
    mState.mMapOffset   = 0;
    mState.mMapLength   = 0;
    mState.mAccess      = GL_WRITE_ONLY_OES;
    mState.mAccessFlags = 0;

    // Notify when data changes.
    onStateChange(angle::SubjectMessage::SubjectUnmapped);

    return angle::Result::Continue;
}

void Buffer::onDataChanged()
{
    mIndexRangeCache.clear();

    // Notify when data changes.
    onStateChange(angle::SubjectMessage::ContentsChanged);

    mImpl->onDataChanged();
}

angle::Result Buffer::getIndexRange(const gl::Context *context,
                                    DrawElementsType type,
                                    size_t offset,
                                    size_t count,
                                    bool primitiveRestartEnabled,
                                    IndexRange *outRange) const
{
    if (mIndexRangeCache.findRange(type, offset, count, primitiveRestartEnabled, outRange))
    {
        return angle::Result::Continue;
    }

    ANGLE_TRY(
        mImpl->getIndexRange(context, type, offset, count, primitiveRestartEnabled, outRange));

    mIndexRangeCache.addRange(type, offset, count, primitiveRestartEnabled, *outRange);

    return angle::Result::Continue;
}

GLint64 Buffer::getMemorySize() const
{
    GLint64 implSize = mImpl->getMemorySize();
    return implSize > 0 ? implSize : mState.mSize;
}

bool Buffer::isDoubleBoundForTransformFeedback() const
{
    return mState.mTransformFeedbackIndexedBindingCount > 1;
}

void Buffer::onTFBindingChanged(const Context *context, bool bound, bool indexed)
{
    ASSERT(bound || mState.mBindingCount > 0);
    mState.mBindingCount += bound ? 1 : -1;
    if (indexed)
    {
        ASSERT(bound || mState.mTransformFeedbackIndexedBindingCount > 0);
        mState.mTransformFeedbackIndexedBindingCount += bound ? 1 : -1;

        onStateChange(angle::SubjectMessage::BindingChanged);
    }
    else
    {
        mState.mTransformFeedbackGenericBindingCount += bound ? 1 : -1;
    }
}

void Buffer::onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message)
{
    // Pass it along!
    ASSERT(index == kImplementationSubjectIndex);
    ASSERT(message == angle::SubjectMessage::SubjectChanged);
    onStateChange(angle::SubjectMessage::SubjectChanged);
}
}  // namespace gl
