//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Buffer.h: Defines the gl::Buffer class, representing storage of vertex and/or
// index data. Implements GL buffer objects and related functionality.
// [OpenGL ES 2.0.24] section 2.9 page 21.

#ifndef LIBANGLE_BUFFER_H_
#define LIBANGLE_BUFFER_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/IndexRangeCache.h"
#include "libANGLE/Observer.h"
#include "libANGLE/RefCountObject.h"

namespace rx
{
class BufferImpl;
class GLImplFactory;
}  // namespace rx

namespace gl
{
class Buffer;
class Context;

class BufferState final : angle::NonCopyable
{
  public:
    BufferState();
    ~BufferState();

    BufferUsage getUsage() const { return mUsage; }
    GLbitfield getAccessFlags() const { return mAccessFlags; }
    GLenum getAccess() const { return mAccess; }
    GLboolean isMapped() const { return mMapped; }
    void *getMapPointer() const { return mMapPointer; }
    GLint64 getMapOffset() const { return mMapOffset; }
    GLint64 getMapLength() const { return mMapLength; }
    GLint64 getSize() const { return mSize; }
    bool isBoundForTransformFeedback() const { return mTransformFeedbackIndexedBindingCount != 0; }

  private:
    friend class Buffer;

    std::string mLabel;

    BufferUsage mUsage;
    GLint64 mSize;
    GLbitfield mAccessFlags;
    GLenum mAccess;
    GLboolean mMapped;
    void *mMapPointer;
    GLint64 mMapOffset;
    GLint64 mMapLength;
    int mBindingCount;
    int mTransformFeedbackIndexedBindingCount;
    int mTransformFeedbackGenericBindingCount;
};

class Buffer final : public RefCountObject<BufferID>,
                     public LabeledObject,
                     public angle::ObserverInterface,
                     public angle::Subject
{
  public:
    Buffer(rx::GLImplFactory *factory, BufferID id);
    ~Buffer() override;
    void onDestroy(const Context *context) override;

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    angle::Result bufferData(Context *context,
                             BufferBinding target,
                             const void *data,
                             GLsizeiptr size,
                             BufferUsage usage);
    angle::Result bufferSubData(const Context *context,
                                BufferBinding target,
                                const void *data,
                                GLsizeiptr size,
                                GLintptr offset);
    angle::Result copyBufferSubData(const Context *context,
                                    Buffer *source,
                                    GLintptr sourceOffset,
                                    GLintptr destOffset,
                                    GLsizeiptr size);
    angle::Result map(const Context *context, GLenum access);
    angle::Result mapRange(const Context *context,
                           GLintptr offset,
                           GLsizeiptr length,
                           GLbitfield access);
    angle::Result unmap(const Context *context, GLboolean *result);

    // These are called when another operation changes Buffer data.
    void onDataChanged();

    angle::Result getIndexRange(const gl::Context *context,
                                DrawElementsType type,
                                size_t offset,
                                size_t count,
                                bool primitiveRestartEnabled,
                                IndexRange *outRange) const;

    BufferUsage getUsage() const { return mState.mUsage; }
    GLbitfield getAccessFlags() const { return mState.mAccessFlags; }
    GLenum getAccess() const { return mState.mAccess; }
    GLboolean isMapped() const { return mState.mMapped; }
    void *getMapPointer() const { return mState.mMapPointer; }
    GLint64 getMapOffset() const { return mState.mMapOffset; }
    GLint64 getMapLength() const { return mState.mMapLength; }
    GLint64 getSize() const { return mState.mSize; }
    GLint64 getMemorySize() const;

    rx::BufferImpl *getImplementation() const { return mImpl; }

    ANGLE_INLINE bool isBound() const { return mState.mBindingCount > 0; }

    ANGLE_INLINE bool isBoundForTransformFeedbackAndOtherUse() const
    {
        // The transform feedback generic binding point is not an indexed binding point but it also
        // does not count as a non-transform-feedback use of the buffer, so we subtract it from the
        // binding count when checking if the buffer is bound to a non-transform-feedback location.
        // See https://crbug.com/853978
        return mState.mTransformFeedbackIndexedBindingCount > 0 &&
               mState.mTransformFeedbackIndexedBindingCount !=
                   mState.mBindingCount - mState.mTransformFeedbackGenericBindingCount;
    }

    bool isDoubleBoundForTransformFeedback() const;
    void onTFBindingChanged(const Context *context, bool bound, bool indexed);
    void onNonTFBindingChanged(int incr) { mState.mBindingCount += incr; }

    // angle::ObserverInterface implementation.
    void onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message) override;

  private:
    BufferState mState;
    rx::BufferImpl *mImpl;
    angle::ObserverBinding mImplObserver;

    mutable IndexRangeCache mIndexRangeCache;
};

}  // namespace gl

#endif  // LIBANGLE_BUFFER_H_
