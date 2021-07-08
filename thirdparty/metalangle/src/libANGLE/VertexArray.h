//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// This class contains prototypes for representing GLES 3 Vertex Array Objects:
//
//   The buffer objects that are to be used by the vertex stage of the GL are collected
//   together to form a vertex array object. All state related to the definition of data used
//   by the vertex processor is encapsulated in a vertex array object.
//

#ifndef LIBANGLE_VERTEXARRAY_H_
#define LIBANGLE_VERTEXARRAY_H_

#include "common/Optional.h"
#include "libANGLE/Constants.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Observer.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/VertexAttribute.h"

#include <vector>

namespace rx
{
class GLImplFactory;
class VertexArrayImpl;
}  // namespace rx

namespace gl
{
class Buffer;

class VertexArrayState final : angle::NonCopyable
{
  public:
    VertexArrayState(VertexArray *vertexArray, size_t maxAttribs, size_t maxBindings);
    ~VertexArrayState();

    const std::string &getLabel() const { return mLabel; }

    Buffer *getElementArrayBuffer() const { return mElementArrayBuffer.get(); }
    size_t getMaxAttribs() const { return mVertexAttributes.size(); }
    size_t getMaxBindings() const { return mVertexBindings.size(); }
    const AttributesMask &getEnabledAttributesMask() const { return mEnabledAttributesMask; }
    const std::vector<VertexAttribute> &getVertexAttributes() const { return mVertexAttributes; }
    const VertexAttribute &getVertexAttribute(size_t attribIndex) const
    {
        return mVertexAttributes[attribIndex];
    }
    const std::vector<VertexBinding> &getVertexBindings() const { return mVertexBindings; }
    const VertexBinding &getVertexBinding(size_t bindingIndex) const
    {
        return mVertexBindings[bindingIndex];
    }
    const VertexBinding &getBindingFromAttribIndex(size_t attribIndex) const
    {
        return mVertexBindings[mVertexAttributes[attribIndex].bindingIndex];
    }
    size_t getBindingIndexFromAttribIndex(size_t attribIndex) const
    {
        return mVertexAttributes[attribIndex].bindingIndex;
    }

    void setAttribBinding(const Context *context, size_t attribIndex, GLuint newBindingIndex);

    // Extra validation performed on the Vertex Array.
    bool hasEnabledNullPointerClientArray() const;

    // Get all the attributes in an AttributesMask that are using the given binding.
    AttributesMask getBindingToAttributesMask(GLuint bindingIndex) const;

  private:
    friend class VertexArray;
    std::string mLabel;
    std::vector<VertexAttribute> mVertexAttributes;
    SubjectBindingPointer<Buffer> mElementArrayBuffer;
    std::vector<VertexBinding> mVertexBindings;
    AttributesMask mEnabledAttributesMask;
    ComponentTypeMask mVertexAttributesTypeMask;

    // This is a performance optimization for buffer binding. Allows element array buffer updates.
    friend class State;

    // From the GLES 3.1 spec:
    // When a generic attribute array is sourced from client memory, the vertex attribute binding
    // state is ignored. Thus we don't have to worry about binding state when using client memory
    // attribs.
    gl::AttributesMask mClientMemoryAttribsMask;
    gl::AttributesMask mNullPointerClientMemoryAttribsMask;

    // Used for validation cache. Indexed by attribute.
    AttributesMask mCachedMappedArrayBuffers;
    AttributesMask mCachedEnabledMappedArrayBuffers;
};

class VertexArray final : public angle::ObserverInterface,
                          public LabeledObject,
                          public angle::Subject
{
  public:
    // Dirty bits for VertexArrays use a heirarchical design. At the top level, each attribute
    // has a single dirty bit. Then an array of MAX_ATTRIBS dirty bits each has a dirty bit for
    // enabled/pointer/format/binding. Bindings are handled similarly. Note that because the
    // total number of dirty bits is 33, it will not be as fast on a 32-bit machine, which
    // can't support the advanced 64-bit scanning intrinsics. We could consider packing the
    // binding and attribute bits together if this becomes a problem.
    //
    // Special note on "DIRTY_ATTRIB_POINTER_BUFFER": this is a special case when the app
    // calls glVertexAttribPointer but only changes a VBO and/or offset binding. This allows
    // the Vulkan back-end to skip performing a pipeline change for performance.
    enum DirtyBitType
    {
        DIRTY_BIT_ELEMENT_ARRAY_BUFFER,
        DIRTY_BIT_ELEMENT_ARRAY_BUFFER_DATA,

        // Dirty bits for attributes.
        DIRTY_BIT_ATTRIB_0,
        DIRTY_BIT_ATTRIB_MAX = DIRTY_BIT_ATTRIB_0 + gl::MAX_VERTEX_ATTRIBS,

        // Dirty bits for bindings.
        DIRTY_BIT_BINDING_0   = DIRTY_BIT_ATTRIB_MAX,
        DIRTY_BIT_BINDING_MAX = DIRTY_BIT_BINDING_0 + gl::MAX_VERTEX_ATTRIB_BINDINGS,

        // We keep separate dirty bits for bound buffers whose data changed since last update.
        DIRTY_BIT_BUFFER_DATA_0   = DIRTY_BIT_BINDING_MAX,
        DIRTY_BIT_BUFFER_DATA_MAX = DIRTY_BIT_BUFFER_DATA_0 + gl::MAX_VERTEX_ATTRIB_BINDINGS,

        DIRTY_BIT_UNKNOWN = DIRTY_BIT_BUFFER_DATA_MAX,
        DIRTY_BIT_MAX     = DIRTY_BIT_UNKNOWN,
    };

    // We want to keep the number of dirty bits within 64 to keep iteration times fast.
    static_assert(DIRTY_BIT_MAX <= 64, "Too many vertex array dirty bits.");

    enum DirtyAttribBitType
    {
        DIRTY_ATTRIB_ENABLED,
        DIRTY_ATTRIB_POINTER,
        DIRTY_ATTRIB_FORMAT,
        DIRTY_ATTRIB_BINDING,
        DIRTY_ATTRIB_POINTER_BUFFER,
        DIRTY_ATTRIB_UNKNOWN,
        DIRTY_ATTRIB_MAX = DIRTY_ATTRIB_UNKNOWN,
    };

    enum DirtyBindingBitType
    {
        DIRTY_BINDING_BUFFER,
        DIRTY_BINDING_DIVISOR,
        DIRTY_BINDING_UNKNOWN,
        DIRTY_BINDING_MAX = DIRTY_BINDING_UNKNOWN,
    };

    using DirtyBits             = angle::BitSet<DIRTY_BIT_MAX>;
    using DirtyAttribBits       = angle::BitSet<DIRTY_ATTRIB_MAX>;
    using DirtyBindingBits      = angle::BitSet<DIRTY_BINDING_MAX>;
    using DirtyAttribBitsArray  = std::array<DirtyAttribBits, gl::MAX_VERTEX_ATTRIBS>;
    using DirtyBindingBitsArray = std::array<DirtyBindingBits, gl::MAX_VERTEX_ATTRIB_BINDINGS>;

    VertexArray(rx::GLImplFactory *factory,
                VertexArrayID id,
                size_t maxAttribs,
                size_t maxAttribBindings);

    void onDestroy(const Context *context);

    VertexArrayID id() const { return mId; }

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    const VertexBinding &getVertexBinding(size_t bindingIndex) const;
    const VertexAttribute &getVertexAttribute(size_t attribIndex) const;
    const VertexBinding &getBindingFromAttribIndex(size_t attribIndex) const
    {
        return mState.getBindingFromAttribIndex(attribIndex);
    }

    // Returns true if the function finds and detaches a bound buffer.
    bool detachBuffer(const Context *context, BufferID bufferID);

    void setVertexAttribDivisor(const Context *context, size_t index, GLuint divisor);
    void enableAttribute(size_t attribIndex, bool enabledState);

    void setVertexAttribPointer(const Context *context,
                                size_t attribIndex,
                                Buffer *boundBuffer,
                                GLint size,
                                VertexAttribType type,
                                bool normalized,
                                GLsizei stride,
                                const void *pointer);

    void setVertexAttribIPointer(const Context *context,
                                 size_t attribIndex,
                                 Buffer *boundBuffer,
                                 GLint size,
                                 VertexAttribType type,
                                 GLsizei stride,
                                 const void *pointer);

    void setVertexAttribFormat(size_t attribIndex,
                               GLint size,
                               VertexAttribType type,
                               bool normalized,
                               bool pureInteger,
                               GLuint relativeOffset);
    void bindVertexBuffer(const Context *context,
                          size_t bindingIndex,
                          Buffer *boundBuffer,
                          GLintptr offset,
                          GLsizei stride);
    void setVertexAttribBinding(const Context *context, size_t attribIndex, GLuint bindingIndex);
    void setVertexBindingDivisor(size_t bindingIndex, GLuint divisor);

    Buffer *getElementArrayBuffer() const { return mState.getElementArrayBuffer(); }
    size_t getMaxAttribs() const { return mState.getMaxAttribs(); }
    size_t getMaxBindings() const { return mState.getMaxBindings(); }

    const std::vector<VertexAttribute> &getVertexAttributes() const
    {
        return mState.getVertexAttributes();
    }
    const std::vector<VertexBinding> &getVertexBindings() const
    {
        return mState.getVertexBindings();
    }

    rx::VertexArrayImpl *getImplementation() const { return mVertexArray; }

    const AttributesMask &getEnabledAttributesMask() const
    {
        return mState.getEnabledAttributesMask();
    }

    gl::AttributesMask getClientAttribsMask() const { return mState.mClientMemoryAttribsMask; }

    bool hasEnabledNullPointerClientArray() const
    {
        return mState.hasEnabledNullPointerClientArray();
    }

    bool hasMappedEnabledArrayBuffer() const
    {
        return mState.mCachedEnabledMappedArrayBuffers.any();
    }

    // Observer implementation
    void onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message) override;

    static size_t GetVertexIndexFromDirtyBit(size_t dirtyBit);

    angle::Result syncState(const Context *context);
    bool hasAnyDirtyBit() const { return mDirtyBits.any(); }

    ComponentTypeMask getAttributesTypeMask() const { return mState.mVertexAttributesTypeMask; }
    AttributesMask getAttributesMask() const { return mState.mEnabledAttributesMask; }

    void onBindingChanged(const Context *context, int incr);
    bool hasTransformFeedbackBindingConflict(const gl::Context *context) const;

    ANGLE_INLINE angle::Result getIndexRange(const Context *context,
                                             DrawElementsType type,
                                             GLsizei indexCount,
                                             const void *indices,
                                             IndexRange *indexRangeOut) const
    {
        Buffer *elementArrayBuffer = mState.mElementArrayBuffer.get();
        if (elementArrayBuffer && mIndexRangeCache.get(type, indexCount, indices, indexRangeOut))
        {
            return angle::Result::Continue;
        }

        return getIndexRangeImpl(context, type, indexCount, indices, indexRangeOut);
    }

    void setBufferAccessValidationEnabled(bool enabled)
    {
        mBufferAccessValidationEnabled = enabled;
    }

  private:
    ~VertexArray() override;

    // This is a performance optimization for buffer binding. Allows element array buffer updates.
    friend class State;

    void setDirtyAttribBit(size_t attribIndex, DirtyAttribBitType dirtyAttribBit);
    void setDirtyBindingBit(size_t bindingIndex, DirtyBindingBitType dirtyBindingBit);

    DirtyBitType getDirtyBitFromIndex(bool contentsChanged, angle::SubjectIndex index) const;
    void setDependentDirtyBit(bool contentsChanged, angle::SubjectIndex index);

    // These are used to optimize draw call validation.
    void updateCachedBufferBindingSize(VertexBinding *binding);
    void updateCachedTransformFeedbackBindingValidation(size_t bindingIndex, const Buffer *buffer);
    void updateCachedMappedArrayBuffers(bool isMapped, const AttributesMask &boundAttributesMask);
    void updateCachedMappedArrayBuffersBinding(const VertexBinding &binding);

    angle::Result getIndexRangeImpl(const Context *context,
                                    DrawElementsType type,
                                    GLsizei indexCount,
                                    const void *indices,
                                    IndexRange *indexRangeOut) const;

    void setVertexAttribPointerImpl(const Context *context,
                                    ComponentType componentType,
                                    bool pureInteger,
                                    size_t attribIndex,
                                    Buffer *boundBuffer,
                                    GLint size,
                                    VertexAttribType type,
                                    bool normalized,
                                    GLsizei stride,
                                    const void *pointer);

    // These two functions return true if the state was dirty.
    bool setVertexAttribFormatImpl(VertexAttribute *attrib,
                                   GLint size,
                                   VertexAttribType type,
                                   bool normalized,
                                   bool pureInteger,
                                   GLuint relativeOffset);
    bool bindVertexBufferImpl(const Context *context,
                              size_t bindingIndex,
                              Buffer *boundBuffer,
                              GLintptr offset,
                              GLsizei stride);

    VertexArrayID mId;

    VertexArrayState mState;
    DirtyBits mDirtyBits;
    DirtyAttribBitsArray mDirtyAttribBits;
    DirtyBindingBitsArray mDirtyBindingBits;
    Optional<DirtyBits> mDirtyBitsGuard;

    rx::VertexArrayImpl *mVertexArray;

    std::vector<angle::ObserverBinding> mArrayBufferObserverBindings;

    AttributesMask mCachedTransformFeedbackConflictedBindingsMask;

    class IndexRangeCache final : angle::NonCopyable
    {
      public:
        IndexRangeCache();

        void invalidate() { mTypeKey = DrawElementsType::InvalidEnum; }

        bool get(DrawElementsType type,
                 GLsizei indexCount,
                 const void *indices,
                 IndexRange *indexRangeOut)
        {
            size_t offset = reinterpret_cast<uintptr_t>(indices);
            if (mTypeKey == type && mIndexCountKey == indexCount && mOffsetKey == offset)
            {
                *indexRangeOut = mPayload;
                return true;
            }

            return false;
        }

        void put(DrawElementsType type,
                 GLsizei indexCount,
                 size_t offset,
                 const IndexRange &indexRange);

      private:
        DrawElementsType mTypeKey;
        GLsizei mIndexCountKey;
        size_t mOffsetKey;
        IndexRange mPayload;
    };

    mutable IndexRangeCache mIndexRangeCache;
    bool mBufferAccessValidationEnabled;
};

}  // namespace gl

#endif  // LIBANGLE_VERTEXARRAY_H_
