//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Implementation of the state class for mananging GLES 3 Vertex Array Objects.
//

#include "libANGLE/VertexArray.h"

#include "common/utilities.h"
#include "libANGLE/Buffer.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/BufferImpl.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/VertexArrayImpl.h"

namespace gl
{
namespace
{
bool IsElementArrayBufferSubjectIndex(angle::SubjectIndex subjectIndex)
{
    return (subjectIndex == MAX_VERTEX_ATTRIBS);
}

constexpr angle::SubjectIndex kElementArrayBufferIndex = MAX_VERTEX_ATTRIBS;
}  // namespace

// VertexArrayState implementation.
VertexArrayState::VertexArrayState(VertexArray *vertexArray,
                                   size_t maxAttribs,
                                   size_t maxAttribBindings)
    : mElementArrayBuffer(vertexArray, kElementArrayBufferIndex)
{
    ASSERT(maxAttribs <= maxAttribBindings);

    for (size_t i = 0; i < maxAttribs; i++)
    {
        mVertexAttributes.emplace_back(static_cast<GLuint>(i));
        mVertexBindings.emplace_back(static_cast<GLuint>(i));
    }

    // Initially all attributes start as "client" with no buffer bound.
    mClientMemoryAttribsMask.set();
}

VertexArrayState::~VertexArrayState() {}

bool VertexArrayState::hasEnabledNullPointerClientArray() const
{
    return (mNullPointerClientMemoryAttribsMask & mEnabledAttributesMask).any();
}

AttributesMask VertexArrayState::getBindingToAttributesMask(GLuint bindingIndex) const
{
    ASSERT(bindingIndex < MAX_VERTEX_ATTRIB_BINDINGS);
    return mVertexBindings[bindingIndex].getBoundAttributesMask();
}

// Set an attribute using a new binding.
void VertexArrayState::setAttribBinding(const Context *context,
                                        size_t attribIndex,
                                        GLuint newBindingIndex)
{
    ASSERT(attribIndex < MAX_VERTEX_ATTRIBS && newBindingIndex < MAX_VERTEX_ATTRIB_BINDINGS);

    VertexAttribute &attrib = mVertexAttributes[attribIndex];

    // Update the binding-attribute map.
    const GLuint oldBindingIndex = attrib.bindingIndex;
    ASSERT(oldBindingIndex != newBindingIndex);

    VertexBinding &oldBinding = mVertexBindings[oldBindingIndex];
    VertexBinding &newBinding = mVertexBindings[newBindingIndex];

    ASSERT(oldBinding.getBoundAttributesMask().test(attribIndex) &&
           !newBinding.getBoundAttributesMask().test(attribIndex));

    oldBinding.resetBoundAttribute(attribIndex);
    newBinding.setBoundAttribute(attribIndex);

    // Set the attribute using the new binding.
    attrib.bindingIndex = newBindingIndex;

    if (context->isBufferAccessValidationEnabled())
    {
        attrib.updateCachedElementLimit(newBinding);
    }

    bool isMapped = newBinding.getBuffer().get() && newBinding.getBuffer()->isMapped();
    mCachedMappedArrayBuffers.set(attribIndex, isMapped);
    mCachedEnabledMappedArrayBuffers.set(attribIndex, isMapped && attrib.enabled);
}

// VertexArray implementation.
VertexArray::VertexArray(rx::GLImplFactory *factory,
                         VertexArrayID id,
                         size_t maxAttribs,
                         size_t maxAttribBindings)
    : mId(id),
      mState(this, maxAttribs, maxAttribBindings),
      mVertexArray(factory->createVertexArray(mState)),
      mBufferAccessValidationEnabled(false)
{
    for (size_t attribIndex = 0; attribIndex < maxAttribBindings; ++attribIndex)
    {
        mArrayBufferObserverBindings.emplace_back(this, attribIndex);
    }
}

void VertexArray::onDestroy(const Context *context)
{
    bool isBound = context->isCurrentVertexArray(this);
    for (VertexBinding &binding : mState.mVertexBindings)
    {
        if (isBound)
        {
            if (binding.getBuffer().get())
                binding.getBuffer()->onNonTFBindingChanged(-1);
        }
        binding.setBuffer(context, nullptr);
    }
    if (isBound && mState.mElementArrayBuffer.get())
        mState.mElementArrayBuffer->onNonTFBindingChanged(-1);
    mState.mElementArrayBuffer.bind(context, nullptr);
    mVertexArray->destroy(context);
    SafeDelete(mVertexArray);
    delete this;
}

VertexArray::~VertexArray()
{
    ASSERT(!mVertexArray);
}

void VertexArray::setLabel(const Context *context, const std::string &label)
{
    mState.mLabel = label;
}

const std::string &VertexArray::getLabel() const
{
    return mState.mLabel;
}

bool VertexArray::detachBuffer(const Context *context, BufferID bufferID)
{
    bool isBound           = context->isCurrentVertexArray(this);
    bool anyBufferDetached = false;
    for (uint32_t bindingIndex = 0; bindingIndex < gl::MAX_VERTEX_ATTRIB_BINDINGS; ++bindingIndex)
    {
        VertexBinding &binding = mState.mVertexBindings[bindingIndex];
        if (binding.getBuffer().id() == bufferID)
        {
            if (isBound)
            {
                if (binding.getBuffer().get())
                    binding.getBuffer()->onNonTFBindingChanged(-1);
            }
            binding.setBuffer(context, nullptr);
            mArrayBufferObserverBindings[bindingIndex].reset();

            if (context->getClientVersion() >= ES_3_1)
            {
                setDirtyBindingBit(bindingIndex, DIRTY_BINDING_BUFFER);
            }
            else
            {
                static_assert(gl::MAX_VERTEX_ATTRIB_BINDINGS < 8 * sizeof(uint32_t),
                              "Not enough bits in bindingIndex");
                // The redundant uint32_t cast here is required to avoid a warning on MSVC.
                ASSERT(binding.getBoundAttributesMask() ==
                       AttributesMask(static_cast<uint32_t>(1 << bindingIndex)));
                setDirtyAttribBit(bindingIndex, DIRTY_ATTRIB_POINTER);
            }

            anyBufferDetached = true;
            mState.mClientMemoryAttribsMask |= binding.getBoundAttributesMask();
        }
    }

    if (mState.mElementArrayBuffer.get() && mState.mElementArrayBuffer->id() == bufferID)
    {
        if (isBound && mState.mElementArrayBuffer.get())
            mState.mElementArrayBuffer->onNonTFBindingChanged(-1);
        mState.mElementArrayBuffer.bind(context, nullptr);
        mDirtyBits.set(DIRTY_BIT_ELEMENT_ARRAY_BUFFER);
        anyBufferDetached = true;
    }

    return anyBufferDetached;
}

const VertexAttribute &VertexArray::getVertexAttribute(size_t attribIndex) const
{
    ASSERT(attribIndex < getMaxAttribs());
    return mState.mVertexAttributes[attribIndex];
}

const VertexBinding &VertexArray::getVertexBinding(size_t bindingIndex) const
{
    ASSERT(bindingIndex < getMaxBindings());
    return mState.mVertexBindings[bindingIndex];
}

size_t VertexArray::GetVertexIndexFromDirtyBit(size_t dirtyBit)
{
    static_assert(gl::MAX_VERTEX_ATTRIBS == gl::MAX_VERTEX_ATTRIB_BINDINGS,
                  "The stride of vertex attributes should equal to that of vertex bindings.");
    ASSERT(dirtyBit > DIRTY_BIT_ELEMENT_ARRAY_BUFFER);
    return (dirtyBit - DIRTY_BIT_ATTRIB_0) % gl::MAX_VERTEX_ATTRIBS;
}

ANGLE_INLINE void VertexArray::setDirtyAttribBit(size_t attribIndex,
                                                 DirtyAttribBitType dirtyAttribBit)
{
    mDirtyBits.set(DIRTY_BIT_ATTRIB_0 + attribIndex);
    mDirtyAttribBits[attribIndex].set(dirtyAttribBit);
}

ANGLE_INLINE void VertexArray::setDirtyBindingBit(size_t bindingIndex,
                                                  DirtyBindingBitType dirtyBindingBit)
{
    mDirtyBits.set(DIRTY_BIT_BINDING_0 + bindingIndex);
    mDirtyBindingBits[bindingIndex].set(dirtyBindingBit);
}

ANGLE_INLINE void VertexArray::updateCachedBufferBindingSize(VertexBinding *binding)
{
    if (!mBufferAccessValidationEnabled)
        return;

    for (size_t boundAttribute : binding->getBoundAttributesMask())
    {
        mState.mVertexAttributes[boundAttribute].updateCachedElementLimit(*binding);
    }
}

ANGLE_INLINE void VertexArray::updateCachedMappedArrayBuffers(
    bool isMapped,
    const AttributesMask &boundAttributesMask)
{
    if (isMapped)
    {
        mState.mCachedMappedArrayBuffers |= boundAttributesMask;
    }
    else
    {
        mState.mCachedMappedArrayBuffers &= ~boundAttributesMask;
    }

    mState.mCachedEnabledMappedArrayBuffers =
        mState.mCachedMappedArrayBuffers & mState.mEnabledAttributesMask;
}

ANGLE_INLINE void VertexArray::updateCachedMappedArrayBuffersBinding(const VertexBinding &binding)
{
    const Buffer *buffer = binding.getBuffer().get();
    return updateCachedMappedArrayBuffers(buffer && buffer->isMapped(),
                                          binding.getBoundAttributesMask());
}

ANGLE_INLINE void VertexArray::updateCachedTransformFeedbackBindingValidation(size_t bindingIndex,
                                                                              const Buffer *buffer)
{
    const bool hasConflict = buffer && buffer->isBoundForTransformFeedbackAndOtherUse();
    mCachedTransformFeedbackConflictedBindingsMask.set(bindingIndex, hasConflict);
}

bool VertexArray::bindVertexBufferImpl(const Context *context,
                                       size_t bindingIndex,
                                       Buffer *boundBuffer,
                                       GLintptr offset,
                                       GLsizei stride)
{
    ASSERT(bindingIndex < getMaxBindings());
    ASSERT(context->isCurrentVertexArray(this));

    VertexBinding *binding = &mState.mVertexBindings[bindingIndex];

    Buffer *oldBuffer = binding->getBuffer().get();

    const bool sameBuffer = oldBuffer == boundBuffer;
    const bool sameStride = static_cast<GLuint>(stride) == binding->getStride();
    const bool sameOffset = offset == binding->getOffset();

    if (sameBuffer && sameStride && sameOffset)
    {
        return false;
    }

    angle::ObserverBinding *observer = &mArrayBufferObserverBindings[bindingIndex];
    observer->assignSubject(boundBuffer);

    // Several nullptr checks are combined here for optimization purposes.
    if (oldBuffer)
    {
        oldBuffer->onNonTFBindingChanged(-1);
        oldBuffer->removeObserver(observer);
        oldBuffer->release(context);
    }

    binding->assignBuffer(boundBuffer);
    binding->setOffset(offset);
    binding->setStride(stride);
    updateCachedBufferBindingSize(binding);

    // Update client memory attribute pointers. Affects all bound attributes.
    if (boundBuffer)
    {
        boundBuffer->addRef();
        boundBuffer->onNonTFBindingChanged(1);
        boundBuffer->addObserver(observer);
        mCachedTransformFeedbackConflictedBindingsMask.set(
            bindingIndex, boundBuffer->isBoundForTransformFeedbackAndOtherUse());
        mState.mClientMemoryAttribsMask &= ~binding->getBoundAttributesMask();
        updateCachedMappedArrayBuffers((boundBuffer->isMapped() == GL_TRUE),
                                       binding->getBoundAttributesMask());
    }
    else
    {
        mCachedTransformFeedbackConflictedBindingsMask.set(bindingIndex, false);
        mState.mClientMemoryAttribsMask |= binding->getBoundAttributesMask();
        updateCachedMappedArrayBuffers(false, binding->getBoundAttributesMask());
    }

    return true;
}

void VertexArray::bindVertexBuffer(const Context *context,
                                   size_t bindingIndex,
                                   Buffer *boundBuffer,
                                   GLintptr offset,
                                   GLsizei stride)
{
    if (bindVertexBufferImpl(context, bindingIndex, boundBuffer, offset, stride))
    {
        setDirtyBindingBit(bindingIndex, DIRTY_BINDING_BUFFER);
    }
}

void VertexArray::setVertexAttribBinding(const Context *context,
                                         size_t attribIndex,
                                         GLuint bindingIndex)
{
    ASSERT(attribIndex < getMaxAttribs() && bindingIndex < getMaxBindings());

    if (mState.mVertexAttributes[attribIndex].bindingIndex != bindingIndex)
    {
        // In ES 3.0 contexts, the binding cannot change, hence the code below is unreachable.
        ASSERT(context->getClientVersion() >= ES_3_1);

        mState.setAttribBinding(context, attribIndex, bindingIndex);

        setDirtyAttribBit(attribIndex, DIRTY_ATTRIB_BINDING);

        // Update client attribs mask.
        bool hasBuffer = mState.mVertexBindings[bindingIndex].getBuffer().get() != nullptr;
        mState.mClientMemoryAttribsMask.set(attribIndex, !hasBuffer);
    }
}

void VertexArray::setVertexBindingDivisor(size_t bindingIndex, GLuint divisor)
{
    ASSERT(bindingIndex < getMaxBindings());

    VertexBinding &binding = mState.mVertexBindings[bindingIndex];

    binding.setDivisor(divisor);
    setDirtyBindingBit(bindingIndex, DIRTY_BINDING_DIVISOR);

    // Trigger updates in all bound attributes.
    for (size_t attribIndex : binding.getBoundAttributesMask())
    {
        mState.mVertexAttributes[attribIndex].updateCachedElementLimit(binding);
    }
}

ANGLE_INLINE bool VertexArray::setVertexAttribFormatImpl(VertexAttribute *attrib,
                                                         GLint size,
                                                         VertexAttribType type,
                                                         bool normalized,
                                                         bool pureInteger,
                                                         GLuint relativeOffset)
{
    angle::FormatID formatID = gl::GetVertexFormatID(type, normalized, size, pureInteger);

    if (formatID != attrib->format->id || attrib->relativeOffset != relativeOffset)
    {
        attrib->relativeOffset = relativeOffset;
        attrib->format         = &angle::Format::Get(formatID);
        return true;
    }

    return false;
}

void VertexArray::setVertexAttribFormat(size_t attribIndex,
                                        GLint size,
                                        VertexAttribType type,
                                        bool normalized,
                                        bool pureInteger,
                                        GLuint relativeOffset)
{
    VertexAttribute &attrib = mState.mVertexAttributes[attribIndex];

    ComponentType componentType = GetVertexAttributeComponentType(pureInteger, type);
    SetComponentTypeMask(componentType, attribIndex, &mState.mVertexAttributesTypeMask);

    if (setVertexAttribFormatImpl(&attrib, size, type, normalized, pureInteger, relativeOffset))
    {
        setDirtyAttribBit(attribIndex, DIRTY_ATTRIB_FORMAT);
    }

    attrib.updateCachedElementLimit(mState.mVertexBindings[attrib.bindingIndex]);
}

void VertexArray::setVertexAttribDivisor(const Context *context, size_t attribIndex, GLuint divisor)
{
    ASSERT(attribIndex < getMaxAttribs());

    setVertexAttribBinding(context, attribIndex, static_cast<GLuint>(attribIndex));
    setVertexBindingDivisor(attribIndex, divisor);
}

void VertexArray::enableAttribute(size_t attribIndex, bool enabledState)
{
    ASSERT(attribIndex < getMaxAttribs());

    VertexAttribute &attrib = mState.mVertexAttributes[attribIndex];

    if (mState.mEnabledAttributesMask.test(attribIndex) == enabledState)
    {
        return;
    }

    attrib.enabled = enabledState;

    setDirtyAttribBit(attribIndex, DIRTY_ATTRIB_ENABLED);

    // Update state cache
    mState.mEnabledAttributesMask.set(attribIndex, enabledState);
    mState.mCachedEnabledMappedArrayBuffers =
        mState.mCachedMappedArrayBuffers & mState.mEnabledAttributesMask;
}

ANGLE_INLINE void VertexArray::setVertexAttribPointerImpl(const Context *context,
                                                          ComponentType componentType,
                                                          bool pureInteger,
                                                          size_t attribIndex,
                                                          Buffer *boundBuffer,
                                                          GLint size,
                                                          VertexAttribType type,
                                                          bool normalized,
                                                          GLsizei stride,
                                                          const void *pointer)
{
    ASSERT(attribIndex < getMaxAttribs());

    VertexAttribute &attrib = mState.mVertexAttributes[attribIndex];

    SetComponentTypeMask(componentType, attribIndex, &mState.mVertexAttributesTypeMask);

    bool attribDirty = setVertexAttribFormatImpl(&attrib, size, type, normalized, pureInteger, 0);

    if (attrib.bindingIndex != attribIndex)
    {
        setVertexAttribBinding(context, attribIndex, static_cast<GLuint>(attribIndex));
    }

    GLsizei effectiveStride =
        stride != 0 ? stride : static_cast<GLsizei>(ComputeVertexAttributeTypeSize(attrib));

    if (attrib.vertexAttribArrayStride != static_cast<GLuint>(stride))
    {
        attribDirty = true;
    }
    attrib.vertexAttribArrayStride = stride;

    // If we switch from an array buffer to a client pointer(or vice-versa), we set the whole
    // attribute dirty. This notifies the Vulkan back-end to update all its caches.
    const VertexBinding &binding = mState.mVertexBindings[attribIndex];
    if ((boundBuffer == nullptr) != (binding.getBuffer().get() == nullptr))
    {
        attribDirty = true;
    }

    // Change of attrib.pointer is not part of attribDirty. Pointer is actually the buffer offset
    // which is handled within bindVertexBufferImpl and reflected in bufferDirty.
    attrib.pointer  = pointer;
    GLintptr offset = boundBuffer ? reinterpret_cast<GLintptr>(pointer) : 0;
    const bool bufferDirty =
        bindVertexBufferImpl(context, attribIndex, boundBuffer, offset, effectiveStride);

    if (attribDirty)
    {
        setDirtyAttribBit(attribIndex, DIRTY_ATTRIB_POINTER);
    }
    else if (bufferDirty)
    {
        setDirtyAttribBit(attribIndex, DIRTY_ATTRIB_POINTER_BUFFER);
    }

    mState.mNullPointerClientMemoryAttribsMask.set(attribIndex,
                                                   boundBuffer == nullptr && pointer == nullptr);
}

void VertexArray::setVertexAttribPointer(const Context *context,
                                         size_t attribIndex,
                                         gl::Buffer *boundBuffer,
                                         GLint size,
                                         VertexAttribType type,
                                         bool normalized,
                                         GLsizei stride,
                                         const void *pointer)
{
    setVertexAttribPointerImpl(context, ComponentType::Float, false, attribIndex, boundBuffer, size,
                               type, normalized, stride, pointer);
}

void VertexArray::setVertexAttribIPointer(const Context *context,
                                          size_t attribIndex,
                                          gl::Buffer *boundBuffer,
                                          GLint size,
                                          VertexAttribType type,
                                          GLsizei stride,
                                          const void *pointer)
{
    ComponentType componentType = GetVertexAttributeComponentType(true, type);
    setVertexAttribPointerImpl(context, componentType, true, attribIndex, boundBuffer, size, type,
                               false, stride, pointer);
}

angle::Result VertexArray::syncState(const Context *context)
{
    if (mDirtyBits.any())
    {
        mDirtyBitsGuard = mDirtyBits;
        ANGLE_TRY(
            mVertexArray->syncState(context, mDirtyBits, &mDirtyAttribBits, &mDirtyBindingBits));
        mDirtyBits.reset();
        mDirtyBitsGuard.reset();

        // The dirty bits should be reset in the back-end. To simplify ASSERTs only check attrib 0.
        ASSERT(mDirtyAttribBits[0].none());
        ASSERT(mDirtyBindingBits[0].none());
    }
    return angle::Result::Continue;
}

void VertexArray::onBindingChanged(const Context *context, int incr)
{
    if (mState.mElementArrayBuffer.get())
        mState.mElementArrayBuffer->onNonTFBindingChanged(incr);
    for (auto &binding : mState.mVertexBindings)
    {
        binding.onContainerBindingChanged(context, incr);
    }
}

VertexArray::DirtyBitType VertexArray::getDirtyBitFromIndex(bool contentsChanged,
                                                            angle::SubjectIndex index) const
{
    if (IsElementArrayBufferSubjectIndex(index))
    {
        mIndexRangeCache.invalidate();
        return contentsChanged ? DIRTY_BIT_ELEMENT_ARRAY_BUFFER_DATA
                               : DIRTY_BIT_ELEMENT_ARRAY_BUFFER;
    }
    else
    {
        // Note: this currently just gets the top-level dirty bit.
        ASSERT(index < mArrayBufferObserverBindings.size());
        return static_cast<DirtyBitType>(
            (contentsChanged ? DIRTY_BIT_BUFFER_DATA_0 : DIRTY_BIT_BINDING_0) + index);
    }
}

void VertexArray::onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message)
{
    switch (message)
    {
        case angle::SubjectMessage::ContentsChanged:
            setDependentDirtyBit(true, index);
            break;

        case angle::SubjectMessage::SubjectChanged:
            if (!IsElementArrayBufferSubjectIndex(index))
            {
                updateCachedBufferBindingSize(&mState.mVertexBindings[index]);
            }
            setDependentDirtyBit(false, index);
            break;

        case angle::SubjectMessage::BindingChanged:
            if (!IsElementArrayBufferSubjectIndex(index))
            {
                const Buffer *buffer = mState.mVertexBindings[index].getBuffer().get();
                updateCachedTransformFeedbackBindingValidation(index, buffer);
            }
            break;

        case angle::SubjectMessage::SubjectMapped:
            if (!IsElementArrayBufferSubjectIndex(index))
            {
                updateCachedMappedArrayBuffersBinding(mState.mVertexBindings[index]);
            }
            onStateChange(angle::SubjectMessage::SubjectMapped);
            break;

        case angle::SubjectMessage::SubjectUnmapped:
            setDependentDirtyBit(true, index);

            if (!IsElementArrayBufferSubjectIndex(index))
            {
                updateCachedMappedArrayBuffersBinding(mState.mVertexBindings[index]);
            }
            onStateChange(angle::SubjectMessage::SubjectUnmapped);
            break;

        default:
            UNREACHABLE();
            break;
    }
}

void VertexArray::setDependentDirtyBit(bool contentsChanged, angle::SubjectIndex index)
{
    DirtyBitType dirtyBit = getDirtyBitFromIndex(contentsChanged, index);
    ASSERT(!mDirtyBitsGuard.valid() || mDirtyBitsGuard.value().test(dirtyBit));
    mDirtyBits.set(dirtyBit);
    onStateChange(angle::SubjectMessage::ContentsChanged);
}

bool VertexArray::hasTransformFeedbackBindingConflict(const gl::Context *context) const
{
    // Fast check first.
    if (!mCachedTransformFeedbackConflictedBindingsMask.any())
    {
        return false;
    }

    const AttributesMask &activeAttribues = context->getStateCache().getActiveBufferedAttribsMask();

    // Slow check. We must ensure that the conflicting attributes are enabled/active.
    for (size_t attribIndex : activeAttribues)
    {
        const VertexAttribute &attrib = mState.mVertexAttributes[attribIndex];
        if (mCachedTransformFeedbackConflictedBindingsMask[attrib.bindingIndex])
        {
            return true;
        }
    }

    return false;
}

angle::Result VertexArray::getIndexRangeImpl(const Context *context,
                                             DrawElementsType type,
                                             GLsizei indexCount,
                                             const void *indices,
                                             IndexRange *indexRangeOut) const
{
    Buffer *elementArrayBuffer = mState.mElementArrayBuffer.get();
    if (!elementArrayBuffer)
    {
        *indexRangeOut = ComputeIndexRange(type, indices, indexCount,
                                           context->getState().isPrimitiveRestartEnabled());
        return angle::Result::Continue;
    }

    size_t offset = reinterpret_cast<uintptr_t>(indices);
    ANGLE_TRY(elementArrayBuffer->getIndexRange(context, type, offset, indexCount,
                                                context->getState().isPrimitiveRestartEnabled(),
                                                indexRangeOut));

    mIndexRangeCache.put(type, indexCount, offset, *indexRangeOut);
    return angle::Result::Continue;
}

VertexArray::IndexRangeCache::IndexRangeCache() = default;

void VertexArray::IndexRangeCache::put(DrawElementsType type,
                                       GLsizei indexCount,
                                       size_t offset,
                                       const IndexRange &indexRange)
{
    ASSERT(type != DrawElementsType::InvalidEnum);

    mTypeKey       = type;
    mIndexCountKey = indexCount;
    mOffsetKey     = offset;
    mPayload       = indexRange;
}
}  // namespace gl
