//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// VertexArrayGL.h: Defines the class interface for VertexArrayGL.

#ifndef LIBANGLE_RENDERER_GL_VERTEXARRAYGL_H_
#define LIBANGLE_RENDERER_GL_VERTEXARRAYGL_H_

#include "libANGLE/renderer/VertexArrayImpl.h"

#include "common/mathutil.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/gl/ContextGL.h"

namespace rx
{

class FunctionsGL;
class StateManagerGL;

class VertexArrayGL : public VertexArrayImpl
{
  public:
    VertexArrayGL(const gl::VertexArrayState &data,
                  const FunctionsGL *functions,
                  StateManagerGL *stateManager);
    ~VertexArrayGL() override;

    void destroy(const gl::Context *context) override;

    angle::Result syncClientSideData(const gl::Context *context,
                                     const gl::AttributesMask &activeAttributesMask,
                                     GLint first,
                                     GLsizei count,
                                     GLsizei instanceCount) const;
    angle::Result syncDrawElementsState(const gl::Context *context,
                                        const gl::AttributesMask &activeAttributesMask,
                                        GLsizei count,
                                        gl::DrawElementsType type,
                                        const void *indices,
                                        GLsizei instanceCount,
                                        bool primitiveRestartEnabled,
                                        const void **outIndices) const;

    GLuint getVertexArrayID() const;
    GLuint getAppliedElementArrayBufferID() const;

    angle::Result syncState(const gl::Context *context,
                            const gl::VertexArray::DirtyBits &dirtyBits,
                            gl::VertexArray::DirtyAttribBitsArray *attribBits,
                            gl::VertexArray::DirtyBindingBitsArray *bindingBits) override;

    void applyNumViewsToDivisor(int numViews);
    void applyActiveAttribLocationsMask(const gl::AttributesMask &activeMask);

    void validateState() const;

  private:
    angle::Result syncDrawState(const gl::Context *context,
                                const gl::AttributesMask &activeAttributesMask,
                                GLint first,
                                GLsizei count,
                                gl::DrawElementsType type,
                                const void *indices,
                                GLsizei instanceCount,
                                bool primitiveRestartEnabled,
                                const void **outIndices) const;

    // Apply index data, only sets outIndexRange if attributesNeedStreaming is true
    angle::Result syncIndexData(const gl::Context *context,
                                GLsizei count,
                                gl::DrawElementsType type,
                                const void *indices,
                                bool primitiveRestartEnabled,
                                bool attributesNeedStreaming,
                                gl::IndexRange *outIndexRange,
                                const void **outIndices) const;

    // Returns the amount of space needed to stream all attributes that need streaming
    // and the data size of the largest attribute
    void computeStreamingAttributeSizes(const gl::AttributesMask &attribsToStream,
                                        GLsizei instanceCount,
                                        const gl::IndexRange &indexRange,
                                        size_t *outStreamingDataSize,
                                        size_t *outMaxAttributeDataSize) const;

    // Stream attributes that have client data
    angle::Result streamAttributes(const gl::Context *context,
                                   const gl::AttributesMask &attribsToStream,
                                   GLsizei instanceCount,
                                   const gl::IndexRange &indexRange) const;
    void syncDirtyAttrib(const gl::Context *context,
                         size_t attribIndex,
                         const gl::VertexArray::DirtyAttribBits &dirtyAttribBits);
    void syncDirtyBinding(const gl::Context *context,
                          size_t bindingIndex,
                          const gl::VertexArray::DirtyBindingBits &dirtyBindingBits);

    void updateAttribEnabled(size_t attribIndex);
    void updateAttribPointer(const gl::Context *context, size_t attribIndex);

    bool supportVertexAttribBinding() const;

    void updateAttribFormat(size_t attribIndex);
    void updateAttribBinding(size_t attribIndex);
    void updateBindingBuffer(const gl::Context *context, size_t bindingIndex);
    void updateBindingDivisor(size_t bindingIndex);

    void updateElementArrayBufferBinding(const gl::Context *context) const;

    void callVertexAttribPointer(GLuint attribIndex,
                                 const gl::VertexAttribute &attrib,
                                 GLsizei stride,
                                 GLintptr offset) const;

    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    GLuint mVertexArrayID;
    int mAppliedNumViews;

    // Remember the program's active attrib location mask so that attributes can be enabled/disabled
    // based on whether they are active in the program
    gl::AttributesMask mProgramActiveAttribLocationsMask;

    mutable gl::BindingPointer<gl::Buffer> mAppliedElementArrayBuffer;

    mutable std::vector<gl::VertexAttribute> mAppliedAttributes;
    mutable std::vector<gl::VertexBinding> mAppliedBindings;

    mutable size_t mStreamingElementArrayBufferSize;
    mutable GLuint mStreamingElementArrayBuffer;

    mutable size_t mStreamingArrayBufferSize;
    mutable GLuint mStreamingArrayBuffer;
};

ANGLE_INLINE angle::Result VertexArrayGL::syncDrawElementsState(
    const gl::Context *context,
    const gl::AttributesMask &activeAttributesMask,
    GLsizei count,
    gl::DrawElementsType type,
    const void *indices,
    GLsizei instanceCount,
    bool primitiveRestartEnabled,
    const void **outIndices) const
{
    return syncDrawState(context, activeAttributesMask, 0, count, type, indices, instanceCount,
                         primitiveRestartEnabled, outIndices);
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_VERTEXARRAYGL_H_
