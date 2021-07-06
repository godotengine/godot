//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef LIBANGLE_TRANSFORM_FEEDBACK_H_
#define LIBANGLE_TRANSFORM_FEEDBACK_H_

#include "libANGLE/RefCountObject.h"

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Debug.h"

#include "angle_gl.h"

namespace rx
{
class GLImplFactory;
class TransformFeedbackImpl;
class TransformFeedbackGL;
}  // namespace rx

namespace gl
{
class Buffer;
struct Caps;
class Context;
class Program;

class TransformFeedbackState final : angle::NonCopyable
{
  public:
    TransformFeedbackState(size_t maxIndexedBuffers);
    ~TransformFeedbackState();

    const OffsetBindingPointer<Buffer> &getIndexedBuffer(size_t idx) const;
    const std::vector<OffsetBindingPointer<Buffer>> &getIndexedBuffers() const;
    const Program *getBoundProgram() const { return mProgram; }
    GLsizeiptr getVerticesDrawn() const { return mVerticesDrawn; }
    GLsizeiptr getPrimitivesDrawn() const;

  private:
    friend class TransformFeedback;

    std::string mLabel;

    bool mActive;
    PrimitiveMode mPrimitiveMode;
    bool mPaused;
    GLsizeiptr mVerticesDrawn;
    GLsizeiptr mVertexCapacity;

    Program *mProgram;

    std::vector<OffsetBindingPointer<Buffer>> mIndexedBuffers;
};

class TransformFeedback final : public RefCountObject<TransformFeedbackID>, public LabeledObject
{
  public:
    TransformFeedback(rx::GLImplFactory *implFactory, TransformFeedbackID id, const Caps &caps);
    ~TransformFeedback() override;
    void onDestroy(const Context *context) override;

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    angle::Result begin(const Context *context, PrimitiveMode primitiveMode, Program *program);
    angle::Result end(const Context *context);
    angle::Result pause(const Context *context);
    angle::Result resume(const Context *context);

    bool isActive() const { return mState.mActive; }

    bool isPaused() const;
    PrimitiveMode getPrimitiveMode() const;
    // Validates that the vertices produced by a draw call will fit in the bound transform feedback
    // buffers.
    bool checkBufferSpaceForDraw(GLsizei count, GLsizei primcount) const;
    // This must be called after each draw call when transform feedback is enabled to keep track of
    // how many vertices have been written to the buffers. This information is needed by
    // checkBufferSpaceForDraw because each draw call appends vertices to the buffers starting just
    // after the last vertex of the previous draw call.
    void onVerticesDrawn(const Context *context, GLsizei count, GLsizei primcount);

    bool hasBoundProgram(ShaderProgramID program) const;

    angle::Result bindIndexedBuffer(const Context *context,
                                    size_t index,
                                    Buffer *buffer,
                                    size_t offset,
                                    size_t size);
    const OffsetBindingPointer<Buffer> &getIndexedBuffer(size_t index) const;
    size_t getIndexedBufferCount() const;

    GLsizeiptr getVerticesDrawn() const { return mState.getVerticesDrawn(); }
    GLsizeiptr getPrimitivesDrawn() const { return mState.getPrimitivesDrawn(); }

    // Returns true if any buffer bound to this object is also bound to another target.
    bool buffersBoundForOtherUse() const;

    angle::Result detachBuffer(const Context *context, BufferID bufferID);

    rx::TransformFeedbackImpl *getImplementation() const;

    void onBindingChanged(const Context *context, bool bound);

  private:
    void bindProgram(const Context *context, Program *program);

    TransformFeedbackState mState;
    rx::TransformFeedbackImpl *mImplementation;
};

}  // namespace gl

#endif  // LIBANGLE_TRANSFORM_FEEDBACK_H_
