//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// validationES2.h:
//  Inlined validation functions for OpenGL ES 2.0 entry points.

#ifndef LIBANGLE_VALIDATION_ES2_H_
#define LIBANGLE_VALIDATION_ES2_H_

#include "libANGLE/ErrorStrings.h"
#include "libANGLE/validationES.h"
#include "libANGLE/validationES2_autogen.h"

namespace gl
{
ANGLE_INLINE bool ValidateDrawArrays(Context *context,
                                     PrimitiveMode mode,
                                     GLint first,
                                     GLsizei count)
{
    return ValidateDrawArraysCommon(context, mode, first, count, 1);
}

ANGLE_INLINE bool ValidateUniform2f(Context *context, GLint location, GLfloat x, GLfloat y)
{
    return ValidateUniform(context, GL_FLOAT_VEC2, location, 1);
}

ANGLE_INLINE bool ValidateBindBuffer(Context *context, BufferBinding target, BufferID buffer)
{
    if (!context->isValidBufferBinding(target))
    {
        context->validationError(GL_INVALID_ENUM, err::kInvalidBufferTypes);
        return false;
    }

    if (!context->getState().isBindGeneratesResourceEnabled() &&
        !context->isBufferGenerated(buffer))
    {
        context->validationError(GL_INVALID_OPERATION, err::kObjectNotGenerated);
        return false;
    }

    return true;
}

ANGLE_INLINE bool ValidateDrawElements(Context *context,
                                       PrimitiveMode mode,
                                       GLsizei count,
                                       DrawElementsType type,
                                       const void *indices)
{
    return ValidateDrawElementsCommon(context, mode, count, type, indices, 1);
}

ANGLE_INLINE bool ValidateVertexAttribPointer(Context *context,
                                              GLuint index,
                                              GLint size,
                                              VertexAttribType type,
                                              GLboolean normalized,
                                              GLsizei stride,
                                              const void *ptr)
{
    if (!ValidateFloatVertexFormat(context, index, size, type))
    {
        return false;
    }

    if (stride < 0)
    {
        context->validationError(GL_INVALID_VALUE, err::kNegativeStride);
        return false;
    }

    if (context->getClientVersion() >= ES_3_1)
    {
        const Caps &caps = context->getCaps();
        if (stride > caps.maxVertexAttribStride)
        {
            context->validationError(GL_INVALID_VALUE, err::kExceedsMaxVertexAttribStride);
            return false;
        }

        if (index >= caps.maxVertexAttribBindings)
        {
            context->validationError(GL_INVALID_VALUE, err::kExceedsMaxVertexAttribBindings);
            return false;
        }
    }

    // [OpenGL ES 3.0.2] Section 2.8 page 24:
    // An INVALID_OPERATION error is generated when a non-zero vertex array object
    // is bound, zero is bound to the ARRAY_BUFFER buffer object binding point,
    // and the pointer argument is not NULL.
    bool nullBufferAllowed = context->getState().areClientArraysEnabled() &&
                             context->getState().getVertexArray()->id().value == 0;
    if (!nullBufferAllowed && context->getState().getTargetBuffer(BufferBinding::Array) == 0 &&
        ptr != nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, err::kClientDataInVertexArray);
        return false;
    }

    if (context->getExtensions().webglCompatibility)
    {
        // WebGL 1.0 [Section 6.14] Fixed point support
        // The WebGL API does not support the GL_FIXED data type.
        if (type == VertexAttribType::Fixed)
        {
            context->validationError(GL_INVALID_ENUM, err::kFixedNotInWebGL);
            return false;
        }

        if (!ValidateWebGLVertexAttribPointer(context, type, normalized, stride, ptr, false))
        {
            return false;
        }
    }

    return true;
}

void RecordBindTextureTypeError(Context *context, TextureType target);

ANGLE_INLINE bool ValidateBindTexture(Context *context, TextureType target, TextureID texture)
{
    if (!context->getStateCache().isValidBindTextureType(target))
    {
        RecordBindTextureTypeError(context, target);
        return false;
    }

    if (texture.value == 0)
    {
        return true;
    }

    Texture *textureObject = context->getTexture(texture);
    if (textureObject && textureObject->getType() != target)
    {
        context->validationError(GL_INVALID_OPERATION, err::kTextureTargetMismatch);
        return false;
    }

    if (!context->getState().isBindGeneratesResourceEnabled() &&
        !context->isTextureGenerated(texture))
    {
        context->validationError(GL_INVALID_OPERATION, err::kObjectNotGenerated);
        return false;
    }

    return true;
}

// Validation of all Tex[Sub]Image2D parameters except TextureTarget.
bool ValidateES2TexImageParametersBase(Context *context,
                                       TextureTarget target,
                                       GLint level,
                                       GLenum internalformat,
                                       bool isCompressed,
                                       bool isSubImage,
                                       GLint xoffset,
                                       GLint yoffset,
                                       GLsizei width,
                                       GLsizei height,
                                       GLint border,
                                       GLenum format,
                                       GLenum type,
                                       GLsizei imageSize,
                                       const void *pixels);

}  // namespace gl

#endif  // LIBANGLE_VALIDATION_ES2_H_
