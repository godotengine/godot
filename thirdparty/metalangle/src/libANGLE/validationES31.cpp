//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationES31.cpp: Validation functions for OpenGL ES 3.1 entry point parameters

#include "libANGLE/validationES31_autogen.h"

#include "libANGLE/Context.h"
#include "libANGLE/ErrorStrings.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/VertexArray.h"
#include "libANGLE/validationES.h"
#include "libANGLE/validationES2_autogen.h"
#include "libANGLE/validationES3_autogen.h"

#include "common/utilities.h"

using namespace angle;

namespace gl
{
using namespace err;

namespace
{

bool ValidateNamedProgramInterface(GLenum programInterface)
{
    switch (programInterface)
    {
        case GL_UNIFORM:
        case GL_UNIFORM_BLOCK:
        case GL_PROGRAM_INPUT:
        case GL_PROGRAM_OUTPUT:
        case GL_TRANSFORM_FEEDBACK_VARYING:
        case GL_BUFFER_VARIABLE:
        case GL_SHADER_STORAGE_BLOCK:
            return true;
        default:
            return false;
    }
}

bool ValidateLocationProgramInterface(GLenum programInterface)
{
    switch (programInterface)
    {
        case GL_UNIFORM:
        case GL_PROGRAM_INPUT:
        case GL_PROGRAM_OUTPUT:
            return true;
        default:
            return false;
    }
}

bool ValidateProgramInterface(GLenum programInterface)
{
    return (programInterface == GL_ATOMIC_COUNTER_BUFFER ||
            ValidateNamedProgramInterface(programInterface));
}

bool ValidateProgramResourceProperty(const Context *context, GLenum prop)
{
    ASSERT(context);
    switch (prop)
    {
        case GL_ACTIVE_VARIABLES:
        case GL_BUFFER_BINDING:
        case GL_NUM_ACTIVE_VARIABLES:

        case GL_ARRAY_SIZE:

        case GL_ARRAY_STRIDE:
        case GL_BLOCK_INDEX:
        case GL_IS_ROW_MAJOR:
        case GL_MATRIX_STRIDE:

        case GL_ATOMIC_COUNTER_BUFFER_INDEX:

        case GL_BUFFER_DATA_SIZE:

        case GL_LOCATION:

        case GL_NAME_LENGTH:

        case GL_OFFSET:

        case GL_REFERENCED_BY_VERTEX_SHADER:
        case GL_REFERENCED_BY_FRAGMENT_SHADER:
        case GL_REFERENCED_BY_COMPUTE_SHADER:

        case GL_TOP_LEVEL_ARRAY_SIZE:
        case GL_TOP_LEVEL_ARRAY_STRIDE:

        case GL_TYPE:
            return true;

        case GL_REFERENCED_BY_GEOMETRY_SHADER_EXT:
            return context->getExtensions().geometryShader;

        case GL_LOCATION_INDEX_EXT:
            return context->getExtensions().blendFuncExtended;

        default:
            return false;
    }
}

// GLES 3.10 spec: Page 82 -- Table 7.2
bool ValidateProgramResourcePropertyByInterface(GLenum prop, GLenum programInterface)
{
    switch (prop)
    {
        case GL_ACTIVE_VARIABLES:
        case GL_BUFFER_BINDING:
        case GL_NUM_ACTIVE_VARIABLES:
        {
            switch (programInterface)
            {
                case GL_ATOMIC_COUNTER_BUFFER:
                case GL_SHADER_STORAGE_BLOCK:
                case GL_UNIFORM_BLOCK:
                    return true;
                default:
                    return false;
            }
        }

        case GL_ARRAY_SIZE:
        {
            switch (programInterface)
            {
                case GL_BUFFER_VARIABLE:
                case GL_PROGRAM_INPUT:
                case GL_PROGRAM_OUTPUT:
                case GL_TRANSFORM_FEEDBACK_VARYING:
                case GL_UNIFORM:
                    return true;
                default:
                    return false;
            }
        }

        case GL_ARRAY_STRIDE:
        case GL_BLOCK_INDEX:
        case GL_IS_ROW_MAJOR:
        case GL_MATRIX_STRIDE:
        {
            switch (programInterface)
            {
                case GL_BUFFER_VARIABLE:
                case GL_UNIFORM:
                    return true;
                default:
                    return false;
            }
        }

        case GL_ATOMIC_COUNTER_BUFFER_INDEX:
        {
            if (programInterface == GL_UNIFORM)
            {
                return true;
            }
            return false;
        }

        case GL_BUFFER_DATA_SIZE:
        {
            switch (programInterface)
            {
                case GL_ATOMIC_COUNTER_BUFFER:
                case GL_SHADER_STORAGE_BLOCK:
                case GL_UNIFORM_BLOCK:
                    return true;
                default:
                    return false;
            }
        }

        case GL_LOCATION:
        {
            return ValidateLocationProgramInterface(programInterface);
        }

        case GL_LOCATION_INDEX_EXT:
        {
            // EXT_blend_func_extended
            return (programInterface == GL_PROGRAM_OUTPUT);
        }

        case GL_NAME_LENGTH:
        {
            return ValidateNamedProgramInterface(programInterface);
        }

        case GL_OFFSET:
        {
            switch (programInterface)
            {
                case GL_BUFFER_VARIABLE:
                case GL_UNIFORM:
                    return true;
                default:
                    return false;
            }
        }

        case GL_REFERENCED_BY_VERTEX_SHADER:
        case GL_REFERENCED_BY_FRAGMENT_SHADER:
        case GL_REFERENCED_BY_COMPUTE_SHADER:
        case GL_REFERENCED_BY_GEOMETRY_SHADER_EXT:
        {
            switch (programInterface)
            {
                case GL_ATOMIC_COUNTER_BUFFER:
                case GL_BUFFER_VARIABLE:
                case GL_PROGRAM_INPUT:
                case GL_PROGRAM_OUTPUT:
                case GL_SHADER_STORAGE_BLOCK:
                case GL_UNIFORM:
                case GL_UNIFORM_BLOCK:
                    return true;
                default:
                    return false;
            }
        }

        case GL_TOP_LEVEL_ARRAY_SIZE:
        case GL_TOP_LEVEL_ARRAY_STRIDE:
        {
            if (programInterface == GL_BUFFER_VARIABLE)
            {
                return true;
            }
            return false;
        }

        case GL_TYPE:
        {
            switch (programInterface)
            {
                case GL_BUFFER_VARIABLE:
                case GL_PROGRAM_INPUT:
                case GL_PROGRAM_OUTPUT:
                case GL_TRANSFORM_FEEDBACK_VARYING:
                case GL_UNIFORM:
                    return true;
                default:
                    return false;
            }
        }

        default:
            return false;
    }
}

bool ValidateProgramResourceIndex(const Program *programObject,
                                  GLenum programInterface,
                                  GLuint index)
{
    switch (programInterface)
    {
        case GL_PROGRAM_INPUT:
            return (index <
                    static_cast<GLuint>(programObject->getState().getProgramInputs().size()));

        case GL_PROGRAM_OUTPUT:
            return (index < static_cast<GLuint>(programObject->getOutputResourceCount()));

        case GL_UNIFORM:
            return (index < static_cast<GLuint>(programObject->getActiveUniformCount()));

        case GL_BUFFER_VARIABLE:
            return (index < static_cast<GLuint>(programObject->getActiveBufferVariableCount()));

        case GL_SHADER_STORAGE_BLOCK:
            return (index < static_cast<GLuint>(programObject->getActiveShaderStorageBlockCount()));

        case GL_UNIFORM_BLOCK:
            return (index < programObject->getActiveUniformBlockCount());

        case GL_ATOMIC_COUNTER_BUFFER:
            return (index < programObject->getActiveAtomicCounterBufferCount());

        case GL_TRANSFORM_FEEDBACK_VARYING:
            return (index < static_cast<GLuint>(programObject->getTransformFeedbackVaryingCount()));

        default:
            UNREACHABLE();
            return false;
    }
}

bool ValidateProgramUniform(Context *context,
                            GLenum valueType,
                            ShaderProgramID program,
                            GLint location,
                            GLsizei count)
{
    // Check for ES31 program uniform entry points
    if (context->getClientVersion() < Version(3, 1))
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    const LinkedUniform *uniform = nullptr;
    Program *programObject       = GetValidProgram(context, program);
    return ValidateUniformCommonBase(context, programObject, location, count, &uniform) &&
           ValidateUniformValue(context, valueType, uniform->type);
}

bool ValidateProgramUniformMatrix(Context *context,
                                  GLenum valueType,
                                  ShaderProgramID program,
                                  GLint location,
                                  GLsizei count,
                                  GLboolean transpose)
{
    // Check for ES31 program uniform entry points
    if (context->getClientVersion() < Version(3, 1))
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    const LinkedUniform *uniform = nullptr;
    Program *programObject       = GetValidProgram(context, program);
    return ValidateUniformCommonBase(context, programObject, location, count, &uniform) &&
           ValidateUniformMatrixValue(context, valueType, uniform->type);
}

bool ValidateVertexAttribFormatCommon(Context *context, GLuint relativeOffset)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    const Caps &caps = context->getCaps();
    if (relativeOffset > static_cast<GLuint>(caps.maxVertexAttribRelativeOffset))
    {
        context->validationError(GL_INVALID_VALUE, kRelativeOffsetTooLarge);
        return false;
    }

    // [OpenGL ES 3.1] Section 10.3.1 page 243:
    // An INVALID_OPERATION error is generated if the default vertex array object is bound.
    if (context->getState().getVertexArrayId().value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultVertexArray);
        return false;
    }

    return true;
}

}  // anonymous namespace

bool ValidateGetBooleani_v(Context *context, GLenum target, GLuint index, GLboolean *data)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (!ValidateIndexedStateQuery(context, target, index, nullptr))
    {
        return false;
    }

    return true;
}

bool ValidateGetBooleani_vRobustANGLE(Context *context,
                                      GLenum target,
                                      GLuint index,
                                      GLsizei bufSize,
                                      GLsizei *length,
                                      GLboolean *data)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateIndexedStateQuery(context, target, index, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);
    return true;
}

bool ValidateDrawIndirectBase(Context *context, PrimitiveMode mode, const void *indirect)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    // Here the third parameter 1 is only to pass the count validation.
    if (!ValidateDrawBase(context, mode))
    {
        return false;
    }

    const State &state = context->getState();

    // An INVALID_OPERATION error is generated if zero is bound to VERTEX_ARRAY_BINDING,
    // DRAW_INDIRECT_BUFFER or to any enabled vertex array.
    if (state.getVertexArrayId().value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultVertexArray);
        return false;
    }

    Buffer *drawIndirectBuffer = state.getTargetBuffer(BufferBinding::DrawIndirect);
    if (!drawIndirectBuffer)
    {
        context->validationError(GL_INVALID_OPERATION, kDrawIndirectBufferNotBound);
        return false;
    }

    // An INVALID_VALUE error is generated if indirect is not a multiple of the size, in basic
    // machine units, of uint.
    GLint64 offset = reinterpret_cast<GLint64>(indirect);
    if ((static_cast<GLuint>(offset) % sizeof(GLuint)) != 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidIndirectOffset);
        return false;
    }

    return true;
}

bool ValidateDrawArraysIndirect(Context *context, PrimitiveMode mode, const void *indirect)
{
    const State &state                      = context->getState();
    TransformFeedback *curTransformFeedback = state.getCurrentTransformFeedback();
    if (curTransformFeedback && curTransformFeedback->isActive() &&
        !curTransformFeedback->isPaused())
    {
        // EXT_geometry_shader allows transform feedback to work with all draw commands.
        // [EXT_geometry_shader] Section 12.1, "Transform Feedback"
        if (context->getExtensions().geometryShader)
        {
            if (!ValidateTransformFeedbackPrimitiveMode(
                    context, curTransformFeedback->getPrimitiveMode(), mode))
            {
                context->validationError(GL_INVALID_OPERATION, kInvalidDrawModeTransformFeedback);
                return false;
            }
        }
        else
        {
            // An INVALID_OPERATION error is generated if transform feedback is active and not
            // paused.
            context->validationError(GL_INVALID_OPERATION,
                                     kUnsupportedDrawModeForTransformFeedback);
            return false;
        }
    }

    if (!ValidateDrawIndirectBase(context, mode, indirect))
        return false;

    Buffer *drawIndirectBuffer = state.getTargetBuffer(BufferBinding::DrawIndirect);
    CheckedNumeric<size_t> checkedOffset(reinterpret_cast<size_t>(indirect));
    // In OpenGL ES3.1 spec, session 10.5, it defines the struct of DrawArraysIndirectCommand
    // which's size is 4 * sizeof(uint).
    auto checkedSum = checkedOffset + 4 * sizeof(GLuint);
    if (!checkedSum.IsValid() ||
        checkedSum.ValueOrDie() > static_cast<size_t>(drawIndirectBuffer->getSize()))
    {
        context->validationError(GL_INVALID_OPERATION, kParamOverflow);
        return false;
    }

    return true;
}

bool ValidateDrawElementsIndirect(Context *context,
                                  PrimitiveMode mode,
                                  DrawElementsType type,
                                  const void *indirect)
{
    if (!ValidateDrawElementsBase(context, mode, type))
    {
        return false;
    }

    const State &state         = context->getState();
    const VertexArray *vao     = state.getVertexArray();
    Buffer *elementArrayBuffer = vao->getElementArrayBuffer();
    if (!elementArrayBuffer)
    {
        context->validationError(GL_INVALID_OPERATION, kMustHaveElementArrayBinding);
        return false;
    }

    if (!ValidateDrawIndirectBase(context, mode, indirect))
        return false;

    Buffer *drawIndirectBuffer = state.getTargetBuffer(BufferBinding::DrawIndirect);
    CheckedNumeric<size_t> checkedOffset(reinterpret_cast<size_t>(indirect));
    // In OpenGL ES3.1 spec, session 10.5, it defines the struct of DrawElementsIndirectCommand
    // which's size is 5 * sizeof(uint).
    auto checkedSum = checkedOffset + 5 * sizeof(GLuint);
    if (!checkedSum.IsValid() ||
        checkedSum.ValueOrDie() > static_cast<size_t>(drawIndirectBuffer->getSize()))
    {
        context->validationError(GL_INVALID_OPERATION, kParamOverflow);
        return false;
    }

    return true;
}

bool ValidateProgramUniform1i(Context *context, ShaderProgramID program, GLint location, GLint v0)
{
    return ValidateProgramUniform1iv(context, program, location, 1, &v0);
}

bool ValidateProgramUniform2i(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLint v0,
                              GLint v1)
{
    GLint xy[2] = {v0, v1};
    return ValidateProgramUniform2iv(context, program, location, 1, xy);
}

bool ValidateProgramUniform3i(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLint v0,
                              GLint v1,
                              GLint v2)
{
    GLint xyz[3] = {v0, v1, v2};
    return ValidateProgramUniform3iv(context, program, location, 1, xyz);
}

bool ValidateProgramUniform4i(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLint v0,
                              GLint v1,
                              GLint v2,
                              GLint v3)
{
    GLint xyzw[4] = {v0, v1, v2, v3};
    return ValidateProgramUniform4iv(context, program, location, 1, xyzw);
}

bool ValidateProgramUniform1ui(Context *context, ShaderProgramID program, GLint location, GLuint v0)
{
    return ValidateProgramUniform1uiv(context, program, location, 1, &v0);
}

bool ValidateProgramUniform2ui(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLuint v0,
                               GLuint v1)
{
    GLuint xy[2] = {v0, v1};
    return ValidateProgramUniform2uiv(context, program, location, 1, xy);
}

bool ValidateProgramUniform3ui(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLuint v0,
                               GLuint v1,
                               GLuint v2)
{
    GLuint xyz[3] = {v0, v1, v2};
    return ValidateProgramUniform3uiv(context, program, location, 1, xyz);
}

bool ValidateProgramUniform4ui(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLuint v0,
                               GLuint v1,
                               GLuint v2,
                               GLuint v3)
{
    GLuint xyzw[4] = {v0, v1, v2, v3};
    return ValidateProgramUniform4uiv(context, program, location, 1, xyzw);
}

bool ValidateProgramUniform1f(Context *context, ShaderProgramID program, GLint location, GLfloat v0)
{
    return ValidateProgramUniform1fv(context, program, location, 1, &v0);
}

bool ValidateProgramUniform2f(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLfloat v0,
                              GLfloat v1)
{
    GLfloat xy[2] = {v0, v1};
    return ValidateProgramUniform2fv(context, program, location, 1, xy);
}

bool ValidateProgramUniform3f(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLfloat v0,
                              GLfloat v1,
                              GLfloat v2)
{
    GLfloat xyz[3] = {v0, v1, v2};
    return ValidateProgramUniform3fv(context, program, location, 1, xyz);
}

bool ValidateProgramUniform4f(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLfloat v0,
                              GLfloat v1,
                              GLfloat v2,
                              GLfloat v3)
{
    GLfloat xyzw[4] = {v0, v1, v2, v3};
    return ValidateProgramUniform4fv(context, program, location, 1, xyzw);
}

bool ValidateProgramUniform1iv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLint *value)
{
    // Check for ES31 program uniform entry points
    if (context->getClientVersion() < Version(3, 1))
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    const LinkedUniform *uniform = nullptr;
    Program *programObject       = GetValidProgram(context, program);
    return ValidateUniformCommonBase(context, programObject, location, count, &uniform) &&
           ValidateUniform1ivValue(context, uniform->type, count, value);
}

bool ValidateProgramUniform2iv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLint *value)
{
    return ValidateProgramUniform(context, GL_INT_VEC2, program, location, count);
}

bool ValidateProgramUniform3iv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLint *value)
{
    return ValidateProgramUniform(context, GL_INT_VEC3, program, location, count);
}

bool ValidateProgramUniform4iv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLint *value)
{
    return ValidateProgramUniform(context, GL_INT_VEC4, program, location, count);
}

bool ValidateProgramUniform1uiv(Context *context,
                                ShaderProgramID program,
                                GLint location,
                                GLsizei count,
                                const GLuint *value)
{
    return ValidateProgramUniform(context, GL_UNSIGNED_INT, program, location, count);
}

bool ValidateProgramUniform2uiv(Context *context,
                                ShaderProgramID program,
                                GLint location,
                                GLsizei count,
                                const GLuint *value)
{
    return ValidateProgramUniform(context, GL_UNSIGNED_INT_VEC2, program, location, count);
}

bool ValidateProgramUniform3uiv(Context *context,
                                ShaderProgramID program,
                                GLint location,
                                GLsizei count,
                                const GLuint *value)
{
    return ValidateProgramUniform(context, GL_UNSIGNED_INT_VEC3, program, location, count);
}

bool ValidateProgramUniform4uiv(Context *context,
                                ShaderProgramID program,
                                GLint location,
                                GLsizei count,
                                const GLuint *value)
{
    return ValidateProgramUniform(context, GL_UNSIGNED_INT_VEC4, program, location, count);
}

bool ValidateProgramUniform1fv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLfloat *value)
{
    return ValidateProgramUniform(context, GL_FLOAT, program, location, count);
}

bool ValidateProgramUniform2fv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLfloat *value)
{
    return ValidateProgramUniform(context, GL_FLOAT_VEC2, program, location, count);
}

bool ValidateProgramUniform3fv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLfloat *value)
{
    return ValidateProgramUniform(context, GL_FLOAT_VEC3, program, location, count);
}

bool ValidateProgramUniform4fv(Context *context,
                               ShaderProgramID program,
                               GLint location,
                               GLsizei count,
                               const GLfloat *value)
{
    return ValidateProgramUniform(context, GL_FLOAT_VEC4, program, location, count);
}

bool ValidateProgramUniformMatrix2fv(Context *context,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT2, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix3fv(Context *context,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT3, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix4fv(Context *context,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT4, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix2x3fv(Context *context,
                                       ShaderProgramID program,
                                       GLint location,
                                       GLsizei count,
                                       GLboolean transpose,
                                       const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT2x3, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix3x2fv(Context *context,
                                       ShaderProgramID program,
                                       GLint location,
                                       GLsizei count,
                                       GLboolean transpose,
                                       const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT3x2, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix2x4fv(Context *context,
                                       ShaderProgramID program,
                                       GLint location,
                                       GLsizei count,
                                       GLboolean transpose,
                                       const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT2x4, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix4x2fv(Context *context,
                                       ShaderProgramID program,
                                       GLint location,
                                       GLsizei count,
                                       GLboolean transpose,
                                       const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT4x2, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix3x4fv(Context *context,
                                       ShaderProgramID program,
                                       GLint location,
                                       GLsizei count,
                                       GLboolean transpose,
                                       const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT3x4, program, location, count,
                                        transpose);
}

bool ValidateProgramUniformMatrix4x3fv(Context *context,
                                       ShaderProgramID program,
                                       GLint location,
                                       GLsizei count,
                                       GLboolean transpose,
                                       const GLfloat *value)
{
    return ValidateProgramUniformMatrix(context, GL_FLOAT_MAT4x3, program, location, count,
                                        transpose);
}

bool ValidateGetTexLevelParameterfv(Context *context,
                                    TextureTarget target,
                                    GLint level,
                                    GLenum pname,
                                    GLfloat *params)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    return ValidateGetTexLevelParameterBase(context, target, level, pname, nullptr);
}

bool ValidateGetTexLevelParameterfvRobustANGLE(Context *context,
                                               TextureTarget target,
                                               GLint level,
                                               GLenum pname,
                                               GLsizei bufSize,
                                               GLsizei *length,
                                               GLfloat *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetTexLevelParameteriv(Context *context,
                                    TextureTarget target,
                                    GLint level,
                                    GLenum pname,
                                    GLint *params)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    return ValidateGetTexLevelParameterBase(context, target, level, pname, nullptr);
}

bool ValidateGetTexLevelParameterivRobustANGLE(Context *context,
                                               TextureTarget target,
                                               GLint level,
                                               GLenum pname,
                                               GLsizei bufSize,
                                               GLsizei *length,
                                               GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateTexStorage2DMultisample(Context *context,
                                     TextureType target,
                                     GLsizei samples,
                                     GLenum internalFormat,
                                     GLsizei width,
                                     GLsizei height,
                                     GLboolean fixedSampleLocations)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    return ValidateTexStorage2DMultisampleBase(context, target, samples, internalFormat, width,
                                               height);
}

bool ValidateTexStorageMem2DMultisampleEXT(Context *context,
                                           TextureType target,
                                           GLsizei samples,
                                           GLenum internalFormat,
                                           GLsizei width,
                                           GLsizei height,
                                           GLboolean fixedSampleLocations,
                                           MemoryObjectID memory,
                                           GLuint64 offset)
{
    if (!context->getExtensions().memoryObject)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    UNIMPLEMENTED();
    return false;
}

bool ValidateGetMultisamplefv(Context *context, GLenum pname, GLuint index, GLfloat *val)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    return ValidateGetMultisamplefvBase(context, pname, index, val);
}

bool ValidateGetMultisamplefvRobustANGLE(Context *context,
                                         GLenum pname,
                                         GLuint index,
                                         GLsizei bufSize,
                                         GLsizei *length,
                                         GLfloat *val)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateFramebufferParameteri(Context *context, GLenum target, GLenum pname, GLint param)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (!ValidFramebufferTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidFramebufferTarget);
        return false;
    }

    switch (pname)
    {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
        {
            GLint maxWidth = context->getCaps().maxFramebufferWidth;
            if (param < 0 || param > maxWidth)
            {
                context->validationError(GL_INVALID_VALUE, kExceedsFramebufferWidth);
                return false;
            }
            break;
        }
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
        {
            GLint maxHeight = context->getCaps().maxFramebufferHeight;
            if (param < 0 || param > maxHeight)
            {
                context->validationError(GL_INVALID_VALUE, kExceedsFramebufferHeight);
                return false;
            }
            break;
        }
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
        {
            GLint maxSamples = context->getCaps().maxFramebufferSamples;
            if (param < 0 || param > maxSamples)
            {
                context->validationError(GL_INVALID_VALUE, kExceedsFramebufferSamples);
                return false;
            }
            break;
        }
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
        {
            break;
        }
        case GL_FRAMEBUFFER_DEFAULT_LAYERS_EXT:
        {
            if (!context->getExtensions().geometryShader)
            {
                context->validationError(GL_INVALID_ENUM, kGeometryShaderExtensionNotEnabled);
                return false;
            }
            GLint maxLayers = context->getCaps().maxFramebufferLayers;
            if (param < 0 || param > maxLayers)
            {
                context->validationError(GL_INVALID_VALUE, kInvalidFramebufferLayer);
                return false;
            }
            break;
        }
        default:
        {
            context->validationError(GL_INVALID_ENUM, kInvalidPname);
            return false;
        }
    }

    const Framebuffer *framebuffer = context->getState().getTargetFramebuffer(target);
    ASSERT(framebuffer);
    if (framebuffer->isDefault())
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultFramebuffer);
        return false;
    }
    return true;
}

bool ValidateGetFramebufferParameteriv(Context *context, GLenum target, GLenum pname, GLint *params)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (!ValidFramebufferTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidFramebufferTarget);
        return false;
    }

    switch (pname)
    {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS_EXT:
            if (!context->getExtensions().geometryShader)
            {
                context->validationError(GL_INVALID_ENUM, kGeometryShaderExtensionNotEnabled);
                return false;
            }
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidPname);
            return false;
    }

    const Framebuffer *framebuffer = context->getState().getTargetFramebuffer(target);
    ASSERT(framebuffer);

    if (framebuffer->isDefault())
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultFramebuffer);
        return false;
    }
    return true;
}

bool ValidateGetFramebufferParameterivRobustANGLE(Context *context,
                                                  GLenum target,
                                                  GLenum pname,
                                                  GLsizei bufSize,
                                                  GLsizei *length,
                                                  GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetProgramResourceIndex(Context *context,
                                     ShaderProgramID program,
                                     GLenum programInterface,
                                     const GLchar *name)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    Program *programObject = GetValidProgram(context, program);
    if (programObject == nullptr)
    {
        return false;
    }

    if (!ValidateNamedProgramInterface(programInterface))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidProgramInterface);
        return false;
    }

    return true;
}

bool ValidateBindVertexBuffer(Context *context,
                              GLuint bindingIndex,
                              BufferID buffer,
                              GLintptr offset,
                              GLsizei stride)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (!context->isBufferGenerated(buffer))
    {
        context->validationError(GL_INVALID_OPERATION, kObjectNotGenerated);
        return false;
    }

    const Caps &caps = context->getCaps();
    if (bindingIndex >= caps.maxVertexAttribBindings)
    {
        context->validationError(GL_INVALID_VALUE, kExceedsMaxVertexAttribBindings);
        return false;
    }

    if (offset < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeOffset);
        return false;
    }

    if (stride < 0 || stride > caps.maxVertexAttribStride)
    {
        context->validationError(GL_INVALID_VALUE, kExceedsMaxVertexAttribStride);
        return false;
    }

    // [OpenGL ES 3.1] Section 10.3.1 page 244:
    // An INVALID_OPERATION error is generated if the default vertex array object is bound.
    if (context->getState().getVertexArrayId().value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultVertexArray);
        return false;
    }

    return true;
}

bool ValidateVertexBindingDivisor(Context *context, GLuint bindingIndex, GLuint divisor)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    const Caps &caps = context->getCaps();
    if (bindingIndex >= caps.maxVertexAttribBindings)
    {
        context->validationError(GL_INVALID_VALUE, kExceedsMaxVertexAttribBindings);
        return false;
    }

    // [OpenGL ES 3.1] Section 10.3.1 page 243:
    // An INVALID_OPERATION error is generated if the default vertex array object is bound.
    if (context->getState().getVertexArrayId().value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultVertexArray);
        return false;
    }

    return true;
}

bool ValidateVertexAttribFormat(Context *context,
                                GLuint attribindex,
                                GLint size,
                                VertexAttribType type,
                                GLboolean normalized,
                                GLuint relativeoffset)
{
    if (!ValidateVertexAttribFormatCommon(context, relativeoffset))
    {
        return false;
    }

    return ValidateFloatVertexFormat(context, attribindex, size, type);
}

bool ValidateVertexAttribIFormat(Context *context,
                                 GLuint attribindex,
                                 GLint size,
                                 VertexAttribType type,
                                 GLuint relativeoffset)
{
    if (!ValidateVertexAttribFormatCommon(context, relativeoffset))
    {
        return false;
    }

    return ValidateIntegerVertexFormat(context, attribindex, size, type);
}

bool ValidateVertexAttribBinding(Context *context, GLuint attribIndex, GLuint bindingIndex)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    // [OpenGL ES 3.1] Section 10.3.1 page 243:
    // An INVALID_OPERATION error is generated if the default vertex array object is bound.
    if (context->getState().getVertexArrayId().value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultVertexArray);
        return false;
    }

    const Caps &caps = context->getCaps();
    if (attribIndex >= caps.maxVertexAttributes)
    {
        context->validationError(GL_INVALID_VALUE, kIndexExceedsMaxVertexAttribute);
        return false;
    }

    if (bindingIndex >= caps.maxVertexAttribBindings)
    {
        context->validationError(GL_INVALID_VALUE, kExceedsMaxVertexAttribBindings);
        return false;
    }

    return true;
}

bool ValidateGetProgramResourceName(Context *context,
                                    ShaderProgramID program,
                                    GLenum programInterface,
                                    GLuint index,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLchar *name)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    Program *programObject = GetValidProgram(context, program);
    if (programObject == nullptr)
    {
        return false;
    }

    if (!ValidateNamedProgramInterface(programInterface))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidProgramInterface);
        return false;
    }

    if (!ValidateProgramResourceIndex(programObject, programInterface, index))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidProgramResourceIndex);
        return false;
    }

    if (bufSize < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeBufferSize);
        return false;
    }

    return true;
}

bool ValidateDispatchCompute(Context *context,
                             GLuint numGroupsX,
                             GLuint numGroupsY,
                             GLuint numGroupsZ)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    const State &state = context->getState();
    Program *program   = state.getLinkedProgram(context);

    if (program == nullptr || !program->hasLinkedShaderStage(ShaderType::Compute))
    {
        context->validationError(GL_INVALID_OPERATION, kNoActiveProgramWithComputeShader);
        return false;
    }

    const Caps &caps = context->getCaps();
    if (numGroupsX > caps.maxComputeWorkGroupCount[0])
    {
        context->validationError(GL_INVALID_VALUE, kExceedsComputeWorkGroupCountX);
        return false;
    }
    if (numGroupsY > caps.maxComputeWorkGroupCount[1])
    {
        context->validationError(GL_INVALID_VALUE, kExceedsComputeWorkGroupCountY);
        return false;
    }
    if (numGroupsZ > caps.maxComputeWorkGroupCount[2])
    {
        context->validationError(GL_INVALID_VALUE, kExceedsComputeWorkGroupCountZ);
        return false;
    }

    return true;
}

bool ValidateDispatchComputeIndirect(Context *context, GLintptr indirect)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    const State &state = context->getState();
    Program *program   = state.getLinkedProgram(context);

    if (program == nullptr || !program->hasLinkedShaderStage(ShaderType::Compute))
    {
        context->validationError(GL_INVALID_OPERATION, kNoActiveProgramWithComputeShader);
        return false;
    }

    if (indirect < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeOffset);
        return false;
    }

    if ((indirect & (sizeof(GLuint) - 1)) != 0)
    {
        context->validationError(GL_INVALID_VALUE, kOffsetMustBeMultipleOfUint);
        return false;
    }

    Buffer *dispatchIndirectBuffer = state.getTargetBuffer(BufferBinding::DispatchIndirect);
    if (!dispatchIndirectBuffer)
    {
        context->validationError(GL_INVALID_OPERATION, kDispatchIndirectBufferNotBound);
        return false;
    }

    CheckedNumeric<GLuint64> checkedOffset(static_cast<GLuint64>(indirect));
    auto checkedSum = checkedOffset + static_cast<GLuint64>(3 * sizeof(GLuint));
    if (!checkedSum.IsValid() ||
        checkedSum.ValueOrDie() > static_cast<GLuint64>(dispatchIndirectBuffer->getSize()))
    {
        context->validationError(GL_INVALID_OPERATION, kInsufficientBufferSize);
        return false;
    }

    return true;
}

bool ValidateBindImageTexture(Context *context,
                              GLuint unit,
                              TextureID texture,
                              GLint level,
                              GLboolean layered,
                              GLint layer,
                              GLenum access,
                              GLenum format)
{
    GLuint maxImageUnits = context->getCaps().maxImageUnits;
    if (unit >= maxImageUnits)
    {
        context->validationError(GL_INVALID_VALUE, kExceedsMaxImageUnits);
        return false;
    }

    if (level < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeLevel);
        return false;
    }

    if (layer < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeLayer);
        return false;
    }

    if (access != GL_READ_ONLY && access != GL_WRITE_ONLY && access != GL_READ_WRITE)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidImageAccess);
        return false;
    }

    switch (format)
    {
        case GL_RGBA32F:
        case GL_RGBA16F:
        case GL_R32F:
        case GL_RGBA32UI:
        case GL_RGBA16UI:
        case GL_RGBA8UI:
        case GL_R32UI:
        case GL_RGBA32I:
        case GL_RGBA16I:
        case GL_RGBA8I:
        case GL_R32I:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
            break;
        default:
            context->validationError(GL_INVALID_VALUE, kInvalidImageFormat);
            return false;
    }

    if (texture.value != 0)
    {
        Texture *tex = context->getTexture(texture);

        if (tex == nullptr)
        {
            context->validationError(GL_INVALID_VALUE, kMissingTextureName);
            return false;
        }

        if (!tex->getImmutableFormat())
        {
            context->validationError(GL_INVALID_OPERATION, kTextureIsNotImmutable);
            return false;
        }
    }

    return true;
}

bool ValidateGetProgramResourceLocation(Context *context,
                                        ShaderProgramID program,
                                        GLenum programInterface,
                                        const GLchar *name)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    Program *programObject = GetValidProgram(context, program);
    if (programObject == nullptr)
    {
        return false;
    }

    if (!programObject->isLinked())
    {
        context->validationError(GL_INVALID_OPERATION, kProgramNotLinked);
        return false;
    }

    if (!ValidateLocationProgramInterface(programInterface))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidProgramInterface);
        return false;
    }
    return true;
}

bool ValidateGetProgramResourceiv(Context *context,
                                  ShaderProgramID program,
                                  GLenum programInterface,
                                  GLuint index,
                                  GLsizei propCount,
                                  const GLenum *props,
                                  GLsizei bufSize,
                                  GLsizei *length,
                                  GLint *params)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    Program *programObject = GetValidProgram(context, program);
    if (programObject == nullptr)
    {
        return false;
    }
    if (!ValidateProgramInterface(programInterface))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidProgramInterface);
        return false;
    }
    if (propCount <= 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidPropCount);
        return false;
    }
    if (bufSize < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeBufSize);
        return false;
    }
    if (!ValidateProgramResourceIndex(programObject, programInterface, index))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidProgramResourceIndex);
        return false;
    }
    for (GLsizei i = 0; i < propCount; i++)
    {
        if (!ValidateProgramResourceProperty(context, props[i]))
        {
            context->validationError(GL_INVALID_ENUM, kInvalidProgramResourceProperty);
            return false;
        }
        if (!ValidateProgramResourcePropertyByInterface(props[i], programInterface))
        {
            context->validationError(GL_INVALID_OPERATION, kInvalidPropertyForProgramInterface);
            return false;
        }
    }
    return true;
}

bool ValidateGetProgramInterfaceiv(Context *context,
                                   ShaderProgramID program,
                                   GLenum programInterface,
                                   GLenum pname,
                                   GLint *params)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    Program *programObject = GetValidProgram(context, program);
    if (programObject == nullptr)
    {
        return false;
    }

    if (!ValidateProgramInterface(programInterface))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidProgramInterface);
        return false;
    }

    switch (pname)
    {
        case GL_ACTIVE_RESOURCES:
        case GL_MAX_NAME_LENGTH:
        case GL_MAX_NUM_ACTIVE_VARIABLES:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidPname);
            return false;
    }

    if (pname == GL_MAX_NAME_LENGTH && programInterface == GL_ATOMIC_COUNTER_BUFFER)
    {
        context->validationError(GL_INVALID_OPERATION, kAtomicCounterResourceName);
        return false;
    }

    if (pname == GL_MAX_NUM_ACTIVE_VARIABLES)
    {
        switch (programInterface)
        {
            case GL_ATOMIC_COUNTER_BUFFER:
            case GL_SHADER_STORAGE_BLOCK:
            case GL_UNIFORM_BLOCK:
                break;

            default:
                context->validationError(GL_INVALID_OPERATION, kMaxActiveVariablesInterface);
                return false;
        }
    }

    return true;
}

bool ValidateGetProgramInterfaceivRobustANGLE(Context *context,
                                              ShaderProgramID program,
                                              GLenum programInterface,
                                              GLenum pname,
                                              GLsizei bufSize,
                                              GLsizei *length,
                                              GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

static bool ValidateGenOrDeleteES31(Context *context, GLint n)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    return ValidateGenOrDelete(context, n);
}

bool ValidateGenProgramPipelines(Context *context, GLint n, ProgramPipelineID *pipelines)
{
    return ValidateGenOrDeleteES31(context, n);
}

bool ValidateDeleteProgramPipelines(Context *context, GLint n, const ProgramPipelineID *pipelines)
{
    return ValidateGenOrDeleteES31(context, n);
}

bool ValidateBindProgramPipeline(Context *context, ProgramPipelineID pipeline)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (!context->isProgramPipelineGenerated({pipeline}))
    {
        context->validationError(GL_INVALID_OPERATION, kObjectNotGenerated);
        return false;
    }

    return true;
}

bool ValidateIsProgramPipeline(Context *context, ProgramPipelineID pipeline)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    return true;
}

bool ValidateUseProgramStages(Context *context,
                              ProgramPipelineID pipeline,
                              GLbitfield stages,
                              ShaderProgramID program)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateActiveShaderProgram(Context *context,
                                 ProgramPipelineID pipeline,
                                 ShaderProgramID program)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateCreateShaderProgramv(Context *context,
                                  ShaderType type,
                                  GLsizei count,
                                  const GLchar *const *strings)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetProgramPipelineiv(Context *context,
                                  ProgramPipelineID pipeline,
                                  GLenum pname,
                                  GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateValidateProgramPipeline(Context *context, ProgramPipelineID pipeline)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetProgramPipelineInfoLog(Context *context,
                                       ProgramPipelineID pipeline,
                                       GLsizei bufSize,
                                       GLsizei *length,
                                       GLchar *infoLog)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateMemoryBarrier(Context *context, GLbitfield barriers)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (barriers == GL_ALL_BARRIER_BITS)
    {
        return true;
    }

    GLbitfield supported_barrier_bits =
        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT |
        GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_COMMAND_BARRIER_BIT |
        GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT |
        GL_FRAMEBUFFER_BARRIER_BIT | GL_TRANSFORM_FEEDBACK_BARRIER_BIT |
        GL_ATOMIC_COUNTER_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT;
    if (barriers == 0 || (barriers & ~supported_barrier_bits) != 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidMemoryBarrierBit);
        return false;
    }

    return true;
}

bool ValidateMemoryBarrierByRegion(Context *context, GLbitfield barriers)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    if (barriers == GL_ALL_BARRIER_BITS)
    {
        return true;
    }

    GLbitfield supported_barrier_bits = GL_ATOMIC_COUNTER_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT |
                                        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                        GL_SHADER_STORAGE_BARRIER_BIT |
                                        GL_TEXTURE_FETCH_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT;
    if (barriers == 0 || (barriers & ~supported_barrier_bits) != 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidMemoryBarrierBit);
        return false;
    }

    return true;
}

bool ValidateSampleMaski(Context *context, GLuint maskNumber, GLbitfield mask)
{
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }

    return ValidateSampleMaskiBase(context, maskNumber, mask);
}

bool ValidateFramebufferTextureEXT(Context *context,
                                   GLenum target,
                                   GLenum attachment,
                                   TextureID texture,
                                   GLint level)
{
    if (!context->getExtensions().geometryShader)
    {
        context->validationError(GL_INVALID_OPERATION, kGeometryShaderExtensionNotEnabled);
        return false;
    }

    if (texture.value != 0)
    {
        gl::Texture *tex = context->getTexture(texture);

        // [EXT_geometry_shader] Section 9.2.8 "Attaching Texture Images to a Framebuffer"
        // An INVALID_VALUE error is generated if <texture> is not the name of a texture object.
        // We put this validation before ValidateFramebufferTextureBase because it is an
        // INVALID_OPERATION error for both FramebufferTexture2D and FramebufferTextureLayer:
        // [OpenGL ES 3.1] Chapter 9.2.8 (FramebufferTexture2D)
        // An INVALID_OPERATION error is generated if texture is not zero, and does not name an
        // existing texture object of type matching textarget.
        // [OpenGL ES 3.1 Chapter 9.2.8 (FramebufferTextureLayer)
        // An INVALID_OPERATION error is generated if texture is non-zero and is not the name of a
        // three-dimensional or two-dimensional array texture.
        if (tex == nullptr)
        {
            context->validationError(GL_INVALID_VALUE, kInvalidTextureName);
            return false;
        }

        if (!ValidMipLevel(context, tex->getType(), level))
        {
            context->validationError(GL_INVALID_VALUE, kInvalidMipLevel);
            return false;
        }
    }

    if (!ValidateFramebufferTextureBase(context, target, attachment, texture, level))
    {
        return false;
    }

    return true;
}

// GL_OES_texture_storage_multisample_2d_array
bool ValidateTexStorage3DMultisampleOES(Context *context,
                                        TextureType target,
                                        GLsizei samples,
                                        GLenum sizedinternalformat,
                                        GLsizei width,
                                        GLsizei height,
                                        GLsizei depth,
                                        GLboolean fixedsamplelocations)
{
    if (!context->getExtensions().textureStorageMultisample2DArray)
    {
        context->validationError(GL_INVALID_ENUM, kMultisampleArrayExtensionRequired);
        return false;
    }

    if (target != TextureType::_2DMultisampleArray)
    {
        context->validationError(GL_INVALID_ENUM, kTargetMustBeTexture2DMultisampleArrayOES);
        return false;
    }

    if (width < 1 || height < 1 || depth < 1)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeSize);
        return false;
    }

    return ValidateTexStorageMultisample(context, target, samples, sizedinternalformat, width,
                                         height);
}

bool ValidateTexStorageMem3DMultisampleEXT(Context *context,
                                           TextureType target,
                                           GLsizei samples,
                                           GLenum internalFormat,
                                           GLsizei width,
                                           GLsizei height,
                                           GLsizei depth,
                                           GLboolean fixedSampleLocations,
                                           MemoryObjectID memory,
                                           GLuint64 offset)
{
    if (!context->getExtensions().memoryObject)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    UNIMPLEMENTED();
    return false;
}

bool ValidateGetProgramResourceLocationIndexEXT(Context *context,
                                                ShaderProgramID program,
                                                GLenum programInterface,
                                                const char *name)
{
    if (!context->getExtensions().blendFuncExtended)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }
    if (context->getClientVersion() < ES_3_1)
    {
        context->validationError(GL_INVALID_OPERATION, kES31Required);
        return false;
    }
    if (programInterface != GL_PROGRAM_OUTPUT)
    {
        context->validationError(GL_INVALID_ENUM, kProgramInterfaceMustBeProgramOutput);
        return false;
    }
    Program *programObject = GetValidProgram(context, program);
    if (!programObject)
    {
        return false;
    }
    if (!programObject->isLinked())
    {
        context->validationError(GL_INVALID_OPERATION, kProgramNotLinked);
        return false;
    }
    return true;
}

}  // namespace gl
