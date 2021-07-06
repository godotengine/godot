//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// utilities.cpp: Conversion functions and other utility routines.

#include "common/utilities.h"
#include <GLSLANG/ShaderVars.h>
#include "common/mathutil.h"
#include "common/platform.h"

#include <set>

#if defined(ANGLE_ENABLE_WINDOWS_UWP)
#    include <windows.applicationmodel.core.h>
#    include <windows.graphics.display.h>
#    include <wrl.h>
#    include <wrl/wrappers/corewrappers.h>
#endif

namespace
{

template <class IndexType>
gl::IndexRange ComputeTypedIndexRange(const IndexType *indices,
                                      size_t count,
                                      bool primitiveRestartEnabled,
                                      GLuint primitiveRestartIndex)
{
    ASSERT(count > 0);

    IndexType minIndex                = 0;
    IndexType maxIndex                = 0;
    size_t nonPrimitiveRestartIndices = 0;

    if (primitiveRestartEnabled)
    {
        // Find the first non-primitive restart index to initialize the min and max values
        size_t i = 0;
        for (; i < count; i++)
        {
            if (indices[i] != primitiveRestartIndex)
            {
                minIndex = indices[i];
                maxIndex = indices[i];
                nonPrimitiveRestartIndices++;
                break;
            }
        }

        // Loop over the rest of the indices
        for (; i < count; i++)
        {
            if (indices[i] != primitiveRestartIndex)
            {
                if (minIndex > indices[i])
                {
                    minIndex = indices[i];
                }
                if (maxIndex < indices[i])
                {
                    maxIndex = indices[i];
                }
                nonPrimitiveRestartIndices++;
            }
        }
    }
    else
    {
        minIndex                   = indices[0];
        maxIndex                   = indices[0];
        nonPrimitiveRestartIndices = count;

        for (size_t i = 1; i < count; i++)
        {
            if (minIndex > indices[i])
            {
                minIndex = indices[i];
            }
            if (maxIndex < indices[i])
            {
                maxIndex = indices[i];
            }
        }
    }

    return gl::IndexRange(static_cast<size_t>(minIndex), static_cast<size_t>(maxIndex),
                          nonPrimitiveRestartIndices);
}

}  // anonymous namespace

namespace gl
{

int VariableComponentCount(GLenum type)
{
    return VariableRowCount(type) * VariableColumnCount(type);
}

GLenum VariableComponentType(GLenum type)
{
    switch (type)
    {
        case GL_BOOL:
        case GL_BOOL_VEC2:
        case GL_BOOL_VEC3:
        case GL_BOOL_VEC4:
            return GL_BOOL;
        case GL_FLOAT:
        case GL_FLOAT_VEC2:
        case GL_FLOAT_VEC3:
        case GL_FLOAT_VEC4:
        case GL_FLOAT_MAT2:
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT4:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT3x2:
        case GL_FLOAT_MAT2x4:
        case GL_FLOAT_MAT4x2:
        case GL_FLOAT_MAT3x4:
        case GL_FLOAT_MAT4x3:
            return GL_FLOAT;
        case GL_INT:
        case GL_SAMPLER_2D:
        case GL_SAMPLER_2D_RECT_ANGLE:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_EXTERNAL_OES:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_INT_VEC2:
        case GL_INT_VEC3:
        case GL_INT_VEC4:
        case GL_IMAGE_2D:
        case GL_INT_IMAGE_2D:
        case GL_UNSIGNED_INT_IMAGE_2D:
        case GL_IMAGE_3D:
        case GL_INT_IMAGE_3D:
        case GL_UNSIGNED_INT_IMAGE_3D:
        case GL_IMAGE_2D_ARRAY:
        case GL_INT_IMAGE_2D_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
        case GL_IMAGE_CUBE:
        case GL_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_ATOMIC_COUNTER:
            return GL_INT;
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_INT_VEC2:
        case GL_UNSIGNED_INT_VEC3:
        case GL_UNSIGNED_INT_VEC4:
            return GL_UNSIGNED_INT;
        default:
            UNREACHABLE();
    }

    return GL_NONE;
}

size_t VariableComponentSize(GLenum type)
{
    switch (type)
    {
        case GL_BOOL:
            return sizeof(GLint);
        case GL_FLOAT:
            return sizeof(GLfloat);
        case GL_INT:
            return sizeof(GLint);
        case GL_UNSIGNED_INT:
            return sizeof(GLuint);
        default:
            UNREACHABLE();
    }

    return 0;
}

size_t VariableInternalSize(GLenum type)
{
    // Expanded to 4-element vectors
    return VariableComponentSize(VariableComponentType(type)) * VariableRowCount(type) * 4;
}

size_t VariableExternalSize(GLenum type)
{
    return VariableComponentSize(VariableComponentType(type)) * VariableComponentCount(type);
}

GLenum VariableBoolVectorType(GLenum type)
{
    switch (type)
    {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            return GL_BOOL;
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return GL_BOOL_VEC2;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return GL_BOOL_VEC3;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return GL_BOOL_VEC4;

        default:
            UNREACHABLE();
            return GL_NONE;
    }
}

int VariableRowCount(GLenum type)
{
    switch (type)
    {
        case GL_NONE:
            return 0;
        case GL_BOOL:
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_BOOL_VEC2:
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC3:
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC4:
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
        case GL_SAMPLER_2D:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_EXTERNAL_OES:
        case GL_SAMPLER_2D_RECT_ANGLE:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_IMAGE_2D:
        case GL_INT_IMAGE_2D:
        case GL_UNSIGNED_INT_IMAGE_2D:
        case GL_IMAGE_2D_ARRAY:
        case GL_INT_IMAGE_2D_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
        case GL_IMAGE_3D:
        case GL_INT_IMAGE_3D:
        case GL_UNSIGNED_INT_IMAGE_3D:
        case GL_IMAGE_CUBE:
        case GL_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_ATOMIC_COUNTER:
            return 1;
        case GL_FLOAT_MAT2:
        case GL_FLOAT_MAT3x2:
        case GL_FLOAT_MAT4x2:
            return 2;
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT4x3:
            return 3;
        case GL_FLOAT_MAT4:
        case GL_FLOAT_MAT2x4:
        case GL_FLOAT_MAT3x4:
            return 4;
        default:
            UNREACHABLE();
    }

    return 0;
}

int VariableColumnCount(GLenum type)
{
    switch (type)
    {
        case GL_NONE:
            return 0;
        case GL_BOOL:
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_SAMPLER_2D:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_EXTERNAL_OES:
        case GL_SAMPLER_2D_RECT_ANGLE:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_IMAGE_2D:
        case GL_INT_IMAGE_2D:
        case GL_UNSIGNED_INT_IMAGE_2D:
        case GL_IMAGE_3D:
        case GL_INT_IMAGE_3D:
        case GL_UNSIGNED_INT_IMAGE_3D:
        case GL_IMAGE_2D_ARRAY:
        case GL_INT_IMAGE_2D_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
        case GL_IMAGE_CUBE:
        case GL_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_ATOMIC_COUNTER:
            return 1;
        case GL_BOOL_VEC2:
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
        case GL_FLOAT_MAT2:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT2x4:
            return 2;
        case GL_BOOL_VEC3:
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT3x2:
        case GL_FLOAT_MAT3x4:
            return 3;
        case GL_BOOL_VEC4:
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
        case GL_FLOAT_MAT4:
        case GL_FLOAT_MAT4x2:
        case GL_FLOAT_MAT4x3:
            return 4;
        default:
            UNREACHABLE();
    }

    return 0;
}

bool IsSamplerType(GLenum type)
{
    switch (type)
    {
        case GL_SAMPLER_2D:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_EXTERNAL_OES:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_2D_RECT_ANGLE:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
            return true;
    }

    return false;
}

bool IsSamplerCubeType(GLenum type)
{
    switch (type)
    {
        case GL_SAMPLER_CUBE:
        case GL_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_SAMPLER_CUBE_SHADOW:
            return true;
    }

    return false;
}

bool IsImageType(GLenum type)
{
    switch (type)
    {
        case GL_IMAGE_2D:
        case GL_INT_IMAGE_2D:
        case GL_UNSIGNED_INT_IMAGE_2D:
        case GL_IMAGE_3D:
        case GL_INT_IMAGE_3D:
        case GL_UNSIGNED_INT_IMAGE_3D:
        case GL_IMAGE_2D_ARRAY:
        case GL_INT_IMAGE_2D_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
        case GL_IMAGE_CUBE:
        case GL_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_IMAGE_CUBE:
            return true;
    }
    return false;
}

bool IsImage2DType(GLenum type)
{
    switch (type)
    {
        case GL_IMAGE_2D:
        case GL_INT_IMAGE_2D:
        case GL_UNSIGNED_INT_IMAGE_2D:
            return true;
        case GL_IMAGE_3D:
        case GL_INT_IMAGE_3D:
        case GL_UNSIGNED_INT_IMAGE_3D:
        case GL_IMAGE_2D_ARRAY:
        case GL_INT_IMAGE_2D_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
        case GL_IMAGE_CUBE:
        case GL_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_IMAGE_CUBE:
            return false;
        default:
            UNREACHABLE();
            return false;
    }
}

bool IsAtomicCounterType(GLenum type)
{
    return type == GL_UNSIGNED_INT_ATOMIC_COUNTER;
}

bool IsOpaqueType(GLenum type)
{
    // ESSL 3.10 section 4.1.7 defines opaque types as: samplers, images and atomic counters.
    return IsImageType(type) || IsSamplerType(type) || IsAtomicCounterType(type);
}

bool IsMatrixType(GLenum type)
{
    return VariableRowCount(type) > 1;
}

GLenum TransposeMatrixType(GLenum type)
{
    if (!IsMatrixType(type))
    {
        return type;
    }

    switch (type)
    {
        case GL_FLOAT_MAT2:
            return GL_FLOAT_MAT2;
        case GL_FLOAT_MAT3:
            return GL_FLOAT_MAT3;
        case GL_FLOAT_MAT4:
            return GL_FLOAT_MAT4;
        case GL_FLOAT_MAT2x3:
            return GL_FLOAT_MAT3x2;
        case GL_FLOAT_MAT3x2:
            return GL_FLOAT_MAT2x3;
        case GL_FLOAT_MAT2x4:
            return GL_FLOAT_MAT4x2;
        case GL_FLOAT_MAT4x2:
            return GL_FLOAT_MAT2x4;
        case GL_FLOAT_MAT3x4:
            return GL_FLOAT_MAT4x3;
        case GL_FLOAT_MAT4x3:
            return GL_FLOAT_MAT3x4;
        default:
            UNREACHABLE();
            return GL_NONE;
    }
}

int MatrixRegisterCount(GLenum type, bool isRowMajorMatrix)
{
    ASSERT(IsMatrixType(type));
    return isRowMajorMatrix ? VariableRowCount(type) : VariableColumnCount(type);
}

int MatrixComponentCount(GLenum type, bool isRowMajorMatrix)
{
    ASSERT(IsMatrixType(type));
    return isRowMajorMatrix ? VariableColumnCount(type) : VariableRowCount(type);
}

int VariableRegisterCount(GLenum type)
{
    return IsMatrixType(type) ? VariableColumnCount(type) : 1;
}

int AllocateFirstFreeBits(unsigned int *bits, unsigned int allocationSize, unsigned int bitsSize)
{
    ASSERT(allocationSize <= bitsSize);

    unsigned int mask = std::numeric_limits<unsigned int>::max() >>
                        (std::numeric_limits<unsigned int>::digits - allocationSize);

    for (unsigned int i = 0; i < bitsSize - allocationSize + 1; i++)
    {
        if ((*bits & mask) == 0)
        {
            *bits |= mask;
            return i;
        }

        mask <<= 1;
    }

    return -1;
}

IndexRange ComputeIndexRange(DrawElementsType indexType,
                             const GLvoid *indices,
                             size_t count,
                             bool primitiveRestartEnabled)
{
    switch (indexType)
    {
        case DrawElementsType::UnsignedByte:
            return ComputeTypedIndexRange(static_cast<const GLubyte *>(indices), count,
                                          primitiveRestartEnabled,
                                          GetPrimitiveRestartIndex(indexType));
        case DrawElementsType::UnsignedShort:
            return ComputeTypedIndexRange(static_cast<const GLushort *>(indices), count,
                                          primitiveRestartEnabled,
                                          GetPrimitiveRestartIndex(indexType));
        case DrawElementsType::UnsignedInt:
            return ComputeTypedIndexRange(static_cast<const GLuint *>(indices), count,
                                          primitiveRestartEnabled,
                                          GetPrimitiveRestartIndex(indexType));
        default:
            UNREACHABLE();
            return IndexRange();
    }
}

GLuint GetPrimitiveRestartIndex(DrawElementsType indexType)
{
    switch (indexType)
    {
        case DrawElementsType::UnsignedByte:
            return 0xFF;
        case DrawElementsType::UnsignedShort:
            return 0xFFFF;
        case DrawElementsType::UnsignedInt:
            return 0xFFFFFFFF;
        default:
            UNREACHABLE();
            return 0;
    }
}

bool IsTriangleMode(PrimitiveMode drawMode)
{
    switch (drawMode)
    {
        case PrimitiveMode::Triangles:
        case PrimitiveMode::TriangleFan:
        case PrimitiveMode::TriangleStrip:
            return true;
        case PrimitiveMode::Points:
        case PrimitiveMode::Lines:
        case PrimitiveMode::LineLoop:
        case PrimitiveMode::LineStrip:
            return false;
        default:
            UNREACHABLE();
    }

    return false;
}

bool IsPolygonMode(PrimitiveMode mode)
{
    switch (mode)
    {
        case PrimitiveMode::Points:
        case PrimitiveMode::Lines:
        case PrimitiveMode::LineStrip:
        case PrimitiveMode::LineLoop:
        case PrimitiveMode::LinesAdjacency:
        case PrimitiveMode::LineStripAdjacency:
            return false;
        default:
            break;
    }

    return true;
}

namespace priv
{
const angle::PackedEnumMap<PrimitiveMode, bool> gLineModes = {
    {{PrimitiveMode::LineLoop, true},
     {PrimitiveMode::LineStrip, true},
     {PrimitiveMode::LineStripAdjacency, true},
     {PrimitiveMode::Lines, true}}};
}  // namespace priv

bool IsIntegerFormat(GLenum unsizedFormat)
{
    switch (unsizedFormat)
    {
        case GL_RGBA_INTEGER:
        case GL_RGB_INTEGER:
        case GL_RG_INTEGER:
        case GL_RED_INTEGER:
            return true;

        default:
            return false;
    }
}

// [OpenGL ES SL 3.00.4] Section 11 p. 120
// Vertex Outs/Fragment Ins packing priorities
int VariableSortOrder(GLenum type)
{
    switch (type)
    {
        // 1. Arrays of mat4 and mat4
        // Non-square matrices of type matCxR consume the same space as a square
        // matrix of type matN where N is the greater of C and R
        case GL_FLOAT_MAT4:
        case GL_FLOAT_MAT2x4:
        case GL_FLOAT_MAT3x4:
        case GL_FLOAT_MAT4x2:
        case GL_FLOAT_MAT4x3:
            return 0;

        // 2. Arrays of mat2 and mat2 (since they occupy full rows)
        case GL_FLOAT_MAT2:
            return 1;

        // 3. Arrays of vec4 and vec4
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_BOOL_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return 2;

        // 4. Arrays of mat3 and mat3
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT3x2:
            return 3;

        // 5. Arrays of vec3 and vec3
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_BOOL_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return 4;

        // 6. Arrays of vec2 and vec2
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_BOOL_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return 5;

        // 7. Single component types
        case GL_FLOAT:
        case GL_INT:
        case GL_BOOL:
        case GL_UNSIGNED_INT:
        case GL_SAMPLER_2D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_EXTERNAL_OES:
        case GL_SAMPLER_2D_RECT_ANGLE:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_3D:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_IMAGE_2D:
        case GL_INT_IMAGE_2D:
        case GL_UNSIGNED_INT_IMAGE_2D:
        case GL_IMAGE_3D:
        case GL_INT_IMAGE_3D:
        case GL_UNSIGNED_INT_IMAGE_3D:
        case GL_IMAGE_2D_ARRAY:
        case GL_INT_IMAGE_2D_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
        case GL_IMAGE_CUBE:
        case GL_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_ATOMIC_COUNTER:
            return 6;

        default:
            UNREACHABLE();
            return 0;
    }
}

std::string ParseResourceName(const std::string &name, std::vector<unsigned int> *outSubscripts)
{
    if (outSubscripts)
    {
        outSubscripts->clear();
    }
    // Strip any trailing array indexing operators and retrieve the subscripts.
    size_t baseNameLength = name.length();
    bool hasIndex         = true;
    while (hasIndex)
    {
        size_t open  = name.find_last_of('[', baseNameLength - 1);
        size_t close = name.find_last_of(']', baseNameLength - 1);
        hasIndex     = (open != std::string::npos) && (close == baseNameLength - 1);
        if (hasIndex)
        {
            baseNameLength = open;
            if (outSubscripts)
            {
                int index = atoi(name.substr(open + 1).c_str());
                if (index >= 0)
                {
                    outSubscripts->push_back(index);
                }
                else
                {
                    outSubscripts->push_back(GL_INVALID_INDEX);
                }
            }
        }
    }

    return name.substr(0, baseNameLength);
}

std::string StripLastArrayIndex(const std::string &name)
{
    size_t strippedNameLength = name.find_last_of('[');
    if (strippedNameLength != std::string::npos && name.back() == ']')
    {
        return name.substr(0, strippedNameLength);
    }
    return name;
}

bool SamplerNameContainsNonZeroArrayElement(const std::string &name)
{
    constexpr char kZERO_ELEMENT[] = "[0]";

    size_t start = 0;
    while (true)
    {
        start = name.find(kZERO_ELEMENT[0], start);
        if (start == std::string::npos)
        {
            break;
        }
        if (name.compare(start, strlen(kZERO_ELEMENT), kZERO_ELEMENT) != 0)
        {
            return true;
        }
        start++;
    }
    return false;
}

const sh::ShaderVariable *FindShaderVarField(const sh::ShaderVariable &var,
                                             const std::string &fullName,
                                             GLuint *fieldIndexOut)
{
    if (var.fields.empty())
    {
        return nullptr;
    }
    size_t pos = fullName.find_first_of(".");
    if (pos == std::string::npos)
    {
        return nullptr;
    }
    std::string topName = fullName.substr(0, pos);
    if (topName != var.name)
    {
        return nullptr;
    }
    std::string fieldName = fullName.substr(pos + 1);
    if (fieldName.empty())
    {
        return nullptr;
    }
    for (size_t field = 0; field < var.fields.size(); ++field)
    {
        if (var.fields[field].name == fieldName)
        {
            *fieldIndexOut = static_cast<GLuint>(field);
            return &var.fields[field];
        }
    }
    return nullptr;
}

unsigned int ArraySizeProduct(const std::vector<unsigned int> &arraySizes)
{
    unsigned int arraySizeProduct = 1u;
    for (unsigned int arraySize : arraySizes)
    {
        arraySizeProduct *= arraySize;
    }
    return arraySizeProduct;
}

unsigned int ParseArrayIndex(const std::string &name, size_t *nameLengthWithoutArrayIndexOut)
{
    ASSERT(nameLengthWithoutArrayIndexOut != nullptr);

    // Strip any trailing array operator and retrieve the subscript
    size_t open = name.find_last_of('[');
    if (open != std::string::npos && name.back() == ']')
    {
        bool indexIsValidDecimalNumber = true;
        for (size_t i = open + 1; i < name.length() - 1u; ++i)
        {
            if (!isdigit(name[i]))
            {
                indexIsValidDecimalNumber = false;
                break;
            }
        }
        if (indexIsValidDecimalNumber)
        {
            errno = 0;  // reset global error flag.
            unsigned long subscript =
                strtoul(name.c_str() + open + 1, /*endptr*/ nullptr, /*radix*/ 10);

            // Check if resulting integer is out-of-range or conversion error.
            if ((subscript <= static_cast<unsigned long>(UINT_MAX)) &&
                !(subscript == ULONG_MAX && errno == ERANGE) && !(errno != 0 && subscript == 0))
            {
                *nameLengthWithoutArrayIndexOut = open;
                return static_cast<unsigned int>(subscript);
            }
        }
    }

    *nameLengthWithoutArrayIndexOut = name.length();
    return GL_INVALID_INDEX;
}

const char *GetGenericErrorMessage(GLenum error)
{
    switch (error)
    {
        case GL_NO_ERROR:
            return "";
        case GL_INVALID_ENUM:
            return "Invalid enum.";
        case GL_INVALID_VALUE:
            return "Invalid value.";
        case GL_INVALID_OPERATION:
            return "Invalid operation.";
        case GL_STACK_OVERFLOW:
            return "Stack overflow.";
        case GL_STACK_UNDERFLOW:
            return "Stack underflow.";
        case GL_OUT_OF_MEMORY:
            return "Out of memory.";
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            return "Invalid framebuffer operation.";
        default:
            UNREACHABLE();
            return "Unknown error.";
    }
}

unsigned int ElementTypeSize(GLenum elementType)
{
    switch (elementType)
    {
        case GL_UNSIGNED_BYTE:
            return sizeof(GLubyte);
        case GL_UNSIGNED_SHORT:
            return sizeof(GLushort);
        case GL_UNSIGNED_INT:
            return sizeof(GLuint);
        default:
            UNREACHABLE();
            return 0;
    }
}

PipelineType GetPipelineType(ShaderType type)
{
    switch (type)
    {
        case ShaderType::Vertex:
        case ShaderType::Fragment:
        case ShaderType::Geometry:
            return PipelineType::GraphicsPipeline;
        case ShaderType::Compute:
            return PipelineType::ComputePipeline;
        default:
            UNREACHABLE();
            return PipelineType::GraphicsPipeline;
    }
}

}  // namespace gl

namespace egl
{
static_assert(EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_X_KHR - EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR == 1,
              "Unexpected EGL cube map enum value.");
static_assert(EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Y_KHR - EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR == 2,
              "Unexpected EGL cube map enum value.");
static_assert(EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_KHR - EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR == 3,
              "Unexpected EGL cube map enum value.");
static_assert(EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Z_KHR - EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR == 4,
              "Unexpected EGL cube map enum value.");
static_assert(EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_KHR - EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR == 5,
              "Unexpected EGL cube map enum value.");

bool IsCubeMapTextureTarget(EGLenum target)
{
    return (target >= FirstCubeMapTextureTarget && target <= LastCubeMapTextureTarget);
}

size_t CubeMapTextureTargetToLayerIndex(EGLenum target)
{
    ASSERT(IsCubeMapTextureTarget(target));
    return target - static_cast<size_t>(FirstCubeMapTextureTarget);
}

EGLenum LayerIndexToCubeMapTextureTarget(size_t index)
{
    ASSERT(index <= (LastCubeMapTextureTarget - FirstCubeMapTextureTarget));
    return FirstCubeMapTextureTarget + static_cast<GLenum>(index);
}

bool IsTextureTarget(EGLenum target)
{
    switch (target)
    {
        case EGL_GL_TEXTURE_2D_KHR:
        case EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR:
        case EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_X_KHR:
        case EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Y_KHR:
        case EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_KHR:
        case EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Z_KHR:
        case EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_KHR:
        case EGL_GL_TEXTURE_3D_KHR:
            return true;

        default:
            return false;
    }
}

bool IsRenderbufferTarget(EGLenum target)
{
    return target == EGL_GL_RENDERBUFFER_KHR;
}

bool IsExternalImageTarget(EGLenum target)
{
    switch (target)
    {
        case EGL_NATIVE_BUFFER_ANDROID:
        case EGL_D3D11_TEXTURE_ANGLE:
        case EGL_MTL_TEXTURE_MGL:
            return true;

        default:
            return false;
    }
}

const char *GetGenericErrorMessage(EGLint error)
{
    switch (error)
    {
        case EGL_SUCCESS:
            return "";
        case EGL_NOT_INITIALIZED:
            return "Not initialized.";
        case EGL_BAD_ACCESS:
            return "Bad access.";
        case EGL_BAD_ALLOC:
            return "Bad allocation.";
        case EGL_BAD_ATTRIBUTE:
            return "Bad attribute.";
        case EGL_BAD_CONFIG:
            return "Bad config.";
        case EGL_BAD_CONTEXT:
            return "Bad context.";
        case EGL_BAD_CURRENT_SURFACE:
            return "Bad current surface.";
        case EGL_BAD_DISPLAY:
            return "Bad display.";
        case EGL_BAD_MATCH:
            return "Bad match.";
        case EGL_BAD_NATIVE_WINDOW:
            return "Bad native window.";
        case EGL_BAD_PARAMETER:
            return "Bad parameter.";
        case EGL_BAD_SURFACE:
            return "Bad surface.";
        case EGL_CONTEXT_LOST:
            return "Context lost.";
        case EGL_BAD_STREAM_KHR:
            return "Bad stream.";
        case EGL_BAD_STATE_KHR:
            return "Bad state.";
        case EGL_BAD_DEVICE_EXT:
            return "Bad device.";
        default:
            UNREACHABLE();
            return "Unknown error.";
    }
}

}  // namespace egl

namespace egl_gl
{
GLuint EGLClientBufferToGLObjectHandle(EGLClientBuffer buffer)
{
    return static_cast<GLuint>(reinterpret_cast<uintptr_t>(buffer));
}
}  // namespace egl_gl

namespace gl_egl
{
EGLenum GLComponentTypeToEGLColorComponentType(GLenum glComponentType)
{
    switch (glComponentType)
    {
        case GL_FLOAT:
            return EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT;

        case GL_UNSIGNED_NORMALIZED:
            return EGL_COLOR_COMPONENT_TYPE_FIXED_EXT;

        default:
            UNREACHABLE();
            return EGL_NONE;
    }
}

EGLClientBuffer GLObjectHandleToEGLClientBuffer(GLuint handle)
{
    return reinterpret_cast<EGLClientBuffer>(static_cast<uintptr_t>(handle));
}

}  // namespace gl_egl

#if !defined(ANGLE_ENABLE_WINDOWS_UWP)
std::string getTempPath()
{
#    ifdef ANGLE_PLATFORM_WINDOWS
    char path[MAX_PATH];
    DWORD pathLen = GetTempPathA(sizeof(path) / sizeof(path[0]), path);
    if (pathLen == 0)
    {
        UNREACHABLE();
        return std::string();
    }

    UINT unique = GetTempFileNameA(path, "sh", 0, path);
    if (unique == 0)
    {
        UNREACHABLE();
        return std::string();
    }

    return path;
#    else
    UNIMPLEMENTED();
    return "";
#    endif
}

void writeFile(const char *path, const void *content, size_t size)
{
    FILE *file = fopen(path, "w");
    if (!file)
    {
        UNREACHABLE();
        return;
    }

    fwrite(content, sizeof(char), size, file);
    fclose(file);
}
#endif  // !ANGLE_ENABLE_WINDOWS_UWP

#if defined(ANGLE_PLATFORM_WINDOWS)

// Causes the thread to relinquish the remainder of its time slice to any
// other thread that is ready to run.If there are no other threads ready
// to run, the function returns immediately, and the thread continues execution.
void ScheduleYield()
{
    Sleep(0);
}

#endif
