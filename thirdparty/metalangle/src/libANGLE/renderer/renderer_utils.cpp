//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// renderer_utils:
//   Helper methods pertaining to most or all back-ends.
//

#include "libANGLE/renderer/renderer_utils.h"

#include "image_util/copyimage.h"
#include "image_util/imageformats.h"

#include "libANGLE/AttributeMap.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/ContextImpl.h"
#include "libANGLE/renderer/Format.h"

#include "platform/Feature.h"

#include <string.h>
#include "common/utilities.h"

namespace rx
{

namespace
{
void CopyColor(gl::ColorF *color)
{
    // No-op
}

void PremultiplyAlpha(gl::ColorF *color)
{
    color->red *= color->alpha;
    color->green *= color->alpha;
    color->blue *= color->alpha;
}

void UnmultiplyAlpha(gl::ColorF *color)
{
    if (color->alpha != 0.0f)
    {
        float invAlpha = 1.0f / color->alpha;
        color->red *= invAlpha;
        color->green *= invAlpha;
        color->blue *= invAlpha;
    }
}

void ClipChannelsR(gl::ColorF *color)
{
    color->green = 0.0f;
    color->blue  = 0.0f;
    color->alpha = 1.0f;
}

void ClipChannelsRG(gl::ColorF *color)
{
    color->blue  = 0.0f;
    color->alpha = 1.0f;
}

void ClipChannelsRGB(gl::ColorF *color)
{
    color->alpha = 1.0f;
}

void ClipChannelsLuminance(gl::ColorF *color)
{
    color->alpha = 1.0f;
}

void ClipChannelsAlpha(gl::ColorF *color)
{
    color->red   = 0.0f;
    color->green = 0.0f;
    color->blue  = 0.0f;
}

void ClipChannelsNoOp(gl::ColorF *color) {}

void WriteUintColor(const gl::ColorF &color,
                    PixelWriteFunction colorWriteFunction,
                    uint8_t *destPixelData)
{
    gl::ColorUI destColor(
        static_cast<unsigned int>(color.red * 255), static_cast<unsigned int>(color.green * 255),
        static_cast<unsigned int>(color.blue * 255), static_cast<unsigned int>(color.alpha * 255));
    colorWriteFunction(reinterpret_cast<const uint8_t *>(&destColor), destPixelData);
}

void WriteFloatColor(const gl::ColorF &color,
                     PixelWriteFunction colorWriteFunction,
                     uint8_t *destPixelData)
{
    colorWriteFunction(reinterpret_cast<const uint8_t *>(&color), destPixelData);
}

template <int cols, int rows, bool IsColumnMajor>
inline int GetFlattenedIndex(int col, int row)
{
    if (IsColumnMajor)
    {
        return col * rows + row;
    }
    else
    {
        return row * cols + col;
    }
}

template <typename T,
          bool IsSrcColumnMajor,
          int colsSrc,
          int rowsSrc,
          bool IsDstColumnMajor,
          int colsDst,
          int rowsDst>
void ExpandMatrix(T *target, const GLfloat *value)
{
    static_assert(colsSrc <= colsDst && rowsSrc <= rowsDst, "Can only expand!");

    constexpr int kDstFlatSize = colsDst * rowsDst;
    T staging[kDstFlatSize]    = {0};

    for (int r = 0; r < rowsSrc; r++)
    {
        for (int c = 0; c < colsSrc; c++)
        {
            int srcIndex = GetFlattenedIndex<colsSrc, rowsSrc, IsSrcColumnMajor>(c, r);
            int dstIndex = GetFlattenedIndex<colsDst, rowsDst, IsDstColumnMajor>(c, r);

            staging[dstIndex] = static_cast<T>(value[srcIndex]);
        }
    }

    memcpy(target, staging, kDstFlatSize * sizeof(T));
}

template <bool IsSrcColumMajor,
          int colsSrc,
          int rowsSrc,
          bool IsDstColumnMajor,
          int colsDst,
          int rowsDst>
void SetFloatUniformMatrix(unsigned int arrayElementOffset,
                           unsigned int elementCount,
                           GLsizei countIn,
                           const GLfloat *value,
                           uint8_t *targetData)
{
    unsigned int count =
        std::min(elementCount - arrayElementOffset, static_cast<unsigned int>(countIn));

    const unsigned int targetMatrixStride = colsDst * rowsDst;
    GLfloat *target                       = reinterpret_cast<GLfloat *>(
        targetData + arrayElementOffset * sizeof(GLfloat) * targetMatrixStride);

    for (unsigned int i = 0; i < count; i++)
    {
        ExpandMatrix<GLfloat, IsSrcColumMajor, colsSrc, rowsSrc, IsDstColumnMajor, colsDst,
                     rowsDst>(target, value);

        target += targetMatrixStride;
        value += colsSrc * rowsSrc;
    }
}

void SetFloatUniformMatrixFast(unsigned int arrayElementOffset,
                               unsigned int elementCount,
                               GLsizei countIn,
                               size_t matrixSize,
                               const GLfloat *value,
                               uint8_t *targetData)
{
    const unsigned int count =
        std::min(elementCount - arrayElementOffset, static_cast<unsigned int>(countIn));

    const uint8_t *valueData = reinterpret_cast<const uint8_t *>(value);
    targetData               = targetData + arrayElementOffset * matrixSize;

    memcpy(targetData, valueData, matrixSize * count);
}

}  // anonymous namespace

PackPixelsParams::PackPixelsParams()
    : destFormat(nullptr), outputPitch(0), packBuffer(nullptr), offset(0)
{}

PackPixelsParams::PackPixelsParams(const gl::Rectangle &areaIn,
                                   const angle::Format &destFormat,
                                   GLuint outputPitchIn,
                                   bool reverseRowOrderIn,
                                   gl::Buffer *packBufferIn,
                                   ptrdiff_t offsetIn)
    : area(areaIn),
      destFormat(&destFormat),
      outputPitch(outputPitchIn),
      packBuffer(packBufferIn),
      reverseRowOrder(reverseRowOrderIn),
      offset(offsetIn)
{}

void PackPixels(const PackPixelsParams &params,
                const angle::Format &sourceFormat,
                int inputPitchIn,
                const uint8_t *sourceIn,
                uint8_t *destWithoutOffset)
{
    uint8_t *destWithOffset = destWithoutOffset + params.offset;

    const uint8_t *source = sourceIn;
    int inputPitch        = inputPitchIn;

    if (params.reverseRowOrder)
    {
        source += inputPitch * (params.area.height - 1);
        inputPitch = -inputPitch;
    }

    if (sourceFormat == *params.destFormat)
    {
        // Direct copy possible
        for (int y = 0; y < params.area.height; ++y)
        {
            memcpy(destWithOffset + y * params.outputPitch, source + y * inputPitch,
                   params.area.width * sourceFormat.pixelBytes);
        }
        return;
    }

    PixelCopyFunction fastCopyFunc = sourceFormat.fastCopyFunctions.get(params.destFormat->id);

    if (fastCopyFunc)
    {
        // Fast copy is possible through some special function
        for (int y = 0; y < params.area.height; ++y)
        {
            for (int x = 0; x < params.area.width; ++x)
            {
                uint8_t *dest =
                    destWithOffset + y * params.outputPitch + x * params.destFormat->pixelBytes;
                const uint8_t *src = source + y * inputPitch + x * sourceFormat.pixelBytes;

                fastCopyFunc(src, dest);
            }
        }
        return;
    }

    PixelWriteFunction pixelWriteFunction = params.destFormat->pixelWriteFunction;
    ASSERT(pixelWriteFunction != nullptr);

    // Maximum size of any Color<T> type used.
    uint8_t temp[16];
    static_assert(sizeof(temp) >= sizeof(gl::ColorF) && sizeof(temp) >= sizeof(gl::ColorUI) &&
                      sizeof(temp) >= sizeof(gl::ColorI) &&
                      sizeof(temp) >= sizeof(angle::DepthStencil),
                  "Unexpected size of pixel struct.");

    PixelReadFunction pixelReadFunction = sourceFormat.pixelReadFunction;
    ASSERT(pixelReadFunction != nullptr);

    for (int y = 0; y < params.area.height; ++y)
    {
        for (int x = 0; x < params.area.width; ++x)
        {
            uint8_t *dest =
                destWithOffset + y * params.outputPitch + x * params.destFormat->pixelBytes;
            const uint8_t *src = source + y * inputPitch + x * sourceFormat.pixelBytes;

            // readFunc and writeFunc will be using the same type of color, CopyTexImage
            // will not allow the copy otherwise.
            pixelReadFunction(src, temp);
            pixelWriteFunction(temp, dest);
        }
    }
}

bool FastCopyFunctionMap::has(angle::FormatID formatID) const
{
    return (get(formatID) != nullptr);
}

PixelCopyFunction FastCopyFunctionMap::get(angle::FormatID formatID) const
{
    for (size_t index = 0; index < mSize; ++index)
    {
        if (mData[index].formatID == formatID)
        {
            return mData[index].func;
        }
    }

    return nullptr;
}

bool ShouldUseDebugLayers(const egl::AttributeMap &attribs)
{
    EGLAttrib debugSetting =
        attribs.get(EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED_ANGLE, EGL_DONT_CARE);

    // Prefer to enable debug layers when available.
#if defined(ANGLE_ENABLE_VULKAN_VALIDATION_LAYERS_BY_DEFAULT)
    return (debugSetting != EGL_FALSE);
#else
    return (debugSetting == EGL_TRUE);
#endif  // defined(ANGLE_ENABLE_VULKAN_VALIDATION_LAYERS_BY_DEFAULT)
}

bool ShouldUseVirtualizedContexts(const egl::AttributeMap &attribs, bool defaultValue)
{
    EGLAttrib virtualizedContextRequest =
        attribs.get(EGL_PLATFORM_ANGLE_CONTEXT_VIRTUALIZATION_ANGLE, EGL_DONT_CARE);
    if (defaultValue)
    {
        return (virtualizedContextRequest != EGL_FALSE);
    }
    else
    {
        return (virtualizedContextRequest == EGL_TRUE);
    }
}

void CopyImageCHROMIUM(const uint8_t *sourceData,
                       size_t sourceRowPitch,
                       size_t sourcePixelBytes,
                       size_t sourceDepthPitch,
                       PixelReadFunction pixelReadFunction,
                       uint8_t *destData,
                       size_t destRowPitch,
                       size_t destPixelBytes,
                       size_t destDepthPitch,
                       PixelWriteFunction pixelWriteFunction,
                       GLenum destUnsizedFormat,
                       GLenum destComponentType,
                       size_t width,
                       size_t height,
                       size_t depth,
                       bool unpackFlipY,
                       bool unpackPremultiplyAlpha,
                       bool unpackUnmultiplyAlpha)
{
    using ConversionFunction              = void (*)(gl::ColorF *);
    ConversionFunction conversionFunction = CopyColor;
    if (unpackPremultiplyAlpha != unpackUnmultiplyAlpha)
    {
        if (unpackPremultiplyAlpha)
        {
            conversionFunction = PremultiplyAlpha;
        }
        else
        {
            conversionFunction = UnmultiplyAlpha;
        }
    }

    auto clipChannelsFunction = ClipChannelsNoOp;
    switch (destUnsizedFormat)
    {
        case GL_RED:
            clipChannelsFunction = ClipChannelsR;
            break;
        case GL_RG:
            clipChannelsFunction = ClipChannelsRG;
            break;
        case GL_RGB:
            clipChannelsFunction = ClipChannelsRGB;
            break;
        case GL_LUMINANCE:
            clipChannelsFunction = ClipChannelsLuminance;
            break;
        case GL_ALPHA:
            clipChannelsFunction = ClipChannelsAlpha;
            break;
    }

    auto writeFunction = (destComponentType == GL_UNSIGNED_INT) ? WriteUintColor : WriteFloatColor;

    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                const uint8_t *sourcePixelData =
                    sourceData + y * sourceRowPitch + x * sourcePixelBytes + z * sourceDepthPitch;

                gl::ColorF sourceColor;
                pixelReadFunction(sourcePixelData, reinterpret_cast<uint8_t *>(&sourceColor));

                conversionFunction(&sourceColor);
                clipChannelsFunction(&sourceColor);

                size_t destY = 0;
                if (unpackFlipY)
                {
                    destY += (height - 1);
                    destY -= y;
                }
                else
                {
                    destY += y;
                }

                uint8_t *destPixelData =
                    destData + destY * destRowPitch + x * destPixelBytes + z * destDepthPitch;
                writeFunction(sourceColor, pixelWriteFunction, destPixelData);
            }
        }
    }
}

// IncompleteTextureSet implementation.
IncompleteTextureSet::IncompleteTextureSet() {}

IncompleteTextureSet::~IncompleteTextureSet() {}

void IncompleteTextureSet::onDestroy(const gl::Context *context)
{
    // Clear incomplete textures.
    for (auto &incompleteTexture : mIncompleteTextures)
    {
        if (incompleteTexture.get() != nullptr)
        {
            incompleteTexture->onDestroy(context);
            incompleteTexture.set(context, nullptr);
        }
    }
}

angle::Result IncompleteTextureSet::getIncompleteTexture(
    const gl::Context *context,
    gl::TextureType type,
    MultisampleTextureInitializer *multisampleInitializer,
    gl::Texture **textureOut)
{
    *textureOut = mIncompleteTextures[type].get();
    if (*textureOut != nullptr)
    {
        return angle::Result::Continue;
    }

    ContextImpl *implFactory = context->getImplementation();

    const GLubyte color[] = {0, 0, 0, 255};
    const gl::Extents colorSize(1, 1, 1);
    gl::PixelUnpackState unpack;
    unpack.alignment = 1;
    const gl::Box area(0, 0, 0, 1, 1, 1);

    // If a texture is external use a 2D texture for the incomplete texture
    gl::TextureType createType = (type == gl::TextureType::External) ? gl::TextureType::_2D : type;

    gl::Texture *tex =
        new gl::Texture(implFactory, {std::numeric_limits<GLuint>::max()}, createType);
    angle::UniqueObjectPointer<gl::Texture, gl::Context> t(tex, context);

    // This is a bit of a kludge but is necessary to consume the error.
    gl::Context *mutableContext = const_cast<gl::Context *>(context);

    if (createType == gl::TextureType::_2DMultisample)
    {
        ANGLE_TRY(
            t->setStorageMultisample(mutableContext, createType, 1, GL_RGBA8, colorSize, true));
    }
    else
    {
        ANGLE_TRY(t->setStorage(mutableContext, createType, 1, GL_RGBA8, colorSize));
    }

    if (type == gl::TextureType::CubeMap)
    {
        for (gl::TextureTarget face : gl::AllCubeFaceTextureTargets())
        {
            ANGLE_TRY(t->setSubImage(mutableContext, unpack, nullptr, face, 0, area, GL_RGBA,
                                     GL_UNSIGNED_BYTE, color));
        }
    }
    else if (type == gl::TextureType::_2DMultisample)
    {
        // Call a specialized clear function to init a multisample texture.
        ANGLE_TRY(multisampleInitializer->initializeMultisampleTextureToBlack(context, t.get()));
    }
    else
    {
        ANGLE_TRY(t->setSubImage(mutableContext, unpack, nullptr,
                                 gl::NonCubeTextureTypeToTarget(createType), 0, area, GL_RGBA,
                                 GL_UNSIGNED_BYTE, color));
    }

    ANGLE_TRY(t->syncState(context));

    mIncompleteTextures[type].set(context, t.release());
    *textureOut = mIncompleteTextures[type].get();
    return angle::Result::Continue;
}

#define ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(api, cols, rows) \
    template void SetFloatUniformMatrix##api<cols, rows>::Run(     \
        unsigned int, unsigned int, GLsizei, GLboolean, const GLfloat *, uint8_t *)

ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(GLSL, 2, 2);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(GLSL, 3, 3);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(GLSL, 2, 3);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(GLSL, 3, 2);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(GLSL, 4, 2);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(GLSL, 4, 3);

ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(HLSL, 2, 2);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(HLSL, 3, 3);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(HLSL, 2, 3);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(HLSL, 3, 2);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(HLSL, 2, 4);
ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC(HLSL, 3, 4);

#undef ANGLE_INSTANTIATE_SET_UNIFORM_MATRIX_FUNC

#define ANGLE_SPECIALIZATION_ROWS_SET_UNIFORM_MATRIX_FUNC(api, cols, rows)                      \
    template void SetFloatUniformMatrix##api<cols, 4>::Run(unsigned int, unsigned int, GLsizei, \
                                                           GLboolean, const GLfloat *, uint8_t *)

template <int cols>
struct SetFloatUniformMatrixGLSL<cols, 4>
{
    static void Run(unsigned int arrayElementOffset,
                    unsigned int elementCount,
                    GLsizei countIn,
                    GLboolean transpose,
                    const GLfloat *value,
                    uint8_t *targetData);
};

ANGLE_SPECIALIZATION_ROWS_SET_UNIFORM_MATRIX_FUNC(GLSL, 2, 4);
ANGLE_SPECIALIZATION_ROWS_SET_UNIFORM_MATRIX_FUNC(GLSL, 3, 4);
ANGLE_SPECIALIZATION_ROWS_SET_UNIFORM_MATRIX_FUNC(GLSL, 4, 4);

#undef ANGLE_SPECIALIZATION_ROWS_SET_UNIFORM_MATRIX_FUNC

#define ANGLE_SPECIALIZATION_COLS_SET_UNIFORM_MATRIX_FUNC(api, cols, rows)                      \
    template void SetFloatUniformMatrix##api<4, rows>::Run(unsigned int, unsigned int, GLsizei, \
                                                           GLboolean, const GLfloat *, uint8_t *)

template <int rows>
struct SetFloatUniformMatrixHLSL<4, rows>
{
    static void Run(unsigned int arrayElementOffset,
                    unsigned int elementCount,
                    GLsizei countIn,
                    GLboolean transpose,
                    const GLfloat *value,
                    uint8_t *targetData);
};

ANGLE_SPECIALIZATION_COLS_SET_UNIFORM_MATRIX_FUNC(HLSL, 4, 2);
ANGLE_SPECIALIZATION_COLS_SET_UNIFORM_MATRIX_FUNC(HLSL, 4, 3);
ANGLE_SPECIALIZATION_COLS_SET_UNIFORM_MATRIX_FUNC(HLSL, 4, 4);

#undef ANGLE_SPECIALIZATION_COLS_SET_UNIFORM_MATRIX_FUNC

template <int cols>
void SetFloatUniformMatrixGLSL<cols, 4>::Run(unsigned int arrayElementOffset,
                                             unsigned int elementCount,
                                             GLsizei countIn,
                                             GLboolean transpose,
                                             const GLfloat *value,
                                             uint8_t *targetData)
{
    const bool isSrcColumnMajor = !transpose;
    if (isSrcColumnMajor)
    {
        // Both src and dst matrixs are has same layout,
        // a single memcpy updates all the matrices
        constexpr size_t srcMatrixSize = sizeof(GLfloat) * cols * 4;
        SetFloatUniformMatrixFast(arrayElementOffset, elementCount, countIn, srcMatrixSize, value,
                                  targetData);
    }
    else
    {
        // fallback to general cases
        SetFloatUniformMatrix<false, cols, 4, true, cols, 4>(arrayElementOffset, elementCount,
                                                             countIn, value, targetData);
    }
}

template <int cols, int rows>
void SetFloatUniformMatrixGLSL<cols, rows>::Run(unsigned int arrayElementOffset,
                                                unsigned int elementCount,
                                                GLsizei countIn,
                                                GLboolean transpose,
                                                const GLfloat *value,
                                                uint8_t *targetData)
{
    const bool isSrcColumnMajor = !transpose;
    // GLSL expects matrix uniforms to be column-major, and each column is padded to 4 rows.
    if (isSrcColumnMajor)
    {
        SetFloatUniformMatrix<true, cols, rows, true, cols, 4>(arrayElementOffset, elementCount,
                                                               countIn, value, targetData);
    }
    else
    {
        SetFloatUniformMatrix<false, cols, rows, true, cols, 4>(arrayElementOffset, elementCount,
                                                                countIn, value, targetData);
    }
}

template <int rows>
void SetFloatUniformMatrixHLSL<4, rows>::Run(unsigned int arrayElementOffset,
                                             unsigned int elementCount,
                                             GLsizei countIn,
                                             GLboolean transpose,
                                             const GLfloat *value,
                                             uint8_t *targetData)
{
    const bool isSrcColumnMajor = !transpose;
    if (!isSrcColumnMajor)
    {
        // Both src and dst matrixs are has same layout,
        // a single memcpy updates all the matrices
        constexpr size_t srcMatrixSize = sizeof(GLfloat) * 4 * rows;
        SetFloatUniformMatrixFast(arrayElementOffset, elementCount, countIn, srcMatrixSize, value,
                                  targetData);
    }
    else
    {
        // fallback to general cases
        SetFloatUniformMatrix<true, 4, rows, false, 4, rows>(arrayElementOffset, elementCount,
                                                             countIn, value, targetData);
    }
}

template <int cols, int rows>
void SetFloatUniformMatrixHLSL<cols, rows>::Run(unsigned int arrayElementOffset,
                                                unsigned int elementCount,
                                                GLsizei countIn,
                                                GLboolean transpose,
                                                const GLfloat *value,
                                                uint8_t *targetData)
{
    const bool isSrcColumnMajor = !transpose;
    // Internally store matrices as row-major to accomodate HLSL matrix indexing.  Each row is
    // padded to 4 columns.
    if (!isSrcColumnMajor)
    {
        SetFloatUniformMatrix<false, cols, rows, false, 4, rows>(arrayElementOffset, elementCount,
                                                                 countIn, value, targetData);
    }
    else
    {
        SetFloatUniformMatrix<true, cols, rows, false, 4, rows>(arrayElementOffset, elementCount,
                                                                countIn, value, targetData);
    }
}

template void GetMatrixUniform<GLint>(GLenum, GLint *, const GLint *, bool);
template void GetMatrixUniform<GLuint>(GLenum, GLuint *, const GLuint *, bool);

void GetMatrixUniform(GLenum type, GLfloat *dataOut, const GLfloat *source, bool transpose)
{
    int columns = gl::VariableColumnCount(type);
    int rows    = gl::VariableRowCount(type);
    for (GLint col = 0; col < columns; ++col)
    {
        for (GLint row = 0; row < rows; ++row)
        {
            GLfloat *outptr = dataOut + ((col * rows) + row);
            const GLfloat *inptr =
                transpose ? source + ((row * 4) + col) : source + ((col * 4) + row);
            *outptr = *inptr;
        }
    }
}

template <typename NonFloatT>
void GetMatrixUniform(GLenum type, NonFloatT *dataOut, const NonFloatT *source, bool transpose)
{
    UNREACHABLE();
}

const angle::Format &GetFormatFromFormatType(GLenum format, GLenum type)
{
    GLenum sizedInternalFormat    = gl::GetInternalFormatInfo(format, type).sizedInternalFormat;
    angle::FormatID angleFormatID = angle::Format::InternalFormatToID(sizedInternalFormat);
    return angle::Format::Get(angleFormatID);
}

angle::Result ComputeStartVertex(ContextImpl *contextImpl,
                                 const gl::IndexRange &indexRange,
                                 GLint baseVertex,
                                 GLint *firstVertexOut)
{
    // The entire index range should be within the limits of a 32-bit uint because the largest
    // GL index type is GL_UNSIGNED_INT.
    ASSERT(indexRange.start <= std::numeric_limits<uint32_t>::max() &&
           indexRange.end <= std::numeric_limits<uint32_t>::max());

    // The base vertex is only used in DrawElementsIndirect. Given the assertion above and the
    // type of mBaseVertex (GLint), adding them both as 64-bit ints is safe.
    int64_t startVertexInt64 =
        static_cast<int64_t>(baseVertex) + static_cast<int64_t>(indexRange.start);

    // OpenGL ES 3.2 spec section 10.5: "Behavior of DrawElementsOneInstance is undefined if the
    // vertex ID is negative for any element"
    ANGLE_CHECK_GL_MATH(contextImpl, startVertexInt64 >= 0);

    // OpenGL ES 3.2 spec section 10.5: "If the vertex ID is larger than the maximum value
    // representable by type, it should behave as if the calculation were upconverted to 32-bit
    // unsigned integers(with wrapping on overflow conditions)." ANGLE does not fully handle
    // these rules, an overflow error is returned if the start vertex cannot be stored in a
    // 32-bit signed integer.
    ANGLE_CHECK_GL_MATH(contextImpl, startVertexInt64 <= std::numeric_limits<GLint>::max());

    *firstVertexOut = static_cast<GLint>(startVertexInt64);
    return angle::Result::Continue;
}

angle::Result GetVertexRangeInfo(const gl::Context *context,
                                 GLint firstVertex,
                                 GLsizei vertexOrIndexCount,
                                 gl::DrawElementsType indexTypeOrInvalid,
                                 const void *indices,
                                 GLint baseVertex,
                                 GLint *startVertexOut,
                                 size_t *vertexCountOut)
{
    if (indexTypeOrInvalid != gl::DrawElementsType::InvalidEnum)
    {
        gl::IndexRange indexRange;
        ANGLE_TRY(context->getState().getVertexArray()->getIndexRange(
            context, indexTypeOrInvalid, vertexOrIndexCount, indices, &indexRange));
        ANGLE_TRY(ComputeStartVertex(context->getImplementation(), indexRange, baseVertex,
                                     startVertexOut));
        *vertexCountOut = indexRange.vertexCount();
    }
    else
    {
        *startVertexOut = firstVertex;
        *vertexCountOut = vertexOrIndexCount;
    }
    return angle::Result::Continue;
}

gl::Rectangle ClipRectToScissor(const gl::State &glState, const gl::Rectangle &rect, bool invertY)
{
    if (glState.isScissorTestEnabled())
    {
        gl::Rectangle clippedRect;
        if (!gl::ClipRectangle(glState.getScissor(), rect, &clippedRect))
        {
            return gl::Rectangle();
        }

        if (invertY)
        {
            clippedRect.y = rect.height - clippedRect.y - clippedRect.height;
        }

        return clippedRect;
    }

    // If the scissor test isn't enabled, assume it has infinite size.  Its intersection with the
    // rect would be the rect itself.
    //
    // Note that on Vulkan, returning this (as opposed to a fixed max-int-sized rect) could lead to
    // unnecessary pipeline creations if two otherwise identical pipelines are used on framebuffers
    // with different sizes.  If such usage is observed in an application, we should investigate
    // possible optimizations.
    return rect;
}

void OverrideFeaturesWithDisplayState(angle::FeatureSetBase *features,
                                      const egl::DisplayState &state)
{
    features->overrideFeatures(state.featureOverridesEnabled, true);
    features->overrideFeatures(state.featureOverridesDisabled, false);
}
}  // namespace rx
