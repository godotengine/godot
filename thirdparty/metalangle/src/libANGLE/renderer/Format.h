//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Format:
//   A universal description of typed GPU storage. Across multiple
//   renderer back-ends, there are common formats and some distinct
//   permutations, this enum encapsulates them all. Formats apply to
//   textures, but could also apply to any typed data.

#ifndef LIBANGLE_RENDERER_FORMAT_H_
#define LIBANGLE_RENDERER_FORMAT_H_

#include "libANGLE/renderer/FormatID_autogen.h"
#include "libANGLE/renderer/renderer_utils.h"

namespace angle
{
enum class FormatID;

extern const Format gFormatInfoTable[];

struct Format final : private angle::NonCopyable
{
    inline constexpr Format(FormatID id,
                            GLenum glFormat,
                            GLenum fboFormat,
                            rx::MipGenerationFunction mipGen,
                            const rx::FastCopyFunctionMap &fastCopyFunctions,
                            rx::PixelReadFunction colorRead,
                            rx::PixelWriteFunction colorWrite,
                            GLenum componentType,
                            GLuint redBits,
                            GLuint greenBits,
                            GLuint blueBits,
                            GLuint alphaBits,
                            GLuint luminanceBits,
                            GLuint depthBits,
                            GLuint stencilBits,
                            GLuint pixelBytes,
                            GLuint componentAlignmentMask,
                            bool isBlock,
                            bool isFixed,
                            bool isScaled,
                            gl::VertexAttribType vertexAttribType);

    static const Format &Get(FormatID id) { return gFormatInfoTable[static_cast<int>(id)]; }

    static FormatID InternalFormatToID(GLenum internalFormat);

    constexpr bool hasDepthOrStencilBits() const;
    constexpr bool isLUMA() const;

    constexpr bool isSint() const;
    constexpr bool isUint() const;
    constexpr bool isSnorm() const;
    constexpr bool isUnorm() const;
    constexpr bool isFloat() const;
    constexpr bool isVertexTypeHalfFloat() const;

    constexpr bool isInt() const { return isSint() || isUint(); }
    constexpr bool isNorm() const { return isSnorm() || isUnorm(); }
    constexpr bool isPureInt() const { return isInt() && !isScaled; }

    bool operator==(const Format &other) const { return this->id == other.id; }

    FormatID id;

    // The closest matching GL internal format for the storage this format uses. Note that this
    // may be a different internal format than the one this ANGLE format is used for.
    GLenum glInternalFormat;

    // The format we should report to the GL layer when querying implementation formats from a FBO.
    // This might not be the same as the glInternalFormat, since some DXGI formats don't have
    // matching GL format enums, like BGRA4, BGR5A1 and B5G6R6.
    GLenum fboImplementationInternalFormat;

    rx::MipGenerationFunction mipGenerationFunction;
    rx::PixelReadFunction pixelReadFunction;
    rx::PixelWriteFunction pixelWriteFunction;

    // A map from a gl::FormatType to a fast pixel copy function for this format.
    const rx::FastCopyFunctionMap &fastCopyFunctions;

    GLenum componentType;

    GLuint redBits;
    GLuint greenBits;
    GLuint blueBits;
    GLuint alphaBits;
    GLuint luminanceBits;
    GLuint depthBits;
    GLuint stencilBits;

    GLuint pixelBytes;

    // For 1-byte components, is 0x0. For 2-byte, is 0x1. For 4-byte, is 0x3. For all others,
    // MAX_UINT.
    GLuint componentAlignmentMask;

    GLuint channelCount;

    bool isBlock;
    bool isFixed;
    bool isScaled;

    // For vertex formats only. Returns the "type" value for glVertexAttribPointer etc.
    gl::VertexAttribType vertexAttribType;
};

constexpr GLuint GetChannelCount(GLuint redBits,
                                 GLuint greenBits,
                                 GLuint blueBits,
                                 GLuint alphaBits,
                                 GLuint luminanceBits,
                                 GLuint depthBits,
                                 GLuint stencilBits)
{
    return (redBits > 0 ? 1 : 0) + (greenBits > 0 ? 1 : 0) + (blueBits > 0 ? 1 : 0) +
           (alphaBits > 0 ? 1 : 0) + (luminanceBits > 0 ? 1 : 0) + (depthBits > 0 ? 1 : 0) +
           (stencilBits > 0 ? 1 : 0);
}

constexpr Format::Format(FormatID id,
                         GLenum glFormat,
                         GLenum fboFormat,
                         rx::MipGenerationFunction mipGen,
                         const rx::FastCopyFunctionMap &fastCopyFunctions,
                         rx::PixelReadFunction colorRead,
                         rx::PixelWriteFunction colorWrite,
                         GLenum componentType,
                         GLuint redBits,
                         GLuint greenBits,
                         GLuint blueBits,
                         GLuint alphaBits,
                         GLuint luminanceBits,
                         GLuint depthBits,
                         GLuint stencilBits,
                         GLuint pixelBytes,
                         GLuint componentAlignmentMask,
                         bool isBlock,
                         bool isFixed,
                         bool isScaled,
                         gl::VertexAttribType vertexAttribType)
    : id(id),
      glInternalFormat(glFormat),
      fboImplementationInternalFormat(fboFormat),
      mipGenerationFunction(mipGen),
      pixelReadFunction(colorRead),
      pixelWriteFunction(colorWrite),
      fastCopyFunctions(fastCopyFunctions),
      componentType(componentType),
      redBits(redBits),
      greenBits(greenBits),
      blueBits(blueBits),
      alphaBits(alphaBits),
      luminanceBits(luminanceBits),
      depthBits(depthBits),
      stencilBits(stencilBits),
      pixelBytes(pixelBytes),
      componentAlignmentMask(componentAlignmentMask),
      channelCount(GetChannelCount(redBits,
                                   greenBits,
                                   blueBits,
                                   alphaBits,
                                   luminanceBits,
                                   depthBits,
                                   stencilBits)),
      isBlock(isBlock),
      isFixed(isFixed),
      isScaled(isScaled),
      vertexAttribType(vertexAttribType)
{}

constexpr bool Format::hasDepthOrStencilBits() const
{
    return depthBits > 0 || stencilBits > 0;
}

constexpr bool Format::isLUMA() const
{
    // There's no format with G or B without R
    ASSERT(redBits > 0 || (greenBits == 0 && blueBits == 0));
    return redBits == 0 && (luminanceBits > 0 || alphaBits > 0);
}

constexpr bool Format::isSint() const
{
    return componentType == GL_INT;
}

constexpr bool Format::isUint() const
{
    return componentType == GL_UNSIGNED_INT;
}

constexpr bool Format::isSnorm() const
{
    return componentType == GL_SIGNED_NORMALIZED;
}

constexpr bool Format::isUnorm() const
{
    return componentType == GL_UNSIGNED_NORMALIZED;
}

constexpr bool Format::isFloat() const
{
    return componentType == GL_FLOAT;
}

constexpr bool Format::isVertexTypeHalfFloat() const
{
    return vertexAttribType == gl::VertexAttribType::HalfFloat;
}

}  // namespace angle

#endif  // LIBANGLE_RENDERER_FORMAT_H_
