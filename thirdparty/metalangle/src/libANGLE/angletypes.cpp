//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// angletypes.h : Defines a variety of structures and enum types that are used throughout libGLESv2

#include "libANGLE/angletypes.h"
#include "libANGLE/Program.h"
#include "libANGLE/State.h"
#include "libANGLE/VertexArray.h"
#include "libANGLE/VertexAttribute.h"

namespace gl
{
RasterizerState::RasterizerState()
{
    memset(this, 0, sizeof(RasterizerState));

    rasterizerDiscard   = false;
    cullFace            = false;
    cullMode            = CullFaceMode::Back;
    frontFace           = GL_CCW;
    polygonOffsetFill   = false;
    polygonOffsetFactor = 0.0f;
    polygonOffsetUnits  = 0.0f;
    pointDrawMode       = false;
    multiSample         = false;
}

bool operator==(const RasterizerState &a, const RasterizerState &b)
{
    return memcmp(&a, &b, sizeof(RasterizerState)) == 0;
}

bool operator!=(const RasterizerState &a, const RasterizerState &b)
{
    return !(a == b);
}

BlendState::BlendState()
{
    memset(this, 0, sizeof(BlendState));

    blend                 = false;
    sourceBlendRGB        = GL_ONE;
    sourceBlendAlpha      = GL_ONE;
    destBlendRGB          = GL_ZERO;
    destBlendAlpha        = GL_ZERO;
    blendEquationRGB      = GL_FUNC_ADD;
    blendEquationAlpha    = GL_FUNC_ADD;
    sampleAlphaToCoverage = false;
    dither                = true;
}

BlendState::BlendState(const BlendState &other)
{
    memcpy(this, &other, sizeof(BlendState));
}

bool BlendState::allChannelsMasked() const
{
    return !colorMaskRed && !colorMaskGreen && !colorMaskBlue && !colorMaskAlpha;
}

bool operator==(const BlendState &a, const BlendState &b)
{
    return memcmp(&a, &b, sizeof(BlendState)) == 0;
}

bool operator!=(const BlendState &a, const BlendState &b)
{
    return !(a == b);
}

DepthStencilState::DepthStencilState()
{
    memset(this, 0, sizeof(DepthStencilState));

    depthTest                = false;
    depthFunc                = GL_LESS;
    depthMask                = true;
    stencilTest              = false;
    stencilFunc              = GL_ALWAYS;
    stencilMask              = static_cast<GLuint>(-1);
    stencilWritemask         = static_cast<GLuint>(-1);
    stencilBackFunc          = GL_ALWAYS;
    stencilBackMask          = static_cast<GLuint>(-1);
    stencilBackWritemask     = static_cast<GLuint>(-1);
    stencilFail              = GL_KEEP;
    stencilPassDepthFail     = GL_KEEP;
    stencilPassDepthPass     = GL_KEEP;
    stencilBackFail          = GL_KEEP;
    stencilBackPassDepthFail = GL_KEEP;
    stencilBackPassDepthPass = GL_KEEP;
}

DepthStencilState::DepthStencilState(const DepthStencilState &other)
{
    memcpy(this, &other, sizeof(DepthStencilState));
}

bool operator==(const DepthStencilState &a, const DepthStencilState &b)
{
    return memcmp(&a, &b, sizeof(DepthStencilState)) == 0;
}

bool operator!=(const DepthStencilState &a, const DepthStencilState &b)
{
    return !(a == b);
}

SamplerState::SamplerState()
{
    memset(this, 0, sizeof(SamplerState));

    setMinFilter(GL_NEAREST_MIPMAP_LINEAR);
    setMagFilter(GL_LINEAR);
    setWrapS(GL_REPEAT);
    setWrapT(GL_REPEAT);
    setWrapR(GL_REPEAT);
    setMaxAnisotropy(1.0f);
    setMinLod(-1000.0f);
    setMaxLod(1000.0f);
    setCompareMode(GL_NONE);
    setCompareFunc(GL_LEQUAL);
    setSRGBDecode(GL_DECODE_EXT);
}

SamplerState::SamplerState(const SamplerState &other) = default;

// static
SamplerState SamplerState::CreateDefaultForTarget(TextureType type)
{
    SamplerState state;

    // According to OES_EGL_image_external and ARB_texture_rectangle: For external textures, the
    // default min filter is GL_LINEAR and the default s and t wrap modes are GL_CLAMP_TO_EDGE.
    if (type == TextureType::External || type == TextureType::Rectangle)
    {
        state.mMinFilter = GL_LINEAR;
        state.mWrapS     = GL_CLAMP_TO_EDGE;
        state.mWrapT     = GL_CLAMP_TO_EDGE;
    }

    return state;
}

void SamplerState::setMinFilter(GLenum minFilter)
{
    mMinFilter                    = minFilter;
    mCompleteness.typed.minFilter = static_cast<uint8_t>(FromGLenum<FilterMode>(minFilter));
}

void SamplerState::setMagFilter(GLenum magFilter)
{
    mMagFilter                    = magFilter;
    mCompleteness.typed.magFilter = static_cast<uint8_t>(FromGLenum<FilterMode>(magFilter));
}

void SamplerState::setWrapS(GLenum wrapS)
{
    mWrapS                    = wrapS;
    mCompleteness.typed.wrapS = static_cast<uint8_t>(FromGLenum<WrapMode>(wrapS));
}

void SamplerState::setWrapT(GLenum wrapT)
{
    mWrapT = wrapT;
    updateWrapTCompareMode();
}

void SamplerState::setWrapR(GLenum wrapR)
{
    mWrapR = wrapR;
}

void SamplerState::setMaxAnisotropy(float maxAnisotropy)
{
    mMaxAnisotropy = maxAnisotropy;
}

void SamplerState::setMinLod(GLfloat minLod)
{
    mMinLod = minLod;
}

void SamplerState::setMaxLod(GLfloat maxLod)
{
    mMaxLod = maxLod;
}

void SamplerState::setCompareMode(GLenum compareMode)
{
    mCompareMode = compareMode;
    updateWrapTCompareMode();
}

void SamplerState::setCompareFunc(GLenum compareFunc)
{
    mCompareFunc = compareFunc;
}

void SamplerState::setSRGBDecode(GLenum sRGBDecode)
{
    mSRGBDecode = sRGBDecode;
}

void SamplerState::setBorderColor(const ColorGeneric &color)
{
    mBorderColor = color;
}

void SamplerState::updateWrapTCompareMode()
{
    uint8_t wrap    = static_cast<uint8_t>(FromGLenum<WrapMode>(mWrapT));
    uint8_t compare = static_cast<uint8_t>(mCompareMode == GL_NONE ? 0x10 : 0x00);
    mCompleteness.typed.wrapTCompareMode = wrap | compare;
}

ImageUnit::ImageUnit()
    : texture(), level(0), layered(false), layer(0), access(GL_READ_ONLY), format(GL_R32UI)
{}

ImageUnit::ImageUnit(const ImageUnit &other) = default;

ImageUnit::~ImageUnit() = default;

static void MinMax(int a, int b, int *minimum, int *maximum)
{
    if (a < b)
    {
        *minimum = a;
        *maximum = b;
    }
    else
    {
        *minimum = b;
        *maximum = a;
    }
}

Rectangle Rectangle::flip(bool flipX, bool flipY) const
{
    Rectangle flipped = *this;
    if (flipX)
    {
        flipped.x     = flipped.x + flipped.width;
        flipped.width = -flipped.width;
    }
    if (flipY)
    {
        flipped.y      = flipped.y + flipped.height;
        flipped.height = -flipped.height;
    }
    return flipped;
}

Rectangle Rectangle::removeReversal() const
{
    return flip(isReversedX(), isReversedY());
}

bool Rectangle::encloses(const gl::Rectangle &inside) const
{
    return x0() <= inside.x0() && y0() <= inside.y0() && x1() >= inside.x1() && y1() >= inside.y1();
}

bool ClipRectangle(const Rectangle &source, const Rectangle &clip, Rectangle *intersection)
{
    int minSourceX, maxSourceX, minSourceY, maxSourceY;
    MinMax(source.x, source.x + source.width, &minSourceX, &maxSourceX);
    MinMax(source.y, source.y + source.height, &minSourceY, &maxSourceY);

    int minClipX, maxClipX, minClipY, maxClipY;
    MinMax(clip.x, clip.x + clip.width, &minClipX, &maxClipX);
    MinMax(clip.y, clip.y + clip.height, &minClipY, &maxClipY);

    if (minSourceX >= maxClipX || maxSourceX <= minClipX || minSourceY >= maxClipY ||
        maxSourceY <= minClipY)
    {
        return false;
    }
    if (intersection)
    {
        intersection->x      = std::max(minSourceX, minClipX);
        intersection->y      = std::max(minSourceY, minClipY);
        intersection->width  = std::min(maxSourceX, maxClipX) - std::max(minSourceX, minClipX);
        intersection->height = std::min(maxSourceY, maxClipY) - std::max(minSourceY, minClipY);
    }
    return true;
}

bool Box::operator==(const Box &other) const
{
    return (x == other.x && y == other.y && z == other.z && width == other.width &&
            height == other.height && depth == other.depth);
}

bool Box::operator!=(const Box &other) const
{
    return !(*this == other);
}

Rectangle Box::toRect() const
{
    ASSERT(z == 0 && depth == 1);
    return Rectangle(x, y, width, height);
}

bool operator==(const Offset &a, const Offset &b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator!=(const Offset &a, const Offset &b)
{
    return !(a == b);
}

bool operator==(const Extents &lhs, const Extents &rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth;
}

bool operator!=(const Extents &lhs, const Extents &rhs)
{
    return !(lhs == rhs);
}

bool ValidateComponentTypeMasks(unsigned long outputTypes,
                                unsigned long inputTypes,
                                unsigned long outputMask,
                                unsigned long inputMask)
{
    static_assert(IMPLEMENTATION_MAX_DRAW_BUFFERS <= kMaxComponentTypeMaskIndex,
                  "Output/input masks should fit into 16 bits - 1 bit per draw buffer. The "
                  "corresponding type masks should fit into 32 bits - 2 bits per draw buffer.");
    static_assert(MAX_VERTEX_ATTRIBS <= kMaxComponentTypeMaskIndex,
                  "Output/input masks should fit into 16 bits - 1 bit per attrib. The "
                  "corresponding type masks should fit into 32 bits - 2 bits per attrib.");

    // For performance reasons, draw buffer and attribute type validation is done using bit masks.
    // We store two bits representing the type split, with the low bit in the lower 16 bits of the
    // variable, and the high bit in the upper 16 bits of the variable. This is done so we can AND
    // with the elswewhere used DrawBufferMask or AttributeMask.

    // OR the masks with themselves, shifted 16 bits. This is to match our split type bits.
    outputMask |= (outputMask << kMaxComponentTypeMaskIndex);
    inputMask |= (inputMask << kMaxComponentTypeMaskIndex);

    // To validate:
    // 1. Remove any indexes that are not enabled in the input (& inputMask)
    // 2. Remove any indexes that exist in output, but not in input (& outputMask)
    // 3. Use == to verify equality
    return (outputTypes & inputMask) == ((inputTypes & outputMask) & inputMask);
}

GLsizeiptr GetBoundBufferAvailableSize(const OffsetBindingPointer<Buffer> &binding)
{
    Buffer *buffer = binding.get();
    if (buffer)
    {
        if (binding.getSize() == 0)
            return static_cast<GLsizeiptr>(buffer->getSize());
        angle::CheckedNumeric<GLintptr> offset       = binding.getOffset();
        angle::CheckedNumeric<GLsizeiptr> size       = binding.getSize();
        angle::CheckedNumeric<GLsizeiptr> bufferSize = buffer->getSize();
        auto end                                     = offset + size;
        auto clampedSize                             = size;
        auto difference                              = end - bufferSize;
        if (!difference.IsValid())
        {
            return 0;
        }
        if (difference.ValueOrDie() > 0)
        {
            clampedSize = size - difference;
        }
        return clampedSize.ValueOrDefault(0);
    }
    else
    {
        return 0;
    }
}

}  // namespace gl
