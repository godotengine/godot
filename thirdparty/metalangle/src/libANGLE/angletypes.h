//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// angletypes.h : Defines a variety of structures and enum types that are used throughout libGLESv2

#ifndef LIBANGLE_ANGLETYPES_H_
#define LIBANGLE_ANGLETYPES_H_

#include "common/Color.h"
#include "common/FixedVector.h"
#include "common/PackedEnums.h"
#include "common/bitset_utils.h"
#include "common/vector_utils.h"
#include "libANGLE/Constants.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

#include <inttypes.h>
#include <stdint.h>

#include <bitset>
#include <map>
#include <unordered_map>

namespace gl
{
class Buffer;
class Texture;

struct Rectangle
{
    Rectangle() : x(0), y(0), width(0), height(0) {}
    constexpr Rectangle(int x_in, int y_in, int width_in, int height_in)
        : x(x_in), y(y_in), width(width_in), height(height_in)
    {}

    int x0() const { return x; }
    int y0() const { return y; }
    int x1() const { return x + width; }
    int y1() const { return y + height; }

    bool isReversedX() const { return width < 0; }
    bool isReversedY() const { return height < 0; }

    // Returns a rectangle with the same area but flipped in X, Y, neither or both.
    Rectangle flip(bool flipX, bool flipY) const;

    // Returns a rectangle with the same area but with height and width guaranteed to be positive.
    Rectangle removeReversal() const;

    bool encloses(const gl::Rectangle &inside) const;

    int x;
    int y;
    int width;
    int height;
};

bool operator==(const Rectangle &a, const Rectangle &b);
bool operator!=(const Rectangle &a, const Rectangle &b);

bool ClipRectangle(const Rectangle &source, const Rectangle &clip, Rectangle *intersection);

struct Offset
{
    constexpr Offset() : x(0), y(0), z(0) {}
    constexpr Offset(int x_in, int y_in, int z_in) : x(x_in), y(y_in), z(z_in) {}

    int x;
    int y;
    int z;
};

constexpr Offset kOffsetZero(0, 0, 0);

bool operator==(const Offset &a, const Offset &b);
bool operator!=(const Offset &a, const Offset &b);

struct Extents
{
    Extents() : width(0), height(0), depth(0) {}
    Extents(int width_, int height_, int depth_) : width(width_), height(height_), depth(depth_) {}

    Extents(const Extents &other) = default;
    Extents &operator=(const Extents &other) = default;

    bool empty() const { return (width * height * depth) == 0; }

    int width;
    int height;
    int depth;
};

bool operator==(const Extents &lhs, const Extents &rhs);
bool operator!=(const Extents &lhs, const Extents &rhs);

struct Box
{
    Box() : x(0), y(0), z(0), width(0), height(0), depth(0) {}
    Box(int x_in, int y_in, int z_in, int width_in, int height_in, int depth_in)
        : x(x_in), y(y_in), z(z_in), width(width_in), height(height_in), depth(depth_in)
    {}
    Box(const Offset &offset, const Extents &size)
        : x(offset.x),
          y(offset.y),
          z(offset.z),
          width(size.width),
          height(size.height),
          depth(size.depth)
    {}
    bool operator==(const Box &other) const;
    bool operator!=(const Box &other) const;
    Rectangle toRect() const;

    int x;
    int y;
    int z;
    int width;
    int height;
    int depth;
};

struct RasterizerState final
{
    // This will zero-initialize the struct, including padding.
    RasterizerState();

    bool cullFace;
    CullFaceMode cullMode;
    GLenum frontFace;

    bool polygonOffsetFill;
    GLfloat polygonOffsetFactor;
    GLfloat polygonOffsetUnits;

    bool pointDrawMode;
    bool multiSample;

    bool rasterizerDiscard;
};

bool operator==(const RasterizerState &a, const RasterizerState &b);
bool operator!=(const RasterizerState &a, const RasterizerState &b);

struct BlendState final
{
    // This will zero-initialize the struct, including padding.
    BlendState();
    BlendState(const BlendState &other);

    bool allChannelsMasked() const;

    bool blend;
    GLenum sourceBlendRGB;
    GLenum destBlendRGB;
    GLenum sourceBlendAlpha;
    GLenum destBlendAlpha;
    GLenum blendEquationRGB;
    GLenum blendEquationAlpha;

    bool colorMaskRed;
    bool colorMaskGreen;
    bool colorMaskBlue;
    bool colorMaskAlpha;

    bool sampleAlphaToCoverage;

    bool dither;
};

bool operator==(const BlendState &a, const BlendState &b);
bool operator!=(const BlendState &a, const BlendState &b);

struct DepthStencilState final
{
    // This will zero-initialize the struct, including padding.
    DepthStencilState();
    DepthStencilState(const DepthStencilState &other);

    bool depthTest;
    GLenum depthFunc;
    bool depthMask;

    bool stencilTest;
    GLenum stencilFunc;
    GLuint stencilMask;
    GLenum stencilFail;
    GLenum stencilPassDepthFail;
    GLenum stencilPassDepthPass;
    GLuint stencilWritemask;
    GLenum stencilBackFunc;
    GLuint stencilBackMask;
    GLenum stencilBackFail;
    GLenum stencilBackPassDepthFail;
    GLenum stencilBackPassDepthPass;
    GLuint stencilBackWritemask;
};

bool operator==(const DepthStencilState &a, const DepthStencilState &b);
bool operator!=(const DepthStencilState &a, const DepthStencilState &b);

// Packs a sampler state for completeness checks:
// * minFilter: 5 values (3 bits)
// * magFilter: 2 values (1 bit)
// * wrapS:     3 values (2 bits)
// * wrapT:     3 values (2 bits)
// * compareMode: 1 bit (for == GL_NONE).
// This makes a total of 9 bits. We can pack this easily into 32 bits:
// * minFilter: 8 bits
// * magFilter: 8 bits
// * wrapS:     8 bits
// * wrapT:     4 bits
// * compareMode: 4 bits

struct PackedSamplerCompleteness
{
    uint8_t minFilter;
    uint8_t magFilter;
    uint8_t wrapS;
    uint8_t wrapTCompareMode;
};

static_assert(sizeof(PackedSamplerCompleteness) == sizeof(uint32_t), "Unexpected size");

// State from Table 6.10 (state per sampler object)
class SamplerState final
{
  public:
    // This will zero-initialize the struct, including padding.
    SamplerState();
    SamplerState(const SamplerState &other);

    static SamplerState CreateDefaultForTarget(TextureType type);

    GLenum getMinFilter() const { return mMinFilter; }

    void setMinFilter(GLenum minFilter);

    GLenum getMagFilter() const { return mMagFilter; }

    void setMagFilter(GLenum magFilter);

    GLenum getWrapS() const { return mWrapS; }

    void setWrapS(GLenum wrapS);

    GLenum getWrapT() const { return mWrapT; }

    void setWrapT(GLenum wrapT);

    GLenum getWrapR() const { return mWrapR; }

    void setWrapR(GLenum wrapR);

    float getMaxAnisotropy() const { return mMaxAnisotropy; }

    void setMaxAnisotropy(float maxAnisotropy);

    GLfloat getMinLod() const { return mMinLod; }

    void setMinLod(GLfloat minLod);

    GLfloat getMaxLod() const { return mMaxLod; }

    void setMaxLod(GLfloat maxLod);

    GLenum getCompareMode() const { return mCompareMode; }

    void setCompareMode(GLenum compareMode);

    GLenum getCompareFunc() const { return mCompareFunc; }

    void setCompareFunc(GLenum compareFunc);

    GLenum getSRGBDecode() const { return mSRGBDecode; }

    void setSRGBDecode(GLenum sRGBDecode);

    void setBorderColor(const ColorGeneric &color);

    const ColorGeneric &getBorderColor() const { return mBorderColor; }

    bool sameCompleteness(const SamplerState &samplerState) const
    {
        return mCompleteness.packed == samplerState.mCompleteness.packed;
    }

  private:
    void updateWrapTCompareMode();

    GLenum mMinFilter;
    GLenum mMagFilter;

    GLenum mWrapS;
    GLenum mWrapT;
    GLenum mWrapR;

    // From EXT_texture_filter_anisotropic
    float mMaxAnisotropy;

    GLfloat mMinLod;
    GLfloat mMaxLod;

    GLenum mCompareMode;
    GLenum mCompareFunc;

    GLenum mSRGBDecode;

    ColorGeneric mBorderColor;

    union Completeness
    {
        uint32_t packed;
        PackedSamplerCompleteness typed;
    };

    Completeness mCompleteness;
};

bool operator==(const SamplerState &a, const SamplerState &b);
bool operator!=(const SamplerState &a, const SamplerState &b);

struct DrawArraysIndirectCommand
{
    GLuint count;
    GLuint instanceCount;
    GLuint first;
    GLuint baseInstance;
};
static_assert(sizeof(DrawArraysIndirectCommand) == 16,
              "Unexpected size of DrawArraysIndirectCommand");

struct DrawElementsIndirectCommand
{
    GLuint count;
    GLuint primCount;
    GLuint firstIndex;
    GLint baseVertex;
    GLuint baseInstance;
};
static_assert(sizeof(DrawElementsIndirectCommand) == 20,
              "Unexpected size of DrawElementsIndirectCommand");

struct ImageUnit
{
    ImageUnit();
    ImageUnit(const ImageUnit &other);
    ~ImageUnit();

    BindingPointer<Texture> texture;
    GLint level;
    GLboolean layered;
    GLint layer;
    GLenum access;
    GLenum format;
};

using ImageUnitTextureTypeMap = std::map<unsigned int, gl::TextureType>;

struct PixelStoreStateBase
{
    GLint alignment   = 4;
    GLint rowLength   = 0;
    GLint skipRows    = 0;
    GLint skipPixels  = 0;
    GLint imageHeight = 0;
    GLint skipImages  = 0;
};

struct PixelUnpackState : PixelStoreStateBase
{};

struct PixelPackState : PixelStoreStateBase
{
    bool reverseRowOrder = false;
};

// Used in Program and VertexArray.
using AttributesMask = angle::BitSet<MAX_VERTEX_ATTRIBS>;

// Used in Program
using UniformBlockBindingMask = angle::BitSet<IMPLEMENTATION_MAX_COMBINED_SHADER_UNIFORM_BUFFERS>;

// Used in Framebuffer / Program
using DrawBufferMask = angle::BitSet<IMPLEMENTATION_MAX_DRAW_BUFFERS>;

// Used in StateCache
using StorageBuffersMask = angle::BitSet<IMPLEMENTATION_MAX_SHADER_STORAGE_BUFFER_BINDINGS>;

template <typename T>
using TexLevelArray = std::array<T, IMPLEMENTATION_MAX_TEXTURE_LEVELS>;

enum class ComponentType
{
    Float       = 0,
    Int         = 1,
    UnsignedInt = 2,
    NoType      = 3,
    EnumCount   = 4,
    InvalidEnum = 4,
};

constexpr ComponentType GLenumToComponentType(GLenum componentType)
{
    switch (componentType)
    {
        case GL_FLOAT:
            return ComponentType::Float;
        case GL_INT:
            return ComponentType::Int;
        case GL_UNSIGNED_INT:
            return ComponentType::UnsignedInt;
        case GL_NONE:
            return ComponentType::NoType;
        default:
            return ComponentType::InvalidEnum;
    }
}

constexpr angle::PackedEnumMap<ComponentType, uint32_t> kComponentMasks = {{
    {ComponentType::Float, 0x10001},
    {ComponentType::Int, 0x00001},
    {ComponentType::UnsignedInt, 0x10000},
}};

constexpr size_t kMaxComponentTypeMaskIndex = 16;
using ComponentTypeMask                     = angle::BitSet<kMaxComponentTypeMaskIndex * 2>;

ANGLE_INLINE void SetComponentTypeMask(ComponentType type, size_t index, ComponentTypeMask *mask)
{
    ASSERT(index <= kMaxComponentTypeMaskIndex);
    *mask &= ~(0x10001 << index);
    *mask |= kComponentMasks[type] << index;
}

ANGLE_INLINE ComponentType GetComponentTypeMask(const ComponentTypeMask &mask, size_t index)
{
    ASSERT(index <= kMaxComponentTypeMaskIndex);
    uint32_t mask_bits = static_cast<uint32_t>((mask.to_ulong() >> index) & 0x10001);
    switch (mask_bits)
    {
        case 0x10001:
            return ComponentType::Float;
        case 0x00001:
            return ComponentType::Int;
        case 0x10000:
            return ComponentType::UnsignedInt;
        default:
            return ComponentType::InvalidEnum;
    }
}

bool ValidateComponentTypeMasks(unsigned long outputTypes,
                                unsigned long inputTypes,
                                unsigned long outputMask,
                                unsigned long inputMask);

using ContextID = uintptr_t;

constexpr size_t kCubeFaceCount = 6;

using TextureMap = angle::PackedEnumMap<TextureType, BindingPointer<Texture>>;

// ShaderVector can contain one item per shader.  It differs from ShaderMap in that the values are
// not indexed by ShaderType.
template <typename T>
using ShaderVector = angle::FixedVector<T, static_cast<size_t>(ShaderType::EnumCount)>;

template <typename T>
using AttachmentArray = std::array<T, IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS>;

template <typename T>
using DrawBuffersArray = std::array<T, IMPLEMENTATION_MAX_DRAW_BUFFERS>;

template <typename T>
using DrawBuffersVector = angle::FixedVector<T, IMPLEMENTATION_MAX_DRAW_BUFFERS>;

template <typename T>
using AttribArray = std::array<T, MAX_VERTEX_ATTRIBS>;

using ActiveTextureMask = angle::BitSet<IMPLEMENTATION_MAX_ACTIVE_TEXTURES>;

template <typename T>
using ActiveTextureArray = std::array<T, IMPLEMENTATION_MAX_ACTIVE_TEXTURES>;

using ActiveTexturePointerArray = ActiveTextureArray<Texture *>;
using ActiveTextureTypeArray    = ActiveTextureArray<TextureType>;

template <typename T>
using UniformBuffersArray = std::array<T, IMPLEMENTATION_MAX_UNIFORM_BUFFER_BINDINGS>;
template <typename T>
using StorageBuffersArray = std::array<T, IMPLEMENTATION_MAX_SHADER_STORAGE_BUFFER_BINDINGS>;
template <typename T>
using AtomicCounterBuffersArray = std::array<T, IMPLEMENTATION_MAX_ATOMIC_COUNTER_BUFFERS>;
using AtomicCounterBufferMask   = angle::BitSet<IMPLEMENTATION_MAX_ATOMIC_COUNTER_BUFFERS>;
template <typename T>
using ImagesArray = std::array<T, IMPLEMENTATION_MAX_IMAGE_UNITS>;

using ImageUnitMask = angle::BitSet<IMPLEMENTATION_MAX_IMAGE_UNITS>;

using SupportedSampleSet = std::set<GLuint>;

template <typename T>
using TransformFeedbackBuffersArray =
    std::array<T, gl::IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_BUFFERS>;

constexpr size_t kBarrierVectorDefaultSize = 16;

template <typename T>
using BarrierVector = angle::FastVector<T, kBarrierVectorDefaultSize>;

using BufferBarrierVector = BarrierVector<Buffer *>;

struct TextureAndLayout
{
    Texture *texture;
    GLenum layout;
};
using TextureBarrierVector = BarrierVector<TextureAndLayout>;

// OffsetBindingPointer.getSize() returns the size specified by the user, which may be larger than
// the size of the bound buffer. This function reduces the returned size to fit the bound buffer if
// necessary. Returns 0 if no buffer is bound or if integer overflow occurs.
GLsizeiptr GetBoundBufferAvailableSize(const OffsetBindingPointer<Buffer> &binding);

}  // namespace gl

namespace rx
{
// A macro that determines whether an object has a given runtime type.
#if defined(__clang__)
#    if __has_feature(cxx_rtti)
#        define ANGLE_HAS_DYNAMIC_CAST 1
#    endif
#elif !defined(NDEBUG) && (!defined(_MSC_VER) || defined(_CPPRTTI)) &&              \
    (!defined(__GNUC__) || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3) || \
     defined(__GXX_RTTI))
#    define ANGLE_HAS_DYNAMIC_CAST 1
#endif

#ifdef ANGLE_HAS_DYNAMIC_CAST
#    define ANGLE_HAS_DYNAMIC_TYPE(type, obj) (dynamic_cast<type>(obj) != nullptr)
#    undef ANGLE_HAS_DYNAMIC_CAST
#else
#    define ANGLE_HAS_DYNAMIC_TYPE(type, obj) (obj != nullptr)
#endif

// Downcast a base implementation object (EG TextureImpl to TextureD3D)
template <typename DestT, typename SrcT>
inline DestT *GetAs(SrcT *src)
{
    ASSERT(ANGLE_HAS_DYNAMIC_TYPE(DestT *, src));
    return static_cast<DestT *>(src);
}

template <typename DestT, typename SrcT>
inline const DestT *GetAs(const SrcT *src)
{
    ASSERT(ANGLE_HAS_DYNAMIC_TYPE(const DestT *, src));
    return static_cast<const DestT *>(src);
}

#undef ANGLE_HAS_DYNAMIC_TYPE

// Downcast a GL object to an Impl (EG gl::Texture to rx::TextureD3D)
template <typename DestT, typename SrcT>
inline DestT *GetImplAs(SrcT *src)
{
    return GetAs<DestT>(src->getImplementation());
}

template <typename DestT, typename SrcT>
inline DestT *SafeGetImplAs(SrcT *src)
{
    return src != nullptr ? GetAs<DestT>(src->getImplementation()) : nullptr;
}

}  // namespace rx

#include "angletypes.inc"

namespace angle
{
// Zero-based for better array indexing
enum FramebufferBinding
{
    FramebufferBindingRead = 0,
    FramebufferBindingDraw,
    FramebufferBindingSingletonMax,
    FramebufferBindingBoth = FramebufferBindingSingletonMax,
    FramebufferBindingMax,
    FramebufferBindingUnknown = FramebufferBindingMax,
};

inline FramebufferBinding EnumToFramebufferBinding(GLenum enumValue)
{
    switch (enumValue)
    {
        case GL_READ_FRAMEBUFFER:
            return FramebufferBindingRead;
        case GL_DRAW_FRAMEBUFFER:
            return FramebufferBindingDraw;
        case GL_FRAMEBUFFER:
            return FramebufferBindingBoth;
        default:
            UNREACHABLE();
            return FramebufferBindingUnknown;
    }
}

inline GLenum FramebufferBindingToEnum(FramebufferBinding binding)
{
    switch (binding)
    {
        case FramebufferBindingRead:
            return GL_READ_FRAMEBUFFER;
        case FramebufferBindingDraw:
            return GL_DRAW_FRAMEBUFFER;
        case FramebufferBindingBoth:
            return GL_FRAMEBUFFER;
        default:
            UNREACHABLE();
            return GL_NONE;
    }
}

template <typename ObjT, typename ContextT>
class DestroyThenDelete
{
  public:
    DestroyThenDelete(const ContextT *context) : mContext(context) {}

    void operator()(ObjT *obj)
    {
        (void)(obj->onDestroy(mContext));
        delete obj;
    }

  private:
    const ContextT *mContext;
};

// Helper class for wrapping an onDestroy function.
template <typename ObjT, typename DeleterT>
class UniqueObjectPointerBase : angle::NonCopyable
{
  public:
    template <typename ContextT>
    UniqueObjectPointerBase(const ContextT *context) : mObject(nullptr), mDeleter(context)
    {}

    template <typename ContextT>
    UniqueObjectPointerBase(ObjT *obj, const ContextT *context) : mObject(obj), mDeleter(context)
    {}

    ~UniqueObjectPointerBase()
    {
        if (mObject)
        {
            mDeleter(mObject);
        }
    }

    ObjT *operator->() const { return mObject; }

    ObjT *release()
    {
        auto obj = mObject;
        mObject  = nullptr;
        return obj;
    }

    ObjT *get() const { return mObject; }

    void reset(ObjT *obj)
    {
        if (mObject)
        {
            mDeleter(mObject);
        }
        mObject = obj;
    }

  private:
    ObjT *mObject;
    DeleterT mDeleter;
};

template <typename ObjT, typename ContextT>
using UniqueObjectPointer = UniqueObjectPointerBase<ObjT, DestroyThenDelete<ObjT, ContextT>>;

}  // namespace angle

namespace gl
{
class State;
}  // namespace gl

#endif  // LIBANGLE_ANGLETYPES_H_
