//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_common.h:
//      Declares common constants, template classes, and mtl::Context - the MTLDevice container &
//      error handler base class.
//

#ifndef LIBANGLE_RENDERER_METAL_MTL_COMMON_H_
#define LIBANGLE_RENDERER_METAL_MTL_COMMON_H_

#import <Metal/Metal.h>

#include <TargetConditionals.h>

#include <string>

#include "common/FastVector.h"
#include "common/Optional.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "common/apple_platform_utils.h"
#include "common/hash_utils.h"
#include "libANGLE/Constants.h"
#include "libANGLE/Version.h"
#include "libANGLE/angletypes.h"

#if TARGET_OS_IPHONE
#    if !defined(__IPHONE_11_0)
#        define __IPHONE_11_0 110000
#    endif
#    if !defined(ANGLE_IOS_DEPLOY_TARGET)
#        define ANGLE_IOS_DEPLOY_TARGET __IPHONE_11_0
#    endif
#    if !defined(__IPHONE_OS_VERSION_MAX_ALLOWED)
#        define __IPHONE_OS_VERSION_MAX_ALLOWED __IPHONE_11_0
#    endif
#    if !defined(__TV_OS_VERSION_MAX_ALLOWED)
#        define __TV_OS_VERSION_MAX_ALLOWED __IPHONE_11_0
#    endif
#endif

#if !defined(TARGET_OS_MACCATALYST)
#    define TARGET_OS_MACCATALYST 0
#endif

#if defined(__ARM_ARCH)
#    define ANGLE_MTL_ARM (__ARM_ARCH != 0)
#else
#    define ANGLE_MTL_ARM 0
#endif

#define ANGLE_MTL_OBJC_SCOPE @autoreleasepool

#if !__has_feature(objc_arc)
#    define ANGLE_MTL_AUTORELEASE autorelease
#    define ANGLE_MTL_RETAIN retain
#    define ANGLE_MTL_RELEASE release
#else
#    define ANGLE_MTL_AUTORELEASE self
#    define ANGLE_MTL_RETAIN self
#    define ANGLE_MTL_RELEASE self
#endif

#define ANGLE_MTL_UNUSED __attribute__((unused))

#if defined(ANGLE_MTL_ENABLE_TRACE)
#    define ANGLE_MTL_LOG(...) NSLog(@__VA_ARGS__)
#else
#    define ANGLE_MTL_LOG(...) (void)0
#endif

namespace egl
{
class Display;
class Image;
}  // namespace egl

#define ANGLE_GL_OBJECTS_X(PROC) \
    PROC(Buffer)                 \
    PROC(Context)                \
    PROC(Framebuffer)            \
    PROC(MemoryObject)           \
    PROC(Query)                  \
    PROC(Program)                \
    PROC(Sampler)                \
    PROC(Semaphore)              \
    PROC(Texture)                \
    PROC(TransformFeedback)      \
    PROC(VertexArray)

#define ANGLE_PRE_DECLARE_OBJECT(OBJ) class OBJ;

namespace gl
{
struct Rectangle;
ANGLE_GL_OBJECTS_X(ANGLE_PRE_DECLARE_OBJECT)
}  // namespace gl

#define ANGLE_PRE_DECLARE_MTL_OBJECT(OBJ) class OBJ##Mtl;

namespace rx
{
class DisplayMtl;
class ContextMtl;
class FramebufferMtl;
class BufferMtl;
class ImageMtl;
class VertexArrayMtl;
class TextureMtl;
class ProgramMtl;
class SamplerMtl;
class TransformFeedbackMtl;

ANGLE_GL_OBJECTS_X(ANGLE_PRE_DECLARE_MTL_OBJECT)

namespace mtl
{

// NOTE(hqle): support variable max number of vertex attributes
constexpr uint32_t kMaxVertexAttribs = gl::MAX_VERTEX_ATTRIBS;
// NOTE(hqle): support variable max number of render targets
constexpr uint32_t kMaxRenderTargets = 4;

constexpr uint32_t kMaxShaderUBOs = 12;
constexpr uint32_t kMaxUBOSize    = 16384;

constexpr uint32_t kMaxShaderXFBs = gl::IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;

constexpr size_t kDefaultAttributeSize = 4 * sizeof(float);

// Metal limits
constexpr uint32_t kMaxShaderBuffers     = 31;
constexpr uint32_t kMaxShaderSamplers    = 16;
constexpr size_t kInlineConstDataMaxSize = 4 * 1024;
constexpr size_t kDefaultUniformsMaxSize = kInlineConstDataMaxSize;
constexpr uint32_t kMaxViewports         = 1;

constexpr uint32_t kVertexAttribBufferStrideAlignment = 4;
// Alignment requirement for offset passed to setVertex|FragmentBuffer
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
constexpr uint32_t kUniformBufferSettingOffsetMinAlignment = 256;
#else
constexpr uint32_t kUniformBufferSettingOffsetMinAlignment = 16;
#endif
constexpr uint32_t kIndexBufferOffsetAlignment       = 4;
constexpr uint32_t kArgumentBufferOffsetAlignment    = kUniformBufferSettingOffsetMinAlignment;
constexpr uint32_t kTextureToBufferBlittingAlignment = 256;

// Font end binding limits
constexpr uint32_t kMaxGLSamplerBindings = 2 * kMaxShaderSamplers;
constexpr uint32_t kMaxGLUBOBindings     = 2 * kMaxShaderUBOs;

// Binding index start for vertex data buffers:
constexpr uint32_t kVboBindingIndexStart = 0;

// Binding index for default attribute buffer:
constexpr uint32_t kDefaultAttribsBindingIndex = kVboBindingIndexStart + kMaxVertexAttribs;
// Binding index for driver uniforms:
constexpr uint32_t kDriverUniformsBindingIndex = kDefaultAttribsBindingIndex + 1;
// Binding index for default uniforms:
constexpr uint32_t kDefaultUniformsBindingIndex = kDefaultAttribsBindingIndex + 3;
// Binding index for shadow samplers' compare modes
constexpr uint32_t kShadowSamplerCompareModesBindingIndex = kDefaultUniformsBindingIndex + 1;
// Binding index for UBO's argument buffer
constexpr uint32_t kUBOArgumentBufferBindingIndex = kShadowSamplerCompareModesBindingIndex + 1;

constexpr uint32_t kStencilMaskAll = 0xff;  // Only 8 bits stencil is supported

// This special constant is used to indicate that a particular vertex descriptor's buffer layout
// index is unused.
constexpr MTLVertexStepFunction kVertexStepFunctionInvalid =
    static_cast<MTLVertexStepFunction>(0xff);

constexpr int kEmulatedAlphaValue = 1;

constexpr uint32_t kOcclusionQueryResultSize = sizeof(uint64_t);

// Work-around the enum is not available on macOS
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
constexpr MTLBlitOption kBlitOptionRowLinearPVRTC = MTLBlitOptionNone;
#else
constexpr MTLBlitOption kBlitOptionRowLinearPVRTC          = MTLBlitOptionRowLinearPVRTC;
#endif

#if defined(__IPHONE_13_0) || defined(__MAC_10_15)
using TextureSwizzleChannels                   = MTLTextureSwizzleChannels;
using RenderStages                             = MTLRenderStages;
constexpr MTLRenderStages kRenderStageVertex   = MTLRenderStageVertex;
constexpr MTLRenderStages kRenderStageFragment = MTLRenderStageFragment;
#else
using TextureSwizzleChannels                               = int;
using RenderStages                                         = int;
constexpr RenderStages kRenderStageVertex                  = 1;
constexpr RenderStages kRenderStageFragment                = 2;
#endif

enum class PixelType
{
    Int,
    UInt,
    Float,
    EnumCount,
};

struct ClearColorValue
{
    PixelType type = PixelType::Float;

    union
    {
        struct
        {
            float red, green, blue, alpha;
        };
        struct
        {
            int32_t redI, greenI, blueI, alphaI;
        };
        struct
        {
            uint32_t redU, greenU, blueU, alphaU;
        };
    };
    // Convert to native Metal value
    inline operator MTLClearColor() const
    {
        switch (type)
        {
            case PixelType::Int:
                return MTLClearColorMake(redI, greenI, blueI, alphaI);
            case PixelType::UInt:
                return MTLClearColorMake(redU, greenU, blueU, alphaU);
            case PixelType::Float:
                return MTLClearColorMake(red, green, blue, alpha);
            default:
                UNREACHABLE();
                return MTLClearColorMake(0, 0, 0, 0);
        }
    }
};

template <typename T>
struct ImplTypeHelper;

// clang-format off
#define ANGLE_IMPL_TYPE_HELPER_GL(OBJ) \
template<>                             \
struct ImplTypeHelper<gl::OBJ>         \
{                                      \
    using ImplType = OBJ##Mtl;         \
};
// clang-format on

ANGLE_GL_OBJECTS_X(ANGLE_IMPL_TYPE_HELPER_GL)

template <>
struct ImplTypeHelper<egl::Display>
{
    using ImplType = DisplayMtl;
};

template <>
struct ImplTypeHelper<egl::Image>
{
    using ImplType = ImageMtl;
};

template <typename T>
using GetImplType = typename ImplTypeHelper<T>::ImplType;

template <typename T>
GetImplType<T> *GetImpl(const T *_Nonnull glObject)
{
    return GetImplAs<GetImplType<T>>(glObject);
}

// A vector that allow containing up to SmallVectorSize bytes without allocating anything on heap.
constexpr size_t kSmallVectorSize = 256;
using SmallVector                 = ::angle::FastVector<uint8_t, kSmallVectorSize>;

}  // namespace mtl
}  // namespace rx

namespace std
{
template <>
struct hash<rx::mtl::SmallVector>
{
    size_t operator()(const rx::mtl::SmallVector &key) const
    {
        return angle::ComputeGenericHash(key.data(), key.size());
    }
};
}  // namespace std

namespace rx
{
namespace mtl
{

// This class wraps Objective-C pointer inside, it will manage the lifetime of
// the Objective-C pointer. Changing pointer is not supported outside subclass.
template <typename T>
class WrappedObject
{
  public:
    WrappedObject() = default;
    ~WrappedObject() { release(); }

    bool valid() const { return (mMetalObject != nil); }

    T get() const { return mMetalObject; }
    inline void reset() { release(); }

    operator T() const { return get(); }

  protected:
    inline void set(T obj) { retainAssign(obj); }

    void retainAssign(T obj)
    {
        T retained = obj;
#if !__has_feature(objc_arc)
        [retained retain];
#endif
        release();
        mMetalObject = obj;
    }

  private:
    void release()
    {
#if !__has_feature(objc_arc)
        [mMetalObject release];
#endif
        mMetalObject = nil;
    }

    T mMetalObject = nil;
};

// This class is similar to WrappedObject, however, it allows changing the
// internal pointer with public methods.
template <typename T>
class AutoObjCPtr : public WrappedObject<T>
{
  public:
    using ParentType = WrappedObject<T>;

    AutoObjCPtr() {}

    AutoObjCPtr(const std::nullptr_t &theNull) {}

    AutoObjCPtr(const AutoObjCPtr &src) { this->retainAssign(src.get()); }

    AutoObjCPtr(AutoObjCPtr &&src) { this->transfer(std::forward<AutoObjCPtr>(src)); }

    // Take ownership of the pointer
    AutoObjCPtr(T &&src)
    {
        this->retainAssign(src);
        src = nil;
    }

    AutoObjCPtr &operator=(const AutoObjCPtr &src)
    {
        this->retainAssign(src.get());
        return *this;
    }

    AutoObjCPtr &operator=(AutoObjCPtr &&src)
    {
        this->transfer(std::forward<AutoObjCPtr>(src));
        return *this;
    }

    // Take ownership of the pointer
    AutoObjCPtr &operator=(T &&src)
    {
        this->retainAssign(src);
        src = nil;
        return *this;
    }

    AutoObjCPtr &operator=(const std::nullptr_t &theNull)
    {
        this->set(nil);
        return *this;
    }

    bool operator==(const AutoObjCPtr &rhs) const { return (*this) == rhs.get(); }

    bool operator==(T rhs) const { return this->get() == rhs; }

    bool operator==(const std::nullptr_t &theNull) const { return this->get(); }

    inline operator bool() { return this->get(); }

    bool operator!=(const AutoObjCPtr &rhs) const { return (*this) != rhs.get(); }

    bool operator!=(T rhs) const { return this->get() != rhs; }

    using ParentType::retainAssign;

  private:
    void transfer(AutoObjCPtr &&src)
    {
        this->retainAssign(std::move(src.get()));
        src.reset();
    }
};

template <typename T>
using AutoObjCObj = AutoObjCPtr<T *>;

// NOTE: SharedEvent is only declared on iOS 12.0+ or mac 10.14+
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
using SharedEventRef = AutoObjCPtr<id<MTLSharedEvent>>;
#else
using SharedEventRef                                       = AutoObjCObj<NSObject>;
#endif

class CommandQueue;
class ErrorHandler
{
  public:
    virtual ~ErrorHandler() {}

    virtual void handleError(GLenum error,
                             const char *file,
                             const char *function,
                             unsigned int line) = 0;

    virtual void handleError(NSError *_Nullable error,
                             const char *file,
                             const char *function,
                             unsigned int line) = 0;
};

class Context : public ErrorHandler
{
  public:
    Context(DisplayMtl *displayMtl);
    _Nullable id<MTLDevice> getMetalDevice() const;
    mtl::CommandQueue &cmdQueue();

    DisplayMtl *getDisplay() const { return mDisplay; }

  protected:
    DisplayMtl *mDisplay;
};

#define ANGLE_MTL_CHECK(context, test, error)                                \
    do                                                                       \
    {                                                                        \
        if (ANGLE_UNLIKELY(!(test)))                                         \
        {                                                                    \
            context->handleError(error, __FILE__, ANGLE_FUNCTION, __LINE__); \
            return angle::Result::Stop;                                      \
        }                                                                    \
    } while (0)

#define ANGLE_MTL_TRY(context, test) ANGLE_MTL_CHECK(context, test, GL_INVALID_OPERATION)

#define ANGLE_MTL_UNREACHABLE(context) \
    UNREACHABLE();                     \
    ANGLE_MTL_TRY(context, false)

}  // namespace mtl
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_MTL_COMMON_H_ */
