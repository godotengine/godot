//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// android_util.cpp: Utilities for the using the Android platform

#include "common/android_util.h"
#include "common/debug.h"

#include <cstdint>

// Taken from cutils/native_handle.h:
// https://android.googlesource.com/platform/system/core/+/master/libcutils/include/cutils/native_handle.h
typedef struct native_handle
{
    int version; /* sizeof(native_handle_t) */
    int numFds;  /* number of file-descriptors at &data[0] */
    int numInts; /* number of ints at &data[numFds] */
    int data[0]; /* numFds + numInts ints */
} native_handle_t;

// Taken from nativebase/nativebase.h
// https://android.googlesource.com/platform/frameworks/native/+/master/libs/nativebase/include/nativebase/nativebase.h
typedef const native_handle_t *buffer_handle_t;

typedef struct android_native_base_t
{
    /* a magic value defined by the actual EGL native type */
    int magic;
    /* the sizeof() of the actual EGL native type */
    int version;
    void *reserved[4];
    /* reference-counting interface */
    void (*incRef)(struct android_native_base_t *base);
    void (*decRef)(struct android_native_base_t *base);
} android_native_base_t;

typedef struct ANativeWindowBuffer
{
    struct android_native_base_t common;
    int width;
    int height;
    int stride;
    int format;
    int usage_deprecated;
    uintptr_t layerCount;
    void *reserved[1];
    const native_handle_t *handle;
    uint64_t usage;
    // we needed extra space for storing the 64-bits usage flags
    // the number of slots to use from reserved_proc depends on the
    // architecture.
    void *reserved_proc[8 - (sizeof(uint64_t) / sizeof(void *))];
} ANativeWindowBuffer_t;

// Taken from android/hardware_buffer.h
// https://android.googlesource.com/platform/frameworks/native/+/master/libs/nativewindow/include/android/hardware_buffer.h

// AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM AHARDWAREBUFFER_FORMAT_B4G4R4A4_UNORM,
// AHARDWAREBUFFER_FORMAT_B5G5R5A1_UNORM formats were deprecated and re-added explicitly.

// clang-format off
/**
 * Buffer pixel formats.
 */
enum {
    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_R8G8B8A8_UNORM
     *   OpenGL ES: GL_RGBA8
     */
    AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM           = 1,

    /**
     * 32 bits per pixel, 8 bits per channel format where alpha values are
     * ignored (always opaque).
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_R8G8B8A8_UNORM
     *   OpenGL ES: GL_RGB8
     */
    AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM           = 2,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_R8G8B8_UNORM
     *   OpenGL ES: GL_RGB8
     */
    AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM             = 3,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_R5G6B5_UNORM_PACK16
     *   OpenGL ES: GL_RGB565
     */
    AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM             = 4,

    AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM           = 5,
    AHARDWAREBUFFER_FORMAT_B5G5R5A1_UNORM           = 6,
    AHARDWAREBUFFER_FORMAT_B4G4R4A4_UNORM           = 7,
    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_R16G16B16A16_SFLOAT
     *   OpenGL ES: GL_RGBA16F
     */
    AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT       = 0x16,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_A2B10G10R10_UNORM_PACK32
     *   OpenGL ES: GL_RGB10_A2
     */
    AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM        = 0x2b,

    /**
     * An opaque binary blob format that must have height 1, with width equal to
     * the buffer size in bytes.
     */
    AHARDWAREBUFFER_FORMAT_BLOB                     = 0x21,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_D16_UNORM
     *   OpenGL ES: GL_DEPTH_COMPONENT16
     */
    AHARDWAREBUFFER_FORMAT_D16_UNORM                = 0x30,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_X8_D24_UNORM_PACK32
     *   OpenGL ES: GL_DEPTH_COMPONENT24
     */
    AHARDWAREBUFFER_FORMAT_D24_UNORM                = 0x31,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_D24_UNORM_S8_UINT
     *   OpenGL ES: GL_DEPTH24_STENCIL8
     */
    AHARDWAREBUFFER_FORMAT_D24_UNORM_S8_UINT        = 0x32,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_D32_SFLOAT
     *   OpenGL ES: GL_DEPTH_COMPONENT32F
     */
    AHARDWAREBUFFER_FORMAT_D32_FLOAT                = 0x33,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_D32_SFLOAT_S8_UINT
     *   OpenGL ES: GL_DEPTH32F_STENCIL8
     */
    AHARDWAREBUFFER_FORMAT_D32_FLOAT_S8_UINT        = 0x34,

    /**
     * Corresponding formats:
     *   Vulkan: VK_FORMAT_S8_UINT
     *   OpenGL ES: GL_STENCIL_INDEX8
     */
    AHARDWAREBUFFER_FORMAT_S8_UINT                  = 0x35,

    /**
     * YUV 420 888 format.
     * Must have an even width and height. Can be accessed in OpenGL
     * shaders through an external sampler. Does not support mip-maps
     * cube-maps or multi-layered textures.
     */
    AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420             = 0x23,
};
// clang-format on

namespace
{

// In the Android system:
// - AHardwareBuffer is essentially a typedef of GraphicBuffer. Conversion functions simply
// reinterpret_cast.
// - GraphicBuffer inherits from two base classes, ANativeWindowBuffer and RefBase.
//
// GraphicBuffer implements a getter for ANativeWindowBuffer (getNativeBuffer) by static_casting
// itself to its base class ANativeWindowBuffer. The offset of the ANativeWindowBuffer pointer
// from the GraphicBuffer pointer is 16 bytes. This is likely due to two pointers: The vtable of
// GraphicBuffer and the one pointer member of the RefBase class.
//
// This is not future proof at all. We need to look into getting utilities added to Android to
// perform this cast for us.
constexpr int kAHardwareBufferToANativeWindowBufferOffset = static_cast<int>(sizeof(void *)) * 2;

template <typename T1, typename T2>
T1 *offsetPointer(T2 *ptr, int bytes)
{
    return reinterpret_cast<T1 *>(reinterpret_cast<intptr_t>(ptr) + bytes);
}

}  // anonymous namespace

namespace angle
{

namespace android
{

ANativeWindowBuffer *ClientBufferToANativeWindowBuffer(EGLClientBuffer clientBuffer)
{
    return reinterpret_cast<ANativeWindowBuffer *>(clientBuffer);
}

void GetANativeWindowBufferProperties(const ANativeWindowBuffer *buffer,
                                      int *width,
                                      int *height,
                                      int *depth,
                                      int *pixelFormat)
{
    *width       = buffer->width;
    *height      = buffer->height;
    *depth       = static_cast<int>(buffer->layerCount);
    *height      = buffer->height;
    *pixelFormat = buffer->format;
}

GLenum NativePixelFormatToGLInternalFormat(int pixelFormat)
{
    switch (pixelFormat)
    {
        case AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM:
            return GL_RGBA8;
        case AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM:
            return GL_RGB8;
        case AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM:
            return GL_RGB8;
        case AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM:
            return GL_RGB565;
        case AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM:
            return GL_BGRA8_EXT;
        case AHARDWAREBUFFER_FORMAT_B5G5R5A1_UNORM:
            return GL_RGB5_A1;
        case AHARDWAREBUFFER_FORMAT_B4G4R4A4_UNORM:
            return GL_RGBA4;
        case AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT:
            return GL_RGBA16F;
        case AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM:
            return GL_RGB10_A2;
        case AHARDWAREBUFFER_FORMAT_BLOB:
            return GL_NONE;
        case AHARDWAREBUFFER_FORMAT_D16_UNORM:
            return GL_DEPTH_COMPONENT16;
        case AHARDWAREBUFFER_FORMAT_D24_UNORM:
            return GL_DEPTH_COMPONENT24;
        case AHARDWAREBUFFER_FORMAT_D24_UNORM_S8_UINT:
            return GL_DEPTH24_STENCIL8;
        case AHARDWAREBUFFER_FORMAT_D32_FLOAT:
            return GL_DEPTH_COMPONENT32F;
        case AHARDWAREBUFFER_FORMAT_D32_FLOAT_S8_UINT:
            return GL_DEPTH32F_STENCIL8;
        case AHARDWAREBUFFER_FORMAT_S8_UINT:
            return GL_STENCIL_INDEX8;
        case AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420:
            return GL_RGB8;
        default:
            // Treat unknown formats as RGB. They are vendor-specific YUV formats that would sample
            // as RGB.
            WARN() << "Unknown pixelFormat: " << pixelFormat << ". Treating as RGB8";
            return GL_RGB8;
    }
}

AHardwareBuffer *ANativeWindowBufferToAHardwareBuffer(ANativeWindowBuffer *windowBuffer)
{
    return offsetPointer<AHardwareBuffer>(windowBuffer,
                                          -kAHardwareBufferToANativeWindowBufferOffset);
}

EGLClientBuffer AHardwareBufferToClientBuffer(const AHardwareBuffer *hardwareBuffer)
{
    return offsetPointer<EGLClientBuffer>(hardwareBuffer,
                                          kAHardwareBufferToANativeWindowBufferOffset);
}

}  // namespace android
}  // namespace angle
