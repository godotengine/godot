//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// renderergl_utils.h: Conversion functions and other utility routines
// specific to the OpenGL renderer.

#ifndef LIBANGLE_RENDERER_GL_RENDERERGLUTILS_H_
#define LIBANGLE_RENDERER_GL_RENDERERGLUTILS_H_

#include "common/debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/Version.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/driver_utils.h"
#include "libANGLE/renderer/gl/functionsgl_typedefs.h"

#include <string>
#include <vector>

namespace angle
{
struct FeaturesGL;
struct FrontendFeatures;
}  // namespace angle

namespace gl
{
struct Caps;
class TextureCapsMap;
struct Extensions;
struct Version;
}  // namespace gl

namespace rx
{
class BlitGL;
class ClearMultiviewGL;
class ContextGL;
class FunctionsGL;
class StateManagerGL;
enum class MultiviewImplementationTypeGL
{
    NV_VIEWPORT_ARRAY2,
    UNSPECIFIED
};

VendorID GetVendorID(const FunctionsGL *functions);

// Helpers for extracting the GL helper objects out of a context
const FunctionsGL *GetFunctionsGL(const gl::Context *context);
StateManagerGL *GetStateManagerGL(const gl::Context *context);
BlitGL *GetBlitGL(const gl::Context *context);
ClearMultiviewGL *GetMultiviewClearer(const gl::Context *context);
const angle::FeaturesGL &GetFeaturesGL(const gl::Context *context);

// Clear all errors on the stored context, emits console warnings
void ClearErrors(const gl::Context *context,
                 const char *file,
                 const char *function,
                 unsigned int line);

// Check for a single error
angle::Result CheckError(const gl::Context *context,
                         const char *call,
                         const char *file,
                         const char *function,
                         unsigned int line);

#define ANGLE_GL_TRY_ALWAYS_CHECK(context, call)                      \
    (ClearErrors(context, __FILE__, __FUNCTION__, __LINE__), (call)); \
    ANGLE_TRY(CheckError(context, #call, __FILE__, __FUNCTION__, __LINE__))

#if defined(ANGLE_ENABLE_ASSERTS)
#    define ANGLE_GL_TRY(context, call) ANGLE_GL_TRY_ALWAYS_CHECK(context, call)
#else
#    define ANGLE_GL_TRY(context, call) call
#endif

namespace nativegl_gl
{

void GenerateCaps(const FunctionsGL *functions,
                  const angle::FeaturesGL &features,
                  gl::Caps *caps,
                  gl::TextureCapsMap *textureCapsMap,
                  gl::Extensions *extensions,
                  gl::Version *maxSupportedESVersion,
                  MultiviewImplementationTypeGL *multiviewImplementationType);

void InitializeFeatures(const FunctionsGL *functions, angle::FeaturesGL *features);
void InitializeFrontendFeatures(const FunctionsGL *functions, angle::FrontendFeatures *features);
}  // namespace nativegl_gl

namespace nativegl
{
bool SupportsDrawIndirect(const FunctionsGL *functions);
bool SupportsCompute(const FunctionsGL *functions);
bool SupportsFenceSync(const FunctionsGL *functions);
bool SupportsOcclusionQueries(const FunctionsGL *functions);
bool SupportsNativeRendering(const FunctionsGL *functions,
                             gl::TextureType type,
                             GLenum internalFormat);
bool SupportsTexImage(gl::TextureType type);
bool UseTexImage2D(gl::TextureType textureType);
bool UseTexImage3D(gl::TextureType textureType);
GLenum GetTextureBindingQuery(gl::TextureType textureType);
GLenum GetBufferBindingQuery(gl::BufferBinding bufferBinding);
std::string GetBufferBindingString(gl::BufferBinding bufferBinding);
}  // namespace nativegl

bool CanMapBufferForRead(const FunctionsGL *functions);
uint8_t *MapBufferRangeWithFallback(const FunctionsGL *functions,
                                    GLenum target,
                                    size_t offset,
                                    size_t length,
                                    GLbitfield access);

angle::Result ShouldApplyLastRowPaddingWorkaround(ContextGL *contextGL,
                                                  const gl::Extents &size,
                                                  const gl::PixelStoreStateBase &state,
                                                  const gl::Buffer *pixelBuffer,
                                                  GLenum format,
                                                  GLenum type,
                                                  bool is3D,
                                                  const void *pixels,
                                                  bool *shouldApplyOut);

struct ContextCreationTry
{
    enum class Type
    {
        DESKTOP_CORE,
        DESKTOP_LEGACY,
        ES,
    };

    ContextCreationTry(EGLint displayType, Type type, gl::Version version)
        : displayType(displayType), type(type), version(version)
    {}

    EGLint displayType;
    Type type;
    gl::Version version;
};

std::vector<ContextCreationTry> GenerateContextCreationToTry(EGLint requestedType, bool isMesaGLX);
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_RENDERERGLUTILS_H_
