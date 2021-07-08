//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// formatutilsgl.cpp: Queries for GL image formats and their translations to native
// GL formats.

#include "libANGLE/renderer/gl/formatutilsgl.h"

#include <limits>

#include "anglebase/no_destructor.h"
#include "common/string_utils.h"
#include "libANGLE/formatutils.h"
#include "platform/FeaturesGL.h"

namespace rx
{

namespace nativegl
{

SupportRequirement::SupportRequirement()
    : version(std::numeric_limits<GLuint>::max(), std::numeric_limits<GLuint>::max()),
      versionExtensions(),
      requiredExtensions()
{}

SupportRequirement::SupportRequirement(const SupportRequirement &other) = default;

SupportRequirement::~SupportRequirement() {}

InternalFormat::InternalFormat() : texture(), filter(), textureAttachment(), renderbuffer() {}

InternalFormat::InternalFormat(const InternalFormat &other) = default;

InternalFormat::~InternalFormat() {}

// supported = version || vertexExt
static inline SupportRequirement VersionOrExts(GLuint major,
                                               GLuint minor,
                                               const std::string &versionExt)
{
    SupportRequirement requirement;
    requirement.version.major = major;
    requirement.version.minor = minor;
    angle::SplitStringAlongWhitespace(versionExt, &requirement.versionExtensions);
    return requirement;
}

// supported = version
static inline SupportRequirement VersionOnly(GLuint major, GLuint minor)
{
    SupportRequirement requirement;
    requirement.version.major = major;
    requirement.version.minor = minor;
    return requirement;
}

// supported = any one of sets in exts
static inline SupportRequirement ExtsOnly(const std::vector<std::string> &exts)
{
    SupportRequirement requirement;
    requirement.version.major = 0;
    requirement.version.minor = 0;
    requirement.requiredExtensions.resize(exts.size());
    for (size_t i = 0; i < exts.size(); i++)
    {
        angle::SplitStringAlongWhitespace(exts[i], &requirement.requiredExtensions[i]);
    }
    return requirement;
}

// supported = ext
static inline SupportRequirement ExtsOnly(const std::string &ext)
{
    return ExtsOnly(std::vector<std::string>({ext}));
}

// supported = ext1 || ext2
static inline SupportRequirement ExtsOnly(const std::string &ext1, const std::string &ext2)
{
    return ExtsOnly(std::vector<std::string>({ext1, ext2}));
}

// supported = true
static inline SupportRequirement AlwaysSupported()
{
    SupportRequirement requirement;
    requirement.version.major = 0;
    requirement.version.minor = 0;
    return requirement;
}

// supported = false
static inline SupportRequirement NeverSupported()
{
    SupportRequirement requirement;
    requirement.version.major = std::numeric_limits<GLuint>::max();
    requirement.version.minor = std::numeric_limits<GLuint>::max();
    return requirement;
}

struct InternalFormatInfo
{
    InternalFormat glesInfo;
    InternalFormat glInfo;
};

typedef std::pair<GLenum, InternalFormatInfo> InternalFormatInfoPair;
typedef std::map<GLenum, InternalFormatInfo> InternalFormatInfoMap;

// A helper function to insert data into the format map with fewer characters.
static inline void InsertFormatMapping(InternalFormatInfoMap *map,
                                       GLenum internalFormat,
                                       const SupportRequirement &desktopTexture,
                                       const SupportRequirement &desktopFilter,
                                       const SupportRequirement &desktopRender,
                                       const SupportRequirement &esTexture,
                                       const SupportRequirement &esFilter,
                                       const SupportRequirement &esTextureAttachment,
                                       const SupportRequirement &esRenderbufferAttachment)
{
    InternalFormatInfo formatInfo;
    formatInfo.glInfo.texture = desktopTexture;
    formatInfo.glInfo.filter  = desktopFilter;
    // No difference spotted yet in Desktop GL texture attachment and renderbuffer capabilities
    formatInfo.glInfo.textureAttachment   = desktopRender;
    formatInfo.glInfo.renderbuffer        = desktopRender;
    formatInfo.glesInfo.texture           = esTexture;
    formatInfo.glesInfo.filter            = esFilter;
    formatInfo.glesInfo.textureAttachment = esTextureAttachment;
    formatInfo.glesInfo.renderbuffer      = esRenderbufferAttachment;
    map->insert(std::make_pair(internalFormat, formatInfo));
}

// Note 1: This map is used to determine extensions support, which is based on checking support for
// sized formats (this is ANGLE implementation limitation - D3D backend supports only sized formats)
// In order to determine support for extensions which introduce unsized formats, this map would say
// that a corresponding sized format is supported, instead. Thus, if this map says that a sized
// format is supported, this means that either the actual sized format or a corresponding unsized
// format is supported by the native driver.
// For example, GL_EXT_texture_rg provides support for RED_EXT format with UNSIGNED_BYTE type.
// Therefore, DetermineRGTextureSupport checks for GL_R8 support. Therefore this map says that
// GL_R8 (and not RED_EXT) is supported if GL_EXT_texture_rg is available. GL_R8 itself
// is supported in ES3, thus the combined condition is VersionOrExts(3, 0, "GL_EXT_texture_rg").
//
// Note 2: Texture Attachment support is checked also by SupportsNativeRendering().
// Unsized formats appear in this map for this reason. The assumption is
// that SupportsNativeRendering() will not check sized formats in the ES2 frontend
// and the information in unsized formats is correct, and not merged like for sized formats.
// In the ES3 frontend, it could happen that SupportsNativeRendering() would be wrong,
// but this will be mitigated by fall back to CPU-readback in TextureGL::copySubTextureHelper().
//
// Note 3: Because creating renderbuffers with unsized formats is impossible,
// the value of renderbuffer support is actually correct for the sized formats.
//
// Note 4: To determine whether a format is filterable, one must check both "Filter" and "Texture"
// support, like it is done in GenerateTextureFormatCaps().
// On the other hand, "Texture Attachment" support formula is self-contained.
//
// TODO(ynovikov): http://anglebug.com/2846 Verify support fields of BGRA, depth, stencil and
// compressed formats, and all formats for Desktop GL.
static InternalFormatInfoMap BuildInternalFormatInfoMap()
{
    InternalFormatInfoMap map;

    // clang-format off
    //                       | Format              | OpenGL texture support                          | Filter           | OpenGL render support                        | OpenGL ES texture support                 | Filter           | OpenGL ES texture attachment support    | OpenGL ES renderbuffer support           |
    InsertFormatMapping(&map, GL_R8,                VersionOrExts(3, 0, "GL_ARB_texture_rg"),         AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOrExts(3, 0, "GL_EXT_texture_rg"),   AlwaysSupported(), VersionOrExts(3, 0, "GL_EXT_texture_rg"), VersionOrExts(3, 0, "GL_EXT_texture_rg")  );
    InsertFormatMapping(&map, GL_R8_SNORM,          VersionOnly(3, 1),                                AlwaysSupported(), NeverSupported(),                              VersionOnly(3, 0),                          AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RG8,               VersionOrExts(3, 0, "GL_ARB_texture_rg"),         AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOrExts(3, 0, "GL_EXT_texture_rg"),   AlwaysSupported(), VersionOrExts(3, 0, "GL_EXT_texture_rg"), VersionOrExts(3, 0, "GL_EXT_texture_rg")  );
    InsertFormatMapping(&map, GL_RG8_SNORM,         VersionOnly(3, 1),                                AlwaysSupported(), NeverSupported(),                              VersionOnly(3, 0),                          AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB8,              AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             AlwaysSupported(),                          AlwaysSupported(), VersionOnly(2, 0),                        VersionOrExts(3, 0, "GL_OES_rgb8_rgba8")  );
    InsertFormatMapping(&map, GL_RGB8_SNORM,        VersionOnly(3, 1),                                AlwaysSupported(), NeverSupported(),                              VersionOnly(3, 0),                          AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB565,            AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             AlwaysSupported(),                          AlwaysSupported(), AlwaysSupported(),                        AlwaysSupported()                         );
    InsertFormatMapping(&map, GL_RGBA4,             AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             AlwaysSupported(),                          AlwaysSupported(), AlwaysSupported(),                        AlwaysSupported()                         );
    InsertFormatMapping(&map, GL_RGB5_A1,           AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             AlwaysSupported(),                          AlwaysSupported(), AlwaysSupported(),                        AlwaysSupported()                         );
    InsertFormatMapping(&map, GL_RGBA8,             AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             AlwaysSupported(),                          AlwaysSupported(), VersionOnly(2, 0),                        VersionOrExts(3, 0, "GL_OES_rgb8_rgba8")  );
    InsertFormatMapping(&map, GL_RGBA8_SNORM,       VersionOnly(3, 1),                                AlwaysSupported(), NeverSupported(),                              VersionOnly(3, 0),                          AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB10_A2,          AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             VersionOnly(3, 0),                          AlwaysSupported(), VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RGB10_A2UI,        VersionOrExts(3, 3, "GL_ARB_texture_rgb10_a2ui"), NeverSupported(),  AlwaysSupported(),                             VersionOnly(3, 0),                          NeverSupported(),  AlwaysSupported(),                        AlwaysSupported()                         );
    InsertFormatMapping(&map, GL_SRGB8,             VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),       AlwaysSupported(), VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),    VersionOrExts(3, 0, "GL_EXT_sRGB"),         AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_SRGB8_ALPHA8,      VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),       AlwaysSupported(), VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),    VersionOrExts(3, 0, "GL_EXT_sRGB"),         AlwaysSupported(), VersionOrExts(3, 0, "GL_EXT_sRGB"),       VersionOrExts(3, 0, "GL_EXT_sRGB")        );
    InsertFormatMapping(&map, GL_R8I,               VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_R8UI,              VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_R16I,              VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_R16UI,             VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_R32I,              VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_R32UI,             VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RG8I,              VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RG8UI,             VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RG16I,             VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RG16UI,            VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RG32I,             VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RG32UI,            VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RGB8I,             VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  NeverSupported(),                              VersionOnly(3, 0),                          NeverSupported(),  NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB8UI,            VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  NeverSupported(),                              VersionOnly(3, 0),                          NeverSupported(),  NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB16I,            VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  NeverSupported(),                              VersionOnly(3, 0),                          NeverSupported(),  NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB16UI,           VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  NeverSupported(),                              VersionOnly(3, 0),                          NeverSupported(),  NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB32I,            VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  NeverSupported(),                              VersionOnly(3, 0),                          NeverSupported(),  NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB32UI,           VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  NeverSupported(),                              VersionOnly(3, 0),                          NeverSupported(),  NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGBA8I,            VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  VersionOrExts(3, 0, "GL_EXT_texture_integer"), VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RGBA8UI,           VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  VersionOrExts(3, 0, "GL_EXT_texture_integer"), VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RGBA16I,           VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  VersionOrExts(3, 0, "GL_EXT_texture_integer"), VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RGBA16UI,          VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  VersionOrExts(3, 0, "GL_EXT_texture_integer"), VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RGBA32I,           VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  VersionOrExts(3, 0, "GL_EXT_texture_integer"), VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );
    InsertFormatMapping(&map, GL_RGBA32UI,          VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  VersionOrExts(3, 0, "GL_EXT_texture_integer"), VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        VersionOnly(3, 0)                         );

    // Unsized formats
    InsertFormatMapping(&map, GL_ALPHA,             NeverSupported(),                                 NeverSupported(),  NeverSupported(),                              AlwaysSupported(),                          AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_LUMINANCE,         NeverSupported(),                                 NeverSupported(),  NeverSupported(),                              AlwaysSupported(),                          AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_LUMINANCE_ALPHA,   NeverSupported(),                                 NeverSupported(),  NeverSupported(),                              AlwaysSupported(),                          AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RED,               VersionOrExts(3, 0, "GL_ARB_texture_rg"),         AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg"),      ExtsOnly("GL_EXT_texture_rg"),              AlwaysSupported(), ExtsOnly("GL_EXT_texture_rg"),            NeverSupported()                          );
    InsertFormatMapping(&map, GL_RG,                VersionOrExts(3, 0, "GL_ARB_texture_rg"),         AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg"),      ExtsOnly("GL_EXT_texture_rg"),              AlwaysSupported(), ExtsOnly("GL_EXT_texture_rg"),            NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB,               AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             AlwaysSupported(),                          AlwaysSupported(), VersionOnly(2, 0),                        NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGBA,              AlwaysSupported(),                                AlwaysSupported(), AlwaysSupported(),                             AlwaysSupported(),                          AlwaysSupported(), VersionOnly(2, 0),                        NeverSupported()                          );
    InsertFormatMapping(&map, GL_RED_INTEGER,       VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        NeverSupported()                          );
    InsertFormatMapping(&map, GL_RG_INTEGER,        VersionOrExts(3, 0, "GL_ARB_texture_rg"),         NeverSupported(),  VersionOrExts(3, 0, "GL_ARB_texture_rg"),      VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGB_INTEGER,       VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  NeverSupported(),                              VersionOnly(3, 0),                          NeverSupported(),  NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_RGBA_INTEGER,      VersionOrExts(3, 0, "GL_EXT_texture_integer"),    NeverSupported(),  VersionOrExts(3, 0, "GL_EXT_texture_integer"), VersionOnly(3, 0),                          NeverSupported(),  VersionOnly(3, 0),                        NeverSupported()                          );
    InsertFormatMapping(&map, GL_SRGB,              VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),       AlwaysSupported(), VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),    ExtsOnly("GL_EXT_sRGB"),                    AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_SRGB_ALPHA,        VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),       AlwaysSupported(), VersionOrExts(2, 1, "GL_EXT_texture_sRGB"),    ExtsOnly("GL_EXT_sRGB"),                    AlwaysSupported(), ExtsOnly("GL_EXT_sRGB"),                  NeverSupported()                          );

    // From GL_EXT_texture_format_BGRA8888
    InsertFormatMapping(&map, GL_BGRA8_EXT,         VersionOrExts(1, 2, "GL_EXT_bgra"),               AlwaysSupported(), VersionOrExts(1, 2, "GL_EXT_bgra"),            ExtsOnly("GL_EXT_texture_format_BGRA8888"), AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );
    InsertFormatMapping(&map, GL_BGRA_EXT,          VersionOrExts(1, 2, "GL_EXT_bgra"),               AlwaysSupported(), VersionOrExts(1, 2, "GL_EXT_bgra"),            ExtsOnly("GL_EXT_texture_format_BGRA8888"), AlwaysSupported(), NeverSupported(),                         NeverSupported()                          );

    // Floating point formats
    // Note 1: GL_EXT_texture_shared_exponent and GL_ARB_color_buffer_float suggest that RGB9_E5
    // would be renderable, but once support for renderable float textures got rolled into core GL
    // spec it wasn't intended to be renderable. In practice it's not reliably renderable even
    // with the extensions, there's a known bug in at least NVIDIA driver version 370.
    //
    // Note 2: It's a bit unclear whether texture attachments with GL_RGB16F should be supported
    // in ES3 with GL_EXT_color_buffer_half_float. Probably not, since in ES3 type is HALF_FLOAT,
    // but GL_EXT_color_buffer_half_float is applicable only to type HALF_FLOAT_OES.
    //
    // Note 3: GL_EXT_color_buffer_float implies that ES3.0 is supported, this simplifies the check.
    //
    //                       | Format              | OpenGL texture support                                       | Filter           | OpenGL render support                                                                  | OpenGL ES texture support                                         | Filter                                                 | OpenGL ES texture attachment support                                                                                                      | OpenGL ES renderbuffer support                                                                                    |
    InsertFormatMapping(&map, GL_R11F_G11F_B10F,    VersionOrExts(3, 0, "GL_EXT_packed_float"),                    AlwaysSupported(), VersionOrExts(3, 0, "GL_EXT_packed_float GL_ARB_color_buffer_float"),                    VersionOnly(3, 0),                                                  AlwaysSupported(),                                       ExtsOnly("GL_EXT_color_buffer_float"),                                                                                                      ExtsOnly("GL_EXT_color_buffer_float")                                                                              );
    InsertFormatMapping(&map, GL_RGB9_E5,           VersionOrExts(3, 0, "GL_EXT_texture_shared_exponent"),         AlwaysSupported(), NeverSupported(),                                                                        VersionOnly(3, 0),                                                  AlwaysSupported(),                                       NeverSupported(),                                                                                                                           NeverSupported()                                                                                                   );
    InsertFormatMapping(&map, GL_R16F,              VersionOrExts(3, 0, "GL_ARB_texture_rg ARB_texture_float"),    AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg GL_ARB_texture_float GL_ARB_color_buffer_float"), VersionOrExts(3, 0, "GL_OES_texture_half_float GL_EXT_texture_rg"), VersionOrExts(3, 0, "GL_OES_texture_half_float_linear"), ExtsOnly("GL_EXT_texture_storage GL_OES_texture_half_float GL_EXT_texture_rg GL_EXT_color_buffer_half_float", "GL_EXT_color_buffer_float"), ExtsOnly("GL_EXT_texture_rg GL_OES_texture_half_float GL_EXT_color_buffer_half_float", "GL_EXT_color_buffer_float"));
    InsertFormatMapping(&map, GL_RG16F,             VersionOrExts(3, 0, "GL_ARB_texture_rg ARB_texture_float"),    AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg GL_ARB_texture_float GL_ARB_color_buffer_float"), VersionOrExts(3, 0, "GL_OES_texture_half_float GL_EXT_texture_rg"), VersionOrExts(3, 0, "GL_OES_texture_half_float_linear"), ExtsOnly("GL_EXT_texture_storage GL_OES_texture_half_float GL_EXT_texture_rg GL_EXT_color_buffer_half_float", "GL_EXT_color_buffer_float"), ExtsOnly("GL_EXT_texture_rg GL_OES_texture_half_float GL_EXT_color_buffer_half_float", "GL_EXT_color_buffer_float"));
    InsertFormatMapping(&map, GL_RGB16F,            VersionOrExts(3, 0, "GL_ARB_texture_float"),                   AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_float GL_ARB_color_buffer_float"),                   VersionOrExts(3, 0, "GL_OES_texture_half_float"),                   VersionOrExts(3, 0, "GL_OES_texture_half_float_linear"), ExtsOnly("GL_EXT_texture_storage GL_OES_texture_half_float GL_EXT_color_buffer_half_float"),                                                ExtsOnly("GL_OES_texture_half_float GL_EXT_color_buffer_half_float")                                               );
    InsertFormatMapping(&map, GL_RGBA16F,           VersionOrExts(3, 0, "GL_ARB_texture_float"),                   AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_float GL_ARB_color_buffer_float"),                   VersionOrExts(3, 0, "GL_OES_texture_half_float"),                   VersionOrExts(3, 0, "GL_OES_texture_half_float_linear"), ExtsOnly("GL_OES_texture_half_float GL_EXT_color_buffer_half_float", "GL_EXT_color_buffer_float"),                                          ExtsOnly("GL_OES_texture_half_float GL_EXT_color_buffer_half_float", "GL_EXT_color_buffer_float")                  );
    InsertFormatMapping(&map, GL_R32F,              VersionOrExts(3, 0, "GL_ARB_texture_rg GL_ARB_texture_float"), AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg GL_ARB_texture_float GL_ARB_color_buffer_float"), VersionOrExts(3, 0, "GL_OES_texture_float GL_EXT_texture_rg"),      ExtsOnly("GL_OES_texture_float_linear"),                 ExtsOnly("GL_EXT_color_buffer_float"),                                                                                                      ExtsOnly("GL_EXT_color_buffer_float")                                                                              );
    InsertFormatMapping(&map, GL_RG32F,             VersionOrExts(3, 0, "GL_ARB_texture_rg GL_ARB_texture_float"), AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_rg GL_ARB_texture_float GL_ARB_color_buffer_float"), VersionOrExts(3, 0, "GL_OES_texture_float GL_EXT_texture_rg"),      ExtsOnly("GL_OES_texture_float_linear"),                 ExtsOnly("GL_EXT_color_buffer_float"),                                                                                                      ExtsOnly("GL_EXT_color_buffer_float")                                                                              );
    InsertFormatMapping(&map, GL_RGB32F,            VersionOrExts(3, 0, "GL_ARB_texture_float"),                   AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_float GL_ARB_color_buffer_float"),                   VersionOrExts(3, 0, "GL_OES_texture_float"),                        ExtsOnly("GL_OES_texture_float_linear"),                 NeverSupported(),                                                                                                                           NeverSupported()                                                                                                   );
    InsertFormatMapping(&map, GL_RGBA32F,           VersionOrExts(3, 0, "GL_ARB_texture_float"),                   AlwaysSupported(), VersionOrExts(3, 0, "GL_ARB_texture_float GL_ARB_color_buffer_float"),                   VersionOrExts(3, 0, "GL_OES_texture_float"),                        ExtsOnly("GL_OES_texture_float_linear"),                 ExtsOnly("GL_EXT_color_buffer_float"),                                                                                                      ExtsOnly("GL_EXT_color_buffer_float")                                                                              );

    // Depth stencil formats
    //                       | Format                  | OpenGL texture support                            | Filter                                     | OpenGL render support                             | OpenGL ES texture support                  | Filter                                     | OpenGL ES texture attachment support                                   | OpenGL ES renderbuffer support                                        |
    InsertFormatMapping(&map, GL_DEPTH_COMPONENT16,     VersionOnly(1, 5),                                  VersionOrExts(1, 5, "GL_ARB_depth_texture"), VersionOnly(1, 5),                                  VersionOnly(2, 0),                           VersionOrExts(3, 0, "GL_OES_depth_texture"), VersionOnly(2, 0),                                                       VersionOnly(2, 0)                                                      );
    InsertFormatMapping(&map, GL_DEPTH_COMPONENT24,     VersionOnly(1, 5),                                  VersionOrExts(1, 5, "GL_ARB_depth_texture"), VersionOnly(1, 5),                                  VersionOnly(2, 0),                           VersionOrExts(3, 0, "GL_OES_depth_texture"), VersionOnly(2, 0),                                                       VersionOnly(2, 0)                                                      );
    InsertFormatMapping(&map, GL_DEPTH_COMPONENT32_OES, VersionOnly(1, 5),                                  VersionOrExts(1, 5, "GL_ARB_depth_texture"), VersionOnly(1, 5),                                  ExtsOnly("GL_OES_depth_texture"),            AlwaysSupported(),                           ExtsOnly("GL_OES_depth32"),                                              ExtsOnly("GL_OES_depth32")                                             );
    InsertFormatMapping(&map, GL_DEPTH_COMPONENT32F,    VersionOrExts(3, 0, "GL_ARB_depth_buffer_float"),   AlwaysSupported(),                           VersionOrExts(3, 0, "GL_ARB_depth_buffer_float"),   VersionOnly(3, 0),                           VersionOrExts(3, 0, "GL_OES_depth_texture"), VersionOnly(3, 0),                                                       VersionOnly(3, 0)                                                      );
    InsertFormatMapping(&map, GL_STENCIL_INDEX8,        VersionOrExts(3, 0, "GL_EXT_packed_depth_stencil"), NeverSupported(),                            VersionOrExts(3, 0, "GL_EXT_packed_depth_stencil"), VersionOnly(2, 0),                           NeverSupported(),                            VersionOnly(2, 0),                                                       VersionOnly(2, 0)                                                      );
    InsertFormatMapping(&map, GL_DEPTH24_STENCIL8,      VersionOrExts(3, 0, "GL_ARB_framebuffer_object"),   VersionOrExts(3, 0, "GL_ARB_depth_texture"), VersionOrExts(3, 0, "GL_ARB_framebuffer_object"),   VersionOrExts(3, 0, "GL_OES_depth_texture"), AlwaysSupported(),                           VersionOrExts(3, 0, "GL_OES_depth_texture GL_OES_packed_depth_stencil"), VersionOrExts(3, 0, "GL_OES_depth_texture GL_OES_packed_depth_stencil"));
    InsertFormatMapping(&map, GL_DEPTH32F_STENCIL8,     VersionOrExts(3, 0, "GL_ARB_depth_buffer_float"),   AlwaysSupported(),                           VersionOrExts(3, 0, "GL_ARB_depth_buffer_float"),   VersionOnly(3, 0),                           AlwaysSupported(),                           VersionOnly(3, 0),                                                       VersionOnly(3, 0)                                                      );
    InsertFormatMapping(&map, GL_DEPTH_COMPONENT,       VersionOnly(1, 5),                                  VersionOrExts(1, 5, "GL_ARB_depth_texture"), VersionOnly(1, 5),                                  VersionOnly(2, 0),                           VersionOrExts(3, 0, "GL_OES_depth_texture"), VersionOnly(2, 0),                                                       VersionOnly(2, 0)                                                      );
    InsertFormatMapping(&map, GL_DEPTH_STENCIL,         VersionOnly(1, 5),                                  VersionOrExts(1, 5, "GL_ARB_depth_texture"), VersionOnly(1, 5),                                  VersionOnly(2, 0),                           VersionOrExts(3, 0, "GL_OES_depth_texture"), VersionOnly(2, 0),                                                       VersionOnly(2, 0)                                                      );

    // Luminance alpha formats
    //                       | Format                  | OpenGL texture support                      | Filter           | Render          | OpenGL ES texture support            | Filter                                      | OpenGL ES texture attachment support | OpenGL ES renderbuffer support |
    InsertFormatMapping(&map, GL_ALPHA8_EXT,             AlwaysSupported(),                           AlwaysSupported(), NeverSupported(), AlwaysSupported(),                     AlwaysSupported(),                            NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_LUMINANCE8_EXT,         AlwaysSupported(),                           AlwaysSupported(), NeverSupported(), AlwaysSupported(),                     AlwaysSupported(),                            NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_LUMINANCE8_ALPHA8_EXT,  AlwaysSupported(),                           AlwaysSupported(), NeverSupported(), AlwaysSupported(),                     AlwaysSupported(),                            NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_ALPHA16F_EXT,           VersionOrExts(3, 0, "GL_ARB_texture_float"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_half_float"), ExtsOnly("GL_OES_texture_half_float_linear"), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_LUMINANCE16F_EXT,       VersionOrExts(3, 0, "GL_ARB_texture_float"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_half_float"), ExtsOnly("GL_OES_texture_half_float_linear"), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_LUMINANCE_ALPHA16F_EXT, VersionOrExts(3, 0, "GL_ARB_texture_float"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_half_float"), ExtsOnly("GL_OES_texture_half_float_linear"), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_ALPHA32F_EXT,           VersionOrExts(3, 0, "GL_ARB_texture_float"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_float"),      ExtsOnly("GL_OES_texture_float_linear"),      NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_LUMINANCE32F_EXT,       VersionOrExts(3, 0, "GL_ARB_texture_float"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_float"),      ExtsOnly("GL_OES_texture_float_linear"),      NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_LUMINANCE_ALPHA32F_EXT, VersionOrExts(3, 0, "GL_ARB_texture_float"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_float"),      ExtsOnly("GL_OES_texture_float_linear"),      NeverSupported(),                      NeverSupported()                );

    // EXT_texture_compression_bptc formats
    //                       | Format                                   | OpenGL texture support                                | Filter           | Render          | OpenGL ES texture support                  | Filter           | OpenGL ES texture attachment support | OpenGL ES renderbuffer support |
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_BPTC_UNORM_EXT,         VersionOrExts(4, 2, "GL_ARB_texture_compression_bptc"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_EXT_texture_compression_bptc"), AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,   VersionOrExts(4, 2, "GL_ARB_texture_compression_bptc"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_EXT_texture_compression_bptc"), AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT,   VersionOrExts(4, 2, "GL_ARB_texture_compression_bptc"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_EXT_texture_compression_bptc"), AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT, VersionOrExts(4, 2, "GL_ARB_texture_compression_bptc"), AlwaysSupported(), NeverSupported(), ExtsOnly("GL_EXT_texture_compression_bptc"), AlwaysSupported(), NeverSupported(),                      NeverSupported()                 );

    // Compressed formats, From ES 3.0.1 spec, table 3.16
    //                       | Format                                      | OpenGL texture support                         | Filter           | Render          | OpenGL ES texture support                                                   | Filter           | OpenGL ES texture attachment support | OpenGL ES renderbuffer support |
    InsertFormatMapping(&map, GL_COMPRESSED_R11_EAC,                        VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_EAC_R11_unsigned_texture"),               AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SIGNED_R11_EAC,                 VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_EAC_R11_signed_texture"),                 AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RG11_EAC,                       VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_EAC_RG11_unsigned_texture"),              AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SIGNED_RG11_EAC,                VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_EAC_RG11_signed_texture"),                AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGB8_ETC2,                      VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_ETC2_RGB8_texture"),                      AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ETC2,                     VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_ETC2_sRGB8_texture"),                     AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,  VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_ETC2_punchthroughA_RGBA8_texture"),       AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_ETC2_punchthroughA_sRGB8_alpha_texture"), AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA8_ETC2_EAC,                 VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_ETC2_RGBA8_texture"),                     AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,          VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "OES_compressed_ETC2_sRGB8_alpha8_texture"),              AlwaysSupported(), NeverSupported(),                      NeverSupported()                );

    // From GL_EXT_texture_compression_dxt1
    //                       | Format                            | OpenGL texture support                         | Filter           | Render          | OpenGL ES texture support                    | Filter           | OpenGL ES texture attachment support | OpenGL ES renderbuffer support |
    InsertFormatMapping(&map, GL_COMPRESSED_RGB_S3TC_DXT1_EXT,    ExtsOnly("GL_EXT_texture_compression_s3tc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_EXT_texture_compression_dxt1"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,   ExtsOnly("GL_EXT_texture_compression_s3tc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_EXT_texture_compression_dxt1"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );

    // From GL_ANGLE_texture_compression_dxt3
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE, ExtsOnly("GL_EXT_texture_compression_s3tc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_ANGLE_texture_compression_dxt3"), AlwaysSupported(), NeverSupported(),                      NeverSupported()                );

    // From GL_ANGLE_texture_compression_dxt5
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE, ExtsOnly("GL_EXT_texture_compression_s3tc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_ANGLE_texture_compression_dxt5"), AlwaysSupported(), NeverSupported(),                      NeverSupported()                );

    // From GL_OES_compressed_ETC1_RGB8_texture
    InsertFormatMapping(&map, GL_ETC1_RGB8_OES,                   VersionOrExts(4, 3, "GL_ARB_ES3_compatibility"), AlwaysSupported(), NeverSupported(), VersionOrExts(3, 0, "GL_OES_compressed_ETC1_RGB8_texture"),       AlwaysSupported(), NeverSupported(),                      NeverSupported()                );

    // From GL_OES_texture_compression_astc
    //                       | Format                                  | OpenGL texture   | Filter         | Render          | OpenGL ES texture support                      | Filter           | ES attachment   | ES renderbuffer |
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_4x4_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_5x4_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_5x5_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_6x5_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_6x6_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_8x5_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_8x6_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_8x8_KHR,           NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_10x5_KHR,          NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_10x6_KHR,          NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_10x8_KHR,          NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_10x10_KHR,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_12x10_KHR,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_12x12_KHR,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR,   NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR,  NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR,  NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR,  NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_KHR_texture_compression_astc_ldr"), AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_3x3x3_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_4x3x3_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_4x4x3_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_4x4x4_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_5x4x4_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_5x5x4_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_5x5x5_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_6x5x5_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_6x6x5_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_ASTC_6x6x6_OES,         NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_3x3x3_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x3x3_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x3_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x4_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4x4_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x4_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x5_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5x5_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x5_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x6_OES, NeverSupported(), NeverSupported(), NeverSupported(), ExtsOnly("GL_OES_texture_compression_astc"),     AlwaysSupported(), NeverSupported(), NeverSupported());

    // From GL_IMG_texture_compression_pvrtc
    //                       | Format                               | OpenGL texture support                         | Filter           | Render          | OpenGL ES texture support                      | Filter           | OpenGL ES texture attachment support | OpenGL ES renderbuffer support |
    InsertFormatMapping(&map, GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG,    ExtsOnly("GL_IMG_texture_compression_pvrtc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_IMG_texture_compression_pvrtc"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG,    ExtsOnly("GL_IMG_texture_compression_pvrtc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_IMG_texture_compression_pvrtc"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG,   ExtsOnly("GL_IMG_texture_compression_pvrtc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_IMG_texture_compression_pvrtc"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG,   ExtsOnly("GL_IMG_texture_compression_pvrtc"),     AlwaysSupported(), NeverSupported(), ExtsOnly("GL_IMG_texture_compression_pvrtc"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );

    // From GL_EXT_pvrtc_sRGB
    //                       | Format                                      | OpenGL texture support  | Filter          | Render          | OpenGL ES texture support      | Filter           | OpenGL ES texture attachment support | OpenGL ES renderbuffer support |
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT,          NeverSupported(),         NeverSupported(), NeverSupported(), ExtsOnly("GL_EXT_pvrtc_sRGB"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT,          NeverSupported(),         NeverSupported(), NeverSupported(), ExtsOnly("GL_EXT_pvrtc_sRGB"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT,    NeverSupported(),         NeverSupported(), NeverSupported(), ExtsOnly("GL_EXT_pvrtc_sRGB"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );
    InsertFormatMapping(&map, GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT,    NeverSupported(),         NeverSupported(), NeverSupported(), ExtsOnly("GL_EXT_pvrtc_sRGB"),   AlwaysSupported(), NeverSupported(),                      NeverSupported()                );

    // clang-format on

    return map;
}

static const InternalFormatInfoMap &GetInternalFormatMap()
{
    static const angle::base::NoDestructor<InternalFormatInfoMap> formatMap(
        BuildInternalFormatInfoMap());
    return *formatMap;
}

const InternalFormat &GetInternalFormatInfo(GLenum internalFormat, StandardGL standard)
{
    const InternalFormatInfoMap &formatMap     = GetInternalFormatMap();
    InternalFormatInfoMap::const_iterator iter = formatMap.find(internalFormat);
    if (iter != formatMap.end())
    {
        const InternalFormatInfo &info = iter->second;
        switch (standard)
        {
            case STANDARD_GL_ES:
                return info.glesInfo;
            case STANDARD_GL_DESKTOP:
                return info.glInfo;
            default:
                UNREACHABLE();
                break;
        }
    }

    static const angle::base::NoDestructor<InternalFormat> defaultInternalFormat;
    return *defaultInternalFormat;
}

static GLenum GetNativeInternalFormat(const FunctionsGL *functions,
                                      const angle::FeaturesGL &features,
                                      const gl::InternalFormat &internalFormat)
{
    GLenum result = internalFormat.internalFormat;

    if (functions->standard == STANDARD_GL_DESKTOP)
    {
        // Use sized internal formats whenever possible to guarantee the requested precision.
        // On Desktop GL, passing an internal format of GL_RGBA will generate a GL_RGBA8 texture
        // even if the provided type is GL_FLOAT.
        result = internalFormat.sizedInternalFormat;

        if (features.avoid1BitAlphaTextureFormats.enabled && internalFormat.alphaBits == 1)
        {
            // Use an 8-bit format instead
            result = GL_RGBA8;
        }

        if (features.rgba4IsNotSupportedForColorRendering.enabled &&
            internalFormat.sizedInternalFormat == GL_RGBA4)
        {
            // Use an 8-bit format instead
            result = GL_RGBA8;
        }

        if (internalFormat.sizedInternalFormat == GL_RGB565 &&
            !functions->isAtLeastGL(gl::Version(4, 1)) &&
            !functions->hasGLExtension("GL_ARB_ES2_compatibility"))
        {
            // GL_RGB565 is required for basic ES2 functionality but was not added to desktop GL
            // until 4.1.
            // Work around this by using an 8-bit format instead.
            result = GL_RGB8;
        }

        if (internalFormat.sizedInternalFormat == GL_BGRA8_EXT)
        {
            // GLES accepts GL_BGRA as an internal format but desktop GL only accepts it as a type.
            // Update the internal format to GL_RGBA.
            result = GL_RGBA8;
        }

        if ((functions->profile & GL_CONTEXT_CORE_PROFILE_BIT) != 0)
        {
            // Work around deprecated luminance alpha formats in the OpenGL core profile by backing
            // them with R or RG textures.
            if (internalFormat.format == GL_LUMINANCE || internalFormat.format == GL_ALPHA)
            {
                result = gl::GetInternalFormatInfo(GL_RED, internalFormat.type).sizedInternalFormat;
            }

            if (internalFormat.format == GL_LUMINANCE_ALPHA)
            {
                result = gl::GetInternalFormatInfo(GL_RG, internalFormat.type).sizedInternalFormat;
            }
        }
    }
    else if (functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        if (internalFormat.componentType == GL_FLOAT && !internalFormat.isLUMA())
        {
            // Use sized internal formats for floating point textures.  Extensions such as
            // EXT_color_buffer_float require the sized formats to be renderable.
            result = internalFormat.sizedInternalFormat;
        }
        else if (internalFormat.format == GL_RED_EXT || internalFormat.format == GL_RG_EXT)
        {
            // Workaround Adreno driver not supporting unsized EXT_texture_rg formats
            result = internalFormat.sizedInternalFormat;
        }
        else if (features.unsizedsRGBReadPixelsDoesntTransform.enabled &&
                 internalFormat.colorEncoding == GL_SRGB)
        {
            // Work around some Adreno driver bugs that don't read back SRGB data correctly when
            // it's in unsized SRGB texture formats.
            result = internalFormat.sizedInternalFormat;
        }
    }

    return result;
}

static GLenum GetNativeFormat(const FunctionsGL *functions,
                              const angle::FeaturesGL &features,
                              GLenum format)
{
    GLenum result = format;

    if (functions->standard == STANDARD_GL_DESKTOP)
    {
        // The ES SRGB extensions require that the provided format is GL_SRGB or SRGB_ALPHA but
        // the desktop GL extensions only accept GL_RGB or GL_RGBA.  Convert them.
        if (format == GL_SRGB)
        {
            result = GL_RGB;
        }

        if (format == GL_SRGB_ALPHA)
        {
            result = GL_RGBA;
        }

        if ((functions->profile & GL_CONTEXT_CORE_PROFILE_BIT) != 0)
        {
            // Work around deprecated luminance alpha formats in the OpenGL core profile by backing
            // them with R or RG textures.
            if (format == GL_LUMINANCE || format == GL_ALPHA)
            {
                result = GL_RED;
            }

            if (format == GL_LUMINANCE_ALPHA)
            {
                result = GL_RG;
            }
        }
    }
    else if (functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        if (features.unsizedsRGBReadPixelsDoesntTransform.enabled)
        {
            if (format == GL_SRGB)
            {
                result = GL_RGB;
            }

            if (format == GL_SRGB_ALPHA)
            {
                result = GL_RGBA;
            }
        }
    }

    return result;
}

static GLenum GetNativeCompressedFormat(const FunctionsGL *functions,
                                        const angle::FeaturesGL &features,
                                        GLenum format)
{
    GLenum result = format;

    if (functions->standard == STANDARD_GL_DESKTOP)
    {
        if (format == GL_ETC1_RGB8_OES)
        {
            // GL_ETC1_RGB8_OES is not available in any desktop GL extension but the compression
            // format is forwards compatible so just use the ETC2 format.
            result = GL_COMPRESSED_RGB8_ETC2;
        }
    }

    if (functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        if (format == GL_ETC1_RGB8_OES)
        {
            // Pass GL_COMPRESSED_RGB8_ETC2 as the target format in ES3 and higher because it
            // becomes a core format.
            result = GL_COMPRESSED_RGB8_ETC2;
        }
    }

    return result;
}

static GLenum GetNativeType(const FunctionsGL *functions,
                            const angle::FeaturesGL &features,
                            GLenum format,
                            GLenum type)
{
    GLenum result = type;

    if (functions->standard == STANDARD_GL_DESKTOP)
    {
        if (type == GL_HALF_FLOAT_OES)
        {
            // The enums differ for the OES half float extensions and desktop GL spec.
            // Update it.
            result = GL_HALF_FLOAT;
        }
    }
    else if (functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        if (type == GL_HALF_FLOAT_OES)
        {
            switch (format)
            {
                case GL_LUMINANCE_ALPHA:
                case GL_LUMINANCE:
                case GL_ALPHA:
                    // In ES3, these formats come from EXT_texture_storage, which uses
                    // HALF_FLOAT_OES. Other formats (like RGBA) use HALF_FLOAT (non-OES) in ES3.
                    break;

                default:
                    result = GL_HALF_FLOAT;
                    break;
            }
        }
    }
    else if (functions->standard == STANDARD_GL_ES && functions->version == gl::Version(2, 0))
    {
        // On ES2, convert GL_HALF_FLOAT to GL_HALF_FLOAT_OES as a convenience for internal
        // functions. It should not be possible to get here by a normal glTexImage2D call.
        if (type == GL_HALF_FLOAT)
        {
            ASSERT(functions->hasGLESExtension("GL_OES_texture_half_float"));
            result = GL_HALF_FLOAT_OES;
        }
    }

    return result;
}

static GLenum GetNativeReadType(const FunctionsGL *functions,
                                const angle::FeaturesGL &features,
                                GLenum type)
{
    GLenum result = type;

    if (functions->standard == STANDARD_GL_DESKTOP || functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        if (type == GL_HALF_FLOAT_OES)
        {
            // The enums differ for the OES half float extensions and desktop GL spec. Update it.
            result = GL_HALF_FLOAT;
        }
    }

    return result;
}

static GLenum GetNativeReadFormat(const FunctionsGL *functions,
                                  const angle::FeaturesGL &features,
                                  GLenum format)
{
    GLenum result = format;
    return result;
}

TexImageFormat GetTexImageFormat(const FunctionsGL *functions,
                                 const angle::FeaturesGL &features,
                                 GLenum internalFormat,
                                 GLenum format,
                                 GLenum type)
{
    TexImageFormat result;
    result.internalFormat = GetNativeInternalFormat(
        functions, features, gl::GetInternalFormatInfo(internalFormat, type));
    result.format = GetNativeFormat(functions, features, format);
    result.type   = GetNativeType(functions, features, format, type);
    return result;
}

TexSubImageFormat GetTexSubImageFormat(const FunctionsGL *functions,
                                       const angle::FeaturesGL &features,
                                       GLenum format,
                                       GLenum type)
{
    TexSubImageFormat result;
    result.format = GetNativeFormat(functions, features, format);
    result.type   = GetNativeType(functions, features, format, type);
    return result;
}

CompressedTexImageFormat GetCompressedTexImageFormat(const FunctionsGL *functions,
                                                     const angle::FeaturesGL &features,
                                                     GLenum internalFormat)
{
    CompressedTexImageFormat result;
    result.internalFormat = GetNativeCompressedFormat(functions, features, internalFormat);
    return result;
}

CompressedTexSubImageFormat GetCompressedSubTexImageFormat(const FunctionsGL *functions,
                                                           const angle::FeaturesGL &features,
                                                           GLenum format)
{
    CompressedTexSubImageFormat result;
    result.format = GetNativeCompressedFormat(functions, features, format);
    return result;
}

CopyTexImageImageFormat GetCopyTexImageImageFormat(const FunctionsGL *functions,
                                                   const angle::FeaturesGL &features,
                                                   GLenum internalFormat,
                                                   GLenum framebufferType)
{
    CopyTexImageImageFormat result;
    result.internalFormat = GetNativeInternalFormat(
        functions, features, gl::GetInternalFormatInfo(internalFormat, framebufferType));
    return result;
}

TexStorageFormat GetTexStorageFormat(const FunctionsGL *functions,
                                     const angle::FeaturesGL &features,
                                     GLenum internalFormat)
{
    TexStorageFormat result;
    result.internalFormat = GetNativeInternalFormat(functions, features,
                                                    gl::GetSizedInternalFormatInfo(internalFormat));
    return result;
}

RenderbufferFormat GetRenderbufferFormat(const FunctionsGL *functions,
                                         const angle::FeaturesGL &features,
                                         GLenum internalFormat)
{
    RenderbufferFormat result;
    result.internalFormat = GetNativeInternalFormat(functions, features,
                                                    gl::GetSizedInternalFormatInfo(internalFormat));
    return result;
}
ReadPixelsFormat GetReadPixelsFormat(const FunctionsGL *functions,
                                     const angle::FeaturesGL &features,
                                     GLenum format,
                                     GLenum type)
{
    ReadPixelsFormat result;
    result.format = GetNativeReadFormat(functions, features, format);
    result.type   = GetNativeReadType(functions, features, type);
    return result;
}
}  // namespace nativegl

}  // namespace rx
