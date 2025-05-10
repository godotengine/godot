/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_VIDEO_RENDER_OGL_ES2

#include <SDL3/SDL_opengles2.h>
#include "SDL_shaders_gles2.h"

/* *INDENT-OFF* */ // clang-format off

/*************************************************************************************************
 * Vertex/fragment shader source                                                                 *
 *************************************************************************************************/

static const char GLES2_Fragment_Include_Best_Texture_Precision[] =
"#ifdef GL_FRAGMENT_PRECISION_HIGH\n"
"#define SDL_TEXCOORD_PRECISION highp\n"
"#else\n"
"#define SDL_TEXCOORD_PRECISION mediump\n"
"#endif\n"
"\n"
"precision mediump float;\n"
"\n"
;

static const char GLES2_Fragment_Include_Medium_Texture_Precision[] =
"#define SDL_TEXCOORD_PRECISION mediump\n"
"precision mediump float;\n"
"\n"
;

static const char GLES2_Fragment_Include_High_Texture_Precision[] =
"#define SDL_TEXCOORD_PRECISION highp\n"
"precision mediump float;\n"
"\n"
;

static const char GLES2_Fragment_Include_Undef_Precision[] =
"#define mediump\n"
"#define highp\n"
"#define lowp\n"
"#define SDL_TEXCOORD_PRECISION\n"
"\n"
;

static const char GLES2_Vertex_Default[] =
"uniform mat4 u_projection;\n"
"attribute vec2 a_position;\n"
"attribute vec4 a_color;\n"
"attribute vec2 a_texCoord;\n"
"varying vec2 v_texCoord;\n"
"varying vec4 v_color;\n"
"\n"
"void main()\n"
"{\n"
"    v_texCoord = a_texCoord;\n"
"    gl_Position = u_projection * vec4(a_position, 0.0, 1.0);\n"
"    gl_PointSize = 1.0;\n"
"    v_color = a_color;\n"
"}\n"
;

static const char GLES2_Fragment_Solid[] =
"varying mediump vec4 v_color;\n"
"\n"
"void main()\n"
"{\n"
"    gl_FragColor = v_color;\n"
"}\n"
;

#define RGB_SHADER_PROLOGUE                                     \
"uniform sampler2D u_texture;\n"                                \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \

#define RGB_PIXELART_SHADER_PROLOGUE                            \
"uniform sampler2D u_texture;\n"                                \
"uniform mediump vec4 u_texel_size;\n"                          \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \

#ifdef OPENGLES_300 // This is required for fwidth() and textureGrad()
#define RGB_PIXELART_GETCOLOR                                                                   \
"    mediump vec2 boxSize = clamp(fwidth(v_texCoord) * u_texel_size.zw, 1e-5, 1.0);\n"          \
"    mediump vec2 tx = v_texCoord * u_texel_size.zw - 0.5 * boxSize;\n"                         \
"    mediump vec2 txOffset = smoothstep(vec2(1.0) - boxSize, vec2(1.0), fract(tx));\n"          \
"    mediump vec2 uv = (floor(tx) + 0.5 + txOffset) * u_texel_size.xy;\n"                       \
"    mediump vec4 color = textureGrad(u_texture, uv, dFdx(v_texCoord), dFdy(v_texCoord));\n"    \
"    mediump vec4 color = texture2D(u_texture, uv);\n"
#else
#define RGB_PIXELART_GETCOLOR                                                                   \
"    mediump vec4 color = texture2D(u_texture, v_texCoord);\n"
#endif

static const char GLES2_Fragment_TextureABGR[] =
    RGB_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
"    mediump vec4 color = texture2D(u_texture, v_texCoord);\n"
"    gl_FragColor = color;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

// ARGB to ABGR conversion
static const char GLES2_Fragment_TextureARGB[] =
    RGB_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
"    mediump vec4 color = texture2D(u_texture, v_texCoord);\n"
"    gl_FragColor = color;\n"
"    gl_FragColor.r = color.b;\n"
"    gl_FragColor.b = color.r;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

// RGB to ABGR conversion
static const char GLES2_Fragment_TextureRGB[] =
    RGB_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
"    mediump vec4 color = texture2D(u_texture, v_texCoord);\n"
"    gl_FragColor = color;\n"
"    gl_FragColor.r = color.b;\n"
"    gl_FragColor.b = color.r;\n"
"    gl_FragColor.a = 1.0;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

// BGR to ABGR conversion
static const char GLES2_Fragment_TextureBGR[] =
    RGB_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
"    mediump vec4 color = texture2D(u_texture, v_texCoord);\n"
"    gl_FragColor = color;\n"
"    gl_FragColor.a = 1.0;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

static const char GLES2_Fragment_TextureABGR_PixelArt[] =
    RGB_PIXELART_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
    RGB_PIXELART_GETCOLOR
"    gl_FragColor = color;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

// ARGB to ABGR conversion
static const char GLES2_Fragment_TextureARGB_PixelArt[] =
    RGB_PIXELART_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
    RGB_PIXELART_GETCOLOR
"    gl_FragColor = color;\n"
"    gl_FragColor.r = color.b;\n"
"    gl_FragColor.b = color.r;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

// RGB to ABGR conversion
static const char GLES2_Fragment_TextureRGB_PixelArt[] =
    RGB_PIXELART_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
    RGB_PIXELART_GETCOLOR
"    gl_FragColor = color;\n"
"    gl_FragColor.r = color.b;\n"
"    gl_FragColor.b = color.r;\n"
"    gl_FragColor.a = 1.0;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

// BGR to ABGR conversion
static const char GLES2_Fragment_TextureBGR_PixelArt[] =
    RGB_PIXELART_SHADER_PROLOGUE
"\n"
"void main()\n"
"{\n"
    RGB_PIXELART_GETCOLOR
"    gl_FragColor = color;\n"
"    gl_FragColor.a = 1.0;\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

#ifdef SDL_HAVE_YUV

#define YUV_SHADER_PROLOGUE                                     \
"uniform sampler2D u_texture;\n"                                \
"uniform sampler2D u_texture_u;\n"                              \
"uniform sampler2D u_texture_v;\n"                              \
"uniform vec3 u_offset;\n"                                      \
"uniform mat3 u_matrix;\n"                                      \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \
"\n"                                                            \

#define YUV_SHADER_BODY                                         \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.y = texture2D(u_texture_u, v_texCoord).r;\n"           \
"    yuv.z = texture2D(u_texture_v, v_texCoord).r;\n"           \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += u_offset;\n"                                        \
"    rgb = yuv * u_matrix;\n"                                   \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

#define NV12_RA_SHADER_BODY                                     \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.yz = texture2D(u_texture_u, v_texCoord).ra;\n"         \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += u_offset;\n"                                        \
"    rgb = yuv * u_matrix;\n"                                   \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

#define NV12_RG_SHADER_BODY                                     \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.yz = texture2D(u_texture_u, v_texCoord).rg;\n"         \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += u_offset;\n"                                        \
"    rgb = yuv * u_matrix;\n"                                   \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

#define NV21_RA_SHADER_BODY                                     \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.yz = texture2D(u_texture_u, v_texCoord).ar;\n"         \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += u_offset;\n"                                        \
"    rgb = yuv * u_matrix;\n"                                   \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

#define NV21_RG_SHADER_BODY                                     \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.yz = texture2D(u_texture_u, v_texCoord).gr;\n"         \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += u_offset;\n"                                        \
"    rgb = yuv * u_matrix;\n"                                   \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

// YUV to ABGR conversion
static const char GLES2_Fragment_TextureYUV[] =
    YUV_SHADER_PROLOGUE
    YUV_SHADER_BODY
;

// NV12 to ABGR conversion
static const char GLES2_Fragment_TextureNV12_RA[] =
    YUV_SHADER_PROLOGUE
    NV12_RA_SHADER_BODY
;
static const char GLES2_Fragment_TextureNV12_RG[] =
    YUV_SHADER_PROLOGUE
    NV12_RG_SHADER_BODY
;

// NV21 to ABGR conversion
static const char GLES2_Fragment_TextureNV21_RA[] =
    YUV_SHADER_PROLOGUE
    NV21_RA_SHADER_BODY
;
static const char GLES2_Fragment_TextureNV21_RG[] =
    YUV_SHADER_PROLOGUE
    NV21_RG_SHADER_BODY
;
#endif

// Custom Android video format texture
static const char GLES2_Fragment_TextureExternalOES_Prologue[] =
"#extension GL_OES_EGL_image_external : require\n"
"\n"
;
static const char GLES2_Fragment_TextureExternalOES[] =
"uniform samplerExternalOES u_texture;\n"
"varying mediump vec4 v_color;\n"
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"
"\n"
"void main()\n"
"{\n"
"    gl_FragColor = texture2D(u_texture, v_texCoord);\n"
"    gl_FragColor *= v_color;\n"
"}\n"
;

/* *INDENT-ON* */ // clang-format on

/*************************************************************************************************
 * Shader selector                                                                               *
 *************************************************************************************************/

const char *GLES2_GetShaderPrologue(GLES2_ShaderType type)
{
    switch (type) {
    case GLES2_SHADER_FRAGMENT_TEXTURE_EXTERNAL_OES:
        return GLES2_Fragment_TextureExternalOES_Prologue;
    default:
        return "";
    }
}

const char *GLES2_GetShaderInclude(GLES2_ShaderIncludeType type)
{
    switch (type) {
    case GLES2_SHADER_FRAGMENT_INCLUDE_UNDEF_PRECISION:
        return GLES2_Fragment_Include_Undef_Precision;
    case GLES2_SHADER_FRAGMENT_INCLUDE_BEST_TEXCOORD_PRECISION:
        return GLES2_Fragment_Include_Best_Texture_Precision;
    case GLES2_SHADER_FRAGMENT_INCLUDE_MEDIUM_TEXCOORD_PRECISION:
        return GLES2_Fragment_Include_Medium_Texture_Precision;
    case GLES2_SHADER_FRAGMENT_INCLUDE_HIGH_TEXCOORD_PRECISION:
        return GLES2_Fragment_Include_High_Texture_Precision;
    default:
        return "";
    }
}

GLES2_ShaderIncludeType GLES2_GetTexCoordPrecisionEnumFromHint(void)
{
    const char *texcoord_hint = SDL_GetHint("SDL_RENDER_OPENGLES2_TEXCOORD_PRECISION");
    GLES2_ShaderIncludeType value = GLES2_SHADER_FRAGMENT_INCLUDE_BEST_TEXCOORD_PRECISION;
    if (texcoord_hint) {
        if (SDL_strcmp(texcoord_hint, "undefined") == 0) {
            return GLES2_SHADER_FRAGMENT_INCLUDE_UNDEF_PRECISION;
        }
        if (SDL_strcmp(texcoord_hint, "high") == 0) {
            return GLES2_SHADER_FRAGMENT_INCLUDE_HIGH_TEXCOORD_PRECISION;
        }
        if (SDL_strcmp(texcoord_hint, "medium") == 0) {
            return GLES2_SHADER_FRAGMENT_INCLUDE_MEDIUM_TEXCOORD_PRECISION;
        }
    }
    return value;
}

const char *GLES2_GetShader(GLES2_ShaderType type)
{
    switch (type) {
    case GLES2_SHADER_VERTEX_DEFAULT:
        return GLES2_Vertex_Default;
    case GLES2_SHADER_FRAGMENT_SOLID:
        return GLES2_Fragment_Solid;
    case GLES2_SHADER_FRAGMENT_TEXTURE_ABGR:
        return GLES2_Fragment_TextureABGR;
    case GLES2_SHADER_FRAGMENT_TEXTURE_ARGB:
        return GLES2_Fragment_TextureARGB;
    case GLES2_SHADER_FRAGMENT_TEXTURE_RGB:
        return GLES2_Fragment_TextureRGB;
    case GLES2_SHADER_FRAGMENT_TEXTURE_BGR:
        return GLES2_Fragment_TextureBGR;
    case GLES2_SHADER_FRAGMENT_TEXTURE_ABGR_PIXELART:
        return GLES2_Fragment_TextureABGR_PixelArt;
    case GLES2_SHADER_FRAGMENT_TEXTURE_ARGB_PIXELART:
        return GLES2_Fragment_TextureARGB_PixelArt;
    case GLES2_SHADER_FRAGMENT_TEXTURE_RGB_PIXELART:
        return GLES2_Fragment_TextureRGB_PixelArt;
    case GLES2_SHADER_FRAGMENT_TEXTURE_BGR_PIXELART:
        return GLES2_Fragment_TextureBGR_PixelArt;
#ifdef SDL_HAVE_YUV
    case GLES2_SHADER_FRAGMENT_TEXTURE_YUV:
        return GLES2_Fragment_TextureYUV;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RA:
        return GLES2_Fragment_TextureNV12_RA;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RG:
        return GLES2_Fragment_TextureNV12_RG;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV21_RA:
        return GLES2_Fragment_TextureNV21_RA;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV21_RG:
        return GLES2_Fragment_TextureNV21_RG;
#endif
    case GLES2_SHADER_FRAGMENT_TEXTURE_EXTERNAL_OES:
        return GLES2_Fragment_TextureExternalOES;
    default:
        return NULL;
    }
}

#endif // SDL_VIDEO_RENDER_OGL_ES2
