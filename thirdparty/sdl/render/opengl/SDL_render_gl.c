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

#ifdef SDL_VIDEO_RENDER_OGL
#include "../../video/SDL_sysvideo.h" // For SDL_RecreateWindow
#include <SDL3/SDL_opengl.h>
#include "../SDL_sysrender.h"
#include "SDL_shaders_gl.h"
#include "../../video/SDL_pixels_c.h"

#ifdef SDL_PLATFORM_MACOS
#include <OpenGL/OpenGL.h>
#endif

#ifdef SDL_VIDEO_VITA_PVR_OGL
#include <GL/gl.h>
#include <GL/glext.h>
#endif

/* To prevent unnecessary window recreation,
 * these should match the defaults selected in SDL_GL_ResetAttributes
 */

#define RENDERER_CONTEXT_MAJOR 2
#define RENDERER_CONTEXT_MINOR 1

// OpenGL renderer implementation

/* Details on optimizing the texture path on macOS:
   http://developer.apple.com/library/mac/#documentation/GraphicsImaging/Conceptual/OpenGL-MacProgGuide/opengl_texturedata/opengl_texturedata.html
*/

typedef struct GL_FBOList GL_FBOList;

struct GL_FBOList
{
    Uint32 w, h;
    GLuint FBO;
    GL_FBOList *next;
};

typedef struct
{
    bool viewport_dirty;
    SDL_Rect viewport;
    SDL_Texture *texture;
    SDL_Texture *target;
    int drawablew;
    int drawableh;
    SDL_BlendMode blend;
    GL_Shader shader;
    float texel_size[4];
    const float *shader_params;
    bool cliprect_enabled_dirty;
    bool cliprect_enabled;
    bool cliprect_dirty;
    SDL_Rect cliprect;
    bool texturing;
    bool texturing_dirty;
    bool vertex_array;
    bool color_array;
    bool texture_array;
    bool color_dirty;
    SDL_FColor color;
    bool clear_color_dirty;
    SDL_FColor clear_color;
} GL_DrawStateCache;

typedef struct
{
    SDL_GLContext context;

    bool debug_enabled;
    bool GL_ARB_debug_output_supported;
    int errors;
    char **error_messages;
    GLDEBUGPROCARB next_error_callback;
    GLvoid *next_error_userparam;

    GLenum textype;

    bool GL_ARB_texture_non_power_of_two_supported;
    bool GL_ARB_texture_rectangle_supported;
    bool GL_EXT_framebuffer_object_supported;
    GL_FBOList *framebuffers;

    // OpenGL functions
#define SDL_PROC(ret, func, params) ret (APIENTRY *func) params;
#include "SDL_glfuncs.h"
#undef SDL_PROC

    // Multitexture support
    bool GL_ARB_multitexture_supported;
    PFNGLACTIVETEXTUREARBPROC glActiveTextureARB;
    GLint num_texture_units;

    PFNGLGENFRAMEBUFFERSEXTPROC glGenFramebuffersEXT;
    PFNGLDELETEFRAMEBUFFERSEXTPROC glDeleteFramebuffersEXT;
    PFNGLFRAMEBUFFERTEXTURE2DEXTPROC glFramebufferTexture2DEXT;
    PFNGLBINDFRAMEBUFFEREXTPROC glBindFramebufferEXT;
    PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC glCheckFramebufferStatusEXT;

    // Shader support
    GL_ShaderContext *shaders;

    GL_DrawStateCache drawstate;
} GL_RenderData;

typedef struct
{
    GLuint texture;
    bool texture_external;
    GLfloat texw;
    GLfloat texh;
    GLenum format;
    GLenum formattype;
    GL_Shader shader;
    float texel_size[4];
    const float *shader_params;
    void *pixels;
    int pitch;
    SDL_Rect locked_rect;
#ifdef SDL_HAVE_YUV
    // YUV texture support
    bool yuv;
    bool nv12;
    GLuint utexture;
    bool utexture_external;
    GLuint vtexture;
    bool vtexture_external;
#endif
    SDL_ScaleMode texture_scale_mode;
    SDL_TextureAddressMode texture_address_mode_u;
    SDL_TextureAddressMode texture_address_mode_v;
    GL_FBOList *fbo;
} GL_TextureData;

static const char *GL_TranslateError(GLenum error)
{
#define GL_ERROR_TRANSLATE(e) \
    case e:                   \
        return #e;
    switch (error) {
        GL_ERROR_TRANSLATE(GL_INVALID_ENUM)
        GL_ERROR_TRANSLATE(GL_INVALID_VALUE)
        GL_ERROR_TRANSLATE(GL_INVALID_OPERATION)
        GL_ERROR_TRANSLATE(GL_OUT_OF_MEMORY)
        GL_ERROR_TRANSLATE(GL_NO_ERROR)
        GL_ERROR_TRANSLATE(GL_STACK_OVERFLOW)
        GL_ERROR_TRANSLATE(GL_STACK_UNDERFLOW)
        GL_ERROR_TRANSLATE(GL_TABLE_TOO_LARGE)
    default:
        return "UNKNOWN";
    }
#undef GL_ERROR_TRANSLATE
}

static void GL_ClearErrors(SDL_Renderer *renderer)
{
    GL_RenderData *data = (GL_RenderData *)renderer->internal;

    if (!data->debug_enabled) {
        return;
    }
    if (data->GL_ARB_debug_output_supported) {
        if (data->errors) {
            int i;
            for (i = 0; i < data->errors; ++i) {
                SDL_free(data->error_messages[i]);
            }
            SDL_free(data->error_messages);

            data->errors = 0;
            data->error_messages = NULL;
        }
    } else if (data->glGetError) {
        while (data->glGetError() != GL_NO_ERROR) {
            // continue;
        }
    }
}

static bool GL_CheckAllErrors(const char *prefix, SDL_Renderer *renderer, const char *file, int line, const char *function)
{
    GL_RenderData *data = (GL_RenderData *)renderer->internal;
    bool result = true;

    if (!data->debug_enabled) {
        return true;
    }
    if (data->GL_ARB_debug_output_supported) {
        if (data->errors) {
            int i;
            for (i = 0; i < data->errors; ++i) {
                SDL_SetError("%s: %s (%d): %s %s", prefix, file, line, function, data->error_messages[i]);
                result = false;
            }
            GL_ClearErrors(renderer);
        }
    } else {
        // check gl errors (can return multiple errors)
        for (;;) {
            GLenum error = data->glGetError();
            if (error != GL_NO_ERROR) {
                if (prefix == NULL || prefix[0] == '\0') {
                    prefix = "generic";
                }
                SDL_SetError("%s: %s (%d): %s %s (0x%X)", prefix, file, line, function, GL_TranslateError(error), error);
                result = false;
            } else {
                break;
            }
        }
    }
    return result;
}

#if 0
#define GL_CheckError(prefix, renderer)
#else
#define GL_CheckError(prefix, renderer) GL_CheckAllErrors(prefix, renderer, SDL_FILE, SDL_LINE, SDL_FUNCTION)
#endif

static bool GL_LoadFunctions(GL_RenderData *data)
{
#ifdef __SDL_NOGETPROCADDR__
#define SDL_PROC(ret, func, params) data->func = func;
#else
    bool result = true;
#define SDL_PROC(ret, func, params)                                                           \
    do {                                                                                      \
        data->func = (ret (APIENTRY *) params)SDL_GL_GetProcAddress(#func);                                            \
        if (!data->func) {                                                                    \
            result = SDL_SetError("Couldn't load GL function %s: %s", #func, SDL_GetError()); \
        }                                                                                     \
    } while (0);
#endif // __SDL_NOGETPROCADDR__

#include "SDL_glfuncs.h"
#undef SDL_PROC
    return result;
}

static bool GL_ActivateRenderer(SDL_Renderer *renderer)
{
    GL_RenderData *data = (GL_RenderData *)renderer->internal;

    if (SDL_GL_GetCurrentContext() != data->context) {
        if (!SDL_GL_MakeCurrent(renderer->window, data->context)) {
            return false;
        }
    }

    GL_ClearErrors(renderer);

    return true;
}

static void APIENTRY GL_HandleDebugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char *message, const void *userParam)
{
    SDL_Renderer *renderer = (SDL_Renderer *)userParam;
    GL_RenderData *data = (GL_RenderData *)renderer->internal;

    if (type == GL_DEBUG_TYPE_ERROR_ARB) {
        // Record this error
        int errors = data->errors + 1;
        char **error_messages = (char **)SDL_realloc(data->error_messages, errors * sizeof(*data->error_messages));
        if (error_messages) {
            data->errors = errors;
            data->error_messages = error_messages;
            data->error_messages[data->errors - 1] = SDL_strdup(message);
        }
    }

    // If there's another error callback, pass it along, otherwise log it
    if (data->next_error_callback) {
        data->next_error_callback(source, type, id, severity, length, message, data->next_error_userparam);
    } else {
        if (type == GL_DEBUG_TYPE_ERROR_ARB) {
            SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s", message);
        } else {
            SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "%s", message);
        }
    }
}

static GL_FBOList *GL_GetFBO(GL_RenderData *data, Uint32 w, Uint32 h)
{
    GL_FBOList *result = data->framebuffers;

    while (result && ((result->w != w) || (result->h != h))) {
        result = result->next;
    }

    if (!result) {
        result = (GL_FBOList *)SDL_malloc(sizeof(GL_FBOList));
        if (result) {
            result->w = w;
            result->h = h;
            data->glGenFramebuffersEXT(1, &result->FBO);
            result->next = data->framebuffers;
            data->framebuffers = result;
        }
    }
    return result;
}

static void GL_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    /* If the window x/y/w/h changed at all, assume the viewport has been
     * changed behind our backs. x/y changes might seem weird but viewport
     * resets have been observed on macOS at minimum!
     */
    if (event->type == SDL_EVENT_WINDOW_RESIZED ||
        event->type == SDL_EVENT_WINDOW_MOVED) {
        GL_RenderData *data = (GL_RenderData *)renderer->internal;
        data->drawstate.viewport_dirty = true;
    }
}

static GLenum GetBlendFunc(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return GL_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return GL_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return GL_SRC_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return GL_ONE_MINUS_SRC_COLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return GL_SRC_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return GL_ONE_MINUS_SRC_ALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return GL_DST_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return GL_ONE_MINUS_DST_COLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return GL_DST_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return GL_ONE_MINUS_DST_ALPHA;
    default:
        return GL_INVALID_ENUM;
    }
}

static GLenum GetBlendEquation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return GL_FUNC_ADD;
    case SDL_BLENDOPERATION_SUBTRACT:
        return GL_FUNC_SUBTRACT;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return GL_FUNC_REVERSE_SUBTRACT;
    case SDL_BLENDOPERATION_MINIMUM:
        return GL_MIN;
    case SDL_BLENDOPERATION_MAXIMUM:
        return GL_MAX;
    default:
        return GL_INVALID_ENUM;
    }
}

static bool GL_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    if (GetBlendFunc(srcColorFactor) == GL_INVALID_ENUM ||
        GetBlendFunc(srcAlphaFactor) == GL_INVALID_ENUM ||
        GetBlendEquation(colorOperation) == GL_INVALID_ENUM ||
        GetBlendFunc(dstColorFactor) == GL_INVALID_ENUM ||
        GetBlendFunc(dstAlphaFactor) == GL_INVALID_ENUM ||
        GetBlendEquation(alphaOperation) == GL_INVALID_ENUM) {
        return false;
    }
    if (colorOperation != alphaOperation) {
        return false;
    }
    return true;
}

static bool convert_format(Uint32 pixel_format, GLint *internalFormat, GLenum *format, GLenum *type)
{
    switch (pixel_format) {
    case SDL_PIXELFORMAT_ARGB8888:
    case SDL_PIXELFORMAT_XRGB8888:
        *internalFormat = GL_RGBA8;
        *format = GL_BGRA;
        *type = GL_UNSIGNED_BYTE; // previously GL_UNSIGNED_INT_8_8_8_8_REV, seeing if this is better in modern times.
        break;
    case SDL_PIXELFORMAT_ABGR8888:
    case SDL_PIXELFORMAT_XBGR8888:
        *internalFormat = GL_RGBA8;
        *format = GL_RGBA;
        *type = GL_UNSIGNED_BYTE; // previously GL_UNSIGNED_INT_8_8_8_8_REV, seeing if this is better in modern times.
        break;
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
        *internalFormat = GL_LUMINANCE;
        *format = GL_LUMINANCE;
        *type = GL_UNSIGNED_BYTE;
        break;
#ifdef SDL_PLATFORM_MACOS
    case SDL_PIXELFORMAT_UYVY:
        *internalFormat = GL_RGB8;
        *format = GL_YCBCR_422_APPLE;
        *type = GL_UNSIGNED_SHORT_8_8_APPLE;
        break;
#endif
    default:
        return false;
    }
    return true;
}

static bool GL_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    GL_RenderData *renderdata = (GL_RenderData *)renderer->internal;
    const GLenum textype = renderdata->textype;
    GL_TextureData *data;
    GLint internalFormat;
    GLenum format, type;
    int texture_w, texture_h;

    GL_ActivateRenderer(renderer);

    renderdata->drawstate.texture = NULL; // we trash this state.
    renderdata->drawstate.texturing_dirty = true; // we trash this state.

    if (texture->access == SDL_TEXTUREACCESS_TARGET &&
        !renderdata->GL_EXT_framebuffer_object_supported) {
        return SDL_SetError("Render targets not supported by OpenGL");
    }

    if (!convert_format(texture->format, &internalFormat, &format, &type)) {
        return SDL_SetError("Texture format %s not supported by OpenGL",
                            SDL_GetPixelFormatName(texture->format));
    }

    data = (GL_TextureData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        size_t size;
        data->pitch = texture->w * SDL_BYTESPERPIXEL(texture->format);
        size = (size_t)texture->h * data->pitch;
        if (texture->format == SDL_PIXELFORMAT_YV12 ||
            texture->format == SDL_PIXELFORMAT_IYUV) {
            // Need to add size for the U and V planes
            size += 2 * ((texture->h + 1) / 2) * ((data->pitch + 1) / 2);
        }
        if (texture->format == SDL_PIXELFORMAT_NV12 ||
            texture->format == SDL_PIXELFORMAT_NV21) {
            // Need to add size for the U/V plane
            size += 2 * ((texture->h + 1) / 2) * ((data->pitch + 1) / 2);
        }
        data->pixels = SDL_calloc(1, size);
        if (!data->pixels) {
            SDL_free(data);
            return false;
        }
    }

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        data->fbo = GL_GetFBO(renderdata, texture->w, texture->h);
    } else {
        data->fbo = NULL;
    }

    data->texture = (GLuint)SDL_GetNumberProperty(create_props, SDL_PROP_TEXTURE_CREATE_OPENGL_TEXTURE_NUMBER, 0);
    if (data->texture) {
        data->texture_external = true;
    } else {
        GL_CheckError("", renderer);
        renderdata->glGenTextures(1, &data->texture);
        if (!GL_CheckError("glGenTextures()", renderer)) {
            if (data->pixels) {
                SDL_free(data->pixels);
            }
            SDL_free(data);
            return false;
        }
    }
    texture->internal = data;

    if (renderdata->GL_ARB_texture_non_power_of_two_supported) {
        texture_w = texture->w;
        texture_h = texture->h;
        data->texw = 1.0f;
        data->texh = 1.0f;
    } else if (renderdata->GL_ARB_texture_rectangle_supported) {
        texture_w = texture->w;
        texture_h = texture->h;
        data->texw = (GLfloat)texture_w;
        data->texh = (GLfloat)texture_h;
    } else {
        texture_w = SDL_powerof2(texture->w);
        texture_h = SDL_powerof2(texture->h);
        data->texw = (GLfloat)(texture->w) / texture_w;
        data->texh = (GLfloat)texture->h / texture_h;
    }
    SDL_PropertiesID props = SDL_GetTextureProperties(texture);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_NUMBER, data->texture);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_TARGET_NUMBER, (Sint64) textype);
    SDL_SetFloatProperty(props, SDL_PROP_TEXTURE_OPENGL_TEX_W_FLOAT, data->texw);
    SDL_SetFloatProperty(props, SDL_PROP_TEXTURE_OPENGL_TEX_H_FLOAT, data->texh);

    data->format = format;
    data->formattype = type;
    data->texture_scale_mode = SDL_SCALEMODE_INVALID;
    data->texture_address_mode_u = SDL_TEXTURE_ADDRESS_INVALID;
    data->texture_address_mode_v = SDL_TEXTURE_ADDRESS_INVALID;
    renderdata->glEnable(textype);
    renderdata->glBindTexture(textype, data->texture);
#ifdef SDL_PLATFORM_MACOS
#ifndef GL_TEXTURE_STORAGE_HINT_APPLE
#define GL_TEXTURE_STORAGE_HINT_APPLE 0x85BC
#endif
#ifndef STORAGE_CACHED_APPLE
#define STORAGE_CACHED_APPLE 0x85BE
#endif
#ifndef STORAGE_SHARED_APPLE
#define STORAGE_SHARED_APPLE 0x85BF
#endif
    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        renderdata->glTexParameteri(textype, GL_TEXTURE_STORAGE_HINT_APPLE,
                                    GL_STORAGE_SHARED_APPLE);
    } else {
        renderdata->glTexParameteri(textype, GL_TEXTURE_STORAGE_HINT_APPLE,
                                    GL_STORAGE_CACHED_APPLE);
    }
    if (texture->access == SDL_TEXTUREACCESS_STREAMING && texture->format == SDL_PIXELFORMAT_ARGB8888 && (texture->w % 8) == 0) {
        renderdata->glPixelStorei(GL_UNPACK_CLIENT_STORAGE_APPLE, GL_TRUE);
        renderdata->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH,
                                  (data->pitch / SDL_BYTESPERPIXEL(texture->format)));
        renderdata->glTexImage2D(textype, 0, internalFormat, texture_w,
                                 texture_h, 0, format, type, data->pixels);
        renderdata->glPixelStorei(GL_UNPACK_CLIENT_STORAGE_APPLE, GL_FALSE);
    } else
#endif
    {
        renderdata->glTexImage2D(textype, 0, internalFormat, texture_w,
                                 texture_h, 0, format, type, NULL);
    }
    renderdata->glDisable(textype);
    if (!GL_CheckError("glTexImage2D()", renderer)) {
        return false;
    }

#ifdef SDL_HAVE_YUV
    if (texture->format == SDL_PIXELFORMAT_YV12 ||
        texture->format == SDL_PIXELFORMAT_IYUV) {
        data->yuv = true;

        data->utexture = (GLuint)SDL_GetNumberProperty(create_props, SDL_PROP_TEXTURE_CREATE_OPENGL_TEXTURE_U_NUMBER, 0);
        if (data->utexture) {
            data->utexture_external = true;
        } else {
            renderdata->glGenTextures(1, &data->utexture);
        }
        data->vtexture = (GLuint)SDL_GetNumberProperty(create_props, SDL_PROP_TEXTURE_CREATE_OPENGL_TEXTURE_V_NUMBER, 0);
        if (data->vtexture) {
            data->vtexture_external = true;
        } else {
            renderdata->glGenTextures(1, &data->vtexture);
        }

        renderdata->glBindTexture(textype, data->utexture);
        renderdata->glTexImage2D(textype, 0, internalFormat, (texture_w + 1) / 2,
                                 (texture_h + 1) / 2, 0, format, type, NULL);
        SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_U_NUMBER, data->utexture);

        renderdata->glBindTexture(textype, data->vtexture);
        renderdata->glTexImage2D(textype, 0, internalFormat, (texture_w + 1) / 2,
                                 (texture_h + 1) / 2, 0, format, type, NULL);
        SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_V_NUMBER, data->vtexture);
    }

    if (texture->format == SDL_PIXELFORMAT_NV12 ||
        texture->format == SDL_PIXELFORMAT_NV21) {
        data->nv12 = true;

        data->utexture = (GLuint)SDL_GetNumberProperty(create_props, SDL_PROP_TEXTURE_CREATE_OPENGL_TEXTURE_UV_NUMBER, 0);
        if (data->utexture) {
            data->utexture_external = true;
        } else {
            renderdata->glGenTextures(1, &data->utexture);
        }
        renderdata->glBindTexture(textype, data->utexture);
        renderdata->glTexImage2D(textype, 0, GL_LUMINANCE_ALPHA, (texture_w + 1) / 2,
                                 (texture_h + 1) / 2, 0, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, NULL);
        SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_UV_NUMBER, data->utexture);
    }
#endif

    if (texture->format == SDL_PIXELFORMAT_ABGR8888 || texture->format == SDL_PIXELFORMAT_ARGB8888) {
        data->shader = SHADER_RGBA;
    } else {
        data->shader = SHADER_RGB;
    }

    data->texel_size[2] = texture->w;
    data->texel_size[3] = texture->h;
    data->texel_size[0] = 1.0f / data->texel_size[2];
    data->texel_size[1] = 1.0f / data->texel_size[3];

#ifdef SDL_HAVE_YUV
    if (data->yuv || data->nv12) {
        if (data->yuv) {
            data->shader = SHADER_YUV;
        } else if (texture->format == SDL_PIXELFORMAT_NV12) {
            if (SDL_GetHintBoolean("SDL_RENDER_OPENGL_NV12_RG_SHADER", false)) {
                data->shader = SHADER_NV12_RG;
            } else {
                data->shader = SHADER_NV12_RA;
            }
        } else {
            if (SDL_GetHintBoolean("SDL_RENDER_OPENGL_NV12_RG_SHADER", false)) {
                data->shader = SHADER_NV21_RG;
            } else {
                data->shader = SHADER_NV21_RA;
            }
        }
        data->shader_params = SDL_GetYCbCRtoRGBConversionMatrix(texture->colorspace, texture->w, texture->h, 8);
        if (!data->shader_params) {
            return SDL_SetError("Unsupported YUV colorspace");
        }
    }
#endif // SDL_HAVE_YUV

    return GL_CheckError("", renderer);
}

static bool GL_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                            const SDL_Rect *rect, const void *pixels, int pitch)
{
    GL_RenderData *renderdata = (GL_RenderData *)renderer->internal;
    const GLenum textype = renderdata->textype;
    GL_TextureData *data = (GL_TextureData *)texture->internal;
    const int texturebpp = SDL_BYTESPERPIXEL(texture->format);

    SDL_assert_release(texturebpp != 0); // otherwise, division by zero later.

    GL_ActivateRenderer(renderer);

    renderdata->drawstate.texture = NULL; // we trash this state.

    renderdata->glBindTexture(textype, data->texture);
    renderdata->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, (pitch / texturebpp));
    renderdata->glTexSubImage2D(textype, 0, rect->x, rect->y, rect->w,
                                rect->h, data->format, data->formattype,
                                pixels);
#ifdef SDL_HAVE_YUV
    if (data->yuv) {
        renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, ((pitch + 1) / 2));

        // Skip to the correct offset into the next texture
        pixels = (const void *)((const Uint8 *)pixels + rect->h * pitch);
        if (texture->format == SDL_PIXELFORMAT_YV12) {
            renderdata->glBindTexture(textype, data->vtexture);
        } else {
            renderdata->glBindTexture(textype, data->utexture);
        }
        renderdata->glTexSubImage2D(textype, 0, rect->x / 2, rect->y / 2,
                                    (rect->w + 1) / 2, (rect->h + 1) / 2,
                                    data->format, data->formattype, pixels);

        // Skip to the correct offset into the next texture
        pixels = (const void *)((const Uint8 *)pixels + ((rect->h + 1) / 2) * ((pitch + 1) / 2));
        if (texture->format == SDL_PIXELFORMAT_YV12) {
            renderdata->glBindTexture(textype, data->utexture);
        } else {
            renderdata->glBindTexture(textype, data->vtexture);
        }
        renderdata->glTexSubImage2D(textype, 0, rect->x / 2, rect->y / 2,
                                    (rect->w + 1) / 2, (rect->h + 1) / 2,
                                    data->format, data->formattype, pixels);
    }

    if (data->nv12) {
        renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, ((pitch + 1) / 2));

        // Skip to the correct offset into the next texture
        pixels = (const void *)((const Uint8 *)pixels + rect->h * pitch);
        renderdata->glBindTexture(textype, data->utexture);
        renderdata->glTexSubImage2D(textype, 0, rect->x / 2, rect->y / 2,
                                    (rect->w + 1) / 2, (rect->h + 1) / 2,
                                    GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, pixels);
    }
#endif
    return GL_CheckError("glTexSubImage2D()", renderer);
}

#ifdef SDL_HAVE_YUV
static bool GL_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                               const SDL_Rect *rect,
                               const Uint8 *Yplane, int Ypitch,
                               const Uint8 *Uplane, int Upitch,
                               const Uint8 *Vplane, int Vpitch)
{
    GL_RenderData *renderdata = (GL_RenderData *)renderer->internal;
    const GLenum textype = renderdata->textype;
    GL_TextureData *data = (GL_TextureData *)texture->internal;

    GL_ActivateRenderer(renderer);

    renderdata->drawstate.texture = NULL; // we trash this state.

    renderdata->glBindTexture(textype, data->texture);
    renderdata->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, Ypitch);
    renderdata->glTexSubImage2D(textype, 0, rect->x, rect->y, rect->w,
                                rect->h, data->format, data->formattype,
                                Yplane);

    renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, Upitch);
    renderdata->glBindTexture(textype, data->utexture);
    renderdata->glTexSubImage2D(textype, 0, rect->x / 2, rect->y / 2,
                                (rect->w + 1) / 2, (rect->h + 1) / 2,
                                data->format, data->formattype, Uplane);

    renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, Vpitch);
    renderdata->glBindTexture(textype, data->vtexture);
    renderdata->glTexSubImage2D(textype, 0, rect->x / 2, rect->y / 2,
                                (rect->w + 1) / 2, (rect->h + 1) / 2,
                                data->format, data->formattype, Vplane);

    return GL_CheckError("glTexSubImage2D()", renderer);
}

static bool GL_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                              const SDL_Rect *rect,
                              const Uint8 *Yplane, int Ypitch,
                              const Uint8 *UVplane, int UVpitch)
{
    GL_RenderData *renderdata = (GL_RenderData *)renderer->internal;
    const GLenum textype = renderdata->textype;
    GL_TextureData *data = (GL_TextureData *)texture->internal;

    GL_ActivateRenderer(renderer);

    renderdata->drawstate.texture = NULL; // we trash this state.

    renderdata->glBindTexture(textype, data->texture);
    renderdata->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, Ypitch);
    renderdata->glTexSubImage2D(textype, 0, rect->x, rect->y, rect->w,
                                rect->h, data->format, data->formattype,
                                Yplane);

    renderdata->glPixelStorei(GL_UNPACK_ROW_LENGTH, UVpitch / 2);
    renderdata->glBindTexture(textype, data->utexture);
    renderdata->glTexSubImage2D(textype, 0, rect->x / 2, rect->y / 2,
                                (rect->w + 1) / 2, (rect->h + 1) / 2,
                                GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, UVplane);

    return GL_CheckError("glTexSubImage2D()", renderer);
}
#endif

static bool GL_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                          const SDL_Rect *rect, void **pixels, int *pitch)
{
    GL_TextureData *data = (GL_TextureData *)texture->internal;

    data->locked_rect = *rect;
    *pixels =
        (void *)((Uint8 *)data->pixels + rect->y * data->pitch +
                 rect->x * SDL_BYTESPERPIXEL(texture->format));
    *pitch = data->pitch;
    return true;
}

static void GL_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GL_TextureData *data = (GL_TextureData *)texture->internal;
    const SDL_Rect *rect;
    void *pixels;

    rect = &data->locked_rect;
    pixels =
        (void *)((Uint8 *)data->pixels + rect->y * data->pitch +
                 rect->x * SDL_BYTESPERPIXEL(texture->format));
    GL_UpdateTexture(renderer, texture, rect, pixels, data->pitch);
}

static bool GL_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GL_RenderData *data = (GL_RenderData *)renderer->internal;
    GL_TextureData *texturedata;
    GLenum status;

    GL_ActivateRenderer(renderer);

    if (!data->GL_EXT_framebuffer_object_supported) {
        return SDL_SetError("Render targets not supported by OpenGL");
    }

    data->drawstate.viewport_dirty = true;

    if (!texture) {
        data->glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        return true;
    }

    texturedata = (GL_TextureData *)texture->internal;
    data->glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, texturedata->fbo->FBO);
    // TODO: check if texture pixel format allows this operation
    data->glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, data->textype, texturedata->texture, 0);
    // Check FBO status
    status = data->glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
        return SDL_SetError("glFramebufferTexture2DEXT() failed");
    }
    return true;
}

/* !!! FIXME: all these Queue* calls set up the vertex buffer the way the immediate mode
   !!! FIXME:  renderer wants it, but this might want to operate differently if we move to
   !!! FIXME:  VBOs at some point. */
static bool GL_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static bool GL_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    GLfloat *verts = (GLfloat *)SDL_AllocateRenderVertices(renderer, count * 2 * sizeof(GLfloat), 0, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    for (i = 0; i < count; i++) {
        *(verts++) = 0.5f + points[i].x;
        *(verts++) = 0.5f + points[i].y;
    }

    return true;
}

static bool GL_QueueDrawLines(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    int i;
    GLfloat prevx, prevy;
    const size_t vertlen = (sizeof(GLfloat) * 2) * count;
    GLfloat *verts = (GLfloat *)SDL_AllocateRenderVertices(renderer, vertlen, 0, &cmd->data.draw.first);

    if (!verts) {
        return false;
    }
    cmd->data.draw.count = count;

    // 0.5f offset to hit the center of the pixel.
    prevx = 0.5f + points->x;
    prevy = 0.5f + points->y;
    *(verts++) = prevx;
    *(verts++) = prevy;

    /* bump the end of each line segment out a quarter of a pixel, to provoke
       the diamond-exit rule. Without this, you won't just drop the last
       pixel of the last line segment, but you might also drop pixels at the
       edge of any given line segment along the way too. */
    for (i = 1; i < count; i++) {
        const GLfloat xstart = prevx;
        const GLfloat ystart = prevy;
        const GLfloat xend = points[i].x + 0.5f; // 0.5f to hit pixel center.
        const GLfloat yend = points[i].y + 0.5f;
        // bump a little in the direction we are moving in.
        const GLfloat deltax = xend - xstart;
        const GLfloat deltay = yend - ystart;
        const GLfloat angle = SDL_atan2f(deltay, deltax);
        prevx = xend + (SDL_cosf(angle) * 0.25f);
        prevy = yend + (SDL_sinf(angle) * 0.25f);
        *(verts++) = prevx;
        *(verts++) = prevy;
    }

    return true;
}

static bool GL_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                            const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                            int num_vertices, const void *indices, int num_indices, int size_indices,
                            float scale_x, float scale_y)
{
    GL_TextureData *texturedata = NULL;
    int i;
    int count = indices ? num_indices : num_vertices;
    GLfloat *verts;
    size_t sz = 2 * sizeof(GLfloat) + 4 * sizeof(GLfloat) + (texture ? 2 : 0) * sizeof(GLfloat);
    const float color_scale = cmd->data.draw.color_scale;

    verts = (GLfloat *)SDL_AllocateRenderVertices(renderer, count * sz, 0, &cmd->data.draw.first);
    if (!verts) {
        return false;
    }

    if (texture) {
        texturedata = (GL_TextureData *)texture->internal;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    for (i = 0; i < count; i++) {
        int j;
        float *xy_;
        SDL_FColor *col_;
        if (size_indices == 4) {
            j = ((const Uint32 *)indices)[i];
        } else if (size_indices == 2) {
            j = ((const Uint16 *)indices)[i];
        } else if (size_indices == 1) {
            j = ((const Uint8 *)indices)[i];
        } else {
            j = i;
        }

        xy_ = (float *)((char *)xy + j * xy_stride);

        *(verts++) = xy_[0] * scale_x;
        *(verts++) = xy_[1] * scale_y;

        col_ = (SDL_FColor *)((char *)color + j * color_stride);
        *(verts++) = col_->r * color_scale;
        *(verts++) = col_->g * color_scale;
        *(verts++) = col_->b * color_scale;
        *(verts++) = col_->a;

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            *(verts++) = uv_[0] * texturedata->texw;
            *(verts++) = uv_[1] * texturedata->texh;
        }
    }
    return true;
}

static bool SetDrawState(GL_RenderData *data, const SDL_RenderCommand *cmd, const GL_Shader shader, const float *shader_params)
{
    const SDL_BlendMode blend = cmd->data.draw.blend;
    bool vertex_array;
    bool color_array;
    bool texture_array;

    if (data->drawstate.viewport_dirty) {
        const bool istarget = data->drawstate.target != NULL;
        const SDL_Rect *viewport = &data->drawstate.viewport;
        data->glMatrixMode(GL_PROJECTION);
        data->glLoadIdentity();
        data->glViewport(viewport->x,
                         istarget ? viewport->y : (data->drawstate.drawableh - viewport->y - viewport->h),
                         viewport->w, viewport->h);
        if (viewport->w && viewport->h) {
            data->glOrtho((GLdouble)0, (GLdouble)viewport->w,
                          (GLdouble)(istarget ? 0 : viewport->h),
                          (GLdouble)(istarget ? viewport->h : 0),
                          0.0, 1.0);
        }
        data->glMatrixMode(GL_MODELVIEW);
        data->drawstate.viewport_dirty = false;
    }

    if (data->drawstate.cliprect_enabled_dirty) {
        if (!data->drawstate.cliprect_enabled) {
            data->glDisable(GL_SCISSOR_TEST);
        } else {
            data->glEnable(GL_SCISSOR_TEST);
        }
        data->drawstate.cliprect_enabled_dirty = false;
    }

    if (data->drawstate.cliprect_enabled && data->drawstate.cliprect_dirty) {
        const SDL_Rect *viewport = &data->drawstate.viewport;
        const SDL_Rect *rect = &data->drawstate.cliprect;
        data->glScissor(viewport->x + rect->x,
                        data->drawstate.target ? viewport->y + rect->y : data->drawstate.drawableh - viewport->y - rect->y - rect->h,
                        rect->w, rect->h);
        data->drawstate.cliprect_dirty = false;
    }

    if (blend != data->drawstate.blend) {
        if (blend == SDL_BLENDMODE_NONE) {
            data->glDisable(GL_BLEND);
        } else {
            data->glEnable(GL_BLEND);
            data->glBlendFuncSeparate(GetBlendFunc(SDL_GetBlendModeSrcColorFactor(blend)),
                                      GetBlendFunc(SDL_GetBlendModeDstColorFactor(blend)),
                                      GetBlendFunc(SDL_GetBlendModeSrcAlphaFactor(blend)),
                                      GetBlendFunc(SDL_GetBlendModeDstAlphaFactor(blend)));
            data->glBlendEquation(GetBlendEquation(SDL_GetBlendModeColorOperation(blend)));
        }
        data->drawstate.blend = blend;
    }

    if (data->shaders &&
        (shader != data->drawstate.shader || shader_params != data->drawstate.shader_params)) {
        GL_SelectShader(data->shaders, shader, shader_params);
        data->drawstate.shader = shader;
        data->drawstate.shader_params = shader_params;
    }

    if (data->drawstate.texturing_dirty || ((cmd->data.draw.texture != NULL) != data->drawstate.texturing)) {
        if (!cmd->data.draw.texture) {
            data->glDisable(data->textype);
            data->drawstate.texturing = false;
        } else {
            data->glEnable(data->textype);
            data->drawstate.texturing = true;
        }
        data->drawstate.texturing_dirty = false;
    }

    vertex_array = cmd->command == SDL_RENDERCMD_DRAW_POINTS || cmd->command == SDL_RENDERCMD_DRAW_LINES || cmd->command == SDL_RENDERCMD_GEOMETRY;
    color_array = cmd->command == SDL_RENDERCMD_GEOMETRY;
    texture_array = cmd->data.draw.texture != NULL;

    if (vertex_array != data->drawstate.vertex_array) {
        if (vertex_array) {
            data->glEnableClientState(GL_VERTEX_ARRAY);
        } else {
            data->glDisableClientState(GL_VERTEX_ARRAY);
        }
        data->drawstate.vertex_array = vertex_array;
    }

    if (color_array != data->drawstate.color_array) {
        if (color_array) {
            data->glEnableClientState(GL_COLOR_ARRAY);
        } else {
            data->glDisableClientState(GL_COLOR_ARRAY);
        }
        data->drawstate.color_array = color_array;
    }

    /* This is a little awkward but should avoid texcoord arrays getting into
       a bad state if the application is manually binding textures */
    if (texture_array != data->drawstate.texture_array) {
        if (texture_array) {
            data->glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        } else {
            data->glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        }
        data->drawstate.texture_array = texture_array;
    }

    return true;
}

static bool SetTextureScaleMode(GL_RenderData *data, GLenum textype, SDL_ScaleMode scaleMode)
{
    switch (scaleMode) {
    case SDL_SCALEMODE_NEAREST:
        data->glTexParameteri(textype, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        data->glTexParameteri(textype, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        break;
    case SDL_SCALEMODE_PIXELART:    // Uses linear sampling
    case SDL_SCALEMODE_LINEAR:
        data->glTexParameteri(textype, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        data->glTexParameteri(textype, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;
    default:
        return SDL_SetError("Unknown texture scale mode: %d", scaleMode);
    }
    return true;
}

static GLint TranslateAddressMode(SDL_TextureAddressMode addressMode)
{
    switch (addressMode) {
    case SDL_TEXTURE_ADDRESS_CLAMP:
        return GL_CLAMP_TO_EDGE;
    case SDL_TEXTURE_ADDRESS_WRAP:
        return GL_REPEAT;
    default:
        SDL_assert(!"Unknown texture address mode");
        return GL_CLAMP_TO_EDGE;
    }
}

static void SetTextureAddressMode(GL_RenderData *data, GLenum textype, SDL_TextureAddressMode addressModeU, SDL_TextureAddressMode addressModeV)
{
    data->glTexParameteri(textype, GL_TEXTURE_WRAP_S, TranslateAddressMode(addressModeU));
    data->glTexParameteri(textype, GL_TEXTURE_WRAP_T, TranslateAddressMode(addressModeV));
}

static bool SetCopyState(GL_RenderData *data, const SDL_RenderCommand *cmd)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    GL_TextureData *texturedata = (GL_TextureData *)texture->internal;
    const GLenum textype = data->textype;
    GL_Shader shader = texturedata->shader;
    const float *shader_params = texturedata->shader_params;

    if (cmd->data.draw.texture_scale_mode == SDL_SCALEMODE_PIXELART) {
        switch (shader) {
        case SHADER_RGB:
            shader = SHADER_RGB_PIXELART;
            shader_params = texturedata->texel_size;
            break;
        case SHADER_RGBA:
            shader = SHADER_RGBA_PIXELART;
            shader_params = texturedata->texel_size;
            break;
        default:
            break;
        }
    }
    SetDrawState(data, cmd, shader, shader_params);

    if (texture != data->drawstate.texture) {
#ifdef SDL_HAVE_YUV
        if (texturedata->yuv) {
            data->glActiveTextureARB(GL_TEXTURE2_ARB);
            data->glBindTexture(textype, texturedata->vtexture);

            data->glActiveTextureARB(GL_TEXTURE1_ARB);
            data->glBindTexture(textype, texturedata->utexture);
        }
        if (texturedata->nv12) {
            data->glActiveTextureARB(GL_TEXTURE1_ARB);
            data->glBindTexture(textype, texturedata->utexture);
        }
#endif
        if (data->GL_ARB_multitexture_supported) {
            data->glActiveTextureARB(GL_TEXTURE0_ARB);
        }
        data->glBindTexture(textype, texturedata->texture);

        data->drawstate.texture = texture;
    }

    if (cmd->data.draw.texture_scale_mode != texturedata->texture_scale_mode) {
#ifdef SDL_HAVE_YUV
        if (texturedata->yuv) {
            data->glActiveTextureARB(GL_TEXTURE2);
            if (!SetTextureScaleMode(data, textype, cmd->data.draw.texture_scale_mode)) {
                return false;
            }

            data->glActiveTextureARB(GL_TEXTURE1);
            if (!SetTextureScaleMode(data, textype, cmd->data.draw.texture_scale_mode)) {
                return false;
            }

            data->glActiveTextureARB(GL_TEXTURE0);
        } else if (texturedata->nv12) {
            data->glActiveTextureARB(GL_TEXTURE1);
            if (!SetTextureScaleMode(data, textype, cmd->data.draw.texture_scale_mode)) {
                return false;
            }

            data->glActiveTextureARB(GL_TEXTURE0);
        }
#endif
        if (!SetTextureScaleMode(data, textype, cmd->data.draw.texture_scale_mode)) {
            return false;
        }

        texturedata->texture_scale_mode = cmd->data.draw.texture_scale_mode;
    }

    if (cmd->data.draw.texture_address_mode_u != texturedata->texture_address_mode_u ||
        cmd->data.draw.texture_address_mode_v != texturedata->texture_address_mode_v) {
#ifdef SDL_HAVE_YUV
        if (texturedata->yuv) {
            data->glActiveTextureARB(GL_TEXTURE2);
            SetTextureAddressMode(data, textype, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);

            data->glActiveTextureARB(GL_TEXTURE1);
            SetTextureAddressMode(data, textype, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);

            data->glActiveTextureARB(GL_TEXTURE0_ARB);
        } else if (texturedata->nv12) {
            data->glActiveTextureARB(GL_TEXTURE1);
            SetTextureAddressMode(data, textype, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);

            data->glActiveTextureARB(GL_TEXTURE0);
        }
#endif
        SetTextureAddressMode(data, textype, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);

        texturedata->texture_address_mode_u = cmd->data.draw.texture_address_mode_u;
        texturedata->texture_address_mode_v = cmd->data.draw.texture_address_mode_v;
    }

    return true;
}

static void GL_InvalidateCachedState(SDL_Renderer *renderer)
{
    GL_DrawStateCache *cache = &((GL_RenderData *)renderer->internal)->drawstate;
    cache->viewport_dirty = true;
    cache->texture = NULL;
    cache->drawablew = 0;
    cache->drawableh = 0;
    cache->blend = SDL_BLENDMODE_INVALID;
    cache->shader = SHADER_INVALID;
    cache->cliprect_enabled_dirty = true;
    cache->cliprect_dirty = true;
    cache->texturing_dirty = true;
    cache->vertex_array = false;  // !!! FIXME: this resets to false at the end of GL_RunCommandQueue, but we could cache this more aggressively.
    cache->color_array = false;  // !!! FIXME: this resets to false at the end of GL_RunCommandQueue, but we could cache this more aggressively.
    cache->texture_array = false;  // !!! FIXME: this resets to false at the end of GL_RunCommandQueue, but we could cache this more aggressively.
    cache->color_dirty = true;
    cache->clear_color_dirty = true;
}

static bool GL_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    // !!! FIXME: it'd be nice to use a vertex buffer instead of immediate mode...
    GL_RenderData *data = (GL_RenderData *)renderer->internal;

    if (!GL_ActivateRenderer(renderer)) {
        return false;
    }

    data->drawstate.target = renderer->target;
    if (!data->drawstate.target) {
        int w, h;
        SDL_GetWindowSizeInPixels(renderer->window, &w, &h);
        if ((w != data->drawstate.drawablew) || (h != data->drawstate.drawableh)) {
            data->drawstate.viewport_dirty = true; // if the window dimensions changed, invalidate the current viewport, etc.
            data->drawstate.cliprect_dirty = true;
            data->drawstate.drawablew = w;
            data->drawstate.drawableh = h;
        }
    }

#ifdef SDL_PLATFORM_MACOS
    // On macOS on older systems, the OpenGL view change and resize events aren't
    // necessarily synchronized, so just always reset it.
    // Workaround for: https://discourse.libsdl.org/t/sdl-2-0-22-prerelease/35306/6
    data->drawstate.viewport_dirty = true;
#endif

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            const float r = cmd->data.color.color.r * cmd->data.color.color_scale;
            const float g = cmd->data.color.color.g * cmd->data.color.color_scale;
            const float b = cmd->data.color.color.b * cmd->data.color.color_scale;
            const float a = cmd->data.color.color.a;
            if (data->drawstate.color_dirty ||
                (r != data->drawstate.color.r) ||
                (g != data->drawstate.color.g) ||
                (b != data->drawstate.color.b) ||
                (a != data->drawstate.color.a)) {
                data->glColor4f(r, g, b, a);
                data->drawstate.color.r = r;
                data->drawstate.color.g = g;
                data->drawstate.color.b = b;
                data->drawstate.color.a = a;
                data->drawstate.color_dirty = false;
            }
            break;
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &data->drawstate.viewport;
            if (SDL_memcmp(viewport, &cmd->data.viewport.rect, sizeof(cmd->data.viewport.rect)) != 0) {
                SDL_copyp(viewport, &cmd->data.viewport.rect);
                data->drawstate.viewport_dirty = true;
                data->drawstate.cliprect_dirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            if (data->drawstate.cliprect_enabled != cmd->data.cliprect.enabled) {
                data->drawstate.cliprect_enabled = cmd->data.cliprect.enabled;
                data->drawstate.cliprect_enabled_dirty = true;
            }

            if (SDL_memcmp(&data->drawstate.cliprect, rect, sizeof(*rect)) != 0) {
                SDL_copyp(&data->drawstate.cliprect, rect);
                data->drawstate.cliprect_dirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            const float r = cmd->data.color.color.r * cmd->data.color.color_scale;
            const float g = cmd->data.color.color.g * cmd->data.color.color_scale;
            const float b = cmd->data.color.color.b * cmd->data.color.color_scale;
            const float a = cmd->data.color.color.a;
            if (data->drawstate.clear_color_dirty ||
                (r != data->drawstate.clear_color.r) ||
                (g != data->drawstate.clear_color.g) ||
                (b != data->drawstate.clear_color.b) ||
                (a != data->drawstate.clear_color.a)) {
                data->glClearColor(r, g, b, a);
                data->drawstate.clear_color.r = r;
                data->drawstate.clear_color.g = g;
                data->drawstate.clear_color.b = b;
                data->drawstate.clear_color.a = a;
                data->drawstate.clear_color_dirty = false;
            }

            if (data->drawstate.cliprect_enabled || data->drawstate.cliprect_enabled_dirty) {
                data->glDisable(GL_SCISSOR_TEST);
                data->drawstate.cliprect_enabled_dirty = data->drawstate.cliprect_enabled;
            }

            data->glClear(GL_COLOR_BUFFER_BIT);
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS: // unused
            break;

        case SDL_RENDERCMD_COPY: // unused
            break;

        case SDL_RENDERCMD_COPY_EX: // unused
            break;

        case SDL_RENDERCMD_DRAW_LINES:
        {
            if (SetDrawState(data, cmd, SHADER_SOLID, NULL)) {
                size_t count = cmd->data.draw.count;
                const GLfloat *verts = (GLfloat *)(((Uint8 *)vertices) + cmd->data.draw.first);

                // SetDrawState handles glEnableClientState.
                data->glVertexPointer(2, GL_FLOAT, sizeof(float) * 2, verts);

                if (count > 2) {
                    // joined lines cannot be grouped
                    data->glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)count);
                } else {
                    // let's group non joined lines
                    SDL_RenderCommand *finalcmd = cmd;
                    SDL_RenderCommand *nextcmd = cmd->next;
                    SDL_BlendMode thisblend = cmd->data.draw.blend;

                    while (nextcmd) {
                        const SDL_RenderCommandType nextcmdtype = nextcmd->command;
                        if (nextcmdtype != SDL_RENDERCMD_DRAW_LINES) {
                            break; // can't go any further on this draw call, different render command up next.
                        } else if (nextcmd->data.draw.count != 2) {
                            break; // can't go any further on this draw call, those are joined lines
                        } else if (nextcmd->data.draw.blend != thisblend) {
                            break; // can't go any further on this draw call, different blendmode copy up next.
                        } else {
                            finalcmd = nextcmd; // we can combine copy operations here. Mark this one as the furthest okay command.
                            count += nextcmd->data.draw.count;
                        }
                        nextcmd = nextcmd->next;
                    }

                    data->glDrawArrays(GL_LINES, 0, (GLsizei)count);
                    cmd = finalcmd; // skip any copy commands we just combined in here.
                }
            }
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        case SDL_RENDERCMD_GEOMETRY:
        {
            /* as long as we have the same copy command in a row, with the
               same texture, we can combine them all into a single draw call. */
            SDL_Texture *thistexture = cmd->data.draw.texture;
            SDL_BlendMode thisblend = cmd->data.draw.blend;
            SDL_ScaleMode thisscalemode = cmd->data.draw.texture_scale_mode;
            SDL_TextureAddressMode thisaddressmode_u = cmd->data.draw.texture_address_mode_u;
            SDL_TextureAddressMode thisaddressmode_v = cmd->data.draw.texture_address_mode_v;
            const SDL_RenderCommandType thiscmdtype = cmd->command;
            SDL_RenderCommand *finalcmd = cmd;
            SDL_RenderCommand *nextcmd = cmd->next;
            size_t count = cmd->data.draw.count;
            int ret;
            while (nextcmd) {
                const SDL_RenderCommandType nextcmdtype = nextcmd->command;
                if (nextcmdtype != thiscmdtype) {
                    break; // can't go any further on this draw call, different render command up next.
                } else if (nextcmd->data.draw.texture != thistexture ||
                           nextcmd->data.draw.texture_scale_mode != thisscalemode ||
                           nextcmd->data.draw.texture_address_mode_u != thisaddressmode_u ||
                           nextcmd->data.draw.texture_address_mode_v != thisaddressmode_v ||
                           nextcmd->data.draw.blend != thisblend) {
                    break; // can't go any further on this draw call, different texture/blendmode copy up next.
                } else {
                    finalcmd = nextcmd; // we can combine copy operations here. Mark this one as the furthest okay command.
                    count += nextcmd->data.draw.count;
                }
                nextcmd = nextcmd->next;
            }

            if (thistexture) {
                ret = SetCopyState(data, cmd);
            } else {
                ret = SetDrawState(data, cmd, SHADER_SOLID, NULL);
            }

            if (ret) {
                const GLfloat *verts = (GLfloat *)(((Uint8 *)vertices) + cmd->data.draw.first);
                int op = GL_TRIANGLES; // SDL_RENDERCMD_GEOMETRY
                if (thiscmdtype == SDL_RENDERCMD_DRAW_POINTS) {
                    op = GL_POINTS;
                }

                if (thiscmdtype == SDL_RENDERCMD_DRAW_POINTS) {
                    // SetDrawState handles glEnableClientState.
                    data->glVertexPointer(2, GL_FLOAT, sizeof(float) * 2, verts);
                } else {
                    // SetDrawState handles glEnableClientState.
                    if (thistexture) {
                        data->glVertexPointer(2, GL_FLOAT, sizeof(float) * 8, verts + 0);
                        data->glColorPointer(4, GL_FLOAT, sizeof(float) * 8, verts + 2);
                        data->glTexCoordPointer(2, GL_FLOAT, sizeof(float) * 8, verts + 6);
                    } else {
                        data->glVertexPointer(2, GL_FLOAT, sizeof(float) * 6, verts + 0);
                        data->glColorPointer(4, GL_FLOAT, sizeof(float) * 6, verts + 2);
                    }
                }

                data->glDrawArrays(op, 0, (GLsizei)count);

                // Restore previously set color when we're done.
                if (thiscmdtype != SDL_RENDERCMD_DRAW_POINTS) {
                    const float r = data->drawstate.color.r;
                    const float g = data->drawstate.color.g;
                    const float b = data->drawstate.color.b;
                    const float a = data->drawstate.color.a;
                    data->glColor4f(r, g, b, a);
                }
            }

            cmd = finalcmd; // skip any copy commands we just combined in here.
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    /* Turn off vertex array state when we're done, in case external code
       relies on it being off. */
    if (data->drawstate.vertex_array) {
        data->glDisableClientState(GL_VERTEX_ARRAY);
        data->drawstate.vertex_array = false;
    }
    if (data->drawstate.color_array) {
        data->glDisableClientState(GL_COLOR_ARRAY);
        data->drawstate.color_array = false;
    }
    if (data->drawstate.texture_array) {
        data->glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        data->drawstate.texture_array = false;
    }

    return GL_CheckError("", renderer);
}

static SDL_Surface *GL_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    GL_RenderData *data = (GL_RenderData *)renderer->internal;
    SDL_PixelFormat format = renderer->target ? renderer->target->format : SDL_PIXELFORMAT_ARGB8888;
    GLint internalFormat;
    GLenum targetFormat, type;
    SDL_Surface *surface;

    GL_ActivateRenderer(renderer);

    if (!convert_format(format, &internalFormat, &targetFormat, &type)) {
        SDL_SetError("Texture format %s not supported by OpenGL", SDL_GetPixelFormatName(format));
        return NULL;
    }

    surface = SDL_CreateSurface(rect->w, rect->h, format);
    if (!surface) {
        return NULL;
    }

    int y = rect->y;
    if (!renderer->target) {
        int w, h;
        SDL_GetRenderOutputSize(renderer, &w, &h);
        y = (h - y) - rect->h;
    }

    data->glPixelStorei(GL_PACK_ALIGNMENT, 1);
    data->glPixelStorei(GL_PACK_ROW_LENGTH, (surface->pitch / SDL_BYTESPERPIXEL(format)));
    data->glReadPixels(rect->x, y, rect->w, rect->h, targetFormat, type, surface->pixels);

    if (!GL_CheckError("glReadPixels()", renderer)) {
        SDL_DestroySurface(surface);
        return NULL;
    }

    // Flip the rows to be top-down if necessary
    if (!renderer->target) {
        SDL_FlipSurface(surface, SDL_FLIP_VERTICAL);
    }
    return surface;
}

static bool GL_RenderPresent(SDL_Renderer *renderer)
{
    GL_ActivateRenderer(renderer);

    return SDL_GL_SwapWindow(renderer->window);
}

static void GL_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GL_RenderData *renderdata = (GL_RenderData *)renderer->internal;
    GL_TextureData *data = (GL_TextureData *)texture->internal;

    GL_ActivateRenderer(renderer);

    if (renderdata->drawstate.texture == texture) {
        renderdata->drawstate.texture = NULL;
    }
    if (renderdata->drawstate.target == texture) {
        renderdata->drawstate.target = NULL;
    }

    if (!data) {
        return;
    }
    if (data->texture && !data->texture_external) {
        renderdata->glDeleteTextures(1, &data->texture);
    }
#ifdef SDL_HAVE_YUV
    if (data->yuv) {
        if (!data->utexture_external) {
            renderdata->glDeleteTextures(1, &data->utexture);
        }
        if (!data->vtexture_external) {
            renderdata->glDeleteTextures(1, &data->vtexture);
        }
    }
    if (data->nv12) {
        if (!data->utexture_external) {
            renderdata->glDeleteTextures(1, &data->utexture);
        }
    }
#endif
    SDL_free(data->pixels);
    SDL_free(data);
    texture->internal = NULL;
}

static void GL_DestroyRenderer(SDL_Renderer *renderer)
{
    GL_RenderData *data = (GL_RenderData *)renderer->internal;

    if (data) {
        if (data->context) {
            // make sure we delete the right resources!
            GL_ActivateRenderer(renderer);
        }

        GL_ClearErrors(renderer);
        if (data->GL_ARB_debug_output_supported) {
            PFNGLDEBUGMESSAGECALLBACKARBPROC glDebugMessageCallbackARBFunc = (PFNGLDEBUGMESSAGECALLBACKARBPROC)SDL_GL_GetProcAddress("glDebugMessageCallbackARB");

            // Uh oh, we don't have a safe way of removing ourselves from the callback chain, if it changed after we set our callback.
            // For now, just always replace the callback with the original one
            glDebugMessageCallbackARBFunc(data->next_error_callback, data->next_error_userparam);
        }
        if (data->shaders) {
            GL_DestroyShaderContext(data->shaders);
        }
        if (data->context) {
            while (data->framebuffers) {
                GL_FBOList *nextnode = data->framebuffers->next;
                // delete the framebuffer object
                data->glDeleteFramebuffersEXT(1, &data->framebuffers->FBO);
                GL_CheckError("", renderer);
                SDL_free(data->framebuffers);
                data->framebuffers = nextnode;
            }
            SDL_GL_DestroyContext(data->context);
        }
        SDL_free(data);
    }
}

static bool GL_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    int interval = 0;

    if (!SDL_GL_SetSwapInterval(vsync)) {
        return false;
    }

    if (!SDL_GL_GetSwapInterval(&interval)) {
        return false;
    }

    if (interval != vsync) {
        return SDL_Unsupported();
    }
    return true;
}

static bool GL_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    GL_RenderData *data = NULL;
    GLint value;
    SDL_WindowFlags window_flags;
    int profile_mask = 0, major = 0, minor = 0;
    bool changed_window = false;
    const char *hint;
    bool non_power_of_two_supported = false;

    SDL_GL_GetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, &profile_mask);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);

#ifndef SDL_VIDEO_VITA_PVR_OGL
    SDL_SyncWindow(window);
    window_flags = SDL_GetWindowFlags(window);
    if (!(window_flags & SDL_WINDOW_OPENGL) ||
        profile_mask == SDL_GL_CONTEXT_PROFILE_ES || major != RENDERER_CONTEXT_MAJOR || minor != RENDERER_CONTEXT_MINOR) {

        changed_window = true;
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, RENDERER_CONTEXT_MAJOR);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, RENDERER_CONTEXT_MINOR);

        if (!SDL_RecreateWindow(window, (window_flags & ~(SDL_WINDOW_VULKAN | SDL_WINDOW_METAL)) | SDL_WINDOW_OPENGL)) {
            goto error;
        }
    }
#endif

    SDL_SetupRendererColorspace(renderer, create_props);

    if (renderer->output_colorspace != SDL_COLORSPACE_SRGB) {
        SDL_SetError("Unsupported output colorspace");
        goto error;
    }

    data = (GL_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        goto error;
    }

    renderer->WindowEvent = GL_WindowEvent;
    renderer->SupportsBlendMode = GL_SupportsBlendMode;
    renderer->CreateTexture = GL_CreateTexture;
    renderer->UpdateTexture = GL_UpdateTexture;
#ifdef SDL_HAVE_YUV
    renderer->UpdateTextureYUV = GL_UpdateTextureYUV;
    renderer->UpdateTextureNV = GL_UpdateTextureNV;
#endif
    renderer->LockTexture = GL_LockTexture;
    renderer->UnlockTexture = GL_UnlockTexture;
    renderer->SetRenderTarget = GL_SetRenderTarget;
    renderer->QueueSetViewport = GL_QueueNoOp;
    renderer->QueueSetDrawColor = GL_QueueNoOp;
    renderer->QueueDrawPoints = GL_QueueDrawPoints;
    renderer->QueueDrawLines = GL_QueueDrawLines;
    renderer->QueueGeometry = GL_QueueGeometry;
    renderer->InvalidateCachedState = GL_InvalidateCachedState;
    renderer->RunCommandQueue = GL_RunCommandQueue;
    renderer->RenderReadPixels = GL_RenderReadPixels;
    renderer->RenderPresent = GL_RenderPresent;
    renderer->DestroyTexture = GL_DestroyTexture;
    renderer->DestroyRenderer = GL_DestroyRenderer;
    renderer->SetVSync = GL_SetVSync;
    renderer->internal = data;
    GL_InvalidateCachedState(renderer);
    renderer->window = window;

    renderer->name = GL_RenderDriver.name;
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XRGB8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XBGR8888);

    data->context = SDL_GL_CreateContext(window);
    if (!data->context) {
        goto error;
    }
    if (!SDL_GL_MakeCurrent(window, data->context)) {
        goto error;
    }

    if (!GL_LoadFunctions(data)) {
        goto error;
    }

#ifdef SDL_PLATFORM_MACOS
    // Enable multi-threaded rendering
    /* Disabled until Ryan finishes his VBO/PBO code...
       CGLEnable(CGLGetCurrentContext(), kCGLCEMPEngine);
     */
#endif

    // Check for debug output support
    if (SDL_GL_GetAttribute(SDL_GL_CONTEXT_FLAGS, &value) &&
        (value & SDL_GL_CONTEXT_DEBUG_FLAG)) {
        data->debug_enabled = true;
    }
    if (data->debug_enabled && SDL_GL_ExtensionSupported("GL_ARB_debug_output")) {
        PFNGLDEBUGMESSAGECALLBACKARBPROC glDebugMessageCallbackARBFunc = (PFNGLDEBUGMESSAGECALLBACKARBPROC)SDL_GL_GetProcAddress("glDebugMessageCallbackARB");

        data->GL_ARB_debug_output_supported = true;
        data->glGetPointerv(GL_DEBUG_CALLBACK_FUNCTION_ARB, (GLvoid **)(char *)&data->next_error_callback);
        data->glGetPointerv(GL_DEBUG_CALLBACK_USER_PARAM_ARB, &data->next_error_userparam);
        glDebugMessageCallbackARBFunc(GL_HandleDebugMessage, renderer);

        // Make sure our callback is called when errors actually happen
        data->glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    }

    hint = SDL_GetHint("GL_ARB_texture_non_power_of_two");
    if (!hint || *hint != '0') {
        bool isGL2 = false;
        const char *verstr = (const char *)data->glGetString(GL_VERSION);
        if (verstr) {
            char verbuf[16];
            char *ptr;
            SDL_strlcpy(verbuf, verstr, sizeof(verbuf));
            ptr = SDL_strchr(verbuf, '.');
            if (ptr) {
                *ptr = '\0';
                if (SDL_atoi(verbuf) >= 2) {
                    isGL2 = true;
                }
            }
        }
        if (isGL2 || SDL_GL_ExtensionSupported("GL_ARB_texture_non_power_of_two")) {
            non_power_of_two_supported = true;
        }
    }

    data->textype = GL_TEXTURE_2D;
    if (non_power_of_two_supported) {
        data->GL_ARB_texture_non_power_of_two_supported = true;
        data->glGetIntegerv(GL_MAX_TEXTURE_SIZE, &value);
        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, value);
    } else if (SDL_GL_ExtensionSupported("GL_ARB_texture_rectangle") ||
               SDL_GL_ExtensionSupported("GL_EXT_texture_rectangle")) {
        data->GL_ARB_texture_rectangle_supported = true;
        data->textype = GL_TEXTURE_RECTANGLE_ARB;
        data->glGetIntegerv(GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB, &value);
        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, value);
    } else {
        data->glGetIntegerv(GL_MAX_TEXTURE_SIZE, &value);
        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, value);
    }

    // Check for multitexture support
    if (SDL_GL_ExtensionSupported("GL_ARB_multitexture")) {
        data->glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)SDL_GL_GetProcAddress("glActiveTextureARB");
        if (data->glActiveTextureARB) {
            data->GL_ARB_multitexture_supported = true;
            data->glGetIntegerv(GL_MAX_TEXTURE_UNITS_ARB, &data->num_texture_units);
        }
    }

    // Check for shader support
    data->shaders = GL_CreateShaderContext();
    SDL_LogInfo(SDL_LOG_CATEGORY_RENDER, "OpenGL shaders: %s",
                data->shaders ? "ENABLED" : "DISABLED");
#ifdef SDL_HAVE_YUV
    // We support YV12 textures using 3 textures and a shader
    if (data->shaders && data->num_texture_units >= 3) {
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_YV12);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_IYUV);
    }

    // We support NV12 textures using 2 textures and a shader
    if (data->shaders && data->num_texture_units >= 2) {
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV12);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV21);
    }
#endif
#ifdef SDL_PLATFORM_MACOS
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_UYVY);
#endif

    if (SDL_GL_ExtensionSupported("GL_EXT_framebuffer_object")) {
        data->GL_EXT_framebuffer_object_supported = true;
        data->glGenFramebuffersEXT = (PFNGLGENFRAMEBUFFERSEXTPROC)
            SDL_GL_GetProcAddress("glGenFramebuffersEXT");
        data->glDeleteFramebuffersEXT = (PFNGLDELETEFRAMEBUFFERSEXTPROC)
            SDL_GL_GetProcAddress("glDeleteFramebuffersEXT");
        data->glFramebufferTexture2DEXT = (PFNGLFRAMEBUFFERTEXTURE2DEXTPROC)
            SDL_GL_GetProcAddress("glFramebufferTexture2DEXT");
        data->glBindFramebufferEXT = (PFNGLBINDFRAMEBUFFEREXTPROC)
            SDL_GL_GetProcAddress("glBindFramebufferEXT");
        data->glCheckFramebufferStatusEXT = (PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC)
            SDL_GL_GetProcAddress("glCheckFramebufferStatusEXT");
    } else {
        SDL_SetError("Can't create render targets, GL_EXT_framebuffer_object not available");
        goto error;
    }

    // Set up parameters for rendering
    data->glMatrixMode(GL_MODELVIEW);
    data->glLoadIdentity();
    data->glDisable(GL_DEPTH_TEST);
    data->glDisable(GL_CULL_FACE);
    data->glDisable(GL_SCISSOR_TEST);
    data->glDisable(data->textype);
    data->glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    data->glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    // This ended up causing video discrepancies between OpenGL and Direct3D
    // data->glEnable(GL_LINE_SMOOTH);

    data->drawstate.color.r = 1.0f;
    data->drawstate.color.g = 1.0f;
    data->drawstate.color.b = 1.0f;
    data->drawstate.color.a = 1.0f;
    data->drawstate.clear_color.r = 1.0f;
    data->drawstate.clear_color.g = 1.0f;
    data->drawstate.clear_color.b = 1.0f;
    data->drawstate.clear_color.a = 1.0f;

    return true;

error:
    if (changed_window) {
        // Uh oh, better try to put it back...
        char *error = SDL_strdup(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, profile_mask);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, major);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor);
        SDL_RecreateWindow(window, window_flags);
        SDL_SetError("%s", error);
        SDL_free(error);
    }
    return false;
}

SDL_RenderDriver GL_RenderDriver = {
    GL_CreateRenderer, "opengl"
};

#endif // SDL_VIDEO_RENDER_OGL
