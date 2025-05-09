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

#include <SDL3/SDL_opengl.h>
#include "SDL_shaders_gl.h"

// OpenGL shader implementation

// #define DEBUG_SHADERS

typedef struct
{
    GLhandleARB program;
    GLhandleARB vert_shader;
    GLhandleARB frag_shader;
} GL_ShaderData;

struct GL_ShaderContext
{
    GLenum (*glGetError)(void);

    PFNGLATTACHOBJECTARBPROC glAttachObjectARB;
    PFNGLCOMPILESHADERARBPROC glCompileShaderARB;
    PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgramObjectARB;
    PFNGLCREATESHADEROBJECTARBPROC glCreateShaderObjectARB;
    PFNGLDELETEOBJECTARBPROC glDeleteObjectARB;
    PFNGLGETINFOLOGARBPROC glGetInfoLogARB;
    PFNGLGETOBJECTPARAMETERIVARBPROC glGetObjectParameterivARB;
    PFNGLGETUNIFORMLOCATIONARBPROC glGetUniformLocationARB;
    PFNGLLINKPROGRAMARBPROC glLinkProgramARB;
    PFNGLSHADERSOURCEARBPROC glShaderSourceARB;
    PFNGLUNIFORM1IARBPROC glUniform1iARB;
    PFNGLUNIFORM1FARBPROC glUniform1fARB;
    PFNGLUNIFORM3FARBPROC glUniform3fARB;
    PFNGLUNIFORM4FARBPROC glUniform4fARB;
    PFNGLUSEPROGRAMOBJECTARBPROC glUseProgramObjectARB;

    bool GL_ARB_texture_rectangle_supported;

    GL_ShaderData shaders[NUM_SHADERS];
    const float *shader_params[NUM_SHADERS];
};

/* *INDENT-OFF* */ // clang-format off

#define COLOR_VERTEX_SHADER                                     \
"varying vec4 v_color;\n"                                       \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n" \
"    v_color = gl_Color;\n"                                     \
"}"                                                             \

#define TEXTURE_VERTEX_SHADER                                   \
"varying vec4 v_color;\n"                                       \
"varying vec2 v_texCoord;\n"                                    \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n" \
"    v_color = gl_Color;\n"                                     \
"    v_texCoord = vec2(gl_MultiTexCoord0);\n"                   \
"}"                                                             \

#define YUV_SHADER_PROLOGUE                                     \
"varying vec4 v_color;\n"                                       \
"varying vec2 v_texCoord;\n"                                    \
"uniform sampler2D tex0; // Y \n"                               \
"uniform sampler2D tex1; // U \n"                               \
"uniform sampler2D tex2; // V \n"                               \
"uniform vec3 Yoffset;\n"                                       \
"uniform vec3 Rcoeff;\n"                                        \
"uniform vec3 Gcoeff;\n"                                        \
"uniform vec3 Bcoeff;\n"                                        \
"\n"                                                            \

#define YUV_SHADER_BODY                                         \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    vec2 tcoord;\n"                                            \
"    vec3 yuv, rgb;\n"                                          \
"\n"                                                            \
"    // Get the Y value \n"                                     \
"    tcoord = v_texCoord;\n"                                    \
"    yuv.x = texture2D(tex0, tcoord).r;\n"                      \
"\n"                                                            \
"    // Get the U and V values \n"                              \
"    tcoord *= UVCoordScale;\n"                                 \
"    yuv.y = texture2D(tex1, tcoord).r;\n"                      \
"    yuv.z = texture2D(tex2, tcoord).r;\n"                      \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += Yoffset;\n"                                         \
"    rgb.r = dot(yuv, Rcoeff);\n"                               \
"    rgb.g = dot(yuv, Gcoeff);\n"                               \
"    rgb.b = dot(yuv, Bcoeff);\n"                               \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1.0) * v_color;\n"                \
"}"                                                             \

#define NV12_SHADER_PROLOGUE                                    \
"varying vec4 v_color;\n"                                       \
"varying vec2 v_texCoord;\n"                                    \
"uniform sampler2D tex0; // Y \n"                               \
"uniform sampler2D tex1; // U/V \n"                             \
"uniform vec3 Yoffset;\n"                                       \
"uniform vec3 Rcoeff;\n"                                        \
"uniform vec3 Gcoeff;\n"                                        \
"uniform vec3 Bcoeff;\n"                                        \
"\n"                                                            \

#define NV12_RA_SHADER_BODY                                     \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    vec2 tcoord;\n"                                            \
"    vec3 yuv, rgb;\n"                                          \
"\n"                                                            \
"    // Get the Y value \n"                                     \
"    tcoord = v_texCoord;\n"                                    \
"    yuv.x = texture2D(tex0, tcoord).r;\n"                      \
"\n"                                                            \
"    // Get the U and V values \n"                              \
"    tcoord *= UVCoordScale;\n"                                 \
"    yuv.yz = texture2D(tex1, tcoord).ra;\n"                    \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += Yoffset;\n"                                         \
"    rgb.r = dot(yuv, Rcoeff);\n"                               \
"    rgb.g = dot(yuv, Gcoeff);\n"                               \
"    rgb.b = dot(yuv, Bcoeff);\n"                               \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1.0) * v_color;\n"                \
"}"                                                             \

#define NV12_RG_SHADER_BODY                                     \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    vec2 tcoord;\n"                                            \
"    vec3 yuv, rgb;\n"                                          \
"\n"                                                            \
"    // Get the Y value \n"                                     \
"    tcoord = v_texCoord;\n"                                    \
"    yuv.x = texture2D(tex0, tcoord).r;\n"                      \
"\n"                                                            \
"    // Get the U and V values \n"                              \
"    tcoord *= UVCoordScale;\n"                                 \
"    yuv.yz = texture2D(tex1, tcoord).rg;\n"                    \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += Yoffset;\n"                                         \
"    rgb.r = dot(yuv, Rcoeff);\n"                               \
"    rgb.g = dot(yuv, Gcoeff);\n"                               \
"    rgb.b = dot(yuv, Bcoeff);\n"                               \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1.0) * v_color;\n"                \
"}"                                                             \

#define NV21_RA_SHADER_BODY                                     \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    vec2 tcoord;\n"                                            \
"    vec3 yuv, rgb;\n"                                          \
"\n"                                                            \
"    // Get the Y value \n"                                     \
"    tcoord = v_texCoord;\n"                                    \
"    yuv.x = texture2D(tex0, tcoord).r;\n"                      \
"\n"                                                            \
"    // Get the U and V values \n"                              \
"    tcoord *= UVCoordScale;\n"                                 \
"    yuv.yz = texture2D(tex1, tcoord).ar;\n"                    \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += Yoffset;\n"                                         \
"    rgb.r = dot(yuv, Rcoeff);\n"                               \
"    rgb.g = dot(yuv, Gcoeff);\n"                               \
"    rgb.b = dot(yuv, Bcoeff);\n"                               \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1.0) * v_color;\n"                \
"}"                                                             \

#define NV21_RG_SHADER_BODY                                     \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    vec2 tcoord;\n"                                            \
"    vec3 yuv, rgb;\n"                                          \
"\n"                                                            \
"    // Get the Y value \n"                                     \
"    tcoord = v_texCoord;\n"                                    \
"    yuv.x = texture2D(tex0, tcoord).r;\n"                      \
"\n"                                                            \
"    // Get the U and V values \n"                              \
"    tcoord *= UVCoordScale;\n"                                 \
"    yuv.yz = texture2D(tex1, tcoord).gr;\n"                    \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += Yoffset;\n"                                         \
"    rgb.r = dot(yuv, Rcoeff);\n"                               \
"    rgb.g = dot(yuv, Gcoeff);\n"                               \
"    rgb.b = dot(yuv, Bcoeff);\n"                               \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1.0) * v_color;\n"                \
"}"                                                             \

/*
 * NOTE: Always use sampler2D, etc here. We'll #define them to the
 *  texture_rectangle versions if we choose to use that extension.
 */
static struct {
    const char *vertex_shader;
    const char *fragment_shader;
    const char *fragment_version;
} shader_source[NUM_SHADERS] = {
    // SHADER_NONE
    { NULL, NULL, NULL },

    // SHADER_SOLID
    {
        // vertex shader
        COLOR_VERTEX_SHADER,
        // fragment shader
"varying vec4 v_color;\n"
"\n"
"void main()\n"
"{\n"
"    gl_FragColor = v_color;\n"
"}",
        // fragment version
        NULL
    },

    // SHADER_RGB
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
"varying vec4 v_color;\n"
"varying vec2 v_texCoord;\n"
"uniform sampler2D tex0;\n"
"uniform vec4 texel_size; // texel size (xy: texel size, zw: texture dimensions)\n"
"\n"
"void main()\n"
"{\n"
"    gl_FragColor = texture2D(tex0, v_texCoord);\n"
"    gl_FragColor.a = 1.0;\n"
"    gl_FragColor *= v_color;\n"
"}",
        // fragment version
        NULL
    },

    // SHADER_RGBA
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
"varying vec4 v_color;\n"
"varying vec2 v_texCoord;\n"
"uniform sampler2D tex0;\n"
"\n"
"void main()\n"
"{\n"
"    gl_FragColor = texture2D(tex0, v_texCoord) * v_color;\n"
"}",
        // fragment version
        NULL
    },

    // SHADER_RGB_PIXELART
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
"varying vec4 v_color;\n"
"varying vec2 v_texCoord;\n"
"uniform sampler2D tex0;\n"
"uniform vec4 texel_size;\n"
"\n"
"void main()\n"
"{\n"
"    vec2 boxSize = clamp(fwidth(v_texCoord) * texel_size.zw, 1e-5, 1.0);\n"
"    vec2 tx = v_texCoord * texel_size.zw - 0.5 * boxSize;\n"
"    vec2 txOffset = smoothstep(vec2(1.0) - boxSize, vec2(1.0), fract(tx));\n"
"    vec2 uv = (floor(tx) + 0.5 + txOffset) * texel_size.xy;\n"
"    gl_FragColor = textureGrad(tex0, uv, dFdx(v_texCoord), dFdy(v_texCoord));\n"
"    gl_FragColor.a = 1.0;\n"
"    gl_FragColor *= v_color;\n"
"}",
        // fragment version
        "#version 130\n"
    },

    // SHADER_RGBA_PIXELART
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
"varying vec4 v_color;\n"
"varying vec2 v_texCoord;\n"
"uniform sampler2D tex0;\n"
"uniform vec4 texel_size;\n"
"\n"
"void main()\n"
"{\n"
"    vec2 boxSize = clamp(fwidth(v_texCoord) * texel_size.zw, 1e-5, 1.0);\n"
"    vec2 tx = v_texCoord * texel_size.zw - 0.5 * boxSize;\n"
"    vec2 txOffset = smoothstep(vec2(1.0) - boxSize, vec2(1.0), fract(tx));\n"
"    vec2 uv = (floor(tx) + 0.5 + txOffset) * texel_size.xy;\n"
"    gl_FragColor = textureGrad(tex0, uv, dFdx(v_texCoord), dFdy(v_texCoord));\n"
"    gl_FragColor *= v_color;\n"
"}",
        // fragment version
        "#version 130\n"
    },

#ifdef SDL_HAVE_YUV
    // SHADER_YUV
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
        YUV_SHADER_PROLOGUE
        YUV_SHADER_BODY,
        // fragment version
        NULL
    },
    // SHADER_NV12_RA
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
        NV12_SHADER_PROLOGUE
        NV12_RA_SHADER_BODY,
        // fragment version
        NULL
    },
    // SHADER_NV12_RG
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
        NV12_SHADER_PROLOGUE
        NV12_RG_SHADER_BODY,
        // fragment version
        NULL
    },
    // SHADER_NV21_RA
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
        NV12_SHADER_PROLOGUE
        NV21_RA_SHADER_BODY,
        // fragment version
        NULL
    },
    // SHADER_NV21_RG
    {
        // vertex shader
        TEXTURE_VERTEX_SHADER,
        // fragment shader
        NV12_SHADER_PROLOGUE
        NV21_RG_SHADER_BODY,
        // fragment version
        NULL
    },
#endif // SDL_HAVE_YUV
};

/* *INDENT-ON* */ // clang-format on

static bool CompileShader(GL_ShaderContext *ctx, GLhandleARB shader, const char *version, const char *defines, const char *source)
{
    GLint status;
    const char *sources[3];

    sources[0] = version;
    sources[1] = defines;
    sources[2] = source;

    ctx->glShaderSourceARB(shader, SDL_arraysize(sources), sources, NULL);
    ctx->glCompileShaderARB(shader);
    ctx->glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &status);
    if (status == 0) {
        bool isstack;
        GLint length;
        char *info;

        ctx->glGetObjectParameterivARB(shader, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
        info = SDL_small_alloc(char, length + 1, &isstack);
        if (info) {
            ctx->glGetInfoLogARB(shader, length, NULL, info);
            SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to compile shader:");
	    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s", defines);
	    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s", source);
	    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s", info);
            SDL_small_free(info, isstack);
        }
        return false;
    } else {
        return true;
    }
}

static bool CompileShaderProgram(GL_ShaderContext *ctx, int index, GL_ShaderData *data)
{
    const int num_tmus_bound = 4;
    const char *vert_defines = "";
    const char *frag_defines = "";
    const char *frag_version = "";
    int i;
    GLint location;

    if (index == SHADER_NONE) {
        return true;
    }

    ctx->glGetError();

    // Make sure we use the correct sampler type for our texture type
    if (ctx->GL_ARB_texture_rectangle_supported) {
        frag_defines =
            "#define sampler2D sampler2DRect\n"
            "#define texture2D texture2DRect\n"
            "#define UVCoordScale 0.5\n";
    } else {
        frag_defines =
            "#define UVCoordScale 1.0\n";
    }
    if (shader_source[index].fragment_version) {
        frag_version = shader_source[index].fragment_version;
    }

    // Create one program object to rule them all
    data->program = ctx->glCreateProgramObjectARB();

    // Create the vertex shader
    data->vert_shader = ctx->glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
    if (!CompileShader(ctx, data->vert_shader, "", vert_defines, shader_source[index].vertex_shader)) {
        return false;
    }

    // Create the fragment shader
    data->frag_shader = ctx->glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    if (!CompileShader(ctx, data->frag_shader, frag_version, frag_defines, shader_source[index].fragment_shader)) {
        return false;
    }

    // ... and in the darkness bind them
    ctx->glAttachObjectARB(data->program, data->vert_shader);
    ctx->glAttachObjectARB(data->program, data->frag_shader);
    ctx->glLinkProgramARB(data->program);

    // Set up some uniform variables
    ctx->glUseProgramObjectARB(data->program);
    for (i = 0; i < num_tmus_bound; ++i) {
        char tex_name[10];
        (void)SDL_snprintf(tex_name, SDL_arraysize(tex_name), "tex%d", i);
        location = ctx->glGetUniformLocationARB(data->program, tex_name);
        if (location >= 0) {
            ctx->glUniform1iARB(location, i);
        }
    }
    ctx->glUseProgramObjectARB(0);

    return ctx->glGetError() == GL_NO_ERROR;
}

static void DestroyShaderProgram(GL_ShaderContext *ctx, GL_ShaderData *data)
{
    ctx->glDeleteObjectARB(data->vert_shader);
    ctx->glDeleteObjectARB(data->frag_shader);
    ctx->glDeleteObjectARB(data->program);
}

GL_ShaderContext *GL_CreateShaderContext(void)
{
    GL_ShaderContext *ctx;
    bool shaders_supported;
    int i;

    ctx = (GL_ShaderContext *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }

    if (!SDL_GL_ExtensionSupported("GL_ARB_texture_non_power_of_two") &&
        (SDL_GL_ExtensionSupported("GL_ARB_texture_rectangle") ||
         SDL_GL_ExtensionSupported("GL_EXT_texture_rectangle"))) {
        ctx->GL_ARB_texture_rectangle_supported = true;
    }

    // Check for shader support
    shaders_supported = false;
    if (SDL_GL_ExtensionSupported("GL_ARB_shader_objects") &&
        SDL_GL_ExtensionSupported("GL_ARB_shading_language_100") &&
        SDL_GL_ExtensionSupported("GL_ARB_vertex_shader") &&
        SDL_GL_ExtensionSupported("GL_ARB_fragment_shader")) {
        ctx->glGetError = (GLenum(*)(void))SDL_GL_GetProcAddress("glGetError");
        ctx->glAttachObjectARB = (PFNGLATTACHOBJECTARBPROC)SDL_GL_GetProcAddress("glAttachObjectARB");
        ctx->glCompileShaderARB = (PFNGLCOMPILESHADERARBPROC)SDL_GL_GetProcAddress("glCompileShaderARB");
        ctx->glCreateProgramObjectARB = (PFNGLCREATEPROGRAMOBJECTARBPROC)SDL_GL_GetProcAddress("glCreateProgramObjectARB");
        ctx->glCreateShaderObjectARB = (PFNGLCREATESHADEROBJECTARBPROC)SDL_GL_GetProcAddress("glCreateShaderObjectARB");
        ctx->glDeleteObjectARB = (PFNGLDELETEOBJECTARBPROC)SDL_GL_GetProcAddress("glDeleteObjectARB");
        ctx->glGetInfoLogARB = (PFNGLGETINFOLOGARBPROC)SDL_GL_GetProcAddress("glGetInfoLogARB");
        ctx->glGetObjectParameterivARB = (PFNGLGETOBJECTPARAMETERIVARBPROC)SDL_GL_GetProcAddress("glGetObjectParameterivARB");
        ctx->glGetUniformLocationARB = (PFNGLGETUNIFORMLOCATIONARBPROC)SDL_GL_GetProcAddress("glGetUniformLocationARB");
        ctx->glLinkProgramARB = (PFNGLLINKPROGRAMARBPROC)SDL_GL_GetProcAddress("glLinkProgramARB");
        ctx->glShaderSourceARB = (PFNGLSHADERSOURCEARBPROC)SDL_GL_GetProcAddress("glShaderSourceARB");
        ctx->glUniform1iARB = (PFNGLUNIFORM1IARBPROC)SDL_GL_GetProcAddress("glUniform1iARB");
        ctx->glUniform1fARB = (PFNGLUNIFORM1FARBPROC)SDL_GL_GetProcAddress("glUniform1fARB");
        ctx->glUniform3fARB = (PFNGLUNIFORM3FARBPROC)SDL_GL_GetProcAddress("glUniform3fARB");
        ctx->glUniform4fARB = (PFNGLUNIFORM4FARBPROC)SDL_GL_GetProcAddress("glUniform4fARB");
        ctx->glUseProgramObjectARB = (PFNGLUSEPROGRAMOBJECTARBPROC)SDL_GL_GetProcAddress("glUseProgramObjectARB");
        if (ctx->glGetError &&
            ctx->glAttachObjectARB &&
            ctx->glCompileShaderARB &&
            ctx->glCreateProgramObjectARB &&
            ctx->glCreateShaderObjectARB &&
            ctx->glDeleteObjectARB &&
            ctx->glGetInfoLogARB &&
            ctx->glGetObjectParameterivARB &&
            ctx->glGetUniformLocationARB &&
            ctx->glLinkProgramARB &&
            ctx->glShaderSourceARB &&
            ctx->glUniform1iARB &&
            ctx->glUniform1fARB &&
            ctx->glUniform3fARB &&
            ctx->glUseProgramObjectARB) {
            shaders_supported = true;
        }
    }

    if (!shaders_supported) {
        SDL_free(ctx);
        return NULL;
    }

    // Compile all the shaders
    for (i = 0; i < NUM_SHADERS; ++i) {
        if (!CompileShaderProgram(ctx, i, &ctx->shaders[i])) {
            GL_DestroyShaderContext(ctx);
            return NULL;
        }
    }

    // We're done!
    return ctx;
}

void GL_SelectShader(GL_ShaderContext *ctx, GL_Shader shader, const float *shader_params)
{
    GLint location;
    GLhandleARB program = ctx->shaders[shader].program;

    ctx->glUseProgramObjectARB(program);

    if (shader_params && shader_params != ctx->shader_params[shader]) {
        if (shader == SHADER_RGB_PIXELART ||
            shader == SHADER_RGBA_PIXELART) {
            location = ctx->glGetUniformLocationARB(program, "texel_size");
            if (location >= 0) {
                ctx->glUniform4fARB(location, shader_params[0], shader_params[1], shader_params[2], shader_params[3]);
            }
        }

#ifdef SDL_HAVE_YUV
        if (shader >= SHADER_YUV) {
            // YUV shader params are Yoffset, 0, Rcoeff, 0, Gcoeff, 0, Bcoeff, 0
            location = ctx->glGetUniformLocationARB(program, "Yoffset");
            if (location >= 0) {
                ctx->glUniform3fARB(location, shader_params[0], shader_params[1], shader_params[2]);
            }
            location = ctx->glGetUniformLocationARB(program, "Rcoeff");
            if (location >= 0) {
                ctx->glUniform3fARB(location, shader_params[4], shader_params[5], shader_params[6]);
            }
            location = ctx->glGetUniformLocationARB(program, "Gcoeff");
            if (location >= 0) {
                ctx->glUniform3fARB(location, shader_params[8], shader_params[9], shader_params[10]);
            }
            location = ctx->glGetUniformLocationARB(program, "Bcoeff");
            if (location >= 0) {
                ctx->glUniform3fARB(location, shader_params[12], shader_params[13], shader_params[14]);
            }
        }
#endif // SDL_HAVE_YUV

        ctx->shader_params[shader] = shader_params;
    }
}

void GL_DestroyShaderContext(GL_ShaderContext *ctx)
{
    int i;

    for (i = 0; i < NUM_SHADERS; ++i) {
        DestroyShaderProgram(ctx, &ctx->shaders[i]);
    }
    SDL_free(ctx);
}

#endif // SDL_VIDEO_RENDER_OGL
