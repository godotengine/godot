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

#ifndef SDL_shaders_gles2_h_
#define SDL_shaders_gles2_h_

#ifdef SDL_VIDEO_RENDER_OGL_ES2

typedef enum
{
    GLES2_SHADER_FRAGMENT_INCLUDE_NONE = 0,
    GLES2_SHADER_FRAGMENT_INCLUDE_BEST_TEXCOORD_PRECISION,
    GLES2_SHADER_FRAGMENT_INCLUDE_MEDIUM_TEXCOORD_PRECISION,
    GLES2_SHADER_FRAGMENT_INCLUDE_HIGH_TEXCOORD_PRECISION,
    GLES2_SHADER_FRAGMENT_INCLUDE_UNDEF_PRECISION,
    GLES2_SHADER_FRAGMENT_INCLUDE_COUNT
} GLES2_ShaderIncludeType;

typedef enum
{
    GLES2_SHADER_VERTEX_DEFAULT = 0,
    GLES2_SHADER_FRAGMENT_SOLID,
    GLES2_SHADER_FRAGMENT_TEXTURE_ABGR,
    GLES2_SHADER_FRAGMENT_TEXTURE_ARGB,
    GLES2_SHADER_FRAGMENT_TEXTURE_BGR,
    GLES2_SHADER_FRAGMENT_TEXTURE_RGB,
    GLES2_SHADER_FRAGMENT_TEXTURE_ABGR_PIXELART,
    GLES2_SHADER_FRAGMENT_TEXTURE_ARGB_PIXELART,
    GLES2_SHADER_FRAGMENT_TEXTURE_BGR_PIXELART,
    GLES2_SHADER_FRAGMENT_TEXTURE_RGB_PIXELART,
#ifdef SDL_HAVE_YUV
    GLES2_SHADER_FRAGMENT_TEXTURE_YUV,
    GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RA,
    GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RG,
    GLES2_SHADER_FRAGMENT_TEXTURE_NV21_RA,
    GLES2_SHADER_FRAGMENT_TEXTURE_NV21_RG,
#endif
    // Shaders beyond this point are optional and not cached at render creation
    GLES2_SHADER_FRAGMENT_TEXTURE_EXTERNAL_OES,
    GLES2_SHADER_COUNT
} GLES2_ShaderType;

extern const char *GLES2_GetShaderPrologue(GLES2_ShaderType type);
extern const char *GLES2_GetShaderInclude(GLES2_ShaderIncludeType type);
extern const char *GLES2_GetShader(GLES2_ShaderType type);
extern GLES2_ShaderIncludeType GLES2_GetTexCoordPrecisionEnumFromHint(void);

#endif // SDL_VIDEO_RENDER_OGL_ES2

#endif // SDL_shaders_gles2_h_
