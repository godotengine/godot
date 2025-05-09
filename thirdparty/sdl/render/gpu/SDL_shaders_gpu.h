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

#ifndef SDL_shaders_gpu_h_
#define SDL_shaders_gpu_h_

#include "SDL_internal.h"

// SDL_GPU shader implementation

typedef enum
{
    VERT_SHADER_INVALID = -1,
    VERT_SHADER_LINEPOINT,
    VERT_SHADER_TRI_COLOR,
    VERT_SHADER_TRI_TEXTURE,

    NUM_VERT_SHADERS,
} GPU_VertexShaderID;

typedef enum
{
    FRAG_SHADER_INVALID = -1,
    FRAG_SHADER_COLOR,
    FRAG_SHADER_TEXTURE_RGB,
    FRAG_SHADER_TEXTURE_RGBA,
    FRAG_SHADER_TEXTURE_RGB_PIXELART,
    FRAG_SHADER_TEXTURE_RGBA_PIXELART,
    FRAG_SHADER_TEXTURE_CUSTOM,

    NUM_FRAG_SHADERS,
} GPU_FragmentShaderID;

struct GPU_Shaders
{
    SDL_GPUShader *vert_shaders[NUM_VERT_SHADERS];
    SDL_GPUShader *frag_shaders[NUM_FRAG_SHADERS];
};

typedef struct GPU_Shaders GPU_Shaders;

void GPU_FillSupportedShaderFormats(SDL_PropertiesID props);
extern bool GPU_InitShaders(GPU_Shaders *shaders, SDL_GPUDevice *device);
extern void GPU_ReleaseShaders(GPU_Shaders *shaders, SDL_GPUDevice *device);
extern SDL_GPUShader *GPU_GetVertexShader(GPU_Shaders *shaders, GPU_VertexShaderID id);
extern SDL_GPUShader *GPU_GetFragmentShader(GPU_Shaders *shaders, GPU_FragmentShaderID id);

#endif // SDL_shaders_gpu_h_
