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

#ifndef SDL_pipeline_gpu_h_
#define SDL_pipeline_gpu_h_

#include "SDL_internal.h"

#include "SDL_shaders_gpu.h"

typedef struct GPU_PipelineParameters
{
    SDL_BlendMode blend_mode;
    GPU_FragmentShaderID frag_shader;
    GPU_VertexShaderID vert_shader;
    SDL_GPUTextureFormat attachment_format;
    SDL_GPUPrimitiveType primitive_type;
    SDL_GPUShader *custom_frag_shader;
} GPU_PipelineParameters;

typedef struct GPU_PipelineCache
{
    SDL_HashTable *table;
} GPU_PipelineCache;

extern bool GPU_InitPipelineCache(GPU_PipelineCache *cache, SDL_GPUDevice *device);
extern void GPU_DestroyPipelineCache(GPU_PipelineCache *cache);
extern SDL_GPUGraphicsPipeline *GPU_GetPipeline(GPU_PipelineCache *cache, GPU_Shaders *shaders, SDL_GPUDevice *device, const GPU_PipelineParameters *params);

#endif // SDL_pipeline_gpu_h_
