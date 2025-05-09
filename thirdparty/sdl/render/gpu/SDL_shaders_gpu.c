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

#ifdef SDL_VIDEO_RENDER_GPU

#include "SDL_shaders_gpu.h"

// SDL_GPU shader implementation

typedef struct GPU_ShaderModuleSource
{
    const unsigned char *code;
    unsigned int code_len;
    SDL_GPUShaderFormat format;
} GPU_ShaderModuleSource;

#if defined(SDL_GPU_VULKAN) && SDL_GPU_VULKAN
#define IF_VULKAN(...)     __VA_ARGS__
#define HAVE_SPIRV_SHADERS 1
#include "shaders/spir-v.h"
#else
#define IF_VULKAN(...)
#define HAVE_SPIRV_SHADERS 0
#endif

#ifdef SDL_GPU_D3D12
#define IF_D3D12(...)       __VA_ARGS__
#define HAVE_DXIL60_SHADERS 1
#include "shaders/dxil.h"
#else
#define IF_D3D12(...)
#define HAVE_DXIL60_SHADERS 0
#endif

#ifdef SDL_GPU_METAL
#define IF_METAL(...)      __VA_ARGS__
#define HAVE_METAL_SHADERS 1
#include "shaders/msl.h"
#else
#define IF_METAL(...)
#define HAVE_METAL_SHADERS 0
#endif

typedef struct GPU_ShaderSources
{
    IF_VULKAN(GPU_ShaderModuleSource spirv;)
    IF_D3D12(GPU_ShaderModuleSource dxil60;)
    IF_METAL(GPU_ShaderModuleSource msl;)
    unsigned int num_samplers;
    unsigned int num_uniform_buffers;
} GPU_ShaderSources;

#define SHADER_SPIRV(code) \
    IF_VULKAN(.spirv = { code, sizeof(code), SDL_GPU_SHADERFORMAT_SPIRV }, )

#define SHADER_DXIL60(code) \
    IF_D3D12(.dxil60 = { code, sizeof(code), SDL_GPU_SHADERFORMAT_DXIL }, )

#define SHADER_METAL(code) \
    IF_METAL(.msl = { code, sizeof(code), SDL_GPU_SHADERFORMAT_MSL }, )

// clang-format off
static const GPU_ShaderSources vert_shader_sources[NUM_VERT_SHADERS] = {
    [VERT_SHADER_LINEPOINT] = {
        .num_samplers = 0,
        .num_uniform_buffers = 1,
        SHADER_SPIRV(linepoint_vert_spv)
        SHADER_DXIL60(linepoint_vert_dxil)
        SHADER_METAL(linepoint_vert_msl)
    },
    [VERT_SHADER_TRI_COLOR] = {
        .num_samplers = 0,
        .num_uniform_buffers = 1,
        SHADER_SPIRV(tri_color_vert_spv)
        SHADER_DXIL60(tri_color_vert_dxil)
        SHADER_METAL(tri_color_vert_msl)
    },
    [VERT_SHADER_TRI_TEXTURE] = {
        .num_samplers = 0,
        .num_uniform_buffers = 1,
        SHADER_SPIRV(tri_texture_vert_spv)
        SHADER_DXIL60(tri_texture_vert_dxil)
        SHADER_METAL(tri_texture_vert_msl)
    },
};

static const GPU_ShaderSources frag_shader_sources[NUM_FRAG_SHADERS] = {
    [FRAG_SHADER_COLOR] = {
        .num_samplers = 0,
        .num_uniform_buffers = 0,
        SHADER_SPIRV(color_frag_spv)
        SHADER_DXIL60(color_frag_dxil)
        SHADER_METAL(color_frag_msl)
    },
    [FRAG_SHADER_TEXTURE_RGB] = {
        .num_samplers = 1,
        .num_uniform_buffers = 0,
        SHADER_SPIRV(texture_rgb_frag_spv)
        SHADER_DXIL60(texture_rgb_frag_dxil)
        SHADER_METAL(texture_rgb_frag_msl)
    },
    [FRAG_SHADER_TEXTURE_RGBA] = {
        .num_samplers = 1,
        .num_uniform_buffers = 0,
        SHADER_SPIRV(texture_rgba_frag_spv)
        SHADER_DXIL60(texture_rgba_frag_dxil)
        SHADER_METAL(texture_rgba_frag_msl)
    },
    [FRAG_SHADER_TEXTURE_RGB_PIXELART] = {
        .num_samplers = 1,
        .num_uniform_buffers = 1,
        SHADER_SPIRV(texture_rgb_pixelart_frag_spv)
        SHADER_DXIL60(texture_rgb_pixelart_frag_dxil)
        SHADER_METAL(texture_rgb_pixelart_frag_msl)
    },
    [FRAG_SHADER_TEXTURE_RGBA_PIXELART] = {
        .num_samplers = 1,
        .num_uniform_buffers = 1,
        SHADER_SPIRV(texture_rgba_pixelart_frag_spv)
        SHADER_DXIL60(texture_rgba_pixelart_frag_dxil)
        SHADER_METAL(texture_rgba_pixelart_frag_msl)
    },
};
// clang-format on

static SDL_GPUShader *CompileShader(const GPU_ShaderSources *sources, SDL_GPUDevice *device, SDL_GPUShaderStage stage)
{
    const GPU_ShaderModuleSource *sms = NULL;
    SDL_GPUShaderFormat formats = SDL_GetGPUShaderFormats(device);

    if (formats == SDL_GPU_SHADERFORMAT_INVALID) {
        // SDL_GetGPUShaderFormats already set the error
        return NULL;
#if HAVE_SPIRV_SHADERS
    } else if (formats & SDL_GPU_SHADERFORMAT_SPIRV) {
        sms = &sources->spirv;
#endif // HAVE_SPIRV_SHADERS
#if HAVE_DXIL60_SHADERS
    } else if (formats & SDL_GPU_SHADERFORMAT_DXIL) {
        sms = &sources->dxil60;
#endif // HAVE_DXIL60_SHADERS
#if HAVE_METAL_SHADERS
    } else if (formats & SDL_GPU_SHADERFORMAT_MSL) {
        sms = &sources->msl;
#endif // HAVE_METAL_SHADERS
    } else {
        SDL_SetError("Unsupported GPU backend");
        return NULL;
    }

    SDL_GPUShaderCreateInfo sci = { 0 };
    sci.code = sms->code;
    sci.code_size = sms->code_len;
    sci.format = sms->format;
    // FIXME not sure if this is correct
    sci.entrypoint =
#if HAVE_METAL_SHADERS
        (sms == &sources->msl) ? "main0" :
#endif // HAVE_METAL_SHADERS
        "main";
    sci.num_samplers = sources->num_samplers;
    sci.num_uniform_buffers = sources->num_uniform_buffers;
    sci.stage = stage;

    return SDL_CreateGPUShader(device, &sci);
}

bool GPU_InitShaders(GPU_Shaders *shaders, SDL_GPUDevice *device)
{
    for (int i = 0; i < SDL_arraysize(vert_shader_sources); ++i) {
        shaders->vert_shaders[i] = CompileShader(
            &vert_shader_sources[i], device, SDL_GPU_SHADERSTAGE_VERTEX);
        if (shaders->vert_shaders[i] == NULL) {
            GPU_ReleaseShaders(shaders, device);
            return false;
        }
    }

    for (int i = 0; i < SDL_arraysize(frag_shader_sources); ++i) {
        if (i == FRAG_SHADER_TEXTURE_CUSTOM) {
            continue;
        }
        shaders->frag_shaders[i] = CompileShader(
            &frag_shader_sources[i], device, SDL_GPU_SHADERSTAGE_FRAGMENT);
        if (shaders->frag_shaders[i] == NULL) {
            GPU_ReleaseShaders(shaders, device);
            return false;
        }
    }

    return true;
}

void GPU_ReleaseShaders(GPU_Shaders *shaders, SDL_GPUDevice *device)
{
    for (int i = 0; i < SDL_arraysize(shaders->vert_shaders); ++i) {
        SDL_ReleaseGPUShader(device, shaders->vert_shaders[i]);
        shaders->vert_shaders[i] = NULL;
    }

    for (int i = 0; i < SDL_arraysize(shaders->frag_shaders); ++i) {
        if (i == FRAG_SHADER_TEXTURE_CUSTOM) {
            continue;
        }
        SDL_ReleaseGPUShader(device, shaders->frag_shaders[i]);
        shaders->frag_shaders[i] = NULL;
    }
}

SDL_GPUShader *GPU_GetVertexShader(GPU_Shaders *shaders, GPU_VertexShaderID id)
{
    SDL_assert((unsigned int)id < SDL_arraysize(shaders->vert_shaders));
    SDL_GPUShader *shader = shaders->vert_shaders[id];
    SDL_assert(shader != NULL);
    return shader;
}

SDL_GPUShader *GPU_GetFragmentShader(GPU_Shaders *shaders, GPU_FragmentShaderID id)
{
    SDL_assert((unsigned int)id < SDL_arraysize(shaders->frag_shaders));
    SDL_GPUShader *shader = shaders->frag_shaders[id];
    SDL_assert(shader != NULL);
    return shader;
}

void GPU_FillSupportedShaderFormats(SDL_PropertiesID props)
{
    bool custom_shaders = false;
    if (SDL_GetBooleanProperty(props, SDL_PROP_RENDERER_CREATE_GPU_SHADERS_SPIRV_BOOLEAN, false)) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, HAVE_SPIRV_SHADERS);
        custom_shaders = true;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_RENDERER_CREATE_GPU_SHADERS_DXIL_BOOLEAN, false)) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXIL_BOOLEAN, HAVE_DXIL60_SHADERS);
        custom_shaders = true;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_RENDERER_CREATE_GPU_SHADERS_MSL_BOOLEAN, false)) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_MSL_BOOLEAN, HAVE_METAL_SHADERS);
        custom_shaders = true;
    }
    if (!custom_shaders) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, HAVE_SPIRV_SHADERS);
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXIL_BOOLEAN, HAVE_DXIL60_SHADERS);
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_MSL_BOOLEAN, HAVE_METAL_SHADERS);
    }
}

#endif // SDL_VIDEO_RENDER_GPU
