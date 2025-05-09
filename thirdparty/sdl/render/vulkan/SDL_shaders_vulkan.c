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

#ifdef SDL_VIDEO_RENDER_VULKAN

#include "SDL_shaders_vulkan.h"

// The shaders here were compiled with compile_shaders.bat
#include "VULKAN_PixelShader_Colors.h"
#include "VULKAN_PixelShader_Textures.h"
#include "VULKAN_PixelShader_Advanced.h"
#include "VULKAN_VertexShader.h"

static struct
{
    const void *ps_shader_data;
    size_t ps_shader_size;
    const void *vs_shader_data;
    size_t vs_shader_size;
} VULKAN_shaders[NUM_SHADERS] = {
    { VULKAN_PixelShader_Colors, sizeof(VULKAN_PixelShader_Colors),
      VULKAN_VertexShader, sizeof(VULKAN_VertexShader) },
    { VULKAN_PixelShader_Textures, sizeof(VULKAN_PixelShader_Textures),
      VULKAN_VertexShader, sizeof(VULKAN_VertexShader) },
    { VULKAN_PixelShader_Advanced, sizeof(VULKAN_PixelShader_Advanced),
      VULKAN_VertexShader, sizeof(VULKAN_VertexShader) },
};

void VULKAN_GetVertexShader(VULKAN_Shader shader, const uint32_t **outBytecode, size_t *outSize)
{
    *outBytecode = (const uint32_t *)VULKAN_shaders[shader].vs_shader_data;
    *outSize = VULKAN_shaders[shader].vs_shader_size;
}

void VULKAN_GetPixelShader(VULKAN_Shader shader, const uint32_t **outBytecode, size_t *outSize)
{
    *outBytecode = (const uint32_t *)VULKAN_shaders[shader].ps_shader_data;
    *outSize = VULKAN_shaders[shader].ps_shader_size;
}

#endif // SDL_VIDEO_RENDER_VULKAN
