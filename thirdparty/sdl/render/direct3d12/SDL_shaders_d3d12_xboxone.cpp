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
#include "../../SDL_internal.h"

#if defined(SDL_VIDEO_RENDER_D3D12) && defined(SDL_PLATFORM_XBOXONE)

#include <SDL3/SDL_stdinc.h>

#include "../../core/windows/SDL_windows.h"
#include "../../video/directx/SDL_d3d12.h"

#include "SDL_shaders_d3d12.h"

#define SDL_COMPOSE_ERROR(str) SDL_STRINGIFY_ARG(__FUNCTION__) ", " str

// Shader blob headers are generated with a pre-build step using compile_shaders_xbox.bat

#define g_main D3D12_PixelShader_Colors
#include "D3D12_PixelShader_Colors_One.h"
#undef g_main

#define g_main D3D12_PixelShader_Textures
#include "D3D12_PixelShader_Textures_One.h"
#undef g_main

#define g_main D3D12_PixelShader_Advanced
#include "D3D12_PixelShader_Advanced_One.h"
#undef g_main


#define g_mainColor D3D12_VertexShader_Colors
#include "D3D12_VertexShader_Color_One.h"
#undef g_mainColor

#define g_mainTexture D3D12_VertexShader_Textures
#include "D3D12_VertexShader_Texture_One.h"
#undef g_mainTexture

#define g_mainAdvanced D3D12_VertexShader_Advanced
#include "D3D12_VertexShader_Advanced_One.h"
#undef g_mainAdvanced


#define g_ColorRS D3D12_RootSig_Color
#include "D3D12_RootSig_Color_One.h"
#undef g_ColorRS

#define g_TextureRS D3D12_RootSig_Texture
#include "D3D12_RootSig_Texture_One.h"
#undef g_TextureRS

#define g_AdvancedRS D3D12_RootSig_Advanced
#include "D3D12_RootSig_Advanced_One.h"
#undef g_AdvancedRS


static struct
{
    const void *ps_shader_data;
    SIZE_T ps_shader_size;
    const void *vs_shader_data;
    SIZE_T vs_shader_size;
    D3D12_RootSignature root_sig;
} D3D12_shaders[NUM_SHADERS] = {
    { D3D12_PixelShader_Colors, sizeof(D3D12_PixelShader_Colors),
      D3D12_VertexShader_Colors, sizeof(D3D12_VertexShader_Colors),
      ROOTSIG_COLOR },
    { D3D12_PixelShader_Textures, sizeof(D3D12_PixelShader_Textures),
      D3D12_VertexShader_Textures, sizeof(D3D12_VertexShader_Textures),
      ROOTSIG_TEXTURE },
    { D3D12_PixelShader_Advanced, sizeof(D3D12_PixelShader_Advanced),
      D3D12_VertexShader_Advanced, sizeof(D3D12_VertexShader_Advanced),
      ROOTSIG_ADVANCED },
};

static struct
{
    const void *rs_shader_data;
    SIZE_T rs_shader_size;
} D3D12_rootsigs[NUM_ROOTSIGS] = {
    { D3D12_RootSig_Color, sizeof(D3D12_RootSig_Color) },
    { D3D12_RootSig_Texture, sizeof(D3D12_RootSig_Texture) },
    { D3D12_RootSig_Advanced, sizeof(D3D12_RootSig_Advanced) },
};

extern "C"
void D3D12_GetVertexShader(D3D12_Shader shader, D3D12_SHADER_BYTECODE *outBytecode)
{
    outBytecode->pShaderBytecode = D3D12_shaders[shader].vs_shader_data;
    outBytecode->BytecodeLength = D3D12_shaders[shader].vs_shader_size;
}

extern "C"
void D3D12_GetPixelShader(D3D12_Shader shader, D3D12_SHADER_BYTECODE *outBytecode)
{
    outBytecode->pShaderBytecode = D3D12_shaders[shader].ps_shader_data;
    outBytecode->BytecodeLength = D3D12_shaders[shader].ps_shader_size;
}

extern "C"
D3D12_RootSignature D3D12_GetRootSignatureType(D3D12_Shader shader)
{
    return D3D12_shaders[shader].root_sig;
}

extern "C"
void D3D12_GetRootSignatureData(D3D12_RootSignature rootSig, D3D12_SHADER_BYTECODE *outBytecode)
{
    outBytecode->pShaderBytecode = D3D12_rootsigs[rootSig].rs_shader_data;
    outBytecode->BytecodeLength = D3D12_rootsigs[rootSig].rs_shader_size;
}

#endif  // SDL_VIDEO_RENDER_D3D12 && SDL_PLATFORM_XBOXONE

