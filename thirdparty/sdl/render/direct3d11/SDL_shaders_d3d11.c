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

#ifdef SDL_VIDEO_RENDER_D3D11

#define COBJMACROS
#include "../../core/windows/SDL_windows.h"
#include <d3d11_1.h>

#include "SDL_shaders_d3d11.h"

#define SDL_COMPOSE_ERROR(str) SDL_STRINGIFY_ARG(__FUNCTION__) ", " str

#if SDL_WINAPI_FAMILY_PHONE
#error Need to build shaders with level_9_3
#endif

// The shaders here were compiled with compile_shaders.bat

#define g_main D3D11_PixelShader_Colors
#include "D3D11_PixelShader_Colors.h"
#undef g_main

#define g_main D3D11_PixelShader_Textures
#include "D3D11_PixelShader_Textures.h"
#undef g_main

#define g_main D3D11_PixelShader_Advanced
#include "D3D11_PixelShader_Advanced.h"
#undef g_main

#define g_main D3D11_VertexShader
#include "D3D11_VertexShader.h"
#undef g_main


static struct
{
    const void *shader_data;
    SIZE_T shader_size;
} D3D11_shaders[] = {
    { NULL, 0 },
    { D3D11_PixelShader_Colors, sizeof(D3D11_PixelShader_Colors) },
    { D3D11_PixelShader_Textures, sizeof(D3D11_PixelShader_Textures) },
    { D3D11_PixelShader_Advanced, sizeof(D3D11_PixelShader_Advanced) },
};
SDL_COMPILE_TIME_ASSERT(D3D11_shaders, SDL_arraysize(D3D11_shaders) == NUM_SHADERS);

bool D3D11_CreateVertexShader(ID3D11Device1 *d3dDevice, ID3D11VertexShader **vertexShader, ID3D11InputLayout **inputLayout)
{
    // Declare how the input layout for SDL's vertex shader will be setup:
    const D3D11_INPUT_ELEMENT_DESC vertexDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    HRESULT result;

    // Load in SDL's one and only vertex shader:
    result = ID3D11Device_CreateVertexShader(d3dDevice,
                                             D3D11_VertexShader,
                                             sizeof(D3D11_VertexShader),
                                             NULL,
                                             vertexShader);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateVertexShader"), result);
    }

    // Create an input layout for SDL's vertex shader:
    result = ID3D11Device_CreateInputLayout(d3dDevice,
                                            vertexDesc,
                                            ARRAYSIZE(vertexDesc),
                                            D3D11_VertexShader,
                                            sizeof(D3D11_VertexShader),
                                            inputLayout);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateInputLayout"), result);
    }
    return true;
}

bool D3D11_CreatePixelShader(ID3D11Device1 *d3dDevice, D3D11_Shader shader, ID3D11PixelShader **pixelShader)
{
    HRESULT result;

    result = ID3D11Device_CreatePixelShader(d3dDevice,
                                            D3D11_shaders[shader].shader_data,
                                            D3D11_shaders[shader].shader_size,
                                            NULL,
                                            pixelShader);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreatePixelShader"), result);
    }
    return true;
}

#endif // SDL_VIDEO_RENDER_D3D11
