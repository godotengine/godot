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

#ifdef SDL_VIDEO_RENDER_D3D

#include "../../core/windows/SDL_windows.h"

#include <d3d9.h>

#include "SDL_shaders_d3d.h"

// The shaders here were compiled with compile_shaders.bat

#define g_ps20_main D3D9_PixelShader_YUV
#include "D3D9_PixelShader_YUV.h"
#undef g_ps20_main

static const BYTE *D3D9_shaders[] = {
    NULL,
    D3D9_PixelShader_YUV
};
SDL_COMPILE_TIME_ASSERT(D3D9_shaders, SDL_arraysize(D3D9_shaders) == NUM_SHADERS);

HRESULT D3D9_CreatePixelShader(IDirect3DDevice9 *d3dDevice, D3D9_Shader shader, IDirect3DPixelShader9 **pixelShader)
{
    return IDirect3DDevice9_CreatePixelShader(d3dDevice, (const DWORD *)D3D9_shaders[shader], pixelShader);
}

#endif // SDL_VIDEO_RENDER_D3D
