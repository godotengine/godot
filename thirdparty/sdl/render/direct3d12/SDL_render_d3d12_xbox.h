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

#ifndef SDL_render_d3d12_xbox_h_
#define SDL_render_d3d12_xbox_h_

#include "../../SDL_internal.h"
#include "../../video/directx/SDL_d3d12.h"

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

extern HRESULT D3D12_XBOX_CreateDevice(ID3D12Device **device, bool createDebug);
extern HRESULT D3D12_XBOX_CreateBackBufferTarget(ID3D12Device1 *device, int width, int height, void **resource);
extern HRESULT D3D12_XBOX_StartFrame(ID3D12Device1 *device, UINT64 *outToken);
extern HRESULT D3D12_XBOX_PresentFrame(ID3D12CommandQueue *commandQueue, UINT64 token, ID3D12Resource *renderTarget);
extern void D3D12_XBOX_GetResolution(Uint32 *width, Uint32 *height);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif
