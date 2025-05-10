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

#ifndef SDL_D3D12_H
#define SDL_D3D12_H

#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))

/* From the DirectX-Headers build system:
 * "MinGW has RPC headers which define old versions, and complain if D3D
 * headers are included before the RPC headers, since D3D headers were
 * generated with new MIDL and "require" new RPC headers."
 */
#define __REQUIRED_RPCNDR_H_VERSION__ 475

// May not be defined in winapifamily.h, can safely be ignored
#ifndef WINAPI_PARTITION_GAMES
#define WINAPI_PARTITION_GAMES 0
#endif // WINAPI_PARTITION_GAMES

#define COBJMACROS
#include "d3d12.h"
#include <dxgi1_6.h>
#include <dxgidebug.h>

#define D3D_GUID(X) &(X)

#define D3D_SAFE_RELEASE(X)      \
    if (X) {                     \
        (X)->lpVtbl->Release(X); \
        X = NULL;                \
    }

/* Some D3D12 calls are mismatched between Windows/Xbox, so we need to wrap the
 * C function ourselves :(
 */
#define D3D_CALL_RET(THIS, FUNC, ...) (THIS)->lpVtbl->FUNC((THIS), ##__VA_ARGS__)

#else // !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))

#if defined(SDL_PLATFORM_XBOXONE)
#include <d3d12_x.h>
#else // SDL_PLATFORM_XBOXSERIES
#include <d3d12_xs.h>
#endif

#define D3D_GUID(X) (X)

#define D3D_SAFE_RELEASE(X) \
    if (X) {                \
        (X)->Release();     \
        X = NULL;           \
    }

// Older versions of the Xbox GDK may not have this defined
#ifndef D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
#define D3D12_TEXTURE_DATA_PITCH_ALIGNMENT 256
#endif
#ifndef D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
#define D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE ((D3D12_RESOURCE_STATES) (0x40 | 0x80))
#endif
#ifndef D3D12_HEAP_TYPE_GPU_UPLOAD
#define D3D12_HEAP_TYPE_GPU_UPLOAD ((D3D12_HEAP_TYPE) 5)
#endif

// DXGI_PRESENT flags are removed on Xbox
#define DXGI_PRESENT_ALLOW_TEARING 0

// Xbox D3D12 does not define the COBJMACROS, so we need to define them ourselves
#include "SDL_d3d12_xbox_cmacros.h"

// They don't even define the CMACROS for ID3DBlob, come on man
#define ID3D10Blob_GetBufferPointer(blob) blob->GetBufferPointer()
#define ID3D10Blob_GetBufferSize(blob) blob->GetBufferSize()
#define ID3D10Blob_Release(blob) blob->Release()

/* Xbox's D3D12 ABI actually varies from Windows, if a function does not exist
 * in the above header then you need to use this instead :(
 */
#define D3D_CALL_RET(THIS, FUNC, RETVAL, ...) *(RETVAL) = (THIS)->FUNC(__VA_ARGS__)

#endif // !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))

#endif // SDL_D3D12_H
