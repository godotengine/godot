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

// D3D12 shader implementation

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    SHADER_SOLID,
    SHADER_RGB,
    SHADER_ADVANCED,
    NUM_SHADERS
} D3D12_Shader;

typedef enum
{
    ROOTSIG_COLOR,
    ROOTSIG_TEXTURE,
    ROOTSIG_ADVANCED,
    NUM_ROOTSIGS
} D3D12_RootSignature;

extern void D3D12_GetVertexShader(D3D12_Shader shader, D3D12_SHADER_BYTECODE *outBytecode);
extern void D3D12_GetPixelShader(D3D12_Shader shader, D3D12_SHADER_BYTECODE *outBytecode);
extern D3D12_RootSignature D3D12_GetRootSignatureType(D3D12_Shader shader);
extern void D3D12_GetRootSignatureData(D3D12_RootSignature rootSig, D3D12_SHADER_BYTECODE *outBytecode);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif
