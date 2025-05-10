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

#if defined(SDL_VIDEO_RENDER_D3D) || \
    defined(SDL_VIDEO_RENDER_D3D11) || \
    defined(SDL_VIDEO_RENDER_D3D12) || \
    defined(SDL_VIDEO_RENDER_GPU) || \
    defined(SDL_VIDEO_RENDER_VULKAN)

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

// Direct3D matrix math functions

typedef struct
{
    float x;
    float y;
} Float2;

typedef struct
{
    float x;
    float y;
    float z;
} Float3;

typedef struct
{
    float x;
    float y;
    float z;
    float w;
} Float4;

typedef struct
{
    union
    {
        struct
        {
            float _11, _12, _13, _14;
            float _21, _22, _23, _24;
            float _31, _32, _33, _34;
            float _41, _42, _43, _44;
        } v;
        float m[4][4];
    };
} Float4X4;

extern Float4X4 MatrixIdentity(void);
extern Float4X4 MatrixMultiply(Float4X4 M1, Float4X4 M2);
extern Float4X4 MatrixScaling(float x, float y, float z);
extern Float4X4 MatrixTranslation(float x, float y, float z);
extern Float4X4 MatrixRotationX(float r);
extern Float4X4 MatrixRotationY(float r);
extern Float4X4 MatrixRotationZ(float r);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_VIDEO_RENDER_D3D || SDL_VIDEO_RENDER_D3D11 || SDL_VIDEO_RENDER_D3D12 || SDL_VIDEO_RENDER_VULKAN
