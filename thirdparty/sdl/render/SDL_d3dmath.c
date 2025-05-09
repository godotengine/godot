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
    defined(SDL_VIDEO_RENDER_VULKAN)

#include "SDL_d3dmath.h"

// Direct3D matrix math functions

Float4X4 MatrixIdentity(void)
{
    Float4X4 m;
    SDL_zero(m);
    m.v._11 = 1.0f;
    m.v._22 = 1.0f;
    m.v._33 = 1.0f;
    m.v._44 = 1.0f;
    return m;
}

Float4X4 MatrixMultiply(Float4X4 M1, Float4X4 M2)
{
    Float4X4 m;
    m.v._11 = M1.v._11 * M2.v._11 + M1.v._12 * M2.v._21 + M1.v._13 * M2.v._31 + M1.v._14 * M2.v._41;
    m.v._12 = M1.v._11 * M2.v._12 + M1.v._12 * M2.v._22 + M1.v._13 * M2.v._32 + M1.v._14 * M2.v._42;
    m.v._13 = M1.v._11 * M2.v._13 + M1.v._12 * M2.v._23 + M1.v._13 * M2.v._33 + M1.v._14 * M2.v._43;
    m.v._14 = M1.v._11 * M2.v._14 + M1.v._12 * M2.v._24 + M1.v._13 * M2.v._34 + M1.v._14 * M2.v._44;
    m.v._21 = M1.v._21 * M2.v._11 + M1.v._22 * M2.v._21 + M1.v._23 * M2.v._31 + M1.v._24 * M2.v._41;
    m.v._22 = M1.v._21 * M2.v._12 + M1.v._22 * M2.v._22 + M1.v._23 * M2.v._32 + M1.v._24 * M2.v._42;
    m.v._23 = M1.v._21 * M2.v._13 + M1.v._22 * M2.v._23 + M1.v._23 * M2.v._33 + M1.v._24 * M2.v._43;
    m.v._24 = M1.v._21 * M2.v._14 + M1.v._22 * M2.v._24 + M1.v._23 * M2.v._34 + M1.v._24 * M2.v._44;
    m.v._31 = M1.v._31 * M2.v._11 + M1.v._32 * M2.v._21 + M1.v._33 * M2.v._31 + M1.v._34 * M2.v._41;
    m.v._32 = M1.v._31 * M2.v._12 + M1.v._32 * M2.v._22 + M1.v._33 * M2.v._32 + M1.v._34 * M2.v._42;
    m.v._33 = M1.v._31 * M2.v._13 + M1.v._32 * M2.v._23 + M1.v._33 * M2.v._33 + M1.v._34 * M2.v._43;
    m.v._34 = M1.v._31 * M2.v._14 + M1.v._32 * M2.v._24 + M1.v._33 * M2.v._34 + M1.v._34 * M2.v._44;
    m.v._41 = M1.v._41 * M2.v._11 + M1.v._42 * M2.v._21 + M1.v._43 * M2.v._31 + M1.v._44 * M2.v._41;
    m.v._42 = M1.v._41 * M2.v._12 + M1.v._42 * M2.v._22 + M1.v._43 * M2.v._32 + M1.v._44 * M2.v._42;
    m.v._43 = M1.v._41 * M2.v._13 + M1.v._42 * M2.v._23 + M1.v._43 * M2.v._33 + M1.v._44 * M2.v._43;
    m.v._44 = M1.v._41 * M2.v._14 + M1.v._42 * M2.v._24 + M1.v._43 * M2.v._34 + M1.v._44 * M2.v._44;
    return m;
}

Float4X4 MatrixScaling(float x, float y, float z)
{
    Float4X4 m;
    SDL_zero(m);
    m.v._11 = x;
    m.v._22 = y;
    m.v._33 = z;
    m.v._44 = 1.0f;
    return m;
}

Float4X4 MatrixTranslation(float x, float y, float z)
{
    Float4X4 m;
    SDL_zero(m);
    m.v._11 = 1.0f;
    m.v._22 = 1.0f;
    m.v._33 = 1.0f;
    m.v._44 = 1.0f;
    m.v._41 = x;
    m.v._42 = y;
    m.v._43 = z;
    return m;
}

Float4X4 MatrixRotationX(float r)
{
    float sinR = SDL_sinf(r);
    float cosR = SDL_cosf(r);
    Float4X4 m;
    SDL_zero(m);
    m.v._11 = 1.0f;
    m.v._22 = cosR;
    m.v._23 = sinR;
    m.v._32 = -sinR;
    m.v._33 = cosR;
    m.v._44 = 1.0f;
    return m;
}

Float4X4 MatrixRotationY(float r)
{
    float sinR = SDL_sinf(r);
    float cosR = SDL_cosf(r);
    Float4X4 m;
    SDL_zero(m);
    m.v._11 = cosR;
    m.v._13 = -sinR;
    m.v._22 = 1.0f;
    m.v._31 = sinR;
    m.v._33 = cosR;
    m.v._44 = 1.0f;
    return m;
}

Float4X4 MatrixRotationZ(float r)
{
    float sinR = SDL_sinf(r);
    float cosR = SDL_cosf(r);
    Float4X4 m;
    SDL_zero(m);
    m.v._11 = cosR;
    m.v._12 = sinR;
    m.v._21 = -sinR;
    m.v._22 = cosR;
    m.v._33 = 1.0f;
    m.v._44 = 1.0f;
    return m;
}

#endif // SDL_VIDEO_RENDER_D3D || SDL_VIDEO_RENDER_D3D11 || SDL_VIDEO_RENDER_D3D12 || SDL_VIDEO_RENDER_VULKAN
