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

#ifndef SDL_gpu_util_h_
#define SDL_gpu_util_h_

#define SDL_GPU_BLENDOP_INVALID     ((SDL_GPUBlendOp)0x7fffffff)
#define SDL_GPU_BLENDFACTOR_INVALID ((SDL_GPUBlendFactor)0x7fffffff)

static SDL_INLINE SDL_GPUBlendFactor GPU_ConvertBlendFactor(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return SDL_GPU_BLENDFACTOR_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return SDL_GPU_BLENDFACTOR_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return SDL_GPU_BLENDFACTOR_SRC_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_COLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return SDL_GPU_BLENDFACTOR_SRC_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return SDL_GPU_BLENDFACTOR_DST_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_COLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return SDL_GPU_BLENDFACTOR_DST_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_ALPHA;
    default:
        return SDL_GPU_BLENDFACTOR_INVALID;
    }
}

static SDL_INLINE SDL_GPUBlendOp GPU_ConvertBlendOperation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return SDL_GPU_BLENDOP_ADD;
    case SDL_BLENDOPERATION_SUBTRACT:
        return SDL_GPU_BLENDOP_SUBTRACT;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return SDL_GPU_BLENDOP_REVERSE_SUBTRACT;
    case SDL_BLENDOPERATION_MINIMUM:
        return SDL_GPU_BLENDOP_MIN;
    case SDL_BLENDOPERATION_MAXIMUM:
        return SDL_GPU_BLENDOP_MAX;
    default:
        return SDL_GPU_BLENDOP_INVALID;
    }
}

#endif // SDL_gpu_util_h
