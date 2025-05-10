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

#ifndef SDL_RENDER_VITA_GXM_TOOLS_H
#define SDL_RENDER_VITA_GXM_TOOLS_H

#include "SDL_internal.h"

#include "../SDL_sysrender.h"

#include <psp2/kernel/processmgr.h>
#include <psp2/appmgr.h>
#include <psp2/display.h>
#include <psp2/gxm.h>
#include <psp2/types.h>
#include <psp2/kernel/sysmem.h>

#include "SDL_render_vita_gxm_types.h"

void init_orthographic_matrix(float *m, float left, float right, float bottom, float top, float near, float far);

void *pool_malloc(VITA_GXM_RenderData *data, unsigned int size);
void *pool_memalign(VITA_GXM_RenderData *data, unsigned int size, unsigned int alignment);

void set_clip_rectangle(VITA_GXM_RenderData *data, int x_min, int y_min, int x_max, int y_max);
void unset_clip_rectangle(VITA_GXM_RenderData *data);

int gxm_init(SDL_Renderer *renderer);
void gxm_finish(SDL_Renderer *renderer);

gxm_texture *create_gxm_texture(VITA_GXM_RenderData *data, unsigned int w, unsigned int h, SceGxmTextureFormat format, unsigned int isRenderTarget, unsigned int *return_w, unsigned int *return_h, unsigned int *return_pitch, float *return_wscale);
void free_gxm_texture(VITA_GXM_RenderData *data, gxm_texture *texture);

void gxm_texture_set_address_mode(gxm_texture *texture, SceGxmTextureAddrMode u_mode, SceGxmTextureAddrMode v_mode);
void gxm_texture_set_filters(gxm_texture *texture, SceGxmTextureFilter min_filter, SceGxmTextureFilter mag_filter);
SceGxmTextureFormat gxm_texture_get_format(const gxm_texture *texture);

void *gxm_texture_get_datap(const gxm_texture *texture);

void gxm_minimal_init_for_common_dialog(void);
void gxm_minimal_term_for_common_dialog(void);
void gxm_init_for_common_dialog(void);
void gxm_swap_for_common_dialog(void);
void gxm_term_for_common_dialog(void);

#endif // SDL_RENDER_VITA_GXM_TOOLS_H
