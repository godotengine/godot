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

#ifndef SDL_kmsdrmvideo_h
#define SDL_kmsdrmvideo_h

#include "../SDL_sysvideo.h"

#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#include <EGL/egl.h>

#ifndef DRM_FORMAT_MOD_INVALID
#define DRM_FORMAT_MOD_INVALID 0x00ffffffffffffffULL
#endif

#ifndef DRM_MODE_FB_MODIFIERS
#define DRM_MODE_FB_MODIFIERS	2
#endif

#ifndef DRM_MODE_PAGE_FLIP_ASYNC
#define DRM_MODE_PAGE_FLIP_ASYNC    2
#endif

#ifndef DRM_MODE_OBJECT_CONNECTOR
#define DRM_MODE_OBJECT_CONNECTOR   0xc0c0c0c0
#endif

#ifndef DRM_MODE_OBJECT_CRTC
#define DRM_MODE_OBJECT_CRTC        0xcccccccc
#endif

#ifndef DRM_CAP_ASYNC_PAGE_FLIP
#define DRM_CAP_ASYNC_PAGE_FLIP 7
#endif

#ifndef DRM_CAP_CURSOR_WIDTH
#define DRM_CAP_CURSOR_WIDTH    8
#endif

#ifndef DRM_CAP_CURSOR_HEIGHT
#define DRM_CAP_CURSOR_HEIGHT   9
#endif

#ifndef GBM_FORMAT_ARGB8888
#define GBM_FORMAT_ARGB8888  ((uint32_t)('A') | ((uint32_t)('R') << 8) | ((uint32_t)('2') << 16) | ((uint32_t)('4') << 24))
#define GBM_BO_USE_CURSOR   (1 << 1)
#define GBM_BO_USE_WRITE    (1 << 3)
#define GBM_BO_USE_LINEAR   (1 << 4)
#endif

struct SDL_VideoData
{
    int devindex;     // device index that was passed on creation
    int drm_fd;       // DRM file desc
    char devpath[32]; // DRM dev path.

    struct gbm_device *gbm_dev;

    bool video_init;             // Has VideoInit succeeded?
    bool vulkan_mode;            // Are we in Vulkan mode? One VK window is enough to be.
    bool async_pageflip_support; // Does the hardware support async. pageflips?

    SDL_Window **windows;
    int max_windows;
    int num_windows;

    /* Even if we have several displays, we only have to
       open 1 FD and create 1 gbm device. */
    bool gbm_init;

};

struct SDL_DisplayModeData
{
    int mode_index;
};

struct SDL_DisplayData
{
    drmModeConnector *connector;
    drmModeCrtc *crtc;
    drmModeModeInfo mode;
    drmModeModeInfo original_mode;
    drmModeModeInfo fullscreen_mode;

    drmModeCrtc *saved_crtc; // CRTC to restore on quit
    bool saved_vrr;

    /* DRM & GBM cursor stuff lives here, not in an SDL_Cursor's internal struct,
       because setting/unsetting up these is done on window creation/destruction,
       where we may not have an SDL_Cursor at all (so no SDL_Cursor internal).
       There's only one cursor GBM BO because we only support one cursor. */
    struct gbm_bo *cursor_bo;
    int cursor_bo_drm_fd;
    uint64_t cursor_w, cursor_h;

    bool default_cursor_init;
};

struct SDL_WindowData
{
    SDL_VideoData *viddata;
    /* SDL internals expect EGL surface to be here, and in KMSDRM the GBM surface is
       what supports the EGL surface on the driver side, so all these surfaces and buffers
       are expected to be here, in the struct pointed by SDL_Window internal pointer:
       this one. So don't try to move these to dispdata!  */
    struct gbm_surface *gs;
    struct gbm_bo *bo;
    struct gbm_bo *next_bo;

    bool waiting_for_flip;
    bool double_buffer;

    EGLSurface egl_surface;
    bool egl_surface_dirty;
};

typedef struct KMSDRM_FBInfo
{
    int drm_fd;     // DRM file desc
    uint32_t fb_id; // DRM framebuffer ID
} KMSDRM_FBInfo;

// Helper functions
extern bool KMSDRM_CreateSurfaces(SDL_VideoDevice *_this, SDL_Window *window);
extern KMSDRM_FBInfo *KMSDRM_FBFromBO(SDL_VideoDevice *_this, struct gbm_bo *bo);
extern KMSDRM_FBInfo *KMSDRM_FBFromBO2(SDL_VideoDevice *_this, struct gbm_bo *bo, int w, int h);
extern bool KMSDRM_WaitPageflip(SDL_VideoDevice *_this, SDL_WindowData *windata);

/****************************************************************************/
// SDL_VideoDevice functions declaration
/****************************************************************************/

// Display and window functions
extern bool KMSDRM_VideoInit(SDL_VideoDevice *_this);
extern void KMSDRM_VideoQuit(SDL_VideoDevice *_this);
extern bool KMSDRM_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display);
extern bool KMSDRM_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
extern bool KMSDRM_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
extern void KMSDRM_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);
extern bool KMSDRM_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window);
extern void KMSDRM_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window);
extern SDL_FullscreenResult KMSDRM_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *_display, SDL_FullscreenOp fullscreen);
extern void KMSDRM_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void KMSDRM_HideWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void KMSDRM_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void KMSDRM_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void KMSDRM_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void KMSDRM_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void KMSDRM_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);

#endif // SDL_kmsdrmvideo_h
