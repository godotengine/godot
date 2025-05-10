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

/* *INDENT-OFF* */ // clang-format off

#ifndef SDL_KMSDRM_MODULE
#define SDL_KMSDRM_MODULE(modname)
#endif

#ifndef SDL_KMSDRM_SYM
#define SDL_KMSDRM_SYM(rc,fn,params)
#endif

#ifndef SDL_KMSDRM_SYM_CONST
#define SDL_KMSDRM_SYM_CONST(type, name)
#endif

#ifndef SDL_KMSDRM_SYM_OPT
#define SDL_KMSDRM_SYM_OPT(rc,fn,params)
#endif


SDL_KMSDRM_MODULE(LIBDRM)
SDL_KMSDRM_SYM(void,drmModeFreeResources,(drmModeResPtr ptr))
SDL_KMSDRM_SYM(void,drmModeFreeFB,(drmModeFBPtr ptr))
SDL_KMSDRM_SYM(void,drmModeFreeCrtc,(drmModeCrtcPtr ptr))
SDL_KMSDRM_SYM(void,drmModeFreeConnector,(drmModeConnectorPtr ptr))
SDL_KMSDRM_SYM(void,drmModeFreeEncoder,(drmModeEncoderPtr ptr))
SDL_KMSDRM_SYM(int,drmGetCap,(int fd, uint64_t capability, uint64_t *value))
SDL_KMSDRM_SYM(int,drmSetMaster,(int fd))
SDL_KMSDRM_SYM(int,drmDropMaster,(int fd))
SDL_KMSDRM_SYM(int,drmAuthMagic,(int fd, drm_magic_t magic))
SDL_KMSDRM_SYM(drmModeResPtr,drmModeGetResources,(int fd))
SDL_KMSDRM_SYM(int,drmModeAddFB,(int fd, uint32_t width, uint32_t height, uint8_t depth,
                         uint8_t bpp, uint32_t pitch, uint32_t bo_handle,
                         uint32_t *buf_id))

SDL_KMSDRM_SYM_OPT(int,drmModeAddFB2,(int fd, uint32_t width, uint32_t height,
                         uint32_t pixel_format, const uint32_t bo_handles[4],
                         const uint32_t pitches[4], const uint32_t offsets[4],
                         uint32_t *buf_id, uint32_t flags))

SDL_KMSDRM_SYM_OPT(int,drmModeAddFB2WithModifiers,(int fd, uint32_t width,
                         uint32_t height, uint32_t pixel_format, const uint32_t bo_handles[4],
                         const uint32_t pitches[4], const uint32_t offsets[4],
                         const uint64_t modifier[4], uint32_t *buf_id, uint32_t flags))

SDL_KMSDRM_SYM_OPT(const char *,drmModeGetConnectorTypeName,(uint32_t connector_type))

SDL_KMSDRM_SYM(int,drmModeRmFB,(int fd, uint32_t bufferId))
SDL_KMSDRM_SYM(drmModeFBPtr,drmModeGetFB,(int fd, uint32_t buf))
SDL_KMSDRM_SYM(drmModeCrtcPtr,drmModeGetCrtc,(int fd, uint32_t crtcId))
SDL_KMSDRM_SYM(int,drmModeSetCrtc,(int fd, uint32_t crtcId, uint32_t bufferId,
                                   uint32_t x, uint32_t y, uint32_t *connectors, int count,
                                   drmModeModeInfoPtr mode))
SDL_KMSDRM_SYM(int,drmModeSetCursor,(int fd, uint32_t crtcId, uint32_t bo_handle,
                                     uint32_t width, uint32_t height))
SDL_KMSDRM_SYM(int,drmModeSetCursor2,(int fd, uint32_t crtcId, uint32_t bo_handle,
                                      uint32_t width, uint32_t height,
                                      int32_t hot_x, int32_t hot_y))
SDL_KMSDRM_SYM(int,drmModeMoveCursor,(int fd, uint32_t crtcId, int x, int y))
SDL_KMSDRM_SYM(drmModeEncoderPtr,drmModeGetEncoder,(int fd, uint32_t encoder_id))
SDL_KMSDRM_SYM(drmModeConnectorPtr,drmModeGetConnector,(int fd, uint32_t connector_id))
SDL_KMSDRM_SYM(int,drmHandleEvent,(int fd,drmEventContextPtr evctx))
SDL_KMSDRM_SYM(int,drmModePageFlip,(int fd, uint32_t crtc_id, uint32_t fb_id,
                                    uint32_t flags, void *user_data))

// Planes stuff.
SDL_KMSDRM_SYM(int,drmSetClientCap,(int fd, uint64_t capability, uint64_t value))
SDL_KMSDRM_SYM(drmModePlaneResPtr,drmModeGetPlaneResources,(int fd))
SDL_KMSDRM_SYM(drmModePlanePtr,drmModeGetPlane,(int fd, uint32_t plane_id))
SDL_KMSDRM_SYM(drmModeObjectPropertiesPtr,drmModeObjectGetProperties,(int fd,uint32_t object_id,uint32_t object_type))
SDL_KMSDRM_SYM(int,drmModeObjectSetProperty,(int fd, uint32_t object_id,
                                             uint32_t object_type, uint32_t property_id,
                                             uint64_t value))
SDL_KMSDRM_SYM(drmModePropertyPtr,drmModeGetProperty,(int fd, uint32_t propertyId))

SDL_KMSDRM_SYM(void,drmModeFreeProperty,(drmModePropertyPtr ptr))
SDL_KMSDRM_SYM(void,drmModeFreeObjectProperties,(drmModeObjectPropertiesPtr ptr))
SDL_KMSDRM_SYM(void,drmModeFreePlane,(drmModePlanePtr ptr))
SDL_KMSDRM_SYM(void,drmModeFreePlaneResources,(drmModePlaneResPtr ptr))
SDL_KMSDRM_SYM(int,drmModeSetPlane,(int fd, uint32_t plane_id, uint32_t crtc_id,
                                    uint32_t fb_id, uint32_t flags,
                                    int32_t crtc_x, int32_t crtc_y,
                                    uint32_t crtc_w, uint32_t crtc_h,
                                    uint32_t src_x, uint32_t src_y,
                                    uint32_t src_w, uint32_t src_h))
// Planes stuff ends.

SDL_KMSDRM_MODULE(GBM)
SDL_KMSDRM_SYM(int,gbm_device_is_format_supported,(struct gbm_device *gbm,
                                                   uint32_t format, uint32_t usage))
SDL_KMSDRM_SYM(void,gbm_device_destroy,(struct gbm_device *gbm))
SDL_KMSDRM_SYM(struct gbm_device *,gbm_create_device,(int fd))
SDL_KMSDRM_SYM(unsigned int,gbm_bo_get_width,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(unsigned int,gbm_bo_get_height,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(uint32_t,gbm_bo_get_stride,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(uint32_t,gbm_bo_get_format,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(union gbm_bo_handle,gbm_bo_get_handle,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(int,gbm_bo_write,(struct gbm_bo *bo, const void *buf, size_t count))
SDL_KMSDRM_SYM(struct gbm_device *,gbm_bo_get_device,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(void,gbm_bo_set_user_data,(struct gbm_bo *bo, void *data,
                                          void (*destroy_user_data)(struct gbm_bo *, void *)))
SDL_KMSDRM_SYM(void *,gbm_bo_get_user_data,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(void,gbm_bo_destroy,(struct gbm_bo *bo))
SDL_KMSDRM_SYM(struct gbm_bo *,gbm_bo_create,(struct gbm_device *gbm,
                                              uint32_t width, uint32_t height,
                                              uint32_t format, uint32_t usage))
SDL_KMSDRM_SYM(struct gbm_surface *,gbm_surface_create,(struct gbm_device *gbm,
                                                        uint32_t width, uint32_t height,
                                                        uint32_t format, uint32_t flags))
SDL_KMSDRM_SYM(void,gbm_surface_destroy,(struct gbm_surface *surf))
SDL_KMSDRM_SYM(struct gbm_bo *,gbm_surface_lock_front_buffer,(struct gbm_surface *surf))
SDL_KMSDRM_SYM(void,gbm_surface_release_buffer,(struct gbm_surface *surf, struct gbm_bo *bo))

SDL_KMSDRM_SYM_OPT(uint64_t,gbm_bo_get_modifier,(struct gbm_bo *bo))
SDL_KMSDRM_SYM_OPT(int,gbm_bo_get_plane_count,(struct gbm_bo *bo))
SDL_KMSDRM_SYM_OPT(uint32_t,gbm_bo_get_offset,(struct gbm_bo *bo, int plane))
SDL_KMSDRM_SYM_OPT(uint32_t,gbm_bo_get_stride_for_plane,(struct gbm_bo *bo, int plane))
SDL_KMSDRM_SYM_OPT(union gbm_bo_handle,gbm_bo_get_handle_for_plane,(struct gbm_bo *bo, int plane))

#undef SDL_KMSDRM_MODULE
#undef SDL_KMSDRM_SYM
#undef SDL_KMSDRM_SYM_CONST
#undef SDL_KMSDRM_SYM_OPT

/* *INDENT-ON* */ // clang-format on
