#ifndef __eglext_h_
#define __eglext_h_ 1

#ifdef __cplusplus
extern "C" {
#endif

/*
** Copyright (c) 2013-2014 The Khronos Group Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and/or associated documentation files (the
** "Materials"), to deal in the Materials without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Materials, and to
** permit persons to whom the Materials are furnished to do so, subject to
** the following conditions:
**
** The above copyright notice and this permission notice shall be included
** in all copies or substantial portions of the Materials.
**
** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
** MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
** IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
** CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
** TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
** MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
*/
/*
** This header is generated from the Khronos OpenGL / OpenGL ES XML
** API Registry. The current version of the Registry, generator scripts
** used to make the header, and the header can be found at
**   http://www.opengl.org/registry/
**
** Khronos $Revision: 27018 $ on $Date: 2014-06-10 08:06:12 -0700 (Tue, 10 Jun 2014) $
*/

#include <EGL/eglplatform.h>

#define EGL_EGLEXT_VERSION 20140610

/* Generated C header for:
 * API: egl
 * Versions considered: .*
 * Versions emitted: _nomatch_^
 * Default extensions included: egl
 * Additional extensions included: _nomatch_^
 * Extensions removed: _nomatch_^
 */

#ifndef EGL_KHR_cl_event
#define EGL_KHR_cl_event 1
#define EGL_CL_EVENT_HANDLE_KHR           0x309C
#define EGL_SYNC_CL_EVENT_KHR             0x30FE
#define EGL_SYNC_CL_EVENT_COMPLETE_KHR    0x30FF
#endif /* EGL_KHR_cl_event */

#ifndef EGL_KHR_cl_event2
#define EGL_KHR_cl_event2 1
typedef void *EGLSyncKHR;
typedef intptr_t EGLAttribKHR;
typedef EGLSyncKHR (EGLAPIENTRYP PFNEGLCREATESYNC64KHRPROC) (EGLDisplay dpy, EGLenum type, const EGLAttribKHR *attrib_list);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLSyncKHR EGLAPIENTRY eglCreateSync64KHR (EGLDisplay dpy, EGLenum type, const EGLAttribKHR *attrib_list);
#endif
#endif /* EGL_KHR_cl_event2 */

#ifndef EGL_KHR_client_get_all_proc_addresses
#define EGL_KHR_client_get_all_proc_addresses 1
#endif /* EGL_KHR_client_get_all_proc_addresses */

#ifndef EGL_KHR_config_attribs
#define EGL_KHR_config_attribs 1
#define EGL_CONFORMANT_KHR                0x3042
#define EGL_VG_COLORSPACE_LINEAR_BIT_KHR  0x0020
#define EGL_VG_ALPHA_FORMAT_PRE_BIT_KHR   0x0040
#endif /* EGL_KHR_config_attribs */

#ifndef EGL_KHR_create_context
#define EGL_KHR_create_context 1
#define EGL_CONTEXT_MAJOR_VERSION_KHR     0x3098
#define EGL_CONTEXT_MINOR_VERSION_KHR     0x30FB
#define EGL_CONTEXT_FLAGS_KHR             0x30FC
#define EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR 0x30FD
#define EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_KHR 0x31BD
#define EGL_NO_RESET_NOTIFICATION_KHR     0x31BE
#define EGL_LOSE_CONTEXT_ON_RESET_KHR     0x31BF
#define EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR  0x00000001
#define EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE_BIT_KHR 0x00000002
#define EGL_CONTEXT_OPENGL_ROBUST_ACCESS_BIT_KHR 0x00000004
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR 0x00000001
#define EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT_KHR 0x00000002
#define EGL_OPENGL_ES3_BIT_KHR            0x00000040
#endif /* EGL_KHR_create_context */

#ifndef EGL_KHR_fence_sync
#define EGL_KHR_fence_sync 1
#ifdef KHRONOS_SUPPORT_INT64
#define EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR 0x30F0
#define EGL_SYNC_CONDITION_KHR            0x30F8
#define EGL_SYNC_FENCE_KHR                0x30F9
#endif /* KHRONOS_SUPPORT_INT64 */
#endif /* EGL_KHR_fence_sync */

#ifndef EGL_KHR_get_all_proc_addresses
#define EGL_KHR_get_all_proc_addresses 1
#endif /* EGL_KHR_get_all_proc_addresses */

#ifndef EGL_KHR_gl_colorspace
#define EGL_KHR_gl_colorspace 1
#define EGL_GL_COLORSPACE_KHR             0x309D
#define EGL_GL_COLORSPACE_SRGB_KHR        0x3089
#define EGL_GL_COLORSPACE_LINEAR_KHR      0x308A
#endif /* EGL_KHR_gl_colorspace */

#ifndef EGL_KHR_gl_renderbuffer_image
#define EGL_KHR_gl_renderbuffer_image 1
#define EGL_GL_RENDERBUFFER_KHR           0x30B9
#endif /* EGL_KHR_gl_renderbuffer_image */

#ifndef EGL_KHR_gl_texture_2D_image
#define EGL_KHR_gl_texture_2D_image 1
#define EGL_GL_TEXTURE_2D_KHR             0x30B1
#define EGL_GL_TEXTURE_LEVEL_KHR          0x30BC
#endif /* EGL_KHR_gl_texture_2D_image */

#ifndef EGL_KHR_gl_texture_3D_image
#define EGL_KHR_gl_texture_3D_image 1
#define EGL_GL_TEXTURE_3D_KHR             0x30B2
#define EGL_GL_TEXTURE_ZOFFSET_KHR        0x30BD
#endif /* EGL_KHR_gl_texture_3D_image */

#ifndef EGL_KHR_gl_texture_cubemap_image
#define EGL_KHR_gl_texture_cubemap_image 1
#define EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR 0x30B3
#define EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_X_KHR 0x30B4
#define EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Y_KHR 0x30B5
#define EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_KHR 0x30B6
#define EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Z_KHR 0x30B7
#define EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_KHR 0x30B8
#endif /* EGL_KHR_gl_texture_cubemap_image */

#ifndef EGL_KHR_image
#define EGL_KHR_image 1
typedef void *EGLImageKHR;
#define EGL_NATIVE_PIXMAP_KHR             0x30B0
#define EGL_NO_IMAGE_KHR                  ((EGLImageKHR)0)
typedef EGLImageKHR (EGLAPIENTRYP PFNEGLCREATEIMAGEKHRPROC) (EGLDisplay dpy, EGLContext ctx, EGLenum target, EGLClientBuffer buffer, const EGLint *attrib_list);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLDESTROYIMAGEKHRPROC) (EGLDisplay dpy, EGLImageKHR image);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLImageKHR EGLAPIENTRY eglCreateImageKHR (EGLDisplay dpy, EGLContext ctx, EGLenum target, EGLClientBuffer buffer, const EGLint *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglDestroyImageKHR (EGLDisplay dpy, EGLImageKHR image);
#endif
#endif /* EGL_KHR_image */

#ifndef EGL_KHR_image_base
#define EGL_KHR_image_base 1
#define EGL_IMAGE_PRESERVED_KHR           0x30D2
#endif /* EGL_KHR_image_base */

#ifndef EGL_KHR_image_pixmap
#define EGL_KHR_image_pixmap 1
#endif /* EGL_KHR_image_pixmap */

#ifndef EGL_KHR_lock_surface
#define EGL_KHR_lock_surface 1
#define EGL_READ_SURFACE_BIT_KHR          0x0001
#define EGL_WRITE_SURFACE_BIT_KHR         0x0002
#define EGL_LOCK_SURFACE_BIT_KHR          0x0080
#define EGL_OPTIMAL_FORMAT_BIT_KHR        0x0100
#define EGL_MATCH_FORMAT_KHR              0x3043
#define EGL_FORMAT_RGB_565_EXACT_KHR      0x30C0
#define EGL_FORMAT_RGB_565_KHR            0x30C1
#define EGL_FORMAT_RGBA_8888_EXACT_KHR    0x30C2
#define EGL_FORMAT_RGBA_8888_KHR          0x30C3
#define EGL_MAP_PRESERVE_PIXELS_KHR       0x30C4
#define EGL_LOCK_USAGE_HINT_KHR           0x30C5
#define EGL_BITMAP_POINTER_KHR            0x30C6
#define EGL_BITMAP_PITCH_KHR              0x30C7
#define EGL_BITMAP_ORIGIN_KHR             0x30C8
#define EGL_BITMAP_PIXEL_RED_OFFSET_KHR   0x30C9
#define EGL_BITMAP_PIXEL_GREEN_OFFSET_KHR 0x30CA
#define EGL_BITMAP_PIXEL_BLUE_OFFSET_KHR  0x30CB
#define EGL_BITMAP_PIXEL_ALPHA_OFFSET_KHR 0x30CC
#define EGL_BITMAP_PIXEL_LUMINANCE_OFFSET_KHR 0x30CD
#define EGL_LOWER_LEFT_KHR                0x30CE
#define EGL_UPPER_LEFT_KHR                0x30CF
typedef EGLBoolean (EGLAPIENTRYP PFNEGLLOCKSURFACEKHRPROC) (EGLDisplay dpy, EGLSurface surface, const EGLint *attrib_list);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLUNLOCKSURFACEKHRPROC) (EGLDisplay dpy, EGLSurface surface);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglLockSurfaceKHR (EGLDisplay dpy, EGLSurface surface, const EGLint *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglUnlockSurfaceKHR (EGLDisplay dpy, EGLSurface surface);
#endif
#endif /* EGL_KHR_lock_surface */

#ifndef EGL_KHR_lock_surface2
#define EGL_KHR_lock_surface2 1
#define EGL_BITMAP_PIXEL_SIZE_KHR         0x3110
#endif /* EGL_KHR_lock_surface2 */

#ifndef EGL_KHR_lock_surface3
#define EGL_KHR_lock_surface3 1
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSURFACE64KHRPROC) (EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLAttribKHR *value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglQuerySurface64KHR (EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLAttribKHR *value);
#endif
#endif /* EGL_KHR_lock_surface3 */

#ifndef EGL_KHR_platform_android
#define EGL_KHR_platform_android 1
#define EGL_PLATFORM_ANDROID_KHR          0x3141
#endif /* EGL_KHR_platform_android */

#ifndef EGL_KHR_platform_gbm
#define EGL_KHR_platform_gbm 1
#define EGL_PLATFORM_GBM_KHR              0x31D7
#endif /* EGL_KHR_platform_gbm */

#ifndef EGL_KHR_platform_wayland
#define EGL_KHR_platform_wayland 1
#define EGL_PLATFORM_WAYLAND_KHR          0x31D8
#endif /* EGL_KHR_platform_wayland */

#ifndef EGL_KHR_platform_x11
#define EGL_KHR_platform_x11 1
#define EGL_PLATFORM_X11_KHR              0x31D5
#define EGL_PLATFORM_X11_SCREEN_KHR       0x31D6
#endif /* EGL_KHR_platform_x11 */

#ifndef EGL_KHR_reusable_sync
#define EGL_KHR_reusable_sync 1
typedef khronos_utime_nanoseconds_t EGLTimeKHR;
#ifdef KHRONOS_SUPPORT_INT64
#define EGL_SYNC_STATUS_KHR               0x30F1
#define EGL_SIGNALED_KHR                  0x30F2
#define EGL_UNSIGNALED_KHR                0x30F3
#define EGL_TIMEOUT_EXPIRED_KHR           0x30F5
#define EGL_CONDITION_SATISFIED_KHR       0x30F6
#define EGL_SYNC_TYPE_KHR                 0x30F7
#define EGL_SYNC_REUSABLE_KHR             0x30FA
#define EGL_SYNC_FLUSH_COMMANDS_BIT_KHR   0x0001
#define EGL_FOREVER_KHR                   0xFFFFFFFFFFFFFFFFull
#define EGL_NO_SYNC_KHR                   ((EGLSyncKHR)0)
typedef EGLSyncKHR (EGLAPIENTRYP PFNEGLCREATESYNCKHRPROC) (EGLDisplay dpy, EGLenum type, const EGLint *attrib_list);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLDESTROYSYNCKHRPROC) (EGLDisplay dpy, EGLSyncKHR sync);
typedef EGLint (EGLAPIENTRYP PFNEGLCLIENTWAITSYNCKHRPROC) (EGLDisplay dpy, EGLSyncKHR sync, EGLint flags, EGLTimeKHR timeout);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSIGNALSYNCKHRPROC) (EGLDisplay dpy, EGLSyncKHR sync, EGLenum mode);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLGETSYNCATTRIBKHRPROC) (EGLDisplay dpy, EGLSyncKHR sync, EGLint attribute, EGLint *value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLSyncKHR EGLAPIENTRY eglCreateSyncKHR (EGLDisplay dpy, EGLenum type, const EGLint *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglDestroySyncKHR (EGLDisplay dpy, EGLSyncKHR sync);
EGLAPI EGLint EGLAPIENTRY eglClientWaitSyncKHR (EGLDisplay dpy, EGLSyncKHR sync, EGLint flags, EGLTimeKHR timeout);
EGLAPI EGLBoolean EGLAPIENTRY eglSignalSyncKHR (EGLDisplay dpy, EGLSyncKHR sync, EGLenum mode);
EGLAPI EGLBoolean EGLAPIENTRY eglGetSyncAttribKHR (EGLDisplay dpy, EGLSyncKHR sync, EGLint attribute, EGLint *value);
#endif
#endif /* KHRONOS_SUPPORT_INT64 */
#endif /* EGL_KHR_reusable_sync */

#ifndef EGL_KHR_stream
#define EGL_KHR_stream 1
typedef void *EGLStreamKHR;
typedef khronos_uint64_t EGLuint64KHR;
#ifdef KHRONOS_SUPPORT_INT64
#define EGL_NO_STREAM_KHR                 ((EGLStreamKHR)0)
#define EGL_CONSUMER_LATENCY_USEC_KHR     0x3210
#define EGL_PRODUCER_FRAME_KHR            0x3212
#define EGL_CONSUMER_FRAME_KHR            0x3213
#define EGL_STREAM_STATE_KHR              0x3214
#define EGL_STREAM_STATE_CREATED_KHR      0x3215
#define EGL_STREAM_STATE_CONNECTING_KHR   0x3216
#define EGL_STREAM_STATE_EMPTY_KHR        0x3217
#define EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR 0x3218
#define EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR 0x3219
#define EGL_STREAM_STATE_DISCONNECTED_KHR 0x321A
#define EGL_BAD_STREAM_KHR                0x321B
#define EGL_BAD_STATE_KHR                 0x321C
typedef EGLStreamKHR (EGLAPIENTRYP PFNEGLCREATESTREAMKHRPROC) (EGLDisplay dpy, const EGLint *attrib_list);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLDESTROYSTREAMKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSTREAMATTRIBKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLint value);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSTREAMKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLint *value);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSTREAMU64KHRPROC) (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLuint64KHR *value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLStreamKHR EGLAPIENTRY eglCreateStreamKHR (EGLDisplay dpy, const EGLint *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglDestroyStreamKHR (EGLDisplay dpy, EGLStreamKHR stream);
EGLAPI EGLBoolean EGLAPIENTRY eglStreamAttribKHR (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLint value);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryStreamKHR (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLint *value);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryStreamu64KHR (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLuint64KHR *value);
#endif
#endif /* KHRONOS_SUPPORT_INT64 */
#endif /* EGL_KHR_stream */

#ifndef EGL_KHR_stream_consumer_gltexture
#define EGL_KHR_stream_consumer_gltexture 1
#ifdef EGL_KHR_stream
#define EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR 0x321E
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSTREAMCONSUMERACQUIREKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSTREAMCONSUMERRELEASEKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglStreamConsumerGLTextureExternalKHR (EGLDisplay dpy, EGLStreamKHR stream);
EGLAPI EGLBoolean EGLAPIENTRY eglStreamConsumerAcquireKHR (EGLDisplay dpy, EGLStreamKHR stream);
EGLAPI EGLBoolean EGLAPIENTRY eglStreamConsumerReleaseKHR (EGLDisplay dpy, EGLStreamKHR stream);
#endif
#endif /* EGL_KHR_stream */
#endif /* EGL_KHR_stream_consumer_gltexture */

#ifndef EGL_KHR_stream_cross_process_fd
#define EGL_KHR_stream_cross_process_fd 1
typedef int EGLNativeFileDescriptorKHR;
#ifdef EGL_KHR_stream
#define EGL_NO_FILE_DESCRIPTOR_KHR        ((EGLNativeFileDescriptorKHR)(-1))
typedef EGLNativeFileDescriptorKHR (EGLAPIENTRYP PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream);
typedef EGLStreamKHR (EGLAPIENTRYP PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC) (EGLDisplay dpy, EGLNativeFileDescriptorKHR file_descriptor);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLNativeFileDescriptorKHR EGLAPIENTRY eglGetStreamFileDescriptorKHR (EGLDisplay dpy, EGLStreamKHR stream);
EGLAPI EGLStreamKHR EGLAPIENTRY eglCreateStreamFromFileDescriptorKHR (EGLDisplay dpy, EGLNativeFileDescriptorKHR file_descriptor);
#endif
#endif /* EGL_KHR_stream */
#endif /* EGL_KHR_stream_cross_process_fd */

#ifndef EGL_KHR_stream_fifo
#define EGL_KHR_stream_fifo 1
#ifdef EGL_KHR_stream
#define EGL_STREAM_FIFO_LENGTH_KHR        0x31FC
#define EGL_STREAM_TIME_NOW_KHR           0x31FD
#define EGL_STREAM_TIME_CONSUMER_KHR      0x31FE
#define EGL_STREAM_TIME_PRODUCER_KHR      0x31FF
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSTREAMTIMEKHRPROC) (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLTimeKHR *value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglQueryStreamTimeKHR (EGLDisplay dpy, EGLStreamKHR stream, EGLenum attribute, EGLTimeKHR *value);
#endif
#endif /* EGL_KHR_stream */
#endif /* EGL_KHR_stream_fifo */

#ifndef EGL_KHR_stream_producer_aldatalocator
#define EGL_KHR_stream_producer_aldatalocator 1
#ifdef EGL_KHR_stream
#endif /* EGL_KHR_stream */
#endif /* EGL_KHR_stream_producer_aldatalocator */

#ifndef EGL_KHR_stream_producer_eglsurface
#define EGL_KHR_stream_producer_eglsurface 1
#ifdef EGL_KHR_stream
#define EGL_STREAM_BIT_KHR                0x0800
typedef EGLSurface (EGLAPIENTRYP PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC) (EGLDisplay dpy, EGLConfig config, EGLStreamKHR stream, const EGLint *attrib_list);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLSurface EGLAPIENTRY eglCreateStreamProducerSurfaceKHR (EGLDisplay dpy, EGLConfig config, EGLStreamKHR stream, const EGLint *attrib_list);
#endif
#endif /* EGL_KHR_stream */
#endif /* EGL_KHR_stream_producer_eglsurface */

#ifndef EGL_KHR_surfaceless_context
#define EGL_KHR_surfaceless_context 1
#endif /* EGL_KHR_surfaceless_context */

#ifndef EGL_KHR_vg_parent_image
#define EGL_KHR_vg_parent_image 1
#define EGL_VG_PARENT_IMAGE_KHR           0x30BA
#endif /* EGL_KHR_vg_parent_image */

#ifndef EGL_KHR_wait_sync
#define EGL_KHR_wait_sync 1
typedef EGLint (EGLAPIENTRYP PFNEGLWAITSYNCKHRPROC) (EGLDisplay dpy, EGLSyncKHR sync, EGLint flags);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLint EGLAPIENTRY eglWaitSyncKHR (EGLDisplay dpy, EGLSyncKHR sync, EGLint flags);
#endif
#endif /* EGL_KHR_wait_sync */

#ifndef EGL_ANDROID_blob_cache
#define EGL_ANDROID_blob_cache 1
typedef khronos_ssize_t EGLsizeiANDROID;
typedef void (*EGLSetBlobFuncANDROID) (const void *key, EGLsizeiANDROID keySize, const void *value, EGLsizeiANDROID valueSize);
typedef EGLsizeiANDROID (*EGLGetBlobFuncANDROID) (const void *key, EGLsizeiANDROID keySize, void *value, EGLsizeiANDROID valueSize);
typedef void (EGLAPIENTRYP PFNEGLSETBLOBCACHEFUNCSANDROIDPROC) (EGLDisplay dpy, EGLSetBlobFuncANDROID set, EGLGetBlobFuncANDROID get);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI void EGLAPIENTRY eglSetBlobCacheFuncsANDROID (EGLDisplay dpy, EGLSetBlobFuncANDROID set, EGLGetBlobFuncANDROID get);
#endif
#endif /* EGL_ANDROID_blob_cache */

#ifndef EGL_ANDROID_framebuffer_target
#define EGL_ANDROID_framebuffer_target 1
#define EGL_FRAMEBUFFER_TARGET_ANDROID    0x3147
#endif /* EGL_ANDROID_framebuffer_target */

#ifndef EGL_ANDROID_image_native_buffer
#define EGL_ANDROID_image_native_buffer 1
#define EGL_NATIVE_BUFFER_ANDROID         0x3140
#endif /* EGL_ANDROID_image_native_buffer */

#ifndef EGL_ANDROID_native_fence_sync
#define EGL_ANDROID_native_fence_sync 1
#define EGL_SYNC_NATIVE_FENCE_ANDROID     0x3144
#define EGL_SYNC_NATIVE_FENCE_FD_ANDROID  0x3145
#define EGL_SYNC_NATIVE_FENCE_SIGNALED_ANDROID 0x3146
#define EGL_NO_NATIVE_FENCE_FD_ANDROID    -1
typedef EGLint (EGLAPIENTRYP PFNEGLDUPNATIVEFENCEFDANDROIDPROC) (EGLDisplay dpy, EGLSyncKHR sync);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLint EGLAPIENTRY eglDupNativeFenceFDANDROID (EGLDisplay dpy, EGLSyncKHR sync);
#endif
#endif /* EGL_ANDROID_native_fence_sync */

#ifndef EGL_ANDROID_recordable
#define EGL_ANDROID_recordable 1
#define EGL_RECORDABLE_ANDROID            0x3142
#endif /* EGL_ANDROID_recordable */

#ifndef EGL_ANGLE_d3d_share_handle_client_buffer
#define EGL_ANGLE_d3d_share_handle_client_buffer 1
#define EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE 0x3200
#endif /* EGL_ANGLE_d3d_share_handle_client_buffer */

#ifndef EGL_ANGLE_window_fixed_size
#define EGL_ANGLE_window_fixed_size 1
#define EGL_FIXED_SIZE_ANGLE        0x3201
#endif /* EGL_ANGLE_window_fixed_size */

#ifndef EGL_ANGLE_query_surface_pointer
#define EGL_ANGLE_query_surface_pointer 1
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSURFACEPOINTERANGLEPROC) (EGLDisplay dpy, EGLSurface surface, EGLint attribute, void **value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglQuerySurfacePointerANGLE (EGLDisplay dpy, EGLSurface surface, EGLint attribute, void **value);
#endif
#endif /* EGL_ANGLE_query_surface_pointer */

#ifndef EGL_ANGLE_software_display
#define EGL_ANGLE_software_display 1
#define EGL_SOFTWARE_DISPLAY_ANGLE ((EGLNativeDisplayType)-1)
#endif /* EGL_ANGLE_software_display */

#ifndef EGL_ANGLE_direct3d_display
#define EGL_ANGLE_direct3d_display 1
#define EGL_D3D11_ELSE_D3D9_DISPLAY_ANGLE ((EGLNativeDisplayType)-2)
#define EGL_D3D11_ONLY_DISPLAY_ANGLE ((EGLNativeDisplayType)-3)
#endif /* EGL_ANGLE_direct3d_display */

#ifndef EGL_ANGLE_surface_d3d_texture_2d_share_handle
#define EGL_ANGLE_surface_d3d_texture_2d_share_handle 1
#endif /* EGL_ANGLE_surface_d3d_texture_2d_share_handle */

#ifndef EGL_ANGLE_surface_d3d_render_to_back_buffer
#define EGL_ANGLE_surface_d3d_render_to_back_buffer 1
#define EGL_ANGLE_DISPLAY_ALLOW_RENDER_TO_BACK_BUFFER 0x320B
#define EGL_ANGLE_SURFACE_RENDER_TO_BACK_BUFFER 0x320C
#endif /* EGL_ANGLE_surface_d3d_render_to_back_buffer */

#ifndef EGL_ANGLE_platform_angle
#define EGL_ANGLE_platform_angle 1
#define EGL_PLATFORM_ANGLE_ANGLE          0x3201
#define EGL_PLATFORM_ANGLE_TYPE_ANGLE     0x3202
#define EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE 0x3203
#define EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE 0x3204
#define EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE 0x3205
#endif /* EGL_ANGLE_platform_angle */

#ifndef EGL_ANGLE_platform_angle_d3d
#define EGL_ANGLE_platform_angle_d3d 1
#define EGL_PLATFORM_ANGLE_TYPE_D3D9_ANGLE 0x3206
#define EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE 0x3207
#define EGL_PLATFORM_ANGLE_USE_WARP_ANGLE 0x3208
#endif /* EGL_ANGLE_platform_angle_d3d */

#ifndef EGL_ANGLE_platform_angle_opengl
#define EGL_ANGLE_platform_angle_opengl 1
#define EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE 0x3209
#define EGL_PLATFORM_ANGLE_TYPE_OPENGLES_ANGLE 0x320A
#endif /* EGL_ANGLE_platform_angle_opengl */

#ifndef EGL_ARM_pixmap_multisample_discard
#define EGL_ARM_pixmap_multisample_discard 1
#define EGL_DISCARD_SAMPLES_ARM           0x3286
#endif /* EGL_ARM_pixmap_multisample_discard */

#ifndef EGL_EXT_buffer_age
#define EGL_EXT_buffer_age 1
#define EGL_BUFFER_AGE_EXT                0x313D
#endif /* EGL_EXT_buffer_age */

#ifndef EGL_EXT_client_extensions
#define EGL_EXT_client_extensions 1
#endif /* EGL_EXT_client_extensions */

#ifndef EGL_EXT_create_context_robustness
#define EGL_EXT_create_context_robustness 1
#define EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT 0x30BF
#define EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT 0x3138
#define EGL_NO_RESET_NOTIFICATION_EXT     0x31BE
#define EGL_LOSE_CONTEXT_ON_RESET_EXT     0x31BF
#endif /* EGL_EXT_create_context_robustness */

#ifndef EGL_EXT_device_base
#define EGL_EXT_device_base 1
typedef void *EGLDeviceEXT;
#define EGL_NO_DEVICE_EXT                 ((EGLDeviceEXT)(0))
#define EGL_BAD_DEVICE_EXT                0x322B
#define EGL_DEVICE_EXT                    0x322C
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYDEVICEATTRIBEXTPROC) (EGLDeviceEXT device, EGLint attribute, EGLAttrib *value);
typedef const char *(EGLAPIENTRYP PFNEGLQUERYDEVICESTRINGEXTPROC) (EGLDeviceEXT device, EGLint name);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYDEVICESEXTPROC) (EGLint max_devices, EGLDeviceEXT *devices, EGLint *num_devices);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYDISPLAYATTRIBEXTPROC) (EGLDisplay dpy, EGLint attribute, EGLAttrib *value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDeviceAttribEXT (EGLDeviceEXT device, EGLint attribute, EGLAttrib *value);
EGLAPI const char *EGLAPIENTRY eglQueryDeviceStringEXT (EGLDeviceEXT device, EGLint name);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDevicesEXT (EGLint max_devices, EGLDeviceEXT *devices, EGLint *num_devices);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDisplayAttribEXT (EGLDisplay dpy, EGLint attribute, EGLAttrib *value);
#endif
#endif /* EGL_EXT_device_base */

#ifndef EGL_EXT_image_dma_buf_import
#define EGL_EXT_image_dma_buf_import 1
#define EGL_LINUX_DMA_BUF_EXT             0x3270
#define EGL_LINUX_DRM_FOURCC_EXT          0x3271
#define EGL_DMA_BUF_PLANE0_FD_EXT         0x3272
#define EGL_DMA_BUF_PLANE0_OFFSET_EXT     0x3273
#define EGL_DMA_BUF_PLANE0_PITCH_EXT      0x3274
#define EGL_DMA_BUF_PLANE1_FD_EXT         0x3275
#define EGL_DMA_BUF_PLANE1_OFFSET_EXT     0x3276
#define EGL_DMA_BUF_PLANE1_PITCH_EXT      0x3277
#define EGL_DMA_BUF_PLANE2_FD_EXT         0x3278
#define EGL_DMA_BUF_PLANE2_OFFSET_EXT     0x3279
#define EGL_DMA_BUF_PLANE2_PITCH_EXT      0x327A
#define EGL_YUV_COLOR_SPACE_HINT_EXT      0x327B
#define EGL_SAMPLE_RANGE_HINT_EXT         0x327C
#define EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT 0x327D
#define EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT 0x327E
#define EGL_ITU_REC601_EXT                0x327F
#define EGL_ITU_REC709_EXT                0x3280
#define EGL_ITU_REC2020_EXT               0x3281
#define EGL_YUV_FULL_RANGE_EXT            0x3282
#define EGL_YUV_NARROW_RANGE_EXT          0x3283
#define EGL_YUV_CHROMA_SITING_0_EXT       0x3284
#define EGL_YUV_CHROMA_SITING_0_5_EXT     0x3285
#endif /* EGL_EXT_image_dma_buf_import */

#ifndef EGL_EXT_multiview_window
#define EGL_EXT_multiview_window 1
#define EGL_MULTIVIEW_VIEW_COUNT_EXT      0x3134
#endif /* EGL_EXT_multiview_window */

#ifndef EGL_EXT_platform_base
#define EGL_EXT_platform_base 1
typedef EGLDisplay (EGLAPIENTRYP PFNEGLGETPLATFORMDISPLAYEXTPROC) (EGLenum platform, void *native_display, const EGLint *attrib_list);
typedef EGLSurface (EGLAPIENTRYP PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC) (EGLDisplay dpy, EGLConfig config, void *native_window, const EGLint *attrib_list);
typedef EGLSurface (EGLAPIENTRYP PFNEGLCREATEPLATFORMPIXMAPSURFACEEXTPROC) (EGLDisplay dpy, EGLConfig config, void *native_pixmap, const EGLint *attrib_list);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLDisplay EGLAPIENTRY eglGetPlatformDisplayEXT (EGLenum platform, void *native_display, const EGLint *attrib_list);
EGLAPI EGLSurface EGLAPIENTRY eglCreatePlatformWindowSurfaceEXT (EGLDisplay dpy, EGLConfig config, void *native_window, const EGLint *attrib_list);
EGLAPI EGLSurface EGLAPIENTRY eglCreatePlatformPixmapSurfaceEXT (EGLDisplay dpy, EGLConfig config, void *native_pixmap, const EGLint *attrib_list);
#endif
#endif /* EGL_EXT_platform_base */

#ifndef EGL_EXT_platform_device
#define EGL_EXT_platform_device 1
#define EGL_PLATFORM_DEVICE_EXT           0x313F
#endif /* EGL_EXT_platform_device */

#ifndef EGL_EXT_platform_wayland
#define EGL_EXT_platform_wayland 1
#define EGL_PLATFORM_WAYLAND_EXT          0x31D8
#endif /* EGL_EXT_platform_wayland */

#ifndef EGL_EXT_platform_x11
#define EGL_EXT_platform_x11 1
#define EGL_PLATFORM_X11_EXT              0x31D5
#define EGL_PLATFORM_X11_SCREEN_EXT       0x31D6
#endif /* EGL_EXT_platform_x11 */

#ifndef EGL_EXT_protected_surface
#define EGL_EXT_protected_surface 1
#define EGL_PROTECTED_CONTENT_EXT         0x32C0
#endif /* EGL_EXT_protected_surface */

#ifndef EGL_EXT_swap_buffers_with_damage
#define EGL_EXT_swap_buffers_with_damage 1
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSWAPBUFFERSWITHDAMAGEEXTPROC) (EGLDisplay dpy, EGLSurface surface, EGLint *rects, EGLint n_rects);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglSwapBuffersWithDamageEXT (EGLDisplay dpy, EGLSurface surface, EGLint *rects, EGLint n_rects);
#endif
#endif /* EGL_EXT_swap_buffers_with_damage */

#ifndef EGL_HI_clientpixmap
#define EGL_HI_clientpixmap 1
struct EGLClientPixmapHI {
    void  *pData;
    EGLint iWidth;
    EGLint iHeight;
    EGLint iStride;
};
#define EGL_CLIENT_PIXMAP_POINTER_HI      0x8F74
typedef EGLSurface (EGLAPIENTRYP PFNEGLCREATEPIXMAPSURFACEHIPROC) (EGLDisplay dpy, EGLConfig config, struct EGLClientPixmapHI *pixmap);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLSurface EGLAPIENTRY eglCreatePixmapSurfaceHI (EGLDisplay dpy, EGLConfig config, struct EGLClientPixmapHI *pixmap);
#endif
#endif /* EGL_HI_clientpixmap */

#ifndef EGL_HI_colorformats
#define EGL_HI_colorformats 1
#define EGL_COLOR_FORMAT_HI               0x8F70
#define EGL_COLOR_RGB_HI                  0x8F71
#define EGL_COLOR_RGBA_HI                 0x8F72
#define EGL_COLOR_ARGB_HI                 0x8F73
#endif /* EGL_HI_colorformats */

#ifndef EGL_IMG_context_priority
#define EGL_IMG_context_priority 1
#define EGL_CONTEXT_PRIORITY_LEVEL_IMG    0x3100
#define EGL_CONTEXT_PRIORITY_HIGH_IMG     0x3101
#define EGL_CONTEXT_PRIORITY_MEDIUM_IMG   0x3102
#define EGL_CONTEXT_PRIORITY_LOW_IMG      0x3103
#endif /* EGL_IMG_context_priority */

#ifndef EGL_MESA_drm_image
#define EGL_MESA_drm_image 1
#define EGL_DRM_BUFFER_FORMAT_MESA        0x31D0
#define EGL_DRM_BUFFER_USE_MESA           0x31D1
#define EGL_DRM_BUFFER_FORMAT_ARGB32_MESA 0x31D2
#define EGL_DRM_BUFFER_MESA               0x31D3
#define EGL_DRM_BUFFER_STRIDE_MESA        0x31D4
#define EGL_DRM_BUFFER_USE_SCANOUT_MESA   0x00000001
#define EGL_DRM_BUFFER_USE_SHARE_MESA     0x00000002
typedef EGLImageKHR (EGLAPIENTRYP PFNEGLCREATEDRMIMAGEMESAPROC) (EGLDisplay dpy, const EGLint *attrib_list);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLEXPORTDRMIMAGEMESAPROC) (EGLDisplay dpy, EGLImageKHR image, EGLint *name, EGLint *handle, EGLint *stride);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLImageKHR EGLAPIENTRY eglCreateDRMImageMESA (EGLDisplay dpy, const EGLint *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglExportDRMImageMESA (EGLDisplay dpy, EGLImageKHR image, EGLint *name, EGLint *handle, EGLint *stride);
#endif
#endif /* EGL_MESA_drm_image */

#ifndef EGL_MESA_platform_gbm
#define EGL_MESA_platform_gbm 1
#define EGL_PLATFORM_GBM_MESA             0x31D7
#endif /* EGL_MESA_platform_gbm */

#ifndef EGL_NOK_swap_region
#define EGL_NOK_swap_region 1
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSWAPBUFFERSREGIONNOKPROC) (EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint *rects);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglSwapBuffersRegionNOK (EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint *rects);
#endif
#endif /* EGL_NOK_swap_region */

#ifndef EGL_NOK_swap_region2
#define EGL_NOK_swap_region2 1
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSWAPBUFFERSREGION2NOKPROC) (EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint *rects);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglSwapBuffersRegion2NOK (EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint *rects);
#endif
#endif /* EGL_NOK_swap_region2 */

#ifndef EGL_NOK_texture_from_pixmap
#define EGL_NOK_texture_from_pixmap 1
#define EGL_Y_INVERTED_NOK                0x307F
#endif /* EGL_NOK_texture_from_pixmap */

#ifndef EGL_NV_3dvision_surface
#define EGL_NV_3dvision_surface 1
#define EGL_AUTO_STEREO_NV                0x3136
#endif /* EGL_NV_3dvision_surface */

#ifndef EGL_NV_coverage_sample
#define EGL_NV_coverage_sample 1
#define EGL_COVERAGE_BUFFERS_NV           0x30E0
#define EGL_COVERAGE_SAMPLES_NV           0x30E1
#endif /* EGL_NV_coverage_sample */

#ifndef EGL_NV_coverage_sample_resolve
#define EGL_NV_coverage_sample_resolve 1
#define EGL_COVERAGE_SAMPLE_RESOLVE_NV    0x3131
#define EGL_COVERAGE_SAMPLE_RESOLVE_DEFAULT_NV 0x3132
#define EGL_COVERAGE_SAMPLE_RESOLVE_NONE_NV 0x3133
#endif /* EGL_NV_coverage_sample_resolve */

#ifndef EGL_NV_depth_nonlinear
#define EGL_NV_depth_nonlinear 1
#define EGL_DEPTH_ENCODING_NV             0x30E2
#define EGL_DEPTH_ENCODING_NONE_NV        0
#define EGL_DEPTH_ENCODING_NONLINEAR_NV   0x30E3
#endif /* EGL_NV_depth_nonlinear */

#ifndef EGL_NV_native_query
#define EGL_NV_native_query 1
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYNATIVEDISPLAYNVPROC) (EGLDisplay dpy, EGLNativeDisplayType *display_id);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYNATIVEWINDOWNVPROC) (EGLDisplay dpy, EGLSurface surf, EGLNativeWindowType *window);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYNATIVEPIXMAPNVPROC) (EGLDisplay dpy, EGLSurface surf, EGLNativePixmapType *pixmap);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglQueryNativeDisplayNV (EGLDisplay dpy, EGLNativeDisplayType *display_id);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryNativeWindowNV (EGLDisplay dpy, EGLSurface surf, EGLNativeWindowType *window);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryNativePixmapNV (EGLDisplay dpy, EGLSurface surf, EGLNativePixmapType *pixmap);
#endif
#endif /* EGL_NV_native_query */

#ifndef EGL_NV_post_convert_rounding
#define EGL_NV_post_convert_rounding 1
#endif /* EGL_NV_post_convert_rounding */

#ifndef EGL_NV_post_sub_buffer
#define EGL_NV_post_sub_buffer 1
#define EGL_POST_SUB_BUFFER_SUPPORTED_NV  0x30BE
typedef EGLBoolean (EGLAPIENTRYP PFNEGLPOSTSUBBUFFERNVPROC) (EGLDisplay dpy, EGLSurface surface, EGLint x, EGLint y, EGLint width, EGLint height);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglPostSubBufferNV (EGLDisplay dpy, EGLSurface surface, EGLint x, EGLint y, EGLint width, EGLint height);
#endif
#endif /* EGL_NV_post_sub_buffer */

#ifndef EGL_NV_stream_sync
#define EGL_NV_stream_sync 1
#define EGL_SYNC_NEW_FRAME_NV             0x321F
typedef EGLSyncKHR (EGLAPIENTRYP PFNEGLCREATESTREAMSYNCNVPROC) (EGLDisplay dpy, EGLStreamKHR stream, EGLenum type, const EGLint *attrib_list);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLSyncKHR EGLAPIENTRY eglCreateStreamSyncNV (EGLDisplay dpy, EGLStreamKHR stream, EGLenum type, const EGLint *attrib_list);
#endif
#endif /* EGL_NV_stream_sync */

#ifndef EGL_NV_sync
#define EGL_NV_sync 1
typedef void *EGLSyncNV;
typedef khronos_utime_nanoseconds_t EGLTimeNV;
#ifdef KHRONOS_SUPPORT_INT64
#define EGL_SYNC_PRIOR_COMMANDS_COMPLETE_NV 0x30E6
#define EGL_SYNC_STATUS_NV                0x30E7
#define EGL_SIGNALED_NV                   0x30E8
#define EGL_UNSIGNALED_NV                 0x30E9
#define EGL_SYNC_FLUSH_COMMANDS_BIT_NV    0x0001
#define EGL_FOREVER_NV                    0xFFFFFFFFFFFFFFFFull
#define EGL_ALREADY_SIGNALED_NV           0x30EA
#define EGL_TIMEOUT_EXPIRED_NV            0x30EB
#define EGL_CONDITION_SATISFIED_NV        0x30EC
#define EGL_SYNC_TYPE_NV                  0x30ED
#define EGL_SYNC_CONDITION_NV             0x30EE
#define EGL_SYNC_FENCE_NV                 0x30EF
#define EGL_NO_SYNC_NV                    ((EGLSyncNV)0)
typedef EGLSyncNV (EGLAPIENTRYP PFNEGLCREATEFENCESYNCNVPROC) (EGLDisplay dpy, EGLenum condition, const EGLint *attrib_list);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLDESTROYSYNCNVPROC) (EGLSyncNV sync);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLFENCENVPROC) (EGLSyncNV sync);
typedef EGLint (EGLAPIENTRYP PFNEGLCLIENTWAITSYNCNVPROC) (EGLSyncNV sync, EGLint flags, EGLTimeNV timeout);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSIGNALSYNCNVPROC) (EGLSyncNV sync, EGLenum mode);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLGETSYNCATTRIBNVPROC) (EGLSyncNV sync, EGLint attribute, EGLint *value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLSyncNV EGLAPIENTRY eglCreateFenceSyncNV (EGLDisplay dpy, EGLenum condition, const EGLint *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglDestroySyncNV (EGLSyncNV sync);
EGLAPI EGLBoolean EGLAPIENTRY eglFenceNV (EGLSyncNV sync);
EGLAPI EGLint EGLAPIENTRY eglClientWaitSyncNV (EGLSyncNV sync, EGLint flags, EGLTimeNV timeout);
EGLAPI EGLBoolean EGLAPIENTRY eglSignalSyncNV (EGLSyncNV sync, EGLenum mode);
EGLAPI EGLBoolean EGLAPIENTRY eglGetSyncAttribNV (EGLSyncNV sync, EGLint attribute, EGLint *value);
#endif
#endif /* KHRONOS_SUPPORT_INT64 */
#endif /* EGL_NV_sync */

#ifndef EGL_NV_system_time
#define EGL_NV_system_time 1
typedef khronos_utime_nanoseconds_t EGLuint64NV;
#ifdef KHRONOS_SUPPORT_INT64
typedef EGLuint64NV (EGLAPIENTRYP PFNEGLGETSYSTEMTIMEFREQUENCYNVPROC) (void);
typedef EGLuint64NV (EGLAPIENTRYP PFNEGLGETSYSTEMTIMENVPROC) (void);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLuint64NV EGLAPIENTRY eglGetSystemTimeFrequencyNV (void);
EGLAPI EGLuint64NV EGLAPIENTRY eglGetSystemTimeNV (void);
#endif
#endif /* KHRONOS_SUPPORT_INT64 */
#endif /* EGL_NV_system_time */

#ifdef __cplusplus
}
#endif

#endif
