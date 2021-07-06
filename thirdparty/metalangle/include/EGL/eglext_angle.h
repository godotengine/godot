//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// eglext_angle.h: ANGLE modifications to the eglext.h header file.
//   Currently we don't include this file directly, we patch eglext.h
//   to include it implicitly so it is visible throughout our code.

#ifndef INCLUDE_EGL_EGLEXT_ANGLE_
#define INCLUDE_EGL_EGLEXT_ANGLE_

// clang-format off

#ifndef EGL_ANGLE_robust_resource_initialization
#define EGL_ANGLE_robust_resource_initialization 1
#define EGL_ROBUST_RESOURCE_INITIALIZATION_ANGLE 0x3453
#endif /* EGL_ANGLE_robust_resource_initialization */

#ifndef EGL_ANGLE_keyed_mutex
#define EGL_ANGLE_keyed_mutex 1
#define EGL_DXGI_KEYED_MUTEX_ANGLE        0x33A2
#endif /* EGL_ANGLE_keyed_mutex */

#ifndef EGL_ANGLE_d3d_texture_client_buffer
#define EGL_ANGLE_d3d_texture_client_buffer 1
#define EGL_D3D_TEXTURE_ANGLE             0x33A3
#endif /* EGL_ANGLE_d3d_texture_client_buffer */

#ifndef EGL_ANGLE_software_display
#define EGL_ANGLE_software_display 1
#define EGL_SOFTWARE_DISPLAY_ANGLE ((EGLNativeDisplayType)-1)
#endif /* EGL_ANGLE_software_display */

#ifndef EGL_ANGLE_direct3d_display
#define EGL_ANGLE_direct3d_display 1
#define EGL_D3D11_ELSE_D3D9_DISPLAY_ANGLE ((EGLNativeDisplayType)-2)
#define EGL_D3D11_ONLY_DISPLAY_ANGLE ((EGLNativeDisplayType)-3)
#endif /* EGL_ANGLE_direct3d_display */

#ifndef EGL_ANGLE_direct_composition
#define EGL_ANGLE_direct_composition 1
#define EGL_DIRECT_COMPOSITION_ANGLE 0x33A5
#endif /* EGL_ANGLE_direct_composition */

#ifndef EGL_ANGLE_platform_angle
#define EGL_ANGLE_platform_angle 1
#define EGL_PLATFORM_ANGLE_ANGLE          0x3202
#define EGL_PLATFORM_ANGLE_TYPE_ANGLE     0x3203
#define EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE 0x3204
#define EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE 0x3205
#define EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE 0x3206
#define EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED_ANGLE 0x3451
#define EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE 0x3209
#define EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE 0x320A
#define EGL_PLATFORM_ANGLE_DEVICE_TYPE_NULL_ANGLE 0x345E
#endif /* EGL_ANGLE_platform_angle */

#ifndef EGL_ANGLE_platform_angle_d3d
#define EGL_ANGLE_platform_angle_d3d 1
#define EGL_PLATFORM_ANGLE_TYPE_D3D9_ANGLE 0x3207
#define EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE 0x3208
#define EGL_PLATFORM_ANGLE_DEVICE_TYPE_D3D_WARP_ANGLE 0x320B
#define EGL_PLATFORM_ANGLE_DEVICE_TYPE_D3D_REFERENCE_ANGLE 0x320C
#define EGL_PLATFORM_ANGLE_ENABLE_AUTOMATIC_TRIM_ANGLE 0x320F
#endif /* EGL_ANGLE_platform_angle_d3d */

#ifndef EGL_ANGLE_platform_angle_opengl
#define EGL_ANGLE_platform_angle_opengl 1
#define EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE 0x320D
#define EGL_PLATFORM_ANGLE_TYPE_OPENGLES_ANGLE 0x320E
#define EGL_PLATFORM_ANGLE_EGL_HANDLE_ANGLE 0x3480
#endif /* EGL_ANGLE_platform_angle_opengl */

#ifndef EGL_ANGLE_platform_angle_null
#define EGL_ANGLE_platform_angle_null 1
#define EGL_PLATFORM_ANGLE_TYPE_NULL_ANGLE 0x33AE
#endif /* EGL_ANGLE_platform_angle_null */

#ifndef EGL_ANGLE_platform_angle_vulkan
#define EGL_ANGLE_platform_angle_vulkan 1
#define EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE 0x3450
#endif /* EGL_ANGLE_platform_angle_vulkan */

#ifndef EGL_ANGLE_platform_angle_metal
#define EGL_ANGLE_platform_angle_metal 1
#define EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE 0x3489
#endif /* EGL_ANGLE_platform_angle_metal  */

#ifndef EGL_ANGLE_platform_angle_device_type_swiftshader
#define EGL_ANGLE_platform_angle_device_type_swiftshader
#define EGL_PLATFORM_ANGLE_DEVICE_TYPE_SWIFTSHADER_ANGLE 0x3487
#endif /* EGL_ANGLE_platform_angle_device_type_swiftshader */

#ifndef EGL_ANGLE_platform_angle_context_virtualization
#define EGL_ANGLE_platform_angle_context_virtualization 1
#define EGL_PLATFORM_ANGLE_CONTEXT_VIRTUALIZATION_ANGLE 0x3481
#endif /* EGL_ANGLE_platform_angle_context_virtualization */

#ifndef EGL_ANGLE_x11_visual
#define EGL_ANGLE_x11_visual
#define EGL_X11_VISUAL_ID_ANGLE 0x33A3
#endif /* EGL_ANGLE_x11_visual */

#ifndef EGL_ANGLE_flexible_surface_compatibility
#define EGL_ANGLE_flexible_surface_compatibility 1
#define EGL_FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE 0x33A6
#endif /* EGL_ANGLE_flexible_surface_compatibility */

#ifndef EGL_ANGLE_surface_orientation
#define EGL_ANGLE_surface_orientation
#define EGL_OPTIMAL_SURFACE_ORIENTATION_ANGLE 0x33A7
#define EGL_SURFACE_ORIENTATION_ANGLE 0x33A8
#define EGL_SURFACE_ORIENTATION_INVERT_X_ANGLE 0x0001
#define EGL_SURFACE_ORIENTATION_INVERT_Y_ANGLE 0x0002
#endif /* EGL_ANGLE_surface_orientation */

#ifndef EGL_ANGLE_experimental_present_path
#define EGL_ANGLE_experimental_present_path
#define EGL_EXPERIMENTAL_PRESENT_PATH_ANGLE 0x33A4
#define EGL_EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE 0x33A9
#define EGL_EXPERIMENTAL_PRESENT_PATH_COPY_ANGLE 0x33AA
#endif /* EGL_ANGLE_experimental_present_path */

#ifndef EGL_ANGLE_stream_producer_d3d_texture
#define EGL_ANGLE_stream_producer_d3d_texture
#define EGL_D3D_TEXTURE_SUBRESOURCE_ID_ANGLE 0x33AB
typedef EGLBoolean(EGLAPIENTRYP PFNEGLCREATESTREAMPRODUCERD3DTEXTUREANGLEPROC)(EGLDisplay dpy, EGLStreamKHR stream, const EGLAttrib *attrib_list);
typedef EGLBoolean(EGLAPIENTRYP PFNEGLSTREAMPOSTD3DTEXTUREANGLEPROC)(EGLDisplay dpy, EGLStreamKHR stream, void *texture, const EGLAttrib *attrib_list);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglCreateStreamProducerD3DTextureANGLE(EGLDisplay dpy, EGLStreamKHR stream, const EGLAttrib *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglStreamPostD3DTextureANGLE(EGLDisplay dpy, EGLStreamKHR stream, void *texture, const EGLAttrib *attrib_list);
#endif
#endif /* EGL_ANGLE_stream_producer_d3d_texture */

#ifndef EGL_ANGLE_create_context_webgl_compatibility
#define EGL_ANGLE_create_context_webgl_compatibility 1
#define EGL_CONTEXT_WEBGL_COMPATIBILITY_ANGLE 0x33AC
#endif /* EGL_ANGLE_create_context_webgl_compatibility */

#ifndef EGL_ANGLE_display_texture_share_group
#define EGL_ANGLE_display_texture_share_group 1
#define EGL_DISPLAY_TEXTURE_SHARE_GROUP_ANGLE 0x33AF
#endif /* EGL_ANGLE_display_texture_share_group */

#ifndef EGL_CHROMIUM_create_context_bind_generates_resource
#define EGL_CHROMIUM_create_context_bind_generates_resource 1
#define EGL_CONTEXT_BIND_GENERATES_RESOURCE_CHROMIUM 0x33AD
#endif /* EGL_CHROMIUM_create_context_bind_generates_resource */

#ifndef EGL_ANGLE_create_context_client_arrays
#define EGL_ANGLE_create_context_client_arrays 1
#define EGL_CONTEXT_CLIENT_ARRAYS_ENABLED_ANGLE 0x3452
#endif /* EGL_ANGLE_create_context_client_arrays */

#ifndef EGL_ANGLE_device_creation
#define EGL_ANGLE_device_creation 1
typedef EGLDeviceEXT(EGLAPIENTRYP PFNEGLCREATEDEVICEANGLEPROC) (EGLint device_type, void *native_device, const EGLAttrib *attrib_list);
typedef EGLBoolean(EGLAPIENTRYP PFNEGLRELEASEDEVICEANGLEPROC) (EGLDeviceEXT device);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLDeviceEXT EGLAPIENTRY eglCreateDeviceANGLE(EGLint device_type, void *native_device, const EGLAttrib *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglReleaseDeviceANGLE(EGLDeviceEXT device);
#endif
#endif /* EGL_ANGLE_device_creation */

#ifndef EGL_ANGLE_program_cache_control
#define EGL_ANGLE_program_cache_control 1
#define EGL_PROGRAM_CACHE_SIZE_ANGLE 0x3455
#define EGL_PROGRAM_CACHE_KEY_LENGTH_ANGLE 0x3456
#define EGL_PROGRAM_CACHE_RESIZE_ANGLE 0x3457
#define EGL_PROGRAM_CACHE_TRIM_ANGLE 0x3458
#define EGL_CONTEXT_PROGRAM_BINARY_CACHE_ENABLED_ANGLE 0x3459
typedef EGLint (EGLAPIENTRYP PFNEGLPROGRAMCACHEGETATTRIBANGLEPROC) (EGLDisplay dpy, EGLenum attrib);
typedef void (EGLAPIENTRYP PFNEGLPROGRAMCACHEQUERYANGLEPROC) (EGLDisplay dpy, EGLint index, void *key, EGLint *keysize, void *binary, EGLint *binarysize);
typedef void (EGLAPIENTRYP PFNEGLPROGRAMCACHEPOPULATEANGLEPROC) (EGLDisplay dpy, const void *key, EGLint keysize, const void *binary, EGLint binarysize);
typedef EGLint (EGLAPIENTRYP PFNEGLPROGRAMCACHERESIZEANGLEPROC) (EGLDisplay dpy, EGLint limit, EGLenum mode);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLint EGLAPIENTRY eglProgramCacheGetAttribANGLE(EGLDisplay dpy, EGLenum attrib);
EGLAPI void EGLAPIENTRY eglProgramCacheQueryANGLE(EGLDisplay dpy, EGLint index, void *key, EGLint *keysize, void *binary, EGLint *binarysize);
EGLAPI void EGLAPIENTRY eglProgramCachePopulateANGLE(EGLDisplay dpy, const void *key, EGLint keysize, const void *binary, EGLint binarysize);
EGLAPI EGLint EGLAPIENTRY eglProgramCacheResizeANGLE(EGLDisplay dpy, EGLint limit, EGLenum mode);
#endif
#endif /* EGL_ANGLE_program_cache_control */

#ifndef EGL_ANGLE_iosurface_client_buffer
#define EGL_ANGLE_iosurface_client_buffer 1
#define EGL_IOSURFACE_ANGLE 0x3454
#define EGL_IOSURFACE_PLANE_ANGLE 0x345A
#define EGL_TEXTURE_RECTANGLE_ANGLE 0x345B
#define EGL_TEXTURE_TYPE_ANGLE 0x345C
#define EGL_TEXTURE_INTERNAL_FORMAT_ANGLE 0x345D
#endif /* EGL_ANGLE_iosurface_client_buffer */

#ifndef MGL_texture_client_buffer
#define MGL_texture_client_buffer 1
#define EGL_MTL_TEXTURE_MGL 0x3456
#define EGL_GL_TEXTURE_MGL 0x3457
#ifndef EGL_TEXTURE_RECTANGLE_ANGLE
#define EGL_TEXTURE_RECTANGLE_ANGLE 0x345B
#endif
#ifndef EGL_TEXTURE_TYPE_ANGLE
#define EGL_TEXTURE_TYPE_ANGLE 0x345C
#endif
#ifndef EGL_TEXTURE_INTERNAL_FORMAT_ANGLE
#define EGL_TEXTURE_INTERNAL_FORMAT_ANGLE 0x345D
#endif
#ifndef EGL_BIND_TO_TEXTURE_TARGET_ANGLE
#define EGL_BIND_TO_TEXTURE_TARGET_ANGLE 0x348D
#endif
#endif /* MGL_texture_client_buffer */

#ifndef EGL_ANGLE_create_context_extensions_enabled
#define EGL_ANGLE_create_context_extensions_enabled 1
#define EGL_EXTENSIONS_ENABLED_ANGLE 0x345F
#endif /* EGL_ANGLE_create_context_extensions_enabled */

#ifndef EGL_CHROMIUM_get_sync_values
#define EGL_CHROMIUM_get_sync_values 1
typedef EGLBoolean (EGLAPIENTRYP PFNEGLGETSYNCVALUESCHROMIUMPROC) (EGLDisplay dpy,
                                                             EGLSurface surface,
                                                             EGLuint64KHR *ust,
                                                             EGLuint64KHR *msc,
                                                             EGLuint64KHR *sbc);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglGetSyncValuesCHROMIUM(EGLDisplay dpy,
                                                             EGLSurface surface,
                                                             EGLuint64KHR *ust,
                                                             EGLuint64KHR *msc,
                                                             EGLuint64KHR *sbc);
#endif
#endif /* EGL_CHROMIUM_get_sync_values */

#ifndef EGL_ANGLE_power_preference
#define EGL_ANGLE_power_preference 1
#define EGL_POWER_PREFERENCE_ANGLE 0x3482
#define EGL_LOW_POWER_ANGLE 0x0001
#define EGL_HIGH_POWER_ANGLE 0x0002
#endif /* EGL_ANGLE_power_preference */

#ifndef EGL_ANGLE_feature_control
#define EGL_ANGLE_feature_control 1
#define EGL_FEATURE_NAME_ANGLE 0x3460
#define EGL_FEATURE_CATEGORY_ANGLE 0x3461
#define EGL_FEATURE_DESCRIPTION_ANGLE 0x3462
#define EGL_FEATURE_BUG_ANGLE 0x3463
#define EGL_FEATURE_STATUS_ANGLE 0x3464
#define EGL_FEATURE_COUNT_ANGLE 0x3465
#define EGL_FEATURE_OVERRIDES_ENABLED_ANGLE 0x3466
#define EGL_FEATURE_OVERRIDES_DISABLED_ANGLE 0x3467
#define EGL_FEATURE_CONDITION_ANGLE 0x3468
typedef const char *(EGLAPIENTRYP PFNEGLQUERYSTRINGIANGLEPROC) (EGLDisplay dpy, EGLint name, EGLint index);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYDISPLAYATTRIBANGLEPROC) (EGLDisplay dpy, EGLint attribute, EGLAttrib *value);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI const char *EGLAPIENTRY eglQueryStringiANGLE(EGLDisplay dpy, EGLint name, EGLint index);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDisplayAttribANGLE(EGLDisplay dpy, EGLint attribute, EGLAttrib *value);
#endif
#endif /* EGL_ANGLE_feature_control */

#ifndef EGL_ANGLE_image_d3d11_texture
#define EGL_D3D11_TEXTURE_ANGLE 0x3484
#define EGL_TEXTURE_INTERNAL_FORMAT_ANGLE 0x345D
#endif /* EGL_ANGLE_image_d3d11_texture */

#ifndef EGL_ANGLE_create_context_backwards_compatible
#define EGL_ANGLE_create_context_backwards_compatible 1
#define EGL_CONTEXT_OPENGL_BACKWARDS_COMPATIBLE_ANGLE 0x3483
#endif /* EGL_ANGLE_create_context_backwards_compatible */

#ifndef EGL_ANGLE_device_cgl
#define EGL_ANGLE_device_cgl 1
#define EGL_CGL_CONTEXT_ANGLE 0x3485
#define EGL_CGL_PIXEL_FORMAT_ANGLE 0x3486
#endif

#ifndef EGL_ANGLE_device_mtl
#define EGL_ANGLE_device_mtl 1
#define EGL_MTL_DEVICE_ANGLE 0x33A2
#endif /* EGL_ANGLE_device_mtl */

// clang-format on

#endif  // INCLUDE_EGL_EGLEXT_ANGLE_
