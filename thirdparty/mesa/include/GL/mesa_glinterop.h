/*
 * Mesa 3-D graphics library
 *
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/* Mesa OpenGL inter-driver interoperability interface designed for but not
 * limited to OpenCL.
 *
 * This is a driver-agnostic, backward-compatible interface. The structures
 * are only allowed to grow. They can never shrink and their members can
 * never be removed, renamed, or redefined.
 *
 * The interface doesn't return a lot of static texture parameters like
 * width, height, etc. It mainly returns mutable buffer and texture view
 * parameters that can't be part of the texture allocation (because they are
 * mutable). If drivers want to return more data or want to return static
 * allocation parameters, they can do it in one of these two ways:
 * - attaching the data to the DMABUF handle in a driver-specific way
 * - passing the data via "out_driver_data" in the "in" structure.
 *
 * Mesa is expected to do a lot of error checking on behalf of OpenCL, such
 * as checking the target, miplevel, and texture completeness.
 *
 * OpenCL, on the other hand, needs to check if the display+context combo
 * is compatible with the OpenCL driver by querying the device information.
 * It also needs to check if the texture internal format and channel ordering
 * (returned in a driver-specific way) is supported by OpenCL, among other
 * things.
 */

#ifndef MESA_GLINTEROP_H
#define MESA_GLINTEROP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations to avoid inclusion of GL/glx.h */
#ifndef GLX_H
struct _XDisplay;
struct __GLXcontextRec;
#endif

/* Forward declarations to avoid inclusion of EGL/egl.h */
#ifndef __egl_h_
typedef void *EGLDisplay;
typedef void *EGLContext;
#endif

#ifndef _WINDEF_
struct HDC__;
typedef struct HDC__ *HDC;
struct HGLRC__;
typedef struct HGLRC__ *HGLRC;
typedef void *HANDLE;
#endif

typedef struct __GLsync *GLsync;

/** Returned error codes. */
enum {
   MESA_GLINTEROP_SUCCESS = 0,
   MESA_GLINTEROP_OUT_OF_RESOURCES,
   MESA_GLINTEROP_OUT_OF_HOST_MEMORY,
   MESA_GLINTEROP_INVALID_OPERATION,
   MESA_GLINTEROP_INVALID_VERSION,
   MESA_GLINTEROP_INVALID_DISPLAY,
   MESA_GLINTEROP_INVALID_CONTEXT,
   MESA_GLINTEROP_INVALID_TARGET,
   MESA_GLINTEROP_INVALID_OBJECT,
   MESA_GLINTEROP_INVALID_MIP_LEVEL,
   MESA_GLINTEROP_UNSUPPORTED
};

/** Access flags. */
enum {
   MESA_GLINTEROP_ACCESS_READ_WRITE = 0,
   MESA_GLINTEROP_ACCESS_READ_ONLY,
   MESA_GLINTEROP_ACCESS_WRITE_ONLY
};

#define MESA_GLINTEROP_DEVICE_INFO_VERSION 2

/**
 * Device information returned by Mesa.
 */
struct mesa_glinterop_device_info {
   /* The caller should set this to the version of the struct they support */
   /* The callee will overwrite it if it supports a lower version.
    *
    * The caller should check the value and access up-to the version supported
    * by the callee.
    */
   /* NOTE: Do not use the MESA_GLINTEROP_DEVICE_INFO_VERSION macro */
   uint32_t version;

   /* PCI location */
   uint32_t pci_segment_group;
   uint32_t pci_bus;
   uint32_t pci_device;
   uint32_t pci_function;

   /* Device identification */
   uint32_t vendor_id;
   uint32_t device_id;

   /* Structure version 1 ends here. */

   /* Size of memory pointed to by out_driver_data. */
   uint32_t driver_data_size;

   /* If the caller wants to query driver-specific data about the OpenGL
   * object, this should point to the memory where that data will be stored.
   * This is expected to be a temporary staging memory. The pointer is not
   * allowed to be saved for later use by Mesa.
   */
   void *driver_data;

   /* Structure version 2 ends here. */
};

#define MESA_GLINTEROP_EXPORT_IN_VERSION 1

/**
 * Input parameters to Mesa interop export functions.
 */
struct mesa_glinterop_export_in {
   /* The caller should set this to the version of the struct they support */
   /* The callee will overwrite it if it supports a lower version.
    *
    * The caller should check the value and access up-to the version supported
    * by the callee.
    */
   /* NOTE: Do not use the MESA_GLINTEROP_EXPORT_IN_VERSION macro */
   uint32_t version;

   /* One of the following:
    * - GL_TEXTURE_BUFFER
    * - GL_TEXTURE_1D
    * - GL_TEXTURE_2D
    * - GL_TEXTURE_3D
    * - GL_TEXTURE_RECTANGLE
    * - GL_TEXTURE_1D_ARRAY
    * - GL_TEXTURE_2D_ARRAY
    * - GL_TEXTURE_CUBE_MAP_ARRAY
    * - GL_TEXTURE_CUBE_MAP
    * - GL_TEXTURE_CUBE_MAP_POSITIVE_X
    * - GL_TEXTURE_CUBE_MAP_NEGATIVE_X
    * - GL_TEXTURE_CUBE_MAP_POSITIVE_Y
    * - GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
    * - GL_TEXTURE_CUBE_MAP_POSITIVE_Z
    * - GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    * - GL_TEXTURE_2D_MULTISAMPLE
    * - GL_TEXTURE_2D_MULTISAMPLE_ARRAY
    * - GL_TEXTURE_EXTERNAL_OES
    * - GL_RENDERBUFFER
    * - GL_ARRAY_BUFFER
    */
   unsigned target;

   /* If target is GL_ARRAY_BUFFER, it's a buffer object.
    * If target is GL_RENDERBUFFER, it's a renderbuffer object.
    * If target is GL_TEXTURE_*, it's a texture object.
    */
   unsigned obj;

   /* Mipmap level. Ignored for non-texture objects. */
   unsigned miplevel;

   /* One of MESA_GLINTEROP_ACCESS_* flags. This describes how the exported
    * object is going to be used.
    */
   uint32_t access;

   /* Size of memory pointed to by out_driver_data. */
   uint32_t out_driver_data_size;

   /* If the caller wants to query driver-specific data about the OpenGL
    * object, this should point to the memory where that data will be stored.
    * This is expected to be a temporary staging memory. The pointer is not
    * allowed to be saved for later use by Mesa.
    */
   void *out_driver_data;
   /* Structure version 1 ends here. */
};

#define MESA_GLINTEROP_EXPORT_OUT_VERSION 1

/**
 * Outputs of Mesa interop export functions.
 */
struct mesa_glinterop_export_out {
   /* The caller should set this to the version of the struct they support */
   /* The callee will overwrite it if it supports a lower version.
    *
    * The caller should check the value and access up-to the version supported
    * by the callee.
    */
   /* NOTE: Do not use the MESA_GLINTEROP_EXPORT_OUT_VERSION macro */
   uint32_t version;

#ifndef _WIN32
   /* The DMABUF handle. It must be closed by the caller using the POSIX
    * close() function when it's not needed anymore. Mesa is not responsible
    * for closing the handle.
    *
    * Not closing the handle by the caller will lead to a resource leak,
    * will prevent releasing the GPU buffer, and may prevent creating new
    * DMABUF handles within the process.
    */
   int dmabuf_fd;
#else
   /* Same concept as a DMABUF, but for Windows/WDDM. It must be closed by
    * the caller using CloseHandle() when it's not needed anymore.
    */
   HANDLE win32_handle;
#endif

   /* The mutable OpenGL internal format specified by glTextureView or
    * glTexBuffer. If the object is not one of those, the original internal
    * format specified by glTexStorage, glTexImage, or glRenderbufferStorage
    * will be returned.
    */
   unsigned internal_format;

   /* Buffer offset and size for GL_ARRAY_BUFFER and GL_TEXTURE_BUFFER.
    * This allows interop with suballocations (a buffer allocated within
    * a larger buffer).
    *
    * Parameters specified by glTexBufferRange for GL_TEXTURE_BUFFER are
    * applied to these and can shrink the range further.
    */
   ptrdiff_t buf_offset;
   ptrdiff_t buf_size;

   /* Parameters specified by glTextureView. If the object is not a texture
    * view, default parameters covering the whole texture will be returned.
    */
   unsigned view_minlevel;
   unsigned view_numlevels;
   unsigned view_minlayer;
   unsigned view_numlayers;

   /* The number of bytes written to out_driver_data. */
   uint32_t out_driver_data_written;
   /* Structure version 1 ends here. */
};


/**
 * Query device information.
 *
 * \param dpy        GLX display
 * \param context    GLX context
 * \param out        where to return the information
 *
 * \return MESA_GLINTEROP_SUCCESS or MESA_GLINTEROP_* != 0 on error
 */
int
MesaGLInteropGLXQueryDeviceInfo(struct _XDisplay *dpy, struct __GLXcontextRec *context,
                                struct mesa_glinterop_device_info *out);


/**
 * Same as MesaGLInteropGLXQueryDeviceInfo except that it accepts EGLDisplay
 * and EGLContext.
 */
int
MesaGLInteropEGLQueryDeviceInfo(EGLDisplay dpy, EGLContext context,
                                struct mesa_glinterop_device_info *out);


/**
* Same as MesaGLInteropGLXQueryDeviceInfo except that it accepts HDC
* and HGLRC.
*/
int
wglMesaGLInteropQueryDeviceInfo(HDC dpy, HGLRC context,
                                struct mesa_glinterop_device_info *out);

/**
 * Create and return a DMABUF handle corresponding to the given OpenGL
 * object, and return other parameters about the OpenGL object.
 *
 * \param dpy        GLX display
 * \param context    GLX context
 * \param in         input parameters
 * \param out        return values
 *
 * \return MESA_GLINTEROP_SUCCESS or MESA_GLINTEROP_* != 0 on error
 */
int
MesaGLInteropGLXExportObject(struct _XDisplay *dpy, struct __GLXcontextRec *context,
                             struct mesa_glinterop_export_in *in,
                             struct mesa_glinterop_export_out *out);


/**
 * Same as MesaGLInteropGLXExportObject except that it accepts
 * EGLDisplay and EGLContext.
 */
int
MesaGLInteropEGLExportObject(EGLDisplay dpy, EGLContext context,
                             struct mesa_glinterop_export_in *in,
                             struct mesa_glinterop_export_out *out);


/**
* Same as MesaGLInteropGLXExportObject except that it accepts
* HDC and HGLRC.
*/
int
wglMesaGLInteropExportObject(HDC dpy, HGLRC context,
                             struct mesa_glinterop_export_in *in,
                             struct mesa_glinterop_export_out *out);


/**
 * Prepare OpenGL resources for being accessed by OpenCL.
 * 
 * \param dpy        GLX display
 * \param context    GLX context
 * \param count      number of resources
 * \param resources  resources to flush
 * \param sync       optional GLsync to map to CL event
 * 
 * \return MESA_GLINTEROP_SUCCESS or MESA_GLINTEROP_* != 0 on error
 */
int
MesaGLInteropGLXFlushObjects(struct _XDisplay *dpy, struct __GLXcontextRec *context,
                             unsigned count, struct mesa_glinterop_export_in *resources,
                             GLsync *sync);

/**
* Same as MesaGLInteropGLXFlushObjects except that it accepts
* EGLDisplay and EGLContext.
*/
int
MesaGLInteropEGLFlushObjects(EGLDisplay dpy, EGLContext context,
                             unsigned count, struct mesa_glinterop_export_in *resources,
                             GLsync *sync);

/**
* Same as MesaGLInteropGLXFlushObjects except that it accepts
* HDC and HGLRC.
*/
int
wglMesaGLInteropFlushObjects(HDC dpy, HGLRC context,
                             unsigned count, struct mesa_glinterop_export_in *resources,
                             GLsync *sync);


typedef int (PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC)(struct _XDisplay *dpy, struct __GLXcontextRec *context,
                                                     struct mesa_glinterop_device_info *out);
typedef int (PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC)(EGLDisplay dpy, EGLContext context,
                                                     struct mesa_glinterop_device_info *out);
typedef int (PFNWGLMESAGLINTEROPQUERYDEVICEINFOPROC)(HDC dpy, HGLRC context,
                                                     struct mesa_glinterop_device_info *out);
typedef int (PFNMESAGLINTEROPGLXEXPORTOBJECTPROC)(struct _XDisplay *dpy, struct __GLXcontextRec *context,
                                                  struct mesa_glinterop_export_in *in,
                                                  struct mesa_glinterop_export_out *out);
typedef int (PFNMESAGLINTEROPEGLEXPORTOBJECTPROC)(EGLDisplay dpy, EGLContext context,
                                                  struct mesa_glinterop_export_in *in,
                                                  struct mesa_glinterop_export_out *out);
typedef int (PFNWGLMESAGLINTEROPEXPORTOBJECTPROC)(HDC dpy, HGLRC context,
                                                  struct mesa_glinterop_export_in *in,
                                                  struct mesa_glinterop_export_out *out);
typedef int (PFNMESAGLINTEROPGLXFLUSHOBJECTSPROC)(struct _XDisplay *dpy, struct __GLXcontextRec *context,
                                                  unsigned count, struct mesa_glinterop_export_in *resources,
                                                  GLsync *sync);
typedef int (PFNMESAGLINTEROPEGLFLUSHOBJECTSPROC)(EGLDisplay dpy, EGLContext context,
                                                  unsigned count, struct mesa_glinterop_export_in *resources,
                                                  GLsync *sync);
typedef int (PFNWGLMESAGLINTEROPFLUSHOBJECTSPROC)(HDC dpy, HGLRC context,
                                                  unsigned count, struct mesa_glinterop_export_in *resources,
                                                  GLsync *sync);

#ifdef __cplusplus
}
#endif

#endif /* MESA_GLINTEROP_H */
