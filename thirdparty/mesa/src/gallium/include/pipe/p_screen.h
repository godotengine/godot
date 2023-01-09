/**************************************************************************
 *
 * Copyright 2007 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/**
 * @file
 *
 * Screen, Adapter or GPU
 *
 * These are driver functions/facilities that are context independent.
 */


#ifndef P_SCREEN_H
#define P_SCREEN_H


#include "pipe/p_compiler.h"
#include "util/format/u_formats.h"
#include "pipe/p_defines.h"
#include "pipe/p_video_enums.h"



#ifdef __cplusplus
extern "C" {
#endif


/** Opaque types */
struct winsys_handle;
struct pipe_fence_handle;
struct pipe_resource;
struct pipe_surface;
struct pipe_transfer;
struct pipe_box;
struct pipe_memory_info;
struct pipe_vertex_buffer;
struct pipe_vertex_element;
struct pipe_vertex_state;
struct disk_cache;
struct driOptionCache;
struct u_transfer_helper;
struct pipe_screen;
struct util_queue_fence;

typedef struct pipe_vertex_state *
   (*pipe_create_vertex_state_func)(struct pipe_screen *screen,
                                    struct pipe_vertex_buffer *buffer,
                                    const struct pipe_vertex_element *elements,
                                    unsigned num_elements,
                                    struct pipe_resource *indexbuf,
                                    uint32_t full_velem_mask);
typedef void (*pipe_vertex_state_destroy_func)(struct pipe_screen *screen,
                                               struct pipe_vertex_state *);
typedef void (*pipe_driver_thread_func)(void *job, void *gdata, int thread_index);



/**
 * Gallium screen/adapter context.  Basically everything
 * hardware-specific that doesn't actually require a rendering
 * context.
 */
struct pipe_screen {
   /**
    * Atomically incremented by drivers to track the number of contexts.
    * If it's 0, it can be assumed that contexts are not tracked.
    * Used by some places to skip locking if num_contexts == 1.
    */
   unsigned num_contexts;

   /**
    * For drivers using u_transfer_helper:
    */
   struct u_transfer_helper *transfer_helper;

   void (*destroy)(struct pipe_screen *);

   const char *(*get_name)(struct pipe_screen *);

   const char *(*get_vendor)(struct pipe_screen *);

   /**
    * Returns the device vendor.
    *
    * The returned value should return the actual device vendor/manufacturer,
    * rather than a potentially generic driver string.
    */
   const char *(*get_device_vendor)(struct pipe_screen *);

   /**
    * Returns the latest OpenCL CTS version passed
    *
    * The returned value should be the git tag used when passing conformance.
    */
   const char *(*get_cl_cts_version)(struct pipe_screen *);

   /**
    * Query an integer-valued capability/parameter/limit
    * \param param  one of PIPE_CAP_x
    */
   int (*get_param)(struct pipe_screen *, enum pipe_cap param);

   /**
    * Query a float-valued capability/parameter/limit
    * \param param  one of PIPE_CAP_x
    */
   float (*get_paramf)(struct pipe_screen *, enum pipe_capf param);

   /**
    * Query a per-shader-stage integer-valued capability/parameter/limit
    * \param param  one of PIPE_CAP_x
    */
   int (*get_shader_param)(struct pipe_screen *, enum pipe_shader_type shader,
                           enum pipe_shader_cap param);

   /**
    * Query an integer-valued capability/parameter/limit for a codec/profile
    * \param param  one of PIPE_VIDEO_CAP_x
    */
   int (*get_video_param)(struct pipe_screen *,
                          enum pipe_video_profile profile,
                          enum pipe_video_entrypoint entrypoint,
                          enum pipe_video_cap param);

   /**
    * Query a compute-specific capability/parameter/limit.
    * \param ir_type shader IR type for which the param applies, or don't care
    *                if the param is not shader related
    * \param param   one of PIPE_COMPUTE_CAP_x
    * \param ret     pointer to a preallocated buffer that will be
    *                initialized to the parameter value, or NULL.
    * \return        size in bytes of the parameter value that would be
    *                returned.
    */
   int (*get_compute_param)(struct pipe_screen *,
                            enum pipe_shader_ir ir_type,
                            enum pipe_compute_cap param,
                            void *ret);

   /**
    * Get the sample pixel grid's size. This function requires
    * PIPE_CAP_PROGRAMMABLE_SAMPLE_LOCATIONS to be callable.
    *
    * \param sample_count - total number of samples
    * \param out_width - the width of the pixel grid
    * \param out_height - the height of the pixel grid
    */
   void (*get_sample_pixel_grid)(struct pipe_screen *, unsigned sample_count,
                                 unsigned *out_width, unsigned *out_height);

   /**
    * Query a timestamp in nanoseconds. The returned value should match
    * PIPE_QUERY_TIMESTAMP. This function returns immediately and doesn't
    * wait for rendering to complete (which cannot be achieved with queries).
    */
   uint64_t (*get_timestamp)(struct pipe_screen *);

   /**
    * Return an equivalent canonical format which has the same component sizes
    * and swizzles as the original, and it is supported by the driver. Gallium
    * already does a first canonicalization step (see get_canonical_format()
    * on st_cb_copyimage.c) and it calls this function (if defined) to get an
    * alternative format if the picked is not supported by the driver.
    */
   enum pipe_format (*get_canonical_format)(struct pipe_screen *,
                                            enum pipe_format format);

   /**
    * Create a context.
    *
    * \param screen      pipe screen
    * \param priv        a pointer to set in pipe_context::priv
    * \param flags       a mask of PIPE_CONTEXT_* flags
    */
   struct pipe_context * (*context_create)(struct pipe_screen *screen,
                                           void *priv, unsigned flags);

   /**
    * Check if the given image copy will be faster on compute
    * \param cpu If true, this is checking against CPU fallback,
    *            otherwise the copy will be on GFX
    */
   bool (*is_compute_copy_faster)(struct pipe_screen *,
                                  enum pipe_format src_format,
                                  enum pipe_format dst_format,
                                  unsigned width,
                                  unsigned height,
                                  unsigned depth,
                                  bool cpu);

   /**
    * Check if the given pipe_format is supported as a texture or
    * drawing surface.
    * \param bindings  bitmask of PIPE_BIND_*
    */
   bool (*is_format_supported)(struct pipe_screen *,
                               enum pipe_format format,
                               enum pipe_texture_target target,
                               unsigned sample_count,
                               unsigned storage_sample_count,
                               unsigned bindings);

   /**
    * Check if the given pipe_format is supported as output for this
    * codec/profile.
    * \param profile  profile to check, may also be PIPE_VIDEO_PROFILE_UNKNOWN
    */
   bool (*is_video_format_supported)(struct pipe_screen *,
                                     enum pipe_format format,
                                     enum pipe_video_profile profile,
                                     enum pipe_video_entrypoint entrypoint);

   /**
    * Check if we can actually create the given resource (test the dimension,
    * overall size, etc).  Used to implement proxy textures.
    * \return TRUE if size is OK, FALSE if too large.
    */
   bool (*can_create_resource)(struct pipe_screen *screen,
                               const struct pipe_resource *templat);

   /**
    * Create a new texture object, using the given template info.
    */
   struct pipe_resource * (*resource_create)(struct pipe_screen *,
                                             const struct pipe_resource *templat);

   struct pipe_resource * (*resource_create_drawable)(struct pipe_screen *,
                                                      const struct pipe_resource *tmpl,
                                                      const void *loader_private);

   struct pipe_resource * (*resource_create_front)(struct pipe_screen *,
                                                   const struct pipe_resource *templat,
                                                   const void *map_front_private);

   /**
    * Create a texture from a winsys_handle. The handle is often created in
    * another process by first creating a pipe texture and then calling
    * resource_get_handle.
    *
    * NOTE: in the case of WINSYS_HANDLE_TYPE_FD handles, the caller
    * retains ownership of the FD.  (This is consistent with
    * EGL_EXT_image_dma_buf_import)
    *
    * \param usage  A combination of PIPE_HANDLE_USAGE_* flags.
    */
   struct pipe_resource * (*resource_from_handle)(struct pipe_screen *,
                                                  const struct pipe_resource *templat,
                                                  struct winsys_handle *handle,
                                                  unsigned usage);

   /**
    * Create a resource from user memory. This maps the user memory into
    * the device address space.
    */
   struct pipe_resource * (*resource_from_user_memory)(struct pipe_screen *,
                                                       const struct pipe_resource *t,
                                                       void *user_memory);

   /**
    * Unlike pipe_resource::bind, which describes what gallium frontends want,
    * resources can have much greater capabilities in practice, often implied
    * by the tiling layout or memory placement. This function allows querying
    * whether a capability is supported beyond what was requested by state
    * trackers. It's also useful for querying capabilities of imported
    * resources where the capabilities are unknown at first.
    *
    * Only these flags are allowed:
    * - PIPE_BIND_SCANOUT
    * - PIPE_BIND_CURSOR
    * - PIPE_BIND_LINEAR
    */
   bool (*check_resource_capability)(struct pipe_screen *screen,
                                     struct pipe_resource *resource,
                                     unsigned bind);

   /**
    * Get a winsys_handle from a texture. Some platforms/winsys requires
    * that the texture is created with a special usage flag like
    * DISPLAYTARGET or PRIMARY.
    *
    * The context parameter can optionally be used to flush the resource and
    * the context to make sure the resource is coherent with whatever user
    * will use it. Some drivers may also use the context to convert
    * the resource into a format compatible for sharing. The use case is
    * OpenGL-OpenCL interop. The context parameter is allowed to be NULL.
    *
    * NOTE: for multi-planar resources (which may or may not have the planes
    * chained through the pipe_resource next pointer) the frontend will
    * always call this function with the first resource of the chain. It is
    * the pipe drivers responsibility to walk the resources as needed when
    * called with handle->plane != 0.
    *
    * NOTE: in the case of WINSYS_HANDLE_TYPE_FD handles, the caller
    * takes ownership of the FD.  (This is consistent with
    * EGL_MESA_image_dma_buf_export)
    *
    * \param usage  A combination of PIPE_HANDLE_USAGE_* flags.
    */
   bool (*resource_get_handle)(struct pipe_screen *,
                               struct pipe_context *context,
                               struct pipe_resource *tex,
                               struct winsys_handle *handle,
                               unsigned usage);

   /**
    * Get info for the given pipe resource without the need to get a
    * winsys_handle.
    *
    * The context parameter can optionally be used to flush the resource and
    * the context to make sure the resource is coherent with whatever user
    * will use it. Some drivers may also use the context to convert
    * the resource into a format compatible for sharing. The context parameter
    * is allowed to be NULL.
    */
   bool (*resource_get_param)(struct pipe_screen *screen,
                              struct pipe_context *context,
                              struct pipe_resource *resource,
                              unsigned plane,
                              unsigned layer,
                              unsigned level,
                              enum pipe_resource_param param,
                              unsigned handle_usage,
                              uint64_t *value);

   /**
    * Get stride and offset for the given pipe resource without the need to get
    * a winsys_handle.
    */
   void (*resource_get_info)(struct pipe_screen *screen,
                             struct pipe_resource *resource,
                             unsigned *stride,
                             unsigned *offset);

   /**
    * Mark the resource as changed so derived internal resources will be
    * recreated on next use.
    *
    * This is necessary when reimporting external images that can't be directly
    * used as texture sampler source, to avoid sampling from old copies.
    */
   void (*resource_changed)(struct pipe_screen *, struct pipe_resource *pt);

   void (*resource_destroy)(struct pipe_screen *,
                            struct pipe_resource *pt);


   /**
    * Do any special operations to ensure frontbuffer contents are
    * displayed, eg copy fake frontbuffer.
    * \param winsys_drawable_handle  an opaque handle that the calling context
    *                                gets out-of-band
    * \param subbox an optional sub region to flush
    */
   void (*flush_frontbuffer)(struct pipe_screen *screen,
                             struct pipe_context *ctx,
                             struct pipe_resource *resource,
                             unsigned level, unsigned layer,
                             void *winsys_drawable_handle,
                             struct pipe_box *subbox);

   /** Set ptr = fence, with reference counting */
   void (*fence_reference)(struct pipe_screen *screen,
                           struct pipe_fence_handle **ptr,
                           struct pipe_fence_handle *fence);

   /**
    * Wait for the fence to finish.
    *
    * If the fence was created with PIPE_FLUSH_DEFERRED, and the context is
    * still unflushed, and the ctx parameter of fence_finish is equal to
    * the context where the fence was created, fence_finish will flush
    * the context prior to waiting for the fence.
    *
    * In all other cases, the ctx parameter has no effect.
    *
    * \param timeout  in nanoseconds (may be PIPE_TIMEOUT_INFINITE).
    */
   bool (*fence_finish)(struct pipe_screen *screen,
                        struct pipe_context *ctx,
                        struct pipe_fence_handle *fence,
                        uint64_t timeout);

   /**
    * For fences created with PIPE_FLUSH_FENCE_FD (exported fd) or
    * by create_fence_fd() (imported fd), return the native fence fd
    * associated with the fence.  This may return -1 for fences
    * created with PIPE_FLUSH_DEFERRED if the fence command has not
    * been flushed yet.
    */
   int (*fence_get_fd)(struct pipe_screen *screen,
                       struct pipe_fence_handle *fence);

   /**
    * Create a fence from an Win32 handle.
    *
    * This is used for importing a foreign/external fence handle.
    *
    * \param fence  if not NULL, an old fence to unref and transfer a
    *    new fence reference to
    * \param handle opaque handle representing the fence object
    * \param type   indicates which fence types backs the handle
    */
   void (*create_fence_win32)(struct pipe_screen *screen,
                              struct pipe_fence_handle **fence,
                              void *handle,
                              const void *name,
                              enum pipe_fd_type type);

   /**
    * Returns a driver-specific query.
    *
    * If \p info is NULL, the number of available queries is returned.
    * Otherwise, the driver query at the specified \p index is returned
    * in \p info. The function returns non-zero on success.
    */
   int (*get_driver_query_info)(struct pipe_screen *screen,
                                unsigned index,
                                struct pipe_driver_query_info *info);

   /**
    * Returns a driver-specific query group.
    *
    * If \p info is NULL, the number of available groups is returned.
    * Otherwise, the driver query group at the specified \p index is returned
    * in \p info. The function returns non-zero on success.
    */
   int (*get_driver_query_group_info)(struct pipe_screen *screen,
                                      unsigned index,
                                      struct pipe_driver_query_group_info *info);

   /**
    * Query information about memory usage.
    */
   void (*query_memory_info)(struct pipe_screen *screen,
                             struct pipe_memory_info *info);

   /**
    * Get IR specific compiler options struct.  For PIPE_SHADER_IR_NIR this
    * returns a 'struct nir_shader_compiler_options'.  Drivers reporting
    * NIR as the preferred IR must implement this.
    */
   const void *(*get_compiler_options)(struct pipe_screen *screen,
                                      enum pipe_shader_ir ir,
                                      enum pipe_shader_type shader);

   /**
    * Returns a pointer to a driver-specific on-disk shader cache. If the
    * driver failed to create the cache or does not support an on-disk shader
    * cache NULL is returned. The callback itself may also be NULL if the
    * driver doesn't support an on-disk shader cache.
    */
   struct disk_cache *(*get_disk_shader_cache)(struct pipe_screen *screen);

   /**
    * Create a new texture object from the given template info, taking
    * format modifiers into account. \p modifiers specifies a list of format
    * modifier tokens, as defined in drm_fourcc.h. The driver then picks the
    * best modifier among these and creates the resource. \p count must
    * contain the size of \p modifiers array.
    *
    * Returns NULL if an entry in \p modifiers is unsupported by the driver,
    * or if only DRM_FORMAT_MOD_INVALID is provided.
    */
   struct pipe_resource * (*resource_create_with_modifiers)(
                           struct pipe_screen *,
                           const struct pipe_resource *templat,
                           const uint64_t *modifiers, int count);

   /**
    * Get supported modifiers for a format.
    * If \p max is 0, the total number of supported modifiers for the supplied
    * format is returned in \p count, with no modification to \p modifiers.
    * Otherwise, \p modifiers is filled with upto \p max supported modifier
    * codes, and \p count with the number of modifiers copied.
    * The \p external_only array is used to return whether the format and
    * modifier combination can only be used with an external texture target.
    */
   void (*query_dmabuf_modifiers)(struct pipe_screen *screen,
                                  enum pipe_format format, int max,
                                  uint64_t *modifiers,
                                  unsigned int *external_only, int *count);

   /**
    * Create a memory object from a winsys handle
    *
    * The underlying memory is most often allocated in by a foregin API.
    * Then the underlying memory object is then exported through interfaces
    * compatible with EXT_external_resources.
    *
    * Note: For WINSYS_HANDLE_TYPE_FD handles, the caller retains ownership
    * of the fd.
    *
    * \param handle  A handle representing the memory object to import
    */
   struct pipe_memory_object *(*memobj_create_from_handle)(struct pipe_screen *screen,
                                                           struct winsys_handle *handle,
                                                           bool dedicated);

   /**
    * Destroy a memory object
    *
    * \param memobj  The memory object to destroy
    */
   void (*memobj_destroy)(struct pipe_screen *screen,
                          struct pipe_memory_object *memobj);

   /**
    * Create a texture from a memory object
    *
    * \param t       texture template
    * \param memobj  The memory object used to back the texture
    */
   struct pipe_resource * (*resource_from_memobj)(struct pipe_screen *screen,
                                                  const struct pipe_resource *t,
                                                  struct pipe_memory_object *memobj,
                                                  uint64_t offset);

   /**
    * Fill @uuid with a unique driver identifier
    *
    * \param uuid    pointer to a memory region of PIPE_UUID_SIZE bytes
    */
   void (*get_driver_uuid)(struct pipe_screen *screen, char *uuid);

   /**
    * Fill @uuid with a unique device identifier
    *
    * \param uuid    pointer to a memory region of PIPE_UUID_SIZE bytes
    */
   void (*get_device_uuid)(struct pipe_screen *screen, char *uuid);

   /**
    * Fill @luid with the locally unique identifier of the context
    * The LUID returned, paired together with the contexts node mask,
    * allows matching the context to an IDXGIAdapter1 object
    *
    * \param luid    pointer to a memory region of PIPE_LUID_SIZE bytes
    */
   void (*get_device_luid)(struct pipe_screen *screen, char *luid);

   /**
    * Return the device node mask identifying the context
    * Together with the contexts LUID, this allows matching
    * the context to an IDXGIAdapter1 object.
    *
    * within a linked device adapter
    */
   uint32_t (*get_device_node_mask)(struct pipe_screen *screen);

   /**
    * Set the maximum number of parallel shader compiler threads.
    */
   void (*set_max_shader_compiler_threads)(struct pipe_screen *screen,
                                           unsigned max_threads);

   /**
    * Return whether parallel shader compilation has finished.
    */
   bool (*is_parallel_shader_compilation_finished)(struct pipe_screen *screen,
                                                   void *shader,
                                                   enum pipe_shader_type shader_type);

   void (*driver_thread_add_job)(struct pipe_screen *screen,
                                 void *job,
                                 struct util_queue_fence *fence,
                                 pipe_driver_thread_func execute,
                                 pipe_driver_thread_func cleanup,
                                 const size_t job_size);

   /**
    * Set the damage region (called when KHR_partial_update() is invoked).
    * This function is passed an array of rectangles encoding the damage area.
    * rects are using the bottom-left origin convention.
    * nrects = 0 means 'reset the damage region'. What 'reset' implies is HW
    * specific. For tile-based renderers, the damage extent is typically set
    * to cover the whole resource with no damage rect (or a 0-size damage
    * rect). This way, the existing resource content is reloaded into the
    * local tile buffer for every tile thus making partial tile update
    * possible. For HW operating in immediate mode, this reset operation is
    * likely to be a NOOP.
    */
   void (*set_damage_region)(struct pipe_screen *screen,
                             struct pipe_resource *resource,
                             unsigned int nrects,
                             const struct pipe_box *rects);

   /**
    * Run driver-specific NIR lowering and optimization passes.
    *
    * gallium frontends should call this before passing shaders to drivers,
    * and ideally also before shader caching.
    *
    * The driver may return a non-NULL string to trigger GLSL link failure
    * and logging of that message in the GLSL linker log.
    */
   char *(*finalize_nir)(struct pipe_screen *screen, void *nir);

   /*Separated memory/resource allocations interfaces for Vulkan */

   /**
    * Create a resource, and retrieve the required size for it but don't
    * allocate any backing memory.
    */
   struct pipe_resource * (*resource_create_unbacked)(struct pipe_screen *,
                                                      const struct pipe_resource *templat,
                                                      uint64_t *size_required);

   /**
    * Allocate backing memory to be bound to resources.
    */
   struct pipe_memory_allocation *(*allocate_memory)(struct pipe_screen *screen,
                                                     uint64_t size);
   /**
    * Free previously allocated backing memory.
    */
   void (*free_memory)(struct pipe_screen *screen,
                       struct pipe_memory_allocation *);

   /**
    * Allocate fd-based memory to be bound to resources.
    */
   struct pipe_memory_allocation *(*allocate_memory_fd)(struct pipe_screen *screen,
                                                        uint64_t size,
                                                        int *fd);

   /**
    * Import memory from an fd-handle.
    */
   bool (*import_memory_fd)(struct pipe_screen *screen,
                            int fd,
                            struct pipe_memory_allocation **pmem,
                            uint64_t *size);

   /**
    * Free previously allocated fd-based memory.
    */
   void (*free_memory_fd)(struct pipe_screen *screen,
                          struct pipe_memory_allocation *pmem);

   /**
    * Bind memory to a resource.
    */
   bool (*resource_bind_backing)(struct pipe_screen *screen,
                                 struct pipe_resource *pt,
                                 struct pipe_memory_allocation *pmem,
                                 uint64_t offset);

   /**
    * Map backing memory.
    */
   void *(*map_memory)(struct pipe_screen *screen,
                       struct pipe_memory_allocation *pmem);

   /**
    * Unmap backing memory.
    */
   void (*unmap_memory)(struct pipe_screen *screen,
                        struct pipe_memory_allocation *pmem);

   /**
    * Determine whether the screen supports the specified modifier
    *
    * Query whether the driver supports a \p modifier in combination with
    * \p format.  If \p external_only is not NULL, the value it points to will
    * be set to 0 or a non-zero value to indicate whether the modifier and
    * format combination is supported only with external, or also with non-
    * external texture targets respectively.  The \p external_only parameter is
    * not used when the function returns false.
    *
    * \return true if the format+modifier pair is supported on \p screen, false
    *         otherwise.
    */
   bool (*is_dmabuf_modifier_supported)(struct pipe_screen *screen,
                                        uint64_t modifier, enum pipe_format,
                                        bool *external_only);

   /**
    * Get the number of planes required for a given modifier/format pair.
    *
    * If not NULL, this function returns the number of planes needed to
    * represent \p format in the layout specified by \p modifier, including
    * any driver-specific auxiliary data planes.
    *
    * Must only be called on a modifier supported by the screen for the
    * specified format.
    *
    * If NULL, no auxiliary planes are required for any modifier+format pairs
    * supported by \p screen.  Hence, the plane count can be derived directly
    * from \p format.
    *
    * \return Number of planes needed to store image data in the layout defined
    *         by \p format and \p modifier.
    */
   unsigned int (*get_dmabuf_modifier_planes)(struct pipe_screen *screen,
                                              uint64_t modifier,
                                              enum pipe_format format);

   /**
    * Get supported page sizes for sparse texture.
    *
    * \p size is the array size of \p x, \p y and \p z.
    *
    * \p offset sets an offset into the possible format page size array,
    *  used to pick a specific xyz size combination.
    *
    * \return Number of supported page sizes, 0 means not support.
    */
   int (*get_sparse_texture_virtual_page_size)(struct pipe_screen *screen,
                                               enum pipe_texture_target target,
                                               bool multi_sample,
                                               enum pipe_format format,
                                               unsigned offset, unsigned size,
                                               int *x, int *y, int *z);

   /**
    * Vertex state CSO functions for precomputing vertex and index buffer
    * states for display lists.
    */
   pipe_create_vertex_state_func create_vertex_state;
   pipe_vertex_state_destroy_func vertex_state_destroy;

   /**
    * Update a timeline semaphore value stored within a driver fence object.
    * Future waits and signals will use the new value.
    */
   void (*set_fence_timeline_value)(struct pipe_screen *screen,
                                    struct pipe_fence_handle *fence,
                                    uint64_t value);

   /**
    * Get additional data for interop_query_device_info
    *
    * \p in_data_size is how much data was allocated by the caller
    * \p data is the buffer to fill
    *
    * \return how much data was written
    */
   uint32_t (*interop_query_device_info)(struct pipe_screen *screen,
                                         uint32_t in_data_size,
                                         void *data);

   /**
    * Get additional data for interop_export_object
    *
    * \p in_data_size is how much data was allocated by the caller
    * \p data is the buffer to fill
    * \p need_export_dmabuf can be set to false to prevent
    *    a following call to resource_get_handle, if the private
    *    data contains the exported data
    *
    * \return how much data was written
    */
   uint32_t (*interop_export_object)(struct pipe_screen *screen,
                                     struct pipe_resource *res,
                                     uint32_t in_data_size,
                                     void *data,
                                     bool *need_export_dmabuf);
};


/**
 * Global configuration options for screen creation.
 */
struct pipe_screen_config {
   struct driOptionCache *options;
   const struct driOptionCache *options_info;
};


#ifdef __cplusplus
}
#endif

#endif /* P_SCREEN_H */
