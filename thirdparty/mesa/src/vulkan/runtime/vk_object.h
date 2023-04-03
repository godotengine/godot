/*
 * Copyright © 2020 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#ifndef VK_OBJECT_H
#define VK_OBJECT_H

#include <vulkan/vulkan.h>
#include <vulkan/vk_icd.h>

#include "c11/threads.h"
#include "util/macros.h"
#include "util/sparse_array.h"

#ifdef __cplusplus
extern "C" {
#endif

struct hash_table;

struct vk_device;

/** Base struct for all Vulkan objects */
struct vk_object_base {
   VK_LOADER_DATA _loader_data;

   /** Type of this object
    *
    * This is used for runtime type checking when casting to and from Vulkan
    * handle types since compile-time type checking doesn't always work.
    */
   VkObjectType type;

   /** Pointer to the device in which this object exists, if any
    *
    * This is NULL for instances and physical devices but should point to a
    * valid vk_device for almost everything else.  (There are a few WSI
    * objects that don't inherit from a device.)
    */
   struct vk_device *device;

   /* True if this object is fully constructed and visible to the client */
   bool client_visible;

   /* For VK_EXT_private_data */
   struct util_sparse_array private_data;

   /* VK_EXT_debug_utils */
   char *object_name;
};

/** Initialize a vk_base_object
 *
 * @param[in]  device   The vk_device this object was created from or NULL
 * @param[out] base     The vk_object_base to initialize
 * @param[in]  obj_type The VkObjectType of the object being initialized
 */
void vk_object_base_init(struct vk_device *device,
                         struct vk_object_base *base,
                         VkObjectType obj_type);

/** Tear down a vk_object_base
 *
 * @param[out] base     The vk_object_base being torn down
 */
void vk_object_base_finish(struct vk_object_base *base);

/** Recycles a vk_object_base
 *
 * This should be called when an object is recycled and handed back to the
 * client as if it were a new object.  When it's called is not important as
 * long as it's called between when the client thinks the object was destroyed
 * and when the client sees it again as a supposedly new object.
 *
 * @param[inout] base   The vk_object_base being recycled
 */
void vk_object_base_recycle(struct vk_object_base *base);

static inline void
vk_object_base_assert_valid(ASSERTED struct vk_object_base *base,
                            ASSERTED VkObjectType obj_type)
{
   assert(base == NULL || base->type == obj_type);
}

static inline struct vk_object_base *
vk_object_base_from_u64_handle(uint64_t handle, VkObjectType obj_type)
{
   struct vk_object_base *base = (struct vk_object_base *)(uintptr_t)handle;
   vk_object_base_assert_valid(base, obj_type);
   return base;
}

/** Define handle cast macros for the given dispatchable handle type
 *
 * For a given `driver_struct`, this defines `driver_struct_to_handle()` and
 * `driver_struct_from_handle()` helpers which provide type-safe (as much as
 * possible with Vulkan handle types) casts to and from the `driver_struct`
 * type.  As an added layer of protection, these casts use the provided
 * `VkObjectType` to assert that the object is of the correct type when
 * running with a debug build.
 *
 * @param __driver_type The name of the driver struct; it is assumed this is
 *                      the name of a struct type and `struct` will be
 *                      prepended automatically
 *
 * @param __base        The name of the vk_base_object member
 *
 * @param __VkType      The Vulkan object type such as VkImage
 *
 * @param __VK_TYPE     The VkObjectType corresponding to __VkType, such as
 *                      VK_OBJECT_TYPE_IMAGE
 */
#define VK_DEFINE_HANDLE_CASTS(__driver_type, __base, __VkType, __VK_TYPE) \
   static inline struct __driver_type *                                    \
   __driver_type ## _from_handle(__VkType _handle)                         \
   {                                                                       \
      struct vk_object_base *base = (struct vk_object_base *)_handle;      \
      vk_object_base_assert_valid(base, __VK_TYPE);                        \
      STATIC_ASSERT(offsetof(struct __driver_type, __base) == 0);          \
      return (struct __driver_type *) base;                                \
   }                                                                       \
                                                                           \
   static inline __VkType                                                  \
   __driver_type ## _to_handle(struct __driver_type *_obj)                 \
   {                                                                       \
      vk_object_base_assert_valid(&_obj->__base, __VK_TYPE);               \
      if (_obj != NULL)                                                    \
         _obj->__base.client_visible = true;                               \
      return (__VkType) _obj;                                              \
   }

/** Define handle cast macros for the given non-dispatchable handle type
 *
 * For a given `driver_struct`, this defines `driver_struct_to_handle()` and
 * `driver_struct_from_handle()` helpers which provide type-safe (as much as
 * possible with Vulkan handle types) casts to and from the `driver_struct`
 * type.  As an added layer of protection, these casts use the provided
 * `VkObjectType` to assert that the object is of the correct type when
 * running with a debug build.
 *
 * @param __driver_type The name of the driver struct; it is assumed this is
 *                      the name of a struct type and `struct` will be
 *                      prepended automatically
 *
 * @param __base        The name of the vk_base_object member
 *
 * @param __VkType      The Vulkan object type such as VkImage
 *
 * @param __VK_TYPE     The VkObjectType corresponding to __VkType, such as
 *                      VK_OBJECT_TYPE_IMAGE
 */
#define VK_DEFINE_NONDISP_HANDLE_CASTS(__driver_type, __base, __VkType, __VK_TYPE) \
   UNUSED static inline struct __driver_type *                             \
   __driver_type ## _from_handle(__VkType _handle)                         \
   {                                                                       \
      struct vk_object_base *base =                                        \
         (struct vk_object_base *)(uintptr_t)_handle;                      \
      vk_object_base_assert_valid(base, __VK_TYPE);                        \
      STATIC_ASSERT(offsetof(struct __driver_type, __base) == 0);          \
      return (struct __driver_type *)base;                                 \
   }                                                                       \
                                                                           \
   UNUSED static inline __VkType                                           \
   __driver_type ## _to_handle(struct __driver_type *_obj)                 \
   {                                                                       \
      vk_object_base_assert_valid(&_obj->__base, __VK_TYPE);               \
      if (_obj != NULL)                                                    \
         _obj->__base.client_visible = true;                               \
      return (__VkType)(uintptr_t) _obj;                                   \
   }

/** Declares a __driver_type pointer which represents __handle
 *
 * @param __driver_type The name of the driver struct; it is assumed this is
 *                      the name of a struct type and `struct` will be
 *                      prepended automatically
 *
 * @param __name        The name of the declared pointer
 *
 * @param __handle      The Vulkan object handle with which to initialize
 *                      `__name`
 */
#define VK_FROM_HANDLE(__driver_type, __name, __handle) \
   struct __driver_type *__name = __driver_type ## _from_handle(__handle)

/* Helpers for vk object (de)allocation and (de)initialization */
void *
vk_object_alloc(struct vk_device *device,
                const VkAllocationCallbacks *alloc,
                size_t size,
                VkObjectType vk_obj_type);

void *
vk_object_zalloc(struct vk_device *device,
                const VkAllocationCallbacks *alloc,
                size_t size,
                VkObjectType vk_obj_type);

struct vk_multialloc;

void *
vk_object_multialloc(struct vk_device *device,
                     struct vk_multialloc *ma,
                     const VkAllocationCallbacks *alloc,
                     VkObjectType vk_obj_type);

void *
vk_object_multizalloc(struct vk_device *device,
                      struct vk_multialloc *ma,
                      const VkAllocationCallbacks *alloc,
                      VkObjectType vk_obj_type);

void
vk_object_free(struct vk_device *device,
               const VkAllocationCallbacks *alloc,
               void *data);


struct vk_private_data_slot {
   struct vk_object_base base;
   uint32_t index;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(vk_private_data_slot, base,
                               VkPrivateDataSlot,
                               VK_OBJECT_TYPE_PRIVATE_DATA_SLOT);

VkResult
vk_private_data_slot_create(struct vk_device *device,
                            const VkPrivateDataSlotCreateInfo* pCreateInfo,
                            const VkAllocationCallbacks* pAllocator,
                            VkPrivateDataSlot* pPrivateDataSlot);
void
vk_private_data_slot_destroy(struct vk_device *device,
                             VkPrivateDataSlot privateDataSlot,
                             const VkAllocationCallbacks *pAllocator);
VkResult
vk_object_base_set_private_data(struct vk_device *device,
                                VkObjectType objectType,
                                uint64_t objectHandle,
                                VkPrivateDataSlot privateDataSlot,
                                uint64_t data);
void
vk_object_base_get_private_data(struct vk_device *device,
                                VkObjectType objectType,
                                uint64_t objectHandle,
                                VkPrivateDataSlot privateDataSlot,
                                uint64_t *pData);

const char *
vk_object_base_name(struct vk_object_base *obj);

#ifdef __cplusplus
}
#endif

#endif /* VK_OBJECT_H */
