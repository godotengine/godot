/*
 * Copyright © 2016 Intel Corporation
 * Copyright © 2019 Google LLC
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef U_FORMAT_VK_H
#define U_FORMAT_VK_H

#include <vulkan/vulkan_core.h>
#include "util/format/u_format.h"
#include "util/u_math.h"

#ifdef __cplusplus
extern "C" {
#endif

enum pipe_format
vk_format_to_pipe_format(enum VkFormat vkformat);

VkImageAspectFlags
vk_format_aspects(VkFormat format);

static inline bool
vk_format_is_color(VkFormat format)
{
   return vk_format_aspects(format) == VK_IMAGE_ASPECT_COLOR_BIT;
}

static inline bool
vk_format_is_depth_or_stencil(VkFormat format)
{
   const VkImageAspectFlags aspects = vk_format_aspects(format);
   return aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
}

static inline bool
vk_format_has_depth(VkFormat format)
{
   const VkImageAspectFlags aspects = vk_format_aspects(format);
   return aspects & VK_IMAGE_ASPECT_DEPTH_BIT;
}

static inline bool
vk_format_has_stencil(VkFormat format)
{
   const VkImageAspectFlags aspects = vk_format_aspects(format);
   return aspects & VK_IMAGE_ASPECT_STENCIL_BIT;
}

static inline VkFormat
vk_format_depth_only(VkFormat format)
{
   assert(vk_format_has_depth(format));
   switch (format) {
   case VK_FORMAT_D16_UNORM_S8_UINT:
      return VK_FORMAT_D16_UNORM;
   case VK_FORMAT_D24_UNORM_S8_UINT:
      return VK_FORMAT_X8_D24_UNORM_PACK32;
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return VK_FORMAT_D32_SFLOAT;
   default:
      return format;
   }
}

static inline VkFormat
vk_format_stencil_only(VkFormat format)
{
   assert(vk_format_has_stencil(format));
   return VK_FORMAT_S8_UINT;
}

void vk_component_mapping_to_pipe_swizzle(VkComponentMapping mapping,
                                          unsigned char out_swizzle[4]);

static inline bool
vk_format_is_int(VkFormat format)
{
   return util_format_is_pure_integer(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_is_sint(VkFormat format)
{
   return util_format_is_pure_sint(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_is_uint(VkFormat format)
{
   return util_format_is_pure_uint(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_is_unorm(VkFormat format)
{
   return util_format_is_unorm(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_is_snorm(VkFormat format)
{
   return util_format_is_snorm(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_is_float(VkFormat format)
{
   return util_format_is_float(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_is_srgb(VkFormat format)
{
   return util_format_is_srgb(vk_format_to_pipe_format(format));
}

static inline unsigned
vk_format_get_blocksize(VkFormat format)
{
   return util_format_get_blocksize(vk_format_to_pipe_format(format));
}

static inline unsigned
vk_format_get_blockwidth(VkFormat format)
{
   return util_format_get_blockwidth(vk_format_to_pipe_format(format));
}

static inline unsigned
vk_format_get_blockheight(VkFormat format)
{
   return util_format_get_blockheight(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_is_compressed(VkFormat format)
{
   /* this includes 4:2:2 formats, which are compressed formats for vulkan */
   return vk_format_get_blockwidth(format) > 1;
}

static inline bool
vk_format_is_block_compressed(VkFormat format)
{
   return util_format_is_compressed(vk_format_to_pipe_format(format));
}

static inline const struct util_format_description *
vk_format_description(VkFormat format)
{
   return util_format_description(vk_format_to_pipe_format(format));
}

static inline unsigned
vk_format_get_component_bits(VkFormat format, enum util_format_colorspace colorspace,
                             unsigned component)
{
   return util_format_get_component_bits(vk_format_to_pipe_format(format),
                                         colorspace,
                                         component);
}

static inline unsigned
vk_format_get_nr_components(VkFormat format)
{
   return util_format_get_nr_components(vk_format_to_pipe_format(format));
}

static inline bool
vk_format_has_alpha(VkFormat format)
{
   return util_format_has_alpha(vk_format_to_pipe_format(format));
}

static inline unsigned
vk_format_get_blocksizebits(VkFormat format)
{
   return util_format_get_blocksizebits(vk_format_to_pipe_format(format));
}

static inline unsigned
vk_format_get_plane_count(VkFormat format)
{
   return util_format_get_num_planes(vk_format_to_pipe_format(format));
}

VkFormat
vk_format_get_plane_format(VkFormat format, unsigned plane_id);

VkFormat
vk_format_get_aspect_format(VkFormat format, const VkImageAspectFlags aspect);

struct vk_format_ycbcr_plane {
   /* RGBA format for this plane */
   VkFormat format;

   /* Whether this plane contains chroma channels */
   bool has_chroma;

   /* For downscaling of YUV planes */
   uint8_t denominator_scales[2];

   /* How to map sampled ycbcr planes to a single 4 component element.
    *
    * We use uint8_t for compactness but it's actually VkComponentSwizzle.
    */
   uint8_t ycbcr_swizzle[4];
};

struct vk_format_ycbcr_info {
   uint8_t n_planes;
   struct vk_format_ycbcr_plane planes[3];
};

const struct vk_format_ycbcr_info *vk_format_get_ycbcr_info(VkFormat format);

#ifdef __cplusplus
}
#endif

#endif
