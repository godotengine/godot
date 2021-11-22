/*
** Copyright (c) 2019-2021 The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

#ifndef VULKAN_VIDEO_CODEC_COMMON_H_
#define VULKAN_VIDEO_CODEC_COMMON_H_ 1

#ifdef __cplusplus
extern "C" {
#endif

#define VK_MAKE_VIDEO_STD_VERSION(major, minor, patch) \
    ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))

#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODEC_COMMON_H_
