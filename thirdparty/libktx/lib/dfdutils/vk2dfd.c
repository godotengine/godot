/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* Copyright 2019-2020 Mark Callow
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @~English
 * @brief Create a DFD for a VkFormat.
 */

#include "dfd.h"

/**
 * @~English
 * @brief Create a DFD matching a VkFormat.
 *
 * @param[in] format    VkFormat for which to create a DFD.
 *
 * @return      pointer to the created DFD or 0 if format not supported or
 *              unrecognized. Caller is responsible for freeing the created
 *              DFD.
 */
uint32_t*
vk2dfd(enum VkFormat format)
 {
     switch (format) {
#include "vk2dfd.inl"
         default: return 0;
     }
 }

