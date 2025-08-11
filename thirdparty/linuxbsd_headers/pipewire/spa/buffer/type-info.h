/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_BUFFER_TYPES_H
#define SPA_BUFFER_TYPES_H

/**
 * \addtogroup spa_buffer
 * \{
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/buffer/buffer.h>
#include <spa/buffer/meta.h>
#include <spa/utils/type.h>

#define SPA_TYPE_INFO_Buffer			SPA_TYPE_INFO_POINTER_BASE "Buffer"
#define SPA_TYPE_INFO_BUFFER_BASE		SPA_TYPE_INFO_Buffer ":"

/** Buffers contain data of a certain type */
#define SPA_TYPE_INFO_Data			SPA_TYPE_INFO_ENUM_BASE "Data"
#define SPA_TYPE_INFO_DATA_BASE			SPA_TYPE_INFO_Data ":"

/** base type for fd based memory */
#define SPA_TYPE_INFO_DATA_Fd			SPA_TYPE_INFO_DATA_BASE "Fd"
#define SPA_TYPE_INFO_DATA_FD_BASE		SPA_TYPE_INFO_DATA_Fd ":"

static const struct spa_type_info spa_type_data_type[] = {
	{ SPA_DATA_Invalid, SPA_TYPE_Int, SPA_TYPE_INFO_DATA_BASE "Invalid", NULL },
	{ SPA_DATA_MemPtr, SPA_TYPE_Int, SPA_TYPE_INFO_DATA_BASE "MemPtr", NULL },
	{ SPA_DATA_MemFd, SPA_TYPE_Int, SPA_TYPE_INFO_DATA_FD_BASE "MemFd", NULL },
	{ SPA_DATA_DmaBuf, SPA_TYPE_Int, SPA_TYPE_INFO_DATA_FD_BASE "DmaBuf", NULL },
	{ SPA_DATA_MemId, SPA_TYPE_Int, SPA_TYPE_INFO_DATA_BASE "MemId", NULL },
	{ 0, 0, NULL, NULL },
};

#define SPA_TYPE_INFO_Meta			SPA_TYPE_INFO_POINTER_BASE "Meta"
#define SPA_TYPE_INFO_META_BASE			SPA_TYPE_INFO_Meta ":"

#define SPA_TYPE_INFO_META_Array		SPA_TYPE_INFO_META_BASE "Array"
#define SPA_TYPE_INFO_META_ARRAY_BASE		SPA_TYPE_INFO_META_Array ":"

#define SPA_TYPE_INFO_META_Region		SPA_TYPE_INFO_META_BASE "Region"
#define SPA_TYPE_INFO_META_REGION_BASE		SPA_TYPE_INFO_META_Region ":"

#define SPA_TYPE_INFO_META_ARRAY_Region		SPA_TYPE_INFO_META_ARRAY_BASE "Region"
#define SPA_TYPE_INFO_META_ARRAY_REGION_BASE	SPA_TYPE_INFO_META_ARRAY_Region ":"

static const struct spa_type_info spa_type_meta_type[] = {
	{ SPA_META_Invalid, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_BASE "Invalid", NULL },
	{ SPA_META_Header, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_BASE "Header", NULL },
	{ SPA_META_VideoCrop, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_REGION_BASE "VideoCrop", NULL },
	{ SPA_META_VideoDamage, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_ARRAY_REGION_BASE "VideoDamage", NULL },
	{ SPA_META_Bitmap, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_BASE "Bitmap", NULL },
	{ SPA_META_Cursor, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_BASE "Cursor", NULL },
	{ SPA_META_Control, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_BASE "Control", NULL },
	{ SPA_META_Busy, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_BASE "Busy", NULL },
	{ SPA_META_VideoTransform, SPA_TYPE_Pointer, SPA_TYPE_INFO_META_BASE "VideoTransform", NULL },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_BUFFER_TYPES_H */
