/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_BUFFERS_TYPES_H
#define SPA_PARAM_BUFFERS_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param-types.h>
#include <spa/node/type-info.h>

#include <spa/param/buffers.h>

#define SPA_TYPE_INFO_PARAM_Meta		SPA_TYPE_INFO_PARAM_BASE "Meta"
#define SPA_TYPE_INFO_PARAM_META_BASE		SPA_TYPE_INFO_PARAM_Meta ":"

static const struct spa_type_info spa_type_param_meta[] = {
	{ SPA_PARAM_META_START, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_META_BASE, spa_type_param },
	{ SPA_PARAM_META_type, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_META_BASE "type", spa_type_meta_type },
	{ SPA_PARAM_META_size, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_META_BASE "size", NULL },
	{ 0, 0, NULL, NULL },
};

/** Base for parameters that describe IO areas to exchange data,
 * control and properties with a node.
 */
#define SPA_TYPE_INFO_PARAM_IO		SPA_TYPE_INFO_PARAM_BASE "IO"
#define SPA_TYPE_INFO_PARAM_IO_BASE		SPA_TYPE_INFO_PARAM_IO ":"

static const struct spa_type_info spa_type_param_io[] = {
	{ SPA_PARAM_IO_START, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_IO_BASE, spa_type_param, },
	{ SPA_PARAM_IO_id, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_IO_BASE "id", spa_type_io },
	{ SPA_PARAM_IO_size, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_IO_BASE "size", NULL },
	{ 0, 0, NULL, NULL },
};

#define SPA_TYPE_INFO_PARAM_Buffers			SPA_TYPE_INFO_PARAM_BASE "Buffers"
#define SPA_TYPE_INFO_PARAM_BUFFERS_BASE		SPA_TYPE_INFO_PARAM_Buffers ":"

#define SPA_TYPE_INFO_PARAM_BlockInfo			SPA_TYPE_INFO_PARAM_BUFFERS_BASE "BlockInfo"
#define SPA_TYPE_INFO_PARAM_BLOCK_INFO_BASE		SPA_TYPE_INFO_PARAM_BlockInfo ":"

static const struct spa_type_info spa_type_param_buffers[] = {
	{ SPA_PARAM_BUFFERS_START,    SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_BUFFERS_BASE, spa_type_param, },
	{ SPA_PARAM_BUFFERS_buffers,  SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BUFFERS_BASE "buffers", NULL },
	{ SPA_PARAM_BUFFERS_blocks,   SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BUFFERS_BASE "blocks", NULL },
	{ SPA_PARAM_BUFFERS_size,     SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BLOCK_INFO_BASE "size", NULL },
	{ SPA_PARAM_BUFFERS_stride,   SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BLOCK_INFO_BASE "stride", NULL },
	{ SPA_PARAM_BUFFERS_align,    SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BLOCK_INFO_BASE "align", NULL },
	{ SPA_PARAM_BUFFERS_dataType, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BLOCK_INFO_BASE "dataType", NULL },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_BUFFERS_TYPES_H */
