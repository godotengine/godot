/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_TYPE_INFO_H
#define SPA_TYPE_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>

/**
 * \addtogroup spa_types
 * \{
 */

#ifndef SPA_TYPE_ROOT
#define SPA_TYPE_ROOT	spa_types
#endif

static inline bool spa_type_is_a(const char *type, const char *parent)
{
	return type != NULL && parent != NULL && strncmp(type, parent, strlen(parent)) == 0;
}

#include <spa/utils/type.h>
#include <spa/utils/enum-types.h>

#include <spa/monitor/type-info.h>
#include <spa/node/type-info.h>
#include <spa/param/type-info.h>
#include <spa/control/type-info.h>

static const struct spa_type_info spa_types[] = {
        /* Basic types */
	{ SPA_TYPE_START, SPA_TYPE_START, SPA_TYPE_INFO_BASE, NULL },
	{ SPA_TYPE_None, SPA_TYPE_None, SPA_TYPE_INFO_BASE "None", NULL },
	{ SPA_TYPE_Bool, SPA_TYPE_Bool, SPA_TYPE_INFO_BASE "Bool", NULL },
	{ SPA_TYPE_Id, SPA_TYPE_Int, SPA_TYPE_INFO_BASE "Id", NULL },
	{ SPA_TYPE_Int, SPA_TYPE_Int, SPA_TYPE_INFO_BASE "Int", NULL },
	{ SPA_TYPE_Long, SPA_TYPE_Long, SPA_TYPE_INFO_BASE "Long", NULL },
	{ SPA_TYPE_Float, SPA_TYPE_Float, SPA_TYPE_INFO_BASE "Float", NULL },
	{ SPA_TYPE_Double, SPA_TYPE_Double, SPA_TYPE_INFO_BASE "Double", NULL },
	{ SPA_TYPE_String, SPA_TYPE_String, SPA_TYPE_INFO_BASE "String", NULL },
	{ SPA_TYPE_Bytes, SPA_TYPE_Bytes, SPA_TYPE_INFO_BASE "Bytes", NULL },
	{ SPA_TYPE_Rectangle, SPA_TYPE_Rectangle, SPA_TYPE_INFO_BASE "Rectangle", NULL },
	{ SPA_TYPE_Fraction, SPA_TYPE_Fraction, SPA_TYPE_INFO_BASE "Fraction", NULL },
	{ SPA_TYPE_Bitmap, SPA_TYPE_Bitmap, SPA_TYPE_INFO_BASE "Bitmap", NULL },
	{ SPA_TYPE_Array, SPA_TYPE_Array, SPA_TYPE_INFO_BASE "Array", NULL },
	{ SPA_TYPE_Pod, SPA_TYPE_Pod, SPA_TYPE_INFO_Pod, NULL },
	{ SPA_TYPE_Struct, SPA_TYPE_Pod, SPA_TYPE_INFO_Struct, NULL },
	{ SPA_TYPE_Object, SPA_TYPE_Pod, SPA_TYPE_INFO_Object, NULL },
	{ SPA_TYPE_Sequence, SPA_TYPE_Pod, SPA_TYPE_INFO_POD_BASE "Sequence", NULL },
	{ SPA_TYPE_Pointer, SPA_TYPE_Pointer, SPA_TYPE_INFO_Pointer, NULL },
	{ SPA_TYPE_Fd, SPA_TYPE_Fd, SPA_TYPE_INFO_BASE "Fd", NULL },
	{ SPA_TYPE_Choice, SPA_TYPE_Pod, SPA_TYPE_INFO_POD_BASE "Choice", NULL },

	{ SPA_TYPE_POINTER_START, SPA_TYPE_Pointer, SPA_TYPE_INFO_Pointer, NULL },
	{ SPA_TYPE_POINTER_Buffer, SPA_TYPE_Pointer, SPA_TYPE_INFO_POINTER_BASE "Buffer", NULL },
	{ SPA_TYPE_POINTER_Meta, SPA_TYPE_Pointer, SPA_TYPE_INFO_POINTER_BASE "Meta", NULL },
	{ SPA_TYPE_POINTER_Dict, SPA_TYPE_Pointer, SPA_TYPE_INFO_POINTER_BASE "Dict", NULL },

	{ SPA_TYPE_EVENT_START, SPA_TYPE_Object, SPA_TYPE_INFO_Event, NULL },
	{ SPA_TYPE_EVENT_Device, SPA_TYPE_Object, SPA_TYPE_INFO_EVENT_BASE "Device", spa_type_device_event },
	{ SPA_TYPE_EVENT_Node, SPA_TYPE_Object, SPA_TYPE_INFO_EVENT_BASE "Node", spa_type_node_event },

	{ SPA_TYPE_COMMAND_START, SPA_TYPE_Object, SPA_TYPE_INFO_Command, NULL },
	{ SPA_TYPE_COMMAND_Device, SPA_TYPE_Object, SPA_TYPE_INFO_COMMAND_BASE "Device", NULL },
	{ SPA_TYPE_COMMAND_Node, SPA_TYPE_Object, SPA_TYPE_INFO_COMMAND_BASE "Node", spa_type_node_command },

	{ SPA_TYPE_OBJECT_START, SPA_TYPE_Object, SPA_TYPE_INFO_Object, NULL },
	{ SPA_TYPE_OBJECT_PropInfo, SPA_TYPE_Object, SPA_TYPE_INFO_PropInfo, spa_type_prop_info, },
	{ SPA_TYPE_OBJECT_Props, SPA_TYPE_Object, SPA_TYPE_INFO_Props, spa_type_props },
	{ SPA_TYPE_OBJECT_Format, SPA_TYPE_Object, SPA_TYPE_INFO_Format, spa_type_format },
	{ SPA_TYPE_OBJECT_ParamBuffers, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_Buffers, spa_type_param_buffers, },
	{ SPA_TYPE_OBJECT_ParamMeta, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_Meta, spa_type_param_meta },
	{ SPA_TYPE_OBJECT_ParamIO, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_IO, spa_type_param_io },
	{ SPA_TYPE_OBJECT_ParamProfile, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_Profile, spa_type_param_profile },
	{ SPA_TYPE_OBJECT_ParamPortConfig, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_PortConfig, spa_type_param_port_config },
	{ SPA_TYPE_OBJECT_ParamRoute, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_Route, spa_type_param_route },
	{ SPA_TYPE_OBJECT_Profiler, SPA_TYPE_Object, SPA_TYPE_INFO_Profiler, spa_type_profiler },
	{ SPA_TYPE_OBJECT_ParamLatency, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_Latency, spa_type_param_latency },
	{ SPA_TYPE_OBJECT_ParamProcessLatency, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_ProcessLatency, spa_type_param_process_latency },
	{ SPA_TYPE_OBJECT_ParamTag, SPA_TYPE_Object, SPA_TYPE_INFO_PARAM_Tag, spa_type_param_tag },

	{ 0, 0, NULL, NULL }
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_TYPE_INFO_H */
