/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_TYPE_H
#define SPA_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>

/** \defgroup spa_types Types
 * Data type information enumerations
 */

/**
 * \addtogroup spa_types
 * \{
 */

enum {
	/* Basic types */
	SPA_TYPE_START = 0x00000,
	SPA_TYPE_None,
	SPA_TYPE_Bool,
	SPA_TYPE_Id,
	SPA_TYPE_Int,
	SPA_TYPE_Long,
	SPA_TYPE_Float,
	SPA_TYPE_Double,
	SPA_TYPE_String,
	SPA_TYPE_Bytes,
	SPA_TYPE_Rectangle,
	SPA_TYPE_Fraction,
	SPA_TYPE_Bitmap,
	SPA_TYPE_Array,
	SPA_TYPE_Struct,
	SPA_TYPE_Object,
	SPA_TYPE_Sequence,
	SPA_TYPE_Pointer,
	SPA_TYPE_Fd,
	SPA_TYPE_Choice,
	SPA_TYPE_Pod,
	_SPA_TYPE_LAST,				/**< not part of ABI */

	/* Pointers */
	SPA_TYPE_POINTER_START = 0x10000,
	SPA_TYPE_POINTER_Buffer,
	SPA_TYPE_POINTER_Meta,
	SPA_TYPE_POINTER_Dict,
	_SPA_TYPE_POINTER_LAST,			/**< not part of ABI */

	/* Events */
	SPA_TYPE_EVENT_START = 0x20000,
	SPA_TYPE_EVENT_Device,
	SPA_TYPE_EVENT_Node,
	_SPA_TYPE_EVENT_LAST,			/**< not part of ABI */

	/* Commands */
	SPA_TYPE_COMMAND_START = 0x30000,
	SPA_TYPE_COMMAND_Device,
	SPA_TYPE_COMMAND_Node,
	_SPA_TYPE_COMMAND_LAST,			/**< not part of ABI */

	/* Objects */
	SPA_TYPE_OBJECT_START = 0x40000,
	SPA_TYPE_OBJECT_PropInfo,
	SPA_TYPE_OBJECT_Props,
	SPA_TYPE_OBJECT_Format,
	SPA_TYPE_OBJECT_ParamBuffers,
	SPA_TYPE_OBJECT_ParamMeta,
	SPA_TYPE_OBJECT_ParamIO,
	SPA_TYPE_OBJECT_ParamProfile,
	SPA_TYPE_OBJECT_ParamPortConfig,
	SPA_TYPE_OBJECT_ParamRoute,
	SPA_TYPE_OBJECT_Profiler,
	SPA_TYPE_OBJECT_ParamLatency,
	SPA_TYPE_OBJECT_ParamProcessLatency,
	SPA_TYPE_OBJECT_ParamTag,
	_SPA_TYPE_OBJECT_LAST,			/**< not part of ABI */

	/* vendor extensions */
	SPA_TYPE_VENDOR_PipeWire	= 0x02000000,

	SPA_TYPE_VENDOR_Other		= 0x7f000000,
};

#define SPA_TYPE_INFO_BASE			"Spa:"

#define SPA_TYPE_INFO_Flags			SPA_TYPE_INFO_BASE "Flags"
#define SPA_TYPE_INFO_FLAGS_BASE		SPA_TYPE_INFO_Flags ":"

#define SPA_TYPE_INFO_Enum			SPA_TYPE_INFO_BASE "Enum"
#define SPA_TYPE_INFO_ENUM_BASE			SPA_TYPE_INFO_Enum ":"

#define SPA_TYPE_INFO_Pod			SPA_TYPE_INFO_BASE "Pod"
#define SPA_TYPE_INFO_POD_BASE			SPA_TYPE_INFO_Pod ":"

#define SPA_TYPE_INFO_Struct			SPA_TYPE_INFO_POD_BASE "Struct"
#define SPA_TYPE_INFO_STRUCT_BASE		SPA_TYPE_INFO_Struct ":"

#define SPA_TYPE_INFO_Object			SPA_TYPE_INFO_POD_BASE "Object"
#define SPA_TYPE_INFO_OBJECT_BASE		SPA_TYPE_INFO_Object ":"

#define SPA_TYPE_INFO_Pointer			SPA_TYPE_INFO_BASE "Pointer"
#define SPA_TYPE_INFO_POINTER_BASE		SPA_TYPE_INFO_Pointer ":"

#define SPA_TYPE_INFO_Interface			SPA_TYPE_INFO_POINTER_BASE "Interface"
#define SPA_TYPE_INFO_INTERFACE_BASE		SPA_TYPE_INFO_Interface ":"

#define SPA_TYPE_INFO_Event			SPA_TYPE_INFO_OBJECT_BASE "Event"
#define SPA_TYPE_INFO_EVENT_BASE		SPA_TYPE_INFO_Event ":"

#define SPA_TYPE_INFO_Command			SPA_TYPE_INFO_OBJECT_BASE "Command"
#define SPA_TYPE_INFO_COMMAND_BASE		SPA_TYPE_INFO_Command ":"

struct spa_type_info {
	uint32_t type;
	uint32_t parent;
	const char *name;
	const struct spa_type_info *values;
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_TYPE_H */
