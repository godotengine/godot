/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_NODE_TYPES_H
#define SPA_NODE_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_node
 * \{
 */

#include <spa/utils/type.h>

#include <spa/node/command.h>
#include <spa/node/event.h>
#include <spa/node/io.h>

#define SPA_TYPE_INFO_IO			SPA_TYPE_INFO_ENUM_BASE "IO"
#define SPA_TYPE_INFO_IO_BASE		SPA_TYPE_INFO_IO ":"

static const struct spa_type_info spa_type_io[] = {
	{ SPA_IO_Invalid, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Invalid", NULL },
	{ SPA_IO_Buffers, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Buffers", NULL },
	{ SPA_IO_Range, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Range", NULL },
	{ SPA_IO_Clock, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Clock", NULL },
	{ SPA_IO_Latency, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Latency", NULL },
	{ SPA_IO_Control, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Control", NULL },
	{ SPA_IO_Notify, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Notify", NULL },
	{ SPA_IO_Position, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Position", NULL },
	{ SPA_IO_RateMatch, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "RateMatch", NULL },
	{ SPA_IO_Memory, SPA_TYPE_Int, SPA_TYPE_INFO_IO_BASE "Memory", NULL },
	{ 0, 0, NULL, NULL },
};

#define SPA_TYPE_INFO_NodeEvent			SPA_TYPE_INFO_EVENT_BASE "Node"
#define SPA_TYPE_INFO_NODE_EVENT_BASE		SPA_TYPE_INFO_NodeEvent ":"

static const struct spa_type_info spa_type_node_event_id[] = {
	{ SPA_NODE_EVENT_Error,		 SPA_TYPE_EVENT_Node, SPA_TYPE_INFO_NODE_EVENT_BASE "Error",   NULL },
	{ SPA_NODE_EVENT_Buffering,	 SPA_TYPE_EVENT_Node, SPA_TYPE_INFO_NODE_EVENT_BASE "Buffering", NULL },
	{ SPA_NODE_EVENT_RequestRefresh, SPA_TYPE_EVENT_Node, SPA_TYPE_INFO_NODE_EVENT_BASE "RequestRefresh", NULL },
	{ SPA_NODE_EVENT_RequestProcess, SPA_TYPE_EVENT_Node, SPA_TYPE_INFO_NODE_EVENT_BASE "RequestProcess", NULL },
	{ 0, 0, NULL, NULL },
};

static const struct spa_type_info spa_type_node_event[] = {
	{ SPA_EVENT_NODE_START, SPA_TYPE_Id, SPA_TYPE_INFO_NODE_EVENT_BASE, spa_type_node_event_id },
	{ 0, 0, NULL, NULL },
};

#define SPA_TYPE_INFO_NodeCommand			SPA_TYPE_INFO_COMMAND_BASE "Node"
#define SPA_TYPE_INFO_NODE_COMMAND_BASE		SPA_TYPE_INFO_NodeCommand ":"

static const struct spa_type_info spa_type_node_command_id[] = {
	{ SPA_NODE_COMMAND_Suspend,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Suspend", NULL },
	{ SPA_NODE_COMMAND_Pause,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Pause",   NULL },
	{ SPA_NODE_COMMAND_Start,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Start",   NULL },
	{ SPA_NODE_COMMAND_Enable,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Enable",  NULL },
	{ SPA_NODE_COMMAND_Disable,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Disable", NULL },
	{ SPA_NODE_COMMAND_Flush,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Flush",   NULL },
	{ SPA_NODE_COMMAND_Drain,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Drain",   NULL },
	{ SPA_NODE_COMMAND_Marker,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "Marker",  NULL },
	{ SPA_NODE_COMMAND_ParamBegin,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "ParamBegin",  NULL },
	{ SPA_NODE_COMMAND_ParamEnd,	SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "ParamEnd",  NULL },
	{ SPA_NODE_COMMAND_RequestProcess, SPA_TYPE_COMMAND_Node, SPA_TYPE_INFO_NODE_COMMAND_BASE "RequestProcess",  NULL },
	{ 0, 0, NULL, NULL },
};

static const struct spa_type_info spa_type_node_command[] = {
	{ 0, SPA_TYPE_Id, SPA_TYPE_INFO_NODE_COMMAND_BASE, spa_type_node_command_id },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_NODE_TYPES_H */
