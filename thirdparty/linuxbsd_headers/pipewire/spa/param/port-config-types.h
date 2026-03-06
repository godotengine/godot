/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PORT_CONFIG_TYPES_H
#define SPA_PARAM_PORT_CONFIG_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/utils/enum-types.h>
#include <spa/param/param-types.h>
#include <spa/param/port-config.h>

#define SPA_TYPE_INFO_ParamPortConfigMode		SPA_TYPE_INFO_ENUM_BASE "ParamPortConfigMode"
#define SPA_TYPE_INFO_PARAM_PORT_CONFIG_MODE_BASE	SPA_TYPE_INFO_ParamPortConfigMode ":"

static const struct spa_type_info spa_type_param_port_config_mode[] = {
	{ SPA_PARAM_PORT_CONFIG_MODE_none, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_PORT_CONFIG_MODE_BASE "none", NULL },
	{ SPA_PARAM_PORT_CONFIG_MODE_passthrough, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_PORT_CONFIG_MODE_BASE "passthrough", NULL },
	{ SPA_PARAM_PORT_CONFIG_MODE_convert, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_PORT_CONFIG_MODE_BASE "convert", NULL },
	{ SPA_PARAM_PORT_CONFIG_MODE_dsp, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_PORT_CONFIG_MODE_BASE "dsp", NULL },
	{ 0, 0, NULL, NULL },
};

#define SPA_TYPE_INFO_PARAM_PortConfig		SPA_TYPE_INFO_PARAM_BASE "PortConfig"
#define SPA_TYPE_INFO_PARAM_PORT_CONFIG_BASE	SPA_TYPE_INFO_PARAM_PortConfig ":"

static const struct spa_type_info spa_type_param_port_config[] = {
	{ SPA_PARAM_PORT_CONFIG_START, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_PORT_CONFIG_BASE, spa_type_param, },
	{ SPA_PARAM_PORT_CONFIG_direction, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_PORT_CONFIG_BASE "direction", spa_type_direction, },
	{ SPA_PARAM_PORT_CONFIG_mode, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_PORT_CONFIG_BASE "mode", spa_type_param_port_config_mode },
	{ SPA_PARAM_PORT_CONFIG_monitor, SPA_TYPE_Bool, SPA_TYPE_INFO_PARAM_PORT_CONFIG_BASE "monitor", NULL },
	{ SPA_PARAM_PORT_CONFIG_control, SPA_TYPE_Bool, SPA_TYPE_INFO_PARAM_PORT_CONFIG_BASE "control", NULL },
	{ SPA_PARAM_PORT_CONFIG_format, SPA_TYPE_OBJECT_Format, SPA_TYPE_INFO_PARAM_PORT_CONFIG_BASE "format", NULL },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PORT_CONFIG_TYPES_H */
