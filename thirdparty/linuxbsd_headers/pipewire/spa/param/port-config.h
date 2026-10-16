/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PORT_CONFIG_H
#define SPA_PARAM_PORT_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param.h>

enum spa_param_port_config_mode {
	SPA_PARAM_PORT_CONFIG_MODE_none,	/**< no configuration */
	SPA_PARAM_PORT_CONFIG_MODE_passthrough,	/**< passthrough configuration */
	SPA_PARAM_PORT_CONFIG_MODE_convert,	/**< convert configuration */
	SPA_PARAM_PORT_CONFIG_MODE_dsp,		/**< dsp configuration, depending on the external
						  *  format. For audio, ports will be configured for
						  *  the given number of channels with F32 format. */
};

/** properties for SPA_TYPE_OBJECT_ParamPortConfig */
enum spa_param_port_config {
	SPA_PARAM_PORT_CONFIG_START,
	SPA_PARAM_PORT_CONFIG_direction,	/**< (Id enum spa_direction) direction */
	SPA_PARAM_PORT_CONFIG_mode,		/**< (Id enum spa_param_port_config_mode) mode */
	SPA_PARAM_PORT_CONFIG_monitor,		/**< (Bool) enable monitor output ports on input ports */
	SPA_PARAM_PORT_CONFIG_control,		/**< (Bool) enable control ports */
	SPA_PARAM_PORT_CONFIG_format,		/**< (Object) format filter */
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PORT_CONFIG_H */
