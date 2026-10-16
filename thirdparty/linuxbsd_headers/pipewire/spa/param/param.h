/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_H
#define SPA_PARAM_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup spa_param Parameters
 * Parameter value enumerations and type information
 */

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/utils/defs.h>

/** different parameter types that can be queried */
enum spa_param_type {
	SPA_PARAM_Invalid,		/**< invalid */
	SPA_PARAM_PropInfo,		/**< property information as SPA_TYPE_OBJECT_PropInfo */
	SPA_PARAM_Props,		/**< properties as SPA_TYPE_OBJECT_Props */
	SPA_PARAM_EnumFormat,		/**< available formats as SPA_TYPE_OBJECT_Format */
	SPA_PARAM_Format,		/**< configured format as SPA_TYPE_OBJECT_Format */
	SPA_PARAM_Buffers,		/**< buffer configurations as SPA_TYPE_OBJECT_ParamBuffers*/
	SPA_PARAM_Meta,			/**< allowed metadata for buffers as SPA_TYPE_OBJECT_ParamMeta*/
	SPA_PARAM_IO,			/**< configurable IO areas as SPA_TYPE_OBJECT_ParamIO */
	SPA_PARAM_EnumProfile,		/**< profile enumeration as SPA_TYPE_OBJECT_ParamProfile */
	SPA_PARAM_Profile,		/**< profile configuration as SPA_TYPE_OBJECT_ParamProfile */
	SPA_PARAM_EnumPortConfig,	/**< port configuration enumeration as SPA_TYPE_OBJECT_ParamPortConfig */
	SPA_PARAM_PortConfig,		/**< port configuration as SPA_TYPE_OBJECT_ParamPortConfig */
	SPA_PARAM_EnumRoute,		/**< routing enumeration as SPA_TYPE_OBJECT_ParamRoute */
	SPA_PARAM_Route,		/**< routing configuration as SPA_TYPE_OBJECT_ParamRoute */
	SPA_PARAM_Control,		/**< Control parameter, a SPA_TYPE_Sequence */
	SPA_PARAM_Latency,		/**< latency reporting, a SPA_TYPE_OBJECT_ParamLatency */
	SPA_PARAM_ProcessLatency,	/**< processing latency, a SPA_TYPE_OBJECT_ParamProcessLatency */
	SPA_PARAM_Tag,			/**< tag reporting, a SPA_TYPE_OBJECT_ParamTag. Since 0.3.79 */
};

/** information about a parameter */
struct spa_param_info {
	uint32_t id;			/**< enum spa_param_type */
#define SPA_PARAM_INFO_SERIAL		(1<<0)	/**< bit to signal update even when the
						 *   read/write flags don't change */
#define SPA_PARAM_INFO_READ		(1<<1)
#define SPA_PARAM_INFO_WRITE		(1<<2)
#define SPA_PARAM_INFO_READWRITE	(SPA_PARAM_INFO_WRITE|SPA_PARAM_INFO_READ)
	uint32_t flags;
	uint32_t user;			/**< private user field. You can use this to keep
					  *  state. */
	int32_t seq;			/**< private seq field. You can use this to keep
					  *  state of a pending update. */
	uint32_t padding[4];
};

#define SPA_PARAM_INFO(id,flags) ((struct spa_param_info){ (id), (flags) })

enum spa_param_bitorder {
	SPA_PARAM_BITORDER_unknown,	/**< unknown bitorder */
	SPA_PARAM_BITORDER_msb,		/**< most significant bit */
	SPA_PARAM_BITORDER_lsb,		/**< least significant bit */
};

enum spa_param_availability {
	SPA_PARAM_AVAILABILITY_unknown,	/**< unknown availability */
	SPA_PARAM_AVAILABILITY_no,	/**< not available */
	SPA_PARAM_AVAILABILITY_yes,	/**< available */
};

#include <spa/param/buffers.h>
#include <spa/param/profile.h>
#include <spa/param/port-config.h>
#include <spa/param/route.h>

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_H */
