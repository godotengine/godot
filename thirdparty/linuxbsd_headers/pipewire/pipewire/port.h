/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_PORT_H
#define PIPEWIRE_PORT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <errno.h>

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>
#include <spa/param/param.h>

#include <pipewire/proxy.h>

/** \defgroup pw_port Port
 * Port interface
 */

/**
 * \addtogroup pw_port
 * \{
 */

#define PW_TYPE_INTERFACE_Port	PW_TYPE_INFO_INTERFACE_BASE "Port"

#define PW_PORT_PERM_MASK	PW_PERM_R|PW_PERM_X|PW_PERM_M

#define PW_VERSION_PORT		3
struct pw_port;

/** The direction of a port */
#define pw_direction spa_direction
#define PW_DIRECTION_INPUT SPA_DIRECTION_INPUT
#define PW_DIRECTION_OUTPUT SPA_DIRECTION_OUTPUT

/** Convert a \ref pw_direction to a readable string */
const char * pw_direction_as_string(enum pw_direction direction);

struct pw_port_info {
	uint32_t id;				/**< id of the global */
	enum pw_direction direction;		/**< port direction */
#define PW_PORT_CHANGE_MASK_PROPS		(1 << 0)
#define PW_PORT_CHANGE_MASK_PARAMS		(1 << 1)
#define PW_PORT_CHANGE_MASK_ALL			((1 << 2)-1)
	uint64_t change_mask;			/**< bitfield of changed fields since last call */
	struct spa_dict *props;			/**< the properties of the port */
	struct spa_param_info *params;		/**< parameters */
	uint32_t n_params;			/**< number of items in \a params */
};

struct pw_port_info *
pw_port_info_update(struct pw_port_info *info,
		const struct pw_port_info *update);

struct pw_port_info *
pw_port_info_merge(struct pw_port_info *info,
		const struct pw_port_info *update, bool reset);

void
pw_port_info_free(struct pw_port_info *info);

#define PW_PORT_EVENT_INFO	0
#define PW_PORT_EVENT_PARAM	1
#define PW_PORT_EVENT_NUM	2

/** Port events */
struct pw_port_events {
#define PW_VERSION_PORT_EVENTS	0
	uint32_t version;
	/**
	 * Notify port info
	 *
	 * \param info info about the port
	 */
	void (*info) (void *data, const struct pw_port_info *info);
	/**
	 * Notify a port param
	 *
	 * Event emitted as a result of the enum_params method.
	 *
	 * \param seq the sequence number of the request
	 * \param id the param id
	 * \param index the param index
	 * \param next the param index of the next param
	 * \param param the parameter
	 */
	void (*param) (void *data, int seq,
		       uint32_t id, uint32_t index, uint32_t next,
		       const struct spa_pod *param);
};

#define PW_PORT_METHOD_ADD_LISTENER	0
#define PW_PORT_METHOD_SUBSCRIBE_PARAMS	1
#define PW_PORT_METHOD_ENUM_PARAMS	2
#define PW_PORT_METHOD_NUM		3

/** Port methods */
struct pw_port_methods {
#define PW_VERSION_PORT_METHODS		0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_port_events *events,
			void *data);
	/**
	 * Subscribe to parameter changes
	 *
	 * Automatically emit param events for the given ids when
	 * they are changed.
	 *
	 * \param ids an array of param ids
	 * \param n_ids the number of ids in \a ids
	 *
	 * This requires X permissions on the port.
	 */
	int (*subscribe_params) (void *object, uint32_t *ids, uint32_t n_ids);

	/**
	 * Enumerate port parameters
	 *
	 * Start enumeration of port parameters. For each param, a
	 * param event will be emitted.
	 *
	 * \param seq a sequence number returned in the reply
	 * \param id the parameter id to enumerate
	 * \param start the start index or 0 for the first param
	 * \param num the maximum number of params to retrieve
	 * \param filter a param filter or NULL
	 *
	 * This requires X permissions on the port.
	 */
	int (*enum_params) (void *object, int seq,
			uint32_t id, uint32_t start, uint32_t num,
			const struct spa_pod *filter);
};

#define pw_port_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_port_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_port_add_listener(c,...)	pw_port_method(c,add_listener,0,__VA_ARGS__)
#define pw_port_subscribe_params(c,...)	pw_port_method(c,subscribe_params,0,__VA_ARGS__)
#define pw_port_enum_params(c,...)	pw_port_method(c,enum_params,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_PORT_H */
