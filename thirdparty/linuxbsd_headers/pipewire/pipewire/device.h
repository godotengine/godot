/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_DEVICE_H
#define PIPEWIRE_DEVICE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>

#include <pipewire/proxy.h>

/** \defgroup pw_device Device
 * Device interface
 */

/**
 * \addtogroup pw_device
 * \{
 */

#define PW_TYPE_INTERFACE_Device	PW_TYPE_INFO_INTERFACE_BASE "Device"

#define PW_DEVICE_PERM_MASK		PW_PERM_RWXM

#define PW_VERSION_DEVICE		3
struct pw_device;

/** The device information. Extra information can be added in later versions */
struct pw_device_info {
	uint32_t id;			/**< id of the global */
#define PW_DEVICE_CHANGE_MASK_PROPS	(1 << 0)
#define PW_DEVICE_CHANGE_MASK_PARAMS	(1 << 1)
#define PW_DEVICE_CHANGE_MASK_ALL	((1 << 2)-1)
	uint64_t change_mask;		/**< bitfield of changed fields since last call */
	struct spa_dict *props;		/**< extra properties */
	struct spa_param_info *params;	/**< parameters */
	uint32_t n_params;		/**< number of items in \a params */
};

/** Update and existing \ref pw_device_info with \a update and reset */
struct pw_device_info *
pw_device_info_update(struct pw_device_info *info,
		const struct pw_device_info *update);
/** Merge and existing \ref pw_device_info with \a update */
struct pw_device_info *
pw_device_info_merge(struct pw_device_info *info,
		const struct pw_device_info *update, bool reset);
/** Free a \ref pw_device_info */
void pw_device_info_free(struct pw_device_info *info);

#define PW_DEVICE_EVENT_INFO	0
#define PW_DEVICE_EVENT_PARAM	1
#define PW_DEVICE_EVENT_NUM	2

/** Device events */
struct pw_device_events {
#define PW_VERSION_DEVICE_EVENTS	0
	uint32_t version;
	/**
	 * Notify device info
	 *
	 * \param info info about the device
	 */
	void (*info) (void *data, const struct pw_device_info *info);
	/**
	 * Notify a device param
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


#define PW_DEVICE_METHOD_ADD_LISTENER		0
#define PW_DEVICE_METHOD_SUBSCRIBE_PARAMS	1
#define PW_DEVICE_METHOD_ENUM_PARAMS		2
#define PW_DEVICE_METHOD_SET_PARAM		3
#define PW_DEVICE_METHOD_NUM			4

/** Device methods */
struct pw_device_methods {
#define PW_VERSION_DEVICE_METHODS	0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_device_events *events,
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
	 * This requires X permissions on the device.
	 */
	int (*subscribe_params) (void *object, uint32_t *ids, uint32_t n_ids);

	/**
	 * Enumerate device parameters
	 *
	 * Start enumeration of device parameters. For each param, a
	 * param event will be emitted.
	 *
	 * \param seq a sequence number to place in the reply
	 * \param id the parameter id to enum or PW_ID_ANY for all
	 * \param start the start index or 0 for the first param
	 * \param num the maximum number of params to retrieve
	 * \param filter a param filter or NULL
	 *
	 * This requires X permissions on the device.
	 */
	int (*enum_params) (void *object, int seq, uint32_t id, uint32_t start, uint32_t num,
			    const struct spa_pod *filter);
	/**
	 * Set a parameter on the device
	 *
	 * \param id the parameter id to set
	 * \param flags extra parameter flags
	 * \param param the parameter to set
	 *
	 * This requires W and X permissions on the device.
	 */
	int (*set_param) (void *object, uint32_t id, uint32_t flags,
			  const struct spa_pod *param);
};

#define pw_device_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_device_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_device_add_listener(c,...)		pw_device_method(c,add_listener,0,__VA_ARGS__)
#define pw_device_subscribe_params(c,...)	pw_device_method(c,subscribe_params,0,__VA_ARGS__)
#define pw_device_enum_params(c,...)		pw_device_method(c,enum_params,0,__VA_ARGS__)
#define pw_device_set_param(c,...)		pw_device_method(c,set_param,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_DEVICE_H */
