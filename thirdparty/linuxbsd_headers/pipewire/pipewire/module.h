/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_MODULE_H
#define PIPEWIRE_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>

#include <pipewire/proxy.h>

/** \defgroup pw_module Module
 * Module interface
 */

/**
 * \addtogroup pw_module
 * \{
 */
#define PW_TYPE_INTERFACE_Module	PW_TYPE_INFO_INTERFACE_BASE "Module"

#define PW_MODULE_PERM_MASK		PW_PERM_R|PW_PERM_M

#define PW_VERSION_MODULE		3
struct pw_module;

/** The module information. Extra information can be added in later versions */
struct pw_module_info {
	uint32_t id;		/**< id of the global */
	const char *name;	/**< name of the module */
	const char *filename;	/**< filename of the module */
	const char *args;	/**< arguments passed to the module */
#define PW_MODULE_CHANGE_MASK_PROPS	(1 << 0)
#define PW_MODULE_CHANGE_MASK_ALL	((1 << 1)-1)
	uint64_t change_mask;	/**< bitfield of changed fields since last call */
	struct spa_dict *props;	/**< extra properties */
};

/** Update and existing \ref pw_module_info with \a update with reset */
struct pw_module_info *
pw_module_info_update(struct pw_module_info *info,
		const struct pw_module_info *update);
/** Merge and existing \ref pw_module_info with \a update */
struct pw_module_info *
pw_module_info_merge(struct pw_module_info *info,
		const struct pw_module_info *update, bool reset);
/** Free a \ref pw_module_info */
void pw_module_info_free(struct pw_module_info *info);

#define PW_MODULE_EVENT_INFO		0
#define PW_MODULE_EVENT_NUM		1

/** Module events */
struct pw_module_events {
#define PW_VERSION_MODULE_EVENTS	0
	uint32_t version;
	/**
	 * Notify module info
	 *
	 * \param info info about the module
	 */
	void (*info) (void *data, const struct pw_module_info *info);
};

#define PW_MODULE_METHOD_ADD_LISTENER	0
#define PW_MODULE_METHOD_NUM		1

/** Module methods */
struct pw_module_methods {
#define PW_VERSION_MODULE_METHODS	0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_module_events *events,
			void *data);
};

#define pw_module_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_module_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_module_add_listener(c,...)	pw_module_method(c,add_listener,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_MODULE_H */
