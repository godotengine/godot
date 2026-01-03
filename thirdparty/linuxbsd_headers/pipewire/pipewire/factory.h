/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_FACTORY_H
#define PIPEWIRE_FACTORY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <errno.h>

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>

#include <pipewire/proxy.h>

/** \defgroup pw_factory Factory
 * Factory interface
 */

/**
 * \addtogroup pw_factory
 * \{
 */
#define PW_TYPE_INTERFACE_Factory	PW_TYPE_INFO_INTERFACE_BASE "Factory"

#define PW_FACTORY_PERM_MASK		PW_PERM_R|PW_PERM_M

#define PW_VERSION_FACTORY		3
struct pw_factory;

/** The factory information. Extra information can be added in later versions */
struct pw_factory_info {
	uint32_t id;			/**< id of the global */
	const char *name;		/**< name the factory */
	const char *type;		/**< type of the objects created by this factory */
	uint32_t version;		/**< version of the objects */
#define PW_FACTORY_CHANGE_MASK_PROPS	(1 << 0)
#define PW_FACTORY_CHANGE_MASK_ALL	((1 << 1)-1)
	uint64_t change_mask;		/**< bitfield of changed fields since last call */
	struct spa_dict *props;		/**< the properties of the factory */
};

struct pw_factory_info *
pw_factory_info_update(struct pw_factory_info *info,
		const struct pw_factory_info *update);
struct pw_factory_info *
pw_factory_info_merge(struct pw_factory_info *info,
		const struct pw_factory_info *update, bool reset);
void
pw_factory_info_free(struct pw_factory_info *info);


#define PW_FACTORY_EVENT_INFO		0
#define PW_FACTORY_EVENT_NUM		1

/** Factory events */
struct pw_factory_events {
#define PW_VERSION_FACTORY_EVENTS	0
	uint32_t version;
	/**
	 * Notify factory info
	 *
	 * \param info info about the factory
	 */
	void (*info) (void *data, const struct pw_factory_info *info);
};

#define PW_FACTORY_METHOD_ADD_LISTENER	0
#define PW_FACTORY_METHOD_NUM		1

/** Factory methods */
struct pw_factory_methods {
#define PW_VERSION_FACTORY_METHODS	0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_factory_events *events,
			void *data);
};

#define pw_factory_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_factory_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_factory_add_listener(c,...)	pw_factory_method(c,add_listener,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_FACTORY_H */
