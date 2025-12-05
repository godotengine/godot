/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_CLIENT_H
#define PIPEWIRE_CLIENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/param/param.h>

#include <pipewire/proxy.h>
#include <pipewire/permission.h>

/** \defgroup pw_client Client
 * Client interface
 */

/**
 * \addtogroup pw_client
 * \{
 */
#define PW_TYPE_INTERFACE_Client	PW_TYPE_INFO_INTERFACE_BASE "Client"

#define PW_CLIENT_PERM_MASK		PW_PERM_RWXM

#define PW_VERSION_CLIENT		3
struct pw_client;

/* default ID of the current client after connect */
#define PW_ID_CLIENT			1

/** The client information. Extra information can be added in later versions */
struct pw_client_info {
	uint32_t id;			/**< id of the global */
#define PW_CLIENT_CHANGE_MASK_PROPS	(1 << 0)
#define PW_CLIENT_CHANGE_MASK_ALL	((1 << 1)-1)
	uint64_t change_mask;		/**< bitfield of changed fields since last call */
	struct spa_dict *props;		/**< extra properties */
};

/** Update an existing \ref pw_client_info with \a update with reset */
struct pw_client_info *
pw_client_info_update(struct pw_client_info *info,
		const struct pw_client_info *update);
/** Merge an existing \ref pw_client_info with \a update */
struct pw_client_info *
pw_client_info_merge(struct pw_client_info *info,
		const struct pw_client_info *update, bool reset);
/** Free a \ref pw_client_info */
void pw_client_info_free(struct pw_client_info *info);


#define PW_CLIENT_EVENT_INFO		0
#define PW_CLIENT_EVENT_PERMISSIONS	1
#define PW_CLIENT_EVENT_NUM		2

/** Client events */
struct pw_client_events {
#define PW_VERSION_CLIENT_EVENTS	0
	uint32_t version;
	/**
	 * Notify client info
	 *
	 * \param info info about the client
	 */
	void (*info) (void *data, const struct pw_client_info *info);
	/**
	 * Notify a client permission
	 *
	 * Event emitted as a result of the get_permissions method.
	 *
	 * \param default_permissions the default permissions
	 * \param index the index of the first permission entry
	 * \param n_permissions the number of permissions
	 * \param permissions the permissions
	 */
	void (*permissions) (void *data,
			     uint32_t index,
			     uint32_t n_permissions,
			     const struct pw_permission *permissions);
};


#define PW_CLIENT_METHOD_ADD_LISTENER		0
#define PW_CLIENT_METHOD_ERROR			1
#define PW_CLIENT_METHOD_UPDATE_PROPERTIES	2
#define PW_CLIENT_METHOD_GET_PERMISSIONS	3
#define PW_CLIENT_METHOD_UPDATE_PERMISSIONS	4
#define PW_CLIENT_METHOD_NUM			5

/** Client methods */
struct pw_client_methods {
#define PW_VERSION_CLIENT_METHODS	0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_client_events *events,
			void *data);
	/**
	 * Send an error to a client
	 *
	 * \param id the global id to report the error on
	 * \param res an errno style error code
	 * \param message an error string
	 *
	 * This requires W and X permissions on the client.
	 */
	int (*error) (void *object, uint32_t id, int res, const char *message);
	/**
	 * Update client properties
	 *
	 * \param props new properties
	 *
	 * This requires W and X permissions on the client.
	 */
	int (*update_properties) (void *object, const struct spa_dict *props);

	/**
	 * Get client permissions
	 *
	 * A permissions event will be emitted with the permissions.
	 *
	 * \param index the first index to query, 0 for first
	 * \param num the maximum number of items to get
	 *
	 * This requires W and X permissions on the client.
	 */
	int (*get_permissions) (void *object, uint32_t index, uint32_t num);
	/**
	 * Manage the permissions of the global objects for this
	 * client
	 *
	 * Update the permissions of the global objects using the
	 * provided array with permissions
	 *
	 * Globals can use the default permissions or can have specific
	 * permissions assigned to them.
	 *
	 * \param n_permissions number of permissions
	 * \param permissions array of permissions
	 *
	 * This requires W and X permissions on the client.
	 */
	int (*update_permissions) (void *object, uint32_t n_permissions,
			const struct pw_permission *permissions);
};

#define pw_client_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_client_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_client_add_listener(c,...)		pw_client_method(c,add_listener,0,__VA_ARGS__)
#define pw_client_error(c,...)			pw_client_method(c,error,0,__VA_ARGS__)
#define pw_client_update_properties(c,...)	pw_client_method(c,update_properties,0,__VA_ARGS__)
#define pw_client_get_permissions(c,...)	pw_client_method(c,get_permissions,0,__VA_ARGS__)
#define pw_client_update_permissions(c,...)	pw_client_method(c,update_permissions,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_CLIENT_H */
