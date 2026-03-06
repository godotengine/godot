/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_RESOURCE_H
#define PIPEWIRE_RESOURCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/hook.h>

/** \defgroup pw_resource Resource
 *
 * \brief Client owned objects
 *
 * Resources represent objects owned by a \ref pw_impl_client. They are
 * the result of binding to a global resource or by calling API that
 * creates client owned objects.
 *
 * The client usually has a proxy object associated with the resource
 * that it can use to communicate with the resource. See \ref page_proxy.
 *
 * Resources are destroyed when the client or the bound object is
 * destroyed.
 *
 */


/**
 * \addtogroup pw_resource
 * \{
 */
struct pw_resource;

#include <pipewire/impl-client.h>

/** Resource events */
struct pw_resource_events {
#define PW_VERSION_RESOURCE_EVENTS	0
	uint32_t version;

	/** The resource is destroyed */
	void (*destroy) (void *data);

	/** a reply to a ping event completed */
        void (*pong) (void *data, int seq);

	/** an error occurred on the resource */
        void (*error) (void *data, int seq, int res, const char *message);
};

/** Make a new resource for client */
struct pw_resource *
pw_resource_new(struct pw_impl_client *client,	/**< the client owning the resource */
		uint32_t id,			/**< the remote per client id */
		uint32_t permissions,		/**< permissions on this resource */
		const char *type,		/**< interface of the resource */
		uint32_t version,		/**< requested interface version */
		size_t user_data_size		/**< extra user data size */);

/** Destroy a resource */
void pw_resource_destroy(struct pw_resource *resource);

/** Remove a resource, like pw_resource_destroy but without sending a
 * remove_id message to the client */
void pw_resource_remove(struct pw_resource *resource);

/** Get the client owning this resource */
struct pw_impl_client *pw_resource_get_client(struct pw_resource *resource);

/** Get the unique id of this resource */
uint32_t pw_resource_get_id(struct pw_resource *resource);

/** Get the permissions of this resource */
uint32_t pw_resource_get_permissions(struct pw_resource *resource);

/** Get the type and optionally the version of this resource */
const char *pw_resource_get_type(struct pw_resource *resource, uint32_t *version);

/** Get the protocol used for this resource */
struct pw_protocol *pw_resource_get_protocol(struct pw_resource *resource);

/** Get the user data for the resource, the size was given in \ref pw_resource_new */
void *pw_resource_get_user_data(struct pw_resource *resource);

/** Add an event listener */
void pw_resource_add_listener(struct pw_resource *resource,
			      struct spa_hook *listener,
			      const struct pw_resource_events *events,
			      void *data);

/** Set the resource implementation. */
void pw_resource_add_object_listener(struct pw_resource *resource,
				struct spa_hook *listener,
				const void *funcs,
				void *data);

/** Generate an ping event for a resource. This will generate a pong event
 * with the same \a sequence number in the return value. */
int pw_resource_ping(struct pw_resource *resource, int seq);

/** ref/unref a resource, Since 0.3.52 */
void pw_resource_ref(struct pw_resource *resource);
void pw_resource_unref(struct pw_resource *resource);

/** Notify global id this resource is bound to */
int pw_resource_set_bound_id(struct pw_resource *resource, uint32_t global_id);

/** Get the global id this resource is bound to or SPA_ID_INVALID when not bound */
uint32_t pw_resource_get_bound_id(struct pw_resource *resource);

/** Generate an error for a resource */
void pw_resource_error(struct pw_resource *resource, int res, const char *error);
void pw_resource_errorf(struct pw_resource *resource, int res, const char *error, ...) SPA_PRINTF_FUNC(3, 4);
void pw_resource_errorf_id(struct pw_resource *resource, uint32_t id, int res, const char *error, ...) SPA_PRINTF_FUNC(4, 5);

/** Get the list of object listeners from a resource */
struct spa_hook_list *pw_resource_get_object_listeners(struct pw_resource *resource);

/** Get the marshal functions for the resource */
const struct pw_protocol_marshal *pw_resource_get_marshal(struct pw_resource *resource);

/** install a marshal function on a resource */
int pw_resource_install_marshal(struct pw_resource *resource, bool implementor);

#define pw_resource_notify(r,type,event,version,...)			\
	spa_hook_list_call(pw_resource_get_object_listeners(r),		\
			type, event, version, ## __VA_ARGS__)

#define pw_resource_call(r,type,method,version,...)			\
	spa_interface_call((struct spa_interface*)r,			\
			type, method, version, ##__VA_ARGS__)

#define pw_resource_call_res(r,type,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)r,		\
			type, _res, method, version, ##__VA_ARGS__);	\
	_res;								\
})


/**
 * \}
 */


#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_RESOURCE_H */
