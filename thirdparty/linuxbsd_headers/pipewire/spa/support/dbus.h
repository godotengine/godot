/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DBUS_H
#define SPA_DBUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/support/loop.h>

/** \defgroup spa_dbus DBus
 * DBus communication
 */

/**
 * \addtogroup spa_dbus
 * \{
 */

#define SPA_TYPE_INTERFACE_DBus		SPA_TYPE_INFO_INTERFACE_BASE "DBus"

#define SPA_VERSION_DBUS		0
struct spa_dbus { struct spa_interface iface; };

enum spa_dbus_type {
	SPA_DBUS_TYPE_SESSION,	/**< The login session bus */
	SPA_DBUS_TYPE_SYSTEM,	/**< The systemwide bus */
	SPA_DBUS_TYPE_STARTER	/**< The bus that started us, if any */
};

#define SPA_DBUS_CONNECTION_EVENT_DESTROY	0
#define SPA_DBUS_CONNECTION_EVENT_DISCONNECTED	1
#define SPA_DBUS_CONNECTION_EVENT_NUM		2

struct spa_dbus_connection_events {
#define SPA_VERSION_DBUS_CONNECTION_EVENTS	0
        uint32_t version;

	/* a connection is destroyed */
	void (*destroy) (void *data);

	/* a connection disconnected */
	void (*disconnected) (void *data);
};

struct spa_dbus_connection {
#define SPA_VERSION_DBUS_CONNECTION	1
        uint32_t version;
	/**
	 * Get the DBusConnection from a wrapper
	 *
	 * Note that the returned handle is closed and unref'd by spa_dbus
	 * immediately before emitting the asynchronous "disconnected" event.
	 * The caller must either deal with the invalidation, or keep an extra
	 * ref on the handle returned.
	 *
	 * \param conn the spa_dbus_connection wrapper
	 * \return a pointer of type DBusConnection
	 */
	void *(*get) (struct spa_dbus_connection *conn);
	/**
	 * Destroy a dbus connection wrapper
	 *
	 * \param conn the wrapper to destroy
	 */
	void (*destroy) (struct spa_dbus_connection *conn);

	/**
	 * Add a listener for events
	 *
	 * Since version 1
	 */
	void (*add_listener) (struct spa_dbus_connection *conn,
			struct spa_hook *listener,
			const struct spa_dbus_connection_events *events,
			void *data);
};

#define spa_dbus_connection_call(c,method,vers,...)			\
({									\
	if (SPA_LIKELY(SPA_CALLBACK_CHECK(c,method,vers)))		\
		c->method((c), ## __VA_ARGS__);				\
})

#define spa_dbus_connection_call_vp(c,method,vers,...)			\
({									\
	void *_res = NULL;						\
	if (SPA_LIKELY(SPA_CALLBACK_CHECK(c,method,vers)))		\
		_res = c->method((c), ## __VA_ARGS__);			\
	_res;								\
})

/** \copydoc spa_dbus_connection.get
 * \sa spa_dbus_connection.get */
#define spa_dbus_connection_get(c)		spa_dbus_connection_call_vp(c,get,0)
/** \copydoc spa_dbus_connection.destroy
 * \sa spa_dbus_connection.destroy */
#define spa_dbus_connection_destroy(c)		spa_dbus_connection_call(c,destroy,0)
/** \copydoc spa_dbus_connection.add_listener
 * \sa spa_dbus_connection.add_listener */
#define spa_dbus_connection_add_listener(c,...)	spa_dbus_connection_call(c,add_listener,1,__VA_ARGS__)

struct spa_dbus_methods {
#define SPA_VERSION_DBUS_METHODS	0
        uint32_t version;

	/**
	 * Get a new connection wrapper for the given bus type.
	 *
	 * The connection wrapper is completely configured to operate
	 * in the main context of the handle that manages the spa_dbus
	 * interface.
	 *
	 * \param dbus the dbus manager
	 * \param type the bus type to wrap
	 * \param error location for the DBusError
	 * \return a new dbus connection wrapper or NULL on error
	 */
	struct spa_dbus_connection * (*get_connection) (void *object,
							enum spa_dbus_type type);
};

/** \copydoc spa_dbus_methods.get_connection
 * \sa spa_dbus_methods.get_connection
 */
static inline struct spa_dbus_connection *
spa_dbus_get_connection(struct spa_dbus *dbus, enum spa_dbus_type type)
{
	struct spa_dbus_connection *res = NULL;
	spa_interface_call_res(&dbus->iface,
                        struct spa_dbus_methods, res,
			get_connection, 0, type);
	return res;
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DBUS_H */
