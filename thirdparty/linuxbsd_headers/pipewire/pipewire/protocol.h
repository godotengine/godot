/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_PROTOCOL_H
#define PIPEWIRE_PROTOCOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/list.h>

/** \defgroup pw_protocol Protocol
 *
 * \brief Manages protocols and their implementation
 */

/**
 * \addtogroup pw_protocol
 * \{
 */

struct pw_protocol;

#include <pipewire/context.h>
#include <pipewire/properties.h>
#include <pipewire/utils.h>

#define PW_TYPE_INFO_Protocol		"PipeWire:Protocol"
#define PW_TYPE_INFO_PROTOCOL_BASE	PW_TYPE_INFO_Protocol ":"

struct pw_protocol_client {
	struct spa_list link;		/**< link in protocol client_list */
	struct pw_protocol *protocol;	/**< the owner protocol */

	struct pw_core *core;

	int (*connect) (struct pw_protocol_client *client,
			const struct spa_dict *props,
			void (*done_callback) (void *data, int result),
			void *data);
	int (*connect_fd) (struct pw_protocol_client *client, int fd, bool close);
	int (*steal_fd) (struct pw_protocol_client *client);
	void (*disconnect) (struct pw_protocol_client *client);
	void (*destroy) (struct pw_protocol_client *client);
	int (*set_paused) (struct pw_protocol_client *client, bool paused);
};

#define pw_protocol_client_connect(c,p,cb,d)	((c)->connect(c,p,cb,d))
#define pw_protocol_client_connect_fd(c,fd,cl)	((c)->connect_fd(c,fd,cl))
#define pw_protocol_client_steal_fd(c)		((c)->steal_fd(c))
#define pw_protocol_client_disconnect(c)	((c)->disconnect(c))
#define pw_protocol_client_destroy(c)		((c)->destroy(c))
#define pw_protocol_client_set_paused(c,p)	((c)->set_paused(c,p))

struct pw_protocol_server {
	struct spa_list link;		/**< link in protocol server_list */
	struct pw_protocol *protocol;	/**< the owner protocol */

	struct pw_impl_core *core;

	struct spa_list client_list;	/**< list of clients of this protocol */

	void (*destroy) (struct pw_protocol_server *listen);
};

#define pw_protocol_server_destroy(l)	((l)->destroy(l))

struct pw_protocol_marshal {
	const char *type;		/**< interface type */
	uint32_t version;		/**< version */
#define PW_PROTOCOL_MARSHAL_FLAG_IMPL	(1 << 0)	/**< marshal for implementations */
	uint32_t flags;			/**< version */
	uint32_t n_client_methods;	/**< number of client methods */
	uint32_t n_server_methods;	/**< number of server methods */
	const void *client_marshal;
	const void *server_demarshal;
	const void *server_marshal;
	const void *client_demarshal;
};

struct pw_protocol_implementation {
#define PW_VERSION_PROTOCOL_IMPLEMENTATION	0
	uint32_t version;

	struct pw_protocol_client * (*new_client) (struct pw_protocol *protocol,
						   struct pw_core *core,
						   const struct spa_dict *props);
	struct pw_protocol_server * (*add_server) (struct pw_protocol *protocol,
						   struct pw_impl_core *core,
						   const struct spa_dict *props);
};

struct pw_protocol_events {
#define PW_VERSION_PROTOCOL_EVENTS		0
	uint32_t version;

	void (*destroy) (void *data);
};

#define pw_protocol_new_client(p,...)	(pw_protocol_get_implementation(p)->new_client(p,__VA_ARGS__))
#define pw_protocol_add_server(p,...)	(pw_protocol_get_implementation(p)->add_server(p,__VA_ARGS__))
#define pw_protocol_ext(p,type,method,...)	(((type*)pw_protocol_get_extension(p))->method( __VA_ARGS__))

struct pw_protocol *pw_protocol_new(struct pw_context *context, const char *name, size_t user_data_size);

void pw_protocol_destroy(struct pw_protocol *protocol);

struct pw_context *pw_protocol_get_context(struct pw_protocol *protocol);

void *pw_protocol_get_user_data(struct pw_protocol *protocol);

const struct pw_protocol_implementation *
pw_protocol_get_implementation(struct pw_protocol *protocol);

const void *
pw_protocol_get_extension(struct pw_protocol *protocol);


void pw_protocol_add_listener(struct pw_protocol *protocol,
                              struct spa_hook *listener,
                              const struct pw_protocol_events *events,
                              void *data);

int pw_protocol_add_marshal(struct pw_protocol *protocol,
			    const struct pw_protocol_marshal *marshal);

const struct pw_protocol_marshal *
pw_protocol_get_marshal(struct pw_protocol *protocol, const char *type, uint32_t version, uint32_t flags);

struct pw_protocol * pw_context_find_protocol(struct pw_context *context, const char *name);

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_PROTOCOL_H */
