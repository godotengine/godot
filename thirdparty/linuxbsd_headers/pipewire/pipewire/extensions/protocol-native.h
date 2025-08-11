/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_EXT_PROTOCOL_NATIVE_H
#define PIPEWIRE_EXT_PROTOCOL_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>

#include <pipewire/proxy.h>
#include <pipewire/resource.h>

/** \defgroup pw_protocol_native Native Protocol
 * PipeWire native protocol interface
 */

/**
 * \addtogroup pw_protocol_native
 * \{
 */
#define PW_TYPE_INFO_PROTOCOL_Native		PW_TYPE_INFO_PROTOCOL_BASE "Native"

struct pw_protocol_native_message {
	uint32_t id;
	uint32_t opcode;
	void *data;
	uint32_t size;
	uint32_t n_fds;
	int *fds;
	int seq;
};

struct pw_protocol_native_demarshal {
	int (*func) (void *object, const struct pw_protocol_native_message *msg);
	uint32_t permissions;
	uint32_t flags;
};

/** \ref pw_protocol_native_ext methods */
struct pw_protocol_native_ext {
#define PW_VERSION_PROTOCOL_NATIVE_EXT	0
	uint32_t version;

	struct spa_pod_builder * (*begin_proxy) (struct pw_proxy *proxy,
			uint8_t opcode, struct pw_protocol_native_message **msg);

	uint32_t (*add_proxy_fd) (struct pw_proxy *proxy, int fd);
	int (*get_proxy_fd) (struct pw_proxy *proxy, uint32_t index);

	int (*end_proxy) (struct pw_proxy *proxy,
			  struct spa_pod_builder *builder);

	struct spa_pod_builder * (*begin_resource) (struct pw_resource *resource,
			uint8_t opcode, struct pw_protocol_native_message **msg);

	uint32_t (*add_resource_fd) (struct pw_resource *resource, int fd);
	int (*get_resource_fd) (struct pw_resource *resource, uint32_t index);

	int (*end_resource) (struct pw_resource *resource,
			     struct spa_pod_builder *builder);
};

#define pw_protocol_native_begin_proxy(p,...)		pw_protocol_ext(pw_proxy_get_protocol(p),struct pw_protocol_native_ext,begin_proxy,p,__VA_ARGS__)
#define pw_protocol_native_add_proxy_fd(p,...)		pw_protocol_ext(pw_proxy_get_protocol(p),struct pw_protocol_native_ext,add_proxy_fd,p,__VA_ARGS__)
#define pw_protocol_native_get_proxy_fd(p,...)		pw_protocol_ext(pw_proxy_get_protocol(p),struct pw_protocol_native_ext,get_proxy_fd,p,__VA_ARGS__)
#define pw_protocol_native_end_proxy(p,...)		pw_protocol_ext(pw_proxy_get_protocol(p),struct pw_protocol_native_ext,end_proxy,p,__VA_ARGS__)

#define pw_protocol_native_begin_resource(r,...)	pw_protocol_ext(pw_resource_get_protocol(r),struct pw_protocol_native_ext,begin_resource,r,__VA_ARGS__)
#define pw_protocol_native_add_resource_fd(r,...)	pw_protocol_ext(pw_resource_get_protocol(r),struct pw_protocol_native_ext,add_resource_fd,r,__VA_ARGS__)
#define pw_protocol_native_get_resource_fd(r,...)	pw_protocol_ext(pw_resource_get_protocol(r),struct pw_protocol_native_ext,get_resource_fd,r,__VA_ARGS__)
#define pw_protocol_native_end_resource(r,...)		pw_protocol_ext(pw_resource_get_protocol(r),struct pw_protocol_native_ext,end_resource,r,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_EXT_PROTOCOL_NATIVE_H */
