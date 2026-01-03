/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_PRIVATE_H
#define PIPEWIRE_PRIVATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/socket.h>
#include <sys/types.h> /* for pthread_t */

#include "pipewire/impl.h"

#include <spa/support/plugin.h>
#include <spa/pod/builder.h>
#include <spa/param/latency-utils.h>
#include <spa/utils/atomic.h>
#include <spa/utils/ratelimit.h>
#include <spa/utils/result.h>
#include <spa/utils/type-info.h>

#if defined(__FreeBSD__) || defined(__MidnightBSD__) || defined(__GNU__)
struct ucred {
};
#endif

#define MAX_RATES				32u
#define CLOCK_MIN_QUANTUM			4u
#define CLOCK_MAX_QUANTUM			65536u

struct settings {
	uint32_t log_level;
	uint32_t clock_rate;			/* default clock rate */
	uint32_t clock_rates[MAX_RATES];	/* allowed clock rates */
	uint32_t n_clock_rates;			/* number of alternative clock rates */
	uint32_t clock_quantum;			/* default quantum */
	uint32_t clock_min_quantum;		/* min quantum */
	uint32_t clock_max_quantum;		/* max quantum */
	uint32_t clock_quantum_limit;		/* quantum limit */
	struct spa_rectangle video_size;
	struct spa_fraction video_rate;
	uint32_t link_max_buffers;
	unsigned int mem_warn_mlock:1;
	unsigned int mem_allow_mlock:1;
	unsigned int clock_power_of_two_quantum:1;
	unsigned int check_quantum:1;
	unsigned int check_rate:1;
#define CLOCK_RATE_UPDATE_MODE_HARD 0
#define CLOCK_RATE_UPDATE_MODE_SOFT 1
	int clock_rate_update_mode;
	uint32_t clock_force_rate;		/* force a clock rate */
	uint32_t clock_force_quantum;		/* force a quantum */
};

#define MAX_PARAMS	32

struct pw_param {
	uint32_t id;
	int32_t seq;
	struct spa_list link;
	struct spa_pod *param;
};

static inline uint32_t pw_param_clear(struct spa_list *param_list, uint32_t id)
{
	struct pw_param *p, *t;
	uint32_t count = 0;

	spa_list_for_each_safe(p, t, param_list, link) {
		if (id == SPA_ID_INVALID || p->id == id) {
			spa_list_remove(&p->link);
			free(p);
			count++;
		}
	}
	return count;
}

static inline struct pw_param *pw_param_add(struct spa_list *params, int32_t seq,
		uint32_t id, const struct spa_pod *param)
{
	struct pw_param *p;

	if (id == SPA_ID_INVALID) {
		if (param == NULL || !spa_pod_is_object(param)) {
			errno = EINVAL;
			return NULL;
		}
		id = SPA_POD_OBJECT_ID(param);
	}

	if ((p = malloc(sizeof(*p) + (param != NULL ? SPA_POD_SIZE(param) : 0))) == NULL)
		return NULL;

	p->id = id;
	p->seq = seq;
	if (param != NULL) {
		p->param = SPA_PTROFF(p, sizeof(*p), struct spa_pod);
		memcpy(p->param, param, SPA_POD_SIZE(param));
	} else {
		pw_param_clear(params, id);
		p->param = NULL;
	}
	spa_list_append(params, &p->link);
	return p;
}

static inline void pw_param_update(struct spa_list *param_list, struct spa_list *pending_list,
			uint32_t n_params, struct spa_param_info *params)
{
	struct pw_param *p, *t;
	uint32_t i;

	for (i = 0; i < n_params; i++) {
		spa_list_for_each_safe(p, t, pending_list, link) {
			if (p->id == params[i].id &&
			    p->seq != params[i].seq &&
			    p->param != NULL) {
				spa_list_remove(&p->link);
				free(p);
			}
		}
	}
	spa_list_consume(p, pending_list, link) {
		spa_list_remove(&p->link);
		if (p->param == NULL) {
			pw_param_clear(param_list, p->id);
			free(p);
		} else {
			spa_list_append(param_list, &p->link);
		}
	}
}

static inline struct spa_param_info *pw_param_info_find(struct spa_param_info info[],
		uint32_t n_info, uint32_t id)
{
	uint32_t i;
	for (i = 0; i < n_info; i++) {
		if (info[i].id == id)
			return &info[i];
	}
	return NULL;
}

#define pw_protocol_emit_destroy(p) spa_hook_list_call(&(p)->listener_list, struct pw_protocol_events, destroy, 0)

struct pw_protocol {
	struct spa_list link;                   /**< link in context protocol_list */
	struct pw_context *context;                   /**< context for this protocol */

	char *name;                             /**< type name of the protocol */

	struct spa_list marshal_list;           /**< list of marshallers for supported interfaces */
	struct spa_list client_list;            /**< list of current clients */
	struct spa_list server_list;            /**< list of current servers */
	struct spa_hook_list listener_list;	/**< event listeners */

	const struct pw_protocol_implementation *implementation; /**< implementation of the protocol */

	const void *extension;  /**< extension API */

	void *user_data;        /**< user data for the implementation */
};

/** the permission function. It returns the allowed access permissions for \a global
  * for \a client */
typedef uint32_t (*pw_permission_func_t) (struct pw_global *global,
					  struct pw_impl_client *client, void *data);

#define pw_impl_client_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_impl_client_events, m, v, ##__VA_ARGS__)

#define pw_impl_client_emit_destroy(o)			pw_impl_client_emit(o, destroy, 0)
#define pw_impl_client_emit_free(o)			pw_impl_client_emit(o, free, 0)
#define pw_impl_client_emit_initialized(o)		pw_impl_client_emit(o, initialized, 0)
#define pw_impl_client_emit_info_changed(o,i)		pw_impl_client_emit(o, info_changed, 0, i)
#define pw_impl_client_emit_resource_added(o,r)		pw_impl_client_emit(o, resource_added, 0, r)
#define pw_impl_client_emit_resource_impl(o,r)		pw_impl_client_emit(o, resource_impl, 0, r)
#define pw_impl_client_emit_resource_removed(o,r)	pw_impl_client_emit(o, resource_removed, 0, r)
#define pw_impl_client_emit_busy_changed(o,b)		pw_impl_client_emit(o, busy_changed, 0, b)

#define pw_impl_core_emit(s,m,v,...) spa_hook_list_call(&s->listener_list, struct pw_impl_core_events, m, v, ##__VA_ARGS__)

#define pw_impl_core_emit_destroy(s)		pw_impl_core_emit(s, destroy, 0)
#define pw_impl_core_emit_free(s)		pw_impl_core_emit(s, free, 0)
#define pw_impl_core_emit_initialized(s)	pw_impl_core_emit(s, initialized, 0)

struct pw_impl_core {
	struct pw_context *context;
	struct spa_list link;			/**< link in context object core_impl list */
	struct pw_global *global;		/**< global object created for this core */
	struct spa_hook global_listener;

	struct pw_properties *properties;	/**< core properties */
	struct pw_core_info info;		/**< core info */

	struct spa_hook_list listener_list;
	void *user_data;			/**< extra user data */

	unsigned int registered:1;
};

#define pw_impl_metadata_emit(s,m,v,...) spa_hook_list_call(&s->listener_list, struct pw_impl_metadata_events, m, v, ##__VA_ARGS__)

#define pw_impl_metadata_emit_destroy(s)	pw_impl_metadata_emit(s, destroy, 0)
#define pw_impl_metadata_emit_free(s)		pw_impl_metadata_emit(s, free, 0)
#define pw_impl_metadata_emit_property(s, ...)	pw_impl_metadata_emit(s, property, 0, __VA_ARGS__)

struct pw_impl_metadata {
	struct pw_context *context;		/**< the context */
	struct spa_list link;			/**< link in context metadata_list */
	struct pw_global *global;		/**< global for this metadata */
	struct spa_hook global_listener;

	struct pw_properties *properties;	/**< properties of the metadata */

	struct pw_metadata *metadata;
	struct spa_hook metadata_listener;

	struct spa_hook_list listener_list;	/**< event listeners */
	void *user_data;

	unsigned int registered:1;
};

struct pw_impl_client {
	struct pw_impl_core *core;		/**< core object */
	struct pw_context *context;		/**< context object */

	struct spa_list link;			/**< link in context object client list */
	struct pw_global *global;		/**< global object created for this client */
	struct spa_hook global_listener;

	pw_permission_func_t permission_func;	/**< get permissions of an object */
	void *permission_data;			/**< data passed to permission function */

	struct pw_properties *properties;	/**< Client properties */

	struct pw_client_info info;	/**< client info */

	struct pw_mempool *pool;		/**< client mempool */
	struct pw_resource *core_resource;	/**< core resource object */
	struct pw_resource *client_resource;	/**< client resource object */

	struct pw_map objects;		/**< list of resource objects */

	struct spa_hook_list listener_list;

	struct pw_protocol *protocol;	/**< protocol in use */
	int recv_seq;			/**< last received sequence number */
	int send_seq;			/**< last sender sequence number */
	uint64_t recv_generation;	/**< last received registry generation */
	uint64_t sent_generation;	/**< last sent registry generation */

	void *user_data;		/**< extra user data */

	struct ucred ucred;		/**< ucred information */
	unsigned int registered:1;
	unsigned int ucred_valid:1;	/**< if the ucred member is valid */
	unsigned int busy:1;
	unsigned int destroyed:1;

	int refcount;

	/* v2 compatibility data */
	void *compat_v2;
};

#define pw_global_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_global_events, m, v, ##__VA_ARGS__)

#define pw_global_emit_registering(g)	pw_global_emit(g, registering, 0)
#define pw_global_emit_destroy(g)	pw_global_emit(g, destroy, 0)
#define pw_global_emit_free(g)		pw_global_emit(g, free, 0)
#define pw_global_emit_permissions_changed(g,...)	pw_global_emit(g, permissions_changed, 0, __VA_ARGS__)

struct pw_global {
	struct pw_context *context;		/**< the context */

	struct spa_list link;		/**< link in context list of globals */
	uint32_t id;			/**< server id of the object */

	struct pw_properties *properties;	/**< properties of the global */

	struct spa_hook_list listener_list;

	const char *type;		/**< type of interface */
	uint32_t version;		/**< version of interface */
	uint32_t permission_mask;	/**< possible permissions */

	pw_global_bind_func_t func;	/**< bind function */
	void *object;			/**< object associated with the interface */
	uint64_t serial;		/**< increasing serial number */
	uint64_t generation;		/**< registry generation number */

	struct spa_list resource_list;	/**< The list of resources of this global */

	unsigned int registered:1;
	unsigned int destroyed:1;
};

#define pw_core_resource(r,m,v,...)	pw_resource_call(r, struct pw_core_events, m, v, ##__VA_ARGS__)
#define pw_core_resource_info(r,...)		pw_core_resource(r,info,0,__VA_ARGS__)
#define pw_core_resource_done(r,...)		pw_core_resource(r,done,0,__VA_ARGS__)
#define pw_core_resource_ping(r,...)		pw_core_resource(r,ping,0,__VA_ARGS__)
#define pw_core_resource_error(r,...)		pw_core_resource(r,error,0,__VA_ARGS__)
#define pw_core_resource_remove_id(r,...)	pw_core_resource(r,remove_id,0,__VA_ARGS__)
#define pw_core_resource_bound_id(r,...)	pw_core_resource(r,bound_id,0,__VA_ARGS__)
#define pw_core_resource_add_mem(r,...)		pw_core_resource(r,add_mem,0,__VA_ARGS__)
#define pw_core_resource_remove_mem(r,...)	pw_core_resource(r,remove_mem,0,__VA_ARGS__)
#define pw_core_resource_bound_props(r,...)	pw_core_resource(r,bound_props,1,__VA_ARGS__)

static inline SPA_PRINTF_FUNC(5,0) void
pw_core_resource_errorv(struct pw_resource *resource, uint32_t id, int seq,
		int res, const char *message, va_list args)
{
	char buffer[1024];
	vsnprintf(buffer, sizeof(buffer), message, args);
	buffer[1023] = '\0';
	pw_log_debug("resource %p: id:%d seq:%d res:%d (%s) msg:\"%s\"",
			resource, id, seq, res, spa_strerror(res), buffer);
	if (resource)
		pw_core_resource_error(resource, id, seq, res, buffer);
	else
		pw_log_error("id:%d seq:%d res:%d (%s) msg:\"%s\"",
				id, seq, res, spa_strerror(res), buffer);
}

static inline SPA_PRINTF_FUNC(5,6) void
pw_core_resource_errorf(struct pw_resource *resource, uint32_t id, int seq,
		int res, const char *message, ...)
{
        va_list args;
	va_start(args, message);
	pw_core_resource_errorv(resource, id, seq, res, message, args);
	va_end(args);
}

struct pw_loop_callbacks {
#define PW_VERSION_LOOP_CALLBACKS	0
	uint32_t version;

	int (*check) (void *data, struct pw_loop *loop);
};

void
pw_loop_set_callbacks(struct pw_loop *loop, const struct pw_loop_callbacks *cb, void *data);

int pw_loop_check(struct pw_loop *loop);

#define ensure_loop(loop,...) ({							\
	int res = pw_loop_check(loop);							\
	if (res != 1) {									\
		pw_log_warn("%s called from wrong context, check thread and locking: %s",	\
				__func__, res < 0 ? spa_strerror(res) : "Not in loop");	\
		fprintf(stderr, "*** %s called from wrong context, check thread and locking: %s\n",\
				__func__, res < 0 ? spa_strerror(res) : "Not in loop");	\
		/* __VA_ARGS__ */							\
	}										\
})

#define pw_registry_resource(r,m,v,...) pw_resource_call(r, struct pw_registry_events,m,v,##__VA_ARGS__)
#define pw_registry_resource_global(r,...)        pw_registry_resource(r,global,0,__VA_ARGS__)
#define pw_registry_resource_global_remove(r,...) pw_registry_resource(r,global_remove,0,__VA_ARGS__)

#define pw_context_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_context_events, m, v, ##__VA_ARGS__)
#define pw_context_emit_destroy(c)		pw_context_emit(c, destroy, 0)
#define pw_context_emit_free(c)			pw_context_emit(c, free, 0)
#define pw_context_emit_info_changed(c,i)	pw_context_emit(c, info_changed, 0, i)
#define pw_context_emit_check_access(c,cl)	pw_context_emit(c, check_access, 0, cl)
#define pw_context_emit_global_added(c,g)	pw_context_emit(c, global_added, 0, g)
#define pw_context_emit_global_removed(c,g)	pw_context_emit(c, global_removed, 0, g)
#define pw_context_emit_driver_added(c,n)	pw_context_emit(c, driver_added, 1, n)
#define pw_context_emit_driver_removed(c,n)	pw_context_emit(c, driver_removed, 1, n)

struct pw_context {
	struct pw_impl_core *core;		/**< core object */

	struct pw_properties *conf;		/**< configuration of the context */
	struct pw_properties *properties;	/**< properties of the context */

	struct settings defaults;		/**< default parameters */
	struct settings settings;		/**< current parameters */

	void *settings_impl;		/**< settings metadata */

	struct pw_mempool *pool;		/**< global memory pool */

	uint64_t stamp;
	uint64_t serial;
	uint64_t generation;			/**< registry generation number */
	struct pw_map globals;			/**< map of globals */

	struct spa_list core_impl_list;		/**< list of core_imp */
	struct spa_list protocol_list;		/**< list of protocols */
	struct spa_list core_list;		/**< list of core connections */
	struct spa_list registry_resource_list;	/**< list of registry resources */
	struct spa_list module_list;		/**< list of modules */
	struct spa_list device_list;		/**< list of devices */
	struct spa_list global_list;		/**< list of globals */
	struct spa_list client_list;		/**< list of clients */
	struct spa_list node_list;		/**< list of nodes */
	struct spa_list factory_list;		/**< list of factories */
	struct spa_list metadata_list;		/**< list of metadata */
	struct spa_list link_list;		/**< list of links */
	struct spa_list control_list[2];	/**< list of controls, indexed by direction */
	struct spa_list export_list;		/**< list of export types */
	struct spa_list driver_list;		/**< list of driver nodes */

	struct spa_hook_list driver_listener_list;
	struct spa_hook_list listener_list;

	struct spa_thread_utils *thread_utils;
	struct pw_loop *main_loop;		/**< main loop for control */
	struct pw_loop *data_loop;		/**< data loop for data passing */
	struct spa_system *data_system;		/**< data system for data passing */
	struct pw_work_queue *work_queue;	/**< work queue */

	struct spa_support support[16];	/**< support for spa plugins */
	uint32_t n_support;		/**< number of support items */
	struct pw_array factory_lib;	/**< mapping of factory_name regexp to library */

	struct pw_array objects;	/**< objects */

	struct pw_impl_client *current_client;	/**< client currently executing code in mainloop */

	long sc_pagesize;
	unsigned int freewheeling:1;

	void *user_data;		/**< extra user data */
};

#define pw_data_loop_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_data_loop_events, m, v, ##__VA_ARGS__)
#define pw_data_loop_emit_destroy(o) pw_data_loop_emit(o, destroy, 0)

struct pw_data_loop {
	struct pw_loop *loop;

	struct spa_hook_list listener_list;

	struct spa_thread_utils *thread_utils;

	pthread_t thread;
	unsigned int cancel:1;
	unsigned int created:1;
	unsigned int running:1;
};

#define pw_main_loop_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_main_loop_events, m, v, ##__VA_ARGS__)
#define pw_main_loop_emit_destroy(o) pw_main_loop_emit(o, destroy, 0)

struct pw_main_loop {
        struct pw_loop *loop;

	struct spa_hook_list listener_list;

	unsigned int created:1;
	unsigned int running:1;
};

#define pw_impl_device_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_impl_device_events, m, v, ##__VA_ARGS__)
#define pw_impl_device_emit_destroy(m)		pw_impl_device_emit(m, destroy, 0)
#define pw_impl_device_emit_free(m)		pw_impl_device_emit(m, free, 0)
#define pw_impl_device_emit_initialized(m)	pw_impl_device_emit(m, initialized, 0)
#define pw_impl_device_emit_info_changed(n,i)	pw_impl_device_emit(n, info_changed, 0, i)

struct pw_impl_device {
	struct pw_context *context;           /**< the context object */
	struct spa_list link;           /**< link in the context device_list */
	struct pw_global *global;       /**< global object for this device */
	struct spa_hook global_listener;

	struct pw_properties *properties;	/**< properties of the device */
	struct pw_device_info info;		/**< introspectable device info */
	struct spa_param_info params[MAX_PARAMS];

	char *name;				/**< device name for debug */

	struct spa_device *device;		/**< device implementation */
	struct spa_hook listener;
	struct spa_hook_list listener_list;

	struct spa_list object_list;

	void *user_data;                /**< device user_data */

	unsigned int registered:1;
};

#define pw_impl_module_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_impl_module_events, m, v, ##__VA_ARGS__)
#define pw_impl_module_emit_destroy(m)		pw_impl_module_emit(m, destroy, 0)
#define pw_impl_module_emit_free(m)		pw_impl_module_emit(m, free, 0)
#define pw_impl_module_emit_initialized(m)	pw_impl_module_emit(m, initialized, 0)
#define pw_impl_module_emit_registered(m)	pw_impl_module_emit(m, registered, 0)

struct pw_impl_module {
	struct pw_context *context;	/**< the context object */
	struct spa_list link;		/**< link in the context module_list */
	struct pw_global *global;	/**< global object for this module */
	struct spa_hook global_listener;

	struct pw_properties *properties;	/**< properties of the module */
	struct pw_module_info info;	/**< introspectable module info */

	struct spa_hook_list listener_list;

	void *user_data;		/**< module user_data */
};

struct pw_node_activation_state {
	int status;                     /**< current status, the result of spa_node_process() */
	int32_t required;		/**< required number of signals */
	int32_t pending;		/**< number of pending signals */
};

static inline void pw_node_activation_state_reset(struct pw_node_activation_state *state)
{
        state->pending = state->required;
}

#define pw_node_activation_state_dec(s) (SPA_ATOMIC_DEC(s->pending) == 0)

struct pw_node_target {
	struct spa_list link;
#define PW_NODE_TARGET_NONE	0
#define PW_NODE_TARGET_PEER	1
	uint32_t flags;
	uint32_t id;
	char name[128];
	struct pw_impl_node *node;
	struct pw_node_activation *activation;
	struct spa_system *system;
	int fd;
	unsigned int active:1;
};

static inline void copy_target(struct pw_node_target *dst, const struct pw_node_target *src)
{
	dst->id = src->id;
	memcpy(dst->name, src->name, sizeof(dst->name));
	dst->node = src->node;
	dst->activation = src->activation;
	dst->system = src->system;
	dst->fd = src->fd;
}

struct pw_node_activation {
#define PW_NODE_ACTIVATION_NOT_TRIGGERED	0
#define PW_NODE_ACTIVATION_TRIGGERED		1
#define PW_NODE_ACTIVATION_AWAKE		2
#define PW_NODE_ACTIVATION_FINISHED		3
	uint32_t status;

	unsigned int version:1;
	unsigned int pending_sync:1;			/* a sync is pending */
	unsigned int pending_new_pos:1;			/* a new position is pending */

	struct pw_node_activation_state state[2];	/* one current state and one next state,
							 * as version flag */
	uint64_t signal_time;
	uint64_t awake_time;
	uint64_t finish_time;
	uint64_t prev_signal_time;

	/* updates */
	struct spa_io_segment reposition;		/* reposition info, used when driver reposition_owner
							 * has this node id */
	struct spa_io_segment segment;			/* update for the extra segment info fields.
							 * used when driver segment_owner has this node id */

	/* for drivers, shared with all nodes */
	uint32_t segment_owner[16];			/* id of owners for each segment info struct.
							 * nodes that want to update segment info need to
							 * CAS their node id in this array. */
	uint32_t padding[15];
#define PW_NODE_ACTIVATION_FLAG_NONE		0
#define PW_NODE_ACTIVATION_FLAG_PROFILER	(1<<0)	/* the profiler is running */
	uint32_t flags;					/* extra flags */
	struct spa_io_position position;		/* contains current position and segment info.
							 * extra info is updated by nodes that have set
							 * themselves as owner in the segment structs */

	uint64_t sync_timeout;				/* sync timeout in nanoseconds
							 * position goes to RUNNING without waiting any
							 * longer for sync clients. */
	uint64_t sync_left;				/* number of cycles before timeout */


	float cpu_load[3];				/* averaged over short, medium, long time */
	uint32_t xrun_count;				/* number of xruns */
	uint64_t xrun_time;				/* time of last xrun in microseconds */
	uint64_t xrun_delay;				/* delay of last xrun in microseconds */
	uint64_t max_delay;				/* max of all xruns in microseconds */

#define PW_NODE_ACTIVATION_COMMAND_NONE		0
#define PW_NODE_ACTIVATION_COMMAND_START	1
#define PW_NODE_ACTIVATION_COMMAND_STOP		2
	uint32_t command;				/* next command */
	uint32_t reposition_owner;			/* owner id with new reposition info, last one
							 * to update wins */
};

#define pw_impl_node_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_impl_node_events, m, v, ##__VA_ARGS__)
#define pw_impl_node_emit_destroy(n)			pw_impl_node_emit(n, destroy, 0)
#define pw_impl_node_emit_free(n)			pw_impl_node_emit(n, free, 0)
#define pw_impl_node_emit_initialized(n)		pw_impl_node_emit(n, initialized, 0)
#define pw_impl_node_emit_port_init(n,p)		pw_impl_node_emit(n, port_init, 0, p)
#define pw_impl_node_emit_port_added(n,p)		pw_impl_node_emit(n, port_added, 0, p)
#define pw_impl_node_emit_port_removed(n,p)		pw_impl_node_emit(n, port_removed, 0, p)
#define pw_impl_node_emit_info_changed(n,i)		pw_impl_node_emit(n, info_changed, 0, i)
#define pw_impl_node_emit_port_info_changed(n,p,i)	pw_impl_node_emit(n, port_info_changed, 0, p, i)
#define pw_impl_node_emit_active_changed(n,a)		pw_impl_node_emit(n, active_changed, 0, a)
#define pw_impl_node_emit_state_request(n,s)		pw_impl_node_emit(n, state_request, 0, s)
#define pw_impl_node_emit_state_changed(n,o,s,e)	pw_impl_node_emit(n, state_changed, 0, o, s, e)
#define pw_impl_node_emit_async_complete(n,s,r)		pw_impl_node_emit(n, async_complete, 0, s, r)
#define pw_impl_node_emit_result(n,s,r,t,result)	pw_impl_node_emit(n, result, 0, s, r, t, result)
#define pw_impl_node_emit_event(n,e)			pw_impl_node_emit(n, event, 0, e)
#define pw_impl_node_emit_driver_changed(n,o,d)		pw_impl_node_emit(n, driver_changed, 0, o, d)
#define pw_impl_node_emit_peer_added(n,p)		pw_impl_node_emit(n, peer_added, 0, p)
#define pw_impl_node_emit_peer_removed(n,p)		pw_impl_node_emit(n, peer_removed, 0, p)

#define pw_impl_node_rt_emit(o,m,v,...) spa_hook_list_call(&o->rt_listener_list, struct pw_impl_node_rt_events, m, v, ##__VA_ARGS__)
#define pw_impl_node_rt_emit_drained(n)			pw_impl_node_rt_emit(n, drained, 0)
#define pw_impl_node_rt_emit_xrun(n)			pw_impl_node_rt_emit(n, xrun, 0)
#define pw_impl_node_rt_emit_start(n)			pw_impl_node_rt_emit(n, start, 0)
#define pw_impl_node_rt_emit_complete(n)		pw_impl_node_rt_emit(n, complete, 0)
#define pw_impl_node_rt_emit_incomplete(n)		pw_impl_node_rt_emit(n, incomplete, 0)
#define pw_impl_node_rt_emit_timeout(n)			pw_impl_node_rt_emit(n, timeout, 0)

struct pw_impl_node {
	struct pw_context *context;		/**< context object */
	struct spa_list link;		/**< link in context node_list */
	struct pw_global *global;	/**< global for this node */
	struct spa_hook global_listener;

	struct pw_properties *properties;	/**< properties of the node */

	struct pw_node_info info;		/**< introspectable node info */
	struct spa_param_info params[MAX_PARAMS];

	char *name;				/** for debug */

	uint32_t priority_driver;	/** priority for being driver */
	char **groups;			/** groups to schedule this node in */
	char **link_groups;		/** groups this node is linked to */
	uint64_t spa_flags;

	unsigned int registered:1;
	unsigned int active:1;		/**< if the node is active */
	unsigned int live:1;		/**< if the node is live */
	unsigned int driver:1;		/**< if the node can drive the graph */
	unsigned int exported:1;	/**< if the node is exported */
	unsigned int remote:1;		/**< if the node is implemented remotely */
	unsigned int driving:1;		/**< a driving node is one of the driver nodes that
					  *  is selected to drive the graph */
	unsigned int visited:1;		/**< for sorting */
	unsigned int want_driver:1;	/**< this node wants to be assigned to a driver */
	unsigned int in_passive:1;	/**< node input links should be passive */
	unsigned int out_passive:1;	/**< node output links should be passive */
	unsigned int runnable:1;	/**< node is runnable */
	unsigned int freewheel:1;	/**< if this is the freewheel driver */
	unsigned int loopchecked:1;	/**< for feedback loop checking */
	unsigned int always_process:1;	/**< this node wants to always be processing, even when idle */
	unsigned int lock_quantum:1;	/**< don't change graph quantum */
	unsigned int lock_rate:1;	/**< don't change graph rate */
	unsigned int transport_sync:1;	/**< supports transport sync */
	unsigned int target_pending:1;	/**< a quantum/rate update is pending */
	unsigned int moved:1;		/**< the node was moved drivers */
	unsigned int added:1;		/**< the node was add to graph */
	unsigned int pause_on_idle:1;	/**< Pause processing when IDLE */
	unsigned int suspend_on_idle:1;
	unsigned int need_resume:1;
	unsigned int forced_rate:1;
	unsigned int forced_quantum:1;
	unsigned int trigger:1;		/**< has the TRIGGER property and needs an extra
					  *  trigger to start processing. */
	unsigned int can_suspend:1;
	unsigned int checked;		/**< for sorting */

	uint32_t port_user_data_size;	/**< extra size for port user data */

	struct spa_list driver_link;
	struct pw_impl_node *driver_node;
	struct spa_list follower_list;
	struct spa_list follower_link;

	struct spa_list sort_link;	/**< link used to sort nodes */

	struct spa_list peer_list;	/* list of peers */

	struct spa_node *node;		/**< SPA node implementation */
	struct spa_hook listener;

	struct spa_list input_ports;		/**< list of input ports */
	struct pw_map input_port_map;		/**< map from port_id to port */
	struct spa_list output_ports;		/**< list of output ports */
	struct pw_map output_port_map;		/**< map from port_id to port */

	struct spa_hook_list listener_list;
	struct spa_hook_list rt_listener_list;

	struct pw_loop *data_loop;		/**< the data loop for this node */
	struct spa_system *data_system;

	struct spa_fraction latency;		/**< requested latency */
	struct spa_fraction max_latency;	/**< maximum latency */
	struct spa_fraction rate;		/**< requested rate */
	uint32_t force_quantum;			/**< forced quantum */
	uint32_t force_rate;			/**< forced rate */
	uint32_t stamp;				/**< stamp of last update */
	struct spa_source source;		/**< source to remotely trigger this node */
	struct pw_memblock *activation;
	struct {
		struct spa_io_clock *clock;	/**< io area of the clock or NULL */
		struct spa_io_position *position;

		struct spa_list target_list;		/* list of targets to signal after
							 * this node */
		struct pw_node_target driver_target;	/* driver target that we signal */
		struct spa_list input_mix;		/* our input ports (and mixers) */
		struct spa_list output_mix;		/* output ports (and mixers) */

		struct pw_node_target target;		/* our target that is signaled by the
							   driver */
		struct spa_list driver_link;		/* our link in driver */

		struct spa_ratelimit rate_limit;
	} rt;
	struct spa_fraction target_rate;
	uint64_t target_quantum;

	uint64_t driver_start;
	uint64_t elapsed;		/* elapsed time in playing */

	void *user_data;                /**< extra user data */
};

struct pw_impl_port_mix {
	struct spa_list link;
	struct spa_list rt_link;
	struct pw_impl_port *p;
	struct {
		enum spa_direction direction;
		uint32_t port_id;
	} port;
	struct spa_io_buffers *io;
	uint32_t id;
	uint32_t peer_id;
	unsigned int have_buffers:1;
	unsigned int active:1;
};

struct pw_impl_port_implementation {
#define PW_VERSION_PORT_IMPLEMENTATION       0
	uint32_t version;

	int (*init_mix) (void *data, struct pw_impl_port_mix *mix);
	int (*release_mix) (void *data, struct pw_impl_port_mix *mix);
};

#define pw_impl_port_call(p,m,v,...)				\
({								\
	int _res = 0;						\
	spa_callbacks_call_res(&(p)->impl,			\
			struct pw_impl_port_implementation,	\
			_res, m, v, ## __VA_ARGS__);		\
	_res;							\
})

#define pw_impl_port_call_init_mix(p,m)		pw_impl_port_call(p,init_mix,0,m)
#define pw_impl_port_call_release_mix(p,m)	pw_impl_port_call(p,release_mix,0,m)

#define pw_impl_port_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_impl_port_events, m, v, ##__VA_ARGS__)
#define pw_impl_port_emit_destroy(p)			pw_impl_port_emit(p, destroy, 0)
#define pw_impl_port_emit_free(p)			pw_impl_port_emit(p, free, 0)
#define pw_impl_port_emit_initialized(p)		pw_impl_port_emit(p, initialized, 0)
#define pw_impl_port_emit_info_changed(p,i)		pw_impl_port_emit(p, info_changed, 0, i)
#define pw_impl_port_emit_link_added(p,l)		pw_impl_port_emit(p, link_added, 0, l)
#define pw_impl_port_emit_link_removed(p,l)		pw_impl_port_emit(p, link_removed, 0, l)
#define pw_impl_port_emit_state_changed(p,o,s,e)	pw_impl_port_emit(p, state_changed, 0, o, s, e)
#define pw_impl_port_emit_control_added(p,c)		pw_impl_port_emit(p, control_added, 0, c)
#define pw_impl_port_emit_control_removed(p,c)		pw_impl_port_emit(p, control_removed, 0, c)
#define pw_impl_port_emit_param_changed(p,i)		pw_impl_port_emit(p, param_changed, 1, i)
#define pw_impl_port_emit_latency_changed(p)		pw_impl_port_emit(p, latency_changed, 2)
#define pw_impl_port_emit_tag_changed(p)		pw_impl_port_emit(p, tag_changed, 3)

#define PW_IMPL_PORT_IS_CONTROL(port)	SPA_FLAG_MASK((port)->flags, \
						PW_IMPL_PORT_FLAG_BUFFERS|PW_IMPL_PORT_FLAG_CONTROL,\
						PW_IMPL_PORT_FLAG_CONTROL)
struct pw_impl_port {
	struct spa_list link;		/**< link in node port_list */

	struct pw_impl_node *node;		/**< owner node */
	struct pw_global *global;	/**< global for this port */
	struct spa_hook global_listener;

#define PW_IMPL_PORT_FLAG_TO_REMOVE		(1<<0)		/**< if the port should be removed from the
								  *  implementation when destroyed */
#define PW_IMPL_PORT_FLAG_BUFFERS		(1<<1)		/**< port has data */
#define PW_IMPL_PORT_FLAG_CONTROL		(1<<2)		/**< port has control */
#define PW_IMPL_PORT_FLAG_NO_MIXER		(1<<3)		/**< don't try to add mixer to port */
	uint32_t flags;
	uint64_t spa_flags;

	enum pw_direction direction;	/**< port direction */
	uint32_t port_id;		/**< port id */

	enum pw_impl_port_state state;	/**< state of the port */
	const char *error;		/**< error state */

	struct pw_properties *properties;	/**< properties of the port */
	struct pw_port_info info;
	struct spa_param_info params[MAX_PARAMS];

	struct pw_buffers buffers;	/**< buffers managed by this port, only on
					  *  output ports, shared with all links */

	struct spa_list links;		/**< list of \ref pw_impl_link */

	struct spa_list control_list[2];/**< list of \ref pw_control indexed by direction */

	struct spa_hook_list listener_list;

	struct spa_callbacks impl;

	struct spa_node *mix;		/**< port buffer mix/split */
#define PW_IMPL_PORT_MIX_FLAG_MULTI	(1<<0)	/**< multi input or output */
#define PW_IMPL_PORT_MIX_FLAG_MIX_ONLY	(1<<1)	/**< only negotiate mix ports */
#define PW_IMPL_PORT_MIX_FLAG_NEGOTIATE	(1<<2)	/**< negotiate buffers  */
	uint32_t mix_flags;		/**< flags for the mixing */
	struct spa_handle *mix_handle;	/**< mix plugin handle */
	struct pw_buffers mix_buffers;	/**< buffers between mixer and node */

	struct spa_list mix_list;	/**< list of \ref pw_impl_port_mix */
	struct pw_map mix_port_map;	/**< map from port_id from mixer */
	uint32_t n_mix;

	struct {
		struct spa_io_buffers io;	/**< io area of the port */
		struct spa_list node_link;
	} rt;					/**< data only accessed from the data thread */
	unsigned int added:1;
	unsigned int destroying:1;
	unsigned int passive:1;
	int busy_count;

	struct spa_latency_info latency[2];	/**< latencies */
	unsigned int have_latency_param:1;
	unsigned int ignore_latency:1;
	unsigned int have_latency:1;

	unsigned int have_tag_param:1;
	struct spa_pod *tag[2];			/**< tags */

	void *owner_data;		/**< extra owner data */
	void *user_data;                /**< extra user data */
};

struct pw_control_link {
	struct spa_list out_link;
	struct spa_list in_link;
	struct pw_control *output;
	struct pw_control *input;
	uint32_t out_port;
	uint32_t in_port;
	unsigned int valid:1;
};

struct pw_node_peer {
	int ref;
	int active_count;
	struct spa_list link;			/**< link in peer list */
	struct pw_impl_node *output;		/**< the output node */
	struct pw_node_target target;		/**< target of the input node */
};

#define pw_impl_link_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_impl_link_events, m, v, ##__VA_ARGS__)
#define pw_impl_link_emit_destroy(l)		pw_impl_link_emit(l, destroy, 0)
#define pw_impl_link_emit_free(l)		pw_impl_link_emit(l, free, 0)
#define pw_impl_link_emit_initialized(l)	pw_impl_link_emit(l, initialized, 0)
#define pw_impl_link_emit_info_changed(l,i)	pw_impl_link_emit(l, info_changed, 0, i)
#define pw_impl_link_emit_state_changed(l,...)	pw_impl_link_emit(l, state_changed, 0, __VA_ARGS__)
#define pw_impl_link_emit_port_unlinked(l,p)	pw_impl_link_emit(l, port_unlinked, 0, p)

struct pw_impl_link {
	struct pw_context *context;		/**< context object */
	struct spa_list link;			/**< link in context link_list */
	struct pw_global *global;		/**< global for this link */
	struct spa_hook global_listener;

	char *name;

	struct pw_link_info info;		/**< introspectable link info */
	struct pw_properties *properties;	/**< extra link properties */

	struct spa_io_buffers *io;		/**< link io area */

	struct pw_impl_port *output;		/**< output port */
	struct spa_list output_link;		/**< link in output port links */
	struct pw_impl_port *input;		/**< input port */
	struct spa_list input_link;		/**< link in input port links */

	struct spa_hook_list listener_list;

	struct pw_control_link control;
	struct pw_control_link notify;

	struct pw_node_peer *peer;

	struct {
		struct pw_impl_port_mix out_mix;	/**< port added to the output mixer */
		struct pw_impl_port_mix in_mix;		/**< port added to the input mixer */
	} rt;

	void *user_data;

	unsigned int registered:1;
	unsigned int feedback:1;
	unsigned int preparing:1;
	unsigned int prepared:1;
	unsigned int passive:1;
	unsigned int destroyed:1;
};

#define pw_resource_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_resource_events, m, v, ##__VA_ARGS__)

#define pw_resource_emit_destroy(o)	pw_resource_emit(o, destroy, 0)
#define pw_resource_emit_pong(o,s)	pw_resource_emit(o, pong, 0, s)
#define pw_resource_emit_error(o,s,r,m)	pw_resource_emit(o, error, 0, s, r, m)

struct pw_resource {
	struct spa_interface impl;	/**< object implementation */

	struct pw_context *context;	/**< the context object */
	struct pw_global *global;	/**< global of resource */
	struct spa_list link;		/**< link in global resource_list */

	struct pw_impl_client *client;	/**< owner client */

	uint32_t id;			/**< per client unique id, index in client objects */
	uint32_t permissions;		/**< resource permissions */
	const char *type;		/**< type of the client interface */
	uint32_t version;		/**< version of the client interface */
	uint32_t bound_id;		/**< global id we are bound to */
	int refcount;

	unsigned int removed:1;		/**< resource was removed from server */
	unsigned int destroyed:1;	/**< resource was destroyed */

	struct spa_hook_list listener_list;
	struct spa_hook_list object_listener_list;

        const struct pw_protocol_marshal *marshal;

	void *user_data;		/**< extra user data */
};

#define pw_proxy_emit(o,m,v,...) spa_hook_list_call(&o->listener_list, struct pw_proxy_events, m, v, ##__VA_ARGS__)
#define pw_proxy_emit_destroy(p)	pw_proxy_emit(p, destroy, 0)
#define pw_proxy_emit_bound(p,g)	pw_proxy_emit(p, bound, 0, g)
#define pw_proxy_emit_removed(p)	pw_proxy_emit(p, removed, 0)
#define pw_proxy_emit_done(p,s)		pw_proxy_emit(p, done, 0, s)
#define pw_proxy_emit_error(p,s,r,m)	pw_proxy_emit(p, error, 0, s, r, m)
#define pw_proxy_emit_bound_props(p,g,r) pw_proxy_emit(p, bound_props, 1, g, r)

struct pw_proxy {
	struct spa_interface impl;	/**< object implementation */

	struct pw_core *core;		/**< the owner core of this proxy */

	uint32_t id;			/**< client side id */
	const char *type;		/**< type of the interface */
	uint32_t version;		/**< client side version */
	uint32_t bound_id;		/**< global id we are bound to */
	int refcount;
	unsigned int zombie:1;		/**< proxy is removed locally and waiting to
					  *  be removed from server */
	unsigned int removed:1;		/**< proxy was removed from server */
	unsigned int destroyed:1;	/**< proxy was destroyed by client */
	unsigned int in_map:1;		/**< proxy is in core object map */

	struct spa_hook_list listener_list;
	struct spa_hook_list object_listener_list;

	const struct pw_protocol_marshal *marshal;	/**< protocol specific marshal functions */

	void *user_data;		/**< extra user data */
};

struct pw_core {
	struct pw_proxy proxy;

	struct pw_context *context;		/**< context */
	struct spa_list link;			/**< link in context core_list */
	struct pw_properties *properties;	/**< extra properties */

	struct pw_mempool *pool;		/**< memory pool */
	struct spa_hook core_listener;
	struct spa_hook proxy_core_listener;

	struct pw_map objects;			/**< map of client side proxy objects
						 *   indexed with the client id */
	struct pw_client *client;		/**< proxy for the client object */

	struct spa_list stream_list;		/**< list of \ref pw_stream objects */
	struct spa_list filter_list;		/**< list of \ref pw_stream objects */

	struct pw_protocol_client *conn;	/**< the protocol client connection */
	int recv_seq;				/**< last received sequence number */
	int send_seq;				/**< last protocol result code */
	uint64_t recv_generation;		/**< last received registry generation */

	unsigned int removed:1;
	unsigned int destroyed:1;

	void *user_data;			/**< extra user data */
};

#define pw_stream_emit(s,m,v,...) spa_hook_list_call(&s->listener_list, struct pw_stream_events, m, v, ##__VA_ARGS__)
#define pw_stream_emit_destroy(s)		pw_stream_emit(s, destroy, 0)
#define pw_stream_emit_state_changed(s,o,n,e)	pw_stream_emit(s, state_changed,0,o,n,e)
#define pw_stream_emit_io_changed(s,i,a,t)	pw_stream_emit(s, io_changed,0,i,a,t)
#define pw_stream_emit_param_changed(s,i,p)	pw_stream_emit(s, param_changed,0,i,p)
#define pw_stream_emit_add_buffer(s,b)		pw_stream_emit(s, add_buffer, 0, b)
#define pw_stream_emit_remove_buffer(s,b)	pw_stream_emit(s, remove_buffer, 0, b)
#define pw_stream_emit_process(s)		pw_stream_emit(s, process, 0)
#define pw_stream_emit_drained(s)		pw_stream_emit(s, drained,0)
#define pw_stream_emit_control_info(s,i,c)	pw_stream_emit(s, control_info, 0, i, c)
#define pw_stream_emit_command(s,c)		pw_stream_emit(s, command,1,c)
#define pw_stream_emit_trigger_done(s)		pw_stream_emit(s, trigger_done,2)


struct pw_stream {
	struct pw_core *core;			/**< the owner core */
	struct spa_hook core_listener;

	struct spa_list link;			/**< link in the core */

	char *name;				/**< the name of the stream */
	struct pw_properties *properties;	/**< properties of the stream */

	uint32_t node_id;			/**< node id for remote node, available from
						  *  CONFIGURE state and higher */
	enum pw_stream_state state;		/**< stream state */
	char *error;				/**< error reason when state is in error */
	int error_res;				/**< error code when in error */

	struct spa_hook_list listener_list;

	struct pw_proxy *proxy;
	struct spa_hook proxy_listener;

	struct pw_impl_node *node;
	struct spa_hook node_listener;
	struct spa_hook node_rt_listener;

	struct spa_list controls;
};

#define pw_filter_emit(s,m,v,...) spa_hook_list_call(&(s)->listener_list, struct pw_filter_events, m, v, ##__VA_ARGS__)
#define pw_filter_emit_destroy(s)		pw_filter_emit(s, destroy, 0)
#define pw_filter_emit_state_changed(s,o,n,e)	pw_filter_emit(s, state_changed,0,o,n,e)
#define pw_filter_emit_io_changed(s,p,i,d,t)	pw_filter_emit(s, io_changed,0,p,i,d,t)
#define pw_filter_emit_param_changed(s,p,i,f)	pw_filter_emit(s, param_changed,0,p,i,f)
#define pw_filter_emit_add_buffer(s,p,b)	pw_filter_emit(s, add_buffer, 0, p, b)
#define pw_filter_emit_remove_buffer(s,p,b)	pw_filter_emit(s, remove_buffer, 0, p, b)
#define pw_filter_emit_process(s,p)		pw_filter_emit(s, process, 0, p)
#define pw_filter_emit_drained(s)		pw_filter_emit(s, drained, 0)
#define pw_filter_emit_command(s,c)		pw_filter_emit(s, command, 1, c)


struct pw_filter {
	struct pw_core *core;	/**< the owner core proxy */
	struct spa_hook core_listener;

	struct spa_list link;			/**< link in the core proxy */

	char *name;				/**< the name of the filter */
	struct pw_properties *properties;	/**< properties of the filter */

	uint32_t node_id;			/**< node id for remote node, available from
						  *  CONFIGURE state and higher */
	enum pw_filter_state state;		/**< filter state */
	char *error;				/**< error reason when state is in error */
	int error_res;				/**< error code when in error */

	struct spa_hook_list listener_list;

	struct pw_proxy *proxy;
	struct spa_hook proxy_listener;

	struct pw_impl_node *node;
	struct spa_hook node_listener;

	struct spa_list controls;
};

#define pw_impl_factory_emit(s,m,v,...) spa_hook_list_call(&s->listener_list, struct pw_impl_factory_events, m, v, ##__VA_ARGS__)

#define pw_impl_factory_emit_destroy(s)		pw_impl_factory_emit(s, destroy, 0)
#define pw_impl_factory_emit_free(s)		pw_impl_factory_emit(s, free, 0)
#define pw_impl_factory_emit_initialized(s)	pw_impl_factory_emit(s, initialized, 0)

struct pw_impl_factory {
	struct pw_context *context;		/**< the context */
	struct spa_list link;		/**< link in context factory_list */
	struct pw_global *global;	/**< global for this factory */
	struct spa_hook global_listener;

	struct pw_factory_info info;	/**< introspectable factory info */
	struct pw_properties *properties;	/**< properties of the factory */

	struct spa_hook_list listener_list;	/**< event listeners */

	struct spa_callbacks impl;

	void *user_data;

	unsigned int registered:1;
};

#define pw_control_emit(c,m,v,...) spa_hook_list_call(&c->listener_list, struct pw_control_events, m, v, ##__VA_ARGS__)
#define pw_control_emit_destroy(c)	pw_control_emit(c, destroy, 0)
#define pw_control_emit_free(c)		pw_control_emit(c, free, 0)
#define pw_control_emit_linked(c,o)	pw_control_emit(c, linked, 0, o)
#define pw_control_emit_unlinked(c,o)	pw_control_emit(c, unlinked, 0, o)

struct pw_control {
	struct spa_list link;		/**< link in context control_list */
	struct pw_context *context;		/**< the context */

	struct pw_impl_port *port;		/**< owner port or NULL */
	struct spa_list port_link;	/**< link in port control_list */

	enum spa_direction direction;	/**< the direction */
	struct spa_list links;		/**< list of pw_control_link */

	uint32_t id;
	int32_t size;

	struct spa_hook_list listener_list;

	void *user_data;
};

/** Find a good format between 2 ports */
int pw_context_find_format(struct pw_context *context,
			struct pw_impl_port *output,
			struct pw_impl_port *input,
			struct pw_properties *props,
			uint32_t n_format_filters,
			struct spa_pod **format_filters,
			struct spa_pod **format,
			struct spa_pod_builder *builder,
			char **error);

int pw_context_debug_port_params(struct pw_context *context,
		struct spa_node *node, enum spa_direction direction,
		uint32_t port_id, uint32_t id, int err, const char *debug, ...);

int pw_proxy_init(struct pw_proxy *proxy, struct pw_core *core, const char *type, uint32_t version);

void pw_proxy_remove(struct pw_proxy *proxy);

int pw_context_recalc_graph(struct pw_context *context, const char *reason);

void pw_impl_port_update_info(struct pw_impl_port *port, const struct spa_port_info *info);

int pw_impl_port_register(struct pw_impl_port *port,
		     struct pw_properties *properties);

/** Get the user data of a port, the size of the memory was given \ref in pw_context_create_port */
void * pw_impl_port_get_user_data(struct pw_impl_port *port);

int pw_impl_port_set_mix(struct pw_impl_port *port, struct spa_node *node, uint32_t flags);

int pw_impl_port_init_mix(struct pw_impl_port *port, struct pw_impl_port_mix *mix);
int pw_impl_port_release_mix(struct pw_impl_port *port, struct pw_impl_port_mix *mix);

void pw_impl_port_update_state(struct pw_impl_port *port, enum pw_impl_port_state state, int res, char *error);

/** Unlink a port */
void pw_impl_port_unlink(struct pw_impl_port *port);

/** Destroy a port */
void pw_impl_port_destroy(struct pw_impl_port *port);

/** Iterate the params of the given port. The callback should return
 * 1 to fetch the next item, 0 to stop iteration or <0 on error.
 * The function returns 0 on success or the error returned by the callback. */
int pw_impl_port_for_each_param(struct pw_impl_port *port,
			   int seq, uint32_t param_id,
			   uint32_t index, uint32_t max,
			   const struct spa_pod *filter,
			   int (*callback) (void *data, int seq,
					    uint32_t id, uint32_t index, uint32_t next,
					    struct spa_pod *param),
			   void *data);

int pw_impl_port_for_each_filtered_param(struct pw_impl_port *in_port,
				    struct pw_impl_port *out_port,
				    int seq,
				    uint32_t in_param_id,
				    uint32_t out_param_id,
				    const struct spa_pod *filter,
				    int (*callback) (void *data, int seq,
						     uint32_t id, uint32_t index, uint32_t next,
						     struct spa_pod *param),
				    void *data);

/** Iterate the links of the port. The callback should return
 * 0 to fetch the next item, any other value stops the iteration and returns
 * the value. When all callbacks return 0, this function returns 0 when all
 * items are iterated. */
int pw_impl_port_for_each_link(struct pw_impl_port *port,
			   int (*callback) (void *data, struct pw_impl_link *link),
			   void *data);

/** Set a param on a port, use SPA_ID_INVALID for mix_id to set
 * the param on all mix ports */
int pw_impl_port_set_param(struct pw_impl_port *port,
		uint32_t id, uint32_t flags, const struct spa_pod *param);

/** Use buffers on a port */
int pw_impl_port_use_buffers(struct pw_impl_port *port, struct pw_impl_port_mix *mix, uint32_t flags,
		struct spa_buffer **buffers, uint32_t n_buffers);

int pw_impl_port_recalc_latency(struct pw_impl_port *port);
int pw_impl_port_recalc_tag(struct pw_impl_port *port);

/** Change the state of the node */
int pw_impl_node_set_state(struct pw_impl_node *node, enum pw_node_state state);


int pw_impl_node_update_ports(struct pw_impl_node *node);

int pw_impl_node_set_driver(struct pw_impl_node *node, struct pw_impl_node *driver);

int pw_impl_node_trigger(struct pw_impl_node *node);

/** Prepare a link
  * Starts the negotiation of formats and buffers on \a link */
int pw_impl_link_prepare(struct pw_impl_link *link);
/** starts streaming on a link */
int pw_impl_link_activate(struct pw_impl_link *link);

/** Deactivate a link */
int pw_impl_link_deactivate(struct pw_impl_link *link);

struct pw_control *
pw_control_new(struct pw_context *context,
	       struct pw_impl_port *owner,		/**< can be NULL */
	       uint32_t id, uint32_t size,
	       size_t user_data_size		/**< extra user data */);

int pw_control_add_link(struct pw_control *control, uint32_t cmix,
		struct pw_control *other, uint32_t omix,
		struct pw_control_link *link);

int pw_control_remove_link(struct pw_control_link *link);

void pw_control_destroy(struct pw_control *control);

void pw_impl_client_unref(struct pw_impl_client *client);

#define PW_LOG_OBJECT_POD	(1<<0)
#define PW_LOG_OBJECT_FORMAT	(1<<1)
void pw_log_log_object(enum spa_log_level level, const struct spa_log_topic *topic,
		const char *file, int line, const char *func, uint32_t flags,
		const void *object);

#define pw_log_object(lev,t,fl,obj)				\
({								\
	if (SPA_UNLIKELY(pw_log_topic_enabled(lev,t)))		\
		pw_log_log_object(lev,t,__FILE__,__LINE__,	\
				__func__,(fl),(obj));		\
})

#define pw_log_pod(lev,pod) pw_log_object(lev,PW_LOG_TOPIC_DEFAULT,PW_LOG_OBJECT_POD,pod)
#define pw_log_format(lev,pod) pw_log_object(lev,PW_LOG_TOPIC_DEFAULT,PW_LOG_OBJECT_FORMAT,pod)

bool pw_log_is_default(void);

void pw_log_init(void);
void pw_log_deinit(void);

void pw_random_init(void);

void pw_settings_init(struct pw_context *context);
int pw_settings_expose(struct pw_context *context);
void pw_settings_clean(struct pw_context *context);

bool pw_should_dlclose(void);

/** \endcond */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_PRIVATE_H */
