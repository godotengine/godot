/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Collabora Ltd. */
/*                         @author George Kiagiadakis <george.kiagiadakis@collabora.com> */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_EXT_SESSION_MANAGER_INTERFACES_H
#define PIPEWIRE_EXT_SESSION_MANAGER_INTERFACES_H

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>

#include "introspect.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup pw_session_manager
 * \{
 */

#define PW_TYPE_INTERFACE_Session		PW_TYPE_INFO_INTERFACE_BASE "Session"
#define PW_SESSION_PERM_MASK			PW_PERM_RWX
#define PW_VERSION_SESSION			0
struct pw_session;

#define PW_TYPE_INTERFACE_Endpoint		PW_TYPE_INFO_INTERFACE_BASE "Endpoint"
#define PW_ENDPOINT_PERM_MASK			PW_PERM_RWX
#define PW_VERSION_ENDPOINT			0
struct pw_endpoint;

#define PW_TYPE_INTERFACE_EndpointStream	PW_TYPE_INFO_INTERFACE_BASE "EndpointStream"
#define PW_ENDPOINT_STREAM_PERM_MASK		PW_PERM_RWX
#define PW_VERSION_ENDPOINT_STREAM		0
struct pw_endpoint_stream;

#define PW_TYPE_INTERFACE_EndpointLink		PW_TYPE_INFO_INTERFACE_BASE "EndpointLink"
#define PW_ENDPOINT_LINK_PERM_MASK		PW_PERM_RWX
#define PW_VERSION_ENDPOINT_LINK		0
struct pw_endpoint_link;

/* Session */

#define PW_SESSION_EVENT_INFO		0
#define PW_SESSION_EVENT_PARAM		1
#define PW_SESSION_EVENT_NUM		2

struct pw_session_events {
#define PW_VERSION_SESSION_EVENTS		0
	uint32_t version;			/**< version of this structure */

	/**
	 * Notify session info
	 *
	 * \param info info about the session
	 */
	void (*info) (void *data, const struct pw_session_info *info);

	/**
	 * Notify a session param
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

#define PW_SESSION_METHOD_ADD_LISTENER		0
#define PW_SESSION_METHOD_SUBSCRIBE_PARAMS	1
#define PW_SESSION_METHOD_ENUM_PARAMS		2
#define PW_SESSION_METHOD_SET_PARAM		3
#define PW_SESSION_METHOD_CREATE_LINK		4
#define PW_SESSION_METHOD_NUM			5

struct pw_session_methods {
#define PW_VERSION_SESSION_METHODS	0
	uint32_t version;			/**< version of this structure */

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_session_events *events,
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
	 * This requires X permissions.
	 */
	int (*subscribe_params) (void *object, uint32_t *ids, uint32_t n_ids);

	/**
	 * Enumerate session parameters
	 *
	 * Start enumeration of session parameters. For each param, a
	 * param event will be emitted.
	 *
	 * \param seq a sequence number returned in the reply
	 * \param id the parameter id to enumerate
	 * \param start the start index or 0 for the first param
	 * \param num the maximum number of params to retrieve
	 * \param filter a param filter or NULL
	 *
	 * This requires X permissions.
	 */
	int (*enum_params) (void *object, int seq,
			uint32_t id, uint32_t start, uint32_t num,
			const struct spa_pod *filter);

	/**
	 * Set a parameter on the session
	 *
	 * \param id the parameter id to set
	 * \param flags extra parameter flags
	 * \param param the parameter to set
	 *
	 * This requires X and W permissions.
	 */
	int (*set_param) (void *object, uint32_t id, uint32_t flags,
			  const struct spa_pod *param);
};

#define pw_session_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_session_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_session_add_listener(c,...)		pw_session_method(c,add_listener,0,__VA_ARGS__)
#define pw_session_subscribe_params(c,...)	pw_session_method(c,subscribe_params,0,__VA_ARGS__)
#define pw_session_enum_params(c,...)		pw_session_method(c,enum_params,0,__VA_ARGS__)
#define pw_session_set_param(c,...)		pw_session_method(c,set_param,0,__VA_ARGS__)


/* Endpoint */

#define PW_ENDPOINT_EVENT_INFO		0
#define PW_ENDPOINT_EVENT_PARAM		1
#define PW_ENDPOINT_EVENT_NUM		2

struct pw_endpoint_events {
#define PW_VERSION_ENDPOINT_EVENTS	0
	uint32_t version;			/**< version of this structure */

	/**
	 * Notify endpoint info
	 *
	 * \param info info about the endpoint
	 */
	void (*info) (void *data, const struct pw_endpoint_info *info);

	/**
	 * Notify a endpoint param
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

#define PW_ENDPOINT_METHOD_ADD_LISTENER		0
#define PW_ENDPOINT_METHOD_SUBSCRIBE_PARAMS	1
#define PW_ENDPOINT_METHOD_ENUM_PARAMS		2
#define PW_ENDPOINT_METHOD_SET_PARAM		3
#define PW_ENDPOINT_METHOD_CREATE_LINK		4
#define PW_ENDPOINT_METHOD_NUM			5

struct pw_endpoint_methods {
#define PW_VERSION_ENDPOINT_METHODS	0
	uint32_t version;			/**< version of this structure */

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_endpoint_events *events,
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
	 * This requires X permissions.
	 */
	int (*subscribe_params) (void *object, uint32_t *ids, uint32_t n_ids);

	/**
	 * Enumerate endpoint parameters
	 *
	 * Start enumeration of endpoint parameters. For each param, a
	 * param event will be emitted.
	 *
	 * \param seq a sequence number returned in the reply
	 * \param id the parameter id to enumerate
	 * \param start the start index or 0 for the first param
	 * \param num the maximum number of params to retrieve
	 * \param filter a param filter or NULL
	 *
	 * This requires X permissions.
	 */
	int (*enum_params) (void *object, int seq,
			uint32_t id, uint32_t start, uint32_t num,
			const struct spa_pod *filter);

	/**
	 * Set a parameter on the endpoint
	 *
	 * \param id the parameter id to set
	 * \param flags extra parameter flags
	 * \param param the parameter to set
	 *
	 * This requires X and W permissions.
	 */
	int (*set_param) (void *object, uint32_t id, uint32_t flags,
			  const struct spa_pod *param);

	/**
	 * Create a link
	 *
	 * This requires X permissions.
	 */
	int (*create_link) (void *object, const struct spa_dict *props);
};

#define pw_endpoint_method(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_endpoint_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_endpoint_add_listener(c,...)		pw_endpoint_method(c,add_listener,0,__VA_ARGS__)
#define pw_endpoint_subscribe_params(c,...)	pw_endpoint_method(c,subscribe_params,0,__VA_ARGS__)
#define pw_endpoint_enum_params(c,...)		pw_endpoint_method(c,enum_params,0,__VA_ARGS__)
#define pw_endpoint_set_param(c,...)		pw_endpoint_method(c,set_param,0,__VA_ARGS__)
#define pw_endpoint_create_link(c,...)		pw_endpoint_method(c,create_link,0,__VA_ARGS__)

/* Endpoint Stream */

#define PW_ENDPOINT_STREAM_EVENT_INFO		0
#define PW_ENDPOINT_STREAM_EVENT_PARAM		1
#define PW_ENDPOINT_STREAM_EVENT_NUM		2

struct pw_endpoint_stream_events {
#define PW_VERSION_ENDPOINT_STREAM_EVENTS	0
	uint32_t version;			/**< version of this structure */

	/**
	 * Notify endpoint stream info
	 *
	 * \param info info about the endpoint stream
	 */
	void (*info) (void *data, const struct pw_endpoint_stream_info *info);

	/**
	 * Notify a endpoint stream param
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

#define PW_ENDPOINT_STREAM_METHOD_ADD_LISTENER		0
#define PW_ENDPOINT_STREAM_METHOD_SUBSCRIBE_PARAMS	1
#define PW_ENDPOINT_STREAM_METHOD_ENUM_PARAMS		2
#define PW_ENDPOINT_STREAM_METHOD_SET_PARAM		3
#define PW_ENDPOINT_STREAM_METHOD_NUM			4

struct pw_endpoint_stream_methods {
#define PW_VERSION_ENDPOINT_STREAM_METHODS	0
	uint32_t version;			/**< version of this structure */

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_endpoint_stream_events *events,
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
	 * This requires X permissions.
	 */
	int (*subscribe_params) (void *object, uint32_t *ids, uint32_t n_ids);

	/**
	 * Enumerate stream parameters
	 *
	 * Start enumeration of stream parameters. For each param, a
	 * param event will be emitted.
	 *
	 * \param seq a sequence number returned in the reply
	 * \param id the parameter id to enumerate
	 * \param start the start index or 0 for the first param
	 * \param num the maximum number of params to retrieve
	 * \param filter a param filter or NULL
	 *
	 * This requires X permissions.
	 */
	int (*enum_params) (void *object, int seq,
			uint32_t id, uint32_t start, uint32_t num,
			const struct spa_pod *filter);

	/**
	 * Set a parameter on the stream
	 *
	 * \param id the parameter id to set
	 * \param flags extra parameter flags
	 * \param param the parameter to set
	 *
	 * This requires X and W permissions.
	 */
	int (*set_param) (void *object, uint32_t id, uint32_t flags,
			  const struct spa_pod *param);
};

#define pw_endpoint_stream_method(o,method,version,...)		\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_endpoint_stream_methods, _res,	\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_endpoint_stream_add_listener(c,...)		pw_endpoint_stream_method(c,add_listener,0,__VA_ARGS__)
#define pw_endpoint_stream_subscribe_params(c,...)	pw_endpoint_stream_method(c,subscribe_params,0,__VA_ARGS__)
#define pw_endpoint_stream_enum_params(c,...)		pw_endpoint_stream_method(c,enum_params,0,__VA_ARGS__)
#define pw_endpoint_stream_set_param(c,...)		pw_endpoint_stream_method(c,set_param,0,__VA_ARGS__)

/* Endpoint Link */

#define PW_ENDPOINT_LINK_EVENT_INFO		0
#define PW_ENDPOINT_LINK_EVENT_PARAM		1
#define PW_ENDPOINT_LINK_EVENT_NUM		2

struct pw_endpoint_link_events {
#define PW_VERSION_ENDPOINT_LINK_EVENTS	0
	uint32_t version;			/**< version of this structure */

	/**
	 * Notify endpoint link info
	 *
	 * \param info info about the endpoint link
	 */
	void (*info) (void *data, const struct pw_endpoint_link_info *info);

	/**
	 * Notify a endpoint link param
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

#define PW_ENDPOINT_LINK_METHOD_ADD_LISTENER		0
#define PW_ENDPOINT_LINK_METHOD_SUBSCRIBE_PARAMS	1
#define PW_ENDPOINT_LINK_METHOD_ENUM_PARAMS		2
#define PW_ENDPOINT_LINK_METHOD_SET_PARAM		3
#define PW_ENDPOINT_LINK_METHOD_REQUEST_STATE		4
#define PW_ENDPOINT_LINK_METHOD_DESTROY			5
#define PW_ENDPOINT_LINK_METHOD_NUM			6

struct pw_endpoint_link_methods {
#define PW_VERSION_ENDPOINT_LINK_METHODS	0
	uint32_t version;			/**< version of this structure */

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_endpoint_link_events *events,
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
	 * This requires X permissions.
	 */
	int (*subscribe_params) (void *object, uint32_t *ids, uint32_t n_ids);

	/**
	 * Enumerate link parameters
	 *
	 * Start enumeration of link parameters. For each param, a
	 * param event will be emitted.
	 *
	 * \param seq a sequence number returned in the reply
	 * \param id the parameter id to enumerate
	 * \param start the start index or 0 for the first param
	 * \param num the maximum number of params to retrieve
	 * \param filter a param filter or NULL
	 *
	 * This requires X permissions.
	 */
	int (*enum_params) (void *object, int seq,
			uint32_t id, uint32_t start, uint32_t num,
			const struct spa_pod *filter);

	/**
	 * Set a parameter on the link
	 *
	 * \param id the parameter id to set
	 * \param flags extra parameter flags
	 * \param param the parameter to set
	 *
	 * This requires X and W permissions.
	 */
	int (*set_param) (void *object, uint32_t id, uint32_t flags,
			  const struct spa_pod *param);

	/**
	 * Request a state on the link.
	 *
	 * This requires X and W permissions.
	 */
	int (*request_state) (void *object, enum pw_endpoint_link_state state);
};

#define pw_endpoint_link_method(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_endpoint_link_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_endpoint_link_add_listener(c,...)		pw_endpoint_link_method(c,add_listener,0,__VA_ARGS__)
#define pw_endpoint_link_subscribe_params(c,...)	pw_endpoint_link_method(c,subscribe_params,0,__VA_ARGS__)
#define pw_endpoint_link_enum_params(c,...)		pw_endpoint_link_method(c,enum_params,0,__VA_ARGS__)
#define pw_endpoint_link_set_param(c,...)		pw_endpoint_link_method(c,set_param,0,__VA_ARGS__)
#define pw_endpoint_link_request_state(c,...)		pw_endpoint_link_method(c,request_state,0,__VA_ARGS__)


/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_EXT_SESSION_MANAGER_INTERFACES_H */
