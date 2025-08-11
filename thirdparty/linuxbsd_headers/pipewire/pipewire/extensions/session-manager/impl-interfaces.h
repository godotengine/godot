/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Collabora Ltd. */
/*                         @author George Kiagiadakis <george.kiagiadakis@collabora.com> */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_EXT_SESSION_MANAGER_IMPL_INTERFACES_H
#define PIPEWIRE_EXT_SESSION_MANAGER_IMPL_INTERFACES_H

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>
#include <errno.h>

#include "introspect.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup pw_session_manager
 * \{
 */

#define PW_TYPE_INTERFACE_ClientEndpoint	PW_TYPE_INFO_INTERFACE_BASE "ClientEndpoint"

#define PW_VERSION_CLIENT_ENDPOINT		0
struct pw_client_endpoint;

#define PW_CLIENT_ENDPOINT_EVENT_SET_SESSION_ID		0
#define PW_CLIENT_ENDPOINT_EVENT_SET_PARAM		1
#define PW_CLIENT_ENDPOINT_EVENT_STREAM_SET_PARAM	2
#define PW_CLIENT_ENDPOINT_EVENT_CREATE_LINK		3
#define PW_CLIENT_ENDPOINT_EVENT_NUM			4

struct pw_client_endpoint_events {
#define PW_VERSION_CLIENT_ENDPOINT_EVENTS		0
	uint32_t version;		/**< version of this structure */

	/**
	 * Sets the session id of the \a endpoint.
	 *
	 * On endpoints that are not session masters, this method notifies
	 * the implementation that it has been associated with a session.
	 * The implementation is obliged to set this id in the
	 * #struct pw_endpoint_info \a session_id field.
	 *
	 * \param endpoint a #pw_endpoint
	 * \param id the session id associated with this endpoint
	 *
	 * \return 0 on success
	 *         -EINVAL when the session id has already been set
	 *         -ENOTSUP when the endpoint is a session master
	 */
	int (*set_session_id) (void *data, uint32_t session_id);

	/**
	 * Set the configurable parameter in \a endpoint.
	 *
	 * Usually, \a param will be obtained from enum_params and then
	 * modified but it is also possible to set another spa_pod
	 * as long as its keys and types match a supported object.
	 *
	 * Objects with property keys that are not known are ignored.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param endpoint a #struct pw_endpoint
	 * \param id the parameter id to configure
	 * \param flags additional flags
	 * \param param the parameter to configure
	 *
	 * \return 0 on success
	 *         -EINVAL when \a endpoint is NULL
	 *         -ENOTSUP when there are no parameters implemented on \a endpoint
	 *         -ENOENT the parameter is unknown
	 */
	int (*set_param) (void *data,
			  uint32_t id, uint32_t flags,
			  const struct spa_pod *param);

	/**
	 * Set a parameter on \a stream_id of \a endpoint.
	 *
	 * When \a param is NULL, the parameter will be unset.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param endpoint a #struct pw_endpoint
	 * \param stream_id the stream to configure
	 * \param id the parameter id to set
	 * \param flags optional flags
	 * \param param a #struct spa_pod with the parameter to set
	 * \return 0 on success
	 *         1 on success, the value of \a param might have been
	 *                changed depending on \a flags and the final value can
	 *                be found by doing stream_enum_params.
	 *         -EINVAL when \a endpoint is NULL or invalid arguments are given
	 *         -ESRCH when the type or size of a property is not correct.
	 *         -ENOENT when the param id is not found
	 */
	int (*stream_set_param) (void *data, uint32_t stream_id,
			         uint32_t id, uint32_t flags,
			         const struct spa_pod *param);

	int (*create_link) (void *data, const struct spa_dict *props);
};

#define PW_CLIENT_ENDPOINT_METHOD_ADD_LISTENER	0
#define PW_CLIENT_ENDPOINT_METHOD_UPDATE	1
#define PW_CLIENT_ENDPOINT_METHOD_STREAM_UPDATE	2
#define PW_CLIENT_ENDPOINT_METHOD_NUM		3

struct pw_client_endpoint_methods {
#define PW_VERSION_CLIENT_ENDPOINT_METHODS	0
	uint32_t version;		/**< version of this structure */

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_client_endpoint_events *events,
			void *data);

	/** Update endpoint information */
	int (*update) (void *object,
#define PW_CLIENT_ENDPOINT_UPDATE_PARAMS	(1 << 0)
#define PW_CLIENT_ENDPOINT_UPDATE_INFO		(1 << 1)
			uint32_t change_mask,
			uint32_t n_params,
			const struct spa_pod **params,
			const struct pw_endpoint_info *info);

	/** Update stream information */
	int (*stream_update) (void *object,
				uint32_t stream_id,
#define PW_CLIENT_ENDPOINT_STREAM_UPDATE_PARAMS		(1 << 0)
#define PW_CLIENT_ENDPOINT_STREAM_UPDATE_INFO		(1 << 1)
#define PW_CLIENT_ENDPOINT_STREAM_UPDATE_DESTROYED	(1 << 2)
				uint32_t change_mask,
				uint32_t n_params,
				const struct spa_pod **params,
				const struct pw_endpoint_stream_info *info);
};

#define pw_client_endpoint_method(o,method,version,...)		\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_client_endpoint_methods, _res,	\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_client_endpoint_add_listener(o,...)	pw_client_endpoint_method(o,add_listener,0,__VA_ARGS__)
#define pw_client_endpoint_update(o,...)	pw_client_endpoint_method(o,update,0,__VA_ARGS__)
#define pw_client_endpoint_stream_update(o,...)	pw_client_endpoint_method(o,stream_update,0,__VA_ARGS__)

#define PW_TYPE_INTERFACE_ClientSession		PW_TYPE_INFO_INTERFACE_BASE "ClientSession"

#define PW_VERSION_CLIENT_SESSION 0
struct pw_client_session;

#define PW_CLIENT_SESSION_EVENT_SET_PARAM		0
#define PW_CLIENT_SESSION_EVENT_LINK_SET_PARAM		1
#define PW_CLIENT_SESSION_EVENT_LINK_REQUEST_STATE	2
#define PW_CLIENT_SESSION_EVENT_NUM			3

struct pw_client_session_events {
#define PW_VERSION_CLIENT_SESSION_EVENTS		0
	uint32_t version;		/**< version of this structure */

	/**
	 * Set the configurable parameter in \a session.
	 *
	 * Usually, \a param will be obtained from enum_params and then
	 * modified but it is also possible to set another spa_pod
	 * as long as its keys and types match a supported object.
	 *
	 * Objects with property keys that are not known are ignored.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param session a #struct pw_session
	 * \param id the parameter id to configure
	 * \param flags additional flags
	 * \param param the parameter to configure
	 *
	 * \return 0 on success
	 *         -EINVAL when \a session is NULL
	 *         -ENOTSUP when there are no parameters implemented on \a session
	 *         -ENOENT the parameter is unknown
	 */
	int (*set_param) (void *data,
			  uint32_t id, uint32_t flags,
			  const struct spa_pod *param);

	/**
	 * Set a parameter on \a link_id of \a session.
	 *
	 * When \a param is NULL, the parameter will be unset.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param session a #struct pw_session
	 * \param link_id the link to configure
	 * \param id the parameter id to set
	 * \param flags optional flags
	 * \param param a #struct spa_pod with the parameter to set
	 * \return 0 on success
	 *         1 on success, the value of \a param might have been
	 *                changed depending on \a flags and the final value can
	 *                be found by doing link_enum_params.
	 *         -EINVAL when \a session is NULL or invalid arguments are given
	 *         -ESRCH when the type or size of a property is not correct.
	 *         -ENOENT when the param id is not found
	 */
	int (*link_set_param) (void *data, uint32_t link_id,
			       uint32_t id, uint32_t flags,
			       const struct spa_pod *param);

	int (*link_request_state) (void *data, uint32_t link_id, uint32_t state);
};

#define PW_CLIENT_SESSION_METHOD_ADD_LISTENER	0
#define PW_CLIENT_SESSION_METHOD_UPDATE		1
#define PW_CLIENT_SESSION_METHOD_LINK_UPDATE	2
#define PW_CLIENT_SESSION_METHOD_NUM		3

struct pw_client_session_methods {
#define PW_VERSION_CLIENT_SESSION_METHODS		0
	uint32_t version;		/**< version of this structure */

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_client_session_events *events,
			void *data);

	/** Update session information */
	int (*update) (void *object,
#define PW_CLIENT_SESSION_UPDATE_PARAMS		(1 << 0)
#define PW_CLIENT_SESSION_UPDATE_INFO		(1 << 1)
			uint32_t change_mask,
			uint32_t n_params,
			const struct spa_pod **params,
			const struct pw_session_info *info);

	/** Update link information */
	int (*link_update) (void *object,
				uint32_t link_id,
#define PW_CLIENT_SESSION_LINK_UPDATE_PARAMS		(1 << 0)
#define PW_CLIENT_SESSION_LINK_UPDATE_INFO		(1 << 1)
#define PW_CLIENT_SESSION_LINK_UPDATE_DESTROYED		(1 << 2)
				uint32_t change_mask,
				uint32_t n_params,
				const struct spa_pod **params,
				const struct pw_endpoint_link_info *info);
};

#define pw_client_session_method(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_client_session_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_client_session_add_listener(o,...)	pw_client_session_method(o,add_listener,0,__VA_ARGS__)
#define pw_client_session_update(o,...)		pw_client_session_method(o,update,0,__VA_ARGS__)
#define pw_client_session_link_update(o,...)	pw_client_session_method(o,link_update,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_EXT_SESSION_MANAGER_IMPL_INTERFACES_H */
