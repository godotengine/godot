/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Collabora Ltd. */
/*                         @author George Kiagiadakis <george.kiagiadakis@collabora.com> */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_EXT_SESSION_MANAGER_KEYS_H
#define PIPEWIRE_EXT_SESSION_MANAGER_KEYS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup pw_session_manager
 * \{
 */

#define PW_KEY_SESSION_ID		"session.id"		/**< id of a session manager */

#define PW_KEY_ENDPOINT_ID		"endpoint.id"		/**< id of an endpoint */
#define PW_KEY_ENDPOINT_NAME		"endpoint.name"		/**< the name of an endpoint */
#define PW_KEY_ENDPOINT_MONITOR		"endpoint.monitor"	/**< endpoint is monitor of given endpoint */
#define PW_KEY_ENDPOINT_CLIENT_ID	"endpoint.client.id"	/**< client of the endpoint */
#define PW_KEY_ENDPOINT_ICON_NAME	"endpoint.icon-name"	/**< an XDG icon name for the device.
								  *  Ex. "sound-card-speakers-usb" */
#define PW_KEY_ENDPOINT_AUTOCONNECT	"endpoint.autoconnect"	/**< try to automatically connect this
								  *  endpoint. */
#define PW_KEY_ENDPOINT_TARGET		"endpoint.target"	/**< the suggested target to connect to */

#define PW_KEY_ENDPOINT_STREAM_ID		"endpoint-stream.id"		/**< id of a stream */
#define PW_KEY_ENDPOINT_STREAM_NAME		"endpoint-stream.name"		/**< unique name of a stream */
#define PW_KEY_ENDPOINT_STREAM_DESCRIPTION	"endpoint-stream.description"	/**< description of a stream */

#define PW_KEY_ENDPOINT_LINK_OUTPUT_ENDPOINT	"endpoint-link.output.endpoint"	/**< output endpoint of link */
#define PW_KEY_ENDPOINT_LINK_OUTPUT_STREAM	"endpoint-link.output.stream"	/**< output stream of link */
#define PW_KEY_ENDPOINT_LINK_INPUT_ENDPOINT	"endpoint-link.input.endpoint"	/**< input endpoint of link */
#define PW_KEY_ENDPOINT_LINK_INPUT_STREAM	"endpoint-link.input.stream"	/**< input stream of link */

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_EXT_SESSION_MANAGER_KEYS_H */
