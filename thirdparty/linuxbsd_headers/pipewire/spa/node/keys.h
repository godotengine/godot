/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_NODE_KEYS_H
#define SPA_NODE_KEYS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_node
 * \{
 */

/** node keys */
#define SPA_KEY_NODE_NAME		"node.name"		/**< a node name */
#define SPA_KEY_NODE_DESCRIPTION	"node.description"	/**< localized human readable node one-line
								  *  description. Ex. "Foobar USB Headset" */
#define SPA_KEY_NODE_LATENCY		"node.latency"		/**< the requested node latency */
#define SPA_KEY_NODE_MAX_LATENCY	"node.max-latency"	/**< maximum supported latency */

#define SPA_KEY_NODE_DRIVER		"node.driver"		/**< the node can be a driver */
#define SPA_KEY_NODE_ALWAYS_PROCESS	"node.always-process"	/**< call the process function even if
								  *  not linked. */
#define SPA_KEY_NODE_PAUSE_ON_IDLE	"node.pause-on-idle"	/**< if the node should be paused
								  *  immediately when idle. */
#define SPA_KEY_NODE_MONITOR		"node.monitor"		/**< the node has monitor ports */


/** port keys */
#define SPA_KEY_PORT_NAME		"port.name"		/**< a port name */
#define SPA_KEY_PORT_ALIAS		"port.alias"		/**< a port alias */
#define SPA_KEY_PORT_MONITOR		"port.monitor"		/**< this port is a monitor port */
#define SPA_KEY_PORT_IGNORE_LATENCY	"port.ignore-latency"	/**< latency ignored by peers */


/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_NODE_KEYS_H */
