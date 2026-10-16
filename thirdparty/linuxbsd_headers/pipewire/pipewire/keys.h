/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_KEYS_H
#define PIPEWIRE_KEYS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <pipewire/utils.h>
/**
 * \defgroup pw_keys Key Names
 *
 * A collection of keys that are used to add extra information on objects.
 *
 * Keys that start with "pipewire." are in general set-once and then
 * read-only. They are usually used for security sensitive information that
 * needs to be fixed.
 *
 * Properties from other objects can also appear. This usually suggests some
 * sort of parent/child or owner/owned relationship.
 *
 * \addtogroup pw_keys
 * \{
 */
#define PW_KEY_PROTOCOL			"pipewire.protocol"	/**< protocol used for connection */
#define PW_KEY_ACCESS			"pipewire.access"	/**< how the client access is controlled */
#define PW_KEY_CLIENT_ACCESS		"pipewire.client.access"/**< how the client wants to be access
								  *  controlled */

/** Various keys related to the identity of a client process and its security.
 * Must be obtained from trusted sources by the protocol and placed as
 * read-only properties. */
#define PW_KEY_SEC_PID			"pipewire.sec.pid"	/**< Client pid, set by protocol */
#define PW_KEY_SEC_UID			"pipewire.sec.uid"	/**< Client uid, set by protocol*/
#define PW_KEY_SEC_GID			"pipewire.sec.gid"	/**< client gid, set by protocol*/
#define PW_KEY_SEC_LABEL		"pipewire.sec.label"	/**< client security label, set by protocol*/

#define PW_KEY_SEC_SOCKET		"pipewire.sec.socket"	/**< client socket name, set by protocol */

#define PW_KEY_LIBRARY_NAME_SYSTEM	"library.name.system"	/**< name of the system library to use */
#define PW_KEY_LIBRARY_NAME_LOOP	"library.name.loop"	/**< name of the loop library to use */
#define PW_KEY_LIBRARY_NAME_DBUS	"library.name.dbus"	/**< name of the dbus library to use */

/** object properties */
#define PW_KEY_OBJECT_PATH		"object.path"		/**< unique path to construct the object */
#define PW_KEY_OBJECT_ID		"object.id"		/**< a global object id */
#define PW_KEY_OBJECT_SERIAL		"object.serial"		/**< a 64 bit object serial number. This is a number
								  *  incremented for each object that is created.
								  *  The lower 32 bits are guaranteed to never be
								  *  SPA_ID_INVALID. */
#define PW_KEY_OBJECT_LINGER		"object.linger"		/**< the object lives on even after the client
								  *  that created it has been destroyed */
#define PW_KEY_OBJECT_REGISTER		"object.register"	/**< If the object should be registered. */
#define PW_KEY_OBJECT_EXPORT		"object.export"		/**< If the object should be exported,
								  *  since 0.3.72 */

/* config */
#define PW_KEY_CONFIG_PREFIX		"config.prefix"		/**< a config prefix directory */
#define PW_KEY_CONFIG_NAME		"config.name"		/**< a config file name */
#define PW_KEY_CONFIG_OVERRIDE_PREFIX	"config.override.prefix"	/**< a config override prefix directory */
#define PW_KEY_CONFIG_OVERRIDE_NAME	"config.override.name"	/**< a config override file name */

/* context */
#define PW_KEY_CONTEXT_PROFILE_MODULES	"context.profile.modules"	/**< a context profile for modules, deprecated */
#define PW_KEY_USER_NAME		"context.user-name"	/**< The user name that runs pipewire */
#define PW_KEY_HOST_NAME		"context.host-name"	/**< The host name of the machine */

/* core */
#define PW_KEY_CORE_NAME		"core.name"		/**< The name of the core. Default is
								  *  `pipewire-<username>-<pid>`, overwritten
								  *  by env(PIPEWIRE_CORE) */
#define PW_KEY_CORE_VERSION		"core.version"		/**< The version of the core. */
#define PW_KEY_CORE_DAEMON		"core.daemon"		/**< If the core is listening for connections. */

#define PW_KEY_CORE_ID			"core.id"		/**< the core id */
#define PW_KEY_CORE_MONITORS		"core.monitors"		/**< the apis monitored by core. */

/* cpu */
#define PW_KEY_CPU_MAX_ALIGN		"cpu.max-align"		/**< maximum alignment needed to support
								  *  all CPU optimizations */
#define PW_KEY_CPU_CORES		"cpu.cores"		/**< number of cores */

/* priorities */
#define PW_KEY_PRIORITY_SESSION		"priority.session"	/**< priority in session manager */
#define PW_KEY_PRIORITY_DRIVER		"priority.driver"	/**< priority to be a driver */

/* remote keys */
#define PW_KEY_REMOTE_NAME		"remote.name"		/**< The name of the remote to connect to,
								  *  default pipewire-0, overwritten by
								  *  env(PIPEWIRE_REMOTE). May also be
								  *  a SPA-JSON array of sockets, to be tried
								  *  in order. */
#define PW_KEY_REMOTE_INTENTION		"remote.intention"	/**< The intention of the remote connection,
								  *  "generic", "screencast" */

/** application keys */
#define PW_KEY_APP_NAME			"application.name"	/**< application name. Ex: "Totem Music Player" */
#define PW_KEY_APP_ID			"application.id"	/**< a textual id for identifying an
								  *  application logically. Ex: "org.gnome.Totem" */
#define PW_KEY_APP_VERSION		"application.version"   /**< application version. Ex: "1.2.0" */
#define PW_KEY_APP_ICON			"application.icon"	/**< aa base64 blob with PNG image data */
#define PW_KEY_APP_ICON_NAME		"application.icon-name"	/**< an XDG icon name for the application.
								  *  Ex: "totem" */
#define PW_KEY_APP_LANGUAGE		"application.language"	/**< application language if applicable, in
								  *  standard POSIX format. Ex: "en_GB" */

#define PW_KEY_APP_PROCESS_ID		"application.process.id"	/**< process id  (pid)*/
#define PW_KEY_APP_PROCESS_BINARY	"application.process.binary"	/**< binary name */
#define PW_KEY_APP_PROCESS_USER		"application.process.user"	/**< user name */
#define PW_KEY_APP_PROCESS_HOST		"application.process.host"	/**< host name */
#define PW_KEY_APP_PROCESS_MACHINE_ID	"application.process.machine-id" /**< the D-Bus host id the
									   *  application runs on */
#define PW_KEY_APP_PROCESS_SESSION_ID	"application.process.session-id" /**< login session of the
									   *  application, on Unix the
									   *  value of $XDG_SESSION_ID. */
/** window system */
#define PW_KEY_WINDOW_X11_DISPLAY	"window.x11.display"	/**< the X11 display string. Ex. ":0.0" */

/** Client properties */
#define PW_KEY_CLIENT_ID		"client.id"		/**< a client id */
#define PW_KEY_CLIENT_NAME		"client.name"		/**< the client name */
#define PW_KEY_CLIENT_API		"client.api"		/**< the client api used to access
								  *  PipeWire */

/** Node keys */
#define PW_KEY_NODE_ID			"node.id"		/**< node id */
#define PW_KEY_NODE_NAME		"node.name"		/**< node name */
#define PW_KEY_NODE_NICK		"node.nick"		/**< short node name */
#define PW_KEY_NODE_DESCRIPTION		"node.description"	/**< localized human readable node one-line
								  *  description. Ex. "Foobar USB Headset" */
#define PW_KEY_NODE_PLUGGED		"node.plugged"		/**< when the node was created. As a uint64 in
								  *  nanoseconds. */

#define PW_KEY_NODE_SESSION		"node.session"		/**< the session id this node is part of */
#define PW_KEY_NODE_GROUP		"node.group"		/**< the group id this node is part of. Nodes
								  *  in the same group are always scheduled
								  *  with the same driver. Can be an array of
								  *  group names. */
#define PW_KEY_NODE_EXCLUSIVE		"node.exclusive"	/**< node wants exclusive access to resources */
#define PW_KEY_NODE_AUTOCONNECT		"node.autoconnect"	/**< node wants to be automatically connected
								  *  to a compatible node */
#define PW_KEY_NODE_LATENCY		"node.latency"		/**< the requested latency of the node as
								  *  a fraction. Ex: 128/48000 */
#define PW_KEY_NODE_MAX_LATENCY		"node.max-latency"	/**< the maximum supported latency of the
								  *  node as a fraction. Ex: 1024/48000 */
#define PW_KEY_NODE_LOCK_QUANTUM	"node.lock-quantum"	/**< don't change quantum when this node
								  *  is active */
#define PW_KEY_NODE_FORCE_QUANTUM	"node.force-quantum"	/**< force a quantum while the node is
								  *  active */
#define PW_KEY_NODE_RATE		"node.rate"		/**< the requested rate of the graph as
								  *  a fraction. Ex: 1/48000 */
#define PW_KEY_NODE_LOCK_RATE		"node.lock-rate"	/**< don't change rate when this node
								  *  is active */
#define PW_KEY_NODE_FORCE_RATE		"node.force-rate"	/**< force a rate while the node is
								  *  active. A value of 0 takes the denominator
								  *  of node.rate */

#define PW_KEY_NODE_DONT_RECONNECT	"node.dont-reconnect"	/**< don't reconnect this node. The node is
								  *  initially linked to target.object or the
								  *  default node. If the target is removed,
								  *  the node is destroyed */
#define PW_KEY_NODE_ALWAYS_PROCESS	"node.always-process"	/**< process even when unlinked */
#define PW_KEY_NODE_WANT_DRIVER		"node.want-driver"	/**< the node wants to be grouped with a driver
								  *  node in order to schedule the graph. */
#define PW_KEY_NODE_PAUSE_ON_IDLE	"node.pause-on-idle"	/**< pause the node when idle */
#define PW_KEY_NODE_SUSPEND_ON_IDLE	"node.suspend-on-idle"	/**< suspend the node when idle */
#define PW_KEY_NODE_CACHE_PARAMS	"node.cache-params"	/**< cache the node params */
#define PW_KEY_NODE_TRANSPORT_SYNC	"node.transport.sync"	/**< the node handles transport sync */
#define PW_KEY_NODE_DRIVER		"node.driver"		/**< node can drive the graph */
#define PW_KEY_NODE_STREAM		"node.stream"		/**< node is a stream, the server side should
								  *  add a converter */
#define PW_KEY_NODE_VIRTUAL		"node.virtual"		/**< the node is some sort of virtual
								  *  object */
#define PW_KEY_NODE_PASSIVE		"node.passive"		/**< indicate that a node wants passive links
								  *  on output/input/all ports when the value is
								  *  "out"/"in"/"true" respectively */
#define PW_KEY_NODE_LINK_GROUP		"node.link-group"	/**< the node is internally linked to
								  *  nodes with the same link-group. Can be an
								  *  array of group names. */
#define PW_KEY_NODE_NETWORK		"node.network"		/**< the node is on a network */
#define PW_KEY_NODE_TRIGGER		"node.trigger"		/**< the node is not scheduled automatically
								  *   based on the dependencies in the graph
								  *   but it will be triggered explicitly. */
#define PW_KEY_NODE_CHANNELNAMES		"node.channel-names"		/**< names of node's
									*   channels (unrelated to positions) */
#define PW_KEY_NODE_DEVICE_PORT_NAME_PREFIX			"node.device-port-name-prefix"		/** override
									*		port name prefix for device ports, like capture and playback
									*		or disable the prefix completely if an empty string is provided */

/** Port keys */
#define PW_KEY_PORT_ID			"port.id"		/**< port id */
#define PW_KEY_PORT_NAME		"port.name"		/**< port name */
#define PW_KEY_PORT_DIRECTION		"port.direction"	/**< the port direction, one of "in" or "out"
								  *  or "control" and "notify" for control ports */
#define PW_KEY_PORT_ALIAS		"port.alias"		/**< port alias */
#define PW_KEY_PORT_PHYSICAL		"port.physical"		/**< if this is a physical port */
#define PW_KEY_PORT_TERMINAL		"port.terminal"		/**< if this port consumes the data */
#define PW_KEY_PORT_CONTROL		"port.control"		/**< if this port is a control port */
#define PW_KEY_PORT_MONITOR		"port.monitor"		/**< if this port is a monitor port */
#define PW_KEY_PORT_CACHE_PARAMS	"port.cache-params"	/**< cache the node port params */
#define PW_KEY_PORT_EXTRA		"port.extra"		/**< api specific extra port info, API name
								  *  should be prefixed. "jack:flags:56" */
#define PW_KEY_PORT_PASSIVE		"port.passive"		/**< the ports wants passive links, since 0.3.67 */
#define PW_KEY_PORT_IGNORE_LATENCY	"port.ignore-latency"	/**< latency ignored by peers, since 0.3.71 */

/** link properties */
#define PW_KEY_LINK_ID			"link.id"		/**< a link id */
#define PW_KEY_LINK_INPUT_NODE		"link.input.node"	/**< input node id of a link */
#define PW_KEY_LINK_INPUT_PORT		"link.input.port"	/**< input port id of a link */
#define PW_KEY_LINK_OUTPUT_NODE		"link.output.node"	/**< output node id of a link */
#define PW_KEY_LINK_OUTPUT_PORT		"link.output.port"	/**< output port id of a link */
#define PW_KEY_LINK_PASSIVE		"link.passive"		/**< indicate that a link is passive and
								  *  does not cause the graph to be
								  *  runnable. */
#define PW_KEY_LINK_FEEDBACK		"link.feedback"		/**< indicate that a link is a feedback
								  *  link and the target will receive data
								  *  in the next cycle */

/** device properties */
#define PW_KEY_DEVICE_ID		"device.id"		/**< device id */
#define PW_KEY_DEVICE_NAME		"device.name"		/**< device name */
#define PW_KEY_DEVICE_PLUGGED		"device.plugged"	/**< when the device was created. As a uint64 in
								  *  nanoseconds. */
#define PW_KEY_DEVICE_NICK		"device.nick"		/**< a short device nickname */
#define PW_KEY_DEVICE_STRING		"device.string"		/**< device string in the underlying layer's
								  *  format. Ex. "surround51:0" */
#define PW_KEY_DEVICE_API		"device.api"		/**< API this device is accessed with.
								  *  Ex. "alsa", "v4l2" */
#define PW_KEY_DEVICE_DESCRIPTION	"device.description"	/**< localized human readable device one-line
								  *  description. Ex. "Foobar USB Headset" */
#define PW_KEY_DEVICE_BUS_PATH		"device.bus-path"	/**< bus path to the device in the OS'
								  *  format. Ex. "pci-0000:00:14.0-usb-0:3.2:1.0" */
#define PW_KEY_DEVICE_SERIAL		"device.serial"		/**< Serial number if applicable */
#define PW_KEY_DEVICE_VENDOR_ID		"device.vendor.id"	/**< vendor ID if applicable */
#define PW_KEY_DEVICE_VENDOR_NAME	"device.vendor.name"	/**< vendor name if applicable */
#define PW_KEY_DEVICE_PRODUCT_ID	"device.product.id"	/**< product ID if applicable */
#define PW_KEY_DEVICE_PRODUCT_NAME	"device.product.name"	/**< product name if applicable */
#define PW_KEY_DEVICE_CLASS		"device.class"		/**< device class */
#define PW_KEY_DEVICE_FORM_FACTOR	"device.form-factor"	/**< form factor if applicable. One of
								  *  "internal", "speaker", "handset", "tv",
								  *  "webcam", "microphone", "headset",
								  *  "headphone", "hands-free", "car", "hifi",
								  *  "computer", "portable" */
#define PW_KEY_DEVICE_BUS		"device.bus"		/**< bus of the device if applicable. One of
								  *  "isa", "pci", "usb", "firewire",
								  *  "bluetooth" */
#define PW_KEY_DEVICE_SUBSYSTEM		"device.subsystem"	/**< device subsystem */
#define PW_KEY_DEVICE_SYSFS_PATH	"device.sysfs.path"	/**< device sysfs path */
#define PW_KEY_DEVICE_ICON		"device.icon"		/**< icon for the device. A base64 blob
								  *  containing PNG image data */
#define PW_KEY_DEVICE_ICON_NAME		"device.icon-name"	/**< an XDG icon name for the device.
								  *  Ex. "sound-card-speakers-usb" */
#define PW_KEY_DEVICE_INTENDED_ROLES	"device.intended-roles"	/**< intended use. A space separated list of
								  *  roles (see PW_KEY_MEDIA_ROLE) this device
								  *  is particularly well suited for, due to
								  *  latency, quality or form factor. */
#define PW_KEY_DEVICE_CACHE_PARAMS	"device.cache-params"	/**< cache the device spa params */

/** module properties */
#define PW_KEY_MODULE_ID		"module.id"		/**< the module id */
#define PW_KEY_MODULE_NAME		"module.name"		/**< the name of the module */
#define PW_KEY_MODULE_AUTHOR		"module.author"		/**< the author's name */
#define PW_KEY_MODULE_DESCRIPTION	"module.description"	/**< a human readable one-line description
								  *  of the module's purpose.*/
#define PW_KEY_MODULE_USAGE		"module.usage"		/**< a human readable usage description of
								  *  the module's arguments. */
#define PW_KEY_MODULE_VERSION		"module.version"	/**< a version string for the module. */

/** Factory properties */
#define PW_KEY_FACTORY_ID		"factory.id"		/**< the factory id */
#define PW_KEY_FACTORY_NAME		"factory.name"		/**< the name of the factory */
#define PW_KEY_FACTORY_USAGE		"factory.usage"		/**< the usage of the factory */
#define PW_KEY_FACTORY_TYPE_NAME	"factory.type.name"	/**< the name of the type created by a factory */
#define PW_KEY_FACTORY_TYPE_VERSION	"factory.type.version"	/**< the version of the type created by a factory */

/** Stream properties */
#define PW_KEY_STREAM_IS_LIVE		"stream.is-live"	/**< Indicates that the stream is live. */
#define PW_KEY_STREAM_LATENCY_MIN	"stream.latency.min"	/**< The minimum latency of the stream. */
#define PW_KEY_STREAM_LATENCY_MAX	"stream.latency.max"	/**< The maximum latency of the stream */
#define PW_KEY_STREAM_MONITOR		"stream.monitor"	/**< Indicates that the stream is monitoring
								  *  and might select a less accurate but faster
								  *  conversion algorithm. Monitor streams are also
								  *  ignored when calculating the latency of their peer
								  *  ports (since 0.3.71).
								  */
#define PW_KEY_STREAM_DONT_REMIX	"stream.dont-remix"	/**< don't remix channels */
#define PW_KEY_STREAM_CAPTURE_SINK	"stream.capture.sink"	/**< Try to capture the sink output instead of
								  *  source output */

/** Media */
#define PW_KEY_MEDIA_TYPE		"media.type"		/**< Media type, one of
								  *  Audio, Video, Midi */
#define PW_KEY_MEDIA_CATEGORY		"media.category"	/**< Media Category:
								  *  Playback, Capture, Duplex, Monitor, Manager */
#define PW_KEY_MEDIA_ROLE		"media.role"		/**< Role: Movie, Music, Camera,
								  *  Screen, Communication, Game,
								  *  Notification, DSP, Production,
								  *  Accessibility, Test */
#define PW_KEY_MEDIA_CLASS		"media.class"		/**< class Ex: "Video/Source" */
#define PW_KEY_MEDIA_NAME		"media.name"		/**< media name. Ex: "Pink Floyd: Time" */
#define PW_KEY_MEDIA_TITLE		"media.title"		/**< title. Ex: "Time" */
#define PW_KEY_MEDIA_ARTIST		"media.artist"		/**< artist. Ex: "Pink Floyd" */
#define PW_KEY_MEDIA_COPYRIGHT		"media.copyright"	/**< copyright string */
#define PW_KEY_MEDIA_SOFTWARE		"media.software"	/**< generator software */
#define PW_KEY_MEDIA_LANGUAGE		"media.language"	/**< language in POSIX format. Ex: en_GB */
#define PW_KEY_MEDIA_FILENAME		"media.filename"	/**< filename */
#define PW_KEY_MEDIA_ICON		"media.icon"		/**< icon for the media, a base64 blob with
								  *  PNG image data */
#define PW_KEY_MEDIA_ICON_NAME		"media.icon-name"	/**< an XDG icon name for the media.
								  *  Ex: "audio-x-mp3" */
#define PW_KEY_MEDIA_COMMENT		"media.comment"		/**< extra comment */
#define PW_KEY_MEDIA_DATE		"media.date"		/**< date of the media */
#define PW_KEY_MEDIA_FORMAT		"media.format"		/**< format of the media */

/** format related properties */
#define PW_KEY_FORMAT_DSP		"format.dsp"		/**< a dsp format.
								  *  Ex: "32 bit float mono audio" */
/** audio related properties */
#define PW_KEY_AUDIO_CHANNEL		"audio.channel"		/**< an audio channel. Ex: "FL" */
#define PW_KEY_AUDIO_RATE		"audio.rate"		/**< an audio samplerate */
#define PW_KEY_AUDIO_CHANNELS		"audio.channels"	/**< number of audio channels */
#define PW_KEY_AUDIO_FORMAT		"audio.format"		/**< an audio format. Ex: "S16LE" */
#define PW_KEY_AUDIO_ALLOWED_RATES	"audio.allowed-rates"	/**< a list of allowed samplerates
								  *  ex. "[ 44100 48000 ]" */

/** video related properties */
#define PW_KEY_VIDEO_RATE		"video.framerate"	/**< a video framerate */
#define PW_KEY_VIDEO_FORMAT		"video.format"		/**< a video format */
#define PW_KEY_VIDEO_SIZE		"video.size"		/**< a video size as "<width>x<height" */

#define PW_KEY_TARGET_OBJECT		"target.object"		/**< a target object to link to. This can be
								  * and object name or object.serial */

#ifndef PW_REMOVE_DEPRECATED
# ifdef PW_ENABLE_DEPRECATED
#  define PW_KEY_PRIORITY_MASTER	"priority.master"	/**< deprecated, use priority.driver */
#  define PW_KEY_NODE_TARGET		"node.target"		/**< deprecated since 0.3.64, use target.object. */
# else
#  define PW_KEY_PRIORITY_MASTER	PW_DEPRECATED("priority.master")
#  define PW_KEY_NODE_TARGET		PW_DEPRECATED("node.target")
# endif /* PW_ENABLE_DEPRECATED */
#endif /* PW_REMOVE_DEPRECATED */

/** \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_KEYS_H */
