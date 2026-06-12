/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_UTILS_KEYS_H
#define SPA_UTILS_KEYS_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup spa_keys  Key Names
 * Key names used by SPA plugins
 */

/**
 * \addtogroup spa_keys
 * \{
 */

/** for objects */
#define SPA_KEY_OBJECT_PATH		"object.path"			/**< a unique path to
									  *  identity the object */

#define SPA_KEY_MEDIA_CLASS		"media.class"			/**< Media class
									  *  Ex. "Audio/Device",
									  *  "Video/Source",... */
#define SPA_KEY_MEDIA_ROLE		"media.role"			/**< Role: Movie, Music, Camera,
									  *  Screen, Communication, Game,
									  *  Notification, DSP, Production,
									  *  Accessibility, Test */
/** keys for udev api */
#define SPA_KEY_API_UDEV		"api.udev"			/**< key for the udev api */
#define SPA_KEY_API_UDEV_MATCH		"api.udev.match"		/**< udev subsystem match */

/** keys for alsa api */
#define SPA_KEY_API_ALSA		"api.alsa"			/**< key for the alsa api */
#define SPA_KEY_API_ALSA_PATH		"api.alsa.path"			/**< alsa device path as can be
									  *  used in snd_pcm_open() and
									  *  snd_ctl_open(). */
#define SPA_KEY_API_ALSA_CARD		"api.alsa.card"			/**< alsa card number */
#define SPA_KEY_API_ALSA_USE_UCM	"api.alsa.use-ucm"		/**< if UCM should be used */
#define SPA_KEY_API_ALSA_IGNORE_DB	"api.alsa.ignore-dB"		/**< if decibel info should be ignored */
#define SPA_KEY_API_ALSA_OPEN_UCM	"api.alsa.open.ucm"		/**< if UCM should be opened card */
#define SPA_KEY_API_ALSA_DISABLE_LONGNAME	\
					"api.alsa.disable-longname"	/**< if card long name should not be passed to MIDI port */
#define SPA_KEY_API_ALSA_BIND_CTLS	"api.alsa.bind-ctls"		/**< alsa controls to bind as params */

/** info from alsa card_info */
#define SPA_KEY_API_ALSA_CARD_ID	"api.alsa.card.id"		/**< id from card_info */
#define SPA_KEY_API_ALSA_CARD_COMPONENTS	\
					"api.alsa.card.components"	/**< components from card_info */
#define SPA_KEY_API_ALSA_CARD_DRIVER	"api.alsa.card.driver"		/**< driver from card_info */
#define SPA_KEY_API_ALSA_CARD_NAME	"api.alsa.card.name"		/**< name from card_info */
#define SPA_KEY_API_ALSA_CARD_LONGNAME	"api.alsa.card.longname"	/**< longname from card_info */
#define SPA_KEY_API_ALSA_CARD_MIXERNAME	"api.alsa.card.mixername"	/**< mixername from card_info */

/** info from alsa pcm_info */
#define SPA_KEY_API_ALSA_PCM_ID		"api.alsa.pcm.id"		/**< id from pcm_info */
#define SPA_KEY_API_ALSA_PCM_CARD	"api.alsa.pcm.card"		/**< card from pcm_info */
#define SPA_KEY_API_ALSA_PCM_NAME	"api.alsa.pcm.name"		/**< name from pcm_info */
#define SPA_KEY_API_ALSA_PCM_SUBNAME	"api.alsa.pcm.subname"		/**< subdevice_name from pcm_info */
#define SPA_KEY_API_ALSA_PCM_STREAM	"api.alsa.pcm.stream"		/**< stream type from pcm_info */
#define SPA_KEY_API_ALSA_PCM_CLASS	"api.alsa.pcm.class"		/**< class from pcm_info as string */
#define SPA_KEY_API_ALSA_PCM_DEVICE	"api.alsa.pcm.device"		/**< device from pcm_info */
#define SPA_KEY_API_ALSA_PCM_SUBDEVICE	"api.alsa.pcm.subdevice"	/**< subdevice from pcm_info */
#define SPA_KEY_API_ALSA_PCM_SUBCLASS	"api.alsa.pcm.subclass"		/**< subclass from pcm_info as string */
#define SPA_KEY_API_ALSA_PCM_SYNC_ID	"api.alsa.pcm.sync-id"		/**< sync id */

/** keys for v4l2 api */
#define SPA_KEY_API_V4L2		"api.v4l2"			/**< key for the v4l2 api */
#define SPA_KEY_API_V4L2_PATH		"api.v4l2.path"			/**< v4l2 device path as can be
									  *  used in open() */

/** keys for libcamera api */
#define SPA_KEY_API_LIBCAMERA		"api.libcamera"			/**< key for the libcamera api */
#define SPA_KEY_API_LIBCAMERA_PATH	"api.libcamera.path"	/**< libcamera device path as can be
									  *  used in open() */
#define SPA_KEY_API_LIBCAMERA_LOCATION	"api.libcamera.location"	/**< location of the camera:
									  * "front", "back" or "external" */

/** info from libcamera_capability */
#define SPA_KEY_API_LIBCAMERA_CAP_DRIVER	"api.libcamera.cap.driver"	/**< driver from capbility */
#define SPA_KEY_API_LIBCAMERA_CAP_CARD	"api.libcamera.cap.card"		/**< caps from capability */
#define SPA_KEY_API_LIBCAMERA_CAP_BUS_INFO	"api.libcamera.cap.bus_info"/**< bus_info from capability */
#define SPA_KEY_API_LIBCAMERA_CAP_VERSION	"api.libcamera.cap.version"	/**< version from capability as %u.%u.%u */
#define SPA_KEY_API_LIBCAMERA_CAP_CAPABILITIES	\
					"api.libcamera.cap.capabilities"	/**< capabilities from capability */
#define SPA_KEY_API_LIBCAMERA_CAP_DEVICE_CAPS	\
					"api.libcamera.cap.device-caps"	/**< device_caps from capability */
/** info from v4l2_capability */
#define SPA_KEY_API_V4L2_CAP_DRIVER	"api.v4l2.cap.driver"		/**< driver from capbility */
#define SPA_KEY_API_V4L2_CAP_CARD	"api.v4l2.cap.card"		/**< caps from capability */
#define SPA_KEY_API_V4L2_CAP_BUS_INFO	"api.v4l2.cap.bus_info"		/**< bus_info from capability */
#define SPA_KEY_API_V4L2_CAP_VERSION	"api.v4l2.cap.version"		/**< version from capability as %u.%u.%u */
#define SPA_KEY_API_V4L2_CAP_CAPABILITIES	\
					"api.v4l2.cap.capabilities"	/**< capabilities from capability */
#define SPA_KEY_API_V4L2_CAP_DEVICE_CAPS	\
					"api.v4l2.cap.device-caps"	/**< device_caps from capability */


/** keys for bluez5 api */
#define SPA_KEY_API_BLUEZ5		"api.bluez5"			/**< key for the bluez5 api */
#define SPA_KEY_API_BLUEZ5_PATH		"api.bluez5.path"		/**< a bluez5 path */
#define SPA_KEY_API_BLUEZ5_DEVICE	"api.bluez5.device"		/**< an internal bluez5 device */
#define SPA_KEY_API_BLUEZ5_CONNECTION	"api.bluez5.connection"		/**< bluez5 device connection status */
#define SPA_KEY_API_BLUEZ5_TRANSPORT	"api.bluez5.transport"		/**< an internal bluez5 transport */
#define SPA_KEY_API_BLUEZ5_PROFILE	"api.bluez5.profile"		/**< a bluetooth profile */
#define SPA_KEY_API_BLUEZ5_ADDRESS	"api.bluez5.address"		/**< a bluetooth address */
#define SPA_KEY_API_BLUEZ5_CODEC	"api.bluez5.codec"		/**< a bluetooth codec */
#define SPA_KEY_API_BLUEZ5_CLASS	"api.bluez5.class"		/**< a bluetooth class */
#define SPA_KEY_API_BLUEZ5_ICON		"api.bluez5.icon"		/**< a bluetooth icon */
#define SPA_KEY_API_BLUEZ5_ROLE		"api.bluez5.role"		/**< "client" or "server" */

/** keys for jack api */
#define SPA_KEY_API_JACK		"api.jack"			/**< key for the JACK api */
#define SPA_KEY_API_JACK_SERVER		"api.jack.server"		/**< a jack server name */
#define SPA_KEY_API_JACK_CLIENT		"api.jack.client"		/**< an internal jack client */

/** keys for glib api */
#define SPA_KEY_API_GLIB_MAINLOOP	"api.glib.mainloop"		/**< whether glib mainloop runs
									 * in same thread as PW loop */

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_UTILS_KEYS_H */
