/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_EXT_METADATA_H
#define PIPEWIRE_EXT_METADATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>

/** \defgroup pw_metadata Metadata
 * Metadata interface
 */

/**
 * \addtogroup pw_metadata
 * \{
 */
#define PW_TYPE_INTERFACE_Metadata		PW_TYPE_INFO_INTERFACE_BASE "Metadata"

#define PW_METADATA_PERM_MASK			PW_PERM_RWX

#define PW_VERSION_METADATA			3
struct pw_metadata;

#define PW_EXTENSION_MODULE_METADATA		PIPEWIRE_MODULE_PREFIX "module-metadata"

#define PW_METADATA_EVENT_PROPERTY		0
#define PW_METADATA_EVENT_NUM			1


/** \ref pw_metadata events */
struct pw_metadata_events {
#define PW_VERSION_METADATA_EVENTS		0
	uint32_t version;

	int (*property) (void *data,
			uint32_t subject,
			const char *key,
			const char *type,
			const char *value);
};

#define PW_METADATA_METHOD_ADD_LISTENER		0
#define PW_METADATA_METHOD_SET_PROPERTY		1
#define PW_METADATA_METHOD_CLEAR		2
#define PW_METADATA_METHOD_NUM			3

/** \ref pw_metadata methods */
struct pw_metadata_methods {
#define PW_VERSION_METADATA_METHODS		0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_metadata_events *events,
			void *data);

	/**
	 * Set a metadata property
	 *
	 * Automatically emit property events for the subject and key
	 * when they are changed.
	 *
	 * \param subject the id of the global to associate the metadata
	 *                with.
	 * \param key the key of the metadata, NULL clears all metadata for
	 *                the subject.
	 * \param type the type of the metadata, this can be blank
	 * \param value the metadata value. NULL clears the metadata.
	 *
	 * This requires X and W permissions on the metadata. It also
	 * requires M permissions on the subject global.
	 */
	int (*set_property) (void *object,
			uint32_t subject,
			const char *key,
			const char *type,
			const char *value);

	/**
	 * Clear all metadata
	 *
	 * This requires X and W permissions on the metadata.
	 */
	int (*clear) (void *object);
};


#define pw_metadata_method(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_metadata_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_metadata_add_listener(c,...)		pw_metadata_method(c,add_listener,0,__VA_ARGS__)
#define pw_metadata_set_property(c,...)		pw_metadata_method(c,set_property,0,__VA_ARGS__)
#define pw_metadata_clear(c)			pw_metadata_method(c,clear,0)

#define PW_KEY_METADATA_NAME		"metadata.name"
#define PW_KEY_METADATA_VALUES		"metadata.values"

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_EXT_METADATA_H */
