/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PLUGIN_H
#define SPA_PLUGIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/utils/dict.h>

/**
 * \defgroup spa_handle Plugin Handle
 * SPA plugin handle and factory interfaces
 */

/**
 * \addtogroup spa_handle
 * \{
 */

struct spa_handle {
	/** Version of this struct */
#define SPA_VERSION_HANDLE	0
	uint32_t version;

	/**
	 * Get the interface provided by \a handle with \a type.
	 *
	 * \a interface is always a struct spa_interface but depending on
	 * \a type, the struct might contain other information.
	 *
	 * \param handle a spa_handle
	 * \param type the interface type
	 * \param interface result to hold the interface.
	 * \return 0 on success
	 *         -ENOTSUP when there are no interfaces
	 *         -EINVAL when handle or info is NULL
	 */
	int (*get_interface) (struct spa_handle *handle, const char *type, void **interface);
	/**
	 * Clean up the memory of \a handle. After this, \a handle should not be used
	 * anymore.
	 *
	 * \param handle a pointer to memory
	 * \return 0 on success
	 */
	int (*clear) (struct spa_handle *handle);
};

#define spa_handle_get_interface(h,...)	(h)->get_interface((h),__VA_ARGS__)
#define spa_handle_clear(h)		(h)->clear((h))

/**
 * This structure lists the information about available interfaces on
 * handles.
 */
struct spa_interface_info {
	const char *type;	/*< the type of the interface, can be
				 *  used to get the interface */
};

/**
 * Extra supporting infrastructure passed to the init() function of
 * a factory. It can be extra information or interfaces such as logging.
 */
struct spa_support {
	const char *type;	/*< the type of the support item */
	void *data;		/*< specific data for the item */
};

/** Find a support item of the given type */
static inline void *spa_support_find(const struct spa_support *support,
				     uint32_t n_support,
				     const char *type)
{
	uint32_t i;
	for (i = 0; i < n_support; i++) {
		if (strcmp(support[i].type, type) == 0)
			return support[i].data;
	}
	return NULL;
}

#define SPA_SUPPORT_INIT(type,data) ((struct spa_support) { (type), (data) })

struct spa_handle_factory {
	/** The version of this structure */
#define SPA_VERSION_HANDLE_FACTORY	1
	uint32_t version;
	/**
	 * The name of the factory contains a logical name that describes
	 * the function of the handle. Other plugins might contain an alternative
	 * implementation with the same name.
	 *
	 * See utils/names.h for the list of standard names.
	 *
	 * Examples include:
	 *
	 *  api.alsa.pcm.sink: an object to write PCM samples to an alsa PLAYBACK
	 *			device
	 *  api.v4l2.source: an object to read from a v4l2 source.
	 */
	const char *name;
	/**
	 * Extra information about the handles of this factory.
	 */
	const struct spa_dict *info;
	/**
	 * Get the size of handles from this factory.
	 *
	 * \param factory a spa_handle_factory
	 * \param params extra parameters that determine the size of the
	 * handle.
	 */
	size_t (*get_size) (const struct spa_handle_factory *factory,
			    const struct spa_dict *params);

	/**
	 * Initialize an instance of this factory. The caller should allocate
	 * memory at least size bytes and pass this as \a handle.
	 *
	 * \a support can optionally contain extra interfaces or data items that the
	 * plugin can use such as a logger.
	 *
	 * \param factory a spa_handle_factory
	 * \param handle a pointer to memory
	 * \param info extra handle specific information, usually obtained
	 *        from a spa_device. This can be used to configure the handle.
	 * \param support support items
	 * \param n_support number of elements in \a support
	 * \return 0 on success
	 *	   < 0 errno type error
	 */
	int (*init) (const struct spa_handle_factory *factory,
		     struct spa_handle *handle,
		     const struct spa_dict *info,
		     const struct spa_support *support,
		     uint32_t n_support);

	/**
	 * spa_handle_factory::enum_interface_info:
	 * \param factory: a #spa_handle_factory
	 * \param info: result to hold spa_interface_info.
	 * \param index: index to keep track of the enumeration, 0 for first item
	 *
	 * Enumerate the interface information for \a factory.
	 *
	 * \return 1 when an item is available
	 *	   0 when no more items are available
	 *	   < 0 errno type error
	 */
	int (*enum_interface_info) (const struct spa_handle_factory *factory,
				    const struct spa_interface_info **info,
				    uint32_t *index);
};

#define spa_handle_factory_get_size(h,...)		(h)->get_size((h),__VA_ARGS__)
#define spa_handle_factory_init(h,...)			(h)->init((h),__VA_ARGS__)
#define spa_handle_factory_enum_interface_info(h,...)	(h)->enum_interface_info((h),__VA_ARGS__)

/**
 * The function signature of the entry point in a plugin.
 *
 * \param factory a location to hold the factory result
 * \param index index to keep track of the enumeration
 * \return 1 on success
 *         0 when there are no more factories
 *         -EINVAL when factory is NULL
 */
typedef int (*spa_handle_factory_enum_func_t) (const struct spa_handle_factory **factory,
					       uint32_t *index);

#define SPA_HANDLE_FACTORY_ENUM_FUNC_NAME "spa_handle_factory_enum"

/**
 * The entry point in a plugin.
 *
 * \param factory a location to hold the factory result
 * \param index index to keep track of the enumeration
 * \return 1 on success
 *	   0 when no more items are available
 *	   < 0 errno type error
 */
int spa_handle_factory_enum(const struct spa_handle_factory **factory, uint32_t *index);



#define SPA_KEY_FACTORY_NAME		"factory.name"		/**< the name of a factory */
#define SPA_KEY_FACTORY_AUTHOR		"factory.author"	/**< a comma separated list of factory authors */
#define SPA_KEY_FACTORY_DESCRIPTION	"factory.description"	/**< description of a factory */
#define SPA_KEY_FACTORY_USAGE		"factory.usage"		/**< usage of a factory */

#define SPA_KEY_LIBRARY_NAME		"library.name"		/**< the name of a library. This is usually
								  *  the filename of the plugin without the
								  *  path or the plugin extension. */

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PLUGIN_H */
