/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PLUGIN_LOADER_H
#define SPA_PLUGIN_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/hook.h>
#include <spa/utils/dict.h>

/** \defgroup spa_plugin_loader Plugin Loader
 * SPA plugin loader
 */

/**
 * \addtogroup spa_plugin_loader
 * \{
 */

#define SPA_TYPE_INTERFACE_PluginLoader	SPA_TYPE_INFO_INTERFACE_BASE "PluginLoader"

#define SPA_VERSION_PLUGIN_LOADER		0
struct spa_plugin_loader { struct spa_interface iface; };

struct spa_plugin_loader_methods {
#define SPA_VERSION_PLUGIN_LOADER_METHODS	0
        uint32_t version;

	/**
	 * Load a SPA plugin.
         *
         * \param factory_name Plugin factory name
         * \param info Info dictionary for plugin. NULL if none.
         * \return plugin handle, or NULL on error
	 */
	struct spa_handle *(*load) (void *object, const char *factory_name, const struct spa_dict *info);

	/**
	 * Unload a SPA plugin.
         *
         * \param handle Plugin handle.
         * \return 0 on success, < 0 on error
	 */
	int (*unload)(void *object, struct spa_handle *handle);
};

static inline struct spa_handle *
spa_plugin_loader_load(struct spa_plugin_loader *loader, const char *factory_name, const struct spa_dict *info)
{
	struct spa_handle *res = NULL;
	if (SPA_LIKELY(loader != NULL))
		spa_interface_call_res(&loader->iface,
				struct spa_plugin_loader_methods, res,
				load, 0, factory_name, info);
	return res;
}

static inline int
spa_plugin_loader_unload(struct spa_plugin_loader *loader, struct spa_handle *handle)
{
	int res = -1;
	if (SPA_LIKELY(loader != NULL))
		spa_interface_call_res(&loader->iface,
				struct spa_plugin_loader_methods, res,
				unload, 0, handle);
	return res;
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PLUGIN_LOADER_H */
