/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2020 Collabora Ltd. */
/*                         @author George Kiagiadakis <george.kiagiadakis@collabora.com> */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_EXT_SESSION_MANAGER_INTROSPECT_FUNCS_H
#define PIPEWIRE_EXT_SESSION_MANAGER_INTROSPECT_FUNCS_H

#include "introspect.h"
#include <spa/pod/builder.h>
#include <pipewire/pipewire.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup pw_session_manager
 * \{
 */

static inline struct pw_session_info *
pw_session_info_update (struct pw_session_info *info,
			const struct pw_session_info *update)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_session_info info;
	} *ext;

	if (update == NULL)
		return info;

	if (info == NULL) {
		ext = (struct extended_info *) calloc(1, sizeof(*ext));
		if (ext == NULL)
			return NULL;

		info = &ext->info;
		info->id = update->id;
	} else {
		ext = SPA_CONTAINER_OF(info, struct extended_info, info);
	}

	info->change_mask = update->change_mask;

	if (update->change_mask & PW_SESSION_CHANGE_MASK_PROPS) {
		if (!ext->props_storage) {
			ext->props_storage = pw_properties_new(NULL, NULL);
			info->props = &ext->props_storage->dict;
		}
		pw_properties_clear(ext->props_storage);
		pw_properties_update(ext->props_storage, update->props);
	}
	if (update->change_mask & PW_SESSION_CHANGE_MASK_PARAMS) {
		info->n_params = update->n_params;
		free((void *) info->params);
		if (update->params) {
			size_t size = info->n_params * sizeof(struct spa_param_info);
			info->params = (struct spa_param_info *) malloc(size);
			memcpy(info->params, update->params, size);
		}
		else
			info->params = NULL;
	}
	return info;
}

static inline void
pw_session_info_free (struct pw_session_info *info)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_session_info info;
	} *ext = SPA_CONTAINER_OF(info, struct extended_info, info);

	pw_properties_free(ext->props_storage);
	free((void *) info->params);
	free(ext);
}

static inline struct pw_endpoint_info *
pw_endpoint_info_update (struct pw_endpoint_info *info,
			const struct pw_endpoint_info *update)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_endpoint_info info;
	} *ext;

	if (update == NULL)
		return info;

	if (info == NULL) {
		ext = (struct extended_info *) calloc(1, sizeof(*ext));
		if (ext == NULL)
			return NULL;

		info = &ext->info;
		info->id = update->id;
		info->name = strdup(update->name);
		info->media_class = strdup(update->media_class);
		info->direction = update->direction;
		info->flags = update->flags;
	} else {
		ext = SPA_CONTAINER_OF(info, struct extended_info, info);
	}

	info->change_mask = update->change_mask;

	if (update->change_mask & PW_ENDPOINT_CHANGE_MASK_STREAMS)
		info->n_streams = update->n_streams;

	if (update->change_mask & PW_ENDPOINT_CHANGE_MASK_SESSION)
		info->session_id = update->session_id;

	if (update->change_mask & PW_ENDPOINT_CHANGE_MASK_PROPS) {
		if (!ext->props_storage) {
			ext->props_storage = pw_properties_new(NULL, NULL);
			info->props = &ext->props_storage->dict;
		}
		pw_properties_clear(ext->props_storage);
		pw_properties_update(ext->props_storage, update->props);
	}
	if (update->change_mask & PW_ENDPOINT_CHANGE_MASK_PARAMS) {
		info->n_params = update->n_params;
		free((void *) info->params);
		if (update->params) {
			size_t size = info->n_params * sizeof(struct spa_param_info);
			info->params = (struct spa_param_info *) malloc(size);
			memcpy(info->params, update->params, size);
		}
		else
			info->params = NULL;
	}
	return info;
}

static inline void
pw_endpoint_info_free (struct pw_endpoint_info *info)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_endpoint_info info;
	} *ext = SPA_CONTAINER_OF(info, struct extended_info, info);

	pw_properties_free(ext->props_storage);
	free(info->name);
	free(info->media_class);
	free((void *) info->params);
	free(ext);
}

static inline struct pw_endpoint_stream_info *
pw_endpoint_stream_info_update (struct pw_endpoint_stream_info *info,
				const struct pw_endpoint_stream_info *update)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_endpoint_stream_info info;
	} *ext;

	if (update == NULL)
		return info;

	if (info == NULL) {
		ext = (struct extended_info *) calloc(1, sizeof(*ext));
		if (ext == NULL)
			return NULL;

		info = &ext->info;
		info->id = update->id;
		info->endpoint_id = update->endpoint_id;
		info->name = strdup(update->name);
	} else {
		ext = SPA_CONTAINER_OF(info, struct extended_info, info);
	}

	info->change_mask = update->change_mask;

	if (update->change_mask & PW_ENDPOINT_STREAM_CHANGE_MASK_LINK_PARAMS) {
		free(info->link_params);
		info->link_params = update->link_params ?
			spa_pod_copy(update->link_params) : NULL;
	}
	if (update->change_mask & PW_ENDPOINT_STREAM_CHANGE_MASK_PROPS) {
		if (!ext->props_storage) {
			ext->props_storage = pw_properties_new(NULL, NULL);
			info->props = &ext->props_storage->dict;
		}
		pw_properties_clear(ext->props_storage);
		pw_properties_update(ext->props_storage, update->props);
	}
	if (update->change_mask & PW_ENDPOINT_STREAM_CHANGE_MASK_PARAMS) {
		info->n_params = update->n_params;
		free((void *) info->params);
		if (update->params) {
			size_t size = info->n_params * sizeof(struct spa_param_info);
			info->params = (struct spa_param_info *) malloc(size);
			memcpy(info->params, update->params, size);
		}
		else
			info->params = NULL;
	}
	return info;
}

static inline void
pw_endpoint_stream_info_free (struct pw_endpoint_stream_info *info)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_endpoint_stream_info info;
	} *ext = SPA_CONTAINER_OF(info, struct extended_info, info);

	pw_properties_free(ext->props_storage);
	free(info->name);
	free(info->link_params);
	free((void *) info->params);
	free(ext);
}


static inline struct pw_endpoint_link_info *
pw_endpoint_link_info_update (struct pw_endpoint_link_info *info,
			const struct pw_endpoint_link_info *update)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_endpoint_link_info info;
	} *ext;

	if (update == NULL)
		return info;

	if (info == NULL) {
		ext = (struct extended_info *) calloc(1, sizeof(*ext));
		if (ext == NULL)
			return NULL;

		info = &ext->info;
		info->id = update->id;
		info->session_id = update->session_id;
		info->output_endpoint_id = update->output_endpoint_id;
		info->output_stream_id = update->output_stream_id;
		info->input_endpoint_id = update->input_endpoint_id;
		info->input_stream_id = update->input_stream_id;
	} else {
		ext = SPA_CONTAINER_OF(info, struct extended_info, info);
	}

	info->change_mask = update->change_mask;

	if (update->change_mask & PW_ENDPOINT_LINK_CHANGE_MASK_STATE) {
		info->state = update->state;
		free(info->error);
		info->error = update->error ? strdup(update->error) : NULL;
	}
	if (update->change_mask & PW_ENDPOINT_LINK_CHANGE_MASK_PROPS) {
		if (!ext->props_storage) {
			ext->props_storage = pw_properties_new(NULL, NULL);
			info->props = &ext->props_storage->dict;
		}
		pw_properties_clear(ext->props_storage);
		pw_properties_update(ext->props_storage, update->props);
	}
	if (update->change_mask & PW_ENDPOINT_LINK_CHANGE_MASK_PARAMS) {
		info->n_params = update->n_params;
		free((void *) info->params);
		if (update->params) {
			size_t size = info->n_params * sizeof(struct spa_param_info);
			info->params = (struct spa_param_info *) malloc(size);
			memcpy(info->params, update->params, size);
		}
		else
			info->params = NULL;
	}
	return info;
}

static inline void
pw_endpoint_link_info_free (struct pw_endpoint_link_info *info)
{
	struct extended_info {
		struct pw_properties *props_storage;
		struct pw_endpoint_link_info info;
	} *ext = SPA_CONTAINER_OF(info, struct extended_info, info);

	pw_properties_free(ext->props_storage);
	free(info->error);
	free((void *) info->params);
	free(ext);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_EXT_SESSION_MANAGER_INTROSPECT_FUNCS_H */
