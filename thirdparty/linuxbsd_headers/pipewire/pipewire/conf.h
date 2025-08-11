/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_CONF_H
#define PIPEWIRE_CONF_H

#include <pipewire/context.h>

/** \defgroup pw_conf Configuration
 * Loading/saving properties from/to configuration files.
 */

/**
 * \addtogroup pw_conf
 * \{
 */

int pw_conf_load_conf_for_context(struct pw_properties *props, struct pw_properties *conf);
int pw_conf_load_conf(const char *prefix, const char *name, struct pw_properties *conf);
int pw_conf_load_state(const char *prefix, const char *name, struct pw_properties *conf);
int pw_conf_save_state(const char *prefix, const char *name, const struct pw_properties *conf);

int pw_conf_section_update_props(const struct spa_dict *conf,
		const char *section, struct pw_properties *props);

int pw_conf_section_for_each(const struct spa_dict *conf, const char *section,
		int (*callback) (void *data, const char *location, const char *section,
			const char *str, size_t len),
		void *data);

int pw_conf_match_rules(const char *str, size_t len, const char *location,
		const struct spa_dict *props,
		int (*callback) (void *data, const char *location, const char *action,
			const char *str, size_t len),
		void *data);

int pw_conf_section_match_rules(const struct spa_dict *conf, const char *section,
		const struct spa_dict *props,
		int (*callback) (void *data, const char *location, const char *action,
			const char *str, size_t len),
		void *data);
/**
 * \}
 */

#endif /* PIPEWIRE_CONF_H */
