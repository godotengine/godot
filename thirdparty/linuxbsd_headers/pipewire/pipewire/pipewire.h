/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_H
#define PIPEWIRE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/support/plugin.h>

#include <pipewire/array.h>
#include <pipewire/client.h>
#include <pipewire/conf.h>
#include <pipewire/context.h>
#include <pipewire/device.h>
#include <pipewire/buffers.h>
#include <pipewire/core.h>
#include <pipewire/factory.h>
#include <pipewire/keys.h>
#include <pipewire/log.h>
#include <pipewire/loop.h>
#include <pipewire/link.h>
#include <pipewire/main-loop.h>
#include <pipewire/map.h>
#include <pipewire/mem.h>
#include <pipewire/module.h>
#include <pipewire/node.h>
#include <pipewire/properties.h>
#include <pipewire/proxy.h>
#include <pipewire/permission.h>
#include <pipewire/protocol.h>
#include <pipewire/port.h>
#include <pipewire/stream.h>
#include <pipewire/filter.h>
#include <pipewire/thread-loop.h>
#include <pipewire/data-loop.h>
#include <pipewire/type.h>
#include <pipewire/utils.h>
#include <pipewire/version.h>

/** \defgroup pw_pipewire Initialization
 * Initializing PipeWire and loading SPA modules.
 */

/**
 * \addtogroup pw_pipewire
 * \{
 */
void
pw_init(int *argc, char **argv[]);

void pw_deinit(void);

bool
pw_debug_is_category_enabled(const char *name);

const char *
pw_get_application_name(void);

const char *
pw_get_prgname(void);

const char *
pw_get_user_name(void);

const char *
pw_get_host_name(void);

const char *
pw_get_client_name(void);

bool pw_check_option(const char *option, const char *value);

enum pw_direction
pw_direction_reverse(enum pw_direction direction);

int pw_set_domain(const char *domain);
const char *pw_get_domain(void);

uint32_t pw_get_support(struct spa_support *support, uint32_t max_support);

struct spa_handle *pw_load_spa_handle(const char *lib,
		const char *factory_name,
		const struct spa_dict *info,
		uint32_t n_support,
		const struct spa_support support[]);

int pw_unload_spa_handle(struct spa_handle *handle);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_H */
