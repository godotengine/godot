/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2023 PipeWire authors */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_CLEANUP_H
#define PIPEWIRE_CLEANUP_H

#include <spa/utils/cleanup.h>

#include <pipewire/properties.h>
#include <pipewire/utils.h>

SPA_DEFINE_AUTOPTR_CLEANUP(pw_properties, struct pw_properties, {
	spa_clear_ptr(*thing, pw_properties_free);
})

SPA_DEFINE_AUTO_CLEANUP(pw_strv, char **, {
	spa_clear_ptr(*thing, pw_free_strv);
})

#endif /* PIPEWIRE_CLEANUP_H */
