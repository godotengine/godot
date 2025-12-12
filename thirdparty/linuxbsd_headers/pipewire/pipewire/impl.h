/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_IMPL_H
#define PIPEWIRE_IMPL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup api_pw_impl
 */

struct pw_impl_client;
struct pw_impl_module;
struct pw_global;
struct pw_node;
struct pw_impl_port;
struct pw_resource;

#include <pipewire/pipewire.h>
#include <pipewire/control.h>
#include <pipewire/impl-core.h>
#include <pipewire/impl-client.h>
#include <pipewire/impl-device.h>
#include <pipewire/impl-factory.h>
#include <pipewire/global.h>
#include <pipewire/impl-link.h>
#include <pipewire/impl-metadata.h>
#include <pipewire/impl-module.h>
#include <pipewire/impl-node.h>
#include <pipewire/impl-port.h>
#include <pipewire/resource.h>
#include <pipewire/work-queue.h>

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_IMPL_H */
