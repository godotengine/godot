/**
 * Copyright (c) 2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_CONN_MUX_H
#define JUICE_CONN_MUX_H

#include "addr.h"
#include "conn.h"
#include "thread.h"
#include "timestamp.h"

#include <stdbool.h>
#include <stdint.h>

int conn_mux_registry_init(conn_registry_t *registry, udp_socket_config_t *config);
void conn_mux_registry_cleanup(conn_registry_t *registry);

int conn_mux_init(juice_agent_t *agent, conn_registry_t *registry, udp_socket_config_t *config);
void conn_mux_cleanup(juice_agent_t *agent);
void conn_mux_lock(juice_agent_t *agent);
void conn_mux_unlock(juice_agent_t *agent);
int conn_mux_interrupt(juice_agent_t *agent);
int conn_mux_send(juice_agent_t *agent, const addr_record_t *dst, const char *data, size_t size,
                        int ds);
int conn_mux_get_addrs(juice_agent_t *agent, addr_record_t *records, size_t size);

#endif
