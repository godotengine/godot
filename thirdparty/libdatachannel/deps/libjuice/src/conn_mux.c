/**
 * Copyright (c) 2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "conn_mux.h"
#include "agent.h"
#include "log.h"
#include "socket.h"
#include "stun.h"
#include "thread.h"
#include "udp.h"

#include <assert.h>
#include <string.h>

#define BUFFER_SIZE 4096
#define INITIAL_MAP_SIZE 16

typedef enum map_entry_type {
	MAP_ENTRY_TYPE_EMPTY = 0,
	MAP_ENTRY_TYPE_DELETED,
	MAP_ENTRY_TYPE_FULL
} map_entry_type_t;

typedef struct map_entry {
	map_entry_type_t type;
	juice_agent_t *agent;
	addr_record_t record;
} map_entry_t;

typedef struct registry_impl {
	thread_t thread;
	socket_t sock;
	mutex_t send_mutex;
	int send_ds;
	map_entry_t *map;
	int map_size;
	int map_count;
} registry_impl_t;

typedef struct conn_impl {
	conn_registry_t *registry;
	timestamp_t next_timestamp;
	bool finished;
} conn_impl_t;

static bool is_ready(const juice_agent_t *agent) {
	if (!agent)
		return false;

	conn_impl_t *conn_impl = agent->conn_impl;
	if (!conn_impl || conn_impl->finished)
		return false;

	return true;
}

static map_entry_t *find_map_entry(registry_impl_t *impl, const addr_record_t *record,
                                   bool allow_deleted);
static int insert_map_entry(registry_impl_t *impl, const addr_record_t *record,
                            juice_agent_t *agent);
static int remove_map_entries(registry_impl_t *impl, juice_agent_t *agent);
static int grow_map(registry_impl_t *impl, int new_size);

static map_entry_t *find_map_entry(registry_impl_t *impl, const addr_record_t *record,
                                   bool allow_deleted) {
	unsigned long key = addr_record_hash(record, false) % impl->map_size;
	unsigned long pos = key;
	while (true) {
		map_entry_t *entry = impl->map + pos;
		if (entry->type == MAP_ENTRY_TYPE_EMPTY ||
		    addr_record_is_equal(&entry->record, record, true)) // compare ports
			break;

		if (entry->type == MAP_ENTRY_TYPE_DELETED && allow_deleted)
			break;

		pos = (pos + 1) % impl->map_size;
		if (pos == key)
			return NULL;
	}
	return impl->map + pos;
}

static int insert_map_entry(registry_impl_t *impl, const addr_record_t *record,
                            juice_agent_t *agent) {

	map_entry_t *entry = find_map_entry(impl, record, true); // allow deleted
	if (!entry || (entry->type != MAP_ENTRY_TYPE_FULL && impl->map_count * 2 >= impl->map_size)) {
		grow_map(impl, impl->map_size * 2);
		return insert_map_entry(impl, record, agent);
	}

	if (entry->type != MAP_ENTRY_TYPE_FULL)
		++impl->map_count;

	entry->type = MAP_ENTRY_TYPE_FULL;
	entry->agent = agent;
	entry->record = *record;

	JLOG_VERBOSE("Added map entry, count=%d", impl->map_count);
	return 0;
}

static int remove_map_entries(registry_impl_t *impl, juice_agent_t *agent) {
	int count = 0;
	for (int i = 0; i < impl->map_size; ++i) {
		map_entry_t *entry = impl->map + i;
		if (entry->type == MAP_ENTRY_TYPE_FULL && entry->agent == agent) {
			entry->type = MAP_ENTRY_TYPE_DELETED;
			entry->agent = NULL;
			++count;
		}
	}

	assert(impl->map_count >= count);
	impl->map_count -= count;

	JLOG_VERBOSE("Removed %d map entries, count=%d", count, impl->map_count);
	return 0;
}

static int grow_map(registry_impl_t *impl, int new_size) {
	if (new_size <= impl->map_size)
		return 0;

	JLOG_DEBUG("Growing map, new_size=%d", new_size);

	map_entry_t *new_map = calloc(1, new_size * sizeof(map_entry_t));
	if (!new_map) {
		JLOG_FATAL("Memory allocation failed for map");
		return -1;
	}

	map_entry_t *old_map = impl->map;
	int old_size = impl->map_size;
	impl->map = new_map;
	impl->map_size = new_size;
	impl->map_count = 0;

	for (int i = 0; i < old_size; ++i) {
		map_entry_t *old_entry = old_map + i;
		if (old_entry->type == MAP_ENTRY_TYPE_FULL)
			insert_map_entry(impl, &old_entry->record, old_entry->agent);
	}

	free(old_map);
	return 0;
}

int conn_mux_prepare(conn_registry_t *registry, struct pollfd *pfd, timestamp_t *next_timestamp);
int conn_mux_process(conn_registry_t *registry, struct pollfd *pfd);
int conn_mux_recv(conn_registry_t *registry, char *buffer, size_t size, addr_record_t *src);
void conn_mux_fail(conn_registry_t *registry);
int conn_mux_run(conn_registry_t *registry);

static thread_return_t THREAD_CALL conn_mux_thread_entry(void *arg) {
	thread_set_name_self("juice mux");
	conn_registry_t *registry = (conn_registry_t *)arg;
	conn_mux_run(registry);
	return (thread_return_t)0;
}

int conn_mux_registry_init(conn_registry_t *registry, udp_socket_config_t *config) {
	(void)config;
	registry_impl_t *registry_impl = calloc(1, sizeof(registry_impl_t));
	if (!registry_impl) {
		JLOG_FATAL("Memory allocation failed for connections registry impl");
		return -1;
	}

	registry_impl->map = calloc(INITIAL_MAP_SIZE, sizeof(map_entry_t));
	if (!registry_impl->map) {
		JLOG_FATAL("Memory allocation failed for map");
		free(registry_impl);
		return -1;
	}
	registry_impl->map_size = INITIAL_MAP_SIZE;
	registry_impl->map_count = 0;

	registry_impl->sock = udp_create_socket(config);
	if (registry_impl->sock == INVALID_SOCKET) {
		JLOG_FATAL("UDP socket creation failed");
		free(registry_impl->map);
		free(registry_impl);
		return -1;
	}

	mutex_init(&registry_impl->send_mutex, 0);
	registry->impl = registry_impl;

	JLOG_DEBUG("Starting connections thread");
	int ret = thread_init(&registry_impl->thread, conn_mux_thread_entry, registry);
	if (ret) {
		JLOG_FATAL("Thread creation failed, error=%d", ret);
		goto error;
	}

	return 0;

error:
	mutex_destroy(&registry_impl->send_mutex);
	closesocket(registry_impl->sock);
	free(registry_impl->map);
	free(registry_impl);
	registry->impl = NULL;
	return -1;
}

void conn_mux_registry_cleanup(conn_registry_t *registry) {
	registry_impl_t *registry_impl = registry->impl;

	JLOG_VERBOSE("Waiting for connections thread");
	thread_join(registry_impl->thread, NULL);

	mutex_destroy(&registry_impl->send_mutex);
	closesocket(registry_impl->sock);
	free(registry_impl->map);
	free(registry->impl);
	registry->impl = NULL;
}

int conn_mux_prepare(conn_registry_t *registry, struct pollfd *pfd, timestamp_t *next_timestamp) {
	timestamp_t now = current_timestamp();
	*next_timestamp = now + 60000;

	mutex_lock(&registry->mutex);
	registry_impl_t *registry_impl = registry->impl;
	pfd->fd = registry_impl->sock;
	pfd->events = POLLIN;

	for (int i = 0; i < registry->agents_size; ++i) {
		juice_agent_t *agent = registry->agents[i];
		if (is_ready(agent)) {
			conn_impl_t *conn_impl = agent->conn_impl;
			if (*next_timestamp > conn_impl->next_timestamp)
				*next_timestamp = conn_impl->next_timestamp;
		}
	}

	int count = registry->agents_count;
	mutex_unlock(&registry->mutex);
	return count;
}

static juice_agent_t *lookup_agent(conn_registry_t *registry, char *buf, size_t len,
                                   const addr_record_t *src) {
	JLOG_VERBOSE("Looking up agent from address");

	registry_impl_t *registry_impl = registry->impl;
	map_entry_t *entry = find_map_entry(registry_impl, src, false);
	juice_agent_t *agent = entry && entry->type == MAP_ENTRY_TYPE_FULL ? entry->agent : NULL;
	if (agent) {
		JLOG_DEBUG("Found agent from address");
		return agent;
	}

	if (!is_stun_datagram(buf, len)) {
		JLOG_INFO("Got non-STUN message from unknown source address");
		return NULL;
	}

	JLOG_VERBOSE("Looking up agent from STUN message content");

	stun_message_t msg;
	if (stun_read(buf, len, &msg) < 0) {
		JLOG_ERROR("STUN message reading failed");
		return NULL;
	}

	if (msg.msg_class == STUN_CLASS_REQUEST && msg.msg_method == STUN_METHOD_BINDING &&
	    msg.has_integrity) {
		// Binding request from peer
		char username[STUN_MAX_USERNAME_LEN];
		strcpy(username, msg.credentials.username);
		char *separator = strchr(username, ':');
		if (!separator) {
			JLOG_WARN("STUN username invalid, username=\"%s\"", username);
			return NULL;
		}
		*separator = '\0';
		const char *local_ufrag = username;
		for (int i = 0; i < registry->agents_size; ++i) {
			agent = registry->agents[i];
			if (is_ready(agent)) {
				if (strcmp(local_ufrag, agent->local.ice_ufrag) == 0) {
					JLOG_DEBUG("Found agent from ICE ufrag");
					insert_map_entry(registry_impl, src, agent);
					return agent;
				}
			}
		}

	} else {
		if (!STUN_IS_RESPONSE(msg.msg_class)) {
			JLOG_INFO("Got unexpected STUN message from unknown source address");
			return NULL;
		}

		for (int i = 0; i < registry->agents_size; ++i) {
			agent = registry->agents[i];
			if (is_ready(agent)) {
				if (agent_find_entry_from_transaction_id(agent, msg.transaction_id)) {
					JLOG_DEBUG("Found agent from transaction ID");
					return agent;
				}
			}
		}
	}

	return NULL;
}

int conn_mux_process(conn_registry_t *registry, struct pollfd *pfd) {
	mutex_lock(&registry->mutex);

	if (pfd->revents & POLLNVAL || pfd->revents & POLLERR) {
		JLOG_ERROR("Error when polling socket");
		conn_mux_fail(registry);
		mutex_unlock(&registry->mutex);
		return -1;
	}

	if (pfd->revents & POLLIN) {
		char buffer[BUFFER_SIZE];
		addr_record_t src;
		int ret;
		while ((ret = conn_mux_recv(registry, buffer, BUFFER_SIZE, &src)) > 0) {
			if (JLOG_DEBUG_ENABLED) {
				char src_str[ADDR_MAX_STRING_LEN];
				addr_record_to_string(&src, src_str, ADDR_MAX_STRING_LEN);
				JLOG_DEBUG("Demultiplexing incoming datagram from %s", src_str);
			}

			juice_agent_t *agent = lookup_agent(registry, buffer, (size_t)ret, &src);
			if (!agent || !is_ready(agent)) {
				JLOG_DEBUG("Agent not found for incoming datagram, dropping");
				continue;
			}

			conn_impl_t *conn_impl = agent->conn_impl;
			if (agent_conn_recv(agent, buffer, (size_t)ret, &src) != 0) {
				JLOG_WARN("Agent receive failed");
				conn_impl->finished = true;
				continue;
			}

			conn_impl->next_timestamp = current_timestamp();
		}

		if (ret < 0) {
			conn_mux_fail(registry);
			mutex_unlock(&registry->mutex);
			return -1;
		}
	}

	for (int i = 0; i < registry->agents_size; ++i) {
		juice_agent_t *agent = registry->agents[i];
		if (is_ready(agent)) {
			conn_impl_t *conn_impl = agent->conn_impl;
			if (conn_impl->next_timestamp <= current_timestamp()) {
				if (agent_conn_update(agent, &conn_impl->next_timestamp) != 0) {
					JLOG_WARN("Agent update failed");
					conn_impl->finished = true;
					continue;
				}
			}
		}
	}

	mutex_unlock(&registry->mutex);
	return 0;
}

int conn_mux_recv(conn_registry_t *registry, char *buffer, size_t size, addr_record_t *src) {
	JLOG_VERBOSE("Receiving datagram");
	registry_impl_t *registry_impl = registry->impl;
	int len;
	while ((len = udp_recvfrom(registry_impl->sock, buffer, size, src)) == 0) {
		// Empty datagram (used to interrupt)
	}

	if (len < 0) {
		if (sockerrno == SEAGAIN || sockerrno == SEWOULDBLOCK) {
			JLOG_VERBOSE("No more datagrams to receive");
			return 0;
		}
		JLOG_ERROR("recvfrom failed, errno=%d", sockerrno);
		return -1;
	}

	addr_unmap_inet6_v4mapped((struct sockaddr *)&src->addr, &src->len);
	return len; // len > 0
}

void conn_mux_fail(conn_registry_t *registry) {
	for (int i = 0; i < registry->agents_size; ++i) {
		juice_agent_t *agent = registry->agents[i];
		if (is_ready(agent)) {
			conn_impl_t *conn_impl = agent->conn_impl;
			agent_conn_fail(agent);
			conn_impl->finished = true;
		}
	}
}

int conn_mux_run(conn_registry_t *registry) {
	struct pollfd pfd[1];
	timestamp_t next_timestamp;
	while (conn_mux_prepare(registry, pfd, &next_timestamp) > 0) {
		timediff_t timediff = next_timestamp - current_timestamp();
		if (timediff < 0)
			timediff = 0;

		JLOG_VERBOSE("Entering poll for %d ms", (int)timediff);
		int ret = poll(pfd, 1, (int)timediff);
		JLOG_VERBOSE("Leaving poll");
		if (ret < 0) {
			if (sockerrno == SEINTR || sockerrno == SEAGAIN) {
				JLOG_VERBOSE("poll interrupted");
				continue;
			} else {
				JLOG_FATAL("poll failed, errno=%d", sockerrno);
				break;
			}
		}

		if (conn_mux_process(registry, pfd) < 0)
			break;
	}

	JLOG_DEBUG("Leaving connections thread");
	return 0;
}

int conn_mux_init(juice_agent_t *agent, conn_registry_t *registry, udp_socket_config_t *config) {
	(void)config; // ignored, only the config from the first connection is used

	conn_impl_t *conn_impl = calloc(1, sizeof(conn_impl_t));
	if (!conn_impl) {
		JLOG_FATAL("Memory allocation failed for connection impl");
		return -1;
	}

	conn_impl->registry = registry;
	agent->conn_impl = conn_impl;
	return 0;
}

void conn_mux_cleanup(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;
	conn_registry_t *registry = conn_impl->registry;

	mutex_lock(&registry->mutex);
	registry_impl_t *registry_impl = registry->impl;
	remove_map_entries(registry_impl, agent);
	mutex_unlock(&registry->mutex);

	conn_mux_interrupt(agent);

	free(agent->conn_impl);
	agent->conn_impl = NULL;
}

void conn_mux_lock(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;
	conn_registry_t *registry = conn_impl->registry;
	mutex_lock(&registry->mutex);
}

void conn_mux_unlock(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;
	conn_registry_t *registry = conn_impl->registry;
	mutex_unlock(&registry->mutex);
}

int conn_mux_interrupt(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;
	conn_registry_t *registry = conn_impl->registry;

	mutex_lock(&registry->mutex);
	conn_impl->next_timestamp = current_timestamp();
	mutex_unlock(&registry->mutex);

	JLOG_VERBOSE("Interrupting connections thread");

	registry_impl_t *registry_impl = registry->impl;
	mutex_lock(&registry_impl->send_mutex);
	if (udp_sendto_self(registry_impl->sock, NULL, 0) < 0) {
		if (sockerrno != SEAGAIN && sockerrno != SEWOULDBLOCK) {
			JLOG_WARN("Failed to interrupt poll by triggering socket, errno=%d", sockerrno);
		}
		mutex_unlock(&registry_impl->send_mutex);
		return -1;
	}
	mutex_unlock(&registry_impl->send_mutex);
	return 0;
}

int conn_mux_send(juice_agent_t *agent, const addr_record_t *dst, const char *data, size_t size,
                  int ds) {
	conn_impl_t *conn_impl = agent->conn_impl;
	registry_impl_t *registry_impl = conn_impl->registry->impl;

	mutex_lock(&registry_impl->send_mutex);

	if (registry_impl->send_ds >= 0 && registry_impl->send_ds != ds) {
		JLOG_VERBOSE("Setting Differentiated Services field to 0x%X", ds);
		if (udp_set_diffserv(registry_impl->sock, ds) == 0)
			registry_impl->send_ds = ds;
		else
			registry_impl->send_ds = -1; // disable for next time
	}

	JLOG_VERBOSE("Sending datagram, size=%d", size);

	int ret = udp_sendto(registry_impl->sock, data, size, dst);
	if (ret < 0) {
		if (sockerrno == SEAGAIN || sockerrno == SEWOULDBLOCK)
			JLOG_INFO("Send failed, buffer is full");
		else if (sockerrno == SEMSGSIZE)
			JLOG_WARN("Send failed, datagram is too large");
		else
			JLOG_WARN("Send failed, errno=%d", sockerrno);
	}

	mutex_unlock(&registry_impl->send_mutex);
	return ret;
}

int conn_mux_get_addrs(juice_agent_t *agent, addr_record_t *records, size_t size) {
	conn_impl_t *conn_impl = agent->conn_impl;
	registry_impl_t *registry_impl = conn_impl->registry->impl;

	return udp_get_addrs(registry_impl->sock, records, size);
}
