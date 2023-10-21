/**
 * Copyright (c) 2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "conn_thread.h"
#include "agent.h"
#include "log.h"
#include "socket.h"
#include "thread.h"
#include "udp.h"

#include <assert.h>
#include <string.h>

#define BUFFER_SIZE 4096

typedef struct conn_impl {
	thread_t thread;
	socket_t sock;
	mutex_t mutex;
	mutex_t send_mutex;
	int send_ds;
	timestamp_t next_timestamp;
	bool stopped;
} conn_impl_t;

int conn_thread_run(juice_agent_t *agent);
int conn_thread_prepare(juice_agent_t *agent, struct pollfd *pfd, timestamp_t *next_timestamp);
int conn_thread_process(juice_agent_t *agent, struct pollfd *pfd);
int conn_thread_recv(socket_t sock, char *buffer, size_t size, addr_record_t *src);

static thread_return_t THREAD_CALL conn_thread_entry(void *arg) {
	thread_set_name_self("juice agent");
	juice_agent_t *agent = (juice_agent_t *)arg;
	conn_thread_run(agent);
	return (thread_return_t)0;
}

int conn_thread_prepare(juice_agent_t *agent, struct pollfd *pfd, timestamp_t *next_timestamp) {
	conn_impl_t *conn_impl = agent->conn_impl;
	mutex_lock(&conn_impl->mutex);
	if (conn_impl->stopped) {
		mutex_unlock(&conn_impl->mutex);
		return 0;
	}

	pfd->fd = conn_impl->sock;
	pfd->events = POLLIN;

	*next_timestamp = conn_impl->next_timestamp;

	mutex_unlock(&conn_impl->mutex);
	return 1;
}

int conn_thread_process(juice_agent_t *agent, struct pollfd *pfd) {
	conn_impl_t *conn_impl = agent->conn_impl;
	mutex_lock(&conn_impl->mutex);
	if (conn_impl->stopped) {
		mutex_unlock(&conn_impl->mutex);
		return -1;
	}

	if (pfd->revents & POLLNVAL || pfd->revents & POLLERR) {
		JLOG_ERROR("Error when polling socket");
		agent_conn_fail(agent);
		mutex_unlock(&conn_impl->mutex);
		return -1;
	}

	if (pfd->revents & POLLIN) {
		char buffer[BUFFER_SIZE];
		addr_record_t src;
		int ret;
		while ((ret = conn_thread_recv(conn_impl->sock, buffer, BUFFER_SIZE, &src)) > 0) {
			if (agent_conn_recv(agent, buffer, (size_t)ret, &src) != 0) {
				JLOG_WARN("Agent receive failed");
				mutex_unlock(&conn_impl->mutex);
				return -1;
			}
		}

		if (ret < 0) {
			agent_conn_fail(agent);
			mutex_unlock(&conn_impl->mutex);
			return -1;
		}

		if (agent_conn_update(agent, &conn_impl->next_timestamp) != 0) {
			JLOG_WARN("Agent update failed");
			mutex_unlock(&conn_impl->mutex);
			return -1;
		}

	} else if (conn_impl->next_timestamp <= current_timestamp()) {
		if (agent_conn_update(agent, &conn_impl->next_timestamp) != 0) {
			JLOG_WARN("Agent update failed");
			mutex_unlock(&conn_impl->mutex);
			return -1;
		}
	}

	mutex_unlock(&conn_impl->mutex);
	return 0;
}

int conn_thread_recv(socket_t sock, char *buffer, size_t size, addr_record_t *src) {
	JLOG_VERBOSE("Receiving datagram");
	int len;
	while ((len = udp_recvfrom(sock, buffer, size, src)) == 0) {
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

int conn_thread_run(juice_agent_t *agent) {
	struct pollfd pfd[1];
	timestamp_t next_timestamp;
	while (conn_thread_prepare(agent, pfd, &next_timestamp) > 0) {
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

		if (conn_thread_process(agent, pfd) < 0)
			break;
	}

	JLOG_DEBUG("Leaving connection thread");
	return 0;
}

int conn_thread_init(juice_agent_t *agent, conn_registry_t *registry, udp_socket_config_t *config) {
	(void)registry;

	conn_impl_t *conn_impl = calloc(1, sizeof(conn_impl_t));
	if (!conn_impl) {
		JLOG_FATAL("Memory allocation failed for connection impl");
		return -1;
	}

	conn_impl->sock = udp_create_socket(config);
	if (conn_impl->sock == INVALID_SOCKET) {
		JLOG_ERROR("UDP socket creation failed");
		free(conn_impl);
		return -1;
	}

	mutex_init(&conn_impl->mutex, 0);
	mutex_init(&conn_impl->send_mutex, 0);

	agent->conn_impl = conn_impl;

	JLOG_DEBUG("Starting connection thread");
	int ret = thread_init(&conn_impl->thread, conn_thread_entry, agent);
	if (ret) {
		JLOG_FATAL("Thread creation failed, error=%d", ret);
		free(conn_impl);
		agent->conn_impl = NULL;
		return -1;
	}

	return 0;
}

void conn_thread_cleanup(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;

	mutex_lock(&conn_impl->mutex);
	conn_impl->stopped = true;
	mutex_unlock(&conn_impl->mutex);

	conn_thread_interrupt(agent);

	JLOG_VERBOSE("Waiting for connection thread");
	thread_join(conn_impl->thread, NULL);

	closesocket(conn_impl->sock);
	mutex_destroy(&conn_impl->mutex);
	mutex_destroy(&conn_impl->send_mutex);
	free(agent->conn_impl);
	agent->conn_impl = NULL;
}

void conn_thread_lock(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;
	mutex_lock(&conn_impl->mutex);
}

void conn_thread_unlock(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;
	mutex_unlock(&conn_impl->mutex);
}

int conn_thread_interrupt(juice_agent_t *agent) {
	conn_impl_t *conn_impl = agent->conn_impl;

	mutex_lock(&conn_impl->mutex);
	conn_impl->next_timestamp = current_timestamp();
	mutex_unlock(&conn_impl->mutex);

	JLOG_VERBOSE("Interrupting connection thread");

	mutex_lock(&conn_impl->send_mutex);
	if (udp_sendto_self(conn_impl->sock, NULL, 0) < 0) {
		if (sockerrno != SEAGAIN && sockerrno != SEWOULDBLOCK) {
			JLOG_WARN("Failed to interrupt poll by triggering socket, errno=%d", sockerrno);
		}
		mutex_unlock(&conn_impl->send_mutex);
		return -1;
	}

	mutex_unlock(&conn_impl->send_mutex);
	return 0;
}

int conn_thread_send(juice_agent_t *agent, const addr_record_t *dst, const char *data, size_t size,
                     int ds) {
	conn_impl_t *conn_impl = agent->conn_impl;

	mutex_lock(&conn_impl->send_mutex);

	if (conn_impl->send_ds >= 0 && conn_impl->send_ds != ds) {
		JLOG_VERBOSE("Setting Differentiated Services field to 0x%X", ds);
		if (udp_set_diffserv(conn_impl->sock, ds) == 0)
			conn_impl->send_ds = ds;
		else
			conn_impl->send_ds = -1; // disable for next time
	}

	JLOG_VERBOSE("Sending datagram, size=%d", size);

	int ret = udp_sendto(conn_impl->sock, data, size, dst);
	if (ret < 0) {
		if (sockerrno == SEAGAIN || sockerrno == SEWOULDBLOCK)
			JLOG_INFO("Send failed, buffer is full");
		else if (sockerrno == SEMSGSIZE)
			JLOG_WARN("Send failed, datagram is too large");
		else
			JLOG_WARN("Send failed, errno=%d", sockerrno);
	}

	mutex_unlock(&conn_impl->send_mutex);
	return ret;
}

int conn_thread_get_addrs(juice_agent_t *agent, addr_record_t *records, size_t size) {
	conn_impl_t *conn_impl = agent->conn_impl;

	return udp_get_addrs(conn_impl->sock, records, size);
}

