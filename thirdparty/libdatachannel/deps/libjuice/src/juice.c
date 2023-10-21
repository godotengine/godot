/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "juice.h"
#include "addr.h"
#include "agent.h"
#include "ice.h"

#ifndef NO_SERVER
#include "server.h"
#endif

#include <stdio.h>

JUICE_EXPORT juice_agent_t *juice_create(const juice_config_t *config) {
	if (!config)
		return NULL;

	return agent_create(config);
}

JUICE_EXPORT void juice_destroy(juice_agent_t *agent) {
	if (agent)
		agent_destroy(agent);
}

JUICE_EXPORT int juice_gather_candidates(juice_agent_t *agent) {
	if (!agent)
		return JUICE_ERR_INVALID;

	if (agent_gather_candidates(agent) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT int juice_get_local_description(juice_agent_t *agent, char *buffer, size_t size) {
	if (!agent || (!buffer && size))
		return JUICE_ERR_INVALID;

	if (agent_get_local_description(agent, buffer, size) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT int juice_set_remote_description(juice_agent_t *agent, const char *sdp) {
	if (!agent || !sdp)
		return JUICE_ERR_INVALID;

	if (agent_set_remote_description(agent, sdp) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT int juice_add_remote_candidate(juice_agent_t *agent, const char *sdp) {
	if (!agent || !sdp)
		return JUICE_ERR_INVALID;

	if (agent_add_remote_candidate(agent, sdp) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT int juice_set_remote_gathering_done(juice_agent_t *agent) {
	if (!agent)
		return JUICE_ERR_INVALID;

	if (agent_set_remote_gathering_done(agent) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT int juice_send(juice_agent_t *agent, const char *data, size_t size) {
	if (!agent || (!data && size))
		return JUICE_ERR_INVALID;

	if (agent_send(agent, data, size, 0) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT int juice_send_diffserv(juice_agent_t *agent, const char *data, size_t size, int ds) {
	if (!agent || (!data && size))
		return JUICE_ERR_INVALID;

	if (agent_send(agent, data, size, ds) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT juice_state_t juice_get_state(juice_agent_t *agent) { return agent_get_state(agent); }

JUICE_EXPORT int juice_get_selected_candidates(juice_agent_t *agent, char *local, size_t local_size,
                                               char *remote, size_t remote_size) {
	if (!agent || (!local && local_size) || (!remote && remote_size))
		return JUICE_ERR_INVALID;

	ice_candidate_t local_cand, remote_cand;
	if (agent_get_selected_candidate_pair(agent, &local_cand, &remote_cand))
		return JUICE_ERR_NOT_AVAIL;

	if (local_size && ice_generate_candidate_sdp(&local_cand, local, local_size) < 0)
		return JUICE_ERR_FAILED;

	if (remote_size && ice_generate_candidate_sdp(&remote_cand, remote, remote_size) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT int juice_get_selected_addresses(juice_agent_t *agent, char *local, size_t local_size,
                                              char *remote, size_t remote_size) {
	if (!agent || (!local && local_size) || (!remote && remote_size))
		return JUICE_ERR_INVALID;

	ice_candidate_t local_cand, remote_cand;
	if (agent_get_selected_candidate_pair(agent, &local_cand, &remote_cand))
		return JUICE_ERR_NOT_AVAIL;

	if (local_size && addr_record_to_string(&local_cand.resolved, local, local_size) < 0)
		return JUICE_ERR_FAILED;

	if (remote_size && addr_record_to_string(&remote_cand.resolved, remote, remote_size) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
}

JUICE_EXPORT const char *juice_state_to_string(juice_state_t state) {
	switch (state) {
	case JUICE_STATE_DISCONNECTED:
		return "disconnected";
	case JUICE_STATE_GATHERING:
		return "gathering";
	case JUICE_STATE_CONNECTING:
		return "connecting";
	case JUICE_STATE_CONNECTED:
		return "connected";
	case JUICE_STATE_COMPLETED:
		return "completed";
	case JUICE_STATE_FAILED:
		return "failed";
	default:
		return "unknown";
	}
}

JUICE_EXPORT juice_server_t *juice_server_create(const juice_server_config_t *config) {
#ifndef NO_SERVER
	if (!config)
		return NULL;

	return server_create(config);
#else
	(void)config;
	JLOG_FATAL("The library was compiled without server support");
	return NULL;
#endif
}

JUICE_EXPORT void juice_server_destroy(juice_server_t *server) {
#ifndef NO_SERVER
	if (server)
		server_destroy(server);
#else
	(void)server;
#endif
}

JUICE_EXPORT uint16_t juice_server_get_port(juice_server_t *server) {
#ifndef NO_SERVER
	return server ? server_get_port(server) : 0;
#else
	(void)server;
	return 0;
#endif
}

JUICE_EXPORT int juice_server_add_credentials(juice_server_t *server,
                                              const juice_server_credentials_t *credentials,
                                              unsigned long lifetime_ms) {
#ifndef NO_SERVER
	if (!server || !credentials)
		return JUICE_ERR_INVALID;

	if (server_add_credentials(server, credentials, (timediff_t)lifetime_ms) < 0)
		return JUICE_ERR_FAILED;

	return JUICE_ERR_SUCCESS;
#else
	(void)server;
	(void)credentials;
	(void)lifetime_ms;
	return JUICE_ERR_INVALID;
#endif
}
