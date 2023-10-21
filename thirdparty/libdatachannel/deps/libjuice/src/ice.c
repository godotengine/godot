/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ice.h"
#include "log.h"
#include "random.h"

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 1024

#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

// See RFC4566 for SDP format: https://www.rfc-editor.org/rfc/rfc4566.html

static const char *skip_prefix(const char *str, const char *prefix) {
	size_t len = strlen(prefix);
	return strncmp(str, prefix, len) == 0 ? str + len : str;
}

static bool match_prefix(const char *str, const char *prefix, const char **end) {
	*end = skip_prefix(str, prefix);
	return *end != str || !*prefix;
}

static int parse_sdp_line(const char *line, ice_description_t *description) {
	const char *arg;
	if (match_prefix(line, "a=ice-ufrag:", &arg)) {
		sscanf(arg, "%256s", description->ice_ufrag);
		return 0;
	}
	if (match_prefix(line, "a=ice-pwd:", &arg)) {
		sscanf(arg, "%256s", description->ice_pwd);
		return 0;
	}
	if (match_prefix(line, "a=ice-lite", &arg)) {
		description->ice_lite = true;
		return 0;
	}
	if (match_prefix(line, "a=end-of-candidates", &arg)) {
		description->finished = true;
		return 0;
	}
	ice_candidate_t candidate;
	if (ice_parse_candidate_sdp(line, &candidate) == 0) {
		ice_add_candidate(&candidate, description);
		return 0;
	}
	return ICE_PARSE_IGNORED;
}

static int parse_sdp_candidate(const char *line, ice_candidate_t *candidate) {
	memset(candidate, 0, sizeof(*candidate));

	line = skip_prefix(line, "a=");
	line = skip_prefix(line, "candidate:");

	char transport[32 + 1];
	char type[32 + 1];
	if (sscanf(line, "%32s %d %32s %u %256s %32s typ %32s", candidate->foundation,
	           &candidate->component, transport, &candidate->priority, candidate->hostname,
	           candidate->service, type) != 7) {
		JLOG_WARN("Failed to parse candidate: %s", line);
		return ICE_PARSE_ERROR;
	}

	for (int i = 0; transport[i]; ++i)
		transport[i] = toupper((unsigned char)transport[i]);

	for (int i = 0; type[i]; ++i)
		type[i] = tolower((unsigned char)type[i]);

	if (strcmp(type, "host") == 0)
		candidate->type = ICE_CANDIDATE_TYPE_HOST;
	else if (strcmp(type, "srflx") == 0)
		candidate->type = ICE_CANDIDATE_TYPE_SERVER_REFLEXIVE;
	else if (strcmp(type, "relay") == 0)
		candidate->type = ICE_CANDIDATE_TYPE_RELAYED;
	else {
		JLOG_WARN("Ignoring candidate with unknown type \"%s\"", type);
		return ICE_PARSE_IGNORED;
	}

	if (strcmp(transport, "UDP") != 0) {
		JLOG_WARN("Ignoring candidate with transport %s", transport);
		return ICE_PARSE_IGNORED;
	}

	return 0;
}

int ice_parse_sdp(const char *sdp, ice_description_t *description) {
	memset(description, 0, sizeof(*description));
	description->ice_lite = false;
	description->candidates_count = 0;
	description->finished = false;

	char buffer[BUFFER_SIZE];
	size_t size = 0;
	while (*sdp) {
		if (*sdp == '\n') {
			if (size) {
				buffer[size++] = '\0';
				if(parse_sdp_line(buffer, description) == ICE_PARSE_ERROR)
					return ICE_PARSE_ERROR;

				size = 0;
			}
		} else if (*sdp != '\r' && size + 1 < BUFFER_SIZE) {
			buffer[size++] = *sdp;
		}
		++sdp;
	}
	ice_sort_candidates(description);

	JLOG_DEBUG("Parsed remote description: ufrag=\"%s\", pwd=\"%s\", candidates=%d",
	           description->ice_ufrag, description->ice_pwd, description->candidates_count);

	if (*description->ice_ufrag == '\0')
		return ICE_PARSE_MISSING_UFRAG;

	if (*description->ice_pwd == '\0')
		return ICE_PARSE_MISSING_PWD;

	return 0;
}

int ice_parse_candidate_sdp(const char *line, ice_candidate_t *candidate) {
	const char *arg;
	if (match_prefix(line, "a=candidate:", &arg)) {
		int ret = parse_sdp_candidate(line, candidate);
		if (ret < 0)
			return ret;
		ice_resolve_candidate(candidate, ICE_RESOLVE_MODE_SIMPLE);
		return 0;
	}
	return ICE_PARSE_ERROR;
}

int ice_create_local_description(ice_description_t *description) {
	memset(description, 0, sizeof(*description));
	juice_random_str64(description->ice_ufrag, 4 + 1);
	juice_random_str64(description->ice_pwd, 22 + 1);
	description->ice_lite = false;
	description->candidates_count = 0;
	description->finished = false;
	JLOG_DEBUG("Created local description: ufrag=\"%s\", pwd=\"%s\"", description->ice_ufrag,
	           description->ice_pwd);
	return 0;
}

int ice_create_local_candidate(ice_candidate_type_t type, int component, int index,
                               const addr_record_t *record, ice_candidate_t *candidate) {
	memset(candidate, 0, sizeof(*candidate));
	candidate->type = type;
	candidate->component = component;
	candidate->resolved = *record;
	strcpy(candidate->foundation, "-");

	candidate->priority = ice_compute_priority(candidate->type, candidate->resolved.addr.ss_family,
	                                           candidate->component, index);

	if (getnameinfo((struct sockaddr *)&record->addr, record->len, candidate->hostname, 256,
	                candidate->service, 32, NI_NUMERICHOST | NI_NUMERICSERV | NI_DGRAM)) {
		JLOG_ERROR("getnameinfo failed, errno=%d", sockerrno);
		return -1;
	}
	return 0;
}

int ice_resolve_candidate(ice_candidate_t *candidate, ice_resolve_mode_t mode) {
	struct addrinfo hints;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_DGRAM;
	hints.ai_protocol = IPPROTO_UDP;
	hints.ai_flags = AI_ADDRCONFIG;
	if (mode != ICE_RESOLVE_MODE_LOOKUP)
		hints.ai_flags |= AI_NUMERICHOST | AI_NUMERICSERV;
	struct addrinfo *ai_list = NULL;
	if (getaddrinfo(candidate->hostname, candidate->service, &hints, &ai_list)) {
		JLOG_INFO("Failed to resolve address: %s:%s", candidate->hostname, candidate->service);
		candidate->resolved.len = 0;
		return -1;
	}
	for (struct addrinfo *ai = ai_list; ai; ai = ai->ai_next) {
		if (ai->ai_family == AF_INET || ai->ai_family == AF_INET6) {
			candidate->resolved.len = (socklen_t)ai->ai_addrlen;
			memcpy(&candidate->resolved.addr, ai->ai_addr, ai->ai_addrlen);
			break;
		}
	}
	freeaddrinfo(ai_list);
	return 0;
}

int ice_add_candidate(ice_candidate_t *candidate, ice_description_t *description) {
	if (candidate->type == ICE_CANDIDATE_TYPE_UNKNOWN)
		return -1;

	if (description->candidates_count >= ICE_MAX_CANDIDATES_COUNT) {
		JLOG_WARN("Description already has the maximum number of candidates");
		return -1;
	}

	if (strcmp(candidate->foundation, "-") == 0)
		snprintf(candidate->foundation, 32, "%u",
		         (unsigned int)(description->candidates_count + 1));

	ice_candidate_t *pos = description->candidates + description->candidates_count;
	*pos = *candidate;
	++description->candidates_count;
	return 0;
}

void ice_sort_candidates(ice_description_t *description) {
	// In-place insertion sort
	ice_candidate_t *begin = description->candidates;
	ice_candidate_t *end = begin + description->candidates_count;
	ice_candidate_t *cur = begin;
	while (++cur < end) {
		uint32_t priority = cur->priority;
		ice_candidate_t *prev = cur;
		ice_candidate_t tmp = *prev;
		while (--prev >= begin && prev->priority < priority) {
			*(prev + 1) = *prev;
		}
		if (prev + 1 != cur)
			*(prev + 1) = tmp;
	}
}

ice_candidate_t *ice_find_candidate_from_addr(ice_description_t *description,
                                              const addr_record_t *record,
                                              ice_candidate_type_t type) {
	ice_candidate_t *cur = description->candidates;
	ice_candidate_t *end = cur + description->candidates_count;
	while (cur != end) {
		if ((type == ICE_CANDIDATE_TYPE_UNKNOWN || cur->type == type) &&
		    addr_is_equal((struct sockaddr *)&record->addr, (struct sockaddr *)&cur->resolved.addr,
		                  true))
			return cur;
		++cur;
	}
	return NULL;
}

int ice_generate_sdp(const ice_description_t *description, char *buffer, size_t size) {
	if (!*description->ice_ufrag || !*description->ice_pwd)
		return -1;

	int len = 0;
	char *begin = buffer;
	char *end = begin + size;

	// Round 0: description
	// Round i with i>0 and i<count+1: candidate i-1
	// Round count + 1: end-of-candidates and ice-options lines
	for (int i = 0; i < description->candidates_count + 2; ++i) {
		int ret;
		if (i == 0) {
			ret = snprintf(begin, end - begin, "a=ice-ufrag:%s\r\na=ice-pwd:%s\r\n",
			               description->ice_ufrag, description->ice_pwd);
			if (description->ice_lite)
				ret = snprintf(begin, end - begin, "a=ice-lite\r\n");

		} else if (i < description->candidates_count + 1) {
			const ice_candidate_t *candidate = description->candidates + i - 1;
			if (candidate->type == ICE_CANDIDATE_TYPE_UNKNOWN ||
			    candidate->type == ICE_CANDIDATE_TYPE_PEER_REFLEXIVE)
				continue;
			char tmp[BUFFER_SIZE];
			if (ice_generate_candidate_sdp(candidate, tmp, BUFFER_SIZE) < 0)
				continue;
			ret = snprintf(begin, end - begin, "%s\r\n", tmp);
		} else { // i == description->candidates_count + 1
			// RFC 8445 10. ICE Option: An agent compliant to this specification MUST inform the
			// peer about the compliance using the 'ice2' option.
			if (description->finished)
				ret = snprintf(begin, end - begin, "a=end-of-candidates\r\na=ice-options:ice2\r\n");
			else
				ret = snprintf(begin, end - begin, "a=ice-options:ice2,trickle\r\n");
		}
		if (ret < 0)
			return -1;

		len += ret;

		if (begin < end)
			begin += ret >= end - begin ? end - begin - 1 : ret;
	}
	return len;
}

int ice_generate_candidate_sdp(const ice_candidate_t *candidate, char *buffer, size_t size) {
	const char *type = NULL;
	const char *suffix = NULL;
	switch (candidate->type) {
	case ICE_CANDIDATE_TYPE_HOST:
		type = "host";
		break;
	case ICE_CANDIDATE_TYPE_PEER_REFLEXIVE:
		type = "prflx";
		break;
	case ICE_CANDIDATE_TYPE_SERVER_REFLEXIVE:
		type = "srflx";
		suffix = "raddr 0.0.0.0 rport 0"; // This is needed for compatibility with Firefox
		break;
	case ICE_CANDIDATE_TYPE_RELAYED:
		type = "relay";
		suffix = "raddr 0.0.0.0 rport 0"; // This is needed for compatibility with Firefox
		break;
	default:
		JLOG_ERROR("Unknown candidate type");
		return -1;
	}
	return snprintf(buffer, size, "a=candidate:%s %u UDP %u %s %s typ %s%s%s",
	                candidate->foundation, candidate->component, candidate->priority,
	                candidate->hostname, candidate->service, type, suffix ? " " : "",
	                suffix ? suffix : "");
}

int ice_create_candidate_pair(ice_candidate_t *local, ice_candidate_t *remote, bool is_controlling,
                              ice_candidate_pair_t *pair) { // local or remote might be NULL
	if (local && remote && local->resolved.addr.ss_family != remote->resolved.addr.ss_family) {
		JLOG_ERROR("Mismatching candidates address families");
		return -1;
	}

	memset(pair, 0, sizeof(*pair));
	pair->local = local;
	pair->remote = remote;
	pair->state = ICE_CANDIDATE_PAIR_STATE_FROZEN;
	return ice_update_candidate_pair(pair, is_controlling);
}

int ice_update_candidate_pair(ice_candidate_pair_t *pair, bool is_controlling) {
	// Compute pair priority according to RFC 8445, extended to support generic pairs missing local
	// or remote See https://www.rfc-editor.org/rfc/rfc8445.html#section-6.1.2.3
	if (!pair->local && !pair->remote)
		return 0;
	uint64_t local_priority =
	    pair->local
	        ? pair->local->priority
	        : ice_compute_priority(ICE_CANDIDATE_TYPE_HOST, pair->remote->resolved.addr.ss_family,
	                               pair->remote->component, 0);
	uint64_t remote_priority =
	    pair->remote
	        ? pair->remote->priority
	        : ice_compute_priority(ICE_CANDIDATE_TYPE_HOST, pair->local->resolved.addr.ss_family,
	                               pair->local->component, 0);
	uint64_t g = is_controlling ? local_priority : remote_priority;
	uint64_t d = is_controlling ? remote_priority : local_priority;
	uint64_t min = g < d ? g : d;
	uint64_t max = g > d ? g : d;
	pair->priority = (min << 32) + (max << 1) + (g > d ? 1 : 0);
	return 0;
}

int ice_candidates_count(const ice_description_t *description, ice_candidate_type_t type) {
	int count = 0;
	for (int i = 0; i < description->candidates_count; ++i) {
		const ice_candidate_t *candidate = description->candidates + i;
		if (candidate->type == type)
			++count;
	}
	return count;
}

uint32_t ice_compute_priority(ice_candidate_type_t type, int family, int component, int index) {
	// Compute candidate priority according to RFC 8445
	// See https://www.rfc-editor.org/rfc/rfc8445.html#section-5.1.2.1
	uint32_t p = 0;

	switch (type) {
	case ICE_CANDIDATE_TYPE_HOST:
		p += ICE_CANDIDATE_PREF_HOST;
		break;
	case ICE_CANDIDATE_TYPE_PEER_REFLEXIVE:
		p += ICE_CANDIDATE_PREF_PEER_REFLEXIVE;
		break;
	case ICE_CANDIDATE_TYPE_SERVER_REFLEXIVE:
		p += ICE_CANDIDATE_PREF_SERVER_REFLEXIVE;
		break;
	case ICE_CANDIDATE_TYPE_RELAYED:
		p += ICE_CANDIDATE_PREF_RELAYED;
		break;
	default:
		break;
	}
	p <<= 16;

	switch (family) {
	case AF_INET:
		p += 32767;
		break;
	case AF_INET6:
		p += 65535;
		break;
	default:
		break;
	}
	p -= CLAMP(index, 0, 32767);
	p <<= 8;

	p += 256 - CLAMP(component, 1, 256);
	return p;
}
