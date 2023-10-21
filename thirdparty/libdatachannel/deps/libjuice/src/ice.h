/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_ICE_H
#define JUICE_ICE_H

#include "addr.h"
#include "juice.h"
#include "timestamp.h"

#include <stdbool.h>
#include <stdint.h>

#define ICE_MAX_CANDIDATES_COUNT 20 // ~ 500B * 20 = 10KB

typedef enum ice_candidate_type {
	ICE_CANDIDATE_TYPE_UNKNOWN,
	ICE_CANDIDATE_TYPE_HOST,
	ICE_CANDIDATE_TYPE_SERVER_REFLEXIVE,
	ICE_CANDIDATE_TYPE_PEER_REFLEXIVE,
	ICE_CANDIDATE_TYPE_RELAYED,
} ice_candidate_type_t;

// RFC 8445: The RECOMMENDED values for type preferences are 126 for host candidates, 110 for
// peer-reflexive candidates, 100 for server-reflexive candidates, and 0 for relayed candidates.
#define ICE_CANDIDATE_PREF_HOST 126
#define ICE_CANDIDATE_PREF_PEER_REFLEXIVE 110
#define ICE_CANDIDATE_PREF_SERVER_REFLEXIVE 100
#define ICE_CANDIDATE_PREF_RELAYED 0

typedef struct ice_candidate {
	ice_candidate_type_t type;
	uint32_t priority;
	int component;
	char foundation[32 + 1]; // 1 to 32 characters
	char transport[32 + 1];
	char hostname[256 + 1];
	char service[32 + 1];
	addr_record_t resolved;
} ice_candidate_t;

typedef struct ice_description {
	char ice_ufrag[256 + 1]; // 4 to 256 characters
	char ice_pwd[256 + 1];   // 22 to 256 characters
	bool ice_lite;
	ice_candidate_t candidates[ICE_MAX_CANDIDATES_COUNT];
	int candidates_count;
	bool finished;
} ice_description_t;

typedef enum ice_candidate_pair_state {
	ICE_CANDIDATE_PAIR_STATE_PENDING,
	ICE_CANDIDATE_PAIR_STATE_SUCCEEDED,
	ICE_CANDIDATE_PAIR_STATE_FAILED,
	ICE_CANDIDATE_PAIR_STATE_FROZEN,
} ice_candidate_pair_state_t;

typedef struct ice_candidate_pair {
	ice_candidate_t *local;
	ice_candidate_t *remote;
	uint64_t priority;
	ice_candidate_pair_state_t state;
	bool nominated;
	bool nomination_requested;
	timestamp_t consent_expiry;
} ice_candidate_pair_t;

typedef enum ice_resolve_mode {
	ICE_RESOLVE_MODE_SIMPLE,
	ICE_RESOLVE_MODE_LOOKUP,
} ice_resolve_mode_t;

#define ICE_PARSE_ERROR -1
#define ICE_PARSE_IGNORED -2
#define ICE_PARSE_MISSING_UFRAG -3
#define ICE_PARSE_MISSING_PWD -4

int ice_parse_sdp(const char *sdp, ice_description_t *description);
int ice_parse_candidate_sdp(const char *line, ice_candidate_t *candidate);
int ice_create_local_description(ice_description_t *description);
int ice_create_local_candidate(ice_candidate_type_t type, int component, int index,
                               const addr_record_t *record, ice_candidate_t *candidate);
int ice_resolve_candidate(ice_candidate_t *candidate, ice_resolve_mode_t mode);
int ice_add_candidate(ice_candidate_t *candidate, ice_description_t *description);
void ice_sort_candidates(ice_description_t *description);
ice_candidate_t *ice_find_candidate_from_addr(ice_description_t *description,
                                              const addr_record_t *record,
                                              ice_candidate_type_t type);
int ice_generate_sdp(const ice_description_t *description, char *buffer, size_t size);
int ice_generate_candidate_sdp(const ice_candidate_t *candidate, char *buffer, size_t size);
int ice_create_candidate_pair(ice_candidate_t *local, ice_candidate_t *remote, bool is_controlling,
                              ice_candidate_pair_t *pair); // local or remote might be NULL
int ice_update_candidate_pair(ice_candidate_pair_t *pair, bool is_controlling);

int ice_candidates_count(const ice_description_t *description, ice_candidate_type_t type);

uint32_t ice_compute_priority(ice_candidate_type_t type, int family, int component, int index);

#endif
