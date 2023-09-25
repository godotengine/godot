/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "turn.h"
#include "log.h"
#include "random.h"
#include "socket.h"

#include <string.h>

static bool memory_is_zero(const void *data, size_t size) {
	const char *d = data;
	for (size_t i = 0; i < size; ++i)
		if (d[i])
			return false;

	return true;
}

static uint16_t random_channel_number() {
	/*
	 * RFC 8656 12. Channels
	 * The ChannelData message (see Section 12.4) starts with a two-byte
	 * field that carries the channel number.  The values of this field are
	 * allocated as follows:
	 *
	 *   +------------------------+--------------------------------------+
	 *   | 0x0000 through 0x3FFF: | These values can never be used for   |
	 *   |                        | channel numbers.                     |
	 *   +------------------------+--------------------------------------+
	 *   | 0x4000 through 0x4FFF: | These values are the allowed channel |
	 *   |                        | numbers (4096 possible values).      |
	 *   +------------------------+--------------------------------------+
	 *   | 0x5000 through 0xFFFF: | Reserved (For DTLS-SRTP multiplexing |
	 *   |                        | collision avoidance, see [RFC7983]). |
	 *   +------------------------+--------------------------------------+
	 */
	uint16_t r;
	juice_random(&r, 2);
	return 0x4000 | (r & 0x0FFF);
}

bool is_channel_data(const void *data, size_t size) {
	// According RFC 8656, first byte in [64..79] is TURN Channel
	if (size == 0)
		return false;
	uint8_t b = *((const uint8_t *)data);
	return b >= 64 && b <= 79;
}

bool is_valid_channel(uint16_t channel) { return channel >= 0x4000; }

int turn_wrap_channel_data(char *buffer, size_t size, const char *data, size_t data_size,
                           uint16_t channel) {
	if (!is_valid_channel(channel)) {
		JLOG_WARN("Invalid channel number: 0x%hX", channel);
		return -1;
	}
	if (data_size >= 65536) {
		JLOG_WARN("ChannelData is too long, size=%zu", size);
		return -1;
	}
	if (size < sizeof(struct channel_data_header) + data_size) {
		JLOG_WARN("Buffer is too small to add ChannelData header, size=%zu, needed=%zu", size,
		          sizeof(struct channel_data_header) + data_size);
		return -1;
	}

	memmove(buffer + sizeof(struct channel_data_header), data, data_size);
	struct channel_data_header *header = (struct channel_data_header *)buffer;
	header->channel_number = htons((uint16_t)channel);
	header->length = htons((uint16_t)data_size);
	return (int)(sizeof(struct channel_data_header) + data_size);
}

static int find_ordered_channel_rec(turn_entry_t *const ordered_channels[], uint16_t channel,
                                    int begin, int end) {
	int d = end - begin;
	if (d <= 0)
		return begin;

	int pivot = begin + d / 2;
	const turn_entry_t *entry = ordered_channels[pivot];
	if (channel < entry->channel)
		return find_ordered_channel_rec(ordered_channels, channel, begin, pivot);
	else if (channel > entry->channel)
		return find_ordered_channel_rec(ordered_channels, channel, pivot + 1, end);
	else
		return pivot;
}

static int find_ordered_channel(const turn_map_t *map, uint16_t channel) {
	return find_ordered_channel_rec(map->ordered_channels, channel, 0, map->channels_count);
}

static int find_ordered_transaction_id_rec(turn_entry_t *const ordered_transaction_ids[],
                                           const uint8_t *transaction_id, int begin, int end) {
	int d = end - begin;
	if (d <= 0)
		return begin;

	int pivot = begin + d / 2;
	const turn_entry_t *entry = ordered_transaction_ids[pivot];
	int ret = memcmp(transaction_id, entry->transaction_id, STUN_TRANSACTION_ID_SIZE);
	if (ret < 0)
		return find_ordered_transaction_id_rec(ordered_transaction_ids, transaction_id, begin,
		                                       pivot);
	else if (ret > 0)
		return find_ordered_transaction_id_rec(ordered_transaction_ids, transaction_id, pivot + 1,
		                                       end);
	else
		return pivot;
}

static int find_ordered_transaction_id(const turn_map_t *map, const uint8_t *transaction_id) {
	return find_ordered_transaction_id_rec(map->ordered_transaction_ids, transaction_id, 0,
	                                       map->transaction_ids_count);
}

static void remove_ordered_transaction_id(turn_map_t *map, const uint8_t *transaction_id) {
	int pos = find_ordered_transaction_id(map, transaction_id);
	if (pos < map->transaction_ids_count) {
		memmove(map->ordered_transaction_ids + pos, map->ordered_transaction_ids + pos + 1,
		        (map->transaction_ids_count - (pos + 1)) * sizeof(turn_entry_t *));
		map->transaction_ids_count--;
	}
}
/*
static void remove_ordered_channel(turn_map_t *map, uint16_t channel) {
    int pos = find_ordered_channel(map, channel);
    if (pos < map->channels_count) {
        memmove(map->ordered_channels + pos, map->ordered_channels + pos + 1,
                (map->channels_count - (pos + 1)) * sizeof(turn_entry_t *));
        map->channels_count--;
    }
}

static void delete_entry(turn_map_t *map, turn_entry_t *entry) {
    if (entry->type == TURN_ENTRY_TYPE_EMPTY || entry->type == TURN_ENTRY_TYPE_DELETED)
        return;

    if (!memory_is_zero(entry->transaction_id, STUN_TRANSACTION_ID_SIZE))
        remove_ordered_transaction_id(map, entry->transaction_id);

    if (entry->type == TURN_ENTRY_TYPE_CHANNEL && entry->channel)
        remove_ordered_channel(map, entry->channel);

    memset(entry, 0, sizeof(*entry));
    entry->type = TURN_ENTRY_TYPE_DELETED;
}
*/
static turn_entry_t *find_entry(turn_map_t *map, const addr_record_t *record,
                                turn_entry_type_t type, bool allow_deleted) {
	unsigned long key = (addr_record_hash(record, false) + (int)type) % map->map_size;
	unsigned long pos = key;
	while (true) {
		turn_entry_t *entry = map->map + pos;
		if (entry->type == TURN_ENTRY_TYPE_EMPTY ||
		    (entry->type == type && addr_record_is_equal(&entry->record, record, false)))
			break;

		if (allow_deleted && entry->type == TURN_ENTRY_TYPE_DELETED)
			break;

		pos = (pos + 1) % map->map_size;
		if (pos == key) {
			JLOG_VERBOSE("TURN map is full");
			return NULL;
		}
	}
	return map->map + pos;
}

static bool update_timestamp(turn_map_t *map, turn_entry_type_t type, const uint8_t *transaction_id,
                             const addr_record_t *record, timediff_t duration) {
	turn_entry_t *entry;
	if (record) {
		entry = find_entry(map, record, type, true);
		if (!entry)
			return false;

		if (entry->type == type) {
			if (memcmp(entry->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE) == 0)
				return true;
		} else {
			entry->type = type;
			entry->record = *record;
		}

		if (!memory_is_zero(entry->transaction_id, STUN_TRANSACTION_ID_SIZE))
			remove_ordered_transaction_id(map, entry->transaction_id);

		memcpy(entry->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE);

	} else {
		int pos = find_ordered_transaction_id(map, transaction_id);
		if (pos == map->transaction_ids_count)
			return false;

		entry = map->ordered_transaction_ids[pos];
		if (entry->type != type ||
		    memcmp(entry->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE) != 0)
			return false;
	}

	entry->timestamp = current_timestamp() + duration;
	entry->fresh_transaction_id = false;
	return true;
}

int turn_init_map(turn_map_t *map, int size) {
	memset(map, 0, sizeof(*map));

	map->map_size = size * 2;
	map->channels_count = 0;
	map->transaction_ids_count = 0;

	map->map = calloc(map->map_size, sizeof(turn_entry_t));
	map->ordered_channels = calloc(map->map_size, sizeof(turn_entry_t *));
	map->ordered_transaction_ids = calloc(map->map_size, sizeof(turn_entry_t *));

	if (!map->map || !map->ordered_channels || !map->ordered_transaction_ids) {
		JLOG_ERROR("Failed to allocate TURN map of size %d", size);
		turn_destroy_map(map);
		return -1;
	}

	return 0;
}

void turn_destroy_map(turn_map_t *map) {
	free(map->map);
	free(map->ordered_channels);
	free(map->ordered_transaction_ids);
}

bool turn_set_permission(turn_map_t *map, const uint8_t *transaction_id,
                         const addr_record_t *record, timediff_t duration) {
	return update_timestamp(map, TURN_ENTRY_TYPE_PERMISSION, transaction_id, record, duration);
}

bool turn_has_permission(turn_map_t *map, const addr_record_t *record) {
	turn_entry_t *entry = find_entry(map, record, TURN_ENTRY_TYPE_PERMISSION, false);
	if (!entry || entry->type != TURN_ENTRY_TYPE_PERMISSION)
		return false;

	return current_timestamp() < entry->timestamp;
}

bool turn_bind_channel(turn_map_t *map, const addr_record_t *record, const uint8_t *transaction_id,
                       uint16_t channel, timediff_t duration) {
	if (!is_valid_channel(channel)) {
		JLOG_ERROR("Invalid channel number: 0x%hX", channel);
		return false;
	}

	turn_entry_t *entry = find_entry(map, record, TURN_ENTRY_TYPE_CHANNEL, true);
	if (!entry)
		return false;

	if (entry->type == TURN_ENTRY_TYPE_CHANNEL && entry->channel) {
		if (entry->channel != channel) {
			JLOG_WARN("The record is already bound to a channel");
			return false;
		}

		entry->timestamp = current_timestamp() + duration;
		return true;
	}

	int pos = find_ordered_channel(map, channel);
	if (pos < map->channels_count) {
		const turn_entry_t *other_entry = map->ordered_channels[pos];
		if (other_entry->channel == channel) {
			JLOG_WARN("The channel is already bound to a record");
			return false;
		}
	}

	if (entry->type != TURN_ENTRY_TYPE_CHANNEL) {
		entry->type = TURN_ENTRY_TYPE_CHANNEL;
		entry->record = *record;
	}

	memmove(map->ordered_channels + pos + 1, map->ordered_channels + pos,
	        (map->channels_count - pos) * sizeof(turn_entry_t *));
	map->ordered_channels[pos] = entry;
	map->channels_count++;

	entry->channel = channel;
	entry->timestamp = current_timestamp() + duration;

	if (transaction_id) {
		memcpy(entry->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE);
		entry->fresh_transaction_id = true;
	}

	return true;
}

bool turn_bind_random_channel(turn_map_t *map, const addr_record_t *record, uint16_t *channel,
                              timediff_t duration) {
	uint16_t c;
	do {
		c = random_channel_number();
	} while (turn_find_channel(map, c, NULL));

	if (!turn_bind_channel(map, record, NULL, c, duration))
		return false;

	if (channel)
		*channel = c;

	return true;
}

bool turn_bind_current_channel(turn_map_t *map, const uint8_t *transaction_id,
                               const addr_record_t *record, timediff_t duration) {
	return update_timestamp(map, TURN_ENTRY_TYPE_CHANNEL, transaction_id, record, duration);
}

bool turn_get_channel(turn_map_t *map, const addr_record_t *record, uint16_t *channel) {
	turn_entry_t *entry = find_entry(map, record, TURN_ENTRY_TYPE_CHANNEL, false);
	if (!entry || entry->type != TURN_ENTRY_TYPE_CHANNEL)
		return false;

	if (channel)
		*channel = entry->channel;

	return true;
}

bool turn_get_bound_channel(turn_map_t *map, const addr_record_t *record, uint16_t *channel) {
	turn_entry_t *entry = find_entry(map, record, TURN_ENTRY_TYPE_CHANNEL, false);
	if (!entry || entry->type != TURN_ENTRY_TYPE_CHANNEL)
		return false;

	if (!entry->channel || current_timestamp() >= entry->timestamp)
		return false;

	if (channel)
		*channel = entry->channel;

	return true;
}

bool turn_find_channel(turn_map_t *map, uint16_t channel, addr_record_t *record) {
	if (!is_valid_channel(channel)) {
		JLOG_WARN("Invalid channel number: 0x%hX", channel);
		return false;
	}

	int pos = find_ordered_channel(map, channel);
	if (pos == map->channels_count)
		return false;

	const turn_entry_t *entry = map->ordered_channels[pos];
	if (entry->channel != channel)
		return false;

	if (record)
		*record = entry->record;

	return true;
}

bool turn_find_bound_channel(turn_map_t *map, uint16_t channel, addr_record_t *record) {
	if (!is_valid_channel(channel)) {
		JLOG_WARN("Invalid channel number: 0x%hX", channel);
		return false;
	}

	int pos = find_ordered_channel(map, channel);
	if (pos == map->channels_count)
		return false;

	const turn_entry_t *entry = map->ordered_channels[pos];
	if (entry->channel != channel || current_timestamp() >= entry->timestamp)
		return false;

	if (record)
		*record = entry->record;

	return true;
}

static bool set_transaction_id(turn_map_t *map, turn_entry_type_t type, const addr_record_t *record,
                               const uint8_t *transaction_id) {
	if (type != TURN_ENTRY_TYPE_PERMISSION && type != TURN_ENTRY_TYPE_CHANNEL)
		return false;

	turn_entry_t *entry = find_entry(map, record, type, true);
	if (!entry)
		return false;

	if (entry->type == type && !memory_is_zero(entry->transaction_id, STUN_TRANSACTION_ID_SIZE))
		remove_ordered_transaction_id(map, entry->transaction_id);

	int pos = find_ordered_transaction_id(map, transaction_id);
	memmove(map->ordered_transaction_ids + pos + 1, map->ordered_transaction_ids + pos,
	        (map->transaction_ids_count - pos) * sizeof(turn_entry_t *));
	map->ordered_transaction_ids[pos] = entry;
	map->transaction_ids_count++;

	if (entry->type != type) {
		entry->type = type;
		entry->record = *record;
	}

	memcpy(entry->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE);
	entry->fresh_transaction_id = true;
	return true;
}

static bool find_transaction_id(turn_map_t *map, const uint8_t *transaction_id,
                                addr_record_t *record) {
	int pos = find_ordered_transaction_id(map, transaction_id);
	if (pos == map->transaction_ids_count)
		return false;

	const turn_entry_t *entry = map->ordered_transaction_ids[pos];
	if (memcmp(entry->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE) != 0)
		return false;

	if (record)
		*record = entry->record;

	return true;
}

static bool set_random_transaction_id(turn_map_t *map, turn_entry_type_t type,
                                      const addr_record_t *record, uint8_t *transaction_id) {
	turn_entry_t *entry = find_entry(map, record, type, false);
	if (entry && entry->fresh_transaction_id) {
		if (transaction_id)
			memcpy(transaction_id, entry->transaction_id, STUN_TRANSACTION_ID_SIZE);

		return true;
	}

	uint8_t tid[STUN_TRANSACTION_ID_SIZE];
	do {
		juice_random(tid, STUN_TRANSACTION_ID_SIZE);
	} while (find_transaction_id(map, tid, NULL));

	if (!set_transaction_id(map, type, record, tid))
		return false;

	if (transaction_id)
		memcpy(transaction_id, tid, STUN_TRANSACTION_ID_SIZE);

	return true;
}

bool turn_set_permission_transaction_id(turn_map_t *map, const addr_record_t *record,
                                        const uint8_t *transaction_id) {
	return set_transaction_id(map, TURN_ENTRY_TYPE_PERMISSION, record, transaction_id);
}

bool turn_set_channel_transaction_id(turn_map_t *map, const addr_record_t *record,
                                     const uint8_t *transaction_id) {
	return set_transaction_id(map, TURN_ENTRY_TYPE_CHANNEL, record, transaction_id);
}

bool turn_set_random_permission_transaction_id(turn_map_t *map, const addr_record_t *record,
                                               uint8_t *transaction_id) {
	return set_random_transaction_id(map, TURN_ENTRY_TYPE_PERMISSION, record, transaction_id);
}

bool turn_set_random_channel_transaction_id(turn_map_t *map, const addr_record_t *record,
                                            uint8_t *transaction_id) {
	return set_random_transaction_id(map, TURN_ENTRY_TYPE_CHANNEL, record, transaction_id);
}

bool turn_retrieve_transaction_id(turn_map_t *map, const uint8_t *transaction_id,
                                  addr_record_t *record) {
	int pos = find_ordered_transaction_id(map, transaction_id);
	if (pos == map->transaction_ids_count)
		return false;

	turn_entry_t *entry = map->ordered_transaction_ids[pos];
	if (memcmp(entry->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE) != 0)
		return false;

	if (record)
		*record = entry->record;

	entry->fresh_transaction_id = false;
	return true;
}
