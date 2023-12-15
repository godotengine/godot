/**************************************************************************/
/*  ogg_packet_sequence.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "ogg_packet_sequence.h"

#include "core/variant/typed_array.h"

void OggPacketSequence::push_page(int64_t p_granule_pos, const Vector<PackedByteArray> &p_data) {
	Vector<PackedByteArray> data_stored;
	for (int i = 0; i < p_data.size(); i++) {
		data_stored.push_back(p_data[i]);
	}
	page_granule_positions.push_back(p_granule_pos);
	page_data.push_back(data_stored);
	data_version++;
}

void OggPacketSequence::set_packet_data(const TypedArray<Array> &p_data) {
	data_version++; // Update the data version so old playbacks know that they can't rely on us anymore.
	page_data.clear();
	for (int page_idx = 0; page_idx < p_data.size(); page_idx++) {
		// Push a new page. We cleared the vector so this will be at index `page_idx`.
		page_data.push_back(Vector<PackedByteArray>());
		TypedArray<PackedByteArray> this_page_data = p_data[page_idx];
		for (int packet = 0; packet < this_page_data.size(); packet++) {
			page_data.write[page_idx].push_back(this_page_data[packet]);
		}
	}
}

TypedArray<Array> OggPacketSequence::get_packet_data() const {
	TypedArray<Array> ret;
	for (const Vector<PackedByteArray> &page : page_data) {
		Array page_variant;
		for (const PackedByteArray &packet : page) {
			page_variant.push_back(packet);
		}
		ret.push_back(page_variant);
	}
	return ret;
}

void OggPacketSequence::set_packet_granule_positions(const PackedInt64Array &p_granule_positions) {
	data_version++; // Update the data version so old playbacks know that they can't rely on us anymore.
	page_granule_positions.clear();
	for (int page_idx = 0; page_idx < p_granule_positions.size(); page_idx++) {
		int64_t granule_pos = p_granule_positions[page_idx];
		page_granule_positions.push_back(granule_pos);
	}
}

PackedInt64Array OggPacketSequence::get_packet_granule_positions() const {
	PackedInt64Array ret;
	for (int64_t granule_pos : page_granule_positions) {
		ret.push_back(granule_pos);
	}
	return ret;
}

void OggPacketSequence::set_sampling_rate(float p_sampling_rate) {
	sampling_rate = p_sampling_rate;
}

float OggPacketSequence::get_sampling_rate() const {
	return sampling_rate;
}

int64_t OggPacketSequence::get_final_granule_pos() const {
	if (!page_granule_positions.is_empty()) {
		return page_granule_positions[page_granule_positions.size() - 1];
	}
	return -1;
}

float OggPacketSequence::get_length() const {
	int64_t granule_pos = get_final_granule_pos();
	if (granule_pos < 0) {
		return 0;
	}
	return granule_pos / sampling_rate;
}

Ref<OggPacketSequencePlayback> OggPacketSequence::instantiate_playback() {
	Ref<OggPacketSequencePlayback> playback;
	playback.instantiate();
	playback->ogg_packet_sequence = Ref<OggPacketSequence>(this);
	playback->data_version = data_version;

	return playback;
}

void OggPacketSequence::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_packet_data", "packet_data"), &OggPacketSequence::set_packet_data);
	ClassDB::bind_method(D_METHOD("get_packet_data"), &OggPacketSequence::get_packet_data);

	ClassDB::bind_method(D_METHOD("set_packet_granule_positions", "granule_positions"), &OggPacketSequence::set_packet_granule_positions);
	ClassDB::bind_method(D_METHOD("get_packet_granule_positions"), &OggPacketSequence::get_packet_granule_positions);

	ClassDB::bind_method(D_METHOD("set_sampling_rate", "sampling_rate"), &OggPacketSequence::set_sampling_rate);
	ClassDB::bind_method(D_METHOD("get_sampling_rate"), &OggPacketSequence::get_sampling_rate);

	ClassDB::bind_method(D_METHOD("get_length"), &OggPacketSequence::get_length);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "packet_data", PROPERTY_HINT_ARRAY_TYPE, "PackedByteArray", PROPERTY_USAGE_NO_EDITOR), "set_packet_data", "get_packet_data");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT64_ARRAY, "granule_positions", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_packet_granule_positions", "get_packet_granule_positions");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sampling_rate", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_sampling_rate", "get_sampling_rate");
}

bool OggPacketSequencePlayback::next_ogg_packet(ogg_packet **p_packet) const {
	ERR_FAIL_COND_V(data_version != ogg_packet_sequence->data_version, false);
	ERR_FAIL_COND_V(ogg_packet_sequence->page_data.is_empty(), false);
	ERR_FAIL_COND_V(ogg_packet_sequence->page_granule_positions.is_empty(), false);
	ERR_FAIL_COND_V(page_cursor >= ogg_packet_sequence->page_data.size(), false);

	// Move on to the next page if need be. This happens first to help simplify seek logic.
	while (packet_cursor >= ogg_packet_sequence->page_data[page_cursor].size()) {
		packet_cursor = 0;
		page_cursor++;
		if (page_cursor >= ogg_packet_sequence->page_data.size()) {
			return false;
		}
	}

	ERR_FAIL_COND_V(page_cursor >= ogg_packet_sequence->page_data.size(), false);

	packet->b_o_s = page_cursor == 0 && packet_cursor == 0;
	packet->e_o_s = page_cursor == ogg_packet_sequence->page_data.size() - 1 && packet_cursor == ogg_packet_sequence->page_data[page_cursor].size() - 1;
	packet->granulepos = packet_cursor == ogg_packet_sequence->page_data[page_cursor].size() - 1 ? ogg_packet_sequence->page_granule_positions[page_cursor] : -1;
	packet->packetno = packetno++;
	packet->bytes = ogg_packet_sequence->page_data[page_cursor][packet_cursor].size();
	packet->packet = (unsigned char *)(ogg_packet_sequence->page_data[page_cursor][packet_cursor].ptr());

	*p_packet = packet;

	packet_cursor++;

	return true;
}

uint32_t OggPacketSequencePlayback::seek_page_internal(int64_t granule, uint32_t after_page_inclusive, uint32_t before_page_inclusive) {
	// FIXME: This function needs better corner case handling.
	if (before_page_inclusive == after_page_inclusive) {
		return before_page_inclusive;
	}
	uint32_t actual_middle_page = after_page_inclusive + (before_page_inclusive - after_page_inclusive) / 2;
	// Complicating the bisection search algorithm, the middle page might not have a packet that ends on it,
	// which means it might not have a correct granule position. Find a nearby page that does have a packet ending on it.
	uint32_t bisection_page = -1;
	// Don't include before_page_inclusive because that always succeeds and will cause infinite recursion later.
	for (uint32_t test_page = actual_middle_page; test_page < before_page_inclusive; test_page++) {
		if (ogg_packet_sequence->page_data[test_page].size() > 0) {
			bisection_page = test_page;
			break;
		}
	}
	// Check if we have to go backwards.
	if (bisection_page == (unsigned int)-1) {
		for (uint32_t test_page = actual_middle_page; test_page >= after_page_inclusive; test_page--) {
			if (ogg_packet_sequence->page_data[test_page].size() > 0) {
				bisection_page = test_page;
				break;
			}
		}
	}
	if (bisection_page == (unsigned int)-1) {
		return -1;
	}

	int64_t bisection_granule_pos = ogg_packet_sequence->page_granule_positions[bisection_page];
	if (granule > bisection_granule_pos) {
		return seek_page_internal(granule, bisection_page + 1, before_page_inclusive);
	} else {
		return seek_page_internal(granule, after_page_inclusive, bisection_page);
	}
}

bool OggPacketSequencePlayback::seek_page(int64_t p_granule_pos) {
	int correct_page = seek_page_internal(p_granule_pos, 0, ogg_packet_sequence->page_data.size() - 1);
	if (correct_page == -1) {
		return false;
	}

	packet_cursor = 0;
	page_cursor = correct_page;

	// Don't pretend subsequent packets are contiguous with previous ones.
	packetno = 0;

	return true;
}

int64_t OggPacketSequencePlayback::get_page_number() const {
	return page_cursor;
}

bool OggPacketSequencePlayback::set_page_number(int64_t p_page_number) {
	if (p_page_number >= 0 && p_page_number < ogg_packet_sequence->page_data.size()) {
		page_cursor = p_page_number;
		packet_cursor = 0;
		packetno = 0;
		return true;
	}
	return false;
}

OggPacketSequencePlayback::OggPacketSequencePlayback() {
	packet = new ogg_packet();
}

OggPacketSequencePlayback::~OggPacketSequencePlayback() {
	delete packet;
}
