/*************************************************************************/
/*  ogg_packet_sequence.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef OGG_PACKET_SEQUENCE_H
#define OGG_PACKET_SEQUENCE_H

#include "core/io/resource.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/variant/native_ptr.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
#include "thirdparty/libogg/ogg/ogg.h"

class OGGPacketSequencePlayback;

class OGGPacketSequence : public Resource {
	GDCLASS(OGGPacketSequence, Resource);

	friend class OGGPacketSequencePlayback;

	// List of pages, each of which is a list of packets on that page. The innermost PackedByteArrays contain complete ogg packets.
	Vector<Vector<PackedByteArray>> page_data;

	// List of the granule position for each page.
	Vector<uint64_t> page_granule_positions;

	// The page after the current last page. Similar semantics to an end() iterator.
	int64_t end_page = 0;

	uint64_t data_version = 0;

	float sampling_rate = 0;
	float length = 0;

protected:
	static void _bind_methods();

public:
	// Pushes information about all the pages that ended on this page.
	// This should be called for each page, even for pages that no packets ended on.
	void push_page(int64_t p_granule_pos, const Vector<PackedByteArray> &p_data);

	void set_packet_data(const Array &p_data);
	Array get_packet_data() const;

	void set_packet_granule_positions(const Array &p_granule_positions);
	Array get_packet_granule_positions() const;

	// Sets a sampling rate associated with this object. OGGPacketSequence doesn't understand codecs,
	// so this value is naively stored as a convenience.
	void set_sampling_rate(float p_sampling_rate);

	// Returns a sampling rate previously set by set_sampling_rate().
	float get_sampling_rate() const;

	// Returns a length previously set by set_length().
	float get_length() const;

	// Returns the granule position of the last page in this sequence.
	int64_t get_final_granule_pos() const;

	Ref<OGGPacketSequencePlayback> instance_playback();

	OGGPacketSequence() {}
	virtual ~OGGPacketSequence() {}
};

class OGGPacketSequencePlayback : public RefCounted {
	GDCLASS(OGGPacketSequencePlayback, RefCounted);

	friend class OGGPacketSequence;

	Ref<OGGPacketSequence> ogg_packet_sequence;

	mutable int64_t page_cursor = 0;
	mutable int32_t packet_cursor = 0;

	mutable ogg_packet *packet;

	uint64_t data_version;

	mutable int64_t packetno = 0;

	// Recursive bisection search for the correct page.
	uint32_t seek_page_internal(int64_t granule, uint32_t after_page_inclusive, uint32_t before_page_inclusive);

public:
	// Calling functions must not modify this packet.
	// Returns true on success, false on error or if there is no next packet.
	bool next_ogg_packet(ogg_packet **p_packet) const;

	// Seeks to the page such that the previous page has a granule position less than or equal to this value,
	// and the current page has a granule position greater than this value.
	// Returns true on success, false on failure.
	bool seek_page(int64_t p_granule_pos);

	OGGPacketSequencePlayback();
	virtual ~OGGPacketSequencePlayback();
};

#endif // OGG_PACKET_SEQUENCE_H
