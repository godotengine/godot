/**************************************************************************/
/*  snapshot.cpp                                                          */
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

#include "snapshot.h"

void Snapshot::set_snapshot(const Dictionary &p_snapshot) {
	snapshot = p_snapshot;
}

Dictionary Snapshot::get_snapshot() const {
	return snapshot;
}

void Snapshot::set_metadata(const Dictionary &p_metadata) {
	metadata = p_metadata;
}

Dictionary Snapshot::get_metadata() const {
	return metadata;
}

void Snapshot::set_tag_slots(const Dictionary &p_tag_slots) {
	tag_slots = p_tag_slots;
}

Dictionary Snapshot::get_tag_slots() const {
	return tag_slots;
}

void Snapshot::set_thumbnail(Ref<Resource> p_thumbnail) {
	thumbnail = p_thumbnail;
}

Ref<Resource> Snapshot::get_thumbnail() const {
	return thumbnail;
}

void Snapshot::set_version(const String &p_version) {
	version = p_version;
}

String Snapshot::get_version() const {
	return version;
}

void Snapshot::set_checksum(const String &p_checksum) {
	checksum = p_checksum;
}

String Snapshot::get_checksum() const {
	return checksum;
}

void Snapshot::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_snapshot", "snapshot"), &Snapshot::set_snapshot);
	ClassDB::bind_method(D_METHOD("get_snapshot"), &Snapshot::get_snapshot);

	ClassDB::bind_method(D_METHOD("set_metadata", "metadata"), &Snapshot::set_metadata);
	ClassDB::bind_method(D_METHOD("get_metadata"), &Snapshot::get_metadata);

	ClassDB::bind_method(D_METHOD("set_thumbnail", "thumbnail"), &Snapshot::set_thumbnail);
	ClassDB::bind_method(D_METHOD("get_thumbnail"), &Snapshot::get_thumbnail);

	ClassDB::bind_method(D_METHOD("set_version", "version"), &Snapshot::set_version);
	ClassDB::bind_method(D_METHOD("get_version"), &Snapshot::get_version);

	ClassDB::bind_method(D_METHOD("set_checksum", "checksum"), &Snapshot::set_checksum);
	ClassDB::bind_method(D_METHOD("get_checksum"), &Snapshot::get_checksum);

	ClassDB::bind_method(D_METHOD("set_tag_slots", "tag_slots"), &Snapshot::set_tag_slots);
	ClassDB::bind_method(D_METHOD("get_tag_slots"), &Snapshot::get_tag_slots);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "snapshot"), "set_snapshot", "get_snapshot");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "metadata"), "set_metadata", "get_metadata");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "tag_slots"), "set_tag_slots", "get_tag_slots");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "thumbnail", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_thumbnail", "get_thumbnail");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "version"), "set_version", "get_version");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "checksum"), "set_checksum", "get_checksum");
}

Snapshot::Snapshot() {
}
