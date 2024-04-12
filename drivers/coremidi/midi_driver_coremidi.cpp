/**************************************************************************/
/*  midi_driver_coremidi.cpp                                              */
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

#include "midi_driver_coremidi.h"

#ifdef COREMIDI_ENABLED

#include "core/string/print_string.h"

#import <CoreAudio/HostTime.h>
#import <CoreServices/CoreServices.h>

void MIDIDriverCoreMidi::read(const MIDIPacketList *packet_list, void *read_proc_ref_con, void *src_conn_ref_con) {
	MIDIPacket *packet = const_cast<MIDIPacket *>(packet_list->packet);
	int *device_index = static_cast<int *>(src_conn_ref_con);
	for (UInt32 i = 0; i < packet_list->numPackets; i++) {
		receive_input_packet(*device_index, packet->timeStamp, packet->data, packet->length);
		packet = MIDIPacketNext(packet);
	}
}

Error MIDIDriverCoreMidi::open() {
	CFStringRef name = CFStringCreateWithCString(nullptr, "Godot", kCFStringEncodingASCII);
	OSStatus result = MIDIClientCreate(name, nullptr, nullptr, &client);
	CFRelease(name);
	if (result != noErr) {
		ERR_PRINT("MIDIClientCreate failed, code: " + itos(result));
		return ERR_CANT_OPEN;
	}

	result = MIDIInputPortCreate(client, CFSTR("Godot Input"), MIDIDriverCoreMidi::read, (void *)this, &port_in);
	if (result != noErr) {
		ERR_PRINT("MIDIInputPortCreate failed, code: " + itos(result));
		return ERR_CANT_OPEN;
	}

	int sources = MIDIGetNumberOfSources();
	for (int i = 0; i < sources; i++) {
		MIDIEndpointRef source = MIDIGetSource(i);
		if (source) {
			MIDIPortConnectSource(port_in, source, static_cast<void *>(&i));
			connected_sources.insert(i, source);
		}
	}

	return OK;
}

void MIDIDriverCoreMidi::close() {
	for (int i = 0; i < connected_sources.size(); i++) {
		MIDIEndpointRef source = connected_sources[i];
		MIDIPortDisconnectSource(port_in, source);
	}
	connected_sources.clear();

	if (port_in != 0) {
		MIDIPortDispose(port_in);
		port_in = 0;
	}

	if (client != 0) {
		MIDIClientDispose(client);
		client = 0;
	}
}

PackedStringArray MIDIDriverCoreMidi::get_connected_inputs() {
	PackedStringArray list;

	for (int i = 0; i < connected_sources.size(); i++) {
		MIDIEndpointRef source = connected_sources[i];
		CFStringRef ref = nullptr;
		char name[256];

		MIDIObjectGetStringProperty(source, kMIDIPropertyDisplayName, &ref);
		CFStringGetCString(ref, name, sizeof(name), kCFStringEncodingUTF8);
		CFRelease(ref);

		list.push_back(name);
	}

	return list;
}

MIDIDriverCoreMidi::MIDIDriverCoreMidi() {}

MIDIDriverCoreMidi::~MIDIDriverCoreMidi() {
	close();
}

#endif // COREMIDI_ENABLED
