/*************************************************************************/
/*  net_utilities.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

/**
	@author AndreaCatania
*/

#include "net_utilities.h"

NetworkTracer::NetworkTracer(int p_packets_to_track) :
		id(0) {
	reset(p_packets_to_track);
}

void NetworkTracer::reset(int p_packets_to_track) {
	id = 0;

	if (p_packets_to_track >= 0) {
		flags.resize(p_packets_to_track);
	}

	// Let's pretend that the connection is good.
	for (size_t i = 0; i < flags.size(); i += 1) {
		flags[i] = true;
	}
}

void NetworkTracer::notify_packet_arrived() {
	if (flags.size() == 0)
		return;
	id = (id + 1) % flags.size();
	flags[id] = true;
}

void NetworkTracer::notify_missing_packet() {
	if (flags.size() == 0)
		return;
	id = (id + 1) % flags.size();
	flags[id] = false;
}

int NetworkTracer::get_missing_packets() const {
	int l = 0;
	for (size_t i = 0; i < flags.size(); i += 1) {
		if (flags[i] == false) {
			l += 1;
		}
	}
	return l;
}
