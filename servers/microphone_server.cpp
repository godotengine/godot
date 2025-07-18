/**************************************************************************/
/*  microphone_server.cpp                                                 */
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

#include "microphone_server.h"
#include "core/variant/typed_array.h"
#include "rendering_server.h"
#include "servers/microphone/microphone_feed.h"

////////////////////////////////////////////////////////
// MicrophoneServer

MicrophoneServer::CreateFunc MicrophoneServer::create_func = nullptr;

void MicrophoneServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_feed", "index"), &MicrophoneServer::get_feed);
	ClassDB::bind_method(D_METHOD("get_feed_count"), &MicrophoneServer::get_feed_count);
}

MicrophoneServer *MicrophoneServer::singleton = nullptr;

MicrophoneServer *MicrophoneServer::get_singleton() {
	return singleton;
}

Ref<MicrophoneFeed> MicrophoneServer::get_feed(int p_index) {
	ERR_FAIL_INDEX_V(p_index, 1, nullptr);

	return default_feed;
}

int MicrophoneServer::get_feed_count() {
	return 1;
}

MicrophoneServer::MicrophoneServer() {
	default_feed = memnew(MicrophoneFeed("Default"));
	singleton = this;
}

MicrophoneServer::~MicrophoneServer() {
	singleton = nullptr;
}
