/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"
#include "core/project_settings.h"
#include "webrtc_data_channel.h"
#include "webrtc_peer_connection.h"

#ifdef JAVASCRIPT_ENABLED
#include "emscripten.h"
#include "webrtc_peer_connection_js.h"
#endif
#ifdef WEBRTC_GDNATIVE_ENABLED
#include "webrtc_data_channel_gdnative.h"
#include "webrtc_peer_connection_gdnative.h"
#endif
#include "webrtc_multiplayer.h"

void register_webrtc_types() {
#define _SET_HINT(NAME, _VAL_, _MAX_) \
	GLOBAL_DEF(NAME, _VAL_);          \
	ProjectSettings::get_singleton()->set_custom_property_info(NAME, PropertyInfo(Variant::INT, NAME, PROPERTY_HINT_RANGE, "2," #_MAX_ ",1,or_greater"));

	_SET_HINT(WRTC_IN_BUF, 64, 4096);

#ifdef JAVASCRIPT_ENABLED
	WebRTCPeerConnectionJS::make_default();
#elif defined(WEBRTC_GDNATIVE_ENABLED)
	WebRTCPeerConnectionGDNative::make_default();
#endif

	ClassDB::register_custom_instance_class<WebRTCPeerConnection>();
#ifdef WEBRTC_GDNATIVE_ENABLED
	ClassDB::register_class<WebRTCPeerConnectionGDNative>();
	ClassDB::register_class<WebRTCDataChannelGDNative>();
#endif
	ClassDB::register_virtual_class<WebRTCDataChannel>();
	ClassDB::register_class<WebRTCMultiplayer>();
}

void unregister_webrtc_types() {}
