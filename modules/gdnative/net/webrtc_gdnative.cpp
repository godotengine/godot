/**************************************************************************/
/*  webrtc_gdnative.cpp                                                   */
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

#include "modules/gdnative/gdnative.h"
#include "modules/gdnative/include/net/godot_net.h"

#ifdef WEBRTC_GDNATIVE_ENABLED
#include "modules/webrtc/webrtc_data_channel_gdnative.h"
#include "modules/webrtc/webrtc_peer_connection_gdnative.h"
#endif

extern "C" {

void GDAPI godot_net_bind_webrtc_peer_connection(godot_object *p_obj, const godot_net_webrtc_peer_connection *p_impl) {
#ifdef WEBRTC_GDNATIVE_ENABLED
	((WebRTCPeerConnectionGDNative *)p_obj)->set_native_webrtc_peer_connection(p_impl);
#endif
}

void GDAPI godot_net_bind_webrtc_data_channel(godot_object *p_obj, const godot_net_webrtc_data_channel *p_impl) {
#ifdef WEBRTC_GDNATIVE_ENABLED
	((WebRTCDataChannelGDNative *)p_obj)->set_native_webrtc_data_channel(p_impl);
#endif
}

godot_error GDAPI godot_net_set_webrtc_library(const godot_net_webrtc_library *p_lib) {
#ifdef WEBRTC_GDNATIVE_ENABLED
	return (godot_error)WebRTCPeerConnectionGDNative::set_default_library(p_lib);
#else
	return (godot_error)ERR_UNAVAILABLE;
#endif
}
}
