/*************************************************************************/
/*  webrtc_peer_connection_gdnative.h                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef WEBRTC_PEER_CONNECTION_GDNATIVE_H
#define WEBRTC_PEER_CONNECTION_GDNATIVE_H

#ifdef WEBRTC_GDNATIVE_ENABLED

#include "modules/gdnative/include/net/godot_net.h"
#include "webrtc_peer_connection.h"

class WebRTCPeerConnectionGDNative : public WebRTCPeerConnection {
	GDCLASS(WebRTCPeerConnectionGDNative, WebRTCPeerConnection);

protected:
	static void _bind_methods();
	static WebRTCPeerConnection *_create();

private:
	static const godot_net_webrtc_library *default_library;
	const godot_net_webrtc_peer_connection *interface;

public:
	static Error set_default_library(const godot_net_webrtc_library *p_library);
	static void make_default() { WebRTCPeerConnection::_create = WebRTCPeerConnectionGDNative::_create; }

	void set_native_webrtc_peer_connection(const godot_net_webrtc_peer_connection *p_impl);

	virtual ConnectionState get_connection_state() const override;

	virtual Error initialize(Dictionary p_config = Dictionary()) override;
	virtual Ref<WebRTCDataChannel> create_data_channel(String p_label, Dictionary p_options = Dictionary()) override;
	virtual Error create_offer() override;
	virtual Error set_remote_description(String type, String sdp) override;
	virtual Error set_local_description(String type, String sdp) override;
	virtual Error add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) override;
	virtual Error poll() override;
	virtual void close() override;

	WebRTCPeerConnectionGDNative();
	~WebRTCPeerConnectionGDNative();
};

#endif // WEBRTC_GDNATIVE_ENABLED

#endif // WEBRTC_PEER_CONNECTION_GDNATIVE_H
