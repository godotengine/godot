/*************************************************************************/
/*  webrtc_peer_connection_extension.h                                   */
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

#ifndef WEBRTC_PEER_CONNECTION_EXTENSION_H
#define WEBRTC_PEER_CONNECTION_EXTENSION_H

#include "webrtc_peer_connection.h"

#include "core/object/gdvirtual.gen.inc"
#include "core/object/script_language.h"
#include "core/variant/native_ptr.h"

class WebRTCPeerConnectionExtension : public WebRTCPeerConnection {
	GDCLASS(WebRTCPeerConnectionExtension, WebRTCPeerConnection);

protected:
	static void _bind_methods();

public:
	void make_default();

	virtual ConnectionState get_connection_state() const override;

	virtual Error initialize(Dictionary p_config = Dictionary()) override;
	virtual Ref<WebRTCDataChannel> create_data_channel(String p_label, Dictionary p_options = Dictionary()) override;
	virtual Error create_offer() override;
	virtual Error set_remote_description(String type, String sdp) override;
	virtual Error set_local_description(String type, String sdp) override;
	virtual Error add_ice_candidate(String p_sdp_mid_name, int p_sdp_mline_index, String p_sdp_name) override;
	virtual Error poll() override;
	virtual void close() override;

	/** GDExtension **/
	GDVIRTUAL0RC(int, _get_connection_state);
	GDVIRTUAL1R(int, _initialize, Dictionary);
	GDVIRTUAL2R(Object *, _create_data_channel, String, Dictionary);
	GDVIRTUAL0R(int, _create_offer);
	GDVIRTUAL2R(int, _set_remote_description, String, String);
	GDVIRTUAL2R(int, _set_local_description, String, String);
	GDVIRTUAL3R(int, _add_ice_candidate, String, int, String);
	GDVIRTUAL0R(int, _poll);
	GDVIRTUAL0(_close);

	WebRTCPeerConnectionExtension() {}
};

#endif // WEBRTC_PEER_CONNECTION_EXTENSION_H
