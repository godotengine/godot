/**************************************************************************/
/*  webrtc_peer_connection_extension.h                                    */
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

#pragma once

#include "webrtc_peer_connection.h"

#include "core/extension/ext_wrappers.gen.inc"
#include "core/object/gdvirtual.gen.inc"

class WebRTCPeerConnectionExtension : public WebRTCPeerConnection {
	GDCLASS(WebRTCPeerConnectionExtension, WebRTCPeerConnection);

protected:
	static void _bind_methods();

public:
	/** GDExtension **/
	EXBIND0RC(ConnectionState, get_connection_state);
	EXBIND0RC(GatheringState, get_gathering_state);
	EXBIND0RC(SignalingState, get_signaling_state);

	class _GDVIRTUAL_initialize : public GDVirtualMethodInfo<false, Error, const Dictionary &> {
	public:
		MethodInfo get_method_info() const override {
			MethodInfo method_info;
			method_info.name = "_initialize";
			method_info.flags = METHOD_FLAG_VIRTUAL | METHOD_FLAG_VIRTUAL_REQUIRED;
			method_info.return_val = GetTypeInfo<Error>::get_class_info();
			method_info.return_val_metadata = GetTypeInfo<Error>::METADATA;
			_gdvirtual_set_method_info_args<const Dictionary &>(method_info);
			return method_info;
		}

		using GDVirtualMethodInfo::GDVirtualMethodInfo;
	};

	static const _GDVIRTUAL_initialize &get_gdvirtual__initialize() {
		static _GDVIRTUAL_initialize instance("_initialize", true, false);
		return instance;
	}

	mutable void *_gdvirtual__initialize_ptr = nullptr;

	virtual Error initialize(const Dictionary &arg1) override {
		Error ret;
		ZeroInitializer<Error>::initialize(ret);
		gdvirtual_call(get_gdvirtual__initialize(), _gdvirtual__initialize_ptr, arg1, ret);
		return ret;
	}
	EXBIND2R(Ref<WebRTCDataChannel>, create_data_channel, const String &, const Dictionary &);
	EXBIND0R(Error, create_offer);
	EXBIND2R(Error, set_remote_description, const String &, const String &);
	EXBIND2R(Error, set_local_description, const String &, const String &);
	EXBIND3R(Error, add_ice_candidate, const String &, int, const String &);
	EXBIND0R(Error, poll);
	EXBIND0(close);

	WebRTCPeerConnectionExtension() {}
};
