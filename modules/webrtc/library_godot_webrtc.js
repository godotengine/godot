/*************************************************************************/
/*  library_godot_webrtc.js                                              */
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

var GodotRTCDataChannel = {
	// Our socket implementation that forwards events to C++.
	$GodotRTCDataChannel__deps: ['$IDHandler', '$GodotOS'],
	$GodotRTCDataChannel: {

		connect: function(p_id, p_on_open, p_on_message, p_on_error, p_on_close) {
			const ref = IDHandler.get(p_id);
			if (!ref) {
				return;
			}

			ref.binaryType = 'arraybuffer';
			ref.onopen = function (event) {
				p_on_open();
			};
			ref.onclose = function (event) {
				p_on_close();
			};
			ref.onerror = function (event) {
				p_on_error();
			};
			ref.onmessage = function(event) {
				var buffer;
				var is_string = 0;
				if (event.data instanceof ArrayBuffer) {
					buffer = new Uint8Array(event.data);
				} else if (event.data instanceof Blob) {
					console.error("Blob type not supported");
					return;
				} else if (typeof event.data === "string") {
					is_string = 1;
					var enc = new TextEncoder("utf-8");
					buffer = new Uint8Array(enc.encode(event.data));
				} else {
					console.error("Unknown message type");
					return;
				}
				var len = buffer.length*buffer.BYTES_PER_ELEMENT;
				var out = _malloc(len);
				HEAPU8.set(buffer, out);
				p_on_message(out, len, is_string);
				_free(out);
			}
		},

		close: function(p_id) {
			const ref = IDHandler.get(p_id);
			if (!ref) {
				return;
			}
			ref.onopen = null;
			ref.onmessage = null;
			ref.onerror = null;
			ref.onclose = null;
			ref.close();
		},

		get_prop: function(p_id, p_prop, p_def) {
			const ref = IDHandler.get(p_id);
			return (ref && ref[p_prop] !== undefined) ? ref[p_prop] : p_def;
		},
	},

	godot_js_rtc_datachannel_ready_state_get: function(p_id) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return 3; // CLOSED
		}

		switch(ref.readyState) {
			case "connecting":
				return 0;
			case "open":
				return 1;
			case "closing":
				return 2;
			case "closed":
				return 3;
		}
		return 3; // CLOSED
	},

	godot_js_rtc_datachannel_send: function(p_id, p_buffer, p_length, p_raw) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return 1;
		}

		const bytes_array = new Uint8Array(p_length);
		for (var i = 0; i < p_length; i++) {
			bytes_array[i] = getValue(p_buffer + i, 'i8');
		}

		if (p_raw) {
			ref.send(bytes_array.buffer);
		} else {
			const string = new TextDecoder('utf-8').decode(bytes_array);
			ref.send(string);
		}
	},

	godot_js_rtc_datachannel_is_ordered: function(p_id) {
		return IDHandler.get_prop(p_id, 'ordered', true);
	},

	godot_js_rtc_datachannel_id_get: function(p_id) {
		return IDHandler.get_prop(p_id, 'id', 65535);
	},

	godot_js_rtc_datachannel_max_packet_lifetime_get: function(p_id) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return 65535;
		}
		if (ref['maxPacketLifeTime'] !== undefined) {
			return ref['maxPacketLifeTime'];
		} else if (ref['maxRetransmitTime'] !== undefined) {
			// Guess someone didn't appreciate the standardization process.
			return ref['maxRetransmitTime'];
		}
		return 65535;
	},

	godot_js_rtc_datachannel_max_retransmits_get: function(p_id) {
		return IDHandler.get_prop(p_id, 'maxRetransmits', 65535);
	},

	godot_js_rtc_datachannel_is_negotiated: function(p_id, p_def) {
		return IDHandler.get_prop(p_id, 'negotiated', 65535);
	},

	godot_js_rtc_datachannel_label_get: function(p_id) {
		const ref = IDHandler.get(p_id);
		if (!ref || !ref.label) {
			return 0;
		}
		return GodotOS.allocString(ref.label);
	},

	godot_js_rtc_datachannel_protocol_get: function(p_id) {
		const ref = IDHandler.get(p_id);
		if (!ref || !ref.protocol) {
			return 0;
		}
		return GodotOS.allocString(ref.protocol);
	},

	godot_js_rtc_datachannel_destroy: function(p_id) {
		GodotRTCDataChannel.close(p_id);
		IDHandler.remove(p_id);
	},

	godot_js_rtc_datachannel_connect: function(p_id, p_ref, p_on_open, p_on_message, p_on_error, p_on_close) {
		const onopen = GodotOS.get_func(p_on_open).bind(null, p_ref);
		const onmessage = GodotOS.get_func(p_on_message).bind(null, p_ref);
		const onerror = GodotOS.get_func(p_on_error).bind(null, p_ref);
		const onclose = GodotOS.get_func(p_on_close).bind(null, p_ref);
		GodotRTCDataChannel.connect(p_id, onopen, onmessage, onerror, onclose);
	},

	godot_js_rtc_datachannel_close: function(p_id) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return;
		}
		GodotRTCDataChannel.close(p_id);
	},
};

autoAddDeps(GodotRTCDataChannel, '$GodotRTCDataChannel');
mergeInto(LibraryManager.library, GodotRTCDataChannel);

var GodotRTCPeerConnection = {

	$GodotRTCPeerConnection__deps: ['$IDHandler', '$GodotOS', '$GodotRTCDataChannel'],
	$GodotRTCPeerConnection: {
		onstatechange: function(p_id, p_conn, callback, event) {
			const ref = IDHandler.get(p_id);
			if (!ref) {
				return;
			}
			var state = 5; // CLOSED
			switch(p_conn.iceConnectionState) {
				case "new":
					state = 0;
				case "checking":
					state = 1;
				case "connected":
				case "completed":
					state = 2;
				case "disconnected":
					state = 3;
				case "failed":
					state = 4;
				case "closed":
					state = 5;
			}
			callback(state);
		},

		onicecandidate: function(p_id, callback, event) {
			const ref = IDHandler.get(p_id);
			if (!ref || !event.candidate) {
				return;
			}

			let c = event.candidate;
			let candidate_str = GodotOS.allocString(c.candidate);
			let mid_str = GodotOS.allocString(c.sdpMid);
			callback(mid_str, c.sdpMLineIndex, candidate_str);
			_free(candidate_str);
			_free(mid_str);
		},

		ondatachannel: function(p_id, callback, event) {
			const ref = IDHandler.get(p_id);
			if (!ref) {
				return;
			}

			const cid = IDHandler.add(event.channel);
			callback(cid);
		},

		onsession: function(p_id, callback, session) {
			const ref = IDHandler.get(p_id);
			if (!ref) {
				return;
			}
			let type_str = GodotOS.allocString(session.type);
			let sdp_str = GodotOS.allocString(session.sdp);
			callback(type_str, sdp_str);
			_free(type_str);
			_free(sdp_str);
		},

		onerror: function(p_id, callback, error) {
			const ref = IDHandler.get(p_id);
			if (!ref) {
				return;
			}
			console.error(error);
			callback();
		},
	},

	godot_js_rtc_pc_create: function(p_config, p_ref, p_on_state_change, p_on_candidate, p_on_datachannel) {
		const onstatechange = GodotOS.get_func(p_on_state_change).bind(null, p_ref);
		const oncandidate = GodotOS.get_func(p_on_candidate).bind(null, p_ref);
		const ondatachannel = GodotOS.get_func(p_on_datachannel).bind(null, p_ref);

		var config = JSON.parse(UTF8ToString(p_config));
		var conn = null;
		try {
			conn = new RTCPeerConnection(config);
		} catch (e) {
			console.error(e);
			return 0;
		}

		const base = GodotRTCPeerConnection;
		const id = IDHandler.add(conn);
		conn.oniceconnectionstatechange = base.onstatechange.bind(null, id, conn, onstatechange);
		conn.onicecandidate = base.onicecandidate.bind(null, id, oncandidate);
		conn.ondatachannel = base.ondatachannel.bind(null, id, ondatachannel);
		return id;
	},

	godot_js_rtc_pc_close: function(p_id) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return;
		}
		ref.close();
	},

	godot_js_rtc_pc_destroy: function(p_id) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return;
		}
		ref.oniceconnectionstatechange = null;
		ref.onicecandidate = null;
		ref.ondatachannel = null;
		IDHandler.remove(p_id);
	},

	godot_js_rtc_pc_offer_create: function(p_id, p_obj, p_on_session, p_on_error) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return;
		}
		const onsession = GodotOS.get_func(p_on_session).bind(null, p_obj);
		const onerror = GodotOS.get_func(p_on_error).bind(null, p_obj);
		ref.createOffer().then(function(session) {
			GodotRTCPeerConnection.onsession(p_id, onsession, session);
		}).catch(function(error) {
			GodotRTCPeerConnection.onerror(p_id, onerror, error);
		});
	},

	godot_js_rtc_pc_local_description_set: function(p_id, p_type, p_sdp, p_obj, p_on_error) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return;
		}
		const type = UTF8ToString(p_type);
		const sdp = UTF8ToString(p_sdp);
		const onerror = GodotOS.get_func(p_on_error).bind(null, p_obj);
		ref.setLocalDescription({
			'sdp': sdp,
			'type': type
		}).catch(function(error) {
			GodotRTCPeerConnection.onerror(p_id, onerror, error);
		});
	},

	godot_js_rtc_pc_remote_description_set: function(p_id, p_type, p_sdp, p_obj, p_session_created, p_on_error) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return;
		}
		const type = UTF8ToString(p_type);
		const sdp = UTF8ToString(p_sdp);
		const onerror = GodotOS.get_func(p_on_error).bind(null, p_obj);
		const onsession = GodotOS.get_func(p_session_created).bind(null, p_obj);
		ref.setRemoteDescription({
			'sdp': sdp,
			'type': type
		}).then(function() {
			if (type != 'offer') {
				return;
			}
			return ref.createAnswer().then(function(session) {
				GodotRTCPeerConnection.onsession(p_id, onsession, session);
			});
		}).catch(function(error) {
			GodotRTCPeerConnection.onerror(p_id, onerror, error);
		});
	},

	godot_js_rtc_pc_ice_candidate_add: function(p_id, p_mid_name, p_mline_idx, p_sdp) {
		const ref = IDHandler.get(p_id);
		if (!ref) {
			return;
		}
		var sdpMidName = UTF8ToString(p_mid_name);
		var sdpName = UTF8ToString(p_sdp);
		ref.addIceCandidate(new RTCIceCandidate({
			"candidate": sdpName,
			"sdpMid": sdpMidName,
			"sdpMlineIndex": p_mline_idx,
		}));
	},

	godot_js_rtc_pc_datachannel_create__deps: ['$GodotRTCDataChannel'],
	godot_js_rtc_pc_datachannel_create: function(p_id, p_label, p_config) {
		try {
			const ref = IDHandler.get(p_id);
			if (!ref) {
				return 0;
			}

			const label = UTF8ToString(p_label);
			const config = JSON.parse(UTF8ToString(p_config));

			const channel = ref.createDataChannel(label, config);
			return IDHandler.add(channel);
		} catch (e) {
			console.error(e);
			return 0;
		}
	},
};

autoAddDeps(GodotRTCPeerConnection, '$GodotRTCPeerConnection')
mergeInto(LibraryManager.library, GodotRTCPeerConnection);
