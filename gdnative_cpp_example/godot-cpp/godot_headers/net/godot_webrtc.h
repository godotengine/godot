/*************************************************************************/
/*  godot_webrtc.h                                                       */
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

#ifndef GODOT_NATIVEWEBRTC_H
#define GODOT_NATIVEWEBRTC_H

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GODOT_NET_WEBRTC_API_MAJOR 3
#define GODOT_NET_WEBRTC_API_MINOR 2

/* Library Interface (used to set default GDNative WebRTC implementation */
typedef struct {
	godot_gdnative_api_version version; /* version of our API */

	/* Called when the library is unset as default interface via godot_net_set_webrtc_library */
	void (*unregistered)();

	/* Used by WebRTCPeerConnection create when GDNative is the default implementation. */
	/* Takes a pointer to WebRTCPeerConnectionGDNative, should bind and return OK, failure if binding was unsuccessful. */
	godot_error (*create_peer_connection)(godot_object *);

	void *next; /* For extension */
} godot_net_webrtc_library;

/* WebRTCPeerConnection interface */
typedef struct {
	godot_gdnative_api_version version; /* version of our API */

	godot_object *data; /* User reference */

	/* This is WebRTCPeerConnection */
	godot_int (*get_connection_state)(const void *);

	godot_error (*initialize)(void *, const godot_dictionary *);
	godot_object *(*create_data_channel)(void *, const char *p_channel_name, const godot_dictionary *);
	godot_error (*create_offer)(void *);
	godot_error (*create_answer)(void *); /* unused for now, should be done automatically on set_local_description */
	godot_error (*set_remote_description)(void *, const char *, const char *);
	godot_error (*set_local_description)(void *, const char *, const char *);
	godot_error (*add_ice_candidate)(void *, const char *, int, const char *);
	godot_error (*poll)(void *);
	void (*close)(void *);

	void *next; /* For extension? */
} godot_net_webrtc_peer_connection;

/* WebRTCDataChannel interface */
typedef struct {
	godot_gdnative_api_version version; /* version of our API */

	godot_object *data; /* User reference */

	/* This is PacketPeer */
	godot_error (*get_packet)(void *, const uint8_t **, int *);
	godot_error (*put_packet)(void *, const uint8_t *, int);
	godot_int (*get_available_packet_count)(const void *);
	godot_int (*get_max_packet_size)(const void *);

	/* This is WebRTCDataChannel */
	void (*set_write_mode)(void *, godot_int);
	godot_int (*get_write_mode)(const void *);
	bool (*was_string_packet)(const void *);

	godot_int (*get_ready_state)(const void *);
	const char *(*get_label)(const void *);
	bool (*is_ordered)(const void *);
	int (*get_id)(const void *);
	int (*get_max_packet_life_time)(const void *);
	int (*get_max_retransmits)(const void *);
	const char *(*get_protocol)(const void *);
	bool (*is_negotiated)(const void *);

	godot_error (*poll)(void *);
	void (*close)(void *);

	void *next; /* For extension? */
} godot_net_webrtc_data_channel;

/* Set the default GDNative library */
godot_error GDAPI godot_net_set_webrtc_library(const godot_net_webrtc_library *);
/* Binds a WebRTCPeerConnectionGDNative to the provided interface */
void GDAPI godot_net_bind_webrtc_peer_connection(godot_object *p_obj, const godot_net_webrtc_peer_connection *);
/* Binds a WebRTCDataChannelGDNative to the provided interface */
void GDAPI godot_net_bind_webrtc_data_channel(godot_object *p_obj, const godot_net_webrtc_data_channel *);

#ifdef __cplusplus
}
#endif

#endif
