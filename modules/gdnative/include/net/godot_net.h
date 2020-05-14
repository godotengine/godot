/*************************************************************************/
/*  godot_net.h                                                          */
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

#ifndef GODOT_NATIVENET_H
#define GODOT_NATIVENET_H

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

// For future versions of the API we should only add new functions at the end of the structure and use the
// version info to detect whether a call is available

// Use these to populate version in your plugin
#define GODOT_NET_API_MAJOR 3
#define GODOT_NET_API_MINOR 1

typedef struct {
	godot_gdnative_api_version version; /* version of our API */
	godot_object *data; /* User reference */

	/* This is StreamPeer */
	godot_error (*get_data)(void *user, uint8_t *p_buffer, int p_bytes);
	godot_error (*get_partial_data)(void *user, uint8_t *p_buffer, int p_bytes, int *r_received);
	godot_error (*put_data)(void *user, const uint8_t *p_data, int p_bytes);
	godot_error (*put_partial_data)(void *user, const uint8_t *p_data, int p_bytes, int *r_sent);

	int (*get_available_bytes)(const void *user);

	void *next; /* For extension? */
} godot_net_stream_peer;

/* Binds a StreamPeerGDNative to the provided interface */
void godot_net_bind_stream_peer(godot_object *p_obj, const godot_net_stream_peer *p_interface);

typedef struct {
	godot_gdnative_api_version version; /* version of our API */

	godot_object *data; /* User reference */

	/* This is PacketPeer */
	godot_error (*get_packet)(void *, const uint8_t **, int *);
	godot_error (*put_packet)(void *, const uint8_t *, int);
	godot_int (*get_available_packet_count)(const void *);
	godot_int (*get_max_packet_size)(const void *);

	void *next; /* For extension? */
} godot_net_packet_peer;

/* Binds a PacketPeerGDNative to the provided interface */
void GDAPI godot_net_bind_packet_peer(godot_object *p_obj, const godot_net_packet_peer *);

typedef struct {
	godot_gdnative_api_version version; /* version of our API */

	godot_object *data; /* User reference */

	/* This is PacketPeer */
	godot_error (*get_packet)(void *, const uint8_t **, int *);
	godot_error (*put_packet)(void *, const uint8_t *, int);
	godot_int (*get_available_packet_count)(const void *);
	godot_int (*get_max_packet_size)(const void *);

	/* This is NetworkedMultiplayerPeer */
	void (*set_transfer_mode)(void *, godot_int);
	godot_int (*get_transfer_mode)(const void *);
	// 0 = broadcast, 1 = server, <0 = all but abs(value)
	void (*set_target_peer)(void *, godot_int);
	godot_int (*get_packet_peer)(const void *);
	godot_bool (*is_server)(const void *);
	void (*poll)(void *);
	// Must be > 0, 1 is for server
	int32_t (*get_unique_id)(const void *);
	void (*set_refuse_new_connections)(void *, godot_bool);
	godot_bool (*is_refusing_new_connections)(const void *);
	godot_int (*get_connection_status)(const void *);

	void *next; /* For extension? Or maybe not... */
} godot_net_multiplayer_peer;

/* Binds a MultiplayerPeerGDNative to the provided interface */
void GDAPI godot_net_bind_multiplayer_peer(godot_object *p_obj, const godot_net_multiplayer_peer *);

#ifdef __cplusplus
}
#endif

// WebRTC Bindings
#include "net/godot_webrtc.h"

#endif /* GODOT_NATIVENET_H */
