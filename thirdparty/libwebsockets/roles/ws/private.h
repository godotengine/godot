/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010 - 2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 *  This is included from core/private.h if LWS_ROLE_WS
 */

extern struct lws_role_ops role_ops_ws;

#define lwsi_role_ws(wsi) (wsi->role_ops == &role_ops_ws)

enum lws_rx_parse_state {
	LWS_RXPS_NEW,

	LWS_RXPS_04_mask_1,
	LWS_RXPS_04_mask_2,
	LWS_RXPS_04_mask_3,

	LWS_RXPS_04_FRAME_HDR_1,
	LWS_RXPS_04_FRAME_HDR_LEN,
	LWS_RXPS_04_FRAME_HDR_LEN16_2,
	LWS_RXPS_04_FRAME_HDR_LEN16_1,
	LWS_RXPS_04_FRAME_HDR_LEN64_8,
	LWS_RXPS_04_FRAME_HDR_LEN64_7,
	LWS_RXPS_04_FRAME_HDR_LEN64_6,
	LWS_RXPS_04_FRAME_HDR_LEN64_5,
	LWS_RXPS_04_FRAME_HDR_LEN64_4,
	LWS_RXPS_04_FRAME_HDR_LEN64_3,
	LWS_RXPS_04_FRAME_HDR_LEN64_2,
	LWS_RXPS_04_FRAME_HDR_LEN64_1,

	LWS_RXPS_07_COLLECT_FRAME_KEY_1,
	LWS_RXPS_07_COLLECT_FRAME_KEY_2,
	LWS_RXPS_07_COLLECT_FRAME_KEY_3,
	LWS_RXPS_07_COLLECT_FRAME_KEY_4,

	LWS_RXPS_WS_FRAME_PAYLOAD
};

enum lws_websocket_opcodes_07 {
	LWSWSOPC_CONTINUATION = 0,
	LWSWSOPC_TEXT_FRAME = 1,
	LWSWSOPC_BINARY_FRAME = 2,

	LWSWSOPC_NOSPEC__MUX = 7,

	/* control extensions 8+ */

	LWSWSOPC_CLOSE = 8,
	LWSWSOPC_PING = 9,
	LWSWSOPC_PONG = 0xa,
};

/* this is not usable directly by user code any more, lws_close_reason() */
#define LWS_WRITE_CLOSE 4

#define ALREADY_PROCESSED_IGNORE_CHAR 1
#define ALREADY_PROCESSED_NO_CB 2

#if !defined(LWS_WITHOUT_EXTENSIONS)
struct lws_vhost_role_ws {
	const struct lws_extension *extensions;
};

struct lws_pt_role_ws {
	struct lws *rx_draining_ext_list;
	struct lws *tx_draining_ext_list;
};
#endif

struct _lws_websocket_related {
	char *rx_ubuf;
#if !defined(LWS_WITHOUT_EXTENSIONS)
	const struct lws_extension *active_extensions[LWS_MAX_EXTENSIONS_ACTIVE];
	void *act_ext_user[LWS_MAX_EXTENSIONS_ACTIVE];
	struct lws *rx_draining_ext_list;
	struct lws *tx_draining_ext_list;
#endif
	/* Also used for close content... control opcode == < 128 */
	uint8_t ping_payload_buf[128 - 3 + LWS_PRE];
	uint8_t mask[4];

	time_t time_next_ping_check;
	size_t rx_packet_length;
	uint32_t rx_ubuf_head;
	uint32_t rx_ubuf_alloc;

	uint8_t ping_payload_len;
	uint8_t mask_idx;
	uint8_t opcode;
	uint8_t rsv;
	uint8_t rsv_first_msg;
	/* zero if no info, or length including 2-byte close code */
	uint8_t close_in_ping_buffer_len;
	uint8_t utf8;
	uint8_t stashed_write_type;
	uint8_t tx_draining_stashed_wp;
	uint8_t ietf_spec_revision;

	unsigned int final:1;
	unsigned int frame_is_binary:1;
	unsigned int all_zero_nonce:1;
	unsigned int this_frame_masked:1;
	unsigned int inside_frame:1; /* next write will be more of frame */
	unsigned int clean_buffer:1; /* buffer not rewritten by extension */
	unsigned int payload_is_close:1; /* process as PONG, but it is close */
	unsigned int ping_pending_flag:1;
	unsigned int continuation_possible:1;
	unsigned int owed_a_fin:1;
	unsigned int check_utf8:1;
	unsigned int defeat_check_utf8:1;
	unsigned int stashed_write_pending:1;
	unsigned int send_check_ping:1;
	unsigned int first_fragment:1;
	unsigned int peer_has_sent_close:1;
#if !defined(LWS_WITHOUT_EXTENSIONS)
	unsigned int extension_data_pending:1;
	unsigned int rx_draining_ext:1;
	unsigned int tx_draining_ext:1;

	uint8_t count_act_ext;
#endif
};

int
lws_ws_handshake_client(struct lws *wsi, unsigned char **buf, size_t len);

#if !defined(LWS_WITHOUT_EXTENSIONS)
LWS_VISIBLE void
lws_context_init_extensions(const struct lws_context_creation_info *info,
			    struct lws_context *context);
LWS_EXTERN int
lws_any_extension_handled(struct lws *wsi, enum lws_extension_callback_reasons r,
			  void *v, size_t len);

LWS_EXTERN int
lws_ext_cb_active(struct lws *wsi, int reason, void *buf, int len);
LWS_EXTERN int
lws_ext_cb_all_exts(struct lws_context *context, struct lws *wsi, int reason,
		    void *arg, int len);
#endif

int
handshake_0405(struct lws_context *context, struct lws *wsi);
int
lws_process_ws_upgrade(struct lws *wsi);
int
lws_server_init_wsi_for_ws(struct lws *wsi);
