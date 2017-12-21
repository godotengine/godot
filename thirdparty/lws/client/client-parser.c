/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2017 Andy Green <andy@warmcat.com>
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
 */

#include "private-libwebsockets.h"

/*
 * parsers.c: lws_rx_sm() needs to be roughly kept in
 *   sync with changes here, esp related to ext draining
 */

int lws_client_rx_sm(struct lws *wsi, unsigned char c)
{
	int callback_action = LWS_CALLBACK_CLIENT_RECEIVE;
	int handled, n, m, rx_draining_ext = 0;
	unsigned short close_code;
	struct lws_tokens eff_buf;
	unsigned char *pp;

	if (wsi->u.ws.rx_draining_ext) {
		assert(!c);
		eff_buf.token = NULL;
		eff_buf.token_len = 0;
		lws_remove_wsi_from_draining_ext_list(wsi);
		rx_draining_ext = 1;
		lwsl_debug("%s: doing draining flow\n", __func__);

		goto drain_extension;
	}

	if (wsi->socket_is_permanently_unusable)
		return -1;

	switch (wsi->lws_rx_parse_state) {
	case LWS_RXPS_NEW:
		/* control frames (PING) may interrupt checkable sequences */
		wsi->u.ws.defeat_check_utf8 = 0;

		switch (wsi->ietf_spec_revision) {
		case 13:
			wsi->u.ws.opcode = c & 0xf;
			/* revisit if an extension wants them... */
			switch (wsi->u.ws.opcode) {
			case LWSWSOPC_TEXT_FRAME:
				wsi->u.ws.rsv_first_msg = (c & 0x70);
				wsi->u.ws.continuation_possible = 1;
				wsi->u.ws.check_utf8 = lws_check_opt(
					wsi->context->options,
					LWS_SERVER_OPTION_VALIDATE_UTF8);
				wsi->u.ws.utf8 = 0;
				break;
			case LWSWSOPC_BINARY_FRAME:
				wsi->u.ws.rsv_first_msg = (c & 0x70);
				wsi->u.ws.check_utf8 = 0;
				wsi->u.ws.continuation_possible = 1;
				break;
			case LWSWSOPC_CONTINUATION:
				if (!wsi->u.ws.continuation_possible) {
					lwsl_info("disordered continuation\n");
					return -1;
				}
				break;
			case LWSWSOPC_CLOSE:
				wsi->u.ws.check_utf8 = 0;
				wsi->u.ws.utf8 = 0;
				break;
			case 3:
			case 4:
			case 5:
			case 6:
			case 7:
			case 0xb:
			case 0xc:
			case 0xd:
			case 0xe:
			case 0xf:
				lwsl_info("illegal opcode\n");
				return -1;
			default:
				wsi->u.ws.defeat_check_utf8 = 1;
				break;
			}
			wsi->u.ws.rsv = (c & 0x70);
			/* revisit if an extension wants them... */
			if (
#ifndef LWS_NO_EXTENSIONS
				!wsi->count_act_ext &&
#endif
				wsi->u.ws.rsv) {
				lwsl_info("illegal rsv bits set\n");
				return -1;
			}
			wsi->u.ws.final = !!((c >> 7) & 1);
			lwsl_ext("%s:    This RX frame Final %d\n", __func__,
				 wsi->u.ws.final);

			if (wsi->u.ws.owed_a_fin &&
			    (wsi->u.ws.opcode == LWSWSOPC_TEXT_FRAME ||
			     wsi->u.ws.opcode == LWSWSOPC_BINARY_FRAME)) {
				lwsl_info("hey you owed us a FIN\n");
				return -1;
			}
			if ((!(wsi->u.ws.opcode & 8)) && wsi->u.ws.final) {
				wsi->u.ws.continuation_possible = 0;
				wsi->u.ws.owed_a_fin = 0;
			}

			if ((wsi->u.ws.opcode & 8) && !wsi->u.ws.final) {
				lwsl_info("control msg can't be fragmented\n");
				return -1;
			}
			if (!wsi->u.ws.final)
				wsi->u.ws.owed_a_fin = 1;

			switch (wsi->u.ws.opcode) {
			case LWSWSOPC_TEXT_FRAME:
			case LWSWSOPC_BINARY_FRAME:
				wsi->u.ws.frame_is_binary = wsi->u.ws.opcode ==
						 LWSWSOPC_BINARY_FRAME;
				break;
			}
			wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN;
			break;

		default:
			lwsl_err("unknown spec version %02d\n",
				 wsi->ietf_spec_revision);
			break;
		}
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN:

		wsi->u.ws.this_frame_masked = !!(c & 0x80);

		switch (c & 0x7f) {
		case 126:
			/* control frames are not allowed to have big lengths */
			if (wsi->u.ws.opcode & 8)
				goto illegal_ctl_length;
			wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN16_2;
			break;
		case 127:
			/* control frames are not allowed to have big lengths */
			if (wsi->u.ws.opcode & 8)
				goto illegal_ctl_length;
			wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_8;
			break;
		default:
			wsi->u.ws.rx_packet_length = c;
			if (wsi->u.ws.this_frame_masked)
				wsi->lws_rx_parse_state =
						LWS_RXPS_07_COLLECT_FRAME_KEY_1;
			else {
				if (c)
					wsi->lws_rx_parse_state =
					LWS_RXPS_PAYLOAD_UNTIL_LENGTH_EXHAUSTED;
				else {
					wsi->lws_rx_parse_state = LWS_RXPS_NEW;
					goto spill;
				}
			}
			break;
		}
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN16_2:
		wsi->u.ws.rx_packet_length = c << 8;
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN16_1;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN16_1:
		wsi->u.ws.rx_packet_length |= c;
		if (wsi->u.ws.this_frame_masked)
			wsi->lws_rx_parse_state = LWS_RXPS_07_COLLECT_FRAME_KEY_1;
		else {
			if (wsi->u.ws.rx_packet_length)
				wsi->lws_rx_parse_state =
					LWS_RXPS_PAYLOAD_UNTIL_LENGTH_EXHAUSTED;
			else {
				wsi->lws_rx_parse_state = LWS_RXPS_NEW;
				goto spill;
			}
		}
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_8:
		if (c & 0x80) {
			lwsl_warn("b63 of length must be zero\n");
			/* kill the connection */
			return -1;
		}
#if defined __LP64__
		wsi->u.ws.rx_packet_length = ((size_t)c) << 56;
#else
		wsi->u.ws.rx_packet_length = 0;
#endif
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_7;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_7:
#if defined __LP64__
		wsi->u.ws.rx_packet_length |= ((size_t)c) << 48;
#endif
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_6;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_6:
#if defined __LP64__
		wsi->u.ws.rx_packet_length |= ((size_t)c) << 40;
#endif
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_5;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_5:
#if defined __LP64__
		wsi->u.ws.rx_packet_length |= ((size_t)c) << 32;
#endif
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_4;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_4:
		wsi->u.ws.rx_packet_length |= ((size_t)c) << 24;
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_3;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_3:
		wsi->u.ws.rx_packet_length |= ((size_t)c) << 16;
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_2;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_2:
		wsi->u.ws.rx_packet_length |= ((size_t)c) << 8;
		wsi->lws_rx_parse_state = LWS_RXPS_04_FRAME_HDR_LEN64_1;
		break;

	case LWS_RXPS_04_FRAME_HDR_LEN64_1:
		wsi->u.ws.rx_packet_length |= (size_t)c;
		if (wsi->u.ws.this_frame_masked)
			wsi->lws_rx_parse_state =
					LWS_RXPS_07_COLLECT_FRAME_KEY_1;
		else {
			if (wsi->u.ws.rx_packet_length)
				wsi->lws_rx_parse_state =
					LWS_RXPS_PAYLOAD_UNTIL_LENGTH_EXHAUSTED;
			else {
				wsi->lws_rx_parse_state = LWS_RXPS_NEW;
				goto spill;
			}
		}
		break;

	case LWS_RXPS_07_COLLECT_FRAME_KEY_1:
		wsi->u.ws.mask[0] = c;
		if (c)
			wsi->u.ws.all_zero_nonce = 0;
		wsi->lws_rx_parse_state = LWS_RXPS_07_COLLECT_FRAME_KEY_2;
		break;

	case LWS_RXPS_07_COLLECT_FRAME_KEY_2:
		wsi->u.ws.mask[1] = c;
		if (c)
			wsi->u.ws.all_zero_nonce = 0;
		wsi->lws_rx_parse_state = LWS_RXPS_07_COLLECT_FRAME_KEY_3;
		break;

	case LWS_RXPS_07_COLLECT_FRAME_KEY_3:
		wsi->u.ws.mask[2] = c;
		if (c)
			wsi->u.ws.all_zero_nonce = 0;
		wsi->lws_rx_parse_state = LWS_RXPS_07_COLLECT_FRAME_KEY_4;
		break;

	case LWS_RXPS_07_COLLECT_FRAME_KEY_4:
		wsi->u.ws.mask[3] = c;
		if (c)
			wsi->u.ws.all_zero_nonce = 0;

		if (wsi->u.ws.rx_packet_length)
			wsi->lws_rx_parse_state =
					LWS_RXPS_PAYLOAD_UNTIL_LENGTH_EXHAUSTED;
		else {
			wsi->lws_rx_parse_state = LWS_RXPS_NEW;
			goto spill;
		}
		break;

	case LWS_RXPS_PAYLOAD_UNTIL_LENGTH_EXHAUSTED:

		assert(wsi->u.ws.rx_ubuf);

		if (wsi->u.ws.rx_draining_ext)
			goto drain_extension;

		if (wsi->u.ws.this_frame_masked && !wsi->u.ws.all_zero_nonce)
			c ^= wsi->u.ws.mask[(wsi->u.ws.mask_idx++) & 3];

		wsi->u.ws.rx_ubuf[LWS_PRE + (wsi->u.ws.rx_ubuf_head++)] = c;

		if (--wsi->u.ws.rx_packet_length == 0) {
			/* spill because we have the whole frame */
			wsi->lws_rx_parse_state = LWS_RXPS_NEW;
			goto spill;
		}

		/*
		 * if there's no protocol max frame size given, we are
		 * supposed to default to context->pt_serv_buf_size
		 */
		if (!wsi->protocol->rx_buffer_size &&
		    wsi->u.ws.rx_ubuf_head != wsi->context->pt_serv_buf_size)
			break;

		if (wsi->protocol->rx_buffer_size &&
		    wsi->u.ws.rx_ubuf_head != wsi->protocol->rx_buffer_size)
			break;

		/* spill because we filled our rx buffer */
spill:

		handled = 0;

		/*
		 * is this frame a control packet we should take care of at this
		 * layer?  If so service it and hide it from the user callback
		 */

		switch (wsi->u.ws.opcode) {
		case LWSWSOPC_CLOSE:
			pp = (unsigned char *)&wsi->u.ws.rx_ubuf[LWS_PRE];
			if (lws_check_opt(wsi->context->options,
					  LWS_SERVER_OPTION_VALIDATE_UTF8) &&
			    wsi->u.ws.rx_ubuf_head > 2 &&
			    lws_check_utf8(&wsi->u.ws.utf8, pp + 2,
					   wsi->u.ws.rx_ubuf_head - 2))
				goto utf8_fail;

			/* is this an acknowledgement of our close? */
			if (wsi->state == LWSS_AWAITING_CLOSE_ACK) {
				/*
				 * fine he has told us he is closing too, let's
				 * finish our close
				 */
				lwsl_parser("seen server's close ack\n");
				return -1;
			}

			lwsl_parser("client sees server close len = %d\n",
						 wsi->u.ws.rx_ubuf_head);
			if (wsi->u.ws.rx_ubuf_head >= 2) {
				close_code = (pp[0] << 8) | pp[1];
				if (close_code < 1000 ||
				    close_code == 1004 ||
				    close_code == 1005 ||
				    close_code == 1006 ||
				    close_code == 1012 ||
				    close_code == 1013 ||
				    close_code == 1014 ||
				    close_code == 1015 ||
				    (close_code >= 1016 && close_code < 3000)
				) {
					pp[0] = (LWS_CLOSE_STATUS_PROTOCOL_ERR >> 8) & 0xff;
					pp[1] = LWS_CLOSE_STATUS_PROTOCOL_ERR & 0xff;
				}
			}
			if (user_callback_handle_rxflow(
					wsi->protocol->callback, wsi,
					LWS_CALLBACK_WS_PEER_INITIATED_CLOSE,
					wsi->user_space, pp,
					wsi->u.ws.rx_ubuf_head))
				return -1;

			if (lws_partial_buffered(wsi))
				/*
				 * if we're in the middle of something,
				 * we can't do a normal close response and
				 * have to just close our end.
				 */
				wsi->socket_is_permanently_unusable = 1;
			else
				/*
				 * parrot the close packet payload back
				 * we do not care about how it went, we are closing
				 * immediately afterwards
				 */
				lws_write(wsi, (unsigned char *)
					  &wsi->u.ws.rx_ubuf[LWS_PRE],
					  wsi->u.ws.rx_ubuf_head,
					  LWS_WRITE_CLOSE);
			wsi->state = LWSS_RETURNED_CLOSE_ALREADY;
			/* close the connection */
			return -1;

		case LWSWSOPC_PING:
			lwsl_info("received %d byte ping, sending pong\n",
				  wsi->u.ws.rx_ubuf_head);

			/* he set a close reason on this guy, ignore PING */
			if (wsi->u.ws.close_in_ping_buffer_len)
				goto ping_drop;

			if (wsi->u.ws.ping_pending_flag) {
				/*
				 * there is already a pending ping payload
				 * we should just log and drop
				 */
				lwsl_parser("DROP PING since one pending\n");
				goto ping_drop;
			}

			/* control packets can only be < 128 bytes long */
			if (wsi->u.ws.rx_ubuf_head > 128 - 3) {
				lwsl_parser("DROP PING payload too large\n");
				goto ping_drop;
			}

			/* stash the pong payload */
			memcpy(wsi->u.ws.ping_payload_buf + LWS_PRE,
			       &wsi->u.ws.rx_ubuf[LWS_PRE],
				wsi->u.ws.rx_ubuf_head);

			wsi->u.ws.ping_payload_len = wsi->u.ws.rx_ubuf_head;
			wsi->u.ws.ping_pending_flag = 1;

			/* get it sent as soon as possible */
			lws_callback_on_writable(wsi);
ping_drop:
			wsi->u.ws.rx_ubuf_head = 0;
			handled = 1;
			break;

		case LWSWSOPC_PONG:
			lwsl_info("client receied pong\n");
			lwsl_hexdump(&wsi->u.ws.rx_ubuf[LWS_PRE],
				     wsi->u.ws.rx_ubuf_head);

			if (wsi->pending_timeout ==
				       PENDING_TIMEOUT_WS_PONG_CHECK_GET_PONG) {
				lwsl_info("%p: received expected PONG\n", wsi);
				lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);
			}

			/* issue it */
			callback_action = LWS_CALLBACK_CLIENT_RECEIVE_PONG;
			break;

		case LWSWSOPC_CONTINUATION:
		case LWSWSOPC_TEXT_FRAME:
		case LWSWSOPC_BINARY_FRAME:
			break;

		default:

			lwsl_parser("Reserved opc 0x%2X\n", wsi->u.ws.opcode);

			/*
			 * It's something special we can't understand here.
			 * Pass the payload up to the extension's parsing
			 * state machine.
			 */

			eff_buf.token = &wsi->u.ws.rx_ubuf[LWS_PRE];
			eff_buf.token_len = wsi->u.ws.rx_ubuf_head;

			if (lws_ext_cb_active(wsi,
				LWS_EXT_CB_EXTENDED_PAYLOAD_RX,
					&eff_buf, 0) <= 0) {
				/* not handled or failed */
				lwsl_ext("Unhandled ext opc 0x%x\n",
					 wsi->u.ws.opcode);
				wsi->u.ws.rx_ubuf_head = 0;

				return 0;
			}
			handled = 1;
			break;
		}

		/*
		 * No it's real payload, pass it up to the user callback.
		 * It's nicely buffered with the pre-padding taken care of
		 * so it can be sent straight out again using lws_write
		 */
		if (handled)
			goto already_done;

		eff_buf.token = &wsi->u.ws.rx_ubuf[LWS_PRE];
		eff_buf.token_len = wsi->u.ws.rx_ubuf_head;

		if (wsi->u.ws.opcode == LWSWSOPC_PONG && !eff_buf.token_len)
			goto already_done;

drain_extension:
		lwsl_ext("%s: passing %d to ext\n", __func__, eff_buf.token_len);

		n = lws_ext_cb_active(wsi, LWS_EXT_CB_PAYLOAD_RX, &eff_buf, 0);
		lwsl_ext("Ext RX returned %d\n", n);
		if (n < 0) {
			wsi->socket_is_permanently_unusable = 1;
			return -1;
		}

		lwsl_ext("post inflate eff_buf len %d\n", eff_buf.token_len);

		if (rx_draining_ext && !eff_buf.token_len) {
			lwsl_debug("   --- ending drain on 0 read result\n");
			goto already_done;
		}

		if (wsi->u.ws.check_utf8 && !wsi->u.ws.defeat_check_utf8) {
			if (lws_check_utf8(&wsi->u.ws.utf8,
					   (unsigned char *)eff_buf.token,
					   eff_buf.token_len))
				goto utf8_fail;

			/* we are ending partway through utf-8 character? */
			if (!wsi->u.ws.rx_packet_length && wsi->u.ws.final &&
			    wsi->u.ws.utf8 && !n) {
				lwsl_info("FINAL utf8 error\n");
utf8_fail:
				lwsl_info("utf8 error\n");
				return -1;
			}
		}

		if (eff_buf.token_len < 0 &&
		    callback_action != LWS_CALLBACK_CLIENT_RECEIVE_PONG)
			goto already_done;

		if (!eff_buf.token)
			goto already_done;

		eff_buf.token[eff_buf.token_len] = '\0';

		if (!wsi->protocol->callback)
			goto already_done;

		if (callback_action == LWS_CALLBACK_CLIENT_RECEIVE_PONG)
			lwsl_info("Client doing pong callback\n");

		if (n && eff_buf.token_len)
			/* extension had more... main loop will come back
			 * we want callback to be done with this set, if so,
			 * because lws_is_final() hides it was final until the
			 * last chunk
			 */
			lws_add_wsi_to_draining_ext_list(wsi);
		else
			lws_remove_wsi_from_draining_ext_list(wsi);

		if (wsi->state == LWSS_RETURNED_CLOSE_ALREADY ||
		    wsi->state == LWSS_WAITING_TO_SEND_CLOSE_NOTIFICATION ||
		    wsi->state == LWSS_AWAITING_CLOSE_ACK)
			goto already_done;

		m = wsi->protocol->callback(wsi,
			(enum lws_callback_reasons)callback_action,
			wsi->user_space, eff_buf.token, eff_buf.token_len);

		/* if user code wants to close, let caller know */
		if (m)
			return 1;

already_done:
		wsi->u.ws.rx_ubuf_head = 0;
		break;
	default:
		lwsl_err("client rx illegal state\n");
		return 1;
	}

	return 0;

illegal_ctl_length:
	lwsl_warn("Control frame asking for extended length is illegal\n");

	/* kill the connection */
	return -1;
}


