/*
 * ./lib/extension-permessage-deflate.c
 *
 *  Copyright (C) 2016 Andy Green <andy@warmcat.com>
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
#include "extension-permessage-deflate.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define LWS_ZLIB_MEMLEVEL 8

const struct lws_ext_options lws_ext_pm_deflate_options[] = {
	/* public RFC7692 settings */
	{ "server_no_context_takeover", EXTARG_NONE },
	{ "client_no_context_takeover", EXTARG_NONE },
	{ "server_max_window_bits",	EXTARG_OPT_DEC },
	{ "client_max_window_bits",	EXTARG_OPT_DEC },
	/* ones only user code can set */
	{ "rx_buf_size",		EXTARG_DEC },
	{ "tx_buf_size",		EXTARG_DEC },
	{ "compression_level",		EXTARG_DEC },
	{ "mem_level",			EXTARG_DEC },
	{ NULL, 0 }, /* sentinel */
};

static void
lws_extension_pmdeflate_restrict_args(struct lws *wsi,
				      struct lws_ext_pm_deflate_priv *priv)
{
	int n, extra;

	/* cap the RX buf at the nearest power of 2 to protocol rx buf */

	n = wsi->context->pt_serv_buf_size;
	if (wsi->protocol->rx_buffer_size)
		n =  wsi->protocol->rx_buffer_size;

	extra = 7;
	while (n >= 1 << (extra + 1))
		extra++;

	if (extra < priv->args[PMD_RX_BUF_PWR2]) {
		priv->args[PMD_RX_BUF_PWR2] = extra;
		lwsl_info(" Capping pmd rx to %d\n", 1 << extra);
	}
}

LWS_VISIBLE int
lws_extension_callback_pm_deflate(struct lws_context *context,
				  const struct lws_extension *ext,
				  struct lws *wsi,
				  enum lws_extension_callback_reasons reason,
				  void *user, void *in, size_t len)
{
	struct lws_ext_pm_deflate_priv *priv =
				     (struct lws_ext_pm_deflate_priv *)user;
	struct lws_tokens *eff_buf = (struct lws_tokens *)in;
	static unsigned char trail[] = { 0, 0, 0xff, 0xff };
	int n, ret = 0, was_fin = 0, extra;
	struct lws_ext_option_arg *oa;

	switch (reason) {
	case LWS_EXT_CB_NAMED_OPTION_SET:
		oa = in;
		if (!oa->option_name)
			break;
		for (n = 0; n < ARRAY_SIZE(lws_ext_pm_deflate_options); n++)
			if (!strcmp(lws_ext_pm_deflate_options[n].name, oa->option_name))
				break;

		if (n == ARRAY_SIZE(lws_ext_pm_deflate_options))
			break;
		oa->option_index = n;

		/* fallthru */

	case LWS_EXT_CB_OPTION_SET:
		oa = in;
		lwsl_notice("%s: option set: idx %d, %s, len %d\n", __func__,
			  oa->option_index, oa->start, oa->len);
		if (oa->start)
			priv->args[oa->option_index] = atoi(oa->start);
		else
			priv->args[oa->option_index] = 1;

		if (priv->args[PMD_CLIENT_MAX_WINDOW_BITS] == 8)
			priv->args[PMD_CLIENT_MAX_WINDOW_BITS] = 9;

		lws_extension_pmdeflate_restrict_args(wsi, priv);
		break;

	case LWS_EXT_CB_OPTION_CONFIRM:
		if (priv->args[PMD_SERVER_MAX_WINDOW_BITS] < 8 ||
		    priv->args[PMD_SERVER_MAX_WINDOW_BITS] > 15 ||
		    priv->args[PMD_CLIENT_MAX_WINDOW_BITS] < 8 ||
		    priv->args[PMD_CLIENT_MAX_WINDOW_BITS] > 15)
			return -1;
		break;

	case LWS_EXT_CB_CLIENT_CONSTRUCT:
	case LWS_EXT_CB_CONSTRUCT:

		n = context->pt_serv_buf_size;
		if (wsi->protocol->rx_buffer_size)
			n =  wsi->protocol->rx_buffer_size;

		if (n < 128) {
			lwsl_info(" permessage-deflate requires the protocol (%s) to have an RX buffer >= 128\n",
					wsi->protocol->name);
			return -1;
		}

		/* fill in **user */
		priv = lws_zalloc(sizeof(*priv), "pmd priv");
		*((void **)user) = priv;
		lwsl_ext("%s: LWS_EXT_CB_*CONSTRUCT\n", __func__);
		memset(priv, 0, sizeof(*priv));

		/* fill in pointer to options list */
		if (in)
			*((const struct lws_ext_options **)in) =
					lws_ext_pm_deflate_options;

		/* fallthru */

	case LWS_EXT_CB_OPTION_DEFAULT:

		/* set the public, RFC7692 defaults... */

		priv->args[PMD_SERVER_NO_CONTEXT_TAKEOVER] = 0,
		priv->args[PMD_CLIENT_NO_CONTEXT_TAKEOVER] = 0;
		priv->args[PMD_SERVER_MAX_WINDOW_BITS] = 15;
		priv->args[PMD_CLIENT_MAX_WINDOW_BITS] = 15;

		/* ...and the ones the user code can override */

		priv->args[PMD_RX_BUF_PWR2] = 10; /* ie, 1024 */
		priv->args[PMD_TX_BUF_PWR2] = 10; /* ie, 1024 */
		priv->args[PMD_COMP_LEVEL] = 1;
		priv->args[PMD_MEM_LEVEL] = 8;

		lws_extension_pmdeflate_restrict_args(wsi, priv);
		break;

	case LWS_EXT_CB_DESTROY:
		lwsl_ext("%s: LWS_EXT_CB_DESTROY\n", __func__);
		lws_free(priv->buf_rx_inflated);
		lws_free(priv->buf_tx_deflated);
		if (priv->rx_init)
			(void)inflateEnd(&priv->rx);
		if (priv->tx_init)
			(void)deflateEnd(&priv->tx);
		lws_free(priv);
		return ret;

	case LWS_EXT_CB_PAYLOAD_RX:
		lwsl_ext(" %s: LWS_EXT_CB_PAYLOAD_RX: in %d, existing in %d\n",
			 __func__, eff_buf->token_len, priv->rx.avail_in);
		if (!(wsi->u.ws.rsv_first_msg & 0x40))
			return 0;

#if 0
		for (n = 0; n < eff_buf->token_len; n++) {
			printf("%02X ", (unsigned char)eff_buf->token[n]);
			if ((n & 15) == 15)
				printf("\n");
		}
		printf("\n");
#endif
		if (!priv->rx_init)
			if (inflateInit2(&priv->rx, -priv->args[PMD_SERVER_MAX_WINDOW_BITS]) != Z_OK) {
				lwsl_err("%s: iniflateInit failed\n", __func__);
				return -1;
			}
		priv->rx_init = 1;
		if (!priv->buf_rx_inflated)
			priv->buf_rx_inflated = lws_malloc(LWS_PRE + 7 + 5 +
					    (1 << priv->args[PMD_RX_BUF_PWR2]), "pmd rx inflate buf");
		if (!priv->buf_rx_inflated) {
			lwsl_err("%s: OOM\n", __func__);
			return -1;
		}

		/*
		 * We have to leave the input stream alone if we didn't
		 * finish with it yet.  The input stream is held in the wsi
		 * rx buffer by the caller, so this assumption is safe while
		 * we block new rx while draining the existing rx
		 */
		if (!priv->rx.avail_in && eff_buf->token && eff_buf->token_len) {
			priv->rx.next_in = (unsigned char *)eff_buf->token;
			priv->rx.avail_in = eff_buf->token_len;
		}
		priv->rx.next_out = priv->buf_rx_inflated + LWS_PRE;
		eff_buf->token = (char *)priv->rx.next_out;
		priv->rx.avail_out = 1 << priv->args[PMD_RX_BUF_PWR2];

		if (priv->rx_held_valid) {
			lwsl_ext("-- RX piling on held byte --\n");
			*(priv->rx.next_out++) = priv->rx_held;
			priv->rx.avail_out--;
			priv->rx_held_valid = 0;
		}

		/* if...
		 *
		 *  - he has no remaining input content for this message, and
		 *  - and this is the final fragment, and
		 *  - we used everything that could be drained on the input side
		 *
		 * ...then put back the 00 00 FF FF the sender stripped as our
		 * input to zlib
		 */
		if (!priv->rx.avail_in && wsi->u.ws.final &&
		    !wsi->u.ws.rx_packet_length) {
			lwsl_ext("RX APPEND_TRAILER-DO\n");
			was_fin = 1;
			priv->rx.next_in = trail;
			priv->rx.avail_in = sizeof(trail);
		}

		n = inflate(&priv->rx, Z_NO_FLUSH);
		lwsl_ext("inflate ret %d, avi %d, avo %d, wsifinal %d\n", n,
			 priv->rx.avail_in, priv->rx.avail_out, wsi->u.ws.final);
		switch (n) {
		case Z_NEED_DICT:
		case Z_STREAM_ERROR:
		case Z_DATA_ERROR:
		case Z_MEM_ERROR:
			lwsl_info("zlib error inflate %d: %s\n",
				  n, priv->rx.msg);
			return -1;
		}
		/*
		 * If we did not already send in the 00 00 FF FF, and he's
		 * out of input, he did not EXACTLY fill the output buffer
		 * (which is ambiguous and we will force it to go around
		 * again by withholding a byte), and he's otherwise working on
		 * being a FIN fragment, then do the FIN message processing
		 * of faking up the 00 00 FF FF that the sender stripped.
		 */
		if (!priv->rx.avail_in && wsi->u.ws.final &&
		    !wsi->u.ws.rx_packet_length && !was_fin &&
		    priv->rx.avail_out /* ambiguous as to if it is the end */
		) {
			lwsl_ext("RX APPEND_TRAILER-DO\n");
			was_fin = 1;
			priv->rx.next_in = trail;
			priv->rx.avail_in = sizeof(trail);
			n = inflate(&priv->rx, Z_SYNC_FLUSH);
			lwsl_ext("RX trailer inf returned %d, avi %d, avo %d\n", n,
				 priv->rx.avail_in, priv->rx.avail_out);
			switch (n) {
			case Z_NEED_DICT:
			case Z_STREAM_ERROR:
			case Z_DATA_ERROR:
			case Z_MEM_ERROR:
				lwsl_info("zlib error inflate %d: %s\n",
					  n, priv->rx.msg);
				return -1;
			}
		}
		/*
		 * we must announce in our returncode now if there is more
		 * output to be expected from inflate, so we can decide to
		 * set the FIN bit on this bufferload or not.  However zlib
		 * is ambiguous when we exactly filled the inflate buffer.  It
		 * does not give us a clue as to whether we should understand
		 * that to mean he ended on a buffer boundary, or if there is
		 * more in the pipeline.
		 *
		 * So to work around that safely, if it used all output space
		 * exactly, we ALWAYS say there is more coming and we withhold
		 * the last byte of the buffer to guarantee that is true.
		 *
		 * That still leaves us at least one byte to finish with a FIN
		 * on, even if actually nothing more is coming from the next
		 * inflate action itself.
		 */
		if (!priv->rx.avail_out) { /* he used all available out buf */
			lwsl_ext("-- rx grabbing held --\n");
			/* snip the last byte and hold it for next time */
			priv->rx_held = *(--priv->rx.next_out);
			priv->rx_held_valid = 1;
		}

		eff_buf->token_len = (char *)priv->rx.next_out - eff_buf->token;
		priv->count_rx_between_fin += eff_buf->token_len;

		lwsl_ext("  %s: RX leaving with new effbuff len %d, "
			 "ret %d, rx.avail_in=%d, TOTAL RX since FIN %lu\n",
			 __func__, eff_buf->token_len, priv->rx_held_valid,
			 priv->rx.avail_in,
			 (unsigned long)priv->count_rx_between_fin);

		if (was_fin) {
			priv->count_rx_between_fin = 0;
			if (priv->args[PMD_SERVER_NO_CONTEXT_TAKEOVER]) {
				(void)inflateEnd(&priv->rx);
				priv->rx_init = 0;
			}
		}
#if 0
		for (n = 0; n < eff_buf->token_len; n++)
			putchar(eff_buf->token[n]);
		puts("\n");
#endif

		return priv->rx_held_valid;

	case LWS_EXT_CB_PAYLOAD_TX:

		if (!priv->tx_init) {
			n = deflateInit2(&priv->tx, priv->args[PMD_COMP_LEVEL],
					 Z_DEFLATED,
					 -priv->args[PMD_SERVER_MAX_WINDOW_BITS +
						     (wsi->vhost->listen_port <= 0)],
					 priv->args[PMD_MEM_LEVEL],
					 Z_DEFAULT_STRATEGY);
			if (n != Z_OK) {
				lwsl_ext("inflateInit2 failed %d\n", n);
				return 1;
			}
		}
		priv->tx_init = 1;
		if (!priv->buf_tx_deflated)
			priv->buf_tx_deflated = lws_malloc(LWS_PRE + 7 + 5 +
					    (1 << priv->args[PMD_TX_BUF_PWR2]), "pmd tx deflate buf");
		if (!priv->buf_tx_deflated) {
			lwsl_err("%s: OOM\n", __func__);
			return -1;
		}

		if (eff_buf->token) {
			lwsl_ext("%s: TX: eff_buf length %d\n", __func__,
				 eff_buf->token_len);
			priv->tx.next_in = (unsigned char *)eff_buf->token;
			priv->tx.avail_in = eff_buf->token_len;
		}

#if 0
		for (n = 0; n < eff_buf->token_len; n++) {
			printf("%02X ", (unsigned char)eff_buf->token[n]);
			if ((n & 15) == 15)
				printf("\n");
		}
		printf("\n");
#endif

		priv->tx.next_out = priv->buf_tx_deflated + LWS_PRE + 5;
		eff_buf->token = (char *)priv->tx.next_out;
		priv->tx.avail_out = 1 << priv->args[PMD_TX_BUF_PWR2];

		n = deflate(&priv->tx, Z_SYNC_FLUSH);
		if (n == Z_STREAM_ERROR) {
			lwsl_ext("%s: Z_STREAM_ERROR\n", __func__);
			return -1;
		}

		if (priv->tx_held_valid) {
			priv->tx_held_valid = 0;
			if (priv->tx.avail_out == 1 << priv->args[PMD_TX_BUF_PWR2])
				/*
				 * we can get a situation he took something in
				 * but did not generate anything out, at the end
				 * of a message (eg, next thing he sends is 80
				 * 00, a zero length FIN, like Authobahn can
				 * send).
				 * If we have come back as a FIN, we must not
				 * place the pending trailer 00 00 FF FF, just
				 * the 1 byte of live data
				 */
				*(--eff_buf->token) = priv->tx_held[0];
			else {
				/* he generated data, prepend whole pending */
				eff_buf->token -= 5;
				for (n = 0; n < 5; n++)
					eff_buf->token[n] = priv->tx_held[n];

			}
		}
		priv->compressed_out = 1;
		eff_buf->token_len = (int)(priv->tx.next_out -
					   (unsigned char *)eff_buf->token);

		/*
		 * we must announce in our returncode now if there is more
		 * output to be expected from inflate, so we can decide to
		 * set the FIN bit on this bufferload or not.  However zlib
		 * is ambiguous when we exactly filled the inflate buffer.  It
		 * does not give us a clue as to whether we should understand
		 * that to mean he ended on a buffer boundary, or if there is
		 * more in the pipeline.
		 *
		 * Worse, the guy providing the stuff we are sending may not
		 * know until after that this was, actually, the last chunk,
		 * that can happen even if we did not fill the output buf, ie
		 * he may send after this a zero-length FIN fragment.
		 *
		 * This is super difficult because we must snip the last 4
		 * bytes in the case this is the last compressed output of the
		 * message.  The only way to deal with it is defer sending the
		 * last 5 bytes of each frame until the next one, when we will
		 * be in a position to understand if that has a FIN or not.
		 */

		extra = !!(len & LWS_WRITE_NO_FIN) || !priv->tx.avail_out;

		if (eff_buf->token_len >= 4 + extra) {
			lwsl_ext("tx held %d\n", 4 + extra);
			priv->tx_held_valid = extra;
			for (n = 3 + extra; n >= 0; n--)
				priv->tx_held[n] = *(--priv->tx.next_out);
			eff_buf->token_len -= 4 + extra;
		}
		lwsl_ext("  TX rewritten with new effbuff len %d, ret %d\n",
			 eff_buf->token_len, !priv->tx.avail_out);

		return !priv->tx.avail_out; /* 1 == have more tx pending */

	case LWS_EXT_CB_PACKET_TX_PRESEND:
		if (!priv->compressed_out)
			break;
		priv->compressed_out = 0;

		if ((*(eff_buf->token) & 0x80) &&
		    priv->args[PMD_CLIENT_NO_CONTEXT_TAKEOVER]) {
			lwsl_debug("PMD_CLIENT_NO_CONTEXT_TAKEOVER\n");
			(void)deflateEnd(&priv->tx);
			priv->tx_init = 0;
		}

		n = *(eff_buf->token) & 15;
		/* set RSV1, but not on CONTINUATION */
		if (n == LWSWSOPC_TEXT_FRAME || n == LWSWSOPC_BINARY_FRAME)
			*eff_buf->token |= 0x40;
#if 0
		for (n = 0; n < eff_buf->token_len; n++) {
			printf("%02X ", (unsigned char)eff_buf->token[n]);
			if ((n & 15) == 15)
				puts("\n");
		}
		puts("\n");
#endif
		lwsl_ext("%s: tx opcode 0x%02X\n", __func__,
			 (unsigned char)*eff_buf->token);
		break;

	default:
		break;
	}

	return 0;
}

