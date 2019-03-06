/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
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

#include <core/private.h>

/*
 * In-place str to lower case
 */

static void
strtolower(char *s)
{
	while (*s) {
#ifdef LWS_PLAT_OPTEE
		int tolower_optee(int c);
		*s = tolower_optee((int)*s);
#else
		*s = tolower((int)*s);
#endif
		s++;
	}
}

int
lws_create_client_ws_object(const struct lws_client_connect_info *i,
			    struct lws *wsi)
{
	int v = SPEC_LATEST_SUPPORTED;

	/* allocate the ws struct for the wsi */
	wsi->ws = lws_zalloc(sizeof(*wsi->ws), "client ws struct");
	if (!wsi->ws) {
		lwsl_notice("OOM\n");
		return 1;
	}

	/* -1 means just use latest supported */
	if (i->ietf_version_or_minus_one != -1 &&
	    i->ietf_version_or_minus_one)
		v = i->ietf_version_or_minus_one;

	wsi->ws->ietf_spec_revision = v;

	return 0;
}

#if !defined(LWS_NO_CLIENT)
int
lws_ws_handshake_client(struct lws *wsi, unsigned char **buf, size_t len)
{
	if ((lwsi_state(wsi) != LRS_WAITING_PROXY_REPLY) &&
	    (lwsi_state(wsi) != LRS_H1C_ISSUE_HANDSHAKE) &&
	    (lwsi_state(wsi) != LRS_WAITING_SERVER_REPLY) &&
	    !lwsi_role_client(wsi))
		return 0;

	// lwsl_notice("%s: hs client gets %d in\n", __func__, (int)len);

	while (len) {
		/*
		 * we were accepting input but now we stopped doing so
		 */
		if (lws_is_flowcontrolled(wsi)) {
			//lwsl_notice("%s: caching %ld\n", __func__, (long)len);
			lws_rxflow_cache(wsi, *buf, 0, (int)len);
			*buf += len;
			return 0;
		}
#if !defined(LWS_WITHOUT_EXTENSIONS)
		if (wsi->ws->rx_draining_ext) {
			int m;

			//lwsl_notice("%s: draining ext\n", __func__);
			if (lwsi_role_client(wsi))
				m = lws_ws_client_rx_sm(wsi, 0);
			else
				m = lws_ws_rx_sm(wsi, 0, 0);
			if (m < 0)
				return -1;
			continue;
		}
#endif
		/* caller will account for buflist usage */

		if (lws_ws_client_rx_sm(wsi, *(*buf)++)) {
			lwsl_notice("%s: client_rx_sm exited, DROPPING %d\n",
				    __func__, (int)len);
			return -1;
		}
		len--;
	}
	// lwsl_notice("%s: finished with %ld\n", __func__, (long)len);

	return 0;
}
#endif

char *
lws_generate_client_ws_handshake(struct lws *wsi, char *p, const char *conn1)
{
	char buf[128], hash[20], key_b64[40];
	int n;
#if !defined(LWS_WITHOUT_EXTENSIONS)
	const struct lws_extension *ext;
	int ext_count = 0;
#endif

	/*
	 * create the random key
	 */
	n = lws_get_random(wsi->context, hash, 16);
	if (n != 16) {
		lwsl_err("Unable to read from random dev %s\n",
			 SYSTEM_RANDOM_FILEPATH);
		return NULL;
	}

	lws_b64_encode_string(hash, 16, key_b64, sizeof(key_b64));

	p += sprintf(p, "Upgrade: websocket\x0d\x0a"
			"Connection: %sUpgrade\x0d\x0a"
			"Sec-WebSocket-Key: ", conn1);
	strcpy(p, key_b64);
	p += strlen(key_b64);
	p += sprintf(p, "\x0d\x0a");
	if (lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS))
		p += sprintf(p, "Sec-WebSocket-Protocol: %s\x0d\x0a",
		     lws_hdr_simple_ptr(wsi,
				     _WSI_TOKEN_CLIENT_SENT_PROTOCOLS));

	/* tell the server what extensions we could support */

#if !defined(LWS_WITHOUT_EXTENSIONS)
	ext = wsi->vhost->ws.extensions;
	while (ext && ext->callback) {

		n = wsi->vhost->protocols[0].callback(wsi,
			LWS_CALLBACK_CLIENT_CONFIRM_EXTENSION_SUPPORTED,
				wsi->user_space, (char *)ext->name, 0);

		/*
		 * zero return from callback means go ahead and allow
		 * the extension, it's what we get if the callback is
		 * unhandled
		 */

		if (n) {
			ext++;
			continue;
		}

		/* apply it */

		if (ext_count)
			*p++ = ',';
		else
			p += sprintf(p, "Sec-WebSocket-Extensions: ");
		p += sprintf(p, "%s", ext->client_offer);
		ext_count++;

		ext++;
	}
	if (ext_count)
		p += sprintf(p, "\x0d\x0a");
#endif

	if (wsi->ws->ietf_spec_revision)
		p += sprintf(p, "Sec-WebSocket-Version: %d\x0d\x0a",
			     wsi->ws->ietf_spec_revision);

	/* prepare the expected server accept response */

	key_b64[39] = '\0'; /* enforce composed length below buf sizeof */
	n = sprintf(buf, "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11",
			  key_b64);

	lws_SHA1((unsigned char *)buf, n, (unsigned char *)hash);

	lws_b64_encode_string(hash, 20,
		  wsi->http.ah->initial_handshake_hash_base64,
		  sizeof(wsi->http.ah->initial_handshake_hash_base64));

	return p;
}

int
lws_client_ws_upgrade(struct lws *wsi, const char **cce)
{
	struct lws_context *context = wsi->context;
	struct lws_tokenize ts;
	int n, len, okay = 0;
	lws_tokenize_elem e;
	char *p, buf[64];
	const char *pc;
#if !defined(LWS_WITHOUT_EXTENSIONS)
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	char *sb = (char *)&pt->serv_buf[0];
	const struct lws_ext_options *opts;
	const struct lws_extension *ext;
	char ext_name[128];
	const char *c, *a;
	int more = 1;
	char ignore;
#endif

	if (wsi->client_h2_substream) {/* !!! client ws-over-h2 not there yet */
		lwsl_warn("%s: client ws-over-h2 upgrade not supported yet\n",
			  __func__);
		*cce = "HS: h2 / ws upgrade unsupported";
		goto bail3;
	}

	if (wsi->http.ah->http_response == 401) {
		lwsl_warn(
		       "lws_client_handshake: got bad HTTP response '%d'\n",
		       wsi->http.ah->http_response);
		*cce = "HS: ws upgrade unauthorized";
		goto bail3;
	}

	if (wsi->http.ah->http_response != 101) {
		lwsl_warn(
		       "lws_client_handshake: got bad HTTP response '%d'\n",
		       wsi->http.ah->http_response);
		*cce = "HS: ws upgrade response not 101";
		goto bail3;
	}

	if (lws_hdr_total_length(wsi, WSI_TOKEN_ACCEPT) == 0) {
		lwsl_info("no ACCEPT\n");
		*cce = "HS: ACCEPT missing";
		goto bail3;
	}

	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_UPGRADE);
	if (!p) {
		lwsl_info("no UPGRADE\n");
		*cce = "HS: UPGRADE missing";
		goto bail3;
	}
	strtolower(p);
	if (strcmp(p, "websocket")) {
		lwsl_warn(
		      "lws_client_handshake: got bad Upgrade header '%s'\n", p);
		*cce = "HS: Upgrade to something other than websocket";
		goto bail3;
	}

	/* connection: must have "upgrade" */

	lws_tokenize_init(&ts, buf, LWS_TOKENIZE_F_COMMA_SEP_LIST |
				    LWS_TOKENIZE_F_MINUS_NONTERM);
	ts.len = lws_hdr_copy(wsi, buf, sizeof(buf) - 1, WSI_TOKEN_CONNECTION);
	if (ts.len <= 0) /* won't fit, or absent */
		goto bad_conn_format;

	do {
		e = lws_tokenize(&ts);
		switch (e) {
		case LWS_TOKZE_TOKEN:
			if (!strcasecmp(ts.token, "upgrade"))
				e = LWS_TOKZE_ENDED;
			break;

		case LWS_TOKZE_DELIMITER:
			break;

		default: /* includes ENDED */
bad_conn_format:
			*cce = "HS: UPGRADE malformed";
			goto bail3;
		}
	} while (e > 0);

	pc = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS);
	if (!pc) {
		lwsl_parser("lws_client_int_s_hs: no protocol list\n");
	} else
		lwsl_parser("lws_client_int_s_hs: protocol list '%s'\n", pc);

	/*
	 * confirm the protocol the server wants to talk was in the list
	 * of protocols we offered
	 */

	len = lws_hdr_total_length(wsi, WSI_TOKEN_PROTOCOL);
	if (!len) {
		lwsl_info("%s: WSI_TOKEN_PROTOCOL is null\n", __func__);
		/*
		 * no protocol name to work from,
		 * default to first protocol
		 */
		n = 0;
		wsi->protocol = &wsi->vhost->protocols[0];
		goto check_extensions;
	}

	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_PROTOCOL);
	len = (int)strlen(p);

	while (pc && *pc && !okay) {
		if (!strncmp(pc, p, len) &&
		    (pc[len] == ',' || pc[len] == '\0')) {
			okay = 1;
			continue;
		}
		while (*pc && *pc++ != ',')
			;
		while (*pc == ' ')
			pc++;
	}

	if (!okay) {
		lwsl_info("%s: got bad protocol %s\n", __func__, p);
		*cce = "HS: PROTOCOL malformed";
		goto bail2;
	}

	/*
	 * identify the selected protocol struct and set it
	 */
	n = 0;
	/* keep client connection pre-bound protocol */
	if (!lwsi_role_client(wsi))
		wsi->protocol = NULL;

	while (wsi->vhost->protocols[n].callback) {
		if (!wsi->protocol &&
		    strcmp(p, wsi->vhost->protocols[n].name) == 0) {
			wsi->protocol = &wsi->vhost->protocols[n];
			break;
		}
		n++;
	}

	if (!wsi->vhost->protocols[n].callback) { /* no match */
		/* if server, that's already fatal */
		if (!lwsi_role_client(wsi)) {
			lwsl_info("%s: fail protocol %s\n", __func__, p);
			*cce = "HS: Cannot match protocol";
			goto bail2;
		}

		/* for client, find the index of our pre-bound protocol */

		n = 0;
		while (wsi->vhost->protocols[n].callback) {
			if (wsi->protocol && strcmp(wsi->protocol->name,
				   wsi->vhost->protocols[n].name) == 0) {
				wsi->protocol = &wsi->vhost->protocols[n];
				break;
			}
			n++;
		}

		if (!wsi->vhost->protocols[n].callback) {
			if (wsi->protocol)
				lwsl_err("Failed to match protocol %s\n",
						wsi->protocol->name);
			else
				lwsl_err("No protocol on client\n");
			goto bail2;
		}
	}

	lwsl_debug("Selected protocol %s\n", wsi->protocol->name);

check_extensions:
	/*
	 * stitch protocol choice into the vh protocol linked list
	 * We always insert ourselves at the start of the list
	 *
	 * X <-> B
	 * X <-> pAn <-> pB
	 */

	lws_same_vh_protocol_insert(wsi, n);

#if !defined(LWS_WITHOUT_EXTENSIONS)
	/* instantiate the accepted extensions */

	if (!lws_hdr_total_length(wsi, WSI_TOKEN_EXTENSIONS)) {
		lwsl_ext("no client extensions allowed by server\n");
		goto check_accept;
	}

	/*
	 * break down the list of server accepted extensions
	 * and go through matching them or identifying bogons
	 */

	if (lws_hdr_copy(wsi, sb, context->pt_serv_buf_size,
			 WSI_TOKEN_EXTENSIONS) < 0) {
		lwsl_warn("ext list from server failed to copy\n");
		*cce = "HS: EXT: list too big";
		goto bail2;
	}

	c = sb;
	n = 0;
	ignore = 0;
	a = NULL;
	while (more) {

		if (*c && (*c != ',' && *c != '\t')) {
			if (*c == ';') {
				ignore = 1;
				if (!a)
					a = c + 1;
			}
			if (ignore || *c == ' ') {
				c++;
				continue;
			}

			ext_name[n] = *c++;
			if (n < (int)sizeof(ext_name) - 1)
				n++;
			continue;
		}
		ext_name[n] = '\0';
		ignore = 0;
		if (!*c)
			more = 0;
		else {
			c++;
			if (!n)
				continue;
		}

		/* check we actually support it */

		lwsl_notice("checking client ext %s\n", ext_name);

		n = 0;
		ext = wsi->vhost->ws.extensions;
		while (ext && ext->callback) {
			if (strcmp(ext_name, ext->name)) {
				ext++;
				continue;
			}

			n = 1;
			lwsl_notice("instantiating client ext %s\n", ext_name);

			/* instantiate the extension on this conn */

			wsi->ws->active_extensions[wsi->ws->count_act_ext] = ext;

			/* allow him to construct his ext instance */

			if (ext->callback(lws_get_context(wsi), ext, wsi,
				   LWS_EXT_CB_CLIENT_CONSTRUCT,
				   (void *)&wsi->ws->act_ext_user[
				                        wsi->ws->count_act_ext],
				   (void *)&opts, 0)) {
				lwsl_info(" ext %s failed construction\n",
					  ext_name);
				ext++;
				continue;
			}

			/*
			 * allow the user code to override ext defaults if it
			 * wants to
			 */
			ext_name[0] = '\0';
			if (user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_WS_EXT_DEFAULTS,
					(char *)ext->name, ext_name,
					sizeof(ext_name))) {
				*cce = "HS: EXT: failed setting defaults";
				goto bail2;
			}

			if (ext_name[0] &&
			    lws_ext_parse_options(ext, wsi,
					          wsi->ws->act_ext_user[
						        wsi->ws->count_act_ext],
					          opts, ext_name,
						  (int)strlen(ext_name))) {
				lwsl_err("%s: unable to parse user defaults '%s'",
					 __func__, ext_name);
				*cce = "HS: EXT: failed parsing defaults";
				goto bail2;
			}

			/*
			 * give the extension the server options
			 */
			if (a && lws_ext_parse_options(ext, wsi,
					wsi->ws->act_ext_user[
					                wsi->ws->count_act_ext],
					opts, a, lws_ptr_diff(c, a))) {
				lwsl_err("%s: unable to parse remote def '%s'",
					 __func__, a);
				*cce = "HS: EXT: failed parsing options";
				goto bail2;
			}

			if (ext->callback(lws_get_context(wsi), ext, wsi,
					LWS_EXT_CB_OPTION_CONFIRM,
				      wsi->ws->act_ext_user[wsi->ws->count_act_ext],
				      NULL, 0)) {
				lwsl_err("%s: ext %s rejects server options %s",
					 __func__, ext->name, a);
				*cce = "HS: EXT: Rejects server options";
				goto bail2;
			}

			wsi->ws->count_act_ext++;

			ext++;
		}

		if (n == 0) {
			lwsl_warn("Unknown ext '%s'!\n", ext_name);
			*cce = "HS: EXT: unknown ext";
			goto bail2;
		}

		a = NULL;
		n = 0;
	}

check_accept:
#endif

	/*
	 * Confirm his accept token is the one we precomputed
	 */

	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_ACCEPT);
	if (strcmp(p, wsi->http.ah->initial_handshake_hash_base64)) {
		lwsl_warn("lws_client_int_s_hs: accept '%s' wrong vs '%s'\n", p,
				  wsi->http.ah->initial_handshake_hash_base64);
		*cce = "HS: Accept hash wrong";
		goto bail2;
	}

	/* allocate the per-connection user memory (if any) */
	if (lws_ensure_user_space(wsi)) {
		lwsl_err("Problem allocating wsi user mem\n");
		*cce = "HS: OOM";
		goto bail2;
	}

	/*
	 * we seem to be good to go, give client last chance to check
	 * headers and OK it
	 */
	if (wsi->protocol->callback(wsi,
				    LWS_CALLBACK_CLIENT_FILTER_PRE_ESTABLISH,
				    wsi->user_space, NULL, 0)) {
		*cce = "HS: Rejected by filter cb";
		goto bail2;
	}

	/* clear his proxy connection timeout */
	lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);

	/* free up his parsing allocations */
	lws_header_table_detach(wsi, 0);

	lws_role_transition(wsi, LWSIFR_CLIENT, LRS_ESTABLISHED,
			    &role_ops_ws);
	lws_restart_ws_ping_pong_timer(wsi);

	wsi->rxflow_change_to = LWS_RXFLOW_ALLOW;

	/*
	 * create the frame buffer for this connection according to the
	 * size mentioned in the protocol definition.  If 0 there, then
	 * use a big default for compatibility
	 */
	n = (int)wsi->protocol->rx_buffer_size;
	if (!n)
		n = context->pt_serv_buf_size;
	n += LWS_PRE;
	wsi->ws->rx_ubuf = lws_malloc(n + 4 /* 0x0000ffff zlib */,
				"client frame buffer");
	if (!wsi->ws->rx_ubuf) {
		lwsl_err("Out of Mem allocating rx buffer %d\n", n);
		*cce = "HS: OOM";
		goto bail2;
	}
	wsi->ws->rx_ubuf_alloc = n;
	lwsl_info("Allocating client RX buffer %d\n", n);

#if !defined(LWS_WITH_ESP32)
	if (setsockopt(wsi->desc.sockfd, SOL_SOCKET, SO_SNDBUF,
		       (const char *)&n, sizeof n)) {
		lwsl_warn("Failed to set SNDBUF to %d", n);
		*cce = "HS: SO_SNDBUF failed";
		goto bail3;
	}
#endif

	lwsl_debug("handshake OK for protocol %s\n", wsi->protocol->name);

	/* call him back to inform him he is up */

	if (wsi->protocol->callback(wsi, LWS_CALLBACK_CLIENT_ESTABLISHED,
				    wsi->user_space, NULL, 0)) {
		*cce = "HS: Rejected at CLIENT_ESTABLISHED";
		goto bail3;
	}

	return 0;

bail3:
	return 3;

bail2:
	return 2;
}
