#include "private-libwebsockets.h"

static int
lws_getaddrinfo46(struct lws *wsi, const char *ads, struct addrinfo **result)
{
	struct addrinfo hints;

	memset(&hints, 0, sizeof(hints));
	*result = NULL;

#ifdef LWS_WITH_IPV6
	if (wsi->ipv6) {

#if !defined(__ANDROID__)
		hints.ai_family = AF_INET6;
		hints.ai_flags = AI_V4MAPPED;
#endif
	} else
#endif
	{
		hints.ai_family = PF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_flags = AI_CANONNAME;
	}

	return getaddrinfo(ads, NULL, &hints, result);
}

struct lws *
lws_client_connect_2(struct lws *wsi)
{
	sockaddr46 sa46;
	struct addrinfo *result;
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	struct lws_pollfd pfd;
	const char *cce = "", *iface;
	int n, port;
	ssize_t plen = 0;
	const char *ads;
#ifdef LWS_WITH_IPV6
	char ipv6only = lws_check_opt(wsi->vhost->options,
			LWS_SERVER_OPTION_IPV6_V6ONLY_MODIFY |
			LWS_SERVER_OPTION_IPV6_V6ONLY_VALUE);

#if defined(__ANDROID__)
	ipv6only = 0;
#endif
#endif

	lwsl_client("%s\n", __func__);

	if (!wsi->u.hdr.ah) {
		cce = "ah was NULL at cc2";
		lwsl_err("%s\n", cce);
		goto oom4;
	}

	/*
	 * start off allowing ipv6 on connection if vhost allows it
	 */
	wsi->ipv6 = LWS_IPV6_ENABLED(wsi->vhost);

	/* Decide what it is we need to connect to:
	 *
	 * Priority 1: connect to http proxy */

	if (wsi->vhost->http_proxy_port) {
		plen = sprintf((char *)pt->serv_buf,
			"CONNECT %s:%u HTTP/1.0\x0d\x0a"
			"User-agent: libwebsockets\x0d\x0a",
			lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS),
			wsi->c_port);

		if (wsi->vhost->proxy_basic_auth_token[0])
			plen += sprintf((char *)pt->serv_buf + plen,
					"Proxy-authorization: basic %s\x0d\x0a",
					wsi->vhost->proxy_basic_auth_token);

		plen += sprintf((char *)pt->serv_buf + plen, "\x0d\x0a");
		ads = wsi->vhost->http_proxy_address;
		port = wsi->vhost->http_proxy_port;

#if defined(LWS_WITH_SOCKS5)

	/* Priority 2: Connect to SOCK5 Proxy */

	} else if (wsi->vhost->socks_proxy_port) {
		socks_generate_msg(wsi, SOCKS_MSG_GREETING, &plen);
		lwsl_client("Sending SOCKS Greeting\n");
		ads = wsi->vhost->socks_proxy_address;
		port = wsi->vhost->socks_proxy_port;
#endif
	} else {

		/* Priority 3: Connect directly */

		ads = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS);
		port = wsi->c_port;
	}

	/*
	 * prepare the actual connection
	 * to whatever we decided to connect to
	 */

       lwsl_notice("%s: %p: address %s\n", __func__, wsi, ads);

       n = lws_getaddrinfo46(wsi, ads, &result);

#ifdef LWS_WITH_IPV6
	if (wsi->ipv6) {

		if (n) {
			/* lws_getaddrinfo46 failed, there is no usable result */
			lwsl_notice("%s: lws_getaddrinfo46 failed %d\n",
					__func__, n);
			cce = "ipv6 lws_getaddrinfo46 failed";
			goto oom4;
		}

		memset(&sa46, 0, sizeof(sa46));

		sa46.sa6.sin6_family = AF_INET6;
		switch (result->ai_family) {
		case AF_INET:
			if (ipv6only)
				break;
			/* map IPv4 to IPv6 */
			bzero((char *)&sa46.sa6.sin6_addr,
						sizeof(sa46.sa6.sin6_addr));
			sa46.sa6.sin6_addr.s6_addr[10] = 0xff;
			sa46.sa6.sin6_addr.s6_addr[11] = 0xff;
			memcpy(&sa46.sa6.sin6_addr.s6_addr[12],
				&((struct sockaddr_in *)result->ai_addr)->sin_addr,
							sizeof(struct in_addr));
			lwsl_notice("uplevelling AF_INET to AF_INET6\n");
			break;

		case AF_INET6:
			memcpy(&sa46.sa6.sin6_addr,
			  &((struct sockaddr_in6 *)result->ai_addr)->sin6_addr,
						sizeof(struct in6_addr));
			sa46.sa6.sin6_scope_id = ((struct sockaddr_in6 *)result->ai_addr)->sin6_scope_id;
			sa46.sa6.sin6_flowinfo = ((struct sockaddr_in6 *)result->ai_addr)->sin6_flowinfo;
			break;
		default:
			lwsl_err("Unknown address family\n");
			freeaddrinfo(result);
			cce = "unknown address family";
			goto oom4;
		}
	} else
#endif /* use ipv6 */

	/* use ipv4 */
	{
		void *p = NULL;

		if (!n) {
			struct addrinfo *res = result;

			/* pick the first AF_INET (IPv4) result */

			while (!p && res) {
				switch (res->ai_family) {
				case AF_INET:
					p = &((struct sockaddr_in *)res->ai_addr)->sin_addr;
					break;
				}

				res = res->ai_next;
			}
#if defined(LWS_FALLBACK_GETHOSTBYNAME)
		} else if (n == EAI_SYSTEM) {
			struct hostent *host;

			lwsl_info("getaddrinfo (ipv4) failed, trying gethostbyname\n");
			host = gethostbyname(ads);
			if (host) {
				p = host->h_addr;
			} else {
				lwsl_err("gethostbyname failed\n");
				cce = "gethostbyname (ipv4) failed";
				goto oom4;
			}
#endif
		} else {
			lwsl_err("getaddrinfo failed\n");
			cce = "getaddrinfo failed";
			goto oom4;
		}

		if (!p) {
			if (result)
				freeaddrinfo(result);
			lwsl_err("Couldn't identify address\n");
			cce = "unable to lookup address";
			goto oom4;
		}

		sa46.sa4.sin_family = AF_INET;
		sa46.sa4.sin_addr = *((struct in_addr *)p);
		bzero(&sa46.sa4.sin_zero, 8);
	}

	if (result)
		freeaddrinfo(result);

	/* now we decided on ipv4 or ipv6, set the port */

	if (!lws_socket_is_valid(wsi->desc.sockfd)) {

#if defined(LWS_WITH_LIBUV)
		if (LWS_LIBUV_ENABLED(context))
			if (lws_libuv_check_watcher_active(wsi)) {
				lwsl_warn("Waiting for libuv watcher to close\n");
				cce = "waiting for libuv watcher to close";
				goto oom4;
			}
#endif

#ifdef LWS_WITH_IPV6
		if (wsi->ipv6)
			wsi->desc.sockfd = socket(AF_INET6, SOCK_STREAM, 0);
		else
#endif
			wsi->desc.sockfd = socket(AF_INET, SOCK_STREAM, 0);

		if (!lws_socket_is_valid(wsi->desc.sockfd)) {
			lwsl_warn("Unable to open socket\n");
			cce = "unable to open socket";
			goto oom4;
		}

		if (lws_plat_set_socket_options(wsi->vhost, wsi->desc.sockfd)) {
			lwsl_err("Failed to set wsi socket options\n");
			compatible_close(wsi->desc.sockfd);
			cce = "set socket opts failed";
			goto oom4;
		}

		wsi->mode = LWSCM_WSCL_WAITING_CONNECT;

		lws_libev_accept(wsi, wsi->desc);
		lws_libuv_accept(wsi, wsi->desc);
		lws_libevent_accept(wsi, wsi->desc);

		if (insert_wsi_socket_into_fds(context, wsi)) {
			compatible_close(wsi->desc.sockfd);
			cce = "insert wsi failed";
			goto oom4;
		}

		lws_change_pollfd(wsi, 0, LWS_POLLIN);

		/*
		 * past here, we can't simply free the structs as error
		 * handling as oom4 does.  We have to run the whole close flow.
		 */

		if (!wsi->protocol)
			wsi->protocol = &wsi->vhost->protocols[0];

		wsi->protocol->callback(wsi, LWS_CALLBACK_WSI_CREATE,
					wsi->user_space, NULL, 0);

		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_CONNECT_RESPONSE,
				AWAITING_TIMEOUT);

		iface = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_IFACE);

		if (iface) {
			n = lws_socket_bind(wsi->vhost, wsi->desc.sockfd, 0, iface);
			if (n < 0) {
				cce = "unable to bind socket";
				goto failed;
			}
		}
	}

#ifdef LWS_WITH_IPV6
	if (wsi->ipv6) {
		sa46.sa6.sin6_port = htons(port);
		n = sizeof(struct sockaddr_in6);
	} else
#endif
	{
		sa46.sa4.sin_port = htons(port);
		n = sizeof(struct sockaddr);
	}

	if (connect(wsi->desc.sockfd, (const struct sockaddr *)&sa46, n) == -1 ||
	    LWS_ERRNO == LWS_EISCONN) {
		if (LWS_ERRNO == LWS_EALREADY ||
		    LWS_ERRNO == LWS_EINPROGRESS ||
		    LWS_ERRNO == LWS_EWOULDBLOCK
#ifdef _WIN32
			|| LWS_ERRNO == WSAEINVAL
#endif
		) {
			lwsl_client("nonblocking connect retry (errno = %d)\n",
				    LWS_ERRNO);

			if (lws_plat_check_connection_error(wsi)) {
				cce = "socket connect failed";
				goto failed;
			}

			/*
			 * must do specifically a POLLOUT poll to hear
			 * about the connect completion
			 */
			if (lws_change_pollfd(wsi, 0, LWS_POLLOUT)) {
				cce = "POLLOUT set failed";
				goto failed;
			}

			return wsi;
		}

		if (LWS_ERRNO != LWS_EISCONN) {
			lwsl_notice("Connect failed errno=%d\n", LWS_ERRNO);
			cce = "connect failed";
			goto failed;
		}
	}

	lwsl_client("connected\n");

	/* we are connected to server, or proxy */

	/* http proxy */
	if (wsi->vhost->http_proxy_port) {

		/*
		 * OK from now on we talk via the proxy, so connect to that
		 *
		 * (will overwrite existing pointer,
		 * leaving old string/frag there but unreferenced)
		 */
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS,
					  wsi->vhost->http_proxy_address))
			goto failed;
		wsi->c_port = wsi->vhost->http_proxy_port;

		n = send(wsi->desc.sockfd, (char *)pt->serv_buf, plen,
			 MSG_NOSIGNAL);
		if (n < 0) {
			lwsl_debug("ERROR writing to proxy socket\n");
			cce = "proxy write failed";
			goto failed;
		}

		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_PROXY_RESPONSE,
				AWAITING_TIMEOUT);

		wsi->mode = LWSCM_WSCL_WAITING_PROXY_REPLY;

		return wsi;
	}
#if defined(LWS_WITH_SOCKS5)
	/* socks proxy */
	else if (wsi->vhost->socks_proxy_port) {
		n = send(wsi->desc.sockfd, (char *)pt->serv_buf, plen,
			 MSG_NOSIGNAL);
		if (n < 0) {
			lwsl_debug("ERROR writing socks greeting\n");
			cce = "socks write failed";
			goto failed;
		}

		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_SOCKS_GREETING_REPLY,
				AWAITING_TIMEOUT);

		wsi->mode = LWSCM_WSCL_WAITING_SOCKS_GREETING_REPLY;

		return wsi;
	}
#endif

	/*
	 * provoke service to issue the handshake directly
	 * we need to do it this way because in the proxy case, this is the
	 * next state and executed only if and when we get a good proxy
	 * response inside the state machine... but notice in SSL case this
	 * may not have sent anything yet with 0 return, and won't until some
	 * many retries from main loop.  To stop that becoming endless,
	 * cover with a timeout.
	 */

	lws_set_timeout(wsi, PENDING_TIMEOUT_SENT_CLIENT_HANDSHAKE,
			AWAITING_TIMEOUT);

	wsi->mode = LWSCM_WSCL_ISSUE_HANDSHAKE;
	pfd.fd = wsi->desc.sockfd;
	pfd.events = LWS_POLLIN;
	pfd.revents = LWS_POLLIN;

	n = lws_service_fd(context, &pfd);
	if (n < 0) {
		cce = "first service failed";
		goto failed;
	}
	if (n) /* returns 1 on failure after closing wsi */
		return NULL;

	return wsi;

oom4:
	/* we're closing, losing some rx is OK */
	lws_header_table_force_to_detachable_state(wsi);

	if (wsi->mode == LWSCM_HTTP_CLIENT ||
	    wsi->mode == LWSCM_HTTP_CLIENT_ACCEPTED ||
	    wsi->mode == LWSCM_WSCL_WAITING_CONNECT) {
		wsi->vhost->protocols[0].callback(wsi,
			LWS_CALLBACK_CLIENT_CONNECTION_ERROR,
			wsi->user_space, (void *)cce, strlen(cce));
		wsi->already_did_cce = 1;
	}
	/* take care that we might be inserted in fds already */
	if (wsi->position_in_fds_table != -1)
		goto failed1;
	lws_remove_from_timeout_list(wsi);
	lws_header_table_detach(wsi, 0);
	lws_free(wsi);

	return NULL;

failed:
	wsi->vhost->protocols[0].callback(wsi,
		LWS_CALLBACK_CLIENT_CONNECTION_ERROR,
		wsi->user_space, (void *)cce, strlen(cce));
	wsi->already_did_cce = 1;
failed1:
	lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS);

	return NULL;
}

/**
 * lws_client_reset() - retarget a connected wsi to start over with a new connection (ie, redirect)
 *			this only works if still in HTTP, ie, not upgraded yet
 * wsi:		connection to reset
 * address:	network address of the new server
 * port:	port to connect to
 * path:	uri path to connect to on the new server
 * host:	host header to send to the new server
 */
LWS_VISIBLE struct lws *
lws_client_reset(struct lws **pwsi, int ssl, const char *address, int port,
		 const char *path, const char *host)
{
	char origin[300] = "", protocol[300] = "", method[32] = "", iface[16] = "", *p;
	struct lws *wsi = *pwsi;

	if (wsi->redirects == 3) {
		lwsl_err("%s: Too many redirects\n", __func__);
		return NULL;
	}
	wsi->redirects++;

	p = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_ORIGIN);
	if (p)
		strncpy(origin, p, sizeof(origin) - 1);

	p = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS);
	if (p)
		strncpy(protocol, p, sizeof(protocol) - 1);

	p = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_METHOD);
	if (p)
		strncpy(method, p, sizeof(method) - 1);

	p = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_IFACE);
	if (p)
		strncpy(method, p, sizeof(iface) - 1);

	lwsl_info("redirect ads='%s', port=%d, path='%s', ssl = %d\n",
		   address, port, path, ssl);

	/* close the connection by hand */

#ifdef LWS_OPENSSL_SUPPORT
	lws_ssl_close(wsi);
#endif

#ifdef LWS_WITH_LIBUV
	if (LWS_LIBUV_ENABLED(wsi->context)) {
		lwsl_debug("%s: lws_libuv_closehandle: wsi %p\n", __func__, wsi);
		/*
		 * libuv has to do his own close handle processing asynchronously
		 * but once it starts we can do everything else synchronously,
		 * including trash wsi->desc.sockfd since it took a copy.
		 *
		 * When it completes it will call compatible_close()
		 */
		lws_libuv_closehandle_manually(wsi);
	} else
#else
	compatible_close(wsi->desc.sockfd);
#endif

	remove_wsi_socket_from_fds(wsi);

#ifdef LWS_OPENSSL_SUPPORT
	wsi->use_ssl = ssl;
#else
	if (ssl) {
		lwsl_err("%s: not configured for ssl\n", __func__);
		return NULL;
	}
#endif

	wsi->desc.sockfd = LWS_SOCK_INVALID;
	wsi->state = LWSS_CLIENT_UNCONNECTED;
	wsi->protocol = NULL;
	wsi->pending_timeout = NO_PENDING_TIMEOUT;
	wsi->c_port = port;
	wsi->hdr_parsing_completed = 0;
	_lws_header_table_reset(wsi->u.hdr.ah);

	if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS, address))
		return NULL;

	if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_HOST, host))
		return NULL;

	if (origin[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_ORIGIN,
					  origin))
			return NULL;
	if (protocol[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS,
					  protocol))
			return NULL;
	if (method[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_METHOD,
					  method))
			return NULL;

	if (iface[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_IFACE,
					  iface))
			return NULL;

	origin[0] = '/';
	strncpy(&origin[1], path, sizeof(origin) - 2);
	if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_URI, origin))
		return NULL;

	*pwsi = lws_client_connect_2(wsi);

	return *pwsi;
}

#ifdef LWS_WITH_HTTP_PROXY
static hubbub_error
html_parser_cb(const hubbub_token *token, void *pw)
{
	struct lws_rewrite *r = (struct lws_rewrite *)pw;
	char buf[1024], *start = buf + LWS_PRE, *p = start,
	     *end = &buf[sizeof(buf) - 1];
	size_t i;

	switch (token->type) {
	case HUBBUB_TOKEN_DOCTYPE:

		p += lws_snprintf(p, end - p, "<!DOCTYPE %.*s %s ",
				(int) token->data.doctype.name.len,
				token->data.doctype.name.ptr,
				token->data.doctype.force_quirks ?
						"(force-quirks) " : "");

		if (token->data.doctype.public_missing)
			lwsl_debug("\tpublic: missing\n");
		else
			p += lws_snprintf(p, end - p, "PUBLIC \"%.*s\"\n",
				(int) token->data.doctype.public_id.len,
				token->data.doctype.public_id.ptr);

		if (token->data.doctype.system_missing)
			lwsl_debug("\tsystem: missing\n");
		else
			p += lws_snprintf(p, end - p, " \"%.*s\">\n",
				(int) token->data.doctype.system_id.len,
				token->data.doctype.system_id.ptr);

		break;
	case HUBBUB_TOKEN_START_TAG:
		p += lws_snprintf(p, end - p, "<%.*s", (int)token->data.tag.name.len,
				token->data.tag.name.ptr);

/*				(token->data.tag.self_closing) ?
						"(self-closing) " : "",
				(token->data.tag.n_attributes > 0) ?
						"attributes:" : "");
*/
		for (i = 0; i < token->data.tag.n_attributes; i++) {
			if (!hstrcmp(&token->data.tag.attributes[i].name, "href", 4) ||
			    !hstrcmp(&token->data.tag.attributes[i].name, "action", 6) ||
			    !hstrcmp(&token->data.tag.attributes[i].name, "src", 3)) {
				const char *pp = (const char *)token->data.tag.attributes[i].value.ptr;
				int plen = (int) token->data.tag.attributes[i].value.len;

				if (strncmp(pp, "http:", 5) && strncmp(pp, "https:", 6)) {

					if (!hstrcmp(&token->data.tag.attributes[i].value,
						     r->from, r->from_len)) {
						pp += r->from_len;
						plen -= r->from_len;
					}
					p += lws_snprintf(p, end - p, " %.*s=\"%s/%.*s\"",
					       (int) token->data.tag.attributes[i].name.len,
					       token->data.tag.attributes[i].name.ptr,
					       r->to, plen, pp);
					continue;
				}
			}

			p += lws_snprintf(p, end - p, " %.*s=\"%.*s\"",
				(int) token->data.tag.attributes[i].name.len,
				token->data.tag.attributes[i].name.ptr,
				(int) token->data.tag.attributes[i].value.len,
				token->data.tag.attributes[i].value.ptr);
		}
		p += lws_snprintf(p, end - p, ">");
		break;
	case HUBBUB_TOKEN_END_TAG:
		p += lws_snprintf(p, end - p, "</%.*s", (int) token->data.tag.name.len,
				token->data.tag.name.ptr);
/*
				(token->data.tag.self_closing) ?
						"(self-closing) " : "",
				(token->data.tag.n_attributes > 0) ?
						"attributes:" : "");
*/
		for (i = 0; i < token->data.tag.n_attributes; i++) {
			p += lws_snprintf(p, end - p, " %.*s='%.*s'\n",
				(int) token->data.tag.attributes[i].name.len,
				token->data.tag.attributes[i].name.ptr,
				(int) token->data.tag.attributes[i].value.len,
				token->data.tag.attributes[i].value.ptr);
		}
		p += lws_snprintf(p, end - p, ">");
		break;
	case HUBBUB_TOKEN_COMMENT:
		p += lws_snprintf(p, end - p, "<!-- %.*s -->\n",
				(int) token->data.comment.len,
				token->data.comment.ptr);
		break;
	case HUBBUB_TOKEN_CHARACTER:
		if (token->data.character.len == 1) {
			if (*token->data.character.ptr == '<') {
				p += lws_snprintf(p, end - p, "&lt;");
				break;
			}
			if (*token->data.character.ptr == '>') {
				p += lws_snprintf(p, end - p, "&gt;");
				break;
			}
			if (*token->data.character.ptr == '&') {
				p += lws_snprintf(p, end - p, "&amp;");
				break;
			}
		}

		p += lws_snprintf(p, end - p, "%.*s", (int) token->data.character.len,
				token->data.character.ptr);
		break;
	case HUBBUB_TOKEN_EOF:
		p += lws_snprintf(p, end - p, "\n");
		break;
	}

	if (user_callback_handle_rxflow(r->wsi->protocol->callback,
			r->wsi, LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ,
			r->wsi->user_space, start, p - start))
		return -1;

	return HUBBUB_OK;
}
#endif

LWS_VISIBLE struct lws *
lws_client_connect_via_info(struct lws_client_connect_info *i)
{
	struct lws *wsi;
	int v = SPEC_LATEST_SUPPORTED;
	const struct lws_protocols *p;

	if (i->context->requested_kill)
		return NULL;

	if (!i->context->protocol_init_done)
		lws_protocol_init(i->context);

	wsi = lws_zalloc(sizeof(struct lws), "client wsi");
	if (wsi == NULL)
		goto bail;

	wsi->context = i->context;
	/* assert the mode and union status (hdr) clearly */
	lws_union_transition(wsi, LWSCM_HTTP_CLIENT);
	wsi->desc.sockfd = LWS_SOCK_INVALID;

	/* 1) fill up the wsi with stuff from the connect_info as far as it
	 * can go.  It's because not only is our connection async, we might
	 * not even be able to get ahold of an ah at this point.
	 */

	/* -1 means just use latest supported */
	if (i->ietf_version_or_minus_one != -1 && i->ietf_version_or_minus_one)
		v = i->ietf_version_or_minus_one;

	wsi->ietf_spec_revision = v;
	wsi->user_space = NULL;
	wsi->state = LWSS_CLIENT_UNCONNECTED;
	wsi->pending_timeout = NO_PENDING_TIMEOUT;
	wsi->position_in_fds_table = -1;
	wsi->c_port = i->port;
	wsi->vhost = i->vhost;
	if (!wsi->vhost)
		wsi->vhost = i->context->vhost_list;

	wsi->protocol = &wsi->vhost->protocols[0];

	/* for http[s] connection, allow protocol selection by name */

	if (i->method && i->vhost && i->protocol) {
		p = lws_vhost_name_to_protocol(i->vhost, i->protocol);
		if (p)
			wsi->protocol = p;
	}

	if (wsi && !wsi->user_space && i->userdata) {
		wsi->user_space_externally_allocated = 1;
		wsi->user_space = i->userdata;
	} else
		/* if we stay in http, we can assign the user space now,
		 * otherwise do it after the protocol negotiated
		 */
		if (i->method)
			if (lws_ensure_user_space(wsi))
				goto bail;

#ifdef LWS_OPENSSL_SUPPORT
	wsi->use_ssl = i->ssl_connection;
#else
	if (i->ssl_connection) {
		lwsl_err("libwebsockets not configured for ssl\n");
		goto bail;
	}
#endif

	/* 2) stash the things from connect_info that we can't process without
	 * an ah.  Because if no ah, we will go on the ah waiting list and
	 * process those things later (after the connect_info and maybe the
	 * things pointed to have gone out of scope.
	 */

	wsi->u.hdr.stash = lws_malloc(sizeof(*wsi->u.hdr.stash), "client stash");
	if (!wsi->u.hdr.stash) {
		lwsl_err("%s: OOM\n", __func__);
		goto bail;
	}

	wsi->u.hdr.stash->origin[0] = '\0';
	wsi->u.hdr.stash->protocol[0] = '\0';
	wsi->u.hdr.stash->method[0] = '\0';
	wsi->u.hdr.stash->iface[0] = '\0';

	strncpy(wsi->u.hdr.stash->address, i->address,
		sizeof(wsi->u.hdr.stash->address) - 1);
	strncpy(wsi->u.hdr.stash->path, i->path,
		sizeof(wsi->u.hdr.stash->path) - 1);
	strncpy(wsi->u.hdr.stash->host, i->host,
		sizeof(wsi->u.hdr.stash->host) - 1);
	if (i->origin)
		strncpy(wsi->u.hdr.stash->origin, i->origin,
			sizeof(wsi->u.hdr.stash->origin) - 1);
	if (i->protocol)
		strncpy(wsi->u.hdr.stash->protocol, i->protocol,
			sizeof(wsi->u.hdr.stash->protocol) - 1);
	if (i->method)
		strncpy(wsi->u.hdr.stash->method, i->method,
			sizeof(wsi->u.hdr.stash->method) - 1);
	if (i->iface)
		strncpy(wsi->u.hdr.stash->iface, i->iface,
			sizeof(wsi->u.hdr.stash->iface) - 1);

	wsi->u.hdr.stash->address[sizeof(wsi->u.hdr.stash->address) - 1] = '\0';
	wsi->u.hdr.stash->path[sizeof(wsi->u.hdr.stash->path) - 1] = '\0';
	wsi->u.hdr.stash->host[sizeof(wsi->u.hdr.stash->host) - 1] = '\0';
	wsi->u.hdr.stash->origin[sizeof(wsi->u.hdr.stash->origin) - 1] = '\0';
	wsi->u.hdr.stash->protocol[sizeof(wsi->u.hdr.stash->protocol) - 1] = '\0';
	wsi->u.hdr.stash->method[sizeof(wsi->u.hdr.stash->method) - 1] = '\0';
	wsi->u.hdr.stash->iface[sizeof(wsi->u.hdr.stash->iface) - 1] = '\0';

	if (i->pwsi)
		*i->pwsi = wsi;

	/* if we went on the waiting list, no probs just return the wsi
	 * when we get the ah, now or later, he will call
	 * lws_client_connect_via_info2() below.
	 */
	if (lws_header_table_attach(wsi, 0) < 0) {
		/*
		 * if we failed here, the connection is already closed
		 * and freed.
		 */
		goto bail1;
	}

	if (i->parent_wsi) {
		lwsl_info("%s: created child %p of parent %p\n", __func__,
				wsi, i->parent_wsi);
		wsi->parent = i->parent_wsi;
		wsi->sibling_list = i->parent_wsi->child_list;
		i->parent_wsi->child_list = wsi;
	}
#ifdef LWS_WITH_HTTP_PROXY
	if (i->uri_replace_to)
		wsi->rw = lws_rewrite_create(wsi, html_parser_cb,
					     i->uri_replace_from,
					     i->uri_replace_to);
#endif

	return wsi;

bail:
	lws_free(wsi);

bail1:
	if (i->pwsi)
		*i->pwsi = NULL;

	return NULL;
}

struct lws *
lws_client_connect_via_info2(struct lws *wsi)
{
	struct client_info_stash *stash = wsi->u.hdr.stash;

	if (!stash)
		return wsi;

	/*
	 * we're not necessarily in a position to action these right away,
	 * stash them... we only need during connect phase so u.hdr is fine
	 */
	if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS,
				  stash->address))
		goto bail1;

	/* these only need u.hdr lifetime as well */

	if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_URI, stash->path))
		goto bail1;

	if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_HOST, stash->host))
		goto bail1;

	if (stash->origin[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_ORIGIN,
					  stash->origin))
			goto bail1;
	/*
	 * this is a list of protocols we tell the server we're okay with
	 * stash it for later when we compare server response with it
	 */
	if (stash->protocol[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS,
					  stash->protocol))
			goto bail1;
	if (stash->method[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_METHOD,
					  stash->method))
			goto bail1;
	if (stash->iface[0])
		if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_IFACE,
					  stash->iface))
			goto bail1;

#if defined(LWS_WITH_SOCKS5)
	if (!wsi->vhost->socks_proxy_port)
		lws_free_set_NULL(wsi->u.hdr.stash);
#endif

	/*
	 * Check with each extension if it is able to route and proxy this
	 * connection for us.  For example, an extension like x-google-mux
	 * can handle this and then we don't need an actual socket for this
	 * connection.
	 */

	if (lws_ext_cb_all_exts(wsi->context, wsi,
				LWS_EXT_CB_CAN_PROXY_CLIENT_CONNECTION,
				(void *)stash->address,
				wsi->c_port) > 0) {
		lwsl_client("lws_client_connect: ext handling conn\n");

		lws_set_timeout(wsi,
			PENDING_TIMEOUT_AWAITING_EXTENSION_CONNECT_RESPONSE,
			        AWAITING_TIMEOUT);

		wsi->mode = LWSCM_WSCL_WAITING_EXTENSION_CONNECT;
		return wsi;
	}
	lwsl_client("lws_client_connect: direct conn\n");
	wsi->context->count_wsi_allocated++;

	return lws_client_connect_2(wsi);

bail1:
#if defined(LWS_WITH_SOCKS5)
	if (!wsi->vhost->socks_proxy_port)
		lws_free_set_NULL(wsi->u.hdr.stash);
#endif

	return NULL;
}

LWS_VISIBLE struct lws *
lws_client_connect_extended(struct lws_context *context, const char *address,
			    int port, int ssl_connection, const char *path,
			    const char *host, const char *origin,
			    const char *protocol, int ietf_version_or_minus_one,
			    void *userdata)
{
	struct lws_client_connect_info i;

	memset(&i, 0, sizeof(i));

	i.context = context;
	i.address = address;
	i.port = port;
	i.ssl_connection = ssl_connection;
	i.path = path;
	i.host = host;
	i.origin = origin;
	i.protocol = protocol;
	i.ietf_version_or_minus_one = ietf_version_or_minus_one;
	i.userdata = userdata;

	return lws_client_connect_via_info(&i);
}

LWS_VISIBLE struct lws *
lws_client_connect(struct lws_context *context, const char *address,
			    int port, int ssl_connection, const char *path,
			    const char *host, const char *origin,
			    const char *protocol, int ietf_version_or_minus_one)
{
	struct lws_client_connect_info i;

	memset(&i, 0, sizeof(i));

	i.context = context;
	i.address = address;
	i.port = port;
	i.ssl_connection = ssl_connection;
	i.path = path;
	i.host = host;
	i.origin = origin;
	i.protocol = protocol;
	i.ietf_version_or_minus_one = ietf_version_or_minus_one;
	i.userdata = NULL;

	return lws_client_connect_via_info(&i);
}

#if defined(LWS_WITH_SOCKS5)
void socks_generate_msg(struct lws *wsi, enum socks_msg_type type,
			ssize_t *msg_len)
{
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	ssize_t len = 0, n, passwd_len;
	short net_num;
	char *p;

	switch (type) {
	case SOCKS_MSG_GREETING:
		/* socks version, version 5 only */
		pt->serv_buf[len++] = SOCKS_VERSION_5;
		/* number of methods */
		pt->serv_buf[len++] = 2;
		/* username password method */
		pt->serv_buf[len++] = SOCKS_AUTH_USERNAME_PASSWORD;
		/* no authentication method */
		pt->serv_buf[len++] = SOCKS_AUTH_NO_AUTH;
		break;

	case SOCKS_MSG_USERNAME_PASSWORD:
		n = strlen(wsi->vhost->socks_user);
		passwd_len = strlen(wsi->vhost->socks_password);

		/* the subnegotiation version */
		pt->serv_buf[len++] = SOCKS_SUBNEGOTIATION_VERSION_1;
		/* length of the user name */
		pt->serv_buf[len++] = n;
		/* user name */
		strncpy((char *)&pt->serv_buf[len], wsi->vhost->socks_user,
			context->pt_serv_buf_size - len);
		len += n;
		/* length of the password */
		pt->serv_buf[len++] = passwd_len;
		/* password */
		strncpy((char *)&pt->serv_buf[len], wsi->vhost->socks_password,
			context->pt_serv_buf_size - len);
		len += passwd_len;
		break;

	case SOCKS_MSG_CONNECT:
		p = (char*)&net_num;

		/* socks version */
		pt->serv_buf[len++] = SOCKS_VERSION_5;
		/* socks command */
		pt->serv_buf[len++] = SOCKS_COMMAND_CONNECT;
		/* reserved */
		pt->serv_buf[len++] = 0;
		/* address type */
		pt->serv_buf[len++] = SOCKS_ATYP_DOMAINNAME;
		/* skip length, we fill it in at the end */
		n = len++;

		/* the address we tell SOCKS proxy to connect to */
		strncpy((char *)&(pt->serv_buf[len]), wsi->u.hdr.stash->address,
			context->pt_serv_buf_size - len);
		len += strlen(wsi->u.hdr.stash->address);
		net_num = htons(wsi->c_port);

		/* the port we tell SOCKS proxy to connect to */
		pt->serv_buf[len++] = p[0];
		pt->serv_buf[len++] = p[1];

		/* the length of the address, excluding port */
		pt->serv_buf[n] = strlen(wsi->u.hdr.stash->address);
		break;
		
	default:
		return;
	}

	*msg_len = len;
}
#endif
