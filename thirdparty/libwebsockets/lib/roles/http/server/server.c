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

#include "core/private.h"

const char * const method_names[] = {
	"GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE", "CONNECT", "HEAD",
#ifdef LWS_WITH_HTTP2
	":path",
#endif
	};

static const char * const intermediates[] = { "private", "public" };

/*
 * return 0: all done
 *        1: nonfatal error
 *       <0: fatal error
 *
 *       REQUIRES CONTEXT LOCK HELD
 */

int
_lws_vhost_init_server(const struct lws_context_creation_info *info,
		       struct lws_vhost *vhost)
{
	int n, opt = 1, limit = 1;
	lws_sockfd_type sockfd;
	struct lws_vhost *vh;
	struct lws *wsi;
	int m = 0, is;

	(void)method_names;
	(void)opt;

	if (info) {
		vhost->iface = info->iface;
		vhost->listen_port = info->port;
	}

	/* set up our external listening socket we serve on */

	if (vhost->listen_port == CONTEXT_PORT_NO_LISTEN ||
	    vhost->listen_port == CONTEXT_PORT_NO_LISTEN_SERVER)
		return 0;

	vh = vhost->context->vhost_list;
	while (vh) {
		if (vh->listen_port == vhost->listen_port) {
			if (((!vhost->iface && !vh->iface) ||
			    (vhost->iface && vh->iface &&
			    !strcmp(vhost->iface, vh->iface))) &&
			   vh->lserv_wsi
			) {
				lwsl_notice(" using listen skt from vhost %s\n",
					    vh->name);
				return 0;
			}
		}
		vh = vh->vhost_next;
	}

	if (vhost->iface) {
		/*
		 * let's check before we do anything else about the disposition
		 * of the interface he wants to bind to...
		 */
		is = lws_socket_bind(vhost, LWS_SOCK_INVALID, vhost->listen_port,
				vhost->iface);
		lwsl_debug("initial if check says %d\n", is);

		if (is == LWS_ITOSA_BUSY)
			/* treat as fatal */
			return -1;

deal:

		lws_start_foreach_llp(struct lws_vhost **, pv,
				      vhost->context->no_listener_vhost_list) {
			if (is >= LWS_ITOSA_USABLE && *pv == vhost) {
				/* on the list and shouldn't be: remove it */
				lwsl_debug("deferred iface: removing vh %s\n",
						(*pv)->name);
				*pv = vhost->no_listener_vhost_list;
				vhost->no_listener_vhost_list = NULL;
				goto done_list;
			}
			if (is < LWS_ITOSA_USABLE && *pv == vhost)
				goto done_list;
		} lws_end_foreach_llp(pv, no_listener_vhost_list);

		/* not on the list... */

		if (is < LWS_ITOSA_USABLE) {

			/* ... but needs to be: so add it */

			lwsl_debug("deferred iface: adding vh %s\n", vhost->name);
			vhost->no_listener_vhost_list =
					vhost->context->no_listener_vhost_list;
			vhost->context->no_listener_vhost_list = vhost;
		}

done_list:

		switch (is) {
		default:
			break;
		case LWS_ITOSA_NOT_EXIST:
			/* can't add it */
			if (info) /* first time */
				lwsl_err("VH %s: iface %s port %d DOESN'T EXIST\n",
				 vhost->name, vhost->iface, vhost->listen_port);
			return 1;
		case LWS_ITOSA_NOT_USABLE:
			/* can't add it */
			if (info) /* first time */
				lwsl_err("VH %s: iface %s port %d NOT USABLE\n",
				 vhost->name, vhost->iface, vhost->listen_port);
			return 1;
		}
	}

	(void)n;
#if defined(__linux__)
#ifdef LWS_WITH_UNIX_SOCK
	/*
	 * A Unix domain sockets cannot be bound for several times, even if we set
	 * the SO_REUSE* options on.
	 * However, fortunately, each thread is able to independently listen when
	 * running on a reasonably new Linux kernel. So we can safely assume
	 * creating just one listening socket for a multi-threaded environment won't
	 * fail in most cases.
	 */
	if (!LWS_UNIX_SOCK_ENABLED(vhost))
#endif
	limit = vhost->context->count_threads;
#endif

	for (m = 0; m < limit; m++) {
#ifdef LWS_WITH_UNIX_SOCK
		if (LWS_UNIX_SOCK_ENABLED(vhost))
			sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
		else
#endif
#ifdef LWS_WITH_IPV6
		if (LWS_IPV6_ENABLED(vhost))
			sockfd = socket(AF_INET6, SOCK_STREAM, 0);
		else
#endif
			sockfd = socket(AF_INET, SOCK_STREAM, 0);

		if (sockfd == LWS_SOCK_INVALID) {
			lwsl_err("ERROR opening socket\n");
			return 1;
		}
#if !defined(LWS_WITH_ESP32)
#if (defined(WIN32) || defined(_WIN32)) && defined(SO_EXCLUSIVEADDRUSE)
		/*
		 * only accept that we are the only listener on the port
		 * https://msdn.microsoft.com/zh-tw/library/
		 *    windows/desktop/ms740621(v=vs.85).aspx
		 *
		 * for lws, to match Linux, we default to exclusive listen
		 */
		if (!lws_check_opt(vhost->options,
				LWS_SERVER_OPTION_ALLOW_LISTEN_SHARE)) {
			if (setsockopt(sockfd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
				       (const void *)&opt, sizeof(opt)) < 0) {
				lwsl_err("reuseaddr failed\n");
				compatible_close(sockfd);
				return -1;
			}
		} else
#endif

		/*
		 * allow us to restart even if old sockets in TIME_WAIT
		 */
		if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR,
			       (const void *)&opt, sizeof(opt)) < 0) {
			lwsl_err("reuseaddr failed\n");
			compatible_close(sockfd);
			return -1;
		}

#if defined(LWS_WITH_IPV6) && defined(IPV6_V6ONLY)
		if (LWS_IPV6_ENABLED(vhost) &&
		    vhost->options & LWS_SERVER_OPTION_IPV6_V6ONLY_MODIFY) {
			int value = (vhost->options &
				LWS_SERVER_OPTION_IPV6_V6ONLY_VALUE) ? 1 : 0;
			if (setsockopt(sockfd, IPPROTO_IPV6, IPV6_V6ONLY,
				      (const void*)&value, sizeof(value)) < 0) {
				compatible_close(sockfd);
				return -1;
			}
		}
#endif

#if defined(__linux__) && defined(SO_REUSEPORT)
		/* keep coverity happy */
#if LWS_MAX_SMP > 1
		n = 1;
#else
		n = lws_check_opt(vhost->options,
				  LWS_SERVER_OPTION_ALLOW_LISTEN_SHARE);
#endif
		if (n && vhost->context->count_threads > 1)
			if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT,
					(const void *)&opt, sizeof(opt)) < 0) {
				compatible_close(sockfd);
				return -1;
			}
#endif
#endif
		lws_plat_set_socket_options(vhost, sockfd, 0);

		is = lws_socket_bind(vhost, sockfd, vhost->listen_port, vhost->iface);
		if (is == LWS_ITOSA_BUSY) {
			/* treat as fatal */
			compatible_close(sockfd);

			return -1;
		}

		/*
		 * There is a race where the network device may come up and then
		 * go away and fail here.  So correctly handle unexpected failure
		 * here despite we earlier confirmed it.
		 */
		if (is < 0) {
			lwsl_info("%s: lws_socket_bind says %d\n", __func__, is);
			compatible_close(sockfd);
			goto deal;
		}

		wsi = lws_zalloc(sizeof(struct lws), "listen wsi");
		if (wsi == NULL) {
			lwsl_err("Out of mem\n");
			goto bail;
		}

#ifdef LWS_WITH_UNIX_SOCK
		if (!LWS_UNIX_SOCK_ENABLED(vhost))
#endif
		{
			wsi->unix_skt = 1;
			vhost->listen_port = is;

			lwsl_debug("%s: lws_socket_bind says %d\n", __func__, is);
		}

		wsi->context = vhost->context;
		wsi->desc.sockfd = sockfd;
		lws_role_transition(wsi, 0, LRS_UNCONNECTED, &role_ops_listen);
		wsi->protocol = vhost->protocols;
		wsi->tsi = m;
		lws_vhost_bind_wsi(vhost, wsi);
		wsi->listener = 1;

		if (wsi->context->event_loop_ops->init_vhost_listen_wsi)
			wsi->context->event_loop_ops->init_vhost_listen_wsi(wsi);

		if (__insert_wsi_socket_into_fds(vhost->context, wsi)) {
			lwsl_notice("inserting wsi socket into fds failed\n");
			goto bail;
		}

		vhost->context->count_wsi_allocated++;
		vhost->lserv_wsi = wsi;

		n = listen(wsi->desc.sockfd, LWS_SOMAXCONN);
		if (n < 0) {
			lwsl_err("listen failed with error %d\n", LWS_ERRNO);
			vhost->lserv_wsi = NULL;
			vhost->context->count_wsi_allocated--;
			__remove_wsi_socket_from_fds(wsi);
			goto bail;
		}
	} /* for each thread able to independently listen */

	if (!lws_check_opt(vhost->context->options, LWS_SERVER_OPTION_EXPLICIT_VHOSTS)) {
#ifdef LWS_WITH_UNIX_SOCK
		if (LWS_UNIX_SOCK_ENABLED(vhost))
			lwsl_info(" Listening on \"%s\"\n", vhost->iface);
		else
#endif
			lwsl_info(" Listening on port %d\n", vhost->listen_port);
        }

	// info->port = vhost->listen_port;

	return 0;

bail:
	compatible_close(sockfd);

	return -1;
}

struct lws_vhost *
lws_select_vhost(struct lws_context *context, int port, const char *servername)
{
	struct lws_vhost *vhost = context->vhost_list;
	const char *p;
	int n, colon;

	n = (int)strlen(servername);
	colon = n;
	p = strchr(servername, ':');
	if (p)
		colon = lws_ptr_diff(p, servername);

	/* Priotity 1: first try exact matches */

	while (vhost) {
		if (port == vhost->listen_port &&
		    !strncmp(vhost->name, servername, colon)) {
			lwsl_info("SNI: Found: %s\n", servername);
			return vhost;
		}
		vhost = vhost->vhost_next;
	}

	/*
	 * Priority 2: if no exact matches, try matching *.vhost-name
	 * unintentional matches are possible but resolve to x.com for *.x.com
	 * which is reasonable.  If exact match exists we already chose it and
	 * never reach here.  SSL will still fail it if the cert doesn't allow
	 * *.x.com.
	 */
	vhost = context->vhost_list;
	while (vhost) {
		int m = (int)strlen(vhost->name);
		if (port && port == vhost->listen_port &&
		    m <= (colon - 2) &&
		    servername[colon - m - 1] == '.' &&
		    !strncmp(vhost->name, servername + colon - m, m)) {
			lwsl_info("SNI: Found %s on wildcard: %s\n",
				    servername, vhost->name);
			return vhost;
		}
		vhost = vhost->vhost_next;
	}

	/* Priority 3: match the first vhost on our port */

	vhost = context->vhost_list;
	while (vhost) {
		if (port && port == vhost->listen_port) {
			lwsl_info("%s: vhost match to %s based on port %d\n",
					__func__, vhost->name, port);
			return vhost;
		}
		vhost = vhost->vhost_next;
	}

	/* no match */

	return NULL;
}

LWS_VISIBLE LWS_EXTERN const char *
lws_get_mimetype(const char *file, const struct lws_http_mount *m)
{
	const struct lws_protocol_vhost_options *pvo = NULL;
	int n = (int)strlen(file);

	if (m)
		pvo = m->extra_mimetypes;

	if (n < 5)
		return NULL;

	if (!strcmp(&file[n - 4], ".ico"))
		return "image/x-icon";

	if (!strcmp(&file[n - 4], ".gif"))
		return "image/gif";

	if (!strcmp(&file[n - 3], ".js"))
		return "text/javascript";

	if (!strcmp(&file[n - 4], ".png"))
		return "image/png";

	if (!strcmp(&file[n - 4], ".jpg"))
		return "image/jpeg";

	if (!strcmp(&file[n - 3], ".gz"))
		return "application/gzip";

	if (!strcmp(&file[n - 4], ".JPG"))
		return "image/jpeg";

	if (!strcmp(&file[n - 5], ".html"))
		return "text/html";

	if (!strcmp(&file[n - 4], ".css"))
		return "text/css";

	if (!strcmp(&file[n - 4], ".txt"))
		return "text/plain";

	if (!strcmp(&file[n - 4], ".svg"))
		return "image/svg+xml";

	if (!strcmp(&file[n - 4], ".ttf"))
		return "application/x-font-ttf";

	if (!strcmp(&file[n - 4], ".otf"))
		return "application/font-woff";

	if (!strcmp(&file[n - 5], ".woff"))
		return "application/font-woff";

	if (!strcmp(&file[n - 4], ".xml"))
		return "application/xml";

	while (pvo) {
		if (pvo->name[0] == '*') /* ie, match anything */
			return pvo->value;

		if (!strcmp(&file[n - strlen(pvo->name)], pvo->name))
			return pvo->value;

		pvo = pvo->next;
	}

	return NULL;
}
static lws_fop_flags_t
lws_vfs_prepare_flags(struct lws *wsi)
{
	lws_fop_flags_t f = 0;

	if (!lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_ACCEPT_ENCODING))
		return f;

	if (strstr(lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_ACCEPT_ENCODING),
		   "gzip")) {
		lwsl_info("client indicates GZIP is acceptable\n");
		f |= LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP;
	}

	return f;
}

static int
lws_http_serve(struct lws *wsi, char *uri, const char *origin,
	       const struct lws_http_mount *m)
{
	const struct lws_protocol_vhost_options *pvo = m->interpret;
	struct lws_process_html_args args;
	const char *mimetype;
#if !defined(_WIN32_WCE)
	const struct lws_plat_file_ops *fops;
	const char *vpath;
	lws_fop_flags_t fflags = LWS_O_RDONLY;
#if defined(WIN32) && defined(LWS_HAVE__STAT32I64)
	struct _stat32i64 st;
#else
	struct stat st;
#endif
	int spin = 0;
#endif
	char path[256], sym[2048];
	unsigned char *p = (unsigned char *)sym + 32 + LWS_PRE, *start = p;
	unsigned char *end = p + sizeof(sym) - 32 - LWS_PRE;
#if !defined(WIN32) && !defined(LWS_WITH_ESP32)
	size_t len;
#endif
	int n;

	wsi->handling_404 = 0;
	if (!wsi->vhost)
		return -1;

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	if (wsi->vhost->http.error_document_404 &&
	    !strcmp(uri, wsi->vhost->http.error_document_404))
		wsi->handling_404 = 1;
#endif

	lws_snprintf(path, sizeof(path) - 1, "%s/%s", origin, uri);

#if !defined(_WIN32_WCE)

	fflags |= lws_vfs_prepare_flags(wsi);

	do {
		spin++;
		fops = lws_vfs_select_fops(wsi->context->fops, path, &vpath);

		if (wsi->http.fop_fd)
			lws_vfs_file_close(&wsi->http.fop_fd);

		wsi->http.fop_fd = fops->LWS_FOP_OPEN(wsi->context->fops,
							path, vpath, &fflags);
		if (!wsi->http.fop_fd) {
			lwsl_info("%s: Unable to open '%s': errno %d\n",
				  __func__, path, errno);

			return 1;
		}

		/* if it can't be statted, don't try */
		if (fflags & LWS_FOP_FLAG_VIRTUAL)
			break;
#if defined(LWS_WITH_ESP32)
		break;
#endif
#if !defined(WIN32)
		if (fstat(wsi->http.fop_fd->fd, &st)) {
			lwsl_info("unable to stat %s\n", path);
			goto notfound;
		}
#else
#if defined(LWS_HAVE__STAT32I64)
		if (_stat32i64(path, &st)) {
			lwsl_info("unable to stat %s\n", path);
			goto notfound;
		}
#else
		if (stat(path, &st)) {
			lwsl_info("unable to stat %s\n", path);
			goto notfound;
		}
#endif
#endif

		wsi->http.fop_fd->mod_time = (uint32_t)st.st_mtime;
		fflags |= LWS_FOP_FLAG_MOD_TIME_VALID;

#if !defined(WIN32) && !defined(LWS_WITH_ESP32)
		if ((S_IFMT & st.st_mode) == S_IFLNK) {
			len = readlink(path, sym, sizeof(sym) - 1);
			if (len) {
				lwsl_err("Failed to read link %s\n", path);
				goto notfound;
			}
			sym[len] = '\0';
			lwsl_debug("symlink %s -> %s\n", path, sym);
			lws_snprintf(path, sizeof(path) - 1, "%s", sym);
		}
#endif
		if ((S_IFMT & st.st_mode) == S_IFDIR) {
			lwsl_debug("default filename append to dir\n");
			lws_snprintf(path, sizeof(path) - 1, "%s/%s/index.html",
				 origin, uri);
		}

	} while ((S_IFMT & st.st_mode) != S_IFREG && spin < 5);

	if (spin == 5)
		lwsl_err("symlink loop %s \n", path);

	n = sprintf(sym, "%08llX%08lX",
		    (unsigned long long)lws_vfs_get_length(wsi->http.fop_fd),
		    (unsigned long)lws_vfs_get_mod_time(wsi->http.fop_fd));

	/* disable ranges if IF_RANGE token invalid */

	if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_IF_RANGE))
		if (strcmp(sym, lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_IF_RANGE)))
			/* differs - defeat Range: */
			wsi->http.ah->frag_index[WSI_TOKEN_HTTP_RANGE] = 0;

	if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_IF_NONE_MATCH)) {
		/*
		 * he thinks he has some version of it already,
		 * check if the tag matches
		 */
		if (!strcmp(sym, lws_hdr_simple_ptr(wsi,
					WSI_TOKEN_HTTP_IF_NONE_MATCH))) {

			char cache_control[50], *cc = "no-store";
			int cclen = 8;

			lwsl_debug("%s: ETAG match %s %s\n", __func__,
				   uri, origin);

			/* we don't need to send the payload */
			if (lws_add_http_header_status(wsi,
					HTTP_STATUS_NOT_MODIFIED, &p, end)) {
				lwsl_err("%s: failed adding not modified\n",
						__func__);
				return -1;
			}

			if (lws_add_http_header_by_token(wsi,
					WSI_TOKEN_HTTP_ETAG,
					(unsigned char *)sym, n, &p, end))
				return -1;

			/* but we still need to send cache control... */

			if (m->cache_max_age && m->cache_reusable) {
				if (!m->cache_revalidate) {
					cc = cache_control;
					cclen = sprintf(cache_control,
						"%s, max-age=%u",
						intermediates[wsi->cache_intermediaries],
						m->cache_max_age);
				} else {
					cc = cache_control;
                                        cclen = sprintf(cache_control,
                                        	"must-revalidate, %s, max-age=%u",
                                                intermediates[wsi->cache_intermediaries],
                                                m->cache_max_age);
				}
			}

			if (lws_add_http_header_by_token(wsi,
					WSI_TOKEN_HTTP_CACHE_CONTROL,
					(unsigned char *)cc, cclen, &p, end))
				return -1;

			if (lws_finalize_http_header(wsi, &p, end))
				return -1;

			n = lws_write(wsi, start, p - start,
				      LWS_WRITE_HTTP_HEADERS |
				      LWS_WRITE_H2_STREAM_END);
			if (n != (p - start)) {
				lwsl_err("_write returned %d from %ld\n", n,
					 (long)(p - start));
				return -1;
			}

			lws_vfs_file_close(&wsi->http.fop_fd);

			if (lws_http_transaction_completed(wsi))
				return -1;

			return 0;
		}
	}

	if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_ETAG,
			(unsigned char *)sym, n, &p, end))
		return -1;
#endif

	mimetype = lws_get_mimetype(path, m);
	if (!mimetype) {
		lwsl_info("unknown mimetype for %s\n", path);
		if (lws_return_http_status(wsi,
				HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE, NULL) ||
		    lws_http_transaction_completed(wsi))
			return -1;

		return 0;
	}
	if (!mimetype[0])
		lwsl_debug("sending no mimetype for %s\n", path);

	wsi->sending_chunked = 0;

	/*
	 * check if this is in the list of file suffixes to be interpreted by
	 * a protocol
	 */
	while (pvo) {
		n = (int)strlen(path);
		if (n > (int)strlen(pvo->name) &&
		    !strcmp(&path[n - strlen(pvo->name)], pvo->name)) {
			wsi->interpreting = 1;
			if (!wsi->http2_substream)
				wsi->sending_chunked = 1;
			wsi->protocol_interpret_idx =
					(char)(lws_intptr_t)pvo->value;
			lwsl_info("want %s interpreted by %s\n", path,
				    wsi->vhost->protocols[
				         (int)(lws_intptr_t)(pvo->value)].name);
			wsi->protocol = &wsi->vhost->protocols[
			                       (int)(lws_intptr_t)(pvo->value)];
			if (lws_ensure_user_space(wsi))
				return -1;
			break;
		}
		pvo = pvo->next;
	}

	if (m->protocol) {
		const struct lws_protocols *pp = lws_vhost_name_to_protocol(
						       wsi->vhost, m->protocol);

		if (lws_bind_protocol(wsi, pp, __func__))
			return -1;
		args.p = (char *)p;
		args.max_len = lws_ptr_diff(end, p);
		if (pp->callback(wsi, LWS_CALLBACK_ADD_HEADERS,
					  wsi->user_space, &args, 0))
			return -1;
		p = (unsigned char *)args.p;
	}

	*p = '\0';
	n = lws_serve_http_file(wsi, path, mimetype, (char *)start,
				lws_ptr_diff(p, start));

	if (n < 0 || ((n > 0) && lws_http_transaction_completed(wsi)))
		return -1; /* error or can't reuse connection: close the socket */

	return 0;

notfound:

	return 1;
}

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
const struct lws_http_mount *
lws_find_mount(struct lws *wsi, const char *uri_ptr, int uri_len)
{
	const struct lws_http_mount *hm, *hit = NULL;
	int best = 0;

	hm = wsi->vhost->http.mount_list;
	while (hm) {
		if (uri_len >= hm->mountpoint_len &&
		    !strncmp(uri_ptr, hm->mountpoint, hm->mountpoint_len) &&
		    (uri_ptr[hm->mountpoint_len] == '\0' ||
		     uri_ptr[hm->mountpoint_len] == '/' ||
		     hm->mountpoint_len == 1)
		    ) {
			if (hm->origin_protocol == LWSMPRO_CALLBACK ||
			    ((hm->origin_protocol == LWSMPRO_CGI ||
			     lws_hdr_total_length(wsi, WSI_TOKEN_GET_URI) ||
			     (wsi->http2_substream &&
				lws_hdr_total_length(wsi,
						WSI_TOKEN_HTTP_COLON_PATH)) ||
			     hm->protocol) &&
			    hm->mountpoint_len > best)) {
				best = hm->mountpoint_len;
				hit = hm;
			}
		}
		hm = hm->mount_next;
	}

	return hit;
}
#endif

#if !defined(LWS_WITH_ESP32)
static int
lws_find_string_in_file(const char *filename, const char *string, int stringlen)
{
	char buf[128];
	int fd, match = 0, pos = 0, n = 0, hit = 0;

	fd = lws_open(filename, O_RDONLY);
	if (fd < 0) {
		lwsl_err("can't open auth file: %s\n", filename);
		return 0;
	}

	while (1) {
		if (pos == n) {
			n = read(fd, buf, sizeof(buf));
			if (n <= 0) {
				if (match == stringlen)
					hit = 1;
				break;
			}
			pos = 0;
		}

		if (match == stringlen) {
			if (buf[pos] == '\r' || buf[pos] == '\n') {
				hit = 1;
				break;
			}
			match = 0;
		}

		if (buf[pos] == string[match])
			match++;
		else
			match = 0;

		pos++;
	}

	close(fd);

	return hit;
}
#endif

static int
lws_unauthorised_basic_auth(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	unsigned char *start = pt->serv_buf + LWS_PRE,
		      *p = start, *end = p + 2048;
	char buf[64];
	int n;

	/* no auth... tell him it is required */

	if (lws_add_http_header_status(wsi, HTTP_STATUS_UNAUTHORIZED, &p, end))
		return -1;

	n = lws_snprintf(buf, sizeof(buf), "Basic realm=\"lwsws\"");
	if (lws_add_http_header_by_token(wsi,
			WSI_TOKEN_HTTP_WWW_AUTHENTICATE,
			(unsigned char *)buf, n, &p, end))
		return -1;

	if (lws_finalize_http_header(wsi, &p, end))
		return -1;

	n = lws_write(wsi, start, p - start, LWS_WRITE_HTTP_HEADERS |
					     LWS_WRITE_H2_STREAM_END);
	if (n < 0)
		return -1;

	return lws_http_transaction_completed(wsi);

}

int lws_clean_url(char *p)
{
	if (p[0] == 'h' && p[1] == 't' && p[2] == 't' && p[3] == 'p') {
		p += 4;
		if (*p == 's')
		p++;
		if (*p == ':') {
			p++;
			if (*p == '/')
			p++;
		}
	}

	while (*p) {
		if (p[0] == '/' && p[1] == '/') {
			char *p1 = p;
			while (*p1) {
				*p1 = p1[1];
				p1++;
			}
			continue;
		}
		p++;
	}

	return 0;
}

static const unsigned char methods[] = {
	WSI_TOKEN_GET_URI,
	WSI_TOKEN_POST_URI,
	WSI_TOKEN_OPTIONS_URI,
	WSI_TOKEN_PUT_URI,
	WSI_TOKEN_PATCH_URI,
	WSI_TOKEN_DELETE_URI,
	WSI_TOKEN_CONNECT,
	WSI_TOKEN_HEAD_URI,
#ifdef LWS_WITH_HTTP2
	WSI_TOKEN_HTTP_COLON_PATH,
#endif
};

static int
lws_http_get_uri_and_method(struct lws *wsi, char **puri_ptr, int *puri_len)
{
	int n, count = 0;

	for (n = 0; n < (int)LWS_ARRAY_SIZE(methods); n++)
		if (lws_hdr_total_length(wsi, methods[n]))
			count++;
	if (!count) {
		lwsl_warn("Missing URI in HTTP request\n");
		return -1;
	}

	if (count != 1 &&
	    !(wsi->http2_substream &&
	      lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_COLON_PATH))) {
		lwsl_warn("multiple methods?\n");
		return -1;
	}

	for (n = 0; n < (int)LWS_ARRAY_SIZE(methods); n++)
		if (lws_hdr_total_length(wsi, methods[n])) {
			*puri_ptr = lws_hdr_simple_ptr(wsi, methods[n]);
			*puri_len = lws_hdr_total_length(wsi, methods[n]);
			return n;
		}

	return -1;
}

static const char * const oprot[] = {
	"http://", "https://"
};

int
lws_http_action(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	const struct lws_http_mount *hit = NULL;
	enum http_version request_version;
	struct lws_process_html_args args;
	enum http_conn_type conn_type;
	char content_length_str[32];
	char http_version_str[12];
	char *uri_ptr = NULL, *s;
	int uri_len = 0, meth, m;
	char http_conn_str[25];
	int http_version_len;
	unsigned int n;

	meth = lws_http_get_uri_and_method(wsi, &uri_ptr, &uri_len);
	if (meth < 0 || meth >= (int)LWS_ARRAY_SIZE(method_names))
		goto bail_nuke_ah;

	/* we insist on absolute paths */

	if (!uri_ptr || uri_ptr[0] != '/') {
		lws_return_http_status(wsi, HTTP_STATUS_FORBIDDEN, NULL);

		goto bail_nuke_ah;
	}

	lwsl_info("Method: '%s' (%d), request for '%s'\n", method_names[meth],
		  meth, uri_ptr);

	if (wsi->role_ops && wsi->role_ops->check_upgrades)
		switch (wsi->role_ops->check_upgrades(wsi)) {
		case LWS_UPG_RET_DONE:
			return 0;
		case LWS_UPG_RET_CONTINUE:
			break;
		case LWS_UPG_RET_BAIL:
			goto bail_nuke_ah;
		}

	if (lws_ensure_user_space(wsi))
		goto bail_nuke_ah;

	/* HTTP header had a content length? */

	wsi->http.rx_content_length = 0;
	wsi->http.content_length_explicitly_zero = 0;
	if (lws_hdr_total_length(wsi, WSI_TOKEN_POST_URI) ||
	    lws_hdr_total_length(wsi, WSI_TOKEN_PATCH_URI) ||
	    lws_hdr_total_length(wsi, WSI_TOKEN_PUT_URI))
		wsi->http.rx_content_length = 100 * 1024 * 1024;

	if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_CONTENT_LENGTH) &&
	    lws_hdr_copy(wsi, content_length_str,
			 sizeof(content_length_str) - 1,
			 WSI_TOKEN_HTTP_CONTENT_LENGTH) > 0) {
		wsi->http.rx_content_length = atoll(content_length_str);
		if (!wsi->http.rx_content_length) {
			wsi->http.content_length_explicitly_zero = 1;
			lwsl_debug("%s: explicit 0 content-length\n", __func__);
		}
	}

	if (wsi->http2_substream) {
		wsi->http.request_version = HTTP_VERSION_2;
	} else {
		/* http_version? Default to 1.0, override with token: */
		request_version = HTTP_VERSION_1_0;

		/* Works for single digit HTTP versions. : */
		http_version_len = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP);
		if (http_version_len > 7 &&
		    lws_hdr_copy(wsi, http_version_str,
				 sizeof(http_version_str) - 1,
				 WSI_TOKEN_HTTP) > 0 &&
		    http_version_str[5] == '1' && http_version_str[7] == '1')
			request_version = HTTP_VERSION_1_1;

		wsi->http.request_version = request_version;

		/* HTTP/1.1 defaults to "keep-alive", 1.0 to "close" */
		if (request_version == HTTP_VERSION_1_1)
			conn_type = HTTP_CONNECTION_KEEP_ALIVE;
		else
			conn_type = HTTP_CONNECTION_CLOSE;

		/* Override default if http "Connection:" header: */
		if (lws_hdr_total_length(wsi, WSI_TOKEN_CONNECTION) &&
		    lws_hdr_copy(wsi, http_conn_str, sizeof(http_conn_str) - 1,
				 WSI_TOKEN_CONNECTION) > 0) {
			http_conn_str[sizeof(http_conn_str) - 1] = '\0';
			if (!strcasecmp(http_conn_str, "keep-alive"))
				conn_type = HTTP_CONNECTION_KEEP_ALIVE;
			else
				if (!strcasecmp(http_conn_str, "close"))
					conn_type = HTTP_CONNECTION_CLOSE;
		}
		wsi->http.conn_type = conn_type;
	}

	n = wsi->protocol->callback(wsi, LWS_CALLBACK_FILTER_HTTP_CONNECTION,
				    wsi->user_space, uri_ptr, uri_len);
	if (n) {
		lwsl_info("LWS_CALLBACK_HTTP closing\n");

		return 1;
	}
	/*
	 * if there is content supposed to be coming,
	 * put a timeout on it having arrived
	 */
	lws_set_timeout(wsi, PENDING_TIMEOUT_HTTP_CONTENT,
			wsi->context->timeout_secs);
#ifdef LWS_WITH_TLS
	if (wsi->tls.redirect_to_https) {
		/*
		 * we accepted http:// only so we could redirect to
		 * https://, so issue the redirect.  Create the redirection
		 * URI from the host: header and ignore the path part
		 */
		unsigned char *start = pt->serv_buf + LWS_PRE, *p = start,
			      *end = p + wsi->context->pt_serv_buf_size - LWS_PRE;

		if (!lws_hdr_total_length(wsi, WSI_TOKEN_HOST))
			goto bail_nuke_ah;

		n = sprintf((char *)end, "https://%s/",
			    lws_hdr_simple_ptr(wsi, WSI_TOKEN_HOST));

		n = lws_http_redirect(wsi, HTTP_STATUS_MOVED_PERMANENTLY,
				      end, n, &p, end);
		if ((int)n < 0)
			goto bail_nuke_ah;

		return lws_http_transaction_completed(wsi);
	}
#endif

#ifdef LWS_WITH_ACCESS_LOG
	lws_prepare_access_log_info(wsi, uri_ptr, uri_len, meth);
#endif

	/* can we serve it from the mount list? */

	hit = lws_find_mount(wsi, uri_ptr, uri_len);
	if (!hit) {
		/* deferred cleanup and reset to protocols[0] */

		lwsl_info("no hit\n");

		if (lws_bind_protocol(wsi, &wsi->vhost->protocols[0],
				      "no mount hit"))
			return 1;

		lwsi_set_state(wsi, LRS_DOING_TRANSACTION);

		m = wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP,
				    wsi->user_space, uri_ptr, uri_len);

		goto after;
	}

	s = uri_ptr + hit->mountpoint_len;

	/*
	 * if we have a mountpoint like https://xxx.com/yyy
	 * there is an implied / at the end for our purposes since
	 * we can only mount on a "directory".
	 *
	 * But if we just go with that, the browser cannot understand
	 * that he is actually looking down one "directory level", so
	 * even though we give him /yyy/abc.html he acts like the
	 * current directory level is /.  So relative urls like "x.png"
	 * wrongly look outside the mountpoint.
	 *
	 * Therefore if we didn't come in on a url with an explicit
	 * / at the end, we must redirect to add it so the browser
	 * understands he is one "directory level" down.
	 */
	if ((hit->mountpoint_len > 1 ||
	     (hit->origin_protocol == LWSMPRO_REDIR_HTTP ||
	      hit->origin_protocol == LWSMPRO_REDIR_HTTPS)) &&
	    (*s != '/' ||
	     (hit->origin_protocol == LWSMPRO_REDIR_HTTP ||
	      hit->origin_protocol == LWSMPRO_REDIR_HTTPS)) &&
	    (hit->origin_protocol != LWSMPRO_CGI &&
	     hit->origin_protocol != LWSMPRO_CALLBACK)) {
		unsigned char *start = pt->serv_buf + LWS_PRE, *p = start,
			      *end = p + wsi->context->pt_serv_buf_size -
			      	     LWS_PRE - 512;

		lwsl_info("Doing 301 '%s' org %s\n", s, hit->origin);

		/* > at start indicates deal with by redirect */
		if (hit->origin_protocol == LWSMPRO_REDIR_HTTP ||
		    hit->origin_protocol == LWSMPRO_REDIR_HTTPS)
			n = lws_snprintf((char *)end, 256, "%s%s",
				    oprot[hit->origin_protocol & 1],
				    hit->origin);
		else {
			if (!lws_hdr_total_length(wsi, WSI_TOKEN_HOST)) {
				if (!lws_hdr_total_length(wsi,
						WSI_TOKEN_HTTP_COLON_AUTHORITY))
					goto bail_nuke_ah;
				n = lws_snprintf((char *)end, 256,
				    "%s%s%s/", oprot[!!lws_is_ssl(wsi)],
				    lws_hdr_simple_ptr(wsi,
						WSI_TOKEN_HTTP_COLON_AUTHORITY),
				    uri_ptr);
			} else
				n = lws_snprintf((char *)end, 256,
				    "%s%s%s/", oprot[!!lws_is_ssl(wsi)],
				    lws_hdr_simple_ptr(wsi, WSI_TOKEN_HOST),
				    uri_ptr);
		}

		lws_clean_url((char *)end);
		n = lws_http_redirect(wsi, HTTP_STATUS_MOVED_PERMANENTLY,
				      end, n, &p, end);
		if ((int)n < 0)
			goto bail_nuke_ah;

		return lws_http_transaction_completed(wsi);
	}

	/* basic auth? */

	if (hit->basic_auth_login_file) {
		char b64[160], plain[(sizeof(b64) * 3) / 4], *pcolon;
		int m, ml, fi;

		/* Did he send auth? */
		ml = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_AUTHORIZATION);
		if (!ml)
			return lws_unauthorised_basic_auth(wsi);

		/* Disallow fragmentation monkey business */

		fi = wsi->http.ah->frag_index[WSI_TOKEN_HTTP_AUTHORIZATION];
		if (wsi->http.ah->frags[fi].nfrag) {
			lwsl_err("fragmented basic auth header not allowed\n");
			return lws_unauthorised_basic_auth(wsi);
		}

		n = HTTP_STATUS_FORBIDDEN;

		m = lws_hdr_copy(wsi, b64, sizeof(b64),
				 WSI_TOKEN_HTTP_AUTHORIZATION);
		if (m < 7) {
			lwsl_err("b64 auth too long\n");
			goto transaction_result_n;
		}

		b64[5] = '\0';
		if (strcasecmp(b64, "Basic")) {
			lwsl_err("auth missing basic: %s\n", b64);
			goto transaction_result_n;
		}

		/* It'll be like Authorization: Basic QWxhZGRpbjpPcGVuU2VzYW1l */

		m = lws_b64_decode_string(b64 + 6, plain, sizeof(plain) - 1);
		if (m < 0) {
			lwsl_err("plain auth too long\n");
			goto transaction_result_n;
		}

		plain[m] = '\0';
		pcolon = strchr(plain, ':');
		if (!pcolon) {
			lwsl_err("basic auth format broken\n");
			return lws_unauthorised_basic_auth(wsi);
		}
		if (!lws_find_string_in_file(hit->basic_auth_login_file,
					     plain, m)) {
			lwsl_err("basic auth lookup failed\n");
			return lws_unauthorised_basic_auth(wsi);
		}

		/*
		 * Rewrite WSI_TOKEN_HTTP_AUTHORIZATION so it is just the
		 * authorized username
		 */

		*pcolon = '\0';
		wsi->http.ah->frags[fi].len = lws_ptr_diff(pcolon, plain);
		pcolon = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_AUTHORIZATION);
		strncpy(pcolon, plain, ml - 1);
		pcolon[ml - 1] = '\0';
		lwsl_info("%s: basic auth accepted for %s\n", __func__,
			 lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_AUTHORIZATION));
	}

#if defined(LWS_WITH_HTTP_PROXY)
	/*
	 * The mount is a reverse proxy?
	 */

	// lwsl_notice("%s: origin_protocol: %d\n", __func__, hit->origin_protocol);

	if (hit->origin_protocol == LWSMPRO_HTTPS ||
	    hit->origin_protocol == LWSMPRO_HTTP)  {
		char ads[96], rpath[256], *pcolon, *pslash, unix_skt = 0;
		struct lws_client_connect_info i;
		struct lws *cwsi;
		int n, na;

		memset(&i, 0, sizeof(i));
		i.context = lws_get_context(wsi);

		if (hit->origin[0] == '+')
			unix_skt = 1;

		pcolon = strchr(hit->origin, ':');
		pslash = strchr(hit->origin, '/');
		if (!pslash) {
			lwsl_err("Proxy mount origin '%s' must have /\n",
				 hit->origin);
			return -1;
		}

		if (unix_skt) {
			if (!pcolon) {
				lwsl_err("Proxy mount origin for unix skt must "
					 "have address delimited by :\n");

				return -1;
			}
			n = lws_ptr_diff(pcolon, hit->origin);
			pslash = pcolon;
		} else {
			if (pcolon > pslash)
				pcolon = NULL;

			if (pcolon)
				n = (int)(pcolon - hit->origin);
			else
				n = (int)(pslash - hit->origin);

			if (n >= (int)sizeof(ads) - 2)
				n = sizeof(ads) - 2;
		}

		memcpy(ads, hit->origin, n);
		ads[n] = '\0';

		i.address = ads;
		i.port = 80;
		if (hit->origin_protocol == LWSMPRO_HTTPS) {
			i.port = 443;
			i.ssl_connection = 1;
		}
		if (pcolon)
			i.port = atoi(pcolon + 1);

		n = lws_snprintf(rpath, sizeof(rpath) - 1, "/%s/%s",
				 pslash + 1, uri_ptr + hit->mountpoint_len) - 2;
		lws_clean_url(rpath);
		na = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_URI_ARGS);
		if (na) {
			char *p = rpath + n;

			if (na >= (int)sizeof(rpath) - n - 2) {
				lwsl_info("%s: query string %d longer "
					  "than we can handle\n", __func__,
					  na);

				return -1;
			}

			*p++ = '?';
			if (lws_hdr_copy(wsi, p,
				     (int)(&rpath[sizeof(rpath) - 1] - p),
				     WSI_TOKEN_HTTP_URI_ARGS) > 0)
				while (na--) {
					if (*p == '\0')
						*p = '&';
					p++;
				}
			*p = '\0';
		}

		i.path = rpath;
		if (i.address[0] != '+' ||
		    !lws_hdr_simple_ptr(wsi, WSI_TOKEN_HOST))
			i.host = i.address;
		else
			i.host = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HOST);
		i.origin = NULL;
		i.method = "GET";
		i.alpn = "http/1.1";
		i.parent_wsi = wsi;
		i.pwsi = &cwsi;

	//	i.uri_replace_from = hit->origin;
	//	i.uri_replace_to = hit->mountpoint;

		lwsl_info("proxying to %s port %d url %s, ssl %d, "
			    "from %s, to %s\n",
			    i.address, i.port, i.path, i.ssl_connection,
			    i.uri_replace_from, i.uri_replace_to);

		if (!lws_client_connect_via_info(&i)) {
			lwsl_err("proxy connect fail\n");

			/*
			 * ... we can't do the proxy action, but we can
			 * cleanly return him a 503 and a description
			 */

			lws_return_http_status(wsi,
				HTTP_STATUS_SERVICE_UNAVAILABLE,
				"<h1>Service Temporarily Unavailable</h1>"
				"The server is temporarily unable to service "
				"your request due to maintenance downtime or "
				"capacity problems. Please try again later.");

			return 1;
		}

		lwsl_info("%s: setting proxy clientside on %p (parent %p)\n",
			  __func__, cwsi, lws_get_parent(cwsi));
		cwsi->http.proxy_clientside = 1;

		return 0;
	}
#endif

	/*
	 * A particular protocol callback is mounted here?
	 *
	 * For the duration of this http transaction, bind us to the
	 * associated protocol
	 */
	if (hit->origin_protocol == LWSMPRO_CALLBACK || hit->protocol) {
		const struct lws_protocols *pp;
		const char *name = hit->origin;
		if (hit->protocol)
			name = hit->protocol;

		pp = lws_vhost_name_to_protocol(wsi->vhost, name);
		if (!pp) {
			n = -1;
			lwsl_err("Unable to find plugin '%s'\n",
				 hit->origin);
			return 1;
		}

		if (lws_bind_protocol(wsi, pp, "http action CALLBACK bind"))
			return 1;

		args.p = uri_ptr;
		args.len = uri_len;
		args.max_len = hit->auth_mask;
		args.final = 0; /* used to signal callback dealt with it */
		args.chunked = 0;

		n = wsi->protocol->callback(wsi,
					    LWS_CALLBACK_CHECK_ACCESS_RIGHTS,
					    wsi->user_space, &args, 0);
		if (n) {
			lws_return_http_status(wsi, HTTP_STATUS_UNAUTHORIZED,
					       NULL);
			goto bail_nuke_ah;
		}
		if (args.final) /* callback completely handled it well */
			return 0;

		if (hit->cgienv && wsi->protocol->callback(wsi,
				LWS_CALLBACK_HTTP_PMO,
				wsi->user_space, (void *)hit->cgienv, 0))
			return 1;

		if (lws_hdr_total_length(wsi, WSI_TOKEN_POST_URI)) {
			m = wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP,
					    wsi->user_space,
					    uri_ptr + hit->mountpoint_len,
					    uri_len - hit->mountpoint_len);
			goto after;
		}
	}

#ifdef LWS_WITH_CGI
	/* did we hit something with a cgi:// origin? */
	if (hit->origin_protocol == LWSMPRO_CGI) {
		const char *cmd[] = {
			NULL, /* replace with cgi path */
			NULL
		};

		lwsl_debug("%s: cgi\n", __func__);
		cmd[0] = hit->origin;

		n = 5;
		if (hit->cgi_timeout)
			n = hit->cgi_timeout;

		n = lws_cgi(wsi, cmd, hit->mountpoint_len, n,
			    hit->cgienv);
		if (n) {
			lwsl_err("%s: cgi failed\n", __func__);
			return -1;
		}

		goto deal_body;
	}
#endif

	n = uri_len - lws_ptr_diff(s, uri_ptr); // (int)strlen(s);
	if (s[0] == '\0' || (n == 1 && s[n - 1] == '/'))
		s = (char *)hit->def;
	if (!s)
		s = "index.html";

	wsi->cache_secs = hit->cache_max_age;
	wsi->cache_reuse = hit->cache_reusable;
	wsi->cache_revalidate = hit->cache_revalidate;
	wsi->cache_intermediaries = hit->cache_intermediaries;

	m = 1;
	if (hit->origin_protocol == LWSMPRO_FILE)
		m = lws_http_serve(wsi, s, hit->origin, hit);

	if (m > 0) {
		/*
		 * lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, NULL);
		 */
		if (hit->protocol) {
			const struct lws_protocols *pp =
					lws_vhost_name_to_protocol(
						wsi->vhost, hit->protocol);

			lwsi_set_state(wsi, LRS_DOING_TRANSACTION);

			if (lws_bind_protocol(wsi, pp, "http_action HTTP"))
				return 1;

			m = pp->callback(wsi, LWS_CALLBACK_HTTP,
					 wsi->user_space,
					 uri_ptr + hit->mountpoint_len,
					 uri_len - hit->mountpoint_len);
		} else
			m = wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP,
				    wsi->user_space, uri_ptr, uri_len);
	}

after:
	if (m) {
		lwsl_info("LWS_CALLBACK_HTTP closing\n");

		return 1;
	}

#ifdef LWS_WITH_CGI
deal_body:
#endif
	/*
	 * If we're not issuing a file, check for content_length or
	 * HTTP keep-alive. No keep-alive header allocation for
	 * ISSUING_FILE, as this uses HTTP/1.0.
	 *
	 * In any case, return 0 and let lws_read decide how to
	 * proceed based on state
	 */
	if (lwsi_state(wsi) != LRS_ISSUING_FILE) {
		/* Prepare to read body if we have a content length: */
		lwsl_debug("wsi->http.rx_content_length %lld %d %d\n",
			   (long long)wsi->http.rx_content_length,
			   wsi->upgraded_to_http2, wsi->http2_substream);

		if (wsi->http.content_length_explicitly_zero &&
		    lws_hdr_total_length(wsi, WSI_TOKEN_POST_URI)) {

			/*
			 * POST with an explicit content-length of zero
			 *
			 * If we don't give the user code the empty HTTP_BODY
			 * callback, he may become confused to hear the
			 * HTTP_BODY_COMPLETION (due to, eg, instantiation of
			 * lws_spa never happened).
			 *
			 * HTTP_BODY_COMPLETION is responsible for sending the
			 * result status code and result body if any, and
			 * do the transaction complete processing.
			 */
			if (wsi->protocol->callback(wsi,
					LWS_CALLBACK_HTTP_BODY,
					wsi->user_space, NULL, 0))
				return 1;
			if (wsi->protocol->callback(wsi,
					LWS_CALLBACK_HTTP_BODY_COMPLETION,
					wsi->user_space, NULL, 0))
				return 1;

			return 0;
		}

		if (wsi->http.rx_content_length > 0) {

			lwsi_set_state(wsi, LRS_BODY);
			lwsl_info("%s: %p: LRS_BODY state set (0x%x)\n",
				    __func__, wsi, wsi->wsistate);
			wsi->http.rx_content_remain =
					wsi->http.rx_content_length;

			/*
			 * At this point we have transitioned from deferred
			 * action to expecting BODY on the stream wsi, if it's
			 * in a bundle like h2.  So if the stream wsi has its
			 * own buflist, we need to deal with that first.
			 */

			while (1) {
				struct lws_tokens ebuf;
				int m;

				ebuf.len = (int)lws_buflist_next_segment_len(
						&wsi->buflist,
						(uint8_t **)&ebuf.token);
				if (!ebuf.len)
					break;
				lwsl_notice("%s: consuming %d\n", __func__,
							(int)ebuf.len);
				m = lws_read_h1(wsi, (uint8_t *)ebuf.token,
						ebuf.len);
				if (m < 0)
					return -1;

				if (lws_buflist_aware_consume(wsi, &ebuf, m, 1))
					return -1;
			}
		}
	}

	return 0;

bail_nuke_ah:
	lws_header_table_detach(wsi, 1);

	return 1;

transaction_result_n:
	lws_return_http_status(wsi, n, NULL);

	return lws_http_transaction_completed(wsi);
}

int
lws_confirm_host_header(struct lws *wsi)
{
	struct lws_tokenize ts;
	lws_tokenize_elem e;
	char buf[128];
	int port = 80;

	/*
	 * this vhost wants us to validate what the
	 * client sent against our vhost name
	 */

	if (!lws_hdr_total_length(wsi, WSI_TOKEN_HOST)) {
		lwsl_info("%s: missing host on upgrade\n", __func__);

		return 1;
	}

#if defined(LWS_WITH_TLS)
	if (wsi->tls.ssl)
		port = 443;
#endif

	lws_tokenize_init(&ts, buf, LWS_TOKENIZE_F_DOT_NONTERM /* server.com */|
				    LWS_TOKENIZE_F_NO_FLOATS /* 1.server.com */|
				    LWS_TOKENIZE_F_MINUS_NONTERM /* a-b.com */);
	ts.len = lws_hdr_copy(wsi, buf, sizeof(buf) - 1, WSI_TOKEN_HOST);
	if (ts.len <= 0) {
		lwsl_info("%s: missing or oversize host header\n", __func__);
		return 1;
	}

	if (lws_tokenize(&ts) != LWS_TOKZE_TOKEN)
		goto bad_format;

	if (strncmp(ts.token, wsi->vhost->name, ts.token_len)) {
		buf[(ts.token - buf) + ts.token_len] = '\0';
		lwsl_info("%s: '%s' in host hdr but vhost name %s\n",
			  __func__, ts.token, wsi->vhost->name);
		return 1;
	}

	e = lws_tokenize(&ts);
	if (e == LWS_TOKZE_DELIMITER && ts.token[0] == ':') {
		if (lws_tokenize(&ts) != LWS_TOKZE_INTEGER)
			goto bad_format;
		else
			port = atoi(ts.token);
	} else
		if (e != LWS_TOKZE_ENDED)
			goto bad_format;

	if (wsi->vhost->listen_port != port) {
		lwsl_info("%s: host port %d mismatches vhost port %d\n",
			  __func__, port, wsi->vhost->listen_port);
		return 1;
	}

	lwsl_debug("%s: host header OK\n", __func__);

	return 0;

bad_format:
	lwsl_info("%s: bad host header format\n", __func__);

	return 1;
}

int
lws_handshake_server(struct lws *wsi, unsigned char **buf, size_t len)
{
	struct lws_context *context = lws_get_context(wsi);
	unsigned char *obuf = *buf;
#if defined(LWS_WITH_HTTP2)
	char tbuf[128], *p;
#endif
	size_t olen = len;
	int n = 0, m, i;

	if (len >= 10000000) {
		lwsl_err("%s: assert: len %ld\n", __func__, (long)len);
		assert(0);
	}

	if (!wsi->http.ah) {
		lwsl_err("%s: assert: NULL ah\n", __func__);
		assert(0);
	}

	while (len) {
		if (!lwsi_role_server(wsi) || !lwsi_role_http(wsi)) {
			lwsl_err("%s: bad wsi role 0x%x\n", __func__,
					lwsi_role(wsi));
			goto bail_nuke_ah;
		}

		i = (int)len;
		m = lws_parse(wsi, *buf, &i);
		lwsl_info("%s: parsed count %d\n", __func__, (int)len - i);
		(*buf) += (int)len - i;
		len = i;
		if (m) {
			if (m == 2) {
				/*
				 * we are transitioning from http with
				 * an AH, to raw.  Drop the ah and set
				 * the mode.
				 */
raw_transition:
				lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);
				lws_bind_protocol(wsi, &wsi->vhost->protocols[
				                        wsi->vhost->
				                        raw_protocol_index],
						__func__);
				lwsl_info("transition to raw vh %s prot %d\n",
					  wsi->vhost->name,
					  wsi->vhost->raw_protocol_index);
				if ((wsi->protocol->callback)(wsi,
						LWS_CALLBACK_RAW_ADOPT,
						wsi->user_space, NULL, 0))
					goto bail_nuke_ah;

				lws_role_transition(wsi, 0, LRS_ESTABLISHED,
						    &role_ops_raw_skt);
				lws_header_table_detach(wsi, 1);

				if (wsi->protocol->callback(wsi,
						LWS_CALLBACK_RAW_RX,
						wsi->user_space, obuf, olen))
					return 1;

				return 0;
			}
			lwsl_info("lws_parse failed\n");
			goto bail_nuke_ah;
		}

		if (wsi->http.ah->parser_state != WSI_PARSING_COMPLETE)
			continue;

		lwsl_parser("%s: lws_parse sees parsing complete\n", __func__);

		/* select vhost */

		if (wsi->vhost->listen_port &&
		    lws_hdr_total_length(wsi, WSI_TOKEN_HOST)) {
			struct lws_vhost *vhost = lws_select_vhost(
				context, wsi->vhost->listen_port,
				lws_hdr_simple_ptr(wsi, WSI_TOKEN_HOST));

			if (vhost)
				lws_vhost_bind_wsi(vhost, wsi);
		} else
			lwsl_info("no host\n");

		if (!lwsi_role_h2(wsi) || !lwsi_role_server(wsi)) {
			wsi->vhost->conn_stats.h1_trans++;
			if (!wsi->conn_stat_done) {
				wsi->vhost->conn_stats.h1_conn++;
				wsi->conn_stat_done = 1;
			}
		}

		/* check for unwelcome guests */

		if (wsi->context->reject_service_keywords) {
			const struct lws_protocol_vhost_options *rej =
					wsi->context->reject_service_keywords;
			char ua[384], *msg = NULL;

			if (lws_hdr_copy(wsi, ua, sizeof(ua) - 1,
					 WSI_TOKEN_HTTP_USER_AGENT) > 0) {
#ifdef LWS_WITH_ACCESS_LOG
				char *uri_ptr = NULL;
				int meth, uri_len;
#endif
				ua[sizeof(ua) - 1] = '\0';
				while (rej) {
					if (!strstr(ua, rej->name)) {
						rej = rej->next;
						continue;
					}

					msg = strchr(rej->value, ' ');
					if (msg)
						msg++;
					lws_return_http_status(wsi,
						atoi(rej->value), msg);
#ifdef LWS_WITH_ACCESS_LOG
					meth = lws_http_get_uri_and_method(wsi,
							&uri_ptr, &uri_len);
					if (meth >= 0)
						lws_prepare_access_log_info(wsi,
							uri_ptr, uri_len, meth);

					/* wsi close will do the log */
#endif
					wsi->vhost->conn_stats.rejected++;
					/*
					 * We don't want anything from
					 * this rejected guy.  Follow
					 * the close flow, not the
					 * transaction complete flow.
					 */
					goto bail_nuke_ah;
				}
			}
		}


		if (lws_hdr_total_length(wsi, WSI_TOKEN_CONNECT)) {
			lwsl_info("Changing to RAW mode\n");
			m = 0;
			goto raw_transition;
		}

		lwsi_set_state(wsi, LRS_PRE_WS_SERVING_ACCEPT);
		lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);

		if (lws_hdr_total_length(wsi, WSI_TOKEN_UPGRADE)) {

			const char *up = lws_hdr_simple_ptr(wsi,
							    WSI_TOKEN_UPGRADE);

			if (strcasecmp(up, "websocket") &&
			    strcasecmp(up, "h2c")) {
				lwsl_info("Unknown upgrade '%s'\n", up);

				if (lws_return_http_status(wsi,
						HTTP_STATUS_FORBIDDEN, NULL) ||
				    lws_http_transaction_completed(wsi))
					goto bail_nuke_ah;
			}

			n = user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_HTTP_CONFIRM_UPGRADE,
					wsi->user_space, (char *)up, 0);

			/* just hang up? */

			if (n < 0)
				goto bail_nuke_ah;

			/* callback returned headers already, do t_c? */

			if (n > 0) {
				if (lws_http_transaction_completed(wsi))
					goto bail_nuke_ah;

				/* continue on */

				return 0;
			}

			/* callback said 0, it was allowed */

			if (wsi->vhost->options &
			    LWS_SERVER_OPTION_VHOST_UPG_STRICT_HOST_CHECK &&
			    lws_confirm_host_header(wsi))
				goto bail_nuke_ah;

			if (!strcasecmp(up, "websocket")) {
#if defined(LWS_ROLE_WS)
				wsi->vhost->conn_stats.ws_upg++;
				lwsl_info("Upgrade to ws\n");
				goto upgrade_ws;
#endif
			}
#if defined(LWS_WITH_HTTP2)
			if (!strcasecmp(up, "h2c")) {
				wsi->vhost->conn_stats.h2_upg++;
				lwsl_info("Upgrade to h2c\n");
				goto upgrade_h2c;
			}
#endif
		}

		/* no upgrade ack... he remained as HTTP */

		lwsl_info("%s: %p: No upgrade\n", __func__, wsi);

		lwsi_set_state(wsi, LRS_ESTABLISHED);
		wsi->http.fop_fd = NULL;

#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
		lws_http_compression_validate(wsi);
#endif

		lwsl_debug("%s: wsi %p: ah %p\n", __func__, (void *)wsi,
			   (void *)wsi->http.ah);

		n = lws_http_action(wsi);

		return n;

#if defined(LWS_WITH_HTTP2)
upgrade_h2c:
		if (!lws_hdr_total_length(wsi, WSI_TOKEN_HTTP2_SETTINGS)) {
			lwsl_info("missing http2_settings\n");
			goto bail_nuke_ah;
		}

		lwsl_info("h2c upgrade...\n");

		p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP2_SETTINGS);
		/* convert the peer's HTTP-Settings */
		n = lws_b64_decode_string(p, tbuf, sizeof(tbuf));
		if (n < 0) {
			lwsl_parser("HTTP2_SETTINGS too long\n");
			return 1;
		}

		/* adopt the header info */

		if (!wsi->h2.h2n) {
			wsi->h2.h2n = lws_zalloc(sizeof(*wsi->h2.h2n),
						   "h2n");
			if (!wsi->h2.h2n)
				return 1;
		}

		lws_h2_init(wsi);

		/* HTTP2 union */

		lws_h2_settings(wsi, &wsi->h2.h2n->set, (unsigned char *)tbuf, n);

		lws_hpack_dynamic_size(wsi, wsi->h2.h2n->set.s[
		                                      H2SET_HEADER_TABLE_SIZE]);

		strcpy(tbuf, "HTTP/1.1 101 Switching Protocols\x0d\x0a"
			      "Connection: Upgrade\x0d\x0a"
			      "Upgrade: h2c\x0d\x0a\x0d\x0a");
		m = (int)strlen(tbuf);
		n = lws_issue_raw(wsi, (unsigned char *)tbuf, m);
		if (n != m) {
			lwsl_debug("http2 switch: ERROR writing to socket\n");
			return 1;
		}

		lwsi_set_state(wsi, LRS_H2_AWAIT_PREFACE);
		wsi->upgraded_to_http2 = 1;

		return 0;
#endif
#if defined(LWS_ROLE_WS)
upgrade_ws:
		if (lws_process_ws_upgrade(wsi))
			goto bail_nuke_ah;

		return 0;
#endif
	} /* while all chars are handled */

	return 0;

bail_nuke_ah:
	/* drop the header info */
	lws_header_table_detach(wsi, 1);

	return 1;
}


LWS_VISIBLE int LWS_WARN_UNUSED_RESULT
lws_http_transaction_completed(struct lws *wsi)
{
	int n = NO_PENDING_TIMEOUT;

	if (lws_has_buffered_out(wsi)
#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
			|| wsi->http.comp_ctx.buflist_comp ||
	    wsi->http.comp_ctx.may_have_more
#endif
	) {
		/*
		 * ...so he tried to send something large as the http reply,
		 * it went as a partial, but he immediately said the
		 * transaction was completed.
		 *
		 * Defer the transaction completed until the last part of the
		 * partial is sent.
		 */
		lwsl_debug("%s: %p: deferring due to partial\n", __func__, wsi);
		wsi->http.deferred_transaction_completed = 1;
		lws_callback_on_writable(wsi);

		return 0;
	}

	lwsl_info("%s: wsi %p\n", __func__, wsi);

#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
	lws_http_compression_destroy(wsi);
#endif
	lws_access_log(wsi);

	if (!wsi->hdr_parsing_completed) {
		char peer[64];
		lws_get_peer_simple(wsi, peer, sizeof(peer) - 1);
		peer[sizeof(peer) - 1] = '\0';
		lwsl_notice("%s: (from %s) ignoring, ah parsing incomplete\n",
				__func__, peer);
		return 0;
	}

	/* if we can't go back to accept new headers, drop the connection */
	if (wsi->http2_substream)
		return 1;

	if (wsi->seen_zero_length_recv)
		return 1;

	if (wsi->http.conn_type != HTTP_CONNECTION_KEEP_ALIVE) {
		lwsl_info("%s: %p: close connection\n", __func__, wsi);
		return 1;
	}

	if (lws_bind_protocol(wsi, &wsi->vhost->protocols[0], __func__))
		return 1;

	/*
	 * otherwise set ourselves up ready to go again, but because we have no
	 * idea about the wsi writability, we make put it in a holding state
	 * until we can verify POLLOUT.  The part of this that confirms POLLOUT
	 * with no partials is in lws_server_socket_service() below.
	 */
	lwsl_debug("%s: %p: setting DEF_ACT from 0x%x\n", __func__,
		   wsi, wsi->wsistate);
	lwsi_set_state(wsi, LRS_DEFERRING_ACTION);
	wsi->http.tx_content_length = 0;
	wsi->http.tx_content_remain = 0;
	wsi->hdr_parsing_completed = 0;
	wsi->sending_chunked = 0;
#ifdef LWS_WITH_ACCESS_LOG
	wsi->http.access_log.sent = 0;
#endif

	if (wsi->vhost->keepalive_timeout)
		n = PENDING_TIMEOUT_HTTP_KEEPALIVE_IDLE;
	lws_set_timeout(wsi, n, wsi->vhost->keepalive_timeout);

	/*
	 * We already know we are on http1.1 / keepalive and the next thing
	 * coming will be another header set.
	 *
	 * If there is no pending rx and we still have the ah, drop it and
	 * reacquire a new ah when the new headers start to arrive.  (Otherwise
	 * we needlessly hog an ah indefinitely.)
	 *
	 * However if there is pending rx and we know from the keepalive state
	 * that is already at least the start of another header set, simply
	 * reset the existing header table and keep it.
	 */
	if (wsi->http.ah) {
		// lws_buflist_describe(&wsi->buflist, wsi);
		if (!lws_buflist_next_segment_len(&wsi->buflist, NULL)) {
			lwsl_debug("%s: %p: nothing in buflist, detaching ah\n",
				  __func__, wsi);
			lws_header_table_detach(wsi, 1);
#ifdef LWS_WITH_TLS
			/*
			 * additionally... if we are hogging an SSL instance
			 * with no pending pipelined headers (or ah now), and
			 * SSL is scarce, drop this connection without waiting
			 */

			if (wsi->vhost->tls.use_ssl &&
			    wsi->context->simultaneous_ssl_restriction &&
			    wsi->context->simultaneous_ssl ==
				   wsi->context->simultaneous_ssl_restriction) {
				lwsl_info("%s: simultaneous_ssl_restriction\n",
					  __func__);
				return 1;
			}
#endif
		} else {
			lwsl_info("%s: %p: resetting/keeping ah as pipeline\n",
				  __func__, wsi);
			lws_header_table_reset(wsi, 0);
			/*
			 * If we kept the ah, we should restrict the amount
			 * of time we are willing to keep it.  Otherwise it
			 * will be bound the whole time the connection remains
			 * open.
			 */
			lws_set_timeout(wsi, PENDING_TIMEOUT_HOLDING_AH,
					wsi->vhost->keepalive_timeout);
		}
		/* If we're (re)starting on headers, need other implied init */
		if (wsi->http.ah)
			wsi->http.ah->ues = URIES_IDLE;

		//lwsi_set_state(wsi, LRS_ESTABLISHED); // !!!
	} else
		if (lws_buflist_next_segment_len(&wsi->buflist, NULL))
			if (lws_header_table_attach(wsi, 0))
				lwsl_debug("acquired ah\n");

	lwsl_debug("%s: %p: keep-alive await new transaction (state 0x%x)\n",
			__func__, wsi, wsi->wsistate);
	lws_callback_on_writable(wsi);

	return 0;
}


LWS_VISIBLE int
lws_serve_http_file(struct lws *wsi, const char *file, const char *content_type,
		    const char *other_headers, int other_headers_len)
{
	struct lws_context *context = lws_get_context(wsi);
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	unsigned char *response = pt->serv_buf + LWS_PRE;
#if defined(LWS_WITH_RANGES)
	struct lws_range_parsing *rp = &wsi->http.range;
#endif
	int ret = 0, cclen = 8, n = HTTP_STATUS_OK;
	char cache_control[50], *cc = "no-store";
	lws_fop_flags_t fflags = LWS_O_RDONLY;
	const struct lws_plat_file_ops *fops;
	lws_filepos_t total_content_length;
	unsigned char *p = response;
	unsigned char *end = p + context->pt_serv_buf_size - LWS_PRE;
	const char *vpath;
#if defined(LWS_WITH_RANGES)
	int ranges;
#endif

	if (wsi->handling_404)
		n = HTTP_STATUS_NOT_FOUND;

	/*
	 * We either call the platform fops .open with first arg platform fops,
	 * or we call fops_zip .open with first arg platform fops, and fops_zip
	 * open will decide whether to switch to fops_zip or stay with fops_def.
	 *
	 * If wsi->http.fop_fd is already set, the caller already opened it
	 */
	if (!wsi->http.fop_fd) {
		fops = lws_vfs_select_fops(wsi->context->fops, file, &vpath);
		fflags |= lws_vfs_prepare_flags(wsi);
		wsi->http.fop_fd = fops->LWS_FOP_OPEN(wsi->context->fops,
							file, vpath, &fflags);
		if (!wsi->http.fop_fd) {
			lwsl_info("%s: Unable to open: '%s': errno %d\n",
				  __func__, file, errno);
			if (lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND,
						   NULL))
						return -1;
			return !wsi->http2_substream;
		}
	}
	wsi->http.filelen = lws_vfs_get_length(wsi->http.fop_fd);
	total_content_length = wsi->http.filelen;

#if defined(LWS_WITH_RANGES)
	ranges = lws_ranges_init(wsi, rp, wsi->http.filelen);

	lwsl_debug("Range count %d\n", ranges);
	/*
	 * no ranges -> 200;
	 *  1 range  -> 206 + Content-Type: normal; Content-Range;
	 *  more     -> 206 + Content-Type: multipart/byteranges
	 *  		Repeat the true Content-Type in each multipart header
	 *  		along with Content-Range
	 */
	if (ranges < 0) {
		/* it means he expressed a range in Range:, but it was illegal */
		lws_return_http_status(wsi,
				HTTP_STATUS_REQ_RANGE_NOT_SATISFIABLE, NULL);
		if (lws_http_transaction_completed(wsi))
			return -1; /* <0 means just hang up */

		lws_vfs_file_close(&wsi->http.fop_fd);

		return 0; /* == 0 means we did the transaction complete */
	}
	if (ranges)
		n = HTTP_STATUS_PARTIAL_CONTENT;
#endif

	if (lws_add_http_header_status(wsi, n, &p, end))
		return -1;

	if ((wsi->http.fop_fd->flags & (LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP |
		       LWS_FOP_FLAG_COMPR_IS_GZIP)) ==
	    (LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP | LWS_FOP_FLAG_COMPR_IS_GZIP)) {
		if (lws_add_http_header_by_token(wsi,
			WSI_TOKEN_HTTP_CONTENT_ENCODING,
			(unsigned char *)"gzip", 4, &p, end))
			return -1;
		lwsl_info("file is being provided in gzip\n");
	}
#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
	else {
		/*
		 * if we know its very compressible, and we can use
		 * compression, then use the most preferred compression
		 * method that the client said he will accept
		 */

		if (!strncmp(content_type, "text/", 5) ||
		    !strcmp(content_type, "application/javascript") ||
		    !strcmp(content_type, "image/svg+xml"))
			lws_http_compression_apply(wsi, NULL, &p, end, 0);
	}
#endif

	if (
#if defined(LWS_WITH_RANGES)
	    ranges < 2 &&
#endif
	    content_type && content_type[0])
		if (lws_add_http_header_by_token(wsi,
						 WSI_TOKEN_HTTP_CONTENT_TYPE,
						 (unsigned char *)content_type,
						 (int)strlen(content_type),
						 &p, end))
			return -1;

#if defined(LWS_WITH_RANGES)
	if (ranges >= 2) { /* multipart byteranges */
		lws_strncpy(wsi->http.multipart_content_type, content_type,
			sizeof(wsi->http.multipart_content_type));

		if (lws_add_http_header_by_token(wsi,
						 WSI_TOKEN_HTTP_CONTENT_TYPE,
						 (unsigned char *)
						 "multipart/byteranges; "
						 "boundary=_lws",
			 	 	 	 20, &p, end))
			return -1;

		/*
		 *  our overall content length has to include
		 *
		 *  - (n + 1) x "_lws\r\n"
		 *  - n x Content-Type: xxx/xxx\r\n
		 *  - n x Content-Range: bytes xxx-yyy/zzz\r\n
		 *  - n x /r/n
		 *  - the actual payloads (aggregated in rp->agg)
		 *
		 *  Precompute it for the main response header
		 */

		total_content_length = (lws_filepos_t)rp->agg +
				       6 /* final _lws\r\n */;

		lws_ranges_reset(rp);
		while (lws_ranges_next(rp)) {
			n = lws_snprintf(cache_control, sizeof(cache_control),
					"bytes %llu-%llu/%llu",
					rp->start, rp->end, rp->extent);

			total_content_length +=
				6 /* header _lws\r\n */ +
				/* Content-Type: xxx/xxx\r\n */
				14 + strlen(content_type) + 2 +
				/* Content-Range: xxxx\r\n */
				15 + n + 2 +
				2; /* /r/n */
		}

		lws_ranges_reset(rp);
		lws_ranges_next(rp);
	}

	if (ranges == 1) {
		total_content_length = (lws_filepos_t)rp->agg;
		n = lws_snprintf(cache_control, sizeof(cache_control),
				 "bytes %llu-%llu/%llu",
				 rp->start, rp->end, rp->extent);

		if (lws_add_http_header_by_token(wsi,
						 WSI_TOKEN_HTTP_CONTENT_RANGE,
						 (unsigned char *)cache_control,
						 n, &p, end))
			return -1;
	}

	wsi->http.range.inside = 0;

	if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_ACCEPT_RANGES,
					 (unsigned char *)"bytes", 5, &p, end))
		return -1;
#endif

	if (!wsi->http2_substream) {
		/* for http/1.1 ... */
		if (!wsi->sending_chunked
#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
				&& !wsi->http.lcs
#endif
		) {
			/* ... if not already using chunked and not using an
			 * http compression translation, then send the naive
			 * content length
			 */
			if (lws_add_http_header_content_length(wsi,
						total_content_length, &p, end))
				return -1;
		} else {

#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
			if (wsi->http.lcs) {

				/* ...otherwise, for http 1 it must go chunked.
				 * For the compression case, the reason is we
				 * compress on the fly and do not know the
				 * compressed content-length until it has all
				 * been sent.  Http/1.1 pipelining must be able
				 * to know where the transaction boundaries are
				 * ... so chunking...
				 */
				if (lws_add_http_header_by_token(wsi,
						WSI_TOKEN_HTTP_TRANSFER_ENCODING,
						(unsigned char *)"chunked", 7,
						&p, end))
					return -1;

				/*
				 * ...this is fun, isn't it :-)  For h1 that is
				 * using an http compression translation, the
				 * compressor must chunk its output privately.
				 *
				 * h2 doesn't need (or support) any of this
				 * crap.
				 */
				lwsl_debug("setting chunking\n");
				wsi->http.comp_ctx.chunking = 1;
			}
#endif
		}
	}

	if (wsi->cache_secs && wsi->cache_reuse) {
		if (!wsi->cache_revalidate) {
			cc = cache_control;
			cclen = sprintf(cache_control, "%s, max-age=%u",
				    intermediates[wsi->cache_intermediaries],
				    wsi->cache_secs);
		} else {
			cc = cache_control;
			cclen = sprintf(cache_control,
					"must-revalidate, %s, max-age=%u",
                                intermediates[wsi->cache_intermediaries],
                                                    wsi->cache_secs);

		}
	}

	/* Only add cache control if its not specified by any other_headers. */
	if (!other_headers ||
	    (!strstr(other_headers, "cache-control") &&
	     !strstr(other_headers, "Cache-Control"))) {
		if (lws_add_http_header_by_token(wsi,
				WSI_TOKEN_HTTP_CACHE_CONTROL,
				(unsigned char *)cc, cclen, &p, end))
			return -1;
	}

	if (other_headers) {
		if ((end - p) < other_headers_len)
			return -1;
		memcpy(p, other_headers, other_headers_len);
		p += other_headers_len;
	}

	if (lws_finalize_http_header(wsi, &p, end))
		return -1;

	ret = lws_write(wsi, response, p - response, LWS_WRITE_HTTP_HEADERS);
	if (ret != (p - response)) {
		lwsl_err("_write returned %d from %ld\n", ret,
			 (long)(p - response));
		return -1;
	}

	wsi->http.filepos = 0;
	lwsi_set_state(wsi, LRS_ISSUING_FILE);

	lws_callback_on_writable(wsi);

	return 0;
}

LWS_VISIBLE int lws_serve_http_file_fragment(struct lws *wsi)
{
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	struct lws_process_html_args args;
	lws_filepos_t amount, poss;
	unsigned char *p, *pstart;
#if defined(LWS_WITH_RANGES)
	unsigned char finished = 0;
#endif
	int n, m;

	lwsl_debug("wsi->http2_substream %d\n", wsi->http2_substream);

	do {

		/* priority 1: buffered output */

		if (lws_has_buffered_out(wsi)) {
			if (lws_issue_raw(wsi, NULL, 0) < 0) {
				lwsl_info("%s: closing\n", __func__);
				goto file_had_it;
			}
			break;
		}

		/* priority 2: buffered pre-compression-transform */

#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
	if (wsi->http.comp_ctx.buflist_comp ||
	    wsi->http.comp_ctx.may_have_more) {
		enum lws_write_protocol wp = LWS_WRITE_HTTP;

		lwsl_info("%s: completing comp partial (buflist %p, may %d)\n",
			   __func__, wsi->http.comp_ctx.buflist_comp,
			   wsi->http.comp_ctx.may_have_more);

		if (wsi->role_ops->write_role_protocol(wsi, NULL, 0, &wp) < 0) {
			lwsl_info("%s signalling to close\n", __func__);
			goto file_had_it;
		}
		lws_callback_on_writable(wsi);

		break;
	}
#endif

		if (wsi->http.filepos == wsi->http.filelen)
			goto all_sent;

		n = 0;

		pstart = pt->serv_buf + LWS_H2_FRAME_HEADER_LENGTH;

		p = pstart;

#if defined(LWS_WITH_RANGES)
		if (wsi->http.range.count_ranges && !wsi->http.range.inside) {

			lwsl_notice("%s: doing range start %llu\n", __func__,
				    wsi->http.range.start);

			if ((long long)lws_vfs_file_seek_cur(wsi->http.fop_fd,
						   wsi->http.range.start -
						   wsi->http.filepos) < 0)
				goto file_had_it;

			wsi->http.filepos = wsi->http.range.start;

			if (wsi->http.range.count_ranges > 1) {
				n =  lws_snprintf((char *)p,
						context->pt_serv_buf_size -
						LWS_H2_FRAME_HEADER_LENGTH,
					"_lws\x0d\x0a"
					"Content-Type: %s\x0d\x0a"
					"Content-Range: bytes "
						"%llu-%llu/%llu\x0d\x0a"
					"\x0d\x0a",
					wsi->http.multipart_content_type,
					wsi->http.range.start,
					wsi->http.range.end,
					wsi->http.range.extent);
				p += n;
			}

			wsi->http.range.budget = wsi->http.range.end -
						   wsi->http.range.start + 1;
			wsi->http.range.inside = 1;
		}
#endif

		poss = context->pt_serv_buf_size - n -
				LWS_H2_FRAME_HEADER_LENGTH;

		if (wsi->http.tx_content_length)
			if (poss > wsi->http.tx_content_remain)
				poss = wsi->http.tx_content_remain;

		/*
		 * if there is a hint about how much we will do well to send at
		 * one time, restrict ourselves to only trying to send that.
		 */
		if (wsi->protocol->tx_packet_size &&
		    poss > wsi->protocol->tx_packet_size)
			poss = wsi->protocol->tx_packet_size;

		if (wsi->role_ops->tx_credit) {
			lws_filepos_t txc = wsi->role_ops->tx_credit(wsi);

			if (!txc) {
				lwsl_info("%s: came here with no tx credit\n",
						__func__);
				return 0;
			}
			if (txc < poss)
				poss = txc;

			/*
			 * consumption of the actual payload amount sent will be
			 * handled when the role data frame is sent
			 */
		}

#if defined(LWS_WITH_RANGES)
		if (wsi->http.range.count_ranges) {
			if (wsi->http.range.count_ranges > 1)
				poss -= 7; /* allow for final boundary */
			if (poss > wsi->http.range.budget)
				poss = wsi->http.range.budget;
		}
#endif
		if (wsi->sending_chunked) {
			/* we need to drop the chunk size in here */
			p += 10;
			/* allow for the chunk to grow by 128 in translation */
			poss -= 10 + 128;
		}

		if (lws_vfs_file_read(wsi->http.fop_fd, &amount, p, poss) < 0)
			goto file_had_it; /* caller will close */

		if (wsi->sending_chunked)
			n = (int)amount;
		else
			n = lws_ptr_diff(p, pstart) + (int)amount;

		lwsl_debug("%s: sending %d\n", __func__, n);

		if (n) {
			lws_set_timeout(wsi, PENDING_TIMEOUT_HTTP_CONTENT,
					context->timeout_secs);

			if (wsi->interpreting) {
				args.p = (char *)p;
				args.len = n;
				args.max_len = (unsigned int)poss + 128;
				args.final = wsi->http.filepos + n ==
					     wsi->http.filelen;
				args.chunked = wsi->sending_chunked;
				if (user_callback_handle_rxflow(
				     wsi->vhost->protocols[
				     (int)wsi->protocol_interpret_idx].callback,
				     wsi, LWS_CALLBACK_PROCESS_HTML,
				     wsi->user_space, &args, 0) < 0)
					goto file_had_it;
				n = args.len;
				p = (unsigned char *)args.p;
			} else
				p = pstart;

#if defined(LWS_WITH_RANGES)
			if (wsi->http.range.send_ctr + 1 ==
				wsi->http.range.count_ranges && // last range
			    wsi->http.range.count_ranges > 1 && // was 2+ ranges (ie, multipart)
			    wsi->http.range.budget - amount == 0) {// final part
				n += lws_snprintf((char *)pstart + n, 6,
					"_lws\x0d\x0a"); // append trailing boundary
				lwsl_debug("added trailing boundary\n");
			}
#endif
			m = lws_write(wsi, p, n, wsi->http.filepos + amount ==
					wsi->http.filelen ?
					 LWS_WRITE_HTTP_FINAL : LWS_WRITE_HTTP);
			if (m < 0)
				goto file_had_it;

			wsi->http.filepos += amount;

#if defined(LWS_WITH_RANGES)
			if (wsi->http.range.count_ranges >= 1) {
				wsi->http.range.budget -= amount;
				if (wsi->http.range.budget == 0) {
					lwsl_notice("range budget exhausted\n");
					wsi->http.range.inside = 0;
					wsi->http.range.send_ctr++;

					if (lws_ranges_next(&wsi->http.range) < 1) {
						finished = 1;
						goto all_sent;
					}
				}
			}
#endif

			if (m != n) {
				/* adjust for what was not sent */
				if (lws_vfs_file_seek_cur(wsi->http.fop_fd,
							   m - n) ==
							     (lws_fileofs_t)-1)
					goto file_had_it;
			}
		}

all_sent:
		if ((!lws_has_buffered_out(wsi)
#if defined(LWS_WITH_HTTP_STREAM_COMPRESSION)
				&& !wsi->http.comp_ctx.buflist_comp &&
		    !wsi->http.comp_ctx.may_have_more
#endif
		    ) && (wsi->http.filepos >= wsi->http.filelen
#if defined(LWS_WITH_RANGES)
		    || finished)
#else
		)
#endif
		)
		     {
			lwsi_set_state(wsi, LRS_ESTABLISHED);
			/* we might be in keepalive, so close it off here */
			lws_vfs_file_close(&wsi->http.fop_fd);

			lwsl_debug("file completed\n");

			if (wsi->protocol->callback &&
			    user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_HTTP_FILE_COMPLETION,
					wsi->user_space, NULL, 0) < 0) {
					/*
					 * For http/1.x, the choices from
					 * transaction_completed are either
					 * 0 to use the connection for pipelined
					 * or nonzero to hang it up.
					 *
					 * However for http/2. while we are
					 * still interested in hanging up the
					 * nwsi if there was a network-level
					 * fatal error, simply completing the
					 * transaction is a matter of the stream
					 * state, not the root connection at the
					 * network level
					 */
					if (wsi->http2_substream)
						return 1;
					else
						return -1;
				}

			return 1;  /* >0 indicates completed */
		}
	} while (1); //(!lws_send_pipe_choked(wsi));

	lws_callback_on_writable(wsi);

	return 0; /* indicates further processing must be done */

file_had_it:
	lws_vfs_file_close(&wsi->http.fop_fd);

	return -1;
}


LWS_VISIBLE void
lws_server_get_canonical_hostname(struct lws_context *context,
				  const struct lws_context_creation_info *info)
{
	if (lws_check_opt(info->options,
			LWS_SERVER_OPTION_SKIP_SERVER_CANONICAL_NAME))
		return;
#if !defined(LWS_WITH_ESP32)
	/* find canonical hostname */
	gethostname((char *)context->canonical_hostname,
		    sizeof(context->canonical_hostname) - 1);

	lwsl_info(" canonical_hostname = %s\n", context->canonical_hostname);
#else
	(void)context;
#endif
}


LWS_VISIBLE LWS_EXTERN int
lws_chunked_html_process(struct lws_process_html_args *args,
			 struct lws_process_html_state *s)
{
	char *sp, buffer[32];
	const char *pc;
	int old_len, n;

	/* do replacements */
	sp = args->p;
	old_len = args->len;
	args->len = 0;
	s->start = sp;
	while (sp < args->p + old_len) {

		if (args->len + 7 >= args->max_len) {
			lwsl_err("Used up interpret padding\n");
			return -1;
		}

		if ((!s->pos && *sp == '$') || s->pos) {
			int hits = 0, hit = 0;

			if (!s->pos)
				s->start = sp;
			s->swallow[s->pos++] = *sp;
			if (s->pos == sizeof(s->swallow) - 1)
				goto skip;
			for (n = 0; n < s->count_vars; n++)
				if (!strncmp(s->swallow, s->vars[n], s->pos)) {
					hits++;
					hit = n;
				}
			if (!hits) {
skip:
				s->swallow[s->pos] = '\0';
				memcpy(s->start, s->swallow, s->pos);
				args->len++;
				s->pos = 0;
				sp = s->start + 1;
				continue;
			}
			if (hits == 1 && s->pos == (int)strlen(s->vars[hit])) {
				pc = s->replace(s->data, hit);
				if (!pc)
					pc = "NULL";
				n = (int)strlen(pc);
				s->swallow[s->pos] = '\0';
				if (n != s->pos) {
					memmove(s->start + n, s->start + s->pos,
						old_len - (sp - args->p));
					old_len += (n - s->pos) + 1;
				}
				memcpy(s->start, pc, n);
				args->len++;
				sp = s->start + 1;

				s->pos = 0;
			}
			sp++;
			continue;
		}

		args->len++;
		sp++;
	}

	if (args->chunked) {
		/* no space left for final chunk trailer */
		if (args->final && args->len + 7 >= args->max_len)
			return -1;

		n = sprintf(buffer, "%X\x0d\x0a", args->len);

		args->p -= n;
		memcpy(args->p, buffer, n);
		args->len += n;

		if (args->final) {
			sp = args->p + args->len;
			*sp++ = '\x0d';
			*sp++ = '\x0a';
			*sp++ = '0';
			*sp++ = '\x0d';
			*sp++ = '\x0a';
			*sp++ = '\x0d';
			*sp++ = '\x0a';
			args->len += 7;
		} else {
			sp = args->p + args->len;
			*sp++ = '\x0d';
			*sp++ = '\x0a';
			args->len += 2;
		}
	}

	return 0;
}
