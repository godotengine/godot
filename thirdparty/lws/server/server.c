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

const char * const method_names[] = {
	"GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE", "CONNECT", "HEAD",
#ifdef LWS_WITH_HTTP2
	":path",
#endif
	};

#if defined (LWS_WITH_ESP8266)
#undef memcpy
void *memcpy(void *dest, const void *src, size_t n)
{
	return ets_memcpy(dest, src, n);
}
#endif

int
lws_context_init_server(struct lws_context_creation_info *info,
			struct lws_vhost *vhost)
{
#if LWS_POSIX
	int n, opt = 1, limit = 1;
#endif
	lws_sockfd_type sockfd;
	struct lws_vhost *vh;
	struct lws *wsi;
	int m = 0;

	(void)method_names;
	(void)opt;
	/* set up our external listening socket we serve on */

	if (info->port == CONTEXT_PORT_NO_LISTEN ||
	    info->port == CONTEXT_PORT_NO_LISTEN_SERVER)
		return 0;

	vh = vhost->context->vhost_list;
	while (vh) {
		if (vh->listen_port == info->port) {
			if ((!info->iface && !vh->iface) ||
			    (info->iface && vh->iface &&
			    !strcmp(info->iface, vh->iface))) {
				vhost->listen_port = info->port;
				vhost->iface = info->iface;
				lwsl_notice(" using listen skt from vhost %s\n",
					    vh->name);
				return 0;
			}
		}
		vh = vh->vhost_next;
	}

#if LWS_POSIX
	(void)n;
#if defined(__linux__)
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

	if (sockfd == -1) {
#else
#if defined(LWS_WITH_ESP8266)
	sockfd = esp8266_create_tcp_listen_socket(vhost);
	if (!lws_sockfd_valid(sockfd)) {
#endif
#endif
		lwsl_err("ERROR opening socket\n");
		return 1;
	}
#if LWS_POSIX && !defined(LWS_WITH_ESP32)

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
			return 1;
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
		return 1;
	}

#if defined(LWS_WITH_IPV6) && defined(IPV6_V6ONLY)
	if (LWS_IPV6_ENABLED(vhost)) {
		if (vhost->options & LWS_SERVER_OPTION_IPV6_V6ONLY_MODIFY) {
			int value = (vhost->options &
				LWS_SERVER_OPTION_IPV6_V6ONLY_VALUE) ? 1 : 0;
			if (setsockopt(sockfd, IPPROTO_IPV6, IPV6_V6ONLY,
				      (const void*)&value, sizeof(value)) < 0) {
				compatible_close(sockfd);
				return 1;
			}
		}
	}
#endif

#if defined(__linux__) && defined(SO_REUSEPORT)
	n = lws_check_opt(vhost->options, LWS_SERVER_OPTION_ALLOW_LISTEN_SHARE);
#if LWS_MAX_SMP > 1
	n = 1;
#endif

	if (n)
		if (vhost->context->count_threads > 1)
			if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT,
					(const void *)&opt, sizeof(opt)) < 0) {
				compatible_close(sockfd);
				return 1;
			}
#endif
#endif
	lws_plat_set_socket_options(vhost, sockfd);

#if LWS_POSIX
	n = lws_socket_bind(vhost, sockfd, info->port, info->iface);
	if (n < 0)
		goto bail;
	info->port = n;
#endif
	vhost->listen_port = info->port;
	vhost->iface = info->iface;

	wsi = lws_zalloc(sizeof(struct lws), "listen wsi");
	if (wsi == NULL) {
		lwsl_err("Out of mem\n");
		goto bail;
	}
	wsi->context = vhost->context;
	wsi->desc.sockfd = sockfd;
	wsi->mode = LWSCM_SERVER_LISTENER;
	wsi->protocol = vhost->protocols;
	wsi->tsi = m;
	wsi->vhost = vhost;
	wsi->listener = 1;

#ifdef LWS_WITH_LIBUV
	if (LWS_LIBUV_ENABLED(vhost->context))
		lws_uv_initvhost(vhost, wsi);
#endif

	if (insert_wsi_socket_into_fds(vhost->context, wsi))
		goto bail;

	vhost->context->count_wsi_allocated++;
	vhost->lserv_wsi = wsi;

#if LWS_POSIX
	n = listen(wsi->desc.sockfd, LWS_SOMAXCONN);
	if (n < 0) {
		lwsl_err("listen failed with error %d\n", LWS_ERRNO);
		vhost->lserv_wsi = NULL;
		vhost->context->count_wsi_allocated--;
		remove_wsi_socket_from_fds(wsi);
		goto bail;
	}
	} /* for each thread able to independently listen */
#else
#if defined(LWS_WITH_ESP8266)
	esp8266_tcp_stream_bind(wsi->desc.sockfd, info->port, wsi);
#endif
#endif
	if (!lws_check_opt(info->options, LWS_SERVER_OPTION_EXPLICIT_VHOSTS)) {
#ifdef LWS_WITH_UNIX_SOCK
		if (LWS_UNIX_SOCK_ENABLED(vhost))
			lwsl_info(" Listening on \"%s\"\n", info->iface);
		else
#endif
			lwsl_info(" Listening on port %d\n", info->port);
        }

	return 0;

bail:
	compatible_close(sockfd);

	return 1;
}

#if defined(LWS_WITH_ESP8266)
#undef strchr
#define strchr ets_strchr
#endif

struct lws_vhost *
lws_select_vhost(struct lws_context *context, int port, const char *servername)
{
	struct lws_vhost *vhost = context->vhost_list;
	const char *p;
	int n, m, colon;

	n = strlen(servername);
	colon = n;
	p = strchr(servername, ':');
	if (p)
		colon = p - servername;

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
		m = strlen(vhost->name);
		if (port == vhost->listen_port &&
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
		if (port == vhost->listen_port) {
			lwsl_info("vhost match to %s based on port %d\n",
					vhost->name, port);
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
	int n = strlen(file);
	const struct lws_protocol_vhost_options *pvo = NULL;

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
#if !defined(_WIN32_WCE) && !defined(LWS_WITH_ESP8266)
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
	char path[256], sym[512];
	unsigned char *p = (unsigned char *)sym + 32 + LWS_PRE, *start = p;
	unsigned char *end = p + sizeof(sym) - 32 - LWS_PRE;
#if !defined(WIN32) && LWS_POSIX && !defined(LWS_WITH_ESP32)
	size_t len;
#endif
	int n;

	lws_snprintf(path, sizeof(path) - 1, "%s/%s", origin, uri);

#if !defined(_WIN32_WCE) && !defined(LWS_WITH_ESP8266)

	fflags |= lws_vfs_prepare_flags(wsi);

	do {
		spin++;
		fops = lws_vfs_select_fops(wsi->context->fops, path, &vpath);

		if (wsi->u.http.fop_fd)
			lws_vfs_file_close(&wsi->u.http.fop_fd);

		wsi->u.http.fop_fd = fops->LWS_FOP_OPEN(wsi->context->fops,
							path, vpath, &fflags);
		if (!wsi->u.http.fop_fd) {
			lwsl_err("Unable to open '%s'\n", path);

			return -1;
		}

		/* if it can't be statted, don't try */
		if (fflags & LWS_FOP_FLAG_VIRTUAL)
			break;
#if defined(LWS_WITH_ESP32)
		break;
#endif
#if !defined(WIN32)
		if (fstat(wsi->u.http.fop_fd->fd, &st)) {
			lwsl_info("unable to stat %s\n", path);
			goto bail;
		}
#else
#if defined(LWS_HAVE__STAT32I64)
		if (_stat32i64(path, &st)) {
			lwsl_info("unable to stat %s\n", path);
			goto bail;
		}
#else
		if (stat(path, &st)) {
			lwsl_info("unable to stat %s\n", path);
			goto bail;
		}
#endif
#endif

		wsi->u.http.fop_fd->mod_time = (uint32_t)st.st_mtime;
		fflags |= LWS_FOP_FLAG_MOD_TIME_VALID;

#if !defined(WIN32) && LWS_POSIX && !defined(LWS_WITH_ESP32)
		if ((S_IFMT & st.st_mode) == S_IFLNK) {
			len = readlink(path, sym, sizeof(sym) - 1);
			if (len) {
				lwsl_err("Failed to read link %s\n", path);
				goto bail;
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
		    (unsigned long long)lws_vfs_get_length(wsi->u.http.fop_fd),
		    (unsigned long)lws_vfs_get_mod_time(wsi->u.http.fop_fd));

	/* disable ranges if IF_RANGE token invalid */

	if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_IF_RANGE))
		if (strcmp(sym, lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_IF_RANGE)))
			/* differs - defeat Range: */
			wsi->u.http.ah->frag_index[WSI_TOKEN_HTTP_RANGE] = 0;

	if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_IF_NONE_MATCH)) {
		/*
		 * he thinks he has some version of it already,
		 * check if the tag matches
		 */
		if (!strcmp(sym, lws_hdr_simple_ptr(wsi,
					WSI_TOKEN_HTTP_IF_NONE_MATCH))) {

			lwsl_debug("%s: ETAG match %s %s\n", __func__,
				   uri, origin);

			/* we don't need to send the payload */
			if (lws_add_http_header_status(wsi,
					HTTP_STATUS_NOT_MODIFIED, &p, end))
				return -1;

			if (lws_add_http_header_by_token(wsi,
					WSI_TOKEN_HTTP_ETAG,
					(unsigned char *)sym, n, &p, end))
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

			lws_vfs_file_close(&wsi->u.http.fop_fd);

			return lws_http_transaction_completed(wsi);
		}
	}

	if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_ETAG,
			(unsigned char *)sym, n, &p, end))
		return -1;
#endif

	mimetype = lws_get_mimetype(path, m);
	if (!mimetype) {
		lwsl_err("unknown mimetype for %s\n", path);
               goto bail;
	}
	if (!mimetype[0])
		lwsl_debug("sending no mimetype for %s\n", path);

	wsi->sending_chunked = 0;

	/*
	 * check if this is in the list of file suffixes to be interpreted by
	 * a protocol
	 */
	while (pvo) {
		n = strlen(path);
		if (n > (int)strlen(pvo->name) &&
		    !strcmp(&path[n - strlen(pvo->name)], pvo->name)) {
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

		if (lws_bind_protocol(wsi, pp))
			return 1;
		args.p = (char *)p;
		args.max_len = end - p;
		if (pp->callback(wsi, LWS_CALLBACK_ADD_HEADERS,
					  wsi->user_space, &args, 0))
			return -1;
		p = (unsigned char *)args.p;
	}

	n = lws_serve_http_file(wsi, path, mimetype, (char *)start, p - start);

	if (n < 0 || ((n > 0) && lws_http_transaction_completed(wsi)))
		return -1; /* error or can't reuse connection: close the socket */

	return 0;
bail:

	return -1;
}

const struct lws_http_mount *
lws_find_mount(struct lws *wsi, const char *uri_ptr, int uri_len)
{
	const struct lws_http_mount *hm, *hit = NULL;
	int best = 0;

	hm = wsi->vhost->mount_list;
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

#if LWS_POSIX

static int
lws_find_string_in_file(const char *filename, const char *string, int stringlen)
{
	char buf[128];
	int fd, match = 0, pos = 0, n = 0, hit = 0;

	fd = open(filename, O_RDONLY);
	if (fd < 0) {
		lwsl_err("can't open auth file: %s\n", filename);
		return 1;
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

static int
lws_unauthorised_basic_auth(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	unsigned char *start = pt->serv_buf + LWS_PRE,
		      *p = start, *end = p + 512;
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

#endif

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

	for (n = 0; n < ARRAY_SIZE(methods); n++)
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

	for (n = 0; n < ARRAY_SIZE(methods); n++)
		if (lws_hdr_total_length(wsi, methods[n])) {
			*puri_ptr = lws_hdr_simple_ptr(wsi, methods[n]);
			*puri_len = lws_hdr_total_length(wsi, methods[n]);
			return n;
		}

	return -1;
}

int
lws_http_action(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	enum http_connection_type connection_type;
	enum http_version request_version;
	char content_length_str[32];
	struct lws_process_html_args args;
	const struct lws_http_mount *hit = NULL;
	unsigned int n;
	char http_version_str[10];
	char http_conn_str[20];
	int http_version_len;
	char *uri_ptr = NULL, *s;
	int uri_len = 0, meth;
	static const char * const oprot[] = {
		"http://", "https://"
	};

	meth = lws_http_get_uri_and_method(wsi, &uri_ptr, &uri_len);
	if (meth < 0 || meth >= ARRAY_SIZE(method_names))
		goto bail_nuke_ah;

	/* we insist on absolute paths */

	if (!uri_ptr || uri_ptr[0] != '/') {
		lws_return_http_status(wsi, HTTP_STATUS_FORBIDDEN, NULL);

		goto bail_nuke_ah;
	}

	lwsl_info("Method: '%s' (%d), request for '%s'\n", method_names[meth],
		  meth, uri_ptr);

	if (lws_ensure_user_space(wsi))
		goto bail_nuke_ah;

	/* HTTP header had a content length? */

	wsi->u.http.rx_content_length = 0;
	if (lws_hdr_total_length(wsi, WSI_TOKEN_POST_URI) ||
		lws_hdr_total_length(wsi, WSI_TOKEN_PATCH_URI) ||
		lws_hdr_total_length(wsi, WSI_TOKEN_PUT_URI))
		wsi->u.http.rx_content_length = 100 * 1024 * 1024;

	if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_CONTENT_LENGTH)) {
		lws_hdr_copy(wsi, content_length_str,
			     sizeof(content_length_str) - 1,
			     WSI_TOKEN_HTTP_CONTENT_LENGTH);
		wsi->u.http.rx_content_length = atoll(content_length_str);
	}

	if (wsi->http2_substream) {
		wsi->u.http.request_version = HTTP_VERSION_2;
	} else {
		/* http_version? Default to 1.0, override with token: */
		request_version = HTTP_VERSION_1_0;

		/* Works for single digit HTTP versions. : */
		http_version_len = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP);
		if (http_version_len > 7) {
			lws_hdr_copy(wsi, http_version_str,
				     sizeof(http_version_str) - 1,
				     WSI_TOKEN_HTTP);
			if (http_version_str[5] == '1' &&
			    http_version_str[7] == '1')
				request_version = HTTP_VERSION_1_1;
		}
		wsi->u.http.request_version = request_version;

		/* HTTP/1.1 defaults to "keep-alive", 1.0 to "close" */
		if (request_version == HTTP_VERSION_1_1)
			connection_type = HTTP_CONNECTION_KEEP_ALIVE;
		else
			connection_type = HTTP_CONNECTION_CLOSE;

		/* Override default if http "Connection:" header: */
		if (lws_hdr_total_length(wsi, WSI_TOKEN_CONNECTION)) {
			lws_hdr_copy(wsi, http_conn_str,
				     sizeof(http_conn_str) - 1,
				     WSI_TOKEN_CONNECTION);
			http_conn_str[sizeof(http_conn_str) - 1] = '\0';
			if (!strcasecmp(http_conn_str, "keep-alive"))
				connection_type = HTTP_CONNECTION_KEEP_ALIVE;
			else
				if (!strcasecmp(http_conn_str, "close"))
					connection_type = HTTP_CONNECTION_CLOSE;
		}
		wsi->u.http.connection_type = connection_type;
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
#ifdef LWS_OPENSSL_SUPPORT
	if (wsi->redirect_to_https) {
		/*
		 * we accepted http:// only so we could redirect to
		 * https://, so issue the redirect.  Create the redirection
		 * URI from the host: header and ignore the path part
		 */
		unsigned char *start = pt->serv_buf + LWS_PRE, *p = start,
			      *end = p + 512;

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
	lws_prepare_access_log_info(wsi, uri_ptr, meth);
#endif

	/* can we serve it from the mount list? */

	hit = lws_find_mount(wsi, uri_ptr, uri_len);
	if (!hit) {
		/* deferred cleanup and reset to protocols[0] */

		lwsl_info("no hit\n");

		if (lws_bind_protocol(wsi, &wsi->vhost->protocols[0]))
			return 1;

		n = wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP,
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
		unsigned char *start = pt->serv_buf + LWS_PRE,
			      *p = start, *end = p + 512;

		lwsl_debug("Doing 301 '%s' org %s\n", s, hit->origin);

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

#if LWS_POSIX
	/* basic auth? */

	if (hit->basic_auth_login_file) {
		char b64[160], plain[(sizeof(b64) * 3) / 4];
		int m;

		/* Did he send auth? */
		if (!lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_AUTHORIZATION))
			return lws_unauthorised_basic_auth(wsi);

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

		m = lws_b64_decode_string(b64 + 6, plain, sizeof(plain));
		if (m < 0) {
			lwsl_err("plain auth too long\n");
			goto transaction_result_n;
		}

		if (!lws_find_string_in_file(hit->basic_auth_login_file,
					     plain, m)) {
			lwsl_err("basic auth lookup failed\n");
			return lws_unauthorised_basic_auth(wsi);
		}

		lwsl_notice("basic auth accepted\n");

		/* accept the auth */
	}
#endif

#if defined(LWS_WITH_HTTP_PROXY)
	/*
	 * The mount is a reverse proxy?
	 */

	if (hit->origin_protocol == LWSMPRO_HTTPS ||
	    hit->origin_protocol == LWSMPRO_HTTP)  {
		struct lws_client_connect_info i;
		char ads[96], rpath[256], *pcolon, *pslash, *p;
		int n, na;

		memset(&i, 0, sizeof(i));
		i.context = lws_get_context(wsi);

		pcolon = strchr(hit->origin, ':');
		pslash = strchr(hit->origin, '/');
		if (!pslash) {
			lwsl_err("Proxy mount origin '%s' must have /\n",
				 hit->origin);
			return -1;
		}
		if (pcolon > pslash)
			pcolon = NULL;
		
		if (pcolon)
			n = pcolon - hit->origin;
		else
			n = pslash - hit->origin;

		if (n >= sizeof(ads) - 2)
			n = sizeof(ads) - 2;

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
		
		lws_snprintf(rpath, sizeof(rpath) - 1, "/%s/%s", pslash + 1,
			     uri_ptr + hit->mountpoint_len);
		lws_clean_url(rpath);
		na = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_URI_ARGS);
		if (na) {
			p = rpath + strlen(rpath);
			*p++ = '?';
			lws_hdr_copy(wsi, p, &rpath[sizeof(rpath) - 1] - p,
				     WSI_TOKEN_HTTP_URI_ARGS);
			while (--na) {
				if (*p == '\0')
					*p = '&';
				p++;
			}
		}
				

		i.path = rpath;
		i.host = i.address;
		i.origin = NULL;
		i.method = "GET";
		i.parent_wsi = wsi;
		i.uri_replace_from = hit->origin;
		i.uri_replace_to = hit->mountpoint;

		lwsl_notice("proxying to %s port %d url %s, ssl %d, from %s, to %s\n",
				i.address, i.port, i.path, i.ssl_connection,
				i.uri_replace_from, i.uri_replace_to);
	
		if (!lws_client_connect_via_info(&i)) {
			lwsl_err("proxy connect fail\n");
			return 1;
		}

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

		if (lws_bind_protocol(wsi, pp))
			return 1;

		args.p = uri_ptr;
		args.len = uri_len;
		args.max_len = hit->auth_mask;
		args.final = 0; /* used to signal callback dealt with it */

		n = wsi->protocol->callback(wsi, LWS_CALLBACK_CHECK_ACCESS_RIGHTS,
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
			n = wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP,
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

	n = strlen(s);
	if (s[0] == '\0' || (n == 1 && s[n - 1] == '/'))
		s = (char *)hit->def;
	if (!s)
		s = "index.html";

	wsi->cache_secs = hit->cache_max_age;
	wsi->cache_reuse = hit->cache_reusable;
	wsi->cache_revalidate = hit->cache_revalidate;
	wsi->cache_intermediaries = hit->cache_intermediaries;

	n = lws_http_serve(wsi, s, hit->origin, hit);
	if (n) {
		/*
		 * 	lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, NULL);
		 */
		if (hit->protocol) {
			const struct lws_protocols *pp = lws_vhost_name_to_protocol(
					wsi->vhost, hit->protocol);

			if (lws_bind_protocol(wsi, pp))
				return 1;

			n = pp->callback(wsi, LWS_CALLBACK_HTTP,
					 wsi->user_space,
					 uri_ptr + hit->mountpoint_len,
					 uri_len - hit->mountpoint_len);
		} else
			n = wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP,
				    wsi->user_space, uri_ptr, uri_len);
	}

after:
	if (n) {
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
	if (wsi->state != LWSS_HTTP_ISSUING_FILE) {
		/* Prepare to read body if we have a content length: */
		lwsl_debug("wsi->u.http.rx_content_length %lld %d %d\n",
			   (long long)wsi->u.http.rx_content_length,
			   wsi->upgraded_to_http2, wsi->http2_substream);
		if (wsi->u.http.rx_content_length > 0) {
			lwsl_notice("%s: %p: LWSS_HTTP_BODY state set\n",
				    __func__, wsi);
			wsi->state = LWSS_HTTP_BODY;
			wsi->u.http.rx_content_remain =
					wsi->u.http.rx_content_length;
		}
	}

	return 0;

bail_nuke_ah:
	/* we're closing, losing some rx is OK */
	lws_header_table_force_to_detachable_state(wsi);
	lws_header_table_detach(wsi, 1);

	return 1;

#if LWS_POSIX
transaction_result_n:
	lws_return_http_status(wsi, n, NULL);

	return lws_http_transaction_completed(wsi);
#endif
}

static int
lws_server_init_wsi_for_ws(struct lws *wsi)
{
	int n;

	wsi->state = LWSS_ESTABLISHED;
	lws_restart_ws_ping_pong_timer(wsi);

	/*
	 * create the frame buffer for this connection according to the
	 * size mentioned in the protocol definition.  If 0 there, use
	 * a big default for compatibility
	 */

	n = wsi->protocol->rx_buffer_size;
	if (!n)
		n = wsi->context->pt_serv_buf_size;
	n += LWS_PRE;
	wsi->u.ws.rx_ubuf = lws_malloc(n + 4 /* 0x0000ffff zlib */, "rx_ubuf");
	if (!wsi->u.ws.rx_ubuf) {
		lwsl_err("Out of Mem allocating rx buffer %d\n", n);
		return 1;
	}
	wsi->u.ws.rx_ubuf_alloc = n;
	lwsl_debug("Allocating RX buffer %d\n", n);

#if LWS_POSIX && !defined(LWS_WITH_ESP32)
	if (!wsi->parent_carries_io)
		if (setsockopt(wsi->desc.sockfd, SOL_SOCKET, SO_SNDBUF,
		       (const char *)&n, sizeof n)) {
			lwsl_warn("Failed to set SNDBUF to %d", n);
			return 1;
		}
#endif

	/* notify user code that we're ready to roll */

	if (wsi->protocol->callback)
		if (wsi->protocol->callback(wsi, LWS_CALLBACK_ESTABLISHED,
					    wsi->user_space,
#ifdef LWS_OPENSSL_SUPPORT
					    wsi->ssl,
#else
					    NULL,
#endif
					    0))
			return 1;

	return 0;
}

int
lws_handshake_server(struct lws *wsi, unsigned char **buf, size_t len)
{
	int protocol_len, n = 0, hit, non_space_char_found = 0, m;
	struct lws_context *context = lws_get_context(wsi);
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	struct _lws_header_related hdr;
	struct allocated_headers *ah;
	unsigned char *obuf = *buf;
	char protocol_list[128];
	char protocol_name[64];
	size_t olen = len;
	char *p;

	if (len >= 10000000) {
		lwsl_err("%s: assert: len %ld\n", __func__, (long)len);
		assert(0);
	}

	if (!wsi->u.hdr.ah) {
		lwsl_err("%s: assert: NULL ah\n", __func__);
		assert(0);
	}

	lwsl_hexdump(*buf, len);

	while (len--) {
		wsi->more_rx_waiting = !!len;

		if (wsi->mode != LWSCM_HTTP_SERVING &&
		    wsi->mode != LWSCM_HTTP2_SERVING &&
		    wsi->mode != LWSCM_HTTP_SERVING_ACCEPTED) {
			lwsl_err("%s: bad wsi mode %d\n", __func__, wsi->mode);
			goto bail_nuke_ah;
		}

		m = lws_parse(wsi, *(*buf)++);
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
				                        raw_protocol_index]);
				lwsl_info("transition to raw vh %s prot %d\n",
					  wsi->vhost->name,
					  wsi->vhost->raw_protocol_index);
				if ((wsi->protocol->callback)(wsi,
						LWS_CALLBACK_RAW_ADOPT,
						wsi->user_space, NULL, 0))
					goto bail_nuke_ah;

				lws_header_table_force_to_detachable_state(wsi);
				lws_union_transition(wsi, LWSCM_RAW);
				lws_header_table_detach(wsi, 1);

				if (m == 2 && (wsi->protocol->callback)(wsi,
						LWS_CALLBACK_RAW_RX,
						wsi->user_space, obuf, olen))
					return 1;

				return 0;
			}
			lwsl_info("lws_parse failed\n");
			goto bail_nuke_ah;
		}

		if (wsi->u.hdr.parser_state != WSI_PARSING_COMPLETE)
			continue;

		lwsl_parser("%s: lws_parse sees parsing complete\n", __func__);
		lwsl_debug("%s: wsi->more_rx_waiting=%d\n", __func__,
				wsi->more_rx_waiting);

		/* select vhost */

		if (lws_hdr_total_length(wsi, WSI_TOKEN_HOST)) {
			struct lws_vhost *vhost = lws_select_vhost(
				context, wsi->vhost->listen_port,
				lws_hdr_simple_ptr(wsi, WSI_TOKEN_HOST));

			if (vhost)
				wsi->vhost = vhost;
		} else
			lwsl_info("no host\n");

		if (wsi->mode != LWSCM_HTTP2_SERVING) {
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
				ua[sizeof(ua) - 1] = '\0';
				while (rej) {
					if (strstr(ua, rej->name)) {
#ifdef LWS_WITH_ACCESS_LOG
						char *uri_ptr = NULL;
						int meth, uri_len;
#endif

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
								uri_ptr, meth);

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
					rej = rej->next;
				}
			}
		}


		if (lws_hdr_total_length(wsi, WSI_TOKEN_CONNECT)) {
			lwsl_info("Changing to RAW mode\n");
			m = 0;
			goto raw_transition;
		}

		wsi->mode = LWSCM_PRE_WS_SERVING_ACCEPT;
		lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);

		/* is this websocket protocol or normal http 1.0? */

		if (lws_hdr_total_length(wsi, WSI_TOKEN_UPGRADE)) {
			if (!strcasecmp(lws_hdr_simple_ptr(wsi, WSI_TOKEN_UPGRADE),
					"websocket")) {
				wsi->vhost->conn_stats.ws_upg++;
				lwsl_info("Upgrade to ws\n");
				goto upgrade_ws;
			}
#ifdef LWS_WITH_HTTP2
			if (!strcasecmp(lws_hdr_simple_ptr(wsi, WSI_TOKEN_UPGRADE),
					"h2c")) {
				wsi->vhost->conn_stats.h2_upg++;
				lwsl_info("Upgrade to h2c\n");
				goto upgrade_h2c;
			}
#endif
			lwsl_info("Unknown upgrade\n");
			/* dunno what he wanted to upgrade to */
			goto bail_nuke_ah;
		}

		/* no upgrade ack... he remained as HTTP */

		lwsl_info("No upgrade\n");
		ah = wsi->u.hdr.ah;

		lws_union_transition(wsi, LWSCM_HTTP_SERVING_ACCEPTED);
		wsi->state = LWSS_HTTP;
		wsi->u.http.fop_fd = NULL;

		/* expose it at the same offset as u.hdr */
		wsi->u.http.ah = ah;
		lwsl_debug("%s: wsi %p: ah %p\n", __func__, (void *)wsi,
			   (void *)wsi->u.hdr.ah);

		n = lws_http_action(wsi);

		return n;

#ifdef LWS_WITH_HTTP2
upgrade_h2c:
		if (!lws_hdr_total_length(wsi, WSI_TOKEN_HTTP2_SETTINGS)) {
			lwsl_info("missing http2_settings\n");
			goto bail_nuke_ah;
		}

		lwsl_info("h2c upgrade...\n");

		p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP2_SETTINGS);
		/* convert the peer's HTTP-Settings */
		n = lws_b64_decode_string(p, protocol_list,
					  sizeof(protocol_list));
		if (n < 0) {
			lwsl_parser("HTTP2_SETTINGS too long\n");
			return 1;
		}

		/* adopt the header info */

		ah = wsi->u.hdr.ah;

		lws_union_transition(wsi, LWSCM_HTTP2_SERVING);

		/* http2 union member has http union struct at start */
		wsi->u.http.ah = ah;

		if (!wsi->u.h2.h2n) {
			wsi->u.h2.h2n = lws_zalloc(sizeof(*wsi->u.h2.h2n), "h2n");
			if (!wsi->u.h2.h2n)
				return 1;
		}

		lws_h2_init(wsi);

		/* HTTP2 union */

		lws_h2_settings(wsi, &wsi->u.h2.h2n->set,
				(unsigned char *)protocol_list, n);

		lws_hpack_dynamic_size(wsi, wsi->u.h2.h2n->set.s[
		                                      H2SET_HEADER_TABLE_SIZE]);

		strcpy(protocol_list, "HTTP/1.1 101 Switching Protocols\x0d\x0a"
				      "Connection: Upgrade\x0d\x0a"
				      "Upgrade: h2c\x0d\x0a\x0d\x0a");
		n = lws_issue_raw(wsi, (unsigned char *)protocol_list,
					strlen(protocol_list));
		if (n != strlen(protocol_list)) {
			lwsl_debug("http2 switch: ERROR writing to socket\n");
			return 1;
		}

		wsi->state = LWSS_HTTP2_AWAIT_CLIENT_PREFACE;

		return 0;
#endif

upgrade_ws:
		if (!wsi->protocol)
			lwsl_err("NULL protocol at lws_read\n");

		/*
		 * It's websocket
		 *
		 * Select the first protocol we support from the list
		 * the client sent us.
		 *
		 * Copy it to remove header fragmentation
		 */

		if (lws_hdr_copy(wsi, protocol_list, sizeof(protocol_list) - 1,
				 WSI_TOKEN_PROTOCOL) < 0) {
			lwsl_err("protocol list too long");
			goto bail_nuke_ah;
		}

		protocol_len = lws_hdr_total_length(wsi, WSI_TOKEN_PROTOCOL);
		protocol_list[protocol_len] = '\0';
		p = protocol_list;
		hit = 0;

		while (*p && !hit) {
			n = 0;
			non_space_char_found = 0;
			while (n < sizeof(protocol_name) - 1 &&
			       *p && *p != ',') {
				/* ignore leading spaces */
				if (!non_space_char_found && *p == ' ') {
					n++;
					continue;
				}
				non_space_char_found = 1;
				protocol_name[n++] = *p++;
			}
			protocol_name[n] = '\0';
			if (*p)
				p++;

			lwsl_info("checking %s\n", protocol_name);

			n = 0;
			while (wsi->vhost->protocols[n].callback) {
				lwsl_info("try %s\n",
					  wsi->vhost->protocols[n].name);

				if (wsi->vhost->protocols[n].name &&
				    !strcmp(wsi->vhost->protocols[n].name,
					    protocol_name)) {
					wsi->protocol = &wsi->vhost->protocols[n];
					hit = 1;
					break;
				}

				n++;
			}
		}

		/* we didn't find a protocol he wanted? */

		if (!hit) {
			if (lws_hdr_simple_ptr(wsi, WSI_TOKEN_PROTOCOL)) {
				lwsl_info("No protocol from \"%s\" supported\n",
					 protocol_list);
				goto bail_nuke_ah;
			}
			/*
			 * some clients only have one protocol and
			 * do not send the protocol list header...
			 * allow it and match to the vhost's default
			 * protocol (which itself defaults to zero)
			 */
			lwsl_info("defaulting to prot handler %d\n",
				wsi->vhost->default_protocol_index);
			n = wsi->vhost->default_protocol_index;
			wsi->protocol = &wsi->vhost->protocols[
				      (int)wsi->vhost->default_protocol_index];
		}

		/* allocate wsi->user storage */
		if (lws_ensure_user_space(wsi))
			goto bail_nuke_ah;

		/*
		 * Give the user code a chance to study the request and
		 * have the opportunity to deny it
		 */
		if ((wsi->protocol->callback)(wsi,
				LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION,
				wsi->user_space,
			      lws_hdr_simple_ptr(wsi, WSI_TOKEN_PROTOCOL), 0)) {
			lwsl_warn("User code denied connection\n");
			goto bail_nuke_ah;
		}

		/*
		 * Perform the handshake according to the protocol version the
		 * client announced
		 */

		switch (wsi->ietf_spec_revision) {
		case 13:
			lwsl_parser("lws_parse calling handshake_04\n");
			if (handshake_0405(context, wsi)) {
				lwsl_info("hs0405 has failed the connection\n");
				goto bail_nuke_ah;
			}
			break;

		default:
			lwsl_info("Unknown client spec version %d\n",
				  wsi->ietf_spec_revision);
			goto bail_nuke_ah;
		}

		lws_same_vh_protocol_insert(wsi, n);

		/* we are upgrading to ws, so http/1.1 and keepalive +
		 * pipelined header considerations about keeping the ah around
		 * no longer apply.  However it's common for the first ws
		 * protocol data to have been coalesced with the browser
		 * upgrade request and to already be in the ah rx buffer.
		 */

		lwsl_info("%s: %p: inheriting ws ah (rxpos:%d, rxlen:%d)\n",
			  __func__, wsi, wsi->u.hdr.ah->rxpos,
			  wsi->u.hdr.ah->rxlen);
		lws_pt_lock(pt);
		hdr = wsi->u.hdr;

		lws_union_transition(wsi, LWSCM_WS_SERVING);
		/*
		 * first service is WS mode will notice this, use the RX and
		 * then detach the ah (caution: we are not in u.hdr union
		 * mode any more then... ah_temp member is at start the same
		 * though)
		 *
		 * Because rxpos/rxlen shows something in the ah, we will get
		 * service guaranteed next time around the event loop
		 *
		 * All union members begin with hdr, so we can use it even
		 * though we transitioned to ws union mode (the ah detach
		 * code uses it anyway).
		 */
		wsi->u.hdr = hdr;
		lws_pt_unlock(pt);

		lws_server_init_wsi_for_ws(wsi);
		lwsl_parser("accepted v%02d connection\n",
			    wsi->ietf_spec_revision);

		/* !!! drop ah unreservedly after ESTABLISHED */
		if (!wsi->more_rx_waiting) {
			lws_header_table_force_to_detachable_state(wsi);
			lws_header_table_detach(wsi, 1);
		}

		return 0;
	} /* while all chars are handled */

	return 0;

bail_nuke_ah:
	/* drop the header info */
	/* we're closing, losing some rx is OK */
	lws_header_table_force_to_detachable_state(wsi);
	lws_header_table_detach(wsi, 1);

	return 1;
}


static int
lws_get_idlest_tsi(struct lws_context *context)
{
	unsigned int lowest = ~0;
	int n = 0, hit = -1;

	for (; n < context->count_threads; n++) {
		if ((unsigned int)context->pt[n].fds_count !=
		    context->fd_limit_per_thread - 1 &&
		    (unsigned int)context->pt[n].fds_count < lowest) {
			lowest = context->pt[n].fds_count;
			hit = n;
		}
	}

	return hit;
}

struct lws *
lws_create_new_server_wsi(struct lws_vhost *vhost)
{
	struct lws *new_wsi;
	int n = lws_get_idlest_tsi(vhost->context);

	if (n < 0) {
		lwsl_err("no space for new conn\n");
		return NULL;
	}

	new_wsi = lws_zalloc(sizeof(struct lws), "new server wsi");
	if (new_wsi == NULL) {
		lwsl_err("Out of memory for new connection\n");
		return NULL;
	}

	new_wsi->tsi = n;
	lwsl_debug("new wsi %p joining vhost %s, tsi %d\n", new_wsi,
		   vhost->name, new_wsi->tsi);

	new_wsi->vhost = vhost;
	new_wsi->context = vhost->context;
	new_wsi->pending_timeout = NO_PENDING_TIMEOUT;
	new_wsi->rxflow_change_to = LWS_RXFLOW_ALLOW;

	/* initialize the instance struct */

	new_wsi->state = LWSS_HTTP;
	new_wsi->mode = LWSCM_HTTP_SERVING;
	new_wsi->hdr_parsing_completed = 0;

#ifdef LWS_OPENSSL_SUPPORT
	new_wsi->use_ssl = LWS_SSL_ENABLED(vhost);
#endif

	/*
	 * these can only be set once the protocol is known
	 * we set an un-established connection's protocol pointer
	 * to the start of the supported list, so it can look
	 * for matching ones during the handshake
	 */
	new_wsi->protocol = vhost->protocols;
	new_wsi->user_space = NULL;
	new_wsi->ietf_spec_revision = 0;
	new_wsi->desc.sockfd = LWS_SOCK_INVALID;
	new_wsi->position_in_fds_table = -1;

	vhost->context->count_wsi_allocated++;

	/*
	 * outermost create notification for wsi
	 * no user_space because no protocol selection
	 */
	vhost->protocols[0].callback(new_wsi, LWS_CALLBACK_WSI_CREATE,
				       NULL, NULL, 0);

	return new_wsi;
}

LWS_VISIBLE int LWS_WARN_UNUSED_RESULT
lws_http_transaction_completed(struct lws *wsi)
{
	int n = NO_PENDING_TIMEOUT;

	lwsl_info("%s: wsi %p\n", __func__, wsi);

	lws_access_log(wsi);

	if (!wsi->hdr_parsing_completed) {
		lwsl_notice("%s: ignoring, ah parsing incomplete\n", __func__);
		return 0;
	}

	lwsl_debug("%s: wsi %p\n", __func__, wsi);
	/* if we can't go back to accept new headers, drop the connection */
	if (wsi->http2_substream)
		return 0;

	if (wsi->seen_zero_length_recv)
		return 1;

	if (wsi->u.http.connection_type != HTTP_CONNECTION_KEEP_ALIVE) {
		lwsl_info("%s: %p: close connection\n", __func__, wsi);
		return 1;
	}

	if (lws_bind_protocol(wsi, &wsi->vhost->protocols[0]))
		return 1;

	/* otherwise set ourselves up ready to go again */
	wsi->state = LWSS_HTTP;
	wsi->mode = LWSCM_HTTP_SERVING;
	wsi->u.http.tx_content_length = 0;
	wsi->u.http.tx_content_remain = 0;
	wsi->hdr_parsing_completed = 0;
#ifdef LWS_WITH_ACCESS_LOG
	wsi->access_log.sent = 0;
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
	if (wsi->u.hdr.ah) {
		lwsl_debug("%s: wsi->more_rx_waiting=%d\n", __func__,
				wsi->more_rx_waiting);

		if (!wsi->more_rx_waiting) {
			lws_header_table_force_to_detachable_state(wsi);
			lws_header_table_detach(wsi, 1);
#ifdef LWS_OPENSSL_SUPPORT
			/*
			 * additionally... if we are hogging an SSL instance
			 * with no pending pipelined headers (or ah now), and
			 * SSL is scarce, drop this connection without waiting
			 */

			if (wsi->vhost->use_ssl &&
			    wsi->context->simultaneous_ssl_restriction &&
			    wsi->context->simultaneous_ssl ==
				   wsi->context->simultaneous_ssl_restriction) {
				lwsl_info("%s: simultaneous_ssl_restriction\n",
					  __func__);
				return 1;
			}
#endif
		} else {
			lws_header_table_reset(wsi, 1);
			/*
			 * If we kept the ah, we should restrict the amount
			 * of time we are willing to keep it.  Otherwise it
			 * will be bound the whole time the connection remains
			 * open.
			 */
			lws_set_timeout(wsi, PENDING_TIMEOUT_HOLDING_AH,
					wsi->vhost->keepalive_timeout);
		}
	}

	/* If we're (re)starting on headers, need other implied init */
	wsi->u.hdr.ues = URIES_IDLE;

	lwsl_info("%s: %p: keep-alive await new transaction\n", __func__, wsi);

	return 0;
}

/* if not a socket, it's a raw, non-ssl file descriptor */

LWS_VISIBLE struct lws *
lws_adopt_descriptor_vhost(struct lws_vhost *vh, lws_adoption_type type,
			   lws_sock_file_fd_type fd, const char *vh_prot_name,
			   struct lws *parent)
{
	struct lws_context *context = vh->context;
	struct lws *new_wsi;
	struct lws_context_per_thread *pt;
	int n, ssl = 0;

#if defined(LWS_WITH_PEER_LIMITS)
	struct lws_peer *peer = NULL;

	if (type & LWS_ADOPT_SOCKET && !(type & LWS_ADOPT_WS_PARENTIO)) {
		peer = lws_get_or_create_peer(vh, fd.sockfd);

		if (!peer) {
			lwsl_err("OOM creating peer\n");
			return NULL;
		}
		if (context->ip_limit_wsi &&
		    peer->count_wsi >= context->ip_limit_wsi) {
			lwsl_notice("Peer reached wsi limit %d\n",
					context->ip_limit_wsi);
			lws_stats_atomic_bump(context, &context->pt[0],
					      LWSSTATS_C_PEER_LIMIT_WSI_DENIED, 1);
			return NULL;
		}
	}
#endif

	new_wsi = lws_create_new_server_wsi(vh);
	if (!new_wsi) {
		if (type & LWS_ADOPT_SOCKET && !(type & LWS_ADOPT_WS_PARENTIO))
			compatible_close(fd.sockfd);
		return NULL;
	}
#if defined(LWS_WITH_PEER_LIMITS)
	if (peer)
		lws_peer_add_wsi(context, peer, new_wsi);
#endif
	pt = &context->pt[(int)new_wsi->tsi];
	lws_stats_atomic_bump(context, pt, LWSSTATS_C_CONNECTIONS, 1);

	if (parent) {
		new_wsi->parent = parent;
		new_wsi->sibling_list = parent->child_list;
		parent->child_list = new_wsi;

		if (type & LWS_ADOPT_WS_PARENTIO)
			new_wsi->parent_carries_io = 1;
	}

	new_wsi->desc = fd;

	if (vh_prot_name) {
		new_wsi->protocol = lws_vhost_name_to_protocol(new_wsi->vhost,
							       vh_prot_name);
		if (!new_wsi->protocol) {
			lwsl_err("Protocol %s not enabled on vhost %s\n",
				 vh_prot_name, new_wsi->vhost->name);
			goto bail;
		}
               if (lws_ensure_user_space(new_wsi)) {
                       lwsl_notice("OOM trying to get user_space\n");
			goto bail;
               }
               if (type & LWS_ADOPT_WS_PARENTIO) {
			new_wsi->desc.sockfd = LWS_SOCK_INVALID;
			lwsl_debug("binding to %s\n", new_wsi->protocol->name);
			lws_bind_protocol(new_wsi, new_wsi->protocol);
			lws_union_transition(new_wsi, LWSCM_WS_SERVING);
			lws_server_init_wsi_for_ws(new_wsi);

			return new_wsi;
               }
	} else
		if (type & LWS_ADOPT_HTTP) /* he will transition later */
			new_wsi->protocol =
				&vh->protocols[vh->default_protocol_index];
		else { /* this is the only time he will transition */
			lws_bind_protocol(new_wsi,
				&vh->protocols[vh->raw_protocol_index]);
			lws_union_transition(new_wsi, LWSCM_RAW);
		}

	if (type & LWS_ADOPT_SOCKET) { /* socket desc */
		lwsl_debug("%s: new wsi %p, sockfd %d\n", __func__, new_wsi,
			   (int)(lws_intptr_t)fd.sockfd);

		if (type & LWS_ADOPT_HTTP)
			/* the transport is accepted...
			 * give him time to negotiate */
			lws_set_timeout(new_wsi,
					PENDING_TIMEOUT_ESTABLISH_WITH_SERVER,
					context->timeout_secs);

#if LWS_POSIX == 0
#if defined(LWS_WITH_ESP8266)
		esp8266_tcp_stream_accept(accept_fd, new_wsi);
#endif
#endif
	} else /* file desc */
		lwsl_debug("%s: new wsi %p, filefd %d\n", __func__, new_wsi,
			   (int)(lws_intptr_t)fd.filefd);

	/*
	 * A new connection was accepted. Give the user a chance to
	 * set properties of the newly created wsi. There's no protocol
	 * selected yet so we issue this to the vhosts's default protocol,
	 * itself by default protocols[0]
	 */
	n = LWS_CALLBACK_SERVER_NEW_CLIENT_INSTANTIATED;
	if (!(type & LWS_ADOPT_HTTP)) {
		if (!(type & LWS_ADOPT_SOCKET))
			n = LWS_CALLBACK_RAW_ADOPT_FILE;
		else
			n = LWS_CALLBACK_RAW_ADOPT;
	}

	if (!LWS_SSL_ENABLED(new_wsi->vhost) || !(type & LWS_ADOPT_ALLOW_SSL) ||
	    !(type & LWS_ADOPT_SOCKET)) {
		/* non-SSL */
		if (!(type & LWS_ADOPT_HTTP)) {
			if (!(type & LWS_ADOPT_SOCKET))
				new_wsi->mode = LWSCM_RAW_FILEDESC;
			else
				new_wsi->mode = LWSCM_RAW;
		}
	} else {
		/* SSL */
		if (!(type & LWS_ADOPT_HTTP))
			new_wsi->mode = LWSCM_SSL_INIT_RAW;
		else
			new_wsi->mode = LWSCM_SSL_INIT;

		ssl = 1;
	}

	lws_libev_accept(new_wsi, new_wsi->desc);
	lws_libuv_accept(new_wsi, new_wsi->desc);
	lws_libevent_accept(new_wsi, new_wsi->desc);

	if (!ssl) {
		if (insert_wsi_socket_into_fds(context, new_wsi)) {
			lwsl_err("%s: fail inserting socket\n", __func__);
			goto fail;
		}
	} else
		if (lws_server_socket_service_ssl(new_wsi, fd.sockfd)) {
			lwsl_info("%s: fail ssl negotiation\n", __func__);
			goto fail;
		}

	/*
	 *  by deferring callback to this point, after insertion to fds,
	 * lws_callback_on_writable() can work from the callback
	 */
	if ((new_wsi->protocol->callback)(
			new_wsi, n, new_wsi->user_space, NULL, 0))
		goto fail;

	if (type & LWS_ADOPT_HTTP) {
		if (!lws_header_table_attach(new_wsi, 0))
			lwsl_debug("Attached ah immediately\n");
		else
			lwsl_info("%s: waiting for ah\n", __func__);
	}

	return new_wsi;

fail:
	if (type & LWS_ADOPT_SOCKET)
		lws_close_free_wsi(new_wsi, LWS_CLOSE_STATUS_NOSTATUS);

	return NULL;

bail:
       lwsl_notice("%s: exiting on bail\n", __func__);
	if (parent)
		parent->child_list = new_wsi->sibling_list;
	if (new_wsi->user_space)
		lws_free(new_wsi->user_space);
	lws_free(new_wsi);
       compatible_close(fd.sockfd);

	return NULL;
}

LWS_VISIBLE struct lws *
lws_adopt_socket_vhost(struct lws_vhost *vh, lws_sockfd_type accept_fd)
{
	lws_sock_file_fd_type fd;

	fd.sockfd = accept_fd;
	return lws_adopt_descriptor_vhost(vh, LWS_ADOPT_SOCKET |
			LWS_ADOPT_HTTP | LWS_ADOPT_ALLOW_SSL, fd, NULL, NULL);
}

LWS_VISIBLE struct lws *
lws_adopt_socket(struct lws_context *context, lws_sockfd_type accept_fd)
{
	return lws_adopt_socket_vhost(context->vhost_list, accept_fd);
}

/* Common read-buffer adoption for lws_adopt_*_readbuf */
static struct lws*
adopt_socket_readbuf(struct lws *wsi, const char *readbuf, size_t len)
{
	struct lws_context_per_thread *pt;
	struct allocated_headers *ah;
	struct lws_pollfd *pfd;

	if (!wsi)
		return NULL;

	if (!readbuf || len == 0)
		return wsi;

	if (len > sizeof(ah->rx)) {
		lwsl_err("%s: rx in too big\n", __func__);
		goto bail;
	}

	/*
	 * we can't process the initial read data until we can attach an ah.
	 *
	 * if one is available, get it and place the data in his ah rxbuf...
	 * wsi with ah that have pending rxbuf get auto-POLLIN service.
	 *
	 * no autoservice because we didn't get a chance to attach the
	 * readbuf data to wsi or ah yet, and we will do it next if we get
	 * the ah.
	 */
	if (wsi->u.hdr.ah || !lws_header_table_attach(wsi, 0)) {
		ah = wsi->u.hdr.ah;
		memcpy(ah->rx, readbuf, len);
		ah->rxpos = 0;
		ah->rxlen = (int16_t)len;

		lwsl_notice("%s: calling service on readbuf ah\n", __func__);
		pt = &wsi->context->pt[(int)wsi->tsi];

		/* unlike a normal connect, we have the headers already
		 * (or the first part of them anyway).
		 * libuv won't come back and service us without a network
		 * event, so we need to do the header service right here.
		 */
		pfd = &pt->fds[wsi->position_in_fds_table];
		pfd->revents |= LWS_POLLIN;
		lwsl_err("%s: calling service\n", __func__);
		if (lws_service_fd_tsi(wsi->context, pfd, wsi->tsi))
			/* service closed us */
			return NULL;

		return wsi;
	}
	lwsl_err("%s: deferring handling ah\n", __func__);
	/*
	 * hum if no ah came, we are on the wait list and must defer
	 * dealing with this until the ah arrives.
	 *
	 * later successful lws_header_table_attach() will apply the
	 * below to the rx buffer (via lws_header_table_reset()).
	 */
	wsi->u.hdr.preamble_rx = lws_malloc(len, "preamble_rx");
	if (!wsi->u.hdr.preamble_rx) {
		lwsl_err("OOM\n");
		goto bail;
	}
	memcpy(wsi->u.hdr.preamble_rx, readbuf, len);
	wsi->u.hdr.preamble_rx_len = len;

	return wsi;

bail:
	lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS);

	return NULL;
}

LWS_VISIBLE struct lws *
lws_adopt_socket_readbuf(struct lws_context *context, lws_sockfd_type accept_fd,
			 const char *readbuf, size_t len)
{
        return adopt_socket_readbuf(lws_adopt_socket(context, accept_fd),
        			    readbuf, len);
}

LWS_VISIBLE struct lws *
lws_adopt_socket_vhost_readbuf(struct lws_vhost *vhost,
			       lws_sockfd_type accept_fd,
			       const char *readbuf, size_t len)
{
        return adopt_socket_readbuf(lws_adopt_socket_vhost(vhost, accept_fd),
        			    readbuf, len);
}

LWS_VISIBLE int
lws_server_socket_service(struct lws_context *context, struct lws *wsi,
			  struct lws_pollfd *pollfd)
{
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	lws_sockfd_type accept_fd = LWS_SOCK_INVALID;
	struct allocated_headers *ah;
	lws_sock_file_fd_type fd;
	int opts = LWS_ADOPT_SOCKET | LWS_ADOPT_ALLOW_SSL;
#if LWS_POSIX
	struct sockaddr_storage cli_addr;
	socklen_t clilen;
#endif
	int n, len;

	switch (wsi->mode) {

	case LWSCM_HTTP_SERVING:
	case LWSCM_HTTP_SERVING_ACCEPTED:
	case LWSCM_HTTP2_SERVING:
	case LWSCM_RAW:

		/* handle http headers coming in */

		/* pending truncated sends have uber priority */

		if (wsi->trunc_len) {
			if (!(pollfd->revents & LWS_POLLOUT))
				break;

			if (lws_issue_raw(wsi, wsi->trunc_alloc +
					       wsi->trunc_offset,
					  wsi->trunc_len) < 0)
				goto fail;
			/*
			 * we can't afford to allow input processing to send
			 * something new, so spin around he event loop until
			 * he doesn't have any partials
			 */
			break;
		}

		/* any incoming data ready? */

		if (!(pollfd->revents & pollfd->events & LWS_POLLIN))
			goto try_pollout;

		/*
		 * If we previously just did POLLIN when IN and OUT were
		 * signalled (because POLLIN processing may have used up
		 * the POLLOUT), don't let that happen twice in a row...
		 * next time we see the situation favour POLLOUT
		 */
#if !defined(LWS_WITH_ESP8266)
		if (wsi->favoured_pollin &&
		    (pollfd->revents & pollfd->events & LWS_POLLOUT)) {
			lwsl_notice("favouring pollout\n");
			wsi->favoured_pollin = 0;
			goto try_pollout;
		}
#endif

		/* these states imply we MUST have an ah attached */

		if (wsi->mode != LWSCM_RAW && (wsi->state == LWSS_HTTP ||
		    wsi->state == LWSS_HTTP_ISSUING_FILE ||
		    wsi->state == LWSS_HTTP_HEADERS)) {
			if (!wsi->u.hdr.ah) {
				/* no autoservice beacuse we will do it next */
				if (lws_header_table_attach(wsi, 0)) {
					lwsl_info("wsi %p: ah get fail\n", wsi);
					goto try_pollout;
				}
			}
			ah = wsi->u.hdr.ah;

			/* if nothing in ah rx buffer, get some fresh rx */
			if (ah->rxpos == ah->rxlen) {
				ah->rxlen = lws_ssl_capable_read(wsi, ah->rx,
						   sizeof(ah->rx));
				ah->rxpos = 0;
				switch (ah->rxlen) {
				case 0:
					lwsl_info("%s: read 0 len a\n", __func__);
					wsi->seen_zero_length_recv = 1;
					lws_change_pollfd(wsi, LWS_POLLIN, 0);
					goto try_pollout;
					/* fallthru */
				case LWS_SSL_CAPABLE_ERROR:
					goto fail;
				case LWS_SSL_CAPABLE_MORE_SERVICE:
					ah->rxlen = ah->rxpos = 0;
					goto try_pollout;
				}

				/*
				 *  make sure ah does not get detached if we
				 * have live data in the rx
				 */
				if (ah->rxlen)
					wsi->more_rx_waiting = 1;
			}

			if (!(ah->rxpos != ah->rxlen && ah->rxlen)) {
				lwsl_err("%s: assert: rxpos %d, rxlen %d\n",
					 __func__, ah->rxpos, ah->rxlen);

				assert(0);
			}
			
			/* just ignore incoming if waiting for close */
			if (wsi->state != LWSS_FLUSHING_STORED_SEND_BEFORE_CLOSE &&
			    wsi->state != LWSS_HTTP_ISSUING_FILE) {
				/*
				 * otherwise give it to whoever wants it
				 * according to the connection state
				 */

				n = lws_read(wsi, ah->rx + ah->rxpos,
					     ah->rxlen - ah->rxpos);
				if (n < 0) /* we closed wsi */
					return 1;

				if (!wsi->u.hdr.ah)
					break;
				if ( wsi->u.hdr.ah->rxlen)
					 wsi->u.hdr.ah->rxpos += n;

				lwsl_debug("%s: wsi %p: ah read rxpos %d, rxlen %d\n",
					   __func__, wsi, wsi->u.hdr.ah->rxpos,
					   wsi->u.hdr.ah->rxlen);

				if (lws_header_table_is_in_detachable_state(wsi) &&
				    (wsi->mode != LWSCM_HTTP_SERVING &&
				     wsi->mode != LWSCM_HTTP_SERVING_ACCEPTED &&
				     wsi->mode != LWSCM_HTTP2_SERVING))
					lws_header_table_detach(wsi, 1);

				break;
			}

			goto try_pollout;
		}

		len = lws_ssl_capable_read(wsi, pt->serv_buf,
					   context->pt_serv_buf_size);
		lwsl_debug("%s: wsi %p read %d\r\n", __func__, wsi, len);
		switch (len) {
		case 0:
			lwsl_info("%s: read 0 len b\n", __func__);

			/* fallthru */
		case LWS_SSL_CAPABLE_ERROR:
			goto fail;
		case LWS_SSL_CAPABLE_MORE_SERVICE:
			goto try_pollout;
		}
		
		if (len < 0) /* coverity */
			goto fail;

		if (wsi->mode == LWSCM_RAW) {
			n = user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_RAW_RX,
					wsi->user_space, pt->serv_buf, len);
			if (n < 0) {
				lwsl_info("LWS_CALLBACK_RAW_RX_fail\n");
				goto fail;
			}
			goto try_pollout;
		}

		/* just ignore incoming if waiting for close */
		if (wsi->state != LWSS_FLUSHING_STORED_SEND_BEFORE_CLOSE &&
		    wsi->state != LWSS_HTTP_ISSUING_FILE) {
			/*
			 * this may want to send
			 * (via HTTP callback for example)
			 */
			n = lws_read(wsi, pt->serv_buf, len);
			if (n < 0) /* we closed wsi */
				return 1;
			/*
			 *  he may have used up the
			 * writability above, if we will defer POLLOUT
			 * processing in favour of POLLIN, note it
			 */
			if (pollfd->revents & LWS_POLLOUT)
				wsi->favoured_pollin = 1;
			break;
		}
		/*
		 *  he may have used up the
		 * writability above, if we will defer POLLOUT
		 * processing in favour of POLLIN, note it
		 */
		if (pollfd->revents & LWS_POLLOUT)
			wsi->favoured_pollin = 1;

try_pollout:
		
		/* this handles POLLOUT for http serving fragments */

		if (!(pollfd->revents & LWS_POLLOUT))
			break;

		/* one shot */
		if (lws_change_pollfd(wsi, LWS_POLLOUT, 0)) {
			lwsl_notice("%s a\n", __func__);
			goto fail;
		}

		if (wsi->mode == LWSCM_RAW) {
			lws_stats_atomic_bump(wsi->context, pt,
						LWSSTATS_C_WRITEABLE_CB, 1);
#if defined(LWS_WITH_STATS)
			if (wsi->active_writable_req_us) {
				uint64_t ul = time_in_microseconds() -
						wsi->active_writable_req_us;

				lws_stats_atomic_bump(wsi->context, pt,
						LWSSTATS_MS_WRITABLE_DELAY, ul);
				lws_stats_atomic_max(wsi->context, pt,
					  LWSSTATS_MS_WORST_WRITABLE_DELAY, ul);
				wsi->active_writable_req_us = 0;
			}
#endif
			n = user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_RAW_WRITEABLE,
					wsi->user_space, NULL, 0);
			if (n < 0) {
				lwsl_info("writeable_fail\n");
				goto fail;
			}
			break;
		}

		if (!wsi->hdr_parsing_completed)
			break;

		if (wsi->state != LWSS_HTTP_ISSUING_FILE) {

			lws_stats_atomic_bump(wsi->context, pt,
						LWSSTATS_C_WRITEABLE_CB, 1);
#if defined(LWS_WITH_STATS)
			if (wsi->active_writable_req_us) {
				uint64_t ul = time_in_microseconds() -
						wsi->active_writable_req_us;

				lws_stats_atomic_bump(wsi->context, pt,
						LWSSTATS_MS_WRITABLE_DELAY, ul);
				lws_stats_atomic_max(wsi->context, pt,
					  LWSSTATS_MS_WORST_WRITABLE_DELAY, ul);
				wsi->active_writable_req_us = 0;
			}
#endif

			n = user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_HTTP_WRITEABLE,
					wsi->user_space, NULL, 0);
			if (n < 0) {
				lwsl_info("writeable_fail\n");
				goto fail;
			}
			break;
		}

		/* >0 == completion, <0 == error
		 *
		 * We'll get a LWS_CALLBACK_HTTP_FILE_COMPLETION callback when
		 * it's done.  That's the case even if we just completed the
		 * send, so wait for that.
		 */
		n = lws_serve_http_file_fragment(wsi);
		if (n < 0)
			goto fail;

		break;

	case LWSCM_SERVER_LISTENER:

#if LWS_POSIX
		/* pollin means a client has connected to us then */

		do {
			if (!(pollfd->revents & LWS_POLLIN) ||
			    !(pollfd->events & LWS_POLLIN))
				break;

#ifdef LWS_OPENSSL_SUPPORT
			/*
			 * can we really accept it, with regards to SSL limit?
			 * another vhost may also have had POLLIN on his listener this
			 * round and used it up already
			 */

			if (wsi->vhost->use_ssl &&
			    context->simultaneous_ssl_restriction &&
			    context->simultaneous_ssl ==
					  context->simultaneous_ssl_restriction)
				/* no... ignore it, he won't come again until we are
				 * below the simultaneous_ssl_restriction limit and
				 * POLLIN is enabled on him again
				 */
				break;
#endif
			/* listen socket got an unencrypted connection... */

			clilen = sizeof(cli_addr);
			lws_latency_pre(context, wsi);

			/*
			 * We cannot identify the peer who is in the listen
			 * socket connect queue before we accept it; even if
			 * we could, not accepting it due to PEER_LIMITS would
			 * block the connect queue for other legit peers.
			 */
			accept_fd  = accept(pollfd->fd, (struct sockaddr *)&cli_addr,
					    &clilen);
			lws_latency(context, wsi, "listener accept", accept_fd,
				    accept_fd >= 0);
			if (accept_fd < 0) {
				if (LWS_ERRNO == LWS_EAGAIN ||
				    LWS_ERRNO == LWS_EWOULDBLOCK) {
					break;
				}
				lwsl_err("ERROR on accept: %s\n", strerror(LWS_ERRNO));
				break;
			}

			lws_plat_set_socket_options(wsi->vhost, accept_fd);

#if defined(LWS_WITH_IPV6)
			lwsl_debug("accepted new conn port %u on fd=%d\n",
				((cli_addr.ss_family == AF_INET6) ?
				ntohs(((struct sockaddr_in6 *) &cli_addr)->sin6_port) :
				ntohs(((struct sockaddr_in *) &cli_addr)->sin_port)),
				accept_fd);
#else
			lwsl_debug("accepted new conn port %u on fd=%d\n",
				   ntohs(((struct sockaddr_in *) &cli_addr)->sin_port),
				   accept_fd);
#endif

#else
			/* not very beautiful... */
			accept_fd = (lws_sockfd_type)pollfd;
#endif
			/*
			 * look at who we connected to and give user code a chance
			 * to reject based on client IP.  There's no protocol selected
			 * yet so we issue this to protocols[0]
			 */
			if ((wsi->vhost->protocols[0].callback)(wsi,
					LWS_CALLBACK_FILTER_NETWORK_CONNECTION,
					NULL, (void *)(lws_intptr_t)accept_fd, 0)) {
				lwsl_debug("Callback denied network connection\n");
				compatible_close(accept_fd);
				break;
			}

			if (!(wsi->vhost->options & LWS_SERVER_OPTION_ONLY_RAW))
				opts |= LWS_ADOPT_HTTP;
			else
				opts = LWS_ADOPT_SOCKET;

			fd.sockfd = accept_fd;
			if (!lws_adopt_descriptor_vhost(wsi->vhost, opts, fd,
							NULL, NULL))
				/* already closed cleanly as necessary */
				return 1;

#if LWS_POSIX
		} while (pt->fds_count < context->fd_limit_per_thread - 1 &&
			 lws_poll_listen_fd(&pt->fds[wsi->position_in_fds_table]) > 0);
#endif
		return 0;

	default:
		break;
	}

	if (!lws_server_socket_service_ssl(wsi, accept_fd))
		return 0;

fail:
	lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS);

	return 1;
}

LWS_VISIBLE int
lws_serve_http_file(struct lws *wsi, const char *file, const char *content_type,
		    const char *other_headers, int other_headers_len)
{
	static const char * const intermediates[] = { "private", "public" };
	struct lws_context *context = lws_get_context(wsi);
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
#if defined(LWS_WITH_RANGES)
	struct lws_range_parsing *rp = &wsi->u.http.range;
#endif
	char cache_control[50], *cc = "no-store";
	unsigned char *response = pt->serv_buf + LWS_PRE;
	unsigned char *p = response;
	unsigned char *end = p + context->pt_serv_buf_size - LWS_PRE;
	lws_filepos_t computed_total_content_length;
	int ret = 0, cclen = 8, n = HTTP_STATUS_OK;
	lws_fop_flags_t fflags = LWS_O_RDONLY;
#if defined(LWS_WITH_RANGES)
	int ranges;
#endif
	const struct lws_plat_file_ops *fops;
	const char *vpath;

	/*
	 * We either call the platform fops .open with first arg platform fops,
	 * or we call fops_zip .open with first arg platform fops, and fops_zip
	 * open will decide whether to switch to fops_zip or stay with fops_def.
	 *
	 * If wsi->u.http.fop_fd is already set, the caller already opened it
	 */
	if (!wsi->u.http.fop_fd) {
		fops = lws_vfs_select_fops(wsi->context->fops, file, &vpath);
		fflags |= lws_vfs_prepare_flags(wsi);
		wsi->u.http.fop_fd = fops->LWS_FOP_OPEN(wsi->context->fops,
							file, vpath, &fflags);
		if (!wsi->u.http.fop_fd) {
			lwsl_err("Unable to open '%s'\n", file);

			return -1;
		}
	}
	wsi->u.http.filelen = lws_vfs_get_length(wsi->u.http.fop_fd);
	computed_total_content_length = wsi->u.http.filelen;

#if defined(LWS_WITH_RANGES)
	ranges = lws_ranges_init(wsi, rp, wsi->u.http.filelen);

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
		lws_return_http_status(wsi, HTTP_STATUS_REQ_RANGE_NOT_SATISFIABLE,
				       NULL);
		if (lws_http_transaction_completed(wsi))
			return -1; /* <0 means just hang up */

		lws_vfs_file_close(&wsi->u.http.fop_fd);

		return 0; /* == 0 means we dealt with the transaction complete */
	}
	if (ranges)
		n = HTTP_STATUS_PARTIAL_CONTENT;
#endif

	if (lws_add_http_header_status(wsi, n, &p, end))
		return -1;

	if ((wsi->u.http.fop_fd->flags & (LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP |
		       LWS_FOP_FLAG_COMPR_IS_GZIP)) ==
	    (LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP | LWS_FOP_FLAG_COMPR_IS_GZIP)) {
		if (lws_add_http_header_by_token(wsi,
			WSI_TOKEN_HTTP_CONTENT_ENCODING,
			(unsigned char *)"gzip", 4, &p, end))
			return -1;
		lwsl_info("file is being provided in gzip\n");
	}

	if (
#if defined(LWS_WITH_RANGES)
	    ranges < 2 &&
#endif
	    content_type && content_type[0])
		if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_CONTENT_TYPE,
						 (unsigned char *)content_type,
						 strlen(content_type), &p, end))
			return -1;

#if defined(LWS_WITH_RANGES)
	if (ranges >= 2) { /* multipart byteranges */
		strncpy(wsi->u.http.multipart_content_type, content_type,
			sizeof(wsi->u.http.multipart_content_type) - 1);
		wsi->u.http.multipart_content_type[
		         sizeof(wsi->u.http.multipart_content_type) - 1] = '\0';
		if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_CONTENT_TYPE,
			 (unsigned char *)"multipart/byteranges; boundary=_lws",
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

		computed_total_content_length = (lws_filepos_t)rp->agg +
						6 /* final _lws\r\n */;

		lws_ranges_reset(rp);
		while (lws_ranges_next(rp)) {
			n = lws_snprintf(cache_control, sizeof(cache_control),
					"bytes %llu-%llu/%llu",
					rp->start, rp->end, rp->extent);

			computed_total_content_length +=
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
		computed_total_content_length = (lws_filepos_t)rp->agg;
		n = lws_snprintf(cache_control, sizeof(cache_control),
				 "bytes %llu-%llu/%llu",
				 rp->start, rp->end, rp->extent);

		if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_CONTENT_RANGE,
						 (unsigned char *)cache_control,
						 n, &p, end))
			return -1;
	}

	wsi->u.http.range.inside = 0;

	if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_ACCEPT_RANGES,
					 (unsigned char *)"bytes", 5, &p, end))
		return -1;
#endif

	if (!wsi->sending_chunked) {
		if (lws_add_http_header_content_length(wsi,
						       computed_total_content_length,
						       &p, end))
			return -1;
	} else {
		if (lws_add_http_header_by_token(wsi,
						 WSI_TOKEN_HTTP_TRANSFER_ENCODING,
						 (unsigned char *)"chunked",
						 7, &p, end))
			return -1;
	}

	if (wsi->cache_secs && wsi->cache_reuse) {
		if (wsi->cache_revalidate) {
			cc = cache_control;
			cclen = sprintf(cache_control, "%s max-age: %u",
				    intermediates[wsi->cache_intermediaries],
				    wsi->cache_secs);
		} else {
			cc = "no-cache";
			cclen = 8;
		}
	}

	if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_CACHE_CONTROL,
			(unsigned char *)cc, cclen, &p, end))
		return -1;

	if (wsi->u.http.connection_type == HTTP_CONNECTION_KEEP_ALIVE)
		if (lws_add_http_header_by_token(wsi, WSI_TOKEN_CONNECTION,
				(unsigned char *)"keep-alive", 10, &p, end))
			return -1;

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

	wsi->u.http.filepos = 0;
	wsi->state = LWSS_HTTP_ISSUING_FILE;

	lws_callback_on_writable(wsi);

	return 0;
}

int
lws_interpret_incoming_packet(struct lws *wsi, unsigned char **buf, size_t len)
{
	int m;

	lwsl_parser("%s: received %d byte packet\n", __func__, (int)len);
#if 0
	lwsl_hexdump(*buf, len);
#endif

	/* let the rx protocol state machine have as much as it needs */

	while (len) {
		/*
		 * we were accepting input but now we stopped doing so
		 */
		if (wsi->rxflow_bitmap) {
			lws_rxflow_cache(wsi, *buf, 0, len);
			lwsl_parser("%s: cached %ld\n", __func__, (long)len);
			return 1;
		}

		if (wsi->u.ws.rx_draining_ext) {
			m = lws_rx_sm(wsi, 0);
			if (m < 0)
				return -1;
			continue;
		}

		/* account for what we're using in rxflow buffer */
		if (wsi->rxflow_buffer) {
			wsi->rxflow_pos++;
			if (wsi->rxflow_pos > wsi->rxflow_len) {
				lwsl_err("bumped rxflow buffer too far (%d / %d)", wsi->rxflow_pos, wsi->rxflow_len);
				assert(0);
			}
		}

		/* consume payload bytes efficiently */
		if (wsi->lws_rx_parse_state ==
		    LWS_RXPS_PAYLOAD_UNTIL_LENGTH_EXHAUSTED) {
			m = lws_payload_until_length_exhausted(wsi, buf, &len);
			if (wsi->rxflow_buffer)
				wsi->rxflow_pos += m;
		}

		if (wsi->rxflow_buffer && wsi->rxflow_pos == wsi->rxflow_len) {
			lwsl_debug("%s: %p flow buf: drained\n", __func__, wsi);
			lws_free_set_NULL(wsi->rxflow_buffer);
			/* having drained the rxflow buffer, can rearm POLLIN */
#ifdef LWS_NO_SERVER
			m =
#endif
			_lws_rx_flow_control(wsi);
			/* m ignored, needed for NO_SERVER case */
		}

		/* process the byte */
		m = lws_rx_sm(wsi, *(*buf)++);
		if (m < 0)
			return -1;
		len--;
	}

	lwsl_parser("%s: exit with %d unused\n", __func__, (int)len);

	return 0;
}

LWS_VISIBLE void
lws_server_get_canonical_hostname(struct lws_context *context,
				  struct lws_context_creation_info *info)
{
	if (lws_check_opt(info->options,
			LWS_SERVER_OPTION_SKIP_SERVER_CANONICAL_NAME))
		return;
#if LWS_POSIX && !defined(LWS_WITH_ESP32)
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
			if (hits == 1 && s->pos == strlen(s->vars[hit])) {
				pc = s->replace(s->data, hit);
				if (!pc)
					pc = "NULL";
				n = strlen(pc);
				s->swallow[s->pos] = '\0';
				if (n != s->pos) {
					memmove(s->start + n,
						s->start + s->pos,
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

	return 0;
}
