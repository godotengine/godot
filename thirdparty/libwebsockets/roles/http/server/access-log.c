/*
 * libwebsockets - server access log handling
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

#include "core/private.h"

/*
 * Produce Apache-compatible log string for wsi, like this:
 *
 * 2.31.234.19 - - [27/Mar/2016:03:22:44 +0800]
 * "GET /aep-screen.png HTTP/1.1"
 * 200 152987 "https://libwebsockets.org/index.html"
 * "Mozilla/5.0 (Macint... Chrome/49.0.2623.87 Safari/537.36"
 *
 */

extern const char * const method_names[];

static const char * const hver[] = {
	"HTTP/1.0", "HTTP/1.1", "HTTP/2"
};

void
lws_prepare_access_log_info(struct lws *wsi, char *uri_ptr, int meth)
{
#ifdef LWS_WITH_IPV6
	char ads[INET6_ADDRSTRLEN];
#else
	char ads[INET_ADDRSTRLEN];
#endif
	char da[64];
	const char *pa, *me;
	struct tm *tmp;
	time_t t = time(NULL);
	int l = 256, m;

	if (!wsi->vhost)
		return;

	/* only worry about preparing it if we store it */
	if (wsi->vhost->log_fd == (int)LWS_INVALID_FILE)
		return;

	if (wsi->access_log_pending)
		lws_access_log(wsi);

	wsi->http.access_log.header_log = lws_malloc(l, "access log");
	if (wsi->http.access_log.header_log) {

		tmp = localtime(&t);
		if (tmp)
			strftime(da, sizeof(da), "%d/%b/%Y:%H:%M:%S %z", tmp);
		else
			strcpy(da, "01/Jan/1970:00:00:00 +0000");

		pa = lws_get_peer_simple(wsi, ads, sizeof(ads));
		if (!pa)
			pa = "(unknown)";

		if (wsi->http2_substream)
			me = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_COLON_METHOD);
		else
			me = method_names[meth];
		if (!me)
			me = "(null)";

		lws_snprintf(wsi->http.access_log.header_log, l,
			 "%s - - [%s] \"%s %s %s\"",
			 pa, da, me, uri_ptr,
			 hver[wsi->http.request_version]);

		l = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_USER_AGENT);
		if (l) {
			wsi->http.access_log.user_agent = lws_malloc(l + 2, "access log");
			if (!wsi->http.access_log.user_agent) {
				lwsl_err("OOM getting user agent\n");
				lws_free_set_NULL(wsi->http.access_log.header_log);
				return;
			}

			lws_hdr_copy(wsi, wsi->http.access_log.user_agent,
					l + 1, WSI_TOKEN_HTTP_USER_AGENT);

			for (m = 0; m < l; m++)
				if (wsi->http.access_log.user_agent[m] == '\"')
					wsi->http.access_log.user_agent[m] = '\'';
		}
		l = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_REFERER);
		if (l) {
			wsi->http.access_log.referrer = lws_malloc(l + 2, "referrer");
			if (!wsi->http.access_log.referrer) {
				lwsl_err("OOM getting user agent\n");
				lws_free_set_NULL(wsi->http.access_log.user_agent);
				lws_free_set_NULL(wsi->http.access_log.header_log);
				return;
			}
			lws_hdr_copy(wsi, wsi->http.access_log.referrer,
					l + 1, WSI_TOKEN_HTTP_REFERER);

			for (m = 0; m < l; m++)
				if (wsi->http.access_log.referrer[m] == '\"')
					wsi->http.access_log.referrer[m] = '\'';
		}
		wsi->access_log_pending = 1;
	}
}


int
lws_access_log(struct lws *wsi)
{
	char *p = wsi->http.access_log.user_agent, ass[512],
	     *p1 = wsi->http.access_log.referrer;
	int l;

	if (!wsi->vhost)
		return 0;

	if (wsi->vhost->log_fd == (int)LWS_INVALID_FILE)
		return 0;

	if (!wsi->access_log_pending)
		return 0;

	if (!wsi->http.access_log.header_log)
		return 0;

	if (!p)
		p = "";

	if (!p1)
		p1 = "";

	/*
	 * We do this in two parts to restrict an oversize referrer such that
	 * we will always have space left to append an empty useragent, while
	 * maintaining the structure of the log text
	 */
	l = lws_snprintf(ass, sizeof(ass) - 7, "%s %d %lu \"%s",
		     wsi->http.access_log.header_log,
		     wsi->http.access_log.response, wsi->http.access_log.sent, p1);
	if (strlen(p) > sizeof(ass) - 6 - l)
		p[sizeof(ass) - 6 - l] = '\0';
	l += lws_snprintf(ass + l, sizeof(ass) - 1 - l, "\" \"%s\"\n", p);

	if (write(wsi->vhost->log_fd, ass, l) != l)
		lwsl_err("Failed to write log\n");

	if (wsi->http.access_log.header_log) {
		lws_free(wsi->http.access_log.header_log);
		wsi->http.access_log.header_log = NULL;
	}
	if (wsi->http.access_log.user_agent) {
		lws_free(wsi->http.access_log.user_agent);
		wsi->http.access_log.user_agent = NULL;
	}
	if (wsi->http.access_log.referrer) {
		lws_free(wsi->http.access_log.referrer);
		wsi->http.access_log.referrer = NULL;
	}
	wsi->access_log_pending = 0;

	return 0;
}

