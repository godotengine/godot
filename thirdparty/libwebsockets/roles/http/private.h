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
 *  This is included from core/private.h if either H1 or H2 roles are
 *  enabled
 */

#if defined(LWS_WITH_HTTP_PROXY)
  #include <hubbub/hubbub.h>
  #include <hubbub/parser.h>
 #endif

#define lwsi_role_http(wsi) (lwsi_role_h1(wsi) || lwsi_role_h2(wsi))

enum http_version {
	HTTP_VERSION_1_0,
	HTTP_VERSION_1_1,
	HTTP_VERSION_2
};

enum http_connection_type {
	HTTP_CONNECTION_CLOSE,
	HTTP_CONNECTION_KEEP_ALIVE
};

/*
 * This is totally opaque to code using the library.  It's exported as a
 * forward-reference pointer-only declaration; the user can use the pointer with
 * other APIs to get information out of it.
 */

#if defined(LWS_WITH_ESP32)
typedef uint16_t ah_data_idx_t;
#else
typedef uint32_t ah_data_idx_t;
#endif

struct lws_fragments {
	ah_data_idx_t	offset;
	uint16_t	len;
	uint8_t		nfrag; /* which ah->frag[] continues this content, or 0 */
	uint8_t		flags; /* only http2 cares */
};

#if defined(LWS_WITH_RANGES)
enum range_states {
	LWSRS_NO_ACTIVE_RANGE,
	LWSRS_BYTES_EQ,
	LWSRS_FIRST,
	LWSRS_STARTING,
	LWSRS_ENDING,
	LWSRS_COMPLETED,
	LWSRS_SYNTAX,
};

struct lws_range_parsing {
	unsigned long long start, end, extent, agg, budget;
	const char buf[128];
	int pos;
	enum range_states state;
	char start_valid, end_valid, ctr, count_ranges, did_try, inside, send_ctr;
};

int
lws_ranges_init(struct lws *wsi, struct lws_range_parsing *rp,
		unsigned long long extent);
int
lws_ranges_next(struct lws_range_parsing *rp);
void
lws_ranges_reset(struct lws_range_parsing *rp);
#endif

/*
 * these are assigned from a pool held in the context.
 * Both client and server mode uses them for http header analysis
 */

struct allocated_headers {
	struct allocated_headers *next; /* linked list */
	struct lws *wsi; /* owner */
	char *data; /* prepared by context init to point to dedicated storage */
	ah_data_idx_t data_length;
	/*
	 * the randomly ordered fragments, indexed by frag_index and
	 * lws_fragments->nfrag for continuation.
	 */
	struct lws_fragments frags[WSI_TOKEN_COUNT];
	time_t assigned;
	/*
	 * for each recognized token, frag_index says which frag[] his data
	 * starts in (0 means the token did not appear)
	 * the actual header data gets dumped as it comes in, into data[]
	 */
	uint8_t frag_index[WSI_TOKEN_COUNT];

#ifndef LWS_NO_CLIENT
	char initial_handshake_hash_base64[30];
#endif

	uint32_t pos;
	uint32_t http_response;
	uint32_t current_token_limit;
	int hdr_token_idx;

	int16_t lextable_pos;

	uint8_t in_use;
	uint8_t nfrag;
	char /*enum uri_path_states */ ups;
	char /*enum uri_esc_states */ ues;

	char esc_stash;
	char post_literal_equal;
	uint8_t /* enum lws_token_indexes */ parser_state;
};



#if defined(LWS_WITH_HTTP_PROXY)
struct lws_rewrite {
	hubbub_parser *parser;
	hubbub_parser_optparams params;
	const char *from, *to;
	int from_len, to_len;
	unsigned char *p, *end;
	struct lws *wsi;
};
static LWS_INLINE int hstrcmp(hubbub_string *s, const char *p, int len)
{
	if ((int)s->len != len)
		return 1;

	return strncmp((const char *)s->ptr, p, len);
}
typedef hubbub_error (*hubbub_callback_t)(const hubbub_token *token, void *pw);
LWS_EXTERN struct lws_rewrite *
lws_rewrite_create(struct lws *wsi, hubbub_callback_t cb, const char *from, const char *to);
LWS_EXTERN void
lws_rewrite_destroy(struct lws_rewrite *r);
LWS_EXTERN int
lws_rewrite_parse(struct lws_rewrite *r, const unsigned char *in, int in_len);
#endif

struct lws_pt_role_http {
	struct allocated_headers *ah_list;
	struct lws *ah_wait_list;
#ifdef LWS_WITH_CGI
	struct lws_cgi *cgi_list;
#endif
	int ah_wait_list_length;
	uint32_t ah_pool_length;

	int ah_count_in_use;
};

struct lws_peer_role_http {
	uint32_t count_ah;
	uint32_t total_ah;
};

struct lws_vhost_role_http {
	char http_proxy_address[128];
	const struct lws_http_mount *mount_list;
	const char *error_document_404;
	unsigned int http_proxy_port;
};

#ifdef LWS_WITH_ACCESS_LOG
struct lws_access_log {
	char *header_log;
	char *user_agent;
	char *referrer;
	unsigned long sent;
	int response;
};
#endif

struct _lws_http_mode_related {
	struct lws *new_wsi_list;

#if defined(LWS_WITH_HTTP_PROXY)
	struct lws_rewrite *rw;
#endif
	struct allocated_headers *ah;
	struct lws *ah_wait_list;

	lws_filepos_t filepos;
	lws_filepos_t filelen;
	lws_fop_fd_t fop_fd;

#if defined(LWS_WITH_RANGES)
	struct lws_range_parsing range;
	char multipart_content_type[64];
#endif

#ifdef LWS_WITH_ACCESS_LOG
	struct lws_access_log access_log;
#endif
#ifdef LWS_WITH_CGI
	struct lws_cgi *cgi; /* wsi being cgi master have one of these */
#endif

	enum http_version request_version;
	enum http_connection_type connection_type;
	lws_filepos_t tx_content_length;
	lws_filepos_t tx_content_remain;
	lws_filepos_t rx_content_length;
	lws_filepos_t rx_content_remain;

#if defined(LWS_WITH_HTTP_PROXY)
	unsigned int perform_rewrite:1;
#endif
};


#ifndef LWS_NO_CLIENT
enum lws_chunk_parser {
	ELCP_HEX,
	ELCP_CR,
	ELCP_CONTENT,
	ELCP_POST_CR,
	ELCP_POST_LF,
};
#endif

enum lws_parse_urldecode_results {
	LPUR_CONTINUE,
	LPUR_SWALLOW,
	LPUR_FORBID,
	LPUR_EXCESSIVE,
};

int
lws_read_h1(struct lws *wsi, unsigned char *buf, lws_filepos_t len);

void
_lws_header_table_reset(struct allocated_headers *ah);

LWS_EXTERN int
_lws_destroy_ah(struct lws_context_per_thread *pt, struct allocated_headers *ah);
