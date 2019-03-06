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
 *
 * included from libwebsockets.h
 */

/* minimal space for typical headers and CSP stuff */

#define LWS_RECOMMENDED_MIN_HEADER_SPACE 2048

/*! \defgroup http HTTP

    Modules related to handling HTTP
*/
//@{

/*! \defgroup httpft HTTP File transfer
 * \ingroup http

    APIs for sending local files in response to HTTP requests
*/
//@{

/**
 * lws_get_mimetype() - Determine mimetype to use from filename
 *
 * \param file:		filename
 * \param m:		NULL, or mount context
 *
 * This uses a canned list of known filetypes first, if no match and m is
 * non-NULL, then tries a list of per-mount file suffix to mimtype mappings.
 *
 * Returns either NULL or a pointer to the mimetype matching the file.
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_get_mimetype(const char *file, const struct lws_http_mount *m);

/**
 * lws_serve_http_file() - Send a file back to the client using http
 * \param wsi:		Websocket instance (available from user callback)
 * \param file:		The file to issue over http
 * \param content_type:	The http content type, eg, text/html
 * \param other_headers:	NULL or pointer to header string
 * \param other_headers_len:	length of the other headers if non-NULL
 *
 *	This function is intended to be called from the callback in response
 *	to http requests from the client.  It allows the callback to issue
 *	local files down the http link in a single step.
 *
 *	Returning <0 indicates error and the wsi should be closed.  Returning
 *	>0 indicates the file was completely sent and
 *	lws_http_transaction_completed() called on the wsi (and close if != 0)
 *	==0 indicates the file transfer is started and needs more service later,
 *	the wsi should be left alone.
 */
LWS_VISIBLE LWS_EXTERN int
lws_serve_http_file(struct lws *wsi, const char *file, const char *content_type,
		    const char *other_headers, int other_headers_len);

LWS_VISIBLE LWS_EXTERN int
lws_serve_http_file_fragment(struct lws *wsi);
//@}


enum http_status {
	HTTP_STATUS_CONTINUE					= 100,

	HTTP_STATUS_OK						= 200,
	HTTP_STATUS_NO_CONTENT					= 204,
	HTTP_STATUS_PARTIAL_CONTENT				= 206,

	HTTP_STATUS_MOVED_PERMANENTLY				= 301,
	HTTP_STATUS_FOUND					= 302,
	HTTP_STATUS_SEE_OTHER					= 303,
	HTTP_STATUS_NOT_MODIFIED				= 304,

	HTTP_STATUS_BAD_REQUEST					= 400,
	HTTP_STATUS_UNAUTHORIZED,
	HTTP_STATUS_PAYMENT_REQUIRED,
	HTTP_STATUS_FORBIDDEN,
	HTTP_STATUS_NOT_FOUND,
	HTTP_STATUS_METHOD_NOT_ALLOWED,
	HTTP_STATUS_NOT_ACCEPTABLE,
	HTTP_STATUS_PROXY_AUTH_REQUIRED,
	HTTP_STATUS_REQUEST_TIMEOUT,
	HTTP_STATUS_CONFLICT,
	HTTP_STATUS_GONE,
	HTTP_STATUS_LENGTH_REQUIRED,
	HTTP_STATUS_PRECONDITION_FAILED,
	HTTP_STATUS_REQ_ENTITY_TOO_LARGE,
	HTTP_STATUS_REQ_URI_TOO_LONG,
	HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE,
	HTTP_STATUS_REQ_RANGE_NOT_SATISFIABLE,
	HTTP_STATUS_EXPECTATION_FAILED,

	HTTP_STATUS_INTERNAL_SERVER_ERROR			= 500,
	HTTP_STATUS_NOT_IMPLEMENTED,
	HTTP_STATUS_BAD_GATEWAY,
	HTTP_STATUS_SERVICE_UNAVAILABLE,
	HTTP_STATUS_GATEWAY_TIMEOUT,
	HTTP_STATUS_HTTP_VERSION_NOT_SUPPORTED,
};
/*! \defgroup html-chunked-substitution HTML Chunked Substitution
 * \ingroup http
 *
 * ##HTML chunked Substitution
 *
 * APIs for receiving chunks of text, replacing a set of variable names via
 * a callback, and then prepending and appending HTML chunked encoding
 * headers.
 */
//@{

struct lws_process_html_args {
	char *p; /**< pointer to the buffer containing the data */
	int len; /**< length of the original data at p */
	int max_len; /**< maximum length we can grow the data to */
	int final; /**< set if this is the last chunk of the file */
	int chunked; /**< 0 == unchunked, 1 == produce chunk headers
			(incompatible with HTTP/2) */
};

typedef const char *(*lws_process_html_state_cb)(void *data, int index);

struct lws_process_html_state {
	char *start; /**< pointer to start of match */
	char swallow[16]; /**< matched character buffer */
	int pos; /**< position in match */
	void *data; /**< opaque pointer */
	const char * const *vars; /**< list of variable names */
	int count_vars; /**< count of variable names */

	lws_process_html_state_cb replace;
		/**< called on match to perform substitution */
};

/*! lws_chunked_html_process() - generic chunked substitution
 * \param args: buffer to process using chunked encoding
 * \param s: current processing state
 */
LWS_VISIBLE LWS_EXTERN int
lws_chunked_html_process(struct lws_process_html_args *args,
			 struct lws_process_html_state *s);
//@}

/** \defgroup HTTP-headers-read HTTP headers: read
 * \ingroup http
 *
 * ##HTTP header releated functions
 *
 *  In lws the client http headers are temporarily stored in a pool, only for the
 *  duration of the http part of the handshake.  It's because in most cases,
 *  the header content is ignored for the whole rest of the connection lifetime
 *  and would then just be taking up space needlessly.
 *
 *  During LWS_CALLBACK_HTTP when the URI path is delivered is the last time
 *  the http headers are still allocated, you can use these apis then to
 *  look at and copy out interesting header content (cookies, etc)
 *
 *  Notice that the header total length reported does not include a terminating
 *  '\0', however you must allocate for it when using the _copy apis.  So the
 *  length reported for a header containing "123" is 3, but you must provide
 *  a buffer of length 4 so that "123\0" may be copied into it, or the copy
 *  will fail with a nonzero return code.
 *
 *  In the special case of URL arguments, like ?x=1&y=2, the arguments are
 *  stored in a token named for the method, eg,  WSI_TOKEN_GET_URI if it
 *  was a GET or WSI_TOKEN_POST_URI if POST.  You can check the total
 *  length to confirm the method.
 *
 *  For URL arguments, each argument is stored urldecoded in a "fragment", so
 *  you can use the fragment-aware api lws_hdr_copy_fragment() to access each
 *  argument in turn: the fragments contain urldecoded strings like x=1 or y=2.
 *
 *  As a convenience, lws has an api that will find the fragment with a
 *  given name= part, lws_get_urlarg_by_name().
 */
///@{

/** struct lws_tokens
 * you need these to look at headers that have been parsed if using the
 * LWS_CALLBACK_FILTER_CONNECTION callback.  If a header from the enum
 * list below is absent, .token = NULL and len = 0.  Otherwise .token
 * points to .len chars containing that header content.
 */
struct lws_tokens {
	char *token; /**< pointer to start of the token */
	int len; /**< length of the token's value */
};

/* enum lws_token_indexes
 * these have to be kept in sync with lextable.h / minilex.c
 *
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
enum lws_token_indexes {
	WSI_TOKEN_GET_URI					=  0,
	WSI_TOKEN_POST_URI					=  1,
	WSI_TOKEN_OPTIONS_URI					=  2,
	WSI_TOKEN_HOST						=  3,
	WSI_TOKEN_CONNECTION					=  4,
	WSI_TOKEN_UPGRADE					=  5,
	WSI_TOKEN_ORIGIN					=  6,
	WSI_TOKEN_DRAFT						=  7,
	WSI_TOKEN_CHALLENGE					=  8,
	WSI_TOKEN_EXTENSIONS					=  9,
	WSI_TOKEN_KEY1						= 10,
	WSI_TOKEN_KEY2						= 11,
	WSI_TOKEN_PROTOCOL					= 12,
	WSI_TOKEN_ACCEPT					= 13,
	WSI_TOKEN_NONCE						= 14,
	WSI_TOKEN_HTTP						= 15,
	WSI_TOKEN_HTTP2_SETTINGS				= 16,
	WSI_TOKEN_HTTP_ACCEPT					= 17,
	WSI_TOKEN_HTTP_AC_REQUEST_HEADERS			= 18,
	WSI_TOKEN_HTTP_IF_MODIFIED_SINCE			= 19,
	WSI_TOKEN_HTTP_IF_NONE_MATCH				= 20,
	WSI_TOKEN_HTTP_ACCEPT_ENCODING				= 21,
	WSI_TOKEN_HTTP_ACCEPT_LANGUAGE				= 22,
	WSI_TOKEN_HTTP_PRAGMA					= 23,
	WSI_TOKEN_HTTP_CACHE_CONTROL				= 24,
	WSI_TOKEN_HTTP_AUTHORIZATION				= 25,
	WSI_TOKEN_HTTP_COOKIE					= 26,
	WSI_TOKEN_HTTP_CONTENT_LENGTH				= 27,
	WSI_TOKEN_HTTP_CONTENT_TYPE				= 28,
	WSI_TOKEN_HTTP_DATE					= 29,
	WSI_TOKEN_HTTP_RANGE					= 30,
	WSI_TOKEN_HTTP_REFERER					= 31,
	WSI_TOKEN_KEY						= 32,
	WSI_TOKEN_VERSION					= 33,
	WSI_TOKEN_SWORIGIN					= 34,

	WSI_TOKEN_HTTP_COLON_AUTHORITY				= 35,
	WSI_TOKEN_HTTP_COLON_METHOD				= 36,
	WSI_TOKEN_HTTP_COLON_PATH				= 37,
	WSI_TOKEN_HTTP_COLON_SCHEME				= 38,
	WSI_TOKEN_HTTP_COLON_STATUS				= 39,

	WSI_TOKEN_HTTP_ACCEPT_CHARSET				= 40,
	WSI_TOKEN_HTTP_ACCEPT_RANGES				= 41,
	WSI_TOKEN_HTTP_ACCESS_CONTROL_ALLOW_ORIGIN		= 42,
	WSI_TOKEN_HTTP_AGE					= 43,
	WSI_TOKEN_HTTP_ALLOW					= 44,
	WSI_TOKEN_HTTP_CONTENT_DISPOSITION			= 45,
	WSI_TOKEN_HTTP_CONTENT_ENCODING				= 46,
	WSI_TOKEN_HTTP_CONTENT_LANGUAGE				= 47,
	WSI_TOKEN_HTTP_CONTENT_LOCATION				= 48,
	WSI_TOKEN_HTTP_CONTENT_RANGE				= 49,
	WSI_TOKEN_HTTP_ETAG					= 50,
	WSI_TOKEN_HTTP_EXPECT					= 51,
	WSI_TOKEN_HTTP_EXPIRES					= 52,
	WSI_TOKEN_HTTP_FROM					= 53,
	WSI_TOKEN_HTTP_IF_MATCH					= 54,
	WSI_TOKEN_HTTP_IF_RANGE					= 55,
	WSI_TOKEN_HTTP_IF_UNMODIFIED_SINCE			= 56,
	WSI_TOKEN_HTTP_LAST_MODIFIED				= 57,
	WSI_TOKEN_HTTP_LINK					= 58,
	WSI_TOKEN_HTTP_LOCATION					= 59,
	WSI_TOKEN_HTTP_MAX_FORWARDS				= 60,
	WSI_TOKEN_HTTP_PROXY_AUTHENTICATE			= 61,
	WSI_TOKEN_HTTP_PROXY_AUTHORIZATION			= 62,
	WSI_TOKEN_HTTP_REFRESH					= 63,
	WSI_TOKEN_HTTP_RETRY_AFTER				= 64,
	WSI_TOKEN_HTTP_SERVER					= 65,
	WSI_TOKEN_HTTP_SET_COOKIE				= 66,
	WSI_TOKEN_HTTP_STRICT_TRANSPORT_SECURITY		= 67,
	WSI_TOKEN_HTTP_TRANSFER_ENCODING			= 68,
	WSI_TOKEN_HTTP_USER_AGENT				= 69,
	WSI_TOKEN_HTTP_VARY					= 70,
	WSI_TOKEN_HTTP_VIA					= 71,
	WSI_TOKEN_HTTP_WWW_AUTHENTICATE				= 72,

	WSI_TOKEN_PATCH_URI					= 73,
	WSI_TOKEN_PUT_URI					= 74,
	WSI_TOKEN_DELETE_URI					= 75,

	WSI_TOKEN_HTTP_URI_ARGS					= 76,
	WSI_TOKEN_PROXY						= 77,
	WSI_TOKEN_HTTP_X_REAL_IP				= 78,
	WSI_TOKEN_HTTP1_0					= 79,
	WSI_TOKEN_X_FORWARDED_FOR				= 80,
	WSI_TOKEN_CONNECT					= 81,
	WSI_TOKEN_HEAD_URI					= 82,
	WSI_TOKEN_TE						= 83,
	WSI_TOKEN_REPLAY_NONCE					= 84,
	WSI_TOKEN_COLON_PROTOCOL				= 85,
	WSI_TOKEN_X_AUTH_TOKEN					= 86,

	/****** add new things just above ---^ ******/

	/* use token storage to stash these internally, not for
	 * user use */

	_WSI_TOKEN_CLIENT_SENT_PROTOCOLS,
	_WSI_TOKEN_CLIENT_PEER_ADDRESS,
	_WSI_TOKEN_CLIENT_URI,
	_WSI_TOKEN_CLIENT_HOST,
	_WSI_TOKEN_CLIENT_ORIGIN,
	_WSI_TOKEN_CLIENT_METHOD,
	_WSI_TOKEN_CLIENT_IFACE,
	_WSI_TOKEN_CLIENT_ALPN,

	/* always last real token index*/
	WSI_TOKEN_COUNT,

	/* parser state additions, no storage associated */
	WSI_TOKEN_NAME_PART,
	WSI_TOKEN_SKIPPING,
	WSI_TOKEN_SKIPPING_SAW_CR,
	WSI_PARSING_COMPLETE,
	WSI_INIT_TOKEN_MUXURL,
};

struct lws_token_limits {
	unsigned short token_limit[WSI_TOKEN_COUNT]; /**< max chars for this token */
};

/**
 * lws_token_to_string() - returns a textual representation of a hdr token index
 *
 * \param token: token index
 */
LWS_VISIBLE LWS_EXTERN const unsigned char *
lws_token_to_string(enum lws_token_indexes token);

/**
 * lws_hdr_total_length: report length of all fragments of a header totalled up
 *		The returned length does not include the space for a
 *		terminating '\0'
 *
 * \param wsi: websocket connection
 * \param h: which header index we are interested in
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_hdr_total_length(struct lws *wsi, enum lws_token_indexes h);

/**
 * lws_hdr_fragment_length: report length of a single fragment of a header
 *		The returned length does not include the space for a
 *		terminating '\0'
 *
 * \param wsi: websocket connection
 * \param h: which header index we are interested in
 * \param frag_idx: which fragment of h we want to get the length of
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_hdr_fragment_length(struct lws *wsi, enum lws_token_indexes h,
			int frag_idx);

/**
 * lws_hdr_copy() - copy all fragments of the given header to a buffer
 *		The buffer length len must include space for an additional
 *		terminating '\0', or it will fail returning -1.
 *
 * \param wsi: websocket connection
 * \param dest: destination buffer
 * \param len: length of destination buffer
 * \param h: which header index we are interested in
 *
 * copies the whole, aggregated header, even if it was delivered in
 * several actual headers piece by piece.  Returns -1 or length of the whole
 * header.
 */
LWS_VISIBLE LWS_EXTERN int
lws_hdr_copy(struct lws *wsi, char *dest, int len, enum lws_token_indexes h);

/**
 * lws_hdr_copy_fragment() - copy a single fragment of the given header to a buffer
 *		The buffer length len must include space for an additional
 *		terminating '\0', or it will fail returning -1.
 *		If the requested fragment index is not present, it fails
 *		returning -1.
 *
 * \param wsi: websocket connection
 * \param dest: destination buffer
 * \param len: length of destination buffer
 * \param h: which header index we are interested in
 * \param frag_idx: which fragment of h we want to copy
 *
 * Normally this is only useful
 * to parse URI arguments like ?x=1&y=2, token index WSI_TOKEN_HTTP_URI_ARGS
 * fragment 0 will contain "x=1" and fragment 1 "y=2"
 */
LWS_VISIBLE LWS_EXTERN int
lws_hdr_copy_fragment(struct lws *wsi, char *dest, int len,
		      enum lws_token_indexes h, int frag_idx);

/**
 * lws_get_urlarg_by_name() - return pointer to arg value if present
 * \param wsi: the connection to check
 * \param name: the arg name, like "token="
 * \param buf: the buffer to receive the urlarg (including the name= part)
 * \param len: the length of the buffer to receive the urlarg
 *
 *     Returns NULL if not found or a pointer inside buf to just after the
 *     name= part.
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_get_urlarg_by_name(struct lws *wsi, const char *name, char *buf, int len);
///@}

/*! \defgroup HTTP-headers-create HTTP headers: create
 *
 * ## HTTP headers: Create
 *
 * These apis allow you to create HTTP response headers in a way compatible with
 * both HTTP/1.x and HTTP/2.
 *
 * They each append to a buffer taking care about the buffer end, which is
 * passed in as a pointer.  When data is written to the buffer, the current
 * position p is updated accordingly.
 *
 * All of these apis are LWS_WARN_UNUSED_RESULT as they can run out of space
 * and fail with nonzero return.
 */
///@{

#define LWSAHH_CODE_MASK			((1 << 16) - 1)
#define LWSAHH_FLAG_NO_SERVER_NAME		(1 << 30)

/**
 * lws_add_http_header_status() - add the HTTP response status code
 *
 * \param wsi: the connection to check
 * \param code: an HTTP code like 200, 404 etc (see enum http_status)
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Adds the initial response code, so should be called first.
 *
 * Code may additionally take OR'd flags:
 *
 *    LWSAHH_FLAG_NO_SERVER_NAME:  don't apply server name header this time
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_status(struct lws *wsi,
			   unsigned int code, unsigned char **p,
			   unsigned char *end);
/**
 * lws_add_http_header_by_name() - append named header and value
 *
 * \param wsi: the connection to check
 * \param name: the hdr name, like "my-header"
 * \param value: the value after the = for this header
 * \param length: the length of the value
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Appends name: value to the headers
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_by_name(struct lws *wsi, const unsigned char *name,
			    const unsigned char *value, int length,
			    unsigned char **p, unsigned char *end);
/**
 * lws_add_http_header_by_token() - append given header and value
 *
 * \param wsi: the connection to check
 * \param token: the token index for the hdr
 * \param value: the value after the = for this header
 * \param length: the length of the value
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Appends name=value to the headers, but is able to take advantage of better
 * HTTP/2 coding mechanisms where possible.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_by_token(struct lws *wsi, enum lws_token_indexes token,
			     const unsigned char *value, int length,
			     unsigned char **p, unsigned char *end);
/**
 * lws_add_http_header_content_length() - append content-length helper
 *
 * \param wsi: the connection to check
 * \param content_length: the content length to use
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Appends content-length: content_length to the headers
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_content_length(struct lws *wsi,
				   lws_filepos_t content_length,
				   unsigned char **p, unsigned char *end);
/**
 * lws_finalize_http_header() - terminate header block
 *
 * \param wsi: the connection to check
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Indicates no more headers will be added
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_finalize_http_header(struct lws *wsi, unsigned char **p,
			 unsigned char *end);

/**
 * lws_finalize_write_http_header() - Helper finializing and writing http headers
 *
 * \param wsi: the connection to check
 * \param start: pointer to the start of headers in the buffer, eg &buf[LWS_PRE]
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Terminates the headers correctly accoring to the protocol in use (h1 / h2)
 * and writes the headers.  Returns nonzero for error.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_finalize_write_http_header(struct lws *wsi, unsigned char *start,
			       unsigned char **p, unsigned char *end);

#define LWS_ILLEGAL_HTTP_CONTENT_LEN ((lws_filepos_t)-1ll)

/**
 * lws_add_http_common_headers() - Helper preparing common http headers
 *
 * \param wsi: the connection to check
 * \param code: an HTTP code like 200, 404 etc (see enum http_status)
 * \param content_type: the content type, like "text/html"
 * \param content_len: the content length, in bytes
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Adds the initial response code, so should be called first.
 *
 * Code may additionally take OR'd flags:
 *
 *    LWSAHH_FLAG_NO_SERVER_NAME:  don't apply server name header this time
 *
 * This helper just calls public apis to simplify adding headers that are
 * commonly needed.  If it doesn't fit your case, or you want to add additional
 * headers just call the public apis directly yourself for what you want.
 *
 * You can miss out the content length header by providing the constant
 * LWS_ILLEGAL_HTTP_CONTENT_LEN for the content_len.
 *
 * It does not call lws_finalize_http_header(), to allow you to add further
 * headers after calling this.  You will need to call that yourself at the end.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_common_headers(struct lws *wsi, unsigned int code,
			    const char *content_type, lws_filepos_t content_len,
			    unsigned char **p, unsigned char *end);

///@}

/*! \defgroup urlendec Urlencode and Urldecode
 * \ingroup http
 *
 * ##HTML chunked Substitution
 *
 * APIs for receiving chunks of text, replacing a set of variable names via
 * a callback, and then prepending and appending HTML chunked encoding
 * headers.
 */
//@{

/**
 * lws_urlencode() - like strncpy but with urlencoding
 *
 * \param escaped: output buffer
 * \param string: input buffer ('/0' terminated)
 * \param len: output buffer max length
 *
 * Because urlencoding expands the output string, it's not
 * possible to do it in-place, ie, with escaped == string
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_urlencode(char *escaped, const char *string, int len);

/*
 * URLDECODE 1 / 2
 *
 * This simple urldecode only operates until the first '\0' and requires the
 * data to exist all at once
 */
/**
 * lws_urldecode() - like strncpy but with urldecoding
 *
 * \param string: output buffer
 * \param escaped: input buffer ('\0' terminated)
 * \param len: output buffer max length
 *
 * This is only useful for '\0' terminated strings
 *
 * Since urldecoding only shrinks the output string, it is possible to
 * do it in-place, ie, string == escaped
 *
 * Returns 0 if completed OK or nonzero for urldecode violation (non-hex chars
 * where hex required, etc)
 */
LWS_VISIBLE LWS_EXTERN int
lws_urldecode(char *string, const char *escaped, int len);
///@}

/**
 * lws_return_http_status() - Return simple http status
 * \param wsi:		Websocket instance (available from user callback)
 * \param code:		Status index, eg, 404
 * \param html_body:		User-readable HTML description < 1KB, or NULL
 *
 *	Helper to report HTTP errors back to the client cleanly and
 *	consistently
 */
LWS_VISIBLE LWS_EXTERN int
lws_return_http_status(struct lws *wsi, unsigned int code,
		       const char *html_body);

/**
 * lws_http_redirect() - write http redirect out on wsi
 *
 * \param wsi:	websocket connection
 * \param code:	HTTP response code (eg, 301)
 * \param loc:	where to redirect to
 * \param len:	length of loc
 * \param p:	pointer current position in buffer (updated as we write)
 * \param end:	pointer to end of buffer
 *
 * Returns amount written, or < 0 indicating fatal write failure.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_http_redirect(struct lws *wsi, int code, const unsigned char *loc, int len,
		  unsigned char **p, unsigned char *end);

/**
 * lws_http_transaction_completed() - wait for new http transaction or close
 * \param wsi:	websocket connection
 *
 *	Returns 1 if the HTTP connection must close now
 *	Returns 0 and resets connection to wait for new HTTP header /
 *	  transaction if possible
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_http_transaction_completed(struct lws *wsi);

/**
 * lws_http_compression_apply() - apply an http compression transform
 *
 * \param wsi: the wsi to apply the compression transform to
 * \param name: NULL, or the name of the compression transform, eg, "deflate"
 * \param p: pointer to pointer to headers buffer
 * \param end: pointer to end of headers buffer
 * \param decomp: 0 = add compressor to wsi, 1 = add decompressor
 *
 * This allows transparent compression of dynamically generated HTTP.  The
 * requested compression (eg, "deflate") is only applied if the client headers
 * indicated it was supported (and it has support in lws), otherwise it's a NOP.
 *
 * If the requested compression method is NULL, then the supported compression
 * formats are tried, and for non-decompression (server) mode the first that's
 * found on the client's accept-encoding header is chosen.
 *
 * NOTE: the compression transform, same as h2 support, relies on the user
 * code using LWS_WRITE_HTTP and then LWS_WRITE_HTTP_FINAL on the last part
 * written.  The internal lws fileserving code already does this.
 *
 * If the library was built without the cmake option
 * LWS_WITH_HTTP_STREAM_COMPRESSION set, then a NOP is provided for this api,
 * allowing user code to build either way and use compression if available.
 */
LWS_VISIBLE int
lws_http_compression_apply(struct lws *wsi, const char *name,
			   unsigned char **p, unsigned char *end, char decomp);
///@}

