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

/*! \defgroup cgi cgi handling
 *
 * ##CGI handling
 *
 * These functions allow low-level control over stdin/out/err of the cgi.
 *
 * However for most cases, binding the cgi to http in and out, the default
 * lws implementation already does the right thing.
 */

enum lws_enum_stdinouterr {
	LWS_STDIN = 0,
	LWS_STDOUT = 1,
	LWS_STDERR = 2,
};

enum lws_cgi_hdr_state {
	LCHS_HEADER,
	LCHS_CR1,
	LCHS_LF1,
	LCHS_CR2,
	LCHS_LF2,
	LHCS_RESPONSE,
	LHCS_DUMP_HEADERS,
	LHCS_PAYLOAD,
	LCHS_SINGLE_0A,
};

struct lws_cgi_args {
	struct lws **stdwsi; /**< get fd with lws_get_socket_fd() */
	enum lws_enum_stdinouterr ch; /**< channel index */
	unsigned char *data; /**< for messages with payload */
	enum lws_cgi_hdr_state hdr_state; /**< track where we are in cgi headers */
	int len; /**< length */
};

#ifdef LWS_WITH_CGI
/**
 * lws_cgi: spawn network-connected cgi process
 *
 * \param wsi: connection to own the process
 * \param exec_array: array of "exec-name" "arg1" ... "argn" NULL
 * \param script_uri_path_len: how many chars on the left of the uri are the
 *        path to the cgi, or -1 to spawn without URL-related env vars
 * \param timeout_secs: seconds script should be allowed to run
 * \param mp_cgienv: pvo list with per-vhost cgi options to put in env
 */
LWS_VISIBLE LWS_EXTERN int
lws_cgi(struct lws *wsi, const char * const *exec_array,
	int script_uri_path_len, int timeout_secs,
	const struct lws_protocol_vhost_options *mp_cgienv);

/**
 * lws_cgi_write_split_stdout_headers: write cgi output accounting for header part
 *
 * \param wsi: connection to own the process
 */
LWS_VISIBLE LWS_EXTERN int
lws_cgi_write_split_stdout_headers(struct lws *wsi);

/**
 * lws_cgi_kill: terminate cgi process associated with wsi
 *
 * \param wsi: connection to own the process
 */
LWS_VISIBLE LWS_EXTERN int
lws_cgi_kill(struct lws *wsi);

/**
 * lws_cgi_get_stdwsi: get wsi for stdin, stdout, or stderr
 *
 * \param wsi: parent wsi that has cgi
 * \param ch: which of LWS_STDIN, LWS_STDOUT or LWS_STDERR
 */
LWS_VISIBLE LWS_EXTERN struct lws *
lws_cgi_get_stdwsi(struct lws *wsi, enum lws_enum_stdinouterr ch);

#endif
///@}

