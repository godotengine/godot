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

/** \defgroup form-parsing  Form Parsing
 * \ingroup http
 * ##POSTed form parsing functions
 *
 * These lws_spa (stateful post arguments) apis let you parse and urldecode
 * POSTed form arguments, both using simple urlencoded and multipart transfer
 * encoding.
 *
 * It's capable of handling file uploads as well a named input parsing,
 * and the apis are the same for both form upload styles.
 *
 * You feed it a list of parameter names and it creates pointers to the
 * urldecoded arguments: file upload parameters pass the file data in chunks to
 * a user-supplied callback as they come.
 *
 * Since it's stateful, it handles the incoming data needing more than one
 * POST_BODY callback and has no limit on uploaded file size.
 */
///@{

/** enum lws_spa_fileupload_states */
enum lws_spa_fileupload_states {
	LWS_UFS_CONTENT,
	/**< a chunk of file content has arrived */
	LWS_UFS_FINAL_CONTENT,
	/**< the last chunk (possibly zero length) of file content has arrived */
	LWS_UFS_OPEN
	/**< a new file is starting to arrive */
};

/**
 * lws_spa_fileupload_cb() - callback to receive file upload data
 *
 * \param data: opt_data pointer set in lws_spa_create
 * \param name: name of the form field being uploaded
 * \param filename: original filename from client
 * \param buf: start of data to receive
 * \param len: length of data to receive
 * \param state: information about how this call relates to file
 *
 * Notice name and filename shouldn't be trusted, as they are passed from
 * HTTP provided by the client.
 */
typedef int (*lws_spa_fileupload_cb)(void *data, const char *name,
				     const char *filename, char *buf, int len,
				     enum lws_spa_fileupload_states state);

/** struct lws_spa - opaque urldecode parser capable of handling multipart
 *			and file uploads */
struct lws_spa;

/**
 * lws_spa_create() - create urldecode parser
 *
 * \param wsi: lws connection (used to find Content Type)
 * \param param_names: array of form parameter names, like "username"
 * \param count_params: count of param_names
 * \param max_storage: total amount of form parameter values we can store
 * \param opt_cb: NULL, or callback to receive file upload data.
 * \param opt_data: NULL, or user pointer provided to opt_cb.
 *
 * Creates a urldecode parser and initializes it.
 *
 * opt_cb can be NULL if you just want normal name=value parsing, however
 * if one or more entries in your form are bulk data (file transfer), you
 * can provide this callback and filter on the name callback parameter to
 * treat that urldecoded data separately.  The callback should return -1
 * in case of fatal error, and 0 if OK.
 */
LWS_VISIBLE LWS_EXTERN struct lws_spa *
lws_spa_create(struct lws *wsi, const char * const *param_names,
	       int count_params, int max_storage, lws_spa_fileupload_cb opt_cb,
	       void *opt_data);

/**
 * lws_spa_process() - parses a chunk of input data
 *
 * \param spa: the parser object previously created
 * \param in: incoming, urlencoded data
 * \param len: count of bytes valid at \param in
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_process(struct lws_spa *spa, const char *in, int len);

/**
 * lws_spa_finalize() - indicate incoming data completed
 *
 * \param spa: the parser object previously created
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_finalize(struct lws_spa *spa);

/**
 * lws_spa_get_length() - return length of parameter value
 *
 * \param spa: the parser object previously created
 * \param n: parameter ordinal to return length of value for
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_get_length(struct lws_spa *spa, int n);

/**
 * lws_spa_get_string() - return pointer to parameter value
 * \param spa: the parser object previously created
 * \param n: parameter ordinal to return pointer to value for
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_spa_get_string(struct lws_spa *spa, int n);

/**
 * lws_spa_destroy() - destroy parser object
 *
 * \param spa: the parser object previously created
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_destroy(struct lws_spa *spa);
///@}
