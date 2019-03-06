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

/*! \defgroup generic-sessions plugin: generic-sessions
 * \ingroup Protocols-and-Plugins
 *
 * ##Plugin Generic-sessions related
 *
 * generic-sessions plugin provides a reusable, generic session and login /
 * register / forgot password framework including email verification.
 */
///@{

#define LWSGS_EMAIL_CONTENT_SIZE 16384
/**< Maximum size of email we might send */

/* SHA-1 binary and hexified versions */
/** typedef struct lwsgw_hash_bin */
typedef struct { unsigned char bin[20]; /**< binary representation of hash */} lwsgw_hash_bin;
/** typedef struct lwsgw_hash */
typedef struct { char id[41]; /**< ascii hex representation of hash */ } lwsgw_hash;

/** enum lwsgs_auth_bits */
enum lwsgs_auth_bits {
	LWSGS_AUTH_LOGGED_IN	= 1, /**< user is logged in as somebody */
	LWSGS_AUTH_ADMIN	= 2, /**< logged in as the admin user */
	LWSGS_AUTH_VERIFIED	= 4, /**< user has verified his email */
	LWSGS_AUTH_FORGOT_FLOW	= 8, /**< just completed "forgot password" */
};

/** struct lws_session_info - information about user session status */
struct lws_session_info {
	char username[32]; /**< username logged in as, or empty string */
	char email[100]; /**< email address associated with login, or empty string */
	char ip[72]; /**< ip address session was started from */
	unsigned int mask; /**< access rights mask associated with session
	 	 	    * see enum lwsgs_auth_bits */
	char session[42]; /**< session id string, usable as opaque uid when not logged in */
};

/** enum lws_gs_event */
enum lws_gs_event {
	LWSGSE_CREATED, /**< a new user was created */
	LWSGSE_DELETED  /**< an existing user was deleted */
};

/** struct lws_gs_event_args */
struct lws_gs_event_args {
	enum lws_gs_event event; /**< which event happened */
	const char *username; /**< which username the event happened to */
	const char *email; /**< the email address of that user */
};

///@}
