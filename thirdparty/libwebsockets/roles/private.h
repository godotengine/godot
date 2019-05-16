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
 *  This is included from core/private.h
 */

typedef uint32_t lws_wsi_state_t;

/*
 * The wsi->role_ops pointer decides almost everything about what role the wsi
 * will play, h2, raw, ws, etc.
 *
 * However there are a few additional flags needed that vary, such as if the
 * role is a client or server side, if it has that concept.  And the connection
 * fulfilling the role, has a separate dynamic state.
 *
 *   31           16 15      0
 *   [  role flags ] [ state ]
 *
 * The role flags part is generally invariant for the lifetime of the wsi,
 * although it can change if the connection role itself does, eg, if the
 * connection upgrades from H1 -> WS1 the role flags may be changed at that
 * point.
 *
 * The state part reflects the dynamic connection state, and the states are
 * reused between roles.
 *
 * None of the internal role or state representations are made available outside
 * of lws internals.  Even for lws internals, if you add stuff here, please keep
 * the constants inside this header only by adding necessary helpers here and
 * use the helpers in the actual code.  This is to ease any future refactors.
 *
 * Notice LWSIFR_ENCAP means we have a parent wsi that actually carries our
 * data as a stream inside a different protocol.
 */

#define _RS 16

#define LWSIFR_CLIENT		(0x1000 << _RS) /* client side */
#define LWSIFR_SERVER		(0x2000 << _RS) /* server side */

#define LWSIFR_P_ENCAP_H2	(0x0100 << _RS) /* we are encapsulated by h2 */

enum lwsi_role {
	LWSI_ROLE_MASK		=			     (0xffff << _RS),
	LWSI_ROLE_ENCAP_MASK	=			     (0x0f00 << _RS),
};

#define lwsi_role(wsi) (wsi->wsistate & LWSI_ROLE_MASK)
#if !defined (_DEBUG)
#define lwsi_set_role(wsi, role) wsi->wsistate = \
				(wsi->wsistate & (~LWSI_ROLE_MASK)) | role
#else
void lwsi_set_role(struct lws *wsi, lws_wsi_state_t role);
#endif

#define lwsi_role_client(wsi) (!!(wsi->wsistate & LWSIFR_CLIENT))
#define lwsi_role_server(wsi) (!!(wsi->wsistate & LWSIFR_SERVER))
#define lwsi_role_h2_ENCAPSULATION(wsi) \
		((wsi->wsistate & LWSI_ROLE_ENCAP_MASK) == LWSIFR_P_ENCAP_H2)

/* Pollout wants a callback in this state */
#define LWSIFS_POCB		(0x100)
/* Before any protocol connection was established */
#define LWSIFS_NOT_EST		(0x200)

enum lwsi_state {

	/* Phase 1: pre-transport */

	LRS_UNCONNECTED				= LWSIFS_NOT_EST | 0,
	LRS_WAITING_CONNECT			= LWSIFS_NOT_EST | 1,

	/* Phase 2: establishing intermediaries on top of transport */

	LRS_WAITING_PROXY_REPLY			= LWSIFS_NOT_EST | 2,
	LRS_WAITING_SSL				= LWSIFS_NOT_EST | 3,
	LRS_WAITING_SOCKS_GREETING_REPLY	= LWSIFS_NOT_EST | 4,
	LRS_WAITING_SOCKS_CONNECT_REPLY		= LWSIFS_NOT_EST | 5,
	LRS_WAITING_SOCKS_AUTH_REPLY		= LWSIFS_NOT_EST | 6,

	/* Phase 3: establishing tls tunnel */

	LRS_SSL_INIT				= LWSIFS_NOT_EST | 7,
	LRS_SSL_ACK_PENDING			= LWSIFS_NOT_EST | 8,
	LRS_PRE_WS_SERVING_ACCEPT		= LWSIFS_NOT_EST | 9,

	/* Phase 4: connected */

	LRS_WAITING_SERVER_REPLY		= LWSIFS_NOT_EST | 10,
	LRS_H2_AWAIT_PREFACE			= LWSIFS_NOT_EST | 11,
	LRS_H2_AWAIT_SETTINGS			= LWSIFS_NOT_EST |
						  LWSIFS_POCB | 12,

	/* Phase 5: protocol logically established */

	LRS_H2_CLIENT_SEND_SETTINGS		= LWSIFS_POCB | 13,
	LRS_H2_WAITING_TO_SEND_HEADERS		= LWSIFS_POCB | 14,
	LRS_DEFERRING_ACTION			= LWSIFS_POCB | 15,
	LRS_IDLING				= 16,
	LRS_H1C_ISSUE_HANDSHAKE			= 17,
	LRS_H1C_ISSUE_HANDSHAKE2		= 18,
	LRS_ISSUE_HTTP_BODY			= 19,
	LRS_ISSUING_FILE			= 20,
	LRS_HEADERS				= 21,
	LRS_BODY				= 22,
	LRS_ESTABLISHED				= LWSIFS_POCB | 23,
	/* we are established, but we have embarked on serving a single
	 * transaction.  Other transaction input may be pending, but we will
	 * not service it while we are busy dealing with the current
	 * transaction.
	 *
	 * When we complete the current transaction, we would reset our state
	 * back to ESTABLISHED and start to process the next transaction.
	 */
	LRS_DOING_TRANSACTION			= LWSIFS_POCB | 24,

	/* Phase 6: finishing */

	LRS_WAITING_TO_SEND_CLOSE		= LWSIFS_POCB | 25,
	LRS_RETURNED_CLOSE			= LWSIFS_POCB | 26,
	LRS_AWAITING_CLOSE_ACK			= LWSIFS_POCB | 27,
	LRS_FLUSHING_BEFORE_CLOSE		= LWSIFS_POCB | 28,
	LRS_SHUTDOWN				= 29,

	/* Phase 7: dead */

	LRS_DEAD_SOCKET				= 30,

	LRS_MASK				= 0xffff
};

#define lwsi_state(wsi) ((enum lwsi_state)(wsi->wsistate & LRS_MASK))
#define lwsi_state_PRE_CLOSE(wsi) ((enum lwsi_state)(wsi->wsistate_pre_close & LRS_MASK))
#define lwsi_state_est(wsi) (!(wsi->wsistate & LWSIFS_NOT_EST))
#define lwsi_state_est_PRE_CLOSE(wsi) (!(wsi->wsistate_pre_close & LWSIFS_NOT_EST))
#define lwsi_state_can_handle_POLLOUT(wsi) (wsi->wsistate & LWSIFS_POCB)
#if !defined (_DEBUG)
#define lwsi_set_state(wsi, lrs) wsi->wsistate = \
			  (wsi->wsistate & (~LRS_MASK)) | lrs
#else
void lwsi_set_state(struct lws *wsi, lws_wsi_state_t lrs);
#endif

/*
 * internal role-specific ops
 */
struct lws_context_per_thread;
struct lws_role_ops {
	const char *name;
	const char *alpn;
	/*
	 * After http headers have parsed, this is the last chance for a role
	 * to upgrade the connection to something else using the headers.
	 * ws-over-h2 is upgraded from h2 like this.
	 */
	int (*check_upgrades)(struct lws *wsi);
	/* role-specific context init during context creation */
	int (*init_context)(struct lws_context *context,
			    const struct lws_context_creation_info *info);
	/* role-specific per-vhost init during vhost creation */
	int (*init_vhost)(struct lws_vhost *vh,
			  const struct lws_context_creation_info *info);
	/* role-specific per-vhost destructor during vhost destroy */
	int (*destroy_vhost)(struct lws_vhost *vh);
	/* generic 1Hz callback for the role itself */
	int (*periodic_checks)(struct lws_context *context, int tsi,
			       time_t now);
	/* chance for the role to force POLLIN without network activity */
	int (*service_flag_pending)(struct lws_context *context, int tsi);
	/* an fd using this role has POLLIN signalled */
	int (*handle_POLLIN)(struct lws_context_per_thread *pt, struct lws *wsi,
			     struct lws_pollfd *pollfd);
	/* an fd using the role wanted a POLLOUT callback and now has it */
	int (*handle_POLLOUT)(struct lws *wsi);
	/* perform user pollout */
	int (*perform_user_POLLOUT)(struct lws *wsi);
	/* do effective callback on writeable */
	int (*callback_on_writable)(struct lws *wsi);
	/* connection-specific tx credit in bytes */
	lws_fileofs_t (*tx_credit)(struct lws *wsi);
	/* role-specific write formatting */
	int (*write_role_protocol)(struct lws *wsi, unsigned char *buf,
				   size_t len, enum lws_write_protocol *wp);

	/* get encapsulation parent */
	struct lws * (*encapsulation_parent)(struct lws *wsi);

	/* role-specific destructor */
	int (*alpn_negotiated)(struct lws *wsi, const char *alpn);

	/* chance for the role to handle close in the protocol */
	int (*close_via_role_protocol)(struct lws *wsi,
				       enum lws_close_status reason);
	/* role-specific close processing */
	int (*close_role)(struct lws_context_per_thread *pt, struct lws *wsi);
	/* role-specific connection close processing */
	int (*close_kill_connection)(struct lws *wsi,
				     enum lws_close_status reason);
	/* role-specific destructor */
	int (*destroy_role)(struct lws *wsi);

	/*
	 * the callback reasons for WRITEABLE for client, server
	 * (just client applies if no concept of client or server)
	 */
	uint16_t writeable_cb[2];
	/*
	 * the callback reasons for CLOSE for client, server
	 * (just client applies if no concept of client or server)
	 */
	uint16_t close_cb[2];

	unsigned int file_handle:1; /* role operates on files not sockets */
};

/* core roles */
extern struct lws_role_ops role_ops_raw_skt, role_ops_raw_file, role_ops_listen,
			   role_ops_pipe;

/* bring in role private declarations */

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
 #include "roles/http/private.h"
#else
 #define lwsi_role_http(wsi) (0)
#endif

#if defined(LWS_ROLE_H1)
 #include "roles/h1/private.h"
#else
 #define lwsi_role_h1(wsi) (0)
#endif

#if defined(LWS_ROLE_H2)
 #include "roles/h2/private.h"
#else
 #define lwsi_role_h2(wsi) (0)
#endif

#if defined(LWS_ROLE_WS)
 #include "roles/ws/private.h"
#else
 #define lwsi_role_ws(wsi) (0)
#endif

#if defined(LWS_ROLE_CGI)
 #include "roles/cgi/private.h"
#else
 #define lwsi_role_cgi(wsi) (0)
#endif

enum {
	LWS_HP_RET_BAIL_OK,
	LWS_HP_RET_BAIL_DIE,
	LWS_HP_RET_USER_SERVICE,

	LWS_HPI_RET_WSI_ALREADY_DIED,	/* we closed it */
	LWS_HPI_RET_HANDLED,		/* no probs */
	LWS_HPI_RET_PLEASE_CLOSE_ME,	/* close it for us */

	LWS_UPG_RET_DONE,
	LWS_UPG_RET_CONTINUE,
	LWS_UPG_RET_BAIL
};
