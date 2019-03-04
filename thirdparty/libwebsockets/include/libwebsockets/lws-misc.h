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

/** \defgroup misc Miscellaneous APIs
* ##Miscellaneous APIs
*
* Various APIs outside of other categories
*/
///@{

/**
 * lws_start_foreach_ll(): linkedlist iterator helper start
 *
 * \param type: type of iteration, eg, struct xyz *
 * \param it: iterator var name to create
 * \param start: start of list
 *
 * This helper creates an iterator and starts a while (it) {
 * loop.  The iterator runs through the linked list starting at start and
 * ends when it gets a NULL.
 * The while loop should be terminated using lws_start_foreach_ll().
 */
#define lws_start_foreach_ll(type, it, start)\
{ \
	type it = start; \
	while (it) {

/**
 * lws_end_foreach_ll(): linkedlist iterator helper end
 *
 * \param it: same iterator var name given when starting
 * \param nxt: member name in the iterator pointing to next list element
 *
 * This helper is the partner for lws_start_foreach_ll() that ends the
 * while loop.
 */

#define lws_end_foreach_ll(it, nxt) \
		it = it->nxt; \
	} \
}

/**
 * lws_start_foreach_ll_safe(): linkedlist iterator helper start safe against delete
 *
 * \param type: type of iteration, eg, struct xyz *
 * \param it: iterator var name to create
 * \param start: start of list
 * \param nxt: member name in the iterator pointing to next list element
 *
 * This helper creates an iterator and starts a while (it) {
 * loop.  The iterator runs through the linked list starting at start and
 * ends when it gets a NULL.
 * The while loop should be terminated using lws_end_foreach_ll_safe().
 * Performs storage of next increment for situations where iterator can become invalidated
 * during iteration.
 */
#define lws_start_foreach_ll_safe(type, it, start, nxt)\
{ \
	type it = start; \
	while (it) { \
		type next_##it = it->nxt;

/**
 * lws_end_foreach_ll_safe(): linkedlist iterator helper end (pre increment storage)
 *
 * \param it: same iterator var name given when starting
 *
 * This helper is the partner for lws_start_foreach_ll_safe() that ends the
 * while loop. It uses the precreated next_ variable already stored during
 * start.
 */

#define lws_end_foreach_ll_safe(it) \
		it = next_##it; \
	} \
}

/**
 * lws_start_foreach_llp(): linkedlist pointer iterator helper start
 *
 * \param type: type of iteration, eg, struct xyz **
 * \param it: iterator var name to create
 * \param start: start of list
 *
 * This helper creates an iterator and starts a while (it) {
 * loop.  The iterator runs through the linked list starting at the
 * address of start and ends when it gets a NULL.
 * The while loop should be terminated using lws_start_foreach_llp().
 *
 * This helper variant iterates using a pointer to the previous linked-list
 * element.  That allows you to easily delete list members by rewriting the
 * previous pointer to the element's next pointer.
 */
#define lws_start_foreach_llp(type, it, start)\
{ \
	type it = &(start); \
	while (*(it)) {

#define lws_start_foreach_llp_safe(type, it, start, nxt)\
{ \
	type it = &(start); \
	type next; \
	while (*(it)) { \
		next = &((*(it))->nxt); \

/**
 * lws_end_foreach_llp(): linkedlist pointer iterator helper end
 *
 * \param it: same iterator var name given when starting
 * \param nxt: member name in the iterator pointing to next list element
 *
 * This helper is the partner for lws_start_foreach_llp() that ends the
 * while loop.
 */

#define lws_end_foreach_llp(it, nxt) \
		it = &(*(it))->nxt; \
	} \
}

#define lws_end_foreach_llp_safe(it) \
		it = next; \
	} \
}

#define lws_ll_fwd_insert(\
	___new_object,	/* pointer to new object */ \
	___m_list,	/* member for next list object ptr */ \
	___list_head	/* list head */ \
		) {\
		___new_object->___m_list = ___list_head; \
		___list_head = ___new_object; \
	}

#define lws_ll_fwd_remove(\
	___type,	/* type of listed object */ \
	___m_list,	/* member for next list object ptr */ \
	___target,	/* object to remove from list */ \
	___list_head	/* list head */ \
	) { \
                lws_start_foreach_llp(___type **, ___ppss, ___list_head) { \
                        if (*___ppss == ___target) { \
                                *___ppss = ___target->___m_list; \
                                break; \
                        } \
                } lws_end_foreach_llp(___ppss, ___m_list); \
	}

/*
 * doubly linked-list
 */

struct lws_dll { /* abstract */
	struct lws_dll *prev;
	struct lws_dll *next;
};

/*
 * these all point to the composed list objects... you have to use the
 * lws_container_of() helper to recover the start of the containing struct
 */

LWS_VISIBLE LWS_EXTERN void
lws_dll_add_front(struct lws_dll *d, struct lws_dll *phead);

LWS_VISIBLE LWS_EXTERN void
lws_dll_remove(struct lws_dll *d);

struct lws_dll_lws { /* typed as struct lws * */
	struct lws_dll_lws *prev;
	struct lws_dll_lws *next;
};

#define lws_dll_is_null(___dll) (!(___dll)->prev && !(___dll)->next)

static LWS_INLINE void
lws_dll_lws_add_front(struct lws_dll_lws *_a, struct lws_dll_lws *_head)
{
	lws_dll_add_front((struct lws_dll *)_a, (struct lws_dll *)_head);
}

static LWS_INLINE void
lws_dll_lws_remove(struct lws_dll_lws *_a)
{
	lws_dll_remove((struct lws_dll *)_a);
}

/*
 * these are safe against the current container object getting deleted,
 * since the hold his next in a temp and go to that next.  ___tmp is
 * the temp.
 */

#define lws_start_foreach_dll_safe(___type, ___it, ___tmp, ___start) \
{ \
	___type ___it = ___start; \
	while (___it) { \
		___type ___tmp = (___it)->next;

#define lws_end_foreach_dll_safe(___it, ___tmp) \
		___it = ___tmp; \
	} \
}

#define lws_start_foreach_dll(___type, ___it, ___start) \
{ \
	___type ___it = ___start; \
	while (___it) {

#define lws_end_foreach_dll(___it) \
		___it = (___it)->next; \
	} \
}

struct lws_buflist;

/**
 * lws_buflist_append_segment(): add buffer to buflist at head
 *
 * \param head: list head
 * \param buf: buffer to stash
 * \param len: length of buffer to stash
 *
 * Returns -1 on OOM, 1 if this was the first segment on the list, and 0 if
 * it was a subsequent segment.
 */
LWS_VISIBLE LWS_EXTERN int
lws_buflist_append_segment(struct lws_buflist **head, const uint8_t *buf,
			   size_t len);
/**
 * lws_buflist_next_segment_len(): number of bytes left in current segment
 *
 * \param head: list head
 * \param buf: if non-NULL, *buf is written with the address of the start of
 *		the remaining data in the segment
 *
 * Returns the number of bytes left in the current segment.  0 indicates
 * that the buflist is empty (there are no segments on the buflist).
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_buflist_next_segment_len(struct lws_buflist **head, uint8_t **buf);
/**
 * lws_buflist_use_segment(): remove len bytes from the current segment
 *
 * \param head: list head
 * \param len: number of bytes to mark as used
 *
 * If len is less than the remaining length of the current segment, the position
 * in the current segment is simply advanced and it returns.
 *
 * If len uses up the remaining length of the current segment, then the segment
 * is deleted and the list head moves to the next segment if any.
 *
 * Returns the number of bytes left in the current segment.  0 indicates
 * that the buflist is empty (there are no segments on the buflist).
 */
LWS_VISIBLE LWS_EXTERN int
lws_buflist_use_segment(struct lws_buflist **head, size_t len);
/**
 * lws_buflist_destroy_all_segments(): free all segments on the list
 *
 * \param head: list head
 *
 * This frees everything on the list unconditionally.  *head is always
 * NULL after this.
 */
LWS_VISIBLE LWS_EXTERN void
lws_buflist_destroy_all_segments(struct lws_buflist **head);

void
lws_buflist_describe(struct lws_buflist **head, void *id);

/**
 * lws_ptr_diff(): helper to report distance between pointers as an int
 *
 * \param head: the pointer with the larger address
 * \param tail: the pointer with the smaller address
 *
 * This helper gives you an int representing the number of bytes further
 * forward the first pointer is compared to the second pointer.
 */
#define lws_ptr_diff(head, tail) \
			((int)((char *)(head) - (char *)(tail)))

/**
 * lws_snprintf(): snprintf that truncates the returned length too
 *
 * \param str: destination buffer
 * \param size: bytes left in destination buffer
 * \param format: format string
 * \param ...: args for format
 *
 * This lets you correctly truncate buffers by concatenating lengths, if you
 * reach the limit the reported length doesn't exceed the limit.
 */
LWS_VISIBLE LWS_EXTERN int
lws_snprintf(char *str, size_t size, const char *format, ...) LWS_FORMAT(3);

/**
 * lws_strncpy(): strncpy that guarantees NUL on truncated copy
 *
 * \param dest: destination buffer
 * \param src: source buffer
 * \param size: bytes left in destination buffer
 *
 * This lets you correctly truncate buffers by concatenating lengths, if you
 * reach the limit the reported length doesn't exceed the limit.
 */
LWS_VISIBLE LWS_EXTERN char *
lws_strncpy(char *dest, const char *src, size_t size);

/**
 * lws_get_random(): fill a buffer with platform random data
 *
 * \param context: the lws context
 * \param buf: buffer to fill
 * \param len: how much to fill
 *
 * This is intended to be called from the LWS_CALLBACK_RECEIVE callback if
 * it's interested to see if the frame it's dealing with was sent in binary
 * mode.
 */
LWS_VISIBLE LWS_EXTERN int
lws_get_random(struct lws_context *context, void *buf, int len);
/**
 * lws_daemonize(): make current process run in the background
 *
 * \param _lock_path: the filepath to write the lock file
 *
 * Spawn lws as a background process, taking care of various things
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_daemonize(const char *_lock_path);
/**
 * lws_get_library_version(): return string describing the version of lws
 *
 * On unix, also includes the git describe
 */
LWS_VISIBLE LWS_EXTERN const char * LWS_WARN_UNUSED_RESULT
lws_get_library_version(void);

/**
 * lws_wsi_user() - get the user data associated with the connection
 * \param wsi: lws connection
 *
 * Not normally needed since it's passed into the callback
 */
LWS_VISIBLE LWS_EXTERN void *
lws_wsi_user(struct lws *wsi);

/**
 * lws_wsi_set_user() - set the user data associated with the client connection
 * \param wsi: lws connection
 * \param user: user data
 *
 * By default lws allocates this and it's not legal to externally set it
 * yourself.  However client connections may have it set externally when the
 * connection is created... if so, this api can be used to modify it at
 * runtime additionally.
 */
LWS_VISIBLE LWS_EXTERN void
lws_set_wsi_user(struct lws *wsi, void *user);

/**
 * lws_parse_uri:	cut up prot:/ads:port/path into pieces
 *			Notice it does so by dropping '\0' into input string
 *			and the leading / on the path is consequently lost
 *
 * \param p:			incoming uri string.. will get written to
 * \param prot:		result pointer for protocol part (https://)
 * \param ads:		result pointer for address part
 * \param port:		result pointer for port part
 * \param path:		result pointer for path part
 *
 * You may also refer to unix socket addresses, using a '+' at the start of
 * the address.  In this case, the address should end with ':', which is
 * treated as the separator between the address and path (the normal separator
 * '/' is a valid part of the socket path).  Eg,
 *
 * http://+/var/run/mysocket:/my/path
 *
 * If the first character after the + is '@', it's interpreted by lws client
 * processing as meaning to use linux abstract namespace sockets, the @ is
 * replaced with a '\0' before use.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_parse_uri(char *p, const char **prot, const char **ads, int *port,
	      const char **path);
/**
 * lws_cmdline_option():	simple commandline parser
 *
 * \param argc:		count of argument strings
 * \param argv:		argument strings
 * \param val:		string to find
 *
 * Returns NULL if the string \p val is not found in the arguments.
 *
 * If it is found, then it returns a pointer to the next character after \p val.
 * So if \p val is "-d", then for the commandlines "myapp -d15" and
 * "myapp -d 15", in both cases the return will point to the "15".
 *
 * In the case there is no argument, like "myapp -d", the return will
 * either point to the '\\0' at the end of -d, or to the start of the
 * next argument, ie, will be non-NULL.
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_cmdline_option(int argc, const char **argv, const char *val);

/**
 * lws_now_secs(): return seconds since 1970-1-1
 */
LWS_VISIBLE LWS_EXTERN unsigned long
lws_now_secs(void);

/**
 * lws_now_usecs(): return useconds since 1970-1-1
 */
LWS_VISIBLE LWS_EXTERN lws_usec_t
lws_now_usecs(void);

/**
 * lws_compare_time_t(): return relationship between two time_t
 *
 * \param context: struct lws_context
 * \param t1: time_t 1
 * \param t2: time_t 2
 *
 * returns <0 if t2 > t1; >0 if t1 > t2; or == 0 if t1 == t2.
 *
 * This is aware of clock discontiguities that may have affected either t1 or
 * t2 and adapts the comparison for them.
 *
 * For the discontiguity detection to work, you must avoid any arithmetic on
 * the times being compared.  For example to have a timeout that triggers
 * 15s from when it was set, store the time it was set and compare like
 * `if (lws_compare_time_t(context, now, set_time) > 15)`
 */
LWS_VISIBLE LWS_EXTERN int
lws_compare_time_t(struct lws_context *context, time_t t1, time_t t2);

/**
 * lws_get_context - Allow getting lws_context from a Websocket connection
 * instance
 *
 * With this function, users can access context in the callback function.
 * Otherwise users may have to declare context as a global variable.
 *
 * \param wsi:	Websocket connection instance
 */
LWS_VISIBLE LWS_EXTERN struct lws_context * LWS_WARN_UNUSED_RESULT
lws_get_context(const struct lws *wsi);

/**
 * lws_get_vhost_listen_port - Find out the port number a vhost is listening on
 *
 * In the case you passed 0 for the port number at context creation time, you
 * can discover the port number that was actually chosen for the vhost using
 * this api.
 *
 * \param vhost:	Vhost to get listen port from
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_get_vhost_listen_port(struct lws_vhost *vhost);

/**
 * lws_get_count_threads(): how many service threads the context uses
 *
 * \param context: the lws context
 *
 * By default this is always 1, if you asked for more than lws can handle it
 * will clip the number of threads.  So you can use this to find out how many
 * threads are actually in use.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_get_count_threads(struct lws_context *context);

/**
 * lws_get_parent() - get parent wsi or NULL
 * \param wsi: lws connection
 *
 * Specialized wsi like cgi stdin/out/err are associated to a parent wsi,
 * this allows you to get their parent.
 */
LWS_VISIBLE LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_get_parent(const struct lws *wsi);

/**
 * lws_get_child() - get child wsi or NULL
 * \param wsi: lws connection
 *
 * Allows you to find a related wsi from the parent wsi.
 */
LWS_VISIBLE LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_get_child(const struct lws *wsi);

/**
 * lws_get_effective_uid_gid() - find out eventual uid and gid while still root
 *
 * \param context: lws context
 * \param uid: pointer to uid result
 * \param gid: pointer to gid result
 *
 * This helper allows you to find out what the uid and gid for the process will
 * be set to after the privileges are dropped, beforehand.  So while still root,
 * eg in LWS_CALLBACK_PROTOCOL_INIT, you can arrange things like cache dir
 * and subdir creation / permissions down /var/cache dynamically.
 */
LWS_VISIBLE LWS_EXTERN void
lws_get_effective_uid_gid(struct lws_context *context, int *uid, int *gid);

/**
 * lws_get_udp() - get wsi's udp struct
 *
 * \param wsi: lws connection
 *
 * Returns NULL or pointer to the wsi's UDP-specific information
 */
LWS_VISIBLE LWS_EXTERN const struct lws_udp * LWS_WARN_UNUSED_RESULT
lws_get_udp(const struct lws *wsi);

LWS_VISIBLE LWS_EXTERN void *
lws_get_opaque_parent_data(const struct lws *wsi);

LWS_VISIBLE LWS_EXTERN void
lws_set_opaque_parent_data(struct lws *wsi, void *data);

LWS_VISIBLE LWS_EXTERN int
lws_get_child_pending_on_writable(const struct lws *wsi);

LWS_VISIBLE LWS_EXTERN void
lws_clear_child_pending_on_writable(struct lws *wsi);

LWS_VISIBLE LWS_EXTERN int
lws_get_close_length(struct lws *wsi);

LWS_VISIBLE LWS_EXTERN unsigned char *
lws_get_close_payload(struct lws *wsi);

/**
 * lws_get_network_wsi() - Returns wsi that has the tcp connection for this wsi
 *
 * \param wsi: wsi you have
 *
 * Returns wsi that has the tcp connection (which may be the incoming wsi)
 *
 * HTTP/1 connections will always return the incoming wsi
 * HTTP/2 connections may return a different wsi that has the tcp connection
 */
LWS_VISIBLE LWS_EXTERN
struct lws *lws_get_network_wsi(struct lws *wsi);

/**
 * lws_set_allocator() - custom allocator support
 *
 * \param realloc
 *
 * Allows you to replace the allocator (and deallocator) used by lws
 */
LWS_VISIBLE LWS_EXTERN void
lws_set_allocator(void *(*realloc)(void *ptr, size_t size, const char *reason));

enum {
	/*
	 * Flags for enable and disable rxflow with reason bitmap and with
	 * backwards-compatible single bool
	 */
	LWS_RXFLOW_REASON_USER_BOOL		= (1 << 0),
	LWS_RXFLOW_REASON_HTTP_RXBUFFER		= (1 << 6),
	LWS_RXFLOW_REASON_H2_PPS_PENDING	= (1 << 7),

	LWS_RXFLOW_REASON_APPLIES		= (1 << 14),
	LWS_RXFLOW_REASON_APPLIES_ENABLE_BIT	= (1 << 13),
	LWS_RXFLOW_REASON_APPLIES_ENABLE	= LWS_RXFLOW_REASON_APPLIES |
						  LWS_RXFLOW_REASON_APPLIES_ENABLE_BIT,
	LWS_RXFLOW_REASON_APPLIES_DISABLE	= LWS_RXFLOW_REASON_APPLIES,
	LWS_RXFLOW_REASON_FLAG_PROCESS_NOW	= (1 << 12),

};

/**
 * lws_rx_flow_control() - Enable and disable socket servicing for
 *				received packets.
 *
 * If the output side of a server process becomes choked, this allows flow
 * control for the input side.
 *
 * \param wsi:	Websocket connection instance to get callback for
 * \param enable:	0 = disable read servicing for this connection, 1 = enable
 *
 * If you need more than one additive reason for rxflow control, you can give
 * iLWS_RXFLOW_REASON_APPLIES_ENABLE or _DISABLE together with one or more of
 * b5..b0 set to idicate which bits to enable or disable.  If any bits are
 * enabled, rx on the connection is suppressed.
 *
 * LWS_RXFLOW_REASON_FLAG_PROCESS_NOW  flag may also be given to force any change
 * in rxflowbstatus to benapplied immediately, this should be used when you are
 * changing a wsi flow control state from outside a callback on that wsi.
 */
LWS_VISIBLE LWS_EXTERN int
lws_rx_flow_control(struct lws *wsi, int enable);

/**
 * lws_rx_flow_allow_all_protocol() - Allow all connections with this protocol to receive
 *
 * When the user server code realizes it can accept more input, it can
 * call this to have the RX flow restriction removed from all connections using
 * the given protocol.
 * \param context:	lws_context
 * \param protocol:	all connections using this protocol will be allowed to receive
 */
LWS_VISIBLE LWS_EXTERN void
lws_rx_flow_allow_all_protocol(const struct lws_context *context,
			       const struct lws_protocols *protocol);

/**
 * lws_remaining_packet_payload() - Bytes to come before "overall"
 *					      rx fragment is complete
 * \param wsi:		Websocket instance (available from user callback)
 *
 * This tracks how many bytes are left in the current ws fragment, according
 * to the ws length given in the fragment header.
 *
 * If the message was in a single fragment, and there is no compression, this
 * is the same as "how much data is left to read for this message".
 *
 * However, if the message is being sent in multiple fragments, this will
 * reflect the unread amount of the current **fragment**, not the message.  With
 * ws, it is legal to not know the length of the message before it completes.
 *
 * Additionally if the message is sent via the negotiated permessage-deflate
 * extension, this number only tells the amount of **compressed** data left to
 * be read, since that is the only information available at the ws layer.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_remaining_packet_payload(struct lws *wsi);



/**
 * lws_is_ssl() - Find out if connection is using SSL
 * \param wsi:	websocket connection to check
 *
 *	Returns 0 if the connection is not using SSL, 1 if using SSL and
 *	using verified cert, and 2 if using SSL but the cert was not
 *	checked (appears for client wsi told to skip check on connection)
 */
LWS_VISIBLE LWS_EXTERN int
lws_is_ssl(struct lws *wsi);
/**
 * lws_is_cgi() - find out if this wsi is running a cgi process
 * \param wsi: lws connection
 */
LWS_VISIBLE LWS_EXTERN int
lws_is_cgi(struct lws *wsi);

/**
 * lws_open() - platform-specific wrapper for open that prepares the fd
 *
 * \param file: the filepath to open
 * \param oflag: option flags
 * \param mode: optional mode of any created file
 *
 * This is a wrapper around platform open() that sets options on the fd
 * according to lws policy.  Currently that is FD_CLOEXEC to stop the opened
 * fd being available to any child process forked by user code.
 */
LWS_VISIBLE LWS_EXTERN int
lws_open(const char *__file, int __oflag, ...);

struct lws_wifi_scan { /* generic wlan scan item */
	struct lws_wifi_scan *next;
	char ssid[32];
	int32_t rssi; /* divide by .count to get db */
	uint8_t bssid[6];
	uint8_t count;
	uint8_t channel;
	uint8_t authmode;
};

#if defined(LWS_WITH_TLS) && !defined(LWS_WITH_MBEDTLS)
/**
 * lws_get_ssl() - Return wsi's SSL context structure
 * \param wsi:	websocket connection
 *
 * Returns pointer to the SSL library's context structure
 */
LWS_VISIBLE LWS_EXTERN SSL*
lws_get_ssl(struct lws *wsi);
#endif

/** \defgroup smtp SMTP related functions
 * ##SMTP related functions
 * \ingroup lwsapi
 *
 * These apis let you communicate with a local SMTP server to send email from
 * lws.  It handles all the SMTP sequencing and protocol actions.
 *
 * Your system should have postfix, sendmail or another MTA listening on port
 * 25 and able to send email using the "mail" commandline app.  Usually distro
 * MTAs are configured for this by default.
 *
 * It runs via its own libuv events if initialized (which requires giving it
 * a libuv loop to attach to).
 *
 * It operates using three callbacks, on_next() queries if there is a new email
 * to send, on_get_body() asks for the body of the email, and on_sent() is
 * called after the email is successfully sent.
 *
 * To use it
 *
 *  - create an lws_email struct
 *
 *  - initialize data, loop, the email_* strings, max_content_size and
 *    the callbacks
 *
 *  - call lws_email_init()
 *
 *  When you have at least one email to send, call lws_email_check() to
 *  schedule starting to send it.
 */
//@{
#ifdef LWS_WITH_SMTP

/** enum lwsgs_smtp_states - where we are in SMTP protocol sequence */
enum lwsgs_smtp_states {
	LGSSMTP_IDLE, /**< awaiting new email */
	LGSSMTP_CONNECTING, /**< opening tcp connection to MTA */
	LGSSMTP_CONNECTED, /**< tcp connection to MTA is connected */
	LGSSMTP_SENT_HELO, /**< sent the HELO */
	LGSSMTP_SENT_FROM, /**< sent FROM */
	LGSSMTP_SENT_TO, /**< sent TO */
	LGSSMTP_SENT_DATA, /**< sent DATA request */
	LGSSMTP_SENT_BODY, /**< sent the email body */
	LGSSMTP_SENT_QUIT, /**< sent the session quit */
};

/** struct lws_email - abstract context for performing SMTP operations */
struct lws_email {
	void *data;
	/**< opaque pointer set by user code and available to the callbacks */
	uv_loop_t *loop;
	/**< the libuv loop we will work on */

	char email_smtp_ip[32]; /**< Fill before init, eg, "127.0.0.1" */
	char email_helo[32];	/**< Fill before init, eg, "myserver.com" */
	char email_from[100];	/**< Fill before init or on_next */
	char email_to[100];	/**< Fill before init or on_next */

	unsigned int max_content_size;
	/**< largest possible email body size */

	/* Fill all the callbacks before init */

	int (*on_next)(struct lws_email *email);
	/**< (Fill in before calling lws_email_init)
	 * called when idle, 0 = another email to send, nonzero is idle.
	 * If you return 0, all of the email_* char arrays must be set
	 * to something useful. */
	int (*on_sent)(struct lws_email *email);
	/**< (Fill in before calling lws_email_init)
	 * called when transfer of the email to the SMTP server was
	 * successful, your callback would remove the current email
	 * from its queue */
	int (*on_get_body)(struct lws_email *email, char *buf, int len);
	/**< (Fill in before calling lws_email_init)
	 * called when the body part of the queued email is about to be
	 * sent to the SMTP server. */


	/* private things */
	uv_timer_t timeout_email; /**< private */
	enum lwsgs_smtp_states estate; /**< private */
	uv_connect_t email_connect_req; /**< private */
	uv_tcp_t email_client; /**< private */
	time_t email_connect_started; /**< private */
	char email_buf[256]; /**< private */
	char *content; /**< private */
};

/**
 * lws_email_init() - Initialize a struct lws_email
 *
 * \param email: struct lws_email to init
 * \param loop: libuv loop to use
 * \param max_content: max email content size
 *
 * Prepares a struct lws_email for use ending SMTP
 */
LWS_VISIBLE LWS_EXTERN int
lws_email_init(struct lws_email *email, uv_loop_t *loop, int max_content);

/**
 * lws_email_check() - Request check for new email
 *
 * \param email: struct lws_email context to check
 *
 * Schedules a check for new emails in 1s... call this when you have queued an
 * email for send.
 */
LWS_VISIBLE LWS_EXTERN void
lws_email_check(struct lws_email *email);
/**
 * lws_email_destroy() - stop using the struct lws_email
 *
 * \param email: the struct lws_email context
 *
 * Stop sending email using email and free allocations
 */
LWS_VISIBLE LWS_EXTERN void
lws_email_destroy(struct lws_email *email);

#endif
//@}

///@}
