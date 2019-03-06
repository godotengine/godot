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

/*! \defgroup sending-data Sending data

    APIs related to writing data on a connection
*/
//@{
#if !defined(LWS_SIZEOFPTR)
#define LWS_SIZEOFPTR ((int)sizeof (void *))
#endif

#if defined(__x86_64__)
#define _LWS_PAD_SIZE 16	/* Intel recommended for best performance */
#else
#define _LWS_PAD_SIZE LWS_SIZEOFPTR   /* Size of a pointer on the target arch */
#endif
#define _LWS_PAD(n) (((n) % _LWS_PAD_SIZE) ? \
		((n) + (_LWS_PAD_SIZE - ((n) % _LWS_PAD_SIZE))) : (n))
/* last 2 is for lws-meta */
#define LWS_PRE _LWS_PAD(4 + 10 + 2)
/* used prior to 1.7 and retained for backward compatibility */
#define LWS_SEND_BUFFER_PRE_PADDING LWS_PRE
#define LWS_SEND_BUFFER_POST_PADDING 0

#define LWS_WRITE_RAW LWS_WRITE_HTTP

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
enum lws_write_protocol {
	LWS_WRITE_TEXT						= 0,
	/**< Send a ws TEXT message,the pointer must have LWS_PRE valid
	 * memory behind it.
	 *
	 * The receiver expects only valid utf-8 in the payload */
	LWS_WRITE_BINARY					= 1,
	/**< Send a ws BINARY message, the pointer must have LWS_PRE valid
	 * memory behind it.
	 *
	 * Any sequence of bytes is valid */
	LWS_WRITE_CONTINUATION					= 2,
	/**< Continue a previous ws message, the pointer must have LWS_PRE valid
	 * memory behind it */
	LWS_WRITE_HTTP						= 3,
	/**< Send HTTP content */

	/* LWS_WRITE_CLOSE is handled by lws_close_reason() */
	LWS_WRITE_PING						= 5,
	LWS_WRITE_PONG						= 6,

	/* Same as write_http but we know this write ends the transaction */
	LWS_WRITE_HTTP_FINAL					= 7,

	/* HTTP2 */

	LWS_WRITE_HTTP_HEADERS					= 8,
	/**< Send http headers (http2 encodes this payload and LWS_WRITE_HTTP
	 * payload differently, http 1.x links also handle this correctly. so
	 * to be compatible with both in the future,header response part should
	 * be sent using this regardless of http version expected)
	 */
	LWS_WRITE_HTTP_HEADERS_CONTINUATION			= 9,
	/**< Continuation of http/2 headers
	 */

	/****** add new things just above ---^ ******/

	/* flags */

	LWS_WRITE_BUFLIST = 0x20,
	/**< Don't actually write it... stick it on the output buflist and
	 *   write it as soon as possible.  Useful if you learn you have to
	 *   write something, have the data to write to hand but the timing is
	 *   unrelated as to whether the connection is writable or not, and were
	 *   otherwise going to have to allocate a temp buffer and write it
	 *   later anyway */

	LWS_WRITE_NO_FIN = 0x40,
	/**< This part of the message is not the end of the message */

	LWS_WRITE_H2_STREAM_END = 0x80,
	/**< Flag indicates this packet should go out with STREAM_END if h2
	 * STREAM_END is allowed on DATA or HEADERS.
	 */

	LWS_WRITE_CLIENT_IGNORE_XOR_MASK = 0x80
	/**< client packet payload goes out on wire unmunged
	 * only useful for security tests since normal servers cannot
	 * decode the content if used */
};

/* used with LWS_CALLBACK_CHILD_WRITE_VIA_PARENT */

struct lws_write_passthru {
	struct lws *wsi;
	unsigned char *buf;
	size_t len;
	enum lws_write_protocol wp;
};


/**
 * lws_write() - Apply protocol then write data to client
 * \param wsi:	Websocket instance (available from user callback)
 * \param buf:	The data to send.  For data being sent on a websocket
 *		connection (ie, not default http), this buffer MUST have
 *		LWS_PRE bytes valid BEFORE the pointer.
 *		This is so the protocol header data can be added in-situ.
 * \param len:	Count of the data bytes in the payload starting from buf
 * \param protocol:	Use LWS_WRITE_HTTP to reply to an http connection, and one
 *		of LWS_WRITE_BINARY or LWS_WRITE_TEXT to send appropriate
 *		data on a websockets connection.  Remember to allow the extra
 *		bytes before and after buf if LWS_WRITE_BINARY or LWS_WRITE_TEXT
 *		are used.
 *
 *	This function provides the way to issue data back to the client
 *	for both http and websocket protocols.
 *
 * IMPORTANT NOTICE!
 *
 * When sending with websocket protocol
 *
 * LWS_WRITE_TEXT,
 * LWS_WRITE_BINARY,
 * LWS_WRITE_CONTINUATION,
 * LWS_WRITE_PING,
 * LWS_WRITE_PONG,
 *
 * or sending on http/2,
 *
 * the send buffer has to have LWS_PRE bytes valid BEFORE the buffer pointer you
 * pass to lws_write().  Since you'll probably want to use http/2 before too
 * long, it's wise to just always do this with lws_write buffers... LWS_PRE is
 * typically 16 bytes it's not going to hurt usually.
 *
 * start of alloc       ptr passed to lws_write      end of allocation
 *       |                         |                         |
 *       v  <-- LWS_PRE bytes -->  v                         v
 *       [----------------  allocated memory  ---------------]
 *              (for lws use)      [====== user buffer ======]
 *
 * This allows us to add protocol info before and after the data, and send as
 * one packet on the network without payload copying, for maximum efficiency.
 *
 * So for example you need this kind of code to use lws_write with a
 * 128-byte payload
 *
 *   char buf[LWS_PRE + 128];
 *
 *   // fill your part of the buffer... for example here it's all zeros
 *   memset(&buf[LWS_PRE], 0, 128);
 *
 *   lws_write(wsi, &buf[LWS_PRE], 128, LWS_WRITE_TEXT);
 *
 * LWS_PRE is at least the frame nonce + 2 header + 8 length
 * LWS_SEND_BUFFER_POST_PADDING is deprecated, it's now 0 and can be left off.
 * The example apps no longer use it.
 *
 * Pad LWS_PRE to the CPU word size, so that word references
 * to the address immediately after the padding won't cause an unaligned access
 * error. Sometimes for performance reasons the recommended padding is even
 * larger than sizeof(void *).
 *
 *	In the case of sending using websocket protocol, be sure to allocate
 *	valid storage before and after buf as explained above.  This scheme
 *	allows maximum efficiency of sending data and protocol in a single
 *	packet while not burdening the user code with any protocol knowledge.
 *
 *	Return may be -1 for a fatal error needing connection close, or the
 *	number of bytes sent.
 *
 * Truncated Writes
 * ================
 *
 * The OS may not accept everything you asked to write on the connection.
 *
 * Posix defines POLLOUT indication from poll() to show that the connection
 * will accept more write data, but it doesn't specifiy how much.  It may just
 * accept one byte of whatever you wanted to send.
 *
 * LWS will buffer the remainder automatically, and send it out autonomously.
 *
 * During that time, WRITABLE callbacks will be suppressed.
 *
 * This is to handle corner cases where unexpectedly the OS refuses what we
 * usually expect it to accept.  You should try to send in chunks that are
 * almost always accepted in order to avoid the inefficiency of the buffering.
 */
LWS_VISIBLE LWS_EXTERN int
lws_write(struct lws *wsi, unsigned char *buf, size_t len,
	  enum lws_write_protocol protocol);

/* helper for case where buffer may be const */
#define lws_write_http(wsi, buf, len) \
	lws_write(wsi, (unsigned char *)(buf), len, LWS_WRITE_HTTP)

/* helper for multi-frame ws message flags */
static LWS_INLINE int
lws_write_ws_flags(int initial, int is_start, int is_end)
{
	int r;

	if (is_start)
		r = initial;
	else
		r = LWS_WRITE_CONTINUATION;

	if (!is_end)
		r |= LWS_WRITE_NO_FIN;

	return r;
}
///@}
