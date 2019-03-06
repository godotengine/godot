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

/*! \defgroup extensions Extension related functions
 * ##Extension releated functions
 *
 *  Ws defines optional extensions, lws provides the ability to implement these
 *  in user code if so desired.
 *
 *  We provide one extensions permessage-deflate.
 */
///@{

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
enum lws_extension_callback_reasons {
	LWS_EXT_CB_CONSTRUCT				=  4,
	LWS_EXT_CB_CLIENT_CONSTRUCT			=  5,
	LWS_EXT_CB_DESTROY				=  8,
	LWS_EXT_CB_PACKET_TX_PRESEND			= 12,
	LWS_EXT_CB_PAYLOAD_TX				= 21,
	LWS_EXT_CB_PAYLOAD_RX				= 22,
	LWS_EXT_CB_OPTION_DEFAULT			= 23,
	LWS_EXT_CB_OPTION_SET				= 24,
	LWS_EXT_CB_OPTION_CONFIRM			= 25,
	LWS_EXT_CB_NAMED_OPTION_SET			= 26,

	/****** add new things just above ---^ ******/
};

/** enum lws_ext_options_types */
enum lws_ext_options_types {
	EXTARG_NONE, /**< does not take an argument */
	EXTARG_DEC,  /**< requires a decimal argument */
	EXTARG_OPT_DEC /**< may have an optional decimal argument */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/** struct lws_ext_options -	Option arguments to the extension.  These are
 *				used in the negotiation at ws upgrade time.
 *				The helper function lws_ext_parse_options()
 *				uses these to generate callbacks */
struct lws_ext_options {
	const char *name; /**< Option name, eg, "server_no_context_takeover" */
	enum lws_ext_options_types type; /**< What kind of args the option can take */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/** struct lws_ext_option_arg */
struct lws_ext_option_arg {
	const char *option_name; /**< may be NULL, option_index used then */
	int option_index; /**< argument ordinal to use if option_name missing */
	const char *start; /**< value */
	int len; /**< length of value */
};

/**
 * typedef lws_extension_callback_function() - Hooks to allow extensions to operate
 * \param context:	Websockets context
 * \param ext:	This extension
 * \param wsi:	Opaque websocket instance pointer
 * \param reason:	The reason for the call
 * \param user:	Pointer to ptr to per-session user data allocated by library
 * \param in:		Pointer used for some callback reasons
 * \param len:	Length set for some callback reasons
 *
 *	Each extension that is active on a particular connection receives
 *	callbacks during the connection lifetime to allow the extension to
 *	operate on websocket data and manage itself.
 *
 *	Libwebsockets takes care of allocating and freeing "user" memory for
 *	each active extension on each connection.  That is what is pointed to
 *	by the user parameter.
 *
 *	LWS_EXT_CB_CONSTRUCT:  called when the server has decided to
 *		select this extension from the list provided by the client,
 *		just before the server will send back the handshake accepting
 *		the connection with this extension active.  This gives the
 *		extension a chance to initialize its connection context found
 *		in user.
 *
 *	LWS_EXT_CB_CLIENT_CONSTRUCT: same as LWS_EXT_CB_CONSTRUCT
 *		but called when client is instantiating this extension.  Some
 *		extensions will work the same on client and server side and then
 *		you can just merge handlers for both CONSTRUCTS.
 *
 *	LWS_EXT_CB_DESTROY:  called when the connection the extension was
 *		being used on is about to be closed and deallocated.  It's the
 *		last chance for the extension to deallocate anything it has
 *		allocated in the user data (pointed to by user) before the
 *		user data is deleted.  This same callback is used whether you
 *		are in client or server instantiation context.
 *
 *	LWS_EXT_CB_PACKET_TX_PRESEND: this works the same way as
 *		LWS_EXT_CB_PACKET_RX_PREPARSE above, except it gives the
 *		extension a chance to change websocket data just before it will
 *		be sent out.  Using the same lws_token pointer scheme in in,
 *		the extension can change the buffer and the length to be
 *		transmitted how it likes.  Again if it wants to grow the
 *		buffer safely, it should copy the data into its own buffer and
 *		set the lws_tokens token pointer to it.
 *
 *	LWS_EXT_CB_ARGS_VALIDATE:
 */
typedef int
lws_extension_callback_function(struct lws_context *context,
			      const struct lws_extension *ext, struct lws *wsi,
			      enum lws_extension_callback_reasons reason,
			      void *user, void *in, size_t len);

/** struct lws_extension -	An extension we support */
struct lws_extension {
	const char *name; /**< Formal extension name, eg, "permessage-deflate" */
	lws_extension_callback_function *callback; /**< Service callback */
	const char *client_offer; /**< String containing exts and options client offers */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/**
 * lws_set_extension_option(): set extension option if possible
 *
 * \param wsi:	websocket connection
 * \param ext_name:	name of ext, like "permessage-deflate"
 * \param opt_name:	name of option, like "rx_buf_size"
 * \param opt_val:	value to set option to
 */
LWS_VISIBLE LWS_EXTERN int
lws_set_extension_option(struct lws *wsi, const char *ext_name,
			 const char *opt_name, const char *opt_val);

/**
 * lws_ext_parse_options() - deal with parsing negotiated extension options
 *
 * \param ext: related extension struct
 * \param wsi:	websocket connection
 * \param ext_user: per-connection extension private data
 * \param opts: list of supported options
 * \param o: option string to parse
 * \param len: length
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ext_parse_options(const struct lws_extension *ext, struct lws *wsi,
		      void *ext_user, const struct lws_ext_options *opts,
		      const char *o, int len);

/** lws_extension_callback_pm_deflate() - extension for RFC7692
 *
 * \param context:	lws context
 * \param ext:	related lws_extension struct
 * \param wsi:	websocket connection
 * \param reason:	incoming callback reason
 * \param user:	per-connection extension private data
 * \param in:	pointer parameter
 * \param len:	length parameter
 *
 * Built-in callback implementing RFC7692 permessage-deflate
 */
LWS_EXTERN int
lws_extension_callback_pm_deflate(struct lws_context *context,
				  const struct lws_extension *ext,
				  struct lws *wsi,
				  enum lws_extension_callback_reasons reason,
				  void *user, void *in, size_t len);

/*
 * The internal exts are part of the public abi
 * If we add more extensions, publish the callback here  ------v
 */
///@}
