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

/*! \defgroup Protocols-and-Plugins Protocols and Plugins
 * \ingroup lwsapi
 *
 * ##Protocol and protocol plugin -related apis
 *
 * Protocols bind ws protocol names to a custom callback specific to that
 * protocol implementaion.
 *
 * A list of protocols can be passed in at context creation time, but it is
 * also legal to leave that NULL and add the protocols and their callback code
 * using plugins.
 *
 * Plugins are much preferable compared to cut and pasting code into an
 * application each time, since they can be used standalone.
 */
///@{
/** struct lws_protocols -	List of protocols and handlers client or server
 *					supports. */

struct lws_protocols {
	const char *name;
	/**< Protocol name that must match the one given in the client
	 * Javascript new WebSocket(url, 'protocol') name. */
	lws_callback_function *callback;
	/**< The service callback used for this protocol.  It allows the
	 * service action for an entire protocol to be encapsulated in
	 * the protocol-specific callback */
	size_t per_session_data_size;
	/**< Each new connection using this protocol gets
	 * this much memory allocated on connection establishment and
	 * freed on connection takedown.  A pointer to this per-connection
	 * allocation is passed into the callback in the 'user' parameter */
	size_t rx_buffer_size;
	/**< lws allocates this much space for rx data and informs callback
	 * when something came.  Due to rx flow control, the callback may not
	 * be able to consume it all without having to return to the event
	 * loop.  That is supported in lws.
	 *
	 * If .tx_packet_size is 0, this also controls how much may be sent at
	 * once for backwards compatibility.
	 */
	unsigned int id;
	/**< ignored by lws, but useful to contain user information bound
	 * to the selected protocol.  For example if this protocol was
	 * called "myprotocol-v2", you might set id to 2, and the user
	 * code that acts differently according to the version can do so by
	 * switch (wsi->protocol->id), user code might use some bits as
	 * capability flags based on selected protocol version, etc. */
	void *user; /**< ignored by lws, but user code can pass a pointer
			here it can later access from the protocol callback */
	size_t tx_packet_size;
	/**< 0 indicates restrict send() size to .rx_buffer_size for backwards-
	 * compatibility.
	 * If greater than zero, a single send() is restricted to this amount
	 * and any remainder is buffered by lws and sent afterwards also in
	 * these size chunks.  Since that is expensive, it's preferable
	 * to restrict one fragment you are trying to send to match this
	 * size.
	 */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/**
 * lws_vhost_name_to_protocol() - get vhost's protocol object from its name
 *
 * \param vh: vhost to search
 * \param name: protocol name
 *
 * Returns NULL or a pointer to the vhost's protocol of the requested name
 */
LWS_VISIBLE LWS_EXTERN const struct lws_protocols *
lws_vhost_name_to_protocol(struct lws_vhost *vh, const char *name);

/**
 * lws_get_protocol() - Returns a protocol pointer from a websocket
 *				  connection.
 * \param wsi:	pointer to struct websocket you want to know the protocol of
 *
 *
 *	Some apis can act on all live connections of a given protocol,
 *	this is how you can get a pointer to the active protocol if needed.
 */
LWS_VISIBLE LWS_EXTERN const struct lws_protocols *
lws_get_protocol(struct lws *wsi);

/** lws_protocol_get() -  deprecated: use lws_get_protocol */
LWS_VISIBLE LWS_EXTERN const struct lws_protocols *
lws_protocol_get(struct lws *wsi) LWS_WARN_DEPRECATED;

/**
 * lws_protocol_vh_priv_zalloc() - Allocate and zero down a protocol's per-vhost
 *				   storage
 * \param vhost:	vhost the instance is related to
 * \param prot:		protocol the instance is related to
 * \param size:		bytes to allocate
 *
 * Protocols often find it useful to allocate a per-vhost struct, this is a
 * helper to be called in the per-vhost init LWS_CALLBACK_PROTOCOL_INIT
 */
LWS_VISIBLE LWS_EXTERN void *
lws_protocol_vh_priv_zalloc(struct lws_vhost *vhost,
			    const struct lws_protocols *prot, int size);

/**
 * lws_protocol_vh_priv_get() - retreive a protocol's per-vhost storage
 *
 * \param vhost:	vhost the instance is related to
 * \param prot:		protocol the instance is related to
 *
 * Recover a pointer to the allocated per-vhost storage for the protocol created
 * by lws_protocol_vh_priv_zalloc() earlier
 */
LWS_VISIBLE LWS_EXTERN void *
lws_protocol_vh_priv_get(struct lws_vhost *vhost,
			 const struct lws_protocols *prot);

/**
 * lws_adjust_protocol_psds - change a vhost protocol's per session data size
 *
 * \param wsi: a connection with the protocol to change
 * \param new_size: the new size of the per session data size for the protocol
 *
 * Returns user_space for the wsi, after allocating
 *
 * This should not be used except to initalize a vhost protocol's per session
 * data size one time, before any connections are accepted.
 *
 * Sometimes the protocol wraps another protocol and needs to discover and set
 * its per session data size at runtime.
 */
LWS_VISIBLE LWS_EXTERN void *
lws_adjust_protocol_psds(struct lws *wsi, size_t new_size);

/**
 * lws_finalize_startup() - drop initial process privileges
 *
 * \param context:	lws context
 *
 * This is called after the end of the vhost protocol initializations, but
 * you may choose to call it earlier
 */
LWS_VISIBLE LWS_EXTERN int
lws_finalize_startup(struct lws_context *context);

/**
 * lws_pvo_search() - helper to find a named pvo in a linked-list
 *
 * \param pvo:	the first pvo in the linked-list
 * \param name: the name of the pvo to return if found
 *
 * Returns NULL, or a pointer to the name pvo in the linked-list
 */
LWS_VISIBLE LWS_EXTERN const struct lws_protocol_vhost_options *
lws_pvo_search(const struct lws_protocol_vhost_options *pvo, const char *name);

/**
 * lws_pvo_get_str() - retreive a string pvo value
 *
 * \param pvo:	the first pvo in the linked-list
 * \param name: the name of the pvo to return if found
 * \param result: pointer to a const char * to get the result if any
 *
 * Returns 0 if found and *result set, or nonzero if not found
 */
LWS_VISIBLE LWS_EXTERN int
lws_pvo_get_str(void *in, const char *name, const char **result);

LWS_VISIBLE LWS_EXTERN int
lws_protocol_init(struct lws_context *context);

#ifdef LWS_WITH_PLUGINS

/* PLUGINS implies LIBUV */

#define LWS_PLUGIN_API_MAGIC 180

/** struct lws_plugin_capability - how a plugin introduces itself to lws */
struct lws_plugin_capability {
	unsigned int api_magic;	/**< caller fills this in, plugin fills rest */
	const struct lws_protocols *protocols; /**< array of supported protocols provided by plugin */
	int count_protocols; /**< how many protocols */
	const struct lws_extension *extensions; /**< array of extensions provided by plugin */
	int count_extensions; /**< how many extensions */
};

typedef int (*lws_plugin_init_func)(struct lws_context *,
				    struct lws_plugin_capability *);
typedef int (*lws_plugin_destroy_func)(struct lws_context *);

/** struct lws_plugin */
struct lws_plugin {
	struct lws_plugin *list; /**< linked list */
#if (UV_VERSION_MAJOR > 0)
	uv_lib_t lib; /**< shared library pointer */
#else
	void *l; /**< so we can compile on ancient libuv */
#endif
	char name[64]; /**< name of the plugin */
	struct lws_plugin_capability caps; /**< plugin capabilities */
};

#endif

///@}
