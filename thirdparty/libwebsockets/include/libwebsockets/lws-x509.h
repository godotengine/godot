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

enum lws_tls_cert_info {
	LWS_TLS_CERT_INFO_VALIDITY_FROM,
	/**< fills .time with the time_t the cert validity started from */
	LWS_TLS_CERT_INFO_VALIDITY_TO,
	/**< fills .time with the time_t the cert validity ends at */
	LWS_TLS_CERT_INFO_COMMON_NAME,
	/**< fills up to len bytes of .ns.name with the cert common name */
	LWS_TLS_CERT_INFO_ISSUER_NAME,
	/**< fills up to len bytes of .ns.name with the cert issuer name */
	LWS_TLS_CERT_INFO_USAGE,
	/**< fills verified with a bitfield asserting the valid uses */
	LWS_TLS_CERT_INFO_VERIFIED,
	/**< fills .verified with a bool representing peer cert validity,
	 *   call returns -1 if no cert */
	LWS_TLS_CERT_INFO_OPAQUE_PUBLIC_KEY,
	/**< the certificate's public key, as an opaque bytestream.  These
	 * opaque bytestreams can only be compared with each other using the
	 * same tls backend, ie, OpenSSL or mbedTLS.  The different backends
	 * produce different, incompatible representations for the same cert.
	 */
};

union lws_tls_cert_info_results {
	unsigned int verified;
	time_t time;
	unsigned int usage;
	struct {
		int len;
		/* KEEP LAST... notice the [64] is only there because
		 * name[] is not allowed in a union.  The actual length of
		 * name[] is arbitrary and is passed into the api using the
		 * len parameter.  Eg
		 *
		 * char big[1024];
		 * union lws_tls_cert_info_results *buf =
		 * 	(union lws_tls_cert_info_results *)big;
		 *
		 * lws_tls_peer_cert_info(wsi, type, buf, sizeof(big) -
		 *			  sizeof(*buf) + sizeof(buf->ns.name));
		 */
		char name[64];
	} ns;
};

/**
 * lws_tls_peer_cert_info() - get information from the peer's TLS cert
 *
 * \param wsi: the connection to query
 * \param type: one of LWS_TLS_CERT_INFO_
 * \param buf: pointer to union to take result
 * \param len: when result is a string, the true length of buf->ns.name[]
 *
 * lws_tls_peer_cert_info() lets you get hold of information from the peer
 * certificate.
 *
 * Return 0 if there is a result in \p buf, or -1 indicating there was no cert
 * or another problem.
 *
 * This function works the same no matter if the TLS backend is OpenSSL or
 * mbedTLS.
 */
LWS_VISIBLE LWS_EXTERN int
lws_tls_peer_cert_info(struct lws *wsi, enum lws_tls_cert_info type,
		       union lws_tls_cert_info_results *buf, size_t len);

/**
 * lws_tls_vhost_cert_info() - get information from the vhost's own TLS cert
 *
 * \param vhost: the vhost to query
 * \param type: one of LWS_TLS_CERT_INFO_
 * \param buf: pointer to union to take result
 * \param len: when result is a string, the true length of buf->ns.name[]
 *
 * lws_tls_vhost_cert_info() lets you get hold of information from the vhost
 * certificate.
 *
 * Return 0 if there is a result in \p buf, or -1 indicating there was no cert
 * or another problem.
 *
 * This function works the same no matter if the TLS backend is OpenSSL or
 * mbedTLS.
 */
LWS_VISIBLE LWS_EXTERN int
lws_tls_vhost_cert_info(struct lws_vhost *vhost, enum lws_tls_cert_info type,
		        union lws_tls_cert_info_results *buf, size_t len);

/**
 * lws_tls_acme_sni_cert_create() - creates a temp selfsigned cert
 *				    and attaches to a vhost
 *
 * \param vhost: the vhost to acquire the selfsigned cert
 * \param san_a: SAN written into the certificate
 * \param san_b: second SAN written into the certificate
 *
 *
 * Returns 0 if created and attached to the vhost.  Returns -1 if problems and
 * frees all allocations before returning.
 *
 * On success, any allocations are destroyed at vhost destruction automatically.
 */
LWS_VISIBLE LWS_EXTERN int
lws_tls_acme_sni_cert_create(struct lws_vhost *vhost, const char *san_a,
			     const char *san_b);

/**
 * lws_tls_acme_sni_csr_create() - creates a CSR and related private key PEM
 *
 * \param context: lws_context used for random
 * \param elements: array of LWS_TLS_REQ_ELEMENT_COUNT const char *
 * \param csr: buffer that will get the b64URL(ASN-1 CSR)
 * \param csr_len: max length of the csr buffer
 * \param privkey_pem: pointer to pointer allocated to hold the privkey_pem
 * \param privkey_len: pointer to size_t set to the length of the privkey_pem
 *
 * Creates a CSR according to the information in \p elements, and a private
 * RSA key used to sign the CSR.
 *
 * The outputs are the b64URL(ASN-1 CSR) into csr, and the PEM private key into
 * privkey_pem.
 *
 * Notice that \p elements points to an array of const char *s pointing to the
 * information listed in the enum above.  If an entry is NULL or an empty
 * string, the element is set to "none" in the CSR.
 *
 * Returns 0 on success or nonzero for failure.
 */
LWS_VISIBLE LWS_EXTERN int
lws_tls_acme_sni_csr_create(struct lws_context *context, const char *elements[],
			    uint8_t *csr, size_t csr_len, char **privkey_pem,
			    size_t *privkey_len);

/**
 * lws_tls_cert_updated() - update every vhost using the given cert path
 *
 * \param context: our lws_context
 * \param certpath: the filepath to the certificate
 * \param keypath: the filepath to the private key of the certificate
 * \param mem_cert: copy of the cert in memory
 * \param len_mem_cert: length of the copy of the cert in memory
 * \param mem_privkey: copy of the private key in memory
 * \param len_mem_privkey: length of the copy of the private key in memory
 *
 * Checks every vhost to see if it is the using certificate described by the
 * the given filepaths.  If so, it attempts to update the vhost ssl_ctx to use
 * the new certificate.
 *
 * Returns 0 on success or nonzero for failure.
 */
LWS_VISIBLE LWS_EXTERN int
lws_tls_cert_updated(struct lws_context *context, const char *certpath,
		     const char *keypath,
		     const char *mem_cert, size_t len_mem_cert,
		     const char *mem_privkey, size_t len_mem_privkey);

