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


/*! \defgroup jwk JSON Web Keys
 * ## JSON Web Keys API
 *
 * Lws provides an API to parse JSON Web Keys into a struct lws_genrsa_elements.
 *
 * "oct" and "RSA" type keys are supported.  For "oct" keys, they are held in
 * the "e" member of the struct lws_genrsa_elements.
 *
 * Keys elements are allocated on the heap.  You must destroy the allocations
 * in the struct lws_genrsa_elements by calling
 * lws_jwk_destroy_genrsa_elements() when you are finished with it.
 */
///@{

struct lws_jwk {
	char keytype[5];		/**< "oct" or "RSA" */
	struct lws_genrsa_elements el;	/**< OCTet key is in el.e */
};

/** lws_jwk_import() - Create a JSON Web key from the textual representation
 *
 * \param s: the JWK object to create
 * \param in: a single JWK JSON stanza in utf-8
 * \param len: the length of the JWK JSON stanza in bytes
 *
 * Creates an lws_jwk struct filled with data from the JSON representation.
 * "oct" and "rsa" key types are supported.
 *
 * For "oct" type keys, it is loaded into el.e.
 */
LWS_VISIBLE LWS_EXTERN int
lws_jwk_import(struct lws_jwk *s, const char *in, size_t len);

/** lws_jwk_destroy() - Destroy a JSON Web key
 *
 * \param s: the JWK object to destroy
 *
 * All allocations in the lws_jwk are destroyed
 */
LWS_VISIBLE LWS_EXTERN void
lws_jwk_destroy(struct lws_jwk *s);

/** lws_jwk_export() - Export a JSON Web key to a textual representation
 *
 * \param s: the JWK object to export
 * \param _private: 0 = just export public parts, 1 = export everything
 * \param p: the buffer to write the exported JWK to
 * \param len: the length of the buffer \p p in bytes
 *
 * Returns length of the used part of the buffer if OK, or -1 for error.
 *
 * Serializes the content of the JWK into a char buffer.
 */
LWS_VISIBLE LWS_EXTERN int
lws_jwk_export(struct lws_jwk *s, int _private, char *p, size_t len);

/** lws_jwk_load() - Import a JSON Web key from a file
 *
 * \param s: the JWK object to load into
 * \param filename: filename to load from
 *
 * Returns 0 for OK or -1 for failure
 */
LWS_VISIBLE int
lws_jwk_load(struct lws_jwk *s, const char *filename);

/** lws_jwk_save() - Export a JSON Web key to a file
 *
 * \param s: the JWK object to save from
 * \param filename: filename to save to
 *
 * Returns 0 for OK or -1 for failure
 */
LWS_VISIBLE int
lws_jwk_save(struct lws_jwk *s, const char *filename);

/** lws_jwk_rfc7638_fingerprint() - jwk to RFC7638 compliant fingerprint
 *
 * \param s: the JWK object to fingerprint
 * \param digest32: buffer to take 32-byte digest
 *
 * Returns 0 for OK or -1 for failure
 */
LWS_VISIBLE int
lws_jwk_rfc7638_fingerprint(struct lws_jwk *s, char *digest32);
///@}
