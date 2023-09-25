/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2001-2007, by Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2008-2012, by Randall Stewart. All rights reserved.
 * Copyright (c) 2008-2012, by Michael Tuexen. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a) Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * b) Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the distribution.
 *
 * c) Neither the name of Cisco Systems, Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(__FreeBSD__) && !defined(__Userspace__)
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");
#endif


#ifndef __NETINET_SCTP_SHA1_H__
#define __NETINET_SCTP_SHA1_H__

#include <sys/types.h>
#if defined(SCTP_USE_NSS_SHA1)
#include <pk11pub.h>
#elif defined(SCTP_USE_OPENSSL_SHA1)
#include <openssl/sha.h>
#elif defined(SCTP_USE_MBEDTLS_SHA1)
#include <mbedtls/sha1.h>
#endif

struct sctp_sha1_context {
#if defined(SCTP_USE_NSS_SHA1)
	struct PK11Context *pk11_ctx;
#elif defined(SCTP_USE_OPENSSL_SHA1)
	SHA_CTX sha_ctx;
#elif defined(SCTP_USE_MBEDTLS_SHA1)
	mbedtls_sha1_context sha1_ctx;
#else
	unsigned int A;
	unsigned int B;
	unsigned int C;
	unsigned int D;
	unsigned int E;
	unsigned int H0;
	unsigned int H1;
	unsigned int H2;
	unsigned int H3;
	unsigned int H4;
	unsigned int words[80];
	unsigned int TEMP;
	/* block I am collecting to process */
	char sha_block[64];
	/* collected so far */
	int how_many_in_block;
	unsigned int running_total;
#endif
};

#if (defined(__APPLE__)  && !defined(__Userspace__) && defined(KERNEL))
#ifndef _KERNEL
#define _KERNEL
#endif
#endif

#if defined(_KERNEL) || defined(__Userspace__)

void sctp_sha1_init(struct sctp_sha1_context *);
void sctp_sha1_update(struct sctp_sha1_context *, const unsigned char *, unsigned int);
void sctp_sha1_final(unsigned char *, struct sctp_sha1_context *);

#endif
#endif
