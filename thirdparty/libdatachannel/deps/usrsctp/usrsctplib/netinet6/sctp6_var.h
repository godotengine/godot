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
 *    this list of conditions and the following disclaimer.
 *
 * b) Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the distribution.
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

#ifndef _NETINET6_SCTP6_VAR_H_
#define _NETINET6_SCTP6_VAR_H_

#if defined(__Userspace__)
#ifdef INET
extern void in6_sin6_2_sin(struct sockaddr_in *, struct sockaddr_in6 *);
extern void in6_sin6_2_sin_in_sock(struct sockaddr *);
extern void in6_sin_2_v4mapsin6(const struct sockaddr_in *, struct sockaddr_in6 *);
#endif
#endif
#if defined(_KERNEL)

#if !defined(__Userspace__)
SYSCTL_DECL(_net_inet6_sctp6);
#if defined(__FreeBSD__)
extern struct protosw sctp6_seqpacket_protosw, sctp6_stream_protosw;
#else
extern struct pr_usrreqs sctp6_usrreqs;
#endif
#else
int sctp6_usrreq(struct socket *, int, struct mbuf *, struct mbuf *, struct mbuf *);
#endif

#if defined(__APPLE__) && !defined(__Userspace__)
int sctp6_input(struct mbuf **, int *);
int sctp6_input_with_port(struct mbuf **, int *, uint16_t);
#else
int sctp6_input(struct mbuf **, int *, int);
int sctp6_input_with_port(struct mbuf **, int *, uint16_t);
#endif
int sctp6_output(struct sctp_inpcb *, struct mbuf *, struct sockaddr *,
                 struct mbuf *, struct proc *);
#if defined(__APPLE__) && !defined(__Userspace__) && !defined(APPLE_LEOPARD) && !defined(APPLE_SNOWLEOPARD) && !defined(APPLE_LION) && !defined(APPLE_MOUNTAINLION) && !defined(APPLE_ELCAPITAN)
void sctp6_ctlinput(int, struct sockaddr *, void *, struct ifnet * SCTP_UNUSED);
#elif defined(__FreeBSD__) && !defined(__Userspace__)
ip6proto_ctlinput_t sctp6_ctlinput;
#else
void sctp6_ctlinput(int, struct sockaddr_in6 *, ip6ctlparam *);
#endif
#if !((defined(__FreeBSD__) || defined(__APPLE__)) && !defined(__Userspace__))
extern void in6_sin_2_v4mapsin6(struct sockaddr_in *, struct sockaddr_in6 *);
extern void in6_sin6_2_sin(struct sockaddr_in *, struct sockaddr_in6 *);
extern void in6_sin6_2_sin_in_sock(struct sockaddr *);
#endif
void sctp6_notify(struct sctp_inpcb *, struct sctp_tcb *, struct sctp_nets *,
                  uint8_t, uint8_t, uint32_t);
#endif
#endif
