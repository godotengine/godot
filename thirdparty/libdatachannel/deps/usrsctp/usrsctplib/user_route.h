/*-
 * Copyright (c) 1980, 1986, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#ifndef _USER_ROUTE_H_
#define _USER_ROUTE_H_

/*
 * Kernel resident routing tables.
 *
 * The routing tables are initialized when interface addresses
 * are set by making entries for all directly connected interfaces.
 */

/*
 * A route consists of a destination address and a reference
 * to a routing entry.  These are often held by protocols
 * in their control blocks, e.g. inpcb.
 */

struct sctp_route {
	struct	sctp_rtentry *ro_rt;
	struct	sockaddr ro_dst;
};

/*
 * These numbers are used by reliable protocols for determining
 * retransmission behavior and are included in the routing structure.
 */
struct sctp_rt_metrics_lite {
	uint32_t rmx_mtu;	/* MTU for this path */
#if 0
	u_long	rmx_expire;	/* lifetime for route, e.g. redirect */
	u_long	rmx_pksent;	/* packets sent using this route */
#endif
};

/*
 * We distinguish between routes to hosts and routes to networks,
 * preferring the former if available.  For each route we infer
 * the interface to use from the gateway address supplied when
 * the route was entered.  Routes that forward packets through
 * gateways are marked so that the output routines know to address the
 * gateway rather than the ultimate destination.
 */
struct sctp_rtentry {
#if 0
	struct	radix_node rt_nodes[2];	/* tree glue, and other values */
	/*
	 * XXX struct rtentry must begin with a struct radix_node (or two!)
	 * because the code does some casts of a 'struct radix_node *'
	 * to a 'struct rtentry *'
	 */
#define	rt_key(r)	(*((struct sockaddr **)(&(r)->rt_nodes->rn_key)))
#define	rt_mask(r)	(*((struct sockaddr **)(&(r)->rt_nodes->rn_mask)))
	struct	sockaddr *rt_gateway;	/* value */
	u_long	rt_flags;		/* up/down?, host/net */
#endif
	struct	ifnet *rt_ifp;		/* the answer: interface to use */
	struct	ifaddr *rt_ifa;		/* the answer: interface address to use */
	struct	sctp_rt_metrics_lite rt_rmx;	/* metrics used by rx'ing protocols */
	long	rt_refcnt;		/* # held references */
#if 0
	struct	sockaddr *rt_genmask;	/* for generation of cloned routes */
	caddr_t	rt_llinfo;		/* pointer to link level info cache */
	struct	rtentry *rt_gwroute;	/* implied entry for gatewayed routes */
	struct	rtentry *rt_parent; 	/* cloning parent of this route */
#endif
	struct	mtx rt_mtx;		/* mutex for routing entry */
};

#define	RT_LOCK_INIT(_rt)	mtx_init(&(_rt)->rt_mtx, "rtentry", NULL, MTX_DEF | MTX_DUPOK)
#define	RT_LOCK(_rt)		mtx_lock(&(_rt)->rt_mtx)
#define	RT_UNLOCK(_rt)		mtx_unlock(&(_rt)->rt_mtx)
#define	RT_LOCK_DESTROY(_rt)	mtx_destroy(&(_rt)->rt_mtx)
#define	RT_LOCK_ASSERT(_rt)	mtx_assert(&(_rt)->rt_mtx, MA_OWNED)

#define	RT_ADDREF(_rt)	do {					\
	RT_LOCK_ASSERT(_rt);					\
	KASSERT((_rt)->rt_refcnt >= 0,				\
		("negative refcnt %ld", (_rt)->rt_refcnt));	\
	(_rt)->rt_refcnt++;					\
} while (0)
#define	RT_REMREF(_rt)	do {					\
	RT_LOCK_ASSERT(_rt);					\
	KASSERT((_rt)->rt_refcnt > 0,				\
		("bogus refcnt %ld", (_rt)->rt_refcnt));	\
	(_rt)->rt_refcnt--;					\
} while (0)
#define	RTFREE_LOCKED(_rt) do {					\
		if ((_rt)->rt_refcnt <= 1) {			\
			rtfree(_rt);				\
		} else {					\
			RT_REMREF(_rt);				\
			RT_UNLOCK(_rt);				\
		}						\
		/* guard against invalid refs */		\
		_rt = NULL;					\
	} while (0)
#define	RTFREE(_rt) do {					\
		RT_LOCK(_rt);					\
		RTFREE_LOCKED(_rt);				\
} while (0)
#endif
