/*-
 * Copyright (c) 1982, 1986, 1988, 1993
 *      The Regents of the University of California.
 * All rights reserved.
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

#ifndef _USER_MBUF_H_
#define _USER_MBUF_H_

/* __Userspace__ header file for mbufs */
#include <stdio.h>
#if !defined(SCTP_SIMPLE_ALLOCATOR)
#include "umem.h"
#endif
#include "user_malloc.h"
#include "netinet/sctp_os_userspace.h"

#define USING_MBUF_CONSTRUCTOR 0

/* For Linux */
#ifndef MSIZE
#define MSIZE 256
/* #define MSIZE 1024 */
#endif
#ifndef MCLBYTES
#define MCLBYTES 2048
#endif

struct mbuf * m_gethdr(int how, short type);
struct mbuf * m_get(int how, short type);
struct mbuf * m_free(struct mbuf *m);
void m_clget(struct mbuf *m, int how);
struct mbuf * m_getm2(struct mbuf *m, int len, int how, short type, int flags, int allonebuf);
struct mbuf *m_uiotombuf(struct uio *uio, int how, int len, int align, int flags);
u_int m_length(struct mbuf *m0, struct mbuf **last);
struct mbuf *m_last(struct mbuf *m);

/* mbuf initialization function */
void mbuf_initialize(void *);

#define	M_MOVE_PKTHDR(to, from)	m_move_pkthdr((to), (from))
#define	MGET(m, how, type)	((m) = m_get((how), (type)))
#define	MGETHDR(m, how, type)	((m) = m_gethdr((how), (type)))
#define	MCLGET(m, how)		m_clget((m), (how))


#define M_HDR_PAD ((sizeof(intptr_t)==4) ? 2 : 6) /* modified for __Userspace__ */

/* Length to m_copy to copy all. */
#define	M_COPYALL	1000000000

/* umem_cache_t is defined in user_include/umem.h as
 * typedef struct umem_cache umem_cache_t;
 * Note:umem_zone_t is a pointer.
 */
#if defined(SCTP_SIMPLE_ALLOCATOR)
typedef size_t sctp_zone_t;
#else
typedef umem_cache_t *sctp_zone_t;
#endif

extern sctp_zone_t zone_mbuf;
extern sctp_zone_t zone_clust;
extern sctp_zone_t zone_ext_refcnt;

/*-
 * Macros for type conversion:
 * mtod(m, t)	-- Convert mbuf pointer to data pointer of correct type.
 * dtom(x)	-- Convert data pointer within mbuf to mbuf pointer (XXX).
 */
#define	mtod(m, t)	((t)((m)->m_data))
#define	dtom(x)		((struct mbuf *)((intptr_t)(x) & ~(MSIZE-1)))

struct mb_args {
	int	flags;	/* Flags for mbuf being allocated */
	short	type;	/* Type of mbuf being allocated */
};

struct clust_args {
	struct mbuf * parent_mbuf;
};

struct mbuf *    m_split(struct mbuf *, int, int);
void             m_cat(struct mbuf *m, struct mbuf *n);
void		 m_adj(struct mbuf *, int);
void  mb_free_ext(struct mbuf *);
void  m_freem(struct mbuf *);
struct m_tag	*m_tag_alloc(uint32_t, int, int, int);
struct mbuf	*m_copym(struct mbuf *, int, int, int);
void		 m_copyback(struct mbuf *, int, int, caddr_t);
int		 m_apply(struct mbuf *, int, int, int (*)(void *, void *, u_int), void *arg);
struct mbuf	*m_pullup(struct mbuf *, int);
struct mbuf	*m_pulldown(struct mbuf *, int off, int len, int *offp);
int		 m_dup_pkthdr(struct mbuf *, struct mbuf *, int);
struct m_tag	*m_tag_copy(struct m_tag *, int);
int		 m_tag_copy_chain(struct mbuf *, struct mbuf *, int);
struct mbuf	*m_prepend(struct mbuf *, int, int);
void		 m_copydata(const struct mbuf *, int, int, caddr_t);

#define MBUF_MEM_NAME "mbuf"
#define MBUF_CLUSTER_MEM_NAME "mbuf_cluster"
#define	MBUF_EXTREFCNT_MEM_NAME	"mbuf_ext_refcnt"

/*
 * Mbufs are of a single size, MSIZE (sys/param.h), which includes overhead.
 * An mbuf may add a single "mbuf cluster" of size MCLBYTES (also in
 * sys/param.h), which has no additional overhead and is used instead of the
 * internal data area; this is done when at least MINCLSIZE of data must be
 * stored.  Additionally, it is possible to allocate a separate buffer
 * externally and attach it to the mbuf in a way similar to that of mbuf
 * clusters.
 */
#define	MLEN		((int)(MSIZE - sizeof(struct m_hdr)))	/* normal data len */
#define	MHLEN		((int)(MLEN - sizeof(struct pkthdr)))	/* data len w/pkthdr */
#define	MINCLSIZE	((int)(MHLEN + 1))	/* smallest amount to put in cluster */
#define	M_MAXCOMPRESS	(MHLEN / 2)	/* max amount to copy for compression */


/*
 * Header present at the beginning of every mbuf.
 */
struct m_hdr {
	struct mbuf	*mh_next;	/* next buffer in chain */
	struct mbuf	*mh_nextpkt;	/* next chain in queue/record */
	caddr_t		 mh_data;	/* location of data */
	int		 mh_len;	/* amount of data in this mbuf */
	int		 mh_flags;	/* flags; see below */
	short		 mh_type;	/* type of data in this mbuf */
	uint8_t          pad[M_HDR_PAD];/* word align                  */
};

/*
 * Packet tag structure (see below for details).
 */
struct m_tag {
	SLIST_ENTRY(m_tag)	m_tag_link;	/* List of packet tags */
	uint16_t		m_tag_id;	/* Tag ID */
	uint16_t		m_tag_len;	/* Length of data */
	uint32_t		m_tag_cookie;	/* ABI/Module ID */
	void			(*m_tag_free)(struct m_tag *);
};

/*
 * Record/packet header in first mbuf of chain; valid only if M_PKTHDR is set.
 */
struct pkthdr {
	struct ifnet	*rcvif;		/* rcv interface */
	/* variables for ip and tcp reassembly */
	void		*header;	/* pointer to packet header */
	int		 len;		/* total packet length */
	/* variables for hardware checksum */
	int		 csum_flags;	/* flags regarding checksum */
	int		 csum_data;	/* data field used by csum routines */
	uint16_t	 tso_segsz;	/* TSO segment size */
	uint16_t	 ether_vtag;	/* Ethernet 802.1p+q vlan tag */
	SLIST_HEAD(packet_tags, m_tag) tags; /* list of packet tags */
};

/*
 * Description of external storage mapped into mbuf; valid only if M_EXT is
 * set.
 */
struct m_ext {
	caddr_t		 ext_buf;	/* start of buffer */
	void		(*ext_free)	/* free routine if not the usual */
			    (void *, void *);
	void		*ext_args;	/* optional argument pointer */
	u_int		 ext_size;	/* size of buffer, for ext_free */
	volatile u_int	*ref_cnt;	/* pointer to ref count info */
	int		 ext_type;	/* type of external storage */
};


/*
 * The core of the mbuf object along with some shortcut defined for practical
 * purposes.
 */
struct mbuf {
	struct m_hdr	m_hdr;
	union {
		struct {
			struct pkthdr	MH_pkthdr;	/* M_PKTHDR set */
			union {
				struct m_ext	MH_ext;	/* M_EXT set */
				char		MH_databuf[MHLEN];
			} MH_dat;
		} MH;
		char	M_databuf[MLEN];		/* !M_PKTHDR, !M_EXT */
	} M_dat;
};

#define	m_next		m_hdr.mh_next
#define	m_len		m_hdr.mh_len
#define	m_data		m_hdr.mh_data
#define	m_type		m_hdr.mh_type
#define	m_flags		m_hdr.mh_flags
#define	m_nextpkt	m_hdr.mh_nextpkt
#define	m_act		m_nextpkt
#define	m_pkthdr	M_dat.MH.MH_pkthdr
#define	m_ext		M_dat.MH.MH_dat.MH_ext
#define	m_pktdat	M_dat.MH.MH_dat.MH_databuf
#define	m_dat		M_dat.M_databuf


/*
 * mbuf flags.
 */
#define	M_EXT		0x0001	/* has associated external storage */
#define	M_PKTHDR	0x0002	/* start of record */
#define	M_EOR		0x0004	/* end of record */
#define	M_RDONLY	0x0008	/* associated data is marked read-only */
#define	M_PROTO1	0x0010	/* protocol-specific */
#define	M_PROTO2	0x0020	/* protocol-specific */
#define	M_PROTO3	0x0040	/* protocol-specific */
#define	M_PROTO4	0x0080	/* protocol-specific */
#define	M_PROTO5	0x0100	/* protocol-specific */
#define	M_FREELIST	0x8000	/* mbuf is on the free list */


/*
 * Flags copied when copying m_pkthdr.
 */
#define	M_COPYFLAGS	(M_PKTHDR|M_EOR|M_RDONLY|M_PROTO1|M_PROTO1|M_PROTO2|\
			    M_PROTO3|M_PROTO4|M_PROTO5|\
			    M_BCAST|M_MCAST|M_FRAG|M_FIRSTFRAG|M_LASTFRAG|\
			    M_VLANTAG|M_PROMISC)


/*
 * mbuf pkthdr flags (also stored in m_flags).
 */
#define	M_BCAST		0x0200	/* send/received as link-level broadcast */
#define	M_MCAST		0x0400	/* send/received as link-level multicast */
#define	M_FRAG		0x0800	/* packet is a fragment of a larger packet */
#define	M_FIRSTFRAG	0x1000	/* packet is first fragment */
#define	M_LASTFRAG	0x2000	/* packet is last fragment */
#define	M_VLANTAG	0x10000	/* ether_vtag is valid */
#define	M_PROMISC	0x20000	/* packet was not for us */
#define	M_NOFREE	0x40000	/* do not free mbuf - it is embedded in the cluster */


/*
 * External buffer types: identify ext_buf type.
 */
#define	EXT_CLUSTER	1	/* mbuf cluster */
#define	EXT_SFBUF	2	/* sendfile(2)'s sf_bufs */
#define	EXT_JUMBOP	3	/* jumbo cluster 4096 bytes */
#define	EXT_JUMBO9	4	/* jumbo cluster 9216 bytes */
#define	EXT_JUMBO16	5	/* jumbo cluster 16184 bytes */
#define	EXT_PACKET	6	/* mbuf+cluster from packet zone */
#define	EXT_MBUF	7	/* external mbuf reference (M_IOVEC) */
#define	EXT_NET_DRV	100	/* custom ext_buf provided by net driver(s) */
#define	EXT_MOD_TYPE	200	/* custom module's ext_buf type */
#define	EXT_DISPOSABLE	300	/* can throw this buffer away w/page flipping */
#define	EXT_EXTREF	400	/* has externally maintained ref_cnt ptr */


/*
 * mbuf types.
 */
#define	MT_NOTMBUF	0	/* USED INTERNALLY ONLY! Object is not mbuf */
#define	MT_DATA		1	/* dynamic (data) allocation */
#define	MT_HEADER	MT_DATA	/* packet header, use M_PKTHDR instead */
#define	MT_SONAME	8	/* socket name */
#define	MT_CONTROL	14	/* extra-data protocol message */
#define	MT_OOBDATA	15	/* expedited data  */
#define	MT_NTYPES	16	/* number of mbuf types for mbtypes[] */

/*
 * __Userspace__ flags like M_NOWAIT are defined in malloc.h
 * Flags like these are used in functions like uma_zalloc()
 * but don't have an equivalent in userland umem
 * Flags specifying how an allocation should be made.
 *
 * The flag to use is as follows:
 * - M_DONTWAIT or M_NOWAIT from an interrupt handler to not block allocation.
 * - M_WAIT or M_WAITOK or M_TRYWAIT from wherever it is safe to block.
 *
 * M_DONTWAIT/M_NOWAIT means that we will not block the thread explicitly and
 * if we cannot allocate immediately we may return NULL, whereas
 * M_WAIT/M_WAITOK/M_TRYWAIT means that if we cannot allocate resources we
 * will block until they are available, and thus never return NULL.
 *
 * XXX Eventually just phase this out to use M_WAITOK/M_NOWAIT.
 */
#define	MBTOM(how)	(how)

void		 m_tag_delete(struct mbuf *, struct m_tag *);
void		 m_tag_delete_chain(struct mbuf *, struct m_tag *);
void		 m_move_pkthdr(struct mbuf *, struct mbuf *);
void		 m_tag_free_default(struct m_tag *);

extern int max_linkhdr;    /* Largest link-level header */
extern int max_protohdr; /* Size of largest protocol layer header. See user_mbuf.c */

/*
 * Evaluate TRUE if it's safe to write to the mbuf m's data region (this can
 * be both the local data payload, or an external buffer area, depending on
 * whether M_EXT is set).
 */
#define	M_WRITABLE(m)	(!((m)->m_flags & M_RDONLY) &&			\
			 (!(((m)->m_flags & M_EXT)) ||			\
			 (*((m)->m_ext.ref_cnt) == 1)) )		\

/* Check if the supplied mbuf has a packet header, or else panic. */
#define M_ASSERTPKTHDR(m)						\
	KASSERT((m) != NULL && (m)->m_flags & M_PKTHDR,			\
	    ("%s: no mbuf packet header!", __func__))

/*
 * Compute the amount of space available before the current start of data in
 * an mbuf.
 *
 * The M_WRITABLE() is a temporary, conservative safety measure: the burden
 * of checking writability of the mbuf data area rests solely with the caller.
 */
#define	M_LEADINGSPACE(m)						\
	(((m)->m_flags & M_EXT) ?					\
	    (M_WRITABLE(m) ? (m)->m_data - (m)->m_ext.ext_buf : 0):	\
	    ((m)->m_flags & M_PKTHDR)? (m)->m_data - (m)->m_pktdat :	\
	    (m)->m_data - (m)->m_dat)

/*
 * Compute the amount of space available after the end of data in an mbuf.
 *
 * The M_WRITABLE() is a temporary, conservative safety measure: the burden
 * of checking writability of the mbuf data area rests solely with the caller.
 */
#define	M_TRAILINGSPACE(m)						\
	(((m)->m_flags & M_EXT) ?					\
	    (M_WRITABLE(m) ? (m)->m_ext.ext_buf + (m)->m_ext.ext_size	\
		- ((m)->m_data + (m)->m_len) : 0) :			\
	    &(m)->m_dat[MLEN] - ((m)->m_data + (m)->m_len))



/*
 * Arrange to prepend space of size plen to mbuf m.  If a new mbuf must be
 * allocated, how specifies whether to wait.  If the allocation fails, the
 * original mbuf chain is freed and m is set to NULL.
 */
#define	M_PREPEND(m, plen, how) do {					\
	struct mbuf **_mmp = &(m);					\
	struct mbuf *_mm = *_mmp;					\
	int _mplen = (plen);						\
	int __mhow = (how);						\
									\
	if (M_LEADINGSPACE(_mm) >= _mplen) {				\
		_mm->m_data -= _mplen;					\
		_mm->m_len += _mplen;					\
	} else								\
		_mm = m_prepend(_mm, _mplen, __mhow);			\
	if (_mm != NULL && _mm->m_flags & M_PKTHDR)			\
		_mm->m_pkthdr.len += _mplen;				\
	*_mmp = _mm;							\
} while (0)

/*
 * Set the m_data pointer of a newly-allocated mbuf (m_get/MGET) to place an
 * object of the specified size at the end of the mbuf, longword aligned.
 */
#define	M_ALIGN(m, len) do {						\
        KASSERT(!((m)->m_flags & (M_PKTHDR|M_EXT)),                     \
                ("%s: M_ALIGN not normal mbuf", __func__));             \
        KASSERT((m)->m_data == (m)->m_dat,                              \
                ("%s: M_ALIGN not a virgin mbuf", __func__));           \
	(m)->m_data += (MLEN - (len)) & ~(sizeof(long) - 1);		\
} while (0)

/*
 * As above, for mbufs allocated with m_gethdr/MGETHDR or initialized by
 * M_DUP/MOVE_PKTHDR.
 */
#define	MH_ALIGN(m, len) do {						\
        KASSERT((m)->m_flags & M_PKTHDR && !((m)->m_flags & M_EXT),     \
                ("%s: MH_ALIGN not PKTHDR mbuf", __func__));            \
        KASSERT((m)->m_data == (m)->m_pktdat,                           \
                ("%s: MH_ALIGN not a virgin mbuf", __func__));          \
	(m)->m_data += (MHLEN - (len)) & ~(sizeof(long) - 1);		\
} while (0)

#define M_SIZE(m)						\
		(((m)->m_flags & M_EXT) ? (m)->m_ext.ext_size :	\
		((m)->m_flags & M_PKTHDR) ? MHLEN :		\
		MLEN)

#endif
