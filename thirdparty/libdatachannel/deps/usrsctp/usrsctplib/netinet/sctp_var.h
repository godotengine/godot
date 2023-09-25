/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2001-2008, by Cisco Systems, Inc. All rights reserved.
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

#ifndef _NETINET_SCTP_VAR_H_
#define _NETINET_SCTP_VAR_H_

#include <netinet/sctp_uio.h>

#if defined(_KERNEL) || defined(__Userspace__)

#if !defined(__Userspace__)
#if defined(__FreeBSD__)
extern struct protosw sctp_seqpacket_protosw, sctp_stream_protosw;
#else
extern struct pr_usrreqs sctp_usrreqs;
#endif
#endif

#define sctp_feature_on(inp, feature)  (inp->sctp_features |= feature)
#define sctp_feature_off(inp, feature) (inp->sctp_features &= ~feature)
#define sctp_is_feature_on(inp, feature) ((inp->sctp_features & feature) == feature)
#define sctp_is_feature_off(inp, feature) ((inp->sctp_features & feature) == 0)

#define sctp_stcb_feature_on(inp, stcb, feature) {\
	if (stcb) { \
		stcb->asoc.sctp_features |= feature; \
	} else if (inp) { \
		inp->sctp_features |= feature; \
	} \
}
#define sctp_stcb_feature_off(inp, stcb, feature) {\
	if (stcb) { \
		stcb->asoc.sctp_features &= ~feature; \
	} else if (inp) { \
		inp->sctp_features &= ~feature; \
	} \
}
#define sctp_stcb_is_feature_on(inp, stcb, feature) \
	(((stcb != NULL) && \
	  ((stcb->asoc.sctp_features & feature) == feature)) || \
	 ((stcb == NULL) && (inp != NULL) && \
	  ((inp->sctp_features & feature) == feature)))
#define sctp_stcb_is_feature_off(inp, stcb, feature) \
	(((stcb != NULL) && \
	  ((stcb->asoc.sctp_features & feature) == 0)) || \
	 ((stcb == NULL) && (inp != NULL) && \
	  ((inp->sctp_features & feature) == 0)) || \
	 ((stcb == NULL) && (inp == NULL)))

/* managing mobility_feature in inpcb (by micchie) */
#define sctp_mobility_feature_on(inp, feature)  (inp->sctp_mobility_features |= feature)
#define sctp_mobility_feature_off(inp, feature) (inp->sctp_mobility_features &= ~feature)
#define sctp_is_mobility_feature_on(inp, feature) (inp->sctp_mobility_features & feature)
#define sctp_is_mobility_feature_off(inp, feature) ((inp->sctp_mobility_features & feature) == 0)

#define sctp_maxspace(sb) (max((sb)->sb_hiwat,SCTP_MINIMAL_RWND))

#define	sctp_sbspace(asoc, sb) ((long) ((sctp_maxspace(sb) > (asoc)->sb_cc) ? (sctp_maxspace(sb) - (asoc)->sb_cc) : 0))

#define	sctp_sbspace_failedmsgs(sb) ((long) ((sctp_maxspace(sb) > SCTP_SBAVAIL(sb)) ? (sctp_maxspace(sb) - SCTP_SBAVAIL(sb)) : 0))

#define sctp_sbspace_sub(a,b) (((a) > (b)) ? ((a) - (b)) : 0)

/*
 * I tried to cache the readq entries at one point. But the reality
 * is that it did not add any performance since this meant we had to
 * lock the STCB on read. And at that point once you have to do an
 * extra lock, it really does not matter if the lock is in the ZONE
 * stuff or in our code. Note that this same problem would occur with
 * an mbuf cache as well so it is not really worth doing, at least
 * right now :-D
 */
#ifdef INVARIANTS
#define sctp_free_a_readq(_stcb, _readq) { \
	if ((_readq)->on_strm_q) \
		panic("On strm q stcb:%p readq:%p", (_stcb), (_readq)); \
	SCTP_ZONE_FREE(SCTP_BASE_INFO(ipi_zone_readq), (_readq)); \
	SCTP_DECR_READQ_COUNT(); \
}
#else
#define sctp_free_a_readq(_stcb, _readq) { \
	SCTP_ZONE_FREE(SCTP_BASE_INFO(ipi_zone_readq), (_readq)); \
	SCTP_DECR_READQ_COUNT(); \
}
#endif

#define sctp_alloc_a_readq(_stcb, _readq) { \
	(_readq) = SCTP_ZONE_GET(SCTP_BASE_INFO(ipi_zone_readq), struct sctp_queued_to_read); \
	if ((_readq)) { \
	     SCTP_INCR_READQ_COUNT(); \
	} \
}

#define sctp_free_a_strmoq(_stcb, _strmoq, _so_locked) { \
	if ((_strmoq)->holds_key_ref) { \
		sctp_auth_key_release(stcb, sp->auth_keyid, _so_locked); \
		(_strmoq)->holds_key_ref = 0; \
	} \
	SCTP_ZONE_FREE(SCTP_BASE_INFO(ipi_zone_strmoq), (_strmoq)); \
	SCTP_DECR_STRMOQ_COUNT(); \
}

#define sctp_alloc_a_strmoq(_stcb, _strmoq) { \
	(_strmoq) = SCTP_ZONE_GET(SCTP_BASE_INFO(ipi_zone_strmoq), struct sctp_stream_queue_pending); \
	if ((_strmoq)) { \
		memset(_strmoq, 0, sizeof(struct sctp_stream_queue_pending)); \
		SCTP_INCR_STRMOQ_COUNT(); \
		(_strmoq)->holds_key_ref = 0; \
	} \
}

#define sctp_free_a_chunk(_stcb, _chk, _so_locked) { \
	if ((_chk)->holds_key_ref) {\
		sctp_auth_key_release((_stcb), (_chk)->auth_keyid, _so_locked); \
		(_chk)->holds_key_ref = 0; \
	} \
	if (_stcb) { \
		SCTP_TCB_LOCK_ASSERT((_stcb)); \
		if ((_chk)->whoTo) { \
			sctp_free_remote_addr((_chk)->whoTo); \
			(_chk)->whoTo = NULL; \
		} \
		if (((_stcb)->asoc.free_chunk_cnt > SCTP_BASE_SYSCTL(sctp_asoc_free_resc_limit)) || \
		    (SCTP_BASE_INFO(ipi_free_chunks) > SCTP_BASE_SYSCTL(sctp_system_free_resc_limit))) { \
			SCTP_ZONE_FREE(SCTP_BASE_INFO(ipi_zone_chunk), (_chk)); \
			SCTP_DECR_CHK_COUNT(); \
		} else { \
			TAILQ_INSERT_TAIL(&(_stcb)->asoc.free_chunks, (_chk), sctp_next); \
			(_stcb)->asoc.free_chunk_cnt++; \
			atomic_add_int(&SCTP_BASE_INFO(ipi_free_chunks), 1); \
		} \
	} else { \
		SCTP_ZONE_FREE(SCTP_BASE_INFO(ipi_zone_chunk), (_chk)); \
		SCTP_DECR_CHK_COUNT(); \
	} \
}

#define sctp_alloc_a_chunk(_stcb, _chk) { \
	if (TAILQ_EMPTY(&(_stcb)->asoc.free_chunks)) { \
		(_chk) = SCTP_ZONE_GET(SCTP_BASE_INFO(ipi_zone_chunk), struct sctp_tmit_chunk); \
		if ((_chk)) { \
			SCTP_INCR_CHK_COUNT(); \
			(_chk)->whoTo = NULL; \
			(_chk)->holds_key_ref = 0; \
		} \
	} else { \
		(_chk) = TAILQ_FIRST(&(_stcb)->asoc.free_chunks); \
		TAILQ_REMOVE(&(_stcb)->asoc.free_chunks, (_chk), sctp_next); \
		atomic_subtract_int(&SCTP_BASE_INFO(ipi_free_chunks), 1); \
		(_chk)->holds_key_ref = 0; \
		SCTP_STAT_INCR(sctps_cached_chk); \
		(_stcb)->asoc.free_chunk_cnt--; \
	} \
}

#if defined(__FreeBSD__) && !defined(__Userspace__)
#define sctp_free_remote_addr(__net) { \
	if ((__net)) {  \
		if (SCTP_DECREMENT_AND_CHECK_REFCOUNT(&(__net)->ref_count)) { \
			RO_NHFREE(&(__net)->ro); \
			if ((__net)->src_addr_selected) { \
				sctp_free_ifa((__net)->ro._s_addr); \
				(__net)->ro._s_addr = NULL; \
			} \
			(__net)->src_addr_selected = 0; \
			(__net)->dest_state &= ~SCTP_ADDR_REACHABLE; \
			SCTP_ZONE_FREE(SCTP_BASE_INFO(ipi_zone_net), (__net)); \
			SCTP_DECR_RADDR_COUNT(); \
		} \
	} \
}

#define sctp_sbfree(ctl, stcb, sb, m) { \
	SCTP_SB_DECR(sb, SCTP_BUF_LEN((m))); \
	SCTP_SAVE_ATOMIC_DECREMENT(&(sb)->sb_mbcnt, MSIZE); \
	if (((ctl)->do_not_ref_stcb == 0) && stcb) {\
		SCTP_SAVE_ATOMIC_DECREMENT(&(stcb)->asoc.sb_cc, SCTP_BUF_LEN((m))); \
		SCTP_SAVE_ATOMIC_DECREMENT(&(stcb)->asoc.my_rwnd_control_len, MSIZE); \
	} \
	if (SCTP_BUF_TYPE(m) != MT_DATA && SCTP_BUF_TYPE(m) != MT_HEADER && \
	    SCTP_BUF_TYPE(m) != MT_OOBDATA) \
		atomic_subtract_int(&(sb)->sb_ctl,SCTP_BUF_LEN((m))); \
}

#define sctp_sballoc(stcb, sb, m) { \
	SCTP_SB_INCR(sb, SCTP_BUF_LEN((m))); \
	atomic_add_int(&(sb)->sb_mbcnt, MSIZE); \
	if (stcb) { \
		atomic_add_int(&(stcb)->asoc.sb_cc, SCTP_BUF_LEN((m))); \
		atomic_add_int(&(stcb)->asoc.my_rwnd_control_len, MSIZE); \
	} \
	if (SCTP_BUF_TYPE(m) != MT_DATA && SCTP_BUF_TYPE(m) != MT_HEADER && \
	    SCTP_BUF_TYPE(m) != MT_OOBDATA) \
		atomic_add_int(&(sb)->sb_ctl,SCTP_BUF_LEN((m))); \
}
#else				/* FreeBSD Version <= 500000 or non-FreeBSD */
#define sctp_free_remote_addr(__net) { \
	if ((__net)) { \
		if (SCTP_DECREMENT_AND_CHECK_REFCOUNT(&(__net)->ref_count)) { \
			if ((__net)->ro.ro_rt) { \
				RTFREE((__net)->ro.ro_rt); \
				(__net)->ro.ro_rt = NULL; \
			} \
			if ((__net)->src_addr_selected) { \
				sctp_free_ifa((__net)->ro._s_addr); \
				(__net)->ro._s_addr = NULL; \
			} \
			(__net)->src_addr_selected = 0; \
			(__net)->dest_state &=~SCTP_ADDR_REACHABLE; \
			SCTP_ZONE_FREE(SCTP_BASE_INFO(ipi_zone_net), (__net)); \
			SCTP_DECR_RADDR_COUNT(); \
		} \
	} \
}

#define sctp_sbfree(ctl, stcb, sb, m) { \
	SCTP_SB_DECR(sb, SCTP_BUF_LEN((m))); \
	SCTP_SAVE_ATOMIC_DECREMENT(&(sb)->sb_mbcnt, MSIZE); \
	if (((ctl)->do_not_ref_stcb == 0) && stcb) { \
		SCTP_SAVE_ATOMIC_DECREMENT(&(stcb)->asoc.sb_cc, SCTP_BUF_LEN((m))); \
		SCTP_SAVE_ATOMIC_DECREMENT(&(stcb)->asoc.my_rwnd_control_len, MSIZE); \
	} \
}

#define sctp_sballoc(stcb, sb, m) { \
	SCTP_SB_INCR(sb, SCTP_BUF_LEN((m))); \
	atomic_add_int(&(sb)->sb_mbcnt, MSIZE); \
	if (stcb) { \
		atomic_add_int(&(stcb)->asoc.sb_cc, SCTP_BUF_LEN((m))); \
		atomic_add_int(&(stcb)->asoc.my_rwnd_control_len, MSIZE); \
	} \
}
#endif

#define sctp_ucount_incr(val) { \
	val++; \
}

#define sctp_ucount_decr(val) { \
	if (val > 0) { \
		val--; \
	} else { \
		val = 0; \
	} \
}

#define sctp_mbuf_crush(data) do { \
	struct mbuf *_m; \
	_m = (data); \
	while (_m && (SCTP_BUF_LEN(_m) == 0)) { \
		(data)  = SCTP_BUF_NEXT(_m); \
		SCTP_BUF_NEXT(_m) = NULL; \
		sctp_m_free(_m); \
		_m = (data); \
	} \
} while (0)

#define sctp_flight_size_decrease(tp1) do { \
	if (tp1->whoTo->flight_size >= tp1->book_size) \
		tp1->whoTo->flight_size -= tp1->book_size; \
	else \
		tp1->whoTo->flight_size = 0; \
} while (0)

#define sctp_flight_size_increase(tp1) do { \
	(tp1)->whoTo->flight_size += (tp1)->book_size; \
} while (0)

#ifdef SCTP_FS_SPEC_LOG
#define sctp_total_flight_decrease(stcb, tp1) do { \
	if (stcb->asoc.fs_index > SCTP_FS_SPEC_LOG_SIZE) \
		stcb->asoc.fs_index = 0;\
	stcb->asoc.fslog[stcb->asoc.fs_index].total_flight = stcb->asoc.total_flight; \
	stcb->asoc.fslog[stcb->asoc.fs_index].tsn = tp1->rec.data.tsn; \
	stcb->asoc.fslog[stcb->asoc.fs_index].book = tp1->book_size; \
	stcb->asoc.fslog[stcb->asoc.fs_index].sent = tp1->sent; \
	stcb->asoc.fslog[stcb->asoc.fs_index].incr = 0; \
	stcb->asoc.fslog[stcb->asoc.fs_index].decr = 1; \
	stcb->asoc.fs_index++; \
	tp1->window_probe = 0; \
	if (stcb->asoc.total_flight >= tp1->book_size) { \
		stcb->asoc.total_flight -= tp1->book_size; \
		if (stcb->asoc.total_flight_count > 0) \
			stcb->asoc.total_flight_count--; \
	} else { \
		stcb->asoc.total_flight = 0; \
		stcb->asoc.total_flight_count = 0; \
	} \
} while (0)

#define sctp_total_flight_increase(stcb, tp1) do { \
	if (stcb->asoc.fs_index > SCTP_FS_SPEC_LOG_SIZE) \
		stcb->asoc.fs_index = 0;\
	stcb->asoc.fslog[stcb->asoc.fs_index].total_flight = stcb->asoc.total_flight; \
	stcb->asoc.fslog[stcb->asoc.fs_index].tsn = tp1->rec.data.tsn; \
	stcb->asoc.fslog[stcb->asoc.fs_index].book = tp1->book_size; \
	stcb->asoc.fslog[stcb->asoc.fs_index].sent = tp1->sent; \
	stcb->asoc.fslog[stcb->asoc.fs_index].incr = 1; \
	stcb->asoc.fslog[stcb->asoc.fs_index].decr = 0; \
	stcb->asoc.fs_index++; \
	(stcb)->asoc.total_flight_count++; \
	(stcb)->asoc.total_flight += (tp1)->book_size; \
} while (0)

#else

#define sctp_total_flight_decrease(stcb, tp1) do { \
	tp1->window_probe = 0; \
	if (stcb->asoc.total_flight >= tp1->book_size) { \
		stcb->asoc.total_flight -= tp1->book_size; \
		if (stcb->asoc.total_flight_count > 0) \
			stcb->asoc.total_flight_count--; \
	} else { \
		stcb->asoc.total_flight = 0; \
		stcb->asoc.total_flight_count = 0; \
	} \
} while (0)

#define sctp_total_flight_increase(stcb, tp1) do { \
	(stcb)->asoc.total_flight_count++; \
	(stcb)->asoc.total_flight += (tp1)->book_size; \
} while (0)

#endif

#define SCTP_PF_ENABLED(_net) (_net->pf_threshold < _net->failure_threshold)
#define SCTP_NET_IS_PF(_net) (_net->pf_threshold < _net->error_count)

struct sctp_nets;
struct sctp_inpcb;
struct sctp_tcb;
struct sctphdr;

#if defined(__FreeBSD__) || defined(_WIN32) || defined(__Userspace__)
void sctp_close(struct socket *so);
#else
int sctp_detach(struct socket *so);
#endif
#if defined(__FreeBSD__) && !defined(__Userspace__)
void sctp_abort(struct socket *so);
#else
int sctp_abort(struct socket *so);
#endif
int sctp_disconnect(struct socket *so);
#if !defined(__Userspace__)
#if defined(__APPLE__) && !defined(APPLE_LEOPARD) && !defined(APPLE_SNOWLEOPARD) && !defined(APPLE_LION) && !defined(APPLE_MOUNTAINLION) && !defined(APPLE_ELCAPITAN)
void sctp_ctlinput(int, struct sockaddr *, void *, struct ifnet * SCTP_UNUSED);
#elif defined(__FreeBSD__)
ipproto_ctlinput_t sctp_ctlinput;
#else
void sctp_ctlinput(int, struct sockaddr *, void *);
#endif
int sctp_ctloutput(struct socket *, struct sockopt *);
#ifdef INET
void sctp_input_with_port(struct mbuf *, int, uint16_t);
#if defined(__FreeBSD__) && !defined(__Userspace__)
int sctp_input(struct mbuf **, int *, int);
#else
void sctp_input(struct mbuf *, int);
#endif
#endif
void sctp_pathmtu_adjustment(struct sctp_tcb *, uint32_t, bool);
#else
#if defined(__Userspace__)
void sctp_pathmtu_adjustment(struct sctp_tcb *, uint32_t, bool);
#else
void sctp_input(struct mbuf *,...);
#endif
void *sctp_ctlinput(int, struct sockaddr *, void *);
int sctp_ctloutput(int, struct socket *, int, int, struct mbuf **);
#endif
#if !(defined(__FreeBSD__) && !defined(__Userspace__))
void sctp_drain(void);
#endif
#if defined(__Userspace__)
void sctp_init(uint16_t,
               int (*)(void *addr, void *buffer, size_t length, uint8_t tos, uint8_t set_df),
               void (*)(const char *, ...), int start_threads);
#elif defined(__APPLE__) && (!defined(APPLE_LEOPARD) && !defined(APPLE_SNOWLEOPARD) &&!defined(APPLE_LION) && !defined(APPLE_MOUNTAINLION))
void sctp_init(struct protosw *pp, struct domain *dp);
#else
#if !defined(__FreeBSD__)
void sctp_init(void);
#endif
void sctp_notify(struct sctp_inpcb *, struct sctp_tcb *, struct sctp_nets *,
    uint8_t, uint8_t, uint16_t, uint32_t);
#endif
#if !defined(__FreeBSD__) && !defined(__Userspace__)
void sctp_finish(void);
#endif
#if defined(__FreeBSD__) || defined(_WIN32) || defined(__Userspace__)
int sctp_flush(struct socket *, int);
#endif
int sctp_shutdown(struct socket *);
int sctp_bindx(struct socket *, int, struct sockaddr_storage *,
	int, int, struct proc *);
/* can't use sctp_assoc_t here */
int sctp_peeloff(struct socket *, struct socket *, int, caddr_t, int *);
#if !defined(__Userspace__)
int sctp_ingetaddr(struct socket *, struct sockaddr **);
#else
int sctp_ingetaddr(struct socket *, struct mbuf *);
#endif
#if !defined(__Userspace__)
int sctp_peeraddr(struct socket *, struct sockaddr **);
#else
int sctp_peeraddr(struct socket *, struct mbuf *);
#endif
#if defined(__FreeBSD__) && !defined(__Userspace__)
int sctp_listen(struct socket *, int, struct thread *);
#elif defined(_WIN32) && !defined(__Userspace__)
int sctp_listen(struct socket *, int, PKTHREAD);
#elif defined(__Userspace__)
int sctp_listen(struct socket *, int, struct proc *);
#else
int sctp_listen(struct socket *, struct proc *);
#endif
int sctp_accept(struct socket *, struct sockaddr **);

#endif /* _KERNEL */

#endif /* !_NETINET_SCTP_VAR_H_ */
