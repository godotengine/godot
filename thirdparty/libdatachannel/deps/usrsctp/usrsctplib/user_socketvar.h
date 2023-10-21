/*-
 * Copyright (c) 1982, 1986, 1990, 1993
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

/* __Userspace__ version of <sys/socketvar.h> goes here.*/

#ifndef _USER_SOCKETVAR_H_
#define _USER_SOCKETVAR_H_

#if defined(__APPLE__)
#include <sys/types.h>
#include <unistd.h>
#endif

/* #include <sys/selinfo.h> */ /*__Userspace__ alternative?*/	/* for struct selinfo */
/* #include <sys/_lock.h>  was 0 byte file */
/* #include <sys/_mutex.h> was 0 byte file */
/* #include <sys/_sx.h> */ /*__Userspace__ alternative?*/
#if !defined(__DragonFly__) && !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(_WIN32) && !defined(__native_client__)
#include <sys/uio.h>
#endif
#define SOCK_MAXADDRLEN 255
#if !defined(MSG_NOTIFICATION)
#define MSG_NOTIFICATION 0x2000         /* SCTP notification */
#endif
#define SCTP_SO_LINGER     0x0001
#define SCTP_SO_ACCEPTCONN 0x0002
#define SS_CANTRCVMORE 0x020
#define SS_CANTSENDMORE 0x010

#if defined(__APPLE__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(_WIN32) || defined(__native_client__)
#define UIO_MAXIOV 1024
#define ERESTART (-1)
#endif

#if !defined(__APPLE__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
enum	uio_rw { UIO_READ, UIO_WRITE };
#endif

#if !defined(__NetBSD__) && !defined(__OpenBSD__)
/* Segment flag values. */
enum uio_seg {
	UIO_USERSPACE,		/* from user data space */
	UIO_SYSSPACE		/* from system space */
};
#endif

struct proc {
	int stub; /* struct proc is a dummy for __Userspace__ */
};

MALLOC_DECLARE(M_ACCF);
MALLOC_DECLARE(M_PCB);
MALLOC_DECLARE(M_SONAME);

/* __Userspace__ Are these all the fields we need?
 * Removing struct thread *uio_td;    owner field
*/
struct uio {
	struct	iovec *uio_iov;		/* scatter/gather list */
	int		uio_iovcnt;		/* length of scatter/gather list */
	off_t	uio_offset;		/* offset in target object */
	ssize_t 	uio_resid;		/* remaining bytes to process */
	enum	uio_seg uio_segflg;	/* address space */
	enum	uio_rw uio_rw;		/* operation */
};


/* __Userspace__ */

/*
 * Kernel structure per socket.
 * Contains send and receive buffer queues,
 * handle on protocol and pointer to protocol
 * private data and error information.
 */
#if defined(_WIN32)
#define AF_ROUTE  17
#if !defined(__MINGW32__)
typedef __int32 pid_t;
#endif
typedef unsigned __int32 uid_t;
enum sigType {
	SIGNAL = 0,
	BROADCAST = 1,
	MAX_EVENTS = 2
};
#endif

/*-
 * Locking key to struct socket:
 * (a) constant after allocation, no locking required.
 * (b) locked by SOCK_LOCK(so).
 * (c) locked by SOCKBUF_LOCK(&so->so_rcv).
 * (d) locked by SOCKBUF_LOCK(&so->so_snd).
 * (e) locked by ACCEPT_LOCK().
 * (f) not locked since integer reads/writes are atomic.
 * (g) used only as a sleep/wakeup address, no value.
 * (h) locked by global mutex so_global_mtx.
 */
struct socket {
	int	so_count;		/* (b) reference count */
	short	so_type;		/* (a) generic type, see socket.h */
	short	so_options;		/* from socket call, see socket.h */
	short	so_linger;		/* time to linger while closing */
	short	so_state;		/* (b) internal state flags SS_* */
	int	so_qstate;		/* (e) internal state flags SQ_* */
	void	*so_pcb;		/* protocol control block */
	int	so_dom;
/*
 * Variables for connection queuing.
 * Socket where accepts occur is so_head in all subsidiary sockets.
 * If so_head is 0, socket is not related to an accept.
 * For head socket so_incomp queues partially completed connections,
 * while so_comp is a queue of connections ready to be accepted.
 * If a connection is aborted and it has so_head set, then
 * it has to be pulled out of either so_incomp or so_comp.
 * We allow connections to queue up based on current queue lengths
 * and limit on number of queued connections for this socket.
 */
	struct	socket *so_head;	/* (e) back pointer to listen socket */
	TAILQ_HEAD(, socket) so_incomp;	/* (e) queue of partial unaccepted connections */
	TAILQ_HEAD(, socket) so_comp;	/* (e) queue of complete unaccepted connections */
	TAILQ_ENTRY(socket) so_list;	/* (e) list of unaccepted connections */
	u_short	so_qlen;		/* (e) number of unaccepted connections */
	u_short	so_incqlen;		/* (e) number of unaccepted incomplete
					   connections */
	u_short	so_qlimit;		/* (e) max number queued connections */
	short	so_timeo;		/* (g) connection timeout */
	userland_cond_t timeo_cond;      /* timeo_cond condition variable being used in wakeup */

	u_short	so_error;		/* (f) error affecting connection */
	struct	sigio *so_sigio;	/* [sg] information for async I/O or
					   out of band data (SIGURG) */
	u_long	so_oobmark;		/* (c) chars to oob mark */
	TAILQ_HEAD(, aiocblist) so_aiojobq; /* AIO ops waiting on socket */
/*
 * Variables for socket buffering.
 */
	struct sockbuf {
		/* __Userspace__ Many of these fields may
		 * not be required for the sctp stack.
		 * Commenting out the following.
		 * Including pthread mutex and condition variable to be
		 * used by sbwait, sorwakeup and sowwakeup.
		*/
		/* struct	selinfo sb_sel;*/ /* process selecting read/write */
		/* struct	mtx sb_mtx;*/	/* sockbuf lock */
		/* struct	sx sb_sx;*/	/* prevent I/O interlacing */
		userland_cond_t sb_cond; /* sockbuf condition variable */
		userland_mutex_t sb_mtx; /* sockbuf lock associated with sb_cond */
		short	sb_state;	/* (c/d) socket state on sockbuf */
#define	sb_startzero	sb_mb
		struct	mbuf *sb_mb;	/* (c/d) the mbuf chain */
		struct	mbuf *sb_mbtail; /* (c/d) the last mbuf in the chain */
		struct	mbuf *sb_lastrecord;	/* (c/d) first mbuf of last
						 * record in socket buffer */
		struct	mbuf *sb_sndptr; /* (c/d) pointer into mbuf chain */
		u_int	sb_sndptroff;	/* (c/d) byte offset of ptr into chain */
		u_int	sb_cc;		/* (c/d) actual chars in buffer */
		u_int	sb_hiwat;	/* (c/d) max actual char count */
		u_int	sb_mbcnt;	/* (c/d) chars of mbufs used */
		u_int	sb_mbmax;	/* (c/d) max chars of mbufs to use */
		u_int	sb_ctl;		/* (c/d) non-data chars in buffer */
		int	sb_lowat;	/* (c/d) low water mark */
		int	sb_timeo;	/* (c/d) timeout for read/write */
		short	sb_flags;	/* (c/d) flags, see below */
	} so_rcv, so_snd;
/*
 * Constants for sb_flags field of struct sockbuf.
 */
#define	SB_MAX		(256*1024)	/* default for max chars in sockbuf */
#define SB_RAW          (64*1024*2)    /*Aligning so->so_rcv.sb_hiwat with the receive buffer size of raw socket*/
/*
 * Constants for sb_flags field of struct sockbuf.
 */
#define	SB_WAIT		0x04		/* someone is waiting for data/space */
#define	SB_SEL		0x08		/* someone is selecting */
#define	SB_ASYNC	0x10		/* ASYNC I/O, need signals */
#define	SB_UPCALL	0x20		/* someone wants an upcall */
#define	SB_NOINTR	0x40		/* operations not interruptible */
#define	SB_AIO		0x80		/* AIO operations queued */
#define	SB_KNOTE	0x100		/* kernel note attached */
#define	SB_AUTOSIZE	0x800		/* automatically size socket buffer */

	void	(*so_upcall)(struct socket *, void *, int);
	void	*so_upcallarg;
	struct	ucred *so_cred;		/* (a) user credentials */
	struct	label *so_label;	/* (b) MAC label for socket */
	struct	label *so_peerlabel;	/* (b) cached MAC label for peer */
	/* NB: generation count must not be first. */
	uint32_t so_gencnt;		/* (h) generation count */
	void	*so_emuldata;		/* (b) private data for emulators */
 	struct so_accf {
		struct	accept_filter *so_accept_filter;
		void	*so_accept_filter_arg;	/* saved filter args */
		char	*so_accept_filter_str;	/* saved user args */
	} *so_accf;
};

#define SB_EMPTY_FIXUP(sb) do {						\
	if ((sb)->sb_mb == NULL) {					\
		(sb)->sb_mbtail = NULL;					\
		(sb)->sb_lastrecord = NULL;				\
	}								\
} while (/*CONSTCOND*/0)

/*
 * Global accept mutex to serialize access to accept queues and
 * fields associated with multiple sockets.  This allows us to
 * avoid defining a lock order between listen and accept sockets
 * until such time as it proves to be a good idea.
 */
#if defined(_WIN32)
extern userland_mutex_t accept_mtx;
extern userland_cond_t accept_cond;
#define ACCEPT_LOCK_ASSERT()
#define	ACCEPT_LOCK() do { \
	EnterCriticalSection(&accept_mtx); \
} while (0)
#define	ACCEPT_UNLOCK()	do { \
	LeaveCriticalSection(&accept_mtx); \
} while (0)
#define	ACCEPT_UNLOCK_ASSERT()
#else
extern userland_mutex_t accept_mtx;

extern userland_cond_t accept_cond;
#ifdef INVARIANTS
#define	ACCEPT_LOCK()	KASSERT(pthread_mutex_lock(&accept_mtx) == 0, ("%s: accept_mtx already locked", __func__))
#define	ACCEPT_UNLOCK()	KASSERT(pthread_mutex_unlock(&accept_mtx) == 0, ("%s: accept_mtx not locked", __func__))
#else
#define	ACCEPT_LOCK()   (void)pthread_mutex_lock(&accept_mtx)
#define	ACCEPT_UNLOCK() (void)pthread_mutex_unlock(&accept_mtx)
#endif
#define	ACCEPT_LOCK_ASSERT() \
          KASSERT(pthread_mutex_trylock(&accept_mtx) == EBUSY, ("%s: accept_mtx not locked", __func__))
#define	ACCEPT_UNLOCK_ASSERT() do {                                                               \
	  KASSERT(pthread_mutex_trylock(&accept_mtx) == 0, ("%s: accept_mtx  locked", __func__)); \
	  (void)pthread_mutex_unlock(&accept_mtx);                                                \
        } while (0)
#endif

/*
 * Per-socket buffer mutex used to protect most fields in the socket
 * buffer.
 */
#define	SOCKBUF_MTX(_sb) (&(_sb)->sb_mtx)
#if defined(_WIN32)
#define SOCKBUF_LOCK_INIT(_sb, _name) \
	InitializeCriticalSection(SOCKBUF_MTX(_sb))
#define SOCKBUF_LOCK_DESTROY(_sb) DeleteCriticalSection(SOCKBUF_MTX(_sb))
#define SOCKBUF_COND_INIT(_sb) InitializeConditionVariable((&(_sb)->sb_cond))
#define SOCKBUF_COND_DESTROY(_sb) DeleteConditionVariable((&(_sb)->sb_cond))
#define SOCK_COND_INIT(_so) InitializeConditionVariable((&(_so)->timeo_cond))
#define SOCK_COND_DESTROY(_so) DeleteConditionVariable((&(_so)->timeo_cond))
#define SOCK_COND(_so) (&(_so)->timeo_cond)
#else
#ifdef INVARIANTS
#define SOCKBUF_LOCK_INIT(_sb, _name) do {                                 \
	pthread_mutexattr_t mutex_attr;                                    \
	                                                                   \
	pthread_mutexattr_init(&mutex_attr);                               \
	pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);  \
	pthread_mutex_init(SOCKBUF_MTX(_sb), &mutex_attr);                 \
	pthread_mutexattr_destroy(&mutex_attr);                            \
} while (0)
#else
#define SOCKBUF_LOCK_INIT(_sb, _name) \
	pthread_mutex_init(SOCKBUF_MTX(_sb), NULL)
#endif
#define SOCKBUF_LOCK_DESTROY(_sb) pthread_mutex_destroy(SOCKBUF_MTX(_sb))
#define SOCKBUF_COND_INIT(_sb) pthread_cond_init((&(_sb)->sb_cond), NULL)
#define SOCKBUF_COND_DESTROY(_sb) pthread_cond_destroy((&(_sb)->sb_cond))
#define SOCK_COND_INIT(_so) pthread_cond_init((&(_so)->timeo_cond), NULL)
#define SOCK_COND_DESTROY(_so) pthread_cond_destroy((&(_so)->timeo_cond))
#define SOCK_COND(_so) (&(_so)->timeo_cond)
#endif
/*__Userspace__ SOCKBUF_LOCK(_sb) is now defined in netinet/sctp_process_lock.h */

/* #define	SOCKBUF_OWNED(_sb)		mtx_owned(SOCKBUF_MTX(_sb)) unused */
/*__Userspace__ SOCKBUF_UNLOCK(_sb) is now defined in netinet/sctp_process_lock.h */

/*__Userspace__ SOCKBUF_LOCK_ASSERT(_sb) is now defined in netinet/sctp_process_lock.h */

/* #define	SOCKBUF_UNLOCK_ASSERT(_sb)	mtx_assert(SOCKBUF_MTX(_sb), MA_NOTOWNED)   unused */

/*
 * Per-socket mutex: we reuse the receive socket buffer mutex for space
 * efficiency.  This decision should probably be revisited as we optimize
 * locking for the socket code.
 */
#define	SOCK_MTX(_so)			SOCKBUF_MTX(&(_so)->so_rcv)
/*__Userspace__ SOCK_LOCK(_so) is now defined in netinet/sctp_process_lock.h */

/* #define	SOCK_OWNED(_so)			SOCKBUF_OWNED(&(_so)->so_rcv) unused */
/*__Userspace__ SOCK_UNLOCK(_so) is now defined in netinet/sctp_process_lock.h */

#define	SOCK_LOCK_ASSERT(_so)		SOCKBUF_LOCK_ASSERT(&(_so)->so_rcv)

/*
 * Socket state bits.
 *
 * Historically, this bits were all kept in the so_state field.  For
 * locking reasons, they are now in multiple fields, as they are
 * locked differently.  so_state maintains basic socket state protected
 * by the socket lock.  so_qstate holds information about the socket
 * accept queues.  Each socket buffer also has a state field holding
 * information relevant to that socket buffer (can't send, rcv).  Many
 * fields will be read without locks to improve performance and avoid
 * lock order issues.  However, this approach must be used with caution.
 */
#define	SS_NOFDREF		0x0001	/* no file table ref any more */
#define	SS_ISCONNECTED		0x0002	/* socket connected to a peer */
#define	SS_ISCONNECTING		0x0004	/* in process of connecting to peer */
#define	SS_ISDISCONNECTING	0x0008	/* in process of disconnecting */
#define	SS_NBIO			0x0100	/* non-blocking ops */
#define	SS_ASYNC		0x0200	/* async i/o notify */
#define	SS_ISCONFIRMING		0x0400	/* deciding to accept connection req */
#define	SS_ISDISCONNECTED	0x2000	/* socket disconnected from peer */
/*
 * Protocols can mark a socket as SS_PROTOREF to indicate that, following
 * pru_detach, they still want the socket to persist, and will free it
 * themselves when they are done.  Protocols should only ever call sofree()
 * following setting this flag in pru_detach(), and never otherwise, as
 * sofree() bypasses socket reference counting.
 */
#define	SS_PROTOREF		0x4000	/* strong protocol reference */

/*
 * Socket state bits now stored in the socket buffer state field.
 */
#define	SBS_CANTSENDMORE	0x0010	/* can't send more data to peer */
#define	SBS_CANTRCVMORE		0x0020	/* can't receive more data from peer */
#define	SBS_RCVATMARK		0x0040	/* at mark on input */

/*
 * Socket state bits stored in so_qstate.
 */
#define	SQ_INCOMP		0x0800	/* unaccepted, incomplete connection */
#define	SQ_COMP			0x1000	/* unaccepted, complete connection */

/*
 * Socket event flags
 */
#define SCTP_EVENT_READ		0x0001	/* socket is readable */
#define SCTP_EVENT_WRITE	0x0002	/* socket is writeable */
#define SCTP_EVENT_ERROR	0x0004	/* socket has an error state */


/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/*                   __Userspace__                             */
/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/* this new __Userspace__ section is to copy portions of the _KERNEL block
 *  above into, avoiding having to port the entire thing at once...
 *  For function prototypes, the full bodies are in user_socket.c .
 */

/* ---------------------------------------------------------- */
/* --- function prototypes (implemented in user_socket.c) --- */
/* ---------------------------------------------------------- */
void	soisconnecting(struct socket *so);
void	soisdisconnecting(struct socket *so);
void	soisconnected(struct socket *so);
struct socket * sonewconn(struct socket *head, int connstatus);
void	socantrcvmore(struct socket *so);
void	socantsendmore(struct socket *so);
void	sofree(struct socket *so);



/* -------------- */
/* --- macros --- */
/* -------------- */

#define	soref(so) do {							\
	SOCK_LOCK_ASSERT(so);						\
	++(so)->so_count;						\
} while (0)

#define	sorele(so) do {							\
	ACCEPT_LOCK_ASSERT();						\
	SOCK_LOCK_ASSERT(so);						\
	KASSERT((so)->so_count > 0, ("sorele"));			\
	if (--(so)->so_count == 0)					\
		sofree(so);						\
	else {								\
		SOCK_UNLOCK(so);					\
		ACCEPT_UNLOCK();					\
	}								\
} while (0)


/* replacing imin with min (user_environment.h) */
#define	sbspace(sb) \
    ((long) min((int)((sb)->sb_hiwat - (sb)->sb_cc), \
	 (int)((sb)->sb_mbmax - (sb)->sb_mbcnt)))

/* do we have to send all at once on a socket? */
#define	sosendallatonce(so) \
    ((so)->so_proto->pr_flags & PR_ATOMIC)

/* can we read something from so? */
#define	soreadable(so) \
    ((int)((so)->so_rcv.sb_cc) >= (so)->so_rcv.sb_lowat || \
	((so)->so_rcv.sb_state & SBS_CANTRCVMORE) || \
	!TAILQ_EMPTY(&(so)->so_comp) || (so)->so_error)

#if 0  /*  original */
#define PR_CONNREQUIRED 0x04  /* from sys/protosw.h "needed" for sowriteable */
#define	sowriteable(so) \
    ((sbspace(&(so)->so_snd) >= (so)->so_snd.sb_lowat && \
	(((so)->so_state&SS_ISCONNECTED) || \
	  ((so)->so_proto->pr_flags&PR_CONNREQUIRED)==0)) || \
     ((so)->so_snd.sb_state & SBS_CANTSENDMORE) || \
     (so)->so_error)
#else  /* line with PR_CONNREQUIRED removed */
/* can we write something to so? */
#define	sowriteable(so) \
    ((sbspace(&(so)->so_snd) >= (so)->so_snd.sb_lowat && \
      (((so)->so_state&SS_ISCONNECTED))) ||              \
     ((so)->so_snd.sb_state & SBS_CANTSENDMORE) || \
     (so)->so_error)
#endif

extern void solisten_proto(struct socket *so, int backlog);
extern int solisten_proto_check(struct socket *so);
extern int sctp_listen(struct socket *so, int backlog, struct proc *p);
extern void socantrcvmore_locked(struct socket *so);
extern int sctp_bind(struct socket *so, struct sockaddr *addr);
extern int sctp6_bind(struct socket *so, struct sockaddr *addr, void *proc);
extern int sctpconn_bind(struct socket *so, struct sockaddr *addr);
extern int sctp_accept(struct socket *so, struct sockaddr **addr);
extern int sctp_attach(struct socket *so, int proto, uint32_t vrf_id);
extern int sctp6_attach(struct socket *so, int proto, uint32_t vrf_id);
extern int sctp_abort(struct socket *so);
extern int sctp6_abort(struct socket *so);
extern void sctp_close(struct socket *so);
extern int soaccept(struct socket *so, struct sockaddr **nam);
extern int solisten(struct socket *so, int backlog);
extern int  soreserve(struct socket *so, u_long sndcc, u_long rcvcc);
extern void sowakeup(struct socket *so, struct sockbuf *sb);
extern void wakeup(void *ident, struct socket *so); /*__Userspace__ */
extern int uiomove(void *cp, int n, struct uio *uio);
extern int sbwait(struct sockbuf *sb);
extern int sodisconnect(struct socket *so);
extern int soconnect(struct socket *so, struct sockaddr *nam);
extern int sctp_disconnect(struct socket *so);
extern int sctp_connect(struct socket *so, struct sockaddr *addr);
extern int sctp6_connect(struct socket *so, struct sockaddr *addr);
extern int sctpconn_connect(struct socket *so, struct sockaddr *addr);
extern void sctp_finish(void);

/* ------------------------------------------------ */
/* -----  macros copied from above ---- */
/* ------------------------------------------------ */

/*
 * Do we need to notify the other side when I/O is possible?
 */
#define	sb_notify(sb)	(((sb)->sb_flags & (SB_WAIT | SB_SEL | SB_ASYNC | \
    SB_UPCALL | SB_AIO | SB_KNOTE)) != 0)


/*
 * In sorwakeup() and sowwakeup(), acquire the socket buffer lock to
 * avoid a non-atomic test-and-wakeup.  However, sowakeup is
 * responsible for releasing the lock if it is called.  We unlock only
 * if we don't call into sowakeup.  If any code is introduced that
 * directly invokes the underlying sowakeup() primitives, it must
 * maintain the same semantics.
 */
#define	sorwakeup_locked(so) do {					\
	SOCKBUF_LOCK_ASSERT(&(so)->so_rcv);				\
	if (sb_notify(&(so)->so_rcv))					\
		sowakeup((so), &(so)->so_rcv);	 			\
	else								\
		SOCKBUF_UNLOCK(&(so)->so_rcv);				\
} while (0)

#define	sorwakeup(so) do {						\
	SOCKBUF_LOCK(&(so)->so_rcv);					\
	sorwakeup_locked(so);						\
} while (0)

#define	sowwakeup_locked(so) do {					\
	SOCKBUF_LOCK_ASSERT(&(so)->so_snd);				\
	if (sb_notify(&(so)->so_snd))					\
		sowakeup((so), &(so)->so_snd); 				\
	else								\
		SOCKBUF_UNLOCK(&(so)->so_snd);				\
} while (0)

#define	sowwakeup(so) do {						\
	SOCKBUF_LOCK(&(so)->so_snd);					\
	sowwakeup_locked(so);						\
} while (0)

#endif /* !_SYS_SOCKETVAR_H_ */
