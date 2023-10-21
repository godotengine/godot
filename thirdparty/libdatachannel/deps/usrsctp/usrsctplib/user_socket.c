/*-
 * Copyright (c) 1982, 1986, 1988, 1990, 1993
 *      The Regents of the University of California.
 * Copyright (c) 2004 The FreeBSD Foundation
 * Copyright (c) 2004-2008 Robert N. M. Watson
 * Copyright (c) 2009-2010 Brad Penoff
 * Copyright (c) 2009-2010 Humaira Kamal
 * Copyright (c) 2011-2012 Irene Ruengeler
 * Copyright (c) 2011-2012 Michael Tuexen
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
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#include <netinet/sctp_os.h>
#include <netinet/sctp_pcb.h>
#include <netinet/sctputil.h>
#include <netinet/sctp_var.h>
#include <netinet/sctp_sysctl.h>
#include <netinet/sctp_input.h>
#include <netinet/sctp_peeloff.h>
#include <netinet/sctp_callout.h>
#include <netinet/sctp_crc32.h>
#ifdef INET6
#include <netinet6/sctp6_var.h>
#endif
#if defined(__FreeBSD__)
#include <sys/param.h>
#endif
#if defined(__linux__)
#define __FAVOR_BSD    /* (on Ubuntu at least) enables UDP header field names like BSD in RFC 768 */
#endif
#if !defined(_WIN32)
#if defined INET || defined INET6
#include <netinet/udp.h>
#endif
#include <arpa/inet.h>
#else
#include <user_socketvar.h>
#endif
userland_mutex_t accept_mtx;
userland_cond_t accept_cond;
#ifdef _WIN32
#include <time.h>
#include <sys/timeb.h>
#endif

MALLOC_DEFINE(M_PCB, "sctp_pcb", "sctp pcb");
MALLOC_DEFINE(M_SONAME, "sctp_soname", "sctp soname");
#define MAXLEN_MBUF_CHAIN  32

/* Prototypes */
extern int sctp_sosend(struct socket *so, struct sockaddr *addr, struct uio *uio,
                       struct mbuf *top, struct mbuf *control, int flags,
                       /* proc is a dummy in __Userspace__ and will not be passed to sctp_lower_sosend */
                       struct proc *p);

extern int sctp_attach(struct socket *so, int proto, uint32_t vrf_id);
extern int sctpconn_attach(struct socket *so, int proto, uint32_t vrf_id);

static void init_sync(void) {
#if defined(_WIN32)
#if defined(INET) || defined(INET6)
	WSADATA wsaData;

	if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
		SCTP_PRINTF("WSAStartup failed\n");
		exit (-1);
	}
#endif
	InitializeConditionVariable(&accept_cond);
	InitializeCriticalSection(&accept_mtx);
#else
	pthread_mutexattr_t mutex_attr;

	pthread_mutexattr_init(&mutex_attr);
#ifdef INVARIANTS
	pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);
#endif
	pthread_mutex_init(&accept_mtx, &mutex_attr);
	pthread_mutexattr_destroy(&mutex_attr);
	pthread_cond_init(&accept_cond, NULL);
#endif
}

void
usrsctp_init(uint16_t port,
             int (*conn_output)(void *addr, void *buffer, size_t length, uint8_t tos, uint8_t set_df),
             void (*debug_printf)(const char *format, ...))
{
	init_sync();
	sctp_init(port, conn_output, debug_printf, 1);
}


void
usrsctp_init_nothreads(uint16_t port,
                       int (*conn_output)(void *addr, void *buffer, size_t length, uint8_t tos, uint8_t set_df),
                       void (*debug_printf)(const char *format, ...))
{
	init_sync();
	sctp_init(port, conn_output, debug_printf, 0);
}


/* Taken from  usr/src/sys/kern/uipc_sockbuf.c and modified for __Userspace__*/
/*
 * Socantsendmore indicates that no more data will be sent on the socket; it
 * would normally be applied to a socket when the user informs the system
 * that no more data is to be sent, by the protocol code (in case
 * PRU_SHUTDOWN).  Socantrcvmore indicates that no more data will be
 * received, and will normally be applied to the socket by a protocol when it
 * detects that the peer will send no more data.  Data queued for reading in
 * the socket may yet be read.
 */

void socantrcvmore_locked(struct socket *so)
{
	SOCKBUF_LOCK_ASSERT(&so->so_rcv);
	so->so_rcv.sb_state |= SBS_CANTRCVMORE;
	sorwakeup_locked(so);
}

void socantrcvmore(struct socket *so)
{
	SOCKBUF_LOCK(&so->so_rcv);
	socantrcvmore_locked(so);
}

void
socantsendmore_locked(struct socket *so)
{
	SOCKBUF_LOCK_ASSERT(&so->so_snd);
	so->so_snd.sb_state |= SBS_CANTSENDMORE;
	sowwakeup_locked(so);
}

void
socantsendmore(struct socket *so)
{
	SOCKBUF_LOCK(&so->so_snd);
	socantsendmore_locked(so);
}



/* Taken from  usr/src/sys/kern/uipc_sockbuf.c and called within sctp_lower_sosend.
 */
int
sbwait(struct sockbuf *sb)
{
	SOCKBUF_LOCK_ASSERT(sb);

	sb->sb_flags |= SB_WAIT;
#if defined(_WIN32)
	if (SleepConditionVariableCS(&(sb->sb_cond), &(sb->sb_mtx), INFINITE))
		return (0);
	else
		return (-1);
#else
	return (pthread_cond_wait(&(sb->sb_cond), &(sb->sb_mtx)));
#endif
}




/* Taken from  /src/sys/kern/uipc_socket.c
 * and modified for __Userspace__
 */
static struct socket *
soalloc(void)
{
	struct socket *so;

	/*
	 * soalloc() sets of socket layer state for a socket,
	 * called only by socreate() and sonewconn().
	 *
	 * sodealloc() tears down socket layer state for a socket,
	 * called only by sofree() and sonewconn().
	 * __Userspace__ TODO : Make sure so is properly deallocated
	 * when tearing down the connection.
	 */

	so = (struct socket *)malloc(sizeof(struct socket));

	if (so == NULL) {
		return (NULL);
	}
	memset(so, 0, sizeof(struct socket));

	/* __Userspace__ Initializing the socket locks here */
	SOCKBUF_LOCK_INIT(&so->so_snd, "so_snd");
	SOCKBUF_LOCK_INIT(&so->so_rcv, "so_rcv");
	SOCKBUF_COND_INIT(&so->so_snd);
	SOCKBUF_COND_INIT(&so->so_rcv);
	SOCK_COND_INIT(so); /* timeo_cond */

	/* __Userspace__ Any ref counting required here? Will we have any use for aiojobq?
	   What about gencnt and numopensockets?*/
	TAILQ_INIT(&so->so_aiojobq);
	return (so);
}

static void
sodealloc(struct socket *so)
{

	KASSERT(so->so_count == 0, ("sodealloc(): so_count %d", so->so_count));
	KASSERT(so->so_pcb == NULL, ("sodealloc(): so_pcb != NULL"));

	SOCKBUF_COND_DESTROY(&so->so_snd);
	SOCKBUF_COND_DESTROY(&so->so_rcv);

	SOCK_COND_DESTROY(so);

	SOCKBUF_LOCK_DESTROY(&so->so_snd);
	SOCKBUF_LOCK_DESTROY(&so->so_rcv);

	free(so);
}

/* Taken from  /src/sys/kern/uipc_socket.c
 * and modified for __Userspace__
 */
void
sofree(struct socket *so)
{
	struct socket *head;

	ACCEPT_LOCK_ASSERT();
	SOCK_LOCK_ASSERT(so);
	/* SS_NOFDREF unset in accept call.  this condition seems irrelevant
	 *  for __Userspace__...
	 */
	if (so->so_count != 0 ||
	    (so->so_state & SS_PROTOREF) || (so->so_qstate & SQ_COMP)) {
		SOCK_UNLOCK(so);
		ACCEPT_UNLOCK();
		return;
	}
	head = so->so_head;
	if (head != NULL) {
		KASSERT((so->so_qstate & SQ_COMP) != 0 ||
		    (so->so_qstate & SQ_INCOMP) != 0,
		    ("sofree: so_head != NULL, but neither SQ_COMP nor "
		    "SQ_INCOMP"));
		KASSERT((so->so_qstate & SQ_COMP) == 0 ||
		    (so->so_qstate & SQ_INCOMP) == 0,
		    ("sofree: so->so_qstate is SQ_COMP and also SQ_INCOMP"));
		TAILQ_REMOVE(&head->so_incomp, so, so_list);
		head->so_incqlen--;
		so->so_qstate &= ~SQ_INCOMP;
		so->so_head = NULL;
	}
	KASSERT((so->so_qstate & SQ_COMP) == 0 &&
	    (so->so_qstate & SQ_INCOMP) == 0,
	    ("sofree: so_head == NULL, but still SQ_COMP(%d) or SQ_INCOMP(%d)",
	    so->so_qstate & SQ_COMP, so->so_qstate & SQ_INCOMP));
	if (so->so_options & SCTP_SO_ACCEPTCONN) {
		KASSERT((TAILQ_EMPTY(&so->so_comp)), ("sofree: so_comp populated"));
		KASSERT((TAILQ_EMPTY(&so->so_incomp)), ("sofree: so_comp populated"));
	}
	SOCK_UNLOCK(so);
	ACCEPT_UNLOCK();
	sctp_close(so); /* was...    sctp_detach(so); */
	/*
	 * From this point on, we assume that no other references to this
	 * socket exist anywhere else in the stack.  Therefore, no locks need
	 * to be acquired or held.
	 *
	 * We used to do a lot of socket buffer and socket locking here, as
	 * well as invoke sorflush() and perform wakeups.  The direct call to
	 * dom_dispose() and sbrelease_internal() are an inlining of what was
	 * necessary from sorflush().
	 *
	 * Notice that the socket buffer and kqueue state are torn down
	 * before calling pru_detach.  This means that protocols should not
	 * assume they can perform socket wakeups, etc, in their detach code.
	 */
	sodealloc(so);
}



/* Taken from  /src/sys/kern/uipc_socket.c */
void
soabort(struct socket *so)
{
	sctp_abort(so);
	ACCEPT_LOCK();
	SOCK_LOCK(so);
	sofree(so);
}


/* Taken from  usr/src/sys/kern/uipc_socket.c and called within sctp_connect (sctp_usrreq.c).
 *  We use sctp_connect for send_one_init_real in ms1.
 */
void
soisconnecting(struct socket *so)
{

	SOCK_LOCK(so);
	so->so_state &= ~(SS_ISCONNECTED|SS_ISDISCONNECTING);
	so->so_state |= SS_ISCONNECTING;
	SOCK_UNLOCK(so);
}

/* Taken from  usr/src/sys/kern/uipc_socket.c and called within sctp_disconnect (sctp_usrreq.c).
 *  TODO Do we use sctp_disconnect?
 */
void
soisdisconnecting(struct socket *so)
{

	/*
	 * Note: This code assumes that SOCK_LOCK(so) and
	 * SOCKBUF_LOCK(&so->so_rcv) are the same.
	 */
	SOCKBUF_LOCK(&so->so_rcv);
	so->so_state &= ~SS_ISCONNECTING;
	so->so_state |= SS_ISDISCONNECTING;
	so->so_rcv.sb_state |= SBS_CANTRCVMORE;
	sorwakeup_locked(so);
	SOCKBUF_LOCK(&so->so_snd);
	so->so_snd.sb_state |= SBS_CANTSENDMORE;
	sowwakeup_locked(so);
	wakeup("dummy",so);
	/* requires 2 args but this was in orig */
	/* wakeup(&so->so_timeo); */
}


/* Taken from sys/kern/kern_synch.c and
   modified for __Userspace__
*/

/*
 * Make all threads sleeping on the specified identifier runnable.
 * Associating wakeup with so_timeo identifier and timeo_cond
 * condition variable. TODO. If we use iterator thread then we need to
 * modify wakeup so it can distinguish between iterator identifier and
 * timeo identifier.
 */
void
wakeup(void *ident, struct socket *so)
{
	SOCK_LOCK(so);
#if defined(_WIN32)
	WakeAllConditionVariable(&(so)->timeo_cond);
#else
	pthread_cond_broadcast(&(so)->timeo_cond);
#endif
	SOCK_UNLOCK(so);
}


/*
 * Make a thread sleeping on the specified identifier runnable.
 * May wake more than one thread if a target thread is currently
 * swapped out.
 */
void
wakeup_one(void *ident)
{
	/* __Userspace__ Check: We are using accept_cond for wakeup_one.
	  It seems that wakeup_one is only called within
	  soisconnected() and sonewconn() with ident &head->so_timeo
	  head is so->so_head, which is back pointer to listen socket
	  This seems to indicate that the use of accept_cond is correct
	  since socket where accepts occur is so_head in all
	  subsidiary sockets.
	 */
	ACCEPT_LOCK();
#if defined(_WIN32)
	WakeAllConditionVariable(&accept_cond);
#else
	pthread_cond_broadcast(&accept_cond);
#endif
	ACCEPT_UNLOCK();
}


/* Called within sctp_process_cookie_[existing/new] */
void
soisconnected(struct socket *so)
{
	struct socket *head;

	ACCEPT_LOCK();
	SOCK_LOCK(so);
	so->so_state &= ~(SS_ISCONNECTING|SS_ISDISCONNECTING|SS_ISCONFIRMING);
	so->so_state |= SS_ISCONNECTED;
	head = so->so_head;
	if (head != NULL && (so->so_qstate & SQ_INCOMP)) {
		SOCK_UNLOCK(so);
		TAILQ_REMOVE(&head->so_incomp, so, so_list);
		head->so_incqlen--;
		so->so_qstate &= ~SQ_INCOMP;
		TAILQ_INSERT_TAIL(&head->so_comp, so, so_list);
		head->so_qlen++;
		so->so_qstate |= SQ_COMP;
		ACCEPT_UNLOCK();
		sorwakeup(head);
		wakeup_one(&head->so_timeo);
		return;
	}
	SOCK_UNLOCK(so);
	ACCEPT_UNLOCK();
	wakeup(&so->so_timeo, so);
	sorwakeup(so);
	sowwakeup(so);

}

/* called within sctp_handle_cookie_echo */

struct socket *
sonewconn(struct socket *head, int connstatus)
{
	struct socket *so;
	int over;

	ACCEPT_LOCK();
	over = (head->so_qlen > 3 * head->so_qlimit / 2);
	ACCEPT_UNLOCK();
#ifdef REGRESSION
	if (regression_sonewconn_earlytest && over)
#else
	if (over)
#endif
		return (NULL);
	so = soalloc();
	if (so == NULL)
		return (NULL);
	so->so_head = head;
	so->so_type = head->so_type;
	so->so_options = head->so_options &~ SCTP_SO_ACCEPTCONN;
	so->so_linger = head->so_linger;
	so->so_state = head->so_state | SS_NOFDREF;
	so->so_dom = head->so_dom;
#ifdef MAC
	SOCK_LOCK(head);
	mac_create_socket_from_socket(head, so);
	SOCK_UNLOCK(head);
#endif
	if (soreserve(so, head->so_snd.sb_hiwat, head->so_rcv.sb_hiwat)) {
		sodealloc(so);
		return (NULL);
	}
	switch (head->so_dom) {
#ifdef INET
	case AF_INET:
		if (sctp_attach(so, IPPROTO_SCTP, SCTP_DEFAULT_VRFID)) {
			sodealloc(so);
			return (NULL);
		}
		break;
#endif
#ifdef INET6
	case AF_INET6:
		if (sctp6_attach(so, IPPROTO_SCTP, SCTP_DEFAULT_VRFID)) {
			sodealloc(so);
			return (NULL);
		}
		break;
#endif
	case AF_CONN:
		if (sctpconn_attach(so, IPPROTO_SCTP, SCTP_DEFAULT_VRFID)) {
			sodealloc(so);
			return (NULL);
		}
		break;
	default:
		sodealloc(so);
		return (NULL);
		break;
	}
	so->so_rcv.sb_lowat = head->so_rcv.sb_lowat;
	so->so_snd.sb_lowat = head->so_snd.sb_lowat;
	so->so_rcv.sb_timeo = head->so_rcv.sb_timeo;
	so->so_snd.sb_timeo = head->so_snd.sb_timeo;
	so->so_rcv.sb_flags |= head->so_rcv.sb_flags & SB_AUTOSIZE;
	so->so_snd.sb_flags |= head->so_snd.sb_flags & SB_AUTOSIZE;
	so->so_state |= connstatus;
	ACCEPT_LOCK();
	if (connstatus) {
		TAILQ_INSERT_TAIL(&head->so_comp, so, so_list);
		so->so_qstate |= SQ_COMP;
		head->so_qlen++;
	} else {
		/*
		 * Keep removing sockets from the head until there's room for
		 * us to insert on the tail.  In pre-locking revisions, this
		 * was a simple if (), but as we could be racing with other
		 * threads and soabort() requires dropping locks, we must
		 * loop waiting for the condition to be true.
		 */
		while (head->so_incqlen > head->so_qlimit) {
			struct socket *sp;
			sp = TAILQ_FIRST(&head->so_incomp);
			TAILQ_REMOVE(&head->so_incomp, sp, so_list);
			head->so_incqlen--;
			sp->so_qstate &= ~SQ_INCOMP;
			sp->so_head = NULL;
			ACCEPT_UNLOCK();
			soabort(sp);
			ACCEPT_LOCK();
		}
		TAILQ_INSERT_TAIL(&head->so_incomp, so, so_list);
		so->so_qstate |= SQ_INCOMP;
		head->so_incqlen++;
	}
	ACCEPT_UNLOCK();
	if (connstatus) {
		sorwakeup(head);
		wakeup_one(&head->so_timeo);
	}
	return (so);

}

 /*
   Source: /src/sys/gnu/fs/xfs/FreeBSD/xfs_ioctl.c
 */
static __inline__ int
copy_to_user(void *dst, void *src, size_t len) {
	memcpy(dst, src, len);
	return 0;
}

static __inline__ int
copy_from_user(void *dst, void *src, size_t len) {
	memcpy(dst, src, len);
	return 0;
}

/*
 References:
 src/sys/dev/lmc/if_lmc.h:
 src/sys/powerpc/powerpc/copyinout.c
 src/sys/sys/systm.h
*/
# define copyin(u, k, len)	copy_from_user(k, u, len)

/* References:
   src/sys/powerpc/powerpc/copyinout.c
   src/sys/sys/systm.h
*/
# define copyout(k, u, len)	copy_to_user(u, k, len)


/* copyiniov definition copied/modified from src/sys/kern/kern_subr.c */
int
copyiniov(struct iovec *iovp, u_int iovcnt, struct iovec **iov, int error)
{
	u_int iovlen;

	*iov = NULL;
	if (iovcnt > UIO_MAXIOV)
		return (error);
	iovlen = iovcnt * sizeof (struct iovec);
	*iov = malloc(iovlen); /*, M_IOV, M_WAITOK); */
	error = copyin(iovp, *iov, iovlen);
	if (error) {
		free(*iov); /*, M_IOV); */
		*iov = NULL;
	}
	return (error);
}

/* (__Userspace__) version of uiomove */
int
uiomove(void *cp, int n, struct uio *uio)
{
	struct iovec *iov;
	size_t cnt;
	int error = 0;

	if ((uio->uio_rw != UIO_READ) &&
	    (uio->uio_rw != UIO_WRITE)) {
		return (EINVAL);
	}

	while (n > 0 && uio->uio_resid) {
		iov = uio->uio_iov;
		cnt = iov->iov_len;
		if (cnt == 0) {
			uio->uio_iov++;
			uio->uio_iovcnt--;
			continue;
		}
		if (cnt > (size_t)n)
			cnt = n;

		switch (uio->uio_segflg) {

		case UIO_USERSPACE:
			if (uio->uio_rw == UIO_READ)
				error = copyout(cp, iov->iov_base, cnt);
			else
				error = copyin(iov->iov_base, cp, cnt);
			if (error)
				goto out;
			break;

		case UIO_SYSSPACE:
			if (uio->uio_rw == UIO_READ)
				memcpy(iov->iov_base, cp, cnt);
			else
				memcpy(cp, iov->iov_base, cnt);
			break;
		}
		iov->iov_base = (char *)iov->iov_base + cnt;
		iov->iov_len -= cnt;
		uio->uio_resid -= cnt;
		uio->uio_offset += (off_t)cnt;
		cp = (char *)cp + cnt;
		n -= (int)cnt;
	}
out:
	return (error);
}


/* Source: src/sys/kern/uipc_syscalls.c */
int
getsockaddr(struct sockaddr **namp, caddr_t uaddr, size_t len)
{
	struct sockaddr *sa;
	int error;

	if (len > SOCK_MAXADDRLEN)
		return (ENAMETOOLONG);
	if (len < offsetof(struct sockaddr, sa_data))
		return (EINVAL);
	MALLOC(sa, struct sockaddr *, len, M_SONAME, M_WAITOK);
	error = copyin(uaddr, sa, len);
	if (error) {
		FREE(sa, M_SONAME);
	} else {
#ifdef HAVE_SA_LEN
		sa->sa_len = len;
#endif
		*namp = sa;
	}
	return (error);
}

int
usrsctp_getsockopt(struct socket *so, int level, int option_name,
                   void *option_value, socklen_t *option_len);

sctp_assoc_t
usrsctp_getassocid(struct socket *sock, struct sockaddr *sa)
{
	struct sctp_paddrinfo sp;
	socklen_t siz;
#ifndef HAVE_SA_LEN
	size_t sa_len;
#endif

	/* First get the assoc id */
	siz = sizeof(sp);
	memset(&sp, 0, sizeof(sp));
#ifdef HAVE_SA_LEN
	memcpy((caddr_t)&sp.spinfo_address, sa, sa->sa_len);
#else
	switch (sa->sa_family) {
#ifdef INET
	case AF_INET:
		sa_len = sizeof(struct sockaddr_in);
		break;
#endif
#ifdef INET6
	case AF_INET6:
		sa_len = sizeof(struct sockaddr_in6);
		break;
#endif
	case AF_CONN:
		sa_len = sizeof(struct sockaddr_conn);
		break;
	default:
		sa_len = 0;
		break;
	}
	memcpy((caddr_t)&sp.spinfo_address, sa, sa_len);
#endif
	if (usrsctp_getsockopt(sock, IPPROTO_SCTP, SCTP_GET_PEER_ADDR_INFO, &sp, &siz) != 0) {
		/* We depend on the fact that 0 can never be returned */
		return ((sctp_assoc_t) 0);
	}
	return (sp.spinfo_assoc_id);
}


/* Taken from  /src/lib/libc/net/sctp_sys_calls.c
 * and modified for __Userspace__
 * calling sctp_generic_sendmsg from this function
 */
ssize_t
userspace_sctp_sendmsg(struct socket *so,
                       const void *data,
                       size_t len,
                       struct sockaddr *to,
                       socklen_t tolen,
                       uint32_t ppid,
                       uint32_t flags,
                       uint16_t stream_no,
                       uint32_t timetolive,
                       uint32_t context)
{
	struct sctp_sndrcvinfo sndrcvinfo, *sinfo = &sndrcvinfo;
	struct uio auio;
	struct iovec iov[1];

	memset(sinfo, 0, sizeof(struct sctp_sndrcvinfo));
	sinfo->sinfo_ppid = ppid;
	sinfo->sinfo_flags = flags;
	sinfo->sinfo_stream = stream_no;
	sinfo->sinfo_timetolive = timetolive;
	sinfo->sinfo_context = context;
	sinfo->sinfo_assoc_id = 0;


	/* Perform error checks on destination (to) */
	if (tolen > SOCK_MAXADDRLEN) {
		errno = ENAMETOOLONG;
		return (-1);
	}
	if ((tolen > 0) &&
	    ((to == NULL) || (tolen < (socklen_t)sizeof(struct sockaddr)))) {
		errno = EINVAL;
		return (-1);
	}
	if (data == NULL) {
		errno = EFAULT;
		return (-1);
	}
	/* Adding the following as part of defensive programming, in case the application
	   does not do it when preparing the destination address.*/
#ifdef HAVE_SA_LEN
	if (to != NULL) {
		to->sa_len = tolen;
	}
#endif

	iov[0].iov_base = (caddr_t)data;
	iov[0].iov_len = len;

	auio.uio_iov =  iov;
	auio.uio_iovcnt = 1;
	auio.uio_segflg = UIO_USERSPACE;
	auio.uio_rw = UIO_WRITE;
	auio.uio_offset = 0;			/* XXX */
	auio.uio_resid = len;
	errno = sctp_lower_sosend(so, to, &auio, NULL, NULL, 0, sinfo);
	if (errno == 0) {
		return (len - auio.uio_resid);
	} else {
		return (-1);
	}
}


ssize_t
usrsctp_sendv(struct socket *so,
              const void *data,
              size_t len,
              struct sockaddr *to,
              int addrcnt,
              void *info,
              socklen_t infolen,
              unsigned int infotype,
              int flags)
{
	struct sctp_sndrcvinfo sinfo;
	struct uio auio;
	struct iovec iov[1];
	int use_sinfo;
	sctp_assoc_t *assoc_id;

	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	if (data == NULL) {
		errno = EFAULT;
		return (-1);
	}
	memset(&sinfo, 0, sizeof(struct sctp_sndrcvinfo));
	assoc_id = NULL;
	use_sinfo = 0;
	switch (infotype) {
	case SCTP_SENDV_NOINFO:
		if ((infolen != 0) || (info != NULL)) {
			errno = EINVAL;
			return (-1);
		}
		break;
	case SCTP_SENDV_SNDINFO:
		if ((info == NULL) || (infolen != sizeof(struct sctp_sndinfo))) {
			errno = EINVAL;
			return (-1);
		}
		sinfo.sinfo_stream = ((struct sctp_sndinfo *)info)->snd_sid;
		sinfo.sinfo_flags = ((struct sctp_sndinfo *)info)->snd_flags;
		sinfo.sinfo_ppid = ((struct sctp_sndinfo *)info)->snd_ppid;
		sinfo.sinfo_context = ((struct sctp_sndinfo *)info)->snd_context;
		sinfo.sinfo_assoc_id = ((struct sctp_sndinfo *)info)->snd_assoc_id;
		assoc_id = &(((struct sctp_sndinfo *)info)->snd_assoc_id);
		use_sinfo = 1;
		break;
	case SCTP_SENDV_PRINFO:
		if ((info == NULL) || (infolen != sizeof(struct sctp_prinfo))) {
			errno = EINVAL;
			return (-1);
		}
		sinfo.sinfo_stream = 0;
		sinfo.sinfo_flags = PR_SCTP_POLICY(((struct sctp_prinfo *)info)->pr_policy);
		sinfo.sinfo_timetolive = ((struct sctp_prinfo *)info)->pr_value;
		use_sinfo = 1;
		break;
	case SCTP_SENDV_AUTHINFO:
		errno = EINVAL;
		return (-1);
	case SCTP_SENDV_SPA:
		if ((info == NULL) || (infolen != sizeof(struct sctp_sendv_spa))) {
			errno = EINVAL;
			return (-1);
		}
		if (((struct sctp_sendv_spa *)info)->sendv_flags & SCTP_SEND_SNDINFO_VALID) {
			sinfo.sinfo_stream = ((struct sctp_sendv_spa *)info)->sendv_sndinfo.snd_sid;
			sinfo.sinfo_flags = ((struct sctp_sendv_spa *)info)->sendv_sndinfo.snd_flags;
			sinfo.sinfo_ppid = ((struct sctp_sendv_spa *)info)->sendv_sndinfo.snd_ppid;
			sinfo.sinfo_context = ((struct sctp_sendv_spa *)info)->sendv_sndinfo.snd_context;
			sinfo.sinfo_assoc_id = ((struct sctp_sendv_spa *)info)->sendv_sndinfo.snd_assoc_id;
			assoc_id = &(((struct sctp_sendv_spa *)info)->sendv_sndinfo.snd_assoc_id);
		} else {
			sinfo.sinfo_flags = 0;
			sinfo.sinfo_stream = 0;
		}
		if (((struct sctp_sendv_spa *)info)->sendv_flags & SCTP_SEND_PRINFO_VALID) {
			sinfo.sinfo_flags |= PR_SCTP_POLICY(((struct sctp_sendv_spa *)info)->sendv_prinfo.pr_policy);
			sinfo.sinfo_timetolive = ((struct sctp_sendv_spa *)info)->sendv_prinfo.pr_value;
		}
		if (((struct sctp_sendv_spa *)info)->sendv_flags & SCTP_SEND_AUTHINFO_VALID) {
			errno = EINVAL;
			return (-1);
		}
		use_sinfo = 1;
		break;
	default:
		errno = EINVAL;
		return (-1);
	}

	/* Perform error checks on destination (to) */
	if (addrcnt > 1) {
		errno = EINVAL;
		return (-1);
	}

	iov[0].iov_base = (caddr_t)data;
	iov[0].iov_len = len;

	auio.uio_iov =  iov;
	auio.uio_iovcnt = 1;
	auio.uio_segflg = UIO_USERSPACE;
	auio.uio_rw = UIO_WRITE;
	auio.uio_offset = 0;			/* XXX */
	auio.uio_resid = len;
	errno = sctp_lower_sosend(so, to, &auio, NULL, NULL, flags, use_sinfo ? &sinfo : NULL);
	if (errno == 0) {
		if ((to != NULL) && (assoc_id != NULL)) {
			*assoc_id = usrsctp_getassocid(so, to);
		}
		return (len - auio.uio_resid);
	} else {
		return (-1);
	}
}


ssize_t
userspace_sctp_sendmbuf(struct socket *so,
    struct mbuf* mbufdata,
    size_t len,
    struct sockaddr *to,
    socklen_t tolen,
    uint32_t ppid,
    uint32_t flags,
    uint16_t stream_no,
    uint32_t timetolive,
    uint32_t context)
{

	struct sctp_sndrcvinfo sndrcvinfo, *sinfo = &sndrcvinfo;
	/*    struct uio auio;
	      struct iovec iov[1]; */
	int error = 0;
	int uflags = 0;
	ssize_t retval;

	sinfo->sinfo_ppid = ppid;
	sinfo->sinfo_flags = flags;
	sinfo->sinfo_stream = stream_no;
	sinfo->sinfo_timetolive = timetolive;
	sinfo->sinfo_context = context;
	sinfo->sinfo_assoc_id = 0;

	/* Perform error checks on destination (to) */
	if (tolen > SOCK_MAXADDRLEN){
		error = (ENAMETOOLONG);
		goto sendmsg_return;
	}
	if (tolen < (socklen_t)offsetof(struct sockaddr, sa_data)){
		error = (EINVAL);
		goto sendmsg_return;
	}
	/* Adding the following as part of defensive programming, in case the application
	   does not do it when preparing the destination address.*/
#ifdef HAVE_SA_LEN
	to->sa_len = tolen;
#endif

	error = sctp_lower_sosend(so, to, NULL/*uio*/,
	                         (struct mbuf *)mbufdata, (struct mbuf *)NULL,
	                         uflags, sinfo);
sendmsg_return:
	/* TODO: Needs a condition for non-blocking when error is EWOULDBLOCK */
	if (0 == error)
		retval = len;
	else if (error == EWOULDBLOCK) {
		errno = EWOULDBLOCK;
		retval = -1;
	} else {
		SCTP_PRINTF("%s: error = %d\n", __func__, error);
		errno = error;
		retval = -1;
	}
	return (retval);
}


/* taken from usr.lib/sctp_sys_calls.c and needed here */
#define        SCTP_SMALL_IOVEC_SIZE 2

/* Taken from  /src/lib/libc/net/sctp_sys_calls.c
 * and modified for __Userspace__
 * calling sctp_generic_recvmsg from this function
 */
ssize_t
userspace_sctp_recvmsg(struct socket *so,
    void *dbuf,
    size_t len,
    struct sockaddr *from,
    socklen_t *fromlenp,
    struct sctp_sndrcvinfo *sinfo,
    int *msg_flags)
{
	struct uio auio;
	struct iovec iov[SCTP_SMALL_IOVEC_SIZE];
	struct iovec *tiov;
	int iovlen = 1;
	int error = 0;
	ssize_t ulen;
	int i;
	socklen_t fromlen;

	iov[0].iov_base = dbuf;
	iov[0].iov_len = len;

	auio.uio_iov = iov;
	auio.uio_iovcnt = iovlen;
	auio.uio_segflg = UIO_USERSPACE;
	auio.uio_rw = UIO_READ;
	auio.uio_offset = 0;			/* XXX */
	auio.uio_resid = 0;
	tiov = iov;
	for (i = 0; i <iovlen; i++, tiov++) {
		if ((auio.uio_resid += tiov->iov_len) < 0) {
			error = EINVAL;
			SCTP_PRINTF("%s: error = %d\n", __func__, error);
			return (-1);
		}
	}
	ulen = auio.uio_resid;
	if (fromlenp != NULL) {
		fromlen = *fromlenp;
	} else {
		fromlen = 0;
	}
	error = sctp_sorecvmsg(so, &auio, (struct mbuf **)NULL,
		    from, fromlen, msg_flags,
		    (struct sctp_sndrcvinfo *)sinfo, 1);

	if (error) {
		if ((auio.uio_resid != ulen) &&
		    (error == EINTR ||
#if !defined(__NetBSD__)
		     error == ERESTART ||
#endif
		     error == EWOULDBLOCK)) {
			error = 0;
		}
	}
	if ((fromlenp != NULL) && (fromlen > 0) && (from != NULL)) {
		switch (from->sa_family) {
#if defined(INET)
		case AF_INET:
			*fromlenp = sizeof(struct sockaddr_in);
			break;
#endif
#if defined(INET6)
		case AF_INET6:
			*fromlenp = sizeof(struct sockaddr_in6);
			break;
#endif
		case AF_CONN:
			*fromlenp = sizeof(struct sockaddr_conn);
			break;
		default:
			*fromlenp = 0;
			break;
		}
		if (*fromlenp > fromlen) {
			*fromlenp = fromlen;
		}
	}
	if (error == 0) {
		/* ready return value */
		return (ulen - auio.uio_resid);
	} else {
		SCTP_PRINTF("%s: error = %d\n", __func__, error);
		return (-1);
	}
}

ssize_t
usrsctp_recvv(struct socket *so,
    void *dbuf,
    size_t len,
    struct sockaddr *from,
    socklen_t *fromlenp,
    void *info,
    socklen_t *infolen,
    unsigned int *infotype,
    int *msg_flags)
{
	struct uio auio;
	struct iovec iov[SCTP_SMALL_IOVEC_SIZE];
	struct iovec *tiov;
	int iovlen = 1;
	ssize_t ulen;
	int i;
	socklen_t fromlen;
	struct sctp_rcvinfo *rcv;
	struct sctp_recvv_rn *rn;
	struct sctp_extrcvinfo seinfo;

	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	iov[0].iov_base = dbuf;
	iov[0].iov_len = len;

	auio.uio_iov = iov;
	auio.uio_iovcnt = iovlen;
	auio.uio_segflg = UIO_USERSPACE;
	auio.uio_rw = UIO_READ;
	auio.uio_offset = 0;			/* XXX */
	auio.uio_resid = 0;
	tiov = iov;
	for (i = 0; i <iovlen; i++, tiov++) {
		if ((auio.uio_resid += tiov->iov_len) < 0) {
			errno = EINVAL;
			return (-1);
		}
	}
	ulen = auio.uio_resid;
	if (fromlenp != NULL) {
		fromlen = *fromlenp;
	} else {
		fromlen = 0;
	}
	errno = sctp_sorecvmsg(so, &auio, (struct mbuf **)NULL,
		    from, fromlen, msg_flags,
		    (struct sctp_sndrcvinfo *)&seinfo, 1);
	if (errno) {
		if ((auio.uio_resid != ulen) &&
		    (errno == EINTR ||
#if !defined(__NetBSD__)
		     errno == ERESTART ||
#endif
		     errno == EWOULDBLOCK)) {
			errno = 0;
		}
	}
	if (errno != 0) {
		goto out;
	}
	if ((*msg_flags & MSG_NOTIFICATION) == 0) {
		struct sctp_inpcb *inp;

		inp = (struct sctp_inpcb *)so->so_pcb;
		if (sctp_is_feature_on(inp, SCTP_PCB_FLAGS_RECVNXTINFO) &&
		    sctp_is_feature_on(inp, SCTP_PCB_FLAGS_RECVRCVINFO) &&
		    *infolen >= (socklen_t)sizeof(struct sctp_recvv_rn) &&
		    seinfo.sreinfo_next_flags & SCTP_NEXT_MSG_AVAIL) {
			rn = (struct sctp_recvv_rn *)info;
			rn->recvv_rcvinfo.rcv_sid = seinfo.sinfo_stream;
			rn->recvv_rcvinfo.rcv_ssn = seinfo.sinfo_ssn;
			rn->recvv_rcvinfo.rcv_flags = seinfo.sinfo_flags;
			rn->recvv_rcvinfo.rcv_ppid = seinfo.sinfo_ppid;
			rn->recvv_rcvinfo.rcv_context = seinfo.sinfo_context;
			rn->recvv_rcvinfo.rcv_tsn = seinfo.sinfo_tsn;
			rn->recvv_rcvinfo.rcv_cumtsn = seinfo.sinfo_cumtsn;
			rn->recvv_rcvinfo.rcv_assoc_id = seinfo.sinfo_assoc_id;
			rn->recvv_nxtinfo.nxt_sid = seinfo.sreinfo_next_stream;
			rn->recvv_nxtinfo.nxt_flags = 0;
			if (seinfo.sreinfo_next_flags & SCTP_NEXT_MSG_IS_UNORDERED) {
				rn->recvv_nxtinfo.nxt_flags |= SCTP_UNORDERED;
			}
			if (seinfo.sreinfo_next_flags & SCTP_NEXT_MSG_IS_NOTIFICATION) {
				rn->recvv_nxtinfo.nxt_flags |= SCTP_NOTIFICATION;
			}
			if (seinfo.sreinfo_next_flags & SCTP_NEXT_MSG_ISCOMPLETE) {
				rn->recvv_nxtinfo.nxt_flags |= SCTP_COMPLETE;
			}
			rn->recvv_nxtinfo.nxt_ppid = seinfo.sreinfo_next_ppid;
			rn->recvv_nxtinfo.nxt_length = seinfo.sreinfo_next_length;
			rn->recvv_nxtinfo.nxt_assoc_id = seinfo.sreinfo_next_aid;
			*infolen = (socklen_t)sizeof(struct sctp_recvv_rn);
			*infotype = SCTP_RECVV_RN;
		} else if (sctp_is_feature_on(inp, SCTP_PCB_FLAGS_RECVRCVINFO) &&
		           *infolen >= (socklen_t)sizeof(struct sctp_rcvinfo)) {
			rcv = (struct sctp_rcvinfo *)info;
			rcv->rcv_sid = seinfo.sinfo_stream;
			rcv->rcv_ssn = seinfo.sinfo_ssn;
			rcv->rcv_flags = seinfo.sinfo_flags;
			rcv->rcv_ppid = seinfo.sinfo_ppid;
			rcv->rcv_context = seinfo.sinfo_context;
			rcv->rcv_tsn = seinfo.sinfo_tsn;
			rcv->rcv_cumtsn = seinfo.sinfo_cumtsn;
			rcv->rcv_assoc_id = seinfo.sinfo_assoc_id;
			*infolen = (socklen_t)sizeof(struct sctp_rcvinfo);
			*infotype = SCTP_RECVV_RCVINFO;
		} else {
			*infotype = SCTP_RECVV_NOINFO;
			*infolen = 0;
		}
	}
	if ((fromlenp != NULL) &&
	    (fromlen > 0) &&
	    (from != NULL) &&
	    (ulen > auio.uio_resid)) {
		switch (from->sa_family) {
#if defined(INET)
		case AF_INET:
			*fromlenp = sizeof(struct sockaddr_in);
			break;
#endif
#if defined(INET6)
		case AF_INET6:
			*fromlenp = sizeof(struct sockaddr_in6);
			break;
#endif
		case AF_CONN:
			*fromlenp = sizeof(struct sockaddr_conn);
			break;
		default:
			*fromlenp = 0;
			break;
		}
		if (*fromlenp > fromlen) {
			*fromlenp = fromlen;
		}
	}
out:
	if (errno == 0) {
		/* ready return value */
		return (ulen - auio.uio_resid);
	} else {
		return (-1);
	}
}




/* Taken from  /src/sys/kern/uipc_socket.c
 * and modified for __Userspace__
 * socreate returns a socket.  The socket should be
 * closed with soclose().
 */
int
socreate(int dom, struct socket **aso, int type, int proto)
{
	struct socket *so;
	int error;

	if ((dom != AF_CONN) && (dom != AF_INET) && (dom != AF_INET6)) {
		return (EINVAL);
	}
	if ((type != SOCK_STREAM) && (type != SOCK_SEQPACKET)) {
		return (EINVAL);
	}
	if (proto != IPPROTO_SCTP) {
		return (EINVAL);
	}

	so = soalloc();
	if (so == NULL) {
		return (ENOBUFS);
	}

	/*
	 * so_incomp represents a queue of connections that
	 * must be completed at protocol level before being
	 * returned. so_comp field heads a list of sockets
	 * that are ready to be returned to the listening process
	 *__Userspace__ These queues are being used at a number of places like accept etc.
	 */
	TAILQ_INIT(&so->so_incomp);
	TAILQ_INIT(&so->so_comp);
	so->so_type = type;
	so->so_count = 1;
	so->so_dom = dom;
	/*
	 * Auto-sizing of socket buffers is managed by the protocols and
	 * the appropriate flags must be set in the pru_attach function.
	 * For __Userspace__ The pru_attach function in this case is sctp_attach.
	 */
	switch (dom) {
#if defined(INET)
	case AF_INET:
		error = sctp_attach(so, proto, SCTP_DEFAULT_VRFID);
		break;
#endif
#if defined(INET6)
	case AF_INET6:
		error = sctp6_attach(so, proto, SCTP_DEFAULT_VRFID);
		break;
#endif
	case AF_CONN:
		error = sctpconn_attach(so, proto, SCTP_DEFAULT_VRFID);
		break;
	default:
		error = EAFNOSUPPORT;
		break;
	}
	if (error) {
		KASSERT(so->so_count == 1, ("socreate: so_count %d", so->so_count));
		so->so_count = 0;
		sodealloc(so);
		return (error);
	}
	*aso = so;
	return (0);
}


/* Taken from  /src/sys/kern/uipc_syscalls.c
 * and modified for __Userspace__
 * Removing struct thread td.
 */
struct socket *
userspace_socket(int domain, int type, int protocol)
{
	struct socket *so = NULL;

	errno = socreate(domain, &so, type, protocol);
	if (errno) {
		return (NULL);
	}
	/*
	 * The original socket call returns the file descriptor fd.
	 * td->td_retval[0] = fd.
	 * We are returning struct socket *so.
	 */
	return (so);
}

struct socket *
usrsctp_socket(int domain, int type, int protocol,
	       int (*receive_cb)(struct socket *sock, union sctp_sockstore addr, void *data,
                                 size_t datalen, struct sctp_rcvinfo, int flags, void *ulp_info),
	       int (*send_cb)(struct socket *sock, uint32_t sb_free, void *ulp_info),
	       uint32_t sb_threshold,
	       void *ulp_info)
{
	struct socket *so = NULL;

	if ((protocol == IPPROTO_SCTP) && (SCTP_BASE_VAR(sctp_pcb_initialized) == 0)) {
		errno = EPROTONOSUPPORT;
		return (NULL);
	}
	if ((receive_cb == NULL) &&
	    ((send_cb != NULL) || (sb_threshold != 0) || (ulp_info != NULL))) {
		errno = EINVAL;
		return (NULL);
	}
	if ((domain == AF_CONN) && (SCTP_BASE_VAR(conn_output) == NULL)) {
		errno = EAFNOSUPPORT;
		return (NULL);
	}
	errno = socreate(domain, &so, type, protocol);
	if (errno) {
		return (NULL);
	}
	/*
	 * The original socket call returns the file descriptor fd.
	 * td->td_retval[0] = fd.
	 * We are returning struct socket *so.
	 */
	register_recv_cb(so, receive_cb);
	register_send_cb(so, sb_threshold, send_cb);
	register_ulp_info(so, ulp_info);
	return (so);
}


u_long	sb_max = SB_MAX;
u_long sb_max_adj =
       SB_MAX * MCLBYTES / (MSIZE + MCLBYTES); /* adjusted sb_max */

static	u_long sb_efficiency = 8;	/* parameter for sbreserve() */

/*
 * Allot mbufs to a sockbuf.  Attempt to scale mbmax so that mbcnt doesn't
 * become limiting if buffering efficiency is near the normal case.
 */
int
sbreserve_locked(struct sockbuf *sb, u_long cc, struct socket *so)
{
	SOCKBUF_LOCK_ASSERT(sb);
	sb->sb_mbmax = (u_int)min(cc * sb_efficiency, sb_max);
	sb->sb_hiwat = (u_int)cc;
	if (sb->sb_lowat > (int)sb->sb_hiwat)
		sb->sb_lowat = (int)sb->sb_hiwat;
	return (1);
}

static int
sbreserve(struct sockbuf *sb, u_long cc, struct socket *so)
{
	int error;

	SOCKBUF_LOCK(sb);
	error = sbreserve_locked(sb, cc, so);
	SOCKBUF_UNLOCK(sb);
	return (error);
}

int
soreserve(struct socket *so, u_long sndcc, u_long rcvcc)
{
	SOCKBUF_LOCK(&so->so_snd);
	SOCKBUF_LOCK(&so->so_rcv);
	so->so_snd.sb_hiwat = (uint32_t)sndcc;
	so->so_rcv.sb_hiwat = (uint32_t)rcvcc;

	if (sbreserve_locked(&so->so_snd, sndcc, so) == 0) {
		goto bad;
	}
	if (sbreserve_locked(&so->so_rcv, rcvcc, so) == 0) {
		goto bad;
	}
	if (so->so_rcv.sb_lowat == 0)
		so->so_rcv.sb_lowat = 1;
	if (so->so_snd.sb_lowat == 0)
		so->so_snd.sb_lowat = MCLBYTES;
	if (so->so_snd.sb_lowat > (int)so->so_snd.sb_hiwat)
		so->so_snd.sb_lowat = (int)so->so_snd.sb_hiwat;
	SOCKBUF_UNLOCK(&so->so_rcv);
	SOCKBUF_UNLOCK(&so->so_snd);
	return (0);

 bad:
	SOCKBUF_UNLOCK(&so->so_rcv);
	SOCKBUF_UNLOCK(&so->so_snd);
	return (ENOBUFS);
}


/* Taken from  /src/sys/kern/uipc_sockbuf.c
 * and modified for __Userspace__
 */

void
sowakeup(struct socket *so, struct sockbuf *sb)
{

	SOCKBUF_LOCK_ASSERT(sb);

	sb->sb_flags &= ~SB_SEL;
	if (sb->sb_flags & SB_WAIT) {
		sb->sb_flags &= ~SB_WAIT;
#if defined(_WIN32)
		WakeAllConditionVariable(&(sb)->sb_cond);
#else
		pthread_cond_broadcast(&(sb)->sb_cond);
#endif
	}
	SOCKBUF_UNLOCK(sb);
}


/* Taken from  /src/sys/kern/uipc_socket.c
 * and modified for __Userspace__
 */

int
sobind(struct socket *so, struct sockaddr *nam)
{
	switch (nam->sa_family) {
#if defined(INET)
	case AF_INET:
		return (sctp_bind(so, nam));
#endif
#if defined(INET6)
	case AF_INET6:
		return (sctp6_bind(so, nam, NULL));
#endif
	case AF_CONN:
		return (sctpconn_bind(so, nam));
	default:
		return EAFNOSUPPORT;
	}
}

/* Taken from  /src/sys/kern/uipc_syscalls.c
 * and modified for __Userspace__
 */

int
usrsctp_bind(struct socket *so, struct sockaddr *name, int namelen)
{
	struct sockaddr *sa;

	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	if ((errno = getsockaddr(&sa, (caddr_t)name, namelen)) != 0)
		return (-1);

	errno = sobind(so, sa);
	FREE(sa, M_SONAME);
	if (errno) {
		return (-1);
	} else {
		return (0);
	}
}

int
userspace_bind(struct socket *so, struct sockaddr *name, int namelen)
{
	return (usrsctp_bind(so, name, namelen));
}

/* Taken from  /src/sys/kern/uipc_socket.c
 * and modified for __Userspace__
 */

int
solisten(struct socket *so, int backlog)
{
	if (so == NULL) {
		return (EBADF);
	} else {
		return (sctp_listen(so, backlog, NULL));
	}
}


int
solisten_proto_check(struct socket *so)
{

	SOCK_LOCK_ASSERT(so);

	if (so->so_state & (SS_ISCONNECTED | SS_ISCONNECTING |
	    SS_ISDISCONNECTING))
		return (EINVAL);
	return (0);
}

static int somaxconn = SOMAXCONN;

void
solisten_proto(struct socket *so, int backlog)
{

	SOCK_LOCK_ASSERT(so);

	if (backlog < 0 || backlog > somaxconn)
		backlog = somaxconn;
	so->so_qlimit = backlog;
	so->so_options |= SCTP_SO_ACCEPTCONN;
}




/* Taken from  /src/sys/kern/uipc_syscalls.c
 * and modified for __Userspace__
 */

int
usrsctp_listen(struct socket *so, int backlog)
{
	errno = solisten(so, backlog);
	if (errno) {
		return (-1);
	} else {
		return (0);
	}
}

int
userspace_listen(struct socket *so, int backlog)
{
	return (usrsctp_listen(so, backlog));
}

/* Taken from  /src/sys/kern/uipc_socket.c
 * and modified for __Userspace__
 */

int
soaccept(struct socket *so, struct sockaddr **nam)
{
	int error;

	SOCK_LOCK(so);
	KASSERT((so->so_state & SS_NOFDREF) != 0, ("soaccept: !NOFDREF"));
	so->so_state &= ~SS_NOFDREF;
	SOCK_UNLOCK(so);
	error = sctp_accept(so, nam);
	return (error);
}



/* Taken from  /src/sys/kern/uipc_syscalls.c
 * kern_accept modified for __Userspace__
 */
int
user_accept(struct socket *head,  struct sockaddr **name, socklen_t *namelen, struct socket **ptr_accept_ret_sock)
{
	struct sockaddr *sa = NULL;
	int error;
	struct socket *so = NULL;


	if (name) {
		*name = NULL;
	}

	if ((head->so_options & SCTP_SO_ACCEPTCONN) == 0) {
		error = EINVAL;
		goto done;
	}

	ACCEPT_LOCK();
	if ((head->so_state & SS_NBIO) && TAILQ_EMPTY(&head->so_comp)) {
		ACCEPT_UNLOCK();
		error = EWOULDBLOCK;
		goto noconnection;
	}
	while (TAILQ_EMPTY(&head->so_comp) && head->so_error == 0) {
		if (head->so_rcv.sb_state & SBS_CANTRCVMORE) {
			head->so_error = ECONNABORTED;
			break;
		}
#if defined(_WIN32)
		if (SleepConditionVariableCS(&accept_cond, &accept_mtx, INFINITE))
			error = 0;
		else
			error = GetLastError();
#else
		error = pthread_cond_wait(&accept_cond, &accept_mtx);
#endif
		if (error) {
			ACCEPT_UNLOCK();
			goto noconnection;
		}
	}
	if (head->so_error) {
		error = head->so_error;
		head->so_error = 0;
		ACCEPT_UNLOCK();
		goto noconnection;
	}
	so = TAILQ_FIRST(&head->so_comp);
	KASSERT(!(so->so_qstate & SQ_INCOMP), ("accept1: so SQ_INCOMP"));
	KASSERT(so->so_qstate & SQ_COMP, ("accept1: so not SQ_COMP"));

	/*
	 * Before changing the flags on the socket, we have to bump the
	 * reference count.  Otherwise, if the protocol calls sofree(),
	 * the socket will be released due to a zero refcount.
	 */
	SOCK_LOCK(so);			/* soref() and so_state update */
	soref(so);			/* file descriptor reference */

	TAILQ_REMOVE(&head->so_comp, so, so_list);
	head->so_qlen--;
	so->so_state |= (head->so_state & SS_NBIO);
	so->so_qstate &= ~SQ_COMP;
	so->so_head = NULL;
	SOCK_UNLOCK(so);
	ACCEPT_UNLOCK();


	/*
	 * The original accept returns fd value via td->td_retval[0] = fd;
	 * we will return the socket for accepted connection.
	 */

	error = soaccept(so, &sa);
	if (error) {
		/*
		 * return a namelen of zero for older code which might
		 * ignore the return value from accept.
		 */
		if (name)
			*namelen = 0;
		goto noconnection;
	}
	if (sa == NULL) {
		if (name)
			*namelen = 0;
		goto done;
	}
	if (name) {
#ifdef HAVE_SA_LEN
		/* check sa_len before it is destroyed */
		if (*namelen > sa->sa_len) {
			*namelen = sa->sa_len;
		}
#else
		socklen_t sa_len;

		switch (sa->sa_family) {
#ifdef INET
		case AF_INET:
			sa_len = sizeof(struct sockaddr_in);
			break;
#endif
#ifdef INET6
		case AF_INET6:
			sa_len = sizeof(struct sockaddr_in6);
			break;
#endif
		case AF_CONN:
			sa_len = sizeof(struct sockaddr_conn);
			break;
		default:
			sa_len = 0;
			break;
		}
		if (*namelen > sa_len) {
			*namelen = sa_len;
		}
#endif
		*name = sa;
		sa = NULL;
	}
noconnection:
	if (sa) {
		FREE(sa, M_SONAME);
	}

done:
	*ptr_accept_ret_sock = so;
	return (error);
}



/* Taken from  /src/sys/kern/uipc_syscalls.c
 * and modified for __Userspace__
 */
/*
 * accept1()
 */
static int
accept1(struct socket *so, struct sockaddr *aname, socklen_t *anamelen, struct socket **ptr_accept_ret_sock)
{
	struct sockaddr *name;
	socklen_t namelen;
	int error;

	if (so == NULL) {
		return (EBADF);
	}
	if (aname == NULL) {
		return (user_accept(so, NULL, NULL, ptr_accept_ret_sock));
	}

	error = copyin(anamelen, &namelen, sizeof (namelen));
	if (error)
		return (error);

	error = user_accept(so, &name, &namelen, ptr_accept_ret_sock);

	/*
	 * return a namelen of zero for older code which might
	 * ignore the return value from accept.
	 */
	if (error) {
		(void) copyout(&namelen,
		    anamelen, sizeof(*anamelen));
		return (error);
	}

	if (error == 0 && name != NULL) {
		error = copyout(name, aname, namelen);
	}
	if (error == 0) {
		error = copyout(&namelen, anamelen, sizeof(namelen));
	}

	if (name) {
		FREE(name, M_SONAME);
	}
	return (error);
}

struct socket *
usrsctp_accept(struct socket *so, struct sockaddr *aname, socklen_t *anamelen)
{
	struct socket *accept_return_sock = NULL;

	errno = accept1(so, aname, anamelen, &accept_return_sock);
	if (errno) {
		return (NULL);
	} else {
		return (accept_return_sock);
	}
}

struct socket *
userspace_accept(struct socket *so, struct sockaddr *aname, socklen_t *anamelen)
{
	return (usrsctp_accept(so, aname, anamelen));
}

struct socket *
usrsctp_peeloff(struct socket *head, sctp_assoc_t id)
{
	struct socket *so;

	if ((errno = sctp_can_peel_off(head, id)) != 0) {
		return (NULL);
	}
	if ((so = sonewconn(head, SS_ISCONNECTED)) == NULL) {
		return (NULL);
	}
	ACCEPT_LOCK();
	SOCK_LOCK(so);
	soref(so);
	TAILQ_REMOVE(&head->so_comp, so, so_list);
	head->so_qlen--;
	so->so_state |= (head->so_state & SS_NBIO);
	so->so_qstate &= ~SQ_COMP;
	so->so_head = NULL;
	SOCK_UNLOCK(so);
	ACCEPT_UNLOCK();
	if ((errno = sctp_do_peeloff(head, so, id)) != 0) {
		so->so_count = 0;
		sodealloc(so);
		return (NULL);
	}
	return (so);
}

int
sodisconnect(struct socket *so)
{
	int error;

	if ((so->so_state & SS_ISCONNECTED) == 0)
		return (ENOTCONN);
	if (so->so_state & SS_ISDISCONNECTING)
		return (EALREADY);
	error = sctp_disconnect(so);
	return (error);
}

int
usrsctp_set_non_blocking(struct socket *so, int onoff)
{
	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	SOCK_LOCK(so);
	if (onoff != 0) {
		so->so_state |= SS_NBIO;
	} else {
		so->so_state &= ~SS_NBIO;
	}
	SOCK_UNLOCK(so);
	return (0);
}

int
usrsctp_get_non_blocking(struct socket *so)
{
	int result;

	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	SOCK_LOCK(so);
	if (so->so_state & SS_NBIO) {
		result = 1;
	} else {
		result = 0;
	}
	SOCK_UNLOCK(so);
	return (result);
}

int
soconnect(struct socket *so, struct sockaddr *nam)
{
	int error;

	if (so->so_options & SCTP_SO_ACCEPTCONN)
		return (EOPNOTSUPP);
	/*
	 * If protocol is connection-based, can only connect once.
	 * Otherwise, if connected, try to disconnect first.  This allows
	 * user to disconnect by connecting to, e.g., a null address.
	 */
	if (so->so_state & (SS_ISCONNECTED|SS_ISCONNECTING) && (sodisconnect(so) != 0)) {
		error = EISCONN;
	} else {
		/*
		 * Prevent accumulated error from previous connection from
		 * biting us.
		 */
		so->so_error = 0;
		switch (nam->sa_family) {
#if defined(INET)
		case AF_INET:
			error = sctp_connect(so, nam);
			break;
#endif
#if defined(INET6)
		case AF_INET6:
			error = sctp6_connect(so, nam);
			break;
#endif
		case AF_CONN:
			error = sctpconn_connect(so, nam);
			break;
		default:
			error = EAFNOSUPPORT;
		}
	}

	return (error);
}



int user_connect(struct socket *so, struct sockaddr *sa)
{
	int error;
	int interrupted = 0;

	if (so == NULL) {
		error = EBADF;
		goto done1;
	}
	if (so->so_state & SS_ISCONNECTING) {
		error = EALREADY;
		goto done1;
	}

	error = soconnect(so, sa);
	if (error) {
		goto bad;
	}
	if ((so->so_state & SS_NBIO) && (so->so_state & SS_ISCONNECTING)) {
		error = EINPROGRESS;
		goto done1;
	}

	SOCK_LOCK(so);
	while ((so->so_state & SS_ISCONNECTING) && so->so_error == 0) {
#if defined(_WIN32)
		if (SleepConditionVariableCS(SOCK_COND(so), SOCK_MTX(so), INFINITE))
			error = 0;
		else
			error = -1;
#else
		error = pthread_cond_wait(SOCK_COND(so), SOCK_MTX(so));
#endif
		if (error) {
#if defined(__NetBSD__)
			if (error == EINTR) {
#else
			if (error == EINTR || error == ERESTART) {
#endif
				interrupted = 1;
			}
			break;
		}
	}
	if (error == 0) {
		error = so->so_error;
		so->so_error = 0;
	}
	SOCK_UNLOCK(so);

bad:
	if (!interrupted) {
		so->so_state &= ~SS_ISCONNECTING;
	}
#if !defined(__NetBSD__)
	if (error == ERESTART) {
		error = EINTR;
	}
#endif
done1:
	return (error);
}

int usrsctp_connect(struct socket *so, struct sockaddr *name, int namelen)
{
	struct sockaddr *sa = NULL;

	errno = getsockaddr(&sa, (caddr_t)name, namelen);
	if (errno)
		return (-1);

	errno = user_connect(so, sa);
	FREE(sa, M_SONAME);
	if (errno) {
		return (-1);
	} else {
		return (0);
	}
}

int userspace_connect(struct socket *so, struct sockaddr *name, int namelen)
{
	return (usrsctp_connect(so, name, namelen));
}

#define SCTP_STACK_BUF_SIZE         2048

void
usrsctp_close(struct socket *so) {
	if (so != NULL) {
		if (so->so_options & SCTP_SO_ACCEPTCONN) {
			struct socket *sp;

			ACCEPT_LOCK();
			while ((sp = TAILQ_FIRST(&so->so_comp)) != NULL) {
				TAILQ_REMOVE(&so->so_comp, sp, so_list);
				so->so_qlen--;
				sp->so_qstate &= ~SQ_COMP;
				sp->so_head = NULL;
				ACCEPT_UNLOCK();
				soabort(sp);
				ACCEPT_LOCK();
			}
			ACCEPT_UNLOCK();
		}
		ACCEPT_LOCK();
		SOCK_LOCK(so);
		sorele(so);
	}
}

void
userspace_close(struct socket *so)
{
	usrsctp_close(so);
}

int
usrsctp_shutdown(struct socket *so, int how)
{
	if (!(how == SHUT_RD || how == SHUT_WR || how == SHUT_RDWR)) {
		errno = EINVAL;
		return (-1);
	}
	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	sctp_flush(so, how);
	if (how != SHUT_WR)
		 socantrcvmore(so);
	if (how != SHUT_RD) {
		errno = sctp_shutdown(so);
		if (errno) {
			return (-1);
		} else {
			return (0);
		}
	}
	return (0);
}

int
userspace_shutdown(struct socket *so, int how)
{
	return (usrsctp_shutdown(so, how));
}

int
usrsctp_finish(void)
{
	if (SCTP_BASE_VAR(sctp_pcb_initialized) == 0) {
		return (0);
	}
	if (SCTP_INP_INFO_TRYLOCK()) {
		if (!LIST_EMPTY(&SCTP_BASE_INFO(listhead))) {
			SCTP_INP_INFO_RUNLOCK();
			return (-1);
		}
		SCTP_INP_INFO_RUNLOCK();
	} else {
		return (-1);
	}
	sctp_finish();
#if defined(_WIN32)
	DeleteConditionVariable(&accept_cond);
	DeleteCriticalSection(&accept_mtx);
#if defined(INET) || defined(INET6)
	WSACleanup();
#endif
#else
	pthread_cond_destroy(&accept_cond);
	pthread_mutex_destroy(&accept_mtx);
#endif
	return (0);
}

int
userspace_finish(void)
{
	return (usrsctp_finish());
}

/* needed from sctp_usrreq.c */
int
sctp_setopt(struct socket *so, int optname, void *optval, size_t optsize, void *p);

int
usrsctp_setsockopt(struct socket *so, int level, int option_name,
                   const void *option_value, socklen_t option_len)
{
	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	switch (level) {
	case SOL_SOCKET:
	{
		switch (option_name) {
		case SO_RCVBUF:
			if (option_len < (socklen_t)sizeof(int)) {
				errno = EINVAL;
				return (-1);
			} else {
				int *buf_size;

				buf_size = (int *)option_value;
				if (*buf_size < 1) {
					errno = EINVAL;
					return (-1);
				}
				sbreserve(&so->so_rcv, (u_long)*buf_size, so);
				return (0);
			}
			break;
		case SO_SNDBUF:
			if (option_len < (socklen_t)sizeof(int)) {
				errno = EINVAL;
				return (-1);
			} else {
				int *buf_size;

				buf_size = (int *)option_value;
				if (*buf_size < 1) {
					errno = EINVAL;
					return (-1);
				}
				sbreserve(&so->so_snd, (u_long)*buf_size, so);
				return (0);
			}
			break;
		case SO_LINGER:
			if (option_len < (socklen_t)sizeof(struct linger)) {
				errno = EINVAL;
				return (-1);
			} else {
				struct linger *l;

				l = (struct linger *)option_value;
				so->so_linger = l->l_linger;
				if (l->l_onoff) {
					so->so_options |= SCTP_SO_LINGER;
				} else {
					so->so_options &= ~SCTP_SO_LINGER;
				}
				return (0);
			}
		default:
			errno = EINVAL;
			return (-1);
		}
	}
	case IPPROTO_SCTP:
		errno = sctp_setopt(so, option_name, (void *) option_value, (size_t)option_len, NULL);
		if (errno) {
			return (-1);
		} else {
			return (0);
		}
	default:
		errno = ENOPROTOOPT;
		return (-1);
	}
}

int
userspace_setsockopt(struct socket *so, int level, int option_name,
                     const void *option_value, socklen_t option_len)
{
	return (usrsctp_setsockopt(so, level, option_name, option_value, option_len));
}

/* needed from sctp_usrreq.c */
int
sctp_getopt(struct socket *so, int optname, void *optval, size_t *optsize,
	    void *p);

int
usrsctp_getsockopt(struct socket *so, int level, int option_name,
                   void *option_value, socklen_t *option_len)
{
	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}
	if (option_len == NULL) {
		errno = EFAULT;
		return (-1);
	}
	switch (level) {
	case SOL_SOCKET:
		switch (option_name) {
		case SO_RCVBUF:
			if (*option_len < (socklen_t)sizeof(int)) {
				errno = EINVAL;
				return (-1);
			} else {
				int *buf_size;

				buf_size = (int *)option_value;
				*buf_size = so->so_rcv.sb_hiwat;
				*option_len = (socklen_t)sizeof(int);
				return (0);
			}
			break;
		case SO_SNDBUF:
			if (*option_len < (socklen_t)sizeof(int)) {
				errno = EINVAL;
				return (-1);
			} else {
				int *buf_size;

				buf_size = (int *)option_value;
				*buf_size = so->so_snd.sb_hiwat;
				*option_len = (socklen_t)sizeof(int);
				return (0);
			}
			break;
		case SO_LINGER:
			if (*option_len < (socklen_t)sizeof(struct linger)) {
				errno = EINVAL;
				return (-1);
			} else {
				struct linger *l;

				l = (struct linger *)option_value;
				l->l_linger = so->so_linger;
				if (so->so_options & SCTP_SO_LINGER) {
					l->l_onoff = 1;
				} else {
					l->l_onoff = 0;
				}
				*option_len = (socklen_t)sizeof(struct linger);
				return (0);
			}
			break;
		case SO_ERROR:
			if (*option_len < (socklen_t)sizeof(int)) {
				errno = EINVAL;
				return (-1);
			} else {
				int *intval;

				intval = (int *)option_value;
				*intval = so->so_error;
				*option_len = (socklen_t)sizeof(int);
				return (0);
			}
			break;
		default:
			errno = EINVAL;
			return (-1);
		}
	case IPPROTO_SCTP:
	{
		size_t len;

		len = (size_t)*option_len;
		errno = sctp_getopt(so, option_name, option_value, &len, NULL);
		*option_len = (socklen_t)len;
		if (errno) {
			return (-1);
		} else {
			return (0);
		}
	}
	default:
		errno = ENOPROTOOPT;
		return (-1);
	}
}

int
userspace_getsockopt(struct socket *so, int level, int option_name,
                     void *option_value, socklen_t *option_len)
{
	return (usrsctp_getsockopt(so, level, option_name, option_value, option_len));
}

int
usrsctp_opt_info(struct socket *so, sctp_assoc_t id, int opt, void *arg, socklen_t *size)
{
	if (arg == NULL) {
		errno = EINVAL;
		return (-1);
	}
	if ((id == SCTP_CURRENT_ASSOC) ||
	    (id == SCTP_ALL_ASSOC)) {
		errno = EINVAL;
		return (-1);
	}
	switch (opt) {
	case SCTP_RTOINFO:
		((struct sctp_rtoinfo *)arg)->srto_assoc_id = id;
		break;
	case SCTP_ASSOCINFO:
		((struct sctp_assocparams *)arg)->sasoc_assoc_id = id;
		break;
	case SCTP_DEFAULT_SEND_PARAM:
		((struct sctp_assocparams *)arg)->sasoc_assoc_id = id;
		break;
	case SCTP_PRIMARY_ADDR:
		((struct sctp_setprim *)arg)->ssp_assoc_id = id;
		break;
	case SCTP_PEER_ADDR_PARAMS:
		((struct sctp_paddrparams *)arg)->spp_assoc_id = id;
		break;
	case SCTP_MAXSEG:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_AUTH_KEY:
		((struct sctp_authkey *)arg)->sca_assoc_id = id;
		break;
	case SCTP_AUTH_ACTIVE_KEY:
		((struct sctp_authkeyid *)arg)->scact_assoc_id = id;
		break;
	case SCTP_DELAYED_SACK:
		((struct sctp_sack_info *)arg)->sack_assoc_id = id;
		break;
	case SCTP_CONTEXT:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_STATUS:
		((struct sctp_status *)arg)->sstat_assoc_id = id;
		break;
	case SCTP_GET_PEER_ADDR_INFO:
		((struct sctp_paddrinfo *)arg)->spinfo_assoc_id = id;
		break;
	case SCTP_PEER_AUTH_CHUNKS:
		((struct sctp_authchunks *)arg)->gauth_assoc_id = id;
		break;
	case SCTP_LOCAL_AUTH_CHUNKS:
		((struct sctp_authchunks *)arg)->gauth_assoc_id = id;
		break;
	case SCTP_TIMEOUTS:
		((struct sctp_timeouts *)arg)->stimo_assoc_id = id;
		break;
	case SCTP_EVENT:
		((struct sctp_event *)arg)->se_assoc_id = id;
		break;
	case SCTP_DEFAULT_SNDINFO:
		((struct sctp_sndinfo *)arg)->snd_assoc_id = id;
		break;
	case SCTP_DEFAULT_PRINFO:
		((struct sctp_default_prinfo *)arg)->pr_assoc_id = id;
		break;
	case SCTP_PEER_ADDR_THLDS:
		((struct sctp_paddrthlds *)arg)->spt_assoc_id = id;
		break;
	case SCTP_REMOTE_UDP_ENCAPS_PORT:
		((struct sctp_udpencaps *)arg)->sue_assoc_id = id;
		break;
	case SCTP_ECN_SUPPORTED:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_PR_SUPPORTED:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_AUTH_SUPPORTED:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_ASCONF_SUPPORTED:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_RECONFIG_SUPPORTED:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_NRSACK_SUPPORTED:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_PKTDROP_SUPPORTED:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_MAX_BURST:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_ENABLE_STREAM_RESET:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	case SCTP_PR_STREAM_STATUS:
		((struct sctp_prstatus *)arg)->sprstat_assoc_id = id;
		break;
	case SCTP_PR_ASSOC_STATUS:
		((struct sctp_prstatus *)arg)->sprstat_assoc_id = id;
		break;
	case SCTP_MAX_CWND:
		((struct sctp_assoc_value *)arg)->assoc_id = id;
		break;
	default:
		break;
	}
	return (usrsctp_getsockopt(so, IPPROTO_SCTP, opt, arg, size));
}

int
usrsctp_set_ulpinfo(struct socket *so, void *ulp_info)
{
	return (register_ulp_info(so, ulp_info));
}


int
usrsctp_get_ulpinfo(struct socket *so, void **pulp_info)
{
	return (retrieve_ulp_info(so, pulp_info));
}

int
usrsctp_bindx(struct socket *so, struct sockaddr *addrs, int addrcnt, int flags)
{
	struct sockaddr *sa;
#ifdef INET
	struct sockaddr_in *sin;
#endif
#ifdef INET6
	struct sockaddr_in6 *sin6;
#endif
	int i;
#if defined(INET) || defined(INET6)
	uint16_t sport;
	bool fix_port;
#endif

	/* validate the flags */
	if ((flags != SCTP_BINDX_ADD_ADDR) &&
	    (flags != SCTP_BINDX_REM_ADDR)) {
		errno = EFAULT;
		return (-1);
	}
	/* validate the address count and list */
	if ((addrcnt <= 0) || (addrs == NULL)) {
		errno = EINVAL;
		return (-1);
	}
#if defined(INET) || defined(INET6)
	sport = 0;
	fix_port = false;
#endif
	/* First pre-screen the addresses */
	sa = addrs;
	for (i = 0; i < addrcnt; i++) {
		switch (sa->sa_family) {
#ifdef INET
		case AF_INET:
#ifdef HAVE_SA_LEN
			if (sa->sa_len != sizeof(struct sockaddr_in)) {
				errno = EINVAL;
				return (-1);
			}
#endif
			sin = (struct sockaddr_in *)sa;
			if (sin->sin_port) {
				/* non-zero port, check or save */
				if (sport) {
					/* Check against our port */
					if (sport != sin->sin_port) {
						errno = EINVAL;
						return (-1);
					}
				} else {
					/* save off the port */
					sport = sin->sin_port;
					fix_port = (i > 0);
				}
			}
#ifndef HAVE_SA_LEN
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_in));
#endif
			break;
#endif
#ifdef INET6
		case AF_INET6:
#ifdef HAVE_SA_LEN
			if (sa->sa_len != sizeof(struct sockaddr_in6)) {
				errno = EINVAL;
				return (-1);
			}
#endif
			sin6 = (struct sockaddr_in6 *)sa;
			if (sin6->sin6_port) {
				/* non-zero port, check or save */
				if (sport) {
					/* Check against our port */
					if (sport != sin6->sin6_port) {
						errno = EINVAL;
						return (-1);
					}
				} else {
					/* save off the port */
					sport = sin6->sin6_port;
					fix_port = (i > 0);
				}
			}
#ifndef HAVE_SA_LEN
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_in6));
#endif
			break;
#endif
		default:
			/* Invalid address family specified. */
			errno = EAFNOSUPPORT;
			return (-1);
		}
#ifdef HAVE_SA_LEN
		sa = (struct sockaddr *)((caddr_t)sa + sa->sa_len);
#endif
	}
	sa = addrs;
	for (i = 0; i < addrcnt; i++) {
#ifndef HAVE_SA_LEN
		size_t sa_len;

#endif
#ifdef HAVE_SA_LEN
#if defined(INET) || defined(INET6)
		if (fix_port) {
			switch (sa->sa_family) {
#ifdef INET
			case AF_INET:
				((struct sockaddr_in *)sa)->sin_port = sport;
				break;
#endif
#ifdef INET6
			case AF_INET6:
				((struct sockaddr_in6 *)sa)->sin6_port = sport;
				break;
#endif
			}
		}
#endif
		if (usrsctp_setsockopt(so, IPPROTO_SCTP, flags, sa, sa->sa_len) != 0) {
			return (-1);
		}
		sa = (struct sockaddr *)((caddr_t)sa + sa->sa_len);
#else
		switch (sa->sa_family) {
#ifdef INET
		case AF_INET:
			sa_len = sizeof(struct sockaddr_in);
			break;
#endif
#ifdef INET6
		case AF_INET6:
			sa_len = sizeof(struct sockaddr_in6);
			break;
#endif
		default:
			sa_len = 0;
			break;
		}
		/*
		 * Now, if there was a port mentioned, assure that the
		 * first address has that port to make sure it fails or
		 * succeeds correctly.
		 */
#if defined(INET) || defined(INET6)
		if (fix_port) {
			switch (sa->sa_family) {
#ifdef INET
			case AF_INET:
				((struct sockaddr_in *)sa)->sin_port = sport;
				break;
#endif
#ifdef INET6
			case AF_INET6:
				((struct sockaddr_in6 *)sa)->sin6_port = sport;
				break;
#endif
			}
		}
#endif
		if (usrsctp_setsockopt(so, IPPROTO_SCTP, flags, sa, (socklen_t)sa_len) != 0) {
			return (-1);
		}
		sa = (struct sockaddr *)((caddr_t)sa + sa_len);
#endif
	}
	return (0);
}

int
usrsctp_connectx(struct socket *so,
                 const struct sockaddr *addrs, int addrcnt,
                 sctp_assoc_t *id)
{
#if defined(INET) || defined(INET6)
	char buf[SCTP_STACK_BUF_SIZE];
	int i, ret, cnt, *aa;
	char *cpto;
	const struct sockaddr *at;
	sctp_assoc_t *p_id;
	size_t len = sizeof(int);

	/* validate the address count and list */
	if ((addrs == NULL) || (addrcnt <= 0)) {
		errno = EINVAL;
		return (-1);
	}
	at = addrs;
	cnt = 0;
	cpto = ((caddr_t)buf + sizeof(int));
	/* validate all the addresses and get the size */
	for (i = 0; i < addrcnt; i++) {
		switch (at->sa_family) {
#ifdef INET
		case AF_INET:
#ifdef HAVE_SA_LEN
			if (at->sa_len != sizeof(struct sockaddr_in)) {
				errno = EINVAL;
				return (-1);
			}
#endif
			len += sizeof(struct sockaddr_in);
			if (len > SCTP_STACK_BUF_SIZE) {
				errno = ENOMEM;
				return (-1);
			}
			memcpy(cpto, at, sizeof(struct sockaddr_in));
			cpto = ((caddr_t)cpto + sizeof(struct sockaddr_in));
			at = (struct sockaddr *)((caddr_t)at + sizeof(struct sockaddr_in));
			break;
#endif
#ifdef INET6
		case AF_INET6:
#ifdef HAVE_SA_LEN
			if (at->sa_len != sizeof(struct sockaddr_in6)) {
				errno = EINVAL;
				return (-1);
			}
#endif
#ifdef INET
			if (IN6_IS_ADDR_V4MAPPED(&((struct sockaddr_in6 *)at)->sin6_addr)) {
				len += sizeof(struct sockaddr_in);
				if (len > SCTP_STACK_BUF_SIZE) {
					errno = ENOMEM;
					return (-1);
				}
				in6_sin6_2_sin((struct sockaddr_in *)cpto, (struct sockaddr_in6 *)at);
				cpto = ((caddr_t)cpto + sizeof(struct sockaddr_in));
			} else {
				len += sizeof(struct sockaddr_in6);
				if (len > SCTP_STACK_BUF_SIZE) {
					errno = ENOMEM;
					return (-1);
				}
				memcpy(cpto, at, sizeof(struct sockaddr_in6));
				cpto = ((caddr_t)cpto + sizeof(struct sockaddr_in6));
			}
#else
			len += sizeof(struct sockaddr_in6);
			if (len > SCTP_STACK_BUF_SIZE) {
				errno = ENOMEM;
				return (-1);
			}
			memcpy(cpto, at, sizeof(struct sockaddr_in6));
			cpto = ((caddr_t)cpto + sizeof(struct sockaddr_in6));
#endif
			at = (struct sockaddr *)((caddr_t)at + sizeof(struct sockaddr_in6));
			break;
#endif
		default:
			errno = EINVAL;
			return (-1);
		}
		cnt++;
	}
	aa = (int *)buf;
	*aa = cnt;
	ret = usrsctp_setsockopt(so, IPPROTO_SCTP, SCTP_CONNECT_X, (void *)buf, (socklen_t)len);
	if ((ret == 0) && id) {
		p_id = (sctp_assoc_t *)buf;
		*id = *p_id;
	}
	return (ret);
#else
	errno = EINVAL;
	return (-1);
#endif
}

int
usrsctp_getpaddrs(struct socket *so, sctp_assoc_t id, struct sockaddr **raddrs)
{
	struct sctp_getaddresses *addrs;
	struct sockaddr *sa;
	caddr_t lim;
	socklen_t opt_len;
	uint32_t size_of_addresses;
	int cnt;

	if (raddrs == NULL) {
		errno = EFAULT;
		return (-1);
	}
	/* When calling getsockopt(), the value contains the assoc_id. */
	size_of_addresses = (uint32_t)id;
	opt_len = (socklen_t)sizeof(uint32_t);
	if (usrsctp_getsockopt(so, IPPROTO_SCTP, SCTP_GET_REMOTE_ADDR_SIZE, &size_of_addresses, &opt_len) != 0) {
		if (errno == ENOENT) {
			return (0);
		} else {
			return (-1);
		}
	}
	opt_len = (socklen_t)((size_t)size_of_addresses + sizeof(struct sctp_getaddresses));
	addrs = calloc(1, (size_t)opt_len);
	if (addrs == NULL) {
		errno = ENOMEM;
		return (-1);
	}
	addrs->sget_assoc_id = id;
	/* Now lets get the array of addresses */
	if (usrsctp_getsockopt(so, IPPROTO_SCTP, SCTP_GET_PEER_ADDRESSES, addrs, &opt_len) != 0) {
		free(addrs);
		return (-1);
	}
	*raddrs = &addrs->addr[0].sa;
	cnt = 0;
	sa = &addrs->addr[0].sa;
	lim = (caddr_t)addrs + opt_len;
#ifdef HAVE_SA_LEN
	while (((caddr_t)sa < lim) && (sa->sa_len > 0)) {
		sa = (struct sockaddr *)((caddr_t)sa + sa->sa_len);
#else
	while ((caddr_t)sa < lim) {
		switch (sa->sa_family) {
#ifdef INET
		case AF_INET:
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_in));
			break;
#endif
#ifdef INET6
		case AF_INET6:
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_in6));
			break;
#endif
		case AF_CONN:
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_conn));
			break;
		default:
			return (cnt);
			break;
		}
#endif
		cnt++;
	}
	return (cnt);
}

void
usrsctp_freepaddrs(struct sockaddr *addrs)
{
	/* Take away the hidden association id */
	void *fr_addr;

	fr_addr = (void *)((caddr_t)addrs - offsetof(struct sctp_getaddresses, addr));
	/* Now free it */
	free(fr_addr);
}

int
usrsctp_getladdrs(struct socket *so, sctp_assoc_t id, struct sockaddr **raddrs)
{
	struct sctp_getaddresses *addrs;
	struct sockaddr *sa;
	caddr_t lim;
	socklen_t opt_len;
	uint32_t size_of_addresses;
	int cnt;

	if (raddrs == NULL) {
		errno = EFAULT;
		return (-1);
	}
	size_of_addresses = 0;
	opt_len = (socklen_t)sizeof(uint32_t);
	if (usrsctp_getsockopt(so, IPPROTO_SCTP, SCTP_GET_LOCAL_ADDR_SIZE, &size_of_addresses, &opt_len) != 0) {
		return (-1);
	}
	opt_len = (socklen_t)(size_of_addresses + sizeof(struct sctp_getaddresses));
	addrs = calloc(1, (size_t)opt_len);
	if (addrs == NULL) {
		errno = ENOMEM;
		return (-1);
	}
	addrs->sget_assoc_id = id;
	/* Now lets get the array of addresses */
	if (usrsctp_getsockopt(so, IPPROTO_SCTP, SCTP_GET_LOCAL_ADDRESSES, addrs, &opt_len) != 0) {
		free(addrs);
		return (-1);
	}
	if (size_of_addresses == 0) {
		free(addrs);
		return (0);
	}
	*raddrs = &addrs->addr[0].sa;
	cnt = 0;
	sa = &addrs->addr[0].sa;
	lim = (caddr_t)addrs + opt_len;
#ifdef HAVE_SA_LEN
	while (((caddr_t)sa < lim) && (sa->sa_len > 0)) {
		sa = (struct sockaddr *)((caddr_t)sa + sa->sa_len);
#else
	while ((caddr_t)sa < lim) {
		switch (sa->sa_family) {
#ifdef INET
		case AF_INET:
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_in));
			break;
#endif
#ifdef INET6
		case AF_INET6:
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_in6));
			break;
#endif
		case AF_CONN:
			sa = (struct sockaddr *)((caddr_t)sa + sizeof(struct sockaddr_conn));
			break;
		default:
			return (cnt);
			break;
		}
#endif
		cnt++;
	}
	return (cnt);
}

void
usrsctp_freeladdrs(struct sockaddr *addrs)
{
	/* Take away the hidden association id */
	void *fr_addr;

	fr_addr = (void *)((caddr_t)addrs - offsetof(struct sctp_getaddresses, addr));
	/* Now free it */
	free(fr_addr);
}

#ifdef INET
void
sctp_userspace_ip_output(int *result, struct mbuf *o_pak,
                         sctp_route_t *ro, void *inp,
                         uint32_t vrf_id)
{
	struct mbuf *m;
	struct mbuf *m_orig;
	int iovcnt;
	int len;
	struct ip *ip;
	struct udphdr *udp;
	struct sockaddr_in dst;
#if defined(_WIN32)
	WSAMSG win_msg_hdr;
	DWORD win_sent_len;
	WSABUF send_iovec[MAXLEN_MBUF_CHAIN];
	WSABUF winbuf;
#else
	struct iovec send_iovec[MAXLEN_MBUF_CHAIN];
	struct msghdr msg_hdr;
#endif
	int use_udp_tunneling;

	*result = 0;

	m = SCTP_HEADER_TO_CHAIN(o_pak);
	m_orig = m;

	len = sizeof(struct ip);
	if (SCTP_BUF_LEN(m) < len) {
		if ((m = m_pullup(m, len)) == 0) {
			SCTP_PRINTF("Can not get the IP header in the first mbuf.\n");
			return;
		}
	}
	ip = mtod(m, struct ip *);
	use_udp_tunneling = (ip->ip_p == IPPROTO_UDP);

	if (use_udp_tunneling) {
		len = sizeof(struct ip) + sizeof(struct udphdr);
		if (SCTP_BUF_LEN(m) < len) {
			if ((m = m_pullup(m, len)) == 0) {
				SCTP_PRINTF("Can not get the UDP/IP header in the first mbuf.\n");
				return;
			}
			ip = mtod(m, struct ip *);
		}
		udp = (struct udphdr *)(ip + 1);
	} else {
		udp = NULL;
	}

	if (!use_udp_tunneling) {
		if (ip->ip_src.s_addr == INADDR_ANY) {
			/* TODO get addr of outgoing interface */
			SCTP_PRINTF("Why did the SCTP implementation did not choose a source address?\n");
		}
		/* TODO need to worry about ro->ro_dst as in ip_output? */
#if defined(__linux__) || defined(_WIN32) || (defined(__FreeBSD__) && (__FreeBSD_version >= 1100030))
		/* need to put certain fields into network order for Linux */
		ip->ip_len = htons(ip->ip_len);
#endif
	}

	memset((void *)&dst, 0, sizeof(struct sockaddr_in));
	dst.sin_family = AF_INET;
	dst.sin_addr.s_addr = ip->ip_dst.s_addr;
#ifdef HAVE_SIN_LEN
	dst.sin_len = sizeof(struct sockaddr_in);
#endif
	if (use_udp_tunneling) {
		dst.sin_port = udp->uh_dport;
	} else {
		dst.sin_port = 0;
	}

	/* tweak the mbuf chain */
	if (use_udp_tunneling) {
		m_adj(m, sizeof(struct ip) + sizeof(struct udphdr));
	}

	for (iovcnt = 0; m != NULL && iovcnt < MAXLEN_MBUF_CHAIN; m = m->m_next, iovcnt++) {
#if !defined(_WIN32)
		send_iovec[iovcnt].iov_base = (caddr_t)m->m_data;
		send_iovec[iovcnt].iov_len = SCTP_BUF_LEN(m);
#else
		send_iovec[iovcnt].buf = (caddr_t)m->m_data;
		send_iovec[iovcnt].len = SCTP_BUF_LEN(m);
#endif
	}

	if (m != NULL) {
		SCTP_PRINTF("mbuf chain couldn't be copied completely\n");
		goto free_mbuf;
	}

#if !defined(_WIN32)
	msg_hdr.msg_name = (struct sockaddr *) &dst;
	msg_hdr.msg_namelen = sizeof(struct sockaddr_in);
	msg_hdr.msg_iov = send_iovec;
	msg_hdr.msg_iovlen = iovcnt;
	msg_hdr.msg_control = NULL;
	msg_hdr.msg_controllen = 0;
	msg_hdr.msg_flags = 0;

	if ((!use_udp_tunneling) && (SCTP_BASE_VAR(userspace_rawsctp) != -1)) {
		if (sendmsg(SCTP_BASE_VAR(userspace_rawsctp), &msg_hdr, MSG_DONTWAIT) < 0) {
			*result = errno;
		}
	}
	if ((use_udp_tunneling) && (SCTP_BASE_VAR(userspace_udpsctp) != -1)) {
		if (sendmsg(SCTP_BASE_VAR(userspace_udpsctp), &msg_hdr, MSG_DONTWAIT) < 0) {
			*result = errno;
		}
	}
#else
	win_msg_hdr.name = (struct sockaddr *) &dst;
	win_msg_hdr.namelen = sizeof(struct sockaddr_in);
	win_msg_hdr.lpBuffers = (LPWSABUF)send_iovec;
	win_msg_hdr.dwBufferCount = iovcnt;
	winbuf.len = 0;
	winbuf.buf = NULL;
	win_msg_hdr.Control = winbuf;
	win_msg_hdr.dwFlags = 0;

	if ((!use_udp_tunneling) && (SCTP_BASE_VAR(userspace_rawsctp) != -1)) {
		if (WSASendTo(SCTP_BASE_VAR(userspace_rawsctp), (LPWSABUF) send_iovec, iovcnt, &win_sent_len, win_msg_hdr.dwFlags, win_msg_hdr.name, (int) win_msg_hdr.namelen, NULL, NULL) != 0) {
			*result = WSAGetLastError();
		}
	}
	if ((use_udp_tunneling) && (SCTP_BASE_VAR(userspace_udpsctp) != -1)) {
		if (WSASendTo(SCTP_BASE_VAR(userspace_udpsctp), (LPWSABUF) send_iovec, iovcnt, &win_sent_len, win_msg_hdr.dwFlags, win_msg_hdr.name, (int) win_msg_hdr.namelen, NULL, NULL) != 0) {
			*result = WSAGetLastError();
		}
	}
#endif
free_mbuf:
	sctp_m_freem(m_orig);
}
#endif

#if defined(INET6)
void sctp_userspace_ip6_output(int *result, struct mbuf *o_pak,
                                            struct route_in6 *ro, void *inp,
                                            uint32_t vrf_id)
{
	struct mbuf *m;
	struct mbuf *m_orig;
	int iovcnt;
	int len;
	struct ip6_hdr *ip6;
	struct udphdr *udp;
	struct sockaddr_in6 dst;
#if defined(_WIN32)
	WSAMSG win_msg_hdr;
	DWORD win_sent_len;
	WSABUF send_iovec[MAXLEN_MBUF_CHAIN];
	WSABUF winbuf;
#else
	struct iovec send_iovec[MAXLEN_MBUF_CHAIN];
	struct msghdr msg_hdr;
#endif
	int use_udp_tunneling;

	*result = 0;

	m = SCTP_HEADER_TO_CHAIN(o_pak);
	m_orig = m;

	len = sizeof(struct ip6_hdr);

	if (SCTP_BUF_LEN(m) < len) {
		if ((m = m_pullup(m, len)) == 0) {
			SCTP_PRINTF("Can not get the IP header in the first mbuf.\n");
			return;
		}
	}

	ip6 = mtod(m, struct ip6_hdr *);
	use_udp_tunneling = (ip6->ip6_nxt == IPPROTO_UDP);

	if (use_udp_tunneling) {
		len = sizeof(struct ip6_hdr) + sizeof(struct udphdr);
		if (SCTP_BUF_LEN(m) < len) {
			if ((m = m_pullup(m, len)) == 0) {
				SCTP_PRINTF("Can not get the UDP/IP header in the first mbuf.\n");
				return;
			}
			ip6 = mtod(m, struct ip6_hdr *);
		}
		udp = (struct udphdr *)(ip6 + 1);
	} else {
		udp = NULL;
	}

	if (!use_udp_tunneling) {
		if (ip6->ip6_src.s6_addr == in6addr_any.s6_addr) {
			/* TODO get addr of outgoing interface */
			SCTP_PRINTF("Why did the SCTP implementation did not choose a source address?\n");
		}
		/* TODO need to worry about ro->ro_dst as in ip_output? */
	}

	memset((void *)&dst, 0, sizeof(struct sockaddr_in6));
	dst.sin6_family = AF_INET6;
	dst.sin6_addr = ip6->ip6_dst;
#ifdef HAVE_SIN6_LEN
	dst.sin6_len = sizeof(struct sockaddr_in6);
#endif

	if (use_udp_tunneling) {
		dst.sin6_port = udp->uh_dport;
	} else {
		dst.sin6_port = 0;
	}

	/* tweak the mbuf chain */
	if (use_udp_tunneling) {
		m_adj(m, sizeof(struct ip6_hdr) + sizeof(struct udphdr));
	} else {
		m_adj(m, sizeof(struct ip6_hdr));
	}

	for (iovcnt = 0; m != NULL && iovcnt < MAXLEN_MBUF_CHAIN; m = m->m_next, iovcnt++) {
#if !defined(_WIN32)
		send_iovec[iovcnt].iov_base = (caddr_t)m->m_data;
		send_iovec[iovcnt].iov_len = SCTP_BUF_LEN(m);
#else
		send_iovec[iovcnt].buf = (caddr_t)m->m_data;
		send_iovec[iovcnt].len = SCTP_BUF_LEN(m);
#endif
	}
	if (m != NULL) {
		SCTP_PRINTF("mbuf chain couldn't be copied completely\n");
		goto free_mbuf;
	}

#if !defined(_WIN32)
	msg_hdr.msg_name = (struct sockaddr *) &dst;
	msg_hdr.msg_namelen = sizeof(struct sockaddr_in6);
	msg_hdr.msg_iov = send_iovec;
	msg_hdr.msg_iovlen = iovcnt;
	msg_hdr.msg_control = NULL;
	msg_hdr.msg_controllen = 0;
	msg_hdr.msg_flags = 0;

	if ((!use_udp_tunneling) && (SCTP_BASE_VAR(userspace_rawsctp6) != -1)) {
		if (sendmsg(SCTP_BASE_VAR(userspace_rawsctp6), &msg_hdr, MSG_DONTWAIT)< 0) {
			*result = errno;
		}
	}
	if ((use_udp_tunneling) && (SCTP_BASE_VAR(userspace_udpsctp6) != -1)) {
		if (sendmsg(SCTP_BASE_VAR(userspace_udpsctp6), &msg_hdr, MSG_DONTWAIT) < 0) {
			*result = errno;
		}
	}
#else
	win_msg_hdr.name = (struct sockaddr *) &dst;
	win_msg_hdr.namelen = sizeof(struct sockaddr_in6);
	win_msg_hdr.lpBuffers = (LPWSABUF)send_iovec;
	win_msg_hdr.dwBufferCount = iovcnt;
	winbuf.len = 0;
	winbuf.buf = NULL;
	win_msg_hdr.Control = winbuf;
	win_msg_hdr.dwFlags = 0;

	if ((!use_udp_tunneling) && (SCTP_BASE_VAR(userspace_rawsctp6) != -1)) {
		if (WSASendTo(SCTP_BASE_VAR(userspace_rawsctp6), (LPWSABUF) send_iovec, iovcnt, &win_sent_len, win_msg_hdr.dwFlags, win_msg_hdr.name, (int) win_msg_hdr.namelen, NULL, NULL) != 0) {
			*result = WSAGetLastError();
		}
	}
	if ((use_udp_tunneling) && (SCTP_BASE_VAR(userspace_udpsctp6) != -1)) {
		if (WSASendTo(SCTP_BASE_VAR(userspace_udpsctp6), (LPWSABUF) send_iovec, iovcnt, &win_sent_len, win_msg_hdr.dwFlags, win_msg_hdr.name, (int) win_msg_hdr.namelen, NULL, NULL) != 0) {
			*result = WSAGetLastError();
		}
	}
#endif
free_mbuf:
	sctp_m_freem(m_orig);
}
#endif

void
usrsctp_register_address(void *addr)
{
	struct sockaddr_conn sconn;

	memset(&sconn, 0, sizeof(struct sockaddr_conn));
	sconn.sconn_family = AF_CONN;
#ifdef HAVE_SCONN_LEN
	sconn.sconn_len = sizeof(struct sockaddr_conn);
#endif
	sconn.sconn_port = 0;
	sconn.sconn_addr = addr;
	sctp_add_addr_to_vrf(SCTP_DEFAULT_VRFID,
	                     NULL,
	                     0xffffffff,
	                     0,
	                     "conn",
	                     NULL,
	                     (struct sockaddr *)&sconn,
	                     0,
	                     0);
}

void
usrsctp_deregister_address(void *addr)
{
	struct sockaddr_conn sconn;

	memset(&sconn, 0, sizeof(struct sockaddr_conn));
	sconn.sconn_family = AF_CONN;
#ifdef HAVE_SCONN_LEN
	sconn.sconn_len = sizeof(struct sockaddr_conn);
#endif
	sconn.sconn_port = 0;
	sconn.sconn_addr = addr;
	sctp_del_addr_from_vrf(SCTP_DEFAULT_VRFID,
	                       (struct sockaddr *)&sconn,
	                       0xffffffff,
	                       "conn");
}

#define PREAMBLE_FORMAT "\n%c %02d:%02d:%02d.%06ld "
#define PREAMBLE_LENGTH 19
#define HEADER "0000 "
#define TRAILER "# SCTP_PACKET\n"

char *
usrsctp_dumppacket(const void *buf, size_t len, int outbound)
{
	size_t i, pos;
	char *dump_buf, *packet;
	struct tm t;
#ifdef _WIN32
	struct timeb tb;
#else
	struct timeval tv;
	time_t sec;
#endif

	if ((len == 0) || (buf == NULL)) {
		return (NULL);
	}
	if ((dump_buf = malloc(PREAMBLE_LENGTH + strlen(HEADER) + 3 * len + strlen(TRAILER) + 1)) == NULL) {
		return (NULL);
	}
	pos = 0;
#ifdef _WIN32
	ftime(&tb);
	localtime_s(&t, &tb.time);
#if defined(__MINGW32__)
	if (snprintf(dump_buf, PREAMBLE_LENGTH + 1, PREAMBLE_FORMAT,
	             outbound ? 'O' : 'I',
	             t.tm_hour, t.tm_min, t.tm_sec, (long)(1000 * tb.millitm)) < 0) {
		free(dump_buf);
		return (NULL);
	}
#else
	if (_snprintf_s(dump_buf, PREAMBLE_LENGTH + 1, PREAMBLE_LENGTH, PREAMBLE_FORMAT,
	                outbound ? 'O' : 'I',
	                t.tm_hour, t.tm_min, t.tm_sec, (long)(1000 * tb.millitm)) < 0) {
		free(dump_buf);
		return (NULL);
	}
#endif
#else
	gettimeofday(&tv, NULL);
	sec = (time_t)tv.tv_sec;
	localtime_r((const time_t *)&sec, &t);
	if (snprintf(dump_buf, PREAMBLE_LENGTH + 1, PREAMBLE_FORMAT,
	             outbound ? 'O' : 'I',
	             t.tm_hour, t.tm_min, t.tm_sec, (long)tv.tv_usec) < 0) {
		free(dump_buf);
		return (NULL);
	}
#endif
	pos += PREAMBLE_LENGTH;
#if defined(_WIN32) && !defined(__MINGW32__)
	strncpy_s(dump_buf + pos, strlen(HEADER) + 1, HEADER, strlen(HEADER));
#else
	strcpy(dump_buf + pos, HEADER);
#endif
	pos += strlen(HEADER);
	packet = (char *)buf;
	for (i = 0; i < len; i++) {
		uint8_t byte, low, high;

		byte = (uint8_t)packet[i];
		high = byte / 16;
		low = byte % 16;
		dump_buf[pos++] = high < 10 ? '0' + high : 'a' + (high - 10);
		dump_buf[pos++] = low < 10 ? '0' + low : 'a' + (low - 10);
		dump_buf[pos++] = ' ';
	}
#if defined(_WIN32) && !defined(__MINGW32__)
	strncpy_s(dump_buf + pos, strlen(TRAILER) + 1, TRAILER, strlen(TRAILER));
#else
	strcpy(dump_buf + pos, TRAILER);
#endif
	pos += strlen(TRAILER);
	dump_buf[pos++] = '\0';
	return (dump_buf);
}

void
usrsctp_freedumpbuffer(char *buf)
{
	free(buf);
}

void
usrsctp_enable_crc32c_offload(void)
{
	SCTP_BASE_VAR(crc32c_offloaded) = 1;
}

void
usrsctp_disable_crc32c_offload(void)
{
	SCTP_BASE_VAR(crc32c_offloaded) = 0;
}

/* Compute the CRC32C in network byte order */
uint32_t
usrsctp_crc32c(void *buffer, size_t length)
{
	uint32_t base = 0xffffffff;

	base = calculate_crc32c(0xffffffff, (unsigned char *)buffer, (unsigned int) length);
	base = sctp_finalize_crc32c(base);
	return (base);
}

void
usrsctp_conninput(void *addr, const void *buffer, size_t length, uint8_t ecn_bits)
{
	struct sockaddr_conn src, dst;
	struct mbuf *m, *mm;
	struct sctphdr *sh;
	struct sctp_chunkhdr *ch;
	int remaining, offset;

	SCTP_STAT_INCR(sctps_recvpackets);
	SCTP_STAT_INCR_COUNTER64(sctps_inpackets);
	memset(&src, 0, sizeof(struct sockaddr_conn));
	src.sconn_family = AF_CONN;
#ifdef HAVE_SCONN_LEN
	src.sconn_len = sizeof(struct sockaddr_conn);
#endif
	src.sconn_addr = addr;
	memset(&dst, 0, sizeof(struct sockaddr_conn));
	dst.sconn_family = AF_CONN;
#ifdef HAVE_SCONN_LEN
	dst.sconn_len = sizeof(struct sockaddr_conn);
#endif
	dst.sconn_addr = addr;
	if ((m = sctp_get_mbuf_for_msg((unsigned int)length, 1, M_NOWAIT, 0, MT_DATA)) == NULL) {
		return;
	}
	/* Set the lengths fields of the mbuf chain.
	 * This is expected by m_copyback().
	 */
	remaining = (int)length;
	for (mm = m; mm != NULL; mm = mm->m_next) {
		mm->m_len = min((int)M_SIZE(mm), remaining);
		m->m_pkthdr.len += mm->m_len;
		remaining -= mm->m_len;
	}
	KASSERT(remaining == 0, ("usrsctp_conninput: %zu bytes left", remaining));
	m_copyback(m, 0, (int)length, (caddr_t)buffer);
	offset = sizeof(struct sctphdr) + sizeof(struct sctp_chunkhdr);
	if (SCTP_BUF_LEN(m) < offset) {
		if ((m = m_pullup(m, offset)) == NULL) {
			SCTP_STAT_INCR(sctps_hdrops);
			return;
		}
	}
	sh = mtod(m, struct sctphdr *);
	ch = (struct sctp_chunkhdr *)((caddr_t)sh + sizeof(struct sctphdr));
	offset -= sizeof(struct sctp_chunkhdr);
	src.sconn_port = sh->src_port;
	dst.sconn_port = sh->dest_port;
	sctp_common_input_processing(&m, 0, offset, (int)length,
	                             (struct sockaddr *)&src,
	                             (struct sockaddr *)&dst,
	                             sh, ch,
	                             SCTP_BASE_VAR(crc32c_offloaded) == 1 ? 0 : 1,
	                             ecn_bits,
	                             SCTP_DEFAULT_VRFID, 0);
	if (m) {
		sctp_m_freem(m);
	}
	return;
}

void usrsctp_handle_timers(uint32_t elapsed_milliseconds)
{
	sctp_handle_tick(sctp_msecs_to_ticks(elapsed_milliseconds));
}

int
usrsctp_get_events(struct socket *so)
{
	int events = 0;

	if (so == NULL) {
		errno = EBADF;
		return -1;
	}

	SOCK_LOCK(so);
	if (soreadable(so)) {
		events |= SCTP_EVENT_READ;
	}
	if (sowriteable(so)) {
		events |= SCTP_EVENT_WRITE;
	}
	if (so->so_error) {
		events |= SCTP_EVENT_ERROR;
	}
	SOCK_UNLOCK(so);

	return events;
}

int
usrsctp_set_upcall(struct socket *so, void (*upcall)(struct socket *, void *, int), void *arg)
{
	if (so == NULL) {
		errno = EBADF;
		return (-1);
	}

	SOCK_LOCK(so);
	so->so_upcall = upcall;
	so->so_upcallarg = arg;
	so->so_snd.sb_flags |= SB_UPCALL;
	so->so_rcv.sb_flags |= SB_UPCALL;
	SOCK_UNLOCK(so);

	return (0);
}

#define USRSCTP_TUNABLE_SET_DEF(__field, __prefix)   \
int usrsctp_tunable_set_ ## __field(uint32_t value)  \
{                                                    \
	if ((value < __prefix##_MIN) ||              \
	    (value > __prefix##_MAX)) {              \
		errno = EINVAL;                      \
		return (-1);                         \
	} else {                                     \
		SCTP_BASE_SYSCTL(__field) = value;   \
		return (0);                          \
	}                                            \
}

USRSCTP_TUNABLE_SET_DEF(sctp_hashtblsize, SCTPCTL_TCBHASHSIZE)
USRSCTP_TUNABLE_SET_DEF(sctp_pcbtblsize, SCTPCTL_PCBHASHSIZE)
USRSCTP_TUNABLE_SET_DEF(sctp_chunkscale, SCTPCTL_CHUNKSCALE)

#define USRSCTP_SYSCTL_SET_DEF(__field, __prefix)    \
int usrsctp_sysctl_set_ ## __field(uint32_t value)   \
{                                                    \
	if ((value < __prefix##_MIN) ||              \
	    (value > __prefix##_MAX)) {              \
		errno = EINVAL;                      \
		return (-1);                         \
	} else {                                     \
		SCTP_BASE_SYSCTL(__field) = value;   \
		return (0);                          \
	}                                            \
}

#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
USRSCTP_SYSCTL_SET_DEF(sctp_sendspace, SCTPCTL_MAXDGRAM)
USRSCTP_SYSCTL_SET_DEF(sctp_recvspace, SCTPCTL_RECVSPACE)
USRSCTP_SYSCTL_SET_DEF(sctp_auto_asconf, SCTPCTL_AUTOASCONF)
USRSCTP_SYSCTL_SET_DEF(sctp_ecn_enable, SCTPCTL_ECN_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_pr_enable, SCTPCTL_PR_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_auth_enable, SCTPCTL_AUTH_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_asconf_enable, SCTPCTL_ASCONF_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_reconfig_enable, SCTPCTL_RECONFIG_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_nrsack_enable, SCTPCTL_NRSACK_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_pktdrop_enable, SCTPCTL_PKTDROP_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_no_csum_on_loopback, SCTPCTL_LOOPBACK_NOCSUM)
USRSCTP_SYSCTL_SET_DEF(sctp_peer_chunk_oh, SCTPCTL_PEER_CHKOH)
USRSCTP_SYSCTL_SET_DEF(sctp_max_burst_default, SCTPCTL_MAXBURST)
USRSCTP_SYSCTL_SET_DEF(sctp_max_chunks_on_queue, SCTPCTL_MAXCHUNKS)
USRSCTP_SYSCTL_SET_DEF(sctp_min_split_point, SCTPCTL_MIN_SPLIT_POINT)
USRSCTP_SYSCTL_SET_DEF(sctp_delayed_sack_time_default, SCTPCTL_DELAYED_SACK_TIME)
USRSCTP_SYSCTL_SET_DEF(sctp_sack_freq_default, SCTPCTL_SACK_FREQ)
USRSCTP_SYSCTL_SET_DEF(sctp_system_free_resc_limit, SCTPCTL_SYS_RESOURCE)
USRSCTP_SYSCTL_SET_DEF(sctp_asoc_free_resc_limit, SCTPCTL_ASOC_RESOURCE)
USRSCTP_SYSCTL_SET_DEF(sctp_heartbeat_interval_default, SCTPCTL_HEARTBEAT_INTERVAL)
USRSCTP_SYSCTL_SET_DEF(sctp_pmtu_raise_time_default, SCTPCTL_PMTU_RAISE_TIME)
USRSCTP_SYSCTL_SET_DEF(sctp_shutdown_guard_time_default, SCTPCTL_SHUTDOWN_GUARD_TIME)
USRSCTP_SYSCTL_SET_DEF(sctp_secret_lifetime_default, SCTPCTL_SECRET_LIFETIME)
USRSCTP_SYSCTL_SET_DEF(sctp_rto_max_default, SCTPCTL_RTO_MAX)
USRSCTP_SYSCTL_SET_DEF(sctp_rto_min_default, SCTPCTL_RTO_MIN)
USRSCTP_SYSCTL_SET_DEF(sctp_rto_initial_default, SCTPCTL_RTO_INITIAL)
USRSCTP_SYSCTL_SET_DEF(sctp_init_rto_max_default, SCTPCTL_INIT_RTO_MAX)
USRSCTP_SYSCTL_SET_DEF(sctp_valid_cookie_life_default, SCTPCTL_VALID_COOKIE_LIFE)
USRSCTP_SYSCTL_SET_DEF(sctp_init_rtx_max_default, SCTPCTL_INIT_RTX_MAX)
USRSCTP_SYSCTL_SET_DEF(sctp_assoc_rtx_max_default, SCTPCTL_ASSOC_RTX_MAX)
USRSCTP_SYSCTL_SET_DEF(sctp_path_rtx_max_default, SCTPCTL_PATH_RTX_MAX)
USRSCTP_SYSCTL_SET_DEF(sctp_add_more_threshold, SCTPCTL_ADD_MORE_ON_OUTPUT)
USRSCTP_SYSCTL_SET_DEF(sctp_nr_incoming_streams_default, SCTPCTL_INCOMING_STREAMS)
USRSCTP_SYSCTL_SET_DEF(sctp_nr_outgoing_streams_default, SCTPCTL_OUTGOING_STREAMS)
USRSCTP_SYSCTL_SET_DEF(sctp_cmt_on_off, SCTPCTL_CMT_ON_OFF)
USRSCTP_SYSCTL_SET_DEF(sctp_cmt_use_dac, SCTPCTL_CMT_USE_DAC)
USRSCTP_SYSCTL_SET_DEF(sctp_use_cwnd_based_maxburst, SCTPCTL_CWND_MAXBURST)
USRSCTP_SYSCTL_SET_DEF(sctp_nat_friendly, SCTPCTL_NAT_FRIENDLY)
USRSCTP_SYSCTL_SET_DEF(sctp_L2_abc_variable, SCTPCTL_ABC_L_VAR)
USRSCTP_SYSCTL_SET_DEF(sctp_mbuf_threshold_count, SCTPCTL_MAX_CHAINED_MBUFS)
USRSCTP_SYSCTL_SET_DEF(sctp_do_drain, SCTPCTL_DO_SCTP_DRAIN)
USRSCTP_SYSCTL_SET_DEF(sctp_hb_maxburst, SCTPCTL_HB_MAX_BURST)
USRSCTP_SYSCTL_SET_DEF(sctp_abort_if_one_2_one_hits_limit, SCTPCTL_ABORT_AT_LIMIT)
USRSCTP_SYSCTL_SET_DEF(sctp_min_residual, SCTPCTL_MIN_RESIDUAL)
USRSCTP_SYSCTL_SET_DEF(sctp_max_retran_chunk, SCTPCTL_MAX_RETRAN_CHUNK)
USRSCTP_SYSCTL_SET_DEF(sctp_logging_level, SCTPCTL_LOGGING_LEVEL)
USRSCTP_SYSCTL_SET_DEF(sctp_default_cc_module, SCTPCTL_DEFAULT_CC_MODULE)
USRSCTP_SYSCTL_SET_DEF(sctp_default_frag_interleave, SCTPCTL_DEFAULT_FRAG_INTERLEAVE)
USRSCTP_SYSCTL_SET_DEF(sctp_mobility_base, SCTPCTL_MOBILITY_BASE)
USRSCTP_SYSCTL_SET_DEF(sctp_mobility_fasthandoff, SCTPCTL_MOBILITY_FASTHANDOFF)
USRSCTP_SYSCTL_SET_DEF(sctp_inits_include_nat_friendly, SCTPCTL_NAT_FRIENDLY_INITS)
USRSCTP_SYSCTL_SET_DEF(sctp_udp_tunneling_port, SCTPCTL_UDP_TUNNELING_PORT)
USRSCTP_SYSCTL_SET_DEF(sctp_enable_sack_immediately, SCTPCTL_SACK_IMMEDIATELY_ENABLE)
USRSCTP_SYSCTL_SET_DEF(sctp_vtag_time_wait, SCTPCTL_TIME_WAIT)
USRSCTP_SYSCTL_SET_DEF(sctp_blackhole, SCTPCTL_BLACKHOLE)
USRSCTP_SYSCTL_SET_DEF(sctp_diag_info_code, SCTPCTL_DIAG_INFO_CODE)
USRSCTP_SYSCTL_SET_DEF(sctp_fr_max_burst_default, SCTPCTL_FRMAXBURST)
USRSCTP_SYSCTL_SET_DEF(sctp_path_pf_threshold, SCTPCTL_PATH_PF_THRESHOLD)
USRSCTP_SYSCTL_SET_DEF(sctp_default_ss_module, SCTPCTL_DEFAULT_SS_MODULE)
USRSCTP_SYSCTL_SET_DEF(sctp_rttvar_bw, SCTPCTL_RTTVAR_BW)
USRSCTP_SYSCTL_SET_DEF(sctp_rttvar_rtt, SCTPCTL_RTTVAR_RTT)
USRSCTP_SYSCTL_SET_DEF(sctp_rttvar_eqret, SCTPCTL_RTTVAR_EQRET)
USRSCTP_SYSCTL_SET_DEF(sctp_steady_step, SCTPCTL_RTTVAR_STEADYS)
USRSCTP_SYSCTL_SET_DEF(sctp_use_dccc_ecn, SCTPCTL_RTTVAR_DCCCECN)
USRSCTP_SYSCTL_SET_DEF(sctp_buffer_splitting, SCTPCTL_BUFFER_SPLITTING)
USRSCTP_SYSCTL_SET_DEF(sctp_initial_cwnd, SCTPCTL_INITIAL_CWND)
#ifdef SCTP_DEBUG
USRSCTP_SYSCTL_SET_DEF(sctp_debug_on, SCTPCTL_DEBUG)
#endif
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#define USRSCTP_SYSCTL_GET_DEF(__field) \
uint32_t usrsctp_sysctl_get_ ## __field(void) { \
	return SCTP_BASE_SYSCTL(__field); \
}

USRSCTP_SYSCTL_GET_DEF(sctp_sendspace)
USRSCTP_SYSCTL_GET_DEF(sctp_recvspace)
USRSCTP_SYSCTL_GET_DEF(sctp_auto_asconf)
USRSCTP_SYSCTL_GET_DEF(sctp_multiple_asconfs)
USRSCTP_SYSCTL_GET_DEF(sctp_ecn_enable)
USRSCTP_SYSCTL_GET_DEF(sctp_pr_enable)
USRSCTP_SYSCTL_GET_DEF(sctp_auth_enable)
USRSCTP_SYSCTL_GET_DEF(sctp_asconf_enable)
USRSCTP_SYSCTL_GET_DEF(sctp_reconfig_enable)
USRSCTP_SYSCTL_GET_DEF(sctp_nrsack_enable)
USRSCTP_SYSCTL_GET_DEF(sctp_pktdrop_enable)
USRSCTP_SYSCTL_GET_DEF(sctp_no_csum_on_loopback)
USRSCTP_SYSCTL_GET_DEF(sctp_peer_chunk_oh)
USRSCTP_SYSCTL_GET_DEF(sctp_max_burst_default)
USRSCTP_SYSCTL_GET_DEF(sctp_max_chunks_on_queue)
USRSCTP_SYSCTL_GET_DEF(sctp_hashtblsize)
USRSCTP_SYSCTL_GET_DEF(sctp_pcbtblsize)
USRSCTP_SYSCTL_GET_DEF(sctp_min_split_point)
USRSCTP_SYSCTL_GET_DEF(sctp_chunkscale)
USRSCTP_SYSCTL_GET_DEF(sctp_delayed_sack_time_default)
USRSCTP_SYSCTL_GET_DEF(sctp_sack_freq_default)
USRSCTP_SYSCTL_GET_DEF(sctp_system_free_resc_limit)
USRSCTP_SYSCTL_GET_DEF(sctp_asoc_free_resc_limit)
USRSCTP_SYSCTL_GET_DEF(sctp_heartbeat_interval_default)
USRSCTP_SYSCTL_GET_DEF(sctp_pmtu_raise_time_default)
USRSCTP_SYSCTL_GET_DEF(sctp_shutdown_guard_time_default)
USRSCTP_SYSCTL_GET_DEF(sctp_secret_lifetime_default)
USRSCTP_SYSCTL_GET_DEF(sctp_rto_max_default)
USRSCTP_SYSCTL_GET_DEF(sctp_rto_min_default)
USRSCTP_SYSCTL_GET_DEF(sctp_rto_initial_default)
USRSCTP_SYSCTL_GET_DEF(sctp_init_rto_max_default)
USRSCTP_SYSCTL_GET_DEF(sctp_valid_cookie_life_default)
USRSCTP_SYSCTL_GET_DEF(sctp_init_rtx_max_default)
USRSCTP_SYSCTL_GET_DEF(sctp_assoc_rtx_max_default)
USRSCTP_SYSCTL_GET_DEF(sctp_path_rtx_max_default)
USRSCTP_SYSCTL_GET_DEF(sctp_add_more_threshold)
USRSCTP_SYSCTL_GET_DEF(sctp_nr_incoming_streams_default)
USRSCTP_SYSCTL_GET_DEF(sctp_nr_outgoing_streams_default)
USRSCTP_SYSCTL_GET_DEF(sctp_cmt_on_off)
USRSCTP_SYSCTL_GET_DEF(sctp_cmt_use_dac)
USRSCTP_SYSCTL_GET_DEF(sctp_use_cwnd_based_maxburst)
USRSCTP_SYSCTL_GET_DEF(sctp_nat_friendly)
USRSCTP_SYSCTL_GET_DEF(sctp_L2_abc_variable)
USRSCTP_SYSCTL_GET_DEF(sctp_mbuf_threshold_count)
USRSCTP_SYSCTL_GET_DEF(sctp_do_drain)
USRSCTP_SYSCTL_GET_DEF(sctp_hb_maxburst)
USRSCTP_SYSCTL_GET_DEF(sctp_abort_if_one_2_one_hits_limit)
USRSCTP_SYSCTL_GET_DEF(sctp_min_residual)
USRSCTP_SYSCTL_GET_DEF(sctp_max_retran_chunk)
USRSCTP_SYSCTL_GET_DEF(sctp_logging_level)
USRSCTP_SYSCTL_GET_DEF(sctp_default_cc_module)
USRSCTP_SYSCTL_GET_DEF(sctp_default_frag_interleave)
USRSCTP_SYSCTL_GET_DEF(sctp_mobility_base)
USRSCTP_SYSCTL_GET_DEF(sctp_mobility_fasthandoff)
USRSCTP_SYSCTL_GET_DEF(sctp_inits_include_nat_friendly)
USRSCTP_SYSCTL_GET_DEF(sctp_udp_tunneling_port)
USRSCTP_SYSCTL_GET_DEF(sctp_enable_sack_immediately)
USRSCTP_SYSCTL_GET_DEF(sctp_vtag_time_wait)
USRSCTP_SYSCTL_GET_DEF(sctp_blackhole)
USRSCTP_SYSCTL_GET_DEF(sctp_diag_info_code)
USRSCTP_SYSCTL_GET_DEF(sctp_fr_max_burst_default)
USRSCTP_SYSCTL_GET_DEF(sctp_path_pf_threshold)
USRSCTP_SYSCTL_GET_DEF(sctp_default_ss_module)
USRSCTP_SYSCTL_GET_DEF(sctp_rttvar_bw)
USRSCTP_SYSCTL_GET_DEF(sctp_rttvar_rtt)
USRSCTP_SYSCTL_GET_DEF(sctp_rttvar_eqret)
USRSCTP_SYSCTL_GET_DEF(sctp_steady_step)
USRSCTP_SYSCTL_GET_DEF(sctp_use_dccc_ecn)
USRSCTP_SYSCTL_GET_DEF(sctp_buffer_splitting)
USRSCTP_SYSCTL_GET_DEF(sctp_initial_cwnd)
#ifdef SCTP_DEBUG
USRSCTP_SYSCTL_GET_DEF(sctp_debug_on)
#endif

void usrsctp_get_stat(struct sctpstat *stat)
{
	*stat = SCTP_BASE_STATS;
}
