/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2006-2007, by Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2008-2011, by Randall Stewart. All rights reserved.
 * Copyright (c) 2008-2011, by Michael Tuexen. All rights reserved.
 * Copyright (c) 2008-2011, by Brad Penoff. All rights reserved.
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

#ifndef __sctp_os_userspace_h__
#define __sctp_os_userspace_h__
/*
 * Userspace includes
 * All the opt_xxx.h files are placed in the kernel build directory.
 * We will place them in userspace stack build directory.
 */

#include <errno.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <mswsock.h>
#include <windows.h>
#include "user_environment.h"
typedef CRITICAL_SECTION userland_mutex_t;
#if WINVER < 0x0600
typedef CRITICAL_SECTION userland_rwlock_t;
enum {
	C_SIGNAL = 0,
	C_BROADCAST = 1,
	C_MAX_EVENTS = 2
};
typedef struct
{
	u_int waiters_count;
	CRITICAL_SECTION waiters_count_lock;
	HANDLE events_[C_MAX_EVENTS];
} userland_cond_t;
void InitializeXPConditionVariable(userland_cond_t *);
void DeleteXPConditionVariable(userland_cond_t *);
int SleepXPConditionVariable(userland_cond_t *, userland_mutex_t *);
void WakeAllXPConditionVariable(userland_cond_t *);
#define InitializeConditionVariable(cond) InitializeXPConditionVariable(cond)
#define DeleteConditionVariable(cond) DeleteXPConditionVariable(cond)
#define SleepConditionVariableCS(cond, mtx, time) SleepXPConditionVariable(cond, mtx)
#define WakeAllConditionVariable(cond) WakeAllXPConditionVariable(cond)
#else
typedef SRWLOCK userland_rwlock_t;
#define DeleteConditionVariable(cond)
typedef CONDITION_VARIABLE userland_cond_t;
#endif
typedef HANDLE userland_thread_t;
#define ADDRESS_FAMILY	unsigned __int8
#define IPVERSION  4
#define MAXTTL     255
/* VS2010 comes with stdint.h */
#if !defined(_MSC_VER) || (_MSC_VER >= 1600)
#include <stdint.h>
#else
typedef unsigned __int64 uint64_t;
typedef unsigned __int32 uint32_t;
typedef __int32          int32_t;
typedef unsigned __int16 uint16_t;
typedef __int16          int16_t;
typedef unsigned __int8  uint8_t;
typedef __int8           int8_t;
#endif
#ifndef _SIZE_T_DEFINED
#typedef __int32         size_t;
#endif
typedef unsigned __int32 u_int;
typedef unsigned char    u_char;
typedef unsigned __int16 u_short;
typedef unsigned __int8  sa_family_t;
#ifndef _SSIZE_T_DEFINED
typedef __int64          ssize_t;
#endif
#if !defined(__MINGW32__)
#define __func__	__FUNCTION__
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK             WSAEWOULDBLOCK
#endif
#ifndef EINPROGRESS
#define EINPROGRESS             WSAEINPROGRESS
#endif
#ifndef EALREADY
#define EALREADY                WSAEALREADY
#endif
#ifndef ENOTSOCK
#define ENOTSOCK                WSAENOTSOCK
#endif
#ifndef EDESTADDRREQ
#define EDESTADDRREQ            WSAEDESTADDRREQ
#endif
#ifndef EMSGSIZE
#define EMSGSIZE                WSAEMSGSIZE
#endif
#ifndef EPROTOTYPE
#define EPROTOTYPE              WSAEPROTOTYPE
#endif
#ifndef ENOPROTOOPT
#define ENOPROTOOPT             WSAENOPROTOOPT
#endif
#ifndef EPROTONOSUPPORT
#define EPROTONOSUPPORT         WSAEPROTONOSUPPORT
#endif
#ifndef ESOCKTNOSUPPORT
#define ESOCKTNOSUPPORT         WSAESOCKTNOSUPPORT
#endif
#ifndef EOPNOTSUPP
#define EOPNOTSUPP              WSAEOPNOTSUPP
#endif
#ifndef ENOTSUP
#define ENOTSUP                 WSAEOPNOTSUPP
#endif
#ifndef EPFNOSUPPORT
#define EPFNOSUPPORT            WSAEPFNOSUPPORT
#endif
#ifndef EAFNOSUPPORT
#define EAFNOSUPPORT            WSAEAFNOSUPPORT
#endif
#ifndef EADDRINUSE
#define EADDRINUSE              WSAEADDRINUSE
#endif
#ifndef EADDRNOTAVAIL
#define EADDRNOTAVAIL           WSAEADDRNOTAVAIL
#endif
#ifndef ENETDOWN
#define ENETDOWN                WSAENETDOWN
#endif
#ifndef ENETUNREACH
#define ENETUNREACH             WSAENETUNREACH
#endif
#ifndef ENETRESET
#define ENETRESET               WSAENETRESET
#endif
#ifndef ECONNABORTED
#define ECONNABORTED            WSAECONNABORTED
#endif
#ifndef ECONNRESET
#define ECONNRESET              WSAECONNRESET
#endif
#ifndef ENOBUFS
#define ENOBUFS                 WSAENOBUFS
#endif
#ifndef EISCONN
#define EISCONN                 WSAEISCONN
#endif
#ifndef ENOTCONN
#define ENOTCONN                WSAENOTCONN
#endif
#ifndef ESHUTDOWN
#define ESHUTDOWN               WSAESHUTDOWN
#endif
#ifndef ETOOMANYREFS
#define ETOOMANYREFS            WSAETOOMANYREFS
#endif
#ifndef ETIMEDOUT
#define ETIMEDOUT               WSAETIMEDOUT
#endif
#ifndef ECONNREFUSED
#define ECONNREFUSED            WSAECONNREFUSED
#endif
#ifndef ELOOP
#define ELOOP                   WSAELOOP
#endif
#ifndef EHOSTDOWN
#define EHOSTDOWN               WSAEHOSTDOWN
#endif
#ifndef EHOSTUNREACH
#define EHOSTUNREACH            WSAEHOSTUNREACH
#endif
#ifndef EPROCLIM
#define EPROCLIM                WSAEPROCLIM
#endif
#ifndef EUSERS
#define EUSERS                  WSAEUSERS
#endif
#ifndef EDQUOT
#define EDQUOT                  WSAEDQUOT
#endif
#ifndef ESTALE
#define ESTALE                  WSAESTALE
#endif
#ifndef EREMOTE
#define EREMOTE                 WSAEREMOTE
#endif

typedef char* caddr_t;

#define bzero(buf, len) memset(buf, 0, len)
#define bcopy(srcKey, dstKey, len) memcpy(dstKey, srcKey, len)

#if defined(_MSC_VER) && (_MSC_VER < 1900) && !defined(__MINGW32__)
#define SCTP_SNPRINTF(data, size, format, ...) 					\
	if (_snprintf_s(data, size, _TRUNCATE, format, __VA_ARGS__) < 0) {	\
		data[0] = '\0';							\
	}
#else
#define SCTP_SNPRINTF(data, ...)						\
	if (snprintf(data, __VA_ARGS__) < 0 ) {					\
		data[0] = '\0';							\
	}
#endif

#define inline __inline
#define __inline__ __inline
#define	MSG_EOR		0x8		/* data completes record */
#define	MSG_DONTWAIT	0x80		/* this message should be nonblocking */

#ifdef CMSG_DATA
#undef CMSG_DATA
#endif
/*
 * The following definitions should apply iff WINVER < 0x0600
 * but that check doesn't work in all cases. So be more pedantic...
 */
#define CMSG_DATA(x) WSA_CMSG_DATA(x)
#define CMSG_ALIGN(x) WSA_CMSGDATA_ALIGN(x)
#ifndef CMSG_FIRSTHDR
#define CMSG_FIRSTHDR(x) WSA_CMSG_FIRSTHDR(x)
#endif
#ifndef CMSG_NXTHDR
#define CMSG_NXTHDR(x, y) WSA_CMSG_NXTHDR(x, y)
#endif
#ifndef CMSG_SPACE
#define CMSG_SPACE(x) WSA_CMSG_SPACE(x)
#endif
#ifndef CMSG_LEN
#define CMSG_LEN(x) WSA_CMSG_LEN(x)
#endif

/****  from sctp_os_windows.h ***************/
#define SCTP_IFN_IS_IFT_LOOP(ifn)	((ifn)->ifn_type == IFT_LOOP)
#define SCTP_ROUTE_IS_REAL_LOOP(ro) ((ro)->ro_rt && (ro)->ro_rt->rt_ifa && (ro)->ro_rt->rt_ifa->ifa_ifp && (ro)->ro_rt->rt_ifa->ifa_ifp->if_type == IFT_LOOP)

/*
 * Access to IFN's to help with src-addr-selection
 */
/* This could return VOID if the index works but for BSD we provide both. */
#define SCTP_GET_IFN_VOID_FROM_ROUTE(ro) \
	((ro)->ro_rt != NULL ? (ro)->ro_rt->rt_ifp : NULL)
#define SCTP_ROUTE_HAS_VALID_IFN(ro) \
	((ro)->ro_rt && (ro)->ro_rt->rt_ifp)
/******************************************/

#define SCTP_GET_IF_INDEX_FROM_ROUTE(ro) 1 /* compiles...  TODO use routing socket to determine */

#define BIG_ENDIAN 1
#define LITTLE_ENDIAN 0
#ifdef WORDS_BIGENDIAN
#define BYTE_ORDER BIG_ENDIAN
#else
#define BYTE_ORDER LITTLE_ENDIAN
#endif

#else /* !defined(Userspace_os_Windows) */
#include <sys/socket.h>

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
#error "Unsupported build configuration."
#endif

#include <pthread.h>

typedef pthread_mutex_t userland_mutex_t;
typedef pthread_rwlock_t userland_rwlock_t;
typedef pthread_cond_t userland_cond_t;
typedef pthread_t userland_thread_t;
#endif

#if defined(_WIN32) || defined(__native_client__)

#define IFNAMSIZ 64

#define random() rand()
#define srandom(s) srand(s)

#define timeradd(tvp, uvp, vvp)   \
	do {                          \
	    (vvp)->tv_sec = (tvp)->tv_sec + (uvp)->tv_sec;  \
		(vvp)->tv_usec = (tvp)->tv_usec + (uvp)->tv_usec;  \
		if ((vvp)->tv_usec >= 1000000) {                   \
		    (vvp)->tv_sec++;                        \
			(vvp)->tv_usec -= 1000000;             \
		}                         \
	} while (0)

#define timersub(tvp, uvp, vvp)   \
	do {                          \
	    (vvp)->tv_sec = (tvp)->tv_sec - (uvp)->tv_sec;  \
		(vvp)->tv_usec = (tvp)->tv_usec - (uvp)->tv_usec;  \
		if ((vvp)->tv_usec < 0) {                   \
		    (vvp)->tv_sec--;                        \
			(vvp)->tv_usec += 1000000;             \
		}                       \
	} while (0)

/*#include <packon.h>
#pragma pack(push, 1)*/
struct ip {
	u_char    ip_hl:4, ip_v:4;
	u_char    ip_tos;
	u_short   ip_len;
	u_short   ip_id;
	u_short   ip_off;
#define IP_RP 0x8000
#define IP_DF 0x4000
#define IP_MF 0x2000
#define IP_OFFMASK 0x1fff
	u_char    ip_ttl;
	u_char    ip_p;
	u_short   ip_sum;
	struct in_addr ip_src, ip_dst;
};

struct ifaddrs {
	struct ifaddrs  *ifa_next;
	char		*ifa_name;
	unsigned int		 ifa_flags;
	struct sockaddr	*ifa_addr;
	struct sockaddr	*ifa_netmask;
	struct sockaddr	*ifa_dstaddr;
	void		*ifa_data;
};

struct udphdr {
	uint16_t uh_sport;
	uint16_t uh_dport;
	uint16_t uh_ulen;
	uint16_t uh_sum;
};

struct iovec {
	size_t len;
	char *buf;
};

#define iov_base buf
#define iov_len	len

struct ifa_msghdr {
	uint16_t         ifam_msglen;
	unsigned char    ifam_version;
	unsigned char    ifam_type;
	uint32_t         ifam_addrs;
	uint32_t         ifam_flags;
	uint16_t         ifam_index;
	uint32_t         ifam_metric;
};

struct ifdevmtu {
	int ifdm_current;
	int ifdm_min;
	int ifdm_max;
};

struct ifkpi {
	unsigned int  ifk_module_id;
	unsigned int  ifk_type;
	union {
		void *ifk_ptr;
		int ifk_value;
	} ifk_data;
};
#endif

#if defined(_WIN32)
int Win_getifaddrs(struct ifaddrs**);
#define getifaddrs(interfaces)  (int)Win_getifaddrs(interfaces)
int win_if_nametoindex(const char *);
#define if_nametoindex(x) win_if_nametoindex(x)
#endif

#define mtx_lock(arg1)
#define mtx_unlock(arg1)
#define mtx_assert(arg1,arg2)
#define MA_OWNED 7 /* sys/mutex.h typically on FreeBSD */
#if !defined(__FreeBSD__)
struct mtx {int dummy;};
#if !defined(__NetBSD__)
struct selinfo {int dummy;};
#endif
struct sx {int dummy;};
#endif

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
/* #include <sys/param.h>  in FreeBSD defines MSIZE */
/* #include <sys/ktr.h> */
/* #include <sys/systm.h> */
#if defined(HAVE_SYS_QUEUE_H)
#include <sys/queue.h>
#else
#include <user_queue.h>
#endif
#include <user_malloc.h>
/* #include <sys/kernel.h> */
/* #include <sys/sysctl.h> */
/* #include <sys/protosw.h> */
/* on FreeBSD, this results in a redefintion of SOCK(BUF)_(UN)LOCK and
 *  uknown type of struct mtx for sb_mtx in struct sockbuf */
#include "user_socketvar.h" /* MALLOC_DECLARE's M_PCB. Replacement for sys/socketvar.h */
/* #include <sys/jail.h> */
/* #include <sys/sysctl.h> */
#include <user_environment.h>
#include <user_atomic.h>
#include <user_mbuf.h>
/* #include <sys/uio.h> */
/* #include <sys/lock.h> */
#if defined(__FreeBSD__) && !defined(__Userspace__)
#include <sys/rwlock.h>
#endif
/* #include <sys/kthread.h> */
#if defined(__FreeBSD__) && !defined(__Userspace__)
#include <sys/priv.h>
#endif
/* #include <sys/random.h> */
#include <limits.h>
/* #include <machine/cpu.h> */

#if defined(__APPLE__)
/* was a 0 byte file.  needed for structs if_data(64) and net_event_data */
#include <net/if_var.h>
#endif
#if defined(__FreeBSD__)
#include <net/if_types.h>
/* #include <net/if_var.h> was a 0 byte file.  causes struct mtx redefinition */
#endif
/* OOTB only - dummy route used at the moment. should we port route to
 *  userspace as well? */
/* on FreeBSD, this results in a redefintion of struct route */
/* #include <net/route.h> */
#if !defined(_WIN32) && !defined(__native_client__)
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/in_systm.h>
#include <netinet/ip.h>
#endif
#if defined(HAVE_NETINET_IP_ICMP_H)
#include <netinet/ip_icmp.h>
#else
#include <user_ip_icmp.h>
#endif
/* #include <netinet/in_pcb.h> ported to userspace */
#include <user_inpcb.h>

/* for getifaddrs */
#include <sys/types.h>
#if !defined(_WIN32)
#if defined(INET) || defined(INET6)
#include <ifaddrs.h>
#endif

/* for ioctl */
#include <sys/ioctl.h>

/* for close, etc. */
#include <unistd.h>
/* for gettimeofday */
#include <sys/time.h>
#endif

/* lots of errno's used and needed in userspace */

/* for offsetof */
#include <stddef.h>

#if defined(SCTP_PROCESS_LEVEL_LOCKS) && !defined(_WIN32)
/* for pthread_mutex_lock, pthread_mutex_unlock, etc. */
#include <pthread.h>
#endif

#ifdef IPSEC
#include <netipsec/ipsec.h>
#include <netipsec/key.h>
#endif				/* IPSEC */

#ifdef INET6
#if defined(__FreeBSD__)
#include <sys/domain.h>
#endif
#ifdef IPSEC
#include <netipsec/ipsec6.h>
#endif
#if !defined(_WIN32)
#include <netinet/ip6.h>
#endif
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__linux__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(_WIN32) || defined(__EMSCRIPTEN__)
#include "user_ip6_var.h"
#else
#include <netinet6/ip6_var.h>
#endif
#if defined(__FreeBSD__)
#include <netinet6/in6_pcb.h>
#include <netinet6/scope6_var.h>
#endif
#endif /* INET6 */

#if defined(HAVE_SCTP_PEELOFF_SOCKOPT)
#include <sys/file.h>
#include <sys/filedesc.h>
#endif

#include "netinet/sctp_sha1.h"

#if defined(__FreeBSD__) && !defined(__Userspace__)
#include <netinet/ip_options.h>
#endif

#define SCTP_PRINTF(...)                                  \
	if (SCTP_BASE_VAR(debug_printf)) {                \
		SCTP_BASE_VAR(debug_printf)(__VA_ARGS__); \
	}

/* Declare all the malloc names for all the various mallocs */
MALLOC_DECLARE(SCTP_M_MAP);
MALLOC_DECLARE(SCTP_M_STRMI);
MALLOC_DECLARE(SCTP_M_STRMO);
MALLOC_DECLARE(SCTP_M_ASC_ADDR);
MALLOC_DECLARE(SCTP_M_ASC_IT);
MALLOC_DECLARE(SCTP_M_AUTH_CL);
MALLOC_DECLARE(SCTP_M_AUTH_KY);
MALLOC_DECLARE(SCTP_M_AUTH_HL);
MALLOC_DECLARE(SCTP_M_AUTH_IF);
MALLOC_DECLARE(SCTP_M_STRESET);
MALLOC_DECLARE(SCTP_M_CMSG);
MALLOC_DECLARE(SCTP_M_COPYAL);
MALLOC_DECLARE(SCTP_M_VRF);
MALLOC_DECLARE(SCTP_M_IFA);
MALLOC_DECLARE(SCTP_M_IFN);
MALLOC_DECLARE(SCTP_M_TIMW);
MALLOC_DECLARE(SCTP_M_MVRF);
MALLOC_DECLARE(SCTP_M_ITER);
MALLOC_DECLARE(SCTP_M_SOCKOPT);

#if defined(SCTP_LOCAL_TRACE_BUF)

#define SCTP_GET_CYCLECOUNT get_cyclecount()
#define SCTP_CTR6 sctp_log_trace

#else
#define SCTP_CTR6 CTR6
#endif

/* Empty ktr statement for _Userspace__ (similar to what is done for mac) */
#define	CTR6(m, d, p1, p2, p3, p4, p5, p6)



#define SCTP_BASE_INFO(__m) system_base_info.sctppcbinfo.__m
#define SCTP_BASE_STATS system_base_info.sctpstat
#define SCTP_BASE_STAT(__m)     system_base_info.sctpstat.__m
#define SCTP_BASE_SYSCTL(__m) system_base_info.sctpsysctl.__m
#define SCTP_BASE_VAR(__m) system_base_info.__m

/*
 *
 */
#if !defined(__APPLE__)
#define USER_ADDR_NULL	(NULL)		/* FIX ME: temp */
#endif

#include <netinet/sctp_constants.h>
#if defined(SCTP_DEBUG)
#define SCTPDBG(level, ...)					\
{								\
	do {							\
		if (SCTP_BASE_SYSCTL(sctp_debug_on) & level) {	\
			SCTP_PRINTF(__VA_ARGS__);		\
		}						\
	} while (0);						\
}
#define SCTPDBG_ADDR(level, addr)				\
{								\
	do {							\
		if (SCTP_BASE_SYSCTL(sctp_debug_on) & level ) {	\
		    sctp_print_address(addr);			\
		}						\
	} while (0);						\
}
#else
#define SCTPDBG(level, ...)
#define SCTPDBG_ADDR(level, addr)
#endif

#ifdef SCTP_LTRACE_CHUNKS
#define SCTP_LTRACE_CHK(a, b, c, d) if(sctp_logging_level & SCTP_LTRACE_CHUNK_ENABLE) CTR6(KTR_SUBSYS, "SCTP:%d[%d]:%x-%x-%x-%x", SCTP_LOG_CHUNK_PROC, 0, a, b, c, d)
#else
#define SCTP_LTRACE_CHK(a, b, c, d)
#endif

#ifdef SCTP_LTRACE_ERRORS
#define SCTP_LTRACE_ERR_RET_PKT(m, inp, stcb, net, file, err) \
	if (sctp_logging_level & SCTP_LTRACE_ERROR_ENABLE) \
		SCTP_PRINTF("mbuf:%p inp:%p stcb:%p net:%p file:%x line:%d error:%d\n", \
		            (void *)m, (void *)inp, (void *)stcb, (void *)net, file, __LINE__, err);
#define SCTP_LTRACE_ERR_RET(inp, stcb, net, file, err) \
	if (sctp_logging_level & SCTP_LTRACE_ERROR_ENABLE) \
		SCTP_PRINTF("inp:%p stcb:%p net:%p file:%x line:%d error:%d\n", \
		            (void *)inp, (void *)stcb, (void *)net, file, __LINE__, err);
#else
#define SCTP_LTRACE_ERR_RET_PKT(m, inp, stcb, net, file, err)
#define SCTP_LTRACE_ERR_RET(inp, stcb, net, file, err)
#endif


/*
 * Local address and interface list handling
 */
#define SCTP_MAX_VRF_ID		0
#define SCTP_SIZE_OF_VRF_HASH	3
#define SCTP_IFNAMSIZ		IFNAMSIZ
#define SCTP_DEFAULT_VRFID	0
#define SCTP_VRF_ADDR_HASH_SIZE	16
#define SCTP_VRF_IFN_HASH_SIZE	3
#define	SCTP_INIT_VRF_TABLEID(vrf)

#if !defined(_WIN32)
#define SCTP_IFN_IS_IFT_LOOP(ifn) (strncmp((ifn)->ifn_name, "lo", 2) == 0)
/* BSD definition */
/* #define SCTP_ROUTE_IS_REAL_LOOP(ro) ((ro)->ro_rt && (ro)->ro_rt->rt_ifa && (ro)->ro_rt->rt_ifa->ifa_ifp && (ro)->ro_rt->rt_ifa->ifa_ifp->if_type == IFT_LOOP) */
/* only used in IPv6 scenario, which isn't supported yet */
#define SCTP_ROUTE_IS_REAL_LOOP(ro) 0

/*
 * Access to IFN's to help with src-addr-selection
 */
/* This could return VOID if the index works but for BSD we provide both. */
#define SCTP_GET_IFN_VOID_FROM_ROUTE(ro) (void *)ro->ro_rt->rt_ifp
#define SCTP_GET_IF_INDEX_FROM_ROUTE(ro) 1 /* compiles...  TODO use routing socket to determine */
#define SCTP_ROUTE_HAS_VALID_IFN(ro) ((ro)->ro_rt && (ro)->ro_rt->rt_ifp)
#endif

/*
 * general memory allocation
 */
#define SCTP_MALLOC(var, type, size, name)				\
	do {								\
		MALLOC(var, type, size, name, M_NOWAIT);		\
	} while (0)

#define SCTP_FREE(var, type)	FREE(var, type)

#define SCTP_MALLOC_SONAME(var, type, size)				\
	do {								\
		MALLOC(var, type, size, M_SONAME, (M_WAITOK | M_ZERO));	\
	} while (0)

#define SCTP_FREE_SONAME(var)	FREE(var, M_SONAME)

#define SCTP_PROCESS_STRUCT struct proc *

/*
 * zone allocation functions
 */


#if defined(SCTP_SIMPLE_ALLOCATOR)
/*typedef size_t sctp_zone_t;*/
#define SCTP_ZONE_INIT(zone, name, size, number) { \
	zone = size; \
}

/* __Userspace__ SCTP_ZONE_GET: allocate element from the zone */
#define SCTP_ZONE_GET(zone, type)  \
        (type *)malloc(zone);


/* __Userspace__ SCTP_ZONE_FREE: free element from the zone */
#define SCTP_ZONE_FREE(zone, element) { \
	free(element);  \
}

#define SCTP_ZONE_DESTROY(zone)
#else
/*__Userspace__
  Compiling & linking notes: Needs libumem, which has been placed in ./user_lib
  All userspace header files are in ./user_include. Makefile will need the
  following.
  CFLAGS = -I./ -Wall
  LDFLAGS = -L./user_lib -R./user_lib -lumem
*/
#include "user_include/umem.h"

/* __Userspace__ SCTP_ZONE_INIT: initialize the zone */
/*
  __Userspace__
  No equivalent function to uma_zone_set_max added yet. (See SCTP_ZONE_INIT in sctp_os_bsd.h
  for reference). It may not be required as mentioned in
  http://nixdoc.net/man-pages/FreeBSD/uma_zalloc.9.html that
  max limits may not enforced on systems with more than one CPU.
*/
#define SCTP_ZONE_INIT(zone, name, size, number) { \
	zone = umem_cache_create(name, size, 0, NULL, NULL, NULL, NULL, NULL, 0); \
  }

/* __Userspace__ SCTP_ZONE_GET: allocate element from the zone */
#define SCTP_ZONE_GET(zone, type) \
        (type *)umem_cache_alloc(zone, UMEM_DEFAULT);


/* __Userspace__ SCTP_ZONE_FREE: free element from the zone */
#define SCTP_ZONE_FREE(zone, element) \
	umem_cache_free(zone, element);


/* __Userspace__ SCTP_ZONE_DESTROY: destroy the zone */
#define SCTP_ZONE_DESTROY(zone) \
	umem_cache_destroy(zone);
#endif

/*
 * __Userspace__ Defining sctp_hashinit_flags() and sctp_hashdestroy() for userland.
 */
void *sctp_hashinit_flags(int elements, struct malloc_type *type,
                    u_long *hashmask, int flags);
void
sctp_hashdestroy(void *vhashtbl, struct malloc_type *type, u_long hashmask);

void
sctp_hashfreedestroy(void *vhashtbl, struct malloc_type *type, u_long hashmask);


#define HASH_NOWAIT 0x00000001
#define HASH_WAITOK 0x00000002

/* M_PCB is MALLOC_DECLARE'd in sys/socketvar.h */
#define SCTP_HASH_INIT(size, hashmark) sctp_hashinit_flags(size, M_PCB, hashmark, HASH_NOWAIT)

#define SCTP_HASH_FREE(table, hashmark) sctp_hashdestroy(table, M_PCB, hashmark)

#define SCTP_HASH_FREE_DESTROY(table, hashmark)  sctp_hashfreedestroy(table, M_PCB, hashmark)
#define SCTP_M_COPYM	m_copym

/*
 * timers
 */
/* __Userspace__
 * user_sctp_callout.h has typedef struct sctp_callout sctp_os_timer_t;
 * which is used in the timer related functions such as
 * SCTP_OS_TIMER_INIT etc.
*/
#include <netinet/sctp_callout.h>

/* __Userspace__ Creating a receive thread */
#include <user_recv_thread.h>

/*__Userspace__ defining KTR_SUBSYS 1 as done in sctp_os_macosx.h */
#define KTR_SUBSYS 1

/* The packed define for 64 bit platforms */
#if !defined(_WIN32)
#define SCTP_PACKED __attribute__((packed))
#define SCTP_UNUSED __attribute__((unused))
#else
#define SCTP_PACKED
#define SCTP_UNUSED
#endif

/*
 * Functions
 */
/* Mbuf manipulation and access macros  */
#define SCTP_BUF_LEN(m) (m->m_len)
#define SCTP_BUF_NEXT(m) (m->m_next)
#define SCTP_BUF_NEXT_PKT(m) (m->m_nextpkt)
#define SCTP_BUF_RESV_UF(m, size) m->m_data += size
#define SCTP_BUF_AT(m, size) m->m_data + size
#define SCTP_BUF_IS_EXTENDED(m) (m->m_flags & M_EXT)
#define SCTP_BUF_EXTEND_SIZE(m) (m->m_ext.ext_size)
#define SCTP_BUF_TYPE(m) (m->m_type)
#define SCTP_BUF_RECVIF(m) (m->m_pkthdr.rcvif)
#define SCTP_BUF_PREPEND	M_PREPEND

#define SCTP_ALIGN_TO_END(m, len) if(m->m_flags & M_PKTHDR) { \
                                     MH_ALIGN(m, len); \
                                  } else if ((m->m_flags & M_EXT) == 0) { \
                                     M_ALIGN(m, len); \
                                  }

#if !defined(_WIN32)
#define SCTP_SNPRINTF(data, ...)						\
	if (snprintf(data, __VA_ARGS__) < 0) {					\
		data[0] = '\0';							\
	}
#endif

/* We make it so if you have up to 4 threads
 * writting based on the default size of
 * the packet log 65 k, that would be
 * 4 16k packets before we would hit
 * a problem.
 */
#define SCTP_PKTLOG_WRITERS_NEED_LOCK 3


/*
 * routes, output, etc.
 */

typedef struct sctp_route	sctp_route_t;
typedef struct sctp_rtentry	sctp_rtentry_t;

static inline void sctp_userspace_rtalloc(sctp_route_t *ro)
{
	if (ro->ro_rt != NULL) {
		ro->ro_rt->rt_refcnt++;
		return;
	}

	ro->ro_rt = (sctp_rtentry_t *) malloc(sizeof(sctp_rtentry_t));
	if (ro->ro_rt == NULL)
		return;

	/* initialize */
	memset(ro->ro_rt, 0, sizeof(sctp_rtentry_t));
	ro->ro_rt->rt_refcnt = 1;

	/* set MTU */
	/* TODO set this based on the ro->ro_dst, looking up MTU with routing socket */
#if 0
	if (userspace_rawroute == -1) {
		userspace_rawroute = socket(AF_ROUTE, SOCK_RAW, 0);
		if (userspace_rawroute == -1)
			return;
	}
#endif
	ro->ro_rt->rt_rmx.rmx_mtu = 1500; /* FIXME temporary solution */

	/* TODO enable the ability to obtain interface index of route for
	 *  SCTP_GET_IF_INDEX_FROM_ROUTE macro.
	 */
}
#define SCTP_RTALLOC(ro, vrf_id, fibnum) sctp_userspace_rtalloc((sctp_route_t *)ro)

/* dummy rtfree needed once user_route.h is included */
static inline void sctp_userspace_rtfree(sctp_rtentry_t *rt)
{
	if(rt == NULL) {
		return;
	}
	if(--rt->rt_refcnt > 0) {
		return;
	}
	free(rt);
}
#define rtfree(arg1) sctp_userspace_rtfree(arg1)


/*************************/
/*      MTU              */
/*************************/
int sctp_userspace_get_mtu_from_ifn(uint32_t if_index);

#define SCTP_GATHER_MTU_FROM_IFN_INFO(ifn, ifn_index) sctp_userspace_get_mtu_from_ifn(ifn_index)

#define SCTP_GATHER_MTU_FROM_ROUTE(sctp_ifa, sa, rt) ((rt != NULL) ? rt->rt_rmx.rmx_mtu : 0)

#define SCTP_SET_MTU_OF_ROUTE(sa, rt, mtu) do { \
                                              if (rt != NULL) \
                                                 rt->rt_rmx.rmx_mtu = mtu; \
                                           } while(0)


/*************************/
/* These are for logging */
/*************************/
/* return the base ext data pointer */
#define SCTP_BUF_EXTEND_BASE(m) (m->m_ext.ext_buf)
 /* return the refcnt of the data pointer */
#define SCTP_BUF_EXTEND_REFCNT(m) (*m->m_ext.ref_cnt)
/* return any buffer related flags, this is
 * used beyond logging for apple only.
 */
#define SCTP_BUF_GET_FLAGS(m) (m->m_flags)

/* For BSD this just accesses the M_PKTHDR length
 * so it operates on an mbuf with hdr flag. Other
 * O/S's may have seperate packet header and mbuf
 * chain pointers.. thus the macro.
 */
#define SCTP_HEADER_TO_CHAIN(m) (m)
#define SCTP_DETACH_HEADER_FROM_CHAIN(m)
#define SCTP_HEADER_LEN(m) ((m)->m_pkthdr.len)
#define SCTP_GET_HEADER_FOR_OUTPUT(o_pak) 0
#define SCTP_RELEASE_HEADER(m)
#define SCTP_RELEASE_PKT(m)	sctp_m_freem(m)

#define SCTP_GET_PKT_VRFID(m, vrf_id)  ((vrf_id = SCTP_DEFAULT_VRFID) != SCTP_DEFAULT_VRFID)



/* Attach the chain of data into the sendable packet. */
#define SCTP_ATTACH_CHAIN(pak, m, packet_length) do { \
                                                  pak = m; \
                                                  pak->m_pkthdr.len = packet_length; \
                          } while(0)

/* Other m_pkthdr type things */
/* FIXME need real definitions */
#define SCTP_IS_IT_BROADCAST(dst, m) 0
/* OOTB only #define SCTP_IS_IT_BROADCAST(dst, m) ((m->m_flags & M_PKTHDR) ? in_broadcast(dst, m->m_pkthdr.rcvif) : 0)  BSD def */
#define SCTP_IS_IT_LOOPBACK(m) 0
/* OOTB ONLY #define SCTP_IS_IT_LOOPBACK(m) ((m->m_flags & M_PKTHDR) && ((m->m_pkthdr.rcvif == NULL) || (m->m_pkthdr.rcvif->if_type == IFT_LOOP)))  BSD def */


/* This converts any input packet header
 * into the chain of data holders, for BSD
 * its a NOP.
 */

/* get the v6 hop limit */
#define SCTP_GET_HLIM(inp, ro) 128
#define IPv6_HOP_LIMIT 128

/* is the endpoint v6only? */
#define SCTP_IPV6_V6ONLY(sctp_inpcb)	((sctp_inpcb)->ip_inp.inp.inp_flags & IN6P_IPV6_V6ONLY)
/* is the socket non-blocking? */
#define SCTP_SO_IS_NBIO(so)	((so)->so_state & SS_NBIO)
#define SCTP_SET_SO_NBIO(so)	((so)->so_state |= SS_NBIO)
#define SCTP_CLEAR_SO_NBIO(so)	((so)->so_state &= ~SS_NBIO)
/* get the socket type */
#define SCTP_SO_TYPE(so)	((so)->so_type)

/* reserve sb space for a socket */
#define SCTP_SORESERVE(so, send, recv)	soreserve(so, send, recv)

/* wakeup a socket */
#define SCTP_SOWAKEUP(so)	wakeup(&(so)->so_timeo, so)
/* number of bytes ready to read */
#define SCTP_SBAVAIL(sb)	(sb)->sb_cc
#define SCTP_SB_INCR(sb, incr)			\
{						\
	atomic_add_int(&(sb)->sb_cc, incr);	\
}
#define SCTP_SB_DECR(sb, decr)					\
{								\
	SCTP_SAVE_ATOMIC_DECREMENT(&(sb)->sb_cc, (int)(decr));	\
}
/* clear the socket buffer state */
#define SCTP_SB_CLEAR(sb)	\
	(sb).sb_cc = 0;		\
	(sb).sb_mb = NULL;	\
	(sb).sb_mbcnt = 0;

#define SCTP_SB_LIMIT_RCV(so) so->so_rcv.sb_hiwat
#define SCTP_SB_LIMIT_SND(so) so->so_snd.sb_hiwat

#define SCTP_READ_RANDOM(buf, len)	read_random(buf, len)

#define SCTP_SHA1_CTX		struct sctp_sha1_context
#define SCTP_SHA1_INIT		sctp_sha1_init
#define SCTP_SHA1_UPDATE	sctp_sha1_update
#define SCTP_SHA1_FINAL(x,y)	sctp_sha1_final((unsigned char *)x, y)

/* start OOTB only stuff */
/* TODO IFT_LOOP is in net/if_types.h on Linux */
#define IFT_LOOP 0x18

/* sctp_pcb.h */

#if defined(_WIN32)
#define SHUT_RD 1
#define SHUT_WR 2
#define SHUT_RDWR 3
#endif
#define PRU_FLUSH_RD SHUT_RD
#define PRU_FLUSH_WR SHUT_WR
#define PRU_FLUSH_RDWR SHUT_RDWR

/* netinet/ip_var.h defintions are behind an if defined for _KERNEL on FreeBSD */
#define	IP_RAWOUTPUT		0x2


/* end OOTB only stuff */

#define AF_CONN 123
struct sockaddr_conn {
#ifdef HAVE_SCONN_LEN
	uint8_t sconn_len;
	uint8_t sconn_family;
#else
	uint16_t sconn_family;
#endif
	uint16_t sconn_port;
	void *sconn_addr;
};

typedef void *(*start_routine_t)(void *);

extern int
sctp_userspace_thread_create(userland_thread_t *thread, start_routine_t start_routine);

void
sctp_userspace_set_threadname(const char *name);

/*
 * SCTP protocol specific mbuf flags.
 */
#define	M_NOTIFICATION		M_PROTO5	/* SCTP notification */

/*
 * IP output routines
 */

/* Defining SCTP_IP_ID macro.
   In netinet/ip_output.c, we have u_short ip_id;
   In netinet/ip_var.h, we have extern u_short	ip_id; (enclosed within _KERNEL_)
   See static __inline uint16_t ip_newid(void) in netinet/ip_var.h
 */
#define SCTP_IP_ID(inp) (ip_id)

/* need sctphdr to get port in SCTP_IP_OUTPUT. sctphdr defined in sctp.h  */
#include <netinet/sctp.h>
extern void sctp_userspace_ip_output(int *result, struct mbuf *o_pak,
                                     sctp_route_t *ro, void *stcb,
                                     uint32_t vrf_id);

#define SCTP_IP_OUTPUT(result, o_pak, ro, inp, vrf_id) sctp_userspace_ip_output(&result, o_pak, ro, inp, vrf_id);

#if defined(INET6)
extern void sctp_userspace_ip6_output(int *result, struct mbuf *o_pak,
                                      struct route_in6 *ro, void *stcb,
                                      uint32_t vrf_id);
#define SCTP_IP6_OUTPUT(result, o_pak, ro, ifp, inp, vrf_id) sctp_userspace_ip6_output(&result, o_pak, ro, inp, vrf_id);
#endif



#if 0
#define SCTP_IP6_OUTPUT(result, o_pak, ro, ifp, stcb, vrf_id) \
{ \
	if (stcb && stcb->sctp_ep) \
		result = ip6_output(o_pak, \
				    ((struct inpcb *)(stcb->sctp_ep))->in6p_outputopts, \
				    (ro), 0, 0, ifp, NULL); \
	else \
		result = ip6_output(o_pak, NULL, (ro), 0, 0, ifp, NULL); \
}
#endif

struct mbuf *
sctp_get_mbuf_for_msg(unsigned int space_needed, int want_header, int how, int allonebuf, int type);


/* with the current included files, this is defined in Linux but
 *  in FreeBSD, it is behind a _KERNEL in sys/socket.h ...
 */
#if defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__native_client__)
/* stolen from /usr/include/sys/socket.h */
#define CMSG_ALIGN(n)   _ALIGN(n)
#elif defined(__NetBSD__)
#define CMSG_ALIGN(n)   (((n) + __ALIGNBYTES) & ~__ALIGNBYTES)
#elif defined(__APPLE__)
#if !defined(__DARWIN_ALIGNBYTES)
#define	__DARWIN_ALIGNBYTES	(sizeof(__darwin_size_t) - 1)
#endif

#if !defined(__DARWIN_ALIGN)
#define	__DARWIN_ALIGN(p)	((__darwin_size_t)((char *)(uintptr_t)(p) + __DARWIN_ALIGNBYTES) &~ __DARWIN_ALIGNBYTES)
#endif

#if !defined(__DARWIN_ALIGNBYTES32)
#define __DARWIN_ALIGNBYTES32     (sizeof(__uint32_t) - 1)
#endif

#if !defined(__DARWIN_ALIGN32)
#define __DARWIN_ALIGN32(p)       ((__darwin_size_t)((char *)(uintptr_t)(p) + __DARWIN_ALIGNBYTES32) &~ __DARWIN_ALIGNBYTES32)
#endif
#define CMSG_ALIGN(n)   __DARWIN_ALIGN32(n)
#endif
#define I_AM_HERE \
                do { \
			SCTP_PRINTF("%s:%d at %s\n", __FILE__, __LINE__ , __func__); \
		} while (0)

#ifndef timevalsub
#define timevalsub(tp1, tp2)                       \
	do {                                       \
		(tp1)->tv_sec -= (tp2)->tv_sec;    \
		(tp1)->tv_usec -= (tp2)->tv_usec;  \
		if ((tp1)->tv_usec < 0) {          \
			(tp1)->tv_sec--;           \
			(tp1)->tv_usec += 1000000; \
		}                                  \
	} while (0)
#endif

#if defined(__linux__)
#if !defined(TAILQ_FOREACH_SAFE)
#define TAILQ_FOREACH_SAFE(var, head, field, tvar)             \
         for ((var) = ((head)->tqh_first);                     \
              (var) && ((tvar) = TAILQ_NEXT((var), field), 1); \
              (var) = (tvar))
#endif
#if !defined(LIST_FOREACH_SAFE)
#define LIST_FOREACH_SAFE(var, head, field, tvar)              \
         for ((var) = ((head)->lh_first);                      \
              (var) && ((tvar) = LIST_NEXT((var), field), 1);  \
              (var) = (tvar))
#endif
#endif
#if defined(__DragonFly__)
#define TAILQ_FOREACH_SAFE TAILQ_FOREACH_MUTABLE
#define LIST_FOREACH_SAFE LIST_FOREACH_MUTABLE
#endif

#if defined(__native_client__)
#define	timercmp(tvp, uvp, cmp)						\
	(((tvp)->tv_sec == (uvp)->tv_sec) ?				\
	    ((tvp)->tv_usec cmp (uvp)->tv_usec) :			\
	    ((tvp)->tv_sec cmp (uvp)->tv_sec))
#endif

#define SCTP_IS_LISTENING(inp) ((inp->sctp_flags & SCTP_PCB_FLAGS_ACCEPTING) != 0)

#if defined(__APPLE__) || defined(__DragonFly__) || defined(__linux__) || defined(__native_client__) || defined(__NetBSD__) || defined(_WIN32) || defined(__Fuchsia__) || defined(__EMSCRIPTEN__)
int
timingsafe_bcmp(const void *, const void *, size_t);
#endif

#endif
