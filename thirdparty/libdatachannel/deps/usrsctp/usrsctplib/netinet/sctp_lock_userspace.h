/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2001-2007, by Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2008-2012, by Randall Stewart. All rights reserved.
 * Copyright (c) 2008-2012, by Michael Tuexen. All rights reserved.
 * Copyright (c) 2008-2012, by Brad Penoff. All rights reserved.
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

#ifndef _NETINET_SCTP_LOCK_EMPTY_H_
#define _NETINET_SCTP_LOCK_EMPTY_H_

/*
 * Empty Lock declarations for all other platforms. Pre-process away to
 * nothing.
 */

/* __Userspace__ putting lock macros in same order as sctp_lock_bsd.h ...*/

#define SCTP_IPI_COUNT_INIT()

#define SCTP_STATLOG_INIT_LOCK()
#define SCTP_STATLOG_LOCK()
#define SCTP_STATLOG_UNLOCK()
#define SCTP_STATLOG_DESTROY()

#define SCTP_INP_INFO_LOCK_DESTROY()

#define SCTP_INP_INFO_LOCK_INIT()
#define SCTP_INP_INFO_RLOCK()
#define SCTP_INP_INFO_WLOCK()
#define SCTP_INP_INFO_TRYLOCK() 1
#define SCTP_INP_INFO_RUNLOCK()
#define SCTP_INP_INFO_WUNLOCK()
#define SCTP_INP_INFO_LOCK_ASSERT()
#define SCTP_INP_INFO_RLOCK_ASSERT()
#define SCTP_INP_INFO_WLOCK_ASSERT()

#define SCTP_WQ_ADDR_INIT()
#define SCTP_WQ_ADDR_DESTROY()
#define SCTP_WQ_ADDR_LOCK()
#define SCTP_WQ_ADDR_UNLOCK()
#define SCTP_WQ_ADDR_LOCK_ASSERT()

#define SCTP_IPI_ADDR_INIT()
#define SCTP_IPI_ADDR_DESTROY()
#define SCTP_IPI_ADDR_RLOCK()
#define SCTP_IPI_ADDR_WLOCK()
#define SCTP_IPI_ADDR_RUNLOCK()
#define SCTP_IPI_ADDR_WUNLOCK()
#define SCTP_IPI_ADDR_LOCK_ASSERT()
#define SCTP_IPI_ADDR_WLOCK_ASSERT()

#define SCTP_IPI_ITERATOR_WQ_INIT()
#define SCTP_IPI_ITERATOR_WQ_DESTROY()
#define SCTP_IPI_ITERATOR_WQ_LOCK()
#define SCTP_IPI_ITERATOR_WQ_UNLOCK()

#define SCTP_IP_PKTLOG_INIT()
#define SCTP_IP_PKTLOG_LOCK()
#define SCTP_IP_PKTLOG_UNLOCK()
#define SCTP_IP_PKTLOG_DESTROY()

#define SCTP_INP_READ_LOCK_INIT(_inp)
#define SCTP_INP_READ_LOCK_DESTROY(_inp)
#define SCTP_INP_READ_LOCK(_inp)
#define SCTP_INP_READ_UNLOCK(_inp)
#define SCTP_INP_READ_LOCK_ASSERT(_inp)

#define SCTP_INP_LOCK_INIT(_inp)
#define SCTP_ASOC_CREATE_LOCK_INIT(_inp)
#define SCTP_INP_LOCK_DESTROY(_inp)
#define SCTP_ASOC_CREATE_LOCK_DESTROY(_inp)

#define SCTP_INP_RLOCK(_inp)
#define SCTP_INP_WLOCK(_inp)
#define SCTP_INP_RLOCK_ASSERT(_inp)
#define SCTP_INP_WLOCK_ASSERT(_inp)

#define SCTP_INP_LOCK_CONTENDED(_inp) (0) /* Don't know if this is possible */

#define SCTP_INP_READ_CONTENDED(_inp) (0) /* Don't know if this is possible */

#define SCTP_ASOC_CREATE_LOCK_CONTENDED(_inp) (0) /* Don't know if this is possible */


#define SCTP_INP_INCR_REF(_inp)
#define SCTP_INP_DECR_REF(_inp)

#define SCTP_ASOC_CREATE_LOCK(_inp)

#define SCTP_INP_RUNLOCK(_inp)
#define SCTP_INP_WUNLOCK(_inp)
#define SCTP_ASOC_CREATE_UNLOCK(_inp)


#define SCTP_TCB_LOCK_INIT(_tcb)
#define SCTP_TCB_LOCK_DESTROY(_tcb)
#define SCTP_TCB_LOCK(_tcb)
#define SCTP_TCB_TRYLOCK(_tcb) 1
#define SCTP_TCB_UNLOCK(_tcb)
#define SCTP_TCB_UNLOCK_IFOWNED(_tcb)
#define SCTP_TCB_LOCK_ASSERT(_tcb)



#define SCTP_ITERATOR_LOCK_INIT()
#define SCTP_ITERATOR_LOCK()
#define SCTP_ITERATOR_UNLOCK()
#define SCTP_ITERATOR_LOCK_DESTROY()



#define SCTP_INCR_EP_COUNT() \
                do { \
		       sctppcbinfo.ipi_count_ep++; \
	        } while (0)

#define SCTP_DECR_EP_COUNT() \
                do { \
		       sctppcbinfo.ipi_count_ep--; \
	        } while (0)

#define SCTP_INCR_ASOC_COUNT() \
                do { \
	               sctppcbinfo.ipi_count_asoc++; \
	        } while (0)

#define SCTP_DECR_ASOC_COUNT() \
                do { \
	               sctppcbinfo.ipi_count_asoc--; \
	        } while (0)

#define SCTP_INCR_LADDR_COUNT() \
                do { \
	               sctppcbinfo.ipi_count_laddr++; \
	        } while (0)

#define SCTP_DECR_LADDR_COUNT() \
                do { \
	               sctppcbinfo.ipi_count_laddr--; \
	        } while (0)

#define SCTP_INCR_RADDR_COUNT() \
                do { \
 	               sctppcbinfo.ipi_count_raddr++; \
	        } while (0)

#define SCTP_DECR_RADDR_COUNT() \
                do { \
 	               sctppcbinfo.ipi_count_raddr--; \
	        } while (0)

#define SCTP_INCR_CHK_COUNT() \
                do { \
  	               sctppcbinfo.ipi_count_chunk++; \
	        } while (0)

#define SCTP_DECR_CHK_COUNT() \
                do { \
  	               sctppcbinfo.ipi_count_chunk--; \
	        } while (0)

#define SCTP_INCR_READQ_COUNT() \
                do { \
		       sctppcbinfo.ipi_count_readq++; \
	        } while (0)

#define SCTP_DECR_READQ_COUNT() \
                do { \
		       sctppcbinfo.ipi_count_readq--; \
	        } while (0)

#define SCTP_INCR_STRMOQ_COUNT() \
                do { \
		       sctppcbinfo.ipi_count_strmoq++; \
	        } while (0)

#define SCTP_DECR_STRMOQ_COUNT() \
                do { \
		       sctppcbinfo.ipi_count_strmoq--; \
	        } while (0)


/* these were in sctp_lock_empty.h but aren't in sctp_lock_bsd.h ... */
#if 0
#define SCTP_IPI_ADDR_LOCK()
#define SCTP_IPI_ADDR_UNLOCK()
#endif


/* These were in sctp_lock_empty.h because they were commented out within
 *  within user_include/user_socketvar.h .  If they are NOT commented out
 *  in user_socketvar.h (because that seems the more natural place for them
 *  to live), then change this "if" to 0.  Keep the "if" as 1 if these ARE
 *  indeed commented out in user_socketvar.h .
 *
 * This modularity is kept so this file can easily be chosen as an alternative
 *  to SCTP_PROCESS_LEVEL_LOCKS.  If one defines SCTP_PROCESS_LEVEL_LOCKS in
 *  user_include/opt_sctp.h, then the file sctp_process_lock.h (which we didn't
 *  implement) is used, and that declares these locks already (so using
 *  SCTP_PROCESS_LEVEL_LOCKS *requires* that these defintions be commented out
 *  in user_socketvar.h).
 */
#if 1
#define SOCK_LOCK(_so)
#define SOCK_UNLOCK(_so)
#define SOCKBUF_LOCK(_so_buf)
#define SOCKBUF_UNLOCK(_so_buf)
#define SOCKBUF_LOCK_ASSERT(_so_buf)
#endif

#endif
