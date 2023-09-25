/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2001-2007, by Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2008-2011, by Randall Stewart. All rights reserved.
 * Copyright (c) 2008-2011, by Michael Tuexen. All rights reserved.
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
#ifndef __sctp_process_lock_h__
#define __sctp_process_lock_h__

/*
 * Need to yet define five atomic fuctions or
 * their equivalant.
 * - atomic_add_int(&foo, val) - add atomically the value
 * - atomic_fetchadd_int(&foo, val) - does same as atomic_add_int
 *				      but value it was is returned.
 * - atomic_subtract_int(&foo, val) - can be made from atomic_add_int()
 *
 * - atomic_cmpset_int(&foo, value, newvalue) - Does a set of newvalue
 *					        in foo if and only if
 *					        foo is value. Returns 0
 *					        on success.
 */

#ifdef SCTP_PER_SOCKET_LOCKING
/*
 * per socket level locking
 */

#if defined(_WIN32)
/* Lock for INFO stuff */
#define SCTP_INP_INFO_LOCK_INIT()
#define SCTP_INP_INFO_RLOCK()
#define SCTP_INP_INFO_RUNLOCK()
#define SCTP_INP_INFO_WLOCK()
#define SCTP_INP_INFO_WUNLOCK()
#define SCTP_INP_INFO_LOCK_ASSERT()
#define SCTP_INP_INFO_RLOCK_ASSERT()
#define SCTP_INP_INFO_WLOCK_ASSERT()
#define SCTP_INP_INFO_LOCK_DESTROY()
#define SCTP_IPI_COUNT_INIT()
#define SCTP_IPI_COUNT_DESTROY()
#else
#define SCTP_INP_INFO_LOCK_INIT()
#define SCTP_INP_INFO_RLOCK()
#define SCTP_INP_INFO_RUNLOCK()
#define SCTP_INP_INFO_WLOCK()
#define SCTP_INP_INFO_WUNLOCK()
#define SCTP_INP_INFO_LOCK_ASSERT()
#define SCTP_INP_INFO_RLOCK_ASSERT()
#define SCTP_INP_INFO_WLOCK_ASSERT()
#define SCTP_INP_INFO_LOCK_DESTROY()
#define SCTP_IPI_COUNT_INIT()
#define SCTP_IPI_COUNT_DESTROY()
#endif

/* Lock for INP */
#define SCTP_INP_LOCK_INIT(_inp)
#define SCTP_INP_LOCK_DESTROY(_inp)

#define SCTP_INP_RLOCK(_inp)
#define SCTP_INP_RUNLOCK(_inp)
#define SCTP_INP_WLOCK(_inp)
#define SCTP_INP_WUNLOCK(_inp)
#define SCTP_INP_RLOCK_ASSERT(_inp)
#define SCTP_INP_WLOCK_ASSERT(_inp)
#define SCTP_INP_INCR_REF(_inp)
#define SCTP_INP_DECR_REF(_inp)

#define SCTP_ASOC_CREATE_LOCK_INIT(_inp)
#define SCTP_ASOC_CREATE_LOCK_DESTROY(_inp)
#define SCTP_ASOC_CREATE_LOCK(_inp)
#define SCTP_ASOC_CREATE_UNLOCK(_inp)

#define SCTP_INP_READ_LOCK_INIT(_inp)
#define SCTP_INP_READ_LOCK_DESTROY(_inp)
#define SCTP_INP_READ_LOCK(_inp)
#define SCTP_INP_READ_UNLOCK(_inp)
#define SCTP_INP_READ_LOCK_ASSERT(_inp)

/* Lock for TCB */
#define SCTP_TCB_LOCK_INIT(_tcb)
#define SCTP_TCB_LOCK_DESTROY(_tcb)
#define SCTP_TCB_LOCK(_tcb)
#define SCTP_TCB_TRYLOCK(_tcb) 1
#define SCTP_TCB_UNLOCK(_tcb)
#define SCTP_TCB_UNLOCK_IFOWNED(_tcb)
#define SCTP_TCB_LOCK_ASSERT(_tcb)

#else
/*
 * per tcb level locking
 */
#define SCTP_IPI_COUNT_INIT()

#if defined(_WIN32)
#define SCTP_WQ_ADDR_INIT() \
	InitializeCriticalSection(&SCTP_BASE_INFO(wq_addr_mtx))
#define SCTP_WQ_ADDR_DESTROY() \
	DeleteCriticalSection(&SCTP_BASE_INFO(wq_addr_mtx))
#define SCTP_WQ_ADDR_LOCK() \
	EnterCriticalSection(&SCTP_BASE_INFO(wq_addr_mtx))
#define SCTP_WQ_ADDR_UNLOCK() \
	LeaveCriticalSection(&SCTP_BASE_INFO(wq_addr_mtx))
#define SCTP_WQ_ADDR_LOCK_ASSERT()

#if WINVER < 0x0600
#define SCTP_INP_INFO_LOCK_INIT() \
	InitializeCriticalSection(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_LOCK_DESTROY() \
	DeleteCriticalSection(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_RLOCK() \
	EnterCriticalSection(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_TRYLOCK()	\
	TryEnterCriticalSection(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_WLOCK() \
	EnterCriticalSection(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_RUNLOCK() \
	LeaveCriticalSection(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_WUNLOCK()	\
	LeaveCriticalSection(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_LOCK_ASSERT()
#define SCTP_INP_INFO_RLOCK_ASSERT()
#define SCTP_INP_INFO_WLOCK_ASSERT()
#else
#define SCTP_INP_INFO_LOCK_INIT() \
	InitializeSRWLock(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_LOCK_DESTROY()
#define SCTP_INP_INFO_RLOCK() \
	AcquireSRWLockShared(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_TRYLOCK() \
	TryAcquireSRWLockShared(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_WLOCK() \
	AcquireSRWLockExclusive(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_RUNLOCK() \
	ReleaseSRWLockShared(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_WUNLOCK() \
	ReleaseSRWLockExclusive(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_LOCK_ASSERT()
#define SCTP_INP_INFO_RLOCK_ASSERT()
#define SCTP_INP_INFO_WLOCK_ASSERT()
#endif

#define SCTP_IP_PKTLOG_INIT() \
	InitializeCriticalSection(&SCTP_BASE_INFO(ipi_pktlog_mtx))
#define SCTP_IP_PKTLOG_DESTROY () \
	DeleteCriticalSection(&SCTP_BASE_INFO(ipi_pktlog_mtx))
#define SCTP_IP_PKTLOG_LOCK() \
	EnterCriticalSection(&SCTP_BASE_INFO(ipi_pktlog_mtx))
#define SCTP_IP_PKTLOG_UNLOCK() \
	LeaveCriticalSection(&SCTP_BASE_INFO(ipi_pktlog_mtx))

/*
 * The INP locks we will use for locking an SCTP endpoint, so for example if
 * we want to change something at the endpoint level for example random_store
 * or cookie secrets we lock the INP level.
 */
#define SCTP_INP_READ_LOCK_INIT(_inp) \
	InitializeCriticalSection(&(_inp)->inp_rdata_mtx)
#define SCTP_INP_READ_LOCK_DESTROY(_inp) \
	DeleteCriticalSection(&(_inp)->inp_rdata_mtx)
#define SCTP_INP_READ_LOCK(_inp) \
	EnterCriticalSection(&(_inp)->inp_rdata_mtx)
#define SCTP_INP_READ_UNLOCK(_inp) \
	LeaveCriticalSection(&(_inp)->inp_rdata_mtx)
#define SCTP_INP_READ_LOCK_ASSERT(_inp)

#define SCTP_INP_LOCK_INIT(_inp) \
	InitializeCriticalSection(&(_inp)->inp_mtx)
#define SCTP_INP_LOCK_DESTROY(_inp) \
	DeleteCriticalSection(&(_inp)->inp_mtx)
#ifdef SCTP_LOCK_LOGGING
#define SCTP_INP_RLOCK(_inp) do { 						\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)	\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_INP);			\
		EnterCriticalSection(&(_inp)->inp_mtx);				\
} while (0)
#define SCTP_INP_WLOCK(_inp) do { 						\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)	\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_INP);			\
	EnterCriticalSection(&(_inp)->inp_mtx);					\
} while (0)
#else
#define SCTP_INP_RLOCK(_inp) \
	EnterCriticalSection(&(_inp)->inp_mtx)
#define SCTP_INP_WLOCK(_inp) \
	EnterCriticalSection(&(_inp)->inp_mtx)
#endif
#define SCTP_INP_RLOCK_ASSERT(_tcb)
#define SCTP_INP_WLOCK_ASSERT(_tcb)

#define SCTP_INP_INCR_REF(_inp) atomic_add_int(&((_inp)->refcount), 1)
#define SCTP_INP_DECR_REF(_inp) atomic_subtract_int(&((_inp)->refcount), 1)

#define SCTP_ASOC_CREATE_LOCK_INIT(_inp) \
	InitializeCriticalSection(&(_inp)->inp_create_mtx)
#define SCTP_ASOC_CREATE_LOCK_DESTROY(_inp) \
	DeleteCriticalSection(&(_inp)->inp_create_mtx)
#ifdef SCTP_LOCK_LOGGING
#define SCTP_ASOC_CREATE_LOCK(_inp) do {					\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)	\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_CREATE);		\
	EnterCriticalSection(&(_inp)->inp_create_mtx);				\
} while (0)
#else
#define SCTP_ASOC_CREATE_LOCK(_inp) \
	EnterCriticalSection(&(_inp)->inp_create_mtx)
#endif

#define SCTP_INP_RUNLOCK(_inp) \
	LeaveCriticalSection(&(_inp)->inp_mtx)
#define SCTP_INP_WUNLOCK(_inp) \
	LeaveCriticalSection(&(_inp)->inp_mtx)
#define SCTP_ASOC_CREATE_UNLOCK(_inp) \
	LeaveCriticalSection(&(_inp)->inp_create_mtx)

/*
 * For the majority of things (once we have found the association) we will
 * lock the actual association mutex. This will protect all the assoiciation
 * level queues and streams and such. We will need to lock the socket layer
 * when we stuff data up into the receiving sb_mb. I.e. we will need to do an
 * extra SOCKBUF_LOCK(&so->so_rcv) even though the association is locked.
 */

#define SCTP_TCB_LOCK_INIT(_tcb) \
	InitializeCriticalSection(&(_tcb)->tcb_mtx)
#define SCTP_TCB_LOCK_DESTROY(_tcb) \
	DeleteCriticalSection(&(_tcb)->tcb_mtx)
#ifdef SCTP_LOCK_LOGGING
#define SCTP_TCB_LOCK(_tcb) do {						\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)	\
		sctp_log_lock(_tcb->sctp_ep, _tcb, SCTP_LOG_LOCK_TCB);		\
	EnterCriticalSection(&(_tcb)->tcb_mtx);					\
} while (0)
#else
#define SCTP_TCB_LOCK(_tcb) \
	EnterCriticalSection(&(_tcb)->tcb_mtx)
#endif
#define SCTP_TCB_TRYLOCK(_tcb) 	((TryEnterCriticalSection(&(_tcb)->tcb_mtx)))
#define SCTP_TCB_UNLOCK(_tcb) \
	LeaveCriticalSection(&(_tcb)->tcb_mtx)
#define SCTP_TCB_LOCK_ASSERT(_tcb)

#else /* all Userspaces except Windows */
#define SCTP_WQ_ADDR_INIT() \
	(void)pthread_mutex_init(&SCTP_BASE_INFO(wq_addr_mtx), &SCTP_BASE_VAR(mtx_attr))
#define SCTP_WQ_ADDR_DESTROY() \
	(void)pthread_mutex_destroy(&SCTP_BASE_INFO(wq_addr_mtx))
#ifdef INVARIANTS
#define SCTP_WQ_ADDR_LOCK() \
	KASSERT(pthread_mutex_lock(&SCTP_BASE_INFO(wq_addr_mtx)) == 0, ("%s:%d: wq_addr_mtx already locked", __FILE__, __LINE__))
#define SCTP_WQ_ADDR_UNLOCK() \
	KASSERT(pthread_mutex_unlock(&SCTP_BASE_INFO(wq_addr_mtx)) == 0, ("%s:%d: wq_addr_mtx not locked", __FILE__, __LINE__))
#else
#define SCTP_WQ_ADDR_LOCK() \
	(void)pthread_mutex_lock(&SCTP_BASE_INFO(wq_addr_mtx))
#define SCTP_WQ_ADDR_UNLOCK() \
	(void)pthread_mutex_unlock(&SCTP_BASE_INFO(wq_addr_mtx))
#endif
#define SCTP_WQ_ADDR_LOCK_ASSERT() \
	KASSERT(pthread_mutex_trylock(&SCTP_BASE_INFO(wq_addr_mtx)) == EBUSY, ("%s:%d: wq_addr_mtx not locked", __FILE__, __LINE__))

#define SCTP_INP_INFO_LOCK_INIT() \
	(void)pthread_rwlock_init(&SCTP_BASE_INFO(ipi_ep_mtx), &SCTP_BASE_VAR(rwlock_attr))
#define SCTP_INP_INFO_LOCK_DESTROY() \
	(void)pthread_rwlock_destroy(&SCTP_BASE_INFO(ipi_ep_mtx))
#ifdef INVARIANTS
#define SCTP_INP_INFO_RLOCK() \
	KASSERT(pthread_rwlock_rdlock(&SCTP_BASE_INFO(ipi_ep_mtx)) == 0, ("%s%d: ipi_ep_mtx already locked", __FILE__, __LINE__))
#define SCTP_INP_INFO_WLOCK() \
	KASSERT(pthread_rwlock_wrlock(&SCTP_BASE_INFO(ipi_ep_mtx)) == 0, ("%s:%d: ipi_ep_mtx already locked", __FILE__, __LINE__))
#define SCTP_INP_INFO_RUNLOCK() \
	KASSERT(pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_ep_mtx)) == 0, ("%s:%d: ipi_ep_mtx not locked", __FILE__, __LINE__))
#define SCTP_INP_INFO_WUNLOCK() \
	KASSERT(pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_ep_mtx)) == 0, ("%s:%d: ipi_ep_mtx not locked", __FILE__, __LINE__))
#else
#define SCTP_INP_INFO_RLOCK() \
	(void)pthread_rwlock_rdlock(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_WLOCK() \
	(void)pthread_rwlock_wrlock(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_RUNLOCK() \
	(void)pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_ep_mtx))
#define SCTP_INP_INFO_WUNLOCK() \
	(void)pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_ep_mtx))
#endif
#define SCTP_INP_INFO_LOCK_ASSERT()
#define SCTP_INP_INFO_RLOCK_ASSERT()
#define SCTP_INP_INFO_WLOCK_ASSERT()
#define SCTP_INP_INFO_TRYLOCK() \
	(!(pthread_rwlock_tryrdlock(&SCTP_BASE_INFO(ipi_ep_mtx))))

#define SCTP_IP_PKTLOG_INIT() \
	(void)pthread_mutex_init(&SCTP_BASE_INFO(ipi_pktlog_mtx), &SCTP_BASE_VAR(mtx_attr))
#define SCTP_IP_PKTLOG_DESTROY() \
	(void)pthread_mutex_destroy(&SCTP_BASE_INFO(ipi_pktlog_mtx))
#ifdef INVARIANTS
#define SCTP_IP_PKTLOG_LOCK() \
	KASSERT(pthread_mutex_lock(&SCTP_BASE_INFO(ipi_pktlog_mtx)) == 0, ("%s:%d: ipi_pktlog_mtx already locked", __FILE__, __LINE__))
#define SCTP_IP_PKTLOG_UNLOCK() \
	KASSERT(pthread_mutex_unlock(&SCTP_BASE_INFO(ipi_pktlog_mtx)) == 0, ("%s:%d: ipi_pktlog_mtx not locked", __FILE__, __LINE__))
#else
#define SCTP_IP_PKTLOG_LOCK() \
	(void)pthread_mutex_lock(&SCTP_BASE_INFO(ipi_pktlog_mtx))
#define SCTP_IP_PKTLOG_UNLOCK() \
	(void)pthread_mutex_unlock(&SCTP_BASE_INFO(ipi_pktlog_mtx))
#endif


/*
 * The INP locks we will use for locking an SCTP endpoint, so for example if
 * we want to change something at the endpoint level for example random_store
 * or cookie secrets we lock the INP level.
 */
#define SCTP_INP_READ_LOCK_INIT(_inp) \
	(void)pthread_mutex_init(&(_inp)->inp_rdata_mtx, &SCTP_BASE_VAR(mtx_attr))
#define SCTP_INP_READ_LOCK_DESTROY(_inp) \
	(void)pthread_mutex_destroy(&(_inp)->inp_rdata_mtx)
#ifdef INVARIANTS
#define SCTP_INP_READ_LOCK(_inp) \
	KASSERT(pthread_mutex_lock(&(_inp)->inp_rdata_mtx) == 0, ("%s:%d: inp_rdata_mtx already locked", __FILE__, __LINE__))
#define SCTP_INP_READ_UNLOCK(_inp) \
	KASSERT(pthread_mutex_unlock(&(_inp)->inp_rdata_mtx) == 0, ("%s:%d: inp_rdata_mtx not locked", __FILE__, __LINE__))
#else
#define SCTP_INP_READ_LOCK(_inp) \
	(void)pthread_mutex_lock(&(_inp)->inp_rdata_mtx)
#define SCTP_INP_READ_UNLOCK(_inp) \
	(void)pthread_mutex_unlock(&(_inp)->inp_rdata_mtx)
#endif
#define SCTP_INP_READ_LOCK_ASSERT(_inp) \
	KASSERT(pthread_mutex_trylock(&(_inp)->inp_rdata_mtx) == EBUSY, ("%s:%d: inp_rdata_mtx not locked", __FILE__, __LINE__))

#define SCTP_INP_LOCK_INIT(_inp) \
	(void)pthread_mutex_init(&(_inp)->inp_mtx, &SCTP_BASE_VAR(mtx_attr))
#define SCTP_INP_LOCK_DESTROY(_inp) \
	(void)pthread_mutex_destroy(&(_inp)->inp_mtx)
#ifdef INVARIANTS
#ifdef SCTP_LOCK_LOGGING
#define SCTP_INP_RLOCK(_inp) do {											\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)						\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_INP);								\
	KASSERT(pthread_mutex_lock(&(_inp)->inp_mtx) == 0, ("%s:%d: inp_mtx already locked", __FILE__, __LINE__));	\
} while (0)
#define SCTP_INP_WLOCK(_inp) do {											\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)						\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_INP);								\
	KASSERT(pthread_mutex_lock(&(_inp)->inp_mtx) == 0, ("%s:%d: inp_mtx already locked", __FILE__, __LINE__));	\
} while (0)
#else
#define SCTP_INP_RLOCK(_inp) \
	KASSERT(pthread_mutex_lock(&(_inp)->inp_mtx) == 0, ("%s:%d: inp_mtx already locked", __FILE__, __LINE__))
#define SCTP_INP_WLOCK(_inp) \
	KASSERT(pthread_mutex_lock(&(_inp)->inp_mtx) == 0, ("%s:%d: inp_mtx already locked", __FILE__, __LINE__))
#endif
#define SCTP_INP_RUNLOCK(_inp) \
	KASSERT(pthread_mutex_unlock(&(_inp)->inp_mtx) == 0, ("%s:%d: inp_mtx not locked", __FILE__, __LINE__))
#define SCTP_INP_WUNLOCK(_inp) \
	KASSERT(pthread_mutex_unlock(&(_inp)->inp_mtx) == 0, ("%s:%d: inp_mtx not locked", __FILE__, __LINE__))
#else
#ifdef SCTP_LOCK_LOGGING
#define SCTP_INP_RLOCK(_inp) do {						\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)	\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_INP);			\
	(void)pthread_mutex_lock(&(_inp)->inp_mtx);				\
} while (0)
#define SCTP_INP_WLOCK(_inp) do {						\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)	\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_INP);			\
	(void)pthread_mutex_lock(&(_inp)->inp_mtx);				\
} while (0)
#else
#define SCTP_INP_RLOCK(_inp) \
	(void)pthread_mutex_lock(&(_inp)->inp_mtx)
#define SCTP_INP_WLOCK(_inp) \
	(void)pthread_mutex_lock(&(_inp)->inp_mtx)
#endif
#define SCTP_INP_RUNLOCK(_inp) \
	(void)pthread_mutex_unlock(&(_inp)->inp_mtx)
#define SCTP_INP_WUNLOCK(_inp) \
	(void)pthread_mutex_unlock(&(_inp)->inp_mtx)
#endif
#define SCTP_INP_RLOCK_ASSERT(_inp) \
	KASSERT(pthread_mutex_trylock(&(_inp)->inp_mtx) == EBUSY, ("%s:%d: inp_mtx not locked", __FILE__, __LINE__))
#define SCTP_INP_WLOCK_ASSERT(_inp) \
	KASSERT(pthread_mutex_trylock(&(_inp)->inp_mtx) == EBUSY, ("%s:%d: inp_mtx not locked", __FILE__, __LINE__))
#define SCTP_INP_INCR_REF(_inp) atomic_add_int(&((_inp)->refcount), 1)
#define SCTP_INP_DECR_REF(_inp) atomic_subtract_int(&((_inp)->refcount), 1)

#define SCTP_ASOC_CREATE_LOCK_INIT(_inp) \
	(void)pthread_mutex_init(&(_inp)->inp_create_mtx, &SCTP_BASE_VAR(mtx_attr))
#define SCTP_ASOC_CREATE_LOCK_DESTROY(_inp) \
	(void)pthread_mutex_destroy(&(_inp)->inp_create_mtx)
#ifdef INVARIANTS
#ifdef SCTP_LOCK_LOGGING
#define SCTP_ASOC_CREATE_LOCK(_inp) do {											\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)							\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_CREATE);								\
	KASSERT(pthread_mutex_lock(&(_inp)->inp_create_mtx) == 0, ("%s:%d: inp_create_mtx already locked", __FILE__, __LINE__));	\
} while (0)
#else
#define SCTP_ASOC_CREATE_LOCK(_inp) \
	KASSERT(pthread_mutex_lock(&(_inp)->inp_create_mtx) == 0, ("%s:%d: inp_create_mtx already locked", __FILE__, __LINE__))
#endif
#define SCTP_ASOC_CREATE_UNLOCK(_inp) \
	KASSERT(pthread_mutex_unlock(&(_inp)->inp_create_mtx) == 0, ("%s:%d: inp_create_mtx not locked", __FILE__, __LINE__))
#else
#ifdef SCTP_LOCK_LOGGING
#define SCTP_ASOC_CREATE_LOCK(_inp) do {					\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE)	\
		sctp_log_lock(_inp, NULL, SCTP_LOG_LOCK_CREATE);		\
	(void)pthread_mutex_lock(&(_inp)->inp_create_mtx);			\
} while (0)
#else
#define SCTP_ASOC_CREATE_LOCK(_inp) \
	(void)pthread_mutex_lock(&(_inp)->inp_create_mtx)
#endif
#define SCTP_ASOC_CREATE_UNLOCK(_inp) \
	(void)pthread_mutex_unlock(&(_inp)->inp_create_mtx)
#endif
/*
 * For the majority of things (once we have found the association) we will
 * lock the actual association mutex. This will protect all the assoiciation
 * level queues and streams and such. We will need to lock the socket layer
 * when we stuff data up into the receiving sb_mb. I.e. we will need to do an
 * extra SOCKBUF_LOCK(&so->so_rcv) even though the association is locked.
 */

#define SCTP_TCB_LOCK_INIT(_tcb) \
	(void)pthread_mutex_init(&(_tcb)->tcb_mtx, &SCTP_BASE_VAR(mtx_attr))
#define SCTP_TCB_LOCK_DESTROY(_tcb) \
	(void)pthread_mutex_destroy(&(_tcb)->tcb_mtx)
#ifdef INVARIANTS
#ifdef SCTP_LOCK_LOGGING
#define SCTP_TCB_LOCK(_tcb) do {											\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE) 						\
		sctp_log_lock(_tcb->sctp_ep, _tcb, SCTP_LOG_LOCK_TCB);							\
	KASSERT(pthread_mutex_lock(&(_tcb)->tcb_mtx) == 0, ("%s:%d: tcb_mtx already locked", __FILE__, __LINE__))	\
} while (0)
#else
#define SCTP_TCB_LOCK(_tcb) \
	KASSERT(pthread_mutex_lock(&(_tcb)->tcb_mtx) == 0, ("%s:%d: tcb_mtx already locked", __FILE__, __LINE__))
#endif
#define SCTP_TCB_UNLOCK(_tcb) \
	KASSERT(pthread_mutex_unlock(&(_tcb)->tcb_mtx) == 0, ("%s:%d: tcb_mtx not locked", __FILE__, __LINE__))
#else
#ifdef SCTP_LOCK_LOGGING
#define SCTP_TCB_LOCK(_tcb) do {						\
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_LOCK_LOGGING_ENABLE) 	\
		sctp_log_lock(_tcb->sctp_ep, _tcb, SCTP_LOG_LOCK_TCB);		\
	(void)pthread_mutex_lock(&(_tcb)->tcb_mtx);				\
} while (0)
#else
#define SCTP_TCB_LOCK(_tcb) \
	(void)pthread_mutex_lock(&(_tcb)->tcb_mtx)
#endif
#define SCTP_TCB_UNLOCK(_tcb) (void)pthread_mutex_unlock(&(_tcb)->tcb_mtx)
#endif
#define SCTP_TCB_LOCK_ASSERT(_tcb) \
	KASSERT(pthread_mutex_trylock(&(_tcb)->tcb_mtx) == EBUSY, ("%s:%d: tcb_mtx not locked", __FILE__, __LINE__))
#define SCTP_TCB_TRYLOCK(_tcb) (!(pthread_mutex_trylock(&(_tcb)->tcb_mtx)))
#endif

#endif /* SCTP_PER_SOCKET_LOCKING */


/*
 * common locks
 */

/* copied over to compile */
#define SCTP_INP_LOCK_CONTENDED(_inp) (0) /* Don't know if this is possible */
#define SCTP_INP_READ_CONTENDED(_inp) (0) /* Don't know if this is possible */
#define SCTP_ASOC_CREATE_LOCK_CONTENDED(_inp) (0) /* Don't know if this is possible */

/* socket locks */

#if defined(_WIN32)
#define SOCKBUF_LOCK_ASSERT(_so_buf)
#define SOCKBUF_LOCK(_so_buf) \
	EnterCriticalSection(&(_so_buf)->sb_mtx)
#define SOCKBUF_UNLOCK(_so_buf) \
	LeaveCriticalSection(&(_so_buf)->sb_mtx)
#define SOCK_LOCK(_so) \
	SOCKBUF_LOCK(&(_so)->so_rcv)
#define SOCK_UNLOCK(_so) \
	SOCKBUF_UNLOCK(&(_so)->so_rcv)
#else
#define SOCKBUF_LOCK_ASSERT(_so_buf) \
	KASSERT(pthread_mutex_trylock(SOCKBUF_MTX(_so_buf)) == EBUSY, ("%s:%d: socket buffer not locked", __FILE__, __LINE__))
#ifdef INVARIANTS
#define SOCKBUF_LOCK(_so_buf) \
	KASSERT(pthread_mutex_lock(SOCKBUF_MTX(_so_buf)) == 0, ("%s:%d: sockbuf_mtx already locked", __FILE__, __LINE__))
#define SOCKBUF_UNLOCK(_so_buf) \
	KASSERT(pthread_mutex_unlock(SOCKBUF_MTX(_so_buf)) == 0, ("%s:%d: sockbuf_mtx not locked", __FILE__, __LINE__))
#else
#define SOCKBUF_LOCK(_so_buf) \
	pthread_mutex_lock(SOCKBUF_MTX(_so_buf))
#define SOCKBUF_UNLOCK(_so_buf) \
	pthread_mutex_unlock(SOCKBUF_MTX(_so_buf))
#endif
#define SOCK_LOCK(_so) \
	SOCKBUF_LOCK(&(_so)->so_rcv)
#define SOCK_UNLOCK(_so) \
	SOCKBUF_UNLOCK(&(_so)->so_rcv)
#endif

#define SCTP_STATLOG_INIT_LOCK()
#define SCTP_STATLOG_LOCK()
#define SCTP_STATLOG_UNLOCK()
#define SCTP_STATLOG_DESTROY()

#if defined(_WIN32)
/* address list locks */
#if WINVER < 0x0600
#define SCTP_IPI_ADDR_INIT() \
	InitializeCriticalSection(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_DESTROY() \
	DeleteCriticalSection(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_RLOCK() \
	EnterCriticalSection(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_RUNLOCK() \
	LeaveCriticalSection(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_WLOCK() \
	EnterCriticalSection(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_WUNLOCK() \
	LeaveCriticalSection(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_LOCK_ASSERT()
#define SCTP_IPI_ADDR_WLOCK_ASSERT()
#else
#define SCTP_IPI_ADDR_INIT() \
	InitializeSRWLock(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_DESTROY()
#define SCTP_IPI_ADDR_RLOCK() \
	AcquireSRWLockShared(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_RUNLOCK() \
	ReleaseSRWLockShared(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_WLOCK() \
	AcquireSRWLockExclusive(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_WUNLOCK() \
	ReleaseSRWLockExclusive(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_LOCK_ASSERT()
#define SCTP_IPI_ADDR_WLOCK_ASSERT()
#endif

/* iterator locks */
#define SCTP_ITERATOR_LOCK_INIT() \
	InitializeCriticalSection(&sctp_it_ctl.it_mtx)
#define SCTP_ITERATOR_LOCK_DESTROY() \
	DeleteCriticalSection(&sctp_it_ctl.it_mtx)
#define SCTP_ITERATOR_LOCK() \
		EnterCriticalSection(&sctp_it_ctl.it_mtx)
#define SCTP_ITERATOR_UNLOCK() \
	LeaveCriticalSection(&sctp_it_ctl.it_mtx)

#define SCTP_IPI_ITERATOR_WQ_INIT() \
	InitializeCriticalSection(&sctp_it_ctl.ipi_iterator_wq_mtx)
#define SCTP_IPI_ITERATOR_WQ_DESTROY() \
	DeleteCriticalSection(&sctp_it_ctl.ipi_iterator_wq_mtx)
#define SCTP_IPI_ITERATOR_WQ_LOCK() \
	EnterCriticalSection(&sctp_it_ctl.ipi_iterator_wq_mtx)
#define SCTP_IPI_ITERATOR_WQ_UNLOCK() \
	LeaveCriticalSection(&sctp_it_ctl.ipi_iterator_wq_mtx)

#else
/* address list locks */
#define SCTP_IPI_ADDR_INIT() \
	(void)pthread_rwlock_init(&SCTP_BASE_INFO(ipi_addr_mtx), &SCTP_BASE_VAR(rwlock_attr))
#define SCTP_IPI_ADDR_DESTROY() \
	(void)pthread_rwlock_destroy(&SCTP_BASE_INFO(ipi_addr_mtx))
#ifdef INVARIANTS
#define SCTP_IPI_ADDR_RLOCK() \
	KASSERT(pthread_rwlock_rdlock(&SCTP_BASE_INFO(ipi_addr_mtx)) == 0, ("%s:%d: ipi_addr_mtx already locked", __FILE__, __LINE__))
#define SCTP_IPI_ADDR_RUNLOCK() \
	KASSERT(pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_addr_mtx)) == 0, ("%s:%d: ipi_addr_mtx not locked", __FILE__, __LINE__))
#define SCTP_IPI_ADDR_WLOCK() \
	KASSERT(pthread_rwlock_wrlock(&SCTP_BASE_INFO(ipi_addr_mtx)) == 0, ("%s:%d: ipi_addr_mtx already locked", __FILE__, __LINE__))
#define SCTP_IPI_ADDR_WUNLOCK() \
	KASSERT(pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_addr_mtx)) == 0, ("%s:%d: ipi_addr_mtx not locked", __FILE__, __LINE__))
#else
#define SCTP_IPI_ADDR_RLOCK() \
	(void)pthread_rwlock_rdlock(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_RUNLOCK() \
	(void)pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_WLOCK() \
	(void)pthread_rwlock_wrlock(&SCTP_BASE_INFO(ipi_addr_mtx))
#define SCTP_IPI_ADDR_WUNLOCK() \
	(void)pthread_rwlock_unlock(&SCTP_BASE_INFO(ipi_addr_mtx))
#endif
#define SCTP_IPI_ADDR_LOCK_ASSERT()
#define SCTP_IPI_ADDR_WLOCK_ASSERT()

/* iterator locks */
#define SCTP_ITERATOR_LOCK_INIT() \
	(void)pthread_mutex_init(&sctp_it_ctl.it_mtx, &SCTP_BASE_VAR(mtx_attr))
#define SCTP_ITERATOR_LOCK_DESTROY() \
	(void)pthread_mutex_destroy(&sctp_it_ctl.it_mtx)
#ifdef INVARIANTS
#define SCTP_ITERATOR_LOCK() \
	KASSERT(pthread_mutex_lock(&sctp_it_ctl.it_mtx) == 0, ("%s:%d: it_mtx already locked", __FILE__, __LINE__))
#define SCTP_ITERATOR_UNLOCK() \
	KASSERT(pthread_mutex_unlock(&sctp_it_ctl.it_mtx) == 0, ("%s:%d: it_mtx not locked", __FILE__, __LINE__))
#else
#define SCTP_ITERATOR_LOCK() \
	(void)pthread_mutex_lock(&sctp_it_ctl.it_mtx)
#define SCTP_ITERATOR_UNLOCK() \
	(void)pthread_mutex_unlock(&sctp_it_ctl.it_mtx)
#endif

#define SCTP_IPI_ITERATOR_WQ_INIT() \
	(void)pthread_mutex_init(&sctp_it_ctl.ipi_iterator_wq_mtx, &SCTP_BASE_VAR(mtx_attr))
#define SCTP_IPI_ITERATOR_WQ_DESTROY() \
	(void)pthread_mutex_destroy(&sctp_it_ctl.ipi_iterator_wq_mtx)
#ifdef INVARIANTS
#define SCTP_IPI_ITERATOR_WQ_LOCK() \
	KASSERT(pthread_mutex_lock(&sctp_it_ctl.ipi_iterator_wq_mtx) == 0, ("%s:%d: ipi_iterator_wq_mtx already locked", __FILE__, __LINE__))
#define SCTP_IPI_ITERATOR_WQ_UNLOCK() \
	KASSERT(pthread_mutex_unlock(&sctp_it_ctl.ipi_iterator_wq_mtx) == 0, ("%s:%d: ipi_iterator_wq_mtx not locked", __FILE__, __LINE__))
#else
#define SCTP_IPI_ITERATOR_WQ_LOCK() \
	(void)pthread_mutex_lock(&sctp_it_ctl.ipi_iterator_wq_mtx)
#define SCTP_IPI_ITERATOR_WQ_UNLOCK() \
	(void)pthread_mutex_unlock(&sctp_it_ctl.ipi_iterator_wq_mtx)
#endif
#endif

#define SCTP_INCR_EP_COUNT() \
	atomic_add_int(&SCTP_BASE_INFO(ipi_count_ep), 1)

#define SCTP_DECR_EP_COUNT() \
	atomic_subtract_int(&SCTP_BASE_INFO(ipi_count_ep), 1)

#define SCTP_INCR_ASOC_COUNT() \
	atomic_add_int(&SCTP_BASE_INFO(ipi_count_asoc), 1)

#define SCTP_DECR_ASOC_COUNT() \
	atomic_subtract_int(&SCTP_BASE_INFO(ipi_count_asoc), 1)

#define SCTP_INCR_LADDR_COUNT() \
	atomic_add_int(&SCTP_BASE_INFO(ipi_count_laddr), 1)

#define SCTP_DECR_LADDR_COUNT() \
	atomic_subtract_int(&SCTP_BASE_INFO(ipi_count_laddr), 1)

#define SCTP_INCR_RADDR_COUNT() \
	atomic_add_int(&SCTP_BASE_INFO(ipi_count_raddr), 1)

#define SCTP_DECR_RADDR_COUNT() \
	atomic_subtract_int(&SCTP_BASE_INFO(ipi_count_raddr), 1)

#define SCTP_INCR_CHK_COUNT() \
	atomic_add_int(&SCTP_BASE_INFO(ipi_count_chunk), 1)

#define SCTP_DECR_CHK_COUNT() \
	atomic_subtract_int(&SCTP_BASE_INFO(ipi_count_chunk), 1)

#define SCTP_INCR_READQ_COUNT() \
	atomic_add_int(&SCTP_BASE_INFO(ipi_count_readq), 1)

#define SCTP_DECR_READQ_COUNT() \
	atomic_subtract_int(&SCTP_BASE_INFO(ipi_count_readq), 1)

#define SCTP_INCR_STRMOQ_COUNT() \
	atomic_add_int(&SCTP_BASE_INFO(ipi_count_strmoq), 1)

#define SCTP_DECR_STRMOQ_COUNT() \
	atomic_subtract_int(&SCTP_BASE_INFO(ipi_count_strmoq), 1)

#endif
