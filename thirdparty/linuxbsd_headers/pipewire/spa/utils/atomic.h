/* Atomic operations */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_ATOMIC_H
#define SPA_ATOMIC_H

#ifdef __cplusplus
extern "C" {
#endif

#define SPA_ATOMIC_CAS(v,ov,nv)						\
({									\
	__typeof__(v) __ov = (ov);					\
	__atomic_compare_exchange_n(&(v), &__ov, (nv),			\
			0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);		\
})

#define SPA_ATOMIC_DEC(s)		__atomic_sub_fetch(&(s), 1, __ATOMIC_SEQ_CST)
#define SPA_ATOMIC_INC(s)		__atomic_add_fetch(&(s), 1, __ATOMIC_SEQ_CST)
#define SPA_ATOMIC_LOAD(s)		__atomic_load_n(&(s), __ATOMIC_SEQ_CST)
#define SPA_ATOMIC_STORE(s,v)		__atomic_store_n(&(s), (v), __ATOMIC_SEQ_CST)
#define SPA_ATOMIC_XCHG(s,v)		__atomic_exchange_n(&(s), (v), __ATOMIC_SEQ_CST)

#define SPA_SEQ_WRITE(s)		SPA_ATOMIC_INC(s)
#define SPA_SEQ_WRITE_SUCCESS(s1,s2)	((s1) + 1 == (s2) && ((s2) & 1) == 0)

#define SPA_SEQ_READ(s)			SPA_ATOMIC_LOAD(s)
#define SPA_SEQ_READ_SUCCESS(s1,s2)	((s1) == (s2) && ((s2) & 1) == 0)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_ATOMIC_H */
