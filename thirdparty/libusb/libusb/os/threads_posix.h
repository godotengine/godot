/*
 * libusb synchronization using POSIX Threads
 *
 * Copyright © 2010 Peter Stuge <peter@stuge.se>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef LIBUSB_THREADS_POSIX_H
#define LIBUSB_THREADS_POSIX_H

#include <pthread.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#define usbi_mutex_static_t		pthread_mutex_t
#define USBI_MUTEX_INITIALIZER		PTHREAD_MUTEX_INITIALIZER
#define usbi_mutex_static_lock		pthread_mutex_lock
#define usbi_mutex_static_unlock	pthread_mutex_unlock

#define usbi_mutex_t			pthread_mutex_t
#define usbi_mutex_init(mutex)		pthread_mutex_init((mutex), NULL)
#define usbi_mutex_lock			pthread_mutex_lock
#define usbi_mutex_unlock		pthread_mutex_unlock
#define usbi_mutex_trylock		pthread_mutex_trylock
#define usbi_mutex_destroy		pthread_mutex_destroy

#define usbi_cond_t			pthread_cond_t
#define usbi_cond_init(cond)		pthread_cond_init((cond), NULL)
#define usbi_cond_wait			pthread_cond_wait
#define usbi_cond_broadcast		pthread_cond_broadcast
#define usbi_cond_destroy		pthread_cond_destroy

#define usbi_tls_key_t			pthread_key_t
#define usbi_tls_key_create(key)	pthread_key_create((key), NULL)
#define usbi_tls_key_get		pthread_getspecific
#define usbi_tls_key_set		pthread_setspecific
#define usbi_tls_key_delete		pthread_key_delete

int usbi_cond_timedwait(pthread_cond_t *cond,
	pthread_mutex_t *mutex, const struct timeval *tv);

int usbi_get_tid(void);

#endif /* LIBUSB_THREADS_POSIX_H */
