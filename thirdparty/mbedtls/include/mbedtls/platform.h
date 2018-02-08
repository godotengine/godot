/**
 * \file platform.h
 *
 * \brief The Mbed TLS platform abstraction layer.
 */
/*
 *  Copyright (C) 2006-2018, Arm Limited (or its affiliates), All Rights Reserved
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  This file is part of Mbed TLS (https://tls.mbed.org)
 */
#ifndef MBEDTLS_PLATFORM_H
#define MBEDTLS_PLATFORM_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_HAVE_TIME)
#include "mbedtls/platform_time.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h or define them on the compiler command line.
 * \{
 */

#if !defined(MBEDTLS_PLATFORM_NO_STD_FUNCTIONS)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if !defined(MBEDTLS_PLATFORM_STD_SNPRINTF)
#if defined(_WIN32)
#define MBEDTLS_PLATFORM_STD_SNPRINTF   mbedtls_platform_win32_snprintf /**< The default \c snprintf function to use.  */
#else
#define MBEDTLS_PLATFORM_STD_SNPRINTF   snprintf /**< The default \c snprintf function to use.  */
#endif
#endif
#if !defined(MBEDTLS_PLATFORM_STD_PRINTF)
#define MBEDTLS_PLATFORM_STD_PRINTF   printf /**< The default \c printf function to use. */
#endif
#if !defined(MBEDTLS_PLATFORM_STD_FPRINTF)
#define MBEDTLS_PLATFORM_STD_FPRINTF fprintf /**< The default \c fprintf function to use. */
#endif
#if !defined(MBEDTLS_PLATFORM_STD_CALLOC)
#define MBEDTLS_PLATFORM_STD_CALLOC   calloc /**< The default \c calloc function to use. */
#endif
#if !defined(MBEDTLS_PLATFORM_STD_FREE)
#define MBEDTLS_PLATFORM_STD_FREE       free /**< The default \c free function to use. */
#endif
#if !defined(MBEDTLS_PLATFORM_STD_EXIT)
#define MBEDTLS_PLATFORM_STD_EXIT      exit /**< The default \c exit function to use. */
#endif
#if !defined(MBEDTLS_PLATFORM_STD_TIME)
#define MBEDTLS_PLATFORM_STD_TIME       time    /**< The default \c time function to use. */
#endif
#if !defined(MBEDTLS_PLATFORM_STD_EXIT_SUCCESS)
#define MBEDTLS_PLATFORM_STD_EXIT_SUCCESS  EXIT_SUCCESS /**< The default exit value to use. */
#endif
#if !defined(MBEDTLS_PLATFORM_STD_EXIT_FAILURE)
#define MBEDTLS_PLATFORM_STD_EXIT_FAILURE  EXIT_FAILURE /**< The default exit value to use. */
#endif
#if defined(MBEDTLS_FS_IO)
#if !defined(MBEDTLS_PLATFORM_STD_NV_SEED_READ)
#define MBEDTLS_PLATFORM_STD_NV_SEED_READ   mbedtls_platform_std_nv_seed_read
#endif
#if !defined(MBEDTLS_PLATFORM_STD_NV_SEED_WRITE)
#define MBEDTLS_PLATFORM_STD_NV_SEED_WRITE  mbedtls_platform_std_nv_seed_write
#endif
#if !defined(MBEDTLS_PLATFORM_STD_NV_SEED_FILE)
#define MBEDTLS_PLATFORM_STD_NV_SEED_FILE   "seedfile"
#endif
#endif /* MBEDTLS_FS_IO */
#else /* MBEDTLS_PLATFORM_NO_STD_FUNCTIONS */
#if defined(MBEDTLS_PLATFORM_STD_MEM_HDR)
#include MBEDTLS_PLATFORM_STD_MEM_HDR
#endif
#endif /* MBEDTLS_PLATFORM_NO_STD_FUNCTIONS */


/* \} name SECTION: Module settings */

/*
 * The function pointers for calloc and free
 */
#if defined(MBEDTLS_PLATFORM_MEMORY)
#if defined(MBEDTLS_PLATFORM_FREE_MACRO) && \
    defined(MBEDTLS_PLATFORM_CALLOC_MACRO)
#define mbedtls_free       MBEDTLS_PLATFORM_FREE_MACRO
#define mbedtls_calloc     MBEDTLS_PLATFORM_CALLOC_MACRO
#else
/* For size_t */
#include <stddef.h>
extern void * (*mbedtls_calloc)( size_t n, size_t size );
extern void (*mbedtls_free)( void *ptr );

/**
 * \brief   This function allows configuring custom memory-management functions.
 *
 * \param calloc_func   The \c calloc function implementation.
 * \param free_func     The \c free function implementation.
 *
 * \return              \c 0.
 */
int mbedtls_platform_set_calloc_free( void * (*calloc_func)( size_t, size_t ),
                              void (*free_func)( void * ) );
#endif /* MBEDTLS_PLATFORM_FREE_MACRO && MBEDTLS_PLATFORM_CALLOC_MACRO */
#else /* !MBEDTLS_PLATFORM_MEMORY */
#define mbedtls_free       free
#define mbedtls_calloc     calloc
#endif /* MBEDTLS_PLATFORM_MEMORY && !MBEDTLS_PLATFORM_{FREE,CALLOC}_MACRO */

/*
 * The function pointers for fprintf
 */
#if defined(MBEDTLS_PLATFORM_FPRINTF_ALT)
/* We need FILE * */
#include <stdio.h>
extern int (*mbedtls_fprintf)( FILE *stream, const char *format, ... );

/**
 * \brief   This function allows configuring a custom \p fprintf function pointer.
 *
 * \param fprintf_func   The \c fprintf function implementation.
 *
 * \return               \c 0.
 */
int mbedtls_platform_set_fprintf( int (*fprintf_func)( FILE *stream, const char *,
                                               ... ) );
#else
#if defined(MBEDTLS_PLATFORM_FPRINTF_MACRO)
#define mbedtls_fprintf    MBEDTLS_PLATFORM_FPRINTF_MACRO
#else
#define mbedtls_fprintf    fprintf
#endif /* MBEDTLS_PLATFORM_FPRINTF_MACRO */
#endif /* MBEDTLS_PLATFORM_FPRINTF_ALT */

/*
 * The function pointers for printf
 */
#if defined(MBEDTLS_PLATFORM_PRINTF_ALT)
extern int (*mbedtls_printf)( const char *format, ... );

/**
 * \brief    This function allows configuring a custom \c printf function
 *           pointer.
 *
 * \param printf_func   The \c printf function implementation.
 *
 * \return              \c 0 on success.
 */
int mbedtls_platform_set_printf( int (*printf_func)( const char *, ... ) );
#else /* !MBEDTLS_PLATFORM_PRINTF_ALT */
#if defined(MBEDTLS_PLATFORM_PRINTF_MACRO)
#define mbedtls_printf     MBEDTLS_PLATFORM_PRINTF_MACRO
#else
#define mbedtls_printf     printf
#endif /* MBEDTLS_PLATFORM_PRINTF_MACRO */
#endif /* MBEDTLS_PLATFORM_PRINTF_ALT */

/*
 * The function pointers for snprintf
 *
 * The snprintf implementation should conform to C99:
 * - it *must* always correctly zero-terminate the buffer
 *   (except when n == 0, then it must leave the buffer untouched)
 * - however it is acceptable to return -1 instead of the required length when
 *   the destination buffer is too short.
 */
#if defined(_WIN32)
/* For Windows (inc. MSYS2), we provide our own fixed implementation */
int mbedtls_platform_win32_snprintf( char *s, size_t n, const char *fmt, ... );
#endif

#if defined(MBEDTLS_PLATFORM_SNPRINTF_ALT)
extern int (*mbedtls_snprintf)( char * s, size_t n, const char * format, ... );

/**
 * \brief   This function allows configuring a custom \c snprintf function
 *          pointer.
 *
 * \param snprintf_func   The \c snprintf function implementation.
 *
 * \return    \c 0 on success.
 */
int mbedtls_platform_set_snprintf( int (*snprintf_func)( char * s, size_t n,
                                                 const char * format, ... ) );
#else /* MBEDTLS_PLATFORM_SNPRINTF_ALT */
#if defined(MBEDTLS_PLATFORM_SNPRINTF_MACRO)
#define mbedtls_snprintf   MBEDTLS_PLATFORM_SNPRINTF_MACRO
#else
#define mbedtls_snprintf   snprintf
#endif /* MBEDTLS_PLATFORM_SNPRINTF_MACRO */
#endif /* MBEDTLS_PLATFORM_SNPRINTF_ALT */

/*
 * The function pointers for exit
 */
#if defined(MBEDTLS_PLATFORM_EXIT_ALT)
extern void (*mbedtls_exit)( int status );

/**
 * \brief   This function allows configuring a custom \c exit function
 *          pointer.
 *
 * \param exit_func   The \c exit function implementation.
 *
 * \return  \c 0 on success.
 */
int mbedtls_platform_set_exit( void (*exit_func)( int status ) );
#else
#if defined(MBEDTLS_PLATFORM_EXIT_MACRO)
#define mbedtls_exit   MBEDTLS_PLATFORM_EXIT_MACRO
#else
#define mbedtls_exit   exit
#endif /* MBEDTLS_PLATFORM_EXIT_MACRO */
#endif /* MBEDTLS_PLATFORM_EXIT_ALT */

/*
 * The default exit values
 */
#if defined(MBEDTLS_PLATFORM_STD_EXIT_SUCCESS)
#define MBEDTLS_EXIT_SUCCESS MBEDTLS_PLATFORM_STD_EXIT_SUCCESS
#else
#define MBEDTLS_EXIT_SUCCESS 0
#endif
#if defined(MBEDTLS_PLATFORM_STD_EXIT_FAILURE)
#define MBEDTLS_EXIT_FAILURE MBEDTLS_PLATFORM_STD_EXIT_FAILURE
#else
#define MBEDTLS_EXIT_FAILURE 1
#endif

/*
 * The function pointers for reading from and writing a seed file to
 * Non-Volatile storage (NV) in a platform-independent way
 *
 * Only enabled when the NV seed entropy source is enabled
 */
#if defined(MBEDTLS_ENTROPY_NV_SEED)
#if !defined(MBEDTLS_PLATFORM_NO_STD_FUNCTIONS) && defined(MBEDTLS_FS_IO)
/* Internal standard platform definitions */
int mbedtls_platform_std_nv_seed_read( unsigned char *buf, size_t buf_len );
int mbedtls_platform_std_nv_seed_write( unsigned char *buf, size_t buf_len );
#endif

#if defined(MBEDTLS_PLATFORM_NV_SEED_ALT)
extern int (*mbedtls_nv_seed_read)( unsigned char *buf, size_t buf_len );
extern int (*mbedtls_nv_seed_write)( unsigned char *buf, size_t buf_len );

/**
 * \brief   This function allows configuring custom seed file writing and
 *          reading functions.
 *
 * \param   nv_seed_read_func   The seed reading function implementation.
 * \param   nv_seed_write_func  The seed writing function implementation.
 *
 * \return  \c 0 on success.
 */
int mbedtls_platform_set_nv_seed(
            int (*nv_seed_read_func)( unsigned char *buf, size_t buf_len ),
            int (*nv_seed_write_func)( unsigned char *buf, size_t buf_len )
            );
#else
#if defined(MBEDTLS_PLATFORM_NV_SEED_READ_MACRO) && \
    defined(MBEDTLS_PLATFORM_NV_SEED_WRITE_MACRO)
#define mbedtls_nv_seed_read    MBEDTLS_PLATFORM_NV_SEED_READ_MACRO
#define mbedtls_nv_seed_write   MBEDTLS_PLATFORM_NV_SEED_WRITE_MACRO
#else
#define mbedtls_nv_seed_read    mbedtls_platform_std_nv_seed_read
#define mbedtls_nv_seed_write   mbedtls_platform_std_nv_seed_write
#endif
#endif /* MBEDTLS_PLATFORM_NV_SEED_ALT */
#endif /* MBEDTLS_ENTROPY_NV_SEED */

#if !defined(MBEDTLS_PLATFORM_SETUP_TEARDOWN_ALT)

/**
 * \brief   The platform context structure.
 *
 * \note    This structure may be used to assist platform-specific
 *          setup or teardown operations.
 */
typedef struct {
    char dummy; /**< Placeholder member, as empty structs are not portable. */
}
mbedtls_platform_context;

#else
#include "platform_alt.h"
#endif /* !MBEDTLS_PLATFORM_SETUP_TEARDOWN_ALT */

/**
 * \brief   This function performs any platform initialization operations.
 *
 * \param   ctx     The Mbed TLS context.
 *
 * \return  \c 0 on success.
 *
 * \note    This function is intended to allow platform-specific initialization,
 *          and should be called before any other library functions. Its
 *          implementation is platform-specific, and unless
 *          platform-specific code is provided, it does nothing.
 *
 *          Its use and whether it is necessary to call it is dependent on the
 *          platform.
 */
int mbedtls_platform_setup( mbedtls_platform_context *ctx );
/**
 * \brief   This function performs any platform teardown operations.
 *
 * \param   ctx     The Mbed TLS context.
 *
 * \note    This function should be called after every other Mbed TLS module
 *          has been correctly freed using the appropriate free function.
 *          Its implementation is platform-specific, and unless
 *          platform-specific code is provided, it does nothing.
 *
 *          Its use and whether it is necessary to call it is dependent on the
 *          platform.
 */
void mbedtls_platform_teardown( mbedtls_platform_context *ctx );

#ifdef __cplusplus
}
#endif

#endif /* platform.h */
