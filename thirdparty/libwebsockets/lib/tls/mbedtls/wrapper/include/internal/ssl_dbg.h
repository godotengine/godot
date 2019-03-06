// Copyright 2015-2016 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _SSL_DEBUG_H_
#define _SSL_DEBUG_H_

#include "platform/ssl_port.h"

#ifdef __cplusplus
 extern "C" {
#endif

#ifdef CONFIG_OPENSSL_DEBUG_LEVEL
    #define SSL_DEBUG_LEVEL CONFIG_OPENSSL_DEBUG_LEVEL
#else
    #define SSL_DEBUG_LEVEL 0
#endif

#define SSL_DEBUG_ON  (SSL_DEBUG_LEVEL + 1)
#define SSL_DEBUG_OFF (SSL_DEBUG_LEVEL - 1)

#ifdef CONFIG_OPENSSL_DEBUG
    #ifndef SSL_DEBUG_LOG
        #error "SSL_DEBUG_LOG is not defined"
    #endif

    #ifndef SSL_DEBUG_FL
        #define SSL_DEBUG_FL "\n"
    #endif

    #define SSL_SHOW_LOCATION()                         \
        SSL_DEBUG_LOG("SSL assert : %s %d\n",           \
            __FILE__, __LINE__)

    #define SSL_DEBUG(level, fmt, ...)                  \
    {                                                   \
        if (level > SSL_DEBUG_LEVEL) {                  \
            SSL_DEBUG_LOG(fmt SSL_DEBUG_FL, ##__VA_ARGS__); \
        }                                               \
    }
#else /* CONFIG_OPENSSL_DEBUG */
    #define SSL_SHOW_LOCATION()

    #define SSL_DEBUG(level, fmt, ...)
#endif /* CONFIG_OPENSSL_DEBUG */

/**
 * OpenSSL assert function
 *
 * if select "CONFIG_OPENSSL_ASSERT_DEBUG", SSL_ASSERT* will show error file name and line
 * if select "CONFIG_OPENSSL_ASSERT_EXIT", SSL_ASSERT* will just return error code.
 * if select "CONFIG_OPENSSL_ASSERT_DEBUG_EXIT" SSL_ASSERT* will show error file name and line,
 * then return error code.
 * if select "CONFIG_OPENSSL_ASSERT_DEBUG_BLOCK", SSL_ASSERT* will show error file name and line,
 * then block here with "while (1)"
 *
 * SSL_ASSERT1 may will return "-1", so function's return argument is integer.
 * SSL_ASSERT2 may will return "NULL", so function's return argument is a point.
 * SSL_ASSERT2 may will return nothing, so function's return argument is "void".
 */
#if defined(CONFIG_OPENSSL_ASSERT_DEBUG)
    #define SSL_ASSERT1(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
        }                                               \
    }

    #define SSL_ASSERT2(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
        }                                               \
    }

    #define SSL_ASSERT3(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
        }                                               \
    }
#elif defined(CONFIG_OPENSSL_ASSERT_EXIT)
    #define SSL_ASSERT1(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            return -1;                                  \
        }                                               \
    }

    #define SSL_ASSERT2(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            return NULL;                                \
        }                                               \
    }

    #define SSL_ASSERT3(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            return ;                                    \
        }                                               \
    }
#elif defined(CONFIG_OPENSSL_ASSERT_DEBUG_EXIT)
    #define SSL_ASSERT1(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
            return -1;                                  \
        }                                               \
    }

    #define SSL_ASSERT2(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
            return NULL;                                \
        }                                               \
    }

    #define SSL_ASSERT3(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
            return ;                                    \
        }                                               \
    }
#elif defined(CONFIG_OPENSSL_ASSERT_DEBUG_BLOCK)
    #define SSL_ASSERT1(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
            while (1);                                  \
        }                                               \
    }

    #define SSL_ASSERT2(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
            while (1);                                  \
        }                                               \
    }

    #define SSL_ASSERT3(s)                              \
    {                                                   \
        if (!(s)) {                                     \
            SSL_SHOW_LOCATION();                        \
            while (1);                                  \
        }                                               \
    }
#else
    #define SSL_ASSERT1(s)
    #define SSL_ASSERT2(s)
    #define SSL_ASSERT3(s)
#endif

#define SSL_PLATFORM_DEBUG_LEVEL SSL_DEBUG_OFF
#define SSL_PLATFORM_ERROR_LEVEL SSL_DEBUG_ON

#define SSL_CERT_DEBUG_LEVEL     SSL_DEBUG_OFF
#define SSL_CERT_ERROR_LEVEL     SSL_DEBUG_ON

#define SSL_PKEY_DEBUG_LEVEL     SSL_DEBUG_OFF
#define SSL_PKEY_ERROR_LEVEL     SSL_DEBUG_ON

#define SSL_X509_DEBUG_LEVEL     SSL_DEBUG_OFF
#define SSL_X509_ERROR_LEVEL     SSL_DEBUG_ON

#define SSL_LIB_DEBUG_LEVEL      SSL_DEBUG_OFF
#define SSL_LIB_ERROR_LEVEL      SSL_DEBUG_ON

#define SSL_STACK_DEBUG_LEVEL    SSL_DEBUG_OFF
#define SSL_STACK_ERROR_LEVEL    SSL_DEBUG_ON

#ifdef __cplusplus
 }
#endif

#endif
