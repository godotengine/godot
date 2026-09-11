/*
 *  Debugging routines
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "ssl_misc.h"

#if defined(MBEDTLS_DEBUG_C)

#include "mbedtls/platform.h"

#include "debug_internal.h"
#include "mbedtls/error.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* DEBUG_BUF_SIZE must be at least 2 */
#define DEBUG_BUF_SIZE      512

/* Temporary hack: on MingW, do not honor the platform.h configuration
 * for snprintf and vsnprintf. Instead, force the native functions,
 * which are the standard ones, not the Windows legacy ones.
 *
 * This hack should be removed once TF-PSA-Crypto has been updated to
 * use the standard printf family.
 */
#if defined(__MINGW32__)
#undef mbedtls_snprintf
#define mbedtls_snprintf snprintf
#undef mbedtls_vsnprintf
#define mbedtls_vsnprintf vsnprintf
#endif

int mbedtls_debug_snprintf(char *dest, size_t maxlen,
                           const char *format, ...)
{
    va_list argp;
    va_start(argp, format);
    int ret = mbedtls_vsnprintf(dest, maxlen, format, argp);
    va_end(argp);
    return ret;
}

static int debug_threshold = 0;

void mbedtls_debug_set_threshold(int threshold)
{
    debug_threshold = threshold;
}

/*
 * All calls to f_dbg must be made via this function
 */
static inline void debug_send_line(const mbedtls_ssl_context *ssl, int level,
                                   const char *file, int line,
                                   const char *str)
{
    /*
     * If in a threaded environment, we need a thread identifier.
     * Since there is no portable way to get one, use the address of the ssl
     * context instead, as it shouldn't be shared between threads.
     */
#if defined(MBEDTLS_THREADING_C)
    char idstr[20 + DEBUG_BUF_SIZE]; /* 0x + 16 nibbles + ': ' */
    mbedtls_snprintf(idstr, sizeof(idstr), "%p: %s", (void *) ssl, str);
    ssl->conf->f_dbg(ssl->conf->p_dbg, level, file, line, idstr);
#else
    ssl->conf->f_dbg(ssl->conf->p_dbg, level, file, line, str);
#endif
}

MBEDTLS_PRINTF_ATTRIBUTE(5, 6)
void mbedtls_debug_print_msg(const mbedtls_ssl_context *ssl, int level,
                             const char *file, int line,
                             const char *format, ...)
{
    va_list argp;
    char str[DEBUG_BUF_SIZE];
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    MBEDTLS_STATIC_ASSERT(DEBUG_BUF_SIZE >= 2, "DEBUG_BUF_SIZE too small");

    if (NULL == ssl              ||
        NULL == ssl->conf        ||
        NULL == ssl->conf->f_dbg ||
        level > debug_threshold) {
        return;
    }

    va_start(argp, format);
    ret = mbedtls_vsnprintf(str, DEBUG_BUF_SIZE, format, argp);
    va_end(argp);

    if (ret < 0) {
        ret = 0;
    } else {
        if (ret >= DEBUG_BUF_SIZE - 1) {
            ret = DEBUG_BUF_SIZE - 2;
        }
    }
    str[ret]     = '\n';
    str[ret + 1] = '\0';

    debug_send_line(ssl, level, file, line, str);
}

void mbedtls_debug_print_ret(const mbedtls_ssl_context *ssl, int level,
                             const char *file, int line,
                             const char *text, int ret)
{
    char str[DEBUG_BUF_SIZE];

    if (NULL == ssl              ||
        NULL == ssl->conf        ||
        NULL == ssl->conf->f_dbg ||
        level > debug_threshold) {
        return;
    }

    /*
     * With non-blocking I/O and examples that just retry immediately,
     * the logs would be quickly flooded with WANT_READ, so ignore that.
     * Don't ignore WANT_WRITE however, since it is usually rare.
     */
    if (ret == MBEDTLS_ERR_SSL_WANT_READ) {
        return;
    }

    mbedtls_snprintf(str, sizeof(str), "%s() returned %d (-0x%04x)\n",
                     text, ret, (unsigned int) -ret);

    debug_send_line(ssl, level, file, line, str);
}

#define MBEDTLS_DEBUG_PRINT_BUF_NO_TEXT       0
#define MBEDTLS_DEBUG_PRINT_BUF_ADD_TEXT      1

static void mbedtls_debug_print_buf_one_line(char *out_buf, size_t out_size,
                                             const unsigned char *in_buf, size_t in_size,
                                             int add_text)
{
    char txt[17] = { 0 };
    size_t i, idx = 0;

    for (i = 0; i < 16; i++) {
        if (i < in_size) {
            idx += mbedtls_snprintf(out_buf + idx, out_size - idx, " %02x",
                                    (unsigned int) in_buf[i]);
            txt[i] = (in_buf[i] > 31 && in_buf[i] < 127) ? in_buf[i] : '.';
        } else {
            /* Just add spaces until the end of the line */
            idx += mbedtls_snprintf(out_buf + idx, out_size - idx, "   ");
        }
    }

    if (add_text) {
        idx += mbedtls_snprintf(out_buf + idx, out_size - idx, "  %s", txt);
    }
    mbedtls_snprintf(out_buf + idx, out_size - idx, "\n");
}

static void mbedtls_debug_print_buf_ext(const mbedtls_ssl_context *ssl, int level,
                                        const char *file, int line, const char *text,
                                        const unsigned char *buf, size_t len,
                                        int add_text)
{
    char str[DEBUG_BUF_SIZE] = { 0 };
    size_t curr_offset = 0, idx = 0, chunk_len;

    if (NULL == ssl              ||
        NULL == ssl->conf        ||
        NULL == ssl->conf->f_dbg ||
        level > debug_threshold) {
        return;
    }

    mbedtls_snprintf(str, sizeof(str), "dumping '%s' (%" MBEDTLS_PRINTF_SIZET " bytes)\n",
                     text, len);
    debug_send_line(ssl, level, file, line, str);

    while (len > 0) {
        memset(str, 0, sizeof(str));
        idx = mbedtls_snprintf(str, sizeof(str), "%04" MBEDTLS_PRINTF_SIZET_HEX ": ", curr_offset);
        chunk_len = (len >= 16) ? 16 : len;
        mbedtls_debug_print_buf_one_line(str + idx, sizeof(str) - idx,
                                         &buf[curr_offset], chunk_len,
                                         add_text);
        debug_send_line(ssl, level, file, line, str);
        curr_offset += 16;
        len -= chunk_len;
    }
}

void mbedtls_debug_print_buf(const mbedtls_ssl_context *ssl, int level,
                             const char *file, int line, const char *text,
                             const unsigned char *buf, size_t len)
{
    mbedtls_debug_print_buf_ext(ssl, level, file, line, text, buf, len,
                                MBEDTLS_DEBUG_PRINT_BUF_ADD_TEXT);
}

#if defined(MBEDTLS_X509_CRT_PARSE_C) && !defined(MBEDTLS_X509_REMOVE_INFO)

#if defined(MBEDTLS_PK_WRITE_C)
static void debug_print_pk(const mbedtls_ssl_context *ssl, int level,
                           const char *file, int line,
                           const char *text, const mbedtls_pk_context *pk)
{
    unsigned char buf[PSA_EXPORT_PUBLIC_KEY_MAX_SIZE];
    size_t buf_len;
    int ret;

    ret = mbedtls_pk_write_pubkey_psa(pk, buf, sizeof(buf), &buf_len);
    if (ret == 0) {
        mbedtls_debug_print_buf_ext(ssl, level, file, line, text, buf, buf_len,
                                    MBEDTLS_DEBUG_PRINT_BUF_NO_TEXT);
    } else {
        mbedtls_debug_print_msg(ssl, level, file, line,
                                "failed to export public key from PK context");
    }
}
#endif /* MBEDTLS_PK_WRITE_C */

static void debug_print_line_by_line(const mbedtls_ssl_context *ssl, int level,
                                     const char *file, int line, const char *text)
{
    char str[DEBUG_BUF_SIZE];
    const char *start, *cur;

    start = text;
    for (cur = text; *cur != '\0'; cur++) {
        if (*cur == '\n') {
            size_t len = (size_t) (cur - start) + 1;
            if (len > DEBUG_BUF_SIZE - 1) {
                len = DEBUG_BUF_SIZE - 1;
            }

            memcpy(str, start, len);
            str[len] = '\0';

            debug_send_line(ssl, level, file, line, str);

            start = cur + 1;
        }
    }
}

void mbedtls_debug_print_crt(const mbedtls_ssl_context *ssl, int level,
                             const char *file, int line,
                             const char *text, const mbedtls_x509_crt *crt)
{
    char str[DEBUG_BUF_SIZE];
    int i = 0;

    if (NULL == ssl              ||
        NULL == ssl->conf        ||
        NULL == ssl->conf->f_dbg ||
        NULL == crt              ||
        level > debug_threshold) {
        return;
    }

    while (crt != NULL) {
        char buf[1024];

        mbedtls_snprintf(str, sizeof(str), "%s #%d:\n", text, ++i);
        debug_send_line(ssl, level, file, line, str);

        mbedtls_x509_crt_info(buf, sizeof(buf) - 1, "", crt);
        debug_print_line_by_line(ssl, level, file, line, buf);

#if defined(MBEDTLS_PK_WRITE_C)
        debug_print_pk(ssl, level, file, line, "crt->PK", &crt->pk);
#endif /* MBEDTLS_PK_WRITE_C */

        crt = crt->next;
    }
}
#endif /* MBEDTLS_X509_CRT_PARSE_C && MBEDTLS_X509_REMOVE_INFO */

#endif /* MBEDTLS_DEBUG_C */
