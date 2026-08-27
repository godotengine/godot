/*
 *  Portable interface to the CPU cycle counter
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "ssl_misc.h"

#if defined(MBEDTLS_TIMING_C)

#include "mbedtls/timing.h"

#if !defined(MBEDTLS_TIMING_ALT)

unsigned long long mbedtls_timing_get_timer(struct mbedtls_timing_hr_time *val, int reset)
{
    if (reset) {
        val->ms = mbedtls_ms_time();
        return 0;
    } else {
        mbedtls_ms_time_t now = mbedtls_ms_time();
        return now - val->ms;
    }
}

/*
 * Set delays to watch
 */
void mbedtls_timing_set_delay(void *data, uint32_t int_ms, uint32_t fin_ms)
{
    mbedtls_timing_delay_context *ctx = (mbedtls_timing_delay_context *) data;

    ctx->int_ms = int_ms;
    ctx->fin_ms = fin_ms;

    if (fin_ms != 0) {
        (void) mbedtls_timing_get_timer(&ctx->timer, 1);
    }
}

/*
 * Get number of delays expired
 */
int mbedtls_timing_get_delay(void *data)
{
    mbedtls_timing_delay_context *ctx = (mbedtls_timing_delay_context *) data;
    unsigned long long elapsed_ms;

    if (ctx->fin_ms == 0) {
        return -1;
    }

    elapsed_ms = mbedtls_timing_get_timer(&ctx->timer, 0);

    if (elapsed_ms >= ctx->fin_ms) {
        return 2;
    }

    if (elapsed_ms >= ctx->int_ms) {
        return 1;
    }

    return 0;
}

/*
 * Get the final delay.
 */
uint32_t mbedtls_timing_get_final_delay(
    const mbedtls_timing_delay_context *data)
{
    return data->fin_ms;
}
#endif /* !MBEDTLS_TIMING_ALT */
#endif /* MBEDTLS_TIMING_C */
