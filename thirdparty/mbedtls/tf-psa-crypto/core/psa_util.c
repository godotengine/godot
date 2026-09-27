/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "tf_psa_crypto_common.h"

/* This is needed for MBEDTLS_ERR_XXX macros */
#include <mbedtls/private/error_common.h>

#if defined(MBEDTLS_ASN1_WRITE_C)
#include <mbedtls/asn1write.h>
#include <psa/crypto_sizes.h>
#endif

#include <mbedtls/psa_util.h>

#if defined(MBEDTLS_PSA_CRYPTO_CLIENT)

#include <psa/crypto.h>
#include <mbedtls/private/entropy.h>

/* Wrapper function allowing the classic API to use the PSA RNG.
 *
 * `mbedtls_psa_get_random(MBEDTLS_PSA_RANDOM_STATE, ...)` calls
 * `psa_generate_random(...)`. The state parameter is ignored since the
 * PSA API doesn't support passing an explicit state.
 */
int mbedtls_psa_get_random(void *p_rng,
                           unsigned char *output,
                           size_t output_size)
{
    /* This function takes a pointer to the RNG state because that's what
     * classic mbedtls functions using an RNG expect. The PSA RNG manages
     * its own state internally and doesn't let the caller access that state.
     * So we just ignore the state parameter, and in practice we'll pass
     * NULL. */
    (void) p_rng;
    psa_status_t status = psa_generate_random(output, output_size);
    if (status == PSA_SUCCESS) {
        return 0;
    } else {
        return MBEDTLS_ERR_ENTROPY_SOURCE_FAILED;
    }
}

#endif /* MBEDTLS_PSA_CRYPTO_CLIENT */

#if defined(PSA_HAVE_ALG_SOME_ECDSA)

/**
 * \brief  Convert a single raw coordinate to DER ASN.1 format. The output der
 *         buffer is filled backward (i.e. starting from its end).
 *
 * \param raw_buf           Buffer containing the raw coordinate to be
 *                          converted.
 * \param raw_len           Length of raw_buf in bytes. This must be > 0.
 * \param der_buf_start     Pointer to the beginning of the buffer which
 *                          will be filled with the DER converted data.
 * \param der_buf_end       End of the buffer used to store the DER output.
 *
 * \return                  On success, the amount of data (in bytes) written to
 *                          the DER buffer.
 * \return                  MBEDTLS_ERR_ASN1_BUF_TOO_SMALL if the provided der
 *                          buffer is too small to contain all the converted data.
 * \return                  MBEDTLS_ERR_ASN1_INVALID_DATA if the input raw
 *                          coordinate is null (i.e. all zeros).
 *
 * \warning                 Raw and der buffer must not be overlapping.
 */
static int convert_raw_to_der_single_int(const unsigned char *raw_buf, size_t raw_len,
                                         unsigned char *der_buf_start,
                                         unsigned char *der_buf_end)
{
    unsigned char *p = der_buf_end;
    int len;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    /* ASN.1 DER encoding requires minimal length, so skip leading 0s.
     * Provided input MPIs should not be 0, but as a failsafe measure, still
     * detect that and return error in case. */
    while (*raw_buf == 0x00) {
        ++raw_buf;
        --raw_len;
        if (raw_len == 0) {
            return MBEDTLS_ERR_ASN1_INVALID_DATA;
        }
    }
    len = (int) raw_len;

    /* Copy the raw coordinate to the end of der_buf. */
    if ((p - der_buf_start) < len) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }
    p -= len;
    memcpy(p, raw_buf, len);

    /* If MSb is 1, ASN.1 requires that we prepend a 0. */
    if (*p & 0x80) {
        if ((p - der_buf_start) < 1) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }
        --p;
        *p = 0x00;
        ++len;
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&p, der_buf_start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(&p, der_buf_start, MBEDTLS_ASN1_INTEGER));

    return len;
}

int mbedtls_ecdsa_raw_to_der(size_t bits, const unsigned char *raw, size_t raw_len,
                             unsigned char *der, size_t der_size, size_t *der_len)
{
    unsigned char r[PSA_BITS_TO_BYTES(PSA_VENDOR_ECC_MAX_CURVE_BITS)];
    unsigned char s[PSA_BITS_TO_BYTES(PSA_VENDOR_ECC_MAX_CURVE_BITS)];
    const size_t coordinate_len = PSA_BITS_TO_BYTES(bits);
    size_t len = 0;
    unsigned char *p = der + der_size;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if (bits == 0) {
        return MBEDTLS_ERR_ASN1_INVALID_DATA;
    }
    if (raw_len != (2 * coordinate_len)) {
        return MBEDTLS_ERR_ASN1_INVALID_DATA;
    }
    if (coordinate_len > sizeof(r)) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    /* Since raw and der buffers might overlap, dump r and s before starting
     * the conversion. */
    memcpy(r, raw, coordinate_len);
    memcpy(s, raw + coordinate_len, coordinate_len);

    /* der buffer will initially be written starting from its end so we pick s
     * first and then r. */
    ret = convert_raw_to_der_single_int(s, coordinate_len, der, p);
    if (ret < 0) {
        return ret;
    }
    p -= ret;
    len += ret;

    ret = convert_raw_to_der_single_int(r, coordinate_len, der, p);
    if (ret < 0) {
        return ret;
    }
    p -= ret;
    len += ret;

    /* Add ASN.1 header (len + tag). */
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&p, der, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(&p, der,
                                                     MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    /* memmove the content of der buffer to its beginnig. */
    memmove(der, p, len);
    *der_len = len;

    return 0;
}

/**
 * \brief Convert a single integer from ASN.1 DER format to raw.
 *
 * \param der               Buffer containing the DER integer value to be
 *                          converted.
 * \param der_len           Length of the der buffer in bytes.
 * \param raw               Output buffer that will be filled with the
 *                          converted data. This should be at least
 *                          coordinate_size bytes and it must be zeroed before
 *                          calling this function.
 * \param coordinate_size   Size (in bytes) of a single coordinate in raw
 *                          format.
 *
 * \return                  On success, the amount of DER data parsed from the
 *                          provided der buffer.
 * \return                  MBEDTLS_ERR_ASN1_UNEXPECTED_TAG if the integer tag
 *                          is missing in the der buffer.
 * \return                  MBEDTLS_ERR_ASN1_LENGTH_MISMATCH if the integer
 *                          is null (i.e. all zeros) or if the output raw buffer
 *                          is too small to contain the converted raw value.
 *
 * \warning                 Der and raw buffers must not be overlapping.
 */
static int convert_der_to_raw_single_int(unsigned char *der, size_t der_len,
                                         unsigned char *raw, size_t coordinate_size)
{
    unsigned char *p = der;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t unpadded_len, padding_len = 0;

    /* Get the length of ASN.1 element (i.e. the integer we need to parse). */
    ret = mbedtls_asn1_get_tag(&p, p + der_len, &unpadded_len,
                               MBEDTLS_ASN1_INTEGER);
    if (ret != 0) {
        return ret;
    }

    /* It's invalid to have:
     * - unpadded_len == 0.
     * - MSb set without a leading 0x00 (leading 0x00 is checked below). */
    if (((unpadded_len == 0) || (*p & 0x80) != 0)) {
        return MBEDTLS_ERR_ASN1_INVALID_DATA;
    }

    /* Skip possible leading zero */
    if (*p == 0x00) {
        p++;
        unpadded_len--;
        /* It is not allowed to have more than 1 leading zero.
         * Ignore the case in which unpadded_len = 0 because that's a 0 encoded
         * in ASN.1 format (i.e. 020100). */
        if ((unpadded_len > 0) && (*p == 0x00)) {
            return MBEDTLS_ERR_ASN1_INVALID_DATA;
        }
    }

    if (unpadded_len > coordinate_size) {
        /* Parsed number is longer than the maximum expected value. */
        return MBEDTLS_ERR_ASN1_INVALID_DATA;
    }
    padding_len = coordinate_size - unpadded_len;
    /* raw buffer was already zeroed by the calling function so zero-padding
     * operation is skipped here. */
    memcpy(raw + padding_len, p, unpadded_len);
    p += unpadded_len;

    return (int) (p - der);
}

int mbedtls_ecdsa_der_to_raw(size_t bits, const unsigned char *der, size_t der_len,
                             unsigned char *raw, size_t raw_size, size_t *raw_len)
{
    unsigned char raw_tmp[PSA_VENDOR_ECDSA_SIGNATURE_MAX_SIZE];
    unsigned char *p = (unsigned char *) der;
    size_t data_len;
    size_t coordinate_size = PSA_BITS_TO_BYTES(bits);
    int ret;

    if (bits == 0) {
        return MBEDTLS_ERR_ASN1_INVALID_DATA;
    }
    /* The output raw buffer should be at least twice the size of a raw
     * coordinate in order to store r and s. */
    if (raw_size < coordinate_size * 2) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }
    if (2 * coordinate_size > sizeof(raw_tmp)) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    /* Check that the provided input DER buffer has the right header. */
    ret = mbedtls_asn1_get_tag(&p, der + der_len, &data_len,
                               MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE);
    if (ret != 0) {
        return ret;
    }

    memset(raw_tmp, 0, 2 * coordinate_size);

    /* Extract r */
    ret = convert_der_to_raw_single_int(p, data_len, raw_tmp, coordinate_size);
    if (ret < 0) {
        return ret;
    }
    p += ret;
    data_len -= ret;

    /* Extract s */
    ret = convert_der_to_raw_single_int(p, data_len, raw_tmp + coordinate_size,
                                        coordinate_size);
    if (ret < 0) {
        return ret;
    }
    p += ret;
    data_len -= ret;

    /* Check that we consumed all the input der data. */
    if ((size_t) (p - der) != der_len) {
        return MBEDTLS_ERR_ASN1_LENGTH_MISMATCH;
    }

    memcpy(raw, raw_tmp, 2 * coordinate_size);
    *raw_len = 2 * coordinate_size;

    return 0;
}

#endif /* PSA_HAVE_ALG_SOME_ECDSA */
