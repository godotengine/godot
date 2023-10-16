/*
 * ASN.1 buffer writing functionality
 *
 *  Copyright The Mbed TLS Contributors
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
 */

#include "common.h"

#if defined(MBEDTLS_ASN1_WRITE_C)

#include "mbedtls/asn1write.h"
#include "mbedtls/error.h"

#include <string.h>

#include "mbedtls/platform.h"

int mbedtls_asn1_write_len(unsigned char **p, unsigned char *start, size_t len)
{
    if (len < 0x80) {
        if (*p - start < 1) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }

        *--(*p) = (unsigned char) len;
        return 1;
    }

    if (len <= 0xFF) {
        if (*p - start < 2) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }

        *--(*p) = (unsigned char) len;
        *--(*p) = 0x81;
        return 2;
    }

    if (len <= 0xFFFF) {
        if (*p - start < 3) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }

        *--(*p) = MBEDTLS_BYTE_0(len);
        *--(*p) = MBEDTLS_BYTE_1(len);
        *--(*p) = 0x82;
        return 3;
    }

    if (len <= 0xFFFFFF) {
        if (*p - start < 4) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }

        *--(*p) = MBEDTLS_BYTE_0(len);
        *--(*p) = MBEDTLS_BYTE_1(len);
        *--(*p) = MBEDTLS_BYTE_2(len);
        *--(*p) = 0x83;
        return 4;
    }

    int len_is_valid = 1;
#if SIZE_MAX > 0xFFFFFFFF
    len_is_valid = (len <= 0xFFFFFFFF);
#endif
    if (len_is_valid) {
        if (*p - start < 5) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }

        *--(*p) = MBEDTLS_BYTE_0(len);
        *--(*p) = MBEDTLS_BYTE_1(len);
        *--(*p) = MBEDTLS_BYTE_2(len);
        *--(*p) = MBEDTLS_BYTE_3(len);
        *--(*p) = 0x84;
        return 5;
    }

    return MBEDTLS_ERR_ASN1_INVALID_LENGTH;
}

int mbedtls_asn1_write_tag(unsigned char **p, unsigned char *start, unsigned char tag)
{
    if (*p - start < 1) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    *--(*p) = tag;

    return 1;
}

int mbedtls_asn1_write_raw_buffer(unsigned char **p, unsigned char *start,
                                  const unsigned char *buf, size_t size)
{
    size_t len = 0;

    if (*p < start || (size_t) (*p - start) < size) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    len = size;
    (*p) -= len;
    memcpy(*p, buf, len);

    return (int) len;
}

#if defined(MBEDTLS_BIGNUM_C)
int mbedtls_asn1_write_mpi(unsigned char **p, unsigned char *start, const mbedtls_mpi *X)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    // Write the MPI
    //
    len = mbedtls_mpi_size(X);

    /* DER represents 0 with a sign bit (0=nonnegative) and 7 value bits, not
     * as 0 digits. We need to end up with 020100, not with 0200. */
    if (len == 0) {
        len = 1;
    }

    if (*p < start || (size_t) (*p - start) < len) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    (*p) -= len;
    MBEDTLS_MPI_CHK(mbedtls_mpi_write_binary(X, *p, len));

    // DER format assumes 2s complement for numbers, so the leftmost bit
    // should be 0 for positive numbers and 1 for negative numbers.
    //
    if (X->s == 1 && **p & 0x80) {
        if (*p - start < 1) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }

        *--(*p) = 0x00;
        len += 1;
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, MBEDTLS_ASN1_INTEGER));

    ret = (int) len;

cleanup:
    return ret;
}
#endif /* MBEDTLS_BIGNUM_C */

int mbedtls_asn1_write_null(unsigned char **p, unsigned char *start)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    // Write NULL
    //
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, 0));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, MBEDTLS_ASN1_NULL));

    return (int) len;
}

int mbedtls_asn1_write_oid(unsigned char **p, unsigned char *start,
                           const char *oid, size_t oid_len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_raw_buffer(p, start,
                                                            (const unsigned char *) oid, oid_len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, MBEDTLS_ASN1_OID));

    return (int) len;
}

int mbedtls_asn1_write_algorithm_identifier(unsigned char **p, unsigned char *start,
                                            const char *oid, size_t oid_len,
                                            size_t par_len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    if (par_len == 0) {
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_null(p, start));
    } else {
        len += par_len;
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_oid(p, start, oid, oid_len));

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start,
                                                     MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    return (int) len;
}

int mbedtls_asn1_write_bool(unsigned char **p, unsigned char *start, int boolean)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    if (*p - start < 1) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    *--(*p) = (boolean) ? 255 : 0;
    len++;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, MBEDTLS_ASN1_BOOLEAN));

    return (int) len;
}

static int asn1_write_tagged_int(unsigned char **p, unsigned char *start, int val, int tag)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    do {
        if (*p - start < 1) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }
        len += 1;
        *--(*p) = val & 0xff;
        val >>= 8;
    } while (val > 0);

    if (**p & 0x80) {
        if (*p - start < 1) {
            return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
        }
        *--(*p) = 0x00;
        len += 1;
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, tag));

    return (int) len;
}

int mbedtls_asn1_write_int(unsigned char **p, unsigned char *start, int val)
{
    return asn1_write_tagged_int(p, start, val, MBEDTLS_ASN1_INTEGER);
}

int mbedtls_asn1_write_enum(unsigned char **p, unsigned char *start, int val)
{
    return asn1_write_tagged_int(p, start, val, MBEDTLS_ASN1_ENUMERATED);
}

int mbedtls_asn1_write_tagged_string(unsigned char **p, unsigned char *start, int tag,
                                     const char *text, size_t text_len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_raw_buffer(p, start,
                                                            (const unsigned char *) text,
                                                            text_len));

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, tag));

    return (int) len;
}

int mbedtls_asn1_write_utf8_string(unsigned char **p, unsigned char *start,
                                   const char *text, size_t text_len)
{
    return mbedtls_asn1_write_tagged_string(p, start, MBEDTLS_ASN1_UTF8_STRING, text, text_len);
}

int mbedtls_asn1_write_printable_string(unsigned char **p, unsigned char *start,
                                        const char *text, size_t text_len)
{
    return mbedtls_asn1_write_tagged_string(p, start, MBEDTLS_ASN1_PRINTABLE_STRING, text,
                                            text_len);
}

int mbedtls_asn1_write_ia5_string(unsigned char **p, unsigned char *start,
                                  const char *text, size_t text_len)
{
    return mbedtls_asn1_write_tagged_string(p, start, MBEDTLS_ASN1_IA5_STRING, text, text_len);
}

int mbedtls_asn1_write_named_bitstring(unsigned char **p,
                                       unsigned char *start,
                                       const unsigned char *buf,
                                       size_t bits)
{
    size_t unused_bits, byte_len;
    const unsigned char *cur_byte;
    unsigned char cur_byte_shifted;
    unsigned char bit;

    byte_len = (bits + 7) / 8;
    unused_bits = (byte_len * 8) - bits;

    /*
     * Named bitstrings require that trailing 0s are excluded in the encoding
     * of the bitstring. Trailing 0s are considered part of the 'unused' bits
     * when encoding this value in the first content octet
     */
    if (bits != 0) {
        cur_byte = buf + byte_len - 1;
        cur_byte_shifted = *cur_byte >> unused_bits;

        for (;;) {
            bit = cur_byte_shifted & 0x1;
            cur_byte_shifted >>= 1;

            if (bit != 0) {
                break;
            }

            bits--;
            if (bits == 0) {
                break;
            }

            if (bits % 8 == 0) {
                cur_byte_shifted = *--cur_byte;
            }
        }
    }

    return mbedtls_asn1_write_bitstring(p, start, buf, bits);
}

int mbedtls_asn1_write_bitstring(unsigned char **p, unsigned char *start,
                                 const unsigned char *buf, size_t bits)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;
    size_t unused_bits, byte_len;

    byte_len = (bits + 7) / 8;
    unused_bits = (byte_len * 8) - bits;

    if (*p < start || (size_t) (*p - start) < byte_len + 1) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    len = byte_len + 1;

    /* Write the bitstring. Ensure the unused bits are zeroed */
    if (byte_len > 0) {
        byte_len--;
        *--(*p) = buf[byte_len] & ~((0x1 << unused_bits) - 1);
        (*p) -= byte_len;
        memcpy(*p, buf, byte_len);
    }

    /* Write unused bits */
    *--(*p) = (unsigned char) unused_bits;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, MBEDTLS_ASN1_BIT_STRING));

    return (int) len;
}

int mbedtls_asn1_write_octet_string(unsigned char **p, unsigned char *start,
                                    const unsigned char *buf, size_t size)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_raw_buffer(p, start, buf, size));

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, MBEDTLS_ASN1_OCTET_STRING));

    return (int) len;
}


/* This is a copy of the ASN.1 parsing function mbedtls_asn1_find_named_data(),
 * which is replicated to avoid a dependency ASN1_WRITE_C on ASN1_PARSE_C. */
static mbedtls_asn1_named_data *asn1_find_named_data(
    mbedtls_asn1_named_data *list,
    const char *oid, size_t len)
{
    while (list != NULL) {
        if (list->oid.len == len &&
            memcmp(list->oid.p, oid, len) == 0) {
            break;
        }

        list = list->next;
    }

    return list;
}

mbedtls_asn1_named_data *mbedtls_asn1_store_named_data(
    mbedtls_asn1_named_data **head,
    const char *oid, size_t oid_len,
    const unsigned char *val,
    size_t val_len)
{
    mbedtls_asn1_named_data *cur;

    if ((cur = asn1_find_named_data(*head, oid, oid_len)) == NULL) {
        // Add new entry if not present yet based on OID
        //
        cur = (mbedtls_asn1_named_data *) mbedtls_calloc(1,
                                                         sizeof(mbedtls_asn1_named_data));
        if (cur == NULL) {
            return NULL;
        }

        cur->oid.len = oid_len;
        cur->oid.p = mbedtls_calloc(1, oid_len);
        if (cur->oid.p == NULL) {
            mbedtls_free(cur);
            return NULL;
        }

        memcpy(cur->oid.p, oid, oid_len);

        cur->val.len = val_len;
        if (val_len != 0) {
            cur->val.p = mbedtls_calloc(1, val_len);
            if (cur->val.p == NULL) {
                mbedtls_free(cur->oid.p);
                mbedtls_free(cur);
                return NULL;
            }
        }

        cur->next = *head;
        *head = cur;
    } else if (val_len == 0) {
        mbedtls_free(cur->val.p);
        cur->val.p = NULL;
    } else if (cur->val.len != val_len) {
        /*
         * Enlarge existing value buffer if needed
         * Preserve old data until the allocation succeeded, to leave list in
         * a consistent state in case allocation fails.
         */
        void *p = mbedtls_calloc(1, val_len);
        if (p == NULL) {
            return NULL;
        }

        mbedtls_free(cur->val.p);
        cur->val.p = p;
        cur->val.len = val_len;
    }

    if (val != NULL && val_len != 0) {
        memcpy(cur->val.p, val, val_len);
    }

    return cur;
}
#endif /* MBEDTLS_ASN1_WRITE_C */
