/*
 *  Error message information
 *
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
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
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_ERROR_C) || defined(MBEDTLS_ERROR_STRERROR_DUMMY)
#include "mbedtls/error.h"
#include <string.h>
#endif

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#define mbedtls_snprintf snprintf
#define mbedtls_time_t   time_t
#endif

#if defined(MBEDTLS_ERROR_C)

#include <stdio.h>

#if defined(MBEDTLS_AES_C)
#include "mbedtls/aes.h"
#endif

#if defined(MBEDTLS_ARC4_C)
#include "mbedtls/arc4.h"
#endif

#if defined(MBEDTLS_ARIA_C)
#include "mbedtls/aria.h"
#endif

#if defined(MBEDTLS_BASE64_C)
#include "mbedtls/base64.h"
#endif

#if defined(MBEDTLS_BIGNUM_C)
#include "mbedtls/bignum.h"
#endif

#if defined(MBEDTLS_BLOWFISH_C)
#include "mbedtls/blowfish.h"
#endif

#if defined(MBEDTLS_CAMELLIA_C)
#include "mbedtls/camellia.h"
#endif

#if defined(MBEDTLS_CCM_C)
#include "mbedtls/ccm.h"
#endif

#if defined(MBEDTLS_CHACHA20_C)
#include "mbedtls/chacha20.h"
#endif

#if defined(MBEDTLS_CHACHAPOLY_C)
#include "mbedtls/chachapoly.h"
#endif

#if defined(MBEDTLS_CIPHER_C)
#include "mbedtls/cipher.h"
#endif

#if defined(MBEDTLS_CMAC_C)
#include "mbedtls/cmac.h"
#endif

#if defined(MBEDTLS_CTR_DRBG_C)
#include "mbedtls/ctr_drbg.h"
#endif

#if defined(MBEDTLS_DES_C)
#include "mbedtls/des.h"
#endif

#if defined(MBEDTLS_DHM_C)
#include "mbedtls/dhm.h"
#endif

#if defined(MBEDTLS_ECP_C)
#include "mbedtls/ecp.h"
#endif

#if defined(MBEDTLS_ENTROPY_C)
#include "mbedtls/entropy.h"
#endif

#if defined(MBEDTLS_GCM_C)
#include "mbedtls/gcm.h"
#endif

#if defined(MBEDTLS_HKDF_C)
#include "mbedtls/hkdf.h"
#endif

#if defined(MBEDTLS_HMAC_DRBG_C)
#include "mbedtls/hmac_drbg.h"
#endif

#if defined(MBEDTLS_MD_C)
#include "mbedtls/md.h"
#endif

#if defined(MBEDTLS_MD2_C)
#include "mbedtls/md2.h"
#endif

#if defined(MBEDTLS_MD4_C)
#include "mbedtls/md4.h"
#endif

#if defined(MBEDTLS_MD5_C)
#include "mbedtls/md5.h"
#endif

#if defined(MBEDTLS_NET_C)
#include "mbedtls/net_sockets.h"
#endif

#if defined(MBEDTLS_OID_C)
#include "mbedtls/oid.h"
#endif

#if defined(MBEDTLS_PADLOCK_C)
#include "mbedtls/padlock.h"
#endif

#if defined(MBEDTLS_PEM_PARSE_C) || defined(MBEDTLS_PEM_WRITE_C)
#include "mbedtls/pem.h"
#endif

#if defined(MBEDTLS_PK_C)
#include "mbedtls/pk.h"
#endif

#if defined(MBEDTLS_PKCS12_C)
#include "mbedtls/pkcs12.h"
#endif

#if defined(MBEDTLS_PKCS5_C)
#include "mbedtls/pkcs5.h"
#endif

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#endif

#if defined(MBEDTLS_POLY1305_C)
#include "mbedtls/poly1305.h"
#endif

#if defined(MBEDTLS_RIPEMD160_C)
#include "mbedtls/ripemd160.h"
#endif

#if defined(MBEDTLS_RSA_C)
#include "mbedtls/rsa.h"
#endif

#if defined(MBEDTLS_SHA1_C)
#include "mbedtls/sha1.h"
#endif

#if defined(MBEDTLS_SHA256_C)
#include "mbedtls/sha256.h"
#endif

#if defined(MBEDTLS_SHA512_C)
#include "mbedtls/sha512.h"
#endif

#if defined(MBEDTLS_SSL_TLS_C)
#include "mbedtls/ssl.h"
#endif

#if defined(MBEDTLS_THREADING_C)
#include "mbedtls/threading.h"
#endif

#if defined(MBEDTLS_X509_USE_C) || defined(MBEDTLS_X509_CREATE_C)
#include "mbedtls/x509.h"
#endif

#if defined(MBEDTLS_XTEA_C)
#include "mbedtls/xtea.h"
#endif


void mbedtls_strerror( int ret, char *buf, size_t buflen )
{
    size_t len;
    int use_ret;

    if( buflen == 0 )
        return;

    memset( buf, 0x00, buflen );

    if( ret < 0 )
        ret = -ret;

    if( ret & 0xFF80 )
    {
        use_ret = ret & 0xFF80;

        // High level error codes
        //
        // BEGIN generated code
#if defined(MBEDTLS_CIPHER_C)
        if( use_ret == -(MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "CIPHER - The selected feature is not available" );
        if( use_ret == -(MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "CIPHER - Bad input parameters" );
        if( use_ret == -(MBEDTLS_ERR_CIPHER_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "CIPHER - Failed to allocate memory" );
        if( use_ret == -(MBEDTLS_ERR_CIPHER_INVALID_PADDING) )
            mbedtls_snprintf( buf, buflen, "CIPHER - Input data contains invalid padding and is rejected" );
        if( use_ret == -(MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED) )
            mbedtls_snprintf( buf, buflen, "CIPHER - Decryption of block requires a full block" );
        if( use_ret == -(MBEDTLS_ERR_CIPHER_AUTH_FAILED) )
            mbedtls_snprintf( buf, buflen, "CIPHER - Authentication failed (for AEAD modes)" );
        if( use_ret == -(MBEDTLS_ERR_CIPHER_INVALID_CONTEXT) )
            mbedtls_snprintf( buf, buflen, "CIPHER - The context is invalid. For example, because it was freed" );
        if( use_ret == -(MBEDTLS_ERR_CIPHER_HW_ACCEL_FAILED) )
            mbedtls_snprintf( buf, buflen, "CIPHER - Cipher hardware accelerator failed" );
#endif /* MBEDTLS_CIPHER_C */

#if defined(MBEDTLS_DHM_C)
        if( use_ret == -(MBEDTLS_ERR_DHM_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "DHM - Bad input parameters" );
        if( use_ret == -(MBEDTLS_ERR_DHM_READ_PARAMS_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - Reading of the DHM parameters failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_MAKE_PARAMS_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - Making of the DHM parameters failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_READ_PUBLIC_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - Reading of the public values failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_MAKE_PUBLIC_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - Making of the public value failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_CALC_SECRET_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - Calculation of the DHM secret failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_INVALID_FORMAT) )
            mbedtls_snprintf( buf, buflen, "DHM - The ASN.1 data is not formatted correctly" );
        if( use_ret == -(MBEDTLS_ERR_DHM_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - Allocation of memory failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_FILE_IO_ERROR) )
            mbedtls_snprintf( buf, buflen, "DHM - Read or write of file failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_HW_ACCEL_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - DHM hardware accelerator failed" );
        if( use_ret == -(MBEDTLS_ERR_DHM_SET_GROUP_FAILED) )
            mbedtls_snprintf( buf, buflen, "DHM - Setting the modulus and generator failed" );
#endif /* MBEDTLS_DHM_C */

#if defined(MBEDTLS_ECP_C)
        if( use_ret == -(MBEDTLS_ERR_ECP_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "ECP - Bad input parameters to function" );
        if( use_ret == -(MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL) )
            mbedtls_snprintf( buf, buflen, "ECP - The buffer is too small to write to" );
        if( use_ret == -(MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "ECP - The requested feature is not available, for example, the requested curve is not supported" );
        if( use_ret == -(MBEDTLS_ERR_ECP_VERIFY_FAILED) )
            mbedtls_snprintf( buf, buflen, "ECP - The signature is not valid" );
        if( use_ret == -(MBEDTLS_ERR_ECP_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "ECP - Memory allocation failed" );
        if( use_ret == -(MBEDTLS_ERR_ECP_RANDOM_FAILED) )
            mbedtls_snprintf( buf, buflen, "ECP - Generation of random value, such as ephemeral key, failed" );
        if( use_ret == -(MBEDTLS_ERR_ECP_INVALID_KEY) )
            mbedtls_snprintf( buf, buflen, "ECP - Invalid private or public key" );
        if( use_ret == -(MBEDTLS_ERR_ECP_SIG_LEN_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "ECP - The buffer contains a valid signature followed by more data" );
        if( use_ret == -(MBEDTLS_ERR_ECP_HW_ACCEL_FAILED) )
            mbedtls_snprintf( buf, buflen, "ECP - The ECP hardware accelerator failed" );
        if( use_ret == -(MBEDTLS_ERR_ECP_IN_PROGRESS) )
            mbedtls_snprintf( buf, buflen, "ECP - Operation in progress, call again with the same parameters to continue" );
#endif /* MBEDTLS_ECP_C */

#if defined(MBEDTLS_MD_C)
        if( use_ret == -(MBEDTLS_ERR_MD_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "MD - The selected feature is not available" );
        if( use_ret == -(MBEDTLS_ERR_MD_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "MD - Bad input parameters to function" );
        if( use_ret == -(MBEDTLS_ERR_MD_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "MD - Failed to allocate memory" );
        if( use_ret == -(MBEDTLS_ERR_MD_FILE_IO_ERROR) )
            mbedtls_snprintf( buf, buflen, "MD - Opening or reading of file failed" );
        if( use_ret == -(MBEDTLS_ERR_MD_HW_ACCEL_FAILED) )
            mbedtls_snprintf( buf, buflen, "MD - MD hardware accelerator failed" );
#endif /* MBEDTLS_MD_C */

#if defined(MBEDTLS_PEM_PARSE_C) || defined(MBEDTLS_PEM_WRITE_C)
        if( use_ret == -(MBEDTLS_ERR_PEM_NO_HEADER_FOOTER_PRESENT) )
            mbedtls_snprintf( buf, buflen, "PEM - No PEM header or footer found" );
        if( use_ret == -(MBEDTLS_ERR_PEM_INVALID_DATA) )
            mbedtls_snprintf( buf, buflen, "PEM - PEM string is not as expected" );
        if( use_ret == -(MBEDTLS_ERR_PEM_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "PEM - Failed to allocate memory" );
        if( use_ret == -(MBEDTLS_ERR_PEM_INVALID_ENC_IV) )
            mbedtls_snprintf( buf, buflen, "PEM - RSA IV is not in hex-format" );
        if( use_ret == -(MBEDTLS_ERR_PEM_UNKNOWN_ENC_ALG) )
            mbedtls_snprintf( buf, buflen, "PEM - Unsupported key encryption algorithm" );
        if( use_ret == -(MBEDTLS_ERR_PEM_PASSWORD_REQUIRED) )
            mbedtls_snprintf( buf, buflen, "PEM - Private key password can't be empty" );
        if( use_ret == -(MBEDTLS_ERR_PEM_PASSWORD_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "PEM - Given private key password does not allow for correct decryption" );
        if( use_ret == -(MBEDTLS_ERR_PEM_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "PEM - Unavailable feature, e.g. hashing/encryption combination" );
        if( use_ret == -(MBEDTLS_ERR_PEM_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "PEM - Bad input parameters to function" );
#endif /* MBEDTLS_PEM_PARSE_C || MBEDTLS_PEM_WRITE_C */

#if defined(MBEDTLS_PK_C)
        if( use_ret == -(MBEDTLS_ERR_PK_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "PK - Memory allocation failed" );
        if( use_ret == -(MBEDTLS_ERR_PK_TYPE_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "PK - Type mismatch, eg attempt to encrypt with an ECDSA key" );
        if( use_ret == -(MBEDTLS_ERR_PK_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "PK - Bad input parameters to function" );
        if( use_ret == -(MBEDTLS_ERR_PK_FILE_IO_ERROR) )
            mbedtls_snprintf( buf, buflen, "PK - Read/write of file failed" );
        if( use_ret == -(MBEDTLS_ERR_PK_KEY_INVALID_VERSION) )
            mbedtls_snprintf( buf, buflen, "PK - Unsupported key version" );
        if( use_ret == -(MBEDTLS_ERR_PK_KEY_INVALID_FORMAT) )
            mbedtls_snprintf( buf, buflen, "PK - Invalid key tag or value" );
        if( use_ret == -(MBEDTLS_ERR_PK_UNKNOWN_PK_ALG) )
            mbedtls_snprintf( buf, buflen, "PK - Key algorithm is unsupported (only RSA and EC are supported)" );
        if( use_ret == -(MBEDTLS_ERR_PK_PASSWORD_REQUIRED) )
            mbedtls_snprintf( buf, buflen, "PK - Private key password can't be empty" );
        if( use_ret == -(MBEDTLS_ERR_PK_PASSWORD_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "PK - Given private key password does not allow for correct decryption" );
        if( use_ret == -(MBEDTLS_ERR_PK_INVALID_PUBKEY) )
            mbedtls_snprintf( buf, buflen, "PK - The pubkey tag or value is invalid (only RSA and EC are supported)" );
        if( use_ret == -(MBEDTLS_ERR_PK_INVALID_ALG) )
            mbedtls_snprintf( buf, buflen, "PK - The algorithm tag or value is invalid" );
        if( use_ret == -(MBEDTLS_ERR_PK_UNKNOWN_NAMED_CURVE) )
            mbedtls_snprintf( buf, buflen, "PK - Elliptic curve is unsupported (only NIST curves are supported)" );
        if( use_ret == -(MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "PK - Unavailable feature, e.g. RSA disabled for RSA key" );
        if( use_ret == -(MBEDTLS_ERR_PK_SIG_LEN_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "PK - The buffer contains a valid signature followed by more data" );
        if( use_ret == -(MBEDTLS_ERR_PK_HW_ACCEL_FAILED) )
            mbedtls_snprintf( buf, buflen, "PK - PK hardware accelerator failed" );
#endif /* MBEDTLS_PK_C */

#if defined(MBEDTLS_PKCS12_C)
        if( use_ret == -(MBEDTLS_ERR_PKCS12_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "PKCS12 - Bad input parameters to function" );
        if( use_ret == -(MBEDTLS_ERR_PKCS12_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "PKCS12 - Feature not available, e.g. unsupported encryption scheme" );
        if( use_ret == -(MBEDTLS_ERR_PKCS12_PBE_INVALID_FORMAT) )
            mbedtls_snprintf( buf, buflen, "PKCS12 - PBE ASN.1 data not as expected" );
        if( use_ret == -(MBEDTLS_ERR_PKCS12_PASSWORD_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "PKCS12 - Given private key password does not allow for correct decryption" );
#endif /* MBEDTLS_PKCS12_C */

#if defined(MBEDTLS_PKCS5_C)
        if( use_ret == -(MBEDTLS_ERR_PKCS5_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "PKCS5 - Bad input parameters to function" );
        if( use_ret == -(MBEDTLS_ERR_PKCS5_INVALID_FORMAT) )
            mbedtls_snprintf( buf, buflen, "PKCS5 - Unexpected ASN.1 data" );
        if( use_ret == -(MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "PKCS5 - Requested encryption or digest alg not available" );
        if( use_ret == -(MBEDTLS_ERR_PKCS5_PASSWORD_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "PKCS5 - Given private key password does not allow for correct decryption" );
#endif /* MBEDTLS_PKCS5_C */

#if defined(MBEDTLS_RSA_C)
        if( use_ret == -(MBEDTLS_ERR_RSA_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "RSA - Bad input parameters to function" );
        if( use_ret == -(MBEDTLS_ERR_RSA_INVALID_PADDING) )
            mbedtls_snprintf( buf, buflen, "RSA - Input data contains invalid padding and is rejected" );
        if( use_ret == -(MBEDTLS_ERR_RSA_KEY_GEN_FAILED) )
            mbedtls_snprintf( buf, buflen, "RSA - Something failed during generation of a key" );
        if( use_ret == -(MBEDTLS_ERR_RSA_KEY_CHECK_FAILED) )
            mbedtls_snprintf( buf, buflen, "RSA - Key failed to pass the validity check of the library" );
        if( use_ret == -(MBEDTLS_ERR_RSA_PUBLIC_FAILED) )
            mbedtls_snprintf( buf, buflen, "RSA - The public key operation failed" );
        if( use_ret == -(MBEDTLS_ERR_RSA_PRIVATE_FAILED) )
            mbedtls_snprintf( buf, buflen, "RSA - The private key operation failed" );
        if( use_ret == -(MBEDTLS_ERR_RSA_VERIFY_FAILED) )
            mbedtls_snprintf( buf, buflen, "RSA - The PKCS#1 verification failed" );
        if( use_ret == -(MBEDTLS_ERR_RSA_OUTPUT_TOO_LARGE) )
            mbedtls_snprintf( buf, buflen, "RSA - The output buffer for decryption is not large enough" );
        if( use_ret == -(MBEDTLS_ERR_RSA_RNG_FAILED) )
            mbedtls_snprintf( buf, buflen, "RSA - The random generator failed to generate non-zeros" );
        if( use_ret == -(MBEDTLS_ERR_RSA_UNSUPPORTED_OPERATION) )
            mbedtls_snprintf( buf, buflen, "RSA - The implementation does not offer the requested operation, for example, because of security violations or lack of functionality" );
        if( use_ret == -(MBEDTLS_ERR_RSA_HW_ACCEL_FAILED) )
            mbedtls_snprintf( buf, buflen, "RSA - RSA hardware accelerator failed" );
#endif /* MBEDTLS_RSA_C */

#if defined(MBEDTLS_SSL_TLS_C)
        if( use_ret == -(MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "SSL - The requested feature is not available" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "SSL - Bad input parameters to function" );
        if( use_ret == -(MBEDTLS_ERR_SSL_INVALID_MAC) )
            mbedtls_snprintf( buf, buflen, "SSL - Verification of the message MAC failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_INVALID_RECORD) )
            mbedtls_snprintf( buf, buflen, "SSL - An invalid SSL record was received" );
        if( use_ret == -(MBEDTLS_ERR_SSL_CONN_EOF) )
            mbedtls_snprintf( buf, buflen, "SSL - The connection indicated an EOF" );
        if( use_ret == -(MBEDTLS_ERR_SSL_UNKNOWN_CIPHER) )
            mbedtls_snprintf( buf, buflen, "SSL - An unknown cipher was received" );
        if( use_ret == -(MBEDTLS_ERR_SSL_NO_CIPHER_CHOSEN) )
            mbedtls_snprintf( buf, buflen, "SSL - The server has no ciphersuites in common with the client" );
        if( use_ret == -(MBEDTLS_ERR_SSL_NO_RNG) )
            mbedtls_snprintf( buf, buflen, "SSL - No RNG was provided to the SSL module" );
        if( use_ret == -(MBEDTLS_ERR_SSL_NO_CLIENT_CERTIFICATE) )
            mbedtls_snprintf( buf, buflen, "SSL - No client certification received from the client, but required by the authentication mode" );
        if( use_ret == -(MBEDTLS_ERR_SSL_CERTIFICATE_TOO_LARGE) )
            mbedtls_snprintf( buf, buflen, "SSL - Our own certificate(s) is/are too large to send in an SSL message" );
        if( use_ret == -(MBEDTLS_ERR_SSL_CERTIFICATE_REQUIRED) )
            mbedtls_snprintf( buf, buflen, "SSL - The own certificate is not set, but needed by the server" );
        if( use_ret == -(MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED) )
            mbedtls_snprintf( buf, buflen, "SSL - The own private key or pre-shared key is not set, but needed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_CA_CHAIN_REQUIRED) )
            mbedtls_snprintf( buf, buflen, "SSL - No CA Chain is set, but required to operate" );
        if( use_ret == -(MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE) )
            mbedtls_snprintf( buf, buflen, "SSL - An unexpected message was received from our peer" );
        if( use_ret == -(MBEDTLS_ERR_SSL_FATAL_ALERT_MESSAGE) )
        {
            mbedtls_snprintf( buf, buflen, "SSL - A fatal alert message was received from our peer" );
            return;
        }
        if( use_ret == -(MBEDTLS_ERR_SSL_PEER_VERIFY_FAILED) )
            mbedtls_snprintf( buf, buflen, "SSL - Verification of our peer failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) )
            mbedtls_snprintf( buf, buflen, "SSL - The peer notified us that the connection is going to be closed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ClientHello handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ServerHello handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the Certificate handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_REQUEST) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the CertificateRequest handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ServerKeyExchange handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO_DONE) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ServerHelloDone handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ClientKeyExchange handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_RP) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ClientKeyExchange handshake message failed in DHM / ECDH Read Public" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_CS) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ClientKeyExchange handshake message failed in DHM / ECDH Calculate Secret" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the CertificateVerify handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_CHANGE_CIPHER_SPEC) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the ChangeCipherSpec handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_FINISHED) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the Finished handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "SSL - Memory allocation failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_HW_ACCEL_FAILED) )
            mbedtls_snprintf( buf, buflen, "SSL - Hardware acceleration function returned with error" );
        if( use_ret == -(MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH) )
            mbedtls_snprintf( buf, buflen, "SSL - Hardware acceleration function skipped / left alone data" );
        if( use_ret == -(MBEDTLS_ERR_SSL_COMPRESSION_FAILED) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the compression / decompression failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION) )
            mbedtls_snprintf( buf, buflen, "SSL - Handshake protocol not within min/max boundaries" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BAD_HS_NEW_SESSION_TICKET) )
            mbedtls_snprintf( buf, buflen, "SSL - Processing of the NewSessionTicket handshake message failed" );
        if( use_ret == -(MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED) )
            mbedtls_snprintf( buf, buflen, "SSL - Session ticket has expired" );
        if( use_ret == -(MBEDTLS_ERR_SSL_PK_TYPE_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "SSL - Public key type mismatch (eg, asked for RSA key exchange and presented EC key)" );
        if( use_ret == -(MBEDTLS_ERR_SSL_UNKNOWN_IDENTITY) )
            mbedtls_snprintf( buf, buflen, "SSL - Unknown identity received (eg, PSK identity)" );
        if( use_ret == -(MBEDTLS_ERR_SSL_INTERNAL_ERROR) )
            mbedtls_snprintf( buf, buflen, "SSL - Internal error (eg, unexpected failure in lower-level module)" );
        if( use_ret == -(MBEDTLS_ERR_SSL_COUNTER_WRAPPING) )
            mbedtls_snprintf( buf, buflen, "SSL - A counter would wrap (eg, too many messages exchanged)" );
        if( use_ret == -(MBEDTLS_ERR_SSL_WAITING_SERVER_HELLO_RENEGO) )
            mbedtls_snprintf( buf, buflen, "SSL - Unexpected message at ServerHello in renegotiation" );
        if( use_ret == -(MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED) )
            mbedtls_snprintf( buf, buflen, "SSL - DTLS client must retry for hello verification" );
        if( use_ret == -(MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL) )
            mbedtls_snprintf( buf, buflen, "SSL - A buffer is too small to receive or write a message" );
        if( use_ret == -(MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE) )
            mbedtls_snprintf( buf, buflen, "SSL - None of the common ciphersuites is usable (eg, no suitable certificate, see debug messages)" );
        if( use_ret == -(MBEDTLS_ERR_SSL_WANT_READ) )
            mbedtls_snprintf( buf, buflen, "SSL - No data of requested type currently available on underlying transport" );
        if( use_ret == -(MBEDTLS_ERR_SSL_WANT_WRITE) )
            mbedtls_snprintf( buf, buflen, "SSL - Connection requires a write call" );
        if( use_ret == -(MBEDTLS_ERR_SSL_TIMEOUT) )
            mbedtls_snprintf( buf, buflen, "SSL - The operation timed out" );
        if( use_ret == -(MBEDTLS_ERR_SSL_CLIENT_RECONNECT) )
            mbedtls_snprintf( buf, buflen, "SSL - The client initiated a reconnect from the same port" );
        if( use_ret == -(MBEDTLS_ERR_SSL_UNEXPECTED_RECORD) )
            mbedtls_snprintf( buf, buflen, "SSL - Record header looks valid but is not expected" );
        if( use_ret == -(MBEDTLS_ERR_SSL_NON_FATAL) )
            mbedtls_snprintf( buf, buflen, "SSL - The alert message received indicates a non-fatal error" );
        if( use_ret == -(MBEDTLS_ERR_SSL_INVALID_VERIFY_HASH) )
            mbedtls_snprintf( buf, buflen, "SSL - Couldn't set the hash for verifying CertificateVerify" );
        if( use_ret == -(MBEDTLS_ERR_SSL_CONTINUE_PROCESSING) )
            mbedtls_snprintf( buf, buflen, "SSL - Internal-only message signaling that further message-processing should be done" );
        if( use_ret == -(MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS) )
            mbedtls_snprintf( buf, buflen, "SSL - The asynchronous operation is not completed yet" );
        if( use_ret == -(MBEDTLS_ERR_SSL_EARLY_MESSAGE) )
            mbedtls_snprintf( buf, buflen, "SSL - Internal-only message signaling that a message arrived early" );
        if( use_ret == -(MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS) )
            mbedtls_snprintf( buf, buflen, "SSL - A cryptographic operation is in progress. Try again later" );
#endif /* MBEDTLS_SSL_TLS_C */

#if defined(MBEDTLS_X509_USE_C) || defined(MBEDTLS_X509_CREATE_C)
        if( use_ret == -(MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE) )
            mbedtls_snprintf( buf, buflen, "X509 - Unavailable feature, e.g. RSA hashing/encryption combination" );
        if( use_ret == -(MBEDTLS_ERR_X509_UNKNOWN_OID) )
            mbedtls_snprintf( buf, buflen, "X509 - Requested OID is unknown" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_FORMAT) )
            mbedtls_snprintf( buf, buflen, "X509 - The CRT/CRL/CSR format is invalid, e.g. different type expected" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_VERSION) )
            mbedtls_snprintf( buf, buflen, "X509 - The CRT/CRL/CSR version element is invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_SERIAL) )
            mbedtls_snprintf( buf, buflen, "X509 - The serial tag or value is invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_ALG) )
            mbedtls_snprintf( buf, buflen, "X509 - The algorithm tag or value is invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_NAME) )
            mbedtls_snprintf( buf, buflen, "X509 - The name tag or value is invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_DATE) )
            mbedtls_snprintf( buf, buflen, "X509 - The date tag or value is invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_SIGNATURE) )
            mbedtls_snprintf( buf, buflen, "X509 - The signature tag or value invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_INVALID_EXTENSIONS) )
            mbedtls_snprintf( buf, buflen, "X509 - The extension tag or value is invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_UNKNOWN_VERSION) )
            mbedtls_snprintf( buf, buflen, "X509 - CRT/CRL/CSR has an unsupported version number" );
        if( use_ret == -(MBEDTLS_ERR_X509_UNKNOWN_SIG_ALG) )
            mbedtls_snprintf( buf, buflen, "X509 - Signature algorithm (oid) is unsupported" );
        if( use_ret == -(MBEDTLS_ERR_X509_SIG_MISMATCH) )
            mbedtls_snprintf( buf, buflen, "X509 - Signature algorithms do not match. (see \\c ::mbedtls_x509_crt sig_oid)" );
        if( use_ret == -(MBEDTLS_ERR_X509_CERT_VERIFY_FAILED) )
            mbedtls_snprintf( buf, buflen, "X509 - Certificate verification failed, e.g. CRL, CA or signature check failed" );
        if( use_ret == -(MBEDTLS_ERR_X509_CERT_UNKNOWN_FORMAT) )
            mbedtls_snprintf( buf, buflen, "X509 - Format not recognized as DER or PEM" );
        if( use_ret == -(MBEDTLS_ERR_X509_BAD_INPUT_DATA) )
            mbedtls_snprintf( buf, buflen, "X509 - Input invalid" );
        if( use_ret == -(MBEDTLS_ERR_X509_ALLOC_FAILED) )
            mbedtls_snprintf( buf, buflen, "X509 - Allocation of memory failed" );
        if( use_ret == -(MBEDTLS_ERR_X509_FILE_IO_ERROR) )
            mbedtls_snprintf( buf, buflen, "X509 - Read/write of file failed" );
        if( use_ret == -(MBEDTLS_ERR_X509_BUFFER_TOO_SMALL) )
            mbedtls_snprintf( buf, buflen, "X509 - Destination buffer is too small" );
        if( use_ret == -(MBEDTLS_ERR_X509_FATAL_ERROR) )
            mbedtls_snprintf( buf, buflen, "X509 - A fatal error occurred, eg the chain is too long or the vrfy callback failed" );
#endif /* MBEDTLS_X509_USE_C || MBEDTLS_X509_CREATE_C */
        // END generated code

        if( strlen( buf ) == 0 )
            mbedtls_snprintf( buf, buflen, "UNKNOWN ERROR CODE (%04X)", use_ret );
    }

    use_ret = ret & ~0xFF80;

    if( use_ret == 0 )
        return;

    // If high level code is present, make a concatenation between both
    // error strings.
    //
    len = strlen( buf );

    if( len > 0 )
    {
        if( buflen - len < 5 )
            return;

        mbedtls_snprintf( buf + len, buflen - len, " : " );

        buf += len + 3;
        buflen -= len + 3;
    }

    // Low level error codes
    //
    // BEGIN generated code
#if defined(MBEDTLS_AES_C)
    if( use_ret == -(MBEDTLS_ERR_AES_INVALID_KEY_LENGTH) )
        mbedtls_snprintf( buf, buflen, "AES - Invalid key length" );
    if( use_ret == -(MBEDTLS_ERR_AES_INVALID_INPUT_LENGTH) )
        mbedtls_snprintf( buf, buflen, "AES - Invalid data input length" );
    if( use_ret == -(MBEDTLS_ERR_AES_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "AES - Invalid input data" );
    if( use_ret == -(MBEDTLS_ERR_AES_FEATURE_UNAVAILABLE) )
        mbedtls_snprintf( buf, buflen, "AES - Feature not available. For example, an unsupported AES key size" );
    if( use_ret == -(MBEDTLS_ERR_AES_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "AES - AES hardware accelerator failed" );
#endif /* MBEDTLS_AES_C */

#if defined(MBEDTLS_ARC4_C)
    if( use_ret == -(MBEDTLS_ERR_ARC4_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "ARC4 - ARC4 hardware accelerator failed" );
#endif /* MBEDTLS_ARC4_C */

#if defined(MBEDTLS_ARIA_C)
    if( use_ret == -(MBEDTLS_ERR_ARIA_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "ARIA - Bad input data" );
    if( use_ret == -(MBEDTLS_ERR_ARIA_INVALID_INPUT_LENGTH) )
        mbedtls_snprintf( buf, buflen, "ARIA - Invalid data input length" );
    if( use_ret == -(MBEDTLS_ERR_ARIA_FEATURE_UNAVAILABLE) )
        mbedtls_snprintf( buf, buflen, "ARIA - Feature not available. For example, an unsupported ARIA key size" );
    if( use_ret == -(MBEDTLS_ERR_ARIA_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "ARIA - ARIA hardware accelerator failed" );
#endif /* MBEDTLS_ARIA_C */

#if defined(MBEDTLS_ASN1_PARSE_C)
    if( use_ret == -(MBEDTLS_ERR_ASN1_OUT_OF_DATA) )
        mbedtls_snprintf( buf, buflen, "ASN1 - Out of data when parsing an ASN1 data structure" );
    if( use_ret == -(MBEDTLS_ERR_ASN1_UNEXPECTED_TAG) )
        mbedtls_snprintf( buf, buflen, "ASN1 - ASN1 tag was of an unexpected value" );
    if( use_ret == -(MBEDTLS_ERR_ASN1_INVALID_LENGTH) )
        mbedtls_snprintf( buf, buflen, "ASN1 - Error when trying to determine the length or invalid length" );
    if( use_ret == -(MBEDTLS_ERR_ASN1_LENGTH_MISMATCH) )
        mbedtls_snprintf( buf, buflen, "ASN1 - Actual length differs from expected length" );
    if( use_ret == -(MBEDTLS_ERR_ASN1_INVALID_DATA) )
        mbedtls_snprintf( buf, buflen, "ASN1 - Data is invalid. (not used)" );
    if( use_ret == -(MBEDTLS_ERR_ASN1_ALLOC_FAILED) )
        mbedtls_snprintf( buf, buflen, "ASN1 - Memory allocation failed" );
    if( use_ret == -(MBEDTLS_ERR_ASN1_BUF_TOO_SMALL) )
        mbedtls_snprintf( buf, buflen, "ASN1 - Buffer too small when writing ASN.1 data structure" );
#endif /* MBEDTLS_ASN1_PARSE_C */

#if defined(MBEDTLS_BASE64_C)
    if( use_ret == -(MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL) )
        mbedtls_snprintf( buf, buflen, "BASE64 - Output buffer too small" );
    if( use_ret == -(MBEDTLS_ERR_BASE64_INVALID_CHARACTER) )
        mbedtls_snprintf( buf, buflen, "BASE64 - Invalid character in input" );
#endif /* MBEDTLS_BASE64_C */

#if defined(MBEDTLS_BIGNUM_C)
    if( use_ret == -(MBEDTLS_ERR_MPI_FILE_IO_ERROR) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - An error occurred while reading from or writing to a file" );
    if( use_ret == -(MBEDTLS_ERR_MPI_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - Bad input parameters to function" );
    if( use_ret == -(MBEDTLS_ERR_MPI_INVALID_CHARACTER) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - There is an invalid character in the digit string" );
    if( use_ret == -(MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - The buffer is too small to write to" );
    if( use_ret == -(MBEDTLS_ERR_MPI_NEGATIVE_VALUE) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - The input arguments are negative or result in illegal output" );
    if( use_ret == -(MBEDTLS_ERR_MPI_DIVISION_BY_ZERO) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - The input argument for division is zero, which is not allowed" );
    if( use_ret == -(MBEDTLS_ERR_MPI_NOT_ACCEPTABLE) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - The input arguments are not acceptable" );
    if( use_ret == -(MBEDTLS_ERR_MPI_ALLOC_FAILED) )
        mbedtls_snprintf( buf, buflen, "BIGNUM - Memory allocation failed" );
#endif /* MBEDTLS_BIGNUM_C */

#if defined(MBEDTLS_BLOWFISH_C)
    if( use_ret == -(MBEDTLS_ERR_BLOWFISH_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "BLOWFISH - Bad input data" );
    if( use_ret == -(MBEDTLS_ERR_BLOWFISH_INVALID_INPUT_LENGTH) )
        mbedtls_snprintf( buf, buflen, "BLOWFISH - Invalid data input length" );
    if( use_ret == -(MBEDTLS_ERR_BLOWFISH_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "BLOWFISH - Blowfish hardware accelerator failed" );
#endif /* MBEDTLS_BLOWFISH_C */

#if defined(MBEDTLS_CAMELLIA_C)
    if( use_ret == -(MBEDTLS_ERR_CAMELLIA_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "CAMELLIA - Bad input data" );
    if( use_ret == -(MBEDTLS_ERR_CAMELLIA_INVALID_INPUT_LENGTH) )
        mbedtls_snprintf( buf, buflen, "CAMELLIA - Invalid data input length" );
    if( use_ret == -(MBEDTLS_ERR_CAMELLIA_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "CAMELLIA - Camellia hardware accelerator failed" );
#endif /* MBEDTLS_CAMELLIA_C */

#if defined(MBEDTLS_CCM_C)
    if( use_ret == -(MBEDTLS_ERR_CCM_BAD_INPUT) )
        mbedtls_snprintf( buf, buflen, "CCM - Bad input parameters to the function" );
    if( use_ret == -(MBEDTLS_ERR_CCM_AUTH_FAILED) )
        mbedtls_snprintf( buf, buflen, "CCM - Authenticated decryption failed" );
    if( use_ret == -(MBEDTLS_ERR_CCM_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "CCM - CCM hardware accelerator failed" );
#endif /* MBEDTLS_CCM_C */

#if defined(MBEDTLS_CHACHA20_C)
    if( use_ret == -(MBEDTLS_ERR_CHACHA20_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "CHACHA20 - Invalid input parameter(s)" );
    if( use_ret == -(MBEDTLS_ERR_CHACHA20_FEATURE_UNAVAILABLE) )
        mbedtls_snprintf( buf, buflen, "CHACHA20 - Feature not available. For example, s part of the API is not implemented" );
    if( use_ret == -(MBEDTLS_ERR_CHACHA20_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "CHACHA20 - Chacha20 hardware accelerator failed" );
#endif /* MBEDTLS_CHACHA20_C */

#if defined(MBEDTLS_CHACHAPOLY_C)
    if( use_ret == -(MBEDTLS_ERR_CHACHAPOLY_BAD_STATE) )
        mbedtls_snprintf( buf, buflen, "CHACHAPOLY - The requested operation is not permitted in the current state" );
    if( use_ret == -(MBEDTLS_ERR_CHACHAPOLY_AUTH_FAILED) )
        mbedtls_snprintf( buf, buflen, "CHACHAPOLY - Authenticated decryption failed: data was not authentic" );
#endif /* MBEDTLS_CHACHAPOLY_C */

#if defined(MBEDTLS_CMAC_C)
    if( use_ret == -(MBEDTLS_ERR_CMAC_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "CMAC - CMAC hardware accelerator failed" );
#endif /* MBEDTLS_CMAC_C */

#if defined(MBEDTLS_CTR_DRBG_C)
    if( use_ret == -(MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED) )
        mbedtls_snprintf( buf, buflen, "CTR_DRBG - The entropy source failed" );
    if( use_ret == -(MBEDTLS_ERR_CTR_DRBG_REQUEST_TOO_BIG) )
        mbedtls_snprintf( buf, buflen, "CTR_DRBG - The requested random buffer length is too big" );
    if( use_ret == -(MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG) )
        mbedtls_snprintf( buf, buflen, "CTR_DRBG - The input (entropy + additional data) is too large" );
    if( use_ret == -(MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR) )
        mbedtls_snprintf( buf, buflen, "CTR_DRBG - Read or write error in file" );
#endif /* MBEDTLS_CTR_DRBG_C */

#if defined(MBEDTLS_DES_C)
    if( use_ret == -(MBEDTLS_ERR_DES_INVALID_INPUT_LENGTH) )
        mbedtls_snprintf( buf, buflen, "DES - The data input has an invalid length" );
    if( use_ret == -(MBEDTLS_ERR_DES_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "DES - DES hardware accelerator failed" );
#endif /* MBEDTLS_DES_C */

#if defined(MBEDTLS_ENTROPY_C)
    if( use_ret == -(MBEDTLS_ERR_ENTROPY_SOURCE_FAILED) )
        mbedtls_snprintf( buf, buflen, "ENTROPY - Critical entropy source failure" );
    if( use_ret == -(MBEDTLS_ERR_ENTROPY_MAX_SOURCES) )
        mbedtls_snprintf( buf, buflen, "ENTROPY - No more sources can be added" );
    if( use_ret == -(MBEDTLS_ERR_ENTROPY_NO_SOURCES_DEFINED) )
        mbedtls_snprintf( buf, buflen, "ENTROPY - No sources have been added to poll" );
    if( use_ret == -(MBEDTLS_ERR_ENTROPY_NO_STRONG_SOURCE) )
        mbedtls_snprintf( buf, buflen, "ENTROPY - No strong sources have been added to poll" );
    if( use_ret == -(MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR) )
        mbedtls_snprintf( buf, buflen, "ENTROPY - Read/write error in file" );
#endif /* MBEDTLS_ENTROPY_C */

#if defined(MBEDTLS_GCM_C)
    if( use_ret == -(MBEDTLS_ERR_GCM_AUTH_FAILED) )
        mbedtls_snprintf( buf, buflen, "GCM - Authenticated decryption failed" );
    if( use_ret == -(MBEDTLS_ERR_GCM_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "GCM - GCM hardware accelerator failed" );
    if( use_ret == -(MBEDTLS_ERR_GCM_BAD_INPUT) )
        mbedtls_snprintf( buf, buflen, "GCM - Bad input parameters to function" );
#endif /* MBEDTLS_GCM_C */

#if defined(MBEDTLS_HKDF_C)
    if( use_ret == -(MBEDTLS_ERR_HKDF_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "HKDF - Bad input parameters to function" );
#endif /* MBEDTLS_HKDF_C */

#if defined(MBEDTLS_HMAC_DRBG_C)
    if( use_ret == -(MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG) )
        mbedtls_snprintf( buf, buflen, "HMAC_DRBG - Too many random requested in single call" );
    if( use_ret == -(MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG) )
        mbedtls_snprintf( buf, buflen, "HMAC_DRBG - Input too large (Entropy + additional)" );
    if( use_ret == -(MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR) )
        mbedtls_snprintf( buf, buflen, "HMAC_DRBG - Read/write error in file" );
    if( use_ret == -(MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED) )
        mbedtls_snprintf( buf, buflen, "HMAC_DRBG - The entropy source failed" );
#endif /* MBEDTLS_HMAC_DRBG_C */

#if defined(MBEDTLS_MD2_C)
    if( use_ret == -(MBEDTLS_ERR_MD2_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "MD2 - MD2 hardware accelerator failed" );
#endif /* MBEDTLS_MD2_C */

#if defined(MBEDTLS_MD4_C)
    if( use_ret == -(MBEDTLS_ERR_MD4_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "MD4 - MD4 hardware accelerator failed" );
#endif /* MBEDTLS_MD4_C */

#if defined(MBEDTLS_MD5_C)
    if( use_ret == -(MBEDTLS_ERR_MD5_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "MD5 - MD5 hardware accelerator failed" );
#endif /* MBEDTLS_MD5_C */

#if defined(MBEDTLS_NET_C)
    if( use_ret == -(MBEDTLS_ERR_NET_SOCKET_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - Failed to open a socket" );
    if( use_ret == -(MBEDTLS_ERR_NET_CONNECT_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - The connection to the given server / port failed" );
    if( use_ret == -(MBEDTLS_ERR_NET_BIND_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - Binding of the socket failed" );
    if( use_ret == -(MBEDTLS_ERR_NET_LISTEN_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - Could not listen on the socket" );
    if( use_ret == -(MBEDTLS_ERR_NET_ACCEPT_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - Could not accept the incoming connection" );
    if( use_ret == -(MBEDTLS_ERR_NET_RECV_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - Reading information from the socket failed" );
    if( use_ret == -(MBEDTLS_ERR_NET_SEND_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - Sending information through the socket failed" );
    if( use_ret == -(MBEDTLS_ERR_NET_CONN_RESET) )
        mbedtls_snprintf( buf, buflen, "NET - Connection was reset by peer" );
    if( use_ret == -(MBEDTLS_ERR_NET_UNKNOWN_HOST) )
        mbedtls_snprintf( buf, buflen, "NET - Failed to get an IP address for the given hostname" );
    if( use_ret == -(MBEDTLS_ERR_NET_BUFFER_TOO_SMALL) )
        mbedtls_snprintf( buf, buflen, "NET - Buffer is too small to hold the data" );
    if( use_ret == -(MBEDTLS_ERR_NET_INVALID_CONTEXT) )
        mbedtls_snprintf( buf, buflen, "NET - The context is invalid, eg because it was free()ed" );
    if( use_ret == -(MBEDTLS_ERR_NET_POLL_FAILED) )
        mbedtls_snprintf( buf, buflen, "NET - Polling the net context failed" );
    if( use_ret == -(MBEDTLS_ERR_NET_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "NET - Input invalid" );
#endif /* MBEDTLS_NET_C */

#if defined(MBEDTLS_OID_C)
    if( use_ret == -(MBEDTLS_ERR_OID_NOT_FOUND) )
        mbedtls_snprintf( buf, buflen, "OID - OID is not found" );
    if( use_ret == -(MBEDTLS_ERR_OID_BUF_TOO_SMALL) )
        mbedtls_snprintf( buf, buflen, "OID - output buffer is too small" );
#endif /* MBEDTLS_OID_C */

#if defined(MBEDTLS_PADLOCK_C)
    if( use_ret == -(MBEDTLS_ERR_PADLOCK_DATA_MISALIGNED) )
        mbedtls_snprintf( buf, buflen, "PADLOCK - Input data should be aligned" );
#endif /* MBEDTLS_PADLOCK_C */

#if defined(MBEDTLS_PLATFORM_C)
    if( use_ret == -(MBEDTLS_ERR_PLATFORM_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "PLATFORM - Hardware accelerator failed" );
    if( use_ret == -(MBEDTLS_ERR_PLATFORM_FEATURE_UNSUPPORTED) )
        mbedtls_snprintf( buf, buflen, "PLATFORM - The requested feature is not supported by the platform" );
#endif /* MBEDTLS_PLATFORM_C */

#if defined(MBEDTLS_POLY1305_C)
    if( use_ret == -(MBEDTLS_ERR_POLY1305_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "POLY1305 - Invalid input parameter(s)" );
    if( use_ret == -(MBEDTLS_ERR_POLY1305_FEATURE_UNAVAILABLE) )
        mbedtls_snprintf( buf, buflen, "POLY1305 - Feature not available. For example, s part of the API is not implemented" );
    if( use_ret == -(MBEDTLS_ERR_POLY1305_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "POLY1305 - Poly1305 hardware accelerator failed" );
#endif /* MBEDTLS_POLY1305_C */

#if defined(MBEDTLS_RIPEMD160_C)
    if( use_ret == -(MBEDTLS_ERR_RIPEMD160_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "RIPEMD160 - RIPEMD160 hardware accelerator failed" );
#endif /* MBEDTLS_RIPEMD160_C */

#if defined(MBEDTLS_SHA1_C)
    if( use_ret == -(MBEDTLS_ERR_SHA1_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "SHA1 - SHA-1 hardware accelerator failed" );
    if( use_ret == -(MBEDTLS_ERR_SHA1_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "SHA1 - SHA-1 input data was malformed" );
#endif /* MBEDTLS_SHA1_C */

#if defined(MBEDTLS_SHA256_C)
    if( use_ret == -(MBEDTLS_ERR_SHA256_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "SHA256 - SHA-256 hardware accelerator failed" );
    if( use_ret == -(MBEDTLS_ERR_SHA256_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "SHA256 - SHA-256 input data was malformed" );
#endif /* MBEDTLS_SHA256_C */

#if defined(MBEDTLS_SHA512_C)
    if( use_ret == -(MBEDTLS_ERR_SHA512_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "SHA512 - SHA-512 hardware accelerator failed" );
    if( use_ret == -(MBEDTLS_ERR_SHA512_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "SHA512 - SHA-512 input data was malformed" );
#endif /* MBEDTLS_SHA512_C */

#if defined(MBEDTLS_THREADING_C)
    if( use_ret == -(MBEDTLS_ERR_THREADING_FEATURE_UNAVAILABLE) )
        mbedtls_snprintf( buf, buflen, "THREADING - The selected feature is not available" );
    if( use_ret == -(MBEDTLS_ERR_THREADING_BAD_INPUT_DATA) )
        mbedtls_snprintf( buf, buflen, "THREADING - Bad input parameters to function" );
    if( use_ret == -(MBEDTLS_ERR_THREADING_MUTEX_ERROR) )
        mbedtls_snprintf( buf, buflen, "THREADING - Locking / unlocking / free failed with error code" );
#endif /* MBEDTLS_THREADING_C */

#if defined(MBEDTLS_XTEA_C)
    if( use_ret == -(MBEDTLS_ERR_XTEA_INVALID_INPUT_LENGTH) )
        mbedtls_snprintf( buf, buflen, "XTEA - The data input has an invalid length" );
    if( use_ret == -(MBEDTLS_ERR_XTEA_HW_ACCEL_FAILED) )
        mbedtls_snprintf( buf, buflen, "XTEA - XTEA hardware accelerator failed" );
#endif /* MBEDTLS_XTEA_C */
    // END generated code

    if( strlen( buf ) != 0 )
        return;

    mbedtls_snprintf( buf, buflen, "UNKNOWN ERROR CODE (%04X)", use_ret );
}

#else /* MBEDTLS_ERROR_C */

#if defined(MBEDTLS_ERROR_STRERROR_DUMMY)

/*
 * Provide an non-function in case MBEDTLS_ERROR_C is not defined
 */
void mbedtls_strerror( int ret, char *buf, size_t buflen )
{
    ((void) ret);

    if( buflen > 0 )
        buf[0] = '\0';
}

#endif /* MBEDTLS_ERROR_STRERROR_DUMMY */

#endif /* MBEDTLS_ERROR_C */
