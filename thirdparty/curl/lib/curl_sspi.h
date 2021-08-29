#ifndef HEADER_CURL_SSPI_H
#define HEADER_CURL_SSPI_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

#include "curl_setup.h"

#ifdef USE_WINDOWS_SSPI

#include <curl/curl.h>

/*
 * When including the following three headers, it is mandatory to define either
 * SECURITY_WIN32 or SECURITY_KERNEL, indicating who is compiling the code.
 */

#undef SECURITY_WIN32
#undef SECURITY_KERNEL
#define SECURITY_WIN32 1
#include <security.h>
#include <sspi.h>
#include <rpc.h>

CURLcode Curl_sspi_global_init(void);
void Curl_sspi_global_cleanup(void);

/* This is used to populate the domain in a SSPI identity structure */
CURLcode Curl_override_sspi_http_realm(const char *chlg,
                                       SEC_WINNT_AUTH_IDENTITY *identity);

/* This is used to generate an SSPI identity structure */
CURLcode Curl_create_sspi_identity(const char *userp, const char *passwdp,
                                   SEC_WINNT_AUTH_IDENTITY *identity);

/* This is used to free an SSPI identity structure */
void Curl_sspi_free_identity(SEC_WINNT_AUTH_IDENTITY *identity);

/* Forward-declaration of global variables defined in curl_sspi.c */
extern HMODULE s_hSecDll;
extern PSecurityFunctionTable s_pSecFn;

/* Provide some definitions missing in old headers */
#define SP_NAME_DIGEST              "WDigest"
#define SP_NAME_NTLM                "NTLM"
#define SP_NAME_NEGOTIATE           "Negotiate"
#define SP_NAME_KERBEROS            "Kerberos"

#ifndef ISC_REQ_USE_HTTP_STYLE
#define ISC_REQ_USE_HTTP_STYLE                0x01000000
#endif

#ifndef ISC_RET_REPLAY_DETECT
#define ISC_RET_REPLAY_DETECT                 0x00000004
#endif

#ifndef ISC_RET_SEQUENCE_DETECT
#define ISC_RET_SEQUENCE_DETECT               0x00000008
#endif

#ifndef ISC_RET_CONFIDENTIALITY
#define ISC_RET_CONFIDENTIALITY               0x00000010
#endif

#ifndef ISC_RET_ALLOCATED_MEMORY
#define ISC_RET_ALLOCATED_MEMORY              0x00000100
#endif

#ifndef ISC_RET_STREAM
#define ISC_RET_STREAM                        0x00008000
#endif

#ifndef SEC_E_INSUFFICIENT_MEMORY
# define SEC_E_INSUFFICIENT_MEMORY            ((HRESULT)0x80090300L)
#endif
#ifndef SEC_E_INVALID_HANDLE
# define SEC_E_INVALID_HANDLE                 ((HRESULT)0x80090301L)
#endif
#ifndef SEC_E_UNSUPPORTED_FUNCTION
# define SEC_E_UNSUPPORTED_FUNCTION           ((HRESULT)0x80090302L)
#endif
#ifndef SEC_E_TARGET_UNKNOWN
# define SEC_E_TARGET_UNKNOWN                 ((HRESULT)0x80090303L)
#endif
#ifndef SEC_E_INTERNAL_ERROR
# define SEC_E_INTERNAL_ERROR                 ((HRESULT)0x80090304L)
#endif
#ifndef SEC_E_SECPKG_NOT_FOUND
# define SEC_E_SECPKG_NOT_FOUND               ((HRESULT)0x80090305L)
#endif
#ifndef SEC_E_NOT_OWNER
# define SEC_E_NOT_OWNER                      ((HRESULT)0x80090306L)
#endif
#ifndef SEC_E_CANNOT_INSTALL
# define SEC_E_CANNOT_INSTALL                 ((HRESULT)0x80090307L)
#endif
#ifndef SEC_E_INVALID_TOKEN
# define SEC_E_INVALID_TOKEN                  ((HRESULT)0x80090308L)
#endif
#ifndef SEC_E_CANNOT_PACK
# define SEC_E_CANNOT_PACK                    ((HRESULT)0x80090309L)
#endif
#ifndef SEC_E_QOP_NOT_SUPPORTED
# define SEC_E_QOP_NOT_SUPPORTED              ((HRESULT)0x8009030AL)
#endif
#ifndef SEC_E_NO_IMPERSONATION
# define SEC_E_NO_IMPERSONATION               ((HRESULT)0x8009030BL)
#endif
#ifndef SEC_E_LOGON_DENIED
# define SEC_E_LOGON_DENIED                   ((HRESULT)0x8009030CL)
#endif
#ifndef SEC_E_UNKNOWN_CREDENTIALS
# define SEC_E_UNKNOWN_CREDENTIALS            ((HRESULT)0x8009030DL)
#endif
#ifndef SEC_E_NO_CREDENTIALS
# define SEC_E_NO_CREDENTIALS                 ((HRESULT)0x8009030EL)
#endif
#ifndef SEC_E_MESSAGE_ALTERED
# define SEC_E_MESSAGE_ALTERED                ((HRESULT)0x8009030FL)
#endif
#ifndef SEC_E_OUT_OF_SEQUENCE
# define SEC_E_OUT_OF_SEQUENCE                ((HRESULT)0x80090310L)
#endif
#ifndef SEC_E_NO_AUTHENTICATING_AUTHORITY
# define SEC_E_NO_AUTHENTICATING_AUTHORITY    ((HRESULT)0x80090311L)
#endif
#ifndef SEC_E_BAD_PKGID
# define SEC_E_BAD_PKGID                      ((HRESULT)0x80090316L)
#endif
#ifndef SEC_E_CONTEXT_EXPIRED
# define SEC_E_CONTEXT_EXPIRED                ((HRESULT)0x80090317L)
#endif
#ifndef SEC_E_INCOMPLETE_MESSAGE
# define SEC_E_INCOMPLETE_MESSAGE             ((HRESULT)0x80090318L)
#endif
#ifndef SEC_E_INCOMPLETE_CREDENTIALS
# define SEC_E_INCOMPLETE_CREDENTIALS         ((HRESULT)0x80090320L)
#endif
#ifndef SEC_E_BUFFER_TOO_SMALL
# define SEC_E_BUFFER_TOO_SMALL               ((HRESULT)0x80090321L)
#endif
#ifndef SEC_E_WRONG_PRINCIPAL
# define SEC_E_WRONG_PRINCIPAL                ((HRESULT)0x80090322L)
#endif
#ifndef SEC_E_TIME_SKEW
# define SEC_E_TIME_SKEW                      ((HRESULT)0x80090324L)
#endif
#ifndef SEC_E_UNTRUSTED_ROOT
# define SEC_E_UNTRUSTED_ROOT                 ((HRESULT)0x80090325L)
#endif
#ifndef SEC_E_ILLEGAL_MESSAGE
# define SEC_E_ILLEGAL_MESSAGE                ((HRESULT)0x80090326L)
#endif
#ifndef SEC_E_CERT_UNKNOWN
# define SEC_E_CERT_UNKNOWN                   ((HRESULT)0x80090327L)
#endif
#ifndef SEC_E_CERT_EXPIRED
# define SEC_E_CERT_EXPIRED                   ((HRESULT)0x80090328L)
#endif
#ifndef SEC_E_ENCRYPT_FAILURE
# define SEC_E_ENCRYPT_FAILURE                ((HRESULT)0x80090329L)
#endif
#ifndef SEC_E_DECRYPT_FAILURE
# define SEC_E_DECRYPT_FAILURE                ((HRESULT)0x80090330L)
#endif
#ifndef SEC_E_ALGORITHM_MISMATCH
# define SEC_E_ALGORITHM_MISMATCH             ((HRESULT)0x80090331L)
#endif
#ifndef SEC_E_SECURITY_QOS_FAILED
# define SEC_E_SECURITY_QOS_FAILED            ((HRESULT)0x80090332L)
#endif
#ifndef SEC_E_UNFINISHED_CONTEXT_DELETED
# define SEC_E_UNFINISHED_CONTEXT_DELETED     ((HRESULT)0x80090333L)
#endif
#ifndef SEC_E_NO_TGT_REPLY
# define SEC_E_NO_TGT_REPLY                   ((HRESULT)0x80090334L)
#endif
#ifndef SEC_E_NO_IP_ADDRESSES
# define SEC_E_NO_IP_ADDRESSES                ((HRESULT)0x80090335L)
#endif
#ifndef SEC_E_WRONG_CREDENTIAL_HANDLE
# define SEC_E_WRONG_CREDENTIAL_HANDLE        ((HRESULT)0x80090336L)
#endif
#ifndef SEC_E_CRYPTO_SYSTEM_INVALID
# define SEC_E_CRYPTO_SYSTEM_INVALID          ((HRESULT)0x80090337L)
#endif
#ifndef SEC_E_MAX_REFERRALS_EXCEEDED
# define SEC_E_MAX_REFERRALS_EXCEEDED         ((HRESULT)0x80090338L)
#endif
#ifndef SEC_E_MUST_BE_KDC
# define SEC_E_MUST_BE_KDC                    ((HRESULT)0x80090339L)
#endif
#ifndef SEC_E_STRONG_CRYPTO_NOT_SUPPORTED
# define SEC_E_STRONG_CRYPTO_NOT_SUPPORTED    ((HRESULT)0x8009033AL)
#endif
#ifndef SEC_E_TOO_MANY_PRINCIPALS
# define SEC_E_TOO_MANY_PRINCIPALS            ((HRESULT)0x8009033BL)
#endif
#ifndef SEC_E_NO_PA_DATA
# define SEC_E_NO_PA_DATA                     ((HRESULT)0x8009033CL)
#endif
#ifndef SEC_E_PKINIT_NAME_MISMATCH
# define SEC_E_PKINIT_NAME_MISMATCH           ((HRESULT)0x8009033DL)
#endif
#ifndef SEC_E_SMARTCARD_LOGON_REQUIRED
# define SEC_E_SMARTCARD_LOGON_REQUIRED       ((HRESULT)0x8009033EL)
#endif
#ifndef SEC_E_SHUTDOWN_IN_PROGRESS
# define SEC_E_SHUTDOWN_IN_PROGRESS           ((HRESULT)0x8009033FL)
#endif
#ifndef SEC_E_KDC_INVALID_REQUEST
# define SEC_E_KDC_INVALID_REQUEST            ((HRESULT)0x80090340L)
#endif
#ifndef SEC_E_KDC_UNABLE_TO_REFER
# define SEC_E_KDC_UNABLE_TO_REFER            ((HRESULT)0x80090341L)
#endif
#ifndef SEC_E_KDC_UNKNOWN_ETYPE
# define SEC_E_KDC_UNKNOWN_ETYPE              ((HRESULT)0x80090342L)
#endif
#ifndef SEC_E_UNSUPPORTED_PREAUTH
# define SEC_E_UNSUPPORTED_PREAUTH            ((HRESULT)0x80090343L)
#endif
#ifndef SEC_E_DELEGATION_REQUIRED
# define SEC_E_DELEGATION_REQUIRED            ((HRESULT)0x80090345L)
#endif
#ifndef SEC_E_BAD_BINDINGS
# define SEC_E_BAD_BINDINGS                   ((HRESULT)0x80090346L)
#endif
#ifndef SEC_E_MULTIPLE_ACCOUNTS
# define SEC_E_MULTIPLE_ACCOUNTS              ((HRESULT)0x80090347L)
#endif
#ifndef SEC_E_NO_KERB_KEY
# define SEC_E_NO_KERB_KEY                    ((HRESULT)0x80090348L)
#endif
#ifndef SEC_E_CERT_WRONG_USAGE
# define SEC_E_CERT_WRONG_USAGE               ((HRESULT)0x80090349L)
#endif
#ifndef SEC_E_DOWNGRADE_DETECTED
# define SEC_E_DOWNGRADE_DETECTED             ((HRESULT)0x80090350L)
#endif
#ifndef SEC_E_SMARTCARD_CERT_REVOKED
# define SEC_E_SMARTCARD_CERT_REVOKED         ((HRESULT)0x80090351L)
#endif
#ifndef SEC_E_ISSUING_CA_UNTRUSTED
# define SEC_E_ISSUING_CA_UNTRUSTED           ((HRESULT)0x80090352L)
#endif
#ifndef SEC_E_REVOCATION_OFFLINE_C
# define SEC_E_REVOCATION_OFFLINE_C           ((HRESULT)0x80090353L)
#endif
#ifndef SEC_E_PKINIT_CLIENT_FAILURE
# define SEC_E_PKINIT_CLIENT_FAILURE          ((HRESULT)0x80090354L)
#endif
#ifndef SEC_E_SMARTCARD_CERT_EXPIRED
# define SEC_E_SMARTCARD_CERT_EXPIRED         ((HRESULT)0x80090355L)
#endif
#ifndef SEC_E_NO_S4U_PROT_SUPPORT
# define SEC_E_NO_S4U_PROT_SUPPORT            ((HRESULT)0x80090356L)
#endif
#ifndef SEC_E_CROSSREALM_DELEGATION_FAILURE
# define SEC_E_CROSSREALM_DELEGATION_FAILURE  ((HRESULT)0x80090357L)
#endif
#ifndef SEC_E_REVOCATION_OFFLINE_KDC
# define SEC_E_REVOCATION_OFFLINE_KDC         ((HRESULT)0x80090358L)
#endif
#ifndef SEC_E_ISSUING_CA_UNTRUSTED_KDC
# define SEC_E_ISSUING_CA_UNTRUSTED_KDC       ((HRESULT)0x80090359L)
#endif
#ifndef SEC_E_KDC_CERT_EXPIRED
# define SEC_E_KDC_CERT_EXPIRED               ((HRESULT)0x8009035AL)
#endif
#ifndef SEC_E_KDC_CERT_REVOKED
# define SEC_E_KDC_CERT_REVOKED               ((HRESULT)0x8009035BL)
#endif
#ifndef SEC_E_INVALID_PARAMETER
# define SEC_E_INVALID_PARAMETER              ((HRESULT)0x8009035DL)
#endif
#ifndef SEC_E_DELEGATION_POLICY
# define SEC_E_DELEGATION_POLICY              ((HRESULT)0x8009035EL)
#endif
#ifndef SEC_E_POLICY_NLTM_ONLY
# define SEC_E_POLICY_NLTM_ONLY               ((HRESULT)0x8009035FL)
#endif

#ifndef SEC_I_CONTINUE_NEEDED
# define SEC_I_CONTINUE_NEEDED                ((HRESULT)0x00090312L)
#endif
#ifndef SEC_I_COMPLETE_NEEDED
# define SEC_I_COMPLETE_NEEDED                ((HRESULT)0x00090313L)
#endif
#ifndef SEC_I_COMPLETE_AND_CONTINUE
# define SEC_I_COMPLETE_AND_CONTINUE          ((HRESULT)0x00090314L)
#endif
#ifndef SEC_I_LOCAL_LOGON
# define SEC_I_LOCAL_LOGON                    ((HRESULT)0x00090315L)
#endif
#ifndef SEC_I_CONTEXT_EXPIRED
# define SEC_I_CONTEXT_EXPIRED                ((HRESULT)0x00090317L)
#endif
#ifndef SEC_I_INCOMPLETE_CREDENTIALS
# define SEC_I_INCOMPLETE_CREDENTIALS         ((HRESULT)0x00090320L)
#endif
#ifndef SEC_I_RENEGOTIATE
# define SEC_I_RENEGOTIATE                    ((HRESULT)0x00090321L)
#endif
#ifndef SEC_I_NO_LSA_CONTEXT
# define SEC_I_NO_LSA_CONTEXT                 ((HRESULT)0x00090323L)
#endif
#ifndef SEC_I_SIGNATURE_NEEDED
# define SEC_I_SIGNATURE_NEEDED               ((HRESULT)0x0009035CL)
#endif

#ifndef CRYPT_E_REVOKED
# define CRYPT_E_REVOKED                      ((HRESULT)0x80092010L)
#endif

#ifdef UNICODE
#  define SECFLAG_WINNT_AUTH_IDENTITY \
     (unsigned long)SEC_WINNT_AUTH_IDENTITY_UNICODE
#else
#  define SECFLAG_WINNT_AUTH_IDENTITY \
     (unsigned long)SEC_WINNT_AUTH_IDENTITY_ANSI
#endif

/*
 * Definitions required from ntsecapi.h are directly provided below this point
 * to avoid including ntsecapi.h due to a conflict with OpenSSL's safestack.h
 */
#define KERB_WRAP_NO_ENCRYPT 0x80000001

#endif /* USE_WINDOWS_SSPI */

#endif /* HEADER_CURL_SSPI_H */
