/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2012 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) 2012 - 2016, Marc Hoersken, <info@marc-hoersken.de>
 * Copyright (C) 2012, Mark Salisbury, <mark.salisbury@hp.com>
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

/*
 * Source file for all Schannel-specific code for the TLS/SSL layer. No code
 * but vtls.c should ever call or use these functions.
 */

#include "curl_setup.h"

#ifdef USE_SCHANNEL

#define EXPOSE_SCHANNEL_INTERNAL_STRUCTS

#ifndef USE_WINDOWS_SSPI
#  error "Can't compile SCHANNEL support without SSPI."
#endif

#include "schannel.h"
#include "vtls.h"
#include "strcase.h"
#include "sendf.h"
#include "connect.h" /* for the connect timeout */
#include "strerror.h"
#include "select.h" /* for the socket readiness */
#include "inet_pton.h" /* for IP addr SNI check */
#include "curl_multibyte.h"
#include "warnless.h"
#include "x509asn1.h"
#include "curl_printf.h"
#include "multiif.h"
#include "version_win32.h"

/* The last #include file should be: */
#include "curl_memory.h"
#include "memdebug.h"

/* ALPN requires version 8.1 of the Windows SDK, which was
   shipped with Visual Studio 2013, aka _MSC_VER 1800:

   https://technet.microsoft.com/en-us/library/hh831771%28v=ws.11%29.aspx
*/
#if defined(_MSC_VER) && (_MSC_VER >= 1800) && !defined(_USING_V110_SDK71_)
#  define HAS_ALPN 1
#endif

#ifndef UNISP_NAME_A
#define UNISP_NAME_A "Microsoft Unified Security Protocol Provider"
#endif

#ifndef UNISP_NAME_W
#define UNISP_NAME_W L"Microsoft Unified Security Protocol Provider"
#endif

#ifndef UNISP_NAME
#ifdef UNICODE
#define UNISP_NAME  UNISP_NAME_W
#else
#define UNISP_NAME  UNISP_NAME_A
#endif
#endif

#if defined(CryptStringToBinary) && defined(CRYPT_STRING_HEX)
#define HAS_CLIENT_CERT_PATH
#endif

#ifdef HAS_CLIENT_CERT_PATH
#ifdef UNICODE
#define CURL_CERT_STORE_PROV_SYSTEM CERT_STORE_PROV_SYSTEM_W
#else
#define CURL_CERT_STORE_PROV_SYSTEM CERT_STORE_PROV_SYSTEM_A
#endif
#endif

#ifndef SP_PROT_SSL2_CLIENT
#define SP_PROT_SSL2_CLIENT             0x00000008
#endif

#ifndef SP_PROT_SSL3_CLIENT
#define SP_PROT_SSL3_CLIENT             0x00000008
#endif

#ifndef SP_PROT_TLS1_CLIENT
#define SP_PROT_TLS1_CLIENT             0x00000080
#endif

#ifndef SP_PROT_TLS1_0_CLIENT
#define SP_PROT_TLS1_0_CLIENT           SP_PROT_TLS1_CLIENT
#endif

#ifndef SP_PROT_TLS1_1_CLIENT
#define SP_PROT_TLS1_1_CLIENT           0x00000200
#endif

#ifndef SP_PROT_TLS1_2_CLIENT
#define SP_PROT_TLS1_2_CLIENT           0x00000800
#endif

#ifndef SCH_USE_STRONG_CRYPTO
#define SCH_USE_STRONG_CRYPTO           0x00400000
#endif

#ifndef SECBUFFER_ALERT
#define SECBUFFER_ALERT                 17
#endif

/* Both schannel buffer sizes must be > 0 */
#define CURL_SCHANNEL_BUFFER_INIT_SIZE   4096
#define CURL_SCHANNEL_BUFFER_FREE_SIZE   1024

#define CERT_THUMBPRINT_STR_LEN 40
#define CERT_THUMBPRINT_DATA_LEN 20

/* Uncomment to force verbose output
 * #define infof(x, y, ...) printf(y, __VA_ARGS__)
 * #define failf(x, y, ...) printf(y, __VA_ARGS__)
 */

#ifndef CALG_SHA_256
#  define CALG_SHA_256 0x0000800c
#endif

/* Work around typo in classic MinGW's w32api up to version 5.0,
   see https://osdn.net/projects/mingw/ticket/38391 */
#if !defined(ALG_CLASS_DHASH) && defined(ALG_CLASS_HASH)
#define ALG_CLASS_DHASH ALG_CLASS_HASH
#endif

#define BACKEND connssl->backend

static Curl_recv schannel_recv;
static Curl_send schannel_send;

static CURLcode pkp_pin_peer_pubkey(struct Curl_easy *data,
                                    struct connectdata *conn, int sockindex,
                                    const char *pinnedpubkey);

static void InitSecBuffer(SecBuffer *buffer, unsigned long BufType,
                          void *BufDataPtr, unsigned long BufByteSize)
{
  buffer->cbBuffer = BufByteSize;
  buffer->BufferType = BufType;
  buffer->pvBuffer = BufDataPtr;
}

static void InitSecBufferDesc(SecBufferDesc *desc, SecBuffer *BufArr,
                              unsigned long NumArrElem)
{
  desc->ulVersion = SECBUFFER_VERSION;
  desc->pBuffers = BufArr;
  desc->cBuffers = NumArrElem;
}

static CURLcode
set_ssl_version_min_max(SCHANNEL_CRED *schannel_cred, struct Curl_easy *data,
                        struct connectdata *conn)
{
  long ssl_version = SSL_CONN_CONFIG(version);
  long ssl_version_max = SSL_CONN_CONFIG(version_max);
  long i = ssl_version;

  switch(ssl_version_max) {
  case CURL_SSLVERSION_MAX_NONE:
  case CURL_SSLVERSION_MAX_DEFAULT:
    ssl_version_max = CURL_SSLVERSION_MAX_TLSv1_2;
    break;
  }
  for(; i <= (ssl_version_max >> 16); ++i) {
    switch(i) {
    case CURL_SSLVERSION_TLSv1_0:
      schannel_cred->grbitEnabledProtocols |= SP_PROT_TLS1_0_CLIENT;
      break;
    case CURL_SSLVERSION_TLSv1_1:
      schannel_cred->grbitEnabledProtocols |= SP_PROT_TLS1_1_CLIENT;
      break;
    case CURL_SSLVERSION_TLSv1_2:
      schannel_cred->grbitEnabledProtocols |= SP_PROT_TLS1_2_CLIENT;
      break;
    case CURL_SSLVERSION_TLSv1_3:
      failf(data, "schannel: TLS 1.3 is not yet supported");
      return CURLE_SSL_CONNECT_ERROR;
    }
  }
  return CURLE_OK;
}

/*longest is 26, buffer is slightly bigger*/
#define LONGEST_ALG_ID 32
#define CIPHEROPTION(X)                         \
  if(strcmp(#X, tmp) == 0)                      \
    return X

static int
get_alg_id_by_name(char *name)
{
  char tmp[LONGEST_ALG_ID] = { 0 };
  char *nameEnd = strchr(name, ':');
  size_t n = nameEnd ? min((size_t)(nameEnd - name), LONGEST_ALG_ID - 1) : \
    min(strlen(name), LONGEST_ALG_ID - 1);
  strncpy(tmp, name, n);
  tmp[n] = 0;
  CIPHEROPTION(CALG_MD2);
  CIPHEROPTION(CALG_MD4);
  CIPHEROPTION(CALG_MD5);
  CIPHEROPTION(CALG_SHA);
  CIPHEROPTION(CALG_SHA1);
  CIPHEROPTION(CALG_MAC);
  CIPHEROPTION(CALG_RSA_SIGN);
  CIPHEROPTION(CALG_DSS_SIGN);
/*ifdefs for the options that are defined conditionally in wincrypt.h*/
#ifdef CALG_NO_SIGN
  CIPHEROPTION(CALG_NO_SIGN);
#endif
  CIPHEROPTION(CALG_RSA_KEYX);
  CIPHEROPTION(CALG_DES);
#ifdef CALG_3DES_112
  CIPHEROPTION(CALG_3DES_112);
#endif
  CIPHEROPTION(CALG_3DES);
  CIPHEROPTION(CALG_DESX);
  CIPHEROPTION(CALG_RC2);
  CIPHEROPTION(CALG_RC4);
  CIPHEROPTION(CALG_SEAL);
#ifdef CALG_DH_SF
  CIPHEROPTION(CALG_DH_SF);
#endif
  CIPHEROPTION(CALG_DH_EPHEM);
#ifdef CALG_AGREEDKEY_ANY
  CIPHEROPTION(CALG_AGREEDKEY_ANY);
#endif
#ifdef CALG_HUGHES_MD5
  CIPHEROPTION(CALG_HUGHES_MD5);
#endif
  CIPHEROPTION(CALG_SKIPJACK);
#ifdef CALG_TEK
  CIPHEROPTION(CALG_TEK);
#endif
  CIPHEROPTION(CALG_CYLINK_MEK);
  CIPHEROPTION(CALG_SSL3_SHAMD5);
#ifdef CALG_SSL3_MASTER
  CIPHEROPTION(CALG_SSL3_MASTER);
#endif
#ifdef CALG_SCHANNEL_MASTER_HASH
  CIPHEROPTION(CALG_SCHANNEL_MASTER_HASH);
#endif
#ifdef CALG_SCHANNEL_MAC_KEY
  CIPHEROPTION(CALG_SCHANNEL_MAC_KEY);
#endif
#ifdef CALG_SCHANNEL_ENC_KEY
  CIPHEROPTION(CALG_SCHANNEL_ENC_KEY);
#endif
#ifdef CALG_PCT1_MASTER
  CIPHEROPTION(CALG_PCT1_MASTER);
#endif
#ifdef CALG_SSL2_MASTER
  CIPHEROPTION(CALG_SSL2_MASTER);
#endif
#ifdef CALG_TLS1_MASTER
  CIPHEROPTION(CALG_TLS1_MASTER);
#endif
#ifdef CALG_RC5
  CIPHEROPTION(CALG_RC5);
#endif
#ifdef CALG_HMAC
  CIPHEROPTION(CALG_HMAC);
#endif
#ifdef CALG_TLS1PRF
  CIPHEROPTION(CALG_TLS1PRF);
#endif
#ifdef CALG_HASH_REPLACE_OWF
  CIPHEROPTION(CALG_HASH_REPLACE_OWF);
#endif
#ifdef CALG_AES_128
  CIPHEROPTION(CALG_AES_128);
#endif
#ifdef CALG_AES_192
  CIPHEROPTION(CALG_AES_192);
#endif
#ifdef CALG_AES_256
  CIPHEROPTION(CALG_AES_256);
#endif
#ifdef CALG_AES
  CIPHEROPTION(CALG_AES);
#endif
#ifdef CALG_SHA_256
  CIPHEROPTION(CALG_SHA_256);
#endif
#ifdef CALG_SHA_384
  CIPHEROPTION(CALG_SHA_384);
#endif
#ifdef CALG_SHA_512
  CIPHEROPTION(CALG_SHA_512);
#endif
#ifdef CALG_ECDH
  CIPHEROPTION(CALG_ECDH);
#endif
#ifdef CALG_ECMQV
  CIPHEROPTION(CALG_ECMQV);
#endif
#ifdef CALG_ECDSA
  CIPHEROPTION(CALG_ECDSA);
#endif
#ifdef CALG_ECDH_EPHEM
  CIPHEROPTION(CALG_ECDH_EPHEM);
#endif
  return 0;
}

static CURLcode
set_ssl_ciphers(SCHANNEL_CRED *schannel_cred, char *ciphers,
                ALG_ID *algIds)
{
  char *startCur = ciphers;
  int algCount = 0;
  while(startCur && (0 != *startCur) && (algCount < NUMOF_CIPHERS)) {
    long alg = strtol(startCur, 0, 0);
    if(!alg)
      alg = get_alg_id_by_name(startCur);
    if(alg)
      algIds[algCount++] = alg;
    else if(!strncmp(startCur, "USE_STRONG_CRYPTO",
                     sizeof("USE_STRONG_CRYPTO") - 1) ||
            !strncmp(startCur, "SCH_USE_STRONG_CRYPTO",
                     sizeof("SCH_USE_STRONG_CRYPTO") - 1))
      schannel_cred->dwFlags |= SCH_USE_STRONG_CRYPTO;
    else
      return CURLE_SSL_CIPHER;
    startCur = strchr(startCur, ':');
    if(startCur)
      startCur++;
  }
  schannel_cred->palgSupportedAlgs = algIds;
  schannel_cred->cSupportedAlgs = algCount;
  return CURLE_OK;
}

#ifdef HAS_CLIENT_CERT_PATH

/* Function allocates memory for store_path only if CURLE_OK is returned */
static CURLcode
get_cert_location(TCHAR *path, DWORD *store_name, TCHAR **store_path,
                  TCHAR **thumbprint)
{
  TCHAR *sep;
  TCHAR *store_path_start;
  size_t store_name_len;

  sep = _tcschr(path, TEXT('\\'));
  if(!sep)
    return CURLE_SSL_CERTPROBLEM;

  store_name_len = sep - path;

  if(_tcsncmp(path, TEXT("CurrentUser"), store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_CURRENT_USER;
  else if(_tcsncmp(path, TEXT("LocalMachine"), store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_LOCAL_MACHINE;
  else if(_tcsncmp(path, TEXT("CurrentService"), store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_CURRENT_SERVICE;
  else if(_tcsncmp(path, TEXT("Services"), store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_SERVICES;
  else if(_tcsncmp(path, TEXT("Users"), store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_USERS;
  else if(_tcsncmp(path, TEXT("CurrentUserGroupPolicy"),
                    store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_CURRENT_USER_GROUP_POLICY;
  else if(_tcsncmp(path, TEXT("LocalMachineGroupPolicy"),
                    store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_LOCAL_MACHINE_GROUP_POLICY;
  else if(_tcsncmp(path, TEXT("LocalMachineEnterprise"),
                    store_name_len) == 0)
    *store_name = CERT_SYSTEM_STORE_LOCAL_MACHINE_ENTERPRISE;
  else
    return CURLE_SSL_CERTPROBLEM;

  store_path_start = sep + 1;

  sep = _tcschr(store_path_start, TEXT('\\'));
  if(!sep)
    return CURLE_SSL_CERTPROBLEM;

  *thumbprint = sep + 1;
  if(_tcslen(*thumbprint) != CERT_THUMBPRINT_STR_LEN)
    return CURLE_SSL_CERTPROBLEM;

  *sep = TEXT('\0');
  *store_path = _tcsdup(store_path_start);
  *sep = TEXT('\\');
  if(!*store_path)
    return CURLE_OUT_OF_MEMORY;

  return CURLE_OK;
}
#endif
static CURLcode
schannel_acquire_credential_handle(struct Curl_easy *data,
                                   struct connectdata *conn,
                                   int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  SCHANNEL_CRED schannel_cred;
  PCCERT_CONTEXT client_certs[1] = { NULL };
  SECURITY_STATUS sspi_status = SEC_E_OK;
  CURLcode result;

  /* setup Schannel API options */
  memset(&schannel_cred, 0, sizeof(schannel_cred));
  schannel_cred.dwVersion = SCHANNEL_CRED_VERSION;

  if(conn->ssl_config.verifypeer) {
#ifdef HAS_MANUAL_VERIFY_API
    if(BACKEND->use_manual_cred_validation)
      schannel_cred.dwFlags = SCH_CRED_MANUAL_CRED_VALIDATION;
    else
#endif
      schannel_cred.dwFlags = SCH_CRED_AUTO_CRED_VALIDATION;

    if(SSL_SET_OPTION(no_revoke)) {
      schannel_cred.dwFlags |= SCH_CRED_IGNORE_NO_REVOCATION_CHECK |
        SCH_CRED_IGNORE_REVOCATION_OFFLINE;

      DEBUGF(infof(data, "schannel: disabled server certificate revocation "
                   "checks"));
    }
    else if(SSL_SET_OPTION(revoke_best_effort)) {
      schannel_cred.dwFlags |= SCH_CRED_IGNORE_NO_REVOCATION_CHECK |
        SCH_CRED_IGNORE_REVOCATION_OFFLINE | SCH_CRED_REVOCATION_CHECK_CHAIN;

      DEBUGF(infof(data, "schannel: ignore revocation offline errors"));
    }
    else {
      schannel_cred.dwFlags |= SCH_CRED_REVOCATION_CHECK_CHAIN;

      DEBUGF(infof(data,
                   "schannel: checking server certificate revocation"));
    }
  }
  else {
    schannel_cred.dwFlags = SCH_CRED_MANUAL_CRED_VALIDATION |
      SCH_CRED_IGNORE_NO_REVOCATION_CHECK |
      SCH_CRED_IGNORE_REVOCATION_OFFLINE;
    DEBUGF(infof(data,
                 "schannel: disabled server cert revocation checks"));
  }

  if(!conn->ssl_config.verifyhost) {
    schannel_cred.dwFlags |= SCH_CRED_NO_SERVERNAME_CHECK;
    DEBUGF(infof(data, "schannel: verifyhost setting prevents Schannel from "
                 "comparing the supplied target name with the subject "
                 "names in server certificates."));
  }

  if(!SSL_SET_OPTION(auto_client_cert)) {
    schannel_cred.dwFlags &= ~SCH_CRED_USE_DEFAULT_CREDS;
    schannel_cred.dwFlags |= SCH_CRED_NO_DEFAULT_CREDS;
    infof(data, "schannel: disabled automatic use of client certificate");
  }
  else
    infof(data, "schannel: enabled automatic use of client certificate");

  switch(conn->ssl_config.version) {
  case CURL_SSLVERSION_DEFAULT:
  case CURL_SSLVERSION_TLSv1:
  case CURL_SSLVERSION_TLSv1_0:
  case CURL_SSLVERSION_TLSv1_1:
  case CURL_SSLVERSION_TLSv1_2:
  case CURL_SSLVERSION_TLSv1_3:
  {
    result = set_ssl_version_min_max(&schannel_cred, data, conn);
    if(result != CURLE_OK)
      return result;
    break;
  }
  case CURL_SSLVERSION_SSLv3:
  case CURL_SSLVERSION_SSLv2:
    failf(data, "SSL versions not supported");
    return CURLE_NOT_BUILT_IN;
  default:
    failf(data, "Unrecognized parameter passed via CURLOPT_SSLVERSION");
    return CURLE_SSL_CONNECT_ERROR;
  }

  if(SSL_CONN_CONFIG(cipher_list)) {
    result = set_ssl_ciphers(&schannel_cred, SSL_CONN_CONFIG(cipher_list),
                             BACKEND->algIds);
    if(CURLE_OK != result) {
      failf(data, "Unable to set ciphers to passed via SSL_CONN_CONFIG");
      return result;
    }
  }


#ifdef HAS_CLIENT_CERT_PATH
  /* client certificate */
  if(data->set.ssl.primary.clientcert || data->set.ssl.primary.cert_blob) {
    DWORD cert_store_name = 0;
    TCHAR *cert_store_path = NULL;
    TCHAR *cert_thumbprint_str = NULL;
    CRYPT_HASH_BLOB cert_thumbprint;
    BYTE cert_thumbprint_data[CERT_THUMBPRINT_DATA_LEN];
    HCERTSTORE cert_store = NULL;
    FILE *fInCert = NULL;
    void *certdata = NULL;
    size_t certsize = 0;
    bool blob = data->set.ssl.primary.cert_blob != NULL;
    TCHAR *cert_path = NULL;
    if(blob) {
      certdata = data->set.ssl.primary.cert_blob->data;
      certsize = data->set.ssl.primary.cert_blob->len;
    }
    else {
      cert_path = curlx_convert_UTF8_to_tchar(
        data->set.ssl.primary.clientcert);
      if(!cert_path)
        return CURLE_OUT_OF_MEMORY;

      result = get_cert_location(cert_path, &cert_store_name,
        &cert_store_path, &cert_thumbprint_str);

      if(result && (data->set.ssl.primary.clientcert[0]!='\0'))
        fInCert = fopen(data->set.ssl.primary.clientcert, "rb");

      if(result && !fInCert) {
        failf(data, "schannel: Failed to get certificate location"
              " or file for %s",
              data->set.ssl.primary.clientcert);
        curlx_unicodefree(cert_path);
        return result;
      }
    }

    if((fInCert || blob) && (data->set.ssl.cert_type) &&
        (!strcasecompare(data->set.ssl.cert_type, "P12"))) {
      failf(data, "schannel: certificate format compatibility error "
              " for %s",
              blob ? "(memory blob)" : data->set.ssl.primary.clientcert);
      curlx_unicodefree(cert_path);
      return CURLE_SSL_CERTPROBLEM;
    }

    if(fInCert || blob) {
      /* Reading a .P12 or .pfx file, like the example at bottom of
           https://social.msdn.microsoft.com/Forums/windowsdesktop/
                          en-US/3e7bc95f-b21a-4bcd-bd2c-7f996718cae5
      */
      CRYPT_DATA_BLOB datablob;
      WCHAR* pszPassword;
      size_t pwd_len = 0;
      int str_w_len = 0;
      const char *cert_showfilename_error = blob ?
        "(memory blob)" : data->set.ssl.primary.clientcert;
      curlx_unicodefree(cert_path);
      if(fInCert) {
        long cert_tell = 0;
        bool continue_reading = fseek(fInCert, 0, SEEK_END) == 0;
        if(continue_reading)
          cert_tell = ftell(fInCert);
        if(cert_tell < 0)
          continue_reading = FALSE;
        else
          certsize = (size_t)cert_tell;
        if(continue_reading)
          continue_reading = fseek(fInCert, 0, SEEK_SET) == 0;
        if(continue_reading)
          certdata = malloc(certsize + 1);
        if((!certdata) ||
           ((int) fread(certdata, certsize, 1, fInCert) != 1))
          continue_reading = FALSE;
        fclose(fInCert);
        if(!continue_reading) {
          failf(data, "schannel: Failed to read cert file %s",
              data->set.ssl.primary.clientcert);
          free(certdata);
          return CURLE_SSL_CERTPROBLEM;
        }
      }

      /* Convert key-pair data to the in-memory certificate store */
      datablob.pbData = (BYTE*)certdata;
      datablob.cbData = (DWORD)certsize;

      if(data->set.ssl.key_passwd != NULL)
        pwd_len = strlen(data->set.ssl.key_passwd);
      pszPassword = (WCHAR*)malloc(sizeof(WCHAR)*(pwd_len + 1));
      if(pszPassword) {
        if(pwd_len > 0)
          str_w_len = MultiByteToWideChar(CP_UTF8,
             MB_ERR_INVALID_CHARS,
             data->set.ssl.key_passwd, (int)pwd_len,
             pszPassword, (int)(pwd_len + 1));

        if((str_w_len >= 0) && (str_w_len <= (int)pwd_len))
          pszPassword[str_w_len] = 0;
        else
          pszPassword[0] = 0;

        cert_store = PFXImportCertStore(&datablob, pszPassword, 0);
        free(pszPassword);
      }
      if(!blob)
        free(certdata);
      if(!cert_store) {
        DWORD errorcode = GetLastError();
        if(errorcode == ERROR_INVALID_PASSWORD)
          failf(data, "schannel: Failed to import cert file %s, "
                "password is bad",
                cert_showfilename_error);
        else
          failf(data, "schannel: Failed to import cert file %s, "
                "last error is 0x%x",
                cert_showfilename_error, errorcode);
        return CURLE_SSL_CERTPROBLEM;
      }

      client_certs[0] = CertFindCertificateInStore(
        cert_store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
        CERT_FIND_ANY, NULL, NULL);

      if(!client_certs[0]) {
        failf(data, "schannel: Failed to get certificate from file %s"
              ", last error is 0x%x",
              cert_showfilename_error, GetLastError());
        CertCloseStore(cert_store, 0);
        return CURLE_SSL_CERTPROBLEM;
      }

      schannel_cred.cCreds = 1;
      schannel_cred.paCred = client_certs;
    }
    else {
      cert_store =
        CertOpenStore(CURL_CERT_STORE_PROV_SYSTEM, 0,
                      (HCRYPTPROV)NULL,
                      CERT_STORE_OPEN_EXISTING_FLAG | cert_store_name,
                      cert_store_path);
      if(!cert_store) {
        failf(data, "schannel: Failed to open cert store %x %s, "
              "last error is 0x%x",
              cert_store_name, cert_store_path, GetLastError());
        free(cert_store_path);
        curlx_unicodefree(cert_path);
        return CURLE_SSL_CERTPROBLEM;
      }
      free(cert_store_path);

      cert_thumbprint.pbData = cert_thumbprint_data;
      cert_thumbprint.cbData = CERT_THUMBPRINT_DATA_LEN;

      if(!CryptStringToBinary(cert_thumbprint_str,
                              CERT_THUMBPRINT_STR_LEN,
                              CRYPT_STRING_HEX,
                              cert_thumbprint_data,
                              &cert_thumbprint.cbData,
                              NULL, NULL)) {
        curlx_unicodefree(cert_path);
        CertCloseStore(cert_store, 0);
        return CURLE_SSL_CERTPROBLEM;
      }

      client_certs[0] = CertFindCertificateInStore(
        cert_store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
        CERT_FIND_HASH, &cert_thumbprint, NULL);

      curlx_unicodefree(cert_path);

      if(client_certs[0]) {
        schannel_cred.cCreds = 1;
        schannel_cred.paCred = client_certs;
      }
      else {
        /* CRYPT_E_NOT_FOUND / E_INVALIDARG */
        CertCloseStore(cert_store, 0);
        return CURLE_SSL_CERTPROBLEM;
      }
    }
    CertCloseStore(cert_store, 0);
  }
#else
  if(data->set.ssl.primary.clientcert || data->set.ssl.primary.cert_blob) {
    failf(data, "schannel: client cert support not built in");
    return CURLE_NOT_BUILT_IN;
  }
#endif

  /* allocate memory for the re-usable credential handle */
  BACKEND->cred = (struct Curl_schannel_cred *)
    calloc(1, sizeof(struct Curl_schannel_cred));
  if(!BACKEND->cred) {
    failf(data, "schannel: unable to allocate memory");

    if(client_certs[0])
      CertFreeCertificateContext(client_certs[0]);

    return CURLE_OUT_OF_MEMORY;
  }
  BACKEND->cred->refcount = 1;

  sspi_status =
    s_pSecFn->AcquireCredentialsHandle(NULL, (TCHAR *)UNISP_NAME,
                                       SECPKG_CRED_OUTBOUND, NULL,
                                       &schannel_cred, NULL, NULL,
                                       &BACKEND->cred->cred_handle,
                                       &BACKEND->cred->time_stamp);

  if(client_certs[0])
    CertFreeCertificateContext(client_certs[0]);

  if(sspi_status != SEC_E_OK) {
    char buffer[STRERROR_LEN];
    failf(data, "schannel: AcquireCredentialsHandle failed: %s",
          Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
    Curl_safefree(BACKEND->cred);
    switch(sspi_status) {
    case SEC_E_INSUFFICIENT_MEMORY:
      return CURLE_OUT_OF_MEMORY;
    case SEC_E_NO_CREDENTIALS:
    case SEC_E_SECPKG_NOT_FOUND:
    case SEC_E_NOT_OWNER:
    case SEC_E_UNKNOWN_CREDENTIALS:
    case SEC_E_INTERNAL_ERROR:
    default:
      return CURLE_SSL_CONNECT_ERROR;
    }
  }

  return CURLE_OK;
}

static CURLcode
schannel_connect_step1(struct Curl_easy *data, struct connectdata *conn,
                       int sockindex)
{
  ssize_t written = -1;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  SecBuffer outbuf;
  SecBufferDesc outbuf_desc;
  SecBuffer inbuf;
  SecBufferDesc inbuf_desc;
#ifdef HAS_ALPN
  unsigned char alpn_buffer[128];
#endif
  SECURITY_STATUS sspi_status = SEC_E_OK;
  struct Curl_schannel_cred *old_cred = NULL;
  struct in_addr addr;
#ifdef ENABLE_IPV6
  struct in6_addr addr6;
#endif
  TCHAR *host_name;
  CURLcode result;
  char * const hostname = SSL_HOST_NAME();

  DEBUGF(infof(data,
               "schannel: SSL/TLS connection with %s port %hu (step 1/3)",
               hostname, conn->remote_port));

  if(curlx_verify_windows_version(5, 1, 0, PLATFORM_WINNT,
                                  VERSION_LESS_THAN_EQUAL)) {
    /* Schannel in Windows XP (OS version 5.1) uses legacy handshakes and
       algorithms that may not be supported by all servers. */
    infof(data, "schannel: Windows version is old and may not be able to "
          "connect to some servers due to lack of SNI, algorithms, etc.");
  }

#ifdef HAS_ALPN
  /* ALPN is only supported on Windows 8.1 / Server 2012 R2 and above.
     Also it doesn't seem to be supported for Wine, see curl bug #983. */
  BACKEND->use_alpn = conn->bits.tls_enable_alpn &&
    !GetProcAddress(GetModuleHandle(TEXT("ntdll")),
                    "wine_get_version") &&
    curlx_verify_windows_version(6, 3, 0, PLATFORM_WINNT,
                                 VERSION_GREATER_THAN_EQUAL);
#else
  BACKEND->use_alpn = false;
#endif

#ifdef _WIN32_WCE
#ifdef HAS_MANUAL_VERIFY_API
  /* certificate validation on CE doesn't seem to work right; we'll
   * do it following a more manual process. */
  BACKEND->use_manual_cred_validation = true;
#else
#error "compiler too old to support requisite manual cert verify for Win CE"
#endif
#else
#ifdef HAS_MANUAL_VERIFY_API
  if(SSL_CONN_CONFIG(CAfile) || SSL_CONN_CONFIG(ca_info_blob)) {
    if(curlx_verify_windows_version(6, 1, 0, PLATFORM_WINNT,
                                    VERSION_GREATER_THAN_EQUAL)) {
      BACKEND->use_manual_cred_validation = true;
    }
    else {
      failf(data, "schannel: this version of Windows is too old to support "
            "certificate verification via CA bundle file.");
      return CURLE_SSL_CACERT_BADFILE;
    }
  }
  else
    BACKEND->use_manual_cred_validation = false;
#else
  if(SSL_CONN_CONFIG(CAfile) || SSL_CONN_CONFIG(ca_info_blob)) {
    failf(data, "schannel: CA cert support not built in");
    return CURLE_NOT_BUILT_IN;
  }
#endif
#endif

  BACKEND->cred = NULL;

  /* check for an existing re-usable credential handle */
  if(SSL_SET_OPTION(primary.sessionid)) {
    Curl_ssl_sessionid_lock(data);
    if(!Curl_ssl_getsessionid(data, conn,
                              SSL_IS_PROXY() ? TRUE : FALSE,
                              (void **)&old_cred, NULL, sockindex)) {
      BACKEND->cred = old_cred;
      DEBUGF(infof(data, "schannel: re-using existing credential handle"));

      /* increment the reference counter of the credential/session handle */
      BACKEND->cred->refcount++;
      DEBUGF(infof(data,
                   "schannel: incremented credential handle refcount = %d",
                   BACKEND->cred->refcount));
    }
    Curl_ssl_sessionid_unlock(data);
  }

  if(!BACKEND->cred) {
    result = schannel_acquire_credential_handle(data, conn, sockindex);
    if(result != CURLE_OK) {
      return result;
    }
  }

  /* Warn if SNI is disabled due to use of an IP address */
  if(Curl_inet_pton(AF_INET, hostname, &addr)
#ifdef ENABLE_IPV6
     || Curl_inet_pton(AF_INET6, hostname, &addr6)
#endif
    ) {
    infof(data, "schannel: using IP address, SNI is not supported by OS.");
  }

#ifdef HAS_ALPN
  if(BACKEND->use_alpn) {
    int cur = 0;
    int list_start_index = 0;
    unsigned int *extension_len = NULL;
    unsigned short* list_len = NULL;

    /* The first four bytes will be an unsigned int indicating number
       of bytes of data in the rest of the buffer. */
    extension_len = (unsigned int *)(&alpn_buffer[cur]);
    cur += sizeof(unsigned int);

    /* The next four bytes are an indicator that this buffer will contain
       ALPN data, as opposed to NPN, for example. */
    *(unsigned int *)&alpn_buffer[cur] =
      SecApplicationProtocolNegotiationExt_ALPN;
    cur += sizeof(unsigned int);

    /* The next two bytes will be an unsigned short indicating the number
       of bytes used to list the preferred protocols. */
    list_len = (unsigned short*)(&alpn_buffer[cur]);
    cur += sizeof(unsigned short);

    list_start_index = cur;

#ifdef USE_HTTP2
    if(data->state.httpwant >= CURL_HTTP_VERSION_2) {
      alpn_buffer[cur++] = ALPN_H2_LENGTH;
      memcpy(&alpn_buffer[cur], ALPN_H2, ALPN_H2_LENGTH);
      cur += ALPN_H2_LENGTH;
      infof(data, "schannel: ALPN, offering %s", ALPN_H2);
    }
#endif

    alpn_buffer[cur++] = ALPN_HTTP_1_1_LENGTH;
    memcpy(&alpn_buffer[cur], ALPN_HTTP_1_1, ALPN_HTTP_1_1_LENGTH);
    cur += ALPN_HTTP_1_1_LENGTH;
    infof(data, "schannel: ALPN, offering %s", ALPN_HTTP_1_1);

    *list_len = curlx_uitous(cur - list_start_index);
    *extension_len = *list_len + sizeof(unsigned int) + sizeof(unsigned short);

    InitSecBuffer(&inbuf, SECBUFFER_APPLICATION_PROTOCOLS, alpn_buffer, cur);
    InitSecBufferDesc(&inbuf_desc, &inbuf, 1);
  }
  else {
    InitSecBuffer(&inbuf, SECBUFFER_EMPTY, NULL, 0);
    InitSecBufferDesc(&inbuf_desc, &inbuf, 1);
  }
#else /* HAS_ALPN */
  InitSecBuffer(&inbuf, SECBUFFER_EMPTY, NULL, 0);
  InitSecBufferDesc(&inbuf_desc, &inbuf, 1);
#endif

  /* setup output buffer */
  InitSecBuffer(&outbuf, SECBUFFER_EMPTY, NULL, 0);
  InitSecBufferDesc(&outbuf_desc, &outbuf, 1);

  /* security request flags */
  BACKEND->req_flags = ISC_REQ_SEQUENCE_DETECT | ISC_REQ_REPLAY_DETECT |
    ISC_REQ_CONFIDENTIALITY | ISC_REQ_ALLOCATE_MEMORY |
    ISC_REQ_STREAM;

  if(!SSL_SET_OPTION(auto_client_cert)) {
    BACKEND->req_flags |= ISC_REQ_USE_SUPPLIED_CREDS;
  }

  /* allocate memory for the security context handle */
  BACKEND->ctxt = (struct Curl_schannel_ctxt *)
    calloc(1, sizeof(struct Curl_schannel_ctxt));
  if(!BACKEND->ctxt) {
    failf(data, "schannel: unable to allocate memory");
    return CURLE_OUT_OF_MEMORY;
  }

  host_name = curlx_convert_UTF8_to_tchar(hostname);
  if(!host_name)
    return CURLE_OUT_OF_MEMORY;

  /* Schannel InitializeSecurityContext:
     https://msdn.microsoft.com/en-us/library/windows/desktop/aa375924.aspx

     At the moment we don't pass inbuf unless we're using ALPN since we only
     use it for that, and Wine (for which we currently disable ALPN) is giving
     us problems with inbuf regardless. https://github.com/curl/curl/issues/983
  */
  sspi_status = s_pSecFn->InitializeSecurityContext(
    &BACKEND->cred->cred_handle, NULL, host_name, BACKEND->req_flags, 0, 0,
    (BACKEND->use_alpn ? &inbuf_desc : NULL),
    0, &BACKEND->ctxt->ctxt_handle,
    &outbuf_desc, &BACKEND->ret_flags, &BACKEND->ctxt->time_stamp);

  curlx_unicodefree(host_name);

  if(sspi_status != SEC_I_CONTINUE_NEEDED) {
    char buffer[STRERROR_LEN];
    Curl_safefree(BACKEND->ctxt);
    switch(sspi_status) {
    case SEC_E_INSUFFICIENT_MEMORY:
      failf(data, "schannel: initial InitializeSecurityContext failed: %s",
            Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
      return CURLE_OUT_OF_MEMORY;
    case SEC_E_WRONG_PRINCIPAL:
      failf(data, "schannel: SNI or certificate check failed: %s",
            Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
      return CURLE_PEER_FAILED_VERIFICATION;
      /*
        case SEC_E_INVALID_HANDLE:
        case SEC_E_INVALID_TOKEN:
        case SEC_E_LOGON_DENIED:
        case SEC_E_TARGET_UNKNOWN:
        case SEC_E_NO_AUTHENTICATING_AUTHORITY:
        case SEC_E_INTERNAL_ERROR:
        case SEC_E_NO_CREDENTIALS:
        case SEC_E_UNSUPPORTED_FUNCTION:
        case SEC_E_APPLICATION_PROTOCOL_MISMATCH:
      */
    default:
      failf(data, "schannel: initial InitializeSecurityContext failed: %s",
            Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
      return CURLE_SSL_CONNECT_ERROR;
    }
  }

  DEBUGF(infof(data, "schannel: sending initial handshake data: "
               "sending %lu bytes.", outbuf.cbBuffer));

  /* send initial handshake data which is now stored in output buffer */
  result = Curl_write_plain(data, conn->sock[sockindex], outbuf.pvBuffer,
                            outbuf.cbBuffer, &written);
  s_pSecFn->FreeContextBuffer(outbuf.pvBuffer);
  if((result != CURLE_OK) || (outbuf.cbBuffer != (size_t) written)) {
    failf(data, "schannel: failed to send initial handshake data: "
          "sent %zd of %lu bytes", written, outbuf.cbBuffer);
    return CURLE_SSL_CONNECT_ERROR;
  }

  DEBUGF(infof(data, "schannel: sent initial handshake data: "
               "sent %zd bytes", written));

  BACKEND->recv_unrecoverable_err = CURLE_OK;
  BACKEND->recv_sspi_close_notify = false;
  BACKEND->recv_connection_closed = false;
  BACKEND->encdata_is_incomplete = false;

  /* continue to second handshake step */
  connssl->connecting_state = ssl_connect_2;

  return CURLE_OK;
}

static CURLcode
schannel_connect_step2(struct Curl_easy *data, struct connectdata *conn,
                       int sockindex)
{
  int i;
  ssize_t nread = -1, written = -1;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  unsigned char *reallocated_buffer;
  SecBuffer outbuf[3];
  SecBufferDesc outbuf_desc;
  SecBuffer inbuf[2];
  SecBufferDesc inbuf_desc;
  SECURITY_STATUS sspi_status = SEC_E_OK;
  CURLcode result;
  bool doread;
  char * const hostname = SSL_HOST_NAME();
  const char *pubkey_ptr;

  doread = (connssl->connecting_state != ssl_connect_2_writing) ? TRUE : FALSE;

  DEBUGF(infof(data,
               "schannel: SSL/TLS connection with %s port %hu (step 2/3)",
               hostname, conn->remote_port));

  if(!BACKEND->cred || !BACKEND->ctxt)
    return CURLE_SSL_CONNECT_ERROR;

  /* buffer to store previously received and decrypted data */
  if(!BACKEND->decdata_buffer) {
    BACKEND->decdata_offset = 0;
    BACKEND->decdata_length = CURL_SCHANNEL_BUFFER_INIT_SIZE;
    BACKEND->decdata_buffer = malloc(BACKEND->decdata_length);
    if(!BACKEND->decdata_buffer) {
      failf(data, "schannel: unable to allocate memory");
      return CURLE_OUT_OF_MEMORY;
    }
  }

  /* buffer to store previously received and encrypted data */
  if(!BACKEND->encdata_buffer) {
    BACKEND->encdata_is_incomplete = false;
    BACKEND->encdata_offset = 0;
    BACKEND->encdata_length = CURL_SCHANNEL_BUFFER_INIT_SIZE;
    BACKEND->encdata_buffer = malloc(BACKEND->encdata_length);
    if(!BACKEND->encdata_buffer) {
      failf(data, "schannel: unable to allocate memory");
      return CURLE_OUT_OF_MEMORY;
    }
  }

  /* if we need a bigger buffer to read a full message, increase buffer now */
  if(BACKEND->encdata_length - BACKEND->encdata_offset <
     CURL_SCHANNEL_BUFFER_FREE_SIZE) {
    /* increase internal encrypted data buffer */
    size_t reallocated_length = BACKEND->encdata_offset +
      CURL_SCHANNEL_BUFFER_FREE_SIZE;
    reallocated_buffer = realloc(BACKEND->encdata_buffer,
                                 reallocated_length);

    if(!reallocated_buffer) {
      failf(data, "schannel: unable to re-allocate memory");
      return CURLE_OUT_OF_MEMORY;
    }
    else {
      BACKEND->encdata_buffer = reallocated_buffer;
      BACKEND->encdata_length = reallocated_length;
    }
  }

  for(;;) {
    TCHAR *host_name;
    if(doread) {
      /* read encrypted handshake data from socket */
      result = Curl_read_plain(conn->sock[sockindex],
                               (char *) (BACKEND->encdata_buffer +
                                         BACKEND->encdata_offset),
                               BACKEND->encdata_length -
                               BACKEND->encdata_offset,
                               &nread);
      if(result == CURLE_AGAIN) {
        if(connssl->connecting_state != ssl_connect_2_writing)
          connssl->connecting_state = ssl_connect_2_reading;
        DEBUGF(infof(data, "schannel: failed to receive handshake, "
                     "need more data"));
        return CURLE_OK;
      }
      else if((result != CURLE_OK) || (nread == 0)) {
        failf(data, "schannel: failed to receive handshake, "
              "SSL/TLS connection failed");
        return CURLE_SSL_CONNECT_ERROR;
      }

      /* increase encrypted data buffer offset */
      BACKEND->encdata_offset += nread;
      BACKEND->encdata_is_incomplete = false;
      DEBUGF(infof(data, "schannel: encrypted data got %zd", nread));
    }

    DEBUGF(infof(data,
                 "schannel: encrypted data buffer: offset %zu length %zu",
                 BACKEND->encdata_offset, BACKEND->encdata_length));

    /* setup input buffers */
    InitSecBuffer(&inbuf[0], SECBUFFER_TOKEN, malloc(BACKEND->encdata_offset),
                  curlx_uztoul(BACKEND->encdata_offset));
    InitSecBuffer(&inbuf[1], SECBUFFER_EMPTY, NULL, 0);
    InitSecBufferDesc(&inbuf_desc, inbuf, 2);

    /* setup output buffers */
    InitSecBuffer(&outbuf[0], SECBUFFER_TOKEN, NULL, 0);
    InitSecBuffer(&outbuf[1], SECBUFFER_ALERT, NULL, 0);
    InitSecBuffer(&outbuf[2], SECBUFFER_EMPTY, NULL, 0);
    InitSecBufferDesc(&outbuf_desc, outbuf, 3);

    if(!inbuf[0].pvBuffer) {
      failf(data, "schannel: unable to allocate memory");
      return CURLE_OUT_OF_MEMORY;
    }

    /* copy received handshake data into input buffer */
    memcpy(inbuf[0].pvBuffer, BACKEND->encdata_buffer,
           BACKEND->encdata_offset);

    host_name = curlx_convert_UTF8_to_tchar(hostname);
    if(!host_name)
      return CURLE_OUT_OF_MEMORY;

    sspi_status = s_pSecFn->InitializeSecurityContext(
      &BACKEND->cred->cred_handle, &BACKEND->ctxt->ctxt_handle,
      host_name, BACKEND->req_flags, 0, 0, &inbuf_desc, 0, NULL,
      &outbuf_desc, &BACKEND->ret_flags, &BACKEND->ctxt->time_stamp);

    curlx_unicodefree(host_name);

    /* free buffer for received handshake data */
    Curl_safefree(inbuf[0].pvBuffer);

    /* check if the handshake was incomplete */
    if(sspi_status == SEC_E_INCOMPLETE_MESSAGE) {
      BACKEND->encdata_is_incomplete = true;
      connssl->connecting_state = ssl_connect_2_reading;
      DEBUGF(infof(data,
                   "schannel: received incomplete message, need more data"));
      return CURLE_OK;
    }

    /* If the server has requested a client certificate, attempt to continue
       the handshake without one. This will allow connections to servers which
       request a client certificate but do not require it. */
    if(sspi_status == SEC_I_INCOMPLETE_CREDENTIALS &&
       !(BACKEND->req_flags & ISC_REQ_USE_SUPPLIED_CREDS)) {
      BACKEND->req_flags |= ISC_REQ_USE_SUPPLIED_CREDS;
      connssl->connecting_state = ssl_connect_2_writing;
      DEBUGF(infof(data,
                   "schannel: a client certificate has been requested"));
      return CURLE_OK;
    }

    /* check if the handshake needs to be continued */
    if(sspi_status == SEC_I_CONTINUE_NEEDED || sspi_status == SEC_E_OK) {
      for(i = 0; i < 3; i++) {
        /* search for handshake tokens that need to be send */
        if(outbuf[i].BufferType == SECBUFFER_TOKEN && outbuf[i].cbBuffer > 0) {
          DEBUGF(infof(data, "schannel: sending next handshake data: "
                       "sending %lu bytes.", outbuf[i].cbBuffer));

          /* send handshake token to server */
          result = Curl_write_plain(data, conn->sock[sockindex],
                                    outbuf[i].pvBuffer, outbuf[i].cbBuffer,
                                    &written);
          if((result != CURLE_OK) ||
             (outbuf[i].cbBuffer != (size_t) written)) {
            failf(data, "schannel: failed to send next handshake data: "
                  "sent %zd of %lu bytes", written, outbuf[i].cbBuffer);
            return CURLE_SSL_CONNECT_ERROR;
          }
        }

        /* free obsolete buffer */
        if(outbuf[i].pvBuffer != NULL) {
          s_pSecFn->FreeContextBuffer(outbuf[i].pvBuffer);
        }
      }
    }
    else {
      char buffer[STRERROR_LEN];
      switch(sspi_status) {
      case SEC_E_INSUFFICIENT_MEMORY:
        failf(data, "schannel: next InitializeSecurityContext failed: %s",
              Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
        return CURLE_OUT_OF_MEMORY;
      case SEC_E_WRONG_PRINCIPAL:
        failf(data, "schannel: SNI or certificate check failed: %s",
              Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
        return CURLE_PEER_FAILED_VERIFICATION;
      case SEC_E_UNTRUSTED_ROOT:
        failf(data, "schannel: %s",
              Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
        return CURLE_PEER_FAILED_VERIFICATION;
        /*
          case SEC_E_INVALID_HANDLE:
          case SEC_E_INVALID_TOKEN:
          case SEC_E_LOGON_DENIED:
          case SEC_E_TARGET_UNKNOWN:
          case SEC_E_NO_AUTHENTICATING_AUTHORITY:
          case SEC_E_INTERNAL_ERROR:
          case SEC_E_NO_CREDENTIALS:
          case SEC_E_UNSUPPORTED_FUNCTION:
          case SEC_E_APPLICATION_PROTOCOL_MISMATCH:
        */
      default:
        failf(data, "schannel: next InitializeSecurityContext failed: %s",
              Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
        return CURLE_SSL_CONNECT_ERROR;
      }
    }

    /* check if there was additional remaining encrypted data */
    if(inbuf[1].BufferType == SECBUFFER_EXTRA && inbuf[1].cbBuffer > 0) {
      DEBUGF(infof(data, "schannel: encrypted data length: %lu",
                   inbuf[1].cbBuffer));
      /*
        There are two cases where we could be getting extra data here:
        1) If we're renegotiating a connection and the handshake is already
        complete (from the server perspective), it can encrypted app data
        (not handshake data) in an extra buffer at this point.
        2) (sspi_status == SEC_I_CONTINUE_NEEDED) We are negotiating a
        connection and this extra data is part of the handshake.
        We should process the data immediately; waiting for the socket to
        be ready may fail since the server is done sending handshake data.
      */
      /* check if the remaining data is less than the total amount
         and therefore begins after the already processed data */
      if(BACKEND->encdata_offset > inbuf[1].cbBuffer) {
        memmove(BACKEND->encdata_buffer,
                (BACKEND->encdata_buffer + BACKEND->encdata_offset) -
                inbuf[1].cbBuffer, inbuf[1].cbBuffer);
        BACKEND->encdata_offset = inbuf[1].cbBuffer;
        if(sspi_status == SEC_I_CONTINUE_NEEDED) {
          doread = FALSE;
          continue;
        }
      }
    }
    else {
      BACKEND->encdata_offset = 0;
    }
    break;
  }

  /* check if the handshake needs to be continued */
  if(sspi_status == SEC_I_CONTINUE_NEEDED) {
    connssl->connecting_state = ssl_connect_2_reading;
    return CURLE_OK;
  }

  /* check if the handshake is complete */
  if(sspi_status == SEC_E_OK) {
    connssl->connecting_state = ssl_connect_3;
    DEBUGF(infof(data, "schannel: SSL/TLS handshake complete"));
  }

  pubkey_ptr = SSL_PINNED_PUB_KEY();
  if(pubkey_ptr) {
    result = pkp_pin_peer_pubkey(data, conn, sockindex, pubkey_ptr);
    if(result) {
      failf(data, "SSL: public key does not match pinned public key!");
      return result;
    }
  }

#ifdef HAS_MANUAL_VERIFY_API
  if(conn->ssl_config.verifypeer && BACKEND->use_manual_cred_validation) {
    return Curl_verify_certificate(data, conn, sockindex);
  }
#endif

  return CURLE_OK;
}

static bool
valid_cert_encoding(const CERT_CONTEXT *cert_context)
{
  return (cert_context != NULL) &&
    ((cert_context->dwCertEncodingType & X509_ASN_ENCODING) != 0) &&
    (cert_context->pbCertEncoded != NULL) &&
    (cert_context->cbCertEncoded > 0);
}

typedef bool(*Read_crt_func)(const CERT_CONTEXT *ccert_context, void *arg);

static void
traverse_cert_store(const CERT_CONTEXT *context, Read_crt_func func,
                    void *arg)
{
  const CERT_CONTEXT *current_context = NULL;
  bool should_continue = true;
  while(should_continue &&
        (current_context = CertEnumCertificatesInStore(
          context->hCertStore,
          current_context)) != NULL)
    should_continue = func(current_context, arg);

  if(current_context)
    CertFreeCertificateContext(current_context);
}

static bool
cert_counter_callback(const CERT_CONTEXT *ccert_context, void *certs_count)
{
  if(valid_cert_encoding(ccert_context))
    (*(int *)certs_count)++;
  return true;
}

struct Adder_args
{
  struct Curl_easy *data;
  CURLcode result;
  int idx;
  int certs_count;
};

static bool
add_cert_to_certinfo(const CERT_CONTEXT *ccert_context, void *raw_arg)
{
  struct Adder_args *args = (struct Adder_args*)raw_arg;
  args->result = CURLE_OK;
  if(valid_cert_encoding(ccert_context)) {
    const char *beg = (const char *) ccert_context->pbCertEncoded;
    const char *end = beg + ccert_context->cbCertEncoded;
    int insert_index = (args->certs_count - 1) - args->idx;
    args->result = Curl_extract_certinfo(args->data, insert_index,
                                         beg, end);
    args->idx++;
  }
  return args->result == CURLE_OK;
}

static CURLcode
schannel_connect_step3(struct Curl_easy *data, struct connectdata *conn,
                       int sockindex)
{
  CURLcode result = CURLE_OK;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  SECURITY_STATUS sspi_status = SEC_E_OK;
  CERT_CONTEXT *ccert_context = NULL;
  bool isproxy = SSL_IS_PROXY();
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  const char * const hostname = SSL_HOST_NAME();
#endif
#ifdef HAS_ALPN
  SecPkgContext_ApplicationProtocol alpn_result;
#endif

  DEBUGASSERT(ssl_connect_3 == connssl->connecting_state);

  DEBUGF(infof(data,
               "schannel: SSL/TLS connection with %s port %hu (step 3/3)",
               hostname, conn->remote_port));

  if(!BACKEND->cred)
    return CURLE_SSL_CONNECT_ERROR;

  /* check if the required context attributes are met */
  if(BACKEND->ret_flags != BACKEND->req_flags) {
    if(!(BACKEND->ret_flags & ISC_RET_SEQUENCE_DETECT))
      failf(data, "schannel: failed to setup sequence detection");
    if(!(BACKEND->ret_flags & ISC_RET_REPLAY_DETECT))
      failf(data, "schannel: failed to setup replay detection");
    if(!(BACKEND->ret_flags & ISC_RET_CONFIDENTIALITY))
      failf(data, "schannel: failed to setup confidentiality");
    if(!(BACKEND->ret_flags & ISC_RET_ALLOCATED_MEMORY))
      failf(data, "schannel: failed to setup memory allocation");
    if(!(BACKEND->ret_flags & ISC_RET_STREAM))
      failf(data, "schannel: failed to setup stream orientation");
    return CURLE_SSL_CONNECT_ERROR;
  }

#ifdef HAS_ALPN
  if(BACKEND->use_alpn) {
    sspi_status =
      s_pSecFn->QueryContextAttributes(&BACKEND->ctxt->ctxt_handle,
                                       SECPKG_ATTR_APPLICATION_PROTOCOL,
                                       &alpn_result);

    if(sspi_status != SEC_E_OK) {
      failf(data, "schannel: failed to retrieve ALPN result");
      return CURLE_SSL_CONNECT_ERROR;
    }

    if(alpn_result.ProtoNegoStatus ==
       SecApplicationProtocolNegotiationStatus_Success) {

      infof(data, "schannel: ALPN, server accepted to use %.*s",
            alpn_result.ProtocolIdSize, alpn_result.ProtocolId);

#ifdef USE_HTTP2
      if(alpn_result.ProtocolIdSize == ALPN_H2_LENGTH &&
         !memcmp(ALPN_H2, alpn_result.ProtocolId, ALPN_H2_LENGTH)) {
        conn->negnpn = CURL_HTTP_VERSION_2;
      }
      else
#endif
        if(alpn_result.ProtocolIdSize == ALPN_HTTP_1_1_LENGTH &&
           !memcmp(ALPN_HTTP_1_1, alpn_result.ProtocolId,
                   ALPN_HTTP_1_1_LENGTH)) {
          conn->negnpn = CURL_HTTP_VERSION_1_1;
        }
    }
    else
      infof(data, "ALPN, server did not agree to a protocol");
    Curl_multiuse_state(data, conn->negnpn == CURL_HTTP_VERSION_2 ?
                        BUNDLE_MULTIPLEX : BUNDLE_NO_MULTIUSE);
  }
#endif

  /* save the current session data for possible re-use */
  if(SSL_SET_OPTION(primary.sessionid)) {
    bool incache;
    bool added = FALSE;
    struct Curl_schannel_cred *old_cred = NULL;

    Curl_ssl_sessionid_lock(data);
    incache = !(Curl_ssl_getsessionid(data, conn, isproxy, (void **)&old_cred,
                                      NULL, sockindex));
    if(incache) {
      if(old_cred != BACKEND->cred) {
        DEBUGF(infof(data,
                     "schannel: old credential handle is stale, removing"));
        /* we're not taking old_cred ownership here, no refcount++ is needed */
        Curl_ssl_delsessionid(data, (void *)old_cred);
        incache = FALSE;
      }
    }
    if(!incache) {
      result = Curl_ssl_addsessionid(data, conn, isproxy, BACKEND->cred,
                                     sizeof(struct Curl_schannel_cred),
                                     sockindex, &added);
      if(result) {
        Curl_ssl_sessionid_unlock(data);
        failf(data, "schannel: failed to store credential handle");
        return result;
      }
      else if(added) {
        /* this cred session is now also referenced by sessionid cache */
        BACKEND->cred->refcount++;
        DEBUGF(infof(data,
                     "schannel: stored credential handle in session cache"));
      }
    }
    Curl_ssl_sessionid_unlock(data);
  }

  if(data->set.ssl.certinfo) {
    int certs_count = 0;
    sspi_status =
      s_pSecFn->QueryContextAttributes(&BACKEND->ctxt->ctxt_handle,
                                       SECPKG_ATTR_REMOTE_CERT_CONTEXT,
                                       &ccert_context);

    if((sspi_status != SEC_E_OK) || !ccert_context) {
      failf(data, "schannel: failed to retrieve remote cert context");
      return CURLE_PEER_FAILED_VERIFICATION;
    }

    traverse_cert_store(ccert_context, cert_counter_callback, &certs_count);

    result = Curl_ssl_init_certinfo(data, certs_count);
    if(!result) {
      struct Adder_args args;
      args.data = data;
      args.idx = 0;
      args.certs_count = certs_count;
      traverse_cert_store(ccert_context, add_cert_to_certinfo, &args);
      result = args.result;
    }
    CertFreeCertificateContext(ccert_context);
    if(result)
      return result;
  }

  connssl->connecting_state = ssl_connect_done;

  return CURLE_OK;
}

static CURLcode
schannel_connect_common(struct Curl_easy *data, struct connectdata *conn,
                        int sockindex, bool nonblocking, bool *done)
{
  CURLcode result;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  curl_socket_t sockfd = conn->sock[sockindex];
  timediff_t timeout_ms;
  int what;

  /* check if the connection has already been established */
  if(ssl_connection_complete == connssl->state) {
    *done = TRUE;
    return CURLE_OK;
  }

  if(ssl_connect_1 == connssl->connecting_state) {
    /* check out how much more time we're allowed */
    timeout_ms = Curl_timeleft(data, NULL, TRUE);

    if(timeout_ms < 0) {
      /* no need to continue if time already is up */
      failf(data, "SSL/TLS connection timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    result = schannel_connect_step1(data, conn, sockindex);
    if(result)
      return result;
  }

  while(ssl_connect_2 == connssl->connecting_state ||
        ssl_connect_2_reading == connssl->connecting_state ||
        ssl_connect_2_writing == connssl->connecting_state) {

    /* check out how much more time we're allowed */
    timeout_ms = Curl_timeleft(data, NULL, TRUE);

    if(timeout_ms < 0) {
      /* no need to continue if time already is up */
      failf(data, "SSL/TLS connection timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    /* if ssl is expecting something, check if it's available. */
    if(connssl->connecting_state == ssl_connect_2_reading
       || connssl->connecting_state == ssl_connect_2_writing) {

      curl_socket_t writefd = ssl_connect_2_writing ==
        connssl->connecting_state ? sockfd : CURL_SOCKET_BAD;
      curl_socket_t readfd = ssl_connect_2_reading ==
        connssl->connecting_state ? sockfd : CURL_SOCKET_BAD;

      what = Curl_socket_check(readfd, CURL_SOCKET_BAD, writefd,
                               nonblocking ? 0 : timeout_ms);
      if(what < 0) {
        /* fatal error */
        failf(data, "select/poll on SSL/TLS socket, errno: %d", SOCKERRNO);
        return CURLE_SSL_CONNECT_ERROR;
      }
      else if(0 == what) {
        if(nonblocking) {
          *done = FALSE;
          return CURLE_OK;
        }
        else {
          /* timeout */
          failf(data, "SSL/TLS connection timeout");
          return CURLE_OPERATION_TIMEDOUT;
        }
      }
      /* socket is readable or writable */
    }

    /* Run transaction, and return to the caller if it failed or if
     * this connection is part of a multi handle and this loop would
     * execute again. This permits the owner of a multi handle to
     * abort a connection attempt before step2 has completed while
     * ensuring that a client using select() or epoll() will always
     * have a valid fdset to wait on.
     */
    result = schannel_connect_step2(data, conn, sockindex);
    if(result || (nonblocking &&
                  (ssl_connect_2 == connssl->connecting_state ||
                   ssl_connect_2_reading == connssl->connecting_state ||
                   ssl_connect_2_writing == connssl->connecting_state)))
      return result;

  } /* repeat step2 until all transactions are done. */

  if(ssl_connect_3 == connssl->connecting_state) {
    result = schannel_connect_step3(data, conn, sockindex);
    if(result)
      return result;
  }

  if(ssl_connect_done == connssl->connecting_state) {
    connssl->state = ssl_connection_complete;
    conn->recv[sockindex] = schannel_recv;
    conn->send[sockindex] = schannel_send;

#ifdef SECPKG_ATTR_ENDPOINT_BINDINGS
    /* When SSPI is used in combination with Schannel
     * we need the Schannel context to create the Schannel
     * binding to pass the IIS extended protection checks.
     * Available on Windows 7 or later.
     */
    conn->sslContext = &BACKEND->ctxt->ctxt_handle;
#endif

    *done = TRUE;
  }
  else
    *done = FALSE;

  /* reset our connection state machine */
  connssl->connecting_state = ssl_connect_1;

  return CURLE_OK;
}

static ssize_t
schannel_send(struct Curl_easy *data, int sockindex,
              const void *buf, size_t len, CURLcode *err)
{
  ssize_t written = -1;
  size_t data_len = 0;
  unsigned char *ptr = NULL;
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  SecBuffer outbuf[4];
  SecBufferDesc outbuf_desc;
  SECURITY_STATUS sspi_status = SEC_E_OK;
  CURLcode result;

  /* check if the maximum stream sizes were queried */
  if(BACKEND->stream_sizes.cbMaximumMessage == 0) {
    sspi_status = s_pSecFn->QueryContextAttributes(
      &BACKEND->ctxt->ctxt_handle,
      SECPKG_ATTR_STREAM_SIZES,
      &BACKEND->stream_sizes);
    if(sspi_status != SEC_E_OK) {
      *err = CURLE_SEND_ERROR;
      return -1;
    }
  }

  /* check if the buffer is longer than the maximum message length */
  if(len > BACKEND->stream_sizes.cbMaximumMessage) {
    len = BACKEND->stream_sizes.cbMaximumMessage;
  }

  /* calculate the complete message length and allocate a buffer for it */
  data_len = BACKEND->stream_sizes.cbHeader + len +
    BACKEND->stream_sizes.cbTrailer;
  ptr = (unsigned char *) malloc(data_len);
  if(!ptr) {
    *err = CURLE_OUT_OF_MEMORY;
    return -1;
  }

  /* setup output buffers (header, data, trailer, empty) */
  InitSecBuffer(&outbuf[0], SECBUFFER_STREAM_HEADER,
                ptr, BACKEND->stream_sizes.cbHeader);
  InitSecBuffer(&outbuf[1], SECBUFFER_DATA,
                ptr + BACKEND->stream_sizes.cbHeader, curlx_uztoul(len));
  InitSecBuffer(&outbuf[2], SECBUFFER_STREAM_TRAILER,
                ptr + BACKEND->stream_sizes.cbHeader + len,
                BACKEND->stream_sizes.cbTrailer);
  InitSecBuffer(&outbuf[3], SECBUFFER_EMPTY, NULL, 0);
  InitSecBufferDesc(&outbuf_desc, outbuf, 4);

  /* copy data into output buffer */
  memcpy(outbuf[1].pvBuffer, buf, len);

  /* https://msdn.microsoft.com/en-us/library/windows/desktop/aa375390.aspx */
  sspi_status = s_pSecFn->EncryptMessage(&BACKEND->ctxt->ctxt_handle, 0,
                                         &outbuf_desc, 0);

  /* check if the message was encrypted */
  if(sspi_status == SEC_E_OK) {
    written = 0;

    /* send the encrypted message including header, data and trailer */
    len = outbuf[0].cbBuffer + outbuf[1].cbBuffer + outbuf[2].cbBuffer;

    /*
      It's important to send the full message which includes the header,
      encrypted payload, and trailer.  Until the client receives all the
      data a coherent message has not been delivered and the client
      can't read any of it.

      If we wanted to buffer the unwritten encrypted bytes, we would
      tell the client that all data it has requested to be sent has been
      sent. The unwritten encrypted bytes would be the first bytes to
      send on the next invocation.
      Here's the catch with this - if we tell the client that all the
      bytes have been sent, will the client call this method again to
      send the buffered data?  Looking at who calls this function, it
      seems the answer is NO.
    */

    /* send entire message or fail */
    while(len > (size_t)written) {
      ssize_t this_write = 0;
      int what;
      timediff_t timeout_ms = Curl_timeleft(data, NULL, FALSE);
      if(timeout_ms < 0) {
        /* we already got the timeout */
        failf(data, "schannel: timed out sending data "
              "(bytes sent: %zd)", written);
        *err = CURLE_OPERATION_TIMEDOUT;
        written = -1;
        break;
      }
      else if(!timeout_ms)
        timeout_ms = TIMEDIFF_T_MAX;
      what = SOCKET_WRITABLE(conn->sock[sockindex], timeout_ms);
      if(what < 0) {
        /* fatal error */
        failf(data, "select/poll on SSL socket, errno: %d", SOCKERRNO);
        *err = CURLE_SEND_ERROR;
        written = -1;
        break;
      }
      else if(0 == what) {
        failf(data, "schannel: timed out sending data "
              "(bytes sent: %zd)", written);
        *err = CURLE_OPERATION_TIMEDOUT;
        written = -1;
        break;
      }
      /* socket is writable */

      result = Curl_write_plain(data, conn->sock[sockindex], ptr + written,
                                len - written, &this_write);
      if(result == CURLE_AGAIN)
        continue;
      else if(result != CURLE_OK) {
        *err = result;
        written = -1;
        break;
      }

      written += this_write;
    }
  }
  else if(sspi_status == SEC_E_INSUFFICIENT_MEMORY) {
    *err = CURLE_OUT_OF_MEMORY;
  }
  else{
    *err = CURLE_SEND_ERROR;
  }

  Curl_safefree(ptr);

  if(len == (size_t)written)
    /* Encrypted message including header, data and trailer entirely sent.
       The return value is the number of unencrypted bytes that were sent. */
    written = outbuf[1].cbBuffer;

  return written;
}

static ssize_t
schannel_recv(struct Curl_easy *data, int sockindex,
              char *buf, size_t len, CURLcode *err)
{
  size_t size = 0;
  ssize_t nread = -1;
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  unsigned char *reallocated_buffer;
  size_t reallocated_length;
  bool done = FALSE;
  SecBuffer inbuf[4];
  SecBufferDesc inbuf_desc;
  SECURITY_STATUS sspi_status = SEC_E_OK;
  /* we want the length of the encrypted buffer to be at least large enough
     that it can hold all the bytes requested and some TLS record overhead. */
  size_t min_encdata_length = len + CURL_SCHANNEL_BUFFER_FREE_SIZE;

  /****************************************************************************
   * Don't return or set BACKEND->recv_unrecoverable_err unless in the cleanup.
   * The pattern for return error is set *err, optional infof, goto cleanup.
   *
   * Our priority is to always return as much decrypted data to the caller as
   * possible, even if an error occurs. The state of the decrypted buffer must
   * always be valid. Transfer of decrypted data to the caller's buffer is
   * handled in the cleanup.
   */

  DEBUGF(infof(data, "schannel: client wants to read %zu bytes", len));
  *err = CURLE_OK;

  if(len && len <= BACKEND->decdata_offset) {
    infof(data, "schannel: enough decrypted data is already available");
    goto cleanup;
  }
  else if(BACKEND->recv_unrecoverable_err) {
    *err = BACKEND->recv_unrecoverable_err;
    infof(data, "schannel: an unrecoverable error occurred in a prior call");
    goto cleanup;
  }
  else if(BACKEND->recv_sspi_close_notify) {
    /* once a server has indicated shutdown there is no more encrypted data */
    infof(data, "schannel: server indicated shutdown in a prior call");
    goto cleanup;
  }

  /* It's debatable what to return when !len. Regardless we can't return
     immediately because there may be data to decrypt (in the case we want to
     decrypt all encrypted cached data) so handle !len later in cleanup.
  */
  else if(len && !BACKEND->recv_connection_closed) {
    /* increase enc buffer in order to fit the requested amount of data */
    size = BACKEND->encdata_length - BACKEND->encdata_offset;
    if(size < CURL_SCHANNEL_BUFFER_FREE_SIZE ||
       BACKEND->encdata_length < min_encdata_length) {
      reallocated_length = BACKEND->encdata_offset +
        CURL_SCHANNEL_BUFFER_FREE_SIZE;
      if(reallocated_length < min_encdata_length) {
        reallocated_length = min_encdata_length;
      }
      reallocated_buffer = realloc(BACKEND->encdata_buffer,
                                   reallocated_length);
      if(!reallocated_buffer) {
        *err = CURLE_OUT_OF_MEMORY;
        failf(data, "schannel: unable to re-allocate memory");
        goto cleanup;
      }

      BACKEND->encdata_buffer = reallocated_buffer;
      BACKEND->encdata_length = reallocated_length;
      size = BACKEND->encdata_length - BACKEND->encdata_offset;
      DEBUGF(infof(data, "schannel: encdata_buffer resized %zu",
                   BACKEND->encdata_length));
    }

    DEBUGF(infof(data,
                 "schannel: encrypted data buffer: offset %zu length %zu",
                 BACKEND->encdata_offset, BACKEND->encdata_length));

    /* read encrypted data from socket */
    *err = Curl_read_plain(conn->sock[sockindex],
                           (char *)(BACKEND->encdata_buffer +
                                    BACKEND->encdata_offset),
                           size, &nread);
    if(*err) {
      nread = -1;
      if(*err == CURLE_AGAIN)
        DEBUGF(infof(data,
                     "schannel: Curl_read_plain returned CURLE_AGAIN"));
      else if(*err == CURLE_RECV_ERROR)
        infof(data, "schannel: Curl_read_plain returned CURLE_RECV_ERROR");
      else
        infof(data, "schannel: Curl_read_plain returned error %d", *err);
    }
    else if(nread == 0) {
      BACKEND->recv_connection_closed = true;
      DEBUGF(infof(data, "schannel: server closed the connection"));
    }
    else if(nread > 0) {
      BACKEND->encdata_offset += (size_t)nread;
      BACKEND->encdata_is_incomplete = false;
      DEBUGF(infof(data, "schannel: encrypted data got %zd", nread));
    }
  }

  DEBUGF(infof(data,
               "schannel: encrypted data buffer: offset %zu length %zu",
               BACKEND->encdata_offset, BACKEND->encdata_length));

  /* decrypt loop */
  while(BACKEND->encdata_offset > 0 && sspi_status == SEC_E_OK &&
        (!len || BACKEND->decdata_offset < len ||
         BACKEND->recv_connection_closed)) {
    /* prepare data buffer for DecryptMessage call */
    InitSecBuffer(&inbuf[0], SECBUFFER_DATA, BACKEND->encdata_buffer,
                  curlx_uztoul(BACKEND->encdata_offset));

    /* we need 3 more empty input buffers for possible output */
    InitSecBuffer(&inbuf[1], SECBUFFER_EMPTY, NULL, 0);
    InitSecBuffer(&inbuf[2], SECBUFFER_EMPTY, NULL, 0);
    InitSecBuffer(&inbuf[3], SECBUFFER_EMPTY, NULL, 0);
    InitSecBufferDesc(&inbuf_desc, inbuf, 4);

    /* https://msdn.microsoft.com/en-us/library/windows/desktop/aa375348.aspx
     */
    sspi_status = s_pSecFn->DecryptMessage(&BACKEND->ctxt->ctxt_handle,
                                           &inbuf_desc, 0, NULL);

    /* check if everything went fine (server may want to renegotiate
       or shutdown the connection context) */
    if(sspi_status == SEC_E_OK || sspi_status == SEC_I_RENEGOTIATE ||
       sspi_status == SEC_I_CONTEXT_EXPIRED) {
      /* check for successfully decrypted data, even before actual
         renegotiation or shutdown of the connection context */
      if(inbuf[1].BufferType == SECBUFFER_DATA) {
        DEBUGF(infof(data, "schannel: decrypted data length: %lu",
                     inbuf[1].cbBuffer));

        /* increase buffer in order to fit the received amount of data */
        size = inbuf[1].cbBuffer > CURL_SCHANNEL_BUFFER_FREE_SIZE ?
          inbuf[1].cbBuffer : CURL_SCHANNEL_BUFFER_FREE_SIZE;
        if(BACKEND->decdata_length - BACKEND->decdata_offset < size ||
           BACKEND->decdata_length < len) {
          /* increase internal decrypted data buffer */
          reallocated_length = BACKEND->decdata_offset + size;
          /* make sure that the requested amount of data fits */
          if(reallocated_length < len) {
            reallocated_length = len;
          }
          reallocated_buffer = realloc(BACKEND->decdata_buffer,
                                       reallocated_length);
          if(!reallocated_buffer) {
            *err = CURLE_OUT_OF_MEMORY;
            failf(data, "schannel: unable to re-allocate memory");
            goto cleanup;
          }
          BACKEND->decdata_buffer = reallocated_buffer;
          BACKEND->decdata_length = reallocated_length;
        }

        /* copy decrypted data to internal buffer */
        size = inbuf[1].cbBuffer;
        if(size) {
          memcpy(BACKEND->decdata_buffer + BACKEND->decdata_offset,
                 inbuf[1].pvBuffer, size);
          BACKEND->decdata_offset += size;
        }

        DEBUGF(infof(data, "schannel: decrypted data added: %zu", size));
        DEBUGF(infof(data,
                     "schannel: decrypted cached: offset %zu length %zu",
                     BACKEND->decdata_offset, BACKEND->decdata_length));
      }

      /* check for remaining encrypted data */
      if(inbuf[3].BufferType == SECBUFFER_EXTRA && inbuf[3].cbBuffer > 0) {
        DEBUGF(infof(data, "schannel: encrypted data length: %lu",
                     inbuf[3].cbBuffer));

        /* check if the remaining data is less than the total amount
         * and therefore begins after the already processed data
         */
        if(BACKEND->encdata_offset > inbuf[3].cbBuffer) {
          /* move remaining encrypted data forward to the beginning of
             buffer */
          memmove(BACKEND->encdata_buffer,
                  (BACKEND->encdata_buffer + BACKEND->encdata_offset) -
                  inbuf[3].cbBuffer, inbuf[3].cbBuffer);
          BACKEND->encdata_offset = inbuf[3].cbBuffer;
        }

        DEBUGF(infof(data,
                     "schannel: encrypted cached: offset %zu length %zu",
                     BACKEND->encdata_offset, BACKEND->encdata_length));
      }
      else {
        /* reset encrypted buffer offset, because there is no data remaining */
        BACKEND->encdata_offset = 0;
      }

      /* check if server wants to renegotiate the connection context */
      if(sspi_status == SEC_I_RENEGOTIATE) {
        infof(data, "schannel: remote party requests renegotiation");
        if(*err && *err != CURLE_AGAIN) {
          infof(data, "schannel: can't renegotiate, an error is pending");
          goto cleanup;
        }
        if(BACKEND->encdata_offset) {
          *err = CURLE_RECV_ERROR;
          infof(data, "schannel: can't renegotiate, "
                "encrypted data available");
          goto cleanup;
        }
        /* begin renegotiation */
        infof(data, "schannel: renegotiating SSL/TLS connection");
        connssl->state = ssl_connection_negotiating;
        connssl->connecting_state = ssl_connect_2_writing;
        *err = schannel_connect_common(data, conn, sockindex, FALSE, &done);
        if(*err) {
          infof(data, "schannel: renegotiation failed");
          goto cleanup;
        }
        /* now retry receiving data */
        sspi_status = SEC_E_OK;
        infof(data, "schannel: SSL/TLS connection renegotiated");
        continue;
      }
      /* check if the server closed the connection */
      else if(sspi_status == SEC_I_CONTEXT_EXPIRED) {
        /* In Windows 2000 SEC_I_CONTEXT_EXPIRED (close_notify) is not
           returned so we have to work around that in cleanup. */
        BACKEND->recv_sspi_close_notify = true;
        if(!BACKEND->recv_connection_closed) {
          BACKEND->recv_connection_closed = true;
          infof(data, "schannel: server closed the connection");
        }
        goto cleanup;
      }
    }
    else if(sspi_status == SEC_E_INCOMPLETE_MESSAGE) {
      BACKEND->encdata_is_incomplete = true;
      if(!*err)
        *err = CURLE_AGAIN;
      infof(data, "schannel: failed to decrypt data, need more data");
      goto cleanup;
    }
    else {
#ifndef CURL_DISABLE_VERBOSE_STRINGS
      char buffer[STRERROR_LEN];
#endif
      *err = CURLE_RECV_ERROR;
      infof(data, "schannel: failed to read data from server: %s",
            Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
      goto cleanup;
    }
  }

  DEBUGF(infof(data,
               "schannel: encrypted data buffer: offset %zu length %zu",
               BACKEND->encdata_offset, BACKEND->encdata_length));

  DEBUGF(infof(data,
               "schannel: decrypted data buffer: offset %zu length %zu",
               BACKEND->decdata_offset, BACKEND->decdata_length));

  cleanup:
  /* Warning- there is no guarantee the encdata state is valid at this point */
  DEBUGF(infof(data, "schannel: schannel_recv cleanup"));

  /* Error if the connection has closed without a close_notify.

     The behavior here is a matter of debate. We don't want to be vulnerable
     to a truncation attack however there's some browser precedent for
     ignoring the close_notify for compatibility reasons.

     Additionally, Windows 2000 (v5.0) is a special case since it seems it
     doesn't return close_notify. In that case if the connection was closed we
     assume it was graceful (close_notify) since there doesn't seem to be a
     way to tell.
  */
  if(len && !BACKEND->decdata_offset && BACKEND->recv_connection_closed &&
     !BACKEND->recv_sspi_close_notify) {
    bool isWin2k = curlx_verify_windows_version(5, 0, 0, PLATFORM_WINNT,
                                                VERSION_EQUAL);

    if(isWin2k && sspi_status == SEC_E_OK)
      BACKEND->recv_sspi_close_notify = true;
    else {
      *err = CURLE_RECV_ERROR;
      infof(data, "schannel: server closed abruptly (missing close_notify)");
    }
  }

  /* Any error other than CURLE_AGAIN is an unrecoverable error. */
  if(*err && *err != CURLE_AGAIN)
    BACKEND->recv_unrecoverable_err = *err;

  size = len < BACKEND->decdata_offset ? len : BACKEND->decdata_offset;
  if(size) {
    memcpy(buf, BACKEND->decdata_buffer, size);
    memmove(BACKEND->decdata_buffer, BACKEND->decdata_buffer + size,
            BACKEND->decdata_offset - size);
    BACKEND->decdata_offset -= size;
    DEBUGF(infof(data, "schannel: decrypted data returned %zu", size));
    DEBUGF(infof(data,
                 "schannel: decrypted data buffer: offset %zu length %zu",
                 BACKEND->decdata_offset, BACKEND->decdata_length));
    *err = CURLE_OK;
    return (ssize_t)size;
  }

  if(!*err && !BACKEND->recv_connection_closed)
    *err = CURLE_AGAIN;

  /* It's debatable what to return when !len. We could return whatever error
     we got from decryption but instead we override here so the return is
     consistent.
  */
  if(!len)
    *err = CURLE_OK;

  return *err ? -1 : 0;
}

static CURLcode schannel_connect_nonblocking(struct Curl_easy *data,
                                             struct connectdata *conn,
                                             int sockindex, bool *done)
{
  return schannel_connect_common(data, conn, sockindex, TRUE, done);
}

static CURLcode schannel_connect(struct Curl_easy *data,
                                 struct connectdata *conn, int sockindex)
{
  CURLcode result;
  bool done = FALSE;

  result = schannel_connect_common(data, conn, sockindex, FALSE, &done);
  if(result)
    return result;

  DEBUGASSERT(done);

  return CURLE_OK;
}

static bool schannel_data_pending(const struct connectdata *conn,
                                  int sockindex)
{
  const struct ssl_connect_data *connssl = &conn->ssl[sockindex];

  if(connssl->use) /* SSL/TLS is in use */
    return (BACKEND->decdata_offset > 0 ||
            (BACKEND->encdata_offset > 0 && !BACKEND->encdata_is_incomplete));
  else
    return FALSE;
}

static void schannel_session_free(void *ptr)
{
  /* this is expected to be called under sessionid lock */
  struct Curl_schannel_cred *cred = ptr;

  if(cred) {
    cred->refcount--;
    if(cred->refcount == 0) {
      s_pSecFn->FreeCredentialsHandle(&cred->cred_handle);
      Curl_safefree(cred);
    }
  }
}

/* shut down the SSL connection and clean up related memory.
   this function can be called multiple times on the same connection including
   if the SSL connection failed (eg connection made but failed handshake). */
static int schannel_shutdown(struct Curl_easy *data, struct connectdata *conn,
                             int sockindex)
{
  /* See https://msdn.microsoft.com/en-us/library/windows/desktop/aa380138.aspx
   * Shutting Down an Schannel Connection
   */
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  char * const hostname = SSL_HOST_NAME();

  DEBUGASSERT(data);

  if(connssl->use) {
    infof(data, "schannel: shutting down SSL/TLS connection with %s port %hu",
          hostname, conn->remote_port);
  }

  if(connssl->use && BACKEND->cred && BACKEND->ctxt) {
    SecBufferDesc BuffDesc;
    SecBuffer Buffer;
    SECURITY_STATUS sspi_status;
    SecBuffer outbuf;
    SecBufferDesc outbuf_desc;
    CURLcode result;
    TCHAR *host_name;
    DWORD dwshut = SCHANNEL_SHUTDOWN;

    InitSecBuffer(&Buffer, SECBUFFER_TOKEN, &dwshut, sizeof(dwshut));
    InitSecBufferDesc(&BuffDesc, &Buffer, 1);

    sspi_status = s_pSecFn->ApplyControlToken(&BACKEND->ctxt->ctxt_handle,
                                              &BuffDesc);

    if(sspi_status != SEC_E_OK) {
      char buffer[STRERROR_LEN];
      failf(data, "schannel: ApplyControlToken failure: %s",
            Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
    }

    host_name = curlx_convert_UTF8_to_tchar(hostname);
    if(!host_name)
      return CURLE_OUT_OF_MEMORY;

    /* setup output buffer */
    InitSecBuffer(&outbuf, SECBUFFER_EMPTY, NULL, 0);
    InitSecBufferDesc(&outbuf_desc, &outbuf, 1);

    sspi_status = s_pSecFn->InitializeSecurityContext(
      &BACKEND->cred->cred_handle,
      &BACKEND->ctxt->ctxt_handle,
      host_name,
      BACKEND->req_flags,
      0,
      0,
      NULL,
      0,
      &BACKEND->ctxt->ctxt_handle,
      &outbuf_desc,
      &BACKEND->ret_flags,
      &BACKEND->ctxt->time_stamp);

    curlx_unicodefree(host_name);

    if((sspi_status == SEC_E_OK) || (sspi_status == SEC_I_CONTEXT_EXPIRED)) {
      /* send close message which is in output buffer */
      ssize_t written;
      result = Curl_write_plain(data, conn->sock[sockindex], outbuf.pvBuffer,
                                outbuf.cbBuffer, &written);

      s_pSecFn->FreeContextBuffer(outbuf.pvBuffer);
      if((result != CURLE_OK) || (outbuf.cbBuffer != (size_t) written)) {
        infof(data, "schannel: failed to send close msg: %s"
              " (bytes written: %zd)", curl_easy_strerror(result), written);
      }
    }
  }

  /* free SSPI Schannel API security context handle */
  if(BACKEND->ctxt) {
    DEBUGF(infof(data, "schannel: clear security context handle"));
    s_pSecFn->DeleteSecurityContext(&BACKEND->ctxt->ctxt_handle);
    Curl_safefree(BACKEND->ctxt);
  }

  /* free SSPI Schannel API credential handle */
  if(BACKEND->cred) {
    Curl_ssl_sessionid_lock(data);
    schannel_session_free(BACKEND->cred);
    Curl_ssl_sessionid_unlock(data);
    BACKEND->cred = NULL;
  }

  /* free internal buffer for received encrypted data */
  if(BACKEND->encdata_buffer != NULL) {
    Curl_safefree(BACKEND->encdata_buffer);
    BACKEND->encdata_length = 0;
    BACKEND->encdata_offset = 0;
    BACKEND->encdata_is_incomplete = false;
  }

  /* free internal buffer for received decrypted data */
  if(BACKEND->decdata_buffer != NULL) {
    Curl_safefree(BACKEND->decdata_buffer);
    BACKEND->decdata_length = 0;
    BACKEND->decdata_offset = 0;
  }

  return CURLE_OK;
}

static void schannel_close(struct Curl_easy *data, struct connectdata *conn,
                           int sockindex)
{
  if(conn->ssl[sockindex].use)
    /* Curl_ssl_shutdown resets the socket state and calls schannel_shutdown */
    Curl_ssl_shutdown(data, conn, sockindex);
  else
    schannel_shutdown(data, conn, sockindex);
}

static int schannel_init(void)
{
  return (Curl_sspi_global_init() == CURLE_OK ? 1 : 0);
}

static void schannel_cleanup(void)
{
  Curl_sspi_global_cleanup();
}

static size_t schannel_version(char *buffer, size_t size)
{
  size = msnprintf(buffer, size, "Schannel");

  return size;
}

static CURLcode schannel_random(struct Curl_easy *data UNUSED_PARAM,
                                unsigned char *entropy, size_t length)
{
  HCRYPTPROV hCryptProv = 0;

  (void)data;

  if(!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL,
                          CRYPT_VERIFYCONTEXT | CRYPT_SILENT))
    return CURLE_FAILED_INIT;

  if(!CryptGenRandom(hCryptProv, (DWORD)length, entropy)) {
    CryptReleaseContext(hCryptProv, 0UL);
    return CURLE_FAILED_INIT;
  }

  CryptReleaseContext(hCryptProv, 0UL);
  return CURLE_OK;
}

static CURLcode pkp_pin_peer_pubkey(struct Curl_easy *data,
                                    struct connectdata *conn, int sockindex,
                                    const char *pinnedpubkey)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  CERT_CONTEXT *pCertContextServer = NULL;

  /* Result is returned to caller */
  CURLcode result = CURLE_SSL_PINNEDPUBKEYNOTMATCH;

  /* if a path wasn't specified, don't pin */
  if(!pinnedpubkey)
    return CURLE_OK;

  do {
    SECURITY_STATUS sspi_status;
    const char *x509_der;
    DWORD x509_der_len;
    struct Curl_X509certificate x509_parsed;
    struct Curl_asn1Element *pubkey;

    sspi_status =
      s_pSecFn->QueryContextAttributes(&BACKEND->ctxt->ctxt_handle,
                                       SECPKG_ATTR_REMOTE_CERT_CONTEXT,
                                       &pCertContextServer);

    if((sspi_status != SEC_E_OK) || !pCertContextServer) {
      char buffer[STRERROR_LEN];
      failf(data, "schannel: Failed to read remote certificate context: %s",
            Curl_sspi_strerror(sspi_status, buffer, sizeof(buffer)));
      break; /* failed */
    }


    if(!(((pCertContextServer->dwCertEncodingType & X509_ASN_ENCODING) != 0) &&
         (pCertContextServer->cbCertEncoded > 0)))
      break;

    x509_der = (const char *)pCertContextServer->pbCertEncoded;
    x509_der_len = pCertContextServer->cbCertEncoded;
    memset(&x509_parsed, 0, sizeof(x509_parsed));
    if(Curl_parseX509(&x509_parsed, x509_der, x509_der + x509_der_len))
      break;

    pubkey = &x509_parsed.subjectPublicKeyInfo;
    if(!pubkey->header || pubkey->end <= pubkey->header) {
      failf(data, "SSL: failed retrieving public key from server certificate");
      break;
    }

    result = Curl_pin_peer_pubkey(data,
                                  pinnedpubkey,
                                  (const unsigned char *)pubkey->header,
                                  (size_t)(pubkey->end - pubkey->header));
    if(result) {
      failf(data, "SSL: public key does not match pinned public key!");
    }
  } while(0);

  if(pCertContextServer)
    CertFreeCertificateContext(pCertContextServer);

  return result;
}

static void schannel_checksum(const unsigned char *input,
                              size_t inputlen,
                              unsigned char *checksum,
                              size_t checksumlen,
                              DWORD provType,
                              const unsigned int algId)
{
  HCRYPTPROV hProv = 0;
  HCRYPTHASH hHash = 0;
  DWORD cbHashSize = 0;
  DWORD dwHashSizeLen = (DWORD)sizeof(cbHashSize);
  DWORD dwChecksumLen = (DWORD)checksumlen;

  /* since this can fail in multiple ways, zero memory first so we never
   * return old data
   */
  memset(checksum, 0, checksumlen);

  if(!CryptAcquireContext(&hProv, NULL, NULL, provType,
                          CRYPT_VERIFYCONTEXT | CRYPT_SILENT))
    return; /* failed */

  do {
    if(!CryptCreateHash(hProv, algId, 0, 0, &hHash))
      break; /* failed */

    /* workaround for original MinGW, should be (const BYTE*) */
    if(!CryptHashData(hHash, (BYTE*)input, (DWORD)inputlen, 0))
      break; /* failed */

    /* get hash size */
    if(!CryptGetHashParam(hHash, HP_HASHSIZE, (BYTE *)&cbHashSize,
                          &dwHashSizeLen, 0))
      break; /* failed */

    /* check hash size */
    if(checksumlen < cbHashSize)
      break; /* failed */

    if(CryptGetHashParam(hHash, HP_HASHVAL, checksum, &dwChecksumLen, 0))
      break; /* failed */
  } while(0);

  if(hHash)
    CryptDestroyHash(hHash);

  if(hProv)
    CryptReleaseContext(hProv, 0);
}

static CURLcode schannel_sha256sum(const unsigned char *input,
                                   size_t inputlen,
                                   unsigned char *sha256sum,
                                   size_t sha256len)
{
  schannel_checksum(input, inputlen, sha256sum, sha256len,
                    PROV_RSA_AES, CALG_SHA_256);
  return CURLE_OK;
}

static void *schannel_get_internals(struct ssl_connect_data *connssl,
                                    CURLINFO info UNUSED_PARAM)
{
  (void)info;
  return &BACKEND->ctxt->ctxt_handle;
}

const struct Curl_ssl Curl_ssl_schannel = {
  { CURLSSLBACKEND_SCHANNEL, "schannel" }, /* info */

  SSLSUPP_CERTINFO |
#ifdef HAS_MANUAL_VERIFY_API
  SSLSUPP_CAINFO_BLOB |
#endif
  SSLSUPP_PINNEDPUBKEY,

  sizeof(struct ssl_backend_data),

  schannel_init,                     /* init */
  schannel_cleanup,                  /* cleanup */
  schannel_version,                  /* version */
  Curl_none_check_cxn,               /* check_cxn */
  schannel_shutdown,                 /* shutdown */
  schannel_data_pending,             /* data_pending */
  schannel_random,                   /* random */
  Curl_none_cert_status_request,     /* cert_status_request */
  schannel_connect,                  /* connect */
  schannel_connect_nonblocking,      /* connect_nonblocking */
  Curl_ssl_getsock,                  /* getsock */
  schannel_get_internals,            /* get_internals */
  schannel_close,                    /* close_one */
  Curl_none_close_all,               /* close_all */
  schannel_session_free,             /* session_free */
  Curl_none_set_engine,              /* set_engine */
  Curl_none_set_engine_default,      /* set_engine_default */
  Curl_none_engines_list,            /* engines_list */
  Curl_none_false_start,             /* false_start */
  schannel_sha256sum,                /* sha256sum */
  NULL,                              /* associate_connection */
  NULL                               /* disassociate_connection */
};

#endif /* USE_SCHANNEL */
