#ifndef HEADER_CURL_SETUP_VMS_H
#define HEADER_CURL_SETUP_VMS_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

/*                                                                         */
/* JEM, 12/30/12, VMS now generates config.h, so only define wrappers for  */
/*                getenv(), getpwuid() and provide is_vms_shell()          */
/*                Also need upper case symbols for system services, and    */
/*                OpenSSL, and some Kerberos image                         */

#ifdef __DECC
#pragma message save
#pragma message disable dollarid
#endif

/* Hide the stuff we are overriding */
#define getenv decc_getenv
#ifdef __DECC
#   if __INITIAL_POINTER_SIZE != 64
#       define getpwuid decc_getpwuid
#   endif
#endif
#include <stdlib.h>
char *decc$getenv(const char *__name);
#include <pwd.h>

#include <string.h>
#include <unixlib.h>

#undef getenv
#undef getpwuid
#define getenv vms_getenv
#define getpwuid vms_getpwuid

/* VAX needs these in upper case when compiling exact case */
#define sys$assign SYS$ASSIGN
#define sys$dassgn SYS$DASSGN
#define sys$qiow SYS$QIOW

#ifdef __DECC
#   if __INITIAL_POINTER_SIZE
#       pragma __pointer_size __save
#   endif
#endif

#if __USE_LONG_GID_T
#   define decc_getpwuid DECC$__LONG_GID_GETPWUID
#else
#   if __INITIAL_POINTER_SIZE
#       define decc_getpwuid decc$__32_getpwuid
#   else
#       define decc_getpwuid decc$getpwuid
#   endif
#endif

    struct passwd *decc_getpwuid(uid_t uid);

#ifdef __DECC
#   if __INITIAL_POINTER_SIZE == 32
/* Translate the path, but only if the path is a VMS file specification */
/* The translation is usually only needed for older versions of VMS */
static char *vms_translate_path(const char *path)
{
  char *unix_path;
  char *test_str;

  /* See if the result is in VMS format, if not, we are done */
  /* Assume that this is a PATH, not just some data */
  test_str = strpbrk(path, ":[<^");
  if(!test_str) {
    return (char *)path;
  }

  unix_path = decc$translate_vms(path);

  if((int)unix_path <= 0) {
    /* We can not translate it, so return the original string */
    return (char *)path;
  }
}
#   else
    /* VMS translate path is actually not needed on the current 64 bit */
    /* VMS platforms, so instead of figuring out the pointer settings */
    /* Change it to a noop */
#   define vms_translate_path(__path) __path
#   endif
#endif

#ifdef __DECC
#   if __INITIAL_POINTER_SIZE
#       pragma __pointer_size __restore
#   endif
#endif

static char *vms_getenv(const char *envvar)
{
  char *result;
  char *vms_path;

  /* first use the DECC getenv() function */
  result = decc$getenv(envvar);
  if(!result) {
    return result;
  }

  vms_path = result;
  result = vms_translate_path(vms_path);

  /* note that if you backport this to use VAX C RTL, that the VAX C RTL */
  /* may do a malloc(2048) for each call to getenv(), so you will need   */
  /* to add a free(vms_path) */
  /* Do not do a free() for DEC C RTL builds, which should be used for */
  /* VMS 5.5-2 and later, even if using GCC */

  return result;
}


static struct passwd vms_passwd_cache;

static struct passwd *vms_getpwuid(uid_t uid)
{
  struct passwd *my_passwd;

/* Hack needed to support 64 bit builds, decc_getpwnam is 32 bit only */
#ifdef __DECC
#   if __INITIAL_POINTER_SIZE
  __char_ptr32 unix_path;
#   else
  char *unix_path;
#   endif
#else
  char *unix_path;
#endif

  my_passwd = decc_getpwuid(uid);
  if(!my_passwd) {
    return my_passwd;
  }

  unix_path = vms_translate_path(my_passwd->pw_dir);

  if((long)unix_path <= 0) {
    /* We can not translate it, so return the original string */
    return my_passwd;
  }

  /* If no changes needed just return it */
  if(unix_path == my_passwd->pw_dir) {
    return my_passwd;
  }

  /* Need to copy the structure returned */
  /* Since curl is only using pw_dir, no need to fix up */
  /* the pw_shell when running under Bash */
  vms_passwd_cache.pw_name = my_passwd->pw_name;
  vms_passwd_cache.pw_uid = my_passwd->pw_uid;
  vms_passwd_cache.pw_gid = my_passwd->pw_uid;
  vms_passwd_cache.pw_dir = unix_path;
  vms_passwd_cache.pw_shell = my_passwd->pw_shell;

  return &vms_passwd_cache;
}

#ifdef __DECC
#pragma message restore
#endif

/* Bug - VMS OpenSSL and Kerberos universal symbols are in uppercase only */
/* VMS libraries should have universal symbols in exact and uppercase */

#define ASN1_INTEGER_get ASN1_INTEGER_GET
#define ASN1_STRING_data ASN1_STRING_DATA
#define ASN1_STRING_length ASN1_STRING_LENGTH
#define ASN1_STRING_print ASN1_STRING_PRINT
#define ASN1_STRING_to_UTF8 ASN1_STRING_TO_UTF8
#define ASN1_STRING_type ASN1_STRING_TYPE
#define BIO_ctrl BIO_CTRL
#define BIO_free BIO_FREE
#define BIO_new BIO_NEW
#define BIO_s_mem BIO_S_MEM
#define BN_bn2bin BN_BN2BIN
#define BN_num_bits BN_NUM_BITS
#define CRYPTO_cleanup_all_ex_data CRYPTO_CLEANUP_ALL_EX_DATA
#define CRYPTO_free CRYPTO_FREE
#define CRYPTO_malloc CRYPTO_MALLOC
#define CONF_modules_load_file CONF_MODULES_LOAD_FILE
#ifdef __VAX
#  ifdef VMS_OLD_SSL
  /* Ancient OpenSSL on VAX/VMS missing this constant */
#    define CONF_MFLAGS_IGNORE_MISSING_FILE 0x10
#    undef CONF_modules_load_file
     static int CONF_modules_load_file(const char *filename,
                                       const char *appname,
                                       unsigned long flags) {
             return 1;
     }
#  endif
#endif
#define DES_ecb_encrypt DES_ECB_ENCRYPT
#define DES_set_key DES_SET_KEY
#define DES_set_odd_parity DES_SET_ODD_PARITY
#define ENGINE_ctrl ENGINE_CTRL
#define ENGINE_ctrl_cmd ENGINE_CTRL_CMD
#define ENGINE_finish ENGINE_FINISH
#define ENGINE_free ENGINE_FREE
#define ENGINE_get_first ENGINE_GET_FIRST
#define ENGINE_get_id ENGINE_GET_ID
#define ENGINE_get_next ENGINE_GET_NEXT
#define ENGINE_init ENGINE_INIT
#define ENGINE_load_builtin_engines ENGINE_LOAD_BUILTIN_ENGINES
#define ENGINE_load_private_key ENGINE_LOAD_PRIVATE_KEY
#define ENGINE_set_default ENGINE_SET_DEFAULT
#define ERR_clear_error ERR_CLEAR_ERROR
#define ERR_error_string ERR_ERROR_STRING
#define ERR_error_string_n ERR_ERROR_STRING_N
#define ERR_free_strings ERR_FREE_STRINGS
#define ERR_get_error ERR_GET_ERROR
#define ERR_peek_error ERR_PEEK_ERROR
#define ERR_remove_state ERR_REMOVE_STATE
#define EVP_PKEY_copy_parameters EVP_PKEY_COPY_PARAMETERS
#define EVP_PKEY_free EVP_PKEY_FREE
#define EVP_cleanup EVP_CLEANUP
#define GENERAL_NAMES_free GENERAL_NAMES_FREE
#define i2d_X509_PUBKEY I2D_X509_PUBKEY
#define MD4_Final MD4_FINAL
#define MD4_Init MD4_INIT
#define MD4_Update MD4_UPDATE
#define MD5_Final MD5_FINAL
#define MD5_Init MD5_INIT
#define MD5_Update MD5_UPDATE
#define OPENSSL_add_all_algo_noconf OPENSSL_ADD_ALL_ALGO_NOCONF
#ifndef __VAX
#define OPENSSL_load_builtin_modules OPENSSL_LOAD_BUILTIN_MODULES
#endif
#define PEM_read_X509 PEM_READ_X509
#define PEM_write_bio_X509 PEM_WRITE_BIO_X509
#define PKCS12_PBE_add PKCS12_PBE_ADD
#define PKCS12_free PKCS12_FREE
#define PKCS12_parse PKCS12_PARSE
#define RAND_add RAND_ADD
#define RAND_bytes RAND_BYTES
#define RAND_egd RAND_EGD
#define RAND_file_name RAND_FILE_NAME
#define RAND_load_file RAND_LOAD_FILE
#define RAND_status RAND_STATUS
#define SSL_CIPHER_get_name SSL_CIPHER_GET_NAME
#define SSL_CTX_add_client_CA SSL_CTX_ADD_CLIENT_CA
#define SSL_CTX_callback_ctrl SSL_CTX_CALLBACK_CTRL
#define SSL_CTX_check_private_key SSL_CTX_CHECK_PRIVATE_KEY
#define SSL_CTX_ctrl SSL_CTX_CTRL
#define SSL_CTX_free SSL_CTX_FREE
#define SSL_CTX_get_cert_store SSL_CTX_GET_CERT_STORE
#define SSL_CTX_load_verify_locations SSL_CTX_LOAD_VERIFY_LOCATIONS
#define SSL_CTX_new SSL_CTX_NEW
#define SSL_CTX_set_cipher_list SSL_CTX_SET_CIPHER_LIST
#define SSL_CTX_set_def_passwd_cb_ud SSL_CTX_SET_DEF_PASSWD_CB_UD
#define SSL_CTX_set_default_passwd_cb SSL_CTX_SET_DEFAULT_PASSWD_CB
#define SSL_CTX_set_msg_callback SSL_CTX_SET_MSG_CALLBACK
#define SSL_CTX_set_verify SSL_CTX_SET_VERIFY
#define SSL_CTX_use_PrivateKey SSL_CTX_USE_PRIVATEKEY
#define SSL_CTX_use_PrivateKey_file SSL_CTX_USE_PRIVATEKEY_FILE
#define SSL_CTX_use_cert_chain_file SSL_CTX_USE_CERT_CHAIN_FILE
#define SSL_CTX_use_certificate SSL_CTX_USE_CERTIFICATE
#define SSL_CTX_use_certificate_file SSL_CTX_USE_CERTIFICATE_FILE
#define SSL_SESSION_free SSL_SESSION_FREE
#define SSL_connect SSL_CONNECT
#define SSL_free SSL_FREE
#define SSL_get1_session SSL_GET1_SESSION
#define SSL_get_certificate SSL_GET_CERTIFICATE
#define SSL_get_current_cipher SSL_GET_CURRENT_CIPHER
#define SSL_get_error SSL_GET_ERROR
#define SSL_get_peer_cert_chain SSL_GET_PEER_CERT_CHAIN
#define SSL_get_peer_certificate SSL_GET_PEER_CERTIFICATE
#define SSL_get_privatekey SSL_GET_PRIVATEKEY
#define SSL_get_session SSL_GET_SESSION
#define SSL_get_shutdown SSL_GET_SHUTDOWN
#define SSL_get_verify_result SSL_GET_VERIFY_RESULT
#define SSL_library_init SSL_LIBRARY_INIT
#define SSL_load_error_strings SSL_LOAD_ERROR_STRINGS
#define SSL_new SSL_NEW
#define SSL_peek SSL_PEEK
#define SSL_pending SSL_PENDING
#define SSL_read SSL_READ
#define SSL_set_connect_state SSL_SET_CONNECT_STATE
#define SSL_set_fd SSL_SET_FD
#define SSL_set_session SSL_SET_SESSION
#define SSL_shutdown SSL_SHUTDOWN
#define SSL_version SSL_VERSION
#define SSL_write SSL_WRITE
#define SSLeay SSLEAY
#define SSLv23_client_method SSLV23_CLIENT_METHOD
#define SSLv3_client_method SSLV3_CLIENT_METHOD
#define TLSv1_client_method TLSV1_CLIENT_METHOD
#define UI_create_method UI_CREATE_METHOD
#define UI_destroy_method UI_DESTROY_METHOD
#define UI_get0_user_data UI_GET0_USER_DATA
#define UI_get_input_flags UI_GET_INPUT_FLAGS
#define UI_get_string_type UI_GET_STRING_TYPE
#define UI_create_method UI_CREATE_METHOD
#define UI_destroy_method UI_DESTROY_METHOD
#define UI_method_get_closer UI_METHOD_GET_CLOSER
#define UI_method_get_opener UI_METHOD_GET_OPENER
#define UI_method_get_reader UI_METHOD_GET_READER
#define UI_method_get_writer UI_METHOD_GET_WRITER
#define UI_method_set_closer UI_METHOD_SET_CLOSER
#define UI_method_set_opener UI_METHOD_SET_OPENER
#define UI_method_set_reader UI_METHOD_SET_READER
#define UI_method_set_writer UI_METHOD_SET_WRITER
#define UI_OpenSSL UI_OPENSSL
#define UI_set_result UI_SET_RESULT
#define X509V3_EXT_print X509V3_EXT_PRINT
#define X509_EXTENSION_get_critical X509_EXTENSION_GET_CRITICAL
#define X509_EXTENSION_get_data X509_EXTENSION_GET_DATA
#define X509_EXTENSION_get_object X509_EXTENSION_GET_OBJECT
#define X509_LOOKUP_file X509_LOOKUP_FILE
#define X509_NAME_ENTRY_get_data X509_NAME_ENTRY_GET_DATA
#define X509_NAME_get_entry X509_NAME_GET_ENTRY
#define X509_NAME_get_index_by_NID X509_NAME_GET_INDEX_BY_NID
#define X509_NAME_print_ex X509_NAME_PRINT_EX
#define X509_STORE_CTX_get_current_cert X509_STORE_CTX_GET_CURRENT_CERT
#define X509_STORE_add_lookup X509_STORE_ADD_LOOKUP
#define X509_STORE_set_flags X509_STORE_SET_FLAGS
#define X509_check_issued X509_CHECK_ISSUED
#define X509_free X509_FREE
#define X509_get_ext_d2i X509_GET_EXT_D2I
#define X509_get_issuer_name X509_GET_ISSUER_NAME
#define X509_get_pubkey X509_GET_PUBKEY
#define X509_get_serialNumber X509_GET_SERIALNUMBER
#define X509_get_subject_name X509_GET_SUBJECT_NAME
#define X509_load_crl_file X509_LOAD_CRL_FILE
#define X509_verify_cert_error_string X509_VERIFY_CERT_ERROR_STRING
#define d2i_PKCS12_fp D2I_PKCS12_FP
#define i2t_ASN1_OBJECT I2T_ASN1_OBJECT
#define sk_num SK_NUM
#define sk_pop SK_POP
#define sk_pop_free SK_POP_FREE
#define sk_value SK_VALUE
#ifdef __VAX
#define OPENSSL_NO_SHA256
#endif
#define SHA256_Final SHA256_FINAL
#define SHA256_Init SHA256_INIT
#define SHA256_Update SHA256_UPDATE

#define USE_UPPERCASE_GSSAPI 1
#define gss_seal GSS_SEAL
#define gss_unseal GSS_UNSEAL

#define USE_UPPERCASE_KRBAPI 1

/* AI_NUMERICHOST needed for IP V6 support in Curl */
#ifdef HAVE_NETDB_H
#include <netdb.h>
#ifndef AI_NUMERICHOST
#ifdef ENABLE_IPV6
#undef ENABLE_IPV6
#endif
#endif
#endif

/* VAX symbols are always in uppercase */
#ifdef __VAX
#define inflate INFLATE
#define inflateEnd INFLATEEND
#define inflateInit2_ INFLATEINIT2_
#define inflateInit_ INFLATEINIT_
#define zlibVersion ZLIBVERSION
#endif

/* Older VAX OpenSSL port defines these as Macros */
/* Need to include the headers first and then redefine */
/* that way a newer port will also work if some one has one */
#ifdef __VAX

#   if (OPENSSL_VERSION_NUMBER < 0x00907001L)
#       define des_set_odd_parity DES_SET_ODD_PARITY
#       define des_set_key DES_SET_KEY
#       define des_ecb_encrypt DES_ECB_ENCRYPT

#   endif
#   include <openssl/evp.h>
#   ifndef OpenSSL_add_all_algorithms
#       define OpenSSL_add_all_algorithms OPENSSL_ADD_ALL_ALGORITHMS
        void OPENSSL_ADD_ALL_ALGORITHMS(void);
#   endif

    /* Curl defines these to lower case and VAX needs them in upper case */
    /* So we need static routines */
#   if (OPENSSL_VERSION_NUMBER < 0x00907001L)

#       undef des_set_odd_parity
#       undef DES_set_odd_parity
#       undef des_set_key
#       undef DES_set_key
#       undef des_ecb_encrypt
#       undef DES_ecb_encrypt

        static void des_set_odd_parity(des_cblock *key) {
            DES_SET_ODD_PARITY(key);
        }

        static int des_set_key(const_des_cblock *key,
                               des_key_schedule schedule) {
            return DES_SET_KEY(key, schedule);
        }

        static void des_ecb_encrypt(const_des_cblock *input,
                                    des_cblock *output,
                                    des_key_schedule ks, int enc) {
            DES_ECB_ENCRYPT(input, output, ks, enc);
        }
#endif
/* Need this to stop a macro redefinition error */
#if OPENSSL_VERSION_NUMBER < 0x00907000L
#   ifdef X509_STORE_set_flags
#       undef X509_STORE_set_flags
#       define X509_STORE_set_flags(x,y) Curl_nop_stmt
#   endif
#endif
#endif

#endif /* HEADER_CURL_SETUP_VMS_H */
