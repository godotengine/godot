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
#include "curl_sspi.h"
#include "curl_multibyte.h"
#include "system_win32.h"
#include "version_win32.h"
#include "warnless.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/* We use our own typedef here since some headers might lack these */
typedef PSecurityFunctionTable (APIENTRY *INITSECURITYINTERFACE_FN)(VOID);

/* See definition of SECURITY_ENTRYPOINT in sspi.h */
#ifdef UNICODE
#  ifdef _WIN32_WCE
#    define SECURITYENTRYPOINT L"InitSecurityInterfaceW"
#  else
#    define SECURITYENTRYPOINT "InitSecurityInterfaceW"
#  endif
#else
#  define SECURITYENTRYPOINT "InitSecurityInterfaceA"
#endif

/* Handle of security.dll or secur32.dll, depending on Windows version */
HMODULE s_hSecDll = NULL;

/* Pointer to SSPI dispatch table */
PSecurityFunctionTable s_pSecFn = NULL;

/*
 * Curl_sspi_global_init()
 *
 * This is used to load the Security Service Provider Interface (SSPI)
 * dynamic link library portably across all Windows versions, without
 * the need to directly link libcurl, nor the application using it, at
 * build time.
 *
 * Once this function has been executed, Windows SSPI functions can be
 * called through the Security Service Provider Interface dispatch table.
 *
 * Parameters:
 *
 * None.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_sspi_global_init(void)
{
  INITSECURITYINTERFACE_FN pInitSecurityInterface;

  /* If security interface is not yet initialized try to do this */
  if(!s_hSecDll) {
    /* Security Service Provider Interface (SSPI) functions are located in
     * security.dll on WinNT 4.0 and in secur32.dll on Win9x. Win2K and XP
     * have both these DLLs (security.dll forwards calls to secur32.dll) */

    /* Load SSPI dll into the address space of the calling process */
    if(curlx_verify_windows_version(4, 0, 0, PLATFORM_WINNT, VERSION_EQUAL))
      s_hSecDll = Curl_load_library(TEXT("security.dll"));
    else
      s_hSecDll = Curl_load_library(TEXT("secur32.dll"));
    if(!s_hSecDll)
      return CURLE_FAILED_INIT;

    /* Get address of the InitSecurityInterfaceA function from the SSPI dll */
    pInitSecurityInterface =
      CURLX_FUNCTION_CAST(INITSECURITYINTERFACE_FN,
                          (GetProcAddress(s_hSecDll, SECURITYENTRYPOINT)));
    if(!pInitSecurityInterface)
      return CURLE_FAILED_INIT;

    /* Get pointer to Security Service Provider Interface dispatch table */
    s_pSecFn = pInitSecurityInterface();
    if(!s_pSecFn)
      return CURLE_FAILED_INIT;
  }

  return CURLE_OK;
}

/*
 * Curl_sspi_global_cleanup()
 *
 * This deinitializes the Security Service Provider Interface from libcurl.
 *
 * Parameters:
 *
 * None.
 */
void Curl_sspi_global_cleanup(void)
{
  if(s_hSecDll) {
    FreeLibrary(s_hSecDll);
    s_hSecDll = NULL;
    s_pSecFn = NULL;
  }
}

/*
 * Curl_create_sspi_identity()
 *
 * This is used to populate a SSPI identity structure based on the supplied
 * username and password.
 *
 * Parameters:
 *
 * userp    [in]     - The user name in the format User or Domain\User.
 * passwdp  [in]     - The user's password.
 * identity [in/out] - The identity structure.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_create_sspi_identity(const char *userp, const char *passwdp,
                                   SEC_WINNT_AUTH_IDENTITY *identity)
{
  xcharp_u useranddomain;
  xcharp_u user, dup_user;
  xcharp_u domain, dup_domain;
  xcharp_u passwd, dup_passwd;
  size_t domlen = 0;

  domain.const_tchar_ptr = TEXT("");

  /* Initialize the identity */
  memset(identity, 0, sizeof(*identity));

  useranddomain.tchar_ptr = curlx_convert_UTF8_to_tchar((char *)userp);
  if(!useranddomain.tchar_ptr)
    return CURLE_OUT_OF_MEMORY;

  user.const_tchar_ptr = _tcschr(useranddomain.const_tchar_ptr, TEXT('\\'));
  if(!user.const_tchar_ptr)
    user.const_tchar_ptr = _tcschr(useranddomain.const_tchar_ptr, TEXT('/'));

  if(user.tchar_ptr) {
    domain.tchar_ptr = useranddomain.tchar_ptr;
    domlen = user.tchar_ptr - useranddomain.tchar_ptr;
    user.tchar_ptr++;
  }
  else {
    user.tchar_ptr = useranddomain.tchar_ptr;
    domain.const_tchar_ptr = TEXT("");
    domlen = 0;
  }

  /* Setup the identity's user and length */
  dup_user.tchar_ptr = _tcsdup(user.tchar_ptr);
  if(!dup_user.tchar_ptr) {
    curlx_unicodefree(useranddomain.tchar_ptr);
    return CURLE_OUT_OF_MEMORY;
  }
  identity->User = dup_user.tbyte_ptr;
  identity->UserLength = curlx_uztoul(_tcslen(dup_user.tchar_ptr));
  dup_user.tchar_ptr = NULL;

  /* Setup the identity's domain and length */
  dup_domain.tchar_ptr = malloc(sizeof(TCHAR) * (domlen + 1));
  if(!dup_domain.tchar_ptr) {
    curlx_unicodefree(useranddomain.tchar_ptr);
    return CURLE_OUT_OF_MEMORY;
  }
  _tcsncpy(dup_domain.tchar_ptr, domain.tchar_ptr, domlen);
  *(dup_domain.tchar_ptr + domlen) = TEXT('\0');
  identity->Domain = dup_domain.tbyte_ptr;
  identity->DomainLength = curlx_uztoul(domlen);
  dup_domain.tchar_ptr = NULL;

  curlx_unicodefree(useranddomain.tchar_ptr);

  /* Setup the identity's password and length */
  passwd.tchar_ptr = curlx_convert_UTF8_to_tchar((char *)passwdp);
  if(!passwd.tchar_ptr)
    return CURLE_OUT_OF_MEMORY;
  dup_passwd.tchar_ptr = _tcsdup(passwd.tchar_ptr);
  if(!dup_passwd.tchar_ptr) {
    curlx_unicodefree(passwd.tchar_ptr);
    return CURLE_OUT_OF_MEMORY;
  }
  identity->Password = dup_passwd.tbyte_ptr;
  identity->PasswordLength = curlx_uztoul(_tcslen(dup_passwd.tchar_ptr));
  dup_passwd.tchar_ptr = NULL;

  curlx_unicodefree(passwd.tchar_ptr);

  /* Setup the identity's flags */
  identity->Flags = SECFLAG_WINNT_AUTH_IDENTITY;

  return CURLE_OK;
}

/*
 * Curl_sspi_free_identity()
 *
 * This is used to free the contents of a SSPI identifier structure.
 *
 * Parameters:
 *
 * identity [in/out] - The identity structure.
 */
void Curl_sspi_free_identity(SEC_WINNT_AUTH_IDENTITY *identity)
{
  if(identity) {
    Curl_safefree(identity->User);
    Curl_safefree(identity->Password);
    Curl_safefree(identity->Domain);
  }
}

#endif /* USE_WINDOWS_SSPI */
