// Windows/SecurityUtils.cpp

#include "StdAfx.h"

#include "SecurityUtils.h"

namespace NWindows {
namespace NSecurity {

/*
bool MyLookupAccountSid(LPCTSTR systemName, PSID sid,
  CSysString &accountName, CSysString &domainName, PSID_NAME_USE sidNameUse)
{
  DWORD accountNameSize = 0, domainNameSize = 0;

  if (!::LookupAccountSid(systemName, sid,
      accountName.GetBuf(0), &accountNameSize,
      domainName.GetBuf(0), &domainNameSize, sidNameUse))
  {
    if (::GetLastError() != ERROR_INSUFFICIENT_BUFFER)
      return false;
  }
  DWORD accountNameSize2 = accountNameSize, domainNameSize2 = domainNameSize;
  bool result = BOOLToBool(::LookupAccountSid(systemName, sid,
      accountName.GetBuf(accountNameSize), &accountNameSize2,
      domainName.GetBuf(domainNameSize), &domainNameSize2, sidNameUse));
  accountName.ReleaseBuf_CalcLen(accountNameSize);
  domainName.ReleaseBuf_CalcLen(domainNameSize);
  return result;
}
*/
  
static void SetLsaString(LPWSTR src, PLSA_UNICODE_STRING dest)
{
  const size_t len = (size_t)wcslen(src);
  dest->Length = (USHORT)(len * sizeof(WCHAR));
  dest->MaximumLength = (USHORT)((len + 1) * sizeof(WCHAR));
  dest->Buffer = src;
}

/*
static void MyLookupSids(CPolicy &policy, PSID ps)
{
  LSA_REFERENCED_DOMAIN_LIST *referencedDomains = NULL;
  LSA_TRANSLATED_NAME *names = NULL;
  NTSTATUS nts = policy.LookupSids(1, &ps, &referencedDomains, &names);
  int res = LsaNtStatusToWinError(nts);
  LsaFreeMemory(referencedDomains);
  LsaFreeMemory(names);
}
*/

extern "C" {

#ifndef _UNICODE
typedef BOOL (WINAPI * Func_LookupAccountNameW)(
    LPCWSTR lpSystemName,
    LPCWSTR lpAccountName,
    PSID Sid,
    LPDWORD cbSid,
    LPWSTR ReferencedDomainName,
    LPDWORD cchReferencedDomainName,
    PSID_NAME_USE peUse
    );
#endif

}

static PSID GetSid(LPWSTR accountName)
{
  #ifndef _UNICODE
  const HMODULE hModule = GetModuleHandle(TEXT("advapi32.dll"));
  if (!hModule)
    return NULL;
  const
  Func_LookupAccountNameW lookupAccountNameW = Z7_GET_PROC_ADDRESS(
  Func_LookupAccountNameW, hModule,
      "LookupAccountNameW");
  if (!lookupAccountNameW)
    return NULL;
  #endif

  DWORD sidLen = 0, domainLen = 0;
  SID_NAME_USE sidNameUse;
  if (!
    #ifdef _UNICODE
    ::LookupAccountNameW
    #else
      lookupAccountNameW
    #endif
        (NULL, accountName, NULL, &sidLen, NULL, &domainLen, &sidNameUse))
  {
    if (::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
    {
      const PSID pSid = ::HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sidLen);
      LPWSTR domainName = (LPWSTR)::HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, (domainLen + 1) * sizeof(WCHAR));
      const BOOL res =
        #ifdef _UNICODE
        ::LookupAccountNameW
        #else
          lookupAccountNameW
        #endif
            (NULL, accountName, pSid, &sidLen, domainName, &domainLen, &sidNameUse);
      ::HeapFree(GetProcessHeap(), 0, domainName);
      if (res)
        return pSid;
    }
  }
  return NULL;
}

#define Z7_WIN_SE_LOCK_MEMORY_NAME L"SeLockMemoryPrivilege"

bool AddLockMemoryPrivilege()
{
  CPolicy policy;
  LSA_OBJECT_ATTRIBUTES attr;
  attr.Length = sizeof(attr);
  attr.RootDirectory = NULL;
  attr.ObjectName  = NULL;
  attr.Attributes = 0;
  attr.SecurityDescriptor = NULL;
  attr.SecurityQualityOfService  = NULL;
  if (policy.Open(NULL, &attr,
      // GENERIC_WRITE)
      POLICY_ALL_ACCESS)
      // STANDARD_RIGHTS_REQUIRED,
      // GENERIC_READ | GENERIC_EXECUTE | POLICY_VIEW_LOCAL_INFORMATION | POLICY_LOOKUP_NAMES)
      != 0)
    return false;
  LSA_UNICODE_STRING userRights;
  wchar_t s[128] = Z7_WIN_SE_LOCK_MEMORY_NAME;
  SetLsaString(s, &userRights);
  WCHAR userName[256 + 2];
  DWORD size = 256;
  if (!GetUserNameW(userName, &size))
    return false;
  const PSID psid = GetSid(userName);
  if (psid == NULL)
    return false;
  bool res = false;

  /*
  PLSA_UNICODE_STRING userRightsArray;
  ULONG countOfRights;
  NTSTATUS status = policy.EnumerateAccountRights(psid, &userRightsArray, &countOfRights);
  if (status != 0)
    return false;
  bool finded = false;
  for (ULONG i = 0; i < countOfRights; i++)
  {
    LSA_UNICODE_STRING &ur = userRightsArray[i];
    if (ur.Length != s.Length() * sizeof(WCHAR))
      continue;
    if (wcsncmp(ur.Buffer, s, s.Length()) != 0)
      continue;
    finded = true;
    res = true;
    break;
  }
  if (!finded)
  */
  {
    /*
    LSA_ENUMERATION_INFORMATION *enums;
    ULONG countReturned;
    NTSTATUS status = policy.EnumerateAccountsWithUserRight(&userRights, &enums, &countReturned);
    if (status == 0)
    {
      for (ULONG i = 0; i < countReturned; i++)
        MyLookupSids(policy, enums[i].Sid);
      if (enums)
        ::LsaFreeMemory(enums);
      res = true;
    }
    */
    const NTSTATUS status = policy.AddAccountRights(psid, &userRights);
    if (status == 0)
      res = true;
    // ULONG res = LsaNtStatusToWinError(status);
  }
  HeapFree(GetProcessHeap(), 0, psid);
  return res;
}

}}
