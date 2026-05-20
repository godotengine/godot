// Windows/SecurityUtils.h

#ifndef ZIP7_INC_WINDOWS_SECURITY_UTILS_H
#define ZIP7_INC_WINDOWS_SECURITY_UTILS_H

#if defined(__MINGW32__) || defined(__MINGW64__)
#include <ntsecapi.h>
#else
#include <NTSecAPI.h>
#endif

#include "Defs.h"

#ifndef _UNICODE

extern "C" {
typedef NTSTATUS (NTAPI *Func_LsaOpenPolicy)(PLSA_UNICODE_STRING SystemName,
    PLSA_OBJECT_ATTRIBUTES ObjectAttributes, ACCESS_MASK DesiredAccess, PLSA_HANDLE PolicyHandle);
typedef NTSTATUS (NTAPI *Func_LsaClose)(LSA_HANDLE ObjectHandle);
typedef NTSTATUS (NTAPI *Func_LsaAddAccountRights)(LSA_HANDLE PolicyHandle,
    PSID AccountSid, PLSA_UNICODE_STRING UserRights, ULONG CountOfRights );
#define MY_STATUS_NOT_IMPLEMENTED  ((NTSTATUS)0xC0000002L)
}

Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

#define POLICY_FUNC_CALL(fff, str)  \
  if (hModule == NULL) return MY_STATUS_NOT_IMPLEMENTED; \
  const Func_ ## fff v = Z7_GET_PROC_ADDRESS(Func_ ## fff, hModule, str); \
  if (!v) return MY_STATUS_NOT_IMPLEMENTED; \
  const NTSTATUS res = v

#else

#define POLICY_FUNC_CALL(fff, str)  \
  const NTSTATUS res = ::fff

#endif


namespace NWindows {
namespace NSecurity {

class CAccessToken
{
  HANDLE _handle;
public:
  CAccessToken(): _handle(NULL) {}
  ~CAccessToken() { Close(); }
  bool Close()
  {
    if (_handle == NULL)
      return true;
    bool res = BOOLToBool(::CloseHandle(_handle));
    if (res)
      _handle = NULL;
    return res;
  }

  bool OpenProcessToken(HANDLE processHandle, DWORD desiredAccess)
  {
    Close();
    return BOOLToBool(::OpenProcessToken(processHandle, desiredAccess, &_handle));
  }

  /*
  bool OpenThreadToken(HANDLE threadHandle, DWORD desiredAccess, bool openAsSelf)
  {
    Close();
    return BOOLToBool(::OpenTreadToken(threadHandle, desiredAccess, BoolToBOOL(anOpenAsSelf), &_handle));
  }
  */

  bool AdjustPrivileges(bool disableAllPrivileges, PTOKEN_PRIVILEGES newState,
      DWORD bufferLength, PTOKEN_PRIVILEGES previousState, PDWORD returnLength)
    { return BOOLToBool(::AdjustTokenPrivileges(_handle, BoolToBOOL(disableAllPrivileges),
      newState, bufferLength, previousState, returnLength)); }
  
  bool AdjustPrivileges(bool disableAllPrivileges, PTOKEN_PRIVILEGES newState)
    { return AdjustPrivileges(disableAllPrivileges, newState, 0, NULL, NULL); }
  
  bool AdjustPrivileges(PTOKEN_PRIVILEGES newState)
    { return AdjustPrivileges(false, newState); }

};



    
struct CPolicy
{
protected:
  LSA_HANDLE _handle;
  #ifndef _UNICODE
  HMODULE hModule;
  #endif
public:
  operator LSA_HANDLE() const { return _handle; }
  CPolicy(): _handle(NULL)
  {
    #ifndef _UNICODE
    hModule = GetModuleHandle(TEXT("advapi32.dll"));
    #endif
  }
  ~CPolicy() { Close(); }

  NTSTATUS Open(PLSA_UNICODE_STRING systemName, PLSA_OBJECT_ATTRIBUTES objectAttributes,
      ACCESS_MASK desiredAccess)
  {
    Close();
    POLICY_FUNC_CALL (LsaOpenPolicy, "LsaOpenPolicy")
      (systemName, objectAttributes, desiredAccess, &_handle);
    return res;
  }
  
  NTSTATUS Close()
  {
    if (_handle == NULL)
      return 0;
    POLICY_FUNC_CALL (LsaClose, "LsaClose")
      (_handle);
    _handle = NULL;
    return res;
  }
  
  NTSTATUS EnumerateAccountsWithUserRight(PLSA_UNICODE_STRING userRights,
      PLSA_ENUMERATION_INFORMATION *enumerationBuffer, PULONG countReturned)
    { return LsaEnumerateAccountsWithUserRight(_handle, userRights, (void **)enumerationBuffer, countReturned); }

  NTSTATUS EnumerateAccountRights(PSID sid, PLSA_UNICODE_STRING* userRights, PULONG countOfRights)
    { return ::LsaEnumerateAccountRights(_handle, sid, userRights, countOfRights); }

  NTSTATUS LookupSids(ULONG count, PSID* sids,
      PLSA_REFERENCED_DOMAIN_LIST* referencedDomains, PLSA_TRANSLATED_NAME* names)
    { return LsaLookupSids(_handle, count, sids, referencedDomains, names); }

  NTSTATUS AddAccountRights(PSID accountSid, PLSA_UNICODE_STRING userRights, ULONG countOfRights)
  {
    POLICY_FUNC_CALL (LsaAddAccountRights, "LsaAddAccountRights")
      (_handle, accountSid, userRights, countOfRights);
    return res;
  }
  NTSTATUS AddAccountRights(PSID accountSid, PLSA_UNICODE_STRING userRights)
    { return AddAccountRights(accountSid, userRights, 1); }

  NTSTATUS RemoveAccountRights(PSID accountSid, bool allRights, PLSA_UNICODE_STRING userRights, ULONG countOfRights)
    { return LsaRemoveAccountRights(_handle, accountSid, (BOOLEAN)(allRights ? TRUE : FALSE), userRights, countOfRights); }
};

bool AddLockMemoryPrivilege();

}}

#endif
