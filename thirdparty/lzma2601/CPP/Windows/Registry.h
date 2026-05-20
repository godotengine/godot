// Windows/Registry.h

#ifndef ZIP7_INC_WINDOWS_REGISTRY_H
#define ZIP7_INC_WINDOWS_REGISTRY_H

#include "../Common/MyBuffer.h"
#include "../Common/MyString.h"

namespace NWindows {
namespace NRegistry {

LONG SetValue(HKEY parentKey, LPCTSTR keyName, LPCTSTR valueName, LPCTSTR value);

class CKey
{
  HKEY _object;

  LONG QueryValueEx(LPCTSTR lpValueName, LPDWORD lpType,
      LPBYTE lpData, LPDWORD lpcbData)
  {
    return RegQueryValueEx(_object, lpValueName, NULL, lpType, lpData, lpcbData);
  }

public:
  CKey(): _object(NULL) {}
  ~CKey() { Close(); }

  operator HKEY() const { return _object; }
  void Attach(HKEY key) { _object = key; }
  HKEY Detach()
  {
    const HKEY key = _object;
    _object = NULL;
    return key;
  }

  LONG Create(HKEY parentKey, LPCTSTR keyName,
      LPTSTR keyClass = REG_NONE,
      DWORD options = REG_OPTION_NON_VOLATILE,
      REGSAM accessMask = KEY_ALL_ACCESS,
      LPSECURITY_ATTRIBUTES securityAttributes = NULL,
      LPDWORD disposition = NULL) throw();
  LONG Open(HKEY parentKey, LPCTSTR keyName, REGSAM accessMask = KEY_ALL_ACCESS) throw();

  LONG Close() throw();

  LONG DeleteSubKey(LPCTSTR subKeyName) throw();
  LONG RecurseDeleteKey(LPCTSTR subKeyName) throw();

  LONG DeleteValue(LPCTSTR name) throw();
#ifndef _UNICODE
  LONG DeleteValue(LPCWSTR name);
#endif

  LONG SetValue(LPCTSTR valueName, UInt32 value) throw();
  LONG SetValue(LPCTSTR valueName, bool value) throw();
  LONG SetValue(LPCTSTR valueName, LPCTSTR value) throw();
  // LONG SetValue(LPCTSTR valueName, const CSysString &value);
#ifndef _UNICODE
  LONG SetValue(LPCWSTR name, LPCWSTR value);
  // LONG SetValue(LPCWSTR name, const UString &value);
#endif

  LONG SetValue(LPCTSTR name, const void *value, UInt32 size) throw();

  LONG SetValue_Strings(LPCTSTR valueName, const UStringVector &strings);
  LONG GetValue_Strings(LPCTSTR valueName, UStringVector &strings);

  LONG SetKeyValue(LPCTSTR keyName, LPCTSTR valueName, LPCTSTR value) throw();

  // GetValue_[type]_IfOk():
  //   if (return_result == ERROR_SUCCESS), (value) variable was read from registry
  //   if (return_result != ERROR_SUCCESS), (value) variable was not changed
  LONG GetValue_UInt32_IfOk(LPCTSTR name, UInt32 &value) throw();
  LONG GetValue_UInt64_IfOk(LPCTSTR name, UInt64 &value) throw();
  LONG GetValue_bool_IfOk(LPCTSTR name, bool &value) throw();

  // QueryValue():
  //   if (return_result == ERROR_SUCCESS), (value) string was read from registry
  //   if (return_result != ERROR_SUCCESS), (value) string was cleared
  LONG QueryValue(LPCTSTR name, CSysString &value);
#ifndef _UNICODE
  LONG QueryValue(LPCWSTR name, UString &value);
#endif

  // QueryValue_Binary():
  //   if (return_result == ERROR_SUCCESS), (value) buffer was read from registry (BINARY data)
  //   if (return_result != ERROR_SUCCESS), (value) buffer was cleared
  LONG QueryValue_Binary(LPCTSTR name, CByteBuffer &value);

  LONG EnumKeys(CSysStringVector &keyNames);
};

}}

#endif
