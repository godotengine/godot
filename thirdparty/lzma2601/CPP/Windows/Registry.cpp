// Windows/Registry.cpp

#include "StdAfx.h"

#include <wchar.h>
// #include <stdio.h>

#ifndef _UNICODE
#include "../Common/StringConvert.h"
#endif
#include "Registry.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {
namespace NRegistry {

#define MYASSERT(expr) // _ASSERTE(expr)
#define MY_ASSUME(expr)

/*
static void Error()
{
  #ifdef _CONSOLE
  printf("\nregistry error\n");
  #else
  MessageBoxW(0, L"registry error", L"", 0);
  // exit(1);
  #endif
}

#define MY_ASSUME(expr) { if (!(expr)) Error(); }
*/

LONG CKey::Create(HKEY parentKey, LPCTSTR keyName,
    LPTSTR keyClass, DWORD options, REGSAM accessMask,
    LPSECURITY_ATTRIBUTES securityAttributes, LPDWORD disposition) throw()
{
  MY_ASSUME(parentKey != NULL);
  DWORD dispositionReal;
  HKEY key = NULL;
  LONG res = RegCreateKeyEx(parentKey, keyName, 0, keyClass,
      options, accessMask, securityAttributes, &key, &dispositionReal);
  if (disposition != NULL)
    *disposition = dispositionReal;
  if (res == ERROR_SUCCESS)
  {
    res = Close();
    _object = key;
  }
  return res;
}

LONG CKey::Open(HKEY parentKey, LPCTSTR keyName, REGSAM accessMask) throw()
{
  MY_ASSUME(parentKey != NULL);
  HKEY key = NULL;
  LONG res = RegOpenKeyEx(parentKey, keyName, 0, accessMask, &key);
  if (res == ERROR_SUCCESS)
  {
    res = Close();
    MYASSERT(res == ERROR_SUCCESS);
    _object = key;
  }
  return res;
}

LONG CKey::Close() throw()
{
  LONG res = ERROR_SUCCESS;
  if (_object != NULL)
  {
    res = RegCloseKey(_object);
    _object = NULL;
  }
  return res;
}

// win95, win98: deletes subkey and all its subkeys
// winNT to be deleted must not have subkeys
LONG CKey::DeleteSubKey(LPCTSTR subKeyName) throw()
{
  MY_ASSUME(_object != NULL);
  return RegDeleteKey(_object, subKeyName);
}

LONG CKey::RecurseDeleteKey(LPCTSTR subKeyName) throw()
{
  {
    CKey key;
    LONG res = key.Open(_object, subKeyName, KEY_READ | KEY_WRITE);
    if (res != ERROR_SUCCESS)
      return res;
    FILETIME fileTime;
    const UInt32 kBufSize = MAX_PATH + 1; // 256 in ATL
    TCHAR buffer[kBufSize];
    // we use loop limit here for some unexpected code failure
    for (unsigned loop_cnt = 0; loop_cnt < (1u << 26); loop_cnt++)
    {
      DWORD size = kBufSize;
      // we always request starting item (index==0) in each iteration,
      // because we remove starting item (index==0) in each loop iteration.
      res = RegEnumKeyEx(key._object, 0, buffer, &size, NULL, NULL, NULL, &fileTime);
      if (res != ERROR_SUCCESS)
      {
        // possible return codes:
        //   ERROR_NO_MORE_ITEMS : are no more subkeys available
        //   ERROR_MORE_DATA     : name buffer is too small
        // we can try to remove (subKeyName), even if there is non ERROR_NO_MORE_ITEMS error.
        // if (res != ERROR_NO_MORE_ITEMS) return res;
        break;
      }
      res = key.RecurseDeleteKey(buffer);
      if (res != ERROR_SUCCESS)
        return res;
    }
    // key.Close();
  }
  return DeleteSubKey(subKeyName);
}


/////////////////////////
// Value Functions

static inline UInt32 BoolToUINT32(bool value) {  return (value ? 1: 0); }
static inline bool UINT32ToBool(UInt32 value) {  return (value != 0); }


LONG CKey::DeleteValue(LPCTSTR name) throw()
{
  MY_ASSUME(_object != NULL);
  return ::RegDeleteValue(_object, name);
}

#ifndef _UNICODE
LONG CKey::DeleteValue(LPCWSTR name)
{
  MY_ASSUME(_object != NULL);
  if (g_IsNT)
    return ::RegDeleteValueW(_object, name);
  return DeleteValue(name == NULL ? NULL : (LPCSTR)GetSystemString(name));
}
#endif

LONG CKey::SetValue(LPCTSTR name, UInt32 value) throw()
{
  MY_ASSUME(_object != NULL);
  return RegSetValueEx(_object, name, 0, REG_DWORD,
      (const BYTE *)&value, sizeof(UInt32));
}

LONG CKey::SetValue(LPCTSTR name, bool value) throw()
{
  return SetValue(name, BoolToUINT32(value));
}


// value must be string that is NULL terminated
LONG CKey::SetValue(LPCTSTR name, LPCTSTR value) throw()
{
  MYASSERT(value != NULL);
  MY_ASSUME(_object != NULL);
  // note: RegSetValueEx supports (value == NULL), if (cbData == 0)
  return RegSetValueEx(_object, name, 0, REG_SZ,
      (const BYTE *)value, (DWORD)(((DWORD)lstrlen(value) + 1) * sizeof(TCHAR)));
}

/*
LONG CKey::SetValue(LPCTSTR name, const CSysString &value)
{
  MYASSERT(value != NULL);
  MY_ASSUME(_object != NULL);
  return RegSetValueEx(_object, name, 0, REG_SZ,
      (const BYTE *)(const TCHAR *)value, (value.Len() + 1) * sizeof(TCHAR));
}
*/

#ifndef _UNICODE

LONG CKey::SetValue(LPCWSTR name, LPCWSTR value)
{
  MYASSERT(value != NULL);
  MY_ASSUME(_object != NULL);
  if (g_IsNT)
    return RegSetValueExW(_object, name, 0, REG_SZ,
        (const BYTE *)value, (DWORD)(((DWORD)wcslen(value) + 1) * sizeof(wchar_t)));
  return SetValue(name == NULL ? NULL :
        (LPCSTR)GetSystemString(name),
        (LPCSTR)GetSystemString(value));
}

#endif


LONG CKey::SetValue(LPCTSTR name, const void *value, UInt32 size) throw()
{
  MYASSERT(value != NULL);
  MY_ASSUME(_object != NULL);
  return RegSetValueEx(_object, name, 0, REG_BINARY,
      (const BYTE *)value, size);
}

LONG SetValue(HKEY parentKey, LPCTSTR keyName, LPCTSTR valueName, LPCTSTR value)
{
  MYASSERT(value != NULL);
  CKey key;
  LONG res = key.Create(parentKey, keyName);
  if (res == ERROR_SUCCESS)
    res = key.SetValue(valueName, value);
  return res;
}

LONG CKey::SetKeyValue(LPCTSTR keyName, LPCTSTR valueName, LPCTSTR value) throw()
{
  MYASSERT(value != NULL);
  CKey key;
  LONG res = key.Create(_object, keyName);
  if (res == ERROR_SUCCESS)
    res = key.SetValue(valueName, value);
  return res;
}


LONG CKey::GetValue_UInt32_IfOk(LPCTSTR name, UInt32 &value) throw()
{
  DWORD type = 0;
  DWORD count = sizeof(value);
  UInt32 value2; // = value;
  const LONG res = QueryValueEx(name, &type, (LPBYTE)&value2, &count);
  if (res == ERROR_SUCCESS)
  {
    // ERROR_UNSUPPORTED_TYPE
    if (count != sizeof(value) || type != REG_DWORD)
      return ERROR_UNSUPPORTED_TYPE; // ERROR_INVALID_DATA;
    value = value2;
  }
  return res;
}

LONG CKey::GetValue_UInt64_IfOk(LPCTSTR name, UInt64 &value) throw()
{
  DWORD type = 0;
  DWORD count = sizeof(value);
  UInt64 value2; // = value;
  const LONG res = QueryValueEx(name, &type, (LPBYTE)&value2, &count);
  if (res == ERROR_SUCCESS)
  {
    if (count != sizeof(value) || type != REG_QWORD)
      return ERROR_UNSUPPORTED_TYPE;
    value = value2;
  }
  return res;
}

LONG CKey::GetValue_bool_IfOk(LPCTSTR name, bool &value) throw()
{
  UInt32 uintValue;
  const LONG res = GetValue_UInt32_IfOk(name, uintValue);
  if (res == ERROR_SUCCESS)
    value = UINT32ToBool(uintValue);
  return res;
}



LONG CKey::QueryValue(LPCTSTR name, CSysString &value)
{
  value.Empty();
  LONG res = ERROR_SUCCESS;
  {
    // if we don't want multiple calls here,
    // we can use big value (264) here.
    // 3 is default available length in new string.
    DWORD size_prev = 3 * sizeof(TCHAR);
    // at least 2 attempts are required. But we use more attempts for cases,
    // where string can be changed by anothner process
    for (unsigned i = 0; i < 2 + 2; i++)
    {
      DWORD type = 0;
      DWORD size = size_prev;
      {
        LPBYTE buf = (LPBYTE)value.GetBuf(size / sizeof(TCHAR));
        res = QueryValueEx(name, &type, size == 0 ? NULL : buf, &size);
        // if (size_prev == 0), then (res == ERROR_SUCCESS) is expected here, because we requested only size.
      }
      if (res == ERROR_SUCCESS || res == ERROR_MORE_DATA)
      {
        if (type != REG_SZ && type != REG_EXPAND_SZ)
        {
          res = ERROR_UNSUPPORTED_TYPE;
          size = 0;
        }
      }
      else
        size = 0;
      if (size > size_prev)
      {
        size_prev = size;
        size = 0;
        res = ERROR_MORE_DATA;
      }
      value.ReleaseBuf_CalcLen(size / sizeof(TCHAR));
      if (res != ERROR_MORE_DATA)
        return res;
    }
  }
  return res;
}


#ifndef _UNICODE

LONG CKey::QueryValue(LPCWSTR name, UString &value)
{
  value.Empty();
  LONG res = ERROR_SUCCESS;
  if (g_IsNT)
  {
    DWORD size_prev = 3 * sizeof(wchar_t);
    for (unsigned i = 0; i < 2 + 2; i++)
    {
      DWORD type = 0;
      DWORD size = size_prev;
      {
        LPBYTE buf = (LPBYTE)value.GetBuf(size / sizeof(wchar_t));
        res = RegQueryValueExW(_object, name, NULL, &type,
            size == 0 ? NULL : buf, &size);
      }
      if (res == ERROR_SUCCESS || res == ERROR_MORE_DATA)
      {
        if (type != REG_SZ && type != REG_EXPAND_SZ)
        {
          res = ERROR_UNSUPPORTED_TYPE;
          size = 0;
        }
      }
      else
        size = 0;
      if (size > size_prev)
      {
        size_prev = size;
        size = 0;
        res = ERROR_MORE_DATA;
      }
      value.ReleaseBuf_CalcLen(size / sizeof(wchar_t));
      if (res != ERROR_MORE_DATA)
        return res;
    }
  }
  else
  {
    AString vTemp;
    res = QueryValue(name == NULL ? NULL : (LPCSTR)GetSystemString(name), vTemp);
    value = GetUnicodeString(vTemp);
  }
  return res;
}

#endif


LONG CKey::QueryValue_Binary(LPCTSTR name, CByteBuffer &value)
{
  // value.Free();
  DWORD size_prev = 0;
  LONG res = ERROR_SUCCESS;
  for (unsigned i = 0; i < 2 + 2; i++)
  {
    DWORD type = 0;
    DWORD size = size_prev;
    value.Alloc(size_prev);
    res = QueryValueEx(name, &type, value.NonConstData(), &size);
    // if (size_prev == 0), then (res == ERROR_SUCCESS) is expected here, because we requested only size.
    if (res == ERROR_SUCCESS || res == ERROR_MORE_DATA)
    {
      if (type != REG_BINARY)
      {
        res = ERROR_UNSUPPORTED_TYPE;
        size = 0;
      }
    }
    else
      size = 0;
    if (size > size_prev)
    {
      size_prev = size;
      size = 0;
      res = ERROR_MORE_DATA;
    }
    if (size < value.Size())
      value.ChangeSize_KeepData(size, size);
    if (res != ERROR_MORE_DATA)
      return res;
  }
  return res;
}


LONG CKey::EnumKeys(CSysStringVector &keyNames)
{
  keyNames.Clear();
  CSysString keyName;
  for (DWORD index = 0; ; index++)
  {
    const unsigned kBufSize = MAX_PATH + 1; // 256 in ATL
    FILETIME lastWriteTime;
    DWORD nameSize = kBufSize;
    const LONG res = ::RegEnumKeyEx(_object, index,
        keyName.GetBuf(kBufSize), &nameSize,
        NULL, NULL, NULL, &lastWriteTime);
    keyName.ReleaseBuf_CalcLen(kBufSize);
    if (res == ERROR_NO_MORE_ITEMS)
      return ERROR_SUCCESS;
    if (res != ERROR_SUCCESS)
      return res;
    keyNames.Add(keyName);
  }
}


LONG CKey::SetValue_Strings(LPCTSTR valueName, const UStringVector &strings)
{
  size_t numChars = 0;
  unsigned i;
  
  for (i = 0; i < strings.Size(); i++)
    numChars += strings[i].Len() + 1;
  
  CObjArray<wchar_t> buffer(numChars);
  size_t pos = 0;
  
  for (i = 0; i < strings.Size(); i++)
  {
    const UString &s = strings[i];
    const size_t size = s.Len() + 1;
    wmemcpy(buffer + pos, s, size);
    pos += size;
  }
  // if (pos != numChars) return E_FAIL;
  return SetValue(valueName, buffer, (UInt32)numChars * sizeof(wchar_t));
}

LONG CKey::GetValue_Strings(LPCTSTR valueName, UStringVector &strings)
{
  strings.Clear();
  CByteBuffer buffer;
  const LONG res = QueryValue_Binary(valueName, buffer);
  if (res != ERROR_SUCCESS)
    return res;
  const size_t dataSize = buffer.Size();
  if (dataSize % sizeof(wchar_t))
    return ERROR_INVALID_DATA;
  const wchar_t *data = (const wchar_t *)(const void *)(const Byte  *)buffer;
  const size_t numChars = dataSize / sizeof(wchar_t);
  // we can check that all names are finished
  // if (numChars != 0 && data[numChars - 1] != 0) return ERROR_INVALID_DATA;
  size_t prev = 0;
  UString s;
  for (size_t i = 0; i < numChars; i++)
  {
    if (data[i] == 0)
    {
      s = data + prev;
      strings.Add(s);
      prev = i + 1;
    }
  }
  return res;
}

}}
