// DLLExports2.cpp

#include "StdAfx.h"

#include "../../Common/MyWindows.h"
#include "../../Common/MyInitGuid.h"

#if defined(Z7_LARGE_PAGES)
#include "../../../C/Alloc.h"
#endif

#include "../../Common/ComTry.h"

#include "../../Windows/NtCheck.h"
#include "../../Windows/PropVariant.h"

#include "../ICoder.h"
#include "../IPassword.h"

#include "../Common/CreateCoder.h"

#include "IArchive.h"


#ifdef _WIN32

#if defined(_UNICODE) && !defined(_WIN64) && !defined(UNDER_CE)
#define NT_CHECK_FAIL_ACTION return FALSE;
#endif

static
HINSTANCE g_hInstance;

extern "C"
BOOL WINAPI DllMain(
  #ifdef UNDER_CE
  HANDLE
  #else
  HINSTANCE
  #endif
  hInstance, DWORD dwReason, LPVOID /*lpReserved*/);

extern "C"
BOOL WINAPI DllMain(
  #ifdef UNDER_CE
  HANDLE
  #else
  HINSTANCE
  #endif
  hInstance, DWORD dwReason, LPVOID /*lpReserved*/)
{
  if (dwReason == DLL_PROCESS_ATTACH)
  {
    // OutputDebugStringA("7z.dll DLL_PROCESS_ATTACH");
    g_hInstance = (HINSTANCE)hInstance;
    NT_CHECK
  }
  /*
  if (dwReason == DLL_PROCESS_DETACH)
  {
    OutputDebugStringA("7z.dll DLL_PROCESS_DETACH");
  }
  */
  return TRUE;
}

#else //  _WIN32

#include "../../Common/StringConvert.h"
// #include <stdio.h>

// STDAPI LibStartup();
static __attribute__((constructor)) void Init_ForceToUTF8();
static __attribute__((constructor)) void Init_ForceToUTF8()
{
  g_ForceToUTF8 = IsNativeUTF8();
  // printf("\nDLLExports2.cpp::Init_ForceToUTF8 =%d\n", g_ForceToUTF8 ? 1 : 0);
}

#endif // _WIN32


Z7_DEFINE_GUID(CLSID_CArchiveHandler,
    k_7zip_GUID_Data1,
    k_7zip_GUID_Data2,
    k_7zip_GUID_Data3_Common,
    0x10, 0x00, 0x00, 0x01, 0x10, 0x00, 0x00, 0x00);

STDAPI CreateCoder(const GUID *clsid, const GUID *iid, void **outObject);
STDAPI CreateHasher(const GUID *clsid, IHasher **hasher);
STDAPI CreateArchiver(const GUID *clsid, const GUID *iid, void **outObject);

STDAPI CreateObject(const GUID *clsid, const GUID *iid, void **outObject);
STDAPI CreateObject(const GUID *clsid, const GUID *iid, void **outObject)
{
  // COM_TRY_BEGIN
  *outObject = NULL;
  if (*iid == IID_ICompressCoder ||
      *iid == IID_ICompressCoder2 ||
      *iid == IID_ICompressFilter)
    return CreateCoder(clsid, iid, outObject);
  if (*iid == IID_IHasher)
    return CreateHasher(clsid, (IHasher **)outObject);
  return CreateArchiver(clsid, iid, outObject);
  // COM_TRY_END
}

STDAPI SetLargePageMode2(UInt32 flags, size_t pageSize, size_t threshold);
STDAPI SetLargePageMode2(UInt32 flags, size_t pageSize, size_t threshold)
{
  #ifdef Z7_LARGE_PAGES
  if (pageSize & (pageSize - 1))
    return E_INVALIDARG;
  z7_LargePage_Set(flags, pageSize, threshold);
  #else
  UNUSED_VAR(flags)
  UNUSED_VAR(pageSize)
  UNUSED_VAR(threshold)
  #endif
  return S_OK;
}

STDAPI SetLargePageMode();
STDAPI SetLargePageMode()
{
  return SetLargePageMode2(0, 0, 0); // default values
}

extern bool g_CaseSensitive;

STDAPI SetCaseSensitive(Int32 caseSensitive);
STDAPI SetCaseSensitive(Int32 caseSensitive)
{
  g_CaseSensitive = (caseSensitive != 0);
  return S_OK;
}

/*
UInt32 g_ClientVersion;
STDAPI SetClientVersion(UInt32 version);
STDAPI SetClientVersion(UInt32 version)
{
  g_ClientVersion = version;
  return S_OK;
}
*/

/*
STDAPI SetProperty(Int32 id, const PROPVARIANT *value);
STDAPI SetProperty(Int32 id, const PROPVARIANT *value)
{
  return S_OK;
}
*/

#ifdef Z7_EXTERNAL_CODECS

CExternalCodecs g_ExternalCodecs;

STDAPI SetCodecs(ICompressCodecsInfo *compressCodecsInfo);
STDAPI SetCodecs(ICompressCodecsInfo *compressCodecsInfo)
{
  COM_TRY_BEGIN

  // OutputDebugStringA(compressCodecsInfo ? "SetCodecs" : "SetCodecs NULL");
  if (compressCodecsInfo)
  {
    g_ExternalCodecs.GetCodecs = compressCodecsInfo;
    return g_ExternalCodecs.Load();
  }
  g_ExternalCodecs.ClearAndRelease();
  return S_OK;

  COM_TRY_END
}

#else

STDAPI SetCodecs(ICompressCodecsInfo *);
STDAPI SetCodecs(ICompressCodecsInfo *)
{
  return S_OK;
}

#endif
