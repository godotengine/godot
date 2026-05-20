// Windows/Shell.cpp

#include "StdAfx.h"

#include "../Common/MyCom.h"
#include "../Common/StringConvert.h"

#include "COM.h"
#include "FileName.h"
#include "MemoryGlobal.h"
#include "Shell.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

// MSVC6 and old SDK don't support this function:
// #define LWSTDAPI  EXTERN_C DECLSPEC_IMPORT HRESULT STDAPICALLTYPE
// LWSTDAPI StrRetToStrW(STRRET *pstr, LPCITEMIDLIST pidl, LPWSTR *ppsz);

// #define SHOW_DEBUG_SHELL

#ifdef SHOW_DEBUG_SHELL

#include "../Common/IntToString.h"

static void Print_Number(UInt32 number, const char *s)
{
  AString s2;
  s2.Add_UInt32(number);
  s2.Add_Space();
  s2 += s;
  OutputDebugStringA(s2);
}

#define ODS(sz) { OutputDebugStringA(sz); }
#define ODS_U(s) { OutputDebugStringW(s); }
#define ODS_(op) { op; }

#else

#define ODS(sz)
#define ODS_U(s)
#define ODS_(op)

#endif


namespace NWindows {
namespace NShell {

#ifndef UNDER_CE

// SHGetMalloc is unsupported in Windows Mobile?

void CItemIDList::Free()
{
  if (!m_Object)
    return;
  /* DOCs:
      SHGetMalloc was introduced in Windows 95 and Microsoft Windows NT 4.0,
      but as of Windows 2000 it is no longer necessary.
      In its place, programs can call the equivalent (and easier to use) CoTaskMemAlloc and CoTaskMemFree.
     Description from oldnewthings:
       shell functions could work without COM (if OLE32.DLL is not loaded),
       but now if OLE32.DLL is loaded, then shell functions and com functions do same things.
     22.02: so we use OLE32.DLL function to free memory:
  */
  /*
  CMyComPtr<IMalloc> shellMalloc;
  if (::SHGetMalloc(&shellMalloc) != NOERROR)
    throw 41099;
  shellMalloc->Free(m_Object);
  */
  CoTaskMemFree(m_Object);
  m_Object = NULL;
}

/*
CItemIDList::(LPCITEMIDLIST itemIDList): m_Object(NULL)
  {  *this = itemIDList; }
CItemIDList::(const CItemIDList& itemIDList): m_Object(NULL)
  {  *this = itemIDList; }

CItemIDList& CItemIDList::operator=(LPCITEMIDLIST object)
{
  Free();
  if (object != 0)
  {
    UINT32 size = GetSize(object);
    m_Object = (LPITEMIDLIST)CoTaskMemAlloc(size);
    if (m_Object != NULL)
      MoveMemory(m_Object, object, size);
  }
  return *this;
}

CItemIDList& CItemIDList::operator=(const CItemIDList &object)
{
  Free();
  if (object.m_Object != NULL)
  {
    UINT32 size = GetSize(object.m_Object);
    m_Object = (LPITEMIDLIST)CoTaskMemAlloc(size);
    if (m_Object != NULL)
      MoveMemory(m_Object, object.m_Object, size);
  }
  return *this;
}
*/


static HRESULT ReadUnicodeStrings(const wchar_t *p, size_t size, UStringVector &names)
{
  names.Clear();
  const wchar_t *lim = p + size;
  UString s;
  /*
  if (size == 0 || p[size - 1] != 0)
    return E_INVALIDARG;
  if (size == 1)
    return S_OK;
  if (p[size - 2] != 0)
    return E_INVALIDARG;
  */
  for (;;)
  {
    const wchar_t *start = p;
    for (;;)
    {
      if (p == lim) return E_INVALIDARG; // S_FALSE
      if (*p++ == 0)
        break;
    }
    const size_t num = (size_t)(p - start);
    if (num == 1)
    {
      if (p != lim) return E_INVALIDARG; // S_FALSE
      return S_OK;
    }
    s.SetFrom(start, (unsigned)(num - 1));
    ODS_U(s)
    names.Add(s);
    // names.ReserveOnePosition();
    // names.AddInReserved_Ptr_of_new(new UString((unsigned)num - 1, start));
  }
}


static HRESULT ReadAnsiStrings(const char *p, size_t size, UStringVector &names)
{
  names.Clear();
  AString name;
  for (; size != 0; size--)
  {
    const char c = *p++;
    if (c == 0)
    {
      if (name.IsEmpty())
        return S_OK;
      names.Add(GetUnicodeString(name));
      name.Empty();
    }
    else
      name.Add_Char(c);
  }
  return E_INVALIDARG;
}


#define INIT_FORMATETC_HGLOBAL(type) { (type), NULL, DVASPECT_CONTENT, -1, TYMED_HGLOBAL }

static HRESULT DataObject_GetData_HGLOBAL(IDataObject *dataObject, CLIPFORMAT cf, NCOM::CStgMedium &medium)
{
  FORMATETC etc = INIT_FORMATETC_HGLOBAL(cf);
  RINOK(dataObject->GetData(&etc, &medium))
  if (medium.tymed != TYMED_HGLOBAL)
    return E_INVALIDARG;
  return S_OK;
}

static HRESULT DataObject_GetData_HDROP_Names(IDataObject *dataObject, UStringVector &names)
{
  names.Clear();
  NCOM::CStgMedium medium;
  
  /* Win10 : if (dataObject) is from IContextMenu::Initialize() and
    if (len_of_path >= MAX_PATH (260) for some file in data object)
    {
      GetData() returns HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)
        "The data area passed to a system call is too small",
      Is there a way to fix this code for long paths?
    } */

  RINOK(DataObject_GetData_HGLOBAL(dataObject, CF_HDROP, medium))
  const size_t blockSize = GlobalSize(medium.hGlobal);
  if (blockSize < sizeof(DROPFILES))
    return E_INVALIDARG;
  NMemory::CGlobalLock dropLock(medium.hGlobal);
  const DROPFILES *dropFiles = (const DROPFILES *)dropLock.GetPointer();
  if (!dropFiles)
    return E_INVALIDARG;
  if (blockSize < dropFiles->pFiles
      || dropFiles->pFiles < sizeof(DROPFILES)
      // || dropFiles->pFiles != sizeof(DROPFILES)
      )
    return E_INVALIDARG;
  const size_t size = blockSize - dropFiles->pFiles;
  const void *namesData = (const Byte *)(const void *)dropFiles + dropFiles->pFiles;
  HRESULT hres;
  if (dropFiles->fWide)
  {
    if (size % sizeof(wchar_t) != 0)
      return E_INVALIDARG;
    hres = ReadUnicodeStrings((const wchar_t *)namesData, size / sizeof(wchar_t), names);
  }
  else
    hres = ReadAnsiStrings((const char *)namesData, size, names);

  ODS_(Print_Number(names.Size(), "DataObject_GetData_HDROP_Names"))
  return hres;
}



// CF_IDLIST:
#define MYWIN_CFSTR_SHELLIDLIST  TEXT("Shell IDList Array")

typedef struct
{
  UINT cidl;
  UINT aoffset[1];
} MYWIN_CIDA;
/*
  cidl : number of PIDLs that are being transferred, not including the parent folder.
  aoffset : An array of offsets, relative to the beginning of this structure.
  aoffset[0] - fully qualified PIDL of a parent folder.
               If this PIDL is empty, the parent folder is the desktop.
  aoffset[1] ... aoffset[cidl] : offset to one of the PIDLs to be transferred.
  All of these PIDLs are relative to the PIDL of the parent folder.
*/

static HRESULT DataObject_GetData_IDLIST(IDataObject *dataObject, UStringVector &names)
{
  names.Clear();
  NCOM::CStgMedium medium;
  RINOK(DataObject_GetData_HGLOBAL(dataObject, (CLIPFORMAT)
      RegisterClipboardFormat(MYWIN_CFSTR_SHELLIDLIST), medium))
  const size_t blockSize = GlobalSize(medium.hGlobal);
  if (blockSize < sizeof(MYWIN_CIDA) || blockSize >= (UInt32)((UInt32)0 - 1))
    return E_INVALIDARG;
  NMemory::CGlobalLock dropLock(medium.hGlobal);
  const MYWIN_CIDA *cida = (const MYWIN_CIDA *)dropLock.GetPointer();
  if (!cida)
    return E_INVALIDARG;
  if (cida->cidl == 0)
  {
    // is it posssible to have no selected items?
    // it's unexpected case.
    return E_INVALIDARG;
  }
  if (cida->cidl >= (blockSize - (UInt32)sizeof(MYWIN_CIDA)) / sizeof(UINT))
    return E_INVALIDARG;
  const UInt32 start = cida->cidl * (UInt32)sizeof(UINT) + (UInt32)sizeof(MYWIN_CIDA);

  STRRET strret;
  CMyComPtr<IShellFolder> parentFolder;
  {
    const UINT offset = cida->aoffset[0];
    if (offset < start || offset >= blockSize
        // || offset != start
        )
      return E_INVALIDARG;

    CMyComPtr<IShellFolder> desktopFolder;
    RINOK(::SHGetDesktopFolder(&desktopFolder))
    if (!desktopFolder)
      return E_FAIL;
    
    LPCITEMIDLIST const lpcItem = (LPCITEMIDLIST)(const void *)((const Byte *)cida + offset);

   #ifdef SHOW_DEBUG_SHELL
    {
      const HRESULT res = desktopFolder->GetDisplayNameOf(
          lpcItem, SHGDN_FORPARSING, &strret);
      if (res == S_OK && strret.uType == STRRET_WSTR)
      {
        ODS_U(strret.pOleStr)
        /* if lpcItem is empty, the path will be
             "C:\Users\user_name\Desktop"
           if lpcItem is "My Computer" folder, the path will be
             "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}" */
        CoTaskMemFree(strret.pOleStr);
      }
    }
   #endif
    
    RINOK(desktopFolder->BindToObject(lpcItem,
        NULL, IID_IShellFolder, (void **)&parentFolder))
    if (!parentFolder)
      return E_FAIL;
  }
  
  names.ClearAndReserve(cida->cidl);
  UString path;
  
  // for (int y = 0; y < 1; y++) // for debug
  for (unsigned i = 1; i <= cida->cidl; i++)
  {
    const UINT offset = cida->aoffset[i];
    if (offset < start || offset >= blockSize)
      return E_INVALIDARG;
    const void *p = (const Byte *)(const void *)cida + offset;
    /* ITEMIDLIST of file can contain more than one SHITEMID item.
       In win10 only SHGDN_FORPARSING returns path that contains
       all path parts related to parts of ITEMIDLIST.
       So we can use only SHGDN_FORPARSING here.
       Don't use (SHGDN_INFOLDER)
       Don't use (SHGDN_INFOLDER | SHGDN_FORPARSING)
    */
    RINOK(parentFolder->GetDisplayNameOf((LPCITEMIDLIST)p, SHGDN_FORPARSING, &strret))

    /*
    // MSVC6 and old SDK do not support StrRetToStrW().
    LPWSTR lpstr;
    RINOK (StrRetToStrW(&strret, NULL, &lpstr))
    ODS_U(lpstr)
    path = lpstr;
    CoTaskMemFree(lpstr);
    */
    if (strret.uType != STRRET_WSTR)
      return E_INVALIDARG;
    ODS_U(strret.pOleStr)
    path = strret.pOleStr;
    // the path could have super path prefix "\\\\?\\"
    // we can remove super path prefix here, if we don't need that prefix
  #ifdef Z7_LONG_PATH
    // we remove super prefix, if we can work without that prefix
    NFile::NName::If_IsSuperPath_RemoveSuperPrefix(path);
  #endif
    names.AddInReserved(path);
    CoTaskMemFree(strret.pOleStr);
  }

  ODS_(Print_Number(cida->cidl, "CFSTR_SHELLIDLIST END"))
  return S_OK;
}


HRESULT DataObject_GetData_HDROP_or_IDLIST_Names(IDataObject *dataObject, UStringVector &paths)
{
  ODS("-- DataObject_GetData_HDROP_or_IDLIST_Names START")
  HRESULT hres = NShell::DataObject_GetData_HDROP_Names(dataObject, paths);
  // if (hres == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER))
  if (hres != S_OK)
  {
    ODS("-- DataObject_GetData_IDLIST START")
    // for (int y = 0; y < 10000; y++) // for debug
    hres = NShell::DataObject_GetData_IDLIST(dataObject, paths);
  }
  ODS("-- DataObject_GetData_HDROP_or_IDLIST_Names END")
  return hres;
}



// #if (NTDDI_VERSION >= NTDDI_VISTA)
typedef struct
{
  UINT cItems;                    // number of items in rgdwFileAttributes array
  DWORD dwSumFileAttributes;      // all of the attributes ORed together
  DWORD dwProductFileAttributes;  // all of the attributes ANDed together
  DWORD rgdwFileAttributes[1];    // array
} MYWIN_FILE_ATTRIBUTES_ARRAY;

#define MYWIN_CFSTR_FILE_ATTRIBUTES_ARRAY  TEXT("File Attributes Array")

HRESULT DataObject_GetData_FILE_ATTRS(IDataObject *dataObject, CFileAttribs &attribs)
{
  attribs.Clear();
  NCOM::CStgMedium medium;
  RINOK(DataObject_GetData_HGLOBAL(dataObject, (CLIPFORMAT)
      RegisterClipboardFormat(MYWIN_CFSTR_FILE_ATTRIBUTES_ARRAY), medium))
  const size_t blockSize = GlobalSize(medium.hGlobal);
  if (blockSize < sizeof(MYWIN_FILE_ATTRIBUTES_ARRAY))
    return E_INVALIDARG;
  NMemory::CGlobalLock dropLock(medium.hGlobal);
  const MYWIN_FILE_ATTRIBUTES_ARRAY *faa = (const MYWIN_FILE_ATTRIBUTES_ARRAY *)dropLock.GetPointer();
  if (!faa)
    return E_INVALIDARG;
  const unsigned numFiles = faa->cItems;
  if (numFiles == 0)
  {
    // is it posssible to have empty array here?
    return E_INVALIDARG;
  }
  if ((blockSize - (sizeof(MYWIN_FILE_ATTRIBUTES_ARRAY) - sizeof(DWORD)))
      / sizeof(DWORD) != numFiles)
    return E_INVALIDARG;
  // attribs.Sum = faa->dwSumFileAttributes;
  // attribs.Product = faa->dwProductFileAttributes;
  // attribs.Vals.SetFromArray(faa->rgdwFileAttributes, numFiles);
  // attribs.IsDirVector.ClearAndSetSize(numFiles);

  if ((faa->dwSumFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
  {
    /* in win10: if selected items are volumes (c:\, d:\ ..) in  My Compter,
       all items have FILE_ATTRIBUTE_DIRECTORY attribute
       ntfs volume also have FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM
       udf volume: FILE_ATTRIBUTE_READONLY
       dvd-rom device: (-1) : all bits are set
    */
    const DWORD *attr = faa->rgdwFileAttributes;
    // DWORD product = (UInt32)0 - 1, sum = 0;
    for (unsigned i = 0; i < numFiles; i++)
    {
      if (attr[i] & FILE_ATTRIBUTE_DIRECTORY)
      {
        // attribs.ThereAreDirs = true;
        attribs.FirstDirIndex = (int)i;
        break;
      }
      // attribs.IsDirVector[i] = (attr[i] & FILE_ATTRIBUTE_DIRECTORY) != 0;
      // product &= v;
      // sum |= v;
    }
    // ODS_(Print_Number(product, "Product calc FILE_ATTRIBUTES_ARRAY ==== DataObject_GetData_HDROP_Names"))
    // ODS_(Print_Number(sum, "Sum calc FILE_ATTRIBUTES_ARRAY ==== DataObject_GetData_HDROP_Names"))
  }
  // ODS_(Print_Number(attribs.Product, "Product FILE_ATTRIBUTES_ARRAY ==== DataObject_GetData_HDROP_Names"))
  // ODS_(Print_Number(attribs.Sum, "Sum FILE_ATTRIBUTES_ARRAY ==== DataObject_GetData_HDROP_Names"))
  ODS_(Print_Number(numFiles, "FILE_ATTRIBUTES_ARRAY ==== DataObject_GetData_HDROP_Names"))
  return S_OK;
}


/////////////////////////////
// CDrop

/*
  win10:
  DragQueryFile() implementation code is not effective because
  there is no pointer inside DROP internal file list, so
  DragQueryFile(fileIndex) runs all names in range [0, fileIndex].
  DragQueryFile(,, buf, bufSize)
  if (buf == NULL) by spec
  {
    returns value is the required size
    in characters, of the buffer, not including the terminating null character
    tests show that if (bufSize == 0), then it also returns  required size.
  }
  if (bufSize != NULL)
  {
    returns: the count of the characters copied, not including null character.
    win10: null character is also  copied at position buf[ret_count];
  }
*/

/*
void CDrop::Attach(HDROP object)
{
  Free();
  m_Object = object;
  m_Assigned = true;
}

void CDrop::Free()
{
  if (m_MustBeFinished && m_Assigned)
    Finish();
  m_Assigned = false;
}

UINT CDrop::QueryCountOfFiles()
{
  return QueryFile(0xFFFFFFFF, (LPTSTR)NULL, 0);
}

void CDrop::QueryFileName(UINT fileIndex, UString &fileName)
{
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    AString fileNameA;
    const UINT len = QueryFile(fileIndex, (LPTSTR)NULL, 0);
    const UINT numCopied = QueryFile(fileIndex, fileNameA.GetBuf(len + 2), len + 2);
    fileNameA.ReleaseBuf_CalcLen(len);
    if (numCopied != len)
      throw 20221223;
    fileName = GetUnicodeString(fileNameA);
  }
  else
  #endif
  {
    // kReserve must be >= 3 for additional buffer size
    //   safety and for optimal performance
    const unsigned kReserve = 3;
    {
      unsigned len = 0;
      wchar_t *buf = fileName.GetBuf_GetMaxAvail(len);
      if (len >= kReserve)
      {
        const UINT numCopied = QueryFile(fileIndex, buf, len);
        if (numCopied < len - 1)
        {
          // (numCopied < len - 1) case means that it have copied full string.
          fileName.ReleaseBuf_CalcLen(numCopied);
          return;
        }
      }
    }
    const UINT len = QueryFile(fileIndex, (LPWSTR)NULL, 0);
    const UINT numCopied = QueryFile(fileIndex,
        fileName.GetBuf(len + kReserve), len + kReserve);
    fileName.ReleaseBuf_CalcLen(len);
    if (numCopied != len)
      throw 20221223;
  }
}


void CDrop::QueryFileNames(UStringVector &fileNames)
{
  UINT numFiles = QueryCountOfFiles();
  
  Print_Number(numFiles, "\n====== CDrop::QueryFileNames START ===== \n");

  fileNames.ClearAndReserve(numFiles);
  UString s;
  for (UINT i = 0; i < numFiles; i++)
  {
    QueryFileName(i, s);
    if (!s.IsEmpty())
      fileNames.AddInReserved(s);
  }
  Print_Number(numFiles, "\n====== CDrop::QueryFileNames END ===== \n");
}
*/


// #if (NTDDI_VERSION >= NTDDI_VISTA)
// SHGetPathFromIDListEx returns a win32 file system path for the item in the name space.
typedef int Z7_WIN_GPFIDL_FLAGS;

extern "C" {
#ifndef _UNICODE
typedef BOOL (WINAPI * Func_SHGetPathFromIDListW)(LPCITEMIDLIST pidl, LPWSTR pszPath); // nt4
#endif

#if !defined(Z7_WIN32_WINNT_MIN) || Z7_WIN32_WINNT_MIN < 0x0600  // Vista
#define Z7_USE_DYN_SHGetPathFromIDListEx
#endif

#ifdef Z7_USE_DYN_SHGetPathFromIDListEx
Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION
typedef BOOL (WINAPI * Func_SHGetPathFromIDListEx)(LPCITEMIDLIST pidl, PWSTR pszPath, DWORD cchPath, Z7_WIN_GPFIDL_FLAGS uOpts); // vista
#endif
}

#ifndef _UNICODE

bool GetPathFromIDList(LPCITEMIDLIST itemIDList, AString &path)
{
  path.Empty();
  const unsigned len = MAX_PATH + 16;
  const bool result = BOOLToBool(::SHGetPathFromIDList(itemIDList, path.GetBuf(len)));
  path.ReleaseBuf_CalcLen(len);
  return result;
}

#endif

bool GetPathFromIDList(LPCITEMIDLIST itemIDList, UString &path)
{
  path.Empty();
  unsigned len = MAX_PATH + 16;

#ifdef _UNICODE
  bool result = BOOLToBool(::SHGetPathFromIDList(itemIDList, path.GetBuf(len)));
#else
  const
  Func_SHGetPathFromIDListW
       shGetPathFromIDListW = Z7_GET_PROC_ADDRESS(
  Func_SHGetPathFromIDListW, ::GetModuleHandleW(L"shell32.dll"),
      "SHGetPathFromIDListW");
  if (!shGetPathFromIDListW)
    return false;
  bool result = BOOLToBool(shGetPathFromIDListW(itemIDList, path.GetBuf(len)));
#endif

  if (!result)
  {
    ODS("==== GetPathFromIDList() SHGetPathFromIDList() returned false")
    /* for long path we need SHGetPathFromIDListEx().
      win10: SHGetPathFromIDListEx() for long path returns path with
             with super path prefix "\\\\?\\". */
#ifdef Z7_USE_DYN_SHGetPathFromIDListEx
    const
    Func_SHGetPathFromIDListEx
    func_SHGetPathFromIDListEx = Z7_GET_PROC_ADDRESS(
    Func_SHGetPathFromIDListEx, ::GetModuleHandleW(L"shell32.dll"),
        "SHGetPathFromIDListEx");
    if (func_SHGetPathFromIDListEx)
#endif
    {
      ODS("==== GetPathFromIDList() (SHGetPathFromIDListEx)")
      do
      {
        len *= 4;
        result = BOOLToBool(
#ifdef Z7_USE_DYN_SHGetPathFromIDListEx
          func_SHGetPathFromIDListEx
#else
          SHGetPathFromIDListEx
#endif
          (itemIDList, path.GetBuf(len), len, 0));
        if (result)
          break;
      }
      while (len <= (1 << 16));
    }
  }

  path.ReleaseBuf_CalcLen(len);
  return result;
}

#endif

#ifdef UNDER_CE

bool BrowseForFolder(LPBROWSEINFO, CSysString)
{
  return false;
}

bool BrowseForFolder(HWND, LPCTSTR, UINT, LPCTSTR, CSysString &)
{
  return false;
}

bool BrowseForFolder(HWND /* owner */, LPCTSTR /* title */,
    LPCTSTR /* initialFolder */, CSysString & /* resultPath */)
{
  /*
  // SHBrowseForFolder doesn't work before CE 6.0 ?
  if (GetProcAddress(LoadLibrary(L"ceshell.dll", L"SHBrowseForFolder") == 0)
    MessageBoxW(0, L"no", L"", 0);
  else
    MessageBoxW(0, L"yes", L"", 0);
  */
  /*
  UString s = "all files";
  s += " (*.*)";
  return MyGetOpenFileName(owner, title, initialFolder, s, resultPath, true);
  */
  return false;
}

#else

/* win10: SHBrowseForFolder() doesn't support long paths,
   even if long path suppport is enabled in registry and in manifest.
   and SHBrowseForFolder() doesn't support super path prefix "\\\\?\\". */

bool BrowseForFolder(LPBROWSEINFO browseInfo, CSysString &resultPath)
{
  resultPath.Empty();
  NWindows::NCOM::CComInitializer comInitializer;
  LPITEMIDLIST itemIDList = ::SHBrowseForFolder(browseInfo);
  if (!itemIDList)
    return false;
  CItemIDList itemIDListHolder;
  itemIDListHolder.Attach(itemIDList);
  return GetPathFromIDList(itemIDList, resultPath);
}


static int CALLBACK BrowseCallbackProc(HWND hwnd, UINT uMsg, LPARAM /* lp */, LPARAM data)
{
  #ifndef UNDER_CE
  switch (uMsg)
  {
    case BFFM_INITIALIZED:
    {
      SendMessage(hwnd, BFFM_SETSELECTION, TRUE, data);
      break;
    }
    /*
    case BFFM_SELCHANGED:
    {
      TCHAR dir[MAX_PATH];
      if (::SHGetPathFromIDList((LPITEMIDLIST) lp , dir))
        SendMessage(hwnd, BFFM_SETSTATUSTEXT, 0, (LPARAM)dir);
      else
        SendMessage(hwnd, BFFM_SETSTATUSTEXT, 0, (LPARAM)TEXT(""));
      break;
    }
    */
    default:
      break;
  }
  #endif
  return 0;
}


static bool BrowseForFolder(HWND owner, LPCTSTR title, UINT ulFlags,
    LPCTSTR initialFolder, CSysString &resultPath)
{
  CSysString displayName;
  BROWSEINFO browseInfo;
  browseInfo.hwndOwner = owner;
  browseInfo.pidlRoot = NULL;

  // there are Unicode/Astring problems in some WinCE SDK ?
  /*
  #ifdef UNDER_CE
  browseInfo.pszDisplayName = (LPSTR)displayName.GetBuf(MAX_PATH);
  browseInfo.lpszTitle = (LPCSTR)title;
  #else
  */
  browseInfo.pszDisplayName = displayName.GetBuf(MAX_PATH);
  browseInfo.lpszTitle = title;
  // #endif
  browseInfo.ulFlags = ulFlags;
  browseInfo.lpfn = initialFolder ? BrowseCallbackProc : NULL;
  browseInfo.lParam = (LPARAM)initialFolder;
  return BrowseForFolder(&browseInfo, resultPath);
}

#ifdef Z7_OLD_WIN_SDK
// ShlObj.h:
#ifndef BIF_NEWDIALOGSTYLE
#define BIF_NEWDIALOGSTYLE     0x0040
#endif
#endif

bool BrowseForFolder(HWND owner, LPCTSTR title,
    LPCTSTR initialFolder, CSysString &resultPath)
{
  return BrowseForFolder(owner, title,
      #ifndef UNDER_CE
      BIF_NEWDIALOGSTYLE |
      #endif
      BIF_RETURNONLYFSDIRS | BIF_STATUSTEXT, initialFolder, resultPath);
  // BIF_STATUSTEXT; BIF_USENEWUI   (Version 5.0)
}

#ifndef _UNICODE

extern "C" {
typedef LPITEMIDLIST (WINAPI * Func_SHBrowseForFolderW)(LPBROWSEINFOW lpbi);
}

static bool BrowseForFolder(LPBROWSEINFOW browseInfo, UString &resultPath)
{
  NWindows::NCOM::CComInitializer comInitializer;
  const
  Func_SHBrowseForFolderW
     f_SHBrowseForFolderW = Z7_GET_PROC_ADDRESS(
  Func_SHBrowseForFolderW, ::GetModuleHandleW(L"shell32.dll"),
      "SHBrowseForFolderW");
  if (!f_SHBrowseForFolderW)
    return false;
  LPITEMIDLIST itemIDList = f_SHBrowseForFolderW(browseInfo);
  if (!itemIDList)
    return false;
  CItemIDList itemIDListHolder;
  itemIDListHolder.Attach(itemIDList);
  return GetPathFromIDList(itemIDList, resultPath);
}

static
int CALLBACK BrowseCallbackProc2(HWND hwnd, UINT uMsg, LPARAM /* lp */, LPARAM data)
{
  switch (uMsg)
  {
    case BFFM_INITIALIZED:
    {
      SendMessageW(hwnd, BFFM_SETSELECTIONW, TRUE, data);
      break;
    }
    /*
    case BFFM_SELCHANGED:
    {
      wchar_t dir[MAX_PATH * 2];

      if (shGetPathFromIDListW((LPITEMIDLIST)lp , dir))
        SendMessageW(hwnd, BFFM_SETSTATUSTEXTW, 0, (LPARAM)dir);
      else
        SendMessageW(hwnd, BFFM_SETSTATUSTEXTW, 0, (LPARAM)L"");
      break;
    }
    */
    default:
      break;
  }
  return 0;
}


static bool BrowseForFolder(HWND owner, LPCWSTR title, UINT ulFlags,
    LPCWSTR initialFolder, UString &resultPath)
{
  UString displayName;
  BROWSEINFOW browseInfo;
  browseInfo.hwndOwner = owner;
  browseInfo.pidlRoot = NULL;
  browseInfo.pszDisplayName = displayName.GetBuf(MAX_PATH);
  browseInfo.lpszTitle = title;
  browseInfo.ulFlags = ulFlags;
  browseInfo.lpfn = initialFolder ? BrowseCallbackProc2 : NULL;
  browseInfo.lParam = (LPARAM)initialFolder;
  return BrowseForFolder(&browseInfo, resultPath);
}

bool BrowseForFolder(HWND owner, LPCWSTR title, LPCWSTR initialFolder, UString &resultPath)
{
  if (g_IsNT)
    return BrowseForFolder(owner, title,
      BIF_NEWDIALOGSTYLE | BIF_RETURNONLYFSDIRS
      //  | BIF_STATUSTEXT // This flag is not supported when BIF_NEWDIALOGSTYLE is specified.
      , initialFolder, resultPath);
  // BIF_STATUSTEXT; BIF_USENEWUI   (Version 5.0)
  CSysString s;
  bool res = BrowseForFolder(owner, GetSystemString(title),
      BIF_NEWDIALOGSTYLE | BIF_RETURNONLYFSDIRS
      // | BIF_STATUSTEXT  // This flag is not supported when BIF_NEWDIALOGSTYLE is specified.
      , GetSystemString(initialFolder), s);
  resultPath = GetUnicodeString(s);
  return res;
}

#endif

#endif

}}
