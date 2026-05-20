// Windows/Shell.h

#ifndef ZIP7_WINDOWS_SHELL_H
#define ZIP7_WINDOWS_SHELL_H

#include "../Common/MyWindows.h"
#if defined(__MINGW32__) || defined(__MINGW64__)
#include <shlobj.h>
#else
#include <ShlObj.h>
#endif

#include "../Common/MyString.h"

#include "Defs.h"

namespace NWindows {
namespace NShell {

/////////////////////////
// CItemIDList
#ifndef UNDER_CE

class CItemIDList
{
  LPITEMIDLIST m_Object;
  Z7_CLASS_NO_COPY(CItemIDList)
public:
  CItemIDList(): m_Object(NULL) {}
  // CItemIDList(LPCITEMIDLIST itemIDList);
  // CItemIDList(const CItemIDList& itemIDList);
  ~CItemIDList() { Free(); }
  void Free();
  void Attach(LPITEMIDLIST object)
  {
    Free();
    m_Object = object;
  }
  LPITEMIDLIST Detach()
  {
    LPITEMIDLIST object = m_Object;
    m_Object = NULL;
    return object;
  }
  operator LPITEMIDLIST() { return m_Object;}
  operator LPCITEMIDLIST() const { return m_Object;}
  LPITEMIDLIST* operator&() { return &m_Object; }
  LPITEMIDLIST operator->() { return m_Object; }

  // CItemIDList& operator=(LPCITEMIDLIST object);
  // CItemIDList& operator=(const CItemIDList &object);
};

/////////////////////////////
// CDrop

/*
class CDrop
{
  HDROP m_Object;
  bool m_MustBeFinished;
  bool m_Assigned;
  void Free();
public:
  CDrop(bool mustBeFinished) : m_MustBeFinished(mustBeFinished), m_Assigned(false) {}
  ~CDrop() { Free(); }

  void Attach(HDROP object);
  operator HDROP() { return m_Object;}
  bool QueryPoint(LPPOINT point)
    { return BOOLToBool(::DragQueryPoint(m_Object, point)); }
  void Finish()
  {
    ::DragFinish(m_Object);
  }
  UINT QueryFile(UINT fileIndex, LPTSTR fileName, UINT bufSize)
    { return ::DragQueryFile(m_Object, fileIndex, fileName, bufSize); }
  #ifndef _UNICODE
  UINT QueryFile(UINT fileIndex, LPWSTR fileName, UINT bufSize)
    { return ::DragQueryFileW(m_Object, fileIndex, fileName, bufSize); }
  #endif
  UINT QueryCountOfFiles();
  void QueryFileName(UINT fileIndex, UString &fileName);
  void QueryFileNames(UStringVector &fileNames);
};
*/
#endif

struct CFileAttribs
{
  int FirstDirIndex;
  // DWORD Sum;
  // DWORD Product;
  // CRecordVector<DWORD> Vals;
  // CRecordVector<bool> IsDirVector;

  CFileAttribs()
  {
    Clear();
  }

  void Clear()
  {
    FirstDirIndex = -1;
    // Sum = 0;
    // Product = 0;
    // IsDirVector.Clear();
  }
};


/* read pathnames from HDROP or SHELLIDLIST.
   The parser can return E_INVALIDARG, if there is some unexpected data in dataObject */
HRESULT DataObject_GetData_HDROP_or_IDLIST_Names(IDataObject *dataObject, UStringVector &names);

HRESULT DataObject_GetData_FILE_ATTRS(IDataObject *dataObject, CFileAttribs &attribs);

bool GetPathFromIDList(LPCITEMIDLIST itemIDList, CSysString &path);
bool BrowseForFolder(LPBROWSEINFO lpbi, CSysString &resultPath);
bool BrowseForFolder(HWND owner, LPCTSTR title, LPCTSTR initialFolder, CSysString &resultPath);

#ifndef _UNICODE
bool GetPathFromIDList(LPCITEMIDLIST itemIDList, UString &path);
bool BrowseForFolder(LPBROWSEINFO lpbi, UString &resultPath);
bool BrowseForFolder(HWND owner, LPCWSTR title, LPCWSTR initialFolder, UString &resultPath);
#endif
}}

#endif
