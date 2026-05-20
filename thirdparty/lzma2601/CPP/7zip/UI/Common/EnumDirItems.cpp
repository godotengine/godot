// EnumDirItems.cpp

#include "StdAfx.h"

#include <wchar.h>
// #include <stdio.h>

#ifndef _WIN32
#include <grp.h>
#include <pwd.h>
#include "../../../Common/UTFConvert.h"
#endif

#include "../../../Common/Wildcard.h"

#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileIO.h"
#include "../../../Windows/FileName.h"

#if defined(_WIN32) && !defined(UNDER_CE)
#define Z7_USE_SECURITY_CODE
#include "../../../Windows/SecurityUtils.h"
#endif

#include "EnumDirItems.h"
#include "SortUtils.h"

using namespace NWindows;
using namespace NFile;
using namespace NName;


static bool FindFile_KeepDots(NFile::NFind::CFileInfo &fi, const FString &path, bool followLink)
{
  const bool res = fi.Find(path, followLink);
  if (!res)
    return res;
  if (path.IsEmpty())
    return res;
  // we keep name "." and "..", if it's without tail slash
  const FChar *p = path.RightPtr(1);
  if (*p != '.')
    return res;
  if (p != path.Ptr())
  {
    FChar c = p[-1];
    if (!IS_PATH_SEPAR(c))
    {
      if (c != '.')
        return res;
      p--;
      if (p != path.Ptr())
      {
        c = p[-1];
        if (!IS_PATH_SEPAR(c))
          return res;
      }
    }
  }
  fi.Name = p;
  return res;
}


void CDirItems::AddDirFileInfo(int phyParent, int logParent, int secureIndex,
    const NFind::CFileInfo &fi)
{
  /*
  CDirItem di(fi);
  di.PhyParent = phyParent;
  di.LogParent = logParent;
  di.SecureIndex = secureIndex;
  Items.Add(di);
  */
  VECTOR_ADD_NEW_OBJECT (Items, CDirItem(fi, phyParent, logParent, secureIndex))
  
  if (fi.IsDir())
    Stat.NumDirs++;
 #ifdef _WIN32
  else if (fi.IsAltStream)
  {
    Stat.NumAltStreams++;
    Stat.AltStreamsSize += fi.Size;
  }
 #endif
  else
  {
    Stat.NumFiles++;
    Stat.FilesSize += fi.Size;
  }
}

// (DWORD)E_FAIL
#define DI_DEFAULT_ERROR  ERROR_INVALID_FUNCTION

HRESULT CDirItems::AddError(const FString &path, DWORD errorCode)
{
  if (errorCode == 0)
    errorCode = DI_DEFAULT_ERROR;
  Stat.NumErrors++;
  if (Callback)
    return Callback->ScanError(path, errorCode);
  return S_OK;
}

HRESULT CDirItems::AddError(const FString &path)
{
  return AddError(path, ::GetLastError());
}

static const unsigned kScanProgressStepMask = (1 << 12) - 1;

HRESULT CDirItems::ScanProgress(const FString &dirPath)
{
  if (Callback)
    return Callback->ScanProgress(Stat, dirPath, true);
  return S_OK;
}

UString CDirItems::GetPrefixesPath(const CIntVector &parents, int index, const UString &name) const
{
  UString path;
  unsigned len = name.Len();
  
  int i;
  for (i = index; i >= 0; i = parents[(unsigned)i])
    len += Prefixes[(unsigned)i].Len();
  
  wchar_t *p = path.GetBuf_SetEnd(len) + len;
  
  p -= name.Len();
  wmemcpy(p, (const wchar_t *)name, name.Len());
  
  for (i = index; i >= 0; i = parents[(unsigned)i])
  {
    const UString &s = Prefixes[(unsigned)i];
    p -= s.Len();
    wmemcpy(p, (const wchar_t *)s, s.Len());
  }
  
  return path;
}

FString CDirItems::GetPhyPath(unsigned index) const
{
  const CDirItem &di = Items[index];
  return us2fs(GetPrefixesPath(PhyParents, di.PhyParent, di.Name));
}

UString CDirItems::GetLogPath(unsigned index) const
{
  const CDirItem &di = Items[index];
  return GetPrefixesPath(LogParents, di.LogParent, di.Name);
}

void CDirItems::ReserveDown()
{
  Prefixes.ReserveDown();
  PhyParents.ReserveDown();
  LogParents.ReserveDown();
  Items.ReserveDown();
}

unsigned CDirItems::AddPrefix(int phyParent, int logParent, const UString &prefix)
{
  PhyParents.Add(phyParent);
  LogParents.Add(logParent);
  return Prefixes.Add(prefix);
}

void CDirItems::DeleteLastPrefix()
{
  PhyParents.DeleteBack();
  LogParents.DeleteBack();
  Prefixes.DeleteBack();
}

bool InitLocalPrivileges();

CDirItems::CDirItems():
    SymLinks(false),
    ScanAltStreams(false)
    , ExcludeDirItems(false)
    , ExcludeFileItems(false)
    , ShareForWrite(false)
   #ifdef Z7_USE_SECURITY_CODE
    , ReadSecure(false)
   #endif
   #ifndef _WIN32
    , StoreOwnerName(false)
   #endif
    , Callback(NULL)
{
  #ifdef Z7_USE_SECURITY_CODE
  _saclEnabled = InitLocalPrivileges();
  #endif
}


#ifdef Z7_USE_SECURITY_CODE

HRESULT CDirItems::AddSecurityItem(const FString &path, int &secureIndex)
{
  secureIndex = -1;

  SECURITY_INFORMATION securInfo =
      DACL_SECURITY_INFORMATION |
      GROUP_SECURITY_INFORMATION |
      OWNER_SECURITY_INFORMATION;
  if (_saclEnabled)
    securInfo |= SACL_SECURITY_INFORMATION;

  DWORD errorCode = 0;
  DWORD secureSize;
  
  BOOL res = ::GetFileSecurityW(fs2us(path), securInfo, (PSECURITY_DESCRIPTOR)(void *)(Byte *)TempSecureBuf, (DWORD)TempSecureBuf.Size(), &secureSize);
  
  if (res)
  {
    if (secureSize == 0)
      return S_OK;
    if (secureSize > TempSecureBuf.Size())
      errorCode = ERROR_INVALID_FUNCTION;
  }
  else
  {
    errorCode = GetLastError();
    if (errorCode == ERROR_INSUFFICIENT_BUFFER)
    {
      if (secureSize <= TempSecureBuf.Size())
        errorCode = ERROR_INVALID_FUNCTION;
      else
      {
        TempSecureBuf.Alloc(secureSize);
        res = ::GetFileSecurityW(fs2us(path), securInfo, (PSECURITY_DESCRIPTOR)(void *)(Byte *)TempSecureBuf, (DWORD)TempSecureBuf.Size(), &secureSize);
        if (res)
        {
          if (secureSize != TempSecureBuf.Size())
            errorCode = ERROR_INVALID_FUNCTION;
        }
        else
          errorCode = GetLastError();
      }
    }
  }
  
  if (res)
  {
    secureIndex = (int)SecureBlocks.AddUniq(TempSecureBuf, secureSize);
    return S_OK;
  }
  
  return AddError(path, errorCode);
}

#endif // Z7_USE_SECURITY_CODE


HRESULT CDirItems::EnumerateOneDir(const FString &phyPrefix, CObjectVector<NFind::CFileInfo> &files)
{
  NFind::CEnumerator enumerator;
  // printf("\n  enumerator.SetDirPrefix(phyPrefix) \n");

  enumerator.SetDirPrefix(phyPrefix);

  #ifdef _WIN32

  NFind::CFileInfo fi;

  for (unsigned ttt = 0; ; ttt++)
  {
    bool found;
    if (!enumerator.Next(fi, found))
      return AddError(phyPrefix);
    if (!found)
      return S_OK;
    files.Add(fi);
    if (Callback && (ttt & kScanProgressStepMask) == kScanProgressStepMask)
    {
      RINOK(ScanProgress(phyPrefix))
    }
  }

  #else // _WIN32

  // enumerator.SolveLinks = !SymLinks;
  
  CObjectVector<NFind::CDirEntry> entries;

  for (;;)
  {
    bool found;
    NFind::CDirEntry de;
    if (!enumerator.Next(de, found))
      return AddError(phyPrefix);
    if (!found)
      break;
    entries.Add(de);
  }

  FOR_VECTOR(i, entries)
  {
    const NFind::CDirEntry &de = entries[i];
    NFind::CFileInfo fi;
    if (!enumerator.Fill_FileInfo(de, fi, !SymLinks))
    // if (!fi.Find_AfterEnumerator(path))
    {
      const FString path = phyPrefix + de.Name;
      {
        RINOK(AddError(path))
        continue;
      }
    }

    files.Add(fi);

    if (Callback && (i & kScanProgressStepMask) == kScanProgressStepMask)
    {
      RINOK(ScanProgress(phyPrefix))
    }
  }

  return S_OK;

  #endif // _WIN32
}




HRESULT CDirItems::EnumerateDir(int phyParent, int logParent, const FString &phyPrefix)
{
  RINOK(ScanProgress(phyPrefix))

  CObjectVector<NFind::CFileInfo> files;
  RINOK(EnumerateOneDir(phyPrefix, files))

  FOR_VECTOR (i, files)
  {
    #ifdef _WIN32
    const NFind::CFileInfo &fi = files[i];
    #else
    const NFind::CFileInfo &fi = files[i];
    /*
    NFind::CFileInfo fi;
    {
      const NFind::CDirEntry &di = files[i];
      const FString path = phyPrefix + di.Name;
      if (!fi.Find_AfterEnumerator(path))
      {
        RINOK(AddError(path));
        continue;
      }
      fi.Name = di.Name;
    }
    */
    #endif

    if (CanIncludeItem(fi.IsDir()))
    {
    int secureIndex = -1;
    #ifdef Z7_USE_SECURITY_CODE
    if (ReadSecure)
    {
      RINOK(AddSecurityItem(phyPrefix + fi.Name, secureIndex))
    }
    #endif
    AddDirFileInfo(phyParent, logParent, secureIndex, fi);
    }
    
    if (Callback && (i & kScanProgressStepMask) == kScanProgressStepMask)
    {
      RINOK(ScanProgress(phyPrefix))
    }

    if (fi.IsDir())
    {
      const FString name2 = fi.Name + FCHAR_PATH_SEPARATOR;
      unsigned parent = AddPrefix(phyParent, logParent, fs2us(name2));
      RINOK(EnumerateDir((int)parent, (int)parent, phyPrefix + name2))
    }
  }
  return S_OK;
}


/*
EnumerateItems2()
  const FStringVector &filePaths - are path without tail slashes.
  All dir prefixes of filePaths will be not stores in logical paths
fix it: we can scan AltStream also.
*/

#ifdef _WIN32
// #define FOLLOW_LINK_PARAM
// #define FOLLOW_LINK_PARAM2
#define FOLLOW_LINK_PARAM , (!SymLinks)
#define FOLLOW_LINK_PARAM2 , (!dirItems.SymLinks)
#else
#define FOLLOW_LINK_PARAM , (!SymLinks)
#define FOLLOW_LINK_PARAM2 , (!dirItems.SymLinks)
#endif

HRESULT CDirItems::EnumerateItems2(
    const FString &phyPrefix,
    const UString &logPrefix,
    const FStringVector &filePaths,
    FStringVector *requestedPaths)
{
  const int phyParent = phyPrefix.IsEmpty() ? -1 : (int)AddPrefix(-1, -1, fs2us(phyPrefix));
  const int logParent = logPrefix.IsEmpty() ? -1 : (int)AddPrefix(-1, -1, logPrefix);

 #ifdef _WIN32
  const bool phyPrefix_isAltStreamPrefix =
      NFile::NName::IsAltStreamPrefixWithColon(fs2us(phyPrefix));
 #endif

  FOR_VECTOR (i, filePaths)
  {
    const FString &filePath = filePaths[i];
    NFind::CFileInfo fi;
    const FString phyPath = phyPrefix + filePath;
    if (!FindFile_KeepDots(fi, phyPath  FOLLOW_LINK_PARAM))
    {
      RINOK(AddError(phyPath))
      continue;
    }
    if (requestedPaths)
      requestedPaths->Add(phyPath);

    const int delimiter = filePath.ReverseFind_PathSepar();
    FString phyPrefixCur;
    int phyParentCur = phyParent;
    if (delimiter >= 0)
    {
      phyPrefixCur.SetFrom(filePath, (unsigned)(delimiter + 1));
      phyParentCur = (int)AddPrefix(phyParent, logParent, fs2us(phyPrefixCur));
    }

    if (CanIncludeItem(fi.IsDir()))
    {
    int secureIndex = -1;
    #ifdef Z7_USE_SECURITY_CODE
    if (ReadSecure)
    {
      RINOK(AddSecurityItem(phyPath, secureIndex))
    }
    #endif
   #ifdef _WIN32
    if (phyPrefix_isAltStreamPrefix && fi.IsAltStream)
    {
      const int pos = fi.Name.Find(FChar(':'));
      if (pos >= 0)
        fi.Name.DeleteFrontal((unsigned)pos + 1);
    }
   #endif
    AddDirFileInfo(phyParentCur, logParent, secureIndex, fi);
    }
    
    if (fi.IsDir())
    {
      const FString name2 = fi.Name + FCHAR_PATH_SEPARATOR;
      const unsigned parent = AddPrefix(phyParentCur, logParent, fs2us(name2));
      RINOK(EnumerateDir((int)parent, (int)parent, phyPrefix + phyPrefixCur + name2))
    }
  }
  
  ReserveDown();
  return S_OK;
}




static HRESULT EnumerateDirItems(
    const NWildcard::CCensorNode &curNode,
    const int phyParent, const int logParent,
    const FString &phyPrefix,
    const UStringVector &addParts, // additional parts from curNode
    CDirItems &dirItems,
    bool enterToSubFolders);


/* EnumerateDirItems_Spec()
   adds new Dir item prefix, and enumerates dir items,
   then it can remove that Dir item prefix, if there are no items in that dir.
*/


/*
  EnumerateDirItems_Spec()
  it's similar to EnumerateDirItems, but phyPrefix doesn't include (curFolderName)
*/

static HRESULT EnumerateDirItems_Spec(
    const NWildcard::CCensorNode &curNode,
    const int phyParent, const int logParent, const FString &curFolderName,
    const FString &phyPrefix,      // without (curFolderName)
    const UStringVector &addParts, // (curNode + addParts) includes (curFolderName)
    CDirItems &dirItems,
    bool enterToSubFolders)
{
  const FString name2 = curFolderName + FCHAR_PATH_SEPARATOR;
  const unsigned parent = dirItems.AddPrefix(phyParent, logParent, fs2us(name2));
  const unsigned numItems = dirItems.Items.Size();
  HRESULT res = EnumerateDirItems(
      curNode, (int)parent, (int)parent, phyPrefix + name2,
      addParts, dirItems, enterToSubFolders);
  if (numItems == dirItems.Items.Size())
    dirItems.DeleteLastPrefix();
  return res;
}


#ifndef UNDER_CE

#ifdef _WIN32

static HRESULT EnumerateAltStreams(
    const NFind::CFileInfo &fi,
    const NWildcard::CCensorNode &curNode,
    const int phyParent, const int logParent,
    const FString &phyPath,         // with (fi.Name), without tail slash for folders
    const UStringVector &addParts,  // with (fi.Name), prefix parts from curNode
    bool addAllSubStreams,
    CDirItems &dirItems)
{
  // we don't use (ExcludeFileItems) rules for AltStreams
  // if (dirItems.ExcludeFileItems) return S_OK;

  NFind::CStreamEnumerator enumerator(phyPath);
  for (;;)
  {
    NFind::CStreamInfo si;
    bool found;
    if (!enumerator.Next(si, found))
    {
      return dirItems.AddError(phyPath + FTEXT(":*")); // , (DWORD)E_FAIL
    }
    if (!found)
      return S_OK;
    if (si.IsMainStream())
      continue;
    UStringVector parts = addParts;
    const UString reducedName = si.GetReducedName();
    parts.Back() += reducedName;
    if (curNode.CheckPathToRoot(false, parts, true))
      continue;
    if (!addAllSubStreams)
      if (!curNode.CheckPathToRoot(true, parts, true))
        continue;

    NFind::CFileInfo fi2 = fi;
    fi2.Name += us2fs(reducedName);
    fi2.Size = si.Size;
    fi2.Attrib &= ~(DWORD)(FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT);
    fi2.IsAltStream = true;
    dirItems.AddDirFileInfo(phyParent, logParent, -1, fi2);
  }
}

#endif // _WIN32


/* We get Reparse data and parse it.
   If there is Reparse error, we free dirItem.Reparse data.
   Do we need to work with empty reparse data?
*/

HRESULT CDirItems::SetLinkInfo(CDirItem &dirItem, const NFind::CFileInfo &fi,
    const FString &phyPrefix)
{
  if (!SymLinks)
    return S_OK;

  #ifdef _WIN32
    if (!fi.HasReparsePoint() || fi.IsAltStream)
  #else // _WIN32
    if (!fi.IsPosixLink())
  #endif // _WIN32
      return S_OK;

  const FString path = phyPrefix + fi.Name;
  CByteBuffer &buf = dirItem.ReparseData;
  if (NIO::GetReparseData(path, buf))
  {
    // if (dirItem.ReparseData.Size() != 0)
    Stat.FilesSize -= fi.Size;
    return S_OK;
  }

  DWORD res = ::GetLastError();
  buf.Free();
  return AddError(path, res);
}

#endif // UNDER_CE



static HRESULT EnumerateForItem(
    const NFind::CFileInfo &fi,
    const NWildcard::CCensorNode &curNode,
    const int phyParent, const int logParent, const FString &phyPrefix,
    const UStringVector &addParts, // additional parts from curNode, without (fi.Name)
    CDirItems &dirItems,
    bool enterToSubFolders)
{
  const UString name = fs2us(fi.Name);
  UStringVector newParts = addParts;
  newParts.Add(name);
  
  // check the path in exclude rules
  if (curNode.CheckPathToRoot(false, newParts, !fi.IsDir()))
    return S_OK;

  #if !defined(UNDER_CE)
  int dirItemIndex = -1;
  #if defined(_WIN32)
  bool addAllSubStreams = false;
  bool needAltStreams = true;
  #endif // _WIN32
  #endif // !defined(UNDER_CE)

  // check the path in inlcude rules
  if (curNode.CheckPathToRoot(true, newParts, !fi.IsDir()))
  {
    #if !defined(UNDER_CE)
    // dirItemIndex = (int)dirItems.Items.Size();
    #if defined(_WIN32)
    // we will not check include rules for substreams.
    addAllSubStreams = true;
    #endif // _WIN32
    #endif // !defined(UNDER_CE)

    if (dirItems.CanIncludeItem(fi.IsDir()))
    {
      int secureIndex = -1;
    #ifdef Z7_USE_SECURITY_CODE
      if (dirItems.ReadSecure)
      {
        RINOK(dirItems.AddSecurityItem(phyPrefix + fi.Name, secureIndex))
      }
    #endif
    #if !defined(UNDER_CE)
      dirItemIndex = (int)dirItems.Items.Size();
    #endif // !defined(UNDER_CE)
      dirItems.AddDirFileInfo(phyParent, logParent, secureIndex, fi);
    }
    else
    {
      #if defined(_WIN32) && !defined(UNDER_CE)
        needAltStreams = false;
      #endif
    }
    
    if (fi.IsDir())
      enterToSubFolders = true;
  }

  #if !defined(UNDER_CE)
  
  // we don't scan AltStreams for link files

  if (dirItemIndex >= 0)
  {
    CDirItem &dirItem = dirItems.Items[(unsigned)dirItemIndex];
    RINOK(dirItems.SetLinkInfo(dirItem, fi, phyPrefix))
    if (dirItem.ReparseData.Size() != 0)
      return S_OK;
  }
  
  #if defined(_WIN32)
  if (needAltStreams && dirItems.ScanAltStreams && !fi.IsAltStream)
  {
    RINOK(EnumerateAltStreams(fi, curNode, phyParent, logParent,
        phyPrefix + fi.Name,    // with (fi.Name)
        newParts,               // with (fi.Name)
        addAllSubStreams,
        dirItems))
  }
  #endif

  #endif // !defined(UNDER_CE)


  #ifndef _WIN32
  if (!fi.IsPosixLink()) // posix link can follow to dir
  #endif
  if (!fi.IsDir())
    return S_OK;

  const NWildcard::CCensorNode *nextNode = NULL;

  if (addParts.IsEmpty())
  {
    int index = curNode.FindSubNode(name);
    if (index >= 0)
    {
      nextNode = &curNode.SubNodes[(unsigned)index];
      newParts.Clear();
    }
  }

  if (!nextNode)
  {
    if (!enterToSubFolders)
      return S_OK;

   #ifndef _WIN32
    if (fi.IsPosixLink())
    {
      // here we can try to resolve posix link
      // if the link to dir, then can we follow it
      return S_OK; // we don't follow posix link
    }
   #else
    if (dirItems.SymLinks && fi.HasReparsePoint())
    {
      /* 20.03: in SymLinks mode: we don't enter to directory that
         has reparse point and has no CCensorNode
         NOTE: (curNode and parent nodes) still can have wildcard rules
         to include some items of target directory (of reparse point),
         but we ignore these rules here.
      */
      return S_OK;
    }
   #endif
    nextNode = &curNode;
  }
  
  return EnumerateDirItems_Spec(
      *nextNode, phyParent, logParent, fi.Name,
      phyPrefix,   // without (fi.Name)
      newParts,    // relative to (*nextNode). (*nextNode + newParts) includes (fi.Name)
      dirItems,
      enterToSubFolders);
}


static bool CanUseFsDirect(const NWildcard::CCensorNode &curNode)
{
  FOR_VECTOR (i, curNode.IncludeItems)
  {
    const NWildcard::CItem &item = curNode.IncludeItems[i];
    if (item.Recursive || item.PathParts.Size() != 1)
      return false;
    const UString &name = item.PathParts.Front();
    /*
    if (name.IsEmpty())
      return false;
    */
    
    /* Windows doesn't support file name with wildcard
       But if another system supports file name with wildcard,
       and wildcard mode is disabled, we can ignore wildcard in name
    */
    /*
    #ifndef _WIN32
    if (!item.WildcardParsing)
      continue;
    #endif
    */
    if (DoesNameContainWildcard(name))
      return false;
  }
  return true;
}


#if defined(_WIN32) && !defined(UNDER_CE)

static bool IsVirtualFsFolder(const FString &prefix, const UString &name)
{
  UString s = fs2us(prefix);
  s += name;
  s.Add_PathSepar();
  // it returns (true) for non real FS folder path like - "\\SERVER\"
  return IsPathSepar(s[0]) && GetRootPrefixSize(s) == 0;
}

#endif



static HRESULT EnumerateDirItems(
    const NWildcard::CCensorNode &curNode,
    const int phyParent, const int logParent, const FString &phyPrefix,
    const UStringVector &addParts,  // prefix from curNode including
    CDirItems &dirItems,
    bool enterToSubFolders)
{
  if (!enterToSubFolders)
  {
    /* if there are IncludeItems censor rules that affect items in subdirs,
       then we will enter to all subfolders */
    if (curNode.NeedCheckSubDirs())
      enterToSubFolders = true;
  }
  
  RINOK(dirItems.ScanProgress(phyPrefix))

  // try direct_names case at first
  if (addParts.IsEmpty() && !enterToSubFolders)
  {
    if (CanUseFsDirect(curNode))
    {
      // all names are direct (no wildcards)
      // so we don't need file_system's dir enumerator
      CRecordVector<bool> needEnterVector;
      unsigned i;

      for (i = 0; i < curNode.IncludeItems.Size(); i++)
      {
        const NWildcard::CItem &item = curNode.IncludeItems[i];
        const UString &name = item.PathParts.Front();
        FString fullPath = phyPrefix + us2fs(name);

        /*
        // not possible now
        if (!item.ForDir && !item.ForFile)
        {
          RINOK(dirItems.AddError(fullPath, ERROR_INVALID_PARAMETER));
          continue;
        }
        */

        #if defined(_WIN32) && !defined(UNDER_CE)
        bool needAltStreams = true;
        #endif

        #ifdef Z7_USE_SECURITY_CODE
        bool needSecurity = true;
        #endif
        
        if (phyPrefix.IsEmpty())
        {
          if (!item.ForFile)
          {
            /* we don't like some names for alt streams inside archive:
               ":sname"     for "\"
               "c:::sname"  for "C:\"
               So we ignore alt streams for these cases */
            if (name.IsEmpty())
            {
              #if defined(_WIN32) && !defined(UNDER_CE)
              needAltStreams = false;
              #endif

              /*
              // do we need to ignore security info for "\\" folder ?
              #ifdef Z7_USE_SECURITY_CODE
              needSecurity = false;
              #endif
              */

              fullPath = CHAR_PATH_SEPARATOR;
            }
            #if defined(_WIN32) && !defined(UNDER_CE)
            else if (item.IsDriveItem())
            {
              needAltStreams = false;
              fullPath.Add_PathSepar();
            }
            #endif
          }
        }

        NFind::CFileInfo fi;
        #if defined(_WIN32) && !defined(UNDER_CE)
        if (IsVirtualFsFolder(phyPrefix, name))
        {
          fi.SetAsDir();
          fi.Name = us2fs(name);
        }
        else
        #endif
        if (!FindFile_KeepDots(fi, fullPath  FOLLOW_LINK_PARAM2))
        {
          RINOK(dirItems.AddError(fullPath))
          continue;
        }

        /*
        #ifdef _WIN32
          #define MY_ERROR_IS_DIR     ERROR_FILE_NOT_FOUND
          #define MY_ERROR_NOT_DIR    DI_DEFAULT_ERROR
        #else
          #define MY_ERROR_IS_DIR     EISDIR
          #define MY_ERROR_NOT_DIR    ENOTDIR
        #endif
        */

        const bool isDir = fi.IsDir();
        if (isDir ? !item.ForDir : !item.ForFile)
        {
          // RINOK(dirItems.AddError(fullPath, isDir ? MY_ERROR_IS_DIR: MY_ERROR_NOT_DIR));
          RINOK(dirItems.AddError(fullPath, DI_DEFAULT_ERROR))
          continue;
        }
        {
          UStringVector pathParts;
          pathParts.Add(fs2us(fi.Name));
          if (curNode.CheckPathToRoot(false, pathParts, !isDir))
            continue;
        }
        

       if (dirItems.CanIncludeItem(fi.IsDir()))
       {
        int secureIndex = -1;
        #ifdef Z7_USE_SECURITY_CODE
        if (needSecurity && dirItems.ReadSecure)
        {
          RINOK(dirItems.AddSecurityItem(fullPath, secureIndex))
        }
        #endif

        dirItems.AddDirFileInfo(phyParent, logParent, secureIndex, fi);

        // we don't scan AltStreams for link files

        #if !defined(UNDER_CE)
        {
          CDirItem &dirItem = dirItems.Items.Back();
          RINOK(dirItems.SetLinkInfo(dirItem, fi, phyPrefix))
          if (dirItem.ReparseData.Size() != 0)
            continue;
        }
        
        #if defined(_WIN32)
        if (needAltStreams && dirItems.ScanAltStreams && !fi.IsAltStream)
        {
          UStringVector pathParts;
          pathParts.Add(fs2us(fi.Name));
          RINOK(EnumerateAltStreams(fi, curNode, phyParent, logParent,
              fullPath,  // including (name)
              pathParts, // including (fi.Name)
              true, /* addAllSubStreams */
              dirItems))
        }
        #endif // defined(_WIN32)

        #endif // !defined(UNDER_CE)
       }


        #ifndef _WIN32
        if (!fi.IsPosixLink()) // posix link can follow to dir
        #endif
        if (!isDir)
          continue;

        UStringVector newParts;
        const NWildcard::CCensorNode *nextNode = NULL;
        int index = curNode.FindSubNode(name);
        if (index >= 0)
        {
          for (int t = (int)needEnterVector.Size(); t <= index; t++)
            needEnterVector.Add(true);
          needEnterVector[(unsigned)index] = false;
          nextNode = &curNode.SubNodes[(unsigned)index];
        }
        else
        {
         #ifndef _WIN32
          if (fi.IsPosixLink())
          {
            // here we can try to resolve posix link
            // if the link to dir, then can we follow it
            continue; // we don't follow posix link
          }
         #else
          if (dirItems.SymLinks)
          {
            if (fi.HasReparsePoint())
            {
              /* 20.03: in SymLinks mode: we don't enter to directory that
              has reparse point and has no CCensorNode */
              continue;
            }
          }
         #endif
          nextNode = &curNode;
          newParts.Add(name); // don't change it to fi.Name. It's for shortnames support
        }

        RINOK(EnumerateDirItems_Spec(*nextNode, phyParent, logParent, fi.Name, phyPrefix,
            newParts, dirItems, true))
      }

      for (i = 0; i < curNode.SubNodes.Size(); i++)
      {
        if (i < needEnterVector.Size())
          if (!needEnterVector[i])
            continue;
        const NWildcard::CCensorNode &nextNode = curNode.SubNodes[i];
        FString fullPath = phyPrefix + us2fs(nextNode.Name);
        NFind::CFileInfo fi;
        
        if (nextNode.Name.IsEmpty())
        {
          if (phyPrefix.IsEmpty())
            fullPath = CHAR_PATH_SEPARATOR;
        }
      #ifdef _WIN32
        else if(phyPrefix.IsEmpty()
            || (phyPrefix.Len() == NName::kSuperPathPrefixSize
                && IsSuperPath(phyPrefix)))
        {
          if (NWildcard::IsDriveColonName(nextNode.Name))
            fullPath.Add_PathSepar();
        }
      #endif

        // we don't want to call fi.Find() for root folder or virtual folder
        if ((phyPrefix.IsEmpty() && nextNode.Name.IsEmpty())
            #if defined(_WIN32) && !defined(UNDER_CE)
            || IsVirtualFsFolder(phyPrefix, nextNode.Name)
            #endif
            )
        {
          fi.SetAsDir();
          fi.Name = us2fs(nextNode.Name);
        }
        else
        {
          if (!FindFile_KeepDots(fi, fullPath  FOLLOW_LINK_PARAM2))
          {
            if (!nextNode.AreThereIncludeItems())
              continue;
            RINOK(dirItems.AddError(fullPath))
            continue;
          }
        
          if (!fi.IsDir())
          {
            RINOK(dirItems.AddError(fullPath, DI_DEFAULT_ERROR))
            continue;
          }
        }

        RINOK(EnumerateDirItems_Spec(nextNode, phyParent, logParent, fi.Name, phyPrefix,
            UStringVector(), dirItems, false))
      }

      return S_OK;
    }
  }

  #ifdef _WIN32
  #ifndef UNDER_CE

  // scan drives, if wildcard is "*:\"

  if (phyPrefix.IsEmpty() && curNode.IncludeItems.Size() > 0)
  {
    unsigned i;
    for (i = 0; i < curNode.IncludeItems.Size(); i++)
    {
      const NWildcard::CItem &item = curNode.IncludeItems[i];
      if (item.PathParts.Size() < 1)
        break;
      const UString &name = item.PathParts.Front();
      if (name.Len() != 2 || name[1] != ':')
        break;
      if (item.PathParts.Size() == 1)
        if (item.ForFile || !item.ForDir)
          break;
      if (NWildcard::IsDriveColonName(name))
        continue;
      if (name[0] != '*' && name[0] != '?')
        break;
    }
    if (i == curNode.IncludeItems.Size())
    {
      FStringVector driveStrings;
      NFind::MyGetLogicalDriveStrings(driveStrings);
      for (i = 0; i < driveStrings.Size(); i++)
      {
        FString driveName = driveStrings[i];
        if (driveName.Len() < 3 || driveName.Back() != '\\')
          return E_FAIL;
        driveName.DeleteBack();
        NFind::CFileInfo fi;
        fi.SetAsDir();
        fi.Name = driveName;

        RINOK(EnumerateForItem(fi, curNode, phyParent, logParent, phyPrefix,
            addParts, dirItems, enterToSubFolders))
      }
      return S_OK;
    }
  }
  
  #endif
  #endif


  CObjectVector<NFind::CFileInfo> files;
  
  // for (int y = 0; y < 1; y++)
  {
    // files.Clear();
    RINOK(dirItems.EnumerateOneDir(phyPrefix, files))
  /*
  FOR_VECTOR (i, files)
  {
    #ifdef _WIN32
    // const NFind::CFileInfo &fi = files[i];
    #else
    NFind::CFileInfo &fi = files[i];
    {
      const NFind::CFileInfo &di = files[i];
      const FString path = phyPrefix + di.Name;
      if (!fi.Find_AfterEnumerator(path))
      {
        RINOK(dirItems.AddError(path));
        continue;
      }
      fi.Name = di.Name;
    }
    #endif

  }
  */
  }

  FOR_VECTOR (i, files)
  {
    #ifdef _WIN32
    const NFind::CFileInfo &fi = files[i];
    #else
    const NFind::CFileInfo &fi = files[i];
    /*
    NFind::CFileInfo fi;
    {
      const NFind::CDirEntry &di = files[i];
      const FString path = phyPrefix + di.Name;
      if (!fi.Find_AfterEnumerator(path))
      {
        RINOK(dirItems.AddError(path));
        continue;
      }
      fi.Name = di.Name;
    }
    */
    #endif

    RINOK(EnumerateForItem(fi, curNode, phyParent, logParent, phyPrefix,
          addParts, dirItems, enterToSubFolders))
    if (dirItems.Callback && (i & kScanProgressStepMask) == kScanProgressStepMask)
    {
      RINOK(dirItems.ScanProgress(phyPrefix))
    }
  }

  return S_OK;
}




HRESULT EnumerateItems(
    const NWildcard::CCensor &censor,
    const NWildcard::ECensorPathMode pathMode,
    const UString &addPathPrefix, // prefix that will be added to Logical Path
    CDirItems &dirItems)
{
  FOR_VECTOR (i, censor.Pairs)
  {
    const NWildcard::CPair &pair = censor.Pairs[i];
    const int phyParent = pair.Prefix.IsEmpty() ? -1 : (int)dirItems.AddPrefix(-1, -1, pair.Prefix);
    int logParent = -1;
    
    if (pathMode == NWildcard::k_AbsPath)
      logParent = phyParent;
    else
    {
      if (!addPathPrefix.IsEmpty())
        logParent = (int)dirItems.AddPrefix(-1, -1, addPathPrefix);
    }
    
    RINOK(EnumerateDirItems(pair.Head, phyParent, logParent, us2fs(pair.Prefix), UStringVector(),
        dirItems,
        false // enterToSubFolders
        ))
  }
  dirItems.ReserveDown();

 #if defined(_WIN32) && !defined(UNDER_CE)
  RINOK(dirItems.FillFixedReparse())
 #endif

 #ifndef _WIN32
  RINOK(dirItems.FillDeviceSizes())
 #endif

  return S_OK;
}


#if defined(_WIN32) && !defined(UNDER_CE)

HRESULT CDirItems::FillFixedReparse()
{
  FOR_VECTOR(i, Items)
  {
    CDirItem &item = Items[i];

    if (!SymLinks)
    {
      // continue; // for debug
      if (!item.Has_Attrib_ReparsePoint())
        continue;
      /*
      We want to get properties of target file instead of properies of symbolic link.
      Probably this code is unused, because
      CFileInfo::Find(with followLink = true) called Fill_From_ByHandleFileInfo() already.
      */
      // if (item.IsDir()) continue;
      const FString phyPath = GetPhyPath(i);
      NFind::CFileInfo fi;
      if (fi.Fill_From_ByHandleFileInfo(phyPath)) // item.IsDir()
      {
        item.Size = fi.Size;
        item.CTime = fi.CTime;
        item.ATime = fi.ATime;
        item.MTime = fi.MTime;
        item.Attrib = fi.Attrib;
        continue;
      }
      RINOK(AddError(phyPath))
      continue;
    }

    // (SymLinks == true)
    if (item.ReparseData.Size() == 0)
      continue;
    // if (item.Size == 0)
    {
      // 20.03: we use Reparse Data instead of real data
      item.Size = item.ReparseData.Size();
    }
    
    CReparseAttr attr;
    if (!attr.Parse(item.ReparseData, item.ReparseData.Size()))
    {
      const FString phyPath = GetPhyPath(i);
      AddError(phyPath, attr.ErrorCode);
      continue;
    }

    /* imagex/WIM reduces absolute paths in links (raparse data),
       if we archive non root folder. We do same thing here */

    // bool isWSL = false;
    if (attr.IsSymLink_WSL())
    {
      // isWSL = true;
      // we don't change WSL symlinks
      continue;
    }
    else
    {
      if (attr.IsRelative_Win())
        continue;
    }

    const UString &link = attr.GetPath();
    if (!IsDrivePath(link))
      continue;
    // maybe we need to support networks paths also ?

    FString fullPathF;
    if (!NDir::MyGetFullPathName(GetPhyPath(i), fullPathF))
      continue;
    const UString fullPath = fs2us(fullPathF);
    const UString logPath = GetLogPath(i);
    if (logPath.Len() >= fullPath.Len())
      continue;
    if (CompareFileNames(logPath, fullPath.RightPtr(logPath.Len())) != 0)
      continue;
    
    const UString prefix = fullPath.Left(fullPath.Len() - logPath.Len());
    if (!IsPathSepar(prefix.Back()))
      continue;

    const unsigned rootPrefixSize = GetRootPrefixSize(prefix);
    if (rootPrefixSize == 0)
      continue;
    if (rootPrefixSize == prefix.Len())
      continue; // simple case: paths are from root
    if (link.Len() <= prefix.Len())
      continue;
    if (CompareFileNames(link.Left(prefix.Len()), prefix) != 0)
      continue;

    UString newLink = prefix.Left(rootPrefixSize);
    newLink += link.Ptr(prefix.Len());

    CByteBuffer &data = item.ReparseData2;
/*
    if (isWSL)
    {
      Convert_WinPath_to_WslLinuxPath(newLink, true); // is absolute : change it
      FillLinkData_WslLink(data, newLink);
    }
    else
*/
      FillLinkData_WinLink(data, newLink, !attr.IsMountPoint());
    if (data.Size() == 0)
      continue;
    // item.ReparseData2 = data;
  }
  return S_OK;
}

#endif


#ifndef _WIN32

HRESULT CDirItems::FillDeviceSizes()
{
  {
    FOR_VECTOR (i, Items)
    {
      CDirItem &item = Items[i];
      
      if (S_ISBLK(item.mode) && item.Size == 0)
      {
        const FString phyPath = GetPhyPath(i);
        NIO::CInFile inFile;
        inFile.PreserveATime = true;
        if (inFile.OpenShared(phyPath, ShareForWrite)) // fixme: OpenShared ??
        {
          UInt64 size = 0;
          if (inFile.GetLength(size))
            item.Size = size;
        }
      }
      if (StoreOwnerName)
      {
        OwnerNameMap.Add_UInt32(item.uid);
        OwnerGroupMap.Add_UInt32(item.gid);
      }
    }
  }

  if (StoreOwnerName)
  {
    UString u;
    AString a;
    {
      FOR_VECTOR (i, OwnerNameMap.Numbers)
      {
        // 200K/sec speed
        u.Empty();
        const passwd *pw = getpwuid(OwnerNameMap.Numbers[i]);
        // printf("\ngetpwuid=%s\n", pw->pw_name);
        if (pw)
        {
          a = pw->pw_name;
          ConvertUTF8ToUnicode(a, u);
        }
        OwnerNameMap.Strings.Add(u);
      }
    }
    {
      FOR_VECTOR (i, OwnerGroupMap.Numbers)
      {
        u.Empty();
        const group *gr = getgrgid(OwnerGroupMap.Numbers[i]);
        if (gr)
        {
          // printf("\ngetgrgid %d %s\n", OwnerGroupMap.Numbers[i], gr->gr_name);
          a = gr->gr_name;
          ConvertUTF8ToUnicode(a, u);
        }
        OwnerGroupMap.Strings.Add(u);
      }
    }
    
    FOR_VECTOR (i, Items)
    {
      CDirItem &item = Items[i];
      {
        const int index = OwnerNameMap.Find(item.uid);
        if (index < 0) throw 1;
        item.OwnerNameIndex = index;
      }
      {
        const int index = OwnerGroupMap.Find(item.gid);
        if (index < 0) throw 1;
        item.OwnerGroupIndex = index;
      }
    }
  }
      

  // if (NeedOwnerNames)
  {
    /*
    {
      for (unsigned i = 0 ; i < 10000; i++)
      {
        const passwd *pw = getpwuid(i);
        if (pw)
        {
          UString u;
          ConvertUTF8ToUnicode(AString(pw->pw_name), u);
          OwnerNameMap.Add(i, u);
          OwnerNameMap.Add(i, u);
          OwnerNameMap.Add(i, u);
        }
        const group *gr = getgrgid(i);
        if (gr)
        {
          // we can use utf-8 here.
          UString u;
          ConvertUTF8ToUnicode(AString(gr->gr_name), u);
          OwnerGroupMap.Add(i, u);
        }
      }
    }
    */
    /*
    {
      FOR_VECTOR (i, OwnerNameMap.Strings)
      {
        AString s;
        ConvertUnicodeToUTF8(OwnerNameMap.Strings[i], s);
        printf("\n%5d %s", (unsigned)OwnerNameMap.Numbers[i], s.Ptr());
      }
    }
    {
      printf("\n\n=========Groups\n");
      FOR_VECTOR (i, OwnerGroupMap.Strings)
      {
        AString s;
        ConvertUnicodeToUTF8(OwnerGroupMap.Strings[i], s);
        printf("\n%5d %s", (unsigned)OwnerGroupMap.Numbers[i], s.Ptr());
      }
    }
    */
  }
      /*
      for (unsigned i = 0 ; i < 100000000; i++)
      {
        // const passwd *pw = getpwuid(1000);
        // pw = pw;
        int pos = OwnerNameMap.Find(1000);
        if (pos < 0 - (int)i)
          throw 1;
      }
      */

  return S_OK;
}

#endif



static const char * const kCannotFindArchive = "Cannot find archive";

HRESULT EnumerateDirItemsAndSort(
    NWildcard::CCensor &censor,
    NWildcard::ECensorPathMode censorPathMode,
    const UString &addPathPrefix,
    UStringVector &sortedPaths,
    UStringVector &sortedFullPaths,
    CDirItemsStat &st,
    IDirItemsCallback *callback)
{
  FStringVector paths;
  
  {
    CDirItems dirItems;
    dirItems.Callback = callback;
    {
      HRESULT res = EnumerateItems(censor, censorPathMode, addPathPrefix, dirItems);
      st = dirItems.Stat;
      RINOK(res)
    }
  
    FOR_VECTOR (i, dirItems.Items)
    {
      const CDirItem &dirItem = dirItems.Items[i];
      if (!dirItem.IsDir())
        paths.Add(dirItems.GetPhyPath(i));
    }
  }
  
  if (paths.Size() == 0)
  {
    // return S_OK;
    throw CMessagePathException(kCannotFindArchive);
  }
  
  UStringVector fullPaths;
  
  unsigned i;
  
  for (i = 0; i < paths.Size(); i++)
  {
    FString fullPath;
    NFile::NDir::MyGetFullPathName(paths[i], fullPath);
    fullPaths.Add(fs2us(fullPath));
  }
  
  CUIntVector indices;
  SortFileNames(fullPaths, indices);
  sortedPaths.ClearAndReserve(indices.Size());
  sortedFullPaths.ClearAndReserve(indices.Size());

  for (i = 0; i < indices.Size(); i++)
  {
    unsigned index = indices[i];
    sortedPaths.AddInReserved(fs2us(paths[index]));
    sortedFullPaths.AddInReserved(fullPaths[index]);
    if (i > 0 && CompareFileNames(sortedFullPaths[i], sortedFullPaths[i - 1]) == 0)
      throw CMessagePathException("Duplicate archive path:", sortedFullPaths[i]);
  }

  return S_OK;
}




#ifdef _WIN32

static bool IsDotsName(const wchar_t *s)
{
  return s[0] == '.' && (s[1] == 0 || (s[1] == '.' && s[2] == 0));
}

// This code converts all short file names to long file names.

static void ConvertToLongName(const UString &prefix, UString &name)
{
  if (name.IsEmpty()
      || DoesNameContainWildcard(name)
      || IsDotsName(name))
    return;
  NFind::CFileInfo fi;
  const FString path (us2fs(prefix + name));
  #ifndef UNDER_CE
  if (NFile::NName::IsDevicePath(path))
    return;
  #endif
  if (fi.Find(path))
    name = fs2us(fi.Name);
}

static void ConvertToLongNames(const UString &prefix, CObjectVector<NWildcard::CItem> &items)
{
  FOR_VECTOR (i, items)
  {
    NWildcard::CItem &item = items[i];
    if (item.Recursive || item.PathParts.Size() != 1)
      continue;
    if (prefix.IsEmpty() && item.IsDriveItem())
      continue;
    ConvertToLongName(prefix, item.PathParts.Front());
  }
}

static void ConvertToLongNames(const UString &prefix, NWildcard::CCensorNode &node)
{
  ConvertToLongNames(prefix, node.IncludeItems);
  ConvertToLongNames(prefix, node.ExcludeItems);
  unsigned i;
  for (i = 0; i < node.SubNodes.Size(); i++)
  {
    UString &name = node.SubNodes[i].Name;
    if (prefix.IsEmpty() && NWildcard::IsDriveColonName(name))
      continue;
    ConvertToLongName(prefix, name);
  }
  // mix folders with same name
  for (i = 0; i < node.SubNodes.Size(); i++)
  {
    NWildcard::CCensorNode &nextNode1 = node.SubNodes[i];
    for (unsigned j = i + 1; j < node.SubNodes.Size();)
    {
      const NWildcard::CCensorNode &nextNode2 = node.SubNodes[j];
      if (nextNode1.Name.IsEqualTo_NoCase(nextNode2.Name))
      {
        nextNode1.IncludeItems += nextNode2.IncludeItems;
        nextNode1.ExcludeItems += nextNode2.ExcludeItems;
        node.SubNodes.Delete(j);
      }
      else
        j++;
    }
  }
  for (i = 0; i < node.SubNodes.Size(); i++)
  {
    NWildcard::CCensorNode &nextNode = node.SubNodes[i];
    ConvertToLongNames(prefix + nextNode.Name + WCHAR_PATH_SEPARATOR, nextNode);
  }
}

void ConvertToLongNames(NWildcard::CCensor &censor)
{
  FOR_VECTOR (i, censor.Pairs)
  {
    NWildcard::CPair &pair = censor.Pairs[i];
    ConvertToLongNames(pair.Prefix, pair.Head);
  }
}

#endif


CMessagePathException::CMessagePathException(const char *a, const wchar_t *u)
{
  (*this) += a;
  if (u)
  {
    Add_LF();
    (*this) += u;
  }
}

CMessagePathException::CMessagePathException(const wchar_t *a, const wchar_t *u)
{
  (*this) += a;
  if (u)
  {
    Add_LF();
    (*this) += u;
  }
}
