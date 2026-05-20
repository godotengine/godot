// UpdateCallback.cpp

#include "StdAfx.h"

// #include <stdio.h>

#ifndef _WIN32
// #include <grp.h>
// #include <pwd.h>
// for major()/minor():
#if defined(__APPLE__) || defined(__DragonFly__) || \
    defined(BSD) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
#include <sys/types.h>
#else
#include <sys/sysmacros.h>
#endif

#endif // _WIN32

#ifndef Z7_ST
#include "../../../Windows/Synchronization.h"
#endif

#include "../../../Common/ComTry.h"
#include "../../../Common/IntToString.h"
#include "../../../Common/StringConvert.h"
#include "../../../Common/Wildcard.h"
#include "../../../Common/UTFConvert.h"

#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileName.h"
#include "../../../Windows/PropVariant.h"

#include "../../Common/StreamObjects.h"
#include "../../Archive/Common/ItemNameUtils.h"

#include "UpdateCallback.h"

#if defined(_WIN32) && !defined(UNDER_CE)
#define Z7_USE_SECURITY_CODE
#include "../../../Windows/SecurityUtils.h"
#endif

using namespace NWindows;
using namespace NFile;

#ifndef Z7_ST
static NSynchronization::CCriticalSection g_CriticalSection;
#define MT_LOCK NSynchronization::CCriticalSectionLock lock(g_CriticalSection);
#else
#define MT_LOCK
#endif


#ifdef Z7_USE_SECURITY_CODE
bool InitLocalPrivileges();
#endif

CArchiveUpdateCallback::CArchiveUpdateCallback():
    PreserveATime(false),
    ShareForWrite(false),
    StopAfterOpenError(false),
    StdInMode(false),
    
    KeepOriginalItemNames(false),
    StoreNtSecurity(false),
    StoreHardLinks(false),
    StoreSymLinks(false),

   #ifndef _WIN32
    StoreOwnerId(false),
    StoreOwnerName(false),
   #endif

    /*
    , Need_ArcMTime_Report(false),
    , ArcMTime_WasReported(false),
    */
    Need_LatestMTime(false),
    LatestMTime_Defined(false),
    
    Callback(NULL),
  
    DirItems(NULL),
    ParentDirItem(NULL),
    
    Arc(NULL),
    ArcItems(NULL),
    UpdatePairs(NULL),
    NewNames(NULL),
    Comment(NULL),
    CommentIndex(-1),
    
    ProcessedItemsStatuses(NULL),
    _hardIndex_From((UInt32)(Int32)-1)
{
  #ifdef Z7_USE_SECURITY_CODE
  _saclEnabled = InitLocalPrivileges();
  #endif
}


Z7_COM7F_IMF(CArchiveUpdateCallback::SetTotal(UInt64 size))
{
  COM_TRY_BEGIN
  return Callback->SetTotal(size);
  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::SetCompleted(const UInt64 *completeValue))
{
  COM_TRY_BEGIN
  return Callback->SetCompleted(completeValue);
  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::SetRatioInfo(const UInt64 *inSize, const UInt64 *outSize))
{
  COM_TRY_BEGIN
  return Callback->SetRatioInfo(inSize, outSize);
  COM_TRY_END
}


/*
static const CStatProp kProps[] =
{
  { NULL, kpidPath, VT_BSTR},
  { NULL, kpidIsDir, VT_BOOL},
  { NULL, kpidSize, VT_UI8},
  { NULL, kpidCTime, VT_FILETIME},
  { NULL, kpidATime, VT_FILETIME},
  { NULL, kpidMTime, VT_FILETIME},
  { NULL, kpidAttrib, VT_UI4},
  { NULL, kpidIsAnti, VT_BOOL}
};

Z7_COM7F_IMF(CArchiveUpdateCallback::EnumProperties(IEnumSTATPROPSTG **)
{
  return CStatPropEnumerator::CreateEnumerator(kProps, Z7_ARRAY_SIZE(kProps), enumerator);
}
*/

Z7_COM7F_IMF(CArchiveUpdateCallback::GetUpdateItemInfo(UInt32 index,
      Int32 *newData, Int32 *newProps, UInt32 *indexInArchive))
{
  COM_TRY_BEGIN
  RINOK(Callback->CheckBreak())
  const CUpdatePair2 &up = (*UpdatePairs)[index];
  if (newData) *newData = BoolToInt(up.NewData);
  if (newProps) *newProps = BoolToInt(up.NewProps);
  if (indexInArchive)
  {
    *indexInArchive = (UInt32)(Int32)-1;
    if (up.ExistInArchive())
      *indexInArchive = ArcItems ? (*ArcItems)[(unsigned)up.ArcIndex].IndexInServer : (UInt32)(Int32)up.ArcIndex;
  }
  return S_OK;
  COM_TRY_END
}


Z7_COM7F_IMF(CArchiveUpdateCallback::GetRootProp(PROPID propID, PROPVARIANT *value))
{
  NCOM::CPropVariant prop;
  switch (propID)
  {
    case kpidIsDir:  prop = true; break;
    case kpidAttrib: if (ParentDirItem) prop = ParentDirItem->GetWinAttrib(); break;
    case kpidCTime:  if (ParentDirItem) PropVariant_SetFrom_FiTime(prop, ParentDirItem->CTime); break;
    case kpidATime:  if (ParentDirItem) PropVariant_SetFrom_FiTime(prop, ParentDirItem->ATime); break;
    case kpidMTime:  if (ParentDirItem) PropVariant_SetFrom_FiTime(prop, ParentDirItem->MTime); break;
    case kpidArcFileName:  if (!ArcFileName.IsEmpty()) prop = ArcFileName; break;
    default: break;
  }
  prop.Detach(value);
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetParent(UInt32 /* index */, UInt32 *parent, UInt32 *parentType))
{
  *parentType = NParentType::kDir;
  *parent = (UInt32)(Int32)-1;
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetNumRawProps(UInt32 *numProps))
{
  *numProps = 0;
  if (StoreNtSecurity)
    *numProps = 1;
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetRawPropInfo(UInt32 /* index */, BSTR *name, PROPID *propID))
{
  *name = NULL;
  *propID = kpidNtSecure;
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetRootRawProp(PROPID
    propID
    , const void **data, UInt32 *dataSize, UInt32 *propType))
{
  #ifndef Z7_USE_SECURITY_CODE
  UNUSED_VAR(propID)
  #endif

  *data = NULL;
  *dataSize = 0;
  *propType = 0;
  if (!StoreNtSecurity)
    return S_OK;
  #ifdef Z7_USE_SECURITY_CODE
  if (propID == kpidNtSecure)
  {
    if (StdInMode)
      return S_OK;

    if (ParentDirItem)
    {
      if (ParentDirItem->SecureIndex < 0)
        return S_OK;
      const CByteBuffer &buf = DirItems->SecureBlocks.Bufs[(unsigned)ParentDirItem->SecureIndex];
      *data = buf;
      *dataSize = (UInt32)buf.Size();
      *propType = NPropDataType::kRaw;
      return S_OK;
    }

    if (Arc && Arc->GetRootProps)
      return Arc->GetRootProps->GetRootRawProp(propID, data, dataSize, propType);
  }
  #endif
  return S_OK;
}


Z7_COM7F_IMF(CArchiveUpdateCallback::GetRawProp(UInt32 index, PROPID propID, const void **data, UInt32 *dataSize, UInt32 *propType))
{
  *data = NULL;
  *dataSize = 0;
  *propType = 0;

  if (propID == kpidNtSecure ||
      propID == kpidNtReparse)
  {
    if (StdInMode)
      return S_OK;

    const CUpdatePair2 &up = (*UpdatePairs)[index];
    if (up.UseArcProps && up.ExistInArchive() && Arc->GetRawProps)
      return Arc->GetRawProps->GetRawProp(
          ArcItems ? (*ArcItems)[(unsigned)up.ArcIndex].IndexInServer : (UInt32)(Int32)up.ArcIndex,
          propID, data, dataSize, propType);
    {
      /*
      if (!up.NewData)
        return E_FAIL;
      */
      if (up.IsAnti)
        return S_OK;
      
      #if defined(_WIN32) && !defined(UNDER_CE)
      const CDirItem &di = DirItems->Items[(unsigned)up.DirIndex];
      #endif

      #ifdef Z7_USE_SECURITY_CODE
      if (propID == kpidNtSecure)
      {
        if (!StoreNtSecurity)
          return S_OK;
        if (di.SecureIndex < 0)
          return S_OK;
        const CByteBuffer &buf = DirItems->SecureBlocks.Bufs[(unsigned)di.SecureIndex];
        *data = buf;
        *dataSize = (UInt32)buf.Size();
        *propType = NPropDataType::kRaw;
      }
      else
      #endif
      if (propID == kpidNtReparse)
      {
        if (!StoreSymLinks)
          return S_OK;
        #if defined(_WIN32) && !defined(UNDER_CE)
        // we use ReparseData2 instead of ReparseData for WIM format
        const CByteBuffer *buf = &di.ReparseData2;
        if (buf->Size() == 0)
          buf = &di.ReparseData;
        if (buf->Size() != 0)
        {
          *data = *buf;
          *dataSize = (UInt32)buf->Size();
          *propType = NPropDataType::kRaw;
        }
        #endif
      }
 
      return S_OK;
    }
  }

  return S_OK;
}

#if defined(_WIN32) && !defined(UNDER_CE)

static UString GetRelativePath(const UString &to, const UString &from, bool isWSL)
{
  UStringVector partsTo, partsFrom;
  SplitPathToParts(to, partsTo);
  SplitPathToParts(from, partsFrom);

  unsigned i;
  for (i = 0;; i++)
  {
    if (i + 1 >= partsFrom.Size() ||
        i + 1 >= partsTo.Size())
      break;
    if (CompareFileNames(partsFrom[i], partsTo[i]) != 0)
      break;
  }

  if (i == 0)
  {
#ifdef _WIN32
    if (isWSL ||
       (NName::IsDrivePath(to) ||
        NName::IsDrivePath(from)))
      return to;
#endif
  }

  UString s;
  unsigned k;
  
  for (k = i + 1; k < partsFrom.Size(); k++)
    s += ".." STRING_PATH_SEPARATOR;
  
  for (k = i; k < partsTo.Size(); k++)
  {
    if (k != i)
      s.Add_PathSepar();
    s += partsTo[k];
  }

  return s;
}

#endif

Z7_COM7F_IMF(CArchiveUpdateCallback::GetProperty(UInt32 index, PROPID propID, PROPVARIANT *value))
{
  COM_TRY_BEGIN
  const CUpdatePair2 &up = (*UpdatePairs)[index];
  NCOM::CPropVariant prop;

  if (up.NewData)
  {
    /*
    if (propID == kpidIsHardLink)
    {
      prop = _isHardLink;
      prop.Detach(value);
      return S_OK;
    }
    */
    if (propID == kpidSymLink)
    {
      if (index == _hardIndex_From)
      {
        prop.Detach(value);
        return S_OK;
      }
      
#if !defined(UNDER_CE)
      if (up.DirIndex >= 0)
      {
        const CDirItem &di = DirItems->Items[(unsigned)up.DirIndex];
        if (di.ReparseData.Size())
        {
#ifdef _WIN32
          CReparseAttr attr;
          if (attr.Parse(di.ReparseData, di.ReparseData.Size()))
          {
            UString path = attr.GetPath();
            if (!path.IsEmpty())
            {
              bool isWSL = attr.IsSymLink_WSL();
              if (isWSL)
                NArchive::NItemName::ReplaceToWinSlashes(path, true); // useBackslashReplacement
              // it's expected that (path) now uses windows slashes.
              // CReparseAttr::IsRelative_Win() returns true if FLAG_RELATIVE is set
              // CReparseAttr::IsRelative_Win() returns true for "\dir1\path"
              // but we want to store real relative paths without "\" root prefix.
              // so we parse path instead of IsRelative_Win() calling.
              if (// attr.IsRelative_Win() ||
                  (isWSL ?
                   IS_PATH_SEPAR(path[0]) :
                   NName::IsAbsolutePath(path)))
              {
                // (path) is abolute path or relative to root: "\path"
                // we try to convert (path) to relative path for writing to archive.
                const FString phyPath = DirItems->GetPhyPath((unsigned)up.DirIndex);
                FString fullPath;
                if (NDir::MyGetFullPathName(phyPath, fullPath))
                {
                  if (IS_PATH_SEPAR(path[0]) &&
                      !IS_PATH_SEPAR(path[1]))
                  {
                    // path is relative to root of (fullPath): "\path"
                    const unsigned prefixSize = NName::GetRootPrefixSize(fullPath);
                    if (prefixSize)
                    {
                      path.DeleteFrontal(1);
                      path.Insert(0, fs2us(fullPath.Left(prefixSize)));
                      // we have changed "\" prefix to drive prefix "c:\" in (path).
                      // (path) is Windows path now.
                      isWSL = false;
                    }
                  }
                }
                path = GetRelativePath(path, fs2us(fullPath), isWSL);
              }
#if WCHAR_PATH_SEPARATOR != L'/'
              // 7-Zip's TAR handler in Windows replaces windows slashes to linux slashes.
              // so we can return any slashes to TAR handler.
              // or we can convert to linux slashes here,
              // because input IInArchive handler uses linux slashes for kpidSymLink.
              // path.Replace(WCHAR_PATH_SEPARATOR, L'/');
#endif
              if (!path.IsEmpty())
                prop = path;
            }
          }
#else // ! _WIN32
          AString utf;
          utf.SetFrom_CalcLen((const char *)(const Byte *)di.ReparseData, (unsigned)di.ReparseData.Size());
    #if 0 // 0 - for debug
          // it's expected that link data uses system codepage.
          // fs2us() ignores conversion errors. But we want correct path
          UString us (fs2us(utf));
    #else
          UString us;
          if (ConvertUTF8ToUnicode(utf, us))
    #endif
          {
            if (!us.IsEmpty())
              prop = us;
          }
#endif // ! _WIN32
        }
        prop.Detach(value);
        return S_OK;
      }
#endif // !defined(UNDER_CE)
    }
    else if (propID == kpidHardLink)
    {
      if (index == _hardIndex_From)
      {
        const CKeyKeyValPair &pair = _map[_hardIndex_To];
        const CUpdatePair2 &up2 = (*UpdatePairs)[pair.Value];
        const UString path = DirItems->GetLogPath((unsigned)up2.DirIndex);
#if WCHAR_PATH_SEPARATOR != L'/'
        // 7-Zip's TAR handler in Windows replaces windows slashes to linux slashes.
        // path.Replace(WCHAR_PATH_SEPARATOR, L'/');
#endif
        prop = path;
        prop.Detach(value);
        return S_OK;
      }
      if (up.DirIndex >= 0)
      {
        prop.Detach(value);
        return S_OK;
      }
    }
  } // if (up.NewData)
  
  if (up.IsAnti
      && propID != kpidIsDir
      && propID != kpidPath
      && propID != kpidIsAltStream)
  {
    switch (propID)
    {
      case kpidSize:  prop = (UInt64)0; break;
      case kpidIsAnti:  prop = true; break;
      default: break;
    }
  }
  else if (propID == kpidPath && up.NewNameIndex >= 0)
    prop = (*NewNames)[(unsigned)up.NewNameIndex];
  else if (propID == kpidComment
      && CommentIndex >= 0
      && (unsigned)CommentIndex == index
      && Comment)
    prop = *Comment;
  else if (propID == kpidShortName && up.NewNameIndex >= 0 && up.IsMainRenameItem)
  {
    // we can generate new ShortName here;
  }
  else if ((up.UseArcProps || (KeepOriginalItemNames && (propID == kpidPath || propID == kpidIsAltStream)))
      && up.ExistInArchive() && Archive)
    return Archive->GetProperty(ArcItems ? (*ArcItems)[(unsigned)up.ArcIndex].IndexInServer : (UInt32)(Int32)up.ArcIndex, propID, value);
  else if (up.ExistOnDisk())
  {
    const CDirItem &di = DirItems->Items[(unsigned)up.DirIndex];
    switch (propID)
    {
      case kpidPath:  prop = DirItems->GetLogPath((unsigned)up.DirIndex); break;
      case kpidIsDir:  prop = di.IsDir(); break;
      case kpidSize:  prop = (UInt64)(di.IsDir() ? (UInt64)0 : di.Size); break;
      case kpidCTime:  PropVariant_SetFrom_FiTime(prop, di.CTime); break;
      case kpidATime:  PropVariant_SetFrom_FiTime(prop, di.ATime); break;
      case kpidMTime:  PropVariant_SetFrom_FiTime(prop, di.MTime); break;
      case kpidAttrib:  /* if (di.Attrib_IsDefined) */ prop = (UInt32)di.GetWinAttrib(); break;
      case kpidPosixAttrib: /* if (di.Attrib_IsDefined) */ prop = (UInt32)di.GetPosixAttrib(); break;
    
    #if defined(_WIN32)
      case kpidIsAltStream:  prop = di.IsAltStream; break;
      // case kpidShortName:  prop = di.ShortName; break;
    #else

        #if defined(__APPLE__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wsign-conversion"
        #endif

      case kpidDeviceMajor:
        /*
        printf("\ndi.mode = %o\n", di.mode);
        printf("\nst.st_rdev major = %d\n", (unsigned)major(di.rdev));
        printf("\nst.st_rdev minor = %d\n", (unsigned)minor(di.rdev));
        */
        if (S_ISCHR(di.mode) || S_ISBLK(di.mode))
          prop = (UInt32)major(di.rdev);
        break;
        
      case kpidDeviceMinor:
        if (S_ISCHR(di.mode) || S_ISBLK(di.mode))
          prop = (UInt32)minor(di.rdev);
        break;

        #if defined(__APPLE__)
        #pragma GCC diagnostic pop
        #endif

      // case kpidDevice: if (S_ISCHR(di.mode) || S_ISBLK(di.mode)) prop = (UInt64)(di.rdev); break;

      case kpidUserId:  if (StoreOwnerId) prop = (UInt32)di.uid; break;
      case kpidGroupId: if (StoreOwnerId) prop = (UInt32)di.gid; break;
      case kpidUser:
        if (di.OwnerNameIndex >= 0)
          prop = DirItems->OwnerNameMap.Strings[(unsigned)di.OwnerNameIndex];
        break;
      case kpidGroup:
        if (di.OwnerGroupIndex >= 0)
          prop = DirItems->OwnerGroupMap.Strings[(unsigned)di.OwnerGroupIndex];
        break;
     #endif
      default: break;
    }
  }
  prop.Detach(value);
  return S_OK;
  COM_TRY_END
}

#ifndef Z7_ST
static NSynchronization::CCriticalSection g_CS;
#endif

void CArchiveUpdateCallback::UpdateProcessedItemStatus(unsigned dirIndex)
{
  if (ProcessedItemsStatuses)
  {
    #ifndef Z7_ST
    NSynchronization::CCriticalSectionLock lock(g_CS);
    #endif
    ProcessedItemsStatuses[dirIndex] = 1;
  }
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetStream2(UInt32 index, ISequentialInStream **inStream, UInt32 mode))
{
  COM_TRY_BEGIN
  *inStream = NULL;
  const CUpdatePair2 &up = (*UpdatePairs)[index];
  if (!up.NewData)
    return E_FAIL;
  
  RINOK(Callback->CheckBreak())
  // RINOK(Callback->Finalize());

  bool isDir = IsDir(up);

  if (up.IsAnti)
  {
    UString name;
    if (up.ArcIndex >= 0)
      name = (*ArcItems)[(unsigned)up.ArcIndex].Name;
    else if (up.DirIndex >= 0)
      name = DirItems->GetLogPath((unsigned)up.DirIndex);
    RINOK(Callback->GetStream(name, isDir, true, mode))
    
    /* 9.33: fixed. Handlers expect real stream object for files, even for anti-file.
       so we return empty stream */

    if (!isDir)
    {
      CBufInStream *inStreamSpec = new CBufInStream();
      CMyComPtr<ISequentialInStream> inStreamLoc = inStreamSpec;
      inStreamSpec->Init(NULL, 0);
      *inStream = inStreamLoc.Detach();
    }
    return S_OK;
  }
  
  RINOK(Callback->GetStream(DirItems->GetLogPath((unsigned)up.DirIndex), isDir, false, mode))
 
  if (isDir)
    return S_OK;

  if (StdInMode)
  {
    if (mode != NUpdateNotifyOp::kAdd &&
        mode != NUpdateNotifyOp::kUpdate)
      return S_OK;

#if 1
    CStdInFileStream *inStreamSpec = new CStdInFileStream;
    CMyComPtr<ISequentialInStream> inStreamLoc(inStreamSpec);
#else
    CMyComPtr<ISequentialInStream> inStreamLoc;
    if (!CreateStdInStream(inStreamLoc))
      return GetLastError_noZero_HRESULT();
#endif
    *inStream = inStreamLoc.Detach();
  }
  else
  {
    #if !defined(UNDER_CE)
    const CDirItem &di = DirItems->Items[(unsigned)up.DirIndex];
    if (di.AreReparseData())
    {
      /*
      // we still need DeviceIoControlOut() instead of Read
      if (!inStreamSpec->File.OpenReparse(path))
      {
        return Callback->OpenFileError(path, ::GetLastError());
      }
      */
      // 20.03: we use Reparse Data instead of real data

      CBufInStream *inStreamSpec = new CBufInStream();
      CMyComPtr<ISequentialInStream> inStreamLoc = inStreamSpec;
      inStreamSpec->Init(di.ReparseData, di.ReparseData.Size());
      *inStream = inStreamLoc.Detach();

      UpdateProcessedItemStatus((unsigned)up.DirIndex);
      return S_OK;
    }
    #endif // !defined(UNDER_CE)

    CInFileStream *inStreamSpec = new CInFileStream;
    CMyComPtr<ISequentialInStream> inStreamLoc(inStreamSpec);

   /*
   // for debug:
   #ifdef _WIN32
    inStreamSpec->StoreOwnerName = true;
    inStreamSpec->OwnerName = "user_name";
    inStreamSpec->OwnerName += di.Name;
    inStreamSpec->OwnerName += "11111111112222222222222333333333333";
    inStreamSpec->OwnerGroup = "gname_";
    inStreamSpec->OwnerGroup += inStreamSpec->OwnerName;
   #endif
   */

   #ifndef _WIN32
    inStreamSpec->StoreOwnerId = StoreOwnerId;
    inStreamSpec->StoreOwnerName = StoreOwnerName;

    // if (StoreOwner)
    {
      inStreamSpec->_uid = di.uid;
      inStreamSpec->_gid = di.gid;
      if (di.OwnerNameIndex >= 0)
        inStreamSpec->OwnerName = DirItems->OwnerNameMap.Strings[(unsigned)di.OwnerNameIndex];
      if (di.OwnerGroupIndex >= 0)
        inStreamSpec->OwnerGroup = DirItems->OwnerGroupMap.Strings[(unsigned)di.OwnerGroupIndex];
    }
   #endif

    inStreamSpec->SupportHardLinks = StoreHardLinks;
    const bool preserveATime = (PreserveATime
        || mode == NUpdateNotifyOp::kAnalyze);   // 22.00 : we don't change access time in Analyze pass.
    inStreamSpec->Set_PreserveATime(preserveATime);

    const FString path = DirItems->GetPhyPath((unsigned)up.DirIndex);
    _openFiles_Indexes.Add(index);
    _openFiles_Paths.Add(path);
    // _openFiles_Streams.Add(inStreamSpec);

    /* 21.02 : we set Callback/CallbackRef after _openFiles_Indexes adding
       for correct working if exception was raised in GetPhyPath */
    inStreamSpec->Callback = this;
    inStreamSpec->CallbackRef = index;

    if (!inStreamSpec->OpenShared(path, ShareForWrite))
    {
      bool isOpen = false;
      if (preserveATime)
      {
        inStreamSpec->Set_PreserveATime(false);
        isOpen = inStreamSpec->OpenShared(path, ShareForWrite);
      }
      if (!isOpen)
      {
        const DWORD error = ::GetLastError();
        const HRESULT hres = Callback->OpenFileError(path, error);
        if (hres == S_OK || hres == S_FALSE)
        if (StopAfterOpenError ||
            // v23: we check also for some critical errors:
            #ifdef _WIN32
              error == ERROR_NO_SYSTEM_RESOURCES
            #else
              error == EMFILE
            #endif
            )
        {
          if (error == 0)
            return E_FAIL;
          return HRESULT_FROM_WIN32(error);
        }
        return hres;
      }
    }

    /*
    {
      // for debug:
      Byte b = 0;
      UInt32 processedSize = 0;
      if (inStreamSpec->Read(&b, 1, &processedSize) != S_OK ||
          processedSize != 1)
        return E_FAIL;
    }
    */

    if (Need_LatestMTime)
    {
      inStreamSpec->ReloadProps();
    }

    // #if defined(Z7_FILE_STREAMS_USE_WIN_FILE) || !defined(_WIN32)
    if (StoreHardLinks)
    {
      CStreamFileProps props;
      if (inStreamSpec->GetProps2(&props) == S_OK)
      {
        if (props.NumLinks > 1)
        {
          CKeyKeyValPair pair;
          pair.Key1 = props.VolID;
          pair.Key2 = props.FileID_Low;
          pair.Value = index;
          const unsigned numItems = _map.Size();
          const unsigned pairIndex = _map.AddToUniqueSorted2(pair);
          if (numItems == _map.Size())
          {
            // const CKeyKeyValPair &pair2 = _map.Pairs[pairIndex];
            _hardIndex_From = index;
            _hardIndex_To = pairIndex;
            // we could return NULL as stream, but it's better to return real stream
            // return S_OK;
          }
        }
      }
    }
    // #endif

    UpdateProcessedItemStatus((unsigned)up.DirIndex);
    *inStream = inStreamLoc.Detach();
  }
  
  return S_OK;
  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::SetOperationResult(Int32 opRes))
{
  COM_TRY_BEGIN
  return Callback->SetOperationResult(opRes);
  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetStream(UInt32 index, ISequentialInStream **inStream))
{
  COM_TRY_BEGIN
  return GetStream2(index, inStream,
      (*UpdatePairs)[index].ArcIndex < 0 ?
          NUpdateNotifyOp::kAdd :
          NUpdateNotifyOp::kUpdate);
  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::ReportOperation(UInt32 indexType, UInt32 index, UInt32 op))
{
  COM_TRY_BEGIN

  // if (op == NUpdateNotifyOp::kOpFinished) return Callback->ReportFinished(indexType, index);

  bool isDir = false;

  if (indexType == NArchive::NEventIndexType::kOutArcIndex)
  {
    UString name;
    if (index != (UInt32)(Int32)-1)
    {
      const CUpdatePair2 &up = (*UpdatePairs)[index];
      if (up.ExistOnDisk())
      {
        name = DirItems->GetLogPath((unsigned)up.DirIndex);
        isDir = DirItems->Items[(unsigned)up.DirIndex].IsDir();
      }
    }
    return Callback->ReportUpdateOperation(op, name.IsEmpty() ? NULL : name.Ptr(), isDir);
  }
  
  wchar_t temp[16];
  UString s2;
  const wchar_t *s = NULL;
  
  if (indexType == NArchive::NEventIndexType::kInArcIndex)
  {
    if (index != (UInt32)(Int32)-1)
    {
      if (ArcItems)
      {
        const CArcItem &ai = (*ArcItems)[index];
        s = ai.Name;
        isDir = ai.IsDir;
      }
      else if (Arc)
      {
        RINOK(Arc->GetItem_Path(index, s2))
        s = s2;
        RINOK(Archive_IsItem_Dir(Arc->Archive, index, isDir))
      }
    }
  }
  else if (indexType == NArchive::NEventIndexType::kBlockIndex)
  {
    temp[0] = '#';
    ConvertUInt32ToString(index, temp + 1);
    s = temp;
  }

  if (!s)
    s = L"";

  return Callback->ReportUpdateOperation(op, s, isDir);

  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::ReportExtractResult(UInt32 indexType, UInt32 index, Int32 opRes))
{
  COM_TRY_BEGIN

  bool isEncrypted = false;
  wchar_t temp[16];
  UString s2;
  const wchar_t *s = NULL;
  
  if (indexType == NArchive::NEventIndexType::kOutArcIndex)
  {
    /*
    UString name;
    if (index != (UInt32)(Int32)-1)
    {
      const CUpdatePair2 &up = (*UpdatePairs)[index];
      if (up.ExistOnDisk())
      {
        s2 = DirItems->GetLogPath(up.DirIndex);
        s = s2;
      }
    }
    */
    return E_FAIL;
  }
 
  if (indexType == NArchive::NEventIndexType::kInArcIndex)
  {
    if (index != (UInt32)(Int32)-1)
    {
      if (ArcItems)
        s = (*ArcItems)[index].Name;
      else if (Arc)
      {
        RINOK(Arc->GetItem_Path(index, s2))
        s = s2;
      }
      if (Archive)
      {
        RINOK(Archive_GetItemBoolProp(Archive, index, kpidEncrypted, isEncrypted))
      }
    }
  }
  else if (indexType == NArchive::NEventIndexType::kBlockIndex)
  {
    temp[0] = '#';
    ConvertUInt32ToString(index, temp + 1);
    s = temp;
  }

  return Callback->ReportExtractResult(opRes, BoolToInt(isEncrypted), s);

  COM_TRY_END
}


/*
Z7_COM7F_IMF(CArchiveUpdateCallback::DoNeedArcProp(PROPID propID, Int32 *answer))
{
  *answer = 0;
  if (Need_ArcMTime_Report && propID == kpidComboMTime)
    *answer = 1;
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::ReportProp(UInt32 indexType, UInt32 index, PROPID propID, const PROPVARIANT *value))
{
  if (indexType == NArchive::NEventIndexType::kArcProp)
  {
    if (propID == kpidComboMTime)
    {
      ArcMTime_WasReported = true;
      if (value->vt == VT_FILETIME)
      {
        Reported_ArcMTime.Set_From_Prop(*value);
        Reported_ArcMTime.Def = true;
      }
      else
      {
        Reported_ArcMTime.Clear();
        if (value->vt != VT_EMPTY)
          return E_FAIL; // for debug
      }
    }
  }
  return Callback->ReportProp(indexType, index, propID, value);
}

Z7_COM7F_IMF(CArchiveUpdateCallback::ReportRawProp(UInt32 indexType, UInt32 index,
    PROPID propID, const void *data, UInt32 dataSize, UInt32 propType))
{
  return Callback->ReportRawProp(indexType, index, propID, data, dataSize, propType);
}

Z7_COM7F_IMF(CArchiveUpdateCallback::ReportFinished(UInt32 indexType, UInt32 index, Int32 opRes))
{
  return Callback->ReportFinished(indexType, index, opRes);
}
*/

Z7_COM7F_IMF(CArchiveUpdateCallback::GetVolumeSize(UInt32 index, UInt64 *size))
{
  if (VolumesSizes.Size() == 0)
    return S_FALSE;
  if (index >= (UInt32)VolumesSizes.Size())
    index = VolumesSizes.Size() - 1;
  *size = VolumesSizes[index];
  return S_OK;
}

Z7_COM7F_IMF(CArchiveUpdateCallback::GetVolumeStream(UInt32 index, ISequentialOutStream **volumeStream))
{
  COM_TRY_BEGIN
  char temp[16];
  ConvertUInt32ToString(index + 1, temp);
  FString res (temp);
  while (res.Len() < 2)
    res.InsertAtFront(FTEXT('0'));
  FString fileName = VolName;
  fileName.Add_Dot();
  fileName += res;
  fileName += VolExt;
  COutFileStream *streamSpec = new COutFileStream;
  CMyComPtr<ISequentialOutStream> streamLoc(streamSpec);
  if (!streamSpec->Create_NEW(fileName))
    return GetLastError_noZero_HRESULT();
  *volumeStream = streamLoc.Detach();
  return S_OK;
  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::CryptoGetTextPassword2(Int32 *passwordIsDefined, BSTR *password))
{
  COM_TRY_BEGIN
  return Callback->CryptoGetTextPassword2(passwordIsDefined, password);
  COM_TRY_END
}

Z7_COM7F_IMF(CArchiveUpdateCallback::CryptoGetTextPassword(BSTR *password))
{
  COM_TRY_BEGIN
  return Callback->CryptoGetTextPassword(password);
  COM_TRY_END
}

HRESULT CArchiveUpdateCallback::InFileStream_On_Error(UINT_PTR val, DWORD error)
{
  #ifdef _WIN32 // FIX IT !!!
  // why did we check only for ERROR_LOCK_VIOLATION ?
  // if (error == ERROR_LOCK_VIOLATION)
  #endif
  {
    MT_LOCK
    const UInt32 index = (UInt32)val;
    FOR_VECTOR(i, _openFiles_Indexes)
    {
      if (_openFiles_Indexes[i] == index)
      {
        RINOK(Callback->ReadingFileError(_openFiles_Paths[i], error))
        break;
      }
    }
  }
  return HRESULT_FROM_WIN32(error);
}

void CArchiveUpdateCallback::InFileStream_On_Destroy(CInFileStream *stream, UINT_PTR val)
{
  MT_LOCK
  if (Need_LatestMTime)
  {
    if (stream->_info_WasLoaded)
    {
      const CFiTime &ft = ST_MTIME(stream->_info);
      if (!LatestMTime_Defined
          || Compare_FiTime(&LatestMTime, &ft) < 0)
        LatestMTime = ft;
      LatestMTime_Defined = true;
    }
  }
  const UInt32 index = (UInt32)val;
  FOR_VECTOR(i, _openFiles_Indexes)
  {
    if (_openFiles_Indexes[i] == index)
    {
      _openFiles_Indexes.Delete(i);
      _openFiles_Paths.Delete(i);
      // _openFiles_Streams.Delete(i);
      return;
    }
  }
  /* 21.02 : this function can be called in destructor.
     And destructor can be called after some exception.
     If we don't want to throw exception in desctructors or after another exceptions,
     we must disable the code below that raises new exception.
  */
  // throw 20141125;
}
