// Update.cpp

#include "StdAfx.h"

// #include  <stdio.h>

#include "Update.h"

#include "../../../Common/StringConvert.h"

#include "../../../Windows/DLL.h"
#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileFind.h"
#include "../../../Windows/FileName.h"
#include "../../../Windows/PropVariant.h"
#include "../../../Windows/PropVariantConv.h"
#include "../../../Windows/TimeUtils.h"

#include "../../Common/FileStreams.h"
#include "../../Common/LimitedStreams.h"
#include "../../Common/MultiOutStream.h"
#include "../../Common/StreamUtils.h"

#include "../../Compress/CopyCoder.h"

#include "../Common/DirItem.h"
#include "../Common/EnumDirItems.h"
#include "../Common/OpenArchive.h"
#include "../Common/UpdateProduce.h"

#include "EnumDirItems.h"
#include "SetProperties.h"
#include "TempFiles.h"
#include "UpdateCallback.h"

static const char * const kUpdateIsNotSupoorted =
  "update operations are not supported for this archive";

static const char * const kUpdateIsNotSupported_MultiVol =
  "Updating for multivolume archives is not implemented";

using namespace NWindows;
using namespace NCOM;
using namespace NFile;
using namespace NDir;
using namespace NName;

#ifdef _WIN32
static CFSTR const kTempFolderPrefix = FTEXT("7zE");
#endif

void CUpdateErrorInfo::SetFromLastError(const char *message)
{
  SystemError = ::GetLastError();
  Message = message;
}

HRESULT CUpdateErrorInfo::SetFromLastError(const char *message, const FString &fileName)
{
  SetFromLastError(message);
  FileNames.Add(fileName);
  return Get_HRESULT_Error();
}

HRESULT CUpdateErrorInfo::SetFromError_DWORD(const char *message, const FString &fileName, DWORD error)
{
  Message = message;
  FileNames.Add(fileName);
  SystemError = error;
  return Get_HRESULT_Error();
}


using namespace NUpdateArchive;

struct CMultiOutStream_Rec
{
  CMultiOutStream *Spec;
  CMyComPtr<IOutStream> Ref;
};

struct CMultiOutStream_Bunch
{
  CObjectVector<CMultiOutStream_Rec> Items;
  
  HRESULT Destruct()
  {
    HRESULT hres = S_OK;
    FOR_VECTOR (i, Items)
    {
      CMultiOutStream_Rec &rec = Items[i];
      if (rec.Ref)
      {
        const HRESULT hres2 = rec.Spec->Destruct();
        if (hres == S_OK)
          hres = hres2;
      }
    }
    Items.Clear();
    return hres;
  }

  void DisableDeletion()
  {
    FOR_VECTOR (i, Items)
    {
      CMultiOutStream_Rec &rec = Items[i];
      if (rec.Ref)
        rec.Spec->NeedDelete = false;
    }
  }
};


void CArchivePath::ParseFromPath(const UString &path, EArcNameMode mode)
{
  OriginalPath = path;
  
  SplitPathToParts_2(path, Prefix, Name);
  
  if (mode == k_ArcNameMode_Add)
    return;
  
  if (mode != k_ArcNameMode_Exact)
  {
    int dotPos = Name.ReverseFind_Dot();
    if (dotPos < 0)
      return;
    if ((unsigned)dotPos == Name.Len() - 1)
      Name.DeleteBack();
    else
    {
      const UString ext = Name.Ptr((unsigned)(dotPos + 1));
      if (BaseExtension.IsEqualTo_NoCase(ext))
      {
        BaseExtension = ext;
        Name.DeleteFrom((unsigned)dotPos);
        return;
      }
    }
  }

  BaseExtension.Empty();
}

UString CArchivePath::GetFinalPath() const
{
  UString path = GetPathWithoutExt();
  if (!BaseExtension.IsEmpty())
  {
    path.Add_Dot();
    path += BaseExtension;
  }
  return path;
}

UString CArchivePath::GetFinalVolPath() const
{
  UString path = GetPathWithoutExt();
  // if BaseExtension is empty, we must ignore VolExtension also.
  if (!BaseExtension.IsEmpty())
  {
    path.Add_Dot();
    path += VolExtension;
  }
  return path;
}

FString CArchivePath::GetTempPath() const
{
  FString path = TempPrefix;
  path += us2fs(Name);
  if (!BaseExtension.IsEmpty())
  {
    path.Add_Dot();
    path += us2fs(BaseExtension);
  }
  path += ".tmp";
  path += TempPostfix;
  return path;
}

static const char * const kDefaultArcType = "7z";
static const char * const kDefaultArcExt = "7z";
static const char * const kSFXExtension =
  #ifdef _WIN32
    "exe";
  #else
    "";
  #endif

bool CUpdateOptions::InitFormatIndex(const CCodecs *codecs,
    const CObjectVector<COpenType> &types, const UString &arcPath)
{
  if (types.Size() > 1)
    return false;
  // int arcTypeIndex = -1;
  if (types.Size() != 0)
  {
    MethodMode.Type = types[0];
    MethodMode.Type_Defined = true;
  }
  if (MethodMode.Type.FormatIndex < 0)
  {
    // MethodMode.Type = -1;
    MethodMode.Type = COpenType();
    if (ArcNameMode != k_ArcNameMode_Add)
    {
      MethodMode.Type.FormatIndex = codecs->FindFormatForArchiveName(arcPath);
      if (MethodMode.Type.FormatIndex >= 0)
        MethodMode.Type_Defined = true;
    }
  }
  return true;
}

bool CUpdateOptions::SetArcPath(const CCodecs *codecs, const UString &arcPath)
{
  UString typeExt;
  int formatIndex = MethodMode.Type.FormatIndex;
  if (formatIndex < 0)
  {
    typeExt = kDefaultArcExt;
  }
  else
  {
    const CArcInfoEx &arcInfo = codecs->Formats[(unsigned)formatIndex];
    if (!arcInfo.UpdateEnabled)
      return false;
    typeExt = arcInfo.GetMainExt();
  }
  UString ext = typeExt;
  if (SfxMode)
    ext = kSFXExtension;
  ArchivePath.BaseExtension = ext;
  ArchivePath.VolExtension = typeExt;
  ArchivePath.ParseFromPath(arcPath, ArcNameMode);
  FOR_VECTOR (i, Commands)
  {
    CUpdateArchiveCommand &uc = Commands[i];
    uc.ArchivePath.BaseExtension = ext;
    uc.ArchivePath.VolExtension = typeExt;
    uc.ArchivePath.ParseFromPath(uc.UserArchivePath, ArcNameMode);
  }
  return true;
}


struct CUpdateProduceCallbackImp Z7_final: public IUpdateProduceCallback
{
  const CObjectVector<CArcItem> *_arcItems;
  CDirItemsStat *_stat;
  IUpdateCallbackUI *_callback;
  
  CUpdateProduceCallbackImp(
      const CObjectVector<CArcItem> *a,
      CDirItemsStat *stat,
      IUpdateCallbackUI *callback):
    _arcItems(a),
    _stat(stat),
    _callback(callback) {}
  
  virtual HRESULT ShowDeleteFile(unsigned arcIndex) Z7_override;
};


HRESULT CUpdateProduceCallbackImp::ShowDeleteFile(unsigned arcIndex)
{
  const CArcItem &ai = (*_arcItems)[arcIndex];
  {
    CDirItemsStat &stat = *_stat;
    if (ai.IsDir)
      stat.NumDirs++;
    else if (ai.IsAltStream)
    {
      stat.NumAltStreams++;
      stat.AltStreamsSize += ai.Size;
    }
    else
    {
      stat.NumFiles++;
      stat.FilesSize += ai.Size;
    }
  }
  return _callback->ShowDeleteFile(ai.Name, ai.IsDir);
}

bool CRenamePair::Prepare()
{
  if (RecursedType != NRecursedType::kNonRecursed)
    return false;
  if (!WildcardParsing)
    return true;
  return !DoesNameContainWildcard(OldName);
}

extern bool g_CaseSensitive;

static unsigned CompareTwoNames(const wchar_t *s1, const wchar_t *s2)
{
  for (unsigned i = 0;; i++)
  {
    wchar_t c1 = s1[i];
    wchar_t c2 = s2[i];
    if (c1 == 0 || c2 == 0)
      return i;
    if (c1 == c2)
      continue;
    if (!g_CaseSensitive && (MyCharUpper(c1) == MyCharUpper(c2)))
      continue;
    if (IsPathSepar(c1) && IsPathSepar(c2))
      continue;
    return i;
  }
}

bool CRenamePair::GetNewPath(bool isFolder, const UString &src, UString &dest) const
{
  unsigned num = CompareTwoNames(OldName, src);
  if (OldName[num] == 0)
  {
    if (src[num] != 0 && !IsPathSepar(src[num]) && num != 0 && !IsPathSepar(src[num - 1]))
      return false;
  }
  else
  {
    // OldName[num] != 0
    // OldName = "1\1a.txt"
    // src = "1"

    if (!isFolder
        || src[num] != 0
        || !IsPathSepar(OldName[num])
        || OldName[num + 1] != 0)
      return false;
  }
  dest = NewName + src.Ptr(num);
  return true;
}

#ifdef SUPPORT_ALT_STREAMS
int FindAltStreamColon_in_Path(const wchar_t *path);
#endif



static HRESULT Compress(
    const CUpdateOptions &options,
    bool isUpdatingItself,
    CCodecs *codecs,
    const CActionSet &actionSet,
    const CArc *arc,
    CArchivePath &archivePath,
    const CObjectVector<CArcItem> &arcItems,
    Byte *processedItemsStatuses,
    const CDirItems &dirItems,
    const CDirItem *parentDirItem,
    CTempFiles &tempFiles,
    CMultiOutStream_Bunch &multiStreams,
    CUpdateErrorInfo &errorInfo,
    IUpdateCallbackUI *callback,
    CFinishArchiveStat &st)
{
  CMyComPtr<IOutArchive> outArchive;
  int formatIndex = options.MethodMode.Type.FormatIndex;
  
  if (arc)
  {
    formatIndex = arc->FormatIndex;
    if (formatIndex < 0)
      return E_NOTIMPL;
    CMyComPtr<IInArchive> archive2 = arc->Archive;
    HRESULT result = archive2.QueryInterface(IID_IOutArchive, &outArchive);
    if (result != S_OK)
      throw kUpdateIsNotSupoorted;
  }
  else
  {
    RINOK(codecs->CreateOutArchive((unsigned)formatIndex, outArchive))

    #ifdef Z7_EXTERNAL_CODECS
    {
      CMyComPtr<ISetCompressCodecsInfo> setCompressCodecsInfo;
      outArchive.QueryInterface(IID_ISetCompressCodecsInfo, (void **)&setCompressCodecsInfo);
      if (setCompressCodecsInfo)
      {
        RINOK(setCompressCodecsInfo->SetCompressCodecsInfo(codecs))
      }
    }
    #endif
  }
  
  if (!outArchive)
    throw kUpdateIsNotSupoorted;

  // we need to set properties to get fileTimeType.
  RINOK(SetProperties(outArchive, options.MethodMode.Properties))

  NFileTimeType::EEnum fileTimeType;
  {
    /*
    how we compare file_in_archive::MTime with dirItem.MTime
    for GetUpdatePairInfoList():
    
    if (kpidMTime is not defined), external MTime of archive is used.
    
    before 22.00:
      if (kpidTimeType is defined)
      {
        kpidTimeType is used as precision.
        (kpidTimeType > kDOS) is not allowed.
      }
      else GetFileTimeType() value is used as precision.

    22.00:
      if (kpidMTime is defined)
      {
        if (kpidMTime::precision != 0), then kpidMTime::precision is used as precision.
        else
        {
          if (kpidTimeType is defined), kpidTimeType is used as precision.
          else GetFileTimeType() value is used as precision.
        }
      }
      else external MTime of archive is used as precision.
    */

    UInt32 value;
    RINOK(outArchive->GetFileTimeType(&value))

    // we support any future fileType here.
    fileTimeType = (NFileTimeType::EEnum)value;
    
    /*
    old 21.07 code:
    switch (value)
    {
      case NFileTimeType::kWindows:
      case NFileTimeType::kUnix:
      case NFileTimeType::kDOS:
        fileTimeType = (NFileTimeType::EEnum)value;
        break;
      default:
        return E_FAIL;
    }
    */
  }

  // bool noTimestampExpected = false;
  {
    const CArcInfoEx &arcInfo = codecs->Formats[(unsigned)formatIndex];

    // if (arcInfo.Flags_KeepName()) noTimestampExpected = true;
    if (arcInfo.Is_Xz() ||
        arcInfo.Is_BZip2())
    {
      /* 7-zip before 22.00 returns NFileTimeType::kUnix for xz and bzip2,
         but we want to set timestamp without reduction to unix. */
      // noTimestampExpected = true;
      fileTimeType = NFileTimeType::kNotDefined; // it means not defined
    }

    if (options.AltStreams.Val && !arcInfo.Flags_AltStreams())
      return E_NOTIMPL;
    if (options.NtSecurity.Val && !arcInfo.Flags_NtSecurity())
      return E_NOTIMPL;
    if (options.DeleteAfterCompressing && arcInfo.Flags_HashHandler())
      return E_NOTIMPL;
  }

  CRecordVector<CUpdatePair2> updatePairs2;

  UStringVector newNames;

  CArcToDoStat stat2;

  if (options.RenameMode || options.RenamePairs.Size() != 0)
  {
    FOR_VECTOR (i, arcItems)
    {
      const CArcItem &ai = arcItems[i];
      bool needRename = false;
      UString dest;
      
      if (ai.Censored)
      {
        FOR_VECTOR (j, options.RenamePairs)
        {
          const CRenamePair &rp = options.RenamePairs[j];
          if (rp.GetNewPath(ai.IsDir, ai.Name, dest))
          {
            needRename = true;
            break;
          }
          
          #ifdef SUPPORT_ALT_STREAMS
          if (ai.IsAltStream)
          {
            int colonPos = FindAltStreamColon_in_Path(ai.Name);
            if (colonPos >= 0)
            {
              UString mainName = ai.Name.Left((unsigned)colonPos);
              /*
              actually we must improve that code to support cases
              with folder renaming like: rn arc dir1\ dir2\
              */
              if (rp.GetNewPath(false, mainName, dest))
              {
                needRename = true;
                dest.Add_Colon();
                dest += ai.Name.Ptr((unsigned)(colonPos + 1));
                break;
              }
            }
          }
          #endif
        }
      }
      
      CUpdatePair2 up2;
      up2.SetAs_NoChangeArcItem(ai.IndexInServer);
      if (needRename)
      {
        up2.NewProps = true;
        RINOK(arc->IsItem_Anti(i, up2.IsAnti))
        up2.NewNameIndex = (int)newNames.Add(dest);
      }
      updatePairs2.Add(up2);
    }
  }
  else
  {
    CRecordVector<CUpdatePair> updatePairs;
    GetUpdatePairInfoList(dirItems, arcItems, fileTimeType, updatePairs); // must be done only once!!!
    CUpdateProduceCallbackImp upCallback(&arcItems, &stat2.DeleteData, callback);
    
    UpdateProduce(updatePairs, actionSet, updatePairs2, isUpdatingItself ? &upCallback : NULL);
  }

  {
    FOR_VECTOR (i, updatePairs2)
    {
      const CUpdatePair2 &up = updatePairs2[i];

      // 17.01: anti-item is (up.NewData && (p.UseArcProps in most cases))

      if (up.NewData && !up.UseArcProps)
      {
        if (up.ExistOnDisk())
        {
          CDirItemsStat2 &stat = stat2.NewData;
          const CDirItem &di = dirItems.Items[(unsigned)up.DirIndex];
          if (di.IsDir())
          {
            if (up.IsAnti)
              stat.Anti_NumDirs++;
            else
              stat.NumDirs++;
          }
         #ifdef _WIN32
          else if (di.IsAltStream)
          {
            if (up.IsAnti)
              stat.Anti_NumAltStreams++;
            else
            {
              stat.NumAltStreams++;
              stat.AltStreamsSize += di.Size;
            }
          }
         #endif
          else
          {
            if (up.IsAnti)
              stat.Anti_NumFiles++;
            else
            {
              stat.NumFiles++;
              stat.FilesSize += di.Size;
            }
          }
        }
      }
      else if (up.ArcIndex >= 0)
      {
        CDirItemsStat2 &stat = *(up.NewData ? &stat2.NewData : &stat2.OldData);
        const CArcItem &ai = arcItems[(unsigned)up.ArcIndex];
        if (ai.IsDir)
        {
          if (up.IsAnti)
            stat.Anti_NumDirs++;
          else
            stat.NumDirs++;
        }
        else if (ai.IsAltStream)
        {
          if (up.IsAnti)
            stat.Anti_NumAltStreams++;
          else
          {
            stat.NumAltStreams++;
            stat.AltStreamsSize += ai.Size;
          }
        }
        else
        {
          if (up.IsAnti)
            stat.Anti_NumFiles++;
          else
          {
            stat.NumFiles++;
            stat.FilesSize += ai.Size;
          }
        }
      }
    }
    RINOK(callback->SetNumItems(stat2))
  }
  
  CArchiveUpdateCallback *updateCallbackSpec = new CArchiveUpdateCallback;
  CMyComPtr<IArchiveUpdateCallback> updateCallback(updateCallbackSpec);
  
  updateCallbackSpec->PreserveATime = options.PreserveATime;
  updateCallbackSpec->ShareForWrite = options.OpenShareForWrite;
  updateCallbackSpec->StopAfterOpenError = options.StopAfterOpenError;
  updateCallbackSpec->StdInMode = options.StdInMode;
  updateCallbackSpec->Callback = callback;

  if (arc)
  {
    // we set Archive to allow to transfer GetProperty requests back to DLL.
    updateCallbackSpec->Archive = arc->Archive;
  }
  
  updateCallbackSpec->DirItems = &dirItems;
  updateCallbackSpec->ParentDirItem = parentDirItem;

  updateCallbackSpec->StoreNtSecurity = options.NtSecurity.Val;
  updateCallbackSpec->StoreHardLinks = options.HardLinks.Val;
  updateCallbackSpec->StoreSymLinks = options.SymLinks.Val;
  updateCallbackSpec->StoreOwnerName = options.StoreOwnerName.Val;
  updateCallbackSpec->StoreOwnerId = options.StoreOwnerId.Val;

  updateCallbackSpec->Arc = arc;
  updateCallbackSpec->ArcItems = &arcItems;
  updateCallbackSpec->UpdatePairs = &updatePairs2;

  updateCallbackSpec->ProcessedItemsStatuses = processedItemsStatuses;

  {
    const UString arcPath = archivePath.GetFinalPath();
    updateCallbackSpec->ArcFileName = ExtractFileNameFromPath(arcPath);
  }

  if (options.RenamePairs.Size() != 0)
    updateCallbackSpec->NewNames = &newNames;

  if (options.SetArcMTime)
  {
    // updateCallbackSpec->Need_ArcMTime_Report = true;
    updateCallbackSpec->Need_LatestMTime = true;
  }

  CMyComPtr<IOutStream> outSeekStream;
  CMyComPtr<ISequentialOutStream> outStream;

  if (!options.StdOutMode)
  {
    FString dirPrefix;
    if (!GetOnlyDirPrefix(us2fs(archivePath.GetFinalPath()), dirPrefix))
      throw 1417161;
    CreateComplexDir(dirPrefix);
  }

  COutFileStream *outStreamSpec = NULL;
  CStdOutFileStream *stdOutFileStreamSpec = NULL;
  CMultiOutStream *volStreamSpec = NULL;

  if (options.VolumesSizes.Size() == 0)
  {
    if (options.StdOutMode)
    {
      stdOutFileStreamSpec = new CStdOutFileStream;
      outStream = stdOutFileStreamSpec;
    }
    else
    {
      outStreamSpec = new COutFileStream;
      outSeekStream = outStreamSpec;
      outStream = outSeekStream;
      bool isOK = false;
      FString realPath;
      
      for (unsigned i = 0; i < (1 << 16); i++)
      {
        if (archivePath.Temp)
        {
          if (i > 0)
          {
            archivePath.TempPostfix.Empty();
            archivePath.TempPostfix.Add_UInt32(i);
          }
          realPath = archivePath.GetTempPath();
        }
        else
          realPath = us2fs(archivePath.GetFinalPath());
        if (outStreamSpec->Create_NEW(realPath))
        {
          tempFiles.Paths.Add(realPath);
          isOK = true;
          break;
        }
        if (::GetLastError() != ERROR_FILE_EXISTS)
          break;
        if (!archivePath.Temp)
          break;
      }
      
      if (!isOK)
        return errorInfo.SetFromLastError("cannot open file", realPath);
    }
  }
  else
  {
    if (options.StdOutMode)
      return E_FAIL;
    if (arc && arc->GetGlobalOffset() > 0)
      return E_NOTIMPL;
      
    volStreamSpec = new CMultiOutStream();
    outSeekStream = volStreamSpec;
    outStream = outSeekStream;
    volStreamSpec->Prefix = us2fs(archivePath.GetFinalVolPath());
    volStreamSpec->Prefix.Add_Dot();
    volStreamSpec->Init(options.VolumesSizes);
    {
      CMultiOutStream_Rec &rec = multiStreams.Items.AddNew();
      rec.Spec = volStreamSpec;
      rec.Ref = rec.Spec;
    }

    /*
    updateCallbackSpec->VolumesSizes = volumesSizes;
    updateCallbackSpec->VolName = archivePath.Prefix + archivePath.Name;
    if (!archivePath.VolExtension.IsEmpty())
      updateCallbackSpec->VolExt = UString('.') + archivePath.VolExtension;
    */
  }

  if (options.SfxMode)
  {
    CInFileStream *sfxStreamSpec = new CInFileStream;
    CMyComPtr<IInStream> sfxStream(sfxStreamSpec);
    if (!sfxStreamSpec->Open(options.SfxModule))
      return errorInfo.SetFromLastError("cannot open SFX module", options.SfxModule);

    CMyComPtr<ISequentialOutStream> sfxOutStream;
    COutFileStream *outStreamSpec2 = NULL;
    if (options.VolumesSizes.Size() == 0)
      sfxOutStream = outStream;
    else
    {
      outStreamSpec2 = new COutFileStream;
      sfxOutStream = outStreamSpec2;
      const FString realPath = us2fs(archivePath.GetFinalPath());
      if (!outStreamSpec2->Create_NEW(realPath))
        return errorInfo.SetFromLastError("cannot open file", realPath);
    }

    {
      UInt64 sfxSize;
      RINOK(sfxStreamSpec->GetSize(&sfxSize))
      RINOK(callback->WriteSfx(fs2us(options.SfxModule), sfxSize))
    }

    RINOK(NCompress::CopyStream(sfxStream, sfxOutStream, NULL))
    
    if (outStreamSpec2)
    {
      RINOK(outStreamSpec2->Close())
    }
  }

  CMyComPtr<ISequentialOutStream> tailStream;

  if (options.SfxMode || !arc || arc->ArcStreamOffset == 0)
    tailStream = outStream;
  else
  {
    // Int64 globalOffset = arc->GetGlobalOffset();
    RINOK(InStream_SeekToBegin(arc->InStream))
    RINOK(NCompress::CopyStream_ExactSize(arc->InStream, outStream, arc->ArcStreamOffset, NULL))
    if (options.StdOutMode)
      tailStream = outStream;
    else
    {
      CTailOutStream *tailStreamSpec = new CTailOutStream;
      tailStream = tailStreamSpec;
      tailStreamSpec->Stream = outSeekStream;
      tailStreamSpec->Offset = arc->ArcStreamOffset;
      tailStreamSpec->Init();
    }
  }

  CFiTime ft;
  FiTime_Clear(ft);
  bool ft_Defined = false;
  {
    FOR_VECTOR (i, updatePairs2)
    {
      const CUpdatePair2 &pair2 = updatePairs2[i];
      CFiTime ft2;
      FiTime_Clear(ft2);
      bool ft2_Defined = false;
      /* we use full precision of dirItem, if dirItem is defined
         and (dirItem will be used or dirItem is sameTime in dir and arc */
      if (pair2.DirIndex >= 0 &&
          (pair2.NewProps || pair2.IsSameTime))
      {
        ft2 = dirItems.Items[(unsigned)pair2.DirIndex].MTime;
        ft2_Defined = true;
      }
      else if (pair2.UseArcProps && pair2.ArcIndex >= 0)
      {
        const CArcItem &arcItem = arcItems[(unsigned)pair2.ArcIndex];
        if (arcItem.MTime.Def)
        {
          arcItem.MTime.Write_To_FiTime(ft2);
          ft2_Defined = true;
        }
      }
      if (ft2_Defined)
      {
        if (!ft_Defined || Compare_FiTime(&ft, &ft2) < 0)
        {
          ft = ft2;
          ft_Defined = true;
        }
      }
    }
    /*
    if (fileTimeType != NFileTimeType::kNotDefined)
    FiTime_Normalize_With_Prec(ft, fileTimeType);
    */
  }

  if (volStreamSpec && options.SetArcMTime && ft_Defined)
  {
    volStreamSpec->MTime = ft;
    volStreamSpec->MTime_Defined = true;
  }

  HRESULT result = outArchive->UpdateItems(tailStream, updatePairs2.Size(), updateCallback);
  // callback->Finalize();
  RINOK(result)

  if (!updateCallbackSpec->AreAllFilesClosed())
  {
    errorInfo.Message = "There are unclosed input files:";
    errorInfo.FileNames = updateCallbackSpec->_openFiles_Paths;
    return E_FAIL;
  }

  if (options.SetArcMTime)
  {
    // bool needNormalizeAfterStream;
    // needParse;
    /*
    if (updateCallbackSpec->ArcMTime_WasReported)
    {
      isDefined = updateCallbackSpec->Reported_ArcMTime.Def;
      if (isDefined)
        updateCallbackSpec->Reported_ArcMTime.Write_To_FiTime(ft);
      else
        fileTimeType = NFileTimeType::kNotDefined;
    }
    if (!isDefined)
    */
    {
      if (updateCallbackSpec->LatestMTime_Defined)
      {
        // CArcTime at = StreamCallback_ArcMTime;
        // updateCallbackSpec->StreamCallback_ArcMTime.Write_To_FiTime(ft);
        // we must normalize with precision from archive;
        if (!ft_Defined || Compare_FiTime(&ft, &updateCallbackSpec->LatestMTime) < 0)
          ft = updateCallbackSpec->LatestMTime;
        ft_Defined = true;
      }
      /*
      if (fileTimeType != NFileTimeType::kNotDefined)
        FiTime_Normalize_With_Prec(ft, fileTimeType);
      */
    }
    // if (ft.dwLowDateTime != 0 || ft.dwHighDateTime != 0)
    if (ft_Defined)
    {
      // we ignore set time errors here.
      // note that user could move some finished volumes to another folder.
      if (outStreamSpec)
        outStreamSpec->SetMTime(&ft);
      else if (volStreamSpec)
        volStreamSpec->SetMTime_Final(ft);
    }
  }

  if (callback)
  {
    UInt64 size = 0;
    if (outStreamSpec)
      outStreamSpec->GetSize(&size);
    else if (stdOutFileStreamSpec)
      size = stdOutFileStreamSpec->GetSize();
    else
      size = volStreamSpec->GetSize();

    st.OutArcFileSize = size;
  }

  if (outStreamSpec)
    result = outStreamSpec->Close();
  else if (volStreamSpec)
  {
    result = volStreamSpec->FinalFlush_and_CloseFiles(st.NumVolumes);
    st.IsMultiVolMode = true;
  }

  RINOK(result)
  
  if (processedItemsStatuses)
  {
    FOR_VECTOR (i, updatePairs2)
    {
      const CUpdatePair2 &up = updatePairs2[i];
      if (up.NewData && up.DirIndex >= 0)
      {
        const CDirItem &di = dirItems.Items[(unsigned)up.DirIndex];
        if (di.AreReparseData() || (!di.IsDir() && di.Size == 0))
          processedItemsStatuses[(unsigned)up.DirIndex] = 1;
      }
    }
  }

  return result;
}



static bool Censor_AreAllAllowed(const NWildcard::CCensor &censor)
{
  if (censor.Pairs.Size() != 1)
    return false;
  const NWildcard::CPair &pair = censor.Pairs[0];
  /* Censor_CheckPath() ignores (CPair::Prefix).
     So we also ignore (CPair::Prefix) here */
  // if (!pair.Prefix.IsEmpty()) return false;
  return pair.Head.AreAllAllowed();
}

bool CensorNode_CheckPath2(const NWildcard::CCensorNode &node, const CReadArcItem &item, bool &include);

static bool Censor_CheckPath(const NWildcard::CCensor &censor, const CReadArcItem &item)
{
  bool finded = false;
  FOR_VECTOR (i, censor.Pairs)
  {
    /* (CPair::Prefix) in not used for matching items in archive.
       So we ignore (CPair::Prefix) here */
    bool include;
    if (CensorNode_CheckPath2(censor.Pairs[i].Head, item, include))
    {
      // Check it and FIXME !!!!
      // here we can exclude item via some Pair, that is still allowed by another Pair
      if (!include)
        return false;
      finded = true;
    }
  }
  return finded;
}

static HRESULT EnumerateInArchiveItems(
    // bool storeStreamsMode,
    const NWildcard::CCensor &censor,
    const CArc &arc,
    CObjectVector<CArcItem> &arcItems)
{
  arcItems.Clear();
  UInt32 numItems;
  IInArchive *archive = arc.Archive;
  RINOK(archive->GetNumberOfItems(&numItems))
  arcItems.ClearAndReserve(numItems);

  CReadArcItem item;

  const bool allFilesAreAllowed = Censor_AreAllAllowed(censor);

  for (UInt32 i = 0; i < numItems; i++)
  {
    CArcItem ai;

    RINOK(arc.GetItem(i, item))
    ai.Name = item.Path;
    ai.IsDir = item.IsDir;
    ai.IsAltStream =
        #ifdef SUPPORT_ALT_STREAMS
          item.IsAltStream;
        #else
          false;
        #endif

    /*
    if (!storeStreamsMode && ai.IsAltStream)
      continue;
    */
    if (allFilesAreAllowed)
      ai.Censored = true;
    else
      ai.Censored = Censor_CheckPath(censor, item);

    // ai.MTime will be set to archive MTime, if not present in archive item
    RINOK(arc.GetItem_MTime(i, ai.MTime))
    RINOK(arc.GetItem_Size(i, ai.Size, ai.Size_Defined))

    ai.IndexInServer = i;
    arcItems.AddInReserved(ai);
  }
  return S_OK;
}

#if defined(_WIN32) && !defined(UNDER_CE)

#if defined(__MINGW32__) || defined(__MINGW64__)
#include <mapi.h>
#else
#include <MAPI.h>
#endif

extern "C" {

#ifdef MAPI_FORCE_UNICODE

#define Z7_WIN_LPMAPISENDMAILW  LPMAPISENDMAILW
#define Z7_WIN_MapiFileDescW    MapiFileDescW
#define Z7_WIN_MapiMessageW     MapiMessageW
#define Z7_WIN_MapiRecipDescW   MapiRecipDescW

#else

typedef struct
{
    ULONG ulReserved;
    ULONG ulRecipClass;
    PWSTR lpszName;
    PWSTR lpszAddress;
    ULONG ulEIDSize;
    PVOID lpEntryID;
} Z7_WIN_MapiRecipDescW, *Z7_WIN_lpMapiRecipDescW;

typedef struct
{
    ULONG ulReserved;
    ULONG flFlags;
    ULONG nPosition;
    PWSTR lpszPathName;
    PWSTR lpszFileName;
    PVOID lpFileType;
} Z7_WIN_MapiFileDescW, *Z7_WIN_lpMapiFileDescW;

typedef struct
{
  ULONG ulReserved;
  PWSTR lpszSubject;
  PWSTR lpszNoteText;
  PWSTR lpszMessageType;
  PWSTR lpszDateReceived;
  PWSTR lpszConversationID;
  FLAGS flFlags;
  Z7_WIN_lpMapiRecipDescW lpOriginator;
  ULONG nRecipCount;
  Z7_WIN_lpMapiRecipDescW lpRecips;
  ULONG nFileCount;
  Z7_WIN_lpMapiFileDescW lpFiles;
} Z7_WIN_MapiMessageW, *Z7_WIN_lpMapiMessageW;

typedef ULONG (FAR PASCAL Z7_WIN_MAPISENDMAILW)(
  LHANDLE lhSession,
  ULONG_PTR ulUIParam,
  Z7_WIN_lpMapiMessageW lpMessage,
  FLAGS flFlags,
  ULONG ulReserved
);
typedef Z7_WIN_MAPISENDMAILW FAR *Z7_WIN_LPMAPISENDMAILW;

#endif // MAPI_FORCE_UNICODE
}
#endif // _WIN32


struct C_CopyFileProgress_to_IUpdateCallbackUI2 Z7_final:
  public ICopyFileProgress
{
  IUpdateCallbackUI2 *Callback;
  HRESULT CallbackResult;
  // bool Disable_Break;

  virtual DWORD CopyFileProgress(UInt64 total, UInt64 current) Z7_override
  {
    const HRESULT res = Callback->MoveArc_Progress(total, current);
    CallbackResult = res;
    // if (Disable_Break && res == E_ABORT) res = S_OK;
    return res == S_OK ? PROGRESS_CONTINUE : PROGRESS_CANCEL;
  }

  C_CopyFileProgress_to_IUpdateCallbackUI2(
      IUpdateCallbackUI2 *callback) :
    Callback(callback),
    CallbackResult(S_OK)
    // , Disable_Break(false)
    {}
};


HRESULT UpdateArchive(
    CCodecs *codecs,
    const CObjectVector<COpenType> &types,
    const UString &cmdArcPath2,
    NWildcard::CCensor &censor,
    CUpdateOptions &options,
    CUpdateErrorInfo &errorInfo,
    IOpenCallbackUI *openCallback,
    IUpdateCallbackUI2 *callback,
    bool needSetPath)
{
  if (options.StdOutMode && options.EMailMode)
    return E_FAIL;

  if (types.Size() > 1)
    return E_NOTIMPL;

  bool renameMode = !options.RenamePairs.IsEmpty();
  if (renameMode)
  {
    if (options.Commands.Size() != 1)
      return E_FAIL;
  }

  if (options.DeleteAfterCompressing)
  {
    if (options.Commands.Size() != 1)
      return E_NOTIMPL;
    const CActionSet &as = options.Commands[0].ActionSet;
    for (unsigned i = 2; i < NPairState::kNumValues; i++)
      if (as.StateActions[i] != NPairAction::kCompress)
        return E_NOTIMPL;
  }

  censor.AddPathsToCensor(options.PathMode);
  #ifdef _WIN32
  ConvertToLongNames(censor);
  #endif
  censor.ExtendExclude();

  
  if (options.VolumesSizes.Size() > 0 && (options.EMailMode /* || options.SfxMode */))
    return E_NOTIMPL;

  if (options.SfxMode)
  {
    CProperty property;
    property.Name = "rsfx";
    options.MethodMode.Properties.Add(property);
    if (options.SfxModule.IsEmpty())
    {
      errorInfo.Message = "SFX file is not specified";
      return E_FAIL;
    }
    bool found = false;
    if (options.SfxModule.Find(FCHAR_PATH_SEPARATOR) < 0)
    {
      const FString fullName = NDLL::GetModuleDirPrefix() + options.SfxModule;
      if (NFind::DoesFileExist_FollowLink(fullName))
      {
        options.SfxModule = fullName;
        found = true;
      }
    }
    if (!found)
    {
      if (!NFind::DoesFileExist_FollowLink(options.SfxModule))
        return errorInfo.SetFromLastError("cannot find specified SFX module", options.SfxModule);
    }
  }

  CArchiveLink arcLink;

  
  if (needSetPath)
  {
    if (!options.InitFormatIndex(codecs, types, cmdArcPath2) ||
        !options.SetArcPath(codecs, cmdArcPath2))
      return E_NOTIMPL;
  }
  
  UString arcPath = options.ArchivePath.GetFinalPath();

  if (!options.VolumesSizes.IsEmpty())
  {
    arcPath = options.ArchivePath.GetFinalVolPath();
    arcPath += ".001";
  }

  if (cmdArcPath2.IsEmpty())
  {
    if (options.MethodMode.Type.FormatIndex < 0)
      throw "type of archive is not specified";
  }
  else
  {
    NFind::CFileInfo fi;
    if (!fi.Find_FollowLink(us2fs(arcPath)))
    {
      if (renameMode)
        throw "can't find archive";
      if (options.MethodMode.Type.FormatIndex < 0)
      {
        if (!options.SetArcPath(codecs, cmdArcPath2))
          return E_NOTIMPL;
      }
    }
    else
    {
      if (fi.IsDir())
        return errorInfo.SetFromError_DWORD("There is a folder with the name of archive",
            us2fs(arcPath),
            #ifdef _WIN32
              ERROR_ACCESS_DENIED
            #else
              EISDIR
            #endif
            );
     #ifdef _WIN32
      if (fi.IsDevice)
        return E_NOTIMPL;
     #endif

      if (!options.StdOutMode && options.UpdateArchiveItself)
        if (fi.IsReadOnly())
        {
          return errorInfo.SetFromError_DWORD("The file is read-only",
              us2fs(arcPath),
              #ifdef _WIN32
                ERROR_ACCESS_DENIED
              #else
                EACCES
              #endif
              );
        }

      if (options.VolumesSizes.Size() > 0)
      {
        errorInfo.FileNames.Add(us2fs(arcPath));
        // errorInfo.SystemError = (DWORD)E_NOTIMPL;
        errorInfo.Message = kUpdateIsNotSupported_MultiVol;
        return E_NOTIMPL;
      }
      CObjectVector<COpenType> types2;
      // change it.
      if (options.MethodMode.Type_Defined)
        types2.Add(options.MethodMode.Type);
      // We need to set Properties to open archive only in some cases (WIM archives).

      CIntVector excl;
      COpenOptions op;
      #ifndef Z7_SFX
      op.props = &options.MethodMode.Properties;
      #endif
      op.codecs = codecs;
      op.types = &types2;
      op.excludedFormats = &excl;
      op.stdInMode = false;
      op.stream = NULL;
      op.filePath = arcPath;

      RINOK(callback->StartOpenArchive(arcPath))

      HRESULT result = arcLink.Open_Strict(op, openCallback);

      if (result == E_ABORT)
        return result;
      
      HRESULT res2 = callback->OpenResult(codecs, arcLink, arcPath, result);
      /*
      if (result == S_FALSE)
        return E_FAIL;
      */
      RINOK(res2)
      RINOK(result)

      if (arcLink.VolumePaths.Size() > 1)
      {
        // errorInfo.SystemError = (DWORD)E_NOTIMPL;
        errorInfo.Message = kUpdateIsNotSupported_MultiVol;
        return E_NOTIMPL;
      }
      
      CArc &arc = arcLink.Arcs.Back();
      arc.MTime.Def =
        #ifdef _WIN32
          !fi.IsDevice;
        #else
          true;
        #endif
      if (arc.MTime.Def)
        arc.MTime.Set_From_FiTime(fi.MTime);

      if (arc.ErrorInfo.ThereIsTail)
      {
        // errorInfo.SystemError = (DWORD)E_NOTIMPL;
        errorInfo.Message = "There is some data block after the end of the archive";
        return E_NOTIMPL;
      }
      if (options.MethodMode.Type.FormatIndex < 0)
      {
        options.MethodMode.Type.FormatIndex = arcLink.GetArc()->FormatIndex;
        if (!options.SetArcPath(codecs, cmdArcPath2))
          return E_NOTIMPL;
      }
    }
  }

  if (options.MethodMode.Type.FormatIndex < 0)
  {
    options.MethodMode.Type.FormatIndex = codecs->FindFormatForArchiveType((UString)kDefaultArcType);
    if (options.MethodMode.Type.FormatIndex < 0)
      return E_NOTIMPL;
  }

  const bool thereIsInArchive = arcLink.IsOpen;
  if (!thereIsInArchive && renameMode)
    return E_FAIL;
  
  CDirItems dirItems;
  dirItems.Callback = callback;

  CDirItem parentDirItem;
  CDirItem *parentDirItem_Ptr = NULL;
  
  /*
  FStringVector requestedPaths;
  FStringVector *requestedPaths_Ptr = NULL;
  if (options.DeleteAfterCompressing)
    requestedPaths_Ptr = &requestedPaths;
  */

  if (options.StdInMode)
  {
    CDirItem di;
    // di.ClearBase();
    // di.Size = (UInt64)(Int64)-1;
    if (!di.SetAs_StdInFile())
      return GetLastError_noZero_HRESULT();
    di.Name = options.StdInFileName;
    // di.Attrib_IsDefined = false;
    // NTime::GetCurUtc_FiTime(di.MTime);
    // di.CTime = di.ATime = di.MTime;
    dirItems.Items.Add(di);
  }
  else
  {
    bool needScanning = false;
    
    if (!renameMode)
    FOR_VECTOR (i, options.Commands)
      if (options.Commands[i].ActionSet.NeedScanning())
        needScanning = true;

    if (needScanning)
    {
      RINOK(callback->StartScanning())

      dirItems.SymLinks = options.SymLinks.Val;

      #if defined(_WIN32) && !defined(UNDER_CE)
      dirItems.ReadSecure = options.NtSecurity.Val;
      #endif

      dirItems.ScanAltStreams = options.AltStreams.Val;
      dirItems.ExcludeDirItems = censor.ExcludeDirItems;
      dirItems.ExcludeFileItems = censor.ExcludeFileItems;
      
      dirItems.ShareForWrite = options.OpenShareForWrite;

     #ifndef _WIN32
      dirItems.StoreOwnerName = options.StoreOwnerName.Val;
     #endif

      const HRESULT res = EnumerateItems(censor,
          options.PathMode,
          UString(), // options.AddPathPrefix,
          dirItems);

      if (res != S_OK)
      {
        if (res != E_ABORT)
          errorInfo.Message = "Scanning error";
        return res;
      }
      
      RINOK(callback->FinishScanning(dirItems.Stat))

      // 22.00: we don't need parent folder, if absolute path mode
      if (options.PathMode != NWildcard::k_AbsPath)
      if (censor.Pairs.Size() == 1)
      {
        NFind::CFileInfo fi;
        FString prefix = us2fs(censor.Pairs[0].Prefix);
        prefix.Add_Dot();
        // UString prefix = censor.Pairs[0].Prefix;
        /*
        if (prefix.Back() == WCHAR_PATH_SEPARATOR)
        {
          prefix.DeleteBack();
        }
        */
        if (fi.Find(prefix))
          if (fi.IsDir())
          {
            parentDirItem.Copy_From_FileInfoBase(fi);
            parentDirItem_Ptr = &parentDirItem;

            int secureIndex = -1;
            #if defined(_WIN32) && !defined(UNDER_CE)
            if (options.NtSecurity.Val)
              dirItems.AddSecurityItem(prefix, secureIndex);
            #endif
            parentDirItem.SecureIndex = secureIndex;
          }
      }
    }
  }

  FString tempDirPrefix;
  bool usesTempDir = false;
  
  #ifdef _WIN32
  CTempDir tempDirectory;
  if (options.EMailMode && options.EMailRemoveAfter)
  {
    tempDirectory.Create(kTempFolderPrefix);
    tempDirPrefix = tempDirectory.GetPath();
    NormalizeDirPathPrefix(tempDirPrefix);
    usesTempDir = true;
  }
  #endif

  CTempFiles tempFiles;

  bool createTempFile = false;

  if (!options.StdOutMode && options.UpdateArchiveItself)
  {
    CArchivePath &ap = options.Commands[0].ArchivePath;
    ap = options.ArchivePath;
    // if ((archive != 0 && !usesTempDir) || !options.WorkingDir.IsEmpty())
    if ((thereIsInArchive || !options.WorkingDir.IsEmpty()) && !usesTempDir && options.VolumesSizes.Size() == 0)
    {
      createTempFile = true;
      ap.Temp = true;
      if (!options.WorkingDir.IsEmpty())
        ap.TempPrefix = options.WorkingDir;
      else
        ap.TempPrefix = us2fs(ap.Prefix);
      NormalizeDirPathPrefix(ap.TempPrefix);
    }
  }

  unsigned ci;


  // self including protection
  if (options.DeleteAfterCompressing)
  {
    for (ci = 0; ci < options.Commands.Size(); ci++)
    {
      CArchivePath &ap = options.Commands[ci].ArchivePath;
      const FString path = us2fs(ap.GetFinalPath());
      // maybe we must compare absolute paths path here
      FOR_VECTOR (i, dirItems.Items)
      {
        const FString phyPath = dirItems.GetPhyPath(i);
        if (phyPath == path)
        {
          UString s;
          s = "It is not allowed to include archive to itself";
          s.Add_LF();
          s += fs2us(path);
          throw s;
        }
      }
    }
  }


  for (ci = 0; ci < options.Commands.Size(); ci++)
  {
    CArchivePath &ap = options.Commands[ci].ArchivePath;
    if (usesTempDir)
    {
      // Check it
      ap.Prefix = fs2us(tempDirPrefix);
      // ap.Temp = true;
      // ap.TempPrefix = tempDirPrefix;
    }
    if (!options.StdOutMode &&
        (ci > 0 || !createTempFile))
    {
      const FString path = us2fs(ap.GetFinalPath());
      if (NFind::DoesFileOrDirExist(path))
      {
        errorInfo.SystemError = ERROR_FILE_EXISTS;
        errorInfo.Message = "The file already exists";
        errorInfo.FileNames.Add(path);
        return errorInfo.Get_HRESULT_Error();
      }
    }
  }

  CObjectVector<CArcItem> arcItems;
  if (thereIsInArchive)
  {
    RINOK(EnumerateInArchiveItems(
      // options.StoreAltStreams,
      censor, arcLink.Arcs.Back(), arcItems))
  }

  /*
  FStringVector processedFilePaths;
  FStringVector *processedFilePaths_Ptr = NULL;
  if (options.DeleteAfterCompressing)
    processedFilePaths_Ptr = &processedFilePaths;
  */

  CByteBuffer processedItems;
  if (options.DeleteAfterCompressing)
  {
    const unsigned num = dirItems.Items.Size();
    processedItems.Alloc(num);
    for (unsigned i = 0; i < num; i++)
      processedItems[i] = 0;
  }

  CMultiOutStream_Bunch multiStreams;

  /*
  #ifndef Z7_NO_CRYPTO
  if (arcLink.PasswordWasAsked)
  {
    // We set password, if open have requested password
    RINOK(callback->SetPassword(arcLink.Password));
  }
  #endif
  */

  for (ci = 0; ci < options.Commands.Size(); ci++)
  {
    const CArc *arc = thereIsInArchive ? arcLink.GetArc() : NULL;
    CUpdateArchiveCommand &command = options.Commands[ci];
    UString name;
    bool isUpdating;
    
    if (options.StdOutMode)
    {
      name = "stdout";
      isUpdating = thereIsInArchive;
    }
    else
    {
      name = command.ArchivePath.GetFinalPath();
      isUpdating = (ci == 0 && options.UpdateArchiveItself && thereIsInArchive);
    }
    
    RINOK(callback->StartArchive(name, isUpdating))

    CFinishArchiveStat st;

    RINOK(Compress(options,
        isUpdating,
        codecs,
        command.ActionSet,
        arc,
        command.ArchivePath,
        arcItems,
        options.DeleteAfterCompressing ? (Byte *)processedItems : NULL,

        dirItems,
        parentDirItem_Ptr,

        tempFiles,
        multiStreams,
        errorInfo, callback, st))

    RINOK(callback->FinishArchive(st))
  }


  if (thereIsInArchive)
  {
    RINOK(arcLink.Close())
    arcLink.Release();
  }

  multiStreams.DisableDeletion();
  RINOK(multiStreams.Destruct())

  // here we disable deleting of temp archives.
  // note: archive moving can fail, or it can be interrupted,
  // if we move new temp update from another volume.
  // And we still want to keep temp archive in that case,
  // because we will have deleted original archive.
  tempFiles.NeedDeleteFiles = false;
  // tempFiles.Paths.Clear();

  if (createTempFile)
  {
    try
    {
      CArchivePath &ap = options.Commands[0].ArchivePath;
      const FString &tempPath = ap.GetTempPath();
      
      // DWORD attrib = 0;
      if (thereIsInArchive)
      {
        // attrib = NFind::GetFileAttrib(us2fs(arcPath));
        if (!DeleteFileAlways(us2fs(arcPath)))
          return errorInfo.SetFromLastError("cannot delete the file", us2fs(arcPath));
      }

      UInt64 totalArcSize = 0;
      {
        NFind::CFileInfo fi;
        if (fi.Find(tempPath))
          totalArcSize = fi.Size;
      }
      RINOK(callback->MoveArc_Start(fs2us(tempPath), arcPath,
          totalArcSize, BoolToInt(thereIsInArchive)))

      C_CopyFileProgress_to_IUpdateCallbackUI2 prox(callback);
      // if we update archive, we have removed original archive.
      // So if we break archive moving, we will have only temporary archive.
      // We can disable breaking here:
      // prox.Disable_Break = thereIsInArchive;

      if (!MyMoveFile_with_Progress(tempPath, us2fs(arcPath), &prox))
      {
        errorInfo.SystemError = ::GetLastError();
        errorInfo.Message = "cannot move the file";
        if (errorInfo.SystemError == ERROR_INVALID_PARAMETER)
        {
          if (totalArcSize > (UInt32)(Int32)-1)
          {
            // bool isFsDetected = false;
            // if (NSystem::Is_File_LimitedBy_4GB(us2fs(arcPath), isFsDetected) || !isFsDetected)
            {
              errorInfo.Message.Add_LF();
              errorInfo.Message += "Archive file size exceeds 4 GB";
            }
          }
        }
        // if there was no input archive, and we have operation breaking.
        // then we can remove temporary archive, because we still have original uncompressed files.
        if (!thereIsInArchive
            && prox.CallbackResult == E_ABORT)
          tempFiles.NeedDeleteFiles = true;
        errorInfo.FileNames.Add(tempPath);
        errorInfo.FileNames.Add(us2fs(arcPath));
        RINOK(prox.CallbackResult)
        return errorInfo.Get_HRESULT_Error();
      }

      // MoveArc_Finish() can return delayed user break (E_ABORT) status,
      // if callback callee ignored interruption to finish archive creation operation.
      RINOK(callback->MoveArc_Finish())
      
      /*
      if (attrib != INVALID_FILE_ATTRIBUTES && (attrib & FILE_ATTRIBUTE_READONLY))
      {
        DWORD attrib2 = NFind::GetFileAttrib(us2fs(arcPath));
        if (attrib2 != INVALID_FILE_ATTRIBUTES)
          NDir::SetFileAttrib(us2fs(arcPath), attrib2 | FILE_ATTRIBUTE_READONLY);
      }
      */
    }
    catch(...)
    {
      throw;
    }
  }


  #if defined(_WIN32) && !defined(UNDER_CE)

Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION
  
  if (options.EMailMode)
  {
    NDLL::CLibrary mapiLib;
    if (!mapiLib.Load(FTEXT("Mapi32.dll")))
    {
      errorInfo.SetFromLastError("cannot load Mapi32.dll");
      return errorInfo.Get_HRESULT_Error();
    }

    FStringVector fullPaths;
    unsigned i;
    
    for (i = 0; i < options.Commands.Size(); i++)
    {
      CArchivePath &ap = options.Commands[i].ArchivePath;
      const FString finalPath = us2fs(ap.GetFinalPath());
      FString arcPath2;
      if (!MyGetFullPathName(finalPath, arcPath2))
        return errorInfo.SetFromLastError("GetFullPathName error", finalPath);
      fullPaths.Add(arcPath2);
    }

    /*
    LPMAPISENDDOCUMENTS fnSend = (LPMAPISENDDOCUMENTS)mapiLib.GetProc("MAPISendDocuments");
    if (fnSend == 0)
    {
      errorInfo.SetFromLastError)("7-Zip cannot find MAPISendDocuments function");
      return errorInfo.Get_HRESULT_Error();
    }
    */
    const
    Z7_WIN_LPMAPISENDMAILW sendMailW = Z7_GET_PROC_ADDRESS(
    Z7_WIN_LPMAPISENDMAILW, mapiLib.Get_HMODULE(),
            "MAPISendMailW");
   if (sendMailW)
   {

    CCurrentDirRestorer curDirRestorer;

    UStringVector paths;
    UStringVector names;
    
    for (i = 0; i < fullPaths.Size(); i++)
    {
      const UString arcPath2 = fs2us(fullPaths[i]);
      const UString fileName = ExtractFileNameFromPath(arcPath2);
      paths.Add(arcPath2);
      names.Add(fileName);
      // Warning!!! MAPISendDocuments function changes Current directory
      // fnSend(0, ";", (LPSTR)(LPCSTR)path, (LPSTR)(LPCSTR)name, 0);
    }

    CRecordVector<Z7_WIN_MapiFileDescW> files;
    files.ClearAndSetSize(paths.Size());
    
    for (i = 0; i < paths.Size(); i++)
    {
      Z7_WIN_MapiFileDescW &f = files[i];
      memset(&f, 0, sizeof(f));
      f.nPosition = 0xFFFFFFFF;
      f.lpszPathName = paths[i].Ptr_non_const();
      f.lpszFileName = names[i].Ptr_non_const();
    }

    {
      Z7_WIN_MapiMessageW m;
      memset(&m, 0, sizeof(m));
      m.nFileCount = files.Size();
      m.lpFiles = files.NonConstData();
      
      const UString addr (options.EMailAddress);
      Z7_WIN_MapiRecipDescW rec;
      if (!addr.IsEmpty())
      {
        memset(&rec, 0, sizeof(rec));
        rec.ulRecipClass = MAPI_TO;
        rec.lpszAddress = addr.Ptr_non_const();
        m.nRecipCount = 1;
        m.lpRecips = &rec;
      }
      
      sendMailW((LHANDLE)0, 0, &m, MAPI_DIALOG, 0);
    }
   }
   else
   {
    const
    LPMAPISENDMAIL sendMail = Z7_GET_PROC_ADDRESS(
    LPMAPISENDMAIL, mapiLib.Get_HMODULE(),
     "MAPISendMail");
    if (!sendMail)
    {
      errorInfo.SetFromLastError("7-Zip cannot find MAPISendMail function");
      return errorInfo.Get_HRESULT_Error();
    }

    CCurrentDirRestorer curDirRestorer;

    AStringVector paths;
    AStringVector names;
    
    for (i = 0; i < fullPaths.Size(); i++)
    {
      const UString arcPath2 = fs2us(fullPaths[i]);
      const UString fileName = ExtractFileNameFromPath(arcPath2);
      paths.Add(GetAnsiString(arcPath2));
      names.Add(GetAnsiString(fileName));
      // const AString path (GetAnsiString(arcPath2));
      // const AString name (GetAnsiString(fileName));
      // Warning!!! MAPISendDocuments function changes Current directory
      // fnSend(0, ";", (LPSTR)(LPCSTR)path, (LPSTR)(LPCSTR)name, 0);
    }

    CRecordVector<MapiFileDesc> files;
    files.ClearAndSetSize(paths.Size());
    
    for (i = 0; i < paths.Size(); i++)
    {
      MapiFileDesc &f = files[i];
      memset(&f, 0, sizeof(f));
      f.nPosition = 0xFFFFFFFF;
      f.lpszPathName = paths[i].Ptr_non_const();
      f.lpszFileName = names[i].Ptr_non_const();
    }

    {
      MapiMessage m;
      memset(&m, 0, sizeof(m));
      m.nFileCount = files.Size();
      m.lpFiles = files.NonConstData();
      
      const AString addr (GetAnsiString(options.EMailAddress));
      MapiRecipDesc rec;
      if (!addr.IsEmpty())
      {
        memset(&rec, 0, sizeof(rec));
        rec.ulRecipClass = MAPI_TO;
        rec.lpszAddress = addr.Ptr_non_const();
        m.nRecipCount = 1;
        m.lpRecips = &rec;
      }
      
      sendMail((LHANDLE)0, 0, &m, MAPI_DIALOG, 0);
    }
   }
  }
  
  #endif

  if (options.DeleteAfterCompressing)
  {
    CRecordVector<CDirPathSortPair> pairs;
    FStringVector foldersNames;

    unsigned i;

    for (i = 0; i < dirItems.Items.Size(); i++)
    {
      const CDirItem &dirItem = dirItems.Items[i];
      const FString phyPath = dirItems.GetPhyPath(i);
      if (dirItem.IsDir())
      {
        CDirPathSortPair pair;
        pair.Index = i;
        pair.SetNumSlashes(phyPath);
        pairs.Add(pair);
      }
      else
      {
        // 21.04: we have set processedItems[*] before for all required items
        if (processedItems[i] != 0
            // || dirItem.Size == 0
            // || dirItem.AreReparseData()
            )
        {
          NFind::CFileInfo fileInfo;
          /* if (!SymLinks), we follow link here, similar to (dirItem) filling */
          if (fileInfo.Find(phyPath, !options.SymLinks.Val))
          {
            bool is_SameSize = false;
            if (options.SymLinks.Val && dirItem.AreReparseData())
            {
              /* (dirItem.Size = dirItem.ReparseData.Size()) was set before.
                 So we don't compare sizes for that case here */
              is_SameSize = fileInfo.IsOsSymLink();
            }
            else
              is_SameSize = (fileInfo.Size == dirItem.Size);

            if (is_SameSize
                && Compare_FiTime(&fileInfo.MTime, &dirItem.MTime) == 0
                && Compare_FiTime(&fileInfo.CTime, &dirItem.CTime) == 0)
            {
              RINOK(callback->DeletingAfterArchiving(phyPath, false))
              DeleteFileAlways(phyPath);
            }
          }
        }
        else
        {
          // file was skipped by some reason. We can throw error for debug:
          /*
          errorInfo.SystemError = 0;
          errorInfo.Message = "file was not processed";
          errorInfo.FileNames.Add(phyPath);
          return E_FAIL;
          */
        }
      }
    }

    pairs.Sort2();
    
    for (i = 0; i < pairs.Size(); i++)
    {
      const FString phyPath = dirItems.GetPhyPath(pairs[i].Index);
      if (NFind::DoesDirExist(phyPath))
      {
        RINOK(callback->DeletingAfterArchiving(phyPath, true))
        RemoveDirAlways_if_Empty(phyPath);
      }
    }

    RINOK(callback->FinishDeletingAfterArchiving())
  }

  return S_OK;
}
