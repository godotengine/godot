// ArchiveExtractCallback.cpp

#include "StdAfx.h"

#undef sprintf
#undef printf

// #include <stdio.h>

#include "../../../../C/Alloc.h"
#include "../../../../C/CpuArch.h"

#include "../../../Common/ComTry.h"
#include "../../../Common/IntToString.h"
#include "../../../Common/StringConvert.h"
#include "../../../Common/UTFConvert.h"
#include "../../../Common/Wildcard.h"

#include "../../../Windows/ErrorMsg.h"
#include "../../../Windows/FileDir.h"
#include "../../../Windows/FileFind.h"
#include "../../../Windows/FileName.h"
#include "../../../Windows/PropVariant.h"
#include "../../../Windows/PropVariantConv.h"

#if defined(_WIN32) && !defined(UNDER_CE)  && !defined(Z7_SFX)
#define Z7_USE_SECURITY_CODE
#include "../../../Windows/SecurityUtils.h"
#endif

#include "../../Common/FilePathAutoRename.h"
#include "../../Common/StreamUtils.h"

#include "../../Archive/Common/ItemNameUtils.h"

#include "../Common/ExtractingFilePath.h"
#include "../Common/PropIDUtils.h"

#include "ArchiveExtractCallback.h"

using namespace NWindows;
using namespace NFile;
using namespace NDir;

static const char * const kCantAutoRename = "Cannot create file with auto name";
static const char * const kCantRenameFile = "Cannot rename existing file";
static const char * const kCantDeleteOutputFile = "Cannot delete output file";
static const char * const kCantDeleteOutputDir = "Cannot delete output folder";
static const char * const kCantOpenOutFile = "Cannot open output file";
#ifndef Z7_SFX
static const char * const kCantOpenInFile = "Cannot open input file";
#endif
static const char * const kCantSetFileLen = "Cannot set length for output file";
#ifdef SUPPORT_LINKS
static const char * const kCantCreateHardLink = "Cannot create hard link";
static const char * const kCantCreateSymLink = "Cannot create symbolic link";
static const char * const k_HardLink_to_SymLink_Ignored = "Hard link to symbolic link was ignored";
static const char * const k_CantDelete_File_for_SymLink = "Cannot delete file for symbolic link creation";
static const char * const k_CantDelete_Dir_for_SymLink = "Cannot delete directory for symbolic link creation";
#endif

static const unsigned k_LinkDataSize_LIMIT = 1 << 12;

#ifdef SUPPORT_LINKS
#if WCHAR_PATH_SEPARATOR != L'/'
  // we convert linux slashes to windows slashes for further processing.
  // also we convert linux backslashes to BackslashReplacement character.
  #define REPLACE_SLASHES_from_Linux_to_Sys(s) \
    { NArchive::NItemName::ReplaceToWinSlashes(s, true); }  // useBackslashReplacement
      // { s.Replace(L'/', WCHAR_PATH_SEPARATOR); }
#else
  #define REPLACE_SLASHES_from_Linux_to_Sys(s)
#endif
#endif

#ifndef Z7_SFX

Z7_COM7F_IMF(COutStreamWithHash::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  HRESULT result = S_OK;
  if (_stream)
    result = _stream->Write(data, size, &size);
  if (_calculate)
    _hash->Update(data, size);
  _size += size;
  if (processedSize)
    *processedSize = size;
  return result;
}

#endif // Z7_SFX


#ifdef Z7_USE_SECURITY_CODE
bool InitLocalPrivileges();
bool InitLocalPrivileges()
{
  NSecurity::CAccessToken token;
  if (!token.OpenProcessToken(GetCurrentProcess(),
      TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES))
    return false;

  TOKEN_PRIVILEGES tp;
 
  tp.PrivilegeCount = 1;
  tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
  
  if  (!::LookupPrivilegeValue(NULL, SE_SECURITY_NAME, &tp.Privileges[0].Luid))
    return false;
  if (!token.AdjustPrivileges(&tp))
    return false;
  return (GetLastError() == ERROR_SUCCESS);
}
#endif // Z7_USE_SECURITY_CODE



#if defined(_WIN32) && !defined(UNDER_CE) && !defined(Z7_SFX)

static const char * const kOfficeExtensions =
  " doc dot wbk"
  " docx docm dotx dotm docb wll wwl"
  " xls xlt xlm"
  " xlsx xlsm xltx xltm xlsb xla xlam"
  " ppt pot pps ppa ppam"
  " pptx pptm potx potm ppam ppsx ppsm sldx sldm"
  " ";

static bool FindExt2(const char *p, const UString &name)
{
  const int pathPos = name.ReverseFind_PathSepar();
  const int dotPos = name.ReverseFind_Dot();
  if (dotPos < 0
      || dotPos < pathPos
      || dotPos == (int)name.Len() - 1)
    return false;

  AString s;
  for (unsigned pos = (unsigned)(dotPos + 1);; pos++)
  {
    const wchar_t c = name[pos];
    if (c <= 0)
      break;
    if (c >= 0x80)
      return false;
    s.Add_Char((char)MyCharLower_Ascii((char)c));
  }
  for (unsigned i = 0; p[i] != 0;)
  {
    unsigned j;
    for (j = i; p[j] != ' '; j++);
    if (s.Len() == j - i && memcmp(p + i, (const char *)s, s.Len()) == 0)
      return true;
    i = j + 1;
  }
  return false;
}


static const char * const k_ZoneId_StreamName_With_Colon_Prefix = ":Zone.Identifier";

bool Is_ZoneId_StreamName(const wchar_t *s)
{
  return StringsAreEqualNoCase_Ascii(s, k_ZoneId_StreamName_With_Colon_Prefix + 1);
}

void ReadZoneFile_Of_BaseFile(CFSTR fileName, CByteBuffer &buf)
{
  buf.Free();
  FString path (fileName);
  path += k_ZoneId_StreamName_With_Colon_Prefix;
  NIO::CInFile file;
  if (!file.Open(path))
    return;
  UInt64 fileSize;
  if (!file.GetLength(fileSize))
    return;
  if (fileSize == 0 || fileSize >= (1u << 15))
    return;
  buf.Alloc((size_t)fileSize);
  size_t processed;
  if (file.ReadFull(buf, (size_t)fileSize, processed) && processed == fileSize)
    return;
  buf.Free();
}

bool WriteZoneFile_To_BaseFile(CFSTR fileName, const CByteBuffer &buf)
{
  FString path (fileName);
  path += k_ZoneId_StreamName_With_Colon_Prefix;
  NIO::COutFile file;
  if (!file.Create_ALWAYS(path))
    return false;
  return file.WriteFull(buf, buf.Size());
}

#endif


#ifdef SUPPORT_LINKS

int CHardLinkNode::Compare(const CHardLinkNode &a) const
{
  if (StreamId < a.StreamId) return -1;
  if (StreamId > a.StreamId) return 1;
  return MyCompare(INode, a.INode);
}

static HRESULT Archive_Get_HardLinkNode(IInArchive *archive, UInt32 index, CHardLinkNode &h, bool &defined)
{
  h.INode = 0;
  h.StreamId = (UInt64)(Int64)-1;
  defined = false;
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidINode, &prop))
    if (!ConvertPropVariantToUInt64(prop, h.INode))
      return S_OK;
  }
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidStreamId, &prop))
    ConvertPropVariantToUInt64(prop, h.StreamId);
  }
  defined = true;
  return S_OK;
}


HRESULT CArchiveExtractCallback::PrepareHardLinks(const CRecordVector<UInt32> *realIndices)
{
  _hardLinks.Clear();

  if (!_arc->Ask_INode)
    return S_OK;
  
  IInArchive * const archive = _arc->Archive;
  CRecordVector<CHardLinkNode> &hardIDs = _hardLinks.IDs;

  {
    UInt32 numItems;
    if (realIndices)
      numItems = realIndices->Size();
    else
    {
      RINOK(archive->GetNumberOfItems(&numItems))
    }

    for (UInt32 i = 0; i < numItems; i++)
    {
      CHardLinkNode h;
      bool defined;
      const UInt32 realIndex = realIndices ? (*realIndices)[i] : i;

      RINOK(Archive_Get_HardLinkNode(archive, realIndex, h, defined))
      if (defined)
      {
        bool isAltStream = false;
        RINOK(Archive_IsItem_AltStream(archive, realIndex, isAltStream))
        if (!isAltStream)
        {
          bool isDir = false;
          RINOK(Archive_IsItem_Dir(archive, realIndex, isDir))
          if (!isDir)
            hardIDs.Add(h);
        }
      }
    }
  }
  
  hardIDs.Sort2();
  
  {
    // we keep only items that have 2 or more items
    unsigned k = 0;
    unsigned numSame = 1;
    for (unsigned i = 1; i < hardIDs.Size(); i++)
    {
      if (hardIDs[i].Compare(hardIDs[i - 1]) != 0)
        numSame = 1;
      else if (++numSame == 2)
      {
        if (i - 1 != k)
          hardIDs[k] = hardIDs[i - 1];
        k++;
      }
    }
    hardIDs.DeleteFrom(k);
  }
  
  _hardLinks.PrepareLinks();
  return S_OK;
}

#endif // SUPPORT_LINKS


CArchiveExtractCallback::CArchiveExtractCallback():
    // Write_CTime(true),
    // Write_ATime(true),
    // Write_MTime(true),
    Is_elimPrefix_Mode(false),
    _arc(NULL),
    _multiArchives(false)
{
  #ifdef Z7_USE_SECURITY_CODE
  _saclEnabled = InitLocalPrivileges();
  #endif
}


void CArchiveExtractCallback::InitBeforeNewArchive()
{
#if defined(_WIN32) && !defined(UNDER_CE) && !defined(Z7_SFX)
  ZoneBuf.Free();
#endif
}

void CArchiveExtractCallback::Init(
    const CExtractNtOptions &ntOptions,
    const NWildcard::CCensorNode *wildcardCensor,
    const CArc *arc,
    IFolderArchiveExtractCallback *extractCallback2,
    bool stdOutMode, bool testMode,
    const FString &directoryPath,
    const UStringVector &removePathParts, bool removePartsForAltStreams,
    UInt64 packSize)
{
  ClearExtractedDirsInfo();
  _outFileStream.Release();
  _bufPtrSeqOutStream.Release();
  
#ifdef SUPPORT_LINKS
  _hardLinks.Clear();
  _postLinks.Clear();
#endif

#ifdef SUPPORT_ALT_STREAMS
  _renamedFiles.Clear();
#endif

  _ntOptions = ntOptions;
  _wildcardCensor = wildcardCensor;
  _stdOutMode = stdOutMode;
  _testMode = testMode;
  _packTotal = packSize;
  _progressTotal = packSize;
  // _progressTotal = 0;
  // _progressTotal_Defined = false;
  // _progressTotal_Defined = true;
  _extractCallback2 = extractCallback2;
  /*
  _compressProgress.Release();
  _extractCallback2.QueryInterface(IID_ICompressProgressInfo, &_compressProgress);
  _callbackMessage.Release();
  _extractCallback2.QueryInterface(IID_IArchiveExtractCallbackMessage2, &_callbackMessage);
  */
  _folderArchiveExtractCallback2.Release();
  _extractCallback2.QueryInterface(IID_IFolderArchiveExtractCallback2, &_folderArchiveExtractCallback2);

  #ifndef Z7_SFX

  ExtractToStreamCallback.Release();
  _extractCallback2.QueryInterface(IID_IFolderExtractToStreamCallback, &ExtractToStreamCallback);
  if (ExtractToStreamCallback)
  {
    Int32 useStreams = 0;
    if (ExtractToStreamCallback->UseExtractToStream(&useStreams) != S_OK)
      useStreams = 0;
    if (useStreams == 0)
      ExtractToStreamCallback.Release();
  }
  
  #endif

  LocalProgressSpec->Init(extractCallback2, true);
  LocalProgressSpec->SendProgress = false;
 
  _removePathParts = removePathParts;
  _removePartsForAltStreams = removePartsForAltStreams;

  #ifndef Z7_SFX
  _baseParentFolder = (UInt32)(Int32)-1;
  _use_baseParentFolder_mode = false;
  #endif

  _arc = arc;
  _dirPathPrefix = directoryPath;
  _dirPathPrefix_Full = directoryPath;
  #if defined(_WIN32) && !defined(UNDER_CE)
  if (!NName::IsAltPathPrefix(_dirPathPrefix))
  #endif
  {
    NName::NormalizeDirPathPrefix(_dirPathPrefix);
    NDir::MyGetFullPathName(directoryPath, _dirPathPrefix_Full);
    NName::NormalizeDirPathPrefix(_dirPathPrefix_Full);
  }
}


Z7_COM7F_IMF(CArchiveExtractCallback::SetTotal(UInt64 size))
{
  COM_TRY_BEGIN
  _progressTotal = size;
  // _progressTotal_Defined = true;
  if (!_multiArchives && _extractCallback2)
    return _extractCallback2->SetTotal(size);
  return S_OK;
  COM_TRY_END
}


static void NormalizeVals(UInt64 &v1, UInt64 &v2)
{
  const UInt64 kMax = (UInt64)1 << 31;
  while (v1 > kMax)
  {
    v1 >>= 1;
    v2 >>= 1;
  }
}


static UInt64 MyMultDiv64(UInt64 unpCur, UInt64 unpTotal, UInt64 packTotal)
{
  NormalizeVals(packTotal, unpTotal);
  NormalizeVals(unpCur, unpTotal);
  if (unpTotal == 0)
    unpTotal = 1;
  return unpCur * packTotal / unpTotal;
}


Z7_COM7F_IMF(CArchiveExtractCallback::SetCompleted(const UInt64 *completeValue))
{
  COM_TRY_BEGIN
  
  if (!_extractCallback2)
    return S_OK;

  UInt64 packCur;
  if (_multiArchives)
  {
    packCur = LocalProgressSpec->InSize;
    if (completeValue /* && _progressTotal_Defined */)
      packCur += MyMultDiv64(*completeValue, _progressTotal, _packTotal);
    completeValue = &packCur;
  }
  return _extractCallback2->SetCompleted(completeValue);
 
  COM_TRY_END
}


Z7_COM7F_IMF(CArchiveExtractCallback::SetRatioInfo(const UInt64 *inSize, const UInt64 *outSize))
{
  COM_TRY_BEGIN
  return LocalProgressSpec.Interface()->SetRatioInfo(inSize, outSize);
  COM_TRY_END
}


void CArchiveExtractCallback::CreateComplexDirectory(
    const UStringVector &dirPathParts, bool isFinal, FString &fullPath)
{
  // we use (_item.IsDir) in this function

  bool isAbsPath = false;
  
  if (!dirPathParts.IsEmpty())
  {
    const UString &s = dirPathParts[0];
    if (s.IsEmpty())
      isAbsPath = true;
    #if defined(_WIN32) && !defined(UNDER_CE)
    else
    {
      if (NName::IsDrivePath2(s))
        isAbsPath = true;
    }
    #endif
  }
  
  if (_pathMode == NExtract::NPathMode::kAbsPaths && isAbsPath)
    fullPath.Empty();
  else
    fullPath = _dirPathPrefix;

  FOR_VECTOR (i, dirPathParts)
  {
    if (i != 0)
      fullPath.Add_PathSepar();
    const UString &s = dirPathParts[i];
    fullPath += us2fs(s);

    const bool isFinalDir = (i == dirPathParts.Size() - 1 && isFinal && _item.IsDir);
    
    if (fullPath.IsEmpty())
    {
      if (isFinalDir)
        _itemFailure = true;
      continue;
    }

    #if defined(_WIN32) && !defined(UNDER_CE)
    if (_pathMode == NExtract::NPathMode::kAbsPaths)
      if (i == 0 && s.Len() == 2 && NName::IsDrivePath2(s))
      {
        if (isFinalDir)
        {
          // we don't want to call SetAttrib() for root drive path
          _itemFailure = true;
        }
        continue;
      }
    #endif

    HRESULT hres = S_OK;
    if (!CreateDir(fullPath))
      hres = GetLastError_noZero_HRESULT();
    if (isFinalDir)
    {
      if (!NFile::NFind::DoesDirExist(fullPath))
      {
        _itemFailure = true;
        SendMessageError_with_Error(hres, "Cannot create folder", fullPath);
      }
    }
  }
}


HRESULT CArchiveExtractCallback::GetTime(UInt32 index, PROPID propID, CArcTime &ft)
{
  ft.Clear();
  NCOM::CPropVariant prop;
  RINOK(_arc->Archive->GetProperty(index, propID, &prop))
  if (prop.vt == VT_FILETIME)
    ft.Set_From_Prop(prop);
  else if (prop.vt != VT_EMPTY)
    return E_FAIL;
  return S_OK;
}


HRESULT CArchiveExtractCallback::GetUnpackSize()
{
  return _arc->GetItem_Size(_index, _curSize, _curSize_Defined);
}

static void AddPathToMessage(UString &s, const FString &path)
{
  s += " : ";
  s += fs2us(path);
}

HRESULT CArchiveExtractCallback::SendMessageError(const char *message, const FString &path) const
{
  UString s (message);
  AddPathToMessage(s, path);
  return _extractCallback2->MessageError(s);
}


HRESULT CArchiveExtractCallback::SendMessageError_with_Error(HRESULT errorCode, const char *message, const FString &path) const
{
  UString s (message);
  if (errorCode != S_OK)
  {
    s += " : ";
    s += NError::MyFormatMessage(errorCode);
  }
  AddPathToMessage(s, path);
  return _extractCallback2->MessageError(s);
}

HRESULT CArchiveExtractCallback::SendMessageError_with_LastError(const char *message, const FString &path) const
{
  const HRESULT errorCode = GetLastError_noZero_HRESULT();
  return SendMessageError_with_Error(errorCode, message, path);
}

HRESULT CArchiveExtractCallback::SendMessageError2(HRESULT errorCode, const char *message, const FString &path1, const FString &path2) const
{
  UString s (message);
  if (errorCode != 0)
  {
    s += " : ";
    s += NError::MyFormatMessage(errorCode);
  }
  AddPathToMessage(s, path1);
  AddPathToMessage(s, path2);
  return _extractCallback2->MessageError(s);
}

HRESULT CArchiveExtractCallback::SendMessageError2_with_LastError(
    const char *message, const FString &path1, const FString &path2) const
{
  const HRESULT errorCode = GetLastError_noZero_HRESULT();
  return SendMessageError2(errorCode, message, path1, path2);
}

#ifndef Z7_SFX

Z7_CLASS_IMP_COM_1(
  CGetProp
  , IGetProp
)
public:
  UInt32 IndexInArc;
  const CArc *Arc;
  // UString BaseName; // relative path
};

Z7_COM7F_IMF(CGetProp::GetProp(PROPID propID, PROPVARIANT *value))
{
  /*
  if (propID == kpidBaseName)
  {
    COM_TRY_BEGIN
    NCOM::CPropVariant prop = BaseName;
    prop.Detach(value);
    return S_OK;
    COM_TRY_END
  }
  */
  return Arc->Archive->GetProperty(IndexInArc, propID, value);
}

#endif // Z7_SFX


struct CLinkLevelsInfo
{
  bool IsAbsolute;
  bool ParentDirDots_after_NonParent;
  int LowLevel;
  int FinalLevel;

  void Parse(const UString &path, bool isWSL);
};

void CLinkLevelsInfo::Parse(const UString &path, bool isWSL)
{
  IsAbsolute = isWSL ?
      IS_PATH_SEPAR(path[0]) :
      NName::IsAbsolutePath(path);
  LowLevel = 0;
  FinalLevel = 0;
  ParentDirDots_after_NonParent = false;
  bool nonParentDir = false;

  UStringVector parts;
  SplitPathToParts(path, parts);
  int level = 0;
  
  FOR_VECTOR (i, parts)
  {
    const UString &s = parts[i];
    if (s.IsEmpty())
    {
      if (i == 0)
        IsAbsolute = true;
      continue;
    }
    if (s.IsEqualTo("."))
      continue;
    if (s.IsEqualTo(".."))
    {
      if (IsAbsolute || nonParentDir)
        ParentDirDots_after_NonParent = true;
      level--;
      if (LowLevel > level)
          LowLevel = level;
    }
    else
    {
      nonParentDir = true;
      level++;
    }
  }
  
  FinalLevel = level;
}


static bool IsSafePath(const UString &path, bool isWSL)
{
  CLinkLevelsInfo levelsInfo;
  levelsInfo.Parse(path, isWSL);
  return !levelsInfo.IsAbsolute
      && levelsInfo.LowLevel >= 0
      && levelsInfo.FinalLevel > 0;
}

bool IsSafePath(const UString &path);
bool IsSafePath(const UString &path)
{
  return IsSafePath(path, false); // isWSL
}

bool CensorNode_CheckPath2(const NWildcard::CCensorNode &node, const CReadArcItem &item, bool &include);
bool CensorNode_CheckPath2(const NWildcard::CCensorNode &node, const CReadArcItem &item, bool &include)
{
  bool found = false;

  // CheckPathVect() doesn't check path to Parent nodes
  if (node.CheckPathVect(item.PathParts, !item.MainIsDir, include))
  {
    if (!include)
      return true;
    
    #ifdef SUPPORT_ALT_STREAMS
    if (!item.IsAltStream)
      return true;
    #endif
    
    found = true;
  }
  
  #ifdef SUPPORT_ALT_STREAMS

  if (!item.IsAltStream)
    return false;
  
  UStringVector pathParts2 = item.PathParts;
  if (pathParts2.IsEmpty())
    pathParts2.AddNew();
  UString &back = pathParts2.Back();
  back.Add_Colon();
  back += item.AltStreamName;
  bool include2;
  
  if (node.CheckPathVect(pathParts2,
      true, // isFile,
      include2))
  {
    include = include2;
    return true;
  }

  #endif // SUPPORT_ALT_STREAMS

  return found;
}


bool CensorNode_CheckPath(const NWildcard::CCensorNode &node, const CReadArcItem &item)
{
  bool include;
  if (CensorNode_CheckPath2(node, item, include))
    return include;
  return false;
}


static FString MakePath_from_2_Parts(const FString &prefix, const FString &path)
{
  FString s (prefix);
  #if defined(_WIN32) && !defined(UNDER_CE)
  if (!path.IsEmpty() && path[0] == ':' && !prefix.IsEmpty() && IsPathSepar(prefix.Back()))
  {
    if (!NName::IsDriveRootPath_SuperAllowed(prefix))
      s.DeleteBack();
  }
  #endif
  s += path;
  return s;
}



#ifdef SUPPORT_LINKS

/*
struct CTempMidBuffer
{
  void *Buf;

  CTempMidBuffer(size_t size): Buf(NULL) { Buf = ::MidAlloc(size); }
  ~CTempMidBuffer() { ::MidFree(Buf); }
};

HRESULT CArchiveExtractCallback::MyCopyFile(ISequentialOutStream *outStream)
{
  const size_t kBufSize = 1 << 16;
  CTempMidBuffer buf(kBufSize);
  if (!buf.Buf)
    return E_OUTOFMEMORY;
  
  NIO::CInFile inFile;
  NIO::COutFile outFile;
  
  if (!inFile.Open(_copyFile_Path))
    return SendMessageError_with_LastError("Open error", _copyFile_Path);
    
  for (;;)
  {
    UInt32 num;
    
    if (!inFile.Read(buf.Buf, kBufSize, num))
      return SendMessageError_with_LastError("Read error", _copyFile_Path);
      
    if (num == 0)
      return S_OK;
      
      
    RINOK(WriteStream(outStream, buf.Buf, num));
  }
}
*/


HRESULT CArchiveExtractCallback::ReadLink()
{
  IInArchive * const archive = _arc->Archive;
  const UInt32 index = _index;
  // _link.Clear(); // _link.Clear() was called already.
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidHardLink, &prop))
    if (prop.vt == VT_BSTR)
    {
      _link.LinkType = k_LinkType_HardLink;
      _link.isRelative = false; // RAR5, TAR: hard links are from root folder of archive
      _link.LinkPath.SetFromBstr(prop.bstrVal);
      // 7-Zip 24-: tar handler returned original path (with linux slash in most case)
      // 7-Zip 24-: rar5 handler returned path with system slash.
      // 7-Zip 25+: tar/rar5 handlers return linux path in most cases.
    }
    else if (prop.vt != VT_EMPTY)
      return E_FAIL;
  }
  /*
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidCopyLink, &prop));
    if (prop.vt == VT_BSTR)
    {
      _link.LinkType = k_LinkType_CopyLink;
      _link.isRelative = false; // RAR5: copy links are from root folder of archive
      _link.LinkPath.SetFromBstr(prop.bstrVal);
    }
    else if (prop.vt != VT_EMPTY)
      return E_FAIL;
  }
  */
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidSymLink, &prop))
    if (prop.vt == VT_BSTR)
    {
      _link.LinkType = k_LinkType_PureSymLink;
      _link.isRelative = true; // RAR5, TAR: symbolic links are relative by default
      _link.LinkPath.SetFromBstr(prop.bstrVal);
      // 7-Zip 24-: (tar, cpio, xar, ext, iso) handlers returned returned original path (with linux slash in most case)
      // 7-Zip 24-: rar5 handler returned path with system slash.
      // 7-Zip 25+: all handlers return linux path in most cases.
    }
    else if (prop.vt != VT_EMPTY)
      return E_FAIL;
  }

  // linux path separator in (_link.LinkPath) is expected for most cases,
  // if new handler code is used, and if data in archive is correct.
  // NtReparse_Data = NULL;
  // NtReparse_Size = 0;
  if (!_link.LinkPath.IsEmpty())
  {
    REPLACE_SLASHES_from_Linux_to_Sys(_link.LinkPath)
  }
  else if (_arc->GetRawProps)
  {
    const void *data;
    UInt32 dataSize, propType;
    if (_arc->GetRawProps->GetRawProp(_index, kpidNtReparse, &data, &dataSize, &propType) == S_OK
        // && dataSize == 1234567 // for debug: unpacking without reparse
        && dataSize)
    {
      if (propType != NPropDataType::kRaw)
        return E_FAIL;
      // 21.06: we need kpidNtReparse in linux for wim archives created in Windows
      // NtReparse_Data = data;
      // NtReparse_Size = dataSize;
      // we ignore error code here, if there is failure of parsing:
      _link.Parse_from_WindowsReparseData((const Byte *)data, dataSize);
    }
  }

  if (_link.LinkPath.IsEmpty())
    return S_OK;
  // (_link.LinkPath) uses system path separator.
  // windows: (_link.LinkPath) doesn't contain linux separator (slash).
  {
    // _link.LinkPath = "\\??\\r:\\1\\2"; // for debug
    // rar5+ returns kpidSymLink absolute link path with "\??\" prefix.
    // we normalize such prefix:
    if (_link.LinkPath.IsPrefixedBy(STRING_PATH_SEPARATOR "??" STRING_PATH_SEPARATOR))
    {
      _link.isRelative = false;
       // we normalize prefix from "\??\" to "\\?\":
      _link.LinkPath.ReplaceOneCharAtPos(1, WCHAR_PATH_SEPARATOR);
      _link.isWindowsPath = true;
      if (_link.LinkPath.IsPrefixedBy_Ascii_NoCase(
          STRING_PATH_SEPARATOR
          STRING_PATH_SEPARATOR "?"
          STRING_PATH_SEPARATOR "UNC"
          STRING_PATH_SEPARATOR))
      {
         // we normalize prefix from "\\?\UNC\path" to "\\path":
        _link.LinkPath.DeleteFrontal(6);
        _link.LinkPath.ReplaceOneCharAtPos(0, WCHAR_PATH_SEPARATOR);
      }
      else
      {
        const unsigned k_prefix_Size = 4;
        if (NName::IsDrivePath(_link.LinkPath.Ptr(k_prefix_Size)))
          _link.LinkPath.DeleteFrontal(k_prefix_Size);
      }
    }
  }
  _link.Normalize_to_RelativeSafe(_removePathParts);
  return S_OK;
}

#endif // SUPPORT_LINKS


#ifndef _WIN32

static HRESULT GetOwner(IInArchive *archive,
    UInt32 index, UInt32 pidName, UInt32 pidId, CProcessedFileInfo::COwnerInfo &res)
{
  {
    NWindows::NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, pidId, &prop))
    if (prop.vt == VT_UI4)
    {
      res.Id_Defined = true;
      res.Id = prop.ulVal;
      // res.Id++; // for debug
      // if (pidId == kpidGroupId) res.Id += 7; // for debug
      // res.Id = 0; // for debug
    }
    else if (prop.vt != VT_EMPTY)
      return E_INVALIDARG;
  }
  {
    NWindows::NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, pidName, &prop))
    if (prop.vt == VT_BSTR)
    {
      const UString s = prop.bstrVal;
      ConvertUnicodeToUTF8(s, res.Name);
    }
    else if (prop.vt == VT_UI4)
    {
      res.Id_Defined = true;
      res.Id = prop.ulVal;
    }
    else if (prop.vt != VT_EMPTY)
      return E_INVALIDARG;
  }
  return S_OK;
}

#endif


HRESULT CArchiveExtractCallback::Read_fi_Props()
{
  IInArchive * const archive = _arc->Archive;
  const UInt32 index = _index;

  _fi.Attrib_Defined = false;

 #ifndef _WIN32
  _fi.Owner.Clear();
  _fi.Group.Clear();
 #endif

  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidPosixAttrib, &prop))
    if (prop.vt == VT_UI4)
    {
      _fi.SetFromPosixAttrib(prop.ulVal);
    }
    else if (prop.vt != VT_EMPTY)
      return E_FAIL;
  }
  
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidAttrib, &prop))
    if (prop.vt == VT_UI4)
    {
      _fi.Attrib = prop.ulVal;
      _fi.Attrib_Defined = true;
    }
    else if (prop.vt != VT_EMPTY)
      return E_FAIL;
  }

  RINOK(GetTime(index, kpidCTime, _fi.CTime))
  RINOK(GetTime(index, kpidATime, _fi.ATime))
  RINOK(GetTime(index, kpidMTime, _fi.MTime))

 #ifndef _WIN32
  if (_ntOptions.ExtractOwner)
  {
    // SendMessageError_with_LastError("_ntOptions.ExtractOwner", _diskFilePath);
    GetOwner(archive, index, kpidUser, kpidUserId, _fi.Owner);
    GetOwner(archive, index, kpidGroup, kpidGroupId, _fi.Group);
  }
 #endif

  return S_OK;
}



void CArchiveExtractCallback::CorrectPathParts()
{
  UStringVector &pathParts = _item.PathParts;

  #ifdef SUPPORT_ALT_STREAMS
  if (!_item.IsAltStream
      || !pathParts.IsEmpty()
      || !(_removePartsForAltStreams || _pathMode == NExtract::NPathMode::kNoPathsAlt))
  #endif
    Correct_FsPath(_pathMode == NExtract::NPathMode::kAbsPaths, _keepAndReplaceEmptyDirPrefixes, pathParts, _item.MainIsDir);
  
  #ifdef SUPPORT_ALT_STREAMS
    
  if (_item.IsAltStream)
  {
    UString s (_item.AltStreamName);
    Correct_AltStream_Name(s);
    bool needColon = true;
    
    if (pathParts.IsEmpty())
    {
      pathParts.AddNew();
      if (_removePartsForAltStreams || _pathMode == NExtract::NPathMode::kNoPathsAlt)
        needColon = false;
    }
    #ifdef _WIN32
    else if (_pathMode == NExtract::NPathMode::kAbsPaths &&
        NWildcard::GetNumPrefixParts_if_DrivePath(pathParts) == pathParts.Size())
      pathParts.AddNew();
    #endif
    
    UString &name = pathParts.Back();
    if (needColon)
      name.Add_Char((char)(_ntOptions.ReplaceColonForAltStream ? '_' : ':'));
    name += s;
  }
    
  #endif // SUPPORT_ALT_STREAMS
}


static void GetFiTimesCAM(const CProcessedFileInfo &fi, CFiTimesCAM &pt, const CArc &arc)
{
  pt.CTime_Defined = false;
  pt.ATime_Defined = false;
  pt.MTime_Defined = false;

  // if (Write_MTime)
  {
    if (fi.MTime.Def)
    {
      fi.MTime.Write_To_FiTime(pt.MTime);
      pt.MTime_Defined = true;
    }
    else if (arc.MTime.Def)
    {
      arc.MTime.Write_To_FiTime(pt.MTime);
      pt.MTime_Defined = true;
    }
  }

  if (/* Write_CTime && */ fi.CTime.Def)
  {
    fi.CTime.Write_To_FiTime(pt.CTime);
    pt.CTime_Defined = true;
  }

  if (/* Write_ATime && */ fi.ATime.Def)
  {
    fi.ATime.Write_To_FiTime(pt.ATime);
    pt.ATime_Defined = true;
  }
}


void CArchiveExtractCallback::CreateFolders()
{
  // 21.04 : we don't change original (_item.PathParts) here
  UStringVector pathParts = _item.PathParts;

  bool isFinal = true;
  // bool is_DirOp = false;
  if (!pathParts.IsEmpty())
  {
    /* v23: if we extract symlink, and we know that it links to dir:
        Linux:   we don't create dir item (symlink_from_path) here.
        Windows: SetReparseData() will create dir item, if it doesn't exist,
                 but if we create dir item here, it's not problem. */
    if (!_item.IsDir
        #ifdef SUPPORT_LINKS
        // #ifndef WIN32
          || !_link.LinkPath.IsEmpty()
        // #endif
        #endif
       )
    {
      pathParts.DeleteBack();
      isFinal = false; // last path part was excluded
    }
    // else is_DirOp = true;
  }
    
  if (pathParts.IsEmpty())
  {
    /* if (_some_pathParts_wereRemoved && Is_elimPrefix_Mode),
       then we can have empty pathParts() here for root folder.
       v24.00: fixed: we set timestamps for such folder still.
    */
    if (!_some_pathParts_wereRemoved ||
        !Is_elimPrefix_Mode)
      return;
    // return; // ignore empty paths case
  }
  /*
  if (is_DirOp)
  {
    RINOK(PrepareOperation(NArchive::NExtract::NAskMode::kExtract))
    _op_WasReported = true;
  }
  */

  FString fullPathNew;
  CreateComplexDirectory(pathParts, isFinal, fullPathNew);

  /*
  if (is_DirOp)
  {
    RINOK(SetOperationResult(
        // _itemFailure ? NArchive::NExtract::NOperationResult::kDataError :
        NArchive::NExtract::NOperationResult::kOK
        ))
  }
  */
  
  if (!_item.IsDir)
    return;
  if (fullPathNew.IsEmpty())
    return;

  if (_itemFailure)
    return;

  CDirPathTime pt;
  GetFiTimesCAM(_fi, pt, *_arc);
 
  if (pt.IsSomeTimeDefined())
  {
    pt.Path = fullPathNew;
    pt.SetDirTime_to_FS_2();
    _extractedFolders.Add(pt);
  }
}



/*
  CheckExistFile(fullProcessedPath)
    it can change: fullProcessedPath, _isRenamed, _overwriteMode
  (needExit = true) means that we must exit GetStream() even for S_OK result.
*/

HRESULT CArchiveExtractCallback::CheckExistFile(FString &fullProcessedPath, bool &needExit)
{
  needExit = true; // it was set already before

  NFind::CFileInfo fileInfo;

  if (fileInfo.Find(fullProcessedPath))
  {
    if (_overwriteMode == NExtract::NOverwriteMode::kSkip)
      return S_OK;
    
    if (_overwriteMode == NExtract::NOverwriteMode::kAsk)
    {
      const int slashPos = fullProcessedPath.ReverseFind_PathSepar();
      const FString realFullProcessedPath = fullProcessedPath.Left((unsigned)(slashPos + 1)) + fileInfo.Name;
  
      /* (fileInfo) can be symbolic link.
         we can show final file properties here. */

      FILETIME ft1;
      FiTime_To_FILETIME(fileInfo.MTime, ft1);

      Int32 overwriteResult;
      RINOK(_extractCallback2->AskOverwrite(
          fs2us(realFullProcessedPath), &ft1, &fileInfo.Size, _item.Path,
          _fi.MTime.Def ? &_fi.MTime.FT : NULL,
          _curSize_Defined ? &_curSize : NULL,
          &overwriteResult))
          
      switch (overwriteResult)
      {
        case NOverwriteAnswer::kCancel:
          return E_ABORT;
        case NOverwriteAnswer::kNo:
          return S_OK;
        case NOverwriteAnswer::kNoToAll:
          _overwriteMode = NExtract::NOverwriteMode::kSkip;
          return S_OK;
    
        case NOverwriteAnswer::kYes:
          break;
        case NOverwriteAnswer::kYesToAll:
          _overwriteMode = NExtract::NOverwriteMode::kOverwrite;
          break;
        case NOverwriteAnswer::kAutoRename:
          _overwriteMode = NExtract::NOverwriteMode::kRename;
          break;
        default:
          return E_FAIL;
      }
    } // NExtract::NOverwriteMode::kAsk

    if (_overwriteMode == NExtract::NOverwriteMode::kRename)
    {
      if (!AutoRenamePath(fullProcessedPath))
      {
        RINOK(SendMessageError(kCantAutoRename, fullProcessedPath))
        return E_FAIL;
      }
      _isRenamed = true;
    }
    else if (_overwriteMode == NExtract::NOverwriteMode::kRenameExisting)
    {
      FString existPath (fullProcessedPath);
      if (!AutoRenamePath(existPath))
      {
        RINOK(SendMessageError(kCantAutoRename, fullProcessedPath))
        return E_FAIL;
      }
      // MyMoveFile can rename folders. So it's OK to use it for folders too
      if (!MyMoveFile(fullProcessedPath, existPath))
      {
        RINOK(SendMessageError2_with_LastError(kCantRenameFile, existPath, fullProcessedPath))
        return E_FAIL;
      }
    }
    else // not Rename*
    {
      if (fileInfo.IsDir())
      {
        // do we need to delete all files in folder?
        if (!RemoveDir(fullProcessedPath))
        {
          RINOK(SendMessageError_with_LastError(kCantDeleteOutputDir, fullProcessedPath))
          return S_OK;
        }
      }
      else // fileInfo is not Dir
      {
        if (NFind::DoesFileExist_Raw(fullProcessedPath))
          if (!DeleteFileAlways(fullProcessedPath))
            if (GetLastError() != ERROR_FILE_NOT_FOUND) // check it in linux
            {
              RINOK(SendMessageError_with_LastError(kCantDeleteOutputFile, fullProcessedPath))
              return S_OK;
              // return E_FAIL;
            }
      } // fileInfo is not Dir
    } // not Rename*
  }
  else // not Find(fullProcessedPath)
  {
    #if defined(_WIN32) && !defined(UNDER_CE)
    // we need to clear READ-ONLY of parent before creating alt stream
    const int colonPos = NName::FindAltStreamColon(fullProcessedPath);
    if (colonPos >= 0 && fullProcessedPath[(unsigned)colonPos + 1] != 0)
    {
      FString parentFsPath (fullProcessedPath);
      parentFsPath.DeleteFrom((unsigned)colonPos);
      NFind::CFileInfo parentFi;
      if (parentFi.Find(parentFsPath))
      {
        if (parentFi.IsReadOnly())
        {
          _altStream_NeedRestore_Attrib_for_parentFsPath = parentFsPath;
          _altStream_NeedRestore_AttribVal = parentFi.Attrib;
          SetFileAttrib(parentFsPath, parentFi.Attrib & ~(DWORD)FILE_ATTRIBUTE_READONLY);
        }
      }
    }
    #endif // defined(_WIN32) && !defined(UNDER_CE)
  }
  
  needExit = false;
  return S_OK;
}



/*
return:
  needExit = false: caller will     use (outStreamLoc) and _hashStreamSpec
  needExit = true : caller will not use (outStreamLoc) and _hashStreamSpec.
*/
HRESULT CArchiveExtractCallback::GetExtractStream(CMyComPtr<ISequentialOutStream> &outStreamLoc, bool &needExit)
{
  needExit = true;
    
  RINOK(Read_fi_Props())

  #ifdef SUPPORT_LINKS
  IInArchive * const archive = _arc->Archive;
  #endif

  const UInt32 index = _index;

  bool isAnti = false;
  RINOK(_arc->IsItem_Anti(index, isAnti))

  CorrectPathParts();
  UString processedPath (MakePathFromParts(_item.PathParts));
  
  if (!isAnti)
  {
    // 21.04: CreateFolders doesn't change (_item.PathParts)
    CreateFolders();
  }
  
  FString fullProcessedPath (us2fs(processedPath));
  if (_pathMode != NExtract::NPathMode::kAbsPaths
      || !NName::IsAbsolutePath(processedPath))
  {
    fullProcessedPath = MakePath_from_2_Parts(_dirPathPrefix, fullProcessedPath);
  }

  #ifdef SUPPORT_ALT_STREAMS
  if (_item.IsAltStream && _item.ParentIndex != (UInt32)(Int32)-1)
  {
    const int renIndex = _renamedFiles.FindInSorted(CIndexToPathPair(_item.ParentIndex));
    if (renIndex != -1)
    {
      const CIndexToPathPair &pair = _renamedFiles[(unsigned)renIndex];
      fullProcessedPath = pair.Path;
      fullProcessedPath.Add_Colon();
      UString s (_item.AltStreamName);
      Correct_AltStream_Name(s);
      fullProcessedPath += us2fs(s);
    }
  }
  #endif // SUPPORT_ALT_STREAMS

  if (_item.IsDir)
  {
    _diskFilePath = fullProcessedPath;
    if (isAnti)
      RemoveDir(_diskFilePath);
    #ifdef SUPPORT_LINKS
    if (_link.LinkPath.IsEmpty())
    #endif
    {
      if (!isAnti)
        SetAttrib();
      return S_OK;
    }
  }
  else if (!_isSplit)
  {
    RINOK(CheckExistFile(fullProcessedPath, needExit))
    if (needExit)
      return S_OK;
    needExit = true;
  }
  
  _diskFilePath = fullProcessedPath;
    

  if (isAnti)
  {
    needExit = false;
    return S_OK;
  }

  // not anti

  #ifdef SUPPORT_LINKS
  
  if (!_link.LinkPath.IsEmpty())
  {
    #ifndef UNDER_CE
    {
      bool linkWasSet = false;
      RINOK(SetLink(fullProcessedPath, _link, linkWasSet))
/*
      // we don't set attributes for placeholder.
      if (linkWasSet)
      {
        _isSymLinkCreated = _link.Is_AnySymLink();
        SetAttrib();
        // printf("\nlinkWasSet %s\n", GetAnsiString(_diskFilePath));
      }
*/
    }
    #endif // UNDER_CE

    // if (_copyFile_Path.IsEmpty())
    {
      needExit = false;
      return S_OK;
    }
  }

  if (!_hardLinks.IDs.IsEmpty() && !_item.IsAltStream && !_item.IsDir)
  {
    CHardLinkNode h;
    bool defined;
    RINOK(Archive_Get_HardLinkNode(archive, index, h, defined))
    if (defined)
    {
      const int linkIndex = _hardLinks.IDs.FindInSorted2(h);
      if (linkIndex != -1)
      {
        FString &hl = _hardLinks.Links[(unsigned)linkIndex];
        if (hl.IsEmpty())
          hl = fullProcessedPath;
        else
        {
          bool link_was_Created = false;
          RINOK(CreateHardLink2(fullProcessedPath, hl, link_was_Created))
          if (!link_was_Created)
            return S_OK;
          // printf("\nHard linkWasSet Archive_Get_HardLinkNode %s\n", GetAnsiString(_diskFilePath));
          // _needSetAttrib = true; // do we need to set attribute ?
          SetAttrib();
          /* if we set (needExit = false) here, _hashStreamSpec will be used,
             and hash will be calulated for all hard links files (it's slower).
             But "Test" operation also calculates hashes.
          */
          needExit = false;
          return S_OK;
        }
      }
    }
  }
  
  #endif // SUPPORT_LINKS


  // ---------- CREATE WRITE FILE -----

  _outFileStreamSpec = new COutFileStream;
  CMyComPtr<IOutStream> outFileStream_Loc(_outFileStreamSpec);
  
  if (!_outFileStreamSpec->Create_ALWAYS_or_Open_ALWAYS(fullProcessedPath, !_isSplit))
  {
    // if (::GetLastError() != ERROR_FILE_EXISTS || !isSplit)
    {
      RINOK(SendMessageError_with_LastError(kCantOpenOutFile, fullProcessedPath))
      return S_OK;
    }
  }
  
  _needSetAttrib = true;

  bool is_SymLink_in_Data = false;

  if (_curSize_Defined && _curSize && _curSize < k_LinkDataSize_LIMIT)
  {
    if (_fi.IsLinuxSymLink())
    {
      is_SymLink_in_Data = true;
      _is_SymLink_in_Data_Linux = true;
    }
    else if (_fi.IsReparse())
    {
      is_SymLink_in_Data = true;
      _is_SymLink_in_Data_Linux = false;
    }
  }

  if (is_SymLink_in_Data)
  {
    _outMemBuf.Alloc((size_t)_curSize);
    _bufPtrSeqOutStream_Spec = new CBufPtrSeqOutStream;
    _bufPtrSeqOutStream = _bufPtrSeqOutStream_Spec;
    _bufPtrSeqOutStream_Spec->Init(_outMemBuf, _outMemBuf.Size());
    outStreamLoc = _bufPtrSeqOutStream;
  }
  else // not reparse
  {
    if (_ntOptions.PreAllocateOutFile && !_isSplit && _curSize_Defined && _curSize > (1 << 12))
    {
      // UInt64 ticks = GetCpuTicks();
      _fileLength_that_WasSet = _curSize;
      bool res = _outFileStreamSpec->File.SetLength(_curSize);
      _fileLength_WasSet = res;
      
      // ticks = GetCpuTicks() - ticks;
      // printf("\nticks = %10d\n", (unsigned)ticks);
      if (!res)
      {
        RINOK(SendMessageError_with_LastError(kCantSetFileLen, fullProcessedPath))
      }
      
      /*
      _outFileStreamSpec->File.Close();
      ticks = GetCpuTicks() - ticks;
      printf("\nticks = %10d\n", (unsigned)ticks);
      return S_FALSE;
      */
      
      /*
      File.SetLength() on FAT (xp64): is fast, but then File.Close() can be slow,
      if we don't write any data.
      File.SetLength() for remote share file (exFAT) can be slow in some cases,
      and the Windows can return "network error" after 1 minute,
      while remote file still can grow.
      We need some way to detect such bad cases and disable PreAllocateOutFile mode.
      */
      
      res = _outFileStreamSpec->SeekToBegin_bool();
      if (!res)
      {
        RINOK(SendMessageError_with_LastError("Cannot seek to begin of file", fullProcessedPath))
      }
    } // PreAllocateOutFile
    
    #ifdef SUPPORT_ALT_STREAMS
    if (_isRenamed && !_item.IsAltStream)
    {
      CIndexToPathPair pair(index, fullProcessedPath);
      unsigned oldSize = _renamedFiles.Size();
      unsigned insertIndex = _renamedFiles.AddToUniqueSorted(pair);
      if (oldSize == _renamedFiles.Size())
        _renamedFiles[insertIndex].Path = fullProcessedPath;
    }
    #endif // SUPPORT_ALT_STREAMS
    
    if (_isSplit)
    {
      RINOK(outFileStream_Loc->Seek((Int64)_position, STREAM_SEEK_SET, NULL))
    }
    outStreamLoc = outFileStream_Loc;
  } // if not reparse

  _outFileStream = outFileStream_Loc;
      
  needExit = false;
  return S_OK;
}



HRESULT CArchiveExtractCallback::GetItem(UInt32 index)
{
  #ifndef Z7_SFX
  _item._use_baseParentFolder_mode = _use_baseParentFolder_mode;
  if (_use_baseParentFolder_mode)
  {
    _item._baseParentFolder = (int)_baseParentFolder;
    if (_pathMode == NExtract::NPathMode::kFullPaths ||
        _pathMode == NExtract::NPathMode::kAbsPaths)
      _item._baseParentFolder = -1;
  }
  #endif // Z7_SFX

  #ifdef SUPPORT_ALT_STREAMS
  _item.WriteToAltStreamIfColon = _ntOptions.WriteToAltStreamIfColon;
  #endif

  return _arc->GetItem(index, _item);
}


Z7_COM7F_IMF(CArchiveExtractCallback::GetStream(UInt32 index, ISequentialOutStream **outStream, Int32 askExtractMode))
{
  COM_TRY_BEGIN

  *outStream = NULL;

  #ifndef Z7_SFX
  if (_hashStream)
    _hashStreamSpec->ReleaseStream();
  _hashStreamWasUsed = false;
  #endif

  _outFileStream.Release();
  _bufPtrSeqOutStream.Release();

  _encrypted = false;
  _isSplit = false;
  _curSize_Defined = false;
  _fileLength_WasSet = false;
  _isRenamed = false;
  // _fi.Clear();
  _extractMode = false;
  _is_SymLink_in_Data_Linux = false;
  _needSetAttrib = false;
  _isSymLinkCreated = false;
  _itemFailure = false;
  _some_pathParts_wereRemoved = false;
  // _op_WasReported = false;

  _position = 0;
  _curSize = 0;
  _fileLength_that_WasSet = 0;
  _index = index;

#if defined(_WIN32) && !defined(UNDER_CE)
  _altStream_NeedRestore_AttribVal = 0;
  _altStream_NeedRestore_Attrib_for_parentFsPath.Empty();
#endif

  _diskFilePath.Empty();

  #ifdef SUPPORT_LINKS
  // _copyFile_Path.Empty();
  _link.Clear();
  #endif


  switch (askExtractMode)
  {
    case NArchive::NExtract::NAskMode::kExtract:
      if (_testMode)
      {
        // askExtractMode = NArchive::NExtract::NAskMode::kTest;
      }
      else
        _extractMode = true;
      break;
    default: break;
  }


  IInArchive * const archive = _arc->Archive;

  RINOK(GetItem(index))

  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetProperty(index, kpidPosition, &prop))
    if (prop.vt != VT_EMPTY)
    {
      if (prop.vt != VT_UI8)
        return E_FAIL;
      _position = prop.uhVal.QuadPart;
      _isSplit = true;
    }
  }

#ifdef SUPPORT_LINKS
  RINOK(ReadLink())
#endif
  
  RINOK(Archive_GetItemBoolProp(archive, index, kpidEncrypted, _encrypted))

  RINOK(GetUnpackSize())

  #ifdef SUPPORT_ALT_STREAMS
  if (!_ntOptions.AltStreams.Val && _item.IsAltStream)
    return S_OK;
  #endif // SUPPORT_ALT_STREAMS

  // we can change (_item.PathParts) in this function
  UStringVector &pathParts = _item.PathParts;

  if (_wildcardCensor)
  {
    if (!CensorNode_CheckPath(*_wildcardCensor, _item))
      return S_OK;
  }

#if defined(_WIN32) && !defined(UNDER_CE) && !defined(Z7_SFX)
  if (askExtractMode == NArchive::NExtract::NAskMode::kExtract
      && !_testMode
      && _item.IsAltStream
      && ZoneBuf.Size() != 0
      && Is_ZoneId_StreamName(_item.AltStreamName))
    if (ZoneMode != NExtract::NZoneIdMode::kOffice
        || _item.PathParts.IsEmpty()
        || FindExt2(kOfficeExtensions, _item.PathParts.Back()))
      return S_OK;
#endif


  #ifndef Z7_SFX
  if (_use_baseParentFolder_mode)
  {
    if (!pathParts.IsEmpty())
    {
      unsigned numRemovePathParts = 0;
      
      #ifdef SUPPORT_ALT_STREAMS
      if (_pathMode == NExtract::NPathMode::kNoPathsAlt && _item.IsAltStream)
        numRemovePathParts = pathParts.Size();
      else
      #endif
      if (_pathMode == NExtract::NPathMode::kNoPaths ||
          _pathMode == NExtract::NPathMode::kNoPathsAlt)
        numRemovePathParts = pathParts.Size() - 1;
      pathParts.DeleteFrontal(numRemovePathParts);
    }
  }
  else
  #endif // Z7_SFX
  {
    if (pathParts.IsEmpty())
    {
      if (_item.IsDir)
        return S_OK;
      /*
      #ifdef SUPPORT_ALT_STREAMS
      if (!_item.IsAltStream)
      #endif
        return E_FAIL;
      */
    }

    unsigned numRemovePathParts = 0;
    
    switch ((int)_pathMode)
    {
      case NExtract::NPathMode::kFullPaths:
      case NExtract::NPathMode::kCurPaths:
      {
        if (_removePathParts.IsEmpty())
          break;
        bool badPrefix = false;
        
        if (pathParts.Size() < _removePathParts.Size())
          badPrefix = true;
        else
        {
          if (pathParts.Size() == _removePathParts.Size())
          {
            if (_removePartsForAltStreams)
            {
              #ifdef SUPPORT_ALT_STREAMS
              if (!_item.IsAltStream)
              #endif
                badPrefix = true;
            }
            else
            {
              if (!_item.MainIsDir)
                badPrefix = true;
            }
          }
          
          if (!badPrefix)
          FOR_VECTOR (i, _removePathParts)
          {
            if (CompareFileNames(_removePathParts[i], pathParts[i]) != 0)
            {
              badPrefix = true;
              break;
            }
          }
        }
        
        if (badPrefix)
        {
          if (askExtractMode == NArchive::NExtract::NAskMode::kExtract && !_testMode)
            return E_FAIL;
        }
        else
        {
          numRemovePathParts = _removePathParts.Size();
          _some_pathParts_wereRemoved = true;
        }
        break;
      }
      
      case NExtract::NPathMode::kNoPaths:
      {
        if (!pathParts.IsEmpty())
          numRemovePathParts = pathParts.Size() - 1;
        break;
      }
      case NExtract::NPathMode::kNoPathsAlt:
      {
        #ifdef SUPPORT_ALT_STREAMS
        if (_item.IsAltStream)
          numRemovePathParts = pathParts.Size();
        else
        #endif
        if (!pathParts.IsEmpty())
          numRemovePathParts = pathParts.Size() - 1;
        break;
      }
      case NExtract::NPathMode::kAbsPaths:
      default:
        break;
    }
    
    pathParts.DeleteFrontal(numRemovePathParts);
  }

  
  #ifndef Z7_SFX

  if (ExtractToStreamCallback)
  {
    CMyComPtr2_Create<IGetProp, CGetProp> GetProp;
    GetProp->Arc = _arc;
    GetProp->IndexInArc = index;
    UString name (MakePathFromParts(pathParts));
    // GetProp->BaseName = name;
    #ifdef SUPPORT_ALT_STREAMS
    if (_item.IsAltStream)
    {
      if (!pathParts.IsEmpty() || (!_removePartsForAltStreams && _pathMode != NExtract::NPathMode::kNoPathsAlt))
        name.Add_Colon();
      name += _item.AltStreamName;
    }
    #endif

    return ExtractToStreamCallback->GetStream7(name, BoolToInt(_item.IsDir), outStream, askExtractMode, GetProp);
  }

  #endif // Z7_SFX


  CMyComPtr<ISequentialOutStream> outStreamLoc;

  if (askExtractMode == NArchive::NExtract::NAskMode::kExtract && !_testMode)
  {
    if (_stdOutMode)
      outStreamLoc = new CStdOutFileStream;
    else
    {
      bool needExit = true;
      RINOK(GetExtractStream(outStreamLoc, needExit))
      if (needExit)
        return S_OK;
    }
  }

  #ifndef Z7_SFX
  if (_hashStream)
  {
    if (askExtractMode == NArchive::NExtract::NAskMode::kExtract ||
        askExtractMode == NArchive::NExtract::NAskMode::kTest)
    {
      _hashStreamSpec->SetStream(outStreamLoc);
      outStreamLoc = _hashStream;
      _hashStreamSpec->Init(true);
      _hashStreamWasUsed = true;
    }
  }
  #endif // Z7_SFX

  if (outStreamLoc)
  {
    /*
    #ifdef SUPPORT_LINKS
    if (!_copyFile_Path.IsEmpty())
    {
      RINOK(PrepareOperation(askExtractMode));
      RINOK(MyCopyFile(outStreamLoc));
      return SetOperationResult(NArchive::NExtract::NOperationResult::kOK);
    }
    if (_link.isCopyLink && _testMode)
      return S_OK;
    #endif
    */
    *outStream = outStreamLoc.Detach();
  }
  
  return S_OK;

  COM_TRY_END
}











Z7_COM7F_IMF(CArchiveExtractCallback::PrepareOperation(Int32 askExtractMode))
{
  COM_TRY_BEGIN

  #ifndef Z7_SFX
  // if (!_op_WasReported)
  if (ExtractToStreamCallback)
    return ExtractToStreamCallback->PrepareOperation7(askExtractMode);
  #endif
  
  _extractMode = false;
  
  switch (askExtractMode)
  {
    case NArchive::NExtract::NAskMode::kExtract:
      if (_testMode)
        askExtractMode = NArchive::NExtract::NAskMode::kTest;
      else
        _extractMode = true;
      break;
    default: break;
  }

  // if (_op_WasReported) return S_OK;
  
  return _extractCallback2->PrepareOperation(_item.Path, BoolToInt(_item.IsDir),
      askExtractMode, _isSplit ? &_position: NULL);
  
  COM_TRY_END
}





HRESULT CArchiveExtractCallback::CloseFile()
{
  if (!_outFileStream)
    return S_OK;
  
  HRESULT hres = S_OK;
  
  const UInt64 processedSize = _outFileStreamSpec->ProcessedSize;
  if (_fileLength_WasSet && _fileLength_that_WasSet > processedSize)
  {
    const bool res = _outFileStreamSpec->File.SetLength(processedSize);
    _fileLength_WasSet = res;
    if (!res)
    {
      const HRESULT hres2 = SendMessageError_with_LastError(kCantSetFileLen, us2fs(_item.Path));
      if (hres == S_OK)
        hres = hres2;
    }
  }

  _curSize = processedSize;
  _curSize_Defined = true;

 #if defined(_WIN32) && !defined(UNDER_CE) && !defined(Z7_SFX)
  if (ZoneBuf.Size() != 0
      && !_item.IsAltStream)
  {
    // if (NFind::DoesFileExist_Raw(tempFilePath))
    if (ZoneMode != NExtract::NZoneIdMode::kOffice ||
        FindExt2(kOfficeExtensions, fs2us(_diskFilePath)))
    {
      // we must write zone file before setting of timestamps
      if (!WriteZoneFile_To_BaseFile(_diskFilePath, ZoneBuf))
      {
        // we can't write it in FAT
        // SendMessageError_with_LastError("Can't write Zone.Identifier stream", path);
      }
    }
  }
 #endif

  CFiTimesCAM t;
  GetFiTimesCAM(_fi, t, *_arc);

  // #ifdef _WIN32
  if (t.IsSomeTimeDefined())
    _outFileStreamSpec->SetTime(
        t.CTime_Defined ? &t.CTime : NULL,
        t.ATime_Defined ? &t.ATime : NULL,
        t.MTime_Defined ? &t.MTime : NULL);
  // #endif

  RINOK(_outFileStreamSpec->Close())
  _outFileStream.Release();

#if defined(_WIN32) && !defined(UNDER_CE)
  if (!_altStream_NeedRestore_Attrib_for_parentFsPath.IsEmpty())
  {
    SetFileAttrib(_altStream_NeedRestore_Attrib_for_parentFsPath, _altStream_NeedRestore_AttribVal);
    _altStream_NeedRestore_Attrib_for_parentFsPath.Empty();
  }
#endif

  return hres;
}


#ifdef SUPPORT_LINKS

static bool CheckLinkPath_in_FS_for_pathParts(const FString &path, const UStringVector &v)
{
  FString path2 = path;
  FOR_VECTOR (i, v)
  {
    // if (i == v.Size() - 1) path = path2; // we don't need last part in returned path
    path2 += us2fs(v[i]);
    NFind::CFileInfo fi;
    // printf("\nCheckLinkPath_in_FS_for_pathParts(): %s\n", GetOemString(path2).Ptr());
    if (fi.Find(path2) && fi.IsOsSymLink())
      return false;
    path2.Add_PathSepar();
  }
  return true;
}

/*
link.isRelative / relative_item_PathPrefix
   false        / empty
   true         / item path without last part
*/
static bool CheckLinkPath_in_FS(
    const FString &pathPrefix_in_FS,
    const CPostLink &postLink,
    const UString &relative_item_PathPrefix)
{
  const CLinkInfo &link = postLink.LinkInfo;
  if (postLink.item_PathParts.IsEmpty() || link.LinkPath.IsEmpty())
    return false;
  FString path;
  {
    const UString &s = postLink.item_PathParts[0];
    if (!s.IsEmpty() && !NName::IsAbsolutePath(s))
      path = pathPrefix_in_FS; // item_PathParts is relative. So we use absolutre prefix
  }
  if (!CheckLinkPath_in_FS_for_pathParts(path, postLink.item_PathParts))
    return false;
  path += us2fs(relative_item_PathPrefix);
  UStringVector v;
  SplitPathToParts(link.LinkPath, v);
  // we check target paths:
  return CheckLinkPath_in_FS_for_pathParts(path, v);
}

static const unsigned k_DangLevel_MAX_for_Link_over_Link = 9;

HRESULT CArchiveExtractCallback::CreateHardLink2(
    const FString &newFilePath, const FString &existFilePath, bool &link_was_Created) const
{
  link_was_Created = false;
  if (_ntOptions.SymLinks_DangerousLevel <= k_DangLevel_MAX_for_Link_over_Link)
  {
    NFind::CFileInfo fi;
    if (fi.Find(existFilePath) && fi.IsOsSymLink())
      return SendMessageError2(0, k_HardLink_to_SymLink_Ignored, newFilePath, existFilePath);
  }
  if (!MyCreateHardLink(newFilePath, existFilePath))
    return SendMessageError2_with_LastError(kCantCreateHardLink, newFilePath, existFilePath);
  link_was_Created = true;
  return S_OK;
}



HRESULT CArchiveExtractCallback::SetLink(
    const FString &fullProcessedPath_from,
    const CLinkInfo &link,
    bool &linkWasSet) // placeholder was created
{
  linkWasSet = false;
  if (link.LinkPath.IsEmpty())
    return S_OK;
  if (!_ntOptions.SymLinks.Val && link.Is_AnySymLink())
    return S_OK;
  CPostLink postLink;
  postLink.Index_in_Arc = _index;
  postLink.item_IsDir = _item.IsDir;
  postLink.item_Path = _item.Path;
  postLink.item_PathParts = _item.PathParts;
  postLink.item_FileInfo = _fi;
  postLink.fullProcessedPath_from = fullProcessedPath_from;
  postLink.LinkInfo = link;
  _postLinks.Add(postLink);
  
  // file doesn't exist in most cases. So we don't check for error.
  DeleteLinkFileAlways_or_RemoveEmptyDir(fullProcessedPath_from, false); // checkThatFileIsEmpty = false

  NIO::COutFile outFile;
  if (!outFile.Create_NEW(fullProcessedPath_from))
    return SendMessageError("Cannot create temporary link file", fullProcessedPath_from);
#if 0 // 1 for debug
  // here we can write link path to temporary link file placeholder,
  // but empty placeholder is better, because we don't want to get any non-eampty data instead of link file.
  AString s;
  ConvertUnicodeToUTF8(link.LinkPath, s);
  outFile.WriteFull(s, s.Len());
#endif
  linkWasSet = true;
  return S_OK;
}


// if file/dir is symbolic link it will remove only link itself
HRESULT CArchiveExtractCallback::DeleteLinkFileAlways_or_RemoveEmptyDir(
    const FString &path, bool checkThatFileIsEmpty) const
{
  NFile::NFind::CFileInfo fi;
  if (fi.Find(path)) // followLink = false
  {
    if (fi.IsDir())
    {
      if (RemoveDirAlways_if_Empty(path))
        return S_OK;
    }
    else
    {
      // link file placeholder must be empty
      if (checkThatFileIsEmpty && !fi.IsOsSymLink() && fi.Size != 0)
        return SendMessageError("Temporary link file is not empty", path);
      if (DeleteFileAlways(path))
        return S_OK;
    }
    if (GetLastError() != ERROR_FILE_NOT_FOUND)
      return SendMessageError_with_LastError(
          fi.IsDir() ?
            k_CantDelete_Dir_for_SymLink:
            k_CantDelete_File_for_SymLink,
          path);
  }
  return S_OK;
}


/*
in:
  link.LinkPath : must be relative (non-absolute) path in any case !!!
  link.isRelative / target path that must stored as created link:
       == false   / _dirPathPrefix_Full + link.LinkPath
       == true    / link.LinkPath
*/
static HRESULT SetLink2(const CArchiveExtractCallback &callback,
    const CPostLink &postLink, bool &linkWasSet)
{
  const CLinkInfo &link = postLink.LinkInfo;
  const FString &fullProcessedPath_from = postLink.fullProcessedPath_from; // full file path in FS (fullProcessedPath_from)

  const unsigned level = callback._ntOptions.SymLinks_DangerousLevel;
  if (level < 20)
  {
    /*
    We want to use additional check for links that can link to directory.
      - linux: all symbolic links are files.
      - windows: we can have file/directory symbolic link,
        but file symbolic link works like directory link in windows.
    So we use additional check for all relative links.

    We don't allow decreasing of final level of link.
    So if some another extracted file will use this link,
    then number of real path parts (after link redirection) cannot be
    smaller than number of requested path parts from archive records.
    
    here we check only (link.LinkPath) without (_item.PathParts).
    */
    CLinkLevelsInfo li;
    li.Parse(link.LinkPath, link.Is_WSL());
    bool isDang;
    UString relativePathPrefix;
    if (li.IsAbsolute // unexpected
        || li.ParentDirDots_after_NonParent
        || (level <= 5 && link.isRelative && li.FinalLevel < 1) // final level lower
        || (level <= 5 && link.isRelative && li.LowLevel < 0)   // negative temporary levels
       )
      isDang = true;
    else // if (!isDang)
    {
      UString path;
      if (link.isRelative)
      {
        // item_PathParts : parts that will be created in output folder.
        // we want to get directory prefix of link item.
        // so we remove file name (last non-empty part) from PathParts:
        UStringVector v = postLink.item_PathParts;
        while (!v.IsEmpty())
        {
          const unsigned len = v.Back().Len();
          v.DeleteBack();
          if (len)
            break;
        }
        path = MakePathFromParts(v);
        NName::NormalizeDirPathPrefix(path);
        relativePathPrefix = path;
      }
      path += link.LinkPath;
      /*
      path is calculated virtual target path of link
      path is relative to root folder of extracted items
      if (!link.isRelative), then (path == link.LinkPath)
      */
      isDang = false;
      if (!IsSafePath(path, link.Is_WSL()))
        isDang = true;
    }
    const char *message = NULL;
    if (isDang)
      message = "Dangerous link path was ignored";
    else if (level <= k_DangLevel_MAX_for_Link_over_Link
        && !CheckLinkPath_in_FS(callback._dirPathPrefix_Full,
            postLink, relativePathPrefix))
      message = "Dangerous link via another link was ignored";
    if (message)
       return callback.SendMessageError2(0, // errorCode
            message, us2fs(postLink.item_Path), us2fs(link.LinkPath));
  }

  FString target; // target path that will be stored to link field
  if (link.Is_HardLink() /* || link.IsCopyLink */ || !link.isRelative)
  {
    // isRelative == false
    // all hard links and absolute symbolic links
    // relatPath == link.LinkPath
    // we get absolute link path for target:
    if (!NName::GetFullPath(callback._dirPathPrefix_Full, us2fs(link.LinkPath), target))
      return callback.SendMessageError("Incorrect link path", us2fs(link.LinkPath));
    // (target) is (_dirPathPrefix_Full + relatPath)
  }
  else
  {
    // link.isRelative == true
    // relative symbolic links only
    target = us2fs(link.LinkPath);
  }
  if (target.IsEmpty())
    return callback.SendMessageError("Empty link", fullProcessedPath_from);

  if (link.Is_HardLink() /* || link.IsCopyLink */)
  {
    // if (link.isHardLink)
    {
      RINOK(callback.DeleteLinkFileAlways_or_RemoveEmptyDir(fullProcessedPath_from, true)) // checkThatFileIsEmpty
      {
        // RINOK(SendMessageError_with_LastError(k_Cant_DeleteTempLinkFile, fullProcessedPath_from))
      }
      return callback.CreateHardLink2(fullProcessedPath_from, target, linkWasSet);
      /*
      RINOK(PrepareOperation(NArchive::NExtract::NAskMode::kExtract))
      _op_WasReported = true;
      RINOK(SetOperationResult(NArchive::NExtract::NOperationResult::kOK))
      linkWasSet = true;
      return S_OK;
      */
    }
    /*
    // IsCopyLink
    {
      NFind::CFileInfo fi;
      if (!fi.Find(target))
      {
        RINOK(SendMessageError2("Cannot find the file for copying", target, fullProcessedPath));
      }
      else
      {
        if (_curSize_Defined && _curSize == fi.Size)
          _copyFile_Path = target;
        else
        {
          RINOK(SendMessageError2("File size collision for file copying", target, fullProcessedPath));
        }
        // RINOK(MyCopyFile(target, fullProcessedPath));
      }
    }
    */
  }

  // is Symbolic link

  /*
  if (_item.IsDir && !isRelative)
  {
    // Windows before Vista doesn't support symbolic links.
    // we could convert such symbolic links to Junction Points
    // isJunction = true;
  }
  */

#ifdef _WIN32
  const bool isDir = (postLink.item_IsDir || link.LinkType == k_LinkType_Junction);
#endif

 
#ifdef _WIN32
  CByteBuffer data;
  // printf("\nFillLinkData(): %s\n", GetOemString(target).Ptr());
  if (link.Is_WSL())
  {
    Convert_WinPath_to_WslLinuxPath(target, !link.isRelative);
    FillLinkData_WslLink(data, fs2us(target));
  }
  else
    FillLinkData_WinLink(data, fs2us(target), link.LinkType != k_LinkType_Junction);
  if (data.Size() == 0)
    return callback.SendMessageError("Cannot fill link data", us2fs(postLink.item_Path));
  /*
  if (NtReparse_Size != data.Size() || memcmp(NtReparse_Data, data, data.Size()) != 0)
    SendMessageError("reconstructed Reparse is different", fs2us(target));
  */
  {
    // we check that reparse data is correct, but we ignore attr.MinorError.
    CReparseAttr attr;
    if (!attr.Parse(data, data.Size()))
      return callback.SendMessageError("Internal error for symbolic link file", us2fs(postLink.item_Path));
  }
#endif

  RINOK(callback.DeleteLinkFileAlways_or_RemoveEmptyDir(fullProcessedPath_from, true)) // checkThatFileIsEmpty
#ifdef _WIN32
  if (!NFile::NIO::SetReparseData(fullProcessedPath_from, isDir, data, (DWORD)data.Size()))
#else // ! _WIN32
  if (!NFile::NIO::SetSymLink(fullProcessedPath_from, target))
#endif // ! _WIN32
  {
    return callback.SendMessageError_with_LastError(kCantCreateSymLink, fullProcessedPath_from);
  }
  linkWasSet = true;
  return S_OK;
}



bool CLinkInfo::Parse_from_WindowsReparseData(const Byte *data, size_t dataSize)
{
  CReparseAttr reparse;
  if (!reparse.Parse(data, dataSize))
    return false;
  // const AString s = GetAnsiString(LinkPath);
  // printf("\nlinkPath: %s\n", s.Ptr());
  LinkPath = reparse.GetPath();
  if (reparse.IsSymLink_WSL())
  {
    LinkType = k_LinkType_WSL;
    isRelative = reparse.IsRelative_WSL(); // detected from LinkPath[0]
    // LinkPath is original raw name converted to UString from AString
    // Linux separator '/' is expected here.
    REPLACE_SLASHES_from_Linux_to_Sys(LinkPath)
  }
  else
  {
    LinkType = reparse.IsMountPoint() ? k_LinkType_Junction : k_LinkType_PureSymLink;
    isRelative = reparse.IsRelative_Win(); // detected by (Flags == Z7_WIN_SYMLINK_FLAG_RELATIVE)
    isWindowsPath = true;
    // LinkPath is original windows link path from raparse data with \??\ prefix removed.
    // windows '\\' separator is expected here.
    // linux '/' separator is not expected here.
    // we translate both types of separators to system separator.
    LinkPath.Replace(
#if WCHAR_PATH_SEPARATOR == L'\\'
        L'/'
#else
        L'\\'
#endif
        , WCHAR_PATH_SEPARATOR);
  }
  // (LinkPath) uses system path separator.
  // windows: (LinkPath) doesn't contain linux separator (slash).
  return true;
}


bool CLinkInfo::Parse_from_LinuxData(const Byte *data, size_t dataSize)
{
  // Clear(); // *this object was cleared by constructor already.
  LinkType = k_LinkType_PureSymLink;
  AString utf;
  if (dataSize >= k_LinkDataSize_LIMIT)
    return false;
  utf.SetFrom_CalcLen((const char *)data, (unsigned)dataSize);
  UString u;
  if (!ConvertUTF8ToUnicode(utf, u))
    return false;
  if (u.IsEmpty())
    return false;
  const wchar_t c = u[0];
  isRelative = (c != L'/');
  // linux path separator is expected
  REPLACE_SLASHES_from_Linux_to_Sys(u)
  LinkPath = u;
  // (LinkPath) uses system path separator.
  // windows: (LinkPath) doesn't contain linux separator (slash).
  return true;
}
    

// in/out:          (LinkPath) uses system path separator
// in/out: windows: (LinkPath) doesn't contain linux separator (slash).
// out: (LinkPath) is relative path, and LinkPath[0] is not path separator
// out: isRelative changed to false, if any prefix was removed.
// note: absolute windows links "c:\" to root will be reduced to empty string:
void CLinkInfo::Remove_AbsPathPrefixes()
{
  while (!LinkPath.IsEmpty())
  {
    unsigned n = 0;
    if (!Is_WSL())
    {
      n =
#ifndef _WIN32
      isWindowsPath ?
        NName::GetRootPrefixSize_WINDOWS(LinkPath) :
#endif
        NName::GetRootPrefixSize(LinkPath);
/*
      // "c:path" will be ignored later as "Dangerous absolute path"
      // so check is not required
      if (n == 0
#ifndef _WIN32
          && isWindowsPath
#endif
          && NName::IsDrivePath2(LinkPath))
        n = 2;
*/
    }
    if (n == 0)
    {
      if (!IS_PATH_SEPAR(LinkPath[0]))
        break;
      n = 1;
    }
    isRelative = false; // (LinkPath) will be treated as relative to root folder of archive
    LinkPath.DeleteFrontal(n);
  }
}


/*
  it removes redundant separators, if there are double separators,
  but it keeps double separators at start of string //name/.
  in/out:    system path separator is used
    windows: slash character (linux separator) is not treated as separator
    windows: (path) doesn't contain linux separator (slash).
*/
static void RemoveRedundantPathSeparators(UString &path)
{
  wchar_t *dest = path.GetBuf();
  const wchar_t * const start = dest;
  const wchar_t *src = dest;
  for (;;)
  {
    wchar_t c = *src++;
    if (c == 0)
      break;
    // if (IS_PATH_SEPAR(c)) // for Windows: we can change (/) to (\).
    if (c == WCHAR_PATH_SEPARATOR)
    {
      if (dest - start >= 2 && dest[-1] == WCHAR_PATH_SEPARATOR)
        continue;
      // c = WCHAR_PATH_SEPARATOR; // for Windows: we can change (/) to (\).
    }
    *dest++ = c;
  }
  *dest = 0;
  path.ReleaseBuf_SetLen((unsigned)(dest - path.Ptr()));
}


// in/out: (LinkPath) uses system path separator
// in/out: windows: (LinkPath) doesn't contain linux separator (slash).
// out: (LinkPath) is relative path, and LinkPath[0] is not path separator
void CLinkInfo::Normalize_to_RelativeSafe(UStringVector &removePathParts)
{
  // We WILL NOT WRITE original absolute link path from archive to filesystem.
  // So here we remove all root prefixes from (LinkPath).
  // If we see any absolute root prefix, then we suppose that this prefix is virtual prefix
  // that shows that link is relative to root folder of archive
  RemoveRedundantPathSeparators(LinkPath);
  // LinkPath = "\\\\?\\r:test\\test2"; // for debug
  Remove_AbsPathPrefixes();
  // (LinkPath) now is relative:
  //  if (isRelative == false), then (LinkPath) is relative to root folder of archive
  //  if (isRelative == true ), then (LinkPath) is relative to current item
  if (LinkPath.IsEmpty() || isRelative || removePathParts.Size() == 0)
    return;

  // if LinkPath is prefixed by _removePathParts, we remove these paths
  UStringVector pathParts;
  SplitPathToParts(LinkPath, pathParts);
  bool badPrefix = false;
  {
    FOR_VECTOR (i, removePathParts)
    {
      if (i >= pathParts.Size()
        || CompareFileNames(removePathParts[i], pathParts[i]) != 0)
      {
        badPrefix = true;
        break;
      }
    }
  }
  if (!badPrefix)
    pathParts.DeleteFrontal(removePathParts.Size());
  LinkPath = MakePathFromParts(pathParts);
  Remove_AbsPathPrefixes();
}

#endif // SUPPORT_LINKS


HRESULT CArchiveExtractCallback::CloseReparseAndFile()
{
  HRESULT res = S_OK;

#ifdef SUPPORT_LINKS

  size_t reparseSize = 0;
  bool repraseMode = false;
  bool needSetReparse = false;
  CLinkInfo link;
  
  if (_bufPtrSeqOutStream)
  {
    repraseMode = true;
    reparseSize = _bufPtrSeqOutStream_Spec->GetPos();
    if (_curSize_Defined && reparseSize == _outMemBuf.Size())
    {
      /*
      CReparseAttr reparse;
      DWORD errorCode = 0;
      needSetReparse = reparse.Parse(_outMemBuf, reparseSize, errorCode);
      if (needSetReparse)
      {
        UString LinkPath = reparse.GetPath();
        #ifndef _WIN32
        LinkPath.Replace(L'\\', WCHAR_PATH_SEPARATOR);
        #endif
      }
      */
      needSetReparse = _is_SymLink_in_Data_Linux ?
          link.Parse_from_LinuxData(_outMemBuf, reparseSize) :
          link.Parse_from_WindowsReparseData(_outMemBuf, reparseSize);
      if (!needSetReparse)
        res = SendMessageError_with_LastError("Incorrect reparse stream", us2fs(_item.Path));
      // (link.LinkPath) uses system path separator.
      // windows: (link.LinkPath) doesn't contain linux separator (slash).
    }
    else
    {
      res = SendMessageError_with_LastError("Unknown reparse stream", us2fs(_item.Path));
    }
    if (!needSetReparse && _outFileStream)
    {
      const HRESULT res2 = WriteStream(_outFileStream, _outMemBuf, reparseSize);
      if (res == S_OK)
        res = res2;
    }
    _bufPtrSeqOutStream.Release();
  }

#endif // SUPPORT_LINKS

  const HRESULT res2 = CloseFile();
  if (res == S_OK)
    res = res2;
  RINOK(res)

#ifdef SUPPORT_LINKS
  if (repraseMode)
  {
    _curSize = reparseSize;
    _curSize_Defined = true;
    if (needSetReparse)
    {
      // empty file was created so we must delete it.
      // in Linux   : we must delete empty file before symbolic link creation
      // in Windows : we can create symbolic link even without file deleting
      if (!DeleteFileAlways(_diskFilePath))
      {
        RINOK(SendMessageError_with_LastError("can't delete file", _diskFilePath))
      }
      {
        bool linkWasSet = false;
        // link.LinkPath = "r:\\1\\2"; // for debug
        // link.isJunction = true; // for debug
        link.Normalize_to_RelativeSafe(_removePathParts);
        RINOK(SetLink(_diskFilePath, link, linkWasSet))
/*
        // we don't set attributes for placeholder.
        if (linkWasSet)
          _isSymLinkCreated = true; // link.IsSymLink();
        else
*/
          _needSetAttrib = false;
      }
    }
  }
#endif // SUPPORT_LINKS
  return res;
}


static void SetAttrib_Base(const FString &path, const CProcessedFileInfo &fi,
    const CArchiveExtractCallback &callback)
{
#ifndef _WIN32
  if (fi.Owner.Id_Defined &&
      fi.Group.Id_Defined)
  {
    if (my_chown(path, fi.Owner.Id, fi.Group.Id) != 0)
      callback.SendMessageError_with_LastError("Cannot set owner", path);
  }
#endif

  if (fi.Attrib_Defined)
  {
    // const AString s = GetAnsiString(_diskFilePath);
    // printf("\nSetFileAttrib_PosixHighDetect: %s: hex:%x\n", s.Ptr(), _fi.Attrib);
    if (!SetFileAttrib_PosixHighDetect(path, fi.Attrib))
    {
      // do we need error message here in Windows and in posix?
      callback.SendMessageError_with_LastError("Cannot set file attribute", path);
    }
  }
}

void CArchiveExtractCallback::SetAttrib() const
{
#ifndef _WIN32
  // Linux now doesn't support permissions for symlinks
  if (_isSymLinkCreated)
    return;
#endif

  if (_itemFailure
      || _diskFilePath.IsEmpty()
      || _stdOutMode
      || !_extractMode)
    return;

  SetAttrib_Base(_diskFilePath, _fi, *this);
}


#ifdef Z7_USE_SECURITY_CODE
HRESULT CArchiveExtractCallback::SetSecurityInfo(UInt32 indexInArc, const FString &path) const
{
  if (!_stdOutMode && _extractMode && _ntOptions.NtSecurity.Val && _arc->GetRawProps)
  {
    const void *data;
    UInt32 dataSize;
    UInt32 propType;
    _arc->GetRawProps->GetRawProp(indexInArc, kpidNtSecure, &data, &dataSize, &propType);
    if (dataSize != 0)
    {
      if (propType != NPropDataType::kRaw)
        return E_FAIL;
      if (CheckNtSecure((const Byte *)data, dataSize))
      {
        SECURITY_INFORMATION securInfo = DACL_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | OWNER_SECURITY_INFORMATION;
        if (_saclEnabled)
          securInfo |= SACL_SECURITY_INFORMATION;
        // if (!
        ::SetFileSecurityW(fs2us(path), securInfo, (PSECURITY_DESCRIPTOR)(void *)(const Byte *)(data));
        {
          // RINOK(SendMessageError_with_LastError("SetFileSecurity FAILS", path))
        }
      }
    }
  }
  return S_OK;
}
#endif // Z7_USE_SECURITY_CODE


Z7_COM7F_IMF(CArchiveExtractCallback::SetOperationResult(Int32 opRes))
{
  COM_TRY_BEGIN

  // printf("\nCArchiveExtractCallback::SetOperationResult: %d %s\n", opRes, GetAnsiString(_diskFilePath));

  #ifndef Z7_SFX
  if (ExtractToStreamCallback)
  {
    GetUnpackSize();
    return ExtractToStreamCallback->SetOperationResult8(opRes, BoolToInt(_encrypted), _curSize);
  }
  #endif

  #ifndef Z7_SFX

  if (_hashStreamWasUsed)
  {
    _hashStreamSpec->_hash->Final(_item.IsDir,
        #ifdef SUPPORT_ALT_STREAMS
          _item.IsAltStream
        #else
          false
        #endif
        , _item.Path);
    _curSize = _hashStreamSpec->GetSize();
    _curSize_Defined = true;
    _hashStreamSpec->ReleaseStream();
    _hashStreamWasUsed = false;
  }

  #endif // Z7_SFX

  RINOK(CloseReparseAndFile())
  
#ifdef Z7_USE_SECURITY_CODE
  RINOK(SetSecurityInfo(_index, _diskFilePath))
#endif

  if (!_curSize_Defined)
    GetUnpackSize();
  
  if (_curSize_Defined)
  {
    #ifdef SUPPORT_ALT_STREAMS
    if (_item.IsAltStream)
      AltStreams_UnpackSize += _curSize;
    else
    #endif
      UnpackSize += _curSize;
  }
    
  if (_item.IsDir)
    NumFolders++;
  #ifdef SUPPORT_ALT_STREAMS
  else if (_item.IsAltStream)
    NumAltStreams++;
  #endif
  else
    NumFiles++;

  if (_needSetAttrib)
    SetAttrib();
  
  RINOK(_extractCallback2->SetOperationResult(opRes, BoolToInt(_encrypted)))
  
  return S_OK;
  
  COM_TRY_END
}



Z7_COM7F_IMF(CArchiveExtractCallback::ReportExtractResult(UInt32 indexType, UInt32 index, Int32 opRes))
{
  if (_folderArchiveExtractCallback2)
  {
    bool isEncrypted = false;
    UString s;
    
    if (indexType == NArchive::NEventIndexType::kInArcIndex && index != (UInt32)(Int32)-1)
    {
      CReadArcItem item;
      RINOK(_arc->GetItem(index, item))
      s = item.Path;
      RINOK(Archive_GetItemBoolProp(_arc->Archive, index, kpidEncrypted, isEncrypted))
    }
    else
    {
      s = '#';
      s.Add_UInt32(index);
      // if (indexType == NArchive::NEventIndexType::kBlockIndex) {}
    }
    
    return _folderArchiveExtractCallback2->ReportExtractResult(opRes, isEncrypted, s);
  }

  return S_OK;
}


Z7_COM7F_IMF(CArchiveExtractCallback::CryptoGetTextPassword(BSTR *password))
{
  COM_TRY_BEGIN
  if (!_cryptoGetTextPassword)
  {
    RINOK(_extractCallback2.QueryInterface(IID_ICryptoGetTextPassword,
        &_cryptoGetTextPassword))
  }
  return _cryptoGetTextPassword->CryptoGetTextPassword(password);
  COM_TRY_END
}


#ifndef Z7_SFX

// ---------- HASH functions ----------

FString CArchiveExtractCallback::Hash_GetFullFilePath()
{
  // this function changes _item.PathParts.
  CorrectPathParts();
  const UStringVector &pathParts = _item.PathParts;
  const UString processedPath (MakePathFromParts(pathParts));
  FString fullProcessedPath (us2fs(processedPath));
  if (_pathMode != NExtract::NPathMode::kAbsPaths
      || !NName::IsAbsolutePath(processedPath))
  {
    fullProcessedPath = MakePath_from_2_Parts(
        DirPathPrefix_for_HashFiles,
        // _dirPathPrefix,
        fullProcessedPath);
  }
  return fullProcessedPath;
}


Z7_COM7F_IMF(CArchiveExtractCallback::GetDiskProperty(UInt32 index, PROPID propID, PROPVARIANT *value))
{
  COM_TRY_BEGIN
  NCOM::CPropVariant prop;
  if (propID == kpidSize)
  {
    RINOK(GetItem(index))
    const FString fullProcessedPath = Hash_GetFullFilePath();
    NFile::NFind::CFileInfo fi;
    if (fi.Find_FollowLink(fullProcessedPath))
      if (!fi.IsDir())
        prop = (UInt64)fi.Size;
  }
  prop.Detach(value);
  return S_OK;
  COM_TRY_END
}


Z7_COM7F_IMF(CArchiveExtractCallback::GetStream2(UInt32 index, ISequentialInStream **inStream, UInt32 mode))
{
  COM_TRY_BEGIN
  *inStream = NULL;
  // if (index != _index) return E_FAIL;
  if (mode != NUpdateNotifyOp::kHashRead)
    return E_FAIL;

  RINOK(GetItem(index))
  const FString fullProcessedPath = Hash_GetFullFilePath();

  CInFileStream *inStreamSpec = new CInFileStream;
  CMyComPtr<ISequentialInStream> inStreamRef = inStreamSpec;
  inStreamSpec->Set_PreserveATime(_ntOptions.PreserveATime);
  if (!inStreamSpec->OpenShared(fullProcessedPath, _ntOptions.OpenShareForWrite))
  {
    RINOK(SendMessageError_with_LastError(kCantOpenInFile, fullProcessedPath))
    return S_OK;
  }
  *inStream = inStreamRef.Detach();
  return S_OK;
  COM_TRY_END
}


Z7_COM7F_IMF(CArchiveExtractCallback::ReportOperation(
    UInt32 /* indexType */, UInt32 /* index */, UInt32 /* op */))
{
  // COM_TRY_BEGIN
  return S_OK;
  // COM_TRY_END
}


Z7_COM7F_IMF(CArchiveExtractCallback::RequestMemoryUse(
    UInt32 flags, UInt32 indexType, UInt32 index, const wchar_t *path,
    UInt64 requiredSize, UInt64 *allowedSize, UInt32 *answerFlags))
{
  if ((flags & NRequestMemoryUseFlags::k_IsReport) == 0)
  {
    const UInt64 memLimit = _ntOptions.MemLimit;
    if (memLimit != (UInt64)(Int64)-1)
    {
      // we overwrite allowedSize
      *allowedSize = memLimit;
      if (requiredSize <= memLimit)
      {
        *answerFlags = NRequestMemoryAnswerFlags::k_Allow;
        return S_OK;
      }
      *answerFlags = NRequestMemoryAnswerFlags::k_Limit_Exceeded;
      if (flags & NRequestMemoryUseFlags::k_SkipArc_IsExpected)
        *answerFlags |= NRequestMemoryAnswerFlags::k_SkipArc;
      flags |= NRequestMemoryUseFlags::k_SLimit_Exceeded
            |  NRequestMemoryUseFlags::k_AllowedSize_WasForced;
    }
  }

  if (!_requestMemoryUseCallback)
  {
    _extractCallback2.QueryInterface(IID_IArchiveRequestMemoryUseCallback,
        &_requestMemoryUseCallback);
    if (!_requestMemoryUseCallback)
    {
      // keep default (answerFlags) from caller or (answerFlags) that was set in this function
      return S_OK;
    }
  }

#if 0
  if ((flags & NRequestMemoryUseFlags::k_IsReport) == 0)
  if (requiredSize <= *allowedSize)
  {
    // it's expected, that *answerFlags was set to NRequestMemoryAnswerFlags::k_Allow already,
    // because it's default answer for (requiredSize <= *allowedSize) case.
    *answerFlags = NRequestMemoryAnswerFlags::k_Allow; // optional code
  }
  else
  {
    // we clear *answerFlags, because we want to disable dafault "Allow", if it's set.
    // *answerFlags = 0;
  /*
      NRequestMemoryAnswerFlags::k_SkipArc |
      NRequestMemoryAnswerFlags::k_Limit_Exceeded;
  */
  }
#endif
  
  UString s;
  if (!path
      && indexType == NArchive::NEventIndexType::kInArcIndex
      && index != (UInt32)(Int32)-1
      && _arc)
  {
    RINOK(_arc->GetItem_Path(index, s))
    path = s.Ptr();
  }
  
  return _requestMemoryUseCallback->RequestMemoryUse(
      flags, indexType, index, path,
      requiredSize, allowedSize, answerFlags);
}

#endif // Z7_SFX



// ------------ After Extracting functions ------------

void CDirPathSortPair::SetNumSlashes(const FChar *s)
{
  for (unsigned numSlashes = 0;;)
  {
    FChar c = *s++;
    if (c == 0)
    {
      Len = numSlashes;
      return;
    }
    if (IS_PATH_SEPAR(c))
      numSlashes++;
  }
}


bool CFiTimesCAM::SetDirTime_to_FS(CFSTR path) const
{
  // it's same function for dir and for file
  return NDir::SetDirTime(path,
      CTime_Defined ? &CTime : NULL,
      ATime_Defined ? &ATime : NULL,
      MTime_Defined ? &MTime : NULL);
}


#ifdef SUPPORT_LINKS

bool CFiTimesCAM::SetLinkFileTime_to_FS(CFSTR path) const
{
  // it's same function for dir and for file
  return NDir::SetLinkFileTime(path,
      CTime_Defined ? &CTime : NULL,
      ATime_Defined ? &ATime : NULL,
      MTime_Defined ? &MTime : NULL);
}

HRESULT CArchiveExtractCallback::SetPostLinks() const
{
  FOR_VECTOR (i, _postLinks)
  {
    const CPostLink &link = _postLinks[i];
    bool linkWasSet = false;
    RINOK(SetLink2(*this, link, linkWasSet))
    if (linkWasSet)
    {
#ifdef _WIN32
      //  Linux now doesn't support permissions for symlinks
      SetAttrib_Base(link.fullProcessedPath_from, link.item_FileInfo, *this);
#endif

      CFiTimesCAM pt;
      GetFiTimesCAM(link.item_FileInfo, pt, *_arc);
      if (pt.IsSomeTimeDefined())
        pt.SetLinkFileTime_to_FS(link.fullProcessedPath_from);

#ifdef Z7_USE_SECURITY_CODE
      // we set security information after timestamps setting
      RINOK(SetSecurityInfo(link.Index_in_Arc, link.fullProcessedPath_from))
#endif
    }
  }
  return S_OK;
}

#endif


HRESULT CArchiveExtractCallback::SetDirsTimes()
{
  if (!_arc)
    return S_OK;

  CRecordVector<CDirPathSortPair> pairs;
  pairs.ClearAndSetSize(_extractedFolders.Size());
  unsigned i;
  
  for (i = 0; i < _extractedFolders.Size(); i++)
  {
    CDirPathSortPair &pair = pairs[i];
    pair.Index = i;
    pair.SetNumSlashes(_extractedFolders[i].Path);
  }
  
  pairs.Sort2();
  
  HRESULT res = S_OK;

  for (i = 0; i < pairs.Size(); i++)
  {
    const CDirPathTime &dpt = _extractedFolders[pairs[i].Index];
    if (!dpt.SetDirTime_to_FS_2())
    {
      // result = E_FAIL;
      // do we need error message here in Windows and in posix?
      // SendMessageError_with_LastError("Cannot set directory time", dpt.Path);
    }
  }

  /*
  #ifndef _WIN32
  for (i = 0; i < _delayedSymLinks.Size(); i++)
  {
    const CDelayedSymLink &link = _delayedSymLinks[i];
    if (!link.Create())
    {
      if (res == S_OK)
        res = GetLastError_noZero_HRESULT();
      // res = E_FAIL;
      // do we need error message here in Windows and in posix?
      SendMessageError_with_LastError("Cannot create Symbolic Link", link._source);
    }
  }
  #endif // _WIN32
  */

  ClearExtractedDirsInfo();
  return res;
}


HRESULT CArchiveExtractCallback::CloseArc()
{
  // we call CloseReparseAndFile() here because we can have non-closed file in some cases?
  HRESULT res = CloseReparseAndFile();
#ifdef SUPPORT_LINKS
  {
    const HRESULT res2 = SetPostLinks();
    if (res == S_OK)
      res = res2;
  }
#endif
  {
    const HRESULT res2 = SetDirsTimes();
    if (res == S_OK)
      res = res2;
  }
  _arc = NULL;
  return res;
}
