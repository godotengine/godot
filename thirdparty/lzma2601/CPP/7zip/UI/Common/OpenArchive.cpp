// OpenArchive.cpp

#include "StdAfx.h"

// #define SHOW_DEBUG_INFO

#ifdef SHOW_DEBUG_INFO
#include <stdio.h>
#endif

#include "../../../../C/CpuArch.h"

#include "../../../Common/ComTry.h"
#include "../../../Common/IntToString.h"
#include "../../../Common/StringConvert.h"
#include "../../../Common/StringToInt.h"
#include "../../../Common/UTFConvert.h"
#include "../../../Common/Wildcard.h"

#include "../../../Windows/FileDir.h"

#include "../../Common/FileStreams.h"
#include "../../Common/LimitedStreams.h"
#include "../../Common/ProgressUtils.h"
#include "../../Common/StreamUtils.h"

#include "../../Compress/CopyCoder.h"

#include "DefaultName.h"
#include "OpenArchive.h"

#ifndef Z7_SFX
#include "SetProperties.h"
#endif

#ifndef Z7_SFX
#ifdef SHOW_DEBUG_INFO
#define PRF(x) x
#else
#define PRF(x)
#endif
#endif

// increase it, if you need to support larger SFX stubs
static const UInt64 kMaxCheckStartPosition = 1 << 23;

/*
Open:
  - formatIndex >= 0 (exact Format)
       1) Open with main type. Archive handler is allowed to use archive start finder.
          Warning, if there is tail.
  
  - formatIndex = -1 (Parser:0) (default)
    - same as #1 but doesn't return Parser

  - formatIndex = -2 (#1)
    - file has supported extension (like a.7z)
      Open with that main type (only starting from start of file).
        - open OK:
            - if there is no tail - return OK
            - if there is tail:
              - archive is not "Self Exe" - return OK with Warning, that there is tail
              - archive is "Self Exe"
                ignore "Self Exe" stub, and tries to open tail
                  - tail can be open as archive - shows that archive and stub size property.
                  - tail can't be open as archive - shows Parser ???
        - open FAIL:
           Try to open with all other types from offset 0 only.
           If some open type is OK and physical archive size is uequal or larger
           than file size, then return that archive with warning that cannot be open as [extension type].
           If extension was EXE, it will try to open as unknown_extension case
    - file has unknown extension (like a.hhh)
       It tries to open via parser code.
         - if there is full archive or tail archive and unknown block or "Self Exe"
           at front, it shows tail archive and stub size property.
         - in another cases, if there is some archive inside file, it returns parser/
         - in another cases, it retuens S_FALSE

       
  - formatIndex = -3 (#2)
    - same as #1, but
    - stub (EXE) + archive is open in Parser

  - formatIndex = -4 (#3)
    - returns only Parser. skip full file archive. And show other sub-archives

  - formatIndex = -5 (#4)
    - returns only Parser. skip full file archive. And show other sub-archives for each byte pos

*/




using namespace NWindows;

/*
#ifdef Z7_SFX
#define OPEN_PROPS_PARAM
#else
#define OPEN_PROPS_PARAM  , props
#endif
*/

/*
CArc::~CArc()
{
  GetRawProps.Release();
  Archive.Release();
  printf("\nCArc::~CArc()\n");
}
*/

#ifndef Z7_SFX

namespace NArchive {
namespace NParser {

struct CParseItem
{
  UInt64 Offset;
  UInt64 Size;
  // UInt64 OkSize;
  UString Name;
  UString Extension;
  FILETIME FileTime;
  UString Comment;
  UString ArcType;
  
  bool FileTime_Defined;
  bool UnpackSize_Defined;
  bool NumSubDirs_Defined;
  bool NumSubFiles_Defined;

  bool IsSelfExe;
  bool IsNotArcType;
  
  UInt64 UnpackSize;
  UInt64 NumSubDirs;
  UInt64 NumSubFiles;

  int FormatIndex;

  bool LenIsUnknown;

  CParseItem():
      // OkSize(0),
      FileTime_Defined(false),
      UnpackSize_Defined(false),
      NumSubDirs_Defined(false),
      NumSubFiles_Defined(false),
      IsSelfExe(false),
      IsNotArcType(false),
      LenIsUnknown(false)
    {}

  /*
  bool IsEqualTo(const CParseItem &item) const
  {
    return Offset == item.Offset && Size == item.Size;
  }
  */

  void NormalizeOffset()
  {
    if ((Int64)Offset < 0)
    {
      Size += Offset;
      // OkSize += Offset;
      Offset = 0;
    }
  }
};

Z7_CLASS_IMP_CHandler_IInArchive_1(
  IInArchiveGetStream
)
public:
  CObjectVector<CParseItem> _items;
  UInt64 _maxEndOffset;
  CMyComPtr<IInStream> _stream;

  UInt64 GetLastEnd() const
  {
    if (_items.IsEmpty())
      return 0;
    const CParseItem &back = _items.Back();
    return back.Offset + back.Size;
  }

  void AddUnknownItem(UInt64 next);
  int FindInsertPos(const CParseItem &item) const;
  void AddItem(const CParseItem &item);
  
  CHandler(): _maxEndOffset(0) {}
};

int CHandler::FindInsertPos(const CParseItem &item) const
{
  unsigned left = 0, right = _items.Size();
  while (left != right)
  {
    const unsigned mid = (unsigned)(((size_t)left + (size_t)right) / 2);
    const CParseItem &midItem = _items[mid];
    if (item.Offset < midItem.Offset)
      right = mid;
    else if (item.Offset > midItem.Offset)
      left = mid + 1;
    else if (item.Size < midItem.Size)
      right = mid;
    /*
    else if (item.Size > midItem.Size)
      left = mid + 1;
    */
    else
    {
      left = mid + 1;
      // return -1;
    }
  }
  return (int)left;
}

void CHandler::AddUnknownItem(UInt64 next)
{
  /*
  UInt64 prevEnd = 0;
  if (!_items.IsEmpty())
  {
    const CParseItem &back = _items.Back();
    prevEnd = back.Offset + back.Size;
  }
  */
  if (_maxEndOffset < next)
  {
    CParseItem item2;
    item2.Offset = _maxEndOffset;
    item2.Size = next - _maxEndOffset;
    _maxEndOffset = next;
    _items.Add(item2);
  }
  else if (_maxEndOffset > next && !_items.IsEmpty())
  {
    CParseItem &back = _items.Back();
    if (back.LenIsUnknown)
    {
      back.Size = next - back.Offset;
      _maxEndOffset = next;
    }
  }
}

void CHandler::AddItem(const CParseItem &item)
{
  AddUnknownItem(item.Offset);
  const int pos = FindInsertPos(item);
  if (pos != -1)
  {
    _items.Insert((unsigned)pos, item);
    UInt64 next = item.Offset + item.Size;
    if (_maxEndOffset < next)
      _maxEndOffset = next;
  }
}

/*
static const CStatProp kProps[] =
{
  { NULL, kpidPath, VT_BSTR},
  { NULL, kpidSize, VT_UI8},
  { NULL, kpidMTime, VT_FILETIME},
  { NULL, kpidType, VT_BSTR},
  { NULL, kpidComment, VT_BSTR},
  { NULL, kpidOffset, VT_UI8},
  { NULL, kpidUnpackSize, VT_UI8},
//   { NULL, kpidNumSubDirs, VT_UI8},
};
*/

static const Byte kProps[] =
{
  kpidPath,
  kpidSize,
  kpidMTime,
  kpidType,
  kpidComment,
  kpidOffset,
  kpidUnpackSize
};

IMP_IInArchive_Props
IMP_IInArchive_ArcProps_NO

Z7_COM7F_IMF(CHandler::Open(IInStream *stream, const UInt64 *, IArchiveOpenCallback * /* openArchiveCallback */))
{
  COM_TRY_BEGIN
  {
    Close();
    _stream = stream;
  }
  return S_OK;
  COM_TRY_END
}

Z7_COM7F_IMF(CHandler::Close())
{
  _items.Clear();
  _stream.Release();
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetNumberOfItems(UInt32 *numItems))
{
  *numItems = _items.Size();
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetProperty(UInt32 index, PROPID propID, PROPVARIANT *value))
{
  COM_TRY_BEGIN
  NCOM::CPropVariant prop;

  const CParseItem &item = _items[index];

  switch (propID)
  {
    case kpidPath:
    {
      char sz[32];
      ConvertUInt32ToString(index + 1, sz);
      UString s(sz);
      if (!item.Name.IsEmpty())
      {
        s.Add_Dot();
        s += item.Name;
      }
      if (!item.Extension.IsEmpty())
      {
        s.Add_Dot();
        s += item.Extension;
      }
      prop = s; break;
    }
    case kpidSize:
    case kpidPackSize: prop = item.Size; break;
    case kpidOffset: prop = item.Offset; break;
    case kpidUnpackSize: if (item.UnpackSize_Defined) prop = item.UnpackSize; break;
    case kpidNumSubFiles: if (item.NumSubFiles_Defined) prop = item.NumSubFiles; break;
    case kpidNumSubDirs: if (item.NumSubDirs_Defined) prop = item.NumSubDirs; break;
    case kpidMTime: if (item.FileTime_Defined) prop = item.FileTime; break;
    case kpidComment: if (!item.Comment.IsEmpty()) prop = item.Comment; break;
    case kpidType: if (!item.ArcType.IsEmpty()) prop = item.ArcType; break;
    default: break;
  }
  prop.Detach(value);
  return S_OK;
  COM_TRY_END
}

Z7_COM7F_IMF(CHandler::Extract(const UInt32 *indices, UInt32 numItems,
    Int32 testMode, IArchiveExtractCallback *extractCallback))
{
  COM_TRY_BEGIN
  
  const bool allFilesMode = (numItems == (UInt32)(Int32)-1);
  if (allFilesMode)
    numItems = _items.Size();
  if (_stream && numItems == 0)
    return S_OK;
  UInt64 totalSize = 0;
  UInt32 i;
  for (i = 0; i < numItems; i++)
    totalSize += _items[allFilesMode ? i : indices[i]].Size;
  extractCallback->SetTotal(totalSize);

  totalSize = 0;
  
  CLocalProgress *lps = new CLocalProgress;
  CMyComPtr<ICompressProgressInfo> progress = lps;
  lps->Init(extractCallback, false);

  CLimitedSequentialInStream *streamSpec = new CLimitedSequentialInStream;
  CMyComPtr<ISequentialInStream> inStream(streamSpec);
  streamSpec->SetStream(_stream);

  CLimitedSequentialOutStream *outStreamSpec = new CLimitedSequentialOutStream;
  CMyComPtr<ISequentialOutStream> outStream(outStreamSpec);

  NCompress::CCopyCoder *copyCoderSpec = new NCompress::CCopyCoder();
  CMyComPtr<ICompressCoder> copyCoder = copyCoderSpec;

  for (i = 0; i < numItems; i++)
  {
    lps->InSize = totalSize;
    lps->OutSize = totalSize;
    RINOK(lps->SetCur())
    CMyComPtr<ISequentialOutStream> realOutStream;
    const Int32 askMode = testMode ?
        NExtract::NAskMode::kTest :
        NExtract::NAskMode::kExtract;
    const UInt32 index = allFilesMode ? i : indices[i];
    const CParseItem &item = _items[index];

    RINOK(extractCallback->GetStream(index, &realOutStream, askMode))
    UInt64 unpackSize = item.Size;
    totalSize += unpackSize;
    bool skipMode = false;
    if (!testMode && !realOutStream)
      continue;
    RINOK(extractCallback->PrepareOperation(askMode))

    outStreamSpec->SetStream(realOutStream);
    realOutStream.Release();
    outStreamSpec->Init(skipMode ? 0 : unpackSize, true);

    Int32 opRes = NExtract::NOperationResult::kOK;
    RINOK(InStream_SeekSet(_stream, item.Offset))
    streamSpec->Init(unpackSize);
    RINOK(copyCoder->Code(inStream, outStream, NULL, NULL, progress))

    if (outStreamSpec->GetRem() != 0)
      opRes = NExtract::NOperationResult::kDataError;
    outStreamSpec->ReleaseStream();
    RINOK(extractCallback->SetOperationResult(opRes))
  }
  
  return S_OK;
  
  COM_TRY_END
}


Z7_COM7F_IMF(CHandler::GetStream(UInt32 index, ISequentialInStream **stream))
{
  COM_TRY_BEGIN
  const CParseItem &item = _items[index];
  return CreateLimitedInStream(_stream, item.Offset, item.Size, stream);
  COM_TRY_END
}

}}

#endif

HRESULT Archive_GetItemBoolProp(IInArchive *arc, UInt32 index, PROPID propID, bool &result) throw()
{
  NCOM::CPropVariant prop;
  result = false;
  RINOK(arc->GetProperty(index, propID, &prop))
  if (prop.vt == VT_BOOL)
    result = VARIANT_BOOLToBool(prop.boolVal);
  else if (prop.vt != VT_EMPTY)
    return E_FAIL;
  return S_OK;
}

HRESULT Archive_IsItem_Dir(IInArchive *arc, UInt32 index, bool &result) throw()
{
  return Archive_GetItemBoolProp(arc, index, kpidIsDir, result);
}

HRESULT Archive_IsItem_Aux(IInArchive *arc, UInt32 index, bool &result) throw()
{
  return Archive_GetItemBoolProp(arc, index, kpidIsAux, result);
}

HRESULT Archive_IsItem_AltStream(IInArchive *arc, UInt32 index, bool &result) throw()
{
  return Archive_GetItemBoolProp(arc, index, kpidIsAltStream, result);
}

HRESULT Archive_IsItem_Deleted(IInArchive *arc, UInt32 index, bool &result) throw()
{
  return Archive_GetItemBoolProp(arc, index, kpidIsDeleted, result);
}

static HRESULT Archive_GetArcProp_Bool(IInArchive *arc, PROPID propid, bool &result) throw()
{
  NCOM::CPropVariant prop;
  result = false;
  RINOK(arc->GetArchiveProperty(propid, &prop))
  if (prop.vt == VT_BOOL)
    result = VARIANT_BOOLToBool(prop.boolVal);
  else if (prop.vt != VT_EMPTY)
    return E_FAIL;
  return S_OK;
}

static HRESULT Archive_GetArcProp_UInt(IInArchive *arc, PROPID propid, UInt64 &result, bool &defined)
{
  defined = false;
  NCOM::CPropVariant prop;
  RINOK(arc->GetArchiveProperty(propid, &prop))
  switch (prop.vt)
  {
    case VT_UI4: result = prop.ulVal; break;
    case VT_I4:  result = (UInt64)(Int64)prop.lVal; break;
    case VT_UI8: result = (UInt64)prop.uhVal.QuadPart; break;
    case VT_I8:  result = (UInt64)prop.hVal.QuadPart; break;
    case VT_EMPTY: return S_OK;
    default: return E_FAIL;
  }
  defined = true;
  return S_OK;
}

static HRESULT Archive_GetArcProp_Int(IInArchive *arc, PROPID propid, Int64 &result, bool &defined)
{
  defined = false;
  NCOM::CPropVariant prop;
  RINOK(arc->GetArchiveProperty(propid, &prop))
  switch (prop.vt)
  {
    case VT_UI4: result = prop.ulVal; break;
    case VT_I4:  result = prop.lVal; break;
    case VT_UI8: result = (Int64)prop.uhVal.QuadPart; break;
    case VT_I8:  result = (Int64)prop.hVal.QuadPart; break;
    case VT_EMPTY: return S_OK;
    default: return E_FAIL;
  }
  defined = true;
  return S_OK;
}

#ifndef Z7_SFX

HRESULT CArc::GetItem_PathToParent(UInt32 index, UInt32 parent, UStringVector &parts) const
{
  if (!GetRawProps)
    return E_FAIL;
  if (index == parent)
    return S_OK;
  UInt32 curIndex = index;
  
  UString s;
  
  bool prevWasAltStream = false;
  
  for (;;)
  {
    #ifdef MY_CPU_LE
    const void *p;
    UInt32 size;
    UInt32 propType;
    RINOK(GetRawProps->GetRawProp(curIndex, kpidName, &p, &size, &propType))
    if (p && propType == PROP_DATA_TYPE_wchar_t_PTR_Z_LE)
      s = (const wchar_t *)p;
    else
    #endif
    {
      NCOM::CPropVariant prop;
      RINOK(Archive->GetProperty(curIndex, kpidName, &prop))
      if (prop.vt == VT_BSTR && prop.bstrVal)
        s.SetFromBstr(prop.bstrVal);
      else if (prop.vt == VT_EMPTY)
        s.Empty();
      else
        return E_FAIL;
    }

    UInt32 curParent = (UInt32)(Int32)-1;
    UInt32 parentType = 0;
    RINOK(GetRawProps->GetParent(curIndex, &curParent, &parentType))

    // 18.06: fixed : we don't want to split name to parts
    /*
    if (parentType != NParentType::kAltStream)
    {
      for (;;)
      {
        int pos = s.ReverseFind_PathSepar();
        if (pos < 0)
        {
          break;
        }
        parts.Insert(0, s.Ptr(pos + 1));
        s.DeleteFrom(pos);
      }
    }
    */
    
    parts.Insert(0, s);

    if (prevWasAltStream)
    {
      {
        UString &s2 = parts[parts.Size() - 2];
        s2.Add_Colon();
        s2 += parts.Back();
      }
      parts.DeleteBack();
    }

    if (parent == curParent)
      return S_OK;
    
    prevWasAltStream = false;
    if (parentType == NParentType::kAltStream)
      prevWasAltStream = true;
    
    if (curParent == (UInt32)(Int32)-1)
      return E_FAIL;
    curIndex = curParent;
  }
}

#endif



HRESULT CArc::GetItem_Path(UInt32 index, UString &result) const
{
  #ifdef MY_CPU_LE
  if (GetRawProps)
  {
    const void *p;
    UInt32 size;
    UInt32 propType;
    if (!IsTree)
    {
      if (GetRawProps->GetRawProp(index, kpidPath, &p, &size, &propType) == S_OK &&
          propType == NPropDataType::kUtf16z)
      {
        unsigned len = size / 2 - 1;
        // (len) doesn't include null terminator

        /*
        #if WCHAR_MAX > 0xffff
        len = (unsigned)Utf16LE__Get_Num_WCHARs(p, len);

        wchar_t *s = result.GetBuf(len);
        wchar_t *sEnd = Utf16LE__To_WCHARs_Sep(p, len, s);
        if (s + len != sEnd) return E_FAIL;
        *sEnd = 0;
        
        #else
        */
        
        wchar_t *s = result.GetBuf(len);
        for (unsigned i = 0; i < len; i++)
        {
          wchar_t c = GetUi16(p);
          p = (const void *)((const Byte *)p + 2);

          #if WCHAR_PATH_SEPARATOR != L'/'
          if (c == L'/')
            c = WCHAR_PATH_SEPARATOR;
          else if (c == L'\\')
            c = WCHAR_IN_FILE_NAME_BACKSLASH_REPLACEMENT; // WSL scheme
          #endif

          *s++ = c;
        }
        *s = 0;
        
        // #endif
        
        result.ReleaseBuf_SetLen(len);

        Convert_UnicodeEsc16_To_UnicodeEscHigh(result);
        if (len != 0)
          return S_OK;
      }
    }
    /*
    else if (GetRawProps->GetRawProp(index, kpidName, &p, &size, &propType) == S_OK &&
        p && propType == NPropDataType::kUtf16z)
    {
      size -= 2;
      UInt32 totalSize = size;
      bool isOK = false;
      
      {
        UInt32 index2 = index;
        for (;;)
        {
          UInt32 parent = (UInt32)(Int32)-1;
          UInt32 parentType = 0;
          if (GetRawProps->GetParent(index2, &parent, &parentType) != S_OK)
            break;
          if (parent == (UInt32)(Int32)-1)
          {
            if (parentType != 0)
              totalSize += 2;
            isOK = true;
            break;
          }
          index2 = parent;
          UInt32 size2;
          const void *p2;
          if (GetRawProps->GetRawProp(index2, kpidName, &p2, &size2, &propType) != S_OK &&
              p2 && propType == NPropDataType::kUtf16z)
            break;
          totalSize += size2;
        }
      }

      if (isOK)
      {
        wchar_t *sz = result.GetBuf_SetEnd(totalSize / 2);
        UInt32 pos = totalSize - size;
        memcpy((Byte *)sz + pos, p, size);
        UInt32 index2 = index;
        for (;;)
        {
          UInt32 parent = (UInt32)(Int32)-1;
          UInt32 parentType = 0;
          if (GetRawProps->GetParent(index2, &parent, &parentType) != S_OK)
            break;
          if (parent == (UInt32)(Int32)-1)
          {
            if (parentType != 0)
              sz[pos / 2 - 1] = L':';
            break;
          }
          index2 = parent;
          UInt32 size2;
          const void *p2;
          if (GetRawProps->GetRawProp(index2, kpidName, &p2, &size2, &propType) != S_OK)
            break;
          pos -= size2;
          memcpy((Byte *)sz + pos, p2, size2);
          sz[(pos + size2 - 2) / 2] = (parentType == 0) ? WCHAR_PATH_SEPARATOR : L':';
        }
        #ifdef _WIN32
        // result.Replace(L'/', WCHAR_PATH_SEPARATOR);
        #endif
        return S_OK;
      }
    }
    */
  }
  #endif
  
  {
    NCOM::CPropVariant prop;
    RINOK(Archive->GetProperty(index, kpidPath, &prop))
    if (prop.vt == VT_BSTR && prop.bstrVal)
      result.SetFromBstr(prop.bstrVal);
    else if (prop.vt == VT_EMPTY)
      result.Empty();
    else
      return E_FAIL;
  }
  
  if (result.IsEmpty())
    return GetItem_DefaultPath(index, result);

  Convert_UnicodeEsc16_To_UnicodeEscHigh(result);
  return S_OK;
}

HRESULT CArc::GetItem_DefaultPath(UInt32 index, UString &result) const
{
  result.Empty();
  bool isDir;
  RINOK(Archive_IsItem_Dir(Archive, index, isDir))
  if (!isDir)
  {
    result = DefaultName;
    NCOM::CPropVariant prop;
    RINOK(Archive->GetProperty(index, kpidExtension, &prop))
    if (prop.vt == VT_BSTR)
    {
      result.Add_Dot();
      result += prop.bstrVal;
    }
    else if (prop.vt != VT_EMPTY)
      return E_FAIL;
  }
  return S_OK;
}

HRESULT CArc::GetItem_Path2(UInt32 index, UString &result) const
{
  RINOK(GetItem_Path(index, result))
  if (Ask_Deleted)
  {
    bool isDeleted = false;
    RINOK(Archive_IsItem_Deleted(Archive, index, isDeleted))
    if (isDeleted)
      result.Insert(0, L"[DELETED]" WSTRING_PATH_SEPARATOR);
  }
  return S_OK;
}

#ifdef SUPPORT_ALT_STREAMS

int FindAltStreamColon_in_Path(const wchar_t *path)
{
  unsigned i = 0;
  int colonPos = -1;
  for (;; i++)
  {
    wchar_t c = path[i];
    if (c == 0)
      return colonPos;
    if (c == ':')
    {
      if (colonPos < 0)
        colonPos = (int)i;
      continue;
    }
    if (c == WCHAR_PATH_SEPARATOR)
      colonPos = -1;
  }
}

#endif

HRESULT CArc::GetItem(UInt32 index, CReadArcItem &item) const
{
  #ifdef SUPPORT_ALT_STREAMS
  item.IsAltStream = false;
  item.AltStreamName.Empty();
  item.MainPath.Empty();
  #endif

  item.IsDir = false;
  item.Path.Empty();
  item.ParentIndex = (UInt32)(Int32)-1;
  
  item.PathParts.Clear();

  RINOK(Archive_IsItem_Dir(Archive, index, item.IsDir))
  item.MainIsDir = item.IsDir;

  RINOK(GetItem_Path2(index, item.Path))

  #ifndef Z7_SFX
  UInt32 mainIndex = index;
  #endif

  #ifdef SUPPORT_ALT_STREAMS

  item.MainPath = item.Path;
  if (Ask_AltStream)
  {
    RINOK(Archive_IsItem_AltStream(Archive, index, item.IsAltStream))
  }
  
  bool needFindAltStream = false;

  if (item.IsAltStream)
  {
    needFindAltStream = true;
    if (GetRawProps)
    {
      UInt32 parentType = 0;
      UInt32 parentIndex;
      RINOK(GetRawProps->GetParent(index, &parentIndex, &parentType))
      if (parentType == NParentType::kAltStream)
      {
        NCOM::CPropVariant prop;
        RINOK(Archive->GetProperty(index, kpidName, &prop))
        if (prop.vt == VT_BSTR && prop.bstrVal)
          item.AltStreamName.SetFromBstr(prop.bstrVal);
        else if (prop.vt != VT_EMPTY)
          return E_FAIL;
        else
        {
          // item.IsAltStream = false;
        }
        /*
        if (item.AltStreamName.IsEmpty())
          item.IsAltStream = false;
        */

        needFindAltStream = false;
        item.ParentIndex = parentIndex;
        mainIndex = parentIndex;

        if (parentIndex == (UInt32)(Int32)-1)
        {
          item.MainPath.Empty();
          item.MainIsDir = true;
        }
        else
        {
          RINOK(GetItem_Path2(parentIndex, item.MainPath))
          RINOK(Archive_IsItem_Dir(Archive, parentIndex, item.MainIsDir))
        }
      }
    }
  }

  if (item.WriteToAltStreamIfColon || needFindAltStream)
  {
    /* Good handler must support GetRawProps::GetParent for alt streams.
       So the following code currently is not used */
    int colon = FindAltStreamColon_in_Path(item.Path);
    if (colon >= 0)
    {
      item.MainPath.DeleteFrom((unsigned)colon);
      item.AltStreamName = item.Path.Ptr((unsigned)(colon + 1));
      item.MainIsDir = (colon == 0 || IsPathSepar(item.Path[(unsigned)colon - 1]));
      item.IsAltStream = true;
    }
  }

  #endif
  
  #ifndef Z7_SFX
  if (item._use_baseParentFolder_mode)
  {
    RINOK(GetItem_PathToParent(mainIndex, (unsigned)item._baseParentFolder, item.PathParts))
    
    #ifdef SUPPORT_ALT_STREAMS
    if ((item.WriteToAltStreamIfColon || needFindAltStream) && !item.PathParts.IsEmpty())
    {
      int colon;
      {
        UString &s = item.PathParts.Back();
        colon = FindAltStreamColon_in_Path(s);
        if (colon >= 0)
        {
          item.AltStreamName = s.Ptr((unsigned)(colon + 1));
          item.MainIsDir = (colon == 0 || IsPathSepar(s[(unsigned)colon - 1]));
          item.IsAltStream = true;
          s.DeleteFrom((unsigned)colon);
        }
      }
      if (colon == 0)
        item.PathParts.DeleteBack();
    }
    #endif
    
  }
  else
  #endif
    SplitPathToParts(
          #ifdef SUPPORT_ALT_STREAMS
            item.MainPath
          #else
            item.Path
          #endif
      , item.PathParts);

  return S_OK;
}

#ifndef Z7_SFX

static HRESULT Archive_GetItem_Size(IInArchive *archive, UInt32 index, UInt64 &size, bool &defined)
{
  NCOM::CPropVariant prop;
  defined = false;
  size = 0;
  RINOK(archive->GetProperty(index, kpidSize, &prop))
  switch (prop.vt)
  {
    case VT_UI1: size = prop.bVal; break;
    case VT_UI2: size = prop.uiVal; break;
    case VT_UI4: size = prop.ulVal; break;
    case VT_UI8: size = (UInt64)prop.uhVal.QuadPart; break;
    case VT_EMPTY: return S_OK;
    default: return E_FAIL;
  }
  defined = true;
  return S_OK;
}

#endif

HRESULT CArc::GetItem_Size(UInt32 index, UInt64 &size, bool &defined) const
{
  NCOM::CPropVariant prop;
  defined = false;
  size = 0;
  RINOK(Archive->GetProperty(index, kpidSize, &prop))
  switch (prop.vt)
  {
    case VT_UI1: size = prop.bVal; break;
    case VT_UI2: size = prop.uiVal; break;
    case VT_UI4: size = prop.ulVal; break;
    case VT_UI8: size = (UInt64)prop.uhVal.QuadPart; break;
    case VT_EMPTY: return S_OK;
    default: return E_FAIL;
  }
  defined = true;
  return S_OK;
}

HRESULT CArc::GetItem_MTime(UInt32 index, CArcTime &at) const
{
  at.Clear();
  NCOM::CPropVariant prop;
  RINOK(Archive->GetProperty(index, kpidMTime, &prop))
  
  if (prop.vt == VT_FILETIME)
  {
    /*
    // for debug
    if (FILETIME_IsZero(prop.at) && MTime.Def)
    {
      at = MTime;
      return S_OK;
    }
    */
    at.Set_From_Prop(prop);
    if (at.Prec == 0)
    {
      // (at.Prec == 0) before version 22.
      // so kpidTimeType is required for that code
      prop.Clear();
      RINOK(Archive->GetProperty(index, kpidTimeType, &prop))
      if (prop.vt == VT_UI4)
      {
        UInt32 val = prop.ulVal;
        if (val == NFileTimeType::kWindows)
          val = k_PropVar_TimePrec_100ns;
        /*
        else if (val > k_PropVar_TimePrec_1ns)
        {
          val = k_PropVar_TimePrec_100ns;
          // val = k_PropVar_TimePrec_1ns;
          // return E_FAIL; // for debug
        }
        */
        at.Prec = (UInt16)val;
      }
    }
    return S_OK;
  }
  
  if (prop.vt != VT_EMPTY)
    return E_FAIL;
  if (MTime.Def)
    at = MTime;
  return S_OK;
}

#ifndef Z7_SFX

static inline bool TestSignature(const Byte *p1, const Byte *p2, size_t size)
{
  for (size_t i = 0; i < size; i++)
    if (p1[i] != p2[i])
      return false;
  return true;
}


static void MakeCheckOrder(CCodecs *codecs,
    CIntVector &orderIndices, unsigned numTypes, CIntVector &orderIndices2,
    const Byte *data, size_t dataSize)
{
  for (unsigned i = 0; i < numTypes; i++)
  {
    const int index = orderIndices[i];
    if (index < 0)
      continue;
    const CArcInfoEx &ai = codecs->Formats[(unsigned)index];
    if (ai.SignatureOffset == 0)
    {
      if (ai.Signatures.IsEmpty())
      {
        if (dataSize != 0) // 21.04: no Signature means Empty Signature
          continue;
      }
      else
      {
        unsigned k;
        const CObjectVector<CByteBuffer> &sigs = ai.Signatures;
        for (k = 0; k < sigs.Size(); k++)
        {
          const CByteBuffer &sig = sigs[k];
          if (sig.Size() <= dataSize && TestSignature(data, sig, sig.Size()))
            break;
        }
        if (k == sigs.Size())
          continue;
      }
    }
    orderIndices2.Add(index);
    orderIndices[i] = -1;
  }
}

#ifdef UNDER_CE
  static const unsigned kNumHashBytes = 1;
  #define HASH_VAL(buf) ((buf)[0])
#else
  static const unsigned kNumHashBytes = 2;
  // #define HASH_VAL(buf) ((buf)[0] | ((UInt32)(buf)[1] << 8))
  #define HASH_VAL(buf) GetUi16(buf)
#endif

static bool IsExeExt(const UString &ext)
{
  return ext.IsEqualTo_Ascii_NoCase("exe");
}

static const char * const k_PreArcFormats[] =
{
    "pe"
  , "elf"
  , "macho"
  , "mub"
  , "te"
};

static bool IsNameFromList(const UString &s, const char * const names[], size_t num)
{
  for (unsigned i = 0; i < num; i++)
    if (StringsAreEqualNoCase_Ascii(s, names[i]))
      return true;
  return false;
}


static bool IsPreArcFormat(const CArcInfoEx &ai)
{
  if (ai.Flags_PreArc())
    return true;
  return IsNameFromList(ai.Name, k_PreArcFormats, Z7_ARRAY_SIZE(k_PreArcFormats));
}

static const char * const k_Formats_with_simple_signuature[] =
{
    "7z"
  , "xz"
  , "rar"
  , "bzip2"
  , "gzip"
  , "cab"
  , "wim"
  , "rpm"
  , "vhd"
  , "xar"
};

static bool IsNewStyleSignature(const CArcInfoEx &ai)
{
  // if (ai.Version >= 0x91F)
  if (ai.NewInterface)
    return true;
  return IsNameFromList(ai.Name, k_Formats_with_simple_signuature, Z7_ARRAY_SIZE(k_Formats_with_simple_signuature));
}



class CArchiveOpenCallback_Offset Z7_final:
  public IArchiveOpenCallback,
  public IArchiveOpenVolumeCallback,
 #ifndef Z7_NO_CRYPTO
  public ICryptoGetTextPassword,
 #endif
  public CMyUnknownImp
{
  Z7_COM_QI_BEGIN2(IArchiveOpenCallback)
  Z7_COM_QI_ENTRY(IArchiveOpenVolumeCallback)
 #ifndef Z7_NO_CRYPTO
  Z7_COM_QI_ENTRY(ICryptoGetTextPassword)
 #endif
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE

  Z7_IFACE_COM7_IMP(IArchiveOpenCallback)
  Z7_IFACE_COM7_IMP(IArchiveOpenVolumeCallback)
 #ifndef Z7_NO_CRYPTO
  Z7_IFACE_COM7_IMP(ICryptoGetTextPassword)
 #endif

public:
  CMyComPtr<IArchiveOpenCallback> Callback;
  CMyComPtr<IArchiveOpenVolumeCallback> OpenVolumeCallback;
  UInt64 Files;
  UInt64 Offset;
  
  #ifndef Z7_NO_CRYPTO
  CMyComPtr<ICryptoGetTextPassword> GetTextPassword;
  #endif
};

#ifndef Z7_NO_CRYPTO
Z7_COM7F_IMF(CArchiveOpenCallback_Offset::CryptoGetTextPassword(BSTR *password))
{
  COM_TRY_BEGIN
  if (GetTextPassword)
    return GetTextPassword->CryptoGetTextPassword(password);
  return E_NOTIMPL;
  COM_TRY_END
}
#endif

Z7_COM7F_IMF(CArchiveOpenCallback_Offset::SetTotal(const UInt64 *, const UInt64 *))
{
  return S_OK;
}

Z7_COM7F_IMF(CArchiveOpenCallback_Offset::SetCompleted(const UInt64 *, const UInt64 *bytes))
{
  if (!Callback)
    return S_OK;
  UInt64 value = Offset;
  if (bytes)
    value += *bytes;
  return Callback->SetCompleted(&Files, &value);
}

Z7_COM7F_IMF(CArchiveOpenCallback_Offset::GetProperty(PROPID propID, PROPVARIANT *value))
{
  if (OpenVolumeCallback)
    return OpenVolumeCallback->GetProperty(propID, value);
  NCOM::PropVariant_Clear(value);
  return S_OK;
  // return E_NOTIMPL;
}

Z7_COM7F_IMF(CArchiveOpenCallback_Offset::GetStream(const wchar_t *name, IInStream **inStream))
{
  if (OpenVolumeCallback)
    return OpenVolumeCallback->GetStream(name, inStream);
  return S_FALSE;
}

#endif


UInt32 GetOpenArcErrorFlags(const NCOM::CPropVariant &prop, bool *isDefinedProp)
{
  if (isDefinedProp != NULL)
    *isDefinedProp = false;

  switch (prop.vt)
  {
    case VT_UI8: if (isDefinedProp) *isDefinedProp = true; return (UInt32)prop.uhVal.QuadPart;
    case VT_UI4: if (isDefinedProp) *isDefinedProp = true; return prop.ulVal;
    case VT_EMPTY: return 0;
    default: throw 151199;
  }
}

void CArcErrorInfo::ClearErrors()
{
  // ErrorFormatIndex = -1; // we don't need to clear ErrorFormatIndex here !!!

  ThereIsTail = false;
  UnexpecedEnd = false;
  IgnoreTail = false;
  // NonZerosTail = false;
  ErrorFlags_Defined = false;
  ErrorFlags = 0;
  WarningFlags = 0;
  TailSize = 0;

  ErrorMessage.Empty();
  WarningMessage.Empty();
}

HRESULT CArc::ReadBasicProps(IInArchive *archive, UInt64 startPos, HRESULT openRes)
{
  // OkPhySize_Defined = false;
  PhySize_Defined = false;
  PhySize = 0;
  Offset = 0;
  AvailPhySize = FileSize - startPos;

  ErrorInfo.ClearErrors();
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidErrorFlags, &prop))
    ErrorInfo.ErrorFlags = GetOpenArcErrorFlags(prop, &ErrorInfo.ErrorFlags_Defined);
  }
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidWarningFlags, &prop))
    ErrorInfo.WarningFlags = GetOpenArcErrorFlags(prop);
  }

  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidError, &prop))
    if (prop.vt != VT_EMPTY)
      ErrorInfo.ErrorMessage = (prop.vt == VT_BSTR ? prop.bstrVal : L"Unknown error");
  }
  
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidWarning, &prop))
    if (prop.vt != VT_EMPTY)
      ErrorInfo.WarningMessage = (prop.vt == VT_BSTR ? prop.bstrVal : L"Unknown warning");
  }
  
  if (openRes == S_OK || ErrorInfo.IsArc_After_NonOpen())
  {
    RINOK(Archive_GetArcProp_UInt(archive, kpidPhySize, PhySize, PhySize_Defined))
    /*
    RINOK(Archive_GetArcProp_UInt(archive, kpidOkPhySize, OkPhySize, OkPhySize_Defined));
    if (!OkPhySize_Defined)
    {
      OkPhySize_Defined = PhySize_Defined;
      OkPhySize = PhySize;
    }
    */

    bool offsetDefined;
    RINOK(Archive_GetArcProp_Int(archive, kpidOffset, Offset, offsetDefined))

    Int64 globalOffset = (Int64)startPos + Offset;
    AvailPhySize = (UInt64)((Int64)FileSize - globalOffset);
    if (PhySize_Defined)
    {
      UInt64 endPos = (UInt64)(globalOffset + (Int64)PhySize);
      if (endPos < FileSize)
      {
        AvailPhySize = PhySize;
        ErrorInfo.ThereIsTail = true;
        ErrorInfo.TailSize = FileSize - endPos;
      }
      else if (endPos > FileSize)
        ErrorInfo.UnexpecedEnd = true;
    }
  }

  return S_OK;
}

/*
static void PrintNumber(const char *s, int n)
{
  char temp[100];
  sprintf(temp, "%s %d", s, n);
  // OutputDebugStringA(temp);
  printf(temp);
}
*/

HRESULT CArc::PrepareToOpen(const COpenOptions &op, unsigned formatIndex, CMyComPtr<IInArchive> &archive)
{
  // OutputDebugStringA("a1");
  // PrintNumber("formatIndex", formatIndex);
    
  RINOK(op.codecs->CreateInArchive(formatIndex, archive))
  // OutputDebugStringA("a2");
  if (!archive)
    return S_OK;

  #ifdef Z7_EXTERNAL_CODECS
  if (op.codecs->NeedSetLibCodecs)
  {
    const CArcInfoEx &ai = op.codecs->Formats[formatIndex];
    if (ai.LibIndex >= 0 ?
        !op.codecs->Libs[(unsigned)ai.LibIndex].SetCodecs :
        !op.codecs->Libs.IsEmpty())
    {
      CMyComPtr<ISetCompressCodecsInfo> setCompressCodecsInfo;
      archive.QueryInterface(IID_ISetCompressCodecsInfo, (void **)&setCompressCodecsInfo);
      if (setCompressCodecsInfo)
      {
        RINOK(setCompressCodecsInfo->SetCompressCodecsInfo(op.codecs))
      }
    }
  }
  #endif
  
  
  #ifndef Z7_SFX

  const CArcInfoEx &ai = op.codecs->Formats[formatIndex];
 
  // OutputDebugStringW(ai.Name);
  // OutputDebugStringA("a3");

  if (ai.Flags_PreArc())
  {
    /* we notify parsers that extract executables, that they don't need
       to open archive, if there is tail after executable (for SFX cases) */
    CMyComPtr<IArchiveAllowTail> allowTail;
    archive.QueryInterface(IID_IArchiveAllowTail, (void **)&allowTail);
    if (allowTail)
      allowTail->AllowTail(BoolToInt(true));
  }

  if (op.props)
  {
    /*
    FOR_VECTOR (y, op.props)
    {
      const COptionalOpenProperties &optProps = (*op.props)[y];
      if (optProps.FormatName.IsEmpty() || optProps.FormatName.CompareNoCase(ai.Name) == 0)
      {
        RINOK(SetProperties(archive, optProps.Props));
        break;
      }
    }
    */
    RINOK(SetProperties(archive, *op.props))
  }
  
  #endif
  return S_OK;
}

#ifndef Z7_SFX

static HRESULT ReadParseItemProps(IInArchive *archive, const CArcInfoEx &ai, NArchive::NParser::CParseItem &pi)
{
  pi.Extension = ai.GetMainExt();
  pi.FileTime_Defined = false;
  pi.ArcType = ai.Name;
  
  RINOK(Archive_GetArcProp_Bool(archive, kpidIsNotArcType, pi.IsNotArcType))

  // RINOK(Archive_GetArcProp_Bool(archive, kpidIsSelfExe, pi.IsSelfExe));
  pi.IsSelfExe = ai.Flags_PreArc();
  
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidMTime, &prop))
    if (prop.vt == VT_FILETIME)
    {
      pi.FileTime_Defined = true;
      pi.FileTime = prop.filetime;
    }
  }
  
  if (!pi.FileTime_Defined)
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidCTime, &prop))
    if (prop.vt == VT_FILETIME)
    {
      pi.FileTime_Defined = true;
      pi.FileTime = prop.filetime;
    }
  }
  
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidName, &prop))
    if (prop.vt == VT_BSTR)
    {
      pi.Name.SetFromBstr(prop.bstrVal);
      pi.Extension.Empty();
    }
    else
    {
      RINOK(archive->GetArchiveProperty(kpidExtension, &prop))
      if (prop.vt == VT_BSTR)
        pi.Extension.SetFromBstr(prop.bstrVal);
    }
  }
  
  {
    NCOM::CPropVariant prop;
    RINOK(archive->GetArchiveProperty(kpidShortComment, &prop))
    if (prop.vt == VT_BSTR)
      pi.Comment.SetFromBstr(prop.bstrVal);
  }


  UInt32 numItems;
  RINOK(archive->GetNumberOfItems(&numItems))
  
  // pi.NumSubFiles = numItems;
  // RINOK(Archive_GetArcProp_UInt(archive, kpidUnpackSize, pi.UnpackSize, pi.UnpackSize_Defined));
  // if (!pi.UnpackSize_Defined)
  {
    pi.NumSubFiles = 0;
    pi.NumSubDirs = 0;
    pi.UnpackSize = 0;
    for (UInt32 i = 0; i < numItems; i++)
    {
      UInt64 size = 0;
      bool defined = false;
      Archive_GetItem_Size(archive, i, size, defined);
      if (defined)
      {
        pi.UnpackSize_Defined = true;
        pi.UnpackSize += size;
      }

      bool isDir = false;
      Archive_IsItem_Dir(archive, i, isDir);
      if (isDir)
        pi.NumSubDirs++;
      else
        pi.NumSubFiles++;
    }
    if (pi.NumSubDirs != 0)
      pi.NumSubDirs_Defined = true;
    pi.NumSubFiles_Defined = true;
  }

  return S_OK;
}

#endif

HRESULT CArc::CheckZerosTail(const COpenOptions &op, UInt64 offset)
{
  if (!op.stream)
    return S_OK;
  RINOK(InStream_SeekSet(op.stream, offset))
  const UInt32 kBufSize = 1 << 11;
  Byte buf[kBufSize];
  
  for (;;)
  {
    UInt32 processed = 0;
    RINOK(op.stream->Read(buf, kBufSize, &processed))
    if (processed == 0)
    {
      // ErrorInfo.NonZerosTail = false;
      ErrorInfo.IgnoreTail = true;
      return S_OK;
    }
    for (size_t i = 0; i < processed; i++)
    {
      if (buf[i] != 0)
      {
        // ErrorInfo.IgnoreTail = false;
        // ErrorInfo.NonZerosTail = true;
        return S_OK;
      }
    }
  }
}



#ifndef Z7_SFX

Z7_CLASS_IMP_COM_2(
  CExtractCallback_To_OpenCallback
  , IArchiveExtractCallback
  , ICompressProgressInfo
)
  Z7_IFACE_COM7_IMP(IProgress)
public:
  CMyComPtr<IArchiveOpenCallback> Callback;
  UInt64 Files;
  UInt64 Offset;

  void Init(IArchiveOpenCallback *callback)
  {
    Callback = callback;
    Files = 0;
    Offset = 0;
  }
};

Z7_COM7F_IMF(CExtractCallback_To_OpenCallback::SetTotal(UInt64 /* size */))
{
  return S_OK;
}

Z7_COM7F_IMF(CExtractCallback_To_OpenCallback::SetCompleted(const UInt64 * /* completeValue */))
{
  return S_OK;
}

Z7_COM7F_IMF(CExtractCallback_To_OpenCallback::SetRatioInfo(const UInt64 *inSize, const UInt64 * /* outSize */))
{
  if (Callback)
  {
    UInt64 value = Offset;
    if (inSize)
      value += *inSize;
    return Callback->SetCompleted(&Files, &value);
  }
  return S_OK;
}

Z7_COM7F_IMF(CExtractCallback_To_OpenCallback::GetStream(UInt32 /* index */, ISequentialOutStream **outStream, Int32 /* askExtractMode */))
{
  *outStream = NULL;
  return S_OK;
}

Z7_COM7F_IMF(CExtractCallback_To_OpenCallback::PrepareOperation(Int32 /* askExtractMode */))
{
  return S_OK;
}

Z7_COM7F_IMF(CExtractCallback_To_OpenCallback::SetOperationResult(Int32 /* operationResult */))
{
  return S_OK;
}


static HRESULT OpenArchiveSpec(IInArchive *archive, bool needPhySize,
    IInStream *stream, const UInt64 *maxCheckStartPosition,
    IArchiveOpenCallback *openCallback,
    IArchiveExtractCallback *extractCallback)
{
  /*
  if (needPhySize)
  {
    Z7_DECL_CMyComPtr_QI_FROM(
        IArchiveOpen2,
        open2, archive)
    if (open2)
      return open2->ArcOpen2(stream, kOpenFlags_RealPhySize, openCallback);
  }
  */
  RINOK(archive->Open(stream, maxCheckStartPosition, openCallback))
  if (needPhySize)
  {
    bool phySize_Defined = false;
    UInt64 phySize = 0;
    RINOK(Archive_GetArcProp_UInt(archive, kpidPhySize, phySize, phySize_Defined))
    if (phySize_Defined)
      return S_OK;

    bool phySizeCantBeDetected = false;
    RINOK(Archive_GetArcProp_Bool(archive, kpidPhySizeCantBeDetected, phySizeCantBeDetected))

    if (!phySizeCantBeDetected)
    {
      PRF(printf("\n-- !phySize_Defined after Open, call archive->Extract()"));
      // It's for bzip2/gz and some xz archives, where Open operation doesn't know phySize.
      // But the Handler will know phySize after full archive testing.
      RINOK(archive->Extract(NULL, (UInt32)(Int32)-1, BoolToInt(true), extractCallback))
      PRF(printf("\n-- OK"));
    }
  }
  return S_OK;
}



static int FindFormatForArchiveType(CCodecs *codecs, CIntVector orderIndices, const char *name)
{
  FOR_VECTOR (i, orderIndices)
  {
    int oi = orderIndices[i];
    if (oi >= 0)
      if (StringsAreEqualNoCase_Ascii(codecs->Formats[(unsigned)oi].Name, name))
        return (int)i;
  }
  return -1;
}

#endif

HRESULT CArc::OpenStream2(const COpenOptions &op)
{
  // fprintf(stdout, "\nOpen: %S", Path); fflush(stdout);

  Archive.Release();
  GetRawProps.Release();
  GetRootProps.Release();

  ErrorInfo.ClearErrors();
  ErrorInfo.ErrorFormatIndex = -1;

  IsParseArc = false;
  ArcStreamOffset = 0;
  
  // OutputDebugStringA("1");
  // OutputDebugStringW(Path);

  const UString fileName = ExtractFileNameFromPath(Path);
  UString extension;
  {
    const int dotPos = fileName.ReverseFind_Dot();
    if (dotPos >= 0)
      extension = fileName.Ptr((unsigned)(dotPos + 1));
  }
  
  CIntVector orderIndices;
  
  bool searchMarkerInHandler = false;
  #ifdef Z7_SFX
    searchMarkerInHandler = true;
  #endif

  CBoolArr isMainFormatArr(op.codecs->Formats.Size());
  {
    FOR_VECTOR(i, op.codecs->Formats)
      isMainFormatArr[i] = false;
  }

  const UInt64 maxStartOffset =
      op.openType.MaxStartOffset_Defined ?
      op.openType.MaxStartOffset :
      kMaxCheckStartPosition;

  #ifndef Z7_SFX
  bool isUnknownExt = false;
  #endif

  #ifndef Z7_SFX
  bool isForced = false;
  #endif

  unsigned numMainTypes = 0;
  const int formatIndex = op.openType.FormatIndex;

  if (formatIndex >= 0)
  {
    #ifndef Z7_SFX
    isForced = true;
    #endif
    orderIndices.Add(formatIndex);
    numMainTypes = 1;
    isMainFormatArr[(unsigned)formatIndex] = true;

    searchMarkerInHandler = true;
  }
  else
  {
    unsigned numFinded = 0;
    #ifndef Z7_SFX
    bool isPrearcExt = false;
    #endif
    
    {
      #ifndef Z7_SFX
      
      bool isZip = false;
      bool isRar = false;
      
      const wchar_t c = extension[0];
      if (c == 'z' || c == 'Z' || c == 'r' || c == 'R')
      {
        bool isNumber = false;
        for (unsigned k = 1;; k++)
        {
          const wchar_t d = extension[k];
          if (d == 0)
            break;
          if (d < '0' || d > '9')
          {
            isNumber = false;
            break;
          }
          isNumber = true;
        }
        if (isNumber)
        {
          if (c == 'z' || c == 'Z')
            isZip = true;
          else
            isRar = true;
        }
      }
      
      #endif

      FOR_VECTOR (i, op.codecs->Formats)
      {
        const CArcInfoEx &ai = op.codecs->Formats[i];

        if (IgnoreSplit || !op.openType.CanReturnArc)
          if (ai.Is_Split())
            continue;
        if (op.excludedFormats->FindInSorted((int)i) >= 0)
          continue;

        #ifndef Z7_SFX
        if (IsPreArcFormat(ai))
          isPrearcExt = true;
        #endif

        if (ai.FindExtension(extension) >= 0
            #ifndef Z7_SFX
            || (isZip && ai.Is_Zip())
            || (isRar && ai.Is_Rar())
            #endif
            )
        {
          // PrintNumber("orderIndices.Insert", i);
          orderIndices.Insert(numFinded++, (int)i);
          isMainFormatArr[i] = true;
        }
        else
          orderIndices.Add((int)i);
      }
    }
  
    if (!op.stream)
    {
      if (numFinded != 1)
        return E_NOTIMPL;
      orderIndices.DeleteFrom(1);
    }
    // PrintNumber("numFinded", numFinded );

    /*
    if (op.openOnlySpecifiedByExtension)
    {
      if (numFinded != 0 && !IsExeExt(extension))
        orderIndices.DeleteFrom(numFinded);
    }
    */

    #ifndef Z7_SFX

      if (op.stream && orderIndices.Size() >= 2)
      {
        RINOK(InStream_SeekToBegin(op.stream))
        CByteBuffer byteBuffer;
        CIntVector orderIndices2;
        if (numFinded == 0 || IsExeExt(extension))
        {
          // signature search was here
        }
        else if (extension.IsEqualTo("000") || extension.IsEqualTo("001"))
        {
          const int i = FindFormatForArchiveType(op.codecs, orderIndices, "rar");
          if (i >= 0)
          {
            const size_t kBufSize = (1 << 10);
            byteBuffer.Alloc(kBufSize);
            size_t processedSize = kBufSize;
            RINOK(ReadStream(op.stream, byteBuffer, &processedSize))
            if (processedSize >= 16)
            {
              const Byte *buf = byteBuffer;
              const Byte kRarHeader[] = { 0x52 , 0x61, 0x72, 0x21, 0x1a, 0x07, 0x00 };
              if (TestSignature(buf, kRarHeader, 7) && buf[9] == 0x73 && (buf[10] & 1) != 0)
              {
                orderIndices2.Add(orderIndices[(unsigned)i]);
                orderIndices[(unsigned)i] = -1;
                if (i >= (int)numFinded)
                  numFinded++;
              }
            }
          }
        }
        else
        {
          const size_t kBufSize = (1 << 10);
          byteBuffer.Alloc(kBufSize);
          size_t processedSize = kBufSize;
          RINOK(ReadStream(op.stream, byteBuffer, &processedSize))
          if (processedSize == 0)
            return S_FALSE;
          
          /*
          check type order:
            0) matched_extension && Backward
            1) matched_extension && (no_signuature || SignatureOffset != 0)
            2) matched_extension && (matched_signature)
            // 3) no signuature
            // 4) matched signuature
          */
          // we move index from orderIndices to orderIndices2 for priority handlers.

          for (unsigned i = 0; i < numFinded; i++)
          {
            const int index = orderIndices[i];
            if (index < 0)
              continue;
            const CArcInfoEx &ai = op.codecs->Formats[(unsigned)index];
            if (ai.Flags_BackwardOpen())
            {
              // backward doesn't need start signatures
              orderIndices2.Add(index);
              orderIndices[i] = -1;
            }
          }

          MakeCheckOrder(op.codecs, orderIndices, numFinded, orderIndices2, NULL, 0);
          MakeCheckOrder(op.codecs, orderIndices, numFinded, orderIndices2, byteBuffer, processedSize);
          // MakeCheckOrder(op.codecs, orderIndices, orderIndices.Size(), orderIndices2, NULL, 0);
          // MakeCheckOrder(op.codecs, orderIndices, orderIndices.Size(), orderIndices2, byteBuffer, processedSize);
        }
      
        FOR_VECTOR (i, orderIndices)
        {
          const int val = orderIndices[i];
          if (val != -1)
            orderIndices2.Add(val);
        }
        orderIndices = orderIndices2;
      }
      
      if (orderIndices.Size() >= 2)
      {
        const int iIso = FindFormatForArchiveType(op.codecs, orderIndices, "iso");
        const int iUdf = FindFormatForArchiveType(op.codecs, orderIndices, "udf");
        if (iUdf > iIso && iIso >= 0)
        {
          const int isoIndex = orderIndices[(unsigned)iIso];
          const int udfIndex = orderIndices[(unsigned)iUdf];
          orderIndices[(unsigned)iUdf] = isoIndex;
          orderIndices[(unsigned)iIso] = udfIndex;
        }
      }

      numMainTypes = numFinded;
      isUnknownExt = (numMainTypes == 0) || isPrearcExt;

    #else // Z7_SFX

      numMainTypes = orderIndices.Size();

      // we need correct numMainTypes for mutlivolume SFX (if some volume is missing)
      if (numFinded != 0)
        numMainTypes = numFinded;
    
    #endif
  }

  UInt64 fileSize = 0;
  if (op.stream)
  {
    RINOK(InStream_GetSize_SeekToBegin(op.stream, fileSize))
  }
  FileSize = fileSize;


  #ifndef Z7_SFX

  CBoolArr skipFrontalFormat(op.codecs->Formats.Size());
  {
    FOR_VECTOR(i, op.codecs->Formats)
      skipFrontalFormat[i] = false;
  }
  
  #endif

  const COpenType &mode = op.openType;

  
  

  
  if (mode.CanReturnArc)
  {
    // ---------- OPEN main type by extenssion ----------
  
    unsigned numCheckTypes = orderIndices.Size();
    if (formatIndex >= 0)
      numCheckTypes = numMainTypes;
    
    for (unsigned i = 0; i < numCheckTypes; i++)
    {
      FormatIndex = orderIndices[i];

      // orderIndices[] item cannot be negative here
      
      bool exactOnly = false;

      #ifndef Z7_SFX
    
      const CArcInfoEx &ai = op.codecs->Formats[(unsigned)FormatIndex];
      // OutputDebugStringW(ai.Name);
      if (i >= numMainTypes)
      {
        // here we allow mismatched extension only for backward handlers
        if (!ai.Flags_BackwardOpen()
            // && !ai.Flags_PureStartOpen()
            )
          continue;
        exactOnly = true;
      }

      #endif
      
      // Some handlers do not set total bytes. So we set it here
      if (op.callback)
        RINOK(op.callback->SetTotal(NULL, &fileSize))

      if (op.stream)
      {
        RINOK(InStream_SeekToBegin(op.stream))
      }
      
      CMyComPtr<IInArchive> archive;
      
      RINOK(PrepareToOpen(op, (unsigned)FormatIndex, archive))
      if (!archive)
        continue;
      
      HRESULT result;
      if (op.stream)
      {
        UInt64 searchLimit = (!exactOnly && searchMarkerInHandler) ? maxStartOffset: 0;
        result = archive->Open(op.stream, &searchLimit, op.callback);
      }
      else
      {
        CMyComPtr<IArchiveOpenSeq> openSeq;
        archive.QueryInterface(IID_IArchiveOpenSeq, (void **)&openSeq);
        if (!openSeq)
          return E_NOTIMPL;
        result = openSeq->OpenSeq(op.seqStream);
      }
      
      RINOK(ReadBasicProps(archive, 0, result))
      
      if (result == S_FALSE)
      {
        bool isArc = ErrorInfo.IsArc_After_NonOpen();

        #ifndef Z7_SFX
        // if it's archive, we allow another open attempt for parser
        if (!mode.CanReturnParser || !isArc)
          skipFrontalFormat[(unsigned)FormatIndex] = true;
        #endif
        
        if (exactOnly)
          continue;
        
        if (i == 0 && numMainTypes == 1)
        {
          // we set NonOpenErrorInfo, only if there is only one main format (defined by extension).
          ErrorInfo.ErrorFormatIndex = FormatIndex;
          NonOpen_ErrorInfo = ErrorInfo;
       
          if (!mode.CanReturnParser && isArc)
          {
            // if (formatIndex < 0 && !searchMarkerInHandler)
            {
              // if bad archive was detected, we don't need additional open attempts
              #ifndef Z7_SFX
              if (!IsPreArcFormat(ai) /* || !mode.SkipSfxStub */)
              #endif
                return S_FALSE;
            }
          }
        }
        
        /*
        #ifndef Z7_SFX
        if (IsExeExt(extension) || ai.Flags_PreArc())
        {
        // openOnlyFullArc = false;
        // canReturnTailArc = true;
        // limitSignatureSearch = true;
        }
        #endif
        */
        
        continue;
      }
      
      RINOK(result)
      
      #ifndef Z7_SFX

      bool isMainFormat = isMainFormatArr[(unsigned)FormatIndex];
      const COpenSpecFlags &specFlags = mode.GetSpec(isForced, isMainFormat, isUnknownExt);

      bool thereIsTail = ErrorInfo.ThereIsTail;
      if (thereIsTail && mode.ZerosTailIsAllowed)
      {
        RINOK(CheckZerosTail(op, (UInt64)(Offset + (Int64)PhySize)))
        if (ErrorInfo.IgnoreTail)
          thereIsTail = false;
      }

      if (Offset > 0)
      {
        if (exactOnly
            || !searchMarkerInHandler
            || !specFlags.CanReturn_NonStart()
            || (mode.MaxStartOffset_Defined && (UInt64)Offset > mode.MaxStartOffset))
          continue;
      }
      if (thereIsTail)
      {
        if (Offset > 0)
        {
          if (!specFlags.CanReturnMid)
            continue;
        }
        else if (!specFlags.CanReturnFrontal)
          continue;
      }

      if (Offset > 0 || thereIsTail)
      {
        if (formatIndex < 0)
        {
          if (IsPreArcFormat(ai))
          {
            // openOnlyFullArc = false;
            // canReturnTailArc = true;
            /*
            if (mode.SkipSfxStub)
            limitSignatureSearch = true;
            */
            // if (mode.SkipSfxStub)
            {
              // skipFrontalFormat[FormatIndex] = true;
              continue;
            }
          }
        }
      }
     
      #endif

      Archive = archive;
      return S_OK;
    }
  }

  

  #ifndef Z7_SFX

  if (!op.stream)
    return S_FALSE;

  if (formatIndex >= 0 && !mode.CanReturnParser)
  {
    if (mode.MaxStartOffset_Defined)
    {
      if (mode.MaxStartOffset == 0)
        return S_FALSE;
    }
    else
    {
      const CArcInfoEx &ai = op.codecs->Formats[(unsigned)formatIndex];
      if (ai.FindExtension(extension) >= 0)
      {
        if (ai.Flags_FindSignature() && searchMarkerInHandler)
          return S_FALSE;
      }
    }
  }

  NArchive::NParser::CHandler *handlerSpec = new NArchive::NParser::CHandler;
  CMyComPtr<IInArchive> handler = handlerSpec;

  CExtractCallback_To_OpenCallback *extractCallback_To_OpenCallback_Spec = new CExtractCallback_To_OpenCallback;
  CMyComPtr<IArchiveExtractCallback> extractCallback_To_OpenCallback = extractCallback_To_OpenCallback_Spec;
  extractCallback_To_OpenCallback_Spec->Init(op.callback);

  {
    // ---------- Check all possible START archives ----------
    // this code is better for full file archives than Parser's code.

    CByteBuffer byteBuffer;
    bool endOfFile = false;
    size_t processedSize;
    {
      size_t bufSize = 1 << 20; // it must be larger than max signature offset or IsArcFunc offset ((1 << 19) + x for UDF)
      if (bufSize > fileSize)
      {
        bufSize = (size_t)fileSize;
        endOfFile = true;
      }
      byteBuffer.Alloc(bufSize);
      RINOK(InStream_SeekToBegin(op.stream))
      processedSize = bufSize;
      RINOK(ReadStream(op.stream, byteBuffer, &processedSize))
      if (processedSize == 0)
        return S_FALSE;
      if (processedSize < bufSize)
        endOfFile = true;
    }
    CUIntVector sortedFormats;

    unsigned i;

    int splitIndex = -1;

    for (i = 0; i < orderIndices.Size(); i++)
    {
      // orderIndices[] item cannot be negative here
      unsigned form = (unsigned)orderIndices[i];
      if (skipFrontalFormat[form])
        continue;
      
      const CArcInfoEx &ai = op.codecs->Formats[form];
      
      if (ai.Is_Split())
      {
        splitIndex = (int)form;
        continue;
      }

      if (ai.Flags_ByExtOnlyOpen())
        continue;

      if (ai.IsArcFunc)
      {
        UInt32 isArcRes = ai.IsArcFunc(byteBuffer, processedSize);
        if (isArcRes == k_IsArc_Res_NO)
          continue;
        if (isArcRes == k_IsArc_Res_NEED_MORE && endOfFile)
          continue;
        // if (isArcRes == k_IsArc_Res_YES_LOW_PROB) continue;
        sortedFormats.Insert(0, form);
        continue;
      }

      const bool isNewStyleSignature = IsNewStyleSignature(ai);
      bool needCheck = !isNewStyleSignature
          || ai.Signatures.IsEmpty()
          || ai.Flags_PureStartOpen()
          || ai.Flags_StartOpen()
          || ai.Flags_BackwardOpen();
    
      if (isNewStyleSignature && !ai.Signatures.IsEmpty())
      {
        unsigned k;
        for (k = 0; k < ai.Signatures.Size(); k++)
        {
          const CByteBuffer &sig = ai.Signatures[k];
          if (processedSize < ai.SignatureOffset + sig.Size())
          {
            if (!endOfFile)
              needCheck = true;
          }
          else if (TestSignature(sig, byteBuffer + ai.SignatureOffset, sig.Size()))
            break;
        }
        if (k != ai.Signatures.Size())
        {
          sortedFormats.Insert(0, form);
          continue;
        }
      }
      if (needCheck)
        sortedFormats.Add(form);
    }

    if (splitIndex >= 0)
      sortedFormats.Insert(0, (unsigned)splitIndex);

    for (i = 0; i < sortedFormats.Size(); i++)
    {
      FormatIndex = (int)sortedFormats[i];
      const CArcInfoEx &ai = op.codecs->Formats[(unsigned)FormatIndex];

      if (op.callback)
        RINOK(op.callback->SetTotal(NULL, &fileSize))

      RINOK(InStream_SeekToBegin(op.stream))

      CMyComPtr<IInArchive> archive;
      RINOK(PrepareToOpen(op, (unsigned)FormatIndex, archive))
      if (!archive)
        continue;
      
      PRF(printf("\nSorted Open %S", (const wchar_t *)ai.Name));
      HRESULT result;
      {
        UInt64 searchLimit = 0;
        /*
        if (mode.CanReturnArc)
          result = archive->Open(op.stream, &searchLimit, op.callback);
        else
        */
        // if (!CanReturnArc), it's ParserMode, and we need phy size
        result = OpenArchiveSpec(archive,
            !mode.CanReturnArc, // needPhySize
            op.stream, &searchLimit, op.callback, extractCallback_To_OpenCallback);
      }
      
      if (result == S_FALSE)
      {
        skipFrontalFormat[(unsigned)FormatIndex] = true;
        // FIXME: maybe we must use LenIsUnknown.
        // printf("  OpenForSize Error");
        continue;
      }
      RINOK(result)

      RINOK(ReadBasicProps(archive, 0, result))

      if (Offset > 0)
      {
        continue; // good handler doesn't return such Offset > 0
        // but there are some cases like false prefixed PK00 archive, when
        // we can support it?
      }

      NArchive::NParser::CParseItem pi;
      pi.Offset = (UInt64)Offset;
      pi.Size = AvailPhySize;
      
      // bool needScan = false;

      if (!PhySize_Defined)
      {
        // it's for Z format
        pi.LenIsUnknown = true;
        // needScan = true;
        // phySize = arcRem;
        // nextNeedCheckStartOpen = false;
      }

      /*
      if (OkPhySize_Defined)
        pi.OkSize = pi.OkPhySize;
      else
        pi.OkSize = pi.Size;
      */

      pi.NormalizeOffset();
      // printf("  phySize = %8d", (unsigned)phySize);


      if (mode.CanReturnArc)
      {
        const bool isMainFormat = isMainFormatArr[(unsigned)FormatIndex];
        const COpenSpecFlags &specFlags = mode.GetSpec(isForced, isMainFormat, isUnknownExt);
        bool openCur = false;

        if (!ErrorInfo.ThereIsTail)
          openCur = true;
        else
        {
          if (mode.ZerosTailIsAllowed)
          {
            RINOK(CheckZerosTail(op, (UInt64)(Offset + (Int64)PhySize)))
            if (ErrorInfo.IgnoreTail)
              openCur = true;
          }
          if (!openCur)
          {
            openCur = specFlags.CanReturnFrontal;
            if (formatIndex < 0) // format is not forced
            {
              if (IsPreArcFormat(ai))
              {
                // if (mode.SkipSfxStub)
                {
                  openCur = false;
                }
              }
            }
          }
        }
        
        if (openCur)
        {
          InStream = op.stream;
          Archive = archive;
          return S_OK;
        }
      }
        
      skipFrontalFormat[(unsigned)FormatIndex] = true;


      // if (!mode.CanReturnArc)
      /*
      if (!ErrorInfo.ThereIsTail)
          continue;
      */
      if (pi.Offset == 0 && !pi.LenIsUnknown && pi.Size >= FileSize)
        continue;

      // printf("\nAdd offset = %d", (int)pi.Offset);
      RINOK(ReadParseItemProps(archive, ai, pi))
      handlerSpec->AddItem(pi);
    }
  }

  

  
  
  // ---------- PARSER ----------

  CUIntVector arc2sig; // formatIndex to signatureIndex
  CUIntVector sig2arc; // signatureIndex to formatIndex;
  {
    unsigned sum = 0;
    FOR_VECTOR (i, op.codecs->Formats)
    {
      arc2sig.Add(sum);
      const CObjectVector<CByteBuffer> &sigs = op.codecs->Formats[i].Signatures;
      sum += sigs.Size();
      FOR_VECTOR (k, sigs)
        sig2arc.Add(i);
    }
  }
  
  {
    const size_t kBeforeSize = 1 << 16;
    const size_t kAfterSize  = 1 << 20;
    const size_t kBufSize = 1 << 22; // it must be more than kBeforeSize + kAfterSize

    const UInt32 kNumVals = (UInt32)1 << (kNumHashBytes * 8);
    CByteArr hashBuffer(kNumVals);
    Byte *hash = hashBuffer;
    memset(hash, 0xFF, kNumVals);
    Byte prevs[256];
    memset(prevs, 0xFF, sizeof(prevs));
    if (sig2arc.Size() >= 0xFF)
      return S_FALSE;

    CUIntVector difficultFormats;
    CBoolArr difficultBools(256);
    {
      for (unsigned i = 0; i < 256; i++)
        difficultBools[i] = false;
    }

    bool thereAreHandlersForSearch = false;

    // UInt32 maxSignatureEnd = 0;
    
    FOR_VECTOR (i, orderIndices)
    {
      int index = orderIndices[i];
      if (index < 0)
        continue;
      const CArcInfoEx &ai = op.codecs->Formats[(unsigned)index];
      if (ai.Flags_ByExtOnlyOpen())
        continue;
      bool isDifficult = false;
      // if (ai.Version < 0x91F) // we don't use parser with old DLL (before 9.31)
      if (!ai.NewInterface)
        isDifficult = true;
      else
      {
        if (ai.Flags_StartOpen())
          isDifficult = true;
        FOR_VECTOR (k, ai.Signatures)
        {
          const CByteBuffer &sig = ai.Signatures[k];
          /*
          UInt32 signatureEnd = ai.SignatureOffset + (UInt32)sig.Size();
          if (maxSignatureEnd < signatureEnd)
            maxSignatureEnd = signatureEnd;
          */
          if (sig.Size() < kNumHashBytes)
          {
            isDifficult = true;
            continue;
          }
          thereAreHandlersForSearch = true;
          UInt32 v = HASH_VAL(sig);
          unsigned sigIndex = arc2sig[(unsigned)index] + k;
          prevs[sigIndex] = hash[v];
          hash[v] = (Byte)sigIndex;
        }
      }
      if (isDifficult)
      {
        difficultFormats.Add((unsigned)index);
        difficultBools[(unsigned)index] = true;
      }
    }
    
    if (!thereAreHandlersForSearch)
    {
      // openOnlyFullArc = true;
      // canReturnTailArc = true;
    }
    
    RINOK(InStream_SeekToBegin(op.stream))

    CLimitedCachedInStream *limitedStreamSpec = new CLimitedCachedInStream;
    CMyComPtr<IInStream> limitedStream = limitedStreamSpec;
    limitedStreamSpec->SetStream(op.stream);

    CArchiveOpenCallback_Offset *openCallback_Offset_Spec = NULL;
    CMyComPtr<IArchiveOpenCallback> openCallback_Offset;
    if (op.callback)
    {
      openCallback_Offset_Spec = new CArchiveOpenCallback_Offset;
      openCallback_Offset = openCallback_Offset_Spec;
      openCallback_Offset_Spec->Callback = op.callback;
      openCallback_Offset_Spec->Callback.QueryInterface(IID_IArchiveOpenVolumeCallback, &openCallback_Offset_Spec->OpenVolumeCallback);
      #ifndef Z7_NO_CRYPTO
      openCallback_Offset_Spec->Callback.QueryInterface(IID_ICryptoGetTextPassword, &openCallback_Offset_Spec->GetTextPassword);
      #endif
    }

    if (op.callback)
      RINOK(op.callback->SetTotal(NULL, &fileSize))
  
    CByteBuffer &byteBuffer = limitedStreamSpec->Buffer;
    byteBuffer.Alloc(kBufSize);

    UInt64 callbackPrev = 0;
    bool needCheckStartOpen = true; // = true, if we need to test all archives types for current pos.

    bool endOfFile = false;
    UInt64 bufPhyPos = 0;
    size_t bytesInBuf = 0;
    // UInt64 prevPos = 0;
    
    // ---------- Main Scan Loop ----------

    UInt64 pos = 0;

    if (!mode.EachPos && handlerSpec->_items.Size() == 1)
    {
      NArchive::NParser::CParseItem &pi = handlerSpec->_items[0];
      if (!pi.LenIsUnknown && pi.Offset == 0)
        pos = pi.Size;
    }

    for (;;)
    {
      // printf("\nPos = %d", (int)pos);
      UInt64 posInBuf = pos - bufPhyPos;
      
      // if (pos > ((UInt64)1 << 35)) break;
      
      if (!endOfFile)
      {
        if (bytesInBuf < kBufSize)
        {
          size_t processedSize = kBufSize - bytesInBuf;
          // printf("\nRead ask = %d", (unsigned)processedSize);
          UInt64 seekPos = bufPhyPos + bytesInBuf;
          RINOK(InStream_SeekSet(op.stream, bufPhyPos + bytesInBuf))
          RINOK(ReadStream(op.stream, byteBuffer.NonConstData() + bytesInBuf, &processedSize))
          // printf("   processed = %d", (unsigned)processedSize);
          if (processedSize == 0)
          {
            fileSize = seekPos;
            endOfFile = true;
          }
          else
          {
            bytesInBuf += processedSize;
            limitedStreamSpec->SetCache(processedSize, (size_t)bufPhyPos);
          }
          continue;
        }
        
        if (bytesInBuf < posInBuf)
        {
          UInt64 skipSize = posInBuf - bytesInBuf;
          if (skipSize <= kBeforeSize)
          {
            size_t keepSize = (size_t)(kBeforeSize - skipSize);
            // printf("\nmemmove skip = %d", (int)keepSize);
            memmove(byteBuffer, byteBuffer.ConstData() + bytesInBuf - keepSize, keepSize);
            bytesInBuf = keepSize;
            bufPhyPos = pos - keepSize;
            continue;
          }
          // printf("\nSkip %d", (int)(skipSize - kBeforeSize));
          // RINOK(op.stream->Seek(skipSize - kBeforeSize, STREAM_SEEK_CUR, NULL));
          bytesInBuf = 0;
          bufPhyPos = pos - kBeforeSize;
          continue;
        }
        
        if (bytesInBuf - posInBuf < kAfterSize)
        {
          size_t beg = (size_t)posInBuf - kBeforeSize;
          // printf("\nmemmove for after beg = %d", (int)beg);
          memmove(byteBuffer, byteBuffer.ConstData() + beg, bytesInBuf - beg);
          bufPhyPos += beg;
          bytesInBuf -= beg;
          continue;
        }
      }

      if (bytesInBuf <= (size_t)posInBuf)
        break;

      bool useOffsetCallback = false;
      if (openCallback_Offset)
      {
        openCallback_Offset_Spec->Files = handlerSpec->_items.Size();
        openCallback_Offset_Spec->Offset = pos;

        useOffsetCallback = (!op.openType.CanReturnArc || handlerSpec->_items.Size() > 1);
 
        if (pos >= callbackPrev + (1 << 23))
        {
          RINOK(openCallback_Offset->SetCompleted(NULL, NULL))
          callbackPrev = pos;
        }
      }

      {
        UInt64 endPos = bufPhyPos + bytesInBuf;
        if (fileSize < endPos)
        {
          FileSize = fileSize; // why ????
          fileSize = endPos;
        }
      }

      const size_t availSize = bytesInBuf - (size_t)posInBuf;
      if (availSize < kNumHashBytes)
        break;
      size_t scanSize = availSize -
          ((availSize >= kAfterSize) ? kAfterSize : kNumHashBytes);
  
      {
        /*
        UInt64 scanLimit = openOnlyFullArc ?
            maxSignatureEnd :
            op.openType.ScanSize + maxSignatureEnd;
        */
        if (!mode.CanReturnParser)
        {
          if (pos > maxStartOffset)
            break;
          UInt64 remScan = maxStartOffset - pos;
          if (scanSize > remScan)
            scanSize = (size_t)remScan;
        }
      }

      scanSize++;

      const Byte *buf = byteBuffer.ConstData() + (size_t)posInBuf;
      const Byte *bufLimit = buf + scanSize;
      size_t ppp = 0;
      
      if (!needCheckStartOpen)
      {
        for (; buf < bufLimit && hash[HASH_VAL(buf)] == 0xFF; buf++);
        ppp = (size_t)(buf - (byteBuffer.ConstData() + (size_t)posInBuf));
        pos += ppp;
        if (buf == bufLimit)
          continue;
      }
      
      UInt32 v = HASH_VAL(buf);
      bool nextNeedCheckStartOpen = true;
      unsigned i = hash[v];
      unsigned indexOfDifficult = 0;

      // ---------- Open Loop for Current Pos ----------
      bool wasOpen = false;
      
      for (;;)
      {
        unsigned index;
        bool isDifficult;
        if (needCheckStartOpen && indexOfDifficult < difficultFormats.Size())
        {
          index = difficultFormats[indexOfDifficult++];
          isDifficult = true;
        }
        else
        {
          if (i == 0xFF)
            break;
          index = sig2arc[i];
          unsigned sigIndex = i - arc2sig[index];
          i = prevs[i];
          if (needCheckStartOpen && difficultBools[index])
            continue;
          const CArcInfoEx &ai = op.codecs->Formats[index];

          if (pos < ai.SignatureOffset)
            continue;

          /*
          if (openOnlyFullArc)
            if (pos != ai.SignatureOffset)
              continue;
          */
  
          const CByteBuffer &sig = ai.Signatures[sigIndex];

          if (ppp + sig.Size() > availSize
              || !TestSignature(buf, sig, sig.Size()))
            continue;
          // printf("\nSignature OK: %10S %8x %5d", (const wchar_t *)ai.Name, (int)pos, (int)(pos - prevPos));
          // prevPos = pos;
          isDifficult = false;
        }

        const CArcInfoEx &ai = op.codecs->Formats[index];


        if ((isDifficult && pos == 0) || ai.SignatureOffset == pos)
        {
          // we don't check same archive second time */
          if (skipFrontalFormat[index])
            continue;
        }

        UInt64 startArcPos = pos;
        if (!isDifficult)
        {
          if (pos < ai.SignatureOffset)
            continue;
          startArcPos = pos - ai.SignatureOffset;
          /*
          // we don't need the check for Z files
          if (startArcPos < handlerSpec->GetLastEnd())
            continue;
          */
        }
        
        if (ai.IsArcFunc && startArcPos >= bufPhyPos)
        {
          const size_t offsetInBuf = (size_t)(startArcPos - bufPhyPos);
          if (offsetInBuf < bytesInBuf)
          {
            const UInt32 isArcRes = ai.IsArcFunc(byteBuffer.ConstData() + offsetInBuf, bytesInBuf - offsetInBuf);
            if (isArcRes == k_IsArc_Res_NO)
              continue;
            if (isArcRes == k_IsArc_Res_NEED_MORE && endOfFile)
              continue;
            /*
            if (isArcRes == k_IsArc_Res_YES_LOW_PROB)
            {
              // if (pos != ai.SignatureOffset)
              continue;
            }
            */
          }
          // printf("\nIsArc OK: %S", (const wchar_t *)ai.Name);
        }
        
        PRF(printf("\npos = %9I64d : %S", pos, (const wchar_t *)ai.Name));

        const bool isMainFormat = isMainFormatArr[index];
        const COpenSpecFlags &specFlags = mode.GetSpec(isForced, isMainFormat, isUnknownExt);
        
        CMyComPtr<IInArchive> archive;
        RINOK(PrepareToOpen(op, index, archive))
        if (!archive)
          return E_FAIL;
        
        // OutputDebugStringW(ai.Name);
        
        const UInt64 rem = fileSize - startArcPos;
        
        UInt64 arcStreamOffset = 0;

        if (ai.Flags_UseGlobalOffset())
        {
          RINOK(limitedStreamSpec->InitAndSeek(0, fileSize))
          RINOK(InStream_SeekSet(limitedStream, startArcPos))
        }
        else
        {
          RINOK(limitedStreamSpec->InitAndSeek(startArcPos, rem))
          arcStreamOffset = startArcPos;
        }
        
        UInt64 maxCheckStartPosition = 0;
        
        if (openCallback_Offset)
        {
          openCallback_Offset_Spec->Files = handlerSpec->_items.Size();
          openCallback_Offset_Spec->Offset = startArcPos;
        }

        // HRESULT result = archive->Open(limitedStream, &maxCheckStartPosition, openCallback_Offset);
        extractCallback_To_OpenCallback_Spec->Files = 0;
        extractCallback_To_OpenCallback_Spec->Offset = startArcPos;

        HRESULT result = OpenArchiveSpec(archive,
            true, // needPhySize
            limitedStream, &maxCheckStartPosition,
            useOffsetCallback ? (IArchiveOpenCallback *)openCallback_Offset : (IArchiveOpenCallback *)op.callback,
            extractCallback_To_OpenCallback);

        RINOK(ReadBasicProps(archive, ai.Flags_UseGlobalOffset() ? 0 : startArcPos, result))

        bool isOpen = false;
      
        if (result == S_FALSE)
        {
          if (!mode.CanReturnParser)
          {
            if (formatIndex < 0 && ErrorInfo.IsArc_After_NonOpen())
            {
              ErrorInfo.ErrorFormatIndex = (int)index;
              NonOpen_ErrorInfo = ErrorInfo;
              // if archive was detected, we don't need additional open attempts
              return S_FALSE;
            }
            continue;
          }
          if (!ErrorInfo.IsArc_After_NonOpen() || !PhySize_Defined || PhySize == 0)
            continue;
        }
        else
        {
          if (PhySize_Defined && PhySize == 0)
          {
            PRF(printf("  phySize_Defined && PhySize == 0 "));
            // we skip that epmty archive case with unusual unexpected (PhySize == 0) from Code function.
            continue;
          }
          isOpen = true;
          RINOK(result)
          PRF(printf("  OK "));
        }

        // fprintf(stderr, "\n %8X  %S", startArcPos, Path);
        // printf("\nOpen OK: %S", ai.Name);
        
        
        NArchive::NParser::CParseItem pi;
        pi.Offset = startArcPos;

        if (ai.Flags_UseGlobalOffset())
          pi.Offset = (UInt64)Offset;
        else if (Offset != 0)
          return E_FAIL;

        const UInt64 arcRem = FileSize - pi.Offset;
        UInt64 phySize = arcRem;
        const bool phySize_Defined = PhySize_Defined;
        if (phySize_Defined)
        {
          if (pi.Offset + PhySize > FileSize)
          {
            // ErrorInfo.ThereIsTail = true;
            PhySize = FileSize - pi.Offset;
          }
          phySize = PhySize;
        }
        if (phySize == 0 || (UInt64)phySize > ((UInt64)1 << 63))
          return E_FAIL;

        /*
        if (!ai.UseGlobalOffset)
        {
          if (phySize > arcRem)
          {
            ThereIsTail = true;
            phySize = arcRem;
          }
        }
        */
        
        bool needScan = false;

 
        if (isOpen && !phySize_Defined)
        {
          // it's for Z format, or bzip2,gz,xz with phySize that was not detected
          pi.LenIsUnknown = true;
          needScan = true;
          phySize = arcRem;
          nextNeedCheckStartOpen = false;
        }

        pi.Size = phySize;
        /*
        if (OkPhySize_Defined)
          pi.OkSize = OkPhySize;
        */
        pi.NormalizeOffset();
        // printf("  phySize = %8d", (unsigned)phySize);

        /*
        if (needSkipFullArc)
          if (pi.Offset == 0 && phySize_Defined && pi.Size >= fileSize)
            continue;
        */
        if (pi.Offset == 0 && !pi.LenIsUnknown && pi.Size >= FileSize)
        {
          // it's possible for dmg archives
          if (!mode.CanReturnArc)
            continue;
        }

        if (mode.EachPos)
          pos++;
        else if (needScan)
        {
          pos++;
          /*
          if (!OkPhySize_Defined)
            pos++;
          else
            pos = pi.Offset + pi.OkSize;
          */
        }
        else
          pos = pi.Offset + pi.Size;

       
        RINOK(ReadParseItemProps(archive, ai, pi))

        if (pi.Offset < startArcPos && !mode.EachPos /* && phySize_Defined */)
        {
          /* It's for DMG format.
          This code deletes all previous items that are included to current item */
            
          while (!handlerSpec->_items.IsEmpty())
          {
            {
              const NArchive::NParser::CParseItem &back = handlerSpec->_items.Back();
              if (back.Offset < pi.Offset)
                break;
              if (back.Offset + back.Size > pi.Offset + pi.Size)
                break;
            }
            handlerSpec->_items.DeleteBack();
          }
        }
        

        if (isOpen && mode.CanReturnArc && phySize_Defined)
        {
          // if (pi.Offset + pi.Size >= fileSize)
          bool openCur = false;

          bool thereIsTail = ErrorInfo.ThereIsTail;
          if (thereIsTail && mode.ZerosTailIsAllowed)
          {
            RINOK(CheckZerosTail(op, (UInt64)((Int64)arcStreamOffset + Offset + (Int64)PhySize)))
            if (ErrorInfo.IgnoreTail)
              thereIsTail = false;
          }

          if (pi.Offset != 0)
          {
            if (!pi.IsNotArcType)
            {
              if (thereIsTail)
                openCur = specFlags.CanReturnMid;
              else
                openCur = specFlags.CanReturnTail;
            }
          }
          else
          {
            if (!thereIsTail)
              openCur = true;
            else
              openCur = specFlags.CanReturnFrontal;

            if (formatIndex >= -2)
              openCur = true;
          }

          if (formatIndex < 0 && pi.IsSelfExe /* && mode.SkipSfxStub */)
            openCur = false;

          // We open file as SFX, if there is front archive or first archive is "Self Executable"
          if (!openCur && !pi.IsSelfExe && !thereIsTail &&
              (!pi.IsNotArcType || pi.Offset == 0))
          {
            if (handlerSpec->_items.IsEmpty())
            {
              if (specFlags.CanReturnTail)
                openCur = true;
            }
            else if (handlerSpec->_items.Size() == 1)
            {
              if (handlerSpec->_items[0].IsSelfExe)
              {
                if (mode.SpecUnknownExt.CanReturnTail)
                  openCur = true;
              }
            }
          }

          if (openCur)
          {
            InStream = op.stream;
            Archive = archive;
            FormatIndex = (int)index;
            ArcStreamOffset = arcStreamOffset;
            return S_OK;
          }
        }

        /*
        if (openOnlyFullArc)
        {
          ErrorInfo.ClearErrors();
          return S_FALSE;
        }
        */

        pi.FormatIndex = (int)index;

        // printf("\nAdd offset = %d", (int)pi.Offset);
        handlerSpec->AddItem(pi);
        wasOpen = true;
        break;
      }
      // ---------- End of Open Loop for Current Pos ----------
     
      if (!wasOpen)
        pos++;
      needCheckStartOpen = (nextNeedCheckStartOpen && wasOpen);
    }
    // ---------- End of Main Scan Loop ----------

    /*
    if (handlerSpec->_items.Size() == 1)
    {
      const NArchive::NParser::CParseItem &pi = handlerSpec->_items[0];
      if (pi.Size == fileSize && pi.Offset == 0)
      {
        Archive = archive;
        FormatIndex2 = pi.FormatIndex;
        return S_OK;
      }
    }
    */

    if (mode.CanReturnParser)
    {
      bool returnParser = (handlerSpec->_items.Size() == 1); // it's possible if fileSize was not correct at start of parsing
      handlerSpec->AddUnknownItem(fileSize);
      if (handlerSpec->_items.Size() == 0)
        return S_FALSE;
      if (returnParser || handlerSpec->_items.Size() != 1)
      {
        // return S_FALSE;
        handlerSpec->_stream = op.stream;
        Archive = handler;
        ErrorInfo.ClearErrors();
        IsParseArc = true;
        FormatIndex = -1; // It's parser
        Offset = 0;
        return S_OK;
      }
    }
  }

  #endif

  if (!Archive)
    return S_FALSE;
  return S_OK;
}




HRESULT CArc::OpenStream(const COpenOptions &op)
{
  RINOK(OpenStream2(op))
  // PrintNumber("op.formatIndex 3", op.formatIndex);

  if (Archive)
  {
    GetRawProps.Release();
    GetRootProps.Release();
    Archive->QueryInterface(IID_IArchiveGetRawProps, (void **)&GetRawProps);
    Archive->QueryInterface(IID_IArchiveGetRootProps, (void **)&GetRootProps);

    RINOK(Archive_GetArcProp_Bool(Archive, kpidIsTree, IsTree))
    RINOK(Archive_GetArcProp_Bool(Archive, kpidIsDeleted, Ask_Deleted))
    RINOK(Archive_GetArcProp_Bool(Archive, kpidIsAltStream, Ask_AltStream))
    RINOK(Archive_GetArcProp_Bool(Archive, kpidIsAux, Ask_Aux))
    RINOK(Archive_GetArcProp_Bool(Archive, kpidINode, Ask_INode))
    RINOK(Archive_GetArcProp_Bool(Archive, kpidReadOnly, IsReadOnly))

    const UString fileName = ExtractFileNameFromPath(Path);
    UString extension;
    {
      int dotPos = fileName.ReverseFind_Dot();
      if (dotPos >= 0)
        extension = fileName.Ptr((unsigned)(dotPos + 1));
    }
    
    DefaultName.Empty();
    if (FormatIndex >= 0)
    {
      const CArcInfoEx &ai = op.codecs->Formats[(unsigned)FormatIndex];
      if (ai.Exts.Size() == 0)
        DefaultName = GetDefaultName2(fileName, UString(), UString());
      else
      {
        int subExtIndex = ai.FindExtension(extension);
        if (subExtIndex < 0)
          subExtIndex = 0;
        const CArcExtInfo &extInfo = ai.Exts[(unsigned)subExtIndex];
        DefaultName = GetDefaultName2(fileName, extInfo.Ext, extInfo.AddExt);
      }
    }
  }

  return S_OK;
}

#ifdef Z7_SFX

#ifdef _WIN32
  #define k_ExeExt ".exe"
  static const unsigned k_ExeExt_Len = 4;
#else
  #define k_ExeExt ""
  static const unsigned k_ExeExt_Len = 0;
#endif

#endif

HRESULT CArc::OpenStreamOrFile(COpenOptions &op)
{
  CMyComPtr<IInStream> fileStream;
  CMyComPtr<ISequentialInStream> seqStream;
  CInFileStream *fileStreamSpec = NULL;
  
  if (op.stdInMode)
  {
#if 1
    seqStream = new CStdInFileStream;
#else
    if (!CreateStdInStream(seqStream))
      return GetLastError_noZero_HRESULT();
#endif
    op.seqStream = seqStream;
  }
  else if (!op.stream)
  {
    fileStreamSpec = new CInFileStream;
    fileStream = fileStreamSpec;
    Path = filePath;
    if (!fileStreamSpec->Open(us2fs(Path)))
      return GetLastError_noZero_HRESULT();
    op.stream = fileStream;
    #ifdef Z7_SFX
    IgnoreSplit = true;
    #endif
  }

  /*
  if (callback)
  {
    UInt64 fileSize;
    RINOK(InStream_GetSize_SeekToEnd(op.stream, fileSize));
    RINOK(op.callback->SetTotal(NULL, &fileSize))
  }
  */

  HRESULT res = OpenStream(op);
  IgnoreSplit = false;
  
  #ifdef Z7_SFX
  
  if (res != S_FALSE
      || !fileStreamSpec
      || !op.callbackSpec
      || NonOpen_ErrorInfo.IsArc_After_NonOpen())
    return res;
  
  {
    if (filePath.Len() > k_ExeExt_Len
        && StringsAreEqualNoCase_Ascii(filePath.RightPtr(k_ExeExt_Len), k_ExeExt))
    {
      const UString path2 = filePath.Left(filePath.Len() - k_ExeExt_Len);
      FOR_VECTOR (i, op.codecs->Formats)
      {
        const CArcInfoEx &ai = op.codecs->Formats[i];
        if (ai.Is_Split())
          continue;
        UString path3 = path2;
        path3.Add_Dot();
        path3 += ai.GetMainExt(); // "7z"  for SFX.
        Path = path3;
        Path += ".001";
        bool isOk = op.callbackSpec->SetSecondFileInfo(us2fs(Path));
        if (!isOk)
        {
          Path = path3;
          isOk = op.callbackSpec->SetSecondFileInfo(us2fs(Path));
        }
        if (isOk)
        {
          if (fileStreamSpec->Open(us2fs(Path)))
          {
            op.stream = fileStream;
            NonOpen_ErrorInfo.ClearErrors_Full();
            if (OpenStream(op) == S_OK)
              return S_OK;
          }
        }
      }
    }
  }
  
  #endif

  return res;
}

void CArchiveLink::KeepModeForNextOpen()
{
  for (unsigned i = Arcs.Size(); i != 0;)
  {
    i--;
    CMyComPtr<IArchiveKeepModeForNextOpen> keep;
    Arcs[i].Archive->QueryInterface(IID_IArchiveKeepModeForNextOpen, (void **)&keep);
    if (keep)
      keep->KeepModeForNextOpen();
  }
}

HRESULT CArchiveLink::Close()
{
  for (unsigned i = Arcs.Size(); i != 0;)
  {
    i--;
    RINOK(Arcs[i].Close())
  }
  IsOpen = false;
  // ErrorsText.Empty();
  return S_OK;
}

void CArchiveLink::Release()
{
  // NonOpenErrorFormatIndex = -1;
  NonOpen_ErrorInfo.ClearErrors();
  NonOpen_ArcPath.Empty();
  while (!Arcs.IsEmpty())
    Arcs.DeleteBack();
}

/*
void CArchiveLink::Set_ErrorsText()
{
  FOR_VECTOR(i, Arcs)
  {
    const CArc &arc = Arcs[i];
    if (!arc.ErrorFlagsText.IsEmpty())
    {
      if (!ErrorsText.IsEmpty())
        ErrorsText.Add_LF();
      ErrorsText += GetUnicodeString(arc.ErrorFlagsText);
    }
    if (!arc.ErrorMessage.IsEmpty())
    {
      if (!ErrorsText.IsEmpty())
        ErrorsText.Add_LF();
      ErrorsText += arc.ErrorMessage;
    }

    if (!arc.WarningMessage.IsEmpty())
    {
      if (!ErrorsText.IsEmpty())
        ErrorsText.Add_LF();
      ErrorsText += arc.WarningMessage;
    }
  }
}
*/

HRESULT CArchiveLink::Open(COpenOptions &op)
{
  Release();
  if (op.types->Size() >= 32)
    return E_NOTIMPL;
  
  HRESULT resSpec;

  for (;;)
  {
    resSpec = S_OK;

    op.openType = COpenType();
    if (op.types->Size() >= 1)
    {
      COpenType latest;
      if (Arcs.Size() < op.types->Size())
        latest = (*op.types)[op.types->Size() - Arcs.Size() - 1];
      else
      {
        latest = (*op.types)[0];
        if (!latest.Recursive)
          break;
      }
      op.openType = latest;
    }
    else if (Arcs.Size() >= 32)
      break;

    /*
    op.formatIndex = -1;
    if (op.types->Size() >= 1)
    {
      int latest;
      if (Arcs.Size() < op.types->Size())
        latest = (*op.types)[op.types->Size() - Arcs.Size() - 1];
      else
      {
        latest = (*op.types)[0];
        if (latest != -2 && latest != -3)
          break;
      }
      if (latest >= 0)
        op.formatIndex = latest;
      else if (latest == -1 || latest == -2)
      {
        // default
      }
      else if (latest == -3)
        op.formatIndex = -2;
      else
        op.formatIndex = latest + 2;
    }
    else if (Arcs.Size() >= 32)
      break;
    */

    if (Arcs.IsEmpty())
    {
      CArc arc;
      arc.filePath = op.filePath;
      arc.Path = op.filePath;
      arc.SubfileIndex = (UInt32)(Int32)-1;
      HRESULT result = arc.OpenStreamOrFile(op);
      if (result != S_OK)
      {
        if (result == S_FALSE)
        {
          NonOpen_ErrorInfo = arc.NonOpen_ErrorInfo;
          // NonOpenErrorFormatIndex = arc.ErrorFormatIndex;
          NonOpen_ArcPath = arc.Path;
        }
        return result;
      }
      Arcs.Add(arc);
      continue;
    }
    
    // PrintNumber("op.formatIndex 11", op.formatIndex);

    const CArc &arc = Arcs.Back();
    
    if (op.types->Size() > Arcs.Size())
      resSpec = E_NOTIMPL;
    
    UInt32 mainSubfile;
    {
      NCOM::CPropVariant prop;
      RINOK(arc.Archive->GetArchiveProperty(kpidMainSubfile, &prop))
      if (prop.vt == VT_UI4)
        mainSubfile = prop.ulVal;
      else
        break;
      UInt32 numItems;
      RINOK(arc.Archive->GetNumberOfItems(&numItems))
      if (mainSubfile >= numItems)
        break;
    }

  
    CMyComPtr<IInArchiveGetStream> getStream;
    if (arc.Archive->QueryInterface(IID_IInArchiveGetStream, (void **)&getStream) != S_OK || !getStream)
      break;
    
    CMyComPtr<ISequentialInStream> subSeqStream;
    if (getStream->GetStream(mainSubfile, &subSeqStream) != S_OK || !subSeqStream)
      break;
    
    CMyComPtr<IInStream> subStream;
    if (subSeqStream.QueryInterface(IID_IInStream, &subStream) != S_OK || !subStream)
      break;
    
    CArc arc2;
    RINOK(arc.GetItem_Path(mainSubfile, arc2.Path))

    bool zerosTailIsAllowed;
    RINOK(Archive_GetItemBoolProp(arc.Archive, mainSubfile, kpidZerosTailIsAllowed, zerosTailIsAllowed))


    if (op.callback)
    {
      Z7_DECL_CMyComPtr_QI_FROM(
          IArchiveOpenSetSubArchiveName,
          setSubArchiveName, op.callback)
      if (setSubArchiveName)
        setSubArchiveName->SetSubArchiveName(arc2.Path);
    }
    
    arc2.SubfileIndex = mainSubfile;

    // CIntVector incl;
    CIntVector excl;

    COpenOptions op2;
    #ifndef Z7_SFX
    op2.props = op.props;
    #endif
    op2.codecs = op.codecs;
    // op2.types = &incl;
    op2.openType = op.openType;
    op2.openType.ZerosTailIsAllowed = zerosTailIsAllowed;
    op2.excludedFormats = &excl;
    op2.stdInMode = false;
    op2.stream = subStream;
    op2.filePath = arc2.Path;
    op2.callback = op.callback;
    op2.callbackSpec = op.callbackSpec;


    HRESULT result = arc2.OpenStream(op2);
    resSpec = (op.types->Size() == 0 ? S_OK : S_FALSE);
    if (result == S_FALSE)
    {
      NonOpen_ErrorInfo = arc2.ErrorInfo;
      NonOpen_ArcPath = arc2.Path;
      break;
    }
    RINOK(result)
    RINOK(arc.GetItem_MTime(mainSubfile, arc2.MTime))
    Arcs.Add(arc2);
  }
  IsOpen = !Arcs.IsEmpty();
  return resSpec;
}

HRESULT CArchiveLink::Open2(COpenOptions &op, IOpenCallbackUI *callbackUI)
{
  VolumesSize = 0;
  COpenCallbackImp *openCallbackSpec = new COpenCallbackImp;
  CMyComPtr<IArchiveOpenCallback> callback = openCallbackSpec;
  openCallbackSpec->Callback = callbackUI;

  FString prefix, name;
  
  if (!op.stream && !op.stdInMode)
  {
    NFile::NDir::GetFullPathAndSplit(us2fs(op.filePath), prefix, name);
    RINOK(openCallbackSpec->Init2(prefix, name))
  }
  else
  {
    openCallbackSpec->SetSubArchiveName(op.filePath);
  }

  op.callback = callback;
  op.callbackSpec = openCallbackSpec;
  
  HRESULT res = Open(op);

  PasswordWasAsked = openCallbackSpec->PasswordWasAsked;
  // Password = openCallbackSpec->Password;

  RINOK(res)
  // VolumePaths.Add(fs2us(prefix + name));

  FOR_VECTOR (i, openCallbackSpec->FileNames_WasUsed)
  {
    if (openCallbackSpec->FileNames_WasUsed[i])
    {
      VolumePaths.Add(fs2us(prefix) + openCallbackSpec->FileNames[i]);
      VolumesSize += openCallbackSpec->FileSizes[i];
    }
  }
  // VolumesSize = openCallbackSpec->TotalSize;
  return S_OK;
}

HRESULT CArc::ReOpen(const COpenOptions &op, IArchiveOpenCallback *openCallback_Additional)
{
  ErrorInfo.ClearErrors();
  ErrorInfo.ErrorFormatIndex = -1;

  UInt64 fileSize = 0;
  if (op.stream)
  {
    RINOK(InStream_SeekToBegin(op.stream))
    RINOK(InStream_AtBegin_GetSize(op.stream, fileSize))
    // RINOK(InStream_GetSize_SeekToBegin(op.stream, fileSize))
  }
  FileSize = fileSize;

  CMyComPtr<IInStream> stream2;
  Int64 globalOffset = GetGlobalOffset();
  if (globalOffset <= 0)
    stream2 = op.stream;
  else
  {
    CTailInStream *tailStreamSpec = new CTailInStream;
    stream2 = tailStreamSpec;
    tailStreamSpec->Stream = op.stream;
    tailStreamSpec->Offset = (UInt64)globalOffset;
    tailStreamSpec->Init();
    RINOK(tailStreamSpec->SeekToStart())
  }

  // There are archives with embedded STUBs (like ZIP), so we must support signature scanning
  // But for another archives we can use 0 here. So the code can be fixed !!!
  UInt64 maxStartPosition = kMaxCheckStartPosition;
  IArchiveOpenCallback *openCallback = openCallback_Additional;
  if (!openCallback)
    openCallback = op.callback;
  HRESULT res = Archive->Open(stream2, &maxStartPosition, openCallback);
  
  if (res == S_OK)
  {
    RINOK(ReadBasicProps(Archive, (UInt64)globalOffset, res))
    ArcStreamOffset = (UInt64)globalOffset;
    if (ArcStreamOffset != 0)
      InStream = op.stream;
  }
  return res;
}

HRESULT CArchiveLink::Open3(COpenOptions &op, IOpenCallbackUI *callbackUI)
{
  HRESULT res = Open2(op, callbackUI);
  if (callbackUI)
  {
    RINOK(callbackUI->Open_Finished())
  }
  return res;
}

HRESULT CArchiveLink::ReOpen(COpenOptions &op)
{
  if (Arcs.Size() > 1)
    return E_NOTIMPL;

  CObjectVector<COpenType> inc;
  CIntVector excl;

  op.types = &inc;
  op.excludedFormats = &excl;
  op.stdInMode = false;
  op.stream = NULL;
  if (Arcs.Size() == 0) // ???
    return Open2(op, NULL);

  /* if archive is multivolume (unsupported here still)
     COpenCallbackImp object will exist after Open stage. */
  COpenCallbackImp *openCallbackSpec = new COpenCallbackImp;
  CMyComPtr<IArchiveOpenCallback> openCallbackNew = openCallbackSpec;

  openCallbackSpec->Callback = NULL;
  openCallbackSpec->ReOpenCallback = op.callback;
  {
    FString dirPrefix, fileName;
    NFile::NDir::GetFullPathAndSplit(us2fs(op.filePath), dirPrefix, fileName);
    RINOK(openCallbackSpec->Init2(dirPrefix, fileName))
  }


  CInFileStream *fileStreamSpec = new CInFileStream;
  CMyComPtr<IInStream> stream(fileStreamSpec);
  if (!fileStreamSpec->Open(us2fs(op.filePath)))
    return GetLastError_noZero_HRESULT();
  op.stream = stream;

  CArc &arc = Arcs[0];
  const HRESULT res = arc.ReOpen(op, openCallbackNew);

  openCallbackSpec->ReOpenCallback = NULL;
  
  PasswordWasAsked = openCallbackSpec->PasswordWasAsked;
  // Password = openCallbackSpec->Password;
  
  IsOpen = (res == S_OK);
  return res;
}

#ifndef Z7_SFX

bool ParseComplexSize(const wchar_t *s, UInt64 &result);
bool ParseComplexSize(const wchar_t *s, UInt64 &result)
{
  result = 0;
  const wchar_t *end;
  UInt64 number = ConvertStringToUInt64(s, &end);
  if (end == s)
    return false;
  if (*end == 0)
  {
    result = number;
    return true;
  }
  if (end[1] != 0)
    return false;
  unsigned numBits;
  switch (MyCharLower_Ascii(*end))
  {
    case 'b': result = number; return true;
    case 'k': numBits = 10; break;
    case 'm': numBits = 20; break;
    case 'g': numBits = 30; break;
    case 't': numBits = 40; break;
    default: return false;
  }
  if (number >= ((UInt64)1 << (64 - numBits)))
    return false;
  result = number << numBits;
  return true;
}

static bool ParseTypeParams(const UString &s, COpenType &type)
{
  if (s[0] == 0)
    return true;
  if (s[1] == 0)
  {
    switch ((unsigned)(Byte)s[0])
    {
      case 'e': type.EachPos = true; return true;
      case 'a': type.CanReturnArc = true; return true;
      case 'r': type.Recursive = true; return true;
      default: break;
    }
    return false;
  }
  if (s[0] == 's')
  {
    UInt64 result;
    if (!ParseComplexSize(s.Ptr(1), result))
      return false;
    type.MaxStartOffset = result;
    type.MaxStartOffset_Defined = true;
    return true;
  }

  return false;
}

static bool ParseType(CCodecs &codecs, const UString &s, COpenType &type)
{
  int pos2 = s.Find(L':');

  {
  UString name;
  if (pos2 < 0)
  {
    name = s;
    pos2 = (int)s.Len();
  }
  else
  {
    name = s.Left((unsigned)pos2);
    pos2++;
  }

  int index = codecs.FindFormatForArchiveType(name);
  type.Recursive = false;

  if (index < 0)
  {
    if (name[0] == '*')
    {
      if (name[1] != 0)
        return false;
    }
    else if (name[0] == '#')
    {
      if (name[1] != 0)
        return false;
      type.CanReturnArc = false;
      type.CanReturnParser = true;
    }
    else if (name.IsEqualTo_Ascii_NoCase("hash"))
    {
      // type.CanReturnArc = false;
      // type.CanReturnParser = false;
      type.IsHashType = true;
    }
    else
      return false;
  }
  
  type.FormatIndex = index;

  }
 
  for (unsigned i = (unsigned)pos2; i < s.Len();)
  {
    int next = s.Find(L':', i);
    if (next < 0)
      next = (int)s.Len();
    const UString name = s.Mid(i, (unsigned)next - i);
    if (name.IsEmpty())
      return false;
    if (!ParseTypeParams(name, type))
      return false;
    i = (unsigned)next + 1;
  }
  
  return true;
}

bool ParseOpenTypes(CCodecs &codecs, const UString &s, CObjectVector<COpenType> &types)
{
  types.Clear();
  bool isHashType = false;
  for (unsigned pos = 0; pos < s.Len();)
  {
    int pos2 = s.Find(L'.', pos);
    if (pos2 < 0)
      pos2 = (int)s.Len();
    UString name = s.Mid(pos, (unsigned)pos2 - pos);
    if (name.IsEmpty())
      return false;
    COpenType type;
    if (!ParseType(codecs, name, type))
      return false;
    if (isHashType)
      return false;
    if (type.IsHashType)
      isHashType = true;
    types.Add(type);
    pos = (unsigned)pos2 + 1;
  }
  return true;
}

/*
bool IsHashType(const CObjectVector<COpenType> &types)
{
  if (types.Size() != 1)
    return false;
  return types[0].IsHashType;
}
*/


#endif
