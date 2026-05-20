// 7zIn.cpp

#include "StdAfx.h"

#ifdef _WIN32
#include <wchar.h>
#else
#include <ctype.h>
#endif

#include "../../../../C/7zCrc.h"
#include "../../../../C/CpuArch.h"

#include "../../../Common/MyBuffer2.h"
// #include "../../../Common/UTFConvert.h"

#include "../../Common/StreamObjects.h"
#include "../../Common/StreamUtils.h"

#include "7zDecode.h"
#include "7zIn.h"

#define Get16(p) GetUi16(p)
#define Get32(p) GetUi32(p)
#define Get64(p) GetUi64(p)

// define FORMAT_7Z_RECOVERY if you want to recover multivolume archives with empty StartHeader
#ifndef Z7_SFX
#define FORMAT_7Z_RECOVERY
#endif

using namespace NWindows;
using namespace NCOM;

unsigned BoolVector_CountSum(const CBoolVector &v);
Z7_NO_INLINE
unsigned BoolVector_CountSum(const CBoolVector &v)
{
  unsigned sum = 0;
  const unsigned size = v.Size();
  if (size)
  {
    const bool *p = v.ConstData();
    const bool * const lim = p + size;
    do
     if (*p)
       sum++;
    while (++p != lim);
  }
  return sum;
}

static inline bool BoolVector_Item_IsValidAndTrue(const CBoolVector &v, unsigned i)
{
  return i < v.Size() ? v[i] : false;
}

Z7_NO_INLINE
static void BoolVector_Fill_False(CBoolVector &v, unsigned size)
{
  v.ClearAndSetSize(size);
  bool *p = v.NonConstData();
  for (unsigned i = 0; i < size; i++)
    p[i] = false;
}


namespace NArchive {
namespace N7z {

#define k_Scan_NumCoders_MAX 64
#define k_Scan_NumCodersStreams_in_Folder_MAX 64

class CInArchiveException {};
class CUnsupportedFeatureException: public CInArchiveException {};

Z7_ATTR_NORETURN
static void ThrowException() { throw CInArchiveException(); }
Z7_ATTR_NORETURN
static inline void ThrowEndOfData()   { ThrowException(); }
Z7_ATTR_NORETURN
static inline void ThrowUnsupported() { throw CUnsupportedFeatureException(); }
Z7_ATTR_NORETURN
static inline void ThrowIncorrect()   { ThrowException(); }

class CStreamSwitch
{
  CInArchive *_archive;
  bool _needRemove;
  bool _needUpdatePos;
public:
  CStreamSwitch(): _needRemove(false), _needUpdatePos(false) {}
  ~CStreamSwitch() { Remove(); }
  void Remove();
  void Set(CInArchive *archive, const Byte *data, size_t size, bool needUpdatePos);
  void Set(CInArchive *archive, const CByteBuffer &byteBuffer);
  void Set(CInArchive *archive, const CObjectVector<CByteBuffer> *dataVector);
};

void CStreamSwitch::Remove()
{
  if (_needRemove)
  {
    if (_archive->_inByteBack->GetRem() != 0)
      _archive->ThereIsHeaderError = true;
    _archive->DeleteByteStream(_needUpdatePos);
    _needRemove = false;
  }
}

void CStreamSwitch::Set(CInArchive *archive, const Byte *data, size_t size, bool needUpdatePos)
{
  Remove();
  _archive = archive;
  _archive->AddByteStream(data, size);
  _needRemove = true;
  _needUpdatePos = needUpdatePos;
}

void CStreamSwitch::Set(CInArchive *archive, const CByteBuffer &byteBuffer)
{
  Set(archive, byteBuffer, byteBuffer.Size(), false);
}

void CStreamSwitch::Set(CInArchive *archive, const CObjectVector<CByteBuffer> *dataVector)
{
  Remove();
  const Byte external = archive->ReadByte();
  if (external != 0)
  {
    if (!dataVector)
      ThrowIncorrect();
    const CNum dataIndex = archive->ReadNum();
    if (dataIndex >= dataVector->Size())
      ThrowIncorrect();
    Set(archive, (*dataVector)[dataIndex]);
  }
}

void CInArchive::AddByteStream(const Byte *buf, size_t size)
{
  if (_numInByteBufs == kNumBufLevelsMax)
    ThrowIncorrect();
  _inByteBack = &_inByteVector[_numInByteBufs++];
  _inByteBack->Init(buf, size);
}
  

Byte CInByte2::ReadByte()
{
  if (_pos >= _size)
    ThrowEndOfData();
  return _buffer[_pos++];
}

void CInByte2::ReadBytes(Byte *data, size_t size)
{
  if (size == 0)
    return;
  if (size > _size - _pos)
    ThrowEndOfData();
  memcpy(data, _buffer + _pos, size);
  _pos += size;
}

void CInByte2::SkipData(UInt64 size)
{
  if (size > _size - _pos)
    ThrowEndOfData();
  _pos += (size_t)size;
}

void CInByte2::SkipData()
{
  SkipData(ReadNumber());
}

static UInt64 ReadNumberSpec(const Byte *p, size_t size, size_t &processed)
{
  if (size == 0)
  {
    processed = 0;
    return 0;
  }
  
  const unsigned b = *p++;
  size--;
  
  if ((b & 0x80) == 0)
  {
    processed = 1;
    return b;
  }
  
  if (size == 0)
  {
    processed = 0;
    return 0;
  }
  
  UInt64 value = (UInt64)*p;
  p++;
  size--;
  
  for (unsigned i = 1; i < 8; i++)
  {
    const unsigned mask = (unsigned)0x80 >> i;
    if ((b & mask) == 0)
    {
      const UInt64 high = b & (mask - 1);
      value |= (high << (i * 8));
      processed = i + 1;
      return value;
    }
    
    if (size == 0)
    {
      processed = 0;
      return 0;
    }
    
    value |= ((UInt64)*p << (i * 8));
    p++;
    size--;
  }
  
  processed = 9;
  return value;
}

UInt64 CInByte2::ReadNumber()
{
  size_t processed;
  const UInt64 res = ReadNumberSpec(_buffer + _pos, _size - _pos, processed);
  if (processed == 0)
    ThrowEndOfData();
  _pos += processed;
  return res;
}

CNum CInByte2::ReadNum()
{
  /*
  if (_pos < _size)
  {
    Byte val = _buffer[_pos];
    if ((unsigned)val < 0x80)
    {
      _pos++;
      return (unsigned)val;
    }
  }
  */
  const UInt64 value = ReadNumber();
  if (value > kNumMax)
    ThrowUnsupported();
  return (CNum)value;
}

UInt32 CInByte2::ReadUInt32()
{
  if (_pos + 4 > _size)
    ThrowEndOfData();
  const UInt32 res = Get32(_buffer + _pos);
  _pos += 4;
  return res;
}

UInt64 CInByte2::ReadUInt64()
{
  if (_pos + 8 > _size)
    ThrowEndOfData();
  const UInt64 res = Get64(_buffer + _pos);
  _pos += 8;
  return res;
}

#define Y0  '7'
#define Y1  'z'
#define Y2  0xBC
#define Y3  0xAF
#define Y4  0x27
#define Y5  0x1C

#define IS_SIGNATURE(p)( \
        (p)[2] == Y2 &&  \
        (p)[3] == Y3 &&  \
        (p)[5] == Y5 &&  \
        (p)[4] == Y4 &&  \
        (p)[1] == Y1 &&  \
        (p)[0] == Y0)

/* FindSignature_10() is allowed to access data up to and including &limit[9].
   limit[10] access is not allowed.
  return:
    (return_ptr <  limit) : signature was found at (return_ptr)
    (return_ptr >= limit) : limit was reached or crossed. So no signature found before limit
*/
Z7_NO_INLINE
static const Byte *FindSignature_10(const Byte *p, const Byte *limit)
{
  for (;;)
  {
    for (;;)
    {
      if (p >= limit)
        return limit;
      const Byte b = p[5];
      p += 6;
      if (b == Y0) {         break; }
      if (b == Y1) { p -= 1; break; }
      if (b == Y2) { p -= 2; break; }
      if (b == Y3) { p -= 3; break; }
      if (b == Y4) { p -= 4; break; }
      if (b == Y5) { p -= 5; break; }
    }
    if (IS_SIGNATURE(p - 1))
      return p - 1;
  }
}


static inline bool TestStartCrc(const Byte *p)
{
  return CrcCalc(p + 12, 20) == Get32(p + 8);
}

static inline bool TestSignature2(const Byte *p)
{
  if (!IS_SIGNATURE(p))
    return false;
 #ifdef FORMAT_7Z_RECOVERY
  if (TestStartCrc(p))
    return true;
  for (unsigned i = 8; i < kHeaderSize; i++)
    if (p[i] != 0)
      return false;
  return (p[6] != 0 || p[7] != 0);
 #else
  return TestStartCrc(p);
 #endif
}


HRESULT CInArchive::FindAndReadSignature(IInStream *stream, const UInt64 *searchHeaderSizeLimit)
{
  RINOK(ReadStream_FALSE(stream, _header, kHeaderSize))

  if (TestSignature2(_header))
    return S_OK;
  if (searchHeaderSizeLimit && *searchHeaderSizeLimit == 0)
    return S_FALSE;

  const UInt32 kBufSize = (1 << 15) + kHeaderSize;  // must be > (kHeaderSize * 2)
  CAlignedBuffer1 buf(kBufSize);
  memcpy(buf, _header, kHeaderSize);
  UInt64 offset = 0;
  
  for (;;)
  {
    UInt32 readSize =
        (offset == 0) ?
          kBufSize - kHeaderSize - kHeaderSize :
          kBufSize - kHeaderSize;
    if (searchHeaderSizeLimit)
    {
      const UInt64 rem = *searchHeaderSizeLimit - offset;
      if (readSize > rem)
        readSize = (UInt32)rem;
      if (readSize == 0)
        return S_FALSE;
    }
    
    UInt32 processed = 0;
    RINOK(stream->Read(buf + kHeaderSize, readSize, &processed))
    if (processed == 0)
      return S_FALSE;

    /* &buf[0] was already tested for signature before.
       So first search here will be for &buf[1] */
    
    for (UInt32 pos = 0;;)
    {
      const Byte *p = buf + pos + 1;
      const Byte *lim = buf + processed + 1;
      /* we have (kHeaderSize - 1 = 31) filled bytes starting from (lim),
         and it's safe to access just 10 bytes in that reserved area */
      p = FindSignature_10(p, lim);
      if (p >= lim)
        break;
      pos = (UInt32)(p - buf);
      if (TestStartCrc(p))
      {
        memcpy(_header, p, kHeaderSize);
        _arhiveBeginStreamPosition += offset + pos;
        return InStream_SeekSet(stream, _arhiveBeginStreamPosition + kHeaderSize);
      }
    }
    
    offset += processed;
    memmove(buf, buf + processed, kHeaderSize);
  }
}

// S_FALSE means that file is not archive
HRESULT CInArchive::Open(IInStream *stream, const UInt64 *searchHeaderSizeLimit)
{
  HeadersSize = 0;
  Close();
  RINOK(InStream_GetPos_GetSize(stream, _arhiveBeginStreamPosition, _fileEndPosition))
  RINOK(FindAndReadSignature(stream, searchHeaderSizeLimit))
  _stream = stream;
  return S_OK;
}
  
void CInArchive::Close()
{
  _numInByteBufs = 0;
  _stream.Release();
  ThereIsHeaderError = false;
}

void CInArchive::ReadArchiveProperties(CInArchiveInfo & /* archiveInfo */)
{
  for (;;)
  {
    if (ReadID() == NID::kEnd)
      break;
    SkipData();
  }
}

// CFolder &folder can be non empty. So we must set all fields

void CInByte2::ParseFolder(CFolder &folder)
{
  const UInt32 numCoders = ReadNum();

  if (numCoders == 0 || numCoders > k_Scan_NumCoders_MAX)
    ThrowUnsupported();

  folder.Coders.SetSize(numCoders);

  UInt32 numInStreams = 0;
  UInt32 i;
  for (i = 0; i < numCoders; i++)
  {
    CCoderInfo &coder = folder.Coders[i];
    {
      const Byte mainByte = ReadByte();
      if ((mainByte & 0xC0) != 0)
        ThrowUnsupported();
      const unsigned idSize = (mainByte & 0xF);
      if (idSize > 8 || idSize > GetRem())
        ThrowUnsupported();
      const Byte *longID = GetPtr();
      UInt64 id = 0;
      for (unsigned j = 0; j < idSize; j++)
        id = ((id << 8) | longID[j]);
      SkipDataNoCheck(idSize);
      coder.MethodID = id;

      if ((mainByte & 0x10) != 0)
      {
        coder.NumStreams = ReadNum();
        // if (coder.NumStreams > k_Scan_NumCodersStreams_in_Folder_MAX) ThrowUnsupported();
        /* numOutStreams = */ ReadNum();
        // if (ReadNum() != 1) // numOutStreams ThrowUnsupported();
      }
      else
      {
        coder.NumStreams = 1;
      }
      
      if ((mainByte & 0x20) != 0)
      {
        const CNum propsSize = ReadNum();
        coder.Props.Alloc((size_t)propsSize);
        ReadBytes((Byte *)coder.Props, (size_t)propsSize);
      }
      else
        coder.Props.Free();
    }
    numInStreams += coder.NumStreams;
  }

  const UInt32 numBonds = numCoders - 1;
  folder.Bonds.SetSize(numBonds);
  for (i = 0; i < numBonds; i++)
  {
    CBond &bp = folder.Bonds[i];
    bp.PackIndex = ReadNum();
    bp.UnpackIndex = ReadNum();
  }

  if (numInStreams < numBonds)
    ThrowUnsupported();
  const UInt32 numPackStreams = numInStreams - numBonds;
  folder.PackStreams.SetSize(numPackStreams);
  
  if (numPackStreams == 1)
  {
    for (i = 0; i < numInStreams; i++)
      if (folder.FindBond_for_PackStream(i) < 0)
      {
        folder.PackStreams[0] = i;
        break;
      }
    if (i == numInStreams)
      ThrowUnsupported();
  }
  else
    for (i = 0; i < numPackStreams; i++)
      folder.PackStreams[i] = ReadNum();
}

void CFolders::ParseFolderInfo(unsigned folderIndex, CFolder &folder) const
{
  const size_t startPos = FoCodersDataOffset[folderIndex];
  CInByte2 inByte;
  inByte.Init(CodersData.ConstData() + startPos, FoCodersDataOffset[folderIndex + 1] - startPos);
  inByte.ParseFolder(folder);
  if (inByte.GetRem() != 0)
    throw 20120424;
}


void CDatabase::GetPath(unsigned index, UString &path) const
{
  path.Empty();
  if (!NameOffsets || !NamesBuf)
    return;

  const size_t offset = NameOffsets[index];
  const size_t size = NameOffsets[index + 1] - offset;

  if (size >= (1 << 28))
    return;

  wchar_t *s = path.GetBuf((unsigned)size - 1);

  const Byte *p = ((const Byte *)NamesBuf + offset * 2);

  #if defined(_WIN32) && defined(MY_CPU_LE)
  
  wmemcpy(s, (const wchar_t *)(const void *)p, size);
  
  #else

  for (size_t i = 0; i < size; i++)
  {
    *s = Get16(p);
    p += 2;
    s++;
  }

  #endif

  path.ReleaseBuf_SetLen((unsigned)size - 1);
}

HRESULT CDatabase::GetPath_Prop(unsigned index, PROPVARIANT *path) const throw()
{
  PropVariant_Clear(path);
  if (!NameOffsets || !NamesBuf)
    return S_OK;

  const size_t offset = NameOffsets[index];
  const size_t size = NameOffsets[index + 1] - offset;

  if (size >= (1 << 14))
    return S_OK;

  // (size) includes null terminator

  /*
  #if WCHAR_MAX > 0xffff
  
  const Byte *p = ((const Byte *)NamesBuf + offset * 2);
  size = Utf16LE__Get_Num_WCHARs(p, size - 1);
  // (size) doesn't include null terminator
  RINOK(PropVarEm_Alloc_Bstr(path, (unsigned)size));
  wchar_t *s = path->bstrVal;
  wchar_t *sEnd = Utf16LE__To_WCHARs_Sep(p, size, s);
  *sEnd = 0;
  if (s + size != sEnd) return E_FAIL;

  #else
  */

  RINOK(PropVarEm_Alloc_Bstr(path, (unsigned)size - 1))
  wchar_t *s = path->bstrVal;
  const Byte *p = ((const Byte *)NamesBuf + offset * 2);
  // Utf16LE__To_WCHARs_Sep(p, size, s);

  for (size_t i = 0; i < size; i++)
  {
    wchar_t c = Get16(p);
    p += 2;
    #if WCHAR_PATH_SEPARATOR != L'/'
    if (c == L'/')
      c = WCHAR_PATH_SEPARATOR;
    else if (c == L'\\')
      c = WCHAR_IN_FILE_NAME_BACKSLASH_REPLACEMENT; // WSL scheme
    #endif
    *s++ = c;
  }

  // #endif

  return S_OK;

  /*
  unsigned cur = index;
  unsigned size = 0;
  
  for (int i = 0;; i++)
  {
    size_t len = NameOffsets[cur + 1] - NameOffsets[cur];
    size += (unsigned)len;
    if (i > 256 || len > (1 << 14) || size > (1 << 14))
      return PropVarEm_Set_Str(path, "[TOO-LONG]");
    cur = Files[cur].Parent;
    if (cur < 0)
      break;
  }
  size--;

  RINOK(PropVarEm_Alloc_Bstr(path, size));
  wchar_t *s = path->bstrVal;
  s += size;
  *s = 0;
  cur = index;
  
  for (;;)
  {
    unsigned len = (unsigned)(NameOffsets[cur + 1] - NameOffsets[cur] - 1);
    const Byte *p = (const Byte *)NamesBuf + (NameOffsets[cur + 1] * 2) - 2;
    for (; len != 0; len--)
    {
      p -= 2;
      --s;
      wchar_t c = Get16(p);
      if (c == '/')
        c = WCHAR_PATH_SEPARATOR;
      *s = c;
    }

    const CFileItem &file = Files[cur];
    cur = file.Parent;
    if (cur < 0)
      return S_OK;
    *(--s) = (file.IsAltStream ? ':' : WCHAR_PATH_SEPARATOR);
  }
  */
}

void CInArchive::WaitId(UInt64 id)
{
  for (;;)
  {
    const UInt64 type = ReadID();
    if (type == id)
      return;
    if (type == NID::kEnd)
      ThrowIncorrect();
    SkipData();
  }
}


void CInArchive::Read_UInt32_Vector(CUInt32DefVector &v)
{
  const unsigned numItems = v.Defs.Size();
  v.Vals.ClearAndSetSize(numItems);
  UInt32 *p = &v.Vals[0];
  const bool *defs = &v.Defs[0];
  for (unsigned i = 0; i < numItems; i++)
  {
    UInt32 a = 0;
    if (defs[i])
      a = ReadUInt32();
    p[i] = a;
  }
}


void CInArchive::ReadHashDigests(unsigned numItems, CUInt32DefVector &crcs)
{
  ReadBoolVector2(numItems, crcs.Defs);
  Read_UInt32_Vector(crcs);
}


void CInArchive::ReadPackInfo(CFolders &f)
{
  const CNum numPackStreams = ReadNum();
  
  WaitId(NID::kSize);
  f.PackPositions.Alloc(numPackStreams + 1);
  f.NumPackStreams = numPackStreams;
  UInt64 sum = 0;
  for (CNum i = 0; i < numPackStreams; i++)
  {
    f.PackPositions[i] = sum;
    const UInt64 packSize = ReadNumber();
    sum += packSize;
    if (sum < packSize)
      ThrowIncorrect();
  }
  f.PackPositions[numPackStreams] = sum;

  UInt64 type;
  for (;;)
  {
    type = ReadID();
    if (type == NID::kEnd)
      return;
    if (type == NID::kCRC)
    {
      CUInt32DefVector PackCRCs;
      ReadHashDigests(numPackStreams, PackCRCs);
      continue;
    }
    SkipData();
  }
}

void CInArchive::ReadUnpackInfo(
    const CObjectVector<CByteBuffer> *dataVector,
    CFolders &folders)
{
  WaitId(NID::kFolder);
  const CNum numFolders = ReadNum();

  CNum numCodersOutStreams = 0;
  {
    CStreamSwitch streamSwitch;
    streamSwitch.Set(this, dataVector);
    const Byte *startBufPtr = _inByteBack->GetPtr();
    folders.NumFolders = numFolders;

    folders.FoStartPackStreamIndex.Alloc(numFolders + 1);
    folders.FoToMainUnpackSizeIndex.Alloc(numFolders);
    folders.FoCodersDataOffset.Alloc(numFolders + 1);
    folders.FoToCoderUnpackSizes.Alloc(numFolders + 1);

    CBoolVector StreamUsed;
    CBoolVector CoderUsed;

    CNum packStreamIndex = 0;
    CNum fo;
    CInByte2 *inByte = _inByteBack;
    
    for (fo = 0; fo < numFolders; fo++)
    {
      UInt32 indexOfMainStream = 0;
      UInt32 numPackStreams = 0;
      folders.FoCodersDataOffset[fo] = (size_t)(_inByteBack->GetPtr() - startBufPtr);

      CNum numInStreams = 0;
      const CNum numCoders = inByte->ReadNum();
    
      if (numCoders == 0 || numCoders > k_Scan_NumCoders_MAX)
        ThrowUnsupported();

      for (CNum ci = 0; ci < numCoders; ci++)
      {
        const Byte mainByte = inByte->ReadByte();
        if ((mainByte & 0xC0) != 0)
          ThrowUnsupported();
        
        const unsigned idSize = (mainByte & 0xF);
        if (idSize > 8)
          ThrowUnsupported();
        if (idSize > inByte->GetRem())
          ThrowEndOfData();
        const Byte *longID = inByte->GetPtr();
        UInt64 id = 0;
        for (unsigned j = 0; j < idSize; j++)
          id = ((id << 8) | longID[j]);
        inByte->SkipDataNoCheck(idSize);
        if (folders.ParsedMethods.IDs.Size() < 128)
          folders.ParsedMethods.IDs.AddToUniqueSorted(id);
        
        CNum coderInStreams = 1;
        if ((mainByte & 0x10) != 0)
        {
          coderInStreams = inByte->ReadNum();
          if (coderInStreams > k_Scan_NumCodersStreams_in_Folder_MAX)
            ThrowUnsupported();
          if (inByte->ReadNum() != 1)
            ThrowUnsupported();
        }

        numInStreams += coderInStreams;
        if (numInStreams > k_Scan_NumCodersStreams_in_Folder_MAX)
          ThrowUnsupported();
        
        if ((mainByte & 0x20) != 0)
        {
          const CNum propsSize = inByte->ReadNum();
          if (propsSize > inByte->GetRem())
            ThrowEndOfData();
          if (id == k_LZMA2 && propsSize == 1)
          {
            const Byte v = *_inByteBack->GetPtr();
            if (folders.ParsedMethods.Lzma2Prop < v)
              folders.ParsedMethods.Lzma2Prop = v;
          }
          else if (id == k_LZMA && propsSize == 5)
          {
            const UInt32 dicSize = GetUi32(_inByteBack->GetPtr() + 1);
            if (folders.ParsedMethods.LzmaDic < dicSize)
              folders.ParsedMethods.LzmaDic = dicSize;
          }
          inByte->SkipDataNoCheck((size_t)propsSize);
        }
      }
      
      if (numCoders == 1 && numInStreams == 1)
      {
        indexOfMainStream = 0;
        numPackStreams = 1;
      }
      else
      {
        UInt32 i;
        const CNum numBonds = numCoders - 1;
        if (numInStreams < numBonds)
          ThrowUnsupported();
        
        BoolVector_Fill_False(StreamUsed, numInStreams);
        BoolVector_Fill_False(CoderUsed, numCoders);
        
        for (i = 0; i < numBonds; i++)
        {
          CNum index = ReadNum();
          if (index >= numInStreams || StreamUsed[index])
            ThrowUnsupported();
          StreamUsed[index] = true;
          
          index = ReadNum();
          if (index >= numCoders || CoderUsed[index])
            ThrowUnsupported();
          CoderUsed[index] = true;
        }
        
        numPackStreams = numInStreams - numBonds;
        
        if (numPackStreams != 1)
          for (i = 0; i < numPackStreams; i++)
          {
            const CNum index = inByte->ReadNum(); // PackStreams
            if (index >= numInStreams || StreamUsed[index])
              ThrowUnsupported();
            StreamUsed[index] = true;
          }
          
        for (i = 0; i < numCoders; i++)
          if (!CoderUsed[i])
          {
            indexOfMainStream = i;
            break;
          }
          
        if (i == numCoders)
          ThrowUnsupported();
      }
      
      folders.FoToCoderUnpackSizes[fo] = numCodersOutStreams;
      numCodersOutStreams += numCoders;
      folders.FoStartPackStreamIndex[fo] = packStreamIndex;
      if (numPackStreams > folders.NumPackStreams - packStreamIndex)
        ThrowIncorrect();
      packStreamIndex += numPackStreams;
      folders.FoToMainUnpackSizeIndex[fo] = (Byte)indexOfMainStream;
    }
    
    const size_t dataSize = (size_t)(_inByteBack->GetPtr() - startBufPtr);
    folders.FoToCoderUnpackSizes[fo] = numCodersOutStreams;
    folders.FoStartPackStreamIndex[fo] = packStreamIndex;
    folders.FoCodersDataOffset[fo] = (size_t)(_inByteBack->GetPtr() - startBufPtr);
    folders.CodersData.CopyFrom(startBufPtr, dataSize);

    // if (folders.NumPackStreams != packStreamIndex) ThrowUnsupported();
  }

  WaitId(NID::kCodersUnpackSize);
  folders.CoderUnpackSizes.Alloc(numCodersOutStreams);
  for (CNum i = 0; i < numCodersOutStreams; i++)
    folders.CoderUnpackSizes[i] = ReadNumber();

  for (;;)
  {
    const UInt64 type = ReadID();
    if (type == NID::kEnd)
      return;
    if (type == NID::kCRC)
    {
      ReadHashDigests(numFolders, folders.FolderCRCs);
      continue;
    }
    SkipData();
  }
}

void CInArchive::ReadSubStreamsInfo(
    CFolders &folders,
    CRecordVector<UInt64> &unpackSizes,
    CUInt32DefVector &digests)
{
  folders.NumUnpackStreamsVector.Alloc(folders.NumFolders);
  CNum i;
  for (i = 0; i < folders.NumFolders; i++)
    folders.NumUnpackStreamsVector[i] = 1;
  
  UInt64 type;
  
  for (;;)
  {
    type = ReadID();
    if (type == NID::kNumUnpackStream)
    {
      for (i = 0; i < folders.NumFolders; i++)
        folders.NumUnpackStreamsVector[i] = ReadNum();
      continue;
    }
    if (type == NID::kCRC || type == NID::kSize || type == NID::kEnd)
      break;
    SkipData();
  }

  if (type == NID::kSize)
  {
    for (i = 0; i < folders.NumFolders; i++)
    {
      // v3.13 incorrectly worked with empty folders
      // v4.07: we check that folder is empty
      const CNum numSubstreams = folders.NumUnpackStreamsVector[i];
      if (numSubstreams == 0)
        continue;
      UInt64 sum = 0;
      for (CNum j = 1; j < numSubstreams; j++)
      {
        const UInt64 size = ReadNumber();
        unpackSizes.Add(size);
        sum += size;
        if (sum < size)
          ThrowIncorrect();
      }
      const UInt64 folderUnpackSize = folders.GetFolderUnpackSize(i);
      if (folderUnpackSize < sum)
        ThrowIncorrect();
      unpackSizes.Add(folderUnpackSize - sum);
    }
    type = ReadID();
  }
  else
  {
    for (i = 0; i < folders.NumFolders; i++)
    {
      /* v9.26 - v9.29 incorrectly worked:
         if (folders.NumUnpackStreamsVector[i] == 0), it threw error */
      const CNum val = folders.NumUnpackStreamsVector[i];
      if (val > 1)
        ThrowIncorrect();
      if (val == 1)
        unpackSizes.Add(folders.GetFolderUnpackSize(i));
    }
  }

  unsigned numDigests = 0;
  for (i = 0; i < folders.NumFolders; i++)
  {
    const CNum numSubstreams = folders.NumUnpackStreamsVector[i];
    if (numSubstreams != 1 || !folders.FolderCRCs.ValidAndDefined(i))
      numDigests += numSubstreams;
  }

  for (;;)
  {
    if (type == NID::kEnd)
      break;
    if (type == NID::kCRC)
    {
      // CUInt32DefVector digests2;
      // ReadHashDigests(numDigests, digests2);
      CBoolVector digests2;
      ReadBoolVector2(numDigests, digests2);

      digests.ClearAndSetSize(unpackSizes.Size());
      
      unsigned k = 0;
      unsigned k2 = 0;
      
      for (i = 0; i < folders.NumFolders; i++)
      {
        const CNum numSubstreams = folders.NumUnpackStreamsVector[i];
        if (numSubstreams == 1 && folders.FolderCRCs.ValidAndDefined(i))
        {
          digests.Defs[k] = true;
          digests.Vals[k] = folders.FolderCRCs.Vals[i];
          k++;
        }
        else for (CNum j = 0; j < numSubstreams; j++)
        {
          bool defined = digests2[k2++];
          digests.Defs[k] = defined;
          UInt32 crc = 0;
          if (defined)
            crc = ReadUInt32();
          digests.Vals[k] = crc;
          k++;
        }
      }
      // if (k != unpackSizes.Size()) throw 1234567;
    }
    else
      SkipData();
    
    type = ReadID();
  }

  if (digests.Defs.Size() != unpackSizes.Size())
  {
    digests.ClearAndSetSize(unpackSizes.Size());
    unsigned k = 0;
    for (i = 0; i < folders.NumFolders; i++)
    {
      const CNum numSubstreams = folders.NumUnpackStreamsVector[i];
      if (numSubstreams == 1 && folders.FolderCRCs.ValidAndDefined(i))
      {
        digests.Defs[k] = true;
        digests.Vals[k] = folders.FolderCRCs.Vals[i];
        k++;
      }
      else for (CNum j = 0; j < numSubstreams; j++)
      {
        digests.Defs[k] = false;
        digests.Vals[k] = 0;
        k++;
      }
    }
  }
}



void CInArchive::ReadStreamsInfo(
    const CObjectVector<CByteBuffer> *dataVector,
    UInt64 &dataOffset,
    CFolders &folders,
    CRecordVector<UInt64> &unpackSizes,
    CUInt32DefVector &digests)
{
  UInt64 type = ReadID();
  
  if (type == NID::kPackInfo)
  {
    dataOffset = ReadNumber();
    if (dataOffset > _rangeLimit)
      ThrowIncorrect();
    ReadPackInfo(folders);
    if (folders.PackPositions[folders.NumPackStreams] > _rangeLimit - dataOffset)
      ThrowIncorrect();
    type = ReadID();
  }

  if (type == NID::kUnpackInfo)
  {
    ReadUnpackInfo(dataVector, folders);
    type = ReadID();
  }

  if (folders.NumFolders != 0 && !folders.PackPositions)
  {
    // if there are folders, we need PackPositions also
    folders.PackPositions.Alloc(1);
    folders.PackPositions[0] = 0;
  }
  
  if (type == NID::kSubStreamsInfo)
  {
    ReadSubStreamsInfo(folders, unpackSizes, digests);
    type = ReadID();
  }
  else
  {
    folders.NumUnpackStreamsVector.Alloc(folders.NumFolders);
    /* If digests.Defs.Size() == 0, it means that there are no crcs.
       So we don't need to fill digests with values. */
    // digests.Vals.ClearAndSetSize(folders.NumFolders);
    // BoolVector_Fill_False(digests.Defs, folders.NumFolders);
    for (CNum i = 0; i < folders.NumFolders; i++)
    {
      folders.NumUnpackStreamsVector[i] = 1;
      unpackSizes.Add(folders.GetFolderUnpackSize(i));
      // digests.Vals[i] = 0;
    }
  }
  
  if (type != NID::kEnd)
    ThrowIncorrect();
}

void CInArchive::ReadBoolVector(unsigned numItems, CBoolVector &v)
{
  v.ClearAndSetSize(numItems);
  Byte b = 0;
  Byte mask = 0;
  bool *p = &v[0];
  for (unsigned i = 0; i < numItems; i++)
  {
    if (mask == 0)
    {
      b = ReadByte();
      mask = 0x80;
    }
    p[i] = ((b & mask) != 0);
    mask = (Byte)(mask >> 1);
  }
}

void CInArchive::ReadBoolVector2(unsigned numItems, CBoolVector &v)
{
  const Byte allAreDefined = ReadByte();
  if (allAreDefined == 0)
  {
    ReadBoolVector(numItems, v);
    return;
  }
  v.ClearAndSetSize(numItems);
  bool *p = &v[0];
  for (unsigned i = 0; i < numItems; i++)
    p[i] = true;
}

void CInArchive::ReadUInt64DefVector(const CObjectVector<CByteBuffer> &dataVector,
    CUInt64DefVector &v, unsigned numItems)
{
  ReadBoolVector2(numItems, v.Defs);

  CStreamSwitch streamSwitch;
  streamSwitch.Set(this, &dataVector);
  
  v.Vals.ClearAndSetSize(numItems);
  UInt64 *p = &v.Vals[0];
  const bool *defs = &v.Defs[0];

  for (unsigned i = 0; i < numItems; i++)
  {
    UInt64 t = 0;
    if (defs[i])
      t = ReadUInt64();
    p[i] = t;
  }
}

HRESULT CInArchive::ReadAndDecodePackedStreams(
    DECL_EXTERNAL_CODECS_LOC_VARS
    UInt64 baseOffset,
    UInt64 &dataOffset, CObjectVector<CByteBuffer> &dataVector
    Z7_7Z_DECODER_CRYPRO_VARS_DECL
    )
{
  CFolders folders;
  CRecordVector<UInt64> unpackSizes;
  CUInt32DefVector  digests;
  
  ReadStreamsInfo(NULL,
    dataOffset,
    folders,
    unpackSizes,
    digests);
  
  CDecoder decoder(_useMixerMT);

  for (CNum i = 0; i < folders.NumFolders; i++)
  {
    CByteBuffer &data = dataVector.AddNew();
    const UInt64 unpackSize64 = folders.GetFolderUnpackSize(i);
    const size_t unpackSize = (size_t)unpackSize64;
    if (unpackSize != unpackSize64)
      ThrowUnsupported();
    data.Alloc(unpackSize);
    
    CMyComPtr2_Create<ISequentialOutStream, CBufPtrSeqOutStream> outStreamSpec;
    outStreamSpec->Init(data, unpackSize);
    
    bool dataAfterEnd_Error = false;

    HRESULT result = decoder.Decode(
        EXTERNAL_CODECS_LOC_VARS
        _stream, baseOffset + dataOffset,
        folders, i,
        NULL, // &unpackSize64
        
        outStreamSpec,
        NULL, // *compressProgress

        NULL  // **inStreamMainRes
        , dataAfterEnd_Error
        
        Z7_7Z_DECODER_CRYPRO_VARS
        #if !defined(Z7_ST)
          , false // mtMode
          , 1     // numThreads
          , 0     // memUsage
        #endif
      );
    
    RINOK(result)
    
    if (dataAfterEnd_Error)
      ThereIsHeaderError = true;
    
    if (unpackSize != outStreamSpec->GetPos())
      ThrowIncorrect();

    if (folders.FolderCRCs.ValidAndDefined(i))
      if (CrcCalc(data, unpackSize) != folders.FolderCRCs.Vals[i])
        ThrowIncorrect();
  }

  if (folders.PackPositions)
    HeadersSize += folders.PackPositions[folders.NumPackStreams];

  return S_OK;
}

HRESULT CInArchive::ReadHeader(
    DECL_EXTERNAL_CODECS_LOC_VARS
    CDbEx &db
    Z7_7Z_DECODER_CRYPRO_VARS_DECL
    )
{
  UInt64 type = ReadID();

  if (type == NID::kArchiveProperties)
  {
    ReadArchiveProperties(db.ArcInfo);
    type = ReadID();
  }
 
  CObjectVector<CByteBuffer> dataVector;
  
  if (type == NID::kAdditionalStreamsInfo)
  {
    const HRESULT result = ReadAndDecodePackedStreams(
        EXTERNAL_CODECS_LOC_VARS
        db.ArcInfo.StartPositionAfterHeader,
        db.ArcInfo.DataStartPosition2,
        dataVector
        Z7_7Z_DECODER_CRYPRO_VARS
        );
    RINOK(result)
    db.ArcInfo.DataStartPosition2 += db.ArcInfo.StartPositionAfterHeader;
    type = ReadID();
  }

  CRecordVector<UInt64> unpackSizes;
  CUInt32DefVector digests;
  
  if (type == NID::kMainStreamsInfo)
  {
    ReadStreamsInfo(&dataVector,
        db.ArcInfo.DataStartPosition,
        (CFolders &)db,
        unpackSizes,
        digests);
    db.ArcInfo.DataStartPosition += db.ArcInfo.StartPositionAfterHeader;
    type = ReadID();
  }

  if (type == NID::kFilesInfo)
  {
  
  const CNum numFiles = ReadNum();

  db.ArcInfo.FileInfoPopIDs.Add(NID::kSize);
  // if (!db.PackSizes.IsEmpty())
    db.ArcInfo.FileInfoPopIDs.Add(NID::kPackInfo);
  if (numFiles > 0 && !digests.Defs.IsEmpty())
    db.ArcInfo.FileInfoPopIDs.Add(NID::kCRC);

  CBoolVector emptyStreamVector;
  CBoolVector emptyFileVector;
  CBoolVector antiFileVector;
  unsigned numEmptyStreams = 0;

  for (;;)
  {
    const UInt64 type2 = ReadID();
    if (type2 == NID::kEnd)
      break;
    const UInt64 size = ReadNumber();
    if (size > _inByteBack->GetRem())
      ThrowIncorrect();
    CStreamSwitch switchProp;
    switchProp.Set(this, _inByteBack->GetPtr(), (size_t)size, true);
    bool addPropIdToList = true;
    bool isKnownType = true;
    if (type2 > ((UInt32)1 << 30))
      isKnownType = false;
    else switch ((UInt32)type2)
    {
      case NID::kName:
      {
        CStreamSwitch streamSwitch;
        streamSwitch.Set(this, &dataVector);
        const size_t rem = _inByteBack->GetRem();
        db.NamesBuf.Alloc(rem);
        ReadBytes(db.NamesBuf, rem);
        db.NameOffsets.Alloc(numFiles + 1);
        size_t pos = 0;
        unsigned i;
        for (i = 0; i < numFiles; i++)
        {
          const size_t curRem = (rem - pos) / 2;
          const UInt16 *buf = (const UInt16 *)(const void *)(db.NamesBuf.ConstData() + pos);
          size_t j;
          for (j = 0; j < curRem && buf[j] != 0; j++);
          if (j == curRem)
            ThrowEndOfData();
          db.NameOffsets[i] = pos / 2;
          pos += j * 2 + 2;
        }
        db.NameOffsets[i] = pos / 2;
        if (pos != rem)
          ThereIsHeaderError = true;
        break;
      }

      case NID::kWinAttrib:
      {
        ReadBoolVector2(numFiles, db.Attrib.Defs);
        CStreamSwitch streamSwitch;
        streamSwitch.Set(this, &dataVector);
        Read_UInt32_Vector(db.Attrib);
        break;
      }
      
      /*
      case NID::kIsAux:
      {
        ReadBoolVector(numFiles, db.IsAux);
        break;
      }
      case NID::kParent:
      {
        db.IsTree = true;
        // CBoolVector boolVector;
        // ReadBoolVector2(numFiles, boolVector);
        // CStreamSwitch streamSwitch;
        // streamSwitch.Set(this, &dataVector);
        CBoolVector boolVector;
        ReadBoolVector2(numFiles, boolVector);

        db.ThereAreAltStreams = false;
        for (i = 0; i < numFiles; i++)
        {
          CFileItem &file = db.Files[i];
          // file.Parent = -1;
          // if (boolVector[i])
          file.Parent = (int)ReadUInt32();
          file.IsAltStream = !boolVector[i];
          if (file.IsAltStream)
            db.ThereAreAltStreams = true;
        }
        break;
      }
      */
      case NID::kEmptyStream:
      {
        ReadBoolVector(numFiles, emptyStreamVector);
        numEmptyStreams = BoolVector_CountSum(emptyStreamVector);
        emptyFileVector.Clear();
        antiFileVector.Clear();
        break;
      }
      case NID::kEmptyFile:  ReadBoolVector(numEmptyStreams, emptyFileVector); break;
      case NID::kAnti:  ReadBoolVector(numEmptyStreams, antiFileVector); break;
      case NID::kStartPos:  ReadUInt64DefVector(dataVector, db.StartPos, (unsigned)numFiles); break;
      case NID::kCTime:  ReadUInt64DefVector(dataVector, db.CTime, (unsigned)numFiles); break;
      case NID::kATime:  ReadUInt64DefVector(dataVector, db.ATime, (unsigned)numFiles); break;
      case NID::kMTime:  ReadUInt64DefVector(dataVector, db.MTime, (unsigned)numFiles); break;
      case NID::kDummy:
      {
        for (UInt64 j = 0; j < size; j++)
          if (ReadByte() != 0)
            ThereIsHeaderError = true;
        addPropIdToList = false;
        break;
      }
      /*
      case NID::kNtSecure:
      {
        try
        {
          {
            CStreamSwitch streamSwitch;
            streamSwitch.Set(this, &dataVector);
            UInt32 numDescriptors = ReadUInt32();
            size_t offset = 0;
            db.SecureOffsets.Clear();
            for (i = 0; i < numDescriptors; i++)
            {
              UInt32 size = ReadUInt32();
              db.SecureOffsets.Add(offset);
              offset += size;
            }
            // ThrowIncorrect();;
            db.SecureOffsets.Add(offset);
            db.SecureBuf.SetCapacity(offset);
            for (i = 0; i < numDescriptors; i++)
            {
              offset = db.SecureOffsets[i];
              ReadBytes(db.SecureBuf + offset, db.SecureOffsets[i + 1] - offset);
            }
            db.SecureIDs.Clear();
            for (unsigned i = 0; i < numFiles; i++)
            {
              db.SecureIDs.Add(ReadNum());
              // db.SecureIDs.Add(ReadUInt32());
            }
            // ReadUInt32();
            if (_inByteBack->GetRem() != 0)
              ThrowIncorrect();;
          }
        }
        catch(CInArchiveException &)
        {
          ThereIsHeaderError = true;
          addPropIdToList = isKnownType = false;
          db.ClearSecure();
        }
        break;
      }
      */
      default:
        addPropIdToList = isKnownType = false;
    }
    if (isKnownType)
    {
      if (addPropIdToList)
        db.ArcInfo.FileInfoPopIDs.Add(type2);
    }
    else
    {
      db.UnsupportedFeatureWarning = true;
      _inByteBack->SkipRem();
    }
    // SkipData worked incorrectly in some versions before v4.59 (7zVer <= 0.02)
    if (_inByteBack->GetRem() != 0)
      ThrowIncorrect();
  }

  type = ReadID(); // Read (NID::kEnd) end of headers

  if (numFiles - numEmptyStreams != unpackSizes.Size())
    ThrowUnsupported();

  CNum emptyFileIndex = 0;
  CNum sizeIndex = 0;

  const unsigned numAntiItems = BoolVector_CountSum(antiFileVector);

  if (numAntiItems != 0)
    db.IsAnti.ClearAndSetSize(numFiles);

  db.Files.ClearAndSetSize(numFiles);

  for (CNum i = 0; i < numFiles; i++)
  {
    CFileItem &file = db.Files[i];
    bool isAnti;
    file.Crc = 0;
    if (!BoolVector_Item_IsValidAndTrue(emptyStreamVector, i))
    {
      file.HasStream = true;
      file.IsDir = false;
      isAnti = false;
      file.Size = unpackSizes[sizeIndex];
      file.CrcDefined = digests.ValidAndDefined(sizeIndex);
      if (file.CrcDefined)
        file.Crc = digests.Vals[sizeIndex];
      sizeIndex++;
    }
    else
    {
      file.HasStream = false;
      file.IsDir = !BoolVector_Item_IsValidAndTrue(emptyFileVector, emptyFileIndex);
      isAnti = BoolVector_Item_IsValidAndTrue(antiFileVector, emptyFileIndex);
      emptyFileIndex++;
      file.Size = 0;
      file.CrcDefined = false;
    }
    if (numAntiItems != 0)
      db.IsAnti[i] = isAnti;
  }
  
  }
  
  db.FillLinks();

  if (type != NID::kEnd || _inByteBack->GetRem() != 0)
  {
    db.UnsupportedFeatureWarning = true;
    // ThrowIncorrect();
  }

  return S_OK;
}


void CDbEx::FillLinks()
{
  FolderStartFileIndex.Alloc(NumFolders);
  FileIndexToFolderIndexMap.Alloc(Files.Size());
  
  CNum folderIndex = 0;
  CNum indexInFolder = 0;
  unsigned i;

  for (i = 0; i < Files.Size(); i++)
  {
    const bool emptyStream = !Files[i].HasStream;
    if (indexInFolder == 0)
    {
      if (emptyStream)
      {
        FileIndexToFolderIndexMap[i] = kNumNoIndex;
        continue;
      }
      // v3.13 incorrectly worked with empty folders
      // v4.07: we skip empty folders
      for (;;)
      {
        if (folderIndex >= NumFolders)
          ThrowIncorrect();
        FolderStartFileIndex[folderIndex] = i;
        if (NumUnpackStreamsVector[folderIndex] != 0)
          break;
        folderIndex++;
      }
    }
    FileIndexToFolderIndexMap[i] = folderIndex;
    if (emptyStream)
      continue;
    if (++indexInFolder >= NumUnpackStreamsVector[folderIndex])
    {
      folderIndex++;
      indexInFolder = 0;
    }
  }

  if (indexInFolder != 0)
  {
    folderIndex++;
    // 18.06
    ThereIsHeaderError = true;
    // ThrowIncorrect();
  }
  
  for (;;)
  {
    if (folderIndex >= NumFolders)
      return;
    FolderStartFileIndex[folderIndex] = i;
    if (NumUnpackStreamsVector[folderIndex] != 0)
    {
      // 18.06
      ThereIsHeaderError = true;
      // ThrowIncorrect();
    }
    folderIndex++;
  }
}


HRESULT CInArchive::ReadDatabase2(
    DECL_EXTERNAL_CODECS_LOC_VARS
    CDbEx &db
    Z7_7Z_DECODER_CRYPRO_VARS_DECL
    )
{
  db.Clear();
  db.ArcInfo.StartPosition = _arhiveBeginStreamPosition;

  db.ArcInfo.Version.Major = _header[6];
  db.ArcInfo.Version.Minor = _header[7];

  if (db.ArcInfo.Version.Major != kMajorVersion)
  {
    // db.UnsupportedVersion = true;
    return S_FALSE;
  }

  UInt64 nextHeaderOffset = Get64(_header + 12);
  UInt64 nextHeaderSize = Get64(_header + 20);
  UInt32 nextHeaderCRC = Get32(_header + 28);

  #ifdef FORMAT_7Z_RECOVERY
  const UInt32 crcFromArc = Get32(_header + 8);
  if (crcFromArc == 0 && nextHeaderOffset == 0 && nextHeaderSize == 0 && nextHeaderCRC == 0)
  {
    UInt64 cur, fileSize;
    RINOK(InStream_GetPos(_stream, cur))
    const unsigned kCheckSize = 512;
    Byte buf[kCheckSize];
    RINOK(InStream_GetSize_SeekToEnd(_stream, fileSize))
    const UInt64 rem = fileSize - cur;
    unsigned checkSize = kCheckSize;
    if (rem < kCheckSize)
      checkSize = (unsigned)(rem);
    if (checkSize < 3)
      return S_FALSE;
    RINOK(InStream_SeekSet(_stream, fileSize - checkSize))
    RINOK(ReadStream_FALSE(_stream, buf, (size_t)checkSize))

    if (buf[checkSize - 1] != 0)
      return S_FALSE;

    unsigned i;
    for (i = checkSize - 2;; i--)
    {
      if ((buf[i] == NID::kEncodedHeader && buf[i + 1] == NID::kPackInfo) ||
          (buf[i] == NID::kHeader        && buf[i + 1] == NID::kMainStreamsInfo))
        break;
      if (i == 0)
        return S_FALSE;
    }
    nextHeaderSize = checkSize - i;
    nextHeaderOffset = rem - nextHeaderSize;
    nextHeaderCRC = CrcCalc(buf + i, (size_t)nextHeaderSize);
    RINOK(InStream_SeekSet(_stream, cur))
    db.StartHeaderWasRecovered = true;
  }
  else
  #endif
  {
    // Crc was tested already at signature check
    // if (CrcCalc(_header + 12, 20) != crcFromArchive) ThrowIncorrect();
  }

  db.ArcInfo.StartPositionAfterHeader = _arhiveBeginStreamPosition + kHeaderSize;
  db.PhySize = kHeaderSize;

  db.IsArc = false;
  if ((Int64)nextHeaderOffset < 0 ||
      nextHeaderSize > ((UInt64)1 << 62))
    return S_FALSE;

  HeadersSize = kHeaderSize;

  if (nextHeaderSize == 0)
  {
    if (nextHeaderOffset != 0 || nextHeaderCRC != 0)
      return S_FALSE;
    db.IsArc = true;
    db.HeadersSize = HeadersSize;
    return S_OK;
  }
  
  if (!db.StartHeaderWasRecovered)
    db.IsArc = true;
  
  HeadersSize += nextHeaderSize;
  // db.EndHeaderOffset = nextHeaderOffset;
  _rangeLimit = nextHeaderOffset;

  db.PhySize = kHeaderSize + nextHeaderOffset + nextHeaderSize;
  if (_fileEndPosition - db.ArcInfo.StartPositionAfterHeader < nextHeaderOffset + nextHeaderSize)
  {
    db.UnexpectedEnd = true;
    return S_FALSE;
  }
  RINOK(_stream->Seek((Int64)nextHeaderOffset, STREAM_SEEK_CUR, NULL))

  const size_t nextHeaderSize_t = (size_t)nextHeaderSize;
  if (nextHeaderSize_t != nextHeaderSize)
    return E_OUTOFMEMORY;
  CByteBuffer buffer2(nextHeaderSize_t);

  RINOK(ReadStream_FALSE(_stream, buffer2, nextHeaderSize_t))

  if (CrcCalc(buffer2, nextHeaderSize_t) != nextHeaderCRC)
    ThrowIncorrect();

  if (!db.StartHeaderWasRecovered)
    db.PhySizeWasConfirmed = true;
  
  CStreamSwitch streamSwitch;
  streamSwitch.Set(this, buffer2);
  
  CObjectVector<CByteBuffer> dataVector;
  
  const UInt64 type = ReadID();
  if (type != NID::kHeader)
  {
    if (type != NID::kEncodedHeader)
      ThrowIncorrect();
    const HRESULT result = ReadAndDecodePackedStreams(
        EXTERNAL_CODECS_LOC_VARS
        db.ArcInfo.StartPositionAfterHeader,
        db.ArcInfo.DataStartPosition2,
        dataVector
        Z7_7Z_DECODER_CRYPRO_VARS
        );
    RINOK(result)
    if (dataVector.Size() == 0)
      return S_OK;
    if (dataVector.Size() > 1)
      ThrowIncorrect();
    streamSwitch.Remove();
    streamSwitch.Set(this, dataVector.Front());
    if (ReadID() != NID::kHeader)
      ThrowIncorrect();
  }

  db.IsArc = true;

  db.HeadersSize = HeadersSize;

  return ReadHeader(
    EXTERNAL_CODECS_LOC_VARS
    db
    Z7_7Z_DECODER_CRYPRO_VARS
    );
}


HRESULT CInArchive::ReadDatabase(
    DECL_EXTERNAL_CODECS_LOC_VARS
    CDbEx &db
    Z7_7Z_DECODER_CRYPRO_VARS_DECL
    )
{
  try
  {
    const HRESULT res = ReadDatabase2(
      EXTERNAL_CODECS_LOC_VARS db
      Z7_7Z_DECODER_CRYPRO_VARS
      );
    if (ThereIsHeaderError)
      db.ThereIsHeaderError = true;
    if (res == E_NOTIMPL)
      ThrowUnsupported();
    return res;
  }
  catch(CUnsupportedFeatureException &)
  {
    db.UnsupportedFeatureError = true;
    return S_FALSE;
  }
  catch(CInArchiveException &)
  {
    db.ThereIsHeaderError = true;
    return S_FALSE;
  }
}

}}
