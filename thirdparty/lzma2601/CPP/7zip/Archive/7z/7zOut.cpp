// 7zOut.cpp

#include "StdAfx.h"

#include "../../../../C/7zCrc.h"
#include "../../../../C/CpuArch.h"

#include "../../../Common/AutoPtr.h"
// #include "../../../Common/UTFConvert.h"

#include "../../Common/StreamObjects.h"
#include "../Common/OutStreamWithCRC.h"

#include "7zOut.h"

unsigned BoolVector_CountSum(const CBoolVector &v);

static UInt64 UInt64Vector_CountSum(const CRecordVector<UInt64> &v)
{
  UInt64 sum = 0;
  const unsigned size = v.Size();
  if (size)
  {
    const UInt64 *p = v.ConstData();
    const UInt64 * const lim = p + size;
    do
      sum += *p++;
    while (p != lim);
  }
  return sum;
}


namespace NArchive {
namespace N7z {

static void FillSignature(Byte *buf)
{
  memcpy(buf, kSignature, kSignatureSize);
  buf[kSignatureSize] = kMajorVersion;
  buf[kSignatureSize + 1] = 4;
}

#ifdef Z7_7Z_VOL
HRESULT COutArchive::WriteFinishSignature()
{
  RINOK(WriteDirect(kFinishSignature, kSignatureSize));
  CArchiveVersion av;
  av.Major = kMajorVersion;
  av.Minor = 2;
  RINOK(WriteDirectByte(av.Major));
  return WriteDirectByte(av.Minor);
}
#endif

static void SetUInt32(Byte *p, UInt32 d)
{
  for (int i = 0; i < 4; i++, d >>= 8)
    p[i] = (Byte)d;
}

static void SetUInt64(Byte *p, UInt64 d)
{
  for (int i = 0; i < 8; i++, d >>= 8)
    p[i] = (Byte)d;
}

HRESULT COutArchive::WriteStartHeader(const CStartHeader &h)
{
  Byte buf[32];
  FillSignature(buf);
  SetUInt64(buf + 8 + 4, h.NextHeaderOffset);
  SetUInt64(buf + 8 + 12, h.NextHeaderSize);
  SetUInt32(buf + 8 + 20, h.NextHeaderCRC);
  SetUInt32(buf + 8, CrcCalc(buf + 8 + 4, 20));
  return WriteDirect(buf, sizeof(buf));
}

#ifdef Z7_7Z_VOL
HRESULT COutArchive::WriteFinishHeader(const CFinishHeader &h)
{
  CCRC crc;
  crc.UpdateUInt64(h.NextHeaderOffset);
  crc.UpdateUInt64(h.NextHeaderSize);
  crc.UpdateUInt32(h.NextHeaderCRC);
  crc.UpdateUInt64(h.ArchiveStartOffset);
  crc.UpdateUInt64(h.AdditionalStartBlockSize);
  RINOK(WriteDirectUInt32(crc.GetDigest()));
  RINOK(WriteDirectUInt64(h.NextHeaderOffset));
  RINOK(WriteDirectUInt64(h.NextHeaderSize));
  RINOK(WriteDirectUInt32(h.NextHeaderCRC));
  RINOK(WriteDirectUInt64(h.ArchiveStartOffset));
  return WriteDirectUInt64(h.AdditionalStartBlockSize);
}
#endif

HRESULT COutArchive::Create_and_WriteStartPrefix(ISequentialOutStream *stream /* , bool endMarker */)
{
  Close();
  #ifdef Z7_7Z_VOL
  // endMarker = false;
  _endMarker = endMarker;
  #endif
  SeqStream = stream;
  // if (!endMarker)
  {
    SeqStream.QueryInterface(IID_IOutStream, &Stream);
    if (!Stream)
    {
      return E_NOTIMPL;
      // endMarker = true;
    }
    RINOK(Stream->Seek(0, STREAM_SEEK_CUR, &_signatureHeaderPos))
    Byte buf[32];
    FillSignature(buf);
    memset(&buf[8], 0, 32 - 8);
    return WriteDirect(buf, sizeof(buf));
  }
  #ifdef Z7_7Z_VOL
  if (endMarker)
  {
    /*
    CStartHeader sh;
    sh.NextHeaderOffset = (UInt32)(Int32)-1;
    sh.NextHeaderSize = (UInt32)(Int32)-1;
    sh.NextHeaderCRC = 0;
    WriteStartHeader(sh);
    return S_OK;
    */
  }
  #endif
}

void COutArchive::Close()
{
  SeqStream.Release();
  Stream.Release();
}

UInt64 COutArchive::GetPos() const
{
  if (_countMode)
    return _countSize;
  if (_writeToStream)
    return _outByte.GetProcessedSize();
  return _outByte2.GetPos();
}

void COutArchive::WriteBytes(const void *data, size_t size)
{
  if (_countMode)
    _countSize += size;
  else if (_writeToStream)
  {
    _outByte.WriteBytes(data, size);
    // _crc = CrcUpdate(_crc, data, size);
  }
  else
    _outByte2.WriteBytes(data, size);
}

void COutArchive::WriteByte(Byte b)
{
  if (_countMode)
    _countSize++;
  else if (_writeToStream)
    WriteByte_ToStream(b);
  else
    _outByte2.WriteByte(b);
}

/*
void COutArchive::WriteUInt32(UInt32 value)
{
  for (int i = 0; i < 4; i++)
  {
    WriteByte((Byte)value);
    value >>= 8;
  }
}

void COutArchive::WriteUInt64(UInt64 value)
{
  for (int i = 0; i < 8; i++)
  {
    WriteByte((Byte)value);
    value >>= 8;
  }
}
*/

void COutArchive::WriteNumber(UInt64 value)
{
  Byte firstByte = 0;
  Byte mask = 0x80;
  int i;
  for (i = 0; i < 8; i++)
  {
    if (value < ((UInt64(1) << ( 7  * (i + 1)))))
    {
      firstByte |= Byte(value >> (8 * i));
      break;
    }
    firstByte |= mask;
    mask = (Byte)(mask >> 1);
  }
  WriteByte(firstByte);
  for (; i > 0; i--)
  {
    WriteByte((Byte)value);
    value >>= 8;
  }
}

static unsigned GetBigNumberSize(UInt64 value)
{
  unsigned i;
  for (i = 1; i < 9; i++)
    if (value < (((UInt64)1 << (i * 7))))
      break;
  return i;
}

#ifdef Z7_7Z_VOL
UInt32 COutArchive::GetVolHeadersSize(UInt64 dataSize, int nameLength, bool props)
{
  UInt32 result = GetBigNumberSize(dataSize) * 2 + 41;
  if (nameLength != 0)
  {
    nameLength = (nameLength + 1) * 2;
    result += nameLength + GetBigNumberSize(nameLength) + 2;
  }
  if (props)
  {
    result += 20;
  }
  if (result >= 128)
    result++;
  result += kSignatureSize + 2 + kFinishHeaderSize;
  return result;
}

UInt64 COutArchive::GetVolPureSize(UInt64 volSize, int nameLength, bool props)
{
  UInt32 headersSizeBase = COutArchive::GetVolHeadersSize(1, nameLength, props);
  int testSize;
  if (volSize > headersSizeBase)
    testSize = volSize - headersSizeBase;
  else
    testSize = 1;
  UInt32 headersSize = COutArchive::GetVolHeadersSize(testSize, nameLength, props);
  UInt64 pureSize = 1;
  if (volSize > headersSize)
    pureSize = volSize - headersSize;
  return pureSize;
}
#endif

void COutArchive::WriteFolder(const CFolder &folder)
{
  WriteNumber(folder.Coders.Size());
  unsigned i;
  
  for (i = 0; i < folder.Coders.Size(); i++)
  {
    const CCoderInfo &coder = folder.Coders[i];
    {
      UInt64 id = coder.MethodID;
      unsigned idSize;
      for (idSize = 1; idSize < sizeof(id); idSize++)
        if ((id >> (8 * idSize)) == 0)
          break;
      // idSize &= 0xF; // idSize is smaller than 16 already
      Byte temp[16];
      for (unsigned t = idSize; t != 0; t--, id >>= 8)
        temp[t] = (Byte)(id & 0xFF);
  
      unsigned b = idSize;
      const bool isComplex = !coder.IsSimpleCoder();
      b |= (isComplex ? 0x10 : 0);

      const size_t propsSize = coder.Props.Size();
      b |= ((propsSize != 0) ? 0x20 : 0);
      temp[0] = (Byte)b;
      WriteBytes(temp, idSize + 1);
      if (isComplex)
      {
        WriteNumber(coder.NumStreams);
        WriteNumber(1); // NumOutStreams;
      }
      if (propsSize == 0)
        continue;
      WriteNumber(propsSize);
      WriteBytes(coder.Props, propsSize);
    }
  }
  
  for (i = 0; i < folder.Bonds.Size(); i++)
  {
    const CBond &bond = folder.Bonds[i];
    WriteNumber(bond.PackIndex);
    WriteNumber(bond.UnpackIndex);
  }
  
  if (folder.PackStreams.Size() > 1)
    for (i = 0; i < folder.PackStreams.Size(); i++)
      WriteNumber(folder.PackStreams[i]);
}

void COutArchive::Write_BoolVector(const CBoolVector &boolVector)
{
  Byte b = 0;
  Byte mask = 0x80;
  FOR_VECTOR (i, boolVector)
  {
    if (boolVector[i])
      b |= mask;
    mask = (Byte)(mask >> 1);
    if (mask == 0)
    {
      WriteByte(b);
      mask = 0x80;
      b = 0;
    }
  }
  if (mask != 0x80)
    WriteByte(b);
}

static inline unsigned Bv_GetSizeInBytes(const CBoolVector &v) { return ((unsigned)v.Size() + 7) / 8; }

void COutArchive::WritePropBoolVector(Byte id, const CBoolVector &boolVector)
{
  WriteByte(id);
  WriteNumber(Bv_GetSizeInBytes(boolVector));
  Write_BoolVector(boolVector);
}

void COutArchive::Write_BoolVector_numDefined(const CBoolVector &boolVector, unsigned numDefined)
{
  if (numDefined == boolVector.Size())
    WriteByte(1);
  else
  {
    WriteByte(0);
    Write_BoolVector(boolVector);
  }
}


void COutArchive::WriteHashDigests(const CUInt32DefVector &digests)
{
  const unsigned numDefined = BoolVector_CountSum(digests.Defs);
  if (numDefined == 0)
    return;
  WriteByte(NID::kCRC);
  Write_BoolVector_numDefined(digests.Defs, numDefined);
  Write_UInt32DefVector_numDefined(digests, numDefined);
}


void COutArchive::WritePackInfo(
    UInt64 dataOffset,
    const CRecordVector<UInt64> &packSizes,
    const CUInt32DefVector &packCRCs)
{
  if (packSizes.IsEmpty())
    return;
  WriteByte(NID::kPackInfo);
  WriteNumber(dataOffset);
  WriteNumber(packSizes.Size());
  WriteByte(NID::kSize);
  FOR_VECTOR (i, packSizes)
    WriteNumber(packSizes[i]);

  WriteHashDigests(packCRCs);
  
  WriteByte(NID::kEnd);
}

void COutArchive::WriteUnpackInfo(const CObjectVector<CFolder> &folders, const COutFolders &outFolders)
{
  if (folders.IsEmpty())
    return;

  WriteByte(NID::kUnpackInfo);

  WriteByte(NID::kFolder);
  WriteNumber(folders.Size());
  {
    WriteByte(0);
    FOR_VECTOR (i, folders)
      WriteFolder(folders[i]);
  }
  
  WriteByte(NID::kCodersUnpackSize);
  FOR_VECTOR (i, outFolders.CoderUnpackSizes)
    WriteNumber(outFolders.CoderUnpackSizes[i]);
  
  WriteHashDigests(outFolders.FolderUnpackCRCs);
  
  WriteByte(NID::kEnd);
}

void COutArchive::WriteSubStreamsInfo(const CObjectVector<CFolder> &folders,
    const COutFolders &outFolders,
    const CRecordVector<UInt64> &unpackSizes,
    const CUInt32DefVector &digests)
{
  const CRecordVector<CNum> &numUnpackStreamsInFolders = outFolders.NumUnpackStreamsVector;
  WriteByte(NID::kSubStreamsInfo);

  unsigned i;
  for (i = 0; i < numUnpackStreamsInFolders.Size(); i++)
    if (numUnpackStreamsInFolders[i] != 1)
    {
      WriteByte(NID::kNumUnpackStream);
      for (i = 0; i < numUnpackStreamsInFolders.Size(); i++)
        WriteNumber(numUnpackStreamsInFolders[i]);
      break;
    }
 
  for (i = 0; i < numUnpackStreamsInFolders.Size(); i++)
    if (numUnpackStreamsInFolders[i] > 1)
    {
      WriteByte(NID::kSize);
      CNum index = 0;
      for (i = 0; i < numUnpackStreamsInFolders.Size(); i++)
      {
        CNum num = numUnpackStreamsInFolders[i];
        for (CNum j = 0; j < num; j++)
        {
          if (j + 1 != num)
            WriteNumber(unpackSizes[index]);
          index++;
        }
      }
      break;
    }

  CUInt32DefVector digests2;

  unsigned digestIndex = 0;
  for (i = 0; i < folders.Size(); i++)
  {
    unsigned numSubStreams = (unsigned)numUnpackStreamsInFolders[i];
    if (numSubStreams == 1 && outFolders.FolderUnpackCRCs.ValidAndDefined(i))
      digestIndex++;
    else
      for (unsigned j = 0; j < numSubStreams; j++, digestIndex++)
      {
        digests2.Defs.Add(digests.Defs[digestIndex]);
        digests2.Vals.Add(digests.Vals[digestIndex]);
      }
  }
  WriteHashDigests(digests2);
  WriteByte(NID::kEnd);
}

// 7-Zip 4.50 - 4.58 contain BUG, so they do not support .7z archives with Unknown field.

void COutArchive::SkipToAligned(unsigned pos, unsigned alignShifts)
{
  if (!_useAlign)
    return;

  const unsigned alignSize = (unsigned)1 << alignShifts;
  pos += (unsigned)GetPos();
  pos &= (alignSize - 1);
  if (pos == 0)
    return;
  unsigned skip = alignSize - pos;
  if (skip < 2)
    skip += alignSize;
  skip -= 2;
  WriteByte(NID::kDummy);
  WriteByte((Byte)skip);
  for (unsigned i = 0; i < skip; i++)
    WriteByte(0);
}

void COutArchive::WriteAlignedBools(const CBoolVector &v, unsigned numDefined, Byte type, unsigned itemSizeShifts)
{
  const unsigned bvSize = (numDefined == v.Size()) ? 0 : Bv_GetSizeInBytes(v);
  const UInt64 dataSize = ((UInt64)numDefined << itemSizeShifts) + bvSize + 2;
  SkipToAligned(3 + bvSize + GetBigNumberSize(dataSize), itemSizeShifts);

  WriteByte(type);
  WriteNumber(dataSize);
  Write_BoolVector_numDefined(v, numDefined);
  WriteByte(0); // 0 means no switching to external stream
}


void COutArchive::Write_UInt32DefVector_numDefined(const CUInt32DefVector &v, unsigned numDefined)
{
  if (_countMode)
  {
    _countSize += (size_t)numDefined * 4;
    return;
  }

  const bool * const defs = v.Defs.ConstData();
  const UInt32 * const vals = v.Vals.ConstData();
  const size_t num = v.Defs.Size();

  for (size_t i = 0; i < num; i++)
    if (defs[i])
    {
      UInt32 value = vals[i];
      for (int k = 0; k < 4; k++)
      {
        if (_writeToStream)
          WriteByte_ToStream((Byte)value);
        else
          _outByte2.WriteByte((Byte)value);
        // WriteByte((Byte)value);
        value >>= 8;
      }
      // WriteUInt32(v.Vals[i]);
    }
}


void COutArchive::Write_UInt64DefVector_type(const CUInt64DefVector &v, Byte type)
{
  const unsigned numDefined = BoolVector_CountSum(v.Defs);
  if (numDefined == 0)
    return;

  WriteAlignedBools(v.Defs, numDefined, type, 3);
  
  if (_countMode)
  {
    _countSize += (size_t)numDefined * 8;
    return;
  }

  const bool * const defs = v.Defs.ConstData();
  const UInt64 * const vals = v.Vals.ConstData();
  const size_t num = v.Defs.Size();

  for (size_t i = 0; i < num; i++)
    if (defs[i])
    {
      UInt64 value = vals[i];
      for (int k = 0; k < 8; k++)
      {
        if (_writeToStream)
          WriteByte_ToStream((Byte)value);
        else
          _outByte2.WriteByte((Byte)value);
        // WriteByte((Byte)value);
        value >>= 8;
      }
      // WriteUInt64(v.Vals[i]);
    }
}


HRESULT COutArchive::EncodeStream(
    DECL_EXTERNAL_CODECS_LOC_VARS
    CEncoder &encoder, const CByteBuffer &data,
    CRecordVector<UInt64> &packSizes, CObjectVector<CFolder> &folders, COutFolders &outFolders)
{
  CMyComPtr2_Create<ISequentialInStream, CBufInStream> streamSpec;
  streamSpec->Init(data, data.Size());
  outFolders.FolderUnpackCRCs.Defs.Add(true);
  outFolders.FolderUnpackCRCs.Vals.Add(CrcCalc(data, data.Size()));
  // outFolders.NumUnpackStreamsVector.Add(1);
  const UInt64 dataSize64 = data.Size();
  const UInt64 expectSize = data.Size();
  RINOK(encoder.Encode1(
      EXTERNAL_CODECS_LOC_VARS
      streamSpec,
      // NULL,
      &dataSize64,  // inSizeForReduce
      expectSize,
      folders.AddNew(),
      // outFolders.CoderUnpackSizes, unpackSize,
      SeqStream, packSizes, NULL))
  if (!streamSpec->WasFinished())
    return E_FAIL;
  encoder.Encode_Post(dataSize64, outFolders.CoderUnpackSizes);
  return S_OK;
}

void COutArchive::WriteHeader(
    const CArchiveDatabaseOut &db,
    // const CHeaderOptions &headerOptions,
    UInt64 &headerOffset)
{
  /*
  bool thereIsSecure = (db.SecureBuf.Size() != 0);
  */
  _useAlign = true;

  headerOffset = UInt64Vector_CountSum(db.PackSizes);

  WriteByte(NID::kHeader);

  /*
  {
    // It's example for per archive properies writing
  
    WriteByte(NID::kArchiveProperties);

    // you must use random 40-bit number that will identify you
    // then you can use same kDeveloperID for any properties and methods
    const UInt64 kDeveloperID = 0x123456789A; // change that value to real random 40-bit number

    #define GENERATE_7Z_ID(developerID, subID) (((UInt64)0x3F << 56) | ((UInt64)developerID << 16) | subID)

    {
      const UInt64 kSubID = 0x1; // you can use small number for subID
      const UInt64 kID = GENERATE_7Z_ID(kDeveloperID, kSubID);
      WriteNumber(kID);
      const unsigned kPropsSize = 3; // it's example size
      WriteNumber(kPropsSize);
      for (unsigned i = 0; i < kPropsSize; i++)
        WriteByte((Byte)(i & 0xFF));
    }
    {
      const UInt64 kSubID = 0x2; // you can use small number for subID
      const UInt64 kID = GENERATE_7Z_ID(kDeveloperID, kSubID);
      WriteNumber(kID);
      const unsigned kPropsSize = 5; // it's example size
      WriteNumber(kPropsSize);
      for (unsigned i = 0; i < kPropsSize; i++)
        WriteByte((Byte)(i + 16));
    }
    WriteByte(NID::kEnd);
  }
  */

  if (db.Folders.Size() > 0)
  {
    WriteByte(NID::kMainStreamsInfo);
    WritePackInfo(0, db.PackSizes, db.PackCRCs);
    WriteUnpackInfo(db.Folders, (const COutFolders &)db);

    CRecordVector<UInt64> unpackSizes;
    CUInt32DefVector digests;
    FOR_VECTOR (i, db.Files)
    {
      const CFileItem &file = db.Files[i];
      if (!file.HasStream)
        continue;
      unpackSizes.Add(file.Size);
      digests.Defs.Add(file.CrcDefined);
      digests.Vals.Add(file.Crc);
    }

    WriteSubStreamsInfo(db.Folders, (const COutFolders &)db, unpackSizes, digests);
    WriteByte(NID::kEnd);
  }

  if (db.Files.IsEmpty())
  {
    WriteByte(NID::kEnd);
    return;
  }

  WriteByte(NID::kFilesInfo);
  WriteNumber(db.Files.Size());

  {
    /* ---------- Empty Streams ---------- */
    CBoolVector emptyStreamVector;
    emptyStreamVector.ClearAndSetSize(db.Files.Size());
    unsigned numEmptyStreams = 0;
    {
      FOR_VECTOR (i, db.Files)
        if (db.Files[i].HasStream)
          emptyStreamVector[i] = false;
        else
        {
          emptyStreamVector[i] = true;
          numEmptyStreams++;
        }
    }

    if (numEmptyStreams != 0)
    {
      WritePropBoolVector(NID::kEmptyStream, emptyStreamVector);
      
      CBoolVector emptyFileVector, antiVector;
      emptyFileVector.ClearAndSetSize(numEmptyStreams);
      antiVector.ClearAndSetSize(numEmptyStreams);
      bool thereAreEmptyFiles = false, thereAreAntiItems = false;
      unsigned cur = 0;
      
      FOR_VECTOR (i, db.Files)
      {
        const CFileItem &file = db.Files[i];
        if (file.HasStream)
          continue;
        emptyFileVector[cur] = !file.IsDir;
        if (!file.IsDir)
          thereAreEmptyFiles = true;
        bool isAnti = db.IsItemAnti(i);
        antiVector[cur] = isAnti;
        if (isAnti)
          thereAreAntiItems = true;
        cur++;
      }
      
      if (thereAreEmptyFiles)
        WritePropBoolVector(NID::kEmptyFile, emptyFileVector);
      if (thereAreAntiItems)
        WritePropBoolVector(NID::kAnti, antiVector);
    }
  }


  {
    /* ---------- Names ---------- */
    
    size_t namesDataSize = 0;
    {
      FOR_VECTOR (i, db.Files)
      {
        const UString &name = db.Names[i];
        const size_t numUtfChars =
        /*
        #if WCHAR_MAX > 0xffff
        Get_Num_Utf16_chars_from_wchar_string(name.Ptr());
        #else
        */
        name.Len();
        // #endif
        namesDataSize += numUtfChars;
      }
    }
    if (namesDataSize)
    {
      namesDataSize += db.Files.Size();  // we will write tail zero wchar for each name
      namesDataSize *= 2;  // 2 bytes per wchar for UTF16 encoding
      namesDataSize++;     // for additional switch byte (zero value)
      SkipToAligned(2 + GetBigNumberSize(namesDataSize), 4);
      WriteByte(NID::kName);
      WriteNumber(namesDataSize);

      if (_countMode)
        _countSize += namesDataSize;
      else
      {
        WriteByte(0);
        FOR_VECTOR (i, db.Files)
        {
          const UString &name = db.Names[i];
          const wchar_t *p = name.Ptr();
          const size_t len = (size_t)name.Len() + 1;
          const wchar_t * const lim = p + len;
          if (_writeToStream)
          {
            do
            {
              const wchar_t c = *p++;
              WriteByte_ToStream((Byte)c);
              WriteByte_ToStream((Byte)(c >> 8));
            }
            while (p != lim);
          }
          else
          {
            Byte *dest = _outByte2.GetDest_and_Update(len * 2);
            do
            {
              /*
              #if WCHAR_MAX > 0xffff
              if (c >= 0x10000)
              {
                c -= 0x10000;
                if (c < (1 << 20))
                {
                  unsigned c0 = 0xd800 + ((c >> 10) & 0x3FF);
                  WriteByte((Byte)c0);
                  WriteByte((Byte)(c0 >> 8));
                  c = 0xdc00 + (c & 0x3FF);
                }
                else
                c = '_'; // we change character unsupported by UTF16
              }
              #endif
              */
              const wchar_t c = *p++;
              SetUi16(dest, (UInt16)c)
              dest += 2;
            }
            while (p != lim);
          }
        }
      }
    }
  }

  /* if (headerOptions.WriteCTime) */ Write_UInt64DefVector_type(db.CTime, NID::kCTime);
  /* if (headerOptions.WriteATime) */ Write_UInt64DefVector_type(db.ATime, NID::kATime);
  /* if (headerOptions.WriteMTime) */ Write_UInt64DefVector_type(db.MTime, NID::kMTime);
  Write_UInt64DefVector_type(db.StartPos, NID::kStartPos);
  
  {
    /* ---------- Write Attrib ---------- */
    const unsigned numDefined = BoolVector_CountSum(db.Attrib.Defs);
    if (numDefined != 0)
    {
      WriteAlignedBools(db.Attrib.Defs, numDefined, NID::kWinAttrib, 2);
      Write_UInt32DefVector_numDefined(db.Attrib, numDefined);
    }
  }

  /*
  {
    // ---------- Write IsAux ----------
    if (BoolVector_CountSum(db.IsAux) != 0)
      WritePropBoolVector(NID::kIsAux, db.IsAux);
  }

  {
    // ---------- Write Parent ----------
    CBoolVector boolVector;
    boolVector.Reserve(db.Files.Size());
    unsigned numIsDir = 0;
    unsigned numParentLinks = 0;
    for (i = 0; i < db.Files.Size(); i++)
    {
      const CFileItem &file = db.Files[i];
      bool defined = !file.IsAltStream;
      boolVector.Add(defined);
      if (defined)
        numIsDir++;
      if (file.Parent >= 0)
        numParentLinks++;
    }
    if (numParentLinks > 0)
    {
      // WriteAlignedBools(boolVector, numDefined, NID::kParent, 2);
      const unsigned bvSize = (numIsDir == boolVector.Size()) ? 0 : Bv_GetSizeInBytes(boolVector);
      const UInt64 dataSize = (UInt64)db.Files.Size() * 4 + bvSize + 1;
      SkipToAligned(2 + (unsigned)bvSize + (unsigned)GetBigNumberSize(dataSize), 2);
      
      WriteByte(NID::kParent);
      WriteNumber(dataSize);
      Write_BoolVector_numDefined(boolVector, numIsDir);
      for (i = 0; i < db.Files.Size(); i++)
      {
        const CFileItem &file = db.Files[i];
        // if (file.Parent >= 0)
          WriteUInt32(file.Parent);
      }
    }
  }

  if (thereIsSecure)
  {
    UInt64 secureDataSize = 1 + 4 +
       db.SecureBuf.Size() +
       db.SecureSizes.Size() * 4;
    // secureDataSize += db.SecureIDs.Size() * 4;
    for (i = 0; i < db.SecureIDs.Size(); i++)
      secureDataSize += GetBigNumberSize(db.SecureIDs[i]);
    SkipToAligned(2 + GetBigNumberSize(secureDataSize), 2);
    WriteByte(NID::kNtSecure);
    WriteNumber(secureDataSize);
    WriteByte(0);
    WriteUInt32(db.SecureSizes.Size());
    for (i = 0; i < db.SecureSizes.Size(); i++)
      WriteUInt32(db.SecureSizes[i]);
    WriteBytes(db.SecureBuf, db.SecureBuf.Size());
    for (i = 0; i < db.SecureIDs.Size(); i++)
    {
      WriteNumber(db.SecureIDs[i]);
      // WriteUInt32(db.SecureIDs[i]);
    }
  }
  */

  WriteByte(NID::kEnd); // for files
  WriteByte(NID::kEnd); // for headers
}

HRESULT COutArchive::WriteDatabase(
    DECL_EXTERNAL_CODECS_LOC_VARS
    const CArchiveDatabaseOut &db,
    const CCompressionMethodMode *options,
    const CHeaderOptions &headerOptions)
{
  if (!db.CheckNumFiles())
    return E_FAIL;

  CStartHeader sh;
  sh.NextHeaderOffset = 0;
  sh.NextHeaderSize = 0;
  sh.NextHeaderCRC = 0; // CrcCalc(NULL, 0);

  if (!db.IsEmpty())
  {
    CMyComPtr2_Create<ISequentialOutStream, COutStreamWithCRC> crcStream;
    crcStream->SetStream(SeqStream);
    crcStream->Init();

    bool encodeHeaders = false;
    if (options)
      if (options->IsEmpty())
        options = NULL;
    if (options)
      if (options->PasswordIsDefined || headerOptions.CompressMainHeader)
        encodeHeaders = true;

    if (!_outByte.Create(1 << 16))
      return E_OUTOFMEMORY;
    _outByte.SetStream(crcStream.Interface());
    _outByte.Init();
    // _crc = CRC_INIT_VAL;
    _countMode = encodeHeaders;
    _writeToStream = true;
    _countSize = 0;
    WriteHeader(db, /* headerOptions, */ sh.NextHeaderOffset);

    if (encodeHeaders)
    {
      CByteBuffer buf(_countSize);
      _outByte2.Init((Byte *)buf, _countSize);
      
      _countMode = false;
      _writeToStream = false;
      WriteHeader(db, /* headerOptions, */ sh.NextHeaderOffset);
      
      if (_countSize != _outByte2.GetPos())
        return E_FAIL;

      CCompressionMethodMode encryptOptions;
      encryptOptions.PasswordIsDefined = options->PasswordIsDefined;
      encryptOptions.Password = options->Password;
      CEncoder encoder(headerOptions.CompressMainHeader ? *options : encryptOptions);
      CRecordVector<UInt64> packSizes;
      CObjectVector<CFolder> folders;
      COutFolders outFolders;

      RINOK(EncodeStream(
          EXTERNAL_CODECS_LOC_VARS
          encoder, buf,
          packSizes, folders, outFolders))

      _writeToStream = true;
      
      if (folders.Size() == 0)
        throw 1;

      WriteID(NID::kEncodedHeader);
      WritePackInfo(sh.NextHeaderOffset, packSizes, CUInt32DefVector());
      WriteUnpackInfo(folders, outFolders);
      WriteByte(NID::kEnd);

      sh.NextHeaderOffset += UInt64Vector_CountSum(packSizes);
    }
    RINOK(_outByte.Flush())
    sh.NextHeaderCRC = crcStream->GetCRC();
    // sh.NextHeaderCRC = CRC_GET_DIGEST(_crc);
    // if (CRC_GET_DIGEST(_crc) != sh.NextHeaderCRC) throw 1;
    sh.NextHeaderSize = _outByte.GetProcessedSize();
  }
  #ifdef Z7_7Z_VOL
  if (_endMarker)
  {
    CFinishHeader h;
    h.NextHeaderSize = headerSize;
    h.NextHeaderCRC = headerCRC;
    h.NextHeaderOffset =
        UInt64(0) - (headerSize +
        4 + kFinishHeaderSize);
    h.ArchiveStartOffset = h.NextHeaderOffset - headerOffset;
    h.AdditionalStartBlockSize = 0;
    RINOK(WriteFinishHeader(h));
    return WriteFinishSignature();
  }
  else
  #endif
  if (Stream)
  {
    RINOK(Stream->Seek((Int64)_signatureHeaderPos, STREAM_SEEK_SET, NULL))
    return WriteStartHeader(sh);
  }
  return S_OK;
}

void CUInt32DefVector::SetItem(unsigned index, bool defined, UInt32 value)
{
  while (index >= Defs.Size())
    Defs.Add(false);
  Defs[index] = defined;
  if (!defined)
    return;
  while (index >= Vals.Size())
    Vals.Add(0);
  Vals[index] = value;
}

void CUInt64DefVector::SetItem(unsigned index, bool defined, UInt64 value)
{
  while (index >= Defs.Size())
    Defs.Add(false);
  Defs[index] = defined;
  if (!defined)
    return;
  while (index >= Vals.Size())
    Vals.Add(0);
  Vals[index] = value;
}

void CArchiveDatabaseOut::AddFile(const CFileItem &file, const CFileItem2 &file2, const UString &name)
{
  unsigned index = Files.Size();
  CTime.SetItem(index, file2.CTimeDefined, file2.CTime);
  ATime.SetItem(index, file2.ATimeDefined, file2.ATime);
  MTime.SetItem(index, file2.MTimeDefined, file2.MTime);
  StartPos.SetItem(index, file2.StartPosDefined, file2.StartPos);
  Attrib.SetItem(index, file2.AttribDefined, file2.Attrib);
  SetItem_Anti(index, file2.IsAnti);
  // SetItem_Aux(index, file2.IsAux);
  Names.Add(name);
  Files.Add(file);
}

}}
