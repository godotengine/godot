// 7zIn.h

#ifndef ZIP7_INC_7Z_IN_H
#define ZIP7_INC_7Z_IN_H

#include "../../../Common/MyCom.h"

#include "../../../Windows/PropVariant.h"

#include "../../IPassword.h"
#include "../../IStream.h"

#include "../../Common/CreateCoder.h"
#include "../../Common/InBuffer.h"

#include "7zItem.h"
 
namespace NArchive {
namespace N7z {

/*
  We don't need to init isEncrypted and passwordIsDefined
  We must upgrade them only */

#ifdef Z7_NO_CRYPTO
#define Z7_7Z_DECODER_CRYPRO_VARS_DECL
#define Z7_7Z_DECODER_CRYPRO_VARS
#else
#define Z7_7Z_DECODER_CRYPRO_VARS_DECL , ICryptoGetTextPassword *getTextPassword, bool &isEncrypted, bool &passwordIsDefined, UString &password
#define Z7_7Z_DECODER_CRYPRO_VARS , getTextPassword, isEncrypted, passwordIsDefined, password
#endif

struct CParsedMethods
{
  Byte Lzma2Prop;
  UInt32 LzmaDic;
  CRecordVector<UInt64> IDs;

  CParsedMethods(): Lzma2Prop(0), LzmaDic(0) {}
};

struct CFolderEx: public CFolder
{
  unsigned UnpackCoder;
};

struct CFolders
{
  CNum NumPackStreams;
  CNum NumFolders;

  CObjArray<UInt64> PackPositions; // NumPackStreams + 1
  // CUInt32DefVector PackCRCs; // we don't use PackCRCs now

  CUInt32DefVector FolderCRCs;             // NumFolders
  CObjArray<CNum> NumUnpackStreamsVector;  // NumFolders

  CObjArray<UInt64> CoderUnpackSizes;      // including unpack sizes of bond coders
  CObjArray<CNum> FoToCoderUnpackSizes;    // NumFolders + 1
  CObjArray<CNum> FoStartPackStreamIndex;  // NumFolders + 1
  CObjArray<Byte> FoToMainUnpackSizeIndex; // NumFolders
  
  CObjArray<size_t> FoCodersDataOffset;    // NumFolders + 1
  CByteBuffer CodersData;

  CParsedMethods ParsedMethods;

  void ParseFolderInfo(unsigned folderIndex, CFolder &folder) const;
  void ParseFolderEx(unsigned folderIndex, CFolderEx &folder) const
  {
    ParseFolderInfo(folderIndex, folder);
    folder.UnpackCoder = FoToMainUnpackSizeIndex[folderIndex];
  }
  
  unsigned GetNumFolderUnpackSizes(unsigned folderIndex) const
  {
    return (unsigned)(FoToCoderUnpackSizes[folderIndex + 1] - FoToCoderUnpackSizes[folderIndex]);
  }

  UInt64 GetFolderUnpackSize(unsigned folderIndex) const
  {
    return CoderUnpackSizes[FoToCoderUnpackSizes[folderIndex] + FoToMainUnpackSizeIndex[folderIndex]];
  }

  UInt64 GetStreamPackSize(unsigned index) const
  {
    return PackPositions[index + 1] - PackPositions[index];
  }

  CFolders(): NumPackStreams(0), NumFolders(0) {}

  void Clear()
  {
    NumPackStreams = 0;
    PackPositions.Free();
    // PackCRCs.Clear();

    NumFolders = 0;
    FolderCRCs.Clear();
    NumUnpackStreamsVector.Free();
    CoderUnpackSizes.Free();
    FoToCoderUnpackSizes.Free();
    FoStartPackStreamIndex.Free();
    FoToMainUnpackSizeIndex.Free();
    FoCodersDataOffset.Free();
    CodersData.Free();
  }
};

struct CDatabase: public CFolders
{
  CRecordVector<CFileItem> Files;

  CUInt64DefVector CTime;
  CUInt64DefVector ATime;
  CUInt64DefVector MTime;
  CUInt64DefVector StartPos;
  CUInt32DefVector Attrib;
  CBoolVector IsAnti;
  /*
  CBoolVector IsAux;
  CByteBuffer SecureBuf;
  CRecordVector<UInt32> SecureIDs;
  */

  CByteBuffer NamesBuf;
  CObjArray<size_t> NameOffsets; // numFiles + 1, offsets of utf-16 symbols

  /*
  void ClearSecure()
  {
    SecureBuf.Free();
    SecureIDs.Clear();
  }
  */

  void Clear()
  {
    CFolders::Clear();
    // ClearSecure();

    NamesBuf.Free();
    NameOffsets.Free();
    
    Files.Clear();
    CTime.Clear();
    ATime.Clear();
    MTime.Clear();
    StartPos.Clear();
    Attrib.Clear();
    IsAnti.Clear();
    // IsAux.Clear();
  }

  bool IsSolid() const
  {
    for (CNum i = 0; i < NumFolders; i++)
      if (NumUnpackStreamsVector[i] > 1)
        return true;
    return false;
  }
  bool IsItemAnti(unsigned index) const { return (index < IsAnti.Size() && IsAnti[index]); }
  // bool IsItemAux(unsigned index) const { return (index < IsAux.Size() && IsAux[index]); }

  /*
  const void* GetName(unsigned index) const
  {
    if (!NameOffsets || !NamesBuf)
      return NULL;
    return (void *)((const Byte *)NamesBuf + NameOffsets[index] * 2);
  };
  */
  void GetPath(unsigned index, UString &path) const;
  HRESULT GetPath_Prop(unsigned index, PROPVARIANT *path) const throw();
};


struct CInArchiveInfo
{
  CArchiveVersion Version;
  UInt64 StartPosition;               // in stream
  UInt64 StartPositionAfterHeader;    // in stream
  UInt64 DataStartPosition;           // in stream
  UInt64 DataStartPosition2;          // in stream. it's for headers
  CRecordVector<UInt64> FileInfoPopIDs;
  
  void Clear()
  {
    StartPosition = 0;
    StartPositionAfterHeader = 0;
    DataStartPosition = 0;
    DataStartPosition2 = 0;
    FileInfoPopIDs.Clear();
  }
};


struct CDbEx: public CDatabase
{
  CInArchiveInfo ArcInfo;
  
  CObjArray<CNum> FolderStartFileIndex;
  CObjArray<CNum> FileIndexToFolderIndexMap;

  UInt64 HeadersSize;
  UInt64 PhySize;
  // UInt64 EndHeaderOffset; // relative to position after StartHeader (32 bytes)

  /*
  CRecordVector<size_t> SecureOffsets;
  bool IsTree;
  bool ThereAreAltStreams;
  */

  bool IsArc;
  bool PhySizeWasConfirmed;

  bool ThereIsHeaderError;
  bool UnexpectedEnd;
  // bool UnsupportedVersion;

  bool StartHeaderWasRecovered;
  bool UnsupportedFeatureWarning;
  bool UnsupportedFeatureError;

  /*
  void ClearSecureEx()
  {
    ClearSecure();
    SecureOffsets.Clear();
  }
  */

  void Clear()
  {
    IsArc = false;
    PhySizeWasConfirmed = false;

    ThereIsHeaderError = false;
    UnexpectedEnd = false;
    // UnsupportedVersion = false;

    StartHeaderWasRecovered = false;
    UnsupportedFeatureError = false;
    UnsupportedFeatureWarning = false;

    /*
    IsTree = false;
    ThereAreAltStreams = false;
    */

    CDatabase::Clear();
    
    // SecureOffsets.Clear();
    ArcInfo.Clear();
    FolderStartFileIndex.Free();
    FileIndexToFolderIndexMap.Free();

    HeadersSize = 0;
    PhySize = 0;
    // EndHeaderOffset = 0;
  }

  bool CanUpdate() const
  {
    if (ThereIsHeaderError
        || UnexpectedEnd
        || StartHeaderWasRecovered
        || UnsupportedFeatureError)
      return false;
    return true;
  }

  void FillLinks();
  
  UInt64 GetFolderStreamPos(size_t folderIndex, size_t indexInFolder) const
  {
    return ArcInfo.DataStartPosition + PackPositions.ConstData()
        [FoStartPackStreamIndex.ConstData()[folderIndex] + indexInFolder];
  }
  
  UInt64 GetFolderFullPackSize(size_t folderIndex) const
  {
    return
      PackPositions[FoStartPackStreamIndex.ConstData()[folderIndex + 1]] -
      PackPositions[FoStartPackStreamIndex.ConstData()[folderIndex]];
  }
  
  UInt64 GetFolderPackStreamSize(size_t folderIndex, size_t streamIndex) const
  {
    const size_t i = FoStartPackStreamIndex.ConstData()[folderIndex] + streamIndex;
    return PackPositions.ConstData()[i + 1] -
           PackPositions.ConstData()[i];
  }

  /*
  UInt64 GetFilePackSize(size_t fileIndex) const
  {
    const CNum folderIndex = FileIndexToFolderIndexMap[fileIndex];
    if (folderIndex != kNumNoIndex)
      if (FolderStartFileIndex[folderIndex] == fileIndex)
        return GetFolderFullPackSize(folderIndex);
    return 0;
  }
  */
};

const unsigned kNumBufLevelsMax = 4;

struct CInByte2
{
  const Byte *_buffer;
public:
  size_t _size;
  size_t _pos;
  
  size_t GetRem() const { return _size - _pos; }
  const Byte *GetPtr() const { return _buffer + _pos; }
  void Init(const Byte *buffer, size_t size)
  {
    _buffer = buffer;
    _size = size;
    _pos = 0;
  }
  Byte ReadByte();
  void ReadBytes(Byte *data, size_t size);
  void SkipDataNoCheck(UInt64 size) { _pos += (size_t)size; }
  void SkipData(UInt64 size);

  void SkipData();
  void SkipRem() { _pos = _size; }
  UInt64 ReadNumber();
  CNum ReadNum();
  UInt32 ReadUInt32();
  UInt64 ReadUInt64();

  void ParseFolder(CFolder &folder);
};

class CStreamSwitch;

const UInt32 kHeaderSize = 32;

class CInArchive
{
  friend class CStreamSwitch;

  CMyComPtr<IInStream> _stream;

  unsigned _numInByteBufs;
  CInByte2 _inByteVector[kNumBufLevelsMax];

  CInByte2 *_inByteBack;
  bool ThereIsHeaderError;
 
  UInt64 _arhiveBeginStreamPosition;
  UInt64 _fileEndPosition;

  UInt64 _rangeLimit; // relative to position after StartHeader (32 bytes)

  Byte _header[kHeaderSize];

  UInt64 HeadersSize;

  bool _useMixerMT;

  void AddByteStream(const Byte *buffer, size_t size);
  
  void DeleteByteStream(bool needUpdatePos)
  {
    _numInByteBufs--;
    if (_numInByteBufs > 0)
    {
      _inByteBack = &_inByteVector[_numInByteBufs - 1];
      if (needUpdatePos)
        _inByteBack->_pos += _inByteVector[_numInByteBufs]._pos;
    }
  }

  HRESULT FindAndReadSignature(IInStream *stream, const UInt64 *searchHeaderSizeLimit);
  
  void ReadBytes(Byte *data, size_t size) { _inByteBack->ReadBytes(data, size); }
  Byte ReadByte() { return _inByteBack->ReadByte(); }
  UInt64 ReadNumber() { return _inByteBack->ReadNumber(); }
  CNum ReadNum() { return _inByteBack->ReadNum(); }
  UInt64 ReadID() { return _inByteBack->ReadNumber(); }
  UInt32 ReadUInt32() { return _inByteBack->ReadUInt32(); }
  UInt64 ReadUInt64() { return _inByteBack->ReadUInt64(); }
  void SkipData(UInt64 size) { _inByteBack->SkipData(size); }
  void SkipData() { _inByteBack->SkipData(); }
  void WaitId(UInt64 id);

  void Read_UInt32_Vector(CUInt32DefVector &v);

  void ReadArchiveProperties(CInArchiveInfo &archiveInfo);
  void ReadHashDigests(unsigned numItems, CUInt32DefVector &crcs);
  
  void ReadPackInfo(CFolders &f);
  
  void ReadUnpackInfo(
      const CObjectVector<CByteBuffer> *dataVector,
      CFolders &folders);
  
  void ReadSubStreamsInfo(
      CFolders &folders,
      CRecordVector<UInt64> &unpackSizes,
      CUInt32DefVector &digests);

  void ReadStreamsInfo(
      const CObjectVector<CByteBuffer> *dataVector,
      UInt64 &dataOffset,
      CFolders &folders,
      CRecordVector<UInt64> &unpackSizes,
      CUInt32DefVector &digests);

  void ReadBoolVector(unsigned numItems, CBoolVector &v);
  void ReadBoolVector2(unsigned numItems, CBoolVector &v);
  void ReadUInt64DefVector(const CObjectVector<CByteBuffer> &dataVector,
      CUInt64DefVector &v, unsigned numItems);
  HRESULT ReadAndDecodePackedStreams(
      DECL_EXTERNAL_CODECS_LOC_VARS
      UInt64 baseOffset, UInt64 &dataOffset,
      CObjectVector<CByteBuffer> &dataVector
      Z7_7Z_DECODER_CRYPRO_VARS_DECL
      );
  HRESULT ReadHeader(
      DECL_EXTERNAL_CODECS_LOC_VARS
      CDbEx &db
      Z7_7Z_DECODER_CRYPRO_VARS_DECL
      );
  HRESULT ReadDatabase2(
      DECL_EXTERNAL_CODECS_LOC_VARS
      CDbEx &db
      Z7_7Z_DECODER_CRYPRO_VARS_DECL
      );
public:
  CInArchive(bool useMixerMT):
      _numInByteBufs(0),
      _useMixerMT(useMixerMT)
      {}
  
  HRESULT Open(IInStream *stream, const UInt64 *searchHeaderSizeLimit); // S_FALSE means is not archive
  void Close();

  HRESULT ReadDatabase(
      DECL_EXTERNAL_CODECS_LOC_VARS
      CDbEx &db
      Z7_7Z_DECODER_CRYPRO_VARS_DECL
      );
};
  
}}
  
#endif
