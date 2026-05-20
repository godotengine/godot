// 7zFolderInStream.cpp

#include "StdAfx.h"

#include "../../../Windows/TimeUtils.h"

#include "7zFolderInStream.h"

namespace NArchive {
namespace N7z {

void CFolderInStream::Init(IArchiveUpdateCallback *updateCallback,
    const UInt32 *indexes, unsigned numFiles)
{
  _updateCallback = updateCallback;
  _indexes = indexes;
  _numFiles = numFiles;

  _totalSize_for_Coder = 0;
  ClearFileInfo();
  
  Processed.ClearAndReserve(numFiles);
  Sizes.ClearAndReserve(numFiles);
  CRCs.ClearAndReserve(numFiles);
  TimesDefined.ClearAndReserve(numFiles);
  MTimes.ClearAndReserve(Need_MTime ? numFiles : (unsigned)0);
  CTimes.ClearAndReserve(Need_CTime ? numFiles : (unsigned)0);
  ATimes.ClearAndReserve(Need_ATime ? numFiles : (unsigned)0);
  Attribs.ClearAndReserve(Need_Attrib ? numFiles : (unsigned)0);

  // FolderCrc = CRC_INIT_VAL;
  _stream.Release();
}

void CFolderInStream::ClearFileInfo()
{
  _pos = 0;
  _crc = CRC_INIT_VAL;
  _size_Defined = false;
  _times_Defined = false;
  _size = 0;
  FILETIME_Clear(_mTime);
  FILETIME_Clear(_cTime);
  FILETIME_Clear(_aTime);
  _attrib = 0;
}

HRESULT CFolderInStream::OpenStream()
{
  while (Processed.Size() < _numFiles)
  {
    CMyComPtr<ISequentialInStream> stream;
    const HRESULT result = _updateCallback->GetStream(_indexes[Processed.Size()], &stream);
    if (result != S_OK && result != S_FALSE)
      return result;

    _stream = stream;
    
    if (stream)
    {
      {
        CMyComPtr<IStreamGetProps> getProps;
        stream.QueryInterface(IID_IStreamGetProps, (void **)&getProps);
        if (getProps)
        {
          // access could be changed in first myx pass
          if (getProps->GetProps(&_size,
              Need_CTime ? &_cTime : NULL,
              Need_ATime ? &_aTime : NULL,
              Need_MTime ? &_mTime : NULL,
              Need_Attrib ? &_attrib : NULL)
              == S_OK)
          {
            _size_Defined = true;
            _times_Defined = true;
          }
          return S_OK;
        }
      }
      {
        CMyComPtr<IStreamGetSize> streamGetSize;
        stream.QueryInterface(IID_IStreamGetSize, &streamGetSize);
        if (streamGetSize)
        {
          if (streamGetSize->GetSize(&_size) == S_OK)
            _size_Defined = true;
        }
        return S_OK;
      }
    }
    
    RINOK(AddFileInfo(result == S_OK))
  }
  return S_OK;
}

static void AddFt(CRecordVector<UInt64> &vec, const FILETIME &ft)
{
  vec.AddInReserved(FILETIME_To_UInt64(ft));
}

/*
HRESULT ReportItemProps(IArchiveUpdateCallbackArcProp *reportArcProp,
    UInt32 index, UInt64 size, const UInt32 *crc)
{
  PROPVARIANT prop;
  prop.vt = VT_EMPTY;
  prop.wReserved1 = 0;
  
  NWindows::NCOM::PropVarEm_Set_UInt64(&prop, size);
  RINOK(reportArcProp->ReportProp(NEventIndexType::kOutArcIndex, index, kpidSize, &prop));
  if (crc)
  {
    NWindows::NCOM::PropVarEm_Set_UInt32(&prop, *crc);
    RINOK(reportArcProp->ReportProp(NEventIndexType::kOutArcIndex, index, kpidCRC, &prop));
  }
  return reportArcProp->ReportFinished(NEventIndexType::kOutArcIndex, index, NUpdate::NOperationResult::kOK);
}
*/

HRESULT CFolderInStream::AddFileInfo(bool isProcessed)
{
  // const UInt32 index = _indexes[Processed.Size()];
  Processed.AddInReserved(isProcessed);
  Sizes.AddInReserved(_pos);
  CRCs.AddInReserved(CRC_GET_DIGEST(_crc));
  if (Need_Attrib) Attribs.AddInReserved(_attrib);
  TimesDefined.AddInReserved(_times_Defined);
  if (Need_MTime) AddFt(MTimes, _mTime);
  if (Need_CTime) AddFt(CTimes, _cTime);
  if (Need_ATime) AddFt(ATimes, _aTime);
  ClearFileInfo();
  /*
  if (isProcessed && _reportArcProp)
    RINOK(ReportItemProps(_reportArcProp, index, _pos, &crc))
  */
  return _updateCallback->SetOperationResult(NArchive::NUpdate::NOperationResult::kOK);
}

Z7_COM7F_IMF(CFolderInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  while (size != 0)
  {
    if (_stream)
    {
      /*
      if (_pos == 0)
      {
        const UInt32 align = (UInt32)1 << AlignLog;
        const UInt32 offset = (UInt32)_totalSize_for_Coder & (align - 1);
        if (offset != 0)
        {
          UInt32 cur = align - offset;
          if (cur > size)
            cur = size;
          memset(data, 0, cur);
          data = (Byte *)data + cur;
          size -= cur;
          // _pos += cur; // for debug
          _totalSize_for_Coder += cur;
          if (processedSize)
            *processedSize += cur;
          continue;
        }
      }
      */
      UInt32 cur = size;
      const UInt32 kMax = (UInt32)1 << 20;
      if (cur > kMax)
        cur = kMax;
      RINOK(_stream->Read(data, cur, &cur))
      if (cur != 0)
      {
        // if (Need_Crc)
        _crc = CrcUpdate(_crc, data, cur);
        /*
        if (FolderCrc)
          FolderCrc = CrcUpdate(FolderCrc, data, cur);
        */
        _pos += cur;
        _totalSize_for_Coder += cur;
        if (processedSize)
          *processedSize = cur; // use +=cur, if continue is possible in loop
        return S_OK;
      }
      
      _stream.Release();
      RINOK(AddFileInfo(true))
    }
    
    if (Processed.Size() >= _numFiles)
      break;
    RINOK(OpenStream())
  }
  return S_OK;
}

Z7_COM7F_IMF(CFolderInStream::GetSubStreamSize(UInt64 subStream, UInt64 *value))
{
  *value = 0;
  if (subStream > Sizes.Size())
    return S_FALSE; // E_FAIL;
  
  const unsigned index = (unsigned)subStream;
  if (index < Sizes.Size())
  {
    *value = Sizes[index];
    return S_OK;
  }
  
  if (!_size_Defined)
  {
    *value = _pos;
    return S_FALSE;
  }
  
  *value = (_pos > _size ? _pos : _size);
  return S_OK;
}


/*
HRESULT CFolderInStream::CloseCrcStream()
{
  if (!_crcStream)
    return S_OK;
  if (!_crcStream_Spec->WasFinished())
    return E_FAIL;
  _crc = _crcStream_Spec->GetCRC() ^ CRC_INIT_VAL; // change it
  const UInt64 size = _crcStream_Spec->GetSize();
  _pos = size;
  _totalSize_for_Coder += size;
  _crcStream.Release();
  // _crcStream->ReleaseStream();
  _stream.Release();
  return AddFileInfo(true);
}

Z7_COM7F_IMF(CFolderInStream::GetNextInSubStream(UInt64 *streamIndexRes, ISequentialInStream **stream)
{
  RINOK(CloseCrcStream())
  *stream = NULL;
  *streamIndexRes = Processed.Size();
  if (Processed.Size() >= _numFiles)
    return S_OK;
  RINOK(OpenStream());
  if (!_stream)
    return S_OK;
  if (!_crcStream)
  {
    _crcStream_Spec = new CSequentialInStreamWithCRC;
    _crcStream = _crcStream_Spec;
  }
  _crcStream_Spec->Init();
  _crcStream_Spec->SetStream(_stream);
  *stream = _crcStream;
  _crcStream->AddRef();
  return S_OK;
}
*/

}}
