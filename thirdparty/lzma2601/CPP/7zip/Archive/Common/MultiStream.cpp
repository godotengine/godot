// MultiStream.cpp

#include "StdAfx.h"

#include "MultiStream.h"

Z7_COM7F_IMF(CMultiStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;
  if (_pos >= _totalLength)
    return S_OK;

  {
    unsigned left = 0, mid = _streamIndex, right = Streams.Size();
    for (;;)
    {
      CSubStreamInfo &m = Streams[mid];
      if (_pos < m.GlobalOffset)
        right = mid;
      else if (_pos >= m.GlobalOffset + m.Size)
        left = mid + 1;
      else
        break;
      mid = (left + right) / 2;
    }
    _streamIndex = mid;
  }
  
  CSubStreamInfo &s = Streams[_streamIndex];
  UInt64 localPos = _pos - s.GlobalOffset;
  if (localPos != s.LocalPos)
  {
    RINOK(s.Stream->Seek((Int64)localPos, STREAM_SEEK_SET, &s.LocalPos))
  }
  {
    const UInt64 rem = s.Size - localPos;
    if (size > rem)
      size = (UInt32)rem;
  }
  const HRESULT result = s.Stream->Read(data, size, &size);
  _pos += size;
  s.LocalPos += size;
  if (processedSize)
    *processedSize = size;
  return result;
}
  
Z7_COM7F_IMF(CMultiStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _pos; break;
    case STREAM_SEEK_END: offset += _totalLength; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _pos = (UInt64)offset;
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}


/*
class COutVolumeStream:
  public ISequentialOutStream,
  public CMyUnknownImp
{
  Z7_COM_UNKNOWN_IMP_0
  Z7_IFACE_COM7_IMP(ISequentialOutStream)

  unsigned _volIndex;
  UInt64 _volSize;
  UInt64 _curPos;
  CMyComPtr<ISequentialOutStream> _volumeStream;
  COutArchive _archive;
  CCRC _crc;

public:
  CFileItem _file;
  CUpdateOptions _options;
  CMyComPtr<IArchiveUpdateCallback2> VolumeCallback;
  void Init(IArchiveUpdateCallback2 *volumeCallback,
      const UString &name)
  {
    _file.Name = name;
    _file.IsStartPosDefined = true;
    _file.StartPos = 0;
    
    VolumeCallback = volumeCallback;
    _volIndex = 0;
    _volSize = 0;
  }
  
  HRESULT Flush();
};

HRESULT COutVolumeStream::Flush()
{
  if (_volumeStream)
  {
    _file.UnPackSize = _curPos;
    _file.FileCRC = _crc.GetDigest();
    RINOK(WriteVolumeHeader(_archive, _file, _options))
    _archive.Close();
    _volumeStream.Release();
    _file.StartPos += _file.UnPackSize;
  }
  return S_OK;
}
*/

/*

#include "../../../Common/Defs.h"

Z7_COM7F_IMF(COutMultiStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  while (size > 0)
  {
    if (_streamIndex >= Streams.Size())
    {
      CSubStreamInfo subStream;
      RINOK(VolumeCallback->GetVolumeSize(Streams.Size(), &subStream.Size))
      RINOK(VolumeCallback->GetVolumeStream(Streams.Size(), &subStream.Stream))
      subStream.Pos = 0;
      Streams.Add(subStream);
      continue;
    }
    CSubStreamInfo &subStream = Streams[_streamIndex];
    if (_offsetPos >= subStream.Size)
    {
      _offsetPos -= subStream.Size;
      _streamIndex++;
      continue;
    }
    if (_offsetPos != subStream.Pos)
    {
      CMyComPtr<IOutStream> outStream;
      RINOK(subStream.Stream.QueryInterface(IID_IOutStream, &outStream))
      RINOK(outStream->Seek((Int64)_offsetPos, STREAM_SEEK_SET, NULL))
      subStream.Pos = _offsetPos;
    }

    const UInt32 curSize = (UInt32)MyMin((UInt64)size, subStream.Size - subStream.Pos);
    UInt32 realProcessed;
    RINOK(subStream.Stream->Write(data, curSize, &realProcessed))
    data = (const void *)((const Byte *)data + realProcessed);
    size -= realProcessed;
    subStream.Pos += realProcessed;
    _offsetPos += realProcessed;
    _absPos += realProcessed;
    if (_absPos > _length)
      _length = _absPos;
    if (processedSize)
      *processedSize += realProcessed;
    if (subStream.Pos == subStream.Size)
    {
      _streamIndex++;
      _offsetPos = 0;
    }
    if (realProcessed != curSize && realProcessed == 0)
      return E_FAIL;
  }
  return S_OK;
}

Z7_COM7F_IMF(COutMultiStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _absPos; break;
    case STREAM_SEEK_END: offset += _length; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _absPos = (UInt64)offset;
  _offsetPos = _absPos;
  _streamIndex = 0;
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}
*/
