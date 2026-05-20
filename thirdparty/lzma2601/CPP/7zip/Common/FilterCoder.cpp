// FilterCoder.cpp

#include "StdAfx.h"

// #include <stdio.h>

#include "../../Common/Defs.h"

#include "FilterCoder.h"
#include "StreamUtils.h"

#ifdef _WIN32
  #define alignedMidBuffer_Alloc g_MidAlloc
#else
  #define alignedMidBuffer_Alloc g_AlignedAlloc
#endif

CAlignedMidBuffer::~CAlignedMidBuffer()
{
  ISzAlloc_Free(&alignedMidBuffer_Alloc, _buf);
}

void CAlignedMidBuffer::AllocAligned(size_t size)
{
  ISzAlloc_Free(&alignedMidBuffer_Alloc, _buf);
  _buf = (Byte *)ISzAlloc_Alloc(&alignedMidBuffer_Alloc, size);
}

/*
  AES filters need 16-bytes alignment for HARDWARE-AES instructions.
  So we call IFilter::Filter(, size), where (size != 16 * N) only for last data block.

  AES-CBC filters need data size aligned for 16-bytes.
  So the encoder can add zeros to the end of original stream.

  Some filters (BCJ and others) don't process data at the end of stream in some cases.
  So the encoder and decoder write such last bytes without change.

  Most filters process all data, if we send aligned size to filter.
     But  BCJ filter can process up 4 bytes less than sent size.
     And ARMT filter can process    2 bytes less than sent size.
*/


static const UInt32 kBufSize = 1 << 21;

Z7_COM7F_IMF(CFilterCoder::SetInBufSize(UInt32 , UInt32 size)) { _inBufSize = size; return S_OK; }
Z7_COM7F_IMF(CFilterCoder::SetOutBufSize(UInt32 , UInt32 size)) { _outBufSize = size; return S_OK; }

HRESULT CFilterCoder::Alloc()
{
  UInt32 size = MyMin(_inBufSize, _outBufSize);
  /* minimal bufSize is 16 bytes for AES and IA64 filter.
     bufSize for AES must be aligned for 16 bytes.
     We use (1 << 12) min size to support future aligned filters. */
  const UInt32 kMinSize = 1 << 12;
  size &= ~(UInt32)(kMinSize - 1);
  if (size < kMinSize)
    size = kMinSize;
  // size = (1 << 12); // + 117; // for debug
  if (!_buf || _bufSize != size)
  {
    AllocAligned(size);
    if (!_buf)
      return E_OUTOFMEMORY;
    _bufSize = size;
  }
  return S_OK;
}

HRESULT CFilterCoder::Init_and_Alloc()
{
  RINOK(Filter->Init())
  return Alloc();
}

CFilterCoder::CFilterCoder(bool encodeMode):
    _bufSize(0),
    _inBufSize(kBufSize),
    _outBufSize(kBufSize),
    _encodeMode(encodeMode),
    _outSize_Defined(false),
    _outSize(0),
    _nowPos64(0)
  {}


Z7_COM7F_IMF(CFilterCoder::Code(ISequentialInStream *inStream, ISequentialOutStream *outStream,
    const UInt64 * /* inSize */, const UInt64 *outSize, ICompressProgressInfo *progress))
{
  RINOK(Init_and_Alloc())

  /*
     It's expected that BCJ/ARMT filter can process up to 4 bytes less
     than sent data size. For such BCJ/ARMT cases with non-filtered data we:
       - write some filtered data to output stream
       - move non-written data (filtered and non-filtered data) to start of buffer
       - read more new data from input stream to position after end of non-filtered data
       - call Filter() for concatenated data in buffer.

     For all cases, even for cases with partial filtering (BCJ/ARMT),
     we try to keep real/virtual alignment for all operations
       (memmove, Read(), Filter(), Write()).
     We use (kAlignSize=64) alignmnent that is larger than (16-bytes)
     required for AES filter alignment.

     AES-CBC uses 16-bytes blocks, that is simple case for processing here,
     if we call Filter() for aligned size for all calls except of last call (last block).
     And now there are no filters that use blocks with non-power2 size,
     but we try to support such non-power2 filters too here at Code().
  */
    
  UInt64 prev = 0;
  UInt64 nowPos64 = 0;
  bool inputFinished = false;
  UInt32 readPos = 0;
  UInt32 filterPos = 0;

  while (!outSize || nowPos64 < *outSize)
  {
    HRESULT hres = S_OK;
    if (!inputFinished)
    {
      size_t processedSize = _bufSize - readPos;
      /* for AES filters we need at least max(16, kAlignSize) bytes in buffer.
         But we try to read full buffer to reduce the number of Filter() and Write() calls.
      */
      hres = ReadStream(inStream, _buf + readPos, &processedSize);
      readPos += (UInt32)processedSize;
      inputFinished = (readPos != _bufSize);
      if (hres != S_OK)
      {
        // do we need to stop encoding after reading error?
        // if (_encodeMode) return hres;
        inputFinished = true;
      }
    }

    if (readPos == 0)
      return hres;

    /* we set (needMoreInput = true), if it's block-filter (like AES-CBC)
         that needs more data for current block filtering:
       We read full input buffer with Read(), and _bufSize is aligned,
       So the possible cases when we set (needMoreInput = true) are:
         1) decode : filter needs more data after the end of input stream.
           another cases are possible for non-power2-block-filter,
           because buffer size is not aligned for filter_non_power2_block_size:
         2) decode/encode : filter needs more data from non-finished input stream
         3) encode        : filter needs more space for zeros after the end of input stream
    */
    bool needMoreInput = false;

    while (readPos != filterPos)
    {
      /* Filter() is allowed to process part of data.
         Here we use the loop to filter as max as possible.
         when we call Filter(data, size):
         if (size < 16), AES-CTR filter uses internal 16-byte buffer.
         new (since v23.00) AES-CTR filter allows (size < 16) for non-last block,
         but it will work less efficiently than calls with aligned (size).
         We still support old (before v23.00) AES-CTR filters here.
         We have aligned (size) for AES-CTR, if it's not last block.
         We have aligned (readPos) for any filter, if (!inputFinished).
         We also meet the requirements for (data) pointer in Filter() call:
         {
           (virtual_stream_offset % aligment_size) == (data_ptr % aligment_size)
           (aligment_size == 2^N)
           (aligment_size  >= 16)
         }
      */
      const UInt32 cur = Filter->Filter(_buf + filterPos, readPos - filterPos);
      if (cur == 0)
        break;
      const UInt32 f = filterPos + cur;
      if (cur > readPos - filterPos)
      {
        // AES-CBC
        if (hres != S_OK)
          break;

        if (!_encodeMode
            || cur > _bufSize - filterPos
            || !inputFinished)
        {
          /* (cur > _bufSize - filterPos) is unexpected for AES filter, if _bufSize is multiply of 16.
             But we support this case, if some future filter will use block with non-power2-size.
          */
          needMoreInput = true;
          break;
        }

        /* (_encodeMode && inputFinished).
           We add zero bytes as pad in current block after the end of read data. */
        Byte *buf = _buf;
        do
          buf[readPos] = 0;
        while (++readPos != f);
        // (readPos) now is (size_of_real_input_data + size_of_zero_pad)
        if (cur != Filter->Filter(buf + filterPos, cur))
          return E_FAIL;
      }
      filterPos = f;
    }

    UInt32 size = filterPos;
    if (hres == S_OK)
    {
      /* If we need more Read() or Filter() calls, then we need to Write()
         some data and move unwritten data to get additional space in buffer.
         We try to keep alignment for data moves, Read(), Filter() and Write() calls.
      */
      const UInt32 kAlignSize = 1 << 6;
      const UInt32 alignedFiltered = filterPos & ~(kAlignSize - 1);
      if (inputFinished)
      {
        if (!needMoreInput)
          size = readPos; // for risc/bcj filters in last block we write data after filterPos.
        else if (_encodeMode)
          size = alignedFiltered; // for non-power2-block-encode-filter
      }
      else
        size = alignedFiltered;
    }

    {
      UInt32 writeSize = size;
      if (outSize)
      {
        const UInt64 rem = *outSize - nowPos64;
        if (writeSize > rem)
          writeSize = (UInt32)rem;
      }
      RINOK(WriteStream(outStream, _buf, writeSize))
      nowPos64 += writeSize;
    }

    if (hres != S_OK)
      return hres;

    if (inputFinished)
    {
      if (readPos == size)
        return hres;
      if (!_encodeMode)
      {
        // block-decode-filter (AES-CBS) has non-full last block
        // we don't want unaligned data move for more iterations with this error case.
        return S_FALSE;
      }
    }

    if (size == 0)
    {
      // it's unexpected that we have no any move in this iteration.
      return E_FAIL;
    }
    // if (size != 0)
    {
      if (filterPos < size)
        return E_FAIL; // filterPos = 0; else
      filterPos -= size;
      readPos -= size;
      if (readPos != 0)
        memmove(_buf, _buf + size, readPos);
    }
    // printf("\nnowPos64=%x, readPos=%x, filterPos=%x\n", (unsigned)nowPos64, (unsigned)readPos, (unsigned)filterPos);

    if (progress && (nowPos64 - prev) >= (1 << 22))
    {
      prev = nowPos64;
      RINOK(progress->SetRatioInfo(&nowPos64, &nowPos64))
    }
  }

  return S_OK;
}



// ---------- Write to Filter ----------

Z7_COM7F_IMF(CFilterCoder::SetOutStream(ISequentialOutStream *outStream))
{
  _outStream = outStream;
  return S_OK;
}

Z7_COM7F_IMF(CFilterCoder::ReleaseOutStream())
{
  _outStream.Release();
  return S_OK;
}

HRESULT CFilterCoder::Flush2()
{
  while (_convSize != 0)
  {
    UInt32 num = _convSize;
    if (_outSize_Defined)
    {
      const UInt64 rem = _outSize - _nowPos64;
      if (num > rem)
        num = (UInt32)rem;
      if (num == 0)
        return k_My_HRESULT_WritingWasCut;
    }
    
    UInt32 processed = 0;
    const HRESULT res = _outStream->Write(_buf + _convPos, num, &processed);
    if (processed == 0)
      return res != S_OK ? res : E_FAIL;
    
    _convPos += processed;
    _convSize -= processed;
    _nowPos64 += processed;
    RINOK(res)
  }
    
  const UInt32 convPos = _convPos;
  if (convPos != 0)
  {
    const UInt32 num = _bufPos - convPos;
    Byte *buf = _buf;
    for (UInt32 i = 0; i < num; i++)
      buf[i] = buf[convPos + i];
    _bufPos = num;
    _convPos = 0;
  }
    
  return S_OK;
}

Z7_COM7F_IMF(CFilterCoder::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  
  while (size != 0)
  {
    RINOK(Flush2())

    // _convSize is 0
    // _convPos is 0
    // _bufPos is small

    if (_bufPos != _bufSize)
    {
      UInt32 num = MyMin(size, _bufSize - _bufPos);
      memcpy(_buf + _bufPos, data, num);
      size -= num;
      data = (const Byte *)data + num;
      if (processedSize)
        *processedSize += num;
      _bufPos += num;
      if (_bufPos != _bufSize)
        continue;
    }

    // _bufPos == _bufSize
    _convSize = Filter->Filter(_buf, _bufPos);
    
    if (_convSize == 0)
      break;
    if (_convSize > _bufPos)
    {
      // that case is not possible.
      _convSize = 0;
      return E_FAIL;
    }
  }

  return S_OK;
}

Z7_COM7F_IMF(CFilterCoder::OutStreamFinish())
{
  for (;;)
  {
    RINOK(Flush2())
    if (_bufPos == 0)
      break;
    const UInt32 convSize = Filter->Filter(_buf, _bufPos);
    _convSize = convSize;
    UInt32 bufPos = _bufPos;
    if (convSize == 0)
      _convSize = bufPos;
    else if (convSize > bufPos)
    {
      // AES
      if (convSize > _bufSize)
      {
        _convSize = 0;
        return E_FAIL;
      }
      if (!_encodeMode)
      {
        _convSize = 0;
        return S_FALSE;
      }
      Byte *buf = _buf;
      for (; bufPos < convSize; bufPos++)
        buf[bufPos] = 0;
      _bufPos = bufPos;
      _convSize = Filter->Filter(_buf, bufPos);
      if (_convSize != _bufPos)
        return E_FAIL;
    }
  }
  
  CMyComPtr<IOutStreamFinish> finish;
  _outStream.QueryInterface(IID_IOutStreamFinish, &finish);
  if (finish)
    return finish->OutStreamFinish();
  return S_OK;
}

// ---------- Init functions ----------

Z7_COM7F_IMF(CFilterCoder::InitEncoder())
{
  InitSpecVars();
  return Init_and_Alloc();
}

HRESULT CFilterCoder::Init_NoSubFilterInit()
{
  InitSpecVars();
  return Alloc();
}

Z7_COM7F_IMF(CFilterCoder::SetOutStreamSize(const UInt64 *outSize))
{
  InitSpecVars();
  if (outSize)
  {
    _outSize = *outSize;
    _outSize_Defined = true;
  }
  return Init_and_Alloc();
}

// ---------- Read from Filter ----------

Z7_COM7F_IMF(CFilterCoder::SetInStream(ISequentialInStream *inStream))
{
  _inStream = inStream;
  return S_OK;
}

Z7_COM7F_IMF(CFilterCoder::ReleaseInStream())
{
  _inStream.Release();
  return S_OK;
}


Z7_COM7F_IMF(CFilterCoder::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  
  while (size != 0)
  {
    if (_convSize != 0)
    {
      if (size > _convSize)
        size = _convSize;
      if (_outSize_Defined)
      {
        const UInt64 rem = _outSize - _nowPos64;
        if (size > rem)
          size = (UInt32)rem;
      }
      memcpy(data, _buf + _convPos, size);
      _convPos += size;
      _convSize -= size;
      _nowPos64 += size;
      if (processedSize)
        *processedSize = size;
      break;
    }
  
    const UInt32 convPos = _convPos;
    if (convPos != 0)
    {
      const UInt32 num = _bufPos - convPos;
      Byte *buf = _buf;
      for (UInt32 i = 0; i < num; i++)
        buf[i] = buf[convPos + i];
      _bufPos = num;
      _convPos = 0;
    }
    
    {
      size_t readSize = _bufSize - _bufPos;
      const HRESULT res = ReadStream(_inStream, _buf + _bufPos, &readSize);
      _bufPos += (UInt32)readSize;
      RINOK(res)
    }
    
    const UInt32 convSize = Filter->Filter(_buf, _bufPos);
    _convSize = convSize;
    
    UInt32 bufPos = _bufPos;

    if (convSize == 0)
    {
      if (bufPos == 0)
        break;
      // BCJ
      _convSize = bufPos;
      continue;
    }
    
    if (convSize > bufPos)
    {
      // AES
      if (convSize > _bufSize)
        return E_FAIL;
      if (!_encodeMode)
        return S_FALSE;
      Byte *buf = _buf;
      do
        buf[bufPos] = 0;
      while (++bufPos != convSize);
      _bufPos = bufPos;
      _convSize = Filter->Filter(_buf, convSize);
      if (_convSize != _bufPos)
        return E_FAIL;
    }
  }
 
  return S_OK;
}


#ifndef Z7_NO_CRYPTO

Z7_COM7F_IMF(CFilterCoder::CryptoSetPassword(const Byte *data, UInt32 size))
  { return _setPassword->CryptoSetPassword(data, size); }

Z7_COM7F_IMF(CFilterCoder::SetKey(const Byte *data, UInt32 size))
  { return _cryptoProperties->SetKey(data, size); }

Z7_COM7F_IMF(CFilterCoder::SetInitVector(const Byte *data, UInt32 size))
  { return _cryptoProperties->SetInitVector(data, size); }

#endif


#ifndef Z7_EXTRACT_ONLY

Z7_COM7F_IMF(CFilterCoder::SetCoderProperties(const PROPID *propIDs,
    const PROPVARIANT *properties, UInt32 numProperties))
  { return _setCoderProperties->SetCoderProperties(propIDs, properties, numProperties); }

Z7_COM7F_IMF(CFilterCoder::WriteCoderProperties(ISequentialOutStream *outStream))
  { return _writeCoderProperties->WriteCoderProperties(outStream); }

Z7_COM7F_IMF(CFilterCoder::SetCoderPropertiesOpt(const PROPID *propIDs,
    const PROPVARIANT *properties, UInt32 numProperties))
  { return _setCoderPropertiesOpt->SetCoderPropertiesOpt(propIDs, properties, numProperties); }

/*
Z7_COM7F_IMF(CFilterCoder::ResetSalt()
  { return _cryptoResetSalt->ResetSalt(); }
*/

Z7_COM7F_IMF(CFilterCoder::ResetInitVector())
  { return _cryptoResetInitVector->ResetInitVector(); }

#endif


Z7_COM7F_IMF(CFilterCoder::SetDecoderProperties2(const Byte *data, UInt32 size))
  { return _setDecoderProperties2->SetDecoderProperties2(data, size); }
