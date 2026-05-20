// Bcj2Coder.cpp

#include "StdAfx.h"

// #include <stdio.h>

#include "../../../C/Alloc.h"

#include "../Common/StreamUtils.h"

#include "Bcj2Coder.h"

namespace NCompress {
namespace NBcj2 {

CBaseCoder::CBaseCoder()
{
  for (unsigned i = 0; i < BCJ2_NUM_STREAMS + 1; i++)
  {
    _bufs[i] = NULL;
    _bufsSizes[i] = 0;
    _bufsSizes_New[i] = (1 << 18);
  }
}

CBaseCoder::~CBaseCoder()
{
  for (unsigned i = 0; i < BCJ2_NUM_STREAMS + 1; i++)
    ::MidFree(_bufs[i]);
}

HRESULT CBaseCoder::Alloc(bool allocForOrig)
{
  const unsigned num = allocForOrig ? BCJ2_NUM_STREAMS + 1 : BCJ2_NUM_STREAMS;
  for (unsigned i = 0; i < num; i++)
  {
    UInt32 size = _bufsSizes_New[i];
    /* buffer sizes for BCJ2_STREAM_CALL and BCJ2_STREAM_JUMP streams
       must be aligned for 4 */
    size &= ~(UInt32)3;
    const UInt32 kMinBufSize = 4;
    if (size < kMinBufSize)
      size = kMinBufSize;
    // size = 4 * 100; // for debug
    // if (BCJ2_IS_32BIT_STREAM(i) == 1) size = 4 * 1; // for debug
    if (!_bufs[i] || size != _bufsSizes[i])
    {
      if (_bufs[i])
      {
        ::MidFree(_bufs[i]);
        _bufs[i] = NULL;
      }
      _bufsSizes[i] = 0;
      Byte *buf = (Byte *)::MidAlloc(size);
      if (!buf)
        return E_OUTOFMEMORY;
      _bufs[i] = buf;
      _bufsSizes[i] = size;
    }
  }
  return S_OK;
}



#ifndef Z7_EXTRACT_ONLY

CEncoder::CEncoder():
    _relatLim(BCJ2_ENC_RELAT_LIMIT_DEFAULT)
    // , _excludeRangeBits(BCJ2_RELAT_EXCLUDE_NUM_BITS)
    {}
CEncoder::~CEncoder() {}

Z7_COM7F_IMF(CEncoder::SetInBufSize(UInt32, UInt32 size))
  { _bufsSizes_New[BCJ2_NUM_STREAMS] = size; return S_OK; }
Z7_COM7F_IMF(CEncoder::SetOutBufSize(UInt32 streamIndex, UInt32 size))
  { _bufsSizes_New[streamIndex] = size; return S_OK; }

Z7_COM7F_IMF(CEncoder::SetCoderProperties(const PROPID *propIDs, const PROPVARIANT *props, UInt32 numProps))
{
  UInt32 relatLim = BCJ2_ENC_RELAT_LIMIT_DEFAULT;
  // UInt32 excludeRangeBits = BCJ2_RELAT_EXCLUDE_NUM_BITS;
  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = props[i];
    const PROPID propID = propIDs[i];
    if (propID >= NCoderPropID::kReduceSize
        // && propID != NCoderPropID::kHashBits
        )
      continue;
    switch (propID)
    {
      /*
      case NCoderPropID::kDefaultProp:
      {
        if (prop.vt != VT_UI4)
          return E_INVALIDARG;
        UInt32 v = prop.ulVal;
        if (v > 31)
          return E_INVALIDARG;
        relatLim = (UInt32)1 << v;
        break;
      }
      case NCoderPropID::kHashBits:
      {
        if (prop.vt != VT_UI4)
          return E_INVALIDARG;
        UInt32 v = prop.ulVal;
        if (v > 31)
          return E_INVALIDARG;
        excludeRangeBits = v;
        break;
      }
      */
      case NCoderPropID::kDictionarySize:
      {
        if (prop.vt != VT_UI4)
          return E_INVALIDARG;
        relatLim = prop.ulVal;
        if (relatLim > BCJ2_ENC_RELAT_LIMIT_MAX)
          return E_INVALIDARG;
        break;
      }
      case NCoderPropID::kNumThreads:
      case NCoderPropID::kLevel:
        continue;
      default: return E_INVALIDARG;
    }
  }
  _relatLim = relatLim;
  // _excludeRangeBits = excludeRangeBits;
  return S_OK;
}


HRESULT CEncoder::CodeReal(
    ISequentialInStream * const *inStreams, const UInt64 * const *inSizes, UInt32 numInStreams,
    ISequentialOutStream * const *outStreams, const UInt64 * const * /* outSizes */, UInt32 numOutStreams,
    ICompressProgressInfo *progress)
{
  if (numInStreams != 1 || numOutStreams != BCJ2_NUM_STREAMS)
    return E_INVALIDARG;

  RINOK(Alloc())

  CBcj2Enc_ip_unsigned fileSize_minus1 = BCJ2_ENC_FileSizeField_UNLIMITED;
  if (inSizes && inSizes[0])
  {
    const UInt64 inSize = *inSizes[0];
   #ifdef BCJ2_ENC_FileSize_MAX
    if (inSize <= BCJ2_ENC_FileSize_MAX)
   #endif
      fileSize_minus1 = BCJ2_ENC_GET_FileSizeField_VAL_FROM_FileSize(inSize);
  }

  Z7_DECL_CMyComPtr_QI_FROM(ICompressGetSubStreamSize, getSubStreamSize, inStreams[0])

  CBcj2Enc enc;
  enc.src = _bufs[BCJ2_NUM_STREAMS];
  enc.srcLim = enc.src;
  {
    for (unsigned i = 0; i < BCJ2_NUM_STREAMS; i++)
    {
      enc.bufs[i] = _bufs[i];
      enc.lims[i] = _bufs[i] + _bufsSizes[i];
    }
  }
  Bcj2Enc_Init(&enc);
  enc.fileIp64 = 0;
  enc.fileSize64_minus1 = fileSize_minus1;
  enc.relatLimit = _relatLim;
  // enc.relatExcludeBits = _excludeRangeBits;
  enc.finishMode = BCJ2_ENC_FINISH_MODE_CONTINUE;

  // Varibales that correspond processed data in input stream:
  UInt64 inPos_without_Temp = 0;  // it doesn't include data in enc.temp[]
  UInt64 inPos_with_Temp = 0;     // it        includes data in enc.temp[]

  UInt64 prevProgress = 0;
  UInt64 totalRead = 0;  // size read from input stream
  UInt64 outSizeRc = 0;
  UInt64 subStream_Index = 0;
  UInt64 subStream_StartPos = 0; // global start offset of subStreams[subStream_Index]
  UInt64 subStream_Size = 0;
  const Byte *srcLim_Read = _bufs[BCJ2_NUM_STREAMS];
  bool readWasFinished = false;
  bool isAccurate = false;
  bool wasUnknownSize = false;

  for (;;)
  {
    if (readWasFinished && enc.srcLim == srcLim_Read)
      enc.finishMode = BCJ2_ENC_FINISH_MODE_END_STREAM;

    // for debug:
    // for (int y=0;y<100;y++) { CBcj2Enc enc2 = enc; Bcj2Enc_Encode(&enc2); }
    
    Bcj2Enc_Encode(&enc);

    inPos_with_Temp = totalRead - (size_t)(srcLim_Read - enc.src);
    inPos_without_Temp = inPos_with_Temp - Bcj2Enc_Get_AvailInputSize_in_Temp(&enc);
    
    // if (inPos_without_Temp != enc.ip64) return E_FAIL;

    if (Bcj2Enc_IsFinished(&enc))
      break;

    if (enc.state < BCJ2_NUM_STREAMS)
    {
      if (enc.bufs[enc.state] != enc.lims[enc.state])
        return E_FAIL;
      const size_t curSize = (size_t)(enc.bufs[enc.state] - _bufs[enc.state]);
      // printf("Write stream = %2d %6d\n", enc.state, curSize);
      RINOK(WriteStream(outStreams[enc.state], _bufs[enc.state], curSize))
      if (enc.state == BCJ2_STREAM_RC)
        outSizeRc += curSize;
      enc.bufs[enc.state] = _bufs[enc.state];
      enc.lims[enc.state] = _bufs[enc.state] + _bufsSizes[enc.state];
    }
    else
    {
      if (enc.state != BCJ2_ENC_STATE_ORIG)
        return E_FAIL;
      // (enc.state == BCJ2_ENC_STATE_ORIG)
      if (enc.src != enc.srcLim)
        return E_FAIL;
      if (enc.finishMode != BCJ2_ENC_FINISH_MODE_CONTINUE
          && Bcj2Enc_Get_AvailInputSize_in_Temp(&enc) != 0)
        return E_FAIL;

      if (enc.src == srcLim_Read)
      {
        if (readWasFinished)
          return E_FAIL;
        UInt32 curSize = _bufsSizes[BCJ2_NUM_STREAMS];
        RINOK(inStreams[0]->Read(_bufs[BCJ2_NUM_STREAMS], curSize, &curSize))
        // printf("Read %6u bytes\n", curSize);
        if (curSize == 0)
          readWasFinished = true;
        totalRead += curSize;
        enc.src     = _bufs[BCJ2_NUM_STREAMS];
        srcLim_Read = _bufs[BCJ2_NUM_STREAMS] + curSize;
      }
      enc.srcLim = srcLim_Read;

      if (getSubStreamSize)
      {
        /* we set base default conversions options that will be used,
           if subStream related options will be not OK */
        enc.fileIp64 = 0;
        enc.fileSize64_minus1 = fileSize_minus1;
        for (;;)
        {
          UInt64 nextPos;
          if (isAccurate)
            nextPos = subStream_StartPos + subStream_Size;
          else
          {
            const HRESULT hres = getSubStreamSize->GetSubStreamSize(subStream_Index, &subStream_Size);
            if (hres != S_OK)
            {
              enc.finishMode = BCJ2_ENC_FINISH_MODE_CONTINUE;
              /* if sub-stream size is unknown, we use default settings.
                 We still can recover to normal mode for next sub-stream,
                 if GetSubStreamSize() will return S_OK, when current
                 sub-stream will be finished.
              */
              if (hres == S_FALSE)
              {
                wasUnknownSize = true;
                break;
              }
              if (hres == E_NOTIMPL)
              {
                getSubStreamSize.Release();
                break;
              }
              return hres;
            }
            // printf("GetSubStreamSize %6u : %6u \n", (unsigned)subStream_Index, (unsigned)subStream_Size);
            nextPos = subStream_StartPos + subStream_Size;
            if ((Int64)subStream_Size == -1)
            {
              /* it's not expected, but (-1) can mean unknown size. */
              enc.finishMode = BCJ2_ENC_FINISH_MODE_CONTINUE;
              wasUnknownSize = true;
              break;
            }
            if (nextPos < subStream_StartPos)
              return E_FAIL;
            isAccurate =
                 (nextPos <  totalRead
              || (nextPos <= totalRead && readWasFinished));
          }
          
          /* (nextPos) is estimated end position of current sub_stream.
             But only (totalRead) and (readWasFinished) values
             can confirm that this estimated end position is accurate.
             That end position is accurate, if it can't be changed in
             further calls of GetSubStreamSize() */

          /* (nextPos < inPos_with_Temp) is unexpected case here, that we
               can get if from some incorrect ICompressGetSubStreamSize object,
               where new GetSubStreamSize() call returns smaller size than
               confirmed by Read() size from previous GetSubStreamSize() call.
          */
          if (nextPos < inPos_with_Temp)
          {
            if (wasUnknownSize)
            {
              /* that case can be complicated for recovering.
                 so we disable sub-streams requesting. */
              enc.finishMode = BCJ2_ENC_FINISH_MODE_CONTINUE;
              getSubStreamSize.Release();
              break;
            }
            return E_FAIL; // to stop after failure
          }

          if (nextPos <= inPos_with_Temp)
          {
            // (nextPos == inPos_with_Temp)
            /* CBcj2Enc encoder requires to finish each [non-empty] block (sub-stream)
                  with BCJ2_ENC_FINISH_MODE_END_BLOCK
               or with BCJ2_ENC_FINISH_MODE_END_STREAM for last block:
               And we send data of new block to CBcj2Enc, only if previous block was finished.
               So we switch to next sub-stream if after Bcj2Enc_Encode() call we have
                 && (enc.finishMode != BCJ2_ENC_FINISH_MODE_CONTINUE)
                 && (nextPos == inPos_with_Temp)
                 && (enc.state == BCJ2_ENC_STATE_ORIG)
            */
            if (enc.finishMode != BCJ2_ENC_FINISH_MODE_CONTINUE)
            {
              /* subStream_StartPos is increased only here.
                   (subStream_StartPos == inPos_with_Temp) : at start
                   (subStream_StartPos <= inPos_with_Temp) : will be later
              */
              subStream_StartPos = nextPos;
              subStream_Size = 0;
              wasUnknownSize = false;
              subStream_Index++;
              isAccurate = false;
              // we don't change finishMode here
              continue;
            }
          }
          
          enc.finishMode = BCJ2_ENC_FINISH_MODE_CONTINUE;
          /* for (!isAccurate) case:
             (totalRead <= real_end_of_subStream)
             so we can use BCJ2_ENC_FINISH_MODE_CONTINUE up to (totalRead)
             // we don't change settings at the end of substream, if settings were unknown,
          */
         
          /* if (wasUnknownSize) then we can't trust size of that sub-stream.
             so we use default settings instead */
          if (!wasUnknownSize)
         #ifdef BCJ2_ENC_FileSize_MAX
          if (subStream_Size <= BCJ2_ENC_FileSize_MAX)
         #endif
          {
            enc.fileIp64 =
                (CBcj2Enc_ip_unsigned)(
                (CBcj2Enc_ip_signed)enc.ip64 +
                (CBcj2Enc_ip_signed)(subStream_StartPos - inPos_without_Temp));
            Bcj2Enc_SET_FileSize(&enc, subStream_Size)
          }

          if (isAccurate)
          {
            /* (real_end_of_subStream == nextPos <= totalRead)
               So we can use BCJ2_ENC_FINISH_MODE_END_BLOCK up to (nextPos). */
            const size_t rem = (size_t)(totalRead - nextPos);
            if ((size_t)(enc.srcLim - enc.src) < rem)
              return E_FAIL;
            enc.srcLim -= rem;
            enc.finishMode = BCJ2_ENC_FINISH_MODE_END_BLOCK;
          }

          break;
        } // for() loop
      } // getSubStreamSize
    }

    if (progress && inPos_without_Temp - prevProgress >= (1 << 22))
    {
      prevProgress = inPos_without_Temp;
      const UInt64 outSize2 = inPos_without_Temp + outSizeRc +
          (size_t)(enc.bufs[BCJ2_STREAM_RC] - _bufs[BCJ2_STREAM_RC]);
      // printf("progress %8u, %8u\n", (unsigned)inSize2, (unsigned)outSize2);
      RINOK(progress->SetRatioInfo(&inPos_without_Temp, &outSize2))
    }
  }

  for (unsigned i = 0; i < BCJ2_NUM_STREAMS; i++)
  {
    RINOK(WriteStream(outStreams[i], _bufs[i], (size_t)(enc.bufs[i] - _bufs[i])))
  }
  // if (inPos_without_Temp != subStream_StartPos + subStream_Size) return E_FAIL;
  return S_OK;
}


Z7_COM7F_IMF(CEncoder::Code(
    ISequentialInStream * const *inStreams, const UInt64 * const *inSizes, UInt32 numInStreams,
    ISequentialOutStream * const *outStreams, const UInt64 * const *outSizes, UInt32 numOutStreams,
    ICompressProgressInfo *progress))
{
  try
  {
    return CodeReal(inStreams, inSizes, numInStreams, outStreams, outSizes,numOutStreams, progress);
  }
  catch(...) { return E_FAIL; }
}

#endif






CDecoder::CDecoder():
    _finishMode(false)
#ifndef Z7_NO_READ_FROM_CODER
    , _outSizeDefined(false)
    , _outSize(0)
    , _outSize_Processed(0)
#endif
{}

Z7_COM7F_IMF(CDecoder::SetInBufSize(UInt32 streamIndex, UInt32 size))
  { _bufsSizes_New[streamIndex] = size; return S_OK; }
Z7_COM7F_IMF(CDecoder::SetOutBufSize(UInt32, UInt32 size))
  { _bufsSizes_New[BCJ2_NUM_STREAMS] = size; return S_OK; }

Z7_COM7F_IMF(CDecoder::SetFinishMode(UInt32 finishMode))
{
  _finishMode = (finishMode != 0);
  return S_OK;
}

void CBaseDecoder::InitCommon()
{
  for (unsigned i = 0; i < BCJ2_NUM_STREAMS; i++)
  {
    dec.lims[i] = dec.bufs[i] = _bufs[i];
    _readRes[i] = S_OK;
    _extraSizes[i] = 0;
    _readSizes[i] = 0;
  }
  Bcj2Dec_Init(&dec);
}


/* call ReadInStream() only after Bcj2Dec_Decode().
   input requirement:
      (dec.state < BCJ2_NUM_STREAMS)
*/
void CBaseDecoder::ReadInStream(ISequentialInStream *inStream)
{
  const unsigned state = dec.state;
  UInt32 total;
  {
    Byte *buf = _bufs[state];
    const Byte *cur = dec.bufs[state];
    // if (cur != dec.lims[state]) throw 1; // unexpected case
    dec.lims[state] =
    dec.bufs[state] = buf;
    total = (UInt32)_extraSizes[state];
    for (UInt32 i = 0; i < total; i++)
      buf[i] = cur[i];
  }
  
  if (_readRes[state] != S_OK)
    return;
  
  do
  {
    UInt32 curSize = _bufsSizes[state] - total;
    // if (state == 0) curSize = 0; // for debug
    // curSize = 7; // for debug
    /* even if we have reached provided inSizes[state] limit,
       we call Read() with (curSize != 0), because
       we want the called handler of stream->Read() could
       execute required Init/Flushing code even for empty stream.
       In another way we could call Read() with (curSize == 0) for
       finished streams, but some Read() handlers can ignore Read(size=0) calls.
    */
    const HRESULT hres = inStream->Read(_bufs[state] + total, curSize, &curSize);
    _readRes[state] = hres;
    if (curSize == 0)
      break;
    _readSizes[state] += curSize;
    total += curSize;
    if (hres != S_OK)
      break;
  }
  while (total < 4 && BCJ2_IS_32BIT_STREAM(state));
  
  /* we exit from decoding loop here, if we can't
     provide new data for input stream.
     Usually it's normal exit after full stream decoding. */
  if (total == 0)
    return;
  
  if (BCJ2_IS_32BIT_STREAM(state))
  {
    const unsigned extra = (unsigned)total & 3;
    _extraSizes[state] = extra;
    if (total < 4)
    {
      if (_readRes[state] == S_OK)
        _readRes[state] = S_FALSE; // actually it's stream error. So maybe we need another error code.
      return;
    }
    total -= (UInt32)extra;
  }
  
  dec.lims[state] += total; // = _bufs[state] + total;
}


Z7_COM7F_IMF(CDecoder::Code(
    ISequentialInStream * const *inStreams, const UInt64 * const *inSizes, UInt32 numInStreams,
    ISequentialOutStream * const *outStreams, const UInt64 * const *outSizes, UInt32 numOutStreams,
    ICompressProgressInfo *progress))
{
  if (numInStreams != BCJ2_NUM_STREAMS || numOutStreams != 1)
    return E_INVALIDARG;

  RINOK(Alloc())
  InitCommon();

  dec.destLim = dec.dest = _bufs[BCJ2_NUM_STREAMS];
  
  UInt64 outSizeWritten = 0;
  UInt64 prevProgress = 0;

  HRESULT hres_Crit = S_OK;  // critical hres status (mostly from input stream reading)
  HRESULT hres_Weak = S_OK;  // first non-critical error code from input stream reading

  for (;;)
  {
    if (Bcj2Dec_Decode(&dec) != SZ_OK)
    {
      /* it's possible only at start (first 5 bytes in RC stream) */
      hres_Crit = S_FALSE;
      break;
    }
    if (dec.state < BCJ2_NUM_STREAMS)
    {
      ReadInStream(inStreams[dec.state]);
      const unsigned state = dec.state;
      const HRESULT hres = _readRes[state];
      if (dec.lims[state] == _bufs[state])
      {
        // we break decoding, if there are no new data in input stream
        hres_Crit = hres;
        break;
      }
      if (hres != S_OK && hres_Weak == S_OK)
        hres_Weak = hres;
    }
    else  // (BCJ2_DEC_STATE_ORIG_0 <= state <= BCJ2_STATE_ORIG)
    {
      {
        const size_t curSize = (size_t)(dec.dest - _bufs[BCJ2_NUM_STREAMS]);
        if (curSize != 0)
        {
          outSizeWritten += curSize;
          RINOK(WriteStream(outStreams[0], _bufs[BCJ2_NUM_STREAMS], curSize))
        }
      }
      {
        UInt32 rem = _bufsSizes[BCJ2_NUM_STREAMS];
        if (outSizes && outSizes[0])
        {
          const UInt64 outSize = *outSizes[0] - outSizeWritten;
          if (rem > outSize)
            rem = (UInt32)outSize;
        }
        dec.dest = _bufs[BCJ2_NUM_STREAMS];
        dec.destLim = dec.dest + rem;
        /* we exit from decoding loop here,
           if (outSizes[0]) limit for output stream was reached */
        if (rem == 0)
          break;
      }
    }

    if (progress)
    {
      // here we don't count additional data in dec.temp (up to 4 bytes for output stream)
      const UInt64 processed = outSizeWritten + (size_t)(dec.dest - _bufs[BCJ2_NUM_STREAMS]);
      if (processed - prevProgress >= (1 << 24))
      {
        prevProgress = processed;
        const UInt64 inSize = processed +
            _readSizes[BCJ2_STREAM_RC] - (size_t)(
              dec.lims[BCJ2_STREAM_RC] -
              dec.bufs[BCJ2_STREAM_RC]);
        RINOK(progress->SetRatioInfo(&inSize, &prevProgress))
      }
    }
  }

  {
    const size_t curSize = (size_t)(dec.dest - _bufs[BCJ2_NUM_STREAMS]);
    if (curSize != 0)
    {
      outSizeWritten += curSize;
      RINOK(WriteStream(outStreams[0], _bufs[BCJ2_NUM_STREAMS], curSize))
    }
  }

  if (hres_Crit == S_OK) hres_Crit = hres_Weak;
  if (hres_Crit != S_OK) return hres_Crit;

  if (_finishMode)
  {
    if (!Bcj2Dec_IsMaybeFinished_code(&dec))
      return S_FALSE;

    /* here we support two correct ways to finish full stream decoding
       with one of the following conditions:
          - the end of input  stream MAIN was reached
          - the end of output stream ORIG was reached
       Currently 7-Zip/7z code ends with (state == BCJ2_STREAM_MAIN),
       because the sizes of MAIN and ORIG streams are known and these
       sizes are stored in 7z archive headers.
       And Bcj2Dec_Decode() exits with (state == BCJ2_STREAM_MAIN),
       if both MAIN and ORIG streams have reached buffers limits.
       But if the size of MAIN stream is not known or if the
       size of MAIN stream includes some padding after payload data,
       then we still can correctly finish decoding with
       (state == BCJ2_DEC_STATE_ORIG), if we know the exact size
       of output ORIG stream.
    */
    if (dec.state != BCJ2_STREAM_MAIN)
    if (dec.state != BCJ2_DEC_STATE_ORIG)
      return S_FALSE;

    /* the caller also will know written size.
       So the following check is optional: */
    if (outSizes && outSizes[0] && *outSizes[0] != outSizeWritten)
      return S_FALSE;

    if (inSizes)
    {
      for (unsigned i = 0; i < BCJ2_NUM_STREAMS; i++)
      {
        /* if (inSizes[i]) is defined, we do full check for processed stream size. */
        if (inSizes[i] && *inSizes[i] != GetProcessedSize_ForInStream(i))
          return S_FALSE;
      }
    }

    /* v23.02: we call Read(0) for BCJ2_STREAM_CALL and BCJ2_STREAM_JUMP streams,
       if there were no Read() calls for such stream.
       So the handlers of these input streams objects can do
       Init/Flushing even for case when stream is empty:
    */
    for (unsigned i = BCJ2_STREAM_CALL; i < BCJ2_STREAM_CALL + 2; i++)
    {
      if (_readSizes[i])
        continue;
      Byte b;
      UInt32 processed;
      RINOK(inStreams[i]->Read(&b, 0, &processed))
    }
  }

  return S_OK;
}


Z7_COM7F_IMF(CDecoder::GetInStreamProcessedSize2(UInt32 streamIndex, UInt64 *value))
{
  *value = GetProcessedSize_ForInStream(streamIndex);
  return S_OK;
}


#ifndef Z7_NO_READ_FROM_CODER

Z7_COM7F_IMF(CDecoder::SetInStream2(UInt32 streamIndex, ISequentialInStream *inStream))
{
  _inStreams[streamIndex] = inStream;
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::ReleaseInStream2(UInt32 streamIndex))
{
  _inStreams[streamIndex].Release();
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::SetOutStreamSize(const UInt64 *outSize))
{
  _outSizeDefined = (outSize != NULL);
  _outSize = 0;
  if (_outSizeDefined)
    _outSize = *outSize;
  _outSize_Processed = 0;

  const HRESULT res = Alloc(false); // allocForOrig
  InitCommon();
  dec.destLim = dec.dest = NULL;
  return res;
}


Z7_COM7F_IMF(CDecoder::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;

  /* Note the case:
     The output (ORIG) stream can be empty.
     But BCJ2_STREAM_RC stream always is not empty.
     And we want to support full data processing for all streams.
     We disable check (size == 0) here.
     So if the caller calls this CDecoder::Read() with (size == 0),
     we execute required Init/Flushing code in this CDecoder object.
     Also this CDecoder::Read() function will call Read() for input streams.
     So the handlers of input streams objects also can do Init/Flushing.
  */
  // if (size == 0) return S_OK;  // disabled to allow (size == 0) processing

  UInt32 totalProcessed = 0;
 
  if (_outSizeDefined)
  {
    const UInt64 rem = _outSize - _outSize_Processed;
    if (size > rem)
      size = (UInt32)rem;
  }
  dec.dest = (Byte *)data;
  dec.destLim = (const Byte *)data + size;

  HRESULT res = S_OK;

  for (;;)
  {
    if (Bcj2Dec_Decode(&dec) != SZ_OK)
      return S_FALSE;  // this error can be only at start of stream
    {
      const UInt32 curSize = (UInt32)(size_t)(dec.dest - (Byte *)data);
      if (curSize != 0)
      {
        data = (void *)((Byte *)data + curSize);
        size -= curSize;
        _outSize_Processed += curSize;
        totalProcessed += curSize;
        if (processedSize)
          *processedSize = totalProcessed;
      }
    }
    if (dec.state >= BCJ2_NUM_STREAMS)
      break;
    ReadInStream(_inStreams[dec.state]);
    if (dec.lims[dec.state] == _bufs[dec.state])
    {
      /* we break decoding, if there are no new data in input stream.
         and we ignore error code, if some data were written to output buffer. */
      if (totalProcessed == 0)
        res = _readRes[dec.state];
      break;
    }
  }

  if (res == S_OK)
  if (_finishMode && _outSizeDefined && _outSize == _outSize_Processed)
  {
    if (!Bcj2Dec_IsMaybeFinished_code(&dec))
      return S_FALSE;
    if (dec.state != BCJ2_STREAM_MAIN)
    if (dec.state != BCJ2_DEC_STATE_ORIG)
      return S_FALSE;
  }

  return res;
}

#endif

}}


/*
extern "C"
{
extern UInt32 bcj2_stats[256 + 2][2];
}

static class CBcj2Stat
{
public:
  ~CBcj2Stat()
  {
    printf("\nBCJ2 stat:");
    unsigned sums[2] = { 0, 0 };
    int i;
    for (i = 2; i < 256 + 2; i++)
    {
      sums[0] += bcj2_stats[i][0];
      sums[1] += bcj2_stats[i][1];
    }
    const unsigned sums2 = sums[0] + sums[1];
    for (int vi = 0; vi < 256 + 3; vi++)
    {
      printf("\n");
      UInt32 n0, n1;
      if (vi < 4)
        printf("\n");
      
      if (vi < 2)
        i = vi;
      else if (vi == 2)
        i = -1;
      else
        i = vi - 1;
  
      if (i < 0)
      {
        n0 = sums[0];
        n1 = sums[1];
        printf("calls   :");
      }
      else
      {
        if (i == 0)
          printf("jcc     :");
        else if (i == 1)
          printf("jump    :");
        else
          printf("call %02x :", i - 2);
        n0 = bcj2_stats[i][0];
        n1 = bcj2_stats[i][1];
      }
      
      const UInt32 sum = n0 + n1;
      printf(" %10u", sum);

    #define PRINT_PERC(val, sum) \
        { UInt32 _sum  = sum; if (_sum == 0) _sum = 1; \
        printf(" %7.3f %%", (double)((double)val * (double)100 / (double)_sum )); }

      if (i >= 2 || i < 0)
      {
        PRINT_PERC(sum, sums2);
      }
      else
        printf("%10s", "");

      printf(" :%10u", n0);
      PRINT_PERC(n0, sum);

      printf(" :%10u", n1);
      PRINT_PERC(n1, sum);
    }
    printf("\n\n");
    fflush(stdout);
  }
} g_CBcjStat;
*/
