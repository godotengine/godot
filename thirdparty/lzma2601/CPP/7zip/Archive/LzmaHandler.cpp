// LzmaHandler.cpp

#include "StdAfx.h"

#include "../../../C/CpuArch.h"

#include "../../Common/ComTry.h"
#include "../../Common/IntToString.h"

#include "../../Windows/PropVariant.h"

#include "../Common/FilterCoder.h"
#include "../Common/ProgressUtils.h"
#include "../Common/RegisterArc.h"
#include "../Common/StreamUtils.h"

#include "../Compress/BcjCoder.h"
#include "../Compress/LzmaDecoder.h"

#include "Common/DummyOutStream.h"

using namespace NWindows;

namespace NArchive {
namespace NLzma {

static bool CheckDicSize(const Byte *p)
{
  UInt32 dicSize = GetUi32(p);
  if (dicSize == 1)
    return true;
  for (unsigned i = 0; i <= 30; i++)
    if (dicSize == ((UInt32)2 << i) || dicSize == ((UInt32)3 << i))
      return true;
  return (dicSize == 0xFFFFFFFF);
}

static const Byte kProps[] =
{
  kpidSize,
  kpidPackSize,
  kpidMethod
};

static const Byte kArcProps[] =
{
  kpidNumStreams,
  kpidMethod
};

struct CHeader
{
  UInt64 Size;
  Byte FilterID;
  Byte LzmaProps[5];

  Byte GetProp() const { return LzmaProps[0]; }
  UInt32 GetDicSize() const { return GetUi32(LzmaProps + 1); }
  bool HasSize() const { return (Size != (UInt64)(Int64)-1); }
  bool Parse(const Byte *buf, bool isThereFilter);
};

bool CHeader::Parse(const Byte *buf, bool isThereFilter)
{
  FilterID = 0;
  if (isThereFilter)
    FilterID = buf[0];
  const Byte *sig = buf + (isThereFilter ? 1 : 0);
  for (int i = 0; i < 5; i++)
    LzmaProps[i] = sig[i];
  Size = GetUi64(sig + 5);
  return
    LzmaProps[0] < 5 * 5 * 9 &&
    FilterID < 2 &&
    (!HasSize() || Size < ((UInt64)1 << 56))
    && CheckDicSize(LzmaProps + 1);
}

class CDecoder Z7_final
{
  CMyComPtr<ISequentialOutStream> _bcjStream;
  CFilterCoder *_filterCoder;
public:
  CMyComPtr2<ICompressCoder, NCompress::NLzma::CDecoder> _lzmaDecoder;

  ~CDecoder();
  HRESULT Create(bool filtered, ISequentialInStream *inStream);

  HRESULT Code(const CHeader &header, ISequentialOutStream *outStream, ICompressProgressInfo *progress);

  UInt64 GetInputProcessedSize() const { return _lzmaDecoder->GetInputProcessedSize(); }

  void ReleaseInStream() { if (_lzmaDecoder) _lzmaDecoder->ReleaseInStream(); }

  HRESULT ReadInput(Byte *data, UInt32 size, UInt32 *processedSize)
    { return _lzmaDecoder->ReadFromInputStream(data, size, processedSize); }
};

HRESULT CDecoder::Create(bool filteredMode, ISequentialInStream *inStream)
{
  _lzmaDecoder.Create_if_Empty();
  _lzmaDecoder->FinishStream = true;

  if (filteredMode)
  {
    if (!_bcjStream)
    {
      _filterCoder = new CFilterCoder(false);
      CMyComPtr<ICompressCoder> coder = _filterCoder;
      _filterCoder->Filter = new NCompress::NBcj::CCoder2(z7_BranchConvSt_X86_Dec);
      _bcjStream = _filterCoder;
    }
  }

  return _lzmaDecoder->SetInStream(inStream);
}

CDecoder::~CDecoder()
{
  ReleaseInStream();
}

HRESULT CDecoder::Code(const CHeader &header, ISequentialOutStream *outStream,
    ICompressProgressInfo *progress)
{
  if (header.FilterID > 1)
    return E_NOTIMPL;

  RINOK(_lzmaDecoder->SetDecoderProperties2(header.LzmaProps, 5))

  bool filteredMode = (header.FilterID == 1);

  if (filteredMode)
  {
    RINOK(_filterCoder->SetOutStream(outStream))
    outStream = _bcjStream;
    RINOK(_filterCoder->SetOutStreamSize(NULL))
  }

  const UInt64 *Size = header.HasSize() ? &header.Size : NULL;
  HRESULT res = _lzmaDecoder->CodeResume(outStream, Size, progress);

  if (filteredMode)
  {
    {
      HRESULT res2 = _filterCoder->OutStreamFinish();
      if (res == S_OK)
        res = res2;
    }
    HRESULT res2 = _filterCoder->ReleaseOutStream();
    if (res == S_OK)
      res = res2;
  }
  
  RINOK(res)

  if (header.HasSize())
    if (_lzmaDecoder->GetOutputProcessedSize() != header.Size)
      return S_FALSE;

  return S_OK;
}


Z7_CLASS_IMP_CHandler_IInArchive_1(
  IArchiveOpenSeq
)
  bool _lzma86;
  bool _isArc;
  bool _needSeekToStart;
  bool _dataAfterEnd;
  bool _needMoreInput;
  bool _unsupported;
  bool _dataError;

  bool _packSize_Defined;
  bool _unpackSize_Defined;
  bool _numStreams_Defined;

  CHeader _header;
  CMyComPtr<IInStream> _stream;
  CMyComPtr<ISequentialInStream> _seqStream;
  
  UInt64 _packSize;
  UInt64 _unpackSize;
  UInt64 _numStreams;

  void GetMethod(NCOM::CPropVariant &prop);

  unsigned GetHeaderSize() const { return 5 + 8 + (_lzma86 ? 1 : 0); }
public:
  CHandler(bool lzma86) { _lzma86 = lzma86; }
};

IMP_IInArchive_Props
IMP_IInArchive_ArcProps

Z7_COM7F_IMF(CHandler::GetArchiveProperty(PROPID propID, PROPVARIANT *value))
{
  NCOM::CPropVariant prop;
  switch (propID)
  {
    case kpidPhySize: if (_packSize_Defined) prop = _packSize; break;
    case kpidNumStreams: if (_numStreams_Defined) prop = _numStreams; break;
    case kpidUnpackSize: if (_unpackSize_Defined) prop = _unpackSize; break;
    case kpidMethod: GetMethod(prop); break;
    case kpidErrorFlags:
    {
      UInt32 v = 0;
      if (!_isArc) v |= kpv_ErrorFlags_IsNotArc;
      if (_needMoreInput) v |= kpv_ErrorFlags_UnexpectedEnd;
      if (_dataAfterEnd) v |= kpv_ErrorFlags_DataAfterEnd;
      if (_unsupported) v |= kpv_ErrorFlags_UnsupportedMethod;
      if (_dataError) v |= kpv_ErrorFlags_DataError;
      prop = v;
      break;
    }
    default: break;
  }
  prop.Detach(value);
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetNumberOfItems(UInt32 *numItems))
{
  *numItems = 1;
  return S_OK;
}


static char * DictSizeToString(UInt32 val, char *s)
{
  for (unsigned i = 0; i < 32; i++)
    if (((UInt32)1 << i) == val)
      return ::ConvertUInt32ToString(i, s);
  char c = 'b';
       if ((val & ((1 << 20) - 1)) == 0) { val >>= 20; c = 'm'; }
  else if ((val & ((1 << 10) - 1)) == 0) { val >>= 10; c = 'k'; }
  s = ::ConvertUInt32ToString(val, s);
  *s++ = c;
  *s = 0;
  return s;
}

static char *AddProp32(char *s, const char *name, UInt32 v)
{
  *s++ = ':';
  s = MyStpCpy(s, name);
  return ::ConvertUInt32ToString(v, s);
}

void CHandler::GetMethod(NCOM::CPropVariant &prop)
{
  if (!_stream)
    return;

  char sz[64];
  char *s = sz;
  if (_header.FilterID != 0)
    s = MyStpCpy(s, "BCJ ");
  s = MyStpCpy(s, "LZMA:");
  s = DictSizeToString(_header.GetDicSize(), s);
  
  UInt32 d = _header.GetProp();
  // if (d != 0x5D)
  {
    UInt32 lc = d % 9;
    d /= 9;
    UInt32 pb = d / 5;
    UInt32 lp = d % 5;
    if (lc != 3) s = AddProp32(s, "lc", lc);
    if (lp != 0) s = AddProp32(s, "lp", lp);
    if (pb != 2) s = AddProp32(s, "pb", pb);
  }
  prop = sz;
}


Z7_COM7F_IMF(CHandler::GetProperty(UInt32 /* index */, PROPID propID, PROPVARIANT *value))
{
  NCOM::CPropVariant prop;
  switch (propID)
  {
    case kpidSize: if (_stream && _header.HasSize()) prop = _header.Size; break;
    case kpidPackSize: if (_packSize_Defined) prop = _packSize; break;
    case kpidMethod: GetMethod(prop); break;
    default: break;
  }
  prop.Detach(value);
  return S_OK;
}

API_FUNC_static_IsArc IsArc_Lzma(const Byte *p, size_t size)
{
  const UInt32 kHeaderSize = 1 + 4 + 8;
  if (size < kHeaderSize)
    return k_IsArc_Res_NEED_MORE;
  if (p[0] >= 5 * 5 * 9)
    return k_IsArc_Res_NO;
  const UInt64 unpackSize = GetUi64(p + 1 + 4);
  if (unpackSize != (UInt64)(Int64)-1)
  {
    if (unpackSize >= ((UInt64)1 << 56))
      return k_IsArc_Res_NO;
  }
  if (unpackSize != 0)
  {
    if (size < kHeaderSize + 2)
      return k_IsArc_Res_NEED_MORE;
    if (p[kHeaderSize] != 0)
      return k_IsArc_Res_NO;
    if (unpackSize != (UInt64)(Int64)-1)
    {
      if ((p[kHeaderSize + 1] & 0x80) != 0)
        return k_IsArc_Res_NO;
    }
  }
  if (!CheckDicSize(p + 1))
    // return k_IsArc_Res_YES_LOW_PROB;
    return k_IsArc_Res_NO;
  return k_IsArc_Res_YES;
}
}

API_FUNC_static_IsArc IsArc_Lzma86(const Byte *p, size_t size)
{
  if (size < 1)
    return k_IsArc_Res_NEED_MORE;
  Byte filterID = p[0];
  if (filterID != 0 && filterID != 1)
    return k_IsArc_Res_NO;
  return IsArc_Lzma(p + 1, size - 1);
}
}



Z7_COM7F_IMF(CHandler::Open(IInStream *inStream, const UInt64 *, IArchiveOpenCallback *))
{
  Close();
  
  const unsigned headerSize = GetHeaderSize();
  const UInt32 kBufSize = 1 << 7;
  Byte buf[kBufSize];
  size_t processedSize = kBufSize;
  RINOK(ReadStream(inStream, buf, &processedSize))
  if (processedSize < headerSize + 2)
    return S_FALSE;
  if (!_header.Parse(buf, _lzma86))
    return S_FALSE;
  const Byte *start = buf + headerSize;
  if (start[0] != 0 /* || (start[1] & 0x80) != 0 */ ) // empty stream with EOS is not 0x80
    return S_FALSE;

  RINOK(InStream_GetSize_SeekToEnd(inStream, _packSize))

  SizeT srcLen = (SizeT)processedSize - headerSize;

  if (srcLen > 10
      && _header.Size == 0
      // && _header.FilterID == 0
      && _header.LzmaProps[0] == 0
      )
    return S_FALSE;

  const UInt32 outLimit = 1 << 11;
  Byte outBuf[outLimit];

  SizeT outSize = outLimit;
  if (outSize > _header.Size)
    outSize = (SizeT)_header.Size;
  SizeT destLen = outSize;
  ELzmaStatus status;
  
  SRes res = LzmaDecode(outBuf, &destLen, start, &srcLen,
      _header.LzmaProps, 5, LZMA_FINISH_ANY,
      &status, &g_Alloc);
  
  if (res != SZ_OK)
    if (res != SZ_ERROR_INPUT_EOF)
      return S_FALSE;

  _isArc = true;
  _stream = inStream;
  _seqStream = inStream;
  _needSeekToStart = true;
  return S_OK;
}

Z7_COM7F_IMF(CHandler::OpenSeq(ISequentialInStream *stream))
{
  Close();
  _isArc = true;
  _seqStream = stream;
  return S_OK;
}

Z7_COM7F_IMF(CHandler::Close())
{
  _isArc = false;
  _needSeekToStart = false;
  _dataAfterEnd = false;
  _needMoreInput = false;
  _unsupported = false;
  _dataError = false;

  _packSize_Defined = false;
  _unpackSize_Defined = false;
  _numStreams_Defined = false;

  _packSize = 0;

  _stream.Release();
  _seqStream.Release();
   return S_OK;
}

Z7_CLASS_IMP_COM_1(
  CCompressProgressInfoImp,
  ICompressProgressInfo
)
  CMyComPtr<IArchiveOpenCallback> Callback;
public:
  UInt64 Offset;

  void Init(IArchiveOpenCallback *callback) { Callback = callback; }
};

Z7_COM7F_IMF(CCompressProgressInfoImp::SetRatioInfo(const UInt64 *inSize, const UInt64 * /* outSize */))
{
  if (Callback)
  {
    const UInt64 files = 0;
    const UInt64 val = Offset + *inSize;
    return Callback->SetCompleted(&files, &val);
  }
  return S_OK;
}

Z7_COM7F_IMF(CHandler::Extract(const UInt32 *indices, UInt32 numItems,
    Int32 testMode, IArchiveExtractCallback *extractCallback))
{
  COM_TRY_BEGIN

  if (numItems == 0)
    return S_OK;
  if (numItems != (UInt32)(Int32)-1 && (numItems != 1 || indices[0] != 0))
    return E_INVALIDARG;

  if (_packSize_Defined)
    RINOK(extractCallback->SetTotal(_packSize))
    
  Int32 opResult;
 {
  CMyComPtr<ISequentialOutStream> realOutStream;
  const Int32 askMode = testMode ?
      NExtract::NAskMode::kTest :
      NExtract::NAskMode::kExtract;
  RINOK(extractCallback->GetStream(0, &realOutStream, askMode))
  if (!testMode && !realOutStream)
    return S_OK;
  
  RINOK(extractCallback->PrepareOperation(askMode))

  CMyComPtr2_Create<ISequentialOutStream, CDummyOutStream> outStream;
  outStream->SetStream(realOutStream);
  outStream->Init();
  realOutStream.Release();

  CMyComPtr2_Create<ICompressProgressInfo, CLocalProgress> lps;
  lps->Init(extractCallback, true);

  if (_needSeekToStart)
  {
    if (!_stream)
      return E_FAIL;
    RINOK(InStream_SeekToBegin(_stream))
  }
  else
    _needSeekToStart = true;

  CDecoder decoder;
  RINOK(decoder.Create(_lzma86, _seqStream))
 
  bool firstItem = true;

  UInt64 packSize = 0;
  UInt64 unpackSize = 0;
  UInt64 numStreams = 0;

  bool dataAfterEnd = false;
  
  HRESULT hres = S_OK;

  for (;;)
  {
    lps->InSize = packSize;
    lps->OutSize = unpackSize;
    RINOK(lps->SetCur())

    const UInt32 kBufSize = 1 + 5 + 8;
    Byte buf[kBufSize];
    const UInt32 headerSize = GetHeaderSize();
    UInt32 processed;
    RINOK(decoder.ReadInput(buf, headerSize, &processed))
    if (processed != headerSize)
    {
      if (processed != 0)
        dataAfterEnd = true;
      break;
    }
  
    CHeader st;
    if (!st.Parse(buf, _lzma86))
    {
      dataAfterEnd = true;
      break;
    }
    numStreams++;
    firstItem = false;

    hres = decoder.Code(st, outStream, lps);

    packSize = decoder.GetInputProcessedSize();
    unpackSize = outStream->GetSize();
    
    if (hres == E_NOTIMPL)
    {
      _unsupported = true;
      hres = S_FALSE;
      break;
    }
    if (hres == S_FALSE)
      break;
    RINOK(hres)
  }

  if (firstItem)
  {
    _isArc = false;
    hres = S_FALSE;
  }
  else if (hres == S_OK || hres == S_FALSE)
  {
    if (dataAfterEnd)
      _dataAfterEnd = true;
    else if (decoder._lzmaDecoder->NeedsMoreInput())
      _needMoreInput = true;

    _packSize = packSize;
    _unpackSize = unpackSize;
    _numStreams = numStreams;
  
    _packSize_Defined = true;
    _unpackSize_Defined = true;
    _numStreams_Defined = true;
  }
  
  opResult = NExtract::NOperationResult::kOK;

  if (!_isArc)
    opResult = NExtract::NOperationResult::kIsNotArc;
  else if (_needMoreInput)
    opResult = NExtract::NOperationResult::kUnexpectedEnd;
  else if (_unsupported)
    opResult = NExtract::NOperationResult::kUnsupportedMethod;
  else if (_dataAfterEnd)
    opResult = NExtract::NOperationResult::kDataAfterEnd;
  else if (hres == S_FALSE)
    opResult = NExtract::NOperationResult::kDataError;
  else if (hres == S_OK)
    opResult = NExtract::NOperationResult::kOK;
  else
    return hres;

  // outStream.Release();
 }
  return extractCallback->SetOperationResult(opResult);

  COM_TRY_END
}

namespace NLzmaAr {

// 2, { 0x5D, 0x00 },

REGISTER_ARC_I_CLS_NO_SIG(
  CHandler(false),
  "lzma", "lzma", NULL, 0xA,
  0,
  NArcInfoFlags::kStartOpen |
  NArcInfoFlags::kKeepName,
  IsArc_Lzma)
 
}

namespace NLzma86Ar {

REGISTER_ARC_I_CLS_NO_SIG(
  CHandler(true),
  "lzma86", "lzma86", NULL, 0xB,
  0,
  NArcInfoFlags::kKeepName,
  IsArc_Lzma86)
 
}

}}
