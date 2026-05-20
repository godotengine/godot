// XzHandler.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "../../Common/ComTry.h"
#include "../../Common/Defs.h"
#include "../../Common/IntToString.h"
#include "../../Common/MyBuffer.h"
#include "../../Common/StringToInt.h"

#include "../../Windows/PropVariant.h"
#include "../../Windows/System.h"

#include "../Common/CWrappers.h"
#include "../Common/ProgressUtils.h"
#include "../Common/RegisterArc.h"
#include "../Common/StreamUtils.h"

#include "../Compress/CopyCoder.h"
#include "../Compress/XzDecoder.h"
#include "../Compress/XzEncoder.h"

#include "IArchive.h"

#include "Common/HandlerOut.h"

using namespace NWindows;

namespace NArchive {
namespace NXz {

#define k_LZMA2_Name "LZMA2"


struct CBlockInfo
{
  unsigned StreamFlags;
  UInt64 PackPos;
  UInt64 PackSize; // pure value from Index record, it doesn't include pad zeros
  UInt64 UnpackPos;
};


Z7_class_CHandler_final:
  public IInArchive,
  public IArchiveOpenSeq,
  public IInArchiveGetStream,
  public ISetProperties,
 #ifndef Z7_EXTRACT_ONLY
  public IOutArchive,
 #endif
  public CMyUnknownImp,
 #ifndef Z7_EXTRACT_ONLY
  public CMultiMethodProps
 #else
  public CCommonMethodProps
 #endif
{
  Z7_COM_QI_BEGIN2(IInArchive)
  Z7_COM_QI_ENTRY(IArchiveOpenSeq)
  Z7_COM_QI_ENTRY(IInArchiveGetStream)
  Z7_COM_QI_ENTRY(ISetProperties)
 #ifndef Z7_EXTRACT_ONLY
  Z7_COM_QI_ENTRY(IOutArchive)
 #endif
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE

  Z7_IFACE_COM7_IMP(IInArchive)
  Z7_IFACE_COM7_IMP(IArchiveOpenSeq)
  Z7_IFACE_COM7_IMP(IInArchiveGetStream)
  Z7_IFACE_COM7_IMP(ISetProperties)
 #ifndef Z7_EXTRACT_ONLY
  Z7_IFACE_COM7_IMP(IOutArchive)
 #endif

  bool _stat_defined;
  bool _stat2_defined;
  bool _isArc;
  bool _needSeekToStart;
  bool _firstBlockWasRead;
  SRes _stat2_decode_SRes;

  CXzStatInfo _stat;    // it's stat from backward parsing
  CXzStatInfo _stat2;   // it's data from forward parsing, if the decoder was called

  const CXzStatInfo *GetStat() const
  {
    if (_stat_defined) return &_stat;
    if (_stat2_defined) return &_stat2;
    return NULL;
  }
  
  AString _methodsString;


  #ifndef Z7_EXTRACT_ONLY

  UInt32 _filterId;
  UInt64 _numSolidBytes;

  void InitXz()
  {
    _filterId = 0;
    _numSolidBytes = XZ_PROPS_BLOCK_SIZE_AUTO;
  }

  #endif


  void Init()
  {
    #ifndef Z7_EXTRACT_ONLY
      InitXz();
      CMultiMethodProps::Init();
    #else
      CCommonMethodProps::InitCommon();
    #endif
  }
  
  HRESULT SetProperty(const wchar_t *name, const PROPVARIANT &value);

  HRESULT Open2(IInStream *inStream, /* UInt32 flags, */ IArchiveOpenCallback *callback);

  HRESULT Decode(NCompress::NXz::CDecoder &decoder,
      ISequentialInStream *seqInStream,
      ISequentialOutStream *outStream,
      ICompressProgressInfo *progress)
  {
    #ifndef Z7_ST
    decoder._numThreads = _numThreads;
    #endif
    decoder._memUsage = _memUsage_Decompress;

    const HRESULT hres = decoder.Decode(seqInStream, outStream,
        NULL, // *outSizeLimit
        true, // finishStream
        progress);
    
    if (decoder.MainDecodeSRes_wasUsed
        && decoder.MainDecodeSRes != SZ_ERROR_MEM
        && decoder.MainDecodeSRes != SZ_ERROR_UNSUPPORTED)
    {
      // if (!_stat2_defined)
      {
        _stat2_decode_SRes = decoder.MainDecodeSRes;
        _stat2 = decoder.Stat;
        _stat2_defined = true;
      }
    }

    if (hres == S_OK && progress)
    {
      // RINOK(
      progress->SetRatioInfo(&decoder.Stat.InSize, &decoder.Stat.OutSize);
    }
    return hres;
  }

public:
  CBlockInfo *_blocks;
  size_t _blocksArraySize;
  UInt64 _maxBlocksSize;
  CMyComPtr<IInStream> _stream;
  CMyComPtr<ISequentialInStream> _seqStream;

  CXzBlock _firstBlock;

  CHandler();
  ~CHandler();

  HRESULT SeekToPackPos(UInt64 pos)
  {
    return InStream_SeekSet(_stream, pos);
  }
};


CHandler::CHandler():
    _blocks(NULL),
    _blocksArraySize(0)
{
  #ifndef Z7_EXTRACT_ONLY
  InitXz();
  #endif
}

CHandler::~CHandler()
{
  MyFree(_blocks);
}


static const Byte kProps[] =
{
  kpidSize,
  kpidPackSize,
  kpidMethod
};

static const Byte kArcProps[] =
{
  kpidMethod,
  kpidNumStreams,
  kpidNumBlocks,
  kpidClusterSize,
  kpidCharacts
};

IMP_IInArchive_Props
IMP_IInArchive_ArcProps

static void Lzma2PropToString(AString &s, unsigned prop)
{
  char c = 0;
  UInt32 size;
  if ((prop & 1) == 0)
    size = prop / 2 + 12;
  else
  {
    c = 'k';
    size = (UInt32)(2 | (prop & 1)) << (prop / 2 + 1);
    if (prop > 17)
    {
      size >>= 10;
      c = 'm';
    }
  }
  s.Add_UInt32(size);
  if (c != 0)
    s.Add_Char(c);
}

struct CMethodNamePair
{
  UInt32 Id;
  const char *Name;
};

static const CMethodNamePair g_NamePairs[] =
{
  { XZ_ID_Subblock, "SB" },
  { XZ_ID_Delta, "Delta" },
  { XZ_ID_X86, "BCJ" },
  { XZ_ID_PPC, "PPC" },
  { XZ_ID_IA64, "IA64" },
  { XZ_ID_ARM, "ARM" },
  { XZ_ID_ARMT, "ARMT" },
  { XZ_ID_SPARC, "SPARC" },
  { XZ_ID_ARM64, "ARM64" },
  { XZ_ID_RISCV, "RISCV" },
  { XZ_ID_LZMA2, "LZMA2" }
};

static void AddMethodString(AString &s, const CXzFilter &f)
{
  const char *p = NULL;
  for (unsigned i = 0; i < Z7_ARRAY_SIZE(g_NamePairs); i++)
    if (g_NamePairs[i].Id == f.id)
    {
      p = g_NamePairs[i].Name;
      break;
    }
  char temp[32];
  if (!p)
  {
    ::ConvertUInt64ToString(f.id, temp);
    p = temp;
  }

  s += p;

  if (f.propsSize > 0)
  {
    s.Add_Colon();
    if (f.id == XZ_ID_LZMA2 && f.propsSize == 1)
      Lzma2PropToString(s, f.props[0]);
    else if (f.id == XZ_ID_Delta && f.propsSize == 1)
      s.Add_UInt32((UInt32)f.props[0] + 1);
    else if (f.id == XZ_ID_ARM64 && f.propsSize == 1)
      s.Add_UInt32((UInt32)f.props[0] + 16 + 2);
    else
    {
      s.Add_Char('[');
      for (UInt32 bi = 0; bi < f.propsSize; bi++)
      {
        const unsigned v = f.props[bi];
        s.Add_Char(GET_HEX_CHAR_UPPER(v >> 4));
        s.Add_Char(GET_HEX_CHAR_UPPER(v & 15));
      }
      s.Add_Char(']');
    }
  }
}

static const char * const kChecks[] =
{
    "NoCheck"
  , "CRC32"
  , NULL
  , NULL
  , "CRC64"
  , NULL
  , NULL
  , NULL
  , NULL
  , NULL
  , "SHA256"
  , NULL
  , NULL
  , NULL
  , NULL
  , NULL
};

static void AddCheckString(AString &s, const CXzs &xzs)
{
  size_t i;
  UInt32 mask = 0;
  for (i = 0; i < xzs.num; i++)
    mask |= ((UInt32)1 << XzFlags_GetCheckType(xzs.streams[i].flags));
  for (i = 0; i <= XZ_CHECK_MASK; i++)
    if (((mask >> i) & 1) != 0)
    {
      s.Add_Space_if_NotEmpty();
      if (kChecks[i])
        s += kChecks[i];
      else
      {
        s += "Check-";
        s.Add_UInt32((UInt32)i);
      }
    }
}

Z7_COM7F_IMF(CHandler::GetArchiveProperty(PROPID propID, PROPVARIANT *value))
{
  COM_TRY_BEGIN
  NCOM::CPropVariant prop;

  const CXzStatInfo *stat = GetStat();
  
  switch (propID)
  {
    case kpidPhySize: if (stat) prop = stat->InSize; break;
    case kpidNumStreams: if (stat && stat->NumStreams_Defined) prop = stat->NumStreams; break;
    case kpidNumBlocks: if (stat && stat->NumBlocks_Defined) prop = stat->NumBlocks; break;
    case kpidUnpackSize: if (stat && stat->UnpackSize_Defined) prop = stat->OutSize; break;
    case kpidClusterSize: if (_stat_defined && _stat.NumBlocks_Defined && stat->NumBlocks > 1) prop = _maxBlocksSize; break;
    case kpidCharacts:
      if (_firstBlockWasRead)
      {
        AString s;
        if (XzBlock_HasPackSize(&_firstBlock))
          s.Add_OptSpaced("BlockPackSize");
        if (XzBlock_HasUnpackSize(&_firstBlock))
          s.Add_OptSpaced("BlockUnpackSize");
        if (!s.IsEmpty())
          prop = s;
      }
      break;
        

    case kpidMethod: if (!_methodsString.IsEmpty()) prop = _methodsString; break;
    case kpidErrorFlags:
    {
      UInt32 v = 0;
      SRes sres = _stat2_decode_SRes;
      if (!_isArc)                      v |= kpv_ErrorFlags_IsNotArc;
      if (sres == SZ_ERROR_INPUT_EOF)   v |= kpv_ErrorFlags_UnexpectedEnd;
      if (_stat2_defined && _stat2.DataAfterEnd) v |= kpv_ErrorFlags_DataAfterEnd;
      if (sres == SZ_ERROR_ARCHIVE)     v |= kpv_ErrorFlags_HeadersError;
      if (sres == SZ_ERROR_UNSUPPORTED) v |= kpv_ErrorFlags_UnsupportedMethod;
      if (sres == SZ_ERROR_DATA)        v |= kpv_ErrorFlags_DataError;
      if (sres == SZ_ERROR_CRC)         v |= kpv_ErrorFlags_CrcError;
      if (v != 0)
        prop = v;
      break;
    }

    case kpidMainSubfile:
    {
      // debug only, comment it:
      // if (_blocks) prop = (UInt32)0;
      break;
    }
    default: break;
  }
  prop.Detach(value);
  return S_OK;
  COM_TRY_END
}

Z7_COM7F_IMF(CHandler::GetNumberOfItems(UInt32 *numItems))
{
  *numItems = 1;
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetProperty(UInt32, PROPID propID, PROPVARIANT *value))
{
  COM_TRY_BEGIN
  const CXzStatInfo *stat = GetStat();
  NCOM::CPropVariant prop;
  switch (propID)
  {
    case kpidSize: if (stat && stat->UnpackSize_Defined) prop = stat->OutSize; break;
    case kpidPackSize: if (stat) prop = stat->InSize; break;
    case kpidMethod: if (!_methodsString.IsEmpty()) prop = _methodsString; break;
    default: break;
  }
  prop.Detach(value);
  return S_OK;
  COM_TRY_END
}


struct COpenCallbackWrap
{
  ICompressProgress vt;
  IArchiveOpenCallback *OpenCallback;
  HRESULT Res;
  
  // new clang shows "non-POD" warning for offsetof(), if we use constructor instead of Init()
  void Init(IArchiveOpenCallback *progress);
};

static SRes OpenCallbackProgress(ICompressProgressPtr pp, UInt64 inSize, UInt64 /* outSize */)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(COpenCallbackWrap)
  if (p->OpenCallback)
    p->Res = p->OpenCallback->SetCompleted(NULL, &inSize);
  return HRESULT_To_SRes(p->Res, SZ_ERROR_PROGRESS);
}

void COpenCallbackWrap::Init(IArchiveOpenCallback *callback)
{
  vt.Progress = OpenCallbackProgress;
  OpenCallback = callback;
  Res = SZ_OK;
}


struct CXzsCPP
{
  CXzs p;
  CXzsCPP() { Xzs_CONSTRUCT(&p) }
  ~CXzsCPP() { Xzs_Free(&p, &g_Alloc); }
};

#define kInputBufSize ((size_t)1 << 10)

struct CLookToRead2_CPP: public CLookToRead2
{
  CLookToRead2_CPP()
  {
    buf = NULL;
    LookToRead2_CreateVTable(this,
        True // Lookahead ?
        );
  }
  void Alloc(size_t allocSize)
  {
    buf = (Byte *)MyAlloc(allocSize);
    if (buf)
      this->bufSize = allocSize;
  }
  ~CLookToRead2_CPP()
  {
    MyFree(buf);
  }
};


static HRESULT SRes_to_Open_HRESULT(SRes res)
{
  switch (res)
  {
    case SZ_OK: return S_OK;
    case SZ_ERROR_MEM: return E_OUTOFMEMORY;
    case SZ_ERROR_PROGRESS: return E_ABORT;
    /*
    case SZ_ERROR_UNSUPPORTED:
    case SZ_ERROR_CRC:
    case SZ_ERROR_DATA:
    case SZ_ERROR_ARCHIVE:
    case SZ_ERROR_NO_ARCHIVE:
      return S_FALSE;
    */
    default: break;
  }
  return S_FALSE;
}



HRESULT CHandler::Open2(IInStream *inStream, /* UInt32 flags, */ IArchiveOpenCallback *callback)
{
  _needSeekToStart = true;

  {
    CXzStreamFlags st;
    CSeqInStreamWrap inStreamWrap;
    
    inStreamWrap.Init(inStream);

    SRes res = Xz_ReadHeader(&st, &inStreamWrap.vt);
    
    if (inStreamWrap.Res != S_OK)
      return inStreamWrap.Res;
    if (res != SZ_OK)
      return SRes_to_Open_HRESULT(res);

    {
      CXzBlock block;
      BoolInt isIndex;
      UInt32 headerSizeRes;
    
      SRes res2 = XzBlock_ReadHeader(&block, &inStreamWrap.vt, &isIndex, &headerSizeRes);
      
      if (inStreamWrap.Res != S_OK)
        return inStreamWrap.Res;
      
      if (res2 != SZ_OK)
      {
        if (res2 == SZ_ERROR_INPUT_EOF)
        {
          _stat2_decode_SRes = res2;
          _stream = inStream;
          _seqStream = inStream;
          _isArc = true;
          return S_OK;
        }

        if (res2 == SZ_ERROR_ARCHIVE)
          return S_FALSE;
        // what codes are possible here ?
        // ?? res2 == SZ_ERROR_MEM           : is possible here
        // ?? res2 == SZ_ERROR_UNSUPPORTED   : is possible here
      }
      else if (!isIndex)
      {
        _firstBlockWasRead = true;
        _firstBlock = block;

        unsigned numFilters = XzBlock_GetNumFilters(&block);
        for (unsigned i = 0; i < numFilters; i++)
        {
          _methodsString.Add_Space_if_NotEmpty();
          AddMethodString(_methodsString, block.filters[i]);
        }
      }
    }
  }

  RINOK(InStream_GetSize_SeekToEnd(inStream, _stat.InSize))
  if (callback)
  {
    RINOK(callback->SetTotal(NULL, &_stat.InSize))
  }

  CSeekInStreamWrap inStreamImp;
  
  inStreamImp.Init(inStream);

  CLookToRead2_CPP lookStream;

  lookStream.Alloc(kInputBufSize);
  
  if (!lookStream.buf)
    return E_OUTOFMEMORY;

  lookStream.realStream = &inStreamImp.vt;
  LookToRead2_INIT(&lookStream)

  COpenCallbackWrap openWrap;
  openWrap.Init(callback);

  CXzsCPP xzs;
  Int64 startPosition;
  SRes res = Xzs_ReadBackward(&xzs.p, &lookStream.vt, &startPosition, &openWrap.vt, &g_Alloc);
  if (res == SZ_ERROR_PROGRESS)
    return (openWrap.Res == S_OK) ? E_FAIL : openWrap.Res;
  /*
  if (res == SZ_ERROR_NO_ARCHIVE && xzs.p.num > 0)
    res = SZ_OK;
  */
  if (res == SZ_OK && startPosition == 0)
  {
    _stat_defined = true;

    _stat.OutSize = Xzs_GetUnpackSize(&xzs.p);
    _stat.UnpackSize_Defined = true;

    _stat.NumStreams = xzs.p.num;
    _stat.NumStreams_Defined = true;
    
    _stat.NumBlocks = Xzs_GetNumBlocks(&xzs.p);
    _stat.NumBlocks_Defined = true;

    AddCheckString(_methodsString, xzs.p);

    const size_t numBlocks = (size_t)_stat.NumBlocks + 1;
    const size_t bytesAlloc = numBlocks * sizeof(CBlockInfo);
    
    if (bytesAlloc / sizeof(CBlockInfo) == _stat.NumBlocks + 1)
    {
      _blocks = (CBlockInfo *)MyAlloc(bytesAlloc);
      if (_blocks)
      {
        unsigned blockIndex = 0;
        UInt64 unpackPos = 0;
        
        for (size_t si = xzs.p.num; si != 0;)
        {
          si--;
          const CXzStream &str = xzs.p.streams[si];
          UInt64 packPos = str.startOffset + XZ_STREAM_HEADER_SIZE;
          
          for (size_t bi = 0; bi < str.numBlocks; bi++)
          {
            const CXzBlockSizes &bs = str.blocks[bi];
            const UInt64 packSizeAligned = bs.totalSize + ((0 - (unsigned)bs.totalSize) & 3);
            
            if (bs.unpackSize != 0)
            {
              if (blockIndex >= _stat.NumBlocks)
                return E_FAIL;

              CBlockInfo &block = _blocks[blockIndex++];
              block.StreamFlags = str.flags;
              block.PackSize = bs.totalSize; // packSizeAligned;
              block.PackPos = packPos;
              block.UnpackPos = unpackPos;
            }
            packPos += packSizeAligned;
            unpackPos += bs.unpackSize;
            if (_maxBlocksSize < bs.unpackSize)
              _maxBlocksSize = bs.unpackSize;
          }
        }
    
        /*
        if (blockIndex != _stat.NumBlocks)
        {
          // there are Empty blocks;
        }
        */
        if (_stat.OutSize != unpackPos)
          return E_FAIL;
        CBlockInfo &block = _blocks[blockIndex++];
        block.StreamFlags = 0;
        block.PackSize = 0;
        block.PackPos = 0;
        block.UnpackPos = unpackPos;
        _blocksArraySize = blockIndex;
      }
    }
  }
  else
  {
    res = SZ_OK;
  }

  RINOK(SRes_to_Open_HRESULT(res))

  _stream = inStream;
  _seqStream = inStream;
  _isArc = true;
  return S_OK;
}



Z7_COM7F_IMF(CHandler::Open(IInStream *inStream, const UInt64 *, IArchiveOpenCallback *callback))
{
  COM_TRY_BEGIN
  {
    Close();
    return Open2(inStream, callback);
  }
  COM_TRY_END
}

Z7_COM7F_IMF(CHandler::OpenSeq(ISequentialInStream *stream))
{
  Close();
  _seqStream = stream;
  _isArc = true;
  _needSeekToStart = false;
  return S_OK;
}

Z7_COM7F_IMF(CHandler::Close())
{
  XzStatInfo_Clear(&_stat);
  XzStatInfo_Clear(&_stat2);
  _stat_defined = false;
  _stat2_defined = false;
  _stat2_decode_SRes = SZ_OK;

  _isArc = false;
  _needSeekToStart = false;
  _firstBlockWasRead = false;

   _methodsString.Empty();
  _stream.Release();
  _seqStream.Release();

  MyFree(_blocks);
  _blocks = NULL;
  _blocksArraySize = 0;
  _maxBlocksSize = 0;

  return S_OK;
}


struct CXzUnpackerCPP2
{
  Byte *InBuf;
  // Byte *OutBuf;
  CXzUnpacker p;
  
  CXzUnpackerCPP2();
  ~CXzUnpackerCPP2();
};

CXzUnpackerCPP2::CXzUnpackerCPP2(): InBuf(NULL)
  // , OutBuf(NULL)
{
  XzUnpacker_Construct(&p, &g_Alloc);
}

CXzUnpackerCPP2::~CXzUnpackerCPP2()
{
  XzUnpacker_Free(&p);
  MidFree(InBuf);
  // MidFree(OutBuf);
}


Z7_CLASS_IMP_IInStream(
  CInStream
)

  UInt64 _virtPos;
public:
  UInt64 Size;
  UInt64 _cacheStartPos;
  size_t _cacheSize;
  CByteBuffer _cache;
  // UInt64 _startPos;
  CXzUnpackerCPP2 xz;

  void InitAndSeek()
  {
    _virtPos = 0;
    _cacheStartPos = 0;
    _cacheSize = 0;
    // _startPos = startPos;
  }

  CMyComPtr2<IInArchive, CHandler> _handlerSpec;
  // ~CInStream();
};

/*
CInStream::~CInStream()
{
  // _cache.Free();
}
*/

static size_t FindBlock(const CBlockInfo *blocks, size_t numBlocks, UInt64 pos)
{
  size_t left = 0, right = numBlocks;
  for (;;)
  {
    size_t mid = (left + right) / 2;
    if (mid == left)
      return left;
    if (pos < blocks[mid].UnpackPos)
      right = mid;
    else
      left = mid;
  }
}



static HRESULT DecodeBlock(CXzUnpackerCPP2 &xzu,
    ISequentialInStream *seqInStream,
    unsigned streamFlags,
    UInt64 packSize, // pure size from Index record, it doesn't include pad zeros
    size_t unpackSize, Byte *dest
    // , ICompressProgressInfo *progress
    )
{
  const size_t kInBufSize = (size_t)1 << 16;

  XzUnpacker_Init(&xzu.p);

  if (!xzu.InBuf)
  {
    xzu.InBuf = (Byte *)MidAlloc(kInBufSize);
    if (!xzu.InBuf)
      return E_OUTOFMEMORY;
  }
  
  xzu.p.streamFlags = (UInt16)streamFlags;
  XzUnpacker_PrepareToRandomBlockDecoding(&xzu.p);

  XzUnpacker_SetOutBuf(&xzu.p, dest, unpackSize);

  const UInt64 packSizeAligned = packSize + ((0 - (unsigned)packSize) & 3);
  UInt64 packRem = packSizeAligned;

  UInt32 inSize = 0;
  SizeT inPos = 0;
  SizeT outPos = 0;

  HRESULT readRes = S_OK;

  for (;;)
  {
    if (inPos == inSize && readRes == S_OK)
    {
      inPos = 0;
      inSize = 0;
      UInt32 rem = kInBufSize;
      if (rem > packRem)
        rem = (UInt32)packRem;
      if (rem != 0)
        readRes = seqInStream->Read(xzu.InBuf, rem, &inSize);
    }

    SizeT inLen = inSize - inPos;
    SizeT outLen = unpackSize - outPos;
    
    ECoderStatus status;

    const SRes res = XzUnpacker_Code(&xzu.p,
        // dest + outPos,
        NULL,
        &outLen,
        xzu.InBuf + inPos, &inLen,
        (inLen == 0), // srcFinished
        CODER_FINISH_END, &status);

    // return E_OUTOFMEMORY;
    // res = SZ_ERROR_CRC;

    if (res != SZ_OK)
    {
      if (res == SZ_ERROR_CRC)
        return S_FALSE;
      return SResToHRESULT(res);
    }

    inPos += inLen;
    outPos += outLen;

    packRem -= inLen;
  
    const BoolInt blockFinished = XzUnpacker_IsBlockFinished(&xzu.p);

    if ((inLen == 0 && outLen == 0) || blockFinished)
    {
      if (packRem != 0 || !blockFinished || unpackSize != outPos)
        return S_FALSE;
      if (XzUnpacker_GetPackSizeForIndex(&xzu.p) != packSize)
        return S_FALSE;
      return S_OK;
    }
  }
}


Z7_COM7F_IMF(CInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  COM_TRY_BEGIN

  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;

  {
    if (_virtPos >= Size)
      return S_OK; // (Size == _virtPos) ? S_OK: E_FAIL;
    {
      UInt64 rem = Size - _virtPos;
      if (size > rem)
        size = (UInt32)rem;
    }
  }

  if (size == 0)
    return S_OK;

  if (_virtPos < _cacheStartPos || _virtPos >= _cacheStartPos + _cacheSize)
  {
    const size_t bi = FindBlock(_handlerSpec->_blocks, _handlerSpec->_blocksArraySize, _virtPos);
    const CBlockInfo &block = _handlerSpec->_blocks[bi];
    const UInt64 unpackSize = _handlerSpec->_blocks[bi + 1].UnpackPos - block.UnpackPos;
    if (_cache.Size() < unpackSize)
      return E_FAIL;

    _cacheSize = 0;

    RINOK(_handlerSpec->SeekToPackPos(block.PackPos))
    RINOK(DecodeBlock(xz, _handlerSpec->_seqStream, block.StreamFlags, block.PackSize,
        (size_t)unpackSize, _cache))
    _cacheStartPos = block.UnpackPos;
    _cacheSize = (size_t)unpackSize;
  }

  {
    const size_t offset = (size_t)(_virtPos - _cacheStartPos);
    const size_t rem = _cacheSize - offset;
    if (size > rem)
      size = (UInt32)rem;
    memcpy(data, _cache.ConstData() + offset, size);
    _virtPos += size;
    if (processedSize)
      *processedSize = size;
    return S_OK;
  }

  COM_TRY_END
}
 

Z7_COM7F_IMF(CInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _virtPos; break;
    case STREAM_SEEK_END: offset += Size; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _virtPos = (UInt64)offset;
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}



static const UInt64 kMaxBlockSize_for_GetStream = (UInt64)1 << 40;

Z7_COM7F_IMF(CHandler::GetStream(UInt32 index, ISequentialInStream **stream))
{
  COM_TRY_BEGIN

  *stream = NULL;

  if (index != 0)
    return E_INVALIDARG;

  if (!_stat.UnpackSize_Defined
      || _maxBlocksSize == 0 // 18.02
      || _maxBlocksSize > kMaxBlockSize_for_GetStream
      || _maxBlocksSize != (size_t)_maxBlocksSize)
    return S_FALSE;

  size_t memSize;
  if (!NSystem::GetRamSize(memSize))
    memSize = (size_t)sizeof(size_t) << 28;
  {
    if (_maxBlocksSize > memSize / 4)
      return S_FALSE;
  }

  CMyComPtr2<ISequentialInStream, CInStream> spec;
  spec.Create_if_Empty();
  spec->_cache.Alloc((size_t)_maxBlocksSize);
  spec->_handlerSpec.SetFromCls(this);
  // spec->_handler = (IInArchive *)this;
  spec->Size = _stat.OutSize;
  spec->InitAndSeek();

  *stream = spec.Detach();
  return S_OK;
  
  COM_TRY_END
}


static Int32 Get_Extract_OperationResult(const NCompress::NXz::CDecoder &decoder)
{
  Int32 opRes;
  SRes sres = decoder.MainDecodeSRes;
  if (sres == SZ_ERROR_NO_ARCHIVE) // (!IsArc)
    opRes = NExtract::NOperationResult::kIsNotArc;
  else if (sres == SZ_ERROR_INPUT_EOF) // (UnexpectedEnd)
    opRes = NExtract::NOperationResult::kUnexpectedEnd;
  else if (decoder.Stat.DataAfterEnd)
    opRes = NExtract::NOperationResult::kDataAfterEnd;
  else if (sres == SZ_ERROR_CRC) // (CrcError)
    opRes = NExtract::NOperationResult::kCRCError;
  else if (sres == SZ_ERROR_UNSUPPORTED) // (Unsupported)
    opRes = NExtract::NOperationResult::kUnsupportedMethod;
  else if (sres == SZ_ERROR_ARCHIVE) //  (HeadersError)
    opRes = NExtract::NOperationResult::kDataError;
  else if (sres == SZ_ERROR_DATA)  // (DataError)
    opRes = NExtract::NOperationResult::kDataError;
  else if (sres != SZ_OK)
    opRes = NExtract::NOperationResult::kDataError;
  else
    opRes = NExtract::NOperationResult::kOK;
  return opRes;
}




Z7_COM7F_IMF(CHandler::Extract(const UInt32 *indices, UInt32 numItems,
    Int32 testMode, IArchiveExtractCallback *extractCallback))
{
  COM_TRY_BEGIN
  if (numItems == 0)
    return S_OK;
  if (numItems != (UInt32)(Int32)-1 && (numItems != 1 || indices[0] != 0))
    return E_INVALIDARG;

  const CXzStatInfo *stat = GetStat();

  if (stat)
    RINOK(extractCallback->SetTotal(stat->InSize))

  UInt64 currentTotalPacked = 0;
  RINOK(extractCallback->SetCompleted(&currentTotalPacked))
  Int32 opRes;
 {
  CMyComPtr<ISequentialOutStream> realOutStream;
  const Int32 askMode = testMode ?
      NExtract::NAskMode::kTest :
      NExtract::NAskMode::kExtract;
  
  RINOK(extractCallback->GetStream(0, &realOutStream, askMode))
  
  if (!testMode && !realOutStream)
    return S_OK;

  RINOK(extractCallback->PrepareOperation(askMode))

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


  NCompress::NXz::CDecoder decoder;

  const HRESULT hres = Decode(decoder, _seqStream, realOutStream, lps);

  if (!decoder.MainDecodeSRes_wasUsed)
    return hres == S_OK ? E_FAIL : hres;

  opRes = Get_Extract_OperationResult(decoder);
  if (opRes == NExtract::NOperationResult::kOK
      && hres != S_OK)
    opRes = NExtract::NOperationResult::kDataError;

  // realOutStream.Release();
 }
  return extractCallback->SetOperationResult(opRes);
  COM_TRY_END
}



#ifndef Z7_EXTRACT_ONLY

Z7_COM7F_IMF(CHandler::GetFileTimeType(UInt32 *timeType))
{
  *timeType = GET_FileTimeType_NotDefined_for_GetFileTimeType;
  // *timeType = NFileTimeType::kUnix;
  return S_OK;
}


Z7_COM7F_IMF(CHandler::UpdateItems(ISequentialOutStream *outStream, UInt32 numItems,
    IArchiveUpdateCallback *updateCallback))
{
  COM_TRY_BEGIN

  if (numItems == 0)
  {
    CSeqOutStreamWrap seqOutStream;
    seqOutStream.Init(outStream);
    SRes res = Xz_EncodeEmpty(&seqOutStream.vt);
    return SResToHRESULT(res);
  }
  
  if (numItems != 1)
    return E_INVALIDARG;

  {
    Z7_DECL_CMyComPtr_QI_FROM(
        IStreamSetRestriction,
        setRestriction, outStream)
    if (setRestriction)
      RINOK(setRestriction->SetRestriction(0, 0))
  }

  Int32 newData, newProps;
  UInt32 indexInArchive;
  if (!updateCallback)
    return E_FAIL;
  RINOK(updateCallback->GetUpdateItemInfo(0, &newData, &newProps, &indexInArchive))

  if (IntToBool(newProps))
  {
    {
      NCOM::CPropVariant prop;
      RINOK(updateCallback->GetProperty(0, kpidIsDir, &prop))
      if (prop.vt != VT_EMPTY)
        if (prop.vt != VT_BOOL || prop.boolVal != VARIANT_FALSE)
          return E_INVALIDARG;
    }
  }

  if (IntToBool(newData))
  {
    UInt64 dataSize;
    {
      NCOM::CPropVariant prop;
      RINOK(updateCallback->GetProperty(0, kpidSize, &prop))
      if (prop.vt != VT_UI8)
        return E_INVALIDARG;
      dataSize = prop.uhVal.QuadPart;
    }

    CMyComPtr2_Create<ICompressCoder, NCompress::NXz::CEncoder> encoder;

    CXzProps &xzProps = encoder->xzProps;
    CLzma2EncProps &lzma2Props = xzProps.lzma2Props;

    lzma2Props.lzmaProps.level = GetLevel();

    xzProps.reduceSize = dataSize;
    /*
    {
      NCOM::CPropVariant prop = (UInt64)dataSize;
      RINOK(encoder->SetCoderProp(NCoderPropID::kReduceSize, prop))
    }
    */

    #ifndef Z7_ST

#ifdef _WIN32
    // we don't use chunk multithreading inside lzma2 stream.
    // so we don't set xzProps.lzma2Props.numThreadGroups.
    if (_numThreadGroups > 1)
      xzProps.numThreadGroups = _numThreadGroups;
#endif
    
    UInt32 numThreads = _numThreads;

    const UInt32 kNumThreads_Max = 1024;
    if (numThreads > kNumThreads_Max)
      numThreads = kNumThreads_Max;

    if (!_numThreads_WasForced
        && _numThreads >= 1
        && _memUsage_WasSet)
    {
      COneMethodInfo oneMethodInfo;
      if (!_methods.IsEmpty())
        oneMethodInfo = _methods[0];

      SetGlobalLevelTo(oneMethodInfo);

      const bool numThreads_WasSpecifiedInMethod = (oneMethodInfo.Get_NumThreads() >= 0);
      if (!numThreads_WasSpecifiedInMethod)
      {
        // here we set the (NCoderPropID::kNumThreads) property in each method, only if there is no such property already
        CMultiMethodProps::SetMethodThreadsTo_IfNotFinded(oneMethodInfo, numThreads);
      }

      // printf("\n====== GetProcessGroupAffinity : \n");

      UInt64 cs = _numSolidBytes;
      if (cs != XZ_PROPS_BLOCK_SIZE_AUTO)
        oneMethodInfo.AddProp_BlockSize2(cs);
      cs = oneMethodInfo.Get_Xz_BlockSize();

      if (cs != XZ_PROPS_BLOCK_SIZE_AUTO &&
          cs != XZ_PROPS_BLOCK_SIZE_SOLID)
      {
        const UInt32 lzmaThreads = oneMethodInfo.Get_Lzma_NumThreads();
        const UInt32 numBlockThreads_Original = numThreads / lzmaThreads;

        if (numBlockThreads_Original > 1)
        {
          UInt32 numBlockThreads = numBlockThreads_Original;
          {
            const UInt64 lzmaMemUsage = oneMethodInfo.Get_Lzma_MemUsage(false);
            for (; numBlockThreads > 1; numBlockThreads--)
            {
              UInt64 size = numBlockThreads * (lzmaMemUsage + cs);
              UInt32 numPackChunks = numBlockThreads + (numBlockThreads / 8) + 1;
              if (cs < ((UInt32)1 << 26)) numPackChunks++;
              if (cs < ((UInt32)1 << 24)) numPackChunks++;
              if (cs < ((UInt32)1 << 22)) numPackChunks++;
              size += numPackChunks * cs;
              // printf("\nnumBlockThreads = %d, size = %d\n", (unsigned)(numBlockThreads), (unsigned)(size >> 20));
              if (size <= _memUsage_Compress)
                break;
            }
          }
          if (numBlockThreads == 0)
            numBlockThreads = 1;
          if (numBlockThreads != numBlockThreads_Original)
            numThreads = numBlockThreads * lzmaThreads;
        }
      }
    }
    xzProps.numTotalThreads = (int)numThreads;

    #endif // Z7_ST


    xzProps.blockSize = _numSolidBytes;
    if (_numSolidBytes == XZ_PROPS_BLOCK_SIZE_SOLID)
    {
      xzProps.lzma2Props.blockSize = LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID;
    }

    RINOK(encoder->SetCheckSize(_crcSize))

    {
      CXzFilterProps &filter = xzProps.filterProps;
      
      if (_filterId == XZ_ID_Delta)
      {
        bool deltaDefined = false;
        FOR_VECTOR (j, _filterMethod.Props)
        {
          const CProp &prop = _filterMethod.Props[j];
          if (prop.Id == NCoderPropID::kDefaultProp && prop.Value.vt == VT_UI4)
          {
            UInt32 delta = (UInt32)prop.Value.ulVal;
            if (delta < 1 || delta > 256)
              return E_INVALIDARG;
            filter.delta = delta;
            deltaDefined = true;
          }
          else
            return E_INVALIDARG;
        }
        if (!deltaDefined)
          return E_INVALIDARG;
      }
      filter.id = _filterId;
    }

    FOR_VECTOR (i, _methods)
    {
      COneMethodInfo &m = _methods[i];

      FOR_VECTOR (j, m.Props)
      {
        const CProp &prop = m.Props[j];
        RINOK(encoder->SetCoderProp(prop.Id, prop.Value))
      }
    }

    {
      CMyComPtr<ISequentialInStream> fileInStream;
      RINOK(updateCallback->GetStream(0, &fileInStream))
      if (!fileInStream)
        return S_FALSE;
      {
        CMyComPtr<IStreamGetSize> streamGetSize;
        fileInStream.QueryInterface(IID_IStreamGetSize, &streamGetSize);
        if (streamGetSize)
        {
          UInt64 size;
          if (streamGetSize->GetSize(&size) == S_OK)
            dataSize = size;
        }
      }
      RINOK(updateCallback->SetTotal(dataSize))
      CMyComPtr2_Create<ICompressProgressInfo, CLocalProgress> lps;
      lps->Init(updateCallback, true);
      RINOK(encoder.Interface()->Code(fileInStream, outStream, NULL, NULL, lps))
    }
      
    return updateCallback->SetOperationResult(NArchive::NUpdate::NOperationResult::kOK);
  }

  if (indexInArchive != 0)
    return E_INVALIDARG;

  Z7_DECL_CMyComPtr_QI_FROM(
      IArchiveUpdateCallbackFile,
      opCallback, updateCallback)
  if (opCallback)
  {
    RINOK(opCallback->ReportOperation(NEventIndexType::kInArcIndex, 0, NUpdateNotifyOp::kReplicate))
  }

  if (_stream)
  {
    const CXzStatInfo *stat = GetStat();
    if (stat)
    {
      RINOK(updateCallback->SetTotal(stat->InSize))
    }
    RINOK(InStream_SeekToBegin(_stream))
  }

  CMyComPtr2_Create<ICompressProgressInfo, CLocalProgress> lps;
  lps->Init(updateCallback, true);

  return NCompress::CopyStream(_stream, outStream, lps);

  COM_TRY_END
}

#endif


HRESULT CHandler::SetProperty(const wchar_t *nameSpec, const PROPVARIANT &value)
{
  UString name = nameSpec;
  name.MakeLower_Ascii();
  if (name.IsEmpty())
    return E_INVALIDARG;
  
  #ifndef Z7_EXTRACT_ONLY

  if (name[0] == L's')
  {
    const wchar_t *s = name.Ptr(1);
    if (*s == 0)
    {
      bool useStr = false;
      bool isSolid;
      switch (value.vt)
      {
        case VT_EMPTY: isSolid = true; break;
        case VT_BOOL: isSolid = (value.boolVal != VARIANT_FALSE); break;
        case VT_BSTR:
          if (!StringToBool(value.bstrVal, isSolid))
            useStr = true;
          break;
        default: return E_INVALIDARG;
      }
      if (!useStr)
      {
        _numSolidBytes = (isSolid ? XZ_PROPS_BLOCK_SIZE_SOLID : XZ_PROPS_BLOCK_SIZE_AUTO);
        return S_OK;
      }
    }
    return ParseSizeString(s, value,
        0, // percentsBase
        _numSolidBytes) ? S_OK: E_INVALIDARG;
  }

  return CMultiMethodProps::SetProperty(name, value);

  #else

  {
    HRESULT hres;
    if (SetCommonProperty(name, value, hres))
      return hres;
  }

  return E_INVALIDARG;
  
  #endif
}



Z7_COM7F_IMF(CHandler::SetProperties(const wchar_t * const *names, const PROPVARIANT *values, UInt32 numProps))
{
  COM_TRY_BEGIN

  Init();

  for (UInt32 i = 0; i < numProps; i++)
  {
    RINOK(SetProperty(names[i], values[i]))
  }

  #ifndef Z7_EXTRACT_ONLY

  if (!_filterMethod.MethodName.IsEmpty())
  {
    unsigned k;
    for (k = 0; k < Z7_ARRAY_SIZE(g_NamePairs); k++)
    {
      const CMethodNamePair &pair = g_NamePairs[k];
      if (StringsAreEqualNoCase_Ascii(_filterMethod.MethodName, pair.Name))
      {
        _filterId = pair.Id;
        break;
      }
    }
    if (k == Z7_ARRAY_SIZE(g_NamePairs))
      return E_INVALIDARG;
  }

  _methods.DeleteFrontal(GetNumEmptyMethods());
  if (_methods.Size() > 1)
    return E_INVALIDARG;
  if (_methods.Size() == 1)
  {
    AString &methodName = _methods[0].MethodName;
    if (methodName.IsEmpty())
      methodName = k_LZMA2_Name;
    else if (
        !methodName.IsEqualTo_Ascii_NoCase(k_LZMA2_Name)
        && !methodName.IsEqualTo_Ascii_NoCase("xz"))
      return E_INVALIDARG;
  }
  
  #endif

  return S_OK;

  COM_TRY_END
}


REGISTER_ARC_IO(
  "xz", "xz txz", "* .tar", 0xC,
  XZ_SIG, 0
  , NArcInfoFlags::kKeepName
  , 0
  , NULL)

}}
