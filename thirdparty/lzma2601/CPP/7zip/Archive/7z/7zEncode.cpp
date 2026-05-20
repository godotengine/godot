// 7zEncode.cpp

#include "StdAfx.h"

#include "../../../Common/ComTry.h"

#include "../../Common/CreateCoder.h"
#include "../../Common/FilterCoder.h"
#include "../../Common/LimitedStreams.h"
#include "../../Common/InOutTempBuffer.h"
#include "../../Common/ProgressUtils.h"
#include "../../Common/StreamObjects.h"

#include "7zEncode.h"
#include "7zSpecStream.h"

namespace NArchive {
namespace N7z {

void CEncoder::InitBindConv()
{
  unsigned numIn = _bindInfo.Coders.Size();
  
  SrcIn_to_DestOut.ClearAndSetSize(numIn);
  DestOut_to_SrcIn.ClearAndSetSize(numIn);

  unsigned numOut = _bindInfo.GetNum_Bonds_and_PackStreams();
  SrcOut_to_DestIn.ClearAndSetSize(numOut);
  // _DestIn_to_SrcOut.ClearAndSetSize(numOut);

  UInt32 destIn = 0;
  UInt32 destOut = 0;

  for (unsigned i = _bindInfo.Coders.Size(); i != 0;)
  {
    i--;

    const NCoderMixer2::CCoderStreamsInfo &coder = _bindInfo.Coders[i];

    numIn--;
    numOut -= coder.NumStreams;
    
    SrcIn_to_DestOut[numIn] = destOut;
    DestOut_to_SrcIn[destOut] = numIn;

    destOut++;
  
    for (UInt32 j = 0; j < coder.NumStreams; j++, destIn++)
    {
      UInt32 index = numOut + j;
      SrcOut_to_DestIn[index] = destIn;
      // _DestIn_to_SrcOut[destIn] = index;
    }
  }
}

void CEncoder::SetFolder(CFolder &folder)
{
  folder.Bonds.SetSize(_bindInfo.Bonds.Size());
  
  unsigned i;

  for (i = 0; i < _bindInfo.Bonds.Size(); i++)
  {
    CBond &fb = folder.Bonds[i];
    const NCoderMixer2::CBond &mixerBond = _bindInfo.Bonds[_bindInfo.Bonds.Size() - 1 - i];
    fb.PackIndex = SrcOut_to_DestIn[mixerBond.PackIndex];
    fb.UnpackIndex = SrcIn_to_DestOut[mixerBond.UnpackIndex];
  }
  
  folder.Coders.SetSize(_bindInfo.Coders.Size());
  
  for (i = 0; i < _bindInfo.Coders.Size(); i++)
  {
    CCoderInfo &coderInfo = folder.Coders[i];
    const NCoderMixer2::CCoderStreamsInfo &coderStreamsInfo = _bindInfo.Coders[_bindInfo.Coders.Size() - 1 - i];
    
    coderInfo.NumStreams = coderStreamsInfo.NumStreams;
    coderInfo.MethodID = _decompressionMethods[i];
    // we don't free coderInfo.Props here. So coderInfo.Props can be non-empty.
  }
  
  folder.PackStreams.SetSize(_bindInfo.PackStreams.Size());
  
  for (i = 0; i < _bindInfo.PackStreams.Size(); i++)
    folder.PackStreams[i] = SrcOut_to_DestIn[_bindInfo.PackStreams[i]];
}



static HRESULT SetCoderProps2(const CProps &props, const UInt64 *dataSizeReduce, IUnknown *coder)
{
  Z7_DECL_CMyComPtr_QI_FROM(
      ICompressSetCoderProperties,
      setCoderProperties, coder)
  if (setCoderProperties)
    return props.SetCoderProps(setCoderProperties, dataSizeReduce);
  return props.AreThereNonOptionalProps() ? E_INVALIDARG : S_OK;
}



void CMtEncMultiProgress::Init(ICompressProgressInfo *progress)
{
  _progress = progress;
  OutSize = 0;
}

Z7_COM7F_IMF(CMtEncMultiProgress::SetRatioInfo(const UInt64 *inSize, const UInt64 * /* outSize */))
{
  UInt64 outSize2;
  {
    #ifndef Z7_ST
    NWindows::NSynchronization::CCriticalSectionLock lock(CriticalSection);
    #endif
    outSize2 = OutSize;
  }
  
  if (_progress)
    return _progress->SetRatioInfo(inSize, &outSize2);
   
  return S_OK;
}



HRESULT CEncoder::CreateMixerCoder(
    DECL_EXTERNAL_CODECS_LOC_VARS
    const UInt64 *inSizeForReduce)
{
  #ifdef USE_MIXER_MT
  #ifdef USE_MIXER_ST
  if (_options.MultiThreadMixer)
  #endif
  {
    _mixerMT = new NCoderMixer2::CMixerMT(true);
    _mixerRef = _mixerMT;
    _mixer = _mixerMT;
  }
  #ifdef USE_MIXER_ST
  else
  #endif
  #endif
  {
    #ifdef USE_MIXER_ST
    _mixerST = new NCoderMixer2::CMixerST(true);
    _mixerRef = _mixerST;
    _mixer = _mixerST;
    #endif
  }

  RINOK(_mixer->SetBindInfo(_bindInfo))

  FOR_VECTOR (m, _options.Methods)
  {
    const CMethodFull &methodFull = _options.Methods[m];

    CCreatedCoder cod;
    
    if (methodFull.CodecIndex >= 0)
    {
      RINOK(CreateCoder_Index(
        EXTERNAL_CODECS_LOC_VARS
        (unsigned)methodFull.CodecIndex, true, cod))
    }
    else
    {
      RINOK(CreateCoder_Id(
        EXTERNAL_CODECS_LOC_VARS
        methodFull.Id, true, cod))
    }

    if (!cod.Coder && !cod.Coder2)
    {
      return E_NOTIMPL; // unsupported method, if encoder
      // return E_FAIL;
    }

    if (cod.NumStreams != methodFull.NumStreams)
      return E_FAIL;

    CMyComPtr<IUnknown> encoderCommon = cod.Coder ? (IUnknown *)cod.Coder : (IUnknown *)cod.Coder2;
   
    #ifndef Z7_ST
    if (methodFull.Set_NumThreads)
    {
      CMyComPtr<ICompressSetCoderMt> setCoderMt;
      encoderCommon.QueryInterface(IID_ICompressSetCoderMt, &setCoderMt);
      if (setCoderMt)
      {
        RINOK(setCoderMt->SetNumberOfThreads(
            /* _options.NumThreads */
            methodFull.NumThreads
            ))
      }
    }
    #endif
        
    RINOK(SetCoderProps2(methodFull, inSizeForReduce, encoderCommon))

    /*
    CMyComPtr<ICryptoResetSalt> resetSalt;
    encoderCommon.QueryInterface(IID_ICryptoResetSalt, (void **)&resetSalt);
    if (resetSalt)
    {
      resetSalt->ResetSalt();
    }
    */

    // now there is no codec that uses another external codec
    /*
    #ifdef Z7_EXTERNAL_CODECS
    CMyComPtr<ISetCompressCodecsInfo> setCompressCodecsInfo;
    encoderCommon.QueryInterface(IID_ISetCompressCodecsInfo, (void **)&setCompressCodecsInfo);
    if (setCompressCodecsInfo)
    {
      // we must use g_ExternalCodecs also
      RINOK(setCompressCodecsInfo->SetCompressCodecsInfo(_externalCodecs->GetCodecs));
    }
    #endif
    */
    
    CMyComPtr<ICryptoSetPassword> cryptoSetPassword;
    encoderCommon.QueryInterface(IID_ICryptoSetPassword, &cryptoSetPassword);

    if (cryptoSetPassword)
    {
      const unsigned sizeInBytes = _options.Password.Len() * 2;
      CByteBuffer_Wipe buffer(sizeInBytes);
      for (unsigned i = 0; i < _options.Password.Len(); i++)
      {
        wchar_t c = _options.Password[i];
        ((Byte *)buffer)[i * 2] = (Byte)c;
        ((Byte *)buffer)[i * 2 + 1] = (Byte)(c >> 8);
      }
      RINOK(cryptoSetPassword->CryptoSetPassword((const Byte *)buffer, (UInt32)sizeInBytes))
    }

    _mixer->AddCoder(cod);
  }
  return S_OK;
}



Z7_CLASS_IMP_COM_1(
  CSequentialOutTempBufferImp2
  , ISequentialOutStream
)
public:
  CInOutTempBuffer TempBuffer;
  CMtEncMultiProgress *_mtProgressSpec;
  
  CSequentialOutTempBufferImp2(): _mtProgressSpec(NULL) {}
};

Z7_COM7F_IMF(CSequentialOutTempBufferImp2::Write(const void *data, UInt32 size, UInt32 *processed))
{
  COM_TRY_BEGIN
  if (processed)
    *processed = 0;
  RINOK(TempBuffer.Write_HRESULT(data, size))
  if (processed)
    *processed = size;
  if (_mtProgressSpec)
    _mtProgressSpec->AddOutSize(size);
  return S_OK;
  COM_TRY_END
}


Z7_CLASS_IMP_COM_1(
  CSequentialOutMtNotify
  , ISequentialOutStream
)
public:
  CMyComPtr<ISequentialOutStream> _stream;
  CMtEncMultiProgress *_mtProgressSpec;
  
  CSequentialOutMtNotify(): _mtProgressSpec(NULL) {}
};

Z7_COM7F_IMF(CSequentialOutMtNotify::Write(const void *data, UInt32 size, UInt32 *processed))
{
  UInt32 realProcessed = 0;
  HRESULT res = _stream->Write(data, size, &realProcessed);
  if (processed)
    *processed = realProcessed;
  if (_mtProgressSpec)
    _mtProgressSpec->AddOutSize(size);
  return res;
}


static HRESULT FillProps_from_Coder(IUnknown *coder, CByteBuffer &props)
{
  Z7_DECL_CMyComPtr_QI_FROM(
      ICompressWriteCoderProperties,
      writeCoderProperties, coder)
  if (writeCoderProperties)
  {
    CMyComPtr2_Create<ISequentialOutStream, CDynBufSeqOutStream> outStreamSpec;
    outStreamSpec->Init();
    RINOK(writeCoderProperties->WriteCoderProperties(outStreamSpec))
    outStreamSpec->CopyToBuffer(props);
  }
  else
    props.Free();
  return S_OK;
}

HRESULT CEncoder::Encode1(
    DECL_EXTERNAL_CODECS_LOC_VARS
    ISequentialInStream *inStream,
    // const UInt64 *inStreamSize,
    const UInt64 *inSizeForReduce,
    UInt64 expectedDataSize,
    CFolder &folderItem,
    // CRecordVector<UInt64> &coderUnpackSizes,
    // UInt64 &unpackSize,
    ISequentialOutStream *outStream,
    CRecordVector<UInt64> &packSizes,
    ICompressProgressInfo *compressProgress)
{
  RINOK(EncoderConstr())

  if (!_mixerRef)
  {
    RINOK(CreateMixerCoder(EXTERNAL_CODECS_LOC_VARS inSizeForReduce))
  }
  
  RINOK(_mixer->ReInit2())

  CMyComPtr2<ICompressProgressInfo, CMtEncMultiProgress> mtProgress;
  CMyComPtr2<ISequentialOutStream, CSequentialOutMtNotify> mtOutStreamNotify;

  CRecordVector<CSequentialOutTempBufferImp2 *> tempBufferSpecs;
  CObjectVector<CMyComPtr<ISequentialOutStream> > tempBuffers;
  
  unsigned i;

  for (i = 1; i < _bindInfo.PackStreams.Size(); i++)
  {
    CSequentialOutTempBufferImp2 *tempBufferSpec = new CSequentialOutTempBufferImp2();
    CMyComPtr<ISequentialOutStream> tempBuffer = tempBufferSpec;
    tempBufferSpecs.Add(tempBufferSpec);
    tempBuffers.Add(tempBuffer);
  }

  const unsigned numMethods = _bindInfo.Coders.Size();

  for (i = 0; i < numMethods; i++)
    _mixer->SetCoderInfo(i, NULL, NULL, false);


  /* inStreamSize can be used by BCJ2 to set optimal range of conversion.
     But current BCJ2 encoder uses also another way to check exact size of current file.
     So inStreamSize is not required. */

  /*
  if (inStreamSize)
    _mixer->SetCoderInfo(_bindInfo.UnpackCoder, inStreamSize, NULL);
  */

  
  /*
  CSequentialInStreamSizeCount2 *inStreamSizeCountSpec = new CSequentialInStreamSizeCount2;
  CMyComPtr<ISequentialInStream> inStreamSizeCount = inStreamSizeCountSpec;
  */

  CSequentialOutStreamSizeCount *outStreamSizeCountSpec = NULL;
  CMyComPtr<ISequentialOutStream> outStreamSizeCount;

  // inStreamSizeCountSpec->Init(inStream);

  // ISequentialInStream *inStreamPointer = inStreamSizeCount;
  ISequentialInStream *inStreamPointer = inStream;

  CRecordVector<ISequentialOutStream *> outStreamPointers;
  
  SetFolder(folderItem);

  for (i = 0; i < numMethods; i++)
  {
    IUnknown *coder = _mixer->GetCoder(i).GetUnknown();
    /*
    {
      CEncoder *sfEncoder = NULL;
      Z7_DECL_CMyComPtr_QI_FROM(
          IGetSfEncoderInternal,
          sf, coder)
      if (sf)
      {
        RINOK(sf->GetSfEncoder(&sfEncoder));
        if (!sfEncoder)
          return E_FAIL;

      }
    }
    */
    /*
    #ifdef Z7_EXTERNAL_CODECS
    {
      Z7_DECL_CMyComPtr_QI_FROM(
          ISetCompressCodecsInfo,
          setCompressCodecsInfo, coder)
      if (setCompressCodecsInfo)
      {
        // we must use g_ExternalCodecs also
        RINOK(setCompressCodecsInfo->SetCompressCodecsInfo(_externalCodecs->GetCodecs))
      }
    }
    #endif
    */
    {
      Z7_DECL_CMyComPtr_QI_FROM(
          ICryptoResetInitVector,
          resetInitVector, coder)
      if (resetInitVector)
      {
        RINOK(resetInitVector->ResetInitVector())
      }
    }
    {
      Z7_DECL_CMyComPtr_QI_FROM(
          ICompressSetCoderPropertiesOpt,
          optProps, coder)
      if (optProps)
      {
        const PROPID propID = NCoderPropID::kExpectedDataSize;
        NWindows::NCOM::CPropVariant prop = (UInt64)expectedDataSize;
        RINOK(optProps->SetCoderPropertiesOpt(&propID, &prop, 1))
      }
    }
    // we must write properties from coder after ResetInitVector()
    RINOK(FillProps_from_Coder(coder, folderItem.Coders[numMethods - 1 - i].Props))
  }

  _mixer->SelectMainCoder(false);
  const UInt32 mainCoder = _mixer->MainCoderIndex;

  bool useMtProgress = false;
  if (!_mixer->Is_PackSize_Correct_for_Coder(mainCoder))
  {
    #ifdef Z7_ST
    if (!_mixer->IsThere_ExternalCoder_in_PackTree(mainCoder))
    #endif
      useMtProgress = true;
  }

  if (useMtProgress)
  {
    mtProgress.SetFromCls(new CMtEncMultiProgress);
    mtProgress->Init(compressProgress);
    
    mtOutStreamNotify.SetFromCls(new CSequentialOutMtNotify);
    mtOutStreamNotify->_stream = outStream;
    mtOutStreamNotify->_mtProgressSpec = mtProgress.ClsPtr();
    
    FOR_VECTOR (t, tempBufferSpecs)
    {
      tempBufferSpecs[t]->_mtProgressSpec = mtProgress.ClsPtr();
    }
  }
  
  
  if (_bindInfo.PackStreams.Size() != 0)
  {
    outStreamSizeCountSpec = new CSequentialOutStreamSizeCount;
    outStreamSizeCount = outStreamSizeCountSpec;
    outStreamSizeCountSpec->SetStream(mtOutStreamNotify.IsDefined() ?
        mtOutStreamNotify.Interface() : outStream);
    outStreamSizeCountSpec->Init();
    outStreamPointers.Add(outStreamSizeCount);
  }

  for (i = 1; i < _bindInfo.PackStreams.Size(); i++)
    outStreamPointers.Add(tempBuffers[i - 1]);

  bool dataAfterEnd_Error;

  RINOK(_mixer->Code(
      &inStreamPointer,
      outStreamPointers.ConstData(),
      mtProgress.IsDefined() ? mtProgress.Interface() :
        compressProgress, dataAfterEnd_Error))
  
  if (_bindInfo.PackStreams.Size() != 0)
    packSizes.Add(outStreamSizeCountSpec->GetSize());
  
  for (i = 1; i < _bindInfo.PackStreams.Size(); i++)
  {
    CInOutTempBuffer &iotb = tempBufferSpecs[i - 1]->TempBuffer;
    RINOK(iotb.WriteToStream(outStream))
    packSizes.Add(iotb.GetDataSize());
  }

  /* Code() in some future codec can change properties.
     v23: so we fill properties again after Code() */
  for (i = 0; i < numMethods; i++)
  {
    IUnknown *coder = _mixer->GetCoder(i).GetUnknown();
    RINOK(FillProps_from_Coder(coder, folderItem.Coders[numMethods - 1 - i].Props))
  }

  return S_OK;
}


void CEncoder::Encode_Post(
      UInt64 unpackSize,
      CRecordVector<UInt64> &coderUnpackSizes)
{
  // unpackSize = 0;
  for (unsigned i = 0; i < _bindInfo.Coders.Size(); i++)
  {
    const int bond = _bindInfo.FindBond_for_UnpackStream(DestOut_to_SrcIn[i]);
    UInt64 streamSize;
    if (bond < 0)
    {
      // streamSize = inStreamSizeCountSpec->GetSize();
      // unpackSize = streamSize;
      streamSize = unpackSize;
    }
    else
      streamSize = _mixer->GetBondStreamSize((unsigned)bond);
    coderUnpackSizes.Add(streamSize);
  }
}


CEncoder::CEncoder(const CCompressionMethodMode &options):
    _constructed(false)
{
  if (options.IsEmpty())
    throw 1;

  _options = options;

  #ifdef USE_MIXER_ST
    _mixerST = NULL;
  #endif
  
  #ifdef USE_MIXER_MT
    _mixerMT = NULL;
  #endif

  _mixer = NULL;
}


HRESULT CEncoder::EncoderConstr()
{
  if (_constructed)
    return S_OK;
  if (_options.Methods.IsEmpty())
  {
    // it has only password method;
    if (!_options.PasswordIsDefined)
      throw 1;
    if (!_options.Bonds.IsEmpty())
      throw 1;

    CMethodFull method;
    method.Id = k_AES;
    method.NumStreams = 1;
    _options.Methods.Add(method);

    NCoderMixer2::CCoderStreamsInfo coderStreamsInfo;
    coderStreamsInfo.NumStreams = 1;
    _bindInfo.Coders.Add(coderStreamsInfo);
  
    _bindInfo.PackStreams.Add(0);
    _bindInfo.UnpackCoder = 0;
  }
  else
  {

  UInt32 numOutStreams = 0;
  unsigned i;
  
  for (i = 0; i < _options.Methods.Size(); i++)
  {
    const CMethodFull &methodFull = _options.Methods[i];
    NCoderMixer2::CCoderStreamsInfo cod;
    
    cod.NumStreams = methodFull.NumStreams;

    if (_options.Bonds.IsEmpty())
    {
      // if there are no bonds in options, we create bonds via first streams of coders
      if (i != _options.Methods.Size() - 1)
      {
        NCoderMixer2::CBond bond;
        bond.PackIndex = numOutStreams;
        bond.UnpackIndex = i + 1; // it's next coder
        _bindInfo.Bonds.Add(bond);
      }
      else if (cod.NumStreams != 0)
        _bindInfo.PackStreams.Insert(0, numOutStreams);
      
      for (UInt32 j = 1; j < cod.NumStreams; j++)
        _bindInfo.PackStreams.Add(numOutStreams + j);
    }
    
    numOutStreams += cod.NumStreams;

    _bindInfo.Coders.Add(cod);
  }

  if (!_options.Bonds.IsEmpty())
  {
    for (i = 0; i < _options.Bonds.Size(); i++)
    {
      NCoderMixer2::CBond mixerBond;
      const CBond2 &bond = _options.Bonds[i];
      if (bond.InCoder >= _bindInfo.Coders.Size()
          || bond.OutCoder >= _bindInfo.Coders.Size()
          || bond.OutStream >= _bindInfo.Coders[bond.OutCoder].NumStreams)
        return E_INVALIDARG;
      mixerBond.PackIndex = _bindInfo.GetStream_for_Coder(bond.OutCoder) + bond.OutStream;
      mixerBond.UnpackIndex = bond.InCoder;
      _bindInfo.Bonds.Add(mixerBond);
    }

    for (i = 0; i < numOutStreams; i++)
      if (_bindInfo.FindBond_for_PackStream(i) == -1)
        _bindInfo.PackStreams.Add(i);
  }

  if (!_bindInfo.SetUnpackCoder())
    return E_INVALIDARG;

  if (!_bindInfo.CalcMapsAndCheck())
    return E_INVALIDARG;

  if (_bindInfo.PackStreams.Size() != 1)
  {
    /* main_PackStream is pack stream of main path of coders tree.
       We find main_PackStream, and place to start of list of out streams.
       It allows to use more optimal memory usage for temp buffers,
       if main_PackStream is largest stream. */

    UInt32 ci = _bindInfo.UnpackCoder;
    
    for (;;)
    {
      if (_bindInfo.Coders[ci].NumStreams == 0)
        break;
      
      const UInt32 outIndex = _bindInfo.Coder_to_Stream[ci];
      const int bond = _bindInfo.FindBond_for_PackStream(outIndex);
      if (bond >= 0)
      {
        ci = _bindInfo.Bonds[(unsigned)bond].UnpackIndex;
        continue;
      }
      
      const int si = _bindInfo.FindStream_in_PackStreams(outIndex);
      if (si >= 0)
        _bindInfo.PackStreams.MoveToFront((unsigned)si);
      break;
    }
  }

  if (_options.PasswordIsDefined)
  {
    unsigned numCryptoStreams = _bindInfo.PackStreams.Size();

    unsigned numInStreams = _bindInfo.Coders.Size();
    
    for (i = 0; i < numCryptoStreams; i++)
    {
      NCoderMixer2::CBond bond;
      bond.UnpackIndex = numInStreams + i;
      bond.PackIndex = _bindInfo.PackStreams[i];
      _bindInfo.Bonds.Add(bond);
    }
    _bindInfo.PackStreams.Clear();

    /*
    if (numCryptoStreams == 0)
      numCryptoStreams = 1;
    */

    for (i = 0; i < numCryptoStreams; i++)
    {
      CMethodFull method;
      method.NumStreams = 1;
      method.Id = k_AES;
      _options.Methods.Add(method);

      NCoderMixer2::CCoderStreamsInfo cod;
      cod.NumStreams = 1;
      _bindInfo.Coders.Add(cod);

      _bindInfo.PackStreams.Add(numOutStreams++);
    }
  }

  }

  for (unsigned i = _options.Methods.Size(); i != 0;)
    _decompressionMethods.Add(_options.Methods[--i].Id);

  if (_bindInfo.Coders.Size() > 16)
    return E_INVALIDARG;
  if (_bindInfo.GetNum_Bonds_and_PackStreams() > 16)
    return E_INVALIDARG;

  if (!_bindInfo.CalcMapsAndCheck())
    return E_INVALIDARG;

  InitBindConv();
  _constructed = true;
  return S_OK;
}

CEncoder::~CEncoder() {}

}}
