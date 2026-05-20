// CoderMixer2.cpp

#include "StdAfx.h"

#include "CoderMixer2.h"

#ifdef USE_MIXER_ST

Z7_COM7F_IMF(CSequentialInStreamCalcSize::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 realProcessed = 0;
  HRESULT result = S_OK;
  if (_stream)
    result = _stream->Read(data, size, &realProcessed);
  _size += realProcessed;
  if (size != 0 && realProcessed == 0)
    _wasFinished = true;
  if (processedSize)
    *processedSize = realProcessed;
  return result;
}


Z7_COM7F_IMF(COutStreamCalcSize::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  HRESULT result = S_OK;
  if (_stream)
    result = _stream->Write(data, size, &size);
  _size += size;
  if (processedSize)
    *processedSize = size;
  return result;
}

Z7_COM7F_IMF(COutStreamCalcSize::OutStreamFinish())
{
  HRESULT result = S_OK;
  if (_stream)
  {
    CMyComPtr<IOutStreamFinish> outStreamFinish;
    _stream.QueryInterface(IID_IOutStreamFinish, &outStreamFinish);
    if (outStreamFinish)
      result = outStreamFinish->OutStreamFinish();
  }
  return result;
}

#endif




namespace NCoderMixer2 {

static void BoolVector_Fill_False(CBoolVector &v, unsigned size)
{
  v.ClearAndSetSize(size);
  bool *p = &v[0];
  for (unsigned i = 0; i < size; i++)
    p[i] = false;
}


HRESULT CCoder::CheckDataAfterEnd(bool &dataAfterEnd_Error /* , bool &InternalPackSizeError */) const
{
  if (Coder)
  {
    if (PackSizePointers.IsEmpty() || !PackSizePointers[0])
      return S_OK;
    CMyComPtr<ICompressGetInStreamProcessedSize> getInStreamProcessedSize;
    Coder.QueryInterface(IID_ICompressGetInStreamProcessedSize, (void **)&getInStreamProcessedSize);
    // if (!getInStreamProcessedSize) return E_FAIL;
    if (getInStreamProcessedSize)
    {
      UInt64 processed;
      RINOK(getInStreamProcessedSize->GetInStreamProcessedSize(&processed))
      if (processed != (UInt64)(Int64)-1)
      {
        const UInt64 size = PackSizes[0];
        if (processed < size && Finish)
          dataAfterEnd_Error = true;
        if (processed > size)
        {
          // InternalPackSizeError = true;
          // return S_FALSE;
        }
      }
    }
  }
  else if (Coder2)
  {
    CMyComPtr<ICompressGetInStreamProcessedSize2> getInStreamProcessedSize2;
    Coder2.QueryInterface(IID_ICompressGetInStreamProcessedSize2, (void **)&getInStreamProcessedSize2);
    if (getInStreamProcessedSize2)
    FOR_VECTOR (i, PackSizePointers)
    {
      if (!PackSizePointers[i])
        continue;
      UInt64 processed;
      RINOK(getInStreamProcessedSize2->GetInStreamProcessedSize2(i, &processed))
      if (processed != (UInt64)(Int64)-1)
      {
        const UInt64 size = PackSizes[i];
        if (processed < size && Finish)
          dataAfterEnd_Error = true;
        else if (processed > size)
        {
          // InternalPackSizeError = true;
          // return S_FALSE;
        }
      }
    }
  }

  return S_OK;
}



class CBondsChecks
{
  CBoolVector _coderUsed;

  bool Init();
  bool CheckCoder(unsigned coderIndex);
public:
  const CBindInfo *BindInfo;

  bool Check();
};

bool CBondsChecks::CheckCoder(unsigned coderIndex)
{
  const CCoderStreamsInfo &coder = BindInfo->Coders[coderIndex];

  if (coderIndex >= _coderUsed.Size() || _coderUsed[coderIndex])
    return false;
  _coderUsed[coderIndex] = true;
  
  const UInt32 start = BindInfo->Coder_to_Stream[coderIndex];

  for (unsigned i = 0; i < coder.NumStreams; i++)
  {
    UInt32 ind = start + i;
    
    if (BindInfo->IsStream_in_PackStreams(ind))
      continue;
    
    const int bond = BindInfo->FindBond_for_PackStream(ind);
    if (bond < 0)
      return false;
    if (!CheckCoder(BindInfo->Bonds[(unsigned)bond].UnpackIndex))
      return false;
  }
  
  return true;
}

bool CBondsChecks::Check()
{
  BoolVector_Fill_False(_coderUsed, BindInfo->Coders.Size());
  
  if (!CheckCoder(BindInfo->UnpackCoder))
    return false;

  FOR_VECTOR(i, _coderUsed)
    if (!_coderUsed[i])
      return false;

  return true;
}

void CBindInfo::ClearMaps()
{
  Coder_to_Stream.Clear();
  Stream_to_Coder.Clear();
}

bool CBindInfo::CalcMapsAndCheck()
{
  ClearMaps();

  UInt32 numStreams = 0;

  if (Coders.Size() == 0)
    return false;
  if (Coders.Size() - 1 != Bonds.Size())
    return false;

  FOR_VECTOR(i, Coders)
  {
    Coder_to_Stream.Add(numStreams);
    
    const CCoderStreamsInfo &c = Coders[i];
    
    for (unsigned j = 0; j < c.NumStreams; j++)
      Stream_to_Coder.Add(i);

    numStreams += c.NumStreams;
  }

  if (numStreams != GetNum_Bonds_and_PackStreams())
    return false;

  CBondsChecks bc;
  bc.BindInfo = this;
  return bc.Check();
}


void CCoder::SetCoderInfo(const UInt64 *unpackSize, const UInt64 * const *packSizes, bool finish)
{
  Finish = finish;

  if (unpackSize)
  {
    UnpackSize = *unpackSize;
    UnpackSizePointer = &UnpackSize;
  }
  else
  {
    UnpackSize = 0;
    UnpackSizePointer = NULL;
  }
  
  PackSizes.ClearAndSetSize((unsigned)NumStreams);
  PackSizePointers.ClearAndSetSize((unsigned)NumStreams);
  
  for (unsigned i = 0; i < NumStreams; i++)
  {
    if (packSizes && packSizes[i])
    {
      PackSizes[i] = *(packSizes[i]);
      PackSizePointers[i] = &PackSizes[i];
    }
    else
    {
      PackSizes[i] = 0;
      PackSizePointers[i] = NULL;
    }
  }
}

bool CMixer::Is_UnpackSize_Correct_for_Coder(UInt32 coderIndex)
{
  if (coderIndex == _bi.UnpackCoder)
    return true;
  
  const int bond = _bi.FindBond_for_UnpackStream(coderIndex);
  if (bond < 0)
    throw 20150213;
  
  /*
  UInt32 coderIndex, coderStreamIndex;
  _bi.GetCoder_for_Stream(_bi.Bonds[(unsigned)bond].PackIndex, coderIndex, coderStreamIndex);
  */
  const UInt32 nextCoder = _bi.Stream_to_Coder[_bi.Bonds[(unsigned)bond].PackIndex];
  
  if (!IsFilter_Vector[nextCoder])
    return false;
  
  return Is_UnpackSize_Correct_for_Coder(nextCoder);
}

bool CMixer::Is_PackSize_Correct_for_Stream(UInt32 streamIndex)
{
  if (_bi.IsStream_in_PackStreams(streamIndex))
    return true;
  
  const int bond = _bi.FindBond_for_PackStream(streamIndex);
  if (bond < 0)
    throw 20150213;

  const UInt32 nextCoder = _bi.Bonds[(unsigned)bond].UnpackIndex;

  if (!IsFilter_Vector[nextCoder])
    return false;
  
  return Is_PackSize_Correct_for_Coder(nextCoder);
}

bool CMixer::Is_PackSize_Correct_for_Coder(UInt32 coderIndex)
{
  const UInt32 startIndex = _bi.Coder_to_Stream[coderIndex];
  const UInt32 numStreams = _bi.Coders[coderIndex].NumStreams;
  for (UInt32 i = 0; i < numStreams; i++)
    if (!Is_PackSize_Correct_for_Stream(startIndex + i))
      return false;
  return true;
}

bool CMixer::IsThere_ExternalCoder_in_PackTree(UInt32 coderIndex)
{
  if (IsExternal_Vector[coderIndex])
    return true;
  const UInt32 startIndex = _bi.Coder_to_Stream[coderIndex];
  const UInt32 numStreams = _bi.Coders[coderIndex].NumStreams;
  for (UInt32 i = 0; i < numStreams; i++)
  {
    const UInt32 si = startIndex + i;
    if (_bi.IsStream_in_PackStreams(si))
      continue;
  
    const int bond = _bi.FindBond_for_PackStream(si);
    if (bond < 0)
      throw 20150213;

    if (IsThere_ExternalCoder_in_PackTree(_bi.Bonds[(unsigned)bond].UnpackIndex))
      return true;
  }
  return false;
}




#ifdef USE_MIXER_ST

CMixerST::CMixerST(bool encodeMode):
    CMixer(encodeMode)
    {}

CMixerST::~CMixerST() {}

void CMixerST::AddCoder(const CCreatedCoder &cod)
{
  IsFilter_Vector.Add(cod.IsFilter);
  IsExternal_Vector.Add(cod.IsExternal);
  // const CCoderStreamsInfo &c = _bi.Coders[_coders.Size()];
  CCoderST &c2 = _coders.AddNew();
  c2.NumStreams = cod.NumStreams;
  c2.Coder = cod.Coder;
  c2.Coder2 = cod.Coder2;

  /*
  if (isFilter)
  {
    c2.CanRead = true;
    c2.CanWrite = true;
  }
  else
  */
  {
    IUnknown *unk = (cod.Coder ? (IUnknown *)cod.Coder : (IUnknown *)cod.Coder2);
    {
      Z7_DECL_CMyComPtr_QI_FROM(ISequentialInStream, s, unk)
      c2.CanRead = (s != NULL);
    }
    {
      Z7_DECL_CMyComPtr_QI_FROM(ISequentialOutStream, s, unk)
      c2.CanWrite = (s != NULL);
    }
  }
}

CCoder &CMixerST::GetCoder(unsigned index)
{
  return _coders[index];
}

HRESULT CMixerST::ReInit2() { return S_OK; }

HRESULT CMixerST::GetInStream2(
    ISequentialInStream * const *inStreams, /* const UInt64 * const *inSizes, */
    UInt32 outStreamIndex, ISequentialInStream **inStreamRes)
{
  UInt32 coderIndex = outStreamIndex, coderStreamIndex = 0;

  if (EncodeMode)
  {
    _bi.GetCoder_for_Stream(outStreamIndex, coderIndex, coderStreamIndex);
    if (coderStreamIndex != 0)
      return E_NOTIMPL;
  }

  const CCoder &coder = _coders[coderIndex];
  
  CMyComPtr<ISequentialInStream> seqInStream;
  coder.QueryInterface(IID_ISequentialInStream, (void **)&seqInStream);
  if (!seqInStream)
    return E_NOTIMPL;

  const UInt32 numInStreams = EncodeMode ? 1 : coder.NumStreams;
  const UInt32 startIndex = EncodeMode ? coderIndex : _bi.Coder_to_Stream[coderIndex];

  bool isSet = false;
  
  if (numInStreams == 1)
  {
    CMyComPtr<ICompressSetInStream> setStream;
    coder.QueryInterface(IID_ICompressSetInStream, (void **)&setStream);
    if (setStream)
    {
      CMyComPtr<ISequentialInStream> seqInStream2;
      RINOK(GetInStream(inStreams, /* inSizes, */ startIndex + 0, &seqInStream2))
      RINOK(setStream->SetInStream(seqInStream2))
      isSet = true;
    }
  }
  
  if (!isSet && numInStreams != 0)
  {
    CMyComPtr<ICompressSetInStream2> setStream2;
    coder.QueryInterface(IID_ICompressSetInStream2, (void **)&setStream2);
    if (!setStream2)
      return E_NOTIMPL;
    
    for (UInt32 i = 0; i < numInStreams; i++)
    {
      CMyComPtr<ISequentialInStream> seqInStream2;
      RINOK(GetInStream(inStreams, /* inSizes, */ startIndex + i, &seqInStream2))
      RINOK(setStream2->SetInStream2(i, seqInStream2))
    }
  }

  *inStreamRes = seqInStream.Detach();
  return S_OK;
}


HRESULT CMixerST::GetInStream(
    ISequentialInStream * const *inStreams, /* const UInt64 * const *inSizes, */
    UInt32 inStreamIndex, ISequentialInStream **inStreamRes)
{
  CMyComPtr<ISequentialInStream> seqInStream;
  
  {
    int index = -1;
    if (EncodeMode)
    {
      if (_bi.UnpackCoder == inStreamIndex)
        index = 0;
    }
    else
      index = _bi.FindStream_in_PackStreams(inStreamIndex);

    if (index >= 0)
    {
      seqInStream = inStreams[(unsigned)index];
      *inStreamRes = seqInStream.Detach();
      return S_OK;
    }
  }
  
  const int bond = FindBond_for_Stream(
      true, // forInputStream
      inStreamIndex);
  if (bond < 0)
    return E_INVALIDARG;

  RINOK(GetInStream2(inStreams, /* inSizes, */
      _bi.Bonds[(unsigned)bond].Get_OutIndex(EncodeMode), &seqInStream))

  while (_binderStreams.Size() <= (unsigned)bond)
    _binderStreams.AddNew();
  CStBinderStream &bs = _binderStreams[(unsigned)bond];

  if (bs.StreamRef || bs.InStreamSpec)
    return E_NOTIMPL;
  
  CSequentialInStreamCalcSize *spec = new CSequentialInStreamCalcSize;
  bs.StreamRef = spec;
  bs.InStreamSpec = spec;
  
  spec->SetStream(seqInStream);
  spec->Init();
  
  seqInStream = bs.InStreamSpec;

  *inStreamRes = seqInStream.Detach();
  return S_OK;
}


HRESULT CMixerST::GetOutStream(
    ISequentialOutStream * const *outStreams, /* const UInt64 * const *outSizes, */
    UInt32 outStreamIndex, ISequentialOutStream **outStreamRes)
{
  CMyComPtr<ISequentialOutStream> seqOutStream;
  
  {
    int index = -1;
    if (!EncodeMode)
    {
      if (_bi.UnpackCoder == outStreamIndex)
        index = 0;
    }
    else
      index = _bi.FindStream_in_PackStreams(outStreamIndex);

    if (index >= 0)
    {
      seqOutStream = outStreams[(unsigned)index];
      *outStreamRes = seqOutStream.Detach();
      return S_OK;
    }
  }
  
  const int bond = FindBond_for_Stream(
      false, // forInputStream
      outStreamIndex);
  if (bond < 0)
    return E_INVALIDARG;

  const UInt32 inStreamIndex = _bi.Bonds[(unsigned)bond].Get_InIndex(EncodeMode);

  UInt32 coderIndex = inStreamIndex;
  UInt32 coderStreamIndex = 0;

  if (!EncodeMode)
    _bi.GetCoder_for_Stream(inStreamIndex, coderIndex, coderStreamIndex);

  CCoder &coder = _coders[coderIndex];

  /*
  if (!coder.Coder)
    return E_NOTIMPL;
  */

  coder.QueryInterface(IID_ISequentialOutStream, (void **)&seqOutStream);
  if (!seqOutStream)
    return E_NOTIMPL;

  const UInt32 numOutStreams = EncodeMode ? coder.NumStreams : 1;
  const UInt32 startIndex = EncodeMode ? _bi.Coder_to_Stream[coderIndex]: coderIndex;

  bool isSet = false;

  if (numOutStreams == 1)
  {
    CMyComPtr<ICompressSetOutStream> setOutStream;
    coder.Coder.QueryInterface(IID_ICompressSetOutStream, &setOutStream);
    if (setOutStream)
    {
      CMyComPtr<ISequentialOutStream> seqOutStream2;
      RINOK(GetOutStream(outStreams, /* outSizes, */ startIndex + 0, &seqOutStream2))
      RINOK(setOutStream->SetOutStream(seqOutStream2))
      isSet = true;
    }
  }

  if (!isSet && numOutStreams != 0)
  {
    return E_NOTIMPL;
    /*
    CMyComPtr<ICompressSetOutStream2> setStream2;
    coder.QueryInterface(IID_ICompressSetOutStream2, (void **)&setStream2);
    if (!setStream2)
      return E_NOTIMPL;
    for (UInt32 i = 0; i < numOutStreams; i++)
    {
      CMyComPtr<ISequentialOutStream> seqOutStream2;
      RINOK(GetOutStream(outStreams, startIndex + i, &seqOutStream2))
      RINOK(setStream2->SetOutStream2(i, seqOutStream2))
    }
    */
  }

  while (_binderStreams.Size() <= (unsigned)bond)
    _binderStreams.AddNew();
  CStBinderStream &bs = _binderStreams[(unsigned)bond];

  if (bs.StreamRef || bs.OutStreamSpec)
    return E_NOTIMPL;
  
  COutStreamCalcSize *spec = new COutStreamCalcSize;
  bs.StreamRef = (ISequentialOutStream *)spec;
  bs.OutStreamSpec = spec;
  
  spec->SetStream(seqOutStream);
  spec->Init();

  seqOutStream = bs.OutStreamSpec;
  
  *outStreamRes = seqOutStream.Detach();
  return S_OK;
}


static HRESULT GetError(HRESULT res, HRESULT res2)
{
  if (res == res2)
    return res;
  if (res == S_OK)
    return res2;
  if (res == k_My_HRESULT_WritingWasCut)
  {
    if (res2 != S_OK)
      return res2;
  }
  return res;
}


HRESULT CMixerST::FinishStream(UInt32 streamIndex)
{
  {
    int index = -1;
    if (!EncodeMode)
    {
      if (_bi.UnpackCoder == streamIndex)
        index = 0;
    }
    else
      index = _bi.FindStream_in_PackStreams(streamIndex);

    if (index >= 0)
      return S_OK;
  }

  const int bond = FindBond_for_Stream(
      false, // forInputStream
      streamIndex);
  if (bond < 0)
    return E_INVALIDARG;

  const UInt32 inStreamIndex = _bi.Bonds[(unsigned)bond].Get_InIndex(EncodeMode);

  UInt32 coderIndex = inStreamIndex;
  UInt32 coderStreamIndex = 0;
  if (!EncodeMode)
    _bi.GetCoder_for_Stream(inStreamIndex, coderIndex, coderStreamIndex);

  CCoder &coder = _coders[coderIndex];
  CMyComPtr<IOutStreamFinish> finish;
  coder.QueryInterface(IID_IOutStreamFinish, (void **)&finish);
  HRESULT res = S_OK;
  if (finish)
  {
    res = finish->OutStreamFinish();
  }
  return GetError(res, FinishCoder(coderIndex));
}


HRESULT CMixerST::FinishCoder(UInt32 coderIndex)
{
  CCoder &coder = _coders[coderIndex];

  const UInt32 numOutStreams = EncodeMode ? coder.NumStreams : 1;
  const UInt32 startIndex = EncodeMode ? _bi.Coder_to_Stream[coderIndex]: coderIndex;

  HRESULT res = S_OK;
  for (unsigned i = 0; i < numOutStreams; i++)
    res = GetError(res, FinishStream(startIndex + i));
  return res;
}


void CMixerST::SelectMainCoder(bool useFirst)
{
  unsigned ci = _bi.UnpackCoder;
  
  int firstNonFilter = -1;
  unsigned firstAllowed = ci;
  
  for (;;)
  {
    const CCoderST &coder = _coders[ci];
    // break;
    
    if (ci != _bi.UnpackCoder)
      if (EncodeMode ? !coder.CanWrite : !coder.CanRead)
      {
        firstAllowed = ci;
        firstNonFilter = -2;
      }
      
    if (coder.NumStreams != 1)
      break;
    
    const UInt32 st = _bi.Coder_to_Stream[ci];
    if (_bi.IsStream_in_PackStreams(st))
      break;
    const int bond = _bi.FindBond_for_PackStream(st);
    if (bond < 0)
      throw 20150213;
    
    if (EncodeMode ? !coder.CanRead : !coder.CanWrite)
      break;
    
    if (firstNonFilter == -1 && !IsFilter_Vector[ci])
      firstNonFilter = (int)ci;
    
    ci = _bi.Bonds[(unsigned)bond].UnpackIndex;
  }
  
  if (useFirst)
    ci = firstAllowed;
  else if (firstNonFilter >= 0)
    ci = (unsigned)firstNonFilter;

  MainCoderIndex = ci;
}


HRESULT CMixerST::Code(
    ISequentialInStream * const *inStreams,
    ISequentialOutStream * const *outStreams,
    ICompressProgressInfo *progress,
    bool &dataAfterEnd_Error)
{
  // InternalPackSizeError = false;
  dataAfterEnd_Error = false;

  _binderStreams.Clear();
  const unsigned ci = MainCoderIndex;
 
  const CCoder &mainCoder = _coders[MainCoderIndex];

  CObjectVector< CMyComPtr<ISequentialInStream> > seqInStreams;
  CObjectVector< CMyComPtr<ISequentialOutStream> > seqOutStreams;
  
  const UInt32 numInStreams  =  EncodeMode ? 1 : mainCoder.NumStreams;
  const UInt32 numOutStreams = !EncodeMode ? 1 : mainCoder.NumStreams;
  
  const UInt32 startInIndex  =  EncodeMode ? ci : _bi.Coder_to_Stream[ci];
  const UInt32 startOutIndex = !EncodeMode ? ci : _bi.Coder_to_Stream[ci];
  
  UInt32 i;

  for (i = 0; i < numInStreams; i++)
  {
    CMyComPtr<ISequentialInStream> seqInStream;
    RINOK(GetInStream(inStreams, /* inSizes, */ startInIndex + i, &seqInStream))
    seqInStreams.Add(seqInStream);
  }
  
  for (i = 0; i < numOutStreams; i++)
  {
    CMyComPtr<ISequentialOutStream> seqOutStream;
    RINOK(GetOutStream(outStreams, /* outSizes, */ startOutIndex + i, &seqOutStream))
    seqOutStreams.Add(seqOutStream);
  }
  
  CRecordVector< ISequentialInStream * > seqInStreamsSpec;
  CRecordVector< ISequentialOutStream * > seqOutStreamsSpec;
  
  for (i = 0; i < numInStreams; i++)
    seqInStreamsSpec.Add(seqInStreams[i]);
  for (i = 0; i < numOutStreams; i++)
    seqOutStreamsSpec.Add(seqOutStreams[i]);

  for (i = 0; i < _coders.Size(); i++)
  {
    if (i == ci)
      continue;
   
    CCoder &coder = _coders[i];

    if (EncodeMode)
    {
      CMyComPtr<ICompressInitEncoder> initEncoder;
      coder.QueryInterface(IID_ICompressInitEncoder, (void **)&initEncoder);
      if (initEncoder)
      {
        RINOK(initEncoder->InitEncoder())
      }
    }
    else
    {
      CMyComPtr<ICompressSetOutStreamSize> setOutStreamSize;
      coder.QueryInterface(IID_ICompressSetOutStreamSize, (void **)&setOutStreamSize);
      if (setOutStreamSize)
      {
        RINOK(setOutStreamSize->SetOutStreamSize(
            EncodeMode ? coder.PackSizePointers[0] : coder.UnpackSizePointer))
      }
    }
  }

  const UInt64 * const *isSizes2 = EncodeMode ? &mainCoder.UnpackSizePointer : mainCoder.PackSizePointers.ConstData();
  const UInt64 * const *outSizes2 = EncodeMode ? mainCoder.PackSizePointers.ConstData() : &mainCoder.UnpackSizePointer;

  HRESULT res;
  if (mainCoder.Coder)
  {
    res = mainCoder.Coder->Code(
        seqInStreamsSpec[0], seqOutStreamsSpec[0],
        isSizes2[0], outSizes2[0],
        progress);
  }
  else
  {
    res = mainCoder.Coder2->Code(
        seqInStreamsSpec.ConstData(), isSizes2, numInStreams,
        seqOutStreamsSpec.ConstData(), outSizes2, numOutStreams,
        progress);
  }

  if (res == k_My_HRESULT_WritingWasCut)
    res = S_OK;

  if (res == S_OK || res == S_FALSE)
  {
    res = GetError(res, FinishCoder(ci));
  }

  for (i = 0; i < _binderStreams.Size(); i++)
  {
    const CStBinderStream &bs = _binderStreams[i];
    if (bs.InStreamSpec)
      bs.InStreamSpec->ReleaseStream();
    else
      bs.OutStreamSpec->ReleaseStream();
  }

  if (res == k_My_HRESULT_WritingWasCut)
    res = S_OK;

  if (res != S_OK)
    return res;

  for (i = 0; i < _coders.Size(); i++)
  {
    RINOK(_coders[i].CheckDataAfterEnd(dataAfterEnd_Error /*, InternalPackSizeError */))
  }

  return S_OK;
}


HRESULT CMixerST::GetMainUnpackStream(
    ISequentialInStream * const *inStreams,
    ISequentialInStream **inStreamRes)
{
  CMyComPtr<ISequentialInStream> seqInStream;

  RINOK(GetInStream2(inStreams, /* inSizes, */
      _bi.UnpackCoder, &seqInStream))
  
  FOR_VECTOR (i, _coders)
  {
    CCoder &coder = _coders[i];
    CMyComPtr<ICompressSetOutStreamSize> setOutStreamSize;
    coder.QueryInterface(IID_ICompressSetOutStreamSize, (void **)&setOutStreamSize);
    if (setOutStreamSize)
    {
      RINOK(setOutStreamSize->SetOutStreamSize(coder.UnpackSizePointer))
    }
  }
  
  *inStreamRes = seqInStream.Detach();
  return S_OK;
}


UInt64 CMixerST::GetBondStreamSize(unsigned bondIndex) const
{
  const CStBinderStream &bs = _binderStreams[bondIndex];
  if (bs.InStreamSpec)
    return bs.InStreamSpec->GetSize();
  return bs.OutStreamSpec->GetSize();
}

#endif






#ifdef USE_MIXER_MT


void CCoderMT::Execute()
{
  try
  {
    Code(NULL);
  }
  catch(...)
  {
    Result = E_FAIL;
  }
}

void CCoderMT::Code(ICompressProgressInfo *progress)
{
  unsigned numInStreams = EncodeMode ? 1 : NumStreams;
  unsigned numOutStreams = EncodeMode ? NumStreams : 1;

  InStreamPointers.ClearAndReserve(numInStreams);
  OutStreamPointers.ClearAndReserve(numOutStreams);

  unsigned i;
  
  for (i = 0; i < numInStreams; i++)
    InStreamPointers.AddInReserved((ISequentialInStream *)InStreams[i]);
  
  for (i = 0; i < numOutStreams; i++)
    OutStreamPointers.AddInReserved((ISequentialOutStream *)OutStreams[i]);

  // we suppose that UnpackSizePointer and PackSizePointers contain correct pointers.
  /*
  if (UnpackSizePointer)
    UnpackSizePointer = &UnpackSize;
  for (i = 0; i < NumStreams; i++)
    if (PackSizePointers[i])
      PackSizePointers[i] = &PackSizes[i];
  */

  CReleaser releaser(*this);
  
  if (Coder)
    Result = Coder->Code(InStreamPointers[0], OutStreamPointers[0],
        EncodeMode ? UnpackSizePointer : PackSizePointers[0],
        EncodeMode ? PackSizePointers[0] : UnpackSizePointer,
        progress);
  else
    Result = Coder2->Code(
        InStreamPointers.ConstData(),  EncodeMode ? &UnpackSizePointer : PackSizePointers.ConstData(), numInStreams,
        OutStreamPointers.ConstData(), EncodeMode ? PackSizePointers.ConstData(): &UnpackSizePointer, numOutStreams,
        progress);
}

HRESULT CMixerMT::SetBindInfo(const CBindInfo &bindInfo)
{
  CMixer::SetBindInfo(bindInfo);
  
  _streamBinders.Clear();
  FOR_VECTOR (i, _bi.Bonds)
  {
    // RINOK(_streamBinders.AddNew().CreateEvents())
    _streamBinders.AddNew();
  }
  return S_OK;
}

void CMixerMT::AddCoder(const CCreatedCoder &cod)
{
  IsFilter_Vector.Add(cod.IsFilter);
  IsExternal_Vector.Add(cod.IsExternal);
  // const CCoderStreamsInfo &c = _bi.Coders[_coders.Size()];
  CCoderMT &c2 = _coders.AddNew();
  c2.NumStreams = cod.NumStreams;
  c2.Coder = cod.Coder;
  c2.Coder2 = cod.Coder2;
  c2.EncodeMode = EncodeMode;
}

CCoder &CMixerMT::GetCoder(unsigned index)
{
  return _coders[index];
}

HRESULT CMixerMT::ReInit2()
{
  FOR_VECTOR (i, _streamBinders)
  {
    RINOK(_streamBinders[i].Create_ReInit())
  }
  return S_OK;
}

void CMixerMT::SelectMainCoder(bool useFirst)
{
  unsigned ci = _bi.UnpackCoder;

  if (!useFirst)
  for (;;)
  {
    if (_coders[ci].NumStreams != 1)
      break;
    if (!IsFilter_Vector[ci])
      break;
    
    UInt32 st = _bi.Coder_to_Stream[ci];
    if (_bi.IsStream_in_PackStreams(st))
      break;
    const int bond = _bi.FindBond_for_PackStream(st);
    if (bond < 0)
      throw 20150213;
    ci = _bi.Bonds[(unsigned)bond].UnpackIndex;
  }
  
  MainCoderIndex = ci;
}

HRESULT CMixerMT::Init(ISequentialInStream * const *inStreams, ISequentialOutStream * const *outStreams)
{
  unsigned i;
  
  for (i = 0; i < _coders.Size(); i++)
  {
    CCoderMT &coderInfo = _coders[i];
    const CCoderStreamsInfo &csi = _bi.Coders[i];
    
    UInt32 j;

    const unsigned numInStreams = EncodeMode ? 1 : csi.NumStreams;
    const unsigned numOutStreams = EncodeMode ? csi.NumStreams : 1;

    coderInfo.InStreams.Clear();
    for (j = 0; j < numInStreams; j++)
      coderInfo.InStreams.AddNew();
    
    coderInfo.OutStreams.Clear();
    for (j = 0; j < numOutStreams; j++)
      coderInfo.OutStreams.AddNew();
  }

  for (i = 0; i < _bi.Bonds.Size(); i++)
  {
    const CBond &bond = _bi.Bonds[i];
   
    UInt32 inCoderIndex, inCoderStreamIndex;
    UInt32 outCoderIndex, outCoderStreamIndex;
    
    {
      UInt32 coderIndex, coderStreamIndex;
      _bi.GetCoder_for_Stream(bond.PackIndex, coderIndex, coderStreamIndex);

      inCoderIndex = EncodeMode ? bond.UnpackIndex : coderIndex;
      outCoderIndex = EncodeMode ? coderIndex : bond.UnpackIndex;

      inCoderStreamIndex = EncodeMode ? 0 : coderStreamIndex;
      outCoderStreamIndex = EncodeMode ? coderStreamIndex : 0;
    }

    _streamBinders[i].CreateStreams2(
        _coders[inCoderIndex].InStreams[inCoderStreamIndex],
        _coders[outCoderIndex].OutStreams[outCoderStreamIndex]);

    CMyComPtr<ICompressSetBufSize> inSetSize, outSetSize;
    _coders[inCoderIndex].QueryInterface(IID_ICompressSetBufSize, (void **)&inSetSize);
    _coders[outCoderIndex].QueryInterface(IID_ICompressSetBufSize, (void **)&outSetSize);
    if (inSetSize && outSetSize)
    {
      const UInt32 kBufSize = 1 << 19;
      inSetSize->SetInBufSize(inCoderStreamIndex, kBufSize);
      outSetSize->SetOutBufSize(outCoderStreamIndex, kBufSize);
    }
  }

  {
    CCoderMT &cod = _coders[_bi.UnpackCoder];
    if (EncodeMode)
      cod.InStreams[0] = inStreams[0];
    else
      cod.OutStreams[0] = outStreams[0];
  }

  for (i = 0; i < _bi.PackStreams.Size(); i++)
  {
    UInt32 coderIndex, coderStreamIndex;
    _bi.GetCoder_for_Stream(_bi.PackStreams[i], coderIndex, coderStreamIndex);
    CCoderMT &cod = _coders[coderIndex];
    if (EncodeMode)
      cod.OutStreams[coderStreamIndex] = outStreams[i];
    else
      cod.InStreams[coderStreamIndex] = inStreams[i];
  }
  
  return S_OK;
}

HRESULT CMixerMT::ReturnIfError(HRESULT code)
{
  FOR_VECTOR (i, _coders)
    if (_coders[i].Result == code)
      return code;
  return S_OK;
}

HRESULT CMixerMT::Code(
    ISequentialInStream * const *inStreams,
    ISequentialOutStream * const *outStreams,
    ICompressProgressInfo *progress,
    bool &dataAfterEnd_Error)
{
  // InternalPackSizeError = false;
  dataAfterEnd_Error = false;

  Init(inStreams, outStreams);

  unsigned i;
  for (i = 0; i < _coders.Size(); i++)
    if (i != MainCoderIndex)
    {
      const WRes wres = _coders[i].Create();
      if (wres != 0)
        return HRESULT_FROM_WIN32(wres);
    }

  for (i = 0; i < _coders.Size(); i++)
    if (i != MainCoderIndex)
    {
      const WRes wres = _coders[i].Start();
      if (wres != 0)
        return HRESULT_FROM_WIN32(wres);
    }

  _coders[MainCoderIndex].Code(progress);

  WRes wres = 0;
  for (i = 0; i < _coders.Size(); i++)
    if (i != MainCoderIndex)
    {
      WRes wres2 = _coders[i].WaitExecuteFinish();
      if (wres == 0)
        wres = wres2;
    }
  if (wres != 0)
    return HRESULT_FROM_WIN32(wres);

  RINOK(ReturnIfError(E_ABORT))
  RINOK(ReturnIfError(E_OUTOFMEMORY))

  for (i = 0; i < _coders.Size(); i++)
  {
    HRESULT result = _coders[i].Result;
    if (result != S_OK
        && result != k_My_HRESULT_WritingWasCut
        && result != S_FALSE
        && result != E_FAIL)
      return result;
  }

  RINOK(ReturnIfError(S_FALSE))

  for (i = 0; i < _coders.Size(); i++)
  {
    HRESULT result = _coders[i].Result;
    if (result != S_OK && result != k_My_HRESULT_WritingWasCut)
      return result;
  }

  for (i = 0; i < _coders.Size(); i++)
  {
    RINOK(_coders[i].CheckDataAfterEnd(dataAfterEnd_Error /* , InternalPackSizeError */))
  }

  return S_OK;
}

UInt64 CMixerMT::GetBondStreamSize(unsigned bondIndex) const
{
  return _streamBinders[bondIndex].ProcessedSize;
}

#endif

}
