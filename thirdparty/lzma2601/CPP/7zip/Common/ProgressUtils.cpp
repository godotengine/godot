// ProgressUtils.cpp

#include "StdAfx.h"

#include "ProgressUtils.h"

CLocalProgress::CLocalProgress():
    SendRatio(true),
    SendProgress(true),
    ProgressOffset(0),
    InSize(0),
    OutSize(0)
  {}

void CLocalProgress::Init(IProgress *progress, bool inSizeIsMain)
{
  _ratioProgress.Release();
  _progress = progress;
  _progress.QueryInterface(IID_ICompressProgressInfo, &_ratioProgress);
  _inSizeIsMain = inSizeIsMain;
}

Z7_COM7F_IMF(CLocalProgress::SetRatioInfo(const UInt64 *inSize, const UInt64 *outSize))
{
  UInt64 inSize2 = InSize;
  UInt64 outSize2 = OutSize;
  
  if (inSize)
    inSize2 += (*inSize);
  if (outSize)
    outSize2 += (*outSize);
  
  if (SendRatio && _ratioProgress)
  {
    RINOK(_ratioProgress->SetRatioInfo(&inSize2, &outSize2))
  }
  
  if (SendProgress)
  {
    inSize2 += ProgressOffset;
    outSize2 += ProgressOffset;
    return _progress->SetCompleted(_inSizeIsMain ? &inSize2 : &outSize2);
  }
  
  return S_OK;
}

HRESULT CLocalProgress::SetCur()
{
  return SetRatioInfo(NULL, NULL);
}
