// BcjCoder.cpp

#include "StdAfx.h"

#include "BcjCoder.h"

namespace NCompress {
namespace NBcj {

Z7_COM7F_IMF(CCoder2::Init())
{
  _pc = 0;
  _state = Z7_BRANCH_CONV_ST_X86_STATE_INIT_VAL;
  return S_OK;
}

Z7_COM7F_IMF2(UInt32, CCoder2::Filter(Byte *data, UInt32 size))
{
  const UInt32 processed = (UInt32)(size_t)(_convFunc(data, size, _pc, &_state) - data);
  _pc += processed;
  return processed;
}

}}
