// ByteSwap.cpp

#include "StdAfx.h"

#include "../../../C/SwapBytes.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

#include "../Common/RegisterCodec.h"

namespace NCompress {
namespace NByteSwap {

Z7_CLASS_IMP_COM_1(CByteSwap2, ICompressFilter) };
Z7_CLASS_IMP_COM_1(CByteSwap4, ICompressFilter) };

Z7_COM7F_IMF(CByteSwap2::Init()) { return S_OK; }

Z7_COM7F_IMF2(UInt32, CByteSwap2::Filter(Byte *data, UInt32 size))
{
  const UInt32 kMask = 2 - 1;
  size &= ~kMask;
  /*
  if ((unsigned)(ptrdiff_t)data & kMask)
  {
    if (size == 0)
      return 0;
    const Byte *end = data + (size_t)size;
    do
    {
      const Byte b0 = data[0];
      data[0] = data[1];
      data[1] = b0;
      data += kStep;
    }
    while (data != end);
  }
  else
  */
  z7_SwapBytes2((UInt16 *)(void *)data, size >> 1);
  return size;
}


Z7_COM7F_IMF(CByteSwap4::Init()) { return S_OK; }

Z7_COM7F_IMF2(UInt32, CByteSwap4::Filter(Byte *data, UInt32 size))
{
  const UInt32 kMask = 4 - 1;
  size &= ~kMask;
  /*
  if ((unsigned)(ptrdiff_t)data & kMask)
  {
    if (size == 0)
      return 0;
    const Byte *end = data + (size_t)size;
    do
    {
      const Byte b0 = data[0];
      const Byte b1 = data[1];
      data[0] = data[3];
      data[1] = data[2];
      data[2] = b1;
      data[3] = b0;
      data += kStep;
    }
    while (data != end);
  }
  else
  */
  z7_SwapBytes4((UInt32 *)(void *)data, size >> 2);
  return size;
}

static struct C_SwapBytesPrepare { C_SwapBytesPrepare() { z7_SwapBytesPrepare(); } } g_SwapBytesPrepare;


REGISTER_FILTER_CREATE(CreateFilter2, CByteSwap2())
REGISTER_FILTER_CREATE(CreateFilter4, CByteSwap4())

REGISTER_CODECS_VAR
{
  REGISTER_FILTER_ITEM(CreateFilter2, CreateFilter2, 0x20302, "Swap2"),
  REGISTER_FILTER_ITEM(CreateFilter4, CreateFilter4, 0x20304, "Swap4"),
};

REGISTER_CODECS(ByteSwap)

}}
