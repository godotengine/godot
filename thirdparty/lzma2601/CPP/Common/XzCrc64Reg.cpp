// XzCrc64Reg.cpp

#include "StdAfx.h"

#include "../../C/CpuArch.h"
#include "../../C/XzCrc64.h"

#include "../Common/MyCom.h"

#include "../7zip/Common/RegisterCodec.h"

Z7_CLASS_IMP_COM_1(
  CXzCrc64Hasher
  , IHasher
)
  UInt64 _crc;
public:
  Byte _mtDummy[1 << 7];  // it's public to eliminate clang warning: unused private field

  CXzCrc64Hasher(): _crc(CRC64_INIT_VAL) {}
};

Z7_COM7F_IMF2(void, CXzCrc64Hasher::Init())
{
  _crc = CRC64_INIT_VAL;
}

Z7_COM7F_IMF2(void, CXzCrc64Hasher::Update(const void *data, UInt32 size))
{
  _crc = Crc64Update(_crc, data, size);
}

Z7_COM7F_IMF2(void, CXzCrc64Hasher::Final(Byte *digest))
{
  const UInt64 val = CRC64_GET_DIGEST(_crc);
  SetUi64(digest, val)
}

REGISTER_HASHER(CXzCrc64Hasher, 0x4, "CRC64", 8)
