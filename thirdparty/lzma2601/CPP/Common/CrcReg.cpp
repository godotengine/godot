// CrcReg.cpp

#include "StdAfx.h"

#include "../../C/7zCrc.h"
#include "../../C/CpuArch.h"

#include "../Common/MyCom.h"

#include "../7zip/Common/RegisterCodec.h"

EXTERN_C_BEGIN

EXTERN_C_END

Z7_CLASS_IMP_COM_2(
  CCrcHasher
  , IHasher
  , ICompressSetCoderProperties
)
  UInt32 _crc;
  Z7_CRC_UPDATE_FUNC _updateFunc;

  Z7_CLASS_NO_COPY(CCrcHasher)

  bool SetFunctions(UInt32 tSize);
public:
  Byte _mtDummy[1 << 7];  // it's public to eliminate clang warning: unused private field

  CCrcHasher(): _crc(CRC_INIT_VAL) { SetFunctions(0); }
};

bool CCrcHasher::SetFunctions(UInt32 tSize)
{
  const Z7_CRC_UPDATE_FUNC f = z7_GetFunc_CrcUpdate(tSize);
  if (!f)
  {
    _updateFunc = CrcUpdate;
    return false;
  }
  _updateFunc = f;
  return true;
}

Z7_COM7F_IMF(CCrcHasher::SetCoderProperties(const PROPID *propIDs, const PROPVARIANT *coderProps, UInt32 numProps))
{
  for (UInt32 i = 0; i < numProps; i++)
  {
    if (propIDs[i] == NCoderPropID::kDefaultProp)
    {
      const PROPVARIANT &prop = coderProps[i];
      if (prop.vt != VT_UI4)
        return E_INVALIDARG;
      if (!SetFunctions(prop.ulVal))
        return E_NOTIMPL;
    }
  }
  return S_OK;
}

Z7_COM7F_IMF2(void, CCrcHasher::Init())
{
  _crc = CRC_INIT_VAL;
}

Z7_COM7F_IMF2(void, CCrcHasher::Update(const void *data, UInt32 size))
{
  _crc = _updateFunc(_crc, data, size);
}

Z7_COM7F_IMF2(void, CCrcHasher::Final(Byte *digest))
{
  const UInt32 val = CRC_GET_DIGEST(_crc);
  SetUi32(digest, val)
}

REGISTER_HASHER(CCrcHasher, 0x1, "CRC32", 4)
