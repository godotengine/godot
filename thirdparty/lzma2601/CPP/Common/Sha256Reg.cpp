// Sha256Reg.cpp

#include "StdAfx.h"

#include "../../C/Sha256.h"

#include "../Common/MyBuffer2.h"
#include "../Common/MyCom.h"

#include "../7zip/Common/RegisterCodec.h"

Z7_CLASS_IMP_COM_2(
  CSha256Hasher
  , IHasher
  , ICompressSetCoderProperties
)
  CAlignedBuffer1 _buf;
public:
  Byte _mtDummy[1 << 7];

  CSha256 *Sha() { return (CSha256 *)(void *)(Byte *)_buf; }
public:
  CSha256Hasher():
    _buf(sizeof(CSha256))
  {
    Sha256_SetFunction(Sha(), 0);
    Sha256_InitState(Sha());
  }
};

Z7_COM7F_IMF2(void, CSha256Hasher::Init())
{
  Sha256_InitState(Sha());
}

Z7_COM7F_IMF2(void, CSha256Hasher::Update(const void *data, UInt32 size))
{
  Sha256_Update(Sha(), (const Byte *)data, size);
}

Z7_COM7F_IMF2(void, CSha256Hasher::Final(Byte *digest))
{
  Sha256_Final(Sha(), digest);
}


Z7_COM7F_IMF(CSha256Hasher::SetCoderProperties(const PROPID *propIDs, const PROPVARIANT *coderProps, UInt32 numProps))
{
  unsigned algo = 0;
  for (UInt32 i = 0; i < numProps; i++)
  {
    if (propIDs[i] == NCoderPropID::kDefaultProp)
    {
      const PROPVARIANT &prop = coderProps[i];
      if (prop.vt != VT_UI4)
        return E_INVALIDARG;
      if (prop.ulVal > 2)
        return E_NOTIMPL;
      algo = (unsigned)prop.ulVal;
    }
  }
  if (!Sha256_SetFunction(Sha(), algo))
    return E_NOTIMPL;
  return S_OK;
}

REGISTER_HASHER(CSha256Hasher, 0xA, "SHA256", SHA256_DIGEST_SIZE)
