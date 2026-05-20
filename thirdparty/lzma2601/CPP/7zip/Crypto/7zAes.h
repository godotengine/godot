// 7zAes.h

#ifndef ZIP7_INC_CRYPTO_7Z_AES_H
#define ZIP7_INC_CRYPTO_7Z_AES_H

#include "../../Common/MyBuffer.h"
#include "../../Common/MyCom.h"
#include "../../Common/MyVector.h"

#include "../ICoder.h"
#include "../IPassword.h"

namespace NCrypto {
namespace N7z {

const unsigned kKeySize = 32;
const unsigned kSaltSizeMax = 16;
const unsigned kIvSizeMax = 16; // AES_BLOCK_SIZE;

class CKeyInfo
{
public:
  unsigned NumCyclesPower;
  unsigned SaltSize;
  Byte Salt[kSaltSizeMax];
  CByteBuffer Password;
  Byte Key[kKeySize];

  bool IsEqualTo(const CKeyInfo &a) const;
  void CalcKey();

  CKeyInfo() { ClearProps(); }
  void ClearProps()
  {
    NumCyclesPower = 0;
    SaltSize = 0;
    for (unsigned i = 0; i < sizeof(Salt); i++)
      Salt[i] = 0;
  }

  void Wipe()
  {
    Password.Wipe();
    NumCyclesPower = 0;
    SaltSize = 0;
    Z7_memset_0_ARRAY(Salt);
    Z7_memset_0_ARRAY(Key);
  }

#ifdef Z7_CPP_IS_SUPPORTED_default
  CKeyInfo(const CKeyInfo &) = default;
#endif
  ~CKeyInfo() { Wipe(); }
};

class CKeyInfoCache
{
  unsigned Size;
  CObjectVector<CKeyInfo> Keys;
public:
  CKeyInfoCache(unsigned size): Size(size) {}
  bool GetKey(CKeyInfo &key);
  void Add(const CKeyInfo &key);
  void FindAndAdd(const CKeyInfo &key);
};

class CBase
{
  CKeyInfoCache _cachedKeys;
protected:
  CKeyInfo _key;
  Byte _iv[kIvSizeMax];
  unsigned _ivSize;
  
  void PrepareKey();
  CBase();
};

class CBaseCoder:
  public ICompressFilter,
  public ICryptoSetPassword,
  public CMyUnknownImp,
  public CBase
{
  Z7_IFACE_COM7_IMP(ICompressFilter)
  Z7_IFACE_COM7_IMP(ICryptoSetPassword)
protected:
  virtual ~CBaseCoder() {}
  CMyComPtr<ICompressFilter> _aesFilter;
};

#ifndef Z7_EXTRACT_ONLY

class CEncoder Z7_final:
  public CBaseCoder,
  public ICompressWriteCoderProperties,
  // public ICryptoResetSalt,
  public ICryptoResetInitVector
{
  Z7_COM_UNKNOWN_IMP_4(
      ICompressFilter,
      ICryptoSetPassword,
      ICompressWriteCoderProperties,
      // ICryptoResetSalt,
      ICryptoResetInitVector)
  Z7_IFACE_COM7_IMP(ICompressWriteCoderProperties)
  // Z7_IFACE_COM7_IMP(ICryptoResetSalt)
  Z7_IFACE_COM7_IMP(ICryptoResetInitVector)
public:
  CEncoder();
};

#endif

class CDecoder Z7_final:
  public CBaseCoder,
  public ICompressSetDecoderProperties2
{
  Z7_COM_UNKNOWN_IMP_3(
      ICompressFilter,
      ICryptoSetPassword,
      ICompressSetDecoderProperties2)
  Z7_IFACE_COM7_IMP(ICompressSetDecoderProperties2)
public:
  CDecoder();
};

}}

#endif
