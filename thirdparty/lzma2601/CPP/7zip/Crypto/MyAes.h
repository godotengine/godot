// Crypto/MyAes.h

#ifndef ZIP7_INC_CRYPTO_MY_AES_H
#define ZIP7_INC_CRYPTO_MY_AES_H

#include "../../../C/Aes.h"

#include "../../Common/MyBuffer2.h"
#include "../../Common/MyCom.h"

#include "../ICoder.h"

namespace NCrypto {

#ifdef Z7_EXTRACT_ONLY
#define Z7_IFACEN_IAesCoderSetFunctions(x)
#else
#define Z7_IFACEN_IAesCoderSetFunctions(x) \
  virtual bool SetFunctions(UInt32 algo) x
#endif


class CAesCoder:
  public ICompressFilter,
  public ICryptoProperties,
 #ifndef Z7_EXTRACT_ONLY
  public ICompressSetCoderProperties,
 #endif
  public CMyUnknownImp
{
  Z7_COM_QI_BEGIN2(ICompressFilter)
  Z7_COM_QI_ENTRY(ICryptoProperties)
 #ifndef Z7_EXTRACT_ONLY
  Z7_COM_QI_ENTRY(ICompressSetCoderProperties)
 #endif
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE
  
public:
  Z7_IFACE_COM7_IMP_NONFINAL(ICompressFilter)
  Z7_IFACE_COM7_IMP(ICryptoProperties)
private:
 #ifndef Z7_EXTRACT_ONLY
  Z7_IFACE_COM7_IMP(ICompressSetCoderProperties)
 #endif

protected:
  bool _keyIsSet;
  // bool _encodeMode;
  // bool _ctrMode;
  // unsigned _offset;
  unsigned _keySize;
  unsigned _ctrPos; // we need _ctrPos here for Init() / SetInitVector()
  AES_CODE_FUNC _codeFunc;
  AES_SET_KEY_FUNC _setKeyFunc;
private:
  // UInt32 _aes[AES_NUM_IVMRK_WORDS + 3];
  CAlignedBuffer1 _aes;

  Byte _iv[AES_BLOCK_SIZE];

  // UInt32 *Aes() { return _aes + _offset; }
protected:
  UInt32 *Aes() { return (UInt32 *)(void *)(Byte *)_aes; }

 Z7_IFACE_PURE(IAesCoderSetFunctions)

public:
  CAesCoder(
      // bool encodeMode,
      unsigned keySize
      // , bool ctrMode
      );
  virtual ~CAesCoder() {}   // we need virtual destructor for derived classes
  void SetKeySize(unsigned size) { _keySize = size; }
};


#ifndef Z7_EXTRACT_ONLY
struct CAesCbcEncoder: public CAesCoder
{
  CAesCbcEncoder(unsigned keySize = 0): CAesCoder(keySize)
  {
    _setKeyFunc = Aes_SetKey_Enc;
    _codeFunc = g_AesCbc_Encode;
  }
  Z7_IFACE_IMP(IAesCoderSetFunctions)
};
#endif

struct CAesCbcDecoder: public CAesCoder
{
  CAesCbcDecoder(unsigned keySize = 0): CAesCoder(keySize)
  {
    _setKeyFunc = Aes_SetKey_Dec;
    _codeFunc = g_AesCbc_Decode;
  }
  Z7_IFACE_IMP(IAesCoderSetFunctions)
};

#ifndef Z7_SFX
struct CAesCtrCoder: public CAesCoder
{
private:
  // unsigned _ctrPos;
  // Z7_IFACE_COM7_IMP(ICompressFilter)
  // Z7_COM7F_IMP(Init())
  Z7_COM7F_IMP2(UInt32, Filter(Byte *data, UInt32 size))
public:
  CAesCtrCoder(unsigned keySize = 0): CAesCoder(keySize)
  {
    _ctrPos = 0;
    _setKeyFunc = Aes_SetKey_Enc;
    _codeFunc = g_AesCtr_Code;
  }
  Z7_IFACE_IMP(IAesCoderSetFunctions)
};
#endif

}

#endif
