// MyAesReg.cpp

#include "StdAfx.h"

#include "../Common/RegisterCodec.h"

#include "MyAes.h"

namespace NCrypto {

#ifndef Z7_SFX

#define REGISTER_AES_2(name, nameString, keySize) \
  REGISTER_FILTER_E(name, \
    CAesCbcDecoder(keySize), \
    CAesCbcEncoder(keySize), \
    0x6F00100 | ((keySize - 16) * 8) | (/* isCtr */ 0 ? 4 : 1), \
    nameString) \

#define REGISTER_AES(name, nameString) \
  /* REGISTER_AES_2(AES128 ## name, "AES128" nameString, 16) */ \
  /* REGISTER_AES_2(AES192 ## name, "AES192" nameString, 24) */ \
  REGISTER_AES_2(AES256 ## name, "AES256" nameString, 32) \

REGISTER_AES(CBC, "CBC")

#endif

}
