// PpmdRegister.cpp

#include "StdAfx.h"

#include "../Common/RegisterCodec.h"

#include "PpmdDecoder.h"

#ifndef Z7_EXTRACT_ONLY
#include "PpmdEncoder.h"
#endif

namespace NCompress {
namespace NPpmd {

REGISTER_CODEC_E(PPMD,
    CDecoder(),
    CEncoder(),
    0x30401,
    "PPMD")

}}
