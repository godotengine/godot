// 7zProperties.h

#ifndef ZIP7_INC_7Z_PROPERTIES_H
#define ZIP7_INC_7Z_PROPERTIES_H

#include "../../PropID.h"

namespace NArchive {
namespace N7z {

// #define Z7_7Z_SHOW_PACK_STREAMS_SIZES // for debug

#ifdef Z7_7Z_SHOW_PACK_STREAMS_SIZES
enum
{
  kpidPackedSize0 = kpidUserDefined,
  kpidPackedSize1,
  kpidPackedSize2,
  kpidPackedSize3,
  kpidPackedSize4
};
#endif

}}

#endif
