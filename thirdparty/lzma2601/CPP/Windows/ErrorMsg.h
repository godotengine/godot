// Windows/ErrorMsg.h

#ifndef ZIP7_INC_WINDOWS_ERROR_MSG_H
#define ZIP7_INC_WINDOWS_ERROR_MSG_H

#include "../Common/MyString.h"

namespace NWindows {
namespace NError {

UString MyFormatMessage(DWORD errorCode);
inline UString MyFormatMessage(HRESULT errorCode) { return MyFormatMessage((DWORD)errorCode); }

}}

#endif
