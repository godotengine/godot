// UserInputUtils.h

#ifndef ZIP7_INC_USER_INPUT_UTILS_H
#define ZIP7_INC_USER_INPUT_UTILS_H

#include "../../../Common/StdOutStream.h"

namespace NUserAnswerMode {

enum EEnum
{
  kYes,
  kNo,
  kYesAll,
  kNoAll,
  kAutoRenameAll,
  kQuit,
  kEof,
  kError
};
}

NUserAnswerMode::EEnum ScanUserYesNoAllQuit(CStdOutStream *outStream);
// bool GetPassword(CStdOutStream *outStream, UString &psw);
HRESULT GetPassword_HRESULT(CStdOutStream *outStream, UString &psw);

#endif
