// MyMessages.h

#ifndef ZIP7_INC_MY_MESSAGES_H
#define ZIP7_INC_MY_MESSAGES_H

#include "../../../Common/MyString.h"

void ShowErrorMessage(HWND window, LPCWSTR message);
inline void ShowErrorMessage(LPCWSTR message) { ShowErrorMessage(NULL, message); }

void ShowErrorMessageHwndRes(HWND window, UInt32 langID);
void ShowErrorMessageRes(UInt32 langID);

void ShowLastErrorMessage(HWND window = NULL);

#endif
