// Windows/ResourceString.h

#ifndef ZIP7_INC_WINDOWS_RESOURCE_STRING_H
#define ZIP7_INC_WINDOWS_RESOURCE_STRING_H

#include "../Common/MyString.h"
#include "../Common/MyWindows.h"

namespace NWindows {

UString MyLoadString(UINT resourceID);
void MyLoadString(HINSTANCE hInstance, UINT resourceID, UString &dest);
void MyLoadString(UINT resourceID, UString &dest);

}

#endif
