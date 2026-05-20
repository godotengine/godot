// Windows/SystemInfo.h

#ifndef ZIP7_INC_WINDOWS_SYSTEM_INFO_H
#define ZIP7_INC_WINDOWS_SYSTEM_INFO_H

#include "../Common/MyString.h"


void GetCpuName_MultiLine(AString &s, AString &registers);

void GetOsInfoText(AString &sRes);
void GetSystemInfoText(AString &s);
void PrintSize_KMGT_Or_Hex(AString &s, UInt64 v);
void Add_LargePages_String(AString &s);

void GetCompiler(AString &s);
void GetVirtCpuid(AString &s);

#endif
