/* DllSecur.h -- DLL loading for security
2023-03-03 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_DLL_SECUR_H
#define ZIP7_INC_DLL_SECUR_H

#include "7zTypes.h"

EXTERN_C_BEGIN

#ifdef _WIN32

void My_SetDefaultDllDirectories(void);
void LoadSecurityDlls(void);

#endif

EXTERN_C_END

#endif
