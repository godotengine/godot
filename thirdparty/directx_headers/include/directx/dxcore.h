/************************************************************
*                                                           *
* Copyright (c) Microsoft Corporation.                      *
* Licensed under the MIT license.                           *
*                                                           *
************************************************************/

#ifndef _DXCOREEXTMODULE_H_
#define _DXCOREEXTMODULE_H_

#include <winapifamily.h>
#include "dxcore_interface.h"

#pragma region Application Family or OneCore Family
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM)

#if (_WIN32_WINNT >= _WIN32_WINNT_WIN10)

STDAPI
DXCoreCreateAdapterFactory(
    REFIID riid,
    _COM_Outptr_ void** ppvFactory
);

template <class T>
HRESULT
DXCoreCreateAdapterFactory(
    _COM_Outptr_ T** ppvFactory
)
{
    return DXCoreCreateAdapterFactory(IID_PPV_ARGS(ppvFactory));
}

#endif // (_WIN32_WINNT >= _WIN32_WINNT_WIN10)

#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM) */
#pragma endregion

#endif // _DXCOREEXTMODULE_H_


