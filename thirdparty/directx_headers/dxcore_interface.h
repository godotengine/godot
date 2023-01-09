//
// DXCore Interface
// Copyright (C) Microsoft Corporation.
// Licensed under the MIT license.
//

#ifndef __dxcore_interface_h__
#define __dxcore_interface_h__

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#include <stdint.h>

#ifdef __cplusplus

#define _FACDXCORE    0x880
#define MAKE_DXCORE_HRESULT( code )     MAKE_HRESULT( 1, _FACDXCORE, code )

enum class DXCoreAdapterProperty : uint32_t
{
    InstanceLuid = 0,
    DriverVersion = 1,
    DriverDescription = 2,
    HardwareID = 3, // Use HardwareIDParts instead, if available.
    KmdModelVersion = 4,
    ComputePreemptionGranularity = 5,
    GraphicsPreemptionGranularity = 6,
    DedicatedAdapterMemory = 7,
    DedicatedSystemMemory = 8,
    SharedSystemMemory = 9,
    AcgCompatible = 10,
    IsHardware = 11,
    IsIntegrated = 12,
    IsDetachable = 13,
    HardwareIDParts = 14
};

enum class DXCoreAdapterState : uint32_t
{
    IsDriverUpdateInProgress = 0,
    AdapterMemoryBudget = 1
};

enum class DXCoreSegmentGroup : uint32_t
{
    Local = 0,
    NonLocal = 1
};

enum class DXCoreNotificationType : uint32_t
{
    AdapterListStale = 0,
    AdapterNoLongerValid = 1,
    AdapterBudgetChange = 2,
    AdapterHardwareContentProtectionTeardown = 3
};

enum class DXCoreAdapterPreference : uint32_t
{
    Hardware = 0,
    MinimumPower = 1,
    HighPerformance = 2
};

struct DXCoreHardwareID
{
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t subSysID;
    uint32_t revision;
};

struct DXCoreHardwareIDParts
{
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t subSystemID;
    uint32_t subVendorID;
    uint32_t revisionID;
};

struct DXCoreAdapterMemoryBudgetNodeSegmentGroup
{
    uint32_t nodeIndex;
    DXCoreSegmentGroup segmentGroup;
};

struct DXCoreAdapterMemoryBudget
{
    uint64_t budget;
    uint64_t currentUsage;
    uint64_t availableForReservation;
    uint64_t currentReservation;
};

typedef void (STDMETHODCALLTYPE *PFN_DXCORE_NOTIFICATION_CALLBACK)(
    DXCoreNotificationType notificationType,
    _In_ IUnknown *object,
    _In_opt_ void *context);

static_assert(sizeof(bool) == 1, "bool assumed as one byte");

DEFINE_GUID(IID_IDXCoreAdapterFactory, 0x78ee5945, 0xc36e, 0x4b13, 0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06);
DEFINE_GUID(IID_IDXCoreAdapterList, 0x526c7776, 0x40e9, 0x459b, 0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28);
DEFINE_GUID(IID_IDXCoreAdapter, 0xf0db4c7f, 0xfe5a, 0x42a2, 0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D11_GRAPHICS, 0x8c47866b, 0x7583, 0x450d, 0xf0, 0xf0, 0x6b, 0xad, 0xa8, 0x95, 0xaf, 0x4b);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS, 0x0c9ece4d, 0x2f6e, 0x4f01, 0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, 0x248e2800, 0xa793, 0x4724, 0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90);

/* interface IDXCoreAdapter */
MIDL_INTERFACE("f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e")
IDXCoreAdapter : public IUnknown
{
public:
    virtual bool STDMETHODCALLTYPE IsValid() = 0;

    virtual bool STDMETHODCALLTYPE IsAttributeSupported( 
        REFGUID attributeGUID) = 0;

    virtual bool STDMETHODCALLTYPE IsPropertySupported( 
        DXCoreAdapterProperty property) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetProperty( 
        DXCoreAdapterProperty property,
        size_t bufferSize,
        _Out_writes_bytes_(bufferSize)  void *propertyData) = 0;

    template <class T>
    HRESULT GetProperty( 
        DXCoreAdapterProperty property,
        _Out_writes_bytes_(sizeof(T))  T *propertyData)
    {
        return GetProperty(property,
                           sizeof(T),
                           (void*)propertyData);
    }

    virtual HRESULT STDMETHODCALLTYPE GetPropertySize( 
        DXCoreAdapterProperty property,
        _Out_ size_t *bufferSize) = 0;

    virtual bool STDMETHODCALLTYPE IsQueryStateSupported( 
        DXCoreAdapterState property) = 0;

    virtual HRESULT STDMETHODCALLTYPE QueryState( 
        DXCoreAdapterState state,
        size_t inputStateDetailsSize,
        _In_reads_bytes_opt_(inputStateDetailsSize) const void *inputStateDetails,
        size_t outputBufferSize,
        _Out_writes_bytes_(outputBufferSize) void *outputBuffer) = 0;

    template <class T1, class T2>
    HRESULT QueryState( 
        DXCoreAdapterState state,
        _In_reads_bytes_opt_(sizeof(T1)) const T1 *inputStateDetails,
        _Out_writes_bytes_(sizeof(T2)) T2 *outputBuffer)
    {
        return QueryState(state,
                          sizeof(T1),
                          (const void*)inputStateDetails,
                          sizeof(T2),
                          (void*)outputBuffer);
    }

    template <class T>
    HRESULT QueryState( 
        DXCoreAdapterState state,
        _Out_writes_bytes_(sizeof(T)) T *outputBuffer)
    {
        return QueryState(state,
                          0,
                          nullptr,
                          sizeof(T),
                          (void*)outputBuffer);
    }

    virtual bool STDMETHODCALLTYPE IsSetStateSupported( 
        DXCoreAdapterState property) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetState( 
        DXCoreAdapterState state,
        size_t inputStateDetailsSize,
        _In_reads_bytes_opt_(inputStateDetailsSize) const void *inputStateDetails,
        size_t inputDataSize,
        _In_reads_bytes_(inputDataSize) const void *inputData) = 0;

    template <class T1, class T2>
    HRESULT SetState( 
        DXCoreAdapterState state,
        const T1 *inputStateDetails,
        const T2 *inputData)
    {
        return SetState(state,
                        sizeof(T1),
                        (const void*)inputStateDetails,
                        sizeof(T2),
                        (const void*)inputData);
    }

    virtual HRESULT STDMETHODCALLTYPE GetFactory(
        REFIID riid,
        _COM_Outptr_ void** ppvFactory
    ) = 0;

    template <class T>
    HRESULT GetFactory(
        _COM_Outptr_ T** ppvFactory
    )
    {
        return GetFactory(IID_PPV_ARGS(ppvFactory));
    }
};

/* interface IDXCoreAdapterList */
MIDL_INTERFACE("526c7776-40e9-459b-b711-f32ad76dfc28")
IDXCoreAdapterList : public IUnknown
{
public:
    virtual HRESULT STDMETHODCALLTYPE GetAdapter( 
        uint32_t index,
        REFIID riid,
        _COM_Outptr_ void **ppvAdapter) = 0;

    template<class T>
    HRESULT STDMETHODCALLTYPE GetAdapter( 
        uint32_t index,
        _COM_Outptr_ T **ppvAdapter)
    {
        return GetAdapter(index,
                          IID_PPV_ARGS(ppvAdapter));
    }

    virtual uint32_t STDMETHODCALLTYPE GetAdapterCount() = 0;

    virtual bool STDMETHODCALLTYPE IsStale() = 0;

    virtual HRESULT STDMETHODCALLTYPE GetFactory(
        REFIID riid,
        _COM_Outptr_ void** ppvFactory
    ) = 0;

    template <class T>
    HRESULT GetFactory(
        _COM_Outptr_ T** ppvFactory
    )
    {
        return GetFactory(IID_PPV_ARGS(ppvFactory));
    }

    virtual HRESULT STDMETHODCALLTYPE Sort(
        uint32_t numPreferences,
        _In_reads_(numPreferences) const DXCoreAdapterPreference* preferences) = 0;

    virtual bool STDMETHODCALLTYPE IsAdapterPreferenceSupported( 
        DXCoreAdapterPreference preference) = 0;
};

/* interface IDXCoreAdapterFactory */
MIDL_INTERFACE("78ee5945-c36e-4b13-a669-005dd11c0f06")
IDXCoreAdapterFactory : public IUnknown
{
public:

    virtual HRESULT STDMETHODCALLTYPE CreateAdapterList( 
        uint32_t numAttributes,
        _In_reads_(numAttributes) const GUID *filterAttributes,
        REFIID riid,
        _COM_Outptr_ void **ppvAdapterList) = 0;

    template<class T>
    HRESULT STDMETHODCALLTYPE CreateAdapterList( 
        uint32_t numAttributes,
        _In_reads_(numAttributes) const GUID *filterAttributes,
        _COM_Outptr_ T **ppvAdapterList)
    {
        return CreateAdapterList(numAttributes,
                                 filterAttributes,
                                 IID_PPV_ARGS(ppvAdapterList));
    }

    virtual HRESULT STDMETHODCALLTYPE GetAdapterByLuid( 
        const LUID &adapterLUID,
        REFIID riid,
        _COM_Outptr_ void **ppvAdapter) = 0;

    template<class T>
    HRESULT STDMETHODCALLTYPE GetAdapterByLuid( 
        const LUID &adapterLUID,
        _COM_Outptr_ T **ppvAdapter)
    {
        return GetAdapterByLuid(adapterLUID,
                                IID_PPV_ARGS(ppvAdapter));
    }

    virtual bool STDMETHODCALLTYPE IsNotificationTypeSupported( 
        DXCoreNotificationType notificationType) = 0;

    virtual HRESULT STDMETHODCALLTYPE RegisterEventNotification( 
        _In_ IUnknown *dxCoreObject,
        DXCoreNotificationType notificationType,
        _In_ PFN_DXCORE_NOTIFICATION_CALLBACK callbackFunction,
        _In_opt_ void *callbackContext,
        _Out_ uint32_t *eventCookie) = 0;

    virtual HRESULT STDMETHODCALLTYPE UnregisterEventNotification( 
        uint32_t eventCookie) = 0;
};

#endif // __cplusplus

#endif // __dxcore_interface_h__


