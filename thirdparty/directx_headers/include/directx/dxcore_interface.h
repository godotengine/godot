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
    HardwareIDParts = 14,
    PhysicalAdapterCount = 15,
    AdapterEngineCount = 16,
    AdapterEngineName = 17,
};

enum class DXCoreAdapterState : uint32_t
{
    IsDriverUpdateInProgress = 0,
    AdapterMemoryBudget = 1,
    AdapterMemoryUsageBytes = 2,
    AdapterMemoryUsageByProcessBytes = 3,
    AdapterEngineRunningTimeMicroseconds = 4,
    AdapterEngineRunningTimeByProcessMicroseconds = 5,
    AdapterTemperatureCelsius = 6,
    AdapterInUseProcessCount = 7,
    AdapterInUseProcessSet = 8,
    AdapterEngineFrequencyHertz = 9,
    AdapterMemoryFrequencyHertz = 10
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

enum class DXCoreWorkload : uint32_t
{
    Graphics = 0,
    Compute = 1,
    Media = 2,
    MachineLearning = 3,
};

enum class DXCoreRuntimeFilterFlags : uint32_t
{
    None = 0x0,
    D3D11 = 0x1,
    D3D12 = 0x2
};

DEFINE_ENUM_FLAG_OPERATORS(DXCoreRuntimeFilterFlags)

enum class DXCoreHardwareTypeFilterFlags : uint32_t
{
    None = 0x0,
    GPU = 0x1,
    ComputeAccelerator = 0x2,
    NPU = 0x4,
    MediaAccelerator = 0x8
};

DEFINE_ENUM_FLAG_OPERATORS(DXCoreHardwareTypeFilterFlags)

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

struct DXCoreAdapterEngineIndex
{
    uint32_t physicalAdapterIndex;
    uint32_t engineIndex;
};

struct DXCoreEngineQueryInput
{
    DXCoreAdapterEngineIndex adapterEngineIndex;
    uint32_t processId;
};

struct DXCoreEngineQueryOutput
{
    uint64_t runningTime;
    bool processQuerySucceeded;
};

enum class DXCoreMemoryType : uint32_t
{
    Dedicated = 0,
    Shared = 1
};

struct DXCoreMemoryUsage
{
    uint64_t committed;
    uint64_t resident;
};

struct DXCoreMemoryQueryInput
{
    uint32_t physicalAdapterIndex;
    DXCoreMemoryType memoryType;
};

struct DXCoreProcessMemoryQueryInput
{
    uint32_t physicalAdapterIndex;
    DXCoreMemoryType memoryType;
    uint32_t processId;
};

struct DXCoreProcessMemoryQueryOutput
{
    DXCoreMemoryUsage memoryUsage;
    bool processQuerySucceeded;
};

struct DXCoreAdapterProcessSetQueryInput
{
    uint32_t arraySize;
    _Field_size_(arraySize) uint32_t* processIds;
};

struct DXCoreAdapterProcessSetQueryOutput
{
    uint32_t processesWritten;
    uint32_t processesTotal;
};

struct DXCoreEngineNamePropertyInput
{
    DXCoreAdapterEngineIndex adapterEngineIndex;
    uint32_t engineNameLength;
    _Field_size_(engineNameLength) wchar_t *engineName;
};

struct DXCoreEngineNamePropertyOutput
{
    uint32_t engineNameLength;
};

struct DXCoreFrequencyQueryOutput
{
    uint64_t frequency;
    uint64_t maxFrequency;
    uint64_t maxOverclockedFrequency;
};

typedef void (STDMETHODCALLTYPE *PFN_DXCORE_NOTIFICATION_CALLBACK)(
    DXCoreNotificationType notificationType,
    _In_ IUnknown *object,
    _In_opt_ void *context);

static_assert(sizeof(bool) == 1, "bool assumed as one byte");

DEFINE_GUID(IID_IDXCoreAdapterFactory, 0x78ee5945, 0xc36e, 0x4b13, 0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06);
DEFINE_GUID(IID_IDXCoreAdapterFactory1, 0xd5682e19, 0x6d21, 0x401c, 0x82, 0x7a, 0x9a, 0x51, 0xa4, 0xea, 0x35, 0xd7);
DEFINE_GUID(IID_IDXCoreAdapterList, 0x526c7776, 0x40e9, 0x459b, 0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28);
DEFINE_GUID(IID_IDXCoreAdapter, 0xf0db4c7f, 0xfe5a, 0x42a2, 0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e);
DEFINE_GUID(IID_IDXCoreAdapter1, 0xa0783366, 0xcfa3, 0x43be, 0x9d, 0x79, 0x55, 0xb2, 0xda, 0x97, 0xc6, 0x3c);

DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D11_GRAPHICS, 0x8c47866b, 0x7583, 0x450d, 0xf0, 0xf0, 0x6b, 0xad, 0xa8, 0x95, 0xaf, 0x4b);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS, 0x0c9ece4d, 0x2f6e, 0x4f01, 0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, 0x248e2800, 0xa793, 0x4724, 0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, 0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_MEDIA, 0x8eb2c848, 0x82f6, 0x4b49, 0xaa, 0x87, 0xae, 0xcf, 0xcf, 0x1, 0x74, 0xc6);

DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU, 0xb69eb219, 0x3ded, 0x4464, 0x97, 0x9f, 0xa0, 0xb, 0xd4, 0x68, 0x70, 0x6);
DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_COMPUTE_ACCELERATOR, 0xe0b195da, 0x58ef, 0x4a22, 0x90, 0xf1, 0x1f, 0x28, 0x16, 0x9c, 0xab, 0x8d);
DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, 0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed);
DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_MEDIA_ACCELERATOR, 0x66bdb96a, 0x50b, 0x44c7, 0xa4, 0xfd, 0xd1, 0x44, 0xce, 0xa, 0xb4, 0x43);

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

/* interface IDXCoreAdapter1 */
MIDL_INTERFACE("a0783366-cfa3-43be-9d79-55b2da97c63c")
IDXCoreAdapter1 : public IDXCoreAdapter
{
public:
    virtual HRESULT STDMETHODCALLTYPE GetPropertyWithInput(
        DXCoreAdapterProperty property,
        size_t inputPropertyDetailsSize,
        _In_reads_bytes_opt_(inputPropertyDetailsSize) const void *inputPropertyDetails,
        size_t outputBufferSize,
        _Out_writes_bytes_(outputBufferSize) void *outputBuffer) = 0;

    template <class T1, class T2>
    HRESULT GetPropertyWithInput(
            DXCoreAdapterProperty property,
            _In_reads_bytes_opt_(sizeof(T1)) const T1 *inputPropertyDetails,
            _Out_writes_bytes_(sizeof(T2)) T2 *outputBuffer)
        {
            return GetPropertyWithInput(property,
                                        sizeof(T1),
                                        (const void*)inputPropertyDetails,
                                        sizeof(T2),
                                        (void*)outputBuffer);
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

/* interface IDXCoreAdapterFactory1 */
MIDL_INTERFACE("d5682e19-6d21-401c-827a-9a51a4ea35d7")
IDXCoreAdapterFactory1 : public IDXCoreAdapterFactory
{
public:
    virtual HRESULT STDMETHODCALLTYPE CreateAdapterListByWorkload(
        DXCoreWorkload workload,
        DXCoreRuntimeFilterFlags runtimeFilter,
        DXCoreHardwareTypeFilterFlags hardwareTypeFilter,
        REFIID riid,
        _COM_Outptr_ void **ppvAdapterList) = 0;

    template<class T>
    HRESULT STDMETHODCALLTYPE CreateAdapterListByWorkload(
        DXCoreWorkload workload,
        DXCoreRuntimeFilterFlags runtimeFilter,
        DXCoreHardwareTypeFilterFlags hardwareTypeFilter,
        _COM_Outptr_ T **ppvAdapterList)
    {
        return CreateAdapterListByWorkload(workload,
                                           runtimeFilter,
                                           hardwareTypeFilter,
                                           IID_PPV_ARGS(ppvAdapterList));
    }
};

#endif // __cplusplus

#endif // __dxcore_interface_h__

