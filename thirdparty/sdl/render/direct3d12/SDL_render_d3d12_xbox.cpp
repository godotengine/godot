/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "../../SDL_internal.h"
#if defined(SDL_VIDEO_RENDER_D3D12) && (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
#include "SDL_render_d3d12_xbox.h"
#include "../../core/windows/SDL_windows.h"
#include <XGameRuntime.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define SDL_COMPOSE_ERROR(str) __FUNCTION__ ", " str
#else
#define SDL_COMPOSE_ERROR(str) SDL_STRINGIFY_ARG(__FUNCTION__) ", " str
#endif

static const GUID SDL_IID_ID3D12Device1 = { 0x77acce80, 0x638e, 0x4e65, { 0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e } };
static const GUID SDL_IID_ID3D12Resource = { 0x696442be, 0xa72e, 0x4059, { 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad } };
static const GUID SDL_IID_IDXGIDevice1 = { 0x77db970f, 0x6276, 0x48ba, { 0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c } };

extern "C"
HRESULT D3D12_XBOX_CreateDevice(ID3D12Device **device, bool createDebug)
{
    HRESULT result;
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params;
    IDXGIDevice1 *dxgiDevice;
    IDXGIAdapter *dxgiAdapter;
    IDXGIOutput *dxgiOutput;
    SDL_zero(params);

    params.Version = D3D12_SDK_VERSION;
    params.ProcessDebugFlags = createDebug ? D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED : D3D12XBOX_PROCESS_DEBUG_FLAG_NONE;
    params.GraphicsCommandQueueRingSizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.GraphicsScratchMemorySizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.ComputeScratchMemorySizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;

    result = D3D12XboxCreateDevice(NULL, &params, SDL_IID_ID3D12Device1, (void **) device);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("[xbox] D3D12XboxCreateDevice"), result);
        goto done;
    }

    result = (*device)->QueryInterface(SDL_IID_IDXGIDevice1, (void **) &dxgiDevice);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("[xbox] ID3D12Device to IDXGIDevice1"), result);
        goto done;
    }

    result = dxgiDevice->GetAdapter(&dxgiAdapter);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("[xbox] dxgiDevice->GetAdapter"), result);
        goto done;
    }

    result = dxgiAdapter->EnumOutputs(0, &dxgiOutput);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("[xbox] dxgiAdapter->EnumOutputs"), result);
        goto done;
    }

    // Set frame interval
    result = (*device)->SetFrameIntervalX(dxgiOutput, D3D12XBOX_FRAME_INTERVAL_60_HZ, 1, D3D12XBOX_FRAME_INTERVAL_FLAG_NONE);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("[xbox] SetFrameIntervalX"), result);
        goto done;
    }

    result = (*device)->ScheduleFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, 0, NULL, D3D12XBOX_SCHEDULE_FRAME_EVENT_FLAG_NONE);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("[xbox] ScheduleFrameEventX"), result);
        goto done;
    }

done:
    return result;
}

extern "C"
HRESULT D3D12_XBOX_CreateBackBufferTarget(ID3D12Device1 *device, int width, int height, void **resource)
{

    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_RESOURCE_DESC resourceDesc;

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    SDL_zero(resourceDesc);
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = width;
    resourceDesc.Height = height;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    return device->CreateCommittedResource(&heapProps,
        D3D12_HEAP_FLAG_ALLOW_DISPLAY,
        &resourceDesc,
        D3D12_RESOURCE_STATE_PRESENT,
        NULL,
        SDL_IID_ID3D12Resource,
        resource
        );
}

extern "C"
HRESULT D3D12_XBOX_StartFrame(ID3D12Device1 *device, UINT64 *outToken)
{
    *outToken = D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL;
    return device->WaitFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, INFINITE, NULL, D3D12XBOX_WAIT_FRAME_EVENT_FLAG_NONE, outToken);
}

extern "C"
HRESULT D3D12_XBOX_PresentFrame(ID3D12CommandQueue *commandQueue, UINT64 token, ID3D12Resource *renderTarget)
{
    D3D12XBOX_PRESENT_PLANE_PARAMETERS planeParameters;
    SDL_zero(planeParameters);
    planeParameters.Token = token;
    planeParameters.ResourceCount = 1;
    planeParameters.ppResources = &renderTarget;
    return commandQueue->PresentX(1, &planeParameters, NULL);
}

extern "C"
void D3D12_XBOX_GetResolution(Uint32 *width, Uint32 *height)
{
    switch (XSystemGetDeviceType()) {
    case XSystemDeviceType::XboxScarlettLockhart:
        *width = 2560;
        *height = 1440;
        break;

    case XSystemDeviceType::XboxOneX:
    case XSystemDeviceType::XboxScarlettAnaconda:
    case XSystemDeviceType::XboxOneXDevkit:
    case XSystemDeviceType::XboxScarlettDevkit:
        *width = 3840;
        *height = 2160;
        break;

    default:
        *width = 1920;
        *height = 1080;
        break;
    }
}

#endif // SDL_VIDEO_RENDER_D3D12 && (SDL_PLATFORM_XBOXONE || SDL_PLATFORM_XBOXSERIES)
