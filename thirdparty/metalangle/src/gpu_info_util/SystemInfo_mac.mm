//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SystemInfo_mac.cpp: implementation of the Mac-specific parts of SystemInfo.h

#if __has_include(<Cocoa/Cocoa.h>)

#    include "gpu_info_util/SystemInfo_internal.h"

#    import <Cocoa/Cocoa.h>
#    import <IOKit/IOKitLib.h>

namespace angle
{

namespace
{

using PlatformDisplayID = uint32_t;

constexpr CGLRendererProperty kCGLRPRegistryIDLow  = static_cast<CGLRendererProperty>(140);
constexpr CGLRendererProperty kCGLRPRegistryIDHigh = static_cast<CGLRendererProperty>(141);

// Code from WebKit to get the active GPU's ID given a display ID.
uint64_t GetGpuIDFromDisplayID(PlatformDisplayID displayID)
{
    GLuint displayMask              = CGDisplayIDToOpenGLDisplayMask(displayID);
    GLint numRenderers              = 0;
    CGLRendererInfoObj rendererInfo = nullptr;
    CGLError error = CGLQueryRendererInfo(displayMask, &rendererInfo, &numRenderers);
    if (!numRenderers || !rendererInfo || error != kCGLNoError)
        return 0;

    // The 0th renderer should not be the software renderer.
    GLint isAccelerated;
    error = CGLDescribeRenderer(rendererInfo, 0, kCGLRPAccelerated, &isAccelerated);
    if (!isAccelerated || error != kCGLNoError)
    {
        CGLDestroyRendererInfo(rendererInfo);
        return 0;
    }

    GLint gpuIDLow  = 0;
    GLint gpuIDHigh = 0;

    error = CGLDescribeRenderer(rendererInfo, 0, kCGLRPRegistryIDLow, &gpuIDLow);

    if (error != kCGLNoError || gpuIDLow < 0)
    {
        CGLDestroyRendererInfo(rendererInfo);
        return 0;
    }

    error = CGLDescribeRenderer(rendererInfo, 0, kCGLRPRegistryIDHigh, &gpuIDHigh);
    if (error != kCGLNoError || gpuIDHigh < 0)
    {
        CGLDestroyRendererInfo(rendererInfo);
        return 0;
    }

    CGLDestroyRendererInfo(rendererInfo);
    return static_cast<uint64_t>(gpuIDHigh) << 32 | gpuIDLow;
}

std::string GetMachineModel()
{
    io_service_t platformExpert = IOServiceGetMatchingService(
        kIOMasterPortDefault, IOServiceMatching("IOPlatformExpertDevice"));

    if (platformExpert == IO_OBJECT_NULL)
    {
        return "";
    }

    CFDataRef modelData = static_cast<CFDataRef>(
        IORegistryEntryCreateCFProperty(platformExpert, CFSTR("model"), kCFAllocatorDefault, 0));
    if (modelData == nullptr)
    {
        IOObjectRelease(platformExpert);
        return "";
    }

    std::string result = reinterpret_cast<const char *>(CFDataGetBytePtr(modelData));

    IOObjectRelease(platformExpert);
    CFRelease(modelData);

    return result;
}

// Extracts one integer property from a registry entry.
bool GetEntryProperty(io_registry_entry_t entry, CFStringRef name, uint32_t *value)
{
    *value = 0;

    CFDataRef data = static_cast<CFDataRef>(
        IORegistryEntrySearchCFProperty(entry, kIOServicePlane, name, kCFAllocatorDefault,
                                        kIORegistryIterateRecursively | kIORegistryIterateParents));

    if (data == nullptr)
    {
        return false;
    }

    const uint32_t *valuePtr = reinterpret_cast<const uint32_t *>(CFDataGetBytePtr(data));

    if (valuePtr == nullptr)
    {
        CFRelease(data);
        return false;
    }

    *value = *valuePtr;
    CFRelease(data);
    return true;
}

// Gathers the vendor and device IDs for the PCI GPUs
bool GetPCIDevices(std::vector<GPUDeviceInfo> *devices)
{
    // matchDictionary will be consumed by IOServiceGetMatchingServices, no need to release it.
    CFMutableDictionaryRef matchDictionary = IOServiceMatching("IOPCIDevice");

    io_iterator_t entryIterator;
    if (IOServiceGetMatchingServices(kIOMasterPortDefault, matchDictionary, &entryIterator) !=
        kIOReturnSuccess)
    {
        return false;
    }

    io_registry_entry_t entry = IO_OBJECT_NULL;

    while ((entry = IOIteratorNext(entryIterator)) != IO_OBJECT_NULL)
    {
        constexpr uint32_t kClassCodeDisplayVGA = 0x30000;
        uint32_t classCode;
        GPUDeviceInfo info;

        if (GetEntryProperty(entry, CFSTR("class-code"), &classCode) &&
            classCode == kClassCodeDisplayVGA &&
            GetEntryProperty(entry, CFSTR("vendor-id"), &info.vendorId) &&
            GetEntryProperty(entry, CFSTR("device-id"), &info.deviceId))
        {
            devices->push_back(info);
        }

        IOObjectRelease(entry);
    }
    IOObjectRelease(entryIterator);

    return true;
}

void SetActiveGPUIndex(SystemInfo *info)
{
    VendorID activeVendor;
    DeviceID activeDevice;

    uint64_t gpuID = GetGpuIDFromDisplayID(kCGDirectMainDisplay);

    if (gpuID == 0)
        return;

    CFMutableDictionaryRef matchDictionary = IORegistryEntryIDMatching(gpuID);
    io_service_t gpuEntry = IOServiceGetMatchingService(kIOMasterPortDefault, matchDictionary);

    if (gpuEntry == IO_OBJECT_NULL)
    {
        IOObjectRelease(gpuEntry);
        return;
    }

    if (!(GetEntryProperty(gpuEntry, CFSTR("vendor-id"), &activeVendor) &&
          GetEntryProperty(gpuEntry, CFSTR("device-id"), &activeDevice)))
    {
        IOObjectRelease(gpuEntry);
        return;
    }

    IOObjectRelease(gpuEntry);

    for (size_t i = 0; i < info->gpus.size(); ++i)
    {
        if (info->gpus[i].vendorId == activeVendor && info->gpus[i].deviceId == activeDevice)
        {
            info->activeGPUIndex = static_cast<int>(i);
            break;
        }
    }
}

}  // anonymous namespace

bool GetSystemInfo(SystemInfo *info)
{
    {
        int32_t major = 0;
        int32_t minor = 0;
        ParseMacMachineModel(GetMachineModel(), &info->machineModelName, &major, &minor);
        info->machineModelVersion = std::to_string(major) + "." + std::to_string(minor);
    }

    if (!GetPCIDevices(&(info->gpus)))
    {
        return false;
    }

    if (info->gpus.empty())
    {
        return false;
    }

    // Call the generic GetDualGPUInfo function to initialize info fields
    // such as isOptimus, isAMDSwitchable, and the activeGPUIndex
    GetDualGPUInfo(info);

    // Then override the activeGPUIndex field of info to reflect the current
    // GPU instead of the non-intel GPU
    if (@available(macOS 10.13, *))
    {
        SetActiveGPUIndex(info);
    }

    // Figure out whether this is a dual-GPU system.
    //
    // TODO(kbr): this code was ported over from Chromium, and its correctness
    // could be improved - need to use Mac-specific APIs to determine whether
    // offline renderers are allowed, and whether these two GPUs are really the
    // integrated/discrete GPUs in a laptop.
    if (info->gpus.size() == 2 &&
        ((IsIntel(info->gpus[0].vendorId) && !IsIntel(info->gpus[1].vendorId)) ||
         (!IsIntel(info->gpus[0].vendorId) && IsIntel(info->gpus[1].vendorId))))
    {
        info->isMacSwitchable = true;
    }

    return true;
}

}  // namespace angle

#endif  // __has_include(<Cocoa/Cocoa.h>)
