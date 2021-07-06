//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SystemInfo.h: gathers information available without starting a GPU driver.

#ifndef GPU_INFO_UTIL_SYSTEM_INFO_H_
#define GPU_INFO_UTIL_SYSTEM_INFO_H_

#include <cstdint>
#include <string>
#include <vector>

namespace angle
{

using VendorID = uint32_t;
using DeviceID = uint32_t;

struct VersionInfo
{
    uint32_t major    = 0;
    uint32_t minor    = 0;
    uint32_t subMinor = 0;
    uint32_t patch    = 0;
};

struct GPUDeviceInfo
{
    GPUDeviceInfo();
    ~GPUDeviceInfo();

    GPUDeviceInfo(const GPUDeviceInfo &other);

    VendorID vendorId = 0;
    DeviceID deviceId = 0;

    std::string driverVendor;
    std::string driverVersion;
    std::string driverDate;

    // Only available on Android
    VersionInfo detailedDriverVersion;
};

struct SystemInfo
{
    SystemInfo();
    ~SystemInfo();

    SystemInfo(const SystemInfo &other);

    bool hasNVIDIAGPU() const;
    bool hasIntelGPU() const;
    bool hasAMDGPU() const;

    std::vector<GPUDeviceInfo> gpus;

    // Index of the GPU expected to be used for 3D graphics. Based on a best-guess heuristic on
    // some platforms. On windows, this is accurate.
    int activeGPUIndex = -1;

    bool isOptimus       = false;
    bool isAMDSwitchable = false;
    // Only true on dual-GPU Mac laptops.
    bool isMacSwitchable = false;

    // Only available on Android
    std::string machineManufacturer;

    // Only available on macOS and Android
    std::string machineModelName;

    // Only available on macOS
    std::string machineModelVersion;
};

// Gathers information about the system without starting a GPU driver and returns them in `info`.
// Returns true if all info was gathered, false otherwise. Even when false is returned, `info` will
// be filled with partial information.
bool GetSystemInfo(SystemInfo *info);

// Known PCI vendor IDs
constexpr VendorID kVendorID_AMD      = 0x1002;
constexpr VendorID kVendorID_ARM      = 0x13B5;
constexpr VendorID kVendorID_ImgTec   = 0x1010;
constexpr VendorID kVendorID_Intel    = 0x8086;
constexpr VendorID kVendorID_NVIDIA   = 0x10DE;
constexpr VendorID kVendorID_Qualcomm = 0x5143;
constexpr VendorID kVendorID_VMWare   = 0x15ad;

// Known non-PCI (i.e. Khronos-registered) vendor IDs
constexpr VendorID kVendorID_Vivante     = 0x10001;
constexpr VendorID kVendorID_VeriSilicon = 0x10002;
constexpr VendorID kVendorID_Kazan       = 0x10003;

// Predicates on vendor IDs
bool IsAMD(VendorID vendorId);
bool IsARM(VendorID vendorId);
bool IsImgTec(VendorID vendorId);
bool IsIntel(VendorID vendorId);
bool IsKazan(VendorID vendorId);
bool IsNVIDIA(VendorID vendorId);
bool IsQualcomm(VendorID vendorId);
bool IsVeriSilicon(VendorID vendorId);
bool IsVMWare(VendorID vendorId);
bool IsVivante(VendorID vendorId);

// Use a heuristic to attempt to find the GPU used for 3D graphics. Sets activeGPUIndex,
// isOptimus, and isAMDSwitchable.
// Always assumes the non-Intel GPU is active on dual-GPU machines.
void GetDualGPUInfo(SystemInfo *info);

// Dumps the system info to stdout.
void PrintSystemInfo(const SystemInfo &info);
}  // namespace angle

#endif  // GPU_INFO_UTIL_SYSTEM_INFO_H_
