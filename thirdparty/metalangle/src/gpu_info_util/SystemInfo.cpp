//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SystemInfo.cpp: implementation of the system-agnostic parts of SystemInfo.h

#include "gpu_info_util/SystemInfo.h"

#include <cstring>
#include <iostream>
#include <sstream>

#include "common/debug.h"
#include "common/string_utils.h"

namespace angle
{
namespace
{
std::string VendorName(VendorID vendor)
{
    switch (vendor)
    {
        case kVendorID_AMD:
            return "AMD";
        case kVendorID_Intel:
            return "Intel";
        case kVendorID_ImgTec:
            return "ImgTec";
        case kVendorID_NVIDIA:
            return "NVIDIA";
        case kVendorID_Qualcomm:
            return "Qualcomm";
        case kVendorID_Vivante:
            return "Vivante";
        case kVendorID_VeriSilicon:
            return "VeriSilicon";
        case kVendorID_VMWare:
            return "VMWare";
        case kVendorID_Kazan:
            return "Kazan";
        default:
            return "Unknown (" + std::to_string(vendor) + ")";
    }
}
}  // anonymous namespace
GPUDeviceInfo::GPUDeviceInfo() = default;

GPUDeviceInfo::~GPUDeviceInfo() = default;

GPUDeviceInfo::GPUDeviceInfo(const GPUDeviceInfo &other) = default;

SystemInfo::SystemInfo() = default;

SystemInfo::~SystemInfo() = default;

SystemInfo::SystemInfo(const SystemInfo &other) = default;

bool SystemInfo::hasNVIDIAGPU() const
{
    for (const GPUDeviceInfo &gpu : gpus)
    {
        if (IsNVIDIA(gpu.vendorId))
        {
            return true;
        }
    }
    return false;
}

bool SystemInfo::hasIntelGPU() const
{
    for (const GPUDeviceInfo &gpu : gpus)
    {
        if (IsIntel(gpu.vendorId))
        {
            return true;
        }
    }
    return false;
}

bool SystemInfo::hasAMDGPU() const
{
    for (const GPUDeviceInfo &gpu : gpus)
    {
        if (IsAMD(gpu.vendorId))
        {
            return true;
        }
    }
    return false;
}

bool IsAMD(VendorID vendorId)
{
    return vendorId == kVendorID_AMD;
}

bool IsARM(VendorID vendorId)
{
    return vendorId == kVendorID_ARM;
}

bool IsImgTec(VendorID vendorId)
{
    return vendorId == kVendorID_ImgTec;
}

bool IsKazan(VendorID vendorId)
{
    return vendorId == kVendorID_Kazan;
}

bool IsIntel(VendorID vendorId)
{
    return vendorId == kVendorID_Intel;
}

bool IsNVIDIA(VendorID vendorId)
{
    return vendorId == kVendorID_NVIDIA;
}

bool IsQualcomm(VendorID vendorId)
{
    return vendorId == kVendorID_Qualcomm;
}

bool IsVeriSilicon(VendorID vendorId)
{
    return vendorId == kVendorID_VeriSilicon;
}

bool IsVMWare(VendorID vendorId)
{
    return vendorId == kVendorID_VMWare;
}

bool IsVivante(VendorID vendorId)
{
    return vendorId == kVendorID_Vivante;
}

bool ParseAMDBrahmaDriverVersion(const std::string &content, std::string *version)
{
    const size_t begin = content.find_first_of("0123456789");
    if (begin == std::string::npos)
    {
        return false;
    }

    const size_t end = content.find_first_not_of("0123456789.", begin);
    if (end == std::string::npos)
    {
        *version = content.substr(begin);
    }
    else
    {
        *version = content.substr(begin, end - begin);
    }
    return true;
}

bool ParseAMDCatalystDriverVersion(const std::string &content, std::string *version)
{
    std::istringstream stream(content);

    std::string line;
    while (std::getline(stream, line))
    {
        static const char kReleaseVersion[] = "ReleaseVersion=";
        if (line.compare(0, std::strlen(kReleaseVersion), kReleaseVersion) != 0)
        {
            continue;
        }

        if (ParseAMDBrahmaDriverVersion(line, version))
        {
            return true;
        }
    }
    return false;
}

bool ParseMacMachineModel(const std::string &identifier,
                          std::string *type,
                          int32_t *major,
                          int32_t *minor)
{
    size_t numberLoc = identifier.find_first_of("0123456789");
    if (numberLoc == std::string::npos)
    {
        return false;
    }

    size_t commaLoc = identifier.find(',', numberLoc);
    if (commaLoc == std::string::npos || commaLoc >= identifier.size())
    {
        return false;
    }

    const char *numberPtr = &identifier[numberLoc];
    const char *commaPtr  = &identifier[commaLoc + 1];
    char *endPtr          = nullptr;

    int32_t majorTmp = static_cast<int32_t>(std::strtol(numberPtr, &endPtr, 10));
    if (endPtr == numberPtr)
    {
        return false;
    }

    int32_t minorTmp = static_cast<int32_t>(std::strtol(commaPtr, &endPtr, 10));
    if (endPtr == commaPtr)
    {
        return false;
    }

    *major = majorTmp;
    *minor = minorTmp;
    *type  = identifier.substr(0, numberLoc);

    return true;
}

bool CMDeviceIDToDeviceAndVendorID(const std::string &id, uint32_t *vendorId, uint32_t *deviceId)
{
    unsigned int vendor = 0;
    unsigned int device = 0;

    bool success = id.length() >= 21 && HexStringToUInt(id.substr(8, 4), &vendor) &&
                   HexStringToUInt(id.substr(17, 4), &device);

    *vendorId = vendor;
    *deviceId = device;
    return success;
}

void GetDualGPUInfo(SystemInfo *info)
{
    ASSERT(!info->gpus.empty());

    // On dual-GPU systems we assume the non-Intel GPU is the graphics one.
    int active    = 0;
    bool hasIntel = false;
    for (size_t i = 0; i < info->gpus.size(); ++i)
    {
        if (IsIntel(info->gpus[i].vendorId))
        {
            hasIntel = true;
        }
        if (IsIntel(info->gpus[active].vendorId))
        {
            active = static_cast<int>(i);
        }
    }

    // Assume that a combination of NVIDIA or AMD with Intel means Optimus or AMD Switchable
    info->activeGPUIndex  = active;
    info->isOptimus       = hasIntel && IsNVIDIA(info->gpus[active].vendorId);
    info->isAMDSwitchable = hasIntel && IsAMD(info->gpus[active].vendorId);
}

void PrintSystemInfo(const SystemInfo &info)
{
    std::cout << info.gpus.size() << " GPUs:\n";

    for (size_t i = 0; i < info.gpus.size(); i++)
    {
        const auto &gpu = info.gpus[i];

        std::cout << "  " << i << " - " << VendorName(gpu.vendorId) << " device id: 0x" << std::hex
                  << std::uppercase << gpu.deviceId << std::dec << "\n";
        if (!gpu.driverVendor.empty())
        {
            std::cout << "       Driver Vendor: " << gpu.driverVendor << "\n";
        }
        if (!gpu.driverVersion.empty())
        {
            std::cout << "       Driver Version: " << gpu.driverVersion << "\n";
        }
        if (!gpu.driverDate.empty())
        {
            std::cout << "       Driver Date: " << gpu.driverDate << "\n";
        }
    }

    std::cout << "\n";
    std::cout << "Active GPU: " << info.activeGPUIndex << "\n";

    std::cout << "\n";
    std::cout << "Optimus: " << (info.isOptimus ? "true" : "false") << "\n";
    std::cout << "AMD Switchable: " << (info.isAMDSwitchable ? "true" : "false") << "\n";

    std::cout << "\n";
    if (!info.machineManufacturer.empty())
    {
        std::cout << "Machine Manufacturer: " << info.machineManufacturer << "\n";
    }
    if (!info.machineModelName.empty())
    {
        std::cout << "Machine Model: " << info.machineModelName << "\n";
    }
    if (!info.machineModelVersion.empty())
    {
        std::cout << "Machine Model Version: " << info.machineModelVersion << "\n";
    }
    std::cout << std::endl;
}
}  // namespace angle
