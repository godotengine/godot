//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Device.cpp: Implements the egl::Device class, representing the abstract
// device. Implements EGLDevice.

#include "libANGLE/Device.h"

#include <iterator>

#include <EGL/eglext.h>
#include <platform/Platform.h>

#include "anglebase/no_destructor.h"
#include "common/debug.h"
#include "common/platform.h"
#include "libANGLE/renderer/DeviceImpl.h"

#if defined(ANGLE_ENABLE_D3D11)
#    include "libANGLE/renderer/d3d/DeviceD3D.h"
#endif

namespace egl
{

template <typename T>
static std::string GenerateExtensionsString(const T &extensions)
{
    std::vector<std::string> extensionsVector = extensions.getStrings();

    std::ostringstream stream;
    std::copy(extensionsVector.begin(), extensionsVector.end(),
              std::ostream_iterator<std::string>(stream, " "));
    return stream.str();
}

typedef std::set<egl::Device *> DeviceSet;
static DeviceSet *GetDeviceSet()
{
    static angle::base::NoDestructor<DeviceSet> devices;
    return devices.get();
}

// Static factory methods
egl::Error Device::CreateDevice(EGLint deviceType, void *nativeDevice, Device **outDevice)
{
    *outDevice = nullptr;

    std::unique_ptr<rx::DeviceImpl> newDeviceImpl;

#if defined(ANGLE_ENABLE_D3D11)
    if (deviceType == EGL_D3D11_DEVICE_ANGLE)
    {
        newDeviceImpl.reset(new rx::DeviceD3D(deviceType, nativeDevice));
    }
#endif

    // Note that creating an EGL device from inputted D3D9 parameters isn't currently supported

    if (newDeviceImpl == nullptr)
    {
        return EglBadAttribute();
    }

    ANGLE_TRY(newDeviceImpl->initialize());
    *outDevice = new Device(nullptr, newDeviceImpl.release());

    return NoError();
}

bool Device::IsValidDevice(const Device *device)
{
    const DeviceSet *deviceSet = GetDeviceSet();
    return deviceSet->find(const_cast<Device *>(device)) != deviceSet->end();
}

Device::Device(Display *owningDisplay, rx::DeviceImpl *impl)
    : mLabel(nullptr), mOwningDisplay(owningDisplay), mImplementation(impl)
{
    ASSERT(GetDeviceSet()->find(this) == GetDeviceSet()->end());
    GetDeviceSet()->insert(this);
    initDeviceExtensions();
}

Device::~Device()
{
    ASSERT(GetDeviceSet()->find(this) != GetDeviceSet()->end());
    GetDeviceSet()->erase(this);
}

void Device::setLabel(EGLLabelKHR label)
{
    mLabel = label;
}

EGLLabelKHR Device::getLabel() const
{
    return mLabel;
}

Error Device::getAttribute(EGLint attribute, EGLAttrib *value)
{
    void *nativeAttribute = nullptr;
    egl::Error error =
        getImplementation()->getAttribute(getOwningDisplay(), attribute, &nativeAttribute);
    *value = reinterpret_cast<EGLAttrib>(nativeAttribute);
    return error;
}

EGLint Device::getType()
{
    return getImplementation()->getType();
}

void Device::initDeviceExtensions()
{
    mImplementation->generateExtensions(&mDeviceExtensions);
    mDeviceExtensionString = GenerateExtensionsString(mDeviceExtensions);
}

const DeviceExtensions &Device::getExtensions() const
{
    return mDeviceExtensions;
}

const std::string &Device::getExtensionString() const
{
    return mDeviceExtensionString;
}
}  // namespace egl
