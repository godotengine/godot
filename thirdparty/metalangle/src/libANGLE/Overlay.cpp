//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Overlay.cpp:
//    Implements the Overlay class.
//

#include "libANGLE/Overlay.h"

#include "common/system_utils.h"
#include "libANGLE/Context.h"
#include "libANGLE/Overlay_font_autogen.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/OverlayImpl.h"

#include <numeric>

namespace gl
{
namespace
{
constexpr std::pair<const char *, WidgetId> kWidgetNames[] = {
    {"FPS", WidgetId::FPS},
    {"VulkanLastValidationMessage", WidgetId::VulkanLastValidationMessage},
    {"VulkanValidationMessageCount", WidgetId::VulkanValidationMessageCount},
    {"VulkanCommandGraphSize", WidgetId::VulkanCommandGraphSize},
    {"VulkanSecondaryCommandBufferPoolWaste", WidgetId::VulkanSecondaryCommandBufferPoolWaste},
};
}  // namespace

OverlayState::OverlayState() : mEnabledWidgetCount(0), mOverlayWidgets{} {}
OverlayState::~OverlayState() = default;

Overlay::Overlay(rx::GLImplFactory *factory)
    : mLastPerSecondUpdate(0), mImplementation(factory->createOverlay(mState))
{}
Overlay::~Overlay() = default;

angle::Result Overlay::init(const Context *context)
{
    initOverlayWidgets();
    mLastPerSecondUpdate = angle::GetCurrentTime();

    ASSERT(std::all_of(
        mState.mOverlayWidgets.begin(), mState.mOverlayWidgets.end(),
        [](const std::unique_ptr<overlay::Widget> &widget) { return widget.get() != nullptr; }));

    enableOverlayWidgetsFromEnvironment();

    return mImplementation->init(context);
}

void Overlay::destroy(const gl::Context *context)
{
    ASSERT(mImplementation);
    mImplementation->onDestroy(context);
}

void Overlay::enableOverlayWidgetsFromEnvironment()
{
    std::istringstream angleOverlayWidgets(angle::GetEnvironmentVar("ANGLE_OVERLAY"));

    std::set<std::string> enabledWidgets;
    std::string widget;
    while (getline(angleOverlayWidgets, widget, ':'))
    {
        enabledWidgets.insert(widget);
    }

    for (const std::pair<const char *, WidgetId> &widgetName : kWidgetNames)
    {
        if (enabledWidgets.count(widgetName.first) > 0)
        {
            mState.mOverlayWidgets[widgetName.second]->enabled = true;
            ++mState.mEnabledWidgetCount;
        }
    }
}

void Overlay::onSwap() const
{
    // Increment FPS counter.
    getPerSecondWidget(WidgetId::FPS)->add(1);

    // Update per second values every second.
    double currentTime = angle::GetCurrentTime();
    double timeDiff    = currentTime - mLastPerSecondUpdate;
    if (timeDiff >= 1.0)
    {
        for (const std::unique_ptr<overlay::Widget> &widget : mState.mOverlayWidgets)
        {
            if (widget->type == WidgetType::PerSecond)
            {
                overlay::PerSecond *perSecond =
                    reinterpret_cast<overlay::PerSecond *>(widget.get());
                perSecond->lastPerSecondCount = static_cast<size_t>(perSecond->count / timeDiff);
                perSecond->count              = 0;
            }
        }
        mLastPerSecondUpdate += 1.0;
    }
}

DummyOverlay::DummyOverlay(rx::GLImplFactory *implFactory) {}
DummyOverlay::~DummyOverlay() = default;

}  // namespace gl
