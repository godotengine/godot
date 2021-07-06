//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Overlay.h:
//    Defines the Overlay class that handles overlay widgets.
//

#ifndef LIBANGLE_OVERLAY_H_
#define LIBANGLE_OVERLAY_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/OverlayWidgets.h"
#include "libANGLE/angletypes.h"

namespace rx
{
class OverlayImpl;
class GLImplFactory;
}  // namespace rx

namespace gl
{
class Context;

class OverlayState : angle::NonCopyable
{
  public:
    OverlayState();
    ~OverlayState();

    size_t getWidgetCoordinatesBufferSize() const;
    size_t getTextWidgetsBufferSize() const;
    size_t getGraphWidgetsBufferSize() const;

    void initFontData(uint8_t *fontData) const;
    void fillEnabledWidgetCoordinates(const gl::Extents &imageExtents,
                                      uint8_t *enabledWidgetsPtr) const;
    void fillWidgetData(const gl::Extents &imageExtents,
                        uint8_t *textData,
                        uint8_t *graphData) const;

    uint32_t getEnabledWidgetCount() const { return mEnabledWidgetCount; }

  private:
    friend class Overlay;

    uint32_t mEnabledWidgetCount;

    angle::PackedEnumMap<WidgetId, std::unique_ptr<overlay::Widget>> mOverlayWidgets;
};

class Overlay : angle::NonCopyable
{
  public:
    Overlay(rx::GLImplFactory *implFactory);
    ~Overlay();

    angle::Result init(const Context *context);
    void destroy(const gl::Context *context);

    void onSwap() const;

    overlay::Text *getTextWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::Text, WidgetType::Text>(id);
    }
    overlay::Count *getCountWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::Count, WidgetType::Count>(id);
    }
    overlay::PerSecond *getPerSecondWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::PerSecond, WidgetType::PerSecond>(id);
    }
    overlay::RunningGraph *getRunningGraphWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::RunningGraph, WidgetType::RunningGraph>(id);
    }
    overlay::RunningHistogram *getRunningHistogramWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::RunningHistogram, WidgetType::RunningHistogram>(id);
    }

    rx::OverlayImpl *getImplementation() const { return mImplementation.get(); }

  private:
    template <typename Widget, WidgetType Type>
    Widget *getWidgetAs(WidgetId id) const
    {
        ASSERT(mState.mOverlayWidgets[id] != nullptr);
        ASSERT(mState.mOverlayWidgets[id]->type == Type);
        return rx::GetAs<Widget>(mState.mOverlayWidgets[id].get());
    }
    void initOverlayWidgets();
    void enableOverlayWidgetsFromEnvironment();

    // Time tracking for PerSecond items.
    mutable double mLastPerSecondUpdate;

    OverlayState mState;
    std::unique_ptr<rx::OverlayImpl> mImplementation;
};

class DummyOverlay
{
  public:
    DummyOverlay(rx::GLImplFactory *implFactory);
    ~DummyOverlay();

    angle::Result init(const Context *context) { return angle::Result::Continue; }
    void destroy(const Context *context) {}

    void onSwap() const {}

    const overlay::Dummy *getTextWidget(WidgetId id) const { return &mDummy; }
    const overlay::Dummy *getCountWidget(WidgetId id) const { return &mDummy; }
    const overlay::Dummy *getPerSecondWidget(WidgetId id) const { return &mDummy; }
    const overlay::Dummy *getRunningGraphWidget(WidgetId id) const { return &mDummy; }
    const overlay::Dummy *getRunningHistogramWidget(WidgetId id) const { return &mDummy; }

  private:
    overlay::Dummy mDummy;
};

#if ANGLE_ENABLE_OVERLAY
using OverlayType = Overlay;
#else   // !ANGLE_ENABLE_OVERLAY
using OverlayType = DummyOverlay;
#endif  // ANGLE_ENABLE_OVERLAY

}  // namespace gl

#endif  // LIBANGLE_OVERLAY_H_
