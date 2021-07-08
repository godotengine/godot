//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// OverlayWidgets.h:
//    Defines the Overlay* widget classes and corresponding enums.
//

#ifndef LIBANGLE_OVERLAYWIDGETS_H_
#define LIBANGLE_OVERLAYWIDGETS_H_

#include "common/angleutils.h"

namespace gl
{
class Overlay;
class OverlayState;

namespace overlay_impl
{
class AppendWidgetDataHelper;
}

enum class WidgetType
{
    // Text types:

    // A total count of some event.
    Count,
    // A single line of ASCII text.  Retains content until changed.
    Text,
    // A per-second value.
    PerSecond,

    // Graph types:

    // A graph of the last N values.
    RunningGraph,
    // A histogram of the last N values (values between 0 and 1).
    RunningHistogram,

    InvalidEnum,
    EnumCount = InvalidEnum,
};

enum class WidgetId
{
    // Front-end widgets:

    // Frames per second (PerSecond).
    FPS,

    // Vulkan backend:

    // Last validation error (Text).
    VulkanLastValidationMessage,
    // Number of validation errors and warnings (Count).
    VulkanValidationMessageCount,
    // Number of nodes in command graph (RunningGraph).
    VulkanCommandGraphSize,
    // Secondary Command Buffer pool memory waste (RunningHistogram).
    VulkanSecondaryCommandBufferPoolWaste,

    InvalidEnum,
    EnumCount = InvalidEnum,
};

namespace overlay
{
class Widget
{
  public:
    virtual ~Widget() {}

  protected:
    WidgetType type;
    // Whether this item should be drawn.
    bool enabled = false;

    // For text items, size of the font.  This is a value in [0, overlay::kFontCount) which
    // determines the font size to use.
    int fontSize;

    // The area covered by the item, predetermined by the overlay class.  Negative values
    // indicate offset from the left/bottom of the image.
    int32_t coords[4];
    float color[4];

    friend class gl::Overlay;
    friend class gl::OverlayState;
    friend class overlay_impl::AppendWidgetDataHelper;
};

class Count : public Widget
{
  public:
    ~Count() override {}
    void add(size_t n) { count += n; }
    void reset() { count = 0; }

  protected:
    size_t count = 0;

    friend class gl::Overlay;
    friend class overlay_impl::AppendWidgetDataHelper;
};

class PerSecond : public Count
{
  public:
    ~PerSecond() override {}

  protected:
    size_t lastPerSecondCount = 0;

    friend class gl::Overlay;
    friend class overlay_impl::AppendWidgetDataHelper;
};

class Text : public Widget
{
  public:
    ~Text() override {}
    void set(std::string &&str) { text = std::move(str); }

  protected:
    std::string text;

    friend class overlay_impl::AppendWidgetDataHelper;
};

class RunningGraph : public Widget
{
  public:
    // Out of line constructor to satisfy chromium-style.
    RunningGraph(size_t n);
    ~RunningGraph() override;
    void add(size_t n) { runningValues[lastValueIndex] += n; }
    void next()
    {
        lastValueIndex                = (lastValueIndex + 1) % runningValues.size();
        runningValues[lastValueIndex] = 0;
    }

  protected:
    std::vector<size_t> runningValues;
    size_t lastValueIndex = 0;
    Text description;

    friend class gl::Overlay;
    friend class gl::OverlayState;
    friend class overlay_impl::AppendWidgetDataHelper;
};

class RunningHistogram : public RunningGraph
{
  public:
    RunningHistogram(size_t n) : RunningGraph(n) {}
    ~RunningHistogram() override {}
    void set(float n)
    {
        ASSERT(n >= 0.0f && n <= 1.0f);
        size_t rank =
            n == 1.0f ? runningValues.size() - 1 : static_cast<size_t>(n * runningValues.size());

        runningValues[lastValueIndex] = rank;
    }
};

// If overlay is disabled, all the above classes would be replaced with Dummy, turning them into
// noop.
class Dummy
{
  public:
    void reset() const {}
    template <typename T>
    void set(T) const
    {}
    template <typename T>
    void add(T) const
    {}
    void next() const {}
};

}  // namespace overlay

}  // namespace gl

#endif  // LIBANGLE_OVERLAYWIDGETS_H_
