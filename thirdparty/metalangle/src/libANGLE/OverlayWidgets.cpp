//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// OverlayWidgets.cpp:
//    Implements functions that interpret widget data.  Data formats and limits correspond to the
//    Vulkan implementation (as the only implementation).  They are generic enough so other backends
//    could respect them too, if they implement the overlay.
//

#include "libANGLE/Overlay.h"
#include "libANGLE/Overlay_font_autogen.h"

namespace gl
{
namespace
{
// Internally, every widget is either Text or Graph.
enum class WidgetInternalType
{
    Text,
    Graph,

    InvalidEnum,
    EnumCount = InvalidEnum,
};

// A map that says how the API-facing widget types map to internal types.
constexpr angle::PackedEnumMap<WidgetType, WidgetInternalType> kWidgetTypeToInternalMap = {
    {WidgetType::Count, WidgetInternalType::Text},
    {WidgetType::Text, WidgetInternalType::Text},
    {WidgetType::PerSecond, WidgetInternalType::Text},
    {WidgetType::RunningGraph, WidgetInternalType::Graph},
    {WidgetType::RunningHistogram, WidgetInternalType::Graph},
};

// Structures and limits matching uniform buffers in vulkan/shaders/src/OverlayDraw.comp.  The size
// of text and graph widgets is chosen such that they could fit in uniform buffers with minimum
// required Vulkan size.
constexpr size_t kMaxRenderableTextWidgets  = 32;
constexpr size_t kMaxRenderableGraphWidgets = 32;
constexpr size_t kMaxTextLength             = 256;
constexpr size_t kMaxGraphDataSize          = 64;

constexpr angle::PackedEnumMap<WidgetInternalType, size_t> kWidgetInternalTypeMaxWidgets = {
    {WidgetInternalType::Text, kMaxRenderableTextWidgets},
    {WidgetInternalType::Graph, kMaxRenderableGraphWidgets},
};

constexpr angle::PackedEnumMap<WidgetInternalType, size_t> kWidgetInternalTypeWidgetOffsets = {
    {WidgetInternalType::Text, 0},
    {WidgetInternalType::Graph, kMaxRenderableTextWidgets},
};

ANGLE_ENABLE_STRUCT_PADDING_WARNINGS

// Structure matching buffer in vulkan/shaders/src/OverlayCull.comp.
struct WidgetCoordinates
{
    uint32_t coordinates[kMaxRenderableTextWidgets + kMaxRenderableGraphWidgets][4];
};

// Structures matching buffers in vulkan/shaders/src/OverlayDraw.comp.
struct TextWidgetData
{
    uint32_t coordinates[4];
    float color[4];
    uint32_t fontSize[3];
    uint32_t padding;
    uint8_t text[kMaxTextLength];
};

struct GraphWidgetData
{
    uint32_t coordinates[4];
    float color[4];
    uint32_t valueWidth;
    uint32_t padding[3];
    uint32_t values[kMaxGraphDataSize];
};

struct TextWidgets
{
    TextWidgetData widgets[kMaxRenderableTextWidgets];
};

struct GraphWidgets
{
    GraphWidgetData widgets[kMaxRenderableGraphWidgets];
};

ANGLE_DISABLE_STRUCT_PADDING_WARNINGS

uint32_t GetWidgetCoord(int32_t src, uint32_t extent)
{
    int32_t dst = src < 0 ? extent + src : src;

    return std::min<uint32_t>(std::max(dst, 0), extent - 1);
}

void GetWidgetCoordinates(const int32_t srcCoords[4],
                          const gl::Extents &imageExtent,
                          uint32_t dstCoordsOut[4])
{
    dstCoordsOut[0] = GetWidgetCoord(srcCoords[0], imageExtent.width);
    dstCoordsOut[1] = GetWidgetCoord(srcCoords[1], imageExtent.height);
    dstCoordsOut[2] = GetWidgetCoord(srcCoords[2], imageExtent.width);
    dstCoordsOut[3] = GetWidgetCoord(srcCoords[3], imageExtent.height);
}

void GetWidgetColor(const float srcColor[4], float dstColor[4])
{
    memcpy(dstColor, srcColor, 4 * sizeof(dstColor[0]));
}

void GetTextFontSize(int srcFontSize, uint32_t dstFontSize[3])
{
    // .xy contains the font glyph width/height
    dstFontSize[0] = overlay::kFontGlyphWidths[srcFontSize];
    dstFontSize[1] = overlay::kFontGlyphHeights[srcFontSize];
    // .z contains the layer
    dstFontSize[2] = srcFontSize;
}

void GetGraphValueWidth(const int32_t srcCoords[4], size_t valueCount, uint32_t *dstValueWidth)
{
    const int32_t graphWidth = std::abs(srcCoords[2] - srcCoords[0]);

    // If valueCount doesn't divide graphWidth, the graph bars won't fit well in its frame.
    // Fix initOverlayWidgets() in that case.
    ASSERT(graphWidth % valueCount == 0);

    *dstValueWidth = graphWidth / valueCount;
}

void GetTextString(const std::string &src, uint8_t textOut[kMaxTextLength])
{
    for (size_t i = 0; i < src.length() && i < kMaxTextLength; ++i)
    {
        // The font image has 96 ASCII characters starting from ' '.
        textOut[i] = src[i] - ' ';
    }
}

void GetGraphValues(const std::vector<size_t> srcValues,
                    size_t startIndex,
                    float scale,
                    uint32_t valuesOut[kMaxGraphDataSize])
{
    ASSERT(srcValues.size() <= kMaxGraphDataSize);

    for (size_t i = 0; i < srcValues.size(); ++i)
    {
        size_t index = (startIndex + i) % srcValues.size();
        valuesOut[i] = static_cast<uint32_t>(srcValues[index] * scale);
    }
}

std::vector<size_t> CreateHistogram(const std::vector<size_t> values)
{
    std::vector<size_t> histogram(values.size(), 0);

    for (size_t rank : values)
    {
        ++histogram[rank];
    }

    return histogram;
}

using OverlayWidgetCounts  = angle::PackedEnumMap<WidgetInternalType, size_t>;
using AppendWidgetDataFunc = void (*)(const overlay::Widget *widget,
                                      const gl::Extents &imageExtent,
                                      TextWidgetData *textWidget,
                                      GraphWidgetData *graphWidget,
                                      OverlayWidgetCounts *widgetCounts);
}  // namespace

namespace overlay_impl
{
// This class interprets the generic data collected in every element into a human-understandable
// widget.  This often means generating text specific to this item and scaling graph data to
// something sensible.
class AppendWidgetDataHelper
{
  public:
    static void AppendFPS(const overlay::Widget *widget,
                          const gl::Extents &imageExtent,
                          TextWidgetData *textWidget,
                          GraphWidgetData *graphWidget,
                          OverlayWidgetCounts *widgetCounts);
    static void AppendVulkanLastValidationMessage(const overlay::Widget *widget,
                                                  const gl::Extents &imageExtent,
                                                  TextWidgetData *textWidget,
                                                  GraphWidgetData *graphWidget,
                                                  OverlayWidgetCounts *widgetCounts);
    static void AppendVulkanValidationMessageCount(const overlay::Widget *widget,
                                                   const gl::Extents &imageExtent,
                                                   TextWidgetData *textWidget,
                                                   GraphWidgetData *graphWidget,
                                                   OverlayWidgetCounts *widgetCounts);
    static void AppendVulkanCommandGraphSize(const overlay::Widget *widget,
                                             const gl::Extents &imageExtent,
                                             TextWidgetData *textWidget,
                                             GraphWidgetData *graphWidget,
                                             OverlayWidgetCounts *widgetCounts);
    static void AppendVulkanSecondaryCommandBufferPoolWaste(const overlay::Widget *widget,
                                                            const gl::Extents &imageExtent,
                                                            TextWidgetData *textWidget,
                                                            GraphWidgetData *graphWidget,
                                                            OverlayWidgetCounts *widgetCounts);

  private:
    static std::ostream &OutputPerSecond(std::ostream &out, const overlay::PerSecond *perSecond);

    static std::ostream &OutputText(std::ostream &out, const overlay::Text *text);

    static std::ostream &OutputCount(std::ostream &out, const overlay::Count *count);

    static void AppendTextCommon(const overlay::Widget *widget,
                                 const gl::Extents &imageExtent,
                                 const std::string &text,
                                 TextWidgetData *textWidget,
                                 OverlayWidgetCounts *widgetCounts);

    static void AppendGraphCommon(const overlay::Widget *widget,
                                  const gl::Extents &imageExtent,
                                  const std::vector<size_t> runningValues,
                                  size_t startIndex,
                                  float scale,
                                  GraphWidgetData *graphWidget,
                                  OverlayWidgetCounts *widgetCounts);
};

void AppendWidgetDataHelper::AppendTextCommon(const overlay::Widget *widget,
                                              const gl::Extents &imageExtent,
                                              const std::string &text,
                                              TextWidgetData *textWidget,
                                              OverlayWidgetCounts *widgetCounts)
{
    GetWidgetCoordinates(widget->coords, imageExtent, textWidget->coordinates);
    GetWidgetColor(widget->color, textWidget->color);
    GetTextFontSize(widget->fontSize, textWidget->fontSize);
    GetTextString(text, textWidget->text);

    ++(*widgetCounts)[WidgetInternalType::Text];
}

void AppendWidgetDataHelper::AppendGraphCommon(const overlay::Widget *widget,
                                               const gl::Extents &imageExtent,
                                               const std::vector<size_t> runningValues,
                                               size_t startIndex,
                                               float scale,
                                               GraphWidgetData *graphWidget,
                                               OverlayWidgetCounts *widgetCounts)
{
    const overlay::RunningGraph *widgetAsGraph = static_cast<const overlay::RunningGraph *>(widget);

    GetWidgetCoordinates(widget->coords, imageExtent, graphWidget->coordinates);
    GetWidgetColor(widget->color, graphWidget->color);
    GetGraphValueWidth(widget->coords, widgetAsGraph->runningValues.size(),
                       &graphWidget->valueWidth);
    GetGraphValues(runningValues, startIndex, scale, graphWidget->values);

    ++(*widgetCounts)[WidgetInternalType::Graph];
}

void AppendWidgetDataHelper::AppendFPS(const overlay::Widget *widget,
                                       const gl::Extents &imageExtent,
                                       TextWidgetData *textWidget,
                                       GraphWidgetData *graphWidget,
                                       OverlayWidgetCounts *widgetCounts)
{
    const overlay::PerSecond *fps = static_cast<const overlay::PerSecond *>(widget);
    std::ostringstream text;
    text << "FPS: ";
    OutputPerSecond(text, fps);

    AppendTextCommon(widget, imageExtent, text.str(), textWidget, widgetCounts);
}

void AppendWidgetDataHelper::AppendVulkanLastValidationMessage(const overlay::Widget *widget,
                                                               const gl::Extents &imageExtent,
                                                               TextWidgetData *textWidget,
                                                               GraphWidgetData *graphWidget,
                                                               OverlayWidgetCounts *widgetCounts)
{
    const overlay::Text *lastValidationMessage = static_cast<const overlay::Text *>(widget);
    std::ostringstream text;
    text << "Last VVL Message: ";
    OutputText(text, lastValidationMessage);

    AppendTextCommon(widget, imageExtent, text.str(), textWidget, widgetCounts);
}

void AppendWidgetDataHelper::AppendVulkanValidationMessageCount(const overlay::Widget *widget,
                                                                const gl::Extents &imageExtent,
                                                                TextWidgetData *textWidget,
                                                                GraphWidgetData *graphWidget,
                                                                OverlayWidgetCounts *widgetCounts)
{
    const overlay::Count *validationMessageCount = static_cast<const overlay::Count *>(widget);
    std::ostringstream text;
    text << "VVL Message Count: ";
    OutputCount(text, validationMessageCount);

    AppendTextCommon(widget, imageExtent, text.str(), textWidget, widgetCounts);
}

void AppendWidgetDataHelper::AppendVulkanCommandGraphSize(const overlay::Widget *widget,
                                                          const gl::Extents &imageExtent,
                                                          TextWidgetData *textWidget,
                                                          GraphWidgetData *graphWidget,
                                                          OverlayWidgetCounts *widgetCounts)
{
    const overlay::RunningGraph *commandGraphSize =
        static_cast<const overlay::RunningGraph *>(widget);

    const size_t maxValue     = *std::max_element(commandGraphSize->runningValues.begin(),
                                              commandGraphSize->runningValues.end());
    const int32_t graphHeight = std::abs(widget->coords[3] - widget->coords[1]);
    const float graphScale    = static_cast<float>(graphHeight) / maxValue;

    AppendGraphCommon(widget, imageExtent, commandGraphSize->runningValues,
                      commandGraphSize->lastValueIndex + 1, graphScale, graphWidget, widgetCounts);

    if ((*widgetCounts)[WidgetInternalType::Text] <
        kWidgetInternalTypeMaxWidgets[WidgetInternalType::Text])
    {
        std::ostringstream text;
        text << "Command Graph Size (Max: " << maxValue << ")";
        AppendTextCommon(&commandGraphSize->description, imageExtent, text.str(), textWidget,
                         widgetCounts);
    }
}

void AppendWidgetDataHelper::AppendVulkanSecondaryCommandBufferPoolWaste(
    const overlay::Widget *widget,
    const gl::Extents &imageExtent,
    TextWidgetData *textWidget,
    GraphWidgetData *graphWidget,
    OverlayWidgetCounts *widgetCounts)
{
    const overlay::RunningHistogram *secondaryCommandBufferPoolWaste =
        static_cast<const overlay::RunningHistogram *>(widget);

    std::vector<size_t> histogram = CreateHistogram(secondaryCommandBufferPoolWaste->runningValues);
    auto maxValueIter             = std::max_element(histogram.rbegin(), histogram.rend());
    const size_t maxValue         = *maxValueIter;
    const int32_t graphHeight     = std::abs(widget->coords[3] - widget->coords[1]);
    const float graphScale        = static_cast<float>(graphHeight) / maxValue;

    AppendGraphCommon(widget, imageExtent, histogram, 0, graphScale, graphWidget, widgetCounts);

    if ((*widgetCounts)[WidgetInternalType::Text] <
        kWidgetInternalTypeMaxWidgets[WidgetInternalType::Text])
    {
        std::ostringstream text;
        size_t peak        = std::distance(maxValueIter, histogram.rend() - 1);
        size_t peakPercent = (peak * 100 + 50) / histogram.size();

        text << "CB Pool Waste (Peak: " << peakPercent << "%)";
        AppendTextCommon(&secondaryCommandBufferPoolWaste->description, imageExtent, text.str(),
                         textWidget, widgetCounts);
    }
}

std::ostream &AppendWidgetDataHelper::OutputPerSecond(std::ostream &out,
                                                      const overlay::PerSecond *perSecond)
{
    return out << perSecond->lastPerSecondCount;
}

std::ostream &AppendWidgetDataHelper::OutputText(std::ostream &out, const overlay::Text *text)
{
    return out << text->text;
}

std::ostream &AppendWidgetDataHelper::OutputCount(std::ostream &out, const overlay::Count *count)
{
    return out << count->count;
}
}  // namespace overlay_impl

namespace
{
constexpr angle::PackedEnumMap<WidgetId, AppendWidgetDataFunc> kWidgetIdToAppendDataFuncMap = {
    {WidgetId::FPS, overlay_impl::AppendWidgetDataHelper::AppendFPS},
    {WidgetId::VulkanLastValidationMessage,
     overlay_impl::AppendWidgetDataHelper::AppendVulkanLastValidationMessage},
    {WidgetId::VulkanValidationMessageCount,
     overlay_impl::AppendWidgetDataHelper::AppendVulkanValidationMessageCount},
    {WidgetId::VulkanCommandGraphSize,
     overlay_impl::AppendWidgetDataHelper::AppendVulkanCommandGraphSize},
    {WidgetId::VulkanSecondaryCommandBufferPoolWaste,
     overlay_impl::AppendWidgetDataHelper::AppendVulkanSecondaryCommandBufferPoolWaste},
};
}

namespace overlay
{
RunningGraph::RunningGraph(size_t n) : runningValues(n, 0) {}
RunningGraph::~RunningGraph() = default;
}  // namespace overlay

size_t OverlayState::getWidgetCoordinatesBufferSize() const
{
    return sizeof(WidgetCoordinates);
}

size_t OverlayState::getTextWidgetsBufferSize() const
{
    return sizeof(TextWidgets);
}

size_t OverlayState::getGraphWidgetsBufferSize() const
{
    return sizeof(GraphWidgets);
}

void OverlayState::fillEnabledWidgetCoordinates(const gl::Extents &imageExtents,
                                                uint8_t *enabledWidgetsPtr) const
{
    WidgetCoordinates *enabledWidgets = reinterpret_cast<WidgetCoordinates *>(enabledWidgetsPtr);
    memset(enabledWidgets, 0, sizeof(*enabledWidgets));

    OverlayWidgetCounts widgetCounts = {};

    for (const std::unique_ptr<overlay::Widget> &widget : mOverlayWidgets)
    {
        if (!widget->enabled)
        {
            continue;
        }

        WidgetInternalType internalType = kWidgetTypeToInternalMap[widget->type];
        ASSERT(internalType != WidgetInternalType::InvalidEnum);

        if (widgetCounts[internalType] >= kWidgetInternalTypeMaxWidgets[internalType])
        {
            continue;
        }

        size_t writeIndex =
            kWidgetInternalTypeWidgetOffsets[internalType] + widgetCounts[internalType]++;

        GetWidgetCoordinates(widget->coords, imageExtents, enabledWidgets->coordinates[writeIndex]);

        // Graph widgets have a text widget attached as well.
        if (internalType == WidgetInternalType::Graph)
        {
            WidgetInternalType textType = WidgetInternalType::Text;
            if (widgetCounts[textType] >= kWidgetInternalTypeMaxWidgets[textType])
            {
                continue;
            }

            const overlay::RunningGraph *widgetAsGraph =
                static_cast<const overlay::RunningGraph *>(widget.get());
            writeIndex = kWidgetInternalTypeWidgetOffsets[textType] + widgetCounts[textType]++;

            GetWidgetCoordinates(widgetAsGraph->description.coords, imageExtents,
                                 enabledWidgets->coordinates[writeIndex]);
        }
    }
}

void OverlayState::fillWidgetData(const gl::Extents &imageExtents,
                                  uint8_t *textData,
                                  uint8_t *graphData) const
{
    TextWidgets *textWidgets   = reinterpret_cast<TextWidgets *>(textData);
    GraphWidgets *graphWidgets = reinterpret_cast<GraphWidgets *>(graphData);

    memset(textWidgets, overlay::kFontCharacters, sizeof(*textWidgets));
    memset(graphWidgets, 0, sizeof(*graphWidgets));

    OverlayWidgetCounts widgetCounts = {};

    for (WidgetId id : angle::AllEnums<WidgetId>())
    {
        const std::unique_ptr<overlay::Widget> &widget = mOverlayWidgets[id];
        if (!widget->enabled)
        {
            continue;
        }

        WidgetInternalType internalType = kWidgetTypeToInternalMap[widget->type];
        ASSERT(internalType != WidgetInternalType::InvalidEnum);

        if (widgetCounts[internalType] >= kWidgetInternalTypeMaxWidgets[internalType])
        {
            continue;
        }

        AppendWidgetDataFunc appendFunc = kWidgetIdToAppendDataFuncMap[id];
        appendFunc(widget.get(), imageExtents,
                   &textWidgets->widgets[widgetCounts[WidgetInternalType::Text]],
                   &graphWidgets->widgets[widgetCounts[WidgetInternalType::Graph]], &widgetCounts);
    }
}

}  // namespace gl
