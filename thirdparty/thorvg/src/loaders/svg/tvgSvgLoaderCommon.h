/*
 * Copyright (c) 2020 - 2022 Samsung Electronics Co., Ltd. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef _TVG_SVG_LOADER_COMMON_H_
#define _TVG_SVG_LOADER_COMMON_H_

#include "tvgCommon.h"
#include "tvgArray.h"

struct SvgNode;
struct SvgStyleGradient;

//NOTE: Please update simpleXmlNodeTypeToString() as well.
enum class SvgNodeType
{
    Doc,
    G,
    Defs,
    Animation,
    Arc,
    Circle,
    Ellipse,
    Image,
    Line,
    Path,
    Polygon,
    Polyline,
    Rect,
    Text,
    TextArea,
    Tspan,
    Use,
    Video,
    ClipPath,
    Mask,
    CssStyle,
    Symbol,
    Unknown
};

/*
// TODO - remove?
enum class SvgLengthType
{
    Percent,
    Px,
    Pc,
    Pt,
    Mm,
    Cm,
    In,
};
*/

enum class SvgFillFlags
{
    Paint = 0x01,
    Opacity = 0x02,
    Gradient = 0x04,
    FillRule = 0x08,
    ClipPath = 0x16
};

enum class SvgStrokeFlags
{
    Paint = 0x1,
    Opacity = 0x2,
    Gradient = 0x4,
    Scale = 0x8,
    Width = 0x10,
    Cap = 0x20,
    Join = 0x40,
    Dash = 0x80,
};

enum class SvgGradientType
{
    Linear,
    Radial
};

enum class SvgStyleFlags
{
    Color = 0x01,
    Fill = 0x02,
    FillRule = 0x04,
    FillOpacity = 0x08,
    Opacity = 0x010,
    Stroke = 0x20,
    StrokeWidth = 0x40,
    StrokeLineJoin = 0x80,
    StrokeLineCap = 0x100,
    StrokeOpacity = 0x200,
    StrokeDashArray = 0x400,
    Transform = 0x800,
    ClipPath = 0x1000,
    Mask = 0x2000,
    MaskType = 0x4000,
    Display = 0x8000
};

enum class SvgStopStyleFlags
{
    StopDefault = 0x0,
    StopOpacity = 0x01,
    StopColor = 0x02
};

enum class SvgFillRule
{
    Winding = 0,
    OddEven = 1
};

enum class SvgMaskType
{
    Luminance = 0,
    Alpha
};

//Length type to recalculate %, pt, pc, mm, cm etc
enum class SvgParserLengthType
{
    Vertical,
    Horizontal,
    //In case of, for example, radius of radial gradient
    Other
};

enum class AspectRatioAlign
{
    None,
    XMinYMin,
    XMidYMin,
    XMaxYMin,
    XMinYMid,
    XMidYMid,
    XMaxYMid,
    XMinYMax,
    XMidYMax,
    XMaxYMax
};

enum class AspectRatioMeetOrSlice
{
    Meet,
    Slice
};

struct SvgDocNode
{
    float w;
    float h;
    float vx;
    float vy;
    float vw;
    float vh;
    SvgNode* defs;
    SvgNode* style;
    AspectRatioAlign align;
    AspectRatioMeetOrSlice meetOrSlice;
};

struct SvgGNode
{
};

struct SvgDefsNode
{
    Array<SvgStyleGradient*> gradients;
};

struct SvgSymbolNode
{
    float w, h;
    float vx, vy, vw, vh;
    AspectRatioAlign align;
    AspectRatioMeetOrSlice meetOrSlice;
    bool overflowVisible;
    bool hasViewBox;
    bool hasWidth;
    bool hasHeight;
};

struct SvgUseNode
{
    float x, y, w, h;
    bool isWidthSet;
    bool isHeightSet;
    SvgNode* symbol;
};

struct SvgEllipseNode
{
    float cx;
    float cy;
    float rx;
    float ry;
};

struct SvgCircleNode
{
    float cx;
    float cy;
    float r;
};

struct SvgRectNode
{
    float x;
    float y;
    float w;
    float h;
    float rx;
    float ry;
    bool hasRx;
    bool hasRy;
};

struct SvgLineNode
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct SvgImageNode
{
    float x, y, w, h;
    char* href;
};

struct SvgPathNode
{
    char* path;
};

struct SvgPolygonNode
{
    int pointsCount;
    float* points;
};

struct SvgClipNode
{
    bool userSpace;
};

struct SvgMaskNode
{
    SvgMaskType type;
    bool userSpace;
};

struct SvgCssStyleNode
{
};

struct SvgLinearGradient
{
    float x1;
    float y1;
    float x2;
    float y2;
    bool isX1Percentage;
    bool isY1Percentage;
    bool isX2Percentage;
    bool isY2Percentage;
};

struct SvgRadialGradient
{
    float cx;
    float cy;
    float fx;
    float fy;
    float r;
    bool isCxPercentage;
    bool isCyPercentage;
    bool isFxPercentage;
    bool isFyPercentage;
    bool isRPercentage;
};

struct SvgComposite
{
    char *url;
    SvgNode* node;
    bool applying;              //flag for checking circular dependency.
};

struct SvgColor
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct SvgPaint
{
    SvgStyleGradient* gradient;
    char *url;
    SvgColor color;
    bool none;
    bool curColor;
};

struct SvgDash
{
    Array<float> array;
};

struct SvgStyleGradient
{
    SvgGradientType type;
    char* id;
    char* ref;
    FillSpread spread;
    SvgRadialGradient* radial;
    SvgLinearGradient* linear;
    Matrix* transform;
    Array<Fill::ColorStop> stops;
    bool userSpace;

    void clear()
    {
        stops.reset();
        free(transform);
        free(radial);
        free(linear);
        free(ref);
        free(id);
    }
};

struct SvgStyleFill
{
    SvgFillFlags flags;
    SvgPaint paint;
    int opacity;
    FillRule fillRule;
};

struct SvgStyleStroke
{
    SvgStrokeFlags flags;
    SvgPaint paint;
    int opacity;
    float scale;
    float width;
    float centered;
    StrokeCap cap;
    StrokeJoin join;
    SvgDash dash;
    int dashCount;
};

struct SvgStyleProperty
{
    SvgStyleFill fill;
    SvgStyleStroke stroke;
    SvgComposite clipPath;
    SvgComposite mask;
    int opacity;
    SvgColor color;
    bool curColorSet;
    char* cssClass;
    SvgStyleFlags flags;
};

struct SvgNode
{
    SvgNodeType type;
    SvgNode* parent;
    Array<SvgNode*> child;
    char *id;
    SvgStyleProperty *style;
    Matrix* transform;
    union {
        SvgGNode g;
        SvgDocNode doc;
        SvgDefsNode defs;
        SvgUseNode use;
        SvgCircleNode circle;
        SvgEllipseNode ellipse;
        SvgPolygonNode polygon;
        SvgPolygonNode polyline;
        SvgRectNode rect;
        SvgPathNode path;
        SvgLineNode line;
        SvgImageNode image;
        SvgMaskNode mask;
        SvgClipNode clip;
        SvgCssStyleNode cssStyle;
        SvgSymbolNode symbol;
    } node;
    bool display;
    ~SvgNode();
};

struct SvgParser
{
    SvgNode* node;
    SvgStyleGradient* styleGrad;
    Fill::ColorStop gradStop;
    SvgStopStyleFlags flags;
    struct
    {
        int x, y;
        uint32_t w, h;
    } global;
    struct
    {
        bool parsedFx;
        bool parsedFy;
    } gradient;
};

struct SvgNodeIdPair
{
    SvgNode* node;
    char *id;
};

struct SvgLoaderData
{
    Array<SvgNode*> stack = {nullptr, 0, 0};
    SvgNode* doc = nullptr;
    SvgNode* def = nullptr;
    SvgNode* cssStyle = nullptr;
    Array<SvgStyleGradient*> gradients;
    SvgStyleGradient* latestGradient = nullptr; //For stops
    SvgParser* svgParse = nullptr;
    Array<SvgNodeIdPair> cloneNodes;
    Array<SvgNodeIdPair> nodesToStyle;
    int level = 0;
    bool result = false;
    bool style = false;
};

/*
 * https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/strtof-strtod-l-wcstod-wcstod-l?view=vs-2017
 *
 * src should be one of the following form :
 *
 * [whitespace] [sign] {digits [radix digits] | radix digits} [{e | E} [sign] digits]
 * [whitespace] [sign] {INF | INFINITY}
 * [whitespace] [sign] NAN [sequence]
 *
 * No hexadecimal form supported
 * no sequence supported after NAN
 */
float customStrtof(const char *nptr, char **endptr);

#endif
