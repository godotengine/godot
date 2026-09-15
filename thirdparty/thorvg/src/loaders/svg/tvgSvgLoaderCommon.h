/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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
#include "tvgInlist.h"
#include "tvgColor.h"

using SvgColor = tvg::RGB;

#define STR_AS(A, B) !strcmp((A), (B))

struct Box
{
    float x, y, w, h;

    void intersect(const Box& box)
    {
        auto x1 = x + w;
        auto y1 = y + h;
        auto x2 = box.x + box.w;
        auto y2 = box.y + box.h;

        x = x > box.x ? x : box.x;
        y = y > box.y ? y : box.y;
        w = (x1 < x2 ? x1 : x2) - x;
        h = (y1 < y2 ? y1 : y2) - y;

        if (w < 0.0f) w = 0.0f;
        if (h < 0.0f) h = 0.0f;
    }
};

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
    Filter,
    GaussianBlur,
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

constexpr bool operator&(SvgFillFlags a, SvgFillFlags b)
{
    return int(a) & int(b);
}

constexpr SvgFillFlags operator|(SvgFillFlags a, SvgFillFlags b)
{
    return SvgFillFlags(int(a) | int(b));
}

constexpr void operator|=(SvgFillFlags& a, const SvgFillFlags b)
{
    a = SvgFillFlags(int(a) | int(b));
}

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
    Miterlimit = 0x100,
    DashOffset = 0x200
};

constexpr bool operator&(SvgStrokeFlags a, SvgStrokeFlags b)
{
    return int(a) & int(b);
}

constexpr SvgStrokeFlags operator|(SvgStrokeFlags a, SvgStrokeFlags b)
{
    return SvgStrokeFlags(int(a) | int(b));
}

constexpr void operator|=(SvgStrokeFlags& a, const SvgStrokeFlags b)
{
    a = SvgStrokeFlags(int(a) | int(b));
}

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
    Display = 0x8000,
    PaintOrder = 0x10000,
    StrokeMiterlimit = 0x20000,
    StrokeDashOffset = 0x40000,
    Filter = 0x80000
};

constexpr bool operator&(SvgStyleFlags a, SvgStyleFlags b)
{
    return int(a) & int(b);
}

constexpr SvgStyleFlags operator|(SvgStyleFlags a, SvgStyleFlags b)
{
    return SvgStyleFlags(int(a) | int(b));
}

constexpr void operator|=(SvgStyleFlags& a, const SvgStyleFlags b)
{
    a = SvgStyleFlags(int(a) | int(b));
}

enum class SvgStopStyleFlags
{
    StopDefault = 0x0,
    StopOpacity = 0x01,
    StopColor = 0x02
};

constexpr bool operator&(SvgStopStyleFlags a, SvgStopStyleFlags b)
{
    return int(a) & int(b);
}

constexpr SvgStopStyleFlags operator|(SvgStopStyleFlags a, SvgStopStyleFlags b)
{
    return SvgStopStyleFlags(int(a) | int(b));
}

enum class SvgGradientFlags
{
    None = 0x0,
    GradientUnits = 0x1,
    SpreadMethod = 0x2,
    X1 = 0x4,
    X2 = 0x8,
    Y1 = 0x10,
    Y2 = 0x20,
    Cx = 0x40,
    Cy = 0x80,
    R = 0x100,
    Fx = 0x200,
    Fy = 0x400,
    Fr = 0x800
};

constexpr bool operator &(SvgGradientFlags a, SvgGradientFlags b)
{
    return int(a) & int(b);
}

constexpr SvgGradientFlags operator |(SvgGradientFlags a, SvgGradientFlags b)
{
    return SvgGradientFlags(int(a) | int(b));
}

enum class SvgMaskType
{
    Luminance = 0,
    Alpha
};

enum class SvgXmlSpace
{
    None,
    Default,
    Preserve
};

//Length type to recalculate %, pt, pc, mm, cm etc
enum class SvgParserLengthType
{
    Vertical,
    Horizontal,
    Diagonal,
    //In case of, for example, radius of radial gradient
    Other
};

enum class SvgViewFlag
{
    None = 0x0,
    Width = 0x01,   //viewPort width
    Height = 0x02,  //viewPort height
    Viewbox = 0x04,  //viewBox x,y,w,h - used only if all 4 are correctly set
    WidthInPercent = 0x08,
    HeightInPercent = 0x10
};

constexpr bool operator &(SvgViewFlag a, SvgViewFlag b)
{
    return static_cast<int>(a) & static_cast<int>(b);
}

constexpr SvgViewFlag operator |(SvgViewFlag a, SvgViewFlag b)
{
    return SvgViewFlag(int(a) | int(b));
}

constexpr SvgViewFlag operator ^(SvgViewFlag a, SvgViewFlag b)
{
    return SvgViewFlag(int(a) ^ int(b));
}

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
    float w, h;   //unit: point or in percentage see: SvgViewFlag
    Box vbox;
    SvgViewFlag viewFlag;
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
    float cx, cy, rx, ry;
};

struct SvgCircleNode
{
    float cx, cy, r;
};

struct SvgRectNode
{
    float x, y, w, h, rx, ry;
    bool hasRx, hasRy;
};

struct SvgLineNode
{
    float x1, y1, x2, y2;
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
    Array<float> pts;
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

struct SvgTextNode
{
    char* text;
    char* fontFamily;
    float x, y;
    float fontSize;
};

struct SvgGaussianBlurNode
{
    float stdDevX, stdDevY;
    Box box;
    bool isPercentage[4];
    bool hasBox;
    bool edgeModeWrap;
};

struct SvgFilterNode
{
    Box box;
    bool isPercentage[4];
    bool filterUserSpace;
    bool primitiveUserSpace;
};

struct SvgLinearGradient
{
    float x1, y1, x2, y2;
    bool isX1Percentage;
    bool isY1Percentage;
    bool isX2Percentage;
    bool isY2Percentage;
};

struct SvgRadialGradient
{
    float cx, cy, fx, fy, r, fr;
    bool isCxPercentage;
    bool isCyPercentage;
    bool isFxPercentage;
    bool isFyPercentage;
    bool isRPercentage;
    bool isFrPercentage;
};

struct SvgComposite
{
    char *url;
    SvgNode* node;
    bool applying;              //flag for checking circular dependency.
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
    float offset;
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
    SvgGradientFlags flags;
    bool userSpace;

    void clear()
    {
        stops.reset();
        tvg::free(transform);
        tvg::free(radial);
        tvg::free(linear);
        tvg::free(ref);
        tvg::free(id);
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
    float miterlimit;
    SvgDash dash;
};

struct SvgFilter
{
    char *url;
    SvgNode* node;
};

struct SvgStyleProperty
{
    SvgStyleFill fill;
    SvgStyleStroke stroke;
    SvgComposite clipPath;
    SvgComposite mask;
    SvgFilter filter;
    int opacity;
    SvgColor color;
    char* cssClass;
    SvgStyleFlags flags;
    SvgStyleFlags flagsImportance; //indicates the importance of the flag - if set, higher priority is applied (https://drafts.csswg.org/css-cascade-4/#importance)
    bool curColorSet;
    bool paintOrder; //true if default (fill, stroke), false otherwise
    bool display;
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
        SvgTextNode text;
        SvgFilterNode filter;
        SvgGaussianBlurNode gaussianBlur;
    } node;
    SvgXmlSpace xmlSpace = SvgXmlSpace::None;
    ~SvgNode();
};

struct SvgParser
{
    SvgNode* node;
    SvgStyleGradient* styleGrad;
    Fill::ColorStop gradStop;
    SvgStopStyleFlags flags;
    Box global;
    struct
    {
        bool parsedFx;
        bool parsedFy;
    } gradient;
};

struct SvgNodeIdPair
{
    INLIST_ITEM(SvgNodeIdPair);
    SvgNodeIdPair(SvgNode* n, char* i) : node{n}, id{i} {}
    SvgNode* node;
    char *id;
};

struct FontFace
{
    char* name = nullptr;
    char* src = nullptr;
    size_t srcLen = 0;
    char* decoded = nullptr;
};

enum class OpenedTagType : uint8_t
{
    Other = 0,
    Style,
    Text
};

struct SvgLoaderData
{
    Array<SvgNode*> stack;
    SvgNode* doc = nullptr;
    SvgNode* def = nullptr; //also used to store nested graphic nodes
    SvgNode* cssStyle = nullptr;
    Array<SvgStyleGradient*> gradients;
    Array<SvgStyleGradient*> gradientStack; //For stops
    SvgParser* svgParse = nullptr;
    Inlist<SvgNodeIdPair> cloneNodes;
    Array<SvgNodeIdPair> nodesToStyle;
    Array<char*> images;        //embedded images
    Array<FontFace> fonts;
    int level = 0;
    bool result = false;
    OpenedTagType openedTag = OpenedTagType::Other;
    SvgNode* currentGraphicsNode = nullptr;
};

#endif
