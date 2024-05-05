/*
 * Copyright (c) 2023 - 2024 the ThorVG project. All rights reserved.

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

#ifndef _TVG_LOTTIE_MODEL_H_
#define _TVG_LOTTIE_MODEL_H_

#include <cstring>

#include "tvgCommon.h"
#include "tvgRender.h"
#include "tvgLottieProperty.h"


struct LottieComposition;

struct LottieStroke
{
    struct DashAttr
    {
        //0: offset, 1: dash, 2: gap
        LottieFloat value[3] = {0.0f, 0.0f, 0.0f};
    };

    virtual ~LottieStroke()
    {
        delete(dashattr);
    }

    LottieFloat& dash(int no)
    {
        if (!dashattr) dashattr = new DashAttr;
        return dashattr->value[no];
    }

    float dashOffset(float frameNo)
    {
        return dash(0)(frameNo);
    }

    float dashGap(float frameNo)
    {
        return dash(2)(frameNo);
    }

    float dashSize(float frameNo)
    {
        auto d = dash(1)(frameNo);
        if (d == 0.0f) return 0.1f;
        else return d;
    }

    bool dynamic()
    {
        if (width.frames || dashattr) return true;
        return false;
    }

    LottieFloat width = 0.0f;
    DashAttr* dashattr = nullptr;
    float miterLimit = 0;
    StrokeCap cap = StrokeCap::Round;
    StrokeJoin join = StrokeJoin::Round;
};


struct LottieMask
{
    LottiePathSet pathset;
    LottieOpacity opacity = 255;
    CompositeMethod method;
    bool inverse = false;
};


struct LottieObject
{
    enum Type : uint8_t
    {
        Composition = 0,
        Layer,
        Group,
        Transform,
        SolidFill,
        SolidStroke,
        GradientFill,
        GradientStroke,
        Rect,
        Ellipse,
        Path,
        Polystar,
        Image,
        Trimpath,
        Text,
        Repeater,
        RoundedCorner
    };

    virtual ~LottieObject()
    {
        free(name);
    }

    virtual void override(LottieProperty* prop)
    {
        TVGERR("LOTTIE", "Unsupported slot type");
    }

    virtual bool mergeable() { return false; }

    char* name = nullptr;
    Type type;
    bool hidden = false;       //remove?
};


struct LottieGlyph
{
    Array<LottieObject*> children;   //glyph shapes.
    float width;
    char* code;
    char* family = nullptr;
    char* style = nullptr;
    uint16_t size;
    uint8_t len;

    void prepare()
    {
        len = strlen(code);
    }

    ~LottieGlyph()
    {
        for (auto p = children.begin(); p < children.end(); ++p) delete(*p);
        free(code);
    }
};


struct LottieFont
{
    enum Origin : uint8_t { Local = 0, CssURL, ScriptURL, FontURL, Embedded };

    ~LottieFont()
    {
        for (auto c = chars.begin(); c < chars.end(); ++c) delete(*c);
        free(style);
        free(family);
        free(name);
    }

    Array<LottieGlyph*> chars;
    char* name = nullptr;
    char* family = nullptr;
    char* style = nullptr;
    float ascent = 0.0f;
    Origin origin = Embedded;
};

struct LottieMarker
{
    char* name = nullptr;
    float time = 0.0f;
    float duration = 0.0f;
    
    ~LottieMarker()
    {
        free(name);
    }
};

struct LottieText : LottieObject
{
    void prepare()
    {
        LottieObject::type = LottieObject::Text;
    }

    void override(LottieProperty* prop) override
    {
        this->doc = *static_cast<LottieTextDoc*>(prop);
        this->prepare();
    }

    LottieTextDoc doc;
    LottieFont* font;
    LottieFloat spacing = 0.0f;  //letter spacing
};


struct LottieTrimpath : LottieObject
{
    enum Type : uint8_t { Simultaneous = 1, Individual = 2 };

    void prepare()
    {
        LottieObject::type = LottieObject::Trimpath;
    }

    bool mergeable() override
    {
        if (!start.frames && start.value == 0.0f && !end.frames && end.value == 100.0f && !offset.frames && offset.value == 0.0f) return true;
        return false;
    }

    void segment(float frameNo, float& start, float& end);

    LottieFloat start = 0.0f;
    LottieFloat end = 100.0f;
    LottieFloat offset = 0.0f;
    Type type = Simultaneous;
};


struct LottieShape : LottieObject
{
    virtual ~LottieShape() {}
    uint8_t direction = 0;   //0: clockwise, 2: counter-clockwise, 3: xor(?)

    bool mergeable() override
    {
        return true;
    }
};


struct LottieRoundedCorner : LottieObject
{
    void prepare()
    {
        LottieObject::type = LottieObject::RoundedCorner;
    }
    LottieFloat radius = 0.0f;
};


struct LottiePath : LottieShape
{
    void prepare()
    {
        LottieObject::type = LottieObject::Path;
    }

    LottiePathSet pathset;
};


struct LottieRect : LottieShape
{
    void prepare()
    {
        LottieObject::type = LottieObject::Rect;
    }

    LottiePosition position = Point{0.0f, 0.0f};
    LottiePoint size = Point{0.0f, 0.0f};
    LottieFloat radius = 0.0f;       //rounded corner radius
};


struct LottiePolyStar : LottieShape
{
    enum Type : uint8_t {Star = 1, Polygon};

    void prepare()
    {
        LottieObject::type = LottieObject::Polystar;
    }

    LottiePosition position = Point{0.0f, 0.0f};
    LottieFloat innerRadius = 0.0f;
    LottieFloat outerRadius = 0.0f;
    LottieFloat innerRoundness = 0.0f;
    LottieFloat outerRoundness = 0.0f;
    LottieFloat rotation = 0.0f;
    LottieFloat ptsCnt = 0.0f;
    Type type = Polygon;
};


struct LottieEllipse : LottieShape
{
    void prepare()
    {
        LottieObject::type = LottieObject::Ellipse;
    }

    LottiePosition position = Point{0.0f, 0.0f};
    LottiePoint size = Point{0.0f, 0.0f};
};


struct LottieTransform : LottieObject
{
    struct SeparateCoord
    {
        LottieFloat x = 0.0f;
        LottieFloat y = 0.0f;
    };

    struct RotationEx
    {
        LottieFloat x = 0.0f;
        LottieFloat y = 0.0f;
    };

    ~LottieTransform()
    {
        delete(coords);
        delete(rotationEx);
    }

    void prepare()
    {
        LottieObject::type = LottieObject::Transform;
    }

    bool mergeable() override
    {
        if (!opacity.frames && opacity.value == 255) return true;
        return false;
    }

    LottiePosition position = Point{0.0f, 0.0f};
    LottieFloat rotation = 0.0f;           //z rotation
    LottiePoint scale = Point{100.0f, 100.0f};
    LottiePoint anchor = Point{0.0f, 0.0f};
    LottieOpacity opacity = 255;

    SeparateCoord* coords = nullptr;       //either a position or separate coordinates
    RotationEx* rotationEx = nullptr;      //extension for 3d rotation
};


struct LottieSolid : LottieObject 
{
    LottieColor color = RGB24{255, 255, 255};
    LottieOpacity opacity = 255;
};


struct LottieSolidStroke : LottieSolid, LottieStroke
{
    void prepare()
    {
        LottieObject::type = LottieObject::SolidStroke;
    }

    void override(LottieProperty* prop) override
    {
        this->color = *static_cast<LottieColor*>(prop);
        this->prepare();
    }
};


struct LottieSolidFill : LottieSolid
{
    void prepare()
    {
        LottieObject::type = LottieObject::SolidFill;
    }

    void override(LottieProperty* prop) override
    {
        this->color = *static_cast<LottieColor*>(prop);
        this->prepare();
    }

    FillRule rule = FillRule::Winding;
};


struct LottieGradient : LottieObject
{
    uint32_t populate(ColorStop& color)
    {
        colorStops.populated = true;
        if (!color.input) return 0;

        uint32_t alphaCnt = (color.input->count - (colorStops.count * 4)) / 2;
        Array<Fill::ColorStop> output(colorStops.count + alphaCnt);
        uint32_t cidx = 0;               //color count
        uint32_t clast = colorStops.count * 4;
        if (clast > color.input->count) clast = color.input->count;
        uint32_t aidx = clast;           //alpha count
        Fill::ColorStop cs;

        //merge color stops.
        for (uint32_t i = 0; i < color.input->count; ++i) {
            if (cidx == clast || aidx == color.input->count) break;
            if ((*color.input)[cidx] == (*color.input)[aidx]) {
                cs.offset = (*color.input)[cidx];
                cs.r = lroundf((*color.input)[cidx + 1] * 255.0f);
                cs.g = lroundf((*color.input)[cidx + 2] * 255.0f);
                cs.b = lroundf((*color.input)[cidx + 3] * 255.0f);
                cs.a = lroundf((*color.input)[aidx + 1] * 255.0f);
                cidx += 4;
                aidx += 2;
            } else if ((*color.input)[cidx] < (*color.input)[aidx]) {
                cs.offset = (*color.input)[cidx];
                cs.r = lroundf((*color.input)[cidx + 1] * 255.0f);
                cs.g = lroundf((*color.input)[cidx + 2] * 255.0f);
                cs.b = lroundf((*color.input)[cidx + 3] * 255.0f);
                //generate alpha value
                if (output.count > 0) {
                    auto p = ((*color.input)[cidx] - output.last().offset) / ((*color.input)[aidx] - output.last().offset);
                    cs.a = mathLerp<uint8_t>(output.last().a, lroundf((*color.input)[aidx + 1] * 255.0f), p);
                } else cs.a = 255;
                cidx += 4;
            } else {
                cs.offset = (*color.input)[aidx];
                cs.a = lroundf((*color.input)[aidx + 1] * 255.0f);
                //generate color value
                if (output.count > 0) {
                    auto p = ((*color.input)[aidx] - output.last().offset) / ((*color.input)[cidx] - output.last().offset);
                    cs.r = mathLerp<uint8_t>(output.last().r, lroundf((*color.input)[cidx + 1] * 255.0f), p);
                    cs.g = mathLerp<uint8_t>(output.last().g, lroundf((*color.input)[cidx + 2] * 255.0f), p);
                    cs.b = mathLerp<uint8_t>(output.last().b, lroundf((*color.input)[cidx + 3] * 255.0f), p);
                } else cs.r = cs.g = cs.b = 255;
                aidx += 2;
            }
            output.push(cs);
        }

        //color remains
        while (cidx + 3 < clast) {
            cs.offset = (*color.input)[cidx];
            cs.r = lroundf((*color.input)[cidx + 1] * 255.0f);
            cs.g = lroundf((*color.input)[cidx + 2] * 255.0f);
            cs.b = lroundf((*color.input)[cidx + 3] * 255.0f);
            cs.a = (output.count > 0) ? output.last().a : 255;
            output.push(cs);
            cidx += 4;
        }

        //alpha remains
        while (aidx < color.input->count) {
            cs.offset = (*color.input)[aidx];
            cs.a = lroundf((*color.input)[aidx + 1] * 255.0f);
            if (output.count > 0) {
                cs.r = output.last().r;
                cs.g = output.last().g;
                cs.b = output.last().b;
            } else cs.r = cs.g = cs.b = 255;
            output.push(cs);
            aidx += 2;
        }

        color.data = output.data;
        output.data = nullptr;

        color.input->reset();
        delete(color.input);

        return output.count;
    }

    bool prepare()
    {
        if (!colorStops.populated) {
            if (colorStops.frames) {
                for (auto v = colorStops.frames->begin(); v < colorStops.frames->end(); ++v) {
                    colorStops.count = populate(v->value);
                }
            } else {
                colorStops.count = populate(colorStops.value);
            }
        }
        if (start.frames || end.frames || height.frames || angle.frames || opacity.frames || colorStops.frames) return true;
        return false;
    }

    Fill* fill(float frameNo);

    LottiePoint start = Point{0.0f, 0.0f};
    LottiePoint end = Point{0.0f, 0.0f};
    LottieFloat height = 0.0f;
    LottieFloat angle = 0.0f;
    LottieOpacity opacity = 255;
    LottieColorStop colorStops;
    uint8_t id = 0;    //1: linear, 2: radial
};


struct LottieGradientFill : LottieGradient
{
    void prepare()
    {
        LottieObject::type = LottieObject::GradientFill;
        LottieGradient::prepare();
    }

    void override(LottieProperty* prop) override
    {
        this->colorStops = *static_cast<LottieColorStop*>(prop);
        this->prepare();
    }

    FillRule rule = FillRule::Winding;
};


struct LottieGradientStroke : LottieGradient, LottieStroke
{
    void prepare()
    {
        LottieObject::type = LottieObject::GradientStroke;
        LottieGradient::prepare();
    }

    void override(LottieProperty* prop) override
    {
        this->colorStops = *static_cast<LottieColorStop*>(prop);
        this->prepare();
    }
};


struct LottieImage : LottieObject
{
    union {
        char* b64Data = nullptr;
        char* path;
    };
    char* mimeType = nullptr;
    uint32_t size = 0;

    Picture* picture = nullptr;   //tvg render data

    ~LottieImage();

    void prepare()
    {
        LottieObject::type = LottieObject::Image;
    }
};


struct LottieRepeater : LottieObject
{
    void prepare()
    {
        LottieObject::type = LottieObject::Repeater;
    }

    LottieFloat copies = 0.0f;
    LottieFloat offset = 0.0f;

    //Transform
    LottiePosition position = Point{0.0f, 0.0f};
    LottieFloat rotation = 0.0f;
    LottiePoint scale = Point{100.0f, 100.0f};
    LottiePoint anchor = Point{0.0f, 0.0f};
    LottieOpacity startOpacity = 255;
    LottieOpacity endOpacity = 255;
    bool inorder = true;        //true: higher,  false: lower
};


struct LottieGroup : LottieObject
{
    virtual ~LottieGroup()
    {
        for (auto p = children.begin(); p < children.end(); ++p) delete(*p);
    }

    void prepare(LottieObject::Type type = LottieObject::Group);
    bool mergeable() override { return allowMerge; }

    Scene* scene = nullptr;               //tvg render data
    Array<LottieObject*> children;

    bool reqFragment = false;   //requirment to fragment the render context
    bool buildDone = false;     //completed in building the composition.
    bool allowMerge = true;     //if this group is consisted of simple (transformed) shapes.
    bool trimpath = false;      //this group has a trimpath.
};


struct LottieLayer : LottieGroup
{
    enum Type : uint8_t {Precomp = 0, Solid, Image, Null, Shape, Text};

    ~LottieLayer();

    uint8_t opacity(float frameNo)
    {
        //return zero if the visibility is false.
        if (type == Null) return 255;
        return transform->opacity(frameNo);
    }

    bool mergeable() override { return false; }

    void prepare();
    float remap(float frameNo);

    struct {
        CompositeMethod type = CompositeMethod::None;
        LottieLayer* target = nullptr;
    } matte;

    BlendMethod blendMethod = BlendMethod::Normal;
    LottieLayer* parent = nullptr;
    LottieFloat timeRemap = 0.0f;
    LottieComposition* comp = nullptr;
    LottieTransform* transform = nullptr;
    Array<LottieMask*> masks;
    RGB24 color;  //used by Solid layer

    float timeStretch = 1.0f;
    float w = 0.0f, h = 0.0f;
    float inFrame = 0.0f;
    float outFrame = 0.0f;
    float startFrame = 0.0f;
    char* refId = nullptr;      //pre-composition reference.
    int16_t pid = -1;           //id of the parent layer.
    int16_t id = -1;            //id of the current layer.

    //cached data
    struct {
        float frameNo = -1.0f;
        Matrix matrix;
        uint8_t opacity;
    } cache;

    Type type = Null;
    bool autoOrient = false;
    bool matteSrc = false;
};


struct LottieSlot
{
    struct Pair {
        LottieObject* obj;
        LottieProperty* prop;
    };

    void assign(LottieObject* target)
    {
        //apply slot object to all targets
        for (auto pair = pairs.begin(); pair < pairs.end(); ++pair) {
            //backup the original properties before overwriting
            switch (type) {
                case LottieProperty::Type::ColorStop: {
                    if (!overriden) {
                        pair->prop = new LottieColorStop;
                        *static_cast<LottieColorStop*>(pair->prop) = static_cast<LottieGradient*>(pair->obj)->colorStops;
                    }

                    pair->obj->override(&static_cast<LottieGradient*>(target)->colorStops);
                    break;
                }
                case LottieProperty::Type::Color: {
                    if (!overriden) {
                        pair->prop = new LottieColor;
                        *static_cast<LottieColor*>(pair->prop) = static_cast<LottieSolid*>(pair->obj)->color;
                    }

                    pair->obj->override(&static_cast<LottieSolid*>(target)->color);
                    break;
                }
                case LottieProperty::Type::TextDoc: {
                    if (!overriden) {
                        pair->prop = new LottieTextDoc;
                        *static_cast<LottieTextDoc*>(pair->prop) = static_cast<LottieText*>(pair->obj)->doc;
                    }

                    pair->obj->override(&static_cast<LottieText*>(target)->doc);
                    break;
                }
                default: break;
            }
        }
        overriden = true;
    }

    void reset()
    {
        if (!overriden) return;

        for (auto pair = pairs.begin(); pair < pairs.end(); ++pair) {
            switch (type) {
                case LottieProperty::Type::ColorStop: {
                    static_cast<LottieGradient*>(pair->obj)->colorStops.release();
                    static_cast<LottieGradient*>(pair->obj)->colorStops = *static_cast<LottieColorStop*>(pair->prop);
                    break;
                }
                case LottieProperty::Type::Color: {
                    static_cast<LottieSolid*>(pair->obj)->color.release();
                    static_cast<LottieSolid*>(pair->obj)->color = *static_cast<LottieColor*>(pair->prop);
                    break;
                }
                case LottieProperty::Type::TextDoc: {
                    static_cast<LottieText*>(pair->obj)->doc.release();
                    static_cast<LottieText*>(pair->obj)->doc = *static_cast<LottieTextDoc*>(pair->prop);
                    break;
                }
                default: break;
            }
            delete(pair->prop);
            pair->prop = nullptr;
        }
        overriden = false;
    }

    LottieSlot(char* sid, LottieObject* obj, LottieProperty::Type type) : sid(sid), type(type)
    {
        pairs.push({obj});
    }

    ~LottieSlot()
    {
        free(sid);
        if (!overriden) return;
        for (auto pair = pairs.begin(); pair < pairs.end(); ++pair) {
            delete(pair->prop);
        }
    }

    char* sid;
    Array<Pair> pairs;
    LottieProperty::Type type;
    bool overriden = false;
};


struct LottieComposition
{
    ~LottieComposition();

    float duration() const
    {
        return frameCnt() / frameRate;  // in second
    }

    float frameAtTime(float timeInSec) const
    {
        auto p = timeInSec / duration();
        if (p < 0.0f) p = 0.0f;
        return p * frameCnt();
    }

    float frameCnt() const
    {
        return endFrame - startFrame;
    }

    LottieLayer* root = nullptr;
    char* version = nullptr;
    char* name = nullptr;
    float w, h;
    float startFrame, endFrame;
    float frameRate;
    Array<LottieObject*> assets;
    Array<LottieInterpolator*> interpolators;
    Array<LottieFont*> fonts;
    Array<LottieSlot*> slots;
    Array<LottieMarker*> markers;
    bool initiated = false;
};

#endif //_TVG_LOTTIE_MODEL_H_
