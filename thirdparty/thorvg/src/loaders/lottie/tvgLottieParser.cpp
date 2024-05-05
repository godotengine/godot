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

#include "tvgStr.h"
#include "tvgCompressor.h"
#include "tvgLottieModel.h"
#include "tvgLottieParser.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#define KEY_AS(name) !strcmp(key, name)


static char* _int2str(int num)
{
    char str[20];
    snprintf(str, 20, "%d", num);
    return strdup(str);
}


CompositeMethod LottieParser::getMaskMethod(bool inversed)
{
    auto mode = getString();
    if (!mode) return CompositeMethod::None;

    switch (mode[0]) {
        case 'a': {
            if (inversed) return CompositeMethod::InvAlphaMask;
            else return CompositeMethod::AddMask;
        }
        case 's': return CompositeMethod::SubtractMask;
        case 'i': return CompositeMethod::IntersectMask;
        case 'f': return CompositeMethod::DifferenceMask;
        default: return CompositeMethod::None;
    }
}


BlendMethod LottieParser::getBlendMethod()
{
    switch (getInt()) {
        case 0: return BlendMethod::Normal;
        case 1: return BlendMethod::Multiply;
        case 2: return BlendMethod::Screen;
        case 3: return BlendMethod::Overlay;
        case 4: return BlendMethod::Darken;
        case 5: return BlendMethod::Lighten;
        case 6: return BlendMethod::ColorDodge;
        case 7: return BlendMethod::ColorBurn;
        case 8: return BlendMethod::HardLight;
        case 9: return BlendMethod::SoftLight;
        case 10: return BlendMethod::Difference;
        case 11: return BlendMethod::Exclusion;
        //case 12: return BlendMethod::Hue:
        //case 13: return BlendMethod::Saturation:
        //case 14: return BlendMethod::Color:
        //case 15: return BlendMethod::Luminosity:
        case 16: return BlendMethod::Add;
        //case 17: return BlendMethod::HardMix:
        default: {
            TVGERR("LOTTIE", "Non-Supported Blend Mode");
            return BlendMethod::Normal;
        }
    }
}


RGB24 LottieParser::getColor(const char *str)
{
    RGB24 color = {0, 0, 0};

    if (!str) return color;

    auto len = strlen(str);

    // some resource has empty color string, return a default color for those cases.
    if (len != 7 || str[0] != '#') return color;

    char tmp[3] = {'\0', '\0', '\0'};
    tmp[0] = str[1];
    tmp[1] = str[2];
    color.rgb[0] = uint8_t(strtol(tmp, nullptr, 16));

    tmp[0] = str[3];
    tmp[1] = str[4];
    color.rgb[1] = uint8_t(strtol(tmp, nullptr, 16));

    tmp[0] = str[5];
    tmp[1] = str[6];
    color.rgb[2] = uint8_t(strtol(tmp, nullptr, 16));

    return color;
}


FillRule LottieParser::getFillRule()
{
    switch (getInt()) {
        case 1: return FillRule::Winding;
        default: return FillRule::EvenOdd;
    }
}


CompositeMethod LottieParser::getMatteType()
{
    switch (getInt()) {
        case 1: return CompositeMethod::AlphaMask;
        case 2: return CompositeMethod::InvAlphaMask;
        case 3: return CompositeMethod::LumaMask;
        case 4: return CompositeMethod::InvLumaMask;
        default: return CompositeMethod::None;
    }
}


StrokeCap LottieParser::getStrokeCap()
{
    switch (getInt()) {
        case 1: return StrokeCap::Butt;
        case 2: return StrokeCap::Round;
        default: return StrokeCap::Square;
    }
}


StrokeJoin LottieParser::getStrokeJoin()
{
    switch (getInt()) {
        case 1: return StrokeJoin::Miter;
        case 2: return StrokeJoin::Round;
        default: return StrokeJoin::Bevel;
    }
}


void LottieParser::getValue(TextDocument& doc)
{
    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("s")) doc.size = getFloat();
        else if (KEY_AS("f")) doc.name = getStringCopy();
        else if (KEY_AS("t")) doc.text = getStringCopy();
        else if (KEY_AS("j")) doc.justify = getInt();
        else if (KEY_AS("tr")) doc.tracking = getInt();
        else if (KEY_AS("lh")) doc.height = getFloat();
        else if (KEY_AS("ls")) doc.shift = getFloat();
        else if (KEY_AS("fc")) getValue(doc.color);
        else if (KEY_AS("ps")) getValue(doc.bbox.pos);
        else if (KEY_AS("sz")) getValue(doc.bbox.size);
        else if (KEY_AS("sc")) getValue(doc.stroke.color);
        else if (KEY_AS("sw")) doc.stroke.width = getFloat();
        else if (KEY_AS("of")) doc.stroke.render = getBool();
        else skip(key);
    }
}


void LottieParser::getValue(PathSet& path)
{
    Array<Point> outs, ins, pts;
    bool closed = false;

    /* The shape object could be wrapped by a array
       if its part of the keyframe object */
    auto arrayWrapper = (peekType() == kArrayType) ? true : false;
    if (arrayWrapper) enterArray();

    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("i")) getValue(ins);
        else if (KEY_AS("o")) getValue(outs);
        else if (KEY_AS("v")) getValue(pts);
        else if (KEY_AS("c")) closed = getBool();
        else skip(key);
    }

    //exit properly from the array
    if (arrayWrapper) nextArrayValue();

    //valid path data?
    if (ins.empty() || outs.empty() || pts.empty()) return;
    if (ins.count != outs.count || outs.count != pts.count) return;

    //convert path
    auto out = outs.begin();
    auto in = ins.begin();
    auto pt = pts.begin();

    //Store manipulated results
    Array<Point> outPts;
    Array<PathCommand> outCmds;

    //Resuse the buffers
    outPts.data = path.pts;
    outPts.reserved = path.ptsCnt;
    outCmds.data = path.cmds;
    outCmds.reserved = path.cmdsCnt;

    size_t extra = closed ? 3 : 0;
    outPts.reserve(pts.count * 3 + 1 + extra);
    outCmds.reserve(pts.count + 2);

    outCmds.push(PathCommand::MoveTo);
    outPts.push(*pt);

    for (++pt, ++out, ++in; pt < pts.end(); ++pt, ++out, ++in) {
        outCmds.push(PathCommand::CubicTo);
        outPts.push(*(pt - 1) + *(out - 1));
        outPts.push(*pt + *in);
        outPts.push(*pt);
    }

    if (closed) {
        outPts.push(pts.last() + outs.last());
        outPts.push(pts.first() + ins.first());
        outPts.push(pts.first());
        outCmds.push(PathCommand::CubicTo);
        outCmds.push(PathCommand::Close);
    }

    path.pts = outPts.data;
    path.cmds = outCmds.data;
    path.ptsCnt = outPts.count;
    path.cmdsCnt = outCmds.count;

    outPts.data = nullptr;
    outCmds.data = nullptr;
}


void LottieParser::getValue(ColorStop& color)
{
    if (peekType() == kArrayType) enterArray();

    color.input = new Array<float>(context.gradient->colorStops.count);

    while (nextArrayValue()) color.input->push(getFloat());
}


void LottieParser::getValue(Array<Point>& pts)
{
    enterArray();
    while (nextArrayValue()) {
        enterArray();
        Point pt;
        getValue(pt);
        pts.push(pt);
    }
}


void LottieParser::getValue(uint8_t& val)
{
    if (peekType() == kArrayType) {
        enterArray();
        if (nextArrayValue()) val = (uint8_t)(getFloat() * 2.55f);
        //discard rest
        while (nextArrayValue()) getFloat();
    } else {
        val = (uint8_t)(getFloat() * 2.55f);
    }
}


void LottieParser::getValue(float& val)
{
    if (peekType() == kArrayType) {
        enterArray();
        if (nextArrayValue()) val = getFloat();
        //discard rest
        while (nextArrayValue()) getFloat();
    } else {
        val = getFloat();
    }
}


void LottieParser::getValue(Point& pt)
{
    int i = 0;
    auto ptr = (float*)(&pt);

    if (peekType() == kArrayType) enterArray();

    while (nextArrayValue()) {
        auto val = getFloat();
        if (i < 2) ptr[i++] = val;
    }
}


void LottieParser::getValue(RGB24& color)
{
    int i = 0;

    if (peekType() == kArrayType) enterArray();

    while (nextArrayValue()) {
        auto val = getFloat();
        if (i < 3) color.rgb[i++] = int32_t(lroundf(val * 255.0f));
    }

    //TODO: color filter?
}


void LottieParser::getInperpolatorPoint(Point& pt)
{
    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("x")) getValue(pt.x);
        else if (KEY_AS("y")) getValue(pt.y);
    }
}


template<typename T>
void LottieParser::parseSlotProperty(T& prop)
{
    while (auto key = nextObjectKey()) {
        if (KEY_AS("p")) parseProperty(prop);
        else skip(key);
    }
}


template<typename T>
bool LottieParser::parseTangent(const char *key, LottieVectorFrame<T>& value)
{
    if (KEY_AS("ti")) {
        value.hasTangent = true;
        getValue(value.inTangent);
    } else if (KEY_AS("to")) {
        value.hasTangent = true;
        getValue(value.outTangent);
    } else return false;

    return true;
}


template<typename T>
bool LottieParser::parseTangent(const char *key, LottieScalarFrame<T>& value)
{
    return false;
}


LottieInterpolator* LottieParser::getInterpolator(const char* key, Point& in, Point& out)
{
    char buf[20];

    if (!key) {
        snprintf(buf, sizeof(buf), "%.2f_%.2f_%.2f_%.2f", in.x, in.y, out.x, out.y);
        key = buf;
    }

    LottieInterpolator* interpolator = nullptr;

    //get a cached interpolator if it has any.
    for (auto i = comp->interpolators.begin(); i < comp->interpolators.end(); ++i) {
        if (!strncmp((*i)->key, key, sizeof(buf))) interpolator = *i;
    }

    //new interpolator
    if (!interpolator) {
        interpolator = static_cast<LottieInterpolator*>(malloc(sizeof(LottieInterpolator)));
        interpolator->set(key, in, out);
        comp->interpolators.push(interpolator);
    }

    return interpolator;
}


template<typename T>
void LottieParser::parseKeyFrame(T& prop)
{
    Point inTangent, outTangent;
    const char* interpolatorKey = nullptr;
    auto& frame = prop.newFrame();
    auto interpolator = false;

    enterObject();

    while (auto key = nextObjectKey()) {
        if (KEY_AS("i")) {
            interpolator = true;
            getInperpolatorPoint(inTangent);
        } else if (KEY_AS("o")) {
            getInperpolatorPoint(outTangent);
        } else if (KEY_AS("n")) {
            if (peekType() == kStringType) {
                interpolatorKey = getString();
            } else {
                enterArray();
                while (nextArrayValue()) {
                    if (!interpolatorKey) interpolatorKey = getString();
                    else skip(nullptr);
                }
            }
        } else if (KEY_AS("t")) {
            frame.no = getFloat();
        } else if (KEY_AS("s")) {
            getValue(frame.value);
        } else if (KEY_AS("e")) {
            //current end frame and the next start frame is duplicated,
            //We propagate the end value to the next frame to avoid having duplicated values.
            auto& frame2 = prop.nextFrame();
            getValue(frame2.value);
        } else if (parseTangent(key, frame)) {
            continue;
        } else if (KEY_AS("h")) {
            frame.hold = getInt();
        } else skip(key);
    }

    if (interpolator) {
        frame.interpolator = getInterpolator(interpolatorKey, inTangent, outTangent);
    }
}

template<typename T>
void LottieParser::parsePropertyInternal(T& prop)
{
    //single value property
    if (peekType() == kNumberType) {
        getValue(prop.value);
    //multi value property
    } else {
        //TODO: Here might be a single frame.
        //Can we figure out the frame number in advance?
        enterArray();
        while (nextArrayValue()) {
            //keyframes value
            if (peekType() == kObjectType) {
                parseKeyFrame(prop);
            //multi value property with no keyframes
            } else {
                getValue(prop.value);
                break;
            }
        }
        prop.prepare();
    }
}


template<LottieProperty::Type type, typename T>
void LottieParser::parseProperty(T& prop, LottieObject* obj)
{
    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("k")) parsePropertyInternal(prop);
        else if (obj && KEY_AS("sid")) {
            auto sid = getStringCopy();
            //append object if the slot already exists.
            for (auto slot = comp->slots.begin(); slot < comp->slots.end(); ++slot) {
                if (strcmp((*slot)->sid, sid)) continue;
                (*slot)->pairs.push({obj});
                return;
            }
            comp->slots.push(new LottieSlot(sid, obj, type));
        } else skip(key);
    }
}


LottieRect* LottieParser::parseRect()
{
    auto rect = new LottieRect;
    if (!rect) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("s")) parseProperty(rect->size);
        else if (KEY_AS("p")) parseProperty(rect->position);
        else if (KEY_AS("r")) parseProperty(rect->radius);
        else if (KEY_AS("nm")) rect->name = getStringCopy();
        else if (KEY_AS("hd")) rect->hidden = getBool();
        else skip(key);
    }
    rect->prepare();
    return rect;
}


LottieEllipse* LottieParser::parseEllipse()
{
    auto ellipse = new LottieEllipse;
    if (!ellipse) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) ellipse->name = getStringCopy();
        else if (KEY_AS("p")) parseProperty(ellipse->position);
        else if (KEY_AS("s")) parseProperty(ellipse->size);
        else if (KEY_AS("hd")) ellipse->hidden = getBool();
        else skip(key);
    }
    ellipse->prepare();
    return ellipse;
}


LottieTransform* LottieParser::parseTransform(bool ddd)
{
    auto transform = new LottieTransform;
    if (!transform) return nullptr;

    if (ddd) {
        transform->rotationEx = new LottieTransform::RotationEx;
        TVGLOG("LOTTIE", "3D transform(ddd) is not totally compatible.");
    }

    while (auto key = nextObjectKey()) {
        if (KEY_AS("p"))
        {
            enterObject();
            while (auto key = nextObjectKey()) {
                if (KEY_AS("k")) parsePropertyInternal(transform->position);
                else if (KEY_AS("s")) {
                    if (getBool()) transform->coords = new LottieTransform::SeparateCoord;
                //check separateCoord to figure out whether "x(expression)" / "x(coord)"
                } else if (transform->coords && KEY_AS("x")) {
                    parseProperty(transform->coords->x);
                } else if (transform->coords && KEY_AS("y")) {
                    parseProperty(transform->coords->y);
                } else skip(key);
            }
        }
        else if (KEY_AS("a")) parseProperty(transform->anchor);
        else if (KEY_AS("s")) parseProperty(transform->scale);
        else if (KEY_AS("r")) parseProperty(transform->rotation);
        else if (KEY_AS("o")) parseProperty(transform->opacity);
        else if (transform->rotationEx && KEY_AS("rx")) parseProperty(transform->rotationEx->x);
        else if (transform->rotationEx && KEY_AS("ry")) parseProperty(transform->rotationEx->y);
        else if (transform->rotationEx && KEY_AS("rz")) parseProperty(transform->rotation);
        else if (KEY_AS("nm")) transform->name = getStringCopy();
        //else if (KEY_AS("sk")) //TODO: skew
        //else if (KEY_AS("sa")) //TODO: skew axis
        else skip(key);
    }
    transform->prepare();
    return transform;
}


LottieSolidFill* LottieParser::parseSolidFill()
{
    auto fill = new LottieSolidFill;
    if (!fill) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) fill->name = getStringCopy();
        else if (KEY_AS("c")) parseProperty<LottieProperty::Type::Color>(fill->color, fill);
        else if (KEY_AS("o")) parseProperty<LottieProperty::Type::Opacity>(fill->opacity, fill);
        else if (KEY_AS("fillEnabled")) fill->hidden |= !getBool();
        else if (KEY_AS("r")) fill->rule = getFillRule();
        else if (KEY_AS("hd")) fill->hidden = getBool();
        else skip(key);
    }
    fill->prepare();
    return fill;
}


void LottieParser::parseStrokeDash(LottieStroke* stroke)
{
    enterArray();
    while (nextArrayValue()) {
        enterObject();
        int idx = 0;
        while (auto key = nextObjectKey()) {
            if (KEY_AS("n")) {
                auto style = getString();
                if (!strcmp("o", style)) idx = 0;           //offset
                else if (!strcmp("d", style)) idx = 1;      //dash
                else if (!strcmp("g", style)) idx = 2;      //gap
            } else if (KEY_AS("v")) {
                parseProperty(stroke->dash(idx));
            } else skip(key);
        }
    }
}


LottieSolidStroke* LottieParser::parseSolidStroke()
{
    auto stroke = new LottieSolidStroke;
    if (!stroke) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("c")) parseProperty<LottieProperty::Type::Color>(stroke->color, stroke);
        else if (KEY_AS("o")) parseProperty<LottieProperty::Type::Opacity>(stroke->opacity, stroke);
        else if (KEY_AS("w")) parseProperty<LottieProperty::Type::Float>(stroke->width, stroke);
        else if (KEY_AS("lc")) stroke->cap = getStrokeCap();
        else if (KEY_AS("lj")) stroke->join = getStrokeJoin();
        else if (KEY_AS("ml")) stroke->miterLimit = getFloat();
        else if (KEY_AS("nm")) stroke->name = getStringCopy();
        else if (KEY_AS("hd")) stroke->hidden = getBool();
        else if (KEY_AS("fillEnabled")) stroke->hidden |= !getBool();
        else if (KEY_AS("d")) parseStrokeDash(stroke);
        else skip(key);
    }
    stroke->prepare();
    return stroke;
}


 void LottieParser::getPathSet(LottiePathSet& path)
{
    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("k")) {
            if (peekType() == kArrayType) {
                enterArray();
                while (nextArrayValue()) parseKeyFrame(path);
            } else {
                getValue(path.value);
            }
        } else skip(key);
    }
}


LottiePath* LottieParser::parsePath()
{
    auto path = new LottiePath;
    if (!path) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) path->name = getStringCopy();
        else if (KEY_AS("ks")) getPathSet(path->pathset);
        else if (KEY_AS("hd")) path->hidden = getBool();
        else skip(key);
    }
    path->prepare();
    return path;
}


LottiePolyStar* LottieParser::parsePolyStar()
{
    auto star = new LottiePolyStar;
    if (!star) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) star->name = getStringCopy();
        else if (KEY_AS("p")) parseProperty(star->position);
        else if (KEY_AS("pt")) parseProperty(star->ptsCnt);
        else if (KEY_AS("ir")) parseProperty(star->innerRadius);
        else if (KEY_AS("is")) parseProperty(star->innerRoundness);
        else if (KEY_AS("or")) parseProperty(star->outerRadius);
        else if (KEY_AS("os")) parseProperty(star->outerRoundness);
        else if (KEY_AS("r")) parseProperty(star->rotation);
        else if (KEY_AS("sy")) star->type = (LottiePolyStar::Type) getInt();
        else if (KEY_AS("hd")) star->hidden = getBool();
        else skip(key);
    }
    star->prepare();
    return star;
}


LottieRoundedCorner* LottieParser::parseRoundedCorner()
{
    auto corner = new LottieRoundedCorner;
    if (!corner) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) corner->name = getStringCopy();
        else if (KEY_AS("r")) parseProperty(corner->radius);
        else if (KEY_AS("hd")) corner->hidden = getBool();
        else skip(key);
    }
    corner->prepare();
    return corner;
}


void LottieParser::parseGradient(LottieGradient* gradient, const char* key)
{
    context.gradient = gradient;

    if (KEY_AS("t")) gradient->id = getInt();
    else if (KEY_AS("o")) parseProperty<LottieProperty::Type::Opacity>(gradient->opacity, gradient);
    else if (KEY_AS("g"))
    {
        enterObject();
        while (auto key = nextObjectKey()) {
            if (KEY_AS("p")) gradient->colorStops.count = getInt();
            else if (KEY_AS("k")) parseProperty<LottieProperty::Type::ColorStop>(gradient->colorStops, gradient);
            else skip(key);
        }
    }
    else if (KEY_AS("s")) parseProperty<LottieProperty::Type::Point>(gradient->start, gradient);
    else if (KEY_AS("e")) parseProperty<LottieProperty::Type::Point>(gradient->end, gradient);
    else if (KEY_AS("h")) parseProperty<LottieProperty::Type::Float>(gradient->height, gradient);
    else if (KEY_AS("a")) parseProperty<LottieProperty::Type::Float>(gradient->angle, gradient);
    else skip(key);
}


LottieGradientFill* LottieParser::parseGradientFill()
{
    auto fill = new LottieGradientFill;
    if (!fill) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) fill->name = getStringCopy();
        else if (KEY_AS("r")) fill->rule = getFillRule();
        else if (KEY_AS("hd")) fill->hidden = getBool();
        else parseGradient(fill, key);
    }

    fill->prepare();

    return fill;
}


LottieGradientStroke* LottieParser::parseGradientStroke()
{
    auto stroke = new LottieGradientStroke;
    if (!stroke) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) stroke->name = getStringCopy();
        else if (KEY_AS("lc")) stroke->cap = getStrokeCap();
        else if (KEY_AS("lj")) stroke->join = getStrokeJoin();
        else if (KEY_AS("ml")) stroke->miterLimit = getFloat();
        else if (KEY_AS("hd")) stroke->hidden = getBool();
        else if (KEY_AS("w")) parseProperty(stroke->width);
        else if (KEY_AS("d")) parseStrokeDash(stroke);
        else parseGradient(stroke, key);
    }
    stroke->prepare();

    return stroke;
}


LottieTrimpath* LottieParser::parseTrimpath()
{
    auto trim = new LottieTrimpath;
    if (!trim) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) trim->name = getStringCopy();
        else if (KEY_AS("s")) parseProperty(trim->start);
        else if (KEY_AS("e")) parseProperty(trim->end);
        else if (KEY_AS("o")) parseProperty(trim->offset);
        else if (KEY_AS("m")) trim->type = static_cast<LottieTrimpath::Type>(getInt());
        else if (KEY_AS("hd")) trim->hidden = getBool();
        else skip(key);
    }
    trim->prepare();

    return trim;
}


LottieRepeater* LottieParser::parseRepeater()
{
    auto repeater = new LottieRepeater;
    if (!repeater) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) repeater->name = getStringCopy();
        else if (KEY_AS("c")) parseProperty(repeater->copies);
        else if (KEY_AS("o")) parseProperty(repeater->offset);
        else if (KEY_AS("m")) repeater->inorder = getInt();
        else if (KEY_AS("tr"))
        {
            enterObject();
            while (auto key = nextObjectKey()) {
                if (KEY_AS("a")) parseProperty(repeater->anchor);
                else if (KEY_AS("p")) parseProperty(repeater->position);
                else if (KEY_AS("r")) parseProperty(repeater->rotation);
                else if (KEY_AS("s")) parseProperty(repeater->scale);
                else if (KEY_AS("so")) parseProperty(repeater->startOpacity);
                else if (KEY_AS("eo")) parseProperty(repeater->endOpacity);
                else skip(key);
            }
        }
        else if (KEY_AS("hd")) repeater->hidden = getBool();
        else skip(key);
    }
    repeater->prepare();

    return repeater;
}


LottieObject* LottieParser::parseObject()
{
    auto type = getString();
    if (!type) return nullptr;

    if (!strcmp(type, "gr")) return parseGroup();
    else if (!strcmp(type, "rc")) return parseRect();
    else if (!strcmp(type, "el")) return parseEllipse();
    else if (!strcmp(type, "tr")) return parseTransform();
    else if (!strcmp(type, "fl")) return parseSolidFill();
    else if (!strcmp(type, "st")) return parseSolidStroke();
    else if (!strcmp(type, "sh")) return parsePath();
    else if (!strcmp(type, "sr")) return parsePolyStar();
    else if (!strcmp(type, "rd")) return parseRoundedCorner();
    else if (!strcmp(type, "gf")) return parseGradientFill();
    else if (!strcmp(type, "gs")) return parseGradientStroke();
    else if (!strcmp(type, "tm")) return parseTrimpath();
    else if (!strcmp(type, "rp")) return parseRepeater();
    else if (!strcmp(type, "mm")) TVGERR("LOTTIE", "MergePath(mm) is not supported yet");
    else if (!strcmp(type, "pb")) TVGERR("LOTTIE", "Puker/Bloat(pb) is not supported yet");
    else if (!strcmp(type, "tw")) TVGERR("LOTTIE", "Twist(tw) is not supported yet");
    else if (!strcmp(type, "op")) TVGERR("LOTTIE", "Offset Path(op) is not supported yet");
    else if (!strcmp(type, "zz")) TVGERR("LOTTIE", "Zig Zag(zz) is not supported yet");
    return nullptr;
}


void LottieParser::parseObject(Array<LottieObject*>& parent)
{
    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("ty")) {
            if (auto child = parseObject()) {
                if (child->hidden) delete(child);
                else parent.push(child);
            }
        } else skip(key);
    }
}


LottieImage* LottieParser::parseImage(const char* data, const char* subPath, bool embedded)
{
    //Used for Image Asset
    auto image = new LottieImage;

    //embeded image resource. should start with "data:"
    //header look like "data:image/png;base64," so need to skip till ','.
    if (embedded && !strncmp(data, "data:", 5)) {
        //figure out the mimetype
        auto mimeType = data + 11;
        auto needle = strstr(mimeType, ";");
        image->mimeType = strDuplicate(mimeType, needle - mimeType);
        //b64 data
        auto b64Data = strstr(data, ",") + 1;
        size_t length = strlen(data) - (b64Data - data);
        image->size = b64Decode(b64Data, length, &image->b64Data);
    //external image resource
    } else {
        auto len = strlen(dirName) + strlen(subPath) + strlen(data) + 1;
        image->path = static_cast<char*>(malloc(len));
        snprintf(image->path, len, "%s%s%s", dirName, subPath, data);
    }

    image->prepare();

    return image;
}


LottieObject* LottieParser::parseAsset()
{
    enterObject();

    LottieObject* obj = nullptr;
    char *id = nullptr;

    //Used for Image Asset
    const char* data = nullptr;
    const char* subPath = nullptr;
    auto embedded = false;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("id"))
        {
            if (peekType() == kStringType) {
                id = getStringCopy();
            } else {
                id = _int2str(getInt());
            }
        }
        else if (KEY_AS("layers")) obj = parseLayers();
        else if (KEY_AS("u")) subPath = getString();
        else if (KEY_AS("p")) data = getString();
        else if (KEY_AS("e")) embedded = getInt();
        else skip(key);
    }
    if (data) obj = parseImage(data, subPath, embedded);
    if (obj) obj->name = id;
    else free(id);
    return obj;
}


LottieFont* LottieParser::parseFont()
{
    enterObject();

    auto font = new LottieFont;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("fName")) font->name = getStringCopy();
        else if (KEY_AS("fFamily")) font->family = getStringCopy();
        else if (KEY_AS("fStyle")) font->style = getStringCopy();
        else if (KEY_AS("ascent")) font->ascent = getFloat();
        else if (KEY_AS("origin")) font->origin = (LottieFont::Origin) getInt();
        else skip(key);
    }
    return font;
}


void LottieParser::parseAssets()
{
    enterArray();
    while (nextArrayValue()) {
        auto asset = parseAsset();
        if (asset) comp->assets.push(asset);
        else TVGERR("LOTTIE", "Invalid Asset!");
    }
}

LottieMarker* LottieParser::parseMarker()
{
    enterObject();
    
    auto marker = new LottieMarker;
    
    while (auto key = nextObjectKey()) {
        if (KEY_AS("cm")) marker->name = getStringCopy();
        else if (KEY_AS("tm")) marker->time = getFloat();
        else if (KEY_AS("dr")) marker->duration = getFloat();
        else skip(key);
    }
    
    return marker;
}

void LottieParser::parseMarkers()
{
    enterArray();
    while (nextArrayValue()) {
        comp->markers.push(parseMarker());
    }
}

void LottieParser::parseChars(Array<LottieGlyph*>& glyphes)
{
    enterArray();
    while (nextArrayValue()) {
        enterObject();
        //a new glyph
        auto glyph = new LottieGlyph;
        while (auto key = nextObjectKey()) {
            if (KEY_AS("ch")) glyph->code = getStringCopy();
            else if (KEY_AS("size")) glyph->size = static_cast<uint16_t>(getFloat());
            else if (KEY_AS("style")) glyph->style = getStringCopy();
            else if (KEY_AS("w")) glyph->width = getFloat();
            else if (KEY_AS("fFamily")) glyph->family = getStringCopy();
            else if (KEY_AS("data"))
            {   //glyph shapes
                enterObject();
                while (auto key = nextObjectKey()) {
                    if (KEY_AS("shapes")) parseShapes(glyph->children);
                }
            } else skip(key);
        }
        glyph->prepare();
        glyphes.push(glyph);
    }
}

void LottieParser::parseFonts()
{
    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("list")) {
            enterArray();
            while (nextArrayValue()) {
                comp->fonts.push(parseFont());
            }
        } else skip(key);
    }
}


LottieObject* LottieParser::parseGroup()
{
    auto group = new LottieGroup;
    if (!group) return nullptr;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("nm")) {
            group->name = getStringCopy();
        } else if (KEY_AS("it")) {
            enterArray();
            while (nextArrayValue()) parseObject(group->children);
        } else skip(key);
    }
    if (group->children.empty()) {
        delete(group);
        return nullptr;
    }
    group->prepare();

    return group;
}


void LottieParser::parseTimeRemap(LottieLayer* layer)
{
    parseProperty(layer->timeRemap);
}


uint8_t LottieParser::getDirection()
{
    auto v = getInt();
    if (v == 1) return 0;
    if (v == 2) return 3;
    if (v == 3) return 2;
    return 0;
}

void LottieParser::parseShapes(Array<LottieObject*>& parent)
{
    uint8_t direction;

    enterArray();
    while (nextArrayValue()) {
        direction = 0;
        enterObject();
        while (auto key = nextObjectKey()) {
            if (KEY_AS("it")) {
                enterArray();
                while (nextArrayValue()) parseObject(parent);
            } else if (KEY_AS("d")) {
                direction = getDirection();
            } else if (KEY_AS("ty")) {
                if (auto child = parseObject()) {
                    if (child->hidden) delete(child);
                    else parent.push(child);
                    if (direction > 0) static_cast<LottieShape*>(child)->direction = direction;
                }
            } else skip(key);
        }
     }
}


void LottieParser::parseTextRange(LottieText* text)
{
    enterArray();
    while (nextArrayValue()) {
        enterObject();
        while (auto key = nextObjectKey()) {
            if (KEY_AS("a")) {  //text style
                enterObject();
                while (auto key = nextObjectKey()) {
                    if (KEY_AS("t")) parseProperty(text->spacing);
                    else skip(key);
                }
            } else skip(key);
        }
    }
}


void LottieParser::parseText(Array<LottieObject*>& parent)
{
    enterObject();

    auto text = new LottieText;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("d")) parseProperty<LottieProperty::Type::TextDoc>(text->doc, text);
        else if (KEY_AS("a")) parseTextRange(text);
        //else if (KEY_AS("p")) TVGLOG("LOTTIE", "Text Follow Path (p) is not supported"); 
        //else if (KEY_AS("m")) TVGLOG("LOTTIE", "Text Alignment Option (m) is not supported");
        else skip(key);
    }

    text->prepare();
    parent.push(text);
}


void LottieParser::getLayerSize(float& val)
{
    if (val == 0.0f) {
        val = getFloat();
    } else {
        //layer might have both w(width) & sw(solid color width)
        //override one if the a new size is smaller.
        auto w = getFloat();
        if (w < val) val = w;
    }
}

LottieMask* LottieParser::parseMask()
{
    auto mask = new LottieMask;
    if (!mask) return nullptr;

    enterObject();
    while (auto key = nextObjectKey()) {
        if (KEY_AS("inv")) mask->inverse = getBool();
        else if (KEY_AS("mode")) mask->method = getMaskMethod(mask->inverse);
        else if (KEY_AS("pt")) getPathSet(mask->pathset);
        else if (KEY_AS("o")) parseProperty(mask->opacity);
        else skip(key);
    }

    return mask;
}


void LottieParser::parseMasks(LottieLayer* layer)
{
    enterArray();
    while (nextArrayValue()) {
        auto mask = parseMask();
        layer->masks.push(mask);
    }
}


LottieLayer* LottieParser::parseLayer()
{
    auto layer = new LottieLayer;
    if (!layer) return nullptr;

    layer->comp = comp;
    context.layer = layer;

    auto ddd = false;

    enterObject();

    while (auto key = nextObjectKey()) {
        if (KEY_AS("ddd")) ddd = getInt();  //3d layer
        else if (KEY_AS("ind")) layer->id = getInt();
        else if (KEY_AS("ty")) layer->type = (LottieLayer::Type) getInt();
        else if (KEY_AS("nm")) layer->name = getStringCopy();
        else if (KEY_AS("sr")) layer->timeStretch = getFloat();
        else if (KEY_AS("ks"))
        {
            enterObject();
            layer->transform = parseTransform(ddd);
        }
        else if (KEY_AS("ao")) layer->autoOrient = getInt();
        else if (KEY_AS("shapes")) parseShapes(layer->children);
        else if (KEY_AS("ip")) layer->inFrame = getFloat();
        else if (KEY_AS("op")) layer->outFrame = getFloat();
        else if (KEY_AS("st")) layer->startFrame = getFloat();
        else if (KEY_AS("bm")) layer->blendMethod = getBlendMethod();
        else if (KEY_AS("parent")) layer->pid = getInt();
        else if (KEY_AS("tm")) parseTimeRemap(layer);
        else if (KEY_AS("w") || KEY_AS("sw")) getLayerSize(layer->w);
        else if (KEY_AS("h") || KEY_AS("sh")) getLayerSize(layer->h);
        else if (KEY_AS("sc")) layer->color = getColor(getString());
        else if (KEY_AS("tt")) layer->matte.type = getMatteType();
        else if (KEY_AS("masksProperties")) parseMasks(layer);
        else if (KEY_AS("hd")) layer->hidden = getBool();
        else if (KEY_AS("refId")) layer->refId = getStringCopy();
        else if (KEY_AS("td")) layer->matteSrc = getInt();      //used for matte layer
        else if (KEY_AS("t")) parseText(layer->children);
        else if (KEY_AS("ef"))
        {
            TVGERR("LOTTIE", "layer effect(ef) is not supported!");
            skip(key);
        }
        else skip(key);
    }

    //Not a valid layer
    if (!layer->transform) {
        delete(layer);
        return nullptr;
    }

    layer->prepare();

    return layer;
}


LottieLayer* LottieParser::parseLayers()
{
    auto root = new LottieLayer;
    if (!root) return nullptr;

    root->type = LottieLayer::Precomp;
    root->comp = comp;

    enterArray();
    while (nextArrayValue()) {
        if (auto layer = parseLayer()) {
            if (layer->matte.type == CompositeMethod::None) {
                root->children.push(layer);
            } else {
                //matte source must be located in the right previous.
                auto matte = static_cast<LottieLayer*>(root->children.last());
                if (matte->matteSrc) {
                    layer->matte.target = matte;
                } else {
                    TVGLOG("LOTTIE", "Matte Source(%s) is not designated?", matte->name);
                }
                root->children.last() = layer;
            }
        }
    }
    root->prepare();
    return root;
}


void LottieParser::postProcess(Array<LottieGlyph*>& glyphes)
{
    //aggregate font characters
    for (uint32_t g = 0; g < glyphes.count; ++g) {
        auto glyph = glyphes[g];
        for (uint32_t i = 0; i < comp->fonts.count; ++i) {
            auto& font = comp->fonts[i];
            if (!strcmp(font->family, glyph->family) && !strcmp(font->style, glyph->style)) {
                font->chars.push(glyph);
                free(glyph->family);
                free(glyph->style);
                break;
            }
        }
    }
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

const char* LottieParser::sid(bool first)
{
    if (first) {
        //verify json
        if (!parseNext()) return nullptr;
        enterObject();
    }
    return nextObjectKey();
}


bool LottieParser::apply(LottieSlot* slot)
{
    enterObject();

    //OPTIMIZE: we can create the property directly, without object
    LottieObject* obj = nullptr;  //slot object

    switch (slot->type) {
        case LottieProperty::Type::ColorStop: {
            obj = new LottieGradient;
            context.gradient = static_cast<LottieGradient*>(obj);
            parseSlotProperty(static_cast<LottieGradient*>(obj)->colorStops);
            break;
        }
        case LottieProperty::Type::Color: {
            obj = new LottieSolid;
            parseSlotProperty(static_cast<LottieSolid*>(obj)->color);
            break;
        }
        case LottieProperty::Type::TextDoc: {
            obj = new LottieText;
            parseSlotProperty(static_cast<LottieText*>(obj)->doc);
            break;
        }
        default: break;
    }

    if (!obj || Invalid()) return false;

    slot->assign(obj);

    delete(obj);

    return true;
}


bool LottieParser::parse()
{
    //verify json.
    if (!parseNext()) return false;

    enterObject();

    if (comp) delete(comp);
    comp = new LottieComposition;
    if (!comp) return false;

    Array<LottieGlyph*> glyphes;

    while (auto key = nextObjectKey()) {
        if (KEY_AS("v")) comp->version = getStringCopy();
        else if (KEY_AS("fr")) comp->frameRate = getFloat();
        else if (KEY_AS("ip")) comp->startFrame = getFloat();
        else if (KEY_AS("op")) comp->endFrame = getFloat();
        else if (KEY_AS("w")) comp->w = getFloat();
        else if (KEY_AS("h")) comp->h = getFloat();
        else if (KEY_AS("nm")) comp->name = getStringCopy();
        else if (KEY_AS("assets")) parseAssets();
        else if (KEY_AS("layers")) comp->root = parseLayers();
        else if (KEY_AS("fonts")) parseFonts();
        else if (KEY_AS("chars")) parseChars(glyphes);
        else if (KEY_AS("markers")) parseMarkers();
        else skip(key);
    }

    if (Invalid() || !comp->root) {
        delete(comp);
        return false;
    }

    comp->root->inFrame = comp->startFrame;
    comp->root->outFrame = comp->endFrame;

    postProcess(glyphes);

    return true;
}
