#include "layoutcontext.h"
#include "parser.h"

#include "maskelement.h"
#include "clippathelement.h"
#include "paintelement.h"
#include "markerelement.h"

#include <cmath>

using namespace lunasvg;

LayoutObject::LayoutObject(LayoutId id)
    : id(id)
{
}

LayoutObject::~LayoutObject()
{
}

void LayoutObject::render(RenderState&) const
{
}

LayoutContainer::LayoutContainer()
{
}

LayoutObject* LayoutContainer::addChild(std::unique_ptr<LayoutObject> child)
{
    children.push_back(std::move(child));
    return &*children.back();
}

LayoutClipPath::LayoutClipPath()
    : LayoutObject(LayoutId::ClipPath)
{
}

void LayoutClipPath::apply(RenderState& state) const
{
    RenderState newState;
    newState.mode = RenderMode::Clipping;
    newState.canvas = Canvas::create(state.canvas->width(), state.canvas->height());
    newState.matrix = state.matrix;
    if(units == Units::ObjectBoundingBox)
    {
        newState.matrix.translate(state.box.x, state.box.y);
        newState.matrix.scale(state.box.w, state.box.h);
    }

    newState.matrix.premultiply(transform);
    for(auto& child : children)
        child->render(newState);

    if(clipper != nullptr)
        clipper->apply(newState);

    state.canvas->blend(*newState.canvas, BlendMode::Dst_In, 1.0);
}

LayoutMask::LayoutMask()
    : LayoutObject(LayoutId::Mask)
{
}

void LayoutMask::apply(RenderState& state) const
{
    RenderState newState;
    newState.canvas = Canvas::create(state.canvas->width(), state.canvas->height());
    newState.matrix = state.matrix;
    if(contentUnits == Units::ObjectBoundingBox)
    {
        newState.matrix.translate(state.box.x, state.box.y);
        newState.matrix.scale(state.box.w, state.box.h);
    }

    for(auto& child : children)
        child->render(newState);

    if(clipper != nullptr)
        clipper->apply(newState);

    if(masker != nullptr)
        masker->apply(newState);

    newState.canvas->luminance();
    state.canvas->blend(*newState.canvas, BlendMode::Dst_In, opacity);
}

LayoutGroup::LayoutGroup()
    : LayoutObject(LayoutId::Group)
{
}

void LayoutGroup::render(RenderState& state) const
{
    RenderState newState;
    newState.mode = state.mode;
    newState.matrix = transform * state.matrix;
    newState.beginGroup(state, clipper, masker, opacity);

    for(auto& child : children)
        child->render(newState);

    newState.endGroup(state, clipper, masker, opacity);
    state.updateBoundingBox(newState);
}

LayoutMarker::LayoutMarker()
    : LayoutObject(LayoutId::Marker)
{
}

void LayoutMarker::apply(RenderState& state, const Point& origin, double angle, double strokeWidth) const
{
    RenderState newState;
    newState.mode = state.mode;
    newState.matrix = state.matrix;
    newState.matrix.translate(origin.x, origin.y);
    if(orient.type() == MarkerOrient::Auto)
        newState.matrix.rotate(angle);
    else
        newState.matrix.rotate(orient.value());

    if(units == MarkerUnits::StrokeWidth)
        newState.matrix.scale(strokeWidth, strokeWidth);

    newState.matrix.translate(-refX, -refY);
    newState.matrix.premultiply(transform);

    newState.beginGroup(state, clipper, masker, opacity);

    for(auto& child : children)
        child->render(newState);

    newState.endGroup(state, clipper, masker, opacity);
    state.updateBoundingBox(newState);
}

LayoutPaint::LayoutPaint(LayoutId id)
    : LayoutObject(id)
{
}

LayoutGradient::LayoutGradient(LayoutId id)
    : LayoutPaint(id)
{
}

LayoutLinearGradient::LayoutLinearGradient()
    : LayoutGradient(LayoutId::LinearGradient)
{
}

void LayoutLinearGradient::apply(RenderState& state) const
{
    Transform matrix;
    if(units == Units::ObjectBoundingBox)
    {
        matrix.translate(state.box.x, state.box.y);
        matrix.scale(state.box.w, state.box.h);
    }

    LinearGradientValues values{x1, y1, x2, y2};
    state.canvas->setGradient(values, transform * matrix, spreadMethod, stops);
}

LayoutRadialGradient::LayoutRadialGradient()
    : LayoutGradient(LayoutId::RadialGradient)
{
}

void LayoutRadialGradient::apply(RenderState& state) const
{
    Transform matrix;
    if(units == Units::ObjectBoundingBox)
    {
        matrix.translate(state.box.x, state.box.y);
        matrix.scale(state.box.w, state.box.h);
    }

    RadialGradientValues values{cx, cy, r, fx, fy};
    state.canvas->setGradient(values, transform * matrix, spreadMethod, stops);
}

LayoutPattern::LayoutPattern()
    : LayoutPaint(LayoutId::Pattern)
{
}

void LayoutPattern::apply(RenderState& state) const
{
    Rect b{x, y, width, height};
    if(units == Units::ObjectBoundingBox)
    {
        b.x = b.x * state.box.w + state.box.x;
        b.y = b.y * state.box.h + state.box.y;
        b.w = b.w * state.box.w;
        b.h = b.h * state.box.h;
    }

    auto scalex = std::sqrt(state.matrix.m00 * state.matrix.m00 + state.matrix.m01 * state.matrix.m01);
    auto scaley = std::sqrt(state.matrix.m10 * state.matrix.m10 + state.matrix.m11 * state.matrix.m11);

    auto width = static_cast<std::uint32_t>(std::ceil(b.w * scalex));
    auto height = static_cast<std::uint32_t>(std::ceil(b.h * scaley));

    RenderState newState;
    newState.canvas = Canvas::create(width, height);
    newState.matrix.scale(scalex, scaley);

    if(!viewBox.empty())
        newState.matrix.premultiply(preserveAspectRatio.getMatrix(Rect{0, 0, b.w, b.h}, viewBox));
    else if(contentUnits == Units::ObjectBoundingBox)
        newState.matrix.scale(state.box.w, state.box.h);

    for(auto& child : children)
        child->render(newState);

    Transform matrix{1.0/scalex, 0, 0, 1.0/scaley, b.x, b.y};
    state.canvas->setPattern(*newState.canvas, matrix * transform, TileMode::Tiled);
}

LayoutSolidColor::LayoutSolidColor()
    : LayoutPaint(LayoutId::SolidColor)
{
}

void LayoutSolidColor::apply(RenderState& state) const
{
    state.canvas->setColor(color);
}

void FillData::render(RenderState& state, const Path& path) const
{
    if(opacity == 0.0 || (painter == nullptr && color.isNone()))
        return;

    if(painter == nullptr)
        state.canvas->setColor(color);
    else
        painter->apply(state);

    state.canvas->setMatrix(state.matrix);
    state.canvas->setOpacity(opacity);
    state.canvas->setWinding(fillRule);
    state.canvas->fill(path);
}

void StrokeData::render(RenderState& state, const Path& path) const
{
    if(opacity == 0.0 || (painter == nullptr && color.isNone()))
        return;

    if(painter == nullptr)
        state.canvas->setColor(color);
    else
        painter->apply(state);

    state.canvas->setMatrix(state.matrix);
    state.canvas->setOpacity(opacity);
    state.canvas->setLineWidth(width);
    state.canvas->setMiterlimit(miterlimit);
    state.canvas->setLineCap(cap);
    state.canvas->setLineJoin(join);
    state.canvas->setDash(dash);
    state.canvas->stroke(path);
}

MarkerPosition::MarkerPosition(const LayoutMarker* marker, const Point& origin, double angle)
    : marker(marker), origin(origin), angle(angle)
{
}

void MarkerPosition::render(RenderState& state, double strokeWidth) const
{
    marker->apply(state, origin, angle, strokeWidth);
}

LayoutShape::LayoutShape()
    : LayoutObject(LayoutId::Shape)
{
}

void LayoutShape::render(RenderState& state) const
{
    RenderState newState;
    newState.mode = state.mode;
    newState.matrix = transform * state.matrix;
    newState.box = box;
    if(visibility == Visibility::Hidden)
    {
        state.updateBoundingBox(newState);
        return;
    }

    newState.beginGroup(state, clipper, masker, 1.0);

    if(newState.mode == RenderMode::Display)
    {
        fillData.render(newState, path);
        strokeData.render(newState, path);
    }
    else if(newState.mode == RenderMode::Clipping)
    {
        newState.canvas->setMatrix(newState.matrix);
        newState.canvas->setColor(Color::Black);
        newState.canvas->setOpacity(1.0);
        newState.canvas->setWinding(clipRule);
        newState.canvas->fill(path);
    }

    for(auto& marker : markers)
        marker.render(newState, strokeData.width);

    newState.endGroup(state, clipper, masker, 1.0);
    state.updateBoundingBox(newState);
}

LayoutRoot::LayoutRoot()
    : LayoutObject(LayoutId::Root)
{
}

void LayoutRoot::render(RenderState& state) const
{
    RenderState newState;
    newState.mode = state.mode;
    newState.matrix = viewTransform * transform * state.matrix;
    newState.beginGroup(state, clipper, masker, opacity);

    for(auto& child : children)
        child->render(newState);

    newState.endGroup(state, clipper, masker, opacity);
    state.updateBoundingBox(newState);
}

void RenderState::beginGroup(RenderState& state, const LayoutClipPath* clipper, const LayoutMask* masker, double opacity)
{
    if(mode == RenderMode::Bounding)
        return;

    if(clipper || (mode == RenderMode::Display && (masker || opacity < 1.0)))
        canvas = Canvas::create(state.canvas->width(), state.canvas->height());
    else
        canvas = state.canvas;
}

void RenderState::endGroup(RenderState& state, const LayoutClipPath* clipper, const LayoutMask* masker, double opacity)
{
    if(mode == RenderMode::Bounding || !(masker || clipper || opacity < 1.0))
        return;

    if(clipper && (mode == RenderMode::Display || mode == RenderMode::Clipping))
        clipper->apply(*this);

    if(masker && mode == RenderMode::Display)
        masker->apply(*this);

    state.canvas->blend(*canvas, BlendMode::Src_Over, mode == RenderMode::Display ? opacity : 1.0);
}

void RenderState::updateBoundingBox(const RenderState& state)
{
    auto matrix = state.matrix * this->matrix.inverted();
    auto box = matrix.map(state.box);

    if(this->box.empty())
    {
        this->box = box;
        return;
    }

    auto l = std::min(this->box.x, box.x);
    auto t = std::min(this->box.y, box.y);
    auto r = std::max(this->box.x + this->box.w, box.x + box.w);
    auto b = std::max(this->box.y + this->box.h, box.y + box.h);

    this->box.x = l;
    this->box.y = t;
    this->box.w = r - l;
    this->box.h = b - t;
}

LayoutContext::LayoutContext(const ParseDocument* document, LayoutRoot* root)
    : m_document(document), m_root(root)
{
}

Element* LayoutContext::getElementById(const std::string& id) const
{
    return m_document->getElementById(id);
}

LayoutObject* LayoutContext::getResourcesById(const std::string& id) const
{
    auto it = m_resourcesCache.find(id);
    if(it == m_resourcesCache.end())
        return nullptr;
    return it->second;
}

LayoutObject* LayoutContext::addToResourcesCache(const std::string& id, std::unique_ptr<LayoutObject> resources)
{
    if(resources == nullptr)
        return nullptr;

    m_resourcesCache.emplace(id, resources.get());
    return m_root->addChild(std::move(resources));
}

LayoutMask* LayoutContext::getMasker(const std::string& id)
{
    if(id.empty())
        return nullptr;

    auto ref = getResourcesById(id);
    if(ref && ref->id == LayoutId::Mask)
        return static_cast<LayoutMask*>(ref);

    auto element = getElementById(id);
    if(element == nullptr || element->id != ElementId::Mask)
        return nullptr;

    auto masker = static_cast<MaskElement*>(element)->getMasker(this);
    return static_cast<LayoutMask*>(addToResourcesCache(id, std::move(masker)));
}

LayoutClipPath* LayoutContext::getClipper(const std::string& id)
{
    if(id.empty())
        return nullptr;

    auto ref = getResourcesById(id);
    if(ref && ref->id == LayoutId::ClipPath)
        return static_cast<LayoutClipPath*>(ref);

    auto element = getElementById(id);
    if(element == nullptr || element->id != ElementId::ClipPath)
        return nullptr;

    auto clipper = static_cast<ClipPathElement*>(element)->getClipper(this);
    return static_cast<LayoutClipPath*>(addToResourcesCache(id, std::move(clipper)));
}

LayoutMarker* LayoutContext::getMarker(const std::string& id)
{
    if(id.empty())
        return nullptr;

    auto ref = getResourcesById(id);
    if(ref && ref->id == LayoutId::Marker)
        return static_cast<LayoutMarker*>(ref);

    auto element = getElementById(id);
    if(element == nullptr || element->id != ElementId::Marker)
        return nullptr;

    auto marker = static_cast<MarkerElement*>(element)->getMarker(this);
    return static_cast<LayoutMarker*>(addToResourcesCache(id, std::move(marker)));
}

LayoutPaint* LayoutContext::getPainter(const std::string& id)
{
    if(id.empty())
        return nullptr;

    auto ref = getResourcesById(id);
    if(ref && (ref->id == LayoutId::LinearGradient || ref->id == LayoutId::RadialGradient || ref->id == LayoutId::Pattern || ref->id == LayoutId::SolidColor))
        return static_cast<LayoutPaint*>(ref);

    auto element = getElementById(id);
    if(element == nullptr || !(element->id == ElementId::LinearGradient || element->id == ElementId::RadialGradient || element->id == ElementId::Pattern || element->id == ElementId::SolidColor))
        return nullptr;

    auto painter = static_cast<PaintElement*>(element)->getPainter(this);
    return static_cast<LayoutPaint*>(addToResourcesCache(id, std::move(painter)));
}

FillData LayoutContext::fillData(const StyledElement* element)
{
    auto fill = element->fill();
    if(fill.isNone())
        return FillData{};

    FillData fillData;
    fillData.painter = getPainter(fill.ref());
    fillData.color = fill.color();
    fillData.opacity = element->opacity() * element->fill_opacity();
    fillData.fillRule = element->fill_rule();
    return fillData;
}

DashData LayoutContext::dashData(const StyledElement* element)
{
    auto dasharray = element->stroke_dasharray();
    if(dasharray.empty())
        return DashData{};

    LengthContext lengthContex(element);
    DashArray dashes;
    for(auto& dash : dasharray)
    {
        auto value = lengthContex.valueForLength(dash, LengthMode::Both);
        dashes.push_back(value);
    }

    auto num_dash = dashes.size();
    if(num_dash % 2)
        num_dash *= 2;

    DashData dashData;
    dashData.array.resize(num_dash);
    double sum = 0.0;
    for(std::size_t i = 0;i < num_dash;i++)
    {
        dashData.array[i] = dashes[i % dashes.size()];
        sum += dashData.array[i];
    }

    if(sum == 0.0)
        return DashData{};

    auto offset = lengthContex.valueForLength(element->stroke_dashoffset(), LengthMode::Both);
    dashData.offset = std::fmod(offset, sum);
    if(dashData.offset < 0.0)
        dashData.offset += sum;

    return dashData;
}

StrokeData LayoutContext::strokeData(const StyledElement* element)
{
    auto stroke = element->stroke();
    if(stroke.isNone())
        return StrokeData{};

    LengthContext lengthContex(element);
    StrokeData strokeData;
    strokeData.painter = getPainter(stroke.ref());
    strokeData.color = stroke.color();
    strokeData.opacity = element->opacity() * element->stroke_opacity();
    strokeData.width = lengthContex.valueForLength(element->stroke_width(), LengthMode::Both);
    strokeData.miterlimit = element->stroke_miterlimit();
    strokeData.cap = element->stroke_linecap();
    strokeData.join = element->stroke_linejoin();
    strokeData.dash = dashData(element);
    return strokeData;
}
