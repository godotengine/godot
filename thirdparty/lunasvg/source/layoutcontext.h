#ifndef LAYOUTCONTEXT_H
#define LAYOUTCONTEXT_H

#include "property.h"
#include "canvas.h"

#include <list>
#include <map>

namespace lunasvg {

enum class LayoutId
{
    Root,
    Group,
    Shape,
    Mask,
    ClipPath,
    Marker,
    LinearGradient,
    RadialGradient,
    Pattern,
    SolidColor
};

class RenderState;

class LayoutObject
{
public:
    LayoutObject(LayoutId id);
    virtual ~LayoutObject();
    virtual void render(RenderState&) const;

public:
    LayoutId id;
};

using LayoutList = std::list<std::unique_ptr<LayoutObject>>;

class LayoutContainer
{
public:
    LayoutContainer();

    LayoutObject* addChild(std::unique_ptr<LayoutObject> child);

public:
    LayoutList children;
};

class LayoutClipPath : public LayoutObject, public LayoutContainer
{
public:
    LayoutClipPath();

    void apply(RenderState& state) const;

public:
    Units units{Units::UserSpaceOnUse};
    Transform transform;
    const LayoutClipPath* clipper{nullptr};
};

class LayoutMask : public LayoutObject, public LayoutContainer
{
public:
    LayoutMask();

    void apply(RenderState& state) const;

public:
    double x{0};
    double y{0};
    double width{0};
    double height{0};
    Units units{Units::ObjectBoundingBox};
    Units contentUnits{Units::UserSpaceOnUse};
    double opacity{1};
    const LayoutMask* masker{nullptr};
    const LayoutClipPath* clipper{nullptr};
};

class LayoutGroup : public LayoutObject, public LayoutContainer
{
public:
    LayoutGroup();

    void render(RenderState& state) const;

public:
    Transform transform;
    double opacity{1};
    const LayoutMask* masker{nullptr};
    const LayoutClipPath* clipper{nullptr};
};

class LayoutMarker : public LayoutObject, public LayoutContainer
{
public:
    LayoutMarker();

    void apply(RenderState& state, const Point& origin, double angle, double strokeWidth) const;

public:
    double refX{0};
    double refY{0};
    Transform transform;
    Angle orient;
    MarkerUnits units{MarkerUnits::StrokeWidth};
    double opacity{1};
    const LayoutMask* masker{nullptr};
    const LayoutClipPath* clipper{nullptr};
};

class LayoutPaint : public LayoutObject
{
public:
    LayoutPaint(LayoutId id);

    virtual void apply(RenderState& state) const = 0;
};

class LayoutGradient : public LayoutPaint
{
public:
    LayoutGradient(LayoutId id);

public:
    Transform transform;
    SpreadMethod spreadMethod{SpreadMethod::Pad};
    Units units{Units::ObjectBoundingBox};
    GradientStops stops;
};

class LayoutLinearGradient : public LayoutGradient
{
public:
    LayoutLinearGradient();

    void apply(RenderState& state) const;

public:
    double x1{0};
    double y1{0};
    double x2{1};
    double y2{0};
};

class LayoutRadialGradient : public LayoutGradient
{
public:
    LayoutRadialGradient();

    void apply(RenderState& state) const;

public:
    double cx{0.5};
    double cy{0.5};
    double r{0.5};
    double fx{0};
    double fy{0};
};

class LayoutPattern : public LayoutPaint, public LayoutContainer
{
public:
    LayoutPattern();

    void apply(RenderState& state) const;

public:
    double x{0};
    double y{0};
    double width{0};
    double height{0};
    Transform transform;
    Units units{Units::ObjectBoundingBox};
    Units contentUnits{Units::UserSpaceOnUse};
    Rect viewBox;
    PreserveAspectRatio preserveAspectRatio;
};

class LayoutSolidColor : public LayoutPaint
{
public:
    LayoutSolidColor();

    void apply(RenderState& state) const;

public:
    Color color;
};

class FillData
{
public:
    FillData() = default;

    void render(RenderState& state, const Path& path) const;

public:
    const LayoutPaint* painter{nullptr};
    Color color{Color::Transparent};
    double opacity{0};
    WindRule fillRule{WindRule::NonZero};
};

class StrokeData
{
public:
    StrokeData() = default;

    void render(RenderState& state, const Path& path) const;

public:
    const LayoutPaint* painter{nullptr};
    Color color{Color::Transparent};
    double opacity{0};
    double width{1};
    double miterlimit{4};
    LineCap cap{LineCap::Butt};
    LineJoin join{LineJoin::Miter};
    DashData dash;
};

class MarkerPosition
{
public:
    MarkerPosition(const LayoutMarker* marker, const Point& origin, double angle);

    void render(RenderState& state, double strokeWidth) const;

public:
    const LayoutMarker* marker{nullptr};
    Point origin;
    double angle{0};
};

class LayoutShape : public LayoutObject
{
public:
    LayoutShape();

    void render(RenderState& state) const;

public:
    Path path;
    Rect box;
    Transform transform;
    FillData fillData;
    StrokeData strokeData;
    Visibility visibility{Visibility::Visible};
    WindRule clipRule{WindRule::NonZero};
    const LayoutMask* masker{nullptr};
    const LayoutClipPath* clipper{nullptr};
    std::vector<MarkerPosition> markers;
};

class LayoutRoot : public LayoutObject, public LayoutContainer
{
public:
    LayoutRoot();

    void render(RenderState& state) const;

public:
    double width{0};
    double height{0};
    Transform transform;
    Transform viewTransform;
    double opacity{1};
    const LayoutMask* masker{nullptr};
    const LayoutClipPath* clipper{nullptr};
};

enum class RenderMode
{
    Display,
    Clipping,
    Bounding
};

class RenderState
{
public:
    RenderState() = default;

    void beginGroup(RenderState& state, const LayoutClipPath* clipper, const LayoutMask* masker, double opacity);
    void endGroup(RenderState& state, const LayoutClipPath* clipper, const LayoutMask* masker, double opacity);
    void updateBoundingBox(const RenderState& state);

public:
    RenderMode mode{RenderMode::Display};
    std::shared_ptr<Canvas> canvas;
    Transform matrix;
    Rect box;
};

class ParseDocument;
class StyledElement;

class LayoutContext
{
public:
    LayoutContext(const ParseDocument* document, LayoutRoot* root);

    Element* getElementById(const std::string& id) const;
    LayoutObject* getResourcesById(const std::string& id) const;
    LayoutObject* addToResourcesCache(const std::string& id, std::unique_ptr<LayoutObject> resources);
    LayoutMask* getMasker(const std::string& id);
    LayoutClipPath* getClipper(const std::string& id);
    LayoutMarker* getMarker(const std::string& id);
    LayoutPaint* getPainter(const std::string& id);

    FillData fillData(const StyledElement* element);
    DashData dashData(const StyledElement* element);
    StrokeData strokeData(const StyledElement* element);

private:
    const ParseDocument* m_document;
    LayoutRoot* m_root;
    std::map<std::string, LayoutObject*> m_resourcesCache;
};

} // namespace lunasvg

#endif // LAYOUTCONTEXT_H
