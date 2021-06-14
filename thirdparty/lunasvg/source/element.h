#ifndef ELEMENT_H
#define ELEMENT_H

#include <memory>
#include <list>
#include <map>

#include "property.h"

namespace lunasvg {

enum class ElementId
{
    Unknown = 0,
    Circle,
    ClipPath,
    Defs,
    Ellipse,
    G,
    Line,
    LinearGradient,
    Marker,
    Mask,
    Path,
    Pattern,
    Polygon,
    Polyline,
    RadialGradient,
    Rect,
    SolidColor,
    Stop,
    Svg,
    Symbol,
    Use
};

enum class PropertyId
{
    Unknown = 0,
    Clip_Path,
    Clip_Rule,
    ClipPathUnits,
    Color,
    Cx,
    Cy,
    D,
    Display,
    Fill,
    Fill_Opacity,
    Fill_Rule,
    Fx,
    Fy,
    GradientTransform,
    GradientUnits,
    Height,
    Href,
    Id,
    Marker_End,
    Marker_Mid,
    Marker_Start,
    MarkerHeight,
    MarkerUnits,
    MarkerWidth,
    Mask,
    MaskContentUnits,
    MaskUnits,
    Offset,
    Opacity,
    Orient,
    PatternContentUnits,
    PatternTransform,
    PatternUnits,
    Points,
    PreserveAspectRatio,
    R,
    RefX,
    RefY,
    Rx,
    Ry,
    Solid_Color,
    Solid_Opacity,
    SpreadMethod,
    Stop_Color,
    Stop_Opacity,
    Stroke,
    Stroke_Dasharray,
    Stroke_Dashoffset,
    Stroke_Linecap,
    Stroke_Linejoin,
    Stroke_Miterlimit,
    Stroke_Opacity,
    Stroke_Width,
    Style,
    Transform,
    ViewBox,
    Visibility,
    Width,
    X,
    X1,
    X2,
    Y,
    Y1,
    Y2
};

class LayoutContext;
class LayoutContainer;
class Element;

class Node
{
public:
    Node() = default;
    virtual ~Node() = default;

    virtual bool isText() const { return false; }
    virtual void layout(LayoutContext*, LayoutContainer*) const;
    virtual std::unique_ptr<Node> clone() const = 0;

public:
    Element* parent = nullptr;
};

class TextNode : public Node
{
public:
    TextNode() = default;

    bool isText() const { return true; }
    std::unique_ptr<Node> clone() const;

public:
    std::string text;
};

using NodeList = std::list<std::unique_ptr<Node>>;
using PropertyMap = std::map<PropertyId, std::string>;

class Element : public Node
{
public:
    Element(ElementId id);

    void set(PropertyId id, const std::string& value);
    const std::string& get(PropertyId id) const;
    const std::string& find(PropertyId id) const;
    bool has(PropertyId id) const;

    Node* addChild(std::unique_ptr<Node> child);
    Rect nearestViewBox() const;
    void layoutChildren(LayoutContext* context, LayoutContainer* current) const;

    template<typename T>
    std::unique_ptr<T> cloneElement() const
    {
        auto element = std::make_unique<T>();
        element->properties = properties;
        for(auto& child : children)
            element->addChild(child->clone());
        return element;
    }

public:
    ElementId id;
    NodeList children;
    PropertyMap properties;
};

} // namespace lunasvg

#endif // ELEMENT_H
