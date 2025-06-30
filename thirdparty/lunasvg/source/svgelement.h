#ifndef LUNASVG_SVGELEMENT_H
#define LUNASVG_SVGELEMENT_H

#include <string>
#include <forward_list>
#include <list>
#include <map>

#include "svgproperty.h"
#include "lunasvg.h"

namespace lunasvg {

class Document;
class SVGElement;
class SVGRootElement;

class SVGNode {
public:
    SVGNode(Document* document)
        : m_document(document)
    {}

    virtual ~SVGNode() = default;
    virtual bool isTextNode() const { return false; }
    virtual bool isElement() const { return false; }
    virtual bool isPaintElement() const { return false; }
    virtual bool isGraphicsElement() const { return false; }
    virtual bool isGeometryElement() const { return false; }
    virtual bool isTextPositioningElement() const { return false; }

    SVGRootElement* rootElement() const;
    Document* document() const { return m_document; }
    void setParent(SVGElement* parent) { m_parent = parent; }
    SVGElement* parent() const { return m_parent; }

    virtual std::unique_ptr<SVGNode> clone(bool deep) const = 0;

private:
    SVGNode(const SVGNode&) = delete;
    SVGNode& operator=(const SVGNode&) = delete;
    Document* m_document;
    SVGElement* m_parent = nullptr;
};

class SVGTextNode final : public SVGNode {
public:
    SVGTextNode(Document* document);

    bool isTextNode() const final { return true; }

    void setData(const std::string& data) { m_data = data; }
    const std::string& data() const { return m_data; }

    std::unique_ptr<SVGNode> clone(bool deep) const final;

private:
    std::string m_data;
};

class Attribute {
public:
    Attribute() = default;
    Attribute(int specificity, PropertyID id, std::string value)
        : m_specificity(specificity), m_id(id), m_value(std::move(value))
    {}

    int specificity() const { return m_specificity; }
    PropertyID id() const { return m_id; }
    const std::string& value() const { return m_value; }

private:
    int m_specificity;
    PropertyID m_id;
    std::string m_value;
};

using AttributeList = std::forward_list<Attribute>;

enum class ElementID : uint8_t {
    Unknown = 0,
    Star,
    Circle,
    ClipPath,
    Defs,
    Ellipse,
    G,
    Image,
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
    Stop,
    Style,
    Svg,
    Symbol,
    Text,
    Tspan,
    Use
};

ElementID elementid(const std::string_view& name);

using SVGNodeList = std::list<std::unique_ptr<SVGNode>>;
using SVGPropertyList = std::forward_list<SVGProperty*>;

class SVGMarkerElement;
class SVGClipPathElement;
class SVGMaskElement;
class SVGPaintElement;
class SVGLayoutState;
class SVGRenderState;

class SVGElement : public SVGNode {
public:
    static std::unique_ptr<SVGElement> create(Document* document, ElementID id);

    SVGElement(Document* document, ElementID id);
    virtual ~SVGElement() = default;

    bool hasAttribute(const std::string_view& name) const;
    const std::string& getAttribute(const std::string_view& name) const;
    bool setAttribute(const std::string_view& name, const std::string& value);

    const Attribute* findAttribute(PropertyID id) const;
    bool hasAttribute(PropertyID id) const;
    const std::string& getAttribute(PropertyID id) const;
    bool setAttribute(int specificity, PropertyID id, const std::string& value);
    void setAttributes(const AttributeList& attributes);
    bool setAttribute(const Attribute& attribute);

    virtual void parseAttribute(PropertyID id, const std::string& value);

    SVGElement* previousElement() const;
    SVGElement* nextElement() const;
    SVGNode* addChild(std::unique_ptr<SVGNode> child);
    SVGNode* firstChild() const;
    SVGNode* lastChild() const;

    ElementID id() const { return m_id; }
    const AttributeList& attributes() const { return m_attributes; }
    const SVGPropertyList& properties() const { return m_properties; }
    const SVGNodeList& children() const { return m_children; }

    virtual Transform localTransform() const { return Transform::Identity; }
    virtual Rect fillBoundingBox() const;
    virtual Rect strokeBoundingBox() const;
    virtual Rect paintBoundingBox() const;

    SVGMarkerElement* getMarker(const std::string_view& id) const;
    SVGClipPathElement* getClipper(const std::string_view& id) const;
    SVGMaskElement* getMasker(const std::string_view& id) const;
    SVGPaintElement* getPainter(const std::string_view& id) const;

    template<typename T>
    void transverse(T callback);

    void addProperty(SVGProperty& value);
    SVGProperty* getProperty(PropertyID id) const;
    Size currentViewportSize() const;
    float font_size() const { return m_font_size; }

    void cloneChildren(SVGElement* parentElement) const;
    std::unique_ptr<SVGNode> clone(bool deep) const final;

    virtual void build();

    virtual void layoutElement(const SVGLayoutState& state);
    void layoutChildren(SVGLayoutState& state);
    virtual void layout(SVGLayoutState& state);

    void renderChildren(SVGRenderState& state) const;
    virtual void render(SVGRenderState& state) const;

    bool isDisplayNone() const { return m_display == Display::None; }
    bool isOverflowHidden() const { return m_overflow == Overflow::Hidden; }
    bool isVisibilityHidden() const { return m_visibility != Visibility::Visible; }

    bool isHiddenElement() const;

    const SVGClipPathElement* clipper() const { return m_clipper; }
    const SVGMaskElement* masker() const { return m_masker; }
    float opacity() const { return m_opacity; }

    bool isElement() const final { return true; }

private:
    mutable Rect m_paintBoundingBox = Rect::Invalid;
    const SVGClipPathElement* m_clipper = nullptr;
    const SVGMaskElement* m_masker = nullptr;
    float m_opacity = 1.f;

    float m_font_size = 12.f;
    Display m_display = Display::Inline;
    Overflow m_overflow = Overflow::Visible;
    Visibility m_visibility = Visibility::Visible;

    ElementID m_id;
    AttributeList m_attributes;
    SVGPropertyList m_properties;
    SVGNodeList m_children;
};

inline const SVGElement* toSVGElement(const SVGNode* node)
{
    if(node && node->isElement())
        return static_cast<const SVGElement*>(node);
    return nullptr;
}

inline SVGElement* toSVGElement(SVGNode* node)
{
    if(node && node->isElement())
        return static_cast<SVGElement*>(node);
    return nullptr;
}

inline SVGElement* toSVGElement(const std::unique_ptr<SVGNode>& node)
{
    return toSVGElement(node.get());
}

template<typename T>
inline void SVGElement::transverse(T callback)
{
    if(!callback(this))
        return;
    for(const auto& child : m_children) {
        if(auto element = toSVGElement(child)) {
            element->transverse(callback);
        } else if(!callback(child.get())) {
            return;
        }
    }
}

class SVGStyleElement final : public SVGElement {
public:
    SVGStyleElement(Document* document);
};

class SVGFitToViewBox {
public:
    SVGFitToViewBox(SVGElement* element);

    const SVGRect& viewBox() const { return m_viewBox; }
    const SVGPreserveAspectRatio& preserveAspectRatio() const { return m_preserveAspectRatio; }
    Transform viewBoxToViewTransform(const Size& viewportSize) const;
    Rect getClipRect(const Size& viewportSize) const;

private:
    SVGRect m_viewBox;
    SVGPreserveAspectRatio m_preserveAspectRatio;
};

class SVGURIReference {
public:
    SVGURIReference(SVGElement* element);

    const SVGString& href() const { return m_href; }
    const std::string& hrefString() const { return m_href.value(); }
    SVGElement* getTargetElement(const Document* document) const;

private:
    SVGString m_href;
};

class SVGPaintServer {
public:
    SVGPaintServer() = default;
    SVGPaintServer(const SVGPaintElement* element, const Color& color, float opacity)
        : m_element(element), m_color(color), m_opacity(opacity)
    {}

    bool isRenderable() const { return m_opacity > 0.f && (m_element || m_color.alpha() > 0); }

    const SVGPaintElement* element() const { return m_element; }
    const Color& color() const { return  m_color; }
    float opacity() const { return m_opacity; }

    bool applyPaint(SVGRenderState& state) const;

private:
    const SVGPaintElement* m_element = nullptr;
    Color m_color = Color::Transparent;
    float m_opacity = 0.f;
};

class SVGGraphicsElement : public SVGElement {
public:
    SVGGraphicsElement(Document* document, ElementID id);

    bool isGraphicsElement() const final { return true; }

    const SVGTransform& transform() const { return m_transform; }
    Transform localTransform() const override { return m_transform.value(); }

    SVGPaintServer getPaintServer(const Paint& paint, float opacity) const;
    StrokeData getStrokeData(const SVGLayoutState& state) const;

private:
    SVGTransform m_transform;
};

class SVGSVGElement : public SVGGraphicsElement, public SVGFitToViewBox {
public:
    SVGSVGElement(Document* document);

    const SVGLength& x() const { return m_x; }
    const SVGLength& y() const { return m_y; }
    const SVGLength& width() const { return m_width; }
    const SVGLength& height() const { return m_height; }

    Transform localTransform() const override;
    void render(SVGRenderState& state) const override;

private:
    SVGLength m_x;
    SVGLength m_y;
    SVGLength m_width;
    SVGLength m_height;
};

class SVGRootElement final : public SVGSVGElement {
public:
    SVGRootElement(Document* document);

    float intrinsicWidth() const { return m_intrinsicWidth; }
    float intrinsicHeight() const { return m_intrinsicHeight; }

    void setNeedsLayout() { m_intrinsicWidth = -1.f; }
    bool needsLayout() const { return m_intrinsicWidth == -1.f; }

    SVGRootElement* updateLayout();

    SVGElement* getElementById(const std::string_view& id) const;
    void addElementById(const std::string& id, SVGElement* element);
    void layout(SVGLayoutState& state) final;

    void forceLayout();

private:
    std::map<std::string, SVGElement*, std::less<>> m_idCache;
    float m_intrinsicWidth{-1.f};
    float m_intrinsicHeight{-1.f};
};

class SVGUseElement final : public SVGGraphicsElement, public SVGURIReference {
public:
    SVGUseElement(Document* document);

    const SVGLength& x() const { return m_x; }
    const SVGLength& y() const { return m_y; }
    const SVGLength& width() const { return m_width; }
    const SVGLength& height() const { return m_height; }

    Transform localTransform() const final;
    void render(SVGRenderState& state) const final;
    void build() final;

private:
    std::unique_ptr<SVGElement> cloneTargetElement(SVGElement* targetElement);
    SVGLength m_x;
    SVGLength m_y;
    SVGLength m_width;
    SVGLength m_height;
};

class SVGImageElement final : public SVGGraphicsElement, public SVGURIReference {
public:
    SVGImageElement(Document* document);

    const SVGLength& x() const { return m_x; }
    const SVGLength& y() const { return m_y; }
    const SVGLength& width() const { return m_width; }
    const SVGLength& height() const { return m_height; }
    const SVGPreserveAspectRatio& preserveAspectRatio() const { return m_preserveAspectRatio; }
    const Bitmap& image() const { return m_image; }

    Rect fillBoundingBox() const final;
    Rect strokeBoundingBox() const final;
    void render(SVGRenderState& state) const final;
    void layoutElement(const SVGLayoutState& state) final;

private:
    SVGLength m_x;
    SVGLength m_y;
    SVGLength m_width;
    SVGLength m_height;
    SVGPreserveAspectRatio m_preserveAspectRatio;
    Bitmap m_image;
};

class SVGSymbolElement final : public SVGGraphicsElement, public SVGFitToViewBox {
public:
    SVGSymbolElement(Document* document);
};

class SVGGElement final : public SVGGraphicsElement {
public:
    SVGGElement(Document* document);

    void render(SVGRenderState& state) const final;
};

class SVGDefsElement final : public SVGGraphicsElement {
public:
    SVGDefsElement(Document* document);
};

class SVGMarkerElement final : public SVGElement, public SVGFitToViewBox {
public:
    SVGMarkerElement(Document* document);

    const SVGLength& refX() const { return m_refX; }
    const SVGLength& refY() const { return m_refY; }
    const SVGLength& markerWidth() const { return m_markerWidth; }
    const SVGLength& markerHeight() const { return m_markerHeight; }
    const SVGEnumeration<MarkerUnits>& markerUnits() const { return m_markerUnits; }
    const SVGAngle& orient() const { return m_orient; }

    Point refPoint() const;
    Size markerSize() const;

    Transform markerTransform(const Point& origin, float angle, float strokeWidth) const;
    Rect markerBoundingBox(const Point& origin, float angle, float strokeWidth) const;
    void renderMarker(SVGRenderState& state, const Point& origin, float angle, float strokeWidth) const;

    Transform localTransform() const final;

private:
    SVGLength m_refX;
    SVGLength m_refY;
    SVGLength m_markerWidth;
    SVGLength m_markerHeight;
    SVGEnumeration<MarkerUnits> m_markerUnits;
    SVGAngle m_orient;
};

class SVGClipPathElement final : public SVGGraphicsElement {
public:
    SVGClipPathElement(Document* document);

    const SVGEnumeration<Units>& clipPathUnits() const { return m_clipPathUnits; }
    Rect clipBoundingBox(const SVGElement* element) const;

    void applyClipMask(SVGRenderState& state) const;
    void applyClipPath(SVGRenderState& state) const;
    bool requiresMasking() const;

private:
    SVGEnumeration<Units> m_clipPathUnits;
};

class SVGMaskElement final : public SVGElement {
public:
    SVGMaskElement(Document* document);

    const SVGLength& x() const { return m_x; }
    const SVGLength& y() const { return m_y; }
    const SVGLength& width() const { return m_width; }
    const SVGLength& height() const { return m_height; }
    const SVGEnumeration<Units>& maskUnits() const { return m_maskUnits; }
    const SVGEnumeration<Units>& maskContentUnits() const { return m_maskContentUnits; }

    Rect maskRect(const SVGElement* element) const;
    Rect maskBoundingBox(const SVGElement* element) const;
    void applyMask(SVGRenderState& state) const;
    void layoutElement(const SVGLayoutState& state) final;

private:
    SVGLength m_x;
    SVGLength m_y;
    SVGLength m_width;
    SVGLength m_height;
    SVGEnumeration<Units> m_maskUnits;
    SVGEnumeration<Units> m_maskContentUnits;
    MaskType m_mask_type = MaskType::Luminance;
};

} // namespace lunasvg

#endif // LUNASVG_SVGELEMENT_H
