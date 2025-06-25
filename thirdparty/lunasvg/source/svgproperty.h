#ifndef LUNASVG_SVGPROPERTY_H
#define LUNASVG_SVGPROPERTY_H

#include "graphics.h"

#include <string>

namespace lunasvg {

enum class PropertyID : uint8_t {
    Unknown = 0,
    Alignment_Baseline,
    Baseline_Shift,
    Class,
    ClipPathUnits,
    Clip_Path,
    Clip_Rule,
    Color,
    Cx,
    Cy,
    D,
    Direction,
    Display,
    Dominant_Baseline,
    Dx,
    Dy,
    Fill,
    Fill_Opacity,
    Fill_Rule,
    Font_Family,
    Font_Size,
    Font_Style,
    Font_Weight,
    Fx,
    Fy,
    GradientTransform,
    GradientUnits,
    Height,
    Href,
    Id,
    MarkerHeight,
    MarkerUnits,
    MarkerWidth,
    Marker_End,
    Marker_Mid,
    Marker_Start,
    Mask,
    MaskContentUnits,
    MaskUnits,
    Mask_Type,
    Offset,
    Opacity,
    Orient,
    Overflow,
    PatternContentUnits,
    PatternTransform,
    PatternUnits,
    Points,
    PreserveAspectRatio,
    R,
    RefX,
    RefY,
    Rotate,
    Rx,
    Ry,
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
    Text_Anchor,
    Transform,
    ViewBox,
    Visibility,
    WhiteSpace,
    Width,
    X,
    X1,
    X2,
    Y,
    Y1,
    Y2
};

PropertyID propertyid(const std::string_view& name);
PropertyID csspropertyid(const std::string_view& name);

class SVGElement;

class SVGProperty {
public:
    SVGProperty(PropertyID id);
    virtual ~SVGProperty() = default;
    PropertyID id() const { return m_id; }

    virtual bool parse(std::string_view input) = 0;

private:
    SVGProperty(const SVGProperty&) = delete;
    SVGProperty& operator=(const SVGProperty&) = delete;
    PropertyID m_id;
};

class SVGString final : public SVGProperty {
public:
    explicit SVGString(PropertyID id)
        : SVGProperty(id)
    {}

    const std::string& value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    std::string m_value;
};

class Paint {
public:
    Paint() = default;
    explicit Paint(const Color& color) : m_color(color) {}
    Paint(const std::string& id, const Color& color)
        : m_id(id), m_color(color)
    {}

    const Color& color() const { return m_color; }
    const std::string& id() const { return m_id; }
    bool isNone() const { return m_id.empty() && !m_color.isVisible(); }

private:
    std::string m_id;
    Color m_color = Color::Transparent;
};

enum class Display : uint8_t {
    Inline,
    None
};

enum class Visibility : uint8_t {
    Visible,
    Hidden,
    Collapse
};

enum class Overflow : uint8_t {
    Visible,
    Hidden
};

enum class FontStyle : uint8_t {
    Normal,
    Italic
};

enum class FontWeight : uint8_t {
    Normal,
    Bold
};

enum class AlignmentBaseline : uint8_t {
    Auto,
    Baseline,
    BeforeEdge,
    TextBeforeEdge,
    Middle,
    Central,
    AfterEdge,
    TextAfterEdge,
    Ideographic,
    Alphabetic,
    Hanging,
    Mathematical
};

enum class DominantBaseline : uint8_t {
    Auto,
    UseScript,
    NoChange,
    ResetSize,
    Ideographic,
    Alphabetic,
    Hanging,
    Mathematical,
    Central,
    Middle,
    TextAfterEdge,
    TextBeforeEdge
};

enum class TextAnchor : uint8_t {
    Start,
    Middle,
    End
};

enum class WhiteSpace : uint8_t {
    Default,
    Preserve
};

enum class Direction : uint8_t {
    Ltr,
    Rtl
};

enum class MaskType : uint8_t {
    Luminance,
    Alpha
};

enum class Units : uint8_t {
    UserSpaceOnUse,
    ObjectBoundingBox
};

enum class MarkerUnits : uint8_t {
    StrokeWidth,
    UserSpaceOnUse
};

template<typename Enum>
using SVGEnumerationEntry = std::pair<Enum, std::string_view>;

template<typename Enum>
class SVGEnumeration final : public SVGProperty {
public:
    explicit SVGEnumeration(PropertyID id, Enum value)
        : SVGProperty(id)
        , m_value(value)
    {}

    Enum value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    template<unsigned int N>
    bool parseEnum(std::string_view input, const SVGEnumerationEntry<Enum>(&entries)[N]);
    Enum m_value;
};

class SVGAngle final : public SVGProperty {
public:
    enum class OrientType {
        Auto,
        AutoStartReverse,
        Angle
    };

    explicit SVGAngle(PropertyID id)
        : SVGProperty(id)
    {}

    float value() const { return m_value; }
    OrientType orientType() const { return m_orientType; }
    bool parse(std::string_view input) final;

private:
    float m_value = 0;
    OrientType m_orientType = OrientType::Angle;
};

enum class LengthUnits : uint8_t {
    None,
    Percent,
    Px,
    Em,
    Ex
};

enum class LengthDirection : uint8_t {
    Horizontal,
    Vertical,
    Diagonal
};

enum class LengthNegativeMode : uint8_t {
    Allow,
    Forbid
};

class Length {
public:
    Length() = default;
    Length(float value, LengthUnits units)
        : m_value(value), m_units(units)
    {}

    float value() const { return m_value; }
    LengthUnits units() const { return m_units; }

    bool parse(std::string_view input, LengthNegativeMode mode);

private:
    float m_value = 0.f;
    LengthUnits m_units = LengthUnits::None;
};

class SVGLength final : public SVGProperty {
public:
    SVGLength(PropertyID id, LengthDirection direction, LengthNegativeMode negativeMode, float value = 0, LengthUnits units = LengthUnits::None)
        : SVGProperty(id)
        , m_direction(direction)
        , m_negativeMode(negativeMode)
        , m_value(value, units)
    {}

    bool isPercent() const { return m_value.units() == LengthUnits::Percent; }

    LengthDirection direction() const { return m_direction; }
    LengthNegativeMode negativeMode() const { return m_negativeMode; }
    const Length& value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    const LengthDirection m_direction;
    const LengthNegativeMode m_negativeMode;
    Length m_value;
};

class LengthContext {
public:
    LengthContext(const SVGElement* element, Units units = Units::UserSpaceOnUse)
        : m_element(element), m_units(units)
    {}

    float valueForLength(const Length& length, LengthDirection direction) const;
    float valueForLength(const SVGLength& length) const { return valueForLength(length.value(), length.direction()); }

private:
    float viewportDimension(LengthDirection direction) const;
    const SVGElement* m_element;
    const Units m_units;
};

using LengthList = std::vector<Length>;

class SVGLengthList final : public SVGProperty {
public:
    SVGLengthList(PropertyID id, LengthDirection direction, LengthNegativeMode negativeMode)
        : SVGProperty(id)
        , m_direction(direction)
        , m_negativeMode(negativeMode)
    {}

    LengthDirection direction() const { return m_direction; }
    LengthNegativeMode negativeMode() const { return m_negativeMode; }
    const LengthList& values() const { return m_values; }
    bool parse(std::string_view input) final;

private:
    const LengthDirection m_direction;
    const LengthNegativeMode m_negativeMode;
    LengthList m_values;
};

class BaselineShift {
public:
    enum class Type {
        Baseline,
        Sub,
        Super,
        Length
    };

    BaselineShift() = default;
    BaselineShift(Type type) : m_type(type) {}
    BaselineShift(const Length& length) : m_type(Type::Length), m_length(length) {}

    Type type() const { return m_type; }
    const Length& length() const { return m_length; }

private:
    Type m_type{Type::Baseline};
    Length m_length;
};

class SVGNumber : public SVGProperty {
public:
    SVGNumber(PropertyID id, float value)
        : SVGProperty(id)
        , m_value(value)
    {}

    float value() const { return m_value; }
    bool parse(std::string_view input) override;

private:
    float m_value;
};

class SVGNumberPercentage final : public SVGProperty {
public:
    SVGNumberPercentage(PropertyID id, float value)
        : SVGProperty(id)
        , m_value(value)
    {}

    float value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    float m_value;
};

using NumberList = std::vector<float>;

class SVGNumberList final : public SVGProperty {
public:
    explicit SVGNumberList(PropertyID id)
        : SVGProperty(id)
    {}

    const NumberList& values() const { return m_values; }
    bool parse(std::string_view input) final;

private:
    NumberList m_values;
};

class SVGPath final : public SVGProperty {
public:
    explicit SVGPath(PropertyID id)
        : SVGProperty(id)
    {}

    const Path& value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    Path m_value;
};

class SVGPoint final : public SVGProperty {
public:
   explicit SVGPoint(PropertyID id)
        : SVGProperty(id)
    {}

    const Point& value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    Point m_value;
};

using PointList = std::vector<Point>;

class SVGPointList final : public SVGProperty {
public:
    explicit SVGPointList(PropertyID id)
        : SVGProperty(id)
    {}

    const PointList& values() const { return m_values; }
    bool parse(std::string_view input) final;

private:
    PointList m_values;
};

class SVGRect final : public SVGProperty {
public:
    explicit SVGRect(PropertyID id)
        : SVGProperty(id)
        , m_value(Rect::Invalid)
    {}

    const Rect& value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    Rect m_value;
};

class SVGTransform final : public SVGProperty {
public:
    explicit SVGTransform(PropertyID id)
        : SVGProperty(id)
    {}

    const Transform& value() const { return m_value; }
    bool parse(std::string_view input) final;

private:
    Transform m_value;
};

class SVGPreserveAspectRatio final : public SVGProperty {
public:
    enum class AlignType {
        None,
        xMinYMin,
        xMidYMin,
        xMaxYMin,
        xMinYMid,
        xMidYMid,
        xMaxYMid,
        xMinYMax,
        xMidYMax,
        xMaxYMax
    };

    enum class MeetOrSlice {
        Meet,
        Slice
    };

    explicit SVGPreserveAspectRatio(PropertyID id)
        : SVGProperty(id)
    {}

    AlignType alignType() const { return m_alignType; }
    MeetOrSlice meetOrSlice() const { return m_meetOrSlice; }
    bool parse(std::string_view input) final;

    Rect getClipRect(const Rect& viewBoxRect, const Size& viewportSize) const;
    Transform getTransform(const Rect& viewBoxRect, const Size& viewportSize) const;
    void transformRect(Rect& dstRect, Rect& srcRect) const;

private:
    AlignType m_alignType = AlignType::xMidYMid;
    MeetOrSlice m_meetOrSlice = MeetOrSlice::Meet;
};

} // namespace lunasvg

#endif // LUNASVG_SVGPROPERTY_H
