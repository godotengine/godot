#include "svglayoutstate.h"
#include "svgelement.h"
#include "svgparserutils.h"

#include <optional>

namespace lunasvg {

static std::optional<Color> parseColorValue(std::string_view& input, const SVGLayoutState* state)
{
    if(skipString(input, "currentColor")) {
        return state->color();
    }

    plutovg_color_t color;
    int length = plutovg_color_parse(&color, input.data(), input.length());
    if(length == 0)
        return std::nullopt;
    input.remove_prefix(length);
    return Color(plutovg_color_to_argb32(&color));
}

static Color parseColor(std::string_view input, const SVGLayoutState* state, const Color& defaultValue)
{
    auto color = parseColorValue(input, state);
    if(!color || !input.empty())
        color = defaultValue;
    return color.value();
}

static Color parseColorOrNone(std::string_view input, const SVGLayoutState* state, const Color& defaultValue)
{
    if(input.compare("none") == 0)
        return Color::Transparent;
    return parseColor(input, state, defaultValue);
}

static bool parseUrlValue(std::string_view& input, std::string& value)
{
    if(!skipString(input, "url")
        || !skipOptionalSpaces(input)
        || !skipDelimiter(input, '(')
        || !skipOptionalSpaces(input)) {
        return false;
    }

    switch(input.front()) {
    case '\'':
    case '\"': {
        auto delim = input.front();
        input.remove_prefix(1);
        skipOptionalSpaces(input);
        if(!skipDelimiter(input, '#'))
            return false;
        while(!input.empty() && input.front() != delim) {
            value += input.front();
            input.remove_prefix(1);
        }

        skipOptionalSpaces(input);
        if(!skipDelimiter(input, delim))
            return false;
        break;
    } case '#': {
        input.remove_prefix(1);
        while(!input.empty() && input.front() != ')') {
            value += input.front();
            input.remove_prefix(1);
        }

        break;
    } default:
        return false;
    }

    return skipOptionalSpaces(input) && skipDelimiter(input, ')');
}

static std::string parseUrl(std::string_view input)
{
    std::string value;
    if(!parseUrlValue(input, value) || !input.empty())
        value.clear();
    return value;
}

static Paint parsePaint(std::string_view input, const SVGLayoutState* state, const Color& defaultValue)
{
    std::string id;
    if(!parseUrlValue(input, id))
        return Paint(parseColorOrNone(input, state, defaultValue));
    if(skipOptionalSpaces(input))
        return Paint(id, parseColorOrNone(input, state, defaultValue));
    return Paint(id, Color::Transparent);
}

static float parseNumberOrPercentage(std::string_view input, bool allowPercentage, float defaultValue)
{
    float value;
    if(!parseNumber(input, value))
        return defaultValue;
    if(allowPercentage) {
        if(skipDelimiter(input, '%'))
            value /= 100.f;
        value = std::clamp(value, 0.f, 1.f);
    }

    if(!input.empty())
        return defaultValue;
    return value;
}

static Length parseLength(const std::string_view& input, LengthNegativeMode mode, const Length& defaultValue)
{
    Length value;
    if(!value.parse(input, mode))
        value = defaultValue;
    return value;
}

static BaselineShift parseBaselineShift(const std::string_view& input)
{
    if(input.compare("baseline") == 0)
        return BaselineShift::Type::Baseline;
    if(input.compare("sub") == 0)
        return BaselineShift::Type::Sub;
    if(input.compare("super") == 0)
        return BaselineShift::Type::Super;
    return parseLength(input, LengthNegativeMode::Allow, Length(0.f, LengthUnits::None));
}

static LengthList parseDashArray(std::string_view input)
{
    if(input.compare("none") == 0)
        return LengthList();
    LengthList values;
    do {
        size_t count = 0;
        while(count < input.length() && input[count] != ',' && !IS_WS(input[count]))
            ++count;
        Length value(0, LengthUnits::None);
        if(!value.parse(input.substr(0, count), LengthNegativeMode::Forbid))
            return LengthList();
        input.remove_prefix(count);
        values.push_back(std::move(value));
    } while(skipOptionalSpacesOrComma(input));
    return values;
}

static float parseFontSize(std::string_view input, const SVGLayoutState* state)
{
    auto length = parseLength(input, LengthNegativeMode::Forbid, Length(12, LengthUnits::None));
    if(length.units() == LengthUnits::Percent)
        return length.value() * state->font_size() / 100.f;
    if(length.units() == LengthUnits::Ex)
        return length.value() * state->font_size() / 2.f;
    if(length.units() == LengthUnits::Em)
        return length.value() * state->font_size();
    return length.value();
}

template<typename Enum, unsigned int N>
static Enum parseEnumValue(const std::string_view& input, const SVGEnumerationEntry<Enum>(&entries)[N], Enum defaultValue)
{
    for(const auto& entry : entries) {
        if(input == entry.second) {
            return entry.first;
        }
    }

    return defaultValue;
}

static Display parseDisplay(const std::string_view& input)
{
    static const SVGEnumerationEntry<Display> entries[] = {
        {Display::Inline, "inline"},
        {Display::None, "none"}
    };

    return parseEnumValue(input, entries, Display::Inline);
}

static Visibility parseVisibility(const std::string_view& input)
{
    static const SVGEnumerationEntry<Visibility> entries[] = {
        {Visibility::Visible, "visible"},
        {Visibility::Hidden, "hidden"},
        {Visibility::Collapse, "collapse"}
    };

    return parseEnumValue(input, entries, Visibility::Visible);
}

static Overflow parseOverflow(const std::string_view& input)
{
    static const SVGEnumerationEntry<Overflow> entries[] = {
        {Overflow::Visible, "visible"},
        {Overflow::Hidden, "hidden"}
    };

    return parseEnumValue(input, entries, Overflow::Visible);
}

static FontWeight parseFontWeight(const std::string_view& input)
{
    static const SVGEnumerationEntry<FontWeight> entries[] = {
        {FontWeight::Normal, "normal"},
        {FontWeight::Bold, "bold"},
        {FontWeight::Normal, "100"},
        {FontWeight::Normal, "200"},
        {FontWeight::Normal, "300"},
        {FontWeight::Normal, "400"},
        {FontWeight::Normal, "500"},
        {FontWeight::Bold, "600"},
        {FontWeight::Bold, "700"},
        {FontWeight::Bold, "800"},
        {FontWeight::Bold, "900"}
    };

    return parseEnumValue(input, entries, FontWeight::Normal);
}

static FontStyle parseFontStyle(const std::string_view& input)
{
    static const SVGEnumerationEntry<FontStyle> entries[] = {
        {FontStyle::Normal, "normal"},
        {FontStyle::Italic, "italic"},
        {FontStyle::Italic, "oblique"}
    };

    return parseEnumValue(input, entries, FontStyle::Normal);
}

static AlignmentBaseline parseAlignmentBaseline(const std::string_view& input)
{
    static const SVGEnumerationEntry<AlignmentBaseline> entries[] = {
        {AlignmentBaseline::Auto, "auto"},
        {AlignmentBaseline::Baseline, "baseline"},
        {AlignmentBaseline::BeforeEdge, "before-edge"},
        {AlignmentBaseline::TextBeforeEdge, "text-before-edge"},
        {AlignmentBaseline::Middle, "middle"},
        {AlignmentBaseline::Central, "central"},
        {AlignmentBaseline::AfterEdge, "after-edge"},
        {AlignmentBaseline::TextAfterEdge, "text-after-edge"},
        {AlignmentBaseline::Ideographic, "ideographic"},
        {AlignmentBaseline::Alphabetic, "alphabetic"},
        {AlignmentBaseline::Hanging, "hanging"},
        {AlignmentBaseline::Mathematical, "mathematical"}
    };

    return parseEnumValue(input, entries, AlignmentBaseline::Auto);
}

static DominantBaseline parseDominantBaseline(const std::string_view& input)
{
    static const SVGEnumerationEntry<DominantBaseline> entries[] = {
        {DominantBaseline::Auto, "auto"},
        {DominantBaseline::UseScript, "use-script"},
        {DominantBaseline::NoChange, "no-change"},
        {DominantBaseline::ResetSize, "reset-size"},
        {DominantBaseline::Ideographic, "ideographic"},
        {DominantBaseline::Alphabetic, "alphabetic"},
        {DominantBaseline::Hanging, "hanging"},
        {DominantBaseline::Mathematical, "mathematical"},
        {DominantBaseline::Central, "central"},
        {DominantBaseline::Middle, "middle"},
        {DominantBaseline::TextAfterEdge, "text-after-edge"},
        {DominantBaseline::TextBeforeEdge, "text-before-edge"}
    };

    return parseEnumValue(input, entries, DominantBaseline::Auto);
}

static Direction parseDirection(const std::string_view& input)
{
    static const SVGEnumerationEntry<Direction> entries[] = {
        {Direction::Ltr, "ltr"},
        {Direction::Rtl, "rtl"}
    };

    return parseEnumValue(input, entries, Direction::Ltr);
}

static TextAnchor parseTextAnchor(const std::string_view& input)
{
    static const SVGEnumerationEntry<TextAnchor> entries[] = {
        {TextAnchor::Start, "start"},
        {TextAnchor::Middle, "middle"},
        {TextAnchor::End, "end"}
    };

    return parseEnumValue(input, entries, TextAnchor::Start);
}

static WhiteSpace parseWhiteSpace(const std::string_view& input)
{
    static const SVGEnumerationEntry<WhiteSpace> entries[] = {
        {WhiteSpace::Default, "default"},
        {WhiteSpace::Preserve, "preserve"},
        {WhiteSpace::Default, "normal"},
        {WhiteSpace::Default, "nowrap"},
        {WhiteSpace::Default, "pre-line"},
        {WhiteSpace::Preserve, "pre-wrap"},
        {WhiteSpace::Preserve, "pre"}
    };

    return parseEnumValue(input, entries, WhiteSpace::Default);
}

static MaskType parseMaskType(const std::string_view& input)
{
    static const SVGEnumerationEntry<MaskType> entries[] = {
        {MaskType::Luminance, "luminance"},
        {MaskType::Alpha, "alpha"}
    };

    return parseEnumValue(input, entries, MaskType::Luminance);
}

static FillRule parseFillRule(const std::string_view& input)
{
    static const SVGEnumerationEntry<FillRule> entries[] = {
        {FillRule::NonZero, "nonzero"},
        {FillRule::EvenOdd, "evenodd"}
    };

    return parseEnumValue(input, entries, FillRule::NonZero);
}

static LineCap parseLineCap(const std::string_view& input)
{
    static const SVGEnumerationEntry<LineCap> entries[] = {
        {LineCap::Butt, "butt"},
        {LineCap::Round, "round"},
        {LineCap::Square, "square"}
    };

    return parseEnumValue(input, entries, LineCap::Butt);
}

static LineJoin parseLineJoin(const std::string_view& input)
{
    static const SVGEnumerationEntry<LineJoin> entries[] = {
        {LineJoin::Miter, "miter"},
        {LineJoin::Round, "round"},
        {LineJoin::Bevel, "bevel"}
    };

    return parseEnumValue(input, entries, LineJoin::Miter);
}

SVGLayoutState::SVGLayoutState(const SVGLayoutState& parent, const SVGElement* element)
    : m_parent(&parent)
    , m_element(element)
    , m_fill(parent.fill())
    , m_stroke(parent.stroke())
    , m_color(parent.color())
    , m_fill_opacity(parent.fill_opacity())
    , m_stroke_opacity(parent.stroke_opacity())
    , m_stroke_miterlimit(parent.stroke_miterlimit())
    , m_font_size(parent.font_size())
    , m_stroke_width(parent.stroke_width())
    , m_stroke_dashoffset(parent.stroke_dashoffset())
    , m_stroke_dasharray(parent.stroke_dasharray())
    , m_stroke_linecap(parent.stroke_linecap())
    , m_stroke_linejoin(parent.stroke_linejoin())
    , m_fill_rule(parent.fill_rule())
    , m_clip_rule(parent.clip_rule())
    , m_font_weight(parent.font_weight())
    , m_font_style(parent.font_style())
    , m_dominant_baseline(parent.dominant_baseline())
    , m_text_anchor(parent.text_anchor())
    , m_white_space(parent.white_space())
    , m_direction(parent.direction())
    , m_visibility(parent.visibility())
    , m_overflow(element->parent() ? Overflow::Hidden : Overflow::Visible)
    , m_marker_start(parent.marker_start())
    , m_marker_mid(parent.marker_mid())
    , m_marker_end(parent.marker_end())
    , m_font_family(parent.font_family())
{
    for(const auto& attribute : element->attributes()) {
        std::string_view input(attribute.value());
        stripLeadingAndTrailingSpaces(input);
        if(input.empty() || input.compare("inherit") == 0)
            continue;
        switch(attribute.id()) {
        case PropertyID::Fill:
            m_fill = parsePaint(input, this, Color::Black);
            break;
        case PropertyID::Stroke:
            m_stroke = parsePaint(input, this, Color::Transparent);
            break;
        case PropertyID::Color:
            m_color = parseColor(input, this, Color::Black);
            break;
        case PropertyID::Stop_Color:
            m_stop_color = parseColor(input, this, Color::Black);
            break;
        case PropertyID::Opacity:
            m_opacity = parseNumberOrPercentage(input, true, 1.f);
            break;
        case PropertyID::Fill_Opacity:
            m_fill_opacity = parseNumberOrPercentage(input, true, 1.f);
            break;
        case PropertyID::Stroke_Opacity:
            m_stroke_opacity = parseNumberOrPercentage(input, true, 1.f);
            break;
        case PropertyID::Stop_Opacity:
            m_stop_opacity = parseNumberOrPercentage(input, true, 1.f);
            break;
        case PropertyID::Stroke_Miterlimit:
            m_stroke_miterlimit = parseNumberOrPercentage(input, false, 4.f);
            break;
        case PropertyID::Font_Size:
            m_font_size = parseFontSize(input, this);
            break;
        case PropertyID::Baseline_Shift:
            m_baseline_shit = parseBaselineShift(input);
            break;
        case PropertyID::Stroke_Width:
            m_stroke_width = parseLength(input, LengthNegativeMode::Forbid, Length(1.f, LengthUnits::None));
            break;
        case PropertyID::Stroke_Dashoffset:
            m_stroke_dashoffset = parseLength(input, LengthNegativeMode::Allow, Length(0.f, LengthUnits::None));
            break;
        case PropertyID::Stroke_Dasharray:
            m_stroke_dasharray = parseDashArray(input);
            break;
        case PropertyID::Stroke_Linecap:
            m_stroke_linecap = parseLineCap(input);
            break;
        case PropertyID::Stroke_Linejoin:
            m_stroke_linejoin = parseLineJoin(input);
            break;
        case PropertyID::Fill_Rule:
            m_fill_rule = parseFillRule(input);
            break;
        case PropertyID::Clip_Rule:
            m_clip_rule = parseFillRule(input);
            break;
        case PropertyID::Font_Weight:
            m_font_weight = parseFontWeight(input);
            break;
        case PropertyID::Font_Style:
            m_font_style = parseFontStyle(input);
            break;
        case PropertyID::Alignment_Baseline:
            m_alignment_baseline = parseAlignmentBaseline(input);
            break;
        case PropertyID::Dominant_Baseline:
            m_dominant_baseline = parseDominantBaseline(input);
            break;
        case PropertyID::Direction:
            m_direction = parseDirection(input);
            break;
        case PropertyID::Text_Anchor:
            m_text_anchor = parseTextAnchor(input);
            break;
        case PropertyID::WhiteSpace:
            m_white_space = parseWhiteSpace(input);
            break;
        case PropertyID::Display:
            m_display = parseDisplay(input);
            break;
        case PropertyID::Visibility:
            m_visibility = parseVisibility(input);
            break;
        case PropertyID::Overflow:
            m_overflow = parseOverflow(input);
            break;
        case PropertyID::Mask_Type:
            m_mask_type = parseMaskType(input);
            break;
        case PropertyID::Mask:
            m_mask = parseUrl(input);
            break;
        case PropertyID::Clip_Path:
            m_clip_path = parseUrl(input);
            break;
        case PropertyID::Marker_Start:
            m_marker_start = parseUrl(input);
            break;
        case PropertyID::Marker_Mid:
            m_marker_mid = parseUrl(input);
            break;
        case PropertyID::Marker_End:
            m_marker_end = parseUrl(input);
            break;
        case PropertyID::Font_Family:
            m_font_family.assign(input);
            break;
        default:
            break;
        }
    }
}

Font SVGLayoutState::font() const
{
    auto bold = m_font_weight == FontWeight::Bold;
    auto italic = m_font_style == FontStyle::Italic;

    FontFace face;
    std::string_view input(m_font_family);
    while(!input.empty() && face.isNull()) {
        auto family = input.substr(0, input.find(','));
        input.remove_prefix(family.length());
        if(!input.empty() && input.front() == ',')
            input.remove_prefix(1);
        stripLeadingAndTrailingSpaces(family);
        if(!family.empty() && (family.front() == '\'' || family.front() == '"')) {
            auto quote = family.front();
            family.remove_prefix(1);
            if(!family.empty() && family.back() == quote)
                family.remove_suffix(1);
            stripLeadingAndTrailingSpaces(family);
        }

        face = fontFaceCache()->getFontFace(family, bold, italic);
    }

    if(face.isNull())
        face = fontFaceCache()->getFontFace(emptyString, bold, italic);
    return Font(face, m_font_size);
}

} // namespace lunasvg
