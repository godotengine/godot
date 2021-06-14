#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <map>
#include <memory>

#include "property.h"

namespace lunasvg {

class Element;
class SVGElement;
class StyledElement;

enum LengthNegativeValuesMode
{
    AllowNegativeLengths,
    ForbidNegativeLengths
};

enum class TransformType
{
    Matrix,
    Rotate,
    Scale,
    SkewX,
    SkewY,
    Translate
};

class Parser
{
public:
    static Length parseLength(const std::string& string, LengthNegativeValuesMode mode);
    static LengthList parseLengthList(const std::string& string, LengthNegativeValuesMode mode);
    static double parseNumber(const std::string& string);
    static double parseNumberPercentage(const std::string& string);
    static PointList parsePointList(const std::string& string);
    static Transform parseTransform(const std::string& string);
    static Path parsePath(const std::string& string);
    static std::string parseUrl(const std::string& string);
    static std::string parseHref(const std::string& string);
    static Rect parseViewBox(const std::string& string);
    static PreserveAspectRatio parsePreserveAspectRatio(const std::string& string);
    static Angle parseAngle(const std::string& string);
    static MarkerUnits parseMarkerUnits(const std::string& string);
    static SpreadMethod parseSpreadMethod(const std::string& string);
    static Units parseUnits(const std::string& string);
    static Color parseColor(const std::string& string, const StyledElement* element);
    static Paint parsePaint(const std::string& string, const StyledElement* element);
    static WindRule parseWindRule(const std::string& string);
    static LineCap parseLineCap(const std::string& string);
    static LineJoin parseLineJoin(const std::string& string);
    static Display parseDisplay(const std::string& string);
    static Visibility parseVisibility(const std::string& string);

private:
    static bool parseLength(const char*& ptr, const char* end, double& value, LengthUnits& units, LengthNegativeValuesMode mode);
    static bool parseNumberList(const char*& ptr, const char* end, double* values, int count);
    static bool parseArcFlag(const char*& ptr, const char* end, bool& flag);
    static bool parseColorComponent(const char*& ptr, const char* end, double& value);
    static bool parseTransform(const char*& ptr, const char* end, TransformType& type, double* values, int& count);
};

class LayoutRoot;

class ParseDocument
{
public:
    ParseDocument();
    ~ParseDocument();

    bool parse(const char* data, std::size_t size);

    SVGElement* rootElement() const { return m_rootElement.get(); }
    Element* getElementById(const std::string& id) const;
    std::unique_ptr<LayoutRoot> layout() const;

private:
    std::unique_ptr<SVGElement> m_rootElement;
    std::map<std::string, Element*> m_idCache;
};

} // namespace lunasvg

#endif // PARSER_H
