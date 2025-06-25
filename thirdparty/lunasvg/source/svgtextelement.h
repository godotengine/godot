#ifndef LUNASVG_SVGTEXTELEMENT_H
#define LUNASVG_SVGTEXTELEMENT_H

#include "svgelement.h"

#include <optional>

namespace lunasvg {

class SVGTextPositioningElement;
class SVGTextElement;

struct SVGCharacterPosition {
    std::optional<float> x;
    std::optional<float> y;
    std::optional<float> dx;
    std::optional<float> dy;
    std::optional<float> rotate;
};

using SVGCharacterPositions = std::map<size_t, SVGCharacterPosition>;

struct SVGTextPosition {
    SVGTextPosition(const SVGNode* node, size_t startOffset, size_t endOffset)
        : node(node), startOffset(startOffset), endOffset(endOffset)
    {}

    const SVGNode* node;
    size_t startOffset;
    size_t endOffset;
};

using SVGTextPositionList = std::vector<SVGTextPosition>;

struct SVGTextFragment {
    explicit SVGTextFragment(const SVGTextPositioningElement* element) : element(element) {}
    const SVGTextPositioningElement* element;
    size_t offset = 0;
    size_t length = 0;
    float x = 0;
    float y = 0;
    float angle = 0;
    float width = 0;
    bool startsNewTextChunk = false;
};

using SVGTextFragmentList = std::vector<SVGTextFragment>;

class SVGTextFragmentsBuilder {
public:
    SVGTextFragmentsBuilder(std::u32string& text, SVGTextFragmentList& fragments);

    void build(const SVGTextElement* textElement);

private:
    void handleText(const SVGTextNode* node);
    void handleElement(const SVGTextPositioningElement* element);
    void fillCharacterPositions(const SVGTextPosition& position);
    std::u32string& m_text;
    SVGTextFragmentList& m_fragments;
    SVGCharacterPositions m_characterPositions;
    SVGTextPositionList m_textPositions;
    size_t m_characterOffset = 0;
    float m_x = 0;
    float m_y = 0;
};

class SVGTextPositioningElement : public SVGGraphicsElement {
public:
    SVGTextPositioningElement(Document* document, ElementID id);

    bool isTextPositioningElement() const final { return true; }

    const LengthList& x() const { return m_x.values(); }
    const LengthList& y() const { return m_y.values(); }
    const LengthList& dx() const { return m_dx.values(); }
    const LengthList& dy() const { return m_dy.values(); }
    const NumberList& rotate() const { return m_rotate.values(); }

    const Font& font() const { return m_font; }
    const SVGPaintServer& fill() const { return m_fill; }
    const SVGPaintServer& stroke() const { return m_stroke; }

    float stroke_width() const { return m_stroke_width; }
    float baseline_offset() const { return m_baseline_offset; }
    AlignmentBaseline alignment_baseline() const { return m_alignment_baseline; }
    DominantBaseline dominant_baseline() const { return m_dominant_baseline; }
    TextAnchor text_anchor() const { return m_text_anchor; }
    WhiteSpace white_space() const { return m_white_space; }
    Direction direction() const { return m_direction; }

    void layoutElement(const SVGLayoutState& state) override;

private:
    float convertBaselineOffset(const BaselineShift& baselineShift) const;
    SVGLengthList m_x;
    SVGLengthList m_y;
    SVGLengthList m_dx;
    SVGLengthList m_dy;
    SVGNumberList m_rotate;

    Font m_font;
    SVGPaintServer m_fill;
    SVGPaintServer m_stroke;

    float m_stroke_width = 1.f;
    float m_baseline_offset = 0.f;
    AlignmentBaseline m_alignment_baseline = AlignmentBaseline::Auto;
    DominantBaseline m_dominant_baseline = DominantBaseline::Auto;
    TextAnchor m_text_anchor = TextAnchor::Start;
    WhiteSpace m_white_space = WhiteSpace::Default;
    Direction m_direction = Direction::Ltr;
};

class SVGTSpanElement final : public SVGTextPositioningElement {
public:
    SVGTSpanElement(Document* document);
};

class SVGTextElement final : public SVGTextPositioningElement {
public:
    SVGTextElement(Document* document);

    Rect fillBoundingBox() const final { return boundingBox(false); }
    Rect strokeBoundingBox() const final { return boundingBox(true); }

    void layout(SVGLayoutState& state) final;
    void render(SVGRenderState& state) const final;

private:
    Rect boundingBox(bool includeStroke) const;
    SVGTextFragmentList m_fragments;
    std::u32string m_text;
};

} // namespace lunasvg

#endif // LUNASVG_SVGTEXTELEMENT_H
