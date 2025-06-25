#include "graphics.h"
#include "lunasvg.h"

#include <cfloat>
#include <cmath>

#include "embedded_cnfont.h"
namespace lunasvg {

const Color Color::Black(0xFF000000);
const Color Color::White(0xFFFFFFFF);
const Color Color::Transparent(0x00000000);

const Rect Rect::Empty(0, 0, 0, 0);
const Rect Rect::Invalid(0, 0, -1, -1);
const Rect Rect::Infinite(-FLT_MAX / 2.f, -FLT_MAX / 2.f, FLT_MAX, FLT_MAX);

Rect::Rect(const Box& box)
    : x(box.x), y(box.y), w(box.w), h(box.h)
{
}

const Transform Transform::Identity(1, 0, 0, 1, 0, 0);

Transform::Transform()
{
    plutovg_matrix_init_identity(&m_matrix);
}

Transform::Transform(float a, float b, float c, float d, float e, float f)
{
    plutovg_matrix_init(&m_matrix, a, b, c, d, e, f);
}

Transform::Transform(const Matrix& matrix)
    : Transform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f)
{
}

Transform Transform::operator*(const Transform& transform) const
{
    plutovg_matrix_t result;
    plutovg_matrix_multiply(&result, &transform.m_matrix, &m_matrix);
    return result;
}

Transform& Transform::operator*=(const Transform& transform)
{
    return (*this = *this * transform);
}

Transform& Transform::multiply(const Transform& transform)
{
    return (*this *= transform);
}

Transform& Transform::translate(float tx, float ty)
{
    return multiply(translated(tx, ty));
}

Transform& Transform::scale(float sx, float sy)
{
    return multiply(scaled(sx, sy));
}

Transform& Transform::rotate(float angle, float cx, float cy)
{
    return multiply(rotated(angle, cx, cy));
}

Transform& Transform::shear(float shx, float shy)
{
    return multiply(sheared(shx, shy));
}

Transform& Transform::postMultiply(const Transform& transform)
{
    return (*this = transform * *this);
}

Transform& Transform::postTranslate(float tx, float ty)
{
    return postMultiply(translated(tx, ty));
}

Transform& Transform::postScale(float sx, float sy)
{
    return postMultiply(scaled(sx, sy));
}

Transform& Transform::postRotate(float angle, float cx, float cy)
{
    return postMultiply(rotated(angle, cx, cy));
}

Transform& Transform::postShear(float shx, float shy)
{
    return postMultiply(sheared(shx, shy));
}

Transform Transform::inverse() const
{
    plutovg_matrix_t inverse;
    plutovg_matrix_invert(&m_matrix, &inverse);
    return inverse;
}

Transform& Transform::invert()
{
    plutovg_matrix_invert(&m_matrix, &m_matrix);
    return *this;
}

void Transform::reset()
{
    plutovg_matrix_init_identity(&m_matrix);
}

Point Transform::mapPoint(float x, float y) const
{
    plutovg_matrix_map(&m_matrix, x, y, &x, &y);
    return Point(x, y);
}

Point Transform::mapPoint(const Point& point) const
{
    return mapPoint(point.x, point.y);
}

Rect Transform::mapRect(const Rect& rect) const
{
    if(!rect.isValid()) {
        return Rect::Invalid;
    }

    plutovg_rect_t result = {rect.x, rect.y, rect.w, rect.h};
    plutovg_matrix_map_rect(&m_matrix, &result, &result);
    return result;
}

float Transform::xScale() const
{
    return std::sqrt(m_matrix.a * m_matrix.a + m_matrix.b * m_matrix.b);
}

float Transform::yScale() const
{
    return std::sqrt(m_matrix.c * m_matrix.c + m_matrix.d * m_matrix.d);
}

bool Transform::parse(const char* data, size_t length)
{
    return plutovg_matrix_parse(&m_matrix, data, length);
}

Transform Transform::rotated(float angle, float cx, float cy)
{
    plutovg_matrix_t matrix;
    if(cx == 0.f && cy == 0.f) {
        plutovg_matrix_init_rotate(&matrix, PLUTOVG_DEG2RAD(angle));
    } else {
        plutovg_matrix_init_translate(&matrix, cx, cy);
        plutovg_matrix_rotate(&matrix, PLUTOVG_DEG2RAD(angle));
        plutovg_matrix_translate(&matrix, -cx, -cy);
    }

    return matrix;
}

Transform Transform::scaled(float sx, float sy)
{
    plutovg_matrix_t matrix;
    plutovg_matrix_init_scale(&matrix, sx, sy);
    return matrix;
}

Transform Transform::sheared(float shx, float shy)
{
    plutovg_matrix_t matrix;
    plutovg_matrix_init_shear(&matrix, PLUTOVG_DEG2RAD(shx), PLUTOVG_DEG2RAD(shy));
    return matrix;
}

Transform Transform::translated(float tx, float ty)
{
    plutovg_matrix_t matrix;
    plutovg_matrix_init_translate(&matrix, tx, ty);
    return matrix;
}

Path::Path(const Path& path)
    : m_data(plutovg_path_reference(path.data()))
{
}

Path::Path(Path&& path)
    : m_data(path.release())
{
}

Path::~Path()
{
    plutovg_path_destroy(m_data);
}

Path& Path::operator=(const Path& path)
{
    Path(path).swap(*this);
    return *this;
}

Path& Path::operator=(Path&& path)
{
    Path(std::move(path)).swap(*this);
    return *this;
}

void Path::moveTo(float x, float y)
{
    plutovg_path_move_to(ensure(), x, y);
}

void Path::lineTo(float x, float y)
{
    plutovg_path_line_to(ensure(), x, y);
}

void Path::quadTo(float x1, float y1, float x2, float y2)
{
    plutovg_path_quad_to(ensure(), x1, y1, x2, y2);
}

void Path::cubicTo(float x1, float y1, float x2, float y2, float x3, float y3)
{
    plutovg_path_cubic_to(ensure(), x1, y1, x2, y2, x3, y3);
}

void Path::arcTo(float rx, float ry, float xAxisRotation, bool largeArcFlag, bool sweepFlag, float x, float y)
{
    plutovg_path_arc_to(ensure(), rx, ry, PLUTOVG_DEG2RAD(xAxisRotation), largeArcFlag, sweepFlag, x, y);
}

void Path::close()
{
    plutovg_path_close(ensure());
}

void Path::addEllipse(float cx, float cy, float rx, float ry)
{
    plutovg_path_add_ellipse(ensure(), cx, cy, rx, ry);
}

void Path::addRoundRect(float x, float y, float w, float h, float rx, float ry)
{
    plutovg_path_add_round_rect(ensure(), x, y, w, h, rx, ry);
}

void Path::addRect(float x, float y, float w, float h)
{
    plutovg_path_add_rect(ensure(), x, y, w, h);
}

void Path::addEllipse(const Point& center, const Size& radii)
{
    addEllipse(center.x, center.y, radii.w, radii.h);
}

void Path::addRoundRect(const Rect& rect, const Size& radii)
{
    addRoundRect(rect.x, rect.y, rect.w, rect.h, radii.w, radii.h);
}

void Path::addRect(const Rect& rect)
{
    addRect(rect.x, rect.y, rect.w, rect.h);
}

void Path::reset()
{
    plutovg_path_destroy(m_data);
    m_data = nullptr;
}

Rect Path::boundingRect() const
{
    if(m_data == nullptr)
        return Rect::Empty;
    plutovg_rect_t extents;
    plutovg_path_extents(m_data, &extents, false);
    return extents;
}

bool Path::isEmpty() const
{
    if(m_data)
        return plutovg_path_get_elements(m_data, nullptr) == 0;
    return true;
}

bool Path::isUnique() const
{
    if(m_data)
        return plutovg_path_get_reference_count(m_data) == 1;
    return true;
}

bool Path::parse(const char* data, size_t length)
{
    plutovg_path_reset(ensure());
    return plutovg_path_parse(m_data, data, length);
}

plutovg_path_t* Path::ensure()
{
    if(isNull()) {
        m_data = plutovg_path_create();
    } else if(!isUnique()) {
        plutovg_path_destroy(m_data);
        m_data = plutovg_path_clone(m_data);
    }

    return m_data;
}

PathIterator::PathIterator(const Path& path)
    : m_size(plutovg_path_get_elements(path.data(), &m_elements))
    , m_index(0)
{
}

PathCommand PathIterator::currentSegment(std::array<Point, 3>& points) const
{
    auto command = m_elements[m_index].header.command;
    switch(command) {
    case PLUTOVG_PATH_COMMAND_MOVE_TO:
        points[0] = m_elements[m_index + 1].point;
        break;
    case PLUTOVG_PATH_COMMAND_LINE_TO:
        points[0] = m_elements[m_index + 1].point;
        break;
    case PLUTOVG_PATH_COMMAND_CUBIC_TO:
        points[0] = m_elements[m_index + 1].point;
        points[1] = m_elements[m_index + 2].point;
        points[2] = m_elements[m_index + 3].point;
        break;
    case PLUTOVG_PATH_COMMAND_CLOSE:
        points[0] = m_elements[m_index + 1].point;
        break;
    }

    return PathCommand(command);
}

void PathIterator::next()
{
    m_index += m_elements[m_index].header.length;
}

const std::string emptyString;

FontFace::FontFace(plutovg_font_face_t* face)
    : m_face(face)
{
}

FontFace::FontFace(const void* data, size_t length, plutovg_destroy_func_t destroy_func, void* closure)
    : m_face(plutovg_font_face_load_from_data(data, length, 0, destroy_func, closure))
{
}

FontFace::FontFace(const char* filename)
    : m_face(plutovg_font_face_load_from_file(filename, 0))
{
}

FontFace::FontFace(const FontFace& face)
    : m_face(plutovg_font_face_reference(face.get()))
{
}

FontFace::FontFace(FontFace&& face)
    : m_face(face.release())
{
}

FontFace::~FontFace()
{
    plutovg_font_face_destroy(m_face);
}

FontFace& FontFace::operator=(const FontFace& face)
{
    FontFace(face).swap(*this);
    return *this;
}

FontFace& FontFace::operator=(FontFace&& face)
{
    FontFace(std::move(face)).swap(*this);
    return *this;
}

void FontFace::swap(FontFace& face)
{
    std::swap(m_face, face.m_face);
}

plutovg_font_face_t* FontFace::release()
{
    return std::exchange(m_face, nullptr);
}

bool FontFaceCache::addFontFace(const std::string& family, bool bold, bool italic, const FontFace& face)
{
    if(!face.isNull())
        m_table[family].emplace_back(bold, italic, face);
    return !face.isNull();
}

FontFace FontFaceCache::getFontFace(const std::string_view& family, bool bold, bool italic)
{
    auto it = m_table.find(family);
    if(it == m_table.end()) {
        return FontFace();
    }

    auto select = [bold, italic](const FontFaceEntry& a, const FontFaceEntry& b) {
        if(std::get<2>(a).isNull())
            return b;
        if(std::get<2>(b).isNull())
            return a;
        int aScore = (bold == std::get<0>(a)) + (italic == std::get<1>(a));
        int bScore = (bold == std::get<0>(b)) + (italic == std::get<1>(b));
        return aScore > bScore ? a : b;
    };

    FontFaceEntry entry;
    for(const auto& item : it->second) {
        entry = select(entry, item);
    }

    return std::get<2>(entry);
}

FontFaceCache::FontFaceCache()
{
    static const struct {
        const char* filename;
        const bool bold;
        const bool italic;
    } entries[] = {
#if defined(_WIN32)
        {"C:/Windows/Fonts/arial.ttf", false, false},
        {"C:/Windows/Fonts/arialbd.ttf", true, false},
        {"C:/Windows/Fonts/ariali.ttf", false, true},
        {"C:/Windows/Fonts/arialbi.ttf", true, true},
#elif defined(__APPLE__)
        {"/Library/Fonts/Arial.ttf", false, false},
        {"/Library/Fonts/Arial Bold.ttf", true, false},
        {"/Library/Fonts/Arial Italic.ttf", false, true},
        {"/Library/Fonts/Arial Bold Italic.ttf", true, true},

        {"/System/Library/Fonts/Supplemental/Arial.ttf", false, false},
        {"/System/Library/Fonts/Supplemental/Arial Bold.ttf", true, false},
        {"/System/Library/Fonts/Supplemental/Arial Italic.ttf", false, true},
        {"/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf", true, true},
#elif defined(__linux__)
        {"/usr/share/fonts/dejavu/DejaVuSans.ttf", false, false},
        {"/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", true, false},
        {"/usr/share/fonts/dejavu/DejaVuSans-Oblique.ttf", false, true},
        {"/usr/share/fonts/dejavu/DejaVuSans-BoldOblique.ttf", true, true},

        {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", false, false},
        {"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", true, false},
        {"/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", false, true},
        {"/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf", true, true},
#endif
    };

#ifndef __EMSCRIPTEN__
    for(const auto& entry : entries) {
        addFontFace(emptyString, entry.bold, entry.italic, FontFace(entry.filename));
    }
#endif

#ifdef LUNASVG_ENABLE_EMBEDDED_FONTS
    // load embedded font after system fonts are loaded,
    // ensure fonts are available in WASM environments
    FontFace cnFont(embedded_cnfont_data, embedded_cnfont_size, nullptr, nullptr);
    if (!cnFont.isNull()) {
        addFontFace("", false, false, cnFont);
    }
#endif

}

FontFaceCache* fontFaceCache()
{
    thread_local FontFaceCache cache;
    return &cache;
}

Font::Font(const FontFace& face, float size)
    : m_face(face), m_size(size)
{
}

float Font::ascent() const
{
    float ascent = 0;
    if(m_size > 0.f && !m_face.isNull())
        plutovg_font_face_get_metrics(m_face.get(), m_size, &ascent, nullptr, nullptr, nullptr);
    return ascent;
}

float Font::descent() const
{
    float descent = 0;
    if(m_size > 0.f && !m_face.isNull())
        plutovg_font_face_get_metrics(m_face.get(), m_size, nullptr, &descent, nullptr, nullptr);
    return descent;
}

float Font::height() const
{
    float ascent = 0, descent = 0;
    if(m_size > 0.f && !m_face.isNull())
        plutovg_font_face_get_metrics(m_face.get(), m_size, &ascent, &descent, nullptr, nullptr);
    return ascent + descent;
}

float Font::xHeight() const
{
    plutovg_rect_t extents = {0};
    if(m_size > 0.f && !m_face.isNull())
        plutovg_font_face_get_glyph_metrics(m_face.get(), m_size, 'x', nullptr, nullptr, &extents);
    return extents.h;
}

float Font::measureText(const std::u32string_view& text) const
{
    if(m_size > 0.f && !m_face.isNull())
        return plutovg_font_face_text_extents(m_face.get(), m_size, text.data(), text.length(), PLUTOVG_TEXT_ENCODING_UTF32, nullptr);
    return 0;
}

std::shared_ptr<Canvas> Canvas::create(const Bitmap& bitmap)
{
    return std::shared_ptr<Canvas>(new Canvas(bitmap));
}

std::shared_ptr<Canvas> Canvas::create(float x, float y, float width, float height)
{
    constexpr int kMaxSize = 1 << 24;
    if(width <= 0 || height <= 0 || width > kMaxSize || height > kMaxSize)
        return std::shared_ptr<Canvas>(new Canvas(0, 0, 1, 1));
    auto l = static_cast<int>(std::floor(x));
    auto t = static_cast<int>(std::floor(y));
    auto r = static_cast<int>(std::ceil(x + width));
    auto b = static_cast<int>(std::ceil(y + height));
    return std::shared_ptr<Canvas>(new Canvas(l, t, r - l, b - t));
}

std::shared_ptr<Canvas> Canvas::create(const Rect& extents)
{
    return create(extents.x, extents.y, extents.w, extents.h);
}

void Canvas::setColor(const Color& color)
{
    setColor(color.redF(), color.greenF(), color.blueF(), color.alphaF());
}

void Canvas::setColor(float r, float g, float b, float a)
{
    plutovg_canvas_set_rgba(m_canvas, r, g, b, a);
}

void Canvas::setLinearGradient(float x1, float y1, float x2, float y2, SpreadMethod spread, const GradientStops& stops, const Transform& transform)
{
    plutovg_canvas_set_linear_gradient(m_canvas, x1, y1, x2, y2, static_cast<plutovg_spread_method_t>(spread), stops.data(), stops.size(), &transform.matrix());
}

void Canvas::setRadialGradient(float cx, float cy, float r, float fx, float fy, SpreadMethod spread, const GradientStops& stops, const Transform& transform)
{
    plutovg_canvas_set_radial_gradient(m_canvas, cx, cy, r, fx, fy, 0.f, static_cast<plutovg_spread_method_t>(spread), stops.data(), stops.size(), &transform.matrix());
}

void Canvas::setTexture(const Canvas& source, TextureType type, float opacity, const Transform& transform)
{
    plutovg_canvas_set_texture(m_canvas, source.surface(), static_cast<plutovg_texture_type_t>(type), opacity, &transform.matrix());
}

void Canvas::fillPath(const Path& path, FillRule fillRule, const Transform& transform)
{
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_transform(m_canvas, &transform.matrix());
    plutovg_canvas_set_fill_rule(m_canvas, static_cast<plutovg_fill_rule_t>(fillRule));
    plutovg_canvas_set_operator(m_canvas, PLUTOVG_OPERATOR_SRC_OVER);
    plutovg_canvas_fill_path(m_canvas, path.data());
}

void Canvas::strokePath(const Path& path, const StrokeData& strokeData, const Transform& transform)
{
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_transform(m_canvas, &transform.matrix());
    plutovg_canvas_set_line_width(m_canvas, strokeData.lineWidth());
    plutovg_canvas_set_miter_limit(m_canvas, strokeData.miterLimit());
    plutovg_canvas_set_line_cap(m_canvas, static_cast<plutovg_line_cap_t>(strokeData.lineCap()));
    plutovg_canvas_set_line_join(m_canvas, static_cast<plutovg_line_join_t>(strokeData.lineJoin()));
    plutovg_canvas_set_dash_offset(m_canvas, strokeData.dashOffset());
    plutovg_canvas_set_dash_array(m_canvas, strokeData.dashArray().data(), strokeData.dashArray().size());
    plutovg_canvas_set_operator(m_canvas, PLUTOVG_OPERATOR_SRC_OVER);
    plutovg_canvas_stroke_path(m_canvas, path.data());
}

void Canvas::fillText(const std::u32string_view& text, const Font& font, const Point& origin, const Transform& transform)
{
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_transform(m_canvas, &transform.matrix());
    plutovg_canvas_set_fill_rule(m_canvas, PLUTOVG_FILL_RULE_NON_ZERO);
    plutovg_canvas_set_operator(m_canvas, PLUTOVG_OPERATOR_SRC_OVER);
    plutovg_canvas_set_font(m_canvas, font.face().get(), font.size());
    plutovg_canvas_fill_text(m_canvas, text.data(), text.length(), PLUTOVG_TEXT_ENCODING_UTF32, origin.x, origin.y);
}

void Canvas::strokeText(const std::u32string_view& text, float strokeWidth, const Font& font, const Point& origin, const Transform& transform)
{
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_transform(m_canvas, &transform.matrix());
    plutovg_canvas_set_line_width(m_canvas, strokeWidth);
    plutovg_canvas_set_miter_limit(m_canvas, 4.f);
    plutovg_canvas_set_line_cap(m_canvas, PLUTOVG_LINE_CAP_BUTT);
    plutovg_canvas_set_line_join(m_canvas, PLUTOVG_LINE_JOIN_MITER);
    plutovg_canvas_set_dash_offset(m_canvas, 0.f);
    plutovg_canvas_set_dash_array(m_canvas, nullptr, 0);
    plutovg_canvas_set_operator(m_canvas, PLUTOVG_OPERATOR_SRC_OVER);
    plutovg_canvas_set_font(m_canvas, font.face().get(), font.size());
    plutovg_canvas_stroke_text(m_canvas, text.data(), text.length(), PLUTOVG_TEXT_ENCODING_UTF32, origin.x, origin.y);
}

void Canvas::clipPath(const Path& path, FillRule clipRule, const Transform& transform)
{
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_transform(m_canvas, &transform.matrix());
    plutovg_canvas_set_fill_rule(m_canvas, static_cast<plutovg_fill_rule_t>(clipRule));
    plutovg_canvas_clip_path(m_canvas, path.data());
}

void Canvas::clipRect(const Rect& rect, FillRule clipRule, const Transform& transform)
{
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_transform(m_canvas, &transform.matrix());
    plutovg_canvas_set_fill_rule(m_canvas, static_cast<plutovg_fill_rule_t>(clipRule));
    plutovg_canvas_clip_rect(m_canvas, rect.x, rect.y, rect.w, rect.h);
}

void Canvas::drawImage(const Bitmap& image, const Rect& dstRect, const Rect& srcRect, const Transform& transform)
{
    auto xScale = dstRect.w / srcRect.w;
    auto yScale = dstRect.h / srcRect.h;
    plutovg_matrix_t matrix = { xScale, 0, 0, yScale, -srcRect.x * xScale, -srcRect.y * yScale };
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_transform(m_canvas, &transform.matrix());
    plutovg_canvas_translate(m_canvas, dstRect.x, dstRect.y);
    plutovg_canvas_set_fill_rule(m_canvas, PLUTOVG_FILL_RULE_NON_ZERO);
    plutovg_canvas_set_operator(m_canvas, PLUTOVG_OPERATOR_SRC_OVER);
    plutovg_canvas_set_texture(m_canvas, image.surface(), PLUTOVG_TEXTURE_TYPE_PLAIN, 1.f, &matrix);
    plutovg_canvas_fill_rect(m_canvas, 0, 0, dstRect.w, dstRect.h);
}

void Canvas::blendCanvas(const Canvas& canvas, BlendMode blendMode, float opacity)
{
    plutovg_matrix_t matrix = { 1, 0, 0, 1, static_cast<float>(canvas.x()), static_cast<float>(canvas.y()) };
    plutovg_canvas_reset_matrix(m_canvas);
    plutovg_canvas_translate(m_canvas, -m_x, -m_y);
    plutovg_canvas_set_operator(m_canvas, static_cast<plutovg_operator_t>(blendMode));
    plutovg_canvas_set_texture(m_canvas, canvas.surface(), PLUTOVG_TEXTURE_TYPE_PLAIN, opacity, &matrix);
    plutovg_canvas_paint(m_canvas);
}

void Canvas::save()
{
    plutovg_canvas_save(m_canvas);
}

void Canvas::restore()
{
    plutovg_canvas_restore(m_canvas);
}

int Canvas::width() const
{
    return plutovg_surface_get_width(m_surface);
}

int Canvas::height() const
{
    return plutovg_surface_get_height(m_surface);
}

void Canvas::convertToLuminanceMask()
{
    auto width = plutovg_surface_get_width(m_surface);
    auto height = plutovg_surface_get_height(m_surface);
    auto stride = plutovg_surface_get_stride(m_surface);
    auto data = plutovg_surface_get_data(m_surface);
    for(int y = 0; y < height; y++) {
        auto pixels = reinterpret_cast<uint32_t*>(data + stride * y);
        for(int x = 0; x < width; x++) {
            auto pixel = pixels[x];
            auto r = (pixel >> 16) & 0xFF;
            auto g = (pixel >> 8) & 0xFF;
            auto b = (pixel >> 0) & 0xFF;
            auto l = (2*r + 3*g + b) / 6;
            pixels[x] = l << 24;
        }
    }
}

Canvas::~Canvas()
{
    plutovg_canvas_destroy(m_canvas);
    plutovg_surface_destroy(m_surface);
}

Canvas::Canvas(const Bitmap& bitmap)
    : m_surface(plutovg_surface_reference(bitmap.surface()))
    , m_canvas(plutovg_canvas_create(m_surface))
    , m_x(0), m_y(0)
{
}

Canvas::Canvas(int x, int y, int width, int height)
    : m_surface(plutovg_surface_create(width, height))
    , m_canvas(plutovg_canvas_create(m_surface))
    , m_x(x), m_y(y)
{
}

} // namespace lunasvg
