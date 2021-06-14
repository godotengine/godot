#include "document.h"
#include "layoutcontext.h"
#include "parser.h"

#include <fstream>
#include <cstring>
#include <cmath>

namespace lunasvg {

struct Bitmap::Impl
{
    Impl(std::uint8_t* data, std::uint32_t width, std::uint32_t height, std::uint32_t stride);
    Impl(std::uint32_t width, std::uint32_t height);

    std::unique_ptr<std::uint8_t[]> ownData;
    std::uint8_t* data;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t stride;
};

Bitmap::Impl::Impl(std::uint8_t* data, std::uint32_t width, std::uint32_t height, std::uint32_t stride)
    : data(data), width(width), height(height), stride(stride)
{
}

Bitmap::Impl::Impl(std::uint32_t width, std::uint32_t height)
    : ownData(new std::uint8_t[width*height*4]), data(nullptr), width(width), height(height), stride(width * 4)
{
}

Bitmap::Bitmap()
{
}

Bitmap::Bitmap(std::uint8_t* data, std::uint32_t width, std::uint32_t height, std::uint32_t stride)
    : m_impl(new Impl(data, width, height, stride))
{
}

Bitmap::Bitmap(std::uint32_t width, std::uint32_t height)
    : m_impl(new Impl(width, height))
{
}

void Bitmap::reset(std::uint8_t* data, std::uint32_t width, std::uint32_t height, std::uint32_t stride)
{
    m_impl.reset(new Impl(data, width, height, stride));
}

void Bitmap::reset(std::uint32_t width, std::uint32_t height)
{
    m_impl.reset(new Impl(width, height));
}

std::uint8_t* Bitmap::data() const
{
    return m_impl ? m_impl->data ? m_impl->data : m_impl->ownData.get() : nullptr;
}

std::uint32_t Bitmap::width() const
{
    return m_impl ? m_impl->width : 0;
}

std::uint32_t Bitmap::height() const
{
    return m_impl ? m_impl->height : 0;
}

std::uint32_t Bitmap::stride() const
{
    return m_impl ? m_impl->stride : 0;
}

bool Bitmap::valid() const
{
    return !!m_impl;
}

Box::Box(double x, double y, double w, double h)
    : x(x), y(y), w(w), h(h)
{
}

Matrix::Matrix(double a, double b, double c, double d, double e, double f)
    : a(a), b(b), c(c), d(d), e(e), f(f)
{
}

std::unique_ptr<Document> Document::loadFromFile(const std::string& filename)
{
    std::ifstream fs;
    fs.open(filename);
    if(!fs.is_open())
        return nullptr;

    std::string content;
    std::getline(fs, content, '\0');
    fs.close();

    return loadFromData(content);
}

std::unique_ptr<Document> Document::loadFromData(const std::string& string)
{
    return loadFromData(string.data(), string.size());
}

std::unique_ptr<Document> Document::loadFromData(const char* data, std::size_t size)
{
    ParseDocument parser;
    if(!parser.parse(data, size))
        return nullptr;

    auto root = parser.layout();
    if(!root || root->children.empty())
        return nullptr;

    std::unique_ptr<Document> document(new Document);
    document->root = std::move(root);
    return document;
}

std::unique_ptr<Document> Document::loadFromData(const char* data)
{
    return loadFromData(data, std::strlen(data));
}

Document* Document::rotate(double angle)
{
    root->transform.rotate(angle);
    return this;
}

Document* Document::rotate(double angle, double cx, double cy)
{
    root->transform.rotate(angle, cx, cy);
    return this;
}

Document* Document::scale(double sx, double sy)
{
    root->transform.scale(sx, sy);
    return this;
}

Document* Document::shear(double shx, double shy)
{
    root->transform.shear(shx, shy);
    return this;
}

Document* Document::translate(double tx, double ty)
{
    root->transform.translate(tx, ty);
    return this;
}

Document* Document::transform(double a, double b, double c, double d, double e, double f)
{
    root->transform.transform(a, b, c, d, e, f);
    return this;
}

Document* Document::identity()
{
    root->transform.identity();
    return this;
}

Matrix Document::matrix() const
{
    Matrix matrix;
    matrix.a = root->transform.m00;
    matrix.b = root->transform.m10;
    matrix.c = root->transform.m01;
    matrix.d = root->transform.m11;
    matrix.e = root->transform.m02;
    matrix.f = root->transform.m12;
    return matrix;
}

Box Document::box() const
{
    RenderState state;
    state.mode = RenderMode::Bounding;
    root->render(state);

    Box box;
    box.x = state.box.x;
    box.y = state.box.y;
    box.w = state.box.w;
    box.h = state.box.h;
    return box;
}

double Document::width() const
{
    return root->width;
}

double Document::height() const
{
    return root->height;
}

void Document::render(Bitmap bitmap, std::uint32_t bgColor) const
{
    RenderState state;
    state.canvas = Canvas::create(bitmap.data(), bitmap.width(), bitmap.height(), bitmap.stride());
    state.canvas->clear(bgColor);
    root->render(state);
    state.canvas->rgba();
}

Bitmap Document::renderToBitmap(std::uint32_t width, std::uint32_t height, std::uint32_t bgColor) const
{
    if(root->width == 0.0 || root->height == 0.0)
        return Bitmap{};

    if(width == 0 && height == 0)
    {
        width = static_cast<std::uint32_t>(std::ceil(root->width));
        height = static_cast<std::uint32_t>(std::ceil(root->height));
    }
    else if(width != 0 && height == 0)
    {
        height = static_cast<std::uint32_t>(std::ceil(width * root->height / root->width));
    }
    else if(height != 0 && width == 0)
    {
        width = static_cast<std::uint32_t>(std::ceil(height * root->width / root->height));
    }

    Bitmap bitmap{width, height};
    RenderState state;
    state.matrix.scale(width / root->width, height / root->height);
    state.canvas = Canvas::create(bitmap.data(), bitmap.width(), bitmap.height(), bitmap.stride());
    state.canvas->clear(bgColor);
    root->render(state);
    state.canvas->rgba();
    return bitmap;
}

Document::Document()
{
}

Document::~Document()
{
}

} // namespace lunasvg
