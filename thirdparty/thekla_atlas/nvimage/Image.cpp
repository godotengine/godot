// This code is in the public domain -- castanyo@yahoo.es

#include "Image.h"
//#include "ImageIO.h"

#include "nvmath/Color.h"

#include "nvcore/Debug.h"
#include "nvcore/Ptr.h"
#include "nvcore/Utils.h" // swap
#include "nvcore/Memory.h" // realloc, free

#include <string.h> // memcpy


using namespace nv;

Image::Image() : m_width(0), m_height(0), m_format(Format_RGB), m_data(NULL)
{
}

Image::Image(const Image & img) : m_data(NULL)
{
	allocate(img.m_width, img.m_height, img.m_depth);
    m_format = img.m_format;
    memcpy(m_data, img.m_data, sizeof(Color32) * m_width * m_height * m_depth);
}

Image::~Image()
{
    free();
}

const Image & Image::operator=(const Image & img)
{
    allocate(img.m_width, img.m_height, m_depth);
    m_format = img.m_format;
    memcpy(m_data, img.m_data, sizeof(Color32) * m_width * m_height * m_depth);
    return *this;
}


void Image::allocate(uint w, uint h, uint d/*= 1*/)
{
    m_width = w;
    m_height = h;
	m_depth = d;
    m_data = realloc<Color32>(m_data, w * h * d);
}

void Image::acquire(Color32 * data, uint w, uint h, uint d/*= 1*/)
{
    free();
    m_width = w;
    m_height = h;
    m_depth = d;
    m_data = data;
}

void Image::resize(uint w, uint h, uint d/*= 1*/) {

    Image img;
    img.allocate(w, h, d);

    Color32 background(0,0,0,0);

    // Copy image.
    uint x, y, z;
    for(z = 0; z < min(d, m_depth); z++) {
        for(y = 0; y < min(h, m_height); y++) {
            for(x = 0; x < min(w, m_width); x++) {
                img.pixel(x, y, z) = pixel(x, y, z);
            }
            for(; x < w; x++) {
                img.pixel(x, y, z) = background;
            }
        }
        for(; y < h; y++) {
            for(x = 0; x < w; x++) {
                img.pixel(x, y, z) = background;
            }
        }
    }
    for(; z < d; z++) {
        for(y = 0; y < h; y++) {
            for(x = 0; x < w; x++) {
                img.pixel(x, y, z) = background;
            }
        }
    }

    swap(m_width, img.m_width);
    swap(m_height, img.m_height);
	swap(m_depth, img.m_depth);
    swap(m_format, img.m_format);
    swap(m_data, img.m_data);
}

/*bool Image::load(const char * name)
{
    free();

    AutoPtr<Image> img(ImageIO::load(name));
    if (img == NULL) {
        return false;
    }

    swap(m_width, img->m_width);
    swap(m_height, img->m_height);
	swap(m_depth, img->m_depth);
    swap(m_format, img->m_format);
    swap(m_data, img->m_data);

    return true;
}*/

void Image::wrap(void * data, uint w, uint h, uint d)
{
    free();
    m_data = (Color32 *)data;
    m_width = w;
    m_height = h;
	m_depth = d;
}

void Image::unwrap()
{
    m_data = NULL;
    m_width = 0;
    m_height = 0;
	m_depth = 0;
}


void Image::free()
{
    ::free(m_data);
    m_data = NULL;
}


uint Image::width() const
{
    return m_width;
}

uint Image::height() const
{
    return m_height;
}

uint Image::depth() const
{
	return m_depth;
}

const Color32 * Image::scanline(uint h) const
{
    nvDebugCheck(h < m_height);
    return m_data + h * m_width;
}

Color32 * Image::scanline(uint h)
{
    nvDebugCheck(h < m_height);
    return m_data + h * m_width;
}

const Color32 * Image::pixels() const
{
    return m_data;
}

Color32 * Image::pixels()
{
    return m_data;
}

const Color32 & Image::pixel(uint idx) const
{
    nvDebugCheck(idx < m_width * m_height * m_depth);
    return m_data[idx];
}

Color32 & Image::pixel(uint idx)
{
    nvDebugCheck(idx < m_width * m_height * m_depth);
    return m_data[idx];
}


Image::Format Image::format() const
{
    return m_format;
}

void Image::setFormat(Image::Format f)
{
    m_format = f;
}

void Image::fill(Color32 c)
{
    const uint size = m_width * m_height * m_depth;
    for (uint i = 0; i < size; ++i)
    {
        m_data[i] = c;
    }
}

