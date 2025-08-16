// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2012, SIL International, All rights reserved.

#include "graphite2/Font.h"

#include "inc/Main.h"
#include "inc/Face.h"     //for the tags
#include "inc/GlyphCache.h"
#include "inc/GlyphFace.h"
#include "inc/Endian.h"
#include "inc/bits.h"

using namespace graphite2;

namespace
{
    // Iterator over version 1 or 2 glat entries which consist of a series of
    //    +-+-+-+-+-+-+-+-+-+-+                +-+-+-+-+-+-+-+-+-+-+-+-+
    // v1 |k|n|v1 |v2 |...|vN |     or    v2   | k | n |v1 |v2 |...|vN |
    //    +-+-+-+-+-+-+-+-+-+-+                +-+-+-+-+-+-+-+-+-+-+-+-+
    // variable length structures.

    template<typename W>
    class _glat_iterator
    {
        unsigned short  key() const             { return uint16(be::peek<W>(_e) + _n); }
        unsigned int    run() const             { return be::peek<W>(_e+sizeof(W)); }
        void            advance_entry()         { _n = 0; _e = _v; be::skip<W>(_v,2); }
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::pair<sparse::key_type, sparse::mapped_type>;
        using difference_type = ptrdiff_t;
        using pointer = value_type *;
        using reference = value_type &;

        _glat_iterator(const void * glat=0) : _e(reinterpret_cast<const byte *>(glat)), _v(_e+2*sizeof(W)), _n(0) {}

        _glat_iterator<W> & operator ++ () {
            ++_n; be::skip<uint16>(_v);
            if (_n == run()) advance_entry();
            return *this;
        }
        _glat_iterator<W>   operator ++ (int)   { _glat_iterator<W> tmp(*this); operator++(); return tmp; }

        // This is strictly a >= operator. A true == operator could be
        // implemented that test for overlap but it would be more expensive a
        // test.
        bool operator == (const _glat_iterator<W> & rhs) { return _v >= rhs._e - 1; }
        bool operator != (const _glat_iterator<W> & rhs) { return !operator==(rhs); }

        value_type          operator * () const {
            return value_type(key(), be::peek<uint16>(_v));
        }

    protected:
        const byte     * _e, * _v;
        size_t        _n;
    };

    typedef _glat_iterator<uint8>   glat_iterator;
    typedef _glat_iterator<uint16>  glat2_iterator;
}

const SlantBox SlantBox::empty = {0,0,0,0};


class GlyphCache::Loader
{
public:
    Loader(const Face & face);    //return result indicates success. Do not use if failed.

    operator bool () const throw();
    unsigned short int units_per_em() const throw();
    unsigned short int num_glyphs() const throw();
    unsigned short int num_attrs() const throw();
    bool has_boxes() const throw();

    const GlyphFace * read_glyph(unsigned short gid, GlyphFace &, int *numsubs) const throw();
    GlyphBox * read_box(uint16 gid, GlyphBox *curr, const GlyphFace & face) const throw();

    CLASS_NEW_DELETE;
private:
    Face::Table _head,
                _hhea,
                _hmtx,
                _glyf,
                _loca,
                m_pGlat,
                m_pGloc;

    bool            _long_fmt;
    bool            _has_boxes;
    unsigned short  _num_glyphs_graphics,        //i.e. boundary box and advance
                    _num_glyphs_attributes,
                    _num_attrs;                    // number of glyph attributes per glyph
};



GlyphCache::GlyphCache(const Face & face, const uint32 face_options)
: _glyph_loader(new Loader(face)),
  _glyphs(_glyph_loader && *_glyph_loader && _glyph_loader->num_glyphs()
        ? grzeroalloc<const GlyphFace *>(_glyph_loader->num_glyphs()) : 0),
  _boxes(_glyph_loader && _glyph_loader->has_boxes() && _glyph_loader->num_glyphs()
        ? grzeroalloc<GlyphBox *>(_glyph_loader->num_glyphs()) : 0),
  _num_glyphs(_glyphs ? _glyph_loader->num_glyphs() : 0),
  _num_attrs(_glyphs ? _glyph_loader->num_attrs() : 0),
  _upem(_glyphs ? _glyph_loader->units_per_em() : 0)
{
    if ((face_options & gr_face_preloadGlyphs) && _glyph_loader && _glyphs)
    {
        int numsubs = 0;
        GlyphFace * const glyphs = new GlyphFace [_num_glyphs];
        if (!glyphs)
            return;

        // The 0 glyph is definately required.
        _glyphs[0] = _glyph_loader->read_glyph(0, glyphs[0], &numsubs);

        // glyphs[0] has the same address as the glyphs array just allocated,
        //  thus assigning the &glyphs[0] to _glyphs[0] means _glyphs[0] points
        //  to the entire array.
        const GlyphFace * loaded = _glyphs[0];
        for (uint16 gid = 1; loaded && gid != _num_glyphs; ++gid)
            _glyphs[gid] = loaded = _glyph_loader->read_glyph(gid, glyphs[gid], &numsubs);

        if (!loaded)
        {
            _glyphs[0] = 0;
            delete [] glyphs;
        }
        else if (numsubs > 0 && _boxes)
        {
            GlyphBox * boxes = (GlyphBox *)gralloc<char>(_num_glyphs * sizeof(GlyphBox) + numsubs * 8 * sizeof(float));
            GlyphBox * currbox = boxes;

            for (uint16 gid = 0; currbox && gid != _num_glyphs; ++gid)
            {
                _boxes[gid] = currbox;
                currbox = _glyph_loader->read_box(gid, currbox, *_glyphs[gid]);
            }
            if (!currbox)
            {
                free(boxes);
                _boxes[0] = 0;
            }
        }
        delete _glyph_loader;
        _glyph_loader = 0;
	// coverity[leaked_storage : FALSE] - calling read_glyph on index 0 saved
	// glyphs as _glyphs[0]. Setting _glyph_loader to nullptr here flags that
	// the dtor needs to call delete[] on _glyphs[0] to release what was allocated
	// as glyphs
    }

    if (_glyphs && glyph(0) == 0)
    {
        free(_glyphs);
        _glyphs = 0;
        if (_boxes)
        {
            free(_boxes);
            _boxes = 0;
        }
        _num_glyphs = _num_attrs = _upem = 0;
    }
}


GlyphCache::~GlyphCache()
{
    if (_glyphs)
    {
        if (_glyph_loader)
        {
            const GlyphFace *  * g = _glyphs;
            for(unsigned short n = _num_glyphs; n; --n, ++g)
                delete *g;
        }
        else
            delete [] _glyphs[0];
        free(_glyphs);
    }
    if (_boxes)
    {
        if (_glyph_loader)
        {
            GlyphBox *  * g = _boxes;
            for (uint16 n = _num_glyphs; n; --n, ++g)
                free(*g);
        }
        else
            free(_boxes[0]);
        free(_boxes);
    }
    delete _glyph_loader;
}

const GlyphFace *GlyphCache::glyph(unsigned short glyphid) const      //result may be changed by subsequent call with a different glyphid
{
    if (glyphid >= numGlyphs())
        return _glyphs[0];
    const GlyphFace * & p = _glyphs[glyphid];
    if (p == 0 && _glyph_loader)
    {
        int numsubs = 0;
        GlyphFace * g = new GlyphFace();
        if (g)  p = _glyph_loader->read_glyph(glyphid, *g, &numsubs);
        if (!p)
        {
            delete g;
            return *_glyphs;
        }
        if (_boxes)
        {
            _boxes[glyphid] = (GlyphBox *)gralloc<char>(sizeof(GlyphBox) + 8 * numsubs * sizeof(float));
            if (!_glyph_loader->read_box(glyphid, _boxes[glyphid], *_glyphs[glyphid]))
            {
                free(_boxes[glyphid]);
                _boxes[glyphid] = 0;
            }
        }
    }
    return p;
}



GlyphCache::Loader::Loader(const Face & face)
: _head(face, Tag::head),
  _hhea(face, Tag::hhea),
  _hmtx(face, Tag::hmtx),
  _glyf(face, Tag::glyf),
  _loca(face, Tag::loca),
  _long_fmt(false),
  _has_boxes(false),
  _num_glyphs_graphics(0),
  _num_glyphs_attributes(0),
  _num_attrs(0)
{
    if (!operator bool())
        return;

    const Face::Table maxp = Face::Table(face, Tag::maxp);
    if (!maxp) { _head = Face::Table(); return; }

    _num_glyphs_graphics = static_cast<unsigned short>(TtfUtil::GlyphCount(maxp));
    // This will fail if the number of glyphs is wildly out of range.
    if (_glyf && TtfUtil::LocaLookup(_num_glyphs_graphics-1, _loca, _loca.size(), _head) == size_t(-2))
    {
        _head = Face::Table();
        return;
    }

    if ((m_pGlat = Face::Table(face, Tag::Glat, 0x00030000)) == NULL
        || (m_pGloc = Face::Table(face, Tag::Gloc)) == NULL
        || m_pGloc.size() < 8)
    {
        _head = Face::Table();
        return;
    }
    const byte    * p = m_pGloc;
    int       version = be::read<uint32>(p);
    const uint16    flags = be::read<uint16>(p);
    _num_attrs = be::read<uint16>(p);
    // We can accurately calculate the number of attributed glyphs by
    //  subtracting the length of the attribids array (numAttribs long if present)
    //  and dividing by either 2 or 4 depending on shor or lonf format
    _long_fmt              = flags & 1;
    ptrdiff_t tmpnumgattrs       = (m_pGloc.size()
                               - (p - m_pGloc)
                               - sizeof(uint16)*(flags & 0x2 ? _num_attrs : 0))
                                   / (_long_fmt ? sizeof(uint32) : sizeof(uint16)) - 1;

    if (version >= 0x00020000 || tmpnumgattrs < 0 || tmpnumgattrs > 65535
        || _num_attrs == 0 || _num_attrs > 0x3000  // is this hard limit appropriate?
        || _num_glyphs_graphics > tmpnumgattrs
        || m_pGlat.size() < 4)
    {
        _head = Face::Table();
        return;
    }

    _num_glyphs_attributes = static_cast<unsigned short>(tmpnumgattrs);
    p = m_pGlat;
    version = be::read<uint32>(p);
    if (version >= 0x00040000 || (version >= 0x00030000 && m_pGlat.size() < 8))       // reject Glat tables that are too new
    {
        _head = Face::Table();
        return;
    }
    else if (version >= 0x00030000)
    {
        unsigned int glatflags = be::read<uint32>(p);
        _has_boxes = glatflags & 1;
        // delete this once the compiler is fixed
        _has_boxes = true;
    }
}

inline
GlyphCache::Loader::operator bool () const throw()
{
    return _head && _hhea && _hmtx && !(bool(_glyf) != bool(_loca));
}

inline
unsigned short int GlyphCache::Loader::units_per_em() const throw()
{
    return _head ? TtfUtil::DesignUnits(_head) : 0;
}

inline
unsigned short int GlyphCache::Loader::num_glyphs() const throw()
{
    return max(_num_glyphs_graphics, _num_glyphs_attributes);
}

inline
unsigned short int GlyphCache::Loader::num_attrs() const throw()
{
    return _num_attrs;
}

inline
bool GlyphCache::Loader::has_boxes () const throw()
{
    return _has_boxes;
}

const GlyphFace * GlyphCache::Loader::read_glyph(unsigned short glyphid, GlyphFace & glyph, int *numsubs) const throw()
{
    Rect        bbox;
    Position    advance;

    if (glyphid < _num_glyphs_graphics)
    {
        int nLsb;
        unsigned int nAdvWid;
        if (_glyf)
        {
            int xMin, yMin, xMax, yMax;
            size_t locidx = TtfUtil::LocaLookup(glyphid, _loca, _loca.size(), _head);
            void *pGlyph = TtfUtil::GlyfLookup(_glyf, locidx, _glyf.size());

            if (pGlyph && TtfUtil::GlyfBox(pGlyph, xMin, yMin, xMax, yMax))
            {
                if ((xMin > xMax) || (yMin > yMax))
                    return 0;
                bbox = Rect(Position(static_cast<float>(xMin), static_cast<float>(yMin)),
                    Position(static_cast<float>(xMax), static_cast<float>(yMax)));
            }
        }
        if (TtfUtil::HorMetrics(glyphid, _hmtx, _hmtx.size(), _hhea, nLsb, nAdvWid))
            advance = Position(static_cast<float>(nAdvWid), 0);
    }

    if (glyphid < _num_glyphs_attributes)
    {
        const byte * gloc = m_pGloc;
        size_t      glocs = 0, gloce = 0;

        be::skip<uint32>(gloc);
        be::skip<uint16>(gloc,2);
        if (_long_fmt)
        {
            if (8 + glyphid * sizeof(uint32) > m_pGloc.size())
                return 0;
            be::skip<uint32>(gloc, glyphid);
            glocs = be::read<uint32>(gloc);
            gloce = be::peek<uint32>(gloc);
        }
        else
        {
            if (8 + glyphid * sizeof(uint16) > m_pGloc.size())
                return 0;
            be::skip<uint16>(gloc, glyphid);
            glocs = be::read<uint16>(gloc);
            gloce = be::peek<uint16>(gloc);
        }

        if (glocs >= m_pGlat.size() - 1 || gloce > m_pGlat.size())
            return 0;

        const uint32 glat_version = be::peek<uint32>(m_pGlat);
        if (glat_version >= 0x00030000)
        {
            if (glocs >= gloce)
                return 0;
            const byte * p = m_pGlat + glocs;
            uint16 bmap = be::read<uint16>(p);
            int num = bit_set_count((uint32)bmap);
            if (numsubs) *numsubs += num;
            glocs += 6 + 8 * num;
            if (glocs > gloce)
                return 0;
        }
        if (glat_version < 0x00020000)
        {
            if (gloce - glocs < 2*sizeof(byte)+sizeof(uint16)
                || gloce - glocs > _num_attrs*(2*sizeof(byte)+sizeof(uint16)))
                    return 0;
            new (&glyph) GlyphFace(bbox, advance, glat_iterator(m_pGlat + glocs), glat_iterator(m_pGlat + gloce));
        }
        else
        {
            if (gloce - glocs < 3*sizeof(uint16)        // can a glyph have no attributes? why not?
                || gloce - glocs > _num_attrs*3*sizeof(uint16)
                || glocs > m_pGlat.size() - 2*sizeof(uint16))
                    return 0;
            new (&glyph) GlyphFace(bbox, advance, glat2_iterator(m_pGlat + glocs), glat2_iterator(m_pGlat + gloce));
        }
        if (!glyph.attrs() || glyph.attrs().capacity() > _num_attrs)
            return 0;
    }
    return &glyph;
}

inline float scale_to(uint8 t, float zmin, float zmax)
{
    return (zmin + t * (zmax - zmin) / 255);
}

Rect readbox(Rect &b, uint8 zxmin, uint8 zymin, uint8 zxmax, uint8 zymax)
{
    return Rect(Position(scale_to(zxmin, b.bl.x, b.tr.x), scale_to(zymin, b.bl.y, b.tr.y)),
                Position(scale_to(zxmax, b.bl.x, b.tr.x), scale_to(zymax, b.bl.y, b.tr.y)));
}

GlyphBox * GlyphCache::Loader::read_box(uint16 gid, GlyphBox *curr, const GlyphFace & glyph) const throw()
{
    if (gid >= _num_glyphs_attributes) return 0;

    const byte * gloc = m_pGloc;
    size_t      glocs = 0, gloce = 0;

    be::skip<uint32>(gloc);
    be::skip<uint16>(gloc,2);
    if (_long_fmt)
    {
        be::skip<uint32>(gloc, gid);
        glocs = be::read<uint32>(gloc);
        gloce = be::peek<uint32>(gloc);
    }
    else
    {
        be::skip<uint16>(gloc, gid);
        glocs = be::read<uint16>(gloc);
        gloce = be::peek<uint16>(gloc);
    }

    if (gloce > m_pGlat.size() || glocs + 6 >= gloce)
        return 0;

    const byte * p = m_pGlat + glocs;
    uint16 bmap = be::read<uint16>(p);
    int num = bit_set_count((uint32)bmap);

    Rect bbox = glyph.theBBox();
    Rect diamax(Position(bbox.bl.x + bbox.bl.y, bbox.bl.x - bbox.tr.y),
                Position(bbox.tr.x + bbox.tr.y, bbox.tr.x - bbox.bl.y));
    Rect diabound = readbox(diamax, p[0], p[2], p[1], p[3]);
    ::new (curr) GlyphBox(num, bmap, &diabound);
    be::skip<uint8>(p, 4);
    if (glocs + 6 + num * 8 >= gloce)
        return 0;

    for (int i = 0; i < num * 2; ++i)
    {
        Rect box = readbox((i & 1) ? diamax : bbox, p[0], p[2], p[1], p[3]);
        curr->addSubBox(i >> 1, i & 1, &box);
        be::skip<uint8>(p, 4);
    }
    return (GlyphBox *)((char *)(curr) + sizeof(GlyphBox) + 2 * num * sizeof(Rect));
}
