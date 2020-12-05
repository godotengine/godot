/*  GRAPHITE2 LICENSING

    Copyright 2010, SIL International
    All rights reserved.

    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation; either version 2.1 of License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should also have received a copy of the GNU Lesser General Public
    License along with this library in the file named "LICENSE".
    If not, write to the Free Software Foundation, 51 Franklin Street,
    Suite 500, Boston, MA 02110-1335, USA or visit their web page on the
    internet at http://www.fsf.org/licenses/lgpl.html.

Alternatively, the contents of this file may be used under the terms of the
Mozilla Public License (http://mozilla.org/MPL) or the GNU General Public
License, as published by the Free Software Foundation, either version 2
of the License or (at your option) any later version.
*/
#include <cstring>
#include "graphite2/Segment.h"
#include "inc/CmapCache.h"
#include "inc/debug.h"
#include "inc/Decompressor.h"
#include "inc/Endian.h"
#include "inc/Face.h"
#include "inc/FileFace.h"
#include "inc/GlyphFace.h"
#include "inc/json.h"
#include "inc/Segment.h"
#include "inc/NameTable.h"
#include "inc/Error.h"

using namespace graphite2;

namespace
{
enum compression
{
    NONE,
    LZ4
};

}

Face::Face(const void* appFaceHandle/*non-NULL*/, const gr_face_ops & ops)
: m_appFaceHandle(appFaceHandle),
  m_pFileFace(NULL),
  m_pGlyphFaceCache(NULL),
  m_cmap(NULL),
  m_pNames(NULL),
  m_logger(NULL),
  m_error(0), m_errcntxt(0),
  m_silfs(NULL),
  m_numSilf(0),
  m_ascent(0),
  m_descent(0)
{
    memset(&m_ops, 0, sizeof m_ops);
    memcpy(&m_ops, &ops, min(sizeof m_ops, ops.size));
}


Face::~Face()
{
    setLogger(0);
    delete m_pGlyphFaceCache;
    delete m_cmap;
    delete[] m_silfs;
#ifndef GRAPHITE2_NFILEFACE
    delete m_pFileFace;
#endif
    delete m_pNames;
}

float Face::default_glyph_advance(const void* font_ptr, gr_uint16 glyphid)
{
    const Font & font = *reinterpret_cast<const Font *>(font_ptr);

    return font.face().glyphs().glyph(glyphid)->theAdvance().x * font.scale();
}

bool Face::readGlyphs(uint32 faceOptions)
{
    Error e;
#ifdef GRAPHITE2_TELEMETRY
    telemetry::category _glyph_cat(tele.glyph);
#endif
    error_context(EC_READGLYPHS);
    m_pGlyphFaceCache = new GlyphCache(*this, faceOptions);

    if (e.test(!m_pGlyphFaceCache, E_OUTOFMEM)
        || e.test(m_pGlyphFaceCache->numGlyphs() == 0, E_NOGLYPHS)
        || e.test(m_pGlyphFaceCache->unitsPerEm() == 0, E_BADUPEM))
    {
        return error(e);
    }

    if (faceOptions & gr_face_cacheCmap)
        m_cmap = new CachedCmap(*this);
    else
        m_cmap = new DirectCmap(*this);
    if (e.test(!m_cmap, E_OUTOFMEM) || e.test(!*m_cmap, E_BADCMAP))
        return error(e);

    if (faceOptions & gr_face_preloadGlyphs)
        nameTable();        // preload the name table along with the glyphs.

    return true;
}

bool Face::readGraphite(const Table & silf)
{
#ifdef GRAPHITE2_TELEMETRY
    telemetry::category _silf_cat(tele.silf);
#endif
    Error e;
    error_context(EC_READSILF);
    const byte * p = silf;
    if (e.test(!p, E_NOSILF) || e.test(silf.size() < 20, E_BADSIZE)) return error(e);

    const uint32 version = be::read<uint32>(p);
    if (e.test(version < 0x00020000, E_TOOOLD)) return error(e);
    if (version >= 0x00030000)
        be::skip<uint32>(p);        // compilerVersion
    m_numSilf = be::read<uint16>(p);

    be::skip<uint16>(p);            // reserved

    bool havePasses = false;
    m_silfs = new Silf[m_numSilf];
    if (e.test(!m_silfs, E_OUTOFMEM)) return error(e);
    for (int i = 0; i < m_numSilf; i++)
    {
        error_context(EC_ASILF + (i << 8));
        const uint32 offset = be::read<uint32>(p),
                     next   = i == m_numSilf - 1 ? uint32(silf.size()) : be::peek<uint32>(p);
        if (e.test(next > silf.size() || offset >= next, E_BADSIZE))
            return error(e);

        if (!m_silfs[i].readGraphite(silf + offset, next - offset, *this, version))
            return false;

        if (m_silfs[i].numPasses())
            havePasses = true;
    }

    return havePasses;
}

bool Face::readFeatures()
{
    return m_Sill.readFace(*this);
}

bool Face::runGraphite(Segment *seg, const Silf *aSilf) const
{
#if !defined GRAPHITE2_NTRACING
    json * dbgout = logger();
    if (dbgout)
    {
        *dbgout << json::object
                    << "id"         << objectid(seg)
                    << "passes"     << json::array;
    }
#endif

//    if ((seg->dir() & 1) != aSilf->dir())
//        seg->reverseSlots();
    if ((seg->dir() & 3) == 3 && aSilf->bidiPass() == 0xFF)
        seg->doMirror(aSilf->aMirror());
    bool res = aSilf->runGraphite(seg, 0, aSilf->positionPass(), true);
    if (res)
    {
        seg->associateChars(0, seg->charInfoCount());
        if (aSilf->flags() & 0x20)
            res &= seg->initCollisions();
        if (res)
            res &= aSilf->runGraphite(seg, aSilf->positionPass(), aSilf->numPasses(), false);
    }

#if !defined GRAPHITE2_NTRACING
    if (dbgout)
{
        seg->positionSlots(0, 0, 0, seg->currdir());
        *dbgout             << json::item
                            << json::close // Close up the passes array
                << "outputdir" << (seg->currdir() ? "rtl" : "ltr")
                << "output" << json::array;
        for(Slot * s = seg->first(); s; s = s->next())
            *dbgout     << dslot(seg, s);
        *dbgout         << json::close
                << "advance" << seg->advance()
                << "chars"   << json::array;
        for(size_t i = 0, n = seg->charInfoCount(); i != n; ++i)
            *dbgout     << json::flat << *seg->charinfo(int(i));
        *dbgout         << json::close  // Close up the chars array
                    << json::close;     // Close up the segment object
    }
#endif

    return res;
}

void Face::setLogger(FILE * log_file GR_MAYBE_UNUSED)
{
#if !defined GRAPHITE2_NTRACING
    delete m_logger;
    m_logger = log_file ? new json(log_file) : 0;
#endif
}

const Silf *Face::chooseSilf(uint32 script) const
{
    if (m_numSilf == 0)
        return NULL;
    else if (m_numSilf == 1 || script == 0)
        return m_silfs;
    else // do more work here
        return m_silfs;
}

uint16 Face::findPseudo(uint32 uid) const
{
    return (m_numSilf) ? m_silfs[0].findPseudo(uid) : 0;
}

int32 Face::getGlyphMetric(uint16 gid, uint8 metric) const
{
    switch (metrics(metric))
    {
        case kgmetAscent : return m_ascent;
        case kgmetDescent : return m_descent;
        default:
            if (gid >= glyphs().numGlyphs()) return 0;
            return glyphs().glyph(gid)->getMetric(metric);
    }
}

void Face::takeFileFace(FileFace* pFileFace GR_MAYBE_UNUSED/*takes ownership*/)
{
#ifndef GRAPHITE2_NFILEFACE
    if (m_pFileFace==pFileFace)
      return;

    delete m_pFileFace;
    m_pFileFace = pFileFace;
#endif
}

NameTable * Face::nameTable() const
{
    if (m_pNames) return m_pNames;
    const Table name(*this, Tag::name);
    if (name)
        m_pNames = new NameTable(name, name.size());
    return m_pNames;
}

uint16 Face::languageForLocale(const char * locale) const
{
    nameTable();
    if (m_pNames)
        return m_pNames->getLanguageId(locale);
    return 0;
}



Face::Table::Table(const Face & face, const Tag n, uint32 version) throw()
: _f(&face), _sz(0), _compressed(false)
{
    _p = static_cast<const byte *>((*_f->m_ops.get_table)(_f->m_appFaceHandle, n, &_sz));

    if (!TtfUtil::CheckTable(n, _p, _sz))
    {
        release();     // Make sure we release the table buffer even if the table failed its checks
        return;
    }

    if (be::peek<uint32>(_p) >= version)
        decompress();
}

void Face::Table::release()
{
    if (_compressed)
        free(const_cast<byte *>(_p));
    else if (_p && _f->m_ops.release_table)
        (*_f->m_ops.release_table)(_f->m_appFaceHandle, _p);
    _p = 0; _sz = 0;
}

Face::Table & Face::Table::operator = (const Table && rhs) throw()
{
    if (this == &rhs)   return *this;
    release();
    new (this) Table(std::move(rhs));
    return *this;
}

Error Face::Table::decompress()
{
    Error e;
    if (e.test(_sz < 5 * sizeof(uint32), E_BADSIZE))
        return e;
    byte * uncompressed_table = 0;
    size_t uncompressed_size = 0;

    const byte * p = _p;
    const uint32 version = be::read<uint32>(p);    // Table version number.

    // The scheme is in the top 5 bits of the 1st uint32.
    const uint32 hdr = be::read<uint32>(p);
    switch(compression(hdr >> 27))
    {
    case NONE: return e;

    case LZ4:
    {
        uncompressed_size  = hdr & 0x07ffffff;
        uncompressed_table = gralloc<byte>(uncompressed_size);
        if (!e.test(!uncompressed_table || uncompressed_size < 4, E_OUTOFMEM))
        {
            memset(uncompressed_table, 0, 4);   // make sure version number is initialised
            // coverity[forward_null : FALSE] - uncompressed_table has been checked so can't be null
            // coverity[checked_return : FALSE] - we test e later
            e.test(lz4::decompress(p, _sz - 2*sizeof(uint32), uncompressed_table, uncompressed_size) != signed(uncompressed_size), E_SHRINKERFAILED);
        }
        break;
    }

    default:
        e.error(E_BADSCHEME);
    };

    // Check the uncompressed version number against the original.
    if (!e)
        // coverity[forward_null : FALSE] - uncompressed_table has already been tested so can't be null
        // coverity[checked_return : FALSE] - we test e later
        e.test(be::peek<uint32>(uncompressed_table) != version, E_SHRINKERFAILED);

    // Tell the provider to release the compressed form since were replacing
    //   it anyway.
    release();

    if (e)
    {
        free(uncompressed_table);
        uncompressed_table = 0;
        uncompressed_size  = 0;
    }

    _p = uncompressed_table;
    _sz = uncompressed_size;
    _compressed = true;

    return e;
}
