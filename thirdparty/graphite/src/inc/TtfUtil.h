// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.
/*
Responsibility: Alan Ward
Last reviewed: Not yet.

Description:
    Utility class for handling TrueType font files.
*/
#pragma once


#include <cstddef>

namespace graphite2
{
namespace TtfUtil
{

#define OVERFLOW_OFFSET_CHECK(p, o) (o + reinterpret_cast<size_t>(p) < reinterpret_cast<size_t>(p))

typedef long fontTableId32;
typedef unsigned short gid16;

#define TTF_TAG(a,b,c,d) ((a << 24UL) + (b << 16UL) + (c << 8UL) + (d))

// Enumeration used to specify a table in a TTF file
class Tag
{
    unsigned int _v;
public:
    Tag(const char n[5]) throw()            : _v(TTF_TAG(n[0],n[1],n[2],n[3])) {}
    Tag(const unsigned int tag) throw()    : _v(tag) {}

    operator unsigned int () const throw () { return _v; }

    enum
    {
        Feat = TTF_TAG('F','e','a','t'),
        Glat = TTF_TAG('G','l','a','t'),
        Gloc = TTF_TAG('G','l','o','c'),
        Sile = TTF_TAG('S','i','l','e'),
        Silf = TTF_TAG('S','i','l','f'),
        Sill = TTF_TAG('S','i','l','l'),
        cmap = TTF_TAG('c','m','a','p'),
        cvt  = TTF_TAG('c','v','t',' '),
        cryp = TTF_TAG('c','r','y','p'),
        head = TTF_TAG('h','e','a','d'),
        fpgm = TTF_TAG('f','p','g','m'),
        gdir = TTF_TAG('g','d','i','r'),
        glyf = TTF_TAG('g','l','y','f'),
        hdmx = TTF_TAG('h','d','m','x'),
        hhea = TTF_TAG('h','h','e','a'),
        hmtx = TTF_TAG('h','m','t','x'),
        loca = TTF_TAG('l','o','c','a'),
        kern = TTF_TAG('k','e','r','n'),
        LTSH = TTF_TAG('L','T','S','H'),
        maxp = TTF_TAG('m','a','x','p'),
        name = TTF_TAG('n','a','m','e'),
        OS_2 = TTF_TAG('O','S','/','2'),
        post = TTF_TAG('p','o','s','t'),
        prep = TTF_TAG('p','r','e','p')
    };
};

/*----------------------------------------------------------------------------------------------
    Class providing utility methods to parse a TrueType font file (TTF).
    Callling application handles all file input and memory allocation.
    Assumes minimal knowledge of TTF file format.
----------------------------------------------------------------------------------------------*/
    ////////////////////////////////// tools to find & check TTF tables
    bool GetHeaderInfo(size_t & lOffset, size_t & lSize);
    bool CheckHeader(const void * pHdr);
    bool GetTableDirInfo(const void * pHdr, size_t & lOffset, size_t & lSize);
    bool GetTableInfo(const Tag TableTag, const void * pHdr, const void * pTableDir,
        size_t & lOffset, size_t & lSize);
    bool CheckTable(const Tag TableId, const void * pTable, size_t lTableSize);

    ////////////////////////////////// simple font wide info
    size_t  GlyphCount(const void * pMaxp);
#ifdef ALL_TTFUTILS
    size_t  MaxCompositeComponentCount(const void * pMaxp);
    size_t  MaxCompositeLevelCount(const void * pMaxp);
    size_t  LocaGlyphCount(size_t lLocaSize, const void * pHead); // throw (std::domain_error);
#endif
    int DesignUnits(const void * pHead);
#ifdef ALL_TTFUTILS
    int HeadTableCheckSum(const void * pHead);
    void HeadTableCreateTime(const void * pHead, unsigned int * pnDateBC, unsigned int * pnDateAD);
    void HeadTableModifyTime(const void * pHead, unsigned int * pnDateBC, unsigned int * pnDateAD);
    bool IsItalic(const void * pHead);
    int FontAscent(const void * pOs2);
    int FontDescent(const void * pOs2);
    bool FontOs2Style(const void *pOs2, bool & fBold, bool & fItalic);
    bool Get31EngFamilyInfo(const void * pName, size_t & lOffset, size_t & lSize);
    bool Get31EngFullFontInfo(const void * pName, size_t & lOffset, size_t & lSize);
    bool Get30EngFamilyInfo(const void * pName, size_t & lOffset, size_t & lSize);
    bool Get30EngFullFontInfo(const void * pName, size_t & lOffset, size_t & lSize);
    int PostLookup(const void * pPost, size_t lPostSize, const void * pMaxp,
        const char * pPostName);
#endif

    ////////////////////////////////// utility methods helpful for name table
    bool GetNameInfo(const void * pName, int nPlatformId, int nEncodingId,
        int nLangId, int nNameId, size_t & lOffset, size_t & lSize);
    //size_t NameTableLength(const byte * pTable);
#ifdef ALL_TTFUTILS
    int GetLangsForNames(const void * pName, int nPlatformId, int nEncodingId,
        int *nameIdList, int cNameIds, short *langIdList);
    void SwapWString(void * pWStr, size_t nSize = 0); // throw (std::invalid_argument);
#endif

    ////////////////////////////////// cmap lookup tools
    const void * FindCmapSubtable(const void * pCmap, int nPlatformId = 3,
        int nEncodingId = 1, size_t length = 0);
    bool CheckCmapSubtable4(const void * pCmap31, const void * pCmapEnd /*, unsigned int maxgid*/);
    gid16 CmapSubtable4Lookup(const void * pCmapSubtabel4, unsigned int nUnicodeId, int rangeKey = 0);
    unsigned int CmapSubtable4NextCodepoint(const void *pCmap31, unsigned int nUnicodeId,
        int * pRangeKey = 0);
    bool CheckCmapSubtable12(const void *pCmap310, const void * pCmapEnd /*, unsigned int maxgid*/);
    gid16 CmapSubtable12Lookup(const void * pCmap310, unsigned int uUnicodeId, int rangeKey = 0);
    unsigned int CmapSubtable12NextCodepoint(const void *pCmap310, unsigned int nUnicodeId,
        int * pRangeKey = 0);

    ///////////////////////////////// horizontal metric data for a glyph
    bool HorMetrics(gid16 nGlyphId, const void * pHmtx, size_t lHmtxSize,
        const void * pHhea, int & nLsb, unsigned int & nAdvWid);

    ////////////////////////////////// primitives for loca and glyf lookup
    size_t LocaLookup(gid16 nGlyphId, const void * pLoca, size_t lLocaSize,
        const void * pHead); // throw (std::out_of_range);
    void * GlyfLookup(const void * pGlyf, size_t lGlyfOffset, size_t lTableLen);

    ////////////////////////////////// primitves for simple glyph data
    bool GlyfBox(const void * pSimpleGlyf, int & xMin, int & yMin,
        int & xMax, int & yMax);

#ifdef ALL_TTFUTILS
    int GlyfContourCount(const void * pSimpleGlyf);
    bool GlyfContourEndPoints(const void * pSimpleGlyf, int * prgnContourEndPoint,
        int cnPointsTotal, size_t & cnPoints);
    bool GlyfPoints(const void * pSimpleGlyf, int * prgnX, int * prgnY,
        char * prgbFlag, int cnPointsTotal, int & cnPoints);

    // primitive to find the glyph ids in a composite glyph
    bool GetComponentGlyphIds(const void * pSimpleGlyf, int * prgnCompId,
        size_t cnCompIdTotal, size_t & cnCompId);
    // primitive to find the placement data for a component in a composite glyph
    bool GetComponentPlacement(const void * pSimpleGlyf, int nCompId,
        bool fOffset, int & a, int & b);
    // primitive to find the transform data for a component in a composite glyph
    bool GetComponentTransform(const void * pSimpleGlyf, int nCompId,
        float & flt11, float & flt12, float & flt21, float & flt22, bool & fTransOffset);
#endif

    ////////////////////////////////// operate on composite or simple glyph (auto glyf lookup)
    void * GlyfLookup(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
        size_t lGlyfSize, size_t lLocaSize, const void * pHead); // primitive used by below methods

#ifdef ALL_TTFUTILS
    // below are primary user methods for handling glyf data
    bool IsSpace(gid16 nGlyphId, const void * pLoca, size_t lLocaSize, const void * pHead);
    bool IsDeepComposite(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
        size_t lGlyfSize, size_t lLocaSize, const void * pHead);

    bool GlyfBox(gid16 nGlyphId, const void * pGlyf, const void * pLoca, size_t lGlyfSize, size_t lLocaSize,
        const void * pHead, int & xMin, int & yMin, int & xMax, int & yMax);
    bool GlyfContourCount(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
        size_t lGlyfSize, size_t lLocaSize, const void *pHead, size_t & cnContours);
    bool GlyfContourEndPoints(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
        size_t lGlyfSize, size_t lLocaSize, const void * pHead, int * prgnContourEndPoint, size_t cnPoints);
    bool GlyfPoints(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
        size_t lGlyfSize, size_t lLocaSize, const void * pHead, const int * prgnContourEndPoint, size_t cnEndPoints,
        int * prgnX, int * prgnY, bool * prgfOnCurve, size_t cnPoints);

    // utitily method used by high-level GlyfPoints
    bool SimplifyFlags(char * prgbFlags, int cnPoints);
    bool CalcAbsolutePoints(int * prgnX, int * prgnY, int cnPoints);
#endif

} // end of namespace TtfUtil
} // end of namespace graphite2
