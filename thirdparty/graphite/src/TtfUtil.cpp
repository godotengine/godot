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
/*--------------------------------------------------------------------*//*:Ignore this sentence.

File: TtfUtil.cpp
Responsibility: Alan Ward
Last reviewed: Not yet.

Description
    Implements the methods for TtfUtil class. This file should remain portable to any C++
    environment by only using standard C++ and the TTF structurs defined in Tt.h.
-------------------------------------------------------------------------------*//*:End Ignore*/


/***********************************************************************************************
    Include files
***********************************************************************************************/
// Language headers
//#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <climits>
#include <cwchar>
//#include <stdexcept>
// Platform headers
// Module headers
#include "inc/TtfUtil.h"
#include "inc/TtfTypes.h"
#include "inc/Endian.h"

/***********************************************************************************************
    Forward declarations
***********************************************************************************************/

/***********************************************************************************************
    Local Constants and static variables
***********************************************************************************************/
namespace
{
#ifdef ALL_TTFUTILS
    // max number of components allowed in composite glyphs
    const int kMaxGlyphComponents = 8;
#endif

    template <int R, typename T>
    inline float fixed_to_float(const T f) {
        return float(f)/float(2^R);
    }

/*----------------------------------------------------------------------------------------------
    Table of standard Postscript glyph names. From Martin Hosken. Disagress with ttfdump.exe
---------------------------------------------------------------------------------------------*/
#ifdef ALL_TTFUTILS
    const int kcPostNames = 258;

    const char * rgPostName[kcPostNames] = {
        ".notdef", ".null", "nonmarkingreturn", "space", "exclam", "quotedbl", "numbersign",
        "dollar", "percent", "ampersand", "quotesingle", "parenleft",
        "parenright", "asterisk", "plus", "comma", "hyphen", "period", "slash",
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "colon", "semicolon", "less", "equal", "greater", "question",
        "at", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "bracketleft", "backslash", "bracketright", "asciicircum",
        "underscore", "grave", "a", "b", "c", "d", "e", "f", "g", "h", "i",
        "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
        "x", "y", "z", "braceleft", "bar", "braceright", "asciitilde",
        "Adieresis", "Aring", "Ccedilla", "Eacute", "Ntilde", "Odieresis",
        "Udieresis", "aacute", "agrave", "acircumflex", "adieresis", "atilde",
        "aring", "ccedilla", "eacute", "egrave", "ecircumflex", "edieresis",
        "iacute", "igrave", "icircumflex", "idieresis", "ntilde", "oacute",
        "ograve", "ocircumflex", "odieresis", "otilde", "uacute", "ugrave",
        "ucircumflex", "udieresis", "dagger", "degree", "cent", "sterling",
        "section", "bullet", "paragraph", "germandbls", "registered",
        "copyright", "trademark", "acute", "dieresis", "notequal", "AE",
        "Oslash", "infinity", "plusminus", "lessequal", "greaterequal", "yen",
        "mu", "partialdiff", "summation", "product", "pi", "integral",
        "ordfeminine", "ordmasculine", "Omega", "ae", "oslash", "questiondown",
        "exclamdown", "logicalnot", "radical", "florin", "approxequal",
        "Delta", "guillemotleft", "guillemotright", "ellipsis", "nonbreakingspace",
        "Agrave", "Atilde", "Otilde", "OE", "oe", "endash", "emdash",
        "quotedblleft", "quotedblright", "quoteleft", "quoteright", "divide",
        "lozenge", "ydieresis", "Ydieresis", "fraction", "currency",
        "guilsinglleft", "guilsinglright", "fi", "fl", "daggerdbl", "periodcentered",
        "quotesinglbase", "quotedblbase", "perthousand", "Acircumflex",
        "Ecircumflex", "Aacute", "Edieresis", "Egrave", "Iacute",
        "Icircumflex", "Idieresis", "Igrave", "Oacute", "Ocircumflex",
        "apple", "Ograve", "Uacute", "Ucircumflex", "Ugrave", "dotlessi",
        "circumflex", "tilde", "macron", "breve", "dotaccent", "ring",
        "cedilla", "hungarumlaut", "ogonek", "caron", "Lslash", "lslash",
        "Scaron", "scaron", "Zcaron", "zcaron", "brokenbar", "Eth", "eth",
        "Yacute", "yacute", "Thorn", "thorn", "minus", "multiply",
        "onesuperior", "twosuperior", "threesuperior", "onehalf", "onequarter",
        "threequarters", "franc", "Gbreve", "gbreve", "Idotaccent", "Scedilla",
        "scedilla", "Cacute", "cacute", "Ccaron", "ccaron",
        "dcroat" };
#endif

} // end of namespace

/***********************************************************************************************
    Methods
***********************************************************************************************/

/* Note on error processing: The code guards against bad glyph ids being used to look up data
in open ended tables (loca, hmtx). If the glyph id comes from a cmap this shouldn't happen
but it seems prudent to check for user errors here. The code does assume that data obtained
from the TTF file is valid otherwise (though the CheckTable method seeks to check for
obvious problems that might accompany a change in table versions). For example an invalid
offset in the loca table which could exceed the size of the glyf table is NOT trapped.
Likewise if numberOf_LongHorMetrics in the hhea table is wrong, this will NOT be trapped,
which could cause a lookup in the hmtx table to exceed the table length. Of course, TTF tables
that are completely corrupt will cause unpredictable results. */

/* Note on composite glyphs: Glyphs that have components that are themselves composites
are not supported. IsDeepComposite can be used to test for this. False is returned from many
of the methods in this cases. It is unclear how to build composite glyphs in some cases,
so this code represents my best guess until test cases can be found. See notes on the high-
level GlyfPoints method. */
namespace graphite2
{
namespace TtfUtil
{


/*----------------------------------------------------------------------------------------------
    Get offset and size of the offset table needed to find table directory.
    Return true if success, false otherwise.
    lSize excludes any table directory entries.
----------------------------------------------------------------------------------------------*/
bool GetHeaderInfo(size_t & lOffset, size_t & lSize)
{
    lOffset = 0;
    lSize   = offsetof(Sfnt::OffsetSubTable, table_directory);
    assert(sizeof(uint32) + 4*sizeof (uint16) == lSize);
    return true;
}

/*----------------------------------------------------------------------------------------------
    Check the offset table for expected data.
    Return true if success, false otherwise.
----------------------------------------------------------------------------------------------*/
bool CheckHeader(const void * pHdr)
{
    const Sfnt::OffsetSubTable * pOffsetTable
        = reinterpret_cast<const Sfnt::OffsetSubTable *>(pHdr);

    return pHdr && be::swap(pOffsetTable->scaler_type) == Sfnt::OffsetSubTable::TrueTypeWin;
}

/*----------------------------------------------------------------------------------------------
    Get offset and size of the table directory.
    Return true if successful, false otherwise.
----------------------------------------------------------------------------------------------*/
bool GetTableDirInfo(const void * pHdr, size_t & lOffset, size_t & lSize)
{
    const Sfnt::OffsetSubTable * pOffsetTable
        = reinterpret_cast<const Sfnt::OffsetSubTable *>(pHdr);

    lOffset = offsetof(Sfnt::OffsetSubTable, table_directory);
    lSize   = be::swap(pOffsetTable->num_tables)
        * sizeof(Sfnt::OffsetSubTable::Entry);

    return true;
}


/*----------------------------------------------------------------------------------------------
    Get offset and size of the specified table.
    Return true if successful, false otherwise. On false, offset and size will be 0.
----------------------------------------------------------------------------------------------*/
bool GetTableInfo(const Tag TableTag, const void * pHdr, const void * pTableDir,
                           size_t & lOffset, size_t & lSize)
{
    const Sfnt::OffsetSubTable * pOffsetTable
        = reinterpret_cast<const Sfnt::OffsetSubTable *>(pHdr);
    const size_t num_tables = be::swap(pOffsetTable->num_tables);
    const Sfnt::OffsetSubTable::Entry
        * entry_itr = reinterpret_cast<const Sfnt::OffsetSubTable::Entry *>(
            pTableDir),
        * const  dir_end = entry_itr + num_tables;

    if (num_tables > 40)
        return false;

    for (;entry_itr != dir_end; ++entry_itr) // 40 - safe guard
    {
        if (be::swap(entry_itr->tag) == TableTag)
        {
            lOffset = be::swap(entry_itr->offset);
            lSize   = be::swap(entry_itr->length);
            return true;
        }
    }

    return false;
}

/*----------------------------------------------------------------------------------------------
    Check the specified table. Tests depend on the table type.
    Return true if successful, false otherwise.
----------------------------------------------------------------------------------------------*/
bool CheckTable(const Tag TableId, const void * pTable, size_t lTableSize)
{
    using namespace Sfnt;

    if (pTable == 0 || lTableSize < 4) return false;

    switch(TableId)
    {
    case Tag::cmap: // cmap
    {
        const Sfnt::CharacterCodeMap * const pCmap
            = reinterpret_cast<const Sfnt::CharacterCodeMap *>(pTable);
        if (lTableSize < sizeof(Sfnt::CharacterCodeMap))
            return false;
        return be::swap(pCmap->version) == 0;
    }

    case Tag::head: // head
    {
        const Sfnt::FontHeader * const pHead
            = reinterpret_cast<const Sfnt::FontHeader *>(pTable);
        if (lTableSize < sizeof(Sfnt::FontHeader))
            return false;
        bool r = be::swap(pHead->version) == OneFix
            && be::swap(pHead->magic_number) == FontHeader::MagicNumber
            && be::swap(pHead->glyph_data_format)
                    == FontHeader::GlypDataFormat
            && (be::swap(pHead->index_to_loc_format)
                    == FontHeader::ShortIndexLocFormat
                || be::swap(pHead->index_to_loc_format)
                    == FontHeader::LongIndexLocFormat)
            && sizeof(FontHeader) <= lTableSize;
        return r;
    }

    case Tag::post: // post
    {
        const Sfnt::PostScriptGlyphName * const pPost
            = reinterpret_cast<const Sfnt::PostScriptGlyphName *>(pTable);
        if (lTableSize < sizeof(Sfnt::PostScriptGlyphName))
            return false;
        const fixed format = be::swap(pPost->format);
        bool r = format == PostScriptGlyphName::Format1
            || format == PostScriptGlyphName::Format2
            || format == PostScriptGlyphName::Format3
            || format == PostScriptGlyphName::Format25;
        return r;
    }

    case Tag::hhea: // hhea
    {
        const Sfnt::HorizontalHeader * pHhea =
            reinterpret_cast<const Sfnt::HorizontalHeader *>(pTable);
        if (lTableSize < sizeof(Sfnt::HorizontalHeader))
            return false;
        bool r = be::swap(pHhea->version) == OneFix
            && be::swap(pHhea->metric_data_format) == 0
            && sizeof (Sfnt::HorizontalHeader) <= lTableSize;
        return r;
    }

    case Tag::maxp: // maxp
    {
        const Sfnt::MaximumProfile * pMaxp =
            reinterpret_cast<const Sfnt::MaximumProfile *>(pTable);
        if (lTableSize < sizeof(Sfnt::MaximumProfile))
            return false;
        bool r = be::swap(pMaxp->version) == OneFix
            && sizeof(Sfnt::MaximumProfile) <= lTableSize;
        return r;
    }

    case Tag::OS_2: // OS/2
    {
        const Sfnt::Compatibility * pOs2
            = reinterpret_cast<const Sfnt::Compatibility *>(pTable);
        if (be::swap(pOs2->version) == 0)
        { // OS/2 table version 1 size
//          if (sizeof(Sfnt::Compatibility)
//                  - sizeof(uint32)*2 - sizeof(int16)*2
//                  - sizeof(uint16)*3 <= lTableSize)
            if (sizeof(Sfnt::Compatibility0) <= lTableSize)
                return true;
        }
        else if (be::swap(pOs2->version) == 1)
        { // OS/2 table version 2 size
//          if (sizeof(Sfnt::Compatibility)
//                  - sizeof(int16) *2
//                  - sizeof(uint16)*3 <= lTableSize)
            if (sizeof(Sfnt::Compatibility1) <= lTableSize)
                return true;
        }
        else if (be::swap(pOs2->version) == 2)
        { // OS/2 table version 3 size
            if (sizeof(Sfnt::Compatibility2) <= lTableSize)
                return true;
        }
        else if (be::swap(pOs2->version) == 3 || be::swap(pOs2->version) == 4)
        { // OS/2 table version 4 size - version 4 changed the meaning of some fields which we don't use
            if (sizeof(Sfnt::Compatibility3) <= lTableSize)
                return true;
        }
        else
            return false;
        break;
    }

    case Tag::name:
    {
        const Sfnt::FontNames * pName
            = reinterpret_cast<const Sfnt::FontNames *>(pTable);
        if (lTableSize < sizeof(Sfnt::FontNames))
            return false;
        return be::swap(pName->format) == 0;
    }

    case Tag::glyf:
    {
        return (lTableSize >= sizeof(Sfnt::Glyph));
    }

    default:
        break;
    }

    return true;
}

/*----------------------------------------------------------------------------------------------
    Return the number of glyphs in the font. Should never be less than zero.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
size_t GlyphCount(const void * pMaxp)
{
    const Sfnt::MaximumProfile * pTable =
            reinterpret_cast<const Sfnt::MaximumProfile *>(pMaxp);
    return be::swap(pTable->num_glyphs);
}

#ifdef ALL_TTFUTILS
/*----------------------------------------------------------------------------------------------
    Return the maximum number of components for any composite glyph in the font.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
size_t  MaxCompositeComponentCount(const void * pMaxp)
{
    const Sfnt::MaximumProfile * pTable =
            reinterpret_cast<const Sfnt::MaximumProfile *>(pMaxp);
    return be::swap(pTable->max_component_elements);
}

/*----------------------------------------------------------------------------------------------
    Composite glyphs can be composed of glyphs that are themselves composites.
    This method returns the maximum number of levels like this for any glyph in the font.
    A non-composite glyph has a level of 1.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
size_t  MaxCompositeLevelCount(const void * pMaxp)
{
    const Sfnt::MaximumProfile * pTable =
            reinterpret_cast<const Sfnt::MaximumProfile *>(pMaxp);
    return be::swap(pTable->max_component_depth);
}

/*----------------------------------------------------------------------------------------------
    Return the number of glyphs in the font according to a differt source.
    Should never be less than zero. Return -1 on failure.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
size_t LocaGlyphCount(size_t lLocaSize, const void * pHead) //throw(std::domain_error)
{

    const Sfnt::FontHeader * pTable
        = reinterpret_cast<const Sfnt::FontHeader *>(pHead);

    if (be::swap(pTable->index_to_loc_format)
        == Sfnt::FontHeader::ShortIndexLocFormat)
    // loca entries are two bytes and have been divided by two
        return (lLocaSize >> 1) - 1;

    if (be::swap(pTable->index_to_loc_format)
        == Sfnt::FontHeader::LongIndexLocFormat)
     // loca entries are four bytes
        return (lLocaSize >> 2) - 1;

    return -1;
    //throw std::domain_error("head table in inconsistent state. The font may be corrupted");
}
#endif

/*----------------------------------------------------------------------------------------------
    Return the design units the font is designed with
----------------------------------------------------------------------------------------------*/
int DesignUnits(const void * pHead)
{
    const Sfnt::FontHeader * pTable =
            reinterpret_cast<const Sfnt::FontHeader *>(pHead);

    return be::swap(pTable->units_per_em);
}

#ifdef ALL_TTFUTILS
/*----------------------------------------------------------------------------------------------
    Return the checksum from the head table, which serves as a unique identifer for the font.
----------------------------------------------------------------------------------------------*/
int HeadTableCheckSum(const void * pHead)
{
    const Sfnt::FontHeader * pTable =
            reinterpret_cast<const Sfnt::FontHeader *>(pHead);

    return be::swap(pTable->check_sum_adjustment);
}

/*----------------------------------------------------------------------------------------------
    Return the create time from the head table. This consists of a 64-bit integer, which
    we return here as two 32-bit integers.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
void HeadTableCreateTime(const void * pHead,
    unsigned int * pnDateBC, unsigned int * pnDateAD)
{
    const Sfnt::FontHeader * pTable =
            reinterpret_cast<const Sfnt::FontHeader *>(pHead);

    *pnDateBC = be::swap(pTable->created[0]);
    *pnDateAD = be::swap(pTable->created[1]);
}

/*----------------------------------------------------------------------------------------------
    Return the modify time from the head table.This consists of a 64-bit integer, which
    we return here as two 32-bit integers.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
void HeadTableModifyTime(const void * pHead,
    unsigned int * pnDateBC, unsigned int *pnDateAD)
{
    const Sfnt::FontHeader * pTable =
            reinterpret_cast<const Sfnt::FontHeader *>(pHead);
   ;
    *pnDateBC = be::swap(pTable->modified[0]);
    *pnDateAD = be::swap(pTable->modified[1]);
}

/*----------------------------------------------------------------------------------------------
    Return true if the font is italic.
----------------------------------------------------------------------------------------------*/
bool IsItalic(const void * pHead)
{
    const Sfnt::FontHeader * pTable =
            reinterpret_cast<const Sfnt::FontHeader *>(pHead);

    return ((be::swap(pTable->mac_style) & 0x00000002) != 0);
}

/*----------------------------------------------------------------------------------------------
    Return the ascent for the font
----------------------------------------------------------------------------------------------*/
int FontAscent(const void * pOs2)
{
    const Sfnt::Compatibility * pTable = reinterpret_cast<const Sfnt::Compatibility *>(pOs2);

    return be::swap(pTable->win_ascent);
}

/*----------------------------------------------------------------------------------------------
    Return the descent for the font
----------------------------------------------------------------------------------------------*/
int FontDescent(const void * pOs2)
{
    const Sfnt::Compatibility * pTable = reinterpret_cast<const Sfnt::Compatibility *>(pOs2);

    return be::swap(pTable->win_descent);
}

/*----------------------------------------------------------------------------------------------
    Get the bold and italic style bits.
    Return true if successful. false otherwise.
    In addition to checking the OS/2 table, one could also check
        the head table's macStyle field (overridden by the OS/2 table on Win)
        the sub-family name in the name table (though this can contain oblique, dark, etc too)
----------------------------------------------------------------------------------------------*/
bool FontOs2Style(const void *pOs2, bool & fBold, bool & fItalic)
{
    const Sfnt::Compatibility * pTable = reinterpret_cast<const Sfnt::Compatibility *>(pOs2);

    fBold = (be::swap(pTable->fs_selection) & Sfnt::Compatibility::Bold) != 0;
    fItalic = (be::swap(pTable->fs_selection) & Sfnt::Compatibility::Italic) != 0;

    return true;
}
#endif

/*----------------------------------------------------------------------------------------------
    Method for searching name table.
----------------------------------------------------------------------------------------------*/
bool GetNameInfo(const void * pName, int nPlatformId, int nEncodingId,
        int nLangId, int nNameId, size_t & lOffset, size_t & lSize)
{
    lOffset = 0;
    lSize = 0;

    const Sfnt::FontNames * pTable = reinterpret_cast<const Sfnt::FontNames *>(pName);
    uint16 cRecord = be::swap(pTable->count);
    uint16 nRecordOffset = be::swap(pTable->string_offset);
    const Sfnt::NameRecord * pRecord = reinterpret_cast<const Sfnt::NameRecord *>(pTable + 1);

    for (int i = 0; i < cRecord; ++i)
    {
        if (be::swap(pRecord->platform_id) == nPlatformId &&
            be::swap(pRecord->platform_specific_id) == nEncodingId &&
            be::swap(pRecord->language_id) == nLangId &&
            be::swap(pRecord->name_id) == nNameId)
        {
            lOffset = be::swap(pRecord->offset) + nRecordOffset;
            lSize = be::swap(pRecord->length);
            return true;
        }
        pRecord++;
    }

    return false;
}

#ifdef ALL_TTFUTILS
/*----------------------------------------------------------------------------------------------
    Return all the lang-IDs that have data for the given name-IDs. Assume that there is room
    in the return array (langIdList) for 128 items. The purpose of this method is to return
    a list of all possible lang-IDs.
----------------------------------------------------------------------------------------------*/
int GetLangsForNames(const void * pName, int nPlatformId, int nEncodingId,
        int * nameIdList, int cNameIds, short * langIdList)
{
    const Sfnt::FontNames * pTable = reinterpret_cast<const Sfnt::FontNames *>(pName);
        int cLangIds = 0;
    uint16 cRecord = be::swap(pTable->count);
        if (cRecord > 127) return cLangIds;
    //uint16 nRecordOffset = swapw(pTable->stringOffset);
    const Sfnt::NameRecord * pRecord = reinterpret_cast<const Sfnt::NameRecord *>(pTable + 1);

    for (int i = 0; i < cRecord; ++i)
    {
        if (be::swap(pRecord->platform_id) == nPlatformId &&
            be::swap(pRecord->platform_specific_id) == nEncodingId)
        {
            bool fNameFound = false;
            int nLangId = be::swap(pRecord->language_id);
            int nNameId = be::swap(pRecord->name_id);
            for (int j = 0; j < cNameIds; j++)
            {
                if (nNameId == nameIdList[j])
                {
                    fNameFound = true;
                    break;
                }
            }
            if (fNameFound)
            {
                // Add it if it's not there.
                int ilang;
                for (ilang = 0; ilang < cLangIds; ilang++)
                    if (langIdList[ilang] == nLangId)
                        break;
                if (ilang >= cLangIds)
                {
                    langIdList[cLangIds] = short(nLangId);
                    cLangIds++;
                }
                if (cLangIds == 128)
                    return cLangIds;
            }
        }
        pRecord++;
    }

    return cLangIds;
}

/*----------------------------------------------------------------------------------------------
    Get the offset and size of the font family name in English for the MS Platform with Unicode
    writing system. The offset is within the pName data. The string is double byte with MSB
    first.
----------------------------------------------------------------------------------------------*/
bool Get31EngFamilyInfo(const void * pName, size_t & lOffset, size_t & lSize)
{
    return GetNameInfo(pName, Sfnt::NameRecord::Microsoft, 1, 1033,
        Sfnt::NameRecord::Family, lOffset, lSize);
}

/*----------------------------------------------------------------------------------------------
    Get the offset and size of the full font name in English for the MS Platform with Unicode
    writing system. The offset is within the pName data. The string is double byte with MSB
    first.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
bool Get31EngFullFontInfo(const void * pName, size_t & lOffset, size_t & lSize)
{
    return GetNameInfo(pName, Sfnt::NameRecord::Microsoft, 1, 1033,
        Sfnt::NameRecord::Fullname, lOffset, lSize);
}

/*----------------------------------------------------------------------------------------------
    Get the offset and size of the font family name in English for the MS Platform with Symbol
    writing system. The offset is within the pName data. The string is double byte with MSB
    first.
----------------------------------------------------------------------------------------------*/
bool Get30EngFamilyInfo(const void * pName, size_t & lOffset, size_t & lSize)
{
    return GetNameInfo(pName, Sfnt::NameRecord::Microsoft, 0, 1033,
        Sfnt::NameRecord::Family, lOffset, lSize);
}

/*----------------------------------------------------------------------------------------------
    Get the offset and size of the full font name in English for the MS Platform with Symbol
    writing system. The offset is within the pName data. The string is double byte with MSB
    first.

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
bool Get30EngFullFontInfo(const void * pName, size_t & lOffset, size_t & lSize)
{
    return GetNameInfo(pName, Sfnt::NameRecord::Microsoft, 0, 1033,
        Sfnt::NameRecord::Fullname, lOffset, lSize);
}

/*----------------------------------------------------------------------------------------------
    Return the Glyph ID for a given Postscript name. This method finds the first glyph which
    matches the requested Postscript name. Ideally every glyph should have a unique Postscript
    name (except for special names such as .notdef), but this is not always true.
    On failure return value less than zero.
       -1 - table search failed
       -2 - format 3 table (no Postscript glyph info)
       -3 - other failures

    Note: this method is not currently used by the Graphite engine.
----------------------------------------------------------------------------------------------*/
int PostLookup(const void * pPost, size_t lPostSize, const void * pMaxp,
                        const char * pPostName)
{
    using namespace Sfnt;

    const Sfnt::PostScriptGlyphName * pTable
        = reinterpret_cast<const Sfnt::PostScriptGlyphName *>(pPost);
    fixed format = be::swap(pTable->format);

    if (format == PostScriptGlyphName::Format3)
    { // format 3 - no Postscript glyph info in font
        return -2;
    }

    // search for given Postscript name among the standard names
    int iPostName = -1; // index in standard names
    for (int i = 0; i < kcPostNames; i++)
    {
        if (!strcmp(pPostName, rgPostName[i]))
        {
            iPostName = i;
            break;
        }
    }

    if (format == PostScriptGlyphName::Format1)
    { // format 1 - use standard Postscript names
        return iPostName;
    }

    if (format == PostScriptGlyphName::Format25)
    {
        if (iPostName == -1)
            return -1;

        const PostScriptGlyphName25 * pTable25
            = static_cast<const PostScriptGlyphName25 *>(pTable);
        int cnGlyphs = GlyphCount(pMaxp);
        for (gid16 nGlyphId = 0; nGlyphId < cnGlyphs && nGlyphId < kcPostNames;
                nGlyphId++)
        { // glyph_name_index25 contains bytes so no byte swapping needed
          // search for first glyph id that uses the standard name
            if (nGlyphId + pTable25->offset[nGlyphId] == iPostName)
                return nGlyphId;
        }
    }

    if (format == PostScriptGlyphName::Format2)
    { // format 2
        const PostScriptGlyphName2 * pTable2
            = static_cast<const PostScriptGlyphName2 *>(pTable);

        int cnGlyphs = be::swap(pTable2->number_of_glyphs);

        if (iPostName != -1)
        { // did match a standard name, look for first glyph id mapped to that name
            for (gid16 nGlyphId = 0; nGlyphId < cnGlyphs; nGlyphId++)
            {
                if (be::swap(pTable2->glyph_name_index[nGlyphId]) == iPostName)
                    return nGlyphId;
            }
        }

        { // did not match a standard name, search font specific names
            size_t nStrSizeGoal = strlen(pPostName);
            const char * pFirstGlyphName = reinterpret_cast<const char *>(
                &pTable2->glyph_name_index[0] + cnGlyphs);
            const char * pGlyphName = pFirstGlyphName;
            int iInNames = 0; // index in font specific names
            bool fFound = false;
            const char * const endOfTable
                = reinterpret_cast<const char *>(pTable2) + lPostSize;
            while (pGlyphName < endOfTable && !fFound)
            { // search Pascal strings for first matching name
                size_t nStringSize = size_t(*pGlyphName);
                if (nStrSizeGoal != nStringSize ||
                    strncmp(pGlyphName + 1, pPostName, nStringSize))
                { // did not match
                    ++iInNames;
                    pGlyphName += nStringSize + 1;
                }
                else
                { // did match
                    fFound = true;
                }
            }
            if (!fFound)
                return -1; // no font specific name matches request

            iInNames += kcPostNames;
            for (gid16 nGlyphId = 0; nGlyphId < cnGlyphs; nGlyphId++)
            { // search for first glyph id that maps to the found string index
                if (be::swap(pTable2->glyph_name_index[nGlyphId]) == iInNames)
                    return nGlyphId;
            }
            return -1; // no glyph mapped to this index (very strange)
        }
    }

    return -3;
}

/*----------------------------------------------------------------------------------------------
    Convert a Unicode character string from big endian (MSB first, Motorola) format to little
    endian (LSB first, Intel) format.
    nSize is the number of Unicode characters in the string. It should not include any
    terminating null. If nSize is 0, it is assumed the string is null terminated. nSize
    defaults to 0.
    Return true if successful, false otherwise.
----------------------------------------------------------------------------------------------*/
void SwapWString(void * pWStr, size_t nSize /* = 0 */) //throw (std::invalid_argument)
{
    if (pWStr == 0)
    {
//      throw std::invalid_argument("null pointer given");
        return;
    }

    uint16 * pStr = reinterpret_cast<uint16 *>(pWStr);
    uint16 * const pStrEnd = pStr + (nSize == 0 ? wcslen((const wchar_t*)pStr) : nSize);

        for (; pStr != pStrEnd; ++pStr)
          *pStr = be::swap(*pStr);
//  std::transform(pStr, pStrEnd, pStr, read<uint16>);

//      for (int i = 0; i < nSize; i++)
//      { // swap the wide characters in the string
//          pStr[i] = utf16(be::swap(uint16(pStr[i])));
//      }
}
#endif

/*----------------------------------------------------------------------------------------------
    Get the left-side bearing and and advance width based on the given tables and Glyph ID
    Return true if successful, false otherwise. On false, one or both value could be INT_MIN
----------------------------------------------------------------------------------------------*/
bool HorMetrics(gid16 nGlyphId, const void * pHmtx, size_t lHmtxSize, const void * pHhea,
                         int & nLsb, unsigned int & nAdvWid)
{
    const Sfnt::HorizontalMetric * phmtx =
        reinterpret_cast<const Sfnt::HorizontalMetric *>(pHmtx);

    const Sfnt::HorizontalHeader * phhea =
        reinterpret_cast<const Sfnt::HorizontalHeader *>(pHhea);

    size_t cLongHorMetrics = be::swap(phhea->num_long_hor_metrics);
    if (nGlyphId < cLongHorMetrics)
    {   // glyph id is acceptable
        if ((nGlyphId + 1) * sizeof(Sfnt::HorizontalMetric) > lHmtxSize) return false;
        nAdvWid = be::swap(phmtx[nGlyphId].advance_width);
        nLsb = be::swap(phmtx[nGlyphId].left_side_bearing);
    }
    else
    {
        // guard against bad glyph id
        size_t lLsbOffset = sizeof(Sfnt::HorizontalMetric) * cLongHorMetrics +
            sizeof(int16) * (nGlyphId - cLongHorMetrics); // offset in bytes
        // We test like this as LsbOffset is an offset not a length.
        if (lLsbOffset >= lHmtxSize - sizeof(int16) || cLongHorMetrics == 0)
        {
            nLsb = 0;
            return false;
        }
        nAdvWid = be::swap(phmtx[cLongHorMetrics - 1].advance_width);
        nLsb = be::peek<int16>(reinterpret_cast<const byte *>(phmtx) + lLsbOffset);
    }

    return true;
}

/*----------------------------------------------------------------------------------------------
    Return a pointer to the requested cmap subtable. By default find the Microsoft Unicode
    subtable. Pass nEncoding as -1 to find first table that matches only nPlatformId.
    Return NULL if the subtable cannot be found.
----------------------------------------------------------------------------------------------*/
const void * FindCmapSubtable(const void * pCmap, int nPlatformId, /* =3 */ int nEncodingId, /* = 1 */ size_t length)
{
    const Sfnt::CharacterCodeMap * pTable = reinterpret_cast<const Sfnt::CharacterCodeMap *>(pCmap);
    uint16 csuPlatforms = be::swap(pTable->num_subtables);
    if (length && (sizeof(Sfnt::CharacterCodeMap) + 8 * (csuPlatforms - 1) > length))
        return NULL;
    for (int i = 0; i < csuPlatforms; i++)
    {
        if (be::swap(pTable->encoding[i].platform_id) == nPlatformId &&
                (nEncodingId == -1 || be::swap(pTable->encoding[i].platform_specific_id) == nEncodingId))
        {
            uint32 offset = be::swap(pTable->encoding[i].offset);
            const uint8 * pRtn = reinterpret_cast<const uint8 *>(pCmap) + offset;
            if (length)
            {
                if (offset > length - 2) return NULL;
                uint16 format = be::read<uint16>(pRtn);
                if (format == 4)
                {
                    if (offset > length - 4) return NULL;
                    uint16 subTableLength = be::peek<uint16>(pRtn);
                    if (i + 1 == csuPlatforms)
                    {
                        if (subTableLength > length - offset)
                            return NULL;
                    }
                    else if (subTableLength > be::swap(pTable->encoding[i+1].offset))
                        return NULL;
                }
                if (format == 12)
                {
                    if (offset > length - 6) return NULL;
                    uint32 subTableLength = be::peek<uint32>(pRtn);
                    if (i + 1 == csuPlatforms)
                    {
                        if (subTableLength > length - offset)
                            return NULL;
                    }
                    else if (subTableLength > be::swap(pTable->encoding[i+1].offset))
                        return NULL;
                }
            }
            return reinterpret_cast<const uint8 *>(pCmap) + offset;
        }
    }

    return 0;
}

/*----------------------------------------------------------------------------------------------
    Check the Microsoft Unicode subtable for expected values
----------------------------------------------------------------------------------------------*/
bool CheckCmapSubtable4(const void * pCmapSubtable4, const void * pCmapEnd /*, unsigned int maxgid*/)
{
    size_t table_len = (const byte *)pCmapEnd - (const byte *)pCmapSubtable4;
    if (!pCmapSubtable4) return false;
    const Sfnt::CmapSubTable * pTable = reinterpret_cast<const Sfnt::CmapSubTable *>(pCmapSubtable4);
    // Bob H say some freeware TT fonts have version 1 (eg, CALIGULA.TTF)
    // so don't check subtable version. 21 Mar 2002 spec changes version to language.
    if (table_len < sizeof(*pTable) || be::swap(pTable->format) != 4) return false;
    const Sfnt::CmapSubTableFormat4 * pTable4 = reinterpret_cast<const Sfnt::CmapSubTableFormat4 *>(pCmapSubtable4);
    if (table_len < sizeof(*pTable4))
        return false;
    uint16 length = be::swap(pTable4->length);
    if (length > table_len)
        return false;
    if (length < sizeof(Sfnt::CmapSubTableFormat4))
        return false;
    uint16 nRanges = be::swap(pTable4->seg_count_x2) >> 1;
    if (!nRanges || length < sizeof(Sfnt::CmapSubTableFormat4) + 4 * nRanges * sizeof(uint16))
        return false;
    // check last range is properly terminated
    uint16 chEnd = be::peek<uint16>(pTable4->end_code + nRanges - 1);
    if (chEnd != 0xFFFF)
        return false;
#if 0
    int lastend = -1;
    for (int i = 0; i < nRanges; ++i)
    {
        uint16 end = be::peek<uint16>(pTable4->end_code + i);
        uint16 start = be::peek<uint16>(pTable4->end_code + nRanges + 1 + i);
        int16 delta = be::peek<int16>(pTable4->end_code + 2*nRanges + 1 + i);
        uint16 offset = be::peek<uint16>(pTable4->end_code + 3*nRanges + 1 + i);
        if (lastend >= end || lastend >= start)
            return false;
        if (offset)
        {
            const uint16 *gstart = pTable4->end_code + 3*nRanges + 1 + i + (offset >> 1);
            const uint16 *gend = gstart + end - start;
            if ((char *)gend >= (char *)pCmapSubtable4 + length)
                return false;
            while (gstart <= gend)
            {
                uint16 g = be::peek<uint16>(gstart++);
                if (g && ((g + delta) & 0xFFFF) > maxgid)
                    return false;
            }
        }
        else if (((delta + end) & 0xFFFF) > maxgid)
            return false;
        lastend = end;
    }
#endif
    return true;
}

/*----------------------------------------------------------------------------------------------
    Return the Glyph ID for the given Unicode ID in the Microsoft Unicode subtable.
    (Actually this code only depends on subtable being format 4.)
    Return 0 if the Unicode ID is not in the subtable.
----------------------------------------------------------------------------------------------*/
gid16 CmapSubtable4Lookup(const void * pCmapSubtabel4, unsigned int nUnicodeId, int rangeKey)
{
    const Sfnt::CmapSubTableFormat4 * pTable = reinterpret_cast<const Sfnt::CmapSubTableFormat4 *>(pCmapSubtabel4);

    uint16 nSeg = be::swap(pTable->seg_count_x2) >> 1;

    uint16 n;
    const uint16 * pLeft, * pMid;
    uint16 cMid, chStart, chEnd;

    if (rangeKey)
    {
        pMid = &(pTable->end_code[rangeKey]);
        chEnd = be::peek<uint16>(pMid);
    }
    else
    {
        // Binary search of the endCode[] array
        pLeft = &(pTable->end_code[0]);
        n = nSeg;
        while (n > 0)
        {
            cMid = n >> 1;           // Pick an element in the middle
            pMid = pLeft + cMid;
            chEnd = be::peek<uint16>(pMid);
            if (nUnicodeId <= chEnd)
            {
                if (cMid == 0 || nUnicodeId > be::peek<uint16>(pMid -1))
                        break;          // Must be this seg or none!
                n = cMid;            // Continue on left side, omitting mid point
            }
            else
            {
                pLeft = pMid + 1;    // Continue on right side, omitting mid point
                n -= (cMid + 1);
            }
        }

        if (!n)
        return 0;
    }

    // Ok, we're down to one segment and pMid points to the endCode element
    // Either this is it or none is.

    chStart = be::peek<uint16>(pMid += nSeg + 1);
    if (chEnd >= nUnicodeId && nUnicodeId >= chStart)
    {
        // Found correct segment. Find Glyph Id
        int16 idDelta = be::peek<uint16>(pMid += nSeg);
        uint16 idRangeOffset = be::peek<uint16>(pMid += nSeg);

        if (idRangeOffset == 0)
            return (uint16)(idDelta + nUnicodeId); // must use modulus 2^16

        // Look up value in glyphIdArray
        const ptrdiff_t offset = (nUnicodeId - chStart) + (idRangeOffset >> 1) +
                (pMid - reinterpret_cast<const uint16 *>(pTable));
        if (offset * 2 + 1 >= be::swap<uint16>(pTable->length))
            return 0;
        gid16 nGlyphId = be::peek<uint16>(reinterpret_cast<const uint16 *>(pTable)+offset);
        // If this value is 0, return 0. Else add the idDelta
        return nGlyphId ? nGlyphId + idDelta : 0;
    }

    return 0;
}

/*----------------------------------------------------------------------------------------------
    Return the next Unicode value in the cmap. Pass 0 to obtain the first item.
    Returns 0xFFFF as the last item.
    pRangeKey is an optional key that is used to optimize the search; its value is the range
    in which the character is found.
----------------------------------------------------------------------------------------------*/
unsigned int CmapSubtable4NextCodepoint(const void *pCmap31, unsigned int nUnicodeId, int * pRangeKey)
{
    const Sfnt::CmapSubTableFormat4 * pTable = reinterpret_cast<const Sfnt::CmapSubTableFormat4 *>(pCmap31);

    uint16 nRange = be::swap(pTable->seg_count_x2) >> 1;

    uint32 nUnicodePrev = (uint32)nUnicodeId;

    const uint16 * pStartCode = &(pTable->end_code[0])
        + nRange // length of end code array
        + 1;   // reserved word

    if (nUnicodePrev == 0)
    {
        // return the first codepoint.
        if (pRangeKey)
            *pRangeKey = 0;
        return be::peek<uint16>(pStartCode);
    }
    else if (nUnicodePrev >= 0xFFFF)
    {
        if (pRangeKey)
            *pRangeKey = nRange - 1;
        return 0xFFFF;
    }

    int iRange = (pRangeKey) ? *pRangeKey : 0;
    // Just in case we have a bad key:
    while (iRange > 0 && be::peek<uint16>(pStartCode + iRange) > nUnicodePrev)
        iRange--;
    while (iRange < nRange - 1 && be::peek<uint16>(pTable->end_code + iRange) < nUnicodePrev)
        iRange++;

    // Now iRange is the range containing nUnicodePrev.
    unsigned int nStartCode = be::peek<uint16>(pStartCode + iRange);
    unsigned int nEndCode = be::peek<uint16>(pTable->end_code + iRange);

    if (nStartCode > nUnicodePrev)
        // Oops, nUnicodePrev is not in the cmap! Adjust so we get a reasonable
        // answer this time around.
        nUnicodePrev = nStartCode - 1;

    if (nEndCode > nUnicodePrev)
    {
        // Next is in the same range; it is the next successive codepoint.
        if (pRangeKey)
            *pRangeKey = iRange;
        return nUnicodePrev + 1;
    }

    // Otherwise the next codepoint is the first one in the next range.
    // There is guaranteed to be a next range because there must be one that
    // ends with 0xFFFF.
    if (pRangeKey)
        *pRangeKey = iRange + 1;
    return (iRange + 1 >= nRange) ? 0xFFFF : be::peek<uint16>(pStartCode + iRange + 1);
}

/*----------------------------------------------------------------------------------------------
    Check the Microsoft UCS-4 subtable for expected values.
----------------------------------------------------------------------------------------------*/
bool CheckCmapSubtable12(const void *pCmapSubtable12, const void *pCmapEnd /*, unsigned int maxgid*/)
{
    size_t table_len = (const byte *)pCmapEnd - (const byte *)pCmapSubtable12;
    if (!pCmapSubtable12)  return false;
    const Sfnt::CmapSubTable * pTable = reinterpret_cast<const Sfnt::CmapSubTable *>(pCmapSubtable12);
    if (table_len < sizeof(*pTable) || be::swap(pTable->format) != 12)
        return false;
    const Sfnt::CmapSubTableFormat12 * pTable12 = reinterpret_cast<const Sfnt::CmapSubTableFormat12 *>(pCmapSubtable12);
    if (table_len < sizeof(*pTable12))
        return false;
    uint32 length = be::swap(pTable12->length);
    if (length > table_len)
        return false;
    if (length < sizeof(Sfnt::CmapSubTableFormat12))
        return false;
    uint32 num_groups = be::swap(pTable12->num_groups);
    if (num_groups > 0x10000000 || length != (sizeof(Sfnt::CmapSubTableFormat12) + (num_groups - 1) * sizeof(uint32) * 3))
        return false;
#if 0
    for (unsigned int i = 0; i < num_groups; ++i)
    {
        if (be::swap(pTable12->group[i].end_char_code)  - be::swap(pTable12->group[i].start_char_code) + be::swap(pTable12->group[i].start_glyph_id) > maxgid)
            return false;
        if (i > 0 && be::swap(pTable12->group[i].start_char_code) <= be::swap(pTable12->group[i-1].end_char_code))
            return false;
    }
#endif
    return true;
}

/*----------------------------------------------------------------------------------------------
    Return the Glyph ID for the given Unicode ID in the Microsoft UCS-4 subtable.
    (Actually this code only depends on subtable being format 12.)
    Return 0 if the Unicode ID is not in the subtable.
----------------------------------------------------------------------------------------------*/
gid16 CmapSubtable12Lookup(const void * pCmap310, unsigned int uUnicodeId, int rangeKey)
{
    const Sfnt::CmapSubTableFormat12 * pTable = reinterpret_cast<const Sfnt::CmapSubTableFormat12 *>(pCmap310);

    //uint32 uLength = be::swap(pTable->length); //could use to test for premature end of table
    uint32 ucGroups = be::swap(pTable->num_groups);

    for (unsigned int i = rangeKey; i < ucGroups; i++)
    {
        uint32 uStartCode = be::swap(pTable->group[i].start_char_code);
        uint32 uEndCode = be::swap(pTable->group[i].end_char_code);
        if (uUnicodeId >= uStartCode && uUnicodeId <= uEndCode)
        {
            uint32 uDiff = uUnicodeId - uStartCode;
            uint32 uStartGid = be::swap(pTable->group[i].start_glyph_id);
            return static_cast<gid16>(uStartGid + uDiff);
        }
    }

    return 0;
}

/*----------------------------------------------------------------------------------------------
    Return the next Unicode value in the cmap. Pass 0 to obtain the first item.
    Returns 0x10FFFF as the last item.
    pRangeKey is an optional key that is used to optimize the search; its value is the range
    in which the character is found.
----------------------------------------------------------------------------------------------*/
unsigned int CmapSubtable12NextCodepoint(const void *pCmap310, unsigned int nUnicodeId, int * pRangeKey)
{
    const Sfnt::CmapSubTableFormat12 * pTable = reinterpret_cast<const Sfnt::CmapSubTableFormat12 *>(pCmap310);

    int nRange = be::swap(pTable->num_groups);

    uint32 nUnicodePrev = (uint32)nUnicodeId;

    if (nUnicodePrev == 0)
    {
        // return the first codepoint.
        if (pRangeKey)
            *pRangeKey = 0;
        return be::swap(pTable->group[0].start_char_code);
    }
    else if (nUnicodePrev >= 0x10FFFF)
    {
        if (pRangeKey)
            *pRangeKey = nRange;
        return 0x10FFFF;
    }

    int iRange = (pRangeKey) ? *pRangeKey : 0;
    // Just in case we have a bad key:
    while (iRange > 0 && be::swap(pTable->group[iRange].start_char_code) > nUnicodePrev)
        iRange--;
    while (iRange < nRange - 1 && be::swap(pTable->group[iRange].end_char_code) < nUnicodePrev)
        iRange++;

    // Now iRange is the range containing nUnicodePrev.

    unsigned int nStartCode = be::swap(pTable->group[iRange].start_char_code);
    unsigned int nEndCode = be::swap(pTable->group[iRange].end_char_code);

    if (nStartCode > nUnicodePrev)
        // Oops, nUnicodePrev is not in the cmap! Adjust so we get a reasonable
        // answer this time around.
        nUnicodePrev = nStartCode - 1;

    if (nEndCode > nUnicodePrev)
    {
        // Next is in the same range; it is the next successive codepoint.
        if (pRangeKey)
            *pRangeKey = iRange;
        return nUnicodePrev + 1;
    }

    // Otherwise the next codepoint is the first one in the next range, or 10FFFF if we're done.
    if (pRangeKey)
        *pRangeKey = iRange + 1;
    return (iRange + 1 >= nRange) ? 0x10FFFF : be::swap(pTable->group[iRange + 1].start_char_code);
}

/*----------------------------------------------------------------------------------------------
    Return the offset stored in the loca table for the given Glyph ID.
    (This offset is into the glyf table.)
    Return -1 if the lookup failed.
    Technically this method should return an unsigned long but it is unlikely the offset will
        exceed 2^31.
----------------------------------------------------------------------------------------------*/
size_t LocaLookup(gid16 nGlyphId,
        const void * pLoca, size_t lLocaSize,
        const void * pHead) // throw (std::out_of_range)
{
    const Sfnt::FontHeader * pTable = reinterpret_cast<const Sfnt::FontHeader *>(pHead);
    size_t res = -2;

    // CheckTable verifies the index_to_loc_format is valid
    if (be::swap(pTable->index_to_loc_format) == Sfnt::FontHeader::ShortIndexLocFormat)
    { // loca entries are two bytes and have been divided by two
        if (lLocaSize > 1 && nGlyphId + 1u < lLocaSize >> 1) // allow sentinel value to be accessed
        {
            const uint16 * pShortTable = reinterpret_cast<const uint16 *>(pLoca);
            res = be::peek<uint16>(pShortTable + nGlyphId) << 1;
            if (res == static_cast<size_t>(be::peek<uint16>(pShortTable + nGlyphId + 1) << 1))
                return -1;
        }
    }
    else if (be::swap(pTable->index_to_loc_format) == Sfnt::FontHeader::LongIndexLocFormat)
    { // loca entries are four bytes
        if (lLocaSize > 3 && nGlyphId + 1u < lLocaSize >> 2)
        {
            const uint32 * pLongTable = reinterpret_cast<const uint32 *>(pLoca);
            res = be::peek<uint32>(pLongTable + nGlyphId);
            if (res == static_cast<size_t>(be::peek<uint32>(pLongTable + nGlyphId + 1)))
                return -1;
        }
    }

    // only get here if glyph id was bad
    return res;
    //throw std::out_of_range("glyph id out of range for font");
}

/*----------------------------------------------------------------------------------------------
    Return a pointer into the glyf table based on the given offset (from LocaLookup).
    Return NULL on error.
----------------------------------------------------------------------------------------------*/
void * GlyfLookup(const void * pGlyf, size_t nGlyfOffset, size_t nTableLen)
{
    const uint8 * pByte = reinterpret_cast<const uint8 *>(pGlyf);
    if (OVERFLOW_OFFSET_CHECK(pByte, nGlyfOffset) || nGlyfOffset >= nTableLen - sizeof(Sfnt::Glyph))
        return NULL;
    return const_cast<uint8 *>(pByte + nGlyfOffset);
}

/*----------------------------------------------------------------------------------------------
    Get the bounding box coordinates for a simple glyf entry (non-composite).
    Return true if successful, false otherwise.
----------------------------------------------------------------------------------------------*/
bool GlyfBox(const void * pSimpleGlyf, int & xMin, int & yMin,
                      int & xMax, int & yMax)
{
    const Sfnt::Glyph * pGlyph = reinterpret_cast<const Sfnt::Glyph *>(pSimpleGlyf);

    xMin = be::swap(pGlyph->x_min);
    yMin = be::swap(pGlyph->y_min);
    xMax = be::swap(pGlyph->x_max);
    yMax = be::swap(pGlyph->y_max);

    return true;
}

#ifdef ALL_TTFUTILS
/*----------------------------------------------------------------------------------------------
    Return the number of contours for a simple glyf entry (non-composite)
    Returning -1 means this is a composite glyph
----------------------------------------------------------------------------------------------*/
int GlyfContourCount(const void * pSimpleGlyf)
{
    const Sfnt::Glyph * pGlyph = reinterpret_cast<const Sfnt::Glyph *>(pSimpleGlyf);
    return be::swap(pGlyph->number_of_contours); // -1 means composite glyph
}

/*----------------------------------------------------------------------------------------------
    Get the point numbers for the end points of the glyph contours for a simple
    glyf entry (non-composite).
    cnPointsTotal - count of contours from GlyfContourCount(); (same as number of end points)
    prgnContourEndPoints - should point to a buffer large enough to hold cnPoints integers
    cnPoints - count of points placed in above range
    Return true if successful, false otherwise.
        False could indicate a multi-level composite glyphs.
----------------------------------------------------------------------------------------------*/
bool GlyfContourEndPoints(const void * pSimpleGlyf, int * prgnContourEndPoint,
                                   int cnPointsTotal, int & cnPoints)
{
    const Sfnt::SimpleGlyph * pGlyph = reinterpret_cast<const Sfnt::SimpleGlyph *>(pSimpleGlyf);

    int cContours = be::swap(pGlyph->number_of_contours);
    if (cContours < 0)
        return false; // this method isn't supposed handle composite glyphs

    for (int i = 0; i < cContours && i < cnPointsTotal; i++)
    {
        prgnContourEndPoint[i] = be::swap(pGlyph->end_pts_of_contours[i]);
    }

    cnPoints = cContours;
    return true;
}

/*----------------------------------------------------------------------------------------------
    Get the points for a simple glyf entry (non-composite)
    cnPointsTotal - count of points from largest end point obtained from GlyfContourEndPoints
    prgnX & prgnY - should point to buffers large enough to hold cnPointsTotal integers
        The ranges are parallel so that coordinates for point(n) are found at offset n in both
        ranges. This is raw point data with relative coordinates.
    prgbFlag - should point to a buffer a large enough to hold cnPointsTotal bytes
        This range is parallel to the prgnX & prgnY
    cnPoints - count of points placed in above ranges
    Return true if successful, false otherwise.
        False could indicate a composite glyph
----------------------------------------------------------------------------------------------*/
bool GlyfPoints(const void * pSimpleGlyf, int * prgnX, int * prgnY,
        char * prgbFlag, int cnPointsTotal, int & cnPoints)
{
    using namespace Sfnt;

    const Sfnt::SimpleGlyph * pGlyph = reinterpret_cast<const Sfnt::SimpleGlyph *>(pSimpleGlyf);
    int cContours = be::swap(pGlyph->number_of_contours);
    // return false for composite glyph
    if (cContours <= 0)
        return false;
    int cPts = be::swap(pGlyph->end_pts_of_contours[cContours - 1]) + 1;
    if (cPts > cnPointsTotal)
        return false;

    // skip over bounding box data & point to byte count of instructions (hints)
    const uint8 * pbGlyph = reinterpret_cast<const uint8 *>
        (&pGlyph->end_pts_of_contours[cContours]);

    // skip over hints & point to first flag
    int cbHints = be::swap(*(uint16 *)pbGlyph);
    pbGlyph += sizeof(uint16);
    pbGlyph += cbHints;

    // load flags & point to first x coordinate
    int iFlag = 0;
    while (iFlag < cPts)
    {
        if (!(*pbGlyph & SimpleGlyph::Repeat))
        { // flag isn't repeated
            prgbFlag[iFlag] = (char)*pbGlyph;
            pbGlyph++;
            iFlag++;
        }
        else
        { // flag is repeated; count specified by next byte
            char chFlag = (char)*pbGlyph;
            pbGlyph++;
            int cFlags = (int)*pbGlyph;
            pbGlyph++;
            prgbFlag[iFlag] = chFlag;
            iFlag++;
            for (int i = 0; i < cFlags; i++)
            {
                prgbFlag[iFlag + i] = chFlag;
            }
            iFlag += cFlags;
        }
    }
    if (iFlag != cPts)
        return false;

    // load x coordinates
    iFlag = 0;
    while (iFlag < cPts)
    {
        if (prgbFlag[iFlag] & SimpleGlyph::XShort)
        {
            prgnX[iFlag] = *pbGlyph;
            if (!(prgbFlag[iFlag] & SimpleGlyph::XIsPos))
            {
                prgnX[iFlag] = -prgnX[iFlag];
            }
            pbGlyph++;
        }
        else
        {
            if (prgbFlag[iFlag] & SimpleGlyph::XIsSame)
            {
                prgnX[iFlag] = 0;
                // do NOT increment pbGlyph
            }
            else
            {
                prgnX[iFlag] = be::swap(*(int16 *)pbGlyph);
                pbGlyph += sizeof(int16);
            }
        }
        iFlag++;
    }

    // load y coordinates
    iFlag = 0;
    while (iFlag < cPts)
    {
        if (prgbFlag[iFlag] & SimpleGlyph::YShort)
        {
            prgnY[iFlag] = *pbGlyph;
            if (!(prgbFlag[iFlag] & SimpleGlyph::YIsPos))
            {
                prgnY[iFlag] = -prgnY[iFlag];
            }
            pbGlyph++;
        }
        else
        {
            if (prgbFlag[iFlag] & SimpleGlyph::YIsSame)
            {
                prgnY[iFlag] = 0;
                // do NOT increment pbGlyph
            }
            else
            {
                prgnY[iFlag] = be::swap(*(int16 *)pbGlyph);
                pbGlyph += sizeof(int16);
            }
        }
        iFlag++;
    }

    cnPoints = cPts;
    return true;
}

/*----------------------------------------------------------------------------------------------
    Fill prgnCompId with the component Glyph IDs from pSimpleGlyf.
    Client must allocate space before calling.
    pSimpleGlyf - assumed to point to a composite glyph
    cCompIdTotal - the number of elements in prgnCompId
    cCompId  - the total number of Glyph IDs stored in prgnCompId
    Return true if successful, false otherwise
        False could indicate a non-composite glyph or the input array was not big enough
----------------------------------------------------------------------------------------------*/
bool GetComponentGlyphIds(const void * pSimpleGlyf, int * prgnCompId,
        size_t cnCompIdTotal, size_t & cnCompId)
{
    using namespace Sfnt;

    if (GlyfContourCount(pSimpleGlyf) >= 0)
        return false;

    const Sfnt::SimpleGlyph * pGlyph = reinterpret_cast<const Sfnt::SimpleGlyph *>(pSimpleGlyf);
    // for a composite glyph, the special data begins here
    const uint8 * pbGlyph = reinterpret_cast<const uint8 *>(&pGlyph->end_pts_of_contours[0]);

    uint16 GlyphFlags;
    size_t iCurrentComp = 0;
    do
    {
        GlyphFlags = be::swap(*((uint16 *)pbGlyph));
        pbGlyph += sizeof(uint16);
        prgnCompId[iCurrentComp++] = be::swap(*((uint16 *)pbGlyph));
        pbGlyph += sizeof(uint16);
        if (iCurrentComp >= cnCompIdTotal)
            return false;
        int nOffset = 0;
        nOffset += GlyphFlags & CompoundGlyph::Arg1Arg2Words ? 4 : 2;
        nOffset += GlyphFlags & CompoundGlyph::HaveScale ? 2 : 0;
        nOffset += GlyphFlags & CompoundGlyph::HaveXAndYScale  ? 4 : 0;
        nOffset += GlyphFlags & CompoundGlyph::HaveTwoByTwo  ? 8 :  0;
        pbGlyph += nOffset;
    } while (GlyphFlags & CompoundGlyph::MoreComponents);

    cnCompId = iCurrentComp;

    return true;
}

/*----------------------------------------------------------------------------------------------
    Return info on how a component glyph is to be placed
    pSimpleGlyph - assumed to point to a composite glyph
    nCompId - glyph id for component of interest
    bOffset - if true, a & b are the x & y offsets for this component
              if false, b is the point on this component that is attaching to point a on the
                preceding glyph
    Return true if successful, false otherwise
        False could indicate a non-composite glyph or that component wasn't found
----------------------------------------------------------------------------------------------*/
bool GetComponentPlacement(const void * pSimpleGlyf, int nCompId,
                                    bool fOffset, int & a, int & b)
{
    using namespace Sfnt;

    if (GlyfContourCount(pSimpleGlyf) >= 0)
        return false;

    const Sfnt::SimpleGlyph * pGlyph = reinterpret_cast<const Sfnt::SimpleGlyph *>(pSimpleGlyf);
    // for a composite glyph, the special data begins here
    const uint8 * pbGlyph = reinterpret_cast<const uint8 *>(&pGlyph->end_pts_of_contours[0]);

    uint16 GlyphFlags;
    do
    {
        GlyphFlags = be::swap(*((uint16 *)pbGlyph));
        pbGlyph += sizeof(uint16);
        if (be::swap(*((uint16 *)pbGlyph)) == nCompId)
        {
            pbGlyph += sizeof(uint16); // skip over glyph id of component
            fOffset = (GlyphFlags & CompoundGlyph::ArgsAreXYValues) == CompoundGlyph::ArgsAreXYValues;

            if (GlyphFlags & CompoundGlyph::Arg1Arg2Words )
            {
                a = be::swap(*(int16 *)pbGlyph);
                pbGlyph += sizeof(int16);
                b = be::swap(*(int16 *)pbGlyph);
                pbGlyph += sizeof(int16);
            }
            else
            { // args are signed bytes
                a = *pbGlyph++;
                b = *pbGlyph++;
            }
            return true;
        }
        pbGlyph += sizeof(uint16); // skip over glyph id of component
        int nOffset = 0;
        nOffset += GlyphFlags & CompoundGlyph::Arg1Arg2Words  ? 4 : 2;
        nOffset += GlyphFlags & CompoundGlyph::HaveScale ? 2 : 0;
        nOffset += GlyphFlags & CompoundGlyph::HaveXAndYScale  ? 4 : 0;
        nOffset += GlyphFlags & CompoundGlyph::HaveTwoByTwo  ? 8 :  0;
        pbGlyph += nOffset;
    } while (GlyphFlags & CompoundGlyph::MoreComponents);

    // didn't find requested component
    fOffset = true;
    a = 0;
    b = 0;
    return false;
}

/*----------------------------------------------------------------------------------------------
    Return info on how a component glyph is to be transformed
    pSimpleGlyph - assumed to point to a composite glyph
    nCompId - glyph id for component of interest
    flt11, flt11, flt11, flt11 - a 2x2 matrix giving the transform
    bTransOffset - whether to transform the offset from above method
        The spec is unclear about the meaning of this flag
        Currently - initialize to true for MS rasterizer and false for Mac rasterizer, then
            on return it will indicate whether transform should apply to offset (MSDN CD 10/99)
    Return true if successful, false otherwise
        False could indicate a non-composite glyph or that component wasn't found
----------------------------------------------------------------------------------------------*/
bool GetComponentTransform(const void * pSimpleGlyf, int nCompId,
                                    float & flt11, float & flt12, float & flt21, float & flt22,
                                    bool & fTransOffset)
{
    using namespace Sfnt;

    if (GlyfContourCount(pSimpleGlyf) >= 0)
        return false;

    const Sfnt::SimpleGlyph * pGlyph = reinterpret_cast<const Sfnt::SimpleGlyph *>(pSimpleGlyf);
    // for a composite glyph, the special data begins here
    const uint8 * pbGlyph = reinterpret_cast<const uint8 *>(&pGlyph->end_pts_of_contours[0]);

    uint16 GlyphFlags;
    do
    {
        GlyphFlags = be::swap(*((uint16 *)pbGlyph));
        pbGlyph += sizeof(uint16);
        if (be::swap(*((uint16 *)pbGlyph)) == nCompId)
        {
            pbGlyph += sizeof(uint16); // skip over glyph id of component
            pbGlyph += GlyphFlags & CompoundGlyph::Arg1Arg2Words  ? 4 : 2; // skip over placement data

            if (fTransOffset) // MS rasterizer
                fTransOffset = !(GlyphFlags & CompoundGlyph::UnscaledOffset);
            else // Apple rasterizer
                fTransOffset = (GlyphFlags & CompoundGlyph::ScaledOffset) != 0;

            if (GlyphFlags & CompoundGlyph::HaveScale)
            {
                flt11 = fixed_to_float<14>(be::swap(*(uint16 *)pbGlyph));
                pbGlyph += sizeof(uint16);
                flt12 = 0;
                flt21 = 0;
                flt22 = flt11;
            }
            else if (GlyphFlags & CompoundGlyph::HaveXAndYScale)
            {
                flt11 = fixed_to_float<14>(be::swap(*(uint16 *)pbGlyph));
                pbGlyph += sizeof(uint16);
                flt12 = 0;
                flt21 = 0;
                flt22 = fixed_to_float<14>(be::swap(*(uint16 *)pbGlyph));
                pbGlyph += sizeof(uint16);
            }
            else if (GlyphFlags & CompoundGlyph::HaveTwoByTwo)
            {
                flt11 = fixed_to_float<14>(be::swap(*(uint16 *)pbGlyph));
                pbGlyph += sizeof(uint16);
                flt12 = fixed_to_float<14>(be::swap(*(uint16 *)pbGlyph));
                pbGlyph += sizeof(uint16);
                flt21 = fixed_to_float<14>(be::swap(*(uint16 *)pbGlyph));
                pbGlyph += sizeof(uint16);
                flt22 = fixed_to_float<14>(be::swap(*(uint16 *)pbGlyph));
                pbGlyph += sizeof(uint16);
            }
            else
            { // identity transform
                flt11 = 1.0;
                flt12 = 0.0;
                flt21 = 0.0;
                flt22 = 1.0;
            }
            return true;
        }
        pbGlyph += sizeof(uint16); // skip over glyph id of component
        int nOffset = 0;
        nOffset += GlyphFlags & CompoundGlyph::Arg1Arg2Words  ? 4 : 2;
        nOffset += GlyphFlags & CompoundGlyph::HaveScale ? 2 : 0;
        nOffset += GlyphFlags & CompoundGlyph::HaveXAndYScale  ? 4 : 0;
        nOffset += GlyphFlags & CompoundGlyph::HaveTwoByTwo  ? 8 :  0;
        pbGlyph += nOffset;
    } while (GlyphFlags & CompoundGlyph::MoreComponents);

    // didn't find requested component
    fTransOffset = false;
    flt11 = 1;
    flt12 = 0;
    flt21 = 0;
    flt22 = 1;
    return false;
}
#endif

/*----------------------------------------------------------------------------------------------
    Return a pointer into the glyf table based on the given tables and Glyph ID
    Since this method doesn't check for spaces, it is good to call IsSpace before using it.
    Return NULL on error.
----------------------------------------------------------------------------------------------*/
void * GlyfLookup(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
                           size_t lGlyfSize, size_t lLocaSize, const void * pHead)
{
    // test for valid glyph id
    // CheckTable verifies the index_to_loc_format is valid

    const Sfnt::FontHeader * pTable
        = reinterpret_cast<const Sfnt::FontHeader *>(pHead);

    if (be::swap(pTable->index_to_loc_format) == Sfnt::FontHeader::ShortIndexLocFormat)
    { // loca entries are two bytes (and have been divided by two)
        if (nGlyphId >= (lLocaSize >> 1) - 1) // don't allow nGlyphId to access sentinel
        {
//          throw std::out_of_range("glyph id out of range for font");
            return NULL;
        }
    }
    if (be::swap(pTable->index_to_loc_format) == Sfnt::FontHeader::LongIndexLocFormat)
    { // loca entries are four bytes
        if (nGlyphId >= (lLocaSize >> 2) - 1)
        {
//          throw std::out_of_range("glyph id out of range for font");
            return NULL;
        }
    }

    size_t lGlyfOffset = LocaLookup(nGlyphId, pLoca, lLocaSize, pHead);
    void * pSimpleGlyf = GlyfLookup(pGlyf, lGlyfOffset, lGlyfSize); // invalid loca offset returns null
    return pSimpleGlyf;
}

#ifdef ALL_TTFUTILS
/*----------------------------------------------------------------------------------------------
    Determine if a particular Glyph ID has any data in the glyf table. If it is white space,
    there will be no glyf data, though there will be metric data in hmtx, etc.
----------------------------------------------------------------------------------------------*/
bool IsSpace(gid16 nGlyphId, const void * pLoca, size_t lLocaSize, const void * pHead)
{
    size_t lGlyfOffset = LocaLookup(nGlyphId, pLoca, lLocaSize, pHead);

    // the +1 should always work because there is a sentinel value at the end of the loca table
    size_t lNextGlyfOffset = LocaLookup(nGlyphId + 1, pLoca, lLocaSize, pHead);

    return (lNextGlyfOffset - lGlyfOffset) == 0;
}

/*----------------------------------------------------------------------------------------------
    Determine if a particular Glyph ID is a multi-level composite.
----------------------------------------------------------------------------------------------*/
bool IsDeepComposite(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
                    size_t lGlyfSize, long lLocaSize, const void * pHead)
{
    if (IsSpace(nGlyphId, pLoca, lLocaSize, pHead)) {return false;}

    void * pSimpleGlyf = GlyfLookup(nGlyphId, pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
    if (pSimpleGlyf == NULL)
        return false; // no way to really indicate an error occured here

    if (GlyfContourCount(pSimpleGlyf) >= 0)
        return false;

    int rgnCompId[kMaxGlyphComponents]; // assumes only a limited number of glyph components
    size_t cCompIdTotal = kMaxGlyphComponents;
    size_t cCompId = 0;

    if (!GetComponentGlyphIds(pSimpleGlyf, rgnCompId, cCompIdTotal, cCompId))
        return false;

    for (size_t i = 0; i < cCompId; i++)
    {
        pSimpleGlyf = GlyfLookup(static_cast<gid16>(rgnCompId[i]),
                            pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
        if (pSimpleGlyf == NULL) {return false;}

        if (GlyfContourCount(pSimpleGlyf) < 0)
            return true;
    }

    return false;
}

/*----------------------------------------------------------------------------------------------
    Get the bounding box coordinates based on the given tables and Glyph ID
    Handles both simple and composite glyphs.
    Return true if successful, false otherwise. On false, all point values will be INT_MIN
        False may indicate a white space glyph
----------------------------------------------------------------------------------------------*/
bool GlyfBox(gid16  nGlyphId, const void * pGlyf, const void * pLoca,
        size_t lGlyfSize, size_t lLocaSize, const void * pHead, int & xMin, int & yMin, int & xMax, int & yMax)
{
    xMin = yMin = xMax = yMax = INT_MIN;

    if (IsSpace(nGlyphId, pLoca, lLocaSize, pHead)) {return false;}

    void * pSimpleGlyf = GlyfLookup(nGlyphId, pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
    if (pSimpleGlyf == NULL) {return false;}

    return GlyfBox(pSimpleGlyf, xMin, yMin, xMax, yMax);
}

/*----------------------------------------------------------------------------------------------
    Get the number of contours based on the given tables and Glyph ID
    Handles both simple and composite glyphs.
    Return true if successful, false otherwise. On false, cnContours will be INT_MIN
        False may indicate a white space glyph or a multi-level composite glyph.
----------------------------------------------------------------------------------------------*/
bool GlyfContourCount(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
    size_t lGlyfSize, size_t lLocaSize, const void * pHead, size_t & cnContours)
{
    cnContours = static_cast<size_t>(INT_MIN);

    if (IsSpace(nGlyphId, pLoca, lLocaSize, pHead)) {return false;}

    void * pSimpleGlyf = GlyfLookup(nGlyphId, pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
    if (pSimpleGlyf == NULL) {return false;}

    int cRtnContours = GlyfContourCount(pSimpleGlyf);
    if (cRtnContours >= 0)
    {
        cnContours = size_t(cRtnContours);
        return true;
    }

    //handle composite glyphs

    int rgnCompId[kMaxGlyphComponents]; // assumes no glyph will be made of more than 8 components
    size_t cCompIdTotal = kMaxGlyphComponents;
    size_t cCompId = 0;

    if (!GetComponentGlyphIds(pSimpleGlyf, rgnCompId, cCompIdTotal, cCompId))
        return false;

    cRtnContours = 0;
    int cTmp = 0;
    for (size_t i = 0; i < cCompId; i++)
    {
        if (IsSpace(static_cast<gid16>(rgnCompId[i]), pLoca, lLocaSize, pHead)) {return false;}
        pSimpleGlyf = GlyfLookup(static_cast<gid16>(rgnCompId[i]),
                                 pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
        if (pSimpleGlyf == 0) {return false;}
        // return false on multi-level composite
        if ((cTmp = GlyfContourCount(pSimpleGlyf)) < 0)
            return false;
        cRtnContours += cTmp;
    }

    cnContours = size_t(cRtnContours);
    return true;
}

/*----------------------------------------------------------------------------------------------
    Get the point numbers for the end points of the glyph contours based on the given tables
    and Glyph ID
    Handles both simple and composite glyphs.
    cnPoints - count of contours from GlyfContourCount (same as number of end points)
    prgnContourEndPoints - should point to a buffer large enough to hold cnPoints integers
    Return true if successful, false otherwise. On false, all end points are INT_MIN
        False may indicate a white space glyph or a multi-level composite glyph.
----------------------------------------------------------------------------------------------*/
bool GlyfContourEndPoints(gid16 nGlyphId, const void * pGlyf, const void * pLoca,
    size_t lGlyfSize, size_t lLocaSize, const void * pHead,
    int * prgnContourEndPoint, size_t cnPoints)
{
        memset(prgnContourEndPoint, 0xFF, cnPoints * sizeof(int));
    // std::fill_n(prgnContourEndPoint, cnPoints, INT_MIN);

    if (IsSpace(nGlyphId, pLoca, lLocaSize, pHead)) {return false;}

    void * pSimpleGlyf = GlyfLookup(nGlyphId, pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
    if (pSimpleGlyf == NULL) {return false;}

    int cContours = GlyfContourCount(pSimpleGlyf);
    int cActualPts = 0;
    if (cContours > 0)
        return GlyfContourEndPoints(pSimpleGlyf, prgnContourEndPoint, cnPoints, cActualPts);

    // handle composite glyphs

    int rgnCompId[kMaxGlyphComponents]; // assumes no glyph will be made of more than 8 components
    size_t cCompIdTotal = kMaxGlyphComponents;
    size_t cCompId = 0;

    if (!GetComponentGlyphIds(pSimpleGlyf, rgnCompId, cCompIdTotal, cCompId))
        return false;

    int * prgnCurrentEndPoint = prgnContourEndPoint;
    int cCurrentPoints = cnPoints;
    int nPrevPt = 0;
    for (size_t i = 0; i < cCompId; i++)
    {
        if (IsSpace(static_cast<gid16>(rgnCompId[i]), pLoca, lLocaSize, pHead)) {return false;}
        pSimpleGlyf = GlyfLookup(static_cast<gid16>(rgnCompId[i]), pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
        if (pSimpleGlyf == NULL) {return false;}
        // returns false on multi-level composite
        if (!GlyfContourEndPoints(pSimpleGlyf, prgnCurrentEndPoint, cCurrentPoints, cActualPts))
            return false;
        // points in composite are numbered sequentially as components are added
        //  must adjust end point numbers for new point numbers
        for (int j = 0; j < cActualPts; j++)
            prgnCurrentEndPoint[j] += nPrevPt;
        nPrevPt = prgnCurrentEndPoint[cActualPts - 1] + 1;

        prgnCurrentEndPoint += cActualPts;
        cCurrentPoints -= cActualPts;
    }

    return true;
}

/*----------------------------------------------------------------------------------------------
    Get the points for a glyph based on the given tables and Glyph ID
    Handles both simple and composite glyphs.
    cnPoints - count of points from largest end point obtained from GlyfContourEndPoints
    prgnX & prgnY - should point to buffers large enough to hold cnPoints integers
        The ranges are parallel so that coordinates for point(n) are found at offset n in
        both ranges. These points are in absolute coordinates.
    prgfOnCurve - should point to a buffer a large enough to hold cnPoints bytes (bool)
        This range is parallel to the prgnX & prgnY
    Return true if successful, false otherwise. On false, all points may be INT_MIN
        False may indicate a white space glyph, a multi-level composite, or a corrupt font
        It's not clear from the TTF spec when the transforms should be applied. Should the
        transform be done before or after attachment point calcs? (current code - before)
        Should the transform be applied to other offsets? (currently - no; however commented
        out code is in place so that if CompoundGlyph::UnscaledOffset on the MS rasterizer is
        clear (typical) then yes, and if CompoundGlyph::ScaledOffset on the Apple rasterizer is
        clear (typical?) then no). See GetComponentTransform.
        It's also unclear where point numbering with attachment poinst starts
        (currently - first point number is relative to whole glyph, second point number is
        relative to current glyph).
----------------------------------------------------------------------------------------------*/
bool GlyfPoints(gid16 nGlyphId, const void * pGlyf,
        const void * pLoca, size_t lGlyfSize, size_t lLocaSize, const void * pHead,
        const int * /*prgnContourEndPoint*/, size_t /*cnEndPoints*/,
        int * prgnX, int * prgnY, bool * prgfOnCurve, size_t cnPoints)
{
        memset(prgnX, 0x7F, cnPoints * sizeof(int));
        memset(prgnY, 0x7F, cnPoints * sizeof(int));

    if (IsSpace(nGlyphId, pLoca, lLocaSize, pHead))
        return false;

    void * pSimpleGlyf = GlyfLookup(nGlyphId, pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
    if (pSimpleGlyf == NULL)
        return false;

    int cContours = GlyfContourCount(pSimpleGlyf);
    int cActualPts;
    if (cContours > 0)
    {
        if (!GlyfPoints(pSimpleGlyf, prgnX, prgnY, (char *)prgfOnCurve, cnPoints, cActualPts))
            return false;
        CalcAbsolutePoints(prgnX, prgnY, cnPoints);
        SimplifyFlags((char *)prgfOnCurve, cnPoints);
        return true;
    }

    // handle composite glyphs
    int rgnCompId[kMaxGlyphComponents]; // assumes no glyph will be made of more than 8 components
    size_t cCompIdTotal = kMaxGlyphComponents;
    size_t cCompId = 0;

    // this will fail if there are more components than there is room for
    if (!GetComponentGlyphIds(pSimpleGlyf, rgnCompId, cCompIdTotal, cCompId))
        return false;

    int * prgnCurrentX = prgnX;
    int * prgnCurrentY = prgnY;
    char * prgbCurrentFlag = (char *)prgfOnCurve; // converting bool to char should be safe
    int cCurrentPoints = cnPoints;
    bool fOffset = true, fTransOff = true;
    int a, b;
    float flt11, flt12, flt21, flt22;
    // int * prgnPrevX = prgnX; // in case first att pt number relative to preceding glyph
    // int * prgnPrevY = prgnY;
    for (size_t i = 0; i < cCompId; i++)
    {
        if (IsSpace(static_cast<gid16>(rgnCompId[i]), pLoca, lLocaSize, pHead)) {return false;}
        void * pCompGlyf = GlyfLookup(static_cast<gid16>(rgnCompId[i]), pGlyf, pLoca, lGlyfSize, lLocaSize, pHead);
        if (pCompGlyf == NULL) {return false;}
        // returns false on multi-level composite
        if (!GlyfPoints(pCompGlyf, prgnCurrentX, prgnCurrentY, prgbCurrentFlag,
            cCurrentPoints, cActualPts))
            return false;
        if (!GetComponentPlacement(pSimpleGlyf, rgnCompId[i], fOffset, a, b))
            return false;
        if (!GetComponentTransform(pSimpleGlyf, rgnCompId[i],
            flt11, flt12, flt21, flt22, fTransOff))
            return false;
        bool fIdTrans = flt11 == 1.0 && flt12 == 0.0 && flt21 == 0.0 && flt22 == 1.0;

        // convert points to absolute coordinates
        // do before transform and attachment point placement are applied
        CalcAbsolutePoints(prgnCurrentX, prgnCurrentY, cActualPts);

        // apply transform - see main method note above
        // do before attachment point calcs
        if (!fIdTrans)
            for (int j = 0; j < cActualPts; j++)
            {
                int x = prgnCurrentX[j]; // store before transform applied
                int y = prgnCurrentY[j];
                prgnCurrentX[j] = (int)(x * flt11 + y * flt12);
                prgnCurrentY[j] = (int)(x * flt21 + y * flt22);
            }

        // apply placement - see main method note above
        int nXOff, nYOff;
        if (fOffset) // explicit x & y offsets
        {
            /* ignore fTransOff for now
            if (fTransOff && !fIdTrans)
            {   // transform x & y offsets
                nXOff = (int)(a * flt11 + b * flt12);
                nYOff = (int)(a * flt21 + b * flt22);
            }
            else */
            { // don't transform offset
                nXOff = a;
                nYOff = b;
            }
        }
        else  // attachment points
        {   // in case first point is relative to preceding glyph and second relative to current
            // nXOff = prgnPrevX[a] - prgnCurrentX[b];
            // nYOff = prgnPrevY[a] - prgnCurrentY[b];
            // first point number relative to whole composite, second relative to current glyph
            nXOff = prgnX[a] - prgnCurrentX[b];
            nYOff = prgnY[a] - prgnCurrentY[b];
        }
        for (int j = 0; j < cActualPts; j++)
        {
            prgnCurrentX[j] += nXOff;
            prgnCurrentY[j] += nYOff;
        }

        // prgnPrevX = prgnCurrentX;
        // prgnPrevY = prgnCurrentY;
        prgnCurrentX += cActualPts;
        prgnCurrentY += cActualPts;
        prgbCurrentFlag += cActualPts;
        cCurrentPoints -= cActualPts;
    }

    SimplifyFlags((char *)prgfOnCurve, cnPoints);

    return true;
}

/*----------------------------------------------------------------------------------------------
    Simplify the meaning of flags to just indicate whether point is on-curve or off-curve.
---------------------------------------------------------------------------------------------*/
bool SimplifyFlags(char * prgbFlags, int cnPoints)
{
    for (int i = 0; i < cnPoints; i++)
        prgbFlags[i] = static_cast<char>(prgbFlags[i] & Sfnt::SimpleGlyph::OnCurve);
    return true;
}

/*----------------------------------------------------------------------------------------------
    Convert relative point coordinates to absolute coordinates
    Points are stored in the font such that they are offsets from one another except for the
        first point of a glyph.
---------------------------------------------------------------------------------------------*/
bool CalcAbsolutePoints(int * prgnX, int * prgnY, int cnPoints)
{
    int nX = prgnX[0];
    int nY = prgnY[0];
    for (int i = 1; i < cnPoints; i++)
    {
        prgnX[i] += nX;
        nX = prgnX[i];
        prgnY[i] += nY;
        nY = prgnY[i];
    }

    return true;
}
#endif

/*----------------------------------------------------------------------------------------------
    Return the length of the 'name' table in bytes.
    Currently used.
---------------------------------------------------------------------------------------------*/
#if 0
size_t NameTableLength(const byte * pTable)
{
    byte * pb = (const_cast<byte *>(pTable)) + 2; // skip format
    size_t cRecords = *pb++ << 8; cRecords += *pb++;
    int dbStringOffset0 = (*pb++) << 8; dbStringOffset0 += *pb++;
    int dbMaxStringOffset = 0;
    for (size_t irec = 0; irec < cRecords; irec++)
    {
        int nPlatform = (*pb++) << 8; nPlatform += *pb++;
        int nEncoding = (*pb++) << 8; nEncoding += *pb++;
        int nLanguage = (*pb++) << 8; nLanguage += *pb++;
        int nName = (*pb++) << 8; nName += *pb++;
        int cbStringLen = (*pb++) << 8; cbStringLen += *pb++;
        int dbStringOffset = (*pb++) << 8; dbStringOffset += *pb++;
        if (dbMaxStringOffset < dbStringOffset + cbStringLen)
            dbMaxStringOffset = dbStringOffset + cbStringLen;
    }
    return dbStringOffset0 + dbMaxStringOffset;
}
#endif

} // end of namespace TtfUtil
} // end of namespace graphite
