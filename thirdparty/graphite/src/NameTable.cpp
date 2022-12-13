// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include "inc/Main.h"
#include "inc/Endian.h"

#include "inc/NameTable.h"
#include "inc/UtfCodec.h"

using namespace graphite2;

NameTable::NameTable(const void* data, size_t length, uint16 platformId, uint16 encodingID)
 : m_platformId(0), m_encodingId(0), m_languageCount(0),
   m_platformOffset(0), m_platformLastRecord(0), m_nameDataLength(0),
   m_table(0), m_nameData(NULL)
{
    void *pdata = gralloc<byte>(length);
    if (!pdata) return;
    memcpy(pdata, data, length);
    m_table = reinterpret_cast<const TtfUtil::Sfnt::FontNames*>(pdata);

    if ((length > sizeof(TtfUtil::Sfnt::FontNames)) &&
        (length > sizeof(TtfUtil::Sfnt::FontNames) +
         sizeof(TtfUtil::Sfnt::NameRecord) * ( be::swap<uint16>(m_table->count) - 1)))
    {
        uint16 offset = be::swap<uint16>(m_table->string_offset);
        if (offset < length)
        {
            m_nameData = reinterpret_cast<const uint8*>(pdata) + offset;
            setPlatformEncoding(platformId, encodingID);
            m_nameDataLength = uint16(length - offset);
            return;
        }
    }
    free(const_cast<TtfUtil::Sfnt::FontNames*>(m_table));
    m_table = NULL;
}

uint16 NameTable::setPlatformEncoding(uint16 platformId, uint16 encodingID)
{
    if (!m_nameData) return 0;
    uint16 i = 0;
    uint16 count = be::swap<uint16>(m_table->count);
    for (; i < count; i++)
    {
        if (be::swap<uint16>(m_table->name_record[i].platform_id) == platformId &&
            be::swap<uint16>(m_table->name_record[i].platform_specific_id) == encodingID)
        {
            m_platformOffset = i;
            break;
        }
    }
    while ((++i < count) &&
           (be::swap<uint16>(m_table->name_record[i].platform_id) == platformId) &&
           (be::swap<uint16>(m_table->name_record[i].platform_specific_id) == encodingID))
    {
        m_platformLastRecord = i;
    }
    m_encodingId = encodingID;
    m_platformId = platformId;
    return 0;
}

void* NameTable::getName(uint16& languageId, uint16 nameId, gr_encform enc, uint32& length)
{
    uint16 anyLang = 0;
    uint16 enUSLang = 0;
    uint16 bestLang = 0;
    if (!m_table)
    {
        languageId = 0;
        length = 0;
        return NULL;
    }
    for (uint16 i = m_platformOffset; i <= m_platformLastRecord; i++)
    {
        if (be::swap<uint16>(m_table->name_record[i].name_id) == nameId)
        {
            uint16 langId = be::swap<uint16>(m_table->name_record[i].language_id);
            if (langId == languageId)
            {
                bestLang = i;
                break;
            }
            // MS language tags have the language in the lower byte, region in the higher
            else if ((langId & 0xFF) == (languageId & 0xFF))
            {
                bestLang = i;
            }
            else if (langId == 0x409)
            {
                enUSLang = i;
            }
            else
            {
                anyLang = i;
            }
        }
    }
    if (!bestLang)
    {
        if (enUSLang) bestLang = enUSLang;
        else
        {
            bestLang = anyLang;
            if (!anyLang)
            {
                languageId = 0;
                length = 0;
                return NULL;
            }
        }
    }
    const TtfUtil::Sfnt::NameRecord & nameRecord = m_table->name_record[bestLang];
    languageId = be::swap<uint16>(nameRecord.language_id);
    uint16 utf16Length = be::swap<uint16>(nameRecord.length);
    uint16 offset = be::swap<uint16>(nameRecord.offset);
    if(offset + utf16Length > m_nameDataLength)
    {
        languageId = 0;
        length = 0;
        return NULL;
    }
    utf16Length >>= 1; // in utf16 units
    utf16::codeunit_t * utf16Name = gralloc<utf16::codeunit_t>(utf16Length + 1);
    if (!utf16Name)
    {
        languageId = 0;
        length = 0;
        return NULL;
    }
    const uint8* pName = m_nameData + offset;
    for (size_t i = 0; i < utf16Length; i++)
    {
        utf16Name[i] = be::read<uint16>(pName);
    }
    utf16Name[utf16Length] = 0;
    if (!utf16::validate(utf16Name, utf16Name + utf16Length))
    {
        free(utf16Name);
        languageId = 0;
        length = 0;
        return NULL;
    }
    switch (enc)
    {
    case gr_utf8:
    {
        utf8::codeunit_t* uniBuffer = gralloc<utf8::codeunit_t>(3 * utf16Length + 1);
        if (!uniBuffer)
        {
            free(utf16Name);
            languageId = 0;
            length = 0;
            return NULL;
        }
        utf8::iterator d = uniBuffer;
        for (utf16::const_iterator s = utf16Name, e = utf16Name + utf16Length; s != e; ++s, ++d)
            *d = *s;
        length = uint32(d - uniBuffer);
        uniBuffer[length] = 0;
        free(utf16Name);
        return uniBuffer;
    }
    case gr_utf16:
        length = utf16Length;
        return utf16Name;
    case gr_utf32:
    {
        utf32::codeunit_t * uniBuffer = gralloc<utf32::codeunit_t>(utf16Length  + 1);
        if (!uniBuffer)
        {
            free(utf16Name);
            languageId = 0;
            length = 0;
            return NULL;
        }
        utf32::iterator d = uniBuffer;
        for (utf16::const_iterator s = utf16Name, e = utf16Name + utf16Length; s != e; ++s, ++d)
            *d = *s;
        length = uint32(d - uniBuffer);
        uniBuffer[length] = 0;
        free(utf16Name);
        return uniBuffer;
    }
    }
    free(utf16Name);
    languageId = 0;
    length = 0;
    return NULL;
}

uint16 NameTable::getLanguageId(const char * bcp47Locale)
{
    size_t localeLength = strlen(bcp47Locale);
    uint16 localeId = m_locale2Lang.getMsId(bcp47Locale);
    if (m_table && (be::swap<uint16>(m_table->format) == 1))
    {
        const uint8 * pLangEntries = reinterpret_cast<const uint8*>(m_table) +
            sizeof(TtfUtil::Sfnt::FontNames)
            + sizeof(TtfUtil::Sfnt::NameRecord) * ( be::swap<uint16>(m_table->count) - 1);
        uint16 numLangEntries = be::read<uint16>(pLangEntries);
        const TtfUtil::Sfnt::LangTagRecord * langTag =
            reinterpret_cast<const TtfUtil::Sfnt::LangTagRecord*>(pLangEntries);
        if (pLangEntries + numLangEntries * sizeof(TtfUtil::Sfnt::LangTagRecord) <= m_nameData)
        {
            for (uint16 i = 0; i < numLangEntries; i++)
            {
                uint16 offset = be::swap<uint16>(langTag[i].offset);
                uint16 length = be::swap<uint16>(langTag[i].length);
                if ((offset + length <= m_nameDataLength) && (length == 2 * localeLength))
                {
                    const uint8* pName = m_nameData + offset;
                    bool match = true;
                    for (size_t j = 0; j < localeLength; j++)
                    {
                        uint16 code = be::read<uint16>(pName);
                        if ((code > 0x7F) || (code != bcp47Locale[j]))
                        {
                            match = false;
                            break;
                        }
                    }
                    if (match)
                        return 0x8000 + i;
                }
            }
        }
    }
    return localeId;
}
