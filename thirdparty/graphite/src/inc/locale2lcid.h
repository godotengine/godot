// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#pragma once
#include <cstring>
#include <cassert>

#include "inc/Main.h"


namespace graphite2 {

struct IsoLangEntry
{
    unsigned short mnLang;
    char maLangStr[4];
    char maCountry[3];
};

// Windows Language ID, Locale ISO-639 language, country code as used in
// naming table of OpenType fonts
const IsoLangEntry LANG_ENTRIES[] = {
    { 0x0401, "ar","SA" }, // Arabic Saudi Arabia
    { 0x0402, "bg","BG" }, // Bulgarian Bulgaria
    { 0x0403, "ca","ES" }, // Catalan Catalan
    { 0x0404, "zh","TW" }, // Chinese Taiwan
    { 0x0405, "cs","CZ" }, // Czech Czech Republic
    { 0x0406, "da","DK" }, // Danish Denmark
    { 0x0407, "de","DE" }, // German Germany
    { 0x0408, "el","GR" }, // Greek Greece
    { 0x0409, "en","US" }, // English United States
    { 0x040A, "es","ES" }, // Spanish (Traditional Sort) Spain
    { 0x040B, "fi","FI" }, // Finnish Finland
    { 0x040C, "fr","FR" }, // French France
    { 0x040D, "he","IL" }, // Hebrew Israel
    { 0x040E, "hu","HU" }, // Hungarian Hungary
    { 0x040F, "is","IS" }, // Icelandic Iceland
    { 0x0410, "it","IT" }, // Italian Italy
    { 0x0411, "jp","JP" }, // Japanese Japan
    { 0x0412, "ko","KR" }, // Korean Korea
    { 0x0413, "nl","NL" }, // Dutch Netherlands
    { 0x0414, "no","NO" }, // Norwegian (Bokmal) Norway
    { 0x0415, "pl","PL" }, // Polish Poland
    { 0x0416, "pt","BR" }, // Portuguese Brazil
    { 0x0417, "rm","CH" }, // Romansh Switzerland
    { 0x0418, "ro","RO" }, // Romanian Romania
    { 0x0419, "ru","RU" }, // Russian Russia
    { 0x041A, "hr","HR" }, // Croatian Croatia
    { 0x041B, "sk","SK" }, // Slovak Slovakia
    { 0x041C, "sq","AL" }, // Albanian Albania
    { 0x041D, "sv","SE" }, // Swedish Sweden
    { 0x041E, "th","TH" }, // Thai Thailand
    { 0x041F, "tr","TR" }, // Turkish Turkey
    { 0x0420, "ur","PK" }, // Urdu Islamic Republic of Pakistan
    { 0x0421, "id","ID" }, // Indonesian Indonesia
    { 0x0422, "uk","UA" }, // Ukrainian Ukraine
    { 0x0423, "be","BY" }, // Belarusian Belarus
    { 0x0424, "sl","SI" }, // Slovenian Slovenia
    { 0x0425, "et","EE" }, // Estonian Estonia
    { 0x0426, "lv","LV" }, // Latvian Latvia
    { 0x0427, "lt","LT" }, // Lithuanian Lithuania
    { 0x0428, "tg","TJ" }, // Tajik (Cyrillic) Tajikistan
    { 0x042A, "vi","VN" }, // Vietnamese Vietnam
    { 0x042B, "hy","AM" }, // Armenian Armenia
    { 0x042C, "az","AZ" }, // Azeri (Latin) Azerbaijan
    { 0x042D, "eu","" }, // Basque Basque
    { 0x042E, "hsb","DE" }, // Upper Sorbian Germany
    { 0x042F, "mk","MK" }, // Macedonian (FYROM) Former Yugoslav Republic of Macedonia
    { 0x0432, "tn","ZA" }, // Setswana South Africa
    { 0x0434, "xh","ZA" }, // isiXhosa South Africa
    { 0x0435, "zu","ZA" }, // isiZulu South Africa
    { 0x0436, "af","ZA" }, // Afrikaans South Africa
    { 0x0437, "ka","GE" }, // Georgian Georgia
    { 0x0438, "fo","FO" }, // Faroese Faroe Islands
    { 0x0439, "hi","IN" }, // Hindi India
    { 0x043A, "mt","MT" }, // Maltese Malta
    { 0x043B, "se","NO" }, // Sami (Northern) Norway
    { 0x043E, "ms","MY" }, // Malay Malaysia
    { 0x043F, "kk","KZ" }, // Kazakh Kazakhstan
    { 0x0440, "ky","KG" }, // Kyrgyz Kyrgyzstan
    { 0x0441, "sw","KE" }, // Kiswahili Kenya
    { 0x0442, "tk","TM" }, // Turkmen Turkmenistan
    { 0x0443, "uz","UZ" }, // Uzbek (Latin) Uzbekistan
    { 0x0444, "tt","RU" }, // Tatar Russia
    { 0x0445, "bn","IN" }, // Bengali India
    { 0x0446, "pa","IN" }, // Punjabi India
    { 0x0447, "gu","IN" }, // Gujarati India
    { 0x0448, "or","IN" }, // Oriya India
    { 0x0448, "wo","SN" }, // Wolof Senegal
    { 0x0449, "ta","IN" }, // Tamil India
    { 0x044A, "te","IN" }, // Telugu India
    { 0x044B, "kn","IN" }, // Kannada India
    { 0x044C, "ml","IN" }, // Malayalam India
    { 0x044D, "as","IN" }, // Assamese India
    { 0x044E, "mr","IN" }, // Marathi India
    { 0x044F, "sa","IN" }, // Sanskrit India
    { 0x0450, "mn","MN" }, // Mongolian (Cyrillic) Mongolia
    { 0x0451, "bo","CN" }, // Tibetan PRC
    { 0x0452, "cy","GB" }, // Welsh United Kingdom
    { 0x0453, "km","KH" }, // Khmer Cambodia
    { 0x0454, "lo","LA" }, // Lao Lao P.D.R.
    { 0x0455, "my","MM" }, // Burmese Myanmar - not listed in Microsoft docs anymore
    { 0x0456, "gl","ES" }, // Galician Galician
    { 0x0457, "kok","IN" }, // Konkani India
    { 0x045A, "syr","TR" }, // Syriac Syria
    { 0x045B, "si","LK" }, // Sinhala Sri Lanka
    { 0x045D, "iu","CA" }, // Inuktitut Canada
    { 0x045E, "am","ET" }, // Amharic Ethiopia
    { 0x0461, "ne","NP" }, // Nepali Nepal
    { 0x0462, "fy","NL" }, // Frisian Netherlands
    { 0x0463, "ps","AF" }, // Pashto Afghanistan
    { 0x0464, "fil","PH" }, // Filipino Philippines
    { 0x0465, "dv","MV" }, // Divehi Maldives
    { 0x0468, "ha","NG" }, // Hausa (Latin) Nigeria
    { 0x046A, "yo","NG" }, // Yoruba Nigeria
    { 0x046B, "qu","BO" }, // Quechua Bolivia
    { 0x046C, "st","ZA" }, // Sesotho sa Leboa South Africa
    { 0x046D, "ba","RU" }, // Bashkir Russia
    { 0x046E, "lb","LU" }, // Luxembourgish Luxembourg
    { 0x046F, "kl","GL" }, // Greenlandic Greenland
    { 0x0470, "ig","NG" }, // Igbo Nigeria
    { 0x0478, "ii","CN" }, // Yi PRC
    { 0x047A, "arn","CL" }, // Mapudungun Chile
    { 0x047C, "moh","CA" }, // Mohawk Mohawk
    { 0x047E, "br","FR" }, // Breton France
    { 0x0480, "ug","CN" }, // Uighur PRC
    { 0x0481, "mi","NZ" }, // Maori New Zealand
    { 0x0482, "oc","FR" }, // Occitan France
    { 0x0483, "co","FR" }, // Corsican France
    { 0x0484, "gsw","FR" }, // Alsatian France
    { 0x0485, "sah","RU" }, // Yakut Russia
    { 0x0486, "qut","GT" }, // K'iche Guatemala
    { 0x0487, "rw","RW" }, // Kinyarwanda Rwanda
    { 0x048C, "gbz","AF" }, // Dari Afghanistan
    { 0x0801, "ar","IQ" }, // Arabic Iraq
    { 0x0804, "zn","CH" }, // Chinese People's Republic of China
    { 0x0807, "de","CH" }, // German Switzerland
    { 0x0809, "en","GB" }, // English United Kingdom
    { 0x080A, "es","MX" }, // Spanish Mexico
    { 0x080C, "fr","BE" }, // French Belgium
    { 0x0810, "it","CH" }, // Italian Switzerland
    { 0x0813, "nl","BE" }, // Dutch Belgium
    { 0x0814, "nn","NO" }, // Norwegian (Nynorsk) Norway
    { 0x0816, "pt","PT" }, // Portuguese Portugal
    { 0x081A, "sh","RS" }, // Serbian (Latin) Serbia
    { 0x081D, "sv","FI" }, // Sweden Finland
    { 0x082C, "az","AZ" }, // Azeri (Cyrillic) Azerbaijan
    { 0x082E, "dsb","DE" }, // Lower Sorbian Germany
    { 0x083B, "se","SE" }, // Sami (Northern) Sweden
    { 0x083C, "ga","IE" }, // Irish Ireland
    { 0x083E, "ms","BN" }, // Malay Brunei Darussalam
    { 0x0843, "uz","UZ" }, // Uzbek (Cyrillic) Uzbekistan
    { 0x0845, "bn","BD" }, // Bengali Bangladesh
    { 0x0850, "mn","MN" }, // Mongolian (Traditional) People's Republic of China
    { 0x085D, "iu","CA" }, // Inuktitut (Latin) Canada
    { 0x085F, "ber","DZ" }, // Tamazight (Latin) Algeria
    { 0x086B, "es","EC" }, // Quechua Ecuador
    { 0x0C01, "ar","EG" }, // Arabic Egypt
    { 0x0C04, "zh","HK" }, // Chinese Hong Kong S.A.R.
    { 0x0C07, "de","AT" }, // German Austria
    { 0x0C09, "en","AU" }, // English Australia
    { 0x0C0A, "es","ES" }, // Spanish (Modern Sort) Spain
    { 0x0C0C, "fr","CA" }, // French Canada
    { 0x0C1A, "sr","CS" }, // Serbian (Cyrillic) Serbia
    { 0x0C3B, "se","FI" }, // Sami (Northern) Finland
    { 0x0C6B, "qu","PE" }, // Quechua Peru
    { 0x1001, "ar","LY" }, // Arabic Libya
    { 0x1004, "zh","SG" }, // Chinese Singapore
    { 0x1007, "de","LU" }, // German Luxembourg
    { 0x1009, "en","CA" }, // English Canada
    { 0x100A, "es","GT" }, // Spanish Guatemala
    { 0x100C, "fr","CH" }, // French Switzerland
    { 0x101A, "hr","BA" }, // Croatian (Latin) Bosnia and Herzegovina
    { 0x103B, "smj","NO" }, // Sami (Lule) Norway
    { 0x1401, "ar","DZ" }, // Arabic Algeria
    { 0x1404, "zh","MO" }, // Chinese Macao S.A.R.
    { 0x1407, "de","LI" }, // German Liechtenstein
    { 0x1409, "en","NZ" }, // English New Zealand
    { 0x140A, "es","CR" }, // Spanish Costa Rica
    { 0x140C, "fr","LU" }, // French Luxembourg
    { 0x141A, "bs","BA" }, // Bosnian (Latin) Bosnia and Herzegovina
    { 0x143B, "smj","SE" }, // Sami (Lule) Sweden
    { 0x1801, "ar","MA" }, // Arabic Morocco
    { 0x1809, "en","IE" }, // English Ireland
    { 0x180A, "es","PA" }, // Spanish Panama
    { 0x180C, "fr","MC" }, // French Principality of Monoco
    { 0x181A, "sh","BA" }, // Serbian (Latin) Bosnia and Herzegovina
    { 0x183B, "sma","NO" }, // Sami (Southern) Norway
    { 0x1C01, "ar","TN" }, // Arabic Tunisia
    { 0x1C09, "en","ZA" }, // English South Africa
    { 0x1C0A, "es","DO" }, // Spanish Dominican Republic
    { 0x1C1A, "sr","BA" }, // Serbian (Cyrillic) Bosnia and Herzegovina
    { 0x1C3B, "sma","SE" }, // Sami (Southern) Sweden
    { 0x2001, "ar","OM" }, // Arabic Oman
    { 0x2009, "en","JM" }, // English Jamaica
    { 0x200A, "es","VE" }, // Spanish Venezuela
    { 0x201A, "bs","BA" }, // Bosnian (Cyrillic) Bosnia and Herzegovina
    { 0x203B, "sms","FI" }, // Sami (Skolt) Finland
    { 0x2401, "ar","YE" }, // Arabic Yemen
    { 0x2409, "en","BS" }, // English Caribbean
    { 0x240A, "es","CO" }, // Spanish Colombia
    { 0x243B, "smn","FI" }, // Sami (Inari) Finland
    { 0x2801, "ar","SY" }, // Arabic Syria
    { 0x2809, "en","BZ" }, // English Belize
    { 0x280A, "es","PE" }, // Spanish Peru
    { 0x2C01, "ar","JO" }, // Arabic Jordan
    { 0x2C09, "en","TT" }, // English Trinidad and Tobago
    { 0x2C0A, "es","AR" }, // Spanish Argentina
    { 0x3001, "ar","LB" }, // Arabic Lebanon
    { 0x3009, "en","ZW" }, // English Zimbabwe
    { 0x300A, "es","EC" }, // Spanish Ecuador
    { 0x3401, "ar","KW" }, // Arabic Kuwait
    { 0x3409, "en","PH" }, // English Republic of the Philippines
    { 0x340A, "es","CL" }, // Spanish Chile
    { 0x3801, "ar","AE" }, // Arabic U.A.E.
    { 0x380A, "es","UY" }, // Spanish Uruguay
    { 0x3C01, "ar","BH" }, // Arabic Bahrain
    { 0x3C0A, "es","PY" }, // Spanish Paraguay
    { 0x4001, "ar","QA" }, // Arabic Qatar
    { 0x4009, "en","IN" }, // English India
    { 0x400A, "es","BO" }, // Spanish Bolivia
    { 0x4409, "en","MY" }, // English Malaysia
    { 0x440A, "es","SV" }, // Spanish El Salvador
    { 0x4809, "en","SG" }, // English Singapore
    { 0x480A, "es","HN" }, // Spanish Honduras
    { 0x4C0A, "es","NI" }, // Spanish Nicaragua
    { 0x500A, "es","PR" }, // Spanish Puerto Rico
    { 0x540A, "es","US" } // Spanish United States
};

class Locale2Lang
{
    Locale2Lang(const Locale2Lang &);
    Locale2Lang & operator = (const Locale2Lang &);

public:
    Locale2Lang() : mSeedPosition(128)
    {
        memset((void*)mLangLookup, 0, sizeof(mLangLookup));
        // create a tri lookup on first 2 letters of language code
        static const int maxIndex = sizeof(LANG_ENTRIES)/sizeof(IsoLangEntry);
        for (int i = 0; i < maxIndex; i++)
        {
            size_t a = LANG_ENTRIES[i].maLangStr[0] - 'a';
            size_t b = LANG_ENTRIES[i].maLangStr[1] - 'a';
            if (mLangLookup[a][b])
            {
                const IsoLangEntry ** old = mLangLookup[a][b];
                int len = 1;
                while (old[len]) len++;
                len += 2;
                mLangLookup[a][b] = gralloc<const IsoLangEntry *>(len);
                if (!mLangLookup[a][b])
                {
                    mLangLookup[a][b] = old;
                    continue;
                }
                mLangLookup[a][b][--len] = NULL;
                mLangLookup[a][b][--len] = &LANG_ENTRIES[i];
                while (--len >= 0)
                {
                    assert(len >= 0);
                    mLangLookup[a][b][len] = old[len];
                }
                free(old);
            }
            else
            {
                mLangLookup[a][b] = gralloc<const IsoLangEntry *>(2);
                if (!mLangLookup[a][b]) continue;
                mLangLookup[a][b][1] = NULL;
                mLangLookup[a][b][0] = &LANG_ENTRIES[i];
            }
        }
        while (2 * mSeedPosition < maxIndex)
            mSeedPosition *= 2;
    };
    ~Locale2Lang()
    {
        for (int i = 0; i != 26; ++i)
            for (int j = 0; j != 26; ++j)
                free(mLangLookup[i][j]);
    }
    unsigned short getMsId(const char * locale) const
    {
        size_t length = strlen(locale);
        size_t langLength = length;
        const char * language = locale;
        const char * script = NULL;
        const char * region = NULL;
        size_t regionLength = 0;
        const char * dash = strchr(locale, '-');
        if (dash && (dash != locale))
        {
            langLength = (dash - locale);
            size_t nextPartLength = length - langLength - 1;
            if (nextPartLength >= 2)
            {
                script = ++dash;
                dash = strchr(dash, '-');
                if (dash)
                {
                    nextPartLength = (dash - script);
                    region = ++dash;
                }
                if (nextPartLength == 2 &&
                    (locale[langLength+1] > 0x40) && (locale[langLength+1] < 0x5B) &&
                    (locale[langLength+2] > 0x40) && (locale[langLength+2] < 0x5B))
                {
                    region = script;
                    regionLength = nextPartLength;
                    script = NULL;
                }
                else if (nextPartLength == 4)
                {
                    if (dash)
                    {
                        dash = strchr(dash, '-');
                        if (dash)
                        {
                            nextPartLength = (dash - region);
                        }
                        else
                        {
                            nextPartLength = langLength - (region - locale);
                        }
                        regionLength = nextPartLength;
                    }
                }
            }
        }
        size_t a = 'e' - 'a';
        size_t b = 'n' - 'a';
        unsigned short langId = 0;
        int i = 0;
        switch (langLength)
        {
            case 2:
            {
                a = language[0] - 'a';
                b = language[1] - 'a';
                if ((a < 26) && (b < 26) && mLangLookup[a][b])
                {
                    while (mLangLookup[a][b][i])
                    {
                        if (mLangLookup[a][b][i]->maLangStr[2] != '\0')
                        {
                            ++i;
                            continue;
                        }
                        if (region && (strncmp(mLangLookup[a][b][i]->maCountry, region, regionLength) == 0))
                        {
                            langId = mLangLookup[a][b][i]->mnLang;
                            break;
                        }
                        else if (langId == 0)
                        {
                            // possible fallback code
                            langId = mLangLookup[a][b][i]->mnLang;
                        }
                        ++i;
                    }
                }
            }
            break;
            case 3:
            {
                a = language[0] - 'a';
                b = language[1] - 'a';
                if (mLangLookup[a][b])
                {
                    while (mLangLookup[a][b][i])
                    {
                        if (mLangLookup[a][b][i]->maLangStr[2] != language[2])
                        {
                            ++i;
                            continue;
                        }
                        if (region && (strncmp(mLangLookup[a][b][i]->maCountry, region, regionLength) == 0))
                        {
                            langId = mLangLookup[a][b][i]->mnLang;
                            break;
                        }
                        else if (langId == 0)
                        {
                            // possible fallback code
                            langId = mLangLookup[a][b][i]->mnLang;
                        }
                        ++i;
                    }
                }
            }
            break;
            default:
                break;
        }
        if (langId == 0) langId = 0x409;
        return langId;
    }
    const IsoLangEntry * findEntryById(unsigned short langId) const
    {
        static const int maxIndex = sizeof(LANG_ENTRIES)/sizeof(IsoLangEntry);
        int window = mSeedPosition;
        int guess = mSeedPosition - 1;
        while (LANG_ENTRIES[guess].mnLang != langId)
        {
            window /= 2;
            if (window == 0) return NULL;
            guess += (LANG_ENTRIES[guess].mnLang > langId)? -window : window;
            while (guess >= maxIndex)
            {
                window /= 2;
                guess -= window;
                assert(window);
            }
        }
        return &LANG_ENTRIES[guess];
    }

    CLASS_NEW_DELETE;

private:
    const IsoLangEntry ** mLangLookup[26][26];
    int mSeedPosition;
};

} // namespace graphite2
