/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <ctype.h>
#include "tvgStr.h"
#include "tvgXmlParser.h"
#include "tvgSvgUtil.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

bool _unsupported(TVG_UNUSED const char* tagAttribute, TVG_UNUSED const char* tagValue)
{
#ifdef THORVG_LOG_ENABLED
    const auto attributesNum = 6;
    const struct
    {
        const char* tag;
        bool tagWildcard; //If true, it is assumed that a wildcard is used after the tag. (ex: tagName*)
        const char* value;
    } attributes[] = {
        {"id", false, nullptr},
        {"data-name", false, nullptr},
        {"overflow", false, "visible"},
        {"version", false, nullptr},
        {"xmlns", true, nullptr},
        {"xml:space", false, nullptr},
    };

    for (unsigned int i = 0; i < attributesNum; ++i) {
        if (tvg::equal(tagAttribute, attributes[i].tag)) {
            if (!attributes[i].value) return false;
            if (tvg::equal(tagValue, attributes[i].value)) return false;
        }
    }
    return true;
#endif
    return false;
}


static const char* _xmlFindWhiteSpace(const char* itr, const char* itrEnd)
{
    for (; itr < itrEnd; itr++) {
        if (isspace((unsigned char)*itr)) break;
    }
    return itr;
}

struct XmlEntity
{
    const char* name;
    uint8_t length;
    char decoded;
};

static constexpr XmlEntity xmlEntities[] = {
    {"&#10;", 5, '\n'},
    {"&#035;", 6, '#'},
    {"&#039;", 6, '\''},
    {"&quot;", 6, '"'},
    {"&nbsp;", 6, ' '},
    {"&apos;", 6, '\''},
    {"&amp;", 5, '&'},
    {"&lt;", 4, '<'},
    {"&gt;", 4, '>'},
};

static const XmlEntity* _xmlFindEntity(const char* itr, const char* itrEnd)
{
    for (auto& entity : xmlEntities) {
        if (itrEnd - itr >= entity.length && !memcmp(itr, entity.name, entity.length)) return &entity;
    }
    return nullptr;
}

static unsigned _xmlDecodeEntities(const char* itr, const char* itrEnd, char* decoded)
{
    auto dst = decoded;
    while (itr < itrEnd) {
        auto entity = (*itr == '&') ? _xmlFindEntity(itr, itrEnd) : nullptr;
        if (!entity) *dst++ = *itr++;
        else {
            *dst++ = entity->decoded;
            itr += entity->length;
        }
    }
    *dst = '\0';
    return dst - decoded;
}

static const char* _xmlFindStartTag(const char* itr, const char* itrEnd)
{
    return (const char*)memchr(itr, '<', itrEnd - itr);
}

static const char* _xmlFindEndTag(const char* itr, const char* itrEnd)
{
    bool insideQuote[2] = {false, false}; // 0: ", 1: '
    for (; itr < itrEnd; itr++) {
        if (*itr == '"' && !insideQuote[1]) insideQuote[0] = !insideQuote[0];
        if (*itr == '\'' && !insideQuote[0]) insideQuote[1] = !insideQuote[1];
        if (!insideQuote[0] && !insideQuote[1]) {
            if ((*itr == '>') || (*itr == '<'))
                return itr;
        }
    }
    return nullptr;
}


static const char* _xmlFindEndCommentTag(const char* itr, const char* itrEnd)
{
    for (; itr < itrEnd; itr++) {
        if ((*itr == '-') && ((itr + 1 < itrEnd) && (*(itr + 1) == '-')) && ((itr + 2 < itrEnd) && (*(itr + 2) == '>'))) return itr + 2;
    }
    return nullptr;
}


static const char* _xmlFindEndCdataTag(const char* itr, const char* itrEnd)
{
    for (; itr < itrEnd; itr++) {
        if ((*itr == ']') && ((itr + 1 < itrEnd) && (*(itr + 1) == ']')) && ((itr + 2 < itrEnd) && (*(itr + 2) == '>'))) return itr + 2;
    }
    return nullptr;
}


static const char* _xmlFindDoctypeChildEndTag(const char* itr, const char* itrEnd)
{
    for (; itr < itrEnd; itr++) {
        if (*itr == '>') return itr;
    }
    return nullptr;
}


static XMLType _getXMLType(const char* itr, const char* itrEnd, size_t &toff)
{
    toff = 0;
    if (itr[1] == '/') {
        toff = 1;
        return XMLType::Close;
    } else if (itr[1] == '?') {
        toff = 1;
        return XMLType::Processing;
    } else if (itr[1] == '!') {
        if ((itr + sizeof("<!DOCTYPE>") - 1 < itrEnd) && (!memcmp(itr + 2, "DOCTYPE", sizeof("DOCTYPE") - 1)) && ((itr[2 + sizeof("DOCTYPE") - 1] == '>') || (isspace((unsigned char)itr[2 + sizeof("DOCTYPE") - 1])))) {
            toff = sizeof("!DOCTYPE") - 1;
            return XMLType::Doctype;
        } else if ((itr + sizeof("<![CDATA[]]>") - 1 < itrEnd) && (!memcmp(itr + 2, "[CDATA[", sizeof("[CDATA[") - 1))) {
            toff = sizeof("![CDATA[") - 1;
            return XMLType::CData;
        } else if ((itr + sizeof("<!---->") - 1 < itrEnd) && (!memcmp(itr + 2, "--", sizeof("--") - 1))) {
            toff = sizeof("!--") - 1;
            return XMLType::Comment;
        } else if (itr + sizeof("<!>") - 1 < itrEnd) {
            toff = sizeof("!") - 1;
            return XMLType::DoctypeChild;
        }
        return XMLType::Open;
    }
    return XMLType::Open;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

const char* xmlNodeTypeToString(TVG_UNUSED SvgNodeType type)
{
#ifdef THORVG_LOG_ENABLED
    static const char* TYPE_NAMES[] = {
        "Svg",
        "G",
        "Defs",
        "Animation",
        "Arc",
        "Circle",
        "Ellipse",
        "Image",
        "Line",
        "Path",
        "Polygon",
        "Polyline",
        "Rect",
        "Text",
        "TextArea",
        "Tspan",
        "Use",
        "Video",
        "ClipPath",
        "Mask",
        "Symbol",
        "Filter",
        "GaussianBlur",
        "Pattern",
        "Unknown",
    };
    return TYPE_NAMES[(int) type];
#endif
    return nullptr;
}


bool isIgnoreUnsupportedLogElements(TVG_UNUSED const char* tagName)
{
#ifdef THORVG_LOG_ENABLED
    const auto elementsNum = 1;
    const char* const elements[] = { "title" };

    for (unsigned int i = 0; i < elementsNum; ++i) {
        if (!strncmp(tagName, elements[i], strlen(tagName))) {
            return true;
        }
    }
    return false;
#else
    return true;
#endif
}


bool xmlParseAttributes(const char* buf, unsigned bufLength, xmlAttributeCb func, const void* data)
{
    if (!buf || !func) return false;

    const char *itr = buf, *itrEnd = buf + bufLength;
    char* tmpBuf = tvg::malloc<char>(bufLength + 1);

    if (!tmpBuf) return false;

    while (itr < itrEnd) {
        const char* p = svgUtilSkipWhiteSpace(itr, itrEnd);
        const char *key, *keyEnd, *value, *valueEnd;
        char* tval;

        if (p == itrEnd) goto success;

        key = p;
        for (keyEnd = key; keyEnd < itrEnd; keyEnd++) {
            if ((*keyEnd == '=') || (isspace((unsigned char)*keyEnd))) break;
        }
        if (keyEnd == itrEnd) goto error;
        if (keyEnd == key) {  // There is no key. This case is invalid, but explores the following syntax.
            itr = keyEnd + 1;
            continue;
        }

        if (*keyEnd == '=') value = keyEnd + 1;
        else {
            value = (const char*)memchr(keyEnd, '=', itrEnd - keyEnd);
            if (!value) goto error;
            value++;
        }
        value = svgUtilSkipWhiteSpace(value, itrEnd);
        if (value == itrEnd) goto error;

        if ((*value == '"') || (*value == '\'')) {
            valueEnd = (const char*)memchr(value + 1, *value, itrEnd - value - 1);
            if (!valueEnd) goto error;
            value++;
        } else {
            valueEnd = _xmlFindWhiteSpace(value, itrEnd);
        }

        itr = (valueEnd < itrEnd) ? valueEnd + 1 : valueEnd;

        value = svgUtilSkipWhiteSpace(value, valueEnd);
        if (value < valueEnd) valueEnd = svgUtilUnskipWhiteSpace(valueEnd, value);

        memcpy(tmpBuf, key, keyEnd - key);
        tmpBuf[keyEnd - key] = '\0';

        tval = tmpBuf + (keyEnd - key) + 1;
        auto decodedLength = _xmlDecodeEntities(value, valueEnd, tval);
        if (decodedLength) {
            auto decodedEnd = svgUtilUnskipWhiteSpace(tval + decodedLength, tval);
            *const_cast<char*>(decodedEnd) = '\0';
            tval = const_cast<char*>(svgUtilSkipWhiteSpace(tval, decodedEnd));
        }

        if (!func((void*)data, tmpBuf, tval)) {
            if (_unsupported(tmpBuf, tval)) {
                TVGLOG("SVG", "Unsupported attributes used [Elements type: %s][Id : %s][Attribute: %s][Value: %s]", xmlNodeTypeToString(((SvgParserContext*)data)->parser->node->type), ((SvgParserContext*)data)->parser->node->id ? ((SvgParserContext*)data)->parser->node->id : "NO_ID", tmpBuf, tval ? tval : "NONE");
            }
        }
    }

success:
    tvg::free(tmpBuf);
    return true;

error:
    tvg::free(tmpBuf);
    return false;
}


bool xmlParse(const char* buf, unsigned bufLength, bool strip, xmlCb func, const void* data)
{
    const char *itr = buf, *itrEnd = buf + bufLength;

    while (itr < itrEnd) {
        if (itr[0] == '<') {
            //Invalid case
            if (itr + 1 >= itrEnd) return false;

            size_t toff = 0;
            XMLType type = _getXMLType(itr, itrEnd, toff);

            const char* p;
            if (type == XMLType::CData) p = _xmlFindEndCdataTag(itr + 1 + toff, itrEnd);
            else if (type == XMLType::DoctypeChild) p = _xmlFindDoctypeChildEndTag(itr + 1 + toff, itrEnd);
            else if (type == XMLType::Comment) p = _xmlFindEndCommentTag(itr + 1 + toff, itrEnd);
            else p = _xmlFindEndTag(itr + 1 + toff, itrEnd);

            if (p) {
                //Invalid case: '<' nested
                if (*p == '<' && type != XMLType::Doctype) return false;
                const char *start, *end;

                start = itr + 1 + toff;
                end = p;

                switch (type) {
                    case XMLType::Open: {
                        if (p[-1] == '/') {
                            type = XMLType::OpenEmpty;
                            end--;
                        }
                        break;
                    }
                    case XMLType::CData: {
                        if (!memcmp(p - 2, "]]", 2)) end -= 2;
                        break;
                    }
                    case XMLType::Processing: {
                        if (p[-1] == '?') end--;
                        break;
                    }
                    case XMLType::Comment: {
                        if (!memcmp(p - 2, "--", 2)) end -= 2;
                        break;
                    }
                    default: {
                        break;
                    }
                }

                if (strip && (type != XMLType::CData)) {
                    start = svgUtilSkipWhiteSpace(start, end);
                    end = svgUtilUnskipWhiteSpace(end, start);
                }

                if (!func((void*)data, type, start, (unsigned int)(end - start))) return false;

                itr = p + 1;
            } else {
                return false;
            }
        } else {
            const char *p, *end;

            if (strip && ((SvgParserContext*)data)->openedTag != OpenedTagType::Text) {
                p = itr;
                p = svgUtilSkipWhiteSpace(p, itrEnd);
                if (p != itr) {
                    if (!func((void*)data, XMLType::Ignored, itr, (unsigned int)(p - itr))) return false;
                    itr = p;
                }
            }

            p = _xmlFindStartTag(itr, itrEnd);
            if (!p) p = itrEnd;

            end = p;
            if (strip && ((SvgParserContext*)data)->openedTag != OpenedTagType::Text) end = svgUtilUnskipWhiteSpace(end, itr);

            if (itr != end) {
                auto length = (unsigned int)(end - itr);
                if (memchr(itr, '&', length)) {
                    auto decoded = tvg::malloc<char>(length + 1);
                    if (!decoded) return false;
                    auto decodedLength = _xmlDecodeEntities(itr, end, decoded);
                    auto result = func((void*)data, XMLType::Data, decoded, decodedLength);
                    tvg::free(decoded);
                    if (!result) return false;
                } else if (!func((void*)data, XMLType::Data, itr, length)) return false;
            }

            if (strip && (end < p) && !func((void*)data, XMLType::Ignored, end, (unsigned int)(p - end))) return false;

            itr = p;
        }
    }
    return true;
}


bool xmlParseW3CAttribute(const char* buf, unsigned bufLength, xmlAttributeCb func, const void* data)
{
    if (!buf) return false;

    const auto end = buf + bufLength;
    if (buf == end) return true;

    auto kmem = tvg::malloc<char>(end - buf + 1);
    auto vmem = tvg::malloc<char>(end - buf + 1);
    auto key = kmem;
    auto val = vmem;

    do {
        auto sep = (const char*)memchr(buf, ':', end - buf);
        auto next = (const char*)memchr(buf, ';', end - buf);
        if (!sep) sep = end;
        if (!next) next = end;
        auto src = buf;
        while (sep - src >= 3 && memcmp(src, "src", 3)) ++src;
        if (sep - src < 3) src = sep;

        if (sep < end && next < end && src < sep) {  // src tag from css font-face contains extra semicolon
            if (next + 1 < end) {
                next = (const char*)memchr(next + 1, ';', end - next - 1);
                if (!next) next = end;
            } else break;
        }

        key[0] = '\0';
        val[0] = '\0';

        if (sep < end && next >= end) {
            memcpy(key, buf, sep - buf);
            key[sep - buf] = '\0';

            memcpy(val, sep + 1, end - sep - 1);
            val[end - sep - 1] = '\0';
        } else if (sep < end && sep < next) {
            memcpy(key, buf, sep - buf);
            key[sep - buf] = '\0';

            memcpy(val, sep + 1, next - sep - 1);
            val[next - sep - 1] = '\0';
        } else if (next < end) {
            memcpy(key, buf, next - buf);
            key[next - buf] = '\0';
        }
        if (key[0]) {
            key = const_cast<char*>(svgUtilSkipWhiteSpace(key, key + strlen(key)));
            key[svgUtilUnskipWhiteSpace(key + strlen(key) , key) - key] = '\0';
            val = const_cast<char*>(svgUtilSkipWhiteSpace(val, val + strlen(val)));
            val[svgUtilUnskipWhiteSpace(val + strlen(val) , val) - val] = '\0';

            if (!func((void*)data, key, val)) {
                if (!_unsupported(key, val)) {
                    TVGLOG("SVG", "Unsupported attributes used [Elements type: %s][Id : %s][Attribute: %s][Value: %s]", xmlNodeTypeToString(((SvgParserContext*)data)->parser->node->type), ((SvgParserContext*)data)->parser->node->id ? ((SvgParserContext*)data)->parser->node->id : "NO_ID", key, val ? val : "NONE");
                }
            }
        }
        if (next >= end) break;
        buf = next + 1;
    } while (true);

    tvg::free(kmem);
    tvg::free(vmem);

    return true;
}


/*
 * Supported formats:
 * tag {}, .name {}, tag.name{}
 */
const char* xmlParseCSSAttribute(const char* buf, unsigned bufLength, char** tag, char** name, const char** attrs, unsigned* attrsLength)
{
    if (!buf) return nullptr;

    *tag = *name = nullptr;
    *attrsLength = 0;

    auto itr = svgUtilSkipWhiteSpace(buf, buf + bufLength);
    auto itrEnd = (const char*)memchr(buf, '{', bufLength);

    if (!itrEnd || itr == itrEnd) return nullptr;

    auto nextElement = (const char*)memchr(itrEnd, '}', bufLength - (itrEnd - buf));
    if (!nextElement) return nullptr;

    *attrs = itrEnd + 1;
    *attrsLength = nextElement - *attrs;

    const char *p;

    itrEnd = svgUtilUnskipWhiteSpace(itrEnd, itr);
    if (*(itrEnd - 1) == '.') return nullptr;

    for (p = itr; p < itrEnd; p++) {
        if (*p == '.') break;
    }

    if (p == itr) *tag = duplicate("all");
    else *tag = duplicate(itr, p - itr);

    if (p == itrEnd) *name = nullptr;
    else *name = duplicate(p + 1, itrEnd - p - 1);

    return (nextElement ? nextElement + 1 : nullptr);
}


const char* xmlFindAttributesTag(const char* buf, unsigned bufLength)
{
    const char *itr = buf, *itrEnd = buf + bufLength;

    for (; itr < itrEnd; itr++) {
        if (!isspace((unsigned char)*itr)) {
            //User skip tagname and already gave it the attributes.
            if (*itr == '=') return buf;
        } else {
            return itr;
        }
    }

    return itrEnd;
}
