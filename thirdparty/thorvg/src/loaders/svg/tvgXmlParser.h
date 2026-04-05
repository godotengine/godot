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

#ifndef _TVG_XML_PARSER_H_
#define _TVG_XML_PARSER_H_

#include "tvgSvgLoaderCommon.h"

#define NUMBER_OF_XML_ENTITIES 9
const char* const xmlEntity[] = {"&#10;", "&quot;", "&nbsp;", "&apos;", "&amp;", "&lt;", "&gt;", "&#035;", "&#039;"};
const int xmlEntityLength[] = {5, 6, 6, 6, 5, 4, 4, 6, 6};

enum class XMLType
{
    Open = 0,     //!< \<tag attribute="value"\>
    OpenEmpty,    //!< \<tag attribute="value" /\>
    Close,        //!< \</tag\>
    Data,         //!< tag text data
    CData,        //!< \<![cdata[something]]\>
    Error,        //!< error contents
    Processing,   //!< \<?xml ... ?\> \<?php .. ?\>
    Doctype,      //!< \<!doctype html
    Comment,      //!< \<!-- something --\>
    Ignored,      //!< whatever is ignored by parser, like whitespace
    DoctypeChild  //!< \<!doctype_child
};

typedef bool (*xmlCb)(void* data, XMLType type, const char* content, unsigned int length);
typedef bool (*xmlAttributeCb)(void* data, const char* key, const char* value);

bool xmlParseAttributes(const char* buf, unsigned bufLength, xmlAttributeCb func, const void* data);
bool xmlParse(const char* buf, unsigned bufLength, bool strip, xmlCb func, const void* data);
bool xmlParseW3CAttribute(const char* buf, unsigned bufLength, xmlAttributeCb func, const void* data);
const char* xmlParseCSSAttribute(const char* buf, unsigned bufLength, char** tag, char** name, const char** attrs, unsigned* attrsLength);
const char* xmlFindAttributesTag(const char* buf, unsigned bufLength);
bool isIgnoreUnsupportedLogElements(const char* tagName);
const char* xmlNodeTypeToString(SvgNodeType type);

#endif //_TVG_XML_PARSER_H_
