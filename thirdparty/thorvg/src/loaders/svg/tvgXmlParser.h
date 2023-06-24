/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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

#ifndef _TVG_SIMPLE_XML_PARSER_H_
#define _TVG_SIMPLE_XML_PARSER_H_

#include "tvgSvgLoaderCommon.h"

#define NUMBER_OF_XML_ENTITIES 8
const char* const xmlEntity[] = {"&quot;", "&nbsp;", "&apos;", "&amp;", "&lt;", "&gt;", "&#035;", "&#039;"};
const int xmlEntityLength[] = {6, 6, 6, 5, 4, 4, 6, 6};

enum class SimpleXMLType
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

typedef bool (*simpleXMLCb)(void* data, SimpleXMLType type, const char* content, unsigned int length);
typedef bool (*simpleXMLAttributeCb)(void* data, const char* key, const char* value);

bool simpleXmlParseAttributes(const char* buf, unsigned bufLength, simpleXMLAttributeCb func, const void* data);
bool simpleXmlParse(const char* buf, unsigned bufLength, bool strip, simpleXMLCb func, const void* data);
bool simpleXmlParseW3CAttribute(const char* buf, unsigned bufLength, simpleXMLAttributeCb func, const void* data);
const char* simpleXmlParseCSSAttribute(const char* buf, unsigned bufLength, char** tag, char** name, const char** attrs, unsigned* attrsLength);
const char* simpleXmlFindAttributesTag(const char* buf, unsigned bufLength);
bool isIgnoreUnsupportedLogElements(const char* tagName);
const char* simpleXmlNodeTypeToString(SvgNodeType type);

#endif //_TVG_SIMPLE_XML_PARSER_H_
