/**
 * pugixml parser - version 1.9
 * --------------------------------------------------------
 * Copyright (C) 2006-2018, by Arseny Kapoulkine (arseny.kapoulkine@gmail.com)
 * Report bugs and download new versions at http://pugixml.org/
 *
 * This library is distributed under the MIT License. See notice at the end
 * of this file.
 *
 * This work is based on the pugxml parser, which is:
 * Copyright (C) 2003, by Kristen Wegner (kristen@tima.net)
 */

#ifndef HEADER_PUGICONFIG_HPP
#define HEADER_PUGICONFIG_HPP

// Uncomment this to enable wchar_t mode
// #define PUGIXML_WCHAR_MODE

// Uncomment this to enable compact mode
// #define PUGIXML_COMPACT

// Uncomment this to disable XPath
// #define PUGIXML_NO_XPATH

// Uncomment this to disable STL
// #define PUGIXML_NO_STL

// Uncomment this to disable exceptions
// #define PUGIXML_NO_EXCEPTIONS

// Set this to control attributes for public classes/functions, i.e.:
// #define PUGIXML_API __declspec(dllexport) // to export all public symbols from DLL
// #define PUGIXML_CLASS __declspec(dllimport) // to import all classes from DLL
// #define PUGIXML_FUNCTION __fastcall // to set calling conventions to all public functions to fastcall
// In absence of PUGIXML_CLASS/PUGIXML_FUNCTION definitions PUGIXML_API is used instead

// Tune these constants to adjust memory-related behavior
// #define PUGIXML_MEMORY_PAGE_SIZE 32768
// #define PUGIXML_MEMORY_OUTPUT_STACK 10240
// #define PUGIXML_MEMORY_XPATH_PAGE_SIZE 4096

// Uncomment this to switch to header-only version
// #define PUGIXML_HEADER_ONLY

// Uncomment this to enable long long support
// #define PUGIXML_HAS_LONG_LONG

#endif

/**
 * Copyright (c) 2006-2018 Arseny Kapoulkine
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
