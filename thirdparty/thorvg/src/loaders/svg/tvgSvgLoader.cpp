/*
 * Copyright (c) 2020-2021 Samsung Electronics Co., Ltd. All rights reserved.

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

/*
 * Copyright notice for the EFL:

 * Copyright (C) EFL developers (see AUTHORS)

 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.

 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#define _USE_MATH_DEFINES       //Math Constants are not defined in Standard C/C++.

#include <cstring>
#include <fstream>
#include <float.h>
#include <math.h>
#include "tvgLoader.h"
#include "tvgXmlParser.h"
#include "tvgSvgLoader.h"
#include "tvgSvgSceneBuilder.h"
#include "tvgSvgUtil.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

/*
 * According to: https://www.w3.org/TR/SVG2/coords.html#Units
 * and: https://www.w3.org/TR/css-values-4/#absolute-lengths
 */
#define PX_PER_IN 96        //1 in = 96 px
#define PX_PER_PC 16        //1 pc = 1/6 in  -> PX_PER_IN/6
#define PX_PER_PT 1.333333f //1 pt = 1/72 in -> PX_PER_IN/72
#define PX_PER_MM 3.779528f //1 in = 25.4 mm -> PX_PER_IN/25.4
#define PX_PER_CM 37.79528f //1 in = 2.54 cm -> PX_PER_IN/2.54


typedef SvgNode* (*FactoryMethod)(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength);
typedef SvgStyleGradient* (*GradientFactoryMethod)(SvgLoaderData* loader, const char* buf, unsigned bufLength);


static char* _skipSpace(const char* str, const char* end)
{
    while (((end && str < end) || (!end && *str != '\0')) && isspace(*str)) {
        ++str;
    }
    return (char*) str;
}


static char* _copyId(const char* str)
{
    if (!str) return nullptr;

    return strdup(str);
}


static const char* _skipComma(const char* content)
{
    content = _skipSpace(content, nullptr);
    if (*content == ',') return content + 1;
    return content;
}


static bool _parseNumber(const char** content, float* number)
{
    char* end = nullptr;

    *number = svgUtilStrtof(*content, &end);
    //If the start of string is not number
    if ((*content) == end) return false;
    //Skip comma if any
    *content = _skipComma(end);
    return true;
}

/**
 * According to https://www.w3.org/TR/SVG/coords.html#Units
 */
static float _toFloat(const SvgParser* svgParse, const char* str, SvgParserLengthType type)
{
    float parsedValue = svgUtilStrtof(str, nullptr);

    if (strstr(str, "cm")) parsedValue *= PX_PER_CM;
    else if (strstr(str, "mm")) parsedValue *= PX_PER_MM;
    else if (strstr(str, "pt")) parsedValue *= PX_PER_PT;
    else if (strstr(str, "pc")) parsedValue *= PX_PER_PC;
    else if (strstr(str, "in")) parsedValue *= PX_PER_IN;
    else if (strstr(str, "%")) {
        if (type == SvgParserLengthType::Vertical) parsedValue = (parsedValue / 100.0) * svgParse->global.h;
        else if (type == SvgParserLengthType::Horizontal) parsedValue = (parsedValue / 100.0) * svgParse->global.w;
        else //if other then it's radius
        {
            float max = (float)svgParse->global.w;
            if (max < svgParse->global.h)
                max = (float)svgParse->global.h;
            parsedValue = (parsedValue / 100.0) * max;
        }
    }
    //TODO: Implement 'em', 'ex' attributes

    return parsedValue;
}


static float _gradientToFloat(const SvgParser* svgParse, const char* str, bool& isPercentage)
{
    char* end = nullptr;

    float parsedValue = svgUtilStrtof(str, &end);
    isPercentage = false;

    if (strstr(str, "%")) {
        parsedValue = parsedValue / 100.0;
        isPercentage = true;
    }
    else if (strstr(str, "cm")) parsedValue *= PX_PER_CM;
    else if (strstr(str, "mm")) parsedValue *= PX_PER_MM;
    else if (strstr(str, "pt")) parsedValue *= PX_PER_PT;
    else if (strstr(str, "pc")) parsedValue *= PX_PER_PC;
    else if (strstr(str, "in")) parsedValue *= PX_PER_IN;
    //TODO: Implement 'em', 'ex' attributes

    return parsedValue;
}


static float _toOffset(const char* str)
{
    char* end = nullptr;
    auto strEnd = str + strlen(str);

    float parsedValue = svgUtilStrtof(str, &end);

    end = _skipSpace(end, nullptr);
    auto ptr = strstr(str, "%");

    if (ptr) {
        parsedValue = parsedValue / 100.0;
        if (end != ptr || (end + 1) != strEnd) return 0;
    } else if (end != strEnd) return 0;

    return parsedValue;
}


static int _toOpacity(const char* str)
{
    char* end = nullptr;
    float opacity = svgUtilStrtof(str, &end);

    if (end) {
        if (end[0] == '%' && end[1] == '\0') return lrint(opacity * 2.55f);
        else if (*end == '\0') return lrint(opacity * 255);
    }
    return 255;
}


#define _PARSE_TAG(Type, Name, Name1, Tags_Array, Default)                        \
    static Type _to##Name1(const char* str)                                       \
    {                                                                             \
        unsigned int i;                                                           \
                                                                                  \
        for (i = 0; i < sizeof(Tags_Array) / sizeof(Tags_Array[0]); i++) {        \
            if (!strcmp(str, Tags_Array[i].tag)) return Tags_Array[i].Name;       \
        }                                                                         \
        return Default;                                                           \
    }


/* parse the line cap used during stroking a path.
 * Value:    butt | round | square | inherit
 * Initial:    butt
 * https://www.w3.org/TR/SVG/painting.html
 */
static constexpr struct
{
    StrokeCap lineCap;
    const char* tag;
} lineCapTags[] = {
    { StrokeCap::Butt, "butt" },
    { StrokeCap::Round, "round" },
    { StrokeCap::Square, "square" }
};


_PARSE_TAG(StrokeCap, lineCap, LineCap, lineCapTags, StrokeCap::Butt)


/* parse the line join used during stroking a path.
 * Value:   miter | round | bevel | inherit
 * Initial:    miter
 * https://www.w3.org/TR/SVG/painting.html
 */
static constexpr struct
{
    StrokeJoin lineJoin;
    const char* tag;
} lineJoinTags[] = {
    { StrokeJoin::Miter, "miter" },
    { StrokeJoin::Round, "round" },
    { StrokeJoin::Bevel, "bevel" }
};


_PARSE_TAG(StrokeJoin, lineJoin, LineJoin, lineJoinTags, StrokeJoin::Miter)


/* parse the fill rule used during filling a path.
 * Value:   nonzero | evenodd | inherit
 * Initial:    nonzero
 * https://www.w3.org/TR/SVG/painting.html
 */
static constexpr struct
{
    FillRule fillRule;
    const char* tag;
} fillRuleTags[] = {
    { FillRule::EvenOdd, "evenodd" }
};


_PARSE_TAG(FillRule, fillRule, FillRule, fillRuleTags, FillRule::Winding)


/* parse the dash pattern used during stroking a path.
 * Value:   none | <dasharray> | inherit
 * Initial:    none
 * https://www.w3.org/TR/SVG/painting.html
 */
static void _parseDashArray(SvgLoaderData* loader, const char *str, SvgDash* dash)
{
    if (!strncmp(str, "none", 4)) return;

    char *end = nullptr;

    while (*str) {
        str = _skipComma(str);
        float parsedValue = svgUtilStrtof(str, &end);
        if (str == end) break;
        if (parsedValue <= 0.0f) break;
        if (*end == '%') {
            ++end;
            //Refers to the diagonal length of the viewport.
            //https://www.w3.org/TR/SVG2/coords.html#Units
            parsedValue = (sqrtf(pow(loader->svgParse->global.w, 2) + pow(loader->svgParse->global.h, 2)) / sqrtf(2.0f)) * (parsedValue / 100.0f);
        }
        (*dash).array.push(parsedValue);
        str = end;
    }
    //If dash array size is 1, it means that dash and gap size are the same.
    if ((*dash).array.count == 1) (*dash).array.push((*dash).array.data[0]);
}


static char* _idFromUrl(const char* url)
{
    url = _skipSpace(url, nullptr);
    if ((*url) == '(') {
        ++url;
        url = _skipSpace(url, nullptr);
    }

    if ((*url) == '\'') ++url;
    if ((*url) == '#') ++url;

    int i = 0;
    while (url[i] > ' ' && url[i] != ')' && url[i] != '\'') ++i;

    //custom strndup() for portability
    int len = strlen(url);
    if (i < len) len = i;

    auto ret = (char*) malloc(len + 1);
    if (!ret) return 0;
    ret[len] = '\0';
    return (char*) memcpy(ret, url, len);
}


static unsigned char _parserColor(const char* value, char** end)
{
    float r;

    r = svgUtilStrtof(value, end);
    *end = _skipSpace(*end, nullptr);
    if (**end == '%') r = 255 * r / 100;
    *end = _skipSpace(*end, nullptr);

    if (r < 0 || r > 255) {
        *end = nullptr;
        return 0;
    }

    return lrint(r);
}


static constexpr struct
{
    const char* name;
    unsigned int value;
} colors[] = {
    { "aliceblue", 0xfff0f8ff },
    { "antiquewhite", 0xfffaebd7 },
    { "aqua", 0xff00ffff },
    { "aquamarine", 0xff7fffd4 },
    { "azure", 0xfff0ffff },
    { "beige", 0xfff5f5dc },
    { "bisque", 0xffffe4c4 },
    { "black", 0xff000000 },
    { "blanchedalmond", 0xffffebcd },
    { "blue", 0xff0000ff },
    { "blueviolet", 0xff8a2be2 },
    { "brown", 0xffa52a2a },
    { "burlywood", 0xffdeb887 },
    { "cadetblue", 0xff5f9ea0 },
    { "chartreuse", 0xff7fff00 },
    { "chocolate", 0xffd2691e },
    { "coral", 0xffff7f50 },
    { "cornflowerblue", 0xff6495ed },
    { "cornsilk", 0xfffff8dc },
    { "crimson", 0xffdc143c },
    { "cyan", 0xff00ffff },
    { "darkblue", 0xff00008b },
    { "darkcyan", 0xff008b8b },
    { "darkgoldenrod", 0xffb8860b },
    { "darkgray", 0xffa9a9a9 },
    { "darkgrey", 0xffa9a9a9 },
    { "darkgreen", 0xff006400 },
    { "darkkhaki", 0xffbdb76b },
    { "darkmagenta", 0xff8b008b },
    { "darkolivegreen", 0xff556b2f },
    { "darkorange", 0xffff8c00 },
    { "darkorchid", 0xff9932cc },
    { "darkred", 0xff8b0000 },
    { "darksalmon", 0xffe9967a },
    { "darkseagreen", 0xff8fbc8f },
    { "darkslateblue", 0xff483d8b },
    { "darkslategray", 0xff2f4f4f },
    { "darkslategrey", 0xff2f4f4f },
    { "darkturquoise", 0xff00ced1 },
    { "darkviolet", 0xff9400d3 },
    { "deeppink", 0xffff1493 },
    { "deepskyblue", 0xff00bfff },
    { "dimgray", 0xff696969 },
    { "dimgrey", 0xff696969 },
    { "dodgerblue", 0xff1e90ff },
    { "firebrick", 0xffb22222 },
    { "floralwhite", 0xfffffaf0 },
    { "forestgreen", 0xff228b22 },
    { "fuchsia", 0xffff00ff },
    { "gainsboro", 0xffdcdcdc },
    { "ghostwhite", 0xfff8f8ff },
    { "gold", 0xffffd700 },
    { "goldenrod", 0xffdaa520 },
    { "gray", 0xff808080 },
    { "grey", 0xff808080 },
    { "green", 0xff008000 },
    { "greenyellow", 0xffadff2f },
    { "honeydew", 0xfff0fff0 },
    { "hotpink", 0xffff69b4 },
    { "indianred", 0xffcd5c5c },
    { "indigo", 0xff4b0082 },
    { "ivory", 0xfffffff0 },
    { "khaki", 0xfff0e68c },
    { "lavender", 0xffe6e6fa },
    { "lavenderblush", 0xfffff0f5 },
    { "lawngreen", 0xff7cfc00 },
    { "lemonchiffon", 0xfffffacd },
    { "lightblue", 0xffadd8e6 },
    { "lightcoral", 0xfff08080 },
    { "lightcyan", 0xffe0ffff },
    { "lightgoldenrodyellow", 0xfffafad2 },
    { "lightgray", 0xffd3d3d3 },
    { "lightgrey", 0xffd3d3d3 },
    { "lightgreen", 0xff90ee90 },
    { "lightpink", 0xffffb6c1 },
    { "lightsalmon", 0xffffa07a },
    { "lightseagreen", 0xff20b2aa },
    { "lightskyblue", 0xff87cefa },
    { "lightslategray", 0xff778899 },
    { "lightslategrey", 0xff778899 },
    { "lightsteelblue", 0xffb0c4de },
    { "lightyellow", 0xffffffe0 },
    { "lime", 0xff00ff00 },
    { "limegreen", 0xff32cd32 },
    { "linen", 0xfffaf0e6 },
    { "magenta", 0xffff00ff },
    { "maroon", 0xff800000 },
    { "mediumaquamarine", 0xff66cdaa },
    { "mediumblue", 0xff0000cd },
    { "mediumorchid", 0xffba55d3 },
    { "mediumpurple", 0xff9370d8 },
    { "mediumseagreen", 0xff3cb371 },
    { "mediumslateblue", 0xff7b68ee },
    { "mediumspringgreen", 0xff00fa9a },
    { "mediumturquoise", 0xff48d1cc },
    { "mediumvioletred", 0xffc71585 },
    { "midnightblue", 0xff191970 },
    { "mintcream", 0xfff5fffa },
    { "mistyrose", 0xffffe4e1 },
    { "moccasin", 0xffffe4b5 },
    { "navajowhite", 0xffffdead },
    { "navy", 0xff000080 },
    { "oldlace", 0xfffdf5e6 },
    { "olive", 0xff808000 },
    { "olivedrab", 0xff6b8e23 },
    { "orange", 0xffffa500 },
    { "orangered", 0xffff4500 },
    { "orchid", 0xffda70d6 },
    { "palegoldenrod", 0xffeee8aa },
    { "palegreen", 0xff98fb98 },
    { "paleturquoise", 0xffafeeee },
    { "palevioletred", 0xffd87093 },
    { "papayawhip", 0xffffefd5 },
    { "peachpuff", 0xffffdab9 },
    { "peru", 0xffcd853f },
    { "pink", 0xffffc0cb },
    { "plum", 0xffdda0dd },
    { "powderblue", 0xffb0e0e6 },
    { "purple", 0xff800080 },
    { "red", 0xffff0000 },
    { "rosybrown", 0xffbc8f8f },
    { "royalblue", 0xff4169e1 },
    { "saddlebrown", 0xff8b4513 },
    { "salmon", 0xfffa8072 },
    { "sandybrown", 0xfff4a460 },
    { "seagreen", 0xff2e8b57 },
    { "seashell", 0xfffff5ee },
    { "sienna", 0xffa0522d },
    { "silver", 0xffc0c0c0 },
    { "skyblue", 0xff87ceeb },
    { "slateblue", 0xff6a5acd },
    { "slategray", 0xff708090 },
    { "slategrey", 0xff708090 },
    { "snow", 0xfffffafa },
    { "springgreen", 0xff00ff7f },
    { "steelblue", 0xff4682b4 },
    { "tan", 0xffd2b48c },
    { "teal", 0xff008080 },
    { "thistle", 0xffd8bfd8 },
    { "tomato", 0xffff6347 },
    { "turquoise", 0xff40e0d0 },
    { "violet", 0xffee82ee },
    { "wheat", 0xfff5deb3 },
    { "white", 0xffffffff },
    { "whitesmoke", 0xfff5f5f5 },
    { "yellow", 0xffffff00 },
    { "yellowgreen", 0xff9acd32 }
};


static void _toColor(const char* str, uint8_t* r, uint8_t* g, uint8_t* b, char** ref)
{
    unsigned int len = strlen(str);
    char *red, *green, *blue;
    unsigned char tr, tg, tb;

    if (len == 4 && str[0] == '#') {
        //Case for "#456" should be interprete as "#445566"
        if (isxdigit(str[1]) && isxdigit(str[2]) && isxdigit(str[3])) {
            char tmp[3] = { '\0', '\0', '\0' };
            tmp[0] = str[1];
            tmp[1] = str[1];
            *r = strtol(tmp, nullptr, 16);
            tmp[0] = str[2];
            tmp[1] = str[2];
            *g = strtol(tmp, nullptr, 16);
            tmp[0] = str[3];
            tmp[1] = str[3];
            *b = strtol(tmp, nullptr, 16);
        }
    } else if (len == 7 && str[0] == '#') {
        if (isxdigit(str[1]) && isxdigit(str[2]) && isxdigit(str[3]) && isxdigit(str[4]) && isxdigit(str[5]) && isxdigit(str[6])) {
            char tmp[3] = { '\0', '\0', '\0' };
            tmp[0] = str[1];
            tmp[1] = str[2];
            *r = strtol(tmp, nullptr, 16);
            tmp[0] = str[3];
            tmp[1] = str[4];
            *g = strtol(tmp, nullptr, 16);
            tmp[0] = str[5];
            tmp[1] = str[6];
            *b = strtol(tmp, nullptr, 16);
        }
    } else if (len >= 10 && (str[0] == 'r' || str[0] == 'R') && (str[1] == 'g' || str[1] == 'G') && (str[2] == 'b' || str[2] == 'B') && str[3] == '(' && str[len - 1] == ')') {
        tr = _parserColor(str + 4, &red);
        if (red && *red == ',') {
            tg = _parserColor(red + 1, &green);
            if (green && *green == ',') {
                tb = _parserColor(green + 1, &blue);
                if (blue && blue[0] == ')' && blue[1] == '\0') {
                    *r = tr;
                    *g = tg;
                    *b = tb;
                }
            }
        }
    } else if (len >= 3 && !strncmp(str, "url", 3)) {
        *ref = _idFromUrl((const char*)(str + 3));
    } else {
        //Handle named color
        for (unsigned int i = 0; i < (sizeof(colors) / sizeof(colors[0])); i++) {
            if (!strcasecmp(colors[i].name, str)) {
                *r = (((uint8_t*)(&(colors[i].value)))[2]);
                *g = (((uint8_t*)(&(colors[i].value)))[1]);
                *b = (((uint8_t*)(&(colors[i].value)))[0]);
                return;
            }
        }
    }
}


static char* _parseNumbersArray(char* str, float* points, int* ptCount, int len)
{
    int count = 0;
    char* end = nullptr;

    str = _skipSpace(str, nullptr);
    while ((count < len) && (isdigit(*str) || *str == '-' || *str == '+' || *str == '.')) {
        points[count++] = svgUtilStrtof(str, &end);
        str = end;
        str = _skipSpace(str, nullptr);
        if (*str == ',') ++str;
        //Eat the rest of space
        str = _skipSpace(str, nullptr);
    }
    *ptCount = count;
    return str;
}


enum class MatrixState {
    Unknown,
    Matrix,
    Translate,
    Rotate,
    Scale,
    SkewX,
    SkewY
};


#define MATRIX_DEF(Name, Value)     \
    {                               \
#Name, sizeof(#Name), Value \
    }


static constexpr struct
{
    const char* tag;
    int sz;
    MatrixState state;
} matrixTags[] = {
    MATRIX_DEF(matrix, MatrixState::Matrix),
    MATRIX_DEF(translate, MatrixState::Translate),
    MATRIX_DEF(rotate, MatrixState::Rotate),
    MATRIX_DEF(scale, MatrixState::Scale),
    MATRIX_DEF(skewX, MatrixState::SkewX),
    MATRIX_DEF(skewY, MatrixState::SkewY)
};


static void _matrixCompose(const Matrix* m1, const Matrix* m2, Matrix* dst)
{
    auto a11 = (m1->e11 * m2->e11) + (m1->e12 * m2->e21) + (m1->e13 * m2->e31);
    auto a12 = (m1->e11 * m2->e12) + (m1->e12 * m2->e22) + (m1->e13 * m2->e32);
    auto a13 = (m1->e11 * m2->e13) + (m1->e12 * m2->e23) + (m1->e13 * m2->e33);

    auto a21 = (m1->e21 * m2->e11) + (m1->e22 * m2->e21) + (m1->e23 * m2->e31);
    auto a22 = (m1->e21 * m2->e12) + (m1->e22 * m2->e22) + (m1->e23 * m2->e32);
    auto a23 = (m1->e21 * m2->e13) + (m1->e22 * m2->e23) + (m1->e23 * m2->e33);

    auto a31 = (m1->e31 * m2->e11) + (m1->e32 * m2->e21) + (m1->e33 * m2->e31);
    auto a32 = (m1->e31 * m2->e12) + (m1->e32 * m2->e22) + (m1->e33 * m2->e32);
    auto a33 = (m1->e31 * m2->e13) + (m1->e32 * m2->e23) + (m1->e33 * m2->e33);

    dst->e11 = a11;
    dst->e12 = a12;
    dst->e13 = a13;
    dst->e21 = a21;
    dst->e22 = a22;
    dst->e23 = a23;
    dst->e31 = a31;
    dst->e32 = a32;
    dst->e33 = a33;
}


/* parse transform attribute
 * https://www.w3.org/TR/SVG/coords.html#TransformAttribute
 */
static Matrix* _parseTransformationMatrix(const char* value)
{
    const int POINT_CNT = 8;

    auto matrix = (Matrix*)malloc(sizeof(Matrix));
    if (!matrix) return nullptr;
    *matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    float points[POINT_CNT];
    int ptCount = 0;
    char* str = (char*)value;
    char* end = str + strlen(str);

    while (str < end) {
        auto state = MatrixState::Unknown;

        if (isspace(*str) || (*str == ',')) {
            ++str;
            continue;
        }
        for (unsigned int i = 0; i < sizeof(matrixTags) / sizeof(matrixTags[0]); i++) {
            if (!strncmp(matrixTags[i].tag, str, matrixTags[i].sz - 1)) {
                state = matrixTags[i].state;
                str += (matrixTags[i].sz - 1);
                break;
            }
        }
        if (state == MatrixState::Unknown) goto error;

        str = _skipSpace(str, end);
        if (*str != '(') goto error;
        ++str;
        str = _parseNumbersArray(str, points, &ptCount, POINT_CNT);
        if (*str != ')') goto error;
        ++str;

        if (state == MatrixState::Matrix) {
            if (ptCount != 6) goto error;
            Matrix tmp = {points[0], points[2], points[4], points[1], points[3], points[5], 0, 0, 1};
            _matrixCompose(matrix, &tmp, matrix);
        } else if (state == MatrixState::Translate) {
            if (ptCount == 1) {
                Matrix tmp = {1, 0, points[0], 0, 1, 0, 0, 0, 1};
                _matrixCompose(matrix, &tmp, matrix);
            } else if (ptCount == 2) {
                Matrix tmp = {1, 0, points[0], 0, 1, points[1], 0, 0, 1};
                _matrixCompose(matrix, &tmp, matrix);
            } else goto error;
        } else if (state == MatrixState::Rotate) {
            //Transform to signed.
            points[0] = fmod(points[0], 360);
            if (points[0] < 0) points[0] += 360;
            auto c = cosf(points[0] * (M_PI / 180.0));
            auto s = sinf(points[0] * (M_PI / 180.0));
            if (ptCount == 1) {
                Matrix tmp = { c, -s, 0, s, c, 0, 0, 0, 1 };
                _matrixCompose(matrix, &tmp, matrix);
            } else if (ptCount == 3) {
                Matrix tmp = { 1, 0, points[1], 0, 1, points[2], 0, 0, 1 };
                _matrixCompose(matrix, &tmp, matrix);
                tmp = { c, -s, 0, s, c, 0, 0, 0, 1 };
                _matrixCompose(matrix, &tmp, matrix);
                tmp = { 1, 0, -points[1], 0, 1, -points[2], 0, 0, 1 };
                _matrixCompose(matrix, &tmp, matrix);
            } else {
                goto error;
            }
        } else if (state == MatrixState::Scale) {
            if (ptCount < 1 || ptCount > 2) goto error;
            auto sx = points[0];
            auto sy = sx;
            if (ptCount == 2) sy = points[1];
            Matrix tmp = { sx, 0, 0, 0, sy, 0, 0, 0, 1 };
            _matrixCompose(matrix, &tmp, matrix);
        }
    }
    return matrix;
error:
    if (matrix) free(matrix);
    return nullptr;
}


#define LENGTH_DEF(Name, Value)     \
    {                               \
#Name, sizeof(#Name), Value \
    }


/*
// TODO - remove?
static constexpr struct
{
    const char* tag;
    int sz;
    SvgLengthType type;
} lengthTags[] = {
    LENGTH_DEF(%, SvgLengthType::Percent),
    LENGTH_DEF(px, SvgLengthType::Px),
    LENGTH_DEF(pc, SvgLengthType::Pc),
    LENGTH_DEF(pt, SvgLengthType::Pt),
    LENGTH_DEF(mm, SvgLengthType::Mm),
    LENGTH_DEF(cm, SvgLengthType::Cm),
    LENGTH_DEF(in, SvgLengthType::In)
};

static float _parseLength(const char* str, SvgLengthType* type)
{
    float value;
    int sz = strlen(str);

    *type = SvgLengthType::Px;
    for (unsigned int i = 0; i < sizeof(lengthTags) / sizeof(lengthTags[0]); i++) {
        if (lengthTags[i].sz - 1 == sz && !strncmp(lengthTags[i].tag, str, sz)) *type = lengthTags[i].type;
    }
    value = svgUtilStrtof(str, nullptr);
    return value;
}
*/

static bool _parseStyleAttr(void* data, const char* key, const char* value);
static bool _parseStyleAttr(void* data, const char* key, const char* value, bool style);


static bool _attrParseSvgNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgDocNode* doc = &(node->node.doc);

    if (!strcmp(key, "width")) {
        doc->w = _toFloat(loader->svgParse, value, SvgParserLengthType::Horizontal);
    } else if (!strcmp(key, "height")) {
        doc->h = _toFloat(loader->svgParse, value, SvgParserLengthType::Vertical);
    } else if (!strcmp(key, "viewBox")) {
        if (_parseNumber(&value, &doc->vx)) {
            if (_parseNumber(&value, &doc->vy)) {
                if (_parseNumber(&value, &doc->vw)) {
                    _parseNumber(&value, &doc->vh);
                    loader->svgParse->global.h = (uint32_t)doc->vh;
                }
                loader->svgParse->global.w = (uint32_t)doc->vw;
            }
            loader->svgParse->global.y = (int)doc->vy;
        }
        loader->svgParse->global.x = (int)doc->vx;
    } else if (!strcmp(key, "preserveAspectRatio")) {
        if (!strcmp(value, "none")) doc->preserveAspect = false;
    } else if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    }
#ifdef THORVG_LOG_ENABLED
    else if ((!strcmp(key, "x") || !strcmp(key, "y")) && fabsf(svgUtilStrtof(value, nullptr)) > FLT_EPSILON ) {
        TVGLOG("SVG", "Unsupported attributes used [Elements type: Svg][Attribute: %s][Value: %s]", key, value);
    }
#endif
    else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


//https://www.w3.org/TR/SVGTiny12/painting.html#SpecifyingPaint
static void _handlePaintAttr(SvgPaint* paint, const char* value)
{
    if (!strcmp(value, "none")) {
        //No paint property
        paint->none = true;
        return;
    }
    paint->none = false;
    if (!strcmp(value, "currentColor")) {
        paint->curColor = true;
        return;
    }
    _toColor(value, &paint->color.r, &paint->color.g, &paint->color.b, &paint->url);
}


static void _handleColorAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    SvgStyleProperty* style = node->style;
    style->curColorSet = true;
    _toColor(value, &style->color.r, &style->color.g, &style->color.b, nullptr);
}


static void _handleFillAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    SvgStyleProperty* style = node->style;
    style->fill.flags = (SvgFillFlags)((int)style->fill.flags | (int)SvgFillFlags::Paint);
    _handlePaintAttr(&style->fill.paint, value);
}


static void _handleStrokeAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    SvgStyleProperty* style = node->style;
    style->stroke.flags = (SvgStrokeFlags)((int)style->stroke.flags | (int)SvgStrokeFlags::Paint);
    _handlePaintAttr(&style->stroke.paint, value);
}


static void _handleStrokeOpacityAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->stroke.flags = (SvgStrokeFlags)((int)node->style->stroke.flags | (int)SvgStrokeFlags::Opacity);
    node->style->stroke.opacity = _toOpacity(value);
}

static void _handleStrokeDashArrayAttr(SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->stroke.flags = (SvgStrokeFlags)((int)node->style->stroke.flags | (int)SvgStrokeFlags::Dash);
    _parseDashArray(loader, value, &node->style->stroke.dash);
}

static void _handleStrokeWidthAttr(SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->stroke.flags = (SvgStrokeFlags)((int)node->style->stroke.flags | (int)SvgStrokeFlags::Width);
    node->style->stroke.width = _toFloat(loader->svgParse, value, SvgParserLengthType::Horizontal);
}


static void _handleStrokeLineCapAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->stroke.flags = (SvgStrokeFlags)((int)node->style->stroke.flags | (int)SvgStrokeFlags::Cap);
    node->style->stroke.cap = _toLineCap(value);
}


static void _handleStrokeLineJoinAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->stroke.flags = (SvgStrokeFlags)((int)node->style->stroke.flags | (int)SvgStrokeFlags::Join);
    node->style->stroke.join = _toLineJoin(value);
}


static void _handleFillRuleAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->fill.flags = (SvgFillFlags)((int)node->style->fill.flags | (int)SvgFillFlags::FillRule);
    node->style->fill.fillRule = _toFillRule(value);
}


static void _handleOpacityAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->opacity = _toOpacity(value);
}


static void _handleFillOpacityAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->style->fill.flags = (SvgFillFlags)((int)node->style->fill.flags | (int)SvgFillFlags::Opacity);
    node->style->fill.opacity = _toOpacity(value);
}


static void _handleTransformAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    node->transform = _parseTransformationMatrix(value);
}


static void _handleClipPathAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    SvgStyleProperty* style = node->style;
    int len = strlen(value);
    if (len >= 3 && !strncmp(value, "url", 3)) {
        if (style->clipPath.url) free(style->clipPath.url);
        style->clipPath.url = _idFromUrl((const char*)(value + 3));
    }
}


static void _handleMaskAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    SvgStyleProperty* style = node->style;
    int len = strlen(value);
    if (len >= 3 && !strncmp(value, "url", 3)) {
        if (style->mask.url) free(style->mask.url);
        style->mask.url = _idFromUrl((const char*)(value + 3));
    }
}


static void _handleDisplayAttr(TVG_UNUSED SvgLoaderData* loader, SvgNode* node, const char* value)
{
    //TODO : The display attribute can have various values as well as "none".
    //       The default is "inline" which means visible and "none" means invisible.
    //       Depending on the type of node, additional functionality may be required.
    //       refer to https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/display
    if (!strcmp(value, "none")) node->display = false;
    else node->display = true;
}


typedef void (*styleMethod)(SvgLoaderData* loader, SvgNode* node, const char* value);

#define STYLE_DEF(Name, Name1, Flag) { #Name, sizeof(#Name), _handle##Name1##Attr, Flag }


static constexpr struct
{
    const char* tag;
    int sz;
    styleMethod tagHandler;
    SvgStyleFlags flag;
} styleTags[] = {
    STYLE_DEF(color, Color, SvgStyleFlags::Color),
    STYLE_DEF(fill, Fill, SvgStyleFlags::Fill),
    STYLE_DEF(fill-rule, FillRule, SvgStyleFlags::FillRule),
    STYLE_DEF(fill-opacity, FillOpacity, SvgStyleFlags::FillOpacity),
    STYLE_DEF(opacity, Opacity, SvgStyleFlags::Opacity),
    STYLE_DEF(stroke, Stroke, SvgStyleFlags::Stroke),
    STYLE_DEF(stroke-width, StrokeWidth, SvgStyleFlags::StrokeWidth),
    STYLE_DEF(stroke-linejoin, StrokeLineJoin, SvgStyleFlags::StrokeLineJoin),
    STYLE_DEF(stroke-linecap, StrokeLineCap, SvgStyleFlags::StrokeLineCap),
    STYLE_DEF(stroke-opacity, StrokeOpacity, SvgStyleFlags::StrokeOpacity),
    STYLE_DEF(stroke-dasharray, StrokeDashArray, SvgStyleFlags::StrokeDashArray),
    STYLE_DEF(transform, Transform, SvgStyleFlags::Transform),
    STYLE_DEF(clip-path, ClipPath, SvgStyleFlags::ClipPath),
    STYLE_DEF(mask, Mask, SvgStyleFlags::Mask),
    STYLE_DEF(display, Display, SvgStyleFlags::Display)
};


static bool _parseStyleAttr(void* data, const char* key, const char* value, bool style)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    int sz;
    if (!key || !value) return false;

    //Trim the white space
    key = _skipSpace(key, nullptr);
    value = _skipSpace(value, nullptr);

    sz = strlen(key);
    for (unsigned int i = 0; i < sizeof(styleTags) / sizeof(styleTags[0]); i++) {
        if (styleTags[i].sz - 1 == sz && !strncmp(styleTags[i].tag, key, sz)) {
            if (style) {
                styleTags[i].tagHandler(loader, node, value);
                node->style->flags = (SvgStyleFlags)((int)node->style->flags | (int)styleTags[i].flag);
            } else if (!((int)node->style->flags & (int)styleTags[i].flag)) {
                styleTags[i].tagHandler(loader, node, value);
            }
            return true;
        }
    }

    return false;
}


static bool _parseStyleAttr(void* data, const char* key, const char* value)
{
    return _parseStyleAttr(data, key, value, true);
}


/* parse g node
 * https://www.w3.org/TR/SVG/struct.html#Groups
 */
static bool _attrParseGNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;

    if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "transform")) {
        node->transform = _parseTransformationMatrix(value);
    } else if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


/* parse clipPath node
 * https://www.w3.org/TR/SVG/struct.html#Groups
 */
static bool _attrParseClipPathNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgCompositeNode* comp = &(node->node.comp);

    if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "transform")) {
        node->transform = _parseTransformationMatrix(value);
    } else if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else if (!strcmp(key, "clipPathUnits")) {
        if (!strcmp(value, "objectBoundingBox")) comp->userSpace = false;
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


static bool _attrParseMaskNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgCompositeNode* comp = &(node->node.comp);

    if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "transform")) {
        node->transform = _parseTransformationMatrix(value);
    } else if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else if (!strcmp(key, "maskContentUnits")) {
        if (!strcmp(value, "objectBoundingBox")) comp->userSpace = false;
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


static SvgNode* _createNode(SvgNode* parent, SvgNodeType type)
{
    SvgNode* node = (SvgNode*)calloc(1, sizeof(SvgNode));

    if (!node) return nullptr;

    //Default fill property
    node->style = (SvgStyleProperty*)calloc(1, sizeof(SvgStyleProperty));

    if (!node->style) {
        free(node);
        return nullptr;
    }

    //Update the default value of stroke and fill
    //https://www.w3.org/TR/SVGTiny12/painting.html#SpecifyingPaint
    node->style->fill.paint.none = false;
    //Default fill opacity is 1
    node->style->fill.opacity = 255;
    node->style->opacity = 255;
    //Default current color is not set
    node->style->fill.paint.curColor = false;
    node->style->curColorSet = false;
    //Default fill rule is nonzero
    node->style->fill.fillRule = FillRule::Winding;

    //Default stroke is none
    node->style->stroke.paint.none = true;
    //Default stroke opacity is 1
    node->style->stroke.opacity = 255;
    //Default stroke current color is not set
    node->style->stroke.paint.curColor = false;
    //Default stroke width is 1
    node->style->stroke.width = 1;
    //Default line cap is butt
    node->style->stroke.cap = StrokeCap::Butt;
    //Default line join is miter
    node->style->stroke.join = StrokeJoin::Miter;
    node->style->stroke.scale = 1.0;

    //Default display is true("inline").
    node->display = true;

    node->parent = parent;
    node->type = type;

    if (parent) parent->child.push(node);
    return node;
}


static SvgNode* _createDefsNode(TVG_UNUSED SvgLoaderData* loader, TVG_UNUSED SvgNode* parent, const char* buf, unsigned bufLength)
{
    if (loader->def && loader->doc->node.doc.defs) return loader->def;
    SvgNode* node = _createNode(nullptr, SvgNodeType::Defs);

    loader->def = node;
    loader->doc->node.doc.defs = node;
    return node;
}


static SvgNode* _createGNode(TVG_UNUSED SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::G);
    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParseGNode, loader);
    return loader->svgParse->node;
}


static SvgNode* _createSvgNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Doc);
    if (!loader->svgParse->node) return nullptr;
    SvgDocNode* doc = &(loader->svgParse->node->node.doc);

    loader->svgParse->global.w = 0;
    loader->svgParse->global.h = 0;

    doc->preserveAspect = true;
    simpleXmlParseAttributes(buf, bufLength, _attrParseSvgNode, loader);

    if (loader->svgParse->global.w == 0) {
        if (doc->w < FLT_EPSILON) loader->svgParse->global.w = 1;
        else loader->svgParse->global.w = (uint32_t)doc->w;
    }
    if (loader->svgParse->global.h == 0) {
        if (doc->h < FLT_EPSILON) loader->svgParse->global.h = 1;
        else loader->svgParse->global.h = (uint32_t)doc->h;
    }

    return loader->svgParse->node;
}


static SvgNode* _createMaskNode(SvgLoaderData* loader, SvgNode* parent, TVG_UNUSED const char* buf, TVG_UNUSED unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Mask);
    if (!loader->svgParse->node) return nullptr;

    loader->svgParse->node->node.comp.userSpace = true;

    simpleXmlParseAttributes(buf, bufLength, _attrParseMaskNode, loader);

    return loader->svgParse->node;
}


static SvgNode* _createClipPathNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::ClipPath);
    if (!loader->svgParse->node) return nullptr;

    loader->svgParse->node->display = false;
    loader->svgParse->node->node.comp.userSpace = true;

    simpleXmlParseAttributes(buf, bufLength, _attrParseClipPathNode, loader);

    return loader->svgParse->node;
}

static bool _attrParsePathNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgPathNode* path = &(node->node.path);

    if (!strcmp(key, "d")) {
        //Temporary: need to copy
        path->path = _copyId(value);
    } else if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


static SvgNode* _createPathNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Path);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParsePathNode, loader);

    return loader->svgParse->node;
}


static constexpr struct
{
    const char* tag;
    SvgParserLengthType type;
    int sz;
    size_t offset;
} circleTags[] = {
    {"cx", SvgParserLengthType::Horizontal, sizeof("cx"), offsetof(SvgCircleNode, cx)},
    {"cy", SvgParserLengthType::Vertical, sizeof("cy"), offsetof(SvgCircleNode, cy)},
    {"r", SvgParserLengthType::Other, sizeof("r"), offsetof(SvgCircleNode, r)}
};


/* parse the attributes for a circle element.
 * https://www.w3.org/TR/SVG/shapes.html#CircleElement
 */
static bool _attrParseCircleNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgCircleNode* circle = &(node->node.circle);
    unsigned char* array;
    int sz = strlen(key);

    array = (unsigned char*)circle;
    for (unsigned int i = 0; i < sizeof(circleTags) / sizeof(circleTags[0]); i++) {
        if (circleTags[i].sz - 1 == sz && !strncmp(circleTags[i].tag, key, sz)) {
            *((float*)(array + circleTags[i].offset)) = _toFloat(loader->svgParse, value, circleTags[i].type);
            return true;
        }
    }

    if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


static SvgNode* _createCircleNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Circle);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParseCircleNode, loader);
    return loader->svgParse->node;
}


static constexpr struct
{
    const char* tag;
    SvgParserLengthType type;
    int sz;
    size_t offset;
} ellipseTags[] = {
    {"cx", SvgParserLengthType::Horizontal, sizeof("cx"), offsetof(SvgEllipseNode, cx)},
    {"cy", SvgParserLengthType::Vertical, sizeof("cy"), offsetof(SvgEllipseNode, cy)},
    {"rx", SvgParserLengthType::Horizontal, sizeof("rx"), offsetof(SvgEllipseNode, rx)},
    {"ry", SvgParserLengthType::Vertical, sizeof("ry"), offsetof(SvgEllipseNode, ry)}
};


/* parse the attributes for an ellipse element.
 * https://www.w3.org/TR/SVG/shapes.html#EllipseElement
 */
static bool _attrParseEllipseNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgEllipseNode* ellipse = &(node->node.ellipse);
    unsigned char* array;
    int sz = strlen(key);

    array = (unsigned char*)ellipse;
    for (unsigned int i = 0; i < sizeof(ellipseTags) / sizeof(ellipseTags[0]); i++) {
        if (ellipseTags[i].sz - 1 == sz && !strncmp(ellipseTags[i].tag, key, sz)) {
            *((float*)(array + ellipseTags[i].offset)) = _toFloat(loader->svgParse, value, ellipseTags[i].type);
            return true;
        }
    }

    if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


static SvgNode* _createEllipseNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Ellipse);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParseEllipseNode, loader);
    return loader->svgParse->node;
}


static bool _attrParsePolygonPoints(const char* str, float** points, int* ptCount)
{
    float tmp[50];
    int tmpCount = 0;
    int count = 0;
    float num;
    float *pointArray = nullptr, *tmpArray;

    while (_parseNumber(&str, &num)) {
        tmp[tmpCount++] = num;
        if (tmpCount == 50) {
            tmpArray = (float*)realloc(pointArray, (count + tmpCount) * sizeof(float));
            if (!tmpArray) goto error_alloc;
            pointArray = tmpArray;
            memcpy(&pointArray[count], tmp, tmpCount * sizeof(float));
            count += tmpCount;
            tmpCount = 0;
        }
    }

    if (tmpCount > 0) {
        tmpArray = (float*)realloc(pointArray, (count + tmpCount) * sizeof(float));
        if (!tmpArray) goto error_alloc;
        pointArray = tmpArray;
        memcpy(&pointArray[count], tmp, tmpCount * sizeof(float));
        count += tmpCount;
    }
    *ptCount = count;
    *points = pointArray;
    return true;

error_alloc:
    return false;
}


/* parse the attributes for a polygon element.
 * https://www.w3.org/TR/SVG/shapes.html#PolylineElement
 */
static bool _attrParsePolygonNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgPolygonNode* polygon = nullptr;

    if (node->type == SvgNodeType::Polygon) polygon = &(node->node.polygon);
    else polygon = &(node->node.polyline);

    if (!strcmp(key, "points")) {
        return _attrParsePolygonPoints(value, &polygon->points, &polygon->pointsCount);
    } else if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


static SvgNode* _createPolygonNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Polygon);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParsePolygonNode, loader);
    return loader->svgParse->node;
}


static SvgNode* _createPolylineNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Polyline);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParsePolygonNode, loader);
    return loader->svgParse->node;
}

static constexpr struct
{
    const char* tag;
    SvgParserLengthType type;
    int sz;
    size_t offset;
} rectTags[] = {
    {"x", SvgParserLengthType::Horizontal, sizeof("x"), offsetof(SvgRectNode, x)},
    {"y", SvgParserLengthType::Vertical, sizeof("y"), offsetof(SvgRectNode, y)},
    {"width", SvgParserLengthType::Horizontal, sizeof("width"), offsetof(SvgRectNode, w)},
    {"height", SvgParserLengthType::Vertical, sizeof("height"), offsetof(SvgRectNode, h)},
    {"rx", SvgParserLengthType::Horizontal, sizeof("rx"), offsetof(SvgRectNode, rx)},
    {"ry", SvgParserLengthType::Vertical, sizeof("ry"), offsetof(SvgRectNode, ry)}
};


/* parse the attributes for a rect element.
 * https://www.w3.org/TR/SVG/shapes.html#RectElement
 */
static bool _attrParseRectNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgRectNode* rect = &(node->node.rect);
    unsigned char* array;
    bool ret = true;
    int sz = strlen(key);

    array = (unsigned char*)rect;
    for (unsigned int i = 0; i < sizeof(rectTags) / sizeof(rectTags[0]); i++) {
        if (rectTags[i].sz - 1 == sz && !strncmp(rectTags[i].tag, key, sz)) {
            *((float*)(array + rectTags[i].offset)) = _toFloat(loader->svgParse, value, rectTags[i].type);

            //Case if only rx or ry is declared
            if (!strncmp(rectTags[i].tag, "rx", sz)) rect->hasRx = true;
            if (!strncmp(rectTags[i].tag, "ry", sz)) rect->hasRy = true;

            if ((rect->rx >= FLT_EPSILON) && (rect->ry < FLT_EPSILON) && rect->hasRx && !rect->hasRy) rect->ry = rect->rx;
            if ((rect->ry >= FLT_EPSILON) && (rect->rx < FLT_EPSILON) && !rect->hasRx && rect->hasRy) rect->rx = rect->ry;
            return ret;
        }
    }

    if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else if (!strcmp(key, "style")) {
        ret = simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else {
        ret = _parseStyleAttr(loader, key, value, false);
    }

    return ret;
}


static SvgNode* _createRectNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Rect);

    if (!loader->svgParse->node) return nullptr;

    loader->svgParse->node->node.rect.hasRx = loader->svgParse->node->node.rect.hasRy = false;

    simpleXmlParseAttributes(buf, bufLength, _attrParseRectNode, loader);
    return loader->svgParse->node;
}


static constexpr struct
{
    const char* tag;
    SvgParserLengthType type;
    int sz;
    size_t offset;
} lineTags[] = {
    {"x1", SvgParserLengthType::Horizontal, sizeof("x1"), offsetof(SvgLineNode, x1)},
    {"y1", SvgParserLengthType::Vertical, sizeof("y1"), offsetof(SvgLineNode, y1)},
    {"x2", SvgParserLengthType::Horizontal, sizeof("x2"), offsetof(SvgLineNode, x2)},
    {"y2", SvgParserLengthType::Vertical, sizeof("y2"), offsetof(SvgLineNode, y2)}
};


/* parse the attributes for a line element.
 * https://www.w3.org/TR/SVG/shapes.html#LineElement
 */
static bool _attrParseLineNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgLineNode* line = &(node->node.line);
    unsigned char* array;
    int sz = strlen(key);

    array = (unsigned char*)line;
    for (unsigned int i = 0; i < sizeof(lineTags) / sizeof(lineTags[0]); i++) {
        if (lineTags[i].sz - 1 == sz && !strncmp(lineTags[i].tag, key, sz)) {
            *((float*)(array + lineTags[i].offset)) = _toFloat(loader->svgParse, value, lineTags[i].type);
            return true;
        }
    }

    if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else {
        return _parseStyleAttr(loader, key, value, false);
    }
    return true;
}


static SvgNode* _createLineNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Line);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParseLineNode, loader);
    return loader->svgParse->node;
}


static char* _idFromHref(const char* href)
{
    href = _skipSpace(href, nullptr);
    if ((*href) == '#') href++;
    return strdup(href);
}


static constexpr struct
{
    const char* tag;
    SvgParserLengthType type;
    int sz;
    size_t offset;
} imageTags[] = {
    {"x", SvgParserLengthType::Horizontal, sizeof("x"), offsetof(SvgRectNode, x)},
    {"y", SvgParserLengthType::Vertical, sizeof("y"), offsetof(SvgRectNode, y)},
    {"width", SvgParserLengthType::Horizontal, sizeof("width"), offsetof(SvgRectNode, w)},
    {"height", SvgParserLengthType::Vertical, sizeof("height"), offsetof(SvgRectNode, h)},
};


/* parse the attributes for a image element.
 * https://www.w3.org/TR/SVG/embedded.html#ImageElement
 */
static bool _attrParseImageNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode* node = loader->svgParse->node;
    SvgImageNode* image = &(node->node.image);
    unsigned char* array;
    int sz = strlen(key);

    array = (unsigned char*)image;
    for (unsigned int i = 0; i < sizeof(imageTags) / sizeof(imageTags[0]); i++) {
        if (imageTags[i].sz - 1 == sz && !strncmp(imageTags[i].tag, key, sz)) {
            *((float*)(array + imageTags[i].offset)) = _toFloat(loader->svgParse, value, imageTags[i].type);
            return true;
        }
    }

    if (!strcmp(key, "href") || !strcmp(key, "xlink:href")) {
        image->href = _idFromHref(value);
    } else if (!strcmp(key, "id")) {
        if (node->id && value) free(node->id);
        node->id = _copyId(value);
    } else if (!strcmp(key, "style")) {
        return simpleXmlParseW3CAttribute(value, _parseStyleAttr, loader);
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else {
        return _parseStyleAttr(loader, key, value);
    }
    return true;
}


static SvgNode* _createImageNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Image);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParseImageNode, loader);
    return loader->svgParse->node;
}


static SvgNode* _getDefsNode(SvgNode* node)
{
    if (!node) return nullptr;

    while (node->parent != nullptr) {
        node = node->parent;
    }

    if (node->type == SvgNodeType::Doc) return node->node.doc.defs;

    return nullptr;
}


static SvgNode* _findChildById(const SvgNode* node, const char* id)
{
    if (!node) return nullptr;

    auto child = node->child.data;
    for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
        if (((*child)->id) && !strcmp((*child)->id, id)) return (*child);
    }
    return nullptr;
}

static SvgNode* _findNodeById(SvgNode *node, const char* id)
{
    SvgNode* result = nullptr;
    if (node->id && !strcmp(node->id, id)) return node;

    if (node->child.count > 0) {
        auto child = node->child.data;
        for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
            result = _findNodeById(*child, id);
            if (result) break;
        }
    }
    return result;
}

static void _cloneGradStops(Array<Fill::ColorStop>& dst, const Array<Fill::ColorStop>& src)
{
    for (uint32_t i = 0; i < src.count; ++i) {
        dst.push(src.data[i]);
    }
}


static SvgStyleGradient* _cloneGradient(SvgStyleGradient* from)
{
    if (!from) return nullptr;

    auto grad = (SvgStyleGradient*)(calloc(1, sizeof(SvgStyleGradient)));
    if (!grad) return nullptr;

    grad->type = from->type;
    grad->id = from->id ? _copyId(from->id) : nullptr;
    grad->ref = from->ref ? _copyId(from->ref) : nullptr;
    grad->spread = from->spread;
    grad->userSpace = from->userSpace;

    if (from->transform) {
        grad->transform = (Matrix*)calloc(1, sizeof(Matrix));
        if (grad->transform) memcpy(grad->transform, from->transform, sizeof(Matrix));
    }

    if (grad->type == SvgGradientType::Linear) {
        grad->linear = (SvgLinearGradient*)calloc(1, sizeof(SvgLinearGradient));
        if (!grad->linear) goto error_grad_alloc;
        memcpy(grad->linear, from->linear, sizeof(SvgLinearGradient));
    } else if (grad->type == SvgGradientType::Radial) {
        grad->radial = (SvgRadialGradient*)calloc(1, sizeof(SvgRadialGradient));
        if (!grad->radial) goto error_grad_alloc;
        memcpy(grad->radial, from->radial, sizeof(SvgRadialGradient));
    }

    _cloneGradStops(grad->stops, from->stops);

    return grad;

error_grad_alloc:
    if (grad) {
        grad->clear();
        free(grad);
    }
    return nullptr;
}


static void _copyAttr(SvgNode* to, const SvgNode* from)
{
    //Copy matrix attribute
    if (from->transform) {
        to->transform = (Matrix*)malloc(sizeof(Matrix));
        if (to->transform) *to->transform = *from->transform;
    }
    //Copy style attribute
    *to->style = *from->style;
    if (from->style->fill.paint.url) to->style->fill.paint.url = strdup(from->style->fill.paint.url);
    if (from->style->stroke.paint.url) to->style->stroke.paint.url = strdup(from->style->stroke.paint.url);
    if (from->style->clipPath.url) to->style->clipPath.url = strdup(from->style->clipPath.url);
    if (from->style->mask.url) to->style->mask.url = strdup(from->style->mask.url);

    //Copy node attribute
    switch (from->type) {
        case SvgNodeType::Circle: {
            to->node.circle.cx = from->node.circle.cx;
            to->node.circle.cy = from->node.circle.cy;
            to->node.circle.r = from->node.circle.r;
            break;
        }
        case SvgNodeType::Ellipse: {
            to->node.ellipse.cx = from->node.ellipse.cx;
            to->node.ellipse.cy = from->node.ellipse.cy;
            to->node.ellipse.rx = from->node.ellipse.rx;
            to->node.ellipse.ry = from->node.ellipse.ry;
            break;
        }
        case SvgNodeType::Rect: {
            to->node.rect.x = from->node.rect.x;
            to->node.rect.y = from->node.rect.y;
            to->node.rect.w = from->node.rect.w;
            to->node.rect.h = from->node.rect.h;
            to->node.rect.rx = from->node.rect.rx;
            to->node.rect.ry = from->node.rect.ry;
            to->node.rect.hasRx = from->node.rect.hasRx;
            to->node.rect.hasRy = from->node.rect.hasRy;
            break;
        }
        case SvgNodeType::Line: {
            to->node.line.x1 = from->node.line.x1;
            to->node.line.y1 = from->node.line.y1;
            to->node.line.x2 = from->node.line.x2;
            to->node.line.y2 = from->node.line.y2;
            break;
        }
        case SvgNodeType::Path: {
            if (from->node.path.path) to->node.path.path = strdup(from->node.path.path);
            break;
        }
        case SvgNodeType::Polygon: {
            to->node.polygon.pointsCount = from->node.polygon.pointsCount;
            to->node.polygon.points = (float*)malloc(to->node.polygon.pointsCount * sizeof(float));
            memcpy(to->node.polygon.points, from->node.polygon.points, to->node.polygon.pointsCount * sizeof(float));
            break;
        }
        case SvgNodeType::Polyline: {
            to->node.polyline.pointsCount = from->node.polyline.pointsCount;
            to->node.polyline.points = (float*)malloc(to->node.polyline.pointsCount * sizeof(float));
            memcpy(to->node.polyline.points, from->node.polyline.points, to->node.polyline.pointsCount * sizeof(float));
            break;
        }
        case SvgNodeType::Image: {
            to->node.image.x = from->node.image.x;
            to->node.image.y = from->node.image.y;
            to->node.image.w = from->node.image.w;
            to->node.image.h = from->node.image.h;
            if (from->node.image.href) to->node.image.href = strdup(from->node.image.href);
            break;
        }
        default: {
            break;
        }
    }
}


static void _cloneNode(SvgNode* from, SvgNode* parent)
{
    SvgNode* newNode;
    if (!from || !parent) return;

    newNode = _createNode(parent, from->type);

    if (!newNode) return;

    _copyAttr(newNode, from);

    auto child = from->child.data;
    for (uint32_t i = 0; i < from->child.count; ++i, ++child) {
        _cloneNode(*child, newNode);
    }
}


static void _postponeCloneNode(SvgLoaderData* loader, SvgNode *node, char* id) {
    loader->cloneNodes.push({node, id});
}


static void _clonePostponedNodes(Array<SvgNodeIdPair>* cloneNodes) {
    for (uint32_t i = 0; i < cloneNodes->count; ++i) {
        auto nodeIdPair = cloneNodes->data[i];
        auto defs = _getDefsNode(nodeIdPair.node);
        auto nodeFrom = _findChildById(defs, nodeIdPair.id);
        _cloneNode(nodeFrom, nodeIdPair.node);
        free(nodeIdPair.id);
    }
}


static constexpr struct
{
    const char* tag;
    SvgParserLengthType type;
    int sz;
    size_t offset;
} useTags[] = {
    {"x", SvgParserLengthType::Horizontal, sizeof("x"), offsetof(SvgRectNode, x)},
    {"y", SvgParserLengthType::Vertical, sizeof("y"), offsetof(SvgRectNode, y)},
    {"width", SvgParserLengthType::Horizontal, sizeof("width"), offsetof(SvgRectNode, w)},
    {"height", SvgParserLengthType::Vertical, sizeof("height"), offsetof(SvgRectNode, h)}
};


static bool _attrParseUseNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgNode *defs, *nodeFrom, *node = loader->svgParse->node;
    char* id;

    SvgUseNode* use = &(node->node.use);
    int sz = strlen(key);
    unsigned char* array = (unsigned char*)use;
    for (unsigned int i = 0; i < sizeof(useTags) / sizeof(useTags[0]); i++) {
        if (useTags[i].sz - 1 == sz && !strncmp(useTags[i].tag, key, sz)) {
            *((float*)(array + useTags[i].offset)) = _toFloat(loader->svgParse, value, useTags[i].type);
            return true;
        }
    }

    if (!strcmp(key, "href") || !strcmp(key, "xlink:href")) {
        id = _idFromHref(value);
        defs = _getDefsNode(node);
        nodeFrom = _findChildById(defs, id);
        if (nodeFrom) {
            _cloneNode(nodeFrom, node);
            free(id);
        } else {
            //some svg export software include <defs> element at the end of the file
            //if so the 'from' element won't be found now and we have to repeat finding
            //after the whole file is parsed
            _postponeCloneNode(loader, node, id);
        }
    } else if (!strcmp(key, "clip-path")) {
        _handleClipPathAttr(loader, node, value);
    } else if (!strcmp(key, "mask")) {
        _handleMaskAttr(loader, node, value);
    } else {
        return _attrParseGNode(data, key, value);
    }
    return true;
}


static SvgNode* _createUseNode(SvgLoaderData* loader, SvgNode* parent, const char* buf, unsigned bufLength)
{
    loader->svgParse->node = _createNode(parent, SvgNodeType::Use);

    if (!loader->svgParse->node) return nullptr;

    simpleXmlParseAttributes(buf, bufLength, _attrParseUseNode, loader);
    return loader->svgParse->node;
}

//TODO: Implement 'text' primitive
static constexpr struct
{
    const char* tag;
    int sz;
    FactoryMethod tagHandler;
} graphicsTags[] = {
    {"use", sizeof("use"), _createUseNode},
    {"circle", sizeof("circle"), _createCircleNode},
    {"ellipse", sizeof("ellipse"), _createEllipseNode},
    {"path", sizeof("path"), _createPathNode},
    {"polygon", sizeof("polygon"), _createPolygonNode},
    {"rect", sizeof("rect"), _createRectNode},
    {"polyline", sizeof("polyline"), _createPolylineNode},
    {"line", sizeof("line"), _createLineNode},
    {"image", sizeof("image"), _createImageNode}
};


static constexpr struct
{
    const char* tag;
    int sz;
    FactoryMethod tagHandler;
} groupTags[] = {
    {"defs", sizeof("defs"), _createDefsNode},
    {"g", sizeof("g"), _createGNode},
    {"svg", sizeof("svg"), _createSvgNode},
    {"mask", sizeof("mask"), _createMaskNode},
    {"clipPath", sizeof("clipPath"), _createClipPathNode}
};


#define FIND_FACTORY(Short_Name, Tags_Array)                                           \
    static FactoryMethod                                                               \
        _find##Short_Name##Factory(const char* name)                                   \
    {                                                                                  \
        unsigned int i;                                                                \
        int sz = strlen(name);                                                         \
                                                                                       \
        for (i = 0; i < sizeof(Tags_Array) / sizeof(Tags_Array[0]); i++) {             \
            if (Tags_Array[i].sz - 1 == sz && !strncmp(Tags_Array[i].tag, name, sz)) { \
                return Tags_Array[i].tagHandler;                                       \
            }                                                                          \
        }                                                                              \
        return nullptr;                                                                \
    }

FIND_FACTORY(Group, groupTags)
FIND_FACTORY(Graphics, graphicsTags)


FillSpread _parseSpreadValue(const char* value)
{
    auto spread = FillSpread::Pad;

    if (!strcmp(value, "reflect")) {
        spread = FillSpread::Reflect;
    } else if (!strcmp(value, "repeat")) {
        spread = FillSpread::Repeat;
    }

    return spread;
}


static void _handleRadialCxAttr(SvgLoaderData* loader, SvgRadialGradient* radial, const char* value)
{
    radial->cx = _gradientToFloat(loader->svgParse, value, radial->isCxPercentage);
    if (!loader->svgParse->gradient.parsedFx) {
        radial->fx = radial->cx;
        radial->isFxPercentage = radial->isCxPercentage;
    }
}


static void _handleRadialCyAttr(SvgLoaderData* loader, SvgRadialGradient* radial, const char* value)
{
    radial->cy = _gradientToFloat(loader->svgParse, value, radial->isCyPercentage);
    if (!loader->svgParse->gradient.parsedFy) {
        radial->fy = radial->cy;
        radial->isFyPercentage = radial->isCyPercentage;
    }
}


static void _handleRadialFxAttr(SvgLoaderData* loader, SvgRadialGradient* radial, const char* value)
{
    radial->fx = _gradientToFloat(loader->svgParse, value, radial->isFxPercentage);
    loader->svgParse->gradient.parsedFx = true;
}


static void _handleRadialFyAttr(SvgLoaderData* loader, SvgRadialGradient* radial, const char* value)
{
    radial->fy = _gradientToFloat(loader->svgParse, value, radial->isFyPercentage);
    loader->svgParse->gradient.parsedFy = true;
}


static void _handleRadialRAttr(SvgLoaderData* loader, SvgRadialGradient* radial, const char* value)
{
    radial->r = _gradientToFloat(loader->svgParse, value, radial->isRPercentage);
}


static void _recalcRadialCxAttr(SvgLoaderData* loader, SvgRadialGradient* radial, bool userSpace)
{
    if (userSpace && !radial->isCxPercentage) radial->cx = radial->cx / loader->svgParse->global.w;
}


static void _recalcRadialCyAttr(SvgLoaderData* loader, SvgRadialGradient* radial, bool userSpace)
{
    if (userSpace && !radial->isCyPercentage) radial->cy = radial->cy / loader->svgParse->global.h;
}


static void _recalcRadialFxAttr(SvgLoaderData* loader, SvgRadialGradient* radial, bool userSpace)
{
    if (userSpace && !radial->isFxPercentage) radial->fx = radial->fx / loader->svgParse->global.w;
}


static void _recalcRadialFyAttr(SvgLoaderData* loader, SvgRadialGradient* radial, bool userSpace)
{
    if (userSpace && !radial->isFyPercentage) radial->fy = radial->fy / loader->svgParse->global.h;
}


static void _recalcRadialRAttr(SvgLoaderData* loader, SvgRadialGradient* radial, bool userSpace)
{
    // scaling factor based on the Units paragraph from : https://www.w3.org/TR/2015/WD-SVG2-20150915/coords.html
    if (userSpace && !radial->isRPercentage) radial->r = radial->r / (sqrtf(pow(loader->svgParse->global.h, 2) + pow(loader->svgParse->global.w, 2)) / sqrtf(2.0));
}


typedef void (*radialMethod)(SvgLoaderData* loader, SvgRadialGradient* radial, const char* value);
typedef void (*radialMethodRecalc)(SvgLoaderData* loader, SvgRadialGradient* radial, bool userSpace);


#define RADIAL_DEF(Name, Name1)                                                          \
    {                                                                                    \
#Name, sizeof(#Name), _handleRadial##Name1##Attr, _recalcRadial##Name1##Attr             \
    }


static constexpr struct
{
    const char* tag;
    int sz;
    radialMethod tagHandler;
    radialMethodRecalc tagRecalc;
} radialTags[] = {
    RADIAL_DEF(cx, Cx),
    RADIAL_DEF(cy, Cy),
    RADIAL_DEF(fx, Fx),
    RADIAL_DEF(fy, Fy),
    RADIAL_DEF(r, R)
};


static bool _attrParseRadialGradientNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgStyleGradient* grad = loader->svgParse->styleGrad;
    SvgRadialGradient* radial = grad->radial;
    int sz = strlen(key);

    for (unsigned int i = 0; i < sizeof(radialTags) / sizeof(radialTags[0]); i++) {
        if (radialTags[i].sz - 1 == sz && !strncmp(radialTags[i].tag, key, sz)) {
            radialTags[i].tagHandler(loader, radial, value);
            return true;
        }
    }

    if (!strcmp(key, "id")) {
        grad->id = _copyId(value);
    } else if (!strcmp(key, "spreadMethod")) {
        grad->spread = _parseSpreadValue(value);
    } else if (!strcmp(key, "href") || !strcmp(key, "xlink:href")) {
        grad->ref = _idFromHref(value);
    } else if (!strcmp(key, "gradientUnits") && !strcmp(value, "userSpaceOnUse")) {
        grad->userSpace = true;
    } else if (!strcmp(key, "gradientTransform")) {
        grad->transform = _parseTransformationMatrix(value);
    } else {
        return false;
    }

    return true;
}


static SvgStyleGradient* _createRadialGradient(SvgLoaderData* loader, const char* buf, unsigned bufLength)
{
    auto grad = (SvgStyleGradient*)(calloc(1, sizeof(SvgStyleGradient)));
    loader->svgParse->styleGrad = grad;

    grad->type = SvgGradientType::Radial;
    grad->userSpace = false;
    grad->radial = (SvgRadialGradient*)calloc(1, sizeof(SvgRadialGradient));
    if (!grad->radial) {
        grad->clear();
        free(grad);
        return nullptr;
    }
    /**
    * Default values of gradient transformed into global percentage
    */
    grad->radial->cx = 0.5f;
    grad->radial->cy = 0.5f;
    grad->radial->fx = 0.5f;
    grad->radial->fy = 0.5f;
    grad->radial->r = 0.5f;
    grad->radial->isCxPercentage = true;
    grad->radial->isCyPercentage = true;
    grad->radial->isFxPercentage = true;
    grad->radial->isFyPercentage = true;
    grad->radial->isRPercentage = true;

    loader->svgParse->gradient.parsedFx = false;
    loader->svgParse->gradient.parsedFy = false;
    simpleXmlParseAttributes(buf, bufLength,
        _attrParseRadialGradientNode, loader);

    for (unsigned int i = 0; i < sizeof(radialTags) / sizeof(radialTags[0]); i++) {
        radialTags[i].tagRecalc(loader, grad->radial, grad->userSpace);
    }

    return loader->svgParse->styleGrad;
}


static bool _attrParseStopsStyle(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    auto stop = &loader->svgParse->gradStop;

    if (!strcmp(key, "stop-opacity")) {
        stop->a = _toOpacity(value);
        loader->svgParse->flags = (SvgStopStyleFlags)((int)loader->svgParse->flags | (int)SvgStopStyleFlags::StopOpacity);
    } else if (!strcmp(key, "stop-color")) {
        _toColor(value, &stop->r, &stop->g, &stop->b, nullptr);
        loader->svgParse->flags = (SvgStopStyleFlags)((int)loader->svgParse->flags | (int)SvgStopStyleFlags::StopColor);
    } else {
        return false;
    }

    return true;
}


static bool _attrParseStops(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    auto stop = &loader->svgParse->gradStop;

    if (!strcmp(key, "offset")) {
        stop->offset = _toOffset(value);
    } else if (!strcmp(key, "stop-opacity")) {
        if (!((int)loader->svgParse->flags & (int)SvgStopStyleFlags::StopOpacity)) {
            stop->a = _toOpacity(value);
        }
    } else if (!strcmp(key, "stop-color")) {
        if (!((int)loader->svgParse->flags & (int)SvgStopStyleFlags::StopColor)) {
            _toColor(value, &stop->r, &stop->g, &stop->b, nullptr);
        }
    } else if (!strcmp(key, "style")) {
        simpleXmlParseW3CAttribute(value, _attrParseStopsStyle, data);
    } else {
        return false;
    }

    return true;
}


static void _handleLinearX1Attr(SvgLoaderData* loader, SvgLinearGradient* linear, const char* value)
{
    linear->x1 = _gradientToFloat(loader->svgParse, value, linear->isX1Percentage);
}


static void _handleLinearY1Attr(SvgLoaderData* loader, SvgLinearGradient* linear, const char* value)
{
    linear->y1 = _gradientToFloat(loader->svgParse, value, linear->isY1Percentage);
}


static void _handleLinearX2Attr(SvgLoaderData* loader, SvgLinearGradient* linear, const char* value)
{
    linear->x2 = _gradientToFloat(loader->svgParse, value, linear->isX2Percentage);
}


static void _handleLinearY2Attr(SvgLoaderData* loader, SvgLinearGradient* linear, const char* value)
{
    linear->y2 = _gradientToFloat(loader->svgParse, value, linear->isY2Percentage);
}


static void _recalcLinearX1Attr(SvgLoaderData* loader, SvgLinearGradient* linear, bool userSpace)
{
    if (userSpace && !linear->isX1Percentage) linear->x1 = linear->x1 / loader->svgParse->global.w;
}


static void _recalcLinearY1Attr(SvgLoaderData* loader, SvgLinearGradient* linear, bool userSpace)
{
    if (userSpace && !linear->isY1Percentage) linear->y1 = linear->y1 / loader->svgParse->global.h;
}


static void _recalcLinearX2Attr(SvgLoaderData* loader, SvgLinearGradient* linear, bool userSpace)
{
    if (userSpace && !linear->isX2Percentage) linear->x2 = linear->x2 / loader->svgParse->global.w;
}


static void _recalcLinearY2Attr(SvgLoaderData* loader, SvgLinearGradient* linear, bool userSpace)
{
    if (userSpace && !linear->isY2Percentage) linear->y2 = linear->y2 / loader->svgParse->global.h;
}


typedef void (*Linear_Method)(SvgLoaderData* loader, SvgLinearGradient* linear, const char* value);
typedef void (*Linear_Method_Recalc)(SvgLoaderData* loader, SvgLinearGradient* linear, bool userSpace);


#define LINEAR_DEF(Name, Name1)                                                          \
    {                                                                                    \
#Name, sizeof(#Name), _handleLinear##Name1##Attr, _recalcLinear##Name1##Attr \
    }


static constexpr struct
{
    const char* tag;
    int sz;
    Linear_Method tagHandler;
    Linear_Method_Recalc tagRecalc;
} linear_tags[] = {
    LINEAR_DEF(x1, X1),
    LINEAR_DEF(y1, Y1),
    LINEAR_DEF(x2, X2),
    LINEAR_DEF(y2, Y2)
};


static bool _attrParseLinearGradientNode(void* data, const char* key, const char* value)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    SvgStyleGradient* grad = loader->svgParse->styleGrad;
    SvgLinearGradient* linear = grad->linear;
    int sz = strlen(key);

    for (unsigned int i = 0; i < sizeof(linear_tags) / sizeof(linear_tags[0]); i++) {
        if (linear_tags[i].sz - 1 == sz && !strncmp(linear_tags[i].tag, key, sz)) {
            linear_tags[i].tagHandler(loader, linear, value);
            return true;
        }
    }

    if (!strcmp(key, "id")) {
        grad->id = _copyId(value);
    } else if (!strcmp(key, "spreadMethod")) {
        grad->spread = _parseSpreadValue(value);
    } else if (!strcmp(key, "href") || !strcmp(key, "xlink:href")) {
        grad->ref = _idFromHref(value);
    } else if (!strcmp(key, "gradientUnits") && !strcmp(value, "userSpaceOnUse")) {
        grad->userSpace = true;
    } else if (!strcmp(key, "gradientTransform")) {
        grad->transform = _parseTransformationMatrix(value);
    } else {
        return false;
    }

    return true;
}


static SvgStyleGradient* _createLinearGradient(SvgLoaderData* loader, const char* buf, unsigned bufLength)
{
    auto grad = (SvgStyleGradient*)(calloc(1, sizeof(SvgStyleGradient)));
    loader->svgParse->styleGrad = grad;

    grad->type = SvgGradientType::Linear;
    grad->userSpace = false;
    grad->linear = (SvgLinearGradient*)calloc(1, sizeof(SvgLinearGradient));
    if (!grad->linear) {
        grad->clear();
        free(grad);
        return nullptr;
    }
    /**
    * Default value of x2 is 100% - transformed to the global percentage
    */
    grad->linear->x2 = 1.0f;
    grad->linear->isX2Percentage = true;

    simpleXmlParseAttributes(buf, bufLength, _attrParseLinearGradientNode, loader);

    for (unsigned int i = 0; i < sizeof(linear_tags) / sizeof(linear_tags[0]); i++) {
        linear_tags[i].tagRecalc(loader, grad->linear, grad->userSpace);
    }

    return loader->svgParse->styleGrad;
}


#define GRADIENT_DEF(Name, Name1)            \
    {                                        \
#Name, sizeof(#Name), _create##Name1         \
    }


/**
 * In the case when the gradients lengths are given as numbers (not percentages)
 * in the current user coordinate system, they are recalculated into percentages
 * related to the canvas width and height.
 */
static constexpr struct
{
    const char* tag;
    int sz;
    GradientFactoryMethod tagHandler;
} gradientTags[] = {
    GRADIENT_DEF(linearGradient, LinearGradient),
    GRADIENT_DEF(radialGradient, RadialGradient)
};


static GradientFactoryMethod _findGradientFactory(const char* name)
{
    int sz = strlen(name);

    for (unsigned int i = 0; i < sizeof(gradientTags) / sizeof(gradientTags[0]); i++) {
        if (gradientTags[i].sz - 1 == sz && !strncmp(gradientTags[i].tag, name, sz)) {
            return gradientTags[i].tagHandler;
        }
    }
    return nullptr;
}


static constexpr struct
{
    const char* tag;
    size_t sz;
} popArray[] = {
    {"g", sizeof("g")},
    {"svg", sizeof("svg")},
    {"defs", sizeof("defs")},
    {"mask", sizeof("mask")},
    {"clipPath", sizeof("clipPath")}
};


static void _svgLoaderParerXmlClose(SvgLoaderData* loader, const char* content)
{
    content = _skipSpace(content, nullptr);

    for (unsigned int i = 0; i < sizeof(popArray) / sizeof(popArray[0]); i++) {
        if (!strncmp(content, popArray[i].tag, popArray[i].sz - 1)) {
            loader->stack.pop();
            break;
        }
    }

    loader->level--;
}


static void _svgLoaderParserXmlOpen(SvgLoaderData* loader, const char* content, unsigned int length, bool empty)
{
    const char* attrs = nullptr;
    int attrsLength = 0;
    int sz = length;
    char tagName[20] = "";
    FactoryMethod method;
    GradientFactoryMethod gradientMethod;
    SvgNode *node = nullptr, *parent = nullptr;
    loader->level++;
    attrs = simpleXmlFindAttributesTag(content, length);

    if (!attrs) {
        //Parse the empty tag
        attrs = content;
        while ((attrs != nullptr) && *attrs != '>') attrs++;
    }

    if (attrs) {
        //Find out the tag name starting from content till sz length
        sz = attrs - content;
        while ((sz > 0) && (isspace(content[sz - 1]))) sz--;
        if ((unsigned)sz >= sizeof(tagName)) return;
        strncpy(tagName, content, sz);
        tagName[sz] = '\0';
        attrsLength = length - sz;
    }

    if ((method = _findGroupFactory(tagName))) {
        //Group
        if (!loader->doc) {
            if (strcmp(tagName, "svg")) return; //Not a valid svg document
            node = method(loader, nullptr, attrs, attrsLength);
            loader->doc = node;
        } else {
            if (!strcmp(tagName, "svg")) return; //Already loaded <svg>(SvgNodeType::Doc) tag
            if (loader->stack.count > 0) parent = loader->stack.data[loader->stack.count - 1];
            else parent = loader->doc;
            node = method(loader, parent, attrs, attrsLength);
        }

        if (!node) return;
        if (node->type != SvgNodeType::Defs || !empty) {
            loader->stack.push(node);
        }
    } else if ((method = _findGraphicsFactory(tagName))) {
        if (loader->stack.count > 0) parent = loader->stack.data[loader->stack.count - 1];
        else parent = loader->doc;
        node = method(loader, parent, attrs, attrsLength);
    } else if ((gradientMethod = _findGradientFactory(tagName))) {
        SvgStyleGradient* gradient;
        gradient = gradientMethod(loader, attrs, attrsLength);
        //FIXME: The current parsing structure does not distinguish end tags.
        //       There is no way to know if the currently parsed gradient is in defs.
        //       If a gradient is declared outside of defs after defs is set, it is included in the gradients of defs.
        //       But finally, the loader has a gradient style list regardless of defs.
        //       This is only to support this when multiple gradients are declared, even if no defs are declared.
        //       refer to: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/defs
        if (loader->def && loader->doc->node.doc.defs) {
            loader->def->node.defs.gradients.push(gradient);
        } else {
            loader->gradients.push(gradient);
        }
        loader->latestGradient = gradient;
    } else if (!strcmp(tagName, "stop")) {
        if (!loader->latestGradient) {
            TVGLOG("SVG", "Stop element is used outside of the Gradient element");
            return;
        }
        /* default value for opacity */
        loader->svgParse->gradStop = {0.0f, 0, 0, 0, 255};
        simpleXmlParseAttributes(attrs, attrsLength, _attrParseStops, loader);
        loader->latestGradient->stops.push(loader->svgParse->gradStop);
    } else if (!isIgnoreUnsupportedLogElements(tagName)) {
        TVGLOG("SVG", "Unsupported elements used [Elements: %s]", tagName);
    }
}


static bool _svgLoaderParser(void* data, SimpleXMLType type, const char* content, unsigned int length)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;

    switch (type) {
        case SimpleXMLType::Open: {
            _svgLoaderParserXmlOpen(loader, content, length, false);
            break;
        }
        case SimpleXMLType::OpenEmpty: {
            _svgLoaderParserXmlOpen(loader, content, length, true);
            break;
        }
        case SimpleXMLType::Close: {
            _svgLoaderParerXmlClose(loader, content);
            break;
        }
        case SimpleXMLType::Data:
        case SimpleXMLType::CData:
        case SimpleXMLType::DoctypeChild: {
            break;
        }
        case SimpleXMLType::Ignored:
        case SimpleXMLType::Comment:
        case SimpleXMLType::Doctype: {
            break;
        }
        default: {
            break;
        }
    }

    return true;
}


static void _styleInherit(SvgStyleProperty* child, const SvgStyleProperty* parent)
{
    if (parent == nullptr) return;
    //Inherit the property of parent if not present in child.
    //Fill
    if (!((int)child->fill.flags & (int)SvgFillFlags::Paint)) {
        child->fill.paint.color = parent->fill.paint.color;
        child->fill.paint.none = parent->fill.paint.none;
        child->fill.paint.curColor = parent->fill.paint.curColor;
        if (parent->fill.paint.url) child->fill.paint.url = _copyId(parent->fill.paint.url);
    } else if (child->fill.paint.curColor && !child->curColorSet) {
        child->color = parent->color;
    }
    if (!((int)child->fill.flags & (int)SvgFillFlags::Opacity)) {
        child->fill.opacity = parent->fill.opacity;
    }
    if (!((int)child->fill.flags & (int)SvgFillFlags::FillRule)) {
        child->fill.fillRule = parent->fill.fillRule;
    }
    //Stroke
    if (!((int)child->stroke.flags & (int)SvgStrokeFlags::Paint)) {
        child->stroke.paint.color = parent->stroke.paint.color;
        child->stroke.paint.none = parent->stroke.paint.none;
        child->stroke.paint.curColor = parent->stroke.paint.curColor;
        child->stroke.paint.url = parent->stroke.paint.url ? _copyId(parent->stroke.paint.url) : nullptr;
    } else if (child->stroke.paint.curColor && !child->curColorSet) {
        child->color = parent->color;
    }
    if (!((int)child->stroke.flags & (int)SvgStrokeFlags::Opacity)) {
        child->stroke.opacity = parent->stroke.opacity;
    }
    if (!((int)child->stroke.flags & (int)SvgStrokeFlags::Width)) {
        child->stroke.width = parent->stroke.width;
    }
    if (!((int)child->stroke.flags & (int)SvgStrokeFlags::Dash)) {
        if (parent->stroke.dash.array.count > 0) {
            child->stroke.dash.array.clear();
            child->stroke.dash.array.reserve(parent->stroke.dash.array.count);
            for (uint32_t i = 0; i < parent->stroke.dash.array.count; ++i) {
                child->stroke.dash.array.push(parent->stroke.dash.array.data[i]);
            }
        }
    }
    if (!((int)child->stroke.flags & (int)SvgStrokeFlags::Cap)) {
        child->stroke.cap = parent->stroke.cap;
    }
    if (!((int)child->stroke.flags & (int)SvgStrokeFlags::Join)) {
        child->stroke.join = parent->stroke.join;
    }
}


static void _inefficientNodeCheck(TVG_UNUSED SvgNode* node){
#ifdef THORVG_LOG_ENABLED
    auto type = simpleXmlNodeTypeToString(node->type);

    if (!node->display && node->type != SvgNodeType::ClipPath) TVGLOG("SVG", "Inefficient elements used [Display is none][Node Type : %s]", type);
    if (node->style->opacity == 0) TVGLOG("SVG", "Inefficient elements used [Opacity is zero][Node Type : %s]", type);
    if (node->style->fill.opacity == 0 && node->style->stroke.opacity == 0) TVGLOG("SVG", "Inefficient elements used [Fill opacity and stroke opacity are zero][Node Type : %s]", type);

    switch (node->type) {
        case SvgNodeType::Path: {
            if (!node->node.path.path) TVGLOG("SVG", "Inefficient elements used [Empty path][Node Type : %s]", type);
            break;
        }
        case SvgNodeType::Ellipse: {
            if (node->node.ellipse.rx == 0 && node->node.ellipse.ry == 0) TVGLOG("SVG", "Inefficient elements used [Size is zero][Node Type : %s]", type);
            break;
        }
        case SvgNodeType::Polygon:
        case SvgNodeType::Polyline: {
            if (node->node.polygon.pointsCount < 2) TVGLOG("SVG", "Inefficient elements used [Invalid Polygon][Node Type : %s]", type);
            break;
        }
        case SvgNodeType::Circle: {
            if (node->node.circle.r == 0) TVGLOG("SVG", "Inefficient elements used [Size is zero][Node Type : %s]", type);
            break;
        }
        case SvgNodeType::Rect: {
            if (node->node.rect.w == 0 && node->node.rect.h) TVGLOG("SVG", "Inefficient elements used [Size is zero][Node Type : %s]", type);
            break;
        }
        case SvgNodeType::Line: {
            if (node->node.line.x1 == node->node.line.x2 && node->node.line.y1 == node->node.line.y2) TVGLOG("SVG", "Inefficient elements used [Size is zero][Node Type : %s]", type);
            break;
        }
        default: break;
    }
#endif
}


static void _updateStyle(SvgNode* node, SvgStyleProperty* parentStyle)
{
    _styleInherit(node->style, parentStyle);
    _inefficientNodeCheck(node);

    auto child = node->child.data;
    for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
        _updateStyle(*child, node->style);
    }
}


static SvgStyleGradient* _gradientDup(Array<SvgStyleGradient*>* gradients, const char* id)
{
    SvgStyleGradient* result = nullptr;

    auto gradList = gradients->data;

    for (uint32_t i = 0; i < gradients->count; ++i) {
        if (!strcmp((*gradList)->id, id)) {
            result = _cloneGradient(*gradList);
            break;
        }
        ++gradList;
    }

    if (result && result->ref) {
        gradList = gradients->data;
        for (uint32_t i = 0; i < gradients->count; ++i) {
            if (!strcmp((*gradList)->id, result->ref)) {
                if (result->stops.count == 0) _cloneGradStops(result->stops, (*gradList)->stops);
                //TODO: Properly inherit other property
                break;
            }
            ++gradList;
        }
    }

    return result;
}


static void _updateGradient(SvgNode* node, Array<SvgStyleGradient*>* gradients)
{
    if (node->child.count > 0) {
        auto child = node->child.data;
        for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
            _updateGradient(*child, gradients);
        }
    } else {
        if (node->style->fill.paint.url) {
            if (node->style->fill.paint.gradient) {
                node->style->fill.paint.gradient->clear();
                free(node->style->fill.paint.gradient);
            }
            node->style->fill.paint.gradient = _gradientDup(gradients, node->style->fill.paint.url);
        }
        if (node->style->stroke.paint.url) {
            if (node->style->stroke.paint.gradient) {
                node->style->stroke.paint.gradient->clear();
                free(node->style->stroke.paint.gradient);
            }
            node->style->stroke.paint.gradient = _gradientDup(gradients, node->style->stroke.paint.url);
        }
    }
}


static void _updateComposite(SvgNode* node, SvgNode* root)
{
    if (node->style->clipPath.url && !node->style->clipPath.node) {
        SvgNode* findResult = _findNodeById(root, node->style->clipPath.url);
        if (findResult) node->style->clipPath.node = findResult;
    }
    if (node->style->mask.url && !node->style->mask.node) {
        SvgNode* findResult = _findNodeById(root, node->style->mask.url);
        if (findResult) node->style->mask.node = findResult;
    }
    if (node->child.count > 0) {
        auto child = node->child.data;
        for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
            _updateComposite(*child, root);
        }
    }
}


static void _freeNodeStyle(SvgStyleProperty* style)
{
    if (!style) return;

    //style->clipPath.node and style->mask.node has only the addresses of node. Therefore, node is released from _freeNode.
    free(style->clipPath.url);
    free(style->mask.url);

    if (style->fill.paint.gradient) {
        style->fill.paint.gradient->clear();
        free(style->fill.paint.gradient);
    }
    if (style->stroke.paint.gradient) {
        style->stroke.paint.gradient->clear();
        free(style->stroke.paint.gradient);
    }
    free(style->fill.paint.url);
    free(style->stroke.paint.url);
    style->stroke.dash.array.reset();
    free(style);
}


static void _freeNode(SvgNode* node)
{
    if (!node) return;

    auto child = node->child.data;
    for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
        _freeNode(*child);
    }
    node->child.reset();

    free(node->id);
    free(node->transform);
    _freeNodeStyle(node->style);
    switch (node->type) {
         case SvgNodeType::Path: {
             free(node->node.path.path);
             break;
         }
         case SvgNodeType::Polygon: {
             free(node->node.polygon.points);
             break;
         }
         case SvgNodeType::Polyline: {
             free(node->node.polyline.points);
             break;
         }
         case SvgNodeType::Doc: {
             _freeNode(node->node.doc.defs);
             break;
         }
         case SvgNodeType::Defs: {
             auto gradients = node->node.defs.gradients.data;
             for (size_t i = 0; i < node->node.defs.gradients.count; ++i) {
                 (*gradients)->clear();
                 free(*gradients);
                 ++gradients;
             }
             node->node.defs.gradients.reset();
             break;
         }
         case SvgNodeType::Image: {
             free(node->node.image.href);
             break;
         }
         default: {
             break;
         }
    }
    free(node);
}


static bool _svgLoaderParserForValidCheckXmlOpen(SvgLoaderData* loader, const char* content, unsigned int length)
{
    const char* attrs = nullptr;
    int sz = length;
    char tagName[20] = "";
    FactoryMethod method;
    SvgNode *node = nullptr;
    int attrsLength = 0;
    loader->level++;
    attrs = simpleXmlFindAttributesTag(content, length);

    if (!attrs) {
        //Parse the empty tag
        attrs = content;
        while ((attrs != nullptr) && *attrs != '>') attrs++;
    }

    if (attrs) {
        sz = attrs - content;
        while ((sz > 0) && (isspace(content[sz - 1]))) sz--;
        if ((unsigned)sz >= sizeof(tagName)) return false;
        strncpy(tagName, content, sz);
        tagName[sz] = '\0';
        attrsLength = length - sz;
    }

    if ((method = _findGroupFactory(tagName))) {
        if (!loader->doc) {
            if (strcmp(tagName, "svg")) return true; //Not a valid svg document
            node = method(loader, nullptr, attrs, attrsLength);
            loader->doc = node;
            loader->stack.push(node);
            return false;
        }
    }
    return true;
}


static bool _svgLoaderParserForValidCheck(void* data, SimpleXMLType type, const char* content, unsigned int length)
{
    SvgLoaderData* loader = (SvgLoaderData*)data;
    bool res = true;;

    switch (type) {
        case SimpleXMLType::Open:
        case SimpleXMLType::OpenEmpty: {
            //If 'res' is false, it means <svg> tag is found.
            res = _svgLoaderParserForValidCheckXmlOpen(loader, content, length);
            break;
        }
        default: {
            break;
        }
    }

    return res;
}


void SvgLoader::clear()
{
    if (copy) free((char*)content);
    size = 0;
    content = nullptr;
    copy = false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

SvgLoader::SvgLoader()
{
}


SvgLoader::~SvgLoader()
{
    close();
}


void SvgLoader::run(unsigned tid)
{
    if (!simpleXmlParse(content, size, true, _svgLoaderParser, &(loaderData))) return;

    if (loaderData.doc) {
        _updateStyle(loaderData.doc, nullptr);
        auto defs = loaderData.doc->node.doc.defs;
        if (defs) _updateGradient(loaderData.doc, &defs->node.defs.gradients);

        if (loaderData.gradients.count > 0) _updateGradient(loaderData.doc, &loaderData.gradients);

        _updateComposite(loaderData.doc, loaderData.doc);
        if (defs) _updateComposite(loaderData.doc, defs);

        if (loaderData.cloneNodes.count > 0) _clonePostponedNodes(&loaderData.cloneNodes);
    }
    root = svgSceneBuild(loaderData.doc, vx, vy, vw, vh, w, h, preserveAspect, svgPath);
}


bool SvgLoader::header()
{
    //For valid check, only <svg> tag is parsed first.
    //If the <svg> tag is found, the loaded file is valid and stores viewbox information.
    //After that, the remaining content data is parsed in order with async.
    loaderData.svgParse = (SvgParser*)malloc(sizeof(SvgParser));
    if (!loaderData.svgParse) return false;

    loaderData.svgParse->flags = SvgStopStyleFlags::StopDefault;

    simpleXmlParse(content, size, true, _svgLoaderParserForValidCheck, &(loaderData));

    if (loaderData.doc && loaderData.doc->type == SvgNodeType::Doc) {
        //Return the brief resource info such as viewbox:
        vx = loaderData.doc->node.doc.vx;
        vy = loaderData.doc->node.doc.vy;
        w = vw = loaderData.doc->node.doc.vw;
        h = vh = loaderData.doc->node.doc.vh;

        //Override size
        if (loaderData.doc->node.doc.w > 0) {
            w = loaderData.doc->node.doc.w;
            if (vw < FLT_EPSILON) vw = w;
        }
        if (loaderData.doc->node.doc.h > 0) {
            h = loaderData.doc->node.doc.h;
            if (vh < FLT_EPSILON) vh = h;
        }

        preserveAspect = loaderData.doc->node.doc.preserveAspect;
    } else {
        TVGLOG("SVG", "No SVG File. There is no <svg/>");
        return false;
    }

    return true;
}


bool SvgLoader::open(const char* data, uint32_t size, bool copy)
{
    clear();

    if (copy) {
        content = (char*)malloc(size);
        if (!content) return false;
        memcpy((char*)content, data, size);
    } else content = data;

    this->size = size;
    this->copy = copy;

    return header();
}


bool SvgLoader::open(const string& path)
{
    clear();

    ifstream f;
    f.open(path);

    if (!f.is_open()) return false;

    svgPath = path;
    getline(f, filePath, '\0');
    f.close();

    if (filePath.empty()) return false;

    content = filePath.c_str();
    size = filePath.size();

    return header();
}


bool SvgLoader::resize(Paint* paint, float w, float h)
{
    if (!paint) return false;

    auto sx = w / this->w;
    auto sy = h / this->h;

    if (preserveAspect) {
        //Scale
        auto scale = sx < sy ? sx : sy;
        paint->scale(scale);
        //Align
        auto tx = 0.0f;
        auto ty = 0.0f;
        auto tw = this->w * scale;
        auto th = this->h * scale;
        if (tw > th) ty -= (h - th) * 0.5f;
        else tx -= (w - tw) * 0.5f;
        paint->translate(-tx, -ty);
    } else {
        //Align
        auto tx = 0.0f;
        auto ty = 0.0f;
        auto tw = this->w * sx;
        auto th = this->h * sy;
        if (tw > th) ty -= (h - th) * 0.5f;
        else tx -= (w - tw) * 0.5f;

        Matrix m = {sx, 0, -tx, 0, sy, -ty, 0, 0, 1};
        paint->transform(m);
    }
    return true;
}


bool SvgLoader::read()
{
    if (!content || size == 0) return false;

    TaskScheduler::request(this);

    return true;
}


bool SvgLoader::close()
{
    this->done();

    if (loaderData.svgParse) {
        free(loaderData.svgParse);
        loaderData.svgParse = nullptr;
    }
    auto gradients = loaderData.gradients.data;
    for (size_t i = 0; i < loaderData.gradients.count; ++i) {
        (*gradients)->clear();
        free(*gradients);
        ++gradients;
    }
    loaderData.gradients.reset();

    _freeNode(loaderData.doc);
    loaderData.doc = nullptr;
    loaderData.stack.reset();

    clear();

    return true;
}


unique_ptr<Paint> SvgLoader::paint()
{
    this->done();
    if (root) return move(root);
    else return nullptr;
}
