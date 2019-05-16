/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
copyright notice, this list of conditions and the
following disclaimer.

* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other
materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
contributors may be used to endorse or promote products
derived from this software without specific prior
written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/
/// \file   FIReader.cpp
/// \brief  Reader for Fast Infoset encoded binary XML files.
/// \date   2017
/// \author Patrick Daehne

#ifndef ASSIMP_BUILD_NO_X3D_IMPORTER

#include "FIReader.hpp"
#include <assimp/StringUtils.h>

// Workaround for issue #1361
// https://github.com/assimp/assimp/issues/1361
#ifdef __ANDROID__
#  define _GLIBCXX_USE_C99 1
#endif

#include <assimp/Exceptional.h>
#include <assimp/IOStream.hpp>
#include <assimp/types.h>
#include <assimp/MemoryIOWrapper.h>
#include <assimp/irrXMLWrapper.h>
#include "../contrib/utf8cpp/source/utf8.h"
#include <assimp/fast_atof.h>
#include <stack>
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace Assimp {

static const std::string parseErrorMessage = "Fast Infoset parse error";

static const char *xmlDeclarations[] = {
    "<?xml encoding='finf'?>",
    "<?xml encoding='finf' standalone='yes'?>",
    "<?xml encoding='finf' standalone='no'?>",
    "<?xml version='1.0' encoding='finf'?>",
    "<?xml version='1.0' encoding='finf' standalone='yes'?>",
    "<?xml version='1.0' encoding='finf' standalone='no'?>",
    "<?xml version='1.1' encoding='finf'?>",
    "<?xml version='1.1' encoding='finf' standalone='yes'?>",
    "<?xml version='1.1' encoding='finf' standalone='no'?>"
};

static size_t parseMagic(const uint8_t *data, const uint8_t *dataEnd) {
    if (dataEnd - data < 4) {
        return 0;
    }
    uint32_t magic = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    switch (magic) {
    case 0xe0000001:
        return 4;
    case 0x3c3f786d: // "<?xm"
        {
            size_t xmlDeclarationsLength = sizeof(xmlDeclarations) / sizeof(xmlDeclarations[0]);
            for (size_t i = 0; i < xmlDeclarationsLength; ++i) {
                auto xmlDeclaration = xmlDeclarations[i];
                ptrdiff_t xmlDeclarationLength = strlen(xmlDeclaration);
                if ((dataEnd - data >= xmlDeclarationLength) && (memcmp(xmlDeclaration, data, xmlDeclarationLength) == 0)) {
                    data += xmlDeclarationLength;
                    if (dataEnd - data < 4) {
                        return 0;
                    }
                    magic = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
                    return magic == 0xe0000001 ? xmlDeclarationLength + 4 : 0;
                }
            }
            return 0;
        }
    default:
        return 0;
    }
}

static std::string parseUTF8String(const uint8_t *data, size_t len) {
    return std::string((char*)data, len);
}

static std::string parseUTF16String(const uint8_t *data, size_t len) {
    if (len & 1) {
        throw DeadlyImportError(parseErrorMessage);
    }
    size_t numShorts = len / 2;
    std::vector<short> utf16;
    utf16.reserve(numShorts);
    for (size_t i = 0; i < numShorts; ++i) {
        short v = (data[0] << 8) | data[1];
        utf16.push_back(v);
        data += 2;
    }
    std::string result;
    utf8::utf16to8(utf16.begin(), utf16.end(), back_inserter(result));
    return result;
}

struct FIStringValueImpl: public FIStringValue {
    inline FIStringValueImpl(std::string &&value_) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ { return value; }
};

std::shared_ptr<FIStringValue> FIStringValue::create(std::string &&value) {
    return std::make_shared<FIStringValueImpl>(std::move(value));
}

struct FIHexValueImpl: public FIHexValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIHexValueImpl(std::vector<uint8_t> &&value_):  strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            os << std::hex << std::uppercase << std::setfill('0');
            std::for_each(value.begin(), value.end(), [&](uint8_t c) { os << std::setw(2) << static_cast<int>(c); });
            strValue = os.str();
        }
        return strValue;
    };
};

std::shared_ptr<FIHexValue> FIHexValue::create(std::vector<uint8_t> &&value) {
    return std::make_shared<FIHexValueImpl>(std::move(value));
}

struct FIBase64ValueImpl: public FIBase64Value {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIBase64ValueImpl(std::vector<uint8_t> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            uint8_t c1 = 0, c2;
            int imod3 = 0;
            std::vector<uint8_t>::size_type valueSize = value.size();
            for (std::vector<uint8_t>::size_type i = 0; i < valueSize; ++i) {
                c2 = value[i];
                switch (imod3) {
                case 0:
                    os << basis_64[c2 >> 2];
                    imod3 = 1;
                    break;
                case 1:
                    os << basis_64[((c1 & 0x03) << 4) | ((c2 & 0xf0) >> 4)];
                    imod3 = 2;
                    break;
                case 2:
                    os << basis_64[((c1 & 0x0f) << 2) | ((c2 & 0xc0) >> 6)] << basis_64[c2 & 0x3f];
                    imod3 = 0;
                    break;
                }
                c1 = c2;
            }
            switch (imod3) {
            case 1:
                os << basis_64[(c1 & 0x03) << 4] << "==";
                break;
            case 2:
                os << basis_64[(c1 & 0x0f) << 2] << '=';
                break;
            }
            strValue = os.str();
        }
        return strValue;
    };
    static const char basis_64[];
};

const char FIBase64ValueImpl::basis_64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::shared_ptr<FIBase64Value> FIBase64Value::create(std::vector<uint8_t> &&value) {
    return std::make_shared<FIBase64ValueImpl>(std::move(value));
}

struct FIShortValueImpl: public FIShortValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIShortValueImpl(std::vector<int16_t> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            int n = 0;
            std::for_each(value.begin(), value.end(), [&](int16_t s) { if (++n > 1) os << ' '; os << s; });
            strValue = os.str();
        }
        return strValue;
    }
};

std::shared_ptr<FIShortValue> FIShortValue::create(std::vector<int16_t> &&value) {
    return std::make_shared<FIShortValueImpl>(std::move(value));
}

struct FIIntValueImpl: public FIIntValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIIntValueImpl(std::vector<int32_t> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            int n = 0;
            std::for_each(value.begin(), value.end(), [&](int32_t i) { if (++n > 1) os << ' '; os << i; });
            strValue = os.str();
        }
        return strValue;
    };
};

std::shared_ptr<FIIntValue> FIIntValue::create(std::vector<int32_t> &&value) {
    return std::make_shared<FIIntValueImpl>(std::move(value));
}

struct FILongValueImpl: public FILongValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FILongValueImpl(std::vector<int64_t> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            int n = 0;
            std::for_each(value.begin(), value.end(), [&](int64_t l) { if (++n > 1) os << ' '; os << l; });
            strValue = os.str();
        }
        return strValue;
    };
};

std::shared_ptr<FILongValue> FILongValue::create(std::vector<int64_t> &&value) {
    return std::make_shared<FILongValueImpl>(std::move(value));
}

struct FIBoolValueImpl: public FIBoolValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIBoolValueImpl(std::vector<bool> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            os << std::boolalpha;
            int n = 0;
            std::for_each(value.begin(), value.end(), [&](bool b) { if (++n > 1) os << ' '; os << b; });
            strValue = os.str();
        }
        return strValue;
    };
};

std::shared_ptr<FIBoolValue> FIBoolValue::create(std::vector<bool> &&value) {
    return std::make_shared<FIBoolValueImpl>(std::move(value));
}

struct FIFloatValueImpl: public FIFloatValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIFloatValueImpl(std::vector<float> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            int n = 0;
            std::for_each(value.begin(), value.end(), [&](float f) { if (++n > 1) os << ' '; os << f; });
            strValue = os.str();
        }
        return strValue;
    }
};

std::shared_ptr<FIFloatValue> FIFloatValue::create(std::vector<float> &&value) {
    return std::make_shared<FIFloatValueImpl>(std::move(value));
}

struct FIDoubleValueImpl: public FIDoubleValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIDoubleValueImpl(std::vector<double> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            int n = 0;
            std::for_each(value.begin(), value.end(), [&](double d) { if (++n > 1) os << ' '; os << d; });
            strValue = os.str();
        }
        return strValue;
    }
};

std::shared_ptr<FIDoubleValue> FIDoubleValue::create(std::vector<double> &&value) {
    return std::make_shared<FIDoubleValueImpl>(std::move(value));
}

struct FIUUIDValueImpl: public FIUUIDValue {
    mutable std::string strValue;
    mutable bool strValueValid;
    inline FIUUIDValueImpl(std::vector<uint8_t> &&value_): strValueValid(false) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ {
        if (!strValueValid) {
            strValueValid = true;
            std::ostringstream os;
            os << std::hex << std::uppercase << std::setfill('0');
            std::vector<uint8_t>::size_type valueSize = value.size();
            for (std::vector<uint8_t>::size_type i = 0; i < valueSize; ++i) {
                switch (i & 15) {
                case 0:
                    if (i > 0) {
                        os << ' ';
                    }
                    os << std::setw(2) << static_cast<int>(value[i]);
                    break;
                case 4:
                case 6:
                case 8:
                case 10:
                    os << '-';
                    // intentionally fall through!
                case 1:
                case 2:
                case 3:
                case 5:
                case 7:
                case 9:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                    os << std::setw(2) << static_cast<int>(value[i]);
                    break;
                }
            }
            strValue = os.str();
        }
        return strValue;
    };
};

std::shared_ptr<FIUUIDValue> FIUUIDValue::create(std::vector<uint8_t> &&value) {
    return std::make_shared<FIUUIDValueImpl>(std::move(value));
}

struct FICDATAValueImpl: public FICDATAValue {
    inline FICDATAValueImpl(std::string &&value_) { value = std::move(value_); }
    virtual const std::string &toString() const /*override*/ { return value; }
};

std::shared_ptr<FICDATAValue> FICDATAValue::create(std::string &&value) {
    return std::make_shared<FICDATAValueImpl>(std::move(value));
}

struct FIHexDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        return FIHexValue::create(std::vector<uint8_t>(data, data + len));
    }
};

struct FIBase64Decoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        return FIBase64Value::create(std::vector<uint8_t>(data, data + len));
    }
};

struct FIShortDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        if (len & 1) {
            throw DeadlyImportError(parseErrorMessage);
        }
        std::vector<int16_t> value;
        size_t numShorts = len / 2;
        value.reserve(numShorts);
        for (size_t i = 0; i < numShorts; ++i) {
            int16_t v = (data[0] << 8) | data[1];
            value.push_back(v);
            data += 2;
        }
        return FIShortValue::create(std::move(value));
    }
};

struct FIIntDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        if (len & 3) {
            throw DeadlyImportError(parseErrorMessage);
        }
        std::vector<int32_t> value;
        size_t numInts = len / 4;
        value.reserve(numInts);
        for (size_t i = 0; i < numInts; ++i) {
            int32_t v = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
            value.push_back(v);
            data += 4;
        }
        return FIIntValue::create(std::move(value));
    }
};

struct FILongDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        if (len & 7) {
            throw DeadlyImportError(parseErrorMessage);
        }
        std::vector<int64_t> value;
        size_t numLongs = len / 8;
        value.reserve(numLongs);
        for (size_t i = 0; i < numLongs; ++i) {
            int64_t b0 = data[0], b1 = data[1], b2 = data[2], b3 = data[3], b4 = data[4], b5 = data[5], b6 = data[6], b7 = data[7];
            int64_t v = (b0 << 56) | (b1 << 48) | (b2 << 40) | (b3 << 32) | (b4 << 24) | (b5 << 16) | (b6 << 8) | b7;
            value.push_back(v);
            data += 8;
        }
        return FILongValue::create(std::move(value));
    }
};

struct FIBoolDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        if (len < 1) {
            throw DeadlyImportError(parseErrorMessage);
        }
        std::vector<bool> value;
        uint8_t b = *data++;
        size_t unusedBits = b >> 4;
        size_t numBools = (len * 8) - 4 - unusedBits;
        value.reserve(numBools);
        uint8_t mask = 1 << 3;
        for (size_t i = 0; i < numBools; ++i) {
            if (!mask) {
                mask = 1 << 7;
                b = *data++;
            }
            value.push_back((b & mask) != 0);
        }
        return FIBoolValue::create(std::move(value));
    }
};

struct FIFloatDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        if (len & 3) {
            throw DeadlyImportError(parseErrorMessage);
        }
        std::vector<float> value;
        size_t numFloats = len / 4;
        value.reserve(numFloats);
        for (size_t i = 0; i < numFloats; ++i) {
            int v = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
            float f;
            memcpy(&f, &v, 4);
            value.push_back(f);
            data += 4;
        }
        return FIFloatValue::create(std::move(value));
    }
};

struct FIDoubleDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        if (len & 7) {
            throw DeadlyImportError(parseErrorMessage);
        }
        std::vector<double> value;
        size_t numDoubles = len / 8;
        value.reserve(numDoubles);
        for (size_t i = 0; i < numDoubles; ++i) {
            long long b0 = data[0], b1 = data[1], b2 = data[2], b3 = data[3], b4 = data[4], b5 = data[5], b6 = data[6], b7 = data[7];
            long long v = (b0 << 56) | (b1 << 48) | (b2 << 40) | (b3 << 32) | (b4 << 24) | (b5 << 16) | (b6 << 8) | b7;
            double f;
            memcpy(&f, &v, 8);
            value.push_back(f);
            data += 8;
        }
        return FIDoubleValue::create(std::move(value));
    }
};

struct FIUUIDDecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        if (len & 15) {
            throw DeadlyImportError(parseErrorMessage);
        }
        return FIUUIDValue::create(std::vector<uint8_t>(data, data + len));
    }
};

struct FICDATADecoder: public FIDecoder {
    virtual std::shared_ptr<const FIValue> decode(const uint8_t *data, size_t len) /*override*/ {
        return FICDATAValue::create(parseUTF8String(data, len));
    }
};

class CFIReaderImpl: public FIReader {
public:

    CFIReaderImpl(std::unique_ptr<uint8_t[]> data_, size_t size):
    data(std::move(data_)), dataP(data.get()), dataEnd(data.get() + size), currentNodeType(irr::io::EXN_NONE),
    emptyElement(false), headerPending(true), terminatorPending(false)
    {}

    virtual ~CFIReaderImpl() {}

    virtual bool read() /*override*/ {
        if (headerPending) {
            headerPending = false;
            parseHeader();
        }
        if (terminatorPending) {
            terminatorPending = false;
            if (elementStack.empty()) {
                return false;
            }
            else {
                nodeName = elementStack.top();
                elementStack.pop();
                currentNodeType = nodeName.empty() ? irr::io::EXN_UNKNOWN : irr::io::EXN_ELEMENT_END;
                return true;
            }
        }
        if (dataP >= dataEnd) {
            return false;
        }
        uint8_t b = *dataP;
        if (b < 0x80) { // Element (C.2.11.2, C.3.7.2)
            // C.3
            parseElement();
            return true;
        }
        else if (b < 0xc0) { // Characters (C.3.7.5)
            // C.7
            auto chars = parseNonIdentifyingStringOrIndex3(vocabulary.charactersTable);
            nodeName = chars->toString();
            currentNodeType = irr::io::EXN_TEXT;
            return true;
        }
        else if (b < 0xe0) {
            if ((b & 0xfc) == 0xc4) { // DTD (C.2.11.5)
                // C.9
                ++dataP;
                if (b & 0x02) {
                    /*const std::string &systemID =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                }
                if (b & 0x01) {
                    /*const std::string &publicID =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                }
                elementStack.push(EmptyString);
                currentNodeType = irr::io::EXN_UNKNOWN;
                return true;
            }
            else if ((b & 0xfc) == 0xc8) { // Unexpanded entity reference (C.3.7.4)
                // C.6
                ++dataP;
                /*const std::string &name =*/ parseIdentifyingStringOrIndex(vocabulary.otherNCNameTable);
                if (b & 0x02) {
                    /*const std::string &systemID =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                }
                if (b & 0x01) {
                    /*const std::string &publicID =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                }
                currentNodeType = irr::io::EXN_UNKNOWN;
                return true;
            }
        }
        else if (b < 0xf0) {
            if (b == 0xe1) { // Processing instruction (C.2.11.3, C.3.7.3)
                // C.5
                ++dataP;
                /*const std::string &target =*/ parseIdentifyingStringOrIndex(vocabulary.otherNCNameTable);
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                /*std::shared_ptr<const FIValue> data =*/ parseNonIdentifyingStringOrIndex1(vocabulary.otherStringTable);
                currentNodeType = irr::io::EXN_UNKNOWN;
                return true;
            }
            else if (b == 0xe2) { // Comment (C.2.11.4, C.3.7.6)
                // C.8
                ++dataP;
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                std::shared_ptr<const FIValue> comment = parseNonIdentifyingStringOrIndex1(vocabulary.otherStringTable);
                nodeName = comment->toString();
                currentNodeType = irr::io::EXN_COMMENT;
                return true;
            }
        }
        else { // Terminator (C.2.12, C.3.8)
            ++dataP;
            if (b == 0xff) {
                terminatorPending = true;
            }
            if (elementStack.empty()) {
                return false;
            }
            else {
                nodeName = elementStack.top();
                elementStack.pop();
                currentNodeType = nodeName.empty() ? irr::io::EXN_UNKNOWN : irr::io::EXN_ELEMENT_END;
                return true;
            }
        }
        throw DeadlyImportError(parseErrorMessage);
    }

    virtual irr::io::EXML_NODE getNodeType() const /*override*/ {
        return currentNodeType;
    }

    virtual int getAttributeCount() const /*override*/ {
        return static_cast<int>(attributes.size());
    }

    virtual const char* getAttributeName(int idx) const /*override*/ {
        if (idx < 0 || idx >= (int)attributes.size()) {
            return nullptr;
        }
        return attributes[idx].name.c_str();
    }

    virtual const char* getAttributeValue(int idx) const /*override*/ {
        if (idx < 0 || idx >= (int)attributes.size()) {
            return nullptr;
        }
        return attributes[idx].value->toString().c_str();
    }

    virtual const char* getAttributeValue(const char* name) const /*override*/ {
        const Attribute* attr = getAttributeByName(name);
        if (!attr) {
            return nullptr;
        }
        return attr->value->toString().c_str();
    }

    virtual const char* getAttributeValueSafe(const char* name) const /*override*/ {
        const Attribute* attr = getAttributeByName(name);
        if (!attr) {
            return EmptyString.c_str();
        }
        return attr->value->toString().c_str();
    }

    virtual int getAttributeValueAsInt(const char* name) const /*override*/ {
        const Attribute* attr = getAttributeByName(name);
        if (!attr) {
            return 0;
        }
        std::shared_ptr<const FIIntValue> intValue = std::dynamic_pointer_cast<const FIIntValue>(attr->value);
        if (intValue) {
            return intValue->value.size() == 1 ? intValue->value.front() : 0;
        }
        return atoi(attr->value->toString().c_str());
    }

    virtual int getAttributeValueAsInt(int idx) const /*override*/ {
        if (idx < 0 || idx >= (int)attributes.size()) {
            return 0;
        }
        std::shared_ptr<const FIIntValue> intValue = std::dynamic_pointer_cast<const FIIntValue>(attributes[idx].value);
        if (intValue) {
            return intValue->value.size() == 1 ? intValue->value.front() : 0;
        }
        return atoi(attributes[idx].value->toString().c_str());
    }

    virtual float getAttributeValueAsFloat(const char* name) const /*override*/ {
        const Attribute* attr = getAttributeByName(name);
        if (!attr) {
            return 0;
        }
        std::shared_ptr<const FIFloatValue> floatValue = std::dynamic_pointer_cast<const FIFloatValue>(attr->value);
        if (floatValue) {
            return floatValue->value.size() == 1 ? floatValue->value.front() : 0;
        }

        return fast_atof(attr->value->toString().c_str());
    }

    virtual float getAttributeValueAsFloat(int idx) const /*override*/ {
        if (idx < 0 || idx >= (int)attributes.size()) {
            return 0;
        }
        std::shared_ptr<const FIFloatValue> floatValue = std::dynamic_pointer_cast<const FIFloatValue>(attributes[idx].value);
        if (floatValue) {
            return floatValue->value.size() == 1 ? floatValue->value.front() : 0;
        }
        return fast_atof(attributes[idx].value->toString().c_str());
    }

    virtual const char* getNodeName() const /*override*/ {
        return nodeName.c_str();
    }

    virtual const char* getNodeData() const /*override*/ {
        return nodeName.c_str();
    }

    virtual bool isEmptyElement() const /*override*/ {
        return emptyElement;
    }

    virtual irr::io::ETEXT_FORMAT getSourceFormat() const /*override*/ {
        return irr::io::ETF_UTF8;
    }

    virtual irr::io::ETEXT_FORMAT getParserFormat() const /*override*/ {
        return irr::io::ETF_UTF8;
    }

    virtual std::shared_ptr<const FIValue> getAttributeEncodedValue(int idx) const /*override*/ {
        if (idx < 0 || idx >= (int)attributes.size()) {
            return nullptr;
        }
        return attributes[idx].value;
    }

    virtual std::shared_ptr<const FIValue> getAttributeEncodedValue(const char* name) const /*override*/ {
        const Attribute* attr = getAttributeByName(name);
        if (!attr) {
            return nullptr;
        }
        return attr->value;
    }

    virtual void registerDecoder(const std::string &algorithmUri, std::unique_ptr<FIDecoder> decoder) /*override*/ {
        decoderMap[algorithmUri] = std::move(decoder);
    }

    virtual void registerVocabulary(const std::string &vocabularyUri, const FIVocabulary *vocabulary) /*override*/ {
        vocabularyMap[vocabularyUri] = vocabulary;
    }

private:

    struct QName {
        std::string prefix;
        std::string uri;
        std::string name;
        inline QName() {}
        inline QName(const FIQName &qname): prefix(qname.prefix ? qname.prefix : ""), uri(qname.uri ? qname.uri : ""), name(qname.name) {}
    };

    struct Attribute {
        QName qname;
        std::string name;
        std::shared_ptr<const FIValue> value;
    };

    struct Vocabulary {
        std::vector<std::string> restrictedAlphabetTable;
        std::vector<std::string> encodingAlgorithmTable;
        std::vector<std::string> prefixTable;
        std::vector<std::string> namespaceNameTable;
        std::vector<std::string> localNameTable;
        std::vector<std::string> otherNCNameTable;
        std::vector<std::string> otherURITable;
        std::vector<std::shared_ptr<const FIValue>> attributeValueTable;
        std::vector<std::shared_ptr<const FIValue>> charactersTable;
        std::vector<std::shared_ptr<const FIValue>> otherStringTable;
        std::vector<QName> elementNameTable;
        std::vector<QName> attributeNameTable;
        Vocabulary() {
            prefixTable.push_back("xml");
            namespaceNameTable.push_back("http://www.w3.org/XML/1998/namespace");
        }
    };

    const Attribute* getAttributeByName(const char* name) const {
        if (!name) {
            return 0;
        }
        std::string n = name;
        for (int i=0; i<(int)attributes.size(); ++i) {
            if (attributes[i].name == n) {
                return &attributes[i];
            }
        }
        return 0;
    }

    size_t parseInt2() { // C.25
        uint8_t b = *dataP++;
        if (!(b & 0x40)) { // x0...... (C.25.2)
            return b & 0x3f;
        }
        else if ((b & 0x60) == 0x40) { // x10..... ........ (C.25.3)
            if (dataEnd - dataP > 0) {
                return (((b & 0x1f) << 8) | *dataP++) + 0x40;
            }
        }
        else if ((b & 0x70) == 0x60) { // x110.... ........ ........ (C.25.4)
            if (dataEnd - dataP > 1) {
                size_t result = (((b & 0x0f) << 16) | (dataP[0] << 8) | dataP[1]) + 0x2040;
                dataP += 2;
                return result;
            }
        }
        throw DeadlyImportError(parseErrorMessage);
    }

    size_t parseInt3() { // C.27
        uint8_t b = *dataP++;
        if (!(b & 0x20)) { // xx0..... (C.27.2)
            return b & 0x1f;
        }
        else if ((b & 0x38) == 0x20) { // xx100... ........ (C.27.3)
            if (dataEnd - dataP > 0) {
                return (((b & 0x07) << 8) | *dataP++) + 0x20;
            }
        }
        else if ((b & 0x38) == 0x28) { // xx101... ........ ........ (C.27.4)
            if (dataEnd - dataP > 1) {
                size_t result = (((b & 0x07) << 16) | (dataP[0] << 8) | dataP[1]) + 0x820;
                dataP += 2;
                return result;
            }
        }
        else if ((b & 0x3f) == 0x30) { // xx110000 0000.... ........ ........ (C.27.5)
            if ((dataEnd - dataP > 2) && !(dataP[0] & 0xf0)) {
                size_t result = (((dataP[0] & 0x0f) << 16) | (dataP[1] << 8) | dataP[2]) + 0x80820;
                dataP += 3;
                return result;
            }
        }
        throw DeadlyImportError(parseErrorMessage);
    }

    size_t parseInt4() { // C.28
        uint8_t b = *dataP++;
        if (!(b & 0x10)) { // xxx0.... (C.28.2)
            return b & 0x0f;
        }
        else if ((b & 0x1c) == 0x10) { // xxx100.. ........ (C.28.3)
            if (dataEnd - dataP > 0) {
                return (((b & 0x03) << 8) | *dataP++) + 0x10;
            }
        }
        else if ((b & 0x1c) == 0x14) { // xxx101.. ........ ........ (C.28.4)
            if (dataEnd - dataP > 1) {
                size_t result = (((b & 0x03) << 16) | (dataP[0] << 8) | dataP[1]) + 0x410;
                dataP += 2;
                return result;
            }
        }
        else if ((b & 0x1f) == 0x18) { // xxx11000 0000.... ........ ........ (C.28.5)
            if ((dataEnd - dataP > 2) && !(dataP[0] & 0xf0)) {
                size_t result = (((dataP[0] & 0x0f) << 16) | (dataP[1] << 8) | dataP[2]) + 0x40410;
                dataP += 3;
                return result;
            }
        }
        throw DeadlyImportError(parseErrorMessage);
    }

    size_t parseSequenceLen() { // C.21
        if (dataEnd - dataP > 0) {
            uint8_t b = *dataP++;
            if (b < 0x80) { // 0....... (C.21.2)
                return b;
            }
            else if ((b & 0xf0) == 0x80) { // 1000.... ........ ........ (C.21.3)
                if (dataEnd - dataP > 1) {
                    size_t result = (((b & 0x0f) << 16) | (dataP[0] << 8) | dataP[1]) + 0x80;
                    dataP += 2;
                    return result;
                }
            }
        }
        throw DeadlyImportError(parseErrorMessage);
    }

    std::string parseNonEmptyOctetString2() { // C.22
        // Parse the length of the string
        uint8_t b = *dataP++ & 0x7f;
        size_t len;
        if (!(b & 0x40)) { // x0...... (C.22.3.1)
            len = b + 1;
        }
        else if (b == 0x40) { // x1000000 ........ (C.22.3.2)
            if (dataEnd - dataP < 1) {
                throw DeadlyImportError(parseErrorMessage);
            }
            len = *dataP++ + 0x41;
        }
        else if (b == 0x60) { // x1100000 ........ ........ ........ ........ (C.22.3.3)
            if (dataEnd - dataP < 4) {
                throw DeadlyImportError(parseErrorMessage);
            }
            len = ((dataP[0] << 24) | (dataP[1] << 16) | (dataP[2] << 8) | dataP[3]) + 0x141;
            dataP += 4;
        }
        else {
            throw DeadlyImportError(parseErrorMessage);
        }

        // Parse the string (C.22.4)
        if (dataEnd - dataP < static_cast<ptrdiff_t>(len)) {
            throw DeadlyImportError(parseErrorMessage);
        }
        std::string s = parseUTF8String(dataP, len);
        dataP += len;

        return s;
    }

    size_t parseNonEmptyOctetString5Length() { // C.23
        // Parse the length of the string
        size_t b = *dataP++ & 0x0f;
        if (!(b & 0x08)) { // xxxx0... (C.23.3.1)
            return b + 1;
        }
        else if (b == 0x08) { // xxxx1000 ........ (C.23.3.2)
            if (dataEnd - dataP > 0) {
                return *dataP++ + 0x09;
            }
        }
        else if (b == 0x0c) { // xxxx1100 ........ ........ ........ ........ (C.23.3.3)
            if (dataEnd - dataP > 3) {
                size_t result = ((dataP[0] << 24) | (dataP[1] << 16) | (dataP[2] << 8) | dataP[3]) + 0x109;
                dataP += 4;
                return result;
            }
        }
        throw DeadlyImportError(parseErrorMessage);
    }

    size_t parseNonEmptyOctetString7Length() { // C.24
        // Parse the length of the string
        size_t b = *dataP++ & 0x03;
        if (!(b & 0x02)) { // xxxxxx0. (C.24.3.1)
            return b + 1;
        }
        else if (b == 0x02) { // xxxxxx10 ........ (C.24.3.2)
            if (dataEnd - dataP > 0) {
                return *dataP++ + 0x3;
            }
        }
        else if (b == 0x03) { // xxxxxx11 ........ ........ ........ ........ (C.24.3.3)
            if (dataEnd - dataP > 3) {
                size_t result = ((dataP[0] << 24) | (dataP[1] << 16) | (dataP[2] << 8) | dataP[3]) + 0x103;
                dataP += 4;
                return result;
            }
        }
        throw DeadlyImportError(parseErrorMessage);
    }

    std::shared_ptr<const FIValue> parseEncodedData(size_t index, size_t len) {
        if (index < 32) {
            FIDecoder *decoder = defaultDecoder[index];
            if (!decoder) {
                throw DeadlyImportError("Invalid encoding algorithm index " + to_string(index));
            }
            return decoder->decode(dataP, len);
        }
        else {
            if (index - 32 >= vocabulary.encodingAlgorithmTable.size()) {
                throw DeadlyImportError("Invalid encoding algorithm index " + to_string(index));
            }
            std::string uri = vocabulary.encodingAlgorithmTable[index - 32];
            auto it = decoderMap.find(uri);
            if (it == decoderMap.end()) {
                throw DeadlyImportError("Unsupported encoding algorithm " + uri);
            }
            else {
                return it->second->decode(dataP, len);
            }
        }
    }

    std::shared_ptr<const FIValue> parseRestrictedAlphabet(size_t index, size_t len) {
        std::string alphabet;
        if (index < 16) {
            switch (index) {
            case 0: // numeric
                alphabet = "0123456789-+.e ";
                break;
            case 1: // date and time
                alphabet = "0123456789-:TZ ";
                break;
            default:
                throw DeadlyImportError("Invalid restricted alphabet index " + to_string(index));
            }
        }
        else {
            if (index - 16 >= vocabulary.restrictedAlphabetTable.size()) {
                throw DeadlyImportError("Invalid restricted alphabet index " + to_string(index));
            }
            alphabet = vocabulary.restrictedAlphabetTable[index - 16];
        }
        std::vector<uint32_t> alphabetUTF32;
        utf8::utf8to32(alphabet.begin(), alphabet.end(), back_inserter(alphabetUTF32));
        std::string::size_type alphabetLength = alphabetUTF32.size();
        if (alphabetLength < 2) {
            throw DeadlyImportError("Invalid restricted alphabet length " + to_string(alphabetLength));
        }
        std::string::size_type bitsPerCharacter = 1;
        while ((1ull << bitsPerCharacter) <= alphabetLength) {
            ++bitsPerCharacter;
        }
        size_t bitsAvail = 0;
        uint8_t mask = (1 << bitsPerCharacter) - 1;
        uint32_t bits = 0;
        std::string s;
        for (size_t i = 0; i < len; ++i) {
            bits = (bits << 8) | dataP[i];
            bitsAvail += 8;
            while (bitsAvail >= bitsPerCharacter) {
                bitsAvail -= bitsPerCharacter;
                size_t charIndex = (bits >> bitsAvail) & mask;
                if (charIndex < alphabetLength) {
                    s.push_back(alphabetUTF32[charIndex]);
                }
                else if (charIndex != mask) {
                    throw DeadlyImportError(parseErrorMessage);
                }
            }
        }
        return FIStringValue::create(std::move(s));
    }

    std::shared_ptr<const FIValue> parseEncodedCharacterString3() { // C.19
        std::shared_ptr<const FIValue> result;
        size_t len;
        uint8_t b = *dataP;
        if (b & 0x20) {
            ++dataP;
            if (dataEnd - dataP < 1) {
                throw DeadlyImportError(parseErrorMessage);
            }
            size_t index = ((b & 0x0f) << 4) | ((*dataP & 0xf0) >> 4); // C.29
            len = parseNonEmptyOctetString5Length();
            if (dataEnd - dataP < static_cast<ptrdiff_t>(len)) {
                throw DeadlyImportError(parseErrorMessage);
            }
            if (b & 0x10) {
                // encoding algorithm (C.19.3.4)
                result = parseEncodedData(index, len);
            }
            else {
                // Restricted alphabet (C.19.3.3)
                result = parseRestrictedAlphabet(index, len);
            }
        }
        else {
            len = parseNonEmptyOctetString5Length();
            if (dataEnd - dataP < static_cast<ptrdiff_t>(len)) {
                throw DeadlyImportError(parseErrorMessage);
            }
            if (b & 0x10) {
                // UTF-16 (C.19.3.2)
                if (len & 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                result = FIStringValue::create(parseUTF16String(dataP, len));
            }
            else {
                // UTF-8 (C.19.3.1)
                result = FIStringValue::create(parseUTF8String(dataP, len));
            }
        }
        dataP += len;
        return result;
    }

    std::shared_ptr<const FIValue> parseEncodedCharacterString5() { // C.20
        std::shared_ptr<const FIValue> result;
        size_t len;
        uint8_t b = *dataP;
        if (b & 0x08) {
            ++dataP;
            if (dataEnd - dataP < 1) {
                throw DeadlyImportError(parseErrorMessage);
            }
            size_t index = ((b & 0x03) << 6) | ((*dataP & 0xfc) >> 2); /* C.29 */
            len = parseNonEmptyOctetString7Length();
            if (dataEnd - dataP < static_cast<ptrdiff_t>(len)) {
                throw DeadlyImportError(parseErrorMessage);
            }
            if (b & 0x04) {
                // encoding algorithm (C.20.3.4)
                result = parseEncodedData(index, len);
            }
            else {
                // Restricted alphabet (C.20.3.3)
                result = parseRestrictedAlphabet(index, len);
            }
        }
        else {
            len = parseNonEmptyOctetString7Length();
            if (dataEnd - dataP < static_cast<ptrdiff_t>(len)) {
                throw DeadlyImportError(parseErrorMessage);
            }
            if (b & 0x04) {
                // UTF-16 (C.20.3.2)
                if (len & 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                result = FIStringValue::create(parseUTF16String(dataP, len));
            }
            else {
                // UTF-8 (C.20.3.1)
                result = FIStringValue::create(parseUTF8String(dataP, len));
            }
        }
        dataP += len;
        return result;
    }

    const std::string &parseIdentifyingStringOrIndex(std::vector<std::string> &stringTable) { // C.13
        if (dataEnd - dataP < 1) {
            throw DeadlyImportError(parseErrorMessage);
        }
        uint8_t b = *dataP;
        if (b & 0x80) {
            // We have an index (C.13.4)
            size_t index = parseInt2();
            if (index >= stringTable.size()) {
                throw DeadlyImportError(parseErrorMessage);
            }
            return stringTable[index];
        }
        else {
            // We have a string (C.13.3)
            stringTable.push_back(parseNonEmptyOctetString2());
            return stringTable.back();
        }
    }

    QName parseNameSurrogate() { // C.16
        if (dataEnd - dataP < 1) {
            throw DeadlyImportError(parseErrorMessage);
        }
        uint8_t b = *dataP++;
        if (b & 0xfc) { // Padding '000000' C.2.5.5
            throw DeadlyImportError(parseErrorMessage);
        }
        QName result;
        size_t index;
        if (b & 0x02) { // prefix (C.16.3)
            if ((dataEnd - dataP < 1) || (*dataP & 0x80)) {
                throw DeadlyImportError(parseErrorMessage);
            }
            index = parseInt2();
            if (index >= vocabulary.prefixTable.size()) {
                throw DeadlyImportError(parseErrorMessage);
            }
            result.prefix = vocabulary.prefixTable[index];
        }
        if (b & 0x01) { // namespace-name (C.16.4)
            if ((dataEnd - dataP < 1) || (*dataP & 0x80)) {
                throw DeadlyImportError(parseErrorMessage);
            }
            index = parseInt2();
            if (index >= vocabulary.namespaceNameTable.size()) {
                throw DeadlyImportError(parseErrorMessage);
            }
            result.uri = vocabulary.namespaceNameTable[index];
        }
        // local-name
        if ((dataEnd - dataP < 1) || (*dataP & 0x80)) {
            throw DeadlyImportError(parseErrorMessage);
        }
        index = parseInt2();
        if (index >= vocabulary.localNameTable.size()) {
            throw DeadlyImportError(parseErrorMessage);
        }
        result.name = vocabulary.localNameTable[index];
        return result;
    }

    const QName &parseQualifiedNameOrIndex2(std::vector<QName> &qNameTable) { // C.17
        uint8_t b = *dataP;
        if ((b & 0x7c) == 0x78) { // x11110..
            // We have a literal (C.17.3)
            ++dataP;
            QName result;
            // prefix (C.17.3.1)
            result.prefix = b & 0x02 ? parseIdentifyingStringOrIndex(vocabulary.prefixTable) : std::string();
            // namespace-name (C.17.3.1)
            result.uri = b & 0x01 ? parseIdentifyingStringOrIndex(vocabulary.namespaceNameTable) : std::string();
            // local-name
            result.name = parseIdentifyingStringOrIndex(vocabulary.localNameTable);
            qNameTable.push_back(result);
            return qNameTable.back();
        }
        else {
            // We have an index (C.17.4)
            size_t index = parseInt2();
            if (index >= qNameTable.size()) {
                throw DeadlyImportError(parseErrorMessage);
            }
            return qNameTable[index];
        }
    }

    const QName &parseQualifiedNameOrIndex3(std::vector<QName> &qNameTable) { // C.18
        uint8_t b = *dataP;
        if ((b & 0x3c) == 0x3c) { // xx1111..
            // We have a literal (C.18.3)
            ++dataP;
            QName result;
            // prefix (C.18.3.1)
            result.prefix = b & 0x02 ? parseIdentifyingStringOrIndex(vocabulary.prefixTable) : std::string();
            // namespace-name (C.18.3.1)
            result.uri = b & 0x01 ? parseIdentifyingStringOrIndex(vocabulary.namespaceNameTable) : std::string();
            // local-name
            result.name = parseIdentifyingStringOrIndex(vocabulary.localNameTable);
            qNameTable.push_back(result);
            return qNameTable.back();
        }
        else {
            // We have an index (C.18.4)
            size_t index = parseInt3();
            if (index >= qNameTable.size()) {
                throw DeadlyImportError(parseErrorMessage);
            }
            return qNameTable[index];
        }
    }

    std::shared_ptr<const FIValue> parseNonIdentifyingStringOrIndex1(std::vector<std::shared_ptr<const FIValue>> &valueTable) { // C.14
        uint8_t b = *dataP;
        if (b == 0xff) { // C.26.2
            // empty string
            ++dataP;
            return EmptyFIString;
        }
        else if (b & 0x80) { // C.14.4
            // We have an index
            size_t index = parseInt2();
            if (index >= valueTable.size()) {
                throw DeadlyImportError(parseErrorMessage);
            }
            return valueTable[index];
        }
        else { // C.14.3
            // We have a literal
            std::shared_ptr<const FIValue> result = parseEncodedCharacterString3();
            if (b & 0x40) { // C.14.3.1
                valueTable.push_back(result);
            }
            return result;
        }
    }

    std::shared_ptr<const FIValue> parseNonIdentifyingStringOrIndex3(std::vector<std::shared_ptr<const FIValue>> &valueTable) { // C.15
        uint8_t b = *dataP;
        if (b & 0x20) { // C.15.4
            // We have an index
            size_t index = parseInt4();
            if (index >= valueTable.size()) {
                throw DeadlyImportError(parseErrorMessage);
            }
            return valueTable[index];
        }
        else { // C.15.3
            // We have a literal
            std::shared_ptr<const FIValue> result = parseEncodedCharacterString5();
            if (b & 0x10) { // C.15.3.1
                valueTable.push_back(result);
            }
            return result;
        }
    }

    void parseElement() {
        // C.3

        attributes.clear();

        uint8_t b = *dataP;
        bool hasAttributes = (b & 0x40) != 0; // C.3.3
        if ((b & 0x3f) == 0x38) { // C.3.4.1
            // Parse namespaces
            ++dataP;
            for (;;) {
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                b = *dataP++;
                if (b == 0xf0) { // C.3.4.3
                    break;
                }
                if ((b & 0xfc) != 0xcc) { // C.3.4.2
                    throw DeadlyImportError(parseErrorMessage);
                }
                // C.12
                Attribute attr;
                attr.qname.prefix = "xmlns";
                attr.qname.name = b & 0x02 ? parseIdentifyingStringOrIndex(vocabulary.prefixTable) : std::string();
                attr.qname.uri = b & 0x01 ? parseIdentifyingStringOrIndex(vocabulary.namespaceNameTable) : std::string();
                attr.name = attr.qname.name.empty() ? "xmlns" : "xmlns:" + attr.qname.name;
                attr.value = FIStringValue::create(std::string(attr.qname.uri));
                attributes.push_back(attr);
            }
            if ((dataEnd - dataP < 1) || (*dataP & 0xc0)) {
                throw DeadlyImportError(parseErrorMessage);
            }
        }

        // Parse Element name (C.3.5)
        const QName &elemName = parseQualifiedNameOrIndex3(vocabulary.elementNameTable);
        nodeName = elemName.prefix.empty() ? elemName.name : elemName.prefix + ':' + elemName.name;

        if (hasAttributes) {
            for (;;) {
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                b = *dataP;
                if (b < 0x80) { // C.3.6.1
                    // C.4
                    Attribute attr;
                    attr.qname = parseQualifiedNameOrIndex2(vocabulary.attributeNameTable);
                    attr.name = attr.qname.prefix.empty() ? attr.qname.name : attr.qname.prefix + ':' + attr.qname.name;
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    attr.value = parseNonIdentifyingStringOrIndex1(vocabulary.attributeValueTable);
                    attributes.push_back(attr);
                }
                else {
                    if ((b & 0xf0) != 0xf0) { // C.3.6.2
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    emptyElement = b == 0xff; // C.3.6.2, C.3.8
                    ++dataP;
                    break;
                }
            }
        }
        else {
            if (dataEnd - dataP < 1) {
                throw DeadlyImportError(parseErrorMessage);
            }
            b = *dataP;
            switch (b) {
            case 0xff:
                terminatorPending = true;
                // Intentionally fall through
            case 0xf0:
                emptyElement = true;
                ++dataP;
                break;
            default:
                emptyElement = false;
            }
        }
        if (!emptyElement) {
            elementStack.push(nodeName);
        }

        currentNodeType = irr::io::EXN_ELEMENT;
    }

    void parseHeader() {
        // Parse header (C.1.3)
        size_t magicSize = parseMagic(dataP, dataEnd);
        if (!magicSize) {
            throw DeadlyImportError(parseErrorMessage);
        }
        dataP += magicSize;
        // C.2.3
        if (dataEnd - dataP < 1) {
            throw DeadlyImportError(parseErrorMessage);
        }
        uint8_t b = *dataP++;
        if (b & 0x40) {
            // Parse additional data (C.2.4)
            size_t len = parseSequenceLen();
            for (size_t i = 0; i < len; ++i) {
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                /*std::string id =*/ parseNonEmptyOctetString2();
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                /*std::string data =*/ parseNonEmptyOctetString2();
            }
        }
        if (b & 0x20) {
            // Parse initial vocabulary (C.2.5)
            if (dataEnd - dataP < 2) {
                throw DeadlyImportError(parseErrorMessage);
            }
            uint16_t b1 = (dataP[0] << 8) | dataP[1];
            dataP += 2;
            if (b1 & 0x1000) {
                // External vocabulary (C.2.5.2)
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                std::string uri = parseNonEmptyOctetString2();
                auto it = vocabularyMap.find(uri);
                if (it == vocabularyMap.end()) {
                    throw DeadlyImportError("Unknown vocabulary " + uri);
                }
                const FIVocabulary *externalVocabulary = it->second;
                if (externalVocabulary->restrictedAlphabetTable) {
                    std::copy(externalVocabulary->restrictedAlphabetTable, externalVocabulary->restrictedAlphabetTable + externalVocabulary->restrictedAlphabetTableSize, std::back_inserter(vocabulary.restrictedAlphabetTable));
                }
                if (externalVocabulary->encodingAlgorithmTable) {
                    std::copy(externalVocabulary->encodingAlgorithmTable, externalVocabulary->encodingAlgorithmTable + externalVocabulary->encodingAlgorithmTableSize, std::back_inserter(vocabulary.encodingAlgorithmTable));
                }
                if (externalVocabulary->prefixTable) {
                    std::copy(externalVocabulary->prefixTable, externalVocabulary->prefixTable + externalVocabulary->prefixTableSize, std::back_inserter(vocabulary.prefixTable));
                }
                if (externalVocabulary->namespaceNameTable) {
                    std::copy(externalVocabulary->namespaceNameTable, externalVocabulary->namespaceNameTable + externalVocabulary->namespaceNameTableSize, std::back_inserter(vocabulary.namespaceNameTable));
                }
                if (externalVocabulary->localNameTable) {
                    std::copy(externalVocabulary->localNameTable, externalVocabulary->localNameTable + externalVocabulary->localNameTableSize, std::back_inserter(vocabulary.localNameTable));
                }
                if (externalVocabulary->otherNCNameTable) {
                    std::copy(externalVocabulary->otherNCNameTable, externalVocabulary->otherNCNameTable + externalVocabulary->otherNCNameTableSize, std::back_inserter(vocabulary.otherNCNameTable));
                }
                if (externalVocabulary->otherURITable) {
                    std::copy(externalVocabulary->otherURITable, externalVocabulary->otherURITable + externalVocabulary->otherURITableSize, std::back_inserter(vocabulary.otherURITable));
                }
                if (externalVocabulary->attributeValueTable) {
                    std::copy(externalVocabulary->attributeValueTable, externalVocabulary->attributeValueTable + externalVocabulary->attributeValueTableSize, std::back_inserter(vocabulary.attributeValueTable));
                }
                if (externalVocabulary->charactersTable) {
                    std::copy(externalVocabulary->charactersTable, externalVocabulary->charactersTable + externalVocabulary->charactersTableSize, std::back_inserter(vocabulary.charactersTable));
                }
                if (externalVocabulary->otherStringTable) {
                    std::copy(externalVocabulary->otherStringTable, externalVocabulary->otherStringTable + externalVocabulary->otherStringTableSize, std::back_inserter(vocabulary.otherStringTable));
                }
                if (externalVocabulary->elementNameTable) {
                    std::copy(externalVocabulary->elementNameTable, externalVocabulary->elementNameTable + externalVocabulary->elementNameTableSize, std::back_inserter(vocabulary.elementNameTable));
                }
                if (externalVocabulary->attributeNameTable) {
                    std::copy(externalVocabulary->attributeNameTable, externalVocabulary->attributeNameTable + externalVocabulary->attributeNameTableSize, std::back_inserter(vocabulary.attributeNameTable));
                }
            }
            if (b1 & 0x0800) {
                // Parse restricted alphabets (C.2.5.3)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.restrictedAlphabetTable.push_back(parseNonEmptyOctetString2());
                }
            }
            if (b1 & 0x0400) {
                // Parse encoding algorithms (C.2.5.3)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.encodingAlgorithmTable.push_back(parseNonEmptyOctetString2());
                }
            }
            if (b1 & 0x0200) {
                // Parse prefixes (C.2.5.3)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.prefixTable.push_back(parseNonEmptyOctetString2());
                }
            }
            if (b1 & 0x0100) {
                // Parse namespace names (C.2.5.3)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.namespaceNameTable.push_back(parseNonEmptyOctetString2());
                }
            }
            if (b1 & 0x0080) {
                // Parse local names (C.2.5.3)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.localNameTable.push_back(parseNonEmptyOctetString2());
                }
            }
            if (b1 & 0x0040) {
                // Parse other ncnames (C.2.5.3)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.otherNCNameTable.push_back(parseNonEmptyOctetString2());
                }
            }
            if (b1 & 0x0020) {
                // Parse other uris (C.2.5.3)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.otherURITable.push_back(parseNonEmptyOctetString2());
                }
            }
            if (b1 & 0x0010) {
                // Parse attribute values (C.2.5.4)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.attributeValueTable.push_back(parseEncodedCharacterString3());
                }
            }
            if (b1 & 0x0008) {
                // Parse content character chunks (C.2.5.4)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.charactersTable.push_back(parseEncodedCharacterString3());
                }
            }
            if (b1 & 0x0004) {
                // Parse other strings (C.2.5.4)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    if (dataEnd - dataP < 1) {
                        throw DeadlyImportError(parseErrorMessage);
                    }
                    vocabulary.otherStringTable.push_back(parseEncodedCharacterString3());
                }
            }
            if (b1 & 0x0002) {
                // Parse element name surrogates (C.2.5.5)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    vocabulary.elementNameTable.push_back(parseNameSurrogate());
                }
            }
            if (b1 & 0x0001) {
                // Parse attribute name surrogates (C.2.5.5)
                for (size_t len = parseSequenceLen(); len > 0; --len) {
                    vocabulary.attributeNameTable.push_back(parseNameSurrogate());
                }
            }
        }
        if (b & 0x10) {
            // Parse notations (C.2.6)
            for (;;) {
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                uint8_t b1 = *dataP++;
                if (b1 == 0xf0) {
                    break;
                }
                if ((b1 & 0xfc) != 0xc0) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                /* C.11 */
                /*const std::string &name =*/ parseIdentifyingStringOrIndex(vocabulary.otherNCNameTable);
                if (b1 & 0x02) {
                    /*const std::string &systemId =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                }
                if (b1 & 0x01) {
                    /*const std::string &publicId =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                }
            }
        }
        if (b & 0x08) {
            // Parse unparsed entities (C.2.7)
            for (;;) {
                if (dataEnd - dataP < 1) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                uint8_t b1 = *dataP++;
                if (b1 == 0xf0) {
                    break;
                }
                if ((b1 & 0xfe) != 0xd0) {
                    throw DeadlyImportError(parseErrorMessage);
                }
                /* C.10 */
                /*const std::string &name =*/ parseIdentifyingStringOrIndex(vocabulary.otherNCNameTable);
                /*const std::string &systemId =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                if (b1 & 0x01) {
                    /*const std::string &publicId =*/ parseIdentifyingStringOrIndex(vocabulary.otherURITable);
                }
                /*const std::string &notationName =*/ parseIdentifyingStringOrIndex(vocabulary.otherNCNameTable);
            }
        }
        if (b & 0x04) {
            // Parse character encoding scheme (C.2.8)
            if (dataEnd - dataP < 1) {
                throw DeadlyImportError(parseErrorMessage);
            }
            /*std::string characterEncodingScheme =*/ parseNonEmptyOctetString2();
        }
        if (b & 0x02) {
            // Parse standalone flag (C.2.9)
            if (dataEnd - dataP < 1) {
                throw DeadlyImportError(parseErrorMessage);
            }
            uint8_t b1 = *dataP++;
            if (b1 & 0xfe) {
                throw DeadlyImportError(parseErrorMessage);
            }
            //bool standalone = b1 & 0x01;
        }
        if (b & 0x01) {
            // Parse version (C.2.10)
            if (dataEnd - dataP < 1) {
                throw DeadlyImportError(parseErrorMessage);
            }
            /*std::shared_ptr<const FIValue> version =*/ parseNonIdentifyingStringOrIndex1(vocabulary.otherStringTable);
        }
    }

    std::unique_ptr<uint8_t[]> data;
    uint8_t *dataP, *dataEnd;
    irr::io::EXML_NODE currentNodeType;
    bool emptyElement;
    bool headerPending;
    bool terminatorPending;
    Vocabulary vocabulary;
    std::vector<Attribute> attributes;
    std::stack<std::string> elementStack;
    std::string nodeName;
    std::map<std::string, std::unique_ptr<FIDecoder>> decoderMap;
    std::map<std::string, const FIVocabulary*> vocabularyMap;

    static const std::string EmptyString;
    static std::shared_ptr<const FIValue> EmptyFIString;

    static FIHexDecoder hexDecoder;
    static FIBase64Decoder base64Decoder;
    static FIShortDecoder shortDecoder;
    static FIIntDecoder intDecoder;
    static FILongDecoder longDecoder;
    static FIBoolDecoder boolDecoder;
    static FIFloatDecoder floatDecoder;
    static FIDoubleDecoder doubleDecoder;
    static FIUUIDDecoder uuidDecoder;
    static FICDATADecoder cdataDecoder;
    static FIDecoder *defaultDecoder[32];
};

const std::string CFIReaderImpl::EmptyString;
std::shared_ptr<const FIValue> CFIReaderImpl::EmptyFIString = FIStringValue::create(std::string());

FIHexDecoder CFIReaderImpl::hexDecoder;
FIBase64Decoder CFIReaderImpl::base64Decoder;
FIShortDecoder CFIReaderImpl::shortDecoder;
FIIntDecoder CFIReaderImpl::intDecoder;
FILongDecoder CFIReaderImpl::longDecoder;
FIBoolDecoder CFIReaderImpl::boolDecoder;
FIFloatDecoder CFIReaderImpl::floatDecoder;
FIDoubleDecoder CFIReaderImpl::doubleDecoder;
FIUUIDDecoder CFIReaderImpl::uuidDecoder;
FICDATADecoder CFIReaderImpl::cdataDecoder;

FIDecoder *CFIReaderImpl::defaultDecoder[32] = {
    &hexDecoder,
    &base64Decoder,
    &shortDecoder,
    &intDecoder,
    &longDecoder,
    &boolDecoder,
    &floatDecoder,
    &doubleDecoder,
    &uuidDecoder,
    &cdataDecoder
};

class CXMLReaderImpl : public FIReader
{
public:

    //! Constructor
    CXMLReaderImpl(std::unique_ptr<irr::io::IIrrXMLReader<char, irr::io::IXMLBase>> reader_)
    : reader(std::move(reader_))
    {}

    virtual ~CXMLReaderImpl() {}

    virtual bool read() /*override*/ {
        return reader->read();
    }

    virtual irr::io::EXML_NODE getNodeType() const /*override*/ {
        return reader->getNodeType();
    }

    virtual int getAttributeCount() const /*override*/ {
        return reader->getAttributeCount();
    }

    virtual const char* getAttributeName(int idx) const /*override*/ {
        return reader->getAttributeName(idx);
    }

    virtual const char* getAttributeValue(int idx) const /*override*/ {
        return reader->getAttributeValue(idx);
    }

    virtual const char* getAttributeValue(const char* name) const /*override*/ {
        return reader->getAttributeValue(name);
    }

    virtual const char* getAttributeValueSafe(const char* name) const /*override*/ {
        return reader->getAttributeValueSafe(name);
    }

    virtual int getAttributeValueAsInt(const char* name) const /*override*/ {
        return reader->getAttributeValueAsInt(name);
    }

    virtual int getAttributeValueAsInt(int idx) const /*override*/ {
        return reader->getAttributeValueAsInt(idx);
    }

    virtual float getAttributeValueAsFloat(const char* name) const /*override*/ {
        return reader->getAttributeValueAsFloat(name);
    }

    virtual float getAttributeValueAsFloat(int idx) const /*override*/ {
        return reader->getAttributeValueAsFloat(idx);
    }

    virtual const char* getNodeName() const /*override*/ {
        return reader->getNodeName();
    }

    virtual const char* getNodeData() const /*override*/ {
        return reader->getNodeData();
    }

    virtual bool isEmptyElement() const /*override*/ {
        return reader->isEmptyElement();
    }

    virtual irr::io::ETEXT_FORMAT getSourceFormat() const /*override*/ {
        return reader->getSourceFormat();
    }

    virtual irr::io::ETEXT_FORMAT getParserFormat() const /*override*/ {
        return reader->getParserFormat();
    }

    virtual std::shared_ptr<const FIValue> getAttributeEncodedValue(int /*idx*/) const /*override*/ {
        return nullptr;
    }

    virtual std::shared_ptr<const FIValue> getAttributeEncodedValue(const char* /*name*/) const /*override*/ {
        return nullptr;
    }

    virtual void registerDecoder(const std::string & /*algorithmUri*/, std::unique_ptr<FIDecoder> /*decoder*/) /*override*/ {}


    virtual void registerVocabulary(const std::string &/*vocabularyUri*/, const FIVocabulary * /*vocabulary*/) /*override*/ {}

private:

    std::unique_ptr<irr::io::IIrrXMLReader<char, irr::io::IXMLBase>> reader;
};

static std::unique_ptr<uint8_t[]> readFile(IOStream *stream, size_t &size, bool &isFI) {
    size = stream->FileSize();
    std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
    if (stream->Read(data.get(), size, 1) != 1) {
        size = 0;
        data.reset();
    }
    isFI = parseMagic(data.get(), data.get() + size) > 0;
    return data;
}

std::unique_ptr<FIReader> FIReader::create(IOStream *stream)
{
    size_t size;
    bool isFI;
    auto data = readFile(stream, size, isFI);
    if (isFI) {
        return std::unique_ptr<FIReader>(new CFIReaderImpl(std::move(data), size));
    }
    else {
        auto memios = std::unique_ptr<MemoryIOStream>(new MemoryIOStream(data.release(), size, true));
        auto callback = std::unique_ptr<CIrrXML_IOStreamReader>(new CIrrXML_IOStreamReader(memios.get()));
        return std::unique_ptr<FIReader>(new CXMLReaderImpl(std::unique_ptr<irr::io::IIrrXMLReader<char, irr::io::IXMLBase>>(createIrrXMLReader(callback.get()))));
    }
}

}// namespace Assimp

#endif // !ASSIMP_BUILD_NO_X3D_IMPORTER
