//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Types.h>

#include <cctype>

MATERIALX_NAMESPACE_BEGIN

const string EMPTY_STRING;

namespace
{

const string LIBRARY_VERSION_STRING = std::to_string(MATERIALX_MAJOR_VERSION) + "." +
                                      std::to_string(MATERIALX_MINOR_VERSION) + "." +
                                      std::to_string(MATERIALX_BUILD_VERSION);

const std::tuple<int, int, int> LIBRARY_VERSION_TUPLE(MATERIALX_MAJOR_VERSION,
                                                      MATERIALX_MINOR_VERSION,
                                                      MATERIALX_BUILD_VERSION);

bool invalidNameChar(char c)
{
    return !isalnum((unsigned char) c) && c != '_' && c != ':';
}

} // anonymous namespace

//
// Utility methods
//

string getVersionString()
{
    return LIBRARY_VERSION_STRING;
}

std::tuple<int, int, int> getVersionIntegers()
{
    return LIBRARY_VERSION_TUPLE;
}

string createValidName(string name, char replaceChar)
{
    std::replace_if(name.begin(), name.end(), invalidNameChar, replaceChar);
    return name;
}

bool isValidName(const string& name)
{
    auto it = std::find_if(name.begin(), name.end(), invalidNameChar);
    return it == name.end();
}

string incrementName(const string& name)
{
    size_t split = name.length();
    while (split > 0)
    {
        if (!isdigit(name[split - 1]))
            break;
        split--;
    }

    if (split < name.length())
    {
        string prefix = name.substr(0, split);
        string suffix = name.substr(split, name.length());
        return prefix + std::to_string(std::stoi(suffix) + 1);
    }
    return name + "2";
}

StringVec splitString(const string& str, const string& sep)
{
    StringVec split;

    string::size_type lastPos = str.find_first_not_of(sep, 0);
    string::size_type pos = str.find_first_of(sep, lastPos);

    while (pos != string::npos || lastPos != string::npos)
    {
        split.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(sep, pos);
        pos = str.find_first_of(sep, lastPos);
    }

    return split;
}

string joinStrings(const StringVec& stringVec, const string& sep)
{
    string res = stringVec.empty() ? EMPTY_STRING : stringVec[0];
    for (size_t i = 1; i < stringVec.size(); i++)
    {
        res += sep + stringVec[i];
    }
    return res;
}

string replaceSubstrings(string str, const StringMap& stringMap)
{
    for (const auto& pair : stringMap)
    {
        if (pair.first.empty())
            continue;

        size_t pos = 0;
        while ((pos = str.find(pair.first, pos)) != string::npos)
        {
            str.replace(pos, pair.first.length(), pair.second);
            pos += pair.second.length();
        }
    }
    return str;
}

string stringToLower(string str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c)
    {
        return (char) std::tolower(c);
    });
    return str;
}

bool stringStartsWith(const std::string& str, const std::string& prefix)
{
    if (str.length() >= prefix.length())
    {
        return !str.compare(0, prefix.length(), prefix);
    }
    return false;
}

bool stringEndsWith(const string& str, const string& suffix)
{
    if (str.length() >= suffix.length())
    {
        return !str.compare(str.length() - suffix.length(), suffix.length(), suffix);
    }
    return false;
}

string trimSpaces(const string& str)
{
    const char SPACE(' ');

    size_t start = str.find_first_not_of(SPACE);
    string result = (start == std::string::npos) ? EMPTY_STRING : str.substr(start);
    size_t end = result.find_last_not_of(SPACE);
    result = (end == std::string::npos) ? EMPTY_STRING : result.substr(0, end + 1);
    return result;
}

StringVec splitNamePath(const string& namePath)
{
    StringVec nameVec = splitString(namePath, NAME_PATH_SEPARATOR);
    return nameVec;
}

string createNamePath(const StringVec& nameVec)
{
    string res = joinStrings(nameVec, NAME_PATH_SEPARATOR);
    return res;
}

string parentNamePath(const string& namePath)
{
    StringVec nameVec = splitNamePath(namePath);
    if (!nameVec.empty())
    {
        nameVec.pop_back();
        return createNamePath(nameVec);
    }
    return EMPTY_STRING;
}

MATERIALX_NAMESPACE_END
