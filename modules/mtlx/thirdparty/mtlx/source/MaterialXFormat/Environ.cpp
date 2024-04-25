//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXFormat/Environ.h>

#include <MaterialXCore/Util.h>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

MATERIALX_NAMESPACE_BEGIN

string getEnviron(const string& name)
{
#if defined(_WIN32)
    if (uint32_t size = GetEnvironmentVariableA(name.c_str(), nullptr, 0))
    {
        vector<char> buffer(size);
        GetEnvironmentVariableA(name.c_str(), buffer.data(), size);
        return string(buffer.data());
    }
#else
    if (const char* const result = getenv(name.c_str()))
    {
        return string(result);
    }
#endif
    return EMPTY_STRING;
}

bool setEnviron(const string& name, const string& value)
{
#if defined(_WIN32)
    return SetEnvironmentVariableA(name.c_str(), value.c_str()) != 0;
#else
    return setenv(name.c_str(), value.c_str(), true);
#endif
}

bool removeEnviron(const string& name)
{
#if defined(_WIN32)
    return SetEnvironmentVariableA(name.c_str(), nullptr) != 0;
#else
    return unsetenv(name.c_str()) == 0;
#endif
}

MATERIALX_NAMESPACE_END
