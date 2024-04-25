//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_FACTORY_H
#define MATERIALX_FACTORY_H

/// @file
/// Class instantiator factory helper class

#include <MaterialXCore/Library.h>

MATERIALX_NAMESPACE_BEGIN

/// @class Factory
/// Factory class for creating instances of classes given their type name.
template <class T> class Factory
{
  public:
    using Ptr = shared_ptr<T>;
    using CreatorFunction = Ptr (*)();
    using CreatorMap = std::unordered_map<string, CreatorFunction>;

    /// Register a new class given a unique type name
    /// and a creator function for the class.
    void registerClass(const string& typeName, CreatorFunction f)
    {
        _creatorMap[typeName] = f;
    }

    /// Determine if a class has been registered for a type name
    bool classRegistered(const string& typeName) const
    {
        return _creatorMap.find(typeName) != _creatorMap.end();
    }

    /// Unregister a registered class
    void unregisterClass(const string& typeName)
    {
        auto it = _creatorMap.find(typeName);
        if (it != _creatorMap.end())
        {
            _creatorMap.erase(it);
        }
    }

    /// Create a new instance of the class with given type name.
    /// Returns nullptr if no class with given name is registered.
    Ptr create(const string& typeName) const
    {
        auto it = _creatorMap.find(typeName);
        return (it != _creatorMap.end() ? it->second() : nullptr);
    }

  private:
    CreatorMap _creatorMap;
};

MATERIALX_NAMESPACE_END

#endif // MATERIALX_FACTORY_H
