//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXFormat/File.h>
#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

template<typename T> struct IsStrContainer : std::false_type {};
template<> struct IsStrContainer<mx::FileSearchPath> : std::true_type {};
template<> struct IsStrContainer<mx::FilePath> : std::true_type {};

using StrContainerIntermediate = std::string;

namespace emscripten 
{

namespace internal 
{

template<typename T>
struct TypeID<T, typename std::enable_if<IsStrContainer<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value, void>::type> {
  static constexpr TYPEID get() {
    return TypeID<StrContainerIntermediate>::get();
  }
};

template<typename T>
struct BindingType<T, typename std::enable_if<IsStrContainer<T>::value, void>::type> {
  typedef typename BindingType<StrContainerIntermediate>::WireType WireType;

  constexpr static WireType toWireType(const T& v) {
    return BindingType<StrContainerIntermediate>::toWireType(v.asString());
  }
  constexpr static T fromWireType(WireType v) {
    return T(BindingType<StrContainerIntermediate>::fromWireType(v));
  }
};

} // namespace internal

} // namespace emscripten
