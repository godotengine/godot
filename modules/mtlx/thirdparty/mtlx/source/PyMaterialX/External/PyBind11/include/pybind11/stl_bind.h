/*
    pybind11/std_bind.h: Binding generators for STL data types

    Copyright (c) 2016 Sergey Lyskov and Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "detail/type_caster_base.h"
#include "cast.h"
#include "operators.h"

#include <algorithm>
#include <sstream>
#include <type_traits>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/* SFINAE helper class used by 'is_comparable */
template <typename T>
struct container_traits {
    template <typename T2>
    static std::true_type
    test_comparable(decltype(std::declval<const T2 &>() == std::declval<const T2 &>()) *);
    template <typename T2>
    static std::false_type test_comparable(...);
    template <typename T2>
    static std::true_type test_value(typename T2::value_type *);
    template <typename T2>
    static std::false_type test_value(...);
    template <typename T2>
    static std::true_type test_pair(typename T2::first_type *, typename T2::second_type *);
    template <typename T2>
    static std::false_type test_pair(...);

    static constexpr const bool is_comparable
        = std::is_same<std::true_type, decltype(test_comparable<T>(nullptr))>::value;
    static constexpr const bool is_pair
        = std::is_same<std::true_type, decltype(test_pair<T>(nullptr, nullptr))>::value;
    static constexpr const bool is_vector
        = std::is_same<std::true_type, decltype(test_value<T>(nullptr))>::value;
    static constexpr const bool is_element = !is_pair && !is_vector;
};

/* Default: is_comparable -> std::false_type */
template <typename T, typename SFINAE = void>
struct is_comparable : std::false_type {};

/* For non-map data structures, check whether operator== can be instantiated */
template <typename T>
struct is_comparable<
    T,
    enable_if_t<container_traits<T>::is_element && container_traits<T>::is_comparable>>
    : std::true_type {};

/* For a vector/map data structure, recursively check the value type
   (which is std::pair for maps) */
template <typename T>
struct is_comparable<T, enable_if_t<container_traits<T>::is_vector>> {
    static constexpr const bool value = is_comparable<typename T::value_type>::value;
};

/* For pairs, recursively check the two data types */
template <typename T>
struct is_comparable<T, enable_if_t<container_traits<T>::is_pair>> {
    static constexpr const bool value = is_comparable<typename T::first_type>::value
                                        && is_comparable<typename T::second_type>::value;
};

/* Fallback functions */
template <typename, typename, typename... Args>
void vector_if_copy_constructible(const Args &...) {}
template <typename, typename, typename... Args>
void vector_if_equal_operator(const Args &...) {}
template <typename, typename, typename... Args>
void vector_if_insertion_operator(const Args &...) {}
template <typename, typename, typename... Args>
void vector_modifiers(const Args &...) {}

template <typename Vector, typename Class_>
void vector_if_copy_constructible(enable_if_t<is_copy_constructible<Vector>::value, Class_> &cl) {
    cl.def(init<const Vector &>(), "Copy constructor");
}

template <typename Vector, typename Class_>
void vector_if_equal_operator(enable_if_t<is_comparable<Vector>::value, Class_> &cl) {
    using T = typename Vector::value_type;

    cl.def(self == self);
    cl.def(self != self);

    cl.def(
        "count",
        [](const Vector &v, const T &x) { return std::count(v.begin(), v.end(), x); },
        arg("x"),
        "Return the number of times ``x`` appears in the list");

    cl.def(
        "remove",
        [](Vector &v, const T &x) {
            auto p = std::find(v.begin(), v.end(), x);
            if (p != v.end()) {
                v.erase(p);
            } else {
                throw value_error();
            }
        },
        arg("x"),
        "Remove the first item from the list whose value is x. "
        "It is an error if there is no such item.");

    cl.def(
        "__contains__",
        [](const Vector &v, const T &x) { return std::find(v.begin(), v.end(), x) != v.end(); },
        arg("x"),
        "Return true the container contains ``x``");
}

// Vector modifiers -- requires a copyable vector_type:
// (Technically, some of these (pop and __delitem__) don't actually require copyability, but it
// seems silly to allow deletion but not insertion, so include them here too.)
template <typename Vector, typename Class_>
void vector_modifiers(
    enable_if_t<is_copy_constructible<typename Vector::value_type>::value, Class_> &cl) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;

    auto wrap_i = [](DiffType i, SizeType n) {
        if (i < 0) {
            i += n;
        }
        if (i < 0 || (SizeType) i >= n) {
            throw index_error();
        }
        return i;
    };

    cl.def(
        "append",
        [](Vector &v, const T &value) { v.push_back(value); },
        arg("x"),
        "Add an item to the end of the list");

    cl.def(init([](const iterable &it) {
        auto v = std::unique_ptr<Vector>(new Vector());
        v->reserve(len_hint(it));
        for (handle h : it) {
            v->push_back(h.cast<T>());
        }
        return v.release();
    }));

    cl.def(
        "clear", [](Vector &v) { v.clear(); }, "Clear the contents");

    cl.def(
        "extend",
        [](Vector &v, const Vector &src) { v.insert(v.end(), src.begin(), src.end()); },
        arg("L"),
        "Extend the list by appending all the items in the given list");

    cl.def(
        "extend",
        [](Vector &v, const iterable &it) {
            const size_t old_size = v.size();
            v.reserve(old_size + len_hint(it));
            try {
                for (handle h : it) {
                    v.push_back(h.cast<T>());
                }
            } catch (const cast_error &) {
                v.erase(v.begin() + static_cast<typename Vector::difference_type>(old_size),
                        v.end());
                try {
                    v.shrink_to_fit();
                } catch (const std::exception &) {
                    // Do nothing
                }
                throw;
            }
        },
        arg("L"),
        "Extend the list by appending all the items in the given list");

    cl.def(
        "insert",
        [](Vector &v, DiffType i, const T &x) {
            // Can't use wrap_i; i == v.size() is OK
            if (i < 0) {
                i += v.size();
            }
            if (i < 0 || (SizeType) i > v.size()) {
                throw index_error();
            }
            v.insert(v.begin() + i, x);
        },
        arg("i"),
        arg("x"),
        "Insert an item at a given position.");

    cl.def(
        "pop",
        [](Vector &v) {
            if (v.empty()) {
                throw index_error();
            }
            T t = std::move(v.back());
            v.pop_back();
            return t;
        },
        "Remove and return the last item");

    cl.def(
        "pop",
        [wrap_i](Vector &v, DiffType i) {
            i = wrap_i(i, v.size());
            T t = std::move(v[(SizeType) i]);
            v.erase(std::next(v.begin(), i));
            return t;
        },
        arg("i"),
        "Remove and return the item at index ``i``");

    cl.def("__setitem__", [wrap_i](Vector &v, DiffType i, const T &t) {
        i = wrap_i(i, v.size());
        v[(SizeType) i] = t;
    });

    /// Slicing protocol
    cl.def(
        "__getitem__",
        [](const Vector &v, const slice &slice) -> Vector * {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength)) {
                throw error_already_set();
            }

            auto *seq = new Vector();
            seq->reserve((size_t) slicelength);

            for (size_t i = 0; i < slicelength; ++i) {
                seq->push_back(v[start]);
                start += step;
            }
            return seq;
        },
        arg("s"),
        "Retrieve list elements using a slice object");

    cl.def(
        "__setitem__",
        [](Vector &v, const slice &slice, const Vector &value) {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength)) {
                throw error_already_set();
            }

            if (slicelength != value.size()) {
                throw std::runtime_error(
                    "Left and right hand size of slice assignment have different sizes!");
            }

            for (size_t i = 0; i < slicelength; ++i) {
                v[start] = value[i];
                start += step;
            }
        },
        "Assign list elements using a slice object");

    cl.def(
        "__delitem__",
        [wrap_i](Vector &v, DiffType i) {
            i = wrap_i(i, v.size());
            v.erase(v.begin() + i);
        },
        "Delete the list elements at index ``i``");

    cl.def(
        "__delitem__",
        [](Vector &v, const slice &slice) {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength)) {
                throw error_already_set();
            }

            if (step == 1 && false) {
                v.erase(v.begin() + (DiffType) start, v.begin() + DiffType(start + slicelength));
            } else {
                for (size_t i = 0; i < slicelength; ++i) {
                    v.erase(v.begin() + DiffType(start));
                    start += step - 1;
                }
            }
        },
        "Delete list elements using a slice object");
}

// If the type has an operator[] that doesn't return a reference (most notably std::vector<bool>),
// we have to access by copying; otherwise we return by reference.
template <typename Vector>
using vector_needs_copy
    = negation<std::is_same<decltype(std::declval<Vector>()[typename Vector::size_type()]),
                            typename Vector::value_type &>>;

// The usual case: access and iterate by reference
template <typename Vector, typename Class_>
void vector_accessor(enable_if_t<!vector_needs_copy<Vector>::value, Class_> &cl) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;
    using ItType = typename Vector::iterator;

    auto wrap_i = [](DiffType i, SizeType n) {
        if (i < 0) {
            i += n;
        }
        if (i < 0 || (SizeType) i >= n) {
            throw index_error();
        }
        return i;
    };

    cl.def(
        "__getitem__",
        [wrap_i](Vector &v, DiffType i) -> T & {
            i = wrap_i(i, v.size());
            return v[(SizeType) i];
        },
        return_value_policy::reference_internal // ref + keepalive
    );

    cl.def(
        "__iter__",
        [](Vector &v) {
            return make_iterator<return_value_policy::reference_internal, ItType, ItType, T &>(
                v.begin(), v.end());
        },
        keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );
}

// The case for special objects, like std::vector<bool>, that have to be returned-by-copy:
template <typename Vector, typename Class_>
void vector_accessor(enable_if_t<vector_needs_copy<Vector>::value, Class_> &cl) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;
    using ItType = typename Vector::iterator;
    cl.def("__getitem__", [](const Vector &v, DiffType i) -> T {
        if (i < 0 && (i += v.size()) < 0) {
            throw index_error();
        }
        if ((SizeType) i >= v.size()) {
            throw index_error();
        }
        return v[(SizeType) i];
    });

    cl.def(
        "__iter__",
        [](Vector &v) {
            return make_iterator<return_value_policy::copy, ItType, ItType, T>(v.begin(), v.end());
        },
        keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );
}

template <typename Vector, typename Class_>
auto vector_if_insertion_operator(Class_ &cl, std::string const &name)
    -> decltype(std::declval<std::ostream &>() << std::declval<typename Vector::value_type>(),
                void()) {
    using size_type = typename Vector::size_type;

    cl.def(
        "__repr__",
        [name](Vector &v) {
            std::ostringstream s;
            s << name << '[';
            for (size_type i = 0; i < v.size(); ++i) {
                s << v[i];
                if (i != v.size() - 1) {
                    s << ", ";
                }
            }
            s << ']';
            return s.str();
        },
        "Return the canonical string representation of this list.");
}

// Provide the buffer interface for vectors if we have data() and we have a format for it
// GCC seems to have "void std::vector<bool>::data()" - doing SFINAE on the existence of data()
// is insufficient, we need to check it returns an appropriate pointer
template <typename Vector, typename = void>
struct vector_has_data_and_format : std::false_type {};
template <typename Vector>
struct vector_has_data_and_format<
    Vector,
    enable_if_t<std::is_same<decltype(format_descriptor<typename Vector::value_type>::format(),
                                      std::declval<Vector>().data()),
                             typename Vector::value_type *>::value>> : std::true_type {};

// [workaround(intel)] Separate function required here
// Workaround as the Intel compiler does not compile the enable_if_t part below
// (tested with icc (ICC) 2021.1 Beta 20200827)
template <typename... Args>
constexpr bool args_any_are_buffer() {
    return detail::any_of<std::is_same<Args, buffer_protocol>...>::value;
}

// [workaround(intel)] Separate function required here
// [workaround(msvc)] Can't use constexpr bool in return type

// Add the buffer interface to a vector
template <typename Vector, typename Class_, typename... Args>
void vector_buffer_impl(Class_ &cl, std::true_type) {
    using T = typename Vector::value_type;

    static_assert(vector_has_data_and_format<Vector>::value,
                  "There is not an appropriate format descriptor for this vector");

    // numpy.h declares this for arbitrary types, but it may raise an exception and crash hard
    // at runtime if PYBIND11_NUMPY_DTYPE hasn't been called, so check here
    format_descriptor<T>::format();

    cl.def_buffer([](Vector &v) -> buffer_info {
        return buffer_info(v.data(),
                           static_cast<ssize_t>(sizeof(T)),
                           format_descriptor<T>::format(),
                           1,
                           {v.size()},
                           {sizeof(T)});
    });

    cl.def(init([](const buffer &buf) {
        auto info = buf.request();
        if (info.ndim != 1 || info.strides[0] % static_cast<ssize_t>(sizeof(T))) {
            throw type_error("Only valid 1D buffers can be copied to a vector");
        }
        if (!detail::compare_buffer_info<T>::compare(info)
            || (ssize_t) sizeof(T) != info.itemsize) {
            throw type_error("Format mismatch (Python: " + info.format
                             + " C++: " + format_descriptor<T>::format() + ")");
        }

        T *p = static_cast<T *>(info.ptr);
        ssize_t step = info.strides[0] / static_cast<ssize_t>(sizeof(T));
        T *end = p + info.shape[0] * step;
        if (step == 1) {
            return Vector(p, end);
        }
        Vector vec;
        vec.reserve((size_t) info.shape[0]);
        for (; p != end; p += step) {
            vec.push_back(*p);
        }
        return vec;
    }));

    return;
}

template <typename Vector, typename Class_, typename... Args>
void vector_buffer_impl(Class_ &, std::false_type) {}

template <typename Vector, typename Class_, typename... Args>
void vector_buffer(Class_ &cl) {
    vector_buffer_impl<Vector, Class_, Args...>(
        cl, detail::any_of<std::is_same<Args, buffer_protocol>...>{});
}

PYBIND11_NAMESPACE_END(detail)

//
// std::vector
//
template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
class_<Vector, holder_type> bind_vector(handle scope, std::string const &name, Args &&...args) {
    using Class_ = class_<Vector, holder_type>;

    // If the value_type is unregistered (e.g. a converting type) or is itself registered
    // module-local then make the vector binding module-local as well:
    using vtype = typename Vector::value_type;
    auto *vtype_info = detail::get_type_info(typeid(vtype));
    bool local = !vtype_info || vtype_info->module_local;

    Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

    // Declare the buffer interface if a buffer_protocol() is passed in
    detail::vector_buffer<Vector, Class_, Args...>(cl);

    cl.def(init<>());

    // Register copy constructor (if possible)
    detail::vector_if_copy_constructible<Vector, Class_>(cl);

    // Register comparison-related operators and functions (if possible)
    detail::vector_if_equal_operator<Vector, Class_>(cl);

    // Register stream insertion operator (if possible)
    detail::vector_if_insertion_operator<Vector, Class_>(cl, name);

    // Modifiers require copyable vector value type
    detail::vector_modifiers<Vector, Class_>(cl);

    // Accessor and iterator; return by value if copyable, otherwise we return by ref + keep-alive
    detail::vector_accessor<Vector, Class_>(cl);

    cl.def(
        "__bool__",
        [](const Vector &v) -> bool { return !v.empty(); },
        "Check whether the list is nonempty");

    cl.def("__len__", &Vector::size);

#if 0
    // C++ style functions deprecated, leaving it here as an example
    cl.def(init<size_type>());

    cl.def("resize",
         (void (Vector::*) (size_type count)) & Vector::resize,
         "changes the number of elements stored");

    cl.def("erase",
        [](Vector &v, SizeType i) {
        if (i >= v.size())
            throw index_error();
        v.erase(v.begin() + i);
    }, "erases element at index ``i``");

    cl.def("empty",         &Vector::empty,         "checks whether the container is empty");
    cl.def("size",          &Vector::size,          "returns the number of elements");
    cl.def("push_back", (void (Vector::*)(const T&)) &Vector::push_back, "adds an element to the end");
    cl.def("pop_back",                               &Vector::pop_back, "removes the last element");

    cl.def("max_size",      &Vector::max_size,      "returns the maximum possible number of elements");
    cl.def("reserve",       &Vector::reserve,       "reserves storage");
    cl.def("capacity",      &Vector::capacity,      "returns the number of elements that can be held in currently allocated storage");
    cl.def("shrink_to_fit", &Vector::shrink_to_fit, "reduces memory usage by freeing unused memory");

    cl.def("clear", &Vector::clear, "clears the contents");
    cl.def("swap",   &Vector::swap, "swaps the contents");

    cl.def("front", [](Vector &v) {
        if (v.size()) return v.front();
        else throw index_error();
    }, "access the first element");

    cl.def("back", [](Vector &v) {
        if (v.size()) return v.back();
        else throw index_error();
    }, "access the last element ");

#endif

    return cl;
}

//
// std::map, std::unordered_map
//

PYBIND11_NAMESPACE_BEGIN(detail)

/* Fallback functions */
template <typename, typename, typename... Args>
void map_if_insertion_operator(const Args &...) {}
template <typename, typename, typename... Args>
void map_assignment(const Args &...) {}

// Map assignment when copy-assignable: just copy the value
template <typename Map, typename Class_>
void map_assignment(
    enable_if_t<is_copy_assignable<typename Map::mapped_type>::value, Class_> &cl) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;

    cl.def("__setitem__", [](Map &m, const KeyType &k, const MappedType &v) {
        auto it = m.find(k);
        if (it != m.end()) {
            it->second = v;
        } else {
            m.emplace(k, v);
        }
    });
}

// Not copy-assignable, but still copy-constructible: we can update the value by erasing and
// reinserting
template <typename Map, typename Class_>
void map_assignment(enable_if_t<!is_copy_assignable<typename Map::mapped_type>::value
                                    && is_copy_constructible<typename Map::mapped_type>::value,
                                Class_> &cl) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;

    cl.def("__setitem__", [](Map &m, const KeyType &k, const MappedType &v) {
        // We can't use m[k] = v; because value type might not be default constructable
        auto r = m.emplace(k, v);
        if (!r.second) {
            // value type is not copy assignable so the only way to insert it is to erase it
            // first...
            m.erase(r.first);
            m.emplace(k, v);
        }
    });
}

template <typename Map, typename Class_>
auto map_if_insertion_operator(Class_ &cl, std::string const &name)
    -> decltype(std::declval<std::ostream &>() << std::declval<typename Map::key_type>()
                                               << std::declval<typename Map::mapped_type>(),
                void()) {

    cl.def(
        "__repr__",
        [name](Map &m) {
            std::ostringstream s;
            s << name << '{';
            bool f = false;
            for (auto const &kv : m) {
                if (f) {
                    s << ", ";
                }
                s << kv.first << ": " << kv.second;
                f = true;
            }
            s << '}';
            return s.str();
        },
        "Return the canonical string representation of this map.");
}

template <typename KeyType>
struct keys_view {
    virtual size_t len() = 0;
    virtual iterator iter() = 0;
    virtual bool contains(const KeyType &k) = 0;
    virtual bool contains(const object &k) = 0;
    virtual ~keys_view() = default;
};

template <typename MappedType>
struct values_view {
    virtual size_t len() = 0;
    virtual iterator iter() = 0;
    virtual ~values_view() = default;
};

template <typename KeyType, typename MappedType>
struct items_view {
    virtual size_t len() = 0;
    virtual iterator iter() = 0;
    virtual ~items_view() = default;
};

template <typename Map, typename KeysView>
struct KeysViewImpl : public KeysView {
    explicit KeysViewImpl(Map &map) : map(map) {}
    size_t len() override { return map.size(); }
    iterator iter() override { return make_key_iterator(map.begin(), map.end()); }
    bool contains(const typename Map::key_type &k) override { return map.find(k) != map.end(); }
    bool contains(const object &) override { return false; }
    Map &map;
};

template <typename Map, typename ValuesView>
struct ValuesViewImpl : public ValuesView {
    explicit ValuesViewImpl(Map &map) : map(map) {}
    size_t len() override { return map.size(); }
    iterator iter() override { return make_value_iterator(map.begin(), map.end()); }
    Map &map;
};

template <typename Map, typename ItemsView>
struct ItemsViewImpl : public ItemsView {
    explicit ItemsViewImpl(Map &map) : map(map) {}
    size_t len() override { return map.size(); }
    iterator iter() override { return make_iterator(map.begin(), map.end()); }
    Map &map;
};

PYBIND11_NAMESPACE_END(detail)

template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
class_<Map, holder_type> bind_map(handle scope, const std::string &name, Args &&...args) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;
    using StrippedKeyType = detail::remove_cvref_t<KeyType>;
    using StrippedMappedType = detail::remove_cvref_t<MappedType>;
    using KeysView = detail::keys_view<StrippedKeyType>;
    using ValuesView = detail::values_view<StrippedMappedType>;
    using ItemsView = detail::items_view<StrippedKeyType, StrippedMappedType>;
    using Class_ = class_<Map, holder_type>;

    // If either type is a non-module-local bound type then make the map binding non-local as well;
    // otherwise (e.g. both types are either module-local or converting) the map will be
    // module-local.
    auto *tinfo = detail::get_type_info(typeid(MappedType));
    bool local = !tinfo || tinfo->module_local;
    if (local) {
        tinfo = detail::get_type_info(typeid(KeyType));
        local = !tinfo || tinfo->module_local;
    }

    Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);
    static constexpr auto key_type_descr = detail::make_caster<KeyType>::name;
    static constexpr auto mapped_type_descr = detail::make_caster<MappedType>::name;
    std::string key_type_name(key_type_descr.text), mapped_type_name(mapped_type_descr.text);

    // If key type isn't properly wrapped, fall back to C++ names
    if (key_type_name == "%") {
        key_type_name = detail::type_info_description(typeid(KeyType));
    }
    // Similarly for value type:
    if (mapped_type_name == "%") {
        mapped_type_name = detail::type_info_description(typeid(MappedType));
    }

    // Wrap KeysView[KeyType] if it wasn't already wrapped
    if (!detail::get_type_info(typeid(KeysView))) {
        class_<KeysView> keys_view(
            scope, ("KeysView[" + key_type_name + "]").c_str(), pybind11::module_local(local));
        keys_view.def("__len__", &KeysView::len);
        keys_view.def("__iter__",
                      &KeysView::iter,
                      keep_alive<0, 1>() /* Essential: keep view alive while iterator exists */
        );
        keys_view.def("__contains__",
                      static_cast<bool (KeysView::*)(const KeyType &)>(&KeysView::contains));
        // Fallback for when the object is not of the key type
        keys_view.def("__contains__",
                      static_cast<bool (KeysView::*)(const object &)>(&KeysView::contains));
    }
    // Similarly for ValuesView:
    if (!detail::get_type_info(typeid(ValuesView))) {
        class_<ValuesView> values_view(scope,
                                       ("ValuesView[" + mapped_type_name + "]").c_str(),
                                       pybind11::module_local(local));
        values_view.def("__len__", &ValuesView::len);
        values_view.def("__iter__",
                        &ValuesView::iter,
                        keep_alive<0, 1>() /* Essential: keep view alive while iterator exists */
        );
    }
    // Similarly for ItemsView:
    if (!detail::get_type_info(typeid(ItemsView))) {
        class_<ItemsView> items_view(
            scope,
            ("ItemsView[" + key_type_name + ", ").append(mapped_type_name + "]").c_str(),
            pybind11::module_local(local));
        items_view.def("__len__", &ItemsView::len);
        items_view.def("__iter__",
                       &ItemsView::iter,
                       keep_alive<0, 1>() /* Essential: keep view alive while iterator exists */
        );
    }

    cl.def(init<>());

    // Register stream insertion operator (if possible)
    detail::map_if_insertion_operator<Map, Class_>(cl, name);

    cl.def(
        "__bool__",
        [](const Map &m) -> bool { return !m.empty(); },
        "Check whether the map is nonempty");

    cl.def(
        "__iter__",
        [](Map &m) { return make_key_iterator(m.begin(), m.end()); },
        keep_alive<0, 1>() /* Essential: keep map alive while iterator exists */
    );

    cl.def(
        "keys",
        [](Map &m) {
            return std::unique_ptr<KeysView>(new detail::KeysViewImpl<Map, KeysView>(m));
        },
        keep_alive<0, 1>() /* Essential: keep map alive while view exists */
    );

    cl.def(
        "values",
        [](Map &m) {
            return std::unique_ptr<ValuesView>(new detail::ValuesViewImpl<Map, ValuesView>(m));
        },
        keep_alive<0, 1>() /* Essential: keep map alive while view exists */
    );

    cl.def(
        "items",
        [](Map &m) {
            return std::unique_ptr<ItemsView>(new detail::ItemsViewImpl<Map, ItemsView>(m));
        },
        keep_alive<0, 1>() /* Essential: keep map alive while view exists */
    );

    cl.def(
        "__getitem__",
        [](Map &m, const KeyType &k) -> MappedType & {
            auto it = m.find(k);
            if (it == m.end()) {
                throw key_error();
            }
            return it->second;
        },
        return_value_policy::reference_internal // ref + keepalive
    );

    cl.def("__contains__", [](Map &m, const KeyType &k) -> bool {
        auto it = m.find(k);
        if (it == m.end()) {
            return false;
        }
        return true;
    });
    // Fallback for when the object is not of the key type
    cl.def("__contains__", [](Map &, const object &) -> bool { return false; });

    // Assignment provided only if the type is copyable
    detail::map_assignment<Map, Class_>(cl);

    cl.def("__delitem__", [](Map &m, const KeyType &k) {
        auto it = m.find(k);
        if (it == m.end()) {
            throw key_error();
        }
        m.erase(it);
    });

    cl.def("__len__", &Map::size);

    return cl;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
