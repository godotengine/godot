// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

#pragma once
#include <assert.h>
#include <jni.h>
#include <jnipp.h>

namespace wrap {

/*!
 * Base class for types wrapping Java types.
 *
 * Derived types are encouraged to have a nested `struct Meta`, inheriting
 * publicly from MetaBaseDroppable or MetaBase, with a singleton accessor named
 * `data()`, and a private constructor (implemented in a .cpp file, not the
 * header) that populates jni::method_t, jni::field_t, etc. members for each
 * method, etc. of interest.
 */
class ObjectWrapperBase {
  public:
    /*!
     * Default constructor.
     */
    ObjectWrapperBase() = default;

    /*!
     * Construct from a jni::Object.
     */
    explicit ObjectWrapperBase(jni::Object obj) : obj_(std::move(obj)) {}

    /*!
     * Construct from a nullptr.
     */
    explicit ObjectWrapperBase(std::nullptr_t const &) : obj_() {}

    /*!
     * Evaluate if this is non-null
     */
    explicit operator bool() const noexcept { return !obj_.isNull(); }

    /*!
     * Is this object null?
     */
    bool isNull() const noexcept { return obj_.isNull(); }

    /*!
     * Get the wrapped jni::Object
     */
    jni::Object &object() noexcept { return obj_; }

    /*!
     * Get the wrapped jni::Object (const overload)
     */
    jni::Object const &object() const noexcept { return obj_; }

  private:
    jni::Object obj_;
};

/*!
 * Equality comparison for a wrapped Java object.
 */
static inline bool operator==(ObjectWrapperBase const &lhs,
                              ObjectWrapperBase const &rhs) noexcept {
    return lhs.object() == rhs.object();
}

/*!
 * Inequality comparison for a wrapped Java object.
 */
static inline bool operator!=(ObjectWrapperBase const &lhs,
                              ObjectWrapperBase const &rhs) noexcept {
    return lhs.object() != rhs.object();
}

/*!
 * Equality comparison between a wrapped Java object and @p nullptr.
 */
static inline bool operator==(ObjectWrapperBase const &obj,
                              std::nullptr_t) noexcept {
    return obj.isNull();
}

/*!
 * Equality comparison between a wrapped Java object and @p nullptr.
 */
static inline bool operator==(std::nullptr_t,
                              ObjectWrapperBase const &obj) noexcept {
    return obj.isNull();
}

/*!
 * Inequality comparison between a wrapped Java object and @p nullptr.
 */
static inline bool operator!=(ObjectWrapperBase const &obj,
                              std::nullptr_t) noexcept {
    return !(obj == nullptr);
}

/*!
 * Inequality comparison between a wrapped Java object and @p nullptr.
 */
static inline bool operator!=(std::nullptr_t,
                              ObjectWrapperBase const &obj) noexcept {
    return !(obj == nullptr);
}

/*!
 * Base class for Meta structs where you want the reference to the Class object
 * to persist (indefinitely).
 *
 * Mostly for classes that would stick around anyway (e.g.
 * @p java.lang.ClassLoader ) where many operations are on static
 * methods/fields. Use of a non-static method or field does not require such a
 * reference, use MetaBaseDroppable in those cases.
 */
class MetaBase {
  public:
    /*!
     * Gets a reference to the class object.
     *
     * Unlike MetaBaseDroppable, here we know that the class object ref is
     * alive.
     */
    jni::Class const &clazz() const noexcept { return clazz_; }
    /*!
     * Gets a reference to the class object.
     *
     * Provided here for parallel API to MetaBaseDroppable, despite being
     * synonymous with clazz() here.
     */
    jni::Class const &classRef() const noexcept { return clazz_; }

    /*!
     * Get the class name, with namespaces delimited by `/`.
     */
    const char *className() const noexcept { return classname_; }

  protected:
    /*!
     * Construct.
     *
     * @param classname The class name, fully qualified, with namespaces
     * delimited by `/`.
     * @param clazz The jclass object for the class in question, if known.
     */
    explicit MetaBase(const char *classname, jni::jclass clazz = nullptr)
        : classname_(classname), clazz_() {
        if (clazz != nullptr) {
            // The 0 makes it a global ref.
            clazz_ = jni::Class{clazz, 0};
        } else {
            clazz_ = jni::Class{classname};
        }
    }

  private:
    const char *classname_;
    jni::Class clazz_;
};

/*!
 * Base class for Meta structs where you don't need the reference to the Class
 * object to persist. (This is most uses.)
 */
class MetaBaseDroppable {
  public:
    /*!
     * Gets the class object.
     *
     * Works regardless of whether dropClassRef() has been called - it's just
     * slower if it has.
     */
    jni::Class clazz() const {
        if (clazz_.isNull()) {
            return {classname_};
        }
        return clazz_;
    }

    /*!
     * Get the class name, with namespaces delimited by `/`.
     */
    const char *className() const noexcept { return classname_; }

    /*!
     * May be called in/after the derived constructor, to drop the reference to
     * the class object if it's no longer needed.
     */
    void dropClassRef() { clazz_ = jni::Class{}; }

  protected:
    /*!
     * Construct.
     *
     * Once you are done constructing your derived struct, you may call
     * dropClassRef() and still safely use non-static method and field IDs
     * retrieved.
     *
     * @param classname The class name, fully qualified, with namespaces
     * delimited by `/`.
     * @param clazz The jclass object for the class in question, if known.
     */
    explicit MetaBaseDroppable(const char *classname,
                               jni::jclass clazz = nullptr)
        : classname_(classname), clazz_() {
        if (clazz != nullptr) {
            // The 0 makes it a global ref.
            clazz_ = jni::Class{clazz, 0};
        } else {
            clazz_ = jni::Class{classname};
        }
    }

    /*!
     * Gets a reference to the class object, but is non-null only if it's still
     * cached.
     *
     * Only for used in derived constructors/initializers, where you know you
     * haven't dropped this yet.
     */
    jni::Class const &classRef() const { return clazz_; }

  private:
    const char *classname_;
    jni::Class clazz_;
};

/*!
 * Implementation namespace for these JNI wrappers.
 *
 * They can be ignored if you aren't adding/extending wrappers.
 */
namespace impl {
/*!
 * Type-aware wrapper for a field ID.
 *
 * This is a smarter alternative to just using jni::field_t since it avoids
 * having to repeat the type in the accessor, without using any more storage.
 *
 * @see StaticFieldId for the equivalent for static fields.
 * @see WrappedFieldId for the equivalent for structures that we wrap.
 */
template <typename T> struct FieldId {
  public:
    FieldId(jni::Class const &clazz, const char *name)
        : id(clazz.getField<T>(name)) {}

    const jni::field_t id;
};

/*!
 * Get the value of field @p field in Java object @p obj.
 *
 * This is found by argument-dependent lookup and can be used unqualified.
 *
 * @relates FieldId
 */
template <typename T>
static inline T get(FieldId<T> const &field, jni::Object const &obj) {
    assert(!obj.isNull());
    return obj.get<T>(field.id);
}
/*!
 * Type-aware wrapper for a static field ID.
 *
 * This is a smarter alternative to just using jni::field_t since it avoids
 * having to repeat the type in the accessor, without using any more storage.
 *
 * @see FieldId
 */
template <typename T> struct StaticFieldId {
  public:
    StaticFieldId(jni::Class const &clazz, const char *name)
        : id(clazz.getStaticField<T>(name)) {}

    const jni::field_t id;
};

/*!
 * Get the value of static field @p field in Java type @p clazz.
 *
 * This is found by argument-dependent lookup and can be used unqualified.
 *
 * @relates FieldId
 */
template <typename T>
static inline T get(StaticFieldId<T> const &field, jni::Class const &clazz) {
    assert(!clazz.isNull());
    return clazz.get<T>(field.id);
}

/*!
 * Type-aware wrapper for a field ID of a wrapped structure type.
 *
 * This is a smarter alternative to just using jni::field_t since it avoids
 * having to repeat the type in the accessor, without using any more storage.
 *
 * Requires that the structure wrapper provides
 * `static constexpr const char *getTypeName() noexcept;`
 *
 * @see FieldId
 */
template <typename T> struct WrappedFieldId {
  public:
    WrappedFieldId(jni::Class const &clazz, const char *name)
        : id(lookupField(clazz, name)) {}

    const jni::field_t id;

  private:
    /*!
     * Helper for field ID lookup, mostly to avoid calling c_str() on a string
     * temporary.
     */
    static jni::field_t lookupField(jni::Class const &clazz, const char *name) {
        std::string fullType = std::string("L") + T::getTypeName() + ";";
        return clazz.getField(name, fullType.c_str());
    }
};

/*!
 * Get the value of field @p field in Java object @p obj.
 *
 * This is found by argument-dependent lookup and can be used unqualified.
 *
 * @relates WrappedFieldId
 */
template <typename T>
static inline T get(WrappedFieldId<T> const &field, jni::Object const &obj) {
    assert(!obj.isNull());
    return T{obj.get<jni::Object>(field.id)};
}
} // namespace impl
} // namespace wrap
