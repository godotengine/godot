// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
// Copyright (c) 2019 Collabora, Ltd.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Authors: Mark Young <marky@lunarg.com>, Rylie Pavlik <rylie.pavlik@collabora.com
//
/*!
 * @file
 *
 * The core of an XR_EXT_debug_utils implementation, used/shared by the loader and several SDK layers.
 */

#pragma once

#include "hex_and_handles.h"

#include <openxr/openxr.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct XrSdkGenericObject {
    //! Type-erased handle value
    uint64_t handle;

    //! Kind of object this handle refers to
    XrObjectType type;
    /// Un-erase the type of the handle and get it properly typed again.
    ///
    /// Note: Does not check the type before doing it!
    template <typename HandleType>
    HandleType& GetTypedHandle() {
        return TreatIntegerAsHandle<HandleType&>(handle);
    }

    //! @overload
    template <typename HandleType>
    HandleType const& GetTypedHandle() const {
        return TreatIntegerAsHandle<HandleType&>(handle);
    }

    //! Create from a typed handle and object type
    template <typename T>
    XrSdkGenericObject(T h, XrObjectType t) : handle(MakeHandleGeneric(h)), type(t) {}

    //! Create from an untyped handle value (integer) and object type
    XrSdkGenericObject(uint64_t h, XrObjectType t) : handle(h), type(t) {}
};

struct XrSdkLogObjectInfo {
    //! Type-erased handle value
    uint64_t handle;

    //! Kind of object this handle refers to
    XrObjectType type;

    //! To be assigned by the application - not part of this object's identity
    std::string name;

    /// Un-erase the type of the handle and get it properly typed again.
    ///
    /// Note: Does not check the type before doing it!
    template <typename HandleType>
    HandleType& GetTypedHandle() {
        return TreatIntegerAsHandle<HandleType&>(handle);
    }

    //! @overload
    template <typename HandleType>
    HandleType const& GetTypedHandle() const {
        return TreatIntegerAsHandle<HandleType&>(handle);
    }

    XrSdkLogObjectInfo() = default;

    //! Create from a typed handle and object type
    template <typename T>
    XrSdkLogObjectInfo(T h, XrObjectType t) : handle(MakeHandleGeneric(h)), type(t) {}

    //! Create from an untyped handle value (integer) and object type
    XrSdkLogObjectInfo(uint64_t h, XrObjectType t) : handle(h), type(t) {}
    //! Create from an untyped handle value (integer), object type, and name
    XrSdkLogObjectInfo(uint64_t h, XrObjectType t, const char* n) : handle(h), type(t), name(n == nullptr ? "" : n) {}

    std::string ToString() const;
};

//! True if the two object infos have the same handle value and handle type
static inline bool Equivalent(XrSdkLogObjectInfo const& a, XrSdkLogObjectInfo const& b) {
    return a.handle == b.handle && a.type == b.type;
}

//! @overload
static inline bool Equivalent(XrDebugUtilsObjectNameInfoEXT const& a, XrSdkLogObjectInfo const& b) {
    return a.objectHandle == b.handle && a.objectType == b.type;
}

//! @overload
static inline bool Equivalent(XrSdkLogObjectInfo const& a, XrDebugUtilsObjectNameInfoEXT const& b) { return Equivalent(b, a); }

/// Object info registered with calls to xrSetDebugUtilsObjectNameEXT
class ObjectInfoCollection {
   public:
    void AddObjectName(uint64_t object_handle, XrObjectType object_type, const std::string& object_name);

    void RemoveObject(uint64_t object_handle, XrObjectType object_type);

    //! Find the stored object info, if any, matching handle and type.
    //! Return nullptr if not found.
    XrSdkLogObjectInfo const* LookUpStoredObjectInfo(XrSdkLogObjectInfo const& info) const;

    //! Find the stored object info, if any, matching handle and type.
    //! Return nullptr if not found.
    XrSdkLogObjectInfo* LookUpStoredObjectInfo(XrSdkLogObjectInfo const& info);

    //! Find the stored object info, if any.
    //! Return nullptr if not found.
    XrSdkLogObjectInfo const* LookUpStoredObjectInfo(uint64_t handle, XrObjectType type) const {
        return LookUpStoredObjectInfo({handle, type});
    }

    //! Find the object name, if any, and update debug utils info accordingly.
    //! Return true if found and updated.
    bool LookUpObjectName(XrDebugUtilsObjectNameInfoEXT& info) const;

    //! Find the object name, if any, and update logging info accordingly.
    //! Return true if found and updated.
    bool LookUpObjectName(XrSdkLogObjectInfo& info) const;

    //! Is the collection empty?
    bool Empty() const { return object_info_.empty(); }

   private:
    // Object names that have been set for given objects
    std::vector<XrSdkLogObjectInfo> object_info_;
};

struct XrSdkSessionLabel;
using XrSdkSessionLabelPtr = std::unique_ptr<XrSdkSessionLabel>;
using XrSdkSessionLabelList = std::vector<XrSdkSessionLabelPtr>;

struct XrSdkSessionLabel {
    static XrSdkSessionLabelPtr make(const XrDebugUtilsLabelEXT& label_info, bool individual);

    std::string label_name;
    XrDebugUtilsLabelEXT debug_utils_label;
    bool is_individual_label;

   private:
    XrSdkSessionLabel(const XrDebugUtilsLabelEXT& label_info, bool individual);
};

/// The metadata for a collection of objects. Must persist unmodified during the entire debug messenger call!
struct NamesAndLabels {
    NamesAndLabels() = default;
    NamesAndLabels(std::vector<XrSdkLogObjectInfo> obj, std::vector<XrDebugUtilsLabelEXT> lab);
    /// C++ structure owning the data (strings) backing the objects vector.
    std::vector<XrSdkLogObjectInfo> sdk_objects;

    std::vector<XrDebugUtilsObjectNameInfoEXT> objects;
    std::vector<XrDebugUtilsLabelEXT> labels;

    /// Populate the debug utils callback data structure.
    void PopulateCallbackData(XrDebugUtilsMessengerCallbackDataEXT& data) const;
    // XrDebugUtilsMessengerCallbackDataEXT MakeCallbackData() const;
};

struct AugmentedCallbackData {
    std::vector<XrDebugUtilsLabelEXT> labels;
    std::vector<XrDebugUtilsObjectNameInfoEXT> new_objects;
    XrDebugUtilsMessengerCallbackDataEXT modified_data;
    const XrDebugUtilsMessengerCallbackDataEXT* exported_data;
};

/// Tracks all the data (handle names and session labels) required to fully augment XR_EXT_debug_utils-related calls.
class DebugUtilsData {
   public:
    DebugUtilsData() = default;

    DebugUtilsData(const DebugUtilsData&) = delete;
    DebugUtilsData& operator=(const DebugUtilsData&) = delete;

    bool Empty() const { return object_info_.Empty() && session_labels_.empty(); }

    //! Core of implementation for xrSetDebugUtilsObjectNameEXT
    void AddObjectName(uint64_t object_handle, XrObjectType object_type, const std::string& object_name);

    /// Core of implementation for xrSessionBeginDebugUtilsLabelRegionEXT
    void BeginLabelRegion(XrSession session, const XrDebugUtilsLabelEXT& label_info);

    /// Core of implementation for xrSessionEndDebugUtilsLabelRegionEXT
    void EndLabelRegion(XrSession session);

    /// Core of implementation for xrSessionInsertDebugUtilsLabelEXT
    void InsertLabel(XrSession session, const XrDebugUtilsLabelEXT& label_info);

    /// Removes all labels associated with a session - call in xrDestroySession and xrDestroyInstance (for all child sessions)
    void DeleteSessionLabels(XrSession session);

    /// Retrieve labels for the given session, if any, and push them in reverse order on the vector.
    void LookUpSessionLabels(XrSession session, std::vector<XrDebugUtilsLabelEXT>& labels) const;

    /// Removes all data related to this object - including session labels if it's a session.
    ///
    /// Does not take care of handling child objects - you must do this yourself.
    void DeleteObject(uint64_t object_handle, XrObjectType object_type);

    /// Given the collection of objects, populate their names and list of labels
    NamesAndLabels PopulateNamesAndLabels(std::vector<XrSdkLogObjectInfo> objects) const;

    void WrapCallbackData(AugmentedCallbackData* aug_data,
                          const XrDebugUtilsMessengerCallbackDataEXT* provided_callback_data) const;

   private:
    void RemoveIndividualLabel(XrSdkSessionLabelList& label_vec);
    XrSdkSessionLabelList* GetSessionLabelList(XrSession session);
    XrSdkSessionLabelList& GetOrCreateSessionLabelList(XrSession session);

    // Session labels: one vector of them per session.
    std::unordered_map<XrSession, std::unique_ptr<XrSdkSessionLabelList>> session_labels_;

    // Names for objects.
    ObjectInfoCollection object_info_;
};
