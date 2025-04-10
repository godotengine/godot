// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
// Copyright (c) 2019 Collabora, Ltd.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Authors: Mark Young <marky@lunarg.com>
//                  Rylie Pavlik <rylie.pavlik@collabora.com>
//                  Dave Houlton <daveh@lunarg.com>
//

#include "object_info.h"

#include "extra_algorithms.h"
#include "hex_and_handles.h"

#include <openxr/openxr.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "memory.h"

std::string XrSdkLogObjectInfo::ToString() const {
    std::ostringstream oss;
    oss << Uint64ToHexString(handle);
    if (!name.empty()) {
        oss << " (" << name << ")";
    }
    return oss.str();
}

void ObjectInfoCollection::AddObjectName(uint64_t object_handle, XrObjectType object_type, const std::string& object_name) {
    // If name is empty, we should erase it
    if (object_name.empty()) {
        RemoveObject(object_handle, object_type);
        return;
    }

    // Otherwise, add it or update the name
    XrSdkLogObjectInfo new_obj = {object_handle, object_type};

    // If it already exists, update the name
    auto lookup_info = LookUpStoredObjectInfo(new_obj);
    if (lookup_info != nullptr) {
        lookup_info->name = object_name;
        return;
    }

    // It doesn't exist, so add a new info block
    new_obj.name = object_name;
    object_info_.push_back(new_obj);
}

void ObjectInfoCollection::RemoveObject(uint64_t object_handle, XrObjectType object_type) {
    vector_remove_if_and_erase(
        object_info_, [=](XrSdkLogObjectInfo const& info) { return info.handle == object_handle && info.type == object_type; });
}

XrSdkLogObjectInfo const* ObjectInfoCollection::LookUpStoredObjectInfo(XrSdkLogObjectInfo const& info) const {
    auto e = object_info_.end();
    auto it = std::find_if(object_info_.begin(), e, [&](XrSdkLogObjectInfo const& stored) { return Equivalent(stored, info); });
    if (it != e) {
        return &(*it);
    }
    return nullptr;
}

XrSdkLogObjectInfo* ObjectInfoCollection::LookUpStoredObjectInfo(XrSdkLogObjectInfo const& info) {
    auto e = object_info_.end();
    auto it = std::find_if(object_info_.begin(), e, [&](XrSdkLogObjectInfo const& stored) { return Equivalent(stored, info); });
    if (it != e) {
        return &(*it);
    }
    return nullptr;
}

bool ObjectInfoCollection::LookUpObjectName(XrDebugUtilsObjectNameInfoEXT& info) const {
    auto info_lookup = LookUpStoredObjectInfo(info.objectHandle, info.objectType);
    if (info_lookup != nullptr) {
        info.objectName = info_lookup->name.c_str();
        return true;
    }
    return false;
}

bool ObjectInfoCollection::LookUpObjectName(XrSdkLogObjectInfo& info) const {
    auto info_lookup = LookUpStoredObjectInfo(info);
    if (info_lookup != nullptr) {
        info.name = info_lookup->name;
        return true;
    }
    return false;
}

static std::vector<XrDebugUtilsObjectNameInfoEXT> PopulateObjectNameInfo(std::vector<XrSdkLogObjectInfo> const& obj) {
    std::vector<XrDebugUtilsObjectNameInfoEXT> ret;
    ret.reserve(obj.size());
    std::transform(obj.begin(), obj.end(), std::back_inserter(ret), [](XrSdkLogObjectInfo const& info) {
        return XrDebugUtilsObjectNameInfoEXT{XR_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, info.type, info.handle,
                                             info.name.c_str()};
    });
    return ret;
}

NamesAndLabels::NamesAndLabels(std::vector<XrSdkLogObjectInfo> obj, std::vector<XrDebugUtilsLabelEXT> lab)
    : sdk_objects(std::move(obj)), objects(PopulateObjectNameInfo(sdk_objects)), labels(std::move(lab)) {}

void NamesAndLabels::PopulateCallbackData(XrDebugUtilsMessengerCallbackDataEXT& callback_data) const {
    callback_data.objects = objects.empty() ? nullptr : const_cast<XrDebugUtilsObjectNameInfoEXT*>(objects.data());
    callback_data.objectCount = static_cast<uint32_t>(objects.size());
    callback_data.sessionLabels = labels.empty() ? nullptr : const_cast<XrDebugUtilsLabelEXT*>(labels.data());
    callback_data.sessionLabelCount = static_cast<uint32_t>(labels.size());
}

void DebugUtilsData::LookUpSessionLabels(XrSession session, std::vector<XrDebugUtilsLabelEXT>& labels) const {
    auto session_label_iterator = session_labels_.find(session);
    if (session_label_iterator != session_labels_.end()) {
        auto& XrSdkSessionLabels = *session_label_iterator->second;
        // Copy the debug utils labels in reverse order in the the labels vector.
        std::transform(XrSdkSessionLabels.rbegin(), XrSdkSessionLabels.rend(), std::back_inserter(labels),
                       [](XrSdkSessionLabelPtr const& label) { return label->debug_utils_label; });
    }
}

XrSdkSessionLabel::XrSdkSessionLabel(const XrDebugUtilsLabelEXT& label_info, bool individual)
    : label_name(label_info.labelName), debug_utils_label(label_info), is_individual_label(individual) {
    // Update the c string pointer to the one we hold.
    debug_utils_label.labelName = label_name.c_str();
    // Zero out the next pointer to avoid a dangling pointer
    debug_utils_label.next = nullptr;
}

XrSdkSessionLabelPtr XrSdkSessionLabel::make(const XrDebugUtilsLabelEXT& label_info, bool individual) {
    XrSdkSessionLabelPtr ret(new XrSdkSessionLabel(label_info, individual));
    return ret;
}
void DebugUtilsData::AddObjectName(uint64_t object_handle, XrObjectType object_type, const std::string& object_name) {
    object_info_.AddObjectName(object_handle, object_type, object_name);
}

// We always want to remove the old individual label before we do anything else.
// So, do that in its own method
void DebugUtilsData::RemoveIndividualLabel(XrSdkSessionLabelList& label_vec) {
    if (!label_vec.empty() && label_vec.back()->is_individual_label) {
        label_vec.pop_back();
    }
}

XrSdkSessionLabelList* DebugUtilsData::GetSessionLabelList(XrSession session) {
    auto session_label_iterator = session_labels_.find(session);
    if (session_label_iterator == session_labels_.end()) {
        return nullptr;
    }
    return session_label_iterator->second.get();
}

XrSdkSessionLabelList& DebugUtilsData::GetOrCreateSessionLabelList(XrSession session) {
    XrSdkSessionLabelList* vec_ptr = GetSessionLabelList(session);
    if (vec_ptr == nullptr) {
        std::unique_ptr<XrSdkSessionLabelList> vec(new XrSdkSessionLabelList);
        vec_ptr = vec.get();
        session_labels_[session] = std::move(vec);
    }
    return *vec_ptr;
}

void DebugUtilsData::BeginLabelRegion(XrSession session, const XrDebugUtilsLabelEXT& label_info) {
    auto& vec = GetOrCreateSessionLabelList(session);

    // Individual labels do not stay around in the transition into a new label region
    RemoveIndividualLabel(vec);

    // Start the new label region
    vec.emplace_back(XrSdkSessionLabel::make(label_info, false));
}

void DebugUtilsData::EndLabelRegion(XrSession session) {
    XrSdkSessionLabelList* vec_ptr = GetSessionLabelList(session);
    if (vec_ptr == nullptr) {
        return;
    }

    // Individual labels do not stay around in the transition out of label region
    RemoveIndividualLabel(*vec_ptr);

    // Remove the last label region
    if (!vec_ptr->empty()) {
        vec_ptr->pop_back();
    }
}

void DebugUtilsData::InsertLabel(XrSession session, const XrDebugUtilsLabelEXT& label_info) {
    auto& vec = GetOrCreateSessionLabelList(session);

    // Remove any individual layer that might already be there
    RemoveIndividualLabel(vec);

    // Insert a new individual label
    vec.emplace_back(XrSdkSessionLabel::make(label_info, true));
}

void DebugUtilsData::DeleteObject(uint64_t object_handle, XrObjectType object_type) {
    object_info_.RemoveObject(object_handle, object_type);

    if (object_type == XR_OBJECT_TYPE_SESSION) {
        auto session = TreatIntegerAsHandle<XrSession>(object_handle);
        XrSdkSessionLabelList* vec_ptr = GetSessionLabelList(session);
        if (vec_ptr != nullptr) {
            session_labels_.erase(session);
        }
    }
}

void DebugUtilsData::DeleteSessionLabels(XrSession session) { session_labels_.erase(session); }

NamesAndLabels DebugUtilsData::PopulateNamesAndLabels(std::vector<XrSdkLogObjectInfo> objects) const {
    std::vector<XrDebugUtilsLabelEXT> labels;
    for (auto& obj : objects) {
        // Check for any names that have been associated with the objects and set them up here
        object_info_.LookUpObjectName(obj);
        // If this is a session, see if there are any labels associated with it for us to add
        // to the callback content.
        if (XR_OBJECT_TYPE_SESSION == obj.type) {
            LookUpSessionLabels(obj.GetTypedHandle<XrSession>(), labels);
        }
    }

    return {objects, labels};
}

void DebugUtilsData::WrapCallbackData(AugmentedCallbackData* aug_data,
                                      const XrDebugUtilsMessengerCallbackDataEXT* callback_data) const {
    // If there's nothing to add, just return the original data as the augmented copy
    aug_data->exported_data = callback_data;
    if (object_info_.Empty() || callback_data->objectCount == 0) {
        return;
    }

    // Inspect each of the callback objects
    bool name_found = false;
    for (uint32_t obj = 0; obj < callback_data->objectCount; ++obj) {
        auto& current_obj = callback_data->objects[obj];
        name_found |= (nullptr != object_info_.LookUpStoredObjectInfo(current_obj.objectHandle, current_obj.objectType));

        // If this is a session, record any labels associated with it
        if (XR_OBJECT_TYPE_SESSION == current_obj.objectType) {
            XrSession session = TreatIntegerAsHandle<XrSession>(current_obj.objectHandle);
            LookUpSessionLabels(session, aug_data->labels);
        }
    }

    // If we found nothing to add, return the original data
    if (!name_found && aug_data->labels.empty()) {
        return;
    }

    // Found additional data - modify an internal copy and return that as the exported data
    memcpy(&aug_data->modified_data, callback_data, sizeof(XrDebugUtilsMessengerCallbackDataEXT));
    aug_data->new_objects.assign(callback_data->objects, callback_data->objects + callback_data->objectCount);

    // Record (overwrite) the names of all incoming objects provided in our internal list
    for (auto& obj : aug_data->new_objects) {
        object_info_.LookUpObjectName(obj);
    }

    // Update local copy & point export to it
    aug_data->modified_data.objects = aug_data->new_objects.data();
    aug_data->modified_data.sessionLabelCount = static_cast<uint32_t>(aug_data->labels.size());
    aug_data->modified_data.sessionLabels = aug_data->labels.empty() ? nullptr : aug_data->labels.data();
    aug_data->exported_data = &aug_data->modified_data;
    return;
}
