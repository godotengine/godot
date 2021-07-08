//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Feature.h: Definition of structs to hold feature/workaround information.
//

#ifndef ANGLE_PLATFORM_FEATURE_H_
#define ANGLE_PLATFORM_FEATURE_H_

#include <map>
#include <string>
#include <vector>

#define ANGLE_FEATURE_CONDITION(set, feature, cond) \
    set->feature.enabled   = cond;                  \
    set->feature.condition = ANGLE_STRINGIFY(cond);

namespace angle
{

enum class FeatureCategory
{
    FrontendWorkarounds,
    OpenGLWorkarounds,
    D3DWorkarounds,
    D3DCompilerWorkarounds,
    VulkanWorkarounds,
    VulkanFeatures,
    MetalFeatures,
};

constexpr char kFeatureCategoryFrontendWorkarounds[]    = "Frontend workarounds";
constexpr char kFeatureCategoryOpenGLWorkarounds[]      = "OpenGL workarounds";
constexpr char kFeatureCategoryD3DWorkarounds[]         = "D3D workarounds";
constexpr char kFeatureCategoryD3DCompilerWorkarounds[] = "D3D compiler workarounds";
constexpr char kFeatureCategoryVulkanWorkarounds[]      = "Vulkan workarounds";
constexpr char kFeatureCategoryVulkanFeatures[]         = "Vulkan features";
constexpr char kFeatureCategoryMetalFeatures[]          = "Metal features";
constexpr char kFeatureCategoryUnknown[]                = "Unknown";

inline const char *FeatureCategoryToString(const FeatureCategory &fc)
{
    switch (fc)
    {
        case FeatureCategory::FrontendWorkarounds:
            return kFeatureCategoryFrontendWorkarounds;
            break;

        case FeatureCategory::OpenGLWorkarounds:
            return kFeatureCategoryOpenGLWorkarounds;
            break;

        case FeatureCategory::D3DWorkarounds:
            return kFeatureCategoryD3DWorkarounds;
            break;

        case FeatureCategory::D3DCompilerWorkarounds:
            return kFeatureCategoryD3DCompilerWorkarounds;
            break;

        case FeatureCategory::VulkanWorkarounds:
            return kFeatureCategoryVulkanWorkarounds;
            break;

        case FeatureCategory::VulkanFeatures:
            return kFeatureCategoryVulkanFeatures;
            break;

        case FeatureCategory::MetalFeatures:
            return kFeatureCategoryMetalFeatures;
            break;

        default:
            return kFeatureCategoryUnknown;
            break;
    }
}

constexpr char kFeatureStatusEnabled[]  = "enabled";
constexpr char kFeatureStatusDisabled[] = "disabled";

inline const char *FeatureStatusToString(const bool &status)
{
    if (status)
    {
        return kFeatureStatusEnabled;
    }
    return kFeatureStatusDisabled;
}

struct Feature;

using FeatureMap  = std::map<std::string, Feature *>;
using FeatureList = std::vector<const Feature *>;

struct Feature
{
    Feature(const Feature &other);
    Feature(const char *name,
            const FeatureCategory &category,
            const char *description,
            FeatureMap *const mapPtr,
            const char *bug);
    ~Feature();

    // The name of the workaround, lowercase, camel_case
    const char *const name;

    // The category that the workaround belongs to. Eg. "Vulkan workarounds"
    const FeatureCategory category;

    // A short description to be read by the user.
    const char *const description;

    // A link to the bug, if any
    const char *const bug;

    // Whether the workaround is enabled or not. Determined by heuristics like vendor ID and
    // version, but may be overriden to any value.
    bool enabled = false;

    // A stingified version of the condition used to set 'enabled'. ie "IsNvidia() && IsApple()"
    const char *condition;
};

inline Feature::Feature(const Feature &other) = default;
inline Feature::Feature(const char *name,
                        const FeatureCategory &category,
                        const char *description,
                        FeatureMap *const mapPtr,
                        const char *bug = "")
    : name(name),
      category(category),
      description(description),
      bug(bug),
      enabled(false),
      condition(nullptr)
{
    if (mapPtr != nullptr)
    {
        (*mapPtr)[std::string(name)] = this;
    }
}

inline Feature::~Feature() = default;

struct FeatureSetBase
{
  public:
    FeatureSetBase();
    ~FeatureSetBase();

  private:
    // Non-copyable
    FeatureSetBase(const FeatureSetBase &other) = delete;
    FeatureSetBase &operator=(const FeatureSetBase &other) = delete;

  protected:
    FeatureMap members = FeatureMap();

  public:
    void overrideFeatures(const std::vector<std::string> &feature_names, const bool enabled)
    {
        for (const std::string &name : feature_names)
        {
            if (members.find(name) != members.end())
            {
                members[name]->enabled = enabled;
            }
        }
    }

    void populateFeatureList(FeatureList *features) const
    {
        for (FeatureMap::const_iterator it = members.begin(); it != members.end(); it++)
        {
            features->push_back(it->second);
        }
    }
};

inline FeatureSetBase::FeatureSetBase()  = default;
inline FeatureSetBase::~FeatureSetBase() = default;

}  // namespace angle

#endif  // ANGLE_PLATFORM_WORKAROUND_H_
