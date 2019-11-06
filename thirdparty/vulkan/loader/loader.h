/*
 *
 * Copyright (c) 2014-2019 The Khronos Group Inc.
 * Copyright (c) 2014-2019 Valve Corporation
 * Copyright (c) 2014-2019 LunarG, Inc.
 * Copyright (C) 2015 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Jon Ashburn <jon@lunarg.com>
 * Author: Courtney Goeltzenleuchter <courtney@LunarG.com>
 * Author: Chia-I Wu <olvaffe@gmail.com>
 * Author: Chia-I Wu <olv@lunarg.com>
 * Author: Mark Lobodzinski <mark@LunarG.com>
 * Author: Lenny Komow <lenny@lunarg.com>
 *
 */

#ifndef LOADER_H
#define LOADER_H

#include <vulkan/vulkan.h>
#include "vk_loader_platform.h"
#include "vk_loader_layer.h"
#include <vulkan/vk_layer.h>
#include <vulkan/vk_icd.h>
#include <assert.h>
#include "vk_layer_dispatch_table.h"
#include "vk_loader_extensions.h"

#if defined(__GNUC__) && __GNUC__ >= 4
#define LOADER_EXPORT __attribute__((visibility("default")))
#elif defined(__SUNPRO_C) && (__SUNPRO_C >= 0x590)
#define LOADER_EXPORT __attribute__((visibility("default")))
#else
#define LOADER_EXPORT
#endif

// A debug option to disable allocators at compile time to investigate future issues.
#define DEBUG_DISABLE_APP_ALLOCATORS 0

#define MAX_STRING_SIZE 1024

// This is defined in vk_layer.h, but if there's problems we need to create the define
// here.
#ifndef MAX_NUM_UNKNOWN_EXTS
#define MAX_NUM_UNKNOWN_EXTS 250
#endif

enum layer_type_flags {
    VK_LAYER_TYPE_FLAG_INSTANCE_LAYER = 0x1,  // If not set, indicates Device layer
    VK_LAYER_TYPE_FLAG_EXPLICIT_LAYER = 0x2,  // If not set, indicates Implicit layer
    VK_LAYER_TYPE_FLAG_META_LAYER = 0x4,      // If not set, indicates standard layer
};

typedef enum VkStringErrorFlagBits {
    VK_STRING_ERROR_NONE = 0x00000000,
    VK_STRING_ERROR_LENGTH = 0x00000001,
    VK_STRING_ERROR_BAD_DATA = 0x00000002,
} VkStringErrorFlagBits;
typedef VkFlags VkStringErrorFlags;

static const int MaxLoaderStringLength = 256;
static const char UTF8_ONE_BYTE_CODE = 0xC0;
static const char UTF8_ONE_BYTE_MASK = 0xE0;
static const char UTF8_TWO_BYTE_CODE = 0xE0;
static const char UTF8_TWO_BYTE_MASK = 0xF0;
static const char UTF8_THREE_BYTE_CODE = 0xF0;
static const char UTF8_THREE_BYTE_MASK = 0xF8;
static const char UTF8_DATA_BYTE_CODE = 0x80;
static const char UTF8_DATA_BYTE_MASK = 0xC0;

// form of all dynamic lists/arrays
// only the list element should be changed
struct loader_generic_list {
    size_t capacity;
    uint32_t count;
    void *list;
};

struct loader_extension_list {
    size_t capacity;
    uint32_t count;
    VkExtensionProperties *list;
};

struct loader_dev_ext_props {
    VkExtensionProperties props;
    uint32_t entrypoint_count;
    char **entrypoints;
};

struct loader_device_extension_list {
    size_t capacity;
    uint32_t count;
    struct loader_dev_ext_props *list;
};

struct loader_name_value {
    char name[MAX_STRING_SIZE];
    char value[MAX_STRING_SIZE];
};

struct loader_layer_functions {
    char str_gipa[MAX_STRING_SIZE];
    char str_gdpa[MAX_STRING_SIZE];
    char str_negotiate_interface[MAX_STRING_SIZE];
    PFN_vkNegotiateLoaderLayerInterfaceVersion negotiate_layer_interface;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
    PFN_GetPhysicalDeviceProcAddr get_physical_device_proc_addr;
};

struct loader_override_expiration {
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t minute;
};

struct loader_layer_properties {
    VkLayerProperties info;
    enum layer_type_flags type_flags;
    uint32_t interface_version;  // PFN_vkNegotiateLoaderLayerInterfaceVersion
    char lib_name[MAX_STRING_SIZE];
    loader_platform_dl_handle lib_handle;
    struct loader_layer_functions functions;
    struct loader_extension_list instance_extension_list;
    struct loader_device_extension_list device_extension_list;
    struct loader_name_value disable_env_var;
    struct loader_name_value enable_env_var;
    uint32_t num_component_layers;
    char (*component_layer_names)[MAX_STRING_SIZE];
    struct {
        char enumerate_instance_extension_properties[MAX_STRING_SIZE];
        char enumerate_instance_layer_properties[MAX_STRING_SIZE];
        char enumerate_instance_version[MAX_STRING_SIZE];
    } pre_instance_functions;
    uint32_t num_override_paths;
    char (*override_paths)[MAX_STRING_SIZE];
    bool is_override;
    bool has_expiration;
    struct loader_override_expiration expiration;
    bool keep;
    uint32_t num_blacklist_layers;
    char (*blacklist_layer_names)[MAX_STRING_SIZE];
};

struct loader_layer_list {
    size_t capacity;
    uint32_t count;
    struct loader_layer_properties *list;
};

struct loader_dispatch_hash_list {
    size_t capacity;
    uint32_t count;
    uint32_t *index;  // index into the dev_ext dispatch table
};

// loader_dispatch_hash_entry and loader_dev_ext_dispatch_table.dev_ext have
// one to one correspondence; one loader_dispatch_hash_entry for one dev_ext
// dispatch entry.
// Also have a one to one correspondence with functions in dev_ext_trampoline.c
struct loader_dispatch_hash_entry {
    char *func_name;
    struct loader_dispatch_hash_list list;  // to handle hashing collisions
};

typedef VkResult(VKAPI_PTR *PFN_vkDevExt)(VkDevice device);
struct loader_dev_ext_dispatch_table {
    PFN_vkDevExt dev_ext[MAX_NUM_UNKNOWN_EXTS];
};

struct loader_dev_dispatch_table {
    VkLayerDispatchTable core_dispatch;
    struct loader_dev_ext_dispatch_table ext_dispatch;
};

// per CreateDevice structure
struct loader_device {
    struct loader_dev_dispatch_table loader_dispatch;
    VkDevice chain_device;  // device object from the dispatch chain
    VkDevice icd_device;    // device object from the icd
    struct loader_physical_device_term *phys_dev_term;

    // List of activated layers.
    //  app_      is the version based on exactly what the application asked for.
    //            This is what must be returned to the application on Enumerate calls.
    //  expanded_ is the version based on expanding meta-layers into their
    //            individual component layers.  This is what is used internally.
    struct loader_layer_list app_activated_layer_list;
    struct loader_layer_list expanded_activated_layer_list;

    VkAllocationCallbacks alloc_callbacks;

    // List of activated device extensions that have terminators implemented in the loader
    struct {
        bool khr_swapchain_enabled;
        bool khr_display_swapchain_enabled;
        bool khr_device_group_enabled;
        bool ext_debug_marker_enabled;
        bool ext_debug_utils_enabled;
        bool ext_full_screen_exclusive_enabled;
    } extensions;

    struct loader_device *next;
};

// Per ICD information

// Per ICD structure
struct loader_icd_term {
    // pointers to find other structs
    const struct loader_scanned_icd *scanned_icd;
    const struct loader_instance *this_instance;
    struct loader_device *logical_device_list;
    VkInstance instance;  // instance object from the icd
    struct loader_icd_term_dispatch dispatch;

    struct loader_icd_term *next;

    PFN_PhysDevExt phys_dev_ext[MAX_NUM_UNKNOWN_EXTS];
};

// Per ICD library structure
struct loader_icd_tramp_list {
    size_t capacity;
    uint32_t count;
    struct loader_scanned_icd *scanned_list;
};

struct loader_instance_dispatch_table {
    VkLayerInstanceDispatchTable layer_inst_disp;  // must be first entry in structure

    // Physical device functions unknown to the loader
    PFN_PhysDevExt phys_dev_ext[MAX_NUM_UNKNOWN_EXTS];
};

// Per instance structure
struct loader_instance {
    struct loader_instance_dispatch_table *disp;  // must be first entry in structure

    // Vulkan API version the app is intending to use.
    uint16_t app_api_major_version;
    uint16_t app_api_minor_version;

    // We need to manually track physical devices over time.  If the user
    // re-queries the information, we don't want to delete old data or
    // create new data unless necessary.
    uint32_t total_gpu_count;
    uint32_t phys_dev_count_term;
    struct loader_physical_device_term **phys_devs_term;
    uint32_t phys_dev_count_tramp;
    struct loader_physical_device_tramp **phys_devs_tramp;

    // We also need to manually track physical device groups, but we don't need
    // loader specific structures since we have that content in the physical
    // device stored internal to the public structures.
    uint32_t phys_dev_group_count_term;
    struct VkPhysicalDeviceGroupProperties **phys_dev_groups_term;
    uint32_t phys_dev_group_count_tramp;
    struct VkPhysicalDeviceGroupProperties **phys_dev_groups_tramp;

    struct loader_instance *next;

    uint32_t total_icd_count;
    struct loader_icd_term *icd_terms;
    struct loader_icd_tramp_list icd_tramp_list;

    struct loader_dispatch_hash_entry dev_ext_disp_hash[MAX_NUM_UNKNOWN_EXTS];
    struct loader_dispatch_hash_entry phys_dev_ext_disp_hash[MAX_NUM_UNKNOWN_EXTS];

    struct loader_msg_callback_map_entry *icd_msg_callback_map;

    struct loader_layer_list instance_layer_list;
    bool override_layer_present;

    // List of activated layers.
    //  app_      is the version based on exactly what the application asked for.
    //            This is what must be returned to the application on Enumerate calls.
    //  expanded_ is the version based on expanding meta-layers into their
    //            individual component layers.  This is what is used internally.
    struct loader_layer_list app_activated_layer_list;
    struct loader_layer_list expanded_activated_layer_list;

    VkInstance instance;  // layers/ICD instance returned to trampoline

    struct loader_extension_list ext_list;  // icds and loaders extensions
    union loader_instance_extension_enables enabled_known_extensions;

    VkLayerDbgFunctionNode *DbgFunctionHead;
    uint32_t num_tmp_report_callbacks;
    VkDebugReportCallbackCreateInfoEXT *tmp_report_create_infos;
    VkDebugReportCallbackEXT *tmp_report_callbacks;
    uint32_t num_tmp_messengers;
    VkDebugUtilsMessengerCreateInfoEXT *tmp_messenger_create_infos;
    VkDebugUtilsMessengerEXT *tmp_messengers;

    VkAllocationCallbacks alloc_callbacks;

    bool wsi_surface_enabled;
#ifdef VK_USE_PLATFORM_WIN32_KHR
    bool wsi_win32_surface_enabled;
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    bool wsi_wayland_surface_enabled;
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
    bool wsi_xcb_surface_enabled;
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
    bool wsi_xlib_surface_enabled;
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    bool wsi_android_surface_enabled;
#endif
#ifdef VK_USE_PLATFORM_MACOS_MVK
    bool wsi_macos_surface_enabled;
#endif
#ifdef VK_USE_PLATFORM_IOS_MVK
    bool wsi_ios_surface_enabled;
#endif
    bool wsi_headless_surface_enabled;
#if defined(VK_USE_PLATFORM_METAL_EXT)
    bool wsi_metal_surface_enabled;
#endif
    bool wsi_display_enabled;
    bool wsi_display_props2_enabled;
};

// VkPhysicalDevice requires special treatment by loader.  Firstly, terminator
// code must be able to get the struct loader_icd_term to call into the proper
// driver  (multiple ICD/gpu case). This can be accomplished by wrapping the
// created VkPhysicalDevice in loader terminate_EnumeratePhysicalDevices().
// Secondly, the loader must be able to handle wrapped by layer VkPhysicalDevice
// in trampoline code.  This implies, that the loader trampoline code must also
// wrap the VkPhysicalDevice object in trampoline code.  Thus, loader has to
// wrap the VkPhysicalDevice created object twice. In trampoline code it can't
// rely on the terminator object wrapping since a layer may also wrap. Since
// trampoline code wraps the VkPhysicalDevice this means all loader trampoline
// code that passes a VkPhysicalDevice should unwrap it.

// Per enumerated PhysicalDevice structure, used to wrap in trampoline code and
// also same structure used to wrap in terminator code
struct loader_physical_device_tramp {
    struct loader_instance_dispatch_table *disp;  // must be first entry in structure
    struct loader_instance *this_instance;
    VkPhysicalDevice phys_dev;  // object from layers/loader terminator
};

// Per enumerated PhysicalDevice structure, used to wrap in terminator code
struct loader_physical_device_term {
    struct loader_instance_dispatch_table *disp;  // must be first entry in structure
    struct loader_icd_term *this_icd_term;
    uint8_t icd_index;
    VkPhysicalDevice phys_dev;  // object from ICD
};

struct loader_struct {
    struct loader_instance *instances;
};

struct loader_scanned_icd {
    char *lib_name;
    loader_platform_dl_handle handle;
    uint32_t api_version;
    uint32_t interface_version;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_GetPhysicalDeviceProcAddr GetPhysicalDeviceProcAddr;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
};

static inline struct loader_instance *loader_instance(VkInstance instance) { return (struct loader_instance *)instance; }

static inline VkPhysicalDevice loader_unwrap_physical_device(VkPhysicalDevice physicalDevice) {
    struct loader_physical_device_tramp *phys_dev = (struct loader_physical_device_tramp *)physicalDevice;
    return phys_dev->phys_dev;
}

static inline void loader_set_dispatch(void *obj, const void *data) { *((const void **)obj) = data; }

static inline VkLayerDispatchTable *loader_get_dispatch(const void *obj) { return *((VkLayerDispatchTable **)obj); }

static inline struct loader_dev_dispatch_table *loader_get_dev_dispatch(const void *obj) {
    return *((struct loader_dev_dispatch_table **)obj);
}

static inline VkLayerInstanceDispatchTable *loader_get_instance_layer_dispatch(const void *obj) {
    return *((VkLayerInstanceDispatchTable **)obj);
}

static inline struct loader_instance_dispatch_table *loader_get_instance_dispatch(const void *obj) {
    return *((struct loader_instance_dispatch_table **)obj);
}

static inline void loader_init_dispatch(void *obj, const void *data) {
#ifdef DEBUG
    assert(valid_loader_magic_value(obj) &&
           "Incompatible ICD, first dword must be initialized to "
           "ICD_LOADER_MAGIC. See loader/README.md for details.");
#endif

    loader_set_dispatch(obj, data);
}

// Global variables used across files
extern struct loader_struct loader;
extern THREAD_LOCAL_DECL struct loader_instance *tls_instance;
#if defined(_WIN32) && !defined(LOADER_DYNAMIC_LIB)
extern LOADER_PLATFORM_THREAD_ONCE_DEFINITION(once_init);
#endif
extern loader_platform_thread_mutex loader_lock;
extern loader_platform_thread_mutex loader_json_lock;

struct loader_msg_callback_map_entry {
    VkDebugReportCallbackEXT icd_obj;
    VkDebugReportCallbackEXT loader_obj;
};

// Helper function definitions
void *loader_instance_heap_alloc(const struct loader_instance *instance, size_t size, VkSystemAllocationScope allocationScope);
void loader_instance_heap_free(const struct loader_instance *instance, void *pMemory);
void *loader_instance_heap_realloc(const struct loader_instance *instance, void *pMemory, size_t orig_size, size_t size,
                                   VkSystemAllocationScope alloc_scope);
void *loader_instance_tls_heap_alloc(size_t size);
void loader_instance_tls_heap_free(void *pMemory);
void *loader_device_heap_alloc(const struct loader_device *device, size_t size, VkSystemAllocationScope allocationScope);
void loader_device_heap_free(const struct loader_device *device, void *pMemory);
void *loader_device_heap_realloc(const struct loader_device *device, void *pMemory, size_t orig_size, size_t size,
                                 VkSystemAllocationScope alloc_scope);

void loader_log(const struct loader_instance *inst, VkFlags msg_type, int32_t msg_code, const char *format, ...);

bool compare_vk_extension_properties(const VkExtensionProperties *op1, const VkExtensionProperties *op2);

VkResult loaderValidateLayers(const struct loader_instance *inst, const uint32_t layer_count,
                              const char *const *ppEnabledLayerNames, const struct loader_layer_list *list);

VkResult loader_validate_instance_extensions(struct loader_instance *inst, const struct loader_extension_list *icd_exts,
                                             const struct loader_layer_list *instance_layer,
                                             const VkInstanceCreateInfo *pCreateInfo);

void loader_initialize(void);
bool has_vk_extension_property_array(const VkExtensionProperties *vk_ext_prop, const uint32_t count,
                                     const VkExtensionProperties *ext_array);
bool has_vk_extension_property(const VkExtensionProperties *vk_ext_prop, const struct loader_extension_list *ext_list);

VkResult loader_add_to_ext_list(const struct loader_instance *inst, struct loader_extension_list *ext_list,
                                uint32_t prop_list_count, const VkExtensionProperties *props);
VkResult loader_add_to_dev_ext_list(const struct loader_instance *inst, struct loader_device_extension_list *ext_list,
                                    const VkExtensionProperties *props, uint32_t entry_count, char **entrys);
VkResult loader_add_device_extensions(const struct loader_instance *inst,
                                      PFN_vkEnumerateDeviceExtensionProperties fpEnumerateDeviceExtensionProperties,
                                      VkPhysicalDevice physical_device, const char *lib_name,
                                      struct loader_extension_list *ext_list);
VkResult loader_init_generic_list(const struct loader_instance *inst, struct loader_generic_list *list_info, size_t element_size);
void loader_destroy_generic_list(const struct loader_instance *inst, struct loader_generic_list *list);
void loaderDestroyLayerList(const struct loader_instance *inst, struct loader_device *device, struct loader_layer_list *layer_list);
void loaderDeleteLayerListAndProperties(const struct loader_instance *inst, struct loader_layer_list *layer_list);
void loaderAddLayerNameToList(const struct loader_instance *inst, const char *name, const enum layer_type_flags type_flags,
                              const struct loader_layer_list *source_list, struct loader_layer_list *target_list,
                              struct loader_layer_list *expanded_target_list);
void loader_scanned_icd_clear(const struct loader_instance *inst, struct loader_icd_tramp_list *icd_tramp_list);
VkResult loader_icd_scan(const struct loader_instance *inst, struct loader_icd_tramp_list *icd_tramp_list);
void loaderScanForLayers(struct loader_instance *inst, struct loader_layer_list *instance_layers);
void loaderScanForImplicitLayers(struct loader_instance *inst, struct loader_layer_list *instance_layers);
bool loaderImplicitLayerIsEnabled(const struct loader_instance *inst, const struct loader_layer_properties *prop);
VkResult loader_get_icd_loader_instance_extensions(const struct loader_instance *inst, struct loader_icd_tramp_list *icd_tramp_list,
                                                   struct loader_extension_list *inst_exts);
struct loader_icd_term *loader_get_icd_and_device(const VkDevice device, struct loader_device **found_dev, uint32_t *icd_index);
void loader_init_dispatch_dev_ext(struct loader_instance *inst, struct loader_device *dev);
void *loader_dev_ext_gpa(struct loader_instance *inst, const char *funcName);
void *loader_get_dev_ext_trampoline(uint32_t index);
bool loader_phys_dev_ext_gpa(struct loader_instance *inst, const char *funcName, bool perform_checking, void **tramp_addr,
                             void **term_addr);
void *loader_get_phys_dev_ext_tramp(uint32_t index);
void *loader_get_phys_dev_ext_termin(uint32_t index);
struct loader_instance *loader_get_instance(const VkInstance instance);
void loaderDeactivateLayers(const struct loader_instance *instance, struct loader_device *device, struct loader_layer_list *list);
struct loader_device *loader_create_logical_device(const struct loader_instance *inst, const VkAllocationCallbacks *pAllocator);
void loader_add_logical_device(const struct loader_instance *inst, struct loader_icd_term *icd_term,
                               struct loader_device *found_dev);
void loader_remove_logical_device(const struct loader_instance *inst, struct loader_icd_term *icd_term,
                                  struct loader_device *found_dev, const VkAllocationCallbacks *pAllocator);
// NOTE: Outside of loader, this entry-point is only provided for error
// cleanup.
void loader_destroy_logical_device(const struct loader_instance *inst, struct loader_device *dev,
                                   const VkAllocationCallbacks *pAllocator);

VkResult loaderEnableInstanceLayers(struct loader_instance *inst, const VkInstanceCreateInfo *pCreateInfo,
                                    const struct loader_layer_list *instance_layers);

VkResult loader_create_instance_chain(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                      struct loader_instance *inst, VkInstance *created_instance);

void loaderActivateInstanceLayerExtensions(struct loader_instance *inst, VkInstance created_inst);

VKAPI_ATTR VkResult VKAPI_CALL loader_layer_create_device(VkInstance instance, VkPhysicalDevice physicalDevice,
                                                          const VkDeviceCreateInfo *pCreateInfo,
                                                          const VkAllocationCallbacks *pAllocator, VkDevice *pDevice,
                                                          PFN_vkGetInstanceProcAddr layerGIPA, PFN_vkGetDeviceProcAddr *nextGDPA);
VKAPI_ATTR void VKAPI_CALL loader_layer_destroy_device(VkDevice device, const VkAllocationCallbacks *pAllocator,
                                                       PFN_vkDestroyDevice destroyFunction);

VkResult loader_create_device_chain(const VkPhysicalDevice pd, const VkDeviceCreateInfo *pCreateInfo,
                                    const VkAllocationCallbacks *pAllocator, const struct loader_instance *inst,
                                    struct loader_device *dev, PFN_vkGetInstanceProcAddr callingLayer,
                                    PFN_vkGetDeviceProcAddr *layerNextGDPA);

VkResult loader_validate_device_extensions(struct loader_instance *this_instance,
                                           const struct loader_layer_list *activated_device_layers,
                                           const struct loader_extension_list *icd_exts, const VkDeviceCreateInfo *pCreateInfo);

VkResult setupLoaderTrampPhysDevs(VkInstance instance);
VkResult setupLoaderTermPhysDevs(struct loader_instance *inst);

VkStringErrorFlags vk_string_validate(const int max_length, const char *char_array);

#endif  // LOADER_H
