/**************************************************************************/
/*  export_plugin.cpp                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "export_plugin.h"

#include "godot_plugin_config.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/io/pck_packer.h"
#include "core/io/plist.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "core/string/translation.h"
#include "core/string/translation_server.h"
#include "core/string/ustring.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/editor_settings.h"
#include "editor/editor_translation.h"
#include "editor/export/editor_export.h"
#include "editor/export/editor_export_platform.h"
#include "editor/export/editor_export_plugin.h"
#include "editor/themes/editor_scale.h"
#include "modules/svg/image_loader_svg.h"
#include "scene/resources/texture.h"

#ifdef MACOS_ENABLED
#include <sys/utsname.h>
#include <unistd.h>
#endif

// Use TranslationServer for TTR macro with proper String conversion
#define TTR(m_text) String(TranslationServer::get_singleton()->translate(m_text))

// Apple TV logo
static const char *_tvos_logo_svg =
		"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
		"<svg\n"
		"   width=\"512\"\n"
		"   height=\"512\"\n"
		"   viewBox=\"0 0 512 512\"\n"
		"   version=\"1.1\"\n"
		"   xmlns=\"http://www.w3.org/2000/svg\">\n"
		"  <rect\n"
		"     style=\"fill:#000000;fill-opacity:1;stroke:none;stroke-width:4.13943;stroke-linecap:round;stroke-opacity:1\"\n"
		"     width=\"512\"\n"
		"     height=\"512\"\n"
		"     x=\"0\"\n"
		"     y=\"0\"\n"
		"     rx=\"45\"\n"
		"     ry=\"45\" />\n"
		"  <path\n"
		"     style=\"fill:#ffffff;fill-opacity:1;stroke-width:0.938225\"\n"
		"     d=\"M 297.07666,113.85838 C 264.83932,114.67547 232.89196,139.34482 221.77802,170.89133 216.70389,184.31151 215.14345,199.1163 215.74571,213.35992 c 0.10882,2.56809 0.31317,5.12671 0.58905,7.67463 -4.31264,0.39346 -8.67713,0.36881 -12.99093,-0.0761 -12.53339,-1.26301 -24.7671,-5.27968 -35.98755,-11.15371 -13.01443,-6.8378 -24.93683,-15.94461 -35.12731,-26.71639 -1.46468,-1.5388 -2.89562,-3.11044 -4.36085,-4.63905 -5.62994,5.58144 -10.29325,12.08319 -13.6924,19.20349 -6.65489,13.66642 -7.96273,29.61343 -3.69239,44.07507 2.78361,9.2518 7.54382,17.8158 13.75947,25.1004 3.48569,4.18045 7.47648,7.90792 11.8033,11.15372 -8.69968,2.79642 -16.89088,7.00292 -24.22687,12.43462 -12.85666,9.35148 -23.12742,22.34242 -28.65584,37.16641 -8.19232,21.73352 -6.39862,46.34735 4.69711,66.3632 7.3584,13.19518 18.16775,24.19143 30.9436,31.44652 20.17168,11.42803 44.58348,13.97507 66.94169,7.00703 12.87926,-3.96893 24.61183,-11.12026 34.10125,-20.75258 2.00168,-2.05642 3.9024,-4.21172 5.69095,-6.45636 1.78956,2.24464 3.68961,4.39994 5.69094,6.45636 9.51081,9.63232 21.24239,16.78365 34.12146,20.75258 20.22694,6.30932 42.14517,5.09605 61.31265,-3.42442 11.86732,-5.24481 22.49273,-13.36025 30.72981,-23.84747 11.32678,-14.50053 17.1779,-32.66881 16.41442,-50.79351 -0.39546,-11.28776 -3.39464,-22.41551 -8.67949,-32.41314 -7.72639,-14.82398 -18.80368,-27.50883 -33.05078,-35.28754 2.38035,-2.30142 4.63905,-4.72876 6.76143,-7.28012 3.91444,-4.67301 7.32897,-9.77891 10.2079,-15.19183 6.66477,-12.55599 9.73274,-26.98898 8.78768,-41.18308 -0.73428,-11.61357 -4.35056,-22.99251 -10.46173,-32.87858 -11.45935,-18.57033 -31.92725,-30.65343 -53.69246,-31.74152 -6.27453,-0.36881 -12.5839,-0.0761 -18.7684,0.93655 -9.14368,-9.45027 -20.35326,-16.92739 -32.61616,-21.60181 -11.58029,-4.42976 -24.00323,-6.54566 -36.38622,-6.25235 -0.8456,-0.0388 -1.69081,-0.0592 -2.53641,-0.0694 z m 0.69724,25.62841 c 10.21791,-0.23481 20.4392,1.63197 29.94457,5.38793 13.01442,5.136 24.59082,13.75946 33.26525,24.94161 -2.89562,2.09966 -5.63022,4.39993 -8.13642,6.92326 -11.05348,11.34524 -17.46351,26.71612 -17.7265,42.29742 -0.21359,10.47133 2.36022,20.93354 7.38854,30.09633 0.02,0.042 0.042,0.0827 0.0639,0.12482 -13.88373,-0.89332 -27.81391,1.91444 -40.37807,8.29079 -9.32902,4.73843 -17.76981,11.23692 -24.80994,19.05215 -0.0416,-0.73428 -0.092,-1.46857 -0.16807,-2.20285 -2.13344,-21.18079 0.46159,-42.76917 7.54981,-62.72626 10.92,-30.81321 32.01453,-57.03626 59.6839,-74.40266 -1.36584,-0.0592 -2.73134,-0.0894 -4.09718,-0.0894 -11.85765,0 -23.62797,2.38034 -34.55668,6.98707 -2.47902,1.05504 -4.89335,2.22309 -7.24724,3.49148 -6.72862,-5.0336 -14.27642,-8.9823 -22.247,-11.59341 -10.23802,-3.40318 -21.13715,-4.13904 -31.8004,-2.14579 -9.62848,1.7896 -18.78942,5.82115 -26.55372,11.63649 -2.35089,-1.26862 -4.75971,-2.43667 -7.24723,-3.49148 -10.92871,-4.60673 -22.69779,-6.98707 -34.55667,-6.98707 -1.36607,0 -2.7318,0.0296 -4.09741,0.0894 27.66936,17.3664 48.76389,43.58945 59.68389,74.40266 7.08855,19.95709 9.68359,41.54547 7.55014,62.72626 -0.076,0.73428 -0.1265,1.46857 -0.16806,2.20285 -7.0398,-7.81523 -15.48116,-14.31372 -24.81018,-19.05215 -12.56383,-6.37635 -26.49402,-7.18411 -40.37807,-8.29079 0.0219,-0.042 0.0439,-0.0827 0.0643,-0.12482 5.02831,-9.16279 7.60189,-19.625 7.3883,-30.09633 -0.26276,-15.5813 -6.67278,-30.95218 -17.72626,-42.29742 -2.5062,-2.52333 -5.24081,-4.8236 -8.13642,-6.92326 8.67443,-11.18215 20.25083,-19.80561 33.26526,-24.94161 9.50559,-3.75596 19.72689,-5.62274 29.94479,-5.38793 10.0469,0.23482 19.95675,2.69793 28.85043,7.10547 l 0.32021,0.16806 0.32022,0.16806 c 7.20468,3.34015 13.81607,7.85831 19.54961,13.31413 l 0.59871,0.59872 0.59872,0.59871 c 0.73428,0.87823 1.43548,1.77975 2.11453,2.70424 0.67905,-0.92449 1.38025,-1.82601 2.11453,-2.70424 l 0.59872,-0.59871 0.59871,-0.59872 c 5.73365,-5.45582 12.34528,-9.97398 19.54985,-13.31413 l 0.32022,-0.16806 0.32021,-0.16806 c 8.89345,-4.40754 18.80329,-6.87065 28.85043,-7.10547 z m -152.95415,169.3944 c 4.82845,0.31317 9.69326,0.0643 14.4401,-0.73428 13.4628,-2.30143 26.13154,-8.66 36.62004,-18.03597 10.26784,-9.14368 17.93439,-21.17125 21.90707,-34.37883 0.51315,6.88762 1.48869,13.73917 2.92873,20.4805 4.64978,22.03245 15.19237,42.3969 30.32147,58.16497 15.12944,15.76806 35.14782,27.15696 56.96212,32.67697 -15.36073,2.09934 -30.12617,9.01941 -41.74636,19.70693 -17.32315,16.09788 -27.20468,39.62417 -26.7537,63.51149 -17.32315,-13.19513 -35.40118,-25.37449 -54.88023,-35.0333 -5.21861,-2.58801 -10.52689,-5.00786 -15.91847,-7.22682 -5.39135,2.21896 -10.69963,4.63881 -15.91846,7.22682 -19.47906,9.65881 -37.55732,21.83817 -54.88047,35.0333 0.45098,-23.88732 -9.43055,-47.41361 -26.75347,-63.51149 -11.62018,-10.68752 -26.38562,-17.60759 -41.74658,-19.70693 21.8143,-5.52001 41.83268,-16.90891 56.96235,-32.67697 15.12886,-15.76807 25.67146,-36.13252 30.32124,-58.16497 1.44026,-6.74133 2.41559,-13.59288 2.92873,-20.4805 3.97268,13.20758 11.63922,25.23515 21.90707,34.37883 10.48849,9.37597 23.15724,15.73454 36.62004,18.03597 1.58264,0.26754 3.17495,0.47982 4.76991,0.63576 1.59451,0.15617 3.19574,0.2337 4.79787,0.29705 1.60212,0.0634 3.20436,0.0564 4.80754,0.0213 0.80241,-0.0172 1.60493,-0.0461 2.40502,-0.0853 0.80032,-0.0394 1.60042,-0.0915 2.39976,-0.15777 z\"\n"
		"     />\n"
		"</svg>";

// Apple TV run icon
static const char *_tvos_run_icon_svg =
		"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
		"<svg\n"
		"   width=\"512\"\n"
		"   height=\"512\"\n"
		"   viewBox=\"0 0 512 512\"\n"
		"   version=\"1.1\"\n"
		"   xmlns=\"http://www.w3.org/2000/svg\">\n"
		"  <rect\n"
		"     style=\"fill:#000000;fill-opacity:1;stroke:none;stroke-width:4.13943;stroke-linecap:round;stroke-opacity:1\"\n"
		"     width=\"512\"\n"
		"     height=\"512\"\n"
		"     x=\"0\"\n"
		"     y=\"0\"\n"
		"     rx=\"45\"\n"
		"     ry=\"45\" />\n"
		"  <path\n"
		"     style=\"fill:#ffffff;fill-opacity:1;stroke-width:1.17647\"\n"
		"     d=\"m 188.36464,127.99984 c -2.75951,0 -5.51957,1.05098 -7.61765,3.15294 -4.21569,4.21566 -4.21569,11.04115 0,15.25683 l 126.58811,126.5881 -126.58811,126.58808 c -4.21569,4.21567 -4.21569,11.04116 0,15.25684 4.21607,4.21607 11.01961,4.21607 15.23529,0 l 134.11762,-134.11758 c 4.21609,-4.21606 4.21609,-11.01959 0,-15.23529 l -134.11762,-134.11762 c -2.09806,-2.10196 -4.85814,-3.1723 -7.61764,-3.1723 z\"\n"
		"     />\n"
		"</svg>";

struct APIAccessInfo {
	String prop_name;
	String type_name;
	Vector<String> prop_flag_value;
	Vector<String> prop_flag_name;
	int default_value;
};

struct DataCollectionInfo {
	String prop_name;
	String type_name;
};

// API Access data
static const APIAccessInfo api_access[] = {
	{ "camera", "NSPrivacyAccessedAPICategoryCamera", { "3C9D.1", "E335.1", "7D7C.1", "A6E4.1" }, { "User-generated content", "Personal health research", "AR/VR health research", "Telemedicine" }, 2 },
	{ "microphone", "NSPrivacyAccessedAPICategoryMicrophone", { "B77B.1", "3C9D.1", "E1AA.1", "A6E4.1" }, { "Audio communication", "User-generated content", "Sound (non speech) analysis", "Telemedicine" }, 2 },
	{ "bluetooth", "NSPrivacyAccessedAPICategoryBluetooth", { "3737.1", "78A6.1", "94F5.1", "3886.1", "D59A.1", "D5D2.1", "BFFA.1" }, { "Connected home", "Multimedia devices", "Nearby sensors", "Controlling medical devices", "Educational hardware", "Remote controlling apps", "Internet of Things" }, 2 },
	{ "address_book", "NSPrivacyAccessedAPICategoryContacts", { "9CCA.1", "F5CC.1", "6F97.1" }, { "App communication", "Finding Friends", "Address import or sync" }, 2 },
	{ "photos", "NSPrivacyAccessedAPICategoryPhotos", { "3C9D.1", "C68D.1", "D510.1", "3E51.1", "EBFD.1" }, { "User-generated content", "Editing and customization", "Media sharing or printing", "Media playback", "Photography" }, 1 },
	{ "photos_add_only", "NSPrivacyAccessedAPICategoryPhotosAddOnly", { "9CDF.1", "3C9D.1", "E7AB.1", "D510.1", "EBFD.1" }, { "QR code scanning", "User-generated content", "Barcode scanning", "Media sharing or printing", "Photography" }, 1 },
	{ "location", "NSPrivacyAccessedAPICategoryLocation", { "5CBA.1", "FAE6.1", "9CCA.1", "8FD9.1", "B2B3.1", "C59E.1", "E351.1", "C07F.1", "F9BF.1", "C5D4.1", "E4F8.1", "F5CC.1", "D3C3.1", "EA0B.1", "A19B.1", "E034.1", "B36B.1", "8354.1", "5342.1" }, { "Store or points of interest finding", "Maps", "App communication", "Timed pick up", "Emergency services", "Booking or ride reservations", "Travel planning", "Food delivery", "Customer service or call routing", "Fitness", "Navigating or routing", "Finding Friends", "News", "Customer rewards", "Event or transportation ticketing", "Shipping or delivery tracking", "Sports", "Indoor mapping", "Ridesharing or hailing" }, 3 },
	{ "tracking", "NSPrivacyAccessedAPICategoryTracking", { "A149.1", "A149.2", "A149.3", "A149.4", "7F47.1" }, { "Analytics", "Product personalization", "Marketing", "Improving the app", "Third-party advertising" }, 1 },
	{ "user_data", "NSPrivacyAccessedAPICategoryUserData", { "35F9.1", "8FFB.1", "3D61.1" }, { "Measure time on-device", "Calculate absolute event timestamps", "User-initiated bug report" }, 1 },
	{ "disk_space", "NSPrivacyAccessedAPICategoryDiskSpace", { "E174.1", "85F4.1", "7D9E.1", "B728.1" }, { "Write or delete file on-device", "Display to user on-device", "User-initiated bug report", "Health research app" }, 3 },
	{ "active_keyboard", "NSPrivacyAccessedAPICategoryActiveKeyboards", { "3EC4.1", "54BD.1" }, { "Custom keyboard app on-device", "Customize UI on-device:2" }, 0 },
	{ "user_defaults", "NSPrivacyAccessedAPICategoryUserDefaults", { "1C8F.1", "AC6B.1", "CA92.1" }, { "Store app settings on-device", "Store app state on-device", "User-initiated bug report" }, 3 },
	{ "home_data", "NSPrivacyAccessedAPICategoryHome", { "3737.1", "9CCA.1", "E9B8.1" }, { "Connected home", "App communication", "Home automation" }, 2 },
	{ "apple_tvs", "NSPrivacyAccessedAPICategoryAppleTVs", { "3E51.1", "78A6.1", "94F5.1", "D5D2.1" }, { "Media playback", "Multimedia devices", "Nearby sensors", "Remote controlling apps" }, 2 },
};

// Data Collection info
static const DataCollectionInfo data_collection_info[] = {
	{ "usage_data", "NSPrivacyCollectedDataTypeUsageData" },
	{ "contact_info", "NSPrivacyCollectedDataTypeContactInfo" },
	{ "identifiers", "NSPrivacyCollectedDataTypeIdentifiers" },
	{ "purchase_history", "NSPrivacyCollectedDataTypePurchaseHistory" },
	{ "financial_info", "NSPrivacyCollectedDataTypeFinancialInfo" },
	{ "location", "NSPrivacyCollectedDataTypeLocation" },
	{ "sensitive_info", "NSPrivacyCollectedDataTypeSensitiveInfo" },
	{ "contacts", "NSPrivacyCollectedDataTypeContacts" },
	{ "audio_data", "NSPrivacyCollectedDataTypeAudioData" },
	{ "gameplay_content", "NSPrivacyCollectedDataTypeGameplayContent" },
	{ "browsing_history", "NSPrivacyCollectedDataTypeBrowsingHistory" },
	{ "user_content", "NSPrivacyCollectedDataTypeUserContent" },
	{ "health", "NSPrivacyCollectedDataTypeHealth" },
	{ "fitness", "NSPrivacyCollectedDataTypeFitness" },
	{ "messages", "NSPrivacyCollectedDataTypeMessages" },
	{ "emails", "NSPrivacyCollectedDataTypeEmails" },
	{ "photos_or_videos", "NSPrivacyCollectedDataTypePhotosOrVideos" },
	{ "search_history", "NSPrivacyCollectedDataTypeSearchHistory" },
};

void EditorExportPlatformTVOS::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) const {
	// Shared iOS and tvOS features
	String driver = ProjectSettings::get_singleton()->get("rendering/renderer/rendering_method");
	if (p_preset->get("texture_format/s3tc")) {
		r_features->push_back("s3tc");
	}
	if (p_preset->get("texture_format/etc")) {
		r_features->push_back("etc");
	}
	if (p_preset->get("texture_format/etc2")) {
		r_features->push_back("etc2");
	}

	// tvOS-specific features
	r_features->push_back("tvos");
	r_features->push_back("arm64");
	
	// Driver feature tags
	if (driver == "vulkan") {
		r_features->push_back("vulkan");
	} else if (driver == "opengl3") {
		r_features->push_back("opengl3");
	} else if (driver == "metal") {
		r_features->push_back("metal");
	}
}

Vector<EditorExportPlatformTVOS::ExportArchitecture> EditorExportPlatformTVOS::_get_supported_architectures() const {
	Vector<ExportArchitecture> archs;
	archs.push_back(ExportArchitecture("arm64", true));
	return archs;
}

Vector<String> EditorExportPlatformTVOS::_get_preset_architectures(const Ref<EditorExportPreset> &p_preset) const {
	Vector<String> architectures;
	Vector<ExportArchitecture> all_archs = _get_supported_architectures();
	for (int i = 0; i < all_archs.size(); ++i) {
		if (p_preset->get("architectures/" + all_archs[i].name)) {
			architectures.push_back(all_archs[i].name);
		}
	}
	return architectures;
}

void EditorExportPlatformTVOS::get_export_options(List<ExportOption> *r_options) const {
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.xcframework"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.xcframework"), ""));

	// Architectures
	const Vector<ExportArchitecture> architectures = _get_supported_architectures();
	for (int i = 0; i < architectures.size(); ++i) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "architectures/" + architectures[i].name), architectures[i].is_default));
	}

	// Application
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/app_store_team_id"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/bundle_identifier", PROPERTY_HINT_PLACEHOLDER_TEXT, "com.example.game"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/signature"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/short_version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/category", PROPERTY_HINT_ENUM, "Business,Education,Entertainment,Games,Music,News"), "Games"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "application/export_project_only"), false));

	// Texture formats
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), true));

	// Launch screen storyboard
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "launch_screens/generate_launch_image", PROPERTY_HINT_ENUM, "Based on Project Icon,Custom"), "Based on Project Icon"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "launch_screens/custom_launch_image", PROPERTY_HINT_GLOBAL_FILE, "*.png"), ""));

	// Privacy usage descriptions
	r_options->push_back(ExportOption(PropertyInfo(Variant::DICTIONARY, "privacy/permissions"), Dictionary()));

	// API access usage descriptions
	int api_access_size = sizeof(api_access) / sizeof(api_access[0]);
	for (int i = 0; i < api_access_size; ++i) {
		const APIAccessInfo &api_info = api_access[i];
		bool default_enabled = api_info.default_value == 2;
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "privacy/usage_descriptions_enabled/" + api_info.prop_name), default_enabled));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "privacy/usage_descriptions/" + api_info.prop_name, PROPERTY_HINT_MULTILINE_TEXT), ""));
	}

	// Data collection settings
	for (int i = 0; i < sizeof(data_collection_info) / sizeof(data_collection_info[0]); ++i) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "privacy/data_collection_enabled/" + data_collection_info[i].prop_name), false));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "privacy/data_collection_use_identifiers/" + data_collection_info[i].prop_name), false));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "privacy/data_collection_use_tracking/" + data_collection_info[i].prop_name), false));
	}

	// Top-shelf image
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "top_shelf/enable"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "top_shelf/image", PROPERTY_HINT_GLOBAL_FILE, "*.png"), ""));

	// Remote control settings
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "gamepad/remote_support"), true));

	// Capabilities
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/push_notifications"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/game_controllers"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/topShelf"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/audio_background"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/network_background"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/metal"), false));

	// Code signing
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "provisioning_profile/debug", PROPERTY_HINT_GLOBAL_FILE, "*.mobileprovision"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "provisioning_profile/release", PROPERTY_HINT_GLOBAL_FILE, "*.mobileprovision"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "provisioning_profile/app_store", PROPERTY_HINT_GLOBAL_FILE, "*.mobileprovision"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "provisioning_profile/ad_hoc", PROPERTY_HINT_GLOBAL_FILE, "*.mobileprovision"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "provisioning_profile/distribution", PROPERTY_HINT_GLOBAL_FILE, "*.mobileprovision"), ""));

	// List plugins
	Vector<PluginConfigIOS> plugin_configs = get_plugins();
	for (int i = 0; i < plugin_configs.size(); i++) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "plugins/" + plugin_configs[i].name), false));
	}
}

bool EditorExportPlatformTVOS::get_export_option_visibility(const EditorExportPreset *p_preset, const String &p_option) const {
	if (p_option == "application/app_store_team_id" || p_option == "application/provisioning_profile_uuid_debug" || p_option == "application/provisioning_profile_uuid_release") {
		return false;
	}

	if (p_option.begins_with("launch_screens/")) {
		return true;
	}

	if (p_option.begins_with("privacy/usage_descriptions/")) {
		String feature_name = p_option.get_slice("/", 2);
		bool enabled = p_preset->get("privacy/usage_descriptions_enabled/" + feature_name);
		return enabled;
	}

	if (p_option.begins_with("privacy/data_collection_use_identifiers/") || p_option.begins_with("privacy/data_collection_use_tracking/")) {
		String feature_name = p_option.get_slice("/", 2);
		bool enabled = p_preset->get("privacy/data_collection_enabled/" + feature_name);
		return enabled;
	}

	if (p_option == "top_shelf/image") {
		bool top_shelf_enabled = p_preset->get("top_shelf/enable");
		return top_shelf_enabled;
	}

	return true;
}

String EditorExportPlatformTVOS::get_export_option_warning(const EditorExportPreset *p_preset, const StringName &p_name) const {
	if (p_name == "application/bundle_identifier" && p_preset) {
		String bundle_id = p_preset->get("application/bundle_identifier");
		if (!is_package_name_valid(bundle_id)) {
			return TTR("Invalid bundle identifier. Valid characters are alphanumeric (A-Z, a-z, 0-9), hyphen (-), and period (.).");
		}
	} else if (p_name == "application/category" && p_preset) {
		String category = p_preset->get("application/category");
		if (category.length() == 0) {
			return TTR("Application category must be specified.");
		}
	}

	return String();
}

void EditorExportPlatformTVOS::_notification(int p_what) {
#ifdef MACOS_ENABLED
	if (p_what == NOTIFICATION_POSTINITIALIZE) {
		if (EditorExport::get_singleton()) {
			EditorExport::get_singleton()->connect_presets_runnable_updated(callable_mp(this, &EditorExportPlatformTVOS::_update_preset_status));
		}
	}
#endif
}

bool EditorExportPlatformTVOS::is_package_name_valid(const String &p_package, String *r_error) const {
	String pname = p_package;

	if (pname.length() == 0) {
		if (r_error) {
			*r_error = TTR("Bundle identifier is missing.");
		}
		return false;
	}

	for (int i = 0; i < pname.length(); i++) {
		char32_t c = pname[i];
		if (!(is_ascii_alphanumeric_char(c) || c == '-' || c == '.')) {
			if (r_error) {
				*r_error = vformat(TTR("The character '%c' is not allowed in bundle identifier."), String::chr(c));
			}
			return false;
		}
	}

	return true;
}

#ifdef MACOS_ENABLED
bool EditorExportPlatformTVOS::_check_xcode_install() {
	// TODO: Implement this
	return true;
}

void EditorExportPlatformTVOS::_check_for_changes_poll_thread(void *ud) {
	EditorExportPlatformTVOS *ea = static_cast<EditorExportPlatformTVOS *>(ud);

	while (!ea->quit_request.is_set()) {
		// TODO: Implement device checking logic
		
		OS::get_singleton()->delay_usec(1000000); // 1 second
	}
}

void EditorExportPlatformTVOS::_update_preset_status() {
	bool has_runnable = false;
	bool devices_available = false;

	// TODO: Check for presets and devices
	
	has_runnable_preset.set_to(has_runnable);
	if (devices_available != devices.size() > 0) {
		devices_changed.set();
	}
}
#endif

Error EditorExportPlatformTVOS::run(const Ref<EditorExportPreset> &p_preset, int p_device, BitField<EditorExportPlatform::DebugFlags> p_debug_flags) {
#ifdef MACOS_ENABLED
	if (!_check_xcode_install()) {
		return Error::ERR_UNAVAILABLE;
	}
	
	// TODO: Implement device running functionality
	
	return OK;
#else
	return Error::ERR_UNCONFIGURED;
#endif
}

int EditorExportPlatformTVOS::get_options_count() const {
	// TODO: Implement this, should handle additional connected devices
	return 0;
}

String EditorExportPlatformTVOS::get_options_tooltip() const {
	return TTR("Run on Apple TV Device");
}

Ref<ImageTexture> EditorExportPlatformTVOS::get_option_icon(int p_index) const {
	// TODO: Implement device-specific icons
	return run_icon;
}

String EditorExportPlatformTVOS::get_option_label(int p_index) const {
	// TODO: Return appropriate device names
	return String();
}

String EditorExportPlatformTVOS::get_option_tooltip(int p_index) const {
	// TODO: Return device info
	return String();
}

HashMap<String, Variant> EditorExportPlatformTVOS::get_custom_project_settings(const Ref<EditorExportPreset> &p_preset) const {
	HashMap<String, Variant> settings;

	// Configurables
	settings["application/bundle_identifier"] = p_preset->get("application/bundle_identifier");
	settings["application/signature"] = p_preset->get("application/signature");
	settings["application/short_version"] = p_preset->get("application/short_version");
	settings["application/version"] = p_preset->get("application/version");
	settings["application/category"] = p_preset->get("application/category");
	settings["application/copyright"] = p_preset->get("application/copyright");

	// Apple TV specific
	settings["application/top_shelf_enabled"] = p_preset->get("top_shelf/enable");
	settings["application/top_shelf_image"] = p_preset->get("top_shelf/image");
	settings["application/gamepad_remote_support"] = p_preset->get("gamepad/remote_support");

	// Capabilities
	settings["capabilities/push_notifications"] = p_preset->get("capabilities/push_notifications");
	settings["capabilities/game_controllers"] = p_preset->get("capabilities/game_controllers");
	settings["capabilities/topShelf"] = p_preset->get("capabilities/topShelf");
	settings["capabilities/audio_background"] = p_preset->get("capabilities/audio_background");
	settings["capabilities/network_background"] = p_preset->get("capabilities/network_background");
	settings["capabilities/metal"] = p_preset->get("capabilities/metal");

	return settings;
}

void EditorExportPlatformTVOS::_blend_and_rotate(Ref<Image> &p_dst, Ref<Image> &p_src, bool p_rot) {
	ERR_FAIL_COND(p_dst.is_null());
	ERR_FAIL_COND(p_src.is_null());

	int sw = p_rot ? p_src->get_height() : p_src->get_width();
	int sh = p_rot ? p_src->get_width() : p_src->get_height();

	int x_pos = (p_dst->get_width() - sw) / 2;
	int y_pos = (p_dst->get_height() - sh) / 2;

	int xs = p_rot ? 0 : 0;
	int ys = p_rot ? p_src->get_width() : 0;
	int ws = p_rot ? p_src->get_height() : p_src->get_width();
	int hs = p_rot ? p_src->get_width() : p_src->get_height();

	for (int y = 0; y < hs; y++) {
		for (int x = 0; x < ws; x++) {
			Color sc = p_src->get_pixel(xs + (p_rot ? y : x), ys + (p_rot ? x : y));
			Color dc = p_dst->get_pixel(x_pos + x, y_pos + y);
			dc.r = (double)(sc.a * sc.r + dc.a * (1.0 - sc.a) * dc.r);
			dc.g = (double)(sc.a * sc.g + dc.a * (1.0 - sc.a) * dc.g);
			dc.b = (double)(sc.a * sc.b + dc.a * (1.0 - sc.a) * dc.b);
			dc.a = (double)(sc.a + dc.a * (1.0 - sc.a));
			p_dst->set_pixel(x_pos + x, y_pos + y, dc);
		}
	}
}

Error EditorExportPlatformTVOS::_export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) {
	// TODO: Implement with tvOS-specific icon sizes
	return OK;
}

Error EditorExportPlatformTVOS::_export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) {
	// TODO: Implement Launch Screen storyboard creation
	return OK;
}

Error EditorExportPlatformTVOS::_walk_dir_recursive(Ref<DirAccess> &p_da, FileHandler p_handler, void *p_userdata) {
	Vector<String> dirs;
	String current_dir = p_da->get_current_dir();

	p_da->list_dir_begin();
	String path = p_da->get_next();
	while (!path.is_empty()) {
		if (path == "." || path == "..") {
			path = p_da->get_next();
			continue;
		}

		if (p_da->current_is_hidden()) {
			path = p_da->get_next();
			continue;
		}

		if (p_da->current_is_dir()) {
			dirs.push_back(path);
		} else {
			Error err = p_handler(current_dir.path_join(path), p_userdata);
			if (err) {
				p_da->list_dir_end();
				return err;
			}
		}

		path = p_da->get_next();
	}
	p_da->list_dir_end();

	for (int i = 0; i < dirs.size(); i++) {
		String dir = dirs[i];
		p_da->change_dir(dir);
		Error err = _walk_dir_recursive(p_da, p_handler, p_userdata);
		if (err) {
			return err;
		}
		p_da->change_dir("..");
	}

	return OK;
}

Error EditorExportPlatformTVOS::_codesign(String p_file, void *p_userdata) {
	struct CodeSignData {
		const Ref<EditorExportPreset> &preset;
		bool development;

		CodeSignData(const Ref<EditorExportPreset> &p_preset, bool p_development) :
				preset(p_preset), development(p_development) {
		}
	};

	ERR_FAIL_COND_V(p_userdata == nullptr, Error::FAILED);
	CodeSignData *data = static_cast<CodeSignData *>(p_userdata);

	String file = p_file.replace(" ", "\\ ");
	String identifier = data->preset->get("application/bundle_identifier");
	String team_id = data->preset->get("application/app_store_team_id");
	
	// TODO: Implement codesigning process
	
	return OK;
}

void EditorExportPlatformTVOS::_check_xcframework_content(const String &p_path, int &r_total_libs, int &r_static_libs, int &r_dylibs, int &r_frameworks) const {
	Ref<PList> plist;
	plist.instantiate();
	plist->load_file(p_path.path_join("Info.plist"));
	Ref<PListNode> root_node = plist->get_root();
	if (root_node.is_null()) {
		return;
	}
	Dictionary root = root_node->get_value();
	if (!root.has("AvailableLibraries")) {
		return;
	}
	Ref<PListNode> libs_node = root["AvailableLibraries"];
	if (libs_node.is_null()) {
		return;
	}
	Array libs = libs_node->get_value();
	r_total_libs = libs.size();
	for (int j = 0; j < libs.size(); j++) {
		Ref<PListNode> lib_node = libs[j];
		if (lib_node.is_null()) {
			return;
		}
		Dictionary lib = lib_node->get_value();
		if (lib.has("BinaryPath")) {
			Ref<PListNode> path_node = lib["BinaryPath"];
			if (path_node.is_valid()) {
				String path = path_node->get_value();
				if (path.ends_with(".a")) {
					r_static_libs++;
				}
				if (path.ends_with(".dylib")) {
					r_dylibs++;
				}
				if (path.ends_with(".framework")) {
					r_frameworks++;
				}
			}
		}
	}
}

Error EditorExportPlatformTVOS::_convert_to_framework(const String &p_source, const String &p_destination, const String &p_id) const {
	// TODO: Implement framework conversion
	return OK;
}

void EditorExportPlatformTVOS::_add_assets_to_project(const String &p_out_dir, const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_project_data, const Vector<TVOSExportAsset> &p_additional_assets) {
	// TODO: Implement asset addition to project
}

Error EditorExportPlatformTVOS::_export_additional_assets(const Ref<EditorExportPreset> &p_preset, const String &p_out_dir, const Vector<String> &p_assets, bool p_is_framework, bool p_should_embed, Vector<TVOSExportAsset> &r_exported_assets) {
	// TODO: Implement additional assets export
	return OK;
}

Error EditorExportPlatformTVOS::_copy_asset(const Ref<EditorExportPreset> &p_preset, const String &p_out_dir, const String &p_asset, const String *p_custom_file_name, bool p_is_framework, bool p_should_embed, Vector<TVOSExportAsset> &r_exported_assets) {
	// TODO: Implement asset copying
	return OK;
}

Error EditorExportPlatformTVOS::_export_additional_assets(const Ref<EditorExportPreset> &p_preset, const String &p_out_dir, const Vector<SharedObject> &p_libraries, Vector<TVOSExportAsset> &r_exported_assets) {
	// TODO: Implement additional assets export
	return OK;
}

Error EditorExportPlatformTVOS::_export_tvos_plugins(const Ref<EditorExportPreset> &p_preset, TVOSConfigData &p_config_data, const String &dest_dir, Vector<TVOSExportAsset> &r_exported_assets, bool p_debug) {
	// TODO: Implement tvOS plugins export
	return OK;
}

String EditorExportPlatformTVOS::_get_additional_plist_content() {
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	String result;
	for (int i = 0; i < export_plugins.size(); ++i) {
		result += export_plugins[i]->get_ios_plist_content();
	}
	return result;
}

String EditorExportPlatformTVOS::_get_linker_flags() {
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	String result;
	for (int i = 0; i < export_plugins.size(); ++i) {
		String flags = export_plugins[i]->get_ios_linker_flags();
		if (!flags.is_empty()) {
			if (!result.is_empty()) {
				result += ' ';
			}
			result += flags;
		}
	}
	return result;
}

String EditorExportPlatformTVOS::_get_cpp_code() {
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	String result;
	for (int i = 0; i < export_plugins.size(); ++i) {
		result += export_plugins[i]->get_ios_cpp_code();
	}
	return result;
}

void EditorExportPlatformTVOS::_fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const TVOSConfigData &p_config, bool p_debug) {
	// TODO: Implement config file fixing
}

Error EditorExportPlatformTVOS::_export_project_helper(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags, bool p_oneclick) {
	// TODO: Implement export project helper
	return OK;
}

bool EditorExportPlatformTVOS::has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug) const {
	String err;
	bool valid = true;
	bool team_id_valid = true;
	r_missing_templates = false;

	// Look for export templates
	String template_err;
	if (p_debug) {
		String template_path = String("libgodot.tvos.debug") + String(".xcframework");
		template_err = find_export_template(template_path, &template_err);
	} else {
		String template_path = String("libgodot.tvos.release") + String(".xcframework");
		template_err = find_export_template(template_path, &template_err);
	}

	if (r_missing_templates) {
		valid = false;
		err += TTR("Export template not found.") + "\n";
	}

	// Bundle ID
	String bundle_id = p_preset->get("application/bundle_identifier");
	if (!is_package_name_valid(bundle_id)) {
		err += TTR("Invalid bundle identifier:") + " " + bundle_id + "\n";
		valid = false;
	}

	// Provisioning profile
	bool uploading_for_distribution = false;
	if (!p_debug && bundle_id.contains("*")) {
		err += TTR("Provisioning profile:") + " " + TTR("App Store distribution profiles aren't supported for wildcard App Bundle ID:") + " " + bundle_id + "\n";
		valid = false;
	}

	if (uploading_for_distribution) {
		if (!p_preset->get("application/app_store_team_id").operator String().length()) {
			err += TTR("App Store Team ID not specified.") + "\n";
			team_id_valid = false;
			valid = false;
		}
	}

	if (!err.is_empty()) {
		r_error = err;
	}
	return valid;
}

bool EditorExportPlatformTVOS::has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const {
	String err;
	bool valid = true;

	// Bundle identifier must be specified
	String bundle_identifier = p_preset->get("application/bundle_identifier");
	if (bundle_identifier.is_empty()) {
		err += TTR("Bundle identifier not specified.") + "\n";
		valid = false;
	}

	// Team ID must be specified
	String team_id = p_preset->get("application/app_store_team_id");
	if (team_id.is_empty()) {
		err += TTR("App Store Team ID not specified.") + "\n";
		valid = false;
	}

	if (!err.is_empty()) {
		r_error = err;
	}
	return valid;
}

Error EditorExportPlatformTVOS::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	// Check if the export configuration is valid
	String error;
	bool missing_templates;
	if (!has_valid_export_configuration(p_preset, error, missing_templates, p_debug)) {
		ERR_PRINT(error);
		return Error::ERR_UNCONFIGURED;
	}

	// Check if the project configuration is valid
	if (!has_valid_project_configuration(p_preset, error)) {
		ERR_PRINT(error);
		return Error::ERR_UNCONFIGURED;
	}

	// Export project using helper
	bool is_one_click = (p_flags & DEBUG_FLAG_ONE_CLICK_DEPLOY) != 0;
	return _export_project_helper(p_preset, p_debug, p_path, p_flags, is_one_click);
}

EditorExportPlatformTVOS::EditorExportPlatformTVOS() {
	if (EditorNode::get_singleton()) {
		Ref<Image> img = memnew(Image);
		const bool upsample = !Math::is_equal_approx(Math::round(EDSCALE), EDSCALE);

		ImageLoaderSVG::create_image_from_string(img, _tvos_logo_svg, EDSCALE, upsample, false);
		logo = ImageTexture::create_from_image(img);

		ImageLoaderSVG::create_image_from_string(img, _tvos_run_icon_svg, EDSCALE, upsample, false);
		run_icon = ImageTexture::create_from_image(img);

		plugins_changed.set();
		devices_changed.set();
#ifdef MACOS_ENABLED
		_update_preset_status();
		check_for_changes_thread.start(_check_for_changes_poll_thread, this);
#endif
	}
}

EditorExportPlatformTVOS::~EditorExportPlatformTVOS() {
#ifdef MACOS_ENABLED
	quit_request.set();
	if (check_for_changes_thread.is_started()) {
		check_for_changes_thread.wait_to_finish();
	}
#endif
}