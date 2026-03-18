#ifndef GAUSSIAN_EDITOR_SERVICES_H
#define GAUSSIAN_EDITOR_SERVICES_H

#ifdef TOOLS_ENABLED

#include "core/error/error_list.h"
#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"

class GaussianSplatAsset;
class GaussianSplatNode3D;
class GaussianSplatRenderer;

namespace GaussianEditorServices {

int64_t dict_get_int(const Dictionary &p_dict, const StringName &p_key, int64_t p_default = 0);
double dict_get_double(const Dictionary &p_dict, const StringName &p_key, double p_default = 0.0);
bool dict_get_bool(const Dictionary &p_dict, const StringName &p_key, bool p_default = false);
String dict_get_string(const Dictionary &p_dict, const StringName &p_key, const String &p_default = String());

String format_gaussian_splat_stats(GaussianSplatNode3D *p_node, const Ref<GaussianSplatRenderer> &p_renderer);
String format_asset_metadata_summary(const Ref<GaussianSplatAsset> &p_asset, const Dictionary &p_metadata, int p_default_thumbnail_size = 128);

String describe_error(Error p_error);
String import_error_recovery_hint(Error p_error, const String &p_extension);
String format_import_failure_message(const String &p_source_path, Error p_error, const String &p_extension, const String &p_context = String());

} // namespace GaussianEditorServices

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_EDITOR_SERVICES_H
