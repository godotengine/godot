#ifndef GS_RENDER_ROUTE_LABELS_H
#define GS_RENDER_ROUTE_LABELS_H

#include "core/string/ustring.h"

namespace GaussianRenderRouteLabels {

String describe_route_uid(const String &p_route_uid);
String describe_sort_route_uid(const String &p_sort_route_uid);
String describe_cull_route_uid(const String &p_cull_route_uid);
String describe_backend_selection_reason(const String &p_reason);
String describe_cull_route_reason(const String &p_reason);

String format_route_uid(const String &p_route_uid);
String format_sort_route_uid(const String &p_sort_route_uid);
String format_cull_route_uid(const String &p_cull_route_uid);
String format_backend_selection_reason(const String &p_reason);
String format_cull_route_reason(const String &p_reason);

} // namespace GaussianRenderRouteLabels

#endif // GS_RENDER_ROUTE_LABELS_H
