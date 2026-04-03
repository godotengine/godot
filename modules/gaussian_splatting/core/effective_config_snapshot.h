#ifndef GAUSSIAN_EFFECTIVE_CONFIG_SNAPSHOT_H
#define GAUSSIAN_EFFECTIVE_CONFIG_SNAPSHOT_H

#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

namespace GaussianEffectiveConfig {

static inline String format_variant_value(const Variant &p_value) {
	switch (p_value.get_type()) {
		case Variant::BOOL:
			return (bool)p_value ? String("On") : String("Off");
		case Variant::INT:
			return itos((int64_t)p_value);
		case Variant::FLOAT:
			return String::num_real((double)p_value);
		case Variant::STRING:
			return String(p_value);
		default:
			return String(p_value.stringify());
	}
}

static inline Dictionary make_entry(const Variant &p_value, const String &p_source,
		const String &p_source_label, const String &p_display_value = String(),
		const String &p_note = String(), bool p_fidelity_limited = false) {
	Dictionary entry;
	entry[StringName("value")] = p_value;
	entry[StringName("source")] = p_source;
	entry[StringName("source_label")] = p_source_label;
	entry[StringName("display_value")] = p_display_value.is_empty() ? format_variant_value(p_value) : p_display_value;
	entry[StringName("note")] = p_note;
	entry[StringName("fidelity_limited")] = p_fidelity_limited;
	return entry;
}

static inline void set_entry(Dictionary &r_snapshot, const StringName &p_key, const Variant &p_value,
		const String &p_source, const String &p_source_label, const String &p_display_value = String(),
		const String &p_note = String(), bool p_fidelity_limited = false) {
	r_snapshot[p_key] = make_entry(p_value, p_source, p_source_label, p_display_value, p_note, p_fidelity_limited);
}

static inline Dictionary get_entry(const Dictionary &p_snapshot, const StringName &p_key) {
	if (!p_snapshot.has(p_key)) {
		return Dictionary();
	}
	const Variant &value = p_snapshot[p_key];
	if (value.get_type() != Variant::DICTIONARY) {
		return Dictionary();
	}
	return value;
}

static inline void merge_into(Dictionary &r_target, const Dictionary &p_source) {
	Array keys = p_source.keys();
	for (int i = 0; i < keys.size(); i++) {
		const Variant &key = keys[i];
		r_target[key] = p_source[key];
	}
}

static inline String get_display_value(const Dictionary &p_entry) {
	return String(p_entry.get(StringName("display_value"), String()));
}

static inline String get_source_label(const Dictionary &p_entry) {
	return String(p_entry.get(StringName("source_label"), String()));
}

static inline String get_source(const Dictionary &p_entry) {
	return String(p_entry.get(StringName("source"), String()));
}

static inline void mark_snapshot_limited(Dictionary &r_snapshot, const String &p_note) {
	Array keys = r_snapshot.keys();
	for (int i = 0; i < keys.size(); i++) {
		const Variant &key = keys[i];
		Dictionary entry = get_entry(r_snapshot, key);
		if (entry.is_empty()) {
			continue;
		}
		entry[StringName("fidelity_limited")] = true;
		if (!p_note.is_empty()) {
			entry[StringName("note")] = p_note;
		}
		r_snapshot[key] = entry;
	}
}

static inline String describe_route_policy_source(const String &p_source) {
	if (p_source == "route_policy") {
		return "route_policy";
	}
	if (p_source == "default_fallback") {
		return "code default";
	}
	if (p_source.is_empty()) {
		return "unknown";
	}
	return p_source;
}

} // namespace GaussianEffectiveConfig

#endif // GAUSSIAN_EFFECTIVE_CONFIG_SNAPSHOT_H
