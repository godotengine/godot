#ifndef COLOR_GRADING_RESOURCE_H
#define COLOR_GRADING_RESOURCE_H

#include "core/io/resource.h"

class ColorGradingResource : public Resource {
	GDCLASS(ColorGradingResource, Resource);

	// Basic color adjustments
	bool enabled = true;
	float exposure = 0.0f; // -5.0 to 5.0 EV
	float contrast = 1.0f; // 0.0 to 2.0
	float saturation = 1.0f; // 0.0 to 2.0
	float temperature = 0.0f; // -100 to 100 (blue to orange)
	float tint = 0.0f; // -100 to 100 (magenta to green)
	float hue_shift = 0.0f; // -180 to 180 degrees

protected:
	static void _bind_methods();

public:
	// Getters/setters
	void set_enabled(bool p_enabled) {
		if (enabled == p_enabled) {
			return;
		}
		enabled = p_enabled;
		emit_changed();
	}
	bool get_enabled() const { return enabled; }

	void set_exposure(float p_exposure) {
		const float clamped = CLAMP(p_exposure, -5.0f, 5.0f);
		if (exposure == clamped) {
			return;
		}
		exposure = clamped;
		emit_changed();
	}
	float get_exposure() const { return exposure; }

	void set_contrast(float p_contrast) {
		const float clamped = CLAMP(p_contrast, 0.0f, 2.0f);
		if (contrast == clamped) {
			return;
		}
		contrast = clamped;
		emit_changed();
	}
	float get_contrast() const { return contrast; }

	void set_saturation(float p_saturation) {
		const float clamped = CLAMP(p_saturation, 0.0f, 2.0f);
		if (saturation == clamped) {
			return;
		}
		saturation = clamped;
		emit_changed();
	}
	float get_saturation() const { return saturation; }

	void set_temperature(float p_temp) {
		const float clamped = CLAMP(p_temp, -100.0f, 100.0f);
		if (temperature == clamped) {
			return;
		}
		temperature = clamped;
		emit_changed();
	}
	float get_temperature() const { return temperature; }

	void set_tint(float p_tint) {
		const float clamped = CLAMP(p_tint, -100.0f, 100.0f);
		if (tint == clamped) {
			return;
		}
		tint = clamped;
		emit_changed();
	}
	float get_tint() const { return tint; }

	void set_hue_shift(float p_hue) {
		const float clamped = CLAMP(p_hue, -180.0f, 180.0f);
		if (hue_shift == clamped) {
			return;
		}
		hue_shift = clamped;
		emit_changed();
	}
	float get_hue_shift() const { return hue_shift; }

	void reset_to_defaults() {
		enabled = true;
		exposure = 0.0f;
		contrast = 1.0f;
		saturation = 1.0f;
		temperature = 0.0f;
		tint = 0.0f;
		hue_shift = 0.0f;
		emit_changed();
	}

	ColorGradingResource();
};

#endif // COLOR_GRADING_RESOURCE_H
