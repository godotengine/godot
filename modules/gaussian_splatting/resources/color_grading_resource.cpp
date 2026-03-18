#include "color_grading_resource.h"

ColorGradingResource::ColorGradingResource() {
	enabled = true;
	exposure = 0.0f;
	contrast = 1.0f;
	saturation = 1.0f;
	temperature = 0.0f;
	tint = 0.0f;
	hue_shift = 0.0f;
}

void ColorGradingResource::_bind_methods() {
	// Reset
	ClassDB::bind_method(D_METHOD("reset_to_defaults"), &ColorGradingResource::reset_to_defaults);

	// Enabled
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &ColorGradingResource::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &ColorGradingResource::get_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");

	// Exposure
	ClassDB::bind_method(D_METHOD("set_exposure", "exposure"), &ColorGradingResource::set_exposure);
	ClassDB::bind_method(D_METHOD("get_exposure"), &ColorGradingResource::get_exposure);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "exposure", PROPERTY_HINT_RANGE, "-5.0,5.0,0.01,or_greater,or_less"),
			"set_exposure", "get_exposure");

	// Contrast
	ClassDB::bind_method(D_METHOD("set_contrast", "contrast"), &ColorGradingResource::set_contrast);
	ClassDB::bind_method(D_METHOD("get_contrast"), &ColorGradingResource::get_contrast);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "contrast", PROPERTY_HINT_RANGE, "0.0,2.0,0.01"),
			"set_contrast", "get_contrast");

	// Saturation
	ClassDB::bind_method(D_METHOD("set_saturation", "saturation"), &ColorGradingResource::set_saturation);
	ClassDB::bind_method(D_METHOD("get_saturation"), &ColorGradingResource::get_saturation);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "saturation", PROPERTY_HINT_RANGE, "0.0,2.0,0.01"),
			"set_saturation", "get_saturation");

	// Temperature
	ClassDB::bind_method(D_METHOD("set_temperature", "temperature"), &ColorGradingResource::set_temperature);
	ClassDB::bind_method(D_METHOD("get_temperature"), &ColorGradingResource::get_temperature);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature", PROPERTY_HINT_RANGE, "-100.0,100.0,1.0"),
			"set_temperature", "get_temperature");

	// Tint
	ClassDB::bind_method(D_METHOD("set_tint", "tint"), &ColorGradingResource::set_tint);
	ClassDB::bind_method(D_METHOD("get_tint"), &ColorGradingResource::get_tint);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tint", PROPERTY_HINT_RANGE, "-100.0,100.0,1.0"),
			"set_tint", "get_tint");

	// Hue Shift
	ClassDB::bind_method(D_METHOD("set_hue_shift", "hue_shift"), &ColorGradingResource::set_hue_shift);
	ClassDB::bind_method(D_METHOD("get_hue_shift"), &ColorGradingResource::get_hue_shift);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "hue_shift", PROPERTY_HINT_RANGE, "-180.0,180.0,1.0,suffix:°"),
			"set_hue_shift", "get_hue_shift");
}
