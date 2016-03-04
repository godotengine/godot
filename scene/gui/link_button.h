#ifndef LINKBUTTON_H
#define LINKBUTTON_H


#include "scene/gui/base_button.h"
#include "scene/resources/bit_mask.h"

class LinkButton : public BaseButton {

	OBJ_TYPE( LinkButton, BaseButton );
public:

	enum UnderlineMode {
		UNDERLINE_MODE_ALWAYS,
		UNDERLINE_MODE_ON_HOVER
	};
private:
	String text;
	UnderlineMode underline_mode;

protected:

	virtual Size2 get_minimum_size() const;
	void _notification(int p_what);
	static void _bind_methods();

public:

	void set_text(const String& p_text);
	String get_text() const;

	void set_underline_mode(UnderlineMode p_underline_mode);
	UnderlineMode get_underline_mode() const;

	LinkButton();
};

VARIANT_ENUM_CAST( LinkButton::UnderlineMode );

#endif // LINKBUTTON_H
