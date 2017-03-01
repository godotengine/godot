#ifndef INPUTACTION_H
#define INPUTACTION_H

#include "resource.h"

class ShortCut : public Resource {

	GDCLASS(ShortCut,Resource);

	InputEvent shortcut;
protected:

	static void _bind_methods();
public:

	void set_shortcut(const InputEvent& p_shortcut);
	InputEvent get_shortcut() const;
	bool is_shortcut(const InputEvent& p_Event) const;
	bool is_valid() const;

	String get_as_text() const;

	ShortCut();
};

#endif // INPUTACTION_H
