#ifndef BACKBUFFERCOPY_H
#define BACKBUFFERCOPY_H

#include "scene/2d/node_2d.h"

class BackBufferCopy : public Node2D {
	OBJ_TYPE( BackBufferCopy,Node2D);
public:
	enum CopyMode {
		COPY_MODE_DISABLED,
		COPY_MODE_RECT,
		COPY_MODE_VIEWPORT
	};
private:

	Rect2 rect;
	CopyMode copy_mode;

	void _update_copy_mode();

protected:

	static void _bind_methods();

public:

	void set_rect(const Rect2& p_rect);
	Rect2 get_rect() const;

	void set_copy_mode(CopyMode p_mode);
	CopyMode get_copy_mode() const;

	Rect2 get_item_rect() const;

	BackBufferCopy();
	~BackBufferCopy();
};

VARIANT_ENUM_CAST(BackBufferCopy::CopyMode);

#endif // BACKBUFFERCOPY_H
