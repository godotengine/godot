#include "back_buffer_copy.h"

void BackBufferCopy::_update_copy_mode() {

	switch(copy_mode) {

		case COPY_MODE_DISABLED: {

			VS::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(),false,Rect2());
		} break;
		case COPY_MODE_RECT: {

			VS::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(),true,rect);
		} break;
		case COPY_MODE_VIEWPORT: {

			VS::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(),true,Rect2());

		} break;

	}
}

Rect2 BackBufferCopy::get_item_rect() const {

	return rect;
}

void BackBufferCopy::set_rect(const Rect2& p_rect) {

	rect=p_rect;
	_update_copy_mode();
}

Rect2 BackBufferCopy::get_rect() const{
	return rect;
}

void BackBufferCopy::set_copy_mode(CopyMode p_mode){

	copy_mode=p_mode;
	_update_copy_mode();
}
BackBufferCopy::CopyMode BackBufferCopy::get_copy_mode() const{

	return copy_mode;
}


void BackBufferCopy::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_rect","rect"),&BackBufferCopy::set_rect);
	ObjectTypeDB::bind_method(_MD("get_rect"),&BackBufferCopy::get_rect);

	ObjectTypeDB::bind_method(_MD("set_copy_mode","copy_mode"),&BackBufferCopy::set_copy_mode);
	ObjectTypeDB::bind_method(_MD("get_copy_mode"),&BackBufferCopy::get_copy_mode);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"copy_mode",PROPERTY_HINT_ENUM,"Disabled,Rect,Viewport"),_SCS("set_copy_mode"),_SCS("get_copy_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::RECT2,"rect"),_SCS("set_rect"),_SCS("get_rect"));

	BIND_CONSTANT( COPY_MODE_DISABLED );
	BIND_CONSTANT( COPY_MODE_RECT );
	BIND_CONSTANT( COPY_MODE_VIEWPORT );

}

BackBufferCopy::BackBufferCopy(){

	rect=Rect2(-100,-100,200,200);
	copy_mode=COPY_MODE_RECT;
	_update_copy_mode();
}
BackBufferCopy::~BackBufferCopy(){

}
