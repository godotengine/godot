/*************************************************************************/
/*  shader_graph_editor_plugin.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "shader_graph_editor_plugin.h"

#if 0
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"

class _ShaderTester : public ShaderCodeGenerator {
public:

	Set<int> *_set;

	virtual void begin() {}
	virtual Error add_node(VS::ShaderNodeType p_type,int p_node_pos,int p_id,const Variant& p_param,const Vector<int>& p_in_connections,const Vector<int>& p_out_connections,const Vector<int>& p_out_connection_outputs) { if (_set) _set->insert(p_id); return OK; }
	virtual void end() {}

	_ShaderTester() { _set=NULL; }
};



void ShaderEditor::edit(Ref<Shader> p_shader) {


	shader=p_shader;

	if (shader.is_null())
		hide();
	else {
		_read_shader_graph();
	}

}

Size2 ShaderEditor::_get_maximum_size() {

	Size2 max;

	for(List<int>::Element *E=order.front();E;E=E->next()) {

		Point2 pos = Point2( shader_graph.node_get_pos_x(E->get()), shader_graph.node_get_pos_y(E->get()) );

		if (click_type==CLICK_NODE && click_node==E->get()) {

			pos+=click_motion-click_pos;
		}
		pos+=get_node_size(E->get());
		if (pos.x>max.x)
			max.x=pos.x;
		if (pos.y>max.y)
			max.y=pos.y;

	}

	return max;
}

Size2 ShaderEditor::get_node_size(int p_node) const {

	VisualServer::ShaderNodeType type=shader_graph.node_get_type(p_node);
	Ref<StyleBox> style = get_stylebox("panel","PopupMenu");
	Ref<Font> font = get_font("font","PopupMenu");
	Color font_color = get_color("font_color","PopupMenu");

	Size2 size = style->get_minimum_size();

	int count=1; // title
	count += VisualServer::shader_get_input_count( type) + VisualServer::shader_get_output_count( type);

	float max_w=font->get_string_size( VisualServer::shader_node_get_type_info(type).name ).width;

	for(int i=0;i<VisualServer::shader_get_input_count(type);i++)
		max_w = MAX( max_w, font->get_string_size( VisualServer::shader_get_input_name(type,i) ).width );


	for(int i=0;i<VisualServer::shader_get_output_count(type);i++)
		max_w = MAX( max_w, font->get_string_size( VisualServer::shader_get_output_name(type,i) ).width );




	switch(type) {

		case VS::NODE_IN:
		case VS::NODE_OUT:
		case VS::NODE_VEC_IN:
		case VS::NODE_VEC_OUT:
		case VS::NODE_PARAMETER:
		case VS::NODE_VEC_PARAMETER:
		case VS::NODE_COLOR_PARAMETER:
		case VS::NODE_TEXTURE_PARAMETER:
		case VS::NODE_TEXTURE_2D_PARAMETER:
		case VS::NODE_TEXTURE_CUBE_PARAMETER:
		case VS::NODE_TRANSFORM_PARAMETER:
		case VS::NODE_LABEL: {

			max_w=MAX( max_w, font->get_string_size( shader_graph.node_get_param(p_node) ).width );
			count++;
		} break;
		case VS::NODE_TIME:
		case VS::NODE_CONSTANT:
		case VS::NODE_VEC_CONSTANT:
		case VS::NODE_COLOR_CONSTANT:
		case VS::NODE_TRANSFORM_CONSTANT: {
			count++;
		} break;
		case VS::NODE_TEXTURE:
		case VS::NODE_VEC_TEXTURE_2D:
		case VS::NODE_VEC_TEXTURE_CUBE: {

			RefPtr res = shader_graph.node_get_param(p_node);
			Ref<Texture> texture = res;
			if (texture.is_null() || texture->get_width()==0) {

				size.y+=max_w;
			} else {

				size.y+=max_w * texture->get_height() / texture->get_width();
			}
		} break;
		default: {}

	}

	size.x+=max_w;
	size.y+=count*(font->get_height()+get_constant("vseparation","PopupMenu"));

	return size;
}


Error ShaderEditor::validate_graph() {

	_ShaderTester st;
	active_nodes.clear();
	st._set=&active_nodes;
	return shader_graph.generate(&st);
}

void ShaderEditor::_draw_node(int p_node) {

	VisualServer::ShaderNodeType type=shader_graph.node_get_type(p_node);
	Ref<StyleBox> style = active_nodes.has(p_node)?get_stylebox("panel","PopupMenu"):get_stylebox("panel_disabled","PopupMenu");
	Ref<Font> font = get_font("font","PopupMenu");
	Color font_color = get_color("font_color","PopupMenu");
	Color font_color_title = get_color("font_color_hover","PopupMenu");
	Size2 size=get_node_size(p_node);
	Point2 pos = Point2( shader_graph.node_get_pos_x(p_node), shader_graph.node_get_pos_y(p_node) )-offset;

	if (click_type==CLICK_NODE && click_node==p_node) {

		pos+=click_motion-click_pos;
	}

	RID ci = get_canvas_item();
	style->draw(ci,Rect2(pos,size));

	Point2 ofs=style->get_offset()+pos;
	Point2 ascent=Point2(0,font->get_ascent());
	float w = size.width-style->get_minimum_size().width;
	float h = font->get_height()+get_constant("vseparation","PopupMenu");

	font->draw_halign( ci, ofs+ascent, HALIGN_CENTER,w, VisualServer::shader_node_get_type_info(type).name,font_color_title);
	ofs.y+=h;

	Ref<Texture> vec_icon = get_icon("NodeVecSlot","EditorIcons");
	Ref<Texture> real_icon = get_icon("NodeRealSlot","EditorIcons");
	float icon_h_ofs = Math::floor(( font->get_height()-vec_icon->get_height())/2.0 )+1;


	for(int i=0;i<VisualServer::shader_get_input_count(type);i++) {

		String name = VisualServer::shader_get_input_name(type,i);
		font->draw_halign( ci, ofs+ascent, HALIGN_LEFT,w, name,font_color);
		Ref<Texture> icon = VisualServer::shader_is_input_vector(type,i)?vec_icon:real_icon;
		icon->draw(ci,ofs+Point2(-real_icon->get_width(),icon_h_ofs));
		ofs.y+=h;
	}

	for(int i=0;i<VisualServer::shader_get_output_count(type);i++) {

		String name = VisualServer::shader_get_output_name(type,i);
		font->draw_halign( ci, ofs+ascent, HALIGN_RIGHT,w, name,font_color);
		Ref<Texture> icon = VisualServer::shader_is_output_vector(type,i)?vec_icon:real_icon;
		icon->draw(ci,ofs+Point2(w,icon_h_ofs));
		ofs.y+=h;
	}

	switch(type) {

		case VS::NODE_IN:
		case VS::NODE_OUT:
		case VS::NODE_PARAMETER:
		case VS::NODE_VEC_IN:
		case VS::NODE_COLOR_PARAMETER:
		case VS::NODE_VEC_OUT:
		case VS::NODE_TEXTURE_PARAMETER:
		case VS::NODE_TEXTURE_2D_PARAMETER:
		case VS::NODE_TEXTURE_CUBE_PARAMETER:
		case VS::NODE_TRANSFORM_CONSTANT:
		case VS::NODE_TRANSFORM_PARAMETER:
		case VS::NODE_VEC_PARAMETER:
		case VS::NODE_LABEL: {
			String text = shader_graph.node_get_param(p_node);
			font->draw_halign( ci, ofs+ascent, HALIGN_CENTER,w, text,font_color);
		} break;
		case VS::NODE_TIME:
		case VS::NODE_CONSTANT: {
			String text = rtos(shader_graph.node_get_param(p_node));
			font->draw_halign( ci, ofs+ascent, HALIGN_CENTER,w, text,font_color);

		} break;
		case VS::NODE_VEC_CONSTANT: {
			String text = Vector3(shader_graph.node_get_param(p_node));
			font->draw_halign( ci, ofs+ascent, HALIGN_CENTER,w, text,font_color);
		} break;
		case VS::NODE_COLOR_CONSTANT: {

			Color color = shader_graph.node_get_param(p_node);
			Rect2 r(ofs,Size2(w,h));
			VisualServer::get_singleton()->canvas_item_add_rect(ci,r,color);
		} break;
		case VS::NODE_TEXTURE:
		case VS::NODE_VEC_TEXTURE_2D:
		case VS::NODE_VEC_TEXTURE_CUBE: {

			Rect2 r(ofs,Size2(w,(pos.y+size.y-style->get_margin(MARGIN_BOTTOM))-ofs.y));
			Vector<Point2> points;
			Vector<Point2> uvs;
			points.resize(4);
			uvs.resize(4);
			points[0]=r.pos;
			points[1]=r.pos+Point2(r.size.x,0);
			points[2]=r.pos+r.size;
			points[3]=r.pos+Point2(0,r.size.y);
			uvs[0]=Point2(0,0);
			uvs[1]=Point2(1,0);
			uvs[2]=Point2(1,1);
			uvs[3]=Point2(0,1);

			Ref<Texture> texture = shader_graph.node_get_param(p_node).operator RefPtr();
			if (texture.is_null() || texture->get_width()==0) {
				texture=get_icon("Click2Edit","EditorIcons");
			}

			VisualServer::get_singleton()->canvas_item_add_primitive(ci,points,Vector<Color>(),uvs,texture->get_rid());
		} break;
		default: {}
	}
}

void ShaderEditor::_node_param_changed() {

	shader_graph.node_set_param( click_node,property_editor->get_variant() );
	update();
	_write_shader_graph();
}

ShaderEditor::ClickType ShaderEditor::_locate_click(const Point2& p_click,int *p_node_id,int *p_slot_index) const {

	Ref<StyleBox> style = get_stylebox("panel","PopupMenu");
	Ref<Texture> real_icon = get_icon("NodeRealSlot","EditorIcons");
	Ref<Font> font = get_font("font","PopupMenu");
	float h = font->get_height()+get_constant("vseparation","PopupMenu");
	float extra_left=MAX( real_icon->get_width()-style->get_margin(MARGIN_LEFT), 0 );
	float extra_right=MAX( real_icon->get_width()-style->get_margin(MARGIN_RIGHT), 0 );


	for(const List<int>::Element *E=order.back();E;E=E->prev()) {

		Size2 size=get_node_size(E->get());
		size.width+=extra_left+extra_right;
		Point2 pos = Point2( shader_graph.node_get_pos_x(E->get()), shader_graph.node_get_pos_y(E->get()) )-offset;
		pos.x-=extra_left;

		Rect2 rect( pos, size );
		if (!rect.has_point(p_click))
			continue;
		VisualServer::ShaderNodeType type=shader_graph.node_get_type(E->get());
		if (p_node_id)
			*p_node_id=E->get();
		float y=p_click.y-(pos.y+style->get_margin(MARGIN_TOP));
		if (y<h)
			return CLICK_NODE;
		y-=h;

		for(int i=0;i<VisualServer::shader_get_input_count(type);i++) {

			if (y<h) {
				if (p_slot_index)
					*p_slot_index=i;
				return CLICK_INPUT_SLOT;
			}
			y-=h;
		}

		for(int i=0;i<VisualServer::shader_get_output_count(type);i++) {

			if (y<h) {
				if (p_slot_index)
					*p_slot_index=i;
				return CLICK_OUTPUT_SLOT;
			}
			y-=h;
		}

		if (p_click.y<(rect.pos.y+rect.size.height-style->get_margin(MARGIN_BOTTOM)))
			return CLICK_PARAMETER;
		else
			return CLICK_NODE;

	}

	return CLICK_NONE;

}

Point2 ShaderEditor::_get_slot_pos(int p_node_id,bool p_input,int p_slot) {

	Ref<StyleBox> style = get_stylebox("panel","PopupMenu");
	float w = get_node_size(p_node_id).width;
	Ref<Font> font = get_font("font","PopupMenu");
	float h = font->get_height()+get_constant("vseparation","PopupMenu");
	Ref<Texture> vec_icon = get_icon("NodeVecSlot","EditorIcons");
	Point2 pos = Point2( shader_graph.node_get_pos_x(p_node_id), shader_graph.node_get_pos_y(p_node_id) )-offset;
	pos+=style->get_offset();
	pos.y+=h;

	if(p_input) {

		pos.y+=p_slot*h;
		pos+=Point2( -vec_icon->get_width()/2.0, h/2.0).floor();
		return pos;
	} else {

		pos.y+=VisualServer::shader_get_input_count( shader_graph.node_get_type(p_node_id ) )*h;
	}

	pos.y+=p_slot*h;
	pos+=Point2( w-style->get_minimum_size().width+vec_icon->get_width()/2.0, h/2.0).floor();

	return pos;

}

void ShaderEditor::_node_edit_property(int p_node) {

	Ref<StyleBox> style = get_stylebox("panel","PopupMenu");
	Size2 size = get_node_size(p_node);
	Point2 pos = Point2( shader_graph.node_get_pos_x(p_node), shader_graph.node_get_pos_y(p_node) )-offset;

	VisualServer::ShaderNodeType type=shader_graph.node_get_type(p_node);

	PropertyInfo ph = VisualServer::get_singleton()->shader_node_get_type_info(type);
	if (ph.type==Variant::NIL)
		return;
	if (ph.type==Variant::_RID)
		ph.type=Variant::OBJECT;

	property_editor->edit(NULL,ph.name,ph.type,shader_graph.node_get_param(p_node),ph.hint,ph.hint_string);

	Point2 popup_pos=Point2( pos.x+(size.width-property_editor->get_size().width)/2.0,pos.y+(size.y-style->get_margin(MARGIN_BOTTOM))).floor();
	popup_pos+=get_global_pos();
	property_editor->set_pos(popup_pos);

	property_editor->popup();

}

bool ShaderEditor::has_point(const Point2& p_point) const {

	int n,si;

	return _locate_click(p_point,&n,&si)!=CLICK_NONE;
}

void ShaderEditor::_input_event(InputEvent p_event) {

	switch(p_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			if (p_event.mouse_button.pressed) {


				if (p_event.mouse_button.button_index==1) {
					click_pos=Point2(p_event.mouse_button.x,p_event.mouse_button.y);
					click_motion=click_pos;
					click_type = _locate_click(click_pos,&click_node,&click_slot);
					if( click_type!=CLICK_NONE) {

						order.erase(click_node);
						order.push_back(click_node);
						update();
					}
					switch(click_type) {
						case CLICK_INPUT_SLOT: {
							click_pos=_get_slot_pos(click_node,true,click_slot);
						} break;
						case CLICK_OUTPUT_SLOT: {
							click_pos=_get_slot_pos(click_node,false,click_slot);
						} break;
						case CLICK_PARAMETER: {
							//open editor
							_node_edit_property(click_node);
						} break;
					}
				}
				if (p_event.mouse_button.button_index==2) {

					if (click_type!=CLICK_NONE) {
						click_type=CLICK_NONE;
						update();
					} else {
						// try to disconnect/remove

						Point2 rclick_pos=Point2(p_event.mouse_button.x,p_event.mouse_button.y);
						rclick_type = _locate_click(rclick_pos,&rclick_node,&rclick_slot);
						if (rclick_type==CLICK_INPUT_SLOT || rclick_type==CLICK_OUTPUT_SLOT) {

							node_popup->clear();
							node_popup->add_item("Disconnect",NODE_DISCONNECT);
							node_popup->set_pos(rclick_pos);
							node_popup->popup();

						}

						if (rclick_type==CLICK_NODE) {
							node_popup->clear();
							node_popup->add_item("Remove",NODE_ERASE);
							node_popup->set_pos(rclick_pos);
							node_popup->popup();
						}


					}
				}
			} else {

				if (p_event.mouse_button.button_index==1 && click_type!=CLICK_NONE) {

					switch(click_type) {
						case CLICK_INPUT_SLOT:
						case CLICK_OUTPUT_SLOT: {

							Point2 dst_click_pos=Point2(p_event.mouse_button.x,p_event.mouse_button.y);
							int id;
							int slot;
							ClickType dst_click_type = _locate_click(dst_click_pos,&id,&slot);
							if (dst_click_type==CLICK_INPUT_SLOT && click_type==CLICK_OUTPUT_SLOT) {

								shader_graph.connect(click_node,click_slot,id,slot);

								Error err = validate_graph();
								if (err==ERR_CYCLIC_LINK)
									shader_graph.disconnect(click_node,click_slot,id,slot);
								_write_shader_graph();

							}
							if (click_type==CLICK_INPUT_SLOT && dst_click_type==CLICK_OUTPUT_SLOT) {

								shader_graph.connect(id,slot,click_node,click_slot);

								Error err = validate_graph();
								if (err==ERR_CYCLIC_LINK)
									shader_graph.disconnect(id,slot,click_node,click_slot);
								_write_shader_graph();
							}

						} break;
						case CLICK_NODE: {
							int new_x=shader_graph.node_get_pos_x(click_node)+(click_motion.x-click_pos.x);
							int new_y=shader_graph.node_get_pos_y(click_node)+(click_motion.y-click_pos.y);
							shader_graph.node_set_pos(click_node,new_x,new_y);
							_write_shader_graph();

						} break;
					}

					click_type=CLICK_NONE;
					update();
				}
			}

		}

		case InputEvent::MOUSE_MOTION: {

			if (p_event.mouse_motion.button_mask&1 && click_type!=CLICK_NONE) {

				click_motion=Point2(p_event.mouse_button.x,p_event.mouse_button.y);
				update();
			}

		} break;
	}
}

void ShaderEditor::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_DRAW: {

			_update_scrollbars();
			//VisualServer::get_singleton()->canvas_item_add_rect(get_canvas_item(),Rect2(Point2(),get_size()),Color(0,0,0,1));

			for(List<int>::Element *E=order.front();E;E=E->next()) {

				_draw_node(E->get());
			}

			if (click_type==CLICK_INPUT_SLOT || click_type==CLICK_OUTPUT_SLOT) {

				VisualServer::get_singleton()->canvas_item_add_line(get_canvas_item(),click_pos,click_motion,Color(0.5,1,0.5,0.8),2);
			}

			List<ShaderGraph::Connection> connections = shader_graph.get_connection_list();
			for(List<ShaderGraph::Connection>::Element *E=connections.front();E;E=E->next()) {

				const ShaderGraph::Connection &c=E->get();
				Point2 source = _get_slot_pos(c.src_id,false,c.src_slot);
				Point2 dest = _get_slot_pos(c.dst_id,true,c.dst_slot);
				bool vec = VisualServer::shader_is_input_vector( shader_graph.node_get_type(c.dst_id), c.dst_slot );
				Color col = vec?Color(1,0.5,0.5,0.8):Color(1,1,0.5,0.8);

				if (click_type==CLICK_NODE && click_node==c.src_id) {

					source+=click_motion-click_pos;
				}

				if (click_type==CLICK_NODE && click_node==c.dst_id) {

					dest+=click_motion-click_pos;
				}

				VisualServer::get_singleton()->canvas_item_add_line(get_canvas_item(),source,dest,col,2);

			}
		} break;
	}

}

void ShaderEditor::_update_scrollbars() {

	Size2 size = get_size();
	Size2 hmin = h_scroll->get_minimum_size();
	Size2 vmin = v_scroll->get_minimum_size();

	v_scroll->set_begin( Point2(size.width - vmin.width, 0) );
	v_scroll->set_end( Point2(size.width, size.height) );

	h_scroll->set_begin( Point2( 0, size.height - hmin.height) );
	h_scroll->set_end( Point2(size.width-vmin.width, size.height) );


	Size2 min = _get_maximum_size();

	if (min.height < size.height - hmin.height) {

		v_scroll->hide();
		offset.y=0;
	} else {

		v_scroll->show();
		v_scroll->set_max(min.height);
		v_scroll->set_page(size.height - hmin.height);
		offset.y=v_scroll->get_val();
	}

	if (min.width < size.width - vmin.width) {

		h_scroll->hide();
		offset.x=0;
	} else {

		h_scroll->show();
		h_scroll->set_max(min.width);
		h_scroll->set_page(size.width - vmin.width);
		offset.x=h_scroll->get_val();
	}
}

void ShaderEditor::_scroll_moved() {

	offset.x=h_scroll->get_val();
	offset.y=v_scroll->get_val();
	update();
}

void ShaderEditor::_bind_methods() {

	ObjectTypeDB::bind_method( "_node_menu_item", &ShaderEditor::_node_menu_item );
	ObjectTypeDB::bind_method( "_node_add_callback", &ShaderEditor::_node_add_callback );
	ObjectTypeDB::bind_method( "_input_event", &ShaderEditor::_input_event );
	ObjectTypeDB::bind_method( "_node_param_changed", &ShaderEditor::_node_param_changed );
	ObjectTypeDB::bind_method( "_scroll_moved", &ShaderEditor::_scroll_moved );
	ObjectTypeDB::bind_method( "_vertex_item", &ShaderEditor::_vertex_item );
	ObjectTypeDB::bind_method( "_fragment_item", &ShaderEditor::_fragment_item );
	ObjectTypeDB::bind_method( "_post_item", &ShaderEditor::_post_item );
}

void ShaderEditor::_read_shader_graph() {

	shader_graph.clear();;
	order.clear();
	List<int> nodes;
	shader->get_node_list(&nodes);
	int larger_id=0;
	for(List<int>::Element *E=nodes.front();E;E=E->next()) {

		if (E->get() > larger_id)
			larger_id = E->get();

		shader_graph.node_add( (VS::ShaderNodeType)shader->node_get_type(E->get()), E->get() );
		shader_graph.node_set_param( E->get(), shader->node_get_param( E->get() ) );
		Point2 pos = shader->node_get_pos(E->get());
		shader_graph.node_set_pos( E->get(), pos.x,pos.y );
		order.push_back(E->get());
	}

	last_id=larger_id+1;

	List<Shader::Connection> connections;
	shader->get_connections(&connections);

	for(List<Shader::Connection>::Element *E=connections.front();E;E=E->next()) {

		Shader::Connection &c=E->get();
		shader_graph.connect(c.src_id,c.src_slot,c.dst_id,c.dst_slot);
	}

	validate_graph();
	update();
}

void ShaderEditor::_write_shader_graph() {

	shader->clear();
	List<int> nodes;
	shader_graph.get_node_list(&nodes);
	for(List<int>::Element *E=nodes.front();E;E=E->next()) {

		shader->node_add((Shader::NodeType)shader_graph.node_get_type(E->get()),E->get());
		shader->node_set_param(E->get(),shader_graph.node_get_param(E->get()));
		shader->node_set_pos(E->get(),Point2( shader_graph.node_get_pos_x(E->get()),shader_graph.node_get_pos_y(E->get()) ) );
	}

	List<ShaderGraph::Connection> connections = shader_graph.get_connection_list();
	for(List<ShaderGraph::Connection>::Element *E=connections.front();E;E=E->next()) {

		const ShaderGraph::Connection &c=E->get();
		shader->connect(c.src_id,c.src_slot,c.dst_id,c.dst_slot);
	}
}

void ShaderEditor::_add_node_from_text(const String& p_text) {

	ERR_FAIL_COND( p_text.get_slice_count(" ") != 3 );
	bool input = p_text.get_slice(" ",0)=="In:";
	String name = p_text.get_slice(" ",1);
	bool vec = p_text.get_slice(" ",2)=="(vec3)";

	_node_add( input?
		( vec? VisualServer::NODE_VEC_IN : VisualServer::NODE_IN ) :
		( vec? VisualServer::NODE_VEC_OUT : VisualServer::NODE_OUT ) );

	shader_graph.node_set_param( last_id-1,name );
	_write_shader_graph();
}

void ShaderEditor::_vertex_item(int p_item) {

	_add_node_from_text(vertex_popup->get_item_text(p_item));
}
void ShaderEditor::_fragment_item(int p_item) {

	_add_node_from_text(fragment_popup->get_item_text(p_item));
}
void ShaderEditor::_post_item(int p_item) {

	_add_node_from_text(post_popup->get_item_text(p_item));
}


void ShaderEditor::_node_menu_item(int p_item) {

	switch(p_item) {

		case GRAPH_ADD_NODE: {
			add_popup->popup_centered_ratio();
			validate_graph();
		} break;
		case NODE_DISCONNECT: {

			if (rclick_type==CLICK_INPUT_SLOT) {

				List<ShaderGraph::Connection> connections = shader_graph.get_connection_list();
				for(List<ShaderGraph::Connection>::Element *E=connections.front();E;E=E->next()) {

					const ShaderGraph::Connection &c=E->get();
					if( c.dst_id==rclick_node && c.dst_slot==rclick_slot) {

						shader_graph.disconnect(c.src_id,c.src_slot,c.dst_id,c.dst_slot);
					}
				}
				update();
				_write_shader_graph();
				validate_graph();
			}

			if (rclick_type==CLICK_OUTPUT_SLOT) {

				List<ShaderGraph::Connection> connections = shader_graph.get_connection_list();
				for(List<ShaderGraph::Connection>::Element *E=connections.front();E;E=E->next()) {

					const ShaderGraph::Connection &c=E->get();
					if( c.src_id==rclick_node && c.src_slot==rclick_slot) {

						shader_graph.disconnect(c.src_id,c.src_slot,c.dst_id,c.dst_slot);
					}
				}
				update();
				_write_shader_graph();
				validate_graph();
			}

		} break;
		case NODE_ERASE: {

			order.erase(rclick_node);
			shader_graph.node_remove(rclick_node);
			update();
			_write_shader_graph();
			validate_graph();
		} break;
		case GRAPH_CLEAR: {

			order.clear();
			shader_graph.clear();
			last_id=1;
			last_x=20;
			last_y=20;
			update();
			_write_shader_graph();
			validate_graph();

		} break;
	}
}

void ShaderEditor::_node_add(VisualServer::ShaderNodeType p_type) {

	shader_graph.node_add(p_type,last_id );
	shader_graph.node_set_pos(last_id ,last_x,last_y);
	String test_param;

	switch(p_type) {
		case VS::NODE_PARAMETER: {

			test_param="param";
		} break;
		case VS::NODE_VEC_PARAMETER: {

			test_param="vec";
		} break;
		case VS::NODE_COLOR_PARAMETER: {

			test_param="color";
		} break;
		case VS::NODE_TEXTURE_PARAMETER: {

			test_param="tex";
		} break;
		case VS::NODE_TEXTURE_2D_PARAMETER: {

			test_param="tex2D";
		} break;
		case VS::NODE_TEXTURE_CUBE_PARAMETER: {

			test_param="cubemap";
		} break;
		case VS::NODE_TRANSFORM_PARAMETER: {
			test_param="xform";
		} break;
		case VS::NODE_LABEL: {

			test_param="label";
		} break;
	}

	if(test_param!="") {

		int iter=0;
		List<int> l;

		shader_graph.get_node_list(&l);

		bool found;
		String test;
		do {
			iter++;
			test=test_param;
			if (iter>1)
				test+="_"+itos(iter);
			found=false;
			for(List<int>::Element *E=l.front();E;E=E->next()) {


				String param = shader_graph.node_get_param( E->get() );
				if (param==test) {
					found=true;
					break;
				}
			}

		} while (found);


		shader_graph.node_set_param(last_id,test);

	}
	order.push_back(last_id);
	last_x+=10;
	last_y+=10;
	last_id++;
	last_x=last_x % (int)get_size().width;
	last_y=last_y % (int)get_size().height;
	update();
	add_popup->hide();;
	_write_shader_graph();

}

void ShaderEditor::_node_add_callback() {

	TreeItem * item = add_types->get_selected();
	ERR_FAIL_COND(!item);
	_node_add((VisualServer::ShaderNodeType)(int)item->get_metadata(0));
	add_popup->hide() ;
}

ShaderEditor::ShaderEditor() {

	set_focus_mode(FOCUS_ALL);

	Panel* menu_panel = memnew( Panel );
	menu_panel->set_anchor( MARGIN_RIGHT, Control::ANCHOR_END );
	menu_panel->set_end( Point2(0,22) );

	add_child( menu_panel );

	PopupMenu *p;
	List<PropertyInfo> defaults;

	MenuButton* node_menu = memnew( MenuButton );
	node_menu->set_text("Graph");
	node_menu->set_pos( Point2( 5,0) );
	menu_panel->add_child( node_menu );

	p=node_menu->get_popup();
	p->add_item("Add Node",GRAPH_ADD_NODE);
	p->add_separator();
	p->add_item("Clear",GRAPH_CLEAR);
	p->connect("item_pressed", this,"_node_menu_item");

	MenuButton* vertex_menu = memnew( MenuButton );
	vertex_menu->set_text("Vertex");
	vertex_menu->set_pos( Point2( 49,0) );
	menu_panel->add_child( vertex_menu );

	p=vertex_menu->get_popup();
	defaults.clear();
	VisualServer::shader_get_default_input_nodes(VisualServer::SHADER_VERTEX,&defaults);

	int id=0;
	for(int i=0;i<defaults.size();i++) {

		p->add_item("In: "+defaults[i].name+(defaults[i].type==Variant::VECTOR3?" (vec3)":" (real)"),id++);
	}
	p->add_separator();
	id++;

	defaults.clear();
	VisualServer::shader_get_default_output_nodes(VisualServer::SHADER_VERTEX,&defaults);

	for(int i=0;i<defaults.size();i++) {

		p->add_item("Out: "+defaults[i].name+(defaults[i].type==Variant::VECTOR3?" (vec3)":" (real)"),id++);
	}

	vertex_popup=p;
	vertex_popup->connect("item_pressed", this,"_vertex_item");
	MenuButton* fragment_menu = memnew( MenuButton );
	fragment_menu->set_text("Fragment");
	fragment_menu->set_pos( Point2( 95 ,0) );
	menu_panel->add_child( fragment_menu );

	p=fragment_menu->get_popup();
	defaults.clear();
	VisualServer::shader_get_default_input_nodes(VisualServer::SHADER_FRAGMENT,&defaults);
	id=0;
	for(int i=0;i<defaults.size();i++) {

		p->add_item("In: "+defaults[i].name+(defaults[i].type==Variant::VECTOR3?" (vec3)":" (real)"),id++);
	}
	p->add_separator();
	id++;
	defaults.clear();
	VisualServer::shader_get_default_output_nodes(VisualServer::SHADER_FRAGMENT,&defaults);

	for(int i=0;i<defaults.size();i++) {

		p->add_item("Out: "+defaults[i].name+(defaults[i].type==Variant::VECTOR3?" (vec3)":" (real)"),id++);
	}

	fragment_popup=p;
	fragment_popup->connect("item_pressed", this,"_fragment_item");

	MenuButton* post_menu = memnew( MenuButton );
	post_menu->set_text("Post");
	post_menu->set_pos( Point2( 161,0) );
	menu_panel->add_child( post_menu );

	p=post_menu->get_popup();
	defaults.clear();
	VisualServer::shader_get_default_input_nodes(VisualServer::SHADER_POST_PROCESS,&defaults);
	id=0;
	for(int i=0;i<defaults.size();i++) {

		p->add_item("In: "+defaults[i].name+(defaults[i].type==Variant::VECTOR3?" (vec3)":" (real)"),id++);
	}
	p->add_separator();
	id++;

	defaults.clear();
	VisualServer::shader_get_default_output_nodes(VisualServer::SHADER_POST_PROCESS,&defaults);

	for(int i=0;i<defaults.size();i++) {

		p->add_item("Out: "+defaults[i].name+(defaults[i].type==Variant::VECTOR3?" (vec3)":" (real)"),id++);
	}

	post_popup=p;
	post_popup->connect("item_pressed", this,"_post_item");


	/* add popup */

	add_popup = memnew( Popup );
	add_child(add_popup);
	add_popup->set_as_toplevel(true);
	Panel *add_panel = memnew( Panel );
	add_popup->add_child(add_panel);
	add_panel->set_area_as_parent_rect();

	Label *add_label = memnew (Label );
	add_label->set_pos(Point2(5,5));
	add_label->set_text("Available Nodes:");
	add_panel->add_child(add_label);


	add_types = memnew( Tree );
	add_types->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	add_types->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	add_types->set_begin( Point2( 20,25 ) );
	add_types->set_end( Point2( 10, 30 ) );
	add_types->set_hide_root(true);
	add_types->set_columns(4);
	add_types->set_select_mode(Tree::SELECT_ROW);


	TreeItem *add_types_root = add_types->create_item(NULL);
	TreeItem *info_item = add_types->create_item(add_types_root);

	for(int i=0;i<VisualServer::NODE_TYPE_MAX;i++) {

		TreeItem *item = add_types->create_item(add_types_root);
		PropertyInfo prop = VisualServer::shader_node_get_type_info((VisualServer::ShaderNodeType)i);
		item->set_text(0,prop.name);
		item->set_text(1,itos(VisualServer::shader_get_input_count((VisualServer::ShaderNodeType)i)));
		item->set_text(2,itos(VisualServer::shader_get_output_count((VisualServer::ShaderNodeType)i)));
		String hint = (prop.type==Variant::_RID)?prop.hint_string:Variant::get_type_name(prop.type);
		item->set_text(3,hint);
		item->set_metadata(0,i);
	}
	info_item->set_text(0,"::NODE::");
	info_item->set_custom_color(0,Color(0.6,0.1,0.1));
	info_item->set_text(1,"::INPUTS::");
	info_item->set_custom_color(1,Color(0.6,0.1,0.1));
	info_item->set_text(2,"::OUTPUTS::");
	info_item->set_custom_color(2,Color(0.6,0.1,0.1));
	info_item->set_text(3,"::PARAM::");
	info_item->set_custom_color(3,Color(0.6,0.1,0.1));
	info_item->set_selectable(0,false);
	info_item->set_selectable(1,false);
	info_item->set_selectable(2,false);
	info_item->set_selectable(3,false);

	add_panel->add_child(add_types);

	add_confirm = memnew( Button );
	add_confirm->set_anchor( MARGIN_LEFT, ANCHOR_END );
	add_confirm->set_anchor( MARGIN_TOP, ANCHOR_END );
	add_confirm->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	add_confirm->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	add_confirm->set_begin( Point2( 75, 29 ) );
	add_confirm->set_end( Point2( 10, 15 ) );
	add_confirm->set_text("Add");
	add_panel->add_child(add_confirm);
	add_confirm->connect("pressed", this,"_node_add_callback");

	last_id=1;
	last_x=20;
	last_y=20;

	property_editor = memnew( CustomPropertyEditor );
	add_child(property_editor);
	property_editor->connect("variant_changed", this,"_node_param_changed");

	h_scroll = memnew( HScrollBar );
	v_scroll = memnew( VScrollBar );

	add_child(h_scroll);
	add_child(v_scroll);

	h_scroll->connect("value_changed", this,"_scroll_moved");
	v_scroll->connect("value_changed", this,"_scroll_moved");

	node_popup= memnew(PopupMenu );
	add_child(node_popup);
	node_popup->set_as_toplevel(true);

	node_popup->connect("item_pressed", this,"_node_menu_item");

}


void ShaderEditorPlugin::edit(Object *p_object) {

	shader_editor->edit(p_object->cast_to<Shader>());
}

bool ShaderEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("Shader");
}

void ShaderEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		shader_editor->show();
		shader_editor->set_process(true);
	} else {

		shader_editor->hide();
		shader_editor->set_process(false);
	}

}

ShaderEditorPlugin::ShaderEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	shader_editor = memnew( ShaderEditor );
	editor->get_viewport()->add_child(shader_editor);
	shader_editor->set_area_as_parent_rect();
	shader_editor->hide();



}


ShaderEditorPlugin::~ShaderEditorPlugin()
{
}


#endif
