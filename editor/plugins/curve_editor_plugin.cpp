#include "curve_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "os/keyboard.h"
#include "spatial_editor_plugin.h"
void CurveTextureEdit::_gui_input(const InputEvent &p_event) {

	if (p_event.type == InputEvent::KEY && p_event.key.pressed && p_event.key.scancode == KEY_DELETE && grabbed != -1) {

		points.remove(grabbed);
		grabbed = -1;
		update();
		emit_signal("curve_changed");
		accept_event();
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == 1 && p_event.mouse_button.pressed) {

		update();
		Ref<Font> font = get_font("font", "Label");

		int font_h = font->get_height();

		Vector2 size = get_size();
		size.y -= font_h;

		Point2 p = Vector2(p_event.mouse_button.x, p_event.mouse_button.y) / size;
		p.y = CLAMP(1.0 - p.y, 0, 1) * (max - min) + min;
		grabbed = -1;
		grabbing = true;

		for (int i = 0; i < points.size(); i++) {

			Vector2 ps = p * get_size();
			Vector2 pt = Vector2(points[i].offset, points[i].height) * get_size();
			if (ps.distance_to(pt) < 4) {
				grabbed = i;
			}
		}

		//grab or select
		if (grabbed != -1) {
			return;
		}
		//insert

		Point np;
		np.offset = p.x;
		np.height = p.y;

		points.push_back(np);
		points.sort();
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == p.x && points[i].height == p.y) {
				grabbed = i;
				break;
			}
		}

		emit_signal("curve_changed");
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == 1 && !p_event.mouse_button.pressed) {

		if (grabbing) {
			grabbing = false;
			emit_signal("curve_changed");
		}
		update();
	}

	if (p_event.type == InputEvent::MOUSE_MOTION && grabbing && grabbed != -1) {

		Ref<Font> font = get_font("font", "Label");
		int font_h = font->get_height();
		Vector2 size = get_size();
		size.y -= font_h;

		Point2 p = Vector2(p_event.mouse_motion.x, p_event.mouse_motion.y) / size;
		p.y = CLAMP(1.0 - p.y, 0, 1) * (max - min) + min;
		p.x = CLAMP(p.x, 0.0, 1.0);

		bool valid = true;

		for (int i = 0; i < points.size(); i++) {

			if (points[i].offset == p.x && points[i].height == p.y && i != grabbed) {
				valid = false;
			}
		}

		if (!valid)
			return;

		points[grabbed].offset = p.x;
		points[grabbed].height = p.y;

		points.sort();
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == p.x && points[i].height == p.y) {
				grabbed = i;
				break;
			}
		}

		emit_signal("curve_changed");

		update();
	}
}

void CurveTextureEdit::_plot_curve(const Vector2 &p_a, const Vector2 &p_b, const Vector2 &p_c, const Vector2 &p_d) {

	Ref<Font> font = get_font("font", "Label");

	int font_h = font->get_height();

	float geometry[4][4];
	float tmp1[4][4];
	float tmp2[4][4];
	float deltas[4][4];
	double x, dx, dx2, dx3;
	double y, dy, dy2, dy3;
	double d, d2, d3;
	int lastx, lasty;
	int newx, newy;
	int ntimes;
	int i, j;

	int xmax = get_size().x;
	int ymax = get_size().y - font_h;

	int vsplits = 4;

	int zero_ofs = (1.0 - (0.0 - min) / (max - min)) * ymax;

	draw_line(Vector2(0, zero_ofs), Vector2(xmax, zero_ofs), Color(0.8, 0.8, 0.8, 0.15), 2.0);

	for (int i = 0; i <= vsplits; i++) {
		float fofs = float(i) / vsplits;
		int yofs = fofs * ymax;
		draw_line(Vector2(xmax, yofs), Vector2(xmax - 4, yofs), Color(0.8, 0.8, 0.8, 0.8), 2.0);

		String text = rtos((1.0 - fofs) * (max - min) + min);
		int ppos = text.find(".");
		if (ppos != -1) {
			if (text.length() > ppos + 2)
				text = text.substr(0, ppos + 2);
		}

		int size = font->get_string_size(text).x;
		int xofs = xmax - size - 4;
		yofs -= font_h / 2;

		if (yofs < 2) {
			yofs = 2;
		} else if (yofs + font_h > ymax - 2) {
			yofs = ymax - font_h - 2;
		}

		draw_string(font, Vector2(xofs, yofs + font->get_ascent()), text, Color(0.8, 0.8, 0.8, 1));
	}

	/* construct the geometry matrix from the segment */
	for (i = 0; i < 4; i++) {
		geometry[i][2] = 0;
		geometry[i][3] = 0;
	}

	geometry[0][0] = (p_a[0] * xmax);
	geometry[1][0] = (p_b[0] * xmax);
	geometry[2][0] = (p_c[0] * xmax);
	geometry[3][0] = (p_d[0] * xmax);

	geometry[0][1] = ((p_a[1] - min) / (max - min) * ymax);
	geometry[1][1] = ((p_b[1] - min) / (max - min) * ymax);
	geometry[2][1] = ((p_c[1] - min) / (max - min) * ymax);
	geometry[3][1] = ((p_d[1] - min) / (max - min) * ymax);

	/* subdivide the curve ntimes (1000) times */
	ntimes = 4 * xmax;
	/* ntimes can be adjusted to give a finer or coarser curve */
	d = 1.0 / ntimes;
	d2 = d * d;
	d3 = d * d * d;

	/* construct a temporary matrix for determining the forward differencing deltas */
	tmp2[0][0] = 0;
	tmp2[0][1] = 0;
	tmp2[0][2] = 0;
	tmp2[0][3] = 1;
	tmp2[1][0] = d3;
	tmp2[1][1] = d2;
	tmp2[1][2] = d;
	tmp2[1][3] = 0;
	tmp2[2][0] = 6 * d3;
	tmp2[2][1] = 2 * d2;
	tmp2[2][2] = 0;
	tmp2[2][3] = 0;
	tmp2[3][0] = 6 * d3;
	tmp2[3][1] = 0;
	tmp2[3][2] = 0;
	tmp2[3][3] = 0;

	/* compose the basis and geometry matrices */

	static const float CR_basis[4][4] = {
		{ -0.5, 1.5, -1.5, 0.5 },
		{ 1.0, -2.5, 2.0, -0.5 },
		{ -0.5, 0.0, 0.5, 0.0 },
		{ 0.0, 1.0, 0.0, 0.0 },
	};

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			tmp1[i][j] = (CR_basis[i][0] * geometry[0][j] +
						  CR_basis[i][1] * geometry[1][j] +
						  CR_basis[i][2] * geometry[2][j] +
						  CR_basis[i][3] * geometry[3][j]);
		}
	}
	/* compose the above results to get the deltas matrix */

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			deltas[i][j] = (tmp2[i][0] * tmp1[0][j] +
							tmp2[i][1] * tmp1[1][j] +
							tmp2[i][2] * tmp1[2][j] +
							tmp2[i][3] * tmp1[3][j]);
		}
	}

	/* extract the x deltas */
	x = deltas[0][0];
	dx = deltas[1][0];
	dx2 = deltas[2][0];
	dx3 = deltas[3][0];

	/* extract the y deltas */
	y = deltas[0][1];
	dy = deltas[1][1];
	dy2 = deltas[2][1];
	dy3 = deltas[3][1];

	lastx = CLAMP(x, 0, xmax);
	lasty = CLAMP(y, 0, ymax);

	/*	if (fix255)
		{
				cd->curve[cd->outline][lastx] = lasty;
		}
		else
		{
				cd->curve_ptr[cd->outline][lastx] = lasty;
				if(gb_debug) printf("bender_plot_curve xmax:%d ymax:%d\n", (int)xmax, (int)ymax);
		}
*/
	/* loop over the curve */
	for (i = 0; i < ntimes; i++) {
		/* increment the x values */
		x += dx;
		dx += dx2;
		dx2 += dx3;

		/* increment the y values */
		y += dy;
		dy += dy2;
		dy2 += dy3;

		newx = CLAMP((Math::round(x)), 0, xmax);
		newy = CLAMP((Math::round(y)), 0, ymax);

		/* if this point is different than the last one...then draw it */
		if ((lastx != newx) || (lasty != newy)) {
#if 0
			if(fix255)
			{
				/* use fixed array size (for the curve graph) */
				cd->curve[cd->outline][newx] = newy;
			}
			else
			{
				/* use dynamic allocated curve_ptr (for the real curve) */
				cd->curve_ptr[cd->outline][newx] = newy;

				if(gb_debug) printf("outline: %d  cX: %d cY: %d\n", (int)cd->outline, (int)newx, (int)newy);
			}
#endif
			draw_line(Vector2(lastx, ymax - lasty), Vector2(newx, ymax - newy), Color(0.8, 0.8, 0.8, 0.8), 2.0);
		}

		lastx = newx;
		lasty = newy;
	}

	int splits = 8;

	draw_line(Vector2(0, ymax - 1), Vector2(xmax, ymax - 1), Color(0.8, 0.8, 0.8, 0.3), 2.0);

	for (int i = 0; i <= splits; i++) {
		float fofs = float(i) / splits;
		draw_line(Vector2(fofs * xmax, ymax), Vector2(fofs * xmax, ymax - 2), Color(0.8, 0.8, 0.8, 0.8), 2.0);

		String text = rtos(fofs);
		int size = font->get_string_size(text).x;
		int ofs = fofs * xmax - size * 0.5;
		if (ofs < 2) {
			ofs = 2;
		} else if (ofs + size > xmax - 2) {
			ofs = xmax - size - 2;
		}

		draw_string(font, Vector2(ofs, ymax + font->get_ascent()), text, Color(0.8, 0.8, 0.8, 1));
	}
}

void CurveTextureEdit::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		Ref<Font> font = get_font("font", "Label");

		int font_h = font->get_height();

		draw_style_box(get_stylebox("bg", "Tree"), Rect2(Point2(), get_size()));

		int w = get_size().x;
		int h = get_size().y;

		Vector2 prev = Vector2(0, 0);
		Vector2 prev2 = Vector2(0, 0);

		for (int i = -1; i < points.size(); i++) {

			Vector2 next;
			Vector2 next2;
			if (i + 1 >= points.size()) {
				next = Vector2(1, 0);
			} else {
				next = Vector2(points[i + 1].offset, points[i + 1].height);
			}

			if (i + 2 >= points.size()) {
				next2 = Vector2(1, 0);
			} else {
				next2 = Vector2(points[i + 2].offset, points[i + 2].height);
			}

			/*if (i==-1 && prev.offset==next.offset) {
								prev=next;
								continue;
						}*/

			_plot_curve(prev2, prev, next, next2);

			prev2 = prev;
			prev = next;
		}

		Vector2 size = get_size();
		size.y -= font_h;
		for (int i = 0; i < points.size(); i++) {

			Color col = i == grabbed ? Color(1, 0.0, 0.0, 0.9) : Color(1, 1, 1, 0.8);

			float h = (points[i].height - min) / (max - min);
			draw_rect(Rect2(Vector2(points[i].offset, 1.0 - h) * size - Vector2(2, 2), Vector2(5, 5)), col);
		}

		/*		if (grabbed!=-1) {

						draw_rect(Rect2(total_w+3,0,h,h),points[grabbed].color);
				}
*/
		if (has_focus()) {

			draw_line(Vector2(-1, -1), Vector2(w + 1, -1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(w + 1, -1), Vector2(w + 1, h + 1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(w + 1, h + 1), Vector2(-1, h + 1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(-1, -1), Vector2(-1, h + 1), Color(1, 1, 1, 0.6));
		}
	}
}

Size2 CurveTextureEdit::get_minimum_size() const {

	return Vector2(64, 64);
}

void CurveTextureEdit::set_range(float p_min, float p_max) {
	max = p_max;
	min = p_min;
	update();
}

void CurveTextureEdit::set_points(const Vector<Vector2> &p_points) {

	points.clear();
	for (int i = 0; i < p_points.size(); i++) {
		Point p;
		p.offset = p_points[i].x;
		p.height = p_points[i].y;
		points.push_back(p);
	}

	points.sort();
	update();
}

Vector<Vector2> CurveTextureEdit::get_points() const {
	Vector<Vector2> ret;
	for (int i = 0; i < points.size(); i++)
		ret.push_back(Vector2(points[i].offset, points[i].height));
	return ret;
}

void CurveTextureEdit::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &CurveTextureEdit::_gui_input);

	ADD_SIGNAL(MethodInfo("curve_changed"));
}

CurveTextureEdit::CurveTextureEdit() {

	grabbed = -1;
	grabbing = false;
	max = 1;
	min = 0;
	set_focus_mode(FOCUS_ALL);
}

void CurveTextureEditorPlugin::_curve_settings_changed() {

	if (!curve_texture_ref.is_valid())
		return;
	curve_editor->set_points(Variant(curve_texture_ref->get_points()));
	curve_editor->set_range(curve_texture_ref->get_min(), curve_texture_ref->get_max());
}

CurveTextureEditorPlugin::CurveTextureEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	curve_editor = memnew(CurveTextureEdit);

	curve_button = editor->add_bottom_panel_item("CurveTexture", curve_editor);

	curve_button->hide();
	curve_editor->set_custom_minimum_size(Size2(100, 128 * EDSCALE));
	curve_editor->hide();
	curve_editor->connect("curve_changed", this, "curve_changed");
}

void CurveTextureEditorPlugin::edit(Object *p_object) {

	if (curve_texture_ref.is_valid()) {
		curve_texture_ref->disconnect("changed", this, "_curve_settings_changed");
	}
	CurveTexture *curve_texture = p_object->cast_to<CurveTexture>();
	if (!curve_texture)
		return;
	curve_texture_ref = Ref<CurveTexture>(curve_texture);
	curve_editor->set_points(Variant(curve_texture_ref->get_points()));
	curve_editor->set_range(curve_texture_ref->get_min(), curve_texture_ref->get_max());
	if (!curve_texture_ref->is_connected("changed", this, "_curve_settings_changed")) {
		curve_texture_ref->connect("changed", this, "_curve_settings_changed");
	}
}

bool CurveTextureEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("CurveTexture");
}

void CurveTextureEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		curve_button->show();
		editor->make_bottom_panel_item_visible(curve_editor);

	} else {

		curve_button->hide();
		if (curve_editor->is_visible_in_tree())
			editor->hide_bottom_panel();
	}
}

void CurveTextureEditorPlugin::_curve_changed() {

	if (curve_texture_ref.is_valid()) {

		UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

		Vector<Vector2> points = curve_editor->get_points();
		PoolVector<Vector2> ppoints = Variant(points);

		ur->create_action(TTR("Modify Curve"), UndoRedo::MERGE_ENDS);
		ur->add_do_method(this, "undo_redo_curve_texture", ppoints);
		ur->add_undo_method(this, "undo_redo_curve_texture", curve_texture_ref->get_points());
		ur->commit_action();
	}
}

void CurveTextureEditorPlugin::_undo_redo_curve_texture(const PoolVector<Vector2> &points) {

	curve_texture_ref->set_points(points);
	curve_editor->set_points(Variant(curve_texture_ref->get_points()));
	curve_editor->update();
}

CurveTextureEditorPlugin::~CurveTextureEditorPlugin() {
}

void CurveTextureEditorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("curve_changed"), &CurveTextureEditorPlugin::_curve_changed);
	ClassDB::bind_method(D_METHOD("_curve_settings_changed"), &CurveTextureEditorPlugin::_curve_settings_changed);
	ClassDB::bind_method(D_METHOD("undo_redo_curve_texture", "points"), &CurveTextureEditorPlugin::_undo_redo_curve_texture);
}
