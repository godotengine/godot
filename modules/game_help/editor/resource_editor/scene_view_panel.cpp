#include "scene_view_panel.h"


#include "scene/resources/packed_scene.h"
#include "core/config/project_settings.h"
#include "editor/themes/editor_scale.h"

void SceneViewPanel::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && (mm->get_button_mask().has_flag(MouseButtonMask::LEFT))) {
		rot_x -= mm->get_relative().y * 0.01;
		rot_y -= mm->get_relative().x * 0.01;

		rot_x = CLAMP(rot_x, -Math_PI / 2, Math_PI / 2);
		_update_rotation();
	}
}

void SceneViewPanel::_update_theme_item_cache() {
	SubViewportContainer::_update_theme_item_cache();

	theme_cache.light_1_icon = get_editor_theme_icon(SNAME("MaterialPreviewLight1"));
	theme_cache.light_2_icon = get_editor_theme_icon(SNAME("MaterialPreviewLight2"));
}

void SceneViewPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			light_1_switch->set_button_icon(theme_cache.light_1_icon);
			light_2_switch->set_button_icon(theme_cache.light_2_icon);
		} break;
	}
}

void SceneViewPanel::_update_rotation() {
	Transform3D t;
	t.basis.rotate(Vector3(0, 1, 0), -rot_y);
	t.basis.rotate(Vector3(1, 0, 0), -rot_x);
	rotation->set_transform(t);
}

void SceneViewPanel::update_aabb(AABB& aabb,Node3D* p_mesh) {

    MeshInstance3D* mi = Object::cast_to<MeshInstance3D>(p_mesh);
    if (mi) {
		aabb = aabb.merge(mi->get_aabb());
    }

    for(int i=0;i<p_mesh->get_child_count();i++)
    {
        Node3D* c = Object::cast_to<Node3D>(p_mesh->get_child(i)) ;
        update_aabb(aabb,c);
    }
    
}
void SceneViewPanel::edit(Node3D* p_mesh) {
	mesh_instance = p_mesh;

	rot_x = Math::deg_to_rad(-15.0);
	rot_y = Math::deg_to_rad(30.0);
	_update_rotation();

	AABB aabb;
    update_aabb(aabb,mesh_instance);
	Vector3 ofs = aabb.get_center();
	float m = aabb.get_longest_axis_size();
	if (m != 0) {
		m = 1.0 / m;
		m *= 0.5;
		Transform3D xform;
		xform.basis.scale(Vector3(m, m, m));
		xform.origin = -xform.basis.xform(ofs); //-ofs*m;
		//xform.origin.z -= aabb.get_longest_axis_size() * 2;
		mesh_instance->set_transform(xform);
	}
}

void SceneViewPanel::_on_light_1_switch_pressed() {
	light1->set_visible(light_1_switch->is_pressed());
}

void SceneViewPanel::_on_light_2_switch_pressed() {
	light2->set_visible(light_2_switch->is_pressed());
}

SceneViewPanel::SceneViewPanel() {
	viewport = memnew(SubViewport);
	Ref<World3D> world_3d;
	world_3d.instantiate();
	viewport->set_world_3d(world_3d); // Use own world.
	add_child(viewport);
	viewport->set_disable_input(true);
	viewport->set_msaa_3d(Viewport::MSAA_4X);
	set_stretch(true);
	camera = memnew(Camera3D);
	camera->set_transform(Transform3D(Basis(), Vector3(0, 0, 1.1)));
	camera->set_perspective(45, 0.1, 10);
	viewport->add_child(camera);

	if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
		camera_attributes.instantiate();
		camera->set_attributes(camera_attributes);
	}

	light1 = memnew(DirectionalLight3D);
	light1->set_transform(Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
	viewport->add_child(light1);

	light2 = memnew(DirectionalLight3D);
	light2->set_transform(Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));
	light2->set_color(Color(0.7, 0.7, 0.7));
	viewport->add_child(light2);

	rotation = memnew(Node3D);
	viewport->add_child(rotation);
	mesh_instance = memnew(MeshInstance3D);
	rotation->add_child(mesh_instance);

	set_custom_minimum_size(Size2(1, 150) );

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	hb->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 2);

	hb->add_spacer();

	VBoxContainer *vb_light = memnew(VBoxContainer);
	hb->add_child(vb_light);

	light_1_switch = memnew(Button);
	light_1_switch->set_theme_type_variation("PreviewLightButton");
	light_1_switch->set_toggle_mode(true);
	light_1_switch->set_pressed(true);
	vb_light->add_child(light_1_switch);
	light_1_switch->connect(SceneStringName(pressed), callable_mp(this, &SceneViewPanel::_on_light_1_switch_pressed));

	light_2_switch = memnew(Button);
	light_2_switch->set_theme_type_variation("PreviewLightButton");
	light_2_switch->set_toggle_mode(true);
	light_2_switch->set_pressed(true);
	vb_light->add_child(light_2_switch);
	light_2_switch->connect(SceneStringName(pressed), callable_mp(this, &SceneViewPanel::_on_light_2_switch_pressed));

	rot_x = 0;
	rot_y = 0;
}

void SceneViewPanel::set_scene_path(const String& p_scene_path) {

	Ref<PackedScene> scene = ResourceLoader::load(p_scene_path);
	if (scene.is_null())
	{
		print_line(L"SceneViewPanel: 路径不存在 :" + p_scene_path);
        return;
	}
	Node* p_node = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
    Node3D* p_mesh = Object::cast_to<Node3D>(p_node);
    if (p_mesh) {
        edit(p_mesh);
    }
}
