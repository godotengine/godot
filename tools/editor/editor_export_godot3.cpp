#include "editor_export_godot3.h"
#include "editor_node.h"
#include "io/resource_format_xml.h"
#include "io/resource_format_binary.h"
#include "scene/resources/scene_format_text.h"

static const char* prop_renames[][2]={
{"script/script","script"},
{"pause/pause_mode","pause_mode"},
{"anchor/left","anchor_left"},
{"anchor/right","anchor_right"},
{"anchor/bottom","anchor_bottom"},
{"anchor/top","anchor_top"},
{"focus_neighbour/left","focus_neighbour_left"},
{"focus_neighbour/right","focus_neighbour_right"},
{"focus_neighbour/bottom","focus_neighbour_bottom"},
{"focus_neighbour/top","focus_neighbour_top"},
{"focus/ignore_mouse","focus_ignore_mouse"},
{"focus/stop_mouse","focus_stop_mouse"},
{"size_flags/horizontal","size_flags_horizontal"},
{"size_flags/vertical","size_flags_vertical"},
{"size_flags/stretch_ratio","size_flags_stretch_ratio"},
{"theme/theme","theme"},
{"visibility/visible","visible"},
{"visibility/behind_parent","show_behind_parent"},
{"visibility/on_top","show_on_top"},
{"visibility/light_mask","light_mask"},
{"material/material","material"},
{"material/use_parent","use_parent_material"},
{"resource/path","resource_path"},
{"resource/name","resource_name"},
{"collision/layers","collision_layers"},
{"collision/mask","collision_mask"},
{"limit/left","limit_left"},
{"limit/right","limit_right"},
{"limit/bottom","limit_bottom"},
{"limit/top","limit_top"},
{"limit/smoothed","limit_smoothed"},
{"draw_margin/h_enabled","draw_margin_h_enabled"},
{"draw_margin/v_enabled","draw_margin_v_enabled"},
{"smoothing/enable","smoothing_enabled"},
{"smoothing/speed","smoothing_speed"},
{"drag_margin/left","drag_margin_left"},
{"drag_margin/top","drag_margin_top"},
{"drag_margin/right","drag_margin_right"},
{"drag_margin/bottom","drag_margin_bottom"},
{"input/pickable","input_pickable"},
{"bias/bias","bias"},
{"collision/exclude_nodes","disable_collision"},
{"range/height","range_height"},
{"range/z_min","range_z_min"},
{"range/z_max","range_z_max"},
{"range/layer_max","range_layer_max"},
{"range/item_cull_mask","range_item_cull_max"},
{"shadow/enabled","shadow_enabled"},
{"shadow/color","shadow_color"},
{"shadow/buffer_size","shadow_buffer_size"},
{"shadow/gradient_length","shadow_gradient_length"},
{"shadow/filter","shadow_filter"},
{"shadow/item_cull_mask","shadow_item_cull_mask"},
{"transform/pos","position"},
{"transform/rot","rotation"},
{"transform/scale","scale"},
{"z/z","z"},
{"z/relative","z_as_relative"},
{"scroll/offset","scroll_offset"},
{"scroll/base_offset","scroll_base_offset"},
{"scroll/base_scale","scroll_base_scale"},
{"scroll/limit_begin","scroll_limit_begin"},
{"scroll/limit_end","scroll_limit_end"},
{"scroll/ignore_camera_zoom","scroll_ignore_camera_zoom"},
{"motion/scale","motion_scale"},
{"motion/offset","motion_offset"},
{"motion/mirroring","motion_mirroring"},
{"collision/layers","collision_layer"},
{"collision/mask","collision_mask"},
{"texture/texture","texture"},
{"texture/offset","texture_offset"},
{"texture/rotation","texture_rotation"},
{"texture/scale","texture_scale"},
{"invert/enable","invert_enable"},
{"invert/border","invert_border"},
{"config/polyphony","polyphony"},
{"config/samples","samples"},
{"config/pitch_random","random_pitch"},
{"params/volume_db","volume_db"},
{"params/pitch_scale","pitch_scale"},
{"params/attenuation/min_distance","attenuation_min_distance"},
{"params/attenuation/max_distance","attenuation_max_distance"},
{"params/attenuation/distance_exp","attenuation_distance_exp"},
{"cell/size","cell_size"},
{"cell/quadrant_size","cell_quadrant_size"},
{"cell/half_offset","cell_half_offset"},
{"cell/tile_origin","cell_tile_origin"},
{"cell/y_sort","cell_y_sort"},
{"collision/use_kinematic","collision_use_kinematic"},
{"collision/friction","collision_friction"},
{"collision/bounce","collision_bounce"},
{"collision/layers","collision_layers"},
{"collision/mask","collision_mask"},
{"occluder/light_mask","occluder_light_mask"},
{"enabler/pause_animations","pause_animations"},
{"enabler/freeze_bodies","freeze_bodies"},
{"enabler/pause_particles","pause_particles"},
{"enabler/pause_animated_sprites","pause_animated_sprites"},
{"enabler/process_parent","process_parent"},
{"enabler/fixed_process_parent","fixed_process_parent"},
{"sort/enabled","sort_enabled"},
{"collision/layers","collision_layers"},
{"collision/mask","collision_mask"},
{"input/ray_pickable","input_ray_pickable"},
{"input/capture_on_drag","input_capture_on_drag"},
{"light/color","light_color"},
{"light/energy","light_energy"},
{"light/negative","light_negative"},
{"light/specular","light_specular"},
{"light/cull_mask","light_cull_mask"},
{"shadow/enabled","shadow_enabled"},
{"shadow/color","shadow_color"},
{"shadow/bias","shadow_bias"},
{"shadow/max_distance","shadow_max_distance"},
{"editor/editor_only","editor_only"},
{"directional_shadow/mode","directional_shadow_mode"},
{"directional_shadow/split_1","directional_shadow_split_1"},
{"directional_shadow/split_2","directional_shadow_split_2"},
{"directional_shadow/split_3","directional_shadow_split_3"},
{"directional_shadow/blend_splits","directional_shadow_blend_splits"},
{"directional_shadow/normal_bias","directional_shadow_normal_bias"},
{"directional_shadow/bias_split_scale","directional_shadow_bias_split_scale"},
{"omni/range","omni_range"},
{"omni/attenuation","omni_attenuation"},
{"omni/shadow_mode","omni_shadow_mode"},
{"omni/shadow_detail","omni_shadow_detail"},
{"spot/range","spot_range"},
{"spot/attenuation","spot_attenuation"},
{"spot/angle","spot_angle"},
{"spot/spot_attenuation","spot_angle_attenuation"},
{"mesh/mesh","mesh"},
{"mesh/skeleton","skeleton"},
{"collision/layers","collision_layer"},
{"collision/mask","collision_mask"},
{"quad/axis","axis"},
{"quad/size","size"},
{"quad/offset","offset"},
{"quad/centered","centered"},
{"transform/local","transform"},
{"transform/transiation","translation"},
{"transform/rotation","rotation"},
{"transform/scale","scale"},
{"visibility/visible","visible"},
{"params/volume_db","volume_db"},
{"params/pitch_scale","pitch_scale"},
{"params/attenuation/min_distance","attenuation_min_distance"},
{"params/attenuation/max_distance","attenuation_max_distance"},
{"params/attenuation/distance_exp","attenuation_distance_exp"},
{"params/emission_cone/degrees","emission_cone_degrees"},
{"params/emission_cone/attenuation_db","emission_cone_attenuation_db"},
{"config/polyphony","polyphony"},
{"config/samples","samples"},
{"flags/transparent","transparent"},
{"flags/shaded","shaded"},
{"flags/alpha_cut","alpha_cut"},
{"type/traction","use_as_traction"},
{"type/steering","use_as_steering"},
{"wheel/radius","wheel_radius"},
{"wheel/rest_length","wheel_rest_length"},
{"wheel/friction_slip","wheel_friction_sleep"},
{"suspension/travel","suspension_travel"},
{"suspension/stiffness","suspension_stiffness"},
{"suspension/max_force","suspension_max_force"},
{"damping/compression","damping_compression"},
{"damping/relaxation","damping_relaxation"},
{"motion/engine_force","engine_force"},
{"motion/breake","breake"},
{"motion/steering","steering"},
{"body/mass","mass"},
{"body/friction","friction"},
{"enabler/pause_animations","pause_animations"},
{"enabler/freeze_bodies","freeze_bodies"},
{"geometry/material_override","material_override"},
{"geometry/cast_shadow","cast_shadow"},
{"geometry/extra_cull_margin","extra_cull_margin"},
{"geometry/billboard","use_as_billboard"},
{"geometry/billboard_y","use_as_y_billboard"},
{"geometry/depth_scale","use_depth_scale"},
{"geometry/visible_in_all_rooms","visible_in_all_rooms"},
{"geometry/use_baked_light","use_in_baked_light"},
{"playback/process_mode","playback_process_mode"},
{"playback/default_blend_time","playback_default_blend_time"},
{"root/root","root_node"},
{"playback/process_mode","playback_process_mode"},
{"stream/stream","stream"},
{"stream/play","play"},
{"stream/loop","loop"},
{"stream/volume_db","volume_db"},
{"stream/pitch_scale","pitch_scale"},
{"stream/tempo_scale","tempo_scale"},
{"stream/autoplay","autoplay"},
{"stream/paused","paused"},
{"stream/stream","stream"},
{"stream/play","play"},
{"stream/loop","loop"},
{"stream/volume_db","volume_db"},
{"stream/autoplay","autoplay"},
{"stream/paused","paused"},
{"stream/loop_restart_time","loop_restart_time"},
{"stream/buffering_ms","buffering_ms"},
{"stream/stream","stream"},
{"stream/play","play"},
{"stream/loop","loop"},
{"stream/volume_db","volume_db"},
{"stream/autoplay","autoplay"},
{"stream/paused","paused"},
{"stream/loop_restart_time","loop_restart_time"},
{"stream/buffering_ms","buffering_ms"},
{"window/title","window_title"},
{"dialog/text","dialog_text"},
{"dialog/hide_on_ok","dialog_hide_on_ok"},
{"placeholder/text","placeholder_text"},
{"placeholder/alpha","placeholder_alpha"},
{"caret/caret_blink","caret_blink"},
{"caret/caret_blink_speed","caret_blink_speed"},
{"patch_margin/left","patch_margin_left"},
{"patch_margin/right","patch_margin_right"},
{"patch_margin/top","patch_margin_top"},
{"patch_margin/bottom","patch_margin_bottom"},
{"popup/exclusive","popup_exclusive"},
{"percent/visible","percent_visible"},
{"range/min","min_value"},
{"range/max","max_value"},
{"range/step","step"},
{"range/page","page"},
{"range/value","value"},
{"range/exp_edit","exp_edit"},
{"range/rounded","rounded"},
{"velocity/linear","linear_velocity"},
{"velocity/angular","angular_velocity"},
{"damp_override_linear","linear_damp"},
{"damp_override_angular","angular_damp"},
{"velocity/linear","linear_velocity"},
{"velocity/angular","angular_velocity"},
{"damp_override_linear","linear_damp"},
{"damp_override_angular","angular_damp"},
{"playback/process_mode","playback_process_mode"},
{"bbcode/enabled","bbcode_enabled"},
{"bbcode/bbcode","bbcode_text"},
{"scroll/horizontal","scroll_horizontal"},
{"scroll/vertical","scroll_vertical"},
{"split/offset","split_offset"},
{"split/collapsed","collapsed"},
{"split/dragger_visibility","dragger_visibility"},
{"caret/block_caret","caret_block_mode"},
{"caret/caret_blink","caret_blink"},
{"caret/caret_blink_speed","caret_blink_speed"},
{"textures/normal","texture_normal"},
{"textures/pressed","texture_pressed"},
{"textures/hover","texture_hover"},
{"textures/disabled","texture_disabled"},
{"textures/focused","texture_focused"},
{"textures/click_mask","texture_click_mask"},
{"params/scale","texture_scale"},
{"params/modulate","self_modulate"},
{"texture/under","texture_under"},
{"texture/over","texture_over"},
{"texture/progress","texture_progress"},
{"mode","fill_mode"},
{"radial_fill/initial_angle","radial_initial_angle"},
{"radial_fill/fill_degrees","radial_fill_degrees"},
{"radial_fill/center_offset","radial_center_offset"},
{"stream/audio_track","audio_track"},
{"stream/stream","stream"},
{"stream/volume_db","volume_db"},
{"stream/autoplay","stream_autoplay"},
{"stream/paused","stream_paused"},
{"font/size","size"},
{"extra_spacing/top","extra_spacing_top"},
{"extra_spacing/bottom","extra_spacing_bottom"},
{"extra_spacing/char","extra_spacing_char"},
{"extra_spacing/space","extra_spacing_space"},
{"font/use_mipmaps","use_mipmaps"},
{"font/use_filter","use_filter"},
{"font/font","font_data"},
{"content_margin/left","content_margin_left"},
{"content_margin/right","content_margin_right"},
{"content_margin/bottom","content_margin_bottom"},
{"content_margin/top","content_margin_top"},
{"margin/left","margin_left"},
{"margin/top","margin_top"},
{"margin/bottom","margin_bottom"},
{"margin/right","margin_right"},
{"expand_margin/left","expand_margin_left"},
{"expand_margin/top","expand_margin_top"},
{"expand_margin/bottom","expand_margin_bottom"},
{"expand_margin/right","expand_margin_right"},
{"modulate/color","modulate_color"},
{"modulate","self_modulate"},
{NULL,NULL}
};



static const char* type_renames[][2]={
{"SpatialPlayer","Spatial"},
{"SpatialSamplePlayer","Spatial"},
{"SpatialStreamPlayer","Spatial"},
{"SamplePlayer","Node"},
{"SamplePlayer2D","Node2D"},
{"SoundPlayer2D","Node2D"},
{"StreamPlayer2D","Node2D"},
{"SampleLibrary","Resource"},
{"TextureFrame","TextureRect"},
{NULL,NULL}
};


static const char* signal_renames[][2]={
{"area_enter","area_entered"},
{"area_exit","area_exited"},
{"area_enter_shape","area_shape_entered"},
{"area_exit_shape","area_shape_exited"},
{"body_enter","body_entered"},
{"body_exit","body_exited"},
{"body_enter_shape","body_shape_entered"},
{"body_exit_shape","body_shape_exited"},
{"mouse_enter","mouse_entered"},
{"mouse_exit","mouse_exited"},
{"focus_enter","focus_entered"},
{"focus_exit","focus_exited"},
{"modal_close","modal_closed"},
{"enter_tree","tree_entered"},
{"exit_tree","tree_exited"},
{NULL,NULL}
};




void EditorExportGodot3::_find_files(EditorFileSystemDirectory *p_dir,List<String> * r_files) {

	for(int i=0;i<p_dir->get_subdir_count();i++) {
		_find_files(p_dir->get_subdir(i),r_files);
	}

	for(int i=0;i<p_dir->get_file_count();i++) {

		r_files->push_back(p_dir->get_file_path(i));
	}
}


void EditorExportGodot3::_rename_properties(const String& p_type,List<ExportData::PropertyData> *p_props) {


	for (List<ExportData::PropertyData>::Element *E=p_props->front();E;E=E->next()) {

		if (prop_rename_map.has(E->get().name)) {
			E->get().name=prop_rename_map[E->get().name];
		}
	}
}

void EditorExportGodot3::_convert_resources(ExportData &resource) {


	for(int i=0;i<resource.resources.size();i++) {

		_rename_properties(resource.resources[i].type,&resource.resources[i].properties);

		if (type_rename_map.has(resource.resources[i].type)) {
			resource.resources[i].type=type_rename_map[resource.resources[i].type];
		}

	}

	for(int i=0;i<resource.nodes.size();i++) {

		_rename_properties(resource.nodes[i].type,&resource.nodes[i].properties);

		if (type_rename_map.has(resource.nodes[i].type)) {
			resource.nodes[i].type=type_rename_map[resource.nodes[i].type];
		}

	}

	for(int i=0;i<resource.connections.size();i++) {

		if (signal_rename_map.has(resource.connections[i].signal)) {
			resource.connections[i].signal=signal_rename_map[resource.connections[i].signal];
		}
	}

}

void EditorExportGodot3::_unpack_packed_scene(ExportData &resource) {
	
	
	Dictionary d;
	for (List<ExportData::PropertyData>::Element *E=resource.resources[resource.resources.size()-1].properties.front();E;E=E->next()) {
		if (E->get().name=="_bundled") {
			d=E->get().value;
		}
	}
	
	ERR_FAIL_COND(d.empty());
	
	ERR_FAIL_COND( !d.has("names"));
	ERR_FAIL_COND( !d.has("variants"));
	ERR_FAIL_COND( !d.has("node_count"));
	ERR_FAIL_COND( !d.has("nodes"));
	ERR_FAIL_COND( !d.has("conn_count"));
	ERR_FAIL_COND( !d.has("conns"));


	Vector<String> names;

	DVector<String> snames = d["names"];
	if (snames.size()) {

		int namecount = snames.size();
		names.resize(namecount);
		DVector<String>::Read r =snames.read();
		for(int i=0;i<names.size();i++)
			names[i]=r[i];
	}

	Array variants = d["variants"];

	resource.nodes.resize(d["node_count"]);
	
	int nc=resource.nodes.size();
	if (nc) {
		DVector<int> snodes = d["nodes"];
		DVector<int>::Read r = snodes.read();
		int idx=0;
		for(int i=0;i<nc;i++) {
			
			int parent = r[idx++];
			int owner = r[idx++];
			int type = r[idx++];
			int name = r[idx++];
			int instance = r[idx++];
			
			ExportData::NodeData node_data;
			
			node_data.text_data=false;
			node_data.name=names[name];
			node_data.type=names[type];
			node_data.parent_int=parent;
			node_data.owner_int=owner;
			if (instance>=0) {
				node_data.instance_is_placeholder=instance&SceneState::FLAG_INSTANCE_IS_PLACEHOLDER;
				node_data.instance=variants[instance&SceneState::FLAG_MASK];
			}
			
			int prop_count = r[idx++];
			
			for(int j=0;j<prop_count;j++) {

				int prop_name = r[idx++];
				int prop_value = r[idx++];

				ExportData::PropertyData pdata;
				pdata.name=names[prop_name];
				pdata.value=variants[prop_value];
				node_data.properties.push_back(pdata);
				
			}

			int group_count = r[idx++];
			for(int j=0;j<group_count;j++) {

				int group_name = r[idx++];
				node_data.groups.push_back(names[group_name]);
			}

			resource.nodes.push_back(node_data);
		}

	}

	int cc=d["conn_count"];

	if (cc) {

		DVector<int> sconns = d["conns"];
		DVector<int>::Read r = sconns.read();
		int idx=0;
		for(int i=0;i<cc;i++) {

			ExportData::Connection conn;

			conn.from_int=r[idx++];
			conn.to_int=r[idx++];
			conn.signal=names[r[idx++]];
			conn.method=names[r[idx++]];
			conn.flags=r[idx++];
			int bindcount = r[idx++];

			for(int j=0;j<bindcount;j++) {

				conn.binds.push_back(variants[r[idx++]]);
			}

			resource.connections.push_back(conn);
		}

	}

	Array np;
	if (d.has("node_paths")) {
		np=d["node_paths"];
	}

	for(int i=0;i<np.size();i++) {
		resource.node_paths.push_back(np[i]);
	}

	Array ei;
	if (d.has("editable_instances")) {		
		ei=d["editable_instances"];
		for(int i=0;i<ei.size();i++) {
			resource.editables.push_back(ei[i]);
		}
	}

	if (d.has("base_scene")) {
		resource.base_scene=variants[d["base_scene"]];
	}

	resource.resources.resize( resource.resources.size() -1 ); //erase packed
}



void EditorExportGodot3::_pack_packed_scene(ExportData &resource) {

	pack_names.clear();
	pack_values.clear();

	Dictionary d;

	d["node_count"]=resource.nodes.size();

	Vector<int> node_data;

	for(int i=0;i<resource.nodes.size();i++) {

		const ExportData::NodeData &node = resource.nodes[i];

		node_data.push_back(node.parent_int);
		node_data.push_back(node.owner_int);
		node_data.push_back(_pack_name(node.type));
		node_data.push_back(_pack_name(node.name));
		int instance=-1;
		if (node.instance!=String()) {
			instance=_pack_value(node.instance);
			if (node.instance_is_placeholder) {
				instance|=SceneState::FLAG_INSTANCE_IS_PLACEHOLDER;
			}
		}
		node_data.push_back(instance);


		node_data.push_back(node.properties.size());

		for(int j=0;j<node.properties.size();j++) {
			node_data.push_back(_pack_name(node.properties[j].name));
			node_data.push_back(_pack_value(node.properties[j].value));
		}

		node_data.push_back(node.groups.size());

		for(int j=0;j<node.groups.size();j++) {

			node_data.push_back(_pack_name(node.groups[j]));
		}

	}

	d["nodes"]=node_data;

	d["conn_count"]=resource.connections.size();

	Vector<int> connections;

	for(int i=0;i<resource.connections.size();i++) {
		const ExportData::Connection &conn=resource.connections[i];

		connections.push_back(conn.from_int);
		connections.push_back(conn.to_int);
		connections.push_back(_pack_name(conn.signal));
		connections.push_back(_pack_name(conn.method));
		connections.push_back(conn.flags);
		connections.push_back(conn.binds.size());
		for(int j=0;j<conn.binds.size();j++) {
			connections.push_back(_pack_value(conn.binds[j]));
		}
	}

	d["connections"]=connections;

	Array np;
	for(int i=0;i<resource.node_paths.size();i++) {
		np.push_back(resource.node_paths[i]);
	}

	d["node_paths"]=np;

	Array ei;
	for(int i=0;i<resource.editables.size();i++) {
		ei.push_back(resource.editables[i]);
	}

	d["editable_instances"]=ei;

	if (resource.base_scene.get_type()) {

		d["base_scene"]=_pack_value(resource.base_scene);
	}


	DVector<String> names;
	names.resize(pack_names.size());
	{
		DVector<String>::Write w=names.write();
		for(Map<String,int>::Element *E=pack_names.front();E;E=E->next()) {
			w[E->get()]=E->key();
		}
	}

	d["names"]=names;

	Array values;

	const Variant*K=NULL;
	while((K=pack_values.next(K))) {

		values[pack_values[*K]]=*K;
	}

	d["variants"]=values;

	ExportData::ResourceData packed_scene;
	packed_scene.type="PackedScene";
	packed_scene.index=-1;
	ExportData::PropertyData pd;
	pd.name="_bundled";
	pd.value=d;
	packed_scene.properties.push_back(pd);

	resource.resources.push_back(packed_scene);
	resource.nodes.clear();
	resource.connections.clear();
	resource.editables.clear();
	resource.node_paths.clear();;
	resource.base_scene=Variant();


}

static String rtosfix(double p_value) {


	if (p_value==0.0)
		return "0"; //avoid negative zero (-0) being written, which may annoy git, svn, etc. for changes when they don't exist.
	else
		return rtoss(p_value);
}

Error EditorExportGodot3::_get_property_as_text(const Variant& p_variant,String&p_string) {

	switch( p_variant.get_type() ) {

		case Variant::NIL: {
			p_string+=("null");
		} break;
		case Variant::BOOL: {

			p_string+=(p_variant.operator bool() ? "true":"false" );
		} break;
		case Variant::INT: {

			p_string+=( itos(p_variant.operator int()) );
		} break;
		case Variant::REAL: {

			String s = rtosfix(p_variant.operator real_t());
			if (s.find(".")==-1 && s.find("e")==-1)
				s+=".0";
			p_string+=( s );
		} break;
		case Variant::STRING: {

			String str=p_variant;
			if (str.begins_with("@RESLOCAL:")) {
				p_string+="SubResource("+str.get_slice(":",1)+")";
			} else if (str.begins_with("@RESEXTERNAL:")) {
				p_string+="ExtResource("+str.get_slice(":",1)+")";
			} else {

				str="\""+str.c_escape_multiline()+"\"";
				p_string+=( str );
			}

		} break;
		case Variant::VECTOR2: {

			Vector2 v = p_variant;
			p_string+=("Vector2( "+rtosfix(v.x) +", "+rtosfix(v.y)+" )" );
		} break;
		case Variant::RECT2: {

			Rect2 aabb = p_variant;
			p_string+=("Rect2( "+rtosfix(aabb.pos.x) +", "+rtosfix(aabb.pos.y) +", "+rtosfix(aabb.size.x) +", "+rtosfix(aabb.size.y)+" )" );

		} break;
		case Variant::VECTOR3: {

			Vector3 v = p_variant;
			p_string+=("Vector3( "+rtosfix(v.x) +", "+rtosfix(v.y)+", "+rtosfix(v.z)+" )");
		} break;
		case Variant::PLANE: {

			Plane p = p_variant;
			p_string+=("Plane( "+rtosfix(p.normal.x) +", "+rtosfix(p.normal.y)+", "+rtosfix(p.normal.z)+", "+rtosfix(p.d)+" )" );

		} break;
		case Variant::_AABB: {

			Rect3 aabb = p_variant;
			p_string+=("Rect3( "+rtosfix(aabb.pos.x) +", "+rtosfix(aabb.pos.y) +", "+rtosfix(aabb.pos.z) +", "+rtosfix(aabb.size.x) +", "+rtosfix(aabb.size.y) +", "+rtosfix(aabb.size.z)+" )"  );

		} break;
		case Variant::QUAT: {

			Quat quat = p_variant;
			p_string+=("Quat( "+rtosfix(quat.x)+", "+rtosfix(quat.y)+", "+rtosfix(quat.z)+", "+rtosfix(quat.w)+" )");

		} break;
		case Variant::MATRIX32: {

			String s="Transform2D( ";
			Matrix32 m3 = p_variant;
			for (int i=0;i<3;i++) {
				for (int j=0;j<2;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtosfix( m3.elements[i][j] );
				}
			}

			p_string+=(s+" )");

		} break;
		case Variant::MATRIX3: {

			String s="Basis( ";
			Matrix3 m3 = p_variant;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtosfix( m3.elements[i][j] );
				}
			}

			p_string+=(s+" )");

		} break;
		case Variant::TRANSFORM: {

			String s="Transform( ";
			Transform t = p_variant;
			Matrix3 &m3 = t.basis;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtosfix( m3.elements[i][j] );
				}
			}

			s=s+", "+rtosfix(t.origin.x) +", "+rtosfix(t.origin.y)+", "+rtosfix(t.origin.z);

			p_string+=(s+" )");
		} break;

			// misc types
		case Variant::COLOR: {

			Color c = p_variant;
			p_string+=("Color( "+rtosfix(c.r) +", "+rtosfix(c.g)+", "+rtosfix(c.b)+", "+rtosfix(c.a)+" )");

		} break;
		case Variant::IMAGE: {


			Image img=p_variant;

			if (img.empty()) {
				p_string+=("Image()");
				break;
			}

			String imgstr="Image()";
			p_string+=imgstr; //do not convert this for now

			/*imgstr+=itos(img.get_width());
			imgstr+=", "+itos(img.get_height());
			imgstr+=", "+String(img.get_mipmaps()?"true":"false");
			imgstr+=", "+Image::get_format_name(img.get_format());

			String s;

			PoolVector<uint8_t> data = img.get_data();
			int len = data.size();
			PoolVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();
			for (int i=0;i<len;i++) {

				if (i>0)
					s+=", ";
				s+=itos(ptr[i]);
			}

			imgstr+=", ";
			p_string+=(imgstr);
			p_string+=(s);
			p_string+=(" )");*/
		} break;
		case Variant::NODE_PATH: {

			String str=p_variant;

			str="NodePath(\""+str.c_escape()+"\")";
			p_string+=(str);

		} break;

		case Variant::OBJECT: {

			//should never arrive here!
			ERR_FAIL_V(ERR_BUG);
		} break;
		case Variant::INPUT_EVENT: {

			String str="InputEvent(";

			InputEvent ev=p_variant;
			switch(ev.type) {
				case InputEvent::KEY: {

					str+="KEY,"+itos(ev.key.scancode);
					String mod;
					if (ev.key.mod.alt)
						mod+="A";
					if (ev.key.mod.shift)
						mod+="S";
					if (ev.key.mod.control)
						mod+="C";
					if (ev.key.mod.meta)
						mod+="M";

					if (mod!=String())
						str+=","+mod;
				} break;
				case InputEvent::MOUSE_BUTTON: {

					str+="MBUTTON,"+itos(ev.mouse_button.button_index);
				} break;
				case InputEvent::JOYSTICK_BUTTON: {
					str+="JBUTTON,"+itos(ev.joy_button.button_index);

				} break;
				case InputEvent::JOYSTICK_MOTION: {
					str+="JAXIS,"+itos(ev.joy_motion.axis)+","+itos(ev.joy_motion.axis_value);
				} break;
				case InputEvent::NONE: {
					str+="NONE";
				} break;
				default: {}
			}

			str+=")";

			p_string+=(str); //will be added later

		} break;
		case Variant::DICTIONARY: {

			Dictionary dict = p_variant;

			List<Variant> keys;
			dict.get_key_list(&keys);
			keys.sort();

			p_string+=("{\n");
			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				/*
				if (!_check_type(dict[E->get()]))
					continue;
				*/
				_get_property_as_text(E->get(),p_string);
				p_string+=(": ");
				_get_property_as_text(dict[E->get()],p_string);
				if (E->next())
					p_string+=(",\n");
			}


			p_string+=("\n}");


		} break;
		case Variant::ARRAY: {

			p_string+=("[ ");
			Array array = p_variant;
			int len=array.size();
			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");
				_get_property_as_text(array[i],p_string);

			}
			p_string+=(" ]");

		} break;

		case Variant::RAW_ARRAY: {

			p_string+=("PoolByteArray( ");
			String s;
			DVector<uint8_t> data = p_variant;
			int len = data.size();
			DVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();
			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");

				p_string+=(itos(ptr[i]));

			}

			p_string+=(" )");

		} break;
		case Variant::INT_ARRAY: {

			p_string+=("PoolIntArray( ");
			DVector<int> data = p_variant;
			int len = data.size();
			DVector<int>::Read r = data.read();
			const int *ptr=r.ptr();

			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");

				p_string+=(itos(ptr[i]));
			}


			p_string+=(" )");

		} break;
		case Variant::REAL_ARRAY: {

			p_string+=("PoolFloatArray( ");
			DVector<real_t> data = p_variant;
			int len = data.size();
			DVector<real_t>::Read r = data.read();
			const real_t *ptr=r.ptr();

			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");
				p_string+=(rtosfix(ptr[i]));
			}

			p_string+=(" )");

		} break;
		case Variant::STRING_ARRAY: {

			p_string+=("PoolStringArray( ");
			DVector<String> data = p_variant;
			int len = data.size();
			DVector<String>::Read r = data.read();
			const String *ptr=r.ptr();
			String s;
			//write_string("\n");



			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");
				String str=ptr[i];
				p_string+=("\""+str.c_escape()+"\"");
			}

			p_string+=(" )");

		} break;
		case Variant::VECTOR2_ARRAY: {

			p_string+=("PoolVector2Array( ");
			DVector<Vector2> data = p_variant;
			int len = data.size();
			DVector<Vector2>::Read r = data.read();
			const Vector2 *ptr=r.ptr();

			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");
				p_string+=(rtosfix(ptr[i].x)+", "+rtosfix(ptr[i].y) );
			}

			p_string+=(" )");

		} break;
		case Variant::VECTOR3_ARRAY: {

			p_string+=("PoolVector3Array( ");
			DVector<Vector3> data = p_variant;
			int len = data.size();
			DVector<Vector3>::Read r = data.read();
			const Vector3 *ptr=r.ptr();

			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");
				p_string+=(rtosfix(ptr[i].x)+", "+rtosfix(ptr[i].y)+", "+rtosfix(ptr[i].z) );
			}

			p_string+=(" )");

		} break;
		case Variant::COLOR_ARRAY: {

			p_string+=("PoolColorArray( ");

			DVector<Color> data = p_variant;
			int len = data.size();
			DVector<Color>::Read r = data.read();
			const Color *ptr=r.ptr();

			for (int i=0;i<len;i++) {

				if (i>0)
					p_string+=(", ");

				p_string+=(rtosfix(ptr[i].r)+", "+rtosfix(ptr[i].g)+", "+rtosfix(ptr[i].b)+", "+rtosfix(ptr[i].a) );

			}
			p_string+=(" )");

		} break;
		default: {}

	}

	return OK;
}

void EditorExportGodot3::_save_text(const String& p_path,ExportData &resource) {

	FileAccessRef f = FileAccess::open(p_path,FileAccess::WRITE);

	if (resource.nodes.size()) {
		f->store_line("[gd_scene load_steps="+itos(resource.nodes.size()+resource.resources.size())+" format=2]\n");
	} else {
		f->store_line("[gd_resource type=\""+resource.resources[resource.resources.size()-1].type+"\" load_steps="+itos(resource.resources.size())+" format=2]\n");
	}

	for (Map<int,ExportData::Dependency>::Element *E=resource.dependencies.front();E;E=E->next()) {

		f->store_line("[ext_resource type=\""+E->get().type+"\" path=\""+E->get().path+"\" id="+itos(E->key())+"]\n");

	}

	for (int i=0;i<resource.resources.size();i++) {

		if (resource.nodes.size() || i<resource.resources.size()-1) {

			f->store_line("\n[sub_resource type=\""+resource.resources[i].type+"\" id="+itos(resource.resources[i].index)+"]\n");
		} else {
			f->store_line("\n[resource]\n");
		}

		for( List<ExportData::PropertyData>::Element *E=resource.resources[i].properties.front();E;E=E->next() ) {

			String prop;
			_get_property_as_text(E->get().value,prop);
			f->store_line(E->get().name+"="+prop);

		}


	}

	for (int i=0;i<resource.nodes.size();i++) {

		String node_txt = "\n[node ";

		if (resource.nodes[i].name!=String()) {
			node_txt+=" name=\""+String(resource.nodes[i].name).c_escape()+"\"";
		}

		if (resource.nodes[i].parent!=NodePath()) {
			node_txt+=" parent=\""+String(resource.nodes[i].parent).c_escape()+"\"";
		}

		if (resource.nodes[i].owner!=NodePath()) {
			node_txt+=" owner=\""+String(resource.nodes[i].owner).c_escape()+"\"";
		}

		if (resource.nodes[i].type!=String()) {
			node_txt+=" type=\""+resource.nodes[i].type+"\"";
		}

		if (resource.nodes[i].instance!=String()) {
			String prop;
			_get_property_as_text(resource.nodes[i].instance,prop);
			node_txt+=" instance="+prop+"";
		}

		node_txt+="]\n";
		f->store_line(node_txt);


		for( List<ExportData::PropertyData>::Element *E=resource.nodes[i].properties.front();E;E=E->next() ) {

			String prop;
			_get_property_as_text(E->get().value,prop);
			f->store_line(E->get().name+"="+prop);
		}


	}

	for(int i=0;i<resource.connections.size();i++) {

		String prop;
		_get_property_as_text(resource.connections[i].binds,prop);

		f->store_line("\n[connection signal=\""+resource.connections[i].signal+"\"  from=\""+String(resource.connections[i].from).c_escape()+"\"  to=\""+String(resource.connections[i].to).c_escape()+"\" method=\""+resource.connections[i].method+"\" binds="+prop+"]");

	}

	for(int i=0;i<resource.editables.size();i++) {

		f->store_line("[editable path=\""+String(resource.editables[i]).c_escape()+"\"]");
	}

}

void EditorExportGodot3::_save_binary(const String& p_path,ExportData &resource) {



}

void EditorExportGodot3::_save_config(const String &p_path) {

	FileAccessRef f = FileAccess::open(p_path.plus_file("godot.cfg"),FileAccess::WRITE);

	f->store_line("[application]\n");
	String name = Globals::get_singleton()->get("application/name");
	f->store_line("name=\""+name.c_escape()+"\"");
	String main_scene = Globals::get_singleton()->get("application/main_scene");
	f->store_line("main_scene=\""+_replace_resource(main_scene).c_escape()+"\"");


}

Error EditorExportGodot3::export_godot3(const String& p_path) {

	List<String> files;
	_find_files(EditorFileSystem::get_singleton()->get_filesystem(),&files);

	EditorProgress progress("exporting","Exporting Godot 3.0",files.size());

	//find XML resources

	resource_replace_map.clear();

	Set<String> xml_extensions;
	Set<String> binary_extensions;
	Set<String> text_extensions;

	{
		List<String> xml_exts;
		ResourceFormatLoaderXML::singleton->get_recognized_extensions(&xml_exts);
		for (List<String>::Element *E=xml_exts.front();E;E=E->next()) {
			xml_extensions.insert(E->get());
		}
	}

	{
		List<String> binary_exts;
		ResourceFormatLoaderBinary::singleton->get_recognized_extensions(&binary_exts);
		for (List<String>::Element *E=binary_exts.front();E;E=E->next()) {
			binary_extensions.insert(E->get());
		}
	}

	{
		List<String> text_exts;
		ResourceFormatLoaderText::singleton->get_recognized_extensions(&text_exts);
		for (List<String>::Element *E=text_exts.front();E;E=E->next()) {
			text_extensions.insert(E->get());
		}
	}

	for(List<String>::Element *E=files.front();E;E=E->next()) {

		String file=E->get();
		if (xml_extensions.has(file.extension().to_lower())) {
			if (ResourceLoader::get_resource_type(file)=="PackedScene") {
				resource_replace_map[file]=file.basename()+".tscn";
			} else {
				resource_replace_map[file]=file.basename()+".tres";
			}
		}
	}

	DirAccess *directory = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	if (directory->change_dir(p_path)!=OK) {
		memdelete(directory);
		ERR_FAIL_V(ERR_CANT_OPEN);
	}


	int idx=0;
	for(List<String>::Element *E=files.front();E;E=E->next()) {

		String path = E->get();
		String extension = path.extension().to_lower();

		String target_path;
		bool repack=false;

		target_path = p_path.plus_file(path.replace("res://",""));

		progress.step(target_path.get_file(),idx++);

		print_line("exporting: "+target_path);

		if (directory->make_dir_recursive(target_path.get_base_dir())!=OK) {
			memdelete(directory);
			ERR_FAIL_V(ERR_CANT_CREATE);
		}

		ExportData resource_data;

		if (xml_extensions.has(extension)) {
			Error err = ResourceLoader::get_export_data(path,resource_data);
			if (err!=OK) {
				memdelete(directory);
				ERR_FAIL_V(err);
			}
		} else if (text_extensions.has(extension)) {
			Error err = ResourceLoader::get_export_data(path,resource_data);
			if (err!=OK) {
				memdelete(directory);
				ERR_FAIL_V(err);
			}
		} else if (binary_extensions.has(extension)) {

			Error err = ResourceLoader::get_export_data(path,resource_data);
			if (err!=OK) {
				memdelete(directory);
				ERR_FAIL_V(err);
			}

		} else {
			//single file, copy it
			Error err = directory->copy(path,target_path);
			if (err!=OK) {
				memdelete(directory);
				ERR_FAIL_V(err);
			}
			continue; //no longer needed to do anything, just copied the file!
		}
		
		if (resource_data.nodes.size()==0 && resource_data.resources[resource_data.resources.size()-1].type=="PackedScene") {
			//must unpack a PackedScene
			_unpack_packed_scene(resource_data);
			repack=true;
		}

		_convert_resources(resource_data);

		if (repack) {

			_pack_packed_scene(resource_data);

		}


		if (xml_extensions.has(extension)) {

			String save_path=resource_replace_map[target_path];
			_save_text(save_path,resource_data);
		} else if (text_extensions.has(extension)) {
			_save_text(target_path,resource_data);
		} else if (binary_extensions.has(extension)) {
			_save_binary(target_path,resource_data);
		}

	}

	memdelete(directory);
	_save_config(p_path);

	return OK;
}

EditorExportGodot3::EditorExportGodot3() {

	int idx=0;
	while(prop_renames[idx][0]!=NULL) {

		prop_rename_map[prop_renames[idx][0]]=prop_renames[idx][1];
		idx++;
	}

	idx=0;

	while(type_renames[idx][0]!=NULL) {

		type_rename_map[type_renames[idx][0]]=type_renames[idx][1];
		idx++;
	}

	idx=0;

	while(signal_renames[idx][0]!=NULL) {

		signal_rename_map[signal_renames[idx][0]]=signal_renames[idx][1];
		idx++;
	}
}
