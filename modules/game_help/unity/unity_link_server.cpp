/**************************************************************************/
/*  editor_file_server.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "unity_link_server.h"

#include "core/io/marshalls.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "../logic/animator/body_animator.h"
#include "../logic/data_table_manager.h"
#include "../logic/character_shape/character_body_prefab.h"

#if TOOLS_ENABLED
#include "editor/import/3d/resource_importer_scene.h"
#include "core/io/config_file.h"
#endif

#define FILESYSTEM_PROTOCOL_VERSION 1
#define PASSWORD_LENGTH 32
#define MAX_FILE_BUFFER_SIZE 100 * 1024 * 1024 // 100mb max file buffer size (description of files to update, compressed).

UnityLinkServer * UnityLinkServer::instance = nullptr;
static bool read_string(StreamPeerConstBuffer& msg_buffer,String &str)
{
	int len = msg_buffer.get_u32();
	if (len < 0 || len > 10240) {
		return false;
	}
	str = msg_buffer.get_utf8_string(len);
	return true;

}


static Node* import_fbx(const String &p_path)
{
	
#if TOOLS_ENABLED
	HashMap<StringName, Variant> defaults;
	Dictionary base_subresource_settings;
	
	base_subresource_settings.clear();

	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load(p_path + ".import");
	if (err == OK) {
		List<String> keys;
		config->get_section_keys("params", &keys);
		for (const String &E : keys) {
			Variant value = config->get_value("params", E);
			if (E == "_subresources") {
				base_subresource_settings = value;
			} else {
				defaults[E] = value;
			}
		}
	}

	return ResourceImporterScene::get_scene_singleton()->pre_import(p_path, defaults); // Use the scene singleton here because we want to see the full thing.
#endif
	return nullptr;

}
// 處理導入的骨架
static Skeleton3D * process_fbx_skeleton(const String &p_path,Node *p_node)
{
	Node * node = p_node->find_child("Skeleton3D");
	Skeleton3D * skeleton = Object::cast_to<Skeleton3D>(node);
	return skeleton;
}
// 保存模型资源
static void save_fbx_res( const String& group_name,const String& sub_path,const Ref<Resource>& p_resource,String& fbx_path, bool is_resource = true)
{
	String export_root_path = "res://Assets/public";
	if (!DirAccess::dir_exists_absolute(export_root_path))
	{
		print_error("Export Root Path不存在:" + export_root_path);
		return ;

	}
	Ref<DirAccess> dir = DirAccess::create_for_path(export_root_path);
	if (!dir->dir_exists(group_name))
	{
		dir->make_dir(group_name);

	}
	dir->change_dir(group_name);
	if (!dir->dir_exists(sub_path))
	{
			dir->make_dir(sub_path);
	}
	dir->change_dir(sub_path);
	fbx_path = export_root_path.path_join(group_name).path_join(sub_path).path_join(p_resource->get_name() + (is_resource ? "tres" :".tscn"));
	ResourceSaver::save(p_resource,fbx_path);
}
static void get_fbx_meshs(Node *p_node,HashMap<String,MeshInstance3D* > &meshs)
{

	for(int i=0;i<p_node->get_child_count();i++)
	{
		Node * child = p_node->get_child(i);
		if(child->get_class() == "MeshInstance3D")
		{
			MeshInstance3D * mesh = Object::cast_to<MeshInstance3D>(child);
			if(!meshs.has(mesh->get_name())){
				meshs[mesh->get_name()] = mesh;
			}
			else{
				String name = mesh->get_name();
				int index = 1;
				while(meshs.has(name +"_"+ itos(index))){
					name = mesh->get_name().str() + "_" + itos(index);
					index++;
				}
				meshs[name] = mesh;
			}
		}
		get_fbx_meshs(child,meshs);
	}
}
// 處理導入的模型
static void process_fbx_mesh(const String &p_path,const String & p_group, Node *p_node)
{
	String name = p_node->get_name();
	// 存儲模型的預製體文件
	Skeleton3D * skeleton = process_fbx_skeleton(p_path,p_node);
	Dictionary bone_map;
	String ske_save_path,bone_map_save_path;
	if(skeleton != nullptr)
	{
		bone_map = skeleton->get_human_bone_mapping();
		skeleton->set_human_bone_mapping(bone_map);

		// 存儲骨架信息
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		packed_scene->pack(skeleton);
		save_fbx_res("skeleton",p_group,packed_scene,ske_save_path,false);
		Ref<CharacterBoneMap> bone_map_ref;
		bone_map_ref.instantiate();
		bone_map_ref->set_bone_map(bone_map);
		save_fbx_res("skeleton",p_group,bone_map_ref,bone_map_save_path,true);
	}
	HashMap<String,MeshInstance3D* > meshs;
	// 便利存儲模型文件
	get_fbx_meshs(p_node,meshs);

	Ref<CharacterBodyPrefab> prefab;
	prefab->set_name("prefab_" + name);
	prefab.instantiate();
	for(auto it = meshs.begin(); it != meshs.end(); ++it){
		Ref<CharacterBodyPart> part;
		part.instantiate();
		MeshInstance3D* mesh = it->value;
		part->init_form_mesh_instance(mesh,bone_map);
		String save_path = p_path + "_" + it->key + ".tres";
		String temp;
		save_fbx_res("mesh", p_group,part,temp,false);
		prefab->parts[save_path] = true;
		print_line("UnityLinkServer: save mesh node :" + save_path);	
	}
	prefab->skeleton_path = ske_save_path;

	// 保存预制体
	save_fbx_res("prefab",p_group,prefab,bone_map_save_path,true);



}
// 創建一個預製體
static void create_fbx_prefab(const String &p_path,const String& name, const LocalVector<String>& p_paths,const String& p_skeleton_path)
{
	Ref<CharacterBodyPrefab> prefab;
	prefab->set_name("prefab_" + name);
	for(int i=0; i<p_paths.size(); i++) {
		prefab->parts[p_paths[i]] = true;
	}
	prefab->skeleton_path = p_skeleton_path;
	String temp;
	// 保存预制体
	save_fbx_res("prefab",p_path,prefab,temp,true);
}

enum class  UnityDataType : int32_t {
	AnimationNode = 1,
	BoneMap,
	AnimationFIle,
	FbxMesh,
	MeshPrefab,
	SkeletonFile,
	ImageFile,
};
static bool poll_client(StreamPeerConstBuffer& msg_buffer) {
	int tag = msg_buffer.get_u32();
	int size = msg_buffer.get_u32();
	UnityDataType type = (UnityDataType)msg_buffer.get_u32();
	String path;
	if (!read_string(msg_buffer, path))
	{
		ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + path);

	}
	path = path.to_lower();
	path = path.replace("assets", "Assets");
	if(!DirAccess::exists("res://" + path))
	{
		// 创建目录
		Ref<DirAccess> dir_access = DirAccess::create(DirAccess::AccessType::ACCESS_RESOURCES);
		dir_access->make_dir_recursive(path);
	}

	if(type == UnityDataType::BoneMap) {
		// 解析骨骼映射文件
		int file_size = msg_buffer.get_32();
		if(file_size < 0 || file_size > 10240){
			ERR_FAIL_V_MSG(false,"UnityLinkServer: create bone map error :" + path);
			
		}
		String file_name = msg_buffer.get_utf8_string(file_size);
		String save_name = file_name.get_basename();

		Ref<CharacterBoneMap> bone_map;
		bone_map.instantiate();
		//bone_map->ref_skeleton_file_path = path + "/" + file_name;
		//bone_map->is_by_sekeleton_file = true;


		String save_path = "res://" + path + "/" + save_name +  ".bone_map.tres";
		ResourceSaver::save(bone_map,save_path);		
		print_line("UnityLinkServer: save bone map :" + save_path);
	}
	else if(type == UnityDataType::AnimationFIle) {
		Callable on_load_animation =  DataTableManager::get_singleton()->get_animation_load_cb();
		if(on_load_animation.is_null()){
			return true;
		}
		String name;
		if (!read_string(msg_buffer, name))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}

		// 解析动画文件
		int mirror = msg_buffer.get_32();
		
		int file_size = size - msg_buffer.get_position();
		String yaml_anim = msg_buffer.get_utf8_string(file_size);
		Ref<Animation> anim;
		anim.instantiate();
		Ref<JSON> json = DataTableManager::get_singleton()->parse_yaml(yaml_anim);
		auto data = json->get_data();
		if(data.get_type() == Variant::DICTIONARY)
		{
			Dictionary dict = data;
			if(dict.has("AnimationClip"))
			{
				Dictionary clip = dict["AnimationClip"];
				on_load_animation.call(clip,mirror,anim);
				anim->optimize();
				String save_path = "res://" + path + "/" + name + ".tres";
				ResourceSaver::save(anim,save_path);
				print_line("UnityLinkServer: save animation node :" + save_path);	
			}
			else
			{
				ERR_FAIL_V_MSG(false,"UnityLinkServer: create animation error " + path);
			}

		}	
		else
		{
			ERR_FAIL_V_MSG(false,"UnityLinkServer: create animation error " + path);
		}
	}
	else if(type == UnityDataType::FbxMesh) {
		// fbx 模型

		String name,group;
		if (!read_string(msg_buffer, name))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}
		if (!read_string(msg_buffer, group))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}
		int file_size = size - msg_buffer.get_position();

		Ref<FileAccess> f = FileAccess::open("res://" + path + "/" + name,FileAccess::WRITE);
		f->store_buffer(msg_buffer.get_u8_ptr(),file_size);	
		Node* p_node = import_fbx("res://" + path + "/" + name);
		if(p_node != nullptr){
			process_fbx_mesh("res://" + path + "/" + name,group,p_node);			
		}
	}
	else if(type == UnityDataType::MeshPrefab) {
		// 预制体信息

		String name,group,skeleton_path;
		if (!read_string(msg_buffer, name))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}
		if (!read_string(msg_buffer, group))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);	
		}
		if (!read_string(msg_buffer, skeleton_path))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);	
		}
		
		Ref<CharacterBodyPrefab> prefab;
		prefab->set_name("prefab_" + name);
		prefab.instantiate();
		prefab->skeleton_path = skeleton_path;

		int mesh_count = msg_buffer.get_32();
		for(int i = 0;i < mesh_count;i++)
		{
			String mesh_path,mesh_name;
			if (!read_string(msg_buffer, mesh_path))
			{
				ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

			}
			if (!read_string(msg_buffer, mesh_name))
			{
				ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);
			}
			String save_path = mesh_path + "_" + mesh_name + ".tres";
			prefab->parts[mesh_path + "/" + mesh_name] = true;
		}
		String bone_map_save_path;
		save_fbx_res("prefab",group,prefab,bone_map_save_path,true);

	}
	else if(type == UnityDataType::SkeletonFile) {

		// 骨架文件
		String name,group;
		if (!read_string(msg_buffer, name))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}
		if (!read_string(msg_buffer, group))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}
		int file_size = size - msg_buffer.get_position();

		Ref<FileAccess> f = FileAccess::open("res://" + path + "/" + name,FileAccess::WRITE);
		f->store_buffer(msg_buffer.get_u8_ptr(),file_size);	
		
		Node* p_node = import_fbx("res://" + path + "/" + name);
		if(p_node != nullptr){
			String name = p_node->get_name();
			// 存儲模型的預製體文件
			Skeleton3D * skeleton = process_fbx_skeleton(path,p_node);
			Dictionary bone_map;
			String ske_save_path,bone_map_save_path;
			if(skeleton != nullptr)
			{
				bone_map = skeleton->get_human_bone_mapping();
				skeleton->set_human_bone_mapping(bone_map);

				// 存儲骨架信息
				Ref<PackedScene> packed_scene;
				packed_scene.instantiate();
				packed_scene->pack(skeleton);
				save_fbx_res("skeleton", group,packed_scene,ske_save_path,false);
				Ref<CharacterBoneMap> bone_map_ref;
				bone_map_ref.instantiate();
				bone_map_ref->set_bone_map(bone_map);
				save_fbx_res("skeleton", group,bone_map_ref,bone_map_save_path,true);
			}

			
		}
	}
	else if(type == UnityDataType::ImageFile) {
		// 直接存儲的文件，png。。。。

		String name;
		if (!read_string(msg_buffer, name))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}
		int file_size = size - msg_buffer.get_position();

		Ref<FileAccess> f = FileAccess::open("res://" + path + "/" + name,FileAccess::WRITE);
		f->store_buffer(msg_buffer.get_u8_ptr(),file_size);	

	}
	return true;
}
bool UnityLinkServer::ClientPeer::poll()
{
	
	connection->poll();
	if(connection->get_status() != StreamPeerTCP::STATUS_CONNECTED){
		if(connection->get_status() == StreamPeerTCP::STATUS_ERROR || connection->get_status() == StreamPeerTCP::STATUS_NONE){
			return false;
		}
		return true;
	}
	if(connection->get_available_bytes() == 0){
		return true;
	}
	print_line(String("UnityLinkServer: poll") + String(connection->get_connected_host()) + ":" + itos(connection->get_connected_port()));
	if(buffer_size == 0){
		buffer_size = 102400;
		data = memnew_arr(uint8_t, buffer_size);
	}
	bool msg_finish = false;
	while (true) {
		int read = 0;
		Error err = connection->get_partial_data(&data[curr_read_count], 1, read);
		if (err != OK) {
			return false;
		} else if (read != 1) { // Busy, wait until next poll
			return true;
		}
		++curr_read_count;
		if(!is_msg_statred && curr_read_count == 8)
		{
			int msg_size = *(((int *)data)+1);
			// 小於0 或者大於40兆代表無效數據
			if(msg_size < 0 || msg_size > 40960000)
			{
				return false;
			}
			if(buffer_size < msg_size)
			{
				buffer_size = msg_size;
				uint8_t* old = data;
				data = memnew_arr(uint8_t, buffer_size);
				
				*(((int *)data)) = *(((int *)old));
				*(((int *)data)+1) = msg_size;
				memdelete_arr(old);
			}
			is_msg_statred = true;

		}
		if(curr_read_count >= 8 && curr_read_count == get_msg_size())
		{
			msg_finish = true;
			break;
		}
	}
	if(!msg_finish){
		return true;
	}
	print_line(String("UnityLinkServer: msg_finish") + String(connection->get_connected_host()) + ":" + itos(connection->get_connected_port()) + " msg size:" + itos(get_msg_size()) + " curr_read_count:" + itos(curr_read_count));
	is_msg_statred = false;
	if(!process_msg()){
		curr_read_count = 0;
		return false;
	}
	curr_read_count = 0;
	return true;
}


bool UnityLinkServer::ClientPeer::process_msg()
{
	StreamPeerConstBuffer buffer;
	buffer.set_data_array(data, get_msg_size());
	return poll_client(buffer);
}
void UnityLinkServer::poll() {
	if (!active) {
		return;
	}

	if (!server->is_connection_available()) {
		return;
	}
	// 接收客户端
	while(true)
	{
		Ref<StreamPeerTCP> tcp_peer = server->take_connection();
		if(tcp_peer.is_null()){
			break;
		}
		// Got a connection!
		tcp_peer->set_no_delay(true);
		print_line(String("UnityLinkServer: Got a connection->") + String(tcp_peer->get_connected_host()) + ":" + itos(tcp_peer->get_connected_port()));
		Ref<ClientPeer> peer ;
		peer.instantiate();
		peer->connection = tcp_peer;
		clients.push_back(peer);

	}
	// 遍历客户端
	auto it = clients.begin();
	while(it != clients.end())
	{
		Ref<ClientPeer>& peer = *it;
		if(!peer->poll()){
			print_line(String("UnityLinkServer: disconnect") + String(peer->connection->get_connected_host()) + ":" + itos(peer->connection->get_connected_port()));
			it = clients.erase(it);
			continue;
		}
		++it;
	}

}

void UnityLinkServer::start() {
	if (active) {
		stop();
	}
	port = 9010;
	Error err = server->listen(port);
	ERR_FAIL_COND_MSG(err != OK, String("UnityLinkServer: Unable to listen on port ") + itos(port));
	active = true;
}

bool UnityLinkServer::is_active() const {
	return active;
}

void UnityLinkServer::stop() {
	if (active) {
		server->stop();
		active = false;
	}
}

UnityLinkServer::UnityLinkServer() {
	instance = this;
	server.instantiate();

}

UnityLinkServer::~UnityLinkServer() {
	instance = nullptr;
	stop();
}


