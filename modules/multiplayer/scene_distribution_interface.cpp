#include "scene_distribution_interface.h"

#include "scene_multiplayer.h"
#include "core/io/marshalls.h"
#include "scene/main/multiplayer_api.h"

#include <stdio.h>
#include <fstream>

#include "modules/gltf/gltf_document.h"

#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/packed_scene.h"
#include "core/io/resource_saver.h"

//perhaps for android _popen() ??
#include <stdlib.h>

void SceneDistributionInterface::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_own_peer_as_glb_creator"), &SceneDistributionInterface::set_own_peer_as_glb_creator);
	ClassDB::bind_method(D_METHOD("request_glb", "glb_name"), &SceneDistributionInterface::request_glb);
	
	ADD_SIGNAL(MethodInfo("_set_glb_creator_peer", PropertyInfo(Variant::INT, "peer")));
}


// Set the caller peer to be a glb creator. That means, you can then request glb files
// Tell all clients that we are the glb creator peer
void SceneDistributionInterface::set_own_peer_as_glb_creator()
{
	printf("SceneDistributionInterface::set_own_peer_as_glb_creator\n");

	//check if the glb_create batch script is there
	Error err;
	Ref<FileAccess> f = FileAccess::open(externally_create_glb_script, FileAccess::READ, &err);
	if (f.is_null()) {
		printf("There is no create glb script at %s , so cannot set glb_creator\n", externally_create_glb_script.ascii().get_data());
		return;
	}


	//check if glb storage path directory is there
	struct stat st;
	int ret;
	ret = stat(externally_created_glb_storage_path.ascii().get_data(), &st);
	if (ret != 0) {
		printf("There is no create glb storage directory at %s , so cannot set glb_creator\n", externally_created_glb_storage_path.ascii().get_data());
		return;
	}

	//set own peer as glb creator
	_glb_creator_peer = multiplayer->get_unique_id();

	//tell all other that we are a glb creator
	emit_signal(SNAME("_set_glb_creator_peer"), _glb_creator_peer);
}


// send request of glb file to the peer that used set_own_peer_as_glb_creator
// The glb_name should only be a name WITHOUT .glb E.g. "Fox"
void SceneDistributionInterface::request_glb(const String& glb_name)
{
	printf("SceneDistributionInterface::request_glb ->%s\n", glb_name.ascii().get_data());

	//check if we got a glb_creator_peer and that we are ourself not the glb_creator_peer
	if (multiplayer->get_distributor()->get_glb_creator_peer() > 0 &&
		multiplayer->get_distributor()->get_glb_creator_peer() != multiplayer->get_unique_id()) {

		int packet_len = SceneMultiplayer::SYS_CMD_SIZE + (glb_name.size() * 4);

		std::vector<uint8_t> buf(packet_len, 0);
		buf[0] = SceneMultiplayer::NETWORK_COMMAND_SYS;
		buf[1] = SceneMultiplayer::SYS_COMMAND_REQUEST_GLB;
		multiplayer->get_multiplayer_peer()->set_transfer_channel(0);
		multiplayer->get_multiplayer_peer()->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
		encode_uint32(multiplayer->get_unique_id(), &buf[2]);

		encode_cstring(glb_name.utf8().get_data(), &buf[6]);

		//send our request to glb_creator_peer
		multiplayer->get_multiplayer_peer()->set_target_peer(multiplayer->get_distributor()->get_glb_creator_peer());
		multiplayer->get_multiplayer_peer()->put_packet(buf.data(), packet_len);
	}
	else if (multiplayer->get_distributor()->get_glb_creator_peer() == multiplayer->get_unique_id()) {
		printf("we are ourself the glb_creator_peer, doing nothing right now\n");
	}
	else if (multiplayer->get_distributor()->get_glb_creator_peer() <= 0) {
		printf("we got NO glb_creator_peer\n");
	}
		

}

void SceneDistributionInterface::_distribute_glb(const String& p_path, int id)
{
	printf("SceneDistributionInterface::distribute_glb\n");
	printf("glb-file-name:%s\n",p_path.ascii().get_data());

	//load glb file into PackedByteArray
	Ref<GLTFDocument> gltf;
	gltf.instantiate();
	Ref<GLTFState> gltf_state;
	gltf_state.instantiate();

	String externally_created_glb_storage_path_file = externally_created_glb_storage_path + p_path;
	Error err;
	PackedByteArray glb_file_PBA;
	Ref<FileAccess> f = FileAccess::open(externally_created_glb_storage_path_file, FileAccess::READ, &err);

	Vector<uint8_t> data;

	if (err != OK) {
		printf("open external glb error\n");
	}
	else if (f.is_valid()) {
		
		data.resize(f->get_length());
		printf("glb file length:%lld\n", f->get_length());
		f->get_buffer(data.ptrw(), f->get_length());

		glb_file_PBA.resize(f->get_length());
		//f->get_buffer(glb_file_PBA.ptrw(), f->get_length());  //does not work
		
		for (uint64_t i = 0; i < f->get_length(); i++) {
			glb_file_PBA.set(i,data[i]);
		}
	}

	//write it to own user://
	//String save_path = "res://" + p_path.replace(".glb", ".gltf");
	String save_path = "res://" + p_path.replace(".glb", ".scn");

	
	gltf->append_from_buffer(glb_file_PBA, "base_path?", gltf_state);

	Node* n = gltf->generate_scene(gltf_state);
	n->set_name(save_path);
	Ref<PackedScene> p = memnew(PackedScene);
	p->pack(n);
	ResourceSaver s;
	Error error = s.save(p, save_path);  // Or "user://..."

	// add new resource to own MultiplayerSpawner.add_spawnable_scene
	Node* root_node = SceneTree::get_singleton()->get_root()->get_node(multiplayer->get_root_path());
	Node* node = root_node->get_node(NodePath("/root/Main/PlayerSpawner"));
	MultiplayerSpawner* spawner = Object::cast_to<MultiplayerSpawner>(node);
	spawner->add_spawnable_scene(save_path);


	//send it to the clients and MultiplayerSpawner.add_spawnable_scene
	int packet_len = SceneMultiplayer::SYS_CMD_SIZE + 4 + 4 + p_path.size() + glb_file_PBA.size() ;
	printf("packet_len:%d\n", packet_len);

	std::vector<uint8_t> buf(packet_len, 0);
	buf[0] = SceneMultiplayer::NETWORK_COMMAND_SYS;
	buf[1] = SceneMultiplayer::SYS_COMMAND_DISTRIBUTE_GLB;
	multiplayer->get_multiplayer_peer()->set_transfer_channel(0);
	multiplayer->get_multiplayer_peer()->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
	encode_uint32(multiplayer->get_unique_id(), &buf[2]);

	//encode length of glb file name
	encode_uint32(p_path.length(), &buf[6]);
	//encode size of glb_file_PBA
	encode_uint32(glb_file_PBA.size(), &buf[10]);
	//encode glb file name
	encode_cstring(p_path.ascii().get_data(), &buf[14]);

	//save glb_file_PBA to buf
	for (int i = 0; i < glb_file_PBA.size(); i++) {
		buf[p_path.length() + SceneMultiplayer::SYS_CMD_SIZE + 4 + 4 + i] = glb_file_PBA[i];
	}

	//printf("buf-out: %x", decode_uint32( &buf[p_path.length() + SceneMultiplayer::SYS_CMD_SIZE + 4 + 4]) );

	//send it to all clients, but not ourself
	int own_peer = multiplayer->get_unique_id();
	for (const int& P : multiplayer->get_connected_peers()) {
		if (P != own_peer) {
			multiplayer->get_multiplayer_peer()->set_target_peer(P);
			multiplayer->get_multiplayer_peer()->put_packet(buf.data(), packet_len);
		}
	}
}



HashSet<String> SceneDistributionInterface::get_requested_glb_files()
{
	return requested_glb_files;
}

// run only by the peer that set itself as glb_creator_peer
// Here we run an external batch file, that will create the glb file.
// Download from sketchfab or create with AI tool e.g. stable-dreamfusion
void SceneDistributionInterface::request_to_externally_create_glb(const String& glb_name)
{
	//printf("SceneDistributionInterface::request_to_externally_create_glb %s\n", glb_name.ascii().get_data());

	FILE* fp;
	int status;
	char return_value[100];

	fp = _popen(externally_create_glb_script.ascii().get_data(), "r");
	if (fp == NULL)
		printf("_popen-open error\n");

	while (fgets(return_value, 200, fp) != NULL)
		printf("%s", return_value);

	// if we get a "bad" return value, we could remove the requested_glb, so
	// that the poll() function in scene_multiplayer.cpp is not looking for the file anymore
	String s(return_value);
	if (s.contains("ERROR")) {
		printf("request_to_externally_create_glb ERROR\n");
		_remove_glb_as_requested(glb_name.ascii().get_data());
	}

	status = _pclose(fp);
	if (status == -1) {
		/* Error reported by pclose() */
		printf("_pclose error\n");
	}
	//else {
	//	/* Use macros described under wait() to inspect `status' in order
	//	   to determine success/failure of command executed by popen() */
	//	printf("_pclose no error\n");
	//}
}

void SceneDistributionInterface::check_if_externally_created_glb_was_created()
{
	//printf("SceneDistributionInterface::check_if_externally_created_glb_was_created");

	HashSet<String>::Iterator it;
	for (it = requested_glb_files.begin(); it; ++it) {
		//printf("it is:%s\n", it->ascii().get_data() );
	
		//check in a pre-specified directory, if there is a file with a name, we got in
		//our requested_glb_files hashset
		String s = externally_created_glb_storage_path + *it;
		//printf("s is:%s\n", s.ascii().get_data() );
		std::ifstream f(s.ascii().get_data());
		if (f.good()) {
			//printf("Found requested glb\n");

			//remove glb-name from requested_glb_files
			requested_glb_files.remove(it);

			//distribute it now
			_distribute_glb(it->ascii().get_data(), 1);
		}
	}
}

void SceneDistributionInterface::set_glb_creator_peer(int peer)
{
	_glb_creator_peer = peer;
}

int SceneDistributionInterface::get_glb_creator_peer()
{
	return _glb_creator_peer;
}

void SceneDistributionInterface::set_glb_as_requested(const String& glb_name)
{
	//check if someone already requested this glb_name file in the past
	if (requested_glb_files.has(glb_name.ascii().get_data())) {
		//printf("glb file %s is already requested\n", glb_name.ascii().get_data());
	}
	else {
		//printf("glb file %s inserted to requested HashSet\n", glb_name.ascii().get_data());
		requested_glb_files.insert(glb_name.ascii().get_data());
	}
}

void SceneDistributionInterface::_remove_glb_as_requested(const String& glb_name)
{
	if (requested_glb_files.has(glb_name.ascii().get_data())) {
		requested_glb_files.remove( requested_glb_files.find(glb_name.ascii().get_data()) );
	}
}
