#ifndef SCENE_DISTRIBUTION_INTERFACE_H
#define SCENE_DISTRIBUTION_INTERFACE_H

#include "core/object/ref_counted.h"
#include <vector>


class SceneMultiplayer;

class SceneDistributionInterface : public RefCounted {
	GDCLASS(SceneDistributionInterface, RefCounted);

private:
	SceneMultiplayer* multiplayer = nullptr;

	static void _bind_methods();

	//The directory where external programs save the created glb file
	String externally_created_glb_storage_path = "C:/Users/inflo/Documents/godot-workspace/godot-data/requested_glb/";
	//The script to call, to create a glb file
	String externally_create_glb_script = "C:/Users/inflo/Documents/godot-workspace/godot-data/scripts/create_glb.bat";

	//here save requested glb files. If they are created and distributed, they are removed. Pending requests
	HashSet<String> requested_glb_files;

	// used only inside scene_distribution_interface.cpp
	void _distribute_glb(const String& p_path, int id);
	void _remove_glb_as_requested(const String& glb_name);

	//the peer that is able to create glb files with external tools
	int _glb_creator_peer = -1;

public:
	// used in _bind_methods() to be used in GDScript
	void set_own_peer_as_glb_creator();
	void request_glb(const String& glb_name);


	// used in scene_multiplayer.cpp
	void set_glb_as_requested(const String& glb_name);
	HashSet<String> get_requested_glb_files();
	void request_to_externally_create_glb(const String& glb_name);
	void check_if_externally_created_glb_was_created();
	void set_glb_creator_peer(int peer);
	int get_glb_creator_peer();


	//will be called in multiplayer constructor
	SceneDistributionInterface(SceneMultiplayer* p_multiplayer) {
		multiplayer = p_multiplayer;
	}
};

#endif // SCENE_DISTRIBUTION_INTERFACE_H
