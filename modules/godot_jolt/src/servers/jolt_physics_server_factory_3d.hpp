#pragma once

#include "servers/jolt_physics_server_3d.hpp"

class JoltPhysicsServerFactory3D final : public Object {
	GDCLASS(JoltPhysicsServerFactory3D, Object)

private:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("create_server"), &JoltPhysicsServerFactory3D::create_server);
	}

public:
	JoltPhysicsServer3D* create_server() {print_line("JoltPhysicsServerFactory3D::create_server()");  return memnew(JoltPhysicsServer3D); }
};
