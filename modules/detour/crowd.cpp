#include "crowd.h"
#include "navmesh_query.h"
#include <DetourCrowd.h>

#define MAX_AGENTS 20

DetourCrowdManager::DetourCrowdManager() :
		Node(),
		crowd(0),
		dirty(false),
		initialized(false),
		query(memnew(DetourNavigationQuery)),
		max_agents(20),
		max_agent_radius(0.0f) {}
DetourCrowdManager::~DetourCrowdManager() {
	if (crowd) {
		if (initialized)
			dtFreeCrowd(crowd);
		else
			dtFree(crowd);
	}
}
void DetourCrowdManager::set_navigation_mesh(
		const Ref<DetourNavigationMesh> &navmesh, const Transform &xform) {
	this->navmesh = navmesh;
	this->transform = xform;
	query->init(navmesh, xform);

	create_crowd();
}
Ref<DetourNavigationMesh> DetourCrowdManager::get_navigation_mesh() const {
	return navmesh;
}
void DetourCrowdManager::add_agent(Object *agent, int mode, bool signals) {
	Spatial *obj = Object::cast_to<Spatial>(agent);
	if (!obj)
		return;
	AgentData *new_agent = memnew(AgentData);
	new_agent->obj = obj;
	new_agent->mode = mode;
	new_agent->radius = 1.0f;
	new_agent->height = 2.0f;
	new_agent->max_accel = 0.5f;
	new_agent->max_speed = 100.0f;
	new_agent->filter_id = 0;
	new_agent->oa_id = 0;
	new_agent->mode = mode;
	new_agent->send_signals = signals;
	Vector3 pos = obj->get_global_transform().origin;
	dtCrowdAgentParams params;
	memset(&params, 0, sizeof(params));
	params.radius = new_agent->radius;
	params.height = new_agent->height;
	params.maxAcceleration = new_agent->max_accel;
	params.maxSpeed = new_agent->max_speed;
	params.pathOptimizationRange = params.radius * 30.0f;
	params.queryFilterType = (unsigned char)new_agent->filter_id;
	params.obstacleAvoidanceType = (unsigned char)new_agent->oa_id;
	agents.push_back(new_agent);
	params.userData = new_agent;
	int id = crowd->addAgent(&pos.coord[0], &params);
	new_agent->id = id;

	if (max_agents < agents.size()) {
		max_agents = agents.size() + 20;
		create_crowd();
	}
	if (signals)
		obj->emit_signal("agent_added", id);
}
void DetourCrowdManager::remove_agent(Object *agent) {
	Spatial *obj = Object::cast_to<Spatial>(agent);
	if (obj)
		return;
	for (int i = 0; i < agents.size(); i++)
		if (agents[i]->obj == obj) {
			agents.remove(i);
			/* remove from crowd too */
			break;
		}
}
void DetourCrowdManager::clear_agent_list() {
	agents.clear();
}
void DetourCrowdManager::set_target(const Vector3 &position) {}
void DetourCrowdManager::set_velocity(const Vector3 &position) {}
void DetourCrowdManager::reset_target() {}
void DetourCrowdManager::set_max_agents(int max_agents) {
	this->max_agents = max_agents;
	create_crowd();
}
void DetourCrowdManager::set_max_agent_radius(float radius) {
	max_agent_radius = radius;
	create_crowd();
}
DetourCrowdManager::AgentData::AgentData() :
		obj(NULL),
		mode(0),
		id(0),
		radius(1.0f),
		height(2.0f),
		max_accel(10.5f),
		max_speed(30.0f),
		filter_id(0),
		oa_id(0) {}
DetourCrowdManager::AgentData::~AgentData() {}
void DetourCrowdManager::process_agent(dtCrowdAgent *agent) {
	if (!agent || !agent->params.userData)
		return;
	AgentData *data = (AgentData *)agent->params.userData;
	bool update_params = false;
	if (!data->obj)
		return;
	if (!data->obj->is_inside_tree())
		return;
	dtCrowdAgentParams params = agent->params;
#if 0
	if (!params.radius > 0.0f) {
		params.radius = data->radius;
		params.height = data->height;
		params.maxAcceleration = data->max_accel;
		params.pathOptimizationRange = data->radius * 30.0f;
		params.queryFilterType = data->filter_id;
		params.obstacleAvoidanceType = data->oa_id;
		update_params = true;
	}
#endif
	if (!params.updateFlags) {
		params.updateFlags = DT_CROWD_OPTIMIZE_TOPO | DT_CROWD_OPTIMIZE_VIS |
							 DT_CROWD_ANTICIPATE_TURNS | DT_CROWD_SEPARATION |
							 DT_CROWD_OBSTACLE_AVOIDANCE;
		update_params = true;
	}
	if (!(params.separationWeight > 0.0f)) {
		params.separationWeight = 4.0f;
		params.collisionQueryRange = 16.0f * data->radius; /* *radius */
		update_params = true;
	}
	if (update_params)
		crowd->updateAgentParameters(data->id, &params);
	Vector3 position;
	Vector3 velocity;
	Vector3 desired_velocity;
	memcpy(&position, agent->npos, sizeof(float) * 3);
	memcpy(&velocity, agent->vel, sizeof(float) * 3);
	memcpy(&desired_velocity, agent->dvel, sizeof(float) * 3);
	Transform transform = data->obj->get_global_transform();
	if (velocity.length() == 0.0f)
		velocity = transform.basis[2];
	if (data->mode == 0)
		data->obj->look_at_from_position(position, position + velocity,
				Vector3(0, 1, 0));
	if (data->send_signals)
		data->obj->emit_signal("agent_position", position, velocity,
				desired_velocity, (int)agent->state);
}
Vector3 DetourCrowdManager::_nearest_point(const Vector3 &point,
		int query_filter,
		polyref_t *nearest_ref) {
	if (!navmesh.is_valid() || !crowd)
		return point;
	polyref_t nearestRef = 0;
	Vector3 ret;
	if (!query) {
		if (nearest_ref)
			*nearest_ref = nearestRef;
		return point;
	}
	ret = query->nearest_point_(point, Vector3(*reinterpret_cast<const Vector3 *>(crowd->getQueryExtents())),
			crowd->getFilter(query_filter), &nearestRef);
	if (nearest_ref)
		*nearest_ref = nearestRef;
	return ret;
}
void DetourCrowdManager::set_agent_target_position(int id,
		const Vector3 &position) {
	uint64_t pref;
	Vector3 close = _nearest_point(position, 0, &pref);
	dtPolyRef nearestRef = (dtPolyRef)pref;
	crowd->requestMoveTarget(agents[id]->id, nearestRef, &close.coord[0]);
}
void DetourCrowdManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
			create_crowd();
			set_process(true);
			break;
		case NOTIFICATION_ENTER_TREE:
			create_crowd();
			break;
		case NOTIFICATION_EXIT_TREE:
			agents.clear();
			break;
		case NOTIFICATION_PROCESS:
			float delta = get_process_delta_time();
			// update_crowd(delta);
			if (crowd && navmesh.is_valid() && agents.size() > 0) {
				crowd->update(delta, NULL);
				Vector<dtCrowdAgent *> active_agents;
				active_agents.resize(agents.size());
				int nactive =
						crowd->getActiveAgents(&active_agents.write[0], agents.size());
				for (int i = 0; i < nactive; i++)
					process_agent(active_agents[i]);
			}
			break;
	}
}
void DetourCrowdManager::update_crowd(float delta) {
	if (dirty)
		create_crowd();
}
bool DetourCrowdManager::create_crowd() {
	dirty = true;
	if (!navmesh.is_valid())
		return false;
	if (crowd) {
		if (initialized)
			dtFreeCrowd(crowd);
		else
			dtFree(crowd);
		initialized = false;
	}
	crowd = dtAllocCrowd();
	if (max_agent_radius == 0.0f)
		max_agent_radius = navmesh->get_agent_radius();
	if (navmesh->get_navmesh() != NULL)
		print_line("good navmesh");
	else
		print_line("bad navmesh");
	if (!crowd->init(max_agents, max_agent_radius, navmesh->get_navmesh()))
		return false;
	dirty = false;
	initialized = true;
	return true;
}
void DetourCrowdManager::set_area_cost(int filter_id, int area_id, float cost) {
	dtQueryFilter *filter = crowd->getEditableFilter(filter_id);
	if (filter)
		filter->setAreaCost(area_id, cost);
}
float DetourCrowdManager::get_area_cost(int filter_id, int area_id) {
	const dtQueryFilter *filter = crowd->getFilter(filter_id);
	if (!filter)
		return 1.0f;
	return filter->getAreaCost(area_id);
}
void DetourCrowdManager::set_include_flags(int filter_id,
		unsigned short flags) {
	dtQueryFilter *filter = crowd->getEditableFilter(filter_id);
	if (filter)
		filter->setIncludeFlags(flags);
}
void DetourCrowdManager::set_exclude_flags(int filter_id,
		unsigned short flags) {
	dtQueryFilter *filter = crowd->getEditableFilter(filter_id);
	if (filter)
		filter->setExcludeFlags(flags);
}
unsigned short DetourCrowdManager::get_include_flags(int filter_id) {
	const dtQueryFilter *filter = crowd->getFilter(filter_id);
	if (!filter)
		return 0U;
	return filter->getIncludeFlags();
}
unsigned short DetourCrowdManager::get_exclude_flags(int filter_id) {
	const dtQueryFilter *filter = crowd->getFilter(filter_id);
	if (!filter)
		return 0U;
	return filter->getExcludeFlags();
}
void DetourCrowdManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_navigation_mesh", "navmesh", "xform"),
			&DetourCrowdManager::set_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_navigation_mesh"),
			&DetourCrowdManager::get_navigation_mesh);
	ClassDB::bind_method(D_METHOD("add_agent", "agent", "mode", "signals"),
			&DetourCrowdManager::add_agent, DEFVAL(0),
			DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_agent", "agent"),
			&DetourCrowdManager::remove_agent);
	ClassDB::bind_method(D_METHOD("clear_agent_list"),
			&DetourCrowdManager::clear_agent_list);
	ClassDB::bind_method(D_METHOD("get_agent_obj", "id"),
			&DetourCrowdManager::get_agent_obj);
	ClassDB::bind_method(D_METHOD("get_agent_mode", "id"),
			&DetourCrowdManager::get_agent_mode);
	ClassDB::bind_method(D_METHOD("get_agent_count"),
			&DetourCrowdManager::get_agent_count);
	ClassDB::bind_method(D_METHOD("set_target", "position"),
			&DetourCrowdManager::set_target);
	ClassDB::bind_method(D_METHOD("set_velocity", "velocity"),
			&DetourCrowdManager::set_velocity);
	ClassDB::bind_method(D_METHOD("reset_target"),
			&DetourCrowdManager::reset_target);
	ClassDB::bind_method(D_METHOD("set_max_agents", "max_agents"),
			&DetourCrowdManager::set_max_agents);
	ClassDB::bind_method(D_METHOD("get_max_agents"),
			&DetourCrowdManager::get_max_agents);
	ClassDB::bind_method(D_METHOD("set_max_agent_radius", "max_agent_radius"),
			&DetourCrowdManager::set_max_agent_radius);
	ClassDB::bind_method(D_METHOD("get_max_agent_radius"),
			&DetourCrowdManager::get_max_agent_radius);
	ClassDB::bind_method(
			D_METHOD("set_area_cost", "filter_id", "area_id", "cost"),
			&DetourCrowdManager::set_area_cost);
	ClassDB::bind_method(D_METHOD("get_area_cost", "filter_id", "area_id"),
			&DetourCrowdManager::get_area_cost);
	ClassDB::bind_method(D_METHOD("set_include_flags", "filter_id", "flags"),
			&DetourCrowdManager::set_include_flags);
	ClassDB::bind_method(D_METHOD("get_include_flags", "filter_id"),
			&DetourCrowdManager::get_include_flags);
	ClassDB::bind_method(D_METHOD("set_exclude_flags", "filter_id", "flags"),
			&DetourCrowdManager::set_exclude_flags);
	ClassDB::bind_method(D_METHOD("get_exclude_flags", "filter_id"),
			&DetourCrowdManager::get_exclude_flags);
	ClassDB::bind_method(D_METHOD("set_agent_target_position", "id", "position"),
			&DetourCrowdManager::set_agent_target_position);
}

#if 0
// void DetourCrowdManager::agent_update_cb(dtCrowdAgent *ag, float dt)
//{
//}
bool Crowd::_set(const StringName &p_name, const Variant &p_value) {
	print_line("_set");
	if (!manager)
		return false;
	String name = p_name;
	print_line(String() + "setting " + name);
	if (name == "add_object") {
		if (p_value.get_type() == Variant::NIL)
			return false;
		String path = p_value;
		NodePath npname = p_value;
		print_line(String() + "setting spatial " + path);
		agent_paths.push_back(npname);
		modes.push_back(0);
		print_line(String() + "agent count: " + itos(agent_paths.size()));
		update_agent_list();
		_change_notify();
		return true;
	} else if (name == "nav_mesh") {
		if (p_value.get_type() == Variant::NODE_PATH) {
			NodePath path = p_value;
			DetourNavigationMeshInstance *nmi = (DetourNavigationMeshInstance*)get_node(path);
			if (nmi) {
				manager->set_navigation_mesh(nmi->get_navmesh());
				_change_notify();
				print_line("navmesh set from path");
				return true;
			}
		} else if (p_value.get_type() == Variant::OBJECT) {
			Ref<Resource> ov = p_value;
			if (ov.is_valid()) {
				manager->set_navigation_mesh((Ref<DetourNavigationMesh>)ov);
				print_line("navmesh set from resource");
				_change_notify();
				return true;
			}
		} else
			print_line(String() + "type: " + itos(p_value.get_type()));
		return false;
	} else if (name.begins_with("agents")) {
		int idx = name.get_slice("/", 1).to_int();
		String what = name.get_slice("/", 2);
		if (what == "path") {
			if (agent_paths.size() > idx) {
				NodePath path = p_value;
				agent_paths.write[idx] = path;
			} else {
				NodePath path = p_value;
				agent_paths.push_back(path);
			}
		} else if (what == "mode") {
			int mode = p_value;
			if (modes.size() > idx)
				modes.write[idx] = mode;
			else
				modes.push_back(mode);
		} else if (what == "remove") {
			bool rm = p_value;
			if (rm) {
				agent_paths.remove(idx);
				modes.remove(idx);
			}
		}
		_change_notify();
		update_agent_list();
		return true;
	}
	return false;
}
bool Crowd::_get(const StringName &p_name, Variant &r_ret) const {
	print_line("_get");
	if (!manager)
		return false;
	String name = p_name;
	if (name == "nav_mesh") {
		r_ret = manager->get_navigation_mesh();
		return true;
	} else if (name.begins_with("agents")) {
		int idx = name.get_slice("/", 1).to_int();
		String what = name.get_slice("/", 2);
		if (what == "path") {
			r_ret = agent_paths[idx];
			return true;
		} else if (what == "mode") {
			if (modes.size() > idx)
				r_ret = modes[idx];
			else
				r_ret = 0;
			return true;
		} else if (what == "remove") {
			r_ret = false;
			return true;
		}
	}
	return false;
}
void Crowd::_get_property_list(List<PropertyInfo> *p_list) const {
	print_line("_get_property_list");
	if (!manager)
		return;
	if (manager->get_agent_count() > 0) {
		for (int i = 0; i < agent_paths.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::NODE_PATH, "agents/" + itos(i) + "/path", PROPERTY_HINT_NONE, ""));
			p_list->push_back(PropertyInfo(Variant::INT, "agents/" + itos(i) + "/mode", PROPERTY_HINT_ENUM, "normal,signal"));
			p_list->push_back(PropertyInfo(Variant::BOOL, "agents/" + itos(i) + "/remove", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		}
	}
	p_list->push_back(PropertyInfo(Variant::NODE_PATH, "add_object", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Spatial"));
	if (!manager->get_navigation_mesh().is_valid())
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "nav_mesh", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "DetourNavigationMeshInstance"));
	else
		p_list->push_back(PropertyInfo(Variant::OBJECT, "nav_mesh", PROPERTY_HINT_RESOURCE_TYPE, "DetourNavigationMesh"));
}

Crowd::Crowd() :
	manager(0)
{
}
Crowd::~Crowd()
{
}

void Crowd::_notification(int p_what) {
	switch(p_what) {
		case NOTIFICATION_READY:
			print_line("a");
			if (!manager)
				return;
			else
				update_agent_list();
			break;
		case NOTIFICATION_ENTER_TREE:
			print_line("b");
			manager = Object::cast_to<DetourCrowdManager>(get_parent());
			if (!manager)
				return;
			else {
				print_line("manager set");
				update_agent_list();
			}
			break;
		case NOTIFICATION_EXIT_TREE:
			print_line("c");
			manager = NULL;
			break;
		case NOTIFICATION_PROCESS:
			float delta = get_process_delta_time();
			// update_crowd(delta);
			break;
	}
}
String Crowd::get_configuration_warning()
{
	String ret;
	print_line("get_configuration_warning");
	if (!is_inside_tree())
		return ret;
	if (!manager)
		ret += TTR("Incorrect instancing. ");
	if (!Object::cast_to<DetourCrowdManager>(get_parent()))
		ret += TTR("Should be parented to DetourCrowdManager. ");
	if (manager && !manager->get_navigation_mesh().is_valid())
		ret += TTR("No navmesh data are set to function. ");
	return ret;
}
void Crowd::update_agent_list()
{
	if (!is_inside_tree())
		return;
	if (!manager)
		return;
	print_line("update_agent_list");
	manager->clear_agent_list();
	print_line("update_agent_list 1");
	for (int i = 0; i < agent_paths.size(); i++) {
		print_line("update_agent_list 2: " + itos(i));
		if (String(agent_paths[i]).length() > 0) {
			print_line("update_agent_list 3: " + itos(i));
			Spatial *obj = (Spatial *)get_node(agent_paths[i]);
			print_line("update_agent_list 4: " + itos(i));
			if (obj) {
				print_line("update_agent_list 5: " + itos(i));
				manager->add_agent(obj, modes[i]);
				print_line("object added ok 0");
			}
			print_line("update_agent_list 6: " + itos(i));
		} else {
			print_line("update_agent_list 7: " + itos(i));
			manager->add_agent(NULL, modes[i]);
		}
	}
	print_line("update_agent_list done");
}
void Crowd::_bind_methods()
{
}
#endif
