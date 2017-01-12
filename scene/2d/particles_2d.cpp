/*************************************************************************/
/*  particles_2d.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "particles_2d.h"



void ParticleAttractor2D::_notification(int p_what) {

	switch(p_what)	 {

		case NOTIFICATION_ENTER_TREE: {

			_update_owner();

		} break;
		case NOTIFICATION_DRAW: {

			if (!get_tree()->is_editor_hint())
				return;

			Vector2 pv;
			float dr = MIN(disable_radius,radius);
			for(int i=0;i<=32;i++) {
				Vector2 v(Math::sin(i/32.0*Math_PI*2),Math::cos(i/32.0*Math_PI*2));
				if (i>0) {
					draw_line(pv*radius,v*radius,Color(0,0,0.5,0.9));
					if (dr>0) {
						draw_line(pv*dr,v*dr,Color(0.5,0,0.0,0.9));
					}
				}
				pv=v;
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (owner) {
				_set_owner(NULL);
			}

		} break;
	}
}

void ParticleAttractor2D::_owner_exited() {

	ERR_FAIL_COND(!owner);
	owner->attractors.erase(this);
	owner=NULL;
}

void ParticleAttractor2D::_update_owner() {

	if (!is_inside_tree() || !has_node(path)) {
		_set_owner(NULL);
		return;
	}

	Node *n = get_node(path);
	ERR_FAIL_COND(!n);
	Particles2D *pn = n->cast_to<Particles2D>();
	if (!pn) {
		_set_owner(NULL);
		return;
	}

	_set_owner(pn);
}

void ParticleAttractor2D::_set_owner(Particles2D* p_owner) {

	if (owner==p_owner)
		return;

	if (owner) {
		owner->disconnect("tree_exited",this,"_owner_exited");
		owner->attractors.erase(this);
		owner=NULL;
	}
	owner=p_owner;

	if (owner) {

		owner->connect("tree_exited",this,"_owner_exited",varray(),CONNECT_ONESHOT);
		owner->attractors.insert(this);
	}
}

void ParticleAttractor2D::_bind_methods() {

	ClassDB::bind_method(_MD("set_enabled","enabled"),&ParticleAttractor2D::set_enabled);
	ClassDB::bind_method(_MD("is_enabled"),&ParticleAttractor2D::is_enabled);

	ClassDB::bind_method(_MD("set_radius","radius"),&ParticleAttractor2D::set_radius);
	ClassDB::bind_method(_MD("get_radius"),&ParticleAttractor2D::get_radius);

	ClassDB::bind_method(_MD("set_disable_radius","radius"),&ParticleAttractor2D::set_disable_radius);
	ClassDB::bind_method(_MD("get_disable_radius"),&ParticleAttractor2D::get_disable_radius);

	ClassDB::bind_method(_MD("set_gravity","gravity"),&ParticleAttractor2D::set_gravity);
	ClassDB::bind_method(_MD("get_gravity"),&ParticleAttractor2D::get_gravity);

	ClassDB::bind_method(_MD("set_absorption","absorption"),&ParticleAttractor2D::set_absorption);
	ClassDB::bind_method(_MD("get_absorption"),&ParticleAttractor2D::get_absorption);

	ClassDB::bind_method(_MD("set_particles_path","path"),&ParticleAttractor2D::set_particles_path);
	ClassDB::bind_method(_MD("get_particles_path"),&ParticleAttractor2D::get_particles_path);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"enabled"),_SCS("set_enabled"),_SCS("is_enabled"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"radius",PROPERTY_HINT_RANGE,"0.1,16000,0.1"),_SCS("set_radius"),_SCS("get_radius"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"disable_radius",PROPERTY_HINT_RANGE,"0.1,16000,0.1"),_SCS("set_disable_radius"),_SCS("get_disable_radius"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"gravity",PROPERTY_HINT_RANGE,"-512,512,0.01"),_SCS("set_gravity"),_SCS("get_gravity"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"absorption",PROPERTY_HINT_RANGE,"0,512,0.01"),_SCS("set_absorption"),_SCS("get_absorption"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"particles_path",PROPERTY_HINT_RESOURCE_TYPE,"Particles2D"),_SCS("set_particles_path"),_SCS("get_particles_path"));



}


void ParticleAttractor2D::set_enabled(bool p_enabled) {

	enabled=p_enabled;
}

bool ParticleAttractor2D::is_enabled() const{

	return enabled;
}

void ParticleAttractor2D::set_radius(float p_radius) {

	radius = p_radius;
	update();
}

float ParticleAttractor2D::get_radius() const {

	return radius;
}

void ParticleAttractor2D::set_disable_radius(float p_disable_radius) {

	disable_radius = p_disable_radius;
	update();
}
float ParticleAttractor2D::get_disable_radius() const {

	return disable_radius;
}

void ParticleAttractor2D::set_gravity(float p_gravity) {

	gravity=p_gravity;

}
float ParticleAttractor2D::get_gravity() const {

	return gravity;
}

void ParticleAttractor2D::set_absorption(float p_absorption) {

	absorption=p_absorption;

}
float ParticleAttractor2D::get_absorption() const {

	return absorption;
}

void ParticleAttractor2D::set_particles_path(NodePath p_path) {

	path=p_path;
	_update_owner();
	update_configuration_warning();
}
NodePath ParticleAttractor2D::get_particles_path() const {

	return path;
}

String ParticleAttractor2D::get_configuration_warning() const {

	if (!has_node(path) || !get_node(path) || !get_node(path)->cast_to<Particles2D>()) {
		return TTR("Path property must point to a valid Particles2D node to work.");
	}

	return String();
}

ParticleAttractor2D::ParticleAttractor2D() {

	owner=NULL;
	radius=50;
	disable_radius=0;
	gravity=100;
	absorption=0;
	path=String("..");
	enabled=true;
}

/****************************************/

_FORCE_INLINE_ static float _rand_from_seed(uint32_t *seed) {

	uint32_t k;
	uint32_t s = (*seed);
	if (s == 0)
		s = 0x12345987;
	k = s / 127773;
	s = 16807 * (s - k * 127773) - 2836 * k;
	if (s < 0)
		s += 2147483647;
	(*seed) = s;

	float v=((float)((*seed) & 0xFFFFF))/(float)0xFFFFF;
	v=v*2.0-1.0;
	return v;
}

void Particles2D::_process_particles(float p_delta) {

	if (particles.size()==0 || lifetime==0)
		return;

	p_delta*=time_scale;

	float frame_time=p_delta;

	if (emit_timeout > 0) {
		time_to_live -= frame_time;
		if (time_to_live < 0) {

			emitting = false;
			_change_notify("config/emitting");
		};
	};

	float next_time = time+frame_time;

	if (next_time > lifetime)
		next_time=Math::fmod(next_time,lifetime);


	Particle *pdata=&particles[0];
	int particle_count=particles.size();
	Transform2D xform;
	if (!local_space)
		xform=get_global_transform();

	active_count=0;

	PoolVector<Point2>::Read r;
	int emission_point_count=0;
	if (emission_points.size()) {

		emission_point_count=emission_points.size();
		r=emission_points.read();
	}

	int attractor_count=0;
	AttractorCache *attractor_ptr=NULL;

	if (attractors.size()) {
		if (attractors.size()!=attractor_cache.size()) {
			attractor_cache.resize(attractors.size());
		}

		int idx=0;
		Transform2D m;
		if (local_space) {
			m= get_global_transform().affine_inverse();
		}
		for (Set<ParticleAttractor2D*>::Element *E=attractors.front();E;E=E->next()) {

			attractor_cache[idx].pos=m.xform( E->get()->get_global_position() );
			attractor_cache[idx].attractor=E->get();
			idx++;
		}

		attractor_ptr=attractor_cache.ptr();
		attractor_count=attractor_cache.size();
	}

	for(int i=0;i<particle_count;i++) {

		Particle &p=pdata[i];

		float restart_time = (i * lifetime / particle_count) * explosiveness;

		bool restart=false;

		if ( next_time < time ) {

			if (restart_time > time || restart_time < next_time )
				restart=true;

		} else if (restart_time > time && restart_time < next_time ) {
			restart=true;
		}

		if (restart) {


			if (emitting) {

				p.pos=emissor_offset;
				if (emission_point_count) {


					Vector2 ep = r[Math::rand()%emission_point_count];
					if (!local_space) {
						p.pos=xform.xform(p.pos+ep*extents);
					} else {
						p.pos+=ep*extents;
					}
				} else {
					if (!local_space) {
						p.pos=xform.xform(p.pos+Vector2(Math::random(-extents.x,extents.x),Math::random(-extents.y,extents.y)));
					} else {
						p.pos+=Vector2(Math::random(-extents.x,extents.x),Math::random(-extents.y,extents.y));
					}
				}
				p.seed=Math::rand() % 12345678;
				uint32_t rand_seed=p.seed*(i+1);

				float angle = Math::deg2rad(param[PARAM_DIRECTION]+_rand_from_seed(&rand_seed)*param[PARAM_SPREAD]);

				p.velocity=Vector2( Math::sin(angle), Math::cos(angle) );
				if (!local_space) {

					p.velocity = xform.basis_xform(p.velocity).normalized();
				}

				p.velocity*=param[PARAM_LINEAR_VELOCITY]+param[PARAM_LINEAR_VELOCITY]*_rand_from_seed(&rand_seed)*randomness[PARAM_LINEAR_VELOCITY];
				p.velocity+=initial_velocity;
				p.active=true;
				p.rot=Math::deg2rad(param[PARAM_INITIAL_ANGLE]+param[PARAM_INITIAL_ANGLE]*randomness[PARAM_INITIAL_ANGLE]*_rand_from_seed(&rand_seed));
				active_count++;

				p.frame=Math::fmod(param[PARAM_ANIM_INITIAL_POS]+randomness[PARAM_ANIM_INITIAL_POS]*_rand_from_seed(&rand_seed),1.0);


			} else {

				p.active=false;
			}

		} else {

			if (!p.active)
				continue;

			uint32_t rand_seed=p.seed*(i+1);

			Vector2 force;

			//apply gravity
			float gravity_dir = Math::deg2rad( param[PARAM_GRAVITY_DIRECTION]+180*randomness[PARAM_GRAVITY_DIRECTION]*_rand_from_seed(&rand_seed));
			force+=Vector2( Math::sin(gravity_dir), Math::cos(gravity_dir) ) * (param[PARAM_GRAVITY_STRENGTH]+param[PARAM_GRAVITY_STRENGTH]*randomness[PARAM_GRAVITY_STRENGTH]*_rand_from_seed(&rand_seed));
			//apply radial
			Vector2 rvec = (p.pos - emissor_offset).normalized();
			force+=rvec*(param[PARAM_RADIAL_ACCEL]+param[PARAM_RADIAL_ACCEL]*randomness[PARAM_RADIAL_ACCEL]*_rand_from_seed(&rand_seed));
			//apply orbit
			float orbitvel = (param[PARAM_ORBIT_VELOCITY]+param[PARAM_ORBIT_VELOCITY]*randomness[PARAM_ORBIT_VELOCITY]*_rand_from_seed(&rand_seed));
			if (orbitvel!=0) {
				Vector2 rel = p.pos - xform.elements[2];
				Transform2D rot(orbitvel*frame_time,Vector2());
				p.pos = rot.xform(rel) + xform.elements[2];

			}

			Vector2 tvec=rvec.tangent();
			force+=tvec*(param[PARAM_TANGENTIAL_ACCEL]+param[PARAM_TANGENTIAL_ACCEL]*randomness[PARAM_TANGENTIAL_ACCEL]*_rand_from_seed(&rand_seed));

			for(int j=0;j<attractor_count;j++) {

				Vector2 vec = (attractor_ptr[j].pos - p.pos);
				float vl = vec.length();

				if (!attractor_ptr[j].attractor->enabled ||  vl==0 || vl > attractor_ptr[j].attractor->radius)
					continue;



				force+=vec*attractor_ptr[j].attractor->gravity;
				float fvl = p.velocity.length();
				if (fvl && attractor_ptr[j].attractor->absorption) {
					Vector2 target = vec.normalized();
					p.velocity = p.velocity.normalized().linear_interpolate(target,MIN(frame_time*attractor_ptr[j].attractor->absorption,1))*fvl;
				}

				if (attractor_ptr[j].attractor->disable_radius && vl < attractor_ptr[j].attractor->disable_radius) {
					p.active=false;
				}
			}

			p.velocity+=force*frame_time;

			if (param[PARAM_DAMPING]) {
				float dmp = param[PARAM_DAMPING]+param[PARAM_DAMPING]*randomness[PARAM_DAMPING]*_rand_from_seed(&rand_seed);
				float v = p.velocity.length();
				v -= dmp * frame_time;
				if (v<=0) {
					p.velocity=Vector2();
				} else {
					p.velocity=p.velocity.normalized() * v;
				}

			}

			p.pos+=p.velocity*frame_time;
			p.rot+=Math::lerp(param[PARAM_SPIN_VELOCITY],param[PARAM_SPIN_VELOCITY]*randomness[PARAM_SPIN_VELOCITY]*_rand_from_seed(&rand_seed),randomness[PARAM_SPIN_VELOCITY])*frame_time;
			float anim_spd=param[PARAM_ANIM_SPEED_SCALE]+param[PARAM_ANIM_SPEED_SCALE]*randomness[PARAM_ANIM_SPEED_SCALE]*_rand_from_seed(&rand_seed);
			p.frame=Math::fposmod(p.frame+(frame_time/lifetime)*anim_spd,1.0);

			active_count++;

		}


	}



	time=Math::fmod( time+frame_time, lifetime );
	if (!emitting && active_count==0) {
		set_process(false);

	}

	update();


}


void Particles2D::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_PROCESS: {

			_process_particles( get_process_delta_time() );
		} break;

		case NOTIFICATION_ENTER_TREE: {

			float ppt=preprocess;
			while(ppt>0) {
				_process_particles(0.1);
				ppt-=0.1;
			}
		} break;
		case NOTIFICATION_DRAW: {


			if (particles.size()==0 || lifetime==0)
				return;

			RID ci=get_canvas_item();
			Size2 size(1,1);
			Point2 center;
			int total_frames=1;

			if (!texture.is_null()) {
				size=texture->get_size();
				size.x/=h_frames;
				size.y/=v_frames;
				total_frames=h_frames*v_frames;
			}


			float time_pos=(time/lifetime);

			Particle *pdata=&particles[0];
			int particle_count=particles.size();

			RID texrid;

			if (texture.is_valid())
				texrid = texture->get_rid();

			Transform2D invxform;
			if (!local_space)
				invxform=get_global_transform().affine_inverse();

			int start_particle = (int)(time * (float)particle_count / lifetime);

			for (int id=0;id<particle_count;++id) {
				int i = start_particle + id;
				if (i >= particle_count) {
					i -= particle_count;
				}

				Particle &p=pdata[i];
				if (!p.active)
					continue;

				float ptime = ((float)i / particle_count)*explosiveness;

				if (ptime<time_pos)
					ptime=time_pos-ptime;
				else
					ptime=(1.0-ptime)+time_pos;

				uint32_t rand_seed=p.seed*(i+1);

				Color color;

				if(color_ramp.is_valid())
				{
					color = color_ramp->get_color_at_offset(ptime);
				} else
				{
					color = default_color;
				}


				{
					float huerand=_rand_from_seed(&rand_seed);
					float huerot = param[PARAM_HUE_VARIATION] + randomness[PARAM_HUE_VARIATION] * huerand;

					if (Math::abs(huerot) > CMP_EPSILON) {

						float h=color.get_h();
						float s=color.get_s();
						float v=color.get_v();
						float a=color.a;
						//float preh=h;
						h+=huerot;
						h=Math::abs(Math::fposmod(h,1.0));
						//print_line("rand: "+rtos(randomness[PARAM_HUE_VARIATION])+" rand: "+rtos(huerand));
						//print_line(itos(i)+":hue: "+rtos(preh)+" + "+rtos(huerot)+" = "+rtos(h));
						color.set_hsv(h,s,v);
						color.a=a;
					}
				}

				float initial_size = param[PARAM_INITIAL_SIZE]+param[PARAM_INITIAL_SIZE]*_rand_from_seed(&rand_seed)*randomness[PARAM_INITIAL_SIZE];
				float final_size = param[PARAM_FINAL_SIZE]+param[PARAM_FINAL_SIZE]*_rand_from_seed(&rand_seed)*randomness[PARAM_FINAL_SIZE];

				float size_mult=initial_size*(1.0-ptime) + final_size*ptime;

				//Size2 rectsize=size * size_mult;
				//rectsize=rectsize.floor();

				//Rect2 r = Rect2(Vecto,rectsize);

				Transform2D xform;

				if (p.rot) {

					xform.set_rotation(p.rot);
					xform.translate(-size*size_mult/2.0);
					xform.elements[2]+=p.pos;
				} else {
					xform.elements[2]=-size*size_mult/2.0;
					xform.elements[2]+=p.pos;
				}

				if (!local_space) {
					xform = invxform * xform;
				}


				xform.scale_basis(Size2(size_mult,size_mult));


				VisualServer::get_singleton()->canvas_item_add_set_transform(ci,xform);


				if (texrid.is_valid()) {

					Rect2 src_rect;
					src_rect.size=size;

					if (total_frames>1) {
						int frame = Math::fast_ftoi(Math::floor(p.frame*total_frames)) % total_frames;
						src_rect.pos.x = size.x * (frame%h_frames);
						src_rect.pos.y = size.y * (frame/h_frames);
					}


					texture->draw_rect_region(ci,Rect2(Point2(),size),src_rect,color);
					//VisualServer::get_singleton()->canvas_item_add_texture_rect(ci,r,texrid,false,color);
				} else {
					VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(Point2(),size),color);

				}

			}


		} break;

	}

}

static const char* _particlesframe_property_names[Particles2D::PARAM_MAX]={
	"params/direction",
	"params/spread",
	"params/linear_velocity",
	"params/spin_velocity",
	"params/orbit_velocity",
	"params/gravity_direction",
	"params/gravity_strength",
	"params/radial_accel",
	"params/tangential_accel",
	"params/damping",
	"params/initial_angle",
	"params/initial_size",
	"params/final_size",
	"params/hue_variation",
	"params/anim_speed_scale",
	"params/anim_initial_pos",
};

static const char* _particlesframe_property_rnames[Particles2D::PARAM_MAX]={
	"randomness/direction",
	"randomness/spread",
	"randomness/linear_velocity",
	"randomness/spin_velocity",
	"randomness/orbit_velocity",
	"randomness/gravity_direction",
	"randomness/gravity_strength",
	"randomness/radial_accel",
	"randomness/tangential_accel",
	"randomness/damping",
	"randomness/initial_angle",
	"randomness/initial_size",
	"randomness/final_size",
	"randomness/hue_variation",
	"randomness/anim_speed_scale",
	"randomness/anim_initial_pos",
};

static const char* _particlesframe_property_ranges[Particles2D::PARAM_MAX]={
	"0,360,0.01",
	"0,180,0.01",
	"-1024,1024,0.01",
	"-1024,1024,0.01",
	"-1024,1024,0.01",
	"0,360,0.01",
	"0,1024,0.01",
	"-128,128,0.01",
	"-128,128,0.01",
	"0,1024,0.001",
	"0,360,0.01",
	"0,1024,0.01",
	"0,1024,0.01",
	"0,1,0.01",
	"0,128,0.01",
	"0,1,0.01",
};


void Particles2D::set_emitting(bool p_emitting) {

	if (emitting==p_emitting)
		return;

	if (p_emitting) {

		if (active_count==0)
			time=0;
		set_process(true);
		time_to_live = emit_timeout;
	};
	emitting=p_emitting;
	_change_notify("config/emitting");
}

bool Particles2D::is_emitting() const {

	return emitting;
}

void Particles2D::set_amount(int p_amount) {

	ERR_FAIL_INDEX(p_amount,1024+1);

	particles.resize(p_amount);
}
int Particles2D::get_amount() const {

	return particles.size();
}

void Particles2D::set_emit_timeout(float p_timeout) {

	emit_timeout = p_timeout;
	time_to_live = p_timeout;
};

float Particles2D::get_emit_timeout() const {

	return emit_timeout;
};

void Particles2D::set_lifetime(float p_lifetime) {

	ERR_FAIL_INDEX(p_lifetime,3600+1);

	lifetime=p_lifetime;
}
float Particles2D::get_lifetime() const {

	return lifetime;
}

void Particles2D::set_time_scale(float p_time_scale) {

	time_scale=p_time_scale;
}
float Particles2D::get_time_scale() const {

	return time_scale;
}

void Particles2D::set_pre_process_time(float p_pre_process_time) {

	preprocess=p_pre_process_time;
}

float Particles2D::get_pre_process_time() const{

	return preprocess;
}


void Particles2D::set_param(Parameter p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,PARAM_MAX);
	param[p_param]=p_value;
}
float Particles2D::get_param(Parameter p_param) const {

	ERR_FAIL_INDEX_V(p_param,PARAM_MAX,0);
	return param[p_param];
}

void Particles2D::set_randomness(Parameter p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,PARAM_MAX);
	randomness[p_param]=p_value;

}
float Particles2D::get_randomness(Parameter p_param) const  {

	ERR_FAIL_INDEX_V(p_param,PARAM_MAX,0);
	return randomness[p_param];

}

void Particles2D::set_texture(const Ref<Texture>& p_texture) {

	texture=p_texture;
}

Ref<Texture> Particles2D::get_texture() const {

	return texture;
}

void Particles2D::set_color(const Color& p_color) {

	default_color = p_color;
}

Color Particles2D::get_color() const {

	return default_color;
}


void Particles2D::set_color_ramp(const Ref<ColorRamp>& p_color_ramp) {

	color_ramp=p_color_ramp;
}

Ref<ColorRamp> Particles2D::get_color_ramp() const {

	return color_ramp;
}

void Particles2D::set_emissor_offset(const Point2& p_offset) {

	emissor_offset=p_offset;
}

Point2 Particles2D::get_emissor_offset() const {

	return emissor_offset;
}


void Particles2D::set_use_local_space(bool p_use) {

	local_space=p_use;
}

bool Particles2D::is_using_local_space() const {

	return local_space;
}

//Deprecated. Converts color phases to color ramp
void Particles2D::set_color_phases(int p_phases) {

	//Create color ramp if we have 2 or more phases.
	//Otherwise first phase phase will be assigned to default color.
	if(p_phases > 1 && color_ramp.is_null())
	{
		color_ramp = Ref<ColorRamp>(memnew (ColorRamp()));
	}
	if(color_ramp.is_valid())
	{
		color_ramp->get_points().resize(p_phases);
	}
}

//Deprecated.
int Particles2D::get_color_phases() const {

	if(color_ramp.is_valid())
	{
		return color_ramp->get_points_count();
	}
	return 0;
}

//Deprecated. Converts color phases to color ramp
void Particles2D::set_color_phase_color(int p_phase,const Color& p_color) {

	ERR_FAIL_INDEX(p_phase,MAX_COLOR_PHASES);
	if(color_ramp.is_valid())
	{
		if(color_ramp->get_points_count() > p_phase)
			color_ramp->set_color(p_phase, p_color);
	} else
	{
		if(p_phase == 0)
			default_color = p_color;
	}
}

//Deprecated.
Color Particles2D::get_color_phase_color(int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase,MAX_COLOR_PHASES,Color());
	if(color_ramp.is_valid())
	{
		return color_ramp->get_color(p_phase);
	}
	return Color(0,0,0,1);
}

//Deprecated. Converts color phases to color ramp
void Particles2D::set_color_phase_pos(int p_phase,float p_pos) {
	ERR_FAIL_INDEX(p_phase,MAX_COLOR_PHASES);
	ERR_FAIL_COND(p_pos<0.0 || p_pos>1.0);
	if(color_ramp.is_valid() && color_ramp->get_points_count() > p_phase)
	{
		return color_ramp->set_offset(p_phase, p_pos);
	}
}

//Deprecated.
float Particles2D::get_color_phase_pos(int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase,MAX_COLOR_PHASES,0);
	if(color_ramp.is_valid())
	{
		return color_ramp->get_offset(p_phase);
	}
	return 0;
}

void Particles2D::set_emission_half_extents(const Vector2& p_extents) {

	extents=p_extents;
}

Vector2 Particles2D::get_emission_half_extents() const {

	return extents;
}

void Particles2D::testee(int a, int b, int c, int d, int e) {

	print_line(itos(a));
	print_line(itos(b));
	print_line(itos(c));
	print_line(itos(d));
	print_line(itos(e));
}

void Particles2D::set_initial_velocity(const Vector2& p_velocity) {


	initial_velocity=p_velocity;
}
Vector2 Particles2D::get_initial_velocity() const{

	return initial_velocity;
}


void Particles2D::pre_process(float p_delta) {

	_process_particles(p_delta);
}


void Particles2D::set_explosiveness(float p_value) {

	explosiveness=p_value;
}

float Particles2D::get_explosiveness() const{

	return explosiveness;
}

void Particles2D::set_flip_h(bool p_flip) {

	flip_h=p_flip;
}

bool Particles2D::is_flipped_h() const{

	return flip_h;
}

void Particles2D::set_flip_v(bool p_flip){

	flip_v=p_flip;
}
bool Particles2D::is_flipped_v() const{

	return flip_v;
}

void Particles2D::set_h_frames(int p_frames) {

	ERR_FAIL_COND(p_frames<1);
	h_frames=p_frames;
}

int Particles2D::get_h_frames() const{

	return h_frames;
}

void Particles2D::set_v_frames(int p_frames){

	ERR_FAIL_COND(p_frames<1);
	v_frames=p_frames;
}
int Particles2D::get_v_frames() const{

	return v_frames;
}



void Particles2D::set_emission_points(const PoolVector<Vector2>& p_points) {

	emission_points=p_points;
}

PoolVector<Vector2> Particles2D::get_emission_points() const{

	return emission_points;
}

void Particles2D::reset() {

	for(int i=0;i<particles.size();i++) {
		particles[i].active=false;
	}
	time=0;
	active_count=0;
}

void Particles2D::_bind_methods() {

	ClassDB::bind_method(_MD("set_emitting","active"),&Particles2D::set_emitting);
	ClassDB::bind_method(_MD("is_emitting"),&Particles2D::is_emitting);

	ClassDB::bind_method(_MD("set_amount","amount"),&Particles2D::set_amount);
	ClassDB::bind_method(_MD("get_amount"),&Particles2D::get_amount);

	ClassDB::bind_method(_MD("set_lifetime","lifetime"),&Particles2D::set_lifetime);
	ClassDB::bind_method(_MD("get_lifetime"),&Particles2D::get_lifetime);

	ClassDB::bind_method(_MD("set_time_scale","time_scale"),&Particles2D::set_time_scale);
	ClassDB::bind_method(_MD("get_time_scale"),&Particles2D::get_time_scale);

	ClassDB::bind_method(_MD("set_pre_process_time","time"),&Particles2D::set_pre_process_time);
	ClassDB::bind_method(_MD("get_pre_process_time"),&Particles2D::get_pre_process_time);

	ClassDB::bind_method(_MD("set_emit_timeout","value"),&Particles2D::set_emit_timeout);
	ClassDB::bind_method(_MD("get_emit_timeout"),&Particles2D::get_emit_timeout);

	ClassDB::bind_method(_MD("set_param","param","value"),&Particles2D::set_param);
	ClassDB::bind_method(_MD("get_param","param"),&Particles2D::get_param);

	ClassDB::bind_method(_MD("set_randomness","param","value"),&Particles2D::set_randomness);
	ClassDB::bind_method(_MD("get_randomness","param"),&Particles2D::get_randomness);

	ClassDB::bind_method(_MD("set_texture:Texture","texture"),&Particles2D::set_texture);
	ClassDB::bind_method(_MD("get_texture:Texture"),&Particles2D::get_texture);

	ClassDB::bind_method(_MD("set_color","color"),&Particles2D::set_color);
	ClassDB::bind_method(_MD("get_color"),&Particles2D::get_color);

	ClassDB::bind_method(_MD("set_color_ramp:ColorRamp","color_ramp"),&Particles2D::set_color_ramp);
	ClassDB::bind_method(_MD("get_color_ramp:ColorRamp"),&Particles2D::get_color_ramp);

	ClassDB::bind_method(_MD("set_emissor_offset","offset"),&Particles2D::set_emissor_offset);
	ClassDB::bind_method(_MD("get_emissor_offset"),&Particles2D::get_emissor_offset);

	ClassDB::bind_method(_MD("set_flip_h","enable"),&Particles2D::set_flip_h);
	ClassDB::bind_method(_MD("is_flipped_h"),&Particles2D::is_flipped_h);

	ClassDB::bind_method(_MD("set_flip_v","enable"),&Particles2D::set_flip_v);
	ClassDB::bind_method(_MD("is_flipped_v"),&Particles2D::is_flipped_v);

	ClassDB::bind_method(_MD("set_h_frames","enable"),&Particles2D::set_h_frames);
	ClassDB::bind_method(_MD("get_h_frames"),&Particles2D::get_h_frames);

	ClassDB::bind_method(_MD("set_v_frames","enable"),&Particles2D::set_v_frames);
	ClassDB::bind_method(_MD("get_v_frames"),&Particles2D::get_v_frames);

	ClassDB::bind_method(_MD("set_emission_half_extents","extents"),&Particles2D::set_emission_half_extents);
	ClassDB::bind_method(_MD("get_emission_half_extents"),&Particles2D::get_emission_half_extents);

	ClassDB::bind_method(_MD("set_color_phases","phases"),&Particles2D::set_color_phases);
	ClassDB::bind_method(_MD("get_color_phases"),&Particles2D::get_color_phases);

	ClassDB::bind_method(_MD("set_color_phase_color","phase","color"),&Particles2D::set_color_phase_color);
	ClassDB::bind_method(_MD("get_color_phase_color","phase"),&Particles2D::get_color_phase_color);

	ClassDB::bind_method(_MD("set_color_phase_pos","phase","pos"),&Particles2D::set_color_phase_pos);
	ClassDB::bind_method(_MD("get_color_phase_pos","phase"),&Particles2D::get_color_phase_pos);

	ClassDB::bind_method(_MD("pre_process","time"),&Particles2D::pre_process);
	ClassDB::bind_method(_MD("reset"),&Particles2D::reset);

	ClassDB::bind_method(_MD("set_use_local_space","enable"),&Particles2D::set_use_local_space);
	ClassDB::bind_method(_MD("is_using_local_space"),&Particles2D::is_using_local_space);

	ClassDB::bind_method(_MD("set_initial_velocity","velocity"),&Particles2D::set_initial_velocity);
	ClassDB::bind_method(_MD("get_initial_velocity"),&Particles2D::get_initial_velocity);

	ClassDB::bind_method(_MD("set_explosiveness","amount"),&Particles2D::set_explosiveness);
	ClassDB::bind_method(_MD("get_explosiveness"),&Particles2D::get_explosiveness);

	ClassDB::bind_method(_MD("set_emission_points","points"),&Particles2D::set_emission_points);
	ClassDB::bind_method(_MD("get_emission_points"),&Particles2D::get_emission_points);

	ADD_PROPERTY(PropertyInfo(Variant::INT,"config/amount",PROPERTY_HINT_EXP_RANGE,"1,1024"),_SCS("set_amount"),_SCS("get_amount") );
	ADD_PROPERTY(PropertyInfo(Variant::REAL,"config/lifetime",PROPERTY_HINT_EXP_RANGE,"0.1,3600,0.1"),_SCS("set_lifetime"),_SCS("get_lifetime") );
	ADD_PROPERTYNO(PropertyInfo(Variant::REAL,"config/time_scale",PROPERTY_HINT_EXP_RANGE,"0.01,128,0.01"),_SCS("set_time_scale"),_SCS("get_time_scale") );
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL,"config/preprocess",PROPERTY_HINT_EXP_RANGE,"0.1,3600,0.1"),_SCS("set_pre_process_time"),_SCS("get_pre_process_time") );
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL,"config/emit_timeout",PROPERTY_HINT_RANGE,"0,3600,0.1"),_SCS("set_emit_timeout"),_SCS("get_emit_timeout") );
	ADD_PROPERTYNO(PropertyInfo(Variant::BOOL,"config/emitting"),_SCS("set_emitting"),_SCS("is_emitting") );
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2,"config/offset"),_SCS("set_emissor_offset"),_SCS("get_emissor_offset"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2,"config/half_extents"),_SCS("set_emission_half_extents"),_SCS("get_emission_half_extents"));
	ADD_PROPERTYNO(PropertyInfo(Variant::BOOL,"config/local_space"),_SCS("set_use_local_space"),_SCS("is_using_local_space"));
	ADD_PROPERTYNO(PropertyInfo(Variant::REAL,"config/explosiveness",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_explosiveness"),_SCS("get_explosiveness"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL,"config/flip_h"),_SCS("set_flip_h"),_SCS("is_flipped_h"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL,"config/flip_v"),_SCS("set_flip_v"),_SCS("is_flipped_v"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT,"config/texture",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_texture"),_SCS("get_texture"));
	ADD_PROPERTYNO(PropertyInfo(Variant::INT,"config/h_frames",PROPERTY_HINT_RANGE,"1,512,1"),_SCS("set_h_frames"),_SCS("get_h_frames"));
	ADD_PROPERTYNO(PropertyInfo(Variant::INT,"config/v_frames",PROPERTY_HINT_RANGE,"1,512,1"),_SCS("set_v_frames"),_SCS("get_v_frames"));


	for(int i=0;i<PARAM_MAX;i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::REAL,_particlesframe_property_names[i],PROPERTY_HINT_RANGE,_particlesframe_property_ranges[i]),_SCS("set_param"),_SCS("get_param"),i);
	}

	for(int i=0;i<PARAM_MAX;i++) {
		ADD_PROPERTYINZ(PropertyInfo(Variant::REAL,_particlesframe_property_rnames[i],PROPERTY_HINT_RANGE,"-1,1,0.01"),_SCS("set_randomness"),_SCS("get_randomness"),i);
	}

	ADD_PROPERTYNZ( PropertyInfo( Variant::INT, "color_phases/count",PROPERTY_HINT_RANGE,"0,4,1", 0), _SCS("set_color_phases"), _SCS("get_color_phases"));

	//Backward compatibility. They will be converted to color ramp
	for(int i=0;i<MAX_COLOR_PHASES;i++) {
		String phase="phase_"+itos(i)+"/";
		ADD_PROPERTYI( PropertyInfo( Variant::REAL, phase+"pos", PROPERTY_HINT_RANGE,"0,1,0.01", 0),_SCS("set_color_phase_pos"),_SCS("get_color_phase_pos"),i );
		ADD_PROPERTYI( PropertyInfo( Variant::COLOR, phase+"color", PROPERTY_HINT_NONE, "", 0),_SCS("set_color_phase_color"),_SCS("get_color_phase_color"),i );
	}

	ADD_PROPERTYNO(PropertyInfo(Variant::COLOR, "color/color"),_SCS("set_color"),_SCS("get_color"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT,"color/color_ramp",PROPERTY_HINT_RESOURCE_TYPE,"ColorRamp"),_SCS("set_color_ramp"),_SCS("get_color_ramp"));

	ADD_PROPERTYNZ(PropertyInfo(Variant::POOL_VECTOR2_ARRAY,"emission_points",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("set_emission_points"),_SCS("get_emission_points"));

	BIND_CONSTANT( PARAM_DIRECTION );
	BIND_CONSTANT( PARAM_SPREAD );
	BIND_CONSTANT( PARAM_LINEAR_VELOCITY );
	BIND_CONSTANT( PARAM_SPIN_VELOCITY );
	BIND_CONSTANT( PARAM_ORBIT_VELOCITY );
	BIND_CONSTANT( PARAM_GRAVITY_DIRECTION );
	BIND_CONSTANT( PARAM_GRAVITY_STRENGTH );
	BIND_CONSTANT( PARAM_RADIAL_ACCEL );
	BIND_CONSTANT( PARAM_TANGENTIAL_ACCEL );
	BIND_CONSTANT( PARAM_DAMPING );
	BIND_CONSTANT( PARAM_INITIAL_ANGLE );
	BIND_CONSTANT( PARAM_INITIAL_SIZE );
	BIND_CONSTANT( PARAM_FINAL_SIZE );
	BIND_CONSTANT( PARAM_HUE_VARIATION );
	BIND_CONSTANT( PARAM_ANIM_SPEED_SCALE );
	BIND_CONSTANT( PARAM_ANIM_INITIAL_POS );
	BIND_CONSTANT( PARAM_MAX );

	BIND_CONSTANT( MAX_COLOR_PHASES );

}

Particles2D::Particles2D() {

	for(int i=0;i<PARAM_MAX;i++) {

		param[i]=0;
		randomness[i]=0;
	}


	set_param(PARAM_SPREAD,10);
	set_param(PARAM_LINEAR_VELOCITY,20);
	set_param(PARAM_GRAVITY_STRENGTH,9.8);
	set_param(PARAM_RADIAL_ACCEL,0);
	set_param(PARAM_TANGENTIAL_ACCEL,0);
	set_param(PARAM_INITIAL_ANGLE,0.0);
	set_param(PARAM_INITIAL_SIZE,1.0);
	set_param(PARAM_FINAL_SIZE,1.0);
	set_param(PARAM_ANIM_SPEED_SCALE,1.0);

	set_color(Color(1,1,1,1));

	time=0;
	lifetime=2;
	emitting=false;
	particles.resize(32);
	active_count=-1;
	set_emitting(true);
	local_space=true;
	preprocess=0;
	time_scale=1.0;


	flip_h=false;
	flip_v=false;

	v_frames=1;
	h_frames=1;

	emit_timeout = 0;
	time_to_live = 0;
	explosiveness=1.0;
}
