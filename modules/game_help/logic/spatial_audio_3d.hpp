
#ifndef SPATIAL_AUDIO_3D_H
#define SPATIAL_AUDIO_3D_H
#include "scene/3d/audio_stream_player_3d.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/3d/physics/ray_cast_3d.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/main/viewport.h"
#include "scene/animation/tween.h"
#include "scene/3d/camera_3d.h"



//@icon("icon.png")

//# Adds spatial acoustics with calculated delays, reverb and occlusion effects based on level geometry.
//#
//# It consists of three components:
//#   * Soundsource
//#   * Reverber
//#   * Soundplayer
//#
//# The *Soundsource* is the original audio emitter, it has a delay based on the player's distance to it.
//# Soundsource also shoots raycasts in 8 different directions + up to determine where to create the Reverbers,
//# and it measures the room size using additional measurement raycasts.
//#
//# The *Reverber* is a sound that is reflected on a wall, it has a delay and reverb effect, based on player's distance and room size.
//#
//# The *Soundplayer* is the entity that actually plays the audio. Why do I need an additional Soundplayer and not playing directly on Soundsource and Reverber?
//# During development, I figured that setting the delay while playing sound generates audible cracks.
//# To mitigate this, each Soundsource has two *Soundplayers*; one active playing a sound, and one in standby where I set delays.
//# Every 10 meters, Soundsource does a crossfade between the active and inactive Soundplayer.
//#
//# A similar problem arises with the Reverber: Imagine it plays a sound in a big room with long reverb.
//# Suddenly, you exit the room. The Reverber with the long reverb is repositioned 1m at the door, and the still playing reverb suddenly abrupts.
//# To mitigate this, each Reverber also has two Soundplayers; one active playing a sound, and one in standby.
//# If the player moves around and the reverber would be placed more than 10m away to it's former position, the standby Soundplayer is placed instead,
//# letting the former reverber hall and fade out.
class Soundsource;
class Soundplayer;
class SpatialAudio3D : public AudioStreamPlayer3D {
	GDCLASS(SpatialAudio3D, AudioStreamPlayer3D);
public:

public:
	bool reverb_enabled = true;//# Enable reverb-effects with all its computations.[br][br]Disable if you just need it for occlusion detection, for example fading ambient rain when entering a house.
	double reverb_volume_db = -6;//# Maximum volume of the reverbs.
	double reverb_fadeintime = 2.0;//# Fade-in time when a reverb is added at a wall.[br][br]The default of 2s is a good value where changes in environment merge well into the overall sound mix.[br]If you have small areas with little changes of distances, set this value to low (0.2s) so that repositioned reverbs don't get stuck for too long in a position. In a large area, when the raycast suddenly hits a close wall, the sudden appearing reverb can sound unnatural.[br][br]Set to -1 for dynamic fade time (based on wetness).
	double reverb_fadeouttime = 2.0;//# Fade-out time when a reverb is removed.[br][br]The default of 2s is a good value where changes in environment merge well into the overall sound mix.[br]If you have small areas with little changes of distances, set this value to low (0.2s) so that repositioned reverbs don't get stuck for too long in a position.[br][br]Set to -1 for dynamic fade time (based on wetness).
	int occlusion_lp_cutoff = 600;//# Frequency when soundsource is occluded behind walls.
	double occlusion_fadetime = 0.5;//# Fadetime when occlusion changes.
	int bass_proximity = 50;//# The closer you are to a wall, the more bass you will hear.[br][br]The effect starts at this value (distance to the wall).[br][br]0 to disable.

	int max_raycast_distance = 100;//# Maximum distance for the reverb raycasts.
	int collision_mask = 1;//# Mask for the raycast where to add reverb.
	double roomsize_multiplicator = 6.0;//# How much hall to add compared to room size.[br]Sometimes you have a small room but you need a long reverb.
	int speed_of_sound = 340;//# How fast sound travels through the air.

	int audiophysics_ticks = 10;//# The number of audio physics calculations per second.[br]Tied to _physics_process().[br][br]Use 30-60 for fast moving sounds like a motorcycle passing by.[br][br]Default: 10.
	bool loop = false;//# Loop audio indefinitely.
	bool shut_up = false;//# Mute output.
	bool debug = false;//# Visualize raycasts, measurement rays and reverb-audioplayers.

	Camera3D* player_camera = nullptr;// store reference to camera so that global_position is always up to date
	LocalVector<Vector3> raycasts_coords;

	// this is the soundsource audioplayer.
	Soundsource* soundsource = nullptr;

	// internal vars
	bool do_update = false;
	int tick_interval;
	int _tick_counter = -1;// initialize the counter with enough time for the engine to initialize
	Dictionary _debug;
	class Debugsphere* debugsphere;
	Ref<Tween> fade_tween;
	Ref<Tween> lowpass_tween;
	double xfadetime = 1.0;

public:
	enum fx { delay, reverb, reverb_hipass, lowpass };

	void _ready() override;

	void physics_process(double _delta) override;

	virtual void do_play();

	virtual void do_stop();

	virtual void do_set_stream(Ref<AudioStream> sound);

	RayCast3D* create_raycast(StringName name_p, const Vector3& target_position);

	LocalVector<RayCast3D*> create_raycast_sector(int start_angle = 0, double width_factor = 1.5, int bearing_raycount = 20, int heading_count = 7);

	Soundplayer* create_soundplayer(String name_p, bool with_reverb_fx = true);

	void create_audiobus(StringName bus_name, int vol_db = 0);

	//# sets the volume.
	//# if you provide fadetime, it will fade to volume using an internal tweener.
	//# if you need to fade more than one value at a time in this player, you can provide a tweener.
	void remove_audiobus(String bus_name);

	// this is a little helper method for tweening.
	// tween_method always tweens the first value.
	// but the first value in set_bus_volume_db() is bus index, so we provide volume as the first value for tween_method.
	void set_audiobus_volume(StringName bus_name, float vol_db, double fadetime = 0, Ref<Tween> tweenvar = Ref<Tween>());

	void tweensetvol(float vol, int bus_index);

	void add_audioeffect(StringName bus_name, fx effect_type);

	void set_audioeffect(StringName bus_name, fx effect_type, Dictionary params);

	void toggle_audioeffect(StringName bus_name, fx effect_type, bool enabled);

	// Soundsource:
	// responsible for: positions and distances
	// room size and wetness (calc_reverb)
	// performs calculations over all raycasts/reverbers every X (0.5) seconds
	// reverber.update_position()
	// spawns reverbers
	float calculate_delay(double distance);

	void dump_debug();

	void print_r(Variant obj);
	void set_reverb_enabled(bool value);
	bool get_reverb_enabled();
	void set_reverb_volume_db(double value);
	double get_reverb_volume_db();
	void set_reverb_fadeintime(double value);
	double get_reverb_fadeintime();
	void set_reverb_fadeouttime(double value);
	double get_reverb_fadeouttime();
	void set_occlusion_lp_cutoff(int value);
	int get_occlusion_lp_cutoff();
	void set_occlusion_fadetime(double value);
	double get_occlusion_fadetime();
	void set_bass_proximity(int value);
	int get_bass_proximity();
	void set_max_raycast_distance(int value);
	int get_max_raycast_distance();
	void set_collision_mask(int value);
	int get_collision_mask();
	void set_roomsize_multiplicator(double value);
	double get_roomsize_multiplicator();
	void set_speed_of_sound(int value);
	int get_speed_of_sound();
	void set_audiophysics_ticks(int value);
	int get_audiophysics_ticks();
	void set_loop(bool value);
	bool get_loop();
	void set_shut_up(bool value);
	bool get_shut_up();
	void set_debug(bool value);
	bool get_debug();

	static void _bind_methods();
};


class Reverber : public SpatialAudio3D {
	GDCLASS(Reverber, SpatialAudio3D);
public:

	int raycast_index;
	double distance_to_soundsource;
	double distance_to_player;
	class Soundplayer* soundplayer_active;
	class Soundplayer* soundplayer_standby;

public:
	void _ready() override;

	// don't inherit this from SpatialAudio3D
	void _exit_tree();

	void physics_process(double _delta) override;

	void do_play() override;

	void do_stop() override;

	// Soundplayer is the actual player that plays a sound.
	// This is because the Soundsource as well as the reverbers need two players and fade between them,
	// e.g. for letting the reverb hall playing out and not stopping abruptly, or for crossfading between different delays.
	// responsible for setup and teardown audio bus, volume, effect parameters and occlusion detection.
	void update_position(Vector3 target_position, bool colliding);

	static void _bind_methods();
};


class Soundsource : public SpatialAudio3D {
	GDCLASS(Soundsource, SpatialAudio3D);
public:

protected:
	LocalVector<RayCast3D*> raycasts;
	LocalVector<Reverber*> reverbers;
	LocalVector<RayCast3D*> measurement_rays;
	Array distances;
	double distance_to_player;
	double distance_to_player_since_last_delay_update;
	int delay_ms;

public:
	void set_delay_ms(int v);
	int get_delay_ms();

public:
	double room_size = 0.0;
	double wetness = 1.0;
	Soundplayer* soundplayer_active = nullptr;
	Soundplayer* soundplayer_standby = nullptr;
	int _playing_since;

public:
	void _ready() override;

	void physics_process(double _delta) override;

	void _exit_tree();

	Reverber* create_reverber(StringName name_p, int raycast_index);

	void do_set_stream(Ref<AudioStream> s) override;

	void do_play() override;

	void do_stop() override;

	void update_run();

	void calculate_all_distances();

// Reverber:
// responsible for playing and positioning Soundplayers
	void calculate_reverb();

	static void _bind_methods();
};

class Soundplayer : public SpatialAudio3D {
	GDCLASS(Soundplayer, SpatialAudio3D);
public:

protected:
	Dictionary ds = Dictionary {/* initializer lists are unsupported */ {0, "active"},{1, "inactive"},{2, "fading_to_active"},{3, "fading_to_inactive"}, };

public:
	enum ss {active, inactive, fading_to_active, fading_to_inactive};

public:
	bool with_reverb_fx;
	ss state;

public:
	void set_state(ss v);
	ss get_state();

public:
	double distance_to_soundsource;
	double distance_to_player;
	Vector3 target_position;

public:
	void set_target_position(Vector3 v);
	Vector3 get_target_position();

protected:
	int delay_ms;

public:
	void set_delay_ms(int v);
	int get_delay_ms();

protected:
	int delay_updated_at;
	double room_size;

public:
	void set_room_size(double v);
	double get_room_size();

protected:
	double wetness;

public:
	void set_wetness(double v);
	double get_wetness();

protected:
	double proximity_volume;

public:
	void set_proximity_volume(double v);
	double get_proximity_volume();

protected:
	double proximity_bass;

public:
	void set_proximity_bass(double v);
	double get_proximity_bass();

protected:
	int lp_cutoff;

public:
	void set_lp_cutoff(int v);
	int get_lp_cutoff();

protected:
	RayCast3D* occlusion_raycast = nullptr;
	String audiobus_name;

public:
	void _ready() override;

	void _on_finished();

	void physics_process(double _delta) override;

	void _exit_tree();

	void do_play() override;

	void do_stop() override;

	void set_active(double fadetime = 0, Tween::EaseType easing = Tween::EaseType::EASE_OUT, Tween::TransitionType transition = Tween::TransitionType::TRANS_QUART);

	void set_inactive(double fadetime = 0, Tween::EaseType easing = Tween::EaseType::EASE_IN, Tween::TransitionType transition = Tween::TransitionType::TRANS_QUINT);

// turn down reverb volume by [reduction] dB when closer to the wall
	void update_effect_params();

	int calculate_proximity_volume();

	double calculate_proximity_bass();

	int calculate_occlusion_lowpass();

	static void _bind_methods();
};
class Debugsphere : public Node3D {
	GDCLASS(Debugsphere, Node3D);
public:

public:
	Color color = "00f";
	double size = 0.5;
	int max_raycast_distance;
	Label3D* label = nullptr;
	String line1;
	String line2;
	String line3;
	String line4;
	Vector3 label_offset;

public:
	void _ready() override;

	void update_label();

	static void _bind_methods();
};

class Debugray : public MeshInstance3D {
	GDCLASS(Debugray, MeshInstance3D);
public:

protected:
	Ref<ImmediateMesh> immediate_mesh;
	Ref<ORMMaterial3D> material;

public:
	void _ready() override;

	void draw(Vector3 pos1, Vector3 pos2, Color color);

	void clear();

	static void _bind_methods();
};


VARIANT_ENUM_CAST(Soundplayer::ss)
VARIANT_ENUM_CAST(SpatialAudio3D::fx)

#endif // SPATIAL_AUDIO_3D_H
