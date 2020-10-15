/*************************************************************************/
/*  game_center.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef GAME_CENTER_ENABLED

#ifndef GAME_CENTER_H
#define GAME_CENTER_H

#include "core/class_db.h"

class GameCenter : public Object {
	GDCLASS(GameCenter, Object);

	static GameCenter *instance;
	static void _bind_methods();

	List<Variant> pending_events;

	bool authenticated;

	void return_connect_error(const char *p_error_description);

public:
	Error authenticate();
	bool is_authenticated();

	Error post_score(Dictionary p_score);
	Error award_achievement(Dictionary p_params);
	void reset_achievements();
	void request_achievements();
	void request_achievement_descriptions();
	Error show_game_center(Dictionary p_params);
	Error request_identity_verification_signature();

	void game_center_closed();

	int get_pending_event_count();
	Variant pop_pending_event();

	static GameCenter *get_singleton();

	GameCenter();
	~GameCenter();
};

#endif

#endif
