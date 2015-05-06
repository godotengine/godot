/*************************************************************************/
/*  game_center.mm                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "game_center.h"

extern "C" {
#import <GameKit/GameKit.h>
};

GameCenter* GameCenter::instance = NULL;

void GameCenter::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("connect"),&GameCenter::connect);
	ObjectTypeDB::bind_method(_MD("is_connected"),&GameCenter::is_connected);

	ObjectTypeDB::bind_method(_MD("post_score"),&GameCenter::post_score);
	ObjectTypeDB::bind_method(_MD("award_achievement"),&GameCenter::award_achievement);

	ObjectTypeDB::bind_method(_MD("get_pending_event_count"),&GameCenter::get_pending_event_count);
	ObjectTypeDB::bind_method(_MD("pop_pending_event"),&GameCenter::pop_pending_event);
};


Error GameCenter::connect() {

	GKLocalPlayer* player = [GKLocalPlayer localPlayer];
	[player authenticateWithCompletionHandler:^(NSError* error) {

		Dictionary ret;
		ret["type"] = "authentication";
		if (player.isAuthenticated) {
			ret["result"] = "ok";
			GameCenter::get_singleton()->connected = true;
		} else {
			ret["result"] = "error";
			ret["error_code"] = error.code;
			ret["error_description"] = [error.localizedDescription UTF8String];
			GameCenter::get_singleton()->connected = false;
		};

		pending_events.push_back(ret);
	}];
	return OK;
};

bool GameCenter::is_connected() {
	return connected;
};

Error GameCenter::post_score(Variant p_score) {

	Dictionary params = p_score;
	ERR_FAIL_COND_V(!params.has("score") || !params.has("category"), ERR_INVALID_PARAMETER);
	float score = params["score"];
	String category = params["category"];

	NSString* cat_str = [[[NSString alloc] initWithUTF8String:category.utf8().get_data()] autorelease];
	GKScore* reporter = [[[GKScore alloc] initWithCategory:cat_str] autorelease];
	reporter.value = score;

	[reporter reportScoreWithCompletionHandler:^(NSError* error) {

		Dictionary ret;
		ret["type"] = "post_score";
		if (error == nil) {
			ret["result"] = "ok";
		} else {
			ret["result"] = "error";
			ret["error_code"] = error.code;
			ret["error_description"] = [error.localizedDescription UTF8String];
		};

		pending_events.push_back(ret);
	}];

	return OK;
};

Error GameCenter::award_achievement(Variant p_params) {

	Dictionary params = p_params;
	ERR_FAIL_COND_V(!params.has("name") || !params.has("progress"), ERR_INVALID_PARAMETER);
	String name = params["name"];
	float progress = params["progress"];

	NSString* name_str = [[[NSString alloc] initWithUTF8String:name.utf8().get_data()] autorelease];
	GKAchievement* achievement = [[[GKAchievement alloc] initWithIdentifier: name_str] autorelease];
	ERR_FAIL_COND_V(!achievement, FAILED);

	achievement.percentComplete = progress;
	[achievement reportAchievementWithCompletionHandler:^(NSError* error) {

		Dictionary ret;
		ret["type"] = "award_achievement";
		if (error == nil) {
			ret["result"] = "ok";
		} else {
			ret["result"] = "error";
			ret["error_code"] = error.code;
		};

		pending_events.push_back(ret);
	}];

	return OK;
};

int GameCenter::get_pending_event_count() {

	return pending_events.size();
};

Variant GameCenter::pop_pending_event() {

	Variant front = pending_events.front()->get();
	pending_events.pop_front();

	return front;
};

GameCenter* GameCenter::get_singleton() {
	return instance;
};

GameCenter::GameCenter() {
	ERR_FAIL_COND(instance != NULL);
	instance = this;
	connected = false;
};


GameCenter::~GameCenter() {

};

#endif
