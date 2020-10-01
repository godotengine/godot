/*************************************************************************/
/*  game_center.mm                                                       */
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

#include "game_center.h"

#ifdef __IPHONE_9_0

#import <GameKit/GameKit.h>
extern "C" {

#else

extern "C" {
#import <GameKit/GameKit.h>

#endif

#import "app_delegate.h"
};

#import "view_controller.h"

GameCenter *GameCenter::instance = NULL;

void GameCenter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("authenticate"), &GameCenter::authenticate);
	ClassDB::bind_method(D_METHOD("is_authenticated"), &GameCenter::is_authenticated);

	ClassDB::bind_method(D_METHOD("post_score"), &GameCenter::post_score);
	ClassDB::bind_method(D_METHOD("award_achievement", "achievement"), &GameCenter::award_achievement);
	ClassDB::bind_method(D_METHOD("reset_achievements"), &GameCenter::reset_achievements);
	ClassDB::bind_method(D_METHOD("request_achievements"), &GameCenter::request_achievements);
	ClassDB::bind_method(D_METHOD("request_achievement_descriptions"), &GameCenter::request_achievement_descriptions);
	ClassDB::bind_method(D_METHOD("show_game_center"), &GameCenter::show_game_center);
	ClassDB::bind_method(D_METHOD("request_identity_verification_signature"), &GameCenter::request_identity_verification_signature);

	ClassDB::bind_method(D_METHOD("get_pending_event_count"), &GameCenter::get_pending_event_count);
	ClassDB::bind_method(D_METHOD("pop_pending_event"), &GameCenter::pop_pending_event);
};

Error GameCenter::authenticate() {
	//if this class isn't available, game center isn't implemented
	if ((NSClassFromString(@"GKLocalPlayer")) == nil) {
		return ERR_UNAVAILABLE;
	}

	GKLocalPlayer *player = [GKLocalPlayer localPlayer];
	ERR_FAIL_COND_V(![player respondsToSelector:@selector(authenticateHandler)], ERR_UNAVAILABLE);

	ViewController *root_controller = (ViewController *)((AppDelegate *)[[UIApplication sharedApplication] delegate]).window.rootViewController;
	ERR_FAIL_COND_V(!root_controller, FAILED);

	// This handler is called several times.  First when the view needs to be shown, then again
	// after the view is cancelled or the user logs in.  Or if the user's already logged in, it's
	// called just once to confirm they're authenticated.  This is why no result needs to be specified
	// in the presentViewController phase. In this case, more calls to this function will follow.
	_weakify(root_controller);
	_weakify(player);
	player.authenticateHandler = (^(UIViewController *controller, NSError *error) {
		_strongify(root_controller);
		_strongify(player);

		if (controller) {
			[root_controller presentViewController:controller animated:YES completion:nil];
		} else {
			Dictionary ret;
			ret["type"] = "authentication";
			if (player.isAuthenticated) {
				ret["result"] = "ok";
				if (@available(iOS 13, *)) {
					ret["player_id"] = [player.teamPlayerID UTF8String];
#if !defined(TARGET_OS_SIMULATOR) || !TARGET_OS_SIMULATOR
				} else {
					ret["player_id"] = [player.playerID UTF8String];
#endif
				}

				GameCenter::get_singleton()->authenticated = true;
			} else {
				ret["result"] = "error";
				ret["error_code"] = (int64_t)error.code;
				ret["error_description"] = [error.localizedDescription UTF8String];
				GameCenter::get_singleton()->authenticated = false;
			};

			pending_events.push_back(ret);
		};
	});

	return OK;
};

bool GameCenter::is_authenticated() {
	return authenticated;
};

Error GameCenter::post_score(Dictionary p_score) {
	ERR_FAIL_COND_V(!p_score.has("score") || !p_score.has("category"), ERR_INVALID_PARAMETER);
	float score = p_score["score"];
	String category = p_score["category"];

	NSString *cat_str = [[NSString alloc] initWithUTF8String:category.utf8().get_data()];
	GKScore *reporter = [[GKScore alloc] initWithLeaderboardIdentifier:cat_str];
	reporter.value = score;

	ERR_FAIL_COND_V([GKScore respondsToSelector:@selector(reportScores)], ERR_UNAVAILABLE);

	[GKScore reportScores:@[ reporter ]
			withCompletionHandler:^(NSError *error) {
				Dictionary ret;
				ret["type"] = "post_score";
				if (error == nil) {
					ret["result"] = "ok";
				} else {
					ret["result"] = "error";
					ret["error_code"] = (int64_t)error.code;
					ret["error_description"] = [error.localizedDescription UTF8String];
				};

				pending_events.push_back(ret);
			}];

	return OK;
};

Error GameCenter::award_achievement(Dictionary p_params) {
	ERR_FAIL_COND_V(!p_params.has("name") || !p_params.has("progress"), ERR_INVALID_PARAMETER);
	String name = p_params["name"];
	float progress = p_params["progress"];

	NSString *name_str = [[NSString alloc] initWithUTF8String:name.utf8().get_data()];
	GKAchievement *achievement = [[GKAchievement alloc] initWithIdentifier:name_str];
	ERR_FAIL_COND_V(!achievement, FAILED);

	ERR_FAIL_COND_V([GKAchievement respondsToSelector:@selector(reportAchievements)], ERR_UNAVAILABLE);

	achievement.percentComplete = progress;
	achievement.showsCompletionBanner = NO;
	if (p_params.has("show_completion_banner")) {
		achievement.showsCompletionBanner = p_params["show_completion_banner"] ? YES : NO;
	}

	[GKAchievement reportAchievements:@[ achievement ]
				withCompletionHandler:^(NSError *error) {
					Dictionary ret;
					ret["type"] = "award_achievement";
					if (error == nil) {
						ret["result"] = "ok";
					} else {
						ret["result"] = "error";
						ret["error_code"] = (int64_t)error.code;
					};

					pending_events.push_back(ret);
				}];

	return OK;
};

void GameCenter::request_achievement_descriptions() {
	[GKAchievementDescription loadAchievementDescriptionsWithCompletionHandler:^(NSArray *descriptions, NSError *error) {
		Dictionary ret;
		ret["type"] = "achievement_descriptions";
		if (error == nil) {
			ret["result"] = "ok";
			PackedStringArray names;
			PackedStringArray titles;
			PackedStringArray unachieved_descriptions;
			PackedStringArray achieved_descriptions;
			PackedInt32Array maximum_points;
			Array hidden;
			Array replayable;

			for (NSUInteger i = 0; i < [descriptions count]; i++) {
				GKAchievementDescription *description = [descriptions objectAtIndex:i];

				const char *str = [description.identifier UTF8String];
				names.push_back(String::utf8(str != NULL ? str : ""));

				str = [description.title UTF8String];
				titles.push_back(String::utf8(str != NULL ? str : ""));

				str = [description.unachievedDescription UTF8String];
				unachieved_descriptions.push_back(String::utf8(str != NULL ? str : ""));

				str = [description.achievedDescription UTF8String];
				achieved_descriptions.push_back(String::utf8(str != NULL ? str : ""));

				maximum_points.push_back(description.maximumPoints);

				hidden.push_back(description.hidden == YES);

				replayable.push_back(description.replayable == YES);
			}

			ret["names"] = names;
			ret["titles"] = titles;
			ret["unachieved_descriptions"] = unachieved_descriptions;
			ret["achieved_descriptions"] = achieved_descriptions;
			ret["maximum_points"] = maximum_points;
			ret["hidden"] = hidden;
			ret["replayable"] = replayable;

		} else {
			ret["result"] = "error";
			ret["error_code"] = (int64_t)error.code;
		};

		pending_events.push_back(ret);
	}];
};

void GameCenter::request_achievements() {
	[GKAchievement loadAchievementsWithCompletionHandler:^(NSArray *achievements, NSError *error) {
		Dictionary ret;
		ret["type"] = "achievements";
		if (error == nil) {
			ret["result"] = "ok";
			PackedStringArray names;
			PackedFloat32Array percentages;

			for (NSUInteger i = 0; i < [achievements count]; i++) {
				GKAchievement *achievement = [achievements objectAtIndex:i];
				const char *str = [achievement.identifier UTF8String];
				names.push_back(String::utf8(str != NULL ? str : ""));

				percentages.push_back(achievement.percentComplete);
			}

			ret["names"] = names;
			ret["progress"] = percentages;

		} else {
			ret["result"] = "error";
			ret["error_code"] = (int64_t)error.code;
		};

		pending_events.push_back(ret);
	}];
};

void GameCenter::reset_achievements() {
	[GKAchievement resetAchievementsWithCompletionHandler:^(NSError *error) {
		Dictionary ret;
		ret["type"] = "reset_achievements";
		if (error == nil) {
			ret["result"] = "ok";
		} else {
			ret["result"] = "error";
			ret["error_code"] = (int64_t)error.code;
		};

		pending_events.push_back(ret);
	}];
};

Error GameCenter::show_game_center(Dictionary p_params) {
	ERR_FAIL_COND_V(!NSProtocolFromString(@"GKGameCenterControllerDelegate"), FAILED);

	GKGameCenterViewControllerState view_state = GKGameCenterViewControllerStateDefault;
	if (p_params.has("view")) {
		String view_name = p_params["view"];
		if (view_name == "default") {
			view_state = GKGameCenterViewControllerStateDefault;
		} else if (view_name == "leaderboards") {
			view_state = GKGameCenterViewControllerStateLeaderboards;
		} else if (view_name == "achievements") {
			view_state = GKGameCenterViewControllerStateAchievements;
		} else if (view_name == "challenges") {
			view_state = GKGameCenterViewControllerStateChallenges;
		} else {
			return ERR_INVALID_PARAMETER;
		}
	}

	GKGameCenterViewController *controller = [[GKGameCenterViewController alloc] init];
	ERR_FAIL_COND_V(!controller, FAILED);

	ViewController *root_controller = (ViewController *)((AppDelegate *)[[UIApplication sharedApplication] delegate]).window.rootViewController;
	ERR_FAIL_COND_V(!root_controller, FAILED);

	controller.gameCenterDelegate = root_controller;
	controller.viewState = view_state;
	if (view_state == GKGameCenterViewControllerStateLeaderboards) {
		controller.leaderboardIdentifier = nil;
		if (p_params.has("leaderboard_name")) {
			String name = p_params["leaderboard_name"];
			NSString *name_str = [[NSString alloc] initWithUTF8String:name.utf8().get_data()];
			controller.leaderboardIdentifier = name_str;
		}
	}

	[root_controller presentViewController:controller animated:YES completion:nil];

	return OK;
};

Error GameCenter::request_identity_verification_signature() {
	ERR_FAIL_COND_V(!is_authenticated(), ERR_UNAUTHORIZED);

	GKLocalPlayer *player = [GKLocalPlayer localPlayer];
	[player generateIdentityVerificationSignatureWithCompletionHandler:^(NSURL *publicKeyUrl, NSData *signature, NSData *salt, uint64_t timestamp, NSError *error) {
		Dictionary ret;
		ret["type"] = "identity_verification_signature";
		if (error == nil) {
			ret["result"] = "ok";
			ret["public_key_url"] = [publicKeyUrl.absoluteString UTF8String];
			ret["signature"] = [[signature base64EncodedStringWithOptions:0] UTF8String];
			ret["salt"] = [[salt base64EncodedStringWithOptions:0] UTF8String];
			ret["timestamp"] = timestamp;
			if (@available(iOS 13, *)) {
				ret["player_id"] = [player.teamPlayerID UTF8String];
#if !defined(TARGET_OS_SIMULATOR) || !TARGET_OS_SIMULATOR
			} else {
				ret["player_id"] = [player.playerID UTF8String];
#endif
			}
		} else {
			ret["result"] = "error";
			ret["error_code"] = (int64_t)error.code;
			ret["error_description"] = [error.localizedDescription UTF8String];
		};

		pending_events.push_back(ret);
	}];

	return OK;
};

void GameCenter::game_center_closed() {
	Dictionary ret;
	ret["type"] = "show_game_center";
	ret["result"] = "ok";
	pending_events.push_back(ret);
}

int GameCenter::get_pending_event_count() {
	return pending_events.size();
};

Variant GameCenter::pop_pending_event() {
	Variant front = pending_events.front()->get();
	pending_events.pop_front();

	return front;
};

GameCenter *GameCenter::get_singleton() {
	return instance;
};

GameCenter::GameCenter() {
	ERR_FAIL_COND(instance != NULL);
	instance = this;
	authenticated = false;
};

GameCenter::~GameCenter() {}

#endif
