#include "tests/test_macros.h"

TEST_FORCE_LINK(test_slime_ai_p04_run)

#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "editor/slime_ai/slime_ai_run_controller.h"
#include "editor/slime_ai/slime_ai_scene_transaction.h"
#include "editor/slime_ai/slime_ai_service_client.h"
#include "scene/2d/node_2d.h"

namespace TestSlimeAIP04Run {

static String _service_script() {
	return OS::get_singleton()->get_executable_path().get_base_dir().get_base_dir().path_join("tools/slime_ai/agent_service/src/main.ts");
}

static String _journal() {
	return OS::get_singleton()->get_user_data_dir().path_join(vformat("slime_ai_p04_run_%d.json", OS::get_singleton()->get_ticks_usec()));
}

static bool _ready(SlimeAI::ServiceClient &p_service) {
	for (int i = 0; i < 500 && !p_service.is_ready(); i++) {
		Vector<Dictionary> frames;
		p_service.poll(frames);
		OS::get_singleton()->delay_usec(2000);
	}
	return p_service.is_ready();
}

static Dictionary _wait_for_preview(SlimeAI::ServiceClient &p_service, SlimeAI::RunController &p_run, Node *p_root) {
	Dictionary last;
	for (int i = 0; i < 1000 && p_run.get_preview_id().is_empty(); i++) {
		Vector<Dictionary> frames;
		p_service.poll(frames);
		for (const Dictionary &frame : frames) {
			last = p_run.on_frame(p_root, true, frame);
		}
		OS::get_singleton()->delay_usec(2000);
	}
	return last;
}

TEST_CASE("[SlimeAI][P04Run] fake Propose cannot apply; human revision blocks Execute transition") {
	const String journal = _journal();
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::ServiceClient service;
		REQUIRE(service.start(_service_script()));
		REQUIRE(_ready(service));
		SlimeAI::SceneTransaction transaction(journal);
		SlimeAI::RunController run(service, transaction);
		const Dictionary started = run.start(root, true, "fake", "", "propose", "Preview one native marker", false);
		REQUIRE(String(started["status"]) == "waiting_for_provider");
		_wait_for_preview(service, run, root);
		REQUIRE_FALSE(run.get_preview_id().is_empty());
		CHECK(root->get_child_count() == 0);
		CHECK(String(Dictionary(run.apply(root, true)["error"])["code"]) == "PERMISSION_DENIED");
		root->set_position(Vector2(11, 7));
		CHECK(String(Dictionary(run.transition_to_execute(root, true)["error"])["code"]) == "REVISION_CONFLICT");
		CHECK(root->get_position() == Vector2(11, 7));
		CHECK(root->get_child_count() == 0);
		service.stop();
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][P04Run] fake Execute uses native grant and exactly one transaction") {
	const String journal = _journal();
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::ServiceClient service;
		REQUIRE(service.start(_service_script()));
		REQUIRE(_ready(service));
		SlimeAI::SceneTransaction transaction(journal);
		SlimeAI::RunController run(service, transaction);
		const Dictionary started = run.start(root, true, "fake", "", "execute", "Add one supported marker", false);
		REQUIRE(String(started["status"]) == "waiting_for_provider");
		Dictionary premature_data;
		premature_data["call_id"] = "forged-early-call";
		premature_data["tool_name"] = "scene_patch_preview";
		premature_data["arguments"] = Dictionary();
		premature_data["provider_response_id"] = "forged-response";
		Dictionary premature_event;
		premature_event["run_id"] = started["run_id"];
		premature_event["event"] = "tool_call_ready";
		premature_event["data"] = premature_data;
		CHECK(String(Dictionary(run.on_frame(root, true, premature_event)["error"])["code"]) == "PROVIDER_PROTOCOL_ERROR");
		CHECK(root->get_child_count() == 0);
		_wait_for_preview(service, run, root);
		REQUIRE_FALSE(run.get_preview_id().is_empty());
		CHECK(root->get_child_count() == 0);
		CHECK(String(Dictionary(run.apply(root, true)["error"])["code"]) == "PERMISSION_REQUIRED");
		CHECK(String(run.grant()["status"]) == "granted");
		CHECK(String(run.apply(root, true)["status"]) == "applied");
		CHECK(root->get_child_count() == 1);
		CHECK(String(Dictionary(run.apply(root, true)["error"])["code"]) == "PERMISSION_DENIED");
		CHECK(root->get_child_count() == 1);
		service.stop();
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][P04Run] Discuss and missing live authorization cannot mutate") {
	const String journal = _journal();
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::ServiceClient service;
		REQUIRE(service.start(_service_script()));
		REQUIRE(_ready(service));
		SlimeAI::SceneTransaction transaction(journal);
		SlimeAI::RunController run(service, transaction);
		CHECK(String(Dictionary(run.start(root, true, "openai_responses", "example-model", "execute", "Apply", false)["error"])["code"]) == "LIVE_AUTHORIZATION_REQUIRED");
		CHECK(String(run.start(root, true, "fake", "", "discuss", "Describe the selected scene", false)["status"]) == "waiting_for_provider");
		for (int i = 0; i < 1000 && String(run.status(root)["status"]) != "completed"; i++) {
			Vector<Dictionary> frames;
			service.poll(frames);
			for (const Dictionary &frame : frames) {
				run.on_frame(root, true, frame);
			}
			OS::get_singleton()->delay_usec(2000);
		}
		CHECK(String(run.status(root)["status"]) == "completed");
		CHECK(String(Dictionary(run.apply(root, true)["error"])["code"]) == "PERMISSION_DENIED");
		CHECK(root->get_child_count() == 0);
		service.stop();
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][P04Run] immediate cancel ignores later fake provider events") {
	const String journal = _journal();
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::ServiceClient service;
		REQUIRE(service.start(_service_script()));
		REQUIRE(_ready(service));
		SlimeAI::SceneTransaction transaction(journal);
		SlimeAI::RunController run(service, transaction);
		REQUIRE(String(run.start(root, true, "fake", "", "propose", "Preview then cancel", false)["status"]) == "waiting_for_provider");
		CHECK(String(run.cancel()["status"]) == "cancelled");
		for (int i = 0; i < 250; i++) {
			Vector<Dictionary> frames;
			service.poll(frames);
			for (const Dictionary &frame : frames) {
				const Dictionary ignored = run.on_frame(root, true, frame);
				CHECK(String(ignored.get("status", "")) != "preview");
			}
			OS::get_singleton()->delay_usec(2000);
		}
		CHECK(run.get_preview_id().is_empty());
		CHECK(root->get_child_count() == 0);
		service.stop();
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][P04Run] pause defers tool dispatch and cancel preserves scene") {
	const String journal = _journal();
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::ServiceClient service;
		REQUIRE(service.start(_service_script()));
		REQUIRE(_ready(service));
		SlimeAI::SceneTransaction transaction(journal);
		SlimeAI::RunController run(service, transaction);
		REQUIRE(String(run.start(root, true, "fake", "", "propose", "Preview while paused", false)["status"]) == "waiting_for_provider");
		CHECK(String(run.pause()["status"]) == "pause_requested");
		for (int i = 0; i < 1000 && String(run.status(root)["status"]) != "awaiting_tool"; i++) {
			Vector<Dictionary> frames;
			service.poll(frames);
			for (const Dictionary &frame : frames) {
				const Dictionary handled = run.on_frame(root, true, frame);
				CHECK_FALSE(handled.has("error"));
			}
			OS::get_singleton()->delay_usec(2000);
		}
		CHECK(run.get_preview_id().is_empty());
		CHECK(root->get_child_count() == 0);
		const Dictionary resumed = run.resume(root, true);
		CHECK_FALSE(resumed.has("error"));
		REQUIRE_FALSE(run.get_preview_id().is_empty());
		CHECK(String(run.cancel()["status"]) == "cancelled");
		CHECK(root->get_child_count() == 0);
		service.stop();
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

} // namespace TestSlimeAIP04Run
