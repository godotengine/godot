/**************************************************************************/
/*  test_control.h                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef TEST_CONTROL_H
#define TEST_CONTROL_H

#include "scene/2d/node_2d.h"
#include "scene/gui/control.h"

#include "tests/test_macros.h"

namespace TestControl {

TEST_CASE("[SceneTree][Control] Transforms") {
	SUBCASE("[Control][Global Transform] Global Transform should be accessible while not in SceneTree.") { // GH-79453
		Control *test_node = memnew(Control);
		Control *test_child = memnew(Control);
		test_node->add_child(test_child);

		test_node->set_global_position(Point2(1, 1));
		CHECK_EQ(test_node->get_global_position(), Point2(1, 1));
		CHECK_EQ(test_child->get_global_position(), Point2(1, 1));
		test_node->set_global_position(Point2(2, 2));
		CHECK_EQ(test_node->get_global_position(), Point2(2, 2));
		test_node->set_scale(Vector2(4, 4));
		CHECK_EQ(test_node->get_global_transform(), Transform2D(0, Size2(4, 4), 0, Vector2(2, 2)));
		test_node->set_scale(Vector2(1, 1));
		test_node->set_rotation_degrees(90);
		CHECK_EQ(test_node->get_global_transform(), Transform2D(Math_PI / 2, Vector2(2, 2)));
		test_node->set_pivot_offset(Vector2(1, 0));
		CHECK_EQ(test_node->get_global_transform(), Transform2D(Math_PI / 2, Vector2(3, 1)));

		memdelete(test_child);
		memdelete(test_node);
	}
}

TEST_CASE("[SceneTree][Control] Focus") {
	Control *ctrl = memnew(Control);
	SceneTree::get_singleton()->get_root()->add_child(ctrl);

	SUBCASE("[SceneTree][Control] Default focus") {
		CHECK_UNARY_FALSE(ctrl->has_focus());
	}

	SUBCASE("[SceneTree][Control] Can't grab focus with default focus mode") {
		ERR_PRINT_OFF
		ctrl->grab_focus();
		ERR_PRINT_ON

		CHECK_UNARY_FALSE(ctrl->has_focus());
	}

	SUBCASE("[SceneTree][Control] Can grab focus") {
		ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
		ctrl->grab_focus();

		CHECK_UNARY(ctrl->has_focus());
	}

	SUBCASE("[SceneTree][Control] Can release focus") {
		ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
		ctrl->grab_focus();
		CHECK_UNARY(ctrl->has_focus());

		ctrl->release_focus();
		CHECK_UNARY_FALSE(ctrl->has_focus());
	}

	SUBCASE("[SceneTree][Control] Only one can grab focus at the same time") {
		ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
		ctrl->grab_focus();
		CHECK_UNARY(ctrl->has_focus());

		Control *other_ctrl = memnew(Control);
		SceneTree::get_singleton()->get_root()->add_child(other_ctrl);
		other_ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
		other_ctrl->grab_focus();

		CHECK_UNARY(other_ctrl->has_focus());
		CHECK_UNARY_FALSE(ctrl->has_focus());

		memdelete(other_ctrl);
	}

	SUBCASE("[SceneTree][Control] Hide control will cause the focus to be released") {
		ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
		ctrl->grab_focus();
		CHECK_UNARY(ctrl->has_focus());

		ctrl->hide();
		CHECK_UNARY_FALSE(ctrl->has_focus());

		ctrl->show();
		CHECK_UNARY_FALSE(ctrl->has_focus());
	}

	SUBCASE("[SceneTree][Control] The parent node is hidden causing the focus to be released") {
		Control *child_ctrl = memnew(Control);
		ctrl->add_child(child_ctrl);

		child_ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
		child_ctrl->grab_focus();
		CHECK_UNARY(child_ctrl->has_focus());

		ctrl->hide();
		CHECK_UNARY_FALSE(child_ctrl->has_focus());

		ctrl->show();
		CHECK_UNARY_FALSE(child_ctrl->has_focus());

		memdelete(child_ctrl);
	}

	memdelete(ctrl);
}

TEST_CASE("[SceneTree][Control] Find next/prev valid focus") {
	Control *ctrl = memnew(Control);
	SceneTree::get_singleton()->get_root()->add_child(ctrl);

	SUBCASE("[SceneTree][Control] In FOCUS_CLICK mode") {
		ctrl->set_focus_mode(Control::FocusMode::FOCUS_CLICK);
		ctrl->grab_focus();
		REQUIRE_UNARY(ctrl->has_focus());

		SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
			SEND_GUI_ACTION("ui_focus_next");
			CHECK_UNARY(ctrl->has_focus());
		}

		SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
			SEND_GUI_ACTION("ui_focus_prev");
			CHECK_UNARY(ctrl->has_focus());
		}

		SUBCASE("[SceneTree][Control] Has a sibling control but the parent node is not a control") {
			Control *other_ctrl = memnew(Control);
			SceneTree::get_singleton()->get_root()->add_child(other_ctrl);

			SUBCASE("[SceneTree][Control] Has a sibling control with FOCUS_ALL") {
				other_ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				REQUIRE_EQ(other_ctrl->get_focus_mode(), Control::FocusMode::FOCUS_ALL);

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Manually specify focus next") {
					ctrl->set_focus_next(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Manually specified focus next is hidden") {
						other_ctrl->hide();
						REQUIRE_UNARY_FALSE(other_ctrl->is_visible());

						SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
							SEND_GUI_ACTION("ui_focus_next");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}

						SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
							SEND_GUI_ACTION("ui_focus_prev");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}
					}
				}

				SUBCASE("[SceneTree][Control] Manually specify focus prev") {
					ctrl->set_focus_previous(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Manually specified focus next is hidden") {
						other_ctrl->hide();
						REQUIRE_UNARY_FALSE(other_ctrl->is_visible());

						SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
							SEND_GUI_ACTION("ui_focus_next");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}

						SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
							SEND_GUI_ACTION("ui_focus_prev");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}
					}
				}
			}

			SUBCASE("[SceneTree][Control] Has a sibling control with FOCUS_CLICK") {
				other_ctrl->set_focus_mode(Control::FocusMode::FOCUS_CLICK);
				REQUIRE_EQ(other_ctrl->get_focus_mode(), Control::FocusMode::FOCUS_CLICK);

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Manually specify focus next") {
					ctrl->set_focus_next(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}
				}

				SUBCASE("[SceneTree][Control] Manually specify focus prev") {
					ctrl->set_focus_previous(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}
				}
			}

			SUBCASE("[SceneTree][Control] Has a sibling control with FOCUS_NONE") {
				other_ctrl->set_focus_mode(Control::FocusMode::FOCUS_NONE);
				REQUIRE_EQ(other_ctrl->get_focus_mode(), Control::FocusMode::FOCUS_NONE);

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Manually specify focus next") {
					ctrl->set_focus_next(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}
				}

				SUBCASE("[SceneTree][Control] Manually specify focus prev") {
					ctrl->set_focus_previous(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}
				}
			}

			memdelete(other_ctrl);
		}
	}

	SUBCASE("[SceneTree][Control] In FOCUS_ALL mode") {
		ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
		REQUIRE_EQ(ctrl->get_focus_mode(), Control::FocusMode::FOCUS_ALL);

		ctrl->grab_focus();
		REQUIRE_UNARY(ctrl->has_focus());

		SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
			SEND_GUI_ACTION("ui_focus_next");
			CHECK_UNARY(ctrl->has_focus());
		}

		SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
			SEND_GUI_ACTION("ui_focus_prev");
			CHECK_UNARY(ctrl->has_focus());
		}

		SUBCASE("[SceneTree][Control] Has a sibling control but the parent node is not a control") {
			Control *other_ctrl = memnew(Control);
			SceneTree::get_singleton()->get_root()->add_child(other_ctrl);

			SUBCASE("[SceneTree][Control] Has a sibling control with FOCUS_ALL") {
				other_ctrl->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				REQUIRE_EQ(other_ctrl->get_focus_mode(), Control::FocusMode::FOCUS_ALL);

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Manually specify focus next") {
					ctrl->set_focus_next(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Manually specified focus next is hidden") {
						other_ctrl->hide();
						REQUIRE_UNARY_FALSE(other_ctrl->is_visible());

						SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
							SEND_GUI_ACTION("ui_focus_next");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}

						SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
							SEND_GUI_ACTION("ui_focus_prev");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}
					}
				}

				SUBCASE("[SceneTree][Control] Manually specify focus prev") {
					ctrl->set_focus_previous(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Manually specified focus next is hidden") {
						other_ctrl->hide();
						REQUIRE_UNARY_FALSE(other_ctrl->is_visible());

						SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
							SEND_GUI_ACTION("ui_focus_next");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}

						SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
							SEND_GUI_ACTION("ui_focus_prev");
							CHECK_UNARY(ctrl->has_focus());
							CHECK_UNARY_FALSE(other_ctrl->has_focus());
						}
					}
				}
			}

			SUBCASE("[SceneTree][Control] Has a sibling control with FOCUS_CLICK") {
				other_ctrl->set_focus_mode(Control::FocusMode::FOCUS_CLICK);
				REQUIRE_EQ(other_ctrl->get_focus_mode(), Control::FocusMode::FOCUS_CLICK);

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Manually specify focus next") {
					ctrl->set_focus_next(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}
				}

				SUBCASE("[SceneTree][Control] Manually specify focus prev") {
					ctrl->set_focus_previous(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY_FALSE(ctrl->has_focus());
						CHECK_UNARY(other_ctrl->has_focus());
					}
				}
			}

			SUBCASE("[SceneTree][Control] Has a sibling control with FOCUS_NONE") {
				other_ctrl->set_focus_mode(Control::FocusMode::FOCUS_NONE);
				REQUIRE_EQ(other_ctrl->get_focus_mode(), Control::FocusMode::FOCUS_NONE);

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl->has_focus());
					CHECK_UNARY_FALSE(other_ctrl->has_focus());
				}

				SUBCASE("[SceneTree][Control] Manually specify focus next") {
					ctrl->set_focus_next(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}
				}

				SUBCASE("[SceneTree][Control] Manually specify focus prev") {
					ctrl->set_focus_previous(ctrl->get_path_to(other_ctrl));

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
						CHECK_UNARY_FALSE(other_ctrl->has_focus());
					}
				}
			}

			memdelete(other_ctrl);
		}

		SUBCASE("[SceneTree][Control] Simple control tree") {
			Control *ctrl_0 = memnew(Control);
			Control *ctrl_1 = memnew(Control);
			Node2D *node_2d_2 = memnew(Node2D);

			ctrl->add_child(ctrl_0);
			ctrl->add_child(ctrl_1);
			ctrl->add_child(node_2d_2);

			ctrl_0->set_focus_mode(Control::FocusMode::FOCUS_ALL);
			ctrl_1->set_focus_mode(Control::FocusMode::FOCUS_ALL);
			REQUIRE_EQ(ctrl_0->get_focus_mode(), Control::FocusMode::FOCUS_ALL);
			REQUIRE_EQ(ctrl_1->get_focus_mode(), Control::FocusMode::FOCUS_ALL);

			SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
				SEND_GUI_ACTION("ui_focus_next");
				CHECK_UNARY(ctrl_0->has_focus());

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl_1->has_focus());

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl->has_focus());
					}
				}
			}

			SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
				SEND_GUI_ACTION("ui_focus_prev");
				CHECK_UNARY(ctrl_1->has_focus());

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl_0->has_focus());

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
					}
				}
			}

			SUBCASE("[SceneTree][Control] Skip next hidden control") {
				ctrl_0->hide();
				REQUIRE_UNARY_FALSE(ctrl_0->is_visible());
				SEND_GUI_ACTION("ui_focus_next");
				CHECK_UNARY_FALSE(ctrl_0->has_focus());
				CHECK_UNARY(ctrl_1->has_focus());
			}

			SUBCASE("[SceneTree][Control] Skip next control with FOCUS_NONE") {
				ctrl_0->set_focus_mode(Control::FocusMode::FOCUS_NONE);
				REQUIRE_EQ(ctrl_0->get_focus_mode(), Control::FocusMode::FOCUS_NONE);
				SEND_GUI_ACTION("ui_focus_next");
				CHECK_UNARY_FALSE(ctrl_0->has_focus());
				CHECK_UNARY(ctrl_1->has_focus());
			}

			SUBCASE("[SceneTree][Control] Skip next control with FOCUS_CLICK") {
				ctrl_0->set_focus_mode(Control::FocusMode::FOCUS_CLICK);
				REQUIRE_EQ(ctrl_0->get_focus_mode(), Control::FocusMode::FOCUS_CLICK);
				SEND_GUI_ACTION("ui_focus_next");
				CHECK_UNARY_FALSE(ctrl_0->has_focus());
				CHECK_UNARY(ctrl_1->has_focus());
			}

			SUBCASE("[SceneTree][Control] Skip next top level control") {
				ctrl_0->set_as_top_level(true);
				REQUIRE_UNARY(ctrl_0->is_set_as_top_level());
				SEND_GUI_ACTION("ui_focus_next");
				CHECK_UNARY_FALSE(ctrl_0->has_focus());
				CHECK_UNARY(ctrl_1->has_focus());
			}

			SUBCASE("[SceneTree][Control] Skip prev hidden control") {
				ctrl_1->hide();
				REQUIRE_UNARY_FALSE(ctrl_1->is_visible());
				SEND_GUI_ACTION("ui_focus_prev");
				CHECK_UNARY_FALSE(ctrl_1->has_focus());
				CHECK_UNARY(ctrl_0->has_focus());
			}

			SUBCASE("[SceneTree][Control] Skip prev control with FOCUS_NONE") {
				ctrl_1->set_focus_mode(Control::FocusMode::FOCUS_NONE);
				REQUIRE_EQ(ctrl_1->get_focus_mode(), Control::FocusMode::FOCUS_NONE);
				SEND_GUI_ACTION("ui_focus_prev");
				CHECK_UNARY_FALSE(ctrl_1->has_focus());
				CHECK_UNARY(ctrl_0->has_focus());
			}

			SUBCASE("[SceneTree][Control] Skip prev control with FOCUS_CLICK") {
				ctrl_1->set_focus_mode(Control::FocusMode::FOCUS_CLICK);
				REQUIRE_EQ(ctrl_1->get_focus_mode(), Control::FocusMode::FOCUS_CLICK);
				SEND_GUI_ACTION("ui_focus_prev");
				CHECK_UNARY_FALSE(ctrl_1->has_focus());
				CHECK_UNARY(ctrl_0->has_focus());
			}

			SUBCASE("[SceneTree][Control] Skip prev top level control") {
				ctrl_1->set_as_top_level(true);
				REQUIRE_UNARY(ctrl_1->is_set_as_top_level());
				SEND_GUI_ACTION("ui_focus_prev");
				CHECK_UNARY_FALSE(ctrl_1->has_focus());
				CHECK_UNARY(ctrl_0->has_focus());
			}

			SUBCASE("[SceneTree][Control] Add more node controls") {
				Control *ctrl_0_0 = memnew(Control);
				Control *ctrl_0_1 = memnew(Control);
				Control *ctrl_0_2 = memnew(Control);
				ctrl_0->add_child(ctrl_0_0);
				ctrl_0->add_child(ctrl_0_1);
				ctrl_0->add_child(ctrl_0_2);
				ctrl_0_0->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				ctrl_0_1->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				ctrl_0_2->set_focus_mode(Control::FocusMode::FOCUS_ALL);

				Control *ctrl_1_0 = memnew(Control);
				Control *ctrl_1_1 = memnew(Control);
				Control *ctrl_1_2 = memnew(Control);
				ctrl_1->add_child(ctrl_1_0);
				ctrl_1->add_child(ctrl_1_1);
				ctrl_1->add_child(ctrl_1_2);
				ctrl_1_0->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				ctrl_1_1->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				ctrl_1_2->set_focus_mode(Control::FocusMode::FOCUS_ALL);

				Control *ctrl_2_0 = memnew(Control);
				Control *ctrl_2_1 = memnew(Control);
				Control *ctrl_2_2 = memnew(Control);
				node_2d_2->add_child(ctrl_2_0);
				node_2d_2->add_child(ctrl_2_1);
				node_2d_2->add_child(ctrl_2_2);
				ctrl_2_0->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				ctrl_2_1->set_focus_mode(Control::FocusMode::FOCUS_ALL);
				ctrl_2_2->set_focus_mode(Control::FocusMode::FOCUS_ALL);

				SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
					SEND_GUI_ACTION("ui_focus_next");
					CHECK_UNARY(ctrl_0->has_focus());

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl_0_0->has_focus());
					}

					SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
						SEND_GUI_ACTION("ui_focus_prev");
						CHECK_UNARY(ctrl->has_focus());
					}
				}

				SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
					SEND_GUI_ACTION("ui_focus_prev");
					CHECK_UNARY(ctrl_1_2->has_focus());
				}

				SUBCASE("[SceneTree][Control] Exist top level tree") {
					ctrl_0->set_as_top_level(true);
					REQUIRE_UNARY(ctrl_0->is_set_as_top_level());

					SUBCASE("[SceneTree][Control] Outside top level tree") {
						ctrl->grab_focus();
						REQUIRE_UNARY(ctrl->has_focus());
						SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
							SEND_GUI_ACTION("ui_focus_next");
							CHECK_UNARY(ctrl_1->has_focus());

							SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
								SEND_GUI_ACTION("ui_focus_prev");
								CHECK_UNARY(ctrl->has_focus());
							}
						}
					}

					SUBCASE("[SceneTree][Control] Inside top level tree") {
						ctrl_0->grab_focus();
						REQUIRE_UNARY(ctrl_0->has_focus());
						SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
							SEND_GUI_ACTION("ui_focus_next");
							CHECK_UNARY(ctrl_0_0->has_focus());

							SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
								SEND_GUI_ACTION("ui_focus_prev");
								CHECK_UNARY(ctrl_0->has_focus());
							}
						}
						SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
							SEND_GUI_ACTION("ui_focus_prev");
							CHECK_UNARY(ctrl_0_2->has_focus());

							SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
								SEND_GUI_ACTION("ui_focus_next");
								CHECK_UNARY(ctrl_0->has_focus());
							}
						}
					}

					SUBCASE("[SceneTree][Control] Manually specified focus next") {
						ctrl->set_focus_next(ctrl->get_path_to(ctrl_2_1));
						ctrl_2_1->set_focus_next(ctrl_2_1->get_path_to(ctrl_1_0));
						ctrl_1_0->set_focus_next(ctrl_1_0->get_path_to(ctrl_0));
						ctrl_0->set_focus_next(ctrl_0->get_path_to(ctrl));

						SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
							SEND_GUI_ACTION("ui_focus_next");
							CHECK_UNARY(ctrl_2_1->has_focus());

							SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
								SEND_GUI_ACTION("ui_focus_next");
								CHECK_UNARY(ctrl_1_0->has_focus());

								SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
									SEND_GUI_ACTION("ui_focus_next");
									CHECK_UNARY(ctrl_0->has_focus());

									SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
										SEND_GUI_ACTION("ui_focus_next");
										CHECK_UNARY(ctrl->has_focus());
									}

									SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
										SEND_GUI_ACTION("ui_focus_prev");
										CHECK_UNARY(ctrl_0_2->has_focus());
									}
								}

								SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
									SEND_GUI_ACTION("ui_focus_prev");
									CHECK_UNARY(ctrl_1->has_focus());
								}
							}

							SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
								SEND_GUI_ACTION("ui_focus_prev");
								CHECK_UNARY(ctrl_2_1->has_focus());
							}
						}

						SUBCASE("[SceneTree][Control] The parent node is not visible") {
							node_2d_2->hide();
							REQUIRE_UNARY(ctrl_2_1->is_visible());
							REQUIRE_UNARY_FALSE(ctrl_2_1->is_visible_in_tree());
							SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
								SEND_GUI_ACTION("ui_focus_next");
								CHECK_UNARY_FALSE(ctrl->has_focus());
								CHECK_UNARY_FALSE(ctrl_2_1->has_focus());
								CHECK_UNARY_FALSE(ctrl_0->has_focus());
								CHECK_UNARY(ctrl_1->has_focus());
							}
						}
					}

					SUBCASE("[SceneTree][Control] Manually specified focus prev") {
						ctrl->set_focus_previous(ctrl->get_path_to(ctrl_0_2));
						ctrl_0_2->set_focus_previous(ctrl_0_2->get_path_to(ctrl_1_1));
						ctrl_1_1->set_focus_previous(ctrl_1_1->get_path_to(ctrl_2_0));
						ctrl_2_0->set_focus_previous(ctrl_2_0->get_path_to(ctrl));

						SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
							SEND_GUI_ACTION("ui_focus_prev");
							CHECK_UNARY(ctrl_0_2->has_focus());

							SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
								SEND_GUI_ACTION("ui_focus_prev");
								CHECK_UNARY(ctrl_1_1->has_focus());

								SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
									SEND_GUI_ACTION("ui_focus_prev");
									CHECK_UNARY(ctrl_2_0->has_focus());

									SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
										SEND_GUI_ACTION("ui_focus_prev");
										CHECK_UNARY(ctrl->has_focus());
									}

									SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
										SEND_GUI_ACTION("ui_focus_next");
										CHECK_UNARY(ctrl_2_0->has_focus());
									}
								}

								SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
									SEND_GUI_ACTION("ui_focus_next");
									CHECK_UNARY(ctrl_1_2->has_focus());
								}
							}

							SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
								SEND_GUI_ACTION("ui_focus_next");
								CHECK_UNARY(ctrl_0->has_focus());
							}
						}

						SUBCASE("[SceneTree][Control] The parent node is not visible") {
							ctrl_0->hide();
							REQUIRE_UNARY(ctrl_0_2->is_visible());
							REQUIRE_UNARY_FALSE(ctrl_0_2->is_visible_in_tree());
							SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
								SEND_GUI_ACTION("ui_focus_prev");
								CHECK_UNARY_FALSE(ctrl->has_focus());
								CHECK_UNARY_FALSE(ctrl_0_2->has_focus());
								CHECK_UNARY(ctrl_1_2->has_focus());
							}
						}
					}
				}

				SUBCASE("[SceneTree][Control] Exist hidden control tree") {
					ctrl_0->hide();
					REQUIRE_UNARY_FALSE(ctrl_0->is_visible());

					SUBCASE("[SceneTree][Control] Simulate ui_focus_next action") {
						SEND_GUI_ACTION("ui_focus_next");
						CHECK_UNARY(ctrl_1->has_focus());

						SUBCASE("[SceneTree][Control] Simulate ui_focus_prev action") {
							SEND_GUI_ACTION("ui_focus_prev");
							CHECK_UNARY(ctrl->has_focus());
						}
					}
				}

				memdelete(ctrl_2_2);
				memdelete(ctrl_2_1);
				memdelete(ctrl_2_0);
				memdelete(ctrl_1_2);
				memdelete(ctrl_1_1);
				memdelete(ctrl_1_0);
				memdelete(ctrl_0_2);
				memdelete(ctrl_0_1);
				memdelete(ctrl_0_0);
			}

			memdelete(node_2d_2);
			memdelete(ctrl_1);
			memdelete(ctrl_0);
		}
	}

	memdelete(ctrl);
}

TEST_CASE("[SceneTree][Control] Anchoring") {
	Control *test_control = memnew(Control);
	Control *test_child = memnew(Control);
	test_control->add_child(test_child);
	test_control->set_size(Size2(2, 2));
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_control);

	SUBCASE("Anchoring without offsets") {
		test_child->set_anchor(SIDE_RIGHT, 0.75);
		test_child->set_anchor(SIDE_BOTTOM, 0.1);
		CHECK_MESSAGE(
				test_child->get_size().is_equal_approx(Vector2(1.5, 0.2)),
				"With no LEFT or TOP anchors, positive RIGHT and BOTTOM anchors should be proportional to the size.");
		CHECK_MESSAGE(
				test_child->get_position().is_equal_approx(Vector2(0, 0)),
				"With positive RIGHT and BOTTOM anchors set and no LEFT or TOP anchors, the position should not change.");

		test_child->set_anchor(SIDE_LEFT, 0.5);
		test_child->set_anchor(SIDE_TOP, 0.01);
		CHECK_MESSAGE(
				test_child->get_size().is_equal_approx(Vector2(0.5, 0.18)),
				"With all anchors set, the size should fit between all four anchors.");
		CHECK_MESSAGE(
				test_child->get_position().is_equal_approx(Vector2(1, 0.02)),
				"With all anchors set, the LEFT and TOP anchors should proportional to the position.");
	}

	SUBCASE("Anchoring with offsets") {
		test_child->set_offset(SIDE_RIGHT, 0.33);
		test_child->set_offset(SIDE_BOTTOM, 0.2);
		CHECK_MESSAGE(
				test_child->get_size().is_equal_approx(Vector2(0.33, 0.2)),
				"With no anchors or LEFT or TOP offsets set, the RIGHT and BOTTOM offsets should be equal to size.");
		CHECK_MESSAGE(
				test_child->get_position().is_equal_approx(Vector2(0, 0)),
				"With only positive RIGHT and BOTTOM offsets set, the position should not change.");

		test_child->set_offset(SIDE_LEFT, 0.1);
		test_child->set_offset(SIDE_TOP, 0.05);
		CHECK_MESSAGE(
				test_child->get_size().is_equal_approx(Vector2(0.23, 0.15)),
				"With no anchors set, the size should fit between all four offsets.");
		CHECK_MESSAGE(
				test_child->get_position().is_equal_approx(Vector2(0.1, 0.05)),
				"With no anchors set, the LEFT and TOP offsets should be equal to the position.");

		test_child->set_anchor(SIDE_RIGHT, 0.5);
		test_child->set_anchor(SIDE_BOTTOM, 0.3);
		test_child->set_anchor(SIDE_LEFT, 0.2);
		test_child->set_anchor(SIDE_TOP, 0.1);
		CHECK_MESSAGE(
				test_child->get_size().is_equal_approx(Vector2(0.83, 0.55)),
				"Anchors adjust size first then it is affected by offsets.");
		CHECK_MESSAGE(
				test_child->get_position().is_equal_approx(Vector2(0.5, 0.25)),
				"Anchors adjust positions first then it is affected by offsets.");

		test_child->set_offset(SIDE_RIGHT, -0.1);
		test_child->set_offset(SIDE_BOTTOM, -0.01);
		test_child->set_offset(SIDE_LEFT, -0.33);
		test_child->set_offset(SIDE_TOP, -0.16);
		CHECK_MESSAGE(
				test_child->get_size().is_equal_approx(Vector2(0.83, 0.55)),
				"Keeping offset distance equal when changing offsets, keeps size equal.");
		CHECK_MESSAGE(
				test_child->get_position().is_equal_approx(Vector2(0.07, 0.04)),
				"Negative offsets move position in top left direction.");
	}

	SUBCASE("Anchoring is preserved on parent size changed") {
		test_child->set_offset(SIDE_RIGHT, -0.05);
		test_child->set_offset(SIDE_BOTTOM, 0.1);
		test_child->set_offset(SIDE_LEFT, 0.05);
		test_child->set_offset(SIDE_TOP, 0.1);
		test_child->set_anchor(SIDE_RIGHT, 0.3);
		test_child->set_anchor(SIDE_BOTTOM, 0.85);
		test_child->set_anchor(SIDE_LEFT, 0.2);
		test_child->set_anchor(SIDE_TOP, 0.55);
		CHECK(test_child->get_rect().is_equal_approx(
				Rect2(Vector2(0.45, 1.2), Size2(0.1, 0.6))));

		test_control->set_size(Size2(4, 1));
		CHECK(test_child->get_rect().is_equal_approx(
				Rect2(Vector2(0.85, 0.65), Size2(0.3, 0.3))));
	}

	memdelete(test_child);
	memdelete(test_control);
}

TEST_CASE("[SceneTree][Control] Custom minimum size") {
	Control *test_control = memnew(Control);
	test_control->set_custom_minimum_size(Size2(4, 2));
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_control);
	CHECK_MESSAGE(
			test_control->get_size().is_equal_approx(Vector2(4, 2)),
			"Size increases to match custom minimum size.");

	test_control->set_size(Size2(5, 4));
	CHECK_MESSAGE(
			test_control->get_size().is_equal_approx(Vector2(5, 4)),
			"Size does not change if above custom minimum size.");

	test_control->set_size(Size2(1, 1));
	CHECK_MESSAGE(
			test_control->get_size().is_equal_approx(Vector2(4, 2)),
			"Size matches minimum size if set below custom minimum size.");

	test_control->set_size(Size2(3, 3));
	CHECK_MESSAGE(
			test_control->get_size().is_equal_approx(Vector2(4, 3)),
			"Adjusts only x axis size if x is below custom minimum size.");

	test_control->set_size(Size2(10, 0.1));
	CHECK_MESSAGE(
			test_control->get_size().is_equal_approx(Vector2(10, 2)),
			"Adjusts only y axis size if y is below custom minimum size.");

	memdelete(test_control);
}

TEST_CASE("[SceneTree][Control] Grow direction") {
	Control *test_control = memnew(Control);
	test_control->set_size(Size2(1, 1));
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_control);

	SUBCASE("Defaults") {
		CHECK(test_control->get_h_grow_direction() == Control::GROW_DIRECTION_END);
		CHECK(test_control->get_v_grow_direction() == Control::GROW_DIRECTION_END);
	}

	SIGNAL_WATCH(test_control, SNAME("minimum_size_changed"))
	Array signal_args;
	signal_args.push_back(Array());

	SUBCASE("Horizontal grow direction begin") {
		test_control->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
		test_control->set_custom_minimum_size(Size2(2, 2));
		SceneTree::get_singleton()->process(0);
		SIGNAL_CHECK("minimum_size_changed", signal_args)
		CHECK_MESSAGE(
				test_control->get_rect().is_equal_approx(
						Rect2(Vector2(-1, 0), Size2(2, 2))),
				"Expand leftwards.");
	}

	SUBCASE("Vertical grow direction begin") {
		test_control->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
		test_control->set_custom_minimum_size(Size2(4, 3));
		SceneTree::get_singleton()->process(0);
		SIGNAL_CHECK("minimum_size_changed", signal_args);
		CHECK_MESSAGE(
				test_control->get_rect().is_equal_approx(
						Rect2(Vector2(0, -2), Size2(4, 3))),
				"Expand upwards.");
	}

	SUBCASE("Horizontal grow direction end") {
		test_control->set_h_grow_direction(Control::GROW_DIRECTION_END);
		test_control->set_custom_minimum_size(Size2(5, 3));
		SceneTree::get_singleton()->process(0);
		SIGNAL_CHECK("minimum_size_changed", signal_args);
		CHECK_MESSAGE(
				test_control->get_rect().is_equal_approx(
						Rect2(Vector2(0, 0), Size2(5, 3))),
				"Expand rightwards.");
	}

	SUBCASE("Vertical grow direction end") {
		test_control->set_v_grow_direction(Control::GROW_DIRECTION_END);
		test_control->set_custom_minimum_size(Size2(4, 4));
		SceneTree::get_singleton()->process(0);
		SIGNAL_CHECK("minimum_size_changed", signal_args);
		CHECK_MESSAGE(
				test_control->get_rect().is_equal_approx(
						Rect2(Vector2(0, 0), Size2(4, 4))),
				"Expand downwards.");
		;
	}

	SUBCASE("Horizontal grow direction both") {
		test_control->set_h_grow_direction(Control::GROW_DIRECTION_BOTH);
		test_control->set_custom_minimum_size(Size2(2, 4));
		SceneTree::get_singleton()->process(0);
		SIGNAL_CHECK("minimum_size_changed", signal_args);
		CHECK_MESSAGE(
				test_control->get_rect().is_equal_approx(
						Rect2(Vector2(-0.5, 0), Size2(2, 4))),
				"Expand equally leftwards and rightwards.");
	}

	SUBCASE("Vertical grow direction both") {
		test_control->set_v_grow_direction(Control::GROW_DIRECTION_BOTH);
		test_control->set_custom_minimum_size(Size2(6, 3));
		SceneTree::get_singleton()->process(0);
		SIGNAL_CHECK("minimum_size_changed", signal_args);
		CHECK_MESSAGE(
				test_control->get_rect().is_equal_approx(
						Rect2(Vector2(0, -1), Size2(6, 3))),
				"Expand equally upwards and downwards.");
	}

	memdelete(test_control);
}

} // namespace TestControl

#endif // TEST_CONTROL_H
