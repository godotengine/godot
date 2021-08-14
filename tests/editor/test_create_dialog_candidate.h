/*************************************************************************/
/*  test_create_dialog_candidate.h                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_CREATE_DIALOG_CANDIDATE_H
#define TEST_CREATE_DIALOG_CANDIDATE_H

#include "thirdparty/doctest/doctest.h"

#include "editor/create_dialog.h"

TEST_SUITE("[Candidate] isValid") {
	TEST_CASE("Substring match") {
		SUBCASE("Candidate with matching query.") {
			CreateDialogCandidate candidate("Node", false, false, false);

			CHECK(candidate.is_valid("node"));
		}

		SUBCASE("Should be valid with query should be case insensitive.") {
			CreateDialogCandidate candidate("Node", false, false, false);

			CHECK(candidate.is_valid("nODe"));
		}

		SUBCASE("Should be valid with query as a substring starting at the beginning.") {
			CreateDialogCandidate candidate("Node", false, false, false);

			CHECK(candidate.is_valid("nod"));
		}

		SUBCASE("Should be valid with query as a substring in the middle.") {
			CreateDialogCandidate candidate("GraphNode", false, false, false);

			CHECK(candidate.is_valid("nod"));
		}

		SUBCASE("Should be invalid with subequence query.") {
			CreateDialogCandidate candidate("GraphNode", false, false, false);

			CHECK(!candidate.is_valid("grph"));
		}

		SUBCASE("Should be invalid with excessive query.") {
			CreateDialogCandidate candidate("GraphNode", false, false, false);

			CHECK(!candidate.is_valid("grph"));
		}
	}

	TEST_CASE("Word boundary") {
		SUBCASE("Should be valid with subsequence match based on word boundary characters.") {
			CreateDialogCandidate candidate("AnimationPlayer", false, false, false);

			CHECK(candidate.is_valid("ap"));
		}

		SUBCASE("Should be valid with digit as word boundary character.") {
			CreateDialogCandidate candidate("Node3D", false, false, false);

			CHECK(candidate.is_valid("N3"));
		}

		SUBCASE("Should be invalid for query with non-word boundary matches.") {
			CreateDialogCandidate candidate("AnimationPlayer", false, false, false);

			CHECK(!candidate.is_valid("apl"));
		}

		SUBCASE("Should be valid with subsequence match on word boundary characters.") {
			CreateDialogCandidate candidate("CharacterBody3D", false, false, false);

			CHECK(candidate.is_valid("cbd"));
		}

		SUBCASE("Should be invalid without word boundary overlap.") {
			CreateDialogCandidate candidate("AnimationPlayer", false, false, false);

			CHECK(!candidate.is_valid("ai"));
		}
	}
}

TEST_SUITE("[Candidate] Score") {
	TEST_CASE("Regular query") {
		SUBCASE("Candidate with matching query should have maximum score.") {
			CreateDialogCandidate candidate("Node", false, false, false);

			CHECK(candidate.compute_score("node") == 1.0);
		}

		SUBCASE("Candidate which was a query match earlier in the string should score higher.") {
			CreateDialogCandidate candidate("NodeGraph", false, false, false);
			CreateDialogCandidate candidate2("GraphNode", false, false, false);

			String query = "node";
			CHECK(candidate.compute_score(query) > candidate2.compute_score(query));
		}

		SUBCASE("Candidate with shorter name should score higher.") {
			CreateDialogCandidate candidate("Path2D", false, false, false);
			CreateDialogCandidate candidate2("PathFollow2D", false, false, false);

			String query = "path";
			CHECK(candidate.compute_score(query) > candidate2.compute_score(query));
		}
	}

	TEST_CASE("Word boundary query") {
		SUBCASE("Substring match in the middle should score lower than word boundary match.") {
			CreateDialogCandidate candidate("CheckBox", false, false, false);
			CreateDialogCandidate candidate2("StaticBody", false, false, false);

			String query = "cb";
			CHECK(candidate.compute_score(query) > candidate2.compute_score(query));
		}
	}

	TEST_CASE("Secondary features") {
		SUBCASE("Candidate with preferred type, should score higher.") {
			CreateDialogCandidate candidate("Node3D", true, false, false);
			CreateDialogCandidate candidate2("Node2D", false, false, false);

			String query = "node";
			CHECK(candidate.compute_score(query) > candidate2.compute_score(query));
		}

		SUBCASE("Candidate which is a favorite should score higher.") {
			CreateDialogCandidate candidate("Node3D", false, true, false);
			CreateDialogCandidate candidate2("Node2D", false, false, false);

			String query = "node";
			CHECK(candidate.compute_score(query) > candidate2.compute_score(query));
		}

		SUBCASE("Candidate which was recently created should score higher.") {
			CreateDialogCandidate candidate("Node3D", false, false, true);
			CreateDialogCandidate candidate2("Node2D", false, false, false);

			String query = "node";
			CHECK(candidate.compute_score(query) > candidate2.compute_score(query));
		}
	}
}

#endif // TEST_CREATE_DIALOG_CANDIDATE_H
