/**************************************************************************/
/*  gdscript_coverage.cpp                                                 */
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

#ifdef TOOLS_ENABLED

#include "gdscript.h"
#include "gdscript_function.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/string/print_string.h"

/*************** Configuration setters ***************/

void GDScriptLanguage::coverage_set_output(const String &p_path) {
	coverage_output_path = p_path;
}

void GDScriptLanguage::coverage_set_mode(const String &p_mode) {
	coverage_mode = (p_mode == "count") ? COVERAGE_MODE_COUNT : COVERAGE_MODE_SET;
}

void GDScriptLanguage::coverage_set_format(const String &p_format) {
	if (p_format == "cobertura") {
		coverage_format = COVERAGE_FORMAT_COBERTURA;
	} else if (p_format == "json") {
		coverage_format = COVERAGE_FORMAT_JSON;
	} else if (p_format == "text") {
		coverage_format = COVERAGE_FORMAT_TEXT;
	} else {
		coverage_format = COVERAGE_FORMAT_LCOV;
	}
}

void GDScriptLanguage::coverage_set_threshold(float p_pct) {
	coverage_threshold = p_pct;
}

void GDScriptLanguage::coverage_add_include(const String &p_glob) {
	coverage_include.push_back(p_glob);
}

void GDScriptLanguage::coverage_add_exclude(const String &p_glob) {
	coverage_exclude.push_back(p_glob);
}

/*************** Runtime control ***************/

void GDScriptLanguage::coverage_start() {
	coverage_hits.clear();
	coverage_func_hits.clear();
	coverage_branch_hits.clear();
	coverage_written = false;
	coverage_enabled.set();
}

/*************** Path and filter helpers ***************/

static bool _coverage_path_included(const String &p_res_path,
		const Vector<String> &p_include, const Vector<String> &p_exclude) {
	for (const String &pat : p_exclude) {
		if (p_res_path.match(pat)) {
			return false;
		}
	}
	if (p_include.is_empty()) {
		return true;
	}
	for (const String &pat : p_include) {
		if (p_res_path.match(pat)) {
			return true;
		}
	}
	return false;
}

// Convert res:// path to an absolute filesystem path.
// Falls back to resource dir prefix when ProjectSettings is unavailable (e.g. engine --test mode).
static String _coverage_globalize(const String &p_res_path) {
	if (ProjectSettings::get_singleton()) {
		String abs = ProjectSettings::get_singleton()->globalize_path(p_res_path);
		if (!abs.begins_with("res://")) {
			return abs;
		}
	}
	return OS::get_singleton()->get_resource_dir().path_join(p_res_path.substr(6));
}

/*************** Hot-path recording (called from VM) ***************/

void GDScriptLanguage::coverage_record_line(const StringName &p_source, int p_line) {
	String path = String(p_source);
	if (!_coverage_path_included(path, coverage_include, coverage_exclude)) {
		return;
	}
	MutexLock lock(coverage_mutex);
	HashMap<int, int> &lines = coverage_hits[path];
	if (coverage_mode == COVERAGE_MODE_COUNT) {
		lines[p_line]++;
	} else {
		lines[p_line] = 1;
	}
}

void GDScriptLanguage::coverage_record_func_entry(const StringName &p_source, const StringName &p_func) {
	String path = String(p_source);
	if (!_coverage_path_included(path, coverage_include, coverage_exclude)) {
		return;
	}
	MutexLock lock(coverage_mutex);
	HashMap<String, int> &funcs = coverage_func_hits[path];
	if (coverage_mode == COVERAGE_MODE_COUNT) {
		funcs[String(p_func)]++;
	} else {
		funcs[String(p_func)] = 1;
	}
}

void GDScriptLanguage::coverage_record_branch(const StringName &p_source, int p_line, int p_ip, bool p_taken) {
	String path = String(p_source);
	if (!_coverage_path_included(path, coverage_include, coverage_exclude)) {
		return;
	}
	MutexLock lock(coverage_mutex);
	BranchResult &br = coverage_branch_hits[path][p_ip];
	br.line = p_line; // store for LCOV BRDA output
	if (p_taken) {
		if (coverage_mode == COVERAGE_MODE_COUNT) {
			br.taken++;
		} else {
			br.taken = 1;
		}
	} else {
		if (coverage_mode == COVERAGE_MODE_COUNT) {
			br.not_taken++;
		} else {
			br.not_taken = 1;
		}
	}
}

/*************** Per-file stats aggregate ***************/

// Aggregate coverage counts for a single source file.
// Used by _gather_file_stats to accumulate line/function/branch totals that
// are then consumed by the format writers and the threshold/summary logic.
struct CoverageFileStats {
	int lines = 0, hit_lines = 0;
	int funcs = 0, hit_funcs = 0;
	int branches = 0, hit_branches = 0;
};

// Accumulate line coverage counts into r_fs.
// p_hits contains recorded hit counts; p_coverable (optional) adds lines that
// were compiled but never reached so they appear as not-hit rather than absent.
static void _compute_line_stats(const HashMap<int, int> &p_hits,
		const HashMap<int, int> *p_coverable, CoverageFileStats &r_fs) {
	HashSet<int> counted;
	for (const KeyValue<int, int> &lv : p_hits) {
		r_fs.lines++;
		if (lv.value > 0) {
			r_fs.hit_lines++;
		}
		counted.insert(lv.key);
	}
	// Count coverable-but-not-hit lines not already in p_hits.
	if (p_coverable) {
		for (const KeyValue<int, int> &cv : *p_coverable) {
			if (!counted.has(cv.key)) {
				r_fs.lines++;
			}
		}
	}
}

// Accumulate function coverage counts from recorded hits into r_fs.
static void _compute_func_stats(const HashMap<String, int> &p_funcs, CoverageFileStats &r_fs) {
	for (const KeyValue<String, int> &fv : p_funcs) {
		r_fs.funcs++;
		if (fv.value > 0) {
			r_fs.hit_funcs++;
		}
	}
}

// Accumulate branch coverage counts from recorded results into r_fs.
// Each IP contributes two branches (taken + not_taken); each counts as hit if > 0.
static void _compute_branch_stats(const HashMap<int, GDScriptLanguage::BranchResult> &p_branches, CoverageFileStats &r_fs) {
	for (const KeyValue<int, GDScriptLanguage::BranchResult> &bv : p_branches) {
		r_fs.branches += 2;
		if (bv.value.taken > 0) {
			r_fs.hit_branches++;
		}
		if (bv.value.not_taken > 0) {
			r_fs.hit_branches++;
		}
	}
}

// Add stats for files compiled and filtered in but never executed at all.
static void _add_unexecuted_file_stats(
		HashMap<String, CoverageFileStats> &r_out,
		const HashMap<String, HashMap<int, int>> &p_coverable) {
	static const HashMap<int, int> empty_hits;
	for (const KeyValue<String, HashMap<int, int>> &cv : p_coverable) {
		if (!r_out.has(cv.key)) {
			_compute_line_stats(empty_hits, &cv.value, r_out[cv.key]);
		}
	}
}

// Count compiled functions absent from p_func_hits so FNF reflects all compiled
// functions, not just those called at least once. Only processes files present in
// p_coverable (i.e. that passed the include/exclude filter).
static void _add_uncalled_func_stats(
		HashMap<String, CoverageFileStats> &r_out,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<String, int>> &p_func_starts,
		const HashMap<String, HashMap<int, int>> *p_coverable) {
	for (const KeyValue<String, HashMap<String, int>> &sv : p_func_starts) {
		if (p_coverable && !p_coverable->has(sv.key)) {
			continue;
		}
		const HashMap<String, int> *fh = p_func_hits.getptr(sv.key);
		for (const KeyValue<String, int> &fv : sv.value) {
			if (!fh || !fh->has(fv.key)) {
				r_out[sv.key].funcs++;
			}
		}
	}
}

// Build per-file coverage stats from the three recorded hit maps.
// When p_coverable is provided, files that were compiled but never executed are
// included as 0%-covered rather than absent. When p_func_starts is provided,
// compiled functions that were never called are counted toward FNF so that the
// reported total matches the number of functions in the bytecode.
static HashMap<String, CoverageFileStats> _gather_file_stats(
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		const HashMap<String, HashMap<int, int>> *p_coverable = nullptr,
		const HashMap<String, HashMap<String, int>> *p_func_starts = nullptr) {
	HashMap<String, CoverageFileStats> out;
	for (const KeyValue<String, HashMap<int, int>> &kv : p_hits) {
		const HashMap<int, int> *cov = p_coverable ? p_coverable->getptr(kv.key) : nullptr;
		_compute_line_stats(kv.value, cov, out[kv.key]);
	}
	for (const KeyValue<String, HashMap<String, int>> &kv : p_func_hits) {
		_compute_func_stats(kv.value, out[kv.key]);
	}
	for (const KeyValue<String, HashMap<int, GDScriptLanguage::BranchResult>> &kv : p_branch_hits) {
		_compute_branch_stats(kv.value, out[kv.key]);
	}
	if (p_coverable) {
		_add_unexecuted_file_stats(out, *p_coverable);
	}
	if (p_func_starts) {
		_add_uncalled_func_stats(out, p_func_hits, *p_func_starts, p_coverable);
	}
	return out;
}

// Sum per-file stats into a single aggregate for threshold checking and summary output.
static CoverageFileStats _sum_totals(const HashMap<String, CoverageFileStats> &p_stats) {
	CoverageFileStats t;
	for (const KeyValue<String, CoverageFileStats> &kv : p_stats) {
		t.lines += kv.value.lines;
		t.hit_lines += kv.value.hit_lines;
		t.funcs += kv.value.funcs;
		t.hit_funcs += kv.value.hit_funcs;
		t.branches += kv.value.branches;
		t.hit_branches += kv.value.hit_branches;
	}
	return t;
}

/*************** Threshold and summary ***************/

bool GDScriptLanguage::coverage_check_threshold() {
	if (coverage_threshold <= 0.0f) {
		return true;
	}
	// Snapshot under coverage_mutex so we don't race with ongoing recording.
	HashMap<String, HashMap<int, int>> hits_snap;
	HashMap<String, HashMap<String, int>> func_hits_snap;
	HashMap<String, HashMap<int, BranchResult>> branch_hits_snap;
	{
		MutexLock lock(coverage_mutex);
		hits_snap = coverage_hits;
		func_hits_snap = coverage_func_hits;
		branch_hits_snap = coverage_branch_hits;
	}
	HashMap<String, HashMap<int, int>> coverable = _coverage_enumerate_coverable_lines();
	HashMap<String, HashMap<String, int>> func_starts = _coverage_enumerate_func_start_lines();
	HashMap<String, CoverageFileStats> stats = _gather_file_stats(hits_snap, func_hits_snap, branch_hits_snap, &coverable, &func_starts);
	CoverageFileStats t = _sum_totals(stats);
	if (t.lines == 0) {
		return true;
	}
	float pct = 100.0f * (float)t.hit_lines / (float)t.lines;
	return pct >= coverage_threshold;
}

static String _format_pct_cell(float p_pct, int p_width) {
	String s = String::num(p_pct, 1) + "%";
	return s.rpad(p_width);
}

static String _format_summary_row(const String &p_label, const CoverageFileStats &p_fs, int p_col_file) {
	float lp = p_fs.lines > 0 ? 100.0f * p_fs.hit_lines / p_fs.lines : 0.0f;
	float fp = p_fs.funcs > 0 ? 100.0f * p_fs.hit_funcs / p_fs.funcs : 0.0f;
	float bp = p_fs.branches > 0 ? 100.0f * p_fs.hit_branches / p_fs.branches : 0.0f;
	return p_label.rpad(p_col_file) + _format_pct_cell(lp, 9) + _format_pct_cell(fp, 9) + _format_pct_cell(bp, 9) + "\n";
}

// Build the human-readable summary string from pre-computed per-file stats.
// Separated from coverage_summary_string() so coverage_write() can pass the
// already-snapshotted maps for consistency instead of taking a second snapshot.
static String _format_coverage_summary(const HashMap<String, CoverageFileStats> &p_stats, float p_threshold) {
	static const int COL_FILE = 40;
	CoverageFileStats totals = _sum_totals(p_stats);

	String out = String("File").rpad(COL_FILE) + String("Lines").rpad(9) + String("Funcs").rpad(9) + String("Branches").rpad(9) + "\n";

	Vector<String> files;
	for (const KeyValue<String, CoverageFileStats> &kv : p_stats) {
		files.push_back(kv.key);
	}
	files.sort();
	for (const String &path : files) {
		out += _format_summary_row(path, *p_stats.getptr(path), COL_FILE);
	}

	String sep(U"────────────────────────────────────────────────────────────");
	out += sep + "\n";
	out += _format_summary_row("Total", totals, COL_FILE);

	if (p_threshold > 0.0f) {
		float pct = totals.lines > 0 ? 100.0f * totals.hit_lines / totals.lines : 0.0f;
		bool pass = pct >= p_threshold;
		out += "Threshold: " + String::num(p_threshold, 1) + "% " + (pass ? U"✓" : U"✗") + "\n";
	}
	return out;
}

String GDScriptLanguage::coverage_summary_string() {
	// Snapshot under coverage_mutex so we don't race with ongoing recording.
	HashMap<String, HashMap<int, int>> hits_snap;
	HashMap<String, HashMap<String, int>> func_hits_snap;
	HashMap<String, HashMap<int, BranchResult>> branch_hits_snap;
	{
		MutexLock lock(coverage_mutex);
		hits_snap = coverage_hits;
		func_hits_snap = coverage_func_hits;
		branch_hits_snap = coverage_branch_hits;
	}
	HashMap<String, HashMap<int, int>> coverable = _coverage_enumerate_coverable_lines();
	HashMap<String, HashMap<String, int>> func_starts = _coverage_enumerate_func_start_lines();
	HashMap<String, CoverageFileStats> stats = _gather_file_stats(hits_snap, func_hits_snap, branch_hits_snap, &coverable, &func_starts);
	return _format_coverage_summary(stats, coverage_threshold);
}

/*************** Bytecode enumeration ***************/

void GDScriptLanguage::_coverage_collect_lines(GDScript *p_script, HashMap<int, int> &r_lines) {
	if (!p_script) {
		return;
	}
	// Scan member functions for OPCODE_LINE entries to find coverable lines.
	for (const KeyValue<StringName, GDScriptFunction *> &kv : p_script->get_member_functions()) {
		GDScriptFunction *func = kv.value;
		if (!func) {
			continue;
		}
		const int *code = func->_code_ptr;
		int code_size = func->_code_size;
		for (int i = 0; i < code_size; i++) {
			if (code[i] == GDScriptFunction::OPCODE_LINE) {
				if (i + 1 < code_size) {
					int ln = code[i + 1];
					if (!r_lines.has(ln)) {
						r_lines[ln] = 0;
					}
				}
				i += 1; // skip line-number operand
			}
		}
	}
	// Recurse into inner classes.
	for (const KeyValue<StringName, Ref<GDScript>> &kv : p_script->get_subclasses()) {
		_coverage_collect_lines(kv.value.ptr(), r_lines);
	}
}

// Walk bytecode to find the first OPCODE_LINE for each function (its start line).
// p_class_prefix is the dot-separated inner-class chain (e.g. "Outer.Inner") for subclasses,
// so that same-named methods in different inner classes get distinct keys.
void GDScriptLanguage::_coverage_collect_func_starts(const GDScript *p_script, HashMap<String, int> &r_starts, const String &p_class_prefix) {
	if (!p_script) {
		return;
	}
	for (const KeyValue<StringName, GDScriptFunction *> &kv : p_script->get_member_functions()) {
		GDScriptFunction *func = kv.value;
		if (!func) {
			continue;
		}
		const int *code = func->_code_ptr;
		int code_size = func->_code_size;
		for (int i = 0; i < code_size; i++) {
			if (code[i] == GDScriptFunction::OPCODE_LINE && i + 1 < code_size) {
				String key = p_class_prefix.is_empty() ? String(func->get_name()) : p_class_prefix + "." + String(func->get_name());
				r_starts[key] = code[i + 1];
				break; // only the first line number is the start line
			}
		}
	}
	for (const KeyValue<StringName, Ref<GDScript>> &kv : p_script->get_subclasses()) {
		String child_prefix = p_class_prefix.is_empty() ? String(kv.key) : p_class_prefix + "." + String(kv.key);
		_coverage_collect_func_starts(kv.value.ptr(), r_starts, child_prefix);
	}
}

// Build a per-file map of function name → start line from script_list.
// Applies the same include/exclude filter as _coverage_enumerate_coverable_lines
// so callers receive a consistent view of which files are in scope.
HashMap<String, HashMap<String, int>> GDScriptLanguage::_coverage_enumerate_func_start_lines() {
	HashMap<String, HashMap<String, int>> result;
	MutexLock lock(mutex);
	const SelfList<GDScript> *s = script_list.first();
	while (s) {
		const GDScript *scr = s->self();
		if (scr) {
			String res_path = scr->get_script_path();
			if (_coverage_path_included(res_path, coverage_include, coverage_exclude)) {
				_coverage_collect_func_starts(scr, result[res_path]);
			}
		}
		s = s->next();
	}
	return result;
}

HashMap<String, HashMap<int, int>> GDScriptLanguage::_coverage_enumerate_coverable_lines() {
	HashMap<String, HashMap<int, int>> coverable;
	MutexLock lock(mutex);
	const SelfList<GDScript> *s = script_list.first();
	while (s) {
		GDScript *scr = s->self();
		if (scr) {
			String res_path = scr->get_script_path();
			if (_coverage_path_included(res_path, coverage_include, coverage_exclude)) {
				_coverage_collect_lines(scr, coverable[res_path]);
			}
		}
		s = s->next();
	}
	return coverable;
}

// Merge recorded hits with coverable-but-not-hit lines and return sorted line numbers.
static Vector<int> _merge_line_hits(const HashMap<int, int> &p_hits,
		const HashMap<String, HashMap<int, int>> &p_coverable, const String &p_path,
		HashMap<int, int> &r_all_lines) {
	r_all_lines = p_hits;
	if (p_coverable.has(p_path)) {
		for (const KeyValue<int, int> &cv : p_coverable[p_path]) {
			if (!r_all_lines.has(cv.key)) {
				r_all_lines[cv.key] = 0;
			}
		}
	}
	Vector<int> sorted_lines;
	for (const KeyValue<int, int> &lv : r_all_lines) {
		sorted_lines.push_back(lv.key);
	}
	sorted_lines.sort();
	return sorted_lines;
}

// Build a sorted list of all file paths appearing in any of the four coverage maps.
// Used by all format writers to ensure files with only function or branch events
// (but no line hits) are not silently dropped from the output.
static Vector<String> _build_sorted_path_union(
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		const HashMap<String, HashMap<int, int>> &p_coverable) {
	HashSet<String> s;
	for (const KeyValue<String, HashMap<int, int>> &kv : p_hits) {
		s.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<String, int>> &kv : p_func_hits) {
		s.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<int, GDScriptLanguage::BranchResult>> &kv : p_branch_hits) {
		s.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<int, int>> &cv : p_coverable) {
		s.insert(cv.key);
	}
	Vector<String> result;
	for (const String &path : s) {
		result.push_back(path);
	}
	result.sort();
	return result;
}

/*************** LCOV writer ***************/

static void _lcov_write_functions(Ref<FileAccess> f, const HashMap<String, int> &p_funcs,
		const HashMap<String, int> &p_starts) {
	// Merge: start with all compiled functions (p_starts), then overlay recorded hits (p_funcs).
	// Functions compiled but never called appear with FNDA:0. FN/FNDA are sorted for determinism.
	HashMap<String, int> all_funcs;
	for (const KeyValue<String, int> &sv : p_starts) {
		all_funcs[sv.key] = 0;
	}
	for (const KeyValue<String, int> &fv : p_funcs) {
		all_funcs[fv.key] = fv.value;
	}
	Vector<String> sorted_funcs;
	for (const KeyValue<String, int> &fv : all_funcs) {
		sorted_funcs.push_back(fv.key);
	}
	sorted_funcs.sort();
	for (const String &fn : sorted_funcs) {
		const int *ln = p_starts.getptr(fn);
		f->store_line("FN:" + itos(ln ? *ln : 0) + "," + fn); // 0 = unknown start line (e.g. lambda)
	}
	int fnh = 0;
	for (const String &fn : sorted_funcs) {
		int cnt = *all_funcs.getptr(fn);
		f->store_line("FNDA:" + itos(cnt) + "," + fn);
		if (cnt > 0) {
			fnh++;
		}
	}
	f->store_line("FNF:" + itos(all_funcs.size()));
	f->store_line("FNH:" + itos(fnh));
}

static void _lcov_write_branches(Ref<FileAccess> f,
		const HashMap<int, GDScriptLanguage::BranchResult> &p_branches,
		int &r_brf, int &r_brh) {
	// Sort by IP for deterministic output across runs.
	Vector<int> sorted_ips;
	for (const KeyValue<int, GDScriptLanguage::BranchResult> &bv : p_branches) {
		sorted_ips.push_back(bv.key);
	}
	sorted_ips.sort();
	for (int ip : sorted_ips) {
		const GDScriptLanguage::BranchResult &br = *p_branches.getptr(ip);
		int ln = br.line;
		// BRDA format: line, block (ip), branch (0=taken/1=not_taken), count
		f->store_line("BRDA:" + itos(ln) + "," + itos(ip) + ",0," + itos(br.taken));
		f->store_line("BRDA:" + itos(ln) + "," + itos(ip) + ",1," + itos(br.not_taken));
		r_brf += 2;
		if (br.taken > 0) {
			r_brh++;
		}
		if (br.not_taken > 0) {
			r_brh++;
		}
	}
}

// Emit DA (line hit) records for all lines in p_sorted_lines, followed by LF/LH totals.
static void _lcov_write_lines(Ref<FileAccess> f, const Vector<int> &p_sorted_lines,
		const HashMap<int, int> &p_all_lines) {
	int lf = 0, lh = 0;
	for (int ln : p_sorted_lines) {
		int hits = *p_all_lines.getptr(ln);
		f->store_line("DA:" + itos(ln) + "," + itos(hits));
		lf++;
		if (hits > 0) {
			lh++;
		}
	}
	f->store_line("LF:" + itos(lf));
	f->store_line("LH:" + itos(lh));
}

// Write one complete LCOV record (TN/SF/FN*/FNDA*/FNF/FNH/BRDA*/BRF/BRH/DA*/LF/LH/end_of_record)
// for a single source file. Merges runtime hits with compile-time coverable data so that
// unexecuted lines and functions appear with count 0 rather than being omitted.
static void _lcov_write_file_record(Ref<FileAccess> f, const String &p_res_path,
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		const HashMap<String, HashMap<int, int>> &p_coverable,
		const HashMap<String, HashMap<String, int>> &p_func_starts) {
	static const HashMap<int, int> empty_hits;
	static const HashMap<String, int> empty_funcs;
	const HashMap<int, int> *hits_ptr = p_hits.getptr(p_res_path);
	f->store_line("TN:");
	f->store_line("SF:" + _coverage_globalize(p_res_path));

	const HashMap<String, int> *funcs = p_func_hits.getptr(p_res_path);
	const HashMap<String, int> *starts = p_func_starts.getptr(p_res_path);
	if (funcs || starts) {
		_lcov_write_functions(f, funcs ? *funcs : empty_funcs, starts ? *starts : empty_funcs);
	}

	int brf = 0, brh = 0;
	const HashMap<int, GDScriptLanguage::BranchResult> *branches = p_branch_hits.getptr(p_res_path);
	if (branches) {
		_lcov_write_branches(f, *branches, brf, brh);
		f->store_line("BRF:" + itos(brf));
		f->store_line("BRH:" + itos(brh));
	}

	HashMap<int, int> all_lines;
	Vector<int> sorted = _merge_line_hits(hits_ptr ? *hits_ptr : empty_hits, p_coverable, p_res_path, all_lines);
	_lcov_write_lines(f, sorted, all_lines);
	f->store_line("end_of_record");
}

static Error _write_lcov(const String &p_path,
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		GDScriptLanguage *p_lang) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open coverage output: " + p_path);

	HashMap<String, HashMap<int, int>> coverable = p_lang->_coverage_enumerate_coverable_lines();
	HashMap<String, HashMap<String, int>> func_starts = p_lang->_coverage_enumerate_func_start_lines();
	Vector<String> sorted_paths = _build_sorted_path_union(p_hits, p_func_hits, p_branch_hits, coverable);
	for (const String &res_path : sorted_paths) {
		_lcov_write_file_record(f, res_path, p_hits, p_func_hits, p_branch_hits, coverable, func_starts);
	}
	return OK;
}

/*************** Cobertura XML writer ***************/

static String _xml_escape(const String &p_str) {
	return p_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;");
}

// Build the merged all_methods map: runtime hits from p_funcs, compile-time entries
// from p_func_starts for functions never called (hits = 0).
static HashMap<String, int> _cobertura_build_all_methods(
		const HashMap<String, int> *p_funcs,
		const HashMap<String, int> *p_func_starts) {
	HashMap<String, int> all;
	if (p_funcs) {
		for (const KeyValue<String, int> &fv : *p_funcs) {
			all[fv.key] = fv.value;
		}
	}
	if (p_func_starts) {
		for (const KeyValue<String, int> &sv : *p_func_starts) {
			if (!all.has(sv.key)) {
				all[sv.key] = 0;
			}
		}
	}
	return all;
}

// Emit the <methods> XML block from a merged method→hits map. No-ops if empty.
static void _cobertura_emit_methods(Ref<FileAccess> f, const HashMap<String, int> &p_all_methods) {
	if (p_all_methods.is_empty()) {
		return;
	}
	Vector<String> sorted;
	for (const KeyValue<String, int> &mv : p_all_methods) {
		sorted.push_back(mv.key);
	}
	sorted.sort();
	f->store_line("      <methods>");
	for (const String &fn : sorted) {
		f->store_line(vformat("        <method name=\"%s\" hits=\"%d\"/>", _xml_escape(fn), *p_all_methods.getptr(fn)));
	}
	f->store_line("      </methods>");
}

// Index branch results by source line number.
// p_branches is keyed by VM instruction pointer; br.line maps each to its source line.
static HashMap<int, Vector<const GDScriptLanguage::BranchResult *>> _cobertura_build_line_branch_index(
		const HashMap<int, GDScriptLanguage::BranchResult> *p_branches) {
	HashMap<int, Vector<const GDScriptLanguage::BranchResult *>> index;
	if (p_branches) {
		for (const KeyValue<int, GDScriptLanguage::BranchResult> &bv : *p_branches) {
			index[bv.value.line].push_back(&bv.value);
		}
	}
	return index;
}

// Emit one <line> element with branch annotation when branch data is present.
static void _cobertura_write_line_element(Ref<FileAccess> f, int p_ln, int p_hits,
		const Vector<const GDScriptLanguage::BranchResult *> *p_lbrs) {
	if (!p_lbrs || p_lbrs->is_empty()) {
		f->store_line(vformat("        <line number=\"%d\" hits=\"%d\" branch=\"false\"/>", p_ln, p_hits));
		return;
	}
	int total = 0, covered = 0;
	for (const GDScriptLanguage::BranchResult *br : *p_lbrs) {
		total += 2;
		if (br->taken > 0) {
			covered++;
		}
		if (br->not_taken > 0) {
			covered++;
		}
	}
	int pct = total > 0 ? 100 * covered / total : 0;
	f->store_line(vformat("        <line number=\"%d\" hits=\"%d\" branch=\"true\" condition-coverage=\"%d%% (%d/%d)\"/>",
			p_ln, p_hits, pct, covered, total));
}

// Emit the <lines> block for all coverable lines (hit and not-hit).
static void _cobertura_write_lines_block(Ref<FileAccess> f,
		const HashMap<int, int> &p_lines,
		const HashMap<int, int> *p_coverable,
		const HashMap<int, Vector<const GDScriptLanguage::BranchResult *>> &p_line_branches) {
	HashMap<int, int> all_lines(p_lines);
	if (p_coverable) {
		for (const KeyValue<int, int> &cv : *p_coverable) {
			if (!all_lines.has(cv.key)) {
				all_lines[cv.key] = 0;
			}
		}
	}
	Vector<int> lnums;
	for (const KeyValue<int, int> &lv : all_lines) {
		lnums.push_back(lv.key);
	}
	lnums.sort();
	f->store_line("      <lines>");
	for (int ln : lnums) {
		_cobertura_write_line_element(f, ln, *all_lines.getptr(ln), p_line_branches.getptr(ln));
	}
	f->store_line("      </lines>");
}

static void _cobertura_write_class(Ref<FileAccess> f, const String &p_res_path,
		const HashMap<int, int> &p_lines,
		const HashMap<String, int> *p_funcs,
		const HashMap<int, GDScriptLanguage::BranchResult> *p_branches,
		const HashMap<int, int> *p_coverable,
		const HashMap<String, int> *p_func_starts = nullptr) {
	CoverageFileStats fs;
	_compute_line_stats(p_lines, p_coverable, fs);
	if (p_funcs) {
		_compute_func_stats(*p_funcs, fs);
	}
	if (p_branches) {
		_compute_branch_stats(*p_branches, fs);
	}
	float flr = fs.lines > 0 ? (float)fs.hit_lines / fs.lines : 0.0f;
	float fbr = fs.branches > 0 ? (float)fs.hit_branches / fs.branches : 0.0f;

	// DTD requires name= (class identifier) and filename= (absolute filesystem path).
	String class_name = _xml_escape(p_res_path.get_file().get_basename());
	f->store_line(vformat("    <class name=\"%s\" filename=\"%s\" line-rate=\"%.4f\" branch-rate=\"%.4f\">",
			class_name, _xml_escape(_coverage_globalize(p_res_path)), flr, fbr));

	HashMap<String, int> all_methods = _cobertura_build_all_methods(p_funcs, p_func_starts);
	_cobertura_emit_methods(f, all_methods);

	HashMap<int, Vector<const GDScriptLanguage::BranchResult *>> line_branches = _cobertura_build_line_branch_index(p_branches);
	_cobertura_write_lines_block(f, p_lines, p_coverable, line_branches);

	f->store_line("    </class>");
}

static Error _write_cobertura(const String &p_path,
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		GDScriptLanguage *p_lang) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open coverage output: " + p_path);

	HashMap<String, HashMap<int, int>> coverable = p_lang->_coverage_enumerate_coverable_lines();
	HashMap<String, HashMap<String, int>> func_starts = p_lang->_coverage_enumerate_func_start_lines();
	HashMap<String, CoverageFileStats> stats = _gather_file_stats(p_hits, p_func_hits, p_branch_hits, &coverable, &func_starts);
	CoverageFileStats t = _sum_totals(stats);
	float line_rate = t.lines > 0 ? (float)t.hit_lines / t.lines : 0.0f;
	float branch_rate = t.branches > 0 ? (float)t.hit_branches / t.branches : 0.0f;
	uint64_t ts = Time::get_singleton()->get_unix_time_from_system();

	f->store_line("<?xml version=\"1.0\" ?>");
	f->store_line("<!DOCTYPE coverage SYSTEM \"http://cobertura.sourceforge.net/xml/coverage-04.dtd\">");
	f->store_line(vformat("<coverage version=\"5.0\" timestamp=\"%d\" line-rate=\"%.4f\" branch-rate=\"%.4f\" lines-covered=\"%d\" lines-valid=\"%d\" branches-covered=\"%d\" branches-valid=\"%d\">",
			(int64_t)ts, line_rate, branch_rate, t.hit_lines, t.lines, t.hit_branches, t.branches));
	f->store_line(vformat("  <packages><package name=\"gdscript\" line-rate=\"%.4f\" branch-rate=\"%.4f\" complexity=\"0\"><classes>",
			line_rate, branch_rate));

	Vector<String> sorted_cobertura = _build_sorted_path_union(p_hits, p_func_hits, p_branch_hits, coverable);
	static const HashMap<int, int> empty_cob_hits;
	for (const String &res_path : sorted_cobertura) {
		const HashMap<int, int> *lines = p_hits.getptr(res_path);
		_cobertura_write_class(f, res_path, lines ? *lines : empty_cob_hits,
				p_func_hits.getptr(res_path),
				p_branch_hits.getptr(res_path),
				coverable.getptr(res_path),
				func_starts.getptr(res_path));
	}

	f->store_line("  </classes></package></packages>");
	f->store_line("</coverage>");
	return OK;
}

/*************** JSON writer ***************/

static void _json_write_lines(Ref<FileAccess> f, const HashMap<int, int> &p_lines,
		const HashMap<int, int> *p_coverable) {
	// Merge hit lines with coverable-but-not-hit lines for a complete picture.
	HashMap<int, int> all_lines;
	all_lines = p_lines;
	if (p_coverable) {
		for (const KeyValue<int, int> &cv : *p_coverable) {
			if (!all_lines.has(cv.key)) {
				all_lines[cv.key] = 0;
			}
		}
	}
	f->store_string("      \"lines\": {");
	Vector<int> lnums;
	for (const KeyValue<int, int> &lv : all_lines) {
		lnums.push_back(lv.key);
	}
	lnums.sort();
	bool first = true;
	for (int ln : lnums) {
		if (!first) {
			f->store_string(",");
		}
		first = false;
		f->store_string("\"" + itos(ln) + "\":" + itos(*all_lines.getptr(ln)));
	}
	f->store_line("},");
}

static void _json_write_funcs(Ref<FileAccess> f, const HashMap<String, int> *p_funcs,
		const HashMap<String, int> *p_func_starts) {
	// Merge: start with all compiled functions (p_func_starts), then overlay recorded hits.
	HashMap<String, int> all_funcs;
	if (p_func_starts) {
		for (const KeyValue<String, int> &kv : *p_func_starts) {
			all_funcs[kv.key] = 0;
		}
	}
	if (p_funcs) {
		for (const KeyValue<String, int> &kv : *p_funcs) {
			all_funcs[kv.key] = kv.value;
		}
	}
	f->store_string("      \"functions\": {");
	Vector<String> sorted_funcs;
	for (const KeyValue<String, int> &fv : all_funcs) {
		sorted_funcs.push_back(fv.key);
	}
	sorted_funcs.sort();
	bool first = true;
	for (const String &fn : sorted_funcs) {
		if (!first) {
			f->store_string(",");
		}
		first = false;
		f->store_string("\"" + fn.c_escape() + "\":" + itos(*all_funcs.getptr(fn)));
	}
	f->store_line("},");
}

static void _json_write_branches(Ref<FileAccess> f,
		const HashMap<int, GDScriptLanguage::BranchResult> &p_branches) {
	f->store_string("      \"branches\": {");
	// Sort by IP for deterministic output, matching the sorted output of _json_write_lines.
	Vector<int> sorted_ips;
	for (const KeyValue<int, GDScriptLanguage::BranchResult> &bv : p_branches) {
		sorted_ips.push_back(bv.key);
	}
	sorted_ips.sort();
	bool first = true;
	for (int ip : sorted_ips) {
		const GDScriptLanguage::BranchResult &br = *p_branches.getptr(ip);
		if (!first) {
			f->store_string(",");
		}
		first = false;
		f->store_string("\"" + itos(ip) + "_taken\":" + itos(br.taken) + ",");
		f->store_string("\"" + itos(ip) + "_not_taken\":" + itos(br.not_taken));
	}
	f->store_line("}");
}

static void _json_write_file_entry(Ref<FileAccess> f, const String &p_path,
		const HashMap<int, int> *p_lines,
		const HashMap<String, int> *p_funcs,
		const HashMap<int, GDScriptLanguage::BranchResult> *p_branches,
		const HashMap<int, int> *p_coverable,
		const HashMap<String, int> *p_func_starts) {
	f->store_line("    \"" + p_path.c_escape() + "\": {");
	if (p_lines || (p_coverable && !p_coverable->is_empty())) {
		static const HashMap<int, int> empty_json_lines;
		_json_write_lines(f, p_lines ? *p_lines : empty_json_lines, p_coverable);
	} else {
		f->store_line("      \"lines\": {},");
	}
	_json_write_funcs(f, p_funcs, p_func_starts);
	if (p_branches) {
		_json_write_branches(f, *p_branches);
	} else {
		f->store_line("      \"branches\": {}");
	}
	f->store_string("    }");
}

static Error _write_json(const String &p_path,
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		GDScriptLanguage *p_lang) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open coverage output: " + p_path);
	HashMap<String, HashMap<int, int>> coverable = p_lang->_coverage_enumerate_coverable_lines();
	HashMap<String, HashMap<String, int>> func_starts = p_lang->_coverage_enumerate_func_start_lines();

	Vector<String> file_list = _build_sorted_path_union(p_hits, p_func_hits, p_branch_hits, coverable);

	f->store_line("{");
	f->store_line("  \"files\": {");
	for (int i = 0; i < file_list.size(); i++) {
		const String &fp = file_list[i];
		_json_write_file_entry(f, fp, p_hits.getptr(fp), p_func_hits.getptr(fp), p_branch_hits.getptr(fp), coverable.getptr(fp), func_starts.getptr(fp));
		if (i + 1 < file_list.size()) {
			f->store_line(",");
		} else {
			f->store_line("");
		}
	}
	f->store_line("  },");

	HashMap<String, CoverageFileStats> stats = _gather_file_stats(p_hits, p_func_hits, p_branch_hits, &coverable, &func_starts);
	CoverageFileStats t = _sum_totals(stats);
	float lp = t.lines > 0 ? 100.0f * t.hit_lines / t.lines : 0.0f;
	float fp2 = t.funcs > 0 ? 100.0f * t.hit_funcs / t.funcs : 0.0f;
	float bp = t.branches > 0 ? 100.0f * t.hit_branches / t.branches : 0.0f;
	f->store_line(vformat("  \"summary\": {\"line_pct\": %.1f, \"func_pct\": %.1f, \"branch_pct\": %.1f}", lp, fp2, bp));
	f->store_line("}");
	return OK;
}

/*************** Text writer ***************/

static Error _write_text(const String &p_path, const String &p_summary) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open coverage output: " + p_path);
	f->store_string(p_summary);
	return OK;
}

/*************** Main write entry point ***************/

Error GDScriptLanguage::coverage_write() {
	if (coverage_output_path.is_empty()) {
		return ERR_UNCONFIGURED;
	}

	// Snapshot the coverage maps under the mutex so writers don't race with
	// concurrent recording still happening on other threads.
	HashMap<String, HashMap<int, int>> hits_snap;
	HashMap<String, HashMap<String, int>> func_hits_snap;
	HashMap<String, HashMap<int, BranchResult>> branch_hits_snap;
	{
		MutexLock lock(coverage_mutex);
		hits_snap = coverage_hits;
		func_hits_snap = coverage_func_hits;
		branch_hits_snap = coverage_branch_hits;
	}

	Error err = OK;
	switch (coverage_format) {
		case COVERAGE_FORMAT_LCOV:
			err = _write_lcov(coverage_output_path, hits_snap, func_hits_snap, branch_hits_snap, this);
			break;
		case COVERAGE_FORMAT_COBERTURA:
			err = _write_cobertura(coverage_output_path, hits_snap, func_hits_snap, branch_hits_snap, this);
			break;
		case COVERAGE_FORMAT_JSON:
			err = _write_json(coverage_output_path, hits_snap, func_hits_snap, branch_hits_snap, this);
			break;
		case COVERAGE_FORMAT_TEXT: {
			HashMap<String, HashMap<int, int>> coverable = _coverage_enumerate_coverable_lines();
			HashMap<String, HashMap<String, int>> func_starts = _coverage_enumerate_func_start_lines();
			HashMap<String, CoverageFileStats> stats = _gather_file_stats(hits_snap, func_hits_snap, branch_hits_snap, &coverable, &func_starts);
			String summary = _format_coverage_summary(stats, coverage_threshold);
			print_line(summary);
			err = _write_text(coverage_output_path, summary);
		} break;
	}

	if (err == OK) {
		coverage_written = true;
	}
	return err;
}

#endif // TOOLS_ENABLED
