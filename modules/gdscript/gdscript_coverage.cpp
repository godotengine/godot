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
	coverage_enabled = true;
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

struct CoverageFileStats {
	int lines = 0, hit_lines = 0;
	int funcs = 0, hit_funcs = 0;
	int branches = 0, hit_branches = 0;
};

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

static void _compute_func_stats(const HashMap<String, int> &p_funcs, CoverageFileStats &r_fs) {
	for (const KeyValue<String, int> &fv : p_funcs) {
		r_fs.funcs++;
		if (fv.value > 0) {
			r_fs.hit_funcs++;
		}
	}
}

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

static HashMap<String, CoverageFileStats> _gather_file_stats(
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		const HashMap<String, HashMap<int, int>> *p_coverable = nullptr) {
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
	// Files that were compiled and filtered-in but never executed at all must
	// appear in stats as 0% covered rather than being invisible.
	if (p_coverable) {
		static const HashMap<int, int> empty_hits;
		for (const KeyValue<String, HashMap<int, int>> &cv : *p_coverable) {
			if (!out.has(cv.key)) {
				_compute_line_stats(empty_hits, &cv.value, out[cv.key]);
			}
		}
	}
	return out;
}

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

bool GDScriptLanguage::coverage_check_threshold() const {
	if (coverage_threshold <= 0.0f) {
		return true;
	}
	HashMap<String, HashMap<int, int>> coverable = _coverage_enumerate_coverable_lines();
	HashMap<String, CoverageFileStats> stats = _gather_file_stats(coverage_hits, coverage_func_hits, coverage_branch_hits, &coverable);
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

String GDScriptLanguage::coverage_summary_string() const {
	static const int COL_FILE = 40;

	HashMap<String, HashMap<int, int>> coverable = _coverage_enumerate_coverable_lines();
	HashMap<String, CoverageFileStats> stats = _gather_file_stats(coverage_hits, coverage_func_hits, coverage_branch_hits, &coverable);
	CoverageFileStats totals = _sum_totals(stats);

	String out = String("File").rpad(COL_FILE) + String("Lines").rpad(9) + String("Funcs").rpad(9) + String("Branches").rpad(9) + "\n";

	Vector<String> files;
	for (const KeyValue<String, CoverageFileStats> &kv : stats) {
		files.push_back(kv.key);
	}
	files.sort();
	for (const String &path : files) {
		out += _format_summary_row(path, stats[path], COL_FILE);
	}

	String sep(U"────────────────────────────────────────────────────────────");
	out += sep + "\n";
	out += _format_summary_row("Total", totals, COL_FILE);

	if (coverage_threshold > 0.0f) {
		float pct = totals.lines > 0 ? 100.0f * totals.hit_lines / totals.lines : 0.0f;
		bool pass = pct >= coverage_threshold;
		out += "Threshold: " + String::num(coverage_threshold, 1) + "% " + (pass ? U"✓" : U"✗") + "\n";
	}
	return out;
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
void GDScriptLanguage::_coverage_collect_func_starts(const GDScript *p_script, HashMap<String, int> &r_starts) {
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
				r_starts[String(func->get_name())] = code[i + 1];
				break; // only the first line number is the start line
			}
		}
	}
	for (const KeyValue<StringName, Ref<GDScript>> &kv : p_script->get_subclasses()) {
		_coverage_collect_func_starts(kv.value.ptr(), r_starts);
	}
}

// Build a per-file map of function name → start line from script_list.
HashMap<String, HashMap<String, int>> GDScriptLanguage::_coverage_enumerate_func_start_lines() const {
	HashMap<String, HashMap<String, int>> result;
	const SelfList<GDScript> *s = script_list.first();
	while (s) {
		const GDScript *scr = s->self();
		if (scr) {
			String res_path = scr->get_script_path();
			_coverage_collect_func_starts(scr, result[res_path]);
		}
		s = s->next();
	}
	return result;
}

HashMap<String, HashMap<int, int>> GDScriptLanguage::_coverage_enumerate_coverable_lines() const {
	HashMap<String, HashMap<int, int>> coverable;
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

/*************** LCOV writer ***************/

static void _lcov_write_functions(Ref<FileAccess> f, const HashMap<String, int> &p_funcs,
		const HashMap<String, int> &p_starts) {
	for (const KeyValue<String, int> &fv : p_funcs) {
		const int *ln = p_starts.getptr(fv.key);
		f->store_line("FN:" + itos(ln ? *ln : 1) + "," + fv.key);
	}
	int fnh = 0;
	for (const KeyValue<String, int> &fv : p_funcs) {
		f->store_line("FNDA:" + itos(fv.value) + "," + fv.key);
		if (fv.value > 0) {
			fnh++;
		}
	}
	f->store_line("FNF:" + itos(p_funcs.size()));
	f->store_line("FNH:" + itos(fnh));
}

static void _lcov_write_branches(Ref<FileAccess> f,
		const HashMap<int, GDScriptLanguage::BranchResult> &p_branches,
		int &r_brf, int &r_brh) {
	for (const KeyValue<int, GDScriptLanguage::BranchResult> &bv : p_branches) {
		int ip = bv.key;
		int ln = bv.value.line;
		// BRDA format: line, block (ip), branch (0=taken/1=not_taken), count
		f->store_line("BRDA:" + itos(ln) + "," + itos(ip) + ",0," + itos(bv.value.taken));
		f->store_line("BRDA:" + itos(ln) + "," + itos(ip) + ",1," + itos(bv.value.not_taken));
		r_brf += 2;
		if (bv.value.taken > 0) {
			r_brh++;
		}
		if (bv.value.not_taken > 0) {
			r_brh++;
		}
	}
}

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

static Error _write_lcov(const String &p_path,
		const HashMap<String, HashMap<int, int>> &p_hits,
		const HashMap<String, HashMap<String, int>> &p_func_hits,
		const HashMap<String, HashMap<int, GDScriptLanguage::BranchResult>> &p_branch_hits,
		GDScriptLanguage *p_lang) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open coverage output: " + p_path);

	HashMap<String, HashMap<int, int>> coverable = p_lang->_coverage_enumerate_coverable_lines();
	HashMap<String, HashMap<String, int>> func_starts = p_lang->_coverage_enumerate_func_start_lines();

	// Sort the union of hit paths and coverable paths for deterministic output.
	// Files compiled but never executed must appear with DA:line,0 entries.
	HashSet<String> path_set;
	for (const KeyValue<String, HashMap<int, int>> &kv : p_hits) {
		path_set.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<int, int>> &cv : coverable) {
		path_set.insert(cv.key);
	}
	Vector<String> sorted_paths;
	for (const String &s : path_set) {
		sorted_paths.push_back(s);
	}
	sorted_paths.sort();

	static const HashMap<int, int> empty_hits;
	for (const String &res_path : sorted_paths) {
		const HashMap<int, int> *hits_ptr = p_hits.getptr(res_path);
		const HashMap<int, int> &lines_for_path = hits_ptr ? *hits_ptr : empty_hits;
		f->store_line("TN:");
		f->store_line("SF:" + _coverage_globalize(res_path));

		const HashMap<String, int> *funcs = p_func_hits.getptr(res_path);
		if (funcs) {
			static const HashMap<String, int> empty_starts;
			const HashMap<String, int> *starts = func_starts.getptr(res_path);
			_lcov_write_functions(f, *funcs, starts ? *starts : empty_starts);
		}

		int brf = 0, brh = 0;
		const HashMap<int, GDScriptLanguage::BranchResult> *branches = p_branch_hits.getptr(res_path);
		if (branches) {
			_lcov_write_branches(f, *branches, brf, brh);
			f->store_line("BRF:" + itos(brf));
			f->store_line("BRH:" + itos(brh));
		}

		HashMap<int, int> all_lines;
		Vector<int> sorted = _merge_line_hits(lines_for_path, coverable, res_path, all_lines);
		_lcov_write_lines(f, sorted, all_lines);

		f->store_line("end_of_record");
	}
	return OK;
}

/*************** Cobertura XML writer ***************/

static String _xml_escape(const String &p_str) {
	return p_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;");
}

static void _cobertura_write_class(Ref<FileAccess> f, const String &p_res_path,
		const HashMap<int, int> &p_lines,
		const HashMap<String, int> *p_funcs,
		const HashMap<int, GDScriptLanguage::BranchResult> *p_branches,
		const HashMap<int, int> *p_coverable) {
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

	f->store_line(vformat("    <class filename=\"%s\" line-rate=\"%.4f\" branch-rate=\"%.4f\">",
			_xml_escape(_coverage_globalize(p_res_path)), flr, fbr));

	if (p_funcs) {
		f->store_line("      <methods>");
		for (const KeyValue<String, int> &fv : *p_funcs) {
			f->store_line(vformat("        <method name=\"%s\" hits=\"%d\"/>", _xml_escape(fv.key), fv.value));
		}
		f->store_line("      </methods>");
	}

	// Emit all coverable lines (hit and not-hit).
	HashMap<int, int> all_lines;
	all_lines = p_lines;
	if (p_coverable) {
		for (const KeyValue<int, int> &cv : *p_coverable) {
			if (!all_lines.has(cv.key)) {
				all_lines[cv.key] = 0;
			}
		}
	}
	f->store_line("      <lines>");
	Vector<int> lnums;
	for (const KeyValue<int, int> &lv : all_lines) {
		lnums.push_back(lv.key);
	}
	lnums.sort();
	for (int ln : lnums) {
		f->store_line(vformat("        <line number=\"%d\" hits=\"%d\" branch=\"false\"/>", ln, *all_lines.getptr(ln)));
	}
	f->store_line("      </lines>");
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
	HashMap<String, CoverageFileStats> stats = _gather_file_stats(p_hits, p_func_hits, p_branch_hits, &coverable);
	CoverageFileStats t = _sum_totals(stats);
	float line_rate = t.lines > 0 ? (float)t.hit_lines / t.lines : 0.0f;
	float branch_rate = t.branches > 0 ? (float)t.hit_branches / t.branches : 0.0f;
	uint64_t ts = Time::get_singleton()->get_unix_time_from_system();

	f->store_line("<?xml version=\"1.0\" ?>");
	f->store_line("<!DOCTYPE coverage SYSTEM \"http://cobertura.sourceforge.net/xml/coverage-04.dtd\">");
	f->store_line(vformat("<coverage version=\"5.0\" timestamp=\"%d\" line-rate=\"%.4f\" branch-rate=\"%.4f\" lines-covered=\"%d\" lines-valid=\"%d\" branches-covered=\"%d\" branches-valid=\"%d\">",
			(int64_t)ts, line_rate, branch_rate, t.hit_lines, t.lines, t.hit_branches, t.branches));
	f->store_line("  <packages><package name=\"gdscript\"><classes>");

	HashSet<String> cobertura_paths;
	for (const KeyValue<String, HashMap<int, int>> &kv : p_hits) {
		cobertura_paths.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<int, int>> &cv : coverable) {
		cobertura_paths.insert(cv.key);
	}
	Vector<String> sorted_cobertura;
	for (const String &s : cobertura_paths) {
		sorted_cobertura.push_back(s);
	}
	sorted_cobertura.sort();

	static const HashMap<int, int> empty_cob_hits;
	for (const String &res_path : sorted_cobertura) {
		const HashMap<int, int> *lines = p_hits.getptr(res_path);
		_cobertura_write_class(f, res_path, lines ? *lines : empty_cob_hits,
				p_func_hits.getptr(res_path),
				p_branch_hits.getptr(res_path),
				coverable.getptr(res_path));
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

static void _json_write_funcs(Ref<FileAccess> f, const HashMap<String, int> &p_funcs) {
	f->store_string("      \"functions\": {");
	bool first = true;
	for (const KeyValue<String, int> &fv : p_funcs) {
		if (!first) {
			f->store_string(",");
		}
		first = false;
		f->store_string("\"" + fv.key.c_escape() + "\":" + itos(fv.value));
	}
	f->store_line("},");
}

static void _json_write_branches(Ref<FileAccess> f,
		const HashMap<int, GDScriptLanguage::BranchResult> &p_branches) {
	f->store_string("      \"branches\": {");
	bool first = true;
	for (const KeyValue<int, GDScriptLanguage::BranchResult> &bv : p_branches) {
		if (!first) {
			f->store_string(",");
		}
		first = false;
		f->store_string("\"" + itos(bv.key) + "_taken\":" + itos(bv.value.taken) + ",");
		f->store_string("\"" + itos(bv.key) + "_not_taken\":" + itos(bv.value.not_taken));
	}
	f->store_line("}");
}

static void _json_write_file_entry(Ref<FileAccess> f, const String &p_path,
		const HashMap<int, int> *p_lines,
		const HashMap<String, int> *p_funcs,
		const HashMap<int, GDScriptLanguage::BranchResult> *p_branches,
		const HashMap<int, int> *p_coverable) {
	f->store_line("    \"" + p_path.c_escape() + "\": {");
	if (p_lines) {
		_json_write_lines(f, *p_lines, p_coverable);
	} else {
		f->store_line("      \"lines\": {},");
	}
	if (p_funcs) {
		_json_write_funcs(f, *p_funcs);
	} else {
		f->store_line("      \"functions\": {},");
	}
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

	// Union of all file paths, including coverable files never executed.
	HashSet<String> all_files;
	for (const KeyValue<String, HashMap<int, int>> &kv : p_hits) {
		all_files.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<String, int>> &kv : p_func_hits) {
		all_files.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<int, GDScriptLanguage::BranchResult>> &kv : p_branch_hits) {
		all_files.insert(kv.key);
	}
	for (const KeyValue<String, HashMap<int, int>> &cv : coverable) {
		all_files.insert(cv.key);
	}
	Vector<String> file_list;
	for (const String &s : all_files) {
		file_list.push_back(s);
	}
	file_list.sort();

	f->store_line("{");
	f->store_line("  \"files\": {");
	for (int i = 0; i < file_list.size(); i++) {
		const String &fp = file_list[i];
		_json_write_file_entry(f, fp, p_hits.getptr(fp), p_func_hits.getptr(fp), p_branch_hits.getptr(fp), coverable.getptr(fp));
		if (i + 1 < file_list.size()) {
			f->store_line(",");
		} else {
			f->store_line("");
		}
	}
	f->store_line("  },");

	HashMap<String, CoverageFileStats> stats = _gather_file_stats(p_hits, p_func_hits, p_branch_hits, &coverable);
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

	Error err = OK;
	switch (coverage_format) {
		case COVERAGE_FORMAT_LCOV:
			err = _write_lcov(coverage_output_path, coverage_hits, coverage_func_hits, coverage_branch_hits, this);
			break;
		case COVERAGE_FORMAT_COBERTURA:
			err = _write_cobertura(coverage_output_path, coverage_hits, coverage_func_hits, coverage_branch_hits, this);
			break;
		case COVERAGE_FORMAT_JSON:
			err = _write_json(coverage_output_path, coverage_hits, coverage_func_hits, coverage_branch_hits, this);
			break;
		case COVERAGE_FORMAT_TEXT: {
			String summary = coverage_summary_string();
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
