/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  22 March 2023                                                   *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2023                                         *
* Purpose   :  Path Offset (Inflate/Shrink)                                    *
* License   :  http://www.boost.org/LICENSE_1_0.txt                            *
*******************************************************************************/

#ifndef CLIPPER_OFFSET_H_
#define CLIPPER_OFFSET_H_

#include "clipper.core.h"
#include "clipper.engine.h"

namespace Clipper2Lib {

enum class JoinType { Square, Round, Miter };

enum class EndType {Polygon, Joined, Butt, Square, Round};
//Butt   : offsets both sides of a path, with square blunt ends
//Square : offsets both sides of a path, with square extended ends
//Round  : offsets both sides of a path, with round extended ends
//Joined : offsets both sides of a path, with joined ends
//Polygon: offsets only one side of a closed path


class ClipperOffset {
private:

	class Group {
	public:
		Paths64 paths_in;
		Paths64 paths_out;
		Path64 path;
		bool is_reversed = false;
		JoinType join_type;
		EndType end_type;
		Group(const Paths64& _paths, JoinType _join_type, EndType _end_type) :
			paths_in(_paths), join_type(_join_type), end_type(_end_type) {}
	};

	int   error_code_ = 0;
	double delta_ = 0.0;
	double group_delta_ = 0.0;
	double abs_group_delta_ = 0.0;
	double temp_lim_ = 0.0;
	double steps_per_rad_ = 0.0;
	double step_sin_ = 0.0;
	double step_cos_ = 0.0;
	PathD norms;
	Paths64 solution;
	std::vector<Group> groups_;
	JoinType join_type_ = JoinType::Square;
	EndType end_type_ = EndType::Polygon;

	double miter_limit_ = 0.0;
	double arc_tolerance_ = 0.0;
	bool preserve_collinear_ = false;
	bool reverse_solution_ = false;

#ifdef USINGZ
	ZCallback64 zCallback64_ = nullptr;
#endif

	void DoSquare(Group& group, const Path64& path, size_t j, size_t k);
	void DoMiter(Group& group, const Path64& path, size_t j, size_t k, double cos_a);
	void DoRound(Group& group, const Path64& path, size_t j, size_t k, double angle);
	void BuildNormals(const Path64& path);
	void OffsetPolygon(Group& group, Path64& path);
	void OffsetOpenJoined(Group& group, Path64& path);
	void OffsetOpenPath(Group& group, Path64& path);
	void OffsetPoint(Group& group, Path64& path, size_t j, size_t& k);
	void DoGroupOffset(Group &group);
	void ExecuteInternal(double delta);
public:
	explicit ClipperOffset(double miter_limit = 2.0,
		double arc_tolerance = 0.0,
		bool preserve_collinear = false, 
		bool reverse_solution = false) :
		miter_limit_(miter_limit), arc_tolerance_(arc_tolerance),
		preserve_collinear_(preserve_collinear),
		reverse_solution_(reverse_solution) { };

	~ClipperOffset() { Clear(); };

	int ErrorCode() { return error_code_; };
	void AddPath(const Path64& path, JoinType jt_, EndType et_);
	void AddPaths(const Paths64& paths, JoinType jt_, EndType et_);
	void Clear() { groups_.clear(); norms.clear(); };
	
	void Execute(double delta, Paths64& paths);
	void Execute(double delta, PolyTree64& polytree);

	double MiterLimit() const { return miter_limit_; }
	void MiterLimit(double miter_limit) { miter_limit_ = miter_limit; }

	//ArcTolerance: needed for rounded offsets (See offset_triginometry2.svg)
	double ArcTolerance() const { return arc_tolerance_; }
	void ArcTolerance(double arc_tolerance) { arc_tolerance_ = arc_tolerance; }

	bool PreserveCollinear() const { return preserve_collinear_; }
	void PreserveCollinear(bool preserve_collinear){preserve_collinear_ = preserve_collinear;}
	
	bool ReverseSolution() const { return reverse_solution_; }
	void ReverseSolution(bool reverse_solution) {reverse_solution_ = reverse_solution;}

#ifdef USINGZ
	void SetZCallback(ZCallback64 cb) { zCallback64_ = cb; }
#endif
};

}
#endif /* CLIPPER_OFFSET_H_ */
