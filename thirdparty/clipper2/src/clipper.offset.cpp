/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  28 November 2023                                                *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2023                                         *
* Purpose   :  Path Offset (Inflate/Shrink)                                    *
* License   :  http://www.boost.org/LICENSE_1_0.txt                            *
*******************************************************************************/

#include <cmath>
#include "clipper2/clipper.h"
#include "clipper2/clipper.offset.h"

namespace Clipper2Lib {

const double default_arc_tolerance = 0.25;
const double floating_point_tolerance = 1e-12;

//------------------------------------------------------------------------------
// Miscellaneous methods
//------------------------------------------------------------------------------

inline bool ToggleBoolIf(bool val, bool condition)
{
	return condition ? !val : val;
}

void GetMultiBounds(const Paths64& paths, std::vector<Rect64>& recList)
{
	recList.reserve(paths.size());
	for (const Path64& path : paths)
	{ 
		if (path.size() < 1)
		{
			recList.push_back(InvalidRect64);
			continue;
		}
		int64_t x = path[0].x, y = path[0].y;
		Rect64 r = Rect64(x, y, x, y);
		for (const Point64& pt : path)
		{
			if (pt.y > r.bottom) r.bottom = pt.y;
			else if (pt.y < r.top) r.top = pt.y;
			if (pt.x > r.right) r.right = pt.x;
			else if (pt.x < r.left) r.left = pt.x;
		}
		recList.push_back(r);
	}
}

bool ValidateBounds(std::vector<Rect64>& recList, double delta)
{
	int64_t int_delta = static_cast<int64_t>(delta);
	int64_t big = MAX_COORD - int_delta;
	int64_t small = MIN_COORD + int_delta;
	for (const Rect64& r : recList)
	{
		if (!r.IsValid()) continue; // ignore invalid paths
		else if (r.left < small || r.right > big ||
			r.top < small || r.bottom > big) return false;
	}
	return true;
}

int GetLowestClosedPathIdx(std::vector<Rect64>& boundsList)
{
	int i = -1, result = -1;
	Point64 botPt = Point64(INT64_MAX, INT64_MIN);
	for (const Rect64& r : boundsList)
	{		
		++i;
		if (!r.IsValid()) continue; // ignore invalid paths
		else if (r.bottom > botPt.y || (r.bottom == botPt.y && r.left < botPt.x))
		{
			botPt = Point64(r.left, r.bottom);
			result = static_cast<int>(i);
		}
	}
	return result;
}

PointD GetUnitNormal(const Point64& pt1, const Point64& pt2)
{
	double dx, dy, inverse_hypot;
	if (pt1 == pt2) return PointD(0.0, 0.0);
	dx = static_cast<double>(pt2.x - pt1.x);
	dy = static_cast<double>(pt2.y - pt1.y);
	inverse_hypot = 1.0 / hypot(dx, dy);
	dx *= inverse_hypot;
	dy *= inverse_hypot;
	return PointD(dy, -dx);
}

inline bool AlmostZero(double value, double epsilon = 0.001)
{
	return std::fabs(value) < epsilon;
}

inline double Hypot(double x, double y) 
{
	//see https://stackoverflow.com/a/32436148/359538
	return std::sqrt(x * x + y * y);
}

inline PointD NormalizeVector(const PointD& vec)
{	
	double h = Hypot(vec.x, vec.y);
	if (AlmostZero(h)) return PointD(0,0);
	double inverseHypot = 1 / h;
	return PointD(vec.x * inverseHypot, vec.y * inverseHypot);
}

inline PointD GetAvgUnitVector(const PointD& vec1, const PointD& vec2)
{
	return NormalizeVector(PointD(vec1.x + vec2.x, vec1.y + vec2.y));
}

inline bool IsClosedPath(EndType et)
{
	return et == EndType::Polygon || et == EndType::Joined;
}

inline Point64 GetPerpendic(const Point64& pt, const PointD& norm, double delta)
{
#ifdef USINGZ
	return Point64(pt.x + norm.x * delta, pt.y + norm.y * delta, pt.z);
#else
	return Point64(pt.x + norm.x * delta, pt.y + norm.y * delta);
#endif
}

inline PointD GetPerpendicD(const Point64& pt, const PointD& norm, double delta)
{
#ifdef USINGZ
	return PointD(pt.x + norm.x * delta, pt.y + norm.y * delta, pt.z);
#else
	return PointD(pt.x + norm.x * delta, pt.y + norm.y * delta);
#endif
}

inline void NegatePath(PathD& path)
{
	for (PointD& pt : path)
	{
		pt.x = -pt.x;
		pt.y = -pt.y;
#ifdef USINGZ
		pt.z = pt.z;
#endif
	}
}


//------------------------------------------------------------------------------
// ClipperOffset::Group methods
//------------------------------------------------------------------------------

ClipperOffset::Group::Group(const Paths64& _paths, JoinType _join_type, EndType _end_type):
	paths_in(_paths), join_type(_join_type), end_type(_end_type)
{
	bool is_joined =
		(end_type == EndType::Polygon) ||
		(end_type == EndType::Joined);
	for (Path64& p: paths_in)
	  StripDuplicates(p, is_joined);

	// get bounds of each path --> bounds_list
	GetMultiBounds(paths_in, bounds_list);

	if (end_type == EndType::Polygon)
	{
		is_hole_list.reserve(paths_in.size());
		for (const Path64& path : paths_in)
			is_hole_list.push_back(Area(path) < 0);
		lowest_path_idx = GetLowestClosedPathIdx(bounds_list);
		// the lowermost path must be an outer path, so if its orientation is negative,
		// then flag the whole group is 'reversed' (will negate delta etc.)
		// as this is much more efficient than reversing every path.
		is_reversed = (lowest_path_idx >= 0) && is_hole_list[lowest_path_idx];
		if (is_reversed) is_hole_list.flip();
	}
	else
	{
		lowest_path_idx = -1;
		is_reversed = false;
		is_hole_list.resize(paths_in.size());
	}
}


//------------------------------------------------------------------------------
// ClipperOffset methods
//------------------------------------------------------------------------------

void ClipperOffset::AddPath(const Path64& path, JoinType jt_, EndType et_)
{
	Paths64 paths;
	paths.push_back(path);
	AddPaths(paths, jt_, et_);
}

void ClipperOffset::AddPaths(const Paths64 &paths, JoinType jt_, EndType et_)
{
	if (paths.size() == 0) return;
	groups_.push_back(Group(paths, jt_, et_));
}

void ClipperOffset::BuildNormals(const Path64& path)
{
	norms.clear();
	norms.reserve(path.size());
	if (path.size() == 0) return;
	Path64::const_iterator path_iter, path_stop_iter = --path.cend();
	for (path_iter = path.cbegin(); path_iter != path_stop_iter; ++path_iter)
		norms.push_back(GetUnitNormal(*path_iter,*(path_iter +1)));
	norms.push_back(GetUnitNormal(*path_stop_iter, *(path.cbegin())));
}

inline PointD TranslatePoint(const PointD& pt, double dx, double dy)
{
#ifdef USINGZ
	return PointD(pt.x + dx, pt.y + dy, pt.z);
#else
	return PointD(pt.x + dx, pt.y + dy);
#endif
}

inline PointD ReflectPoint(const PointD& pt, const PointD& pivot)
{
#ifdef USINGZ
	return PointD(pivot.x + (pivot.x - pt.x), pivot.y + (pivot.y - pt.y), pt.z);
#else
	return PointD(pivot.x + (pivot.x - pt.x), pivot.y + (pivot.y - pt.y));
#endif
}

PointD IntersectPoint(const PointD& pt1a, const PointD& pt1b,
	const PointD& pt2a, const PointD& pt2b)
{
	if (pt1a.x == pt1b.x) //vertical
	{
		if (pt2a.x == pt2b.x) return PointD(0, 0);

		double m2 = (pt2b.y - pt2a.y) / (pt2b.x - pt2a.x);
		double b2 = pt2a.y - m2 * pt2a.x;
		return PointD(pt1a.x, m2 * pt1a.x + b2);
	}
	else if (pt2a.x == pt2b.x) //vertical
	{
		double m1 = (pt1b.y - pt1a.y) / (pt1b.x - pt1a.x);
		double b1 = pt1a.y - m1 * pt1a.x;
		return PointD(pt2a.x, m1 * pt2a.x + b1);
	}
	else
	{
		double m1 = (pt1b.y - pt1a.y) / (pt1b.x - pt1a.x);
		double b1 = pt1a.y - m1 * pt1a.x;
		double m2 = (pt2b.y - pt2a.y) / (pt2b.x - pt2a.x);
		double b2 = pt2a.y - m2 * pt2a.x;
		if (m1 == m2) return PointD(0, 0);
		double x = (b2 - b1) / (m1 - m2);
		return PointD(x, m1 * x + b1);
	}
}

void ClipperOffset::DoBevel(const Path64& path, size_t j, size_t k)
{
	PointD pt1, pt2;
	if (j == k)
	{
		double abs_delta = std::abs(group_delta_);
		pt1 = PointD(path[j].x - abs_delta * norms[j].x, path[j].y - abs_delta * norms[j].y);
		pt2 = PointD(path[j].x + abs_delta * norms[j].x, path[j].y + abs_delta * norms[j].y);
	} 
	else
	{
		pt1 = PointD(path[j].x + group_delta_ * norms[k].x, path[j].y + group_delta_ * norms[k].y);
		pt2 = PointD(path[j].x + group_delta_ * norms[j].x, path[j].y + group_delta_ * norms[j].y);
	}
	path_out.push_back(Point64(pt1));
	path_out.push_back(Point64(pt2));
}

void ClipperOffset::DoSquare(const Path64& path, size_t j, size_t k)
{
	PointD vec;
	if (j == k) 
		vec = PointD(norms[j].y, -norms[j].x);
	else
		vec = GetAvgUnitVector(
			PointD(-norms[k].y, norms[k].x),
			PointD(norms[j].y, -norms[j].x));

	double abs_delta = std::abs(group_delta_);

	// now offset the original vertex delta units along unit vector
	PointD ptQ = PointD(path[j]);
	ptQ = TranslatePoint(ptQ, abs_delta * vec.x, abs_delta * vec.y);
	// get perpendicular vertices
	PointD pt1 = TranslatePoint(ptQ, group_delta_ * vec.y, group_delta_ * -vec.x);
	PointD pt2 = TranslatePoint(ptQ, group_delta_ * -vec.y, group_delta_ * vec.x);
	// get 2 vertices along one edge offset
	PointD pt3 = GetPerpendicD(path[k], norms[k], group_delta_);
	if (j == k)
	{
		PointD pt4 = PointD(pt3.x + vec.x * group_delta_, pt3.y + vec.y * group_delta_);
		PointD pt = IntersectPoint(pt1, pt2, pt3, pt4);
#ifdef USINGZ
		pt.z = ptQ.z;
#endif
		//get the second intersect point through reflecion
		path_out.push_back(Point64(ReflectPoint(pt, ptQ)));
		path_out.push_back(Point64(pt));
	}
	else
	{
		PointD pt4 = GetPerpendicD(path[j], norms[k], group_delta_);
		PointD pt = IntersectPoint(pt1, pt2, pt3, pt4);
#ifdef USINGZ
		pt.z = ptQ.z;
#endif
		path_out.push_back(Point64(pt));
		//get the second intersect point through reflecion
		path_out.push_back(Point64(ReflectPoint(pt, ptQ)));
	}
}

void ClipperOffset::DoMiter(const Path64& path, size_t j, size_t k, double cos_a)
{
	double q = group_delta_ / (cos_a + 1);
#ifdef USINGZ
	path_out.push_back(Point64(
		path[j].x + (norms[k].x + norms[j].x) * q,
		path[j].y + (norms[k].y + norms[j].y) * q,
		path[j].z));
#else
	path_out.push_back(Point64(
		path[j].x + (norms[k].x + norms[j].x) * q,
		path[j].y + (norms[k].y + norms[j].y) * q));
#endif
}

void ClipperOffset::DoRound(const Path64& path, size_t j, size_t k, double angle)
{
	if (deltaCallback64_) {
		// when deltaCallback64_ is assigned, group_delta_ won't be constant, 
		// so we'll need to do the following calculations for *every* vertex.
		double abs_delta = std::fabs(group_delta_);
		double arcTol = (arc_tolerance_ > floating_point_tolerance ?
			std::min(abs_delta, arc_tolerance_) :
			std::log10(2 + abs_delta) * default_arc_tolerance);
		double steps_per_360 = std::min(PI / std::acos(1 - arcTol / abs_delta), abs_delta * PI);
		step_sin_ = std::sin(2 * PI / steps_per_360);
		step_cos_ = std::cos(2 * PI / steps_per_360);
		if (group_delta_ < 0.0) step_sin_ = -step_sin_;
		steps_per_rad_ = steps_per_360 / (2 * PI);
	}

	Point64 pt = path[j];
	PointD offsetVec = PointD(norms[k].x * group_delta_, norms[k].y * group_delta_);

	if (j == k) offsetVec.Negate();
#ifdef USINGZ
	path_out.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y, pt.z));
#else
	path_out.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y));
#endif
	int steps = static_cast<int>(std::ceil(steps_per_rad_ * std::abs(angle))); // #448, #456
	for (int i = 1; i < steps; ++i) // ie 1 less than steps
	{
		offsetVec = PointD(offsetVec.x * step_cos_ - step_sin_ * offsetVec.y,
			offsetVec.x * step_sin_ + offsetVec.y * step_cos_);
#ifdef USINGZ
		path_out.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y, pt.z));
#else
		path_out.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y));
#endif
	}
	path_out.push_back(GetPerpendic(path[j], norms[j], group_delta_));
}

void ClipperOffset::OffsetPoint(Group& group, const Path64& path, size_t j, size_t k)
{
	// Let A = change in angle where edges join
	// A == 0: ie no change in angle (flat join)
	// A == PI: edges 'spike'
	// sin(A) < 0: right turning
	// cos(A) < 0: change in angle is more than 90 degree

	if (path[j] == path[k]) { k = j; return; }

	double sin_a = CrossProduct(norms[j], norms[k]);
	double cos_a = DotProduct(norms[j], norms[k]);
	if (sin_a > 1.0) sin_a = 1.0;
	else if (sin_a < -1.0) sin_a = -1.0;

	if (deltaCallback64_) {
		group_delta_ = deltaCallback64_(path, norms, j, k);
		if (group.is_reversed) group_delta_ = -group_delta_;
	}
	if (std::fabs(group_delta_) <= floating_point_tolerance)
	{
		path_out.push_back(path[j]);
		return;
	}

	if (cos_a > -0.99 && (sin_a * group_delta_ < 0)) // test for concavity first (#593)
	{
		// is concave
		path_out.push_back(GetPerpendic(path[j], norms[k], group_delta_));
		// this extra point is the only (simple) way to ensure that
	  // path reversals are fully cleaned with the trailing clipper		
		path_out.push_back(path[j]); // (#405)
		path_out.push_back(GetPerpendic(path[j], norms[j], group_delta_));
	}
	else if (cos_a > 0.999 && join_type_ != JoinType::Round) 
	{
		// almost straight - less than 2.5 degree (#424, #482, #526 & #724) 
		DoMiter(path, j, k, cos_a);
	}
	else if (join_type_ == JoinType::Miter)
	{
		// miter unless the angle is sufficiently acute to exceed ML
		if (cos_a > temp_lim_ - 1) DoMiter(path, j, k, cos_a);
		else DoSquare(path, j, k);
	}
	else if (join_type_ == JoinType::Round)
		DoRound(path, j, k, std::atan2(sin_a, cos_a));
	else if ( join_type_ == JoinType::Bevel)
		DoBevel(path, j, k);
	else
		DoSquare(path, j, k);
}

void ClipperOffset::OffsetPolygon(Group& group, const Path64& path)
{
	path_out.clear();
	for (Path64::size_type j = 0, k = path.size() -1; j < path.size(); k = j, ++j)
		OffsetPoint(group, path, j, k);
	solution.push_back(path_out);
}

void ClipperOffset::OffsetOpenJoined(Group& group, const Path64& path)
{
	OffsetPolygon(group, path);
	Path64 reverse_path(path);
	std::reverse(reverse_path.begin(), reverse_path.end());
	
	//rebuild normals // BuildNormals(path);
	std::reverse(norms.begin(), norms.end());
	norms.push_back(norms[0]);
	norms.erase(norms.begin());
	NegatePath(norms);

	OffsetPolygon(group, reverse_path);
}

void ClipperOffset::OffsetOpenPath(Group& group, const Path64& path)
{
	// do the line start cap
	if (deltaCallback64_) group_delta_ = deltaCallback64_(path, norms, 0, 0);
	
	if (std::fabs(group_delta_) <= floating_point_tolerance)
		path_out.push_back(path[0]);
	else
	{
		switch (end_type_)
		{
		case EndType::Butt:
			DoBevel(path, 0, 0);
			break;
		case EndType::Round:
			DoRound(path, 0, 0, PI);
			break;
		default:
			DoSquare(path, 0, 0);
			break;
		}
	}
	
	size_t highI = path.size() - 1;
	// offset the left side going forward
	for (Path64::size_type j = 1, k = 0; j < highI; k = j, ++j)
		OffsetPoint(group, path, j, k);

	// reverse normals 
	for (size_t i = highI; i > 0; --i)
		norms[i] = PointD(-norms[i - 1].x, -norms[i - 1].y);
	norms[0] = norms[highI];

	// do the line end cap
	if (deltaCallback64_)
		group_delta_ = deltaCallback64_(path, norms, highI, highI);

	if (std::fabs(group_delta_) <= floating_point_tolerance)
		path_out.push_back(path[highI]);
	else
	{
		switch (end_type_)
		{
		case EndType::Butt:
			DoBevel(path, highI, highI);
			break;
		case EndType::Round:
			DoRound(path, highI, highI, PI);
			break;
		default:
			DoSquare(path, highI, highI);
			break;
		}
	}

	for (size_t j = highI, k = 0; j > 0; k = j, --j)
		OffsetPoint(group, path, j, k);
	solution.push_back(path_out);
}

void ClipperOffset::DoGroupOffset(Group& group)
{
	if (group.end_type == EndType::Polygon)
	{
		// a straight path (2 points) can now also be 'polygon' offset 
		// where the ends will be treated as (180 deg.) joins
		if (group.lowest_path_idx < 0) delta_ = std::abs(delta_);
		group_delta_ = (group.is_reversed) ? -delta_ : delta_;
	}
	else
		group_delta_ = std::abs(delta_);// *0.5;

	double abs_delta = std::fabs(group_delta_);
	if (!ValidateBounds(group.bounds_list, abs_delta))
	{
		DoError(range_error_i);
		error_code_ |= range_error_i;
		return;
	}

	join_type_	= group.join_type;
	end_type_ = group.end_type;

	if (group.join_type == JoinType::Round || group.end_type == EndType::Round)
	{
		// calculate a sensible number of steps (for 360 deg for the given offset)
		// arcTol - when arc_tolerance_ is undefined (0), the amount of 
		// curve imprecision that's allowed is based on the size of the 
		// offset (delta). Obviously very large offsets will almost always 
		// require much less precision. See also offset_triginometry2.svg
		double arcTol = (arc_tolerance_ > floating_point_tolerance ?
			std::min(abs_delta, arc_tolerance_) :
			std::log10(2 + abs_delta) * default_arc_tolerance);

		double steps_per_360 = std::min(PI / std::acos(1 - arcTol / abs_delta), abs_delta * PI);
		step_sin_ = std::sin(2 * PI / steps_per_360);
		step_cos_ = std::cos(2 * PI / steps_per_360);
		if (group_delta_ < 0.0) step_sin_ = -step_sin_;
		steps_per_rad_ = steps_per_360 / (2 * PI);
	}

	std::vector<Rect64>::const_iterator path_rect_it = group.bounds_list.cbegin();
	std::vector<bool>::const_iterator is_hole_it = group.is_hole_list.cbegin();
	Paths64::const_iterator path_in_it = group.paths_in.cbegin();
	for ( ; path_in_it != group.paths_in.cend(); ++path_in_it, ++path_rect_it, ++is_hole_it)
	{
		if (!path_rect_it->IsValid()) continue;
		Path64::size_type pathLen = path_in_it->size();
		path_out.clear();

		if (pathLen == 1) // single point
		{
			if (group_delta_ < 1) continue;
			const Point64& pt = (*path_in_it)[0];
			//single vertex so build a circle or square ...
			if (group.join_type == JoinType::Round)
			{
				double radius = abs_delta;
				int steps = static_cast<int>(std::ceil(steps_per_rad_ * 2 * PI)); //#617
				path_out = Ellipse(pt, radius, radius, steps);
#ifdef USINGZ
				for (auto& p : path_out) p.z = pt.z;
#endif
			}
			else
			{
				int d = (int)std::ceil(abs_delta);
				Rect64 r = Rect64(pt.x - d, pt.y - d, pt.x + d, pt.y + d);
				path_out = r.AsPath();
#ifdef USINGZ
				for (auto& p : path_out) p.z = pt.z;
#endif
			}
			solution.push_back(path_out);
			continue;
		} // end of offsetting a single point 

		// when shrinking outer paths, make sure they can shrink this far (#593)
		// also when shrinking holes, make sure they too can shrink this far (#715)
		if ((group_delta_ > 0) == ToggleBoolIf(*is_hole_it, group.is_reversed) &&
			(std::min(path_rect_it->Width(), path_rect_it->Height()) <= -group_delta_ * 2) )
				  continue;

		if ((pathLen == 2) && (group.end_type == EndType::Joined))
			end_type_ = (group.join_type == JoinType::Round) ? 
			  EndType::Round : 
			  EndType::Square;

		BuildNormals(*path_in_it);
		if (end_type_ == EndType::Polygon) OffsetPolygon(group, *path_in_it);
		else if (end_type_ == EndType::Joined) OffsetOpenJoined(group, *path_in_it);
		else OffsetOpenPath(group, *path_in_it);
	}
}


size_t ClipperOffset::CalcSolutionCapacity()
{
	size_t result = 0;
	for (const Group& g : groups_)
		result += (g.end_type == EndType::Joined) ? g.paths_in.size() * 2 : g.paths_in.size();
	return result;
}

bool ClipperOffset::CheckReverseOrientation()
{
	// nb: this assumes there's consistency in orientation between groups
	bool is_reversed_orientation = false;
	for (const Group& g : groups_)
		if (g.end_type == EndType::Polygon)
		{
			is_reversed_orientation = g.is_reversed;
			break;
		}
	return is_reversed_orientation;
}

void ClipperOffset::ExecuteInternal(double delta)
{
	error_code_ = 0;
	solution.clear();
	if (groups_.size() == 0) return;
	solution.reserve(CalcSolutionCapacity());

	if (std::abs(delta) < 0.5) // ie: offset is insignificant 
	{
		Paths64::size_type sol_size = 0;
		for (const Group& group : groups_) sol_size += group.paths_in.size();
		solution.reserve(sol_size);
		for (const Group& group : groups_)
			copy(group.paths_in.begin(), group.paths_in.end(), back_inserter(solution));
		return;
	}

	temp_lim_ = (miter_limit_ <= 1) ?
		2.0 :
		2.0 / (miter_limit_ * miter_limit_);

	delta_ = delta;
	std::vector<Group>::iterator git;
	for (git = groups_.begin(); git != groups_.end(); ++git)
	{
		DoGroupOffset(*git);
		if (!error_code_) continue; // all OK
		solution.clear();
	}
}

void ClipperOffset::Execute(double delta, Paths64& paths)
{
	paths.clear();

	ExecuteInternal(delta);
	if (!solution.size()) return;

	bool paths_reversed = CheckReverseOrientation();
	//clean up self-intersections ...
	Clipper64 c;
	c.PreserveCollinear(false);
	//the solution should retain the orientation of the input
	c.ReverseSolution(reverse_solution_ != paths_reversed);
#ifdef USINGZ
	if (zCallback64_) { c.SetZCallback(zCallback64_); }
#endif
	c.AddSubject(solution);
	if (paths_reversed)
		c.Execute(ClipType::Union, FillRule::Negative, paths);
	else
		c.Execute(ClipType::Union, FillRule::Positive, paths);
}


void ClipperOffset::Execute(double delta, PolyTree64& polytree)
{
	polytree.Clear();

	ExecuteInternal(delta);
	if (!solution.size()) return;

	bool paths_reversed = CheckReverseOrientation();
	//clean up self-intersections ...
	Clipper64 c;
	c.PreserveCollinear(false);
	//the solution should retain the orientation of the input
	c.ReverseSolution (reverse_solution_ != paths_reversed);
#ifdef USINGZ
	if (zCallback64_) {
		c.SetZCallback(zCallback64_);
	}
#endif
	c.AddSubject(solution);


	if (paths_reversed)
		c.Execute(ClipType::Union, FillRule::Negative, polytree);
	else
		c.Execute(ClipType::Union, FillRule::Positive, polytree);
}

void ClipperOffset::Execute(DeltaCallback64 delta_cb, Paths64& paths)
{
	deltaCallback64_ = delta_cb;
	Execute(1.0, paths);
}

} // namespace
