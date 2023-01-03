/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  27 January 2023                                                 *
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

void GetBoundsAndLowestPolyIdx(const Paths64& paths, Rect64& r, int & idx)
{
	idx = -1;
	r = MaxInvalidRect64;
	int64_t lpx = 0;
	for (int i = 0; i < static_cast<int>(paths.size()); ++i)
		for (const Point64& p : paths[i])
		{
			if (p.y >= r.bottom)
			{
				if (p.y > r.bottom || p.x < lpx)
				{
					idx = i;
					lpx = p.x;
					r.bottom = p.y;
				}
			}
			else if (p.y < r.top) r.top = p.y;
			if (p.x > r.right) r.right = p.x;
			else if (p.x < r.left) r.left = p.x;
		}
	if (idx < 0) r = Rect64(0, 0, 0, 0);
	if (r.top == INT64_MIN) r.bottom = r.top;
	if (r.left == INT64_MIN) r.left = r.right;
}

bool IsSafeOffset(const Rect64& r, int64_t delta)
{
	if (delta < 0) return true;
	return r.left > INT64_MIN + delta &&
		r.right < INT64_MAX - delta &&
		r.top > INT64_MIN + delta &&
		r.bottom < INT64_MAX - delta;
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
	return Point64(pt.x + norm.x * delta, pt.y + norm.y * delta);
}

inline PointD GetPerpendicD(const Point64& pt, const PointD& norm, double delta)
{
	return PointD(pt.x + norm.x * delta, pt.y + norm.y * delta);
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

void ClipperOffset::AddPath(const Clipper2Lib::PathD& path, JoinType jt_, EndType et_)
{
	PathsD paths;
	paths.push_back(path);
	AddPaths(paths, jt_, et_);
}

void ClipperOffset::AddPaths(const PathsD& paths, JoinType jt_, EndType et_)
{
	if (paths.size() == 0) return;
	groups_.push_back(Group(PathsDToPaths64(paths), jt_, et_));
}

void ClipperOffset::BuildNormals(const Path64& path)
{
	norms.clear();
	norms.reserve(path.size());
	if (path.size() == 0) return;
	Path64::const_iterator path_iter, path_last_iter = --path.cend();
	for (path_iter = path.cbegin(); path_iter != path_last_iter; ++path_iter)
		norms.push_back(GetUnitNormal(*path_iter,*(path_iter +1)));
	norms.push_back(GetUnitNormal(*path_last_iter, *(path.cbegin())));
}

inline PointD TranslatePoint(const PointD& pt, double dx, double dy)
{
	return PointD(pt.x + dx, pt.y + dy);
}

inline PointD ReflectPoint(const PointD& pt, const PointD& pivot)
{
	return PointD(pivot.x + (pivot.x - pt.x), pivot.y + (pivot.y - pt.y));
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

void ClipperOffset::DoSquare(Group& group, const Path64& path, size_t j, size_t k)
{
	PointD vec;
	if (j == k) 
		vec = PointD(norms[0].y, -norms[0].x);
	else
		vec = GetAvgUnitVector(
			PointD(-norms[k].y, norms[k].x),
			PointD(norms[j].y, -norms[j].x));

	// now offset the original vertex delta units along unit vector
	PointD ptQ = PointD(path[j]);
	ptQ = TranslatePoint(ptQ, abs_group_delta_ * vec.x, abs_group_delta_ * vec.y);
	// get perpendicular vertices
	PointD pt1 = TranslatePoint(ptQ, group_delta_ * vec.y, group_delta_ * -vec.x);
	PointD pt2 = TranslatePoint(ptQ, group_delta_ * -vec.y, group_delta_ * vec.x);
	// get 2 vertices along one edge offset
	PointD pt3 = GetPerpendicD(path[k], norms[k], group_delta_);
	if (j == k)
	{
		PointD pt4 = PointD(pt3.x + vec.x * group_delta_, pt3.y + vec.y * group_delta_);
		PointD pt = IntersectPoint(pt1, pt2, pt3, pt4);
		//get the second intersect point through reflecion
		group.path_.push_back(Point64(ReflectPoint(pt, ptQ)));
		group.path_.push_back(Point64(pt));
	}
	else
	{
		PointD pt4 = GetPerpendicD(path[j], norms[k], group_delta_);
		PointD pt = IntersectPoint(pt1, pt2, pt3, pt4);
		group.path_.push_back(Point64(pt));
		//get the second intersect point through reflecion
		group.path_.push_back(Point64(ReflectPoint(pt, ptQ)));
	}
}

void ClipperOffset::DoMiter(Group& group, const Path64& path, size_t j, size_t k, double cos_a)
{
	double q = group_delta_ / (cos_a + 1);
	group.path_.push_back(Point64(
		path[j].x + (norms[k].x + norms[j].x) * q,
		path[j].y + (norms[k].y + norms[j].y) * q));
}

void ClipperOffset::DoRound(Group& group, const Path64& path, size_t j, size_t k, double angle)
{
	//even though angle may be negative this is a convex join
	Point64 pt = path[j];
	int steps = static_cast<int>(std::floor(steps_per_rad_ * std::abs(angle)));
	double step_sin = std::sin(angle / steps);
	double step_cos = std::cos(angle / steps);
	
	PointD pt2 = PointD(norms[k].x * group_delta_, norms[k].y * group_delta_);
	if (j == k) pt2.Negate();

	group.path_.push_back(Point64(pt.x + pt2.x, pt.y + pt2.y));
	for (int i = 0; i < steps; ++i)
	{
		pt2 = PointD(pt2.x * step_cos - step_sin * pt2.y,
			pt2.x * step_sin + pt2.y * step_cos);
		group.path_.push_back(Point64(pt.x + pt2.x, pt.y + pt2.y));
	}
	group.path_.push_back(GetPerpendic(path[j], norms[j], group_delta_));
}

void ClipperOffset::OffsetPoint(Group& group, 
	Path64& path, size_t j, size_t& k, bool reversing)
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

	bool almostNoAngle = AlmostZero(cos_a - 1);
	bool is180DegSpike = AlmostZero(cos_a + 1) && reversing;
	// when there's almost no angle of deviation or it's concave
	if (almostNoAngle || is180DegSpike || (sin_a * group_delta_ < 0))
	{
    //almost no angle or concave
		group.path_.push_back(GetPerpendic(path[j], norms[k], group_delta_));
		// create a simple self-intersection that will be cleaned up later
		if (!almostNoAngle) group.path_.push_back(path[j]);
		group.path_.push_back(GetPerpendic(path[j], norms[j], group_delta_));
	}
	else 
	{
		// it's convex 
		if (join_type_ == JoinType::Round)
			DoRound(group, path, j, k, std::atan2(sin_a, cos_a));
		else if (join_type_ == JoinType::Miter)
		{
			// miter unless the angle is so acute the miter would exceeds ML
			if (cos_a > temp_lim_ - 1) DoMiter(group, path, j, k, cos_a);
			else DoSquare(group, path, j, k);
		}
		// don't bother squaring angles that deviate < ~20 degrees because
		// squaring will be indistinguishable from mitering and just be a lot slower
		else if (cos_a > 0.9)
			DoMiter(group, path, j, k, cos_a);			
		else
			DoSquare(group, path, j, k);			
	}
	k = j;
}

void ClipperOffset::OffsetPolygon(Group& group, Path64& path)
{
	group.path_.clear();
	for (Path64::size_type i = 0, j = path.size() -1; i < path.size(); j = i, ++i)
		OffsetPoint(group, path, i, j);
	group.paths_out_.push_back(group.path_);
}

void ClipperOffset::OffsetOpenJoined(Group& group, Path64& path)
{
	OffsetPolygon(group, path);
	std::reverse(path.begin(), path.end());
	BuildNormals(path);
	OffsetPolygon(group, path);
}

void ClipperOffset::OffsetOpenPath(Group& group, Path64& path, EndType end_type)
{
	group.path_.clear();

	// do the line start cap
	switch (end_type)
	{
	case EndType::Butt:
		group.path_.push_back(Point64(
			path[0].x - norms[0].x * group_delta_,
			path[0].y - norms[0].y * group_delta_));
		group.path_.push_back(GetPerpendic(path[0], norms[0], group_delta_));
		break;
	case EndType::Round:
		DoRound(group, path, 0,0, PI);
		break;
	default:
		DoSquare(group, path, 0, 0);
		break;
	}

	size_t highI = path.size() - 1;

	// offset the left side going forward
	for (Path64::size_type i = 1, k = 0; i < highI; ++i)
		OffsetPoint(group, path, i, k);

	// reverse normals 
	for (size_t i = highI; i > 0; --i)
		norms[i] = PointD(-norms[i - 1].x, -norms[i - 1].y);
	norms[0] = norms[highI];

	// do the line end cap
	switch (end_type)
	{
	case EndType::Butt:
		group.path_.push_back(Point64(
			path[highI].x - norms[highI].x * group_delta_,
			path[highI].y - norms[highI].y * group_delta_));
		group.path_.push_back(GetPerpendic(path[highI], norms[highI], group_delta_));
		break;
	case EndType::Round:
		DoRound(group, path, highI, highI, PI);
		break;
	default:
		DoSquare(group, path, highI, highI);
		break;
	}

	for (size_t i = highI, k = 0; i > 0; --i)
		OffsetPoint(group, path, i, k, true);
	group.paths_out_.push_back(group.path_);
}

void ClipperOffset::DoGroupOffset(Group& group, double delta)
{
	if (group.end_type_ != EndType::Polygon) delta = std::abs(delta) * 0.5;
	bool isClosedPaths = IsClosedPath(group.end_type_);

	//the lowermost polygon must be an outer polygon. So we can use that as the
	//designated orientation for outer polygons (needed for tidy-up clipping)
	Rect64 r;
	int idx = 0;
	GetBoundsAndLowestPolyIdx(group.paths_in_, r, idx);
	if (!IsSafeOffset(r, static_cast<int64_t>(std::ceil(delta))))
#if __cpp_exceptions
	throw Clipper2Exception(range_error);
#else
		error_code_ |= range_error_i;
		return;
#endif
	if (isClosedPaths)
	{
		double area = Area(group.paths_in_[idx]);
		if (area == 0) return;	
		group.is_reversed_ = (area < 0);
		if (group.is_reversed_) delta = -delta;
	} 
	else
		group.is_reversed_ = false;

	group_delta_ = delta;
	abs_group_delta_ = std::abs(group_delta_);
	join_type_ = group.join_type_;

	double arcTol = (arc_tolerance_ > floating_point_tolerance ? arc_tolerance_
		: std::log10(2 + abs_group_delta_) * default_arc_tolerance); // empirically derived

//calculate a sensible number of steps (for 360 deg for the given offset
	if (group.join_type_ == JoinType::Round || group.end_type_ == EndType::Round)
	{
		steps_per_rad_ = PI / std::acos(1 - arcTol / abs_group_delta_) / (PI *2);
	}

	bool is_closed_path = IsClosedPath(group.end_type_);
	Paths64::const_iterator path_iter;
	for(path_iter = group.paths_in_.cbegin(); path_iter != group.paths_in_.cend(); ++path_iter)
	{
		Path64 path = StripDuplicates(*path_iter, is_closed_path);
		Path64::size_type cnt = path.size();
		if (cnt == 0) continue;

		if (cnt == 1) // single point - only valid with open paths
		{
			group.path_ = Path64();
			//single vertex so build a circle or square ...
			if (group.join_type_ == JoinType::Round)
			{
				double radius = abs_group_delta_;
				group.path_ = Ellipse(path[0], radius, radius);
			}
			else
			{
				int d = (int)std::ceil(abs_group_delta_);
				r = Rect64(path[0].x - d, path[0].y - d, path[0].x + d, path[0].y + d);
				group.path_ = r.AsPath();
			}
			group.paths_out_.push_back(group.path_);
		}
		else
		{
			BuildNormals(path);
			if (group.end_type_ == EndType::Polygon) OffsetPolygon(group, path);
			else if (group.end_type_ == EndType::Joined) OffsetOpenJoined(group, path);
			else OffsetOpenPath(group, path, group.end_type_);
		}
	}
	solution.reserve(solution.size() + group.paths_out_.size());
	copy(group.paths_out_.begin(), group.paths_out_.end(), back_inserter(solution));
	group.paths_out_.clear();
}

Paths64 ClipperOffset::Execute(double delta)
{
	error_code_ = 0;
	solution.clear();
	if (groups_.size() == 0) return solution;

	if (std::abs(delta) < default_arc_tolerance)
	{
		for (const Group& group : groups_)
		{
			solution.reserve(solution.size() + group.paths_in_.size());
			copy(group.paths_in_.begin(), group.paths_in_.end(), back_inserter(solution));
		}
		return solution;
	}

	temp_lim_ = (miter_limit_ <= 1) ? 
		2.0 : 
		2.0 / (miter_limit_ * miter_limit_);

	std::vector<Group>::iterator git;
	for (git = groups_.begin(); git != groups_.end(); ++git)
	{
		DoGroupOffset(*git, delta);
		if (!error_code_) continue;
		solution.clear();
		return solution;
	}

	//clean up self-intersections ...
	Clipper64 c;
	c.PreserveCollinear = false;
	//the solution should retain the orientation of the input
	c.ReverseSolution = reverse_solution_ != groups_[0].is_reversed_;
	c.AddSubject(solution);
	if (groups_[0].is_reversed_)
		c.Execute(ClipType::Union, FillRule::Negative, solution);
	else
		c.Execute(ClipType::Union, FillRule::Positive, solution);

	return solution;
}

} // namespace
