/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  22 March 2023                                                   *
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
	//if (idx < 0) r = Rect64(0, 0, 0, 0);
	//if (r.top == INT64_MIN) r.bottom = r.top;
	//if (r.left == INT64_MIN) r.left = r.right;
}

bool IsSafeOffset(const Rect64& r, double abs_delta)
{
	return r.left > min_coord + abs_delta &&
		r.right < max_coord - abs_delta &&
		r.top > min_coord + abs_delta &&
		r.bottom < max_coord - abs_delta;
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
	Path64::const_iterator path_iter, path_last_iter = --path.cend();
	for (path_iter = path.cbegin(); path_iter != path_last_iter; ++path_iter)
		norms.push_back(GetUnitNormal(*path_iter,*(path_iter +1)));
	norms.push_back(GetUnitNormal(*path_last_iter, *(path.cbegin())));
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
#ifdef USINGZ
		pt.z = ptQ.z;
#endif
		//get the second intersect point through reflecion
		group.path.push_back(Point64(ReflectPoint(pt, ptQ)));
		group.path.push_back(Point64(pt));
	}
	else
	{
		PointD pt4 = GetPerpendicD(path[j], norms[k], group_delta_);
		PointD pt = IntersectPoint(pt1, pt2, pt3, pt4);
#ifdef USINGZ
		pt.z = ptQ.z;
#endif
		group.path.push_back(Point64(pt));
		//get the second intersect point through reflecion
		group.path.push_back(Point64(ReflectPoint(pt, ptQ)));
	}
}

void ClipperOffset::DoMiter(Group& group, const Path64& path, size_t j, size_t k, double cos_a)
{
	double q = group_delta_ / (cos_a + 1);
#ifdef USINGZ
	group.path.push_back(Point64(
		path[j].x + (norms[k].x + norms[j].x) * q,
		path[j].y + (norms[k].y + norms[j].y) * q,
		path[j].z));
#else
	group.path.push_back(Point64(
		path[j].x + (norms[k].x + norms[j].x) * q,
		path[j].y + (norms[k].y + norms[j].y) * q));
#endif
}

void ClipperOffset::DoRound(Group& group, const Path64& path, size_t j, size_t k, double angle)
{
	Point64 pt = path[j];
	PointD offsetVec = PointD(norms[k].x * group_delta_, norms[k].y * group_delta_);

	if (j == k) offsetVec.Negate();
#ifdef USINGZ
	group.path.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y, pt.z));
#else
	group.path.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y));
#endif
	if (angle > -PI + 0.01)	// avoid 180deg concave
	{
		int steps = static_cast<int>(std::ceil(steps_per_rad_ * std::abs(angle))); // #448, #456
		for (int i = 1; i < steps; ++i) // ie 1 less than steps
		{
			offsetVec = PointD(offsetVec.x * step_cos_ - step_sin_ * offsetVec.y,
				offsetVec.x * step_sin_ + offsetVec.y * step_cos_);
#ifdef USINGZ
			group.path.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y, pt.z));
#else
			group.path.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y));
#endif

		}
	}
	group.path.push_back(GetPerpendic(path[j], norms[j], group_delta_));
}

void ClipperOffset::OffsetPoint(Group& group, Path64& path, size_t j, size_t& k)
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

	if (cos_a > 0.99) // almost straight - less than 8 degrees
	{
		group.path.push_back(GetPerpendic(path[j], norms[k], group_delta_));
		if (cos_a < 0.9998) // greater than 1 degree (#424)
			group.path.push_back(GetPerpendic(path[j], norms[j], group_delta_)); // (#418)
	}
	else if (cos_a > -0.99 && (sin_a * group_delta_ < 0))
	{
		// is concave
		group.path.push_back(GetPerpendic(path[j], norms[k], group_delta_));
		// this extra point is the only (simple) way to ensure that
		// path reversals are fully cleaned with the trailing clipper
		group.path.push_back(path[j]); // (#405)
		group.path.push_back(GetPerpendic(path[j], norms[j], group_delta_));
	}	
	else if (join_type_ == JoinType::Round)
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

	k = j;
}

void ClipperOffset::OffsetPolygon(Group& group, Path64& path)
{
	for (Path64::size_type i = 0, j = path.size() -1; i < path.size(); j = i, ++i)
		OffsetPoint(group, path, i, j);
	group.paths_out.push_back(group.path);
}

void ClipperOffset::OffsetOpenJoined(Group& group, Path64& path)
{
	OffsetPolygon(group, path);
	std::reverse(path.begin(), path.end());
	
	//rebuild normals // BuildNormals(path);
	std::reverse(norms.begin(), norms.end());
	norms.push_back(norms[0]);
	norms.erase(norms.begin());
	NegatePath(norms);

	group.path.clear();
	OffsetPolygon(group, path);
}

void ClipperOffset::OffsetOpenPath(Group& group, Path64& path)
{
	// do the line start cap
	switch (end_type_)
	{
	case EndType::Butt:
#ifdef USINGZ
		group.path.push_back(Point64(
			path[0].x - norms[0].x * group_delta_,
			path[0].y - norms[0].y * group_delta_,
			path[0].z));
#else
		group.path.push_back(Point64(
			path[0].x - norms[0].x * group_delta_,
			path[0].y - norms[0].y * group_delta_));
#endif
		group.path.push_back(GetPerpendic(path[0], norms[0], group_delta_));
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
	switch (end_type_)
	{
	case EndType::Butt:
#ifdef USINGZ
		group.path.push_back(Point64(
			path[highI].x - norms[highI].x * group_delta_,
			path[highI].y - norms[highI].y * group_delta_,
			path[highI].z));
#else
		group.path.push_back(Point64(
			path[highI].x - norms[highI].x * group_delta_,
			path[highI].y - norms[highI].y * group_delta_));
#endif
		group.path.push_back(GetPerpendic(path[highI], norms[highI], group_delta_));
		break;
	case EndType::Round:
		DoRound(group, path, highI, highI, PI);
		break;
	default:
		DoSquare(group, path, highI, highI);
		break;
	}

	for (size_t i = highI, k = 0; i > 0; --i)
		OffsetPoint(group, path, i, k);
	group.paths_out.push_back(group.path);
}

void ClipperOffset::DoGroupOffset(Group& group)
{
	Rect64 r;
	int idx = -1;
	//the lowermost polygon must be an outer polygon. So we can use that as the
	//designated orientation for outer polygons (needed for tidy-up clipping)
	GetBoundsAndLowestPolyIdx(group.paths_in, r, idx);
	if (idx < 0) return;

	if (group.end_type == EndType::Polygon)
	{
		double area = Area(group.paths_in[idx]);
		//if (area == 0) return; // probably unhelpful (#430)
		group.is_reversed = (area < 0);
		if (group.is_reversed) group_delta_ = -delta_;
		else group_delta_ = delta_;
	} 
	else
	{
		group.is_reversed = false;
		group_delta_ = std::abs(delta_) * 0.5;
	}
	abs_group_delta_ = std::fabs(group_delta_);

	// do range checking
	if (!IsSafeOffset(r, abs_group_delta_))
	{
		DoError(range_error_i);
		error_code_ |= range_error_i;
		return;
	}

	join_type_	= group.join_type;
	end_type_ = group.end_type;

	//calculate a sensible number of steps (for 360 deg for the given offset
	if (group.join_type == JoinType::Round || group.end_type == EndType::Round)
	{
		// arcTol - when arc_tolerance_ is undefined (0), the amount of 
		// curve imprecision that's allowed is based on the size of the 
		// offset (delta). Obviously very large offsets will almost always 
		// require much less precision. See also offset_triginometry2.svg
		double arcTol = (arc_tolerance_ > floating_point_tolerance ?
			std::min(abs_group_delta_, arc_tolerance_) :
			std::log10(2 + abs_group_delta_) * default_arc_tolerance); 
		double steps_per_360 = PI / std::acos(1 - arcTol / abs_group_delta_);
		if (steps_per_360 > abs_group_delta_ * PI)
			steps_per_360 = abs_group_delta_ * PI;  //ie avoids excessive precision

		step_sin_ = std::sin(2 * PI / steps_per_360);
		step_cos_ = std::cos(2 * PI / steps_per_360);
		if (group_delta_ < 0.0) step_sin_ = -step_sin_;		
		steps_per_rad_ = steps_per_360 / (2 *PI);
	}

	bool is_joined =
		(end_type_ == EndType::Polygon) ||
		(end_type_ == EndType::Joined);
	Paths64::const_iterator path_iter;
	for(path_iter = group.paths_in.cbegin(); path_iter != group.paths_in.cend(); ++path_iter)
	{
		Path64 path = StripDuplicates(*path_iter, is_joined);
		Path64::size_type cnt = path.size();
		if (cnt == 0 || ((cnt < 3) && group.end_type == EndType::Polygon)) 
			continue;

		group.path.clear();
		if (cnt == 1) // single point - only valid with open paths
		{
			if (group_delta_ < 1) continue;
			//single vertex so build a circle or square ...
			if (group.join_type == JoinType::Round)
			{
				double radius = abs_group_delta_;
				group.path = Ellipse(path[0], radius, radius);
#ifdef USINGZ
				for (auto& p : group.path) p.z = path[0].z;
#endif
			}
			else
			{
				int d = (int)std::ceil(abs_group_delta_);
				r = Rect64(path[0].x - d, path[0].y - d, path[0].x + d, path[0].y + d);
				group.path = r.AsPath();
#ifdef USINGZ
				for (auto& p : group.path) p.z = path[0].z;
#endif
			}
			group.paths_out.push_back(group.path);
		}
		else
		{
			if ((cnt == 2) && (group.end_type == EndType::Joined))
			{
				if (group.join_type == JoinType::Round)
					end_type_ = EndType::Round;
				else
					end_type_ = EndType::Square;
			}

			BuildNormals(path);
			if (end_type_ == EndType::Polygon) OffsetPolygon(group, path);
			else if (end_type_ == EndType::Joined) OffsetOpenJoined(group, path);
			else OffsetOpenPath(group, path);
		}
	}
	solution.reserve(solution.size() + group.paths_out.size());
	copy(group.paths_out.begin(), group.paths_out.end(), back_inserter(solution));
	group.paths_out.clear();
}

void ClipperOffset::ExecuteInternal(double delta)
{
	error_code_ = 0;
	solution.clear();
	if (groups_.size() == 0) return;

	if (std::abs(delta) < 0.5)
	{
		for (const Group& group : groups_)
		{
			solution.reserve(solution.size() + group.paths_in.size());
			copy(group.paths_in.begin(), group.paths_in.end(), back_inserter(solution));
		}
	} 
	else
	{
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
}

void ClipperOffset::Execute(double delta, Paths64& paths)
{
	paths.clear();

	ExecuteInternal(delta);
	if (!solution.size()) return;

	paths = solution;
	//clean up self-intersections ...
	Clipper64 c;
	c.PreserveCollinear = false;
	//the solution should retain the orientation of the input
	c.ReverseSolution = reverse_solution_ != groups_[0].is_reversed;
#ifdef USINGZ
	if (zCallback64_) {
		c.SetZCallback(zCallback64_);
	}
#endif
	c.AddSubject(solution);
	if (groups_[0].is_reversed)
		c.Execute(ClipType::Union, FillRule::Negative, paths);
	else
		c.Execute(ClipType::Union, FillRule::Positive, paths);
}


void ClipperOffset::Execute(double delta, PolyTree64& polytree)
{
	polytree.Clear();

	ExecuteInternal(delta);
	if (!solution.size()) return;

	//clean up self-intersections ...
	Clipper64 c;
	c.PreserveCollinear = false;
	//the solution should retain the orientation of the input
	c.ReverseSolution = reverse_solution_ != groups_[0].is_reversed;
#ifdef USINGZ
	if (zCallback64_) {
		c.SetZCallback(zCallback64_);
	}
#endif
	c.AddSubject(solution);
	if (groups_[0].is_reversed)
		c.Execute(ClipType::Union, FillRule::Negative, polytree);
	else
		c.Execute(ClipType::Union, FillRule::Positive, polytree);
}

} // namespace
