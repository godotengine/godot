/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Version   :  10.0 (beta)                                                     *
* Date      :  8 Noveber 2017                                                  *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
* Purpose   :  Triangulate clipping solutions                                  *
* License   : http://www.boost.org/LICENSE_1_0.txt                             *
*******************************************************************************/

#include "core/error_macros.h"

#include <stdlib.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "clipper_triangulation.h"
#include "clipper.h"

namespace clipperlib {

  //------------------------------------------------------------------------------
  // Miscellaneous functions ...
  //------------------------------------------------------------------------------

  inline  int64_t CrossProductVal(const Point64 pt1,
    const Point64 pt2, const Point64 pt3, int64_t &val)
  {
    val = ((pt2.x - pt1.x) * (pt1.y - pt3.y) - (pt1.x - pt3.x) * (pt2.y - pt1.y));
    return val;
  }
  //------------------------------------------------------------------------------

  inline bool IsHotEdge(const Active &e) { return (e.outrec); }
  //------------------------------------------------------------------------------

  inline bool IsStartSide(const Active &e) { return (&e == e.outrec->start_e); }
  //------------------------------------------------------------------------------

  inline OutPt* GetOutPt(Active &e)
  {
    return (IsStartSide(e)) ? e.outrec->pts : e.outrec->pts->next;
  }
  //------------------------------------------------------------------------------

  inline Active* GetLeftAdjacentHotEdge(Active &e)
  {
    Active * result = e.prev_in_ael;
    while (result && !IsHotEdge(*result)) result = result->prev_in_ael;
    return result;
  }
  //------------------------------------------------------------------------------

  inline Active* GetRightAdjacentHotEdge(Active &e)
  {
    Active * result = e.next_in_ael;
    while (result && !IsHotEdge(*result)) result = result->next_in_ael;
    return result;
  }
  //------------------------------------------------------------------------------

  inline void DisposeOutPt(OutPt *op)
  {
    if (op->prev) op->prev->next = op->next;
    if (op->next) op->next->prev = op->prev;
    OutPtTri *opt = static_cast<OutPtTri *>(op);
    if (opt->right_outrec) opt->right_outrec->left_outpt = NULL;
    delete op;
  }
  //------------------------------------------------------------------------------

  void UpdateHelper(OutRec *right_outrec, OutPt *left_outpt)
  {
    OutPtTri *left_opt = static_cast<OutPtTri *>(left_outpt);
    OutRecTri *right_ort = static_cast<OutRecTri *>(right_outrec);
    if (left_opt && left_opt->right_outrec)
      left_opt->right_outrec->left_outpt = NULL;
    if (right_ort->left_outpt)
      static_cast<OutPtTri *>(right_ort->left_outpt)->right_outrec = NULL;
    right_ort->left_outpt = left_opt;
    if (left_opt) left_opt->right_outrec = right_ort;
  }
  //------------------------------------------------------------------------------

  void Update(OutPt *op, OutRec *outrec)
  {
    OutPt *op2 = op;
    do {
      OutPtTri *opt = static_cast<OutPtTri *>(op2);
      if (opt->right_outrec)
        UpdateHelper(opt->right_outrec, NULL);
      opt->outrec = outrec;
      op2 = op2->next;
    } while (op2 != op);
  }
  //------------------------------------------------------------------------------

  inline int PointCount(OutPt *op)
  {
    if (!op) return 0;
    OutPt *p = op;
    int cnt = 0;
    do {
      cnt++;
      p = p->next;
    } while (p != op);
    return cnt;
  }
  //------------------------------------------------------------------------------

  OutPtTri* InsertPt(const Point64 &pt, OutPt *insert_after)
  {
    OutPtTri *result = new OutPtTri();
    result->pt = pt;
    result->prev = insert_after;
    result->next = insert_after->next;
    result->outrec = static_cast<OutPtTri *>(insert_after)->outrec;
    static_cast<OutPtTri *>(result)->right_outrec = NULL;
    insert_after->next->prev = result;
    insert_after->next = result;
    return result;
  }

  //------------------------------------------------------------------------------
  // ClipperTri methods ...
  //------------------------------------------------------------------------------

  void ClipperTri::AddPolygon(const Point64 pt1, const Point64 pt2, const Point64 pt3)
  {
    Path p;
    p.reserve(3);
    p.push_back(pt3);
    p.push_back(pt2);
    p.push_back(pt1);
    triangles_.push_back(p);
  }
  //------------------------------------------------------------------------------

  OutPt* ClipperTri::CreateOutPt()
  {
    OutPtTri *result = new OutPtTri();
    result->outrec = NULL;
    result->right_outrec = NULL;
    return result;
  }
  //------------------------------------------------------------------------------

  OutRec* ClipperTri::CreateOutRec()
  {
    OutRecTri *result = new OutRecTri();
    result->left_outpt = NULL;
    return result;
  }
  //------------------------------------------------------------------------------

  void ClipperTri::Triangulate(OutRec *outrec)
  {
    OutPt *op = outrec->pts;
    if (op->next == op->prev) return;
    OutPt *end_op = op->next;
    for (;;)
    {
      OutPt *op2 = op;
      int64_t cpval = 0;
      while (op->prev != end_op) {
        if (CrossProductVal(op->pt, op->prev->pt, op->prev->prev->pt, cpval) >= 0)
          break;
        if (op2 != op) {
          //Due to rounding, the clipping algorithm can occasionally produce
          //tiny self-intersections and these need removing ...
          if (CrossProductVal(op2->pt, op->pt, op->prev->prev->pt, cpval) > 0) {
            OutPtTri *opt = static_cast<OutPtTri *>(op);
            if (opt->outrec) UpdateHelper(opt->outrec, op2);
            DisposeOutPt(op);
            op = op2;
            continue;
          }
        }
        op = op->prev;
      }

      if (op->prev == end_op) break;
      if (cpval) AddPolygon(op->pt, op->prev->pt, op->prev->prev->pt);
      OutPtTri *opt = static_cast<OutPtTri *>(op->prev);
      if (opt->outrec) UpdateHelper(opt->outrec, op);
      DisposeOutPt(op->prev);
      if (op != outrec->pts) op = op->next;
    }
  }
  //------------------------------------------------------------------------------

  OutPt* ClipperTri::AddOutPt(Active &e, const Point64 pt) {

    OutPt *result = Clipper::AddOutPt(e, pt);
    OutPtTri *opt = static_cast<OutPtTri *>(result);
    opt->outrec = e.outrec;
    last_op_ = result;
    Triangulate(e.outrec);
    //Triangulate() above may assign Result.OutRecRt so ...
    if (IsStartSide(e) && ! opt->right_outrec) {
      Active *e2 = GetRightAdjacentHotEdge(e);
    if (e2) UpdateHelper(e2->outrec, result);
    }
    return result;
  }
  //------------------------------------------------------------------------------

  void ClipperTri::AddLocalMinPoly(Active &e1, Active &e2, const Point64 pt)
  {
    Clipper::AddLocalMinPoly(e1, e2, pt);
    OutRec *locMinOr = e1.outrec;
    static_cast<OutPtTri *>(locMinOr->pts)->outrec = locMinOr;
    UpdateHelper(locMinOr, locMinOr->pts);
    if (locMinOr->flag == orOuter)return;

    //do 'keyholing' ...
    Active *e = GetRightAdjacentHotEdge(e1);
    if (e == &e2) e = GetRightAdjacentHotEdge(e2);
    if (!e) e = GetLeftAdjacentHotEdge(e1);
    OutPt *botLft = static_cast<OutRecTri *>(e->outrec)->left_outpt;
    OutPt *botRt = GetOutPt(*e);

    if (!botLft || botRt->pt.y < botLft->pt.y) botLft = botRt;

    botRt = InsertPt(botLft->pt, botLft->prev);
    OutRec *botOr = static_cast<OutPtTri *>(botLft)->outrec;
    if (!botOr->pts) botOr = botOr->owner;

    OutPt *startOp = botOr->pts;
    OutPt *endOp = startOp->next;

    locMinOr->flag = orOuter;
    locMinOr->owner = NULL;
    OutPt *locMinLft = locMinOr->pts;
    OutPt *locMinRt = InsertPt(locMinLft->pt, locMinLft);

    //locMinOr will contain the polygon to the right of the join (ascending),
    //and botOr will contain the polygon to the left of the join (descending).

    //tail -> botRt -> locMinRt : locMinRt is joined by botRt tail
    locMinRt->next = endOp;
    endOp->prev = locMinRt;
    botRt->next = locMinRt;
    locMinRt->prev = botRt;
    locMinOr->pts = locMinRt;

    //locMinLft -> botLft -> head : locMinLft joins behind botLft (left)
    startOp->next = locMinLft;
    locMinLft->prev = startOp;
    botLft->prev = locMinLft;
    locMinLft->next = botLft;
    static_cast<OutPtTri *>(locMinLft)->outrec = botOr; //ie abreviated update()

    Update(locMinRt, locMinOr); //updates the outrec for each op

    //exchange endE's ...
    e = botOr->end_e;
    botOr->end_e = locMinOr->end_e;
    locMinOr->end_e = e;
    botOr->end_e->outrec = botOr;
    locMinOr->end_e->outrec = locMinOr;

    //update helper info  ...
    UpdateHelper(locMinOr, locMinRt);
    UpdateHelper(botOr, botOr->pts);
    Triangulate(locMinOr);
    Triangulate(botOr);
  }
  //------------------------------------------------------------------------------

  void ClipperTri::AddLocalMaxPoly(Active &e1, Active &e2, const Point64 pt)
  {
    OutRec *outrec = e1.outrec;
    //very occasionally IsStartSide(e1) is wrong so ...
    bool is_outer = IsStartSide(e1) || (e1.outrec == e2.outrec);
    if (is_outer) {
      OutRecTri *ort = static_cast<OutRecTri *>(e1.outrec);
      if (ort->left_outpt) UpdateHelper(outrec, NULL);
      UpdateHelper(e2.outrec, NULL);
    }

    Clipper::AddLocalMaxPoly(e1, e2, pt);

    if (!outrec->pts) outrec = outrec->owner;

    if (is_outer) {
      if (static_cast<OutPtTri *>(outrec->pts)->right_outrec)
          UpdateHelper(static_cast<OutPtTri *>(outrec->pts)->right_outrec, NULL);
      else if (static_cast<OutPtTri *>(outrec->pts->next)->right_outrec)
        UpdateHelper(static_cast<OutPtTri *>(outrec->pts->next)->right_outrec, NULL);
    }
    else {
      Active *e = GetRightAdjacentHotEdge(e2);
      if (e) UpdateHelper(e->outrec, last_op_);
      Update(outrec->pts, outrec);
    }
    Triangulate(outrec);
  }
  //------------------------------------------------------------------------------

  bool ClipperTri::Execute(ClipType clipType, Paths &solution, FillRule fr)
  {
    solution.clear();
    if (clipType == ctNone) return true;
    bool result = ExecuteInternal(clipType, fr);
    if (result) BuildResult(solution);
    CleanUp();
    return result;
  }
  //------------------------------------------------------------------------------

  void ClipperTri::BuildResult(Paths &paths) {
    paths.clear();
    paths.reserve(triangles_.size());
    for (Paths::iterator it = triangles_.begin(); it != triangles_.end(); ++it) {
      ERR_FAIL_COND((*it).size() != 3);
      paths.push_back(*it);
    }
  }
  //------------------------------------------------------------------------------

} //namespace
