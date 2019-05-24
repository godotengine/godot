// Copyright (c) 2018 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_OPT_LOOP_DEPENDENCE_H_
#define SOURCE_OPT_LOOP_DEPENDENCE_H_

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/scalar_analysis.h"

namespace spvtools {
namespace opt {

// Stores information about dependence between a load and a store wrt a single
// loop in a loop nest.
// DependenceInformation
// * UNKNOWN if no dependence information can be gathered or is gathered
//   for it.
// * DIRECTION if a dependence direction could be found, but not a
//   distance.
// * DISTANCE if a dependence distance could be found.
// * PEEL if peeling either the first or last iteration will break
//   dependence between the given load and store.
// * IRRELEVANT if it has no effect on the dependence between the given
//   load and store.
//
// If peel_first == true, the analysis has found that peeling the first
// iteration of this loop will break dependence.
//
// If peel_last == true, the analysis has found that peeling the last iteration
// of this loop will break dependence.
class DistanceEntry {
 public:
  enum DependenceInformation {
    UNKNOWN = 0,
    DIRECTION = 1,
    DISTANCE = 2,
    PEEL = 3,
    IRRELEVANT = 4,
    POINT = 5
  };
  enum Directions {
    NONE = 0,
    LT = 1,
    EQ = 2,
    LE = 3,
    GT = 4,
    NE = 5,
    GE = 6,
    ALL = 7
  };
  DependenceInformation dependence_information;
  Directions direction;
  int64_t distance;
  bool peel_first;
  bool peel_last;
  int64_t point_x;
  int64_t point_y;

  DistanceEntry()
      : dependence_information(DependenceInformation::UNKNOWN),
        direction(Directions::ALL),
        distance(0),
        peel_first(false),
        peel_last(false),
        point_x(0),
        point_y(0) {}

  explicit DistanceEntry(Directions direction_)
      : dependence_information(DependenceInformation::DIRECTION),
        direction(direction_),
        distance(0),
        peel_first(false),
        peel_last(false),
        point_x(0),
        point_y(0) {}

  DistanceEntry(Directions direction_, int64_t distance_)
      : dependence_information(DependenceInformation::DISTANCE),
        direction(direction_),
        distance(distance_),
        peel_first(false),
        peel_last(false),
        point_x(0),
        point_y(0) {}

  DistanceEntry(int64_t x, int64_t y)
      : dependence_information(DependenceInformation::POINT),
        direction(Directions::ALL),
        distance(0),
        peel_first(false),
        peel_last(false),
        point_x(x),
        point_y(y) {}

  bool operator==(const DistanceEntry& rhs) const {
    return direction == rhs.direction && peel_first == rhs.peel_first &&
           peel_last == rhs.peel_last && distance == rhs.distance &&
           point_x == rhs.point_x && point_y == rhs.point_y;
  }

  bool operator!=(const DistanceEntry& rhs) const { return !(*this == rhs); }
};

// Stores a vector of DistanceEntrys, one per loop in the analysis.
// A DistanceVector holds all of the information gathered in a dependence
// analysis wrt the loops stored in the LoopDependenceAnalysis performing the
// analysis.
class DistanceVector {
 public:
  explicit DistanceVector(size_t size) : entries(size, DistanceEntry{}) {}

  explicit DistanceVector(std::vector<DistanceEntry> entries_)
      : entries(entries_) {}

  DistanceEntry& GetEntry(size_t index) { return entries[index]; }
  const DistanceEntry& GetEntry(size_t index) const { return entries[index]; }

  std::vector<DistanceEntry>& GetEntries() { return entries; }
  const std::vector<DistanceEntry>& GetEntries() const { return entries; }

  bool operator==(const DistanceVector& rhs) const {
    if (entries.size() != rhs.entries.size()) {
      return false;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
      if (entries[i] != rhs.entries[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const DistanceVector& rhs) const { return !(*this == rhs); }

 private:
  std::vector<DistanceEntry> entries;
};

class DependenceLine;
class DependenceDistance;
class DependencePoint;
class DependenceNone;
class DependenceEmpty;

class Constraint {
 public:
  explicit Constraint(const Loop* loop) : loop_(loop) {}
  enum ConstraintType { Line, Distance, Point, None, Empty };

  virtual ConstraintType GetType() const = 0;

  virtual ~Constraint() {}

  // Get the loop this constraint belongs to.
  const Loop* GetLoop() const { return loop_; }

  bool operator==(const Constraint& other) const;

  bool operator!=(const Constraint& other) const;

#define DeclareCastMethod(target)                  \
  virtual target* As##target() { return nullptr; } \
  virtual const target* As##target() const { return nullptr; }
  DeclareCastMethod(DependenceLine);
  DeclareCastMethod(DependenceDistance);
  DeclareCastMethod(DependencePoint);
  DeclareCastMethod(DependenceNone);
  DeclareCastMethod(DependenceEmpty);
#undef DeclareCastMethod

 protected:
  const Loop* loop_;
};

class DependenceLine : public Constraint {
 public:
  DependenceLine(SENode* a, SENode* b, SENode* c, const Loop* loop)
      : Constraint(loop), a_(a), b_(b), c_(c) {}

  ConstraintType GetType() const final { return Line; }

  DependenceLine* AsDependenceLine() final { return this; }
  const DependenceLine* AsDependenceLine() const final { return this; }

  SENode* GetA() const { return a_; }
  SENode* GetB() const { return b_; }
  SENode* GetC() const { return c_; }

 private:
  SENode* a_;
  SENode* b_;
  SENode* c_;
};

class DependenceDistance : public Constraint {
 public:
  DependenceDistance(SENode* distance, const Loop* loop)
      : Constraint(loop), distance_(distance) {}

  ConstraintType GetType() const final { return Distance; }

  DependenceDistance* AsDependenceDistance() final { return this; }
  const DependenceDistance* AsDependenceDistance() const final { return this; }

  SENode* GetDistance() const { return distance_; }

 private:
  SENode* distance_;
};

class DependencePoint : public Constraint {
 public:
  DependencePoint(SENode* source, SENode* destination, const Loop* loop)
      : Constraint(loop), source_(source), destination_(destination) {}

  ConstraintType GetType() const final { return Point; }

  DependencePoint* AsDependencePoint() final { return this; }
  const DependencePoint* AsDependencePoint() const final { return this; }

  SENode* GetSource() const { return source_; }
  SENode* GetDestination() const { return destination_; }

 private:
  SENode* source_;
  SENode* destination_;
};

class DependenceNone : public Constraint {
 public:
  DependenceNone() : Constraint(nullptr) {}
  ConstraintType GetType() const final { return None; }

  DependenceNone* AsDependenceNone() final { return this; }
  const DependenceNone* AsDependenceNone() const final { return this; }
};

class DependenceEmpty : public Constraint {
 public:
  DependenceEmpty() : Constraint(nullptr) {}
  ConstraintType GetType() const final { return Empty; }

  DependenceEmpty* AsDependenceEmpty() final { return this; }
  const DependenceEmpty* AsDependenceEmpty() const final { return this; }
};

// Provides dependence information between a store instruction and a load
// instruction inside the same loop in a loop nest.
//
// The analysis can only check dependence between stores and loads with regard
// to the loop nest it is created with.
//
// The analysis can output debugging information to a stream. The output
// describes the control flow of the analysis and what information it can deduce
// at each step.
// SetDebugStream and ClearDebugStream are provided for this functionality.
//
// The dependency algorithm is based on the 1990 Paper
//   Practical Dependence Testing
//   Gina Goff, Ken Kennedy, Chau-Wen Tseng
//
// The algorithm first identifies subscript pairs between the load and store.
// Each pair is tested until all have been tested or independence is found.
// The number of induction variables in a pair determines which test to perform
// on it;
// Zero Index Variable (ZIV) is used when no induction variables are present
// in the pair.
// Single Index Variable (SIV) is used when only one induction variable is
// present, but may occur multiple times in the pair.
// Multiple Index Variable (MIV) is used when more than one induction variable
// is present in the pair.
class LoopDependenceAnalysis {
 public:
  LoopDependenceAnalysis(IRContext* context, std::vector<const Loop*> loops)
      : context_(context),
        loops_(loops),
        scalar_evolution_(context),
        debug_stream_(nullptr),
        constraints_{} {}

  // Finds the dependence between |source| and |destination|.
  // |source| should be an OpLoad.
  // |destination| should be an OpStore.
  // Any direction and distance information found will be stored in
  // |distance_vector|.
  // Returns true if independence is found, false otherwise.
  bool GetDependence(const Instruction* source, const Instruction* destination,
                     DistanceVector* distance_vector);

  // Returns true if |subscript_pair| represents a Zero Index Variable pair
  // (ZIV)
  bool IsZIV(const std::pair<SENode*, SENode*>& subscript_pair);

  // Returns true if |subscript_pair| represents a Single Index Variable
  // (SIV) pair
  bool IsSIV(const std::pair<SENode*, SENode*>& subscript_pair);

  // Returns true if |subscript_pair| represents a Multiple Index Variable
  // (MIV) pair
  bool IsMIV(const std::pair<SENode*, SENode*>& subscript_pair);

  // Finds the lower bound of |loop| as an SENode* and returns the result.
  // The lower bound is the starting value of the loops induction variable
  SENode* GetLowerBound(const Loop* loop);

  // Finds the upper bound of |loop| as an SENode* and returns the result.
  // The upper bound is the last value before the loop exit condition is met.
  SENode* GetUpperBound(const Loop* loop);

  // Returns true if |value| is between |bound_one| and |bound_two| (inclusive).
  bool IsWithinBounds(int64_t value, int64_t bound_one, int64_t bound_two);

  // Finds the bounds of |loop| as upper_bound - lower_bound and returns the
  // resulting SENode.
  // If the operations can not be completed a nullptr is returned.
  SENode* GetTripCount(const Loop* loop);

  // Returns the SENode* produced by building an SENode from the result of
  // calling GetInductionInitValue on |loop|.
  // If the operation can not be completed a nullptr is returned.
  SENode* GetFirstTripInductionNode(const Loop* loop);

  // Returns the SENode* produced by building an SENode from the result of
  // GetFirstTripInductionNode + (GetTripCount - 1) * induction_coefficient.
  // If the operation can not be completed a nullptr is returned.
  SENode* GetFinalTripInductionNode(const Loop* loop,
                                    SENode* induction_coefficient);

  // Returns all the distinct loops that appear in |nodes|.
  std::set<const Loop*> CollectLoops(
      const std::vector<SERecurrentNode*>& nodes);

  // Returns all the distinct loops that appear in |source| and |destination|.
  std::set<const Loop*> CollectLoops(SENode* source, SENode* destination);

  // Returns true if |distance| is provably outside the loop bounds.
  // |coefficient| must be an SENode representing the coefficient of the
  // induction variable of |loop|.
  // This method is able to handle some symbolic cases which IsWithinBounds
  // can't handle.
  bool IsProvablyOutsideOfLoopBounds(const Loop* loop, SENode* distance,
                                     SENode* coefficient);

  // Sets the ostream for debug information for the analysis.
  void SetDebugStream(std::ostream& debug_stream) {
    debug_stream_ = &debug_stream;
  }

  // Clears the stored ostream to stop debug information printing.
  void ClearDebugStream() { debug_stream_ = nullptr; }

  // Returns the ScalarEvolutionAnalysis used by this analysis.
  ScalarEvolutionAnalysis* GetScalarEvolution() { return &scalar_evolution_; }

  // Creates a new constraint of type |T| and returns the pointer to it.
  template <typename T, typename... Args>
  Constraint* make_constraint(Args&&... args) {
    constraints_.push_back(
        std::unique_ptr<Constraint>(new T(std::forward<Args>(args)...)));

    return constraints_.back().get();
  }

  // Subscript partitioning as described in Figure 1 of 'Practical Dependence
  // Testing' by Gina Goff, Ken Kennedy, and Chau-Wen Tseng from PLDI '91.
  // Partitions the subscripts into independent subscripts and minimally coupled
  // sets of subscripts.
  // Returns the partitioning of subscript pairs. Sets of size 1 indicates an
  // independent subscript-pair and others indicate coupled sets.
  using PartitionedSubscripts =
      std::vector<std::set<std::pair<Instruction*, Instruction*>>>;
  PartitionedSubscripts PartitionSubscripts(
      const std::vector<Instruction*>& source_subscripts,
      const std::vector<Instruction*>& destination_subscripts);

  // Returns the Loop* matching the loop for |subscript_pair|.
  // |subscript_pair| must be an SIV pair.
  const Loop* GetLoopForSubscriptPair(
      const std::pair<SENode*, SENode*>& subscript_pair);

  // Returns the DistanceEntry matching the loop for |subscript_pair|.
  // |subscript_pair| must be an SIV pair.
  DistanceEntry* GetDistanceEntryForSubscriptPair(
      const std::pair<SENode*, SENode*>& subscript_pair,
      DistanceVector* distance_vector);

  // Returns the DistanceEntry matching |loop|.
  DistanceEntry* GetDistanceEntryForLoop(const Loop* loop,
                                         DistanceVector* distance_vector);

  // Returns a vector of Instruction* which form the subscripts of the array
  // access defined by the access chain |instruction|.
  std::vector<Instruction*> GetSubscripts(const Instruction* instruction);

  // Delta test as described in Figure 3 of 'Practical Dependence
  // Testing' by Gina Goff, Ken Kennedy, and Chau-Wen Tseng from PLDI '91.
  bool DeltaTest(
      const std::vector<std::pair<SENode*, SENode*>>& coupled_subscripts,
      DistanceVector* dv_entry);

  // Constraint propagation as described in Figure 5 of 'Practical Dependence
  // Testing' by Gina Goff, Ken Kennedy, and Chau-Wen Tseng from PLDI '91.
  std::pair<SENode*, SENode*> PropagateConstraints(
      const std::pair<SENode*, SENode*>& subscript_pair,
      const std::vector<Constraint*>& constraints);

  // Constraint intersection as described in Figure 4 of 'Practical Dependence
  // Testing' by Gina Goff, Ken Kennedy, and Chau-Wen Tseng from PLDI '91.
  Constraint* IntersectConstraints(Constraint* constraint_0,
                                   Constraint* constraint_1,
                                   const SENode* lower_bound,
                                   const SENode* upper_bound);

  // Returns true if each loop in |loops| is in a form supported by this
  // analysis.
  // A loop is supported if it has a single induction variable and that
  // induction variable has a step of +1 or -1 per loop iteration.
  bool CheckSupportedLoops(std::vector<const Loop*> loops);

  // Returns true if |loop| is in a form supported by this analysis.
  // A loop is supported if it has a single induction variable and that
  // induction variable has a step of +1 or -1 per loop iteration.
  bool IsSupportedLoop(const Loop* loop);

 private:
  IRContext* context_;

  // The loop nest we are analysing the dependence of.
  std::vector<const Loop*> loops_;

  // The ScalarEvolutionAnalysis used by this analysis to store and perform much
  // of its logic.
  ScalarEvolutionAnalysis scalar_evolution_;

  // The ostream debug information for the analysis to print to.
  std::ostream* debug_stream_;

  // Stores all the constraints created by the analysis.
  std::list<std::unique_ptr<Constraint>> constraints_;

  // Returns true if independence can be proven and false if it can't be proven.
  bool ZIVTest(const std::pair<SENode*, SENode*>& subscript_pair);

  // Analyzes the subscript pair to find an applicable SIV test.
  // Returns true if independence can be proven and false if it can't be proven.
  bool SIVTest(const std::pair<SENode*, SENode*>& subscript_pair,
               DistanceVector* distance_vector);

  // Takes the form a*i + c1, a*i + c2
  // When c1 and c2 are loop invariant and a is constant
  // distance = (c1 - c2)/a
  //              < if distance > 0
  // direction =  = if distance = 0
  //              > if distance < 0
  // Returns true if independence is proven and false if it can't be proven.
  bool StrongSIVTest(SENode* source, SENode* destination, SENode* coeff,
                     DistanceEntry* distance_entry);

  // Takes for form a*i + c1, a*i + c2
  // where c1 and c2 are loop invariant and a is constant.
  // c1 and/or c2 contain one or more SEValueUnknown nodes.
  bool SymbolicStrongSIVTest(SENode* source, SENode* destination,
                             SENode* coefficient,
                             DistanceEntry* distance_entry);

  // Takes the form a1*i + c1, a2*i + c2
  // where a1 = 0
  // distance = (c1 - c2) / a2
  // Returns true if independence is proven and false if it can't be proven.
  bool WeakZeroSourceSIVTest(SENode* source, SERecurrentNode* destination,
                             SENode* coefficient,
                             DistanceEntry* distance_entry);

  // Takes the form a1*i + c1, a2*i + c2
  // where a2 = 0
  // distance = (c2 - c1) / a1
  // Returns true if independence is proven and false if it can't be proven.
  bool WeakZeroDestinationSIVTest(SERecurrentNode* source, SENode* destination,
                                  SENode* coefficient,
                                  DistanceEntry* distance_entry);

  // Takes the form a1*i + c1, a2*i + c2
  // where a1 = -a2
  // distance = (c2 - c1) / 2*a1
  // Returns true if independence is proven and false if it can't be proven.
  bool WeakCrossingSIVTest(SENode* source, SENode* destination,
                           SENode* coefficient, DistanceEntry* distance_entry);

  // Uses the def_use_mgr to get the instruction referenced by
  // SingleWordInOperand(|id|) when called on |instruction|.
  Instruction* GetOperandDefinition(const Instruction* instruction, int id);

  // Perform the GCD test if both, the source and the destination nodes, are in
  // the form a0*i0 + a1*i1 + ... an*in + c.
  bool GCDMIVTest(const std::pair<SENode*, SENode*>& subscript_pair);

  // Finds the number of induction variables in |node|.
  // Returns -1 on failure.
  int64_t CountInductionVariables(SENode* node);

  // Finds the number of induction variables shared between |source| and
  // |destination|.
  // Returns -1 on failure.
  int64_t CountInductionVariables(SENode* source, SENode* destination);

  // Takes the offset from the induction variable and subtracts the lower bound
  // from it to get the constant term added to the induction.
  // Returns the resuting constant term, or nullptr if it could not be produced.
  SENode* GetConstantTerm(const Loop* loop, SERecurrentNode* induction);

  // Marks all the distance entries in |distance_vector| that were relate to
  // loops in |loops_| but were not used in any subscripts as irrelevant to the
  // to the dependence test.
  void MarkUnsusedDistanceEntriesAsIrrelevant(const Instruction* source,
                                              const Instruction* destination,
                                              DistanceVector* distance_vector);

  // Converts |value| to a std::string and returns the result.
  // This is required because Android does not compile std::to_string.
  template <typename valueT>
  std::string ToString(valueT value) {
    std::ostringstream string_stream;
    string_stream << value;
    return string_stream.str();
  }

  // Prints |debug_msg| and "\n" to the ostream pointed to by |debug_stream_|.
  // Won't print anything if |debug_stream_| is nullptr.
  void PrintDebug(std::string debug_msg);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_DEPENDENCE_H_
