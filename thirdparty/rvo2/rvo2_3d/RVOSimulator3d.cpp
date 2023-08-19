/*
 * RVOSimulator.cpp
 * RVO2-3D Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

#include "RVOSimulator3d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Agent3d.h"
#include "KdTree3d.h"

namespace RVO3D {
	RVOSimulator3D::RVOSimulator3D() : defaultAgent_(NULL), kdTree_(NULL), globalTime_(0.0f), timeStep_(0.0f)
	{
		kdTree_ = new KdTree3D(this);
	}

	RVOSimulator3D::RVOSimulator3D(float timeStep, float neighborDist, size_t maxNeighbors, float timeHorizon, float radius, float maxSpeed, const Vector3 &velocity) : defaultAgent_(NULL), kdTree_(NULL), globalTime_(0.0f), timeStep_(timeStep)
	{
		kdTree_ = new KdTree3D(this);
		defaultAgent_ = new Agent3D();

		defaultAgent_->maxNeighbors_ = maxNeighbors;
		defaultAgent_->maxSpeed_ = maxSpeed;
		defaultAgent_->neighborDist_ = neighborDist;
		defaultAgent_->radius_ = radius;
		defaultAgent_->timeHorizon_ = timeHorizon;
		defaultAgent_->velocity_ = velocity;
	}

	RVOSimulator3D::~RVOSimulator3D()
	{
		if (defaultAgent_ != NULL) {
			delete defaultAgent_;
		}

		for (size_t i = 0; i < agents_.size(); ++i) {
			delete agents_[i];
		}

		if (kdTree_ != NULL) {
			delete kdTree_;
		}
	}

	size_t RVOSimulator3D::getAgentNumAgentNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->agentNeighbors_.size();
	}

	size_t RVOSimulator3D::getAgentAgentNeighbor(size_t agentNo, size_t neighborNo) const
	{
		return agents_[agentNo]->agentNeighbors_[neighborNo].second->id_;
	}

	size_t RVOSimulator3D::getAgentNumORCAPlanes(size_t agentNo) const
	{
		return agents_[agentNo]->orcaPlanes_.size();
	}

	const Plane &RVOSimulator3D::getAgentORCAPlane(size_t agentNo, size_t planeNo) const
	{
		return agents_[agentNo]->orcaPlanes_[planeNo];
	}

	void RVOSimulator3D::removeAgent(size_t agentNo)
	{
		delete agents_[agentNo];
		agents_[agentNo] = agents_.back();
		agents_.pop_back();
	}

	size_t RVOSimulator3D::addAgent(const Vector3 &position)
	{
		if (defaultAgent_ == NULL) {
			return RVO3D_ERROR;
		}

		Agent3D *agent = new Agent3D();

		agent->position_ = position;
		agent->maxNeighbors_ = defaultAgent_->maxNeighbors_;
		agent->maxSpeed_ = defaultAgent_->maxSpeed_;
		agent->neighborDist_ = defaultAgent_->neighborDist_;
		agent->radius_ = defaultAgent_->radius_;
		agent->timeHorizon_ = defaultAgent_->timeHorizon_;
		agent->velocity_ = defaultAgent_->velocity_;

		agent->id_ = agents_.size();

		agents_.push_back(agent);

		return agents_.size() - 1;
	}

	size_t RVOSimulator3D::addAgent(const Vector3 &position, float neighborDist, size_t maxNeighbors, float timeHorizon, float radius, float maxSpeed, const Vector3 &velocity)
	{
		Agent3D *agent = new Agent3D();

		agent->position_ = position;
		agent->maxNeighbors_ = maxNeighbors;
		agent->maxSpeed_ = maxSpeed;
		agent->neighborDist_ = neighborDist;
		agent->radius_ = radius;
		agent->timeHorizon_ = timeHorizon;
		agent->velocity_ = velocity;

		agent->id_ = agents_.size();

		agents_.push_back(agent);

		return agents_.size() - 1;
	}

	void RVOSimulator3D::doStep()
	{
		kdTree_->buildAgentTree(agents_);

		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->computeNeighbors(this);
			agents_[i]->computeNewVelocity(this);
		}

		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->update(this);
		}

		globalTime_ += timeStep_;
	}

	size_t RVOSimulator3D::getAgentMaxNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->maxNeighbors_;
	}

	float RVOSimulator3D::getAgentMaxSpeed(size_t agentNo) const
	{
		return agents_[agentNo]->maxSpeed_;
	}

	float RVOSimulator3D::getAgentNeighborDist(size_t agentNo) const
	{
		return agents_[agentNo]->neighborDist_;
	}

	const Vector3 &RVOSimulator3D::getAgentPosition(size_t agentNo) const
	{
		return agents_[agentNo]->position_;
	}

	const Vector3 &RVOSimulator3D::getAgentPrefVelocity(size_t agentNo) const
	{
		return agents_[agentNo]->prefVelocity_;
	}

	float RVOSimulator3D::getAgentRadius(size_t agentNo) const
	{
		return agents_[agentNo]->radius_;
	}

	float RVOSimulator3D::getAgentTimeHorizon(size_t agentNo) const
	{
		return agents_[agentNo]->timeHorizon_;
	}

	const Vector3 &RVOSimulator3D::getAgentVelocity(size_t agentNo) const
	{
		return agents_[agentNo]->velocity_;
	}

	float RVOSimulator3D::getGlobalTime() const
	{
		return globalTime_;
	}

	size_t RVOSimulator3D::getNumAgents() const
	{
		return agents_.size();
	}

	float RVOSimulator3D::getTimeStep() const
	{
		return timeStep_;
	}

	void RVOSimulator3D::setAgentDefaults(float neighborDist, size_t maxNeighbors, float timeHorizon, float radius, float maxSpeed, const Vector3 &velocity)
	{
		if (defaultAgent_ == NULL) {
			defaultAgent_ = new Agent3D();
		}

		defaultAgent_->maxNeighbors_ = maxNeighbors;
		defaultAgent_->maxSpeed_ = maxSpeed;
		defaultAgent_->neighborDist_ = neighborDist;
		defaultAgent_->radius_ = radius;
		defaultAgent_->timeHorizon_ = timeHorizon;
		defaultAgent_->velocity_ = velocity;
	}

	void RVOSimulator3D::setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors)
	{
		agents_[agentNo]->maxNeighbors_ = maxNeighbors;
	}

	void RVOSimulator3D::setAgentMaxSpeed(size_t agentNo, float maxSpeed)
	{
		agents_[agentNo]->maxSpeed_ = maxSpeed;
	}

	void RVOSimulator3D::setAgentNeighborDist(size_t agentNo, float neighborDist)
	{
		agents_[agentNo]->neighborDist_ = neighborDist;
	}

	void RVOSimulator3D::setAgentPosition(size_t agentNo, const Vector3 &position)
	{
		agents_[agentNo]->position_ = position;
	}

	void RVOSimulator3D::setAgentPrefVelocity(size_t agentNo, const Vector3 &prefVelocity)
	{
		agents_[agentNo]->prefVelocity_ = prefVelocity;
	}

	void RVOSimulator3D::setAgentRadius(size_t agentNo, float radius)
	{
		agents_[agentNo]->radius_ = radius;
	}

	void RVOSimulator3D::setAgentTimeHorizon(size_t agentNo, float timeHorizon)
	{
		agents_[agentNo]->timeHorizon_ = timeHorizon;
	}

	void RVOSimulator3D::setAgentVelocity(size_t agentNo, const Vector3 &velocity)
	{
		agents_[agentNo]->velocity_ = velocity;
	}

	void RVOSimulator3D::setTimeStep(float timeStep)
	{
		timeStep_ = timeStep;
	}
}
