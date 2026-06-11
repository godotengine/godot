// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// File defining a coherent interface for different queuing strategies.

#ifndef DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_QUEUING_POLICY_H_
#define DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_QUEUING_POLICY_H_

#include <queue>
#include <stack>
#include <utility>

namespace draco {

template <class T>
class Queue {
 public:
  bool empty() const { return q_.empty(); }
  typename std::queue<T>::size_type size() const { return q_.size(); }
  void clear() { return q_.clear(); }
  void push(const T &value) { q_.push(value); }
  void push(T &&value) { q_.push(std::move(value)); }
  void pop() { q_.pop(); }
  typename std::queue<T>::const_reference front() const { return q_.front(); }

 private:
  std::queue<T> q_;
};

template <class T>
class Stack {
 public:
  bool empty() const { return s_.empty(); }
  typename std::stack<T>::size_type size() const { return s_.size(); }
  void clear() { return s_.clear(); }
  void push(const T &value) { s_.push(value); }
  void push(T &&value) { s_.push(std::move(value)); }
  void pop() { s_.pop(); }
  typename std::stack<T>::const_reference front() const { return s_.top(); }

 private:
  std::stack<T> s_;
};

template <class T, class Compare = std::less<T> >
class PriorityQueue {
  typedef std::priority_queue<T, std::vector<T>, Compare> QType;

 public:
  bool empty() const { return s_.empty(); }
  typename QType::size_type size() const { return s_.size(); }
  void clear() { return s_.clear(); }
  void push(const T &value) { s_.push(value); }
  void push(T &&value) { s_.push(std::move(value)); }
  void pop() { s_.pop(); }
  typename QType::const_reference front() const { return s_.top(); }

 private:
  QType s_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_QUEUING_POLICY_H_
