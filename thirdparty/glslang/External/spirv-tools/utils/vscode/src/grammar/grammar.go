// Copyright (C) 2019 Google Inc.
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

// Package grammar holds the JSON type definitions for the SPIR-V grammar schema.
//
// See https://www.khronos.org/registry/spir-v/specs/unified1/MachineReadableGrammar.html
// for more information.
package grammar

// Root is the top-level structure of the JSON grammar.
type Root struct {
	MagicNumber  string        `json:"magic_number"`
	MajorVersion int           `json:"major_version"`
	MinorVersion int           `json:"minor_version"`
	Revision     int           `json:"revision"`
	Instructions []Instruction `json:"instructions"`
	OperandKinds []OperandKind `json:"operand_kinds"`
}

// Instruction holds information about a specific SPIR-V instruction.
type Instruction struct {
	Opname   string    `json:"opname"`
	Class    string    `json:"class"`
	Opcode   int       `json:"opcode"`
	Operands []Operand `json:"operands"`
}

// Operand contains information about a logical operand for an instruction.
type Operand struct {
	Kind       string     `json:"kind"`
	Name       string     `json:"name"`
	Quantifier Quantifier `json:"quantifier"`
}

// OperandKind contains information about a specific operand kind.
type OperandKind struct {
	Category   string      `json:"category"`
	Kind       string      `json:"kind"`
	Enumerants []Enumerant `json:"enumerants"`
	Bases      []string    `json:"bases"`
}

// Enumerant contains information about an enumerant in an enum.
type Enumerant struct {
	Enumerant    string      `json:"enumerant"`
	Value        interface{} `json:"value"`
	Capabilities []string    `json:"capabilities"`
	Parameters   []Parameter `json:"parameters"`
	Version      string      `json:"version"`
}

// Parameter contains information about a logical parameter for an enumerant.
type Parameter struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
}

// Quantifier indicates the number of times the quantified term may appear.
type Quantifier string

const (
	// Once indicates the quantified term may appear exactly once.
	Once Quantifier = ""
	// ZeroOrOnce indicates the quantified term may appear zero or one
	// time; an optional term.
	ZeroOrOnce Quantifier = "?"
	// ZeroOrMany indicates the quantified term may appear any number of
	// times.
	ZeroOrMany Quantifier = "*"
)
