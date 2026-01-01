@tool
extends Node

## Conversation Manager
## Manages multi-turn conversations with AI, maintaining context and history

class_name ConversationManager

signal conversation_updated()
signal ai_question(question: String)

var conversation_history: Array = []
var current_conversation_id: String = ""
var context_summary: String = ""
var max_history_length: int = 20  # Keep last 20 messages

## Start a new conversation
func start_conversation(conversation_id: String = "") -> void:
	if conversation_id == "":
		conversation_id = _generate_conversation_id()
	
	current_conversation_id = conversation_id
	conversation_history.clear()
	context_summary = ""
	
	# Add system message
	add_message("system", """You are an AI assistant integrated into Godot Engine. You help developers build games through natural conversation.

You can:
- Generate and modify code
- Create scenes and game systems
- Answer questions about the project
- Debug issues
- Suggest improvements
- Ask clarifying questions when needed

Be conversational, helpful, and proactive. When you need more information, ask questions.
When generating code, explain what you're doing and why.""")

## Add a message to conversation
func add_message(role: String, content: String, metadata: Dictionary = {}) -> void:
	var message = {
		"role": role,  # "user", "assistant", "system"
		"content": content,
		"timestamp": Time.get_unix_time_from_system(),
		"metadata": metadata
	}
	
	conversation_history.append(message)
	
	# Trim history if too long
	if conversation_history.size() > max_history_length:
		# Keep system message and recent messages
		var system_msg = conversation_history[0] if conversation_history[0].role == "system" else null
		var recent_messages = conversation_history.slice(-max_history_length + 1)
		conversation_history.clear()
		if system_msg:
			conversation_history.append(system_msg)
		conversation_history.append_array(recent_messages)
	
	# Update context summary periodically
	if conversation_history.size() % 5 == 0:
		_update_context_summary()
	
	conversation_updated.emit()

## Get conversation history for AI (formatted for Claude API)
func get_conversation_context() -> Array:
	var formatted = []
	for msg in conversation_history:
		# Claude API expects {"role": "...", "content": "..."}
		formatted.append({
			"role": msg.role,
			"content": msg.content
		})
	return formatted

## Get formatted conversation for display
func get_formatted_conversation() -> Array:
	var formatted = []
	for msg in conversation_history:
		if msg.role == "system":
			continue  # Skip system messages in display
		
		formatted.append({
			"role": msg.role,
			"content": msg.content,
			"timestamp": msg.timestamp
		})
	return formatted

## Update context summary for long conversations
func _update_context_summary() -> void:
	if conversation_history.size() < 10:
		return
	
	# Create a summary of earlier conversation
	var summary_parts = []
	var recent_count = 5
	
	# Summarize older messages
	var older_messages = conversation_history.slice(0, -recent_count)
	if older_messages.size() > 0:
		summary_parts.append("Earlier in this conversation:")
		for msg in older_messages:
			if msg.role != "system":
				var role_label = "User" if msg.role == "user" else "Assistant"
				summary_parts.append(f"{role_label}: {msg.content.substr(0, 100)}...")
	
	context_summary = "\n".join(summary_parts)

## Check if AI is asking a question
func detect_ai_question(response: String) -> bool:
	var question_patterns = [
		"\\?",
		"what",
		"which",
		"how",
		"when",
		"where",
		"why",
		"can you",
		"could you",
		"would you like"
	]
	
	var response_lower = response.to_lower()
	for pattern in question_patterns:
		if pattern in response_lower:
			# Check if it's actually a question (not just mentioning question words)
			if "?" in response or response_lower.begins_with(pattern):
				return true
	
	return false

## Extract question from response
func extract_question(response: String) -> String:
	# Look for question marks
	if "?" in response:
		var sentences = response.split("?")
		if sentences.size() > 0:
			return sentences[0].strip_edges() + "?"
	
	# Look for question patterns
	var question_patterns = [
		"what (.*)\\?",
		"which (.*)\\?",
		"how (.*)\\?",
		"when (.*)\\?",
		"where (.*)\\?",
		"why (.*)\\?"
	]
	
	for pattern in question_patterns:
		var regex = RegEx.create_from_string(pattern)
		var result = regex.search(response)
		if result:
			return result.get_string(0)
	
	return ""

## Clear conversation history
func clear_history() -> void:
	var system_msg = null
	if conversation_history.size() > 0 and conversation_history[0].role == "system":
		system_msg = conversation_history[0]
	
	conversation_history.clear()
	if system_msg:
		conversation_history.append(system_msg)
	
	context_summary = ""
	conversation_updated.emit()

## Generate conversation ID
func _generate_conversation_id() -> String:
	return "conv_" + str(Time.get_unix_time_from_system()) + "_" + str(randi() % 10000)

## Get conversation summary
func get_summary() -> String:
	if context_summary != "":
		return context_summary
	
	var summary = "Conversation with " + str(conversation_history.size()) + " messages"
	return summary
