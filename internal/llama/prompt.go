package llama

import (
	"fmt"
	"time"
)

func DefaultSystemPrompt() string {
	return "Você é um assistente útil e objetivo."
}

func TodayString() string {
	// Keep similar to model card example.
	return time.Now().Format("02 Jan 2006")
}

func BuildSystemPrompt(systemPrompt string) string {
	today := TodayString()
	// Prompt format from model card.
	return fmt.Sprintf("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nCutting Knowledge Date: December 2023\nToday Date: %s\n\n%s<|eot_id|>", today, systemPrompt)
}

func BuildUserTurn(user string) string {
	return fmt.Sprintf("<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>", user)
}

func BuildAssistantHeader() string {
	return "<|start_header_id|>assistant<|end_header_id|>\n\n"
}
