/**
 * Agent loop: sends messages to the model, handles tool calls,
 * and continues until the model stops calling tools or limits are hit.
 */

import OpenAI from "openai";
import type {
  ChatCompletionMessageParam,
  ChatCompletionToolMessageParam,
} from "openai/resources/chat/completions";
import { toolDefinitions, executeTool } from "./tools.js";
import type { ConversationMessage, ToolCall } from "./types.js";

const SERVER_URL = process.env.SERVER_URL || "http://localhost:8000";
const MAX_TURNS = 10;
const TIMEOUT_MS = 120_000;

// System prompt matching training_data/tool_formatter.py:98-103
const SYSTEM_PROMPT =
  "You are a SQL assistant that builds pipe SQL queries. " +
  "You have access to tools for exploring database schemas and executing queries. " +
  "Pipe SQL uses |> to chain operators: FROM, WHERE, SELECT, AGGREGATE, JOIN, ORDER BY, LIMIT, EXTEND. " +
  "First explore the schema, then write the final pipe SQL query.";

const client = new OpenAI({
  baseURL: `${SERVER_URL}/v1`,
  apiKey: "not-needed", // Local server, no auth
});

export interface AgentResult {
  conversation: ConversationMessage[];
  predictedPipeSQL: string | null;
  numTurns: number;
  status: "success" | "timeout" | "error" | "no_prediction";
  error?: string;
}

/** Run the agent loop for a single question. */
export async function runAgent(
  question: string,
  dbId: string,
): Promise<AgentResult> {
  const messages: ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: `Database: ${dbId}\nQuestion: ${question}` },
  ];

  const conversation: ConversationMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: `Database: ${dbId}\nQuestion: ${question}` },
  ];

  let numTurns = 0;
  let lastExecutePipeSQL: string | null = null;
  const startTime = Date.now();

  try {
    while (numTurns < MAX_TURNS) {
      // Check timeout
      if (Date.now() - startTime > TIMEOUT_MS) {
        return {
          conversation,
          predictedPipeSQL: lastExecutePipeSQL,
          numTurns,
          status: lastExecutePipeSQL ? "success" : "timeout",
        };
      }

      numTurns++;

      // Call the model
      const response = await client.chat.completions.create({
        model: "pipe-sql",
        messages,
        tools: toolDefinitions,
        max_tokens: 512,
        temperature: 0.1,
      });

      const choice = response.choices[0];
      if (!choice) break;

      const assistantMsg = choice.message;
      const content = assistantMsg.content || "";
      const toolCalls = assistantMsg.tool_calls;

      // Record in conversation
      const convMsg: ConversationMessage = {
        role: "assistant",
        content,
      };
      if (toolCalls && toolCalls.length > 0) {
        convMsg.tool_calls = toolCalls.map((tc) => ({
          id: tc.id,
          type: "function" as const,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          },
        }));
      }
      conversation.push(convMsg);

      // Add to messages for next turn
      messages.push({
        role: "assistant",
        content: content,
        tool_calls: toolCalls,
      } as ChatCompletionMessageParam);

      // If no tool calls, agent is done
      if (!toolCalls || toolCalls.length === 0) {
        break;
      }

      // Execute each tool call
      for (const tc of toolCalls) {
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(tc.function.arguments);
        } catch {
          args = {};
        }

        // Track execute_pipe_sql calls for prediction extraction
        if (tc.function.name === "execute_pipe_sql" && args.pipe_sql) {
          lastExecutePipeSQL = args.pipe_sql as string;
        }

        const result = await executeTool(tc.function.name, args);

        // Add tool response
        const toolMsg: ChatCompletionToolMessageParam = {
          role: "tool",
          tool_call_id: tc.id,
          content: result,
        };
        messages.push(toolMsg);
        conversation.push({
          role: "tool",
          content: result,
          tool_call_id: tc.id,
        });
      }
    }
  } catch (err) {
    return {
      conversation,
      predictedPipeSQL: lastExecutePipeSQL,
      numTurns,
      status: "error",
      error: String(err),
    };
  }

  // Extract predicted pipe SQL
  const predictedSQL = extractPipeSQL(conversation, lastExecutePipeSQL);

  return {
    conversation,
    predictedPipeSQL: predictedSQL,
    numTurns,
    status: predictedSQL ? "success" : "no_prediction",
  };
}

/**
 * Extract the final pipe SQL prediction from the conversation.
 * Priority: (1) last execute_pipe_sql arg, (2) SQL code block, (3) text with |>
 */
function extractPipeSQL(
  conversation: ConversationMessage[],
  lastExecuteArg: string | null,
): string | null {
  // Priority 1: last execute_pipe_sql tool call argument
  if (lastExecuteArg) return lastExecuteArg;

  // Look at assistant messages in reverse for SQL
  for (let i = conversation.length - 1; i >= 0; i--) {
    const msg = conversation[i];
    if (msg.role !== "assistant") continue;

    // Priority 2: SQL code block
    const codeBlockMatch = msg.content.match(/```sql\s*\n([\s\S]*?)\n```/);
    if (codeBlockMatch) return codeBlockMatch[1].trim();

    // Priority 3: text containing |>
    const lines = msg.content.split("\n");
    for (const line of lines) {
      if (line.includes("|>") && line.trim().length > 10) {
        return line.trim();
      }
    }
  }

  return null;
}
