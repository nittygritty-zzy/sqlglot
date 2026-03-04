/**
 * Tool definitions matching the training data format.
 * Each tool calls the Python server's /tools/ endpoints.
 */

import type { ChatCompletionTool } from "openai/resources/chat/completions";

const SERVER_URL = process.env.SERVER_URL || "http://localhost:8000";

/** OpenAI function tool definitions matching training_data/tool_formatter.py */
export const toolDefinitions: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "list_tables",
      description: "List all table names in a database.",
      parameters: {
        type: "object",
        properties: {
          db_id: { type: "string", description: "Database identifier" },
        },
        required: ["db_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "describe_table",
      description:
        "Get columns, types, primary keys, and foreign keys for a table.",
      parameters: {
        type: "object",
        properties: {
          db_id: { type: "string", description: "Database identifier" },
          table_name: { type: "string", description: "Table name" },
        },
        required: ["db_id", "table_name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "sample_data",
      description: "Return sample rows from a table.",
      parameters: {
        type: "object",
        properties: {
          db_id: { type: "string", description: "Database identifier" },
          table_name: { type: "string", description: "Table name" },
          limit: {
            type: "integer",
            description: "Max rows to return",
            default: 5,
          },
        },
        required: ["db_id", "table_name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "execute_pipe_sql",
      description: "Execute a pipe SQL query and return results.",
      parameters: {
        type: "object",
        properties: {
          db_id: { type: "string", description: "Database identifier" },
          pipe_sql: { type: "string", description: "Pipe SQL query" },
        },
        required: ["db_id", "pipe_sql"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "validate_pipe_sql",
      description: "Check pipe SQL syntax validity.",
      parameters: {
        type: "object",
        properties: {
          pipe_sql: {
            type: "string",
            description: "Pipe SQL query to validate",
          },
        },
        required: ["pipe_sql"],
      },
    },
  },
];

/** Execute a tool by calling the Python server. */
export async function executeTool(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  const url = `${SERVER_URL}/tools/${name}`;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: args }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      return `Error (HTTP ${resp.status}): ${text}`;
    }

    const data = (await resp.json()) as { result: string };
    return data.result;
  } catch (err) {
    return `Error calling tool ${name}: ${err}`;
  }
}
