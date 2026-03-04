/** Benchmark question from the server. */
export interface BenchmarkQuestion {
  id: string;
  question: string;
  db_id: string;
  gold_sql: string;
  source: string;
}

/** Result for a single benchmark question. */
export interface BenchmarkResult {
  id: string;
  question: string;
  db_id: string;
  gold_sql: string;
  predicted_pipe_sql: string | null;
  conversation: ConversationMessage[];
  num_turns: number;
  duration_ms: number;
  status: "success" | "timeout" | "error" | "no_prediction";
  error?: string;
}

export interface ConversationMessage {
  role: string;
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}
