import axios from "axios";
import { ENV } from "../config/env.js";

export interface EvalResult {
  metric_name: string;
  score?: number;
  explanation?: string;
  error?: string;
}

/**
 * Call Deepeval service to evaluate using specified metric
 */
export async function evalWithMetric(
  source: string,
  output: string,
  metric: string = "faithfulness",
  provider?: string
): Promise<EvalResult> {
  if (!source || source.trim() === "") {
    throw new Error("Source cannot be empty");
  }

  if (!output || output.trim() === "") {
    throw new Error("Output cannot be empty");
  }

  try {
    const payload: any = {
      source,
      output,
      metric
    };

    // Include provider if specified
    if (provider) {
      payload.provider = provider;
    }

    const res = await axios.post<EvalResult>(ENV.DEEPEVAL_URL, payload);

    return res.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      if ((err as any).code === "ECONNREFUSED") {
        throw new Error(
          `Deepeval service unavailable at ${ENV.DEEPEVAL_URL}. Is it running?`
        );
      }
      throw new Error(
        `Deepeval Error: ${err.response?.status} - ${err.message}`
      );
    }
    throw err;
  }
}

/**
 * Legacy function for backward compatibility - defaults to faithfulness
 */
export async function evalFaithfulness(
  source: string,
  output: string,
  provider?: string
): Promise<EvalResult> {
  return evalWithMetric(source, output, "faithfulness", provider);
}
