import axios from "axios";
import { ENV } from "../config/env.js";

/**
 * Call OpenAI API
 */
async function callOpenAI(prompt: string, model: string, temperature: number = 0.7): Promise<string> {
  const res = await axios.post(
    "https://api.openai.com/v1/chat/completions",
    {
      model,
      messages: [{ role: "user", content: prompt }],
      temperature
    },
    {
      headers: {
        Authorization: `Bearer ${ENV.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      }
    }
  );

  return res.data.choices[0].message.content as string;
}



/**
 * Call LLM with provider selection
 */
export async function callLLM(
  prompt: string,
  model?: string,
  temperature?: number
): Promise<string> {
  const selectedModel = model || "gpt-4o-mini"; // Default model if not provided
  const selectedTemperature = temperature !== undefined ? temperature : 0.7; // Default temperature

  if (!prompt || prompt.trim() === "") {
    throw new Error("Prompt cannot be empty");
  }

  if (!selectedModel || selectedModel.trim() === "") {
    throw new Error(`Invalid model: "${selectedModel}". Model cannot be empty.`);
  }

  console.log(`Using model: ${selectedModel}, temperature: ${selectedTemperature}`);

  try {
    return await callOpenAI(prompt, selectedModel, selectedTemperature);
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      throw new Error(
        `LLM API Error: ${err.response?.status} - ${err.response?.data?.error?.message || err.message}`
      );
    }
    throw err;
  }
}
