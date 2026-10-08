import { env, SYSTEM_PROMPT } from '../config'

export type GeminiMessage = {
  role: 'user' | 'model'
  parts: Array<{ text: string }>
}

export type AskHandlers = {
  onText: (delta: string) => void
  onTool: (name: string) => void
}

let activeAbort: AbortController | null = null

export function cancel(): void {
  if (activeAbort) {
    activeAbort.abort()
    activeAbort = null
  }
}

const FALLBACK_MODELS = [
  'gemini-3.5-flash',
  'gemini-3.1-flash-lite',
  'gemini-3.8-flash',
  'gemini-flash-latest',
]

/**
 * Calls Google Gemini API with streaming SSE deltas.
 */
export async function ask(
  history: Array<{ role: string; content?: any; text?: string }>,
  handlers: AskHandlers,
): Promise<{ text: string; tools: string[] }> {
  cancel()
  const abortController = new AbortController()
  activeAbort = abortController

  const apiKey = env.geminiKey
  if (!apiKey) {
    const msg = 'No Gemini API key configured. Please set VITE_GEMINI_API_KEY in .env.local.'
    handlers.onText(msg)
    return { text: msg, tools: [] }
  }

  // Convert conversation history to Gemini contents format
  const contents: GeminiMessage[] = []
  for (const item of history) {
    const role = item.role === 'user' ? 'user' : 'model'
    let text = ''
    if (typeof item.content === 'string') {
      text = item.content
    } else if (typeof item.text === 'string') {
      text = item.text
    } else if (Array.isArray(item.content)) {
      text = item.content
        .map((p: any) => (typeof p === 'string' ? p : p?.text || ''))
        .filter(Boolean)
        .join(' ')
    }
    if (text.trim()) {
      contents.push({
        role,
        parts: [{ text: text.trim() }],
      })
    }
  }

  // Model candidate selection
  const chosenModel = env.geminiModel || 'gemini-3.5-flash'
  const modelsToTry = [chosenModel, ...FALLBACK_MODELS.filter((m) => m !== chosenModel)]

  let fullText = ''
  let lastError: Error | null = null

  for (const model of modelsToTry) {
    if (abortController.signal.aborted) break

    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${encodeURIComponent(
        apiKey,
      )}`

      const body = {
        systemInstruction: {
          parts: [{ text: SYSTEM_PROMPT }],
        },
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 2048,
        },
      }

      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
        },
        body: JSON.stringify(body),
        signal: abortController.signal,
      })

      if (!res.ok) {
        const errorText = await res.text()
        let errObj: any = {}
        try {
          errObj = JSON.parse(errorText)
        } catch {}
        const code = res.status
        const detail = errObj.error?.message || errorText || `HTTP ${code}`
        
        // If 503 (high demand) or 404, try next candidate model
        if ((code === 503 || code === 404) && model !== modelsToTry[modelsToTry.length - 1]) {
          console.warn(`[gemini] ${model} returned ${code}, attempting fallback model...`)
          continue
        }
        throw new Error(`Gemini API error (${code}): ${detail}`)
      }

      const reader = res.body?.getReader()
      if (!reader) {
        throw new Error('No readable response stream received from Gemini API.')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('data: ')) continue
          const rawJson = trimmed.slice(6).trim()
          if (!rawJson || rawJson === '[DONE]') continue

          try {
            const data = JSON.parse(rawJson)
            const parts = data.candidates?.[0]?.content?.parts ?? []
            for (const part of parts) {
              if (part.text) {
                fullText += part.text
                handlers.onText(part.text)
              }
            }
          } catch {
            // Partial JSON across chunks or heartbeat, continue
          }
        }
      }

      // If text was generated successfully, return
      if (fullText.trim()) {
        activeAbort = null
        return { text: fullText.trim(), tools: [] }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return { text: fullText.trim(), tools: [] }
      }
      lastError = err
      console.error(`[gemini] error with model ${model}:`, err.message)
    }
  }

  activeAbort = null
  if (lastError && !fullText.trim()) {
    const errorMsg = `Unable to generate response: ${lastError.message}`
    handlers.onText(errorMsg)
    return { text: errorMsg, tools: [] }
  }

  return { text: fullText.trim(), tools: [] }
}
