require("dotenv").config();

const express = require("express");
const { GoogleGenAI } = require("@google/genai");
const fs = require("fs/promises");
const os = require("os");
const path = require("path");
const { execFile, spawn } = require("child_process");
const crypto = require("crypto");

const app = express();

app.use(express.json());
app.use(express.static("public"));

const PORT = process.env.PORT || 3000;

const AI_PROVIDER = (process.env.AI_PROVIDER || "ollama").toLowerCase();
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen2.5:0.5b";
const OLLAMA_URL =
    process.env.OLLAMA_URL || "http://127.0.0.1:11434/api/generate";
const GEMINI_TEXT_MODEL =
    process.env.GEMINI_TEXT_MODEL ||
    process.env.GEMINI_MODEL ||
    "gemini-3.1-flash-lite";
const GEMINI_IMAGE_MODEL =
    process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const PYTHON = process.env.PYTHON || "python";
const FFMPEG = process.env.FFMPEG_PATH || "ffmpeg";
const voiceWorker = path.join(__dirname, "voice_worker.py");
const instagramAudioFiles = new Map();
const instagramImageFiles = new Map();
const instagramConversations = new Map();
const INSTAGRAM_AUDIO_MAX_BYTES = 25 * 1024 * 1024;
const INSTAGRAM_IMAGE_MAX_BYTES = 8 * 1024 * 1024;
const INSTAGRAM_AUDIO_RETENTION_MS = 60 * 1000;
const INSTAGRAM_CONTEXT_LIMIT = 4;


/* =========================================================
   RESPONSE FORMAT
========================================================= */

const replyFormat = {
    type: "object",
    properties: {
        reply: { type: "string" },
        language: { type: "string" },
        script: { type: "string" }
    },
    required: ["reply", "language", "script"],
    additionalProperties: false
};


/* =========================================================
   SYSTEM PROMPT
========================================================= */

const systemPrompt = `
You are a capable, accurate, friendly, multilingual general-purpose assistant.
Answer normal conversational questions naturally and directly. Do not unnecessarily say that you are an AI.
Be warm and helpful without being overly casual, repetitive, or theatrical.

CODING ASSISTANT:
- You can write complete, runnable programming solutions when requested.
- Support JavaScript, Node.js, Express, React, HTML, CSS, Python, SQL, MongoDB, and other common languages and tools.
- Debug code provided by the user: identify the likely cause, explain the error clearly, and provide corrected complete code when practical.
- For coding requests, organize the answer naturally as: a short explanation, complete code, how to run or use it, and important notes only when necessary.
- Use Markdown-style formatting when it improves readability.
- Put code in fenced Markdown blocks with the correct language tag, such as a JavaScript code fence.
- Do not omit essential code with placeholders unless the user asks for a sketch.
- When code is requested, include the complete implementation in a fenced code block before usage notes.
- When debugging, inspect every referenced variable, function, import, and syntax element before declaring the code correct.
- For a broken example, explicitly name the error, show corrected complete code, and explain the correction briefly.

REASONING AND ACCURACY:
- Understand the user's actual goal, including follow-up messages and prior conversation context.
- Give concise answers for simple questions and more detailed answers for complex coding requests.
- Do not invent APIs, facts, error causes, test results, or configuration values.
- When uncertain, say what is uncertain and state the safest next step.
- Ask a focused clarification only when the missing information prevents a useful answer.

LANGUAGE RULES:
- Detect the language of the user's latest message.
- Always reply in the SAME language as the user.
- If the user writes a language using English/Roman letters, reply using English/Roman letters.
- If the user writes using native script, reply using the same native script.
- Understand transliterated languages.
- Understand mixed-language messages naturally.
- Understand slang, abbreviations, spelling mistakes and casual speech.
- Do not translate unless the user asks.
- Do not switch to English unless the user uses English or asks for English.
- Do not unnecessarily mix languages.

ROMANIZED LANGUAGE:
- Romanized Telugu -> Romanized Telugu.
- Romanized Hindi -> Romanized Hindi.
- Romanized Tamil -> Romanized Tamil.
- Romanized Kannada -> Romanized Kannada.
- Romanized Malayalam -> Romanized Malayalam.
- Romanized Marathi -> Romanized Marathi.
- Romanized Bengali -> Romanized Bengali.

NATIVE SCRIPT:
- Telugu -> Telugu script.
- Hindi -> Devanagari.
- Tamil -> Tamil script.
- Kannada -> Kannada script.
- Malayalam -> Malayalam script.
- Bengali -> Bengali script.
- Marathi -> Devanagari.

NATURAL RESPONSE:
- Understand the meaning, not just individual words.
- Reply naturally and conversationally.
- Do not give literal translations.
- Do not repeat the user's message.
- Answer the actual question.

ACCURACY:
- Give factual answers carefully.
- Do not invent facts.
- Never guess when you are unsure.
- If uncertain, clearly say that you are uncertain.
- Do not present guesses as facts.
- For calculations, calculate carefully.

CONVERSATION:
- Understand the context of the conversation.
- Give natural, clear and helpful answers.
- Maintain useful context across follow-up messages when conversation history is provided.
- Do not unnecessarily repeat information.

INSTAGRAM:
- Keep replies suitable for Instagram DMs.
- Keep ordinary Instagram replies concise, but include complete code when a coding request requires it.

IMPORTANT:
- Never output analysis.
- Never output chain-of-thought.
- Never output <think>, </think>, /think or /no_think.
- Return ONLY valid JSON.
`;


/* =========================================================
   CLEAN MODEL OUTPUT
========================================================= */

const controlTokenPattern =
    /(?:<think>[\s\S]*?<\/think>|<think>|<\/think>|(?<![\w])\/?no_think(?![\w])|(?<![\w])\/?think(?![\w])|<\|[^|]+\|>)/gi;


function cleanReply(text) {
    const cleaned = String(text || "")
        .replace(controlTokenPattern, "")
        .trim();

    try {
        const envelope = JSON.parse(cleaned);

        if (envelope && typeof envelope.reply === "string") {
            return cleanReply(envelope.reply);
        }
    } catch {
        // Normal response may be plain text.
    }

    return cleaned;
}


/* =========================================================
   CHECK IF MODEL JUST COPIED USER
========================================================= */

function sameMeaningInput(reply, input) {
    const normalize = value =>
        cleanReply(value)
            .toLowerCase()
            .replace(/[?!.,]/g, "")
            .replace(/\s+/g, " ")
            .trim();

    return !normalize(reply) || normalize(reply) === normalize(input);
}


/* =========================================================
   SCRIPT DETECTION
========================================================= */

function detectScript(text) {
    const scriptRanges = [
        [0x0900, 0x097f, "Devanagari"],
        [0x0980, 0x09ff, "Bengali"],
        [0x0a00, 0x0a7f, "Gurmukhi"],
        [0x0a80, 0x0aff, "Gujarati"],
        [0x0b00, 0x0b7f, "Odia"],
        [0x0b80, 0x0bff, "Tamil"],
        [0x0c00, 0x0c7f, "Telugu"],
        [0x0c80, 0x0cff, "Kannada"],
        [0x0d00, 0x0d7f, "Malayalam"],
        [0x0d80, 0x0dff, "Sinhala"],
        [0x0600, 0x06ff, "Arabic"],
        [0x0700, 0x074f, "Syriac"],
        [0x0780, 0x07bf, "Thaana"],
        [0x0400, 0x04ff, "Cyrillic"],
        [0x0370, 0x03ff, "Greek"],
        [0x0590, 0x05ff, "Hebrew"],
        [0x0530, 0x058f, "Armenian"],
        [0x10a0, 0x10ff, "Georgian"],
        [0x0e00, 0x0e7f, "Thai"],
        [0x0e80, 0x0eff, "Lao"],
        [0x1000, 0x109f, "Myanmar"],
        [0x1100, 0x11ff, "Hangul"],
        [0x3040, 0x30ff, "Japanese"],
        [0x3400, 0x9fff, "CJK"],
        [0xac00, 0xd7af, "Hangul"]
    ];

    for (const character of text) {
        const code = character.codePointAt(0);

        const match = scriptRanges.find(
            ([start, end]) => code >= start && code <= end
        );

        if (match) return match[2];
    }

    return "Latin/Roman";
}


/* =========================================================
   ROMANIZED LANGUAGE DETECTION
========================================================= */

function romanLanguageHint(text) {
    const hints = [
        [
            "Telugu",
            "te",
            /\b(ela\s+vunnav|ela\s+unnav|tinnava|nuvvu|naaku|naku|enta|ipudu|ippudu|ante|matladut|unnav|vunnav|chesav|chesavu|emi|enti|peru|ekkada|enduku|ledu|avunu)\b/i
        ],
        [
            "Hindi",
            "hi",
            /\b(tum|aap|kya|kaise|mujhe|mera|meri|mere|hai|hain|nahi|nahin|hoon|hu|kahan|kyun|kyon|kab|kaun|acha|achha|bahut|mujhko|aapka|aapki)\b/i
        ],
        [
            "Tamil",
            "ta",
            /\b(nee|enna|eppadi|epdi|irukka|irukku|panra|pannu|naan|neenga|unga|ungal|sollu|sollunga|enge|yen|epdi|saptiya)\b/i
        ],
        [
            "Kannada",
            "kn",
            /\b(neenu|hegiddiya|hege|nanage|maadi|madthiya|enu|yenu|illa|howdu|ninna|nimma|elli|yaake|oota|tindya)\b/i
        ],
        [
            "Malayalam",
            "ml",
            /\b(nee|engane|sukham|enikku|enikk|aano|entha|evide|ninte|ningal|illa|athe|kazhicho|sugamano)\b/i
        ],
        [
            "Marathi",
            "mr",
            /\b(tu|tumhi|kasa|kashi|ahe|mala|majha|majhi|kay|kuthe|kaay|nahi|aahes|ahes|tula|tumcha)\b/i
        ],
        [
            "Bengali",
            "bn",
            /\b(tumi|kemon|acho|amar|bangla|ki|kothay|keno|tomar|tomake|bhalo|nei|ache)\b/i
        ],
        [
            "Punjabi",
            "pa",
            /\b(tusi|tuhada|tuhanu|ki|kive|kithon|kithe|mera|meri|hanji|nahi|theek|changa)\b/i
        ],
        [
            "Gujarati",
            "gu",
            /\b(tame|tamaro|tamari|kem|cho|chhe|mane|maru|shu|kya|kyare|nathi|saru|majama)\b/i
        ]
    ];

    const match = hints.find(([, , pattern]) => pattern.test(text));

    return match
        ? {
              language: match[0],
              code: match[1],
              confidence: 0.9
          }
        : null;
}


/* =========================================================
   FRANC LANGUAGE DETECTION
========================================================= */

async function detectLanguageWithFranc(text) {
    try {
        // franc is an ES module, so use dynamic import.
        const francModule = await import("franc");

        const code = francModule.franc(text);

        if (!code || code === "und") {
            return null;
        }

        const languages = {
            eng: {
                language: "English",
                code: "en"
            },
            hin: {
                language: "Hindi",
                code: "hi"
            },
            tel: {
                language: "Telugu",
                code: "te"
            },
            tam: {
                language: "Tamil",
                code: "ta"
            },
            kan: {
                language: "Kannada",
                code: "kn"
            },
            mal: {
                language: "Malayalam",
                code: "ml"
            },
            mar: {
                language: "Marathi",
                code: "mr"
            },
            ben: {
                language: "Bengali",
                code: "bn"
            },
            guj: {
                language: "Gujarati",
                code: "gu"
            },
            pan: {
                language: "Punjabi",
                code: "pa"
            },
            urd: {
                language: "Urdu",
                code: "ur"
            },
            spa: {
                language: "Spanish",
                code: "es"
            },
            fra: {
                language: "French",
                code: "fr"
            },
            deu: {
                language: "German",
                code: "de"
            },
            ita: {
                language: "Italian",
                code: "it"
            },
            por: {
                language: "Portuguese",
                code: "pt"
            },
            rus: {
                language: "Russian",
                code: "ru"
            },
            jpn: {
                language: "Japanese",
                code: "ja"
            },
            kor: {
                language: "Korean",
                code: "ko"
            },
            zho: {
                language: "Chinese",
                code: "zh"
            },
            ara: {
                language: "Arabic",
                code: "ar"
            },
            tur: {
                language: "Turkish",
                code: "tr"
            },
            vie: {
                language: "Vietnamese",
                code: "vi"
            },
            ind: {
                language: "Indonesian",
                code: "id"
            },
            nld: {
                language: "Dutch",
                code: "nl"
            },
            pol: {
                language: "Polish",
                code: "pl"
            },
            ukr: {
                language: "Ukrainian",
                code: "uk"
            }
        };

        return languages[code] || null;
    } catch (error) {
        console.error("Franc detection error:", error.message);
        return null;
    }
}


/* =========================================================
   ASSISTANT JSON PARSER
========================================================= */

function parseAssistantResult(text, profile) {
    try {
        const cleaned = text
            .trim()
            .replace(/^```json\s*|\s*```$/g, "");

        const jsonText = cleaned.startsWith("{")
            ? cleaned
            : cleaned.match(/\{[\s\S]*\}/)?.[0];

        const result = JSON.parse(jsonText || cleaned);

        if (result.reply) {
            return {
                reply: String(result.reply),
                language: String(
                    result.language || profile.language
                )
                    .replace(/[{}"']/g, "")
                    .trim(),
                script: String(result.script || profile.script),
                speechText: String(
                    result.speech_text || result.reply
                )
            };
        }
    } catch {
        // Fall back if model does not return perfect JSON.
    }

    return {
        reply: text.trim(),
        language: profile.code,
        script: profile.script,
        speechText: text.trim()
    };
}


/* =========================================================
   OLLAMA CALL
========================================================= */

async function callOllama(requestPrompt) {
    const response = await fetch(OLLAMA_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model: OLLAMA_MODEL,
            prompt: requestPrompt,
            stream: false,
            format: "json",
            keep_alive: "10m",
            options: {
                temperature: 0.2,
                num_ctx: 4096,
                num_predict: 768
            }
        })
    });

    if (!response.ok) {
        const errorText = await response.text();

        throw new Error(
            `Ollama HTTP error ${response.status}: ${errorText}`
        );
    }

    const data = await response.json();

    if (!data.response) {
        throw new Error("Ollama returned an empty response");
    }

    return data.response;
}


function promptWithoutSystemInstruction(requestPrompt) {
    return requestPrompt.startsWith(systemPrompt)
        ? requestPrompt.slice(systemPrompt.length).trimStart()
        : requestPrompt;
}


/* =========================================================
   GEMINI CALL WITH RETRY
========================================================= */

async function callGemini(requestPrompt) {
    if (!GEMINI_API_KEY) {
        throw new Error(
            "GEMINI_API_KEY is required when AI_PROVIDER=gemini"
        );
    }

    const ai = new GoogleGenAI({
        apiKey: GEMINI_API_KEY
    });

    const maxRetries = 2;

    for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
        try {
            console.log(
                `Gemini: generating response (attempt ${attempt}/${maxRetries + 1})`
            );

            const response = await ai.models.generateContent({
                model: GEMINI_TEXT_MODEL,
                contents: promptWithoutSystemInstruction(requestPrompt),
                config: {
                    systemInstruction: systemPrompt,
                    temperature: 0.2,
                    maxOutputTokens: 768,
                    responseMimeType: "application/json"
                }
            });

            if (!response.text) {
                throw new Error("Gemini returned an empty response");
            }

            return response.text;

        } catch (error) {
            console.error(
                `Gemini attempt ${attempt} failed:`,
                error.message
            );

            const errorMessage = String(
                error?.message || ""
            ).toLowerCase();

            const isTemporaryError =
                error?.status === 503 ||
                error?.code === 503 ||
                errorMessage.includes("high demand") ||
                errorMessage.includes("unavailable") ||
                errorMessage.includes("temporarily") ||
                errorMessage.includes("overloaded");

            if (!isTemporaryError || attempt > maxRetries) {
                throw error;
            }

            const delay = attempt * 3000;

            console.log(
                `Gemini temporarily unavailable. Retrying in ${delay / 1000}s...`
            );

            await new Promise(resolve =>
                setTimeout(resolve, delay)
            );
        }
    }

    throw new Error("Gemini failed after all retry attempts");
}


/* =========================================================
   AI PROVIDER WITH OLLAMA FALLBACK
========================================================= */

async function callAI(requestPrompt) {
    if (AI_PROVIDER === "gemini") {
        try {
            return await callGemini(requestPrompt);

        } catch (error) {
            console.error(
                "Gemini unavailable. Falling back to Ollama:",
                error.message
            );

            try {
                return await callOllama(requestPrompt);

            } catch (ollamaError) {
                console.error(
                    "Ollama fallback also failed:",
                    ollamaError.message
                );

                throw new Error(
                    `Gemini failed: ${error.message}\n` +
                    `Ollama fallback failed: ${ollamaError.message}`
                );
            }
        }
    }

    if (AI_PROVIDER === "ollama") {
        return callOllama(requestPrompt);
    }

    throw new Error(
        `Unsupported AI_PROVIDER: ${AI_PROVIDER}. Use gemini or ollama.`
    );
}


function getGeminiClient() {
    if (!GEMINI_API_KEY) {
        throw new Error("GEMINI_API_KEY is required for image features");
    }

    return new GoogleGenAI({
        apiKey: GEMINI_API_KEY
    });
}


async function callGeminiImageUnderstanding(
    imageBuffer,
    mimeType,
    instruction
) {
    const ai = getGeminiClient();

    const response = await ai.models.generateContent({
        model: GEMINI_TEXT_MODEL,
        contents: [
            {
                role: "user",
                parts: [
                    {
                        inlineData: {
                            mimeType,
                            data: imageBuffer.toString("base64")
                        }
                    },
                    {
                        text:
                            instruction ||
                            "Describe this image and point out the most useful details for the user."
                    }
                ]
            }
        ],
        config: {
            systemInstruction: `${systemPrompt}

IMAGE UNDERSTANDING:
Analyze the supplied image carefully. If it contains code, transcribe and explain the relevant code. Do not claim to see details that are not visible. Return a concise, useful answer in plain text with Markdown when helpful.`,
            temperature: 0.2,
            maxOutputTokens: 768
        }
    });

    if (!response.text) {
        throw new Error("Gemini returned an empty image analysis");
    }

    return response.text.trim();
}


async function generateGeminiImage(prompt, sourceImage) {
    const ai = getGeminiClient();
    const parts = [];

    if (sourceImage) {
        parts.push({
            inlineData: {
                mimeType: sourceImage.mimeType,
                data: sourceImage.buffer.toString("base64")
            }
        });
    }

    parts.push({
        text: prompt
    });

    const response = await ai.models.generateContent({
        model: GEMINI_IMAGE_MODEL,
        contents: [
            {
                role: "user",
                parts
            }
        ],
        config: {
            responseModalities: ["TEXT", "IMAGE"]
        }
    });

    const imagePart = response.candidates
        ?.flatMap(candidate => candidate.content?.parts || [])
        .find(part => part.inlineData?.data);

    if (!imagePart?.inlineData?.data) {
        throw new Error(
            "Gemini image generation returned no image data"
        );
    }

    return {
        buffer: Buffer.from(
            imagePart.inlineData.data,
            "base64"
        ),
        mimeType:
            imagePart.inlineData.mimeType ||
            "image/png"
    };
}


/* =========================================================
   GENERATE REPLY
========================================================= */

async function generateReply(messages) {
    const latestUserMessage = [...messages]
        .reverse()
        .find(message => message.role === "user");

    if (!latestUserMessage?.content) {
        throw new Error("A user message is required");
    }

    const userText = latestUserMessage.content.trim();


    /* -----------------------------------------
       Detect writing system
    ----------------------------------------- */

    const localScript = detectScript(userText);


    /* -----------------------------------------
       Detect Romanized Indian languages
       First because franc can struggle with
       short Romanized messages.
    ----------------------------------------- */

    const localHint =
        localScript === "Latin/Roman"
            ? romanLanguageHint(userText)
            : null;


    /* -----------------------------------------
       Use franc as additional detection
    ----------------------------------------- */

    let francLanguage = null;

    if (!localHint && userText.length >= 15) {
        francLanguage =
            await detectLanguageWithFranc(userText);
    }


    /* -----------------------------------------
       Select language
    ----------------------------------------- */

    let languageHint;
    let detectedCode = "en";

    if (localHint) {
        languageHint =
            `${localHint.language} (${localHint.code})`;
        detectedCode = localHint.code;
    } else if (francLanguage) {
        languageHint =
            `${francLanguage.language} (${francLanguage.code})`;
        detectedCode = francLanguage.code;
    } else {
        languageHint =
            "infer from the message and conversation";
    }


    /* -----------------------------------------
       Conversation context
    ----------------------------------------- */

    const conversation = messages
        .slice(-4)
        .map(
            message =>
                `${message.role === "assistant"
                    ? "Assistant"
                    : "User"}: ${message.content}`
        )
        .join("\n");


    /* -----------------------------------------
       Final prompt
    ----------------------------------------- */

    const prompt = `${systemPrompt}

LANGUAGE DETECTION:
Writing system: ${localScript}
Likely language: ${languageHint}

USER MESSAGE:
${userText}

IMPORTANT LANGUAGE INSTRUCTION:
Reply in the SAME language as the user's latest message.

If the user writes Romanized language:
- Keep the reply Romanized.
- Do not convert it to native script.

If the user writes native script:
- Keep the reply in that native script.

Examples:

Romanized Telugu:
User: "tinnava?"
Assistant: "Avunu, tinnanu. Nuvvu tinnava?"

Romanized Telugu:
User: "ni peru enti?"
Assistant: "Na peru AI assistant."

Romanized Hindi:
User: "tum kya kar rahe ho?"
Assistant: "Main tumse baat kar raha hoon."

Telugu script:
User: "మీరు ఎలా ఉన్నారు?"
Assistant: "నేను బాగున్నాను. మీరు ఎలా ఉన్నారు?"

Hindi script:
User: "आप कैसे हैं?"
Assistant: "मैं ठीक हूँ। आप कैसे हैं?"

Spanish:
User: "¿Cómo estás?"
Assistant: "Estoy bien. ¿Cómo estás tú?"

French:
User: "Comment ça va ?"
Assistant: "Ça va bien. Et toi ?"

Japanese:
User: "元気ですか？"
Assistant: "はい、元気です。"

STRICT:
- Understand the meaning of the message.
- Do not translate unless requested.
- Do not repeat the user's message.
- Do not switch to English unnecessarily.
- Do not mix languages unnecessarily.
- Do not output analysis.
- Do not output chain-of-thought.
- Never output <think>, </think>, /think or /no_think.
- Keep the answer natural and concise.
- Return ONLY valid JSON.

Required JSON:
{
    "reply": "your natural answer",
    "language": "${detectedCode}",
    "script": "${localScript === "Latin/Roman"
        ? "latin"
        : "native"}"
}

Conversation:
${conversation}`;


    /* -----------------------------------------
       Ask AI
    ----------------------------------------- */

    const rawResponse =
        await callAI(prompt);


    /* -----------------------------------------
       Profile fallback
    ----------------------------------------- */

    const profile = localHint
        ? {
              language: localHint.code,
              script:
                  localScript === "Latin/Roman"
                      ? "latin"
                      : "native",
              code: localHint.code
          }
        : francLanguage
        ? {
              language: francLanguage.code,
              script:
                  localScript === "Latin/Roman"
                      ? "latin"
                      : "native",
              code: francLanguage.code
          }
        : {
              language: "en",
              script:
                  localScript === "Latin/Roman"
                      ? "latin"
                      : "native",
              code: "en"
          };


    /* -----------------------------------------
       Parse response
    ----------------------------------------- */

    let result =
        parseAssistantResult(
            rawResponse,
            profile
        );

    let reply =
        cleanReply(result.reply);

    let language =
        String(
            result.language ||
            profile.code
        );

    let script =
        String(
            result.script ||
            profile.script
        );


    /* -----------------------------------------
       Retry if model copied user
    ----------------------------------------- */

    if (sameMeaningInput(reply, userText)) {
        const retryPrompt = `${systemPrompt}

The previous response was incorrect because it repeated the user's message.

User message:
"${userText}"

Detected language:
${languageHint}

Writing system:
${localScript}

Answer the meaning of the user's message.

STRICT:
- Reply in the SAME language.
- Keep the SAME writing system.
- Romanized input = Romanized reply.
- Native script input = native script reply.
- Do NOT repeat the user's message.
- Do NOT translate unless requested.
- Keep the answer concise.
- Return ONLY valid JSON.

{
    "reply": "your natural answer",
    "language": "${detectedCode}",
    "script": "${localScript === "Latin/Roman"
        ? "latin"
        : "native"}"
}`;

        const retryRaw =
            await callAI(retryPrompt);

        result =
            parseAssistantResult(
                retryRaw,
                profile
            );

        reply =
            cleanReply(result.reply);

        language =
            String(
                result.language ||
                profile.code
            );

        script =
            String(
                result.script ||
                profile.script
            );
    }


    return {
        reply,
        language,
        script,
        speechText: reply
    };
}


/* =========================================================
   VOICE WORKER
========================================================= */

function runVoiceWorker(args) {
    return new Promise((resolve, reject) => {
        const child = spawn(
            PYTHON,
            [voiceWorker, ...args],
            {
                windowsHide: true,
                env: {
                    ...process.env,
                    PYTHONIOENCODING: "utf-8",
                    PYTHONUTF8: "1"
                }
            }
        );

        let stdout = "";
        let stderr = "";

        child.stdout.setEncoding("utf8");
        child.stderr.setEncoding("utf8");

        child.stdout.on("data", chunk => {
            stdout += chunk;
        });

        child.stderr.on("data", chunk => {
            stderr += chunk;
        });

        child.on("error", reject);

        child.on("close", code =>
            code === 0
                ? resolve(stdout.trim())
                : reject(
                      new Error(
                          stderr.trim() ||
                              `Voice worker exited with ${code}`
                      )
                  )
        );
    });
}


function runProcess(command, args, label) {
    return new Promise((resolve, reject) => {
        const configuredExecutable =
            path.resolve(command);

        const executableCandidates = [
            configuredExecutable
        ];

        if (process.platform === "win32") {
            executableCandidates.push(
                path.join(
                    path.dirname(configuredExecutable),
                    "bin",
                    path.basename(configuredExecutable)
                )
            );
        }

        const findExecutable = async () => {
            for (
                const executable
                of executableCandidates
            ) {
                try {
                    const stats =
                        await fs.stat(executable);

                    if (stats.isFile()) {
                        return executable;
                    }
                } catch {
                    // Try the next known Windows package layout.
                }
            }

            throw new Error(
                `no executable file found at: ${executableCandidates.join(", ")}`
            );
        };

        findExecutable()
            .then(executable => {
                console.log(
                    `${label}: using ${executable}`
                );

                const processEnv = {
                    ...process.env,
                    PATH:
                        process.platform === "win32"
                            ? `${path.dirname(executable)};${process.env.PATH || ""}`
                            : process.env.PATH
                };

                execFile(
                    executable,
                    args,
                    {
                        windowsHide: true,
                        env: processEnv,
                        encoding: "utf8",
                        maxBuffer: 1024 * 1024
                    },
                    (
                        error,
                        stdout,
                        stderr
                    ) => {
                        if (!error) {
                            return resolve();
                        }

                        reject(
                            new Error(
                                `${label} failed: ${
                                    String(
                                        stderr ||
                                            stdout ||
                                            ""
                                    ).trim() ||
                                    error.message
                                }`
                            )
                        );
                    }
                );
            })
            .catch(error =>
                reject(
                    new Error(
                        `${label} could not start: ${error.message}`
                    )
                )
            );
    });
}


function publicInstagramAudioUrl(fileName) {
    const baseUrl =
        process.env.INSTAGRAM_PUBLIC_BASE_URL
            ?.replace(/\/$/, "");

    if (!baseUrl) {
        throw new Error(
            "Instagram audio reply requires INSTAGRAM_PUBLIC_BASE_URL so Meta can fetch the converted audio over HTTPS"
        );
    }

    return `${baseUrl}/instagram-audio/${encodeURIComponent(
        fileName
    )}`;
}


function retainInstagramAudio(
    fileName,
    filePath
) {
    instagramAudioFiles.set(
        fileName,
        filePath
    );

    setTimeout(
        async () => {
            instagramAudioFiles.delete(
                fileName
            );

            await fs.rm(
                filePath,
                {
                    force: true
                }
            );
        },
        INSTAGRAM_AUDIO_RETENTION_MS
    ).unref();
}


function splitInstagramText(
    text,
    maxCharacters = 1000
) {
    const source = String(text ?? "");

    if (
        source.length <=
        maxCharacters
    ) {
        return [source];
    }

    const chunks = [];
    let remaining =
        Array.from(source);

    while (remaining.length) {
        let characterCount = 0;
        let end = 0;

        while (
            end < remaining.length &&
            characterCount +
                remaining[end].length <=
                maxCharacters
        ) {
            characterCount +=
                remaining[end].length;
            end += 1;
        }

        if (end === 0) {
            throw new Error(
                `Unable to split Instagram text within ${maxCharacters} characters`
            );
        }

        const candidate =
            remaining
                .slice(0, end)
                .join("");

        let splitAt =
            candidate.lastIndexOf("\n") +
            1;

        if (splitAt <= 0) {
            const sentenceBoundary =
                /[.!?。！？](?:\s|$)/g;

            let match;

            while (
                (match =
                    sentenceBoundary.exec(
                        candidate
                    ))
            ) {
                splitAt =
                    match.index + 1;
            }
        }

        if (
            splitAt <= 0 ||
            splitAt <
                Math.floor(
                    candidate.length / 2
                )
        ) {
            splitAt =
                candidate.length;
        }

        while (
            splitAt > 0 &&
            candidate
                .slice(0, splitAt)
                .length >
                maxCharacters
        ) {
            splitAt -= 1;
        }

        const chunk =
            candidate.slice(
                0,
                splitAt
            );

        if (!chunk) {
            throw new Error(
                "Instagram text splitter produced an empty chunk"
            );
        }

        chunks.push(chunk);

        remaining =
            Array.from(
                source.slice(
                    chunks.join("").length
                )
            );
    }

    return chunks;
}


async function sendInstagramTextChunk(
    recipientId,
    text
) {
    const response = await fetch(
        `https://graph.instagram.com/${
            process.env.INSTAGRAM_GRAPH_VERSION ||
            "v25.0"
        }/me/messages`,
        {
            method: "POST",
            headers: {
                Authorization:
                    `Bearer ${process.env.INSTAGRAM_ACCESS_TOKEN}`,
                "Content-Type":
                    "application/json"
            },
            body: JSON.stringify({
                recipient: {
                    id: recipientId
                },
                message: {
                    text
                }
            })
        }
    );

    if (!response.ok) {
        throw new Error(
            `Instagram text API error: ${response.status} ${await response.text()}`
        );
    }
}


async function sendInstagramTextReplyLong(
    recipientId,
    text
) {
    const chunks =
        splitInstagramText(text);

    for (
        const [
            index,
            chunk
        ]
        of chunks.entries()
    ) {
        await sendInstagramTextChunk(
            recipientId,
            chunk
        );

        if (chunks.length > 1) {
            console.log(
                `Instagram text reply chunk ${index + 1}/${chunks.length} sent (${chunk.length} characters)`
            );
        }
    }
}


async function sendInstagramAudioReply(
    recipientId,
    text,
    language
) {
    const fileId =
        crypto.randomUUID();

    const mp3Path =
        path.join(
            os.tmpdir(),
            `instagram-reply-${fileId}.mp3`
        );

    const m4aPath =
        path.join(
            os.tmpdir(),
            `instagram-reply-${fileId}.m4a`
        );

    let retained = false;

    try {
        const audioUrl =
            publicInstagramAudioUrl(
                `${fileId}.m4a`
            );

        console.log(
            "Instagram voice: generating audio reply"
        );

        await runVoiceWorker([
            "speak",
            "--text",
            text,
            "--language",
            language || "en",
            "--output",
            mp3Path
        ]);

        console.log(
            "Instagram voice: converting audio to M4A/AAC"
        );

        await runProcess(
            FFMPEG,
            [
                "-y",
                "-i",
                mp3Path,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                m4aPath
            ],
            "FFmpeg audio conversion"
        );

        const { size } =
            await fs.stat(
                m4aPath
            );

        if (
            size >
            INSTAGRAM_AUDIO_MAX_BYTES
        ) {
            throw new Error(
                "Instagram audio reply exceeds the 25 MB limit"
            );
        }

        retainInstagramAudio(
            `${fileId}.m4a`,
            m4aPath
        );

        retained = true;

        const response = await fetch(
            `https://graph.instagram.com/${
                process.env.INSTAGRAM_GRAPH_VERSION ||
                "v25.0"
            }/me/messages`,
            {
                method: "POST",
                headers: {
                    Authorization:
                        `Bearer ${process.env.INSTAGRAM_ACCESS_TOKEN}`,
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    recipient: {
                        id: recipientId
                    },
                    message: {
                        attachment: {
                            type: "audio",
                            payload: {
                                url: audioUrl
                            }
                        }
                    }
                })
            }
        );

        if (!response.ok) {
            throw new Error(
                `Instagram audio API error: ${response.status} ${await response.text()}`
            );
        }
    } catch (error) {
        if (retained) {
            instagramAudioFiles.delete(
                `${fileId}.m4a`
            );
        }

        throw error;
    } finally {
        await fs.rm(
            mp3Path,
            {
                force: true
            }
        );

        if (!retained) {
            await fs.rm(
                m4aPath,
                {
                    force: true
                }
            );
        }
    }
}


function getInstagramAudioUrl(
    message
) {
    const attachments =
        message?.message?.attachments;

    if (!Array.isArray(attachments)) {
        return null;
    }

    const audioAttachment =
        attachments.find(
            attachment =>
                attachment?.type ===
                    "audio" &&
                typeof attachment
                    ?.payload?.url ===
                    "string"
        );

    return (
        audioAttachment
            ?.payload?.url || null
    );
}


function getInstagramImageUrl(
    message
) {
    const attachments =
        message?.message?.attachments;

    if (!Array.isArray(attachments)) {
        return null;
    }

    const imageAttachment =
        attachments.find(
            attachment =>
                attachment?.type ===
                    "image" &&
                typeof attachment
                    ?.payload?.url ===
                    "string"
        );

    return (
        imageAttachment
            ?.payload?.url || null
    );
}


async function downloadInstagramImage(
    url
) {
    const response = await fetch(
        url,
        {
            headers:
                process.env
                    .INSTAGRAM_ACCESS_TOKEN
                    ? {
                          Authorization:
                              `Bearer ${process.env.INSTAGRAM_ACCESS_TOKEN}`
                      }
                    : undefined
        }
    );

    if (!response.ok) {
        throw new Error(
            `Instagram image download error: ${response.status} ${await response.text()}`
        );
    }

    const contentLength =
        Number(
            response.headers.get(
                "content-length"
            ) || 0
        );

    if (
        contentLength >
        INSTAGRAM_IMAGE_MAX_BYTES
    ) {
        throw new Error(
            "Instagram image exceeds the 8 MB limit"
        );
    }

    const buffer =
        Buffer.from(
            await response.arrayBuffer()
        );

    if (!buffer.length) {
        throw new Error(
            "Instagram image download returned an empty file"
        );
    }

    if (
        buffer.length >
        INSTAGRAM_IMAGE_MAX_BYTES
    ) {
        throw new Error(
            "Instagram image exceeds the 8 MB limit"
        );
    }

    const contentType =
        response.headers.get(
            "content-type"
        ) || "image/jpeg";

    const mimeType =
        contentType
            .split(";", 1)[0]
            .toLowerCase();

    if (
        !mimeType.startsWith("image/")
    ) {
        throw new Error(
            `Instagram attachment is not an image: ${mimeType}`
        );
    }

    return {
        buffer,
        mimeType
    };
}


function imageExtension(
    mimeType
) {
    return mimeType ===
        "image/jpeg"
        ? "jpg"
        : mimeType.split(
              "/"
          )[1] || "png";
}


function publicInstagramImageUrl(
    fileName
) {
    const baseUrl =
        process.env
            .INSTAGRAM_PUBLIC_BASE_URL
            ?.replace(/\/$/, "");

    if (!baseUrl) {
        throw new Error(
            "Image replies require INSTAGRAM_PUBLIC_BASE_URL so Meta can fetch the image over HTTPS"
        );
    }

    return `${baseUrl}/instagram-image/${encodeURIComponent(
        fileName
    )}`;
}


function retainInstagramImage(
    fileName,
    filePath,
    mimeType
) {
    instagramImageFiles.set(
        fileName,
        {
            filePath,
            mimeType
        }
    );

    setTimeout(
        async () => {
            instagramImageFiles.delete(
                fileName
            );

            await fs.rm(
                filePath,
                {
                    force: true
                }
            );
        },
        INSTAGRAM_AUDIO_RETENTION_MS
    ).unref();
}


async function handleInstagramImageMessage(
    message,
    imageUrl,
    caption
) {
    const instruction =
        caption?.trim() ||
        "Describe this image and explain the most useful details you can see.";

    const downloadedImage =
        imageUrl
            ? await downloadInstagramImage(
                  imageUrl
              )
            : null;

    const conversation =
        getInstagramConversation(
            message.sender.id,
            instruction
        );

    if (
        isImageGenerationRequest(
            instruction,
            Boolean(downloadedImage)
        )
    ) {
        console.log(
            "Instagram image: generating or editing image"
        );

        const generatedImage =
            await generateGeminiImage(
                instruction,
                downloadedImage
            );

        const fileId =
            crypto.randomUUID();

        const fileName =
            `${fileId}.${imageExtension(
                generatedImage.mimeType
            )}`;

        const filePath =
            path.join(
                os.tmpdir(),
                `instagram-${fileName}`
            );

        await fs.writeFile(
            filePath,
            generatedImage.buffer
        );

        retainInstagramImage(
            fileName,
            filePath,
            generatedImage.mimeType
        );

        try {
            const publicUrl =
                publicInstagramImageUrl(
                    fileName
                );

            await sendInstagramImageReply(
                message.sender.id,
                publicUrl
            );

            saveInstagramConversation(
                message.sender.id,
                conversation,
                "[Generated image sent]"
            );

            console.log(
                "Instagram image reply sent"
            );
        } catch (error) {
            instagramImageFiles.delete(
                fileName
            );

            await fs.rm(
                filePath,
                {
                    force: true
                }
            );

            throw error;
        }

        return;
    }

    if (!downloadedImage) {
        throw new Error(
            "An image is required for image understanding"
        );
    }

    const context =
        conversation
            .slice(0, -1)
            .map(
                item =>
                    `${item.role}: ${item.content}`
            )
            .join("\n");

    const contextualInstruction =
        context
            ? `Conversation context:\n${context}\n\nCurrent image instruction:\n${instruction}`
            : instruction;

    const answer =
        await callGeminiImageUnderstanding(
            downloadedImage.buffer,
            downloadedImage.mimeType,
            contextualInstruction
        );

    saveInstagramConversation(
        message.sender.id,
        conversation,
        answer
    );

    await sendInstagramTextReplyLong(
        message.sender.id,
        answer
    );

    console.log(
        "Instagram image understanding reply sent"
    );
}


function isImageGenerationRequest(
    text,
    hasSourceImage
) {
    const normalized =
        String(text || "")
            .toLowerCase();

    if (
        hasSourceImage &&
        /\b(make|turn|remove|change|edit|convert|transform|replace|add|erase)\b/.test(
            normalized
        )
    ) {
        return true;
    }

    return /\b(generate|create|draw|make|design|produce)\b[\s\S]*\b(image|picture|photo|illustration|artwork|logo)\b/.test(
        normalized
    );
}


async function sendInstagramImageReply(
    recipientId,
    imageUrl
) {
    const response = await fetch(
        `https://graph.instagram.com/${
            process.env.INSTAGRAM_GRAPH_VERSION ||
            "v25.0"
        }/me/messages`,
        {
            method: "POST",
            headers: {
                Authorization:
                    `Bearer ${process.env.INSTAGRAM_ACCESS_TOKEN}`,
                "Content-Type":
                    "application/json"
            },
            body: JSON.stringify({
                recipient: {
                    id: recipientId
                },
                message: {
                    attachment: {
                        type: "image",
                        payload: {
                            url: imageUrl
                        }
                    }
                }
            })
        }
    );

    if (!response.ok) {
        throw new Error(
            `Instagram image API error: ${response.status} ${await response.text()}`
        );
    }
}


function getInstagramConversation(
    senderId,
    userText
) {
    const conversation =
        instagramConversations.get(
            senderId
        ) || [];

    return [
        ...conversation,
        {
            role: "user",
            content: userText
        }
    ].slice(
        -INSTAGRAM_CONTEXT_LIMIT
    );
}


function saveInstagramConversation(
    senderId,
    conversation,
    reply
) {
    instagramConversations.set(
        senderId,
        [
            ...conversation,
            {
                role: "assistant",
                content: reply
            }
        ].slice(
            -INSTAGRAM_CONTEXT_LIMIT
        )
    );
}


async function downloadInstagramAudio(
    url
) {
    const response = await fetch(
        url,
        {
            headers:
                process.env
                    .INSTAGRAM_ACCESS_TOKEN
                    ? {
                          Authorization:
                              `Bearer ${process.env.INSTAGRAM_ACCESS_TOKEN}`
                      }
                    : undefined
        }
    );

    if (!response.ok) {
        throw new Error(
            `Instagram audio download error: ${response.status} ${await response.text()}`
        );
    }

    const contentLength =
        Number(
            response.headers.get(
                "content-length"
            ) || 0
        );

    if (
        contentLength >
        25 * 1024 * 1024
    ) {
        throw new Error(
            "Instagram audio file exceeds the 25 MB limit"
        );
    }

    const audioBuffer =
        Buffer.from(
            await response.arrayBuffer()
        );

    if (!audioBuffer.length) {
        throw new Error(
            "Instagram audio download returned an empty file"
        );
    }

    const contentType =
        response.headers.get(
            "content-type"
        ) || "";

    const extension =
        contentType.includes("mp4") ||
        contentType.includes("m4a")
            ? "m4a"
            : contentType.includes(
                  "mpeg"
              ) ||
              contentType.includes("mp3")
            ? "mp3"
            : contentType.includes(
                  "ogg"
              ) ||
              contentType.includes("opus")
            ? "ogg"
            : "webm";

    return {
        audioBuffer,
        extension
    };
}


async function transcribeInstagramAudio(
    url
) {
    let inputPath;

    try {
        console.log(
            "Instagram voice: downloading audio"
        );

        const {
            audioBuffer,
            extension
        } =
            await downloadInstagramAudio(
                url
            );

        inputPath =
            path.join(
                os.tmpdir(),
                `instagram-audio-${Date.now()}-${process.pid}.${extension}`
            );

        await fs.writeFile(
            inputPath,
            audioBuffer
        );

        console.log(
            "Instagram voice: transcribing audio"
        );

        const transcription =
            JSON.parse(
                await runVoiceWorker([
                    "transcribe",
                    inputPath
                ])
            );

        if (
            !transcription.text?.trim()
        ) {
            throw new Error(
                "Voice worker returned an empty transcription"
            );
        }

        return transcription.text.trim();
    } finally {
        if (inputPath) {
            await fs.rm(
                inputPath,
                {
                    force: true
                }
            );
        }
    }
}


/* =========================================================
   HOME PAGE
========================================================= */

app.get(
    "/instagram-audio/:fileName",
    (req, res) => {
        const filePath =
            instagramAudioFiles.get(
                req.params.fileName
            );

        if (!filePath) {
            return res.sendStatus(404);
        }

        res.sendFile(
            filePath,
            {
                headers: {
                    "Content-Type":
                        "audio/mp4",
                    "Cache-Control":
                        "no-store"
                }
            },
            error => {
                if (
                    error &&
                    !res.headersSent
                ) {
                    res.sendStatus(
                        error.statusCode ||
                            500
                    );
                }
            }
        );
    }
);


app.get(
    "/instagram-image/:fileName",
    (req, res) => {
        const image =
            instagramImageFiles.get(
                req.params.fileName
            );

        if (!image) {
            return res.sendStatus(404);
        }

        res.sendFile(
            image.filePath,
            {
                headers: {
                    "Content-Type":
                        image.mimeType,
                    "Cache-Control":
                        "no-store"
                }
            },
            error => {
                if (
                    error &&
                    !res.headersSent
                ) {
                    res.sendStatus(
                        error.statusCode ||
                            500
                    );
                }
            }
        );
    }
);


app.get(
    "/",
    (req, res) =>
        res.sendFile(
            path.join(
                __dirname,
                "public/index.html"
            )
        )
);


/* =========================================================
   GENERATE API
========================================================= */

app.post(
    "/api/generate",
    async (req, res) => {
        try {
            if (!req.body.prompt) {
                return res
                    .status(400)
                    .json({
                        error:
                            "Prompt is required"
                    });
            }

            const result =
                await generateReply([
                    {
                        role: "user",
                        content:
                            req.body.prompt
                    }
                ]);

            res.json({
                response:
                    result.reply,
                ...result
            });
        } catch (error) {
            console.error(
                "Generate error:",
                error
            );

            res.status(500).json({
                error:
                    error.message
            });
        }
    }
);


/* =========================================================
   CHAT API
========================================================= */

app.post(
    "/api/chat",
    async (req, res) => {
        try {
            const messages =
                Array.isArray(
                    req.body.messages
                )
                    ? req.body.messages
                    : [];

            if (!messages.length) {
                return res
                    .status(400)
                    .json({
                        error:
                            "At least one message is required"
                    });
            }

            const result =
                await generateReply(
                    messages.slice(-4)
                );

            res.json({
                response:
                    result.reply,
                ...result
            });
        } catch (error) {
            console.error(
                "Chat error:",
                error
            );

            res.status(500).json({
                error:
                    error.message
            });
        }
    }
);


/* =========================================================
   TRANSCRIBE AUDIO
========================================================= */

app.post(
    "/api/transcribe",
    express.raw({
        type: [
            "audio/*",
            "video/webm"
        ],
        limit: "25mb"
    }),
    async (req, res) => {
        const inputPath =
            path.join(
                os.tmpdir(),
                `chat-audio-${Date.now()}.webm`
            );

        try {
            if (
                !req.body?.length
            ) {
                return res
                    .status(400)
                    .json({
                        error:
                            "Audio body is required"
                    });
            }

            await fs.writeFile(
                inputPath,
                req.body
            );

            res.json(
                JSON.parse(
                    await runVoiceWorker([
                        "transcribe",
                        inputPath
                    ])
                )
            );
        } catch (error) {
            console.error(
                "Transcription error:",
                error
            );

            res.status(500).json({
                error:
                    error.message
            });
        } finally {
            await fs.rm(
                inputPath,
                {
                    force: true
                }
            );
        }
    }
);


/* =========================================================
   TEXT TO SPEECH
========================================================= */

app.post(
    "/api/speak",
    async (req, res) => {
        const outputPath =
            path.join(
                os.tmpdir(),
                `chat-voice-${Date.now()}.mp3`
            );

        try {
            const {
                text,
                language
            } = req.body || {};

            if (!text) {
                return res
                    .status(400)
                    .json({
                        error:
                            "Text is required"
                    });
            }

            await runVoiceWorker([
                "speak",
                "--text",
                text,
                "--language",
                language || "en",
                "--output",
                outputPath
            ]);

            res.sendFile(
                outputPath,
                async () =>
                    fs.rm(
                        outputPath,
                        {
                            force: true
                        }
                    )
            );
        } catch (error) {
            console.error(
                "Speech error:",
                error
            );

            res.status(500).json({
                error:
                    error.message
            });
        }
    }
);


/* =========================================================
   INSTAGRAM WEBHOOK VERIFICATION
========================================================= */

const VERIFY_TOKEN =
    process.env.INSTAGRAM_VERIFY_TOKEN ||
    "instagram_bot_verify_2026";

app.get(
    "/webhook",
    (req, res) => {
        console.log(
            "Instagram webhook verification:",
            req.query
        );

        if (
            req.query["hub.mode"] ===
                "subscribe" &&
            req.query[
                "hub.verify_token"
            ] === VERIFY_TOKEN
        ) {
            return res
                .status(200)
                .send(
                    req.query[
                        "hub.challenge"
                    ]
                );
        }

        res.sendStatus(403);
    }
);


/* =========================================================
   INSTAGRAM WEBHOOK
========================================================= */

app.post(
    "/webhook",
    async (req, res) => {
        console.log(
            "Instagram webhook received:",
            JSON.stringify(
                req.body,
                null,
                2
            )
        );

        const message =
            req.body.entry?.[0]
                ?.messaging?.[0];

        const text =
            message?.message?.text?.trim();

        const audioUrl =
            getInstagramAudioUrl(
                message
            );

        const imageUrl =
            getInstagramImageUrl(
                message
            );

        const imageGenerationRequest =
            isImageGenerationRequest(
                text,
                false
            );

        if (
            (!text &&
                !audioUrl &&
                !imageUrl &&
                !imageGenerationRequest) ||
            message.message.is_echo
        ) {
            return res.sendStatus(200);
        }

        res.sendStatus(200);

        try {
            if (
                imageUrl ||
                imageGenerationRequest
            ) {
                await handleInstagramImageMessage(
                    message,
                    imageUrl,
                    text
                );

                return;
            }

            let userText = text;

            if (
                !userText &&
                audioUrl
            ) {
                userText =
                    await transcribeInstagramAudio(
                        audioUrl
                    );
            }

            console.log(
                `Instagram ${
                    audioUrl
                        ? "voice"
                        : "text"
                } message transcribed/received:`,
                userText
            );

            const conversation =
                getInstagramConversation(
                    message.sender.id,
                    userText
                );

            const result =
                await generateReply(
                    conversation
                );

            saveInstagramConversation(
                message.sender.id,
                conversation,
                result.reply
            );

            if (audioUrl) {
                try {
                    console.log(
                        "Instagram voice: sending audio reply"
                    );

                    await sendInstagramAudioReply(
                        message.sender.id,
                        result.speechText,
                        result.language
                    );

                    console.log(
                        `Instagram audio reply sent in ${result.language}`
                    );
                } catch (
                    audioError
                ) {
                    console.error(
                        "Instagram audio reply unavailable; sending text fallback:",
                        audioError
                    );

                    await sendInstagramTextReplyLong(
                        message.sender.id,
                        result.reply
                    );

                    console.log(
                        `Instagram text fallback sent in ${result.language}:`,
                        result.reply
                    );
                }
            } else {
                console.log(
                    "Instagram: sending text reply"
                );

                await sendInstagramTextReplyLong(
                    message.sender.id,
                    result.reply
                );

                console.log(
                    `Instagram reply sent in ${result.language}:`,
                    result.reply
                );
            }
        } catch (error) {
            console.error(
                "Ollama/Reply error:",
                error
            );
        }
    }
);


/* =========================================================
   START SERVER
========================================================= */
app.listen(PORT, "0.0.0.0", () => { console.log("Server running at http://localhost:"); });
