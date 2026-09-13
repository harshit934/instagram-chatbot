require("dotenv").config();
const express = require("express");

const app = express();

app.use(express.json());

const PORT = process.env.PORT || 3000;

// Test route
app.get("/", (req, res) => {
    res.send("Instagram chatbot server is running!");
});

// Instagram webhook verification
app.get("/webhook", (req, res) => {
    const VERIFY_TOKEN = "instagram_bot_verify_2026";

    const mode = req.query["hub.mode"];
    const token = req.query["hub.verify_token"];
    const challenge = req.query["hub.challenge"];

    if (mode === "subscribe" && token === VERIFY_TOKEN) {
        console.log("Webhook verified!");
        res.status(200).send(challenge);
    } else {
        res.sendStatus(403);
    }
});

// Receive Instagram messages
app.post("/webhook", async (req, res) => {
    console.log("Instagram webhook received:");
    console.log(JSON.stringify(req.body, null, 2));

    try {
        const message = req.body.entry?.[0]?.messaging?.[0];

        // Ignore read events and other events
        if (!message?.message?.text || message.message.is_echo) {
            return res.sendStatus(200);
        }

        const senderId = message.sender.id;
        const receivedText = message.message.text;

        // Tell Instagram we received the webhook
        res.sendStatus(200);

        console.log("User said:", receivedText);
        console.log("Sender ID:", senderId);

        // Send message to Ollama through ngrok
        const ollamaResponse = await fetch(
            "https://cough-until-record.ngrok-free.dev/api/generate",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    model: "qwen2.5:0.5b",
                    prompt: `You are a friendly Instagram chatbot.
Give short, natural and helpful replies.
Keep replies suitable for Instagram DMs.

User message: ${receivedText}`,
                    stream: false
                })
            }
        );

        const ollamaData = await ollamaResponse.json();

        console.log("Ollama response:");
        console.log(JSON.stringify(ollamaData, null, 2));

        const aiReply = ollamaData.response;

        console.log("AI reply:", aiReply);

        // Send AI reply back to Instagram
        const instagramResponse = await fetch(
            "https://graph.instagram.com/v25.0/me/messages",
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${process.env.INSTAGRAM_ACCESS_TOKEN}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    recipient: {
                        id: senderId
                    },
                    message: {
                        text: aiReply
                    }
                })
            }
        );

        const instagramData = await instagramResponse.json();

        console.log("Instagram API response:");
        console.log(JSON.stringify(instagramData, null, 2));

    } catch (error) {
        console.error("AI/Reply error:", error);
    }
});

// Start server
app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running at http://localhost:${PORT}`);
});