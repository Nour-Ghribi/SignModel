import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const text = typeof body?.text === "string" ? body.text.trim() : "";

    if (!text) {
      return NextResponse.json({ reply: "Texte manquant." }, { status: 400 });
    }

    const mockReply = `J'ai compris: ${text}. Voici une réponse adaptée.`;
    const apiKey = process.env.OPENAI_API_KEY;

    if (!apiKey) {
      return NextResponse.json({ reply: mockReply });
    }

    try {
      const resp = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: "gpt-3.5-turbo",
          temperature: 0.4,
          messages: [
            {
              role: "system",
              content: "Tu es HandiBot, assistant clair et bienveillant.",
            },
            {
              role: "user",
              content: `L'utilisateur a signé: ${text}. Réponds en une phrase utile.`,
            },
          ],
        }),
      });

      if (!resp.ok) {
        return NextResponse.json({ reply: mockReply });
      }

      const data = await resp.json();
      const reply = data?.choices?.[0]?.message?.content?.trim() || mockReply;
      return NextResponse.json({ reply });
    } catch {
      return NextResponse.json({ reply: mockReply });
    }
  } catch {
    return NextResponse.json({ reply: "Requête invalide." }, { status: 400 });
  }
}
