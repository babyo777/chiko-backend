import express from "express";
import bodyParser from "body-parser";
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import { OpenAIEmbeddings } from "@langchain/openai";
// import { MemoryVectorStore } from "langchain/vectorstores/memory";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
// import * as cheerio from "cheerio";
import dotenv from "dotenv";
import cors from "cors";
dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const app = express();
const port = process.env.PORT || 4000;
app.use(cors());
app.use(bodyParser.json());
let openai = new OpenAI({
  baseURL: "https://api.groq.com/openai/v1",
  apiKey: process.env.GROQ_API_KEY,
});

app.post("/", async (req, res) => {
  const { message, history } = req.body;
  if (message.trim().length === 0) return res.status(400).send("Bad Request");

  const chatCompletion = await openai.chat.completions.create({
    model: "mixtral-8x7b-32768",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `
        Context: You are an intelligent assistant capable of analyzing user queries and determining the best way to respond. The goal is to categorize queries based on response needs, such as whether they require a straightforward answer, a Socratic method of engagement, images, videos, or source links.

        Objective: Your task is to generate a JSON object for each query, indicating whether the query can be answered with a normal chat, requires the Socratic method, or needs images, videos, or source links.

        Style: The response should be in a clear, technical format, suitable for developers implementing this logic in their assistant.

        Tone: Maintain a formal and precise tone to ensure clarity.

        Audience: Developers or AI engineers designing intelligent response systems for assistants, interested in the logic behind determining the type of response needed.

        Response: Analyze the user's query, then return a JSON object with five properties: 
          - 'normal_chat': Whether the query can be answered directly.
          - 'socratic_learning': Whether the query requires a back-and-forth engagement.
          - 'requires_images': Whether images are needed for the response.
          - 'requires_video': Whether a video is necessary.
          - 'requires_source_links': Whether source links should be provided.
          `,
      },
      {
        role: "user",
        content: message, // User's message here
      },
    ],
  });
  console.log(chatCompletion.choices[0].message.content);
  const data = JSON.parse(chatCompletion.choices[0].message.content);

  const responseObj = {};

  if (data?.response?.normal_chat || data.normal_chat) {
    responseObj.answer = await socratic_learning_message(message, history);
  } else {
    responseObj.answer = await socratic_learning_message(message, history);
    // responseObj.videos = await getVideos(message);
    responseObj.followUpQuestions = await generateFollowUpQuestions(
      responseObj.answer
    );
  }

  if (data.requires_images) {
    responseObj.images = await getImages(responseObj.answer);
  }
  if (data.requires_video) {
    responseObj.videos = await getVideos(responseObj.answer);
  }

  res.status(200).json(responseObj);
});

async function getVideos(message) {
  const url = "https://google.serper.dev/videos";
  const data = JSON.stringify({
    q: message,
  });
  const requestOptions = {
    method: "POST",
    headers: {
      "X-API-KEY": process.env.SERPER_API_KEY,
      "Content-Type": "application/json",
    },
    body: data,
  };
  try {
    const response = await fetch(url, requestOptions);
    if (!response.ok) {
      throw new Error(
        `Network response was not ok. Status: ${response.status}`
      );
    }
    const responseData = await response.json();
    const validLinks = await Promise.all(
      responseData.videos.map(async (video) => {
        const imageUrl = video.imageUrl;
        if (typeof imageUrl === "string") {
          try {
            const imageResponse = await fetch(imageUrl, { method: "HEAD" });
            if (imageResponse.ok) {
              const contentType = imageResponse.headers.get("content-type");
              if (contentType && contentType.startsWith("image/")) {
                return { imageUrl, link: video.link };
              }
            }
          } catch (error) {
            console.error(`Error fetching image link ${imageUrl}:`, error);
          }
        }
        return null;
      })
    );
    const filteredLinks = validLinks.filter((link) => link !== null);
    return filteredLinks.slice(0, 9);
  } catch (error) {
    console.error("Error fetching videos:", error);
    return null;
  }
}

async function getImages(message) {
  const url = "https://google.serper.dev/images";
  const data = JSON.stringify({
    q: message,
  });
  const requestOptions = {
    method: "POST",
    headers: {
      "X-API-KEY": process.env.SERPER_API_KEY,
      "Content-Type": "application/json",
    },
    body: data,
  };
  try {
    const response = await fetch(url, requestOptions);
    if (!response.ok) {
      throw new Error(
        `Network response was not ok. Status: ${response.status}`
      );
    }
    const responseData = await response.json();
    const validLinks = await Promise.all(
      responseData.images.map(async (image) => {
        const link = image.imageUrl;
        if (typeof link === "string") {
          try {
            const imageResponse = await fetch(link, { method: "HEAD" });
            if (imageResponse.ok) {
              const contentType = imageResponse.headers.get("content-type");
              if (contentType && contentType.startsWith("image/")) {
                return {
                  title: image.title,
                  link: link,
                };
              }
            }
          } catch (error) {
            console.error(`Error fetching image link ${link}:`, error);
          }
        }
        return null;
      })
    );
    const filteredLinks = validLinks.filter((link) => link !== null);
    return filteredLinks.slice(0, 9);
  } catch (error) {
    console.error("Error fetching images:", error);
    return null;
  }
}

async function generateFollowUpQuestions(responseText) {
  const groqResponse = await openai.chat.completions.create({
    model: "llama-3.1-8b-instant",
    messages: [
      {
        role: "system",
        content:
          "You are a question generator. Generate 3 follow-up questions based on the provided text. Return the questions in an array format.",
      },
      {
        role: "user",
        content: `Generate 3 follow-up questions based on the following text:\n\n${responseText}\n\nReturn the questions in the following format: ["Question 1", "Question 2", "Question 3"]`,
      },
    ],
  });

  try {
    return JSON.parse(groqResponse.choices[0].message.content);
  } catch (error) {
    return null;
  }
}

async function normal_chat_response(message) {
  const chatCompletion = await openai.chat.completions.create({
    model: "llama-3.1-8b-instant", // Using the GPT-4 model for better understanding
    messages: [
      {
        role: "system",
        content: ``,
      },
      {
        role: "user",
        content: message,
      },
    ],
  });
  return chatCompletion.choices[0].message.content;
}

async function socratic_learning_message(
  message,
  history
  // textChunkSize = 800,
  // textChunkOverlap = 200,
  // numberOfSimilarityResults = 2,
  // numberOfPagesToScan = 1
) {
  // async function searchEngineForSources(message) {
  //   console.log(`3. Initializing Search Engine Process`);
  //   const serperKey = process.env.SERPER_API_KEY;
  //   const response = await fetch("https://google.serper.dev/search", {
  //     method: "POST",
  //     headers: {
  //       "Content-Type": "application/json",
  //       "X-API-KEY": serperKey,
  //     },
  //     body: JSON.stringify({ q: message }),
  //   });
  //   const docs = await response.json();
  //   const normalizedData = normalizeData(docs.organic);
  //   return await Promise.all(normalizedData.map(fetchAndProcess));
  // }

  // function normalizeData(docs) {
  //   return docs
  //     .filter((doc) => doc.title && doc.link)
  //     .slice(0, numberOfPagesToScan)
  //     .map(({ title, link }) => ({ title, link }));
  // }

  // const fetchPageContent = async (link) => {
  //   try {
  //     const response = await fetch(link);
  //     if (!response.ok) {
  //       return "";
  //     }
  //     const text = await response.text();
  //     return extractMainContent(text, link);
  //   } catch (error) {
  //     console.error(`Error fetching page content for ${link}:`, error);
  //     return "";
  //   }
  // };

  // function extractMainContent(html, link) {
  //   const $ = html.length ? cheerio.load(html) : null;
  //   $("script, style, head, nav, footer, iframe, img").remove();
  //   return $("body").text().replace(/\s+/g, " ").trim();
  // }

  // let vectorCount = 0;
  // const fetchAndProcess = async (item) => {
  //   const htmlContent = await fetchPageContent(item.link);
  //   if (htmlContent && htmlContent.length < 250) return null;
  //   const splitText = await new RecursiveCharacterTextSplitter({
  //     chunkSize: textChunkSize,
  //     chunkOverlap: textChunkOverlap,
  //   }).splitText(htmlContent);
  //   const vectorStore = await MemoryVectorStore.fromTexts(
  //     splitText,
  //     { link: item.link, title: item.title },
  //     embeddings
  //   );
  //   vectorCount++;

  //   return await vectorStore.similaritySearch(
  //     message,
  //     numberOfSimilarityResults
  //   );
  // };

  // const sources = await searchEngineForSources(
  //   message,
  //   textChunkSize,
  //   textChunkOverlap
  // );
  console.log("calling gemini", history);
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const chat = model.startChat({
    history: [
      {
        role: "user",
        parts: [
          {
            text: `
You are a Socratic chatbot designed to engage users in critical thinking through dialogue. Your goal is to help users deepen their understanding of concepts by asking probing questions instead of providing direct answers.

	1.	Context: A user has a question or is seeking help on a particular topic.
	2.	Response Style:
	•	Begin by acknowledging the user’s question and providing a brief context if necessary.
	•	Follow up with an one open-ended question that encourages the user to reflect on their understanding and assumptions.
	•	Based on the user’s responses, ask further probing questions to guide them toward deeper insights.
	3.	Example Interaction:
	•	User: “Why is it important to understand time complexity in algorithms?”
	•	Chatbot: “That’s an important question! What do you think time complexity helps you understand about how algorithms perform?”
	4.	Continue the dialogue by encouraging the user to explore their thoughts, consider implications, and make connections without directly answering their questions.

Additional Notes:

	•	Maintain a curious and engaging tone throughout the conversation.
	•	Keep your questions focused on guiding the user to articulate their thoughts and explore the topic more deeply.
`,
          },
        ],
      },
      ...(history || null),
    ],
  });

  const result = await chat.sendMessage(message);
  const res = await result.response;
  const output = await res.text();

  return output;
}
app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
