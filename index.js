import express from "express";
import bodyParser from "body-parser";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import OpenAI from "openai";
import * as cheerio from "cheerio";
import dotenv from "dotenv";
import cors from "cors";
dotenv.config();
const app = express();
const port = 4000;
app.use(cors());
app.use(bodyParser.json());
let openai = new OpenAI({
  baseURL: "https://api.groq.com/openai/v1",
  apiKey: process.env.GROQ_API_KEY,
});
const embeddings = new OpenAIEmbeddings();
app.post("/", async (req, res) => {
  const { message } = req.body;
  if (message.trim().length === 0) return res.status(400).send("Bad Request");

  const chatCompletion = await openai.chat.completions.create({
    model: "mixtral-8x7b-32768", // Using the GPT-4 model for better understanding
    messages: [
      {
        role: "system",
        content: `
            You are an intelligent assistant. When a user submits a query, analyze it and return only a valid JSON object with the following properties:
    
            normal_chat: (boolean) true if the query can be answered in a straightforward manner; false otherwise.
            socratic_learning: (boolean) true if the query requires a Socratic method of engagement; false otherwise.
            requires_images: (boolean) true if the response needs images; false otherwise.
            requires_video: (boolean) true if the response needs a video; false otherwise.
            requires_source_links: (boolean) true if the response requires website or source links; false otherwise.
    
            Only respond with a valid JSON object without any additional commentary or explanation.
          `,
      },
      {
        role: "user",
        content: message, // User's message here
      },
    ],
  });

  const data = JSON.parse(chatCompletion.choices[0].message.content);

  const responseObj = {};

  responseObj.followUpQuestions = await generateFollowUpQuestions(message);

  if (data.normal_chat) {
    responseObj.answer = await normal_chat_response(message);
  } else {
    responseObj.answer = await socratic_learning_message(message);
  }

  if (data.requires_images) {
    responseObj.images = await getImages(message);
  }
  if (data.requires_video) {
    responseObj.videos = await getVideos(message);
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
    model: "mixtral-8x7b-32768",
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
    model: "mixtral-8x7b-32768", // Using the GPT-4 model for better understanding
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
  textChunkSize = 800,
  textChunkOverlap = 200,
  numberOfSimilarityResults = 2,
  numberOfPagesToScan = 1
) {
  async function searchEngineForSources(message) {
    console.log(`3. Initializing Search Engine Process`);
    const serperKey = process.env.SERPER_API_KEY;
    const response = await fetch("https://google.serper.dev/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-KEY": serperKey,
      },
      body: JSON.stringify({ q: message }),
    });
    const docs = await response.json();
    const normalizedData = normalizeData(docs.organic);
    return await Promise.all(normalizedData.map(fetchAndProcess));
  }

  function normalizeData(docs) {
    return docs
      .filter((doc) => doc.title && doc.link)
      .slice(0, numberOfPagesToScan)
      .map(({ title, link }) => ({ title, link }));
  }

  const fetchPageContent = async (link) => {
    try {
      const response = await fetch(link);
      if (!response.ok) {
        return "";
      }
      const text = await response.text();
      return extractMainContent(text, link);
    } catch (error) {
      console.error(`Error fetching page content for ${link}:`, error);
      return "";
    }
  };

  function extractMainContent(html, link) {
    const $ = html.length ? cheerio.load(html) : null;
    $("script, style, head, nav, footer, iframe, img").remove();
    return $("body").text().replace(/\s+/g, " ").trim();
  }

  let vectorCount = 0;
  const fetchAndProcess = async (item) => {
    const htmlContent = await fetchPageContent(item.link);
    if (htmlContent && htmlContent.length < 250) return null;
    const splitText = await new RecursiveCharacterTextSplitter({
      chunkSize: textChunkSize,
      chunkOverlap: textChunkOverlap,
    }).splitText(htmlContent);
    const vectorStore = await MemoryVectorStore.fromTexts(
      splitText,
      { link: item.link, title: item.title },
      embeddings
    );
    vectorCount++;

    return await vectorStore.similaritySearch(
      message,
      numberOfSimilarityResults
    );
  };

  const sources = await searchEngineForSources(
    message,
    textChunkSize,
    textChunkOverlap
  );

  const chatCompletion = await openai.chat.completions.create({
    messages: [
      {
        role: "system",
        content: `
          - Here is my query "${message}", respond back with an answer that is as short possible and to the point in Markdown. If you can't find any relevant results, respond with "No relevant results found." 
          `,
      },
      {
        role: "user",
        content: ` - Here are the top results from a similarity search: ${JSON.stringify(
          sources
        )}. `,
      },
    ],

    model: "mixtral-8x7b-32768",
  });

  return chatCompletion.choices[0].message.content;
}
app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
