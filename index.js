// 1. Import necessary modules
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
// 2. Initialize Express
const app = express();
const port = 4000;
// 3. Middleware
app.use(cors());
app.use(bodyParser.json());
// 4. Initialize Groq and embeddings
let openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});
const embeddings = new OpenAIEmbeddings();
// 5. Define the route for POST requests
app.post("/", async (req, res) => {
  // 6. Handle POST requests
  console.log(`1. Received POST request`);
  // 7. Extract request data
  const {
    message,
    returnSources = false,
    embedSourcesInLLMResponse = false,
    textChunkSize = 800,
    textChunkOverlap = 200,
    numberOfSimilarityResults = 2,
    numberOfPagesToScan = 1,
  } = req.body;
  if (message.trim().length === 0) return res.status(400).send("Bad Request");
  console.log(`2. Destructured request data`);

  // 10. Define search engine function
  async function searchEngineForSources(message) {
    console.log(`3. Initializing Search Engine Process`);
    // 11. Initialize Serper API
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
    console.log(docs);
    // 13. Get documents from Serper API
    const normalizedData = normalizeData(docs.organic);
    // 14. Process and vectorize the content
    return await Promise.all(normalizedData.map(fetchAndProcess));
  }
  // 15. Normalize data
  function normalizeData(docs) {
    return docs
      .filter((doc) => doc.title && doc.link)
      .slice(0, numberOfPagesToScan)
      .map(({ title, link }) => ({ title, link }));
  }
  // 16. Fetch page content
  const fetchPageContent = async (link) => {
    console.log(`7. Fetching page content for ${link}`);
    try {
      const response = await fetch(link);
      if (!response.ok) {
        return ""; // skip if fetch fails
      }
      const text = await response.text();
      return extractMainContent(text, link);
    } catch (error) {
      console.error(`Error fetching page content for ${link}:`, error);
      return "";
    }
  };
  // 17. Extract main content from the HTML page
  function extractMainContent(html, link) {
    console.log(`8. Extracting main content from HTML for ${link}`);
    const $ = html.length ? cheerio.load(html) : null;
    $("script, style, head, nav, footer, iframe, img").remove();
    return $("body").text().replace(/\s+/g, " ").trim();
  }
  // 18. Process and vectorize the content
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
    console.log(`9. Processed ${vectorCount} sources for ${item.link}`);
    return await vectorStore.similaritySearch(
      message,
      numberOfSimilarityResults
    );
  };
  // 19. Fetch and process sources
  const sources = await searchEngineForSources(
    message,
    textChunkSize,
    textChunkOverlap
  );
  const sourcesParsed = sources
    .filter((group) => group !== null) // Filter out null groups
    .map(
      (group) =>
        group
          .filter((doc) => doc !== null) // Ensure no null documents in each group
          .map((doc) => {
            const title = doc.metadata.title;
            const link = doc.metadata.link;
            return { title, link };
          })
          .filter(
            (doc, index, self) =>
              self.findIndex((d) => d.link === doc.link) === index
          ) // Filter out duplicates
    );
  const chatCompletion = await openai.chat.completions.create({
    model: "gpt-4", // Using GPT-4 for better language understanding and responses
    messages: [
      {
        role: "system",
        content: `
            - When the user asks a question, respond concisely with relevant information in Markdown format.
            - If no relevant results are available, respond with "No relevant results found. Can you clarify or ask another related question?" and engage using the Socratic method to encourage deeper understanding.
          `,
      },
      {
        role: "user",
        content: `- Use the top results from a similarity search to respond if the user asks a question. Otherwise, respond normally (e.g., if the user says "hey" or "hi", simply reply with "hi"). The results from the search are: ${JSON.stringify(
          sources
        )}`,
      },
    ],
    stream: true, // Enable streaming for real-time responses
  });
  console.log(`11. Sent content to Groq for chat completion.`);
  let responseTotal = "";
  console.log(`12. Streaming response from Groq... \n`);
  for await (const chunk of chatCompletion) {
    if (chunk.choices[0].delta && chunk.choices[0].finish_reason !== "stop") {
      process.stdout.write(chunk.choices[0].delta.content);
      responseTotal += chunk.choices[0].delta.content;
    } else {
      let responseObj = {};
      returnSources ? (responseObj.sources = sourcesParsed) : null;
      responseObj.answer = responseTotal;

      responseObj.followUpQuestions = await generateFollowUpQuestions(
        responseTotal
      );

      responseObj.videos = await getVideos(message);
      responseObj.videos = await getImages(message);
      console.log(responseObj);
      res.status(200).json(responseObj);
    }
  }
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
// 22. Notify when the server starts listening

async function generateFollowUpQuestions(responseText) {
  const groqResponse = await openai.chat.completions.create({
    model: "gpt-4",
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
    // Try parsing the response to JSON
    return JSON.parse(groqResponse.choices[0].message.content);
  } catch (error) {
    return null;
  }
}

app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
