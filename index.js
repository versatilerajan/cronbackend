require("dotenv").config();
const path = require("path");
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const helmet = require("helmet");
const compression = require("compression");
const rateLimit = require("express-rate-limit");
const admin = require("firebase-admin");
const Razorpay = require("razorpay");
const crypto = require("crypto");
const { QdrantClient } = require("@qdrant/js-client-rest");
const { EmbeddingModel, FlagEmbedding } = require("fastembed");
const app = express();
app.use(
  express.json({
    limit: "10kb",
    verify: (req, res, buf) => {
      if (req.originalUrl === "/subscription/webhook") {
        req.rawBody = buf.toString("utf8");
      }
    },
  })
);
app.use(cors({
  origin: true,
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true,
  optionsSuccessStatus: 204
}));
app.options(/.*/, (req, res) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.sendStatus(204);
});
app.use(helmet());
app.use(compression());
app.use(rateLimit({ windowMs: 15 * 60 * 1000, max: 500 }));

const RAZORPAY_KEY_ID = process.env.RAZORPAY_KEY_ID;
const RAZORPAY_KEY_SECRET = process.env.RAZORPAY_KEY_SECRET;
const RAZORPAY_WEBHOOK_SECRET = process.env.RAZORPAY_WEBHOOK_SECRET;

const PLAN_CONFIG = {
  monthly: { planId: process.env.RAZORPAY_PLAN_ID_MONTHLY, totalCount: 120 },
  yearly:  { planId: process.env.RAZORPAY_PLAN_ID_YEARLY,  totalCount: 10  },
};

function resolvePlan(planType) {
  const key = String(planType || "monthly").toLowerCase();
  const plan = PLAN_CONFIG[key];
  if (!plan || !plan.planId) return null;
  return { planType: key, planId: plan.planId, totalCount: plan.totalCount };
}

const razorpay = new Razorpay({
  key_id: RAZORPAY_KEY_ID,
  key_secret: RAZORPAY_KEY_SECRET,
});

const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

const QDRANT_KNOWLEDGE_BASE_COLLECTION = process.env.QDRANT_KNOWLEDGE_BASE_COLLECTION || "knowledge_base";
const QDRANT_QUESTION_BANK_COLLECTION = process.env.QDRANT_QUESTION_BANK_COLLECTION || "question_bank";

const PSMODEL_ENDPOINT = process.env.PSMODEL_ENDPOINT;
const PSMODEL_API_KEY = process.env.PSMODEL_API_KEY;
const PSMODEL_MODEL = process.env.PSMODEL_MODEL;

const PSMODEL_TIMEOUT_MS = parseInt(process.env.PSMODEL_TIMEOUT_MS || "60000", 10);
const PSMODEL_TEMPERATURE = parseFloat(process.env.PSMODEL_TEMPERATURE || "0.3");
const QA_MAX_TOKENS = parseInt(process.env.QA_MAX_TOKENS || "1200", 10);

const EMBEDDING_MODEL_NAME = process.env.EMBEDDING_MODEL_NAME || "BAAI/bge-base-en-v1.5";
const EMBEDDING_CACHE_DIR = process.env.EMBEDDING_CACHE_DIR || path.join(process.cwd(), ".fastembed_cache");

const QA_KNOWLEDGE_BASE_TOP_K = parseInt(process.env.QA_KNOWLEDGE_BASE_TOP_K || "8", 10);
const QA_QUESTION_BANK_TOP_K = parseInt(process.env.QA_QUESTION_BANK_TOP_K || "5", 10);

const FREE_DAILY_QUERY_LIMIT = parseInt(process.env.FREE_DAILY_QUERY_LIMIT || "1", 10);
const ENABLE_QUERY_DAILY_LIMIT = process.env.ENABLE_QUERY_DAILY_LIMIT !== "false";

const qdrant = new QdrantClient({ url: QDRANT_URL, apiKey: QDRANT_API_KEY });

let embedderPromise = null;
function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = FlagEmbedding.init({
      model: EmbeddingModel.BGEBaseENV15,
      cacheDir: EMBEDDING_CACHE_DIR,
      maxLength: 512,
    });
  }
  return embedderPromise;
}

async function embedOne(text) {
  const embedder = await getEmbedder();
  const out = [];
  for await (const batch of embedder.embed([text], 32)) {
    for (const vec of batch) out.push(Array.from(vec));
  }
  return out[0];
}

async function searchKnowledgeBase(vector, topK) {
  const result = await qdrant.query(QDRANT_KNOWLEDGE_BASE_COLLECTION, {
    query: vector,
    limit: topK,
    with_payload: true,
  });
  return result.points || [];
}

async function searchQuestionBank(vector, topK) {
  const result = await qdrant.query(QDRANT_QUESTION_BANK_COLLECTION, {
    query: vector,
    limit: topK,
    with_payload: true,
  });
  return result.points || [];
}

function formatKnowledge(points) {
  if (!points.length) return "None found";
  return points
    .map((p, i) => {
      const pl = p.payload || {};
      const label = pl.chapter || pl.topic || pl.subject || pl.source || "Reference";
      return `[${i + 1}] (${label}) ${pl.text || ""}`;
    })
    .join("\n\n");
}

function buildAnswerPrompt({ question, kbText }) {
  const system = `You are Ask Cron, a knowledgeable tutor for Civil Services (UPSC/State PSC) aspirants. Answer the student's question clearly, accurately, and directly using the supplied reference material as your factual source. If the material does not contain enough information to answer confidently, say so plainly instead of guessing or inventing facts. Keep the answer well-structured and exam-relevant. Never mention "context", "knowledge base", "retrieval", or any internal system detail; just answer naturally as a tutor would. Respond in plain text, no markdown code fences.`;
  const user = `Student question: "${question}"

Reference material:
${kbText}

Write a clear, accurate, well-organized answer for a Civil Services exam aspirant. If the material is insufficient to fully answer, say what is missing rather than fabricating details.`;
  return { system, user };
}

async function streamPSModel(system, user, onToken) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PSMODEL_TIMEOUT_MS);
  let full = "";
  try {
    const response = await fetch(PSMODEL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${PSMODEL_API_KEY}`,
      },
      body: JSON.stringify({
        model: PSMODEL_MODEL,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        temperature: PSMODEL_TEMPERATURE,
        max_tokens: QA_MAX_TOKENS,
        stream: true,
      }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      const errText = await response.text().catch(() => "");
      throw new Error(`PSMODEL request failed with status ${response.status}: ${errText}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sepIndex;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);
        const lines = rawEvent.split("\n");
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const json = JSON.parse(payload);
            const delta = json?.choices?.[0]?.delta?.content;
            if (delta) {
              full += delta;
              if (onToken) onToken(delta);
            }
          } catch (e) {}
        }
      }
    }
    return full;
  } finally {
    clearTimeout(timer);
  }
}

let cached = global.mongoose || { conn: null, promise: null };
global.mongoose = cached;

async function connectDB() {
  if (cached.conn) return cached.conn;
  if (!cached.promise) {
    if (!process.env.MONGO_URI) throw new Error("MONGO_URI is missing");
    cached.promise = mongoose.connect(process.env.MONGO_URI, {
      serverSelectionTimeoutMS: 15000,
      connectTimeoutMS: 15000,
    })
      .then(m => {
        console.log("MongoDB connected successfully");
        return m;
      })
      .catch(err => {
        console.error("MongoDB connection FAILED:", err.message);
        cached.promise = null;
        throw err;
      });
  }
  cached.conn = await cached.promise;

  await mongoose.model("Result").collection.createIndex(
    { userId: 1, testId: 1, phase: 1, isLate: 1, score: -1, submittedAt: -1 },
    { background: true }
  );

  return cached.conn;
}

let aiChatHisCached = global.aiChatHisConn || { conn: null, promise: null };
global.aiChatHisConn = aiChatHisCached;

async function connectAIChatHistoryDB() {
  if (aiChatHisCached.conn) return aiChatHisCached.conn;
  if (!aiChatHisCached.promise) {
    if (!process.env.AICHATHIS_URI) throw new Error("AICHATHIS_URI is missing");
    aiChatHisCached.promise = mongoose.createConnection(process.env.AICHATHIS_URI, {
      serverSelectionTimeoutMS: 15000,
      connectTimeoutMS: 15000,
    }).asPromise()
      .then(conn => {
        console.log("AI chat history DB connected successfully");
        return conn;
      })
      .catch(err => {
        console.error("AI chat history DB connection FAILED:", err.message);
        aiChatHisCached.promise = null;
        throw err;
      });
  }
  aiChatHisCached.conn = await aiChatHisCached.promise;
  return aiChatHisCached.conn;
}

const aiChatHistorySchema = new mongoose.Schema({
  userId: { type: String, index: true },
  question: String,
  answer: String,
  relatedPyqs: [mongoose.Schema.Types.Mixed],
}, { timestamps: true });

function getAIChatHistoryModel(conn) {
  return conn.models.AIChatHistory || conn.model("AIChatHistory", aiChatHistorySchema, "ai_chat_history");
}

let firebaseInitialized = false;
if (!admin.apps.length) {
  try {
    const raw = process.env.FIREBASE_SERVICE_ACCOUNT;
    if (!raw || raw.trim() === "") throw new Error("FIREBASE_SERVICE_ACCOUNT missing");
    const serviceAccount = JSON.parse(raw);
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    firebaseInitialized = true;
    console.log("Firebase Admin Initialized successfully (Auth only)");
  } catch (err) {
    console.error("Firebase Admin Initialization FAILED:", err.message);
  }
}

const testSchema = new mongoose.Schema({
  title: String,
  date: String,
  startTime: Date,
  endTime: Date,
  totalQuestions: Number,
  testType: { type: String, enum: ["paid", "free"], required: true },
  isSundayFullTest: { type: Boolean, default: false },
}, { timestamps: true });

const questionSchema = new mongoose.Schema({
  testId: mongoose.Schema.Types.ObjectId,
  questionNumber: Number,
  questionStatement: String,
  options: { option1: String, option2: String, option3: String, option4: String },
  correctOption: String,
  phase: { type: String, enum: ["GS", "CSAT"], default: "GS" }
});

const resultSchema = new mongoose.Schema({
  userId: String,
  testId: mongoose.Schema.Types.ObjectId,
  phase: { type: String, enum: ["GS", "CSAT"], required: true },
  score: Number,
  correct: Number,
  incorrect: Number,
  unattempted: Number,
  attempted: Number,
  totalQuestions: Number,
  submittedAt: { type: Date, default: Date.now },
  startedAt: Date,
  isLate: { type: Boolean, default: false },
  answers: [{ questionId: String, selectedOption: String }],
  timeTakenSeconds: { type: Number, default: 0 }
}, { timestamps: true });

const freeResultSchema = new mongoose.Schema({
  testId: mongoose.Schema.Types.ObjectId,
  score: Number,
  totalQuestions: Number,
  submittedAt: { type: Date, default: Date.now },
}, { timestamps: true });

const userSchema = new mongoose.Schema({
  uid: { type: String, required: true, unique: true },
  displayName: String,
  email: String,
  premiumPeriods: [{
    from: Date,
    to: Date,
  }],
  premiumExpiresAt: Date,
  isPremium: { type: Boolean, default: false },
  subscriptionId: { type: String, default: null },
  subscriptionStatus: { type: String, default: null },
  planType: { type: String, enum: ["monthly", "yearly", null], default: null },
  lastPaymentId: { type: String, default: null },
  qaUsage: {
    date: { type: String, default: null },
    count: { type: Number, default: 0 },
  },
}, { timestamps: true });

const Test       = mongoose.models.Test       || mongoose.model("Test",       testSchema);
const Question   = mongoose.models.Question   || mongoose.model("Question",   questionSchema);
const Result     = mongoose.models.Result     || mongoose.model("Result",     resultSchema);
const FreeResult = mongoose.models.FreeResult || mongoose.model("FreeResult", freeResultSchema);
const User       = mongoose.models.User       || mongoose.model("User",       userSchema);

const userAuth = async (req, res, next) => {
  if (!firebaseInitialized) return res.status(503).json({ message: "Auth service unavailable" });
  const authHeader = req.headers.authorization;
  if (!authHeader) return res.status(401).json({ message: "No token provided" });
  const token = authHeader.startsWith("Bearer ") ? authHeader.split(" ")[1] : authHeader;
  try {
    const decoded = await admin.auth().verifyIdToken(token);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ message: "Invalid Firebase token" });
  }
};

function calculateNetScore(correct, incorrect) {
  return (correct * 2) - (incorrect * (2 / 3));
}

const IST_OFFSET_MS = 5.5 * 60 * 60 * 1000;

function nowIST() {
  return new Date(Date.now() + IST_OFFSET_MS);
}

function toISTISOString(utcDate) {
  const shifted = new Date(utcDate.getTime() + IST_OFFSET_MS);
  return shifted.toISOString().replace("Z", "+05:30");
}

function isRankRevealTime() {
  const ist = nowIST();
  const hours   = ist.getUTCHours();
  const minutes = ist.getUTCMinutes();
  return (hours > 17) || (hours === 17 && minutes >= 0);
}

function rankRevealTimeIST() {
  return "5:00 PM IST";
}

function toISTDateKey(utcMs) {
  const ist = new Date(utcMs + IST_OFFSET_MS);
  return [
    ist.getUTCFullYear(),
    String(ist.getUTCMonth() + 1).padStart(2, '0'),
    String(ist.getUTCDate()).padStart(2, '0'),
  ].join('-');
}

function computeStreak(results, premiumPeriods, nowMs) {
  const NOW = nowMs || Date.now();

  const testedSet = new Set();
  results.forEach(r => {
    testedSet.add(toISTDateKey(r.submittedAt.getTime()));
  });

  if (testedSet.size === 0) return { currentStreak: 0, longestStreak: 0 };

  const GRACE_MAX_DAYS = 2;
  const graceDaySet = new Set();

  if (premiumPeriods && premiumPeriods.length >= 2) {
    const sorted = [...premiumPeriods].sort((a, b) => new Date(a.from) - new Date(b.from));

    for (let i = 0; i < sorted.length - 1; i++) {
      const endOfPeriod   = new Date(sorted[i].to);
      const startOfRenew  = new Date(sorted[i + 1].from);

      const endKey    = toISTDateKey(endOfPeriod.getTime());
      const renewKey  = toISTDateKey(startOfRenew.getTime());

      const gapDays = Math.round(
        (new Date(renewKey).getTime() - new Date(endKey).getTime()) / 86400000
      ) - 1;

      if (gapDays >= 1 && gapDays <= GRACE_MAX_DAYS) {
        for (let g = 1; g <= gapDays; g++) {
          const gMs  = new Date(endKey).getTime() + g * 86400000;
          const gKey = toISTDateKey(gMs + IST_OFFSET_MS);
          graceDaySet.add(gKey);
        }
      }
    }
  }

  const bridgedSet = new Set(testedSet);
  for (const gDay of graceDaySet) {
    const gMs    = new Date(gDay).getTime();
    const prevKey = toISTDateKey(gMs - 86400000 + IST_OFFSET_MS);
    const nextKey = toISTDateKey(gMs + 86400000 + IST_OFFSET_MS);
    if (testedSet.has(prevKey) && testedSet.has(nextKey)) {
      bridgedSet.add(gDay);
    }
  }

  const sorted = [...bridgedSet].sort();
  let longestStreak = 1;
  let runLen = 1;
  for (let i = 1; i < sorted.length; i++) {
    const diffDays = Math.round(
      (new Date(sorted[i]) - new Date(sorted[i - 1])) / 86400000
    );
    if (diffDays === 1) {
      runLen++;
      if (runLen > longestStreak) longestStreak = runLen;
    } else {
      runLen = 1;
    }
  }

  const todayKey = toISTDateKey(NOW);
  const ydayKey  = toISTDateKey(NOW - 86400000);

  let currentStreak = 0;
  if (bridgedSet.has(todayKey) || bridgedSet.has(ydayKey)) {
    let cursorMs = new Date(bridgedSet.has(todayKey) ? todayKey : ydayKey).getTime();
    while (true) {
      const k = toISTDateKey(cursorMs + IST_OFFSET_MS);
      if (bridgedSet.has(k)) {
        currentStreak++;
        cursorMs -= 86400000;
      } else {
        break;
      }
    }
  }

  return { currentStreak, longestStreak };
}

async function computeRank(testId, phase, score, submittedAt) {
  const better = await Result.countDocuments({
    testId,
    phase,
    isLate: false,
    $or: [
      { score: { $gt: score } },
      { score: score, submittedAt: { $lt: submittedAt } }
    ]
  });
  const total = await Result.countDocuments({ testId, phase, isLate: false });
  return { rank: better + 1, totalParticipants: total };
}

async function buildTestResponse(test, userId, istNow) {
  const startIST = new Date(test.startTime.getTime() + IST_OFFSET_MS);
  const endIST   = new Date(test.endTime.getTime()   + IST_OFFSET_MS);

  const testDateParts = (test.date || "").split("-");
  let deadlineIST = endIST;
  if (testDateParts.length === 3) {
    const [y, m, d] = testDateParts.map(Number);
    const endOfDayUTC = new Date(Date.UTC(y, m - 1, d, 18, 29, 59));
    const endOfDayIST = new Date(endOfDayUTC.getTime() + IST_OFFSET_MS);
    deadlineIST = endOfDayIST > endIST ? endOfDayIST : endIST;
  }

  const existingPhases = ["GS"];
  if (test.isSundayFullTest) existingPhases.push("CSAT");

  const userResults = await Result.find({
    userId,
    testId: test._id,
    phase: { $in: existingPhases },
  }).lean();

  const submittedPhases = userResults.map(r => r.phase);
  const hasSubmitted    = submittedPhases.length > 0;

  if (istNow < startIST) {
    return {
      status: "not_started",
      testId: test._id.toString(),
      title: test.title,
      startTimeIST: toISTISOString(test.startTime),
      endTimeIST: toISTISOString(test.endTime),
      deadlineIST: toISTISOString(new Date(deadlineIST.getTime() - IST_OFFSET_MS)),
      totalQuestions: test.totalQuestions,
      isSundayFullTest: !!test.isSundayFullTest,
      hasSubmitted: false,
      submittedPhases: [],
      message: "Test has not started yet"
    };
  }

  if (istNow > deadlineIST) {
    return {
      status: "archived",
      testId: test._id.toString(),
      title: test.title,
      startTimeIST: toISTISOString(test.startTime),
      endTimeIST: toISTISOString(test.endTime),
      deadlineIST: toISTISOString(new Date(deadlineIST.getTime() - IST_OFFSET_MS)),
      totalQuestions: test.totalQuestions,
      isSundayFullTest: !!test.isSundayFullTest,
      hasSubmitted,
      submittedPhases,
      canReview: hasSubmitted,
      message: "Test has ended and is now archived."
    };
  }

  const questions = await Question.find({ testId: test._id })
    .select("-correctOption")
    .sort({ questionNumber: 1 })
    .lean();

  const response = {
    status: "active",
    testId: test._id.toString(),
    title: test.title,
    totalQuestions: test.totalQuestions,
    startTimeIST: toISTISOString(test.startTime),
    endTimeIST: toISTISOString(test.endTime),
    deadlineIST: toISTISOString(new Date(deadlineIST.getTime() - IST_OFFSET_MS)),
    isSundayFullTest: !!test.isSundayFullTest,
    hasSubmitted,
    submittedPhases,
  };

  if (test.isSundayFullTest) {
    const gs   = questions.filter(q => q.phase === "GS");
    const csat = questions.filter(q => q.phase === "CSAT");
    response.phases = {
      GS:   { count: gs.length,   questions: gs   },
      CSAT: { count: csat.length, questions: csat },
    };
  } else {
    response.questions = questions;
  }

  return response;
}

function verifyWebhookSignature(rawBody, signature) {
  const expected = crypto
    .createHmac("sha256", RAZORPAY_WEBHOOK_SECRET)
    .update(rawBody)
    .digest("hex");

  if (expected.length !== signature.length) return false;

  return crypto.timingSafeEqual(
    Buffer.from(expected),
    Buffer.from(signature)
  );
}

async function syncSubscription(uid) {
  const user = await User.findOne({ uid });
  if (!user) throw new Error("User not found");
  if (!user.subscriptionId) throw new Error("No subscription found");

  const subscription = await razorpay.subscriptions.fetch(user.subscriptionId);
  user.subscriptionStatus = subscription.status;
  await user.save();

  return {
    premium: user.isPremium,
    status: user.subscriptionStatus,
    expiry: user.premiumExpiresAt,
    subscriptionId: user.subscriptionId,
  };
}

app.get("/", async (req, res) => {
  try {
    await connectDB();
    res.json({
      status: "User Backend Running",
      firebaseReady: firebaseInitialized,
      mongoReady: mongoose.connection.readyState === 1 ? "connected" : "not connected",
      currentServerDateUTC: new Date().toISOString().split("T")[0],
      currentISTTime: toISTISOString(new Date()),
    });
  } catch (err) {
    res.status(500).json({ status: "Error", error: err.message });
  }
});

app.get("/subscription/config", userAuth, async (req, res) => {
  try {
    res.json({
      success: true,
      key: RAZORPAY_KEY_ID,
      plans: {
        monthly: { planId: PLAN_CONFIG.monthly.planId },
        yearly:  { planId: PLAN_CONFIG.yearly.planId },
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.post("/subscription/create", userAuth, async (req, res) => {
  try {
    await connectDB();

    const resolved = resolvePlan(req.body?.planType);
    if (!resolved) {
      return res.status(400).json({
        success: false,
        message: "planType must be 'monthly' or 'yearly', and the matching plan id must be configured",
      });
    }

    const user = await User.findOneAndUpdate(
      { uid: req.user.uid },
      {
        $setOnInsert: { uid: req.user.uid },
        $set: { displayName: req.user.name || "", email: req.user.email || "" },
      },
      { upsert: true, new: true }
    );

    if (user.subscriptionId) {
      try {
        const existing = await razorpay.subscriptions.fetch(user.subscriptionId);
        user.subscriptionStatus = existing.status;
        await user.save();

        const reusableStatuses = ["created", "authenticated", "active", "pending"];
        const samePlan = existing.plan_id === resolved.planId;
        if (samePlan && reusableStatuses.includes(existing.status)) {
          return res.json({
            success: true,
            alreadyExists: true,
            subscriptionId: existing.id,
            status: existing.status,
            planType: resolved.planType,
            key: RAZORPAY_KEY_ID,
          });
        }

        if (!samePlan && reusableStatuses.includes(existing.status)) {
          return res.status(409).json({
            success: false,
            message: "An active subscription on a different plan already exists. Cancel it before switching plans.",
            currentSubscriptionId: existing.id,
            currentStatus: existing.status,
          });
        }
      } catch (e) {
        return res.status(500).json({ success: false, message: e.message });
      }
    }

    const subscription = await razorpay.subscriptions.create({
      plan_id: resolved.planId,
      total_count: resolved.totalCount,
      quantity: 1,
      customer_notify: 1,
      notes: {
        firebase_uid: req.user.uid,
        plan_type: resolved.planType,
      },
    });

    user.subscriptionId = subscription.id;
    user.subscriptionStatus = subscription.status;
    user.planType = resolved.planType;
    await user.save();

    res.json({
      success: true,
      subscriptionId: subscription.id,
      status: subscription.status,
      planType: resolved.planType,
      key: RAZORPAY_KEY_ID,
    });
  } catch (err) {
    console.error("/subscription/create error:", err.message);
    res.status(500).json({ success: false, message: err.message });
  }
});

app.get("/subscription/status", userAuth, async (req, res) => {
  try {
    await connectDB();
    const user = await User.findOne({ uid: req.user.uid });

    if (!user) {
      return res.json({ success: true, premium: false });
    }

    res.json({
      success: true,
      premium: user.isPremium,
      status: user.subscriptionStatus,
      expiry: user.premiumExpiresAt,
      subscriptionId: user.subscriptionId,
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.post("/subscription/refresh", userAuth, async (req, res) => {
  try {
    await connectDB();
    const result = await syncSubscription(req.user.uid);
    res.json({ success: true, ...result });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.post("/subscription/verify", userAuth, async (req, res) => {
  try {
    const { razorpay_payment_id, razorpay_subscription_id, razorpay_signature } = req.body;

    const generated = crypto
      .createHmac("sha256", RAZORPAY_KEY_SECRET)
      .update(razorpay_payment_id + "|" + razorpay_subscription_id)
      .digest("hex");

    if (generated !== razorpay_signature) {
      return res.status(400).json({ success: false, message: "Invalid signature" });
    }

    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.post("/subscription/webhook", async (req, res) => {
  try {
    const signature = req.headers["x-razorpay-signature"];

    if (!signature || !verifyWebhookSignature(req.rawBody, signature)) {
      return res.status(400).json({ success: false, message: "Invalid webhook signature" });
    }

    await connectDB();

    const payload = req.body;
    const event = payload.event;
    const subscription = payload.payload?.subscription?.entity;

    if (!subscription) {
      return res.json({ success: true });
    }

    const firebaseUid = subscription.notes?.firebase_uid;

    let user = null;
    if (firebaseUid) {
      user = await User.findOne({ uid: firebaseUid });
    }
    if (!user) {
      user = await User.findOne({ subscriptionId: subscription.id });
    }
    if (!user) {
      return res.json({ success: true, message: "User not found" });
    }

    user.subscriptionId = subscription.id;
    user.subscriptionStatus = subscription.status;

    if (subscription.plan_id === PLAN_CONFIG.monthly.planId) user.planType = "monthly";
    else if (subscription.plan_id === PLAN_CONFIG.yearly.planId) user.planType = "yearly";

    switch (event) {
      case "subscription.activated": {
        user.isPremium = true;
        if (subscription.current_start && subscription.current_end) {
          user.premiumPeriods.push({
            from: new Date(subscription.current_start * 1000),
            to: new Date(subscription.current_end * 1000),
          });
          user.premiumExpiresAt = new Date(subscription.current_end * 1000);
        }
        break;
      }

      case "subscription.charged": {
        user.isPremium = true;
        if (subscription.current_start && subscription.current_end) {
          user.premiumPeriods.push({
            from: new Date(subscription.current_start * 1000),
            to: new Date(subscription.current_end * 1000),
          });
          user.premiumExpiresAt = new Date(subscription.current_end * 1000);
        }
        if (payload.payload.payment && payload.payload.payment.entity) {
          user.lastPaymentId = payload.payload.payment.entity.id;
        }
        break;
      }

      case "subscription.completed":
      case "subscription.cancelled":
      case "subscription.halted":
        user.isPremium = false;
        user.subscriptionId = null;
        user.subscriptionStatus = null;
        break;

      case "subscription.paused":
        user.isPremium = false;
        break;

      case "payment.failed":
        user.isPremium = false;
        break;

      default:
        break;
    }

    await user.save();
    res.json({ success: true });
  } catch (err) {
    console.error("/subscription/webhook error:", err.message);
    res.status(500).json({ success: false, message: err.message });
  }
});

app.get("/user/analytics/summary", userAuth, async (req, res) => {
  try {
    await connectDB();
    const uid = req.user.uid;

    const [results, userDoc] = await Promise.all([
      Result.find({ userId: uid }).lean(),
      User.findOne({ uid }).lean(),
    ]);

    if (results.length === 0) {
      return res.json({
        testsGiven: 0,
        totalCorrect: 0,
        totalIncorrect: 0,
        totalUnattempted: 0,
        totalTimeTakenSeconds: 0,
        avgPercentage: 0,
        bestPercentage: 0,
        currentStreak: 0,
        longestStreak: 0,
        quizTypeStats: {},
      });
    }

    let totalCorrect     = 0;
    let totalIncorrect   = 0;
    let totalUnattempted = 0;
    let totalTime        = 0;
    let bestPct          = 0;
    let sumPct           = 0;

    const quizTypeStats = {
      paidDaily:  { count: 0, totalCorrect: 0, totalIncorrect: 0, totalMarks: 0, bestPercentage: 0, avgPercentage: 0, totalTimeSeconds: 0, minTimeSeconds: undefined, maxTimeSeconds: undefined },
      paidPhase1: { count: 0, totalCorrect: 0, totalIncorrect: 0, totalMarks: 0, bestPercentage: 0, avgPercentage: 0, totalTimeSeconds: 0, minTimeSeconds: undefined, maxTimeSeconds: undefined },
      paidPhase2: { count: 0, totalCorrect: 0, totalIncorrect: 0, totalMarks: 0, bestPercentage: 0, avgPercentage: 0, totalTimeSeconds: 0, minTimeSeconds: undefined, maxTimeSeconds: undefined },
    };

    results.forEach(r => {
      totalCorrect      += r.correct      || 0;
      totalIncorrect    += r.incorrect    || 0;
      totalUnattempted  += r.unattempted  || 0;
      totalTime         += r.timeTakenSeconds || 0;

      const pct = r.totalQuestions > 0 ? (r.correct / r.totalQuestions) * 100 : 0;
      sumPct  += pct;
      bestPct  = Math.max(bestPct, pct);

      let qtKey = 'paidDaily';
      if (r.phase === 'GS' && r.totalQuestions === 100) qtKey = 'paidPhase1';
      if (r.phase === 'CSAT') qtKey = 'paidPhase2';

      const qt = quizTypeStats[qtKey];
      qt.count++;
      qt.totalCorrect   += r.correct   || 0;
      qt.totalIncorrect += r.incorrect || 0;
      qt.totalMarks     += (r.correct * 2) - (r.incorrect * (2 / 3));
      qt.bestPercentage  = Math.max(qt.bestPercentage, pct);

      const timeSec = r.timeTakenSeconds || 0;
      qt.totalTimeSeconds += timeSec;
      if (qt.minTimeSeconds === undefined || timeSec < qt.minTimeSeconds) qt.minTimeSeconds = timeSec;
      if (qt.maxTimeSeconds === undefined || timeSec > qt.maxTimeSeconds) qt.maxTimeSeconds = timeSec;
    });

    Object.values(quizTypeStats).forEach(qt => {
      if (qt.count > 0) {
        qt.avgPercentage  = (qt.totalCorrect / (qt.totalCorrect + qt.totalIncorrect)) * 100 || 0;
        qt.avgTimeSeconds = qt.totalTimeSeconds / qt.count;
        qt.avgTimeMinutes = Math.floor(qt.avgTimeSeconds / 60);
        qt.avgTimeSecs    = Math.round(qt.avgTimeSeconds % 60);
      }
    });

    const { currentStreak, longestStreak } = computeStreak(
      results,
      userDoc?.premiumPeriods || [],
    );

    const testsGiven    = results.length;
    const avgPercentage = testsGiven > 0 ? sumPct / testsGiven : 0;

    res.json({
      testsGiven,
      totalCorrect,
      totalIncorrect,
      totalUnattempted,
      totalTimeTakenSeconds: totalTime,
      avgPercentage:  Math.round(avgPercentage * 10) / 10,
      bestPercentage: Math.round(bestPct        * 10) / 10,
      currentStreak,
      longestStreak,
      quizTypeStats,
    });
  } catch (err) {
    console.error("/user/analytics/summary error:", err.message);
    res.status(500).json({ message: "Failed to fetch analytics summary" });
  }
});

app.get("/user/analytics/attempts", userAuth, async (req, res) => {
  try {
    await connectDB();
    const uid   = req.user.uid;
    const limit = parseInt(req.query.limit) || 30;

    const attempts = await Result.aggregate([
      { $match: { userId: uid } },
      { $sort: { submittedAt: -1 } },
      { $limit: limit },
      {
        $lookup: {
          from: "tests",
          localField: "testId",
          foreignField: "_id",
          as: "testDoc"
        }
      },
      {
        $addFields: {
          testTitle: {
            $ifNull: [{ $arrayElemAt: ["$testDoc.title", 0] }, "Paid Test"]
          }
        }
      }
    ]);

    res.json(attempts.map(a => ({
      _id:             a._id.toString(),
      testId:          a.testId.toString(),
      testTitle:       a.testTitle,
      phase:           a.phase,
      score:           a.score,
      correct:         a.correct,
      incorrect:       a.incorrect,
      unattempted:     a.unattempted,
      totalQuestions:  a.totalQuestions,
      submittedAt:     a.submittedAt.toISOString(),
      timeTakenSeconds: a.timeTakenSeconds || 0,
      isLate:          a.isLate || false,
    })));
  } catch (err) {
    console.error("/user/analytics/attempts error:", err.message);
    res.status(500).json({ message: "Failed to fetch attempts" });
  }
});

app.get("/user/today-test", userAuth, async (req, res) => {
  try {
    await connectDB();
    const istNow  = nowIST();
    const todayIST = istNow.toISOString().split("T")[0];

    const test = await Test.findOne({ date: todayIST, testType: "paid" });
    if (!test) return res.status(404).json({ message: "No paid test available today" });

    const response = await buildTestResponse(test, req.user.uid, istNow);
    res.json(response);
  } catch (err) {
    console.error("/user/today-test error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/user/today-tests", userAuth, async (req, res) => {
  try {
    await connectDB();
    const istNow   = nowIST();
    const todayIST = istNow.toISOString().split("T")[0];

    const tests = await Test.find({ date: todayIST, testType: "paid" })
      .sort({ startTime: 1 })
      .lean();

    if (!tests.length) return res.status(404).json({ message: "No paid tests available today" });

    const responses = await Promise.all(
      tests.map(test => buildTestResponse(test, req.user.uid, istNow))
    );

    res.json({ tests: responses, count: responses.length });
  } catch (err) {
    console.error("/user/today-tests error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/user/submission-status/:testId", userAuth, async (req, res) => {
  try {
    await connectDB();
    const test = await Test.findById(req.params.testId);
    if (!test) return res.status(404).json({ message: "Test not found" });

    const phases = test.isSundayFullTest ? ["GS", "CSAT"] : ["GS"];

    const userResults = await Result.find({
      userId: req.user.uid,
      testId: test._id,
      phase: { $in: phases },
    }).lean();

    const submittedPhases = userResults.map(r => r.phase);
    const hasSubmitted    = submittedPhases.length > 0;
    const rankRevealNow   = isRankRevealTime();

    const response = {
      hasSubmitted,
      submittedPhases,
      rankRevealTime: rankRevealTimeIST(),
      rankRevealNow,
    };

    if (hasSubmitted && rankRevealNow) {
      const rankData = {};
      for (const r of userResults) {
        const { rank, totalParticipants } = await computeRank(test._id, r.phase, r.score, r.submittedAt);
        rankData[r.phase] = {
          score:             Math.round(r.score * 100) / 100,
          correct:           r.correct,
          incorrect:         r.incorrect,
          unattempted:       r.unattempted,
          rank,
          totalParticipants,
          isLate:            r.isLate || false,
        };
      }
      response.rankData = rankData;

      if (test.isSundayFullTest && userResults.length === 2) {
        const combinedScore = userResults.reduce((sum, r) => sum + r.score, 0);
        const betterCombined = await Result.aggregate([
          { $match: { testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] } } },
          { $group: { _id: "$userId", total: { $sum: "$score" } } },
          { $match: { total: { $gt: combinedScore } } },
          { $count: "count" }
        ]);
        const combinedRank = (betterCombined[0]?.count || 0) + 1;
        const totalUsers = await Result.distinct("userId", {
          testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] }
        }).then(ids => new Set(ids).size);

        response.combinedRank = {
          score: Math.round(combinedScore * 100) / 100,
          rank:  combinedRank,
          totalParticipants: totalUsers
        };
      }
    }

    res.json(response);
  } catch (err) {
    console.error("/user/submission-status error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/user/overall-rank", userAuth, async (req, res) => {
  try {
    await connectDB();
    const uid = req.user.uid;

    const userResults = await Result.find({ userId: uid }).lean();
    if (userResults.length === 0) {
      return res.json({
        hasRank: false,
        message: "Complete at least one paid test to see your overall rank.",
        totalMarks:   0,
        totalCorrect: 0,
        testsGiven:   0,
      });
    }

    let totalMarks   = 0;
    let totalCorrect = 0;
    userResults.forEach(r => {
      totalMarks   += (r.correct * 2) - (r.incorrect * (2 / 3));
      totalCorrect += r.correct || 0;
    });

    const betterUsers = await Result.aggregate([
      { $match: { isLate: false } },
      {
        $group: {
          _id: "$userId",
          totalMarks: {
            $sum: { $subtract: [{ $multiply: ["$correct", 2] }, { $multiply: ["$incorrect", 2 / 3] }] }
          }
        }
      },
      { $match: { totalMarks: { $gt: totalMarks } } },
      { $count: "count" }
    ]);

    const rank              = (betterUsers[0]?.count || 0) + 1;
    const totalParticipants = await Result.distinct("userId", { isLate: false })
      .then(ids => new Set(ids).size);

    res.json({
      hasRank: true,
      rank,
      totalMarks:       Math.round(totalMarks * 100) / 100,
      totalCorrect,
      testsGiven:       userResults.length,
      totalParticipants,
      percentile: totalParticipants > 0
        ? Math.round(((totalParticipants - rank) / totalParticipants) * 100)
        : 0,
      message: "Your overall rank among all participants"
    });
  } catch (err) {
    console.error("/user/overall-rank error:", err.message);
    res.status(500).json({ message: "Failed to calculate overall rank" });
  }
});

app.get("/leaderboard/global", async (req, res) => {
  try {
    await connectDB();
    const limit = parseInt(req.query.limit) || 50;

    const leaderboard = await Result.aggregate([
      { $match: { isLate: false } },
      {
        $group: {
          _id: "$userId",
          totalMarks: {
            $sum: { $subtract: [{ $multiply: ["$correct", 2] }, { $multiply: ["$incorrect", 2 / 3] }] }
          },
          totalCorrect: { $sum: "$correct" },
          testsGiven:   { $sum: 1 }
        }
      },
      { $sort: { totalMarks: -1 } },
      { $limit: limit },
      {
        $project: {
          userId:      "$_id",
          totalMarks:  { $round: ["$totalMarks", 2] },
          totalCorrect: 1,
          testsGiven:   1,
          rank:         { $literal: 0 }
        }
      }
    ]);

    leaderboard.forEach((entry, index) => { entry.rank = index + 1; });

    res.json({
      leaderboard,
      totalParticipants: await Result.distinct("userId", { isLate: false })
        .then(ids => new Set(ids).size)
    });
  } catch (err) {
    console.error("/leaderboard/global error:", err.message);
    res.status(500).json({ message: "Failed to fetch global leaderboard" });
  }
});

app.post("/user/submit-test/:testId", userAuth, async (req, res) => {
  try {
    await connectDB();

    const { phase, answers, timeTakenSeconds } = req.body;

    if (!["GS", "CSAT"].includes(phase)) {
      return res.status(400).json({ message: "phase must be 'GS' or 'CSAT'" });
    }
    if (!Array.isArray(answers)) {
      return res.status(400).json({ message: "answers must be an array of objects" });
    }

    const test = await Test.findById(req.params.testId);
    if (!test || test.testType !== "paid") {
      return res.status(404).json({ message: "Paid test not found" });
    }

    const istNow  = nowIST();
    const endIST  = new Date(test.endTime.getTime() + IST_OFFSET_MS);
    const testDateParts = (test.date || "").split("-");
    let deadlineIST = endIST;
    if (testDateParts.length === 3) {
      const [y, m, d] = testDateParts.map(Number);
      const endOfDayUTC = new Date(Date.UTC(y, m - 1, d, 18, 29, 59));
      const endOfDayIST = new Date(endOfDayUTC.getTime() + IST_OFFSET_MS);
      deadlineIST = endOfDayIST > endIST ? endOfDayIST : endIST;
    }

    const startIST = new Date(test.startTime.getTime() + IST_OFFSET_MS);
    if (istNow < startIST) {
      return res.status(403).json({ message: "Test has not started yet." });
    }

    const isLate = istNow > deadlineIST;
    const now    = new Date();

    const existing = await Result.findOne({ userId: req.user.uid, testId: test._id, phase });
    if (existing) {
      return res.status(403).json({
        message: `You have already submitted ${phase} phase. You can only preview your attempt.`,
        alreadySubmitted: true
      });
    }

    let qFilter = { testId: test._id };
    if (test.isSundayFullTest) qFilter.phase = phase;

    const questions = await Question.find(qFilter).lean();
    if (questions.length === 0) {
      return res.status(404).json({ message: "No questions found for this phase" });
    }

    let correct = 0, incorrect = 0, unattempted = 0, attempted = 0;

    const savedAnswers = answers.map(ans => {
      const q = questions.find(qq => qq._id.toString() === ans.questionId);
      if (!q) return { questionId: ans.questionId, selectedOption: null };

      const selected = ans.selectedOption;
      if (!selected) { unattempted++; return { questionId: ans.questionId, selectedOption: null }; }

      attempted++;
      if (selected === q.correctOption) correct++;
      else incorrect++;

      return { questionId: ans.questionId, selectedOption: selected };
    });

    const score = calculateNetScore(correct, incorrect);

    await Result.create({
      userId: req.user.uid,
      testId: test._id,
      phase,
      score,
      correct,
      incorrect,
      unattempted,
      attempted,
      totalQuestions: questions.length,
      submittedAt: now,
      startedAt:   now,
      isLate,
      answers: savedAnswers,
      timeTakenSeconds: timeTakenSeconds || 0
    });

    await User.findOneAndUpdate(
      { uid: req.user.uid },
      {
        $setOnInsert: { uid: req.user.uid },
        $set: { displayName: req.user.name || "", email: req.user.email || "" },
      },
      { upsert: true }
    );

    const rankRevealNow = isRankRevealTime();
    const { rank, totalParticipants } = await computeRank(test._id, phase, score, now);

    const responseBase = {
      phase,
      score:           Math.round(score * 100) / 100,
      correct,
      incorrect,
      unattempted,
      totalQuestions:  questions.length,
      isLate,
      ranked:          true,
      rankRevealTime:  rankRevealTimeIST(),
      rankRevealNow,
    };

    if (!rankRevealNow) {
      return res.json({
        ...responseBase,
        message: isLate
          ? `Submitted after test window but your attempt is saved! Your rank will be visible at ${rankRevealTimeIST()}.`
          : `Test submitted successfully! Your rank will be available at ${rankRevealTimeIST()} today.`,
      });
    }

    responseBase.rank                    = rank;
    responseBase.totalRankedParticipants = totalParticipants;
    responseBase.message = isLate
      ? "Submitted! Your rank is inserted among on-time participants."
      : "Test submitted! Here is your rank.";

    if (test.isSundayFullTest) {
      const gsResult   = await Result.findOne({ userId: req.user.uid, testId: test._id, phase: "GS" });
      const csatResult = await Result.findOne({ userId: req.user.uid, testId: test._id, phase: "CSAT" });

      if (gsResult && csatResult) {
        const combinedScore = gsResult.score + csatResult.score;
        const betterCombined = await Result.aggregate([
          { $match: { testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] } } },
          { $group: { _id: "$userId", totalScore: { $sum: "$score" } } },
          { $match: { totalScore: { $gt: combinedScore } } },
          { $count: "count" }
        ]);
        const combinedRank = (betterCombined[0]?.count || 0) + 1;
        const totalUnique  = await Result.distinct("userId", {
          testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] }
        }).then(arr => new Set(arr).size);

        responseBase.combined = {
          score: Math.round(combinedScore * 100) / 100,
          rank:  combinedRank,
          totalParticipants: totalUnique
        };
      }
    }

    res.json(responseBase);
  } catch (err) {
    console.error("/user/submit-test error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/user/my-rank/:testId", userAuth, async (req, res) => {
  try {
    await connectDB();
    const test = await Test.findById(req.params.testId);
    if (!test) return res.status(404).json({ message: "Test not found" });

    const phases = test.isSundayFullTest ? ["GS", "CSAT"] : ["GS"];

    const userResults = await Result.find({
      userId: req.user.uid,
      testId: test._id,
      phase: { $in: phases },
    }).lean();

    if (userResults.length === 0) {
      return res.status(404).json({ message: "No attempt found for this test" });
    }

    const rankRevealNow = isRankRevealTime();

    if (!rankRevealNow) {
      return res.json({
        rankRevealNow:   false,
        rankRevealTime:  rankRevealTimeIST(),
        message:         `Ranks will be revealed at ${rankRevealTimeIST()} today. Come back then!`,
        hasSubmitted:    true,
      });
    }

    const response = { phases: {}, rankRevealNow: true };

    for (const r of userResults) {
      const { rank, totalParticipants } = await computeRank(test._id, r.phase, r.score, r.submittedAt);
      response.phases[r.phase] = {
        score:             Math.round(r.score * 100) / 100,
        correct:           r.correct,
        incorrect:         r.incorrect,
        unattempted:       r.unattempted,
        rank,
        totalParticipants,
        isLate:            r.isLate || false,
      };
    }

    if (test.isSundayFullTest && userResults.length === 2) {
      const combinedScore = userResults.reduce((sum, r) => sum + r.score, 0);
      const betterCombined = await Result.aggregate([
        { $match: { testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] } } },
        { $group: { _id: "$userId", total: { $sum: "$score" } } },
        { $match: { total: { $gt: combinedScore } } },
        { $count: "count" }
      ]);
      const combinedRank = (betterCombined[0]?.count || 0) + 1;
      const totalUsers   = await Result.distinct("userId", {
        testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] }
      }).then(ids => new Set(ids).size);

      response.combined = {
        score: Math.round(combinedScore * 100) / 100,
        rank:  combinedRank,
        totalParticipants: totalUsers
      };
    }

    res.json(response);
  } catch (err) {
    console.error("/user/my-rank error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/user/leaderboard/:testId", userAuth, async (req, res) => {
  try {
    await connectDB();
    const test = await Test.findById(req.params.testId);
    if (!test) return res.status(404).json({ message: "Test not found" });

    if (!test.isSundayFullTest) {
      const results = await Result.find({ testId: test._id, phase: "GS", isLate: false })
        .sort({ score: -1, submittedAt: 1 })
        .limit(50)
        .lean();
      const total = await Result.countDocuments({ testId: test._id, phase: "GS", isLate: false });
      return res.json({
        phase: "GS",
        leaderboard: results.map(r => ({
          userId: r.userId,
          score:  Math.round(r.score * 100) / 100,
          submittedAt: r.submittedAt
        })),
        totalRankedParticipants: total,
        note: "On-time GS attempts"
      });
    }

    const gsResults   = await Result.find({ testId: test._id, phase: "GS",   isLate: false }).sort({ score: -1, submittedAt: 1 }).limit(20).lean();
    const csatResults = await Result.find({ testId: test._id, phase: "CSAT", isLate: false }).sort({ score: -1, submittedAt: 1 }).limit(20).lean();
    const combined    = await Result.aggregate([
      { $match: { testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] } } },
      { $group: { _id: "$userId", totalScore: { $sum: "$score" }, gsScore: { $sum: { $cond: [{ $eq: ["$phase", "GS"] }, "$score", 0] } }, csatScore: { $sum: { $cond: [{ $eq: ["$phase", "CSAT"] }, "$score", 0] } } } },
      { $sort: { totalScore: -1 } },
      { $limit: 20 }
    ]);

    res.json({
      isSundayFullTest: true,
      gs:   { leaderboard: gsResults.map(r => ({ userId: r.userId, score: Math.round(r.score*100)/100 })), total: await Result.countDocuments({ testId: test._id, phase: "GS", isLate: false }) },
      csat: { leaderboard: csatResults.map(r => ({ userId: r.userId, score: Math.round(r.score*100)/100 })), total: await Result.countDocuments({ testId: test._id, phase: "CSAT", isLate: false }) },
      combined: {
        leaderboard: combined.map((e, i) => ({ rank: i+1, userId: e._id, totalScore: Math.round(e.totalScore*100)/100, gs: Math.round(e.gsScore*100)/100, csat: Math.round(e.csatScore*100)/100 })),
        totalUniqueParticipants: await Result.distinct("userId", { testId: test._id, isLate: false, phase: { $in: ["GS", "CSAT"] } }).then(a => a.length)
      }
    });
  } catch (err) {
    console.error("/user/leaderboard error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/user/review-test/:testId", userAuth, async (req, res) => {
  try {
    await connectDB();
    const test = await Test.findById(req.params.testId);
    if (!test || test.testType !== "paid") {
      return res.status(404).json({ message: "Test not found or not paid" });
    }

    const phase = req.query.phase || "GS";

    let filter = { userId: req.user.uid, testId: test._id };
    if (test.isSundayFullTest) filter.phase = phase;

    const result = await Result.findOne(filter);
    if (!result) {
      return res.status(404).json({ message: `No submission found for phase ${phase}` });
    }

    let qFilter = { testId: test._id };
    if (test.isSundayFullTest) qFilter.phase = phase;

    const questions = await Question.find(qFilter).sort({ questionNumber: 1 }).lean();

    const reviewQuestions = questions.map(q => {
      const userAns = result.answers.find(a => a.questionId === q._id.toString());
      return {
        questionNumber:    q.questionNumber,
        questionStatement: q.questionStatement,
        options:           q.options,
        yourAnswer:        userAns?.selectedOption || null,
        correctAnswer:     q.correctOption,
        isCorrect:         userAns ? userAns.selectedOption === q.correctOption : false
      };
    });

    const rankRevealNow = isRankRevealTime();
    let rankInfo = null;
    if (rankRevealNow) {
      const { rank, totalParticipants } = await computeRank(test._id, result.phase, result.score, result.submittedAt);
      rankInfo = {
        phase:             result.phase,
        score:             Math.round(result.score * 100) / 100,
        rank,
        totalParticipants,
        isLate:            result.isLate || false,
      };
    }

    res.json({
      title:       test.title,
      phase:       result.phase,
      score:       Math.round(result.score * 100) / 100,
      correct:     result.correct,
      incorrect:   result.incorrect,
      unattempted: result.unattempted,
      submittedAt: result.submittedAt.toISOString(),
      isLate:      result.isLate || false,
      rankRevealNow,
      rankRevealTime: rankRevealTimeIST(),
      rankInfo,
      questions: reviewQuestions,
      message: rankRevealNow
        ? result.isLate
          ? "Your rank is inserted among on-time participants."
          : "Review your answers and performance"
        : `Rank will be available at ${rankRevealTimeIST()}. Come back then!`
    });
  } catch (err) {
    console.error("/user/review-test error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/free/tests", async (req, res) => {
  try {
    await connectDB();
    const tests = await Test.find({ testType: "free" })
      .sort({ createdAt: -1 })
      .select("title date totalQuestions startTime endTime createdAt")
      .lean();

    if (!tests.length) return res.status(404).json({ message: "No free tests available" });

    res.json({
      success: true,
      tests: tests.map(t => ({
        testId:         t._id.toString(),
        title:          t.title || "BPSC Free Practice Test",
        date:           t.date  || "—",
        totalQuestions: t.totalQuestions,
        createdAt:      t.createdAt ? new Date(t.createdAt).toISOString() : null,
        startTimeIST:   t.startTime ? toISTISOString(t.startTime) : null,
        endTimeIST:     t.endTime   ? toISTISOString(t.endTime)   : null,
      }))
    });
  } catch (err) {
    console.error("/free/tests error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/free/test/:testId", async (req, res) => {
  try {
    await connectDB();
    const test = await Test.findOne({ _id: req.params.testId, testType: "free" }).lean();
    if (!test) return res.status(404).json({ message: "Free test not found" });

    const questions = await Question.find({ testId: test._id }).select("-correctOption").lean();

    res.json({
      status:         "active",
      testId:         test._id.toString(),
      title:          test.title || "BPSC Free Practice Test",
      totalQuestions: test.totalQuestions,
      date:           test.date,
      questions,
      note:                 "Persistent free practice test — available anytime until removed by admin",
      isPersistentFreeTest: true
    });
  } catch (err) {
    console.error("/free/test/:testId error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/free/today-test", async (req, res) => {
  try {
    await connectDB();
    const test = await Test.findOne({ testType: "free" }).sort({ createdAt: -1 }).lean();
    if (!test) return res.status(404).json({ message: "No free test available at the moment" });

    const questions = await Question.find({ testId: test._id }).select("-correctOption").lean();

    res.json({
      status:         "active",
      testId:         test._id.toString(),
      title:          test.title || "BPSC Free Practice Test",
      totalQuestions: test.totalQuestions,
      date:           test.date,
      questions,
      note:                 "Persistent free practice test — available anytime until removed by admin",
      isPersistentFreeTest: true
    });
  } catch (err) {
    console.error("/free/today-test error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.post("/free/submit-test/:testId", async (req, res) => {
  try {
    await connectDB();
    const { answers, timeTakenSeconds } = req.body;
    if (!Array.isArray(answers)) return res.status(400).json({ message: "answers must be array" });

    const questions = await Question.find({ testId: req.params.testId });
    if (!questions.length) return res.status(404).json({ message: "Test not found" });

    let score = 0;
    questions.forEach(q => {
      const ua = answers.find(a => a.questionId === q._id.toString());
      if (ua && ua.selectedOption === q.correctOption) score++;
    });

    const result = await FreeResult.create({
      testId: req.params.testId,
      score,
      totalQuestions: questions.length
    });

    const betterCount = await FreeResult.countDocuments({
      testId: req.params.testId,
      $or: [
        { score: { $gt: score } },
        { score, submittedAt: { $lt: result.submittedAt } }
      ]
    });
    const total = await FreeResult.countDocuments({ testId: req.params.testId });

    res.json({
      score,
      total,
      yourRank:    betterCount + 1,
      rankDisplay: `${betterCount + 1} / ${total}`,
      message:     "Submitted – your rank is visible on the public leaderboard"
    });
  } catch (err) {
    console.error("/free/submit error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/free/leaderboard/:testId", async (req, res) => {
  try {
    await connectDB();
    const results = await FreeResult.find({ testId: req.params.testId })
      .sort({ score: -1, submittedAt: 1 })
      .limit(100)
      .lean();
    const total = await FreeResult.countDocuments({ testId: req.params.testId });

    res.json({
      leaderboard: results.map((r, idx) => ({
        rank:           idx + 1,
        score:          r.score,
        totalQuestions: r.totalQuestions,
        submittedAt:    r.submittedAt
      })),
      totalParticipants: total
    });
  } catch (err) {
    console.error("/free/leaderboard error:", err.message);
    res.status(500).json({ message: "Server error" });
  }
});

function todayISTDateKey() {
  return toISTDateKey(Date.now());
}

app.post("/qa/ask", userAuth, async (req, res) => {
  const question = (req.body?.question || "").trim();
  if (!question) return res.status(400).json({ message: "question is required" });
  if (question.length > 500) return res.status(400).json({ message: "question is too long (max 500 characters)" });

  try {
    await connectDB();

    const user = await User.findOneAndUpdate(
      { uid: req.user.uid },
      {
        $setOnInsert: { uid: req.user.uid },
        $set: { displayName: req.user.name || "", email: req.user.email || "" },
      },
      { upsert: true, new: true }
    );

    if (ENABLE_QUERY_DAILY_LIMIT && !user.isPremium) {
      const today = todayISTDateKey();
      const usage = user.qaUsage || {};
      const usedToday = usage.date === today ? (usage.count || 0) : 0;

      if (usedToday >= FREE_DAILY_QUERY_LIMIT) {
        return res.status(429).json({
          message: "You have used your free question for today. Upgrade to premium for unlimited questions with Ask Cron.",
          dailyLimit: FREE_DAILY_QUERY_LIMIT,
          used: usedToday,
        });
      }

      user.qaUsage = { date: today, count: usedToday + 1 };
      await user.save();
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    function sendEvent(event, data) {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    }

    const queryVector = await embedOne(question);

    const [kbPoints, pyqPoints] = await Promise.all([
      searchKnowledgeBase(queryVector, QA_KNOWLEDGE_BASE_TOP_K),
      searchQuestionBank(queryVector, QA_QUESTION_BANK_TOP_K),
    ]);

    const relatedPyqs = pyqPoints.map(p => {
      const pl = p.payload || {};
      return {
        exam: pl.exam || null,
        year: pl.year || null,
        question: pl.question || null,
        options: pl.options || null,
        answer: pl.answer || null,
      };
    });

    const kbText = formatKnowledge(kbPoints);
    const { system, user: userPrompt } = buildAnswerPrompt({ question, kbText });
    const finalAnswer = await streamPSModel(system, userPrompt, delta => {
      sendEvent("token", { content: delta });
    });

    sendEvent("done", {
      answer: finalAnswer,
      relatedPyqs,
    });
    res.end();

    try {
      const aiConn = await connectAIChatHistoryDB();
      const AIChatHistory = getAIChatHistoryModel(aiConn);
      await AIChatHistory.create({
        userId: req.user.uid,
        question,
        answer: finalAnswer,
        relatedPyqs,
      });
    } catch (e) {
      console.error("[AIChatHistory save]", e.message);
    }
  } catch (err) {
    console.error("/qa/ask error:", err.message);
    if (!res.headersSent) {
      res.status(500).json({ message: "Server error" });
    } else {
      try { res.end(); } catch (_) {}
    }
  }
});

app.get("/qa/history", userAuth, async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 50, 200);
    const before = req.query.before;

    const aiConn = await connectAIChatHistoryDB();
    const AIChatHistory = getAIChatHistoryModel(aiConn);

    const filter = { userId: req.user.uid };
    if (before) {
      filter.createdAt = { $lt: new Date(before) };
    }

    const history = await AIChatHistory.find(filter)
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean();

    res.json({
      success: true,
      history: history.map(h => ({
        id: h._id.toString(),
        question: h.question,
        answer: h.answer,
        relatedPyqs: h.relatedPyqs || [],
        createdAt: h.createdAt,
      })),
      count: history.length,
    });
  } catch (err) {
    console.error("/qa/history error:", err.message);
    res.status(500).json({ success: false, message: "Failed to fetch chat history" });
  }
});

app.delete("/qa/history/:id", userAuth, async (req, res) => {
  try {
    const aiConn = await connectAIChatHistoryDB();
    const AIChatHistory = getAIChatHistoryModel(aiConn);

    const deleted = await AIChatHistory.findOneAndDelete({
      _id: req.params.id,
      userId: req.user.uid,
    });

    if (!deleted) {
      return res.status(404).json({ success: false, message: "History entry not found" });
    }

    res.json({ success: true });
  } catch (err) {
    console.error("/qa/history delete error:", err.message);
    res.status(500).json({ success: false, message: "Failed to delete history entry" });
  }
});

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdDev(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const variance = mean(arr.map(x => (x - m) ** 2));
  return Math.sqrt(variance);
}

function trendSlope(percentagesChronological) {
  const n = percentagesChronological.length;
  if (n < 2) return 0;
  const xs = percentagesChronological.map((_, i) => i);
  const xMean = mean(xs);
  const yMean = mean(percentagesChronological);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - xMean) * (percentagesChronological[i] - yMean);
    den += (xs[i] - xMean) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

app.get("/user/analytics/selection-probability", userAuth, async (req, res) => {
  try {
    await connectDB();
    const uid = req.user.uid;

    const results = await Result.find({ userId: uid, isLate: false })
      .sort({ submittedAt: 1 })
      .lean();

    if (results.length === 0) {
      return res.json({
        hasEnoughData: false,
        message: "Attempt at least one paid test to see your selection probability estimate.",
        probability: null,
      });
    }

    let totalMarks = 0;
    results.forEach(r => {
      totalMarks += calculateNetScore(r.correct || 0, r.incorrect || 0);
    });

    const betterUsers = await Result.aggregate([
      { $match: { isLate: false } },
      {
        $group: {
          _id: "$userId",
          totalMarks: {
            $sum: { $subtract: [{ $multiply: ["$correct", 2] }, { $multiply: ["$incorrect", 2 / 3] }] }
          }
        }
      },
      { $match: { totalMarks: { $gt: totalMarks } } },
      { $count: "count" }
    ]);

    const rank = (betterUsers[0]?.count || 0) + 1;
    const totalParticipants = await Result.distinct("userId", { isLate: false })
      .then(ids => new Set(ids).size);

    const percentileScore = totalParticipants > 1
      ? clamp01((totalParticipants - rank) / (totalParticipants - 1))
      : 0.5;

    let totalCorrect = 0, totalIncorrect = 0, totalAttempted = 0, totalQuestions = 0;
    const percentages = [];

    results.forEach(r => {
      totalCorrect    += r.correct     || 0;
      totalIncorrect  += r.incorrect   || 0;
      totalAttempted  += r.attempted   || 0;
      totalQuestions  += r.totalQuestions || 0;
      const pct = r.totalQuestions > 0 ? (r.correct / r.totalQuestions) * 100 : 0;
      percentages.push(pct);
    });

    const accuracyScore = (totalCorrect + totalIncorrect) > 0
      ? clamp01(totalCorrect / (totalCorrect + totalIncorrect))
      : 0;

    const sd = stdDev(percentages);
    const consistencyScore = clamp01(1 - sd / 40);

    const attemptRateScore = totalQuestions > 0
      ? clamp01(totalAttempted / totalQuestions)
      : 0;

    let trendScore = 0.5;
    if (percentages.length >= 3) {
      const slope = trendSlope(percentages);
      const normalizedSlope = clamp01((slope / 5 + 1) / 2);
      trendScore = normalizedSlope;
    }

    const weights = {
      percentile:  0.35,
      accuracy:    0.25,
      consistency: 0.15,
      attemptRate: 0.10,
      trend:       0.15,
    };

    const weightedSum =
      percentileScore  * weights.percentile +
      accuracyScore    * weights.accuracy +
      consistencyScore * weights.consistency +
      attemptRateScore * weights.attemptRate +
      trendScore        * weights.trend;

    const z = (weightedSum - 0.5) * 6;
    const probability = clamp01(sigmoid(z));

    const dataConfidence = results.length >= 10 ? "high" : results.length >= 5 ? "medium" : "low";

    res.json({
      hasEnoughData: true,
      probability: Math.round(probability * 1000) / 10,
      confidence: dataConfidence,
      testsAnalyzed: results.length,
      breakdown: {
        percentile:  { score: Math.round(percentileScore * 100), rank, totalParticipants },
        accuracy:    { score: Math.round(accuracyScore * 100), totalCorrect, totalIncorrect },
        consistency: { score: Math.round(consistencyScore * 100), stdDevPercentage: Math.round(sd * 10) / 10 },
        attemptRate: { score: Math.round(attemptRateScore * 100), totalAttempted, totalQuestions },
        trend:       { score: Math.round(trendScore * 100) },
      },
      weights,
      message: "This is a statistical estimate based on your mock test performance relative to other aspirants — not an official prediction.",
    });
  } catch (err) {
    console.error("/user/analytics/selection-probability error:", err.message);
    res.status(500).json({ message: "Failed to calculate selection probability" });
  }
});

app.use((req, res) => {
  res.status(404).json({ success: false, message: "Route not found" });
});

app.use((err, req, res, next) => {
  res.status(500).json({ success: false, message: "Internal server error" });
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

module.exports = app;
