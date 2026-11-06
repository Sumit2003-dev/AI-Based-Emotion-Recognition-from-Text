export const EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "love", "disgust", "neutral"] as const;

export type Emotion = (typeof EMOTIONS)[number];

/** Limits shared by the analyzer UI and the server-side classifier. */
export const MAX_BATCH_TEXTS = 40;
export const MAX_TEXT_CHARS = 1200;

export const EMOTION_META: Record<Emotion, { label: string; blurb: string }> = {
  happy: { label: "Joy & Elation", blurb: "Joy, delight, elation, triumph or satisfaction." },
  sad: { label: "Sadness & Grief", blurb: "Loss, disappointment, grief or emotional fatigue." },
  angry: { label: "Anger & Frustration", blurb: "Irritation, outrage, boundary violation or frustration." },
  fear: { label: "Fear & Anxiety", blurb: "Worry, apprehension, threat detection or anxiety." },
  surprise: { label: "Surprise & Astonishment", blurb: "Astonishment, unexpected turns or sudden shock." },
  love: { label: "Love & Affection", blurb: "Affection, deep warmth, romance, devotion or intimacy." },
  disgust: { label: "Disgust & Aversion", blurb: "Revulsion, aversion, foulness, moral recoil or contempt." },
  neutral: { label: "Neutral & Objective", blurb: "Plainly stated, objective pragmatic communication." },
};

export type EmotionScores = Record<Emotion, number>;

export type AffectiveDimensions = {
  valence: number;
  arousal: number;
  arousal_label: string;
  dominance: number;
  dominance_label: string;
  subjectivity: number;
  subjectivity_label: string;
};

export type EmotionalGuidance = {
  psychology?: string | undefined;
  empathy_tip?: string | undefined;
  suggested_reply?: string | undefined;
};

export type EmotionToken = {
  token: string;
  is_emotional: boolean;
  emotion: string;
  color?: string | null | undefined;
  weight: number;
};

export type EmotionReading = {
  text: string;
  emotion: Emotion;
  confidence: number;
  scores: EmotionScores;
  detectedLanguage?: string | undefined;
  languageName?: string | undefined;
  isMultilingual?: boolean | undefined;
  translatedText?: string | null | undefined;
  sentiment?: string | undefined;
  sentimentScore?: number | undefined;
  intensityLevel?: string | undefined;
  intensityScore?: number | undefined;
  dimensions?: AffectiveDimensions | undefined;
  guidance?: EmotionalGuidance | undefined;
  tokens?: EmotionToken[] | undefined;
  isMixedEmotion?: boolean | undefined;
  secondaryEmotion?: Emotion | undefined;
  secondaryConfidence?: number | undefined;
  blendLabel?: string | undefined;
  blendExplanation?: string | undefined;
};

export type RawReading = {
  index: number;
  emotion: string;
  scores: Partial<EmotionScores>;
  detectedLanguage?: string | undefined;
  languageName?: string | undefined;
  isMultilingual?: boolean | undefined;
  translatedText?: string | null | undefined;
  sentiment?: string | undefined;
  sentimentScore?: number | undefined;
  intensityLevel?: string | undefined;
  intensityScore?: number | undefined;
  dimensions?: AffectiveDimensions | undefined;
  guidance?: EmotionalGuidance | undefined;
  tokens?: EmotionToken[] | undefined;
  isMixedEmotion?: boolean | undefined;
  secondaryEmotion?: Emotion | undefined;
  secondaryConfidence?: number | undefined;
  blendLabel?: string | undefined;
  blendExplanation?: string | undefined;
};

/** Compute fallback affective dimensions for an emotion when backend dimensions are absent. */
function computeDefaultDimensions(emotion: Emotion, confidence: number): AffectiveDimensions {
  const vadMap: Record<Emotion, [number, number, string, number, string]> = {
    happy: [0.85, 72, "High Surge", 75, "Assertive / Empowered"],
    sad: [-0.8, 25, "Calm / Low", 20, "Submissive / Helpless"],
    angry: [-0.75, 88, "High Surge", 70, "Assertive / Empowered"],
    surprise: [0.2, 82, "High Surge", 45, "Balanced Agency"],
    fear: [-0.82, 85, "High Surge", 15, "Submissive / Helpless"],
    love: [0.92, 65, "High Surge", 68, "Assertive / Empowered"],
    disgust: [-0.8, 60, "High Surge", 55, "Assertive / Empowered"],
    neutral: [0.0, 15, "Calm / Low", 50, "Balanced Agency"],
  };
  const [val, aro, aroLabel, dom, domLabel] = vadMap[emotion];
  const confScale = Math.min(1, Math.max(0.3, confidence));
  const subjPct = emotion === "neutral" ? 5 : Math.round(confScale * 90);
  return {
    valence: parseFloat((val * confScale).toFixed(3)),
    arousal: Math.round(aro * confScale),
    arousal_label: aroLabel,
    dominance: Math.round(dom * confScale),
    dominance_label: domLabel,
    subjectivity: subjPct,
    subjectivity_label:
      subjPct >= 65 ? "High Emotion" : subjPct >= 35 ? "Balanced" : "Objective Fact",
  };
}

/** Compute fallback empathy guidance when backend guidance is absent, matched to the input language. */
const GUIDANCE_BY_LANGUAGE: Record<string, Record<Emotion, EmotionalGuidance>> = {
  hi: {
    happy: {
      psychology: "संतुष्टि, आभार, उपलब्धि और सामाजिक जुड़ाव को दर्शाता है।",
      empathy_tip: "उनकी खुशी में शामिल हों और उनकी सफलता की सराहना करें।",
      suggested_reply: "यह तो बहुत ही बढ़िया खबर है! आपको ढेर सारी शुभकामनाएं! 🎉",
    },
    sad: {
      psychology: "दुख, अकेलापन, निराशा या नियंत्रण खोने की भावना को दर्शाता है।",
      empathy_tip: "बिना किसी निर्णय के उनका साथ दें और सांत्वना प्रदान करें।",
      suggested_reply:
        "यह सुनकर बहुत दुख हुआ। अगर किसी भी चीज़ की ज़रूरत हो तो मैं हमेशा आपके साथ हूँ।",
    },
    angry: {
      psychology: "अन्याय, उम्मीद टूटने, काम में रुकावट या सीमा के उल्लंघन से उत्पन्न।",
      empathy_tip: "माहौल को तुरंत शांत करें और उनकी परेशानी को ध्यानपूर्वक स्वीकार करें।",
      suggested_reply:
        "मैं समझ सकता हूँ कि आप परेशान क्यों हैं। चलिए इसे तुरंत मिलकर सुलझाते हैं।",
    },
    surprise: {
      psychology: "किसी अप्रत्याशित घटना या नई बात के सामने आने पर आश्चर्य की प्रतिक्रिया।",
      empathy_tip: "बात को स्पष्ट करने में मदद करें और उन्हें स्थिति को समझने का समय दें।",
      suggested_reply: "अरे वाह, यह तो वाकई अप्रत्याशित था! आप इसे कैसे संभाल रहे हैं?",
    },
    fear: {
      psychology: "भविष्य के खतरे, अनिश्चितता, असुरक्षा या चिंता की आशंका।",
      empathy_tip: "धैर्य और आश्वासन दें ताकि वे सुरक्षित महसूस कर सकें।",
      suggested_reply: "घबराइए मत, गहरी सांस लीजिए। हम मिलकर इसे संभाल लेंगे, आप अकेले नहीं हैं।",
    },
    love: {
      psychology: "गहरा भावनात्मक जुड़ाव और स्नेह।",
      empathy_tip: "सद्भावना और गर्मजोशी से उत्तर दें।",
      suggested_reply: "आपके इस स्नेह के लिए बहुत-बहुत धन्यवाद, यह दिल को छू गया। ❤️",
    },
    disgust: {
      psychology: "अप्रिय या असहज स्थिति से दूरी बनाने की इच्छा।",
      empathy_tip: "उनकी सीमा और भावनाओं का सम्मान करें।",
      suggested_reply: "मैं आपकी इस परेशानी को समझता हूँ, आपकी सीमा का पूरा सम्मान है।",
    },
    neutral: {
      psychology: "तथ्यात्मक, स्पष्ट और बिना किसी भावनात्मक उत्तेजना के बातचीत।",
      empathy_tip: "बातचीत को सीधा, स्पष्ट और उपयोगी बनाए रखें।",
      suggested_reply: "जानकारी के लिए धन्यवाद, नोट कर लिया गया है।",
    },
  },
  hinglish: {
    happy: {
      psychology: "Khushi, satisfaction, achievement ya celebration reflect karta hai.",
      empathy_tip: "Unki khushi ko acknowledge karo aur celebrate karne me unka sath do.",
      suggested_reply:
        "Aree waah, bohot badiya news hai! Party kab de rahe ho? Congratulations! 🎉",
    },
    sad: {
      psychology: "Disappointment, dukh, stress ya thoda low feel hone ka signal hai.",
      empathy_tip: "Judgemental hue bina unhe space aur emotional support provide karo.",
      suggested_reply:
        "Sunke bura laga bro. Tension mat le, sab theek ho jayega, main hu na yahan.",
    },
    angry: {
      psychology: "Frustration, expectation break hone ya unfair situation ki wajah se gussa.",
      empathy_tip: "Defensive mat bano, calmly suno aur solution nikalne pe focus karo.",
      suggested_reply:
        "Sahi me bohot frustrating hai bro. Gussa mat ho, shanti se baat karke sort karte hain.",
    },
    surprise: {
      psychology: "Achanak se kuch unexpected revelation ya event hone ka shock/surprise.",
      empathy_tip:
        "Unhe thoda time do absorb karne ka aur next steps figure out karne me help karo.",
      suggested_reply: "Arrey baap re! Yeh to bilkul unexpected tha! Ab kya plan hai?",
    },
    fear: {
      psychology: "Anxiety, tension, exam/future uncertainty ya risk ka darr.",
      empathy_tip: "Calmness aur assurance do taaki panic na ho.",
      suggested_reply:
        "Chill kar bhai, itna darr mat. Hum milke sambhal lenge step-by-step, tension mat le.",
    },
    love: {
      psychology: "Emotional warmth aur care.",
      empathy_tip: "Warmth reciprocate karo.",
      suggested_reply: "Dil se shukriya bhai, aapki baat bohot achhi lagi ❤️",
    },
    disgust: {
      psychology: "Strong dislike ya avoidance.",
      empathy_tip: "Boundaries respect karo.",
      suggested_reply: "Totally understandable hai. Apni boundaries maintain karna zaroori hai.",
    },
    neutral: {
      psychology: "To-the-point factual communication bina kisi heavy emotion ke.",
      empathy_tip: "Direct, clear aur professional tone maintain karo.",
      suggested_reply: "Theek hai bhai, samajh gaya. Update ke liye shukriya.",
    },
  },
  es: {
    happy: {
      psychology: "Refleja satisfacción, gratitud, logro o celebración compartida.",
      empathy_tip: "Valida y amplifica su alegría. Celebra su logro genuinamente.",
      suggested_reply: "¡Qué fantástica noticia! ¡Muchas felicidades por este gran logro! 🎉",
    },
    sad: {
      psychology: "Indica tristeza, decepción, duelo o vulnerabilidad emocional.",
      empathy_tip: "Ofrece compañía sin juzgar ni apresurarte a solucionar todo.",
      suggested_reply: "Siento mucho por lo que estás pasando. Estoy aquí para lo que necesites.",
    },
    angry: {
      psychology: "Surge por injusticias, expectativas rotas o límites violados.",
      empathy_tip: "Desescala la tensión de inmediato y valida su molestia.",
      suggested_reply:
        "Entiendo perfectamente tu frustración. Vamos a resolver esto de inmediato.",
    },
    surprise: {
      psychology: "Disrupción cognitiva ante un acontecimiento inesperado o revelación.",
      empathy_tip: "Ayuda a asimilar la novedad y aclarar los pasos a seguir.",
      suggested_reply: "¡Guau, eso no me lo esperaba para nada! ¿Cómo lo estás asimilando?",
    },
    fear: {
      psychology: "Anticipación de amenaza, incertidumbre o ansiedad ante lo desconocido.",
      empathy_tip: "Brinda seguridad, hechos claros y tranquilidad.",
      suggested_reply: "Respira hondo, todo saldrá bien. Vamos paso a paso, no estás solo.",
    },
    love: {
      psychology: "Afecto y cariño recíproco.",
      empathy_tip: "Responde con calidez.",
      suggested_reply: "Tus palabras significan mucho para mí. ¡Muchas gracias por tu cariño! ❤️",
    },
    disgust: {
      psychology: "Rechazo hacia una situación.",
      empathy_tip: "Respeta sus límites.",
      suggested_reply: "Comprendo completamente tu rechazo. Establecer límites es lo correcto.",
    },
    neutral: {
      psychology: "Comunicación objetiva, práctica e informativa.",
      empathy_tip: "Mantén un tono claro, directo y profesional.",
      suggested_reply: "Entendido y anotado. Muchas gracias por la información.",
    },
  },
  fr: {
    happy: {
      psychology: "Témoigne de joie, de satisfaction et d'accomplissement personnel.",
      empathy_tip: "Partagez et célébrez son enthousiasme de manière chaleureuse.",
      suggested_reply: "C'est une excellente nouvelle ! Toutes mes félicitations ! 🎉",
    },
    sad: {
      psychology: "Signe de tristesse, de déception ou de fatigue émotionnelle.",
      empathy_tip: "Offrez une écoute bienveillante sans chercher à tout régler de suite.",
      suggested_reply: "Je suis vraiment désolé d'apprendre cela. Je suis de tout cœur avec toi.",
    },
    angry: {
      psychology: "Causé par une frustration, un sentiment d'injustice ou des limites franchies.",
      empathy_tip: "Désamorcez la colère sans vous mettre sur la défensive.",
      suggested_reply: "Je comprends parfaitement votre frustration. Réglons cela rapidement.",
    },
    surprise: {
      psychology: "Réaction spontanée face à une situation totalement inattendue.",
      empathy_tip: "Donnez du temps pour digérer la nouvelle et clarifiez la suite.",
      suggested_reply: "Ouah, c'était totalement inattendu ! Comment tu réagis à ça ?",
    },
    fear: {
      psychology: "Anticipation d'un danger, anxiété face à l'avenir ou insécurité.",
      empathy_tip: "Rassurez avec calme et proposez des repères stables.",
      suggested_reply: "Prends une grande inspiration, tout va bien se passer. Tu n'es pas seul.",
    },
    love: {
      psychology: "Affection profonde et attachement sincère.",
      empathy_tip: "Répondez avec chaleur et reconnaissance.",
      suggested_reply: "Merci infiniment pour vos mots attentionnés et votre affection. ❤️",
    },
    disgust: {
      psychology: "Rejet face à une situation inappropriée ou désagréable.",
      empathy_tip: "Respectez ses limites et son recul légitime.",
      suggested_reply: "Je comprends totalement votre réaction. Poser des limites est sain.",
    },
    neutral: {
      psychology: "Message purement informatif, direct et factuel.",
      empathy_tip: "Restez concis, professionnel et constructif.",
      suggested_reply: "Bien reçu et noté. Merci pour cette information.",
    },
  },
  en: {
    happy: {
      psychology: "Reflects satisfaction, gratitude, accomplishment, or shared social bonding.",
      empathy_tip: "Validate and amplify their excitement. Celebrate their milestone genuinely.",
      suggested_reply: "That is fantastic news! So excited for you, congratulations! 🎉",
    },
    sad: {
      psychology:
        "Indicates feelings of grief, isolation, disappointment, or perceived loss of control.",
      empathy_tip: "Offer non-judgmental presence without rushing to fix everything immediately.",
      suggested_reply:
        "I am really sorry to hear you are dealing with this. Sending strength and here if you need anything.",
    },
    angry: {
      psychology:
        "Driven by perceived injustice, broken expectations, blocked goals, or violated boundaries.",
      empathy_tip:
        "De-escalate immediately. Acknowledge their frustration without being defensive.",
      suggested_reply:
        "I completely understand why this is frustrating. Let's make this right immediately.",
    },
    surprise: {
      psychology: "Cognitive schema disruption following an unexpected event or revelation.",
      empathy_tip: "Help anchor the surprise and clarify next steps or consequences.",
      suggested_reply: "Wow, that really came out of nowhere! How are you processing it so far?",
    },
    fear: {
      psychology: "Anticipation of threat, uncertainty, vulnerability, or impending loss.",
      empathy_tip: "Provide grounding assurance, transparency, and safety cues.",
      suggested_reply:
        "Take a deep breath — we will get through this step by step. You are not alone in this.",
    },
    love: {
      psychology: "Attachment, deep trust, warmth, reciprocal intimacy, and emotional safety.",
      empathy_tip: "Express warmth in return, reciprocate appreciation, and honor the relational bond.",
      suggested_reply: "Your words mean the world to me. Thank you for your kindness, love, and constant care. ❤️",
    },
    disgust: {
      psychology: "Protective recoil from physical contaminants or moral degradation and hypocrisy.",
      empathy_tip: "Validate boundary setting and support physical or emotional distance from the trigger.",
      suggested_reply: "I completely understand your revulsion. Setting clear boundaries is 100% the right call.",
    },
    neutral: {
      psychology: "Informational, pragmatic, and objective communication without affective charge.",
      empathy_tip: "Keep the dialogue clear, efficient, and direct.",
      suggested_reply: "Understood, noted. Thank you for the update.",
    },
  },
};

export function computeDefaultGuidance(
  emotion: Emotion,
  lang?: string | undefined,
): EmotionalGuidance {
  const normLang = (lang || "en").toLowerCase();
  return (
    GUIDANCE_BY_LANGUAGE[normLang]?.[emotion] ?? GUIDANCE_BY_LANGUAGE.en[emotion]
  );
}

/** Characterizes co-occurring affective blends when multiple emotions co-exist. */
export function computeEmotionBlend(
  primary: Emotion,
  secondary: Emotion,
): { label: string; explanation: string } {
  const pair = [primary, secondary].sort().join("+");
  const blendMap: Record<string, { label: string; explanation: string }> = {
    "angry+sad": {
      label: "Bitter Disappointment & Hurt",
      explanation:
        "Underlying hurt or unacknowledged effort expressed with outward frustration, grievance, or resentment.",
    },
    "happy+surprise": {
      label: "Delighted Astonishment",
      explanation:
        "Joy, elation, and positive excitement provoked by an unexpected revelation or pleasant turn of events.",
    },
    "fear+sad": {
      label: "Anxious Despair",
      explanation:
        "Feelings of loss, grief, or isolation combined with apprehension and vulnerability about the future.",
    },
    "angry+fear": {
      label: "Defensive Vigilance",
      explanation:
        "Heightened threat-perception provoking a defensive, guarded, or combative affective state.",
    },
    "happy+sad": {
      label: "Bittersweet Ambivalence",
      explanation:
        "Complex dual emotional valence — fond gratitude or love intertwined with sorrow or nostalgia.",
    },
    "fear+surprise": {
      label: "Alarmed Disorientation",
      explanation:
        "Sudden disruption to expectations accompanied by acute wariness or uncertainty.",
    },
    "angry+happy": {
      label: "Triumphant Aggression / Spite",
      explanation:
        "Assertive satisfaction derived from vindication, confrontation, or overcoming opposition.",
    },
  };

  return (
    blendMap[pair] || {
      label: `Mixed ${primary} & ${secondary}`,
      explanation: `Co-occurring affective signals where ${primary} dominates alongside measurable ${secondary}.`,
    }
  );
}

/** Clamp scores into 0..1, renormalize to sum 1, and pick the argmax emotion. */
export function normalizeReading(text: string, raw: RawReading): EmotionReading {
  const scores = {} as EmotionScores;
  let total = 0;
  for (const emotion of EMOTIONS) {
    const value = raw.scores[emotion];
    const clamped =
      typeof value === "number" && Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
    scores[emotion] = clamped;
    total += clamped;
  }
  if (total <= 0) {
    scores.neutral = 1;
    total = 1;
  }
  for (const emotion of EMOTIONS) {
    scores[emotion] = scores[emotion] / total;
  }

  let best: Emotion = EMOTIONS.includes(raw.emotion as Emotion)
    ? (raw.emotion as Emotion)
    : "neutral";
  for (const emotion of EMOTIONS) {
    if (scores[emotion] > scores[best]) best = emotion;
  }

  const confidence = scores[best];

  // Identify secondary co-occurring emotion for mixed-emotion detection
  const sortedDesc = [...EMOTIONS].sort((a, b) => scores[b] - scores[a]);
  const runnerUp = sortedDesc.find((e) => e !== best) ?? "neutral";
  const runnerUpScore = scores[runnerUp];
  const isMixed =
    runnerUpScore >= 0.18 &&
    runnerUp !== "neutral" &&
    best !== "neutral" &&
    (runnerUpScore / (confidence || 1) >= 0.35 || runnerUpScore >= 0.25);

  const blend = isMixed ? computeEmotionBlend(best, runnerUp) : undefined;

  const sentiment =
    raw.sentiment || (best === "happy" ? "Positive" : best === "neutral" ? "Neutral" : "Negative");
  const sentimentScore =
    raw.sentimentScore !== undefined
      ? raw.sentimentScore
      : best === "happy"
        ? 0.75
        : best === "neutral"
          ? 0.0
          : -0.75;
  const intensityScore = raw.intensityScore ?? Math.round(confidence * 100);
  const intensityLevel =
    raw.intensityLevel ||
    (intensityScore >= 75 ? "High / Strong" : intensityScore >= 45 ? "Moderate" : "Subtle / Low");

  return {
    text,
    emotion: best,
    confidence,
    scores,
    detectedLanguage: raw.detectedLanguage,
    languageName: raw.languageName,
    isMultilingual: raw.isMultilingual,
    translatedText: raw.translatedText,
    sentiment,
    sentimentScore,
    intensityLevel,
    intensityScore,
    dimensions: raw.dimensions || computeDefaultDimensions(best, confidence),
    guidance: raw.guidance || computeDefaultGuidance(best, raw.detectedLanguage),
    tokens: raw.tokens || [],
    isMixedEmotion: raw.isMixedEmotion ?? isMixed,
    secondaryEmotion: raw.secondaryEmotion ?? (isMixed ? runnerUp : undefined),
    secondaryConfidence: raw.secondaryConfidence ?? (isMixed ? runnerUpScore : undefined),
    blendLabel: raw.blendLabel ?? blend?.label,
    blendExplanation: raw.blendExplanation ?? blend?.explanation,
  };
}

export function topEmotionDistribution(readings: EmotionReading[]): Record<Emotion, number> {
  const counts = Object.fromEntries(EMOTIONS.map((e) => [e, 0])) as Record<Emotion, number>;
  for (const reading of readings) counts[reading.emotion] += 1;
  return counts;
}
