# BeyondWord Emotion Annotation Guidelines (Gold Standard Protocol v1.0)

These guidelines govern human and algorithmic emotion annotation for BeyondWord dataset curation and evaluation. All annotators must follow these rules blindly without access to model predictions or unverified raw source labels.

---

## 1. Emotion Taxonomy (8 Classes)

| Emotion | Core Affective Definition | Positive Identification Signals | Negative Boundaries (What it is NOT) |
|---|---|---|---|
| **Joy** | Feelings of happiness, pleasure, delight, relief, triumphant success, or celebration. | *"We won the championship!"*, *"Finally got promoted!"*, *"I am so happy today."* | NOT mere factual statement of positive business events (*"Stock market rose 1%"* is Neutral). |
| **Sadness** | Feelings of sorrow, grief, loss, loneliness, hopelessness, dejection, or de-energized defeat. | *"I feel so lonely and empty."*, *"Lost someone very dear."*, *"Umeed toot gayi."* | NOT mild disinterest or constructive criticism. |
| **Anger** | Feelings of annoyance, frustration, irritation, outrage, fury, or resentment toward an agent/situation. | *"This delay is ridiculous, fix it!"*, *"You lied to my face."*, *"Dimag kharab mat karo."* | NOT quiet despair (Sadness) or physical nausea (Disgust). |
| **Fear** | Anticipation of threat, danger, anxiety, panic, terror, dread, or impending doom. | *"My hands are trembling in the dark."*, *"Terrified of exam results."*, *"Darr lag raha hai."* | NOT mild cautious scheduling or routine risk assessments. |
| **Surprise** | Astonishment, bewilderment, unexpected revelation, or shock at an unforeseen event. | *"Wait, what?! I never expected this!"*, *"Achanak se shock laga!"* | **CRITICAL:** Plain questions (`?`) or rhetorical comments are **NOT** surprise (*"Were minor children part of the divorce?"* is Neutral). |
| **Love** | Intense interpersonal affection, adoration, deep platonic/romantic connection, warmth, or devotion. | *"I love you forever, soulmate."*, *"So grateful for my mom's unconditional care."* | NOT casual approval (*"I love tacos"* $\rightarrow$ Joy). |
| **Disgust** | Physical revulsion, nausea, moral aversion, sickening uncleanliness, or visceral contempt. | *"The rotten meat was covered in slime."*, *"Vile and sickening behavior."* | NOT simple policy disagreement (*"Terrible idea"* $\rightarrow$ Disgust/Anger only if visceral revulsion is expressed). |
| **Neutral** | Factual, declarative, informational, procedural, scheduling, or emotionally devoid text. | *"Meeting is at 2:00 PM."*, *"No, they are not free."*, *"Document submitted."* | **EXCLUSIVITY RULE:** Must not coexist with strong positive or negative affective labels. |

---

## 2. Special Contextual Protocols

### Protocol A: Sarcasm & Irony
- **Rule:** When positive vocabulary (*"great"*, *"superb"*, *"pure bliss"*, *"kya shandaar"*) is paired with an unambiguously negative situation (*flight delayed, coffee spilled on laptop, stuck in traffic*), **do not annotate as Joy or Surprise**.
- **Label Assignment:** Primary label is **Anger** (or **Disgust** for vile situations), with an optional secondary tag of **Sarcasm/Irony**.

### Protocol B: Mixed Emotions & Ambivalence
- Sentences often convey multiple simultaneous emotional currents.
- In multi-label annotation, assign:
  - **Primary Emotion ($y_1$):** The governing or final resolution emotion.
  - **Secondary Emotion ($y_2$):** The co-occurring emotion with substantial emotional presence.
- **Canonical Compound Pairs:**
  - `Joy + Sadness`: Bittersweet (e.g., *"Graduation is exciting, but I'll miss everyone."*)
  - `Anger + Sadness`: Hurt Resentment (e.g., *"Gussa bhi aa raha hai aur rona bhi."*)
  - `Joy + Fear`: Apprehensive Thrill (e.g., *"Excited for college, but terrified of leaving home."*)
  - `Love + Sadness`: Longing / Heartache (e.g., *"Love my puppy, but exhausted by continuous crying."*)

### Protocol C: Contrast & Concession ("But", "Par", "Lekin", "Fir Bhi")
- In compound sentences joined by contrastive conjunctions, the **resolution clause (typically after the conjunction)** governs the primary emotion.
  - *"Worked hard all year (Joy/Effort), BUT failed the exam (Sadness)"* $\rightarrow$ **Primary: Sadness**.
  - *"Faced tough obstacles (Fear/Sadness), BUT we finally triumphed! (Joy)"* $\rightarrow$ **Primary: Joy**.

### Protocol D: Negation Resolution
- Negated emotion terms reverse or extinguish the emotion:
  - *"Not happy"* $\rightarrow$ **Sadness / Neutral** (never Joy).
  - *"Daant nahi padi, mazaak kar raha tha"* $\rightarrow$ **Neutral / Relief** (never Sadness).
  - *"Don't feel angry anymore"* $\rightarrow$ **Neutral** (never Anger).
  - *"Not bad"* $\rightarrow$ **Neutral / Mild Joy** (Litotes).

### Protocol E: The Neutral Exclusivity Constraint
- `Neutral` represents the absence of emotional valence ($V \approx 0, A \approx 0$).
- Therefore, in multi-label representations:
  $$\text{If } y_{\text{Neutral}} = 1 \implies \forall k \ne \text{Neutral}, \; y_k = 0$$
- If any emotion (Joy, Sadness, Anger, Fear, etc.) is verified active, `Neutral` **must be 0**.

---

## 3. Disagreement Resolution & Adjudication Protocol

1. **Independent Blind Passes:** Annotator 1 and Annotator 2 independently annotate all texts without viewing each other's work or model inferences.
2. **Inter-Annotator Agreement:** Compute Cohen's Kappa ($\kappa$) across the full label matrix. Target: $\kappa \ge 0.75$.
3. **Adjudication Meeting:** Any item with disagreement ($\text{Annotator 1} \ne \text{Annotator 2}$) is brought to adjudication. A lead adjudicator applies the decision trees above to record the final consensus ground truth label.

