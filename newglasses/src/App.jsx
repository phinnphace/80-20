import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, LabelList, Legend } from "recharts";
import cartoonImg from './assets/cartoon.png';


const DARK = "#1c1f2e";
const TEAL = "#0d9488";
const ROSE = "#e11d48";
const AMBER = "#d97706";
const MUTED = "#94a3b8";
const LIGHT_PANEL = "#f1f5f9";

const chartData = [
  { split: "50/50", condA: 64.5, condB: 93 },
  { split: "60/40", condA: 60.2, condB: 95.7 },
  { split: "70/30", condA: 67.4, condB: 94.2 },
  { split: "80/20", condA: 73.5, condB: 94.8 },
  { split: "90/10", condA: 81.6, condB: 100 },
];

/* ───────── Slide Panels ───────── */

function TitleSlide() {
  return (
    <div style={{
      background: DARK, color: "#fff", padding: "80px 48px 60px",
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", minHeight: 340, textAlign: "center",
    }}>
      <h1 style={{ color: "#fff", fontSize: 36, fontWeight: 800, lineHeight: 1.15, margin: 0, letterSpacing: "-0.02em" }}>
        We Don't Need More.<br />We Need to Look.
      </h1>
      <p style={{ color: MUTED, fontSize: 15, marginTop: 28, fontStyle: "italic", margin: "28px 0 0" }}>
        Allocation, audit, and the cost of unexamined defaults
      </p>
      <p style={{ color: "#64748b", fontSize: 13, marginTop: 20 }}>
        Phinn Markson &nbsp;|&nbsp; July 2026
      </p>
    </div>
  );
}

function CartoonSlide() {
  return (
    <div style={{
      background: "#000", color: "#fff", padding: "24px",
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", minHeight: 300,
    }}>
      <img
        src={cartoonImg}
        alt="A child reading a toy catalog while a parent points outside and says: Be grateful for what you have."
        style={{
          maxWidth: "100%", maxHeight: 400, borderRadius: 4,
          boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
        }}
      />
    </div>
  );
}

function ConversationSlide() {
  return (
    <div style={{
      background: DARK, color: "#fff", padding: "48px",
      display: "flex", gap: 40, alignItems: "flex-start", minHeight: 300,
    }}>
      <div style={{ flex: 1 }}>
        <p style={{ color: MUTED, fontSize: 13, margin: "0 0 16px" }}>The current conversation:</p>
        {["More data centers.", "More energy.", "More GPUs.", "More data.", "More training.", "More."].map((t, i) => (
          <p key={i} style={{ fontSize: 22, fontWeight: 700, margin: "6px 0", lineHeight: 1.3 }}>{t}</p>
        ))}
      </div>
      <div style={{ flex: 0.7, paddingTop: 32 }}>
        <p style={{ color: MUTED, fontSize: 15, lineHeight: 1.6 }}>
          The assumption underneath:<br />
          the problem is scarcity,<br />
          and the solution is scale.
        </p>
      </div>
    </div>
  );
}

function AllocationSlide() {
  return (
    <div style={{
      background: DARK, color: "#fff", padding: "64px 48px",
      textAlign: "center", minHeight: 300,
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    }}>
      <h2 style={{ color: "#fff", fontSize: 32, fontWeight: 800, margin: 0 }}>
        The problem is not scarcity.<br />It's allocation.
      </h2>
      <div style={{ marginTop: 32 }}>
        <p style={{ color: MUTED, fontSize: 15, lineHeight: 1.7, margin: 0 }}>
          We have plenty.<br />
          We just keep reaching for more instead of<br />
          examining what's here.
        </p>
      </div>
    </div>
  );
}

function CNNSlide() {
  const cardStyle = (accent) => ({
    background: "#f1f5f9", borderRadius: 12, padding: "24px 28px", flex: 1,
    boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
  });
  return (
    <div style={{ background: "#fff", padding: "40px 48px", minHeight: 320 }}>
      <h2 style={{ fontSize: 28, fontWeight: 800, margin: "0 0 16px", color: "#1a1a2e" }}>
        One small example.
      </h2>
      <p style={{ color: "#475569", fontSize: 14, margin: "0 0 4px" }}>
        Two identical CNNs. 93K parameters. Chinese character recognition.
      </p>
      <p style={{ color: "#475569", fontSize: 14, margin: "0 0 28px" }}>
        Same architecture, same seed, same data. Standard 80/20 split.
      </p>
      <div style={{ display: "flex", gap: 20, marginBottom: 28 }}>
        <div style={cardStyle(ROSE)}>
          <p style={{ fontSize: 13, fontWeight: 700, color: "#1a1a2e", margin: "0 0 4px" }}>Condition A: Isolated</p>
          <p style={{ fontSize: 28, fontWeight: 800, color: ROSE, margin: "0 0 2px", lineHeight: 1.1 }}>
            61.2% internal<br />validation
          </p>
          <p style={{ fontSize: 12, color: TEAL, margin: "4px 0 0", fontWeight: 600 }}>
            Transfers to calligraphy: 100%
          </p>
        </div>
        <div style={cardStyle(TEAL)}>
          <p style={{ fontSize: 13, fontWeight: 700, color: "#1a1a2e", margin: "0 0 4px" }}>Condition B: Contextual</p>
          <p style={{ fontSize: 28, fontWeight: 800, color: TEAL, margin: "0 0 2px", lineHeight: 1.1 }}>
            91.4% internal<br />validation
          </p>
          <p style={{ fontSize: 12, color: ROSE, margin: "4px 0 0", fontWeight: 600 }}>
            Transfers to calligraphy: 5.3%
          </p>
        </div>
      </div>
      <p style={{ color: MUTED, fontSize: 14, fontStyle: "italic", margin: 0 }}>
        Internal validation told one story. The world told another.
      </p>
    </div>
  );
}

function ChartSlide() {
  return (
    <div style={{ background: "#fff", padding: "40px 28px 32px 28px", minHeight: 360 }}>
      <h2 style={{ fontSize: 28, fontWeight: 800, margin: "0 0 8px", color: "#1a1a2e", paddingLeft: 20 }}>
        Then I tested the default.
      </h2>
      <p style={{ color: MUTED, fontSize: 13, margin: "0 0 20px", paddingLeft: 20 }}>
        Same experiment. Five split ratios. Same seed, same hyperparameters.
      </p>
      <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
        <div style={{ flex: 1, height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} barGap={2} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="split" tick={{ fontSize: 11, fill: "#64748b" }} />
              <YAxis domain={[0, 105]} tick={{ fontSize: 11, fill: "#64748b" }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="condA" name="Condition A" fill={ROSE} radius={[3, 3, 0, 0]}>
                <LabelList dataKey="condA" position="top" style={{ fontSize: 10, fill: "#64748b" }} />
              </Bar>
              <Bar dataKey="condB" name="Condition B" fill={TEAL} radius={[3, 3, 0, 0]}>
                <LabelList dataKey="condB" position="top" style={{ fontSize: 10, fill: "#64748b" }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={{
          background: LIGHT_PANEL, borderRadius: 8, padding: "20px 24px",
          minWidth: 140, textAlign: "center",
        }}>
          <p style={{ fontSize: 32, fontWeight: 800, color: ROSE, margin: "0 0 4px" }}>21.4 pt</p>
          <p style={{ fontSize: 12, color: "#475569", margin: "0 0 20px" }}>swing in Condition A<br />across splits</p>
          <p style={{ fontSize: 32, fontWeight: 800, color: TEAL, margin: "0 0 4px" }}>7.0 pt</p>
          <p style={{ fontSize: 12, color: "#475569", margin: 0 }}>swing in Condition B<br />across splits</p>
        </div>
      </div>
      <p style={{ textAlign: "center", fontWeight: 700, fontStyle: "italic", color: "#1a1a2e", fontSize: 14, marginTop: 16 }}>
        Nobody needed more data.<br />The waste was in the assumption.
      </p>
    </div>
  );
}

function ThreePapersSlide() {
  const findingStyle = (bg) => ({
    background: bg, borderRadius: 6, padding: "16px 24px", marginBottom: 12,
  });
  return (
    <div style={{ background: DARK, color: "#fff", padding: "44px 48px", minHeight: 320 }}>
      <h2 style={{ color: "#fff", fontSize: 28, fontWeight: 800, margin: "0 0 24px" }}>
        Three papers. One dataset.
      </h2>
      <div style={findingStyle("rgba(255,255,255,0.06)")}>
        <p style={{ fontSize: 12, color: MUTED, margin: "0 0 6px" }}>80/20, internal validation only</p>
        <p style={{ fontSize: 15, fontStyle: "italic", fontWeight: 600, margin: 0, color: "#e2e8f0" }}>
          "Contextual training improves accuracy by 21 points."
        </p>
      </div>
      <div style={findingStyle("rgba(255,255,255,0.06)")}>
        <p style={{ fontSize: 12, color: MUTED, margin: "0 0 6px" }}>70/30, CASIA transfer</p>
        <p style={{ fontSize: 15, fontStyle: "italic", fontWeight: 600, margin: 0, color: "#e2e8f0" }}>
          "Contextual training transfers perfectly (100%) while isolated training fails (16.9%)."
        </p>
      </div>
      <div style={findingStyle("rgba(255,255,255,0.06)")}>
        <p style={{ fontSize: 12, color: MUTED, margin: "0 0 6px" }}>50/50, CalliBench transfer</p>
        <p style={{ fontSize: 15, fontStyle: "italic", fontWeight: 600, margin: 0, color: "#e2e8f0" }}>
          "Isolated training is the only path to robust generalization."
        </p>
      </div>
      <p style={{ color: ROSE, fontSize: 13, fontWeight: 600, marginTop: 16, marginBottom: 0 }}>
        Same data. Same model. The split ratio determined which truth you'd find.
      </p>
    </div>
  );
}

function LoopSlide() {
  const boxStyle = {
    background: LIGHT_PANEL, borderRadius: 8, padding: "16px 20px", width: 220,
  };
  const labelStyle = { fontSize: 14, fontWeight: 700, color: "#1a1a2e", margin: "0 0 6px" };
  const descStyle = { fontSize: 11, color: "#64748b", margin: 0, lineHeight: 1.4 };
  const arrowStyle = { fontSize: 24, color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center" };

  return (
    <div style={{ background: "#fff", padding: "40px 48px", minHeight: 340 }}>
      <h2 style={{ fontSize: 28, fontWeight: 800, margin: "0 0 28px", color: "#1a1a2e" }}>The Loop</h2>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={boxStyle}>
            <p style={labelStyle}>Methodology</p>
            <p style={descStyle}>Preprocess, 80/20 split, internal validation</p>
          </div>
          <div style={arrowStyle}>→</div>
          <div style={boxStyle}>
            <p style={labelStyle}>Conclusions</p>
            <p style={descStyle}>"Results" that are tautologies of the method</p>
          </div>
        </div>
        <div style={{ display: "flex", width: 520, justifyContent: "space-between", padding: "0 20px" }}>
          <div style={arrowStyle}>↑</div>
          <div style={arrowStyle}>↓</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={boxStyle}>
            <p style={labelStyle}>Demand for More</p>
            <p style={descStyle}>More compute, more data, more infrastructure</p>
          </div>
          <div style={arrowStyle}>←</div>
          <div style={boxStyle}>
            <p style={labelStyle}>Policy / Allocation</p>
            <p style={descStyle}>Decisions downstream of those conclusions</p>
          </div>
        </div>
      </div>
      <p style={{
        textAlign: "center", fontSize: 12, color: MUTED, fontStyle: "italic",
        marginTop: 20, lineHeight: 1.5, marginBottom: 0,
      }}>
        The methodology produces the conclusions. The conclusions justify the methodology.<br />
        The loop sustains itself. Nobody is acting in bad faith. That's what makes it sticky.
      </p>
    </div>
  );
}

function PortfolioSlide() {
  const tools = [
    { name: "geoDeltaAudit", platform: "R / CRAN", color: "#1a1a2e",
      desc: "Audits how variables change when data crosses administrative boundaries. The labels stay. The quality doesn't." },
    { name: "shellgame", platform: "R / CRAN", color: "#1a1a2e",
      desc: "Shows the degradation is agnostic to variable and tool. Population, income, R, Python — same shell game." },
    { name: "nocando", platform: "Python / PyPI", color: "#1a1a2e",
      desc: "Catches code/hardware mismatches before you idle the GPU. Audit before you burn." },
    { name: "Split ratio audit", platform: "This talk", color: ROSE,
      desc: "Tests the methodology itself. The 80/20 default determined which conclusion you'd find." },
  ];
  return (
    <div style={{ background: "#fff", padding: "40px 48px", minHeight: 320 }}>
      <h2 style={{ fontSize: 28, fontWeight: 800, margin: "0 0 24px", color: "#1a1a2e" }}>
        Same principle. Different layers.
      </h2>
      {tools.map((t, i) => (
        <div key={i} style={{
          display: "flex", gap: 20, padding: "14px 20px",
          background: i % 2 === 0 ? "#f8fafc" : "#fff",
          borderRadius: 6, marginBottom: 4, alignItems: "flex-start",
        }}>
          <div style={{ minWidth: 140 }}>
            <p style={{ fontSize: 14, fontWeight: 700, color: t.color, margin: "0 0 2px" }}>{t.name}</p>
            <p style={{ fontSize: 11, color: MUTED, margin: 0 }}>{t.platform}</p>
          </div>
          <p style={{ fontSize: 13, color: "#475569", margin: 0, lineHeight: 1.5 }}>{t.desc}</p>
        </div>
      ))}
      <p style={{
        textAlign: "center", fontSize: 13, fontStyle: "italic", color: MUTED,
        marginTop: 20, marginBottom: 0, lineHeight: 1.5,
      }}>
        Each audit costs essentially nothing. Nobody gives anything up.<br />
        The only thing eliminated is waste.
      </p>
    </div>
  );
}

function QuestionSlide() {
  return (
    <div style={{
      background: DARK, color: "#fff", padding: "64px 48px",
      textAlign: "center", minHeight: 300,
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    }}>
      <h2 style={{ color: "#fff", fontSize: 28, fontWeight: 800, margin: 0, lineHeight: 1.35 }}>
        How much of the demand for<br />
        "more" is downstream of<br />
        conclusions that were artifacts<br />
        of unexamined defaults?
      </h2>
      <div style={{ marginTop: 28 }}>
        <p style={{ color: MUTED, fontSize: 14, margin: 0, lineHeight: 1.6 }}>
          We can't quantify the total. That's part of the problem.<br />
          The waste is invisible because it looks like results.
        </p>
      </div>
    </div>
  );
}

function AskSlide() {
  const items = [
    { label: "Test the split ratio.", detail: "Five runs. Same seed. One afternoon.", color: ROSE },
    { label: "Transfer test.", detail: "Internal validation is a checkpoint, not a conclusion.", color: AMBER },
    { label: "Audit your inputs.", detail: "Check your data, your transforms, your hardware, before you occupy them.", color: TEAL },
  ];
  return (
    <div style={{ background: "#fff", padding: "40px 48px", minHeight: 300 }}>
      <h2 style={{ fontSize: 28, fontWeight: 800, margin: "0 0 24px", color: "#1a1a2e" }}>
        The ask is small.
      </h2>
      {items.map((it, i) => (
        <div key={i} style={{
          background: "#f8fafc", borderRadius: 8, padding: "18px 24px", marginBottom: 12,
        }}>
          <p style={{ fontSize: 16, fontWeight: 700, color: it.color, margin: "0 0 4px" }}>{it.label}</p>
          <p style={{ fontSize: 13, color: "#64748b", margin: 0 }}>{it.detail}</p>
        </div>
      ))}
      <p style={{
        textAlign: "center", fontSize: 13, fontStyle: "italic", color: MUTED,
        marginTop: 16, marginBottom: 0,
      }}>
        These are not heroic acts. They're checking your mirrors before you change lanes.
      </p>
    </div>
  );
}

function ClosingSlide() {
  return (
    <div style={{
      background: DARK, color: "#fff", padding: "64px 48px",
      textAlign: "center", minHeight: 300,
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    }}>
      <h2 style={{ color: "#fff", fontSize: 28, fontWeight: 800, margin: "0 0 12px" }}>Simple, not easy.</h2>
      <p style={{ color: "#cbd5e1", fontSize: 15, margin: "0 0 32px", lineHeight: 1.6 }}>
        We don't need a breakthrough.<br />
        We need to look at what we already have.
      </p>
      <p style={{ color: "#fff", fontSize: 22, fontWeight: 700, fontStyle: "italic", margin: "0 0 32px", lineHeight: 1.4 }}>
        The unexamined default<br />is not worth inheriting.
      </p>
      <p style={{ color: MUTED, fontSize: 13, margin: 0 }}>
        Phinn Markson<br />
        <span style={{ color: "#60a5fa" }}>markson.2@osu.edu</span> | <span style={{ color: "#60a5fa" }}>dataacorns.com</span>
      </p>
    </div>
  );
}

/* ───────── Notes content ───────── */

const notes = [
  // Slide 1: Title
  `The current conversation about AI and compute goes like this: we need more data centers, more energy, more GPUs, more data, more training, more parameters, more. The assumption underneath is that the problem is scarcity and the solution is scale. I think the assumption is wrong.`,

  // Slide 2: Cartoon
  `I don't know how you grew up but saying things such as "I'm bored", or "I don't know what to do" did not end well. What I failed to recognize as a kid is this thread… We don't have a scarcity problem. We have an allocation problem. We have plenty. We just keep reaching for more instead of examining what's already here.`,

  // Slide 3: The current conversation
  `This is a side quest from a project so if it seems lacking full context I wanted to acknowledge that you are correct. Given the current environment and conversations happening, I do not hear this one being had and I wanted to add this in hopes it would become part of our conversation and beyond.`,

  // Slide 4: Allocation
  `I want to show you what I mean with one small example, and then I want to zoom out.`,

  // Slide 5: One small example
  `No need to dwell on the Chinese characters — they're the vehicle for this story. In fact most of the technical jargon you do not need pay attention to, the "thing" is not the point. The point is: internal validation is a tautology. The model passed its own exam. Of course it did.

I trained two identical convolutional neural networks — tiny ones, 93K parameters each — on Chinese character recognition. Same architecture, same seed, same data. One trained on isolated characters (Condition A), one on characters in natural handwriting context (Condition B). Standard setup, standard 80/20 train/validation split. The contextual model won by 30 points on internal validation: 91.4% vs. 61.2%.

Paper-worthy finding. Publishable. Next project.`,

  // Slide 6: Then I tested the default
  `This is the first receipt. 21 points of swing from a choice nobody tests. So I ran transfer tests — external data the model had never seen — and the story flipped. The isolated model, the one that "lost" internal validation, generalized perfectly to calligraphy (100%). The contextual model, the one that "won," failed almost completely (5.3%). Internal validation told one story. The world told another.

Still not the finding.

The finding came when I did something most papers don't: I tested the split ratio itself. That 80/20 split everyone uses — I ran the same experiment at 50/50, 60/40, 70/30, 80/20, and 90/10. Same seed, same hyperparameters, same data. Only the split changed.

The results:

Condition A swings 21.4 percentage points across splits. Condition B swings only 7. Same data, same model. Only the split changed. And when I ran transfer tests at each split, the conclusions didn't just shift — they contradicted each other:

At 80/20, Condition B's transfer to isolated handwriting (CASIA) was 26.6%. At 60/40, it was 98.3%. The contextual model transfers near-perfectly — our original conclusion that it "doesn't generalize" was an artifact of the split we inherited without testing.`,

  // Slide 7: Three papers
  `You can write three defensible, contradictory papers from this one dataset depending on which split and which test metric you choose:

80/20, internal validation only: "Contextual training improves accuracy by 21 points."

70/30, CASIA transfer: "Contextual training transfers perfectly (100%) while isolated training fails (16.9%)."

50/50, CalliBench transfer: "Isolated training is the only path to robust generalization."

All true. All incomplete. The split ratio — a choice most researchers make without examining it — determined which truth you'd find.

One thing the split can't fix: calligraphy transfer. Condition B never exceeds 15.8% on CalliBench at any split. That's not a split artifact. That's a genuine domain gap. And that distinction matters — the split ratio audit is how you tell artifact from finding. Without it, you can't tell which of your conclusions are real and which are inherited from a default nobody tested. Even this intermediate finding is conclusive of anything, it is only a sign post for further exploration within this study context.`,

  // Slide 8: The Loop
  `We are naming the system so we can get to solutions. The circularity is the feature, not a bug someone introduced.

This is where it gets uncomfortable, because the split ratio isn't an isolated problem. It's one instance of a pattern: methodology mistakes its own preprocessing for discovery.

The standard ML pipeline runs like this: collect data, preprocess it extensively, split it (80/20, because that's what we do), train a model, report internal validation as results. The model passes because it was built to pass — you engineered the conditions for it. That's not science. That's a tautology wearing a lab coat.

But it gets published. And because it got published, the methodology gets repeated. And because the methodology gets repeated, it produces more results of the same kind, which get published, which validate the methodology, which gets repeated. The methodology produces the conclusions. The conclusions justify the methodology. The loop sustains itself.

The downstream effects compound. Conclusions from internally validated models inform policy decisions. Policy decisions drive resource allocation. Resource allocation drives demand for more compute, more data, more infrastructure. And that demand funds the same methodology that produced the conclusions. The loop doesn't just repeat — it scales.

Nobody in this loop is acting in bad faith. That's what makes it sticky. Everyone is following best practices — because best practices are what everyone follows. The 80/20 split is standard because it's standard. Internal validation is accepted because it's accepted. The circularity is the feature, not a bug someone introduced.`,

  // Slide 9: Portfolio
  `I keep building the same thing in different domains because the pattern keeps showing up. I did not intend on this. I need to name it.

Four tools, four layers, one principle. And now the split ratio work. Same principle, different layer. The methodology itself contains unexamined defaults that determine your conclusions before you run the experiment.

Each of these tools does the same thing: looks at what's already there before reaching for more. Each audit costs essentially nothing. Nobody gives anything up. Nobody's workflow degrades. The only thing eliminated is waste — and it's waste composed entirely of moments nobody valued because nobody was watching.`,

  // Slide 10: Question
  null, // No notes — just the slide

  // Slide 11: The ask is small
  `We talk about the AI energy crisis as though it's a supply problem. Build more data centers. Source more power. Scale more infrastructure. The assumption is that the demand is legitimate — that all this compute is producing proportional value, and we simply need more capacity to meet it.

But how much of that demand is downstream of conclusions that were artifacts of unexamined defaults? How many models were trained on data that silently degraded during geographic transformation? How many GPUs idled because the code couldn't address them? How many papers reported results that would flip if the split ratio changed?

I can't quantify the total. Nobody can — that's part of the problem. The waste is invisible because it's composed of things that look like results. A model that reports 91.4% accuracy looks like it works. A paper that gets published looks like it's been validated. A policy informed by that paper looks like it's evidence-based. None of these things are false. They're just not examined.

The ask is small. Test the split ratio — five runs, same seed, one afternoon. Transfer test on external data. Audit your geographic transformations. Check your code against your hardware before you occupy it. These are not heroic acts. They're the equivalent of checking your mirrors before you change lanes. They cost nothing, they catch waste, and the only thing lost is heat that was never going to become work.`,

  // Slide 12: Closing
  `We don't need a breakthrough. We don't need a new framework, a new paradigm, a new model architecture. We need to look at what we already have. That's the turn. The unexamined default is not worth inheriting.

I keep building audit tools not because I think a linter can fix an industry but simply, this is easy.`,
];

const slideComponents = [
  TitleSlide, CartoonSlide, ConversationSlide, AllocationSlide,
  CNNSlide, ChartSlide, ThreePapersSlide, LoopSlide,
  PortfolioSlide, QuestionSlide, AskSlide, ClosingSlide,
];

/* ───────── Main Component ───────── */

export default function NewGlasses() {
  const [mode, setMode] = useState("read"); // "read" or "present"
  const [activeSlide, setActiveSlide] = useState(0);

  const handlePrint = () => window.print();

  return (
    <div style={{ fontFamily: "'Inter', system-ui, -apple-system, sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

        * { box-sizing: border-box; }

        .notes-text {
          font-family: 'Source Serif 4', Georgia, serif;
          font-size: 15.5px;
          line-height: 1.72;
          color: #334155;
          white-space: pre-line;
        }

        .slide-panel {
          border-radius: 6px;
          overflow: hidden;
          box-shadow: 0 1px 4px rgba(0,0,0,0.1), 0 4px 16px rgba(0,0,0,0.06);
        }

        .controls { display: flex; align-items: center; gap: 12px; }

        .control-btn {
          background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px;
          padding: 7px 16px; font-size: 13px; cursor: pointer; color: #475569;
          font-family: Inter, system-ui, sans-serif; font-weight: 500;
          transition: background 0.15s;
        }
        .control-btn:hover { background: #e2e8f0; }
        .control-btn.active { background: #1c1f2e; color: #fff; border-color: #1c1f2e; }

        .slide-unit { margin-bottom: 48px; }

        .nav-dot {
          width: 8px; height: 8px; border-radius: 50%; border: none;
          cursor: pointer; padding: 0; transition: all 0.15s;
        }

        .present-nav {
          display: flex; align-items: center; gap: 8px;
          justify-content: center; margin-top: 16px;
        }
        .present-nav button {
          background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px;
          padding: 6px 14px; font-size: 13px; cursor: pointer; color: #475569;
          font-family: Inter, system-ui, sans-serif;
        }
        .present-nav button:hover { background: #e2e8f0; }
        .present-nav button:disabled { opacity: 0.4; cursor: default; }

        @media print {
          .controls, .present-nav, .mode-bar { display: none !important; }
          .slide-unit {
            page-break-before: always;
            margin-bottom: 0;
          }
          .slide-unit:first-child { page-break-before: auto; }
          .slide-panel {
            page-break-inside: avoid;
            box-shadow: none;
            border: 1px solid #ddd;
          }
          .slide-notes {
            page-break-inside: auto;
          }
          body { margin: 0; }
        }
      `}</style>

      {/* ── Header bar ── */}
      <div className="mode-bar" style={{
        position: "sticky", top: 0, zIndex: 100,
        background: "rgba(255,255,255,0.92)", backdropFilter: "blur(8px)",
        borderBottom: "1px solid #e2e8f0",
        padding: "10px 24px", display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#1c1f2e", letterSpacing: "0.02em" }}>
            NEW GLASSES
          </span>
          <span style={{ fontSize: 12, color: MUTED }}>
            {mode === "read" ? "Reading mode" : `Slide ${activeSlide + 1} of ${slideComponents.length}`}
          </span>
        </div>
        <div className="controls">
          <button
            className={`control-btn ${mode === "read" ? "active" : ""}`}
            onClick={() => setMode("read")}
          >
            Read
          </button>
          <button
            className={`control-btn ${mode === "present" ? "active" : ""}`}
            onClick={() => setMode("present")}
          >
            Present
          </button>
          <button className="control-btn" onClick={handlePrint}>
            ⎙ Print / PDF
          </button>
        </div>
      </div>

      {/* ── Content ── */}
      <div style={{ maxWidth: 780, margin: "0 auto", padding: "32px 24px 64px" }}>

        {mode === "read" ? (
          /* ── Reading mode: all slides ── */
          slideComponents.map((SlideComp, i) => (
            <section key={i} className="slide-unit">
              <div style={{
                display: "flex", alignItems: "center", gap: 8, marginBottom: 12,
              }}>
                <span style={{
                  fontSize: 11, fontWeight: 600, color: MUTED,
                  background: "#f1f5f9", borderRadius: 4, padding: "2px 8px",
                  letterSpacing: "0.04em",
                }}>
                  {String(i + 1).padStart(2, "0")}
                </span>
              </div>
              <div className="slide-panel">
                <SlideComp />
              </div>
              {notes[i] && (
                <div className="slide-notes" style={{ padding: "24px 8px 0" }}>
                  <div className="notes-text">{notes[i]}</div>
                </div>
              )}
            </section>
          ))
        ) : (
          /* ── Presentation mode: one slide ── */
          <div>
            <section className="slide-unit" style={{ marginBottom: 24 }}>
              <div className="slide-panel">
                {(() => { const C = slideComponents[activeSlide]; return <C />; })()}
              </div>
              {notes[activeSlide] && (
                <div className="slide-notes" style={{ padding: "24px 8px 0" }}>
                  <div className="notes-text">{notes[activeSlide]}</div>
                </div>
              )}
            </section>
            <div className="present-nav">
              <button
                disabled={activeSlide === 0}
                onClick={() => setActiveSlide(s => Math.max(0, s - 1))}
              >
                ← Prev
              </button>
              <div style={{ display: "flex", gap: 6, alignItems: "center", margin: "0 8px" }}>
                {slideComponents.map((_, i) => (
                  <button
                    key={i}
                    className="nav-dot"
                    style={{
                      background: i === activeSlide ? "#1c1f2e" : "#cbd5e1",
                      transform: i === activeSlide ? "scale(1.3)" : "scale(1)",
                    }}
                    onClick={() => setActiveSlide(i)}
                    title={`Slide ${i + 1}`}
                  />
                ))}
              </div>
              <button
                disabled={activeSlide === slideComponents.length - 1}
                onClick={() => setActiveSlide(s => Math.min(slideComponents.length - 1, s + 1))}
              >
                Next →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
