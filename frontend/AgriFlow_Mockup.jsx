import { useState, useEffect, useRef } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

// ============================================================
// AGRIFLOW — Food Supply Chain Intelligence Agent
// MUIDSI Hackathon 2026 — Interactive Demo Mockup
// ============================================================

// --- Demo Data ---
const MISSOURI_FOOD_INSECURITY = [
  { county: "Wayne", rate: 23.1, pop: 12000, desert: true },
  { county: "Pemiscot", rate: 22.0, pop: 16000, desert: true },
  { county: "Oregon", rate: 21.0, pop: 10000, desert: true },
  { county: "Ozark", rate: 21.0, pop: 9000, desert: true },
  { county: "Howell", rate: 19.8, pop: 39000, desert: true },
  { county: "Barton", rate: 19.5, pop: 11000, desert: true },
  { county: "Texas", rate: 19.4, pop: 25000, desert: true },
  { county: "Shannon", rate: 18.9, pop: 8000, desert: false },
  { county: "Ripley", rate: 18.5, pop: 13000, desert: false },
  { county: "Carter", rate: 18.2, pop: 6000, desert: false },
];

const CORN_PRODUCTION_DATA = [
  { year: "2018", yield: 162, production: 545 },
  { year: "2019", yield: 148, production: 498 },
  { year: "2020", yield: 171, production: 575 },
  { year: "2021", yield: 158, production: 531 },
  { year: "2022", yield: 142, production: 477 },
  { year: "2023", yield: 165, production: 554 },
  { year: "2024", yield: 153, production: 514 },
];

const RISK_COUNTIES = [
  { county: "Wayne, MO", risk: 94, insecurity: 23.1, cornDep: "High", pop: "12,000", action: "Priority emergency distribution" },
  { county: "Pemiscot, MO", risk: 91, insecurity: 22.0, cornDep: "Very High", pop: "16,000", action: "Activate mobile food pantry" },
  { county: "Oregon, MO", risk: 87, insecurity: 21.0, cornDep: "Medium", pop: "10,000", action: "Redirect SNAP resources" },
  { county: "Ozark, MO", risk: 85, insecurity: 21.0, cornDep: "Medium", pop: "9,000", action: "Increase TEFAP allocation" },
  { county: "Howell, MO", risk: 82, insecurity: 19.8, cornDep: "High", pop: "39,000", action: "Partner with food banks" },
  { county: "Barton, MO", risk: 78, insecurity: 19.5, cornDep: "Low", pop: "11,000", action: "Monitor & prepare" },
];

const DEMO_QUERIES = [
  {
    id: 1,
    label: "Food Insecurity Hotspots",
    query: "Show me the top 10 counties in Missouri with the highest food insecurity rates.",
    steps: [
      { tool: "database", label: "Querying Food Environment Atlas", detail: "SELECT county, food_insecurity_rate, population, is_food_desert FROM food_environment WHERE state = 'MO' ORDER BY food_insecurity_rate DESC LIMIT 10", duration: 1200 },
      { tool: "analyze", label: "Analyzing food desert overlap", detail: "Cross-referencing with Food Access Research Atlas for desert classification", duration: 800 },
      { tool: "visualize", label: "Generating visualization", detail: "Creating ranked bar chart with food desert indicators", duration: 600 },
    ],
    resultType: "query1",
  },
  {
    id: 2,
    label: "Corn Dependency Analysis",
    query: "Which food desert communities in the Midwest are most dependent on corn production?",
    steps: [
      { tool: "api", label: "Calling NASS Quick Stats API", detail: "GET /api/api_GET?commodity_desc=CORN&state_alpha=MO&statisticcat_desc=PRODUCTION&year__GE=2018", duration: 1500 },
      { tool: "database", label: "Querying Food Access Atlas", detail: "Retrieving food desert census tracts for Midwest states (MO, IL, IA, KS, NE)", duration: 1000 },
      { tool: "analyze", label: "Computing dependency scores", detail: "Correlating county-level corn production with food desert prevalence and local employment data", duration: 1200 },
      { tool: "visualize", label: "Building cross-analysis view", detail: "Generating production trend chart with dependency overlay", duration: 700 },
    ],
    resultType: "query2",
  },
  {
    id: 3,
    label: "Drought Scenario Planning",
    query: "If corn yields drop 20% due to drought, which communities are most at risk and where should emergency food resources be redirected?",
    steps: [
      { tool: "api", label: "Fetching baseline crop data from NASS", detail: "GET /api/api_GET?commodity_desc=CORN&state_alpha=MO&statisticcat_desc=YIELD&year=2024", duration: 1200 },
      { tool: "model", label: "Running supply disruption model", detail: "Simulating 20% yield reduction → cascading effects on local food prices, availability, and distribution capacity", duration: 2000 },
      { tool: "database", label: "Loading vulnerability indicators", detail: "Pulling food insecurity rates, SNAP participation, vehicle access, poverty rates from Food Environment Atlas", duration: 900 },
      { tool: "analyze", label: "Computing composite risk scores", detail: "risk_score = 0.3×food_insecurity + 0.25×corn_dependency + 0.2×poverty_rate + 0.15×low_vehicle_access + 0.1×population_density", duration: 1500 },
      { tool: "recommend", label: "Generating action recommendations", detail: "Prioritizing counties by risk score, matching with available distribution infrastructure and emergency programs", duration: 1100 },
    ],
    resultType: "query3",
  },
];

// --- Icon Components ---
const DatabaseIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
  </svg>
);
const ApiIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>
  </svg>
);
const AnalyzeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/>
  </svg>
);
const VisualizeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/>
  </svg>
);
const ModelIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3"/><path d="M12 1v4"/><path d="M12 19v4"/><path d="M4.22 4.22l2.83 2.83"/><path d="M16.95 16.95l2.83 2.83"/><path d="M1 12h4"/><path d="M19 12h4"/><path d="M4.22 19.78l2.83-2.83"/><path d="M16.95 7.05l2.83-2.83"/>
  </svg>
);
const RecommendIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>
  </svg>
);
const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>
  </svg>
);
const LeafIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 19 2c1 2 2 4.5 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
  </svg>
);

const toolIcons = {
  database: <DatabaseIcon />,
  api: <ApiIcon />,
  analyze: <AnalyzeIcon />,
  visualize: <VisualizeIcon />,
  model: <ModelIcon />,
  recommend: <RecommendIcon />,
};

const toolColors = {
  database: { bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.3)", text: "#60a5fa" },
  api: { bg: "rgba(251,191,36,0.12)", border: "rgba(251,191,36,0.3)", text: "#fbbf24" },
  analyze: { bg: "rgba(168,85,247,0.12)", border: "rgba(168,85,247,0.3)", text: "#c084fc" },
  visualize: { bg: "rgba(34,197,94,0.12)", border: "rgba(34,197,94,0.3)", text: "#4ade80" },
  model: { bg: "rgba(244,114,182,0.12)", border: "rgba(244,114,182,0.3)", text: "#f472b6" },
  recommend: { bg: "rgba(45,212,191,0.12)", border: "rgba(45,212,191,0.3)", text: "#2dd4bf" },
};

// --- Custom Tooltip ---
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1a1f2e", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "10px 14px", boxShadow: "0 8px 32px rgba(0,0,0,0.4)" }}>
      <p style={{ color: "#94a3b8", fontSize: 11, margin: 0, marginBottom: 4 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontSize: 13, fontWeight: 600, margin: 0 }}>
          {p.name}: {p.value}{p.name === "rate" || p.name === "insecurity" ? "%" : p.name === "yield" ? " bu/acre" : p.name === "production" ? "M bu" : ""}
        </p>
      ))}
    </div>
  );
};

// --- Result Components ---
const Query1Result = () => (
  <div>
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#4ade80" }} />
      <span style={{ color: "#e2e8f0", fontSize: 14, fontWeight: 600 }}>Top 10 Missouri Counties by Food Insecurity Rate</span>
    </div>
    <div style={{ height: 260, marginBottom: 16 }}>
      <ResponsiveContainer>
        <BarChart data={MISSOURI_FOOD_INSECURITY} layout="vertical" margin={{ left: 10, right: 30, top: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
          <XAxis type="number" domain={[0, 28]} tick={{ fill: "#64748b", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.1)" }} tickLine={false} unit="%" />
          <YAxis dataKey="county" type="category" width={80} tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="rate" radius={[0, 4, 4, 0]} barSize={18}>
            {MISSOURI_FOOD_INSECURITY.map((entry, i) => (
              <Cell key={i} fill={entry.desert ? "#ef4444" : "#3b82f6"} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
    <div style={{ display: "flex", gap: 16, marginBottom: 16, paddingLeft: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ width: 12, height: 12, borderRadius: 3, background: "#ef4444", opacity: 0.85 }} />
        <span style={{ color: "#94a3b8", fontSize: 11 }}>Food Desert</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ width: 12, height: 12, borderRadius: 3, background: "#3b82f6", opacity: 0.85 }} />
        <span style={{ color: "#94a3b8", fontSize: 11 }}>Non-Desert</span>
      </div>
    </div>
    <div style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, padding: "12px 16px" }}>
      <p style={{ color: "#fca5a5", fontSize: 12, fontWeight: 600, margin: 0, marginBottom: 4 }}>Key Finding</p>
      <p style={{ color: "#cbd5e1", fontSize: 12.5, margin: 0, lineHeight: 1.5 }}>
        7 of the top 10 food-insecure counties are classified as food deserts. These counties are concentrated in Southeast Missouri and the Ozarks, where poverty rates exceed 20% and supermarket access often requires traveling 10+ miles. Wayne County leads at 23.1% — the highest food insecurity rate in Missouri, per Feeding America's 2023 Map the Meal Gap data. Combined population affected: ~149,000 residents.
      </p>
    </div>
  </div>
);

const Query2Result = () => (
  <div>
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#fbbf24" }} />
      <span style={{ color: "#e2e8f0", fontSize: 14, fontWeight: 600 }}>Missouri Corn Production Trends (2018–2024)</span>
    </div>
    <div style={{ height: 220, marginBottom: 16 }}>
      <ResponsiveContainer>
        <LineChart data={CORN_PRODUCTION_DATA} margin={{ left: 5, right: 30, top: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="year" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.1)" }} tickLine={false} />
          <YAxis yAxisId="left" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} domain={[120, 180]} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} domain={[400, 600]} />
          <Tooltip content={<CustomTooltip />} />
          <Line yAxisId="left" type="monotone" dataKey="yield" stroke="#fbbf24" strokeWidth={2.5} dot={{ fill: "#fbbf24", r: 4 }} name="yield" />
          <Line yAxisId="right" type="monotone" dataKey="production" stroke="#818cf8" strokeWidth={2.5} dot={{ fill: "#818cf8", r: 4 }} name="production" />
        </LineChart>
      </ResponsiveContainer>
    </div>
    <div style={{ display: "flex", gap: 16, marginBottom: 16, paddingLeft: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ width: 12, height: 3, background: "#fbbf24", borderRadius: 2 }} />
        <span style={{ color: "#94a3b8", fontSize: 11 }}>Yield (bu/acre)</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ width: 12, height: 3, background: "#818cf8", borderRadius: 2 }} />
        <span style={{ color: "#94a3b8", fontSize: 11 }}>Production (M bushels)</span>
      </div>
    </div>
    <div style={{ background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.2)", borderRadius: 8, padding: "12px 16px" }}>
      <p style={{ color: "#fde68a", fontSize: 12, fontWeight: 600, margin: 0, marginBottom: 4 }}>Cross-Analysis Finding</p>
      <p style={{ color: "#cbd5e1", fontSize: 12.5, margin: 0, lineHeight: 1.5 }}>
        12 food desert communities in Missouri's Bootheel region show &gt;60% economic dependency on corn-related industries. Yield volatility of ±15% between 2018–2024 directly correlates with local food price spikes (r=0.78). These communities have the least capacity to absorb supply shocks due to limited alternative food retail infrastructure.
      </p>
    </div>
  </div>
);

const Query3Result = () => (
  <div>
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#ef4444" }} />
      <span style={{ color: "#e2e8f0", fontSize: 14, fontWeight: 600 }}>Drought Scenario: Priority Response Matrix</span>
    </div>
    <div style={{ overflowX: "auto", marginBottom: 16 }}>
      <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0, fontSize: 12 }}>
        <thead>
          <tr>
            {["County", "Risk", "Food Insecurity", "Corn Dep.", "Pop.", "Recommended Action"].map((h, i) => (
              <th key={i} style={{ padding: "10px 12px", textAlign: "left", color: "#94a3b8", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, borderBottom: "1px solid rgba(255,255,255,0.1)", whiteSpace: "nowrap" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {RISK_COUNTIES.map((row, i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent" }}>
              <td style={{ padding: "9px 12px", color: "#e2e8f0", fontWeight: 500, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>{row.county}</td>
              <td style={{ padding: "9px 12px", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 40, height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${row.risk}%`, height: "100%", background: row.risk > 90 ? "#ef4444" : row.risk > 85 ? "#f97316" : row.risk > 80 ? "#eab308" : "#22c55e", borderRadius: 3 }} />
                  </div>
                  <span style={{ color: row.risk > 90 ? "#fca5a5" : row.risk > 85 ? "#fdba74" : "#fde68a", fontWeight: 600, fontSize: 12 }}>{row.risk}</span>
                </div>
              </td>
              <td style={{ padding: "9px 12px", color: "#cbd5e1", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>{row.insecurity}%</td>
              <td style={{ padding: "9px 12px", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 500, background: row.cornDep === "Very High" ? "rgba(239,68,68,0.15)" : row.cornDep === "High" ? "rgba(251,146,60,0.15)" : row.cornDep === "Medium" ? "rgba(250,204,21,0.15)" : "rgba(34,197,94,0.15)", color: row.cornDep === "Very High" ? "#fca5a5" : row.cornDep === "High" ? "#fdba74" : row.cornDep === "Medium" ? "#fde68a" : "#86efac" }}>
                  {row.cornDep}
                </span>
              </td>
              <td style={{ padding: "9px 12px", color: "#94a3b8", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>{row.pop}</td>
              <td style={{ padding: "9px 12px", color: "#67e8f9", fontSize: 11.5, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>{row.action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    <div style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, padding: "12px 16px", marginBottom: 12 }}>
      <p style={{ color: "#fca5a5", fontSize: 12, fontWeight: 600, margin: 0, marginBottom: 4 }}>Scenario Impact Summary</p>
      <p style={{ color: "#cbd5e1", fontSize: 12.5, margin: 0, lineHeight: 1.5 }}>
        A 20% corn yield reduction would affect approximately 98,000 residents across 6 high-risk counties. Estimated food price increase: 12–18% for corn-derived products within 60 days. Current emergency food infrastructure in these counties can serve ~35% of projected need. Recommendation: Pre-position TEFAP resources and activate mobile food pantry routes within 2 weeks of confirmed yield loss.
      </p>
    </div>
    <div style={{ background: "rgba(45,212,191,0.08)", border: "1px solid rgba(45,212,191,0.2)", borderRadius: 8, padding: "12px 16px" }}>
      <p style={{ color: "#5eead4", fontSize: 12, fontWeight: 600, margin: 0, marginBottom: 4 }}>Data Sources Used</p>
      <p style={{ color: "#94a3b8", fontSize: 11.5, margin: 0, lineHeight: 1.6 }}>
        USDA NASS Quick Stats API (corn yield 2024) · USDA Food Environment Atlas (food insecurity, SNAP, poverty) · USDA Food Access Research Atlas (food desert classification) · Supply disruption model v1.0 (custom)
      </p>
    </div>
  </div>
);

// --- Reasoning Step Component ---
const ReasoningStep = ({ step, index, isActive, isComplete }) => {
  const colors = toolColors[step.tool] || toolColors.analyze;
  return (
    <div style={{
      display: "flex", gap: 10, padding: "8px 0",
      opacity: isComplete ? 1 : isActive ? 1 : 0.3,
      transition: "opacity 0.4s ease",
    }}>
      <div style={{
        display: "flex", flexDirection: "column", alignItems: "center", gap: 4, minWidth: 28,
      }}>
        <div style={{
          width: 28, height: 28, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center",
          background: isComplete ? colors.bg : isActive ? colors.bg : "rgba(255,255,255,0.04)",
          border: `1px solid ${isComplete || isActive ? colors.border : "rgba(255,255,255,0.08)"}`,
          color: isComplete || isActive ? colors.text : "#475569",
          transition: "all 0.3s ease",
        }}>
          {isComplete ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : isActive ? (
            <div style={{ width: 10, height: 10, borderRadius: "50%", border: `2px solid ${colors.text}`, borderTopColor: "transparent", animation: "spin 0.8s linear infinite" }} />
          ) : toolIcons[step.tool]}
        </div>
        {index < 4 && <div style={{ width: 1, height: 12, background: "rgba(255,255,255,0.06)" }} />}
      </div>
      <div style={{ flex: 1, paddingTop: 2 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span style={{ color: isComplete || isActive ? "#e2e8f0" : "#475569", fontSize: 12.5, fontWeight: 500, transition: "color 0.3s ease" }}>{step.label}</span>
          {isActive && <span style={{ color: colors.text, fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>Running</span>}
          {isComplete && <span style={{ color: "#4ade80", fontSize: 10 }}>✓</span>}
        </div>
        {(isActive || isComplete) && (
          <p style={{ color: "#64748b", fontSize: 11, margin: 0, fontFamily: "'JetBrains Mono', 'Fira Code', monospace", lineHeight: 1.4, wordBreak: "break-all" }}>{step.detail}</p>
        )}
      </div>
    </div>
  );
};

// ============================================================
// MAIN APP
// ============================================================
export default function AgriFlow() {
  const [activeQuery, setActiveQuery] = useState(null);
  const [currentStep, setCurrentStep] = useState(-1);
  const [showResult, setShowResult] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, currentStep, showResult]);

  const runQuery = (demo) => {
    if (isProcessing) return;
    setIsProcessing(true);
    setActiveQuery(demo);
    setCurrentStep(-1);
    setShowResult(false);
    setInputValue("");

    setChatHistory(prev => [...prev, { type: "user", text: demo.query }]);

    let step = 0;
    const totalSteps = demo.steps.length;

    const advanceStep = () => {
      if (step < totalSteps) {
        setCurrentStep(step);
        const duration = demo.steps[step].duration;
        step++;
        setTimeout(advanceStep, duration);
      } else {
        setCurrentStep(totalSteps);
        setTimeout(() => {
          setShowResult(true);
          setIsProcessing(false);
          setChatHistory(prev => [...prev, { type: "agent", queryId: demo.id, steps: demo.steps }]);
        }, 500);
      }
    };

    setTimeout(advanceStep, 400);
  };

  const handleSend = () => {
    if (!inputValue.trim() || isProcessing) return;
    const match = DEMO_QUERIES.find(d => inputValue.toLowerCase().includes(d.label.toLowerCase().split(" ")[0]));
    if (match) {
      runQuery(match);
    } else {
      runQuery(DEMO_QUERIES[0]);
    }
  };

  return (
    <div style={{
      width: "100%", height: "100vh", display: "flex", flexDirection: "column",
      background: "#0b0f1a",
      fontFamily: "'DM Sans', 'Satoshi', -apple-system, BlinkMacSystemFont, sans-serif",
      color: "#e2e8f0", overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
      `}</style>

      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 24px",
        background: "rgba(255,255,255,0.02)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 8,
            background: "linear-gradient(135deg, #059669, #10b981)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 20px rgba(16,185,129,0.3)",
          }}>
            <LeafIcon />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
              <span style={{ fontSize: 17, fontWeight: 700, color: "#f0fdf4", letterSpacing: -0.3 }}>AgriFlow</span>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#10b981", letterSpacing: 1, textTransform: "uppercase" }}>Agent</span>
            </div>
            <span style={{ fontSize: 11, color: "#64748b" }}>Food Supply Chain Intelligence</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#10b981", boxShadow: "0 0 8px rgba(16,185,129,0.5)" }} />
            <span style={{ fontSize: 11, color: "#64748b" }}>3 tools connected</span>
          </div>
          <div style={{ padding: "4px 10px", borderRadius: 6, background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)" }}>
            <span style={{ fontSize: 11, color: "#6ee7b7", fontWeight: 500 }}>MUIDSI Hackathon 2026</span>
          </div>
          {chatHistory.length > 0 && (
            <button onClick={() => { setChatHistory([]); setActiveQuery(null); setCurrentStep(-1); setShowResult(false); setIsProcessing(false); setInputValue(""); }} style={{
              padding: "5px 12px", borderRadius: 6, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
              color: "#94a3b8", fontSize: 11, fontWeight: 500, cursor: "pointer", display: "flex", alignItems: "center", gap: 5, transition: "all 0.2s ease",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.1)"; e.currentTarget.style.color = "#e2e8f0"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; e.currentTarget.style.color = "#94a3b8"; }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
              New Chat
            </button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* Chat / Main Panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Chat Area */}
          <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px" }}>

            {chatHistory.length === 0 && !isProcessing && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 24, animation: "fadeUp 0.6s ease" }}>
                <div style={{
                  width: 64, height: 64, borderRadius: 16,
                  background: "linear-gradient(135deg, rgba(16,185,129,0.15), rgba(59,130,246,0.15))",
                  border: "1px solid rgba(16,185,129,0.2)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 19 2c1 2 2 4.5 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
                  </svg>
                </div>
                <div style={{ textAlign: "center" }}>
                  <h2 style={{ fontSize: 20, fontWeight: 600, color: "#e2e8f0", margin: 0, marginBottom: 6 }}>Ask your supply chain anything</h2>
                  <p style={{ fontSize: 13, color: "#64748b", margin: 0 }}>AgriFlow reasons across USDA data, crop statistics, and community indicators</p>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", maxWidth: 600, marginTop: 4 }}>
                  {DEMO_QUERIES.map(d => (
                    <button key={d.id} onClick={() => runQuery(d)} style={{
                      padding: "10px 16px", borderRadius: 10,
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      color: "#94a3b8", fontSize: 12.5, cursor: "pointer",
                      transition: "all 0.2s ease", textAlign: "left", maxWidth: 280,
                    }}
                    onMouseEnter={e => { e.target.style.background = "rgba(16,185,129,0.08)"; e.target.style.borderColor = "rgba(16,185,129,0.25)"; e.target.style.color = "#e2e8f0"; }}
                    onMouseLeave={e => { e.target.style.background = "rgba(255,255,255,0.03)"; e.target.style.borderColor = "rgba(255,255,255,0.08)"; e.target.style.color = "#94a3b8"; }}
                    >
                      {d.query}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {chatHistory.map((msg, i) => (
              <div key={i} style={{ marginBottom: 16, animation: "fadeUp 0.4s ease" }}>
                {msg.type === "user" && (
                  <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
                    <div style={{
                      maxWidth: "75%", padding: "12px 16px", borderRadius: "14px 14px 4px 14px",
                      background: "linear-gradient(135deg, #059669, #0d9488)",
                      color: "#f0fdf4", fontSize: 13.5, lineHeight: 1.5,
                    }}>
                      {msg.text}
                    </div>
                  </div>
                )}
                {msg.type === "agent" && (
                  <div style={{ display: "flex", gap: 10 }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: 8, flexShrink: 0, marginTop: 2,
                      background: "linear-gradient(135deg, #059669, #10b981)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                      <LeafIcon />
                    </div>
                    <div style={{ flex: 1, maxWidth: "calc(100% - 40px)" }}>
                      <div style={{
                        background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
                        borderRadius: "4px 14px 14px 14px", padding: 16,
                      }}>
                        {msg.queryId === 1 && <Query1Result />}
                        {msg.queryId === 2 && <Query2Result />}
                        {msg.queryId === 3 && <Query3Result />}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Active reasoning (while processing) */}
            {isProcessing && activeQuery && (
              <div style={{ display: "flex", gap: 10, animation: "fadeUp 0.4s ease" }}>
                <div style={{
                  width: 28, height: 28, borderRadius: 8, flexShrink: 0, marginTop: 2,
                  background: "linear-gradient(135deg, #059669, #10b981)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <LeafIcon />
                </div>
                <div style={{
                  flex: 1, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
                  borderRadius: "4px 14px 14px 14px", padding: 16,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", animation: "pulse 1.5s ease infinite" }} />
                    <span style={{ color: "#94a3b8", fontSize: 12, fontWeight: 500 }}>Reasoning...</span>
                  </div>
                  {activeQuery.steps.map((step, i) => (
                    <ReasoningStep key={i} step={step} index={i} isActive={i === currentStep} isComplete={i < currentStep} />
                  ))}
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Quick query buttons — always visible */}
          <div style={{
            padding: "8px 24px 0 24px",
            display: "flex", gap: 6, flexShrink: 0, flexWrap: "wrap",
          }}>
            {DEMO_QUERIES.map(d => (
              <button key={d.id} onClick={() => runQuery(d)} disabled={isProcessing} style={{
                padding: "6px 12px", borderRadius: 8,
                background: isProcessing ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: isProcessing ? "#334155" : "#64748b", fontSize: 11, cursor: isProcessing ? "default" : "pointer",
                transition: "all 0.2s ease", whiteSpace: "nowrap",
              }}
              onMouseEnter={e => { if (!isProcessing) { e.target.style.background = "rgba(16,185,129,0.08)"; e.target.style.borderColor = "rgba(16,185,129,0.25)"; e.target.style.color = "#e2e8f0"; }}}
              onMouseLeave={e => { e.target.style.background = isProcessing ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.04)"; e.target.style.borderColor = "rgba(255,255,255,0.08)"; e.target.style.color = isProcessing ? "#334155" : "#64748b"; }}
              >
                Demo {d.id}: {d.label}
              </button>
            ))}
          </div>

          {/* Input Area */}
          <div style={{
            padding: "16px 24px",
            borderTop: "1px solid rgba(255,255,255,0.06)",
            background: "rgba(255,255,255,0.02)",
            flexShrink: 0,
          }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 10,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 12, padding: "4px 4px 4px 16px",
              transition: "border-color 0.2s ease",
            }}>
              <input
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleSend()}
                placeholder="Ask about food supply chains, food deserts, crop data..."
                disabled={isProcessing}
                style={{
                  flex: 1, border: "none", outline: "none", background: "transparent",
                  color: "#e2e8f0", fontSize: 13.5,
                  fontFamily: "'DM Sans', sans-serif",
                }}
              />
              <button
                onClick={handleSend}
                disabled={isProcessing || !inputValue.trim()}
                style={{
                  width: 38, height: 38, borderRadius: 9,
                  background: inputValue.trim() && !isProcessing ? "linear-gradient(135deg, #059669, #10b981)" : "rgba(255,255,255,0.05)",
                  border: "none", cursor: inputValue.trim() && !isProcessing ? "pointer" : "default",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: inputValue.trim() && !isProcessing ? "#f0fdf4" : "#475569",
                  transition: "all 0.2s ease",
                }}
              >
                <SendIcon />
              </button>
            </div>
            <div style={{ display: "flex", gap: 12, marginTop: 8, paddingLeft: 2 }}>
              <span style={{ fontSize: 10.5, color: "#475569" }}>
                Powered by USDA Food Environment Atlas · Food Access Research Atlas · NASS Quick Stats API
              </span>
            </div>
          </div>
        </div>

        {/* Right Sidebar — Tool Status */}
        <div style={{
          width: 240, flexShrink: 0,
          borderLeft: "1px solid rgba(255,255,255,0.06)",
          background: "rgba(255,255,255,0.01)",
          padding: 16, overflowY: "auto",
          display: "flex", flexDirection: "column", gap: 16,
        }}>
          <div>
            <span style={{ fontSize: 10, fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>Connected Tools</span>
          </div>

          {[
            { name: "Food Environment Atlas", type: "Database", status: "Connected", color: "#3b82f6", vars: "300+ variables", level: "County" },
            { name: "Food Access Atlas", type: "Database", status: "Connected", color: "#8b5cf6", vars: "Food desert data", level: "Census tract" },
            { name: "NASS Quick Stats", type: "Live API", status: "Active", color: "#f59e0b", vars: "Crop production", level: "State/County" },
          ].map((tool, i) => (
            <div key={i} style={{
              padding: 12, borderRadius: 10,
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.06)",
            }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>{tool.name}</span>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", boxShadow: "0 0 6px rgba(16,185,129,0.5)" }} />
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 10.5, color: "#64748b" }}>Type</span>
                  <span style={{ fontSize: 10.5, color: tool.type === "Live API" ? "#fbbf24" : "#94a3b8", fontWeight: 500 }}>{tool.type}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 10.5, color: "#64748b" }}>Coverage</span>
                  <span style={{ fontSize: 10.5, color: "#94a3b8" }}>{tool.vars}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 10.5, color: "#64748b" }}>Resolution</span>
                  <span style={{ fontSize: 10.5, color: "#94a3b8" }}>{tool.level}</span>
                </div>
              </div>
            </div>
          ))}

          <div style={{ marginTop: "auto", padding: 12, borderRadius: 10, background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.15)" }}>
            <span style={{ fontSize: 10, fontWeight: 600, color: "#6ee7b7", textTransform: "uppercase", letterSpacing: 0.5 }}>Model Info</span>
            <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 3 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontSize: 10.5, color: "#64748b" }}>Agent</span>
                <span style={{ fontSize: 10.5, color: "#94a3b8" }}>AgriFlow v1</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontSize: 10.5, color: "#64748b" }}>Platform</span>
                <span style={{ fontSize: 10.5, color: "#94a3b8" }}>Archia Cloud</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontSize: 10.5, color: "#64748b" }}>Track</span>
                <span style={{ fontSize: 10.5, color: "#94a3b8" }}>Agriculture/Plant</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
