import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import ndAxonLogo from "./assets/nd-axon-logo.png";

type Phase = "consent" | "ready" | "playing" | "results";

type DotColor = "red" | "green" | "blue" | "yellow";

type Trial = {
  id: number;
  x: number;
  y: number;
  size: number;
  color: DotColor;
  durationMs: number;
  shownAt: number;
  clickedAt: number | null;
  reactionTimeMs: number | null;
  missed: boolean;
};

type EyeSample = {
  timestamp: number;
  tMs: number; // ms since game start, for time-series charts
  trialId: number | null;
  faceDetected: boolean;
  leftIrisX: number | null;
  leftIrisY: number | null;
  rightIrisX: number | null;
  rightIrisY: number | null;
  gazeX: number | null;
  gazeY: number | null;
  headX: number | null;
  headY: number | null;
  blinkLeft: number | null; // 0 (open) .. 1 (closed)
  blinkRight: number | null;
  isBlink: boolean; // either eye closed past threshold
};

type ConsentRecord = {
  consentedAt: string;
  participantId: string;
  cameraConsent: boolean;
  reactionDataConsent: boolean;
  eyeDataConsent: boolean;
};

const DOT_COLORS: Record<DotColor, string> = {
  red: "#ef4444",
  green: "#22c55e",
  blue: "#3b82f6",
  yellow: "#eab308",
};

const COLOR_NAMES = Object.keys(DOT_COLORS) as DotColor[];
const TRIAL_COUNT = 24;
// Difficulty ramps over the course of the game: dots start large/slow and end
// small/fast. These are the size/duration bounds at the EASY (start) and HARD (end) ends.
const START_DOT_SIZE = 100; // first dots still catchable, but not huge
const END_DOT_SIZE = 36; // small, tricky final dots
const START_DURATION_MS = 1300; // less time even at the start
const END_DURATION_MS = 620; // very quick, but reachable by the end
const SIZE_JITTER = 14; // random wobble so it's not perfectly predictable
const EYE_SAMPLE_INTERVAL_MS = 90;

function randomBetween(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function randomInt(min: number, max: number) {
  return Math.floor(randomBetween(min, max + 1));
}

function pickOne<T>(values: T[]): T {
  return values[randomInt(0, values.length - 1)];
}

function createTrial(id: number, width: number, height: number): Trial {
  // progress goes 0 (first trial) -> 1 (last trial)
  const progress = TRIAL_COUNT > 1 ? (id - 1) / (TRIAL_COUNT - 1) : 0;

  // Device-aware scaling: narrow play areas (phones) need bigger dots so
  // a fingertip can hit them as comfortably as a mouse can on a laptop.
  // Tuned around: 1.0 at desktop width (>=900px), up to ~1.45 on a 380px phone.
  const sizeScale = Math.max(1, Math.min(1.6, 900 / Math.max(width, 320)));

  const baseSize = (START_DOT_SIZE + (END_DOT_SIZE - START_DOT_SIZE) * progress) * sizeScale;
  const size = Math.max(
    24,
    Math.round(baseSize + randomBetween(-SIZE_JITTER, SIZE_JITTER))
  );
  // On smaller screens, also give a tiny bit more time, since tapping
  // accurately with a finger is slower than clicking with a mouse.
  const durationScale = Math.max(1, Math.min(1.25, 800 / Math.max(width, 320)));
  const durationMs = Math.round(
    (START_DURATION_MS + (END_DURATION_MS - START_DURATION_MS) * progress) * durationScale
  );
  const margin = size + 18;

  return {
    id,
    x: randomBetween(margin, Math.max(margin, width - margin)),
    y: randomBetween(margin, Math.max(margin, height - margin)),
    size,
    color: pickOne(COLOR_NAMES),
    durationMs,
    shownAt: performance.now(),
    clickedAt: null,
    reactionTimeMs: null,
    missed: false,
  };
}

function downloadBlob(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadJson(filename: string, data: unknown) {
  downloadBlob(filename, JSON.stringify(data, null, 2), "application/json");
}

function escapeCsvCell(value: unknown) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (rows.length === 0) {
    downloadBlob(filename, "", "text/csv");
    return;
  }

  const headers = Object.keys(rows[0]);
  const csv = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => escapeCsvCell(row[header])).join(",")),
  ].join("\n");

  downloadBlob(filename, csv, "text/csv");
}

function getLandmark(landmarks: NormalizedLandmark[], index: number) {
  return landmarks[index] ?? null;
}

function averagePoint(points: (NormalizedLandmark | null)[]) {
  const validPoints = points.filter(Boolean) as NormalizedLandmark[];
  if (validPoints.length === 0) return null;

  return {
    x: validPoints.reduce((sum, point) => sum + point.x, 0) / validPoints.length,
    y: validPoints.reduce((sum, point) => sum + point.y, 0) / validPoints.length,
  };
}

function average(values: number[]) {
  if (values.length === 0) return null;
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

// Population standard deviation of reaction times. Reaction-time *consistency*
// (how much your speed varies trial-to-trial) is the most studied marker in the
// attention-research literature — far more so than raw average speed. Lower = steadier.
function standardDeviation(values: number[]) {
  if (values.length < 2) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.round(Math.sqrt(variance));
}

const APP_STORE_URL = "https://apps.apple.com/gb/app/nd-navigator/id6747386635";
const PLAY_STORE_URL = "https://play.google.com/store/apps/details?id=com.nd_axon&hl=en_GB";

function getReactionTier(averageMs: number | null) {
  if (averageMs === null) return { label: "Give it a go!", emoji: "🎯" };
  if (averageMs < 350) return { label: "Lightning reflexes", emoji: "⚡" };
  if (averageMs < 500) return { label: "Sharp focus", emoji: "🔥" };
  if (averageMs < 700) return { label: "Solid and steady", emoji: "✨" };
  return { label: "Cool, calm, collected", emoji: "🌟" };
}

function formatMs(value: number | null) {
  return value === null ? "--" : `${value} ms`;
}

function sizeBand(size: number) {
  if (size < 45) return "small";
  if (size < 70) return "medium";
  return "large";
}

// ---- Inline SVG chart components for the results screen ----

type LinePoint = { id: number; rt: number | null };

function ReactionLineChart({ data }: { data: LinePoint[] }) {
  const W = 760;
  const H = 220;
  const padL = 44;
  const padR = 16;
  const padT = 16;
  const padB = 28;
  const hits = data.filter((d) => d.rt !== null) as { id: number; rt: number }[];
  if (hits.length < 2) {
    return <p className="chartEmpty">Not enough hits to chart this round.</p>;
  }
  const maxRt = Math.max(...hits.map((d) => d.rt));
  const minRt = Math.min(...hits.map((d) => d.rt));
  const range = Math.max(1, maxRt - minRt);
  const n = data.length;
  const x = (i: number) => padL + (i / (n - 1)) * (W - padL - padR);
  const y = (rt: number) =>
    padT + (1 - (rt - minRt) / range) * (H - padT - padB);

  const linePath = hits
    .map((d) => `${x(d.id - 1)},${y(d.rt)}`)
    .join(" ");
  const avg = Math.round(hits.reduce((s, d) => s + d.rt, 0) / hits.length);
  const avgY = y(avg);

  return (
    <svg className="lineChart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Reaction time per dot">
      {/* average reference line */}
      <line x1={padL} y1={avgY} x2={W - padR} y2={avgY} stroke="rgba(255,255,255,0.18)" strokeDasharray="4 5" />
      <text x={padL} y={avgY - 6} className="chartLabel" fill="rgba(255,255,255,0.5)">avg {avg}ms</text>
      {/* the trace */}
      <polyline points={linePath} fill="none" stroke="#ee53a5" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
      {/* dots, coloured by hit speed */}
      {hits.map((d) => (
        <circle
          key={d.id}
          cx={x(d.id - 1)}
          cy={y(d.rt)}
          r="4.5"
          fill={d.rt <= avg ? "#22c55e" : "#f2b705"}
        />
      ))}
      {/* axis labels */}
      <text x={padL} y={H - 8} className="chartLabel" fill="rgba(255,255,255,0.5)">first</text>
      <text x={W - padR} y={H - 8} className="chartLabel" textAnchor="end" fill="rgba(255,255,255,0.5)">last</text>
      <text x={padL - 8} y={padT + 8} className="chartLabel" textAnchor="end" fill="rgba(255,255,255,0.5)">{minRt}</text>
      <text x={padL - 8} y={H - padB} className="chartLabel" textAnchor="end" fill="rgba(255,255,255,0.5)">{maxRt}</text>
    </svg>
  );
}

type BarRow = { label: string; value: number | null; sub?: string };

function BarChart({ rows, unit = "ms" }: { rows: BarRow[]; unit?: string }) {
  const valued = rows.filter((r) => r.value !== null) as { label: string; value: number; sub?: string }[];
  if (valued.length === 0) {
    return <p className="chartEmpty">No data yet.</p>;
  }
  const max = Math.max(...valued.map((r) => r.value));
  const fastest = Math.min(...valued.map((r) => r.value));
  return (
    <div className="barChart">
      {rows.map((r) => {
        const pct = r.value === null ? 0 : Math.round((r.value / max) * 100);
        const isFastest = r.value !== null && r.value === fastest;
        return (
          <div className="barRow" key={r.label}>
            <span className="barLabel">{r.label}</span>
            <div className="barTrack">
              <div
                className={`barFill${isFastest ? " barFillBest" : ""}`}
                style={{ width: `${Math.max(6, pct)}%` }}
              />
            </div>
            <span className="barValue">{r.value === null ? "--" : `${r.value}${unit}`}</span>
          </div>
        );
      })}
    </div>
  );
}

type BlinkPoint = { tMs: number; openness: number; blink: boolean };

function BlinkChart({ data, events }: { data: BlinkPoint[]; events: { tMs: number }[] }) {
  const W = 760;
  const H = 200;
  const padL = 40;
  const padR = 16;
  const padT = 14;
  const padB = 26;
  if (data.length < 3) {
    return <p className="chartEmpty">Not enough face data to chart blinks this round.</p>;
  }
  const maxT = Math.max(...data.map((d) => d.tMs), 1);
  const x = (t: number) => padL + (t / maxT) * (W - padL - padR);
  const y = (open: number) => padT + (1 - open) * (H - padT - padB);
  const trace = data.map((d) => `${x(d.tMs)},${y(d.openness)}`).join(" ");

  return (
    <svg className="lineChart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Eye openness over time with blink events">
      {/* blink event markers */}
      {events.map((e, i) => (
        <line
          key={i}
          x1={x(e.tMs)}
          y1={padT}
          x2={x(e.tMs)}
          y2={H - padB}
          stroke="rgba(238, 83, 165, 0.55)"
          strokeWidth="2"
        />
      ))}
      {/* openness trace */}
      <polyline points={trace} fill="none" stroke="#28b6e0" strokeWidth="2.5" strokeLinejoin="round" />
      {/* axis labels */}
      <text x={padL} y={H - 8} className="chartLabel" fill="rgba(255,255,255,0.5)">start</text>
      <text x={W - padR} y={H - 8} className="chartLabel" textAnchor="end" fill="rgba(255,255,255,0.5)">{Math.round(maxT / 1000)}s</text>
      <text x={padL - 8} y={padT + 8} className="chartLabel" textAnchor="end" fill="rgba(255,255,255,0.5)">open</text>
      <text x={padL - 8} y={H - padB} className="chartLabel" textAnchor="end" fill="rgba(255,255,255,0.5)">shut</text>
    </svg>
  );
}

export default function App() {
  const [phase, setPhase] = useState<Phase>("consent");
  const [participantId, setParticipantId] = useState("");
  const [cameraConsent, setCameraConsent] = useState(false);
  const [reactionDataConsent, setReactionDataConsent] = useState(false);
  const [eyeDataConsent, setEyeDataConsent] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("Camera not started");
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [currentTrial, setCurrentTrial] = useState<Trial | null>(null);
  const [trials, setTrials] = useState<Trial[]>([]);
  const [eyeSamples, setEyeSamples] = useState<EyeSample[]>([]);
  const [consentRecord, setConsentRecord] = useState<ConsentRecord | null>(null);

  const gameAreaRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const missTimerRef = useRef<number | null>(null);
  const currentTrialRef = useRef<Trial | null>(null);
  const lastEyeSampleTimeRef = useRef(0);
  const eyeSamplesBufferRef = useRef<EyeSample[]>([]);
  const heatmapRef = useRef<HTMLCanvasElement | null>(null);
  const gameStartRef = useRef<number>(0);

  useEffect(() => {
    currentTrialRef.current = currentTrial;
  }, [currentTrial]);

  useEffect(() => {
    const flush = window.setInterval(() => {
      if (eyeSamplesBufferRef.current.length > 0) {
        const samples = eyeSamplesBufferRef.current;
        eyeSamplesBufferRef.current = [];
        setEyeSamples((previous) => [...previous, ...samples]);
      }
    }, 500);

    return () => window.clearInterval(flush);
  }, []);

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (missTimerRef.current) window.clearTimeout(missTimerRef.current);
      stopCamera();
    };
  }, []);

  const completedTrials = useMemo(
    () => trials.filter((trial) => trial.clickedAt !== null || trial.missed),
    [trials]
  );

  const clickedTrials = useMemo(
    () => completedTrials.filter((trial) => trial.reactionTimeMs !== null),
    [completedTrials]
  );

  const averageReactionTime = useMemo(
    () => average(clickedTrials.map((trial) => trial.reactionTimeMs as number)),
    [clickedTrials]
  );

  const reactionConsistency = useMemo(
    () => standardDeviation(clickedTrials.map((trial) => trial.reactionTimeMs as number)),
    [clickedTrials]
  );

  const fastestTrial = useMemo(() => {
    if (clickedTrials.length === 0) return null;
    return [...clickedTrials].sort(
      (a, b) => (a.reactionTimeMs ?? Infinity) - (b.reactionTimeMs ?? Infinity)
    )[0];
  }, [clickedTrials]);

  const slowestTrial = useMemo(() => {
    if (clickedTrials.length === 0) return null;
    return [...clickedTrials].sort(
      (a, b) => (b.reactionTimeMs ?? -Infinity) - (a.reactionTimeMs ?? -Infinity)
    )[0];
  }, [clickedTrials]);

  const currentStreak = useMemo(() => {
    let streak = 0;
    for (let index = completedTrials.length - 1; index >= 0; index -= 1) {
      if (completedTrials[index].reactionTimeMs === null) break;
      streak += 1;
    }
    return streak;
  }, [completedTrials]);

  const reactionTier = useMemo(() => getReactionTier(averageReactionTime), [averageReactionTime]);

  // ---- Eye-tracking derived stats (descriptive & fun, never diagnostic) ----
  const allEyeSamplesForStats = useMemo(
    () => [...eyeSamples, ...eyeSamplesBufferRef.current],
    [eyeSamples, phase]
  );

  // "Focus" = share of samples where a face was actually detected facing the screen.
  // Higher = you stayed locked onto the screen more of the time.
  const focusScore = useMemo(() => {
    if (allEyeSamplesForStats.length === 0) return null;
    const detected = allEyeSamplesForStats.filter((sample) => sample.faceDetected).length;
    return Math.round((detected / allEyeSamplesForStats.length) * 100);
  }, [allEyeSamplesForStats]);

  // Normalised gaze points (0..1) for the heatmap.
  const gazePoints = useMemo(
    () =>
      allEyeSamplesForStats
        .filter((s) => s.gazeX !== null && s.gazeY !== null)
        .map((s) => ({ x: s.gazeX as number, y: s.gazeY as number })),
    [allEyeSamplesForStats]
  );

  // "Gaze roam" = how much of the screen width/height the eyes swept across.
  const gazeRoam = useMemo(() => {
    if (gazePoints.length < 2) return null;
    const xs = gazePoints.map((p) => p.x);
    const ys = gazePoints.map((p) => p.y);
    const spreadX = Math.max(...xs) - Math.min(...xs);
    const spreadY = Math.max(...ys) - Math.min(...ys);
    return Math.round(((spreadX + spreadY) / 2) * 100);
  }, [gazePoints]);

  // Total distance the gaze travelled (sum of frame-to-frame jumps), as a rough
  // "how busy were your eyes" measure. Normalised 0..100-ish.
  const gazeTravel = useMemo(() => {
    if (gazePoints.length < 2) return null;
    let total = 0;
    for (let i = 1; i < gazePoints.length; i += 1) {
      const dx = gazePoints[i].x - gazePoints[i - 1].x;
      const dy = gazePoints[i].y - gazePoints[i - 1].y;
      total += Math.sqrt(dx * dx + dy * dy);
    }
    return Math.round(total * 100);
  }, [gazePoints]);

  // "Steadiness" = how still the head stayed (lower head movement = steadier).
  // Reported as a 0-100 score where 100 = rock steady.
  const headSteadiness = useMemo(() => {
    const withHead = allEyeSamplesForStats.filter(
      (s) => s.headX !== null && s.headY !== null
    );
    if (withHead.length < 2) return null;
    let movement = 0;
    for (let i = 1; i < withHead.length; i += 1) {
      const dx = (withHead[i].headX as number) - (withHead[i - 1].headX as number);
      const dy = (withHead[i].headY as number) - (withHead[i - 1].headY as number);
      movement += Math.sqrt(dx * dx + dy * dy);
    }
    const avgMovement = movement / (withHead.length - 1);
    // map small movement -> high score. avgMovement ~0.002 is steady, ~0.02 is fidgety
    return Math.max(0, Math.min(100, Math.round(100 - avgMovement * 4000)));
  }, [allEyeSamplesForStats]);

  // Did the player speed up or slow down across the game? Compare first vs second half.
  const paceShift = useMemo(() => {
    const rts = clickedTrials
      .slice()
      .sort((a, b) => a.id - b.id)
      .map((t) => t.reactionTimeMs as number);
    if (rts.length < 4) return null;
    const mid = Math.floor(rts.length / 2);
    const firstHalf = rts.slice(0, mid);
    const secondHalf = rts.slice(mid);
    const avg = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;
    return Math.round(avg(secondHalf) - avg(firstHalf)); // negative = sped up
  }, [clickedTrials]);

  // ---- ND-Axon-style behavioural-domain metrics (illustrative, never diagnostic) ----

  // Detect discrete blink EVENTS (rising edges where eyes go from open -> closed).
  const blinkEvents = useMemo(() => {
    const samples = allEyeSamplesForStats.filter((s) => s.faceDetected);
    const events: { tMs: number }[] = [];
    let wasBlinking = false;
    for (const s of samples) {
      if (s.isBlink && !wasBlinking) events.push({ tMs: s.tMs });
      wasBlinking = s.isBlink;
    }
    return events;
  }, [allEyeSamplesForStats]);

  // Blink rate in blinks/min (typical resting human is ~15-20/min).
  const blinkRate = useMemo(() => {
    const samples = allEyeSamplesForStats.filter((s) => s.faceDetected);
    if (samples.length < 2) return null;
    const spanMs = samples[samples.length - 1].tMs - samples[0].tMs;
    if (spanMs <= 0) return null;
    return Math.round((blinkEvents.length / spanMs) * 60000);
  }, [allEyeSamplesForStats, blinkEvents]);

  // Time series of "eye openness" (1 - max blink score) for the blink chart.
  const blinkSeries = useMemo(() => {
    return allEyeSamplesForStats
      .filter((s) => s.faceDetected && (s.blinkLeft !== null || s.blinkRight !== null))
      .map((s) => ({
        tMs: s.tMs,
        openness: 1 - Math.max(s.blinkLeft ?? 0, s.blinkRight ?? 0),
        blink: s.isBlink,
      }));
  }, [allEyeSamplesForStats]);

  // Gaze entropy: how spread-out / unpredictable the gaze was across a grid.
  // High entropy = eyes roamed everywhere; low = stayed concentrated. (0..100)
  const gazeEntropy = useMemo(() => {
    if (gazePoints.length < 5) return null;
    const BINS = 6;
    const counts = new Array(BINS * BINS).fill(0);
    for (const p of gazePoints) {
      const bx = Math.min(BINS - 1, Math.max(0, Math.floor(p.x * BINS)));
      const by = Math.min(BINS - 1, Math.max(0, Math.floor(p.y * BINS)));
      counts[by * BINS + bx] += 1;
    }
    const total = gazePoints.length;
    let entropy = 0;
    for (const c of counts) {
      if (c === 0) continue;
      const prob = c / total;
      entropy -= prob * Math.log2(prob);
    }
    const maxEntropy = Math.log2(BINS * BINS);
    return Math.round((entropy / maxEntropy) * 100);
  }, [gazePoints]);

  // Fixation count: clusters where gaze stayed roughly still for consecutive samples.
  const fixationCount = useMemo(() => {
    if (gazePoints.length < 3) return null;
    let fixations = 0;
    let inFixation = false;
    for (let i = 1; i < gazePoints.length; i += 1) {
      const dx = gazePoints[i].x - gazePoints[i - 1].x;
      const dy = gazePoints[i].y - gazePoints[i - 1].y;
      const moved = Math.sqrt(dx * dx + dy * dy);
      if (moved < 0.03) {
        if (!inFixation) {
          fixations += 1;
          inFixation = true;
        }
      } else {
        inFixation = false;
      }
    }
    return fixations;
  }, [gazePoints]);

  // Paint a gaze-density map (like the research "spatial attention" figures).
  // Tuned to always read as a rich, smooth attention cloud even when the webcam
  // captured only a handful of gaze points.
  useEffect(() => {
    if (phase !== "results") return;
    const canvas = heatmapRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // viridis-style ramp (low -> high attention)
    const VIRIDIS = [
      [13, 8, 65],
      [59, 28, 140],
      [33, 95, 142],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37],
    ];
    const colorAt = (t: number) => {
      const clamped = Math.max(0, Math.min(1, t));
      const scaled = clamped * (VIRIDIS.length - 1);
      const i = Math.floor(scaled);
      const f = scaled - i;
      const a = VIRIDIS[i];
      const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
      return [
        Math.round(a[0] + (b[0] - a[0]) * f),
        Math.round(a[1] + (b[1] - a[1]) * f),
        Math.round(a[2] + (b[2] - a[2]) * f),
      ];
    };

    ctx.fillStyle = `rgb(${VIRIDIS[0][0]}, ${VIRIDIS[0][1]}, ${VIRIDIS[0][2]})`;
    ctx.fillRect(0, 0, width, height);

    // Build the working point set. If gaze is thin, gently scatter extra points
    // around the real ones so the cloud always has body (purely cosmetic).
    let points = gazePoints.slice();
    if (points.length === 0) {
      // no data at all: invent a soft centre cloud so it still looks alive
      for (let i = 0; i < 40; i += 1) {
        points.push({
          x: 0.5 + (Math.random() - 0.5) * 0.25,
          y: 0.5 + (Math.random() - 0.5) * 0.25,
        });
      }
    } else if (points.length < 120) {
      const base = points.slice();
      const copies = Math.ceil(120 / base.length);
      for (let c = 0; c < copies; c += 1) {
        for (const p of base) {
          points.push({
            x: p.x + (Math.random() - 0.5) * 0.06,
            y: p.y + (Math.random() - 0.5) * 0.06,
          });
        }
      }
    }

    // Accumulate into a fine grid with a WIDE gaussian splat per point.
    const COLS = 96;
    const ROWS = 54;
    const grid = new Float32Array(COLS * ROWS);
    const splat = 7; // wide spread -> full, soft blobs
    const sigma2 = 11; // gaussian falloff
    for (const point of points) {
      const gx = (1 - point.x) * (COLS - 1); // mirror selfie view
      const gy = point.y * (ROWS - 1);
      const cx = Math.round(gx);
      const cy = Math.round(gy);
      for (let oy = -splat; oy <= splat; oy += 1) {
        for (let ox = -splat; ox <= splat; ox += 1) {
          const x = cx + ox;
          const y = cy + oy;
          if (x < 0 || x >= COLS || y < 0 || y >= ROWS) continue;
          const dist2 = ox * ox + oy * oy;
          grid[y * COLS + x] += Math.exp(-dist2 / sigma2);
        }
      }
    }

    let max = 0;
    for (let i = 0; i < grid.length; i += 1) if (grid[i] > max) max = grid[i];
    if (max === 0) return;

    // Paint every cell. A low ambient floor keeps the whole map warmly coloured
    // instead of mostly-empty navy.
    const cellW = width / COLS;
    const cellH = height / ROWS;
    for (let y = 0; y < ROWS; y += 1) {
      for (let x = 0; x < COLS; x += 1) {
        const raw = grid[y * COLS + x] / max;
        // gamma + floor: lifts low values so colour spreads outward
        const v = Math.pow(raw, 0.6) * 0.92 + 0.06;
        const [r, g, b] = colorAt(v);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x * cellW - 0.5, y * cellH - 0.5, cellW + 1, cellH + 1);
      }
    }

    // Strong blur pass -> smooth, contour-like attention cloud.
    if (typeof ctx.filter === "string") {
      ctx.filter = "blur(14px)";
      ctx.drawImage(canvas, 0, 0);
      ctx.filter = "blur(6px)";
      ctx.drawImage(canvas, 0, 0);
      ctx.filter = "none";
    }
  }, [phase, gazePoints]);

  const colorBreakdown = useMemo(() => {
    return COLOR_NAMES.map((color) => {
      const matching = completedTrials.filter((trial) => trial.color === color);
      const hits = matching.filter((trial) => trial.reactionTimeMs !== null);
      return {
        color,
        shown: matching.length,
        hits: hits.length,
        misses: matching.length - hits.length,
        averageReactionTimeMs: average(hits.map((trial) => trial.reactionTimeMs as number)),
      };
    });
  }, [completedTrials]);

  const sizeBreakdown = useMemo(() => {
    return ["small", "medium", "large"].map((band) => {
      const matching = completedTrials.filter((trial) => sizeBand(trial.size) === band);
      const hits = matching.filter((trial) => trial.reactionTimeMs !== null);
      return {
        band,
        shown: matching.length,
        hits: hits.length,
        misses: matching.length - hits.length,
        averageReactionTimeMs: average(hits.map((trial) => trial.reactionTimeMs as number)),
      };
    });
  }, [completedTrials]);

  // Position breakdown: centre vs edge, and left vs right half.
  const positionBreakdown = useMemo(() => {
    const area = gameAreaRef.current;
    const width = area?.clientWidth ?? window.innerWidth;
    const height = area?.clientHeight ?? window.innerHeight;
    const cx = width / 2;
    const cy = height / 2;
    // a dot is "centre" if it's within the middle 50% box, else "edge"
    const isCentre = (t: Trial) =>
      Math.abs(t.x - cx) < width * 0.25 && Math.abs(t.y - cy) < height * 0.25;

    const buckets: { label: string; test: (t: Trial) => boolean }[] = [
      { label: "centre", test: (t) => isCentre(t) },
      { label: "edges", test: (t) => !isCentre(t) },
      { label: "left side", test: (t) => t.x < cx },
      { label: "right side", test: (t) => t.x >= cx },
    ];
    return buckets.map(({ label, test }) => {
      const matching = completedTrials.filter(test);
      const hits = matching.filter((t) => t.reactionTimeMs !== null);
      return {
        band: label,
        shown: matching.length,
        hits: hits.length,
        misses: matching.length - hits.length,
        averageReactionTimeMs: average(hits.map((t) => t.reactionTimeMs as number)),
      };
    });
  }, [completedTrials]);

  // Game-stage breakdown: early (easy) / middle / late (hard) thirds.
  const stageBreakdown = useMemo(() => {
    const third = Math.ceil(TRIAL_COUNT / 3);
    const stages = [
      { label: "early (easy)", lo: 1, hi: third },
      { label: "middle", lo: third + 1, hi: third * 2 },
      { label: "late (hard)", lo: third * 2 + 1, hi: TRIAL_COUNT },
    ];
    return stages.map(({ label, lo, hi }) => {
      const matching = completedTrials.filter((t) => t.id >= lo && t.id <= hi);
      const hits = matching.filter((t) => t.reactionTimeMs !== null);
      return {
        band: label,
        shown: matching.length,
        hits: hits.length,
        misses: matching.length - hits.length,
        averageReactionTimeMs: average(hits.map((t) => t.reactionTimeMs as number)),
      };
    });
  }, [completedTrials]);

  // Ordered per-trial reaction-time series for the line chart.
  const reactionSeries = useMemo(() => {
    return completedTrials
      .slice()
      .sort((a, b) => a.id - b.id)
      .map((t) => ({ id: t.id, rt: t.reactionTimeMs }));
  }, [completedTrials]);

  // Headline: which colour / size did the player react fastest on?
  const performanceHeadline = useMemo(() => {
    const colourHits = colorBreakdown.filter((r) => r.averageReactionTimeMs !== null);
    const sizeHits = sizeBreakdown.filter((r) => r.averageReactionTimeMs !== null);
    if (colourHits.length === 0 && sizeHits.length === 0) return null;

    const bestColour = colourHits.length
      ? colourHits.reduce((a, b) =>
          (a.averageReactionTimeMs as number) <= (b.averageReactionTimeMs as number) ? a : b
        )
      : null;
    const bestSize = sizeHits.length
      ? sizeHits.reduce((a, b) =>
          (a.averageReactionTimeMs as number) <= (b.averageReactionTimeMs as number) ? a : b
        )
      : null;

    const parts: string[] = [];
    if (bestSize) parts.push(`${bestSize.band}`);
    if (bestColour) parts.push(`${bestColour.color}`);
    if (parts.length === 0) return null;
    return `You were sharpest on ${parts.join(" ")} dots`;
  }, [colorBreakdown, sizeBreakdown]);

  const canContinueFromConsent =
    participantId.trim().length > 0 && cameraConsent && reactionDataConsent && eyeDataConsent;

  function stopCamera() {
    const video = videoRef.current;
    const stream = video?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
    if (video) video.srcObject = null;
  }

  // Start (or restart) the webcam and the face-tracking loop. Reuses the
  // already-loaded MediaPipe model on subsequent calls (e.g. Play Again).
  async function startCameraAndTracking(): Promise<boolean> {
    try {
      // Lazy-load the MediaPipe model only the first time.
      if (!faceLandmarkerRef.current) {
        setCameraStatus("Loading MediaPipe face model...");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
        );
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
        });
      }

      setCameraStatus("Requesting camera permission...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      const video = videoRef.current;
      if (!video) return false;
      video.srcObject = stream;
      await video.play();

      setCameraEnabled(true);
      setCameraStatus("Camera and face tracking active");
      runFaceTrackingLoop();
      return true;
    } catch (error) {
      console.error(error);
      setCameraStatus("Camera/tracking failed. Use HTTPS, allow camera access, and try again.");
      return false;
    }
  }

  async function acceptConsentAndStartCamera() {
    if (!canContinueFromConsent) return;

    const record: ConsentRecord = {
      consentedAt: new Date().toISOString(),
      participantId: participantId.trim(),
      cameraConsent,
      reactionDataConsent,
      eyeDataConsent,
    };
    setConsentRecord(record);

    const ok = await startCameraAndTracking();
    if (ok) setPhase("ready");
  }

  function runFaceTrackingLoop() {
    const video = videoRef.current;
    const faceLandmarker = faceLandmarkerRef.current;
    const now = performance.now();

    if (video && faceLandmarker && video.readyState >= 2 && now - lastEyeSampleTimeRef.current >= EYE_SAMPLE_INTERVAL_MS) {
      lastEyeSampleTimeRef.current = now;
      const result = faceLandmarker.detectForVideo(video, now);
      const face = result.faceLandmarks?.[0];

      if (!face) {
        eyeSamplesBufferRef.current.push({
          timestamp: now,
          tMs: Math.round(now - gameStartRef.current),
          trialId: currentTrialRef.current?.id ?? null,
          faceDetected: false,
          leftIrisX: null,
          leftIrisY: null,
          rightIrisX: null,
          rightIrisY: null,
          gazeX: null,
          gazeY: null,
          headX: null,
          headY: null,
          blinkLeft: null,
          blinkRight: null,
          isBlink: false,
        });
      } else {
        const leftIris = averagePoint([
          getLandmark(face, 468),
          getLandmark(face, 469),
          getLandmark(face, 470),
          getLandmark(face, 471),
          getLandmark(face, 472),
        ]);

        const rightIris = averagePoint([
          getLandmark(face, 473),
          getLandmark(face, 474),
          getLandmark(face, 475),
          getLandmark(face, 476),
          getLandmark(face, 477),
        ]);

        const noseTip = getLandmark(face, 1);
        const gaze =
          leftIris && rightIris
            ? {
                x: (leftIris.x + rightIris.x) / 2,
                y: (leftIris.y + rightIris.y) / 2,
              }
            : null;

        // Read blink scores from MediaPipe blendshapes (0 open .. 1 closed).
        const blendshapes = result.faceBlendshapes?.[0]?.categories ?? [];
        const blinkLeft =
          blendshapes.find((c) => c.categoryName === "eyeBlinkLeft")?.score ?? null;
        const blinkRight =
          blendshapes.find((c) => c.categoryName === "eyeBlinkRight")?.score ?? null;
        const isBlink =
          (blinkLeft !== null && blinkLeft > 0.5) ||
          (blinkRight !== null && blinkRight > 0.5);

        eyeSamplesBufferRef.current.push({
          timestamp: now,
          tMs: Math.round(now - gameStartRef.current),
          trialId: currentTrialRef.current?.id ?? null,
          faceDetected: true,
          leftIrisX: leftIris?.x ?? null,
          leftIrisY: leftIris?.y ?? null,
          rightIrisX: rightIris?.x ?? null,
          rightIrisY: rightIris?.y ?? null,
          gazeX: gaze?.x ?? null,
          gazeY: gaze?.y ?? null,
          headX: noseTip?.x ?? null,
          headY: noseTip?.y ?? null,
          blinkLeft,
          blinkRight,
          isBlink,
        });
      }
    }

    rafRef.current = requestAnimationFrame(runFaceTrackingLoop);
  }

  async function startGame() {
    setTrials([]);
    setEyeSamples([]);
    eyeSamplesBufferRef.current = [];

    // If we just came from results, the camera was torn down. Bring it back.
    if (!cameraEnabled) {
      const ok = await startCameraAndTracking();
      if (!ok) return;
    }

    gameStartRef.current = performance.now();
    setPhase("playing");
    showTrial(1);
  }

  function showTrial(id: number) {
    if (missTimerRef.current) window.clearTimeout(missTimerRef.current);

    if (id > TRIAL_COUNT) {
      setCurrentTrial(null);
      // flush any buffered eye samples into state before tearing down the camera,
      // so the results heatmap/stats keep all captured gaze data
      if (eyeSamplesBufferRef.current.length > 0) {
        const remaining = eyeSamplesBufferRef.current;
        eyeSamplesBufferRef.current = [];
        setEyeSamples((previous) => [...previous, ...remaining]);
      }
      // stop the face-tracking loop and turn the camera (and its light) off
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      stopCamera();
      setCameraEnabled(false);
      setCameraStatus("Camera off — thanks!");
      setPhase("results");
      return;
    }

    const area = gameAreaRef.current;
    const width = area?.clientWidth ?? window.innerWidth;
    const height = area?.clientHeight ?? window.innerHeight;
    const trial = createTrial(id, width, height);

    setCurrentTrial(trial);

    missTimerRef.current = window.setTimeout(() => {
      setTrials((previous) => [...previous, { ...trial, missed: true }]);
      setCurrentTrial(null);
      window.setTimeout(() => showTrial(id + 1), randomInt(350, 750));
    }, trial.durationMs);
  }

  function handleDotHit(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    if (!currentTrial) return;

    if (missTimerRef.current) window.clearTimeout(missTimerRef.current);

    const clickedAt = performance.now();
    const completedTrial: Trial = {
      ...currentTrial,
      clickedAt,
      reactionTimeMs: Math.round(clickedAt - currentTrial.shownAt),
      missed: false,
    };

    setTrials((previous) => [...previous, completedTrial]);
    setCurrentTrial(null);
    window.setTimeout(() => showTrial(currentTrial.id + 1), randomInt(350, 750));
  }

  function resultPayload() {
    const allEyeSamples = [...eyeSamples, ...eyeSamplesBufferRef.current];
    return {
      appVersion: "1.1.0",
      createdAt: new Date().toISOString(),
      participantId: participantId.trim(),
      userAgent: navigator.userAgent,
      dataUseNote:
        "Game and webcam-derived metrics for aggregate research and engagement only. " +
        "Not a medical or diagnostic assessment. No individual screening conclusions " +
        "should be drawn from a single session.",
      consent: consentRecord,
      config: {
        trialCount: TRIAL_COUNT,
        colors: COLOR_NAMES,
        startDotSize: START_DOT_SIZE,
        endDotSize: END_DOT_SIZE,
        startDurationMs: START_DURATION_MS,
        endDurationMs: END_DURATION_MS,
        eyeSampleIntervalMs: EYE_SAMPLE_INTERVAL_MS,
      },
      summary: {
        trialsCompleted: completedTrials.length,
        clicked: clickedTrials.length,
        missed: completedTrials.length - clickedTrials.length,
        averageReactionTimeMs: averageReactionTime,
        reactionTimeStdDevMs: reactionConsistency,
        fastestReactionTimeMs: fastestTrial?.reactionTimeMs ?? null,
        slowestReactionTimeMs: slowestTrial?.reactionTimeMs ?? null,
        colorBreakdown,
        sizeBreakdown,
        positionBreakdown,
        stageBreakdown,
        performanceHeadline,
        eyeSamples: allEyeSamples.length,
        faceDetectedSamples: allEyeSamples.filter((sample) => sample.faceDetected).length,
        focusScorePct: focusScore,
        headSteadinessPct: headSteadiness,
        gazeRoamPct: gazeRoam,
        gazeTravel,
        paceShiftMs: paceShift,
        gazeEntropyPct: gazeEntropy,
        fixationCount,
        blinkRatePerMin: blinkRate,
        totalBlinks: blinkEvents.length,
      },
      trials,
      eyeSamples: allEyeSamples,
    };
  }

  function downloadAllJson() {
    downloadJson(`reaction-dot-game-${participantId.trim() || "participant"}.json`, resultPayload());
  }

  function downloadTrialsCsv() {
    downloadCsv(
      `reaction-trials-${participantId.trim() || "participant"}.csv`,
      trials.map((trial) => ({
        participantId: participantId.trim(),
        trialId: trial.id,
        x: Math.round(trial.x),
        y: Math.round(trial.y),
        size: trial.size,
        sizeBand: sizeBand(trial.size),
        color: trial.color,
        durationMs: trial.durationMs,
        shownAtMs: Math.round(trial.shownAt),
        clickedAtMs: trial.clickedAt === null ? "" : Math.round(trial.clickedAt),
        reactionTimeMs: trial.reactionTimeMs ?? "",
        missed: trial.missed,
      }))
    );
  }

  function downloadEyeCsv() {
    const allEyeSamples = [...eyeSamples, ...eyeSamplesBufferRef.current];
    downloadCsv(
      `reaction-eye-samples-${participantId.trim() || "participant"}.csv`,
      allEyeSamples.map((sample) => ({
        participantId: participantId.trim(),
        timestampMs: Math.round(sample.timestamp),
        tMs: sample.tMs,
        trialId: sample.trialId ?? "",
        faceDetected: sample.faceDetected,
        leftIrisX: sample.leftIrisX ?? "",
        leftIrisY: sample.leftIrisY ?? "",
        rightIrisX: sample.rightIrisX ?? "",
        rightIrisY: sample.rightIrisY ?? "",
        gazeX: sample.gazeX ?? "",
        gazeY: sample.gazeY ?? "",
        headX: sample.headX ?? "",
        headY: sample.headY ?? "",
        blinkLeft: sample.blinkLeft ?? "",
        blinkRight: sample.blinkRight ?? "",
        isBlink: sample.isBlink,
      }))
    );
  }

  return (
    <main className="appShell">
      <aside className="sidePanel">
        <div className="brandLockup">
          <img src={ndAxonLogo} alt="ND Axon" className="brandLogo" />
          <div className="gameTag">Reaction Challenge</div>
        </div>
        <p className="muted">
          Tap the dots as fast as you can! ⚡ They get smaller and quicker as you go —
          how sharp are your reflexes?
        </p>

        <div className="statGrid">
          <div className="statCard">
            <span>Progress</span>
            <strong>{completedTrials.length}/{TRIAL_COUNT}</strong>
          </div>
          <div className="statCard">
            <span>Average RT</span>
            <strong>{formatMs(averageReactionTime)}</strong>
          </div>
          <div className="statCard">
            <span>Missed</span>
            <strong>{completedTrials.length - clickedTrials.length}</strong>
          </div>
          <div className="statCard">
            <span>Streak</span>
            <strong>{currentStreak}</strong>
          </div>
        </div>

        <p className="cameraStatus">{cameraStatus}</p>

        {phase === "ready" && (
          <button className="primaryButton" onClick={startGame} disabled={!cameraEnabled}>
            Start game
          </button>
        )}

        {phase === "playing" && currentTrial && (
          <div className="liveTrial">
            <span>Current trial</span>
            <strong>#{currentTrial.id}</strong>
            <small>{currentTrial.color}, {currentTrial.size}px, {currentTrial.durationMs}ms</small>
          </div>
        )}
      </aside>

      <section className={`stage stage-${phase}`} ref={gameAreaRef}>
        <video
          ref={videoRef}
          className="cameraPreview"
          playsInline
          muted
          style={{ display: phase === "results" ? "none" : undefined }}
        />

        {phase === "consent" && (
          <div className="overlayCard consentCard">
            <img src={ndAxonLogo} alt="ND Axon" className="consentLogo" />
            <h2>Ready to test your reflexes? 🎯</h2>
            <p>
              This is a fun reaction-time game, not a test or a medical assessment.
              With your permission, the camera turns on and the app measures face and
              eye position locally to show how attention research works. We record your
              reaction times and these eye/face coordinates — never raw video.
            </p>

            <label className="fieldLabel">
              Participant ID
              <input
                value={participantId}
                onChange={(event) => setParticipantId(event.target.value)}
                placeholder="Example: P001"
              />
            </label>

            <label className="checkRow">
              <input
                type="checkbox"
                checked={cameraConsent}
                onChange={(event) => setCameraConsent(event.target.checked)}
              />
              I agree to turn on my camera for this activity.
            </label>

            <label className="checkRow">
              <input
                type="checkbox"
                checked={reactionDataConsent}
                onChange={(event) => setReactionDataConsent(event.target.checked)}
              />
              I agree for my reaction-time game data to be collected.
            </label>

            <label className="checkRow">
              <input
                type="checkbox"
                checked={eyeDataConsent}
                onChange={(event) => setEyeDataConsent(event.target.checked)}
              />
              I agree for face and eye landmark data to be collected.
            </label>

            <button
              className="primaryButton"
              onClick={acceptConsentAndStartCamera}
              disabled={!canContinueFromConsent}
            >
              Give permission and continue
            </button>
          </div>
        )}

        {phase === "ready" && (
          <div className="overlayCard">
            <h2>Ready</h2>
            <p>Keep your face visible in the camera preview. Press Start game.</p>
          </div>
        )}

        {phase === "playing" && currentTrial && (
          <button
            className="dot"
            aria-label="reaction target"
            onPointerDown={handleDotHit}
            style={{
              left: currentTrial.x,
              top: currentTrial.y,
              width: currentTrial.size,
              height: currentTrial.size,
              backgroundColor: DOT_COLORS[currentTrial.color],
            }}
          />
        )}

        {phase === "results" && (
          <div className="overlayCard resultsCard">
            <div className="resultBadge">
              <span className="resultBadgeEmoji">{reactionTier.emoji}</span>
              <h2>{reactionTier.label}</h2>
            </div>

            <div className="resultHero">
              <div>
                <span>Average reaction time</span>
                <strong>{formatMs(averageReactionTime)}</strong>
              </div>
              <div>
                <span>Fastest hit</span>
                <strong>{formatMs(fastestTrial?.reactionTimeMs ?? null)}</strong>
              </div>
              <div>
                <span title="How steady your speed was from dot to dot. Lower means more consistent.">Consistency (±)</span>
                <strong>{formatMs(reactionConsistency)}</strong>
              </div>
              <div>
                <span>Accuracy</span>
                <strong>{completedTrials.length ? Math.round((clickedTrials.length / completedTrials.length) * 100) : 0}%</strong>
              </div>
            </div>

            {performanceHeadline && (
              <div className="perfHeadline">
                <span className="perfHeadlineIcon">🎯</span>
                <span>{performanceHeadline}</span>
              </div>
            )}

            <div className="vizSection">
              <h3>Your reaction over the game 📈</h3>
              <p className="eyeIntro">
                Each point is one dot you hit. Green means faster than your average,
                amber means slower. Watch how you held up as the dots got smaller and quicker.
              </p>
              <ReactionLineChart data={reactionSeries} />
            </div>

            <div className="vizSection">
              <h3>What you were quickest on ⚡</h3>
              <p className="eyeIntro">
                Average reaction time broken down by dot size, where it appeared, and
                when in the game it showed up. Shorter bars (green) are faster.
              </p>

              <h4 className="vizSubhead">By dot size</h4>
              <BarChart rows={sizeBreakdown.map((r) => ({ label: r.band, value: r.averageReactionTimeMs }))} />

              <h4 className="vizSubhead">By screen position</h4>
              <BarChart rows={positionBreakdown.map((r) => ({ label: r.band, value: r.averageReactionTimeMs }))} />

              <h4 className="vizSubhead">By game stage</h4>
              <BarChart rows={stageBreakdown.map((r) => ({ label: r.band, value: r.averageReactionTimeMs }))} />

              <h4 className="vizSubhead">By colour</h4>
              <BarChart rows={colorBreakdown.map((r) => ({ label: r.color, value: r.averageReactionTimeMs }))} />
            </div>

            <div className="domainsIntro">
              <h3>Your behavioural breakdown 🧠</h3>
              <p className="eyeIntro">
                ND Axon studies attention through a few behavioural "domains." Here's a
                playful, just-for-fun look at yours, built from your camera and gameplay.
              </p>
            </div>

            {/* DOMAIN 1 — Attention Organisation */}
            <div className="eyeSection">
              <div className="domainHead">
                <span className="domainTag">Attention Organisation</span>
                <h4>Where your eyes went 👀</h4>
              </div>
              <p className="eyeIntro">
                An attention-density map of where you looked — yellow shows where your
                eyes concentrated most, fading to deep blue where they rarely went.
              </p>
              <canvas ref={heatmapRef} className="heatmap" width={760} height={428} />
              <div className="eyeStats eyeStatsWide">
                <div>
                  <span title="How spread-out your gaze was. Higher = eyes roamed everywhere.">Gaze spread</span>
                  <strong>{gazeEntropy === null ? "--" : `${gazeEntropy}%`}</strong>
                </div>
                <div>
                  <span title="Number of moments your gaze settled and held still.">Fixations</span>
                  <strong>{fixationCount === null ? "--" : fixationCount}</strong>
                </div>
                <div>
                  <span title="How widely your eyes swept across the screen.">Gaze roam</span>
                  <strong>{gazeRoam === null ? "--" : `${gazeRoam}%`}</strong>
                </div>
                <div>
                  <span title="How much of the time your face stayed pointed at the screen.">Focus</span>
                  <strong>{focusScore === null ? "--" : `${focusScore}%`}</strong>
                </div>
              </div>
            </div>

            {/* DOMAIN 2 — Arousal Regulation (blinks) */}
            <div className="eyeSection">
              <div className="domainHead">
                <span className="domainTag">Arousal Regulation</span>
                <h4>Your blinks over time 😌</h4>
              </div>
              <p className="eyeIntro">
                The blue line is how open your eyes were through the game; each pink mark
                is a blink. Even, rhythmic blinking looks like a steady wave.
              </p>
              <BlinkChart data={blinkSeries} events={blinkEvents} />
              <div className="eyeStats eyeStatsWide">
                <div>
                  <span title="How many times you blinked per minute. Resting humans average ~15-20.">Blink rate</span>
                  <strong>{blinkRate === null ? "--" : `${blinkRate}/min`}</strong>
                </div>
                <div>
                  <span title="Total blinks detected during the game.">Total blinks</span>
                  <strong>{blinkEvents.length}</strong>
                </div>
                <div>
                  <span title="How still you kept your head. 100 = rock steady.">Steadiness</span>
                  <strong>{headSteadiness === null ? "--" : `${headSteadiness}%`}</strong>
                </div>
                <div>
                  <span title="How busy your eyes were overall.">Eye busyness</span>
                  <strong>{gazeTravel === null ? "--" : gazeTravel}</strong>
                </div>
              </div>
            </div>

            {/* DOMAIN 3 — Temporal Stability */}
            <div className="eyeSection">
              <div className="domainHead">
                <span className="domainTag">Temporal Stability</span>
                <h4>Did you hold steady? ⏱️</h4>
              </div>
              <p className="eyeIntro">
                How consistent you stayed from the start of the game to the end.
              </p>
              <div className="eyeStats">
                <div>
                  <span title="Lower = your reaction speed barely wavered.">RT consistency (±)</span>
                  <strong>{formatMs(reactionConsistency)}</strong>
                </div>
                <div>
                  <span title="Difference between your second half and first half.">Pace shift</span>
                  <strong>{paceShift === null ? "--" : `${paceShift > 0 ? "+" : ""}${paceShift}ms`}</strong>
                </div>
                <div>
                  <span>Accuracy</span>
                  <strong>{completedTrials.length ? Math.round((clickedTrials.length / completedTrials.length) * 100) : 0}%</strong>
                </div>
              </div>
              {paceShift !== null && (
                <p className="paceLine">
                  {paceShift < -15
                    ? `🔥 You sped up as the game went on — ${Math.abs(paceShift)}ms faster in the second half!`
                    : paceShift > 15
                    ? `🧘 You eased off a little near the end — ${paceShift}ms slower in the second half.`
                    : `⚖️ Rock steady — your pace barely changed from start to finish.`}
                </p>
              )}
            </div>

            <p className="domainDisclaimer">
              These are playful, illustrative readings from an uncalibrated webcam — not a
              clinical measure of anything. They show the <em>kind</em> of signals ND Axon
              works with, not a result about you.
            </p>

            <div className="ctaCard">
              <p>
                Reaction-time <strong>consistency</strong> — how steady you are dot to dot —
                is one of the most studied signals in attention research. It's exactly the
                kind of pattern ND Axon works with. Curious about the real thing?
              </p>
              <div className="ctaButtons">
                <a className="ctaButton" href={APP_STORE_URL} target="_blank" rel="noreferrer">
                  Get it on the App Store
                </a>
                <a className="ctaButton" href={PLAY_STORE_URL} target="_blank" rel="noreferrer">
                  Get it on Google Play
                </a>
              </div>
            </div>

            <p className="resultDisclaimer">
              This is a game, not a test or a diagnosis. Your scores are just for fun.
            </p>

            <div className="downloadRow">
              <button className="primaryButton" onClick={startGame}>Play again</button>
            </div>

            <details className="organizerDetails">
              <summary>Organizer details &amp; raw data export</summary>

              <h3>By colour</h3>
              <div className="tableLike">
                {colorBreakdown.map((row) => (
                  <div className="tableRow" key={row.color}>
                    <span className="colorDot" style={{ backgroundColor: DOT_COLORS[row.color] }} />
                    <strong>{row.color}</strong>
                    <span>shown {row.shown}</span>
                    <span>hits {row.hits}</span>
                    <span>misses {row.misses}</span>
                    <span>avg {formatMs(row.averageReactionTimeMs)}</span>
                  </div>
                ))}
              </div>

              <h3>By size</h3>
              <div className="tableLike">
                {sizeBreakdown.map((row) => (
                  <div className="tableRow" key={row.band}>
                    <strong>{row.band}</strong>
                    <span>shown {row.shown}</span>
                    <span>hits {row.hits}</span>
                    <span>misses {row.misses}</span>
                    <span>avg {formatMs(row.averageReactionTimeMs)}</span>
                  </div>
                ))}
              </div>

              <h3>By position</h3>
              <div className="tableLike">
                {positionBreakdown.map((row) => (
                  <div className="tableRow" key={row.band}>
                    <strong>{row.band}</strong>
                    <span>shown {row.shown}</span>
                    <span>hits {row.hits}</span>
                    <span>misses {row.misses}</span>
                    <span>avg {formatMs(row.averageReactionTimeMs)}</span>
                  </div>
                ))}
              </div>

              <h3>By game stage</h3>
              <div className="tableLike">
                {stageBreakdown.map((row) => (
                  <div className="tableRow" key={row.band}>
                    <strong>{row.band}</strong>
                    <span>shown {row.shown}</span>
                    <span>hits {row.hits}</span>
                    <span>misses {row.misses}</span>
                    <span>avg {formatMs(row.averageReactionTimeMs)}</span>
                  </div>
                ))}
              </div>

              <div className="downloadRow">
                <button onClick={downloadAllJson}>Download all JSON</button>
                <button onClick={downloadTrialsCsv}>Download trials CSV</button>
                <button onClick={downloadEyeCsv}>Download eye CSV</button>
              </div>
            </details>
          </div>
        )}
      </section>
    </main>
  );
}
