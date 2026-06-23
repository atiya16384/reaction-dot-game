import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";

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
const START_DOT_SIZE = 110; // big, forgiving first dots
const END_DOT_SIZE = 46; // small but still hittable final dots
const START_DURATION_MS = 1700; // plenty of time early on
const END_DURATION_MS = 950; // quick, but reachable near the end
const SIZE_JITTER = 16; // random wobble so it's not perfectly predictable
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

  const baseSize = START_DOT_SIZE + (END_DOT_SIZE - START_DOT_SIZE) * progress;
  const size = Math.max(
    18,
    Math.round(baseSize + randomBetween(-SIZE_JITTER, SIZE_JITTER))
  );
  const durationMs = Math.round(
    START_DURATION_MS + (END_DURATION_MS - START_DURATION_MS) * progress
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

  // Paint a gaze-density map (like the research "spatial attention" figures):
  // bin gaze into a grid, smooth it, then colour by concentration.
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

    // lowest viridis colour as the backdrop
    ctx.fillStyle = `rgb(${VIRIDIS[0][0]}, ${VIRIDIS[0][1]}, ${VIRIDIS[0][2]})`;
    ctx.fillRect(0, 0, width, height);

    if (gazePoints.length < 3) {
      ctx.fillStyle = "#a9c0c6";
      ctx.font = "14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Not enough gaze data captured this round", width / 2, height / 2);
      return;
    }

    // 1) accumulate gaze into a coarse grid with a soft splat around each point
    const COLS = 64;
    const ROWS = 36;
    const grid = new Float32Array(COLS * ROWS);
    const splat = 2; // spread each gaze sample over neighbouring cells
    for (const point of gazePoints) {
      // gaze x is mirrored (selfie view) -> flip to match what the player saw
      const gx = (1 - point.x) * (COLS - 1);
      const gy = point.y * (ROWS - 1);
      const cx = Math.round(gx);
      const cy = Math.round(gy);
      for (let oy = -splat; oy <= splat; oy += 1) {
        for (let ox = -splat; ox <= splat; ox += 1) {
          const x = cx + ox;
          const y = cy + oy;
          if (x < 0 || x >= COLS || y < 0 || y >= ROWS) continue;
          const dist2 = ox * ox + oy * oy;
          grid[y * COLS + x] += Math.exp(-dist2 / 2.2);
        }
      }
    }

    // 2) normalise
    let max = 0;
    for (let i = 0; i < grid.length; i += 1) if (grid[i] > max) max = grid[i];
    if (max === 0) return;

    // 3) paint cells, upscaled to canvas, with bilinear-ish smoothing via cell size
    const cellW = width / COLS;
    const cellH = height / ROWS;
    for (let y = 0; y < ROWS; y += 1) {
      for (let x = 0; x < COLS; x += 1) {
        const v = grid[y * COLS + x] / max;
        if (v <= 0.02) continue; // leave backdrop showing for empty areas
        const [r, g, b] = colorAt(v);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.85 * Math.min(1, v + 0.15)})`;
        ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
      }
    }

    // 4) gentle blur pass to turn blocks into smooth contours
    if (typeof ctx.filter === "string") {
      const snapshot = ctx.getImageData(0, 0, width, height);
      ctx.putImageData(snapshot, 0, 0);
      ctx.filter = "blur(8px)";
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

  const canContinueFromConsent =
    participantId.trim().length > 0 && cameraConsent && reactionDataConsent && eyeDataConsent;

  function stopCamera() {
    const video = videoRef.current;
    const stream = video?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
    if (video) video.srcObject = null;
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
    setCameraStatus("Loading MediaPipe face model...");

    try {
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
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });

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
      if (!video) return;
      video.srcObject = stream;
      await video.play();

      setCameraEnabled(true);
      setCameraStatus("Camera and face tracking active");
      setPhase("ready");
      runFaceTrackingLoop();
    } catch (error) {
      console.error(error);
      setCameraStatus("Camera/tracking failed. Use HTTPS, allow camera access, and try again.");
    }
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

        eyeSamplesBufferRef.current.push({
          timestamp: now,
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
        });
      }
    }

    rafRef.current = requestAnimationFrame(runFaceTrackingLoop);
  }

  function startGame() {
    setTrials([]);
    setEyeSamples([]);
    eyeSamplesBufferRef.current = [];
    setPhase("playing");
    showTrial(1);
  }

  function showTrial(id: number) {
    if (missTimerRef.current) window.clearTimeout(missTimerRef.current);

    if (id > TRIAL_COUNT) {
      setCurrentTrial(null);
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
        eyeSamples: allEyeSamples.length,
        faceDetectedSamples: allEyeSamples.filter((sample) => sample.faceDetected).length,
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
      }))
    );
  }

  return (
    <main className="appShell">
      <aside className="sidePanel">
        <h1>Reaction Dot Game</h1>
        <p className="muted">
          Tap the dots as fast as you can. Different colours, sizes, and speeds —
          see how quick you really are.
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

      <section className="stage" ref={gameAreaRef}>
        <video ref={videoRef} className="cameraPreview" playsInline muted />

        {phase === "consent" && (
          <div className="overlayCard consentCard">
            <h2>Before you begin</h2>
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

            <div className="eyeSection">
              <h3>Where your eyes went 👀</h3>
              <p className="eyeIntro">
                While you played, the camera tracked your gaze. This is a
                attention-density map of where you looked — yellow shows where your
                eyes concentrated most, fading to dark where they rarely went.
              </p>
              <canvas ref={heatmapRef} className="heatmap" width={760} height={428} />
              <div className="eyeStats">
                <div>
                  <span title="How much of the time your face stayed pointed at the screen.">Focus</span>
                  <strong>{focusScore === null ? "--" : `${focusScore}%`}</strong>
                </div>
                <div>
                  <span title="How widely your eyes swept across the screen.">Gaze roam</span>
                  <strong>{gazeRoam === null ? "--" : `${gazeRoam}%`}</strong>
                </div>
                <div>
                  <span>Gaze points</span>
                  <strong>{gazePoints.length}</strong>
                </div>
              </div>
            </div>

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
