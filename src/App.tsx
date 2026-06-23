import { useEffect, useMemo, useRef, useState } from "react";
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
const MIN_DOT_SIZE = 28;
const MAX_DOT_SIZE = 88;
const MIN_DURATION_MS = 750;
const MAX_DURATION_MS = 1800;
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
  const size = randomInt(MIN_DOT_SIZE, MAX_DOT_SIZE);
  const margin = size + 18;

  return {
    id,
    x: randomBetween(margin, Math.max(margin, width - margin)),
    y: randomBetween(margin, Math.max(margin, height - margin)),
    size,
    color: pickOne(COLOR_NAMES),
    durationMs: randomInt(MIN_DURATION_MS, MAX_DURATION_MS),
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

  function handleDotClick() {
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
      appVersion: "1.0.0",
      createdAt: new Date().toISOString(),
      participantId: participantId.trim(),
      userAgent: navigator.userAgent,
      consent: consentRecord,
      config: {
        trialCount: TRIAL_COUNT,
        colors: COLOR_NAMES,
        minDotSize: MIN_DOT_SIZE,
        maxDotSize: MAX_DOT_SIZE,
        minDurationMs: MIN_DURATION_MS,
        maxDurationMs: MAX_DURATION_MS,
        eyeSampleIntervalMs: EYE_SAMPLE_INTERVAL_MS,
      },
      summary: {
        trialsCompleted: completedTrials.length,
        clicked: clickedTrials.length,
        missed: completedTrials.length - clickedTrials.length,
        averageReactionTimeMs: averageReactionTime,
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
          Tap or click the red, green, blue, and yellow dots as quickly as possible.
          The camera is used to estimate face and eye landmarks during the task.
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
            <span>Eye samples</span>
            <strong>{eyeSamples.length + eyeSamplesBufferRef.current.length}</strong>
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
              This app asks for camera permission, processes video locally with MediaPipe,
              and saves reaction-time plus landmark-derived eye/face coordinates in the browser.
              This starter app does not upload raw video or results to a server.
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
            onClick={handleDotClick}
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
            <h2>Results breakdown</h2>
            <div className="resultHero">
              <div>
                <span>Average reaction time</span>
                <strong>{formatMs(averageReactionTime)}</strong>
              </div>
              <div>
                <span>Fastest</span>
                <strong>{formatMs(fastestTrial?.reactionTimeMs ?? null)}</strong>
              </div>
              <div>
                <span>Slowest</span>
                <strong>{formatMs(slowestTrial?.reactionTimeMs ?? null)}</strong>
              </div>
              <div>
                <span>Accuracy</span>
                <strong>{completedTrials.length ? Math.round((clickedTrials.length / completedTrials.length) * 100) : 0}%</strong>
              </div>
            </div>

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
              <button onClick={startGame}>Run again</button>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
