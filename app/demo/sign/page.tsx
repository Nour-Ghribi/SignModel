"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

type Landmark = { x: number; y: number; z: number };
type Mode = "A" | "C";

type SignModelAssets = {
  labels: string[];
  mean: number[];
  std: number[];
};

type WeightSpec = {
  name: string;
  shape: number[];
  dtype: string;
};

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const NUM_HANDS = 2;
const LANDMARKS_PER_HAND = 21;
const COORDS_PER_LANDMARK = 3;
const FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK;
const SEQUENCE_LENGTH = 16;
const DETECT_INTERVAL_MS = 50;
const PREDICT_INTERVAL_MS = 150;

const dist = (a: Landmark, b: Landmark) => {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

function classifyRuleBased(landmarks: Landmark[]) {
  const palm = landmarks[0];
  const tips = [4, 8, 12, 16, 20].map((i) => landmarks[i]);
  const palmDists = tips.map((tip) => dist(tip, palm));
  const avgPalmDist = palmDists.reduce((sum, value) => sum + value, 0) / palmDists.length;

  const indexExtended = dist(landmarks[8], landmarks[5]) > 0.22;
  const middleFolded = dist(landmarks[12], landmarks[9]) < 0.18;
  const ringFolded = dist(landmarks[16], landmarks[13]) < 0.17;
  const pinkyFolded = dist(landmarks[20], landmarks[17]) < 0.16;

  if (avgPalmDist > 0.3) return "BONJOUR";
  if (avgPalmDist < 0.1) return "STOP";
  if (indexExtended && middleFolded && ringFolded && pinkyFolded) return "AIDE";
  return "...";
}

function normalizeHand(landmarks: Landmark[]) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale = Math.max(dist(wrist, middleMcp), 1e-6);
  const values: number[] = [];

  landmarks.forEach((point) => {
    values.push((point.x - wrist.x) / scale);
    values.push((point.y - wrist.y) / scale);
    values.push((point.z - wrist.z) / scale);
  });

  return values;
}

function buildFrameFeatures(allHands: Landmark[][]) {
  const ordered = [...allHands].sort((a, b) => a[0].x - b[0].x).slice(0, NUM_HANDS);
  const values: number[] = [];

  ordered.forEach((hand) => values.push(...normalizeHand(hand)));
  while (values.length < FEATURES_PER_FRAME) {
    values.push(0);
  }

  return values.slice(0, FEATURES_PER_FRAME);
}

function buildSignModel() {
  const input = tf.input({ shape: [SEQUENCE_LENGTH, FEATURES_PER_FRAME] });
  let x = tf.layers
    .conv1d({ name: "conv1d", filters: 64, kernelSize: 3, padding: "same", activation: "relu" })
    .apply(input) as tf.SymbolicTensor;
  x = tf.layers.batchNormalization({ name: "batch_normalization" }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.maxPooling1d({ name: "max_pooling1d", poolSize: 2 }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.dropout({ name: "dropout", rate: 0.2 }).apply(x) as tf.SymbolicTensor;

  x = tf.layers
    .conv1d({ name: "conv1d_1", filters: 128, kernelSize: 3, padding: "same", activation: "relu" })
    .apply(x) as tf.SymbolicTensor;
  x = tf.layers.batchNormalization({ name: "batch_normalization_1" }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.maxPooling1d({ name: "max_pooling1d_1", poolSize: 2 }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.dropout({ name: "dropout_1", rate: 0.25 }).apply(x) as tf.SymbolicTensor;

  x = tf.layers
    .conv1d({ name: "conv1d_2", filters: 128, kernelSize: 3, padding: "same", activation: "relu" })
    .apply(x) as tf.SymbolicTensor;
  x = tf.layers.batchNormalization({ name: "batch_normalization_2" }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.globalAveragePooling1d({ name: "global_average_pooling1d" }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.dense({ name: "dense", units: 128, activation: "relu" }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.dropout({ name: "dropout_2", rate: 0.3 }).apply(x) as tf.SymbolicTensor;
  const output = tf.layers.dense({ name: "dense_1", units: 20, activation: "softmax" }).apply(x) as tf.SymbolicTensor;

  return tf.model({ inputs: input, outputs: output });
}

async function loadWeightMap(specs: WeightSpec[], binUrl: string) {
  const res = await fetch(binUrl);
  if (!res.ok) {
    throw new Error("Failed to load model weights.");
  }

  const buffer = await res.arrayBuffer();
  const values = new Float32Array(buffer);
  const tensors = new Map<string, tf.Tensor>();
  let offset = 0;

  specs.forEach((spec) => {
    const size = spec.shape.reduce((acc, value) => acc * value, 1);
    const slice = values.slice(offset, offset + size);
    offset += size;
    tensors.set(spec.name, tf.tensor(slice, spec.shape, "float32"));
  });

  return tensors;
}

export default function SignDemoPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const tfModelRef = useRef<tf.LayersModel | null>(null);
  const modelAssetsRef = useRef<SignModelAssets | null>(null);
  const sequenceBufferRef = useRef<number[][]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastDetectMsRef = useRef(0);
  const lastPredictMsRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const cameraOnRef = useRef(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("Camera off");
  const [error, setError] = useState("");
  const [handDetected, setHandDetected] = useState(false);
  const [subtitle, setSubtitle] = useState("...");
  const [mode, setMode] = useState<Mode>("C");
  const [botReply, setBotReply] = useState("HandiBot reply will appear here.");
  const [botBusy, setBotBusy] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  const handStatus = useMemo(() => (handDetected ? "Hand detected" : "No hand"), [handDetected]);

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const width = Math.floor(video.clientWidth);
    const height = Math.floor(video.clientHeight);
    if (width > 0 && height > 0 && (canvas.width !== width || canvas.height !== height)) {
      canvas.width = width;
      canvas.height = height;
    }
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const drawHands = useCallback((allHands: Landmark[][]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#3D3D9E";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#E91E8C";

    allHands.forEach((landmarks) => {
      HAND_CONNECTIONS.forEach(([a, b]) => {
        const p1 = landmarks[a];
        const p2 = landmarks[b];
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      });

      landmarks.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    sequenceBufferRef.current = [];
    cameraOnRef.current = false;
    setCameraOn(false);
    setHandDetected(false);
    setSubtitle("...");
    setStatus("Camera off");
    clearCanvas();
  }, [clearCanvas]);

  const predictSequence = useCallback(() => {
    if (sequenceBufferRef.current.length < SEQUENCE_LENGTH) return;
    if (!tfModelRef.current || !modelAssetsRef.current) return;

    const now = performance.now();
    if (now - lastPredictMsRef.current < PREDICT_INTERVAL_MS) return;
    lastPredictMsRef.current = now;

    const assets = modelAssetsRef.current;
    const flatMean = assets.mean;
    const flatStd = assets.std;
    const sequence = sequenceBufferRef.current.slice(-SEQUENCE_LENGTH);

    const result = tf.tidy(() => {
      const input = tf.tensor3d([sequence], [1, SEQUENCE_LENGTH, FEATURES_PER_FRAME], "float32");
      const mean = tf.tensor3d(flatMean, [1, 1, FEATURES_PER_FRAME], "float32");
      const std = tf.tensor3d(flatStd, [1, 1, FEATURES_PER_FRAME], "float32");
      const normalized = input.sub(mean).div(std);
      const output = tfModelRef.current!.predict(normalized) as tf.Tensor;
      const probabilities = output.dataSync();
      let bestIndex = 0;
      let bestScore = -1;
      for (let i = 0; i < probabilities.length; i += 1) {
        if (probabilities[i] > bestScore) {
          bestScore = probabilities[i];
          bestIndex = i;
        }
      }
      return {
        label: assets.labels[bestIndex] ?? "...",
        score: bestScore,
      };
    });

    setSubtitle(result.score >= 0.35 ? `${result.label.toUpperCase()} (${(result.score * 100).toFixed(0)}%)` : "...");
  }, []);

  const detectLoop = useCallback(() => {
    const video = videoRef.current;
    const detector = handLandmarkerRef.current;
    if (!video || !detector || !cameraOnRef.current) return;

    const now = performance.now();
    if (now - lastDetectMsRef.current >= DETECT_INTERVAL_MS) {
      lastDetectMsRef.current = now;
      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const result = detector.detectForVideo(video, now);
        const allHands = (result.landmarks ?? []) as Landmark[][];

        if (allHands.length > 0) {
          setHandDetected(true);
          drawHands(allHands);

          const frameFeatures = buildFrameFeatures(allHands);
          const nextSequence = [...sequenceBufferRef.current, frameFeatures].slice(-SEQUENCE_LENGTH);
          sequenceBufferRef.current = nextSequence;

          if (mode === "A") {
            setSubtitle(allHands.map((hand) => classifyRuleBased(hand)).join(" | "));
          } else {
            predictSequence();
          }
        } else {
          setHandDetected(false);
          sequenceBufferRef.current = [];
          clearCanvas();
          setSubtitle("...");
        }
      }
    }

    rafRef.current = requestAnimationFrame(detectLoop);
  }, [clearCanvas, drawHands, mode, predictSequence]);

  const initHandLandmarker = useCallback(async () => {
    if (handLandmarkerRef.current) return;
    setStatus("Loading hand tracker...");

    const vision = await FilesetResolver.forVisionTasks("/wasm");
    try {
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "/wasm/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
    } catch {
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "/wasm/hand_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
    }
  }, []);

  const initSignModel = useCallback(async () => {
    if (tfModelRef.current && modelAssetsRef.current) return;
    const [modelRes, labelsRes, normRes] = await Promise.all([
      fetch("/models/sign-cnn/model.json"),
      fetch("/models/sign-cnn/labels.json"),
      fetch("/models/sign-cnn/normalization.json"),
    ]);

    if (!modelRes.ok || !labelsRes.ok || !normRes.ok) {
      throw new Error("Failed to load model assets.");
    }

    await tf.ready();
    const modelJson = await modelRes.json();
    const labelsJson = await labelsRes.json();
    const normJson = await normRes.json();
    const specs = (modelJson.weightsManifest?.[0]?.weights ?? []) as WeightSpec[];
    const shardPath = modelJson.weightsManifest?.[0]?.paths?.[0];

    if (!specs.length || !shardPath) {
      throw new Error("Invalid model manifest.");
    }

    const model = buildSignModel();
    const tensorMap = await loadWeightMap(specs, `/models/sign-cnn/${shardPath}`);
    const namedLayers = [
      "conv1d",
      "batch_normalization",
      "conv1d_1",
      "batch_normalization_1",
      "conv1d_2",
      "batch_normalization_2",
      "dense",
      "dense_1",
    ];

    namedLayers.forEach((layerName) => {
      const layer = model.getLayer(layerName);
      const weightNames = layer.weights.map((weight) => {
        const path = (weight as unknown as { originalName?: string; name: string }).originalName ?? weight.name;
        return path.replace(/.*\//, `${layerName}/`).replace(/:0$/, "");
      });
      const layerWeights = weightNames.map((name) => {
        const tensor = tensorMap.get(name);
        if (!tensor) {
          throw new Error(`Missing weight: ${name}`);
        }
        return tensor;
      });
      layer.setWeights(layerWeights);
    });
    tensorMap.forEach((tensor) => tensor.dispose());

    tfModelRef.current = model;
    modelAssetsRef.current = {
      labels: labelsJson.labels,
      mean: normJson.mean[0][0],
      std: normJson.std[0][0],
    };
    setModelReady(true);
  }, []);

  const startCamera = useCallback(async () => {
    if (cameraOnRef.current || streamRef.current) return;

    try {
      setError("");
      await Promise.all([initHandLandmarker(), initSignModel()]);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      syncCanvasSize();

      sequenceBufferRef.current = [];
      cameraOnRef.current = true;
      setCameraOn(true);
      setStatus(modelReady ? "Camera on" : "Camera on / model loading");
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(detectLoop);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Camera/model error";
      setError(message);
      setStatus("Error");
      cameraOnRef.current = false;
      setCameraOn(false);
    }
  }, [detectLoop, initHandLandmarker, initSignModel, modelReady, syncCanvasSize]);

  const sendToHandibot = useCallback(async () => {
    if (!subtitle || subtitle === "...") return;
    setBotBusy(true);
    try {
      const res = await fetch("/api/handibot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: subtitle }),
      });
      const data = await res.json();
      setBotReply(data.reply ?? "Empty reply from bot.");
    } catch {
      setBotReply("Network error while calling HandiBot.");
    } finally {
      setBotBusy(false);
    }
  }, [subtitle]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const observer = new ResizeObserver(() => syncCanvasSize());
    observer.observe(video);

    initSignModel().catch((err) => {
      const message = err instanceof Error ? err.message : "Model load error";
      setError(message);
    });

    return () => {
      observer.disconnect();
      stopCamera();
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
      tfModelRef.current?.dispose();
      tfModelRef.current = null;
      modelAssetsRef.current = null;
    };
  }, [initSignModel, stopCamera, syncCanvasSize]);

  return (
    <main className="min-h-screen bg-slate-100 p-4 md:p-6">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 md:flex-row">
        <section className="md:w-3/5">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <button
              onClick={startCamera}
              className="rounded px-4 py-2 text-white"
              style={{ backgroundColor: "#3D3D9E" }}
            >
              Start Camera
            </button>
            <button
              onClick={stopCamera}
              className="rounded px-4 py-2 text-white"
              style={{ backgroundColor: "#E91E8C" }}
            >
              Stop Camera
            </button>
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value as Mode)}
              className="rounded border px-3 py-2"
            >
              <option value="C">Mode C (1D-CNN sign model)</option>
              <option value="A">Mode A (rules fallback)</option>
            </select>
            <span className="rounded bg-white px-3 py-2 text-sm">{status}</span>
            <span className="rounded bg-white px-3 py-2 text-sm">{handStatus}</span>
            <span className="rounded bg-white px-3 py-2 text-sm">{cameraOn ? "Camera ON" : "Camera OFF"}</span>
            <span className="rounded bg-white px-3 py-2 text-sm">{modelReady ? "Model READY" : "Model LOADING"}</span>
          </div>

          <div className="relative overflow-hidden rounded-xl bg-black">
            <video ref={videoRef} className="h-auto w-full" playsInline muted />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
            <div className="absolute bottom-0 left-0 w-full bg-black/60 px-3 py-2 text-center text-lg font-semibold text-white">
              {subtitle}
            </div>
          </div>

          {error && <p className="mt-3 rounded bg-red-100 p-2 text-sm text-red-700">{error}</p>}
        </section>

        <aside className="md:w-2/5">
          <div className="rounded-xl bg-white p-4 shadow">
            <h2 className="mb-3 text-lg font-bold" style={{ color: "#3D3D9E" }}>
              HandiBot
            </h2>
            <div className="mb-4 flex items-center gap-3">
              <div
                className={`h-12 w-12 rounded-full ${botBusy ? "animate-pulse" : ""}`}
                style={{ background: "linear-gradient(135deg, #E91E8C, #3D3D9E)" }}
                aria-label="avatar"
              />
              <div className="max-w-[85%] rounded-2xl bg-slate-100 px-3 py-2 text-sm">{botReply}</div>
            </div>
            <button
              onClick={sendToHandibot}
              className="w-full rounded px-4 py-2 font-semibold text-white"
              style={{ backgroundColor: "#E91E8C" }}
            >
              Send to HandiBot
            </button>
          </div>
        </aside>
      </div>
    </main>
  );
}
