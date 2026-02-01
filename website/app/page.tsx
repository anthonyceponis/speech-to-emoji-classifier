"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";

// Emoji Mapping
const KEYWORD_TO_EMOJI: Record<string, string> = {
  down: "⬇️",
  go: "🟢",
  left: "⬅️",
  no: "❌",
  right: "➡️",
  stop: "🛑",
  up: "⬆️",
  yes: "✅",
};

export default function AudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [audioSamples, setAudioSamples] = useState<number[] | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);

  // Audio & UI Refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const progressBarRef = useRef<HTMLDivElement>(null); // Ref for the progress bar
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null); // Ref to track start time

  useEffect(() => {
    return () => {
      cancelAnimation();
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, []);

  const cancelAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    // Reset progress bar to empty when stopped
    if (progressBarRef.current) {
      progressBarRef.current.style.width = "0%";
    }
  };

  const drawWaveform = () => {
    if (!analyserRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    if (!canvasCtx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);

      // --- 1. Update Progress Bar ---
      if (startTimeRef.current && progressBarRef.current) {
        const elapsed = Date.now() - startTimeRef.current;
        const duration = 1000; // 1 second
        const remainingPercentage = Math.max(0, 100 - (elapsed / duration) * 100);
        progressBarRef.current.style.width = `${remainingPercentage}%`;
      }

      // --- 2. Draw Waveform ---
      analyser.getByteTimeDomainData(dataArray);

      canvasCtx.fillStyle = "rgb(24 24 27)"; // bg-zinc-950
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = "rgb(45 212 191)"; // teal-400
      canvasCtx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
    };
    draw();
  };

  const recordAudio = async () => {
    try {
      setIsRecording(true);
      setAudioSamples(null);
      setPrediction(null);
      cancelAnimation();

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const context = new window.AudioContext({ sampleRate: 16000 });
      audioContextRef.current = context;

      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Start Visuals & Timer
      startTimeRef.current = Date.now();
      drawWaveform();

      const processor = context.createScriptProcessor(4096, 1, 1);
      const samples: number[] = [];
      const TARGET_SAMPLES = 16000;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < inputData.length; i++) {
          if (samples.length < TARGET_SAMPLES) samples.push(inputData[i]);
        }
        if (samples.length >= TARGET_SAMPLES) {
          source.disconnect();
          processor.disconnect();
          stream.getTracks().forEach((track) => track.stop());
          context.close();
          cancelAnimation(); // This resets the bar
          setAudioSamples(samples);
          setIsRecording(false);
        }
      };

      source.connect(processor);
      processor.connect(context.destination);
    } catch (err) {
      console.error("Error recording audio:", err);
      setIsRecording(false);
    }
  };

  const playRecording = () => {
    if (!audioSamples) return;
    cancelAnimation();

    const playbackContext = new window.AudioContext({ sampleRate: 16000 });
    const buffer = playbackContext.createBuffer(1, audioSamples.length, 16000);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < audioSamples.length; i++) channelData[i] = audioSamples[i];

    const source = playbackContext.createBufferSource();
    source.buffer = buffer;
    const analyser = playbackContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    analyser.connect(playbackContext.destination);
    analyserRef.current = analyser;

    // Start Visuals & Timer
    startTimeRef.current = Date.now();
    drawWaveform();

    source.start();
    source.onended = () => {
      cancelAnimation();
      playbackContext.close();
    };
  };

  const inferAudio = async () => {
    if (!audioSamples) return;
    try {
      setIsInferring(true);
      setPrediction(null);
      const response = await fetch("http://localhost:8000/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ samples: audioSamples, sample_rate: 16000 }),
      });
      const data = await response.json();
      const label = data.label || data.prediction || data.class || "unknown";
      setPrediction(label.toLowerCase());
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      setIsInferring(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center py-12 gap-8 bg-zinc-950 text-white font-mono">

      {/* --- INSTRUCTIONS & OPTIONS --- */}
      <div className="flex flex-col items-center space-y-4 max-w-2xl px-4 text-center">
        <h1 className="text-2xl font-bold tracking-tight">speech to emoji</h1>


        <p className="text-zinc-400 text-xs">
          click record, say a word, replay to check your recording, and infer.
        </p>

        <div className="grid grid-cols-4 gap-3 mt-4 w-full">
          {Object.entries(KEYWORD_TO_EMOJI).map(([word, emoji]) => (
            <div
              key={word}
              className="flex flex-col items-center justify-center p-3 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-teal-500/50 transition-colors"
            >
              <span className="text-2xl mb-1">{emoji}</span>
              <span className="text-xs text-zinc-500 tracking-wider">{word}</span>
            </div>
          ))}
        </div>
      </div>

      {/* --- PREDICTION DISPLAY --- */}
      <div className="h-24 flex items-center justify-center w-full">
        {prediction ? (
          <div className="flex flex-col items-center animate-in fade-in zoom-in duration-300">
            <span className="text-6xl" role="img" aria-label={prediction}>
              {KEYWORD_TO_EMOJI[prediction] || "❓"}
            </span>
            <span className="text-teal-400 mt-2 text-xl font-bold tracking-widest">
              {prediction}
            </span>
          </div>
        ) : (
          <div className="text-zinc-700 text-sm border-b border-zinc-800 pb-1">
            {audioSamples ? "ready to infer" : "waiting for audio..."}
          </div>
        )}
      </div>

      {/* --- VISUALIZER & PROGRESS BAR --- */}
      <div className="relative w-full max-w-[600px] mx-4">
        {/* Progress Bar Container */}
        <div className="absolute top-0 left-0 w-full h-1 bg-zinc-800 rounded-t-md overflow-hidden z-10">
          {/* Progress Bar Fill */}
          <div
            ref={progressBarRef}
            className="h-full bg-teal-400 transition-all duration-75 ease-linear"
            style={{ width: "0%" }}
          />
        </div>

        <canvas
          ref={canvasRef}
          width={600}
          height={150}
          className="w-full border border-zinc-700 rounded-md bg-zinc-900 shadow-inner"
        />
      </div>

      {/* --- CONTROLS --- */}
      <div className="flex gap-4">
        <Button
          onClick={recordAudio}
          disabled={isRecording || isInferring}
          variant={isRecording ? "destructive" : "default"}
          className="w-32 relative overflow-hidden"
        >
          {isRecording ? "recording..." : "record"}
        </Button>

        <Button
          onClick={playRecording}
          disabled={!audioSamples || isRecording || isInferring}
          variant="outline"
          className="text-black bg-white hover:bg-zinc-200 w-24"
        >
          replay
        </Button>

        <Button
          onClick={inferAudio}
          disabled={!audioSamples || isRecording || isInferring}
          variant="secondary"
          className="w-24"
        >
          {isInferring ? "..." : "infer"}
        </Button>
      </div>
    </div>
  );
}
