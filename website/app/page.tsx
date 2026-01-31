"use client"
import { Button } from "@/components/ui/button"
import { Mic, Square, Play } from "lucide-react"
import { useRef, useState } from "react"

export default function Home() {
  const [isRecording, setIsRecording] = useState(false)
  const [audioURL, setAudioURL] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animationRef = useRef<number | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)

  const initAudio = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext()
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 2048
    }
    return { ctx: audioContextRef.current, analyser: analyserRef.current! }
  }

  const startRecording = async () => {
    const { ctx, analyser } = initAudio()
    if (ctx.state === 'suspended') await ctx.resume()

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const source = ctx.createMediaStreamSource(stream)

    // Connect Mic -> Analyser (Don't connect to ctx.destination or you'll hear yourself)
    source.connect(analyser)
    draw()

    const recorder = new MediaRecorder(stream)
    mediaRecorderRef.current = recorder
    chunksRef.current = []

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" })
      setAudioURL(URL.createObjectURL(blob))
      stream.getTracks().forEach(track => track.stop())
      source.disconnect() // Stop analyzing mic
    }

    recorder.start()
    setIsRecording(true)
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
  }

  const playBack = async () => {
    if (!audioURL) return
    const { ctx, analyser } = initAudio()
    if (ctx.state === 'suspended') await ctx.resume()

    const audio = new Audio(audioURL)
    const source = ctx.createMediaElementSource(audio)

    // Connect File -> Analyser -> Speakers
    source.connect(analyser)
    analyser.connect(ctx.destination)

    audio.play()
    draw()

    audio.onended = () => {
      source.disconnect()
      analyser.disconnect() // Clean up destination connection
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }

  const draw = () => {
    const canvas = canvasRef.current
    const analyser = analyserRef.current
    if (!canvas || !analyser) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const renderFrame = () => {
      animationRef.current = requestAnimationFrame(renderFrame)
      analyser.getByteTimeDomainData(dataArray)

      ctx.fillStyle = "rgb(0, 0, 0)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.lineWidth = 2
      ctx.strokeStyle = "#3b82f6"
      ctx.beginPath()

      const sliceWidth = canvas.width / bufferLength
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * canvas.height) / 2
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
        x += sliceWidth
      }
      ctx.lineTo(canvas.width, canvas.height / 2)
      ctx.stroke()
    }
    renderFrame()
  }

  return (
    <div className="flex flex-col min-h-screen items-center justify-center bg-black gap-8 p-4">
      <div className="w-full max-w-2xl bg-zinc-900 border border-zinc-800 rounded-xl p-2 shadow-2xl">
        <canvas ref={canvasRef} width={800} height={200} className="w-full h-48 rounded-lg" />
      </div>

      <div className="flex gap-4">
        <Button
          size="lg"
          variant={isRecording ? "destructive" : "default"}
          onClick={isRecording ? stopRecording : startRecording}
        >
          {isRecording ? <Square className="mr-2" /> : <Mic className="mr-2" />}
          {isRecording ? "Stop Recording" : "Record New Sample"}
        </Button>

        {audioURL && !isRecording && (
          <Button size="lg" variant="secondary" onClick={playBack}>
            <Play className="mr-2" /> Playback Sample
          </Button>
        )}
      </div>
    </div>
  )
}
