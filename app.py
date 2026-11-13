import asyncio
import base64
import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import edge_tts
import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from PIL import Image

app = FastAPI(title="AI Luxury Product Ad Studio")

WORK_DIR = Path("/tmp/ad_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODAL_VIDEO_URL = os.environ.get("MODAL_VIDEO_URL", "").strip().strip('"').strip("'")
if MODAL_VIDEO_URL and not MODAL_VIDEO_URL.startswith(("http://", "https://")):
    MODAL_VIDEO_URL = "https://" + MODAL_VIDEO_URL


def get_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def analyze_product_with_gemini(image_path: Path, user_hint: str, language: str) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(image_path).convert("RGB")

    lang_instruction = (
        "Write the voiceover_script in crisp, luxurious conversational Hinglish (Hindi in Roman script mixed with stylish English words, confident Indian female luxury ad voice)."
        if language == "hinglish"
        else "Write the voiceover_script in sleek, confident, luxury English."
    )

    system_prompt = f"""
You are a top 3D CGI commercial director (like viral Dior / Nike Blender commercials).
Analyze the exact product in the uploaded image and the user's hint: "{user_hint or 'Make a viral luxury 3D CGI commercial'}".

{lang_instruction}

Return ONLY valid JSON matching this exact structure:
{{
  "hero_brand": "1-2 word bold brand/product title (e.g. NALKY or SAUVAGE)",
  "tagline": "2-4 word punchy tagline (e.g. BUILT TO WIN)",
  "voiceover_script": "28 to 36 words total (approx 11-13 seconds spoken crisply). Confident luxury hook, bold feature line, punchy call-to-action.",
  "scenes": [
    {{
      "scene_style": "shatter_reveal",
      "video_prompt": "High-end 3D CGI luxury commercial shot: dark blue and metallic fragments shattering outward off the floating product against misty studio light, smooth camera push-in, photorealistic Cinema4D Octane render."
    }},
    {{
      "scene_style": "macro_detail",
      "video_prompt": "Extreme macro 3D commercial close-up gliding smoothly across the glossy details and texture of the product, crisp studio rim light sweeping over reflections, ultra sharp."
    }},
    {{
      "scene_style": "pedestal_blinds",
      "video_prompt": "Sleek architectural luxury commercial shot: three products standing on white stepped pedestals, diagonal window blind sunlight shadows shifting gently across the background, blurred green leaves swaying softly, smooth horizontal dolly."
    }},
    {{
      "scene_style": "twin_float",
      "video_prompt": "Minimalist bright grey studio commercial: two floating products levitating in mid-air and rotating smoothly past each other with crisp reflections, high-fashion CGI aesthetic."
    }},
    {{
      "scene_style": "chrome_finale",
      "video_prompt": "Dramatic finale 3D CGI shot: silver chrome shatter particles swirling cleanly around the hero product under bright spotlight, luxury commercial packshot."
    }}
  ]
}}
"""
    # Use free-tier supported Gemini 3.5 Flash-Lite with safe fallback
    for model_name in ("gemini-3.5-flash-lite", "gemini-3-flash-preview"):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[system_prompt, img],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return json.loads(response.text)
        except Exception:
            continue
    raise RuntimeError("Gemini API call failed. Check GEMINI_API_KEY.")


async def create_female_voiceover(text: str, out_mp3: Path, language: str):
    voice = "hi-IN-SwaraNeural" if language == "hinglish" else "en-IN-NeerjaNeural"
    communicate = edge_tts.Communicate(text=text, voice=voice, rate="+8%")
    await communicate.save(str(out_mp3))


def generate_ai_video_clip(image_bytes: bytes, prompt: str, scene_style: str, aspect_ratio: str, out_mp4: Path):
    if not MODAL_VIDEO_URL:
        raise RuntimeError("MODAL_VIDEO_URL environment variable is not set.")

    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "prompt": prompt,
        "scene_style": scene_style,
        "aspect_ratio": aspect_ratio,
    }
    with httpx.Client(timeout=420.0) as client:
        resp = client.post(MODAL_VIDEO_URL, json=payload)
        resp.raise_for_status()
        out_mp4.write_bytes(resp.content)


def build_luxury_commercial(
    job_dir: Path,
    raw_clips: list[Path],
    voice_mp3: Path,
    plan: dict,
    aspect_ratio: str,
) -> Path:
    w, h = (720, 1280) if aspect_ratio == "9:16" else (1280, 720)
    sans_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    serif_font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    if not Path(serif_font).exists():
        serif_font = sans_font

    hero_brand = plan.get("hero_brand", "LUXURY").upper()[:18]
    tagline = plan.get("tagline", "PURE PERFECTION").upper()[:28]

    hero_file = job_dir / "hero.txt"
    tag_file = job_dir / "tag.txt"
    outro_file = job_dir / "outro.txt"
    hero_file.write_text(hero_brand, encoding="utf-8")
    tag_file.write_text(tagline, encoding="utf-8")
    outro_file.write_text(f"PRODUCED FOR {hero_brand}", encoding="utf-8")

    # Snappy 2.0s - 2.3s luxury commercial cuts + Dior-style centered typography
    styled_clips = []
    title_size = 96 if aspect_ratio == "9:16" else 112

    for idx, clip in enumerate(raw_clips):
        styled = job_dir / f"styled_{idx}.mp4"
        overlay = ""
        if idx == 0:
            # Giant centered luxury Serif title on opening hero reveal
            overlay = (
                f",drawtext=fontfile={serif_font}:textfile={hero_file}:fontcolor=white@0.95:"
                f"fontsize={title_size}:shadowcolor=black@0.45:shadowx=3:shadowy=4:"
                f"x=(w-text_w)/2:y=(h-text_h)/2+40"
            )
        elif idx == 2:
            # Clean minimal caption on architectural pedestal shot
            overlay = (
                f",drawtext=fontfile={sans_font}:textfile={tag_file}:fontcolor=white:"
                f"fontsize=34:box=1:boxcolor=black@0.55:boxborderw=14:x=(w-text_w)/2:y=h-170"
            )
        elif idx == 4:
            # Bold finale lockup
            overlay = (
                f",drawtext=fontfile={sans_font}:textfile={outro_file}:fontcolor=white@0.95:"
                f"fontsize=36:shadowcolor=black@0.5:shadowx=2:shadowy=2:"
                f"x=(w-text_w)/2:y=(h-text_h)/2"
            )

        # Speed up slightly (setpts=0.82*PTS) for ultra-snappy commercial pacing (~2.2s/shot)
        vf = (
            f"setpts=0.82*PTS,scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},fps=24,eq=contrast=1.10:saturation=1.15{overlay}"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(clip),
                "-vf", vf, "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "2",
                str(styled),
            ],
            check=True,
        )
        styled_clips.append(styled)

    # Fast beat-synced commercial transitions across all 5 shots
    d = [get_duration(c) for c in styled_clips]
    fade = 0.22
    o1 = max(0.5, d[0] - fade)
    o2 = max(1.0, o1 + d[1] - fade)
    o3 = max(1.5, o2 + d[2] - fade)
    o4 = max(2.0, o3 + d[3] - fade)

    merged_video = job_dir / "merged_video.mp4"
    filter_complex = (
        f"[0:v][1:v]xfade=transition=fadewhite:duration={fade}:offset={o1}[v01];"
        f"[v01][2:v]xfade=transition=slideleft:duration={fade}:offset={o2}[v02];"
        f"[v02][3:v]xfade=transition=fade:duration={fade}:offset={o3}[v03];"
        f"[v03][4:v]xfade=transition=fadewhite:duration={fade}:offset={o4}[vout]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(styled_clips[0]),
            "-i", str(styled_clips[1]),
            "-i", str(styled_clips[2]),
            "-i", str(styled_clips[3]),
            "-i", str(styled_clips[4]),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "2",
            str(merged_video),
        ],
        check=True,
    )

    # Sync duration with Female Voiceover + deep modern electronic commercial bass beat
    vid_dur = get_duration(merged_video)
    aud_dur = get_duration(voice_mp3)
    target_dur = max(vid_dur, aud_dur + 0.5)
    pad_sec = max(0.0, target_dur - vid_dur)

    final_mp4 = job_dir / "final_ad.mp4"
    audio_filter = (
        f"aevalsrc=0.09*sin(2*PI*55*t)*exp(-4*mod(t,0.5))+0.04*sin(2*PI*165*t)*exp(-8*mod(t,0.25)):s=44100:d={target_dur}[bg];"
        f"[1:a]aresample=44100,volume=1.40[vo];"
        f"[vo][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    video_filter = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_sec}[vfinal]"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(merged_video),
            "-i", str(voice_mp3),
            "-filter_complex", f"{video_filter};{audio_filter}",
            "-map", "[vfinal]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-b:a", "128k",
            "-t", f"{target_dur:.2f}", "-movflags", "+faststart", "-threads", "2",
            str(final_mp4),
        ],
        check=True,
    )
    return final_mp4


def run_pipeline(job_id: str, image_path: Path, hint: str, language: str, aspect_ratio: str):
    job_dir = image_path.parent
    try:
        JOBS[job_id]["step"] = "AI product analyze karke 5 CGI scenes design kar raha hai..."
        JOBS[job_id]["progress"] = 10
        plan = analyze_product_with_gemini(image_path, hint, language)
        JOBS[job_id]["script"] = plan["voiceover_script"]

        JOBS[job_id]["step"] = "Women's AI Voiceover record ho raha hai..."
        JOBS[job_id]["progress"] = 22
        voice_mp3 = job_dir / "voice.mp3"
        asyncio.run(create_female_voiceover(plan["voiceover_script"], voice_mp3, language))

        img_bytes = image_path.read_bytes()
        raw_clips = []
        for idx, scene in enumerate(plan["scenes"]):
            JOBS[job_id]["step"] = f"3D CGI AI Scene {idx + 1}/5 ({scene['scene_style']}) render ho raha hai..."
            JOBS[job_id]["progress"] = 30 + (idx * 12)
            clip_path = job_dir / f"raw_{idx}.mp4"
            generate_ai_video_clip(
                img_bytes,
                scene["video_prompt"],
                scene["scene_style"],
                aspect_ratio,
                clip_path,
            )
            raw_clips.append(clip_path)

        JOBS[job_id]["step"] = "Beat-synced luxury transitions & typography edit ho rahe hain..."
        JOBS[job_id]["progress"] = 92
        final_mp4 = build_luxury_commercial(job_dir, raw_clips, voice_mp3, plan, aspect_ratio)

        JOBS[job_id].update({
            "status": "done",
            "progress": 100,
            "step": "Luxury CGI Ad ready hai!",
            "video_url": f"/download/{job_id}",
        })
    except Exception as exc:
        JOBS[job_id].update({
            "status": "error",
            "step": f"Error: {str(exc)}",
        })


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/generate")
async def generate_ad(
    image: UploadFile = File(...),
    hint: str = Form(""),
    language: str = Form("hinglish"),
    aspect_ratio: str = Form("9:16"),
):
    job_id = uuid.uuid4().hex[:10]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    img_path = job_dir / "product.png"
    with open(img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    JOBS[job_id] = {"status": "processing", "progress": 5, "step": "Starting...", "script": ""}
    threading.Thread(
        target=run_pipeline,
        args=(job_id, img_path, hint, language, aspect_ratio),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"status": "error", "step": "Job not found"}, status_code=404)
    return JOBS[job_id]


@app.get("/download/{job_id}")
def download_video(job_id: str):
    video_path = WORK_DIR / job_id / "final_ad.mp4"
    if not video_path.exists():
        return JSONResponse({"error": "Video not ready"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4", filename=f"luxury_ad_{job_id}.mp4")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI 3D CGI Ad Studio</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #070b14; color: #f8fafc; margin: 0; padding: 24px; }
    .card { max-width: 620px; margin: 0 auto; background: #111827; padding: 28px; border-radius: 16px; border: 1px solid #1f2937; }
    h1 { margin-top: 0; font-size: 1.55rem; color: #38bdf8; }
    label { display: block; margin-top: 16px; font-weight: 600; font-size: 0.92rem; color: #cbd5e1; }
    input, textarea, select { width: 100%; margin-top: 6px; padding: 10px 12px; border-radius: 10px; border: 1px solid #334155; background: #0b1120; color: #fff; box-sizing: border-box; }
    button { width: 100%; margin-top: 22px; padding: 14px; border: none; border-radius: 10px; background: linear-gradient(90deg, #0ea5e9, #6366f1); color: #fff; font-weight: 700; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .progress-box { display: none; margin-top: 20px; background: #0b1120; padding: 14px; border-radius: 10px; }
    .bar-bg { width: 100%; height: 10px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-top: 8px; }
    .bar-fill { height: 100%; width: 0%; background: #38bdf8; transition: width 0.4s; }
    video { width: 100%; max-height: 520px; border-radius: 12px; margin-top: 16px; background: #000; }
    .dl-btn { display: inline-block; margin-top: 12px; padding: 10px 18px; background: #10b981; color: #fff; text-decoration: none; border-radius: 8px; font-weight: 600; }
  </style>
</head>
<body>
  <div class="card">
    <h1>✨ AI Luxury 3D CGI Ad Studio</h1>
    <p style="color:#94a3b8;font-size:0.9rem;">Product photo daalein + ek hint likhein. AI khud product nikal kar Shatter Reveal, Pedestal Blinds Shadows, Twin Float & Women's Voiceover ke saath Viral Ad bana dega.</p>
    <form id="adForm">
      <label>1. Product Photo Upload Karein</label>
      <input type="file" name="image" accept="image/*" required />

      <label>2. Ad Hint Likhein (Optional)</label>
      <textarea name="hint" rows="2" placeholder="Jaise: Luxury Dior style commercial banao, energetic mood"></textarea>

      <label>3. Voiceover Language (Women's Voice)</label>
      <select name="language">
        <option value="hinglish">Hinglish (Swara Female Voice)</option>
        <option value="english">Indian English (Neerja Female Voice)</option>
      </select>

      <label>4. Format</label>
      <select name="aspect_ratio">
        <option value="9:16" selected>Instagram Reels / Shorts (9:16 Viral Format)</option>
        <option value="16:9">YouTube / Amazon / Landscape (16:9)</option>
      </select>

      <button type="submit" id="submitBtn">🎬 Generate Viral CGI Ad</button>
    </form>

    <div class="progress-box" id="progressBox">
      <div id="statusText">Starting...</div>
      <div class="bar-bg"><div class="bar-fill" id="barFill"></div></div>
      <p id="scriptBox" style="font-size:0.85rem;color:#94a3b8;margin-top:10px;"></p>
    </div>

    <div id="resultBox" style="display:none;">
      <video id="adVideo" controls></video>
      <a id="dlLink" class="dl-btn" download>⬇️ Download MP4 Ad</a>
    </div>
  </div>

  <script>
    const form = document.getElementById("adForm");
    const btn = document.getElementById("submitBtn");
    const progressBox = document.getElementById("progressBox");
    const statusText = document.getElementById("statusText");
    const barFill = document.getElementById("barFill");
    const scriptBox = document.getElementById("scriptBox");
    const resultBox = document.getElementById("resultBox");
    const adVideo = document.getElementById("adVideo");
    const dlLink = document.getElementById("dlLink");

    form.onsubmit = async (e) => {
      e.preventDefault();
      btn.disabled = true;
      resultBox.style.display = "none";
      progressBox.style.display = "block";
      statusText.innerText = "Photo upload ho rahi hai...";
      barFill.style.width = "5%";

      const fd = new FormData(form);
      const res = await fetch("/generate", { method: "POST", body: fd });
      const { job_id } = await res.json();

      const timer = setInterval(async () => {
        const sRes = await fetch(`/status/${job_id}`);
        const data = await sRes.json();
        statusText.innerText = data.step || "Processing...";
        barFill.style.width = (data.progress || 10) + "%";
        if (data.script) scriptBox.innerText = "🎙️ Voiceover: " + data.script;

        if (data.status === "done") {
          clearInterval(timer);
          btn.disabled = false;
          adVideo.src = data.video_url;
          dlLink.href = data.video_url;
          resultBox.style.display = "block";
        } else if (data.status === "error") {
          clearInterval(timer);
          btn.disabled = false;
        }
      }, 3000);
    };
  </script>
</body>
</html>
"""
