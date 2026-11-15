import asyncio
import base64
import gc
import io
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import edge_tts
import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont, ImageFilter

app = FastAPI(title="AI Cinematic 3D Product Commercial Studio")

WORK_DIR = Path("/tmp/ad_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
FAL_KEY = os.environ.get("FAL_KEY", "").strip().strip('"').strip("'")


def save_job(job_id: str, **updates):
    if job_id not in JOBS:
        JOBS[job_id] = {}
    JOBS[job_id].update(updates)
    status_file = WORK_DIR / job_id / "status.json"
    if status_file.parent.exists():
        try:
            status_file.write_text(json.dumps(JOBS[job_id]), encoding="utf-8")
        except Exception:
            pass


def load_job(job_id: str) -> dict | None:
    if job_id in JOBS:
        return JOBS[job_id]
    status_file = WORK_DIR / job_id / "status.json"
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            JOBS[job_id] = data
            return data
        except Exception:
            pass
    final_mp4 = WORK_DIR / job_id / "final_ad.mp4"
    if final_mp4.exists():
        return {"status": "done", "progress": 100, "step": "Your Commercial Ad is ready!", "video_url": f"/download/{job_id}"}
    return None


def get_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def prepare_product_frame(image_path: Path, output_png: Path, scene_idx: int) -> Path:
    """प्रोडक्ट को सिनेमैटिक 3D स्टूडियो लाइटिंग और पार्टिकल इफेक्ट्स के साथ तैयार करता है"""
    img = Image.open(image_path).convert("RGBA")
    W, H = 540, 960

    # सीन के हिसाब से अलग-अलग ज़ूम और कंपोजीशन ताकि 3D कमर्शियल जैसा लगे
    if scene_idx == 0:
        img.thumbnail((int(W * 0.65), int(H * 0.65)), Image.Resampling.LANCZOS)
    elif scene_idx == 1:
        img.thumbnail((int(W * 0.85), int(H * 0.85)), Image.Resampling.LANCZOS) # Macro close-up feel
    elif scene_idx == 2:
        img.thumbnail((int(W * 0.70), int(H * 0.70)), Image.Resampling.LANCZOS)
    else:
        img.thumbnail((int(W * 0.75), int(H * 0.75)), Image.Resampling.LANCZOS)

    # सिनेमैटिक डार्क लग्जरी स्टूडियो बैकग्राउंड (वियो और सोरा जैसे विज्ञापनों जैसा)
    bg = Image.new("RGBA", (W, H), (10, 14, 24, 255))
    draw = ImageDraw.Draw(bg)
    
    # रेडियल ग्लो / स्पॉटलाइट इफेक्ट
    for y in range(H):
        factor = 1.0 - abs(y - H // 2) / (H // 2) * 0.3
        r = int(15 * factor)
        g = int(22 * factor)
        b = int(40 * factor)
        draw.line([(0, y), (W, y)], fill=(r, g, b, 255))

    # हवा में उड़ते हुए 3D पार्टिकल्स / स्पार्क्स (CGI Simulation)
    for _ in range(35):
        px = random.randint(10, W - 10)
        py = random.randint(10, H - 10)
        p_size = random.randint(2, 5)
        alpha = random.randint(60, 180)
        draw.ellipse([px, py, px + p_size, py + p_size], fill=(200, 220, 255, alpha))

    px = (W - img.width) // 2
    py = (H - img.height) // 2 + (20 if scene_idx == 1 else 0)

    # रियलिस्टिक ड्रॉप शैडो और फ्लोर रिफ्लेक्शन
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh_layer = Image.new("RGBA", img.size, (0, 0, 0, 180))
    shadow.paste(sh_layer, (px, py + 30), img)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))

    bg.alpha_composite(shadow)
    bg.alpha_composite(img, (px, py))

    # सीन 1 पर स्टाइलिश टेक्स्ट ओवरले ("DAY 08 - MAKING PRODUCT ADS")
    if scene_idx == 0:
        f_sub = _load_font(20)
        draw_txt = ImageDraw.Draw(bg)
        draw_txt.text((35, 120), "MAKING PRODUCT ADS", font=f_sub, fill=(180, 200, 230, 220))
        f_big = _load_font(36)
        draw_txt.text((35, 150), "UNTIL BRAND NOTICES", font=f_big, fill=(255, 255, 255, 255))
        draw_txt.text((35, 195), "DAY 08 🎬", font=f_big, fill=(250, 204, 21, 255))

    final_img = bg.convert("RGB")
    final_img.save(output_png, format="PNG")
    return output_png


def plan_commercial(image_path: Path, user_hint: str) -> dict:
    raw = image_path.stem.replace("_", " ").replace("-", " ").strip()
    brand = "".join(c for c in (raw.split()[0] if raw else "LUXE") if c.isalnum()).upper() or "LUXE"
    hint_clean = (user_hint or "unmatched cinematic quality and premium daily lifestyle").strip()

    if GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            prompt = f"""
You are an elite CGI Commercial Director (like Dior Sauvage or Apple 3D ads).
Product image provided. User description: "{hint_clean}".
Create a 4-scene high-converting commercial plan in JSON format:
{{
  "brand_name": "{brand}",
  "headline_top": "PURE ELEGANCE",
  "headline_bottom": "EXPERIENCE NOW",
  "signposts": ["Pro Quality", "CGI Grade", "Best Choice"],
  "voiceover_script": "Making product ads until the world notices. Day eight. Behold the all-new {brand}! Engineered for {hint_clean}, bringing breathtaking 3D design and unmatched performance. Upgrade your standard today!"
}}
"""
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, img],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            txt = (resp.text or "").strip()
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(txt)
            if "signposts" in data and "voiceover_script" in data:
                return data
        except Exception:
            pass

    return {
        "brand_name": brand[:16],
        "headline_top": "PURE ELEGANCE",
        "headline_bottom": "BUILT TO PERFECTION",
        "signposts": ["Pro Quality", "CGI Grade", "Best Choice"],
        "voiceover_script": f"Making product ads until the world notices. Day eight. Behold the all-new {brand}! Engineered for {hint_clean}, bringing breathtaking design and unmatched performance. Upgrade your standard today!"
    }


async def create_voiceover(text: str, out_mp3: Path):
    try:
        communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural", rate="+5%")
        await communicate.save(str(out_mp3))
    except Exception:
        gTTS(text=text, lang="en", tld="com").save(str(out_mp3))


def animate_scene(keyframe_png: Path, mode: str, aspect_ratio: str, duration: float, out_mp4: Path):
    W, H = (540, 960) if aspect_ratio == "9:16" else (960, 540)
    fps = 24
    total_frames = int(duration * fps)

    src = Image.open(keyframe_png).convert("RGB")
    src = src.resize((int(W * 1.18), int(H * 1.18)), Image.Resampling.LANCZOS)
    sw, sh = src.size

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency", "-crf", "18",
        "-pix_fmt", "yuv420p", "-threads", "1", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for f in range(total_frames):
        p = f / max(1, total_frames - 1)
        if mode == "zoom_in":
            scale = 1.0 + 0.14 * p
            ox = int((sw - W / scale) * 0.5)
            oy = int((sh - H / scale) * 0.5)
        elif mode == "pan_right":
            scale = 1.08
            ox = int((sw - W / scale) * (0.2 + 0.6 * p))
            oy = int((sh - H / scale) * 0.5)
        elif mode == "dramatic_push":
            scale = 1.02 + 0.15 * (p ** 0.85)
            ox = int((sw - W / scale) * (0.5 + 0.05 * math.sin(p * math.pi)))
            oy = int((sh - H / scale) * 0.48)
        else:
            scale = 1.12 - 0.1 * p
            ox = int((sw - W / scale) * 0.5)
            oy = int((sh - H / scale) * 0.5)

        cw, ch = int(W / scale), int(H / scale)
        ox = max(0, min(sw - cw, ox))
        oy = max(0, min(sh - ch, oy))
        frame = src.crop((ox, oy, ox + cw, oy + ch)).resize((W, H), Image.Resampling.BILINEAR)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()


def _make_signpost_png(label: str, angle: float, out_png: Path):
    font = _load_font(26)
    words = label.strip().split()
    line1 = words[0] if words else "PRO"
    line2 = " ".join(words[1:]) if len(words) > 1 else ""
    bw, bh = 190, 115 if line2 else 78

    card = Image.new("RGBA", (bw, bh + 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rectangle([bw // 2 - 5, bh - 3, bw // 2 + 5, bh + 115], fill=(212, 175, 55, 255), outline=(100, 80, 20, 255), width=2)
    d.rounded_rectangle([3, 3, bw - 3, bh - 3], radius=14, fill=(20, 25, 40, 240), outline=(212, 175, 55, 255), width=3)

    if line2:
        d.text((bw // 2 - int(d.textlength(line1, font=font)) // 2, 16), line1, font=font, fill=(255, 255, 255, 255))
        d.text((bw // 2 - int(d.textlength(line2, font=font)) // 2, 54), line2, font=font, fill=(250, 204, 21, 255))
    else:
        d.text((bw // 2 - int(d.textlength(line1, font=font)) // 2, 22), line1, font=font, fill=(255, 255, 255, 255))

    rot = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    rot.save(out_png, format="PNG")


def _create_headline_png(W: int, H: int, top_line: str, bottom_line: str, out_png: Path):
    banner = Image.new("RGBA", (W, int(H * 0.34)), (0, 0, 0, 0))
    f_top = _load_font(42)
    f_bot = _load_font(40)

    txt_layer = Image.new("RGBA", (W, int(H * 0.32)), (0, 0, 0, 0))
    d = ImageDraw.Draw(txt_layer)
    t1, t2 = top_line.upper()[:24], bottom_line.upper()[:24]

    d.text((W // 2 - int(d.textlength(t1, font=f_top)) // 2 + 2, 36), t1, font=f_top, fill=(0, 0, 0, 180))
    d.text((W // 2 - int(d.textlength(t1, font=f_top)) // 2, 34), t1, font=f_top, fill=(250, 204, 21, 255))

    d.text((W // 2 - int(d.textlength(t2, font=f_bot)) // 2 + 2, 90), t2, font=f_bot, fill=(0, 0, 0, 180))
    d.text((W // 2 - int(d.textlength(t2, font=f_bot)) // 2, 88), t2, font=f_bot, fill=(255, 255, 255, 255))

    tilted = txt_layer.rotate(4, expand=False, resample=Image.Resampling.BICUBIC)
    banner.alpha_composite(tilted, (0, 8))
    banner.save(out_png, format="PNG")


def run_pipeline(job_id: str, image_path: Path, hint: str):
    job_dir = image_path.parent
    try:
        save_job(job_id, step="Planning 3D CGI Commercial & script...", progress=15)
        plan = plan_commercial(image_path, hint)
        save_job(job_id, script=plan["voiceover_script"])

        save_job(job_id, step="Recording cinematic voiceover...", progress=30)
        voice_mp3 = job_dir / "voice.mp3"
        asyncio.run(create_voiceover(plan["voiceover_script"], voice_mp3))

        scene_videos = []
        modes = [("zoom_in", 3.8), ("pan_right", 3.5), ("dramatic_push", 4.2), ("zoom_in", 3.5)]

        for idx in range(4):
            save_job(job_id, step=f"Rendering CGI Scene {idx + 1}/4...", progress=40 + idx * 12)
            kf_png = job_dir / f"kf_{idx}.png"
            prepare_product_frame(image_path, kf_png, idx)

            sc_mp4 = job_dir / f"scene_{idx}.mp4"
            mode, dur = modes[idx]
            animate_scene(kf_png, mode, "9:16", dur, sc_mp4)
            kf_png.unlink(missing_ok=True)
            scene_videos.append(sc_mp4)

        save_job(job_id, step="Adding CGI signposts, headline & assembling final ad...", progress=88)
        W, H = 540, 960
        norm_clips = []

        for idx, sc in enumerate(scene_videos):
            norm = job_dir / f"norm_{idx}.mp4"
            dur = get_duration(sc)
            base_vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps=24"

            if idx == 2:
                labels = plan.get("signposts", ["Pro Quality", "CGI Grade", "Best Choice"])[:3]
                sp_paths = []
                for s_i, lbl in enumerate(labels):
                    sp = job_dir / f"sign_{s_i}.png"
                    _make_signpost_png(lbl, (-6 if s_i % 2 == 0 else 6), sp)
                    sp_paths.append(sp)

                seg = dur / 3.0
                fc = (
                    f"[0:v]{base_vf}[b0];"
                    f"[b0][1:v]overlay=x=30:y=340:enable='between(t\\,0.05\\,{seg})'[b1];"
                    f"[b1][2:v]overlay=x=310:y=260:enable='between(t\\,{seg}\\,{seg*2})'[b2];"
                    f"[b2][3:v]overlay=x=170:y=310:enable='between(t\\,{seg*2}\\,{dur})'[v]"
                )
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(sc), "-i", str(sp_paths[0]), "-i", str(sp_paths[1]), "-i", str(sp_paths[2]),
                    "-filter_complex", fc, "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "superfast", "-crf", "18", str(norm)
                ], check=True)
                for sp in sp_paths:
                    sp.unlink(missing_ok=True)
            elif idx == 3:
                head_png = job_dir / "headline.png"
                _create_headline_png(W, H, plan.get("headline_top", "PURE ELEGANCE"), plan.get("headline_bottom", "EXPERIENCE NOW"), head_png)
                fc = f"[0:v]{base_vf}[b0];[b0][1:v]overlay=0:25[v]"
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(sc), "-i", str(head_png),
                    "-filter_complex", fc, "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "superfast", "-crf", "18", str(norm)
                ], check=True)
                head_png.unlink(missing_ok=True)
            else:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(sc), "-vf", base_vf, "-an",
                    "-c:v", "libx264", "-preset", "superfast", "-crf", "18", str(norm)
                ], check=True)

            norm_clips.append(norm)

        merged = job_dir / "merged.mp4"
        concat_txt = job_dir / "concat.txt"
        concat_txt.write_text("\n".join(f"file '{c.resolve()}'" for c in norm_clips), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(merged)], check=True)

        final_mp4 = job_dir / "final_ad.mp4"
        aud_dur = get_duration(voice_mp3)
        vid_dur = get_duration(merged)
        target_dur = max(vid_dur, aud_dur + 0.5)

        audio_filter = f"[1:a]aresample=44100,volume=1.6[vo]"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(merged), "-i", str(voice_mp3),
            "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={max(0, target_dur - vid_dur):.2f}[vfinal];{audio_filter}",
            "-map", "[vfinal]", "-map", "[vo]", "-c:v", "libx264", "-preset", "superfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k", "-t", f"{target_dur:.2f}", "-movflags", "+faststart", str(final_mp4)
        ], check=True)

        save_job(job_id, status="done", progress=100, step="Your Commercial Ad is ready!", video_url=f"/download/{job_id}")
    except Exception as e:
        save_job(job_id, status="error", step=f"Error: {str(e)}")


@app.post("/generate")
async def generate_ad(image: UploadFile = File(...), hint: str = Form("")):
    job_id = uuid.uuid4().hex[:10]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    img_path = job_dir / "product.png"
    with open(img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    save_job(job_id, status="processing", progress=5, step="Starting...")
    threading.Thread(target=run_pipeline, args=(job_id, img_path, hint), daemon=True).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    return load_job(job_id) or {"status": "error", "step": "Not found"}


@app.get("/download/{job_id}")
def download_video(job_id: str):
    return FileResponse(WORK_DIR / job_id / "final_ad.mp4", media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Cinematic 3D Product Commercial Studio</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #070b14; color: #f8fafc; margin: 0; padding: 24px; }
    .card { max-width: 600px; margin: 0 auto; background: #111827; padding: 28px; border-radius: 16px; border: 1px solid #1f2937; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h1 { color: #facc15; font-size: 1.5rem; margin-top: 0; }
    label { display: block; margin-top: 16px; font-weight: 600; color: #cbd5e1; }
    input, textarea { width: 100%; margin-top: 6px; padding: 10px; border-radius: 8px; border: 1px solid #334155; background: #0b1120; color: #fff; box-sizing: border-box; }
    button { width: 100%; margin-top: 22px; padding: 14px; border: none; border-radius: 8px; background: linear-gradient(90deg, #f59e0b, #ef4444); color: #fff; font-weight: 700; cursor: pointer; font-size: 1rem; }
    button:hover { opacity: 0.9; }
    .progress-box { display: none; margin-top: 20px; background: #0b1120; padding: 14px; border-radius: 8px; }
    .bar-bg { width: 100%; height: 10px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-top: 8px; }
    .bar-fill { height: 100%; width: 0%; background: #facc15; transition: width 0.4s; }
    video { width: 100%; border-radius: 8px; margin-top: 16px; display: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎬 AI 3D Cinematic Commercial Studio</h1>
    <p style="color:#94a3b8;font-size:0.9rem;">Upload a product photo. AI automatically places it on a 3D luxury CGI studio background with glowing particles, dynamic camera pans, signposts & voiceover!</p>
    <form id="adForm">
      <label>Product Photo</label>
      <input type="file" name="image" accept="image/*" required />
      <label>Product Benefits / Description</label>
      <textarea name="hint" rows="2" placeholder="e.g. Luxury perfume bottle, long-lasting fragrance, elegant design"></textarea>
      <button type="submit" id="submitBtn">🚀 Generate 3D Cinematic Ad</button>
    </form>
    <div class="progress-box" id="progressBox">
      <div id="stepText">Processing...</div>
      <div class="bar-bg"><div class="bar-fill" id="barFill"></div></div>
    </div>
    <video id="resultVideo" controls></video>
  </div>
  <script>
    document.getElementById('adForm').onsubmit = async (e) => {
      e.preventDefault();
      const btn = document.getElementById('submitBtn');
      btn.disabled = true;
      document.getElementById('progressBox').style.display = 'block';
      
      const res = await fetch('/generate', { method: 'POST', body: new FormData(e.target) });
      const data = await res.json();
      const jobId = data.job_id;

      const interval = setInterval(async () => {
        const stRes = await fetch('/status/' + jobId);
        const st = await stRes.json();
        document.getElementById('stepText').innerText = st.step || "Processing...";
        document.getElementById('barFill').style.width = (st.progress || 10) + '%';
        if (st.status === 'done') {
          clearInterval(interval);
          document.getElementById('progressBox').style.display = 'none';
          const vid = document.getElementById('resultVideo');
          vid.src = st.video_url;
          vid.style.display = 'block';
          btn.disabled = false;
        } else if (st.status === 'error') {
          clearInterval(interval);
          alert(st.step);
          btn.disabled = false;
        }
      }, 2000);
    };
  </script>
</body>
</html>
"""
