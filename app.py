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
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFilter

app = FastAPI(title="AI Luxury Product Ad Studio")

WORK_DIR = Path("/tmp/ad_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
MODAL_VIDEO_URL = os.environ.get("MODAL_VIDEO_URL", "").strip().strip('"').strip("'")
if MODAL_VIDEO_URL and not MODAL_VIDEO_URL.startswith(("http://", "https://")):
    MODAL_VIDEO_URL = "https://" + MODAL_VIDEO_URL


def get_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def _fallback_ad_plan(image_path: Path, user_hint: str, language: str) -> dict:
    raw = image_path.stem.replace("_", " ").replace("-", " ").strip()
    brand = "".join(c for c in (raw.split()[0] if raw else "LUXURY") if c.isalnum()).upper() or "LUXURY"
    hint_clean = (user_hint or "luxury perfection").strip()

    if language == "hinglish":
        script = (
            f"Jab style aur class ki baat ho, toh dekho {brand}. "
            f"Pure {hint_clean.lower()} ke saath unmatched design aur royal finishing. "
            f"Apna standard elevate karo, grab your {brand} today!"
        )
    else:
        script = (
            f"When luxury meets pure performance, meet {brand}. "
            f"Crafted with {hint_clean.lower()}, precision engineering, and timeless elegance. "
            f"Elevate your everyday style and claim yours today."
        )

    return {
        "hero_brand": brand[:18],
        "tagline": "BUILT TO STAND OUT",
        "voiceover_script": script,
        "scenes": [
            {"scene_style": "shatter_reveal", "video_prompt": f"3D CGI shatter reveal for {brand}"},
            {"scene_style": "macro_detail", "video_prompt": f"Macro close-up for {brand}"},
            {"scene_style": "pedestal_blinds", "video_prompt": f"Pedestal blinds shadows for {brand}"},
            {"scene_style": "twin_float", "video_prompt": f"Twin float studio shot for {brand}"},
            {"scene_style": "chrome_finale", "video_prompt": f"Chrome finale packshot for {brand}"},
        ],
    }


def analyze_product_with_gemini(image_path: Path, user_hint: str, language: str) -> dict:
    if not GEMINI_API_KEY:
        return _fallback_ad_plan(image_path, user_hint, language)

    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)

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
    {{"scene_style": "shatter_reveal", "video_prompt": "3D CGI shatter reveal"}},
    {{"scene_style": "macro_detail", "video_prompt": "Macro detail sweep"}},
    {{"scene_style": "pedestal_blinds", "video_prompt": "Pedestal window blinds shadows"}},
    {{"scene_style": "twin_float", "video_prompt": "Twin floating rotation"}},
    {{"scene_style": "chrome_finale", "video_prompt": "Chrome sparkle finale"}}
  ]
}}
"""
    for model_name in ("gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-2.5-flash"):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[system_prompt, img],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            txt = (response.text or "").strip()
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(txt)
        except Exception as e:
            print(f"Gemini failed on {model_name}: {repr(e)}", flush=True)
            continue
    return _fallback_ad_plan(image_path, user_hint, language)


async def create_female_voiceover(text: str, out_mp3: Path, language: str):
    voice = "hi-IN-SwaraNeural" if language == "hinglish" else "en-IN-NeerjaNeural"
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+8%")
        await communicate.save(str(out_mp3))
    except Exception as e:
        print(f"Edge-TTS failed ({e}), using gTTS fallback...", flush=True)
        tts_lang = "hi" if language == "hinglish" else "en"
        gTTS(text=text, lang=tts_lang, tld="co.in").save(str(out_mp3))


def _extract_product_light(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img.thumbnail((480, 480), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape

    border = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]], axis=0)
    bg_rgb = np.median(border, axis=0)
    std = float(np.mean(np.std(border, axis=0)))

    if std < 36.0:
        dist = np.linalg.norm(arr[:, :, :3] - bg_rgb, axis=2)
        alpha = np.clip((dist - 18.0) / 28.0, 0.0, 1.0) * 255.0
        if np.mean(alpha > 128) > 0.12:
            mask = Image.fromarray(alpha.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1))
            img.putalpha(mask)
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            return img

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=24, fill=255)
    img.putalpha(mask)
    return img


def _make_gradient_bg(w: int, h: int, c_edge: tuple, c_center: tuple) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((x - w * 0.5) / (w * 0.65)) ** 2 + ((y - h * 0.44) / (h * 0.55)) ** 2)
    t = np.clip(1.0 - dist, 0.0, 1.0) ** 1.6
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(3):
        rgb[:, :, ch] = np.clip(c_edge[ch] + t * (c_center[ch] - c_edge[ch]), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").convert("RGBA")


def generate_ai_video_clip(product: Image.Image, scene_style: str, aspect_ratio: str, out_mp4: Path):
    dims = {"9:16": (540, 960), "16:9": (960, 540)}
    W, H = dims.get(aspect_ratio, (540, 960))
    fps = 24
    total_frames = 54

    hero = product.copy()
    hero.thumbnail((int(W * 0.62), int(H * 0.46)), Image.Resampling.LANCZOS)
    pw, ph = hero.size
    rng = random.Random(hash(scene_style) & 0xFFFF)

    if scene_style == "pedestal_blinds":
        bg = _make_gradient_bg(W, H, (198, 192, 182), (242, 238, 230))
        d = ImageDraw.Draw(bg)
        ped_y = int(H * 0.72)
        d.rectangle([int(W * 0.10), ped_y + 24, int(W * 0.34), H], fill=(225, 222, 216, 255))
        d.rectangle([int(W * 0.66), ped_y + 24, int(W * 0.90), H], fill=(225, 222, 216, 255))
        d.rectangle([int(W * 0.30), ped_y, int(W * 0.70), H], fill=(246, 244, 240, 255), outline=(210, 206, 198, 255), width=2)

        blinds = Image.new("RGBA", (W + 240, H + 240), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blinds)
        for off in range(-H - 240, W + H + 240, 95):
            bd.polygon([(off, 0), (off + 46, 0), (off - H - 240 + 46, H + 240), (off - H - 240, H + 240)], fill=(25, 22, 18, 42))
        blinds = blinds.filter(ImageFilter.GaussianBlur(6))

        side_prod = product.copy()
        side_prod.thumbnail((int(pw * 0.56), int(ph * 0.56)), Image.Resampling.LANCZOS)
        sw, sh = side_prod.size

    elif scene_style == "twin_float":
        bg = _make_gradient_bg(W, H, (190, 198, 208), (241, 245, 249))
        twin_b = hero.copy()
        twin_b.thumbnail((int(pw * 0.68), int(ph * 0.68)), Image.Resampling.LANCZOS)

    elif scene_style == "macro_detail":
        bg = _make_gradient_bg(W, H, (10, 14, 24), (28, 38, 58))
        macro = product.copy()
        macro.thumbnail((int(W * 1.35), int(H * 1.15)), Image.Resampling.LANCZOS)
        mw, mh = macro.size

    elif scene_style == "chrome_finale":
        bg = _make_gradient_bg(W, H, (5, 7, 13), (26, 36, 58))
        sd = ImageDraw.Draw(bg)
        sd.polygon([(W * 0.25, 0), (W * 0.75, 0), (W * 0.92, H), (W * 0.08, H)], fill=(148, 197, 255, 20))
        sparks = [(rng.uniform(0, 2 * math.pi), rng.uniform(W * 0.22, W * 0.44), rng.randint(3, 8), rng.uniform(0.8, 2.0)) for _ in range(30)]

    else:
        bg = _make_gradient_bg(W, H, (6, 11, 22), (20, 40, 78))
        grid = 6
        tiles = []
        tw_t, th_t = max(1, pw // grid), max(1, ph // grid)
        for gy in range(grid):
            for gx in range(grid):
                x0, y0 = gx * tw_t, gy * th_t
                x1 = pw if gx == grid - 1 else (gx + 1) * tw_t
                y1 = ph if gy == grid - 1 else (gy + 1) * th_t
                tiles.append((hero.crop((x0, y0, x1, y1)), x0, y0, rng.randint(-W // 2, W // 2), rng.randint(-H // 3, H // 3)))
        frags = [(rng.randint(0, W), rng.randint(0, H), rng.randint(6, 14), rng.uniform(1.0, 2.4)) for _ in range(24)]

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-threads", "1", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for f in range(total_frames):
        p = f / max(1, total_frames - 1)
        frame = bg.copy()

        if scene_style == "pedestal_blinds":
            pan = int((p - 0.5) * 32)
            frame.paste(side_prod, (int(W * 0.22 - sw // 2) + pan, int(H * 0.72 + 24 - sh)), side_prod)
            frame.paste(side_prod, (int(W * 0.78 - sw // 2) + pan, int(H * 0.72 + 24 - sh)), side_prod)
            frame.paste(hero, (W // 2 - pw // 2 + pan, int(H * 0.72 - ph)), hero)
            shift = int(p * 85)
            frame = Image.alpha_composite(frame, blinds.crop((shift, shift, shift + W, shift + H)))

        elif scene_style == "twin_float":
            bob1 = int(math.sin(p * math.pi * 2) * 18)
            bob2 = int(math.sin(p * math.pi * 2 + math.pi) * 20)
            rot1 = hero.rotate(math.sin(p * math.pi * 2) * 5, expand=True, resample=Image.Resampling.BILINEAR)
            rot2 = twin_b.rotate(-math.sin(p * math.pi * 2) * 7, expand=True, resample=Image.Resampling.BILINEAR)
            frame.paste(rot2, (int(W * 0.68 - rot2.width // 2), int(H * 0.34 - rot2.height // 2 + bob2)), rot2)
            frame.paste(rot1, (int(W * 0.38 - rot1.width // 2), int(H * 0.50 - rot1.height // 2 + bob1)), rot1)

        elif scene_style == "macro_detail":
            ox = int((W - mw) * (0.2 + 0.6 * p))
            oy = int((H - mh) * (0.25 + 0.5 * p))
            frame.paste(macro, (ox, oy), macro)
            streak = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sx = int(-W * 0.3 + p * W * 1.6)
            ImageDraw.Draw(streak).polygon([(sx, 0), (sx + 90, 0), (sx - H // 2 + 90, H), (sx - H // 2, H)], fill=(255, 255, 255, 34))
            frame = Image.alpha_composite(frame, streak.filter(ImageFilter.GaussianBlur(10)))

        elif scene_style == "chrome_finale":
            sc = 0.96 + 0.08 * p
            hp = hero.resize((int(pw * sc), int(ph * sc)), Image.Resampling.BILINEAR)
            frame.paste(hp, (W // 2 - hp.width // 2, H // 2 - hp.height // 2), hp)
            sd = ImageDraw.Draw(frame)
            for ang, rad, sz, spd in sparks:
                a = ang + p * spd * 3.2
                cx = int(W // 2 + math.cos(a) * rad)
                cy = int(H // 2 + math.sin(a) * rad * 0.65)
                sd.polygon([(cx, cy - sz), (cx + sz, cy), (cx, cy + sz), (cx - sz, cy)], fill=(226, 232, 240, 210))

        else:
            ease = 1.0 - (1.0 - min(1.0, p / 0.55)) ** 3
            hx, hy = W // 2 - pw // 2, int(H * 0.48 - ph // 2 + math.sin(p * 6) * 8)
            if p < 0.55:
                for piece, x0, y0, dx, dy in tiles:
                    frame.paste(piece, (int(hx + x0 + dx * (1 - ease)), int(hy + y0 + dy * (1 - ease))), piece)
            else:
                sc = 1.0 + (p - 0.55) * 0.18
                hp = hero.resize((int(pw * sc), int(ph * sc)), Image.Resampling.BILINEAR)
                frame.paste(hp, (W // 2 - hp.width // 2, int(H * 0.48 - hp.height // 2)), hp)
            fd = ImageDraw.Draw(frame)
            for fx, fy, fsz, fspd in frags:
                ny = int((fy - p * 120 * fspd) % H)
                fd.polygon([(fx, ny), (fx + fsz, ny + fsz // 2), (fx + fsz // 3, ny + fsz)], fill=(56, 189, 248, 150))

        proc.stdin.write(frame.convert("RGB").tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode(errors="ignore")
    if proc.wait() != 0:
        raise RuntimeError("FFmpeg clip render failed: " + err[-250:])
    del bg
    gc.collect()


def build_luxury_commercial(
    job_dir: Path,
    raw_clips: list[Path],
    voice_mp3: Path,
    plan: dict,
    aspect_ratio: str,
) -> Path:
    w, h = (540, 960) if aspect_ratio == "9:16" else (960, 540)
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

    styled_clips = []
    title_size = 72 if aspect_ratio == "9:16" else 84

    for idx, clip in enumerate(raw_clips):
        styled = job_dir / f"styled_{idx}.mp4"
        overlay = ""
        if idx == 0:
            overlay = (
                f",drawtext=fontfile={serif_font}:textfile={hero_file}:fontcolor=white@0.95:"
                f"fontsize={title_size}:shadowcolor=black@0.45:shadowx=3:shadowy=4:"
                f"x=(w-text_w)/2:y=(h-text_h)/2+30"
            )
        elif idx == 2:
            overlay = (
                f",drawtext=fontfile={sans_font}:textfile={tag_file}:fontcolor=white:"
                f"fontsize=26:box=1:boxcolor=black@0.55:boxborderw=10:x=(w-text_w)/2:y=h-120"
            )
        elif idx == 4:
            overlay = (
                f",drawtext=fontfile={sans_font}:textfile={outro_file}:fontcolor=white@0.95:"
                f"fontsize=28:shadowcolor=black@0.5:shadowx=2:shadowy=2:"
                f"x=(w-text_w)/2:y=(h-text_h)/2"
            )

        out_st = max(0.5, get_duration(clip) * 0.85 - 0.25)
        vf = (
            f"setpts=0.85*PTS,scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},fps=24,eq=contrast=1.10:saturation=1.15{overlay},"
            f"fade=t=in:st=0:d=0.18,fade=t=out:st={out_st:.2f}:d=0.20"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(clip),
                "-vf", vf, "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-pix_fmt", "yuv420p", "-threads", "1", str(styled),
            ],
            check=True,
        )
        clip.unlink(missing_ok=True)
        styled_clips.append(styled)
        gc.collect()

    merged_video = job_dir / "merged_video.mp4"
    concat_file = job_dir / "concat.txt"
    concat_file.write_text("\n".join(f"file '{c.resolve()}'" for c in styled_clips), encoding="utf-8")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(merged_video)],
        check=True,
    )
    for c in styled_clips:
        c.unlink(missing_ok=True)

    vid_dur = get_duration(merged_video)
    aud_dur = get_duration(voice_mp3)
    target_dur = max(vid_dur, aud_dur + 0.5)
    pad_sec = round(max(0.0, target_dur - vid_dur), 2)

    final_mp4 = job_dir / "final_ad.mp4"
    audio_filter = (
        f"aevalsrc='0.09*sin(2*PI*55*t)*exp(-4*mod(t,0.5))+0.04*sin(2*PI*165*t)*exp(-8*mod(t,0.25))':s=44100:d={target_dur:.2f}[bg];"
        f"[1:a]aresample=44100,volume=1.40[vo];"
        f"[vo][bg]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
    )
    video_filter = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_sec:.2f}[vfinal]"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(merged_video),
                "-i", str(voice_mp3),
                "-filter_complex", f"{video_filter};{audio_filter}",
                "-map", "[vfinal]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-t", f"{target_dur:.2f}", "-movflags", "+faststart", "-threads", "1",
                str(final_mp4),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(merged_video),
                "-i", str(voice_mp3),
                "-vf", f"tpad=stop_mode=clone:stop_duration={pad_sec:.2f}",
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-t", f"{target_dur:.2f}", "-movflags", "+faststart", "-threads", "1",
                str(final_mp4),
            ],
            check=True,
        )
    merged_video.unlink(missing_ok=True)
    gc.collect()
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

        product_img = _extract_product_light(image_path.read_bytes())
        raw_clips = []
        for idx, scene in enumerate(plan["scenes"]):
            JOBS[job_id]["step"] = f"3D CGI AI Scene {idx + 1}/5 ({scene['scene_style']}) render ho raha hai..."
            JOBS[job_id]["progress"] = 30 + (idx * 12)
            clip_path = job_dir / f"raw_{idx}.mp4"
            generate_ai_video_clip(product_img, scene["scene_style"], aspect_ratio, clip_path)
            raw_clips.append(clip_path)

        del product_img
        gc.collect()

        JOBS[job_id]["step"] = "Beat-synced luxury transitions & typography edit ho rahe hain..."
        JOBS[job_id]["progress"] = 92
        build_luxury_commercial(job_dir, raw_clips, voice_mp3, plan, aspect_ratio)

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
    now = time.time()
    for old in WORK_DIR.iterdir():
        if old.is_dir() and (now - old.stat().st_mtime) > 600:
            shutil.rmtree(old, ignore_errors=True)

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
