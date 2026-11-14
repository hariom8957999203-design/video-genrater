import asyncio
import base64
import gc
import io
import json
import math
import os
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
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="AI TV Commercial Ad Studio")

WORK_DIR = Path("/tmp/ad_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
FAL_KEY = os.environ.get("FAL_KEY", "").strip().strip('"').strip("'")
FAL_VIDEO_MODEL = os.environ.get("FAL_VIDEO_MODEL", "fal-ai/wan/v2.2-a14b/image-to-video").strip()
USE_GEMINI_OMNI = os.environ.get("USE_GEMINI_OMNI", "false").lower() in ("1", "true", "yes")


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


def _fallback_commercial_plan(image_path: Path, user_hint: str) -> dict:
    raw = image_path.stem.replace("_", " ").replace("-", " ").strip()
    brand = "".join(c for c in (raw.split()[0] if raw else "PRIME") if c.isalnum()).upper() or "PRIME"
    hint_clean = (user_hint or "everyday luxury and smart performance").strip()

    return {
        "brand_name": brand[:16],
        "headline_top": "NEXT-GEN POWER",
        "headline_bottom": "BUILT TO WIN",
        "signposts": ["Pro Quality", "Smart Design", "Built Strong"],
        "voiceover_script": (
            f"Looking for something truly unstoppable? Meet the all-new {brand}! "
            f"Engineered for {hint_clean}, it brings together pro quality, smart design, and unmatched performance. "
            f"Upgrade your everyday — grab your {brand} today!"
        ),
        "scene_prompts": [
            (
                f"Photorealistic TV commercial frame: Warm sunlit modern lifestyle scene where happy real people "
                f"are actively using and admiring the EXACT product from the reference image. Keep product identical, no text overlays."
            ),
            (
                f"Photorealistic extreme macro commercial close-up showing the EXACT product from the reference image "
                f"in active slow-motion use, dramatic studio rim light, rich contrast, no text overlays."
            ),
            (
                f"High-speed 3D CGI commercial explosion shot: dynamic glowing particles, energy waves, or dramatic studio light streaks "
                f"swirling around the EXACT product from the reference image, no text overlays."
            ),
            (
                f"Clean hero TV commercial packshot: the EXACT product from the reference image standing boldly on a sleek "
                f"warm tabletop with soft sunlight in the background, clear space at the top for headline text, no text overlays."
            ),
        ],
    }


def plan_commercial_with_gemini(image_path: Path, user_hint: str) -> dict:
    if not GEMINI_API_KEY:
        return _fallback_commercial_plan(image_path, user_hint)

    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)

    prompt = f"""
You are an award-winning TV Commercial Director (like viral Kellogg's / Apple / Nike 20-second ads).
Look carefully at the exact product in the uploaded image and the user's note: "{user_hint or 'Make a high-converting 3D cinematic TV commercial in English'}".

Design a 4-scene commercial:
- Scene 1 (Lifestyle Hook): Real people naturally excited and using THIS EXACT product.
- Scene 2 (Hero Action Macro): Close-up shot demonstrating THIS EXACT product's main function in action.
- Scene 3 (3D CGI Splash/Energy): Dramatic 3D CGI slow-motion splash/particles/energy swirling around THIS EXACT product.
- Scene 4 (Final Packshot): Clean tabletop product packshot with space at top for bold headline text.

Also provide:
- 3 ultra-short punchy signpost callout labels (2 words max each, like "High Protein", "High Fibre", "No Maida" — tailored to THIS product!).
- A bold 2-line packshot headline (`headline_top` and `headline_bottom`, 2-3 words each).
- An energetic, persuasive ENGLISH commercial voiceover script (35-45 words).

Return ONLY valid JSON:
{{
  "brand_name": "Short product/brand name",
  "headline_top": "2-3 WORD UPPERCASE LINE 1",
  "headline_bottom": "2-3 WORD UPPERCASE LINE 2",
  "signposts": ["Benefit 1", "Benefit 2", "Benefit 3"],
  "voiceover_script": "Energetic 35-45 word English TV commercial voiceover script with hook, 3 benefits, and buy-now CTA.",
  "scene_prompts": [
    "Detailed prompt for Scene 1 keeping the exact reference product identical, no text overlays.",
    "Detailed prompt for Scene 2 keeping the exact reference product identical, no text overlays.",
    "Detailed prompt for Scene 3 keeping the exact reference product identical, no text overlays.",
    "Detailed prompt for Scene 4 clean tabletop packshot keeping the exact reference product identical, space at top, no text overlays."
  ]
}}
"""
    for model_name in ("gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite"):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[prompt, img],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            txt = (resp.text or "").strip()
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(txt)
            if len(data.get("scene_prompts", [])) >= 4 and len(data.get("signposts", [])) >= 3:
                return data
        except Exception as e:
            print(f"Planner failed on {model_name}: {repr(e)}", flush=True)
            continue

    return _fallback_commercial_plan(image_path, user_hint)


async def create_english_voiceover(text: str, out_mp3: Path, voice_accent: str):
    voice_map = {
        "us_female": "en-US-AriaNeural",
        "uk_female": "en-GB-SoniaNeural",
        "in_female": "en-IN-NeerjaNeural",
        "us_male": "en-US-GuyNeural",
    }
    voice = voice_map.get(voice_accent, "en-US-AriaNeural")
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+7%")
        await communicate.save(str(out_mp3))
    except Exception as e:
        print(f"Edge-TTS failed ({e}), using gTTS fallback...", flush=True)
        gTTS(text=text, lang="en", tld="com").save(str(out_mp3))


def generate_scene_keyframe(ref_image: Image.Image, prompt: str, aspect_ratio: str, out_png: Path) -> Path:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for img_model in ("gemini-3.1-flash-image-preview", "gemini-2.5-flash-image"):
            try:
                resp = client.models.generate_content(
                    model=img_model,
                    contents=[prompt, ref_image],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                    ),
                )
                for cand in getattr(resp, "candidates", []) or []:
                    for part in getattr(cand.content, "parts", []) or []:
                        if getattr(part, "inline_data", None) and part.inline_data.data:
                            gen = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                            gen.save(out_png, format="PNG")
                            return out_png
            except Exception as e:
                print(f"Gemini Image ({img_model}) failed: {repr(e)}", flush=True)
                continue

    if FAL_KEY:
        try:
            buf = io.BytesIO()
            ref_image.save(buf, format="PNG")
            data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
            with httpx.Client(timeout=180.0) as hc:
                sub = hc.post(
                    "https://queue.fal.run/fal-ai/flux-pro/kontext",
                    headers=headers,
                    json={"prompt": prompt, "image_url": data_uri, "aspect_ratio": aspect_ratio},
                )
                sub.raise_for_status()
                q = sub.json()
                for _ in range(60):
                    time.sleep(2)
                    st = hc.get(q["status_url"], headers=headers).json()
                    if st.get("status") == "COMPLETED":
                        res = hc.get(q["response_url"], headers=headers).json()
                        img_url = res["images"][0]["url"]
                        out_png.write_bytes(hc.get(img_url).content)
                        return out_png
                    if st.get("status") in ("FAILED", "CANCELLED"):
                        break
        except Exception as e:
            print(f"Fal Kontext failed: {repr(e)}", flush=True)

    W, H = (540, 960) if aspect_ratio == "9:16" else (960, 540)
    bg = Image.new("RGB", (W, H), (14, 18, 28))
    prod = ref_image.copy()
    prod.thumbnail((W, H), Image.Resampling.LANCZOS)
    bg.paste(prod, ((W - prod.width) // 2, (H - prod.height) // 2))
    bg.save(out_png, format="PNG")
    return out_png


def animate_keyframe_with_ai_video(keyframe_png: Path, motion_prompt: str, aspect_ratio: str, out_mp4: Path) -> bool:
    if FAL_KEY:
        try:
            data_uri = "data:image/png;base64," + base64.b64encode(keyframe_png.read_bytes()).decode("utf-8")
            headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
            with httpx.Client(timeout=240.0) as hc:
                sub = hc.post(
                    f"https://queue.fal.run/{FAL_VIDEO_MODEL}",
                    headers=headers,
                    json={"image_url": data_uri, "prompt": motion_prompt, "aspect_ratio": aspect_ratio},
                )
                sub.raise_for_status()
                q = sub.json()
                for _ in range(90):
                    time.sleep(3)
                    st = hc.get(q["status_url"], headers=headers).json()
                    if st.get("status") == "COMPLETED":
                        res = hc.get(q["response_url"], headers=headers).json()
                        vurl = res.get("video", {}).get("url")
                        if vurl:
                            out_mp4.write_bytes(hc.get(vurl).content)
                            return True
                        break
                    if st.get("status") in ("FAILED", "CANCELLED"):
                        break
        except Exception as e:
            print(f"Fal video failed: {repr(e)}", flush=True)

    if USE_GEMINI_OMNI and GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            img_b64 = base64.b64encode(keyframe_png.read_bytes()).decode("utf-8")
            interaction = client.interactions.create(
                model="gemini-omni-1.2-flash",
                input=[
                    {"type": "image", "data": img_b64, "mime_type": "image/png"},
                    {"type": "text", "text": motion_prompt},
                ],
                response_format={"type": "video", "aspect_ratio": aspect_ratio, "resolution": "720p", "duration": "5s"},
            )
            vbytes = None
            if getattr(interaction, "output_video", None) and getattr(interaction.output_video, "data", None):
                vbytes = base64.b64decode(interaction.output_video.data)
            elif getattr(interaction, "steps", None):
                for step in interaction.steps:
                    for item in getattr(step, "content", []) or []:
                        if getattr(item, "type", None) == "video" and getattr(item, "data", None):
                            vbytes = base64.b64decode(item.data)
                            break
            if vbytes:
                out_mp4.write_bytes(vbytes)
                return True
        except Exception as e:
            print(f"Gemini Omni failed: {repr(e)}", flush=True)

    return False


def animate_keyframe_cinematic(keyframe_png: Path, mode: str, aspect_ratio: str, duration_sec: float, out_mp4: Path):
    """Low-RAM, sharp 540p/960p cinematic camera motion."""
    W, H = (540, 960) if aspect_ratio == "9:16" else (960, 540)
    fps = 24
    total_frames = int(duration_sec * fps)

    src = Image.open(keyframe_png).convert("RGB")
    src = src.resize((int(W * 1.16), int(H * 1.16)), Image.Resampling.LANCZOS)
    sw, sh = src.size

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency", "-crf", "20",
        "-pix_fmt", "yuv420p", "-threads", "1", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for f in range(total_frames):
        p = f / max(1, total_frames - 1)
        if mode == "zoom_in":
            scale = 1.0 + 0.13 * p
            ox = int((sw - W / scale) * 0.5)
            oy = int((sh - H / scale) * 0.5)
        elif mode == "pan_right":
            scale = 1.08
            ox = int((sw - W / scale) * (0.15 + 0.7 * p))
            oy = int((sh - H / scale) * 0.5)
        elif mode == "dramatic_push":
            scale = 1.02 + 0.14 * (p ** 0.85)
            ox = int((sw - W / scale) * (0.5 + 0.05 * math.sin(p * math.pi * 2)))
            oy = int((sh - H / scale) * 0.48)
        else:
            scale = 1.11 - 0.09 * p
            ox = int((sw - W / scale) * 0.5)
            oy = int((sh - H / scale) * 0.5)

        cw, ch = int(W / scale), int(H / scale)
        ox = max(0, min(sw - cw, ox))
        oy = max(0, min(sh - ch, oy))
        frame = src.crop((ox, oy, ox + cw, oy + ch)).resize((W, H), Image.Resampling.BILINEAR)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode(errors="ignore")
    src.close()
    del src
    gc.collect()
    if proc.wait() != 0:
        raise RuntimeError("FFmpeg render failed: " + err[-250:])


def _make_signpost_png(label: str, angle: float, out_png: Path) -> tuple[int, int]:
    """Creates a lightweight static transparent signpost PNG (zero RAM overhead)."""
    font = _load_font(26)
    words = label.strip().split()
    line1 = words[0] if words else "PRO"
    line2 = " ".join(words[1:]) if len(words) > 1 else ""
    bw, bh = 185, 110 if line2 else 74

    card = Image.new("RGBA", (bw, bh + 125), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rectangle([bw // 2 - 5, bh - 3, bw // 2 + 5, bh + 120], fill=(203, 166, 134, 255), outline=(92, 61, 46, 255), width=2)
    d.rounded_rectangle([3, 3, bw - 3, bh - 3], radius=12, fill=(255, 251, 240, 255), outline=(74, 44, 28, 255), width=4)

    if line2:
        w1 = d.textlength(line1, font=font)
        w2 = d.textlength(line2, font=font)
        d.text((bw // 2 - w1 // 2, 14), line1, font=font, fill=(60, 32, 18, 255))
        d.text((bw // 2 - w2 // 2, 52), line2, font=font, fill=(60, 32, 18, 255))
    else:
        w1 = d.textlength(line1, font=font)
        d.text((bw // 2 - w1 // 2, 20), line1, font=font, fill=(60, 32, 18, 255))

    rot = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    rot.save(out_png, format="PNG")
    return rot.size


def _create_packshot_headline_png(W: int, H: int, top_line: str, bottom_line: str, out_png: Path):
    banner = Image.new("RGBA", (W, int(H * 0.34)), (0, 0, 0, 0))
    f_top = _load_font(40 if W < H else 48)
    f_bot = _load_font(38 if W < H else 46)

    txt_layer = Image.new("RGBA", (W, int(H * 0.32)), (0, 0, 0, 0))
    d = ImageDraw.Draw(txt_layer)
    t1 = top_line.upper()[:24]
    t2 = bottom_line.upper()[:24]

    w1 = d.textlength(t1, font=f_top)
    w2 = d.textlength(t2, font=f_bot)
    y1, y2 = 34, 88

    d.text((W // 2 - w1 // 2 + 3, y1 + 4), t1, font=f_top, fill=(0, 0, 0, 175))
    d.text((W // 2 - w1 // 2, y1), t1, font=f_top, fill=(250, 204, 21, 255))

    d.text((W // 2 - w2 // 2 + 3, y2 + 4), t2, font=f_bot, fill=(0, 0, 0, 175))
    d.text((W // 2 - w2 // 2, y2), t2, font=f_bot, fill=(255, 255, 255, 255))

    tilted = txt_layer.rotate(5, expand=False, resample=Image.Resampling.BICUBIC)
    banner.alpha_composite(tilted, (0, 8))
    banner.save(out_png, format="PNG")


def assemble_commercial(
    job_dir: Path,
    scene_videos: list[Path],
    voice_mp3: Path,
    plan: dict,
    aspect_ratio: str,
) -> Path:
    W, H = (540, 960) if aspect_ratio == "9:16" else (960, 540)

    # 1) Normalize Scene 1 & 2 cleanly (low RAM, 1 thread)
    norm_clips = []
    for idx in range(4):
        sc = scene_videos[idx]
        norm = job_dir / f"norm_{idx}.mp4"
        dur = get_duration(sc)
        out_st = max(0.4, dur - 0.16)
        base_vf = (
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps=24,"
            f"eq=contrast=1.05:saturation=1.08,fade=t=in:st=0:d=0.14,fade=t=out:st={out_st:.2f}:d=0.14"
        )

        if idx == 2:
            # Overlay 3 lightweight static signpost PNGs directly in FFmpeg!
            labels = plan.get("signposts", ["Pro Quality", "Smart Design", "Built Strong"])[:3]
            angles = (-6, 5, -4)
            sign_paths = []
            sign_sizes = []
            for s_i, lbl in enumerate(labels):
                sp = job_dir / f"sign_{s_i}.png"
                sz = _make_signpost_png(lbl, angles[s_i], sp)
                sign_paths.append(sp)
                sign_sizes.append(sz)

            seg = dur / 3.0
            t_bounds = [(0.05, seg), (seg, seg * 2), (seg * 2, dur)]
            coords = [
                (int(W * 0.08), int(H * 0.36)),
                (max(12, W - sign_sizes[1][0] - int(W * 0.08)), int(H * 0.25)),
                ((W - sign_sizes[2][0]) // 2, int(H * 0.33)),
            ]
            fc = (
                f"[0:v]{base_vf}[b0];"
                f"[b0][1:v]overlay=x={coords[0][0]}:y='{coords[0][1]}+70*max(0\\,1-(t-{t_bounds[0][0]:.2f})/0.22)':enable='between(t\\,{t_bounds[0][0]:.2f}\\,{t_bounds[0][1]:.2f})'[b1];"
                f"[b1][2:v]overlay=x={coords[1][0]}:y='{coords[1][1]}+70*max(0\\,1-(t-{t_bounds[1][0]:.2f})/0.22)':enable='between(t\\,{t_bounds[1][0]:.2f}\\,{t_bounds[1][1]:.2f})'[b2];"
                f"[b2][3:v]overlay=x={coords[2][0]}:y='{coords[2][1]}+70*max(0\\,1-(t-{t_bounds[2][0]:.2f})/0.22)':enable='between(t\\,{t_bounds[2][0]:.2f}\\,{t_bounds[2][1]:.2f})'[v]"
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(sc),
                    "-i", str(sign_paths[0]), "-i", str(sign_paths[1]), "-i", str(sign_paths[2]),
                    "-filter_complex", fc, "-map", "[v]", "-an",
                    "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-threads", "1", str(norm),
                ],
                check=True,
            )
            for sp in sign_paths:
                sp.unlink(missing_ok=True)

        elif idx == 3:
            headline_png = job_dir / "headline.png"
            _create_packshot_headline_png(
                W, H,
                plan.get("headline_top", "PURE ENERGY"),
                plan.get("headline_bottom", "BUILT TO WIN"),
                headline_png,
            )
            fc = f"[0:v]{base_vf}[b0];[b0][1:v]overlay=0:18[v]"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(sc), "-i", str(headline_png),
                    "-filter_complex", fc, "-map", "[v]", "-an",
                    "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-threads", "1", str(norm),
                ],
                check=True,
            )
            headline_png.unlink(missing_ok=True)

        else:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(sc), "-vf", base_vf, "-an",
                    "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-threads", "1", str(norm),
                ],
                check=True,
            )

        sc.unlink(missing_ok=True)
        norm_clips.append(norm)
        gc.collect()

    merged_video = job_dir / "merged.mp4"
    concat_txt = job_dir / "concat.txt"
    concat_txt.write_text("\n".join(f"file '{c.resolve()}'" for c in norm_clips), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(merged_video)],
        check=True,
    )
    for c in norm_clips:
        c.unlink(missing_ok=True)

    vid_dur = get_duration(merged_video)
    aud_dur = get_duration(voice_mp3)
    target_dur = max(vid_dur, aud_dur + 0.6)
    pad_sec = round(max(0.0, target_dur - vid_dur), 2)

    final_mp4 = job_dir / "final_ad.mp4"
    audio_filter = (
        f"aevalsrc='0.05*sin(2*PI*65*t)*exp(-5*mod(t,0.45))+0.025*sin(2*PI*195*t)*exp(-9*mod(t,0.225))':s=44100:d={target_dur:.2f}[bg];"
        f"[1:a]aresample=44100,volume=1.55[vo];"
        f"[vo][bg]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
    )
    video_filter = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_sec:.2f}[vfinal]"

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(merged_video), "-i", str(voice_mp3),
            "-filter_complex", f"{video_filter};{audio_filter}",
            "-map", "[vfinal]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-t", f"{target_dur:.2f}",
            "-movflags", "+faststart", "-threads", "1", str(final_mp4),
        ],
        check=True,
    )
    merged_video.unlink(missing_ok=True)
    gc.collect()
    return final_mp4


def run_pipeline(job_id: str, image_path: Path, hint: str, voice_accent: str, aspect_ratio: str):
    job_dir = image_path.parent
    try:
        save_job(job_id, step="Planning 4-scene TV commercial & English script...", progress=10)
        plan = plan_commercial_with_gemini(image_path, hint)
        save_job(job_id, script=plan["voiceover_script"])

        save_job(job_id, step="Recording energetic English commercial voiceover...", progress=20)
        voice_mp3 = job_dir / "voice.mp3"
        asyncio.run(create_english_voiceover(plan["voiceover_script"], voice_mp3, voice_accent))

        ref_img = Image.open(image_path).convert("RGB")
        ref_img.thumbnail((800, 800), Image.Resampling.LANCZOS)

        scene_prompts = plan["scene_prompts"][:4]
        motion_modes = [("zoom_in", 4.5), ("pan_right", 3.8), ("dramatic_push", 5.2), ("gentle_settle", 3.5)]
        scene_videos = []

        for idx, sc_prompt in enumerate(scene_prompts):
            save_job(job_id, step=f"Generating AI Commercial Scene {idx + 1}/4...", progress=28 + idx * 14)
            kf_png = job_dir / f"keyframe_{idx}.png"
            generate_scene_keyframe(ref_img, sc_prompt, aspect_ratio, kf_png)

            sc_mp4 = job_dir / f"scene_{idx}.mp4"
            save_job(job_id, step=f"Animating Scene {idx + 1}/4...", progress=34 + idx * 14)
            mode, dur = motion_modes[idx]
            if not animate_keyframe_with_ai_video(kf_png, sc_prompt, aspect_ratio, sc_mp4):
                animate_keyframe_cinematic(kf_png, mode, aspect_ratio, dur, sc_mp4)

            kf_png.unlink(missing_ok=True)
            scene_videos.append(sc_mp4)
            gc.collect()

        ref_img.close()
        del ref_img
        gc.collect()

        save_job(job_id, step="Adding Signpost Callouts, Packshot Headline & English Voiceover...", progress=90)
        assemble_commercial(job_dir, scene_videos, voice_mp3, plan, aspect_ratio)

        save_job(
            job_id,
            status="done",
            progress=100,
            step="Your TV-style 3D Commercial Ad is ready!",
            video_url=f"/download/{job_id}",
        )
    except Exception as exc:
        print(f"Pipeline error ({job_id}): {repr(exc)}", flush=True)
        save_job(job_id, status="error", step=f"Error: {str(exc)}")


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "gemini_configured": bool(GEMINI_API_KEY),
        "fal_configured": bool(FAL_KEY),
        "gemini_omni": USE_GEMINI_OMNI,
    }


@app.post("/generate")
async def generate_ad(
    image: UploadFile = File(...),
    hint: str = Form(""),
    voice_accent: str = Form("us_female"),
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

    save_job(job_id, status="processing", progress=5, step="Starting...", script="")
    threading.Thread(
        target=run_pipeline,
        args=(job_id, img_path, hint, voice_accent, aspect_ratio),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    data = load_job(job_id)
    if not data:
        return JSONResponse({"status": "error", "step": "Job not found"}, status_code=404)
    return data


@app.get("/download/{job_id}")
def download_video(job_id: str):
    video_path = WORK_DIR / job_id / "final_ad.mp4"
    if not video_path.exists():
        return JSONResponse({"error": "Video not ready"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4", filename=f"tv_commercial_{job_id}.mp4")


@app.get("/", response_class=HTMLResponse)
def home():
    video_ai = bool(FAL_KEY or USE_GEMINI_OMNI)
    badge = (
        '<span style="color:#10b981;font-weight:700;">● Full AI Video Motion + Scene Generation Active</span>'
        if video_ai
        else '<span style="color:#38bdf8;font-weight:700;">● Gemini AI Scene Staging + Cinematic Motion Active (Add FAL_KEY for full AI video motion)</span>'
    )
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI TV Commercial Ad Studio</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #070b14; color: #f8fafc; margin: 0; padding: 24px; }}
    .card {{ max-width: 620px; margin: 0 auto; background: #111827; padding: 28px; border-radius: 16px; border: 1px solid #1f2937; }}
    h1 {{ margin-top: 0; font-size: 1.55rem; color: #facc15; }}
    label {{ display: block; margin-top: 16px; font-weight: 600; font-size: 0.92rem; color: #cbd5e1; }}
    input, textarea, select {{ width: 100%; margin-top: 6px; padding: 10px 12px; border-radius: 10px; border: 1px solid #334155; background: #0b1120; color: #fff; box-sizing: border-box; }}
    button {{ width: 100%; margin-top: 22px; padding: 14px; border: none; border-radius: 10px; background: linear-gradient(90deg, #f59e0b, #ef4444); color: #fff; font-weight: 700; font-size: 1rem; cursor: pointer; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .progress-box {{ display: none; margin-top: 20px; background: #0b1120; padding: 14px; border-radius: 10px; }}
    .bar-bg {{ width: 100%; height: 10px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-top: 8px; }}
    .bar-fill {{ height: 100%; width: 0%; background: #facc15; transition: width 0.4s; }}
    video {{ width: 100%; max-height: 540px; border-radius: 12px; margin-top: 16px; background: #000; }}
    .dl-btn {{ display: inline-block; margin-top: 12px; padding: 10px 18px; background: #10b981; color: #fff; text-decoration: none; border-radius: 8px; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>🎬 AI TV-Style Product Commercial Studio</h1>
    <p style="font-size:0.85rem;margin-top:-6px;">{badge}</p>
    <p style="color:#94a3b8;font-size:0.9rem;">Upload your product photo. AI generates a 4-scene lifestyle + 3D CGI splash commercial with pop-up benefit signposts, slanted packshot headline & English voiceover.</p>
    <form id="adForm">
      <label>1. Upload Product Photo</label>
      <input type="file" name="image" accept="image/*" required />

      <label>2. Product Benefits / Ad Concept (Optional)</label>
      <textarea name="hint" rows="2" placeholder="e.g. Chocolate multigrain snack, high protein, no maida, mom & kid breakfast vibe"></textarea>

      <label>3. English Voiceover</label>
      <select name="voice_accent">
        <option value="us_female" selected>US Commercial Female (Aria)</option>
        <option value="uk_female">British Commercial Female (Sonia)</option>
        <option value="in_female">Indian English Female (Neerja)</option>
        <option value="us_male">US Energetic Male (Guy)</option>
      </select>

      <label>4. Format</label>
      <select name="aspect_ratio">
        <option value="9:16" selected>Reels / Shorts (9:16)</option>
        <option value="16:9">YouTube / TV (16:9)</option>
      </select>

      <button type="submit" id="submitBtn">🚀 Generate Commercial Ad</button>
    </form>

    <div class="progress-box" id="progressBox">
      <div id="statusText">Starting...</div>
      <div class="bar-bg"><div class="bar-fill" id="barFill"></div></div>
      <p id="scriptBox" style="font-size:0.85rem;color:#94a3b8;margin-top:10px;"></p>
    </div>

    <div id="resultBox" style="display:none;">
      <video id="adVideo" controls></video>
      <a id="dlLink" class="dl-btn" download>⬇️ Download Commercial MP4</a>
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

    form.onsubmit = async (e) => {{
      e.preventDefault();
      btn.disabled = true;
      resultBox.style.display = "none";
      progressBox.style.display = "block";
      statusText.innerText = "Uploading product photo...";
      barFill.style.width = "5%";

      const fd = new FormData(form);
      const res = await fetch("/generate", {{ method: "POST", body: fd }});
      const {{ job_id }} = await res.json();

      const timer = setInterval(async () => {{
        const sRes = await fetch(`/status/${{job_id}}`);
        const data = await sRes.json();
        statusText.innerText = data.step || "Processing...";
        barFill.style.width = (data.progress || 10) + "%";
        if (data.script) scriptBox.innerText = "🎙️ English Voiceover: " + data.script;

        if (data.status === "done") {{
          clearInterval(timer);
          btn.disabled = false;
          adVideo.src = data.video_url;
          dlLink.href = data.video_url;
          resultBox.style.display = "block";
        }} else if (data.status === "error") {{
          clearInterval(timer);
          btn.disabled = false;
        }}
      }}, 3000);
    }};
  </script>
</body>
</html>
"""
