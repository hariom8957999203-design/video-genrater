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
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

REMBG_MODEL = os.environ.get("REMBG_MODEL", "isnet-general-use")
try:
    from rembg import new_session, remove as rembg_remove
    try:
        REMBG_SESSION = new_session(REMBG_MODEL)
    except Exception:
        REMBG_SESSION = new_session("u2netp")
    HAS_REMBG = True
except Exception:
    REMBG_SESSION = None
    HAS_REMBG = False

app = FastAPI(title="AI 3D CGI Product Ad Studio")

WORK_DIR = Path("/tmp/ad_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
FAL_KEY = os.environ.get("FAL_KEY", "").strip().strip('"').strip("'")
FAL_MODEL = os.environ.get("FAL_VIDEO_MODEL", "fal-ai/wan/v2.2-a14b/image-to-video").strip()
MODAL_VIDEO_URL = os.environ.get("MODAL_VIDEO_URL", "").strip().strip('"').strip("'")
if MODAL_VIDEO_URL and not MODAL_VIDEO_URL.startswith(("http://", "https://")):
    MODAL_VIDEO_URL = "https://" + MODAL_VIDEO_URL

USE_GEMINI_OMNI = os.environ.get("USE_GEMINI_OMNI", "false").lower() in ("1", "true", "yes")


def get_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def _load_font(size: int, serif: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if serif
        else ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def extract_product_hd(image_bytes: bytes) -> Image.Image:
    """Extracts ONLY the product cleanly at high resolution (no background clutter)."""
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    orig.thumbnail((1280, 1280), Image.Resampling.LANCZOS)

    if HAS_REMBG:
        try:
            buf = io.BytesIO()
            orig.save(buf, format="PNG")
            cutout_bytes = rembg_remove(buf.getvalue(), session=REMBG_SESSION)
            img = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            # Subtle 0.6px anti-alias on alpha edge to remove rough cutout stair-stepping
            r, g, b, a = img.split()
            a = a.filter(ImageFilter.GaussianBlur(0.6))
            img.putalpha(a)
            return img
        except Exception as e:
            print(f"rembg failed, falling back to color/edge cutout: {e}", flush=True)

    arr = np.array(orig, dtype=np.float32)
    h, w, _ = arr.shape
    border = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]], axis=0)
    bg_rgb = np.median(border, axis=0)
    dist = np.linalg.norm(arr[:, :, :3] - bg_rgb, axis=2)

    if float(np.mean(np.std(border, axis=0))) < 48.0:
        alpha = np.clip((dist - 20.0) / 30.0, 0.0, 1.0) * 255.0
        if np.mean(alpha > 128) > 0.10:
            orig.putalpha(Image.fromarray(alpha.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1)))
            bbox = orig.getbbox()
            if bbox:
                orig = orig.crop(bbox)
            return orig

    return orig


def _fallback_ad_plan(image_path: Path, user_hint: str) -> dict:
    raw = image_path.stem.replace("_", " ").replace("-", " ").strip()
    brand = "".join(c for c in (raw.split()[0] if raw else "LUXURY") if c.isalnum()).upper() or "LUXURY"
    hint_clean = (user_hint or "precision 3D engineering and luxury design").strip()

    script = (
        f"Stop settling for ordinary. Meet the all-new {brand}. "
        f"Crafted with {hint_clean}, delivering unmatched precision, effortless performance, and bold luxury design. "
        f"Elevate your everyday — order your {brand} today."
    )

    return {
        "hero_brand": brand[:18],
        "product_name": brand,
        "tagline": "ENGINEERED TO STAND OUT",
        "key_functions": [
            {"title": "PRECISION BUILD", "detail": "Crafted for peak durability & luxury"},
            {"title": "SMART PERFORMANCE", "detail": "Effortless power in every detail"},
            {"title": "STUDIO GRADE FINISH", "detail": "Designed to turn heads everywhere"},
        ],
        "voiceover_script": script,
        "cgi_prompts": [
            f"Ultra-photorealistic 3D CGI commercial shot of this exact {brand} product levitating above a dark obsidian marble pedestal, dramatic rim lighting, floating silver light particles, slow cinematic 3D camera orbit, Unreal Engine 5 Octane render style, 8k sharp product details.",
            f"Cinematic 3D CGI commercial close-up of this exact {brand} product in motion, dynamic volumetric spotlight sweep, subtle mist and micro-droplets in slow motion highlighting its build and function, luxury Apple/Dior commercial aesthetic.",
            f"Hero 3D CGI luxury packshot of this exact {brand} product centered on a sleek reflective chrome surface, dramatic dark studio gradient, glowing volumetric light beams behind product, smooth push-in camera motion."
        ],
        "scenes": [
            {"scene_style": "3d_orbit_reveal", "feature_idx": 0},
            {"scene_style": "function_showcase", "feature_idx": 0},
            {"scene_style": "macro_3d_sweep", "feature_idx": 1},
            {"scene_style": "function_showcase", "feature_idx": 2},
            {"scene_style": "luxury_cta_finale", "feature_idx": 2},
        ],
    }


def analyze_product_with_gemini(image_path: Path, user_hint: str) -> dict:
    if not GEMINI_API_KEY:
        return _fallback_ad_plan(image_path, user_hint)

    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)

    system_prompt = f"""
You are an elite 3D CGI commercial director (like Apple, Nike, Dior, and Sony commercials).
Analyze the EXACT product in the image and the user's brief: "{user_hint or 'Make a cinematic 3D CGI luxury ad showcasing its real functions so customers buy it'}".

Write EVERYTHING in sleek, persuasive, high-converting commercial ENGLISH.

Return ONLY valid JSON:
{{
  "hero_brand": "1-2 word bold brand/product title",
  "product_name": "2-4 word product name",
  "tagline": "2-4 word punchy sales tagline",
  "key_functions": [
    {{"title": "2-3 WORD FUNCTION", "detail": "4-6 word buyer benefit"}},
    {{"title": "2-3 WORD FUNCTION", "detail": "4-6 word buyer benefit"}},
    {{"title": "2-3 WORD FUNCTION", "detail": "4-6 word buyer benefit"}}
  ],
  "voiceover_script": "30 to 38 words in crisp, confident luxury English (approx 12 seconds). Strong hook, real product benefits, urgent call-to-action.",
  "cgi_prompts": [
    "3D CGI commercial prompt 1: 360-degree 3D studio reveal of the exact product on a dark luxury pedestal, volumetric lighting, Octane render, sharp product fidelity, no text.",
    "3D CGI commercial prompt 2: Dramatic 3D CGI action shot showing the exact product demonstrating its primary function in slow motion (e.g., water splash / glowing display / mist spray / precision motion), cinematic lighting, no text.",
    "3D CGI commercial prompt 3: Macro 3D hero packshot of the exact product with gleaming studio reflections, smooth dolly zoom, luxury commercial finish, no text."
  ],
  "scenes": [
    {{"scene_style": "3d_orbit_reveal", "feature_idx": 0}},
    {{"scene_style": "function_showcase", "feature_idx": 0}},
    {{"scene_style": "macro_3d_sweep", "feature_idx": 1}},
    {{"scene_style": "function_showcase", "feature_idx": 2}},
    {{"scene_style": "luxury_cta_finale", "feature_idx": 2}}
  ]
}}
"""
    for model_name in ("gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite"):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[system_prompt, img],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            txt = (response.text or "").strip()
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(txt)
            if "voiceover_script" in data and "cgi_prompts" in data:
                return data
        except Exception as e:
            print(f"Gemini failed on {model_name}: {repr(e)}", flush=True)
            continue
    return _fallback_ad_plan(image_path, user_hint)


async def create_english_voiceover(text: str, out_mp3: Path, voice_accent: str):
    voice_map = {
        "us_female": "en-US-AriaNeural",
        "uk_female": "en-GB-SoniaNeural",
        "in_female": "en-IN-NeerjaNeural",
        "us_male": "en-US-GuyNeural",
    }
    voice = voice_map.get(voice_accent, "en-US-AriaNeural")
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+5%")
        await communicate.save(str(out_mp3))
    except Exception as e:
        print(f"Edge-TTS failed ({e}), using gTTS English fallback...", flush=True)
        gTTS(text=text, lang="en", tld="com").save(str(out_mp3))


def _prep_studio_reference_png(clean_product: Image.Image, aspect_ratio: str) -> bytes:
    """Places the isolated HD product onto a clean dark luxury studio canvas for Image-to-Video AI."""
    W, H = (720, 1280) if aspect_ratio == "9:16" else (1280, 720)
    canvas = Image.new("RGB", (W, H), (12, 16, 26))
    prod = clean_product.copy()
    prod.thumbnail((int(W * 0.62), int(H * 0.56)), Image.Resampling.LANCZOS)
    px = (W - prod.width) // 2
    py = (H - prod.height) // 2
    canvas.paste(prod, (px, py), prod)
    out = io.BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def _generate_fal_cgi_clips(
    clean_product: Image.Image,
    plan: dict,
    aspect_ratio: str,
    job_dir: Path,
    job_id: str,
) -> list[Path] | None:
    """Calls Fal.ai Image-to-Video (Wan 2.2 / Kling) to generate TRUE 3D CGI cinematic clips."""
    if not FAL_KEY:
        return None

    ref_png = _prep_studio_reference_png(clean_product, aspect_ratio)
    data_uri = "data:image/png;base64," + base64.b64encode(ref_png).decode("utf-8")
    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    prompts = plan.get("cgi_prompts", [])[:3]
    clips = []

    with httpx.Client(timeout=240.0) as client:
        for idx, prompt in enumerate(prompts):
            JOBS[job_id]["step"] = f"Generating TRUE 3D CGI Scene {idx + 1}/{len(prompts)} via AI Video..."
            JOBS[job_id]["progress"] = 35 + idx * 18
            try:
                submit_res = client.post(
                    f"https://queue.fal.run/{FAL_MODEL}",
                    headers=headers,
                    json={
                        "image_url": data_uri,
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,
                    },
                )
                submit_res.raise_for_status()
                q = submit_res.json()
                status_url = q["status_url"]
                response_url = q["response_url"]

                for _ in range(90):
                    time.sleep(3)
                    st = client.get(status_url, headers=headers).json()
                    if st.get("status") == "COMPLETED":
                        break
                    if st.get("status") in ("FAILED", "CANCELLED"):
                        raise RuntimeError(f"Fal job failed: {st}")

                res_data = client.get(response_url, headers=headers).json()
                video_url = res_data.get("video", {}).get("url")
                if not video_url:
                    raise RuntimeError(f"Missing video URL in Fal response: {res_data}")

                clip_path = job_dir / f"cgi_{idx}.mp4"
                clip_path.write_bytes(client.get(video_url).content)
                clips.append(clip_path)
            except Exception as e:
                print(f"Fal CGI generation failed on scene {idx}: {repr(e)}", flush=True)
                return None
    return clips if clips else None


def _generate_gemini_omni_clip(
    clean_product: Image.Image,
    plan: dict,
    aspect_ratio: str,
    out_mp4: Path,
) -> bool:
    """Calls Google Gemini Omni 1.2 Flash (requires Paid Tier billing) for TRUE 3D CGI video."""
    if not (USE_GEMINI_OMNI and GEMINI_API_KEY):
        return False

    try:
        ref_png = _prep_studio_reference_png(clean_product, aspect_ratio)
        img_b64 = base64.b64encode(ref_png).decode("utf-8")
        client = genai.Client(api_key=GEMINI_API_KEY)
        full_prompt = " ".join(plan.get("cgi_prompts", []))[:800]
        interaction = client.interactions.create(
            model="gemini-omni-1.2-flash",
            input=[
                {"type": "image", "data": img_b64, "mime_type": "image/png"},
                {"type": "text", "text": full_prompt},
            ],
            response_format={
                "type": "video",
                "aspect_ratio": aspect_ratio,
                "resolution": "720p",
                "duration": "12s",
            },
        )
        video_bytes = None
        if getattr(interaction, "output_video", None) and getattr(interaction.output_video, "data", None):
            video_bytes = base64.b64decode(interaction.output_video.data)
        elif getattr(interaction, "steps", None):
            for step in interaction.steps:
                for item in getattr(step, "content", []) or []:
                    if getattr(item, "type", None) == "video" and getattr(item, "data", None):
                        video_bytes = base64.b64decode(item.data)
                        break
        if video_bytes:
            out_mp4.write_bytes(video_bytes)
            return True
    except Exception as e:
        print(f"Gemini Omni 1.2 Flash failed: {repr(e)}", flush=True)
    return False


def _generate_modal_clip(clean_product: Image.Image, plan: dict, aspect_ratio: str, out_mp4: Path) -> bool:
    if not MODAL_VIDEO_URL:
        return False
    try:
        ref_png = _prep_studio_reference_png(clean_product, aspect_ratio)
        img_b64 = base64.b64encode(ref_png).decode("utf-8")
        payload = {
            "image_base64": img_b64,
            "prompt": " ".join(plan.get("cgi_prompts", []))[:800],
            "aspect_ratio": aspect_ratio,
            "duration": 10,
        }
        with httpx.Client(timeout=240.0) as client_http:
            resp = client_http.post(MODAL_VIDEO_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "video_base64" in data:
                out_mp4.write_bytes(base64.b64decode(data["video_base64"]))
                return True
            if "video_url" in data:
                out_mp4.write_bytes(client_http.get(data["video_url"]).content)
                return True
    except Exception as e:
        print(f"MODAL_VIDEO_URL failed: {repr(e)}", flush=True)
    return False


def _find_perspective_coeffs(dst_pts, src_pts):
    matrix = []
    for (dx, dy), (sx, sy) in zip(dst_pts, src_pts):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for pt in src_pts for c in pt], dtype=np.float64)
    return np.linalg.solve(A, B).tolist()


def _render_3d_tilted_hd(product: Image.Image, tilt_deg: float, scale: float = 1.0) -> Image.Image:
    pw, ph = product.size
    sw, sh = max(40, int(pw * scale)), max(40, int(ph * scale))
    img = product.resize((sw, sh), Image.Resampling.LANCZOS)
    img = ImageOps.expand(img, border=6, fill=(0, 0, 0, 0))
    iw, ih = img.size

    rad = math.radians(max(-30.0, min(30.0, tilt_deg)))
    pad_x = int(iw * 0.14)
    pad_y = int(ih * 0.14)
    cw, ch = iw + pad_x * 2, ih + pad_y * 2

    inset_x = iw * (1.0 - math.cos(rad)) * 0.40
    shift_y = ih * math.sin(rad) * 0.20

    src_pts = [(0, 0), (iw, 0), (iw, ih), (0, ih)]
    dst_pts = [
        (pad_x + inset_x, pad_y - shift_y),
        (pad_x + iw - inset_x, pad_y + shift_y),
        (pad_x + iw - inset_x, pad_y + ih - shift_y),
        (pad_x + inset_x, pad_y + ih + shift_y),
    ]
    coeffs = _find_perspective_coeffs(dst_pts, src_pts)
    return img.transform((cw, ch), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def _add_studio_reflection(canvas: Image.Image, tilted_prod: Image.Image, x: int, y: int) -> None:
    pw, ph = tilted_prod.size
    shadow = Image.new("RGBA", (pw, 52), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([int(pw * 0.14), 10, int(pw * 0.86), 42], fill=(0, 0, 0, 165))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    canvas.alpha_composite(shadow, (x, y + ph - int(ph * 0.15)))

    refl_h = int(ph * 0.36)
    refl = ImageOps.flip(tilted_prod).crop((0, 0, pw, refl_h))
    _, _, _, a = refl.split()
    a_np = np.array(a, dtype=np.float32)
    grad = np.linspace(0.30, 0.0, refl_h)[:, None]
    refl.putalpha(Image.fromarray(np.clip(a_np * grad, 0, 255).astype(np.uint8), "L"))
    canvas.alpha_composite(refl.filter(ImageFilter.GaussianBlur(1.5)), (x, y + ph - int(ph * 0.13)))


def _make_studio_bg(w: int, h: int, c_edge: tuple, c_center: tuple, floor_y: float = 0.73) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((x - w * 0.5) / (w * 0.62)) ** 2 + ((y - h * 0.42) / (h * 0.50)) ** 2)
    t = np.clip(1.0 - dist, 0.0, 1.0) ** 1.7
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(3):
        rgb[:, :, ch] = np.clip(c_edge[ch] + t * (c_center[ch] - c_edge[ch]), 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    fy = int(h * floor_y)
    ImageDraw.Draw(img).line([(0, fy), (w, fy)], fill=(148, 163, 184, 55), width=2)
    return img


def generate_hd_studio_fallback_clip(
    product: Image.Image,
    scene: dict,
    plan: dict,
    aspect_ratio: str,
    out_mp4: Path,
):
    """High-definition 720p Studio Preview fallback when no paid 3D Video API key is configured."""
    W, H = (720, 1280) if aspect_ratio == "9:16" else (1280, 720)
    fps = 24
    total_frames = 54

    scene_style = scene.get("scene_style", "3d_orbit_reveal")
    features = plan.get("key_functions", [{"title": "PRO QUALITY", "detail": "Crafted for performance"}])
    feat = features[int(scene.get("feature_idx", 0)) % max(1, len(features))]

    hero = product.copy()
    hero.thumbnail((int(W * 0.62), int(H * 0.48)), Image.Resampling.LANCZOS)
    rng = random.Random(hash(scene_style) & 0xFFFF)

    font_title = _load_font(28)
    font_sub = _load_font(20)
    font_big = _load_font(56, serif=True)
    font_cta = _load_font(30)

    if scene_style == "macro_3d_sweep":
        bg = _make_studio_bg(W, H, (12, 10, 22), (38, 28, 66), floor_y=0.78)
        macro = product.copy()
        macro.thumbnail((int(W * 1.10), int(H * 0.90)), Image.Resampling.LANCZOS)
        mw, mh = macro.size
    elif scene_style == "luxury_cta_finale":
        bg = _make_studio_bg(W, H, (6, 9, 16), (26, 38, 62), floor_y=0.73)
        sparks = [(rng.uniform(0, 2 * math.pi), rng.uniform(W * 0.22, W * 0.42), rng.randint(4, 9), rng.uniform(0.8, 1.8)) for _ in range(28)]
    else:
        bg = _make_studio_bg(W, H, (5, 8, 15), (22, 46, 82), floor_y=0.73)

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-threads", "2", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for f in range(total_frames):
        p = f / max(1, total_frames - 1)
        ease = 1.0 - (1.0 - p) ** 3
        frame = bg.copy()

        sweep = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sx = int(-W * 0.4 + p * W * 1.8)
        ImageDraw.Draw(sweep).polygon(
            [(sx, 0), (sx + 150, 0), (sx - H // 2 + 150, H), (sx - H // 2, H)],
            fill=(186, 230, 253, 24),
        )
        frame = Image.alpha_composite(frame, sweep.filter(ImageFilter.GaussianBlur(22)))

        if scene_style == "macro_3d_sweep":
            ox = int((W - mw) * (0.15 + 0.7 * p))
            oy = int((H - mh) * (0.25 + 0.35 * math.sin(p * math.pi)))
            frame.alpha_composite(macro, (ox, oy))
            d = ImageDraw.Draw(frame)
            bx0, by0, bx1, by1 = int(W * 0.08), int(H * 0.78), int(W * 0.92), int(H * 0.90)
            d.rounded_rectangle([bx0, by0, bx1, by1], radius=20, fill=(10, 15, 28, 210), outline=(56, 189, 248, 200), width=2)
            d.text((bx0 + 28, by0 + 20), f"✦ {feat['title'].upper()}", font=font_title, fill=(56, 189, 248, 255))
            d.text((bx0 + 28, by0 + 62), feat["detail"], font=font_sub, fill=(226, 232, 240, 240))

        elif scene_style == "function_showcase":
            tilt = math.sin(p * math.pi * 1.5 - 0.5) * 20.0
            bob = int(math.sin(p * math.pi * 2) * 12)
            tilted = _render_3d_tilted_hd(hero, tilt_deg=tilt, scale=0.95 + 0.05 * ease)
            px = W // 2 - tilted.width // 2
            py = int(H * 0.46 - tilted.height // 2) + bob
            _add_studio_reflection(frame, tilted, px, py)
            frame.alpha_composite(tilted, (px, py))

            card_alpha = int(min(1.0, p / 0.25) * 225)
            hud = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            hd = ImageDraw.Draw(hud)
            card_y = int(H * 0.11)
            card_x0 = int(W * 0.08)
            card_x1 = int(W * 0.92) if aspect_ratio == "9:16" else int(W * 0.50)
            hd.rounded_rectangle(
                [card_x0, card_y, card_x1, card_y + 116],
                radius=20,
                fill=(12, 19, 34, card_alpha),
                outline=(56, 189, 248, card_alpha),
                width=2,
            )
            hd.text((card_x0 + 26, card_y + 20), f"✦ {feat['title'].upper()}", font=font_title, fill=(56, 189, 248, 255))
            hd.text((card_x0 + 26, card_y + 62), feat["detail"], font=font_sub, fill=(241, 245, 249, 240))
            frame = Image.alpha_composite(frame, hud)

        elif scene_style == "luxury_cta_finale":
            tilt = (1.0 - ease) * -16.0
            tilted = _render_3d_tilted_hd(hero, tilt_deg=tilt, scale=0.94 + 0.10 * ease)
            px = W // 2 - tilted.width // 2
            py = int(H * 0.44 - tilted.height // 2)
            _add_studio_reflection(frame, tilted, px, py)
            frame.alpha_composite(tilted, (px, py))

            sd = ImageDraw.Draw(frame)
            for ang, rad, sz, spd in sparks:
                a = ang + p * spd * 3.0
                sx_p = int(W // 2 + math.cos(a) * rad)
                sy_p = int(H * 0.42 + math.sin(a) * rad * 0.6)
                sd.polygon([(sx_p, sy_p - sz), (sx_p + sz, sy_p), (sx_p, sy_p + sz), (sx_p - sz, sy_p)], fill=(226, 232, 240, 210))

            btn_w, btn_h = int(W * 0.58), 72
            bx0, by0 = W // 2 - btn_w // 2, int(H * 0.80)
            sd.rounded_rectangle([bx0, by0, bx0 + btn_w, by0 + btn_h], radius=36, fill=(14, 165, 233, 245), outline=(255, 255, 255, 210), width=2)
            cta_txt = "ORDER NOW →"
            tw = sd.textlength(cta_txt, font=font_cta)
            sd.text((W // 2 - tw // 2, by0 + 18), cta_txt, font=font_cta, fill=(255, 255, 255, 255))

        else:
            tilt = -24.0 + 36.0 * ease
            scale = 0.84 + 0.20 * ease
            tilted = _render_3d_tilted_hd(hero, tilt_deg=tilt, scale=scale)
            px = W // 2 - tilted.width // 2
            py = int(H * 0.47 - tilted.height // 2)
            _add_studio_reflection(frame, tilted, px, py)
            frame.alpha_composite(tilted, (px, py))

            d = ImageDraw.Draw(frame)
            brand = plan.get("hero_brand", "LUXURY").upper()[:18]
            bw = d.textlength(brand, font=font_big)
            d.text((W // 2 - bw // 2, int(H * 0.11)), brand, font=font_big, fill=(255, 255, 255, int(235 * ease)))

        proc.stdin.write(frame.convert("RGB").tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode(errors="ignore")
    if proc.wait() != 0:
        raise RuntimeError("FFmpeg clip render failed: " + err[-250:])
    del bg
    gc.collect()


def finalize_audio_video(job_dir: Path, raw_clips: list[Path], voice_mp3: Path, aspect_ratio: str) -> Path:
    w, h = (720, 1280) if aspect_ratio == "9:16" else (1280, 720)
    styled_clips = []

    for idx, clip in enumerate(raw_clips):
        styled = job_dir / f"styled_{idx}.mp4"
        dur = get_duration(clip)
        out_st = max(0.5, dur - 0.25)
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=24,"
            f"eq=contrast=1.06:saturation=1.10,"
            f"fade=t=in:st=0:d=0.18,fade=t=out:st={out_st:.2f}:d=0.20"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(clip),
                "-vf", vf, "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-threads", "2", str(styled),
            ],
            check=True,
        )
        clip.unlink(missing_ok=True)
        styled_clips.append(styled)

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
    target_dur = max(vid_dur, aud_dur + 0.6)
    pad_sec = round(max(0.0, target_dur - vid_dur), 2)

    final_mp4 = job_dir / "final_ad.mp4"
    audio_filter = (
        f"aevalsrc='0.06*sin(2*PI*55*t)*exp(-4*mod(t,0.5))+0.02*sin(2*PI*165*t)*exp(-8*mod(t,0.25))':s=44100:d={target_dur:.2f}[bg];"
        f"[1:a]aresample=44100,volume=1.50[vo];"
        f"[vo][bg]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
    )
    video_filter = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_sec:.2f}[vfinal]"

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(merged_video), "-i", str(voice_mp3),
            "-filter_complex", f"{video_filter};{audio_filter}",
            "-map", "[vfinal]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-t", f"{target_dur:.2f}",
            "-movflags", "+faststart", "-threads", "2", str(final_mp4),
        ],
        check=True,
    )
    merged_video.unlink(missing_ok=True)
    gc.collect()
    return final_mp4


def run_pipeline(job_id: str, image_path: Path, hint: str, voice_accent: str, aspect_ratio: str):
    job_dir = image_path.parent
    try:
        JOBS[job_id]["step"] = "Analyzing product & writing 3D CGI commercial script..."
        JOBS[job_id]["progress"] = 10
        plan = analyze_product_with_gemini(image_path, hint)
        JOBS[job_id]["script"] = plan["voiceover_script"]

        JOBS[job_id]["step"] = "Isolating product at HD resolution..."
        JOBS[job_id]["progress"] = 20
        product_img = extract_product_hd(image_path.read_bytes())

        JOBS[job_id]["step"] = "Recording English commercial voiceover..."
        JOBS[job_id]["progress"] = 28
        voice_mp3 = job_dir / "voice.mp3"
        asyncio.run(create_english_voiceover(plan["voiceover_script"], voice_mp3, voice_accent))

        raw_clips = None
        mode_used = "True 3D CGI AI Video"

        # 1) Try Fal.ai Image-to-Video (Wan 2.2 / Kling)
        if FAL_KEY:
            raw_clips = _generate_fal_cgi_clips(product_img, plan, aspect_ratio, job_dir, job_id)

        # 2) Try Google Gemini Omni 1.2 Flash (Paid Tier)
        if not raw_clips and USE_GEMINI_OMNI:
            JOBS[job_id]["step"] = "Generating 3D CGI commercial via Gemini Omni..."
            JOBS[job_id]["progress"] = 50
            omni_mp4 = job_dir / "omni.mp4"
            if _generate_gemini_omni_clip(product_img, plan, aspect_ratio, omni_mp4):
                raw_clips = [omni_mp4]

        # 3) Try Modal GPU Backend
        if not raw_clips and MODAL_VIDEO_URL:
            JOBS[job_id]["step"] = "Generating 3D CGI commercial via Modal GPU..."
            JOBS[job_id]["progress"] = 50
            modal_mp4 = job_dir / "modal.mp4"
            if _generate_modal_clip(product_img, plan, aspect_ratio, modal_mp4):
                raw_clips = [modal_mp4]

        # 4) Fallback: High-Definition 720p Studio Preview
        if not raw_clips:
            mode_used = "HD Studio Preview (Set FAL_KEY or USE_GEMINI_OMNI=true for true 3D CGI AI motion)"
            raw_clips = []
            for idx, scene in enumerate(plan["scenes"]):
                JOBS[job_id]["step"] = f"Rendering HD Studio Scene {idx + 1}/5..."
                JOBS[job_id]["progress"] = 35 + (idx * 10)
                clip_path = job_dir / f"raw_{idx}.mp4"
                generate_hd_studio_fallback_clip(product_img, scene, plan, aspect_ratio, clip_path)
                raw_clips.append(clip_path)

        del product_img
        gc.collect()

        JOBS[job_id]["step"] = "Mixing English voiceover & exporting HD MP4..."
        JOBS[job_id]["progress"] = 90
        finalize_audio_video(job_dir, raw_clips, voice_mp3, aspect_ratio)

        JOBS[job_id].update({
            "status": "done",
            "progress": 100,
            "step": f"Done! ({mode_used})",
            "video_url": f"/download/{job_id}",
        })
    except Exception as exc:
        JOBS[job_id].update({"status": "error", "step": f"Error: {str(exc)}"})


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "rembg": HAS_REMBG,
        "rembg_model": REMBG_MODEL,
        "fal_configured": bool(FAL_KEY),
        "gemini_omni": USE_GEMINI_OMNI,
        "modal_configured": bool(MODAL_VIDEO_URL),
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

    JOBS[job_id] = {"status": "processing", "progress": 5, "step": "Starting...", "script": ""}
    threading.Thread(
        target=run_pipeline,
        args=(job_id, img_path, hint, voice_accent, aspect_ratio),
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
    return FileResponse(video_path, media_type="video/mp4", filename=f"3d_cgi_ad_{job_id}.mp4")


@app.get("/", response_class=HTMLResponse)
def home():
    cgi_ready = bool(FAL_KEY or USE_GEMINI_OMNI or MODAL_VIDEO_URL)
    badge = (
        '<span style="color:#10b981;font-weight:700;">● True 3D CGI AI Engine Active</span>'
        if cgi_ready
        else '<span style="color:#f59e0b;font-weight:700;">● HD Studio Preview Mode (Add FAL_KEY or USE_GEMINI_OMNI=true for True 3D CGI AI Motion)</span>'
    )
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI 3D CGI Product Ad Studio</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #070b14; color: #f8fafc; margin: 0; padding: 24px; }}
    .card {{ max-width: 620px; margin: 0 auto; background: #111827; padding: 28px; border-radius: 16px; border: 1px solid #1f2937; }}
    h1 {{ margin-top: 0; font-size: 1.55rem; color: #38bdf8; }}
    label {{ display: block; margin-top: 16px; font-weight: 600; font-size: 0.92rem; color: #cbd5e1; }}
    input, textarea, select {{ width: 100%; margin-top: 6px; padding: 10px 12px; border-radius: 10px; border: 1px solid #334155; background: #0b1120; color: #fff; box-sizing: border-box; }}
    button {{ width: 100%; margin-top: 22px; padding: 14px; border: none; border-radius: 10px; background: linear-gradient(90deg, #0ea5e9, #6366f1); color: #fff; font-weight: 700; font-size: 1rem; cursor: pointer; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .progress-box {{ display: none; margin-top: 20px; background: #0b1120; padding: 14px; border-radius: 10px; }}
    .bar-bg {{ width: 100%; height: 10px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-top: 8px; }}
    .bar-fill {{ height: 100%; width: 0%; background: #38bdf8; transition: width 0.4s; }}
    video {{ width: 100%; max-height: 540px; border-radius: 12px; margin-top: 16px; background: #000; }}
    .dl-btn {{ display: inline-block; margin-top: 12px; padding: 10px 18px; background: #10b981; color: #fff; text-decoration: none; border-radius: 8px; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>✨ AI 3D CGI Product Ad Studio</h1>
    <p style="font-size:0.85rem;margin-top:-6px;">{badge}</p>
    <p style="color:#94a3b8;font-size:0.9rem;">Upload any product photo. AI extracts only the product in HD, turns it into a cinematic 3D CGI commercial, and adds a sleek English voiceover.</p>
    <form id="adForm">
      <label>1. Upload Product Photo</label>
      <input type="file" name="image" accept="image/*" required />

      <label>2. Product Features / CGI Direction (Optional)</label>
      <textarea name="hint" rows="2" placeholder="e.g. Waterproof smartwatch, show water splashes and glowing AMOLED screen"></textarea>

      <label>3. English Voiceover Style</label>
      <select name="voice_accent">
        <option value="us_female" selected>US Luxury Female (Aria)</option>
        <option value="uk_female">British Luxury Female (Sonia)</option>
        <option value="in_female">Indian English Female (Neerja)</option>
        <option value="us_male">US Deep Commercial Male (Guy)</option>
      </select>

      <label>4. Aspect Ratio</label>
      <select name="aspect_ratio">
        <option value="9:16" selected>Instagram Reels / Shorts (9:16)</option>
        <option value="16:9">YouTube / Amazon / Landscape (16:9)</option>
      </select>

      <button type="submit" id="submitBtn">🎬 Generate 3D CGI Ad</button>
    </form>

    <div class="progress-box" id="progressBox">
      <div id="statusText">Starting...</div>
      <div class="bar-bg"><div class="bar-fill" id="barFill"></div></div>
      <p id="scriptBox" style="font-size:0.85rem;color:#94a3b8;margin-top:10px;"></p>
    </div>

    <div id="resultBox" style="display:none;">
      <video id="adVideo" controls></video>
      <a id="dlLink" class="dl-btn" download>⬇️ Download HD MP4 Ad</a>
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
      statusText.innerText = "Uploading product image...";
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
