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

try:
    from rembg import new_session, remove as rembg_remove
    REMBG_SESSION = new_session("u2netp")
    HAS_REMBG = True
except Exception:
    REMBG_SESSION = None
    HAS_REMBG = False

app = FastAPI(title="AI 3D Cinematic Product Ad Studio")

WORK_DIR = Path("/tmp/ad_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
MODAL_VIDEO_URL = os.environ.get("MODAL_VIDEO_URL", "").strip().strip('"').strip("'")
if MODAL_VIDEO_URL and not MODAL_VIDEO_URL.startswith(("http://", "https://")):
    MODAL_VIDEO_URL = "https://" + MODAL_VIDEO_URL

# Set USE_GEMINI_OMNI=true if your GEMINI_API_KEY has Paid Tier 1 enabled
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


def extract_product_clean(image_bytes: bytes) -> Image.Image:
    """Extracts ONLY the main product from any background using AI segmentation (rembg)."""
    if HAS_REMBG:
        try:
            cutout_bytes = rembg_remove(image_bytes, session=REMBG_SESSION)
            img = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            img.thumbnail((640, 640), Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print(f"rembg failed, falling back to edge segmentation: {e}", flush=True)

    # Fallback if rembg is unavailable
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img.thumbnail((640, 640), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape

    border = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]], axis=0)
    bg_rgb = np.median(border, axis=0)
    dist = np.linalg.norm(arr[:, :, :3] - bg_rgb, axis=2)

    if float(np.mean(np.std(border, axis=0))) < 48.0:
        alpha = np.clip((dist - 20.0) / 30.0, 0.0, 1.0) * 255.0
        if np.mean(alpha > 128) > 0.10:
            mask = Image.fromarray(alpha.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1))
            img.putalpha(mask)
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            return img

    # Center elliptical vignette fallback for cluttered backgrounds without rembg
    yy, xx = np.mgrid[0:h, 0:w]
    ell = (((xx - w * 0.5) / (w * 0.44)) ** 2 + ((yy - h * 0.5) / (h * 0.44)) ** 2)
    alpha = np.clip((1.15 - ell) / 0.35, 0.0, 1.0) * 255.0
    img.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    return img


def _fallback_ad_plan(image_path: Path, user_hint: str, language: str) -> dict:
    raw = image_path.stem.replace("_", " ").replace("-", " ").strip()
    brand = "".join(c for c in (raw.split()[0] if raw else "LUXURY") if c.isalnum()).upper() or "LUXURY"
    hint_clean = (user_hint or "next-gen luxury engineering").strip()

    if language == "hinglish":
        script = (
            f"Ordinary ko bhool jaaiye, miliye naye {brand} se! "
            f"{hint_clean} ke saath iska smart design aur powerhouse performance har moment ko effortless banata hai. "
            f"Limited stock available — apna {brand} abhi order karein!"
        )
    else:
        script = (
            f"Stop settling for ordinary — meet the all-new {brand}. "
            f"Engineered with {hint_clean}, precision craftsmanship, and effortless everyday performance. "
            f"Upgrade your lifestyle today and claim yours now."
        )

    return {
        "hero_brand": brand[:18],
        "product_name": brand,
        "tagline": "ENGINEERED TO PERFORM",
        "key_functions": [
            {"title": "PRECISION DESIGN", "detail": "Ultra-durable luxury finish"},
            {"title": "SMART PERFORMANCE", "detail": "Built for effortless daily use"},
            {"title": "PRO GRADE QUALITY", "detail": "Trusted by top creators"},
        ],
        "voiceover_script": script,
        "omni_video_prompt": (
            f"A high-budget 3D CGI luxury product commercial for {brand}. "
            f"Isolate the exact product in a sleek dark studio with volumetric rim lighting, 360-degree 3D camera orbit, "
            f"macro close-ups demonstrating its real-world functions in action, floating specular reflections, and a dramatic hero packshot."
        ),
        "scenes": [
            {"scene_style": "3d_orbit_reveal", "feature_idx": 0, "video_prompt": f"3D orbit reveal for {brand}"},
            {"scene_style": "function_showcase", "feature_idx": 0, "video_prompt": f"Function showcase 1 for {brand}"},
            {"scene_style": "macro_3d_sweep", "feature_idx": 1, "video_prompt": f"Macro 3D sweep for {brand}"},
            {"scene_style": "function_showcase", "feature_idx": 2, "video_prompt": f"Function showcase 2 for {brand}"},
            {"scene_style": "luxury_cta_finale", "feature_idx": 2, "video_prompt": f"Luxury CTA packshot for {brand}"},
        ],
    }


def analyze_product_with_gemini(image_path: Path, user_hint: str, language: str) -> dict:
    if not GEMINI_API_KEY:
        return _fallback_ad_plan(image_path, user_hint, language)

    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)

    lang_instruction = (
        "Write voiceover_script in crisp, high-converting luxury Hinglish (Hindi in Roman script mixed with English, persuasive Indian female commercial voice)."
        if language == "hinglish"
        else "Write voiceover_script in sleek, persuasive luxury English."
    )

    system_prompt = f"""
You are an elite 3D CGI commercial director (like Apple / Dior / Sony product launch ads).
Look closely at the exact product inside the uploaded image and the user's hint: "{user_hint or 'Create a 3D cinematic commercial showing product functions so customers buy it'}".

Identify:
1. Exact product type and brand/name.
2. Its 3 REAL practical functions/features that make customers buy it (e.g., for smartwatch: "HEART & SPO2 MONITOR", "IP68 WATERPROOF", "7-DAY BATTERY"; for perfume: "24H LONG-LASTING", "MICRO-MIST ATOMIZER", "FRENCH OUD NOTES"; for blender/gadget: "HIGH-TORQUE MOTOR", "ONE-TOUCH BLEND", "USB-C FAST CHARGE").
3. A vivid 3D CGI commercial prompt (`omni_video_prompt`) showing the product rotating in 3D space and actively demonstrating its functions in a cinematic studio (no hands/clutter from original background).

{lang_instruction}

Return ONLY valid JSON:
{{
  "hero_brand": "1-2 word bold brand/product title",
  "product_name": "2-4 word product name",
  "tagline": "2-4 word punchy sales tagline",
  "key_functions": [
    {{"title": "2-3 WORD FEATURE", "detail": "4-6 word customer benefit"}},
    {{"title": "2-3 WORD FEATURE", "detail": "4-6 word customer benefit"}},
    {{"title": "2-3 WORD FEATURE", "detail": "4-6 word customer benefit"}}
  ],
  "voiceover_script": "30 to 38 words total (approx 12 seconds spoken). Hook + product functions + urgent buy CTA.",
  "omni_video_prompt": "Detailed 60-word 3D CGI commercial direction describing camera orbit, studio lighting, and the exact product performing its real functions in slow-motion action.",
  "scenes": [
    {{"scene_style": "3d_orbit_reveal", "feature_idx": 0, "video_prompt": "3D studio reveal"}},
    {{"scene_style": "function_showcase", "feature_idx": 0, "video_prompt": "Show function 1"}},
    {{"scene_style": "macro_3d_sweep", "feature_idx": 1, "video_prompt": "Macro 3D detail + function 2"}},
    {{"scene_style": "function_showcase", "feature_idx": 2, "video_prompt": "Show function 3"}},
    {{"scene_style": "luxury_cta_finale", "feature_idx": 2, "video_prompt": "Hero packshot + Shop Now CTA"}}
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
            if "key_functions" in data and len(data["key_functions"]) >= 3:
                return data
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


def try_generate_real_ai_video(
    clean_product: Image.Image,
    plan: dict,
    aspect_ratio: str,
    out_mp4: Path,
) -> bool:
    """
    Attempts TRUE 3D CGI Image-to-Video generation using:
    1) Modal GPU endpoint (MODAL_VIDEO_URL) if configured
    2) Google Gemini Omni 1.2 Flash (USE_GEMINI_OMNI=true with paid tier)
    Returns True if a real AI video was generated, else False (falls back to 3D Studio Engine).
    """
    buf = io.BytesIO()
    # Place isolated product on clean studio dark slate for image-to-video models
    studio_bg = Image.new("RGB", (clean_product.width + 120, clean_product.height + 120), (14, 18, 28))
    studio_bg.paste(clean_product, (60, 60), clean_product)
    studio_bg.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 1) Modal GPU Backend (Wan2.1 / LTX / Kling etc.)
    if MODAL_VIDEO_URL:
        try:
            payload = {
                "image_base64": img_b64,
                "prompt": plan.get("omni_video_prompt", "3D cinematic luxury product commercial"),
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
                elif "video_url" in data:
                    v_resp = client_http.get(data["video_url"])
                    v_resp.raise_for_status()
                    out_mp4.write_bytes(v_resp.content)
                    return True
        except Exception as e:
            print(f"MODAL_VIDEO_URL failed, falling back: {repr(e)}", flush=True)

    # 2) Google Gemini Omni 1.2 Flash (Interactions API)
    if USE_GEMINI_OMNI and GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = (
                f"{plan.get('omni_video_prompt', '')} "
                f"Focus strictly on the product in the reference image placed in a luxury 3D CGI commercial environment. "
                f"Smooth 3D camera orbit, dramatic studio lighting, show the product's features in motion, no text overlays."
            )
            interaction = client.interactions.create(
                model="gemini-omni-1.2-flash",
                input=[
                    {"type": "image", "data": img_b64, "mime_type": "image/png"},
                    {"type": "text", "text": prompt},
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
            print(f"Gemini Omni 1.2 Flash failed (paid tier required), falling back: {repr(e)}", flush=True)

    return False


def _find_perspective_coeffs(dst_pts, src_pts):
    matrix = []
    for (dx, dy), (sx, sy) in zip(dst_pts, src_pts):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for pt in src_pts for c in pt], dtype=np.float64)
    return np.linalg.solve(A, B).tolist()


def _render_3d_tilted_product(product: Image.Image, tilt_deg: float, scale: float = 1.0) -> Image.Image:
    """Applies true projective 3D Y-axis perspective tilt + mirror studio reflection."""
    pw, ph = product.size
    sw, sh = max(20, int(pw * scale)), max(20, int(ph * scale))
    img = product.resize((sw, sh), Image.Resampling.LANCZOS)
    img = ImageOps.expand(img, border=4, fill=(0, 0, 0, 0))
    iw, ih = img.size

    rad = math.radians(max(-35.0, min(35.0, tilt_deg)))
    pad_x = int(iw * 0.16)
    pad_y = int(ih * 0.18)
    cw, ch = iw + pad_x * 2, ih + pad_y * 2

    inset_x = iw * (1.0 - math.cos(rad)) * 0.42
    shift_y = ih * math.sin(rad) * 0.22

    src_pts = [(0, 0), (iw, 0), (iw, ih), (0, ih)]
    dst_pts = [
        (pad_x + inset_x, pad_y - shift_y),
        (pad_x + iw - inset_x, pad_y + shift_y),
        (pad_x + iw - inset_x, pad_y + ih - shift_y),
        (pad_x + inset_x, pad_y + ih + shift_y),
    ]
    coeffs = _find_perspective_coeffs(dst_pts, src_pts)
    tilted = img.transform((cw, ch), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    return tilted


def _add_studio_reflection(canvas: Image.Image, tilted_prod: Image.Image, x: int, y: int) -> None:
    """Draws soft ground shadow + glossy mirror floor reflection directly beneath the 3D product."""
    pw, ph = tilted_prod.size
    # Soft contact shadow
    shadow = Image.new("RGBA", (pw, 40), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([int(pw * 0.15), 8, int(pw * 0.85), 32], fill=(0, 0, 0, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(shadow, (x, y + ph - int(ph * 0.16)))

    # Glossy mirror reflection
    refl_h = int(ph * 0.40)
    refl = ImageOps.flip(tilted_prod).crop((0, 0, pw, refl_h))
    r, g, b, a = refl.split()
    a_np = np.array(a, dtype=np.float32)
    grad = np.linspace(0.32, 0.0, refl_h)[:, None]
    a_np = np.clip(a_np * grad, 0, 255).astype(np.uint8)
    refl.putalpha(Image.fromarray(a_np, "L"))
    refl = refl.filter(ImageFilter.GaussianBlur(1.5))
    canvas.alpha_composite(refl, (x, y + ph - int(ph * 0.14)))


def _make_studio_bg(w: int, h: int, c_edge: tuple, c_center: tuple, floor_y: float = 0.72) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((x - w * 0.5) / (w * 0.62)) ** 2 + ((y - h * 0.42) / (h * 0.50)) ** 2)
    t = np.clip(1.0 - dist, 0.0, 1.0) ** 1.7
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(3):
        rgb[:, :, ch] = np.clip(c_edge[ch] + t * (c_center[ch] - c_edge[ch]), 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    # Horizon studio floor glow
    d = ImageDraw.Draw(img)
    fy = int(h * floor_y)
    d.line([(0, fy), (w, fy)], fill=(148, 163, 184, 55), width=2)
    return img


def generate_3d_studio_clip(
    product: Image.Image,
    scene: dict,
    plan: dict,
    aspect_ratio: str,
    out_mp4: Path,
):
    """Renders 3D-perspective animated studio scenes + interactive function callout HUD."""
    W, H = (540, 960) if aspect_ratio == "9:16" else (960, 540)
    fps = 24
    total_frames = 56

    scene_style = scene.get("scene_style", "3d_orbit_reveal")
    feat_idx = int(scene.get("feature_idx", 0)) % max(1, len(plan.get("key_functions", [])))
    features = plan.get("key_functions", [{"title": "PRO QUALITY", "detail": "Crafted for performance"}])
    feat = features[feat_idx]

    hero = product.copy()
    hero.thumbnail((int(W * 0.58), int(H * 0.44)), Image.Resampling.LANCZOS)
    rng = random.Random(hash(scene_style + str(feat_idx)) & 0xFFFF)

    font_title = _load_font(22 if aspect_ratio == "9:16" else 24)
    font_sub = _load_font(15 if aspect_ratio == "9:16" else 16)
    font_big = _load_font(44 if aspect_ratio == "9:16" else 52, serif=True)
    font_cta = _load_font(24)

    if scene_style == "function_showcase":
        bg = _make_studio_bg(W, H, (8, 13, 24), (24, 42, 74), floor_y=0.75)
    elif scene_style == "macro_3d_sweep":
        bg = _make_studio_bg(W, H, (12, 10, 22), (38, 28, 66), floor_y=0.78)
        macro = product.copy()
        macro.thumbnail((int(W * 1.15), int(H * 0.95)), Image.Resampling.LANCZOS)
        mw, mh = macro.size
    elif scene_style == "luxury_cta_finale":
        bg = _make_studio_bg(W, H, (6, 9, 16), (26, 38, 62), floor_y=0.73)
        sparks = [(rng.uniform(0, 2 * math.pi), rng.uniform(W * 0.20, W * 0.42), rng.randint(3, 7), rng.uniform(0.8, 1.8)) for _ in range(28)]
    else:  # 3d_orbit_reveal
        bg = _make_studio_bg(W, H, (5, 8, 15), (22, 46, 82), floor_y=0.73)

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-threads", "1", str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for f in range(total_frames):
        p = f / max(1, total_frames - 1)
        ease = 1.0 - (1.0 - p) ** 3
        frame = bg.copy()

        # Volumetric diagonal studio light sweep
        sweep = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sx = int(-W * 0.4 + p * W * 1.8)
        ImageDraw.Draw(sweep).polygon(
            [(sx, 0), (sx + 110, 0), (sx - H // 2 + 110, H), (sx - H // 2, H)],
            fill=(186, 230, 253, 26),
        )
        frame = Image.alpha_composite(frame, sweep.filter(ImageFilter.GaussianBlur(18)))

        if scene_style == "macro_3d_sweep":
            ox = int((W - mw) * (0.15 + 0.7 * p))
            oy = int((H - mh) * (0.25 + 0.35 * math.sin(p * math.pi)))
            frame.alpha_composite(macro, (ox, oy))
            # Feature glass banner at bottom
            d = ImageDraw.Draw(frame)
            bx0, by0, bx1, by1 = int(W * 0.1), int(H * 0.78), int(W * 0.9), int(H * 0.90)
            d.rounded_rectangle([bx0, by0, bx1, by1], radius=16, fill=(10, 15, 28, 200), outline=(56, 189, 248, 190), width=2)
            d.text((bx0 + 22, by0 + 16), f"⚡ {feat['title'].upper()}", font=font_title, fill=(56, 189, 248, 255))
            d.text((bx0 + 22, by0 + 50), feat["detail"], font=font_sub, fill=(226, 232, 240, 240))

        elif scene_style == "function_showcase":
            tilt = math.sin(p * math.pi * 1.5 - 0.5) * 24.0
            bob = int(math.sin(p * math.pi * 2) * 10)
            tilted = _render_3d_tilted_product(hero, tilt_deg=tilt, scale=0.95 + 0.06 * ease)
            px = W // 2 - tilted.width // 2
            py = int(H * 0.44 - tilted.height // 2) + bob
            _add_studio_reflection(frame, tilted, px, py)
            frame.alpha_composite(tilted, (px, py))

            # Animated pointer line + Glassmorphic Function Callout Card
            card_alpha = int(min(1.0, p / 0.25) * 220)
            hud = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            hd = ImageDraw.Draw(hud)
            cx, cy = int(W * 0.54), int(H * 0.42)
            card_y = int(H * 0.12) if aspect_ratio == "9:16" else int(H * 0.14)
            card_x0 = int(W * 0.08)
            card_x1 = int(W * 0.92) if aspect_ratio == "9:16" else int(W * 0.52)

            # Target dot & pointer line
            hd.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(56, 189, 248, card_alpha))
            hd.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], outline=(56, 189, 248, card_alpha), width=2)
            hd.line([(cx, cy), (int(W * 0.50), card_y + 92)], fill=(56, 189, 248, card_alpha), width=2)

            hd.rounded_rectangle(
                [card_x0, card_y, card_x1, card_y + 94],
                radius=16,
                fill=(12, 19, 34, card_alpha),
                outline=(56, 189, 248, card_alpha),
                width=2,
            )
            hd.text((card_x0 + 20, card_y + 16), f"✦ {feat['title'].upper()}", font=font_title, fill=(56, 189, 248, 255))
            hd.text((card_x0 + 20, card_y + 50), feat["detail"], font=font_sub, fill=(241, 245, 249, 240))
            frame = Image.alpha_composite(frame, hud)

        elif scene_style == "luxury_cta_finale":
            tilt = (1.0 - ease) * -18.0
            tilted = _render_3d_tilted_product(hero, tilt_deg=tilt, scale=0.92 + 0.12 * ease)
            px = W // 2 - tilted.width // 2
            py = int(H * 0.43 - tilted.height // 2)
            _add_studio_reflection(frame, tilted, px, py)
            frame.alpha_composite(tilted, (px, py))

            sd = ImageDraw.Draw(frame)
            for ang, rad, sz, spd in sparks:
                a = ang + p * spd * 3.0
                sx_p = int(W // 2 + math.cos(a) * rad)
                sy_p = int(H * 0.42 + math.sin(a) * rad * 0.6)
                sd.polygon([(sx_p, sy_p - sz), (sx_p + sz, sy_p), (sx_p, sy_p + sz), (sx_p - sz, sy_p)], fill=(226, 232, 240, 210))

            # High-converting CTA pill
            btn_w, btn_h = int(W * 0.56), 58
            bx0, by0 = W // 2 - btn_w // 2, int(H * 0.80)
            sd.rounded_rectangle([bx0, by0, bx0 + btn_w, by0 + btn_h], radius=29, fill=(14, 165, 233, 240), outline=(255, 255, 255, 200), width=2)
            cta_txt = "ORDER YOURS NOW →"
            tw = sd.textlength(cta_txt, font=font_cta)
            sd.text((W // 2 - tw // 2, by0 + 15), cta_txt, font=font_cta, fill=(255, 255, 255, 255))

        else:  # 3d_orbit_reveal
            tilt = -28.0 + 42.0 * ease
            scale = 0.78 + 0.28 * ease
            tilted = _render_3d_tilted_product(hero, tilt_deg=tilt, scale=scale)
            px = W // 2 - tilted.width // 2
            py = int(H * 0.46 - tilted.height // 2)
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
    w, h = (540, 960) if aspect_ratio == "9:16" else (960, 540)
    styled_clips = []

    for idx, clip in enumerate(raw_clips):
        styled = job_dir / f"styled_{idx}.mp4"
        dur = get_duration(clip)
        out_st = max(0.5, dur - 0.25)
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=24,"
            f"eq=contrast=1.08:saturation=1.14,"
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
        f"aevalsrc='0.08*sin(2*PI*55*t)*exp(-4*mod(t,0.5))+0.03*sin(2*PI*165*t)*exp(-8*mod(t,0.25))':s=44100:d={target_dur:.2f}[bg];"
        f"[1:a]aresample=44100,volume=1.45[vo];"
        f"[vo][bg]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
    )
    video_filter = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_sec:.2f}[vfinal]"

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(merged_video), "-i", str(voice_mp3),
            "-filter_complex", f"{video_filter};{audio_filter}",
            "-map", "[vfinal]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-t", f"{target_dur:.2f}",
            "-movflags", "+faststart", "-threads", "1", str(final_mp4),
        ],
        check=True,
    )
    merged_video.unlink(missing_ok=True)
    gc.collect()
    return final_mp4


def run_pipeline(job_id: str, image_path: Path, hint: str, language: str, aspect_ratio: str):
    job_dir = image_path.parent
    try:
        JOBS[job_id]["step"] = "AI product aur uske functions ko analyze kar raha hai..."
        JOBS[job_id]["progress"] = 10
        plan = analyze_product_with_gemini(image_path, hint, language)
        JOBS[job_id]["script"] = plan["voiceover_script"]

        JOBS[job_id]["step"] = "Photo ke andar se sirf product extract ho raha hai..."
        JOBS[job_id]["progress"] = 20
        product_img = extract_product_clean(image_path.read_bytes())

        JOBS[job_id]["step"] = "Women's AI Voiceover record ho raha hai..."
        JOBS[job_id]["progress"] = 30
        voice_mp3 = job_dir / "voice.mp3"
        asyncio.run(create_female_voiceover(plan["voiceover_script"], voice_mp3, language))

        JOBS[job_id]["step"] = "3D Cinematic AI Video render check ho raha hai..."
        JOBS[job_id]["progress"] = 40
        ai_full_mp4 = job_dir / "ai_full.mp4"
        if try_generate_real_ai_video(product_img, plan, aspect_ratio, ai_full_mp4):
            raw_clips = [ai_full_mp4]
        else:
            raw_clips = []
            for idx, scene in enumerate(plan["scenes"]):
                JOBS[job_id]["step"] = f"3D Studio Scene {idx + 1}/5 ({scene['scene_style']}) render ho raha hai..."
                JOBS[job_id]["progress"] = 45 + (idx * 9)
                clip_path = job_dir / f"raw_{idx}.mp4"
                generate_3d_studio_clip(product_img, scene, plan, aspect_ratio, clip_path)
                raw_clips.append(clip_path)

        del product_img
        gc.collect()

        JOBS[job_id]["step"] = "Cinematic audio-video mix & final export ho raha hai..."
        JOBS[job_id]["progress"] = 92
        finalize_audio_video(job_dir, raw_clips, voice_mp3, aspect_ratio)

        JOBS[job_id].update({
            "status": "done",
            "progress": 100,
            "step": "3D Cinematic Product Ad ready hai!",
            "video_url": f"/download/{job_id}",
        })
    except Exception as exc:
        JOBS[job_id].update({"status": "error", "step": f"Error: {str(exc)}"})


@app.get("/healthz")
def healthz():
    return {"ok": True, "rembg": HAS_REMBG, "gemini_omni": USE_GEMINI_OMNI}


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
    return FileResponse(video_path, media_type="video/mp4", filename=f"3d_product_ad_{job_id}.mp4")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI 3D Product Ad Studio</title>
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
    <h1>✨ AI 3D Cinematic Product Ad Studio</h1>
    <p style="color:#94a3b8;font-size:0.9rem;">Koi bhi product photo upload karein — AI usme se sirf product extract karke uske functions, 3D tilt, studio reflection & high-converting Voiceover ke saath cinematic ad bana dega.</p>
    <form id="adForm">
      <label>1. Product Photo Upload Karein</label>
      <input type="file" name="image" accept="image/*" required />

      <label>2. Product Features / Ad Hint Likhein (Optional)</label>
      <textarea name="hint" rows="2" placeholder="Jaise: Waterproof smartwatch hai, heart rate & fast charging dikhao"></textarea>

      <label>3. Voiceover Language</label>
      <select name="language">
        <option value="hinglish">Hinglish (Swara Female Voice)</option>
        <option value="english">Indian English (Neerja Female Voice)</option>
      </select>

      <label>4. Video Format</label>
      <select name="aspect_ratio">
        <option value="9:16" selected>Instagram Reels / Shorts (9:16)</option>
        <option value="16:9">YouTube / Amazon / Landscape (16:9)</option>
      </select>

      <button type="submit" id="submitBtn">🎬 Generate 3D Cinematic Ad</button>
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
