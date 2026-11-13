import io
import math
import os
import random
import modal

app = modal.App("ai-product-ad-video")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "diffusers==0.32.2",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "sentencepiece==0.2.0",
        "pillow==11.1.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "rembg[cpu]==2.0.62",
        "fastapi[standard]==0.115.8",
    )
)

volume = modal.Volume.from_name("ltx-model-cache", create_if_missing=True)


def extract_product_cutout(image_bytes: bytes, session):
    from PIL import Image
    from rembg import remove

    raw = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    cutout = remove(raw, session=session)
    bbox = cutout.getbbox()
    if bbox:
        cutout = cutout.crop(bbox)
    return cutout


def paste_with_shadow(canvas, item, cx: int, cy: int, angle: float = 0.0, add_shadow: bool = True):
    from PIL import Image, ImageFilter

    rotated = item.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x = cx - rotated.width // 2
    y = cy - rotated.height // 2

    if add_shadow:
        alpha = rotated.getchannel("A")
        shadow = Image.new("RGBA", rotated.size, (10, 15, 25, 0))
        shadow.putalpha(alpha.point(lambda p: int(p * 0.42)))
        shadow = shadow.filter(ImageFilter.GaussianBlur(14))
        canvas.alpha_composite(shadow, (x + 12, y + 22))

    canvas.alpha_composite(rotated, (x, y))


def scale_to_height(img, target_h: int):
    from PIL import Image

    ratio = target_h / max(1, img.height)
    new_w = max(1, int(img.width * ratio))
    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


def build_luxury_keyframe(cutout, scene_style: str, w: int, h: int):
    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(42)

    if scene_style == "shatter_reveal":
        # Cool slate-blue misty studio + floating tilted product + shattering shards
        bg = Image.new("RGBA", (w, h), (155, 175, 195, 255))
        draw = ImageDraw.Draw(bg)
        for y in range(h):
            shade = int(185 - (y / h) * 65)
            draw.line([(0, y), (w, y)], fill=(shade - 25, shade - 10, shade + 10, 255))
        # Diagonal light shaft
        beam = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(beam)
        bdraw.polygon([(0, 0), (w // 2, 0), (w, h), (w // 3, h)], fill=(230, 240, 255, 55))
        bg.alpha_composite(beam.filter(ImageFilter.GaussianBlur(30)))

        prod = scale_to_height(cutout, int(h * 0.56))
        paste_with_shadow(bg, prod, w // 2 + 10, h // 2, angle=-10)

        # Shattering dark-blue & silver fragments on left
        shards = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shards)
        for _ in range(42):
            sx = rng.randint(int(w * 0.08), int(w * 0.52))
            sy = rng.randint(int(h * 0.18), int(h * 0.82))
            sz = rng.randint(8, 28)
            pts = [(sx, sy), (sx + sz, sy + rng.randint(-10, 12)), (sx + rng.randint(-8, 12), sy + sz)]
            col = (20, 55, 105, 220) if rng.random() > 0.3 else (210, 225, 240, 200)
            sdraw.polygon(pts, fill=col)
        bg.alpha_composite(shards.filter(ImageFilter.GaussianBlur(0.8)))
        return bg.convert("RGB")

    elif scene_style == "pedestal_blinds":
        # Architectural white stepped pedestals + Venetian blind gobo shadows + foreground leaves
        bg = Image.new("RGBA", (w, h), (235, 238, 242, 255))
        draw = ImageDraw.Draw(bg)

        # Venetian blind diagonal shadow stripes on backdrop
        blinds = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(blinds)
        for i in range(-h, w + h, 58):
            bdraw.polygon([(i, 0), (i + 24, 0), (i - 140, h), (i - 164, h)], fill=(120, 130, 145, 70))
        bg.alpha_composite(blinds.filter(ImageFilter.GaussianBlur(8)))

        # Stepped white pedestals
        draw.rectangle([0, int(h * 0.62), w, h], fill=(215, 220, 226, 255))
        draw.rectangle([0, int(h * 0.72), w, h], fill=(238, 240, 244, 255))
        draw.rectangle([int(w * 0.28), int(h * 0.57), int(w * 0.72), int(h * 0.72)], fill=(246, 248, 250, 255))

        # 3 Products on stepped pedestals
        side_prod = scale_to_height(cutout, int(h * 0.34))
        center_prod = scale_to_height(cutout, int(h * 0.42))
        paste_with_shadow(bg, side_prod, int(w * 0.22), int(h * 0.46), angle=0)
        paste_with_shadow(bg, side_prod, int(w * 0.78), int(h * 0.46), angle=0)
        paste_with_shadow(bg, center_prod, int(w * 0.50), int(h * 0.40), angle=0)

        # Out-of-focus foreground tropical leaves
        leaves = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(leaves)
        ldraw.polygon([(0, int(h * 0.78)), (int(w * 0.26), int(h * 0.68)), (int(w * 0.14), h), (0, h)], fill=(25, 55, 38, 220))
        ldraw.polygon([(w, int(h * 0.20)), (int(w * 0.76), int(h * 0.32)), (w, int(h * 0.45))], fill=(25, 55, 38, 210))
        bg.alpha_composite(leaves.filter(ImageFilter.GaussianBlur(10)))
        return bg.convert("RGB")

    elif scene_style == "macro_detail":
        # Tight luxury macro angle with dramatic studio rim lighting
        bg = Image.new("RGBA", (w, h), (16, 22, 34, 255))
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse([int(w * 0.1), int(h * 0.15), int(w * 0.9), int(h * 0.85)], fill=(70, 125, 185, 120))
        bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(55)))

        prod = scale_to_height(cutout, int(h * 0.92))
        paste_with_shadow(bg, prod, int(w * 0.55), int(h * 0.56), angle=-16, add_shadow=False)
        return bg.convert("RGB")

    elif scene_style == "twin_float":
        # Clean bright minimalist studio + 2 angled floating products
        bg = Image.new("RGBA", (w, h), (236, 240, 245, 255))
        draw = ImageDraw.Draw(bg)
        for y in range(h):
            v = int(242 - (y / h) * 20)
            draw.line([(0, y), (w, y)], fill=(v, v + 2, v + 5, 255))

        prod_a = scale_to_height(cutout, int(h * 0.44))
        prod_b = scale_to_height(cutout, int(h * 0.44))
        paste_with_shadow(bg, prod_a, int(w * 0.30), int(h * 0.58), angle=18)
        paste_with_shadow(bg, prod_b, int(w * 0.70), int(h * 0.42), angle=-24)
        return bg.convert("RGB")

    else:  # chrome_finale
        # Hero centered spotlight + metallic silver shatter energy
        bg = Image.new("RGBA", (w, h), (185, 202, 218, 255))
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse([int(w * 0.15), int(h * 0.2), int(w * 0.85), int(h * 0.8)], fill=(255, 255, 255, 170))
        bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(45)))

        prod = scale_to_height(cutout, int(h * 0.58))
        paste_with_shadow(bg, prod, w // 2, h // 2, angle=0)

        shards = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shards)
        for _ in range(36):
            sx = rng.randint(int(w * 0.08), int(w * 0.45))
            sy = rng.randint(int(h * 0.20), int(h * 0.80))
            sz = rng.randint(10, 26)
            pts = [(sx, sy), (sx + sz, sy + 6), (sx + 4, sy + sz)]
            sdraw.polygon(pts, fill=(235, 240, 248, 230))
        bg.alpha_composite(shards.filter(ImageFilter.GaussianBlur(0.6)))
        return bg.convert("RGB")


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/root/.cache/huggingface": volume},
    scaledown_window=120,
    timeout=600,
)
class VideoGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import LTXImageToVideoPipeline
        from rembg import new_session

        os.environ["U2NET_HOME"] = "/root/.cache/huggingface/u2net"
        self.rembg_session = new_session("u2net")

        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")

    @modal.method()
    def generate_clip(
        self,
        image_bytes: bytes,
        prompt: str,
        scene_style: str = "shatter_reveal",
        aspect_ratio: str = "9:16",
    ) -> bytes:
        import tempfile
        from diffusers.utils import export_to_video

        width, height = (480, 704) if aspect_ratio == "9:16" else (704, 480)
        cutout = extract_product_cutout(image_bytes, self.rembg_session)
        keyframe = build_luxury_keyframe(cutout, scene_style, width, height)

        frames = self.pipe(
            image=keyframe,
            prompt=prompt,
            negative_prompt="worst quality, blurry, jittery, distorted text, watermark, cheap look, deformed object",
            width=width,
            height=height,
            num_frames=65,  # ~2.7s snappy luxury shot at 24fps (must be 8k+1)
            num_inference_steps=28,
            guidance_scale=3.5,
        ).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            export_to_video(frames, tmp.name, fps=24)
            tmp.seek(0)
            return tmp.read()


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
async def generate_video_api(request: dict):
    import base64
    from fastapi import Response

    img_bytes = base64.b64decode(request["image_base64"])
    prompt = request["prompt"]
    scene_style = request.get("scene_style", "shatter_reveal")
    aspect_ratio = request.get("aspect_ratio", "9:16")

    mp4_bytes = VideoGenerator().generate_clip.remote(img_bytes, prompt, scene_style, aspect_ratio)
    return Response(content=mp4_bytes, media_type="video/mp4")
