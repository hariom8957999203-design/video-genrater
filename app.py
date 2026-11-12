from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

app = FastAPI()
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/output", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
VOICE = "hi-IN-SwaraNeural"

def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

def make_cinematic_poster(product_path, hint, out_path):
    W, H = 1080, 1920
    # 1. CINEMATIC DARK BACKGROUND - Example jaisa
    bg = Image.new("RGB", (W, H), (12, 12, 18))
    draw = ImageDraw.Draw(bg)
    # Gold side light lines
    draw.rectangle([30, 0, 38, H], fill=(255, 195, 0))
    draw.rectangle([W-38, 0, W-30, H], fill=(255, 195, 0))
    # Bottom glow
    for y in range(1500, H):
        bg.paste((30, 20, 10), (0, y, W, y+1))

    # 2. TERA PRODUCT - Bada, Floating, Glow ke saath
    prod = Image.open(product_path).convert("RGB")
    # Product ko thoda bright + contrast badhao taaki pro lage
    prod = ImageEnhance.Contrast(prod).enhance(1.15)
    prod = ImageEnhance.Color(prod).enhance(1.15)
    # Thoda rotate karo - 8 degree - Floating feel ke liye
    prod = prod.rotate(-8, expand=True, fillcolor=(12,12,18))
    prod.thumbnail((950, 1050))

    # Shadow + Gold Glow
    shadow = Image.new("RGBA", (W, H), (0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse([120, 1450, 960, 1580], fill=(0,0,0,180))
    shadow = shadow.filter(ImageFilter.GaussianBlur(30))
    bg = Image.alpha_composite(bg.convert("RGBA"), shadow).convert("RGB")
    bg.paste(prod, ((W-prod.width)//2, 380))

    draw = ImageDraw.Draw(bg)
    # 3. TOP TEXT - Example jaisa
    draw.text((90, 70), "MAKING PRODUCT ADS UNTIL", font=get_font(30), fill=(255,215,150))
    draw.text((90, 115), "THE COMPANY NOTICES ME", font=get_font(30), fill=(255,215,150))

    # 4. BOTTOM - DAY 8 - Bada Gold
    draw.text((120, 1520), "DAY 8", font=get_font(180), fill=(255,195,0))
    draw.text((60, 1720), "NALKY RACKET SET", font=get_font(52), fill=(255,255,255))
    # Hint - 2 line
    words = hint.split()
    lines, cur = [], ""
    for w in words:
        if len(cur+w) < 30: cur += w + " "
        else: lines.append(cur); cur = w + " "
    lines.append(cur)
    y = 1790
    for line in lines[:1]:
        draw.text((60, y), line.upper(), font=get_font(32), fill=(200,200,200))
    draw.text((60, 1860), "UNIVERSAL STATIONARY - MOTIPUR", font=get_font(28), fill=(255,215,0))
    bg.save(out_path)
    return out_path

@app.post("/create-ad", response_class=HTMLResponse)
async def create_ad(image: UploadFile = File(...), hint: str = Form(...)):
    uid = str(uuid.uuid4())[:8]
    img_path = f"static/uploads/{uid}.jpg"
    with open(img_path, "wb") as f: f.write(await image.read())
    script = f"Day 8. Making product ads until the company notices me. Aaj hai Nalky racket set. {hint}. Lightweight, strong grip, best for beginners. Sirf Universal Stationary Motipur me."
    audio_path = f"static/output/{uid}.mp3"
    try: await edge_tts.Communicate(script, VOICE).save(audio_path)
    except: pass
    poster = f"static/output/{uid}.jpg"
    make_cinematic_poster(img_path, hint, poster)

    # VIDEO - Slow Zoom + Floating feel
    video_html = ""
    try:
        import numpy as np
        import imageio.v2 as imageio
        video_path = f"static/output/{uid}.mp4"
        base = imageio.imread(poster)
        base_small = np.array(Image.fromarray(base).resize((720,1280)))
        writer = imageio.get_writer(video_path, fps=24, codec='libx264')
        for i in range(96): # 4 sec
            zoom = 1.0 + (i/96)*0.12
            h, w = base_small.shape[:2]
            nh, nw = int(h/zoom), int(w/zoom)
            y0, x0 = (h-nh)//2, (w-nw)//2
            crop = base_small[y0:y0+nh, x0:x0+nw]
            frame = np.array(Image.fromarray(crop).resize((720,1280)))
            writer.append_data(frame)
        writer.close()
        video_html = f'<video controls autoplay loop style="width:90%;max-width:400px;border-radius:15px" src="/static/output/{uid}.mp4"></video><br><br><a href="/static/output/{uid}.mp4" download style="background:#ff0050;padding:15px 30px;color:white;text-decoration:none;font-weight:bold;border-radius:10px">VIDEO DOWNLOAD KARO</a>'
    except Exception as e:
        print(e)
        video_html = f'<img src="/static/output/{uid}.jpg" style="width:90%;max-width:400px"><br><br><a href="/static/output/{uid}.jpg" download style="background:gold;padding:15px 30px;color:black;text-decoration:none;font-weight:bold;border-radius:10px">IMAGE DOWNLOAD KARO</a>'

    return f"""<html><body style="background:#0a0f2d;color:white;text-align:center;font-family:sans-serif;padding:20px">
    <h1 style="color:gold">CINEMATIC AD BAN GAYA!</h1>{video_html}<br><br>
    <audio controls src="/static/output/{uid}.mp3"></audio><br><br><p>{script}</p><br><a href="/" style="color:white">Naya Ad Banao</a></body></html>"""

@app.get("/")
async def home():
    return FileResponse("templates/index.html")
