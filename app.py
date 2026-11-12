from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageFilter

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

def make_3d_poster(product_path, hint, out_path):
    W, H = 1080, 1920
    bg = Image.new("RGB", (W, H))
    for y in range(H):
        r = int(20 + (y/H)*30)
        g = int(10 + (y/H)*20)
        b = int(80 + (y/H)*60)
        bg.paste((r,g,b), (0,y,W,1))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([0,0,W,180], fill=(255, 195, 0))
    draw.text((60, 50), "UNIVERSAL STATIONARY", font=get_font(58), fill=(0,0,0))
    draw.text((60, 115), "MOTIPUR - BEST QUALITY - LOW PRICE", font=get_font(28), fill=(0,0,0))

    prod = Image.open(product_path).convert("RGB")
    card = Image.new("RGB", (940, 940), (255,255,255))
    prod.thumbnail((900, 900))
    card.paste(prod, ((940-prod.width)//2, (940-prod.height)//2))
    bg.paste(card, (70, 280))
    draw = ImageDraw.Draw(bg)
    draw.text((60, 1320), "NALKY RACKET SET", font=get_font(68), fill=(255,215,0))
    # Hint wrap
    words = hint.split()
    lines, cur = [], ""
    for w in words:
        if len(cur+w) < 28: cur += w + " "
        else: lines.append(cur); cur = w + " "
    lines.append(cur)
    y = 1420
    for line in lines[:2]:
        draw.text((60, y), line.upper(), font=get_font(42), fill=(255,255,255))
        y += 60
    draw.ellipse([60, 1620, 360, 1760], fill=(255,0,80))
    draw.text((90, 1665), "50% OFF", font=get_font(42), fill=(255,255,255))
    draw.text((400, 1650), "DM TO ORDER NOW", font=get_font(42), fill=(255,255,255))
    draw.text((60, 1800), "@stationersuniversal - Motipur", font=get_font(32), fill=(255,215,0))
    bg.save(out_path)
    return out_path

async def make_voice(text, out_path):
    try:
        await edge_tts.Communicate(text, VOICE).save(out_path)
        return True
    except:
        return False

@app.post("/create-ad", response_class=HTMLResponse)
async def create_ad(image: UploadFile = File(...), hint: str = Form(...)):
    uid = str(uuid.uuid4())[:8]
    img_path = f"static/uploads/{uid}.jpg"
    with open(img_path, "wb") as f: f.write(await image.read())
    script = f"Namaste dosto! {hint}. Nalky racket set ab Universal Stationary Motipur me aa gaya hai. Best quality, best price. Stock khatam hone se pehle DM karo."
    audio_path = f"static/output/{uid}.mp3"
    await make_voice(script, audio_path)
    poster = f"static/output/{uid}.jpg"
    make_3d_poster(img_path, hint, poster)

    # VIDEO - Lazy import, agar numpy nahi hai to bhi crash nahi hoga
    video_path = ""
    try:
        import numpy as np
        import imageio.v2 as imageio
        video_path = f"static/output/{uid}.mp4"
        base = imageio.imread(poster)
        kwargs = {"fps": 24, "codec": "libx264"}
        if os.path.exists(audio_path):
            kwargs["audio_path"] = audio_path
            kwargs["audio_codec"] = "aac"
        writer = imageio.get_writer(video_path, **kwargs)
        for i in range(96):
            zoom = 1.0 + (i/96)*0.12
            h, w = base.shape[:2]
            nh, nw = int(h/zoom), int(w/zoom)
            y0, x0 = (h-nh)//2, (w-nw)//2
            crop = base[y0:y0+nh, x0:x0+nw]
            frame = np.array(Image.fromarray(crop).resize((1080,1920)))
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        print("Video skip:", e)
        video_path = ""

    if video_path and os.path.exists(video_path):
        media = f'<video controls autoplay loop style="width:90%;max-width:400px;border-radius:15px" src="/static/output/{uid}.mp4"></video><br><br><a href="/static/output/{uid}.mp4" download style="background:#ff0050;padding:15px 30px;color:white;text-decoration:none;font-weight:bold;border-radius:10px">VIDEO DOWNLOAD KARO</a>'
    else:
        media = f'<img src="/static/output/{uid}.jpg" style="width:90%;max-width:400px;border-radius:15px"><br><br><a href="/static/output/{uid}.jpg" download style="background:gold;padding:15px 30px;color:black;text-decoration:none;font-weight:bold;border-radius:10px">IMAGE DOWNLOAD KARO</a>'

    return f"""<html><body style="background:#0a0f2d;color:white;text-align:center;font-family:sans-serif;padding:20px">
    <h1 style="color:gold">3D AD BAN GAYA!</h1>{media}<br><br>
    <audio controls src="/static/output/{uid}.mp3"></audio><br><br><p>{script}</p><br><a href="/" style="color:white">Naya Ad Banao</a></body></html>"""

@app.get("/")
async def home():
    return FileResponse("templates/index.html")
