from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, asyncio, edge_tts
from PIL import Image, ImageDraw

app = FastAPI()
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/output", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

VOICE = "hi-IN-SwaraNeural"

async def make_voice(text, out_path):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(out_path)

def make_ad_image(product_path, hint, out_path):
    img = Image.open(product_path).convert("RGB")
    bg = Image.new("RGB", (1080, 1920), (10, 15, 45))
    img = img.resize((900, 900))
    bg.paste(img, (90, 350))
    draw = ImageDraw.Draw(bg)
    draw.text((90, 100), "UNIVERSAL STATIONARY", fill=(255, 215, 0))
    draw.text((90, 1400), hint, fill=(255, 255, 255))
    draw.text((90, 1550), "stationersuniversal | DM to Order", fill=(255, 215, 0))
    bg.save(out_path)
    return out_path

@app.post("/create-ad")
async def create_ad(image: UploadFile = File(...), hint: str = Form(...)):
    uid = str(uuid.uuid4())[:8]
    img_path = f"static/uploads/{uid}.jpg"
    with open(img_path, "wb") as f:
        f.write(await image.read())

    script = f"Namaste dosto! {hint}. Ye product ab Universal Stationary and Journal Store me available hai. Quality best hai, price bilkul kam hai. Order karne ke liye abhi DM karo."
    
    audio_path = f"static/output/{uid}.mp3"
    await make_voice(script, audio_path)

    ad_img = f"static/output/{uid}.jpg"
    make_ad_image(img_path, hint, ad_img)

    video_path = f"static/output/{uid}.mp4"
    cmd = f"ffmpeg -y -loop 1 -i {ad_img} -i {audio_path} -vf \"scale=1080:1920,zoompan=z='min(zoom+0.0015,1.3)':d=250:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920\" -shortest -c:v libx264 -pix_fmt yuv420p {video_path}"
    os.system(cmd)
    
    return {"video_url": f"/static/output/{uid}.mp4", "script": script}

@app.get("/")
async def home():
    return FileResponse("templates/index.html")
