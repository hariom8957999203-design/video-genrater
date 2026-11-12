from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, asyncio, edge_tts
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/output", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

VOICE = "hi-IN-SwaraNeural"

@app.get("/health")
def health():
    return {"status": "ok"}

async def make_voice(text, out_path):
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(out_path)
        return True
    except Exception as e:
        print("Voice Error:", e)
        return False

def make_ad_image(product_path, hint, out_path):
    img = Image.open(product_path).convert("RGB")
    bg = Image.new("RGB", (1080, 1920), (10, 15, 45))
    # Product ko center me fit karo
    img.thumbnail((900, 900))
    bg.paste(img, (90, 450))
    draw = ImageDraw.Draw(bg)
    draw.text((90, 100), "UNIVERSAL STATIONARY", fill=(255, 215, 0))
    # Hint ko 2 line me todo
    draw.text((90, 1450), hint[:40], fill=(255, 255, 255))
    draw.text((90, 1550), "stationersuniversal | DM to Order", fill=(255, 215, 0))
    bg.save(out_path)
    return out_path

@app.post("/create-ad", response_class=HTMLResponse)
async def create_ad(image: UploadFile = File(...), hint: str = Form(...)):
    try:
        uid = str(uuid.uuid4())[:8]
        img_path = f"static/uploads/{uid}.jpg"
        with open(img_path, "wb") as f:
            f.write(await image.read())

        script = f"Namaste dosto! {hint}. Ye product ab Universal Stationary Store me available hai. Quality best hai, price bilkul kam hai. Order ke liye DM karo."
        
        audio_path = f"static/output/{uid}.mp3"
        await make_voice(script, audio_path)

        ad_img = f"static/output/{uid}.jpg"
        make_ad_image(img_path, hint, ad_img)

        return f"""
        <html><body style="background:#0a0f2d;color:white;text-align:center;font-family:sans-serif;padding:20px">
        <h1 style="color:gold">AD BAN GAYA!</h1>
        <img src="/static/output/{uid}.jpg" style="width:90%;max-width:400px;border-radius:15px"><br><br>
        <audio controls src="/static/output/{uid}.mp3"></audio><br><br>
        <p>{script}</p>
        <a href="/static/output/{uid}.jpg" download style="background:gold;padding:15px 30px;color:black;text-decoration:none;font-weight:bold;border-radius:10px">AD IMAGE DOWNLOAD KARO</a>
        <br><br><a href="/" style="color:white">Naya Ad Banao</a>
        </body></html>
        """
    except Exception as e:
        return f"<h1>Error: {e}</h1><a href='/'>Wapas Jao</a>"

@app.get("/")
async def home():
    return FileResponse("templates/index.html")
async def home():
    return FileResponse("templates/index.html")
