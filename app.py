from flask import Flask, render_template, request, url_for, session
import os, json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from openai import OpenAI
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret-key"   # مهم للـ session

# ===== OpenAI Client =====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== PATHS =====
MODEL_PATH = "models/effnetB0_final.h5"
CLASSES_PATH = "models/classes.json"
UPLOAD_FOLDER = os.path.join("static", "uploads")
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== LOAD MODEL =====
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ===== LOAD CLASSES =====
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes_obj = json.load(f)

if isinstance(classes_obj, dict):
    idx2label = {int(k): v for k, v in classes_obj.items()}
    CLASS_NAMES = [idx2label[i] for i in range(len(idx2label))]
elif isinstance(classes_obj, list):
    CLASS_NAMES = classes_obj
else:
    raise ValueError("classes.json لازم يكون dict أو list")

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ===== IMAGE PREPROCESS =====
def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def safe_openai(system_msg, user_msg):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "حدث خطأ أثناء توليد الإجابة."

# ===== MAIN PAGE =====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename.strip() == "":
            return render_template("index.html", error="لم يتم اختيار صورة")

        if not allowed_file(file.filename):
            return render_template("index.html", error="صيغة الصورة غير مدعومة")

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        img = preprocess_image(save_path)
        preds = model.predict(img, verbose=0)[0]

        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        if top_idx >= len(CLASS_NAMES):
            return render_template("index.html", error="خطأ في عدد الكلاسات")

        if confidence < 0.5:
            return render_template("index.html", error="الصورة غير واضحة أو ليست مرضًا جلديًا")

        label = CLASS_NAMES[top_idx]

        # تخزين المرض للـ chat
        session["last_label"] = label
        session["last_confidence"] = round(confidence * 100, 2)

        result = {
            "label": label,
            "confidence": round(confidence * 100, 2)
        }

        # شرح أطول شوي
        prompt = f"""
اشرح مرض {label} بشرح تثقيفي واضح.

اذكر:
- ما هو المرض
- الأسباب الشائعة
- الأعراض
- طرق العناية العامة
- متى يجب زيارة الطبيب

بدون تشخيص طبي.
"""
        ai_text = safe_openai("أنت مساعد طبي تثقيفي.", prompt)

        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "index.html",
            image_url=image_url,
            result=result,
            ai_text=ai_text
        )

    return render_template("index.html")

# ===== CHAT ROUTE (مربوط بالمرض تلقائيًا) =====
@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return {"reply": "اكتبي سؤالك."}

    label = session.get("last_label")
    confidence = session.get("last_confidence")

    if not label:
        return {"reply": "ارفعي صورة أولاً حتى أحدد المرض ثم اسألي."}

    prompt = f"""
أنت مساعد طبي تثقيفي.
اعتبر أن كل سؤال متعلق بمرض: {label}
نسبة الثقة: {confidence}%

سؤال المستخدم: {user_msg}

قواعد:
- أجب عن مرض {label} فقط
- اذكر علاجات أو أدوية عامة بدون جرعات
- لا تشخص
- أجب بالعربية وبوضوح
"""

    reply = safe_openai("أنت مساعد طبي تثقيفي.", prompt)
    return {"reply": reply}

if __name__ == "__main__":
    app.run(debug=True)