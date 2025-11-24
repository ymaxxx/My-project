import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import tensorflow as tf
import os

# 尝试加载模型，如果失败则打印警告，但仍然启动服务器（方便调试页面）
model = None
try:
    model = tf.keras.models.load_model("mnist_cnn.h5")
except Exception as e:
    print("Warning: could not load model mnist_cnn.h5:", e)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    # 返回 frontend/index.html，方便直接通过浏览器访问页面
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.normpath(os.path.join(base_dir, '..', 'frontend'))
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.normpath(os.path.join(base_dir, '..', 'frontend'))
    return send_from_directory(frontend_dir, filename)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 500

    data = request.json.get("image")
    if not data:
        return jsonify({"error": "no image provided"}), 400

    img_str = data.split(",")[-1]

    img_bytes = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    # Pillow 版本差异：Image.ANTIALIAS 在新版中移到 Image.Resampling
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = getattr(Image, 'LANCZOS', Image.NEAREST)
    img = img.resize((28, 28), resample)

    arr = np.array(img) / 255.0
    # 如果训练时模型接受 (28,28,1)，则 reshape 成 (1,28,28,1)
    arr = arr.reshape(1, 28, 28, 1).astype('float32')

    # 视训练数据而定，可能需要反转颜色：arr = 1.0 - arr

    pred = model.predict(arr)
    result = int(np.argmax(pred))

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
