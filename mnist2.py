import os
import logging
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ログの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'  # flashメッセージのために必要

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

try:
    model = load_model('./model.keras')  # 学習済みモデルをロード
    logger.info("モデルが正常にロードされました。")
except Exception as e:
    logger.error(f"モデルのロード中にエラーが発生しました: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルが選択されていません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"ファイルが保存されました: {filepath}")

                # 受け取った画像を読み込み、np形式に変換
                img = image.load_img(filepath, color_mode="grayscale", target_size=(image_size,image_size))
                img = image.img_to_array(img)
                data = np.array([img])
                
                # 変換したデータをモデルに渡して予測する
                result = model.predict(data)[0]
                predicted = result.argmax()
                pred_answer = f"これは {classes[predicted]} です"

                logger.info(f"予測結果: {pred_answer}")
                return render_template("index.html", answer=pred_answer)
        except Exception as e:
            logger.error(f"エラーが発生しました: {str(e)}")
            return render_template("index.html", answer=f"エラーが発生しました: {str(e)}")

    return render_template("index.html", answer="")

if __name__ == "__main__":
    # UPLOADフォルダが存在しない場合は作成
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        logger.info(f"アップロードフォルダを作成しました: {UPLOAD_FOLDER}")
    
    app.run(debug=True)