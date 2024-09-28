from tensorflow.keras.models import load_model

# .h5 ファイルをロード
model = load_model('model.h5')

# .keras 形式で保存
model.save('model2.keras')