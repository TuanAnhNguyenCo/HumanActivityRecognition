from flask import Flask, request, jsonify
from util.img2bone import HandDetector
import numpy as np


hand = HandDetector()
app = Flask("HAR")
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Lấy dữ liệu gửi từ client
    return jsonify({"prediction": "OK"})
    img = data['img']
    print(np.array(img).shape)
    bones = []
    for i in range(5):
        bone = hand.findHands(np.array(img[i],dtype=np.uint8))
        if bone is not None:
            bones.append(bone)
    bones = np.array(bones)
    if (bones.shape[0] == 5 and bones.shape[1] == 21):
        return jsonify({"prediction": "OK"})
    else:
        return jsonify({"prediction": "Fail"})
    
        
    # Thực hiện xử lý dữ liệu và dự đoán bằng mô hình AI của bạn
    

if __name__ == '__main__':
    app.run(debug=True)


