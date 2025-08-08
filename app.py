from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PyPDF2 import PdfReader
import os
from utils import generatetextquestion,generatemcqquestion, generatetruefalsequestion, generatebloomscore


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify(message="Hello, Flask!")

@app.route('/generatequestion', methods=['POST'])
def generatequestion():
    data = request.get_json()
    print(data['config'])
    file_data = data.get('file', {})
    filename = file_data.get('filename')
    base64_content = file_data.get('content')
    pdf_bytes = base64.b64decode(base64_content)
    pdf_stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)
    all_text = ''
    for page in reader.pages:
        all_text += page.extract_text() or ''
    testquestion = []
    truefalsequestion = []
    mcqquestion = []
    if data['config']['numQuestions']['text'] > 0:
        testquestion = generatetextquestion(context=all_text,length=data['config']['questionLength'],instruction=data['config']['userInput'],number=data['config']['numQuestions']['text'],info=data['config']['bloomWeights'])
    if data['config']['numQuestions']['mcq'] > 0:
        mcqquestion = generatemcqquestion(context=all_text,length=data['config']['questionLength'],instruction=data['config']['userInput'],number=data['config']['numQuestions']['mcq'],info=data['config']['bloomWeights'])
    if data['config']['numQuestions']['trueFalse'] > 0:
        truefalsequestion = generatetruefalsequestion(context=all_text,length=data['config']['questionLength'],instruction=data['config']['userInput'],number=data['config']['numQuestions']['trueFalse'],info=data['config']['bloomWeights'])
    print(testquestion)
    print(truefalsequestion)
    print(mcqquestion)
    response = []
    for i in testquestion:
        response.append({"question_type":"text","questioninfo":i})
    for i in truefalsequestion:
        response.append({"question_type":"TrueFalse","questioninfo":i})
    for i in mcqquestion:
        response.append({"question_type":"MCQ","questioninfo":i})
    print(response)
    return jsonify(response)

@app.route('/analysequestion', methods=['POST'])
def analysequestion():
    data = request.get_json()
    print(data)
    json = generatebloomscore(data['file']['mimeType'],data['file']['content'])
    response = json
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
