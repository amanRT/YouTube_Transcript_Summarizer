import json
from flask import Flask
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)


model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
article = """
Justin Timberlake and Jessica Biel, welcome to parenthood. 
The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People. 
"Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports. 
The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both.
"""
def YtTrans(video_id):
    output=YouTubeTranscriptApi.get_transcript(video_id,languages=['de', 'en'])
    print(len(output))
    for i in range(0,len(output)):
        test=output[i]['text']
        print(test)
    str=output[4]['text']
    return str
def YtTranscript(transcript):
    inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)
    print(outputs)


@app.route('/')
def home():
    YtTranscript(article)
    return YtTrans("ppJy5uGZLi4")

if __name__ == '__main__':
    app.run(debug=True)