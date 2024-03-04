from flask import Flask, render_template, request
from main_model import summarizer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        s = summarizer(rawtext)
        # summary, original_txt, len_orig_txt, len_summary = summarizer(rawtext)
          
    return render_template('summary.html', s=s)
    # return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_txt=len_orig_txt, len_summary=len_summary)

if __name__ == "__main__":
    app.run(debug=True)
